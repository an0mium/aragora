"""
SQLite integration store backend.

Persisted to disk, survives restarts. Suitable for single-instance
production deployments.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

from aragora.config import resolve_db_path
from aragora.storage.integration_backends import IntegrationStoreBackend
from aragora.storage.integration_models import (
    IntegrationConfig,
    UserIdMapping,
    _decrypt_settings,
    _encrypt_settings,
    _record_user_mapping_operation,
)

logger = logging.getLogger(__name__)


class SQLiteIntegrationStore(IntegrationStoreBackend):
    """
    SQLite-backed integration store.

    Persisted to disk, survives restarts. Suitable for single-instance
    production deployments.

    Raises:
        DistributedStateError: In production if PostgreSQL is not available
    """

    def __init__(self, db_path: Path | str):
        # SECURITY: Check production guards for SQLite usage
        try:
            from aragora.storage.production_guards import (
                require_distributed_store,
                StorageMode,
            )

            require_distributed_store(
                "integration_store",
                StorageMode.SQLITE,
                "Integration store using SQLite - use PostgreSQL for multi-instance deployments",
            )
        except ImportError:
            pass  # Guards not available, allow SQLite

        self.db_path = Path(resolve_db_path(db_path))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # ContextVar for per-async-context connection (async-safe replacement for threading.local)
        self._conn_var: contextvars.ContextVar[sqlite3.Connection | None] = contextvars.ContextVar(
            f"integration_conn_{id(self)}", default=None
        )
        # Track all connections for proper cleanup
        self._connections: set[sqlite3.Connection] = set()
        self._connections_lock = threading.Lock()
        self._init_schema()
        logger.info("SQLiteIntegrationStore initialized: %s", self.db_path)

    def _get_conn(self) -> sqlite3.Connection:
        """Get per-context database connection."""
        conn = self._conn_var.get()
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._conn_var.set(conn)
            with self._connections_lock:
                self._connections.add(conn)
        return conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS integrations (
                integration_type TEXT NOT NULL,
                user_id TEXT NOT NULL DEFAULT 'default',
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                notify_on_consensus INTEGER DEFAULT 1,
                notify_on_debate_end INTEGER DEFAULT 1,
                notify_on_error INTEGER DEFAULT 0,
                notify_on_leaderboard INTEGER DEFAULT 0,
                settings_json TEXT,
                messages_sent INTEGER DEFAULT 0,
                errors_24h INTEGER DEFAULT 0,
                last_activity REAL,
                last_error TEXT,
                workspace_id TEXT,
                PRIMARY KEY (user_id, integration_type)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_integrations_user ON integrations(user_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_integrations_type ON integrations(integration_type)"
        )

        # User ID mappings table (cross-platform identity resolution)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_id_mappings (
                email TEXT NOT NULL,
                platform TEXT NOT NULL,
                platform_user_id TEXT NOT NULL,
                display_name TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                user_id TEXT NOT NULL DEFAULT 'default',
                PRIMARY KEY (user_id, platform, email)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mappings_email ON user_id_mappings(email)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mappings_platform ON user_id_mappings(platform)"
        )
        conn.commit()
        conn.close()

    async def get(
        self, integration_type: str, user_id: str = "default"
    ) -> IntegrationConfig | None:
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT integration_type, enabled, created_at, updated_at,
                      notify_on_consensus, notify_on_debate_end, notify_on_error,
                      notify_on_leaderboard, settings_json, messages_sent,
                      errors_24h, last_activity, last_error, user_id, workspace_id
               FROM integrations WHERE user_id = ? AND integration_type = ?""",
            (user_id, integration_type),
        )
        row = cursor.fetchone()
        if row:
            config = IntegrationConfig.from_row(row)
            # Decrypt settings with AAD for integrity verification
            config.settings = _decrypt_settings(config.settings, user_id, integration_type)
            return config
        return None

    async def save(self, config: IntegrationConfig) -> None:
        conn = self._get_conn()
        config.updated_at = time.time()
        user_id = config.user_id or "default"
        # Encrypt settings with AAD binding to user + integration type
        encrypted_settings = _encrypt_settings(config.settings, user_id, config.type)
        conn.execute(
            """INSERT OR REPLACE INTO integrations
               (integration_type, user_id, enabled, created_at, updated_at,
                notify_on_consensus, notify_on_debate_end, notify_on_error,
                notify_on_leaderboard, settings_json, messages_sent, errors_24h,
                last_activity, last_error, workspace_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                config.type,
                user_id,
                int(config.enabled),
                config.created_at,
                config.updated_at,
                int(config.notify_on_consensus),
                int(config.notify_on_debate_end),
                int(config.notify_on_error),
                int(config.notify_on_leaderboard),
                json.dumps(encrypted_settings),
                config.messages_sent,
                config.errors_24h,
                config.last_activity,
                config.last_error,
                config.workspace_id,
            ),
        )
        conn.commit()
        logger.debug("Saved integration: %s for user %s", config.type, user_id)

    async def delete(self, integration_type: str, user_id: str = "default") -> bool:
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM integrations WHERE user_id = ? AND integration_type = ?",
            (user_id, integration_type),
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.debug("Deleted integration: %s for user %s", integration_type, user_id)
        return deleted

    async def list_for_user(self, user_id: str = "default") -> list[IntegrationConfig]:
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT integration_type, enabled, created_at, updated_at,
                      notify_on_consensus, notify_on_debate_end, notify_on_error,
                      notify_on_leaderboard, settings_json, messages_sent,
                      errors_24h, last_activity, last_error, user_id, workspace_id
               FROM integrations WHERE user_id = ?""",
            (user_id,),
        )
        configs = []
        for row in cursor.fetchall():
            config = IntegrationConfig.from_row(row)
            config.settings = _decrypt_settings(config.settings, user_id, config.type)
            configs.append(config)
        return configs

    async def list_all(self, limit: int = 1000) -> list[IntegrationConfig]:
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT integration_type, enabled, created_at, updated_at,
                      notify_on_consensus, notify_on_debate_end, notify_on_error,
                      notify_on_leaderboard, settings_json, messages_sent,
                      errors_24h, last_activity, last_error, user_id, workspace_id
               FROM integrations
               LIMIT ?""",
            (limit,),
        )
        configs = []
        for row in cursor.fetchall():
            config = IntegrationConfig.from_row(row)
            config.settings = _decrypt_settings(
                config.settings, config.user_id or "default", config.type
            )
            configs.append(config)
        return configs

    async def get_user_mapping(
        self, email: str, platform: str, user_id: str = "default"
    ) -> UserIdMapping | None:
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT email, platform, platform_user_id, display_name,
                      created_at, updated_at, user_id
               FROM user_id_mappings
               WHERE user_id = ? AND platform = ? AND email = ?""",
            (user_id, platform, email),
        )
        row = cursor.fetchone()
        if row:
            return UserIdMapping.from_row(row)
        return None

    async def save_user_mapping(self, mapping: UserIdMapping) -> None:
        conn = self._get_conn()
        mapping.updated_at = time.time()
        conn.execute(
            """INSERT OR REPLACE INTO user_id_mappings
               (email, platform, platform_user_id, display_name,
                created_at, updated_at, user_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                mapping.email,
                mapping.platform,
                mapping.platform_user_id,
                mapping.display_name,
                mapping.created_at,
                mapping.updated_at,
                mapping.user_id,
            ),
        )
        conn.commit()
        _record_user_mapping_operation("save", mapping.platform, True)
        logger.debug("Saved user mapping: %s -> %s", mapping.email, mapping.platform)

    async def delete_user_mapping(
        self, email: str, platform: str, user_id: str = "default"
    ) -> bool:
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM user_id_mappings WHERE user_id = ? AND platform = ? AND email = ?",
            (user_id, platform, email),
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        _record_user_mapping_operation("delete", platform, deleted)
        return deleted

    async def list_user_mappings(
        self, platform: str | None = None, user_id: str = "default"
    ) -> list[UserIdMapping]:
        conn = self._get_conn()
        if platform:
            cursor = conn.execute(
                """SELECT email, platform, platform_user_id, display_name,
                          created_at, updated_at, user_id
                   FROM user_id_mappings
                   WHERE user_id = ? AND platform = ?""",
                (user_id, platform),
            )
        else:
            cursor = conn.execute(
                """SELECT email, platform, platform_user_id, display_name,
                          created_at, updated_at, user_id
                   FROM user_id_mappings
                   WHERE user_id = ?""",
                (user_id,),
            )
        return [UserIdMapping.from_row(row) for row in cursor.fetchall()]

    async def close(self) -> None:
        # Close all tracked connections
        with self._connections_lock:
            for conn in self._connections:
                try:
                    conn.close()
                except (OSError, RuntimeError, ValueError) as e:
                    logger.debug("Error closing connection: %s", e)
                    pass
            self._connections.clear()


__all__ = [
    "SQLiteIntegrationStore",
]
