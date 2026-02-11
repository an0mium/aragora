"""
PostgreSQL integration store backend.

Async implementation for production multi-instance deployments
with horizontal scaling and concurrent writes.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from aragora.storage.integration_backends import IntegrationStoreBackend
from aragora.storage.integration_models import (
    IntegrationConfig,
    UserIdMapping,
    _decrypt_settings,
    _encrypt_settings,
    _record_user_mapping_operation,
)
from aragora.utils.async_utils import run_async

if TYPE_CHECKING:
    from asyncpg import Pool

logger = logging.getLogger(__name__)


class PostgresIntegrationStore(IntegrationStoreBackend):
    """
    PostgreSQL-backed integration store.

    Async implementation for production multi-instance deployments
    with horizontal scaling and concurrent writes.
    """

    SCHEMA_NAME = "integrations"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS integrations (
            integration_type TEXT NOT NULL,
            user_id TEXT NOT NULL DEFAULT 'default',
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            notify_on_consensus BOOLEAN DEFAULT TRUE,
            notify_on_debate_end BOOLEAN DEFAULT TRUE,
            notify_on_error BOOLEAN DEFAULT FALSE,
            notify_on_leaderboard BOOLEAN DEFAULT FALSE,
            settings_json JSONB,
            messages_sent INTEGER DEFAULT 0,
            errors_24h INTEGER DEFAULT 0,
            last_activity TIMESTAMPTZ,
            last_error TEXT,
            workspace_id TEXT,
            PRIMARY KEY (user_id, integration_type)
        );
        CREATE INDEX IF NOT EXISTS idx_integrations_user ON integrations(user_id);
        CREATE INDEX IF NOT EXISTS idx_integrations_type ON integrations(integration_type);

        CREATE TABLE IF NOT EXISTS user_id_mappings (
            email TEXT NOT NULL,
            platform TEXT NOT NULL,
            platform_user_id TEXT NOT NULL,
            display_name TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            user_id TEXT NOT NULL DEFAULT 'default',
            PRIMARY KEY (user_id, platform, email)
        );
        CREATE INDEX IF NOT EXISTS idx_mappings_email ON user_id_mappings(email);
        CREATE INDEX IF NOT EXISTS idx_mappings_platform ON user_id_mappings(platform);
    """

    def __init__(self, pool: "Pool"):
        self._pool = pool
        self._initialized = False
        logger.info("PostgresIntegrationStore initialized")

    async def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        async with self._pool.acquire() as conn:
            await conn.execute(self.INITIAL_SCHEMA)

        self._initialized = True
        logger.debug(f"[{self.SCHEMA_NAME}] Schema initialized")

    def _row_to_config(self, row: Any) -> IntegrationConfig:
        """Convert database row to IntegrationConfig (settings decryption done at store level)."""
        return IntegrationConfig(
            type=row["integration_type"],
            enabled=bool(row["enabled"]),
            created_at=row["created_at"] or time.time(),
            updated_at=row["updated_at"] or time.time(),
            notify_on_consensus=bool(row["notify_on_consensus"]),
            notify_on_debate_end=bool(row["notify_on_debate_end"]),
            notify_on_error=bool(row["notify_on_error"]),
            notify_on_leaderboard=bool(row["notify_on_leaderboard"]),
            settings=json.loads(row["settings_json"]) if row["settings_json"] else {},
            messages_sent=row["messages_sent"] or 0,
            errors_24h=row["errors_24h"] or 0,
            last_activity=row["last_activity"],
            last_error=row["last_error"],
            user_id=row["user_id"],
            workspace_id=row["workspace_id"],
        )

    def _row_to_mapping(self, row: Any) -> UserIdMapping:
        """Convert database row to UserIdMapping."""
        return UserIdMapping(
            email=row["email"],
            platform=row["platform"],
            platform_user_id=row["platform_user_id"],
            display_name=row["display_name"],
            created_at=row["created_at"] or time.time(),
            updated_at=row["updated_at"] or time.time(),
            user_id=row["user_id"],
        )

    async def get(
        self, integration_type: str, user_id: str = "default"
    ) -> IntegrationConfig | None:
        """Get integration configuration (async)."""
        return await self.get_async(integration_type, user_id)

    async def get_async(
        self, integration_type: str, user_id: str = "default"
    ) -> IntegrationConfig | None:
        """Get integration configuration asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT integration_type, enabled,
                          EXTRACT(EPOCH FROM created_at) as created_at,
                          EXTRACT(EPOCH FROM updated_at) as updated_at,
                          notify_on_consensus, notify_on_debate_end, notify_on_error,
                          notify_on_leaderboard, settings_json, messages_sent,
                          errors_24h,
                          EXTRACT(EPOCH FROM last_activity) as last_activity,
                          last_error, user_id, workspace_id
                   FROM integrations WHERE user_id = $1 AND integration_type = $2""",
                user_id,
                integration_type,
            )
            if row:
                config = self._row_to_config(row)
                # Decrypt settings with AAD for integrity verification
                config.settings = _decrypt_settings(config.settings, user_id, integration_type)
                return config
            return None

    def get_sync(self, integration_type: str, user_id: str = "default") -> IntegrationConfig | None:
        """Get integration configuration (sync wrapper for async)."""
        return run_async(self.get_async(integration_type, user_id))

    async def save(self, config: IntegrationConfig) -> None:
        """Save integration configuration (async)."""
        await self.save_async(config)

    async def save_async(self, config: IntegrationConfig) -> None:
        """Save integration configuration asynchronously."""
        config.updated_at = time.time()
        user_id = config.user_id or "default"
        # Encrypt settings with AAD binding to user + integration type
        encrypted_settings = _encrypt_settings(config.settings, user_id, config.type)

        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO integrations
                   (integration_type, user_id, enabled, created_at, updated_at,
                    notify_on_consensus, notify_on_debate_end, notify_on_error,
                    notify_on_leaderboard, settings_json, messages_sent, errors_24h,
                    last_activity, last_error, workspace_id)
                   VALUES ($1, $2, $3, to_timestamp($4), to_timestamp($5),
                           $6, $7, $8, $9, $10, $11, $12,
                           CASE WHEN $13::float IS NOT NULL THEN to_timestamp($13) ELSE NULL END,
                           $14, $15)
                   ON CONFLICT (user_id, integration_type) DO UPDATE SET
                    enabled = EXCLUDED.enabled,
                    updated_at = EXCLUDED.updated_at,
                    notify_on_consensus = EXCLUDED.notify_on_consensus,
                    notify_on_debate_end = EXCLUDED.notify_on_debate_end,
                    notify_on_error = EXCLUDED.notify_on_error,
                    notify_on_leaderboard = EXCLUDED.notify_on_leaderboard,
                    settings_json = EXCLUDED.settings_json,
                    messages_sent = EXCLUDED.messages_sent,
                    errors_24h = EXCLUDED.errors_24h,
                    last_activity = EXCLUDED.last_activity,
                    last_error = EXCLUDED.last_error,
                    workspace_id = EXCLUDED.workspace_id""",
                config.type,
                user_id,
                config.enabled,
                config.created_at,
                config.updated_at,
                config.notify_on_consensus,
                config.notify_on_debate_end,
                config.notify_on_error,
                config.notify_on_leaderboard,
                json.dumps(encrypted_settings),
                config.messages_sent,
                config.errors_24h,
                config.last_activity,
                config.last_error,
                config.workspace_id,
            )
        logger.debug(f"Saved integration: {config.type} for user {user_id}")

    def save_sync(self, config: IntegrationConfig) -> None:
        """Save integration configuration (sync wrapper for async)."""
        run_async(self.save_async(config))

    async def delete(self, integration_type: str, user_id: str = "default") -> bool:
        """Delete integration configuration (async)."""
        return await self.delete_async(integration_type, user_id)

    async def delete_async(self, integration_type: str, user_id: str = "default") -> bool:
        """Delete integration configuration asynchronously."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM integrations WHERE user_id = $1 AND integration_type = $2",
                user_id,
                integration_type,
            )
            deleted = result != "DELETE 0"
            if deleted:
                logger.debug(f"Deleted integration: {integration_type} for user {user_id}")
            return deleted

    def delete_sync(self, integration_type: str, user_id: str = "default") -> bool:
        """Delete integration configuration (sync wrapper for async)."""
        return run_async(self.delete_async(integration_type, user_id))

    async def list_for_user(self, user_id: str = "default") -> list[IntegrationConfig]:
        """List all integrations for a user (async)."""
        return await self.list_for_user_async(user_id)

    async def list_for_user_async(self, user_id: str = "default") -> list[IntegrationConfig]:
        """List all integrations for a user asynchronously."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT integration_type, enabled,
                          EXTRACT(EPOCH FROM created_at) as created_at,
                          EXTRACT(EPOCH FROM updated_at) as updated_at,
                          notify_on_consensus, notify_on_debate_end, notify_on_error,
                          notify_on_leaderboard, settings_json, messages_sent,
                          errors_24h,
                          EXTRACT(EPOCH FROM last_activity) as last_activity,
                          last_error, user_id, workspace_id
                   FROM integrations WHERE user_id = $1""",
                user_id,
            )
            configs = []
            for row in rows:
                config = self._row_to_config(row)
                config.settings = _decrypt_settings(config.settings, user_id, config.type)
                configs.append(config)
            return configs

    def list_for_user_sync(self, user_id: str = "default") -> list[IntegrationConfig]:
        """List all integrations for a user (sync wrapper for async)."""
        return run_async(self.list_for_user_async(user_id))

    async def list_all(self, limit: int = 1000) -> list[IntegrationConfig]:
        """List all integrations (async)."""
        return await self.list_all_async(limit)

    async def list_all_async(self, limit: int = 1000) -> list[IntegrationConfig]:
        """List all integrations asynchronously."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT integration_type, enabled,
                          EXTRACT(EPOCH FROM created_at) as created_at,
                          EXTRACT(EPOCH FROM updated_at) as updated_at,
                          notify_on_consensus, notify_on_debate_end, notify_on_error,
                          notify_on_leaderboard, settings_json, messages_sent,
                          errors_24h,
                          EXTRACT(EPOCH FROM last_activity) as last_activity,
                          last_error, user_id, workspace_id
                   FROM integrations
                   LIMIT $1""",
                limit,
            )
            configs = []
            for row in rows:
                config = self._row_to_config(row)
                config.settings = _decrypt_settings(
                    config.settings, config.user_id or "default", config.type
                )
                configs.append(config)
            return configs

    def list_all_sync(self, limit: int = 1000) -> list[IntegrationConfig]:
        """List all integrations (sync wrapper for async)."""
        return run_async(self.list_all_async(limit))

    async def get_user_mapping(
        self, email: str, platform: str, user_id: str = "default"
    ) -> UserIdMapping | None:
        """Get user ID mapping for a platform (async)."""
        return await self.get_user_mapping_async(email, platform, user_id)

    async def get_user_mapping_async(
        self, email: str, platform: str, user_id: str = "default"
    ) -> UserIdMapping | None:
        """Get user ID mapping for a platform asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT email, platform, platform_user_id, display_name,
                          EXTRACT(EPOCH FROM created_at) as created_at,
                          EXTRACT(EPOCH FROM updated_at) as updated_at,
                          user_id
                   FROM user_id_mappings
                   WHERE user_id = $1 AND platform = $2 AND email = $3""",
                user_id,
                platform,
                email,
            )
            if row:
                _record_user_mapping_operation("get", platform, True)
                return self._row_to_mapping(row)
            _record_user_mapping_operation("get", platform, False)
            return None

    def get_user_mapping_sync(
        self, email: str, platform: str, user_id: str = "default"
    ) -> UserIdMapping | None:
        """Get user ID mapping (sync wrapper for async)."""
        return run_async(self.get_user_mapping_async(email, platform, user_id))

    async def save_user_mapping(self, mapping: UserIdMapping) -> None:
        """Save user ID mapping (async)."""
        await self.save_user_mapping_async(mapping)

    async def save_user_mapping_async(self, mapping: UserIdMapping) -> None:
        """Save user ID mapping asynchronously."""
        mapping.updated_at = time.time()

        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO user_id_mappings
                   (email, platform, platform_user_id, display_name,
                    created_at, updated_at, user_id)
                   VALUES ($1, $2, $3, $4, to_timestamp($5), to_timestamp($6), $7)
                   ON CONFLICT (user_id, platform, email) DO UPDATE SET
                    platform_user_id = EXCLUDED.platform_user_id,
                    display_name = EXCLUDED.display_name,
                    updated_at = EXCLUDED.updated_at""",
                mapping.email,
                mapping.platform,
                mapping.platform_user_id,
                mapping.display_name,
                mapping.created_at,
                mapping.updated_at,
                mapping.user_id,
            )
        _record_user_mapping_operation("save", mapping.platform, True)
        logger.debug(f"Saved user mapping: {mapping.email} -> {mapping.platform}")

    def save_user_mapping_sync(self, mapping: UserIdMapping) -> None:
        """Save user ID mapping (sync wrapper for async)."""
        run_async(self.save_user_mapping_async(mapping))

    async def delete_user_mapping(
        self, email: str, platform: str, user_id: str = "default"
    ) -> bool:
        """Delete user ID mapping (async)."""
        return await self.delete_user_mapping_async(email, platform, user_id)

    async def delete_user_mapping_async(
        self, email: str, platform: str, user_id: str = "default"
    ) -> bool:
        """Delete user ID mapping asynchronously."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM user_id_mappings WHERE user_id = $1 AND platform = $2 AND email = $3",
                user_id,
                platform,
                email,
            )
            deleted = result != "DELETE 0"
            _record_user_mapping_operation("delete", platform, deleted)
            return deleted

    def delete_user_mapping_sync(self, email: str, platform: str, user_id: str = "default") -> bool:
        """Delete user ID mapping (sync wrapper for async)."""
        return run_async(self.delete_user_mapping_async(email, platform, user_id))

    async def list_user_mappings(
        self, platform: str | None = None, user_id: str = "default"
    ) -> list[UserIdMapping]:
        """List user ID mappings (async)."""
        return await self.list_user_mappings_async(platform, user_id)

    async def list_user_mappings_async(
        self, platform: str | None = None, user_id: str = "default"
    ) -> list[UserIdMapping]:
        """List user ID mappings asynchronously."""
        async with self._pool.acquire() as conn:
            if platform:
                rows = await conn.fetch(
                    """SELECT email, platform, platform_user_id, display_name,
                              EXTRACT(EPOCH FROM created_at) as created_at,
                              EXTRACT(EPOCH FROM updated_at) as updated_at,
                              user_id
                       FROM user_id_mappings
                       WHERE user_id = $1 AND platform = $2""",
                    user_id,
                    platform,
                )
            else:
                rows = await conn.fetch(
                    """SELECT email, platform, platform_user_id, display_name,
                              EXTRACT(EPOCH FROM created_at) as created_at,
                              EXTRACT(EPOCH FROM updated_at) as updated_at,
                              user_id
                       FROM user_id_mappings
                       WHERE user_id = $1""",
                    user_id,
                )
            return [self._row_to_mapping(row) for row in rows]

    def list_user_mappings_sync(
        self, platform: str | None = None, user_id: str = "default"
    ) -> list[UserIdMapping]:
        """List user ID mappings (sync wrapper for async)."""
        return run_async(self.list_user_mappings_async(platform, user_id))

    async def close(self) -> None:
        """Close is a no-op for pool-based stores (pool managed externally)."""
        pass


__all__ = [
    "PostgresIntegrationStore",
]
