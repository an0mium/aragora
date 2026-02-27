"""
VerticalsConfigRepository - Persistence for vertical configuration overrides.

Stores user-customized vertical configs (tools, compliance frameworks, model
config) in SQLite so they survive server restarts. On startup, saved configs
are loaded and applied on top of registry defaults.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aragora.config import resolve_db_path
from aragora.persistence.db_config import DatabaseType

logger = logging.getLogger(__name__)


class VerticalsConfigRepository:
    """Repository for persisting vertical configuration overrides."""

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            self._db_path = self._get_default_db_path()
        else:
            self._db_path = Path(resolve_db_path(db_path))
        self._init_schema()

    def _get_default_db_path(self) -> Path:
        from aragora.persistence.db_config import get_db_path

        return get_db_path(DatabaseType.ONBOARDING)

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vertical_configs (
                    vertical_id TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    updated_by TEXT DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (vertical_id)
                )
            """)
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def save_config(
        self,
        vertical_id: str,
        config: dict[str, Any],
        updated_by: str = "",
    ) -> None:
        """Save or update vertical config override."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO vertical_configs (vertical_id, config_json, updated_by, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(vertical_id) DO UPDATE SET
                    config_json = excluded.config_json,
                    updated_by = excluded.updated_by,
                    updated_at = excluded.updated_at
                """,
                (vertical_id, json.dumps(config), updated_by, now),
            )
            conn.commit()
        logger.info("Saved vertical config for %s", vertical_id)

    def get_config(self, vertical_id: str) -> dict[str, Any] | None:
        """Load saved config override for a vertical."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT config_json FROM vertical_configs WHERE vertical_id = ?",
                (vertical_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["config_json"])

    def get_all_configs(self) -> dict[str, dict[str, Any]]:
        """Load all saved config overrides."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT vertical_id, config_json FROM vertical_configs").fetchall()
        return {row["vertical_id"]: json.loads(row["config_json"]) for row in rows}

    def delete_config(self, vertical_id: str) -> bool:
        """Delete a saved config override."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM vertical_configs WHERE vertical_id = ?",
                (vertical_id,),
            )
            conn.commit()
        return cursor.rowcount > 0


_repository: VerticalsConfigRepository | None = None


def get_verticals_config_repository() -> VerticalsConfigRepository:
    """Get or create the singleton repository."""
    global _repository
    if _repository is None:
        _repository = VerticalsConfigRepository()
    return _repository
