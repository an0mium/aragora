"""
Schema versioning for SQLite databases.

Provides a simple migration framework for tracking and upgrading
database schemas across versions.

Usage:
    from aragora.storage.schema import SchemaManager

    manager = SchemaManager(conn, "my_module", current_version=2)

    manager.register_migration(1, 2, '''
        ALTER TABLE my_table ADD COLUMN new_field TEXT;
    ''')

    manager.ensure_schema()  # Runs any pending migrations
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """A database migration from one version to another."""

    from_version: int
    to_version: int
    sql: Optional[str] = None
    function: Optional[Callable[[sqlite3.Connection], None]] = None
    description: str = ""

    def apply(self, conn: sqlite3.Connection) -> None:
        """Apply this migration to the database."""
        if self.sql:
            conn.executescript(self.sql)
        elif self.function:
            self.function(conn)
        else:
            raise ValueError("Migration must have either sql or function")


class SchemaManager:
    """
    Manages schema versioning and migrations for a SQLite database.

    Tracks version in a _schema_versions table and runs pending migrations
    to bring the database up to the current version.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        module_name: str,
        current_version: int = 1,
    ):
        """
        Initialize schema manager.

        Args:
            conn: SQLite connection
            module_name: Unique identifier for this schema (e.g., "elo", "memory")
            current_version: The version this code expects
        """
        self.conn = conn
        self.module_name = module_name
        self.current_version = current_version
        self.migrations: list[Migration] = []

        self._ensure_version_table()

    def _ensure_version_table(self) -> None:
        """Create the schema versions table if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS _schema_versions (
                module TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def get_version(self) -> int:
        """Get the current schema version for this module."""
        cursor = self.conn.execute(
            "SELECT version FROM _schema_versions WHERE module = ?",
            (self.module_name,)
        )
        row = cursor.fetchone()
        return row[0] if row else 0

    def set_version(self, version: int) -> None:
        """Set the schema version for this module."""
        self.conn.execute("""
            INSERT INTO _schema_versions (module, version, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(module) DO UPDATE SET
                version = excluded.version,
                updated_at = CURRENT_TIMESTAMP
        """, (self.module_name, version))
        self.conn.commit()

    def register_migration(
        self,
        from_version: int,
        to_version: int,
        sql: Optional[str] = None,
        function: Optional[Callable[[sqlite3.Connection], None]] = None,
        description: str = "",
    ) -> None:
        """
        Register a migration between versions.

        Args:
            from_version: Version to migrate from
            to_version: Version to migrate to
            sql: SQL script to execute (either sql or function required)
            function: Python function to execute
            description: Human-readable description
        """
        migration = Migration(
            from_version=from_version,
            to_version=to_version,
            sql=sql,
            function=function,
            description=description,
        )
        self.migrations.append(migration)
        # Keep migrations sorted by from_version
        self.migrations.sort(key=lambda m: m.from_version)

    def get_pending_migrations(self) -> list[Migration]:
        """Get list of migrations needed to reach current version."""
        current = self.get_version()
        pending = []

        for migration in self.migrations:
            if migration.from_version >= current and migration.to_version <= self.current_version:
                pending.append(migration)

        return pending

    def ensure_schema(self, initial_schema: Optional[str] = None) -> bool:
        """
        Ensure the database schema is up to date.

        Args:
            initial_schema: SQL to create initial tables (version 1)

        Returns:
            True if migrations were applied, False if already up to date
        """
        current = self.get_version()

        applied = False

        if current == 0 and initial_schema:
            # Fresh database - create initial schema
            logger.info(f"[{self.module_name}] Creating initial schema (v1)")
            self.conn.executescript(initial_schema)
            self.set_version(1)
            current = 1
            applied = True

        if current == self.current_version:
            return applied  # Already at target version

        if current > self.current_version:
            logger.warning(
                f"[{self.module_name}] Database version ({current}) is newer than "
                f"code version ({self.current_version}). Skipping migrations."
            )
            return False

        # Apply pending migrations
        pending = self.get_pending_migrations()
        if not pending:
            # No registered migrations, just update version
            self.set_version(self.current_version)
            return True

        for migration in pending:
            if migration.from_version == current:
                desc = migration.description or f"v{migration.from_version} -> v{migration.to_version}"
                logger.info(f"[{self.module_name}] Running migration: {desc}")
                try:
                    migration.apply(self.conn)
                    current = migration.to_version
                    self.set_version(current)
                except Exception as e:
                    logger.error(f"[{self.module_name}] Migration failed: {e}")
                    raise

        return True

    def validate_schema(self, expected_tables: list[str]) -> dict:
        """
        Validate that expected tables exist.

        Args:
            expected_tables: List of table names that should exist

        Returns:
            Dict with validation results
        """
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        existing = {row[0] for row in cursor.fetchall()}

        missing = [t for t in expected_tables if t not in existing]
        extra = [t for t in existing if t not in expected_tables and not t.startswith("_")]

        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "extra": extra,
            "version": self.get_version(),
        }


def safe_add_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    column_type: str,
    default: Optional[str] = None,
) -> bool:
    """
    Safely add a column to a table if it doesn't exist.

    Args:
        conn: SQLite connection
        table: Table name
        column: Column name to add
        column_type: SQL type (e.g., "TEXT", "INTEGER")
        default: Optional default value

    Returns:
        True if column was added, False if it already existed
    """
    # Check if column exists
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = {row[1] for row in cursor.fetchall()}

    if column in columns:
        return False

    # Add the column
    sql = f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
    if default is not None:
        sql += f" DEFAULT {default}"

    conn.execute(sql)
    conn.commit()
    logger.debug(f"Added column {column} to {table}")
    return True
