"""
Migration runner for Aragora database schema management.

Provides a lightweight alternative to Alembic for managing schema changes
across SQLite and PostgreSQL backends.

Features:
- Advisory locking for PostgreSQL to prevent concurrent migration runs
- Version tracking with applied_by metadata
- Support for SQL and Python migration functions
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from aragora.storage.backends import (
    POSTGRESQL_AVAILABLE,
    DatabaseBackend,
    PostgreSQLBackend,
    SQLiteBackend,
)

logger = logging.getLogger(__name__)

# Advisory lock ID for migration coordination (hash of 'aragora_migration')
MIGRATION_LOCK_ID = 2089872453


@dataclass
class Migration:
    """
    Represents a database migration.

    Attributes:
        version: Unique version number (use timestamps like 20240115120000)
        name: Human-readable name
        up_sql: SQL to apply the migration (can be None if using up_fn)
        down_sql: SQL to rollback the migration (can be None if using down_fn)
        up_fn: Python function to apply migration (alternative to up_sql)
        down_fn: Python function to rollback migration (alternative to down_sql)
    """

    version: int
    name: str
    up_sql: str | None = None
    down_sql: str | None = None
    up_fn: Optional[Callable[[DatabaseBackend], None]] = None
    down_fn: Optional[Callable[[DatabaseBackend], None]] = None

    def __post_init__(self) -> None:
        if not self.up_sql and not self.up_fn:
            raise ValueError(f"Migration {self.version} must have up_sql or up_fn")


class MigrationRunner:
    """
    Manages and executes database migrations.

    Tracks applied migrations in a _migrations table and supports both
    SQLite and PostgreSQL backends.
    """

    MIGRATIONS_TABLE = "_aragora_migrations"

    def __init__(
        self,
        backend: DatabaseBackend | None = None,
        db_path: str = "aragora.db",
        database_url: str | None = None,
    ):
        """
        Initialize the migration runner.

        Args:
            backend: Existing database backend to use.
            db_path: SQLite database path (if no backend provided).
            database_url: PostgreSQL URL (if no backend provided).
        """
        if backend:
            self._backend = backend
        else:
            # Create backend from parameters
            env_url = os.environ.get("DATABASE_URL") or os.environ.get("ARAGORA_DATABASE_URL")
            actual_url = database_url or env_url

            if actual_url:
                if not POSTGRESQL_AVAILABLE:
                    raise ImportError(
                        "psycopg2 required for PostgreSQL. Install with: pip install psycopg2-binary"
                    )
                self._backend = PostgreSQLBackend(actual_url)
            else:
                self._backend = SQLiteBackend(db_path)

        self._migrations: list[Migration] = []
        self._init_migrations_table()

    def _init_migrations_table(self) -> None:
        """Create the migrations tracking table if it doesn't exist."""
        # Use BIGINT for PostgreSQL to support timestamp-based version numbers
        version_type = "BIGINT" if isinstance(self._backend, PostgreSQLBackend) else "INTEGER"
        sql = f"""
            CREATE TABLE IF NOT EXISTS {self.MIGRATIONS_TABLE} (
                version {version_type} PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                applied_by TEXT DEFAULT NULL
            )
        """
        self._backend.execute_write(sql)

    def _acquire_migration_lock(self, timeout_seconds: float = 30.0) -> bool:
        """
        Acquire an advisory lock for running migrations (PostgreSQL only).

        Uses pg_try_advisory_lock to prevent concurrent migration runs across
        multiple pods/instances. SQLite uses file-level locking inherently.

        Args:
            timeout_seconds: Maximum time to wait for lock acquisition.

        Returns:
            True if lock acquired.

        Raises:
            RuntimeError: If lock cannot be acquired within timeout.
        """
        if not isinstance(self._backend, PostgreSQLBackend):
            # SQLite has inherent file locking, no advisory lock needed
            return True

        start_time = time.time()
        poll_interval = 0.5  # seconds between retry attempts

        while True:
            # Try to acquire advisory lock (non-blocking)
            result = self._backend.fetch_one(f"SELECT pg_try_advisory_lock({MIGRATION_LOCK_ID})")

            if result and result[0]:
                logger.info("Acquired migration advisory lock")
                return True

            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                logger.error(
                    f"Failed to acquire migration lock after {elapsed:.1f}s. "
                    "Another migration may be in progress."
                )
                raise RuntimeError(
                    f"Migration lock acquisition timeout after {timeout_seconds}s. "
                    "Ensure no other migration is running and try again."
                )

            logger.debug(f"Migration lock held by another process, retrying in {poll_interval}s...")
            time.sleep(poll_interval)

    def _release_migration_lock(self) -> None:
        """
        Release the advisory lock for migrations (PostgreSQL only).

        Safe to call even if lock was not acquired (no-op for SQLite).
        """
        if not isinstance(self._backend, PostgreSQLBackend):
            return

        try:
            self._backend.execute_write(f"SELECT pg_advisory_unlock({MIGRATION_LOCK_ID})")
            logger.info("Released migration advisory lock")
        except Exception as e:
            # Log but don't raise - lock will be released on connection close anyway
            logger.warning(f"Failed to release migration lock: {e}")

    def _get_applied_by(self) -> str:
        """Get identifier for who applied the migration."""
        hostname = socket.gethostname()
        pid = os.getpid()
        return f"{hostname}:{pid}"

    def register(self, migration: Migration) -> None:
        """
        Register a migration.

        Args:
            migration: Migration to register.
        """
        # Insert in version order
        self._migrations.append(migration)
        self._migrations.sort(key=lambda m: m.version)

    def get_applied_versions(self) -> set[int]:
        """Get set of applied migration versions."""
        rows = self._backend.fetch_all(f"SELECT version FROM {self.MIGRATIONS_TABLE}")
        return {row[0] for row in rows}

    def get_pending_migrations(self) -> list[Migration]:
        """Get list of migrations that haven't been applied."""
        applied = self.get_applied_versions()
        return [m for m in self._migrations if m.version not in applied]

    def upgrade(
        self,
        target_version: int | None = None,
        lock_timeout: float = 30.0,
    ) -> list[Migration]:
        """
        Apply pending migrations up to target version.

        Acquires an advisory lock (PostgreSQL) to prevent concurrent migrations
        across multiple pods/instances.

        Args:
            target_version: Maximum version to apply (None = all pending).
            lock_timeout: Maximum seconds to wait for migration lock.

        Returns:
            List of applied migrations.

        Raises:
            RuntimeError: If migration lock cannot be acquired.
        """
        applied: list[Migration] = []
        pending = self.get_pending_migrations()

        if not pending:
            return applied

        # Acquire lock before running migrations
        self._acquire_migration_lock(timeout_seconds=lock_timeout)

        try:
            applied_by = self._get_applied_by()

            for migration in pending:
                if target_version and migration.version > target_version:
                    break

                logger.info(f"Applying migration {migration.version}: {migration.name}")

                try:
                    if migration.up_fn:
                        migration.up_fn(self._backend)
                    elif migration.up_sql:
                        # Split by semicolon and execute each statement
                        for stmt in migration.up_sql.split(";"):
                            stmt = stmt.strip()
                            if stmt:
                                self._backend.execute_write(stmt)

                    # Record migration with applied_by metadata
                    self._backend.execute_write(
                        f"INSERT INTO {self.MIGRATIONS_TABLE} (version, name, applied_by) "
                        "VALUES (?, ?, ?)",
                        (migration.version, migration.name, applied_by),
                    )
                    applied.append(migration)
                    logger.info(f"Applied migration {migration.version}")

                except Exception as e:
                    logger.error(f"Failed to apply migration {migration.version}: {e}")
                    raise
        finally:
            # Always release lock
            self._release_migration_lock()

        return applied

    def downgrade(
        self,
        target_version: int | None = None,
        lock_timeout: float = 30.0,
    ) -> list[Migration]:
        """
        Rollback migrations down to target version.

        Acquires an advisory lock (PostgreSQL) to prevent concurrent migrations
        across multiple pods/instances.

        Args:
            target_version: Minimum version to keep (None = rollback one).
            lock_timeout: Maximum seconds to wait for migration lock.

        Returns:
            List of rolled back migrations.

        Raises:
            RuntimeError: If migration lock cannot be acquired.
        """
        rolled_back: list[Migration] = []
        applied = self.get_applied_versions()

        # Get applied migrations in reverse order
        to_rollback = [m for m in reversed(self._migrations) if m.version in applied]

        if not to_rollback:
            logger.info("No migrations to rollback")
            return rolled_back

        # Acquire lock before rolling back
        self._acquire_migration_lock(timeout_seconds=lock_timeout)

        try:
            for migration in to_rollback:
                if target_version and migration.version <= target_version:
                    break

                if not migration.down_sql and not migration.down_fn:
                    logger.warning(f"Migration {migration.version} has no rollback")
                    break

                logger.info(f"Rolling back migration {migration.version}: {migration.name}")

                try:
                    if migration.down_fn:
                        migration.down_fn(self._backend)
                    elif migration.down_sql:
                        for stmt in migration.down_sql.split(";"):
                            stmt = stmt.strip()
                            if stmt:
                                self._backend.execute_write(stmt)

                    # Remove migration record
                    self._backend.execute_write(
                        f"DELETE FROM {self.MIGRATIONS_TABLE} WHERE version = ?",
                        (migration.version,),
                    )
                    rolled_back.append(migration)
                    logger.info(f"Rolled back migration {migration.version}")

                except Exception as e:
                    logger.error(f"Failed to rollback migration {migration.version}: {e}")
                    raise

                # Only rollback one if no target specified
                if target_version is None:
                    break
        finally:
            # Always release lock
            self._release_migration_lock()

        return rolled_back

    def status(self) -> dict:
        """
        Get migration status.

        Returns:
            Dict with applied, pending, and latest version info.
        """
        applied = self.get_applied_versions()
        pending = self.get_pending_migrations()

        return {
            "applied_count": len(applied),
            "pending_count": len(pending),
            "applied_versions": sorted(applied),
            "pending_versions": [m.version for m in pending],
            "latest_applied": max(applied) if applied else None,
            "latest_available": self._migrations[-1].version if self._migrations else None,
        }

    def close(self) -> None:
        """Close the database connection."""
        self._backend.close()


# Global runner instance with thread-safe initialization
_runner: MigrationRunner | None = None
_runner_lock = threading.Lock()


def get_migration_runner(
    db_path: str = "aragora.db",
    database_url: str | None = None,
) -> MigrationRunner:
    """
    Get or create the global migration runner (thread-safe).

    Automatically loads migrations from the migrations package.
    """
    global _runner
    if _runner is None:
        with _runner_lock:
            # Double-checked locking pattern
            if _runner is None:
                _runner = MigrationRunner(db_path=db_path, database_url=database_url)
                _load_migrations(_runner)
    return _runner


def _load_migrations(runner: MigrationRunner) -> None:
    """Load all migration modules from the versions package."""
    try:
        from aragora.migrations import versions

        versions_path = Path(versions.__file__).parent

        for _, name, _ in pkgutil.iter_modules([str(versions_path)]):
            if name.startswith("v"):
                module = importlib.import_module(f"aragora.migrations.versions.{name}")
                if hasattr(module, "migration"):
                    runner.register(module.migration)
                    logger.debug(f"Loaded migration: {name}")
    except ImportError:
        logger.debug("No migrations.versions package found")


def apply_migrations(
    db_path: str = "aragora.db",
    database_url: str | None = None,
    target_version: int | None = None,
) -> list[Migration]:
    """
    Apply all pending migrations.

    Args:
        db_path: SQLite database path.
        database_url: PostgreSQL URL.
        target_version: Maximum version to apply.

    Returns:
        List of applied migrations.
    """
    runner = get_migration_runner(db_path, database_url)
    return runner.upgrade(target_version)


def rollback_migration(
    db_path: str = "aragora.db",
    database_url: str | None = None,
    target_version: int | None = None,
) -> list[Migration]:
    """
    Rollback the last migration.

    Args:
        db_path: SQLite database path.
        database_url: PostgreSQL URL.
        target_version: Minimum version to keep.

    Returns:
        List of rolled back migrations.
    """
    runner = get_migration_runner(db_path, database_url)
    return runner.downgrade(target_version)


def get_migration_status(
    db_path: str = "aragora.db",
    database_url: str | None = None,
) -> dict:
    """
    Get migration status.

    Returns:
        Dict with applied/pending counts and versions.
    """
    runner = get_migration_runner(db_path, database_url)
    return runner.status()


def reset_runner() -> None:
    """Reset the global runner (for testing)."""
    global _runner
    if _runner:
        _runner.close()
        _runner = None
