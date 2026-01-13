"""
Database Migration Runner for Aragora.

Provides CLI tools for running database migrations in production.

Usage:
    # Check migration status
    python -m aragora.persistence.migrations.runner --status

    # Dry-run migrations (show what would be done)
    python -m aragora.persistence.migrations.runner --dry-run

    # Run migrations
    python -m aragora.persistence.migrations.runner --migrate

    # Create a new migration
    python -m aragora.persistence.migrations.runner --create "Add user lockout fields"

    # Rollback last migration (not recommended in production)
    python -m aragora.persistence.migrations.runner --rollback
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from aragora.storage.schema import SchemaManager, get_wal_connection

logger = logging.getLogger(__name__)

# Default database paths (relative to ARAGORA_DATA_DIR or absolute)
DEFAULT_DB_PATHS = {
    "elo": "aragora_elo.db",
    "memory": "aragora_memory.db",
    "users": "aragora_users.db",
    "positions": "aragora_positions.db",
    "replay": "aragora_replay.db",
    "tokens": "aragora_tokens.db",
}


@dataclass
class MigrationFile:
    """Represents a migration file."""

    version: int
    name: str
    path: Path
    description: str = ""

    @property
    def module_name(self) -> str:
        """Get the Python module name for this migration."""
        return self.path.stem


@dataclass
class MigrationStatus:
    """Status of a database's migrations."""

    db_name: str
    db_path: str
    current_version: int
    target_version: int
    pending_count: int
    applied_migrations: list[int]
    pending_migrations: list[int]


class MigrationRunner:
    """
    Runs database migrations for Aragora.

    Discovers migration files and applies them in order.
    """

    MIGRATIONS_DIR = Path(__file__).parent

    def __init__(
        self,
        nomic_dir: Optional[Path] = None,
        db_paths: Optional[dict[str, str]] = None,
    ):
        """
        Initialize the migration runner.

        Args:
            nomic_dir: Base directory for database files
            db_paths: Override default database paths
        """
        self.nomic_dir = nomic_dir or self._get_nomic_dir()
        self.db_paths = db_paths or DEFAULT_DB_PATHS
        self._discovered_migrations: dict[str, list[MigrationFile]] = {}

    @staticmethod
    def _get_nomic_dir() -> Path:
        """Get the data directory from environment or default."""
        env_dir = (
            os.environ.get("ARAGORA_DATA_DIR")
            or os.environ.get("ARAGORA_NOMIC_DIR")
            or os.environ.get("NOMIC_DIR")
        )
        if env_dir:
            return Path(env_dir)

        # Default to .nomic (relative to current working directory)
        return Path(".nomic")

    def discover_migrations(self, db_name: str) -> list[MigrationFile]:
        """
        Discover migration files for a specific database.

        Migration files should be named like:
        - 001_initial.py
        - 002_add_lockout.py
        - 003_add_analytics.py

        Args:
            db_name: Name of the database module (e.g., "elo", "users")

        Returns:
            List of MigrationFile objects sorted by version
        """
        if db_name in self._discovered_migrations:
            return self._discovered_migrations[db_name]

        migrations = []
        migrations_path = self.MIGRATIONS_DIR / db_name

        if not migrations_path.exists():
            logger.debug(f"No migrations directory for {db_name}")
            return []

        # Pattern: NNN_description.py
        pattern = re.compile(r"^(\d{3})_(.+)\.py$")

        for file_path in sorted(migrations_path.glob("*.py")):
            if file_path.name.startswith("_"):
                continue

            match = pattern.match(file_path.name)
            if match:
                version = int(match.group(1))
                name = match.group(2)
                migrations.append(
                    MigrationFile(
                        version=version,
                        name=name,
                        path=file_path,
                        description=name.replace("_", " ").title(),
                    )
                )

        # Sort by version
        migrations.sort(key=lambda m: m.version)
        self._discovered_migrations[db_name] = migrations
        return migrations

    def get_db_path(self, db_name: str) -> Path:
        """Get the full path to a database file."""
        db_file = self.db_paths.get(db_name, f"aragora_{db_name}.db")
        return self.nomic_dir / db_file

    def get_status(self, db_name: str) -> Optional[MigrationStatus]:
        """
        Get migration status for a database.

        Args:
            db_name: Name of the database module

        Returns:
            MigrationStatus or None if database doesn't exist
        """
        db_path = self.get_db_path(db_name)
        if not db_path.exists():
            return None

        migrations = self.discover_migrations(db_name)
        target_version = max((m.version for m in migrations), default=0)

        conn = get_wal_connection(db_path)
        try:
            manager = SchemaManager(conn, db_name, current_version=target_version)
            current_version = manager.get_version()

            applied = [m.version for m in migrations if m.version <= current_version]
            pending = [m.version for m in migrations if m.version > current_version]

            return MigrationStatus(
                db_name=db_name,
                db_path=str(db_path),
                current_version=current_version,
                target_version=target_version,
                pending_count=len(pending),
                applied_migrations=applied,
                pending_migrations=pending,
            )
        finally:
            conn.close()

    def get_all_status(self) -> dict[str, Optional[MigrationStatus]]:
        """Get migration status for all known databases."""
        status = {}
        for db_name in self.db_paths:
            status[db_name] = self.get_status(db_name)
        return status

    def migrate(
        self,
        db_name: str,
        dry_run: bool = False,
        target_version: Optional[int] = None,
    ) -> dict:
        """
        Run migrations for a database.

        Args:
            db_name: Name of the database module
            dry_run: If True, show what would be done without executing
            target_version: Migrate to specific version (None = latest)

        Returns:
            Dict with migration results
        """
        db_path = self.get_db_path(db_name)
        migrations = self.discover_migrations(db_name)

        if not migrations:
            return {
                "db_name": db_name,
                "status": "no_migrations",
                "message": f"No migrations found for {db_name}",
            }

        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        max_version = max(m.version for m in migrations)
        target = target_version if target_version is not None else max_version

        if dry_run:
            return self._dry_run_migrate(db_name, db_path, migrations, target)

        return self._execute_migrate(db_name, db_path, migrations, target)

    def _dry_run_migrate(
        self,
        db_name: str,
        db_path: Path,
        migrations: list[MigrationFile],
        target_version: int,
    ) -> dict:
        """Show what migrations would be run without executing."""
        if not db_path.exists():
            pending = [m for m in migrations if m.version <= target_version]
            return {
                "db_name": db_name,
                "status": "dry_run",
                "current_version": 0,
                "target_version": target_version,
                "would_create_db": True,
                "pending_migrations": [
                    {"version": m.version, "name": m.name, "description": m.description}
                    for m in pending
                ],
            }

        conn = get_wal_connection(db_path)
        try:
            manager = SchemaManager(conn, db_name, current_version=target_version)
            current = manager.get_version()

            pending = [m for m in migrations if m.version > current and m.version <= target_version]

            return {
                "db_name": db_name,
                "status": "dry_run",
                "current_version": current,
                "target_version": target_version,
                "would_create_db": False,
                "pending_migrations": [
                    {"version": m.version, "name": m.name, "description": m.description}
                    for m in pending
                ],
            }
        finally:
            conn.close()

    def _execute_migrate(
        self,
        db_name: str,
        db_path: Path,
        migrations: list[MigrationFile],
        target_version: int,
    ) -> dict:
        """Execute migrations."""
        conn = get_wal_connection(db_path)
        applied = []
        errors = []

        try:
            manager = SchemaManager(conn, db_name, current_version=target_version)
            current = manager.get_version()

            pending = [m for m in migrations if m.version > current and m.version <= target_version]

            if not pending:
                return {
                    "db_name": db_name,
                    "status": "up_to_date",
                    "current_version": current,
                    "message": f"{db_name} is already at version {current}",
                }

            for migration in pending:
                logger.info(
                    f"[{db_name}] Applying migration {migration.version}: "
                    f"{migration.description}"
                )

                try:
                    # Load and execute migration module
                    module = self._load_migration_module(migration)
                    if hasattr(module, "upgrade"):
                        # Check if migration is empty (only has pass)
                        if self._is_empty_migration(module.upgrade):
                            logger.warning(
                                f"[{db_name}] Migration {migration.version} has empty "
                                f"upgrade() function. Consider implementing it."
                            )
                        module.upgrade(conn)
                        conn.commit()
                    else:
                        raise ValueError(f"Migration {migration.path} missing upgrade() function")

                    # Update version
                    manager.set_version(migration.version)
                    applied.append(migration.version)

                    logger.info(f"[{db_name}] Applied migration {migration.version} successfully")

                except Exception as e:
                    conn.rollback()
                    errors.append(
                        {
                            "version": migration.version,
                            "error": str(e),
                        }
                    )
                    logger.error(f"[{db_name}] Migration {migration.version} failed: {e}")
                    break  # Stop on first error

            final_version = manager.get_version()

            return {
                "db_name": db_name,
                "status": "completed" if not errors else "partial",
                "initial_version": current,
                "final_version": final_version,
                "applied": applied,
                "errors": errors,
            }

        finally:
            conn.close()

    def _load_migration_module(self, migration: MigrationFile):
        """Load a migration module."""
        # Build module path relative to package
        db_name = migration.path.parent.name
        module_path = f"aragora.persistence.migrations.{db_name}.{migration.module_name}"

        # Import the module
        return importlib.import_module(module_path)

    def _is_empty_migration(self, func: Callable) -> bool:
        """Check if a migration function is effectively empty (only pass/pass-like)."""
        import inspect

        try:
            source = inspect.getsource(func)
            # Check if body is just 'pass' or comments + pass
            lines = [
                line.strip()
                for line in source.split("\n")
                if line.strip()
                and not line.strip().startswith("#")
                and not line.strip().startswith('"""')
                and not line.strip().startswith("'''")
                and not line.strip().startswith("def ")
            ]
            # Filter out docstrings
            non_doc_lines = [l for l in lines if not l.startswith('"') and not l.startswith("'")]
            return len(non_doc_lines) == 0 or all(l == "pass" for l in non_doc_lines)
        except Exception:
            # If we can't inspect, assume it's not empty
            return False

    def migrate_all(self, dry_run: bool = False) -> dict[str, dict]:
        """Run migrations for all databases."""
        results = {}
        for db_name in self.db_paths:
            results[db_name] = self.migrate(db_name, dry_run=dry_run)
        return results

    def create_migration(
        self,
        db_name: str,
        description: str,
    ) -> Path:
        """
        Create a new migration file.

        Args:
            db_name: Name of the database module
            description: Human-readable description

        Returns:
            Path to the created migration file
        """
        migrations_path = self.MIGRATIONS_DIR / db_name
        migrations_path.mkdir(parents=True, exist_ok=True)

        # Find next version number
        existing = self.discover_migrations(db_name)
        next_version = max((m.version for m in existing), default=0) + 1

        # Create filename
        slug = re.sub(r"[^a-z0-9]+", "_", description.lower()).strip("_")
        filename = f"{next_version:03d}_{slug}.py"
        file_path = migrations_path / filename

        # Create migration file
        template = f'''"""
Migration {next_version}: {description}

Created: {datetime.utcnow().isoformat()}

IMPORTANT: Replace the 'pass' statements below with actual SQL.
Empty migrations will generate warnings when applied.
"""

import sqlite3


def upgrade(conn: sqlite3.Connection) -> None:
    """
    Apply this migration.

    Common patterns:
    - Add column: ALTER TABLE tablename ADD COLUMN colname TYPE
    - Create table: CREATE TABLE IF NOT EXISTS ...
    - Create index: CREATE INDEX IF NOT EXISTS ...
    - Insert data: INSERT INTO tablename VALUES ...

    SQLite limitations (use workarounds):
    - Cannot DROP COLUMN directly (recreate table)
    - Cannot RENAME COLUMN in older SQLite (recreate table)
    - Cannot add constraints to existing table (recreate table)
    """
    # Example: Add a new column
    # conn.execute("""
    #     ALTER TABLE users ADD COLUMN locked_until TIMESTAMP
    # """)

    # Example: Create a new table
    # conn.execute("""
    #     CREATE TABLE IF NOT EXISTS audit_log (
    #         id INTEGER PRIMARY KEY AUTOINCREMENT,
    #         action TEXT NOT NULL,
    #         user_id TEXT,
    #         timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    #         details TEXT
    #     )
    # """)

    # Example: Create an index
    # conn.execute("""
    #     CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp
    #     ON audit_log(timestamp)
    # """)

    # TODO: Replace this pass with your migration SQL
    pass


def downgrade(conn: sqlite3.Connection) -> None:
    """
    Reverse this migration (optional, for development only).

    WARNING: Downgrade is not recommended in production.
    SQLite has limited ALTER TABLE support - you may need to
    recreate tables to reverse changes.
    """
    # Example: Drop a table (if you created one in upgrade)
    # conn.execute("DROP TABLE IF EXISTS audit_log")

    # TODO: Replace this pass if you need rollback support
    pass
'''

        file_path.write_text(template)
        logger.info(f"Created migration: {file_path}")

        # Clear cache
        self._discovered_migrations.pop(db_name, None)

        return file_path


def print_status(runner: MigrationRunner) -> None:
    """Print migration status for all databases."""
    status = runner.get_all_status()

    print("\n=== Migration Status ===\n")

    for db_name, db_status in status.items():
        if db_status is None:
            print(f"[{db_name}] Database does not exist")
            continue

        status_icon = "✓" if db_status.pending_count == 0 else "!"
        print(f"[{db_name}] {status_icon}")
        print(f"  Path: {db_status.db_path}")
        print(f"  Current version: {db_status.current_version}")
        print(f"  Target version: {db_status.target_version}")

        if db_status.pending_count > 0:
            print(f"  Pending migrations: {db_status.pending_migrations}")
        else:
            print("  Status: Up to date")
        print()


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Aragora Database Migration Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show migration status for all databases",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what migrations would be run without executing",
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Run pending migrations",
    )
    parser.add_argument(
        "--create",
        metavar="DESCRIPTION",
        help="Create a new migration with the given description",
    )
    parser.add_argument(
        "--db",
        metavar="NAME",
        help="Target specific database (default: all)",
    )
    parser.add_argument(
        "--nomic-dir",
        metavar="PATH",
        help="Override ARAGORA_DATA_DIR/ARAGORA_NOMIC_DIR (legacy: NOMIC_DIR)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Initialize runner
    nomic_dir = Path(args.nomic_dir) if args.nomic_dir else None
    runner = MigrationRunner(nomic_dir=nomic_dir)

    # Handle commands
    if args.status:
        print_status(runner)
        return 0

    if args.dry_run:
        if args.db:
            result = runner.migrate(args.db, dry_run=True)
            print(f"\n[{args.db}] Dry run:")
            if result.get("pending_migrations"):
                for m in result["pending_migrations"]:
                    print(f"  Would apply: {m['version']:03d}_{m['name']} - {m['description']}")
            else:
                print("  No pending migrations")
        else:
            results = runner.migrate_all(dry_run=True)
            for db_name, result in results.items():
                print(f"\n[{db_name}] Dry run:")
                if result.get("pending_migrations"):
                    for m in result["pending_migrations"]:
                        print(f"  Would apply: {m['version']:03d}_{m['name']} - {m['description']}")
                else:
                    print("  No pending migrations")
        return 0

    if args.migrate:
        if args.db:
            result = runner.migrate(args.db)
            print(f"\n[{args.db}] Migration result: {result['status']}")
            if result.get("applied"):
                print(f"  Applied: {result['applied']}")
            if result.get("errors"):
                for err in result["errors"]:
                    print(f"  Error in v{err['version']}: {err['error']}")
                return 1
        else:
            results = runner.migrate_all()
            has_errors = False
            for db_name, result in results.items():
                print(f"\n[{db_name}] {result['status']}")
                if result.get("applied"):
                    print(f"  Applied: {result['applied']}")
                if result.get("errors"):
                    has_errors = True
                    for err in result["errors"]:
                        print(f"  Error in v{err['version']}: {err['error']}")
            return 1 if has_errors else 0

    if args.create:
        if not args.db:
            print("Error: --db is required when creating a migration")
            return 1
        path = runner.create_migration(args.db, args.create)
        print(f"Created migration: {path}")
        return 0

    # No command specified, show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
