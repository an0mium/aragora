"""
PostgreSQL-specific migrations for Aragora.

These migrations create the same schema as SQLite migrations but with
PostgreSQL-compatible syntax and optimizations.

Usage:
    # Run schema migrations
    from aragora.persistence.migrations.postgres import PostgresMigrationRunner

    runner = PostgresMigrationRunner()
    await runner.migrate()

    # Migrate data from SQLite
    from aragora.persistence.migrations.postgres import DataMigrator

    migrator = DataMigrator(
        sqlite_path="path/to/db.sqlite",
        postgres_dsn="postgresql://user:pass@host/db"
    )
    await migrator.migrate_all()
"""

from aragora.persistence.migrations.postgres.runner import (
    PostgresMigrationRunner,
    get_postgres_migration_runner,
)
from aragora.persistence.migrations.postgres.data_migrator import (
    DataMigrator,
    MigrationStats,
)

__all__ = [
    "PostgresMigrationRunner",
    "get_postgres_migration_runner",
    "DataMigrator",
    "MigrationStats",
]
