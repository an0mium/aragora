"""
Database migrations for Aragora.

Provides version-controlled schema management for production databases.

Usage:
    # Check migration status
    python -m aragora.persistence.migrations.runner --status

    # Dry-run migrations
    python -m aragora.persistence.migrations.runner --dry-run

    # Run migrations
    python -m aragora.persistence.migrations.runner --migrate

    # Create new migration
    python -m aragora.persistence.migrations.runner --create "Add new field" --db users

See aragora/persistence/migrations/runner.py for full documentation.
"""

from aragora.persistence.migrations.runner import (
    MigrationRunner,
    MigrationFile,
    MigrationStatus,
)

__all__ = [
    "MigrationRunner",
    "MigrationFile",
    "MigrationStatus",
]
