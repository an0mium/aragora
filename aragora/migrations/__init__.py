"""
Aragora Database Migrations.

A lightweight migration system for managing database schema changes across
SQLite and PostgreSQL backends.

Usage:
    # Apply all pending migrations
    python -m aragora.migrations upgrade

    # Rollback last migration
    python -m aragora.migrations downgrade

    # Check migration status
    python -m aragora.migrations status

    # Create a new migration
    python -m aragora.migrations create "Add user preferences table"
"""

from .runner import (
    Migration,
    MigrationRunner,
    apply_migrations,
    get_migration_runner,
    get_migration_status,
    rollback_migration,
)

__all__ = [
    "MigrationRunner",
    "Migration",
    "get_migration_runner",
    "apply_migrations",
    "rollback_migration",
    "get_migration_status",
]
