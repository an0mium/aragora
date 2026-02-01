"""
Aragora Database Migrations.

A lightweight migration system for managing database schema changes across
SQLite and PostgreSQL backends.

Features:
- Version tracking with automatic detection of applied/pending migrations
- Checksum verification to detect modified migration files
- Rollback SQL storage for disaster recovery
- Advisory locking for PostgreSQL concurrent safety
- Zero-downtime migration patterns

Usage:
    # Apply all pending migrations
    python -m aragora.migrations upgrade

    # Rollback last migration
    python -m aragora.migrations downgrade

    # Check migration status
    python -m aragora.migrations status

    # Create a new migration
    python -m aragora.migrations create "Add user preferences table"

Version Tracking:
    from aragora.migrations.tracker import MigrationTracker, AppliedMigration

    tracker = MigrationTracker(backend)
    if not tracker.is_applied("20240101000000"):
        tracker.mark_applied(
            version="20240101000000",
            name="Initial schema",
            checksum="abc123...",
            rollback_sql="DROP TABLE IF EXISTS users;"
        )

Zero-Downtime Patterns:
    from aragora.migrations.patterns import (
        safe_add_nullable_column,  # Expand phase
        backfill_column,           # Data migration
        safe_set_not_null,         # Contract phase
        validate_migration_safety, # Pre-migration validation
    )
"""

from .runner import (
    Migration,
    MigrationRunner,
    RollbackRecord,
    RollbackValidation,
    apply_migrations,
    compute_checksum,
    get_migration_runner,
    get_migration_status,
    rollback_migration,
)
from .tracker import (
    AppliedMigration,
    MigrationTracker,
    compute_migration_checksum,
)
from .patterns import (
    MigrationRisk,
    MigrationValidation,
    backfill_column,
    safe_add_column,
    safe_add_nullable_column,
    safe_create_index,
    safe_drop_column,
    safe_drop_index,
    safe_rename_column,
    safe_set_not_null,
    validate_migration_safety,
)

__all__ = [
    # Runner
    "MigrationRunner",
    "Migration",
    "RollbackValidation",
    "RollbackRecord",
    "get_migration_runner",
    "apply_migrations",
    "rollback_migration",
    "get_migration_status",
    "compute_checksum",
    # Tracker
    "MigrationTracker",
    "AppliedMigration",
    "compute_migration_checksum",
    # Zero-downtime patterns
    "MigrationRisk",
    "MigrationValidation",
    "safe_add_column",
    "safe_add_nullable_column",
    "safe_drop_column",
    "safe_rename_column",
    "backfill_column",
    "safe_set_not_null",
    "safe_create_index",
    "safe_drop_index",
    "validate_migration_safety",
]
