"""
Aragora Backup Module.

Provides automated backup with verification:
- SQLite database backups with compression
- Integrity verification (checksums, restore tests)
- Retention policy enforcement
- Prometheus metrics for monitoring

Quick Start:
    from aragora.backup import BackupManager, get_backup_manager

    # Get the global backup manager
    manager = get_backup_manager()

    # Create a backup
    backup = manager.create_backup("/path/to/database.db")

    # Verify the backup
    result = manager.verify_backup(backup.id)

    # Restore (dry run first)
    manager.restore_backup(backup.id, "/path/to/restore.db", dry_run=True)
"""

from .manager import (
    BackupManager,
    BackupMetadata,
    BackupStatus,
    BackupType,
    RetentionPolicy,
    VerificationResult,
    get_backup_manager,
    set_backup_manager,
)

__all__ = [
    "BackupManager",
    "BackupMetadata",
    "BackupStatus",
    "BackupType",
    "RetentionPolicy",
    "VerificationResult",
    "get_backup_manager",
    "set_backup_manager",
]
