#!/usr/bin/env python3
"""
Aragora Backup CLI - Simple backup and restore operations.

Usage:
    python scripts/backup_cli.py backup <database_path>
    python scripts/backup_cli.py restore <backup_id> <target_path>
    python scripts/backup_cli.py list [--status verified]
    python scripts/backup_cli.py verify <backup_id>
    python scripts/backup_cli.py cleanup [--dry-run]

Examples:
    # Create a backup
    python scripts/backup_cli.py backup ~/.aragora/aragora.db

    # List all verified backups
    python scripts/backup_cli.py list --status verified

    # Restore a backup (dry-run first)
    python scripts/backup_cli.py restore abc123 ./restored.db --dry-run
    python scripts/backup_cli.py restore abc123 ./restored.db

    # Verify backup integrity
    python scripts/backup_cli.py verify abc123

    # Clean up old backups according to retention policy
    python scripts/backup_cli.py cleanup --dry-run
    python scripts/backup_cli.py cleanup
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from aragora.backup.manager import (
    BackupManager,
    BackupStatus,
    BackupType,
    RetentionPolicy,
)


def get_backup_manager(backup_dir: str | None = None) -> BackupManager:
    """Get or create backup manager with default settings."""
    if backup_dir is None:
        backup_dir = Path.home() / ".aragora" / "backups"
    return BackupManager(
        backup_dir=backup_dir,
        retention_policy=RetentionPolicy(
            keep_daily=7,
            keep_weekly=4,
            keep_monthly=3,
            min_backups=1,
        ),
        compression=True,
        verify_after_backup=True,
    )


def cmd_backup(args: argparse.Namespace) -> int:
    """Create a new backup."""
    manager = get_backup_manager(args.backup_dir)
    source = Path(args.database_path)

    if not source.exists():
        print(f"Error: Database not found: {source}")
        return 1

    print(f"Creating backup of {source}...")
    try:
        backup = manager.create_backup(
            source_path=source,
            backup_type=BackupType.FULL,
            metadata={"cli_version": "1.0", "source": str(source)},
        )
        print("Backup created successfully!")
        print(f"  ID: {backup.id}")
        print(f"  Path: {backup.backup_path}")
        print(f"  Size: {backup.compressed_size_bytes / 1024 / 1024:.2f} MB")
        print(f"  Status: {backup.status.value}")
        print(f"  Verified: {backup.verified}")
        return 0
    except Exception as e:
        print(f"Error creating backup: {e}")
        return 1


def cmd_restore(args: argparse.Namespace) -> int:
    """Restore a backup."""
    manager = get_backup_manager(args.backup_dir)

    if args.dry_run:
        print(f"Dry run: Would restore backup {args.backup_id} to {args.target_path}")

    try:
        success = manager.restore_backup(
            backup_id=args.backup_id,
            target_path=args.target_path,
            dry_run=args.dry_run,
        )
        if success:
            if args.dry_run:
                print("Dry run successful - backup is valid and can be restored")
            else:
                print(f"Restored backup {args.backup_id} to {args.target_path}")
            return 0
        else:
            print("Restore failed")
            return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    """List backups."""
    manager = get_backup_manager(args.backup_dir)

    status_filter = None
    if args.status:
        try:
            status_filter = BackupStatus(args.status)
        except ValueError:
            print(f"Invalid status: {args.status}")
            print(f"Valid values: {', '.join(s.value for s in BackupStatus)}")
            return 1

    backups = manager.list_backups(status=status_filter)

    if not backups:
        print("No backups found")
        return 0

    print(f"Found {len(backups)} backup(s):\n")
    print(f"{'ID':<10} {'Status':<12} {'Size (MB)':<10} {'Verified':<10} {'Created':<20}")
    print("-" * 70)

    for b in backups:
        size_mb = b.compressed_size_bytes / 1024 / 1024
        created = b.created_at.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{b.id:<10} {b.status.value:<12} {size_mb:<10.2f} {str(b.verified):<10} {created:<20}"
        )

    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify a backup."""
    manager = get_backup_manager(args.backup_dir)

    print(f"Verifying backup {args.backup_id}...")
    try:
        if args.comprehensive:
            result = manager.verify_restore_comprehensive(args.backup_id)
            print(f"Verification result: {'PASSED' if result.verified else 'FAILED'}")
            print(f"  Basic verification: {result.basic_verification.verified}")
            if result.schema_validation:
                print(f"  Schema validation: {result.schema_validation.valid}")
            if result.integrity_check:
                print(f"  Integrity check: {result.integrity_check.valid}")
            print(f"  Table checksums: {result.table_checksums_valid}")
            if result.all_errors:
                print(f"  Errors: {result.all_errors}")
            if result.all_warnings:
                print(f"  Warnings: {result.all_warnings}")
        else:
            result = manager.verify_backup(args.backup_id)
            print(f"Verification result: {'PASSED' if result.verified else 'FAILED'}")
            print(f"  Checksum valid: {result.checksum_valid}")
            print(f"  Restore tested: {result.restore_tested}")
            print(f"  Tables valid: {result.tables_valid}")
            print(f"  Row counts valid: {result.row_counts_valid}")
            if result.errors:
                print(f"  Errors: {result.errors}")

        return 0 if result.verified else 1
    except Exception as e:
        print(f"Error verifying backup: {e}")
        return 1


def cmd_cleanup(args: argparse.Namespace) -> int:
    """Clean up old backups according to retention policy."""
    manager = get_backup_manager(args.backup_dir)

    if args.dry_run:
        print("Dry run - showing backups that would be deleted:\n")
        to_delete = manager.apply_retention_policy(dry_run=True)
        if not to_delete:
            print("No backups to delete")
        else:
            for backup_id in to_delete:
                print(f"  Would delete: {backup_id}")
        return 0
    else:
        print("Cleaning up old backups...")
        deleted = manager.cleanup_expired()
        if not deleted:
            print("No backups were deleted")
        else:
            print(f"Deleted {len(deleted)} backup(s):")
            for backup_id in deleted:
                print(f"  {backup_id}")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aragora Backup CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        help="Backup directory (default: ~/.aragora/backups)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # backup command
    backup_parser = subparsers.add_parser("backup", help="Create a new backup")
    backup_parser.add_argument("database_path", help="Path to database to backup")

    # restore command
    restore_parser = subparsers.add_parser("restore", help="Restore a backup")
    restore_parser.add_argument("backup_id", help="Backup ID to restore")
    restore_parser.add_argument("target_path", help="Target path for restore")
    restore_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify restore without applying",
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List backups")
    list_parser.add_argument(
        "--status",
        choices=[s.value for s in BackupStatus],
        help="Filter by status",
    )

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a backup")
    verify_parser.add_argument("backup_id", help="Backup ID to verify")
    verify_parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive verification (slower)",
    )

    # cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old backups")
    cleanup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting",
    )

    args = parser.parse_args()

    if args.command == "backup":
        return cmd_backup(args)
    elif args.command == "restore":
        return cmd_restore(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "verify":
        return cmd_verify(args)
    elif args.command == "cleanup":
        return cmd_cleanup(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
