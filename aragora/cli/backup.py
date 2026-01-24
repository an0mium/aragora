"""
Aragora Backup CLI - Database backup and restore commands.

Provides commands for:
- Creating backups (full/incremental)
- Listing backups
- Restoring from backup
- Verifying backup integrity
- Managing retention policies
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_BACKUP_DIR = ".aragora/backups"
DEFAULT_DB_PATH = "aragora.db"


def _format_size(size_bytes: int | float) -> str:
    """Format bytes as human-readable size."""
    size: float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def cmd_backup_create(args: argparse.Namespace) -> int:
    """Create a new backup."""
    from aragora.backup.manager import BackupManager, BackupType, RetentionPolicy

    backup_dir = Path(args.output or DEFAULT_BACKUP_DIR)
    db_path = Path(args.database or DEFAULT_DB_PATH)

    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        print("Specify database with --database or run from project directory")
        return 1

    # Configure retention policy
    retention = RetentionPolicy(
        keep_daily=args.keep_daily,
        keep_weekly=args.keep_weekly,
        keep_monthly=args.keep_monthly,
    )

    manager = BackupManager(
        backup_dir=backup_dir,
        retention_policy=retention,
        compression=not args.no_compress,
        verify_after_backup=not args.skip_verify,
    )

    backup_type = BackupType.INCREMENTAL if args.incremental else BackupType.FULL

    print(f"Creating {'incremental' if args.incremental else 'full'} backup...")
    print(f"  Source: {db_path}")
    print(f"  Destination: {backup_dir}")

    if args.dry_run:
        print("\n[DRY RUN] Would create backup, but --dry-run specified")
        return 0

    try:
        result = manager.create_backup(
            source_path=db_path,
            backup_type=backup_type,
            metadata={"created_by": "cli", "notes": args.notes} if args.notes else None,
        )

        print("\nBackup created successfully!")
        print(f"  ID: {result.id}")
        print(f"  Path: {result.backup_path}")
        print(f"  Size: {_format_size(result.compressed_size_bytes)}")
        print(f"  Duration: {_format_duration(result.duration_seconds)}")
        print(f"  Status: {result.status.value}")
        if result.verified:
            print("  Verified: Yes")
        print(f"  Tables: {len(result.tables)}")

        return 0

    except Exception as e:
        print(f"\nError: Backup failed - {e}")
        return 1


def cmd_backup_list(args: argparse.Namespace) -> int:
    """List available backups."""
    from aragora.backup.manager import BackupManager

    backup_dir = Path(args.backup_dir or DEFAULT_BACKUP_DIR)

    if not backup_dir.exists():
        print(f"No backups found (directory does not exist: {backup_dir})")
        return 0

    manager = BackupManager(backup_dir=backup_dir)
    backups = list(manager._backups.values())

    if not backups:
        print("No backups found.")
        return 0

    # Sort by creation time (newest first)
    backups.sort(key=lambda b: b.created_at, reverse=True)

    if args.json:
        output = [b.to_dict() for b in backups]
        print(json.dumps(output, indent=2, default=str))
        return 0

    # Table output
    print(f"\nBackups in {backup_dir}:")
    print("-" * 80)
    print(f"{'ID':<10} {'Created':<20} {'Type':<12} {'Size':<10} {'Status':<10}")
    print("-" * 80)

    for backup in backups[: args.limit]:
        created = backup.created_at.strftime("%Y-%m-%d %H:%M:%S")
        size = _format_size(backup.compressed_size_bytes)
        print(
            f"{backup.id:<10} {created:<20} {backup.backup_type.value:<12} {size:<10} {backup.status.value:<10}"
        )

    if len(backups) > args.limit:
        print(f"\n... and {len(backups) - args.limit} more backups")

    print(f"\nTotal: {len(backups)} backups")
    total_size = sum(b.compressed_size_bytes for b in backups)
    print(f"Total size: {_format_size(total_size)}")

    return 0


def cmd_backup_restore(args: argparse.Namespace) -> int:
    """Restore from a backup."""
    from aragora.backup.manager import BackupManager

    backup_dir = Path(args.backup_dir or DEFAULT_BACKUP_DIR)

    if not backup_dir.exists():
        print(f"Error: Backup directory not found: {backup_dir}")
        return 1

    manager = BackupManager(backup_dir=backup_dir)

    # Find backup
    backup = manager._backups.get(args.backup_id)
    if not backup:
        # Try to find by partial ID
        matches = [b for bid, b in manager._backups.items() if bid.startswith(args.backup_id)]
        if len(matches) == 1:
            backup = matches[0]
        elif len(matches) > 1:
            print(f"Error: Multiple backups match '{args.backup_id}':")
            for m in matches:
                print(f"  {m.id} - {m.created_at}")
            return 1
        else:
            print(f"Error: Backup not found: {args.backup_id}")
            return 1

    target_path = Path(args.output or backup.source_path)

    print("Restoring backup...")
    print(f"  Backup ID: {backup.id}")
    print(f"  Backup path: {backup.backup_path}")
    print(f"  Target: {target_path}")
    print(f"  Created: {backup.created_at}")

    if args.dry_run:
        print("\n[DRY RUN] Would restore backup, but --dry-run specified")
        return 0

    # Check if target exists
    if target_path.exists() and not args.force:
        print(f"\nError: Target already exists: {target_path}")
        print("Use --force to overwrite")
        return 1

    try:
        manager.restore_backup(backup.id, target_path, verify_after=not args.skip_verify)
        print("\nRestore completed successfully!")
        print(f"  Restored to: {target_path}")
        return 0

    except Exception as e:
        print(f"\nError: Restore failed - {e}")
        return 1


def cmd_backup_verify(args: argparse.Namespace) -> int:
    """Verify backup integrity."""
    from aragora.backup.manager import BackupManager

    backup_dir = Path(args.backup_dir or DEFAULT_BACKUP_DIR)

    if not backup_dir.exists():
        print(f"Error: Backup directory not found: {backup_dir}")
        return 1

    manager = BackupManager(backup_dir=backup_dir)

    # Find backup
    backup = manager._backups.get(args.backup_id)
    if not backup:
        matches = [b for bid, b in manager._backups.items() if bid.startswith(args.backup_id)]
        if len(matches) == 1:
            backup = matches[0]
        elif len(matches) > 1:
            print(f"Error: Multiple backups match '{args.backup_id}'")
            return 1
        else:
            print(f"Error: Backup not found: {args.backup_id}")
            return 1

    print(f"Verifying backup {backup.id}...")

    result = manager.verify_backup(backup.id, test_restore=not args.skip_restore_test)

    if args.json:
        print(
            json.dumps(
                {
                    "backup_id": result.backup_id,
                    "verified": result.verified,
                    "checksum_valid": result.checksum_valid,
                    "restore_tested": result.restore_tested,
                    "tables_valid": result.tables_valid,
                    "row_counts_valid": result.row_counts_valid,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "duration_seconds": result.duration_seconds,
                },
                indent=2,
            )
        )
        return 0 if result.verified else 1

    print("\nVerification Results:")
    print(f"  Checksum: {'Valid' if result.checksum_valid else 'INVALID'}")
    print(f"  Tables: {'Valid' if result.tables_valid else 'INVALID'}")
    print(f"  Row counts: {'Valid' if result.row_counts_valid else 'INVALID'}")
    print(f"  Restore test: {'Passed' if result.restore_tested else 'Skipped'}")
    print(f"  Duration: {_format_duration(result.duration_seconds)}")

    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  - {error}")

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    if result.verified:
        print("\nBackup verified successfully!")
        return 0
    else:
        print("\nBackup verification FAILED!")
        return 1


def cmd_backup_cleanup(args: argparse.Namespace) -> int:
    """Apply retention policy to clean up old backups."""
    from aragora.backup.manager import BackupManager, RetentionPolicy

    backup_dir = Path(args.backup_dir or DEFAULT_BACKUP_DIR)

    if not backup_dir.exists():
        print(f"No backups found (directory does not exist: {backup_dir})")
        return 0

    retention = RetentionPolicy(
        keep_daily=args.keep_daily,
        keep_weekly=args.keep_weekly,
        keep_monthly=args.keep_monthly,
    )

    manager = BackupManager(backup_dir=backup_dir, retention_policy=retention)

    # Get backups that would be removed
    to_remove = manager.apply_retention_policy(dry_run=True)

    if not to_remove:
        print("No backups need to be removed based on retention policy.")
        return 0

    print(f"\nBackups to remove ({len(to_remove)}):")
    total_size = 0
    for backup_id in to_remove:
        backup = manager._backups.get(backup_id)
        if backup:
            total_size += backup.compressed_size_bytes
            print(
                f"  {backup.id} - {backup.created_at.strftime('%Y-%m-%d %H:%M')} ({_format_size(backup.compressed_size_bytes)})"
            )

    print(f"\nTotal space to reclaim: {_format_size(total_size)}")

    if args.dry_run:
        print("\n[DRY RUN] Would remove backups, but --dry-run specified")
        return 0

    if not args.force:
        try:
            response = input("\nProceed with cleanup? [y/N]: ").strip().lower()
            if response not in ("y", "yes"):
                print("Cleanup cancelled.")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\nCleanup cancelled.")
            return 0

    removed = manager.apply_retention_policy(dry_run=False)
    print(f"\nRemoved {len(removed)} backups, reclaimed {_format_size(total_size)}")

    return 0


def add_backup_subparsers(subparsers: Any) -> None:
    """Add backup-related subparsers."""
    # Main backup command with subcommands
    backup_parser = subparsers.add_parser(
        "backup",
        help="Database backup and restore commands",
        description="Manage database backups including creation, restoration, and verification.",
    )
    backup_subparsers = backup_parser.add_subparsers(dest="backup_command", help="Backup commands")

    # backup create
    create_parser = backup_subparsers.add_parser("create", help="Create a new backup")
    create_parser.add_argument(
        "--database", "-d", help=f"Database path (default: {DEFAULT_DB_PATH})"
    )
    create_parser.add_argument(
        "--output", "-o", help=f"Backup directory (default: {DEFAULT_BACKUP_DIR})"
    )
    create_parser.add_argument(
        "--incremental", "-i", action="store_true", help="Create incremental backup"
    )
    create_parser.add_argument("--no-compress", action="store_true", help="Disable compression")
    create_parser.add_argument(
        "--skip-verify", action="store_true", help="Skip verification after backup"
    )
    create_parser.add_argument("--notes", help="Add notes to backup metadata")
    create_parser.add_argument(
        "--keep-daily", type=int, default=7, help="Keep N daily backups (default: 7)"
    )
    create_parser.add_argument(
        "--keep-weekly", type=int, default=4, help="Keep N weekly backups (default: 4)"
    )
    create_parser.add_argument(
        "--keep-monthly", type=int, default=3, help="Keep N monthly backups (default: 3)"
    )
    create_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without doing it"
    )
    create_parser.set_defaults(func=cmd_backup_create)

    # backup list
    list_parser = backup_subparsers.add_parser("list", help="List available backups")
    list_parser.add_argument(
        "--backup-dir", "-d", help=f"Backup directory (default: {DEFAULT_BACKUP_DIR})"
    )
    list_parser.add_argument("--limit", "-l", type=int, default=20, help="Maximum backups to show")
    list_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    list_parser.set_defaults(func=cmd_backup_list)

    # backup restore
    restore_parser = backup_subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument("backup_id", help="Backup ID to restore")
    restore_parser.add_argument(
        "--backup-dir", "-d", help=f"Backup directory (default: {DEFAULT_BACKUP_DIR})"
    )
    restore_parser.add_argument("--output", "-o", help="Target path (default: original location)")
    restore_parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite existing target"
    )
    restore_parser.add_argument(
        "--skip-verify", action="store_true", help="Skip verification after restore"
    )
    restore_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without doing it"
    )
    restore_parser.set_defaults(func=cmd_backup_restore)

    # backup verify
    verify_parser = backup_subparsers.add_parser("verify", help="Verify backup integrity")
    verify_parser.add_argument("backup_id", help="Backup ID to verify")
    verify_parser.add_argument(
        "--backup-dir", "-d", help=f"Backup directory (default: {DEFAULT_BACKUP_DIR})"
    )
    verify_parser.add_argument("--skip-restore-test", action="store_true", help="Skip restore test")
    verify_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    verify_parser.set_defaults(func=cmd_backup_verify)

    # backup cleanup
    cleanup_parser = backup_subparsers.add_parser(
        "cleanup", help="Remove old backups per retention policy"
    )
    cleanup_parser.add_argument(
        "--backup-dir", "-d", help=f"Backup directory (default: {DEFAULT_BACKUP_DIR})"
    )
    cleanup_parser.add_argument(
        "--keep-daily", type=int, default=7, help="Keep N daily backups (default: 7)"
    )
    cleanup_parser.add_argument(
        "--keep-weekly", type=int, default=4, help="Keep N weekly backups (default: 4)"
    )
    cleanup_parser.add_argument(
        "--keep-monthly", type=int, default=3, help="Keep N monthly backups (default: 3)"
    )
    cleanup_parser.add_argument(
        "--force", "-f", action="store_true", help="Don't prompt for confirmation"
    )
    cleanup_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without doing it"
    )
    cleanup_parser.set_defaults(func=cmd_backup_cleanup)


def cmd_backup(args: argparse.Namespace) -> int:
    """Handle backup command (dispatcher)."""
    if not hasattr(args, "backup_command") or args.backup_command is None:
        print("Usage: aragora backup <command>")
        print("\nCommands:")
        print("  create   Create a new backup")
        print("  list     List available backups")
        print("  restore  Restore from backup")
        print("  verify   Verify backup integrity")
        print("  cleanup  Remove old backups")
        print("\nRun 'aragora backup <command> --help' for more information")
        return 0

    if hasattr(args, "func"):
        return args.func(args)

    return 0
