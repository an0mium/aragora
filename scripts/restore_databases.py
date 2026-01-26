#!/usr/bin/env python3
"""
Database restore script for Aragora.

Restores SQLite and PostgreSQL databases from backups created by backup_databases.py.
Supports verification after restore and dry-run mode.

Usage:
    python scripts/restore_databases.py --input /path/to/backup_dir [--verify] [--dry-run]
    python scripts/restore_databases.py --input /path/to/backup.db.gz --target ./restored.db

PostgreSQL usage:
    python scripts/restore_databases.py --input /path/to/backup_dir --postgres --pg-url postgresql://...
"""

import argparse
import gzip
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def decompress_file(compressed_path: Path, output_path: Path) -> None:
    """Decompress a gzipped file."""
    with gzip.open(compressed_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def verify_sqlite_database(db_path: Path) -> Tuple[bool, str]:
    """Verify a SQLite database is valid and not corrupted.

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

        # Run integrity check
        cursor = conn.execute("PRAGMA integrity_check")
        result = cursor.fetchone()

        if result[0] != "ok":
            conn.close()
            return False, f"Integrity check failed: {result[0]}"

        # Get table count
        cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]

        conn.close()
        return True, f"Valid database with {table_count} tables"

    except sqlite3.Error as e:
        return False, f"SQLite error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def restore_sqlite_database(
    backup_path: Path,
    target_path: Path,
    verify: bool = True,
    dry_run: bool = False,
) -> Tuple[bool, str]:
    """Restore a SQLite database from backup.

    Args:
        backup_path: Path to backup file (.db or .db.gz)
        target_path: Where to restore the database
        verify: Run integrity check after restore
        dry_run: Only simulate, don't actually restore

    Returns:
        Tuple of (success, message)
    """
    if dry_run:
        return True, f"[DRY RUN] Would restore {backup_path} to {target_path}"

    try:
        # Handle compressed backups
        if backup_path.suffix == ".gz":
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            decompress_file(backup_path, tmp_path)
            source_path = tmp_path
        else:
            source_path = backup_path
            tmp_path = None

        # Backup existing database if it exists
        if target_path.exists():
            backup_existing = target_path.with_suffix(
                f".db.pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copy2(target_path, backup_existing)

        # Use SQLite backup API for atomic restore
        source_conn = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
        target_conn = sqlite3.connect(target_path)

        source_conn.backup(target_conn)

        source_conn.close()
        target_conn.close()

        # Cleanup temp file
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()

        # Verify if requested
        if verify:
            is_valid, msg = verify_sqlite_database(target_path)
            if not is_valid:
                return False, f"Verification failed: {msg}"
            return True, f"Restored and verified: {msg}"

        return True, f"Restored to {target_path}"

    except Exception as e:
        return False, f"Restore failed: {e}"


def restore_postgres_database(
    backup_path: Path,
    pg_url: str,
    database_name: Optional[str] = None,
    verify: bool = True,
    dry_run: bool = False,
) -> Tuple[bool, str]:
    """Restore a PostgreSQL database from pg_dump backup.

    Args:
        backup_path: Path to backup file (.sql, .sql.gz, or .dump)
        pg_url: PostgreSQL connection URL
        database_name: Target database name (extracted from URL if not provided)
        verify: Run basic verification after restore
        dry_run: Only simulate, don't actually restore

    Returns:
        Tuple of (success, message)
    """
    if dry_run:
        return True, f"[DRY RUN] Would restore {backup_path} to PostgreSQL"

    try:
        # Parse database name from URL if not provided
        if database_name is None:
            # postgresql://user:pass@host:port/dbname
            database_name = pg_url.rsplit("/", 1)[-1].split("?")[0]

        # Handle compressed backups
        if backup_path.suffix == ".gz":
            with tempfile.NamedTemporaryFile(suffix=".sql", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            decompress_file(backup_path, tmp_path)
            restore_file = tmp_path
        else:
            restore_file = backup_path
            tmp_path = None

        # Determine restore command based on file type
        if backup_path.name.endswith(".dump") or backup_path.name.endswith(".dump.gz"):
            # Custom format - use pg_restore
            cmd = [
                "pg_restore",
                "--clean",
                "--if-exists",
                "--no-owner",
                "--no-acl",
                "-d",
                pg_url,
                str(restore_file),
            ]
        else:
            # SQL format - use psql
            cmd = [
                "psql",
                pg_url,
                "-f",
                str(restore_file),
            ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        # Cleanup temp file
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()

        if result.returncode != 0:
            return False, f"Restore failed: {result.stderr}"

        # Verify if requested
        if verify:
            verify_cmd = ["psql", pg_url, "-c", "SELECT 1"]
            verify_result = subprocess.run(
                verify_cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if verify_result.returncode != 0:
                return False, f"Verification failed: {verify_result.stderr}"

        return True, f"Restored to PostgreSQL database: {database_name}"

    except subprocess.TimeoutExpired:
        return False, "Restore timed out after 5 minutes"
    except FileNotFoundError as e:
        return False, f"PostgreSQL tools not found: {e}"
    except Exception as e:
        return False, f"Restore failed: {e}"


def restore_from_manifest(
    backup_dir: Path,
    target_dir: Optional[Path] = None,
    verify: bool = True,
    dry_run: bool = False,
    quiet: bool = False,
) -> Tuple[int, int]:
    """Restore all databases from a backup directory using its manifest.

    Args:
        backup_dir: Directory containing backup files and manifest.json
        target_dir: Where to restore databases (uses original paths if not specified)
        verify: Verify databases after restore
        dry_run: Only simulate restore
        quiet: Suppress output

    Returns:
        Tuple of (successful_count, error_count)
    """
    manifest_path = backup_dir / "manifest.json"

    if not manifest_path.exists():
        if not quiet:
            print(f"No manifest found at {manifest_path}", file=sys.stderr)
        return 0, 1

    with open(manifest_path) as f:
        manifest = json.load(f)

    if not quiet:
        print(f"Restoring from backup created at: {manifest['created_at']}")
        print(f"Original root: {manifest['root']}")
        print()

    successful = 0
    errors = 0

    for entry in manifest.get("databases", []):
        source_name = entry["source"]
        backup_name = entry["backup"]
        backup_path = backup_dir / backup_name

        if not backup_path.exists():
            if not quiet:
                print(f"  ✗ {source_name}: Backup file not found", file=sys.stderr)
            errors += 1
            continue

        # Determine target path
        if target_dir:
            target_path = target_dir / source_name
        else:
            # Use original path from manifest
            original_root = Path(manifest["root"])
            target_path = original_root / source_name

        # Ensure parent directory exists
        if not dry_run:
            target_path.parent.mkdir(parents=True, exist_ok=True)

        success, msg = restore_sqlite_database(
            backup_path, target_path, verify=verify, dry_run=dry_run
        )

        if success:
            successful += 1
            if not quiet:
                print(f"  ✓ {source_name}: {msg}")
        else:
            errors += 1
            if not quiet:
                print(f"  ✗ {source_name}: {msg}", file=sys.stderr)

    return successful, errors


def main():
    parser = argparse.ArgumentParser(
        description="Restore Aragora databases from backup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Restore all databases from a backup directory
  python scripts/restore_databases.py --input ./backups/db_20240101

  # Restore to a different location
  python scripts/restore_databases.py --input ./backups/db_20240101 --target ./restored

  # Restore a single database file
  python scripts/restore_databases.py --input ./backups/mydb_20240101.db.gz --target ./mydb.db

  # Dry run to see what would be restored
  python scripts/restore_databases.py --input ./backups/db_20240101 --dry-run

  # Restore PostgreSQL database
  python scripts/restore_databases.py --input ./backup.sql.gz --postgres --pg-url postgresql://localhost/mydb
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Backup directory (with manifest.json) or single backup file",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Target directory or file path for restore",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify database integrity after restore (default: True)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification after restore",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be restored without actually restoring",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )

    # PostgreSQL options
    parser.add_argument(
        "--postgres",
        action="store_true",
        help="Restore PostgreSQL database instead of SQLite",
    )
    parser.add_argument(
        "--pg-url",
        type=str,
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL connection URL (default: $DATABASE_URL)",
    )
    parser.add_argument(
        "--pg-database",
        type=str,
        default=None,
        help="Target PostgreSQL database name",
    )

    args = parser.parse_args()

    verify = args.verify and not args.no_verify
    input_path = args.input.resolve()

    if not input_path.exists():
        print(f"Input path does not exist: {input_path}", file=sys.stderr)
        return 1

    # PostgreSQL restore
    if args.postgres:
        if not args.pg_url:
            print("PostgreSQL URL required (--pg-url or $DATABASE_URL)", file=sys.stderr)
            return 1

        success, msg = restore_postgres_database(
            backup_path=input_path,
            pg_url=args.pg_url,
            database_name=args.pg_database,
            verify=verify,
            dry_run=args.dry_run,
        )

        if not args.quiet:
            print(msg)

        return 0 if success else 1

    # SQLite restore - single file
    if input_path.is_file():
        if not args.target:
            print("Target path required when restoring a single file", file=sys.stderr)
            return 1

        target_path = args.target.resolve()

        success, msg = restore_sqlite_database(
            backup_path=input_path,
            target_path=target_path,
            verify=verify,
            dry_run=args.dry_run,
        )

        if not args.quiet:
            print(msg)

        return 0 if success else 1

    # SQLite restore - directory with manifest
    if input_path.is_dir():
        target_dir = args.target.resolve() if args.target else None

        successful, errors = restore_from_manifest(
            backup_dir=input_path,
            target_dir=target_dir,
            verify=verify,
            dry_run=args.dry_run,
            quiet=args.quiet,
        )

        if not args.quiet:
            print()
            print(f"Restored: {successful} databases")
            if errors:
                print(f"Errors: {errors} databases", file=sys.stderr)

        return 1 if errors else 0

    print(f"Input must be a file or directory: {input_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
