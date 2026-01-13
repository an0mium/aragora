#!/usr/bin/env python3
"""
Database backup script for Aragora.

Creates compressed backups of all SQLite databases with rotation.
Can be run manually or via cron/systemd timer.

Usage:
    python scripts/backup_databases.py [--backup-dir /path/to/backups] [--keep 7]
"""

import argparse
import gzip
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
import sys

# Directories containing databases
DB_DIRECTORIES = [
    ".nomic",
    "consolidated",
    ".",  # Root directory databases
]

# Database file patterns
DB_PATTERNS = ["*.db", "*.sqlite", "*.sqlite3"]

# Exclude patterns (test databases, temporary files)
EXCLUDE_PATTERNS = ["test_*.db", "*_test.db", "*.db-journal", "*.db-wal", "*.db-shm"]


def find_databases(root: Path) -> list[Path]:
    """Find all SQLite databases in the project."""
    databases = []

    for db_dir in DB_DIRECTORIES:
        search_path = root / db_dir
        if not search_path.exists():
            continue

        for pattern in DB_PATTERNS:
            for db_file in search_path.glob(pattern):
                # Skip excluded patterns
                skip = False
                for exclude in EXCLUDE_PATTERNS:
                    if db_file.match(exclude):
                        skip = True
                        break

                if not skip and db_file.is_file():
                    # Verify it's a valid SQLite database
                    try:
                        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
                        conn.execute("SELECT 1")
                        conn.close()
                        databases.append(db_file)
                    except sqlite3.Error:
                        pass  # Not a valid SQLite database

    return sorted(set(databases))


def backup_database(db_path: Path, backup_dir: Path, compress: bool = True) -> Path:
    """Create a backup of a single database.

    Uses SQLite's backup API for consistency during active writes.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_name = db_path.stem

    if compress:
        backup_name = f"{db_name}_{timestamp}.db.gz"
    else:
        backup_name = f"{db_name}_{timestamp}.db"

    backup_path = backup_dir / backup_name

    # Use SQLite backup API for consistency
    source_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    if compress:
        # Backup to temp file then compress
        temp_backup = backup_dir / f"{db_name}_{timestamp}.db"
        dest_conn = sqlite3.connect(temp_backup)
        source_conn.backup(dest_conn)
        dest_conn.close()

        # Compress the backup
        with open(temp_backup, "rb") as f_in:
            with gzip.open(backup_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        temp_backup.unlink()
    else:
        dest_conn = sqlite3.connect(backup_path)
        source_conn.backup(dest_conn)
        dest_conn.close()

    source_conn.close()
    return backup_path


def rotate_backups(backup_dir: Path, db_name: str, keep: int = 7):
    """Remove old backups, keeping only the most recent N."""
    pattern = f"{db_name}_*.db*"
    backups = sorted(backup_dir.glob(pattern), reverse=True)

    for old_backup in backups[keep:]:
        old_backup.unlink()


def main():
    parser = argparse.ArgumentParser(description="Backup Aragora databases")
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Directory to store backups (default: ./backups/db_YYYYMMDD)",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=7,
        help="Number of backups to keep per database (default: 7)",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Don't compress backups",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Project root directory",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )
    args = parser.parse_args()

    root = args.root.resolve()

    # Set default backup directory with date
    if args.backup_dir is None:
        date_str = datetime.now().strftime("%Y%m%d")
        backup_dir = root / "backups" / f"db_{date_str}"
    else:
        backup_dir = args.backup_dir

    backup_dir.mkdir(parents=True, exist_ok=True)

    # Find all databases
    databases = find_databases(root)

    if not databases:
        if not args.quiet:
            print("No databases found to backup")
        return 0

    if not args.quiet:
        print(f"Found {len(databases)} databases to backup")
        print(f"Backup directory: {backup_dir}")
        print()

    # Backup each database
    backed_up = []
    errors = []

    for db_path in databases:
        try:
            backup_path = backup_database(db_path, backup_dir, compress=not args.no_compress)
            backed_up.append((db_path.name, backup_path.name))

            # Rotate old backups
            rotate_backups(backup_dir, db_path.stem, args.keep)

            if not args.quiet:
                size_kb = backup_path.stat().st_size / 1024
                print(f"  ✓ {db_path.name} -> {backup_path.name} ({size_kb:.1f} KB)")

        except Exception as e:
            errors.append((db_path.name, str(e)))
            if not args.quiet:
                print(f"  ✗ {db_path.name}: {e}", file=sys.stderr)

    if not args.quiet:
        print()
        print(f"Backed up: {len(backed_up)} databases")
        if errors:
            print(f"Errors: {len(errors)} databases", file=sys.stderr)

    # Write manifest
    manifest_path = backup_dir / "manifest.json"
    import json

    manifest = {
        "created_at": datetime.now().isoformat(),
        "root": str(root),
        "databases": [{"source": src, "backup": bak} for src, bak in backed_up],
        "errors": [{"source": src, "error": err} for src, err in errors],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
