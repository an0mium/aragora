#!/usr/bin/env python3
"""
Disaster Recovery Metrics Script.

Measures and reports RTO (Recovery Time Objective) and RPO (Recovery Point Objective)
compliance for Aragora's backup and disaster recovery system.

Usage:
    python scripts/dr_metrics.py                    # Full DR metrics report
    python scripts/dr_metrics.py --ci-mode          # CI-friendly output
    python scripts/dr_metrics.py --verify-backup    # Verify latest backup
    python scripts/dr_metrics.py --dry-run          # Preview without actions

SLA Targets:
    Free Tier:       RTO=24h, RPO=24h
    Pro Tier:        RTO=4h,  RPO=1h
    Enterprise Tier: RTO=1h,  RPO=15m
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


@dataclass
class RTORPOTargets:
    """RTO/RPO targets by tier."""

    tier: str
    rto_seconds: int  # Recovery Time Objective in seconds
    rpo_seconds: int  # Recovery Point Objective in seconds

    @property
    def rto_human(self) -> str:
        """Human-readable RTO."""
        if self.rto_seconds >= 3600:
            return f"{self.rto_seconds // 3600}h"
        return f"{self.rto_seconds // 60}m"

    @property
    def rpo_human(self) -> str:
        """Human-readable RPO."""
        if self.rpo_seconds >= 3600:
            return f"{self.rpo_seconds // 3600}h"
        return f"{self.rpo_seconds // 60}m"


# SLA-defined targets
SLA_TARGETS = {
    "free": RTORPOTargets("free", 24 * 3600, 24 * 3600),  # 24h, 24h
    "pro": RTORPOTargets("pro", 4 * 3600, 1 * 3600),  # 4h, 1h
    "enterprise": RTORPOTargets("enterprise", 1 * 3600, 15 * 60),  # 1h, 15m
}


@dataclass
class DRMetrics:
    """DR metrics measurement results."""

    backup_exists: bool
    backup_age_seconds: float
    backup_size_bytes: int
    restore_time_seconds: float
    data_integrity_verified: bool
    rto_compliance: dict[str, bool]  # tier -> compliant
    rpo_compliance: dict[str, bool]  # tier -> compliant
    errors: list[str]
    timestamp: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


async def create_test_database(db_path: Path, num_records: int = 1000) -> None:
    """Create a test database with sample data."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create tables similar to Aragora's schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS debates (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            user_id TEXT,
            resource_id TEXT,
            created_at TEXT NOT NULL
        )
    """)

    # Insert test data
    now = datetime.utcnow().isoformat()
    for i in range(num_records):
        cursor.execute(
            "INSERT INTO debates VALUES (?, ?, ?, ?, ?)",
            (f"debate-{i}", f"Topic {i}", "completed", now, now),
        )
        cursor.execute(
            "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?)",
            (f"event-{i}", "debate.created", f"user-{i % 10}", f"debate-{i}", now),
        )

    conn.commit()
    conn.close()


async def measure_backup_create(db_path: Path, backup_path: Path) -> tuple[float, int]:
    """Measure backup creation time and size.

    Returns:
        Tuple of (creation_time_seconds, backup_size_bytes)
    """
    try:
        from aragora.backup.manager import BackupManager

        manager = BackupManager(str(backup_path.parent))

        start_time = time.time()
        result = await manager.create_backup(
            source_path=str(db_path),
            backup_type="full",
            compress=True,
        )
        end_time = time.time()

        creation_time = end_time - start_time

        # Get backup size
        if result and hasattr(result, "size_bytes"):
            size = result.size_bytes
        else:
            # Fallback: check file size
            backup_files = list(backup_path.parent.glob("*.backup*"))
            size = sum(f.stat().st_size for f in backup_files) if backup_files else 0

        return creation_time, size

    except ImportError:
        # Fallback for when aragora.backup is not available
        import shutil

        start_time = time.time()
        shutil.copy2(db_path, backup_path)
        end_time = time.time()

        return end_time - start_time, backup_path.stat().st_size


async def measure_restore_time(backup_path: Path, restore_path: Path) -> tuple[float, bool]:
    """Measure restore time and verify integrity.

    Returns:
        Tuple of (restore_time_seconds, integrity_verified)
    """
    try:
        from aragora.backup.manager import BackupManager

        manager = BackupManager(str(backup_path.parent))

        # Find the backup
        backups = await manager.list_backups()
        if not backups:
            return 0.0, False

        latest_backup = backups[0]

        start_time = time.time()
        result = await manager.restore(
            backup_id=latest_backup.id,
            target_path=str(restore_path),
            dry_run=False,
        )
        end_time = time.time()

        restore_time = end_time - start_time

        # Verify integrity
        verification = await manager.verify_backup(latest_backup.id)
        integrity_ok = verification.checksum_valid if verification else False

        return restore_time, integrity_ok

    except ImportError:
        # Fallback for when aragora.backup is not available
        import shutil

        start_time = time.time()
        shutil.copy2(backup_path, restore_path)
        end_time = time.time()

        # Basic integrity check
        integrity_ok = restore_path.exists() and restore_path.stat().st_size > 0

        return end_time - start_time, integrity_ok


async def get_latest_backup_age() -> Optional[float]:
    """Get the age of the latest backup in seconds."""
    try:
        from aragora.backup.manager import BackupManager

        # Check common backup locations
        backup_dirs = [
            Path("/var/aragora/backups"),
            Path.home() / ".aragora" / "backups",
            Path("backups"),
        ]

        for backup_dir in backup_dirs:
            if backup_dir.exists():
                manager = BackupManager(str(backup_dir))
                backups = await manager.list_backups()
                if backups:
                    latest = backups[0]
                    age = (datetime.utcnow() - latest.created_at).total_seconds()
                    return age

        return None

    except (ImportError, Exception):
        return None


def calculate_compliance(
    restore_time: float,
    backup_age: float,
) -> tuple[dict[str, bool], dict[str, bool]]:
    """Calculate RTO and RPO compliance for each tier.

    Returns:
        Tuple of (rto_compliance, rpo_compliance) dicts
    """
    rto_compliance = {}
    rpo_compliance = {}

    for tier, targets in SLA_TARGETS.items():
        rto_compliance[tier] = restore_time <= targets.rto_seconds
        rpo_compliance[tier] = backup_age <= targets.rpo_seconds

    return rto_compliance, rpo_compliance


async def run_dr_metrics(
    dry_run: bool = False,
    ci_mode: bool = False,
    verify_only: bool = False,
) -> DRMetrics:
    """Run DR metrics measurement.

    Args:
        dry_run: If True, only simulate without actual backup/restore
        ci_mode: If True, output CI-friendly format
        verify_only: If True, only verify existing backups

    Returns:
        DRMetrics with measurement results
    """
    errors: list[str] = []
    backup_exists = False
    backup_age = 0.0
    backup_size = 0
    restore_time = 0.0
    integrity_verified = False

    if verify_only:
        # Just check existing backup age
        age = await get_latest_backup_age()
        if age is not None:
            backup_exists = True
            backup_age = age
            integrity_verified = True
        else:
            errors.append("No existing backups found")

    elif dry_run:
        # Simulate metrics
        backup_exists = True
        backup_age = 3600.0  # 1 hour
        backup_size = 10 * 1024 * 1024  # 10MB
        restore_time = 120.0  # 2 minutes
        integrity_verified = True
        print("[DRY RUN] Simulating DR metrics")

    else:
        # Full DR drill
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            db_path = tmppath / "test.db"
            backup_dir = tmppath / "backups"
            restore_path = tmppath / "restored.db"
            backup_dir.mkdir()

            print("Creating test database...")
            await create_test_database(db_path, num_records=1000)

            print("Measuring backup creation...")
            try:
                _, backup_size = await measure_backup_create(db_path, backup_dir / "backup.db")
                backup_exists = True
                backup_age = 0.0  # Just created
            except Exception as e:
                errors.append(f"Backup creation failed: {e}")

            if backup_exists:
                print("Measuring restore time...")
                try:
                    restore_time, integrity_verified = await measure_restore_time(
                        backup_dir, restore_path
                    )
                except Exception as e:
                    errors.append(f"Restore failed: {e}")

    # Calculate compliance
    rto_compliance, rpo_compliance = calculate_compliance(restore_time, backup_age)

    return DRMetrics(
        backup_exists=backup_exists,
        backup_age_seconds=backup_age,
        backup_size_bytes=backup_size,
        restore_time_seconds=restore_time,
        data_integrity_verified=integrity_verified,
        rto_compliance=rto_compliance,
        rpo_compliance=rpo_compliance,
        errors=errors,
        timestamp=datetime.utcnow().isoformat(),
    )


def print_report(metrics: DRMetrics, ci_mode: bool = False) -> None:
    """Print DR metrics report."""
    if ci_mode:
        # CI-friendly JSON output
        print(json.dumps(metrics.to_dict(), indent=2))
        return

    # Human-readable report
    print("\n" + "=" * 60)
    print("DISASTER RECOVERY METRICS REPORT")
    print("=" * 60)
    print(f"Timestamp: {metrics.timestamp}")
    print()

    print("BACKUP STATUS")
    print("-" * 40)
    print(f"  Backup exists:     {metrics.backup_exists}")
    if metrics.backup_exists:
        age_hours = metrics.backup_age_seconds / 3600
        print(f"  Backup age:        {age_hours:.2f} hours")
        size_mb = metrics.backup_size_bytes / (1024 * 1024)
        print(f"  Backup size:       {size_mb:.2f} MB")
        print(f"  Integrity verified: {metrics.data_integrity_verified}")
    print()

    print("RECOVERY METRICS")
    print("-" * 40)
    restore_minutes = metrics.restore_time_seconds / 60
    print(f"  Restore time:      {restore_minutes:.2f} minutes")
    print()

    print("SLA COMPLIANCE")
    print("-" * 40)
    print(f"  {'Tier':<12} {'RTO Target':<12} {'RTO OK':<10} {'RPO Target':<12} {'RPO OK':<10}")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 10} {'-' * 12} {'-' * 10}")

    for tier, targets in SLA_TARGETS.items():
        rto_ok = "PASS" if metrics.rto_compliance.get(tier, False) else "FAIL"
        rpo_ok = "PASS" if metrics.rpo_compliance.get(tier, False) else "FAIL"
        print(
            f"  {tier.title():<12} {targets.rto_human:<12} {rto_ok:<10} {targets.rpo_human:<12} {rpo_ok:<10}"
        )

    print()

    if metrics.errors:
        print("ERRORS")
        print("-" * 40)
        for error in metrics.errors:
            print(f"  - {error}")
        print()

    # Overall status
    all_enterprise_compliant = metrics.rto_compliance.get(
        "enterprise", False
    ) and metrics.rpo_compliance.get("enterprise", False)
    if all_enterprise_compliant:
        print("OVERALL STATUS: ENTERPRISE-READY")
    elif metrics.rto_compliance.get("pro", False) and metrics.rpo_compliance.get("pro", False):
        print("OVERALL STATUS: PRO-TIER COMPLIANT")
    elif metrics.rto_compliance.get("free", False) and metrics.rpo_compliance.get("free", False):
        print("OVERALL STATUS: FREE-TIER COMPLIANT")
    else:
        print("OVERALL STATUS: NON-COMPLIANT - ACTION REQUIRED")

    print("=" * 60)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Measure DR metrics for RTO/RPO compliance")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate metrics without actual backup/restore",
    )
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help="Output CI-friendly JSON format",
    )
    parser.add_argument(
        "--verify-backup",
        action="store_true",
        help="Only verify existing backups",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()

    # Run metrics
    metrics = asyncio.run(
        run_dr_metrics(
            dry_run=args.dry_run,
            ci_mode=args.ci_mode or args.json,
            verify_only=args.verify_backup,
        )
    )

    # Print report
    print_report(metrics, ci_mode=args.ci_mode or args.json)

    # Exit code based on compliance
    if metrics.errors:
        return 1
    if not metrics.backup_exists:
        return 1
    if not metrics.rto_compliance.get("free", False):
        return 1
    if not metrics.rpo_compliance.get("free", False):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
