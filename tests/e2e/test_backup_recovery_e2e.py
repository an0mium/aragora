"""
E2E Backup and Disaster Recovery Tests for Aragora.

Validates backup and recovery functionality:
- Create backup success
- Restore backup integrity
- Incremental backup chain
- Backup during active debate
- RTO validation (recovery time < 5 min)

Run with: pytest tests/e2e/test_backup_recovery_e2e.py -v
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field

import pytest
import pytest_asyncio

from tests.e2e.harness import (
    E2ETestConfig,
    E2ETestHarness,
    e2e_environment,
)

# Mark all tests in this module as e2e
pytestmark = [pytest.mark.e2e]


# ============================================================================
# SLA Targets
# ============================================================================


@dataclass
class BackupRecoverySLAs:
    """Backup and recovery SLA targets."""

    # Recovery Time Objective (RTO) - maximum time to restore service
    rto_seconds: float = 300.0  # 5 minutes

    # Recovery Point Objective (RPO) - maximum data loss window
    rpo_seconds: float = 3600.0  # 1 hour

    # Backup creation time
    max_backup_time_seconds: float = 60.0

    # Verification time
    max_verification_time_seconds: float = 30.0

    # Backup integrity
    required_integrity_score: float = 1.0  # 100% integrity required


SLAS = BackupRecoverySLAs()


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_backup_dir():
    """Create a temporary backup directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_database(temp_backup_dir: Path):
    """Create a sample SQLite database for testing."""
    db_path = temp_backup_dir / "sample.db"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE debates (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            debate_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            content TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (debate_id) REFERENCES debates(id)
        )
    """)

    # Insert sample data
    for i in range(10):
        cursor.execute(
            "INSERT INTO debates (id, topic, status) VALUES (?, ?, ?)",
            (f"debate-{i}", f"Topic {i}", "completed"),
        )
        for j in range(5):
            cursor.execute(
                "INSERT INTO messages (id, debate_id, agent_id, content) VALUES (?, ?, ?, ?)",
                (f"msg-{i}-{j}", f"debate-{i}", f"agent-{j % 3}", f"Message content {j}"),
            )

    conn.commit()
    conn.close()

    return db_path


@pytest_asyncio.fixture
async def backup_harness():
    """Harness configured for backup tests."""
    config = E2ETestConfig(
        num_agents=2,
        agent_capabilities=["debate", "general"],
        agent_response_delay=0.01,
        timeout_seconds=60.0,
        task_timeout_seconds=30.0,
        heartbeat_interval=2.0,
        default_debate_rounds=1,
    )
    async with e2e_environment(config) as harness:
        yield harness


# ============================================================================
# Backup Creation Tests
# ============================================================================


class TestBackupCreation:
    """Test backup creation functionality."""

    def test_create_backup_success(self, temp_backup_dir: Path, sample_database: Path):
        """Test successful backup creation."""
        from aragora.backup.manager import BackupManager, BackupStatus, BackupType

        manager = BackupManager(backup_dir=temp_backup_dir)

        # Create backup
        backup = manager.create_backup(
            source_path=str(sample_database),
            backup_type=BackupType.FULL,
        )

        # Verify backup was created
        assert backup is not None
        assert backup.status in (BackupStatus.COMPLETED, BackupStatus.VERIFIED)
        assert backup.backup_type == BackupType.FULL
        assert backup.size_bytes > 0
        assert Path(backup.backup_path).exists()

    def test_backup_creates_checksum(self, temp_backup_dir: Path, sample_database: Path):
        """Test backup creates valid checksum."""
        from aragora.backup.manager import BackupManager, BackupType

        manager = BackupManager(backup_dir=temp_backup_dir)
        backup = manager.create_backup(
            source_path=str(sample_database),
            backup_type=BackupType.FULL,
        )

        assert backup.checksum
        assert len(backup.checksum) == 64  # SHA-256 hex digest

    def test_backup_captures_table_info(self, temp_backup_dir: Path, sample_database: Path):
        """Test backup captures table information."""
        from aragora.backup.manager import BackupManager, BackupType

        manager = BackupManager(backup_dir=temp_backup_dir)
        backup = manager.create_backup(
            source_path=str(sample_database),
            backup_type=BackupType.FULL,
        )

        # Should have table info
        assert "debates" in backup.tables
        assert "messages" in backup.tables
        assert backup.row_counts.get("debates", 0) == 10
        assert backup.row_counts.get("messages", 0) == 50

    def test_backup_completes_within_sla(self, temp_backup_dir: Path, sample_database: Path):
        """Test backup completes within time SLA."""
        from aragora.backup.manager import BackupManager, BackupType

        manager = BackupManager(backup_dir=temp_backup_dir)

        start = time.time()
        backup = manager.create_backup(
            source_path=str(sample_database),
            backup_type=BackupType.FULL,
        )
        elapsed = time.time() - start

        assert elapsed < SLAS.max_backup_time_seconds, (
            f"Backup took {elapsed:.2f}s, exceeds SLA {SLAS.max_backup_time_seconds}s"
        )
        assert backup.duration_seconds < SLAS.max_backup_time_seconds


# ============================================================================
# Backup Verification Tests
# ============================================================================


class TestBackupVerification:
    """Test backup verification functionality."""

    def test_verify_backup_integrity(self, temp_backup_dir: Path, sample_database: Path):
        """Test backup verification passes for valid backup."""
        from aragora.backup.manager import BackupManager, BackupType

        manager = BackupManager(backup_dir=temp_backup_dir)
        backup = manager.create_backup(
            source_path=str(sample_database),
            backup_type=BackupType.FULL,
        )

        # Verify the backup
        result = manager.verify_backup(backup.id)

        assert result.verified
        assert result.checksum_valid
        assert result.tables_valid
        assert result.row_counts_valid

    def test_verify_detects_corruption(self, temp_backup_dir: Path, sample_database: Path):
        """Test verification detects corrupted backup."""
        from aragora.backup.manager import BackupManager, BackupType

        manager = BackupManager(backup_dir=temp_backup_dir)
        backup = manager.create_backup(
            source_path=str(sample_database),
            backup_type=BackupType.FULL,
        )

        # Corrupt the backup file (append random data)
        backup_path = Path(backup.backup_path)
        with open(backup_path, "ab") as f:
            f.write(b"corrupted data")

        # Verification should detect corruption
        result = manager.verify_backup(backup.id)

        # Checksum should fail
        assert not result.checksum_valid

    def test_verification_completes_within_sla(self, temp_backup_dir: Path, sample_database: Path):
        """Test verification completes within time SLA."""
        from aragora.backup.manager import BackupManager, BackupType

        manager = BackupManager(backup_dir=temp_backup_dir)
        backup = manager.create_backup(
            source_path=str(sample_database),
            backup_type=BackupType.FULL,
        )

        start = time.time()
        result = manager.verify_backup(backup.id)
        elapsed = time.time() - start

        assert elapsed < SLAS.max_verification_time_seconds, (
            f"Verification took {elapsed:.2f}s, exceeds SLA"
        )


# ============================================================================
# Restore Tests
# ============================================================================


class TestBackupRestore:
    """Test backup restore functionality."""

    def test_restore_backup_integrity(self, temp_backup_dir: Path, sample_database: Path):
        """Test restored database has correct data."""
        from aragora.backup.manager import BackupManager, BackupType

        manager = BackupManager(backup_dir=temp_backup_dir)
        backup = manager.create_backup(
            source_path=str(sample_database),
            backup_type=BackupType.FULL,
        )

        # Restore to new location
        restore_path = temp_backup_dir / "restored.db"
        success = manager.restore_backup(backup.id, str(restore_path))

        assert success
        assert restore_path.exists()

        # Verify restored data
        conn = sqlite3.connect(str(restore_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM debates")
        debate_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM messages")
        message_count = cursor.fetchone()[0]

        conn.close()

        assert debate_count == 10
        assert message_count == 50

    def test_rto_validation(self, temp_backup_dir: Path, sample_database: Path):
        """Test Recovery Time Objective (RTO) is met."""
        from aragora.backup.manager import BackupManager, BackupType

        manager = BackupManager(backup_dir=temp_backup_dir)
        backup = manager.create_backup(
            source_path=str(sample_database),
            backup_type=BackupType.FULL,
        )

        restore_path = temp_backup_dir / "rto_test.db"

        # Measure restore time
        start = time.time()
        success = manager.restore_backup(backup.id, str(restore_path))
        restore_time = time.time() - start

        assert success
        assert restore_time < SLAS.rto_seconds, (
            f"Restore took {restore_time:.2f}s, exceeds RTO of {SLAS.rto_seconds}s"
        )

    def test_restore_preserves_schema(self, temp_backup_dir: Path, sample_database: Path):
        """Test restore preserves database schema."""
        from aragora.backup.manager import BackupManager, BackupType

        manager = BackupManager(backup_dir=temp_backup_dir)
        backup = manager.create_backup(
            source_path=str(sample_database),
            backup_type=BackupType.FULL,
        )

        restore_path = temp_backup_dir / "schema_test.db"
        manager.restore_backup(backup.id, str(restore_path))

        # Verify schema
        conn = sqlite3.connect(str(restore_path))
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]

        conn.close()

        assert "debates" in tables
        assert "messages" in tables


# ============================================================================
# Incremental Backup Tests
# ============================================================================


class TestIncrementalBackup:
    """Test incremental backup functionality."""

    def test_incremental_backup_chain(self, temp_backup_dir: Path, sample_database: Path):
        """Test incremental backup chain works correctly."""
        from aragora.backup.manager import BackupManager, BackupType

        manager = BackupManager(backup_dir=temp_backup_dir)

        # Create full backup
        full_backup = manager.create_backup(
            source_path=str(sample_database),
            backup_type=BackupType.FULL,
        )
        assert full_backup is not None

        # Add more data
        conn = sqlite3.connect(str(sample_database))
        cursor = conn.cursor()
        for i in range(10, 15):
            cursor.execute(
                "INSERT INTO debates (id, topic, status) VALUES (?, ?, ?)",
                (f"debate-{i}", f"New Topic {i}", "pending"),
            )
        conn.commit()
        conn.close()

        # Create incremental backup
        incr_backup = manager.create_backup(
            source_path=str(sample_database),
            backup_type=BackupType.INCREMENTAL,
        )

        assert incr_backup is not None
        assert incr_backup.backup_type == BackupType.INCREMENTAL
        # Incremental should be smaller than full (though for small DBs may not be)
        assert incr_backup.size_bytes > 0

    def test_list_backups_returns_all(self, temp_backup_dir: Path, sample_database: Path):
        """Test list_backups returns all backups."""
        from aragora.backup.manager import BackupManager, BackupType

        manager = BackupManager(backup_dir=temp_backup_dir)

        # Create multiple backups
        manager.create_backup(str(sample_database), BackupType.FULL)
        manager.create_backup(str(sample_database), BackupType.FULL)
        manager.create_backup(str(sample_database), BackupType.INCREMENTAL)

        backups = manager.list_backups()

        assert len(backups) >= 3


# ============================================================================
# Backup During Active Operations Tests
# ============================================================================


class TestBackupDuringActiveOperations:
    """Test backup behavior during active operations."""

    @pytest.mark.asyncio
    async def test_backup_during_active_debate(
        self, backup_harness: E2ETestHarness, temp_backup_dir: Path, sample_database: Path
    ):
        """Test backup can be created during active debate."""
        from aragora.backup.manager import BackupManager, BackupType

        manager = BackupManager(backup_dir=temp_backup_dir)

        # Start a debate in background
        async def run_debate():
            await backup_harness.run_debate("Background debate", rounds=2)

        debate_task = asyncio.create_task(run_debate())

        # Give debate a moment to start
        await asyncio.sleep(0.1)

        # Create backup while debate is running
        backup = manager.create_backup(
            source_path=str(sample_database),
            backup_type=BackupType.FULL,
        )

        assert backup is not None
        assert backup.size_bytes > 0

        # Wait for debate to finish
        await debate_task

    @pytest.mark.asyncio
    async def test_concurrent_backup_operations(self, temp_backup_dir: Path, sample_database: Path):
        """Test system handles concurrent backup requests."""
        from aragora.backup.manager import BackupManager, BackupType

        manager = BackupManager(backup_dir=temp_backup_dir)

        # Create backups concurrently (simulated with threads since BackupManager is sync)
        async def create_backup_async(backup_type: BackupType):
            # Run sync backup in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: manager.create_backup(str(sample_database), backup_type),
            )

        # Create multiple backups concurrently
        tasks = [
            create_backup_async(BackupType.FULL),
            create_backup_async(BackupType.FULL),
            create_backup_async(BackupType.FULL),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # At least some backups should succeed
        successful = [r for r in results if not isinstance(r, Exception) and r is not None]
        assert len(successful) >= 1


# ============================================================================
# Retention Policy Tests
# ============================================================================


class TestRetentionPolicy:
    """Test backup retention policy enforcement."""

    def test_cleanup_expired_backups(self, temp_backup_dir: Path, sample_database: Path):
        """Test expired backups are cleaned up."""
        from aragora.backup.manager import BackupManager, BackupType, RetentionPolicy

        # Create manager with short retention
        policy = RetentionPolicy(
            keep_daily=2,
            keep_weekly=1,
            keep_monthly=1,
            min_backups=2,
        )
        manager = BackupManager(backup_dir=temp_backup_dir, retention_policy=policy)

        # Create several backups
        for _ in range(5):
            manager.create_backup(str(sample_database), BackupType.FULL)

        # Apply retention policy
        removed = manager.apply_retention_policy()

        # Should have removed some backups (or kept at least min_backups)
        remaining = manager.list_backups()
        assert len(remaining) >= policy.min_backups

    def test_get_latest_backup(self, temp_backup_dir: Path, sample_database: Path):
        """Test get_latest_backup returns most recent."""
        from aragora.backup.manager import BackupManager, BackupType

        manager = BackupManager(backup_dir=temp_backup_dir)

        # Create multiple backups
        backup1 = manager.create_backup(str(sample_database), BackupType.FULL)
        time.sleep(0.1)  # Ensure different timestamps
        backup2 = manager.create_backup(str(sample_database), BackupType.FULL)
        time.sleep(0.1)
        backup3 = manager.create_backup(str(sample_database), BackupType.FULL)

        latest = manager.get_latest_backup()

        assert latest is not None
        assert latest.id == backup3.id


# ============================================================================
# Disaster Recovery Simulation Tests
# ============================================================================


class TestDisasterRecoverySimulation:
    """Simulate disaster recovery scenarios."""

    def test_full_recovery_workflow(self, temp_backup_dir: Path, sample_database: Path):
        """Test complete disaster recovery workflow."""
        from aragora.backup.manager import BackupManager, BackupType

        manager = BackupManager(backup_dir=temp_backup_dir)

        # Step 1: Create backup of current state
        backup = manager.create_backup(str(sample_database), BackupType.FULL)
        assert backup is not None

        # Step 2: Verify backup
        verification = manager.verify_backup(backup.id)
        assert verification.verified

        # Step 3: Simulate disaster (delete original database)
        original_data_path = sample_database
        os.remove(original_data_path)
        assert not original_data_path.exists()

        # Step 4: Restore from backup
        restored_path = temp_backup_dir / "recovered.db"
        success = manager.restore_backup(backup.id, str(restored_path))
        assert success

        # Step 5: Verify restored data
        conn = sqlite3.connect(str(restored_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM debates")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 10  # Original data restored

    def test_recovery_from_latest_backup(self, temp_backup_dir: Path, sample_database: Path):
        """Test recovery automatically uses latest backup."""
        from aragora.backup.manager import BackupManager, BackupType

        manager = BackupManager(backup_dir=temp_backup_dir)

        # Create multiple backups
        manager.create_backup(str(sample_database), BackupType.FULL)

        # Add more data
        conn = sqlite3.connect(str(sample_database))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO debates (id, topic, status) VALUES (?, ?, ?)",
            ("new-debate", "New Topic", "pending"),
        )
        conn.commit()
        conn.close()

        # Create another backup with new data
        latest_backup = manager.create_backup(str(sample_database), BackupType.FULL)

        # Restore from latest
        restore_path = temp_backup_dir / "latest_restore.db"
        success = manager.restore_backup(latest_backup.id, str(restore_path))

        assert success

        # Verify new data is present
        conn = sqlite3.connect(str(restore_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM debates")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 11  # Original 10 + 1 new


# ============================================================================
# Metrics and Monitoring Tests
# ============================================================================


class TestBackupMetrics:
    """Test backup metrics and monitoring."""

    def test_backup_records_duration(self, temp_backup_dir: Path, sample_database: Path):
        """Test backup records duration correctly."""
        from aragora.backup.manager import BackupManager, BackupType

        manager = BackupManager(backup_dir=temp_backup_dir)
        backup = manager.create_backup(str(sample_database), BackupType.FULL)

        assert backup.duration_seconds > 0
        assert backup.duration_seconds < 60  # Should be fast for small DB

    def test_backup_records_sizes(self, temp_backup_dir: Path, sample_database: Path):
        """Test backup records size information."""
        from aragora.backup.manager import BackupManager, BackupType

        manager = BackupManager(backup_dir=temp_backup_dir)
        backup = manager.create_backup(str(sample_database), BackupType.FULL)

        assert backup.size_bytes > 0

        # If compressed, compressed size should be recorded
        if backup.compressed_size_bytes > 0:
            # Compressed should typically be smaller or equal
            assert backup.compressed_size_bytes <= backup.size_bytes * 2
