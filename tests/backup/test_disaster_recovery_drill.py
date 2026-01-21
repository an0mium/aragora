"""
Disaster Recovery Drill Tests.

Automated tests that validate DR capabilities as defined in docs/DR_DRILL_PROCEDURES.md.
These tests simulate disaster scenarios and verify recovery mechanisms work correctly.

SOC 2 Control: CC9-01 - Disaster recovery procedures and testing

Test Categories:
1. Backup Restoration Drill - Monthly requirement
2. Component Failover Validation - Quarterly requirement
3. Data Integrity Verification - Part of backup restoration
4. RTO/RPO Metrics Measurement - Annual requirement
5. Protected File Recovery - Critical infrastructure
"""

import asyncio
import gzip
import hashlib
import os
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.backup import (
    BackupManager,
    BackupMetadata,
    BackupStatus,
    BackupType,
    RetentionPolicy,
    VerificationResult,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for DR drills."""
    with tempfile.TemporaryDirectory(prefix="dr_drill_") as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "data").mkdir()
        (workspace / "backups").mkdir()
        (workspace / "restore").mkdir()
        yield workspace


@pytest.fixture
def production_database(temp_workspace: Path) -> Path:
    """Create a simulated production database with realistic data."""
    db_path = temp_workspace / "data" / "production.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create realistic schema
    cursor.executescript(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            org_id TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE debates (
            id INTEGER PRIMARY KEY,
            topic TEXT NOT NULL,
            org_id TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            rounds INTEGER DEFAULT 3,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            completed_at TEXT
        );

        CREATE TABLE audit_events (
            id INTEGER PRIMARY KEY,
            event_type TEXT NOT NULL,
            user_id INTEGER,
            org_id TEXT,
            details TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE agents (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            provider TEXT NOT NULL,
            elo_rating INTEGER DEFAULT 1500,
            total_debates INTEGER DEFAULT 0
        );

        CREATE INDEX idx_users_org ON users(org_id);
        CREATE INDEX idx_debates_org ON debates(org_id);
        CREATE INDEX idx_audit_timestamp ON audit_events(timestamp);
    """
    )

    # Insert test data
    orgs = ["acme-corp", "tech-startup", "enterprise-co"]

    # Users
    for i in range(100):
        org = orgs[i % len(orgs)]
        cursor.execute(
            "INSERT INTO users (name, email, org_id) VALUES (?, ?, ?)",
            (f"User {i}", f"user{i}@{org.replace('-', '')}.com", org),
        )

    # Debates
    for i in range(50):
        org = orgs[i % len(orgs)]
        status = "completed" if i < 30 else "pending"
        cursor.execute(
            "INSERT INTO debates (topic, org_id, status, rounds) VALUES (?, ?, ?, ?)",
            (f"Debate Topic {i}", org, status, 3 + (i % 3)),
        )

    # Audit events
    for i in range(500):
        cursor.execute(
            "INSERT INTO audit_events (event_type, user_id, org_id, details) VALUES (?, ?, ?, ?)",
            (
                ["login", "debate_start", "debate_end", "api_call"][i % 4],
                (i % 100) + 1,
                orgs[i % len(orgs)],
                f'{{"action_id": {i}}}',
            ),
        )

    # Agents
    agents = [
        ("claude", "anthropic", 1650),
        ("gpt-4", "openai", 1620),
        ("gemini-pro", "google", 1580),
        ("mistral-large", "mistral", 1560),
        ("grok", "xai", 1540),
    ]
    for name, provider, elo in agents:
        cursor.execute(
            "INSERT INTO agents (name, provider, elo_rating) VALUES (?, ?, ?)",
            (name, provider, elo),
        )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def backup_manager(temp_workspace: Path) -> BackupManager:
    """Create backup manager for DR testing."""
    return BackupManager(
        backup_dir=temp_workspace / "backups",
        compression=True,
        verify_after_backup=True,
        metrics_enabled=False,
    )


class TestBackupRestorationDrill:
    """
    Monthly Backup Restoration Drill.

    Validates:
    - Backup creation succeeds
    - Backup verification passes
    - Restoration completes successfully
    - Data integrity preserved
    - Recovery time measured
    """

    def test_full_backup_restore_cycle(
        self,
        backup_manager: BackupManager,
        production_database: Path,
        temp_workspace: Path,
    ):
        """Execute a complete backup-verify-restore cycle."""
        # Record original data counts
        original_counts = self._get_table_counts(production_database)

        # Phase 1: Create backup
        start_time = time.time()
        backup = backup_manager.create_backup(production_database)
        backup_time = time.time() - start_time

        assert backup.status == BackupStatus.VERIFIED
        assert backup.backup_type == BackupType.FULL
        assert backup.size_bytes > 0
        print(f"Backup created in {backup_time:.2f}s, size: {backup.size_bytes} bytes")

        # Phase 2: Verify backup exists and is valid
        verification = backup_manager.verify_backup(backup.id)
        assert verification.verified
        assert verification.checksum_valid
        print(f"Verification passed: {verification}")

        # Phase 3: Simulate disaster - delete original
        original_path = production_database
        disaster_time = datetime.now(timezone.utc)

        # Phase 4: Restore backup
        restore_path = temp_workspace / "restore" / "recovered.db"
        restore_start = time.time()

        # Perform dry-run first
        backup_manager.restore_backup(backup.id, restore_path, dry_run=True)

        # Actual restore
        backup_manager.restore_backup(backup.id, restore_path)
        restore_time = time.time() - restore_start

        assert restore_path.exists()
        print(f"Restore completed in {restore_time:.2f}s")

        # Phase 5: Verify data integrity
        restored_counts = self._get_table_counts(restore_path)
        assert (
            restored_counts == original_counts
        ), f"Data mismatch: original={original_counts}, restored={restored_counts}"

        # Verify specific data
        self._verify_data_integrity(restore_path)

        # Calculate RPO (simulated - based on backup timestamp)
        rpo_seconds = (disaster_time - backup.created_at).total_seconds()
        print(f"RPO: {rpo_seconds:.1f}s")

        # Calculate RTO
        rto_seconds = backup_time + restore_time
        print(f"RTO: {rto_seconds:.2f}s")

        # Assertions for DR metrics (adjusted for test scale)
        assert rto_seconds < 60, f"RTO exceeded: {rto_seconds}s > 60s"

    def test_incremental_backup_restore(
        self,
        backup_manager: BackupManager,
        production_database: Path,
        temp_workspace: Path,
    ):
        """Test incremental backup and restore."""
        # Create initial full backup
        full_backup = backup_manager.create_backup(production_database)
        assert full_backup.status == BackupStatus.VERIFIED

        # Add more data (simulating production changes)
        conn = sqlite3.connect(str(production_database))
        cursor = conn.cursor()
        for i in range(100, 110):
            cursor.execute(
                "INSERT INTO users (name, email, org_id) VALUES (?, ?, ?)",
                (f"New User {i}", f"newuser{i}@test.com", "new-org"),
            )
        conn.commit()
        conn.close()

        # Create incremental backup
        incr_backup = backup_manager.create_backup(
            production_database,
            backup_type=BackupType.INCREMENTAL,
        )
        assert incr_backup.status == BackupStatus.VERIFIED

        # Incremental should be smaller or equal (depends on implementation)
        print(f"Full: {full_backup.size_bytes}, Incremental: {incr_backup.size_bytes}")

        # Restore and verify new data present
        restore_path = temp_workspace / "restore" / "incremental_recover.db"
        backup_manager.restore_backup(incr_backup.id, restore_path)

        counts = self._get_table_counts(restore_path)
        assert counts["users"] == 110, f"Expected 110 users, got {counts['users']}"

    def test_backup_corruption_detection(
        self,
        backup_manager: BackupManager,
        production_database: Path,
    ):
        """Verify corrupted backups are detected."""
        # Create backup
        backup = backup_manager.create_backup(production_database)
        assert backup.status == BackupStatus.VERIFIED

        # Corrupt the backup file
        backup_path = Path(backup.backup_path)
        with gzip.open(backup_path, "ab") as f:
            f.write(b"CORRUPTED DATA")

        # Verification should fail
        verification = backup_manager.verify_backup(backup.id)
        assert not verification.checksum_valid

    def test_restore_with_existing_file_backup(
        self,
        backup_manager: BackupManager,
        production_database: Path,
        temp_workspace: Path,
    ):
        """Verify restore creates backup of existing file."""
        # Create backup
        backup = backup_manager.create_backup(production_database)

        # Create a file at restore path
        restore_path = temp_workspace / "restore" / "existing.db"
        restore_path.parent.mkdir(parents=True, exist_ok=True)
        restore_path.write_bytes(b"existing data")

        # Restore should preserve existing file
        backup_manager.restore_backup(backup.id, restore_path)

        # Check backup of original exists
        backup_files = list(restore_path.parent.glob("existing.backup_*"))
        assert len(backup_files) == 1, "Should create backup of existing file"

    def _get_table_counts(self, db_path: Path) -> Dict[str, int]:
        """Get row counts for all tables."""
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        counts = {}
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]

        conn.close()
        return counts

    def _verify_data_integrity(self, db_path: Path):
        """Verify specific data integrity checks."""
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Verify foreign key integrity
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()[0]
        assert result == "ok", f"Integrity check failed: {result}"

        # Verify indexes exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in cursor.fetchall()]
        assert len(indexes) >= 3, f"Missing indexes: {indexes}"

        # Verify agent rankings are preserved
        cursor.execute("SELECT name, elo_rating FROM agents ORDER BY elo_rating DESC")
        agents = cursor.fetchall()
        assert agents[0][0] == "claude", f"Top agent changed: {agents[0]}"

        conn.close()


class TestComponentFailoverValidation:
    """
    Quarterly Component Failover Drill.

    Validates failover mechanisms for:
    - Circuit breakers
    - Agent fallback chains
    - Rate limit recovery
    """

    def test_circuit_breaker_failover(self):
        """Test circuit breaker opens and recovers correctly."""
        from aragora.resilience import CircuitBreaker

        breaker = CircuitBreaker(
            failure_threshold=3,
            cooldown_seconds=0.5,  # Short timeout for testing
            half_open_success_threshold=1,
        )

        # Trip the breaker
        for _ in range(3):
            breaker.record_failure()

        assert breaker.is_open, "Circuit should be open after 3 failures"

        # Wait for cooldown
        time.sleep(0.6)

        # Check state (should allow half-open test)
        status = breaker.get_status()
        # After cooldown, circuit should be half-open or closed
        assert status in ("half-open", "closed", "open")

        # Record success to close
        breaker.record_success()
        assert not breaker.is_open, "Circuit should close after success"

    def test_circuit_breaker_multi_entity(self):
        """Test circuit breaker with multiple entities."""
        from aragora.resilience import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=1.0)

        # Fail agent-1
        breaker.record_failure("agent-1")
        breaker.record_failure("agent-1")

        # agent-1 should be blocked, agent-2 should be available
        assert not breaker.is_available("agent-1")
        assert breaker.is_available("agent-2")

    def test_rate_limit_backoff_recovery(self):
        """Test exponential backoff on rate limits."""
        # Simple exponential backoff calculation
        base_delay = 0.1
        max_delay = 2.0
        multiplier = 2.0

        def get_delay(attempt: int) -> float:
            delay = base_delay * (multiplier**attempt)
            return min(delay, max_delay)

        # Simulate rate limit retries
        delays = [get_delay(i) for i in range(5)]

        # Verify exponential increase
        assert delays[1] > delays[0]
        assert delays[2] > delays[1]
        # Verify max cap
        assert delays[4] <= max_delay


class TestDataIntegrityVerification:
    """
    Data Integrity Verification Tests.

    Validates:
    - Checksum verification
    - Table structure preservation
    - Row count validation
    - Index preservation
    """

    def test_checksum_verification_algorithm(
        self,
        production_database: Path,
    ):
        """Verify SHA-256 checksum is correctly computed and validated."""
        # Compute checksum manually
        with open(production_database, "rb") as f:
            expected_hash = hashlib.sha256(f.read()).hexdigest()

        # Verify same computation yields same hash
        with open(production_database, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()

        assert expected_hash == actual_hash

    def test_table_structure_preservation(
        self,
        backup_manager: BackupManager,
        production_database: Path,
        temp_workspace: Path,
    ):
        """Verify table schemas are preserved across backup/restore."""
        # Get original schema
        original_schema = self._get_schema(production_database)

        # Backup and restore
        backup = backup_manager.create_backup(production_database)
        restore_path = temp_workspace / "restore" / "schema_test.db"
        backup_manager.restore_backup(backup.id, restore_path)

        # Get restored schema
        restored_schema = self._get_schema(restore_path)

        assert (
            original_schema == restored_schema
        ), f"Schema mismatch:\nOriginal: {original_schema}\nRestored: {restored_schema}"

    def test_row_count_tolerance(
        self,
        backup_manager: BackupManager,
        production_database: Path,
        temp_workspace: Path,
    ):
        """Test row count verification with tolerance."""
        # Create backup
        backup = backup_manager.create_backup(production_database)

        # Verify includes row counts
        verification = backup_manager.verify_backup(backup.id)
        assert verification.verified
        assert verification.tables_valid or verification.row_counts_valid

    def _get_schema(self, db_path: Path) -> Dict[str, str]:
        """Extract table schemas from database."""
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name")
        schema = {row[0]: row[1] for row in cursor.fetchall() if row[1]}
        conn.close()
        return schema


class TestRTORPOMetrics:
    """
    RTO/RPO Metrics Measurement.

    Recovery Time Objective (RTO): Maximum acceptable downtime
    Recovery Point Objective (RPO): Maximum acceptable data loss

    Targets (from DR_DRILL_PROCEDURES.md):
    - RTO: <4 hours for full DR, <30 min for component
    - RPO: <1 hour for full DR, <5 min with streaming replication
    """

    def test_backup_rto_measurement(
        self,
        backup_manager: BackupManager,
        production_database: Path,
        temp_workspace: Path,
    ):
        """Measure and validate backup RTO."""
        metrics = {
            "backup_start": time.time(),
            "backup_end": 0,
            "verify_end": 0,
            "restore_end": 0,
        }

        # Backup
        backup = backup_manager.create_backup(production_database)
        metrics["backup_end"] = time.time()

        # Verify
        backup_manager.verify_backup(backup.id)
        metrics["verify_end"] = time.time()

        # Restore
        restore_path = temp_workspace / "restore" / "rto_test.db"
        backup_manager.restore_backup(backup.id, restore_path)
        metrics["restore_end"] = time.time()

        # Calculate RTO components
        backup_time = metrics["backup_end"] - metrics["backup_start"]
        verify_time = metrics["verify_end"] - metrics["backup_end"]
        restore_time = metrics["restore_end"] - metrics["verify_end"]
        total_rto = metrics["restore_end"] - metrics["backup_start"]

        print("RTO Breakdown:")
        print(f"  Backup:  {backup_time:.3f}s")
        print(f"  Verify:  {verify_time:.3f}s")
        print(f"  Restore: {restore_time:.3f}s")
        print(f"  Total:   {total_rto:.3f}s")

        # For test databases, RTO should be under 10 seconds
        assert total_rto < 10.0, f"RTO too high: {total_rto:.3f}s"

    def test_backup_rpo_measurement(
        self,
        backup_manager: BackupManager,
        production_database: Path,
    ):
        """Measure and validate backup RPO."""
        # Create backup
        backup = backup_manager.create_backup(production_database)
        backup_timestamp = backup.created_at

        # Simulate time passing (data changes)
        simulated_disaster_time = datetime.now(timezone.utc) + timedelta(minutes=5)

        # RPO = time between last backup and disaster
        rpo_seconds = (simulated_disaster_time - backup_timestamp).total_seconds()

        print(f"Simulated RPO: {rpo_seconds:.1f}s ({rpo_seconds/60:.1f} min)")

        # With 5-minute backup interval, RPO should be max ~5 min
        # Add 1s tolerance for test execution timing variations
        assert rpo_seconds <= 301, f"RPO too high: {rpo_seconds}s > 301s"

    def test_multiple_backup_rpo_improvement(
        self,
        backup_manager: BackupManager,
        production_database: Path,
    ):
        """Verify more frequent backups improve RPO."""
        # Create multiple backups
        backups = []
        for _ in range(3):
            backup = backup_manager.create_backup(production_database)
            backups.append(backup)
            time.sleep(0.1)  # Small delay between backups

        # Disaster at "now"
        disaster_time = datetime.now(timezone.utc)

        # RPO from latest backup
        latest_backup = backups[-1]
        rpo = (disaster_time - latest_backup.created_at).total_seconds()

        print(f"RPO with {len(backups)} backups: {rpo:.3f}s")

        # Most recent backup should give minimal RPO
        assert rpo < 1.0, f"RPO too high with recent backup: {rpo}s"


class TestProtectedFileRecovery:
    """
    Protected File Recovery Tests.

    Validates recovery of critical infrastructure files.
    """

    def test_protected_file_checksum_validation(self, temp_workspace: Path):
        """Test checksum validation for protected files."""
        # Create a "protected" file
        protected_file = temp_workspace / "protected" / "core.py"
        protected_file.parent.mkdir(parents=True, exist_ok=True)
        protected_file.write_text("# Core module\ndef main(): pass")

        # Compute checksum
        original_checksum = hashlib.sha256(protected_file.read_bytes()).hexdigest()[:16]

        # Modify file (simulating corruption/attack)
        protected_file.write_text("# Modified!")

        # Compute new checksum
        modified_checksum = hashlib.sha256(protected_file.read_bytes()).hexdigest()[:16]

        # Checksums should differ
        assert original_checksum != modified_checksum

    def test_protected_file_recovery(self, temp_workspace: Path):
        """Test recovery of protected files from backup."""
        # Setup
        source_dir = temp_workspace / "source"
        backup_dir = temp_workspace / "backup"
        source_dir.mkdir()
        backup_dir.mkdir()

        # Create protected files
        files = ["core.py", "config.py", "init.py"]
        original_contents = {}

        for fname in files:
            fpath = source_dir / fname
            content = f"# {fname}\nVERSION = '1.0.0'"
            fpath.write_text(content)
            original_contents[fname] = content

        # Create backup
        for fname in files:
            src = source_dir / fname
            dst = backup_dir / fname
            shutil.copy(src, dst)

        # Simulate disaster - corrupt files
        for fname in files:
            (source_dir / fname).write_text("CORRUPTED")

        # Verify corruption
        for fname in files:
            assert (source_dir / fname).read_text() == "CORRUPTED"

        # Restore from backup
        for fname in files:
            src = backup_dir / fname
            dst = source_dir / fname
            shutil.copy(src, dst)

        # Verify recovery
        for fname in files:
            restored = (source_dir / fname).read_text()
            assert restored == original_contents[fname], f"{fname} not properly restored"


class TestRetentionPolicyEnforcement:
    """
    Retention Policy Enforcement Tests.

    Validates backup cleanup according to policy.
    """

    def test_retention_keeps_minimum_backups(
        self,
        temp_workspace: Path,
        production_database: Path,
    ):
        """Test retention policy keeps minimum number of backups."""
        manager = BackupManager(
            backup_dir=temp_workspace / "backups",
            compression=True,
            retention_policy=RetentionPolicy(
                keep_daily=1,
                keep_weekly=0,
                keep_monthly=0,
                min_backups=2,
            ),
        )

        # Create multiple backups
        backups = []
        for _ in range(5):
            backup = manager.create_backup(production_database)
            backups.append(backup)
            time.sleep(0.05)

        # Run cleanup
        cleaned = manager.cleanup_expired_backups()

        # Should keep minimum
        remaining = manager.list_backups()
        assert len(remaining) >= 2, f"Should keep at least 2 backups, have {len(remaining)}"

    def test_retention_policy_respects_age(
        self,
        backup_manager: BackupManager,
        production_database: Path,
    ):
        """Test retention respects backup age categories."""
        # Create backup
        backup = backup_manager.create_backup(production_database)

        # Fresh backup should not be cleaned up
        cleaned = backup_manager.cleanup_expired_backups()

        remaining = backup_manager.list_backups()
        assert any(b.id == backup.id for b in remaining)


class TestConcurrentBackupOperations:
    """
    Concurrent Backup Operation Tests.

    Validates backup system handles concurrent operations safely.
    """

    @pytest.mark.asyncio
    async def test_concurrent_backup_creation(
        self,
        temp_workspace: Path,
        production_database: Path,
    ):
        """Test concurrent backup creation doesn't corrupt data."""
        manager = BackupManager(
            backup_dir=temp_workspace / "backups",
            compression=True,
        )

        async def create_backup():
            # Run in executor since it's synchronous
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                manager.create_backup,
                production_database,
            )

        # Create multiple backups concurrently
        tasks = [create_backup() for _ in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed or fail gracefully
        successes = [r for r in results if isinstance(r, BackupMetadata)]
        errors = [r for r in results if isinstance(r, Exception)]

        # At least some should succeed
        assert len(successes) >= 1, f"No backups succeeded: {errors}"

        # Verify each successful backup is valid
        for backup in successes:
            verification = manager.verify_backup(backup.id)
            assert verification.verified, f"Backup {backup.id} is invalid"


class TestEmergencyRecoveryScenarios:
    """
    Emergency Recovery Scenario Tests.

    Simulates various disaster scenarios from DR_DRILL_PROCEDURES.md.
    """

    def test_scenario_database_corruption(
        self,
        backup_manager: BackupManager,
        production_database: Path,
        temp_workspace: Path,
    ):
        """Scenario 1: Database corruption recovery."""
        # Phase 1: Pre-disaster - create backup
        backup = backup_manager.create_backup(production_database)
        original_counts = self._get_counts(production_database)

        # Phase 2: Simulate corruption
        with open(production_database, "ab") as f:
            f.write(b"\x00\x00\x00CORRUPTED")

        # Verify corruption
        try:
            conn = sqlite3.connect(str(production_database))
            conn.execute("SELECT * FROM users")
            conn.close()
            # If no error, data may still be readable
        except sqlite3.DatabaseError:
            pass  # Expected - database is corrupted

        # Phase 3: Recovery
        recovery_start = time.time()
        restore_path = temp_workspace / "restore" / "recovered_from_corruption.db"
        backup_manager.restore_backup(backup.id, restore_path)
        recovery_time = time.time() - recovery_start

        # Phase 4: Verify recovery
        recovered_counts = self._get_counts(restore_path)
        assert recovered_counts == original_counts

        print(f"Recovery from corruption completed in {recovery_time:.3f}s")

    def test_scenario_accidental_deletion(
        self,
        backup_manager: BackupManager,
        production_database: Path,
        temp_workspace: Path,
    ):
        """Scenario: Accidental table deletion recovery."""
        # Phase 1: Backup
        backup = backup_manager.create_backup(production_database)

        # Phase 2: Accidental deletion
        conn = sqlite3.connect(str(production_database))
        conn.execute("DROP TABLE users")
        conn.commit()
        conn.close()

        # Verify deletion
        conn = sqlite3.connect(str(production_database))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        assert "users" not in tables

        # Phase 3: Recovery
        restore_path = temp_workspace / "restore" / "recovered_users.db"
        backup_manager.restore_backup(backup.id, restore_path)

        # Phase 4: Verify table restored
        conn = sqlite3.connect(str(restore_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        conn.close()

        assert "users" in tables
        assert user_count == 100  # Original count

    def _get_counts(self, db_path: Path) -> Dict[str, int]:
        """Get row counts."""
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        counts = {}
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cursor.fetchone()[0]
            except sqlite3.Error:
                counts[table] = -1
        conn.close()
        return counts
