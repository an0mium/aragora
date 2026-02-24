"""
Tests for Offsite Backup Automation and Restore Drill Evidence.

Tests cover:
- BackupManager.restore_drill() method
- RestoreDrillReport data class serialization
- Drill history persistence and retrieval
- Backup status reporting
- Scheduled offsite backup integration
- BackupOffsiteHandler HTTP endpoints
- Error handling for edge cases

SOC 2 Compliance: CC9.1, CC9.2 (Business Continuity)
"""

import asyncio
import gzip
import json
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.backup.manager import (
    BackupManager,
    BackupMetadata,
    BackupStatus,
    BackupType,
    RestoreDrillReport,
    RetentionPolicy,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory(prefix="offsite_auto_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def backup_dir(temp_dir):
    """Create a backup directory."""
    d = temp_dir / "backups"
    d.mkdir()
    return d


@pytest.fixture
def sample_db(temp_dir) -> Path:
    """Create a sample SQLite database for testing."""
    db_path = temp_dir / "test.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
    cursor.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT)")
    for i in range(10):
        cursor.execute("INSERT INTO users (name) VALUES (?)", (f"User {i}",))
    cursor.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?)",
        ("version", "1.0"),
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def manager(backup_dir) -> BackupManager:
    """Create a BackupManager instance."""
    return BackupManager(
        backup_dir=backup_dir,
        compression=True,
        verify_after_backup=True,
        metrics_enabled=False,
    )


@pytest.fixture
def manager_with_backup(manager, sample_db):
    """Create a BackupManager with an existing verified backup."""
    manager.create_backup(sample_db)
    return manager


# =============================================================================
# RestoreDrillReport Tests
# =============================================================================


class TestRestoreDrillReport:
    """Tests for RestoreDrillReport data class."""

    def test_to_dict_basic(self):
        now = datetime.now(timezone.utc)
        report = RestoreDrillReport(
            drill_id="drill-001",
            backup_id="bk-001",
            started_at=now,
            completed_at=now,
            duration_seconds=1.5,
            status="passed",
            tables_verified=3,
            rows_verified=100,
            checksum_valid=True,
            schema_valid=True,
            integrity_valid=True,
        )
        d = report.to_dict()
        assert d["drill_id"] == "drill-001"
        assert d["backup_id"] == "bk-001"
        assert d["status"] == "passed"
        assert d["tables_verified"] == 3
        assert d["rows_verified"] == 100
        assert d["checksum_valid"] is True
        assert d["schema_valid"] is True
        assert d["integrity_valid"] is True
        assert d["errors"] == []

    def test_to_dict_with_errors(self):
        now = datetime.now(timezone.utc)
        report = RestoreDrillReport(
            drill_id="drill-002",
            backup_id="bk-002",
            started_at=now,
            status="failed",
            errors=["Checksum mismatch", "Schema changed"],
        )
        d = report.to_dict()
        assert d["status"] == "failed"
        assert len(d["errors"]) == 2
        assert d["completed_at"] is None

    def test_from_dict_round_trip(self):
        now = datetime.now(timezone.utc)
        original = RestoreDrillReport(
            drill_id="drill-003",
            backup_id="bk-003",
            started_at=now,
            completed_at=now,
            duration_seconds=2.5,
            status="passed",
            tables_verified=5,
            rows_verified=500,
            checksum_valid=True,
            schema_valid=True,
            integrity_valid=True,
            metadata={"rto_seconds": 2.5},
        )
        d = original.to_dict()
        restored = RestoreDrillReport.from_dict(d)

        assert restored.drill_id == original.drill_id
        assert restored.backup_id == original.backup_id
        assert restored.status == original.status
        assert restored.tables_verified == original.tables_verified
        assert restored.rows_verified == original.rows_verified
        assert restored.checksum_valid == original.checksum_valid
        assert restored.metadata == original.metadata

    def test_from_dict_minimal(self):
        """Test from_dict with minimal required fields."""
        data = {
            "drill_id": "d-min",
            "backup_id": "b-min",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        report = RestoreDrillReport.from_dict(data)
        assert report.drill_id == "d-min"
        assert report.status == "pending"
        assert report.errors == []

    def test_default_values(self):
        now = datetime.now(timezone.utc)
        report = RestoreDrillReport(
            drill_id="d-default",
            backup_id="b-default",
            started_at=now,
        )
        assert report.status == "pending"
        assert report.tables_verified == 0
        assert report.rows_verified == 0
        assert report.checksum_valid is False
        assert report.schema_valid is False
        assert report.integrity_valid is False
        assert report.errors == []
        assert report.metadata == {}


# =============================================================================
# BackupManager.restore_drill() Tests
# =============================================================================


class TestRestoreDrill:
    """Tests for BackupManager.restore_drill() method."""

    def test_restore_drill_success(self, manager_with_backup):
        """Test successful restore drill with verified backup."""
        report = manager_with_backup.restore_drill()

        assert report.status == "passed"
        assert report.drill_id is not None
        assert report.backup_id is not None
        assert report.started_at is not None
        assert report.completed_at is not None
        assert report.duration_seconds > 0
        assert report.tables_verified == 2  # users + settings
        assert report.rows_verified == 11  # 10 users + 1 setting
        assert report.checksum_valid is True
        assert report.schema_valid is True
        assert report.integrity_valid is True
        assert report.errors == []

    def test_restore_drill_with_specific_backup_id(self, manager, sample_db):
        """Test restore drill targeting a specific backup."""
        backup = manager.create_backup(sample_db)
        report = manager.restore_drill(backup_id=backup.id)

        assert report.status == "passed"
        assert report.backup_id == backup.id

    def test_restore_drill_no_backups(self, manager):
        """Test restore drill when no backups exist."""
        report = manager.restore_drill()

        assert report.status == "failed"
        assert report.backup_id == "none"
        assert "No verified backups available" in report.errors[0]

    def test_restore_drill_backup_not_found(self, manager):
        """Test restore drill with non-existent backup ID."""
        report = manager.restore_drill(backup_id="nonexistent")

        assert report.status == "failed"
        assert "Backup not found" in report.errors[0]

    def test_restore_drill_missing_backup_file(self, manager_with_backup):
        """Test restore drill when backup file has been deleted."""
        latest = manager_with_backup.get_latest_backup()
        # Delete the backup file
        Path(latest.backup_path).unlink()

        report = manager_with_backup.restore_drill(backup_id=latest.id)

        assert report.status == "failed"
        assert any("not found" in e for e in report.errors)

    def test_restore_drill_corrupted_backup(self, manager, sample_db):
        """Test restore drill detects corrupted backup (checksum mismatch)."""
        backup = manager.create_backup(sample_db)

        # Corrupt the backup file by appending garbage
        backup_path = Path(backup.backup_path)
        with open(backup_path, "ab") as f:
            f.write(b"CORRUPTED_DATA")

        report = manager.restore_drill(backup_id=backup.id)

        assert report.status == "failed"
        assert report.checksum_valid is False
        assert any("Checksum mismatch" in e for e in report.errors)

    def test_restore_drill_records_in_history(self, manager_with_backup):
        """Test that drill results are stored in history."""
        manager_with_backup.restore_drill()
        manager_with_backup.restore_drill()

        history = manager_with_backup.get_drill_history()
        assert len(history) == 2
        # Most recent first
        assert history[0].started_at >= history[1].started_at

    def test_restore_drill_history_limit(self, manager_with_backup):
        """Test drill history respects limit parameter."""
        for _ in range(5):
            manager_with_backup.restore_drill()

        assert len(manager_with_backup.get_drill_history(limit=3)) == 3
        assert len(manager_with_backup.get_drill_history(limit=10)) == 5

    def test_restore_drill_multiple_backups(self, manager, sample_db):
        """Test drill with multiple backups uses latest when no ID given."""
        backup1 = manager.create_backup(sample_db)
        time.sleep(0.01)
        backup2 = manager.create_backup(sample_db)

        report = manager.restore_drill()

        # Should use the latest verified backup
        assert report.backup_id == backup2.id
        assert report.status == "passed"

    def test_restore_drill_uncompressed_backup(self, temp_dir, sample_db):
        """Test drill with uncompressed backup."""
        backup_dir = temp_dir / "backups_uncomp"
        backup_dir.mkdir()
        manager = BackupManager(
            backup_dir=backup_dir,
            compression=False,
            metrics_enabled=False,
        )
        manager.create_backup(sample_db)

        report = manager.restore_drill()
        assert report.status == "passed"
        assert report.tables_verified == 2


# =============================================================================
# Drill History Persistence Tests
# =============================================================================


class TestDrillHistoryPersistence:
    """Tests for drill history persistence across manager restarts."""

    def test_drill_history_persisted_to_disk(self, backup_dir, sample_db):
        """Test that drill history survives manager restarts."""
        mgr1 = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        mgr1.create_backup(sample_db)
        mgr1.restore_drill()

        # Create a new manager instance to test persistence
        mgr2 = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        history = mgr2.get_drill_history()

        assert len(history) == 1
        assert history[0].status == "passed"

    def test_corrupted_drill_history_handled(self, backup_dir):
        """Test graceful handling of corrupted drill history file."""
        (backup_dir / "drill_history.json").write_text("NOT VALID JSON")

        # Should not raise
        mgr = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        assert mgr.get_drill_history() == []

    def test_drill_history_file_created(self, backup_dir, sample_db):
        """Test that drill history file is created on first drill."""
        mgr = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        mgr.create_backup(sample_db)
        mgr.restore_drill()

        drills_path = backup_dir / "drill_history.json"
        assert drills_path.exists()

        data = json.loads(drills_path.read_text())
        assert len(data) == 1
        assert data[0]["status"] == "passed"

    def test_failed_drills_also_persisted(self, backup_dir):
        """Test that failed drills are saved to history."""
        mgr = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        mgr.restore_drill()  # No backups -> should fail

        history = mgr.get_drill_history()
        assert len(history) == 1
        assert history[0].status == "failed"

        # Verify persisted
        mgr2 = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        assert len(mgr2.get_drill_history()) == 1


# =============================================================================
# BackupManager.get_backup_status() Tests
# =============================================================================


class TestGetBackupStatus:
    """Tests for BackupManager.get_backup_status() method."""

    def test_status_empty(self, manager):
        """Test status when no backups exist."""
        status = manager.get_backup_status()

        assert status["total_backups"] == 0
        assert status["verified_backups"] == 0
        assert status["failed_backups"] == 0
        assert status["total_size_bytes"] == 0
        assert status["last_successful_backup"] is None
        assert status["last_drill"] is None
        assert "retention_policy" in status

    def test_status_with_backup(self, manager_with_backup):
        """Test status after creating a backup."""
        status = manager_with_backup.get_backup_status()

        assert status["total_backups"] == 1
        assert status["verified_backups"] == 1
        assert status["last_successful_backup"] is not None
        assert status["last_successful_backup"]["status"] == "verified"

    def test_status_with_drill(self, manager_with_backup):
        """Test status includes latest drill result."""
        manager_with_backup.restore_drill()
        status = manager_with_backup.get_backup_status()

        assert status["last_drill"] is not None
        assert status["last_drill"]["status"] == "passed"

    def test_status_retention_policy(self, manager):
        """Test status includes retention policy details."""
        status = manager.get_backup_status()
        rp = status["retention_policy"]

        assert rp["keep_daily"] == 7
        assert rp["keep_weekly"] == 4
        assert rp["keep_monthly"] == 3
        assert rp["min_backups"] == 1

    def test_status_multiple_backups(self, manager, sample_db):
        """Test status with multiple backups."""
        manager.create_backup(sample_db)
        manager.create_backup(sample_db)
        manager.create_backup(sample_db)

        status = manager.get_backup_status()
        assert status["total_backups"] == 3
        assert status["total_size_bytes"] > 0


# =============================================================================
# BackupOffsiteHandler Tests
# =============================================================================


class TestBackupOffsiteHandler:
    """Tests for BackupOffsiteHandler HTTP endpoint handler."""

    @pytest.fixture
    def handler_cls(self):
        """Import and return the handler class."""
        from aragora.server.handlers.backup_offsite_handler import (
            BackupOffsiteHandler,
        )

        return BackupOffsiteHandler

    @pytest.fixture
    def mock_manager(self, manager_with_backup):
        """Create a mock-ready manager with a backup."""
        return manager_with_backup

    @pytest.fixture
    def handler(self, handler_cls, mock_manager):
        """Create a handler instance with injected manager."""
        h = handler_cls({})
        h._manager = mock_manager
        return h

    def test_can_handle_status(self, handler):
        assert handler.can_handle("/api/v1/backup/status", "GET") is True

    def test_can_handle_drills(self, handler):
        assert handler.can_handle("/api/v1/backup/drills", "GET") is True

    def test_can_handle_drill_post(self, handler):
        assert handler.can_handle("/api/v1/backup/drill", "POST") is True

    def test_cannot_handle_unknown(self, handler):
        assert handler.can_handle("/api/v1/backup/unknown", "GET") is False
        assert handler.can_handle("/api/v2/backups", "GET") is False
        assert handler.can_handle("/api/v1/backup/status", "POST") is False

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_get_status_endpoint(self, handler):
        """Test GET /api/v1/backup/status returns correct format."""
        result = await handler._get_status()
        body = result[0]

        # Response should have {"data": {...}} envelope
        assert "data" in body
        data = body["data"]
        assert "total_backups" in data
        assert "last_successful_backup" in data
        assert "retention_policy" in data

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_list_drills_empty(self, handler_cls, backup_dir):
        """Test GET /api/v1/backup/drills with no drills."""
        mgr = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        h = handler_cls({})
        h._manager = mgr

        result = await h._list_drills({})
        body = result[0]

        assert "data" in body
        assert body["data"]["drills"] == []
        assert body["data"]["total"] == 0

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_list_drills_with_results(self, handler):
        """Test GET /api/v1/backup/drills after running drills."""
        handler._manager.restore_drill()
        handler._manager.restore_drill()

        result = await handler._list_drills({})
        body = result[0]

        assert body["data"]["total"] == 2
        drills = body["data"]["drills"]
        assert len(drills) == 2
        assert drills[0]["status"] == "passed"

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_list_drills_with_limit(self, handler):
        """Test GET /api/v1/backup/drills with limit parameter."""
        for _ in range(5):
            handler._manager.restore_drill()

        result = await handler._list_drills({"limit": "3"})
        body = result[0]

        assert body["data"]["total"] == 3

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_trigger_drill_endpoint(self, handler):
        """Test POST /api/v1/backup/drill triggers a restore drill."""
        result = await handler._trigger_drill({})
        body = result[0]

        assert "data" in body
        data = body["data"]
        assert data["status"] == "passed"
        assert data["drill_id"] is not None
        assert data["tables_verified"] == 2
        assert data["rows_verified"] == 11

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_trigger_drill_with_backup_id(self, handler):
        """Test POST /api/v1/backup/drill with specific backup_id."""
        latest = handler._manager.get_latest_backup()
        result = await handler._trigger_drill({"backup_id": latest.id})
        body = result[0]

        assert body["data"]["backup_id"] == latest.id
        assert body["data"]["status"] == "passed"

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_trigger_drill_no_backups(self, handler_cls, backup_dir):
        """Test POST /api/v1/backup/drill when no backups exist."""
        mgr = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        h = handler_cls({})
        h._manager = mgr

        result = await h._trigger_drill({})
        body = result[0]

        assert body["data"]["status"] == "failed"

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_handle_routes_status(self, handler):
        """Test handle() dispatches GET /api/v1/backup/status."""
        mock_handler = MagicMock()
        mock_handler.command = "GET"

        result = await handler.handle("/api/v1/backup/status", {}, mock_handler)
        body = result[0]
        assert "data" in body

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_handle_routes_drills(self, handler):
        """Test handle() dispatches GET /api/v1/backup/drills."""
        mock_handler = MagicMock()
        mock_handler.command = "GET"

        result = await handler.handle("/api/v1/backup/drills", {}, mock_handler)
        body = result[0]
        assert "data" in body

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_handle_unknown_path(self, handler):
        """Test handle() returns 404 for unknown paths."""
        mock_handler = MagicMock()
        mock_handler.command = "GET"

        result = await handler.handle("/api/v1/backup/unknown", {}, mock_handler)
        body = result[0]
        status = result[1]
        assert status == 404


# =============================================================================
# Scheduler Offsite Integration Tests
# =============================================================================


class TestSchedulerOffsiteIntegration:
    """Tests for BackupScheduler offsite backup integration."""

    @pytest.fixture
    def mock_offsite_manager(self):
        """Create a mock offsite manager."""
        mock = MagicMock()
        mock.upload_backup.return_value = MagicMock(id="offsite-001")
        return mock

    def test_scheduler_accepts_offsite_manager(self, manager, mock_offsite_manager):
        """Test that BackupScheduler accepts an offsite_manager parameter."""
        from aragora.backup.scheduler import BackupScheduler, BackupSchedule

        scheduler = BackupScheduler(
            backup_manager=manager,
            schedule=BackupSchedule(enable_offsite=True),
            offsite_manager=mock_offsite_manager,
        )
        assert scheduler._offsite_manager is mock_offsite_manager

    def test_schedule_offsite_flags(self):
        """Test BackupSchedule offsite configuration flags."""
        from aragora.backup.scheduler import BackupSchedule

        schedule = BackupSchedule(
            enable_offsite=True,
            offsite_after_backup=True,
        )
        assert schedule.enable_offsite is True
        assert schedule.offsite_after_backup is True

        # Defaults
        default_schedule = BackupSchedule()
        assert default_schedule.enable_offsite is False
        assert default_schedule.offsite_after_backup is True


# =============================================================================
# Scheduler DR Drill with restore_drill() Tests
# =============================================================================


class TestSchedulerDrillIntegration:
    """Tests for scheduler using BackupManager.restore_drill()."""

    @pytest.mark.asyncio
    async def test_dr_drill_uses_restore_drill_method(self, manager_with_backup):
        """Test that _execute_dr_drill uses restore_drill() when available."""
        from aragora.backup.scheduler import BackupScheduler, BackupSchedule

        scheduler = BackupScheduler(
            backup_manager=manager_with_backup,
            schedule=BackupSchedule(enable_dr_drills=True),
        )

        result = await scheduler._execute_dr_drill()

        assert result["success"] is True
        assert "drill_id" in result
        assert result["steps"][0]["step"] == "restore_drill"
        assert result["steps"][0]["status"] == "passed"

    @pytest.mark.asyncio
    async def test_dr_drill_fallback_without_restore_drill(self):
        """Test fallback when manager lacks restore_drill() method."""
        from aragora.backup.scheduler import BackupScheduler, BackupSchedule

        # Manager without restore_drill
        mock_manager = MagicMock()
        mock_manager.list_backups.return_value = []
        # Remove restore_drill attribute
        del mock_manager.restore_drill

        scheduler = BackupScheduler(
            backup_manager=mock_manager,
            schedule=BackupSchedule(enable_dr_drills=True),
        )

        result = await scheduler._execute_dr_drill()

        assert result["success"] is False
        assert "No backups available" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_dr_drill_increments_stats(self, manager_with_backup):
        """Test that successful drill increments scheduler stats."""
        from aragora.backup.scheduler import BackupScheduler, BackupSchedule

        scheduler = BackupScheduler(
            backup_manager=manager_with_backup,
            schedule=BackupSchedule(enable_dr_drills=True),
        )

        assert scheduler.get_stats().dr_drills_completed == 0

        await scheduler._execute_dr_drill()

        assert scheduler.get_stats().dr_drills_completed == 1
        assert scheduler.get_stats().last_dr_drill_at is not None


# =============================================================================
# End-to-End Integration Tests
# =============================================================================


class TestEndToEndDrillWorkflow:
    """End-to-end tests for the complete backup + drill workflow."""

    def test_full_backup_drill_cycle(self, temp_dir):
        """Test complete: create DB -> backup -> drill -> verify evidence."""
        # Step 1: Create a production-like database
        db_path = temp_dir / "production.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE
            );
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                amount REAL NOT NULL
            );
            CREATE INDEX idx_orders_user ON orders(user_id);
        """)
        for i in range(50):
            cursor.execute(
                "INSERT INTO users (name, email) VALUES (?, ?)",
                (f"User {i}", f"user{i}@test.com"),
            )
        for i in range(200):
            cursor.execute(
                "INSERT INTO orders (user_id, amount) VALUES (?, ?)",
                ((i % 50) + 1, 10.0 + i),
            )
        conn.commit()
        conn.close()

        # Step 2: Create backup
        backup_dir = temp_dir / "backups"
        backup_dir.mkdir()
        mgr = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        backup = mgr.create_backup(db_path)
        assert backup.status == BackupStatus.VERIFIED

        # Step 3: Run restore drill
        report = mgr.restore_drill()
        assert report.status == "passed"
        assert report.tables_verified == 2  # users + orders
        assert report.rows_verified == 250  # 50 + 200
        assert report.checksum_valid is True
        assert report.schema_valid is True
        assert report.integrity_valid is True

        # Step 4: Check status includes drill evidence
        status = mgr.get_backup_status()
        assert status["total_backups"] == 1
        assert status["last_drill"]["status"] == "passed"
        assert status["last_drill"]["tables_verified"] == 2

        # Step 5: Verify drill history persists
        history = mgr.get_drill_history()
        assert len(history) == 1
        assert history[0].drill_id == report.drill_id

    def test_drill_after_data_change_detects_stale_backup(self, temp_dir):
        """Test that drill on old backup still works after DB changes."""
        db_path = temp_dir / "data.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, val TEXT)")
        for i in range(10):
            cursor.execute("INSERT INTO items (val) VALUES (?)", (f"item_{i}",))
        conn.commit()
        conn.close()

        backup_dir = temp_dir / "backups"
        backup_dir.mkdir()
        mgr = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        backup = mgr.create_backup(db_path)

        # Modify database after backup (simulating production changes)
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        for i in range(100):
            cursor.execute("INSERT INTO items (val) VALUES (?)", (f"new_{i}",))
        conn.commit()
        conn.close()

        # Drill on old backup should still pass (verifies backup integrity)
        report = mgr.restore_drill(backup_id=backup.id)
        assert report.status == "passed"
        assert report.rows_verified == 10  # Original count, not 110


# =============================================================================
# Handler Factory Tests
# =============================================================================


class TestHandlerFactory:
    """Tests for handler factory function."""

    def test_create_handler(self):
        from aragora.server.handlers.backup_offsite_handler import (
            create_backup_offsite_handler,
        )

        handler = create_backup_offsite_handler({})
        assert handler is not None
        assert handler.can_handle("/api/v1/backup/status", "GET")
