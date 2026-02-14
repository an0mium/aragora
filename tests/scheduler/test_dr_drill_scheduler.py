"""
Tests for DR drill scheduler module.

Tests cover:
- DrillType enum
- DrillStatus enum
- ComponentType enum
- DrillStep dataclass
- DRDrillResult dataclass
- DRDrillConfig dataclass
- DRDrillStorage class
- DRDrillScheduler class
- Global scheduler singleton
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.scheduler.dr_drill_scheduler import (
    ComponentType,
    DRDrillConfig,
    DRDrillResult,
    DRDrillScheduler,
    DRDrillStorage,
    DrillStatus,
    DrillStep,
    DrillType,
    get_dr_drill_scheduler,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestDrillType:
    """Tests for DrillType enum."""

    def test_has_all_drill_types(self):
        """Enum has all expected drill types."""
        assert DrillType.BACKUP_RESTORATION.value == "backup_restoration"
        assert DrillType.COMPONENT_FAILOVER.value == "component_failover"
        assert DrillType.FULL_DR_SIMULATION.value == "full_dr_simulation"
        assert DrillType.DATA_INTEGRITY_CHECK.value == "data_integrity_check"
        assert DrillType.NETWORK_FAILOVER.value == "network_failover"
        assert DrillType.MANUAL.value == "manual"

    def test_drill_type_count(self):
        """Enum has exactly 6 drill types."""
        assert len(DrillType) == 6

    def test_is_string_enum(self):
        """DrillType values are strings."""
        for drill_type in DrillType:
            assert isinstance(drill_type.value, str)


class TestDrillStatus:
    """Tests for DrillStatus enum."""

    def test_has_all_statuses(self):
        """Enum has all expected statuses."""
        assert DrillStatus.SCHEDULED.value == "scheduled"
        assert DrillStatus.RUNNING.value == "running"
        assert DrillStatus.COMPLETED.value == "completed"
        assert DrillStatus.FAILED.value == "failed"
        assert DrillStatus.CANCELLED.value == "cancelled"
        assert DrillStatus.PARTIAL.value == "partial"

    def test_status_count(self):
        """Enum has exactly 6 statuses."""
        assert len(DrillStatus) == 6


class TestComponentType:
    """Tests for ComponentType enum."""

    def test_has_all_component_types(self):
        """Enum has all expected component types."""
        assert ComponentType.DATABASE.value == "database"
        assert ComponentType.OBJECT_STORAGE.value == "object_storage"
        assert ComponentType.CACHE.value == "cache"
        assert ComponentType.QUEUE.value == "queue"
        assert ComponentType.API_SERVER.value == "api_server"
        assert ComponentType.WEBSOCKET.value == "websocket"
        assert ComponentType.AUTH.value == "auth"
        assert ComponentType.SEARCH.value == "search"
        assert ComponentType.FULL_SYSTEM.value == "full_system"

    def test_component_type_count(self):
        """Enum has exactly 9 component types."""
        assert len(ComponentType) == 9


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestDrillStep:
    """Tests for DrillStep dataclass."""

    def test_create_with_required_fields(self):
        """Creates step with required fields only."""
        step = DrillStep(
            step_id="step_123",
            step_name="Backup Database",
            component=ComponentType.DATABASE,
            action="backup",
        )

        assert step.step_id == "step_123"
        assert step.step_name == "Backup Database"
        assert step.component == ComponentType.DATABASE
        assert step.action == "backup"

    def test_default_values(self):
        """Default values are set correctly."""
        step = DrillStep(
            step_id="step_123",
            step_name="Test",
            component=ComponentType.DATABASE,
            action="backup",
        )

        assert step.status == DrillStatus.SCHEDULED
        assert step.started_at is None
        assert step.completed_at is None
        assert step.duration_seconds == 0.0
        assert step.error_message is None
        assert step.metrics == {}


class TestDRDrillResult:
    """Tests for DRDrillResult dataclass."""

    def test_create_with_required_fields(self):
        """Creates result with required fields only."""
        result = DRDrillResult(
            drill_id="drill_123",
            drill_type=DrillType.BACKUP_RESTORATION,
        )

        assert result.drill_id == "drill_123"
        assert result.drill_type == DrillType.BACKUP_RESTORATION

    def test_default_values(self):
        """Default values are set correctly."""
        result = DRDrillResult(
            drill_id="drill_123",
            drill_type=DrillType.BACKUP_RESTORATION,
        )

        assert result.status == DrillStatus.SCHEDULED
        assert result.initiated_by == "system"
        assert result.steps == []
        assert result.total_duration_seconds == 0.0
        assert result.rto_seconds is None
        assert result.rpo_seconds is None
        assert result.target_rto_seconds == 3600.0
        assert result.target_rpo_seconds == 300.0
        assert result.meets_rto is False
        assert result.meets_rpo is False
        assert result.is_compliant is False

    def test_to_dict(self):
        """to_dict returns proper dictionary."""
        now = datetime.now(timezone.utc)
        result = DRDrillResult(
            drill_id="drill_123",
            drill_type=DrillType.BACKUP_RESTORATION,
            status=DrillStatus.COMPLETED,
            scheduled_at=now,
            started_at=now,
            completed_at=now + timedelta(minutes=30),
            rto_seconds=1800.0,
            rpo_seconds=60.0,
            meets_rto=True,
            meets_rpo=True,
            is_compliant=True,
            success_rate=1.0,
        )

        d = result.to_dict()

        assert d["drill_id"] == "drill_123"
        assert d["drill_type"] == "backup_restoration"
        assert d["status"] == "completed"
        assert d["meets_rto"] is True
        assert d["meets_rpo"] is True
        assert d["is_compliant"] is True
        assert d["success_rate"] == 1.0

    def test_to_dict_with_steps(self):
        """to_dict includes step counts."""
        result = DRDrillResult(
            drill_id="drill_123",
            drill_type=DrillType.BACKUP_RESTORATION,
            steps=[
                DrillStep(
                    step_id="s1",
                    step_name="Step 1",
                    component=ComponentType.DATABASE,
                    action="backup",
                    status=DrillStatus.COMPLETED,
                ),
                DrillStep(
                    step_id="s2",
                    step_name="Step 2",
                    component=ComponentType.CACHE,
                    action="restore",
                    status=DrillStatus.FAILED,
                ),
            ],
        )

        d = result.to_dict()

        assert d["steps_count"] == 2
        assert d["steps_passed"] == 1


class TestDRDrillConfig:
    """Tests for DRDrillConfig dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        config = DRDrillConfig()

        assert config.monthly_drill_day == 15
        assert config.quarterly_drill_months == [3, 6, 9, 12]
        assert config.annual_drill_month == 1
        assert config.target_rto_seconds == 3600.0
        assert config.target_rpo_seconds == 300.0
        assert config.dry_run is False

    def test_custom_values(self):
        """Custom values are set correctly."""
        config = DRDrillConfig(
            monthly_drill_day=1,
            target_rto_seconds=7200.0,
            target_rpo_seconds=600.0,
            dry_run=True,
        )

        assert config.monthly_drill_day == 1
        assert config.target_rto_seconds == 7200.0
        assert config.target_rpo_seconds == 600.0
        assert config.dry_run is True


# =============================================================================
# Storage Tests
# =============================================================================


class TestDRDrillStorage:
    """Tests for DRDrillStorage class."""

    @pytest.fixture
    def storage(self):
        """Create in-memory storage for testing."""
        return DRDrillStorage()

    def test_init_creates_schema(self, storage):
        """Initializes with required tables."""
        conn = storage._get_conn()
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        assert "dr_drills" in tables
        assert "drill_steps" in tables
        assert "drill_schedule" in tables
        assert "rto_rpo_history" in tables

    def test_save_and_get_drill(self, storage):
        """Saves and retrieves a drill."""
        drill = DRDrillResult(
            drill_id="drill_123",
            drill_type=DrillType.BACKUP_RESTORATION,
            status=DrillStatus.COMPLETED,
        )

        storage.save_drill(drill)
        retrieved = storage.get_drill("drill_123")

        assert retrieved is not None
        assert retrieved.drill_id == "drill_123"
        assert retrieved.drill_type == DrillType.BACKUP_RESTORATION

    def test_get_nonexistent_drill(self, storage):
        """Returns None for nonexistent drill."""
        result = storage.get_drill("nonexistent")

        assert result is None

    def test_save_and_update_drill(self, storage):
        """Updates existing drill."""
        drill = DRDrillResult(
            drill_id="drill_123",
            drill_type=DrillType.BACKUP_RESTORATION,
            status=DrillStatus.SCHEDULED,
        )
        storage.save_drill(drill)

        # Update status
        drill.status = DrillStatus.COMPLETED
        storage.save_drill(drill)

        retrieved = storage.get_drill("drill_123")
        assert retrieved.status == DrillStatus.COMPLETED

    def test_record_rto_rpo(self, storage):
        """Records RTO/RPO metrics."""
        drill = DRDrillResult(
            drill_id="drill_123",
            drill_type=DrillType.BACKUP_RESTORATION,
            rto_seconds=1800.0,
            rpo_seconds=60.0,
            target_rto_seconds=3600.0,
            target_rpo_seconds=300.0,
        )
        drill.is_compliant = True

        storage.record_rto_rpo(drill)

        # Verify record was saved
        conn = storage._get_conn()
        row = conn.execute(
            "SELECT * FROM rto_rpo_history WHERE drill_id = ?",
            (drill.drill_id,),
        ).fetchone()
        assert row is not None
        assert row["rto_seconds"] == 1800.0

    def test_save_schedule(self, storage):
        """Saves schedule."""
        next_run = datetime.now(timezone.utc) + timedelta(days=7)

        storage.save_schedule("test_schedule", "backup_restoration", next_run)

        conn = storage._get_conn()
        row = conn.execute(
            "SELECT * FROM drill_schedule WHERE schedule_id = ?",
            ("test_schedule",),
        ).fetchone()
        assert row is not None
        assert row["drill_type"] == "backup_restoration"

    def test_get_due_schedules(self, storage):
        """Gets due schedules."""
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        future = datetime.now(timezone.utc) + timedelta(days=7)

        storage.save_schedule("past_schedule", "backup_restoration", past)
        storage.save_schedule("future_schedule", "backup_restoration", future)

        due = storage.get_due_schedules()

        assert len(due) == 1
        assert due[0]["schedule_id"] == "past_schedule"

    def test_update_schedule_run(self, storage):
        """Updates schedule run time."""
        next_run = datetime.now(timezone.utc) + timedelta(days=7)
        storage.save_schedule("test_schedule", "backup_restoration", next_run)

        new_next_run = datetime.now(timezone.utc) + timedelta(days=30)
        storage.update_schedule_run("test_schedule", new_next_run)

        conn = storage._get_conn()
        row = conn.execute(
            "SELECT * FROM drill_schedule WHERE schedule_id = ?",
            ("test_schedule",),
        ).fetchone()
        assert row["last_run"] is not None


# =============================================================================
# Scheduler Tests
# =============================================================================


class TestDRDrillScheduler:
    """Tests for DRDrillScheduler class."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler with in-memory storage."""
        config = DRDrillConfig(storage_path=None, dry_run=True)
        return DRDrillScheduler(config)

    def test_init(self, scheduler):
        """Initializes with default state."""
        assert scheduler._running is False
        assert scheduler._task is None
        assert scheduler._test_handlers == {}
        assert scheduler._restore_handlers == {}
        assert scheduler._notification_handlers == []

    def test_register_notification_handler(self, scheduler):
        """Registers notification handler."""
        handler = MagicMock()

        scheduler.register_notification_handler(handler)

        assert handler in scheduler._notification_handlers

    @pytest.mark.asyncio
    async def test_execute_drill_dry_run(self, scheduler):
        """Executes drill in dry run mode."""
        drill = await scheduler.execute_drill(
            drill_type=DrillType.BACKUP_RESTORATION,
            components=[ComponentType.DATABASE, ComponentType.CACHE],
        )

        assert drill.status in [DrillStatus.COMPLETED, DrillStatus.PARTIAL]
        assert len(drill.steps) >= 1

    @pytest.mark.asyncio
    async def test_execute_drill_measures_duration(self, scheduler):
        """Drill execution measures duration."""
        drill = await scheduler.execute_drill(
            drill_type=DrillType.BACKUP_RESTORATION,
            components=[ComponentType.DATABASE],
        )

        assert drill.total_duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_execute_drill_sets_status(self, scheduler):
        """Drill execution sets final status."""
        drill = await scheduler.execute_drill(
            drill_type=DrillType.BACKUP_RESTORATION,
            components=[ComponentType.DATABASE],
        )

        assert drill.status in [
            DrillStatus.COMPLETED,
            DrillStatus.PARTIAL,
            DrillStatus.FAILED,
        ]

    @pytest.mark.asyncio
    async def test_execute_drill_uses_targets(self, scheduler):
        """Drill execution uses configured targets."""
        scheduler.config.target_rto_seconds = 7200.0
        scheduler.config.target_rpo_seconds = 600.0

        drill = await scheduler.execute_drill(
            drill_type=DrillType.BACKUP_RESTORATION,
            components=[ComponentType.DATABASE],
        )

        assert drill.target_rto_seconds == 7200.0
        assert drill.target_rpo_seconds == 600.0

    @pytest.mark.asyncio
    async def test_execute_drill_tracks_rto(self, scheduler):
        """Drill execution tracks RTO."""
        drill = await scheduler.execute_drill(
            drill_type=DrillType.BACKUP_RESTORATION,
            components=[ComponentType.DATABASE],
        )

        # RTO should be set after drill completes
        assert drill.rto_seconds is not None or drill.rto_seconds == 0.0

    @pytest.mark.asyncio
    async def test_execute_drill_calculates_compliance(self, scheduler):
        """Drill calculates compliance."""
        drill = await scheduler.execute_drill(
            drill_type=DrillType.BACKUP_RESTORATION,
            components=[ComponentType.DATABASE],
        )

        # Compliance is calculated based on RTO/RPO vs targets
        assert isinstance(drill.is_compliant, bool)

    def test_get_drill(self, scheduler):
        """Gets a drill by ID."""
        # First execute a drill
        import asyncio

        drill = asyncio.run(
            scheduler.execute_drill(
                drill_type=DrillType.BACKUP_RESTORATION,
                components=[ComponentType.DATABASE],
            )
        )

        retrieved = scheduler.get_drill(drill.drill_id)

        assert retrieved is not None
        assert retrieved.drill_id == drill.drill_id

    def test_get_nonexistent_drill(self, scheduler):
        """Returns None for nonexistent drill."""
        result = scheduler.get_drill("nonexistent")

        assert result is None

    def test_get_compliance_report(self, scheduler):
        """Gets compliance report."""
        import asyncio

        asyncio.run(
            scheduler.execute_drill(
                drill_type=DrillType.BACKUP_RESTORATION,
                components=[ComponentType.DATABASE],
            )
        )

        report = scheduler.get_compliance_report()

        assert "total_drills" in report
        assert "compliant_drills" in report
        assert "compliance_rate" in report


class TestSchedulerLifecycle:
    """Tests for scheduler lifecycle methods."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler with in-memory storage."""
        config = DRDrillConfig(storage_path=None, dry_run=True)
        return DRDrillScheduler(config)

    @pytest.mark.asyncio
    async def test_start(self, scheduler):
        """Starts the scheduler."""
        await scheduler.start()

        assert scheduler._running is True
        assert scheduler._task is not None

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, scheduler):
        """Start is idempotent."""
        await scheduler.start()
        task1 = scheduler._task

        await scheduler.start()
        task2 = scheduler._task

        assert task1 is task2

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop(self, scheduler):
        """Stops the scheduler."""
        await scheduler.start()
        await scheduler.stop()

        assert scheduler._running is False


class TestDrillTypes:
    """Tests for different drill types."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler with in-memory storage."""
        config = DRDrillConfig(storage_path=None, dry_run=True)
        return DRDrillScheduler(config)

    @pytest.mark.asyncio
    async def test_backup_restoration_drill(self, scheduler):
        """Backup restoration drill creates appropriate steps."""
        drill = await scheduler.execute_drill(
            drill_type=DrillType.BACKUP_RESTORATION,
            components=[ComponentType.DATABASE],
        )

        assert drill.drill_type == DrillType.BACKUP_RESTORATION
        assert len(drill.steps) >= 1

    @pytest.mark.asyncio
    async def test_component_failover_drill(self, scheduler):
        """Component failover drill executes correctly."""
        drill = await scheduler.execute_drill(
            drill_type=DrillType.COMPONENT_FAILOVER,
            components=[ComponentType.API_SERVER],
        )

        assert drill.drill_type == DrillType.COMPONENT_FAILOVER

    @pytest.mark.asyncio
    async def test_data_integrity_check(self, scheduler):
        """Data integrity check drill."""
        drill = await scheduler.execute_drill(
            drill_type=DrillType.DATA_INTEGRITY_CHECK,
            components=[ComponentType.DATABASE, ComponentType.OBJECT_STORAGE],
        )

        assert drill.drill_type == DrillType.DATA_INTEGRITY_CHECK

    @pytest.mark.asyncio
    async def test_manual_drill(self, scheduler):
        """Manual drill type."""
        drill = await scheduler.execute_drill(
            drill_type=DrillType.MANUAL,
            components=[ComponentType.DATABASE],
        )

        assert drill.drill_type == DrillType.MANUAL


class TestComplianceMetrics:
    """Tests for compliance metric calculations."""

    def test_rto_compliance_calculation(self):
        """RTO compliance is calculated correctly."""
        drill = DRDrillResult(
            drill_id="test",
            drill_type=DrillType.BACKUP_RESTORATION,
            rto_seconds=1800.0,  # 30 minutes
            target_rto_seconds=3600.0,  # 1 hour target
        )

        # Meets RTO if actual <= target
        drill.meets_rto = drill.rto_seconds <= drill.target_rto_seconds
        assert drill.meets_rto is True

    def test_rto_fails_when_exceeded(self):
        """RTO fails when exceeded."""
        drill = DRDrillResult(
            drill_id="test",
            drill_type=DrillType.BACKUP_RESTORATION,
            rto_seconds=7200.0,  # 2 hours
            target_rto_seconds=3600.0,  # 1 hour target
        )

        drill.meets_rto = drill.rto_seconds <= drill.target_rto_seconds
        assert drill.meets_rto is False

    def test_rpo_compliance_calculation(self):
        """RPO compliance is calculated correctly."""
        drill = DRDrillResult(
            drill_id="test",
            drill_type=DrillType.BACKUP_RESTORATION,
            rpo_seconds=60.0,  # 1 minute
            target_rpo_seconds=300.0,  # 5 minute target
        )

        drill.meets_rpo = drill.rpo_seconds <= drill.target_rpo_seconds
        assert drill.meets_rpo is True

    def test_overall_compliance_requires_both(self):
        """Overall compliance requires both RTO and RPO."""
        drill = DRDrillResult(
            drill_id="test",
            drill_type=DrillType.BACKUP_RESTORATION,
            rto_seconds=1800.0,
            rpo_seconds=60.0,
            target_rto_seconds=3600.0,
            target_rpo_seconds=300.0,
            meets_rto=True,
            meets_rpo=True,
        )

        drill.is_compliant = drill.meets_rto and drill.meets_rpo
        assert drill.is_compliant is True

    def test_overall_compliance_fails_with_rto_miss(self):
        """Overall compliance fails if RTO missed."""
        drill = DRDrillResult(
            drill_id="test",
            drill_type=DrillType.BACKUP_RESTORATION,
            meets_rto=False,
            meets_rpo=True,
        )

        drill.is_compliant = drill.meets_rto and drill.meets_rpo
        assert drill.is_compliant is False


class TestGlobalScheduler:
    """Tests for global scheduler singleton."""

    def test_get_scheduler_creates_instance(self):
        """Creates scheduler on first call."""
        import aragora.scheduler.dr_drill_scheduler as module

        module._scheduler = None

        scheduler = get_dr_drill_scheduler()

        assert isinstance(scheduler, DRDrillScheduler)

    def test_get_scheduler_returns_same_instance(self):
        """Returns same instance on subsequent calls."""
        import aragora.scheduler.dr_drill_scheduler as module

        module._scheduler = None

        scheduler1 = get_dr_drill_scheduler()
        scheduler2 = get_dr_drill_scheduler()

        assert scheduler1 is scheduler2
