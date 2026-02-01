"""
Tests for Stuck Detector Module.

Comprehensive tests for:
- Stuck detection logic
- Recovery trigger conditions
- Timeout handling
- False positive prevention
- State tracking
- Recovery action execution
- Reset after recovery
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.nomic.stuck_detector import (
    HealthSummary,
    RecoveryAction,
    StuckDetector,
    StuckDetectorConfig,
    StuckWorkItem,
    WorkAge,
    get_stuck_detector,
    reset_stuck_detector,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_config():
    """Create default stuck detector config."""
    return StuckDetectorConfig(
        green_threshold_minutes=2.0,
        yellow_threshold_minutes=5.0,
        red_threshold_minutes=10.0,
        max_recoveries=3,
        auto_reassign_on_red=True,
        escalate_after_recoveries=2,
        cancel_after_recoveries=3,
        check_interval_seconds=1,  # Fast for tests
    )


@pytest.fixture
def mock_bead_store():
    """Create mock bead store."""
    store = MagicMock()
    store.list_by_status = AsyncMock(return_value=[])
    store.get = AsyncMock(return_value=None)
    store.update = AsyncMock()
    return store


@pytest.fixture
def mock_coordinator():
    """Create mock convoy coordinator."""
    coordinator = MagicMock()
    coordinator._assignments = {}
    coordinator.get_assignment = AsyncMock(return_value=None)
    coordinator.update_assignment_status = AsyncMock()
    return coordinator


@pytest.fixture
def mock_escalation_store():
    """Create mock escalation store."""
    store = MagicMock()
    store.create_chain = AsyncMock()
    return store


@pytest.fixture
async def detector(default_config):
    """Create stuck detector instance."""
    reset_stuck_detector()
    detector = StuckDetector(config=default_config)
    await detector.initialize()
    yield detector
    await detector.stop_monitoring()
    reset_stuck_detector()


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    reset_stuck_detector()
    yield
    reset_stuck_detector()


def _now() -> datetime:
    """Get current time."""
    return datetime.now(timezone.utc)


# ============================================================================
# Test WorkAge Enum
# ============================================================================


class TestWorkAge:
    """Tests for WorkAge enum."""

    def test_work_age_values(self):
        """Should have correct values."""
        assert WorkAge.GREEN.value == "green"
        assert WorkAge.YELLOW.value == "yellow"
        assert WorkAge.RED.value == "red"

    def test_work_age_is_string_enum(self):
        """Should be a string enum."""
        assert isinstance(WorkAge.GREEN, str)
        assert WorkAge.GREEN == "green"


# ============================================================================
# Test RecoveryAction Enum
# ============================================================================


class TestRecoveryAction:
    """Tests for RecoveryAction enum."""

    def test_recovery_action_values(self):
        """Should have correct values."""
        assert RecoveryAction.NONE.value == "none"
        assert RecoveryAction.NOTIFY.value == "notify"
        assert RecoveryAction.ESCALATE.value == "escalate"
        assert RecoveryAction.REASSIGN.value == "reassign"
        assert RecoveryAction.CANCEL.value == "cancel"
        assert RecoveryAction.RETRY.value == "retry"


# ============================================================================
# Test StuckWorkItem
# ============================================================================


class TestStuckWorkItem:
    """Tests for StuckWorkItem dataclass."""

    def test_stuck_work_item_creation(self):
        """Should create work item with defaults."""
        now = _now()
        item = StuckWorkItem(
            id="item-001",
            work_type="bead",
            title="Test work",
            agent_id="agent-001",
            age=WorkAge.GREEN,
            last_update=now,
            time_since_update=timedelta(minutes=1),
        )
        assert item.id == "item-001"
        assert item.work_type == "bead"
        assert item.previous_recoveries == 0
        assert item.recommended_action == RecoveryAction.NONE

    def test_age_minutes(self):
        """Should calculate age in minutes."""
        item = StuckWorkItem(
            id="item-001",
            work_type="bead",
            title="Test",
            agent_id="agent-001",
            age=WorkAge.YELLOW,
            last_update=_now(),
            time_since_update=timedelta(minutes=5, seconds=30),
        )
        assert 5.4 < item.age_minutes < 5.6

    def test_is_stuck_true_when_red(self):
        """Should be stuck when age is RED."""
        item = StuckWorkItem(
            id="item-001",
            work_type="bead",
            title="Test",
            agent_id="agent-001",
            age=WorkAge.RED,
            last_update=_now(),
            time_since_update=timedelta(minutes=15),
        )
        assert item.is_stuck is True

    def test_is_stuck_false_when_green(self):
        """Should not be stuck when age is GREEN."""
        item = StuckWorkItem(
            id="item-001",
            work_type="bead",
            title="Test",
            agent_id="agent-001",
            age=WorkAge.GREEN,
            last_update=_now(),
            time_since_update=timedelta(minutes=1),
        )
        assert item.is_stuck is False

    def test_is_stuck_false_when_yellow(self):
        """Should not be stuck when age is YELLOW."""
        item = StuckWorkItem(
            id="item-001",
            work_type="bead",
            title="Test",
            agent_id="agent-001",
            age=WorkAge.YELLOW,
            last_update=_now(),
            time_since_update=timedelta(minutes=3),
        )
        assert item.is_stuck is False


# ============================================================================
# Test StuckDetectorConfig
# ============================================================================


class TestStuckDetectorConfig:
    """Tests for StuckDetectorConfig dataclass."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = StuckDetectorConfig()
        assert config.green_threshold_minutes == 2.0
        assert config.yellow_threshold_minutes == 5.0
        assert config.red_threshold_minutes == 10.0
        assert config.max_recoveries == 3
        assert config.auto_reassign_on_red is True
        assert config.check_interval_seconds == 30

    def test_custom_config(self):
        """Should accept custom values."""
        config = StuckDetectorConfig(
            green_threshold_minutes=5.0,
            yellow_threshold_minutes=10.0,
            red_threshold_minutes=20.0,
            max_recoveries=5,
        )
        assert config.green_threshold_minutes == 5.0
        assert config.red_threshold_minutes == 20.0
        assert config.max_recoveries == 5


# ============================================================================
# Test HealthSummary
# ============================================================================


class TestHealthSummary:
    """Tests for HealthSummary dataclass."""

    def test_health_summary_creation(self):
        """Should create health summary."""
        summary = HealthSummary(
            total_items=10,
            green_count=7,
            yellow_count=2,
            red_count=1,
            recovered_count=3,
            failed_recoveries=1,
            by_agent={},
            by_convoy={},
        )
        assert summary.total_items == 10
        assert summary.green_count == 7

    def test_health_percentage_all_green(self):
        """Should calculate 100% when all green."""
        summary = HealthSummary(
            total_items=10,
            green_count=10,
            yellow_count=0,
            red_count=0,
            recovered_count=0,
            failed_recoveries=0,
            by_agent={},
            by_convoy={},
        )
        assert summary.health_percentage == 100.0

    def test_health_percentage_mixed(self):
        """Should calculate correct percentage."""
        summary = HealthSummary(
            total_items=10,
            green_count=7,
            yellow_count=2,
            red_count=1,
            recovered_count=0,
            failed_recoveries=0,
            by_agent={},
            by_convoy={},
        )
        assert summary.health_percentage == 70.0

    def test_health_percentage_empty(self):
        """Should return 100% for empty items."""
        summary = HealthSummary(
            total_items=0,
            green_count=0,
            yellow_count=0,
            red_count=0,
            recovered_count=0,
            failed_recoveries=0,
            by_agent={},
            by_convoy={},
        )
        assert summary.health_percentage == 100.0

    def test_stuck_count(self):
        """Should return red count as stuck count."""
        summary = HealthSummary(
            total_items=10,
            green_count=5,
            yellow_count=3,
            red_count=2,
            recovered_count=0,
            failed_recoveries=0,
            by_agent={},
            by_convoy={},
        )
        assert summary.stuck_count == 2


# ============================================================================
# Test StuckDetector Initialization
# ============================================================================


class TestStuckDetectorInit:
    """Tests for StuckDetector initialization."""

    @pytest.mark.asyncio
    async def test_initialization(self, default_config):
        """Should initialize detector."""
        detector = StuckDetector(config=default_config)
        await detector.initialize()

        assert detector._running is False
        assert detector._monitoring_task is None
        assert detector._recovery_counts == {}

    @pytest.mark.asyncio
    async def test_default_config_used(self):
        """Should use default config if not provided."""
        detector = StuckDetector()
        assert detector.config is not None
        assert isinstance(detector.config, StuckDetectorConfig)


# ============================================================================
# Test StuckDetector Monitoring Lifecycle
# ============================================================================


class TestStuckDetectorMonitoring:
    """Tests for monitoring lifecycle."""

    @pytest.mark.asyncio
    async def test_start_monitoring(self, detector):
        """Should start monitoring loop."""
        await detector.start_monitoring()
        assert detector._running is True
        assert detector._monitoring_task is not None

    @pytest.mark.asyncio
    async def test_start_monitoring_idempotent(self, detector):
        """Should not restart if already running."""
        await detector.start_monitoring()
        task1 = detector._monitoring_task

        await detector.start_monitoring()
        task2 = detector._monitoring_task

        assert task1 is task2

    @pytest.mark.asyncio
    async def test_stop_monitoring(self, detector):
        """Should stop monitoring loop."""
        await detector.start_monitoring()
        await detector.stop_monitoring()

        assert detector._running is False
        assert detector._monitoring_task is None

    @pytest.mark.asyncio
    async def test_stop_monitoring_when_not_running(self, detector):
        """Should handle stopping when not running."""
        await detector.stop_monitoring()  # Should not raise
        assert detector._running is False


# ============================================================================
# Test StuckDetector Age Classification
# ============================================================================


class TestStuckDetectorAgeClassification:
    """Tests for age classification."""

    def test_classify_green(self, detector):
        """Should classify recent updates as GREEN."""
        age = detector._classify_age(timedelta(minutes=1))
        assert age == WorkAge.GREEN

    def test_classify_yellow(self, detector):
        """Should classify medium-aged as YELLOW."""
        age = detector._classify_age(timedelta(minutes=3))
        assert age == WorkAge.YELLOW

    def test_classify_red(self, detector):
        """Should classify old updates as RED."""
        age = detector._classify_age(timedelta(minutes=15))
        assert age == WorkAge.RED

    def test_classify_at_boundaries(self, detector):
        """Should handle boundary conditions."""
        # At green threshold (2 min) - should be yellow
        age = detector._classify_age(timedelta(minutes=2))
        assert age == WorkAge.YELLOW

        # At yellow threshold (5 min) - should be red
        age = detector._classify_age(timedelta(minutes=5))
        assert age == WorkAge.RED


# ============================================================================
# Test StuckDetector Action Determination
# ============================================================================


class TestStuckDetectorActionDetermination:
    """Tests for recovery action determination."""

    def test_action_green_is_none(self, detector):
        """Should return NONE for GREEN items."""
        item = StuckWorkItem(
            id="item-001",
            work_type="bead",
            title="Test",
            agent_id="agent-001",
            age=WorkAge.GREEN,
            last_update=_now(),
            time_since_update=timedelta(minutes=1),
        )
        action = detector._determine_action(item)
        assert action == RecoveryAction.NONE

    def test_action_yellow_is_notify(self, detector):
        """Should return NOTIFY for YELLOW items."""
        item = StuckWorkItem(
            id="item-001",
            work_type="bead",
            title="Test",
            agent_id="agent-001",
            age=WorkAge.YELLOW,
            last_update=_now(),
            time_since_update=timedelta(minutes=3),
        )
        action = detector._determine_action(item)
        assert action == RecoveryAction.NOTIFY

    def test_action_red_is_reassign(self, detector):
        """Should return REASSIGN for RED items."""
        item = StuckWorkItem(
            id="item-001",
            work_type="bead",
            title="Test",
            agent_id="agent-001",
            age=WorkAge.RED,
            last_update=_now(),
            time_since_update=timedelta(minutes=15),
        )
        action = detector._determine_action(item)
        assert action == RecoveryAction.REASSIGN

    def test_action_escalate_after_recoveries(self, detector):
        """Should ESCALATE after multiple recoveries."""
        detector._recovery_counts["item-001"] = 2

        item = StuckWorkItem(
            id="item-001",
            work_type="bead",
            title="Test",
            agent_id="agent-001",
            age=WorkAge.RED,
            last_update=_now(),
            time_since_update=timedelta(minutes=15),
            previous_recoveries=2,
        )
        action = detector._determine_action(item)
        assert action == RecoveryAction.ESCALATE

    def test_action_cancel_after_max_recoveries(self, detector):
        """Should CANCEL after max recoveries."""
        detector._recovery_counts["item-001"] = 3

        item = StuckWorkItem(
            id="item-001",
            work_type="bead",
            title="Test",
            agent_id="agent-001",
            age=WorkAge.RED,
            last_update=_now(),
            time_since_update=timedelta(minutes=15),
            previous_recoveries=3,
        )
        action = detector._determine_action(item)
        assert action == RecoveryAction.CANCEL

    def test_action_escalate_when_auto_reassign_disabled(self, default_config):
        """Should ESCALATE when auto_reassign_on_red is False."""
        default_config.auto_reassign_on_red = False
        detector = StuckDetector(config=default_config)

        item = StuckWorkItem(
            id="item-001",
            work_type="bead",
            title="Test",
            agent_id="agent-001",
            age=WorkAge.RED,
            last_update=_now(),
            time_since_update=timedelta(minutes=15),
        )
        action = detector._determine_action(item)
        assert action == RecoveryAction.ESCALATE


# ============================================================================
# Test StuckDetector Detection
# ============================================================================


class TestStuckDetectorDetection:
    """Tests for stuck work detection."""

    @pytest.mark.asyncio
    async def test_detect_stuck_work_empty(self, detector):
        """Should return empty list when no work items."""
        items = await detector.detect_stuck_work()
        assert items == []

    @pytest.mark.asyncio
    async def test_detect_stuck_beads(self, detector, mock_bead_store):
        """Should detect stuck beads."""
        detector.bead_store = mock_bead_store

        # Create a running bead that's stuck
        mock_bead = MagicMock()
        mock_bead.id = "bead-001"
        mock_bead.title = "Test Bead"
        mock_bead.claimed_by = "agent-001"
        mock_bead.updated_at = _now() - timedelta(minutes=15)
        mock_bead.created_at = _now() - timedelta(minutes=20)
        mock_bead.bead_type = MagicMock(value="task")

        mock_bead_store.list_by_status = AsyncMock(return_value=[mock_bead])

        items = await detector.detect_stuck_work()
        assert len(items) == 1
        assert items[0].id == "bead-001"
        assert items[0].is_stuck is True

    @pytest.mark.asyncio
    async def test_detect_stuck_assignments(self, detector, mock_coordinator):
        """Should detect stuck assignments."""
        detector.coordinator = mock_coordinator

        # Create a stuck assignment
        from aragora.nomic.convoy_coordinator import AssignmentStatus

        mock_assignment = MagicMock()
        mock_assignment.id = "assignment-001"
        mock_assignment.bead_id = "bead-001"
        mock_assignment.agent_id = "agent-001"
        mock_assignment.convoy_id = "convoy-001"
        mock_assignment.status = AssignmentStatus.ACTIVE
        mock_assignment.updated_at = _now() - timedelta(minutes=15)
        mock_assignment.assigned_at = _now() - timedelta(minutes=20)
        mock_assignment.estimated_duration_minutes = 30
        mock_assignment.previous_agents = []

        mock_coordinator._assignments = {"assignment-001": mock_assignment}

        items = await detector.detect_stuck_work()
        assert len(items) == 1
        assert items[0].id == "assignment-001"


# ============================================================================
# Test StuckDetector Callbacks
# ============================================================================


class TestStuckDetectorCallbacks:
    """Tests for callback registration and invocation."""

    @pytest.mark.asyncio
    async def test_register_callback(self, detector):
        """Should register callback."""
        callback = MagicMock()
        detector.register_callback(callback)
        assert callback in detector._callbacks

    @pytest.mark.asyncio
    async def test_callback_invoked_on_handling(self, detector):
        """Should invoke callback when handling stuck item."""
        received = []

        def callback(item, action):
            received.append((item, action))

        detector.register_callback(callback)

        item = StuckWorkItem(
            id="item-001",
            work_type="bead",
            title="Test",
            agent_id="agent-001",
            age=WorkAge.YELLOW,
            last_update=_now(),
            time_since_update=timedelta(minutes=3),
            recommended_action=RecoveryAction.NOTIFY,
        )

        await detector._handle_stuck_item(item)
        assert len(received) == 1
        assert received[0][1] == RecoveryAction.NOTIFY

    @pytest.mark.asyncio
    async def test_async_callback_invoked(self, detector):
        """Should handle async callbacks."""
        received = []

        async def async_callback(item, action):
            received.append((item, action))

        detector.register_callback(async_callback)

        item = StuckWorkItem(
            id="item-001",
            work_type="bead",
            title="Test",
            agent_id="agent-001",
            age=WorkAge.YELLOW,
            last_update=_now(),
            time_since_update=timedelta(minutes=3),
            recommended_action=RecoveryAction.NOTIFY,
        )

        await detector._handle_stuck_item(item)
        assert len(received) == 1


# ============================================================================
# Test StuckDetector Recovery Actions
# ============================================================================


class TestStuckDetectorRecovery:
    """Tests for recovery actions."""

    @pytest.mark.asyncio
    async def test_handle_stuck_item_notify(self, detector):
        """Should handle NOTIFY action."""
        item = StuckWorkItem(
            id="item-001",
            work_type="bead",
            title="Test",
            agent_id="agent-001",
            age=WorkAge.YELLOW,
            last_update=_now(),
            time_since_update=timedelta(minutes=3),
            recommended_action=RecoveryAction.NOTIFY,
        )

        result = await detector._handle_stuck_item(item)
        assert result is True
        assert detector._recovery_counts.get("item-001") == 1

    @pytest.mark.asyncio
    async def test_handle_stuck_item_none_action(self, detector):
        """Should not take action for NONE."""
        item = StuckWorkItem(
            id="item-001",
            work_type="bead",
            title="Test",
            agent_id="agent-001",
            age=WorkAge.GREEN,
            last_update=_now(),
            time_since_update=timedelta(minutes=1),
            recommended_action=RecoveryAction.NONE,
        )

        result = await detector._handle_stuck_item(item)
        assert result is False

    @pytest.mark.asyncio
    async def test_reassign_bead(self, detector, mock_bead_store):
        """Should reset bead to pending for reassignment."""
        detector.bead_store = mock_bead_store

        mock_bead = MagicMock()
        mock_bead.id = "bead-001"
        mock_bead_store.get = AsyncMock(return_value=mock_bead)

        item = StuckWorkItem(
            id="bead-001",
            work_type="bead",
            title="Test",
            agent_id="agent-001",
            age=WorkAge.RED,
            last_update=_now(),
            time_since_update=timedelta(minutes=15),
            recommended_action=RecoveryAction.REASSIGN,
            metadata={},
        )

        result = await detector._do_reassign(item)
        assert result is True
        mock_bead_store.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_escalate_creates_chain(self, detector, mock_escalation_store):
        """Should create escalation chain."""
        detector.escalation_store = mock_escalation_store

        item = StuckWorkItem(
            id="item-001",
            work_type="bead",
            title="Test",
            agent_id="agent-001",
            age=WorkAge.RED,
            last_update=_now(),
            time_since_update=timedelta(minutes=15),
            previous_recoveries=2,
        )

        result = await detector._do_escalate(item)
        assert result is True
        mock_escalation_store.create_chain.assert_called_once()

    @pytest.mark.asyncio
    async def test_escalate_no_store(self, detector):
        """Should return False without escalation store."""
        detector.escalation_store = None

        item = StuckWorkItem(
            id="item-001",
            work_type="bead",
            title="Test",
            agent_id="agent-001",
            age=WorkAge.RED,
            last_update=_now(),
            time_since_update=timedelta(minutes=15),
            previous_recoveries=2,
        )

        result = await detector._do_escalate(item)
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_bead(self, detector, mock_bead_store):
        """Should cancel stuck bead."""
        detector.bead_store = mock_bead_store

        mock_bead = MagicMock()
        mock_bead.id = "bead-001"
        mock_bead_store.get = AsyncMock(return_value=mock_bead)

        item = StuckWorkItem(
            id="bead-001",
            work_type="bead",
            title="Test",
            agent_id="agent-001",
            age=WorkAge.RED,
            last_update=_now(),
            time_since_update=timedelta(minutes=15),
            previous_recoveries=3,
            metadata={},
        )

        result = await detector._do_cancel(item)
        assert result is True
        mock_bead_store.update.assert_called_once()


# ============================================================================
# Test StuckDetector Item Check
# ============================================================================


class TestStuckDetectorItemCheck:
    """Tests for checking specific items."""

    @pytest.mark.asyncio
    async def test_check_item_bead_found(self, detector, mock_bead_store):
        """Should return item for found bead."""
        detector.bead_store = mock_bead_store

        mock_bead = MagicMock()
        mock_bead.id = "bead-001"
        mock_bead.title = "Test Bead"
        mock_bead.claimed_by = "agent-001"
        mock_bead.updated_at = _now() - timedelta(minutes=5)
        mock_bead.created_at = _now() - timedelta(minutes=10)

        mock_bead_store.get = AsyncMock(return_value=mock_bead)

        item = await detector.check_item("bead-001", "bead")
        assert item is not None
        assert item.id == "bead-001"

    @pytest.mark.asyncio
    async def test_check_item_bead_not_found(self, detector, mock_bead_store):
        """Should return None for missing bead."""
        detector.bead_store = mock_bead_store
        mock_bead_store.get = AsyncMock(return_value=None)

        item = await detector.check_item("unknown", "bead")
        assert item is None

    @pytest.mark.asyncio
    async def test_check_item_assignment_found(self, detector, mock_coordinator):
        """Should return item for found assignment."""
        detector.coordinator = mock_coordinator

        from aragora.nomic.convoy_coordinator import AssignmentStatus

        mock_assignment = MagicMock()
        mock_assignment.id = "assignment-001"
        mock_assignment.bead_id = "bead-001"
        mock_assignment.agent_id = "agent-001"
        mock_assignment.status = AssignmentStatus.ACTIVE
        mock_assignment.updated_at = _now() - timedelta(minutes=5)
        mock_assignment.assigned_at = _now() - timedelta(minutes=10)
        mock_assignment.previous_agents = []

        mock_coordinator._assignments = {"assignment-001": mock_assignment}

        item = await detector.check_item("assignment-001", "assignment")
        assert item is not None
        assert item.id == "assignment-001"

    @pytest.mark.asyncio
    async def test_check_item_unknown_type(self, detector):
        """Should return None for unknown type."""
        item = await detector.check_item("item-001", "unknown")
        assert item is None


# ============================================================================
# Test StuckDetector Health Summary
# ============================================================================


class TestStuckDetectorHealthSummary:
    """Tests for health summary generation."""

    @pytest.mark.asyncio
    async def test_get_health_summary_empty(self, detector):
        """Should return summary for no items."""
        summary = await detector.get_health_summary()

        assert summary.total_items == 0
        assert summary.green_count == 0
        assert summary.health_percentage == 100.0
        assert summary.oldest_stuck is None

    @pytest.mark.asyncio
    async def test_get_health_summary_with_items(self, detector, mock_bead_store):
        """Should return correct summary."""
        detector.bead_store = mock_bead_store

        # Create beads at different ages
        beads = []
        for i, minutes_old in enumerate([1, 3, 15]):
            mock_bead = MagicMock()
            mock_bead.id = f"bead-{i}"
            mock_bead.title = f"Test Bead {i}"
            mock_bead.claimed_by = "agent-001"
            mock_bead.updated_at = _now() - timedelta(minutes=minutes_old)
            mock_bead.created_at = _now() - timedelta(minutes=minutes_old + 5)
            mock_bead.bead_type = MagicMock(value="task")
            beads.append(mock_bead)

        mock_bead_store.list_by_status = AsyncMock(return_value=beads)

        summary = await detector.get_health_summary()

        assert summary.total_items == 3
        assert summary.green_count == 1  # 1 minute old
        assert summary.yellow_count == 1  # 3 minutes old
        assert summary.red_count == 1  # 15 minutes old

    @pytest.mark.asyncio
    async def test_get_health_summary_oldest_stuck(self, detector, mock_bead_store):
        """Should identify oldest stuck item."""
        detector.bead_store = mock_bead_store

        mock_bead1 = MagicMock()
        mock_bead1.id = "bead-1"
        mock_bead1.title = "Older Bead"
        mock_bead1.claimed_by = "agent-001"
        mock_bead1.updated_at = _now() - timedelta(minutes=20)
        mock_bead1.created_at = _now() - timedelta(minutes=25)
        mock_bead1.bead_type = MagicMock(value="task")

        mock_bead2 = MagicMock()
        mock_bead2.id = "bead-2"
        mock_bead2.title = "Newer Bead"
        mock_bead2.claimed_by = "agent-001"
        mock_bead2.updated_at = _now() - timedelta(minutes=15)
        mock_bead2.created_at = _now() - timedelta(minutes=20)
        mock_bead2.bead_type = MagicMock(value="task")

        mock_bead_store.list_by_status = AsyncMock(return_value=[mock_bead1, mock_bead2])

        summary = await detector.get_health_summary()

        assert summary.oldest_stuck is not None
        assert summary.oldest_stuck.id == "bead-1"

    @pytest.mark.asyncio
    async def test_health_summary_by_agent(self, detector, mock_bead_store):
        """Should group by agent."""
        detector.bead_store = mock_bead_store

        beads = []
        for i in range(2):
            mock_bead = MagicMock()
            mock_bead.id = f"bead-{i}"
            mock_bead.title = f"Test Bead {i}"
            mock_bead.claimed_by = f"agent-{i % 2}"  # Distribute across agents
            mock_bead.updated_at = _now() - timedelta(minutes=1)
            mock_bead.created_at = _now() - timedelta(minutes=5)
            mock_bead.bead_type = MagicMock(value="task")
            beads.append(mock_bead)

        mock_bead_store.list_by_status = AsyncMock(return_value=beads)

        summary = await detector.get_health_summary()

        assert "agent-0" in summary.by_agent or "agent-1" in summary.by_agent


# ============================================================================
# Test StuckDetector Statistics
# ============================================================================


class TestStuckDetectorStatistics:
    """Tests for statistics gathering."""

    @pytest.mark.asyncio
    async def test_get_statistics(self, detector):
        """Should return statistics."""
        stats = await detector.get_statistics()

        assert "total_items" in stats
        assert "healthy_percentage" in stats
        assert "stuck_count" in stats
        assert "recovered_count" in stats
        assert "monitoring_active" in stats
        assert stats["monitoring_active"] is False

    @pytest.mark.asyncio
    async def test_statistics_monitoring_active(self, detector):
        """Should reflect monitoring state."""
        await detector.start_monitoring()
        stats = await detector.get_statistics()
        assert stats["monitoring_active"] is True

        await detector.stop_monitoring()
        stats = await detector.get_statistics()
        assert stats["monitoring_active"] is False


# ============================================================================
# Test Singleton Factory
# ============================================================================


class TestStuckDetectorSingleton:
    """Tests for singleton factory functions."""

    @pytest.mark.asyncio
    async def test_get_stuck_detector(self):
        """Should return detector instance."""
        reset_stuck_detector()
        detector = await get_stuck_detector()

        assert detector is not None
        assert isinstance(detector, StuckDetector)

    @pytest.mark.asyncio
    async def test_get_stuck_detector_returns_same_instance(self):
        """Should return same instance on subsequent calls."""
        reset_stuck_detector()
        detector1 = await get_stuck_detector()
        detector2 = await get_stuck_detector()

        assert detector1 is detector2

    def test_reset_stuck_detector(self):
        """Should reset singleton."""
        reset_stuck_detector()
        # After reset, next call should create new instance


# ============================================================================
# Test False Positive Prevention
# ============================================================================


class TestStuckDetectorFalsePositives:
    """Tests for false positive prevention."""

    @pytest.mark.asyncio
    async def test_completed_assignments_not_detected(self, detector, mock_coordinator):
        """Should not flag completed assignments as stuck."""
        detector.coordinator = mock_coordinator

        from aragora.nomic.convoy_coordinator import AssignmentStatus

        mock_assignment = MagicMock()
        mock_assignment.id = "assignment-001"
        mock_assignment.bead_id = "bead-001"
        mock_assignment.agent_id = "agent-001"
        mock_assignment.status = AssignmentStatus.COMPLETED  # Not ACTIVE
        mock_assignment.updated_at = _now() - timedelta(minutes=15)
        mock_assignment.assigned_at = _now() - timedelta(minutes=20)

        mock_coordinator._assignments = {"assignment-001": mock_assignment}

        items = await detector.detect_stuck_work()
        # Completed assignments should not be detected
        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_fresh_items_not_stuck(self, detector, mock_bead_store):
        """Should not flag recent items as stuck."""
        detector.bead_store = mock_bead_store

        mock_bead = MagicMock()
        mock_bead.id = "bead-001"
        mock_bead.title = "Fresh Bead"
        mock_bead.claimed_by = "agent-001"
        mock_bead.updated_at = _now() - timedelta(seconds=30)  # Very recent
        mock_bead.created_at = _now() - timedelta(minutes=1)
        mock_bead.bead_type = MagicMock(value="task")

        mock_bead_store.list_by_status = AsyncMock(return_value=[mock_bead])

        items = await detector.detect_stuck_work()
        assert len(items) == 1
        assert items[0].is_stuck is False
        assert items[0].age == WorkAge.GREEN


# ============================================================================
# Test Recovery Count Tracking
# ============================================================================


class TestRecoveryCountTracking:
    """Tests for recovery count tracking."""

    @pytest.mark.asyncio
    async def test_recovery_count_increments(self, detector):
        """Should increment recovery count on handling."""
        item = StuckWorkItem(
            id="item-001",
            work_type="bead",
            title="Test",
            agent_id="agent-001",
            age=WorkAge.YELLOW,
            last_update=_now(),
            time_since_update=timedelta(minutes=3),
            previous_recoveries=0,
            recommended_action=RecoveryAction.NOTIFY,
        )

        await detector._handle_stuck_item(item)
        assert detector._recovery_counts.get("item-001") == 1

        # Handle again
        item.previous_recoveries = 1
        await detector._handle_stuck_item(item)
        assert detector._recovery_counts.get("item-001") == 2

    @pytest.mark.asyncio
    async def test_recovery_count_affects_action(self, detector):
        """Should use recovery count to determine action."""
        # Set high recovery count
        detector._recovery_counts["item-001"] = 3

        item = StuckWorkItem(
            id="item-001",
            work_type="bead",
            title="Test",
            agent_id="agent-001",
            age=WorkAge.RED,
            last_update=_now(),
            time_since_update=timedelta(minutes=15),
            previous_recoveries=3,
        )

        action = detector._determine_action(item)
        assert action == RecoveryAction.CANCEL
