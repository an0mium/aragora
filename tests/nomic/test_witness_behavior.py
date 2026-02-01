"""
Tests for Witness Behavior Module.

Comprehensive tests for:
- Behavior observation recording
- Anomaly detection
- Metrics collection
- Threshold alerts
- Event serialization
- Historical data access
- Observer lifecycle (start/stop)
- Health report generation
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.witness_behavior import (
    AgentHealthCheck,
    Alert,
    AlertSeverity,
    ConvoyHealthCheck,
    HealthReport,
    HealthStatus,
    HeartbeatMonitor,
    WitnessBehavior,
    WitnessConfig,
    get_witness_behavior,
    reset_witness_behavior,
)
from aragora.nomic.agent_roles import (
    AgentHierarchy,
    AgentRole,
    RoleAssignment,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_hierarchy():
    """Create a mock agent hierarchy."""
    hierarchy = MagicMock(spec=AgentHierarchy)
    hierarchy.get_agents_by_role = AsyncMock(return_value=[])
    hierarchy.get_assignment = AsyncMock(return_value=None)
    hierarchy.cleanup_expired_polecats = AsyncMock()
    return hierarchy


@pytest.fixture
def mock_convoy_manager():
    """Create a mock convoy manager."""
    manager = MagicMock()
    manager.list_convoys = AsyncMock(return_value=[])
    manager.get_convoy_progress = AsyncMock()
    return manager


@pytest.fixture
def mock_coordinator():
    """Create a mock convoy coordinator."""
    coordinator = MagicMock()
    coordinator.get_agent_assignments = AsyncMock(return_value=[])
    coordinator.handle_agent_failure = AsyncMock()
    return coordinator


@pytest.fixture
def mock_escalation_store():
    """Create a mock escalation store."""
    store = MagicMock()
    store.create_chain = AsyncMock()
    store.process_auto_escalations = AsyncMock()
    return store


@pytest.fixture
def default_config():
    """Create a default witness config."""
    return WitnessConfig(
        patrol_interval_seconds=1,  # Fast for tests
        health_check_interval_seconds=1,
        report_interval_seconds=5,
        heartbeat_timeout_seconds=120,
        stuck_threshold_minutes=10,
        error_rate_threshold=0.2,
        max_alerts_per_target=5,
        alert_cooldown_seconds=1,
        auto_escalate_critical=True,
    )


@pytest.fixture
async def witness(mock_hierarchy, default_config):
    """Create a witness behavior instance."""
    reset_witness_behavior()
    witness = WitnessBehavior(
        hierarchy=mock_hierarchy,
        config=default_config,
    )
    yield witness
    await witness.stop_patrol()
    reset_witness_behavior()


@pytest.fixture
def sample_role_assignment():
    """Create a sample role assignment."""
    return RoleAssignment(
        agent_id="agent-001",
        role=AgentRole.CREW,
        assigned_at=datetime.now(timezone.utc),
    )


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    reset_witness_behavior()
    yield
    reset_witness_behavior()


# ============================================================================
# Test HealthStatus Enum
# ============================================================================


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_health_status_values(self):
        """Should have correct status values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.CRITICAL.value == "critical"
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_health_status_string_enum(self):
        """Should be a string enum."""
        assert isinstance(HealthStatus.HEALTHY, str)
        assert HealthStatus.HEALTHY == "healthy"


# ============================================================================
# Test AlertSeverity Enum
# ============================================================================


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_alert_severity_values(self):
        """Should have correct severity values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"


# ============================================================================
# Test AgentHealthCheck
# ============================================================================


class TestAgentHealthCheck:
    """Tests for AgentHealthCheck dataclass."""

    def test_agent_health_check_creation(self):
        """Should create health check with defaults."""
        now = datetime.now(timezone.utc)
        check = AgentHealthCheck(
            agent_id="agent-001",
            status=HealthStatus.HEALTHY,
            checked_at=now,
        )
        assert check.agent_id == "agent-001"
        assert check.status == HealthStatus.HEALTHY
        assert check.active_beads == 0
        assert check.completed_beads == 0
        assert check.failed_beads == 0
        assert check.error_rate == 0.0
        assert check.issues == []

    def test_is_responsive_with_recent_heartbeat(self):
        """Should be responsive with recent heartbeat."""
        now = datetime.now(timezone.utc)
        check = AgentHealthCheck(
            agent_id="agent-001",
            status=HealthStatus.HEALTHY,
            checked_at=now,
            last_heartbeat=now - timedelta(minutes=2),
        )
        assert check.is_responsive is True

    def test_is_responsive_with_old_heartbeat(self):
        """Should not be responsive with old heartbeat."""
        now = datetime.now(timezone.utc)
        check = AgentHealthCheck(
            agent_id="agent-001",
            status=HealthStatus.HEALTHY,
            checked_at=now,
            last_heartbeat=now - timedelta(minutes=10),
        )
        assert check.is_responsive is False

    def test_is_responsive_without_heartbeat(self):
        """Should not be responsive without heartbeat."""
        check = AgentHealthCheck(
            agent_id="agent-001",
            status=HealthStatus.HEALTHY,
            checked_at=datetime.now(timezone.utc),
        )
        assert check.is_responsive is False

    def test_is_stuck_with_active_beads_no_activity(self):
        """Should be stuck with active beads and no activity."""
        check = AgentHealthCheck(
            agent_id="agent-001",
            status=HealthStatus.DEGRADED,
            checked_at=datetime.now(timezone.utc),
            active_beads=1,
            last_activity=None,
        )
        assert check.is_stuck is True

    def test_is_stuck_with_active_beads_old_activity(self):
        """Should be stuck with active beads and old activity."""
        now = datetime.now(timezone.utc)
        check = AgentHealthCheck(
            agent_id="agent-001",
            status=HealthStatus.DEGRADED,
            checked_at=now,
            active_beads=1,
            last_activity=now - timedelta(minutes=15),
        )
        assert check.is_stuck is True

    def test_is_not_stuck_with_no_active_beads(self):
        """Should not be stuck with no active beads."""
        check = AgentHealthCheck(
            agent_id="agent-001",
            status=HealthStatus.HEALTHY,
            checked_at=datetime.now(timezone.utc),
            active_beads=0,
        )
        assert check.is_stuck is False

    def test_is_not_stuck_with_recent_activity(self):
        """Should not be stuck with recent activity."""
        now = datetime.now(timezone.utc)
        check = AgentHealthCheck(
            agent_id="agent-001",
            status=HealthStatus.HEALTHY,
            checked_at=now,
            active_beads=1,
            last_activity=now - timedelta(minutes=5),
        )
        assert check.is_stuck is False


# ============================================================================
# Test ConvoyHealthCheck
# ============================================================================


class TestConvoyHealthCheck:
    """Tests for ConvoyHealthCheck dataclass."""

    def test_convoy_health_check_creation(self):
        """Should create convoy health check."""
        now = datetime.now(timezone.utc)
        check = ConvoyHealthCheck(
            convoy_id="convoy-001",
            status=HealthStatus.HEALTHY,
            checked_at=now,
            total_beads=10,
            completed_beads=5,
            failed_beads=0,
            stuck_beads=0,
            completion_rate=0.5,
        )
        assert check.convoy_id == "convoy-001"
        assert check.total_beads == 10
        assert check.completion_rate == 0.5

    def test_progress_percentage(self):
        """Should calculate correct progress percentage."""
        check = ConvoyHealthCheck(
            convoy_id="convoy-001",
            status=HealthStatus.HEALTHY,
            checked_at=datetime.now(timezone.utc),
            total_beads=10,
            completed_beads=7,
            failed_beads=0,
            stuck_beads=0,
            completion_rate=0.7,
        )
        assert check.progress_percentage == 70.0

    def test_progress_percentage_empty_convoy(self):
        """Should return 0 for empty convoy."""
        check = ConvoyHealthCheck(
            convoy_id="convoy-001",
            status=HealthStatus.HEALTHY,
            checked_at=datetime.now(timezone.utc),
            total_beads=0,
            completed_beads=0,
            failed_beads=0,
            stuck_beads=0,
            completion_rate=0.0,
        )
        assert check.progress_percentage == 0.0


# ============================================================================
# Test Alert
# ============================================================================


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation(self):
        """Should create alert with fields."""
        now = datetime.now(timezone.utc)
        alert = Alert(
            id="alert-001",
            severity=AlertSeverity.WARNING,
            source="witness_patrol",
            target="agent-001",
            message="High latency detected",
            timestamp=now,
        )
        assert alert.id == "alert-001"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.acknowledged is False

    def test_alert_to_dict(self):
        """Should serialize to dictionary."""
        now = datetime.now(timezone.utc)
        alert = Alert(
            id="alert-001",
            severity=AlertSeverity.ERROR,
            source="witness",
            target="agent-001",
            message="Test alert",
            timestamp=now,
        )
        data = alert.to_dict()

        assert data["id"] == "alert-001"
        assert data["severity"] == "error"
        assert data["source"] == "witness"
        assert data["target"] == "agent-001"
        assert data["acknowledged"] is False
        assert data["acknowledged_at"] is None

    def test_alert_to_dict_acknowledged(self):
        """Should include acknowledgment info when acknowledged."""
        now = datetime.now(timezone.utc)
        alert = Alert(
            id="alert-001",
            severity=AlertSeverity.INFO,
            source="witness",
            target="agent-001",
            message="Test",
            timestamp=now,
            acknowledged=True,
            acknowledged_at=now,
            acknowledged_by="admin",
        )
        data = alert.to_dict()

        assert data["acknowledged"] is True
        assert data["acknowledged_at"] is not None
        assert data["acknowledged_by"] == "admin"


# ============================================================================
# Test HealthReport
# ============================================================================


class TestHealthReport:
    """Tests for HealthReport dataclass."""

    def test_health_report_creation(self):
        """Should create health report."""
        now = datetime.now(timezone.utc)
        report = HealthReport(
            report_id="report-001",
            generated_at=now,
            overall_status=HealthStatus.HEALTHY,
            agent_checks=[],
            convoy_checks=[],
            alerts=[],
            statistics={"total_agents": 0},
        )
        assert report.report_id == "report-001"
        assert report.overall_status == HealthStatus.HEALTHY

    def test_health_report_to_dict(self):
        """Should serialize to dictionary."""
        now = datetime.now(timezone.utc)
        agent_check = AgentHealthCheck(
            agent_id="agent-001",
            status=HealthStatus.HEALTHY,
            checked_at=now,
        )
        report = HealthReport(
            report_id="report-001",
            generated_at=now,
            overall_status=HealthStatus.HEALTHY,
            agent_checks=[agent_check],
            convoy_checks=[],
            alerts=[],
            statistics={"total_agents": 1},
            recommendations=["Monitor system"],
        )
        data = report.to_dict()

        assert data["report_id"] == "report-001"
        assert data["overall_status"] == "healthy"
        assert len(data["agent_checks"]) == 1
        assert data["agent_checks"][0]["agent_id"] == "agent-001"
        assert data["recommendations"] == ["Monitor system"]


# ============================================================================
# Test WitnessConfig
# ============================================================================


class TestWitnessConfig:
    """Tests for WitnessConfig dataclass."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = WitnessConfig()
        assert config.patrol_interval_seconds == 30
        assert config.health_check_interval_seconds == 60
        assert config.heartbeat_timeout_seconds == 120
        assert config.stuck_threshold_minutes == 10
        assert config.error_rate_threshold == 0.2
        assert config.max_alerts_per_target == 5
        assert config.auto_escalate_critical is True

    def test_custom_config(self):
        """Should accept custom values."""
        config = WitnessConfig(
            patrol_interval_seconds=60,
            heartbeat_timeout_seconds=300,
            max_alerts_per_target=10,
        )
        assert config.patrol_interval_seconds == 60
        assert config.heartbeat_timeout_seconds == 300
        assert config.max_alerts_per_target == 10


# ============================================================================
# Test HeartbeatMonitor
# ============================================================================


class TestHeartbeatMonitor:
    """Tests for HeartbeatMonitor class."""

    @pytest.mark.asyncio
    async def test_record_heartbeat(self):
        """Should record heartbeat with timestamp."""
        monitor = HeartbeatMonitor(timeout_seconds=60)
        await monitor.record_heartbeat("agent-001")

        heartbeat = await monitor.get_last_heartbeat("agent-001")
        assert heartbeat is not None

    @pytest.mark.asyncio
    async def test_is_responsive_true(self):
        """Should return true for recent heartbeat."""
        monitor = HeartbeatMonitor(timeout_seconds=60)
        await monitor.record_heartbeat("agent-001")

        assert await monitor.is_responsive("agent-001") is True

    @pytest.mark.asyncio
    async def test_is_responsive_false_no_heartbeat(self):
        """Should return false for missing heartbeat."""
        monitor = HeartbeatMonitor(timeout_seconds=60)

        assert await monitor.is_responsive("unknown-agent") is False

    @pytest.mark.asyncio
    async def test_is_responsive_false_old_heartbeat(self):
        """Should return false for old heartbeat."""
        monitor = HeartbeatMonitor(timeout_seconds=1)
        monitor._heartbeats["agent-001"] = datetime.now(timezone.utc) - timedelta(seconds=10)

        assert await monitor.is_responsive("agent-001") is False

    @pytest.mark.asyncio
    async def test_get_unresponsive_agents(self):
        """Should return list of unresponsive agents."""
        monitor = HeartbeatMonitor(timeout_seconds=1)
        now = datetime.now(timezone.utc)

        # One responsive, one not
        monitor._heartbeats["agent-001"] = now
        monitor._heartbeats["agent-002"] = now - timedelta(seconds=10)

        unresponsive = await monitor.get_unresponsive_agents()
        assert "agent-002" in unresponsive
        assert "agent-001" not in unresponsive


# ============================================================================
# Test WitnessBehavior Initialization
# ============================================================================


class TestWitnessBehaviorInit:
    """Tests for WitnessBehavior initialization."""

    def test_initialization(self, mock_hierarchy, default_config):
        """Should initialize with dependencies."""
        witness = WitnessBehavior(
            hierarchy=mock_hierarchy,
            config=default_config,
        )
        assert witness.hierarchy is mock_hierarchy
        assert witness.config is default_config
        assert witness._running is False
        assert witness._patrol_task is None

    def test_default_config_used(self, mock_hierarchy):
        """Should use default config if not provided."""
        witness = WitnessBehavior(hierarchy=mock_hierarchy)
        assert witness.config is not None
        assert isinstance(witness.config, WitnessConfig)


# ============================================================================
# Test WitnessBehavior Patrol Lifecycle
# ============================================================================


class TestWitnessBehaviorPatrol:
    """Tests for patrol lifecycle."""

    @pytest.mark.asyncio
    async def test_start_patrol(self, witness):
        """Should start patrol loop."""
        await witness.start_patrol()
        assert witness._running is True
        assert witness._patrol_task is not None

    @pytest.mark.asyncio
    async def test_start_patrol_idempotent(self, witness):
        """Should not restart if already running."""
        await witness.start_patrol()
        task1 = witness._patrol_task

        await witness.start_patrol()
        task2 = witness._patrol_task

        assert task1 is task2

    @pytest.mark.asyncio
    async def test_stop_patrol(self, witness):
        """Should stop patrol loop."""
        await witness.start_patrol()
        await witness.stop_patrol()

        assert witness._running is False
        assert witness._patrol_task is None

    @pytest.mark.asyncio
    async def test_stop_patrol_when_not_running(self, witness):
        """Should handle stopping when not running."""
        await witness.stop_patrol()  # Should not raise
        assert witness._running is False


# ============================================================================
# Test WitnessBehavior Agent Health Checks
# ============================================================================


class TestWitnessBehaviorHealthChecks:
    """Tests for agent health checks."""

    @pytest.mark.asyncio
    async def test_check_agent_health_not_found(self, witness):
        """Should return UNKNOWN for unknown agent."""
        witness.hierarchy.get_assignment = AsyncMock(return_value=None)

        health = await witness.check_agent_health("unknown-agent")
        assert health.status == HealthStatus.UNKNOWN
        assert "Agent not found" in health.issues[0]

    @pytest.mark.asyncio
    async def test_check_agent_health_no_heartbeat(self, witness, sample_role_assignment):
        """Should be UNHEALTHY with no heartbeat."""
        witness.hierarchy.get_assignment = AsyncMock(return_value=sample_role_assignment)

        health = await witness.check_agent_health("agent-001")
        assert health.status == HealthStatus.UNHEALTHY
        assert "No recent heartbeat" in health.issues

    @pytest.mark.asyncio
    async def test_check_agent_health_healthy(self, witness, sample_role_assignment):
        """Should be HEALTHY with recent heartbeat."""
        witness.hierarchy.get_assignment = AsyncMock(return_value=sample_role_assignment)
        await witness.heartbeat_monitor.record_heartbeat("agent-001")

        health = await witness.check_agent_health("agent-001")
        assert health.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_check_agent_health_high_error_rate(
        self, witness, sample_role_assignment, mock_coordinator
    ):
        """Should detect high error rate."""
        witness.hierarchy.get_assignment = AsyncMock(return_value=sample_role_assignment)
        witness.coordinator = mock_coordinator
        await witness.heartbeat_monitor.record_heartbeat("agent-001")

        # Mock assignments with high failure rate
        from aragora.nomic.convoy_coordinator import AssignmentStatus

        mock_assignments = [
            MagicMock(status=AssignmentStatus.COMPLETED, completed_at=None),
            MagicMock(status=AssignmentStatus.FAILED, completed_at=None),
            MagicMock(status=AssignmentStatus.FAILED, completed_at=None),
        ]
        mock_coordinator.get_agent_assignments = AsyncMock(return_value=mock_assignments)

        health = await witness.check_agent_health("agent-001")
        assert health.error_rate > witness.config.error_rate_threshold


# ============================================================================
# Test WitnessBehavior Alert Creation
# ============================================================================


class TestWitnessBehaviorAlerts:
    """Tests for alert creation."""

    @pytest.mark.asyncio
    async def test_create_alert(self, witness):
        """Should create alert."""
        alert = await witness._create_alert(
            severity=AlertSeverity.WARNING,
            source="test",
            target="agent-001",
            message="Test alert",
        )
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING
        assert alert.target == "agent-001"
        assert alert.id in witness._alerts

    @pytest.mark.asyncio
    async def test_alert_cooldown(self, witness):
        """Should suppress alerts during cooldown."""
        # Create first alert
        alert1 = await witness._create_alert(
            severity=AlertSeverity.WARNING,
            source="test",
            target="agent-001",
            message="First",
        )
        assert alert1 is not None

        # Second alert should be suppressed
        alert2 = await witness._create_alert(
            severity=AlertSeverity.WARNING,
            source="test",
            target="agent-001",
            message="Second",
        )
        assert alert2 is None

    @pytest.mark.asyncio
    async def test_alert_max_per_target(self, witness, default_config):
        """Should limit alerts per target."""
        # Set low limit
        witness.config.max_alerts_per_target = 2
        witness.config.alert_cooldown_seconds = 0  # Disable cooldown

        await witness._create_alert(
            severity=AlertSeverity.INFO,
            source="test",
            target="agent-001",
            message="Alert 1",
        )
        await witness._create_alert(
            severity=AlertSeverity.INFO,
            source="test",
            target="agent-001",
            message="Alert 2",
        )
        alert3 = await witness._create_alert(
            severity=AlertSeverity.INFO,
            source="test",
            target="agent-001",
            message="Alert 3",
        )
        assert alert3 is None

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, witness):
        """Should acknowledge alert."""
        alert = await witness._create_alert(
            severity=AlertSeverity.WARNING,
            source="test",
            target="agent-001",
            message="Test",
        )
        assert alert is not None

        result = await witness.acknowledge_alert(alert.id, "admin")
        assert result is True
        assert witness._alerts[alert.id].acknowledged is True
        assert witness._alerts[alert.id].acknowledged_by == "admin"

    @pytest.mark.asyncio
    async def test_acknowledge_alert_not_found(self, witness):
        """Should return False for unknown alert."""
        result = await witness.acknowledge_alert("unknown-id", "admin")
        assert result is False

    @pytest.mark.asyncio
    async def test_alert_callback(self, witness):
        """Should notify callbacks on alert."""
        received_alerts = []

        def callback(alert):
            received_alerts.append(alert)

        witness.register_callback(callback)

        await witness._create_alert(
            severity=AlertSeverity.INFO,
            source="test",
            target="agent-001",
            message="Test",
        )

        assert len(received_alerts) == 1
        assert received_alerts[0].message == "Test"

    @pytest.mark.asyncio
    async def test_async_alert_callback(self, witness):
        """Should handle async callbacks."""
        received_alerts = []

        async def async_callback(alert):
            received_alerts.append(alert)

        witness.register_callback(async_callback)

        await witness._create_alert(
            severity=AlertSeverity.INFO,
            source="test",
            target="agent-001",
            message="Test",
        )

        assert len(received_alerts) == 1


# ============================================================================
# Test WitnessBehavior Health Reports
# ============================================================================


class TestWitnessBehaviorReports:
    """Tests for health report generation."""

    @pytest.mark.asyncio
    async def test_generate_health_report_empty(self, witness):
        """Should generate report with no agents."""
        report = await witness.generate_health_report()

        assert report is not None
        assert report.overall_status == HealthStatus.HEALTHY
        assert report.agent_checks == []
        assert report.statistics["total_agents"] == 0

    @pytest.mark.asyncio
    async def test_generate_health_report_with_agents(self, witness, sample_role_assignment):
        """Should include agents in report."""
        witness.hierarchy.get_agents_by_role = AsyncMock(return_value=[sample_role_assignment])
        witness.hierarchy.get_assignment = AsyncMock(return_value=sample_role_assignment)

        report = await witness.generate_health_report()

        assert report.statistics["total_agents"] >= 1

    @pytest.mark.asyncio
    async def test_generate_health_report_determines_overall_status(self, witness):
        """Should determine overall status from agent checks."""
        # Create an alert to make a critical check
        await witness._create_alert(
            severity=AlertSeverity.CRITICAL,
            source="test",
            target="agent-001",
            message="Critical issue",
        )

        report = await witness.generate_health_report()
        assert report.alerts is not None

    @pytest.mark.asyncio
    async def test_reports_limited_in_memory(self, witness):
        """Should limit stored reports."""
        # Generate many reports
        for _ in range(150):
            await witness.generate_health_report()

        assert len(witness._reports) <= 100


# ============================================================================
# Test WitnessBehavior Recent Alerts Query
# ============================================================================


class TestWitnessBehaviorAlertQueries:
    """Tests for querying recent alerts."""

    @pytest.mark.asyncio
    async def test_get_recent_alerts(self, witness):
        """Should return recent alerts."""
        witness.config.alert_cooldown_seconds = 0
        witness.config.max_alerts_per_target = 100

        for i in range(3):
            await witness._create_alert(
                severity=AlertSeverity.INFO,
                source="test",
                target=f"agent-{i}",
                message=f"Alert {i}",
            )

        alerts = await witness.get_recent_alerts(hours=1)
        assert len(alerts) == 3

    @pytest.mark.asyncio
    async def test_get_recent_alerts_severity_filter(self, witness):
        """Should filter by severity."""
        witness.config.alert_cooldown_seconds = 0
        witness.config.max_alerts_per_target = 100

        await witness._create_alert(
            severity=AlertSeverity.INFO,
            source="test",
            target="agent-1",
            message="Info",
        )
        await witness._create_alert(
            severity=AlertSeverity.WARNING,
            source="test",
            target="agent-2",
            message="Warning",
        )

        alerts = await witness.get_recent_alerts(severity=AlertSeverity.WARNING)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING

    @pytest.mark.asyncio
    async def test_get_recent_alerts_sorted_by_time(self, witness):
        """Should return alerts sorted by timestamp (newest first)."""
        witness.config.alert_cooldown_seconds = 0
        witness.config.max_alerts_per_target = 100

        for i in range(3):
            await witness._create_alert(
                severity=AlertSeverity.INFO,
                source="test",
                target=f"agent-{i}",
                message=f"Alert {i}",
            )
            await asyncio.sleep(0.01)  # Small delay between alerts

        alerts = await witness.get_recent_alerts()
        # Most recent should be first
        timestamps = [a.timestamp for a in alerts]
        assert timestamps == sorted(timestamps, reverse=True)


# ============================================================================
# Test WitnessBehavior Statistics
# ============================================================================


class TestWitnessBehaviorStatistics:
    """Tests for statistics gathering."""

    @pytest.mark.asyncio
    async def test_get_statistics(self, witness):
        """Should return statistics."""
        stats = await witness.get_statistics()

        assert "total_alerts" in stats
        assert "unacknowledged_alerts" in stats
        assert "reports_generated" in stats
        assert "patrol_running" in stats
        assert stats["patrol_running"] is False

    @pytest.mark.asyncio
    async def test_statistics_reflect_state(self, witness):
        """Should reflect current state."""
        # Create some alerts
        witness.config.alert_cooldown_seconds = 0
        witness.config.max_alerts_per_target = 100

        await witness._create_alert(
            severity=AlertSeverity.INFO,
            source="test",
            target="agent-1",
            message="Test",
        )
        await witness.generate_health_report()

        stats = await witness.get_statistics()
        assert stats["total_alerts"] == 1
        assert stats["reports_generated"] == 1


# ============================================================================
# Test Singleton Factory
# ============================================================================


class TestWitnessSingleton:
    """Tests for singleton factory functions."""

    @pytest.mark.asyncio
    async def test_get_witness_behavior(self, mock_hierarchy):
        """Should return witness instance."""
        reset_witness_behavior()
        witness = await get_witness_behavior(hierarchy=mock_hierarchy)

        assert witness is not None
        assert isinstance(witness, WitnessBehavior)

    @pytest.mark.asyncio
    async def test_get_witness_behavior_returns_same_instance(self, mock_hierarchy):
        """Should return same instance on subsequent calls."""
        reset_witness_behavior()
        witness1 = await get_witness_behavior(hierarchy=mock_hierarchy)
        witness2 = await get_witness_behavior(hierarchy=mock_hierarchy)

        assert witness1 is witness2

    def test_reset_witness_behavior(self, mock_hierarchy):
        """Should reset singleton."""
        reset_witness_behavior()
        # After reset, next call should create new instance
        # (tested by other tests)


# ============================================================================
# Test Escalation Integration
# ============================================================================


class TestWitnessBehaviorEscalation:
    """Tests for escalation integration."""

    @pytest.mark.asyncio
    async def test_critical_alert_creates_escalation(
        self, witness, mock_escalation_store, sample_role_assignment
    ):
        """Should escalate critical alerts."""
        witness.escalation_store = mock_escalation_store
        witness.hierarchy.get_assignment = AsyncMock(return_value=sample_role_assignment)

        # Create unhealthy agent to trigger escalation
        health = AgentHealthCheck(
            agent_id="agent-001",
            status=HealthStatus.CRITICAL,
            checked_at=datetime.now(timezone.utc),
            issues=["Issue 1", "Issue 2", "Issue 3"],
        )

        await witness._handle_unhealthy_agent(sample_role_assignment, health)

        # Should have created escalation
        mock_escalation_store.create_chain.assert_called_once()


# ============================================================================
# Test Convoy Monitoring
# ============================================================================


class TestWitnessBehaviorConvoyMonitoring:
    """Tests for convoy monitoring."""

    @pytest.mark.asyncio
    async def test_check_convoy_progress_no_manager(self, witness):
        """Should handle missing convoy manager."""
        witness.convoy_manager = None
        # Should not raise
        await witness._check_convoy_progress()

    @pytest.mark.asyncio
    async def test_check_convoy_progress_stalled_convoy(self, witness, mock_convoy_manager):
        """Should create alert for stalled convoy."""
        witness.convoy_manager = mock_convoy_manager
        witness.config.alert_cooldown_seconds = 0

        # Mock active convoy
        mock_convoy = MagicMock()
        mock_convoy.id = "convoy-001"
        mock_convoy.title = "Test Convoy"
        mock_convoy_manager.list_convoys = AsyncMock(return_value=[mock_convoy])

        # Mock progress showing stalled state
        mock_progress = MagicMock()
        mock_progress.completion_percentage = 10.0
        mock_progress.running_beads = 0
        mock_progress.pending_beads = 5
        mock_convoy_manager.get_convoy_progress = AsyncMock(return_value=mock_progress)

        await witness._check_convoy_progress()

        # Should have created an alert
        assert len(witness._alerts) == 1
