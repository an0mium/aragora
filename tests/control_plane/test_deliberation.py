"""
Tests for Control Plane Deliberation Integration.

Tests cover:
- DeliberationStatus enum
- SLAComplianceLevel enum
- DeliberationSLA dataclass
- DeliberationMetrics dataclass
- DeliberationTask dataclass
- DeliberationManager integration
"""

import pytest
from datetime import datetime, timezone

from aragora.control_plane.deliberation import (
    DeliberationStatus,
    SLAComplianceLevel,
    DeliberationSLA,
    DeliberationMetrics,
    DeliberationTask,
    DELIBERATION_TASK_TYPE,
    DEFAULT_DELIBERATION_TIMEOUT,
)

# Import threshold constants
DEFAULT_SLA_WARNING_THRESHOLD = 0.8
DEFAULT_SLA_CRITICAL_THRESHOLD = 0.95


class TestDeliberationConstants:
    """Tests for deliberation module constants."""

    def test_task_type_constant(self):
        """Test deliberation task type constant."""
        assert DELIBERATION_TASK_TYPE == "deliberation"

    def test_default_timeout(self):
        """Test default timeout value."""
        assert DEFAULT_DELIBERATION_TIMEOUT == 300.0  # 5 minutes

    def test_sla_thresholds(self):
        """Test SLA threshold values."""
        assert DEFAULT_SLA_WARNING_THRESHOLD == 0.8  # 80%
        assert DEFAULT_SLA_CRITICAL_THRESHOLD == 0.95  # 95%


class TestDeliberationStatusEnum:
    """Tests for DeliberationStatus enum."""

    def test_all_statuses_defined(self):
        """Test that all statuses are defined."""
        expected = [
            "pending",
            "scheduled",
            "in_progress",
            "consensus_reached",
            "no_consensus",
            "failed",
            "timeout",
            "cancelled",
        ]
        for status in expected:
            assert DeliberationStatus(status) is not None

    def test_status_values(self):
        """Test status enum values."""
        assert DeliberationStatus.PENDING.value == "pending"
        assert DeliberationStatus.SCHEDULED.value == "scheduled"
        assert DeliberationStatus.IN_PROGRESS.value == "in_progress"
        assert DeliberationStatus.CONSENSUS_REACHED.value == "consensus_reached"
        assert DeliberationStatus.NO_CONSENSUS.value == "no_consensus"
        assert DeliberationStatus.FAILED.value == "failed"
        assert DeliberationStatus.TIMEOUT.value == "timeout"


class TestSLAComplianceLevelEnum:
    """Tests for SLAComplianceLevel enum."""

    def test_all_levels_defined(self):
        """Test that all compliance levels are defined."""
        expected = ["compliant", "warning", "critical", "violated"]
        for level in expected:
            assert SLAComplianceLevel(level) is not None

    def test_level_values(self):
        """Test compliance level enum values."""
        assert SLAComplianceLevel.COMPLIANT.value == "compliant"
        assert SLAComplianceLevel.WARNING.value == "warning"
        assert SLAComplianceLevel.CRITICAL.value == "critical"
        assert SLAComplianceLevel.VIOLATED.value == "violated"


class TestDeliberationSLA:
    """Tests for DeliberationSLA dataclass."""

    def test_sla_defaults(self):
        """Test SLA default values."""
        sla = DeliberationSLA()

        assert sla.timeout_seconds == DEFAULT_DELIBERATION_TIMEOUT
        assert sla.warning_threshold == DEFAULT_SLA_WARNING_THRESHOLD
        assert sla.critical_threshold == DEFAULT_SLA_CRITICAL_THRESHOLD
        assert sla.max_rounds == 5
        assert sla.min_agents == 2
        assert sla.consensus_required is True
        assert sla.notify_on_warning is True
        assert sla.notify_on_violation is True

    def test_custom_sla(self):
        """Test custom SLA configuration."""
        sla = DeliberationSLA(
            timeout_seconds=600.0,
            warning_threshold=0.7,
            critical_threshold=0.9,
            max_rounds=10,
            min_agents=3,
            consensus_required=False,
        )

        assert sla.timeout_seconds == 600.0
        assert sla.warning_threshold == 0.7
        assert sla.max_rounds == 10
        assert sla.consensus_required is False

    def test_get_compliance_level_compliant(self):
        """Test compliance level when within limits."""
        sla = DeliberationSLA(timeout_seconds=300.0)

        # 50% of timeout = 150s -> COMPLIANT
        level = sla.get_compliance_level(150.0)
        assert level == SLAComplianceLevel.COMPLIANT

    def test_get_compliance_level_warning(self):
        """Test compliance level at warning threshold."""
        sla = DeliberationSLA(
            timeout_seconds=300.0,
            warning_threshold=0.8,
        )

        # 85% of 300s = 255s -> WARNING
        level = sla.get_compliance_level(255.0)
        assert level == SLAComplianceLevel.WARNING

    def test_get_compliance_level_critical(self):
        """Test compliance level at critical threshold."""
        sla = DeliberationSLA(
            timeout_seconds=300.0,
            warning_threshold=0.8,
            critical_threshold=0.95,
        )

        # 96% of 300s = 288s -> CRITICAL
        level = sla.get_compliance_level(288.0)
        assert level == SLAComplianceLevel.CRITICAL

    def test_get_compliance_level_violated(self):
        """Test compliance level when timeout exceeded."""
        sla = DeliberationSLA(timeout_seconds=300.0)

        # 310s > 300s -> VIOLATED
        level = sla.get_compliance_level(310.0)
        assert level == SLAComplianceLevel.VIOLATED

    def test_compliance_boundary_exact_timeout(self):
        """Test compliance level at exact timeout."""
        sla = DeliberationSLA(timeout_seconds=300.0)

        level = sla.get_compliance_level(300.0)
        assert level == SLAComplianceLevel.VIOLATED


class TestDeliberationMetrics:
    """Tests for DeliberationMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating deliberation metrics."""
        import time

        now = time.time()
        metrics = DeliberationMetrics(
            started_at=now,
            completed_at=now + 120.5,
            rounds_completed=3,
            total_agent_responses=12,
            consensus_confidence=0.85,
        )

        assert metrics.rounds_completed == 3
        assert metrics.total_agent_responses == 12
        assert metrics.consensus_confidence == 0.85
        assert metrics.duration_seconds is not None
        assert abs(metrics.duration_seconds - 120.5) < 0.1

    def test_metrics_defaults(self):
        """Test metrics default values."""
        metrics = DeliberationMetrics()

        assert metrics.rounds_completed == 0
        assert metrics.total_agent_responses == 0
        assert metrics.consensus_confidence is None
        assert metrics.started_at is None
        assert metrics.duration_seconds is None

    def test_metrics_duration_property(self):
        """Test duration_seconds computed property."""
        import time

        now = time.time()
        metrics = DeliberationMetrics(started_at=now - 60.0)

        # Duration is computed from started_at to now
        assert metrics.duration_seconds is not None
        assert metrics.duration_seconds >= 60.0


class TestDeliberationTask:
    """Tests for DeliberationTask dataclass."""

    def test_task_creation(self):
        """Test creating a deliberation task."""
        task = DeliberationTask(
            question="Should we refactor the authentication module?",
            agents=["claude", "gpt-4", "gemini"],
            sla=DeliberationSLA(timeout_seconds=600.0),
            priority="high",
        )

        assert task.question == "Should we refactor the authentication module?"
        assert len(task.agents) == 3
        assert task.sla.timeout_seconds == 600.0
        assert task.priority == "high"
        # task_id and request_id are auto-generated
        assert task.task_id is not None
        assert task.request_id is not None

    def test_task_defaults(self):
        """Test task default values."""
        task = DeliberationTask(
            question="Test topic",
        )

        assert task.agents == []
        assert task.status == DeliberationStatus.PENDING
        assert task.sla is not None
        assert task.priority == "normal"
        assert task.result is None
        assert task.context is None

    def test_task_status_transitions(self):
        """Test task status can be updated."""
        task = DeliberationTask(
            question="Status test",
        )

        assert task.status == DeliberationStatus.PENDING

        task.status = DeliberationStatus.SCHEDULED
        assert task.status == DeliberationStatus.SCHEDULED

        task.status = DeliberationStatus.IN_PROGRESS
        assert task.status == DeliberationStatus.IN_PROGRESS

        task.status = DeliberationStatus.CONSENSUS_REACHED
        assert task.status == DeliberationStatus.CONSENSUS_REACHED

    def test_task_to_payload(self):
        """Test task payload serialization."""
        task = DeliberationTask(
            question="Serialization test",
            agents=["agent1"],
            priority="high",
            metadata={"source": "api"},
        )

        data = task.to_payload()
        assert data["question"] == "Serialization test"
        assert data["agents"] == ["agent1"]
        assert data["metadata"]["source"] == "api"

    def test_task_with_metrics(self):
        """Test task with completed metrics."""
        import time

        now = time.time()
        metrics = DeliberationMetrics(
            started_at=now - 240.0,
            completed_at=now,
            rounds_completed=4,
            total_agent_responses=12,
            consensus_confidence=0.88,
        )

        task = DeliberationTask(
            question="Metrics test",
            status=DeliberationStatus.CONSENSUS_REACHED,
            metrics=metrics,
        )

        assert task.metrics.rounds_completed == 4
        assert task.metrics.consensus_confidence == 0.88


class TestSLAComplianceTracking:
    """Integration tests for SLA compliance tracking."""

    def test_sla_progression(self):
        """Test SLA compliance as time progresses."""
        sla = DeliberationSLA(
            timeout_seconds=100.0,
            warning_threshold=0.8,
            critical_threshold=0.95,
        )

        # Start -> COMPLIANT
        assert sla.get_compliance_level(0.0) == SLAComplianceLevel.COMPLIANT

        # 50% -> COMPLIANT
        assert sla.get_compliance_level(50.0) == SLAComplianceLevel.COMPLIANT

        # 79% -> COMPLIANT (just under warning)
        assert sla.get_compliance_level(79.0) == SLAComplianceLevel.COMPLIANT

        # 80% -> WARNING
        assert sla.get_compliance_level(80.0) == SLAComplianceLevel.WARNING

        # 90% -> WARNING
        assert sla.get_compliance_level(90.0) == SLAComplianceLevel.WARNING

        # 95% -> CRITICAL
        assert sla.get_compliance_level(95.0) == SLAComplianceLevel.CRITICAL

        # 99% -> CRITICAL
        assert sla.get_compliance_level(99.0) == SLAComplianceLevel.CRITICAL

        # 100% -> VIOLATED
        assert sla.get_compliance_level(100.0) == SLAComplianceLevel.VIOLATED

        # 150% -> VIOLATED
        assert sla.get_compliance_level(150.0) == SLAComplianceLevel.VIOLATED


class TestDeliberationPriorities:
    """Tests for deliberation priority handling."""

    @pytest.mark.parametrize(
        "priority",
        ["low", "normal", "high", "urgent"],
    )
    def test_valid_priorities(self, priority):
        """Test that valid priority values work."""
        task = DeliberationTask(
            question="Priority test",
            priority=priority,
        )
        assert task.priority == priority
