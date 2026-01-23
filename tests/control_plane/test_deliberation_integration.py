"""
Tests for Deliberation-as-Task Integration.

Tests cover:
- Deliberation submission through coordinator
- SLA tracking and compliance
- ELO callback integration
- Notification callbacks
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.control_plane.deliberation import (
    AgentPerformance,
    DeliberationManager,
    DeliberationMetrics,
    DeliberationOutcome,
    DeliberationSLA,
    DeliberationStatus,
    DeliberationTask,
    DELIBERATION_TASK_TYPE,
    SLAComplianceLevel,
)


class TestDeliberationSLA:
    """Tests for DeliberationSLA."""

    def test_default_sla(self):
        """Test default SLA values."""
        sla = DeliberationSLA()

        assert sla.timeout_seconds == 300.0
        assert sla.warning_threshold == 0.8
        assert sla.critical_threshold == 0.95
        assert sla.max_rounds == 5

    def test_compliance_compliant(self):
        """Test compliant SLA level."""
        sla = DeliberationSLA(timeout_seconds=100.0)

        level = sla.get_compliance_level(50.0)  # 50% of timeout

        assert level == SLAComplianceLevel.COMPLIANT

    def test_compliance_warning(self):
        """Test warning SLA level."""
        sla = DeliberationSLA(timeout_seconds=100.0, warning_threshold=0.8)

        level = sla.get_compliance_level(85.0)  # 85% > 80% threshold

        assert level == SLAComplianceLevel.WARNING

    def test_compliance_critical(self):
        """Test critical SLA level."""
        sla = DeliberationSLA(timeout_seconds=100.0, critical_threshold=0.95)

        level = sla.get_compliance_level(96.0)  # 96% > 95% threshold

        assert level == SLAComplianceLevel.CRITICAL

    def test_compliance_violated(self):
        """Test violated SLA level."""
        sla = DeliberationSLA(timeout_seconds=100.0)

        level = sla.get_compliance_level(105.0)  # Over timeout

        assert level == SLAComplianceLevel.VIOLATED


class TestDeliberationTask:
    """Tests for DeliberationTask."""

    def test_create_task(self):
        """Test task creation."""
        task = DeliberationTask(
            question="What is the best approach?",
            context="We need to decide on architecture",
            agents=["claude", "gpt-4"],
        )

        assert task.question == "What is the best approach?"
        assert "claude" in task.agents
        assert task.status == DeliberationStatus.PENDING
        assert task.task_id is not None

    def test_to_payload(self):
        """Test payload generation."""
        task = DeliberationTask(
            question="Test question",
            sla=DeliberationSLA(timeout_seconds=120.0, max_rounds=3),
        )

        payload = task.to_payload()

        assert payload["question"] == "Test question"
        assert payload["sla"]["timeout_seconds"] == 120.0
        assert payload["sla"]["max_rounds"] == 3

    def test_from_payload_roundtrip(self):
        """Test payload roundtrip."""
        original = DeliberationTask(
            question="Roundtrip test",
            context="Some context",
            agents=["agent1", "agent2"],
            sla=DeliberationSLA(timeout_seconds=60.0),
        )

        payload = original.to_payload()
        restored = DeliberationTask.from_payload("task-123", payload)

        assert restored.task_id == "task-123"
        assert restored.question == original.question
        assert restored.context == original.context
        assert restored.sla.timeout_seconds == original.sla.timeout_seconds


class TestDeliberationMetrics:
    """Tests for DeliberationMetrics."""

    def test_duration_not_started(self):
        """Test duration when not started."""
        metrics = DeliberationMetrics()

        assert metrics.duration_seconds is None

    def test_duration_in_progress(self):
        """Test duration while in progress."""
        import time

        metrics = DeliberationMetrics(started_at=time.time() - 10.0)

        assert metrics.duration_seconds is not None
        assert metrics.duration_seconds >= 10.0

    def test_duration_completed(self):
        """Test duration when completed."""
        import time

        now = time.time()
        metrics = DeliberationMetrics(
            started_at=now - 30.0,
            completed_at=now - 10.0,
        )

        assert metrics.duration_seconds == 20.0


class TestDeliberationOutcome:
    """Tests for DeliberationOutcome."""

    def test_successful_outcome(self):
        """Test successful outcome creation."""
        outcome = DeliberationOutcome(
            task_id="task-123",
            request_id="req-456",
            success=True,
            consensus_reached=True,
            consensus_confidence=0.85,
            winning_position="Option A is better",
            duration_seconds=45.0,
            sla_compliant=True,
        )

        assert outcome.success is True
        assert outcome.consensus_reached is True
        assert outcome.consensus_confidence == 0.85

    def test_outcome_with_agent_performances(self):
        """Test outcome with agent performance metrics."""
        outcome = DeliberationOutcome(
            task_id="task-123",
            request_id="req-456",
            success=True,
            consensus_reached=True,
            agent_performances={
                "claude": AgentPerformance(
                    agent_id="claude",
                    contributed_to_consensus=True,
                    response_count=3,
                    final_position_correct=True,
                ),
                "gpt-4": AgentPerformance(
                    agent_id="gpt-4",
                    contributed_to_consensus=False,
                    response_count=3,
                    final_position_correct=False,
                ),
            },
        )

        assert len(outcome.agent_performances) == 2
        assert outcome.agent_performances["claude"].contributed_to_consensus is True


class TestDeliberationManager:
    """Tests for DeliberationManager."""

    @pytest.fixture
    def mock_coordinator(self):
        """Create a mock coordinator."""
        coordinator = AsyncMock()
        coordinator.submit_task = AsyncMock(return_value="task-123")
        coordinator.wait_for_result = AsyncMock(
            return_value=MagicMock(
                status=MagicMock(value="completed"),
                result={"consensus_reached": True, "confidence": 0.9},
            )
        )
        return coordinator

    @pytest.fixture
    def manager(self, mock_coordinator):
        """Create a manager with mock coordinator."""
        return DeliberationManager(coordinator=mock_coordinator)

    @pytest.mark.asyncio
    async def test_submit_deliberation(self, manager, mock_coordinator):
        """Test submitting a deliberation."""
        task_id = await manager.submit_deliberation(
            question="What approach should we use?",
            agents=["claude", "gpt-4"],
            priority="high",
            timeout_seconds=120.0,
        )

        assert task_id == "task-123"
        mock_coordinator.submit_task.assert_called_once()

        # Verify task type
        call_kwargs = mock_coordinator.submit_task.call_args.kwargs
        assert call_kwargs["task_type"] == DELIBERATION_TASK_TYPE

    @pytest.mark.asyncio
    async def test_submit_requires_coordinator(self):
        """Test error when coordinator not set."""
        manager = DeliberationManager()

        with pytest.raises(RuntimeError, match="Coordinator not set"):
            await manager.submit_deliberation(question="Test?")

    @pytest.mark.asyncio
    async def test_wait_for_outcome(self, manager, mock_coordinator):
        """Test waiting for deliberation outcome."""
        # First submit
        task_id = await manager.submit_deliberation(question="Test?")

        # Then wait
        outcome = await manager.wait_for_outcome(task_id)

        assert outcome is not None
        assert outcome.success is True
        assert outcome.consensus_reached is True

    def test_get_active_deliberations(self, manager):
        """Test listing active deliberations."""
        # Add some tasks to internal tracking
        manager._active_deliberations["task-1"] = DeliberationTask(
            question="Q1",
            status=DeliberationStatus.IN_PROGRESS,
        )
        manager._active_deliberations["task-2"] = DeliberationTask(
            question="Q2",
            status=DeliberationStatus.CONSENSUS_REACHED,
        )
        manager._active_deliberations["task-3"] = DeliberationTask(
            question="Q3",
            status=DeliberationStatus.SCHEDULED,
        )

        active = manager.get_active_deliberations()

        assert len(active) == 2  # IN_PROGRESS and SCHEDULED
        assert any(t.question == "Q1" for t in active)
        assert any(t.question == "Q3" for t in active)

    def test_get_deliberation_stats(self, manager):
        """Test statistics retrieval."""
        manager._active_deliberations["task-1"] = DeliberationTask(
            question="Q1",
            status=DeliberationStatus.IN_PROGRESS,
        )
        manager._active_deliberations["task-1"].metrics.sla_compliance = SLAComplianceLevel.VIOLATED

        stats = manager.get_deliberation_stats()

        assert stats["total_active"] == 1
        assert stats["sla_violations"] == 1


class TestEloCallback:
    """Tests for ELO callback integration."""

    def test_elo_callback_called(self):
        """Test that ELO callback is invoked on completion."""
        callback_called = False
        callback_outcome = None

        def elo_callback(outcome):
            nonlocal callback_called, callback_outcome
            callback_called = True
            callback_outcome = outcome

        manager = DeliberationManager(elo_callback=elo_callback)

        # Simulate outcome
        outcome = DeliberationOutcome(
            task_id="task-123",
            request_id="req-456",
            success=True,
            consensus_reached=True,
            consensus_confidence=0.9,
        )

        # Call the callback manually (normally done by execute_deliberation)
        if manager._elo_callback:
            manager._elo_callback(outcome)

        assert callback_called is True
        assert callback_outcome.task_id == "task-123"


class TestNotificationCallback:
    """Tests for notification callback integration."""

    def test_notification_callback_called(self):
        """Test that notification callback is invoked."""
        notifications = []

        def notification_callback(event_type: str, data: dict):
            notifications.append((event_type, data))

        manager = DeliberationManager(notification_callback=notification_callback)

        # Simulate SLA warning notification
        if manager._notification_callback:
            manager._notification_callback(
                "sla_warning",
                {
                    "task_id": "task-123",
                    "elapsed_seconds": 240.0,
                    "timeout_seconds": 300.0,
                },
            )

        assert len(notifications) == 1
        assert notifications[0][0] == "sla_warning"
        assert notifications[0][1]["task_id"] == "task-123"


class TestDeliberationStatus:
    """Tests for DeliberationStatus enum."""

    def test_all_statuses(self):
        """Test all status values exist."""
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


class TestAgentPerformance:
    """Tests for AgentPerformance dataclass."""

    def test_default_performance(self):
        """Test default performance values."""
        perf = AgentPerformance(agent_id="test-agent")

        assert perf.agent_id == "test-agent"
        assert perf.contributed_to_consensus is False
        assert perf.response_count == 0
        assert perf.average_confidence == 0.0
        assert perf.position_changed is False
        assert perf.final_position_correct is False

    def test_performance_with_values(self):
        """Test performance with custom values."""
        perf = AgentPerformance(
            agent_id="claude",
            contributed_to_consensus=True,
            response_count=5,
            average_confidence=0.85,
            position_changed=True,
            final_position_correct=True,
        )

        assert perf.contributed_to_consensus is True
        assert perf.response_count == 5
        assert perf.average_confidence == 0.85
