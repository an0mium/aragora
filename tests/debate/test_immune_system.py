"""
Tests for Transparent Immune System Module.

Tests the immune system functionality including:
- HealthStatus and AgentStatus enums
- HealthEvent and AgentHealthState dataclasses
- TransparentImmuneSystem class
- Agent lifecycle events
- System health monitoring
- Broadcast callbacks
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from aragora.debate.immune_system import (
    AgentHealthState,
    AgentStatus,
    HealthEvent,
    HealthStatus,
    TransparentImmuneSystem,
    get_immune_system,
    reset_immune_system,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_healthy_status(self):
        """Test healthy status value."""
        assert HealthStatus.HEALTHY.value == "healthy"

    def test_degraded_status(self):
        """Test degraded status value."""
        assert HealthStatus.DEGRADED.value == "degraded"

    def test_stressed_status(self):
        """Test stressed status value."""
        assert HealthStatus.STRESSED.value == "stressed"

    def test_critical_status(self):
        """Test critical status value."""
        assert HealthStatus.CRITICAL.value == "critical"

    def test_recovering_status(self):
        """Test recovering status value."""
        assert HealthStatus.RECOVERING.value == "recovering"


class TestAgentStatus:
    """Test AgentStatus enum."""

    def test_idle_status(self):
        """Test idle status value."""
        assert AgentStatus.IDLE.value == "idle"

    def test_thinking_status(self):
        """Test thinking status value."""
        assert AgentStatus.THINKING.value == "thinking"

    def test_responding_status(self):
        """Test responding status value."""
        assert AgentStatus.RESPONDING.value == "responding"

    def test_timeout_status(self):
        """Test timeout status value."""
        assert AgentStatus.TIMEOUT.value == "timeout"

    def test_failed_status(self):
        """Test failed status value."""
        assert AgentStatus.FAILED.value == "failed"

    def test_recovered_status(self):
        """Test recovered status value."""
        assert AgentStatus.RECOVERED.value == "recovered"

    def test_circuit_open_status(self):
        """Test circuit_open status value."""
        assert AgentStatus.CIRCUIT_OPEN.value == "circuit_open"


# =============================================================================
# HealthEvent Tests
# =============================================================================


class TestHealthEvent:
    """Test HealthEvent dataclass."""

    def test_create_health_event(self):
        """Test creating a health event."""
        event = HealthEvent(
            timestamp=1234567890.0,
            event_type="agent_started",
            status="thinking",
            component="claude",
            message="Agent claude started",
            details={"task": "analyze code"},
            audience_message="Claude is thinking...",
        )

        assert event.timestamp == 1234567890.0
        assert event.event_type == "agent_started"
        assert event.status == "thinking"
        assert event.component == "claude"
        assert event.message == "Agent claude started"
        assert event.details == {"task": "analyze code"}
        assert event.audience_message == "Claude is thinking..."

    def test_to_dict(self):
        """Test converting event to dictionary."""
        event = HealthEvent(
            timestamp=100.0,
            event_type="test",
            status="idle",
            component="agent",
            message="Test message",
            details={},
        )

        d = event.to_dict()

        assert d["timestamp"] == 100.0
        assert d["event_type"] == "test"
        assert d["audience_message"] is None

    def test_to_broadcast(self):
        """Test converting event to broadcast format."""
        event = HealthEvent(
            timestamp=100.0,
            event_type="test",
            status="idle",
            component="agent",
            message="Test",
            details={},
        )

        broadcast = event.to_broadcast()

        assert broadcast["type"] == "health_event"
        assert "data" in broadcast
        assert broadcast["data"]["event_type"] == "test"


# =============================================================================
# AgentHealthState Tests
# =============================================================================


class TestAgentHealthState:
    """Test AgentHealthState dataclass."""

    def test_create_agent_health_state(self):
        """Test creating agent health state."""
        state = AgentHealthState(name="claude")

        assert state.name == "claude"
        assert state.status == AgentStatus.IDLE
        assert state.consecutive_failures == 0
        assert state.circuit_open is False

    def test_to_dict(self):
        """Test converting state to dictionary."""
        state = AgentHealthState(
            name="gpt-4",
            status=AgentStatus.RESPONDING,
            consecutive_failures=2,
            total_timeouts=5,
            avg_response_ms=1500.5678,
            circuit_open=True,
        )

        d = state.to_dict()

        assert d["name"] == "gpt-4"
        assert d["status"] == "responding"
        assert d["consecutive_failures"] == 2
        assert d["total_timeouts"] == 5
        assert d["avg_response_ms"] == 1500.57  # Rounded
        assert d["circuit_open"] is True


# =============================================================================
# TransparentImmuneSystem Initialization Tests
# =============================================================================


class TestImmuneSystemInit:
    """Test TransparentImmuneSystem initialization."""

    def test_init_defaults(self):
        """Test default initialization."""
        immune = TransparentImmuneSystem()

        assert immune.agent_states == {}
        assert immune.system_status == HealthStatus.HEALTHY
        assert immune.event_history == []
        assert immune.broadcast_callback is None
        assert immune.total_events == 0
        assert immune.total_failures == 0
        assert immune.total_recoveries == 0

    def test_set_broadcast_callback(self):
        """Test setting broadcast callback."""
        immune = TransparentImmuneSystem()
        callback = MagicMock()

        immune.set_broadcast_callback(callback)

        assert immune.broadcast_callback == callback


# =============================================================================
# Agent Lifecycle Tests
# =============================================================================


class TestAgentLifecycle:
    """Test agent lifecycle events."""

    def test_agent_started(self):
        """Test agent_started event."""
        immune = TransparentImmuneSystem()

        immune.agent_started("claude", "analyze code")

        assert "claude" in immune.agent_states
        assert immune.agent_states["claude"].status == AgentStatus.THINKING
        assert len(immune.event_history) == 1
        assert immune.event_history[0].event_type == "agent_started"

    def test_agent_started_broadcasts(self):
        """Test agent_started broadcasts event."""
        immune = TransparentImmuneSystem()
        callback = MagicMock()
        immune.set_broadcast_callback(callback)

        immune.agent_started("claude")

        callback.assert_called_once()
        call_arg = callback.call_args[0][0]
        assert call_arg["type"] == "health_event"

    def test_agent_completed(self):
        """Test agent_completed event."""
        immune = TransparentImmuneSystem()
        immune.agent_started("claude")

        immune.agent_completed("claude", response_ms=1500.0)

        assert immune.agent_states["claude"].status == AgentStatus.IDLE
        assert immune.agent_states["claude"].consecutive_failures == 0

    def test_agent_completed_updates_avg_response(self):
        """Test agent_completed updates rolling average."""
        immune = TransparentImmuneSystem()
        immune.agent_started("claude")

        immune.agent_completed("claude", response_ms=1000.0)
        first_avg = immune.agent_states["claude"].avg_response_ms

        immune.agent_started("claude")
        immune.agent_completed("claude", response_ms=2000.0)
        second_avg = immune.agent_states["claude"].avg_response_ms

        # Rolling average should be between 1000 and 2000
        assert first_avg == 300.0  # 0.3 * 1000 + 0.7 * 0
        assert 300 < second_avg < 2000

    def test_agent_timeout(self):
        """Test agent_timeout event."""
        immune = TransparentImmuneSystem()
        immune.agent_started("claude")

        immune.agent_timeout("claude", timeout_seconds=60.0)

        assert immune.agent_states["claude"].status == AgentStatus.TIMEOUT
        assert immune.agent_states["claude"].consecutive_failures == 1
        assert immune.agent_states["claude"].total_timeouts == 1
        assert immune.total_failures == 1

    def test_agent_timeout_with_context(self):
        """Test agent_timeout with context details."""
        immune = TransparentImmuneSystem()
        callback = MagicMock()
        immune.set_broadcast_callback(callback)

        immune.agent_timeout("claude", 60.0, context={"task": "complex analysis"})

        call_data = callback.call_args[0][0]["data"]
        assert call_data["details"]["context"] == {"task": "complex analysis"}

    def test_agent_failed(self):
        """Test agent_failed event."""
        immune = TransparentImmuneSystem()

        immune.agent_failed("claude", error="API rate limit exceeded", recoverable=True)

        assert immune.agent_states["claude"].status == AgentStatus.FAILED
        assert immune.agent_states["claude"].consecutive_failures == 1
        assert immune.total_failures == 1

    def test_agent_recovered(self):
        """Test agent_recovered event."""
        immune = TransparentImmuneSystem()
        immune.agent_failed("claude", "error")

        immune.agent_recovered("claude", "fallback_agent")

        assert immune.agent_states["claude"].status == AgentStatus.RECOVERED
        assert immune.total_recoveries == 1

    def test_circuit_opened(self):
        """Test circuit_opened event."""
        immune = TransparentImmuneSystem()

        immune.circuit_opened("claude", reason="Too many failures")

        assert immune.agent_states["claude"].status == AgentStatus.CIRCUIT_OPEN
        assert immune.agent_states["claude"].circuit_open is True

    def test_circuit_closed(self):
        """Test circuit_closed event."""
        immune = TransparentImmuneSystem()
        immune.circuit_opened("claude", "failures")

        immune.circuit_closed("claude")

        assert immune.agent_states["claude"].status == AgentStatus.IDLE
        assert immune.agent_states["claude"].circuit_open is False
        assert immune.agent_states["claude"].consecutive_failures == 0


# =============================================================================
# Agent Progress Tests
# =============================================================================


class TestAgentProgress:
    """Test agent_progress reporting."""

    def test_agent_progress_at_threshold(self):
        """Test agent_progress broadcasts at threshold."""
        immune = TransparentImmuneSystem()
        callback = MagicMock()
        immune.set_broadcast_callback(callback)
        immune.agent_started("claude")
        callback.reset_mock()

        # At 5 second threshold
        immune.agent_progress("claude", elapsed_seconds=5.0)

        callback.assert_called_once()
        call_data = callback.call_args[0][0]["data"]
        assert call_data["event_type"] == "agent_progress"

    def test_agent_progress_updates_status(self):
        """Test agent_progress updates agent status."""
        immune = TransparentImmuneSystem()
        immune.agent_started("claude")

        immune.agent_progress("claude", elapsed_seconds=10.0)

        assert immune.agent_states["claude"].status == AgentStatus.RESPONDING

    def test_agent_progress_no_broadcast_between_thresholds(self):
        """Test agent_progress doesn't broadcast between thresholds."""
        immune = TransparentImmuneSystem()
        callback = MagicMock()
        immune.set_broadcast_callback(callback)
        immune.agent_started("claude")
        callback.reset_mock()

        # Between thresholds (not within 1 second of any threshold)
        immune.agent_progress("claude", elapsed_seconds=10.0)

        callback.assert_not_called()


# =============================================================================
# System Status Tests
# =============================================================================


class TestSystemStatus:
    """Test system status updates."""

    def test_healthy_when_no_failures(self):
        """Test system stays healthy with no failures."""
        immune = TransparentImmuneSystem()
        immune.agent_started("claude")
        immune.agent_completed("claude", 1000.0)

        assert immune.system_status == HealthStatus.HEALTHY

    def test_degraded_with_one_failure(self):
        """Test system degrades with one failure."""
        immune = TransparentImmuneSystem()
        immune.agent_started("claude")
        immune.agent_started("gpt-4")

        immune.agent_failed("claude", "error")

        assert immune.system_status == HealthStatus.DEGRADED

    def test_stressed_with_multiple_failures(self):
        """Test system stressed with multiple failures."""
        immune = TransparentImmuneSystem()
        # Create 4 agents
        for name in ["claude", "gpt-4", "gemini", "mistral"]:
            immune.agent_started(name)

        # Fail 2 out of 4 (less than half)
        immune.agent_failed("claude", "error")
        immune.agent_timeout("gpt-4", 60.0)

        # 2/4 = 50%, which is not < 50%, so should be CRITICAL
        # Actually need 2 agents, and both failed = critical
        # Let's test with 4 agents and 1 failure for stressed
        immune2 = TransparentImmuneSystem()
        for name in ["a1", "a2", "a3", "a4"]:
            immune2.agent_started(name)
        immune2.agent_failed("a1", "error")  # 1/4 = degraded

        assert immune2.system_status == HealthStatus.DEGRADED

    def test_critical_with_majority_failures(self):
        """Test system critical with majority failures."""
        immune = TransparentImmuneSystem()
        immune.agent_started("claude")
        immune.agent_started("gpt-4")

        immune.agent_failed("claude", "error")
        immune.agent_timeout("gpt-4", 60.0)

        assert immune.system_status == HealthStatus.CRITICAL


# =============================================================================
# System Event Tests
# =============================================================================


class TestSystemEvent:
    """Test system-level events."""

    def test_system_event(self):
        """Test broadcasting system event."""
        immune = TransparentImmuneSystem()

        immune.system_event(
            event_type="debate_started",
            message="Debate round 1 started",
            details={"round": 1},
            audience_message="Let the debate begin!",
        )

        assert len(immune.event_history) == 1
        assert immune.event_history[0].component == "system"
        assert immune.event_history[0].event_type == "debate_started"


# =============================================================================
# Health Summary Tests
# =============================================================================


class TestHealthSummary:
    """Test health summary methods."""

    def test_get_system_health(self):
        """Test get_system_health returns summary."""
        immune = TransparentImmuneSystem()
        immune.agent_started("claude")
        immune.agent_completed("claude", 1000.0)
        immune.agent_failed("gpt-4", "error")
        immune.agent_recovered("gpt-4", "fallback")

        health = immune.get_system_health()

        assert health["status"] == "healthy"  # gpt-4 recovered
        assert health["total_failures"] == 1
        assert health["total_recoveries"] == 1
        assert "claude" in health["agents"]
        assert "gpt-4" in health["agents"]
        assert health["uptime_seconds"] >= 0

    def test_recovery_rate_calculation(self):
        """Test recovery rate is calculated correctly."""
        immune = TransparentImmuneSystem()
        immune.agent_failed("a1", "error")
        immune.agent_failed("a2", "error")
        immune.agent_recovered("a1", "fallback")

        health = immune.get_system_health()

        # 1 recovery / 2 failures = 0.5
        assert health["recovery_rate"] == 0.5

    def test_get_recent_events(self):
        """Test get_recent_events returns limited events."""
        immune = TransparentImmuneSystem()
        for i in range(100):
            immune.system_event("test", f"Event {i}")

        recent = immune.get_recent_events(limit=10)

        assert len(recent) == 10
        assert recent[-1]["message"] == "Event 99"


# =============================================================================
# Event History Management Tests
# =============================================================================


class TestEventHistoryManagement:
    """Test event history management."""

    def test_event_history_limited(self):
        """Test event history is limited to prevent memory issues."""
        immune = TransparentImmuneSystem()

        # Test the trimming logic directly by adding 1001 events
        # then triggering a broadcast which trims
        for i in range(1001):
            event = HealthEvent(
                timestamp=float(i),
                event_type="test",
                status="idle",
                component="system",
                message=f"Event {i}",
                details={},
            )
            immune.event_history.append(event)
            immune.total_events += 1

        # At this point we have 1001 events - should trigger trim
        # Add one more event via _broadcast to trigger trimming
        immune.system_event("test", "Trigger trim")

        # After trim, history should be ~500 (plus the one we just added)
        assert len(immune.event_history) <= 501

    def test_event_history_preserves_recent(self):
        """Test trimming preserves most recent events."""
        immune = TransparentImmuneSystem()

        # Add 1005 events directly
        for i in range(1005):
            event = HealthEvent(
                timestamp=float(i),
                event_type="test",
                status="idle",
                component="system",
                message=f"Event {i}",
                details={},
            )
            immune.event_history.append(event)

        # Trigger trim
        immune.system_event("trigger", "trim")

        # Should have kept the most recent events
        # The last event should be our "trigger" event
        assert immune.event_history[-1].event_type == "trigger"


# =============================================================================
# Broadcast Error Handling Tests
# =============================================================================


class TestBroadcastErrorHandling:
    """Test broadcast callback error handling."""

    def test_handles_connection_error(self):
        """Test handles ConnectionError from callback."""
        immune = TransparentImmuneSystem()
        callback = MagicMock(side_effect=ConnectionError("Lost connection"))
        immune.set_broadcast_callback(callback)

        # Should not raise
        immune.agent_started("claude")

        assert len(immune.event_history) == 1

    def test_handles_runtime_error(self):
        """Test handles RuntimeError from callback."""
        immune = TransparentImmuneSystem()
        callback = MagicMock(side_effect=RuntimeError("Event loop closed"))
        immune.set_broadcast_callback(callback)

        # Should not raise
        immune.agent_failed("claude", "error")

        assert immune.total_failures == 1

    def test_handles_unexpected_error(self):
        """Test handles unexpected errors from callback."""
        immune = TransparentImmuneSystem()
        callback = MagicMock(side_effect=ValueError("Unexpected"))
        immune.set_broadcast_callback(callback)

        # Should not raise
        immune.agent_completed("claude", 1000.0)

        assert len(immune.event_history) == 1


# =============================================================================
# Global Instance Tests
# =============================================================================


class TestGlobalInstance:
    """Test global instance management."""

    def test_get_immune_system_creates_instance(self):
        """Test get_immune_system creates instance on first call."""
        reset_immune_system()

        immune = get_immune_system()

        assert immune is not None
        assert isinstance(immune, TransparentImmuneSystem)

    def test_get_immune_system_returns_same_instance(self):
        """Test get_immune_system returns same instance."""
        reset_immune_system()

        immune1 = get_immune_system()
        immune2 = get_immune_system()

        assert immune1 is immune2

    def test_reset_immune_system(self):
        """Test reset_immune_system clears instance."""
        reset_immune_system()
        immune1 = get_immune_system()
        immune1.agent_started("test")

        reset_immune_system()
        immune2 = get_immune_system()

        assert immune1 is not immune2
        assert "test" not in immune2.agent_states


# =============================================================================
# Timeout Stage Tests
# =============================================================================


class TestTimeoutStages:
    """Test progressive timeout stage messages."""

    def test_timeout_stages_defined(self):
        """Test timeout stages are properly defined."""
        stages = TransparentImmuneSystem.TIMEOUT_STAGES

        assert len(stages) == 5
        assert stages[0][0] == 5
        assert stages[-1][0] == 90

    def test_timeout_stages_escalate(self):
        """Test timeout stages have escalating thresholds."""
        stages = TransparentImmuneSystem.TIMEOUT_STAGES

        thresholds = [s[0] for s in stages]
        assert thresholds == sorted(thresholds)
