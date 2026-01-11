"""
Tests for Transparent Immune System module.

Tests cover:
- HealthStatus enum values
- AgentStatus enum values
- HealthEvent dataclass operations
- AgentHealthState tracking
- TransparentImmuneSystem lifecycle methods
- System health status transitions
- Broadcasting and event history
- Global singleton access
"""

import time
import pytest
from unittest.mock import Mock, patch

from aragora.debate.immune_system import (
    HealthStatus,
    AgentStatus,
    HealthEvent,
    AgentHealthState,
    TransparentImmuneSystem,
    get_immune_system,
    reset_immune_system,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global immune system before and after each test."""
    reset_immune_system()
    yield
    reset_immune_system()


@pytest.fixture
def immune_system():
    """Create a fresh immune system for testing."""
    return TransparentImmuneSystem()


@pytest.fixture
def health_event():
    """Create a sample health event."""
    return HealthEvent(
        timestamp=1234567890.0,
        event_type="test_event",
        status="healthy",
        component="test_agent",
        message="Test message",
        details={"key": "value"},
        audience_message="Human-friendly message",
    )


# ============================================================================
# HealthStatus Enum Tests
# ============================================================================

class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_healthy_value(self):
        """Test HEALTHY status value."""
        assert HealthStatus.HEALTHY.value == "healthy"

    def test_degraded_value(self):
        """Test DEGRADED status value."""
        assert HealthStatus.DEGRADED.value == "degraded"

    def test_stressed_value(self):
        """Test STRESSED status value."""
        assert HealthStatus.STRESSED.value == "stressed"

    def test_critical_value(self):
        """Test CRITICAL status value."""
        assert HealthStatus.CRITICAL.value == "critical"

    def test_recovering_value(self):
        """Test RECOVERING status value."""
        assert HealthStatus.RECOVERING.value == "recovering"

    def test_all_statuses_unique(self):
        """Test all status values are unique."""
        values = [s.value for s in HealthStatus]
        assert len(values) == len(set(values))


# ============================================================================
# AgentStatus Enum Tests
# ============================================================================

class TestAgentStatus:
    """Tests for AgentStatus enum."""

    def test_idle_value(self):
        """Test IDLE status value."""
        assert AgentStatus.IDLE.value == "idle"

    def test_thinking_value(self):
        """Test THINKING status value."""
        assert AgentStatus.THINKING.value == "thinking"

    def test_responding_value(self):
        """Test RESPONDING status value."""
        assert AgentStatus.RESPONDING.value == "responding"

    def test_timeout_value(self):
        """Test TIMEOUT status value."""
        assert AgentStatus.TIMEOUT.value == "timeout"

    def test_failed_value(self):
        """Test FAILED status value."""
        assert AgentStatus.FAILED.value == "failed"

    def test_recovered_value(self):
        """Test RECOVERED status value."""
        assert AgentStatus.RECOVERED.value == "recovered"

    def test_circuit_open_value(self):
        """Test CIRCUIT_OPEN status value."""
        assert AgentStatus.CIRCUIT_OPEN.value == "circuit_open"


# ============================================================================
# HealthEvent Dataclass Tests
# ============================================================================

class TestHealthEvent:
    """Tests for HealthEvent dataclass."""

    def test_create_event(self, health_event):
        """Test creating a health event."""
        assert health_event.event_type == "test_event"
        assert health_event.component == "test_agent"
        assert health_event.message == "Test message"

    def test_to_dict(self, health_event):
        """Test converting event to dictionary."""
        data = health_event.to_dict()

        assert isinstance(data, dict)
        assert data["timestamp"] == 1234567890.0
        assert data["event_type"] == "test_event"
        assert data["status"] == "healthy"
        assert data["component"] == "test_agent"
        assert data["message"] == "Test message"
        assert data["details"]["key"] == "value"
        assert data["audience_message"] == "Human-friendly message"

    def test_to_broadcast(self, health_event):
        """Test formatting event for WebSocket broadcast."""
        broadcast = health_event.to_broadcast()

        assert broadcast["type"] == "health_event"
        assert "data" in broadcast
        assert broadcast["data"]["event_type"] == "test_event"

    def test_event_without_audience_message(self):
        """Test event without audience message."""
        event = HealthEvent(
            timestamp=time.time(),
            event_type="test",
            status="ok",
            component="agent",
            message="Internal message",
            details={},
        )
        assert event.audience_message is None
        data = event.to_dict()
        assert data["audience_message"] is None


# ============================================================================
# AgentHealthState Dataclass Tests
# ============================================================================

class TestAgentHealthState:
    """Tests for AgentHealthState dataclass."""

    def test_create_state(self):
        """Test creating an agent health state."""
        state = AgentHealthState(name="claude")

        assert state.name == "claude"
        assert state.status == AgentStatus.IDLE
        assert state.consecutive_failures == 0
        assert state.total_timeouts == 0
        assert state.circuit_open is False

    def test_to_dict(self):
        """Test converting state to dictionary."""
        state = AgentHealthState(
            name="gpt4",
            status=AgentStatus.THINKING,
            consecutive_failures=2,
            avg_response_ms=1234.5678,
        )
        data = state.to_dict()

        assert data["name"] == "gpt4"
        assert data["status"] == "thinking"
        assert data["consecutive_failures"] == 2
        assert data["avg_response_ms"] == 1234.57  # Rounded

    def test_state_with_circuit_open(self):
        """Test state with circuit breaker open."""
        state = AgentHealthState(
            name="failing_agent",
            status=AgentStatus.CIRCUIT_OPEN,
            circuit_open=True,
        )
        data = state.to_dict()

        assert data["circuit_open"] is True
        assert data["status"] == "circuit_open"


# ============================================================================
# TransparentImmuneSystem Tests
# ============================================================================

class TestTransparentImmuneSystem:
    """Tests for TransparentImmuneSystem class."""

    def test_initialization(self, immune_system):
        """Test immune system initialization."""
        assert immune_system.system_status == HealthStatus.HEALTHY
        assert len(immune_system.agent_states) == 0
        assert len(immune_system.event_history) == 0
        assert immune_system.total_events == 0

    def test_set_broadcast_callback(self, immune_system):
        """Test setting broadcast callback."""
        callback = Mock()
        immune_system.set_broadcast_callback(callback)

        assert immune_system.broadcast_callback == callback

    def test_agent_started(self, immune_system):
        """Test agent_started event."""
        callback = Mock()
        immune_system.set_broadcast_callback(callback)

        immune_system.agent_started("claude", task="analyze code")

        # Check agent state
        state = immune_system.agent_states["claude"]
        assert state.status == AgentStatus.THINKING

        # Check broadcast was called
        callback.assert_called_once()
        event_data = callback.call_args[0][0]
        assert event_data["type"] == "health_event"
        assert event_data["data"]["event_type"] == "agent_started"

    def test_agent_completed(self, immune_system):
        """Test agent_completed event."""
        immune_system.agent_started("claude")
        immune_system.agent_completed("claude", response_ms=1500)

        state = immune_system.agent_states["claude"]
        assert state.status == AgentStatus.IDLE
        assert state.avg_response_ms > 0
        assert state.consecutive_failures == 0

    def test_agent_timeout(self, immune_system):
        """Test agent_timeout event."""
        immune_system.agent_started("slow_agent")
        immune_system.agent_timeout("slow_agent", timeout_seconds=90.0)

        state = immune_system.agent_states["slow_agent"]
        assert state.status == AgentStatus.TIMEOUT
        assert state.consecutive_failures == 1
        assert state.total_timeouts == 1
        assert immune_system.total_failures == 1

    def test_agent_failed(self, immune_system):
        """Test agent_failed event."""
        immune_system.agent_started("broken_agent")
        immune_system.agent_failed("broken_agent", error="API error", recoverable=True)

        state = immune_system.agent_states["broken_agent"]
        assert state.status == AgentStatus.FAILED
        assert state.consecutive_failures == 1

    def test_agent_recovered(self, immune_system):
        """Test agent_recovered event."""
        immune_system.agent_started("recovering_agent")
        immune_system.agent_failed("recovering_agent", error="temp error")
        immune_system.agent_recovered("recovering_agent", "fallback")

        state = immune_system.agent_states["recovering_agent"]
        assert state.status == AgentStatus.RECOVERED
        assert immune_system.total_recoveries == 1

    def test_circuit_opened(self, immune_system):
        """Test circuit_opened event."""
        immune_system.circuit_opened("overloaded_agent", reason="too many failures")

        state = immune_system.agent_states["overloaded_agent"]
        assert state.status == AgentStatus.CIRCUIT_OPEN
        assert state.circuit_open is True

    def test_circuit_closed(self, immune_system):
        """Test circuit_closed event."""
        immune_system.circuit_opened("agent", reason="failures")
        immune_system.circuit_closed("agent")

        state = immune_system.agent_states["agent"]
        assert state.status == AgentStatus.IDLE
        assert state.circuit_open is False
        assert state.consecutive_failures == 0

    def test_system_event(self, immune_system):
        """Test generic system event."""
        callback = Mock()
        immune_system.set_broadcast_callback(callback)

        immune_system.system_event(
            "debate_started",
            "Debate initiated",
            details={"topic": "AI safety"},
            audience_message="Welcome to the debate!",
        )

        callback.assert_called_once()
        event_data = callback.call_args[0][0]["data"]
        assert event_data["event_type"] == "debate_started"
        assert event_data["component"] == "system"

    def test_get_system_health(self, immune_system):
        """Test getting system health summary."""
        immune_system.agent_started("agent1")
        immune_system.agent_completed("agent1", 500)
        immune_system.agent_started("agent2")
        immune_system.agent_timeout("agent2", 30.0)

        health = immune_system.get_system_health()

        assert "status" in health
        assert "uptime_seconds" in health
        assert "agents" in health
        assert len(health["agents"]) == 2

    def test_get_recent_events(self, immune_system):
        """Test getting recent events."""
        immune_system.agent_started("a1")
        immune_system.agent_completed("a1", 100)
        immune_system.agent_started("a2")

        events = immune_system.get_recent_events(limit=2)

        assert len(events) == 2
        assert all(isinstance(e, dict) for e in events)


# ============================================================================
# System Status Update Tests
# ============================================================================

class TestSystemStatusUpdates:
    """Tests for system status transitions."""

    def test_healthy_with_no_agents(self, immune_system):
        """Test healthy status with no agents."""
        immune_system._update_system_status()
        assert immune_system.system_status == HealthStatus.HEALTHY

    def test_healthy_with_all_working(self, immune_system):
        """Test healthy status when all agents work."""
        immune_system.agent_started("a1")
        immune_system.agent_completed("a1", 100)
        immune_system.agent_started("a2")
        immune_system.agent_completed("a2", 100)

        assert immune_system.system_status == HealthStatus.HEALTHY

    def test_degraded_with_one_failure(self, immune_system):
        """Test degraded status with one failed agent."""
        immune_system.agent_started("a1")
        immune_system.agent_completed("a1", 100)
        immune_system.agent_started("a2")
        immune_system.agent_timeout("a2", 30.0)

        assert immune_system.system_status == HealthStatus.DEGRADED

    def test_critical_with_majority_failures(self, immune_system):
        """Test critical status when majority of agents fail."""
        immune_system.agent_started("a1")
        immune_system.agent_timeout("a1", 30.0)
        immune_system.agent_started("a2")
        immune_system.agent_failed("a2", "error")

        assert immune_system.system_status == HealthStatus.CRITICAL


# ============================================================================
# Broadcasting Tests
# ============================================================================

class TestBroadcasting:
    """Tests for event broadcasting."""

    def test_broadcast_callback_receives_events(self, immune_system):
        """Test that callback receives all events."""
        events = []
        immune_system.set_broadcast_callback(lambda e: events.append(e))

        immune_system.agent_started("test")
        immune_system.agent_completed("test", 500)

        assert len(events) == 2

    def test_broadcast_callback_error_handling(self, immune_system):
        """Test that broadcast errors don't crash the system."""
        def failing_callback(event):
            raise RuntimeError("Broadcast failed")

        immune_system.set_broadcast_callback(failing_callback)

        # Should not raise
        immune_system.agent_started("test")

        # Event should still be recorded
        assert len(immune_system.event_history) == 1

    def test_event_history_limit(self, immune_system):
        """Test that event history is limited to prevent memory leaks."""
        # Generate more than 1000 events
        for i in range(1100):
            immune_system.agent_started(f"agent_{i}")

        # History should be trimmed
        assert len(immune_system.event_history) <= 1000


# ============================================================================
# Agent Progress Tests
# ============================================================================

class TestAgentProgress:
    """Tests for agent progress reporting."""

    def test_progress_at_threshold(self, immune_system):
        """Test progress broadcast at timeout thresholds."""
        events = []
        immune_system.set_broadcast_callback(lambda e: events.append(e))

        immune_system.agent_started("slow")
        immune_system.agent_progress("slow", 5.0)  # First threshold

        # Should broadcast at 5 second threshold
        progress_events = [e for e in events if e["data"]["event_type"] == "agent_progress"]
        assert len(progress_events) == 1

    def test_progress_updates_status(self, immune_system):
        """Test progress updates agent status to RESPONDING."""
        immune_system.agent_started("agent")
        immune_system.agent_progress("agent", 10.0)

        state = immune_system.agent_states["agent"]
        assert state.status == AgentStatus.RESPONDING


# ============================================================================
# Global Singleton Tests
# ============================================================================

class TestGlobalSingleton:
    """Tests for global immune system singleton."""

    def test_get_immune_system_returns_singleton(self):
        """Test that get_immune_system returns same instance."""
        instance1 = get_immune_system()
        instance2 = get_immune_system()

        assert instance1 is instance2

    def test_reset_immune_system(self):
        """Test resetting global immune system."""
        instance1 = get_immune_system()
        instance1.agent_started("test")

        reset_immune_system()

        instance2 = get_immune_system()
        assert instance1 is not instance2
        assert len(instance2.agent_states) == 0


# ============================================================================
# Rolling Average Tests
# ============================================================================

class TestRollingAverage:
    """Tests for response time rolling average."""

    def test_rolling_average_calculation(self, immune_system):
        """Test rolling average smoothing.

        Uses exponential moving average with alpha=0.3:
        - First: 0.3 * 1000 + 0.7 * 0 = 300
        - Second: 0.3 * 2000 + 0.7 * 300 = 810
        """
        immune_system.agent_started("agent")
        immune_system.agent_completed("agent", 1000)

        state = immune_system.agent_states["agent"]
        assert state.avg_response_ms == 300.0  # 0.3 * 1000 + 0.7 * 0

        immune_system.agent_started("agent")
        immune_system.agent_completed("agent", 2000)

        # 0.3 * 2000 + 0.7 * 300 = 810
        assert state.avg_response_ms == 810.0
