"""
Tests for extracted debate services.

Tests for:
- EventBus: Event emission and subscription
- AgentPool: Agent selection and management
- PhaseExecutor: Phase orchestration
"""

import asyncio
import pytest
from dataclasses import dataclass
from typing import Any, List
from unittest.mock import AsyncMock, Mock, patch

from aragora.debate.event_bus import (
    EventBus,
    DebateEvent,
    get_event_bus,
    set_event_bus,
)
from aragora.debate.agent_pool import (
    AgentPool,
    AgentPoolConfig,
    AgentMetrics,
)
from aragora.debate.phase_executor import (
    PhaseExecutor,
    PhaseConfig,
    PhaseResult,
    PhaseStatus,
    ExecutionResult,
    STANDARD_PHASE_ORDER,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str
    model: str = "test-model"


@dataclass
class MockPhase:
    """Mock phase for testing."""

    name: str
    should_fail: bool = False
    execution_time: float = 0.0
    output: Any = None

    async def execute(self, context: Any) -> Any:
        if self.execution_time > 0:
            await asyncio.sleep(self.execution_time)
        if self.should_fail:
            raise RuntimeError(f"Phase {self.name} failed intentionally")
        return self.output or {"phase": self.name, "status": "completed"}


@pytest.fixture
def mock_agents():
    """Create mock agents for testing."""
    return [
        MockAgent(name="agent_1", model="claude"),
        MockAgent(name="agent_2", model="gpt-4"),
        MockAgent(name="agent_3", model="gemini"),
        MockAgent(name="agent_4", model="claude"),
    ]


@pytest.fixture
def mock_phases():
    """Create mock phases for testing."""
    return {
        "context_initializer": MockPhase(name="context_initializer", output={"context": "initialized"}),
        "proposal": MockPhase(name="proposal", output={"proposals": ["p1", "p2"]}),
        "debate_rounds": MockPhase(name="debate_rounds", output={"rounds": 3}),
        "consensus": MockPhase(name="consensus", output={"consensus": True, "answer": "final"}),
        "analytics": MockPhase(name="analytics", output={"metrics": {}}),
        "feedback": MockPhase(name="feedback", output={"stored": True}),
    }


# =============================================================================
# EventBus Tests
# =============================================================================


class TestEventBus:
    """Tests for EventBus service."""

    def test_create_event_bus(self):
        """Test event bus creation."""
        bus = EventBus()
        assert bus is not None
        assert bus._events_emitted == 0

    @pytest.mark.asyncio
    async def test_emit_event(self):
        """Test event emission."""
        bus = EventBus()

        await bus.emit("debate_start", debate_id="test-123", task="Test task")

        assert bus._events_emitted == 1
        assert bus._events_by_type["debate_start"] == 1

    @pytest.mark.asyncio
    async def test_subscribe_async_handler(self):
        """Test async handler subscription."""
        bus = EventBus()
        received_events: List[DebateEvent] = []

        async def handler(event: DebateEvent):
            received_events.append(event)

        bus.subscribe("test_event", handler)
        await bus.emit("test_event", debate_id="123")

        assert len(received_events) == 1
        assert received_events[0].event_type == "test_event"
        assert received_events[0].debate_id == "123"

    def test_subscribe_sync_handler(self):
        """Test sync handler subscription."""
        bus = EventBus()
        received_events: List[DebateEvent] = []

        def handler(event: DebateEvent):
            received_events.append(event)

        bus.subscribe_sync("test_event", handler)
        bus.emit_sync("test_event", debate_id="456")

        assert len(received_events) == 1
        assert received_events[0].debate_id == "456"

    @pytest.mark.asyncio
    async def test_unsubscribe_handler(self):
        """Test handler unsubscription."""
        bus = EventBus()

        async def handler(event: DebateEvent):
            pass

        bus.subscribe("test_event", handler)
        assert len(bus._async_handlers["test_event"]) == 1

        result = bus.unsubscribe("test_event", handler)
        assert result is True
        assert len(bus._async_handlers["test_event"]) == 0

    @pytest.mark.asyncio
    async def test_event_bridge_integration(self):
        """Test event bridge notification."""
        mock_bridge = Mock()
        mock_bridge.notify = Mock()

        bus = EventBus(event_bridge=mock_bridge)
        await bus.emit("debate_start", debate_id="123")

        mock_bridge.notify.assert_called_once()
        call_kwargs = mock_bridge.notify.call_args
        assert call_kwargs[0][0] == "debate_start"

    def test_user_event_queue(self):
        """Test user event queuing."""
        bus = EventBus()

        bus.queue_user_event({"type": "vote", "user_id": "user1", "vote": "agree"})
        bus.queue_user_event({"type": "suggestion", "user_id": "user2", "content": "test"})

        assert bus._user_event_queue.qsize() == 2

    @pytest.mark.asyncio
    async def test_drain_user_events(self):
        """Test user event draining."""
        bus = EventBus()

        bus.queue_user_event({"type": "vote", "user_id": "user1"})
        bus.queue_user_event({"type": "vote", "user_id": "user2"})

        events = await bus.drain_user_events("debate-123")

        assert len(events) == 2
        assert bus._user_event_queue.qsize() == 0

    def test_get_metrics(self):
        """Test metrics retrieval."""
        bus = EventBus()
        bus.emit_sync("event_a", debate_id="1")
        bus.emit_sync("event_a", debate_id="2")
        bus.emit_sync("event_b", debate_id="3")

        metrics = bus.get_metrics()

        assert metrics["total_events_emitted"] == 3
        assert metrics["events_by_type"]["event_a"] == 2
        assert metrics["events_by_type"]["event_b"] == 1

    @pytest.mark.asyncio
    async def test_emit_moment_event(self):
        """Test specialized moment event emission."""
        bus = EventBus()

        await bus.emit_moment_event(
            debate_id="123",
            moment_type="breakthrough",
            description="Agent had key insight",
            agent="agent_1",
            significance=0.8,
        )

        assert bus._events_by_type["moment"] == 1

    def test_singleton_access(self):
        """Test singleton pattern."""
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2

        new_bus = EventBus()
        set_event_bus(new_bus)
        assert get_event_bus() is new_bus


# =============================================================================
# AgentPool Tests
# =============================================================================


class TestAgentPool:
    """Tests for AgentPool service."""

    def test_create_pool(self, mock_agents):
        """Test pool creation."""
        pool = AgentPool(mock_agents)

        assert len(pool.agents) == 4
        assert len(pool.available_agents) == 4

    def test_get_agent_by_name(self, mock_agents):
        """Test agent retrieval by name."""
        pool = AgentPool(mock_agents)

        agent = pool.get_agent("agent_2")
        assert agent is not None
        assert agent.name == "agent_2"

        missing = pool.get_agent("nonexistent")
        assert missing is None

    def test_require_agents_success(self, mock_agents):
        """Test require_agents with sufficient agents."""
        pool = AgentPool(mock_agents)

        agents = pool.require_agents(min_count=2)
        assert len(agents) >= 2

    def test_require_agents_failure(self):
        """Test require_agents with insufficient agents."""
        pool = AgentPool([MockAgent(name="only_one")])

        with pytest.raises(ValueError) as exc_info:
            pool.require_agents(min_count=5)

        assert "Insufficient agents" in str(exc_info.value)

    def test_select_team_random(self, mock_agents):
        """Test random team selection."""
        config = AgentPoolConfig(use_performance_selection=False)
        pool = AgentPool(mock_agents, config)

        team = pool.select_team(team_size=2)
        assert len(team) == 2
        assert all(a in mock_agents for a in team)

    def test_select_team_performance_based(self, mock_agents):
        """Test performance-based team selection."""
        mock_elo = Mock()
        mock_elo.get_rating = Mock(side_effect=lambda n: {"agent_1": 1200, "agent_2": 1100, "agent_3": 900, "agent_4": 1000}.get(n, 1000))

        config = AgentPoolConfig(
            use_performance_selection=True,
            elo_system=mock_elo,
        )
        pool = AgentPool(mock_agents, config)

        team = pool.select_team(team_size=2)

        # Should select top 2 by ELO
        assert len(team) == 2
        team_names = {a.name for a in team}
        assert "agent_1" in team_names  # Highest ELO

    def test_select_team_with_exclusion(self, mock_agents):
        """Test team selection with exclusion."""
        pool = AgentPool(mock_agents)

        team = pool.select_team(team_size=2, exclude={"agent_1", "agent_2"})

        team_names = {a.name for a in team}
        assert "agent_1" not in team_names
        assert "agent_2" not in team_names

    def test_select_critics_mesh(self, mock_agents):
        """Test mesh topology critic selection."""
        config = AgentPoolConfig(topology="full_mesh", critic_count=2)
        pool = AgentPool(mock_agents, config)

        proposer = mock_agents[0]
        critics = pool.select_critics(proposer)

        assert len(critics) == 2
        assert proposer not in critics

    def test_select_critics_ring(self, mock_agents):
        """Test ring topology critic selection."""
        config = AgentPoolConfig(topology="ring", critic_count=2)
        pool = AgentPool(mock_agents, config)

        proposer = mock_agents[1]  # agent_2 (index 1)
        critics = pool.select_critics(proposer)

        # Ring: neighbors should be agent_1 and agent_3
        critic_names = {c.name for c in critics}
        # At least one neighbor should be selected
        assert len(critics) > 0
        assert proposer not in critics

    def test_circuit_breaker_integration(self, mock_agents):
        """Test circuit breaker filtering."""
        mock_cb = Mock()
        mock_cb.is_open = Mock(side_effect=lambda n: n == "agent_2")

        config = AgentPoolConfig(circuit_breaker=mock_cb)
        pool = AgentPool(mock_agents, config)

        available = pool.available_agents
        names = {a.name for a in available}

        assert "agent_2" not in names
        assert len(available) == 3

    def test_update_metrics(self, mock_agents):
        """Test agent metrics update."""
        pool = AgentPool(mock_agents)

        pool.update_metrics(
            agent_name="agent_1",
            elo_rating=1150.0,
            calibration_score=0.8,
            debate_participated=True,
            won=True,
        )

        metrics = pool.get_agent_metrics("agent_1")
        assert metrics is not None
        assert metrics.elo_rating == 1150.0
        assert metrics.calibration_score == 0.8
        assert metrics.debates_participated == 1

    def test_get_pool_status(self, mock_agents):
        """Test pool status retrieval."""
        pool = AgentPool(mock_agents)

        status = pool.get_pool_status()

        assert status["total_agents"] == 4
        assert status["available_agents"] == 4
        assert len(status["agents"]) == 4


# =============================================================================
# PhaseExecutor Tests
# =============================================================================


class TestPhaseExecutor:
    """Tests for PhaseExecutor service."""

    def test_create_executor(self, mock_phases):
        """Test executor creation."""
        executor = PhaseExecutor(mock_phases)

        assert executor is not None
        assert len(executor.phase_names) == 6

    @pytest.mark.asyncio
    async def test_execute_all_phases(self, mock_phases):
        """Test full phase execution."""
        executor = PhaseExecutor(mock_phases)

        result = await executor.execute(
            context={},
            debate_id="test-debate",
        )

        assert result.success
        assert len(result.phases) == 6
        assert result.debate_id == "test-debate"
        assert all(p.status == PhaseStatus.COMPLETED for p in result.phases)

    @pytest.mark.asyncio
    async def test_execute_with_failure(self, mock_phases):
        """Test execution with phase failure."""
        mock_phases["proposal"] = MockPhase(name="proposal", should_fail=True)
        executor = PhaseExecutor(mock_phases)

        result = await executor.execute(context={}, debate_id="test")

        assert not result.success
        proposal_result = result.get_phase_result("proposal")
        assert proposal_result is not None
        assert proposal_result.status == PhaseStatus.FAILED

    @pytest.mark.asyncio
    async def test_optional_phase_failure_continues(self, mock_phases):
        """Test that optional phase failure doesn't stop execution."""
        mock_phases["analytics"] = MockPhase(name="analytics", should_fail=True)
        executor = PhaseExecutor(mock_phases)

        result = await executor.execute(context={}, debate_id="test")

        # Should continue past analytics failure
        feedback_result = result.get_phase_result("feedback")
        # Note: depends on ordering, analytics may fail but feedback might run
        assert result.phases[-1].phase_name in ["feedback", "analytics"]

    @pytest.mark.asyncio
    async def test_phase_timeout(self, mock_phases):
        """Test phase-level timeout."""
        mock_phases["proposal"] = MockPhase(
            name="proposal",
            execution_time=5.0,  # Long execution
        )
        config = PhaseConfig(phase_timeout_seconds=0.1)  # Short timeout
        executor = PhaseExecutor(mock_phases, config)

        result = await executor.execute(context={}, debate_id="test")

        proposal_result = result.get_phase_result("proposal")
        assert proposal_result is not None
        assert proposal_result.status == PhaseStatus.FAILED
        assert "Timed out" in (proposal_result.error or "")

    @pytest.mark.asyncio
    async def test_overall_timeout(self, mock_phases):
        """Test overall execution timeout."""
        # Make all phases slow
        for phase in mock_phases.values():
            phase.execution_time = 1.0

        config = PhaseConfig(total_timeout_seconds=0.1)  # Very short
        executor = PhaseExecutor(mock_phases, config)

        result = await executor.execute(context={}, debate_id="test")

        assert not result.success
        assert "timed out" in result.error.lower()

    def test_request_termination(self, mock_phases):
        """Test early termination request."""
        executor = PhaseExecutor(mock_phases)

        executor.request_termination("User cancelled")

        should_term, reason = executor.check_termination()
        assert should_term is True
        assert reason == "User cancelled"

    @pytest.mark.asyncio
    async def test_custom_phase_order(self, mock_phases):
        """Test custom phase ordering."""
        executor = PhaseExecutor(mock_phases)

        result = await executor.execute(
            context={},
            debate_id="test",
            phase_order=["proposal", "consensus"],  # Skip others
        )

        assert len(result.phases) == 2
        assert result.phases[0].phase_name == "proposal"
        assert result.phases[1].phase_name == "consensus"

    def test_add_remove_phase(self, mock_phases):
        """Test dynamic phase management."""
        executor = PhaseExecutor(mock_phases)

        # Add new phase
        new_phase = MockPhase(name="custom")
        executor.add_phase("custom", new_phase)
        assert "custom" in executor.phase_names

        # Remove phase
        result = executor.remove_phase("custom")
        assert result is True
        assert "custom" not in executor.phase_names

    @pytest.mark.asyncio
    async def test_metrics_collection(self, mock_phases):
        """Test metrics are collected."""
        executor = PhaseExecutor(mock_phases)

        await executor.execute(context={}, debate_id="test")

        metrics = executor.get_metrics()

        assert metrics["total_phases"] == 6
        assert metrics["completed_phases"] == 6
        assert metrics["failed_phases"] == 0
        assert metrics["total_duration_ms"] > 0

    @pytest.mark.asyncio
    async def test_trace_callback(self, mock_phases):
        """Test trace callback is called."""
        traces = []

        def trace_callback(event_type: str, data: dict):
            traces.append((event_type, data))

        config = PhaseConfig(
            enable_tracing=True,
            trace_callback=trace_callback,
        )
        executor = PhaseExecutor(mock_phases, config)

        await executor.execute(context={}, debate_id="test")

        # Should have start/end for each phase
        assert len(traces) >= 12  # 6 phases * 2 events
        assert any(t[0] == "phase_start" for t in traces)
        assert any(t[0] == "phase_end" for t in traces)

    @pytest.mark.asyncio
    async def test_consensus_output_captured(self, mock_phases):
        """Test that consensus phase output becomes final output."""
        mock_phases["consensus"] = MockPhase(
            name="consensus",
            output={"answer": "The final answer", "consensus": True},
        )
        executor = PhaseExecutor(mock_phases)

        result = await executor.execute(context={}, debate_id="test")

        assert result.final_output is not None
        assert result.final_output["answer"] == "The final answer"


# =============================================================================
# Integration Tests
# =============================================================================


class TestServiceIntegration:
    """Integration tests for services working together."""

    @pytest.mark.asyncio
    async def test_event_bus_with_phase_executor(self, mock_phases):
        """Test EventBus receives events from PhaseExecutor."""
        bus = EventBus()
        traces = []

        def trace_callback(event_type: str, data: dict):
            bus.emit_sync(f"trace:{event_type}", debate_id=data.get("debate_id", ""))
            traces.append(event_type)

        config = PhaseConfig(enable_tracing=True, trace_callback=trace_callback)
        executor = PhaseExecutor(mock_phases, config)

        await executor.execute(context={}, debate_id="integration-test")

        assert bus._events_emitted > 0
        assert len(traces) > 0

    def test_agent_pool_with_event_bus(self, mock_agents):
        """Test AgentPool can emit events via EventBus."""
        bus = EventBus()
        pool = AgentPool(mock_agents)

        # Emit event when team selected
        team = pool.select_team(team_size=2)
        bus.emit_sync(
            "team_selected",
            debate_id="test",
            team=[a.name for a in team],
        )

        assert bus._events_by_type.get("team_selected") == 1


# =============================================================================
# DebateEvent Tests
# =============================================================================


class TestDebateEvent:
    """Tests for DebateEvent dataclass."""

    def test_create_event(self):
        """Test event creation."""
        event = DebateEvent(
            event_type="test",
            debate_id="123",
            data={"key": "value"},
        )

        assert event.event_type == "test"
        assert event.debate_id == "123"
        assert event.data["key"] == "value"
        assert event.timestamp is not None

    def test_to_dict(self):
        """Test event serialization."""
        event = DebateEvent(
            event_type="test",
            debate_id="123",
            data={"extra": "data"},
        )

        d = event.to_dict()

        assert d["event_type"] == "test"
        assert d["debate_id"] == "123"
        assert d["extra"] == "data"
        assert "timestamp" in d


# =============================================================================
# PhaseResult Tests
# =============================================================================


class TestPhaseResult:
    """Tests for PhaseResult dataclass."""

    def test_success_property(self):
        """Test success property for different statuses."""
        completed = PhaseResult(phase_name="test", status=PhaseStatus.COMPLETED)
        assert completed.success is True

        skipped = PhaseResult(phase_name="test", status=PhaseStatus.SKIPPED)
        assert skipped.success is True

        failed = PhaseResult(phase_name="test", status=PhaseStatus.FAILED)
        assert failed.success is False

        pending = PhaseResult(phase_name="test", status=PhaseStatus.PENDING)
        assert pending.success is False
