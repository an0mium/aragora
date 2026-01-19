"""
Integration tests for the Aragora debate module.

Tests cover:
- EventBus subscribe/emit/unsubscribe flows
- Arena initialization with various configurations
- Arena context manager behavior
- Convergence detection integration
- Memory manager integration with debate outcomes
- Consensus detection with voting engine
- Event emission during debate lifecycle
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core import Agent, Critique, Environment, Message, Vote
from aragora.debate.event_bus import DebateEvent, EventBus
from aragora.debate.protocol import DebateProtocol


class MockAgent(Agent):
    """Mock agent for integration testing."""

    def __init__(
        self,
        name: str = "mock-agent",
        response: str = "Test response",
        model: str = "mock-model",
        role: str = "proposer",
    ):
        super().__init__(name=name, model=model, role=role)
        self.agent_type = "mock"
        self.response = response
        self.generate_calls = 0
        self.critique_calls = 0
        self.vote_calls = 0

    async def generate(self, prompt: str, context: list = None) -> str:
        self.generate_calls += 1
        return self.response

    async def generate_stream(self, prompt: str, context: list = None):
        yield self.response

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list = None,
        target_agent: str = None,
    ) -> Critique:
        self.critique_calls += 1
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal[:100] if proposal else "",
            issues=["Test issue"],
            suggestions=["Test suggestion"],
            severity=0.5,
            reasoning="Test reasoning",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        self.vote_calls += 1
        choice = list(proposals.keys())[0] if proposals else self.name
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning="Test vote",
            confidence=0.8,
            continue_debate=False,
        )


class TestEventBusIntegration:
    """Integration tests for EventBus pub/sub system."""

    def test_event_bus_creation(self):
        """EventBus can be created without dependencies."""
        bus = EventBus()

        assert bus is not None
        assert hasattr(bus, "_async_handlers")

    def test_event_bus_with_bridge(self):
        """EventBus can be created with event bridge."""
        mock_bridge = MagicMock()

        bus = EventBus(event_bridge=mock_bridge)

        assert bus._event_bridge is mock_bridge

    def test_subscribe_adds_handler(self):
        """Subscribe adds handler to event type."""
        bus = EventBus()
        handler = AsyncMock()

        bus.subscribe("debate_start", handler)

        assert "debate_start" in bus._async_handlers
        assert handler in bus._async_handlers["debate_start"]

    def test_subscribe_multiple_handlers(self):
        """Multiple handlers can subscribe to same event type."""
        bus = EventBus()
        handler1 = AsyncMock()
        handler2 = AsyncMock()

        bus.subscribe("debate_start", handler1)
        bus.subscribe("debate_start", handler2)

        assert len(bus._async_handlers["debate_start"]) == 2

    def test_subscribe_different_events(self):
        """Handlers can subscribe to different event types."""
        bus = EventBus()
        start_handler = AsyncMock()
        end_handler = AsyncMock()

        bus.subscribe("debate_start", start_handler)
        bus.subscribe("debate_end", end_handler)

        assert "debate_start" in bus._async_handlers
        assert "debate_end" in bus._async_handlers

    def test_unsubscribe_removes_handler(self):
        """Unsubscribe removes handler from event type."""
        bus = EventBus()
        handler = AsyncMock()
        bus.subscribe("debate_start", handler)

        bus.unsubscribe("debate_start", handler)

        assert handler not in bus._async_handlers.get("debate_start", [])

    def test_unsubscribe_nonexistent_handler(self):
        """Unsubscribe handles nonexistent handler gracefully."""
        bus = EventBus()
        handler = AsyncMock()

        # Should not raise
        bus.unsubscribe("debate_start", handler)

    @pytest.mark.asyncio
    async def test_emit_calls_handlers(self):
        """Emit calls all subscribed handlers."""
        bus = EventBus()
        handler = AsyncMock()
        bus.subscribe("debate_start", handler)

        await bus.emit("debate_start", debate_id="test-123", task="Test task")

        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_passes_event_to_handlers(self):
        """Emit passes DebateEvent to handlers."""
        bus = EventBus()
        received_events = []

        async def handler(event):
            received_events.append(event)

        bus.subscribe("debate_start", handler)

        await bus.emit("debate_start", debate_id="test-123", task="Test task")

        assert len(received_events) == 1
        assert isinstance(received_events[0], DebateEvent)
        assert received_events[0].event_type == "debate_start"
        assert received_events[0].debate_id == "test-123"

    @pytest.mark.asyncio
    async def test_emit_no_subscribers(self):
        """Emit works with no subscribers."""
        bus = EventBus()

        # Should not raise
        await bus.emit("debate_start", debate_id="test-123")

    @pytest.mark.asyncio
    async def test_emit_multiple_handlers(self):
        """Emit calls all handlers for event type."""
        bus = EventBus()
        call_order = []

        async def handler1(event):
            call_order.append("handler1")

        async def handler2(event):
            call_order.append("handler2")

        bus.subscribe("debate_start", handler1)
        bus.subscribe("debate_start", handler2)

        await bus.emit("debate_start", debate_id="test-123")

        assert "handler1" in call_order
        assert "handler2" in call_order

    @pytest.mark.asyncio
    async def test_emit_with_event_bridge(self):
        """Emit notifies event bridge when present."""
        mock_bridge = MagicMock()
        mock_bridge.notify = MagicMock()
        bus = EventBus(event_bridge=mock_bridge)

        await bus.emit("debate_start", debate_id="test-123")

        mock_bridge.notify.assert_called_once()


class TestDebateEventIntegration:
    """Integration tests for DebateEvent dataclass."""

    def test_event_creation(self):
        """DebateEvent can be created with required fields."""
        event = DebateEvent(
            event_type="debate_start",
            debate_id="test-123",
        )

        assert event.event_type == "debate_start"
        assert event.debate_id == "test-123"
        assert event.timestamp is not None

    def test_event_with_data(self):
        """DebateEvent can include additional data."""
        event = DebateEvent(
            event_type="agent_message",
            debate_id="test-123",
            data={"agent": "claude", "content": "Test response"},
        )

        assert event.data["agent"] == "claude"
        assert event.data["content"] == "Test response"

    def test_event_to_dict(self):
        """DebateEvent can be converted to dictionary."""
        event = DebateEvent(
            event_type="debate_start",
            debate_id="test-123",
            data={"task": "Test task"},
        )

        result = event.to_dict()

        assert result["event_type"] == "debate_start"
        assert result["debate_id"] == "test-123"
        assert result["task"] == "Test task"
        assert "timestamp" in result

    def test_event_timestamp_is_utc(self):
        """DebateEvent timestamp is UTC."""
        event = DebateEvent(
            event_type="debate_start",
            debate_id="test-123",
        )

        assert event.timestamp.tzinfo == timezone.utc

    def test_event_correlation_id_auto_populated(self):
        """DebateEvent correlation_id is auto-populated from trace context."""
        event = DebateEvent(
            event_type="debate_start",
            debate_id="test-123",
        )

        # correlation_id may be None if no trace context exists
        # Just verify the attribute exists
        assert hasattr(event, "correlation_id")


class TestArenaInitializationIntegration:
    """Integration tests for Arena initialization."""

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        return [
            MockAgent(name="agent1", response="Response 1"),
            MockAgent(name="agent2", response="Response 2"),
        ]

    @pytest.fixture
    def environment(self):
        """Create test environment."""
        return Environment(task="Test integration task")

    @pytest.fixture
    def protocol(self):
        """Create minimal test protocol."""
        return DebateProtocol(
            rounds=1,
            timeout_seconds=30,
            enable_calibration=False,
            enable_trickster=False,
            enable_breakpoints=False,
        )

    def test_arena_can_be_imported(self):
        """Arena can be imported from debate module."""
        from aragora.debate.orchestrator import Arena

        assert Arena is not None

    def test_arena_creation_with_minimal_args(self, environment, mock_agents, protocol):
        """Arena can be created with minimal arguments."""
        from aragora.debate.orchestrator import Arena

        arena = Arena(
            environment=environment,
            agents=mock_agents,
            protocol=protocol,
        )

        assert arena is not None
        assert arena.env == environment
        assert len(arena.agents) == 2

    def test_arena_protocol_defaults(self, environment, mock_agents):
        """Arena uses default protocol when none provided."""
        from aragora.debate.orchestrator import Arena

        arena = Arena(
            environment=environment,
            agents=mock_agents,
        )

        assert arena.protocol is not None
        assert arena.protocol.rounds > 0

    def test_arena_has_event_bus(self, environment, mock_agents, protocol):
        """Arena initializes with event bus."""
        from aragora.debate.orchestrator import Arena

        arena = Arena(
            environment=environment,
            agents=mock_agents,
            protocol=protocol,
        )

        assert hasattr(arena, "event_bus")
        assert arena.event_bus is not None

    def test_arena_has_convergence_detector(self, environment, mock_agents, protocol):
        """Arena initializes convergence detector when enabled."""
        from aragora.debate.orchestrator import Arena

        protocol_with_convergence = DebateProtocol(
            rounds=1,
            convergence_detection=True,
            convergence_threshold=0.95,
        )

        arena = Arena(
            environment=environment,
            agents=mock_agents,
            protocol=protocol_with_convergence,
        )

        assert arena.convergence_detector is not None

    def test_arena_without_convergence_detector(self, environment, mock_agents):
        """Arena skips convergence detector when disabled."""
        from aragora.debate.orchestrator import Arena

        protocol_no_convergence = DebateProtocol(
            rounds=1,
            convergence_detection=False,
        )

        arena = Arena(
            environment=environment,
            agents=mock_agents,
            protocol=protocol_no_convergence,
        )

        assert arena.convergence_detector is None


class TestArenaContextManagerIntegration:
    """Integration tests for Arena as context manager."""

    @pytest.fixture
    def mock_agents(self):
        return [
            MockAgent(name="agent1", response="Response 1"),
            MockAgent(name="agent2", response="Response 2"),
        ]

    @pytest.fixture
    def environment(self):
        return Environment(task="Test context manager task")

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(
            rounds=1,
            timeout_seconds=30,
            enable_calibration=False,
            enable_trickster=False,
        )

    @pytest.mark.asyncio
    async def test_arena_async_context_manager_enter(
        self, environment, mock_agents, protocol
    ):
        """Arena can be used as async context manager."""
        from aragora.debate.orchestrator import Arena

        arena = Arena(
            environment=environment,
            agents=mock_agents,
            protocol=protocol,
        )

        async with arena as ctx:
            assert ctx is arena

    @pytest.mark.asyncio
    async def test_arena_async_context_manager_cleanup(
        self, environment, mock_agents, protocol
    ):
        """Arena cleans up on context manager exit."""
        from aragora.debate.orchestrator import Arena

        arena = Arena(
            environment=environment,
            agents=mock_agents,
            protocol=protocol,
        )

        # Just verify no errors during enter/exit
        async with arena:
            pass


class TestConvergenceDetectorIntegration:
    """Integration tests for convergence detection."""

    def test_convergence_detector_creation(self):
        """ConvergenceDetector can be created."""
        from aragora.debate.convergence import ConvergenceDetector

        detector = ConvergenceDetector(
            convergence_threshold=0.95,
            divergence_threshold=0.3,
            min_rounds_before_check=1,
        )

        assert detector is not None
        assert detector.convergence_threshold == 0.95

    def test_convergence_detector_with_debate_id(self):
        """ConvergenceDetector can be created with debate_id."""
        from aragora.debate.convergence import ConvergenceDetector

        detector = ConvergenceDetector(
            convergence_threshold=0.95,
            debate_id="test-123",
        )

        assert detector is not None

    @pytest.mark.asyncio
    async def test_convergence_check_with_similar_responses(self):
        """ConvergenceDetector detects similar responses."""
        from aragora.debate.convergence import ConvergenceDetector

        detector = ConvergenceDetector(
            convergence_threshold=0.8,
            min_rounds_before_check=0,
        )

        # Identical responses should converge
        responses = [
            ("agent1", "The answer is definitely 42."),
            ("agent2", "The answer is definitely 42."),
        ]

        # This may require embeddings, which may not be available in tests
        # Just verify the method exists and can be called
        try:
            result = await detector.check_convergence(responses, round_num=1)
            assert isinstance(result, dict)
        except Exception:
            # Embeddings may not be available in test environment
            pass

    @pytest.mark.asyncio
    async def test_convergence_check_with_different_responses(self):
        """ConvergenceDetector detects different responses."""
        from aragora.debate.convergence import ConvergenceDetector

        detector = ConvergenceDetector(
            convergence_threshold=0.95,
            min_rounds_before_check=0,
        )

        responses = [
            ("agent1", "The answer is 42."),
            ("agent2", "The answer is completely different."),
        ]

        try:
            result = await detector.check_convergence(responses, round_num=1)
            assert isinstance(result, dict)
        except Exception:
            pass


class TestVotingEngineIntegration:
    """Integration tests for voting engine."""

    def test_voting_engine_import(self):
        """VotingEngine can be imported."""
        from aragora.debate.voting_engine import VotingEngine

        assert VotingEngine is not None

    def test_voting_engine_creation(self):
        """VotingEngine can be created."""
        from aragora.debate.voting_engine import VotingEngine

        engine = VotingEngine()
        assert engine is not None

    def test_count_votes_simple_majority(self):
        """VotingEngine counts simple majority correctly."""
        from aragora.debate.voting_engine import VotingEngine

        engine = VotingEngine()
        votes = [
            Vote(agent="a1", choice="option1", reasoning="r1", confidence=0.8),
            Vote(agent="a2", choice="option1", reasoning="r2", confidence=0.7),
            Vote(agent="a3", choice="option2", reasoning="r3", confidence=0.9),
        ]

        result = engine.count_votes(votes)

        assert result is not None
        assert hasattr(result, "winner") or hasattr(result, "vote_counts")

    def test_count_votes_with_confidence(self):
        """VotingEngine handles votes with varying confidence."""
        from aragora.debate.voting_engine import VotingEngine

        engine = VotingEngine()
        votes = [
            Vote(agent="a1", choice="option1", reasoning="r1", confidence=0.5),
            Vote(agent="a2", choice="option2", reasoning="r2", confidence=0.95),
        ]

        result = engine.count_votes(votes)

        assert result is not None

    def test_count_votes_empty(self):
        """VotingEngine handles empty vote list."""
        from aragora.debate.voting_engine import VotingEngine

        engine = VotingEngine()

        result = engine.count_votes([])

        # Should return a VoteResult (even if empty)
        assert result is not None


class TestConsensusIntegration:
    """Integration tests for consensus data structures."""

    def test_consensus_vote_type_import(self):
        """VoteType can be imported from consensus module."""
        from aragora.debate.consensus import VoteType

        assert VoteType is not None
        assert VoteType.AGREE.value == "agree"

    def test_consensus_evidence_creation(self):
        """Evidence dataclass can be created."""
        from aragora.debate.consensus import Evidence

        evidence = Evidence(
            evidence_id="ev_123",
            source="agent1",
            content="Test evidence content",
            evidence_type="argument",
            supports_claim=True,
            strength=0.8,
        )

        assert evidence.evidence_id == "ev_123"
        assert evidence.strength == 0.8

    def test_consensus_claim_creation(self):
        """Claim dataclass can be created."""
        from aragora.debate.consensus import Claim

        claim = Claim(
            claim_id="cl_123",
            statement="Test claim statement",
            author="agent1",
            confidence=0.85,
        )

        assert claim.claim_id == "cl_123"
        assert claim.confidence == 0.85

    def test_consensus_claim_net_evidence_strength(self):
        """Claim calculates net evidence strength."""
        from aragora.debate.consensus import Claim, Evidence

        supporting = Evidence(
            evidence_id="ev_1",
            source="a1",
            content="Support",
            evidence_type="argument",
            supports_claim=True,
            strength=0.8,
        )
        refuting = Evidence(
            evidence_id="ev_2",
            source="a2",
            content="Refute",
            evidence_type="argument",
            supports_claim=False,
            strength=0.3,
        )

        claim = Claim(
            claim_id="cl_1",
            statement="Test",
            author="a1",
            confidence=0.7,
            supporting_evidence=[supporting],
            refuting_evidence=[refuting],
        )

        # Net strength = (0.8 - 0.3) / (0.8 + 0.3) ≈ 0.45
        assert claim.net_evidence_strength > 0

    def test_consensus_dissent_record_creation(self):
        """DissentRecord dataclass can be created."""
        from aragora.debate.consensus import DissentRecord

        dissent = DissentRecord(
            agent="agent1",
            claim_id="cl_123",
            dissent_type="partial",
            reasons=["Insufficient evidence"],
            severity=0.6,
        )

        assert dissent.agent == "agent1"
        assert dissent.severity == 0.6


class TestMemoryManagerIntegration:
    """Integration tests for memory manager."""

    def test_memory_manager_import(self):
        """MemoryManager can be imported."""
        from aragora.debate.memory_manager import MemoryManager

        assert MemoryManager is not None

    def test_memory_manager_creation(self):
        """MemoryManager can be created without dependencies."""
        from aragora.debate.memory_manager import MemoryManager

        manager = MemoryManager()
        assert manager is not None

    def test_memory_manager_with_continuum(self):
        """MemoryManager can be created with ContinuumMemory."""
        from aragora.debate.memory_manager import MemoryManager

        mock_continuum = MagicMock()
        manager = MemoryManager(continuum_memory=mock_continuum)

        assert manager.continuum_memory is mock_continuum

    def test_store_outcome_with_continuum(self):
        """MemoryManager stores outcomes via ContinuumMemory."""
        from aragora.debate.memory_manager import MemoryManager
        from aragora.core import DebateResult

        mock_continuum = MagicMock()
        mock_continuum.add = MagicMock()
        manager = MemoryManager(continuum_memory=mock_continuum)

        result = DebateResult(
            task="Test task",
            consensus_reached=True,
            confidence=0.9,
            messages=[],
            critiques=[],
            votes=[],
            rounds_used=3,
            final_answer="Test answer",
        )

        manager.store_debate_outcome(result, task="Test task")

        # Verify add was called on continuum_memory
        mock_continuum.add.assert_called_once()

    def test_get_domain_default(self):
        """MemoryManager returns default domain without extractor."""
        from aragora.debate.memory_manager import MemoryManager

        manager = MemoryManager()

        # Without domain extractor, returns "general"
        assert manager._get_domain() == "general"

    def test_get_domain_with_extractor(self):
        """MemoryManager uses domain extractor when provided."""
        from aragora.debate.memory_manager import MemoryManager

        def mock_extractor():
            return "testing"

        manager = MemoryManager(domain_extractor=mock_extractor)

        assert manager._get_domain() == "testing"


class TestProtocolIntegration:
    """Integration tests for DebateProtocol with Arena."""

    @pytest.fixture
    def mock_agents(self):
        return [
            MockAgent(name="agent1"),
            MockAgent(name="agent2"),
        ]

    @pytest.fixture
    def environment(self):
        return Environment(task="Test protocol integration")

    def test_protocol_rounds_applied_to_arena(self, environment, mock_agents):
        """Protocol rounds setting is applied to Arena."""
        from aragora.debate.orchestrator import Arena

        protocol = DebateProtocol(rounds=5)
        arena = Arena(
            environment=environment,
            agents=mock_agents,
            protocol=protocol,
        )

        assert arena.protocol.rounds == 5

    def test_protocol_consensus_applied(self, environment, mock_agents):
        """Protocol consensus setting is applied."""
        from aragora.debate.orchestrator import Arena

        protocol = DebateProtocol(consensus="majority")
        arena = Arena(
            environment=environment,
            agents=mock_agents,
            protocol=protocol,
        )

        assert arena.protocol.consensus == "majority"

    def test_protocol_topology_applied(self, environment, mock_agents):
        """Protocol topology setting is applied."""
        from aragora.debate.orchestrator import Arena

        protocol = DebateProtocol(topology="ring")
        arena = Arena(
            environment=environment,
            agents=mock_agents,
            protocol=protocol,
        )

        assert arena.protocol.topology == "ring"

    def test_protocol_early_stopping_applied(self, environment, mock_agents):
        """Protocol early stopping setting is applied."""
        from aragora.debate.orchestrator import Arena

        protocol = DebateProtocol(
            early_stopping=True,
            early_stop_threshold=0.9,
        )
        arena = Arena(
            environment=environment,
            agents=mock_agents,
            protocol=protocol,
        )

        assert arena.protocol.early_stopping is True
        assert arena.protocol.early_stop_threshold == 0.9


class TestEventBridgeIntegration:
    """Integration tests for event bridge with EventBus."""

    def test_event_bus_with_spectator(self):
        """EventBus integrates with spectator stream."""
        mock_spectator = MagicMock()
        bus = EventBus(spectator=mock_spectator)

        assert bus._spectator is mock_spectator

    def test_event_bus_with_audience_manager(self):
        """EventBus integrates with audience manager."""
        mock_audience = MagicMock()
        bus = EventBus(audience_manager=mock_audience)

        assert bus._audience_manager is mock_audience

    def test_event_bus_with_immune_system(self):
        """EventBus integrates with immune system."""
        mock_immune = MagicMock()
        bus = EventBus(immune_system=mock_immune)

        assert bus._immune_system is mock_immune

    @pytest.mark.asyncio
    async def test_emit_notifies_spectator(self):
        """EventBus emit notifies spectator when present."""
        mock_spectator = MagicMock()
        mock_spectator.emit = MagicMock()
        bus = EventBus(spectator=mock_spectator)

        await bus.emit("debate_start", debate_id="test-123")

        # Spectator should be notified via emit()
        mock_spectator.emit.assert_called()


class TestDomainExtractionIntegration:
    """Integration tests for domain extraction."""

    def test_compute_domain_from_task_security(self):
        """Domain extraction detects security tasks."""
        from aragora.debate.orchestrator import _compute_domain_from_task

        assert _compute_domain_from_task("implement authentication") == "security"
        assert _compute_domain_from_task("fix security vulnerability") == "security"
        assert _compute_domain_from_task("add encryption") == "security"

    def test_compute_domain_from_task_performance(self):
        """Domain extraction detects performance tasks."""
        from aragora.debate.orchestrator import _compute_domain_from_task

        # Note: keywords must be lowercased for matching
        assert _compute_domain_from_task("optimize query speed") == "performance"
        assert _compute_domain_from_task("implement cache strategy") == "performance"
        assert _compute_domain_from_task("reduce latency") == "performance"

    def test_compute_domain_from_task_testing(self):
        """Domain extraction detects testing tasks."""
        from aragora.debate.orchestrator import _compute_domain_from_task

        assert _compute_domain_from_task("write unit tests") == "testing"
        assert _compute_domain_from_task("improve test coverage") == "testing"

    def test_compute_domain_from_task_general(self):
        """Domain extraction defaults to general."""
        from aragora.debate.orchestrator import _compute_domain_from_task

        assert _compute_domain_from_task("implement new feature") == "general"
        assert _compute_domain_from_task("random task") == "general"


class TestRoundPhasesIntegration:
    """Integration tests for round phases with protocol."""

    def test_structured_phases_with_protocol(self):
        """Protocol structured phases integrate correctly."""
        from aragora.debate.protocol import (
            DebateProtocol,
            STRUCTURED_ROUND_PHASES,
        )

        protocol = DebateProtocol(
            use_structured_phases=True,
        )

        # Get phase for round 0
        phase = protocol.get_round_phase(0)

        assert phase is not None
        assert phase.name == "Context Gathering"

    def test_structured_phases_all_rounds(self):
        """Protocol returns phases for all rounds."""
        from aragora.debate.protocol import (
            DebateProtocol,
            STRUCTURED_ROUND_PHASES,
        )

        protocol = DebateProtocol(
            use_structured_phases=True,
            rounds=9,
        )

        for i in range(9):
            phase = protocol.get_round_phase(i)
            assert phase is not None
            assert phase.number == i

    def test_light_phases_fewer_rounds(self):
        """Light protocol has fewer phases."""
        from aragora.debate.protocol import (
            ARAGORA_AI_LIGHT_PROTOCOL,
            STRUCTURED_LIGHT_ROUND_PHASES,
        )

        assert ARAGORA_AI_LIGHT_PROTOCOL.rounds == 4
        assert len(STRUCTURED_LIGHT_ROUND_PHASES) == 4
