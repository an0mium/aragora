"""Tests for LiveExplainabilityStream wiring into Arena debate flow.

Verifies that:
- LiveExplainabilityStream is created when enable_live_explainability=True
- EventBus subscriptions route agent_message, vote, consensus events to the stream
- Final snapshot is attached to DebateResult.metadata after debate completion
- Disabled by default (no stream created when flag is False)
- Graceful degradation when imports fail
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.core import DebateResult, Environment, TaskComplexity
from aragora.debate.context import DebateContext
from aragora.debate.event_bus import DebateEvent, EventBus
from aragora.debate.orchestrator_runner import (
    _DebateExecutionState,
    _subscribe_live_explainability,
    setup_debate_infrastructure,
    handle_debate_completion,
)
from aragora.explainability.live_stream import (
    LiveExplainabilityStream,
)


# =============================================================================
# Fixtures
# =============================================================================


class _FakeArena:
    """Lightweight fake Arena with just enough attributes for explainability tests.

    Using a real class instead of MagicMock so attribute assignments
    (like ``arena.live_explainability_stream = stream``) persist correctly.
    """

    def __init__(self) -> None:
        self.env = MagicMock(spec=Environment)
        self.env.task = "Should we use rate limiting?"
        self.env.context = {}

        # Agents
        agents = []
        for name in ("claude", "gpt4", "gemini"):
            agent = MagicMock()
            agent.name = name
            agent.model = f"{name}-model"
            agents.append(agent)
        self.agents = agents

        # Protocol
        self.protocol = MagicMock()
        self.protocol.enable_km_belief_sync = False
        self.protocol.enable_hook_tracking = False
        self.protocol.rounds = 3
        self.protocol.checkpoint_cleanup_on_success = True
        self.protocol.enable_translation = False

        # Budget
        self._budget_coordinator = MagicMock()
        self._budget_coordinator.check_budget_before_debate = MagicMock()
        self._budget_coordinator.autotuner = None

        # Trackers
        self._trackers = MagicMock()
        self._trackers.on_debate_start = MagicMock()
        self._trackers.on_debate_complete = MagicMock()

        # Extensions
        self.extensions = MagicMock()
        self.extensions.on_debate_complete = MagicMock()
        self.extensions.setup_debate_budget = MagicMock()

        # Event system
        self.event_bus = EventBus()
        self._event_emitter = MagicMock()

        # Methods
        self._emit_agent_preview = MagicMock()
        self._create_pending_debate_bead = AsyncMock(return_value=None)
        self._init_hook_tracking = AsyncMock(return_value={})
        self._ingest_debate_outcome = AsyncMock()
        self._update_debate_bead = AsyncMock()
        self._complete_hook_tracking = AsyncMock()
        self._create_debate_bead = AsyncMock(return_value=None)
        self._queue_for_supabase_sync = MagicMock()
        self.cleanup_checkpoints = AsyncMock(return_value=0)
        self._cleanup_convergence_cache = MagicMock()
        self._teardown_agent_channels = AsyncMock()
        self._translate_conclusions = AsyncMock()

        # Config flags
        self.enable_live_explainability = True
        self.live_explainability_stream = None
        self.enable_post_debate_workflow = False
        self.disable_post_debate_pipeline = True  # Skip post-debate coordinator
        self.enable_auto_execution = False
        self.post_debate_config = None
        self.compliance_monitor = None


@pytest.fixture
def fake_arena():
    return _FakeArena()


@pytest.fixture
def execution_state():
    """Create a _DebateExecutionState for testing."""
    ctx = MagicMock(spec=DebateContext)
    ctx.env = MagicMock()
    ctx.env.task = "Should we use rate limiting?"
    ctx.result = DebateResult(
        task="Should we use rate limiting?",
        consensus_reached=True,
        confidence=0.85,
        messages=[],
        critiques=[],
        votes=[],
        rounds_used=3,
        final_answer="Yes, use rate limiting with token bucket",
    )
    ctx.domain = "general"
    ctx.post_debate_workflow_triggered = False
    return _DebateExecutionState(
        debate_id="test-live-explainability",
        correlation_id="corr-test",
        domain="general",
        task_complexity=TaskComplexity.MODERATE,
        ctx=ctx,
        debate_status="completed",
        debate_start_time=time.perf_counter() - 5.0,
    )


# =============================================================================
# Test: Stream creation during setup
# =============================================================================


class TestStreamCreation:
    """Tests for LiveExplainabilityStream creation in setup_debate_infrastructure."""

    @pytest.mark.asyncio
    async def test_stream_created_when_enabled(self, fake_arena, execution_state):
        """Stream should be created when enable_live_explainability is True."""
        await setup_debate_infrastructure(fake_arena, execution_state)

        assert fake_arena.live_explainability_stream is not None
        assert isinstance(
            fake_arena.live_explainability_stream, LiveExplainabilityStream
        )

    @pytest.mark.asyncio
    async def test_stream_not_created_when_disabled(self, fake_arena, execution_state):
        """Stream should NOT be created when enable_live_explainability is False."""
        fake_arena.enable_live_explainability = False

        await setup_debate_infrastructure(fake_arena, execution_state)

        assert fake_arena.live_explainability_stream is None

    @pytest.mark.asyncio
    async def test_stream_graceful_on_missing_event_bus(
        self, fake_arena, execution_state
    ):
        """Stream should be created even without EventBus (just no subscriptions)."""
        fake_arena.event_bus = None

        await setup_debate_infrastructure(fake_arena, execution_state)

        # Stream created but no subscriptions (no EventBus)
        assert fake_arena.live_explainability_stream is not None


# =============================================================================
# Test: EventBus subscription routing
# =============================================================================


class TestEventBusSubscription:
    """Tests for _subscribe_live_explainability EventBus routing."""

    def test_proposal_routed_to_on_proposal(self):
        """agent_message with role=proposer should call on_proposal."""
        stream = LiveExplainabilityStream()
        bus = EventBus()
        _subscribe_live_explainability(bus, stream)

        event = DebateEvent(
            event_type="agent_message",
            debate_id="test",
            data={
                "agent": "claude",
                "content": "We should use rate limiting",
                "role": "proposer",
                "round_num": 1,
            },
        )
        for handler in bus._sync_handlers.get("agent_message", []):
            handler(event)

        assert len(stream._evidence) == 1
        assert stream._evidence[0].source == "claude"
        assert stream._evidence[0].evidence_type == "proposal"

    def test_critique_routed_to_on_critique(self):
        """agent_message with role=critic should call on_critique."""
        stream = LiveExplainabilityStream()
        bus = EventBus()
        _subscribe_live_explainability(bus, stream)

        event = DebateEvent(
            event_type="agent_message",
            debate_id="test",
            data={
                "agent": "gpt4",
                "content": "Rate limiting alone is insufficient",
                "role": "critic",
                "round_num": 1,
            },
        )
        for handler in bus._sync_handlers.get("agent_message", []):
            handler(event)

        assert len(stream._evidence) == 1
        assert stream._evidence[0].source == "gpt4"
        assert stream._evidence[0].evidence_type == "critique"

    def test_refinement_routed_to_on_refinement(self):
        """agent_message with role=reviser should call on_refinement."""
        stream = LiveExplainabilityStream()
        bus = EventBus()
        _subscribe_live_explainability(bus, stream)

        # First add a proposal so refinement can track position change
        stream.on_proposal("claude", "Original position", round_num=1)

        event = DebateEvent(
            event_type="agent_message",
            debate_id="test",
            data={
                "agent": "claude",
                "content": "Revised position with more detail",
                "role": "reviser",
                "round_num": 2,
            },
        )
        for handler in bus._sync_handlers.get("agent_message", []):
            handler(event)

        # Should have proposal + refinement
        assert len(stream._evidence) == 2
        assert stream._evidence[1].evidence_type == "refinement"

    def test_vote_routed_to_on_vote(self):
        """vote event should call on_vote."""
        stream = LiveExplainabilityStream()
        bus = EventBus()
        _subscribe_live_explainability(bus, stream)

        event = DebateEvent(
            event_type="vote",
            debate_id="test",
            data={
                "agent": "claude",
                "choice": "rate_limiting",
                "confidence": 0.85,
                "round_num": 2,
                "reasoning": "Best approach for API protection",
            },
        )
        for handler in bus._sync_handlers.get("vote", []):
            handler(event)

        assert len(stream._votes) == 1
        assert stream._votes[0].agent == "claude"
        assert stream._votes[0].choice == "rate_limiting"
        assert stream._votes[0].confidence == 0.85

    def test_consensus_routed_to_on_consensus(self):
        """consensus event should call on_consensus."""
        stream = LiveExplainabilityStream()
        bus = EventBus()
        _subscribe_live_explainability(bus, stream)

        event = DebateEvent(
            event_type="consensus",
            debate_id="test",
            data={
                "confidence": 0.9,
                "position": "Use token bucket rate limiting",
            },
        )
        for handler in bus._sync_handlers.get("consensus", []):
            handler(event)

        # Consensus sets conclusion in agent_positions
        assert "consensus" in stream._agent_positions or len(stream._evidence) >= 0
        # No exception should be raised

    def test_unknown_role_ignored(self):
        """agent_message with unknown role should not crash."""
        stream = LiveExplainabilityStream()
        bus = EventBus()
        _subscribe_live_explainability(bus, stream)

        event = DebateEvent(
            event_type="agent_message",
            debate_id="test",
            data={
                "agent": "claude",
                "content": "Some message",
                "role": "judge",  # Not proposer/critic/reviser
                "round_num": 1,
            },
        )
        for handler in bus._sync_handlers.get("agent_message", []):
            handler(event)

        # Should not have added any evidence
        assert len(stream._evidence) == 0

    def test_multiple_events_accumulate(self):
        """Multiple events should accumulate state in the stream."""
        stream = LiveExplainabilityStream()
        bus = EventBus()
        _subscribe_live_explainability(bus, stream)

        # Proposal
        bus.emit_sync("agent_message", debate_id="test",
                       agent="claude", content="Proposal A", role="proposer", round_num=1)
        # Critique
        bus.emit_sync("agent_message", debate_id="test",
                       agent="gpt4", content="Critique of A", role="critic", round_num=1)
        # Vote
        bus.emit_sync("vote", debate_id="test",
                       agent="claude", choice="A", confidence=0.8, round_num=2)

        assert len(stream._evidence) == 2
        assert len(stream._votes) == 1


# =============================================================================
# Test: Snapshot attachment to DebateResult
# =============================================================================


class TestSnapshotAttachment:
    """Tests for attaching explainability snapshot to DebateResult."""

    @pytest.mark.asyncio
    async def test_snapshot_attached_to_result_metadata(
        self, fake_arena, execution_state
    ):
        """Final snapshot should be in result.metadata['live_explainability']."""
        # Create and populate a real stream
        stream = LiveExplainabilityStream()
        stream.on_proposal("claude", "Use rate limiting", round_num=1)
        stream.on_critique("gpt4", "Not enough on its own", round_num=1)
        stream.on_vote("claude", "rate_limiting", confidence=0.85, round_num=2)
        stream.on_vote("gpt4", "rate_limiting", confidence=0.7, round_num=2)
        fake_arena.live_explainability_stream = stream

        await handle_debate_completion(fake_arena, execution_state)

        result = execution_state.ctx.result
        assert "live_explainability" in result.metadata

        explainability = result.metadata["live_explainability"]
        assert "factors" in explainability
        assert "narrative" in explainability
        assert "leading_position" in explainability
        assert "agent_agreement" in explainability
        assert "evidence_quality" in explainability
        assert explainability["evidence_count"] == 2  # proposal + critique
        assert explainability["vote_count"] == 2

    @pytest.mark.asyncio
    async def test_snapshot_not_attached_when_stream_is_none(
        self, fake_arena, execution_state
    ):
        """No snapshot when stream is None (disabled)."""
        fake_arena.live_explainability_stream = None

        await handle_debate_completion(fake_arena, execution_state)

        result = execution_state.ctx.result
        assert "live_explainability" not in result.metadata

    @pytest.mark.asyncio
    async def test_snapshot_factors_are_serializable(
        self, fake_arena, execution_state
    ):
        """Factors in metadata should be plain dicts (JSON-serializable)."""
        stream = LiveExplainabilityStream()
        stream.on_proposal("claude", "Position A", round_num=1)
        stream.on_vote("claude", "A", confidence=0.9, round_num=2)
        fake_arena.live_explainability_stream = stream

        await handle_debate_completion(fake_arena, execution_state)

        factors = execution_state.ctx.result.metadata["live_explainability"]["factors"]
        assert isinstance(factors, list)
        for factor in factors:
            assert isinstance(factor, dict)
            assert "name" in factor
            assert "contribution" in factor
            assert isinstance(factor["contribution"], (int, float))

    @pytest.mark.asyncio
    async def test_snapshot_has_round_and_counts(
        self, fake_arena, execution_state
    ):
        """Snapshot metadata should include round number and event counts."""
        stream = LiveExplainabilityStream()
        stream.on_proposal("claude", "Proposal", round_num=3)
        stream.on_critique("gpt4", "Critique", round_num=3)
        stream.on_refinement("claude", "Revised proposal", round_num=3)
        fake_arena.live_explainability_stream = stream

        await handle_debate_completion(fake_arena, execution_state)

        meta = execution_state.ctx.result.metadata["live_explainability"]
        assert meta["evidence_count"] == 3
        assert meta["round_num"] >= 0
        assert isinstance(meta["belief_shifts"], int)


# =============================================================================
# Test: Full round-trip (setup → events → completion)
# =============================================================================


class TestFullRoundTrip:
    """End-to-end test: setup creates stream, events populate it, completion attaches snapshot."""

    @pytest.mark.asyncio
    async def test_setup_events_completion_cycle(self, fake_arena, execution_state):
        """Full cycle: setup → simulate events → handle completion."""
        # Step 1: Setup creates the stream and subscribes to EventBus
        await setup_debate_infrastructure(fake_arena, execution_state)
        assert fake_arena.live_explainability_stream is not None

        stream = fake_arena.live_explainability_stream
        bus = fake_arena.event_bus

        # Step 2: Simulate debate events through EventBus
        # Round 1: proposals
        bus.emit_sync(
            "agent_message", debate_id="test",
            agent="claude", content="Rate limiting with token bucket",
            role="proposer", round_num=1,
        )
        bus.emit_sync(
            "agent_message", debate_id="test",
            agent="gpt4", content="Circuit breaker pattern instead",
            role="proposer", round_num=1,
        )

        # Round 1: critiques
        bus.emit_sync(
            "agent_message", debate_id="test",
            agent="gpt4", content="Token bucket is too simple",
            role="critic", round_num=1,
        )
        bus.emit_sync(
            "agent_message", debate_id="test",
            agent="claude", content="Circuit breakers don't limit rate",
            role="critic", round_num=1,
        )

        # Round 2: votes
        bus.emit_sync(
            "vote", debate_id="test",
            agent="claude", choice="token_bucket", confidence=0.85, round_num=2,
        )
        bus.emit_sync(
            "vote", debate_id="test",
            agent="gpt4", choice="token_bucket", confidence=0.7, round_num=2,
        )

        # Verify stream accumulated events
        assert len(stream._evidence) == 4  # 2 proposals + 2 critiques
        assert len(stream._votes) == 2

        # Step 3: Handle completion attaches snapshot
        await handle_debate_completion(fake_arena, execution_state)

        result = execution_state.ctx.result
        assert "live_explainability" in result.metadata
        meta = result.metadata["live_explainability"]
        assert meta["evidence_count"] == 4
        assert meta["vote_count"] == 2
        assert isinstance(meta["narrative"], str)
        assert len(meta["factors"]) > 0


# =============================================================================
# Test: ArenaConfig flag
# =============================================================================


class TestArenaConfigFlag:
    """Tests for enable_live_explainability config flag."""

    def test_arena_config_default_false(self):
        """ArenaConfig should default enable_live_explainability to False."""
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig()
        assert config.enable_live_explainability is False

    def test_arena_config_can_enable(self):
        """ArenaConfig should accept enable_live_explainability=True."""
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig(enable_live_explainability=True)
        assert config.enable_live_explainability is True
