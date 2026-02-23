"""Tests for ActiveIntrospectionTracker wiring into Arena debate flow.

Verifies that:
- ActiveIntrospectionTracker is created when enable_introspection=True
- EventBus subscriptions route agent_message, round_start, round_end events to the tracker
- Introspection summaries are attached to DebateResult.metadata after debate completion
- Disabled when flag is False (no tracker created)
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
    _subscribe_active_introspection,
    setup_debate_infrastructure,
    handle_debate_completion,
)
from aragora.introspection.active import (
    ActiveIntrospectionTracker,
    RoundMetrics,
)


# =============================================================================
# Fixtures
# =============================================================================


class _FakeArena:
    """Lightweight fake Arena with just enough attributes for introspection tests.

    Using a real class instead of MagicMock so attribute assignments
    (like ``arena.active_introspection_tracker = tracker``) persist correctly.
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
        self.enable_introspection = True
        self.active_introspection_tracker = None
        self.enable_live_explainability = False
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
        debate_id="test-introspection",
        correlation_id="corr-test",
        domain="general",
        task_complexity=TaskComplexity.MODERATE,
        ctx=ctx,
        debate_status="completed",
        debate_start_time=time.perf_counter() - 5.0,
    )


# =============================================================================
# Test: Tracker creation during setup
# =============================================================================


class TestTrackerCreation:
    """Tests for ActiveIntrospectionTracker creation in setup_debate_infrastructure."""

    @pytest.mark.asyncio
    async def test_tracker_created_when_enabled(self, fake_arena, execution_state):
        """Tracker should be created when enable_introspection is True."""
        await setup_debate_infrastructure(fake_arena, execution_state)

        assert fake_arena.active_introspection_tracker is not None
        assert isinstance(fake_arena.active_introspection_tracker, ActiveIntrospectionTracker)

    @pytest.mark.asyncio
    async def test_tracker_not_created_when_disabled(self, fake_arena, execution_state):
        """Tracker should NOT be created when enable_introspection is False."""
        fake_arena.enable_introspection = False

        await setup_debate_infrastructure(fake_arena, execution_state)

        assert fake_arena.active_introspection_tracker is None

    @pytest.mark.asyncio
    async def test_tracker_graceful_on_missing_event_bus(self, fake_arena, execution_state):
        """Tracker should be created even without EventBus (just no subscriptions)."""
        fake_arena.event_bus = None

        await setup_debate_infrastructure(fake_arena, execution_state)

        # Tracker created but no subscriptions (no EventBus)
        assert fake_arena.active_introspection_tracker is not None


# =============================================================================
# Test: EventBus subscription routing
# =============================================================================


class TestEventBusSubscription:
    """Tests for _subscribe_active_introspection EventBus routing."""

    def test_proposal_tracked(self):
        """agent_message with role=proposer should update tracker with proposals_made."""
        tracker = ActiveIntrospectionTracker()
        bus = EventBus()
        _subscribe_active_introspection(bus, tracker)

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

        summary = tracker.get_summary("claude")
        assert summary is not None
        assert summary.total_proposals == 1

    def test_critique_tracked(self):
        """agent_message with role=critic should update tracker with critiques_given."""
        tracker = ActiveIntrospectionTracker()
        bus = EventBus()
        _subscribe_active_introspection(bus, tracker)

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

        summary = tracker.get_summary("gpt4")
        assert summary is not None
        assert summary.total_critiques == 1

    def test_unknown_role_ignored(self):
        """agent_message with unknown role should not crash or add metrics."""
        tracker = ActiveIntrospectionTracker()
        bus = EventBus()
        _subscribe_active_introspection(bus, tracker)

        event = DebateEvent(
            event_type="agent_message",
            debate_id="test",
            data={
                "agent": "claude",
                "content": "Some message",
                "role": "judge",  # Not proposer/critic
                "round_num": 1,
            },
        )
        for handler in bus._sync_handlers.get("agent_message", []):
            handler(event)

        # No metrics should be recorded for unknown roles
        summary = tracker.get_summary("claude")
        assert summary is None

    def test_round_end_updates_rounds_completed(self):
        """round_end event should ensure all tracked agents have round recorded."""
        tracker = ActiveIntrospectionTracker()
        bus = EventBus()
        _subscribe_active_introspection(bus, tracker)

        # First, add an agent via a proposal
        bus.emit_sync(
            "agent_message",
            debate_id="test",
            agent="claude",
            content="Proposal",
            role="proposer",
            round_num=1,
        )

        # Now emit round_end for round 2
        event = DebateEvent(
            event_type="round_end",
            debate_id="test",
            data={"round_num": 2},
        )
        for handler in bus._sync_handlers.get("round_end", []):
            handler(event)

        summary = tracker.get_summary("claude")
        assert summary is not None
        assert summary.rounds_completed == 2

    def test_round_start_does_not_crash(self):
        """round_start event should be handled without errors."""
        tracker = ActiveIntrospectionTracker()
        bus = EventBus()
        _subscribe_active_introspection(bus, tracker)

        event = DebateEvent(
            event_type="round_start",
            debate_id="test",
            data={"round_num": 1},
        )
        # Should not raise
        for handler in bus._sync_handlers.get("round_start", []):
            handler(event)

    def test_multiple_events_accumulate(self):
        """Multiple events should accumulate metrics in the tracker."""
        tracker = ActiveIntrospectionTracker()
        bus = EventBus()
        _subscribe_active_introspection(bus, tracker)

        # Proposal from claude
        bus.emit_sync(
            "agent_message",
            debate_id="test",
            agent="claude",
            content="Proposal A",
            role="proposer",
            round_num=1,
        )
        # Critique from gpt4
        bus.emit_sync(
            "agent_message",
            debate_id="test",
            agent="gpt4",
            content="Critique of A",
            role="critic",
            round_num=1,
        )
        # Another proposal from claude
        bus.emit_sync(
            "agent_message",
            debate_id="test",
            agent="claude",
            content="Proposal B",
            role="proposer",
            round_num=2,
        )

        claude_summary = tracker.get_summary("claude")
        assert claude_summary is not None
        assert claude_summary.total_proposals == 2

        gpt4_summary = tracker.get_summary("gpt4")
        assert gpt4_summary is not None
        assert gpt4_summary.total_critiques == 1


# =============================================================================
# Test: Summary attachment to DebateResult
# =============================================================================


class TestSummaryAttachment:
    """Tests for attaching introspection summaries to DebateResult."""

    @pytest.mark.asyncio
    async def test_introspection_attached_to_result_metadata(self, fake_arena, execution_state):
        """Introspection summaries should be in result.metadata['introspection']."""
        # Create and populate a tracker
        tracker = ActiveIntrospectionTracker()
        tracker.update_round(
            "claude",
            1,
            RoundMetrics(
                round_number=1,
                proposals_made=2,
                proposals_accepted=1,
            ),
        )
        tracker.update_round(
            "gpt4",
            1,
            RoundMetrics(
                round_number=1,
                critiques_given=3,
                critiques_led_to_changes=2,
            ),
        )
        fake_arena.active_introspection_tracker = tracker

        await handle_debate_completion(fake_arena, execution_state)

        result = execution_state.ctx.result
        assert "introspection" in result.metadata

        introspection = result.metadata["introspection"]
        assert "claude" in introspection
        assert "gpt4" in introspection
        assert introspection["claude"]["total_proposals"] == 2
        assert introspection["claude"]["total_accepted"] == 1
        assert introspection["gpt4"]["total_critiques"] == 3
        assert introspection["gpt4"]["total_critiques_effective"] == 2

    @pytest.mark.asyncio
    async def test_introspection_not_attached_when_tracker_is_none(
        self, fake_arena, execution_state
    ):
        """No introspection when tracker is None (disabled)."""
        fake_arena.active_introspection_tracker = None

        await handle_debate_completion(fake_arena, execution_state)

        result = execution_state.ctx.result
        assert "introspection" not in result.metadata

    @pytest.mark.asyncio
    async def test_introspection_not_attached_when_no_data(self, fake_arena, execution_state):
        """No introspection metadata when tracker has no data."""
        tracker = ActiveIntrospectionTracker()
        # Empty tracker -- no agents tracked
        fake_arena.active_introspection_tracker = tracker

        await handle_debate_completion(fake_arena, execution_state)

        result = execution_state.ctx.result
        # Empty summaries dict is falsy, so no metadata attached
        assert "introspection" not in result.metadata

    @pytest.mark.asyncio
    async def test_introspection_summaries_are_serializable(self, fake_arena, execution_state):
        """Summaries in metadata should be plain dicts (JSON-serializable)."""
        tracker = ActiveIntrospectionTracker()
        tracker.update_round(
            "claude",
            1,
            RoundMetrics(
                round_number=1,
                proposals_made=1,
                proposals_accepted=1,
                argument_influence=0.8,
            ),
        )
        fake_arena.active_introspection_tracker = tracker

        await handle_debate_completion(fake_arena, execution_state)

        introspection = execution_state.ctx.result.metadata["introspection"]
        assert isinstance(introspection, dict)
        claude_data = introspection["claude"]
        assert isinstance(claude_data, dict)
        assert "agent_name" in claude_data
        assert "rounds_completed" in claude_data
        assert "total_proposals" in claude_data
        assert "proposal_acceptance_rate" in claude_data
        assert "average_influence" in claude_data
        assert isinstance(claude_data["proposal_acceptance_rate"], (int, float))

    @pytest.mark.asyncio
    async def test_introspection_has_round_history(self, fake_arena, execution_state):
        """Introspection metadata should include round history."""
        tracker = ActiveIntrospectionTracker()
        tracker.update_round(
            "claude",
            1,
            RoundMetrics(
                round_number=1,
                proposals_made=1,
            ),
        )
        tracker.update_round(
            "claude",
            2,
            RoundMetrics(
                round_number=2,
                proposals_made=1,
                proposals_accepted=1,
            ),
        )
        fake_arena.active_introspection_tracker = tracker

        await handle_debate_completion(fake_arena, execution_state)

        meta = execution_state.ctx.result.metadata["introspection"]
        claude_data = meta["claude"]
        assert len(claude_data["round_history"]) == 2
        assert claude_data["round_history"][0]["round_number"] == 1
        assert claude_data["round_history"][1]["round_number"] == 2


# =============================================================================
# Test: Full round-trip (setup -> events -> completion)
# =============================================================================


class TestFullRoundTrip:
    """End-to-end test: setup creates tracker, events populate it, completion attaches summary."""

    @pytest.mark.asyncio
    async def test_setup_events_completion_cycle(self, fake_arena, execution_state):
        """Full cycle: setup -> simulate events -> handle completion."""
        # Step 1: Setup creates the tracker and subscribes to EventBus
        await setup_debate_infrastructure(fake_arena, execution_state)
        assert fake_arena.active_introspection_tracker is not None

        tracker = fake_arena.active_introspection_tracker
        bus = fake_arena.event_bus

        # Step 2: Simulate debate events through EventBus
        # Round 1: proposals
        bus.emit_sync(
            "agent_message",
            debate_id="test",
            agent="claude",
            content="Rate limiting with token bucket",
            role="proposer",
            round_num=1,
        )
        bus.emit_sync(
            "agent_message",
            debate_id="test",
            agent="gpt4",
            content="Circuit breaker pattern instead",
            role="proposer",
            round_num=1,
        )

        # Round 1: critiques
        bus.emit_sync(
            "agent_message",
            debate_id="test",
            agent="gpt4",
            content="Token bucket is too simple",
            role="critic",
            round_num=1,
        )
        bus.emit_sync(
            "agent_message",
            debate_id="test",
            agent="claude",
            content="Circuit breakers don't limit rate",
            role="critic",
            round_num=1,
        )

        # Verify tracker accumulated events
        claude_summary = tracker.get_summary("claude")
        assert claude_summary is not None
        assert claude_summary.total_proposals == 1
        assert claude_summary.total_critiques == 1

        gpt4_summary = tracker.get_summary("gpt4")
        assert gpt4_summary is not None
        assert gpt4_summary.total_proposals == 1
        assert gpt4_summary.total_critiques == 1

        # Step 3: Handle completion attaches summary
        await handle_debate_completion(fake_arena, execution_state)

        result = execution_state.ctx.result
        assert "introspection" in result.metadata
        meta = result.metadata["introspection"]
        assert "claude" in meta
        assert "gpt4" in meta
        assert meta["claude"]["total_proposals"] == 1
        assert meta["gpt4"]["total_critiques"] == 1


# =============================================================================
# Test: ArenaConfig flag
# =============================================================================


class TestArenaConfigFlag:
    """Tests for enable_introspection config flag."""

    def test_arena_config_default_true(self):
        """ArenaConfig should default enable_introspection to True."""
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig()
        assert config.enable_introspection is True

    def test_arena_config_can_disable(self):
        """ArenaConfig should accept enable_introspection=False."""
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig(enable_introspection=False)
        assert config.enable_introspection is False
