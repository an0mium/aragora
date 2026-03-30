"""E2E test for live explainability: EventBus → factor snapshot → receipt metadata.

Verifies the full chain when ``enable_live_explainability=True``:
1. ``setup_debate_infrastructure`` creates a ``LiveExplainabilityStream`` and
   subscribes it to the ``EventBus``.
2. Debate events (proposals, critiques, refinements, votes, consensus) flow
   through the ``EventBus`` and accumulate in the stream.
3. ``handle_debate_completion`` snapshots the stream and attaches factor data
   to ``DebateResult.metadata["live_explainability"]``.
4. The receipt metadata contains factors, narrative, counts, and leading
   position — everything downstream consumers need for audit-ready receipts.

Uses mock agents only — no real API calls.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.core import DebateResult, Environment, TaskComplexity
from aragora.debate.context import DebateContext
from aragora.debate.event_bus import EventBus
from aragora.debate.orchestrator_runner import (
    _DebateExecutionState,
    handle_debate_completion,
    setup_debate_infrastructure,
)
from aragora.explainability.live_stream import LiveExplainabilityStream


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockArena:
    """Minimal Arena stand-in with the attributes the runner inspects."""

    def __init__(self) -> None:
        self.env = MagicMock(spec=Environment)
        self.env.task = "Should we adopt microservices?"
        self.env.context = {}

        # Agents
        agents = []
        for name in ("agent_a", "agent_b", "agent_c"):
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

        # Methods the runner calls
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
        self.disable_post_debate_pipeline = True
        self.enable_auto_execution = False
        self.post_debate_config = None
        self.compliance_monitor = None


def _make_execution_state() -> _DebateExecutionState:
    ctx = MagicMock(spec=DebateContext)
    ctx.env = MagicMock()
    ctx.env.task = "Should we adopt microservices?"
    ctx.result = DebateResult(
        task="Should we adopt microservices?",
        consensus_reached=True,
        confidence=0.82,
        messages=[],
        critiques=[],
        votes=[],
        rounds_used=3,
        final_answer="Adopt microservices with bounded contexts",
    )
    ctx.domain = "general"
    ctx.post_debate_workflow_triggered = False
    return _DebateExecutionState(
        debate_id="e2e-live-explain",
        correlation_id="corr-e2e",
        domain="general",
        task_complexity=TaskComplexity.MODERATE,
        ctx=ctx,
        debate_status="completed",
        debate_start_time=time.perf_counter() - 10.0,
    )


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------


class TestLiveExplainabilityE2E:
    """Full end-to-end: setup → EventBus events → completion → receipt metadata."""

    @pytest.mark.asyncio
    async def test_eventbus_to_receipt_metadata(self):
        """Events pumped through EventBus appear as factors in receipt metadata."""
        arena = _MockArena()
        state = _make_execution_state()

        # ── Step 1: setup creates stream + subscribes to EventBus ──
        await setup_debate_infrastructure(arena, state)

        assert arena.live_explainability_stream is not None
        assert isinstance(arena.live_explainability_stream, LiveExplainabilityStream)
        stream = arena.live_explainability_stream
        bus = arena.event_bus

        # ── Step 2: simulate a multi-round debate via EventBus ──
        # Round 1 — proposals
        bus.emit_sync(
            "agent_message",
            debate_id="e2e-live-explain",
            agent="agent_a",
            content="Microservices improve scalability",
            role="proposer",
            round_num=1,
        )
        bus.emit_sync(
            "agent_message",
            debate_id="e2e-live-explain",
            agent="agent_b",
            content="Monolith is simpler to operate",
            role="proposer",
            round_num=1,
        )

        # Round 1 — critiques
        bus.emit_sync(
            "agent_message",
            debate_id="e2e-live-explain",
            agent="agent_c",
            content="Microservices add network complexity",
            role="critic",
            round_num=1,
        )
        bus.emit_sync(
            "agent_message",
            debate_id="e2e-live-explain",
            agent="agent_a",
            content="Monolith doesn't scale past a certain size",
            role="critic",
            round_num=1,
        )

        # Round 2 — refinement
        bus.emit_sync(
            "agent_message",
            debate_id="e2e-live-explain",
            agent="agent_b",
            content="Adopt microservices with bounded contexts",
            role="reviser",
            round_num=2,
        )

        # Round 2 — votes
        bus.emit_sync(
            "vote",
            debate_id="e2e-live-explain",
            agent="agent_a",
            choice="microservices_bounded",
            confidence=0.9,
            round_num=2,
        )
        bus.emit_sync(
            "vote",
            debate_id="e2e-live-explain",
            agent="agent_b",
            choice="microservices_bounded",
            confidence=0.85,
            round_num=2,
        )
        bus.emit_sync(
            "vote",
            debate_id="e2e-live-explain",
            agent="agent_c",
            choice="microservices_bounded",
            confidence=0.75,
            round_num=2,
        )

        # Consensus
        bus.emit_sync(
            "consensus",
            debate_id="e2e-live-explain",
            confidence=0.85,
            position="microservices_bounded",
        )

        # Verify stream accumulated the events
        assert len(stream._evidence) == 5  # 2 proposals + 2 critiques + 1 refinement
        assert len(stream._votes) == 3

        # ── Step 3: handle_debate_completion attaches snapshot ──
        await handle_debate_completion(arena, state)

        # ── Step 4: verify receipt metadata ──
        result = state.ctx.result
        assert "live_explainability" in result.metadata, (
            "live_explainability key must be in receipt metadata"
        )

        meta = result.metadata["live_explainability"]

        # Structural checks
        assert isinstance(meta["factors"], list)
        assert len(meta["factors"]) > 0, "At least one factor must be present"
        assert isinstance(meta["narrative"], str)
        assert len(meta["narrative"]) > 0, "Narrative must be non-empty"

        # Count checks
        assert meta["evidence_count"] == 5
        assert meta["vote_count"] == 3
        assert isinstance(meta["belief_shifts"], int)

        # Position checks
        assert meta["leading_position"] is not None
        assert meta["position_confidence"] > 0

        # Agreement checks — all three agents voted the same choice
        assert meta["agent_agreement"] == 1.0

        # Factor content checks
        factor_names = {f["name"] for f in meta["factors"]}
        assert "agent_agreement" in factor_names, (
            "agent_agreement factor expected when votes are present"
        )
        assert "evidence_quality" in factor_names, (
            "evidence_quality factor expected when evidence is present"
        )

        # Each factor must have the required keys
        for factor in meta["factors"]:
            assert "name" in factor
            assert "contribution" in factor
            assert "explanation" in factor
            assert "trend" in factor
            assert isinstance(factor["contribution"], (int, float))

    @pytest.mark.asyncio
    async def test_no_metadata_when_disabled(self):
        """When enable_live_explainability=False, no receipt metadata is added."""
        arena = _MockArena()
        arena.enable_live_explainability = False
        state = _make_execution_state()

        await setup_debate_infrastructure(arena, state)
        assert arena.live_explainability_stream is None

        await handle_debate_completion(arena, state)
        assert "live_explainability" not in state.ctx.result.metadata

    @pytest.mark.asyncio
    async def test_factors_present_with_votes_only(self):
        """Even with just votes (no critiques/refinements), factors appear."""
        arena = _MockArena()
        state = _make_execution_state()

        await setup_debate_infrastructure(arena, state)
        bus = arena.event_bus

        # Single proposal + single vote
        bus.emit_sync(
            "agent_message",
            debate_id="e2e",
            agent="agent_a",
            content="Simple proposal",
            role="proposer",
            round_num=1,
        )
        bus.emit_sync(
            "vote",
            debate_id="e2e",
            agent="agent_a",
            choice="yes",
            confidence=0.8,
            round_num=1,
        )

        await handle_debate_completion(arena, state)

        meta = state.ctx.result.metadata["live_explainability"]
        assert meta["evidence_count"] == 1
        assert meta["vote_count"] == 1
        assert len(meta["factors"]) >= 1

    @pytest.mark.asyncio
    async def test_vote_flip_tracked_as_belief_shift(self):
        """A vote flip produces a belief shift counted in receipt metadata."""
        arena = _MockArena()
        state = _make_execution_state()

        await setup_debate_infrastructure(arena, state)
        bus = arena.event_bus

        bus.emit_sync(
            "agent_message",
            debate_id="e2e",
            agent="agent_a",
            content="Position X",
            role="proposer",
            round_num=1,
        )

        # First vote
        bus.emit_sync(
            "vote",
            debate_id="e2e",
            agent="agent_a",
            choice="X",
            confidence=0.6,
            round_num=1,
        )
        # Flip vote
        bus.emit_sync(
            "vote",
            debate_id="e2e",
            agent="agent_a",
            choice="Y",
            confidence=0.8,
            round_num=2,
        )

        await handle_debate_completion(arena, state)

        meta = state.ctx.result.metadata["live_explainability"]
        assert meta["belief_shifts"] >= 1, "Vote flip should register as a belief shift"

    @pytest.mark.asyncio
    async def test_snapshot_confidence_weighted_consensus_factor(self):
        """Confidence-weighted consensus factor appears when votes are present."""
        arena = _MockArena()
        state = _make_execution_state()

        await setup_debate_infrastructure(arena, state)
        bus = arena.event_bus

        # Votes with different confidence levels
        for agent, conf in [("agent_a", 0.95), ("agent_b", 0.6), ("agent_c", 0.5)]:
            bus.emit_sync(
                "vote",
                debate_id="e2e",
                agent=agent,
                choice="consensus_choice",
                confidence=conf,
                round_num=1,
            )

        await handle_debate_completion(arena, state)

        meta = state.ctx.result.metadata["live_explainability"]
        factor_names = {f["name"] for f in meta["factors"]}
        assert "confidence_weighted_consensus" in factor_names
