"""End-to-end test for live explainability: EventBus -> factor snapshot -> receipt metadata.

Verifies the full chain:
1. EventBus emits agent_message (proposals, critiques, refinements), vote, and consensus events
2. LiveExplainabilityStream accumulates evidence, votes, and belief shifts
3. Factor snapshot is computed with evidence_quality, agent_agreement, etc.
4. Snapshot is attached to DebateResult.metadata["live_explainability"]
5. Factors appear in final receipt metadata with correct structure

Uses mock agents — no real API calls.
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
    _subscribe_live_explainability,
    setup_debate_infrastructure,
    handle_debate_completion,
)
from aragora.explainability.live_stream import (
    LiveExplainabilityStream,
    ExplanationSnapshot,
)


# =============================================================================
# Helpers
# =============================================================================


class _FakeArena:
    """Minimal Arena stub for E2E explainability tests."""

    def __init__(self) -> None:
        self.env = MagicMock(spec=Environment)
        self.env.task = "Should we adopt microservices?"
        self.env.context = {}

        agents = []
        for name in ("claude", "gpt4", "gemini"):
            agent = MagicMock()
            agent.name = name
            agent.model = f"{name}-model"
            agents.append(agent)
        self.agents = agents

        self.protocol = MagicMock()
        self.protocol.enable_km_belief_sync = False
        self.protocol.enable_hook_tracking = False
        self.protocol.rounds = 3
        self.protocol.checkpoint_cleanup_on_success = True
        self.protocol.enable_translation = False

        self._budget_coordinator = MagicMock()
        self._budget_coordinator.check_budget_before_debate = MagicMock()
        self._budget_coordinator.autotuner = None

        self._trackers = MagicMock()
        self._trackers.on_debate_start = MagicMock()
        self._trackers.on_debate_complete = MagicMock()

        self.extensions = MagicMock()
        self.extensions.on_debate_complete = MagicMock()
        self.extensions.setup_debate_budget = MagicMock()

        self.event_bus = EventBus()
        self._event_emitter = MagicMock()

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
        final_answer="Yes, adopt microservices with a strangler-fig migration",
    )
    ctx.domain = "general"
    ctx.post_debate_workflow_triggered = False
    return _DebateExecutionState(
        debate_id="e2e-live-explainability",
        correlation_id="corr-e2e",
        domain="general",
        task_complexity=TaskComplexity.MODERATE,
        ctx=ctx,
        debate_status="completed",
        debate_start_time=time.perf_counter() - 10.0,
    )


# =============================================================================
# E2E test: full chain from EventBus to receipt metadata
# =============================================================================


class TestLiveExplainabilityE2E:
    """End-to-end: setup -> EventBus events -> factors -> receipt metadata."""

    @pytest.mark.asyncio
    async def test_eventbus_to_receipt_metadata_full_chain(self):
        """The full chain: setup creates stream, EventBus events flow through,
        factors are computed, and the snapshot lands in result.metadata."""
        arena = _FakeArena()
        state = _make_execution_state()

        # Step 1: setup_debate_infrastructure creates the stream and subscribes
        await setup_debate_infrastructure(arena, state)
        assert arena.live_explainability_stream is not None
        assert isinstance(arena.live_explainability_stream, LiveExplainabilityStream)

        bus = arena.event_bus

        # Step 2: Simulate a multi-round debate via EventBus
        # Round 1 — proposals
        bus.emit_sync(
            "agent_message",
            debate_id="e2e-live-explainability",
            agent="claude",
            content="We should adopt microservices for independent scaling",
            role="proposer",
            round_num=1,
        )
        bus.emit_sync(
            "agent_message",
            debate_id="e2e-live-explainability",
            agent="gpt4",
            content="Monolith-first is safer for small teams",
            role="proposer",
            round_num=1,
        )
        bus.emit_sync(
            "agent_message",
            debate_id="e2e-live-explainability",
            agent="gemini",
            content="Modular monolith is a good middle ground",
            role="proposer",
            round_num=1,
        )

        # Round 1 — critiques
        bus.emit_sync(
            "agent_message",
            debate_id="e2e-live-explainability",
            agent="gpt4",
            content="Microservices add operational complexity",
            role="critic",
            round_num=1,
        )
        bus.emit_sync(
            "agent_message",
            debate_id="e2e-live-explainability",
            agent="claude",
            content="Monolith coupling slows feature delivery",
            role="critic",
            round_num=1,
        )

        # Round 2 — refinements
        bus.emit_sync(
            "agent_message",
            debate_id="e2e-live-explainability",
            agent="claude",
            content="Strangler-fig pattern for gradual migration",
            role="reviser",
            round_num=2,
        )

        # Round 2 — votes
        bus.emit_sync(
            "vote",
            debate_id="e2e-live-explainability",
            agent="claude",
            choice="microservices_strangler",
            confidence=0.88,
            round_num=2,
            reasoning="Gradual migration reduces risk",
        )
        bus.emit_sync(
            "vote",
            debate_id="e2e-live-explainability",
            agent="gpt4",
            choice="microservices_strangler",
            confidence=0.72,
            round_num=2,
            reasoning="Compromise approach is pragmatic",
        )
        bus.emit_sync(
            "vote",
            debate_id="e2e-live-explainability",
            agent="gemini",
            choice="modular_monolith",
            confidence=0.65,
            round_num=2,
            reasoning="Simpler to operate",
        )

        # Consensus
        bus.emit_sync(
            "consensus",
            debate_id="e2e-live-explainability",
            confidence=0.82,
            position="Adopt microservices with strangler-fig migration",
        )

        # Verify stream accumulated everything
        stream = arena.live_explainability_stream
        assert len(stream._evidence) == 6  # 3 proposals + 2 critiques + 1 refinement
        assert len(stream._votes) == 3

        # Step 3: handle_debate_completion attaches snapshot to result metadata
        await handle_debate_completion(arena, state)

        result = state.ctx.result
        assert "live_explainability" in result.metadata

        meta = result.metadata["live_explainability"]

        # Verify structural completeness
        assert "factors" in meta
        assert "narrative" in meta
        assert "leading_position" in meta
        assert "agent_agreement" in meta
        assert "evidence_quality" in meta
        assert "position_confidence" in meta
        assert "round_num" in meta
        assert "evidence_count" in meta
        assert "vote_count" in meta
        assert "belief_shifts" in meta

        # Verify counts match what we fed through EventBus
        assert meta["evidence_count"] == 6
        assert meta["vote_count"] == 3

        # Verify factors are present and well-formed
        factors = meta["factors"]
        assert isinstance(factors, list)
        assert len(factors) > 0

        factor_names = {f["name"] for f in factors}
        # With 6 evidence + 3 votes + belief shifts, we expect these factors
        assert "evidence_quality" in factor_names
        assert "agent_agreement" in factor_names

        for factor in factors:
            assert "name" in factor
            assert "contribution" in factor
            assert "explanation" in factor
            assert "trend" in factor
            assert isinstance(factor["contribution"], (int, float))

        # Verify narrative is non-empty
        assert len(meta["narrative"]) > 0

        # Verify leading position reflects majority vote
        assert meta["leading_position"] is not None

        # Verify agreement is > 0 (2/3 agents agree on microservices_strangler)
        assert meta["agent_agreement"] > 0

    @pytest.mark.asyncio
    async def test_factors_include_confidence_weighted_consensus(self):
        """When votes are present, confidence_weighted_consensus factor should appear."""
        arena = _FakeArena()
        state = _make_execution_state()

        await setup_debate_infrastructure(arena, state)

        bus = arena.event_bus
        bus.emit_sync(
            "agent_message",
            debate_id="e2e",
            agent="claude",
            content="Proposal A",
            role="proposer",
            round_num=1,
        )
        bus.emit_sync(
            "vote",
            debate_id="e2e",
            agent="claude",
            choice="A",
            confidence=0.9,
            round_num=2,
        )
        bus.emit_sync(
            "vote",
            debate_id="e2e",
            agent="gpt4",
            choice="A",
            confidence=0.8,
            round_num=2,
        )

        await handle_debate_completion(arena, state)

        factors = state.ctx.result.metadata["live_explainability"]["factors"]
        factor_names = {f["name"] for f in factors}
        assert "confidence_weighted_consensus" in factor_names

    @pytest.mark.asyncio
    async def test_belief_shifts_tracked_through_refinement(self):
        """Refinements that change position should generate belief_shifts in metadata."""
        arena = _FakeArena()
        state = _make_execution_state()

        await setup_debate_infrastructure(arena, state)

        bus = arena.event_bus
        # Initial proposal
        bus.emit_sync(
            "agent_message",
            debate_id="e2e",
            agent="claude",
            content="Original position on microservices",
            role="proposer",
            round_num=1,
        )
        # Critique forces revision
        bus.emit_sync(
            "agent_message",
            debate_id="e2e",
            agent="gpt4",
            content="That approach has scaling issues",
            role="critic",
            round_num=1,
        )
        # Refinement changes position
        bus.emit_sync(
            "agent_message",
            debate_id="e2e",
            agent="claude",
            content="Revised: use strangler-fig pattern instead",
            role="reviser",
            round_num=2,
        )

        await handle_debate_completion(arena, state)

        meta = state.ctx.result.metadata["live_explainability"]
        # At least one belief shift from the refinement
        assert meta["belief_shifts"] >= 1

    @pytest.mark.asyncio
    async def test_vote_flip_detected_in_factors(self):
        """When an agent flips vote, belief_stability factor should appear."""
        arena = _FakeArena()
        state = _make_execution_state()

        await setup_debate_infrastructure(arena, state)

        bus = arena.event_bus
        bus.emit_sync(
            "agent_message",
            debate_id="e2e",
            agent="claude",
            content="Proposal A",
            role="proposer",
            round_num=1,
        )
        # First vote
        bus.emit_sync(
            "vote",
            debate_id="e2e",
            agent="claude",
            choice="monolith",
            confidence=0.6,
            round_num=1,
        )
        # Flipped vote
        bus.emit_sync(
            "vote",
            debate_id="e2e",
            agent="claude",
            choice="microservices",
            confidence=0.8,
            round_num=2,
        )

        await handle_debate_completion(arena, state)

        meta = state.ctx.result.metadata["live_explainability"]
        assert meta["belief_shifts"] >= 1

        factor_names = {f["name"] for f in meta["factors"]}
        assert "belief_stability" in factor_names

    @pytest.mark.asyncio
    async def test_metadata_is_json_serializable(self):
        """Receipt metadata must be JSON-serializable for storage."""
        import json

        arena = _FakeArena()
        state = _make_execution_state()

        await setup_debate_infrastructure(arena, state)

        bus = arena.event_bus
        bus.emit_sync(
            "agent_message",
            debate_id="e2e",
            agent="claude",
            content="Proposal",
            role="proposer",
            round_num=1,
        )
        bus.emit_sync(
            "vote",
            debate_id="e2e",
            agent="claude",
            choice="yes",
            confidence=0.9,
            round_num=2,
        )

        await handle_debate_completion(arena, state)

        meta = state.ctx.result.metadata["live_explainability"]
        # Must not raise
        serialized = json.dumps(meta)
        assert isinstance(serialized, str)
        roundtripped = json.loads(serialized)
        assert roundtripped["evidence_count"] == meta["evidence_count"]

    @pytest.mark.asyncio
    async def test_no_metadata_when_explainability_disabled(self):
        """When enable_live_explainability is False, no metadata should be attached."""
        arena = _FakeArena()
        arena.enable_live_explainability = False
        state = _make_execution_state()

        await setup_debate_infrastructure(arena, state)
        assert arena.live_explainability_stream is None

        await handle_debate_completion(arena, state)
        assert "live_explainability" not in state.ctx.result.metadata
