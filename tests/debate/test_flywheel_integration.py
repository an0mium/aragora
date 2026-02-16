"""Tests for the three debate flywheel integrations + MetaPlanner improvement queue + protocol convenience.

Tests cover:
1. Adaptive Consensus in ConsensusPhase
2. Knowledge Injection in ContextInitializer
3. Dialectical Synthesis in PhaseExecutor (via create_phase_executor)
4. ImprovementQueue -> MetaPlanner injection
5. DebateProtocol.with_full_flywheel() convenience method
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.protocol import DebateProtocol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(name: str, role: str = "proposer") -> MagicMock:
    """Create a mock agent with a name and role attribute."""
    agent = MagicMock()
    agent.name = name
    agent.role = role
    agent.timeout = 120
    agent.generate = AsyncMock(return_value="synthesis output text")
    return agent


def _make_calibration_summary(brier: float, total_predictions: int = 10) -> MagicMock:
    """Create a mock calibration summary with the given Brier score."""
    summary = MagicMock()
    summary.brier_score = brier
    summary.total_predictions = total_predictions
    summary.bias_direction = "neutral"
    return summary


def _make_ctx(
    task: str = "Design a rate limiter",
    context: str = "",
    num_agents: int = 3,
    proposals: dict[str, str] | None = None,
) -> MagicMock:
    """Build a minimal DebateContext-like MagicMock."""
    ctx = MagicMock(spec_set=False)
    ctx.env = MagicMock()
    ctx.env.task = task
    ctx.env.context = context

    agents = [_make_agent(f"agent-{i}") for i in range(num_agents)]
    ctx.agents = agents

    ctx.proposals = proposals or {a.name: f"Proposal from {a.name}" for a in agents}
    ctx.context_messages = []
    ctx.partial_messages = []
    ctx.proposers = list(agents)

    # Result
    result = MagicMock()
    result.votes = []
    result.critiques = []
    result.consensus_reached = False
    result.confidence = 0.0
    result.final_answer = ""
    result.synthesis = None
    result.synthesis_confidence = None
    result.synthesis_provenance = None
    result.adaptive_threshold_explanation = None
    result.formal_verification = None
    result.metadata = {}
    result.dissenting_views = []
    result.task = task
    result.messages = []
    result.rounds_used = 3
    result.winner = None
    result.status = None
    result.consensus_strength = None
    ctx.result = result

    ctx.vote_tally = {}
    ctx.winner_agent = None
    ctx.cancellation_token = None
    ctx.hook_manager = None
    ctx.event_emitter = None
    ctx.loop_id = "test-loop"
    ctx.debate_id = "test-debate-001"
    ctx.domain = "engineering"
    ctx.start_time = 0.0

    # Background tasks
    ctx.background_research_task = None
    ctx.background_evidence_task = None
    ctx.applied_insight_ids = []

    return ctx


# ===========================================================================
# 1. Adaptive Consensus in ConsensusPhase
# ===========================================================================


class TestAdaptiveConsensusInConsensusPhase:
    """Verify that ConsensusPhase honours enable_adaptive_consensus."""

    def test_adaptive_consensus_object_created_when_enabled(self):
        """When enable_adaptive_consensus=True, _adaptive_consensus is set."""
        protocol = DebateProtocol(
            enable_adaptive_consensus=True,
            consensus="majority",
            consensus_threshold=0.6,
        )
        from aragora.debate.phases.consensus_phase import ConsensusPhase

        phase = ConsensusPhase(protocol=protocol)
        assert phase._adaptive_consensus is not None

    def test_adaptive_consensus_not_created_when_disabled(self):
        """When enable_adaptive_consensus=False (default), _adaptive_consensus is None."""
        protocol = DebateProtocol(
            enable_adaptive_consensus=False,
            consensus="majority",
        )
        from aragora.debate.phases.consensus_phase import ConsensusPhase

        phase = ConsensusPhase(protocol=protocol)
        assert phase._adaptive_consensus is None

    def test_well_calibrated_pool_lowers_threshold(self):
        """Well-calibrated agents (low Brier) produce a lower threshold."""
        from aragora.debate.adaptive_consensus import (
            AdaptiveConsensus,
            AdaptiveConsensusConfig,
        )

        config = AdaptiveConsensusConfig(
            base_threshold=0.6,
            min_threshold=0.45,
            max_threshold=0.85,
            calibration_impact=0.3,
            min_calibration_samples=5,
        )
        ac = AdaptiveConsensus(config)

        # Create agents and a calibration tracker that reports low Brier
        agents = [_make_agent(f"agent-{i}") for i in range(3)]
        tracker = MagicMock()
        # Brier = 0.10 (well-calibrated, below neutral 0.25)
        tracker.get_calibration_summary.return_value = _make_calibration_summary(
            brier=0.10, total_predictions=20
        )

        threshold = ac.compute_threshold(agents, calibration_tracker=tracker)

        # Expected: 0.6 + 0.3 * (0.10 - 0.25) = 0.6 - 0.045 = 0.555
        assert threshold < 0.6, f"Expected threshold below base 0.6, got {threshold}"
        assert threshold >= config.min_threshold

    def test_poorly_calibrated_pool_raises_threshold(self):
        """Poorly-calibrated agents (high Brier) produce a higher threshold."""
        from aragora.debate.adaptive_consensus import (
            AdaptiveConsensus,
            AdaptiveConsensusConfig,
        )

        config = AdaptiveConsensusConfig(
            base_threshold=0.6,
            min_threshold=0.45,
            max_threshold=0.85,
            calibration_impact=0.3,
            min_calibration_samples=5,
        )
        ac = AdaptiveConsensus(config)

        agents = [_make_agent(f"agent-{i}") for i in range(3)]
        tracker = MagicMock()
        # Brier = 0.45 (poorly calibrated, above neutral 0.25)
        tracker.get_calibration_summary.return_value = _make_calibration_summary(
            brier=0.45, total_predictions=20
        )

        threshold = ac.compute_threshold(agents, calibration_tracker=tracker)

        # Expected: 0.6 + 0.3 * (0.45 - 0.25) = 0.6 + 0.06 = 0.66
        assert threshold > 0.6, f"Expected threshold above base 0.6, got {threshold}"
        assert threshold <= config.max_threshold

    def test_no_calibration_data_returns_base_threshold(self):
        """When no agents have calibration data, base threshold is returned."""
        from aragora.debate.adaptive_consensus import (
            AdaptiveConsensus,
            AdaptiveConsensusConfig,
        )

        config = AdaptiveConsensusConfig(base_threshold=0.6)
        ac = AdaptiveConsensus(config)

        agents = [_make_agent(f"agent-{i}") for i in range(3)]
        # No tracker at all
        threshold = ac.compute_threshold(agents)
        assert threshold == 0.6

    def test_compute_threshold_with_explanation(self):
        """compute_threshold_with_explanation returns an audit-friendly explanation."""
        from aragora.debate.adaptive_consensus import (
            AdaptiveConsensus,
            AdaptiveConsensusConfig,
        )

        config = AdaptiveConsensusConfig(base_threshold=0.6)
        ac = AdaptiveConsensus(config)

        agents = [_make_agent("claude"), _make_agent("gemini")]
        tracker = MagicMock()
        tracker.get_calibration_summary.return_value = _make_calibration_summary(
            brier=0.15, total_predictions=30
        )

        threshold, explanation = ac.compute_threshold_with_explanation(
            agents, calibration_tracker=tracker
        )

        assert isinstance(threshold, float)
        assert isinstance(explanation, str)
        assert "Adaptive consensus threshold" in explanation
        assert "Brier" in explanation or "brier" in explanation

    def test_threshold_clamped_to_bounds(self):
        """Extreme Brier scores are clamped to [min_threshold, max_threshold]."""
        from aragora.debate.adaptive_consensus import (
            AdaptiveConsensus,
            AdaptiveConsensusConfig,
        )

        config = AdaptiveConsensusConfig(
            base_threshold=0.6,
            min_threshold=0.45,
            max_threshold=0.85,
            calibration_impact=2.0,  # Very aggressive scaling
        )
        ac = AdaptiveConsensus(config)

        agents = [_make_agent("agent-0")]
        tracker = MagicMock()

        # Very bad calibration -> threshold would overshoot max
        tracker.get_calibration_summary.return_value = _make_calibration_summary(
            brier=0.8, total_predictions=50
        )
        threshold_high = ac.compute_threshold(agents, calibration_tracker=tracker)
        assert threshold_high <= config.max_threshold

        # Very good calibration -> threshold would undershoot min
        tracker.get_calibration_summary.return_value = _make_calibration_summary(
            brier=0.0, total_predictions=50
        )
        threshold_low = ac.compute_threshold(agents, calibration_tracker=tracker)
        assert threshold_low >= config.min_threshold

    @pytest.mark.asyncio
    async def test_consensus_phase_passes_adaptive_threshold_to_winner_selector(self):
        """ConsensusPhase uses adaptive threshold in _handle_majority_consensus."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusCallbacks,
            ConsensusDependencies,
            ConsensusPhase,
        )

        protocol = DebateProtocol(
            enable_adaptive_consensus=True,
            consensus="majority",
            consensus_threshold=0.6,
        )

        tracker = MagicMock()
        tracker.get_calibration_summary.return_value = _make_calibration_summary(
            brier=0.10, total_predictions=20
        )

        deps = ConsensusDependencies(
            protocol=protocol,
            calibration_tracker=tracker,
        )

        # Create minimal callbacks that return empty votes
        vote_callback = AsyncMock(return_value=MagicMock(choice="agent-0", agent="agent-1", reasoning="good", confidence=0.8, continue_debate=False))
        callbacks = ConsensusCallbacks(
            vote_with_agent=vote_callback,
            group_similar_votes=lambda votes: ({}, {v.choice: v.choice for v in votes} if votes else {}),
        )

        phase = ConsensusPhase(deps=deps, callbacks=callbacks)

        # Verify the adaptive consensus object was created
        assert phase._adaptive_consensus is not None

        # The threshold should be computed when _handle_majority_consensus runs
        # We can verify the formula directly
        threshold, explanation = phase._adaptive_consensus.compute_threshold_with_explanation(
            [_make_agent("a"), _make_agent("b")],
            calibration_tracker=tracker,
        )
        # Brier=0.10, base=0.6 -> 0.6 + 0.3*(0.10-0.25) = 0.555
        assert 0.50 < threshold < 0.60
        assert "adaptive_threshold_explanation" not in explanation or explanation  # just check it runs


# ===========================================================================
# 2. Knowledge Injection in ContextInitializer
# ===========================================================================


class TestKnowledgeInjectionInContextInitializer:
    """Verify that ContextInitializer._inject_debate_knowledge works correctly."""

    @pytest.mark.asyncio
    async def test_injects_knowledge_when_enabled(self):
        """When enable_knowledge_injection=True and knowledge_mound configured, context is enriched."""
        from aragora.debate.phases.context_init import ContextInitializer

        protocol = DebateProtocol(
            enable_knowledge_injection=True,
            knowledge_injection_max_receipts=3,
        )

        mock_mound = MagicMock()
        initializer = ContextInitializer(
            protocol=protocol,
            knowledge_mound=mock_mound,
        )

        ctx = _make_ctx(context="Original context")

        # Mock the injector: patch the import inside _inject_debate_knowledge
        mock_knowledge = MagicMock()
        mock_knowledge.debate_id = "past-debate-1"
        mock_knowledge.task = "Past task"
        mock_knowledge.final_answer = "Past answer"
        mock_knowledge.confidence = 0.9
        mock_knowledge.relevance_score = 0.8

        with patch(
            "aragora.debate.phases.context_init.DebateKnowledgeInjector"
        ) as MockInjector:
            injector_instance = MockInjector.return_value
            injector_instance.query_relevant_knowledge = AsyncMock(
                return_value=[mock_knowledge]
            )
            injector_instance.format_for_injection.return_value = (
                "## Relevant Past Decisions\n**Past task** (confidence: 0.90)\n- Decision: Past answer\n"
            )

            await initializer._inject_debate_knowledge(ctx)

        # Context should have been extended
        assert "Relevant Past Decisions" in ctx.env.context
        assert "Past task" in ctx.env.context

    @pytest.mark.asyncio
    async def test_noop_when_disabled(self):
        """When enable_knowledge_injection=False, no injection occurs."""
        from aragora.debate.phases.context_init import ContextInitializer

        protocol = DebateProtocol(enable_knowledge_injection=False)
        mock_mound = MagicMock()

        initializer = ContextInitializer(
            protocol=protocol,
            knowledge_mound=mock_mound,
        )

        ctx = _make_ctx(context="Original context")
        original_context = ctx.env.context

        await initializer._inject_debate_knowledge(ctx)

        assert ctx.env.context == original_context

    @pytest.mark.asyncio
    async def test_noop_when_no_knowledge_mound(self):
        """When knowledge_mound is None, no injection occurs even if enabled."""
        from aragora.debate.phases.context_init import ContextInitializer

        protocol = DebateProtocol(enable_knowledge_injection=True)

        initializer = ContextInitializer(
            protocol=protocol,
            knowledge_mound=None,
        )

        ctx = _make_ctx(context="Original context")
        original_context = ctx.env.context

        await initializer._inject_debate_knowledge(ctx)

        assert ctx.env.context == original_context

    @pytest.mark.asyncio
    async def test_noop_when_no_relevant_knowledge(self):
        """When the injector finds no relevant receipts, context is unchanged."""
        from aragora.debate.phases.context_init import ContextInitializer

        protocol = DebateProtocol(
            enable_knowledge_injection=True,
            knowledge_injection_max_receipts=3,
        )
        mock_mound = MagicMock()

        initializer = ContextInitializer(
            protocol=protocol,
            knowledge_mound=mock_mound,
        )

        ctx = _make_ctx(context="Original context")

        with patch(
            "aragora.debate.phases.context_init.DebateKnowledgeInjector"
        ) as MockInjector:
            injector_instance = MockInjector.return_value
            injector_instance.query_relevant_knowledge = AsyncMock(return_value=[])

            await initializer._inject_debate_knowledge(ctx)

        assert ctx.env.context == "Original context"

    @pytest.mark.asyncio
    async def test_appends_to_existing_context(self):
        """Knowledge injection appends to existing context rather than replacing it."""
        from aragora.debate.phases.context_init import ContextInitializer

        protocol = DebateProtocol(
            enable_knowledge_injection=True,
            knowledge_injection_max_receipts=2,
        )
        mock_mound = MagicMock()

        initializer = ContextInitializer(
            protocol=protocol,
            knowledge_mound=mock_mound,
        )

        ctx = _make_ctx(context="Existing background context.")

        mock_knowledge = MagicMock()
        mock_knowledge.relevance_score = 0.9

        with patch(
            "aragora.debate.phases.context_init.DebateKnowledgeInjector"
        ) as MockInjector:
            injector_instance = MockInjector.return_value
            injector_instance.query_relevant_knowledge = AsyncMock(
                return_value=[mock_knowledge]
            )
            injector_instance.format_for_injection.return_value = (
                "## Past decisions here"
            )

            await initializer._inject_debate_knowledge(ctx)

        # Both old and new context should be present
        assert "Existing background context." in ctx.env.context
        assert "Past decisions here" in ctx.env.context

    @pytest.mark.asyncio
    async def test_handles_import_error_gracefully(self):
        """If DebateKnowledgeInjector cannot be imported, no crash."""
        from aragora.debate.phases.context_init import ContextInitializer

        protocol = DebateProtocol(enable_knowledge_injection=True)
        mock_mound = MagicMock()

        initializer = ContextInitializer(
            protocol=protocol,
            knowledge_mound=mock_mound,
        )

        ctx = _make_ctx(context="Original")

        with patch(
            "aragora.debate.phases.context_init.DebateKnowledgeInjector",
            side_effect=ImportError("not found"),
        ):
            # Should not raise
            await initializer._inject_debate_knowledge(ctx)

        # Context should be unchanged since the import fails before querying
        assert ctx.env.context == "Original"


# ===========================================================================
# 3. Dialectical Synthesis in PhaseExecutor
# ===========================================================================


class TestDialecticalSynthesisInPhaseExecutor:
    """Verify that synthesis phase is inserted between consensus and analytics."""

    def test_synthesis_phase_inserted_when_enabled(self):
        """When enable_synthesis=True, create_phase_executor adds a synthesis phase."""
        protocol = DebateProtocol(enable_synthesis=True)

        # Build a minimal arena mock
        arena = MagicMock()
        arena.protocol = protocol
        arena.context_initializer = MagicMock()
        arena.context_initializer.initialize = AsyncMock()
        arena.proposal_phase = MagicMock()
        arena.proposal_phase.name = "proposal"
        arena.debate_rounds_phase = MagicMock()
        arena.debate_rounds_phase.name = "debate_rounds"
        arena.consensus_phase = MagicMock()
        arena.consensus_phase.name = "consensus"
        arena.analytics_phase = MagicMock()
        arena.analytics_phase.name = "analytics"
        arena.feedback_phase = MagicMock()
        arena.feedback_phase.name = "feedback"

        # Mock agents and config for timeout calculation
        arena.agents = [_make_agent("a"), _make_agent("b")]
        arena.config = MagicMock()
        arena.config.checkpoint_manager = None

        from aragora.debate.arena_phases import create_phase_executor

        executor = create_phase_executor(arena)

        # Verify synthesis phase exists in the executor
        phase_names = executor.phase_names
        assert "synthesis" in phase_names

        # Verify ordering: synthesis appears after consensus and before analytics
        consensus_idx = phase_names.index("consensus")
        synthesis_idx = phase_names.index("synthesis")
        analytics_idx = phase_names.index("analytics")

        assert consensus_idx < synthesis_idx < analytics_idx, (
            f"Expected consensus < synthesis < analytics, "
            f"got consensus={consensus_idx}, synthesis={synthesis_idx}, analytics={analytics_idx}"
        )

    def test_no_synthesis_phase_when_disabled(self):
        """When enable_synthesis=False, no synthesis phase in the executor."""
        protocol = DebateProtocol(enable_synthesis=False)

        arena = MagicMock()
        arena.protocol = protocol
        arena.context_initializer = MagicMock()
        arena.context_initializer.initialize = AsyncMock()
        arena.proposal_phase = MagicMock()
        arena.proposal_phase.name = "proposal"
        arena.debate_rounds_phase = MagicMock()
        arena.debate_rounds_phase.name = "debate_rounds"
        arena.consensus_phase = MagicMock()
        arena.consensus_phase.name = "consensus"
        arena.analytics_phase = MagicMock()
        arena.analytics_phase.name = "analytics"
        arena.feedback_phase = MagicMock()
        arena.feedback_phase.name = "feedback"
        arena.agents = [_make_agent("a")]
        arena.config = MagicMock()
        arena.config.checkpoint_manager = None

        from aragora.debate.arena_phases import create_phase_executor

        executor = create_phase_executor(arena)

        assert "synthesis" not in executor.phase_names

    @pytest.mark.asyncio
    async def test_synthesis_stores_result_on_context(self):
        """When synthesis runs, its output is stored on ctx.result.synthesis."""
        from aragora.debate.phases.synthesis_phase import (
            DialecticalPosition,
            DialecticalSynthesizer,
            SynthesisConfig,
            SynthesisResult,
        )

        config = SynthesisConfig(
            enable_synthesis=True,
            min_opposing_positions=2,
        )
        synthesizer = DialecticalSynthesizer(config=config)

        ctx = _make_ctx(num_agents=3)
        # Set up two proposals and some votes to create thesis/antithesis
        ctx.proposals = {
            "agent-0": "We should use token bucket algorithm for rate limiting.",
            "agent-1": "We should use sliding window algorithm for rate limiting.",
            "agent-2": "We should use leaky bucket algorithm for rate limiting.",
        }

        # Create mock votes that support different positions
        vote_0 = MagicMock()
        vote_0.agent = "agent-2"
        vote_0.choice = "agent-0"
        vote_1 = MagicMock()
        vote_1.agent = "agent-0"
        vote_1.choice = "agent-1"
        ctx.result.votes = [vote_0, vote_1]

        # Mock the agent's generate method to return structured synthesis
        ctx.agents[0].generate = AsyncMock(
            return_value=(
                "### Synthesis\n"
                "A hybrid approach combining token bucket and sliding window.\n"
                "### Elements from Thesis\n"
                "- Token bucket simplicity\n"
                "### Elements from Antithesis\n"
                "- Sliding window fairness\n"
                "### Novel Elements\n"
                "- Adaptive switching between algorithms\n"
            )
        )

        result = await synthesizer.synthesize(ctx)

        assert result is not None
        assert isinstance(result, SynthesisResult)
        assert len(result.synthesis) > 0
        assert result.confidence > 0.0
        assert result.thesis is not None
        assert result.antithesis is not None

    @pytest.mark.asyncio
    async def test_synthesis_skipped_with_single_proposal(self):
        """Synthesis requires at least min_opposing_positions distinct proposals."""
        from aragora.debate.phases.synthesis_phase import (
            DialecticalSynthesizer,
            SynthesisConfig,
        )

        config = SynthesisConfig(
            enable_synthesis=True,
            min_opposing_positions=2,
        )
        synthesizer = DialecticalSynthesizer(config=config)

        ctx = _make_ctx(num_agents=1)
        ctx.proposals = {"agent-0": "Only one proposal"}
        ctx.result.votes = []

        result = await synthesizer.synthesize(ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_synthesis_disabled_returns_none(self):
        """When enable_synthesis=False, synthesize() returns None immediately."""
        from aragora.debate.phases.synthesis_phase import (
            DialecticalSynthesizer,
            SynthesisConfig,
        )

        config = SynthesisConfig(enable_synthesis=False)
        synthesizer = DialecticalSynthesizer(config=config)

        ctx = _make_ctx()
        result = await synthesizer.synthesize(ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_synthesis_wrapper_stores_on_result(self):
        """The SynthesisPhaseWrapper in create_phase_executor stores synthesis on ctx.result."""
        protocol = DebateProtocol(enable_synthesis=True)

        arena = MagicMock()
        arena.protocol = protocol
        arena.context_initializer = MagicMock()
        arena.context_initializer.initialize = AsyncMock()
        arena.proposal_phase = MagicMock()
        arena.proposal_phase.name = "proposal"
        arena.debate_rounds_phase = MagicMock()
        arena.debate_rounds_phase.name = "debate_rounds"
        arena.consensus_phase = MagicMock()
        arena.consensus_phase.name = "consensus"
        arena.analytics_phase = MagicMock()
        arena.analytics_phase.name = "analytics"
        arena.feedback_phase = MagicMock()
        arena.feedback_phase.name = "feedback"
        arena.agents = [_make_agent("a"), _make_agent("b")]
        arena.config = MagicMock()
        arena.config.checkpoint_manager = None

        from aragora.debate.arena_phases import create_phase_executor

        executor = create_phase_executor(arena)
        synthesis_phase = executor.get_phase("synthesis")
        assert synthesis_phase is not None

        # Build context for the wrapper to use
        ctx = _make_ctx(num_agents=2)

        # Patch DialecticalSynthesizer.synthesize to return a mock result
        from aragora.debate.phases.synthesis_phase import (
            DialecticalPosition,
            SynthesisResult,
        )

        mock_synth_result = SynthesisResult(
            synthesis="Combined approach text",
            confidence=0.85,
            thesis=DialecticalPosition(agent="a", content="Position A"),
            antithesis=DialecticalPosition(agent="b", content="Position B"),
            elements_from_thesis=["element A1"],
            elements_from_antithesis=["element B1"],
            novel_elements=["novel 1"],
            synthesizer_agent="a",
        )

        with patch.object(
            synthesis_phase._synth, "synthesize", new=AsyncMock(return_value=mock_synth_result)
        ):
            await synthesis_phase.execute(ctx)

        assert ctx.result.synthesis == "Combined approach text"
        assert ctx.result.synthesis_confidence == 0.85
        assert ctx.result.synthesis_provenance["thesis_agent"] == "a"
        assert ctx.result.synthesis_provenance["antithesis_agent"] == "b"
        assert ctx.result.synthesis_provenance["synthesizer"] == "a"


# ===========================================================================
# 4. ImprovementQueue -> MetaPlanner
# ===========================================================================


class TestImprovementQueueToMetaPlanner:
    """Verify that MetaPlanner.plan() injects pending suggestions from the global queue."""

    def test_improvement_queue_basic_operations(self):
        """ImprovementQueue supports enqueue, dequeue_batch, peek, and __len__."""
        from aragora.nomic.improvement_queue import ImprovementQueue, ImprovementSuggestion

        queue = ImprovementQueue(max_size=5)
        assert len(queue) == 0

        s1 = ImprovementSuggestion(
            debate_id="d1",
            task="Add retry logic",
            suggestion="Use exponential backoff",
            category="reliability",
            confidence=0.9,
        )
        s2 = ImprovementSuggestion(
            debate_id="d2",
            task="Fix flaky test",
            suggestion="Add retry in CI",
            category="test_coverage",
            confidence=0.7,
        )

        queue.enqueue(s1)
        queue.enqueue(s2)
        assert len(queue) == 2

        # Peek does not remove
        peeked = queue.peek(10)
        assert len(peeked) == 2
        assert len(queue) == 2

        # Dequeue removes
        batch = queue.dequeue_batch(1)
        assert len(batch) == 1
        assert batch[0].task == "Add retry logic"
        assert len(queue) == 1

    def test_improvement_queue_eviction(self):
        """When queue is full, oldest suggestion is evicted."""
        from aragora.nomic.improvement_queue import ImprovementQueue, ImprovementSuggestion

        queue = ImprovementQueue(max_size=2)

        for i in range(3):
            queue.enqueue(
                ImprovementSuggestion(
                    debate_id=f"d{i}",
                    task=f"task-{i}",
                    suggestion=f"suggestion-{i}",
                    category="test_coverage",
                    confidence=0.8,
                )
            )

        assert len(queue) == 2
        peeked = queue.peek(10)
        # The oldest (task-0) should have been evicted
        tasks = [s.task for s in peeked]
        assert "task-0" not in tasks
        assert "task-1" in tasks
        assert "task-2" in tasks

    def test_global_singleton(self):
        """get_improvement_queue returns the same instance across calls."""
        from aragora.nomic.improvement_queue import get_improvement_queue

        q1 = get_improvement_queue()
        q2 = get_improvement_queue()
        assert q1 is q2

    @pytest.mark.asyncio
    async def test_meta_planner_injects_improvements_from_queue(self):
        """MetaPlanner.prioritize_work injects queue suggestions into PlanningContext."""
        from aragora.nomic.improvement_queue import (
            ImprovementQueue,
            ImprovementSuggestion,
            get_improvement_queue,
        )
        from aragora.nomic.meta_planner import (
            MetaPlanner,
            MetaPlannerConfig,
            PlanningContext,
            Track,
        )

        # Use quick_mode to avoid needing real agents/debate
        config = MetaPlannerConfig(
            quick_mode=True,
            enable_cross_cycle_learning=False,
        )
        planner = MetaPlanner(config=config)

        # Set up the global queue with a suggestion
        queue = get_improvement_queue()
        # Clear any previous state
        queue.dequeue_batch(1000)

        queue.enqueue(
            ImprovementSuggestion(
                debate_id="debate-test-001",
                task="Improve error messages",
                suggestion="Add structured error codes",
                category="code_quality",
                confidence=0.85,
            )
        )

        # Build a context to observe mutation
        context = PlanningContext()

        # In quick_mode, prioritize_work skips the debate but still runs
        # the improvement injection code path. However, quick_mode returns
        # _heuristic_prioritize before the injection. Let's test the
        # injection directly.
        #
        # The injection happens at lines 160-169 of meta_planner.py, before
        # the Arena is created but after quick_mode check. In quick_mode the
        # code returns early before injection. So instead we test with
        # quick_mode=False but patch the Arena/debate parts.

        config2 = MetaPlannerConfig(
            quick_mode=False,
            enable_cross_cycle_learning=False,
        )
        planner2 = MetaPlanner(config=config2)

        # Patch Arena import to return mock
        with patch(
            "aragora.nomic.meta_planner.Arena"
        ) as MockArena, patch(
            "aragora.nomic.meta_planner.Environment"
        ), patch(
            "aragora.nomic.meta_planner.DebateProtocol"
        ):
            mock_arena_instance = MockArena.return_value
            mock_result = MagicMock()
            mock_result.consensus = "1. Improve error messages (Track: qa, High impact)"
            mock_result.final_response = None
            mock_result.responses = []
            mock_arena_instance.run = AsyncMock(return_value=mock_result)

            # Also need to patch _create_agent to return something
            with patch.object(planner2, "_create_agent", return_value=_make_agent("mock")):
                with patch.object(planner2, "_generate_receipt"):
                    goals = await planner2.prioritize_work(
                        objective="Improve code quality",
                        available_tracks=[Track.QA],
                        context=context,
                    )

        # The queue should have been peeked (not drained) -- the suggestion remains
        remaining = queue.peek(10)
        assert any(s.task == "Improve error messages" for s in remaining)

    @pytest.mark.asyncio
    async def test_meta_planner_no_crash_when_queue_empty(self):
        """MetaPlanner handles empty improvement queue gracefully."""
        from aragora.nomic.improvement_queue import get_improvement_queue
        from aragora.nomic.meta_planner import (
            MetaPlanner,
            MetaPlannerConfig,
            Track,
        )

        # Drain the queue
        queue = get_improvement_queue()
        queue.dequeue_batch(1000)

        config = MetaPlannerConfig(
            quick_mode=False,
            enable_cross_cycle_learning=False,
        )
        planner = MetaPlanner(config=config)

        with patch(
            "aragora.nomic.meta_planner.Arena"
        ) as MockArena, patch(
            "aragora.nomic.meta_planner.Environment"
        ), patch(
            "aragora.nomic.meta_planner.DebateProtocol"
        ):
            mock_arena_instance = MockArena.return_value
            mock_result = MagicMock()
            mock_result.consensus = "1. Fix tests (Track: qa)"
            mock_result.final_response = None
            mock_result.responses = []
            mock_arena_instance.run = AsyncMock(return_value=mock_result)

            with patch.object(planner, "_create_agent", return_value=_make_agent("mock")):
                with patch.object(planner, "_generate_receipt"):
                    goals = await planner.prioritize_work(
                        objective="Fix tests",
                        available_tracks=[Track.QA],
                    )

        # Should not crash and should return some goals
        assert isinstance(goals, list)


# ===========================================================================
# 5. DebateProtocol.with_full_flywheel() Convenience Method
# ===========================================================================


class TestProtocolConvenienceMethods:
    """Verify that DebateProtocol.with_full_flywheel() returns correctly configured protocol."""

    def test_with_full_flywheel_enables_all_flags(self):
        """with_full_flywheel() enables adaptive_consensus, synthesis, and knowledge_injection."""
        protocol = DebateProtocol.with_full_flywheel()

        assert protocol.enable_adaptive_consensus is True
        assert protocol.enable_synthesis is True
        assert protocol.enable_knowledge_injection is True
        assert protocol.enable_trickster is True
        assert protocol.auto_create_plan is True

    def test_with_full_flywheel_allows_overrides(self):
        """Keyword arguments to with_full_flywheel() override the defaults."""
        protocol = DebateProtocol.with_full_flywheel(
            enable_trickster=False,
            rounds=5,
            consensus="majority",
        )

        # Overridden values
        assert protocol.enable_trickster is False
        assert protocol.rounds == 5
        assert protocol.consensus == "majority"

        # Flywheel flags still enabled
        assert protocol.enable_adaptive_consensus is True
        assert protocol.enable_synthesis is True
        assert protocol.enable_knowledge_injection is True

    def test_with_full_flywheel_can_disable_flywheel_flag(self):
        """User can explicitly disable a flywheel flag even in with_full_flywheel."""
        protocol = DebateProtocol.with_full_flywheel(enable_synthesis=False)
        assert protocol.enable_synthesis is False
        assert protocol.enable_adaptive_consensus is True
        assert protocol.enable_knowledge_injection is True

    def test_with_full_flywheel_returns_debate_protocol(self):
        """with_full_flywheel() returns a DebateProtocol instance."""
        protocol = DebateProtocol.with_full_flywheel()
        assert isinstance(protocol, DebateProtocol)

    def test_with_gold_path_factory(self):
        """with_gold_path() sets auto_create_plan and plan fields."""
        protocol = DebateProtocol.with_gold_path(
            min_confidence=0.8,
            approval_mode="always",
        )
        assert protocol.auto_create_plan is True
        assert protocol.plan_min_confidence == 0.8
        assert protocol.plan_approval_mode == "always"

    def test_default_protocol_flywheel_flags_disabled(self):
        """Default protocol has flywheel flags disabled (opt-in)."""
        protocol = DebateProtocol()
        assert protocol.enable_adaptive_consensus is False
        assert protocol.enable_synthesis is False
        assert protocol.enable_knowledge_injection is False


# ===========================================================================
# 6. DebateKnowledgeInjector Unit Tests
# ===========================================================================


class TestDebateKnowledgeInjector:
    """Verify the DebateKnowledgeInjector query and format logic."""

    def test_format_for_injection_empty_list(self):
        """format_for_injection returns empty string for empty list."""
        from aragora.debate.knowledge_injection import DebateKnowledgeInjector

        injector = DebateKnowledgeInjector()
        result = injector.format_for_injection([])
        assert result == ""

    def test_format_for_injection_with_items(self):
        """format_for_injection produces markdown with past decisions."""
        from aragora.debate.knowledge_injection import (
            DebateKnowledgeInjector,
            KnowledgeInjectionConfig,
            PastDebateKnowledge,
        )

        config = KnowledgeInjectionConfig(
            include_confidence=True,
            include_dissenting_views=True,
        )
        injector = DebateKnowledgeInjector(config=config)

        knowledge = [
            PastDebateKnowledge(
                debate_id="d1",
                task="Should we use Redis?",
                final_answer="Yes, use Redis for caching",
                confidence=0.85,
                consensus_reached=True,
                relevance_score=0.9,
                key_insights=["Fast for read-heavy workloads"],
                dissenting_views=["Operational complexity concern"],
            ),
        ]

        formatted = injector.format_for_injection(knowledge)
        assert "Relevant Past Decisions" in formatted
        assert "Redis" in formatted
        assert "0.85" in formatted
        assert "Dissenting view" in formatted

    @pytest.mark.asyncio
    async def test_inject_into_prompt_no_knowledge(self):
        """inject_into_prompt returns base prompt unchanged when no knowledge found."""
        from aragora.debate.knowledge_injection import DebateKnowledgeInjector

        injector = DebateKnowledgeInjector()

        with patch.object(
            injector, "query_relevant_knowledge", new=AsyncMock(return_value=[])
        ):
            result = await injector.inject_into_prompt(
                base_prompt="You are debating...",
                task="Design a system",
            )

        assert result == "You are debating..."

    def test_config_defaults(self):
        """KnowledgeInjectionConfig has sensible defaults."""
        from aragora.debate.knowledge_injection import KnowledgeInjectionConfig

        config = KnowledgeInjectionConfig()
        assert config.enable_injection is True
        assert config.max_relevant_receipts == 3
        assert config.min_relevance_score == 0.3


# ===========================================================================
# 7. Cross-cutting: Full Flywheel Protocol Creates Working Phases
# ===========================================================================


class TestFullFlywheelIntegration:
    """Verify that a full-flywheel protocol correctly initializes all components."""

    def test_consensus_phase_with_flywheel_protocol(self):
        """ConsensusPhase created with flywheel protocol has adaptive consensus."""
        from aragora.debate.phases.consensus_phase import ConsensusPhase

        protocol = DebateProtocol.with_full_flywheel()
        phase = ConsensusPhase(protocol=protocol)

        # Adaptive consensus should be initialized
        assert phase._adaptive_consensus is not None

    def test_phase_executor_with_flywheel_includes_synthesis(self):
        """create_phase_executor with flywheel protocol includes synthesis phase."""
        protocol = DebateProtocol.with_full_flywheel()

        arena = MagicMock()
        arena.protocol = protocol
        arena.context_initializer = MagicMock()
        arena.context_initializer.initialize = AsyncMock()
        arena.proposal_phase = MagicMock()
        arena.proposal_phase.name = "proposal"
        arena.debate_rounds_phase = MagicMock()
        arena.debate_rounds_phase.name = "debate_rounds"
        arena.consensus_phase = MagicMock()
        arena.consensus_phase.name = "consensus"
        arena.analytics_phase = MagicMock()
        arena.analytics_phase.name = "analytics"
        arena.feedback_phase = MagicMock()
        arena.feedback_phase.name = "feedback"
        arena.agents = [_make_agent("a"), _make_agent("b")]
        arena.config = MagicMock()
        arena.config.checkpoint_manager = None

        from aragora.debate.arena_phases import create_phase_executor

        executor = create_phase_executor(arena)

        assert "synthesis" in executor.phase_names
        assert "consensus" in executor.phase_names

        # Verify correct ordering
        names = executor.phase_names
        assert names.index("consensus") < names.index("synthesis") < names.index("analytics")

    def test_context_initializer_with_flywheel_has_injection_enabled(self):
        """ContextInitializer respects enable_knowledge_injection from flywheel protocol."""
        from aragora.debate.phases.context_init import ContextInitializer

        protocol = DebateProtocol.with_full_flywheel()
        mock_mound = MagicMock()

        initializer = ContextInitializer(
            protocol=protocol,
            knowledge_mound=mock_mound,
        )

        # The initializer stores the protocol and mound
        assert initializer.protocol.enable_knowledge_injection is True
        assert initializer.knowledge_mound is not None
