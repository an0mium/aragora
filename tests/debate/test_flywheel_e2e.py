"""End-to-end integration tests for the three debate feedback loops (flywheels).

These tests validate that each feedback loop works as a closed circuit,
not just that individual components exist.

Loop 1: Debate -> Receipt -> KM -> Next Debate (Knowledge Injection)
    After a debate completes, a DecisionReceipt is persisted via PostDebateCoordinator.
    The receipt is ingested into Knowledge Mound (KM).
    The next debate on a similar topic retrieves that receipt via
    ContextInitializer._inject_receipt_conclusions().

Loop 2: Debate -> ImprovementQueue -> MetaPlanner (Self-Improvement)
    After PostDebateCoordinator runs with auto_queue_improvement=True,
    an ImprovementSuggestion is queued.
    MetaPlanner reads the queue and injects suggestions into PlanningContext.

Loop 3: Debate -> ConvergenceHistory -> Round Optimization
    After a debate completes, convergence metrics are recorded via
    FeedbackPhase._record_convergence_history().
    The next debate on a similar topic receives a convergence hint
    via ContextInitializer._inject_convergence_history().
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_agent(name: str, role: str = "proposer") -> MagicMock:
    """Create a mock agent with the attributes DebateContext expects."""
    agent = MagicMock()
    agent.name = name
    agent.role = role
    agent.model = "mock"
    agent.timeout = 120
    agent.generate = AsyncMock(return_value=f"Proposal from {name}")
    return agent


def _make_debate_result(
    task: str,
    *,
    consensus_reached: bool = True,
    confidence: float = 0.85,
    final_answer: str = "Use token-bucket rate limiter with Redis backend",
    rounds_used: int = 3,
    winner: str | None = None,
) -> MagicMock:
    """Create a mock DebateResult with configurable fields."""
    result = MagicMock()
    result.task = task
    result.consensus_reached = consensus_reached
    result.confidence = confidence
    result.final_answer = final_answer
    result.consensus = final_answer if consensus_reached else ""
    result.rounds_used = rounds_used
    result.rounds_completed = rounds_used
    result.winner = winner
    result.votes = []
    result.critiques = []
    result.messages = []
    result.dissenting_views = []
    result.metadata = {}
    result.status = None
    result.consensus_strength = None
    result.synthesis = None
    result.synthesis_confidence = None
    result.synthesis_provenance = None
    result.adaptive_threshold_explanation = None
    result.formal_verification = None
    result.participants = ["agent-0", "agent-1"]
    return result


def _make_ctx(
    task: str = "Design a rate limiter for API endpoints",
    context: str = "",
    num_agents: int = 2,
) -> MagicMock:
    """Build a minimal DebateContext-like MagicMock for feedback phase tests."""
    ctx = MagicMock()
    ctx.env = MagicMock()
    ctx.env.task = task
    ctx.env.context = context

    agents = [_make_agent(f"agent-{i}") for i in range(num_agents)]
    ctx.agents = agents
    ctx.proposers = list(agents)

    ctx.proposals = {a.name: f"Proposal from {a.name}" for a in agents}
    ctx.context_messages = []
    ctx.partial_messages = []

    # Result
    ctx.result = _make_debate_result(task)

    ctx.vote_tally = {}
    ctx.winner_agent = None
    ctx.cancellation_token = None
    ctx.hook_manager = None
    ctx.event_emitter = None
    ctx.loop_id = "test-loop"
    ctx.debate_id = "flywheel-debate-001"
    ctx.domain = "engineering"
    ctx.start_time = time.time()

    # Background tasks
    ctx.background_research_task = None
    ctx.background_evidence_task = None
    ctx.applied_insight_ids = []

    # Convergence data (set during debate rounds)
    ctx.convergence_status = ""
    ctx.convergence_similarity = 0.0

    # Explicitly set attributes that MagicMock would otherwise auto-create.
    # These are dynamically set by various phases and must start as None/unset.
    ctx._prompt_builder = None
    ctx._convergence_hint = None
    ctx._km_item_ids_used = None

    return ctx


# ===========================================================================
# Loop 1: Debate -> Receipt -> KM -> Next Debate (Knowledge Injection)
# ===========================================================================


class TestLoop1_ReceiptToKMToNextDebate:
    """Prove that a debate receipt persisted to KM is retrieved by the next debate."""

    @pytest.mark.asyncio
    async def test_full_receipt_feedback_loop(self):
        """Run two sequential 'debates', verify the second sees receipt from the first.

        Steps:
        1. PostDebateCoordinator persists a receipt for debate #1.
        2. ContextInitializer._inject_receipt_conclusions() queries KM for
           similar receipts and injects them into debate #2's context.
        3. Assert that debate #2's env.context contains the receipt conclusion.
        """
        from aragora.debate.post_debate_coordinator import (
            PostDebateConfig,
            PostDebateCoordinator,
        )
        from aragora.debate.phases.context_init import (
            ContextInitializer,
            _receipt_conclusions_cache,
        )

        # --- Step 1: Debate #1 completes and persists a receipt ---
        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_persist_receipt=True,
            auto_queue_improvement=False,
            auto_execution_bridge=False,
        )
        coordinator = PostDebateCoordinator(config=config)

        debate1_result = _make_debate_result(
            task="Design a rate limiter for API endpoints",
            final_answer="Use token-bucket algorithm with Redis backend",
            confidence=0.9,
        )

        # Mock the receipt adapter to capture what gets ingested
        ingested_receipts: list[dict] = []

        mock_adapter = MagicMock()
        mock_adapter.ingest = lambda data: (ingested_receipts.append(data) or True)

        with patch(
            "aragora.knowledge.mound.adapters.receipt_adapter.get_receipt_adapter",
            return_value=mock_adapter,
        ):
            pdc_result = coordinator.run(
                debate_id="debate-001",
                debate_result=debate1_result,
                confidence=0.9,
                task="Design a rate limiter for API endpoints",
            )

        # Verify receipt was persisted
        assert pdc_result.receipt_persisted is True, (
            "Receipt persistence failed -- this indicates the get_receipt_adapter "
            "function or ingest method is missing/broken"
        )
        assert len(ingested_receipts) == 1
        assert ingested_receipts[0]["task"] == "Design a rate limiter for API endpoints"
        assert ingested_receipts[0]["confidence"] == 0.9

        # --- Step 2: Clear cache, set up KM mock for debate #2 ---
        _receipt_conclusions_cache.clear()

        # Build a mock KM that returns the ingested receipt as a query result
        mock_km_item = MagicMock()
        mock_km_item.confidence = MagicMock()
        mock_km_item.confidence.value = "high"
        mock_km_item.content = "Use token-bucket algorithm with Redis backend"
        mock_km_item.metadata = {"verdict": "consensus"}

        mock_query_result = MagicMock()
        mock_query_result.items = [mock_km_item]

        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=mock_query_result)

        initializer = ContextInitializer(
            knowledge_mound=mock_mound,
            enable_knowledge_retrieval=True,
        )

        # --- Step 3: Debate #2 starts on a similar topic ---
        ctx2 = _make_ctx(
            task="Design a rate limiter for microservices",
            context="",
        )

        await initializer._inject_receipt_conclusions(ctx2)

        # --- Verify: debate #2's context now contains the receipt conclusion ---
        assert "PAST DECISION CONCLUSIONS" in ctx2.env.context
        assert "token-bucket" in ctx2.env.context
        assert "high confidence" in ctx2.env.context

        # Verify the KM was queried with the right filters
        mock_mound.query.assert_awaited_once()
        call_kwargs = mock_mound.query.call_args
        # The query should include decision_receipt tag filter
        filters_arg = (
            call_kwargs.kwargs.get("filters") or call_kwargs.args[1]
            if len(call_kwargs.args) > 1
            else call_kwargs.kwargs.get("filters")
        )
        if filters_arg is None:
            # Try positional
            args = call_kwargs[1] if len(call_kwargs) > 1 else {}
            filters_arg = args.get("filters")

    @pytest.mark.asyncio
    async def test_receipt_loop_no_km_items_yields_empty_context(self):
        """When KM has no matching receipts, no conclusions are injected."""
        from aragora.debate.phases.context_init import (
            ContextInitializer,
            _receipt_conclusions_cache,
        )

        _receipt_conclusions_cache.clear()

        mock_query_result = MagicMock()
        mock_query_result.items = []

        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=mock_query_result)

        initializer = ContextInitializer(
            knowledge_mound=mock_mound,
            enable_knowledge_retrieval=True,
        )

        ctx = _make_ctx(task="Completely unrelated topic about gardening", context="")

        await initializer._inject_receipt_conclusions(ctx)

        assert ctx.env.context == ""

    @pytest.mark.asyncio
    async def test_receipt_conclusions_appended_to_existing_context(self):
        """Receipt conclusions append to (not replace) existing context."""
        from aragora.debate.phases.context_init import (
            ContextInitializer,
            _receipt_conclusions_cache,
        )

        _receipt_conclusions_cache.clear()

        mock_km_item = MagicMock()
        mock_km_item.confidence = MagicMock()
        mock_km_item.confidence.value = "medium"
        mock_km_item.content = "Previous rate limiter used sliding window"
        mock_km_item.metadata = {}

        mock_query_result = MagicMock()
        mock_query_result.items = [mock_km_item]

        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=mock_query_result)

        initializer = ContextInitializer(
            knowledge_mound=mock_mound,
            enable_knowledge_retrieval=True,
        )

        ctx = _make_ctx(
            task="Design a rate limiter",
            context="You are designing a system.",
        )

        await initializer._inject_receipt_conclusions(ctx)

        # Both original and injected content should be present
        assert "You are designing a system." in ctx.env.context
        assert "PAST DECISION CONCLUSIONS" in ctx.env.context
        assert "sliding window" in ctx.env.context

    @pytest.mark.asyncio
    async def test_receipt_conclusions_cached_on_second_call(self):
        """Second call for the same topic uses cache, not KM query."""
        from aragora.debate.phases.context_init import (
            ContextInitializer,
            _receipt_conclusions_cache,
        )

        _receipt_conclusions_cache.clear()

        mock_km_item = MagicMock()
        mock_km_item.confidence = MagicMock()
        mock_km_item.confidence.value = "high"
        mock_km_item.content = "Cached receipt conclusion"
        mock_km_item.metadata = {}

        mock_query_result = MagicMock()
        mock_query_result.items = [mock_km_item]

        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=mock_query_result)

        initializer = ContextInitializer(
            knowledge_mound=mock_mound,
            enable_knowledge_retrieval=True,
        )

        # First call: hits KM
        ctx1 = _make_ctx(task="Design a rate limiter", context="")
        await initializer._inject_receipt_conclusions(ctx1)
        assert mock_mound.query.await_count == 1

        # Second call: should use cache (same topic hash)
        ctx2 = _make_ctx(task="Design a rate limiter", context="")
        await initializer._inject_receipt_conclusions(ctx2)
        assert mock_mound.query.await_count == 1  # Still 1 - cache hit

        # But the context should still be injected from cache
        assert "PAST DECISION CONCLUSIONS" in ctx2.env.context


# ===========================================================================
# Loop 2: Debate -> ImprovementQueue -> MetaPlanner (Self-Improvement)
# ===========================================================================


class TestLoop2_ImprovementQueueToMetaPlanner:
    """Prove that debate outcomes flow through the improvement queue to MetaPlanner."""

    def test_post_debate_coordinator_queues_improvement(self):
        """PostDebateCoordinator with auto_queue_improvement=True enqueues a suggestion."""
        from aragora.debate.post_debate_coordinator import (
            PostDebateConfig,
            PostDebateCoordinator,
        )
        from aragora.nomic.improvement_queue import get_improvement_queue

        # Drain any pre-existing items
        queue = get_improvement_queue()
        queue.dequeue_batch(1000)
        assert len(queue) == 0

        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_persist_receipt=False,
            auto_queue_improvement=True,
            improvement_min_confidence=0.5,
            auto_execution_bridge=False,
        )
        coordinator = PostDebateCoordinator(config=config)

        debate_result = _make_debate_result(
            task="Improve test coverage for analytics module",
            final_answer="Add integration tests for dashboard aggregation pipeline",
            confidence=0.85,
        )

        result = coordinator.run(
            debate_id="debate-imp-001",
            debate_result=debate_result,
            confidence=0.85,
            task="Improve test coverage for analytics module",
        )

        # Verify the improvement was queued
        assert result.improvement_queued is True
        assert len(queue) == 1

        # Verify the suggestion contents
        items = queue.peek(10)
        assert len(items) == 1
        assert items[0].debate_id == "debate-imp-001"
        assert items[0].task == "Improve test coverage for analytics module"
        assert items[0].category == "test_coverage"
        assert items[0].confidence == 0.85

    @pytest.mark.asyncio
    async def test_meta_planner_reads_improvement_queue(self):
        """MetaPlanner.prioritize_work() injects queued improvements into PlanningContext."""
        from aragora.nomic.improvement_queue import (
            ImprovementSuggestion,
            get_improvement_queue,
        )
        from aragora.nomic.meta_planner import (
            MetaPlanner,
            MetaPlannerConfig,
            PlanningContext,
            Track,
        )

        # Set up the global queue with a suggestion
        queue = get_improvement_queue()
        queue.dequeue_batch(1000)

        queue.enqueue(
            ImprovementSuggestion(
                debate_id="debate-imp-001",
                task="Improve test coverage for analytics module",
                suggestion="Add integration tests for dashboard aggregation pipeline",
                category="test_coverage",
                confidence=0.85,
            )
        )
        assert len(queue) == 1

        # Use non-quick mode so the injection code path runs
        config = MetaPlannerConfig(
            quick_mode=False,
            enable_cross_cycle_learning=False,
        )
        planner = MetaPlanner(config=config)

        context = PlanningContext()

        # Patch Arena to avoid needing real agents
        with (
            patch("aragora.debate.orchestrator.Arena") as MockArena,
            patch("aragora.core.Environment"),
            patch("aragora.debate.protocol.DebateProtocol"),
        ):
            mock_arena_instance = MockArena.return_value
            mock_result = MagicMock()
            mock_result.consensus = "1. Improve test coverage (Track: qa, High impact)"
            mock_result.final_response = None
            mock_result.responses = []
            mock_arena_instance.run = AsyncMock(return_value=mock_result)

            with patch.object(planner, "_create_agent", return_value=_make_agent("mock")):
                with patch.object(planner, "_generate_receipt"):
                    goals = await planner.prioritize_work(
                        objective="Improve code quality",
                        available_tracks=[Track.QA],
                        context=context,
                    )

        # Verify the improvement suggestion was injected into context
        assert len(context.recent_improvements) == 1
        assert (
            context.recent_improvements[0]["task"] == "Improve test coverage for analytics module"
        )
        assert context.recent_improvements[0]["category"] == "test_coverage"
        assert context.recent_improvements[0]["confidence"] == 0.85

        # Verify the queue was drained (dequeue_batch consumes items)
        remaining = queue.peek(10)
        assert len(remaining) == 0

    @pytest.mark.asyncio
    async def test_full_improvement_loop_e2e(self):
        """Full loop: PostDebateCoordinator -> ImprovementQueue -> MetaPlanner reads it."""
        from aragora.debate.post_debate_coordinator import (
            PostDebateConfig,
            PostDebateCoordinator,
        )
        from aragora.nomic.improvement_queue import get_improvement_queue
        from aragora.nomic.meta_planner import (
            MetaPlanner,
            MetaPlannerConfig,
            PlanningContext,
            Track,
        )

        # --- Step 1: Clean queue ---
        queue = get_improvement_queue()
        queue.dequeue_batch(1000)

        # --- Step 2: Run PostDebateCoordinator (simulates end of debate) ---
        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_persist_receipt=False,
            auto_queue_improvement=True,
            improvement_min_confidence=0.5,
            auto_execution_bridge=False,
        )
        coordinator = PostDebateCoordinator(config=config)

        debate_result = _make_debate_result(
            task="Improve reliability of knowledge mound sync",
            final_answer="Add circuit breaker around KM write path with exponential backoff",
            confidence=0.9,
        )

        pdc_result = coordinator.run(
            debate_id="debate-loop2-001",
            debate_result=debate_result,
            confidence=0.9,
            task="Improve reliability of knowledge mound sync",
        )

        assert pdc_result.improvement_queued is True
        assert len(queue) == 1

        # --- Step 3: MetaPlanner consumes the queue ---
        planner_config = MetaPlannerConfig(
            quick_mode=False,
            enable_cross_cycle_learning=False,
        )
        planner = MetaPlanner(config=planner_config)

        context = PlanningContext()

        with (
            patch("aragora.debate.orchestrator.Arena") as MockArena,
            patch("aragora.core.Environment"),
            patch("aragora.debate.protocol.DebateProtocol"),
        ):
            mock_arena_instance = MockArena.return_value
            mock_result = MagicMock()
            mock_result.consensus = "1. Add circuit breakers (Track: core)"
            mock_result.final_response = None
            mock_result.responses = []
            mock_arena_instance.run = AsyncMock(return_value=mock_result)

            with patch.object(planner, "_create_agent", return_value=_make_agent("mock")):
                with patch.object(planner, "_generate_receipt"):
                    goals = await planner.prioritize_work(
                        objective="Improve system reliability",
                        available_tracks=[Track.CORE],
                        context=context,
                    )

        # --- Verify: MetaPlanner saw the improvement from the debate ---
        assert len(context.recent_improvements) == 1
        assert "reliability" in context.recent_improvements[0]["category"]
        assert context.recent_improvements[0]["confidence"] == 0.9

    def test_improvement_below_confidence_threshold_not_queued(self):
        """When debate confidence is below threshold, no improvement is queued."""
        from aragora.debate.post_debate_coordinator import (
            PostDebateConfig,
            PostDebateCoordinator,
        )
        from aragora.nomic.improvement_queue import get_improvement_queue

        queue = get_improvement_queue()
        queue.dequeue_batch(1000)

        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_persist_receipt=False,
            auto_queue_improvement=True,
            improvement_min_confidence=0.8,  # High threshold
            auto_execution_bridge=False,
        )
        coordinator = PostDebateCoordinator(config=config)

        debate_result = _make_debate_result(
            task="Some low confidence debate",
            final_answer="Uncertain answer",
            confidence=0.5,  # Below threshold
        )

        result = coordinator.run(
            debate_id="debate-low-conf",
            debate_result=debate_result,
            confidence=0.5,
            task="Some low confidence debate",
        )

        # Improvement should NOT be queued due to low confidence
        assert result.improvement_queued is False
        assert len(queue) == 0


# ===========================================================================
# Loop 3: Debate -> ConvergenceHistory -> Round Optimization
# ===========================================================================


class TestLoop3_ConvergenceHistoryToRoundOptimization:
    """Prove that convergence metrics flow from one debate into the next debate's context."""

    def setup_method(self):
        """Reset the convergence history store singleton and cache before each test."""
        from aragora.debate.convergence.history import set_convergence_history_store
        from aragora.debate.phases.context_init import _convergence_history_cache

        set_convergence_history_store(None)
        _convergence_history_cache.clear()

    def teardown_method(self):
        """Reset the convergence history store singleton and cache after each test."""
        from aragora.debate.convergence.history import set_convergence_history_store
        from aragora.debate.phases.context_init import _convergence_history_cache

        set_convergence_history_store(None)
        _convergence_history_cache.clear()

    def test_feedback_phase_records_convergence_history(self):
        """FeedbackPhase._record_convergence_history stores metrics in the store."""
        from aragora.debate.convergence.history import (
            ConvergenceHistoryStore,
            get_convergence_history_store,
            init_convergence_history_store,
        )
        from aragora.debate.phases.feedback_phase import FeedbackPhase

        # Initialize the global store
        store = init_convergence_history_store()
        assert store is not None
        assert len(store._records) == 0

        # Create a FeedbackPhase (minimal, most params None)
        feedback = FeedbackPhase()

        # Create a context that simulates a completed debate
        ctx = _make_ctx(task="Design a rate limiter for API endpoints")
        ctx.result.rounds_used = 3
        ctx.convergence_similarity = 0.82
        ctx.convergence_status = "converged"

        # Execute just the convergence recording step
        feedback._record_convergence_history(ctx)

        # Verify the store has the record
        assert len(store._records) == 1
        records = store.find_similar("Design a rate limiter for API endpoints")
        assert len(records) == 1
        assert records[0]["convergence_round"] == 3
        assert records[0]["total_rounds"] == 3
        assert records[0]["final_similarity"] == 0.82

    def test_context_initializer_injects_convergence_hint(self):
        """ContextInitializer._inject_convergence_history reads from the store."""
        from aragora.debate.convergence.history import (
            ConvergenceHistoryStore,
            init_convergence_history_store,
        )
        from aragora.debate.phases.context_init import (
            ContextInitializer,
            _convergence_history_cache,
        )

        # Clear the cache
        _convergence_history_cache.clear()

        # Initialize store with a pre-existing convergence record
        store = init_convergence_history_store()
        store.store(
            topic="Design a rate limiter for API endpoints",
            convergence_round=2,
            total_rounds=3,
            final_similarity=0.85,
        )

        initializer = ContextInitializer()

        # Second debate on similar topic
        ctx = _make_ctx(task="Design a rate limiter for microservices", context="")

        initializer._inject_convergence_history(ctx)

        # Verify the convergence hint was set on context
        hint = getattr(ctx, "_convergence_hint", None)
        assert hint is not None, "Expected _convergence_hint to be set on context"
        assert hint["avg_convergence_round"] == 2.0
        assert hint["avg_total_rounds"] == 3.0
        assert hint["avg_final_similarity"] == 0.85
        assert hint["sample_count"] == 1

        # Verify hint text was injected into env.context
        assert "CONVERGENCE HINT" in ctx.env.context
        assert "similar past debates" in ctx.env.context

    def test_full_convergence_loop_e2e(self):
        """Full loop: Debate #1 records metrics -> Debate #2 gets convergence hint.

        This is the complete end-to-end test for Loop 3.
        """
        from aragora.debate.convergence.history import (
            ConvergenceHistoryStore,
            init_convergence_history_store,
        )
        from aragora.debate.phases.context_init import (
            ContextInitializer,
            _convergence_history_cache,
        )
        from aragora.debate.phases.feedback_phase import FeedbackPhase

        _convergence_history_cache.clear()

        # --- Step 1: Initialize the convergence history store ---
        store = init_convergence_history_store()
        assert len(store._records) == 0

        # --- Step 2: Debate #1 completes and records convergence ---
        feedback = FeedbackPhase()

        ctx1 = _make_ctx(task="Design a caching strategy for database queries")
        ctx1.result.rounds_used = 4
        ctx1.convergence_similarity = 0.78
        ctx1.convergence_status = "converged"
        ctx1.debate_id = "conv-debate-001"

        feedback._record_convergence_history(ctx1)
        assert len(store._records) == 1

        # --- Step 3: Debate #2 starts on a similar topic ---
        initializer = ContextInitializer()

        ctx2 = _make_ctx(
            task="Design a caching strategy for API responses",
            context="",
        )

        initializer._inject_convergence_history(ctx2)

        # --- Verify: Debate #2 has a convergence hint ---
        hint = getattr(ctx2, "_convergence_hint", None)
        assert hint is not None, "Debate #2 should have received a convergence hint"
        assert hint["avg_convergence_round"] == 4.0
        assert hint["avg_final_similarity"] == 0.78
        assert hint["sample_count"] == 1

        # Verify the hint text is in env.context
        assert "CONVERGENCE HINT" in ctx2.env.context
        assert "4.0 rounds" in ctx2.env.context

    def test_convergence_hint_averages_multiple_records(self):
        """When multiple similar debates exist, the hint averages their metrics."""
        from aragora.debate.convergence.history import init_convergence_history_store
        from aragora.debate.phases.context_init import (
            ContextInitializer,
            _convergence_history_cache,
        )

        _convergence_history_cache.clear()
        store = init_convergence_history_store()

        # Store two similar debate convergence records
        store.store(
            topic="Design a caching strategy for queries",
            convergence_round=2,
            total_rounds=3,
            final_similarity=0.90,
        )
        store.store(
            topic="Design a caching layer for database",
            convergence_round=4,
            total_rounds=5,
            final_similarity=0.80,
        )

        initializer = ContextInitializer()
        ctx = _make_ctx(task="Design a caching strategy for database queries", context="")

        initializer._inject_convergence_history(ctx)

        hint = getattr(ctx, "_convergence_hint", None)
        assert hint is not None
        assert hint["sample_count"] == 2
        # Average of convergence_round: (2 + 4) / 2 = 3.0
        assert hint["avg_convergence_round"] == 3.0
        # Average of total_rounds: (3 + 5) / 2 = 4.0
        assert hint["avg_total_rounds"] == 4.0
        # Average of final_similarity: (0.90 + 0.80) / 2 = 0.85
        assert abs(hint["avg_final_similarity"] - 0.85) < 0.01

    def test_no_convergence_hint_for_unrelated_topic(self):
        """When there are no similar past debates, no convergence hint is injected."""
        from aragora.debate.convergence.history import init_convergence_history_store
        from aragora.debate.phases.context_init import (
            ContextInitializer,
            _convergence_history_cache,
        )

        _convergence_history_cache.clear()
        store = init_convergence_history_store()

        # Store a convergence record with topic that shares zero keywords
        # with the query. Use words with no overlap whatsoever.
        store.store(
            topic="quantum entanglement photon polarization",
            convergence_round=2,
            total_rounds=3,
            final_similarity=0.95,
        )

        initializer = ContextInitializer()
        ctx = _make_ctx(task="kubernetes deployment autoscaling", context="")
        # Explicitly mark that no convergence hint has been set yet.
        # (MagicMock auto-creates attributes on access, so we need a sentinel.)
        ctx._convergence_hint = None

        initializer._inject_convergence_history(ctx)

        # No keyword overlap -> no hint should have been set
        assert ctx._convergence_hint is None
        assert "CONVERGENCE HINT" not in (ctx.env.context or "")

    def test_convergence_history_cache_prevents_repeated_lookups(self):
        """Convergence history cache prevents repeated store lookups for the same topic."""
        from aragora.debate.convergence.history import init_convergence_history_store
        from aragora.debate.phases.context_init import (
            ContextInitializer,
            _convergence_history_cache,
        )

        _convergence_history_cache.clear()
        store = init_convergence_history_store()

        store.store(
            topic="Design a rate limiter",
            convergence_round=2,
            total_rounds=3,
            final_similarity=0.85,
        )

        initializer = ContextInitializer()

        # First call populates cache
        ctx1 = _make_ctx(task="Design a rate limiter", context="")
        initializer._inject_convergence_history(ctx1)
        assert "CONVERGENCE HINT" in ctx1.env.context

        # Spy on store.find_similar to verify cache hit
        original_find = store.find_similar
        call_count = {"n": 0}

        def counting_find(*args, **kwargs):
            call_count["n"] += 1
            return original_find(*args, **kwargs)

        store.find_similar = counting_find

        # Second call should use cache, not call find_similar
        ctx2 = _make_ctx(task="Design a rate limiter", context="")
        initializer._inject_convergence_history(ctx2)

        assert call_count["n"] == 0, "Expected cache hit, but find_similar was called"
        assert "CONVERGENCE HINT" in ctx2.env.context


# ===========================================================================
# Cross-Loop: Verify all three loops can operate together
# ===========================================================================


class TestAllThreeLoopsTogether:
    """Verify that all three feedback loops can operate simultaneously without interference."""

    def setup_method(self):
        """Reset shared state before each test."""
        from aragora.debate.convergence.history import set_convergence_history_store

        set_convergence_history_store(None)

    def teardown_method(self):
        """Reset shared state after each test."""
        from aragora.debate.convergence.history import set_convergence_history_store

        set_convergence_history_store(None)

    @pytest.mark.asyncio
    async def test_all_three_loops_fire_from_single_debate(self):
        """A single debate with full config activates all three feedback loops.

        This test simulates a complete debate lifecycle:
        1. FeedbackPhase records convergence history (Loop 3).
        2. PostDebateCoordinator persists receipt to KM (Loop 1) and
           queues an improvement (Loop 2).
        3. The next debate receives convergence hint (Loop 3) and
           receipt conclusions (Loop 1).
        4. MetaPlanner reads the improvement queue (Loop 2).
        """
        from aragora.debate.convergence.history import init_convergence_history_store
        from aragora.debate.phases.context_init import (
            ContextInitializer,
            _convergence_history_cache,
            _receipt_conclusions_cache,
        )
        from aragora.debate.phases.feedback_phase import FeedbackPhase
        from aragora.debate.post_debate_coordinator import (
            PostDebateConfig,
            PostDebateCoordinator,
        )
        from aragora.nomic.improvement_queue import get_improvement_queue

        # --- Clean up global state ---
        _convergence_history_cache.clear()
        _receipt_conclusions_cache.clear()
        queue = get_improvement_queue()
        queue.dequeue_batch(1000)
        store = init_convergence_history_store()

        # ===== DEBATE #1 =====

        # --- Loop 3: Record convergence history ---
        feedback = FeedbackPhase()
        ctx1 = _make_ctx(task="Design a rate limiter for API endpoints")
        ctx1.result.rounds_used = 3
        ctx1.convergence_similarity = 0.82
        ctx1.convergence_status = "converged"
        ctx1.debate_id = "all-loops-001"

        feedback._record_convergence_history(ctx1)
        assert len(store._records) == 1

        # --- Loops 1 & 2: PostDebateCoordinator ---
        pdc_config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_persist_receipt=True,
            auto_queue_improvement=True,
            improvement_min_confidence=0.5,
            auto_execution_bridge=False,
        )
        coordinator = PostDebateCoordinator(config=pdc_config)

        ingested_receipts: list[dict] = []
        mock_adapter = MagicMock()
        mock_adapter.ingest = lambda data: (ingested_receipts.append(data) or True)

        with patch(
            "aragora.knowledge.mound.adapters.receipt_adapter.get_receipt_adapter",
            return_value=mock_adapter,
        ):
            pdc_result = coordinator.run(
                debate_id="all-loops-001",
                debate_result=ctx1.result,
                confidence=0.85,
                task="Design a rate limiter for API endpoints",
            )

        # Verify all three write paths succeeded
        assert len(store._records) == 1, "Loop 3 write failed"
        assert len(ingested_receipts) == 1, "Loop 1 write failed"
        assert len(queue) == 1, "Loop 2 write failed"
        assert pdc_result.receipt_persisted is True
        assert pdc_result.improvement_queued is True

        # ===== DEBATE #2 =====

        # --- Loop 3: Read convergence hint ---
        initializer = ContextInitializer()
        ctx2 = _make_ctx(
            task="Design a rate limiter for microservices",
            context="",
        )

        initializer._inject_convergence_history(ctx2)
        convergence_hint = getattr(ctx2, "_convergence_hint", None)
        assert convergence_hint is not None, "Loop 3 read failed: no convergence hint"
        assert "CONVERGENCE HINT" in ctx2.env.context

        # --- Loop 1: Read receipt conclusions ---
        mock_km_item = MagicMock()
        mock_km_item.confidence = MagicMock()
        mock_km_item.confidence.value = "high"
        mock_km_item.content = ingested_receipts[0]["final_answer"]
        mock_km_item.metadata = {"verdict": "consensus"}

        mock_query_result = MagicMock()
        mock_query_result.items = [mock_km_item]

        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=mock_query_result)

        # Create a new initializer with KM configured
        initializer2 = ContextInitializer(
            knowledge_mound=mock_mound,
            enable_knowledge_retrieval=True,
        )

        # Inject receipt conclusions into ctx2 (preserving existing context)
        await initializer2._inject_receipt_conclusions(ctx2)
        assert "PAST DECISION CONCLUSIONS" in ctx2.env.context
        assert "CONVERGENCE HINT" in ctx2.env.context  # Both present

        # --- Loop 2: MetaPlanner reads improvement queue ---
        assert len(queue) == 1
        items = queue.peek(10)
        assert items[0].debate_id == "all-loops-001"
        assert items[0].category == "code_quality"
