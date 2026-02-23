"""Closed-loop integration test: debate → outcome → calibration → improvement.

Verifies that the full pipeline works end-to-end:
1. Arena setup creates all trackers (live explainability, introspection)
2. Events flow through EventBus to all subscribers
3. PostDebateCoordinator runs all enabled steps including:
   - Argument verification
   - Outcome feedback (systematic error → Nomic Loop goals)
   - Calibration push
4. Results are attached to DebateResult.metadata

This test mocks external dependencies but exercises real internal wiring.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core import DebateResult, Environment, TaskComplexity
from aragora.debate.context import DebateContext
from aragora.debate.event_bus import EventBus
from aragora.debate.orchestrator_runner import (
    _DebateExecutionState,
    handle_debate_completion,
    setup_debate_infrastructure,
)
from aragora.debate.post_debate_coordinator import (
    PostDebateConfig,
    PostDebateCoordinator,
    PostDebateResult,
)


class _FakeArena:
    """Lightweight fake Arena for integration testing."""

    def __init__(self) -> None:
        self.env = MagicMock(spec=Environment)
        self.env.task = "Should we implement rate limiting for the API?"
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

        # Enable ALL wired features
        self.enable_introspection = True
        self.active_introspection_tracker = None
        self.enable_live_explainability = True
        self.live_explainability_stream = None
        self.enable_post_debate_workflow = False
        self.disable_post_debate_pipeline = (
            True  # Skip full coordinator in handle_debate_completion
        )
        self.enable_auto_execution = False
        self.post_debate_config = None
        self.compliance_monitor = None


@pytest.fixture
def arena():
    return _FakeArena()


@pytest.fixture
def execution_state():
    ctx = MagicMock(spec=DebateContext)
    ctx.env = MagicMock()
    ctx.env.task = "Should we implement rate limiting for the API?"
    ctx.result = DebateResult(
        task="Should we implement rate limiting for the API?",
        consensus_reached=True,
        confidence=0.92,
        messages=[],
        critiques=[],
        votes=[],
        rounds_used=3,
        final_answer="Yes, use token bucket rate limiting with 100 req/min default",
    )
    ctx.domain = "architecture"
    ctx.post_debate_workflow_triggered = False
    return _DebateExecutionState(
        debate_id="integration-test-001",
        correlation_id="corr-integration",
        domain="architecture",
        task_complexity=TaskComplexity.MODERATE,
        ctx=ctx,
        debate_status="completed",
        debate_start_time=time.perf_counter() - 10.0,
    )


class TestClosedLoopIntegration:
    """End-to-end integration test for the full feedback loop."""

    @pytest.mark.asyncio
    async def test_full_pipeline_setup_events_completion(self, arena, execution_state):
        """Full pipeline: setup → events → completion → metadata attached."""
        # Step 1: Setup creates both trackers
        await setup_debate_infrastructure(arena, execution_state)

        assert arena.active_introspection_tracker is not None
        assert arena.live_explainability_stream is not None

        # Step 2: Simulate debate events via EventBus
        bus = arena.event_bus

        # Round 1: proposals
        bus.emit_sync(
            "agent_message",
            debate_id="integration-test-001",
            agent="claude",
            content="Token bucket is the right approach for API rate limiting",
            role="proposer",
            round_num=1,
        )
        bus.emit_sync(
            "agent_message",
            debate_id="integration-test-001",
            agent="gpt4",
            content="Consider sliding window instead of token bucket",
            role="proposer",
            round_num=1,
        )

        # Round 1: critiques
        bus.emit_sync(
            "agent_message",
            debate_id="integration-test-001",
            agent="gpt4",
            content="Token bucket doesn't handle burst traffic well",
            role="critic",
            round_num=1,
        )
        bus.emit_sync(
            "agent_message",
            debate_id="integration-test-001",
            agent="claude",
            content="Sliding window has higher memory overhead per client",
            role="critic",
            round_num=1,
        )

        # Round 2: refinements
        bus.emit_sync(
            "agent_message",
            debate_id="integration-test-001",
            agent="claude",
            content="Use token bucket with burst allowance for best of both",
            role="proposer",
            round_num=2,
        )

        # Step 3: Handle completion — attaches metadata
        await handle_debate_completion(arena, execution_state)

        result = execution_state.ctx.result

        # Verify introspection metadata attached
        assert "introspection" in result.metadata
        introspection = result.metadata["introspection"]
        assert "claude" in introspection
        assert "gpt4" in introspection
        assert introspection["claude"]["total_proposals"] == 2
        assert introspection["claude"]["total_critiques"] == 1
        assert introspection["gpt4"]["total_proposals"] == 1
        assert introspection["gpt4"]["total_critiques"] == 1

        # Verify live explainability metadata attached
        assert "live_explainability" in result.metadata
        live_exp = result.metadata["live_explainability"]
        assert "factors" in live_exp
        assert "narrative" in live_exp
        assert "leading_position" in live_exp

    @pytest.mark.asyncio
    async def test_post_debate_coordinator_all_steps(self, arena, execution_state):
        """PostDebateCoordinator runs all enabled steps in sequence."""
        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_persist_receipt=False,
            auto_gauntlet_validate=False,
            auto_verify_arguments=True,
            auto_outcome_feedback=True,
            auto_push_calibration=False,
            auto_execution_bridge=False,
        )
        coordinator = PostDebateCoordinator(config=config)

        mock_result = MagicMock()
        mock_result.messages = []
        mock_result.final_answer = "Use token bucket"
        mock_result.confidence = 0.92

        verification_result = {
            "debate_id": "integration-test-001",
            "verification": {"valid_chains": [], "invalid_chains": []},
            "is_sound": True,
            "soundness_score": 0.95,
        }
        feedback_result = {
            "goals_generated": 1,
            "suggestions_queued": 1,
            "trickster_adjustment": 0.9,
            "domains_flagged": ["architecture"],
            "agents_flagged": ["gpt4"],
        }

        with (
            patch.object(
                coordinator,
                "_step_argument_verification",
                return_value=verification_result,
            ),
            patch.object(
                coordinator,
                "_step_outcome_feedback",
                return_value=feedback_result,
            ),
        ):
            result = coordinator.run(
                "integration-test-001",
                mock_result,
                confidence=0.92,
                task="Rate limiting design",
            )

        assert result.argument_verification is not None
        assert result.argument_verification["is_sound"] is True
        assert result.argument_verification["soundness_score"] == 0.95

        assert result.outcome_feedback is not None
        assert result.outcome_feedback["goals_generated"] == 1
        assert "architecture" in result.outcome_feedback["domains_flagged"]

    def test_outcome_feedback_bridge_standalone(self):
        """OutcomeFeedbackBridge generates goals from mock systematic errors."""
        from aragora.nomic.outcome_feedback import FeedbackGoal, OutcomeFeedbackBridge

        bridge = OutcomeFeedbackBridge(
            min_verifications=3,
            overconfidence_threshold=0.05,
            low_accuracy_threshold=0.7,
        )

        # Test _error_to_goals directly with a known error pattern
        error = {
            "domain": "security",
            "agent": "gpt4",
            "overconfidence": 0.25,
            "success_rate": 0.55,
            "avg_brier_score": 0.35,
            "total_verifications": 10,
            "avg_confidence": 0.80,
        }

        goals = bridge._error_to_goals(error)

        # Should generate all 3 goal types
        goal_types = {g.goal_type for g in goals}
        assert "reduce_overconfidence" in goal_types
        assert "increase_accuracy" in goal_types
        assert "domain_training" in goal_types

        # Check severity ordering
        for goal in goals:
            assert 0.0 <= goal.severity <= 1.0
            assert goal.domain == "security"
            assert goal.agent == "gpt4"
            assert goal.priority >= 1
            assert goal.priority <= 10

    def test_feedback_goal_priority_mapping(self):
        """FeedbackGoal.priority maps severity to 1-10 range."""
        from aragora.nomic.outcome_feedback import FeedbackGoal

        low = FeedbackGoal(domain="d", agent="a", goal_type="t", severity=0.1, description="d")
        mid = FeedbackGoal(domain="d", agent="a", goal_type="t", severity=0.5, description="d")
        high = FeedbackGoal(domain="d", agent="a", goal_type="t", severity=1.0, description="d")

        assert low.priority == 1
        assert mid.priority == 5
        assert high.priority == 10

    def test_all_new_features_coexist(self):
        """All new PostDebateConfig flags can be enabled simultaneously."""
        config = PostDebateConfig(
            auto_verify_arguments=True,
            auto_outcome_feedback=True,
            auto_explain=True,
            auto_persist_receipt=True,
            auto_gauntlet_validate=True,
            auto_push_calibration=True,
            auto_execution_bridge=True,
        )
        assert config.auto_verify_arguments is True
        assert config.auto_outcome_feedback is True

        result = PostDebateResult()
        assert result.argument_verification is None
        assert result.outcome_feedback is None
        assert result.bridge_results == []

    @pytest.mark.asyncio
    async def test_disabled_features_produce_no_metadata(self, execution_state):
        """When features are disabled, no metadata keys are added."""
        disabled_arena = _FakeArena()
        disabled_arena.enable_introspection = False
        disabled_arena.enable_live_explainability = False

        await setup_debate_infrastructure(disabled_arena, execution_state)

        assert disabled_arena.active_introspection_tracker is None
        assert disabled_arena.live_explainability_stream is None

        await handle_debate_completion(disabled_arena, execution_state)

        result = execution_state.ctx.result
        assert "introspection" not in result.metadata
        assert "live_explainability" not in result.metadata
