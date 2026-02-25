"""
Tests for per-debate cost accounting end-to-end flow (issue #263).

Covers:
- AutonomicExecutor inline cost recording via _record_call_cost
- set_debate_cost_tracker / debate_id wiring
- Cost accumulation across generate, critique, and vote calls
- setup_debate_infrastructure wires cost tracker to executor
- cleanup_debate_resources clears cost tracker reference
- Receipt generation includes cost_summary from DebateCostTracker
- Debate costs endpoint returns correct data
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from aragora.billing.debate_costs import (
    DebateCostTracker,
    get_debate_cost_tracker,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tracker():
    """Fresh DebateCostTracker."""
    return DebateCostTracker()


@pytest.fixture
def mock_agent():
    """Mock agent with token usage tracking attributes."""
    agent = MagicMock()
    agent.name = "claude"
    agent.provider = "anthropic"
    agent.model = "claude-sonnet-4"
    # Simulate the last_tokens_in/out properties set by APIAgent._record_token_usage
    agent.last_tokens_in = 1000
    agent.last_tokens_out = 500
    # generate returns a coroutine
    agent.generate = AsyncMock(return_value="Test proposal response")
    agent.critique = AsyncMock(
        return_value=MagicMock(
            score=0.8,
            reasoning="Good proposal",
            suggestions=["Minor improvement"],
        )
    )
    agent.vote = AsyncMock(
        return_value=MagicMock(
            choice="claude",
            confidence=0.9,
            reasoning="Best answer",
        )
    )
    return agent


@pytest.fixture
def mock_gpt_agent():
    """Mock GPT agent."""
    agent = MagicMock()
    agent.name = "gpt"
    agent.provider = "openai"
    agent.model = "gpt-4o"
    agent.last_tokens_in = 800
    agent.last_tokens_out = 400
    agent.generate = AsyncMock(return_value="GPT proposal response")
    agent.critique = AsyncMock(
        return_value=MagicMock(
            score=0.7,
            reasoning="Needs work",
            suggestions=["Improve clarity"],
        )
    )
    agent.vote = AsyncMock(
        return_value=MagicMock(
            choice="gpt",
            confidence=0.85,
            reasoning="Comprehensive",
        )
    )
    return agent


# =============================================================================
# AutonomicExecutor._record_call_cost tests
# =============================================================================


class TestAutonomicExecutorCostRecording:
    """Tests for inline cost recording in AutonomicExecutor."""

    def _make_executor(self):
        """Create a minimal AutonomicExecutor."""
        from aragora.debate.autonomic_executor import AutonomicExecutor

        return AutonomicExecutor(circuit_breaker=None)

    def test_set_debate_cost_tracker(self, tracker):
        """set_debate_cost_tracker stores tracker and debate_id."""
        executor = self._make_executor()
        executor.set_debate_cost_tracker(tracker, "debate-123")

        assert executor._debate_cost_tracker is tracker
        assert executor._debate_id == "debate-123"

    def test_record_call_cost_records_to_tracker(self, tracker, mock_agent):
        """_record_call_cost writes to the DebateCostTracker."""
        executor = self._make_executor()
        executor.set_debate_cost_tracker(tracker, "debate-abc")

        executor._record_call_cost(mock_agent, "proposal", round_num=1)

        summary = tracker.get_debate_cost("debate-abc")
        assert summary.total_calls == 1
        assert summary.total_cost_usd > Decimal("0")
        assert "claude" in summary.per_agent
        assert 1 in summary.per_round

        agent_bd = summary.per_agent["claude"]
        assert agent_bd.total_tokens_in == 1000
        assert agent_bd.total_tokens_out == 500
        assert agent_bd.call_count == 1

        round_bd = summary.per_round[1]
        assert round_bd.call_count == 1
        assert round_bd.total_tokens_in == 1000

    def test_record_call_cost_skips_zero_tokens(self, tracker):
        """_record_call_cost does nothing when tokens are 0."""
        executor = self._make_executor()
        executor.set_debate_cost_tracker(tracker, "debate-xyz")

        agent = MagicMock()
        agent.name = "empty"
        agent.last_tokens_in = 0
        agent.last_tokens_out = 0

        executor._record_call_cost(agent, "proposal", round_num=1)

        summary = tracker.get_debate_cost("debate-xyz")
        assert summary.total_calls == 0

    def test_record_call_cost_noop_without_tracker(self, mock_agent):
        """_record_call_cost does nothing when tracker is not set."""
        executor = self._make_executor()
        # No tracker set
        executor._record_call_cost(mock_agent, "proposal", round_num=1)
        # Should not raise

    def test_record_call_cost_noop_without_debate_id(self, tracker, mock_agent):
        """_record_call_cost does nothing when debate_id is empty."""
        executor = self._make_executor()
        executor._debate_cost_tracker = tracker
        executor._debate_id = ""

        executor._record_call_cost(mock_agent, "proposal", round_num=1)

        # Tracker should have no records
        assert tracker.get_all_debate_ids() == []

    def test_record_call_cost_uses_operation_as_phase(self, tracker, mock_agent):
        """Operation field in the record matches the phase parameter."""
        executor = self._make_executor()
        executor.set_debate_cost_tracker(tracker, "d1")

        executor._record_call_cost(mock_agent, "critique", round_num=2)

        records = tracker.get_call_records("d1")
        assert len(records) == 1
        assert records[0].operation == "critique"
        assert records[0].round_number == 2

    def test_record_call_cost_handles_missing_agent_attrs(self, tracker):
        """_record_call_cost handles agents without provider/model attrs."""
        executor = self._make_executor()
        executor.set_debate_cost_tracker(tracker, "d1")

        agent = MagicMock(spec=[])  # Empty spec = no attributes
        agent.name = "bare"
        agent.last_tokens_in = 100
        agent.last_tokens_out = 50

        # Should not raise even without provider/model
        executor._record_call_cost(agent, "generate", round_num=1)

        summary = tracker.get_debate_cost("d1")
        assert summary.total_calls == 1

    def test_clear_tracker_after_debate(self, tracker, mock_agent):
        """set_debate_cost_tracker(None, '') clears the reference."""
        executor = self._make_executor()
        executor.set_debate_cost_tracker(tracker, "d1")
        executor._record_call_cost(mock_agent, "proposal", round_num=1)

        assert tracker.get_debate_cost("d1").total_calls == 1

        # Clear
        executor.set_debate_cost_tracker(None, "")

        assert executor._debate_cost_tracker is None
        assert executor._debate_id == ""

        # Further calls should be no-ops
        executor._record_call_cost(mock_agent, "revision", round_num=2)

        # Still only the original call
        assert tracker.get_debate_cost("d1").total_calls == 1


# =============================================================================
# AutonomicExecutor.generate() inline cost recording
# =============================================================================


class TestGenerateInlineCostRecording:
    """Tests that generate() automatically records costs."""

    @pytest.mark.asyncio
    async def test_generate_records_cost(self, tracker, mock_agent):
        """Successful generate() should record cost to tracker."""
        from aragora.debate.autonomic_executor import AutonomicExecutor

        executor = AutonomicExecutor(circuit_breaker=None)
        executor.set_debate_cost_tracker(tracker, "debate-gen-1")

        result = await executor.generate(
            mock_agent,
            "Test prompt",
            [],
            phase="proposal",
            round_num=1,
        )

        assert result is not None
        assert "Test proposal response" in result

        summary = tracker.get_debate_cost("debate-gen-1")
        assert summary.total_calls == 1
        assert "claude" in summary.per_agent
        assert 1 in summary.per_round

    @pytest.mark.asyncio
    async def test_generate_failure_does_not_record_cost(self, tracker):
        """Failed generate() should not record cost."""
        from aragora.debate.autonomic_executor import AutonomicExecutor

        executor = AutonomicExecutor(circuit_breaker=None)
        executor.set_debate_cost_tracker(tracker, "debate-fail")

        agent = MagicMock()
        agent.name = "failing"
        agent.provider = "test"
        agent.model = "test"
        agent.last_tokens_in = 0
        agent.last_tokens_out = 0
        agent.generate = AsyncMock(side_effect=RuntimeError("API error"))

        result = await executor.generate(agent, "Test prompt", [])
        # Should return error message, not raise
        assert "error" in result.lower() or "system" in result.lower()

        summary = tracker.get_debate_cost("debate-fail")
        assert summary.total_calls == 0

    @pytest.mark.asyncio
    async def test_multiple_agents_accumulate_costs(self, tracker, mock_agent, mock_gpt_agent):
        """Multiple agents accumulate separate per-agent costs."""
        from aragora.debate.autonomic_executor import AutonomicExecutor

        executor = AutonomicExecutor(circuit_breaker=None)
        executor.set_debate_cost_tracker(tracker, "debate-multi")

        await executor.generate(mock_agent, "Prompt A", [], phase="proposal", round_num=1)
        await executor.generate(mock_gpt_agent, "Prompt B", [], phase="proposal", round_num=1)
        await executor.generate(mock_agent, "Prompt C", [], phase="revision", round_num=2)

        summary = tracker.get_debate_cost("debate-multi")
        assert summary.total_calls == 3
        assert summary.total_cost_usd > Decimal("0")

        # Per-agent
        assert "claude" in summary.per_agent
        assert "gpt" in summary.per_agent
        assert summary.per_agent["claude"].call_count == 2
        assert summary.per_agent["gpt"].call_count == 1

        # Per-round
        assert 1 in summary.per_round
        assert 2 in summary.per_round
        assert summary.per_round[1].call_count == 2
        assert summary.per_round[2].call_count == 1


# =============================================================================
# Critique and vote cost recording
# =============================================================================


class TestCritiqueAndVoteCostRecording:
    """Tests that critique() and vote() also record costs."""

    @pytest.mark.asyncio
    async def test_critique_records_cost(self, tracker, mock_agent):
        """Successful critique() records cost to tracker."""
        from aragora.debate.autonomic_executor import AutonomicExecutor

        executor = AutonomicExecutor(circuit_breaker=None)
        executor.set_debate_cost_tracker(tracker, "debate-crit")

        result = await executor.critique(
            mock_agent,
            "Proposal text",
            "Task",
            [],
            phase="critique",
            round_num=1,
        )

        assert result is not None

        summary = tracker.get_debate_cost("debate-crit")
        assert summary.total_calls == 1
        records = tracker.get_call_records("debate-crit")
        assert records[0].operation == "critique"

    @pytest.mark.asyncio
    async def test_vote_records_cost(self, tracker, mock_agent):
        """Successful vote() records cost to tracker."""
        from aragora.debate.autonomic_executor import AutonomicExecutor

        executor = AutonomicExecutor(circuit_breaker=None)
        executor.set_debate_cost_tracker(tracker, "debate-vote")

        result = await executor.vote(
            mock_agent,
            {"claude": "Answer A", "gpt": "Answer B"},
            "Task",
            phase="voting",
            round_num=3,
        )

        assert result is not None

        summary = tracker.get_debate_cost("debate-vote")
        assert summary.total_calls == 1
        records = tracker.get_call_records("debate-vote")
        assert records[0].operation == "voting"
        assert records[0].round_number == 3


# =============================================================================
# Receipt cost_summary integration
# =============================================================================


class TestReceiptCostSummaryIntegration:
    """Tests that debate cost data flows into DecisionReceipt."""

    def test_cost_summary_includes_all_breakdowns(self, tracker):
        """Full cost summary has per-agent, per-round, and model_usage."""
        debate_id = "receipt-test"

        tracker.record_agent_call(
            debate_id=debate_id,
            agent_name="claude",
            provider="anthropic",
            tokens_in=2000,
            tokens_out=800,
            model="claude-sonnet-4",
            round_number=1,
            operation="proposal",
        )
        tracker.record_agent_call(
            debate_id=debate_id,
            agent_name="gpt",
            provider="openai",
            tokens_in=1500,
            tokens_out=600,
            model="gpt-4o",
            round_number=1,
            operation="critique",
        )
        tracker.record_agent_call(
            debate_id=debate_id,
            agent_name="claude",
            provider="anthropic",
            tokens_in=2500,
            tokens_out=1000,
            model="claude-sonnet-4",
            round_number=2,
            operation="revision",
        )

        summary = tracker.get_debate_cost(debate_id)
        cost_dict = summary.to_dict()

        # Build receipt
        from aragora.gauntlet.receipt_models import DecisionReceipt

        mock_result = MagicMock()
        mock_result.debate_id = debate_id
        mock_result.id = debate_id
        mock_result.messages = []
        mock_result.votes = []
        mock_result.consensus_reached = True
        mock_result.confidence = 0.85
        mock_result.participants = ["claude", "gpt"]
        mock_result.dissenting_views = []
        mock_result.consensus_strength = "strong"
        mock_result.final_answer = "Rate limiter with token bucket"
        mock_result.task = "Design a rate limiter"
        mock_result.winner = "claude"
        mock_result.rounds_used = 2
        mock_result.duration_seconds = 30.0
        mock_result.convergence_similarity = 0.9

        receipt = DecisionReceipt.from_debate_result(
            mock_result,
            cost_summary=cost_dict,
        )

        # Verify receipt has cost data
        assert receipt.cost_summary is not None
        assert receipt.cost_summary["debate_id"] == debate_id
        assert Decimal(receipt.cost_summary["total_cost_usd"]) > 0
        assert receipt.cost_summary["total_calls"] == 3

        # Verify per-agent has operations recorded
        per_agent = receipt.cost_summary["per_agent"]
        assert per_agent["claude"]["call_count"] == 2
        assert per_agent["gpt"]["call_count"] == 1

        # Verify per-round
        per_round = receipt.cost_summary["per_round"]
        assert per_round["1"]["call_count"] == 2  # proposal + critique
        assert per_round["2"]["call_count"] == 1  # revision

        # Verify model_usage
        model_usage = receipt.cost_summary["model_usage"]
        assert "anthropic/claude-sonnet-4" in model_usage
        assert "openai/gpt-4o" in model_usage

        # Round-trip preservation
        receipt_dict = receipt.to_dict()
        restored = DecisionReceipt.from_dict(receipt_dict)
        assert restored.cost_summary == receipt.cost_summary

    def test_receipt_to_dict_preserves_cost_in_json(self, tracker):
        """cost_summary in to_dict is JSON-serializable."""
        import json

        debate_id = "json-test"
        tracker.record_agent_call(
            debate_id=debate_id,
            agent_name="claude",
            provider="anthropic",
            tokens_in=1000,
            tokens_out=500,
            model="claude-sonnet-4",
            round_number=1,
        )

        summary = tracker.get_debate_cost(debate_id)
        cost_dict = summary.to_dict()

        from aragora.gauntlet.receipt_models import DecisionReceipt

        receipt = DecisionReceipt(
            receipt_id="r-json",
            gauntlet_id=debate_id,
            timestamp="2026-02-23T00:00:00Z",
            input_summary="test",
            input_hash="abc",
            risk_summary={"total": 0},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=0,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.8,
            cost_summary=cost_dict,
        )

        # Should be JSON-serializable
        json_str = json.dumps(receipt.to_dict())
        parsed = json.loads(json_str)
        assert parsed["cost_summary"]["debate_id"] == debate_id
        assert float(parsed["cost_summary"]["total_cost_usd"]) > 0


# =============================================================================
# Handler endpoint integration
# =============================================================================


class TestDebateCostsEndpoint:
    """Tests for the /api/v1/debates/{id}/costs endpoint."""

    def test_costs_mixin_returns_summary(self):
        """CostsMixin._get_debate_costs returns cost data."""
        from aragora.server.handlers.debates.costs import CostsMixin

        tracker = DebateCostTracker()
        debate_id = "endpoint-test"

        tracker.record_agent_call(
            debate_id=debate_id,
            agent_name="claude",
            provider="anthropic",
            tokens_in=2000,
            tokens_out=800,
            model="claude-sonnet-4",
            round_number=1,
            operation="proposal",
        )

        # Create a mock handler that implements the protocol
        handler = MagicMock()
        handler.ctx = {}

        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"id": debate_id, "status": "completed"}
        handler.get_storage.return_value = mock_storage

        # Bind the method to our mock
        with patch("aragora.billing.debate_costs.get_debate_cost_tracker", return_value=tracker):
            result = CostsMixin._get_debate_costs(handler, debate_id)

        assert result is not None
        # The result should be a tuple of (data, status_code) or a HandlerResult
        # depending on implementation, but should contain the cost data
        body = result[0] if isinstance(result, tuple) else result
        if isinstance(body, dict):
            data = body.get("data", body)
        else:
            data = body

        # Verify cost data is present
        assert data is not None

    def test_costs_mixin_returns_empty_for_unknown_debate(self):
        """CostsMixin returns empty summary for unknown debate."""
        from aragora.server.handlers.debates.costs import CostsMixin

        tracker = DebateCostTracker()

        handler = MagicMock()
        handler.ctx = {}

        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"id": "unknown", "status": "completed"}
        handler.get_storage.return_value = mock_storage

        with patch("aragora.billing.debate_costs.get_debate_cost_tracker", return_value=tracker):
            result = CostsMixin._get_debate_costs(handler, "unknown")

        assert result is not None

    def test_costs_mixin_returns_404_for_missing_debate(self):
        """CostsMixin returns 404 when debate doesn't exist in storage."""
        from aragora.server.handlers.debates.costs import CostsMixin

        handler = MagicMock()
        handler.ctx = {}

        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = None
        handler.get_storage.return_value = mock_storage

        result = CostsMixin._get_debate_costs(handler, "nonexistent")

        # Should be a 404 error response
        assert result is not None
        if isinstance(result, tuple):
            status = result[1]
            assert status == 404


# =============================================================================
# Infrastructure wiring tests
# =============================================================================


class TestDebateCostInfrastructureWiring:
    """Tests that setup/cleanup properly wire cost tracking."""

    @pytest.mark.asyncio
    async def test_setup_debate_infrastructure_wires_tracker(self):
        """setup_debate_infrastructure should wire DebateCostTracker to executor."""
        from aragora.debate.orchestrator_runner import setup_debate_infrastructure

        # Create a minimal mock arena
        arena = MagicMock()
        arena.autonomic = MagicMock()
        arena.autonomic.set_debate_cost_tracker = MagicMock()
        arena.protocol = MagicMock()
        arena.protocol.enable_hook_tracking = False
        arena.protocol.enable_compliance_check = False
        arena.agents = []
        arena.env = MagicMock()
        arena.env.task = "Test task"
        arena.extensions = MagicMock()
        arena.extensions.setup_debate_budget = MagicMock()
        arena._trackers = MagicMock()
        arena._budget_coordinator = MagicMock()
        arena._budget_coordinator.check_budget_before_debate = MagicMock()
        arena._emit_agent_preview = MagicMock()

        state = MagicMock()
        state.debate_id = "wiring-test"
        state.ctx = MagicMock()
        state.ctx.result = None
        state.domain = "general"
        state.task_complexity = MagicMock()
        state.task_complexity.value = "medium"
        state.correlation_id = "corr-123"

        mock_tracker = MagicMock()
        with patch(
            "aragora.billing.debate_costs.get_debate_cost_tracker",
            return_value=mock_tracker,
        ):
            await setup_debate_infrastructure(arena, state)

        # Verify the tracker was wired
        arena.autonomic.set_debate_cost_tracker.assert_called_once_with(mock_tracker, "wiring-test")

    @pytest.mark.asyncio
    async def test_cleanup_clears_tracker_reference(self):
        """cleanup_debate_resources clears the cost tracker from executor."""
        from aragora.debate.orchestrator_runner import cleanup_debate_resources

        arena = MagicMock()
        arena.autonomic = MagicMock()
        arena.autonomic.set_debate_cost_tracker = MagicMock()
        arena.protocol = MagicMock()
        arena.protocol.checkpoint_cleanup_on_success = False
        arena.protocol.enable_translation = False
        arena.protocol.enable_result_routing = False
        arena._cleanup_convergence_cache = MagicMock()
        arena._teardown_agent_channels = AsyncMock()
        arena.enable_auto_execution = False
        arena.enable_result_routing = False

        ctx = MagicMock()
        ctx.result = MagicMock()
        ctx.result.to_dict = MagicMock(return_value={})

        state = MagicMock()
        state.ctx = ctx
        state.debate_status = "completed"
        state.debate_id = "cleanup-test"

        ctx.finalize_result = MagicMock(return_value=ctx.result)

        await cleanup_debate_resources(arena, state)

        # Verify the tracker was cleared
        arena.autonomic.set_debate_cost_tracker.assert_called_once_with(None, "")


# =============================================================================
# Cost propagation to DebateResult
# =============================================================================


class TestCostPropagationToResult:
    """Tests that _populate_result_cost fills DebateResult cost fields."""

    @pytest.mark.asyncio
    async def test_populate_result_from_debate_cost_tracker(self):
        """_populate_result_cost fills result from DebateCostSummary."""
        from aragora.debate.orchestrator_runner import _populate_result_cost

        tracker = DebateCostTracker()
        debate_id = "pop-test"

        tracker.record_agent_call(
            debate_id=debate_id,
            agent_name="claude",
            provider="anthropic",
            tokens_in=2000,
            tokens_out=800,
            model="claude-sonnet-4",
            round_number=1,
            operation="proposal",
        )
        tracker.record_agent_call(
            debate_id=debate_id,
            agent_name="gpt",
            provider="openai",
            tokens_in=1500,
            tokens_out=600,
            model="gpt-4o",
            round_number=1,
            operation="critique",
        )

        summary = tracker.get_debate_cost(debate_id)

        # Mock result
        result = MagicMock()
        result.total_cost_usd = 0.0
        result.total_tokens = 0
        result.per_agent_cost = {}
        result.budget_limit_usd = None

        # Mock extensions with get_debate_cost_summary
        extensions = MagicMock()
        extensions.get_debate_cost_summary.return_value = summary
        extensions.debate_budget_limit_usd = 5.0

        await _populate_result_cost(result, debate_id, extensions)

        assert result.total_cost_usd > 0
        assert result.total_tokens == (2000 + 800 + 1500 + 600)
        assert "claude" in result.per_agent_cost
        assert "gpt" in result.per_agent_cost
        assert result.budget_limit_usd == 5.0
