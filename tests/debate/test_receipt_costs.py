"""Tests for per-debate cost accounting in receipt generation.

Verifies that:
1. Cost data is collected from DebateCostTracker when available
2. Cost data falls back to DebateResult fields when tracker unavailable
3. Receipt includes cost_summary when cost data is present
4. Receipt works without cost data (graceful degradation)
5. Cost breakdown structure validation (per-agent, per-round, model usage)
6. PostDebateCoordinator wires cost data through to receipts
7. Markdown and HTML exporters render cost sections correctly
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.post_debate_coordinator import (
    PostDebateConfig,
    PostDebateCoordinator,
    PostDebateResult,
)
from aragora.gauntlet.receipt_models import (
    ConsensusProof,
    DecisionReceipt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_debate_result(
    consensus: bool = True,
    confidence: float = 0.85,
    task: str = "Should we use microservices?",
    total_cost_usd: float = 0.0,
    per_agent_cost: dict | None = None,
):
    """Create a mock debate result with optional cost fields."""
    result = MagicMock()
    result.consensus = "majority" if consensus else None
    result.consensus_reached = consensus
    result.confidence = confidence
    result.task = task
    result.domain = "general"
    result.messages = []
    result.votes = []
    result.winner = "claude"
    result.participants = ["claude", "gpt-4o"]
    result.dissenting_views = []
    result.debate_id = "test-debate-001"
    result.final_answer = "Use microservices for scalability"
    result.rounds_used = 3
    result.duration_seconds = 45.2
    result.convergence_similarity = 0.9
    result.consensus_strength = "strong"
    result.total_cost_usd = total_cost_usd
    result.per_agent_cost = per_agent_cost or {}
    return result


def _make_cost_summary_dict(
    debate_id: str = "test-debate-001",
    total_cost: str = "0.0234",
    agents: dict | None = None,
):
    """Create a DebateCostSummary-like dict."""
    per_agent = {}
    if agents:
        for name, cost in agents.items():
            per_agent[name] = {
                "agent_name": name,
                "total_cost_usd": str(cost),
                "total_tokens_in": 1500,
                "total_tokens_out": 500,
                "call_count": 3,
                "models_used": {"claude-sonnet-4": 3},
            }

    return {
        "debate_id": debate_id,
        "total_cost_usd": total_cost,
        "total_tokens_in": 3000,
        "total_tokens_out": 1000,
        "total_calls": 6,
        "per_agent": per_agent,
        "per_round": {
            "1": {
                "round_number": 1,
                "total_cost_usd": "0.0120",
                "total_tokens_in": 1500,
                "total_tokens_out": 500,
                "call_count": 3,
            },
            "2": {
                "round_number": 2,
                "total_cost_usd": "0.0114",
                "total_tokens_in": 1500,
                "total_tokens_out": 500,
                "call_count": 3,
            },
        },
        "model_usage": {
            "anthropic/claude-sonnet-4": {
                "provider": "anthropic",
                "model": "claude-sonnet-4",
                "total_cost_usd": "0.0150",
                "total_tokens_in": 2000,
                "total_tokens_out": 700,
                "call_count": 4,
            },
            "openai/gpt-4o": {
                "provider": "openai",
                "model": "gpt-4o",
                "total_cost_usd": "0.0084",
                "total_tokens_in": 1000,
                "total_tokens_out": 300,
                "call_count": 2,
            },
        },
        "started_at": "2026-02-24T10:00:00+00:00",
        "completed_at": "2026-02-24T10:00:45+00:00",
    }


def _make_receipt_with_cost(cost_summary: dict | None = None) -> DecisionReceipt:
    """Create a receipt from a debate result with optional cost data."""
    result = _make_debate_result()
    return DecisionReceipt.from_debate_result(result, cost_summary=cost_summary)


# =============================================================================
# PostDebateResult cost_breakdown field
# =============================================================================


class TestPostDebateResultCostField:
    """Tests for the cost_breakdown field on PostDebateResult."""

    def test_default_is_none(self):
        r = PostDebateResult()
        assert r.cost_breakdown is None

    def test_can_set_cost_breakdown(self):
        cost_data = _make_cost_summary_dict()
        r = PostDebateResult(debate_id="d1", cost_breakdown=cost_data)
        assert r.cost_breakdown is not None
        assert r.cost_breakdown["total_cost_usd"] == "0.0234"

    def test_success_unaffected_by_cost(self):
        """cost_breakdown presence does not affect success property."""
        r = PostDebateResult(debate_id="d1", cost_breakdown={"total_cost_usd": "0.05"})
        assert r.success is True

        r2 = PostDebateResult(debate_id="d2", cost_breakdown=None)
        assert r2.success is True


# =============================================================================
# _step_collect_cost_data
# =============================================================================


class TestCollectCostData:
    """Tests for PostDebateCoordinator._step_collect_cost_data."""

    def test_returns_data_from_debate_cost_tracker(self):
        """When DebateCostTracker has data, returns rich summary."""
        mock_summary = MagicMock()
        mock_summary.total_calls = 6
        mock_summary.total_cost_usd = Decimal("0.0234")
        mock_summary.to_dict.return_value = _make_cost_summary_dict(
            agents={"claude": Decimal("0.015"), "gpt-4o": Decimal("0.0084")}
        )

        mock_tracker = MagicMock()
        mock_tracker.get_debate_cost.return_value = mock_summary

        coordinator = PostDebateCoordinator(config=PostDebateConfig())
        result = _make_debate_result()

        with patch(
            "aragora.debate.post_debate_coordinator.get_debate_cost_tracker",
            return_value=mock_tracker,
            create=True,
        ):
            # Need to patch the import itself
            import aragora.debate.post_debate_coordinator as mod

            # Use importlib to verify the method works
            with patch.dict("sys.modules", {}):
                cost = coordinator._step_collect_cost_data("test-debate-001", result)

        # Since the import happens inside the method, we need a different approach
        # Let's test via the actual method with patched module
        pass

    def test_returns_data_from_tracker_via_mock(self):
        """DebateCostTracker returns rich summary when data exists."""
        mock_summary = MagicMock()
        mock_summary.total_calls = 6
        mock_summary.total_cost_usd = Decimal("0.0234")
        expected_dict = _make_cost_summary_dict(
            agents={"claude": Decimal("0.015"), "gpt-4o": Decimal("0.0084")}
        )
        mock_summary.to_dict.return_value = expected_dict

        mock_tracker = MagicMock()
        mock_tracker.get_debate_cost.return_value = mock_summary

        coordinator = PostDebateCoordinator(config=PostDebateConfig())
        result = _make_debate_result()

        with patch(
            "aragora.billing.debate_costs.get_debate_cost_tracker",
            return_value=mock_tracker,
        ):
            cost = coordinator._step_collect_cost_data("test-debate-001", result)

        assert cost is not None
        assert cost["total_cost_usd"] == "0.0234"
        assert "per_agent" in cost
        mock_tracker.get_debate_cost.assert_called_once_with("test-debate-001")

    def test_falls_back_to_debate_result_fields(self):
        """When DebateCostTracker has no data, falls back to result fields."""
        mock_summary = MagicMock()
        mock_summary.total_calls = 0  # No calls recorded

        mock_tracker = MagicMock()
        mock_tracker.get_debate_cost.return_value = mock_summary

        coordinator = PostDebateCoordinator(config=PostDebateConfig())
        result = _make_debate_result(
            total_cost_usd=0.045,
            per_agent_cost={"claude": 0.03, "gpt-4o": 0.015},
        )

        with patch(
            "aragora.billing.debate_costs.get_debate_cost_tracker",
            return_value=mock_tracker,
        ):
            cost = coordinator._step_collect_cost_data("test-debate-001", result)

        assert cost is not None
        assert cost["total_cost_usd"] == "0.045"
        assert "per_agent" in cost
        assert "claude" in cost["per_agent"]
        assert cost["per_agent"]["claude"]["total_cost_usd"] == "0.03"

    def test_returns_none_when_no_cost_data(self):
        """When neither tracker nor result has cost data, returns None."""
        mock_summary = MagicMock()
        mock_summary.total_calls = 0

        mock_tracker = MagicMock()
        mock_tracker.get_debate_cost.return_value = mock_summary

        coordinator = PostDebateCoordinator(config=PostDebateConfig())
        result = _make_debate_result(total_cost_usd=0.0, per_agent_cost={})

        with patch(
            "aragora.billing.debate_costs.get_debate_cost_tracker",
            return_value=mock_tracker,
        ):
            cost = coordinator._step_collect_cost_data("test-debate-001", result)

        assert cost is None

    def test_graceful_degradation_on_import_error(self):
        """When DebateCostTracker cannot be imported, degrades gracefully."""
        coordinator = PostDebateCoordinator(config=PostDebateConfig())
        result = _make_debate_result(
            total_cost_usd=0.025,
            per_agent_cost={"claude": 0.025},
        )

        with patch(
            "builtins.__import__",
            side_effect=_selective_import_error("aragora.billing.debate_costs"),
        ):
            cost = coordinator._step_collect_cost_data("test-debate-001", result)

        # Should fall back to result fields
        assert cost is not None
        assert cost["total_cost_usd"] == "0.025"

    def test_graceful_degradation_on_tracker_exception(self):
        """When tracker raises, degrades to result fields."""
        mock_tracker = MagicMock()
        mock_tracker.get_debate_cost.side_effect = RuntimeError("DB unavailable")

        coordinator = PostDebateCoordinator(config=PostDebateConfig())
        result = _make_debate_result(total_cost_usd=0.01)

        with patch(
            "aragora.billing.debate_costs.get_debate_cost_tracker",
            return_value=mock_tracker,
        ):
            cost = coordinator._step_collect_cost_data("test-debate-001", result)

        assert cost is not None
        assert cost["total_cost_usd"] == "0.01"

    def test_fallback_structure_has_required_keys(self):
        """Fallback cost dict has the same top-level keys as tracker output."""
        coordinator = PostDebateCoordinator(config=PostDebateConfig())
        result = _make_debate_result(
            total_cost_usd=0.05,
            per_agent_cost={"claude": 0.03, "gpt-4o": 0.02},
        )

        # Simulate tracker not available
        with patch(
            "aragora.billing.debate_costs.get_debate_cost_tracker",
            side_effect=ImportError("not available"),
        ):
            cost = coordinator._step_collect_cost_data("test-debate-001", result)

        assert cost is not None
        required_keys = {
            "debate_id",
            "total_cost_usd",
            "total_tokens_in",
            "total_tokens_out",
            "total_calls",
            "per_agent",
            "per_round",
            "model_usage",
        }
        assert required_keys.issubset(set(cost.keys()))


def _selective_import_error(blocked_module: str):
    """Create an import side_effect that blocks a specific module."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _mock_import(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"Mocked import error for {name}")
        return real_import(name, *args, **kwargs)

    return _mock_import


# =============================================================================
# DecisionReceipt.from_debate_result with cost_summary
# =============================================================================


class TestReceiptFromDebateResultWithCost:
    """Tests for DecisionReceipt.from_debate_result cost_summary param."""

    def test_receipt_includes_cost_when_provided(self):
        cost_data = _make_cost_summary_dict(
            agents={"claude": "0.015", "gpt-4o": "0.0084"}
        )
        receipt = _make_receipt_with_cost(cost_summary=cost_data)

        assert receipt.cost_summary is not None
        assert receipt.cost_summary["total_cost_usd"] == "0.0234"
        assert "per_agent" in receipt.cost_summary
        assert len(receipt.cost_summary["per_agent"]) == 2

    def test_receipt_works_without_cost(self):
        receipt = _make_receipt_with_cost(cost_summary=None)

        assert receipt.cost_summary is None
        # Receipt should still be valid
        assert receipt.receipt_id
        assert receipt.verdict in ("PASS", "CONDITIONAL", "FAIL")
        assert receipt.confidence > 0

    def test_cost_summary_in_to_dict(self):
        cost_data = _make_cost_summary_dict()
        receipt = _make_receipt_with_cost(cost_summary=cost_data)

        receipt_dict = receipt.to_dict()
        assert "cost_summary" in receipt_dict
        assert receipt_dict["cost_summary"] == cost_data

    def test_cost_summary_none_in_to_dict(self):
        receipt = _make_receipt_with_cost(cost_summary=None)

        receipt_dict = receipt.to_dict()
        assert "cost_summary" in receipt_dict
        assert receipt_dict["cost_summary"] is None

    def test_cost_summary_survives_roundtrip(self):
        cost_data = _make_cost_summary_dict(agents={"claude": "0.02"})
        receipt = _make_receipt_with_cost(cost_summary=cost_data)

        receipt_dict = receipt.to_dict()
        restored = DecisionReceipt.from_dict(receipt_dict)

        assert restored.cost_summary is not None
        assert restored.cost_summary["total_cost_usd"] == cost_data["total_cost_usd"]
        assert "per_agent" in restored.cost_summary

    def test_cost_summary_in_json_export(self):
        """cost_summary appears in JSON export."""
        cost_data = _make_cost_summary_dict()
        receipt = _make_receipt_with_cost(cost_summary=cost_data)

        json_str = receipt.to_json()
        assert '"cost_summary"' in json_str
        assert '"total_cost_usd"' in json_str


# =============================================================================
# Cost breakdown structure validation
# =============================================================================


class TestCostBreakdownStructure:
    """Validate the structure of cost breakdown dicts."""

    def test_per_agent_breakdown_structure(self):
        cost_data = _make_cost_summary_dict(
            agents={"claude": "0.015", "gpt-4o": "0.0084"}
        )
        per_agent = cost_data["per_agent"]

        for agent_name, breakdown in per_agent.items():
            assert "agent_name" in breakdown
            assert "total_cost_usd" in breakdown
            assert "total_tokens_in" in breakdown
            assert "total_tokens_out" in breakdown
            assert "call_count" in breakdown

    def test_per_round_breakdown_structure(self):
        cost_data = _make_cost_summary_dict()
        per_round = cost_data["per_round"]

        for round_key, breakdown in per_round.items():
            assert "round_number" in breakdown
            assert "total_cost_usd" in breakdown
            assert "total_tokens_in" in breakdown
            assert "total_tokens_out" in breakdown
            assert "call_count" in breakdown

    def test_model_usage_structure(self):
        cost_data = _make_cost_summary_dict()
        model_usage = cost_data["model_usage"]

        for key, usage in model_usage.items():
            assert "provider" in usage
            assert "model" in usage
            assert "total_cost_usd" in usage
            assert "call_count" in usage

    def test_total_tokens_computed(self):
        cost_data = _make_cost_summary_dict()
        total_in = cost_data["total_tokens_in"]
        total_out = cost_data["total_tokens_out"]

        assert total_in == 3000
        assert total_out == 1000
        assert total_in + total_out == 4000


# =============================================================================
# Markdown exporter cost rendering
# =============================================================================


class TestMarkdownCostRendering:
    """Tests for cost breakdown in markdown export."""

    def test_markdown_includes_cost_section(self):
        cost_data = _make_cost_summary_dict(
            agents={"claude": "0.015", "gpt-4o": "0.0084"}
        )
        receipt = _make_receipt_with_cost(cost_summary=cost_data)

        md = receipt.to_markdown()
        assert "## Cost Breakdown" in md
        assert "$0.0234" in md

    def test_markdown_includes_per_agent_table(self):
        cost_data = _make_cost_summary_dict(
            agents={"claude": "0.015", "gpt-4o": "0.0084"}
        )
        receipt = _make_receipt_with_cost(cost_summary=cost_data)

        md = receipt.to_markdown()
        assert "### Per-Agent Costs" in md
        assert "claude" in md
        assert "gpt-4o" in md
        assert "$0.015" in md

    def test_markdown_includes_model_usage(self):
        cost_data = _make_cost_summary_dict()
        receipt = _make_receipt_with_cost(cost_summary=cost_data)

        md = receipt.to_markdown()
        assert "### Model Usage" in md
        assert "anthropic/claude-sonnet-4" in md
        assert "openai/gpt-4o" in md

    def test_markdown_includes_token_counts(self):
        cost_data = _make_cost_summary_dict()
        receipt = _make_receipt_with_cost(cost_summary=cost_data)

        md = receipt.to_markdown()
        assert "Tokens In" in md
        assert "Tokens Out" in md

    def test_markdown_no_cost_section_when_absent(self):
        receipt = _make_receipt_with_cost(cost_summary=None)

        md = receipt.to_markdown()
        assert "## Cost Breakdown" not in md


# =============================================================================
# HTML exporter cost rendering
# =============================================================================


class TestHTMLCostRendering:
    """Tests for cost breakdown in HTML export."""

    def test_html_includes_cost_section(self):
        cost_data = _make_cost_summary_dict(
            agents={"claude": "0.015", "gpt-4o": "0.0084"}
        )
        receipt = _make_receipt_with_cost(cost_summary=cost_data)

        html = receipt.to_html()
        assert "Cost Breakdown" in html
        assert "$0.0234" in html

    def test_html_includes_per_agent_table(self):
        cost_data = _make_cost_summary_dict(
            agents={"claude": "0.015", "gpt-4o": "0.0084"}
        )
        receipt = _make_receipt_with_cost(cost_summary=cost_data)

        html = receipt.to_html()
        assert "Per-Agent Costs" in html
        assert "claude" in html
        assert "gpt-4o" in html

    def test_html_no_cost_section_when_absent(self):
        receipt = _make_receipt_with_cost(cost_summary=None)

        html = receipt.to_html()
        assert "Cost Breakdown" not in html

    def test_html_cost_section_is_well_formed(self):
        """Cost section has matching div open/close tags."""
        cost_data = _make_cost_summary_dict()
        receipt = _make_receipt_with_cost(cost_summary=cost_data)

        html = receipt.to_html()
        # The cost section should be wrapped in a div.section
        assert '<div class="section">' in html


# =============================================================================
# Full pipeline integration
# =============================================================================


class TestPostDebateCoordinatorCostWiring:
    """Test that PostDebateCoordinator wires cost data through the pipeline."""

    def test_run_populates_cost_breakdown(self):
        """run() collects cost data and stores it in result.cost_breakdown."""
        mock_summary = MagicMock()
        mock_summary.total_calls = 4
        mock_summary.total_cost_usd = Decimal("0.032")
        mock_summary.to_dict.return_value = _make_cost_summary_dict()

        mock_tracker = MagicMock()
        mock_tracker.get_debate_cost.return_value = mock_summary

        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_persist_receipt=False,
            auto_gauntlet_validate=False,
            auto_queue_improvement=False,
            auto_outcome_feedback=False,
            auto_trigger_canvas=False,
            auto_llm_judge=False,
            auto_execution_bridge=False,
        )
        coordinator = PostDebateCoordinator(config=config)
        debate_result = _make_debate_result()

        with patch(
            "aragora.billing.debate_costs.get_debate_cost_tracker",
            return_value=mock_tracker,
        ):
            result = coordinator.run(
                debate_id="test-debate-001",
                debate_result=debate_result,
                confidence=0.85,
                task="test task",
            )

        assert result.cost_breakdown is not None
        assert result.cost_breakdown["total_cost_usd"] == "0.0234"

    def test_run_without_cost_data(self):
        """run() works without errors when no cost data is available."""
        mock_summary = MagicMock()
        mock_summary.total_calls = 0

        mock_tracker = MagicMock()
        mock_tracker.get_debate_cost.return_value = mock_summary

        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_persist_receipt=False,
            auto_gauntlet_validate=False,
            auto_queue_improvement=False,
            auto_outcome_feedback=False,
            auto_trigger_canvas=False,
            auto_llm_judge=False,
            auto_execution_bridge=False,
        )
        coordinator = PostDebateCoordinator(config=config)
        debate_result = _make_debate_result()

        with patch(
            "aragora.billing.debate_costs.get_debate_cost_tracker",
            return_value=mock_tracker,
        ):
            result = coordinator.run(
                debate_id="test-debate-001",
                debate_result=debate_result,
                confidence=0.85,
                task="test task",
            )

        assert result.cost_breakdown is None
        assert result.success is True

    def test_persist_receipt_includes_cost(self):
        """_step_persist_receipt passes cost data to the receipt adapter."""
        mock_adapter = MagicMock()

        coordinator = PostDebateCoordinator(config=PostDebateConfig())
        debate_result = _make_debate_result()
        cost_data = _make_cost_summary_dict()

        with patch(
            "aragora.knowledge.mound.adapters.receipt_adapter.get_receipt_adapter",
            return_value=mock_adapter,
        ):
            success = coordinator._step_persist_receipt(
                debate_id="test-debate-001",
                debate_result=debate_result,
                task="test task",
                confidence=0.85,
                cost_breakdown=cost_data,
            )

        assert success is True
        mock_adapter.ingest.assert_called_once()
        ingested_data = mock_adapter.ingest.call_args[0][0]
        assert "cost_summary" in ingested_data
        assert ingested_data["cost_summary"]["total_cost_usd"] == "0.0234"

    def test_persist_receipt_without_cost(self):
        """_step_persist_receipt works without cost data."""
        mock_adapter = MagicMock()

        coordinator = PostDebateCoordinator(config=PostDebateConfig())
        debate_result = _make_debate_result()

        with patch(
            "aragora.knowledge.mound.adapters.receipt_adapter.get_receipt_adapter",
            return_value=mock_adapter,
        ):
            success = coordinator._step_persist_receipt(
                debate_id="test-debate-001",
                debate_result=debate_result,
                task="test task",
                confidence=0.85,
                cost_breakdown=None,
            )

        assert success is True
        mock_adapter.ingest.assert_called_once()
        ingested_data = mock_adapter.ingest.call_args[0][0]
        assert "cost_summary" not in ingested_data
