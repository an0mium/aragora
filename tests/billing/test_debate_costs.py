"""
Tests for aragora.billing.debate_costs module.

Tests cover:
- DebateCostTracker: recording agent calls, cost computation, provider rates
- DebateCostSummary: per-agent, per-round, and per-model breakdowns
- DebateCostSummary: serialization (to_dict / from_dict round-trip)
- AgentCallRecord: creation and serialization
- Global singleton: get_debate_cost_tracker
- Edge cases: empty debates, unknown providers, clearing
- Integration with DecisionReceipt: cost_summary inclusion
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from aragora.billing.debate_costs import (
    AgentCallRecord,
    AgentCostBreakdown,
    DebateCostSummary,
    DebateCostTracker,
    ModelUsage,
    RoundCostBreakdown,
    get_debate_cost_tracker,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tracker():
    """Fresh DebateCostTracker with no external dependencies."""
    return DebateCostTracker()


@pytest.fixture
def populated_tracker(tracker):
    """Tracker with sample calls for debate-1."""
    tracker.record_agent_call(
        debate_id="debate-1",
        agent_name="claude",
        provider="anthropic",
        tokens_in=2000,
        tokens_out=800,
        model="claude-sonnet-4",
        round_number=1,
        operation="proposal",
    )
    tracker.record_agent_call(
        debate_id="debate-1",
        agent_name="gpt",
        provider="openai",
        tokens_in=1500,
        tokens_out=600,
        model="gpt-4o",
        round_number=1,
        operation="critique",
    )
    tracker.record_agent_call(
        debate_id="debate-1",
        agent_name="claude",
        provider="anthropic",
        tokens_in=2500,
        tokens_out=1000,
        model="claude-sonnet-4",
        round_number=2,
        operation="revision",
    )
    return tracker


# =============================================================================
# AgentCallRecord Tests
# =============================================================================


class TestAgentCallRecord:
    """Tests for AgentCallRecord dataclass."""

    def test_creation_and_cost(self):
        """Record stores cost correctly."""
        record = AgentCallRecord(
            debate_id="d1",
            agent_name="claude",
            provider="anthropic",
            model="claude-sonnet-4",
            tokens_in=1000,
            tokens_out=500,
            cost_usd=Decimal("0.01050"),
            round_number=1,
        )
        assert record.debate_id == "d1"
        assert record.agent_name == "claude"
        assert record.cost_usd == Decimal("0.01050")
        assert record.round_number == 1

    def test_to_dict(self):
        """to_dict produces expected keys."""
        record = AgentCallRecord(
            debate_id="d1",
            agent_name="gpt",
            provider="openai",
            model="gpt-4o",
            tokens_in=500,
            tokens_out=200,
            cost_usd=Decimal("0.003250"),
            round_number=2,
            operation="critique",
        )
        d = record.to_dict()
        assert d["debate_id"] == "d1"
        assert d["agent_name"] == "gpt"
        assert d["cost_usd"] == "0.003250"
        assert d["round_number"] == 2
        assert d["operation"] == "critique"
        assert "timestamp" in d


# =============================================================================
# DebateCostTracker Tests
# =============================================================================


class TestDebateCostTracker:
    """Tests for the DebateCostTracker class."""

    def test_record_agent_call_returns_record(self, tracker):
        """record_agent_call returns an AgentCallRecord with computed cost."""
        record = tracker.record_agent_call(
            debate_id="d1",
            agent_name="claude",
            provider="anthropic",
            tokens_in=1000,
            tokens_out=500,
            model="claude-sonnet-4",
        )
        assert isinstance(record, AgentCallRecord)
        assert record.debate_id == "d1"
        assert record.cost_usd > Decimal("0")

    def test_record_agent_call_computes_cost_from_usage(self, tracker):
        """Cost should be computed via calculate_token_cost."""
        record = tracker.record_agent_call(
            debate_id="d1",
            agent_name="gpt",
            provider="openai",
            tokens_in=1_000_000,  # 1M tokens
            tokens_out=0,
            model="gpt-4o",
        )
        # gpt-4o input rate is $2.50/1M tokens
        assert record.cost_usd == Decimal("2.50")

    def test_record_multiple_calls(self, tracker):
        """Multiple calls are tracked separately."""
        tracker.record_agent_call(
            debate_id="d1",
            agent_name="a",
            provider="openai",
            tokens_in=100,
            tokens_out=50,
            model="gpt-4o",
        )
        tracker.record_agent_call(
            debate_id="d1",
            agent_name="b",
            provider="anthropic",
            tokens_in=200,
            tokens_out=100,
            model="claude-sonnet-4",
        )
        records = tracker.get_call_records("d1")
        assert len(records) == 2

    def test_get_all_debate_ids(self, populated_tracker):
        """get_all_debate_ids returns tracked debates."""
        ids = populated_tracker.get_all_debate_ids()
        assert "debate-1" in ids

    def test_clear_debate(self, populated_tracker):
        """clear_debate removes all records."""
        populated_tracker.clear_debate("debate-1")
        summary = populated_tracker.get_debate_cost("debate-1")
        assert summary.total_calls == 0
        assert summary.total_cost_usd == Decimal("0")

    def test_unknown_provider_uses_default_rates(self, tracker):
        """Unknown provider falls back to openrouter default rates."""
        record = tracker.record_agent_call(
            debate_id="d1",
            agent_name="custom",
            provider="unknown_provider",
            tokens_in=1_000_000,
            tokens_out=0,
            model="custom-model",
        )
        # Default input rate is $2.00/1M tokens
        assert record.cost_usd == Decimal("2.00")


# =============================================================================
# DebateCostSummary Tests
# =============================================================================


class TestDebateCostSummary:
    """Tests for the DebateCostSummary output."""

    def test_empty_debate(self, tracker):
        """Empty debate returns zero-cost summary."""
        summary = tracker.get_debate_cost("nonexistent")
        assert summary.debate_id == "nonexistent"
        assert summary.total_cost_usd == Decimal("0")
        assert summary.total_calls == 0
        assert summary.per_agent == {}
        assert summary.per_round == {}
        assert summary.model_usage == {}

    def test_total_cost_accumulates(self, populated_tracker):
        """Total cost is the sum of all call costs."""
        summary = populated_tracker.get_debate_cost("debate-1")
        assert summary.total_calls == 3
        assert summary.total_cost_usd > Decimal("0")

        # Verify it sums correctly
        records = populated_tracker.get_call_records("debate-1")
        expected_total = sum(r.cost_usd for r in records)
        assert summary.total_cost_usd == expected_total

    def test_per_agent_breakdown(self, populated_tracker):
        """Per-agent breakdown groups costs by agent name."""
        summary = populated_tracker.get_debate_cost("debate-1")
        assert "claude" in summary.per_agent
        assert "gpt" in summary.per_agent

        claude = summary.per_agent["claude"]
        assert claude.call_count == 2
        assert claude.total_tokens_in == 4500  # 2000 + 2500
        assert claude.total_tokens_out == 1800  # 800 + 1000
        assert claude.total_cost_usd > Decimal("0")
        assert "claude-sonnet-4" in claude.models_used
        assert claude.models_used["claude-sonnet-4"] == 2

        gpt = summary.per_agent["gpt"]
        assert gpt.call_count == 1
        assert gpt.total_tokens_in == 1500

    def test_per_round_breakdown(self, populated_tracker):
        """Per-round breakdown groups costs by round number."""
        summary = populated_tracker.get_debate_cost("debate-1")
        assert 1 in summary.per_round
        assert 2 in summary.per_round

        round1 = summary.per_round[1]
        assert round1.call_count == 2  # claude + gpt in round 1
        assert round1.total_tokens_in == 3500  # 2000 + 1500

        round2 = summary.per_round[2]
        assert round2.call_count == 1
        assert round2.total_tokens_in == 2500

    def test_model_usage(self, populated_tracker):
        """Model usage tracks unique provider/model combos."""
        summary = populated_tracker.get_debate_cost("debate-1")
        assert "anthropic/claude-sonnet-4" in summary.model_usage
        assert "openai/gpt-4o" in summary.model_usage

        anthropic = summary.model_usage["anthropic/claude-sonnet-4"]
        assert anthropic.call_count == 2
        assert anthropic.provider == "anthropic"
        assert anthropic.model == "claude-sonnet-4"

    def test_timestamps(self, populated_tracker):
        """started_at and completed_at track call time range."""
        summary = populated_tracker.get_debate_cost("debate-1")
        assert summary.started_at is not None
        assert summary.completed_at is not None
        assert summary.started_at <= summary.completed_at

    def test_to_dict(self, populated_tracker):
        """to_dict produces JSON-serializable output."""
        summary = populated_tracker.get_debate_cost("debate-1")
        d = summary.to_dict()

        assert d["debate_id"] == "debate-1"
        assert isinstance(d["total_cost_usd"], str)
        assert d["total_calls"] == 3
        assert "claude" in d["per_agent"]
        assert "1" in d["per_round"]
        assert "anthropic/claude-sonnet-4" in d["model_usage"]

    def test_from_dict_round_trip(self, populated_tracker):
        """from_dict recovers a DebateCostSummary from to_dict output."""
        original = populated_tracker.get_debate_cost("debate-1")
        d = original.to_dict()
        restored = DebateCostSummary.from_dict(d)

        assert restored.debate_id == original.debate_id
        assert restored.total_cost_usd == original.total_cost_usd
        assert restored.total_calls == original.total_calls
        assert set(restored.per_agent.keys()) == set(original.per_agent.keys())
        assert set(restored.per_round.keys()) == set(original.per_round.keys())
        assert set(restored.model_usage.keys()) == set(original.model_usage.keys())

        # Verify agent details survive round-trip
        for name in original.per_agent:
            assert restored.per_agent[name].total_cost_usd == original.per_agent[name].total_cost_usd
            assert restored.per_agent[name].call_count == original.per_agent[name].call_count

    def test_from_dict_empty(self):
        """from_dict handles empty/minimal dict."""
        summary = DebateCostSummary.from_dict({})
        assert summary.debate_id == ""
        assert summary.total_cost_usd == Decimal("0")
        assert summary.total_calls == 0


# =============================================================================
# DecisionReceipt Integration
# =============================================================================


class TestReceiptCostIntegration:
    """Tests for cost_summary inclusion in DecisionReceipt."""

    def test_receipt_to_dict_includes_cost_summary(self):
        """DecisionReceipt.to_dict() includes cost_summary field."""
        from aragora.gauntlet.receipt_models import DecisionReceipt

        receipt = DecisionReceipt(
            receipt_id="r-1",
            gauntlet_id="d-1",
            timestamp="2026-01-01T00:00:00Z",
            input_summary="test",
            input_hash="abc123",
            risk_summary={"total": 0},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=0,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.8,
            cost_summary={
                "total_cost_usd": "0.045",
                "per_agent": {"claude": {"total_cost_usd": "0.030"}},
            },
        )
        d = receipt.to_dict()
        assert "cost_summary" in d
        assert d["cost_summary"]["total_cost_usd"] == "0.045"

    def test_receipt_to_dict_cost_summary_none(self):
        """cost_summary is None when not provided."""
        from aragora.gauntlet.receipt_models import DecisionReceipt

        receipt = DecisionReceipt(
            receipt_id="r-2",
            gauntlet_id="d-2",
            timestamp="2026-01-01T00:00:00Z",
            input_summary="test",
            input_hash="def456",
            risk_summary={"total": 0},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=0,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.8,
        )
        d = receipt.to_dict()
        assert "cost_summary" in d
        assert d["cost_summary"] is None

    def test_receipt_from_dict_preserves_cost_summary(self):
        """from_dict recovers cost_summary."""
        from aragora.gauntlet.receipt_models import DecisionReceipt

        data = {
            "receipt_id": "r-3",
            "gauntlet_id": "d-3",
            "timestamp": "2026-01-01T00:00:00Z",
            "input_summary": "test",
            "input_hash": "ghi789",
            "risk_summary": {"total": 0},
            "attacks_attempted": 0,
            "attacks_successful": 0,
            "probes_run": 0,
            "vulnerabilities_found": 0,
            "verdict": "PASS",
            "confidence": 0.9,
            "robustness_score": 0.8,
            "cost_summary": {
                "total_cost_usd": "0.10",
                "per_agent": {"gpt": {"total_cost_usd": "0.10"}},
            },
        }
        receipt = DecisionReceipt.from_dict(data)
        assert receipt.cost_summary is not None
        assert receipt.cost_summary["total_cost_usd"] == "0.10"

    def test_from_debate_result_with_cost_summary(self):
        """from_debate_result passes cost_summary through."""
        from aragora.gauntlet.receipt_models import DecisionReceipt

        mock_result = MagicMock()
        mock_result.debate_id = "debate-99"
        mock_result.id = "debate-99"
        mock_result.messages = []
        mock_result.votes = []
        mock_result.consensus_reached = True
        mock_result.confidence = 0.85
        mock_result.participants = ["claude", "gpt"]
        mock_result.dissenting_views = []
        mock_result.consensus_strength = "strong"
        mock_result.final_answer = "Test answer"
        mock_result.task = "Test task"
        mock_result.winner = "claude"
        mock_result.rounds_used = 3
        mock_result.duration_seconds = 45.0
        mock_result.convergence_similarity = 0.9

        cost_data = {
            "debate_id": "debate-99",
            "total_cost_usd": "0.025",
            "total_calls": 6,
            "per_agent": {"claude": {"total_cost_usd": "0.015"}},
        }

        receipt = DecisionReceipt.from_debate_result(
            mock_result,
            cost_summary=cost_data,
        )
        assert receipt.cost_summary == cost_data
        assert receipt.cost_summary["total_cost_usd"] == "0.025"


# =============================================================================
# Global Singleton
# =============================================================================


class TestGlobalSingleton:
    """Tests for the get_debate_cost_tracker singleton."""

    def test_get_debate_cost_tracker_returns_instance(self):
        """Singleton returns a DebateCostTracker."""
        with patch("aragora.billing.debate_costs._debate_cost_tracker", None):
            tracker = get_debate_cost_tracker()
            assert isinstance(tracker, DebateCostTracker)

    def test_get_debate_cost_tracker_returns_same_instance(self):
        """Singleton returns the same instance on subsequent calls."""
        with patch("aragora.billing.debate_costs._debate_cost_tracker", None):
            t1 = get_debate_cost_tracker()
            t2 = get_debate_cost_tracker()
            assert t1 is t2


# =============================================================================
# Dataclass Unit Tests
# =============================================================================


class TestAgentCostBreakdown:
    """Tests for AgentCostBreakdown dataclass."""

    def test_to_dict(self):
        breakdown = AgentCostBreakdown(
            agent_name="claude",
            total_cost_usd=Decimal("0.050"),
            total_tokens_in=5000,
            total_tokens_out=2000,
            call_count=3,
            models_used={"claude-sonnet-4": 2, "claude-opus-4": 1},
        )
        d = breakdown.to_dict()
        assert d["agent_name"] == "claude"
        assert d["total_cost_usd"] == "0.050"
        assert d["call_count"] == 3
        assert d["models_used"]["claude-sonnet-4"] == 2


class TestRoundCostBreakdown:
    """Tests for RoundCostBreakdown dataclass."""

    def test_to_dict(self):
        bd = RoundCostBreakdown(
            round_number=1,
            total_cost_usd=Decimal("0.012"),
            total_tokens_in=2000,
            total_tokens_out=800,
            call_count=2,
        )
        d = bd.to_dict()
        assert d["round_number"] == 1
        assert d["total_cost_usd"] == "0.012"


class TestModelUsage:
    """Tests for ModelUsage dataclass."""

    def test_to_dict(self):
        mu = ModelUsage(
            provider="anthropic",
            model="claude-sonnet-4",
            total_cost_usd=Decimal("0.035"),
            total_tokens_in=3500,
            total_tokens_out=1400,
            call_count=2,
        )
        d = mu.to_dict()
        assert d["provider"] == "anthropic"
        assert d["model"] == "claude-sonnet-4"
        assert d["call_count"] == 2


# =============================================================================
# Multi-Debate Isolation
# =============================================================================


class TestMultiDebateIsolation:
    """Tests that costs for different debates are isolated."""

    def test_separate_debates_have_separate_costs(self, tracker):
        """Recording in debate-A does not affect debate-B."""
        tracker.record_agent_call(
            debate_id="A",
            agent_name="claude",
            provider="anthropic",
            tokens_in=1000,
            tokens_out=500,
            model="claude-sonnet-4",
        )
        tracker.record_agent_call(
            debate_id="B",
            agent_name="gpt",
            provider="openai",
            tokens_in=2000,
            tokens_out=1000,
            model="gpt-4o",
        )

        summary_a = tracker.get_debate_cost("A")
        summary_b = tracker.get_debate_cost("B")

        assert summary_a.total_calls == 1
        assert summary_b.total_calls == 1
        assert "claude" in summary_a.per_agent
        assert "claude" not in summary_b.per_agent
        assert "gpt" in summary_b.per_agent
        assert "gpt" not in summary_a.per_agent

    def test_clear_one_debate_preserves_other(self, tracker):
        """Clearing debate-A does not affect debate-B."""
        tracker.record_agent_call(
            debate_id="A",
            agent_name="claude",
            provider="anthropic",
            tokens_in=1000,
            tokens_out=500,
            model="claude-sonnet-4",
        )
        tracker.record_agent_call(
            debate_id="B",
            agent_name="gpt",
            provider="openai",
            tokens_in=2000,
            tokens_out=1000,
            model="gpt-4o",
        )

        tracker.clear_debate("A")

        summary_a = tracker.get_debate_cost("A")
        summary_b = tracker.get_debate_cost("B")

        assert summary_a.total_calls == 0
        assert summary_b.total_calls == 1
