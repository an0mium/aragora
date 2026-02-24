"""
Tests for the differentiation dashboard REST handler.

Covers all 5 endpoints under /api/v1/differentiation/*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.differentiation import DifferentiationHandler


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------


@dataclass
class MockReceipt:
    id: str = "receipt-001"
    question: str = "Should we adopt microservices?"
    dissenting_views: list | None = None
    unresolved_tensions: list | None = None
    verified_claims: list | None = None
    robustness_score: float | None = 0.82
    topic: str = ""

    def __post_init__(self):
        if self.dissenting_views is None:
            self.dissenting_views = ["Agent-B disagrees due to complexity"]
        if self.unresolved_tensions is None:
            self.unresolved_tensions = []
        if self.verified_claims is None:
            self.verified_claims = ["Scalability claim verified"]


class MockReceiptStore:
    def __init__(self, receipts: list | None = None):
        self._receipts = (
            receipts
            if receipts is not None
            else [
                MockReceipt(id="r1", robustness_score=0.85),
                MockReceipt(id="r2", robustness_score=0.65, dissenting_views=[]),
                MockReceipt(
                    id="r3",
                    robustness_score=0.92,
                    dissenting_views=["View A", "View B"],
                    unresolved_tensions=["Tension 1"],
                    verified_claims=["C1", "C2", "C3"],
                ),
            ]
        )

    def list_receipts(self, limit: int = 20) -> list:
        return self._receipts[:limit]


_SENTINEL = object()


class MockEloSystem:
    def __init__(self, agents: dict | None | object = _SENTINEL):
        if agents is _SENTINEL:
            self._agents = {
                "claude-4": {"elo": 1650, "wins": 30, "losses": 10},
                "gpt-5": {"elo": 1580, "wins": 25, "losses": 15},
                "gemini-3": {"elo": 1520, "wins": 20, "losses": 20},
                "mistral-l": {"elo": 1450, "wins": 15, "losses": 25},
            }
        else:
            self._agents = agents or {}

    def get_all_ratings(self) -> dict:
        return self._agents

    def get_calibration_stats(self) -> dict:
        return {"avg_ece": 0.045, "avg_brier": 0.18}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(
    receipt_store: Any = None,
    elo_system: Any = None,
) -> DifferentiationHandler:
    ctx: dict[str, Any] = {}
    if receipt_store is not None:
        ctx["receipt_store"] = receipt_store
    if elo_system is not None:
        ctx["elo_system"] = elo_system
    return DifferentiationHandler(ctx=ctx)


def _parse_body(result: Any) -> dict:
    return json.loads(result.body)


def _parse_data(result: Any) -> dict:
    body = _parse_body(result)
    return body.get("data", body)


# ---------------------------------------------------------------------------
# can_handle / routing
# ---------------------------------------------------------------------------


class TestDifferentiationRouting:
    """Test route matching."""

    def test_handles_summary(self):
        h = _make_handler()
        assert h.can_handle("/api/v1/differentiation/summary")

    def test_handles_vetting(self):
        h = _make_handler()
        assert h.can_handle("/api/v1/differentiation/vetting")

    def test_handles_calibration(self):
        h = _make_handler()
        assert h.can_handle("/api/v1/differentiation/calibration")

    def test_handles_memory(self):
        h = _make_handler()
        assert h.can_handle("/api/v1/differentiation/memory")

    def test_handles_benchmarks(self):
        h = _make_handler()
        assert h.can_handle("/api/v1/differentiation/benchmarks")

    def test_does_not_handle_unrelated(self):
        h = _make_handler()
        assert not h.can_handle("/api/v1/debates")

    def test_does_not_handle_partial_match(self):
        h = _make_handler()
        assert not h.can_handle("/api/v1/differentiation")


# ---------------------------------------------------------------------------
# GET /api/v1/differentiation/summary
# ---------------------------------------------------------------------------


class TestSummaryEndpoint:
    """Test the summary endpoint."""

    def test_returns_summary_with_receipt_data(self):
        h = _make_handler(
            receipt_store=MockReceiptStore(),
            elo_system=MockEloSystem(),
        )
        result = h.handle("/api/v1/differentiation/summary", {}, None)
        data = _parse_data(result)

        assert data["total_decisions"] == 3
        assert data["active_agent_count"] == 4
        assert data["adversarial_vetting_enabled"] is True
        assert data["multi_model_consensus"] is True

    def test_dissent_preserved_rate(self):
        h = _make_handler(receipt_store=MockReceiptStore())
        result = h.handle("/api/v1/differentiation/summary", {}, None)
        data = _parse_data(result)

        # 2 of 3 receipts have dissenting views (r1 has default, r3 has views, r2 has empty)
        assert 0.0 <= data["dissent_preserved_rate"] <= 1.0

    def test_avg_robustness_score(self):
        h = _make_handler(receipt_store=MockReceiptStore())
        result = h.handle("/api/v1/differentiation/summary", {}, None)
        data = _parse_data(result)

        expected = (0.85 + 0.65 + 0.92) / 3
        assert abs(data["avg_robustness_score"] - expected) < 0.01

    def test_calibration_error_from_elo(self):
        h = _make_handler(elo_system=MockEloSystem())
        result = h.handle("/api/v1/differentiation/summary", {}, None)
        data = _parse_data(result)

        assert data["avg_calibration_error"] == 0.045

    def test_empty_stores(self):
        h = _make_handler(receipt_store=MockReceiptStore(receipts=[]))
        result = h.handle("/api/v1/differentiation/summary", {}, None)
        data = _parse_data(result)

        assert data["total_decisions"] == 0
        assert data["dissent_preserved_rate"] == 0.0
        assert data["avg_robustness_score"] == 0.0

    def test_no_stores_graceful(self):
        h = _make_handler()
        result = h.handle("/api/v1/differentiation/summary", {}, None)
        data = _parse_data(result)

        assert data["total_decisions"] == 0


# ---------------------------------------------------------------------------
# GET /api/v1/differentiation/vetting
# ---------------------------------------------------------------------------


class TestVettingEndpoint:
    """Test adversarial vetting evidence endpoint."""

    def test_returns_evidence_list(self):
        h = _make_handler(receipt_store=MockReceiptStore())
        result = h.handle("/api/v1/differentiation/vetting", {}, None)
        data = _parse_data(result)

        assert "evidence" in data
        assert "aggregates" in data
        assert len(data["evidence"]) == 3

    def test_evidence_fields(self):
        h = _make_handler(receipt_store=MockReceiptStore())
        result = h.handle("/api/v1/differentiation/vetting", {}, None)
        data = _parse_data(result)

        item = data["evidence"][0]
        assert "receipt_id" in item
        assert "dissenting_views_count" in item
        assert "unresolved_tensions_count" in item
        assert "verified_claims_count" in item
        assert "robustness_score" in item
        assert "has_adversarial_challenge" in item

    def test_aggregates_computed(self):
        h = _make_handler(receipt_store=MockReceiptStore())
        result = h.handle("/api/v1/differentiation/vetting", {}, None)
        data = _parse_data(result)

        agg = data["aggregates"]
        assert agg["total_decisions"] == 3
        assert 0 <= agg["adversarial_rate"] <= 1.0
        assert agg["adversarially_vetted"] >= 0

    def test_limit_param(self):
        receipts = [MockReceipt(id=f"r{i}") for i in range(30)]
        h = _make_handler(receipt_store=MockReceiptStore(receipts=receipts))
        result = h.handle("/api/v1/differentiation/vetting", {"limit": "5"}, None)
        data = _parse_data(result)

        assert len(data["evidence"]) == 5

    def test_limit_capped_at_50(self):
        receipts = [MockReceipt(id=f"r{i}") for i in range(60)]
        h = _make_handler(receipt_store=MockReceiptStore(receipts=receipts))
        result = h.handle("/api/v1/differentiation/vetting", {"limit": "100"}, None)
        data = _parse_data(result)

        assert len(data["evidence"]) == 50

    def test_empty_receipts(self):
        h = _make_handler(receipt_store=MockReceiptStore(receipts=[]))
        result = h.handle("/api/v1/differentiation/vetting", {}, None)
        data = _parse_data(result)

        assert data["evidence"] == []
        assert data["aggregates"]["total_decisions"] == 0
        assert data["aggregates"]["adversarial_rate"] == 0.0

    def test_receipt_with_no_dissent_not_adversarial(self):
        receipts = [
            MockReceipt(
                id="clean",
                dissenting_views=[],
                unresolved_tensions=[],
            )
        ]
        h = _make_handler(receipt_store=MockReceiptStore(receipts=receipts))
        result = h.handle("/api/v1/differentiation/vetting", {}, None)
        data = _parse_data(result)

        assert data["evidence"][0]["has_adversarial_challenge"] is False


# ---------------------------------------------------------------------------
# GET /api/v1/differentiation/calibration
# ---------------------------------------------------------------------------


class TestCalibrationEndpoint:
    """Test multi-agent calibration data endpoint."""

    def test_returns_agents_and_ensemble(self):
        h = _make_handler(elo_system=MockEloSystem())
        result = h.handle("/api/v1/differentiation/calibration", {}, None)
        data = _parse_data(result)

        assert "agents" in data
        assert "ensemble_metrics" in data
        assert len(data["agents"]) == 4

    def test_agents_sorted_by_elo(self):
        h = _make_handler(elo_system=MockEloSystem())
        result = h.handle("/api/v1/differentiation/calibration", {}, None)
        data = _parse_data(result)

        elos = [a["elo"] for a in data["agents"]]
        assert elos == sorted(elos, reverse=True)

    def test_ensemble_metrics(self):
        h = _make_handler(elo_system=MockEloSystem())
        result = h.handle("/api/v1/differentiation/calibration", {}, None)
        data = _parse_data(result)

        ensemble = data["ensemble_metrics"]
        assert ensemble["agent_count"] == 4
        assert ensemble["best_single_elo"] == 1650.0
        assert ensemble["elo_spread"] == 200.0
        assert ensemble["diversity_score"] == 0.4  # 4/10

    def test_win_rate_calculation(self):
        h = _make_handler(elo_system=MockEloSystem())
        result = h.handle("/api/v1/differentiation/calibration", {}, None)
        data = _parse_data(result)

        claude = next(a for a in data["agents"] if a["agent_id"] == "claude-4")
        assert claude["win_rate"] == 0.75  # 30/(30+10)
        assert claude["games_played"] == 40

    def test_empty_elo(self):
        empty_elo = MockEloSystem(agents={})
        # Ensure get_all_ratings returns empty dict
        assert empty_elo.get_all_ratings() == {}
        h = _make_handler(elo_system=empty_elo)
        result = h.handle("/api/v1/differentiation/calibration", {}, None)
        data = _parse_data(result)

        assert data["agents"] == []
        assert data["ensemble_metrics"]["agent_count"] == 0

    def test_no_elo_system_graceful(self):
        h = _make_handler()
        result = h.handle("/api/v1/differentiation/calibration", {}, None)
        data = _parse_data(result)

        assert data["agents"] == []


# ---------------------------------------------------------------------------
# GET /api/v1/differentiation/memory
# ---------------------------------------------------------------------------


class TestMemoryEndpoint:
    """Test institutional memory growth endpoint."""

    def test_returns_memory_structure(self):
        h = _make_handler()
        result = h.handle("/api/v1/differentiation/memory", {}, None)
        data = _parse_data(result)

        assert "memory" in data
        assert "knowledge_mound" in data
        assert "learning_indicators" in data

    def test_memory_tier_fields(self):
        h = _make_handler()
        result = h.handle("/api/v1/differentiation/memory", {}, None)
        data = _parse_data(result)

        mem = data["memory"]
        for field in ["total_entries", "fast_tier", "medium_tier", "slow_tier", "glacial_tier"]:
            assert field in mem

    def test_knowledge_mound_fields(self):
        h = _make_handler()
        result = h.handle("/api/v1/differentiation/memory", {}, None)
        data = _parse_data(result)

        km = data["knowledge_mound"]
        assert "total_artifacts" in km
        assert "adapter_count" in km
        assert "cross_debate_links" in km

    def test_default_adapter_count(self):
        h = _make_handler()
        result = h.handle("/api/v1/differentiation/memory", {}, None)
        data = _parse_data(result)

        assert data["knowledge_mound"]["adapter_count"] == 34

    def test_learning_indicators(self):
        h = _make_handler()
        result = h.handle("/api/v1/differentiation/memory", {}, None)
        data = _parse_data(result)

        learning = data["learning_indicators"]
        assert "decisions_informing_future" in learning
        assert "knowledge_reuse_rate" in learning
        assert "memory_quality_score" in learning


# ---------------------------------------------------------------------------
# GET /api/v1/differentiation/benchmarks
# ---------------------------------------------------------------------------


class TestBenchmarksEndpoint:
    """Test industry benchmark comparison endpoint."""

    def test_returns_benchmark_structure(self):
        h = _make_handler()
        result = h.handle("/api/v1/differentiation/benchmarks", {}, None)
        data = _parse_data(result)

        assert "category" in data
        assert "benchmarks" in data
        assert "available_categories" in data

    def test_default_category_is_technology(self):
        h = _make_handler()
        result = h.handle("/api/v1/differentiation/benchmarks", {}, None)
        data = _parse_data(result)

        assert data["category"] == "technology"

    def test_custom_category(self):
        h = _make_handler()
        result = h.handle(
            "/api/v1/differentiation/benchmarks",
            {"category": "healthcare"},
            None,
        )
        data = _parse_data(result)

        assert data["category"] == "healthcare"

    def test_available_categories_list(self):
        h = _make_handler()
        result = h.handle("/api/v1/differentiation/benchmarks", {}, None)
        data = _parse_data(result)

        cats = data["available_categories"]
        assert "healthcare" in cats
        assert "financial" in cats
        assert "legal" in cats
        assert "technology" in cats

    def test_benchmarks_graceful_without_aggregator(self):
        h = _make_handler()
        result = h.handle("/api/v1/differentiation/benchmarks", {}, None)
        data = _parse_data(result)

        # Should return empty benchmarks, not error
        assert isinstance(data["benchmarks"], list)
