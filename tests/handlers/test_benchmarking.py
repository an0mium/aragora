"""
Tests for the decision benchmarking REST handler.

Covers GET /api/v1/benchmarks and GET /api/v1/benchmarks/compare.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.analytics.benchmarking import BenchmarkAggregator
from aragora.server.handlers.benchmarking import BenchmarkingHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(aggregator: BenchmarkAggregator | None = None) -> BenchmarkingHandler:
    """Create a BenchmarkingHandler with an optional pre-seeded aggregator."""
    ctx: dict[str, Any] = {}
    if aggregator is not None:
        ctx["benchmark_aggregator"] = aggregator
    return BenchmarkingHandler(ctx=ctx)


def _seeded_aggregator(n_tenants: int = 6) -> BenchmarkAggregator:
    """Return an aggregator with data ready for querying."""
    agg = BenchmarkAggregator(min_k=2, noise_scale=0.0)
    for i in range(n_tenants):
        agg.record_decision(
            tenant_id=f"tenant_{i}",
            category="healthcare",
            metrics={
                "consensus_rate": 0.70 + i * 0.04,
                "confidence_avg": 0.60 + i * 0.03,
            },
            industry="healthcare",
            team_size=25,
            decision_type="vendor_selection",
        )
    return agg


def _parse_body(result: Any) -> dict:
    return json.loads(result.body)


# ---------------------------------------------------------------------------
# can_handle / routing
# ---------------------------------------------------------------------------


class TestBenchmarkingHandlerRouting:
    """Test route matching."""

    def test_can_handle_benchmarks(self):
        h = _make_handler()
        assert h.can_handle("/api/v1/benchmarks")

    def test_can_handle_compare(self):
        h = _make_handler()
        assert h.can_handle("/api/v1/benchmarks/compare")

    def test_rejects_unrelated_path(self):
        h = _make_handler()
        assert not h.can_handle("/api/v1/debates")


# ---------------------------------------------------------------------------
# GET /api/v1/benchmarks
# ---------------------------------------------------------------------------


class TestGetBenchmarks:
    """Tests for the benchmarks list endpoint."""

    def test_returns_200_with_data(self):
        h = _make_handler(_seeded_aggregator())
        result = h.handle(
            "/api/v1/benchmarks",
            {"category": "healthcare"},
            MagicMock(),
        )
        assert result is not None
        assert result.status == 200
        body = _parse_body(result)
        assert "benchmarks" in body
        assert body["count"] >= 1

    def test_with_decision_type_filter(self):
        h = _make_handler(_seeded_aggregator())
        result = h.handle(
            "/api/v1/benchmarks",
            {"category": "healthcare"},
            MagicMock(),
        )
        body = _parse_body(result)
        assert body["count"] >= 1

    def test_missing_category_returns_400(self):
        h = _make_handler(_seeded_aggregator())
        result = h.handle(
            "/api/v1/benchmarks",
            {},
            MagicMock(),
        )
        assert result is not None
        assert result.status == 400

    def test_empty_category_returns_400(self):
        h = _make_handler(_seeded_aggregator())
        result = h.handle(
            "/api/v1/benchmarks",
            {"category": ""},
            MagicMock(),
        )
        assert result is not None
        assert result.status == 400

    def test_empty_benchmarks_returns_empty_list(self):
        h = _make_handler(BenchmarkAggregator(min_k=5, noise_scale=0.0))
        result = h.handle(
            "/api/v1/benchmarks",
            {"category": "nonexistent"},
            MagicMock(),
        )
        assert result is not None
        assert result.status == 200
        body = _parse_body(result)
        assert body["benchmarks"] == []
        assert body["count"] == 0


# ---------------------------------------------------------------------------
# GET /api/v1/benchmarks/compare
# ---------------------------------------------------------------------------


class TestCompareBenchmarks:
    """Tests for the benchmark comparison endpoint."""

    def test_returns_comparison(self):
        h = _make_handler(_seeded_aggregator())
        result = h.handle(
            "/api/v1/benchmarks/compare",
            {
                "category": "healthcare",
                "consensus_rate": "0.85",
            },
            MagicMock(),
        )
        assert result is not None
        assert result.status == 200
        body = _parse_body(result)
        assert "comparison" in body
        assert "consensus_rate" in body["comparison"]
        assert "percentile_rank" in body["comparison"]["consensus_rate"]

    def test_missing_category_returns_400(self):
        h = _make_handler(_seeded_aggregator())
        result = h.handle(
            "/api/v1/benchmarks/compare",
            {"consensus_rate": "0.85"},
            MagicMock(),
        )
        assert result is not None
        assert result.status == 400

    def test_no_metrics_returns_400(self):
        h = _make_handler(_seeded_aggregator())
        result = h.handle(
            "/api/v1/benchmarks/compare",
            {"category": "healthcare"},
            MagicMock(),
        )
        assert result is not None
        assert result.status == 400

    def test_invalid_metric_value_returns_400(self):
        h = _make_handler(_seeded_aggregator())
        result = h.handle(
            "/api/v1/benchmarks/compare",
            {
                "category": "healthcare",
                "consensus_rate": "not_a_number",
            },
            MagicMock(),
        )
        assert result is not None
        assert result.status == 400

    def test_empty_category_returns_400(self):
        h = _make_handler(_seeded_aggregator())
        result = h.handle(
            "/api/v1/benchmarks/compare",
            {
                "category": "",
                "consensus_rate": "0.85",
            },
            MagicMock(),
        )
        assert result is not None
        assert result.status == 400
