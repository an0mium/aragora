"""
Tests for the decision benchmarking analytics module.

Covers BenchmarkMetric dataclass, BenchmarkAggregator, k-anonymity enforcement,
differential noise (Laplace mechanism), category listing, comparison, and
handler endpoint tests.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.analytics.benchmarking import (
    BENCHMARK_METRICS,
    DECISION_TYPE_CATEGORIES,
    INDUSTRY_CATEGORIES,
    TEAM_SIZE_BUCKETS,
    BenchmarkAggregate,
    BenchmarkAggregator,
    BenchmarkMetric,
    _bucket_team_size,
    _laplace_noise,
    get_benchmark_aggregator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_aggregator(
    agg: BenchmarkAggregator,
    *,
    n_tenants: int = 6,
    category: str = "healthcare",
) -> None:
    """Populate an aggregator with n_tenants worth of decisions."""
    for i in range(n_tenants):
        agg.record_decision(
            tenant_id=f"tenant_{i}",
            category=category,
            metrics={
                "consensus_rate": 0.70 + i * 0.05,
                "confidence_avg": 0.60 + i * 0.04,
                "time_to_decision": 120.0 + i * 30,
                "cost_per_decision": 5.0 + i * 2,
                "calibration_score": 0.50 + i * 0.08,
            },
        )


def _seed_legacy(
    agg: BenchmarkAggregator,
    *,
    n_tenants: int = 6,
    industry: str = "healthcare",
    team_size: int = 15,
    decision_type: str = "vendor_selection",
) -> None:
    """Populate an aggregator using the legacy record_decision API."""
    for i in range(n_tenants):
        agg.record_decision(
            tenant_id=f"tenant_{i}",
            category="unused",  # ignored when industry is set
            industry=industry,
            team_size=team_size,
            decision_type=decision_type,
            metrics={
                "consensus_rate": 0.70 + i * 0.05,
                "confidence_avg": 0.60 + i * 0.04,
                "time_to_decision": 120.0 + i * 30,
                "cost_per_decision": 5.0 + i * 2,
                "calibration_score": 0.50 + i * 0.08,
            },
        )


# ===========================================================================
# BenchmarkMetric dataclass tests (3 tests)
# ===========================================================================


class TestBenchmarkMetric:
    """Tests for the BenchmarkMetric dataclass."""

    def test_fields(self):
        now = datetime.now(timezone.utc)
        metric = BenchmarkMetric(
            category="healthcare",
            metric="consensus_rate",
            p25=0.70,
            p50=0.80,
            p75=0.90,
            p90=0.95,
            sample_count=10,
            computed_at=now,
        )
        assert metric.category == "healthcare"
        assert metric.metric == "consensus_rate"
        assert metric.p25 == 0.70
        assert metric.p50 == 0.80
        assert metric.p75 == 0.90
        assert metric.p90 == 0.95
        assert metric.sample_count == 10
        assert metric.computed_at == now

    def test_default_computed_at(self):
        metric = BenchmarkMetric(
            category="test",
            metric="test",
            p25=0,
            p50=0,
            p75=0,
            p90=0,
            sample_count=0,
        )
        assert isinstance(metric.computed_at, datetime)
        assert metric.computed_at.tzinfo is not None

    def test_to_dict(self):
        now = datetime.now(timezone.utc)
        metric = BenchmarkMetric(
            category="financial",
            metric="cost_per_decision",
            p25=1.2345,
            p50=2.3456,
            p75=3.4567,
            p90=4.5678,
            sample_count=20,
            computed_at=now,
        )
        d = metric.to_dict()
        assert d["category"] == "financial"
        assert d["metric"] == "cost_per_decision"
        assert d["p25"] == 1.2345
        assert d["p50"] == 2.3456
        assert d["p75"] == 3.4567
        assert d["p90"] == 4.5678
        assert d["sample_count"] == 20
        assert d["computed_at"] == now.isoformat()

    def test_backward_compat_alias(self):
        """BenchmarkAggregate should be the same class as BenchmarkMetric."""
        assert BenchmarkAggregate is BenchmarkMetric


# ===========================================================================
# _bucket_team_size tests
# ===========================================================================


class TestBucketTeamSize:
    """Tests for team size bucketing per spec."""

    @pytest.mark.parametrize(
        "size,expected",
        [
            (1, "1-5"),
            (3, "1-5"),
            (5, "1-5"),
            (6, "6-20"),
            (15, "6-20"),
            (20, "6-20"),
            (21, "21-100"),
            (50, "21-100"),
            (100, "21-100"),
            (101, "100+"),
            (500, "100+"),
            (10000, "100+"),
        ],
    )
    def test_buckets(self, size: int, expected: str):
        assert _bucket_team_size(size) == expected


# ===========================================================================
# BenchmarkAggregator.compute_benchmarks tests (5 tests)
# ===========================================================================


class TestComputeBenchmarks:
    """Tests for BenchmarkAggregator.compute_benchmarks()."""

    def test_returns_all_metrics_with_enough_data(self):
        agg = BenchmarkAggregator(min_k=5, epsilon=0)
        _seed_aggregator(agg, n_tenants=6)
        results = agg.compute_benchmarks("healthcare")
        metrics_found = {r.metric for r in results}
        assert metrics_found == set(BENCHMARK_METRICS)

    def test_returns_correct_category(self):
        agg = BenchmarkAggregator(min_k=2, epsilon=0)
        _seed_aggregator(agg, n_tenants=3, category="legal")
        results = agg.compute_benchmarks("legal")
        assert all(r.category == "legal" for r in results)

    def test_empty_for_nonexistent_category(self):
        agg = BenchmarkAggregator(min_k=2, epsilon=0)
        _seed_aggregator(agg, n_tenants=5, category="healthcare")
        results = agg.compute_benchmarks("nonexistent")
        assert results == []

    def test_percentile_ordering_without_noise(self):
        """p25 <= p50 <= p75 <= p90 (without noise)."""
        agg = BenchmarkAggregator(min_k=1, epsilon=0)
        _seed_aggregator(agg, n_tenants=20)
        for r in agg.compute_benchmarks("healthcare"):
            assert r.p25 <= r.p50 <= r.p75 <= r.p90, f"Ordering violated for {r.metric}"

    def test_sample_count_is_correct(self):
        agg = BenchmarkAggregator(min_k=1, epsilon=0)
        _seed_aggregator(agg, n_tenants=7)
        for r in agg.compute_benchmarks("healthcare"):
            assert r.sample_count == 7


# ===========================================================================
# BenchmarkAggregator.compare tests (5 tests)
# ===========================================================================


class TestCompare:
    """Tests for BenchmarkAggregator.compare()."""

    def test_returns_percentile_ranks(self):
        agg = BenchmarkAggregator(min_k=2, epsilon=0)
        _seed_aggregator(agg, n_tenants=10, category="healthcare")

        comparison = agg.compare(
            tenant_metrics={"consensus_rate": 0.95, "confidence_avg": 0.50},
            category="healthcare",
        )
        assert "consensus_rate" in comparison
        assert "confidence_avg" in comparison
        assert "percentile_rank" in comparison["consensus_rate"]
        assert "benchmark_p50" in comparison["consensus_rate"]
        assert "benchmark_p75" in comparison["consensus_rate"]
        assert "value" in comparison["consensus_rate"]

    def test_high_value_gets_high_rank(self):
        agg = BenchmarkAggregator(min_k=2, epsilon=0)
        _seed_aggregator(agg, n_tenants=10, category="healthcare")

        comparison = agg.compare(
            tenant_metrics={"consensus_rate": 99.0},
            category="healthcare",
        )
        assert comparison["consensus_rate"]["percentile_rank"] == 95

    def test_low_value_gets_low_rank(self):
        agg = BenchmarkAggregator(min_k=2, epsilon=0)
        _seed_aggregator(agg, n_tenants=10, category="healthcare")

        comparison = agg.compare(
            tenant_metrics={"consensus_rate": 0.01},
            category="healthcare",
        )
        assert comparison["consensus_rate"]["percentile_rank"] == 25

    def test_unknown_metric_skipped(self):
        agg = BenchmarkAggregator(min_k=2, epsilon=0)
        _seed_aggregator(agg, n_tenants=5, category="healthcare")

        comparison = agg.compare(
            tenant_metrics={"nonexistent_metric": 42.0},
            category="healthcare",
        )
        assert comparison == {}

    def test_empty_benchmarks_returns_empty(self):
        agg = BenchmarkAggregator(min_k=5, epsilon=0)
        # Not enough tenants to pass k-anonymity
        _seed_aggregator(agg, n_tenants=3, category="healthcare")

        comparison = agg.compare(
            tenant_metrics={"consensus_rate": 0.8},
            category="healthcare",
        )
        assert comparison == {}


# ===========================================================================
# k-anonymity enforcement tests (3 tests)
# ===========================================================================


class TestKAnonymity:
    """Tests for k-anonymity enforcement."""

    def test_below_k_returns_empty(self):
        agg = BenchmarkAggregator(min_k=5, epsilon=0)
        _seed_aggregator(agg, n_tenants=4)
        results = agg.compute_benchmarks("healthcare")
        assert results == []

    def test_exactly_k_returns_results(self):
        agg = BenchmarkAggregator(min_k=5, epsilon=0)
        _seed_aggregator(agg, n_tenants=5)
        results = agg.compute_benchmarks("healthcare")
        assert len(results) > 0

    def test_duplicate_tenant_ids_not_double_counted(self):
        """Same tenant_id recorded twice should still count as 1 tenant."""
        agg = BenchmarkAggregator(min_k=3, epsilon=0)
        # Only 2 unique tenants, even though 4 records
        for _ in range(2):
            agg.record_decision(
                tenant_id="t1",
                category="test",
                metrics={"consensus_rate": 0.8},
            )
            agg.record_decision(
                tenant_id="t2",
                category="test",
                metrics={"consensus_rate": 0.9},
            )
        results = agg.compute_benchmarks("test")
        assert results == []  # Only 2 distinct tenants, need 3


# ===========================================================================
# Differential noise tests (2 tests)
# ===========================================================================


class TestDifferentialNoise:
    """Tests for differential noise (Laplace mechanism)."""

    def test_laplace_noise_nonzero(self):
        """With epsilon > 0, noise should be generated."""
        values = [_laplace_noise(1.0) for _ in range(100)]
        # Not all values should be zero
        assert any(v != 0.0 for v in values)

    def test_noise_perturbs_benchmarks(self):
        """Benchmarks with noise should differ from exact values."""
        agg_no_noise = BenchmarkAggregator(min_k=1, epsilon=0)
        agg_with_noise = BenchmarkAggregator(min_k=1, epsilon=1.0)

        for a in [agg_no_noise, agg_with_noise]:
            _seed_aggregator(a, n_tenants=10)

        clean = {(r.metric, r.p50) for r in agg_no_noise.compute_benchmarks("healthcare")}
        noisy = {(r.metric, r.p50) for r in agg_with_noise.compute_benchmarks("healthcare")}

        # At least some values should differ due to noise
        assert clean != noisy


# ===========================================================================
# Category listing tests (2 tests)
# ===========================================================================


class TestGetCategories:
    """Tests for BenchmarkAggregator.get_categories()."""

    def test_returns_categories_with_enough_data(self):
        agg = BenchmarkAggregator(min_k=3, epsilon=0)
        _seed_aggregator(agg, n_tenants=5, category="healthcare")
        _seed_aggregator(agg, n_tenants=5, category="financial")
        _seed_aggregator(agg, n_tenants=2, category="legal")  # too few

        categories = agg.get_categories()
        assert "healthcare" in categories
        assert "financial" in categories
        assert "legal" not in categories

    def test_empty_when_no_data(self):
        agg = BenchmarkAggregator(min_k=5, epsilon=0)
        assert agg.get_categories() == []


# ===========================================================================
# Legacy API backward compatibility
# ===========================================================================


class TestLegacyAPI:
    """Tests for backward-compatible record_decision with industry/team_size/decision_type."""

    def test_legacy_record_creates_composite_category(self):
        agg = BenchmarkAggregator(min_k=1, noise_scale=0.0)
        agg.record_decision(
            tenant_id="t1",
            category="ignored",
            industry="finance",
            team_size=5,
            decision_type="risk_assessment",
            metrics={"consensus_rate": 0.9},
        )
        # Should record under composite AND individual categories
        categories = {r.category for r in agg._records}
        assert "finance/1-5/risk_assessment" in categories
        assert "finance" in categories

    def test_legacy_get_benchmarks(self):
        agg = BenchmarkAggregator(min_k=2, noise_scale=0.0)
        _seed_legacy(agg, n_tenants=3, industry="healthcare", decision_type="vendor")

        # Legacy get_benchmarks uses composite prefix
        healthcare = agg.get_benchmarks("healthcare", "6-20")
        assert len(healthcare) > 0
        assert all("healthcare" in r.category for r in healthcare)


# ===========================================================================
# Thread safety
# ===========================================================================


class TestThreadSafety:
    """Tests for thread-safe recording."""

    def test_concurrent_recording(self):
        agg = BenchmarkAggregator(min_k=1, epsilon=0)
        errors: list[Exception] = []

        def _record(tid: int) -> None:
            try:
                for j in range(50):
                    agg.record_decision(
                        tenant_id=f"t{tid}_{j}",
                        category="test",
                        metrics={"consensus_rate": 0.5},
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_record, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(agg._records) == 200


# ===========================================================================
# Constants
# ===========================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_industry_categories(self):
        assert "healthcare" in INDUSTRY_CATEGORIES
        assert "financial" in INDUSTRY_CATEGORIES
        assert "legal" in INDUSTRY_CATEGORIES
        assert "technology" in INDUSTRY_CATEGORIES

    def test_decision_type_categories(self):
        assert "vendor" in DECISION_TYPE_CATEGORIES
        assert "hiring" in DECISION_TYPE_CATEGORIES
        assert "architecture" in DECISION_TYPE_CATEGORIES
        assert "pricing" in DECISION_TYPE_CATEGORIES
        assert "incident" in DECISION_TYPE_CATEGORIES

    def test_team_size_buckets(self):
        assert "1-5" in TEAM_SIZE_BUCKETS
        assert "6-20" in TEAM_SIZE_BUCKETS
        assert "21-100" in TEAM_SIZE_BUCKETS
        assert "100+" in TEAM_SIZE_BUCKETS


# ===========================================================================
# Handler endpoint tests (5 tests)
# ===========================================================================


class TestBenchmarkingHandler:
    """Tests for BenchmarkingHandler REST endpoints."""

    def _make_handler(self, aggregator=None):
        from aragora.server.handlers.benchmarking import BenchmarkingHandler

        ctx = {}
        if aggregator is not None:
            ctx["benchmark_aggregator"] = aggregator
        handler = BenchmarkingHandler(ctx)
        return handler

    def _mock_http_handler(self):
        """Create a mock HTTP handler with auth context for RBAC."""
        h = MagicMock()
        h.headers = {"Authorization": "Bearer test-token"}
        return h

    def _seed_agg(self):
        """Create and seed an aggregator with enough data."""
        agg = BenchmarkAggregator(min_k=3, epsilon=0)
        _seed_aggregator(agg, n_tenants=5, category="healthcare")
        _seed_aggregator(agg, n_tenants=5, category="financial")
        return agg

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.utils.decorators.has_permission", return_value=True)
    @patch("aragora.server.handlers.benchmarking.require_permission", lambda perm: lambda fn: fn)
    def test_get_benchmarks(self, _mock_perm):
        """GET /api/v1/benchmarks?category=healthcare returns benchmark data."""
        from aragora.server.handlers.benchmarking import BenchmarkingHandler

        agg = self._seed_agg()
        ctx = {"benchmark_aggregator": agg}
        handler = BenchmarkingHandler(ctx)

        result = handler._get_benchmarks({"category": "healthcare"})
        assert result is not None
        status, body_bytes, _ = result
        assert status == 200

        import json

        body = json.loads(body_bytes)
        assert "benchmarks" in body
        assert body["count"] == len(body["benchmarks"])
        assert body["count"] > 0

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.utils.decorators.has_permission", return_value=True)
    @patch("aragora.server.handlers.benchmarking.require_permission", lambda perm: lambda fn: fn)
    def test_get_benchmarks_missing_category(self, _mock_perm):
        """GET /api/v1/benchmarks without category returns 400."""
        from aragora.server.handlers.benchmarking import BenchmarkingHandler

        handler = BenchmarkingHandler({})
        result = handler._get_benchmarks({})
        assert result is not None
        assert result.status_code == 400

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.utils.decorators.has_permission", return_value=True)
    @patch("aragora.server.handlers.benchmarking.require_permission", lambda perm: lambda fn: fn)
    def test_get_categories(self, _mock_perm):
        """GET /api/v1/benchmarks/categories returns list of categories."""
        from aragora.server.handlers.benchmarking import BenchmarkingHandler

        agg = self._seed_agg()
        ctx = {"benchmark_aggregator": agg}
        handler = BenchmarkingHandler(ctx)

        result = handler._get_categories()
        assert result is not None
        status, body_bytes, _ = result
        assert status == 200

        import json

        body = json.loads(body_bytes)
        assert "categories" in body
        assert "healthcare" in body["categories"]
        assert "financial" in body["categories"]

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.utils.decorators.has_permission", return_value=True)
    @patch("aragora.server.handlers.benchmarking.require_permission", lambda perm: lambda fn: fn)
    def test_get_compare(self, _mock_perm):
        """GET /api/v1/benchmarks/compare returns comparison data."""
        from aragora.server.handlers.benchmarking import BenchmarkingHandler

        agg = self._seed_agg()
        ctx = {"benchmark_aggregator": agg}
        handler = BenchmarkingHandler(ctx)

        result = handler._get_compare(
            {
                "category": "healthcare",
                "consensus_rate": "0.85",
                "tenant_id": "test-tenant",
            }
        )
        assert result is not None
        status, body_bytes, _ = result
        assert status == 200

        import json

        body = json.loads(body_bytes)
        assert "comparison" in body
        assert "category" in body
        assert body["category"] == "healthcare"
        assert body["tenant_id"] == "test-tenant"
        assert "consensus_rate" in body["comparison"]

    @pytest.mark.no_auto_auth
    @patch("aragora.server.handlers.utils.decorators.has_permission", return_value=True)
    @patch("aragora.server.handlers.benchmarking.require_permission", lambda perm: lambda fn: fn)
    def test_get_compare_no_metrics(self, _mock_perm):
        """GET /api/v1/benchmarks/compare without metrics returns 400."""
        from aragora.server.handlers.benchmarking import BenchmarkingHandler

        handler = BenchmarkingHandler({})
        result = handler._get_compare({"category": "healthcare"})
        assert result is not None
        assert result.status_code == 400
