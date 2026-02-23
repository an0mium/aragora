"""
Tests for the decision benchmarking analytics module.

Covers BenchmarkAggregator, BenchmarkAggregate, k-anonymity,
differential noise, filtering, and comparison.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aragora.analytics.benchmarking import (
    BENCHMARK_METRICS,
    BenchmarkAggregate,
    BenchmarkAggregator,
    _bucket_team_size,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_aggregator(
    agg: BenchmarkAggregator,
    *,
    n_tenants: int = 6,
    industry: str = "healthcare",
    team_size: int = 15,
    decision_type: str = "vendor_selection",
) -> None:
    """Populate an aggregator with n_tenants worth of decisions."""
    for i in range(n_tenants):
        agg.record_decision(
            tenant_id=f"tenant_{i}",
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


# ---------------------------------------------------------------------------
# BenchmarkAggregate dataclass
# ---------------------------------------------------------------------------


class TestBenchmarkAggregate:
    """Tests for the BenchmarkAggregate dataclass."""

    def test_fields(self):
        now = datetime.now(timezone.utc)
        agg = BenchmarkAggregate(
            category="healthcare/medium/vendor_selection",
            metric="consensus_rate",
            p25=0.70,
            p50=0.80,
            p75=0.90,
            p90=0.95,
            sample_count=10,
            computed_at=now,
        )
        assert agg.category == "healthcare/medium/vendor_selection"
        assert agg.metric == "consensus_rate"
        assert agg.p25 == 0.70
        assert agg.p50 == 0.80
        assert agg.p75 == 0.90
        assert agg.p90 == 0.95
        assert agg.sample_count == 10
        assert agg.computed_at == now

    def test_default_computed_at(self):
        agg = BenchmarkAggregate(
            category="test",
            metric="test",
            p25=0,
            p50=0,
            p75=0,
            p90=0,
            sample_count=0,
        )
        assert isinstance(agg.computed_at, datetime)
        assert agg.computed_at.tzinfo is not None


# ---------------------------------------------------------------------------
# _bucket_team_size helper
# ---------------------------------------------------------------------------


class TestBucketTeamSize:
    """Tests for team size bucketing."""

    @pytest.mark.parametrize(
        "size,expected",
        [
            (1, "small"),
            (5, "small"),
            (10, "small"),
            (11, "medium"),
            (50, "medium"),
            (51, "large"),
            (200, "large"),
            (201, "enterprise"),
            (1000, "enterprise"),
        ],
    )
    def test_buckets(self, size: int, expected: str):
        assert _bucket_team_size(size) == expected


# ---------------------------------------------------------------------------
# BenchmarkAggregator
# ---------------------------------------------------------------------------


class TestBenchmarkAggregator:
    """Tests for the main aggregator class."""

    def test_record_decision_stores_data(self):
        agg = BenchmarkAggregator(min_k=1, noise_scale=0.0)
        agg.record_decision(
            tenant_id="t1",
            industry="finance",
            team_size=5,
            decision_type="risk_assessment",
            metrics={"consensus_rate": 0.9},
        )
        assert len(agg._records) == 1
        assert agg._records[0].tenant_id == "t1"
        assert agg._records[0].industry == "finance"
        assert agg._records[0].team_size_bucket == "small"

    def test_record_decision_filters_unknown_metrics(self):
        agg = BenchmarkAggregator(min_k=1, noise_scale=0.0)
        agg.record_decision(
            tenant_id="t1",
            industry="finance",
            team_size=5,
            decision_type="test",
            metrics={"consensus_rate": 0.9, "unknown_metric": 42.0},
        )
        assert "unknown_metric" not in agg._records[0].metrics
        assert "consensus_rate" in agg._records[0].metrics

    def test_compute_aggregates_with_enough_tenants(self):
        agg = BenchmarkAggregator(min_k=5, noise_scale=0.0)
        _seed_aggregator(agg, n_tenants=6)
        results = agg.compute_aggregates()
        assert len(results) > 0
        # Should have one aggregate per metric
        metrics_found = {r.metric for r in results}
        assert metrics_found == set(BENCHMARK_METRICS)

    def test_compute_aggregates_too_few_tenants_returns_empty(self):
        agg = BenchmarkAggregator(min_k=5, noise_scale=0.0)
        _seed_aggregator(agg, n_tenants=3)
        results = agg.compute_aggregates()
        assert results == []

    def test_k_anonymity_boundary(self):
        """Exactly min_k tenants should produce results."""
        agg = BenchmarkAggregator(min_k=5, noise_scale=0.0)
        _seed_aggregator(agg, n_tenants=5)
        results = agg.compute_aggregates()
        assert len(results) > 0

    def test_differential_noise_is_applied(self):
        """With noise_scale > 0, percentiles should differ from exact values."""
        agg_no_noise = BenchmarkAggregator(min_k=1, noise_scale=0.0)
        agg_with_noise = BenchmarkAggregator(min_k=1, noise_scale=0.1)

        for a in [agg_no_noise, agg_with_noise]:
            _seed_aggregator(a, n_tenants=10)

        clean = {(r.metric, r.p50) for r in agg_no_noise.compute_aggregates()}
        noisy = {(r.metric, r.p50) for r in agg_with_noise.compute_aggregates()}

        # At least some values should differ due to noise
        assert clean != noisy

    def test_percentile_ordering(self):
        """p25 <= p50 <= p75 <= p90 (without noise)."""
        agg = BenchmarkAggregator(min_k=1, noise_scale=0.0)
        _seed_aggregator(agg, n_tenants=20)
        for r in agg.compute_aggregates():
            assert r.p25 <= r.p50 <= r.p75 <= r.p90, f"Ordering violated for {r.metric}"

    def test_multiple_groups_computed_independently(self):
        agg = BenchmarkAggregator(min_k=2, noise_scale=0.0)
        _seed_aggregator(agg, n_tenants=3, industry="healthcare", decision_type="vendor")
        _seed_aggregator(agg, n_tenants=3, industry="finance", decision_type="risk")

        results = agg.compute_aggregates()
        categories = {r.category for r in results}
        assert "healthcare/medium/vendor" in categories
        assert "finance/medium/risk" in categories

    def test_get_benchmarks_filtering(self):
        agg = BenchmarkAggregator(min_k=2, noise_scale=0.0)
        _seed_aggregator(agg, n_tenants=3, industry="healthcare", decision_type="vendor")
        _seed_aggregator(agg, n_tenants=3, industry="finance", decision_type="risk")

        healthcare = agg.get_benchmarks("healthcare", "medium")
        assert all("healthcare" in r.category for r in healthcare)
        assert len(healthcare) > 0

        finance = agg.get_benchmarks("finance", "medium", "risk")
        assert all("finance/medium/risk" in r.category for r in finance)

    def test_get_benchmarks_no_match_returns_empty(self):
        agg = BenchmarkAggregator(min_k=2, noise_scale=0.0)
        _seed_aggregator(agg, n_tenants=3, industry="healthcare", decision_type="vendor")
        results = agg.get_benchmarks("nonexistent", "small")
        assert results == []

    def test_compare_returns_percentile_ranks(self):
        agg = BenchmarkAggregator(min_k=2, noise_scale=0.0)
        _seed_aggregator(agg, n_tenants=10, industry="healthcare", decision_type="vendor")

        comparison = agg.compare(
            tenant_metrics={"consensus_rate": 0.95, "confidence_avg": 0.50},
            industry="healthcare",
            team_size_bucket="medium",
        )
        assert "consensus_rate" in comparison
        assert "confidence_avg" in comparison
        assert "percentile_rank" in comparison["consensus_rate"]
        assert "benchmark_p50" in comparison["consensus_rate"]

    def test_compare_unknown_metric_skipped(self):
        agg = BenchmarkAggregator(min_k=2, noise_scale=0.0)
        _seed_aggregator(agg, n_tenants=5, industry="healthcare", decision_type="vendor")

        comparison = agg.compare(
            tenant_metrics={"nonexistent_metric": 42.0},
            industry="healthcare",
            team_size_bucket="medium",
        )
        assert comparison == {}

    def test_sample_count_is_correct(self):
        agg = BenchmarkAggregator(min_k=1, noise_scale=0.0)
        _seed_aggregator(agg, n_tenants=7)
        for r in agg.compute_aggregates():
            assert r.sample_count == 7

    def test_thread_safety(self):
        """Recording from multiple threads should not crash."""
        import threading

        agg = BenchmarkAggregator(min_k=1, noise_scale=0.0)
        errors = []

        def _record(tid: int):
            try:
                for j in range(50):
                    agg.record_decision(
                        tenant_id=f"t{tid}_{j}",
                        industry="test",
                        team_size=10,
                        decision_type="dt",
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
