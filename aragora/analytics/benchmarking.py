"""
Decision Benchmarking Module.

Provides anonymized, k-anonymous decision-quality benchmarks so tenants
can compare their consensus rates, confidence, time-to-decision, cost,
and calibration against industry peers without leaking individual data.

Usage:
    from aragora.analytics.benchmarking import (
        BenchmarkAggregator,
        BenchmarkAggregate,
    )

    agg = BenchmarkAggregator(min_k=5)
    agg.record_decision(
        tenant_id="t1",
        industry="healthcare",
        team_size=12,
        decision_type="vendor_selection",
        metrics={"consensus_rate": 0.85, "confidence_avg": 0.72},
    )
    benchmarks = agg.compute_aggregates()
"""

from __future__ import annotations

import logging
import random
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Canonical metric names tracked by the aggregator
BENCHMARK_METRICS = (
    "consensus_rate",
    "confidence_avg",
    "time_to_decision",
    "cost_per_decision",
    "calibration_score",
)

# Team-size bucket boundaries
_TEAM_SIZE_BUCKETS = [
    (1, 10, "small"),
    (11, 50, "medium"),
    (51, 200, "large"),
    (201, float("inf"), "enterprise"),
]


def _bucket_team_size(size: int) -> str:
    """Return a privacy-safe team-size bucket label.

    Args:
        size: Number of team members.

    Returns:
        One of "small", "medium", "large", "enterprise".
    """
    for lo, hi, label in _TEAM_SIZE_BUCKETS:
        if lo <= size <= hi:
            return label
    return "enterprise"


@dataclass
class BenchmarkAggregate:
    """Aggregated benchmark percentiles for one category + metric."""

    category: str  # e.g. "healthcare/small/vendor_selection"
    metric: str  # e.g. "consensus_rate"
    p25: float
    p50: float
    p75: float
    p90: float
    sample_count: int
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class _DecisionRecord:
    """Internal record of a single decision's metrics."""

    tenant_id: str
    industry: str
    team_size_bucket: str
    decision_type: str
    metrics: dict[str, float]
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BenchmarkAggregator:
    """Computes anonymized decision-quality percentiles with k-anonymity.

    Only emits aggregates when at least *min_k* distinct tenants contribute
    to a bucket. Small differential noise is added to each percentile value
    to further protect individual tenant data.
    """

    def __init__(self, min_k: int = 5, noise_scale: float = 0.01) -> None:
        self.min_k = max(1, min_k)
        self.noise_scale = max(0.0, noise_scale)
        self._records: list[_DecisionRecord] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_decision(
        self,
        tenant_id: str,
        industry: str,
        team_size: int,
        decision_type: str,
        metrics: dict[str, float],
    ) -> None:
        """Record a single decision's quality metrics.

        Args:
            tenant_id: Opaque tenant identifier (used only for k-counting).
            industry: Industry vertical (e.g. "healthcare", "finance").
            team_size: Number of participants in the decision.
            decision_type: Category of decision (e.g. "vendor_selection").
            metrics: Mapping of metric name to float value.
        """
        bucket = _bucket_team_size(team_size)
        record = _DecisionRecord(
            tenant_id=tenant_id,
            industry=industry,
            team_size_bucket=bucket,
            decision_type=decision_type,
            metrics={k: v for k, v in metrics.items() if k in BENCHMARK_METRICS},
        )
        with self._lock:
            self._records.append(record)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def compute_aggregates(self) -> list[BenchmarkAggregate]:
        """Compute percentile benchmarks across all groups.

        Returns only groups that satisfy k-anonymity (>= min_k distinct tenants).
        """
        with self._lock:
            records = list(self._records)

        # Group records by (industry, team_size_bucket, decision_type)
        groups: dict[tuple[str, str, str], list[_DecisionRecord]] = defaultdict(list)
        for r in records:
            key = (r.industry, r.team_size_bucket, r.decision_type)
            groups[key].append(r)

        now = datetime.now(timezone.utc)
        aggregates: list[BenchmarkAggregate] = []

        for (industry, bucket, dtype), group_records in sorted(groups.items()):
            # k-anonymity check: count distinct tenants
            distinct_tenants = {r.tenant_id for r in group_records}
            if len(distinct_tenants) < self.min_k:
                logger.debug(
                    "Skipping group %s/%s/%s: only %d tenants (need %d)",
                    industry,
                    bucket,
                    dtype,
                    len(distinct_tenants),
                    self.min_k,
                )
                continue

            category = f"{industry}/{bucket}/{dtype}"

            for metric_name in BENCHMARK_METRICS:
                values = [
                    r.metrics[metric_name]
                    for r in group_records
                    if metric_name in r.metrics
                ]
                if not values:
                    continue

                values.sort()
                aggregates.append(
                    BenchmarkAggregate(
                        category=category,
                        metric=metric_name,
                        p25=self._percentile(values, 25),
                        p50=self._percentile(values, 50),
                        p75=self._percentile(values, 75),
                        p90=self._percentile(values, 90),
                        sample_count=len(values),
                        computed_at=now,
                    )
                )

        return aggregates

    def get_benchmarks(
        self,
        industry: str,
        team_size_bucket: str,
        decision_type: str | None = None,
    ) -> list[BenchmarkAggregate]:
        """Return benchmarks filtered by industry and team-size bucket.

        Args:
            industry: Industry to filter on.
            team_size_bucket: Team-size bucket ("small", "medium", etc.).
            decision_type: Optional decision-type filter.
        """
        all_aggs = self.compute_aggregates()
        prefix = f"{industry}/{team_size_bucket}/"
        results = [a for a in all_aggs if a.category.startswith(prefix)]
        if decision_type:
            results = [
                a for a in results if a.category == f"{industry}/{team_size_bucket}/{decision_type}"
            ]
        return results

    def compare(
        self,
        tenant_metrics: dict[str, float],
        industry: str,
        team_size_bucket: str,
    ) -> dict[str, Any]:
        """Compare a tenant's metrics against industry benchmarks.

        Returns a dict mapping each metric name to its percentile rank
        relative to the benchmark distribution.
        """
        benchmarks = self.get_benchmarks(industry, team_size_bucket)
        # Build a lookup: metric_name -> BenchmarkAggregate
        by_metric: dict[str, BenchmarkAggregate] = {}
        for b in benchmarks:
            by_metric.setdefault(b.metric, b)

        comparison: dict[str, Any] = {}
        for metric_name, value in tenant_metrics.items():
            if metric_name not in by_metric:
                continue
            agg = by_metric[metric_name]
            rank = self._estimate_percentile_rank(value, agg)
            comparison[metric_name] = {
                "value": value,
                "percentile_rank": rank,
                "benchmark_p50": agg.p50,
                "benchmark_p75": agg.p75,
            }
        return comparison

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _percentile(self, sorted_values: list[float], pct: int) -> float:
        """Compute a percentile with linear interpolation + differential noise."""
        n = len(sorted_values)
        if n == 0:
            return 0.0
        if n == 1:
            return sorted_values[0] + self._noise()

        k = (pct / 100.0) * (n - 1)
        lo = int(k)
        hi = min(lo + 1, n - 1)
        frac = k - lo
        raw = sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])
        return raw + self._noise()

    def _noise(self) -> float:
        """Generate small differential privacy noise."""
        if self.noise_scale <= 0:
            return 0.0
        return random.gauss(0, self.noise_scale)

    @staticmethod
    def _estimate_percentile_rank(value: float, agg: BenchmarkAggregate) -> int:
        """Estimate which percentile a value falls into."""
        if value <= agg.p25:
            return 25
        if value <= agg.p50:
            return 50
        if value <= agg.p75:
            return 75
        if value <= agg.p90:
            return 90
        return 95
