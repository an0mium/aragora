"""
Decision Benchmarking endpoint handler.

Provides REST APIs for anonymized decision-quality benchmarks:

- GET /api/v1/benchmarks         — List benchmarks filtered by industry/team_size/decision_type
- GET /api/v1/benchmarks/compare — Compare tenant metrics against industry benchmarks
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.server.versioning.compat import strip_version_prefix

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from .utils.decorators import require_permission

logger = logging.getLogger(__name__)


class BenchmarkingHandler(BaseHandler):
    """Handler for decision benchmarking endpoints."""

    ROUTES = [
        "/api/benchmarks",
        "/api/benchmarks/compare",
    ]

    def __init__(self, ctx: dict | None = None):
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        normalized = strip_version_prefix(path)
        # Match both /api/benchmarks and /api/benchmarks/compare
        return normalized in self.ROUTES

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Route GET requests."""
        normalized = strip_version_prefix(path)

        if normalized == "/api/benchmarks/compare":
            return self._get_compare(query_params)
        if normalized == "/api/benchmarks":
            return self._get_benchmarks(query_params)

        return None

    @handle_errors("get benchmarks")
    def _get_benchmarks(self, query_params: dict[str, Any]) -> HandlerResult:
        """GET /api/v1/benchmarks — list aggregated benchmarks.

        Query params:
            industry (required): Industry vertical.
            team_size: Team-size bucket (small/medium/large/enterprise).
            decision_type: Optional decision-type filter.
        """
        industry = query_params.get("industry", "")
        team_size = query_params.get("team_size", "")
        decision_type = query_params.get("decision_type")

        if not industry or not team_size:
            return error_response(
                "Missing required query parameters: industry, team_size",
                400,
            )

        valid_buckets = {"small", "medium", "large", "enterprise"}
        if team_size not in valid_buckets:
            return error_response(
                f"Invalid team_size. Must be one of: {', '.join(sorted(valid_buckets))}",
                400,
            )

        from aragora.analytics.benchmarking import BenchmarkAggregator

        aggregator = self._get_aggregator()
        benchmarks = aggregator.get_benchmarks(industry, team_size, decision_type)

        items = [
            {
                "category": b.category,
                "metric": b.metric,
                "p25": b.p25,
                "p50": b.p50,
                "p75": b.p75,
                "p90": b.p90,
                "sample_count": b.sample_count,
                "computed_at": b.computed_at.isoformat(),
            }
            for b in benchmarks
        ]

        return json_response({"benchmarks": items, "count": len(items)})

    @handle_errors("compare benchmarks")
    def _get_compare(self, query_params: dict[str, Any]) -> HandlerResult:
        """GET /api/v1/benchmarks/compare — compare tenant metrics to benchmarks.

        Query params:
            industry (required): Industry vertical.
            team_size (required): Team-size bucket.
            consensus_rate, confidence_avg, time_to_decision,
            cost_per_decision, calibration_score: Metric values to compare.
        """
        industry = query_params.get("industry", "")
        team_size = query_params.get("team_size", "")

        if not industry or not team_size:
            return error_response(
                "Missing required query parameters: industry, team_size",
                400,
            )

        valid_buckets = {"small", "medium", "large", "enterprise"}
        if team_size not in valid_buckets:
            return error_response(
                f"Invalid team_size. Must be one of: {', '.join(sorted(valid_buckets))}",
                400,
            )

        from aragora.analytics.benchmarking import BENCHMARK_METRICS

        tenant_metrics: dict[str, float] = {}
        for metric in BENCHMARK_METRICS:
            raw = query_params.get(metric)
            if raw is not None:
                try:
                    tenant_metrics[metric] = float(raw)
                except (ValueError, TypeError):
                    return error_response(
                        f"Invalid value for {metric}: must be a number",
                        400,
                    )

        if not tenant_metrics:
            return error_response(
                "At least one metric must be provided for comparison",
                400,
            )

        aggregator = self._get_aggregator()
        comparison = aggregator.compare(tenant_metrics, industry, team_size)

        return json_response({"comparison": comparison, "industry": industry, "team_size": team_size})

    def _get_aggregator(self) -> Any:
        """Get or create the BenchmarkAggregator from server context."""
        aggregator = self.ctx.get("benchmark_aggregator")
        if aggregator is None:
            from aragora.analytics.benchmarking import BenchmarkAggregator

            aggregator = BenchmarkAggregator()
            self.ctx["benchmark_aggregator"] = aggregator
        return aggregator
