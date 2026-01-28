"""
Composite Analytics Module for Knowledge Mound Phase A3.

This module provides cross-adapter analytics and composite SLO monitoring.
It identifies bottlenecks, computes parallel efficiency, and recommends
optimizations across the adapter ecosystem.

Key Components:
- CompositeMetrics: Cross-adapter metric aggregation
- BottleneckAnalysis: Adapter performance bottleneck detection
- CompositeAnalytics: Main analytics engine

Usage:
    from aragora.knowledge.mound.ops.composite_analytics import (
        CompositeAnalytics,
        CompositeMetrics,
    )

    analytics = CompositeAnalytics()
    metrics = analytics.compute_composite_metrics(sync_results)
    bottleneck = analytics.identify_bottleneck(sync_results)
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SLOStatus(Enum):
    """Status of SLO compliance."""

    MET = "met"
    """SLO is being met."""

    WARNING = "warning"
    """SLO is at risk (within 20% of threshold)."""

    VIOLATED = "violated"
    """SLO has been violated."""


class BottleneckSeverity(Enum):
    """Severity of identified bottleneck."""

    NONE = "none"
    """No significant bottleneck."""

    MINOR = "minor"
    """Minor bottleneck, some impact."""

    MODERATE = "moderate"
    """Moderate bottleneck, noticeable impact."""

    SEVERE = "severe"
    """Severe bottleneck, significant impact."""

    CRITICAL = "critical"
    """Critical bottleneck, system degradation."""


class OptimizationType(Enum):
    """Types of optimization recommendations."""

    PARALLELIZE = "parallelize"
    """Parallelize adapter execution."""

    CACHE = "cache"
    """Add caching for frequently accessed data."""

    BATCH = "batch"
    """Batch operations for efficiency."""

    TIMEOUT_ADJUST = "timeout_adjust"
    """Adjust timeout settings."""

    RETRY_ADJUST = "retry_adjust"
    """Adjust retry configuration."""

    SCALE = "scale"
    """Scale resources for adapter."""


@dataclass
class AdapterMetrics:
    """Metrics for a single adapter."""

    adapter_name: str
    """Name of the adapter."""

    sync_count: int = 0
    """Number of sync operations."""

    success_count: int = 0
    """Number of successful syncs."""

    failure_count: int = 0
    """Number of failed syncs."""

    total_time_ms: float = 0.0
    """Total time spent in milliseconds."""

    avg_time_ms: float = 0.0
    """Average sync time in milliseconds."""

    min_time_ms: float = 0.0
    """Minimum sync time."""

    max_time_ms: float = 0.0
    """Maximum sync time."""

    p95_time_ms: float = 0.0
    """95th percentile sync time."""

    p99_time_ms: float = 0.0
    """99th percentile sync time."""

    items_processed: int = 0
    """Total items processed."""

    items_updated: int = 0
    """Total items updated."""

    error_rate: float = 0.0
    """Failure rate (0-1)."""

    throughput_per_sec: float = 0.0
    """Items processed per second."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "adapter_name": self.adapter_name,
            "sync_count": self.sync_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_time_ms": round(self.avg_time_ms, 2),
            "min_time_ms": round(self.min_time_ms, 2),
            "max_time_ms": round(self.max_time_ms, 2),
            "p95_time_ms": round(self.p95_time_ms, 2),
            "p99_time_ms": round(self.p99_time_ms, 2),
            "items_processed": self.items_processed,
            "items_updated": self.items_updated,
            "error_rate": round(self.error_rate, 4),
            "throughput_per_sec": round(self.throughput_per_sec, 2),
        }


@dataclass
class SLOConfig:
    """SLO configuration thresholds."""

    sync_time_target_ms: float = 5000.0
    """Target total sync time in milliseconds."""

    adapter_time_target_ms: float = 1000.0
    """Target per-adapter time in milliseconds."""

    error_rate_threshold: float = 0.05
    """Maximum acceptable error rate (5%)."""

    success_rate_threshold: float = 0.95
    """Minimum success rate (95%)."""

    warning_threshold_ratio: float = 0.8
    """Ratio at which to trigger warning (80% of limit)."""

    throughput_min_per_sec: float = 10.0
    """Minimum items per second."""


@dataclass
class SLOResult:
    """Result of SLO evaluation."""

    slo_name: str
    """Name of the SLO."""

    target: float
    """Target value."""

    actual: float
    """Actual value."""

    status: SLOStatus
    """Compliance status."""

    margin: float = 0.0
    """Margin from threshold (positive = headroom, negative = violation)."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "slo_name": self.slo_name,
            "target": self.target,
            "actual": round(self.actual, 4),
            "status": self.status.value,
            "margin": round(self.margin, 4),
        }


@dataclass
class BottleneckAnalysis:
    """Analysis of adapter bottlenecks."""

    bottleneck_adapter: Optional[str] = None
    """Name of the bottleneck adapter (if any)."""

    severity: BottleneckSeverity = BottleneckSeverity.NONE
    """Severity of the bottleneck."""

    time_contribution_pct: float = 0.0
    """Percentage of total time attributed to bottleneck."""

    avg_time_ms: float = 0.0
    """Average time for bottleneck adapter."""

    second_slowest: Optional[str] = None
    """Second slowest adapter for comparison."""

    gap_ms: float = 0.0
    """Time gap between slowest and second slowest."""

    recommendation: str = ""
    """Recommendation for addressing bottleneck."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bottleneck_adapter": self.bottleneck_adapter,
            "severity": self.severity.value,
            "time_contribution_pct": round(self.time_contribution_pct, 2),
            "avg_time_ms": round(self.avg_time_ms, 2),
            "second_slowest": self.second_slowest,
            "gap_ms": round(self.gap_ms, 2),
            "recommendation": self.recommendation,
        }


@dataclass
class OptimizationRecommendation:
    """Recommendation for optimization."""

    adapter_name: str
    """Adapter this recommendation applies to."""

    optimization_type: OptimizationType
    """Type of optimization."""

    priority: int = 3
    """Priority (1=highest, 5=lowest)."""

    expected_improvement_pct: float = 0.0
    """Expected improvement percentage."""

    description: str = ""
    """Human-readable description."""

    implementation_notes: str = ""
    """Notes for implementation."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "adapter_name": self.adapter_name,
            "optimization_type": self.optimization_type.value,
            "priority": self.priority,
            "expected_improvement_pct": round(self.expected_improvement_pct, 2),
            "description": self.description,
            "implementation_notes": self.implementation_notes,
        }


@dataclass
class CompositeMetrics:
    """Composite metrics across all adapters."""

    total_sync_time_ms: float = 0.0
    """Total time for all syncs."""

    avg_sync_time_ms: float = 0.0
    """Average time per sync cycle."""

    parallel_efficiency: float = 0.0
    """Actual vs ideal parallel time (0-1)."""

    critical_path_time_ms: float = 0.0
    """Time of longest adapter (critical path)."""

    theoretical_parallel_time_ms: float = 0.0
    """Theoretical minimum with perfect parallelization."""

    adapter_count: int = 0
    """Number of adapters in the system."""

    successful_adapters: int = 0
    """Number of adapters with successful syncs."""

    adapter_metrics: Dict[str, AdapterMetrics] = field(default_factory=dict)
    """Per-adapter metrics."""

    composite_slo_met: bool = True
    """Whether composite SLO is met."""

    slo_results: List[SLOResult] = field(default_factory=list)
    """Individual SLO results."""

    bottleneck_analysis: Optional[BottleneckAnalysis] = None
    """Bottleneck analysis."""

    recommendations: List[OptimizationRecommendation] = field(default_factory=list)
    """Optimization recommendations."""

    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When metrics were computed."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_sync_time_ms": round(self.total_sync_time_ms, 2),
            "avg_sync_time_ms": round(self.avg_sync_time_ms, 2),
            "parallel_efficiency": round(self.parallel_efficiency, 4),
            "critical_path_time_ms": round(self.critical_path_time_ms, 2),
            "theoretical_parallel_time_ms": round(self.theoretical_parallel_time_ms, 2),
            "adapter_count": self.adapter_count,
            "successful_adapters": self.successful_adapters,
            "adapter_metrics": {k: v.to_dict() for k, v in self.adapter_metrics.items()},
            "composite_slo_met": self.composite_slo_met,
            "slo_results": [s.to_dict() for s in self.slo_results],
            "bottleneck_analysis": (
                self.bottleneck_analysis.to_dict() if self.bottleneck_analysis else None
            ),
            "recommendations": [r.to_dict() for r in self.recommendations],
            "computed_at": self.computed_at.isoformat(),
        }


@dataclass
class SyncResultInput:
    """Input format for sync results (matches BidirectionalCoordinator.SyncResult)."""

    adapter_name: str
    direction: str  # "forward" or "reverse"
    success: bool
    items_processed: int = 0
    items_updated: int = 0
    errors: List[str] = field(default_factory=list)
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CompositeAnalytics:
    """Engine for computing cross-adapter analytics.

    This engine aggregates metrics across adapters, identifies bottlenecks,
    and generates optimization recommendations.
    """

    def __init__(self, slo_config: Optional[SLOConfig] = None):
        """Initialize the composite analytics engine.

        Args:
            slo_config: Optional SLO configuration.
        """
        self.slo_config = slo_config or SLOConfig()
        self._historical_times: Dict[str, List[float]] = defaultdict(list)
        self._adapter_dependencies: Dict[str, Set[str]] = {}
        self._max_history = 100  # Keep last 100 measurements

    def compute_composite_metrics(
        self,
        sync_results: List[Any],
        include_recommendations: bool = True,
    ) -> CompositeMetrics:
        """Compute composite metrics from sync results.

        Args:
            sync_results: List of SyncResult objects or dicts.
            include_recommendations: Whether to generate recommendations.

        Returns:
            CompositeMetrics with aggregated metrics.
        """
        metrics = CompositeMetrics()

        if not sync_results:
            return metrics

        # Normalize input to standard format
        normalized = self._normalize_sync_results(sync_results)

        # Aggregate per-adapter metrics
        adapter_data: Dict[str, List[SyncResultInput]] = defaultdict(list)
        for result in normalized:
            adapter_data[result.adapter_name].append(result)

        total_time = 0.0
        max_time = 0.0
        successful = 0

        for adapter_name, results in adapter_data.items():
            adapter_metrics = self._compute_adapter_metrics(adapter_name, results)
            metrics.adapter_metrics[adapter_name] = adapter_metrics

            total_time += adapter_metrics.total_time_ms
            max_time = max(max_time, adapter_metrics.max_time_ms)

            if adapter_metrics.success_count > 0:
                successful += 1

            # Store in history
            for r in results:
                self._historical_times[adapter_name].append(float(r.duration_ms))
                # Trim history
                if len(self._historical_times[adapter_name]) > self._max_history:
                    self._historical_times[adapter_name] = self._historical_times[adapter_name][
                        -self._max_history :
                    ]

        metrics.adapter_count = len(adapter_data)
        metrics.successful_adapters = successful
        metrics.total_sync_time_ms = total_time
        metrics.avg_sync_time_ms = total_time / len(normalized) if normalized else 0
        metrics.critical_path_time_ms = max_time

        # Compute parallel efficiency
        # Theoretical parallel time = time of slowest adapter
        # Actual total time = sum of all adapter times (sequential)
        if total_time > 0 and metrics.adapter_count > 0:
            metrics.theoretical_parallel_time_ms = max_time
            # Efficiency: how close actual to ideal parallel
            # 1.0 = perfect parallel, lower = more sequential overhead
            if metrics.adapter_count > 1:
                # Ideal parallel would take max_time
                # Sequential takes total_time
                # Efficiency = ideal / actual (capped at 1.0)
                metrics.parallel_efficiency = min(
                    1.0, max_time / (total_time / metrics.adapter_count)
                )
            else:
                metrics.parallel_efficiency = 1.0

        # Evaluate SLOs
        metrics.slo_results = self._evaluate_slos(metrics)
        metrics.composite_slo_met = all(s.status != SLOStatus.VIOLATED for s in metrics.slo_results)

        # Identify bottleneck
        metrics.bottleneck_analysis = self._identify_bottleneck(metrics)

        # Generate recommendations
        if include_recommendations:
            metrics.recommendations = self._generate_recommendations(metrics)

        return metrics

    def _normalize_sync_results(self, sync_results: List[Any]) -> List[SyncResultInput]:
        """Normalize sync results to standard format.

        Args:
            sync_results: List of results in various formats.

        Returns:
            List of SyncResultInput.
        """
        normalized = []

        for result in sync_results:
            if isinstance(result, SyncResultInput):
                normalized.append(result)
            elif hasattr(result, "adapter_name"):
                # Duck-type SyncResult dataclass
                normalized.append(
                    SyncResultInput(
                        adapter_name=result.adapter_name,
                        direction=getattr(result, "direction", "forward"),
                        success=getattr(result, "success", True),
                        items_processed=getattr(result, "items_processed", 0),
                        items_updated=getattr(result, "items_updated", 0),
                        errors=getattr(result, "errors", []),
                        duration_ms=getattr(result, "duration_ms", 0),
                        metadata=getattr(result, "metadata", {}),
                    )
                )
            elif isinstance(result, dict):
                normalized.append(
                    SyncResultInput(
                        adapter_name=result.get("adapter_name", "unknown"),
                        direction=result.get("direction", "forward"),
                        success=result.get("success", True),
                        items_processed=result.get("items_processed", 0),
                        items_updated=result.get("items_updated", 0),
                        errors=result.get("errors", []),
                        duration_ms=result.get("duration_ms", 0),
                        metadata=result.get("metadata", {}),
                    )
                )

        return normalized

    def _compute_adapter_metrics(
        self,
        adapter_name: str,
        results: List[SyncResultInput],
    ) -> AdapterMetrics:
        """Compute metrics for a single adapter.

        Args:
            adapter_name: Name of the adapter.
            results: List of sync results for this adapter.

        Returns:
            AdapterMetrics for the adapter.
        """
        metrics = AdapterMetrics(adapter_name=adapter_name)

        if not results:
            return metrics

        times = [float(r.duration_ms) for r in results]
        successes = sum(1 for r in results if r.success)

        metrics.sync_count = len(results)
        metrics.success_count = successes
        metrics.failure_count = len(results) - successes
        metrics.total_time_ms = sum(times)
        metrics.avg_time_ms = statistics.mean(times) if times else 0
        metrics.min_time_ms = min(times) if times else 0
        metrics.max_time_ms = max(times) if times else 0

        # Percentiles
        sorted_times = sorted(times)
        if len(sorted_times) >= 2:
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)
            metrics.p95_time_ms = sorted_times[min(p95_idx, len(sorted_times) - 1)]
            metrics.p99_time_ms = sorted_times[min(p99_idx, len(sorted_times) - 1)]
        else:
            metrics.p95_time_ms = metrics.max_time_ms
            metrics.p99_time_ms = metrics.max_time_ms

        metrics.items_processed = sum(r.items_processed for r in results)
        metrics.items_updated = sum(r.items_updated for r in results)
        metrics.error_rate = (
            metrics.failure_count / metrics.sync_count if metrics.sync_count > 0 else 0
        )

        # Throughput
        total_time_sec = metrics.total_time_ms / 1000.0 if metrics.total_time_ms > 0 else 1.0
        metrics.throughput_per_sec = metrics.items_processed / total_time_sec

        return metrics

    def _evaluate_slos(self, metrics: CompositeMetrics) -> List[SLOResult]:
        """Evaluate SLOs against metrics.

        Args:
            metrics: Computed composite metrics.

        Returns:
            List of SLO results.
        """
        results = []
        config = self.slo_config

        # Total sync time SLO
        sync_time_status = self._compute_slo_status(
            actual=metrics.total_sync_time_ms,
            target=config.sync_time_target_ms,
            lower_is_better=True,
        )
        results.append(
            SLOResult(
                slo_name="total_sync_time",
                target=config.sync_time_target_ms,
                actual=metrics.total_sync_time_ms,
                status=sync_time_status,
                margin=(config.sync_time_target_ms - metrics.total_sync_time_ms)
                / config.sync_time_target_ms,
            )
        )

        # Per-adapter time SLOs
        for adapter_name, adapter_metrics in metrics.adapter_metrics.items():
            adapter_status = self._compute_slo_status(
                actual=adapter_metrics.avg_time_ms,
                target=config.adapter_time_target_ms,
                lower_is_better=True,
            )
            results.append(
                SLOResult(
                    slo_name=f"adapter_{adapter_name}_time",
                    target=config.adapter_time_target_ms,
                    actual=adapter_metrics.avg_time_ms,
                    status=adapter_status,
                    margin=(config.adapter_time_target_ms - adapter_metrics.avg_time_ms)
                    / config.adapter_time_target_ms,
                )
            )

            # Error rate SLO
            error_status = self._compute_slo_status(
                actual=adapter_metrics.error_rate,
                target=config.error_rate_threshold,
                lower_is_better=True,
            )
            results.append(
                SLOResult(
                    slo_name=f"adapter_{adapter_name}_error_rate",
                    target=config.error_rate_threshold,
                    actual=adapter_metrics.error_rate,
                    status=error_status,
                    margin=(config.error_rate_threshold - adapter_metrics.error_rate)
                    / config.error_rate_threshold
                    if config.error_rate_threshold > 0
                    else 1.0,
                )
            )

        # Overall success rate
        total_syncs = sum(m.sync_count for m in metrics.adapter_metrics.values())
        total_successes = sum(m.success_count for m in metrics.adapter_metrics.values())
        overall_success_rate = total_successes / total_syncs if total_syncs > 0 else 1.0

        success_status = self._compute_slo_status(
            actual=overall_success_rate,
            target=config.success_rate_threshold,
            lower_is_better=False,
        )
        results.append(
            SLOResult(
                slo_name="overall_success_rate",
                target=config.success_rate_threshold,
                actual=overall_success_rate,
                status=success_status,
                margin=(overall_success_rate - config.success_rate_threshold)
                / (1.0 - config.success_rate_threshold)
                if config.success_rate_threshold < 1.0
                else 1.0,
            )
        )

        return results

    def _compute_slo_status(
        self,
        actual: float,
        target: float,
        lower_is_better: bool,
    ) -> SLOStatus:
        """Compute SLO status for a metric.

        Args:
            actual: Actual value.
            target: Target threshold.
            lower_is_better: Whether lower values are better.

        Returns:
            SLO status.
        """
        warning_threshold = target * self.slo_config.warning_threshold_ratio

        if lower_is_better:
            if actual > target:
                return SLOStatus.VIOLATED
            elif actual > warning_threshold:
                return SLOStatus.WARNING
            else:
                return SLOStatus.MET
        else:
            if actual < target:
                return SLOStatus.VIOLATED
            elif actual < target / self.slo_config.warning_threshold_ratio:
                return SLOStatus.WARNING
            else:
                return SLOStatus.MET

    def _identify_bottleneck(self, metrics: CompositeMetrics) -> BottleneckAnalysis:
        """Identify the bottleneck adapter.

        Args:
            metrics: Computed composite metrics.

        Returns:
            BottleneckAnalysis.
        """
        analysis = BottleneckAnalysis()

        if not metrics.adapter_metrics:
            return analysis

        # Sort adapters by average time
        sorted_adapters = sorted(
            metrics.adapter_metrics.items(),
            key=lambda x: x[1].avg_time_ms,
            reverse=True,
        )

        if not sorted_adapters:
            return analysis

        slowest_name, slowest_metrics = sorted_adapters[0]
        analysis.bottleneck_adapter = slowest_name
        analysis.avg_time_ms = slowest_metrics.avg_time_ms

        # Time contribution
        total_time = sum(m.total_time_ms for m in metrics.adapter_metrics.values())
        if total_time > 0:
            analysis.time_contribution_pct = (slowest_metrics.total_time_ms / total_time) * 100

        # Second slowest for comparison
        if len(sorted_adapters) > 1:
            second_name, second_metrics = sorted_adapters[1]
            analysis.second_slowest = second_name
            analysis.gap_ms = slowest_metrics.avg_time_ms - second_metrics.avg_time_ms

        # Determine severity
        if analysis.time_contribution_pct > 50:
            analysis.severity = BottleneckSeverity.CRITICAL
            analysis.recommendation = (
                f"Critical: {slowest_name} accounts for {analysis.time_contribution_pct:.0f}% "
                "of sync time. Consider parallelization or caching."
            )
        elif analysis.time_contribution_pct > 35:
            analysis.severity = BottleneckSeverity.SEVERE
            analysis.recommendation = (
                f"Severe: {slowest_name} is a significant bottleneck. "
                "Investigate performance optimizations."
            )
        elif analysis.time_contribution_pct > 25:
            analysis.severity = BottleneckSeverity.MODERATE
            analysis.recommendation = (
                f"Moderate: {slowest_name} is slower than average. "
                "Consider optimization if time permits."
            )
        elif analysis.gap_ms > 100:
            analysis.severity = BottleneckSeverity.MINOR
            analysis.recommendation = (
                f"Minor: {slowest_name} is {analysis.gap_ms:.0f}ms slower than next. "
                "Low priority optimization opportunity."
            )
        else:
            analysis.severity = BottleneckSeverity.NONE
            analysis.recommendation = "No significant bottleneck detected."

        return analysis

    def _generate_recommendations(
        self,
        metrics: CompositeMetrics,
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations.

        Args:
            metrics: Computed composite metrics.

        Returns:
            List of recommendations.
        """
        recommendations = []

        # Check parallelization opportunities
        parallelizable = self.recommend_parallelization()
        if parallelizable:
            recommendations.append(
                OptimizationRecommendation(
                    adapter_name="system",
                    optimization_type=OptimizationType.PARALLELIZE,
                    priority=2,
                    expected_improvement_pct=min(50, (1 - metrics.parallel_efficiency) * 100),
                    description=f"Parallelize adapters: {', '.join(parallelizable)}",
                    implementation_notes=(
                        "These adapters have no dependencies and can run concurrently. "
                        "Use asyncio.gather() for parallel execution."
                    ),
                )
            )

        # Check for high-error adapters
        for name, adapter_metrics in metrics.adapter_metrics.items():
            if adapter_metrics.error_rate > 0.1:
                recommendations.append(
                    OptimizationRecommendation(
                        adapter_name=name,
                        optimization_type=OptimizationType.RETRY_ADJUST,
                        priority=1,
                        expected_improvement_pct=adapter_metrics.error_rate * 50,
                        description=f"High error rate ({adapter_metrics.error_rate:.1%}) for {name}",
                        implementation_notes=(
                            "Consider implementing exponential backoff, "
                            "circuit breaker, or health checks."
                        ),
                    )
                )

        # Check for slow adapters
        for name, adapter_metrics in metrics.adapter_metrics.items():
            if adapter_metrics.avg_time_ms > self.slo_config.adapter_time_target_ms:
                recommendations.append(
                    OptimizationRecommendation(
                        adapter_name=name,
                        optimization_type=OptimizationType.CACHE,
                        priority=2,
                        expected_improvement_pct=30,
                        description=f"Slow adapter: {name} ({adapter_metrics.avg_time_ms:.0f}ms avg)",
                        implementation_notes=(
                            "Consider adding caching for frequently accessed data "
                            "or batching operations."
                        ),
                    )
                )

        # Check low throughput
        for name, adapter_metrics in metrics.adapter_metrics.items():
            if (
                adapter_metrics.throughput_per_sec < self.slo_config.throughput_min_per_sec
                and adapter_metrics.items_processed > 0
            ):
                recommendations.append(
                    OptimizationRecommendation(
                        adapter_name=name,
                        optimization_type=OptimizationType.BATCH,
                        priority=3,
                        expected_improvement_pct=20,
                        description=f"Low throughput for {name}: {adapter_metrics.throughput_per_sec:.1f}/sec",
                        implementation_notes="Consider batching operations for better efficiency.",
                    )
                )

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority)

        return recommendations

    def set_adapter_dependencies(
        self,
        dependencies: Dict[str, List[str]],
    ) -> None:
        """Set adapter dependencies for parallelization analysis.

        Args:
            dependencies: Dict of adapter_name -> list of adapters it depends on.
        """
        self._adapter_dependencies = {adapter: set(deps) for adapter, deps in dependencies.items()}

    def recommend_parallelization(self) -> List[str]:
        """Recommend adapters safe to parallelize.

        Returns:
            List of adapter names safe for parallel execution.
        """
        if not self._historical_times:
            return []

        # Adapters with no dependencies can run in parallel
        all_adapters = set(self._historical_times.keys())
        dependent_adapters = set()

        for adapter, deps in self._adapter_dependencies.items():
            if deps:
                dependent_adapters.add(adapter)

        # Independent adapters can be parallelized
        parallelizable = all_adapters - dependent_adapters

        # Also include adapters that are dependencies but have no dependencies themselves
        for adapter in all_adapters:
            if adapter not in self._adapter_dependencies:
                parallelizable.add(adapter)

        return sorted(parallelizable)

    def get_historical_stats(self, adapter_name: str) -> Dict[str, float]:
        """Get historical statistics for an adapter.

        Args:
            adapter_name: Name of the adapter.

        Returns:
            Dict with historical statistics.
        """
        times = self._historical_times.get(adapter_name, [])

        if not times:
            return {"count": 0}

        sorted_times = sorted(times)
        count = len(times)

        return {
            "count": count,
            "avg_ms": statistics.mean(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "stddev_ms": statistics.stdev(times) if count > 1 else 0,
            "p50_ms": sorted_times[count // 2],
            "p95_ms": sorted_times[int(count * 0.95)] if count >= 20 else sorted_times[-1],
            "p99_ms": sorted_times[int(count * 0.99)] if count >= 100 else sorted_times[-1],
        }

    def compute_trend(
        self,
        adapter_name: str,
        window_size: int = 10,
    ) -> Dict[str, Any]:
        """Compute performance trend for an adapter.

        Args:
            adapter_name: Name of the adapter.
            window_size: Size of the sliding window.

        Returns:
            Dict with trend information.
        """
        times = self._historical_times.get(adapter_name, [])

        if len(times) < window_size * 2:
            return {"trend": "insufficient_data", "samples": len(times)}

        recent = times[-window_size:]
        previous = times[-window_size * 2 : -window_size]

        recent_avg = statistics.mean(recent)
        previous_avg = statistics.mean(previous)

        change_pct = ((recent_avg - previous_avg) / previous_avg) * 100 if previous_avg > 0 else 0

        if change_pct > 10:
            trend = "degrading"
        elif change_pct < -10:
            trend = "improving"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "recent_avg_ms": round(recent_avg, 2),
            "previous_avg_ms": round(previous_avg, 2),
            "change_pct": round(change_pct, 2),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked adapters.

        Returns:
            Summary dict with adapter statistics.
        """
        summary = {
            "adapter_count": len(self._historical_times),
            "adapters": {},
        }

        for adapter_name in self._historical_times:
            summary["adapters"][adapter_name] = {
                "stats": self.get_historical_stats(adapter_name),
                "trend": self.compute_trend(adapter_name),
            }

        return summary


# Singleton instance
_composite_analytics: Optional[CompositeAnalytics] = None


def get_composite_analytics(
    slo_config: Optional[SLOConfig] = None,
) -> CompositeAnalytics:
    """Get or create the singleton composite analytics engine.

    Args:
        slo_config: Optional SLO configuration (only used on first call).

    Returns:
        CompositeAnalytics instance.
    """
    global _composite_analytics
    if _composite_analytics is None:
        _composite_analytics = CompositeAnalytics(slo_config)
    return _composite_analytics


__all__ = [
    # Enums
    "SLOStatus",
    "BottleneckSeverity",
    "OptimizationType",
    # Dataclasses
    "AdapterMetrics",
    "SLOConfig",
    "SLOResult",
    "BottleneckAnalysis",
    "OptimizationRecommendation",
    "CompositeMetrics",
    "SyncResultInput",
    # Engine
    "CompositeAnalytics",
    "get_composite_analytics",
]
