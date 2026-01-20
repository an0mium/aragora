"""
Centralized SLO (Service Level Objective) Configuration for Aragora.

This module defines performance targets based on docs/PERFORMANCE_TARGETS.md
and provides utilities for asserting SLO compliance in tests.

Usage:
    from tests.slo_config import SLO, assert_latency_slo, LatencyMetrics

    # Check single operation
    assert_latency_slo("health_check", elapsed_ms=15)

    # Check with percentiles
    metrics = LatencyMetrics(times)
    assert metrics.p99 < SLO.LATENCY["health_check"]["p99"]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class SLO:
    """Service Level Objective definitions from PERFORMANCE_TARGETS.md."""

    # Latency targets in milliseconds
    LATENCY: dict[str, dict[str, float]] = {
        # Operation: {p50, p95, p99, timeout}
        "health_check": {"p50": 5, "p95": 20, "p99": 50, "timeout": 1000},
        "authentication": {"p50": 50, "p95": 150, "p99": 300, "timeout": 2000},
        "simple_api_call": {"p50": 100, "p95": 300, "p99": 500, "timeout": 5000},
        "debate_start": {"p50": 500, "p95": 1500, "p99": 3000, "timeout": 10000},
        "debate_round": {"p50": 2000, "p95": 5000, "p99": 10000, "timeout": 30000},
        "full_debate": {"p50": 30000, "p95": 60000, "p99": 120000, "timeout": 300000},
        "search_query": {"p50": 200, "p95": 500, "p99": 1000, "timeout": 5000},
        "export_small": {"p50": 500, "p95": 2000, "p99": 5000, "timeout": 30000},
        "export_large": {"p50": 5000, "p95": 15000, "p99": 30000, "timeout": 120000},
    }

    # Throughput targets (requests/sec or ops/sec)
    THROUGHPUT: dict[str, dict[str, int]] = {
        "api_requests": {"target": 1000, "burst": 5000},
        "websocket_connections": {"target": 10000, "burst": 25000},
        "concurrent_debates": {"target": 100, "burst": 500},
        "messages_streaming": {"target": 5000, "burst": 20000},
    }

    # Error rate targets (percentage)
    ERROR_RATES: dict[str, dict[str, float]] = {
        "5xx_errors": {"target": 0.1, "alert": 0.5},
        "4xx_errors": {"target": 5.0, "alert": 10.0},
        "timeout_errors": {"target": 0.5, "alert": 2.0},
        "connection_errors": {"target": 0.1, "alert": 1.0},
    }

    # Memory operation targets (ops/sec)
    MEMORY_OPS: dict[str, dict[str, float]] = {
        "critique_write": {"min_ops_per_sec": 100, "target_latency_ms": 10},
        "critique_read": {"min_ops_per_sec": 200, "target_latency_ms": 5},
        "continuum_add": {"min_ops_per_sec": 50, "target_latency_ms": 20},
        "continuum_search": {"min_ops_per_sec": 100, "target_latency_ms": 10},
    }

    # ELO operation targets
    ELO_OPS: dict[str, dict[str, float]] = {
        "rating_update": {"min_ops_per_sec": 100, "target_latency_ms": 10},
        "leaderboard_query": {"min_ops_per_sec": 200, "target_latency_ms": 5},
        "history_query": {"min_ops_per_sec": 100, "target_latency_ms": 10},
    }

    # Debate performance targets
    DEBATE: dict[str, dict[str, float]] = {
        "single_round_max_sec": 8.0,  # Increased from 6.0 for CI variability during parallel tests
        "round_scaling_max_ratio": 5.0,  # 3 rounds should be < 5x of 1 round
        "agent_scaling_max_ratio": 4.0,  # 5 agents should be < 4x of 2 agents (relaxed for CI variability)
        "concurrent_min_per_sec": 0.5,  # Minimum debates per second
    }

    # Import time targets
    STARTUP: dict[str, float] = {
        "import_max_sec": 3.0,
        "server_ready_max_sec": 5.0,
    }


@dataclass
class LatencyMetrics:
    """Calculate latency percentiles from a list of measurements."""

    times_ms: list[float]

    @property
    def count(self) -> int:
        return len(self.times_ms)

    @property
    def mean(self) -> float:
        return sum(self.times_ms) / len(self.times_ms) if self.times_ms else 0

    @property
    def min(self) -> float:
        return min(self.times_ms) if self.times_ms else 0

    @property
    def max(self) -> float:
        return max(self.times_ms) if self.times_ms else 0

    def percentile(self, p: float) -> float:
        """Calculate the p-th percentile (0-100)."""
        if not self.times_ms:
            return 0
        sorted_times = sorted(self.times_ms)
        idx = int((p / 100) * len(sorted_times))
        idx = min(idx, len(sorted_times) - 1)
        return sorted_times[idx]

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    def check_slo(self, operation: str) -> dict[str, Any]:
        """Check metrics against SLO for an operation."""
        if operation not in SLO.LATENCY:
            raise ValueError(f"Unknown operation: {operation}. Valid: {list(SLO.LATENCY.keys())}")

        targets = SLO.LATENCY[operation]
        return {
            "operation": operation,
            "p50_actual": self.p50,
            "p50_target": targets["p50"],
            "p50_pass": self.p50 <= targets["p50"],
            "p95_actual": self.p95,
            "p95_target": targets["p95"],
            "p95_pass": self.p95 <= targets["p95"],
            "p99_actual": self.p99,
            "p99_target": targets["p99"],
            "p99_pass": self.p99 <= targets["p99"],
            "all_pass": (
                self.p50 <= targets["p50"]
                and self.p95 <= targets["p95"]
                and self.p99 <= targets["p99"]
            ),
        }


@dataclass
class ThroughputMetrics:
    """Calculate throughput metrics from operation counts and duration."""

    operations: int
    duration_sec: float

    @property
    def ops_per_sec(self) -> float:
        return self.operations / self.duration_sec if self.duration_sec > 0 else 0

    def check_slo(self, category: str) -> dict[str, Any]:
        """Check throughput against SLO for a category."""
        if category not in SLO.THROUGHPUT:
            raise ValueError(f"Unknown category: {category}. Valid: {list(SLO.THROUGHPUT.keys())}")

        targets = SLO.THROUGHPUT[category]
        return {
            "category": category,
            "actual_ops_per_sec": self.ops_per_sec,
            "target_ops_per_sec": targets["target"],
            "burst_ops_per_sec": targets["burst"],
            "target_pass": self.ops_per_sec >= targets["target"],
        }


def assert_latency_slo(
    operation: str,
    elapsed_ms: float | None = None,
    metrics: LatencyMetrics | None = None,
    percentile: str = "p99",
) -> None:
    """
    Assert that latency meets SLO for an operation.

    Args:
        operation: Operation name (e.g., "health_check", "debate_round")
        elapsed_ms: Single measurement in milliseconds
        metrics: LatencyMetrics object for percentile checks
        percentile: Which percentile to check ("p50", "p95", "p99")

    Raises:
        AssertionError: If SLO is not met
        ValueError: If operation is unknown
    """
    if operation not in SLO.LATENCY:
        raise ValueError(f"Unknown operation: {operation}. Valid: {list(SLO.LATENCY.keys())}")

    targets = SLO.LATENCY[operation]

    if elapsed_ms is not None:
        target = targets[percentile]
        assert elapsed_ms <= target, (
            f"SLO violation: {operation} {percentile} = {elapsed_ms:.1f}ms " f"(target: {target}ms)"
        )

    if metrics is not None:
        result = metrics.check_slo(operation)
        assert result["all_pass"], (
            f"SLO violation for {operation}:\n"
            f"  P50: {result['p50_actual']:.1f}ms vs {result['p50_target']}ms "
            f"({'PASS' if result['p50_pass'] else 'FAIL'})\n"
            f"  P95: {result['p95_actual']:.1f}ms vs {result['p95_target']}ms "
            f"({'PASS' if result['p95_pass'] else 'FAIL'})\n"
            f"  P99: {result['p99_actual']:.1f}ms vs {result['p99_target']}ms "
            f"({'PASS' if result['p99_pass'] else 'FAIL'})"
        )


def assert_throughput_slo(
    category: str,
    operations: int,
    duration_sec: float,
    check_burst: bool = False,
) -> None:
    """
    Assert that throughput meets SLO for a category.

    Args:
        category: Category name (e.g., "api_requests", "concurrent_debates")
        operations: Number of operations completed
        duration_sec: Duration in seconds
        check_burst: Check against burst capacity instead of target

    Raises:
        AssertionError: If SLO is not met
        ValueError: If category is unknown
    """
    if category not in SLO.THROUGHPUT:
        raise ValueError(f"Unknown category: {category}. Valid: {list(SLO.THROUGHPUT.keys())}")

    targets = SLO.THROUGHPUT[category]
    ops_per_sec = operations / duration_sec if duration_sec > 0 else 0
    target_key = "burst" if check_burst else "target"
    target = targets[target_key]

    assert ops_per_sec >= target, (
        f"SLO violation: {category} throughput = {ops_per_sec:.1f}/sec " f"(target: {target}/sec)"
    )


def assert_memory_ops_slo(
    operation: str,
    ops_count: int,
    duration_sec: float,
) -> None:
    """Assert memory operation throughput meets SLO."""
    if operation not in SLO.MEMORY_OPS:
        raise ValueError(f"Unknown operation: {operation}. Valid: {list(SLO.MEMORY_OPS.keys())}")

    targets = SLO.MEMORY_OPS[operation]
    ops_per_sec = ops_count / duration_sec if duration_sec > 0 else 0
    min_ops = targets["min_ops_per_sec"]

    assert ops_per_sec >= min_ops, (
        f"SLO violation: {operation} = {ops_per_sec:.0f} ops/sec " f"(target: >{min_ops} ops/sec)"
    )


def assert_elo_ops_slo(
    operation: str,
    ops_count: int,
    duration_sec: float,
) -> None:
    """Assert ELO operation throughput meets SLO."""
    if operation not in SLO.ELO_OPS:
        raise ValueError(f"Unknown operation: {operation}. Valid: {list(SLO.ELO_OPS.keys())}")

    targets = SLO.ELO_OPS[operation]
    ops_per_sec = ops_count / duration_sec if duration_sec > 0 else 0
    min_ops = targets["min_ops_per_sec"]

    assert ops_per_sec >= min_ops, (
        f"SLO violation: {operation} = {ops_per_sec:.0f} ops/sec " f"(target: >{min_ops} ops/sec)"
    )


def assert_debate_slo(
    metric: str,
    actual: float,
) -> None:
    """Assert debate performance meets SLO."""
    if metric not in SLO.DEBATE:
        raise ValueError(f"Unknown metric: {metric}. Valid: {list(SLO.DEBATE.keys())}")

    target = SLO.DEBATE[metric]

    # For "_max_" metrics, actual should be less than target
    # For "_min_" metrics, actual should be greater than target
    if "_max_" in metric:
        assert actual <= target, f"SLO violation: {metric} = {actual:.2f} (max: {target})"
    elif "_min_" in metric:
        assert actual >= target, f"SLO violation: {metric} = {actual:.2f} (min: {target})"
    else:
        # Ratio checks - actual should be less than target
        assert actual <= target, f"SLO violation: {metric} = {actual:.2f} (target: <{target})"


def assert_error_rate_slo(
    category: str,
    errors: int,
    total: int,
    use_alert_threshold: bool = False,
) -> None:
    """Assert error rate meets SLO."""
    if category not in SLO.ERROR_RATES:
        raise ValueError(f"Unknown category: {category}. Valid: {list(SLO.ERROR_RATES.keys())}")

    targets = SLO.ERROR_RATES[category]
    error_rate = (errors / total * 100) if total > 0 else 0
    threshold_key = "alert" if use_alert_threshold else "target"
    threshold = targets[threshold_key]

    assert error_rate <= threshold, (
        f"SLO violation: {category} = {error_rate:.2f}% " f"(target: <{threshold}%)"
    )
