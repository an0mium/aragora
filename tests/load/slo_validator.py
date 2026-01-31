"""
SLO Validation Module for Aragora Load Tests.

Provides utilities for validating Service Level Objectives during load tests.
Integrates with Locust, pytest, and k6 test results.

Usage:
    from tests.load.slo_validator import SLOValidator, SLOResult

    validator = SLOValidator.from_profile("medium")
    result = validator.validate(metrics)

    if not result.passed:
        for violation in result.violations:
            print(f"SLO violation: {violation}")
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from tests.load.profiles import LoadProfile, SLOThresholds, get_profile


class SLOCategory(str, Enum):
    """Categories of SLO metrics."""

    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    AVAILABILITY = "availability"
    CAPACITY = "capacity"


class SLOSeverity(str, Enum):
    """Severity levels for SLO violations."""

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SLOViolation:
    """Represents a single SLO violation."""

    category: SLOCategory
    metric_name: str
    threshold: float
    actual_value: float
    severity: SLOSeverity
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "metric_name": self.metric_name,
            "threshold": self.threshold,
            "actual_value": self.actual_value,
            "severity": self.severity.value,
            "message": self.message,
        }


@dataclass
class SLOResult:
    """Result of SLO validation."""

    passed: bool
    violations: list[SLOViolation] = field(default_factory=list)
    warnings: list[SLOViolation] = field(default_factory=list)
    metrics_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def critical_violations(self) -> list[SLOViolation]:
        return [v for v in self.violations if v.severity == SLOSeverity.CRITICAL]

    @property
    def error_violations(self) -> list[SLOViolation]:
        return [v for v in self.violations if v.severity == SLOSeverity.ERROR]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": [v.to_dict() for v in self.warnings],
            "metrics_summary": self.metrics_summary,
            "critical_count": len(self.critical_violations),
            "error_count": len(self.error_violations),
            "warning_count": len(self.warnings),
        }

    def format_report(self) -> str:
        """Format a human-readable report."""
        lines = []
        lines.append("=" * 60)
        lines.append("SLO VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Result: {'PASSED' if self.passed else 'FAILED'}")
        lines.append("")

        if self.violations:
            lines.append("VIOLATIONS:")
            for v in self.violations:
                severity_marker = {
                    SLOSeverity.CRITICAL: "[CRITICAL]",
                    SLOSeverity.ERROR: "[ERROR]",
                    SLOSeverity.WARNING: "[WARNING]",
                }.get(v.severity, "")
                lines.append(f"  {severity_marker} {v.message}")
                lines.append(f"      Metric: {v.metric_name}")
                lines.append(f"      Threshold: {v.threshold}, Actual: {v.actual_value}")
            lines.append("")

        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  [WARNING] {w.message}")
            lines.append("")

        if self.metrics_summary:
            lines.append("METRICS SUMMARY:")
            for key, value in self.metrics_summary.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.2f}")
                else:
                    lines.append(f"  {key}: {value}")

        lines.append("=" * 60)
        return "\n".join(lines)


class SLOValidator:
    """Validates metrics against SLO thresholds."""

    def __init__(self, thresholds: SLOThresholds, profile_name: str = "custom"):
        self.thresholds = thresholds
        self.profile_name = profile_name

    @classmethod
    def from_profile(cls, profile_name: str) -> "SLOValidator":
        """Create validator from a named profile."""
        profile = get_profile(profile_name)
        return cls(profile.slo_thresholds, profile_name)

    @classmethod
    def from_thresholds(
        cls,
        http_p95_ms: int = 500,
        http_p99_ms: int = 1000,
        max_error_rate: float = 0.01,
        min_throughput_rps: float = 10.0,
    ) -> "SLOValidator":
        """Create validator with custom thresholds."""
        thresholds = SLOThresholds(
            http_p95_ms=http_p95_ms,
            http_p99_ms=http_p99_ms,
            max_error_rate=max_error_rate,
            min_throughput_rps=min_throughput_rps,
        )
        return cls(thresholds)

    def validate(
        self,
        response_times_ms: list[float],
        total_requests: int,
        failed_requests: int,
        duration_seconds: float,
        additional_metrics: Optional[dict[str, Any]] = None,
    ) -> SLOResult:
        """
        Validate metrics against SLO thresholds.

        Args:
            response_times_ms: List of response times in milliseconds
            total_requests: Total number of requests made
            failed_requests: Number of failed requests
            duration_seconds: Total test duration in seconds
            additional_metrics: Optional additional metrics to validate

        Returns:
            SLOResult with validation outcome
        """
        violations: list[SLOViolation] = []
        warnings: list[SLOViolation] = []
        metrics_summary: dict[str, Any] = {}

        # Calculate metrics
        if response_times_ms:
            sorted_times = sorted(response_times_ms)
            p50_idx = int(len(sorted_times) * 0.50)
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)

            p50_ms = sorted_times[min(p50_idx, len(sorted_times) - 1)]
            p95_ms = sorted_times[min(p95_idx, len(sorted_times) - 1)]
            p99_ms = sorted_times[min(p99_idx, len(sorted_times) - 1)]
            avg_ms = sum(response_times_ms) / len(response_times_ms)

            metrics_summary["response_time_p50_ms"] = p50_ms
            metrics_summary["response_time_p95_ms"] = p95_ms
            metrics_summary["response_time_p99_ms"] = p99_ms
            metrics_summary["response_time_avg_ms"] = avg_ms

            # Validate p95 latency
            if p95_ms > self.thresholds.http_p95_ms:
                severity = (
                    SLOSeverity.CRITICAL
                    if p95_ms > self.thresholds.http_p95_ms * 2
                    else SLOSeverity.ERROR
                )
                violations.append(
                    SLOViolation(
                        category=SLOCategory.LATENCY,
                        metric_name="http_response_time_p95",
                        threshold=self.thresholds.http_p95_ms,
                        actual_value=p95_ms,
                        severity=severity,
                        message=f"p95 response time {p95_ms:.0f}ms exceeds threshold {self.thresholds.http_p95_ms}ms",
                    )
                )

            # Validate p99 latency
            if p99_ms > self.thresholds.http_p99_ms:
                severity = (
                    SLOSeverity.CRITICAL
                    if p99_ms > self.thresholds.http_p99_ms * 2
                    else SLOSeverity.ERROR
                )
                violations.append(
                    SLOViolation(
                        category=SLOCategory.LATENCY,
                        metric_name="http_response_time_p99",
                        threshold=self.thresholds.http_p99_ms,
                        actual_value=p99_ms,
                        severity=severity,
                        message=f"p99 response time {p99_ms:.0f}ms exceeds threshold {self.thresholds.http_p99_ms}ms",
                    )
                )

            # Warning for high p50 (indicates general slowness)
            if p50_ms > self.thresholds.http_p50_ms:
                warnings.append(
                    SLOViolation(
                        category=SLOCategory.LATENCY,
                        metric_name="http_response_time_p50",
                        threshold=self.thresholds.http_p50_ms,
                        actual_value=p50_ms,
                        severity=SLOSeverity.WARNING,
                        message=f"p50 response time {p50_ms:.0f}ms exceeds expected {self.thresholds.http_p50_ms}ms",
                    )
                )

        # Calculate error rate
        if total_requests > 0:
            error_rate = failed_requests / total_requests
            metrics_summary["error_rate"] = error_rate
            metrics_summary["total_requests"] = total_requests
            metrics_summary["failed_requests"] = failed_requests

            if error_rate > self.thresholds.max_error_rate:
                severity = (
                    SLOSeverity.CRITICAL
                    if error_rate > self.thresholds.max_error_rate * 3
                    else SLOSeverity.ERROR
                )
                violations.append(
                    SLOViolation(
                        category=SLOCategory.ERROR_RATE,
                        metric_name="http_error_rate",
                        threshold=self.thresholds.max_error_rate,
                        actual_value=error_rate,
                        severity=severity,
                        message=f"Error rate {error_rate:.2%} exceeds threshold {self.thresholds.max_error_rate:.2%}",
                    )
                )

        # Calculate throughput
        if duration_seconds > 0 and total_requests > 0:
            throughput_rps = total_requests / duration_seconds
            metrics_summary["throughput_rps"] = throughput_rps
            metrics_summary["duration_seconds"] = duration_seconds

            if throughput_rps < self.thresholds.min_throughput_rps:
                violations.append(
                    SLOViolation(
                        category=SLOCategory.THROUGHPUT,
                        metric_name="throughput_rps",
                        threshold=self.thresholds.min_throughput_rps,
                        actual_value=throughput_rps,
                        severity=SLOSeverity.ERROR,
                        message=f"Throughput {throughput_rps:.1f} rps below threshold {self.thresholds.min_throughput_rps} rps",
                    )
                )

        # Validate additional metrics
        if additional_metrics:
            self._validate_additional_metrics(
                additional_metrics, violations, warnings, metrics_summary
            )

        # Determine overall pass/fail
        # Fail if any ERROR or CRITICAL violations
        passed = not any(
            v.severity in (SLOSeverity.ERROR, SLOSeverity.CRITICAL) for v in violations
        )

        return SLOResult(
            passed=passed,
            violations=violations,
            warnings=warnings,
            metrics_summary=metrics_summary,
        )

    def _validate_additional_metrics(
        self,
        metrics: dict[str, Any],
        violations: list[SLOViolation],
        warnings: list[SLOViolation],
        summary: dict[str, Any],
    ) -> None:
        """Validate additional custom metrics."""

        # Debate creation latency
        if "debate_create_p95_ms" in metrics:
            value = metrics["debate_create_p95_ms"]
            summary["debate_create_p95_ms"] = value
            if value > self.thresholds.debate_create_p95_ms:
                violations.append(
                    SLOViolation(
                        category=SLOCategory.LATENCY,
                        metric_name="debate_create_p95_ms",
                        threshold=self.thresholds.debate_create_p95_ms,
                        actual_value=value,
                        severity=SLOSeverity.ERROR,
                        message=f"Debate creation p95 latency {value:.0f}ms exceeds {self.thresholds.debate_create_p95_ms}ms",
                    )
                )

        # Search latency
        if "search_p95_ms" in metrics:
            value = metrics["search_p95_ms"]
            summary["search_p95_ms"] = value
            if value > self.thresholds.search_p95_ms:
                violations.append(
                    SLOViolation(
                        category=SLOCategory.LATENCY,
                        metric_name="search_p95_ms",
                        threshold=self.thresholds.search_p95_ms,
                        actual_value=value,
                        severity=SLOSeverity.ERROR,
                        message=f"Search p95 latency {value:.0f}ms exceeds {self.thresholds.search_p95_ms}ms",
                    )
                )

        # Auth latency
        if "auth_p95_ms" in metrics:
            value = metrics["auth_p95_ms"]
            summary["auth_p95_ms"] = value
            if value > self.thresholds.auth_p95_ms:
                violations.append(
                    SLOViolation(
                        category=SLOCategory.LATENCY,
                        metric_name="auth_p95_ms",
                        threshold=self.thresholds.auth_p95_ms,
                        actual_value=value,
                        severity=SLOSeverity.ERROR,
                        message=f"Auth p95 latency {value:.0f}ms exceeds {self.thresholds.auth_p95_ms}ms",
                    )
                )

        # Timeout rate
        if "timeout_rate" in metrics:
            value = metrics["timeout_rate"]
            summary["timeout_rate"] = value
            if value > self.thresholds.max_timeout_rate:
                violations.append(
                    SLOViolation(
                        category=SLOCategory.AVAILABILITY,
                        metric_name="timeout_rate",
                        threshold=self.thresholds.max_timeout_rate,
                        actual_value=value,
                        severity=SLOSeverity.CRITICAL,
                        message=f"Timeout rate {value:.2%} exceeds {self.thresholds.max_timeout_rate:.2%}",
                    )
                )


def validate_locust_stats(
    stats: dict[str, Any],
    profile_name: str = "light",
) -> SLOResult:
    """
    Validate Locust test statistics against SLO thresholds.

    Args:
        stats: Locust statistics dictionary
        profile_name: Load profile to use for thresholds

    Returns:
        SLOResult with validation outcome
    """
    validator = SLOValidator.from_profile(profile_name)

    # Extract metrics from Locust stats
    total_stats = stats.get("Total", stats.get("total", {}))

    response_times_ms = []
    if "response_times" in total_stats:
        # Locust provides response times as a distribution
        rt = total_stats["response_times"]
        for time_ms, count in rt.items():
            response_times_ms.extend([float(time_ms)] * count)
    elif "avg_response_time" in total_stats:
        # Fallback: create synthetic list from aggregates
        avg = total_stats["avg_response_time"]
        count = total_stats.get("num_requests", 100)
        response_times_ms = [avg] * count

    total_requests = total_stats.get("num_requests", 0)
    failed_requests = total_stats.get("num_failures", 0)

    # Duration might be in different places
    duration = stats.get("duration", stats.get("last_request_timestamp", 60))
    if isinstance(duration, str):
        # Parse duration string like "5m30s"
        duration = 300  # Default fallback

    return validator.validate(
        response_times_ms=response_times_ms,
        total_requests=total_requests,
        failed_requests=failed_requests,
        duration_seconds=duration,
    )


def validate_pytest_metrics(
    metrics_dict: dict[str, Any],
    profile_name: str = "light",
) -> SLOResult:
    """
    Validate pytest load test metrics against SLO thresholds.

    Args:
        metrics_dict: Metrics dictionary from pytest load tests
        profile_name: Load profile to use for thresholds

    Returns:
        SLOResult with validation outcome
    """
    validator = SLOValidator.from_profile(profile_name)

    # Convert response times from seconds to milliseconds if needed
    response_times = metrics_dict.get("response_times", [])
    if response_times and max(response_times) < 100:  # Likely in seconds
        response_times_ms = [t * 1000 for t in response_times]
    else:
        response_times_ms = response_times

    total_requests = metrics_dict.get("total_operations", metrics_dict.get("total_requests", 0))
    failed_requests = metrics_dict.get("api_errors", metrics_dict.get("failed_requests", 0))
    duration = metrics_dict.get("duration_seconds", metrics_dict.get("duration", 60))

    # Additional metrics
    additional = {}
    if "search_latencies" in metrics_dict:
        search_times = metrics_dict["search_latencies"]
        if search_times:
            sorted_times = sorted(search_times)
            p95_idx = int(len(sorted_times) * 0.95)
            p95 = sorted_times[min(p95_idx, len(sorted_times) - 1)]
            if p95 < 100:  # Likely in seconds
                p95 *= 1000
            additional["search_p95_ms"] = p95

    return validator.validate(
        response_times_ms=response_times_ms,
        total_requests=total_requests,
        failed_requests=failed_requests,
        duration_seconds=duration,
        additional_metrics=additional,
    )


# =============================================================================
# CLI Support
# =============================================================================


if __name__ == "__main__":
    import argparse
    import random

    parser = argparse.ArgumentParser(description="SLO Validator")
    parser.add_argument(
        "--profile",
        "-p",
        default="light",
        help="Load profile for thresholds",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo validation with synthetic data",
    )

    args = parser.parse_args()

    if args.demo:
        print(f"Running demo validation with '{args.profile}' profile...")
        print()

        # Generate synthetic metrics
        response_times = [random.gauss(200, 100) for _ in range(1000)]
        response_times = [max(10, t) for t in response_times]  # Ensure positive

        validator = SLOValidator.from_profile(args.profile)
        result = validator.validate(
            response_times_ms=response_times,
            total_requests=1000,
            failed_requests=5,
            duration_seconds=60,
        )

        print(result.format_report())
    else:
        print("Use --demo to run demo validation")
        print("Profiles available: smoke, light, medium, heavy, spike, soak")
