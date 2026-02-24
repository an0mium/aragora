"""
Load testing harness and SLO enforcement for Aragora.

Provides a pure-Python load testing framework and an SLO enforcement system
that validates service metrics against configurable targets.

Usage:
    from aragora.observability.load_test import (
        LoadTestConfig,
        LoadTestRunner,
        SLOEnforcer,
    )

    # Run a load test
    runner = LoadTestRunner()
    config = LoadTestConfig(target_rps=100, duration_seconds=30)
    result = await runner.run("http://localhost:8080/api/v1/health", config)
    print(f"p95: {result.p95_ms:.1f}ms, achieved: {result.rps_achieved:.0f} rps")

    # Enforce SLO targets
    enforcer = SLOEnforcer()
    for violation in enforcer.get_violations():
        print(f"SLO violated: {violation.target_name} - {violation.message}")

Environment Variables:
    SLO_DEBATE_P95_MS: Override debate p95 target (default: 5000)
    SLO_API_P95_MS: Override API p95 target (default: 200)
    SLO_ERROR_RATE_TARGET: Override error rate target (default: 0.01)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


# =============================================================================
# Load Test Configuration and Results
# =============================================================================


@dataclass
class LoadTestConfig:
    """Configuration for a load test run.

    Attributes:
        target_rps: Target requests per second to sustain.
        duration_seconds: Total duration to run the test.
        concurrency: Maximum number of concurrent in-flight requests.
        warmup_seconds: Warmup period before collecting metrics.
    """

    target_rps: int = 100
    duration_seconds: int = 30
    concurrency: int = 10
    warmup_seconds: float = 2.0


@dataclass
class LoadTestResult:
    """Aggregated results from a load test run.

    Attributes:
        total_requests: Total number of requests sent.
        successful: Number of successful (non-error) requests.
        failed: Number of failed requests.
        p50_ms: 50th percentile latency in milliseconds.
        p95_ms: 95th percentile latency in milliseconds.
        p99_ms: 99th percentile latency in milliseconds.
        max_ms: Maximum observed latency in milliseconds.
        rps_achieved: Actual requests per second achieved.
        errors_by_type: Count of errors grouped by type string.
    """

    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0
    rps_achieved: float = 0.0
    errors_by_type: dict[str, int] = field(default_factory=dict)


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Calculate a percentile from a pre-sorted list.

    Args:
        sorted_values: Sorted list of float values.
        pct: Percentile to compute (0-100).

    Returns:
        The percentile value, or 0.0 if list is empty.
    """
    if not sorted_values:
        return 0.0
    idx = int(len(sorted_values) * pct / 100.0)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


# =============================================================================
# Load Test Runner
# =============================================================================


# Type alias for request functions used by the runner
RequestFunc = Callable[[], Awaitable[bool]]


class LoadTestRunner:
    """Pure-Python async load test runner.

    Uses ``asyncio.Semaphore`` for concurrency control and collects
    latency samples to compute percentile statistics.

    Example::

        runner = LoadTestRunner()
        config = LoadTestConfig(target_rps=50, duration_seconds=10)
        result = await runner.run("http://localhost:8080/health", config)
    """

    def __init__(self, http_client: Any | None = None) -> None:
        """Initialize the load test runner.

        Args:
            http_client: Optional async HTTP client. If not provided, a
                lightweight stub is used (suitable for unit testing).
        """
        self._http_client = http_client

    async def _make_request(self, endpoint: str) -> bool:
        """Send a single HTTP request to the endpoint.

        Args:
            endpoint: URL to request.

        Returns:
            True if the request was successful, False otherwise.
        """
        if self._http_client is not None:
            try:
                response = await self._http_client.get(endpoint)
                status = getattr(response, "status", getattr(response, "status_code", 500))
                return 200 <= status < 400
            except Exception as exc:  # noqa: BLE001 - load test must collect all errors
                logger.debug("Request to %s failed: %s", endpoint, exc)
                return False

        # Fallback: attempt aiohttp if available
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    return 200 <= resp.status < 400
        except ImportError:
            logger.warning("No HTTP client available; install aiohttp or provide http_client")
            return False
        except Exception as exc:  # noqa: BLE001
            logger.debug("Request to %s failed: %s", endpoint, exc)
            return False

    async def run(
        self,
        endpoint: str,
        config: LoadTestConfig | None = None,
        request_func: RequestFunc | None = None,
    ) -> LoadTestResult:
        """Run a load test against an endpoint.

        Args:
            endpoint: URL to load test.
            config: Load test configuration. Uses defaults if not provided.
            request_func: Optional custom request function. If provided, it is
                called instead of the built-in HTTP request logic. Must return
                True for success, False for failure.

        Returns:
            LoadTestResult with aggregated statistics.
        """
        if config is None:
            config = LoadTestConfig()

        semaphore = asyncio.Semaphore(config.concurrency)
        latencies: list[float] = []
        errors: dict[str, int] = defaultdict(int)
        successful = 0
        failed = 0
        lock = asyncio.Lock()

        warmup_end = time.monotonic() + config.warmup_seconds
        test_end = warmup_end + config.duration_seconds

        async def _send_one(is_warmup: bool) -> None:
            nonlocal successful, failed
            async with semaphore:
                start = time.monotonic()
                try:
                    if request_func is not None:
                        ok = await request_func()
                    else:
                        ok = await self._make_request(endpoint)
                except Exception as exc:  # noqa: BLE001
                    ok = False
                    error_type = type(exc).__name__
                    async with lock:
                        errors[error_type] += 1

                elapsed_ms = (time.monotonic() - start) * 1000.0

                if is_warmup:
                    return

                async with lock:
                    if ok:
                        successful += 1
                    else:
                        failed += 1
                        if "request_error" not in errors and not ok:
                            errors["request_error"] += 1
                    latencies.append(elapsed_ms)

        # Calculate interval between requests
        interval = 1.0 / config.target_rps if config.target_rps > 0 else 0.01
        tasks: list[asyncio.Task] = []

        # Warmup phase
        now = time.monotonic()
        while now < warmup_end:
            task = asyncio.create_task(_send_one(is_warmup=True))
            tasks.append(task)
            await asyncio.sleep(interval)
            now = time.monotonic()

        # Measurement phase
        measure_start = time.monotonic()
        while time.monotonic() < test_end:
            task = asyncio.create_task(_send_one(is_warmup=False))
            tasks.append(task)
            await asyncio.sleep(interval)

        # Wait for all in-flight tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        measure_duration = time.monotonic() - measure_start

        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        total = successful + failed

        return LoadTestResult(
            total_requests=total,
            successful=successful,
            failed=failed,
            p50_ms=_percentile(sorted_latencies, 50),
            p95_ms=_percentile(sorted_latencies, 95),
            p99_ms=_percentile(sorted_latencies, 99),
            max_ms=sorted_latencies[-1] if sorted_latencies else 0.0,
            rps_achieved=total / measure_duration if measure_duration > 0 else 0.0,
            errors_by_type=dict(errors),
        )

    async def run_debate_load_test(
        self,
        config: LoadTestConfig | None = None,
        debate_endpoint: str = "http://localhost:8080/api/v1/debates",
    ) -> LoadTestResult:
        """Run a load test specialized for debate creation.

        Sends POST requests to the debate creation endpoint with a
        minimal payload.

        Args:
            config: Load test configuration.
            debate_endpoint: URL of the debate creation endpoint.

        Returns:
            LoadTestResult for debate creation load.
        """

        async def _create_debate() -> bool:
            if self._http_client is not None:
                try:
                    payload = {"task": "load-test-debate", "rounds": 1}
                    response = await self._http_client.post(debate_endpoint, json=payload)
                    status = getattr(response, "status", getattr(response, "status_code", 500))
                    return 200 <= status < 400
                except Exception:  # noqa: BLE001
                    return False

            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    payload = {"task": "load-test-debate", "rounds": 1}
                    async with session.post(
                        debate_endpoint,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        return 200 <= resp.status < 400
            except ImportError:
                logger.warning("No HTTP client available for debate load test")
                return False
            except Exception:  # noqa: BLE001
                return False

        return await self.run(
            endpoint=debate_endpoint,
            config=config,
            request_func=_create_debate,
        )


# =============================================================================
# SLO Enforcement
# =============================================================================


@dataclass
class SLOTargetDef:
    """Definition of an SLO enforcement target.

    Attributes:
        name: Human-readable name for the SLO.
        metric: Metric identifier (e.g. ``debate_p95``, ``api_p95``).
        threshold: Numeric threshold value.
        window_seconds: Evaluation window in seconds.
    """

    name: str
    metric: str
    threshold: float
    window_seconds: int = 3600


@dataclass
class SLOCheckResult:
    """Result of checking a single SLO target.

    Attributes:
        target_name: Name of the SLO target that was checked.
        metric: Metric identifier.
        threshold: Configured threshold.
        current_value: Observed metric value.
        compliant: Whether the SLO is currently met.
        checked_at: Timestamp of the check.
    """

    target_name: str
    metric: str
    threshold: float
    current_value: float
    compliant: bool
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SLOViolation:
    """Record of an SLO violation.

    Attributes:
        target_name: Name of the SLO target that was violated.
        metric: Metric identifier.
        threshold: Configured threshold.
        observed_value: Value that violated the SLO.
        message: Human-readable description of the violation.
        occurred_at: Timestamp when the violation was detected.
    """

    target_name: str
    metric: str
    threshold: float
    observed_value: float
    message: str
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Default SLO targets for enforcement
_DEFAULT_SLO_TARGETS: list[SLOTargetDef] = [
    SLOTargetDef(
        name="debate_p95_latency",
        metric="debate_p95",
        threshold=float(os.getenv("SLO_DEBATE_P95_MS", "5000")),
        window_seconds=3600,
    ),
    SLOTargetDef(
        name="api_p95_latency",
        metric="api_p95",
        threshold=float(os.getenv("SLO_API_P95_MS", "200")),
        window_seconds=3600,
    ),
    SLOTargetDef(
        name="error_rate",
        metric="error_rate",
        threshold=float(os.getenv("SLO_ERROR_RATE_TARGET", "0.01")),
        window_seconds=3600,
    ),
]


class SLOEnforcer:
    """Enforces SLO targets against observed metrics.

    Maintains a registry of SLO targets, records metric observations,
    and tracks violations over configurable time windows.

    Example::

        enforcer = SLOEnforcer()
        enforcer.record_metric("api_p95", 180.0)  # 180ms, within target
        enforcer.record_metric("api_p95", 250.0)  # 250ms, violation!

        violations = enforcer.get_violations()
        for v in violations:
            print(f"{v.target_name}: {v.message}")
    """

    def __init__(self) -> None:
        """Initialize with default SLO targets."""
        self._targets: list[SLOTargetDef] = list(_DEFAULT_SLO_TARGETS)
        self._violations: list[SLOViolation] = []
        self._observations: dict[str, list[tuple[datetime, float]]] = defaultdict(list)

    def register_target(self, target: SLOTargetDef) -> None:
        """Register a new SLO target for enforcement.

        Args:
            target: SLO target definition to register.
        """
        # Replace existing target with the same name if present
        self._targets = [t for t in self._targets if t.name != target.name]
        self._targets.append(target)
        logger.info(
            "Registered SLO target: %s (metric=%s, threshold=%s)",
            target.name,
            target.metric,
            target.threshold,
        )

    def get_targets(self) -> list[SLOTargetDef]:
        """Return a copy of all registered SLO targets.

        Returns:
            List of SLOTargetDef instances.
        """
        return list(self._targets)

    def record_metric(self, metric: str, value: float) -> list[SLOViolation]:
        """Record a metric observation and check against targets.

        Args:
            metric: Metric identifier (must match a registered target's metric).
            value: Observed value.

        Returns:
            List of any new violations triggered by this observation.
        """
        now = datetime.now(timezone.utc)
        self._observations[metric].append((now, value))

        new_violations: list[SLOViolation] = []

        for target in self._targets:
            if target.metric != metric:
                continue

            violated = False
            if target.metric == "error_rate":
                # error_rate: value must be <= threshold
                violated = value > target.threshold
            else:
                # Latency metrics: value must be <= threshold (in ms)
                violated = value > target.threshold

            if violated:
                violation = SLOViolation(
                    target_name=target.name,
                    metric=target.metric,
                    threshold=target.threshold,
                    observed_value=value,
                    message=(
                        f"{target.name}: observed {value:.2f} exceeds "
                        f"threshold {target.threshold:.2f}"
                    ),
                    occurred_at=now,
                )
                self._violations.append(violation)
                new_violations.append(violation)
                logger.warning("SLO violation: %s", violation.message)

        return new_violations

    def check_slo(self, target: SLOTargetDef) -> SLOCheckResult:
        """Check whether an SLO target is currently met.

        Uses the most recent observation for the target's metric within
        the configured window. If no observations exist, the SLO is
        considered compliant (no data = no violation).

        Args:
            target: SLO target to check.

        Returns:
            SLOCheckResult with compliance status.
        """
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=target.window_seconds)

        # Get observations within window
        observations = [
            (ts, val) for ts, val in self._observations.get(target.metric, []) if ts >= window_start
        ]

        if not observations:
            return SLOCheckResult(
                target_name=target.name,
                metric=target.metric,
                threshold=target.threshold,
                current_value=0.0,
                compliant=True,
                checked_at=now,
            )

        # Use the latest observation
        _, latest_value = observations[-1]
        compliant = latest_value <= target.threshold

        return SLOCheckResult(
            target_name=target.name,
            metric=target.metric,
            threshold=target.threshold,
            current_value=latest_value,
            compliant=compliant,
            checked_at=now,
        )

    def check_all(self) -> list[SLOCheckResult]:
        """Check all registered SLO targets.

        Returns:
            List of SLOCheckResult for each registered target.
        """
        return [self.check_slo(target) for target in self._targets]

    def get_violations(self, period: str = "24h") -> list[SLOViolation]:
        """Get recent SLO violations within a time period.

        Args:
            period: Time period string. Supported formats: ``"1h"``, ``"24h"``,
                ``"7d"``. Defaults to ``"24h"``.

        Returns:
            List of SLOViolation records within the period.
        """
        now = datetime.now(timezone.utc)
        delta = _parse_period(period)
        cutoff = now - delta

        return [v for v in self._violations if v.occurred_at >= cutoff]

    def clear_violations(self) -> None:
        """Clear all recorded violations."""
        self._violations.clear()

    def prune_observations(self, max_age_seconds: int = 7200) -> int:
        """Remove observations older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age of observations to retain.

        Returns:
            Number of observations pruned.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
        pruned = 0
        for metric in self._observations:
            before = len(self._observations[metric])
            self._observations[metric] = [
                (ts, val) for ts, val in self._observations[metric] if ts >= cutoff
            ]
            pruned += before - len(self._observations[metric])
        return pruned


def _parse_period(period: str) -> timedelta:
    """Parse a period string into a timedelta.

    Supports ``h`` (hours) and ``d`` (days) suffixes.

    Args:
        period: Period string (e.g. ``"24h"``, ``"7d"``).

    Returns:
        Corresponding timedelta.

    Raises:
        ValueError: If the period string is not recognized.
    """
    period = period.strip().lower()
    if period.endswith("h"):
        return timedelta(hours=int(period[:-1]))
    elif period.endswith("d"):
        return timedelta(days=int(period[:-1]))
    else:
        raise ValueError(f"Unsupported period format: {period!r}. Use e.g. '24h' or '7d'.")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Load test
    "LoadTestConfig",
    "LoadTestResult",
    "LoadTestRunner",
    # SLO enforcement
    "SLOTargetDef",
    "SLOCheckResult",
    "SLOViolation",
    "SLOEnforcer",
]
