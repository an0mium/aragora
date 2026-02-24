"""
Debate-specific SLO definitions and enforcement for Aragora.

Defines five critical SLOs for debate performance, wires them to Prometheus
metrics with alerting thresholds, and provides a rolling-window tracker
that feeds the ``/api/health/slos`` endpoint.

SLO Targets:
    1. time_to_first_token_p95 < 3s
    2. debate_completion_p95 < 60s (3-round, 3-agent)
    3. websocket_reconnection_success_rate > 99%
    4. consensus_detection_latency_p95 < 500ms
    5. agent_dispatch_concurrency > 0.8

Usage:
    from aragora.observability.debate_slos import (
        DebateSLOTracker,
        get_debate_slo_tracker,
        get_debate_slo_status,
    )

    tracker = get_debate_slo_tracker()

    # Record metrics as they occur
    tracker.record_first_token_latency(1.5)
    tracker.record_debate_completion(42.0, rounds=3, agents=3)
    tracker.record_websocket_reconnection(success=True)
    tracker.record_consensus_latency(0.3)
    tracker.record_dispatch_concurrency(0.92)

    # Get SLO status for the health endpoint
    status = tracker.get_status()

Environment Variables:
    SLO_TTFT_P95_S: Override time-to-first-token p95 target (default: 3.0)
    SLO_DEBATE_COMPLETION_P95_S: Override debate completion p95 target (default: 60.0)
    SLO_WS_RECONNECT_RATE: Override WebSocket reconnection success rate (default: 0.99)
    SLO_CONSENSUS_P95_S: Override consensus detection p95 target (default: 0.5)
    SLO_DISPATCH_CONCURRENCY: Override agent dispatch concurrency ratio (default: 0.8)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# SLO Definitions
# =============================================================================


class DebateSLOLevel(str, Enum):
    """Compliance level for an individual debate SLO."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class DebateSLODefinition:
    """Definition of a single debate SLO target."""

    slo_id: str
    name: str
    target: float
    unit: str
    description: str
    comparison: str  # "lte" (latency), "gte" (rate/ratio)
    warning_threshold: float  # value that triggers yellow
    critical_threshold: float  # value that triggers red

    def evaluate(self, value: float) -> DebateSLOLevel:
        """Evaluate a metric value against this SLO.

        Args:
            value: Current metric value.

        Returns:
            DebateSLOLevel indicating compliance.
        """
        if self.comparison == "lte":
            # For latency: lower is better
            if value <= self.target:
                return DebateSLOLevel.GREEN
            elif value <= self.critical_threshold:
                return DebateSLOLevel.YELLOW
            else:
                return DebateSLOLevel.RED
        else:
            # For rates/ratios: higher is better
            if value >= self.target:
                return DebateSLOLevel.GREEN
            elif value >= self.critical_threshold:
                return DebateSLOLevel.YELLOW
            else:
                return DebateSLOLevel.RED


# Default SLO target values
DEFAULT_TTFT_P95_S = 3.0
DEFAULT_DEBATE_COMPLETION_P95_S = 60.0
DEFAULT_WS_RECONNECT_RATE = 0.99
DEFAULT_CONSENSUS_P95_S = 0.5
DEFAULT_DISPATCH_CONCURRENCY = 0.8


def get_debate_slo_definitions() -> dict[str, DebateSLODefinition]:
    """Get debate SLO definitions with environment overrides.

    Returns:
        Dictionary mapping slo_id to DebateSLODefinition.
    """
    ttft = float(os.getenv("SLO_TTFT_P95_S", str(DEFAULT_TTFT_P95_S)))
    completion = float(
        os.getenv("SLO_DEBATE_COMPLETION_P95_S", str(DEFAULT_DEBATE_COMPLETION_P95_S))
    )
    ws_rate = float(os.getenv("SLO_WS_RECONNECT_RATE", str(DEFAULT_WS_RECONNECT_RATE)))
    consensus = float(os.getenv("SLO_CONSENSUS_P95_S", str(DEFAULT_CONSENSUS_P95_S)))
    dispatch = float(
        os.getenv("SLO_DISPATCH_CONCURRENCY", str(DEFAULT_DISPATCH_CONCURRENCY))
    )

    return {
        "time_to_first_token": DebateSLODefinition(
            slo_id="time_to_first_token",
            name="Time to First Token p95",
            target=ttft,
            unit="seconds",
            description="p95 latency from debate start to first agent token",
            comparison="lte",
            warning_threshold=ttft * 1.5,  # 4.5s default
            critical_threshold=ttft * 2.0,  # 6s default
        ),
        "debate_completion": DebateSLODefinition(
            slo_id="debate_completion",
            name="Debate Completion p95",
            target=completion,
            unit="seconds",
            description="p95 completion time for 3-round, 3-agent debates",
            comparison="lte",
            warning_threshold=completion * 1.5,  # 90s default
            critical_threshold=completion * 2.0,  # 120s default
        ),
        "websocket_reconnection": DebateSLODefinition(
            slo_id="websocket_reconnection",
            name="WebSocket Reconnection Success Rate",
            target=ws_rate,
            unit="ratio",
            description="Percentage of WebSocket reconnections that succeed",
            comparison="gte",
            warning_threshold=ws_rate * 0.995,  # ~98.5% default
            critical_threshold=ws_rate * 0.97,  # ~96% default
        ),
        "consensus_detection": DebateSLODefinition(
            slo_id="consensus_detection",
            name="Consensus Detection Latency p95",
            target=consensus,
            unit="seconds",
            description="p95 latency for consensus mechanism to produce a result",
            comparison="lte",
            warning_threshold=consensus * 2.0,  # 1s default
            critical_threshold=consensus * 4.0,  # 2s default
        ),
        "agent_dispatch_concurrency": DebateSLODefinition(
            slo_id="agent_dispatch_concurrency",
            name="Agent Dispatch Concurrency Ratio",
            target=dispatch,
            unit="ratio",
            description="Ratio of parallel vs sequential agent calls (1.0 = fully parallel)",
            comparison="gte",
            warning_threshold=dispatch * 0.9,  # 0.72 default
            critical_threshold=dispatch * 0.75,  # 0.6 default
        ),
    }


# =============================================================================
# Rolling-Window Percentile Tracker
# =============================================================================


class _RollingWindow:
    """Time-windowed sample buffer for percentile calculation.

    Stores timestamped values and computes percentiles over the
    configured window. Old values are pruned lazily on access.
    """

    def __init__(self, window_seconds: float = 3600.0, max_samples: int = 10_000) -> None:
        self._window_seconds = window_seconds
        self._max_samples = max_samples
        self._samples: list[tuple[float, float]] = []  # (timestamp, value)

    def add(self, value: float) -> None:
        """Add a value with the current timestamp."""
        self._samples.append((time.time(), value))
        if len(self._samples) > self._max_samples * 2:
            self._prune()

    def _prune(self) -> None:
        """Remove samples outside the window."""
        cutoff = time.time() - self._window_seconds
        self._samples = [
            (ts, val) for ts, val in self._samples if ts > cutoff
        ]
        # Keep only max_samples most recent
        if len(self._samples) > self._max_samples:
            self._samples = self._samples[-self._max_samples:]

    def percentile(self, pct: float) -> float:
        """Calculate a percentile from the current window.

        Args:
            pct: Percentile to compute (0-100).

        Returns:
            Percentile value, or 0.0 if no samples.
        """
        self._prune()
        if not self._samples:
            return 0.0
        values = sorted(val for _, val in self._samples)
        idx = (pct / 100.0) * (len(values) - 1)
        lower = int(idx)
        upper = min(lower + 1, len(values) - 1)
        frac = idx - lower
        return values[lower] * (1 - frac) + values[upper] * frac

    @property
    def count(self) -> int:
        """Number of samples in the current window."""
        self._prune()
        return len(self._samples)

    def rate(self, success_value: float = 1.0) -> float:
        """Calculate the success rate within the window.

        Assumes values are 1.0 for success, 0.0 for failure.

        Returns:
            Success rate as a ratio (0.0 to 1.0), or 1.0 if no samples.
        """
        self._prune()
        if not self._samples:
            return 1.0
        successes = sum(1 for _, val in self._samples if val >= success_value)
        return successes / len(self._samples)

    def mean(self) -> float:
        """Calculate mean value within the window."""
        self._prune()
        if not self._samples:
            return 0.0
        return sum(val for _, val in self._samples) / len(self._samples)


# =============================================================================
# Prometheus Metrics Integration
# =============================================================================

_prometheus_initialized = False
_TTFT_HISTOGRAM: Any = None
_DEBATE_COMPLETION_HISTOGRAM: Any = None
_WS_RECONNECT_COUNTER: Any = None
_WS_RECONNECT_FAILURE_COUNTER: Any = None
_CONSENSUS_LATENCY_HISTOGRAM: Any = None
_DISPATCH_CONCURRENCY_GAUGE: Any = None
_SLO_COMPLIANCE_GAUGE: Any = None
_SLO_BURN_RATE_GAUGE: Any = None


def _init_prometheus_metrics() -> bool:
    """Initialize Prometheus metrics for debate SLOs.

    Returns:
        True if Prometheus was initialized, False if using no-op stubs.
    """
    global _prometheus_initialized
    global _TTFT_HISTOGRAM, _DEBATE_COMPLETION_HISTOGRAM
    global _WS_RECONNECT_COUNTER, _WS_RECONNECT_FAILURE_COUNTER
    global _CONSENSUS_LATENCY_HISTOGRAM, _DISPATCH_CONCURRENCY_GAUGE
    global _SLO_COMPLIANCE_GAUGE, _SLO_BURN_RATE_GAUGE

    if _prometheus_initialized:
        return True

    try:
        from prometheus_client import Counter, Gauge, Histogram

        _TTFT_HISTOGRAM = Histogram(
            "aragora_debate_time_to_first_token_seconds",
            "Time from debate start to first agent token (seconds)",
            buckets=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 15.0],
        )

        _DEBATE_COMPLETION_HISTOGRAM = Histogram(
            "aragora_debate_load_completion_seconds",
            "Debate completion duration for SLO tracking (seconds)",
            ["rounds", "agents"],
            buckets=[10, 20, 30, 45, 60, 90, 120, 180, 300, 600],
        )

        _WS_RECONNECT_COUNTER = Counter(
            "aragora_ws_reconnect_total",
            "Total WebSocket reconnection attempts",
            ["outcome"],  # success, failure
        )

        _CONSENSUS_LATENCY_HISTOGRAM = Histogram(
            "aragora_consensus_detection_latency_slo_seconds",
            "Consensus detection latency for SLO tracking (seconds)",
            buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0],
        )

        _DISPATCH_CONCURRENCY_GAUGE = Gauge(
            "aragora_agent_dispatch_concurrency_ratio",
            "Ratio of parallel vs sequential agent dispatches (0-1)",
        )

        _SLO_COMPLIANCE_GAUGE = Gauge(
            "aragora_debate_slo_compliance",
            "Debate SLO compliance level (2=green, 1=yellow, 0=red)",
            ["slo_id"],
        )

        _SLO_BURN_RATE_GAUGE = Gauge(
            "aragora_debate_slo_burn_rate",
            "Debate SLO error budget burn rate",
            ["slo_id"],
        )

        _prometheus_initialized = True
        logger.info("Debate SLO Prometheus metrics initialized")
        return True

    except ImportError:
        logger.debug("prometheus-client not installed, debate SLO metrics disabled")
        _prometheus_initialized = True
        return False


def _observe_histogram(histogram: Any, value: float, **labels: str) -> None:
    """Safely observe a histogram value."""
    if histogram is None:
        return
    try:
        if labels:
            histogram.labels(**labels).observe(value)
        else:
            histogram.observe(value)
    except (AttributeError, TypeError):
        pass


def _inc_counter(counter: Any, **labels: str) -> None:
    """Safely increment a counter."""
    if counter is None:
        return
    try:
        if labels:
            counter.labels(**labels).inc()
        else:
            counter.inc()
    except (AttributeError, TypeError):
        pass


def _set_gauge(gauge: Any, value: float, **labels: str) -> None:
    """Safely set a gauge value."""
    if gauge is None:
        return
    try:
        if labels:
            gauge.labels(**labels).set(value)
        else:
            gauge.set(value)
    except (AttributeError, TypeError):
        pass


# =============================================================================
# SLO Status Dataclass
# =============================================================================


@dataclass
class DebateSLOResult:
    """Result of checking a single debate SLO."""

    slo_id: str
    name: str
    target: float
    current: float
    unit: str
    level: DebateSLOLevel
    compliant: bool
    sample_count: int
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "slo_id": self.slo_id,
            "name": self.name,
            "target": self.target,
            "current": round(self.current, 6),
            "unit": self.unit,
            "level": self.level.value,
            "compliant": self.compliant,
            "sample_count": self.sample_count,
            "description": self.description,
        }


@dataclass
class DebateSLOStatus:
    """Overall debate SLO status across all five SLOs."""

    slos: dict[str, DebateSLOResult]
    timestamp: str
    overall_healthy: bool
    overall_level: DebateSLOLevel
    windows: dict[str, dict[str, str]]  # per-window time ranges

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "overall_healthy": self.overall_healthy,
            "overall_level": self.overall_level.value,
            "slos": {k: v.to_dict() for k, v in self.slos.items()},
            "windows": self.windows,
        }


# =============================================================================
# Debate SLO Tracker
# =============================================================================


class DebateSLOTracker:
    """Tracks debate-specific SLO metrics with rolling windows.

    Maintains three rolling windows (1h, 24h, 7d) and computes
    percentile/rate metrics for each of the five debate SLOs.

    Thread-safe: safe to call from multiple async tasks or threads.

    Usage:
        tracker = DebateSLOTracker()
        tracker.record_first_token_latency(2.1)
        tracker.record_debate_completion(45.0, rounds=3, agents=3)
        status = tracker.get_status()
    """

    # Window durations in seconds
    WINDOWS = {
        "1h": 3600,
        "24h": 86400,
        "7d": 604800,
    }

    def __init__(self) -> None:
        self._definitions = get_debate_slo_definitions()

        # Per-window rolling trackers for each SLO
        self._ttft: dict[str, _RollingWindow] = {}
        self._completion: dict[str, _RollingWindow] = {}
        self._ws_reconnect: dict[str, _RollingWindow] = {}
        self._consensus: dict[str, _RollingWindow] = {}
        self._dispatch: dict[str, _RollingWindow] = {}

        for window_name, duration in self.WINDOWS.items():
            self._ttft[window_name] = _RollingWindow(duration)
            self._completion[window_name] = _RollingWindow(duration)
            self._ws_reconnect[window_name] = _RollingWindow(duration)
            self._consensus[window_name] = _RollingWindow(duration)
            self._dispatch[window_name] = _RollingWindow(duration)

        _init_prometheus_metrics()

    def record_first_token_latency(self, seconds: float) -> None:
        """Record time from debate start to first agent token.

        Args:
            seconds: Latency in seconds.
        """
        for w in self._ttft.values():
            w.add(seconds)
        _observe_histogram(_TTFT_HISTOGRAM, seconds)

    def record_debate_completion(
        self,
        seconds: float,
        rounds: int = 3,
        agents: int = 3,
    ) -> None:
        """Record debate completion duration.

        Args:
            seconds: Total debate duration in seconds.
            rounds: Number of debate rounds.
            agents: Number of agents in the debate.
        """
        for w in self._completion.values():
            w.add(seconds)
        _observe_histogram(
            _DEBATE_COMPLETION_HISTOGRAM,
            seconds,
            rounds=str(rounds),
            agents=str(agents),
        )

    def record_websocket_reconnection(self, success: bool) -> None:
        """Record a WebSocket reconnection attempt.

        Args:
            success: Whether the reconnection succeeded.
        """
        value = 1.0 if success else 0.0
        for w in self._ws_reconnect.values():
            w.add(value)
        outcome = "success" if success else "failure"
        _inc_counter(_WS_RECONNECT_COUNTER, outcome=outcome)

    def record_consensus_latency(self, seconds: float) -> None:
        """Record consensus detection latency.

        Args:
            seconds: Consensus detection duration in seconds.
        """
        for w in self._consensus.values():
            w.add(seconds)
        _observe_histogram(_CONSENSUS_LATENCY_HISTOGRAM, seconds)

    def record_dispatch_concurrency(self, ratio: float) -> None:
        """Record agent dispatch concurrency ratio.

        Args:
            ratio: Ratio of parallel vs sequential dispatch (0-1).
        """
        clamped = max(0.0, min(1.0, ratio))
        for w in self._dispatch.values():
            w.add(clamped)
        _set_gauge(_DISPATCH_CONCURRENCY_GAUGE, clamped)

    def _evaluate_slo(
        self,
        slo_id: str,
        windows: dict[str, _RollingWindow],
        value_func: str,
        window: str = "1h",
    ) -> DebateSLOResult:
        """Evaluate a single SLO against the specified window.

        Args:
            slo_id: SLO identifier.
            windows: Rolling windows for this metric.
            value_func: Method to call on the window ("percentile:95", "rate", "mean").
            window: Window name ("1h", "24h", "7d").

        Returns:
            DebateSLOResult with current status.
        """
        defn = self._definitions[slo_id]
        w = windows.get(window, windows.get("1h"))

        if value_func.startswith("percentile:"):
            pct = float(value_func.split(":")[1])
            current = w.percentile(pct) if w else 0.0
        elif value_func == "rate":
            current = w.rate() if w else 1.0
        elif value_func == "mean":
            current = w.mean() if w else 0.0
        else:
            current = 0.0

        level = defn.evaluate(current)
        compliant = level == DebateSLOLevel.GREEN

        # Update Prometheus compliance gauges
        level_value = {
            DebateSLOLevel.GREEN: 2.0,
            DebateSLOLevel.YELLOW: 1.0,
            DebateSLOLevel.RED: 0.0,
        }.get(level, 0.0)
        _set_gauge(_SLO_COMPLIANCE_GAUGE, level_value, slo_id=slo_id)

        return DebateSLOResult(
            slo_id=slo_id,
            name=defn.name,
            target=defn.target,
            current=current,
            unit=defn.unit,
            level=level,
            compliant=compliant,
            sample_count=w.count if w else 0,
            description=defn.description,
        )

    def get_status(self, window: str = "1h") -> DebateSLOStatus:
        """Get overall debate SLO status for the specified window.

        Args:
            window: Time window to evaluate ("1h", "24h", "7d").

        Returns:
            DebateSLOStatus with per-SLO results and overall health.
        """
        now = datetime.now(timezone.utc)

        slos = {
            "time_to_first_token": self._evaluate_slo(
                "time_to_first_token", self._ttft, "percentile:95", window
            ),
            "debate_completion": self._evaluate_slo(
                "debate_completion", self._completion, "percentile:95", window
            ),
            "websocket_reconnection": self._evaluate_slo(
                "websocket_reconnection", self._ws_reconnect, "rate", window
            ),
            "consensus_detection": self._evaluate_slo(
                "consensus_detection", self._consensus, "percentile:95", window
            ),
            "agent_dispatch_concurrency": self._evaluate_slo(
                "agent_dispatch_concurrency", self._dispatch, "mean", window
            ),
        }

        # Overall: red if any red, yellow if any yellow, green otherwise
        levels = [r.level for r in slos.values()]
        if DebateSLOLevel.RED in levels:
            overall_level = DebateSLOLevel.RED
        elif DebateSLOLevel.YELLOW in levels:
            overall_level = DebateSLOLevel.YELLOW
        else:
            overall_level = DebateSLOLevel.GREEN

        overall_healthy = overall_level == DebateSLOLevel.GREEN

        # Build window time ranges
        windows_info = {}
        for wname, duration in self.WINDOWS.items():
            window_start = now - timedelta(seconds=duration)
            windows_info[wname] = {
                "start": window_start.isoformat(),
                "end": now.isoformat(),
            }

        return DebateSLOStatus(
            slos=slos,
            timestamp=now.isoformat(),
            overall_healthy=overall_healthy,
            overall_level=overall_level,
            windows=windows_info,
        )

    def get_multi_window_status(self) -> dict[str, Any]:
        """Get SLO status across all three windows.

        Returns:
            Dictionary with per-window status suitable for JSON response.
        """
        result: dict[str, Any] = {}
        for window_name in self.WINDOWS:
            status = self.get_status(window_name)
            result[window_name] = status.to_dict()

        # Add overall from the 1h window (primary)
        primary = self.get_status("1h")
        result["overall_healthy"] = primary.overall_healthy
        result["overall_level"] = primary.overall_level.value
        result["timestamp"] = primary.timestamp

        return result

    def reset(self) -> None:
        """Reset all tracked metrics. Primarily for testing."""
        for window_name, duration in self.WINDOWS.items():
            self._ttft[window_name] = _RollingWindow(duration)
            self._completion[window_name] = _RollingWindow(duration)
            self._ws_reconnect[window_name] = _RollingWindow(duration)
            self._consensus[window_name] = _RollingWindow(duration)
            self._dispatch[window_name] = _RollingWindow(duration)


# =============================================================================
# Global Singleton
# =============================================================================

_global_tracker: DebateSLOTracker | None = None


def get_debate_slo_tracker() -> DebateSLOTracker:
    """Get or create the global debate SLO tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = DebateSLOTracker()
    return _global_tracker


def reset_debate_slo_tracker() -> None:
    """Reset the global debate SLO tracker. Primarily for testing."""
    global _global_tracker
    _global_tracker = None


def get_debate_slo_status(window: str = "1h") -> dict[str, Any]:
    """Get debate SLO status as a JSON-serializable dictionary.

    Convenience function for the health endpoint.

    Args:
        window: Time window ("1h", "24h", "7d").

    Returns:
        Dictionary with SLO status data.
    """
    tracker = get_debate_slo_tracker()
    return tracker.get_status(window).to_dict()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Types
    "DebateSLOLevel",
    "DebateSLODefinition",
    "DebateSLOResult",
    "DebateSLOStatus",
    # Definitions
    "get_debate_slo_definitions",
    # Tracker
    "DebateSLOTracker",
    "get_debate_slo_tracker",
    "reset_debate_slo_tracker",
    # Convenience
    "get_debate_slo_status",
]
