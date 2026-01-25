"""
Decision Pipeline Metrics and Observability.

Provides Prometheus metrics and instrumentation for the decision routing pipeline:
- Request counts by type, source, and priority
- Latency histograms
- Confidence distribution
- Cache hit/miss rates
- Error tracking

Usage:
    from aragora.observability.decision_metrics import (
        record_decision_request,
        record_decision_result,
        get_decision_metrics,
    )

    # Record a decision request
    record_decision_request(
        decision_type="debate",
        source="slack",
        priority="high",
    )

    # Record a decision result
    record_decision_result(
        decision_type="debate",
        source="slack",
        success=True,
        confidence=0.85,
        duration_seconds=4.5,
        cache_hit=False,
    )

Requirements:
    pip install prometheus-client

Environment Variables:
    DECISION_METRICS_ENABLED: Set to "true" to enable metrics (default: true)
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

# Prometheus metrics - initialized lazily
_initialized = False

# Metric instances (will be set during initialization)
DECISION_REQUESTS: Any = None
DECISION_RESULTS: Any = None
DECISION_LATENCY: Any = None
DECISION_CONFIDENCE: Any = None
DECISION_CACHE_HITS: Any = None
DECISION_CACHE_MISSES: Any = None
DECISION_DEDUP_HITS: Any = None
DECISION_ACTIVE: Any = None
DECISION_ERRORS: Any = None
DECISION_CONSENSUS_RATE: Any = None
DECISION_AGENTS_USED: Any = None


def _init_metrics() -> bool:
    """Initialize Prometheus metrics lazily."""
    global _initialized
    global DECISION_REQUESTS, DECISION_RESULTS, DECISION_LATENCY
    global DECISION_CONFIDENCE, DECISION_CACHE_HITS, DECISION_CACHE_MISSES
    global DECISION_DEDUP_HITS, DECISION_ACTIVE, DECISION_ERRORS
    global DECISION_CONSENSUS_RATE, DECISION_AGENTS_USED

    if _initialized:
        return True

    try:
        from aragora.observability.config import get_metrics_config

        config = get_metrics_config()
        if not config.enabled:
            _init_noop_metrics()
            _initialized = True
            return False
    except ImportError:
        _init_noop_metrics()
        _initialized = True
        return False

    try:
        from prometheus_client import Counter, Gauge, Histogram, Summary

        # Request metrics
        DECISION_REQUESTS = Counter(
            "aragora_decision_requests_total",
            "Total decision requests received",
            ["decision_type", "source", "priority"],
        )

        DECISION_RESULTS = Counter(
            "aragora_decision_results_total",
            "Total decision results by outcome",
            ["decision_type", "source", "success", "consensus_reached"],
        )

        # Latency histogram with buckets for different decision types
        DECISION_LATENCY = Histogram(
            "aragora_decision_latency_seconds",
            "Decision processing latency",
            ["decision_type", "source"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
        )

        # Confidence distribution
        DECISION_CONFIDENCE = Histogram(
            "aragora_decision_confidence",
            "Decision confidence score distribution",
            ["decision_type"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        # Cache metrics
        DECISION_CACHE_HITS = Counter(
            "aragora_decision_cache_hits_total",
            "Total decision cache hits",
            ["decision_type"],
        )

        DECISION_CACHE_MISSES = Counter(
            "aragora_decision_cache_misses_total",
            "Total decision cache misses",
            ["decision_type"],
        )

        DECISION_DEDUP_HITS = Counter(
            "aragora_decision_dedup_hits_total",
            "Total deduplication hits (waited for in-flight request)",
            ["decision_type"],
        )

        # Active decisions gauge
        DECISION_ACTIVE = Gauge(
            "aragora_decision_active",
            "Number of currently active decision requests",
            ["decision_type"],
        )

        # Error tracking
        DECISION_ERRORS = Counter(
            "aragora_decision_errors_total",
            "Total decision errors by type",
            ["decision_type", "error_type"],
        )

        # Consensus rate (for debates)
        DECISION_CONSENSUS_RATE = Summary(
            "aragora_decision_consensus_rate",
            "Rate of consensus achieved in debates",
        )

        # Agents used per decision
        DECISION_AGENTS_USED = Histogram(
            "aragora_decision_agents_used",
            "Number of agents used per decision",
            ["decision_type"],
            buckets=[1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20],
        )

        _initialized = True
        logger.info("Decision Prometheus metrics initialized")
        return True

    except ImportError:
        logger.warning("prometheus_client not installed, decision metrics disabled")
        _init_noop_metrics()
        _initialized = True
        return False


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global DECISION_REQUESTS, DECISION_RESULTS, DECISION_LATENCY
    global DECISION_CONFIDENCE, DECISION_CACHE_HITS, DECISION_CACHE_MISSES
    global DECISION_DEDUP_HITS, DECISION_ACTIVE, DECISION_ERRORS
    global DECISION_CONSENSUS_RATE, DECISION_AGENTS_USED

    class NoopMetric:
        """No-op metric that accepts any method call and supports chaining."""

        def __getattr__(self, name: str) -> Any:
            # Return self to allow method chaining like .labels().inc()
            return lambda *args, **kwargs: self

    noop = NoopMetric()
    DECISION_REQUESTS = noop
    DECISION_RESULTS = noop
    DECISION_LATENCY = noop
    DECISION_CONFIDENCE = noop
    DECISION_CACHE_HITS = noop
    DECISION_CACHE_MISSES = noop
    DECISION_DEDUP_HITS = noop
    DECISION_ACTIVE = noop
    DECISION_ERRORS = noop
    DECISION_CONSENSUS_RATE = noop
    DECISION_AGENTS_USED = noop


# =============================================================================
# Recording Functions
# =============================================================================


def record_decision_request(
    decision_type: str,
    source: str,
    priority: str = "normal",
) -> None:
    """Record an incoming decision request."""
    _init_metrics()
    DECISION_REQUESTS.labels(
        decision_type=decision_type,
        source=source,
        priority=priority,
    ).inc()
    DECISION_ACTIVE.labels(decision_type=decision_type).inc()


def record_decision_result(
    decision_type: str,
    source: str,
    success: bool,
    confidence: float,
    duration_seconds: float,
    consensus_reached: bool = False,
    cache_hit: bool = False,
    dedup_hit: bool = False,
    agent_count: int = 0,
    error_type: Optional[str] = None,
) -> None:
    """Record a decision result with all metrics."""
    _init_metrics()

    # Record result
    DECISION_RESULTS.labels(
        decision_type=decision_type,
        source=source,
        success=str(success).lower(),
        consensus_reached=str(consensus_reached).lower(),
    ).inc()

    # Record latency
    DECISION_LATENCY.labels(
        decision_type=decision_type,
        source=source,
    ).observe(duration_seconds)

    # Record confidence
    if confidence > 0:
        DECISION_CONFIDENCE.labels(decision_type=decision_type).observe(confidence)

    # Cache metrics
    if cache_hit:
        DECISION_CACHE_HITS.labels(decision_type=decision_type).inc()
    else:
        DECISION_CACHE_MISSES.labels(decision_type=decision_type).inc()

    # Deduplication metrics
    if dedup_hit:
        DECISION_DEDUP_HITS.labels(decision_type=decision_type).inc()

    # Agent count
    if agent_count > 0:
        DECISION_AGENTS_USED.labels(decision_type=decision_type).observe(agent_count)

    # Error tracking
    if error_type:
        DECISION_ERRORS.labels(
            decision_type=decision_type,
            error_type=error_type,
        ).inc()

    # Consensus for debates
    if decision_type == "debate":
        DECISION_CONSENSUS_RATE.observe(1.0 if consensus_reached else 0.0)

    # Decrement active count
    DECISION_ACTIVE.labels(decision_type=decision_type).dec()


def record_decision_error(
    decision_type: str,
    error_type: str,
) -> None:
    """Record a decision error."""
    _init_metrics()
    DECISION_ERRORS.labels(
        decision_type=decision_type,
        error_type=error_type,
    ).inc()


def record_decision_cache_hit(decision_type: str) -> None:
    """Record a decision cache hit."""
    _init_metrics()
    DECISION_CACHE_HITS.labels(decision_type=decision_type).inc()


def record_decision_cache_miss(decision_type: str) -> None:
    """Record a decision cache miss."""
    _init_metrics()
    DECISION_CACHE_MISSES.labels(decision_type=decision_type).inc()


def record_decision_dedup_hit(decision_type: str) -> None:
    """Record a deduplication hit."""
    _init_metrics()
    DECISION_DEDUP_HITS.labels(decision_type=decision_type).inc()


# =============================================================================
# Context Managers
# =============================================================================


@contextmanager
def track_decision(
    decision_type: str,
    source: str,
    priority: str = "normal",
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for tracking a decision through its lifecycle.

    Usage:
        with track_decision("debate", "slack", "high") as ctx:
            result = await router.route(request)
            ctx["success"] = result.success
            ctx["confidence"] = result.confidence
            ctx["consensus_reached"] = result.consensus_reached
            ctx["agent_count"] = len(result.agent_contributions)

    Args:
        decision_type: Type of decision
        source: Input source
        priority: Request priority

    Yields:
        Context dict to populate with result data
    """
    start_time = time.perf_counter()
    record_decision_request(decision_type, source, priority)

    context: Dict[str, Any] = {
        "success": True,
        "confidence": 0.0,
        "consensus_reached": False,
        "cache_hit": False,
        "dedup_hit": False,
        "agent_count": 0,
        "error_type": None,
    }

    try:
        yield context
    except Exception as e:
        context["success"] = False
        context["error_type"] = type(e).__name__
        raise
    finally:
        duration = time.perf_counter() - start_time
        record_decision_result(
            decision_type=decision_type,
            source=source,
            success=context.get("success", False),
            confidence=context.get("confidence", 0.0),
            duration_seconds=duration,
            consensus_reached=context.get("consensus_reached", False),
            cache_hit=context.get("cache_hit", False),
            dedup_hit=context.get("dedup_hit", False),
            agent_count=context.get("agent_count", 0),
            error_type=context.get("error_type"),
        )


# =============================================================================
# Metrics Retrieval
# =============================================================================


def get_decision_metrics() -> Dict[str, Any]:
    """Get current decision metrics summary."""
    _init_metrics()

    try:
        from prometheus_client import REGISTRY

        metrics = {}

        # Collect all decision metrics
        for metric in REGISTRY.collect():
            if metric.name.startswith("aragora_decision"):
                samples = list(metric.samples)
                if samples:
                    metrics[metric.name] = {
                        "help": metric.documentation,
                        "type": metric.type,
                        "samples": [{"labels": dict(s.labels), "value": s.value} for s in samples],
                    }

        return metrics

    except ImportError:
        return {"error": "prometheus_client not installed"}
    except Exception as e:
        return {"error": str(e)}


def get_decision_summary() -> Dict[str, Any]:
    """Get a human-readable summary of decision metrics."""
    _init_metrics()

    try:
        from prometheus_client import REGISTRY

        summary: dict[str, dict[str, Any]] = {
            "requests": {},
            "latency": {},
            "confidence": {},
            "errors": {},
            "cache": {},
        }

        for metric in REGISTRY.collect():
            name = metric.name
            samples = list(metric.samples)

            if name == "aragora_decision_requests_total":
                for sample in samples:
                    key = f"{sample.labels.get('decision_type', 'unknown')}"
                    summary["requests"][key] = summary["requests"].get(key, 0) + sample.value

            elif name == "aragora_decision_latency_seconds":
                for sample in samples:
                    if "_sum" in sample.name:
                        key = sample.labels.get("decision_type", "unknown")
                        summary["latency"][f"{key}_sum"] = sample.value
                    elif "_count" in sample.name:
                        key = sample.labels.get("decision_type", "unknown")
                        summary["latency"][f"{key}_count"] = sample.value

            elif name == "aragora_decision_errors_total":
                for sample in samples:
                    dt = sample.labels.get("decision_type", "unknown")
                    et = sample.labels.get("error_type", "unknown")
                    summary["errors"][f"{dt}_{et}"] = sample.value

            elif name == "aragora_decision_cache_hits_total":
                for sample in samples:
                    key = sample.labels.get("decision_type", "unknown")
                    summary["cache"][f"{key}_hits"] = sample.value

            elif name == "aragora_decision_cache_misses_total":
                for sample in samples:
                    key = sample.labels.get("decision_type", "unknown")
                    summary["cache"][f"{key}_misses"] = sample.value

        # Calculate averages
        for key in list(summary["latency"].keys()):
            if key.endswith("_sum"):
                base_key = key[:-4]
                count_key = f"{base_key}_count"
                if count_key in summary["latency"]:
                    count = summary["latency"][count_key]
                    if count > 0:
                        avg = summary["latency"][key] / count
                        summary["latency"][f"{base_key}_avg"] = round(avg, 3)

        return summary

    except ImportError:
        return {"error": "prometheus_client not installed"}
    except Exception as e:
        return {"error": str(e)}


__all__ = [
    "record_decision_request",
    "record_decision_result",
    "record_decision_error",
    "record_decision_cache_hit",
    "record_decision_cache_miss",
    "record_decision_dedup_hit",
    "track_decision",
    "get_decision_metrics",
    "get_decision_summary",
]
