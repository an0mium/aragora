"""
Prometheus metrics for rate limiting.

Provides detailed observability for rate limiting decisions, including:
- Rate limit decisions (allowed/rejected) by endpoint
- Bucket fill levels
- Per-tenant rejection tracking
- Redis backend status
- Distributed coordination metrics
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Try to import prometheus_client
try:
    from prometheus_client import Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# ============================================================================
# Metric Definitions
# ============================================================================

if PROMETHEUS_AVAILABLE:
    # Core rate limit decision metrics
    RATE_LIMIT_DECISIONS = Counter(
        "aragora_rate_limit_decisions_total",
        "Total rate limit decisions by endpoint and decision outcome",
        ["endpoint", "decision"],  # decision: allowed, rejected
    )

    # Bucket usage metrics (current fill level as percentage)
    RATE_LIMIT_BUCKET_USAGE = Gauge(
        "aragora_rate_limit_bucket_usage",
        "Current bucket fill level as percentage (0-100)",
        ["endpoint"],
    )

    # Per-tenant rejection tracking
    RATE_LIMIT_REJECTIONS = Counter(
        "aragora_rate_limit_rejections_total",
        "Total rate limit rejections by endpoint and tenant",
        ["endpoint", "tenant_id"],
    )

    # Redis backend metrics
    RATE_LIMIT_REDIS_OPERATIONS = Counter(
        "aragora_rate_limit_redis_operations_total",
        "Total Redis operations for rate limiting",
        ["operation", "status"],  # operation: check, sync; status: success, failure
    )

    RATE_LIMIT_REDIS_LATENCY = Histogram(
        "aragora_rate_limit_redis_latency_seconds",
        "Redis operation latency for rate limiting",
        ["operation"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    )

    RATE_LIMIT_BACKEND_STATUS = Gauge(
        "aragora_rate_limit_backend_status",
        "Current rate limit backend status (1=redis, 0=memory)",
        ["instance_id"],
    )

    # Circuit breaker metrics
    RATE_LIMIT_CIRCUIT_BREAKER_STATE = Gauge(
        "aragora_rate_limit_circuit_breaker_state",
        "Circuit breaker state (0=closed, 1=open, 2=half_open)",
        ["instance_id"],
    )

    RATE_LIMIT_FALLBACK_REQUESTS = Counter(
        "aragora_rate_limit_fallback_requests_total",
        "Requests handled by in-memory fallback due to Redis issues",
        ["instance_id"],
    )

    # Distributed coordination metrics
    RATE_LIMIT_INSTANCE_COUNT = Gauge(
        "aragora_rate_limit_instance_count",
        "Number of rate limiter instances coordinating via Redis",
    )

    RATE_LIMIT_DISTRIBUTED_REJECTIONS = Counter(
        "aragora_rate_limit_distributed_rejections_total",
        "Total rejections across all instances (aggregated from Redis)",
    )


# ============================================================================
# Metric Recording Functions
# ============================================================================


def record_rate_limit_decision(
    endpoint: str,
    allowed: bool,
    remaining: int = 0,
    limit: int = 0,
    tenant_id: str | None = None,
) -> None:
    """Record a rate limit decision for metrics.

    Args:
        endpoint: The API endpoint being rate limited
        allowed: Whether the request was allowed
        remaining: Remaining requests in the bucket
        limit: Total bucket limit
        tenant_id: Optional tenant ID for per-tenant tracking
    """
    if not PROMETHEUS_AVAILABLE:
        return

    decision = "allowed" if allowed else "rejected"
    normalized_endpoint = _normalize_endpoint_for_metrics(endpoint)

    try:
        RATE_LIMIT_DECISIONS.labels(endpoint=normalized_endpoint, decision=decision).inc()

        # Track bucket usage as percentage
        if limit > 0:
            usage_pct = ((limit - remaining) / limit) * 100
            RATE_LIMIT_BUCKET_USAGE.labels(endpoint=normalized_endpoint).set(usage_pct)

        # Track per-tenant rejections
        if not allowed and tenant_id:
            RATE_LIMIT_REJECTIONS.labels(
                endpoint=normalized_endpoint,
                tenant_id=tenant_id,
            ).inc()

    except Exception as e:
        logger.debug(f"Failed to record rate limit metrics: {e}")


def record_redis_operation(
    operation: str,
    success: bool,
    latency_seconds: float | None = None,
) -> None:
    """Record a Redis operation for rate limiting.

    Args:
        operation: The type of operation (check, sync, etc.)
        success: Whether the operation succeeded
        latency_seconds: Optional operation latency
    """
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        status = "success" if success else "failure"
        RATE_LIMIT_REDIS_OPERATIONS.labels(operation=operation, status=status).inc()

        if latency_seconds is not None:
            RATE_LIMIT_REDIS_LATENCY.labels(operation=operation).observe(latency_seconds)

    except Exception as e:
        logger.debug(f"Failed to record Redis operation metrics: {e}")


def record_backend_status(instance_id: str, using_redis: bool) -> None:
    """Record the current rate limit backend status.

    Args:
        instance_id: The server instance ID
        using_redis: Whether Redis is being used (vs memory fallback)
    """
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        RATE_LIMIT_BACKEND_STATUS.labels(instance_id=instance_id).set(1 if using_redis else 0)
    except Exception as e:
        logger.debug(f"Failed to record backend status: {e}")


def record_circuit_breaker_state(instance_id: str, state: str) -> None:
    """Record circuit breaker state.

    Args:
        instance_id: The server instance ID
        state: The circuit breaker state (closed, open, half_open)
    """
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        state_value = {"closed": 0, "open": 1, "half_open": 2}.get(state, -1)
        RATE_LIMIT_CIRCUIT_BREAKER_STATE.labels(instance_id=instance_id).set(state_value)
    except Exception as e:
        logger.debug(f"Failed to record circuit breaker state: {e}")


def record_fallback_request(instance_id: str) -> None:
    """Record a request that fell back to in-memory rate limiting.

    Args:
        instance_id: The server instance ID
    """
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        RATE_LIMIT_FALLBACK_REQUESTS.labels(instance_id=instance_id).inc()
    except Exception as e:
        logger.debug(f"Failed to record fallback request: {e}")


def record_distributed_metrics(instance_count: int, total_rejections: int) -> None:
    """Record aggregated distributed rate limiting metrics.

    Args:
        instance_count: Number of coordinated instances
        total_rejections: Total rejections across all instances
    """
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        RATE_LIMIT_INSTANCE_COUNT.set(instance_count)
        # Note: Counter can only increment, so we track delta
        # This is called periodically with the current total
    except Exception as e:
        logger.debug(f"Failed to record distributed metrics: {e}")


def _normalize_endpoint_for_metrics(endpoint: str) -> str:
    """Normalize endpoint path for metrics labels.

    Reduces cardinality by:
    - Replacing numeric path segments with placeholders
    - Limiting depth
    - Normalizing case
    """
    if not endpoint:
        return "unknown"

    # Lowercase and limit length
    normalized = endpoint.lower()[:100]

    # Replace UUIDs and numeric IDs with placeholders
    import re

    # UUID pattern
    normalized = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "{id}",
        normalized,
    )
    # Numeric IDs
    normalized = re.sub(r"/\d+(?=/|$)", "/{id}", normalized)

    return normalized


def get_rate_limit_metrics() -> dict[str, Any]:
    """Get current rate limit metrics as a dictionary.

    Useful for health checks and debugging when Prometheus is not available.

    Returns:
        Dictionary of current metric values
    """
    if not PROMETHEUS_AVAILABLE:
        return {"prometheus_available": False}

    try:
        # Collect samples from counters/gauges
        backend_status: dict[str, str] = {}
        circuit_breaker_states: dict[str, str] = {}

        # Get backend status by instance
        backend_status_metrics = list(RATE_LIMIT_BACKEND_STATUS.collect())
        if backend_status_metrics:
            for sample in backend_status_metrics[0].samples:
                instance_id = sample.labels.get("instance_id", "unknown")
                backend_status[instance_id] = "redis" if sample.value == 1 else "memory"

        # Get circuit breaker states
        circuit_breaker_metrics = list(RATE_LIMIT_CIRCUIT_BREAKER_STATE.collect())
        if circuit_breaker_metrics:
            for sample in circuit_breaker_metrics[0].samples:
                instance_id = sample.labels.get("instance_id", "unknown")
                state_map = {0: "closed", 1: "open", 2: "half_open"}
                circuit_breaker_states[instance_id] = state_map.get(int(sample.value), "unknown")

        return {
            "prometheus_available": True,
            "backend_status": backend_status,
            "circuit_breaker_states": circuit_breaker_states,
        }

    except Exception as e:
        logger.debug(f"Failed to collect rate limit metrics: {e}")
        return {"prometheus_available": True, "error": "Failed to collect rate limit metrics"}


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "record_rate_limit_decision",
    "record_redis_operation",
    "record_backend_status",
    "record_circuit_breaker_state",
    "record_fallback_request",
    "record_distributed_metrics",
    "get_rate_limit_metrics",
]
