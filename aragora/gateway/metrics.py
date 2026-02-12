"""
Gateway Metrics for Aragora.

Provides Prometheus metrics specifically for the gateway subsystem:
- Request tracking (by framework, method, status)
- Action execution (by type, outcome)
- Circuit breaker state (per framework)
- Credential vault gauges (per tenant)
- Session tracking
- Policy decision tracking
- Audit event tracking

Uses the established observability pattern: lazy initialization with NoOpMetric
fallback when prometheus_client is not available.

Usage:
    from aragora.gateway.metrics import (
        record_gateway_request,
        record_gateway_action,
        record_policy_decision,
        record_audit_event,
        set_circuit_breaker_state,
        set_credentials_stored,
        set_active_sessions,
    )

    # Record a proxied request
    record_gateway_request(
        framework="openai",
        method="POST",
        status="200",
        duration_seconds=1.23,
    )

    # Record an action execution
    record_gateway_action(
        action_type="browser_navigate",
        outcome="completed",
        duration_seconds=5.4,
    )

    # Track circuit breaker state changes
    set_circuit_breaker_state("openai", "closed")

    # Track credential counts
    set_credentials_stored("tenant-123", 5)

    # Track active sessions
    set_active_sessions(42)
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

# Module-level initialization state
_initialized = False

# ---------------------------------------------------------------------------
# Request metrics
# ---------------------------------------------------------------------------
GATEWAY_REQUESTS_TOTAL: Any = None
GATEWAY_REQUEST_DURATION_SECONDS: Any = None

# ---------------------------------------------------------------------------
# Action metrics
# ---------------------------------------------------------------------------
GATEWAY_ACTIONS_TOTAL: Any = None
GATEWAY_ACTION_DURATION_SECONDS: Any = None

# ---------------------------------------------------------------------------
# Infrastructure gauges
# ---------------------------------------------------------------------------
GATEWAY_CIRCUIT_BREAKER_STATE: Any = None
GATEWAY_CREDENTIALS_STORED: Any = None
GATEWAY_ACTIVE_SESSIONS: Any = None

# ---------------------------------------------------------------------------
# Policy and audit counters
# ---------------------------------------------------------------------------
GATEWAY_POLICY_DECISIONS_TOTAL: Any = None
GATEWAY_AUDIT_EVENTS_TOTAL: Any = None


# ===========================================================================
# NoOp fallback
# ===========================================================================


class _NoOpMetric:
    """No-op metric for when prometheus_client is unavailable."""

    def labels(self, *args: Any, **kwargs: Any) -> _NoOpMetric:
        return self

    def inc(self, amount: float = 1) -> None:
        pass

    def dec(self, amount: float = 1) -> None:
        pass

    def set(self, value: float) -> None:
        pass

    def observe(self, value: float) -> None:
        pass


# ===========================================================================
# Initialization
# ===========================================================================


def _init_noop_metrics() -> None:
    """Assign NoOp stubs to every metric global."""
    global GATEWAY_REQUESTS_TOTAL, GATEWAY_REQUEST_DURATION_SECONDS
    global GATEWAY_ACTIONS_TOTAL, GATEWAY_ACTION_DURATION_SECONDS
    global GATEWAY_CIRCUIT_BREAKER_STATE, GATEWAY_CREDENTIALS_STORED
    global GATEWAY_ACTIVE_SESSIONS
    global GATEWAY_POLICY_DECISIONS_TOTAL, GATEWAY_AUDIT_EVENTS_TOTAL

    noop = _NoOpMetric()
    GATEWAY_REQUESTS_TOTAL = noop
    GATEWAY_REQUEST_DURATION_SECONDS = noop
    GATEWAY_ACTIONS_TOTAL = noop
    GATEWAY_ACTION_DURATION_SECONDS = noop
    GATEWAY_CIRCUIT_BREAKER_STATE = noop
    GATEWAY_CREDENTIALS_STORED = noop
    GATEWAY_ACTIVE_SESSIONS = noop
    GATEWAY_POLICY_DECISIONS_TOTAL = noop
    GATEWAY_AUDIT_EVENTS_TOTAL = noop


def init_gateway_metrics() -> bool:
    """Initialize gateway Prometheus metrics (lazy, idempotent).

    Returns:
        True if real Prometheus metrics are active, False otherwise.
    """
    global _initialized
    global GATEWAY_REQUESTS_TOTAL, GATEWAY_REQUEST_DURATION_SECONDS
    global GATEWAY_ACTIONS_TOTAL, GATEWAY_ACTION_DURATION_SECONDS
    global GATEWAY_CIRCUIT_BREAKER_STATE, GATEWAY_CREDENTIALS_STORED
    global GATEWAY_ACTIVE_SESSIONS
    global GATEWAY_POLICY_DECISIONS_TOTAL, GATEWAY_AUDIT_EVENTS_TOTAL

    if _initialized:
        return True

    try:
        from aragora.observability.config import get_metrics_config

        config = get_metrics_config()
        if not config.enabled:
            _init_noop_metrics()
            _initialized = True
            return False
    except Exception as exc:
        # If observability config is unavailable fall through to Prometheus
        logger.debug("Observability config unavailable: %s", exc)

    try:
        from prometheus_client import Counter, Gauge, Histogram

        # -- Request metrics --------------------------------------------------
        GATEWAY_REQUESTS_TOTAL = Counter(
            "aragora_gateway_requests_total",
            "Total requests proxied through the gateway",
            ["framework", "method", "status"],
        )

        GATEWAY_REQUEST_DURATION_SECONDS = Histogram(
            "aragora_gateway_request_duration_seconds",
            "Gateway request duration in seconds",
            ["framework", "method"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )

        # -- Action metrics ---------------------------------------------------
        GATEWAY_ACTIONS_TOTAL = Counter(
            "aragora_gateway_actions_total",
            "Total actions executed through the gateway",
            ["action_type", "outcome"],
        )

        GATEWAY_ACTION_DURATION_SECONDS = Histogram(
            "aragora_gateway_action_duration_seconds",
            "Gateway action execution duration in seconds",
            ["action_type"],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
        )

        # -- Infrastructure gauges --------------------------------------------
        GATEWAY_CIRCUIT_BREAKER_STATE = Gauge(
            "aragora_gateway_circuit_breaker_state",
            "Circuit breaker state per framework (0=closed, 1=open, 2=half-open)",
            ["framework"],
        )

        GATEWAY_CREDENTIALS_STORED = Gauge(
            "aragora_gateway_credentials_stored",
            "Number of credentials stored per tenant",
            ["tenant"],
        )

        GATEWAY_ACTIVE_SESSIONS = Gauge(
            "aragora_gateway_active_sessions",
            "Number of currently active gateway sessions",
        )

        # -- Policy and audit counters ----------------------------------------
        GATEWAY_POLICY_DECISIONS_TOTAL = Counter(
            "aragora_gateway_policy_decisions_total",
            "Total policy decisions made by the gateway",
            ["decision"],  # allow, deny, require_approval
        )

        GATEWAY_AUDIT_EVENTS_TOTAL = Counter(
            "aragora_gateway_audit_events_total",
            "Total audit events emitted by the gateway",
            ["event_type"],
        )

        _initialized = True
        logger.debug("Gateway metrics initialized (Prometheus)")
        return True

    except ImportError as e:
        logger.warning(f"prometheus_client not installed, gateway metrics disabled: {e}")
        _init_noop_metrics()
        _initialized = True
        return False
    except Exception as e:
        logger.error(f"Failed to initialize gateway metrics: {e}")
        _init_noop_metrics()
        _initialized = True
        return False


def _ensure_init() -> None:
    """Ensure metrics are initialized before use."""
    if not _initialized:
        init_gateway_metrics()


# ===========================================================================
# Recording functions -- Requests
# ===========================================================================


def record_gateway_request(
    framework: str,
    method: str,
    status: str,
    duration_seconds: float | None = None,
) -> None:
    """Record a gateway proxy request.

    Args:
        framework: Target framework name (e.g. "openai", "anthropic").
        method: HTTP method (e.g. "GET", "POST").
        status: HTTP status code as string (e.g. "200", "429").
        duration_seconds: Optional request duration in seconds.
    """
    _ensure_init()
    GATEWAY_REQUESTS_TOTAL.labels(framework=framework, method=method, status=status).inc()
    if duration_seconds is not None:
        GATEWAY_REQUEST_DURATION_SECONDS.labels(framework=framework, method=method).observe(
            duration_seconds
        )


@contextmanager
def track_gateway_request(
    framework: str,
    method: str,
) -> Generator[dict[str, Any], None, None]:
    """Context manager to track a gateway request end-to-end.

    Yields a mutable dict where the caller should set ``status`` before
    the block exits.  Duration is measured automatically.

    Example::

        with track_gateway_request("openai", "POST") as ctx:
            response = await do_request()
            ctx["status"] = str(response.status_code)

    Args:
        framework: Target framework name.
        method: HTTP method.
    """
    _ensure_init()
    ctx: dict[str, Any] = {"status": "500"}
    start = time.perf_counter()
    try:
        yield ctx
    finally:
        duration = time.perf_counter() - start
        record_gateway_request(framework, method, ctx["status"], duration)


# ===========================================================================
# Recording functions -- Actions
# ===========================================================================


def record_gateway_action(
    action_type: str,
    outcome: str,
    duration_seconds: float | None = None,
) -> None:
    """Record a gateway action execution.

    Args:
        action_type: Action type (e.g. "browser_navigate", "code_run").
        outcome: Execution outcome (e.g. "completed", "failed", "timeout",
                 "cancelled").
        duration_seconds: Optional action duration in seconds.
    """
    _ensure_init()
    GATEWAY_ACTIONS_TOTAL.labels(action_type=action_type, outcome=outcome).inc()
    if duration_seconds is not None:
        GATEWAY_ACTION_DURATION_SECONDS.labels(action_type=action_type).observe(duration_seconds)


@contextmanager
def track_gateway_action(
    action_type: str,
) -> Generator[dict[str, Any], None, None]:
    """Context manager to track a gateway action.

    Yields a mutable dict where the caller should set ``outcome`` before
    the block exits.  Duration is measured automatically.

    Args:
        action_type: Action type string.
    """
    _ensure_init()
    ctx: dict[str, Any] = {"outcome": "failed"}
    start = time.perf_counter()
    try:
        yield ctx
    finally:
        duration = time.perf_counter() - start
        record_gateway_action(action_type, ctx["outcome"], duration)


# ===========================================================================
# Recording functions -- Infrastructure gauges
# ===========================================================================

# State string -> numeric mapping for Prometheus gauge
_CB_STATE_MAP: dict[str, float] = {
    "closed": 0.0,
    "open": 1.0,
    "half-open": 2.0,
    "half_open": 2.0,
}


def set_circuit_breaker_state(framework: str, state: str) -> None:
    """Set the circuit breaker state gauge for a framework.

    Args:
        framework: Framework name (e.g. "openai").
        state: One of "closed", "open", "half-open".
    """
    _ensure_init()
    value = _CB_STATE_MAP.get(state, 0.0)
    GATEWAY_CIRCUIT_BREAKER_STATE.labels(framework=framework).set(value)


def set_credentials_stored(tenant: str, count: int) -> None:
    """Set the number of stored credentials for a tenant.

    Args:
        tenant: Tenant identifier.
        count: Number of credentials stored.
    """
    _ensure_init()
    GATEWAY_CREDENTIALS_STORED.labels(tenant=tenant).set(count)


def set_active_sessions(count: int) -> None:
    """Set the current number of active gateway sessions.

    Args:
        count: Number of active sessions.
    """
    _ensure_init()
    GATEWAY_ACTIVE_SESSIONS.set(count)


def inc_active_sessions() -> None:
    """Increment the active sessions gauge by one."""
    _ensure_init()
    GATEWAY_ACTIVE_SESSIONS.inc()


def dec_active_sessions() -> None:
    """Decrement the active sessions gauge by one."""
    _ensure_init()
    GATEWAY_ACTIVE_SESSIONS.dec()


# ===========================================================================
# Recording functions -- Policy and audit
# ===========================================================================


def record_policy_decision(decision: str) -> None:
    """Record a gateway policy decision.

    Args:
        decision: Decision type (e.g. "allow", "deny", "require_approval").
    """
    _ensure_init()
    GATEWAY_POLICY_DECISIONS_TOTAL.labels(decision=decision).inc()


def record_audit_event(event_type: str) -> None:
    """Record a gateway audit event.

    Args:
        event_type: Audit event type string (e.g. "openclaw.task.submitted").
    """
    _ensure_init()
    GATEWAY_AUDIT_EVENTS_TOTAL.labels(event_type=event_type).inc()


# ===========================================================================
# Exports
# ===========================================================================

__all__ = [
    # Initialization
    "init_gateway_metrics",
    # Metric instances (for advanced direct access)
    "GATEWAY_REQUESTS_TOTAL",
    "GATEWAY_REQUEST_DURATION_SECONDS",
    "GATEWAY_ACTIONS_TOTAL",
    "GATEWAY_ACTION_DURATION_SECONDS",
    "GATEWAY_CIRCUIT_BREAKER_STATE",
    "GATEWAY_CREDENTIALS_STORED",
    "GATEWAY_ACTIVE_SESSIONS",
    "GATEWAY_POLICY_DECISIONS_TOTAL",
    "GATEWAY_AUDIT_EVENTS_TOTAL",
    # Recording functions
    "record_gateway_request",
    "record_gateway_action",
    "record_policy_decision",
    "record_audit_event",
    "set_circuit_breaker_state",
    "set_credentials_stored",
    "set_active_sessions",
    "inc_active_sessions",
    "dec_active_sessions",
    # Context managers
    "track_gateway_request",
    "track_gateway_action",
]
