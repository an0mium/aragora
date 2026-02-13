"""
Resilience patterns for fault-tolerant systems.

Provides circuit breaker and other resilience patterns for graceful
failure handling in API calls and agent interactions.

This package provides:
- CircuitBreaker: Core circuit breaker implementation
- Registry: Global circuit breaker management
- Metrics: Monitoring and observability
- Persistence: SQLite-based state persistence
- Decorator: @with_resilience for async functions

Usage:
    from aragora.resilience import CircuitBreaker, get_circuit_breaker

    # Using the registry
    cb = get_circuit_breaker("my-service", provider="anthropic")

    # Direct instantiation
    cb = CircuitBreaker(failure_threshold=5, cooldown_seconds=30)

    # With decorator
    @with_resilience(circuit_name="api_call")
    async def call_api():
        ...
"""

from __future__ import annotations

# Core circuit breaker
from .circuit_breaker import CircuitBreaker, CircuitOpenError

# Registry management
from .registry import (
    MAX_CIRCUIT_BREAKERS,
    STALE_THRESHOLD_SECONDS,
    _circuit_breakers,
    _circuit_breakers_lock,
    _prune_stale_circuit_breakers,
    get_circuit_breaker,
    get_circuit_breakers,
    prune_circuit_breakers,
    reset_all_circuit_breakers,
)

# Metrics and status
from .metrics import (
    emit_metrics as _emit_metrics,
    get_all_circuit_breakers_status,
    get_circuit_breaker_metrics,
    get_circuit_breaker_status,
    get_circuit_breaker_summary,
    set_metrics_callback,
)

# Persistence
from .persistence import (
    cleanup_stale_persisted,
    init_circuit_breaker_persistence,
    load_circuit_breakers,
    persist_all_circuit_breakers,
    persist_circuit_breaker,
)

# Simple (sync-only) circuit breaker for feature handlers
from .simple_circuit_breaker import SimpleCircuitBreaker

# Decorator
from .decorator import with_resilience

# Wire up metrics callback in circuit_breaker module
from .circuit_breaker import _set_metrics_callback

_set_metrics_callback(_emit_metrics)

# =============================================================================
# RESILIENCE PATTERNS (retry, timeout, health, circuit breaker v2)
# =============================================================================
# These were originally in aragora.resilience_patterns and are now consolidated
# into this package. The aragora.resilience_patterns module is a backward-compat
# shim that re-exports from here.

# Retry patterns
from .retry import (
    ExponentialBackoff,
    JitterMode,
    RetryConfig,
    RetryStrategy,
    calculate_backoff_delay,
    with_retry,
    with_retry_sync,
)

# Timeout patterns
from .timeout import (
    TimeoutConfig,
    asyncio_timeout,
    is_timeout_available,
    timeout_context,
    timeout_context_sync,
    with_timeout,
    with_timeout_sync,
)

# Circuit breaker v2 (async-first, configurable)
from .circuit_breaker_v2 import (
    BaseCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerConfig as CircuitBreakerConfigV2,
    CircuitBreakerOpenError,
    CircuitBreakerStats,
    CircuitState,
    get_all_circuit_breakers as get_all_v2_circuit_breakers,
    get_circuit_breaker as get_v2_circuit_breaker,
    reset_all_circuit_breakers as reset_all_v2_circuit_breakers,
    with_circuit_breaker,
    with_circuit_breaker_sync,
)

# Health monitoring
from .health import (
    HealthChecker,
    HealthReport,
    HealthStatus,
    get_global_health_registry,
)

# HTTP client resilience utilities
from .http_client import (
    TRANSIENT_HTTP_STATUSES,
    CircuitBreakerMixin,
    HttpErrorInfo,
    ResilientRequestConfig,
    classify_http_error,
    make_resilient_request,
)

__all__ = [
    # Simple (sync-only) circuit breaker
    "SimpleCircuitBreaker",
    # Core v1
    "CircuitBreaker",
    "CircuitOpenError",
    # Registry
    "MAX_CIRCUIT_BREAKERS",
    "STALE_THRESHOLD_SECONDS",
    "_circuit_breakers",
    "_circuit_breakers_lock",
    "_prune_stale_circuit_breakers",
    "get_circuit_breaker",
    "get_circuit_breakers",
    "prune_circuit_breakers",
    "reset_all_circuit_breakers",
    # Metrics
    "_emit_metrics",
    "get_all_circuit_breakers_status",
    "get_circuit_breaker_metrics",
    "get_circuit_breaker_status",
    "get_circuit_breaker_summary",
    "set_metrics_callback",
    # Persistence
    "cleanup_stale_persisted",
    "init_circuit_breaker_persistence",
    "load_circuit_breakers",
    "persist_all_circuit_breakers",
    "persist_circuit_breaker",
    # Decorator
    "with_resilience",
    # Retry patterns
    "JitterMode",
    "RetryConfig",
    "RetryStrategy",
    "ExponentialBackoff",
    "calculate_backoff_delay",
    "with_retry",
    "with_retry_sync",
    # Timeout patterns
    "TimeoutConfig",
    "asyncio_timeout",
    "is_timeout_available",
    "timeout_context",
    "timeout_context_sync",
    "with_timeout",
    "with_timeout_sync",
    # Circuit breaker v2
    "BaseCircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerConfigV2",
    "CircuitBreakerOpenError",
    "CircuitBreakerStats",
    "CircuitState",
    "get_all_v2_circuit_breakers",
    "get_v2_circuit_breaker",
    "reset_all_v2_circuit_breakers",
    "with_circuit_breaker",
    "with_circuit_breaker_sync",
    # Health monitoring
    "HealthChecker",
    "HealthReport",
    "HealthStatus",
    "get_global_health_registry",
    # HTTP client resilience
    "TRANSIENT_HTTP_STATUSES",
    "CircuitBreakerMixin",
    "HttpErrorInfo",
    "ResilientRequestConfig",
    "classify_http_error",
    "make_resilient_request",
]
