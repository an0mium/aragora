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

# Decorator
from .decorator import with_resilience

# Wire up metrics callback in circuit_breaker module
from .circuit_breaker import _set_metrics_callback

_set_metrics_callback(_emit_metrics)

# =============================================================================
# NEW MODULE COMPATIBILITY ALIASES
# =============================================================================
# Re-export key components from resilience_patterns module for unified API.
# This allows connectors to use either old or new patterns during migration.

try:
    from aragora.resilience_patterns import (
        # Retry patterns
        RetryConfig,
        RetryStrategy,
        with_retry,
        with_retry_sync,
        # Timeout patterns
        TimeoutConfig,
        with_timeout,
        with_timeout_sync,
        # Circuit breaker v2
        BaseCircuitBreaker,
        CircuitBreakerConfig as CircuitBreakerConfigV2,
        CircuitState,
        with_circuit_breaker,
        # Health monitoring
        HealthChecker,
        HealthReport,
        HealthStatus,
    )

    __all__ = [
        # Core
        "CircuitBreaker",
        "CircuitOpenError",
        # Registry
        "MAX_CIRCUIT_BREAKERS",
        "STALE_THRESHOLD_SECONDS",
        "_circuit_breakers",
        "_circuit_breakers_lock",
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
        "_prune_stale_circuit_breakers",
        # NEW - Re-exports from resilience_patterns
        "RetryConfig",
        "RetryStrategy",
        "with_retry",
        "with_retry_sync",
        "TimeoutConfig",
        "with_timeout",
        "with_timeout_sync",
        "BaseCircuitBreaker",
        "CircuitBreakerConfigV2",
        "CircuitState",
        "with_circuit_breaker",
        "HealthChecker",
        "HealthReport",
        "HealthStatus",
    ]

except ImportError:
    # resilience_patterns not available - only export core patterns
    __all__ = [
        # Core
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
    ]
