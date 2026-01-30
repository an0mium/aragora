"""
Unified Resilience Patterns for Aragora.

DEPRECATED: This package is a backward compatibility shim.
Import from `aragora.resilience` instead:

    # New style (preferred)
    from aragora.resilience import RetryConfig, with_retry, with_timeout

    # Old style (still works)
    from aragora.resilience_patterns import RetryConfig, with_retry, with_timeout

Both imports resolve to the same code.
"""

import warnings

warnings.warn(
    "aragora.resilience_patterns is deprecated. "
    "Import from aragora.resilience instead. "
    "This package will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the consolidated resilience package.
# Submodule imports (e.g., from aragora.resilience_patterns.retry import ...)
# still work because the original .py files remain in this directory.

from aragora.resilience import (
    # Retry
    RetryConfig,
    RetryStrategy,
    with_retry,
    with_retry_sync,
    # Timeout
    TimeoutConfig,
    with_timeout,
    with_timeout_sync,
    # Circuit breaker v2
    BaseCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerConfigV2,
    CircuitState,
    with_circuit_breaker,
    # Health
    HealthChecker,
    HealthReport,
    HealthStatus,
)

# Re-export items that consumers import by name from this package
from .retry import (
    JitterMode,
    ExponentialBackoff,
    calculate_backoff_delay,
)

from .circuit_breaker import (
    CircuitBreakerOpenError,
    CircuitBreakerStats,
    get_circuit_breaker,
    reset_all_circuit_breakers,
    get_all_circuit_breakers,
    with_circuit_breaker_sync,
)

from .timeout import asyncio_timeout

from .health import HealthRegistry, get_global_health_registry

from .metrics import (
    circuit_breaker_state_changed,
    retry_attempt,
    retry_exhausted,
    timeout_occurred,
    health_status_changed,
    operation_duration,
    create_metrics_callbacks,
    reset_metrics,
)

__all__ = [
    # Retry
    "RetryStrategy",
    "JitterMode",
    "RetryConfig",
    "ExponentialBackoff",
    "with_retry",
    "with_retry_sync",
    "calculate_backoff_delay",
    # Timeout
    "TimeoutConfig",
    "with_timeout",
    "with_timeout_sync",
    "asyncio_timeout",
    # Circuit breaker
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreakerConfigV2",
    "CircuitBreakerOpenError",
    "CircuitBreakerStats",
    "BaseCircuitBreaker",
    "with_circuit_breaker",
    "with_circuit_breaker_sync",
    "get_circuit_breaker",
    "reset_all_circuit_breakers",
    "get_all_circuit_breakers",
    # Health
    "HealthStatus",
    "HealthChecker",
    "HealthReport",
    "HealthRegistry",
    "get_global_health_registry",
    # Metrics
    "circuit_breaker_state_changed",
    "retry_attempt",
    "retry_exhausted",
    "timeout_occurred",
    "health_status_changed",
    "operation_duration",
    "create_metrics_callbacks",
    "reset_metrics",
]
