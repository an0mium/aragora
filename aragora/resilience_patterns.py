"""Backward compatibility shim for aragora.resilience_patterns.

Re-exports all resilience patterns from the consolidated
aragora.resilience package.
"""

from aragora.resilience import (
    BaseCircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    ExponentialBackoff,
    HealthChecker,
    HealthReport,
    HealthStatus,
    RetryConfig,
    RetryStrategy,
    TimeoutConfig,
    calculate_backoff_delay,
    with_circuit_breaker,
    with_circuit_breaker_sync,
    with_retry,
    with_retry_sync,
    with_timeout,
    with_timeout_sync,
)

__all__ = [
    "BaseCircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "ExponentialBackoff",
    "HealthChecker",
    "HealthReport",
    "HealthStatus",
    "RetryConfig",
    "RetryStrategy",
    "TimeoutConfig",
    "calculate_backoff_delay",
    "with_circuit_breaker",
    "with_circuit_breaker_sync",
    "with_retry",
    "with_retry_sync",
    "with_timeout",
    "with_timeout_sync",
]
