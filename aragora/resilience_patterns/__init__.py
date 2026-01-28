"""
Unified Resilience Patterns for Aragora.

This module provides consolidated resilience patterns used across the codebase:
- Circuit breakers for failure isolation
- Retry logic with configurable backoff strategies
- Timeout management for operations
- Health monitoring utilities

Design Goals:
- Single source of truth for resilience patterns
- Backward compatibility with existing code
- Configurable via dataclasses
- Async-first with sync support

Usage:
    from aragora.resilience_patterns import (
        CircuitBreakerConfig,
        RetryConfig,
        with_retry,
        with_timeout,
    )

    # Circuit breaker
    @with_circuit_breaker("my_service")
    async def call_service():
        ...

    # Retry with exponential backoff
    @with_retry(RetryConfig(max_retries=3, strategy="exponential"))
    async def flaky_operation():
        ...

    # Timeout
    @with_timeout(5.0)
    async def bounded_operation():
        ...
"""

from .retry import (
    RetryStrategy,
    RetryConfig,
    ExponentialBackoff,
    with_retry,
    with_retry_sync,
    calculate_backoff_delay,
)

from .timeout import (
    TimeoutConfig,
    with_timeout,
    with_timeout_sync,
    asyncio_timeout,
)

from .circuit_breaker import (
    CircuitState,
    CircuitBreakerConfig,
    BaseCircuitBreaker,
    with_circuit_breaker,
)

from .health import (
    HealthStatus,
    HealthChecker,
    HealthReport,
)

from .metrics import (
    circuit_breaker_state_changed,
    retry_attempt,
    retry_exhausted,
    timeout_occurred,
    health_status_changed,
    operation_duration,
    create_metrics_callbacks,
)

__all__ = [
    # Retry
    "RetryStrategy",
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
    "BaseCircuitBreaker",
    "with_circuit_breaker",
    # Health
    "HealthStatus",
    "HealthChecker",
    "HealthReport",
    # Metrics
    "circuit_breaker_state_changed",
    "retry_attempt",
    "retry_exhausted",
    "timeout_occurred",
    "health_status_changed",
    "operation_duration",
    "create_metrics_callbacks",
]
