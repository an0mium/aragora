"""DEPRECATED: circuit_breaker_v2 has been consolidated (no code deleted).

This module is a backward-compatibility shim.  The implementation that
previously lived here (BaseCircuitBreaker, CircuitBreakerConfig, the
with_circuit_breaker decorators, and the global named registry) now lives in
aragora.resilience.simple_circuit_breaker, the canonical home for the
thread-safe synchronous-core circuit breakers.

Migration:
    # Before
    from aragora.resilience.circuit_breaker_v2 import BaseCircuitBreaker

    # After
    from aragora.resilience.simple_circuit_breaker import BaseCircuitBreaker

All names re-exported below are identity-equal to their canonical targets:
registry state is shared, isinstance checks keep working, and behavior is
unchanged.  For the asyncio-native multi-entity breaker, see
aragora.resilience.circuit_breaker.CircuitBreaker.
"""

from __future__ import annotations

import warnings

from aragora.resilience.simple_circuit_breaker import (  # noqa: F401
    BaseCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerStats,
    CircuitState,
    _circuit_breakers,
    _circuit_breakers_lock,
    get_all_circuit_breakers,
    get_circuit_breaker,
    reset_all_circuit_breakers,
    with_circuit_breaker,
    with_circuit_breaker_sync,
)

warnings.warn(
    "aragora.resilience.circuit_breaker_v2 is deprecated; import from "
    "aragora.resilience.simple_circuit_breaker (sync-core breakers, "
    "BaseCircuitBreaker, decorators, registry) or "
    "aragora.resilience.circuit_breaker (async multi-entity CircuitBreaker) "
    "instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CircuitState",
    "CircuitBreakerOpenError",
    "CircuitBreakerConfig",
    "CircuitBreakerStats",
    "BaseCircuitBreaker",
    "with_circuit_breaker",
    "with_circuit_breaker_sync",
    "get_circuit_breaker",
    "reset_all_circuit_breakers",
    "get_all_circuit_breakers",
]
