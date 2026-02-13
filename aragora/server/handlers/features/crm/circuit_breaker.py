"""CRM Circuit Breaker - Resilience Pattern for CRM Platform Access.

Delegates to the canonical SimpleCircuitBreaker implementation.

Stability: STABLE
"""

from __future__ import annotations

import threading

from aragora.resilience.simple_circuit_breaker import SimpleCircuitBreaker

# Backwards-compatible alias
CRMCircuitBreaker = SimpleCircuitBreaker

# Global circuit breaker instance
_circuit_breaker = SimpleCircuitBreaker("CRM", half_open_max_calls=2)
_circuit_breaker_lock = threading.Lock()


def get_crm_circuit_breaker() -> SimpleCircuitBreaker:
    """Get the global circuit breaker for CRM platform access."""
    return _circuit_breaker


def reset_crm_circuit_breaker() -> None:
    """Reset the global circuit breaker (for testing)."""
    with _circuit_breaker_lock:
        _circuit_breaker.reset()


__all__ = [
    "CRMCircuitBreaker",
    "get_crm_circuit_breaker",
    "reset_crm_circuit_breaker",
]
