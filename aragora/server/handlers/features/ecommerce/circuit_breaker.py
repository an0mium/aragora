"""Circuit breaker for e-commerce platform API access.

Delegates to the canonical SimpleCircuitBreaker implementation.
"""

from __future__ import annotations

import threading

from aragora.resilience.simple_circuit_breaker import SimpleCircuitBreaker

# Backwards-compatible alias
EcommerceCircuitBreaker = SimpleCircuitBreaker

# Global circuit breaker instance
_circuit_breaker = SimpleCircuitBreaker("Ecommerce", half_open_max_calls=2)
_circuit_breaker_lock = threading.Lock()


def get_ecommerce_circuit_breaker() -> SimpleCircuitBreaker:
    """Get the global circuit breaker for e-commerce platform access."""
    return _circuit_breaker


def reset_ecommerce_circuit_breaker() -> None:
    """Reset the global circuit breaker (for testing)."""
    with _circuit_breaker_lock:
        _circuit_breaker.reset()
