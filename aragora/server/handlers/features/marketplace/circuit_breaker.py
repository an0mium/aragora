"""Marketplace circuit breaker for template loading resilience.

Delegates to the canonical SimpleCircuitBreaker implementation.
"""

from __future__ import annotations

import threading
from typing import Any

from aragora.resilience.simple_circuit_breaker import SimpleCircuitBreaker

# Backwards-compatible alias
MarketplaceCircuitBreaker = SimpleCircuitBreaker

# Global circuit breaker instance
_circuit_breaker: SimpleCircuitBreaker | None = None
_circuit_breaker_lock = threading.Lock()


def _get_circuit_breaker() -> SimpleCircuitBreaker:
    """Get or create the marketplace circuit breaker."""
    global _circuit_breaker
    with _circuit_breaker_lock:
        if _circuit_breaker is None:
            _circuit_breaker = SimpleCircuitBreaker("Marketplace", half_open_max_calls=3)
        return _circuit_breaker


def _get_marketplace_circuit_breaker() -> SimpleCircuitBreaker:
    """Public accessor for the marketplace circuit breaker (for testing)."""
    return _get_circuit_breaker()


def get_marketplace_circuit_breaker_status() -> dict[str, Any]:
    """Get the marketplace circuit breaker status."""
    return _get_marketplace_circuit_breaker().get_status()


def _reset_circuit_breaker() -> None:
    """Reset the global circuit breaker instance (for testing)."""
    global _circuit_breaker
    with _circuit_breaker_lock:
        _circuit_breaker = None
