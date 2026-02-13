"""Circuit breaker for PagerDuty API access.

Delegates to the canonical SimpleCircuitBreaker implementation.
"""

from __future__ import annotations

import threading
from typing import Any

from aragora.resilience.simple_circuit_breaker import SimpleCircuitBreaker

# Backwards-compatible alias
DevOpsCircuitBreaker = SimpleCircuitBreaker

# Global circuit breaker instance
_devops_circuit_breaker: SimpleCircuitBreaker | None = None
_circuit_breaker_lock = threading.Lock()


def get_devops_circuit_breaker() -> SimpleCircuitBreaker:
    """Get or create the global DevOps circuit breaker."""
    global _devops_circuit_breaker
    with _circuit_breaker_lock:
        if _devops_circuit_breaker is None:
            _devops_circuit_breaker = SimpleCircuitBreaker("DevOps", half_open_max_calls=3)
        return _devops_circuit_breaker


def get_devops_circuit_breaker_status() -> dict[str, Any]:
    """Get the current status of the DevOps circuit breaker."""
    return get_devops_circuit_breaker().get_status()
