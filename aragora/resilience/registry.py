"""
Global circuit breaker registry for shared state across components.

Thread-safe management of circuit breaker instances with automatic pruning.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from aragora.resilience_config import (
    CircuitBreakerConfig,
    get_circuit_breaker_config,
)

if TYPE_CHECKING:
    from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

# Global circuit breaker registry for shared state across components (thread-safe)
_circuit_breakers: dict[str, CircuitBreaker] = {}
_circuit_breakers_lock = threading.Lock()

# Configuration for circuit breaker pruning
MAX_CIRCUIT_BREAKERS = 1000  # Maximum registry size before forced pruning
STALE_THRESHOLD_SECONDS = 24 * 60 * 60  # 24 hours - prune if not accessed


def _prune_stale_circuit_breakers() -> int:
    """Remove circuit breakers not accessed within STALE_THRESHOLD_SECONDS.

    Called automatically when registry exceeds MAX_CIRCUIT_BREAKERS.
    Must be called with _circuit_breakers_lock held.

    Returns:
        Number of circuit breakers pruned.
    """
    now = time.time()
    stale_names = []
    for name, cb in _circuit_breakers.items():
        if hasattr(cb, "_last_accessed") and (now - cb._last_accessed) > STALE_THRESHOLD_SECONDS:
            stale_names.append(name)

    for name in stale_names:
        del _circuit_breakers[name]

    if stale_names:
        logger.info(f"Pruned {len(stale_names)} stale circuit breakers: {stale_names[:5]}...")
    return len(stale_names)


def get_circuit_breaker(
    name: str,
    failure_threshold: int | None = None,
    cooldown_seconds: float | None = None,
    provider: str | None = None,
    config: CircuitBreakerConfig | None = None,
) -> CircuitBreaker:
    """
    Get or create a named circuit breaker from the global registry (thread-safe).

    This ensures consistent circuit breaker state across components
    for the same service/agent.

    Automatically prunes stale circuit breakers (not accessed in 24h) when
    the registry exceeds MAX_CIRCUIT_BREAKERS entries.

    Configuration resolution order:
    1. Explicit config parameter (if provided)
    2. Provider-based config lookup (if provider provided)
    3. Explicit failure_threshold/cooldown_seconds parameters
    4. Default config from get_circuit_breaker_config()

    Args:
        name: Unique identifier for this circuit breaker (e.g., "agent_claude")
        failure_threshold: Failures before opening circuit (legacy, prefer config)
        cooldown_seconds: Seconds before attempting recovery (legacy, prefer config)
        provider: Provider name for automatic config lookup (e.g., "anthropic", "openai")
        config: Explicit CircuitBreakerConfig to use

    Returns:
        CircuitBreaker instance (shared if already exists)

    Example:
        # Using provider-based config
        cb = get_circuit_breaker("anthropic_agent", provider="anthropic")

        # Using explicit config
        cb = get_circuit_breaker("custom", config=CircuitBreakerConfig(failure_threshold=10))

        # Legacy parameters still work
        cb = get_circuit_breaker("legacy", failure_threshold=3, cooldown_seconds=60)
    """
    # Import here to avoid circular imports
    from .circuit_breaker import CircuitBreaker

    with _circuit_breakers_lock:
        # Prune if registry is getting too large
        if len(_circuit_breakers) >= MAX_CIRCUIT_BREAKERS:
            pruned = _prune_stale_circuit_breakers()
            # If still too large after pruning, log warning
            if len(_circuit_breakers) >= MAX_CIRCUIT_BREAKERS:
                logger.warning(
                    f"Circuit breaker registry still large after pruning {pruned}: "
                    f"{len(_circuit_breakers)} entries"
                )

        if name not in _circuit_breakers:
            # Resolve configuration
            resolved_config: CircuitBreakerConfig
            if config is not None:
                resolved_config = config
            elif provider is not None:
                resolved_config = get_circuit_breaker_config(provider=provider, agent_name=name)
            elif failure_threshold is not None or cooldown_seconds is not None:
                # Legacy parameters provided - use them with defaults for unset values
                base_config = get_circuit_breaker_config()
                resolved_config = base_config.with_overrides(
                    failure_threshold=failure_threshold,
                    timeout_seconds=cooldown_seconds,
                )
            else:
                # Use default config (respects environment variables)
                resolved_config = get_circuit_breaker_config()

            _circuit_breakers[name] = CircuitBreaker.from_config(
                config=resolved_config,
                name=name,
            )
            logger.debug(f"Created circuit breaker: {name} with config: {resolved_config}")

        cb = _circuit_breakers[name]
        cb._last_accessed = time.time()  # Update access timestamp
        return cb


def reset_all_circuit_breakers() -> None:
    """Reset all global circuit breakers (thread-safe). Useful for testing."""
    with _circuit_breakers_lock:
        for cb in _circuit_breakers.values():
            cb.reset()
        count = len(_circuit_breakers)
    logger.info(f"Reset {count} circuit breakers")


def get_circuit_breakers() -> dict[str, CircuitBreaker]:
    """Get all registered circuit breakers (thread-safe).

    Returns:
        Dict mapping circuit breaker names to CircuitBreaker instances.
    """
    with _circuit_breakers_lock:
        return dict(_circuit_breakers)


def prune_circuit_breakers() -> int:
    """Manually prune stale circuit breakers from the registry.

    Removes circuit breakers not accessed within STALE_THRESHOLD_SECONDS (24h).

    Returns:
        Number of circuit breakers pruned.
    """
    with _circuit_breakers_lock:
        return _prune_stale_circuit_breakers()


__all__ = [
    "MAX_CIRCUIT_BREAKERS",
    "STALE_THRESHOLD_SECONDS",
    "get_circuit_breaker",
    "reset_all_circuit_breakers",
    "get_circuit_breakers",
    "prune_circuit_breakers",
    # Internal but needed by other modules
    "_circuit_breakers",
    "_circuit_breakers_lock",
]
