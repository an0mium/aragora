"""
Cross-Subsystem Event Subscribers.

Handles event propagation between subsystems to enable cross-pollination:
- Memory → RLM: Retrieval patterns inform compression strategies
- Agent ELO → Debate: Performance updates team selection weights
- Knowledge → Memory: Index updates sync to memory insights

This module bridges the event system with subsystem-specific actions,
enabling loose coupling while maintaining data flow.

Usage:
    from aragora.events.cross_subscribers import (
        CrossSubscriberManager,
        get_cross_subscriber_manager,
    )

    # Initialize and connect to event stream
    manager = get_cross_subscriber_manager()
    manager.connect(event_emitter)

    # Subscribers automatically process relevant events
"""

from __future__ import annotations

import logging

from aragora.events.subscribers.config import (
    AsyncDispatchConfig,
    RetryConfig,
    SubscriberStats,
)
from aragora.observability.metrics import record_km_inbound_event

from .manager import CrossSubscriberManager
from .registry import (
    CrossSubscriber,
    get_registered_subscribers,
    register_factory,
    register_subscriber,
    registered_subscriber_names,
    reset_registry,
)

logger = logging.getLogger(__name__)

# Global manager instance
_global_manager: CrossSubscriberManager | None = None
_bootstrapped = False


def get_cross_subscriber_manager() -> CrossSubscriberManager:
    """Get or create the global cross-subscriber manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = CrossSubscriberManager()
    return _global_manager


def reset_cross_subscriber_manager() -> None:
    """Reset the global manager (for testing)."""
    global _global_manager, _bootstrapped
    _global_manager = None
    _bootstrapped = False


def bootstrap(*, include_names: set[str] | None = None) -> CrossSubscriberManager:
    """Ensure the manager exists and has applied every registered subscriber.

    Domain-free composition primitive: this imports NO domain code. Home modules
    (domain/application/interface) self-register at import time; the layered
    composition-root bootstraps
    (:func:`aragora.debate.event_subscribers.bootstrap_debate_event_subscribers`
    and
    :func:`aragora.server.startup.event_subscribers.bootstrap_event_subscribers`)
    import those home modules and then call this to wire them into the manager.
    Idempotent - repeated calls only apply newly registered subscribers.

    Args:
        include_names: Forwarded to
            :meth:`CrossSubscriberManager.apply_registered_subscribers` - see
            that method for why a subset bootstrap needs this to stay narrow
            regardless of what else has already been imported in-process.

    See docs/architecture/P4A_EVENTS_QUEUE_INVERSION.md §4.4.
    """
    global _bootstrapped
    manager = get_cross_subscriber_manager()
    newly_applied = manager.apply_registered_subscribers(include_names=include_names)
    count = len(manager.get_stats())
    if not _bootstrapped or newly_applied:
        logger.info(
            "cross-subscriber bootstrap: %d subscriber(s) registered (+%d newly applied)",
            count,
            newly_applied,
        )
        _bootstrapped = True
    else:
        logger.debug("cross-subscriber bootstrap: %d subscriber(s) already registered", count)
    return manager


__all__ = [
    "CrossSubscriber",
    "CrossSubscriberManager",
    "SubscriberStats",
    "RetryConfig",
    "AsyncDispatchConfig",
    "bootstrap",
    "get_cross_subscriber_manager",
    "get_registered_subscribers",
    "register_factory",
    "register_subscriber",
    "registered_subscriber_names",
    "reset_cross_subscriber_manager",
    "reset_registry",
    "record_km_inbound_event",
]
