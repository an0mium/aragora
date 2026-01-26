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

from typing import Optional

from aragora.events.subscribers.config import (
    AsyncDispatchConfig,
    RetryConfig,
    SubscriberStats,
)

from .manager import CrossSubscriberManager

# Global manager instance
_global_manager: Optional[CrossSubscriberManager] = None


def get_cross_subscriber_manager() -> CrossSubscriberManager:
    """Get or create the global cross-subscriber manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = CrossSubscriberManager()
    return _global_manager


def reset_cross_subscriber_manager() -> None:
    """Reset the global manager (for testing)."""
    global _global_manager
    _global_manager = None


__all__ = [
    "CrossSubscriberManager",
    "SubscriberStats",
    "RetryConfig",
    "AsyncDispatchConfig",
    "get_cross_subscriber_manager",
    "reset_cross_subscriber_manager",
]
