"""Domain-free cross-subscriber registry (P4a EventBus inversion, E1).

Holds cross-subsystem event subscribers keyed by name so that domain,
application, and interface *home modules* can self-register their subscribers at
import time. Registration is decoupled from dispatch: a subscriber wires its own
handlers into the manager via ``manager.register(...)`` when applied, so neither
this registry nor the manager needs to import any domain code.

This module carries ZERO domain imports (eager OR lazy) - it only stores and
returns opaque subscriber objects. Registration is keyed by name and idempotent,
so repeated bootstrap calls are no-ops. See
docs/architecture/P4A_EVENTS_QUEUE_INVERSION.md §4.1 and §4.4.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .manager import CrossSubscriberManager


@runtime_checkable
class CrossSubscriber(Protocol):
    """A cross-subsystem event subscriber.

    A subscriber owns its own reactions and wires them into the manager via
    ``manager.register(name, event_type, handler)`` when applied. Keeping the
    wiring on the subscriber lets home modules own their reactions without the
    registry (or the manager) importing any domain code.
    """

    def register(self, manager: CrossSubscriberManager) -> None: ...


_SUBSCRIBERS: dict[str, CrossSubscriber] = {}
_FACTORIES: dict[str, Callable[[], CrossSubscriber]] = {}


def register_subscriber(name: str, subscriber: CrossSubscriber) -> None:
    """Register a subscriber instance under ``name`` (keyed, idempotent)."""
    _SUBSCRIBERS[name] = subscriber
    _FACTORIES.pop(name, None)


def register_factory(name: str, factory: Callable[[], CrossSubscriber]) -> None:
    """Register a zero-arg factory that lazily builds a subscriber.

    The factory is invoked at most once, the first time the subscriber is
    materialized by :func:`get_registered_subscribers`.
    """
    if name not in _SUBSCRIBERS:
        _FACTORIES[name] = factory


def get_registered_subscribers() -> dict[str, CrossSubscriber]:
    """Return all registered subscribers, materializing factories once."""
    for name, factory in list(_FACTORIES.items()):
        if name not in _SUBSCRIBERS:
            _SUBSCRIBERS[name] = factory()
        _FACTORIES.pop(name, None)
    return dict(_SUBSCRIBERS)


def registered_subscriber_names() -> list[str]:
    """Return the sorted names of all registered subscribers and factories."""
    return sorted(set(_SUBSCRIBERS) | set(_FACTORIES))


def reset_registry() -> None:
    """Clear the registry (for tests)."""
    _SUBSCRIBERS.clear()
    _FACTORIES.clear()
