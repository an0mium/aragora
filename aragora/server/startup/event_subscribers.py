"""Superset bootstrap for cross-subsystem event subscribers (P4a E1).

Composition root for the *running product* (server startup, CLI). It imports the
domain, application, AND interface event-subscriber home modules so every reaction
self-registers, then wires them into the cross-subscriber manager. Interface may
import every layer, so this superset lives here (in ``interface``); it must NOT
live in ``events``/``debate``, which would recreate an upward import.

See docs/architecture/P4A_EVENTS_QUEUE_INVERSION.md §4.4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.events.cross_subscribers import CrossSubscriberManager


def bootstrap_event_subscribers() -> CrossSubscriberManager:
    """Import all event-subscriber home modules and wire them into the manager.

    Superset of the domain-subset bootstrap: it also imports application and
    interface home modules. Idempotent. E1 is a pure enabler: no home modules
    exist yet, so this currently composes the domain-subset bootstrap and ensures
    the manager is initialized. E2-E7 add application/interface
    ``import aragora.<pkg>.event_subscribers`` lines here (e.g. workflow,
    notifications, server) as handlers relocate to their home layers.

    Returns:
        The registry-backed cross-subscriber manager singleton.
    """
    from aragora.debate.event_subscribers import bootstrap_debate_event_subscribers

    # Application + interface home-module imports are added here by E2-E7, e.g.
    # ``import aragora.workflow.event_subscribers`` (a side-effect import for
    # registration; mark it with a noqa F401 to silence unused-import lint).
    return bootstrap_debate_event_subscribers()
