"""Domain-subset bootstrap for cross-subsystem event subscribers (P4a E1).

Composition root for the *domain* layer (Arena / pure-library debate). It imports
only domain home modules so their subscribers self-register, then wires them into
the cross-subscriber manager. Application/interface reactions (workflow, nomic,
server) register when *those* subsystems initialize; a pure-library debate with no
server/workflow engine simply has no such reactions (matching today's try/except
fallbacks).

This module lives in ``debate`` (domain) and imports only sibling-domain home
modules plus ``aragora.events.cross_subscribers`` (domain -> infrastructure,
downward = legal). It must NOT be promoted to the interface superset - that would
recreate an upward import. See docs/architecture/P4A_EVENTS_QUEUE_INVERSION.md §4.4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.events.cross_subscribers import CrossSubscriberManager


def bootstrap_debate_event_subscribers() -> CrossSubscriberManager:
    """Import domain event-subscriber home modules and wire them into the manager.

    Idempotent (registration is keyed by name). E1 is a pure enabler: no domain
    home modules exist yet, so this currently only ensures the manager is
    initialized. E2-E6 add ``import aragora.<domain>.event_subscribers`` lines
    here (e.g. knowledge/memory/debate/reasoning/ranking) as handlers relocate to
    their home layers.

    Returns:
        The registry-backed cross-subscriber manager singleton.
    """
    # Domain home-module imports are added here by E2-E6 as handlers relocate,
    # e.g. ``import aragora.knowledge.event_subscribers`` (a side-effect import
    # for registration; mark it with a noqa F401 to silence unused-import lint).
    from aragora.events.cross_subscribers import bootstrap

    return bootstrap()
