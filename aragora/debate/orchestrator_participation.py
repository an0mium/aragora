"""User participation lifecycle helpers for Arena debates.

Extracted from orchestrator.py to reduce its size. These functions handle
AudienceManager initialization and EventBus setup for pub/sub event handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aragora.debate.audience_manager import AudienceManager
from aragora.debate.event_bus import EventBus

if TYPE_CHECKING:
    from aragora.debate.orchestrator import Arena


def init_user_participation(arena: Arena) -> None:
    """Initialize user participation tracking and event subscription.

    Sets up the AudienceManager with loop scoping and notification callbacks,
    then subscribes it to the event emitter if available.

    Args:
        arena: Arena instance to initialize.
    """
    arena.audience_manager = AudienceManager(
        loop_id=arena.loop_id,
        strict_loop_scoping=arena.strict_loop_scoping,
    )
    arena.audience_manager.set_notify_callback(arena._notify_spectator)
    if arena.event_emitter:
        arena.audience_manager.subscribe_to_emitter(arena.event_emitter)


def init_event_bus(arena: Arena) -> None:
    """Initialize EventBus for pub/sub event handling.

    Creates the EventBus with connections to the event bridge, audience manager,
    immune system, and spectator for unified event distribution.

    Args:
        arena: Arena instance to initialize.
    """
    arena.event_bus = EventBus(
        event_bridge=arena.event_bridge,
        audience_manager=arena.audience_manager,
        immune_system=arena.immune_system,
        spectator=arena.spectator,
    )


__all__ = [
    "init_user_participation",
    "init_event_bus",
]
