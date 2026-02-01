"""Lifecycle and cache management helpers for Arena debates.

Extracted from orchestrator.py to reduce its size. These functions handle
DebateStateCache, LifecycleManager, EventEmitter, and CheckpointOperations
initialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aragora.debate.checkpoint_ops import CheckpointOperations
from aragora.debate.event_emission import EventEmitter
from aragora.debate.lifecycle_manager import LifecycleManager
from aragora.debate.state_cache import DebateStateCache

if TYPE_CHECKING:
    from aragora.debate.orchestrator import Arena


def init_caches(arena: Arena) -> None:
    """Initialize caches for computed values.

    Creates the DebateStateCache for caching debate state computations.

    Args:
        arena: Arena instance to initialize.
    """
    arena._cache = DebateStateCache()


def init_lifecycle_manager(arena: Arena) -> None:
    """Initialize LifecycleManager for cleanup and task cancellation.

    Creates the LifecycleManager with references to the cache, circuit breaker,
    and checkpoint manager for coordinated lifecycle operations.

    Args:
        arena: Arena instance to initialize.
    """
    arena._lifecycle = LifecycleManager(
        cache=arena._cache,
        circuit_breaker=arena.circuit_breaker,
        checkpoint_manager=arena.checkpoint_manager,
    )


def init_event_emitter(arena: Arena) -> None:
    """Initialize EventEmitter for spectator/websocket events.

    Creates the EventEmitter with connections to event bus, event bridge,
    hooks, and persona manager for broadcasting debate events.

    Args:
        arena: Arena instance to initialize.
    """
    arena._event_emitter = EventEmitter(
        event_bus=arena.event_bus,
        event_bridge=arena.event_bridge,
        hooks=arena.hooks,
        persona_manager=arena.persona_manager,
    )


def init_checkpoint_ops(arena: Arena) -> None:
    """Initialize CheckpointOperations for checkpoint and memory operations.

    Creates the CheckpointOperations helper. Note: memory_manager is set to None
    initially and should be updated after _init_phases when memory_manager exists.

    Args:
        arena: Arena instance to initialize.
    """
    arena._checkpoint_ops = CheckpointOperations(
        checkpoint_manager=arena.checkpoint_manager,
        memory_manager=None,  # Set after _init_phases when memory_manager exists
        cache=arena._cache,
    )


__all__ = [
    "init_caches",
    "init_lifecycle_manager",
    "init_event_emitter",
    "init_checkpoint_ops",
]
