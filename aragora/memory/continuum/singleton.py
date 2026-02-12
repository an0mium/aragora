"""
Singleton instance management for ContinuumMemory.

Provides global access to a shared ContinuumMemory instance for
cross-subsystem integration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aragora.types.protocols import EventEmitterProtocol
    from .core import ContinuumMemory

logger = logging.getLogger(__name__)

# Singleton instance for cross-subsystem access
_global_continuum_memory: ContinuumMemory | None = None


def get_continuum_memory(
    db_path: str | None = None,
    event_emitter: EventEmitterProtocol | None = None,
) -> ContinuumMemory:
    """Get the global ContinuumMemory singleton instance.

    Creates a new instance if one doesn't exist, or returns the existing one.
    Useful for cross-subsystem integration where modules need shared memory access.

    Args:
        db_path: Optional database path (only used on first call)
        event_emitter: Optional event emitter for cross-subsystem events

    Returns:
        ContinuumMemory singleton instance
    """
    global _global_continuum_memory

    if _global_continuum_memory is None:
        # Import here to avoid circular imports
        from .core import ContinuumMemory

        _global_continuum_memory = ContinuumMemory(
            db_path=db_path,
            event_emitter=event_emitter,
        )
        logger.debug("Created global ContinuumMemory instance")

    return _global_continuum_memory


def reset_continuum_memory() -> None:
    """Reset the global ContinuumMemory instance (for testing)."""
    global _global_continuum_memory
    if _global_continuum_memory:
        if hasattr(_global_continuum_memory, "close"):
            getattr(_global_continuum_memory, "close")()
    _global_continuum_memory = None


__all__ = [
    "get_continuum_memory",
    "reset_continuum_memory",
]
