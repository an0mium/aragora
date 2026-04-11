"""
Handler registry - centralized handler list to avoid circular imports.

This module provides centralized access to ALL_HANDLERS and HANDLER_STABILITY
without causing circular imports when features.py needs to enumerate handlers.

Usage:
    # From features.py or other modules that need handler enumeration
    from aragora.server.handlers._registry import ALL_HANDLERS, HANDLER_STABILITY

    # The main __init__.py populates these after all handlers are imported
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aragora.config.stability import Stability

if TYPE_CHECKING:
    from aragora.server.handlers.base import BaseHandler

# Handler list - populated by __init__.py after imports complete
ALL_HANDLERS: list[type[BaseHandler]] = []

# Handler stability classifications - populated by __init__.py
HANDLER_STABILITY: dict[str, Stability] = {}


def get_handler_stability(handler_name: str) -> Stability:
    """Get the stability level for a handler.

    Args:
        handler_name: Handler class name (e.g., 'DebatesHandler')

    Returns:
        Stability level, defaults to EXPERIMENTAL if not classified
    """
    return HANDLER_STABILITY.get(handler_name, Stability.EXPERIMENTAL)


def get_all_handler_stability() -> dict[str, str]:
    """Get all handler stability levels as strings for API response.

    Registered handlers that do not have an explicit classification default to
    EXPERIMENTAL to match get_handler_stability().
    """
    all_stability = {
        handler.__name__: get_handler_stability(handler.__name__).value for handler in ALL_HANDLERS
    }
    for name, stability in HANDLER_STABILITY.items():
        all_stability.setdefault(name, stability.value)
    return all_stability


def register_handler(
    handler: type[BaseHandler],
    stability: Stability = Stability.EXPERIMENTAL,
) -> None:
    """Register a handler and its stability classification.

    Args:
        handler: The handler class to register.
        stability: Stability level for the handler.
    """
    if handler not in ALL_HANDLERS:
        ALL_HANDLERS.append(handler)
    HANDLER_STABILITY[handler.__name__] = stability


def reset_registry() -> None:
    """Clear all registered handlers and stability classifications.

    Intended for use in unit tests to isolate registry state between tests.
    """
    ALL_HANDLERS.clear()
    HANDLER_STABILITY.clear()


__all__ = [
    "ALL_HANDLERS",
    "HANDLER_STABILITY",
    "get_all_handler_stability",
    "get_handler_stability",
    "register_handler",
    "reset_registry",
]
