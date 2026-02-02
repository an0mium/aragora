"""
Handler registry for modular HTTP endpoint routing.

This module provides centralized initialization and routing for all modular
HTTP handlers. The HandlerRegistryMixin can be mixed into request handler
classes to add modular routing capabilities.

Features:
- O(1) exact path lookup via route index
- LRU cached prefix matching for dynamic routes
- Lazy handler initialization
- API versioning support (/api/v1/... paths)

Usage:
    class MyHandler(HandlerRegistryMixin, BaseHTTPRequestHandler):
        pass

NOTE: This module has been decomposed into the handler_registry/ package.
This file re-exports all public symbols for backward compatibility.
See handler_registry/ for the implementation:
- handler_registry/core.py: Core infrastructure
- handler_registry/debates.py: Debate handlers
- handler_registry/agents.py: Agent handlers
- handler_registry/memory.py: Memory/knowledge handlers
- handler_registry/analytics.py: Analytics/metrics handlers
- handler_registry/social.py: Social/chat handlers
- handler_registry/admin.py: Admin/enterprise handlers
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Re-export everything from the new package structure
from aragora.server.handler_registry import (
    HANDLER_REGISTRY,
    HANDLERS_AVAILABLE,
    HandlerRegistryMixin,
    HandlerValidationError,
    RouteIndex,
    get_route_index,
    validate_all_handlers,
    validate_handler_class,
    validate_handler_instance,
    validate_handlers_on_init,
)

if TYPE_CHECKING:
    from aragora.server.unified_server import UnifiedHandler  # noqa: F401

__all__ = [
    "HandlerRegistryMixin",
    "HANDLER_REGISTRY",
    "HANDLERS_AVAILABLE",
    "RouteIndex",
    "get_route_index",
    # Validation functions
    "HandlerValidationError",
    "validate_handler_class",
    "validate_handler_instance",
    "validate_all_handlers",
    "validate_handlers_on_init",
    # Note: UnifiedHandler is re-exported via __getattr__ for lazy loading
]


# Re-export UnifiedHandler from unified_server for backward compatibility
def __getattr__(name: str) -> Any:
    """Lazy import for UnifiedHandler to avoid circular import."""
    if name == "UnifiedHandler":
        from aragora.server.unified_server import UnifiedHandler

        return UnifiedHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
