"""
Aragora Services Package.

Provides centralized service management via the ServiceRegistry pattern.
This replaces scattered global singletons with a unified, testable registry.

Usage:
    from aragora.services import ServiceRegistry, get_service, register_service

    # Register a service
    register_service(TierManager, my_tier_manager)

    # Get a service
    tier_manager = get_service(TierManager)

    # For testing - reset all services
    ServiceRegistry.reset()
"""

from .registry import (
    ServiceRegistry,
    ServiceNotFoundError,
    get_service,
    register_service,
    has_service,
)

__all__ = [
    "ServiceRegistry",
    "ServiceNotFoundError",
    "get_service",
    "register_service",
    "has_service",
]
