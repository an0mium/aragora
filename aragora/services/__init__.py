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

    # Get registry stats
    stats = ServiceRegistry.get().stats()

    # Graceful shutdown
    ServiceRegistry.get().shutdown()

    # For testing - reset all services
    ServiceRegistry.reset()
"""

from .registry import (
    RegistryStats,
    ServiceDescriptor,
    ServiceNotFoundError,
    ServiceRegistry,
    ServiceScope,
    get_service,
    has_service,
    register_service,
)

# =============================================================================
# Marker Types for Cache Services
# =============================================================================
# These marker types allow registering different cache instances with the
# ServiceRegistry while maintaining type safety.


class MethodCacheService:
    """Marker type for the global method cache (utils/cache.py)."""

    pass


class QueryCacheService:
    """Marker type for the global query cache (utils/cache.py)."""

    pass


class EmbeddingCacheService:
    """Marker type for the embedding cache (memory/embeddings.py)."""

    pass


class HandlerCacheService:
    """Marker type for the HTTP handler cache (server/handlers/base.py)."""

    pass


class EmbeddingProviderService:
    """Marker type for the embedding provider reference (memory/streams.py)."""

    pass


__all__ = [
    # Registry
    "ServiceRegistry",
    "ServiceNotFoundError",
    "ServiceScope",
    "ServiceDescriptor",
    "RegistryStats",
    "get_service",
    "register_service",
    "has_service",
    # Cache markers
    "MethodCacheService",
    "QueryCacheService",
    "EmbeddingCacheService",
    "HandlerCacheService",
    "EmbeddingProviderService",
]
