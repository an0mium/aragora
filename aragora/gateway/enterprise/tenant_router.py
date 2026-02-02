"""
Multi-Tenant Router for Enterprise Gateway.

This module provides backwards compatibility for imports from the original
tenant_router.py location. All functionality has been moved to the
routing/ subpackage.

Usage (backwards compatible):
    from aragora.gateway.enterprise.tenant_router import (
        TenantRouter,
        TenantRoutingConfig,
        EndpointConfig,
    )

New preferred import:
    from aragora.gateway.enterprise.routing import (
        TenantRouter,
        TenantRoutingConfig,
        EndpointConfig,
    )
"""

# Re-export everything from the routing package for backwards compatibility
from aragora.gateway.enterprise.routing import (
    # Exceptions
    TenantRoutingError,
    TenantNotFoundError,
    NoAvailableEndpointError,
    QuotaExceededError,
    CrossTenantAccessError,
    # Enums
    LoadBalancingStrategy,
    EndpointStatus,
    RoutingEventType,
    # Data classes
    EndpointConfig,
    EndpointHealth,
    TenantQuotas,
    QuotaStatus,
    TenantRoutingConfig,
    RoutingDecision,
    RoutingAuditEntry,
    # Core classes
    QuotaTracker,
    EndpointHealthTracker,
    TenantRouter,
    TenantRoutingContext,
)


__all__ = [
    # Exceptions
    "TenantRoutingError",
    "TenantNotFoundError",
    "NoAvailableEndpointError",
    "QuotaExceededError",
    "CrossTenantAccessError",
    # Enums
    "LoadBalancingStrategy",
    "EndpointStatus",
    "RoutingEventType",
    # Data classes
    "EndpointConfig",
    "EndpointHealth",
    "TenantQuotas",
    "QuotaStatus",
    "TenantRoutingConfig",
    "RoutingDecision",
    "RoutingAuditEntry",
    # Core classes
    "QuotaTracker",
    "EndpointHealthTracker",
    "TenantRouter",
    "TenantRoutingContext",
]
