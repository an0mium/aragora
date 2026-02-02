"""
Multi-Tenant Routing Package for Enterprise Gateway.

This package provides comprehensive tenant routing capabilities including:
- Main TenantRouter for routing requests to tenant endpoints
- Quota management with sliding window rate limiting
- Data isolation and cross-tenant access control
- Specialized handlers for enterprise, team, and individual tenants

Usage:
    from aragora.gateway.enterprise.routing import (
        TenantRouter,
        TenantRoutingConfig,
        EndpointConfig,
        TenantQuotas,
    )

    # Configure and use the router
    router = TenantRouter(
        configs=[
            TenantRoutingConfig(
                tenant_id="acme-corp",
                endpoints=[
                    EndpointConfig(url="https://acme.api.example.com"),
                ],
            ),
        ]
    )

    decision = await router.route("acme-corp", request_data)
"""

# Router core
from .router import (
    # Exceptions
    TenantRoutingError,
    TenantNotFoundError,
    NoAvailableEndpointError,
    QuotaExceededError,
    # Enums
    LoadBalancingStrategy,
    EndpointStatus,
    RoutingEventType,
    # Data classes
    EndpointConfig,
    EndpointHealth,
    TenantRoutingConfig,
    RoutingDecision,
    RoutingAuditEntry,
    # Core classes
    EndpointHealthTracker,
    TenantRouter,
    TenantRoutingContext,
)

# Quota management
from .quotas import (
    TenantQuotas,
    QuotaStatus,
    QuotaTracker,
)

# Isolation
from .isolation import (
    CrossTenantAccessError,
    IsolationConfig,
    TenantAccessContext,
    TenantContextBuilder,
    TenantRoutingContextManager,
    validate_tenant_access,
    get_tenant_from_context,
)

# Enterprise tenant handling
from .enterprise import (
    SLATier,
    ComplianceRequirement,
    EnterpriseRoutingConfig,
    EnterpriseEndpoint,
    EnterpriseRoutingDecision,
    EnterpriseTenantHandler,
)

# Team tenant handling
from .team import (
    TeamPlan,
    MemberRole,
    TeamMember,
    TeamRoutingConfig,
    TeamEndpoint,
    TeamRoutingDecision,
    TeamTenantHandler,
)

# Individual tenant handling
from .individual import (
    IndividualPlan,
    AccountStatus,
    IndividualRoutingConfig,
    IndividualEndpoint,
    IndividualRoutingDecision,
    IndividualTenantHandler,
)


__all__ = [
    # Router exceptions
    "TenantRoutingError",
    "TenantNotFoundError",
    "NoAvailableEndpointError",
    "QuotaExceededError",
    "CrossTenantAccessError",
    # Router enums
    "LoadBalancingStrategy",
    "EndpointStatus",
    "RoutingEventType",
    # Router data classes
    "EndpointConfig",
    "EndpointHealth",
    "TenantRoutingConfig",
    "RoutingDecision",
    "RoutingAuditEntry",
    # Router core classes
    "EndpointHealthTracker",
    "TenantRouter",
    "TenantRoutingContext",
    "TenantRoutingContextManager",
    # Quota management
    "TenantQuotas",
    "QuotaStatus",
    "QuotaTracker",
    # Isolation
    "IsolationConfig",
    "TenantAccessContext",
    "TenantContextBuilder",
    "validate_tenant_access",
    "get_tenant_from_context",
    # Enterprise tenant
    "SLATier",
    "ComplianceRequirement",
    "EnterpriseRoutingConfig",
    "EnterpriseEndpoint",
    "EnterpriseRoutingDecision",
    "EnterpriseTenantHandler",
    # Team tenant
    "TeamPlan",
    "MemberRole",
    "TeamMember",
    "TeamRoutingConfig",
    "TeamEndpoint",
    "TeamRoutingDecision",
    "TeamTenantHandler",
    # Individual tenant
    "IndividualPlan",
    "AccountStatus",
    "IndividualRoutingConfig",
    "IndividualEndpoint",
    "IndividualRoutingDecision",
    "IndividualTenantHandler",
]
