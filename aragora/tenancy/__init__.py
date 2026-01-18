"""
Multi-tenant isolation for Aragora enterprise deployments.

Provides:
- Tenant context management
- Data isolation
- Resource quotas
- Cross-tenant security

Usage:
    from aragora.tenancy import TenantContext, get_current_tenant

    with TenantContext(tenant_id="acme-corp"):
        # All operations scoped to this tenant
        result = await arena.run()
"""

from aragora.tenancy.context import (
    TenantContext,
    get_current_tenant,
    require_tenant,
    set_tenant,
)
from aragora.tenancy.isolation import (
    TenantDataIsolation,
    TenantIsolationConfig,
)
from aragora.tenancy.quotas import (
    QuotaConfig,
    QuotaManager,
    QuotaExceeded,
)
from aragora.tenancy.tenant import (
    Tenant,
    TenantConfig,
    TenantStatus,
    TenantTier,
)

__all__ = [
    # Context
    "TenantContext",
    "get_current_tenant",
    "require_tenant",
    "set_tenant",
    # Isolation
    "TenantDataIsolation",
    "TenantIsolationConfig",
    # Quotas
    "QuotaConfig",
    "QuotaExceeded",
    "QuotaManager",
    # Tenant
    "Tenant",
    "TenantConfig",
    "TenantStatus",
    "TenantTier",
]
