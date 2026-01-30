# Multi-Tenancy

Enterprise multi-tenant isolation for Aragora deployments.

## Overview

This package provides complete tenant isolation:

| Component | Module | Purpose |
|-----------|--------|---------|
| Context | `context.py` | Tenant context propagation |
| Isolation | `isolation.py` | Data isolation enforcement |
| Quotas | `quotas.py` | Resource quota management |
| Limits | `limits.py` | Rate and usage limits |
| Tenant | `tenant.py` | Tenant lifecycle management |
| Provisioning | `provisioning.py` | Automated tenant setup |

## Quick Start

```python
from aragora.tenancy import (
    TenantContext,
    get_current_tenant,
    TenantManager,
    QuotaManager,
)

# Context manager for tenant scoping
with TenantContext(tenant_id="acme-corp"):
    # All operations automatically scoped to this tenant
    result = await arena.run()

    # Get current tenant anywhere in call stack
    tenant = get_current_tenant()
    print(f"Running for: {tenant.name}")
```

## Tenant Tiers

```python
from aragora.tenancy import Tenant, TenantTier, TenantConfig

# Create tenant with tier
tenant = Tenant(
    id="acme-corp",
    name="Acme Corporation",
    tier=TenantTier.ENTERPRISE,
    config=TenantConfig(
        max_concurrent_debates=50,
        max_agents_per_debate=10,
        storage_quota_gb=100,
    ),
)

# Tier-based features
if tenant.tier >= TenantTier.BUSINESS:
    enable_advanced_analytics()
```

## Data Isolation

```python
from aragora.tenancy import TenantDataIsolation, TenantIsolationEnforcer

# Configure isolation
isolation = TenantDataIsolation(
    enforce_row_level_security=True,
    isolate_file_storage=True,
    separate_encryption_keys=True,
)

# Enforcer validates all data access
enforcer = TenantIsolationEnforcer(isolation)

@enforcer.require_tenant_scope
async def get_debates(tenant_id: str):
    # Automatically filtered to tenant's data
    return await db.query("SELECT * FROM debates")
```

## Quota Management

```python
from aragora.tenancy import QuotaManager, QuotaConfig, QuotaExceeded

# Configure quotas
config = QuotaConfig(
    max_debates_per_day=100,
    max_api_calls_per_hour=10000,
    max_storage_bytes=10 * 1024**3,  # 10 GB
)

quota_manager = QuotaManager(config)

# Check before operation
try:
    await quota_manager.check_and_increment("debates", tenant_id)
    result = await arena.run()
except QuotaExceeded as e:
    return error_response(f"Quota exceeded: {e.quota_name}")
```

## Tenant Lifecycle

```python
from aragora.tenancy import TenantManager, TenantStatus

manager = TenantManager()

# Create tenant
tenant = await manager.create_tenant(
    name="New Customer",
    tier=TenantTier.BUSINESS,
    admin_email="admin@newcustomer.com",
)

# Suspend tenant
await manager.update_status(tenant.id, TenantStatus.SUSPENDED)

# Delete tenant (with data cleanup)
await manager.delete_tenant(tenant.id, purge_data=True)
```

## Provisioning

```python
from aragora.tenancy import TenantProvisioner

provisioner = TenantProvisioner()

# Automated setup
await provisioner.provision(
    tenant_id="new-tenant",
    tier=TenantTier.ENTERPRISE,
    options={
        "create_default_users": True,
        "setup_integrations": True,
        "copy_templates": True,
    },
)
```

## Integration with RBAC

Tenancy works with the RBAC system:

```python
from aragora.rbac import require_permission
from aragora.tenancy import require_tenant

@require_tenant
@require_permission("debates:create")
async def create_debate(request):
    tenant = get_current_tenant()
    # User authenticated + tenant scoped + permission checked
    ...
```
