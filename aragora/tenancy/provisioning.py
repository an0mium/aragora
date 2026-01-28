"""
Tenant provisioning and lifecycle management.

Provides tenant creation, upgrade, suspension, and deletion capabilities.

Usage:
    from aragora.tenancy.provisioning import TenantProvisioner

    provisioner = TenantProvisioner()
    tenant = await provisioner.create_tenant(
        name="Acme Corp",
        domain="acme.com",
        admin_email="admin@acme.com",
    )
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from aragora.tenancy.tenant import (
    Tenant,
    TenantConfig,
    TenantStatus,
    TenantTier,
)

logger = logging.getLogger(__name__)


class TenantProvisioner:
    """
    Manages tenant provisioning and lifecycle.

    Handles creation, tier upgrades, suspension, and deletion of tenants.
    """

    def __init__(self):
        """Initialize the provisioner."""
        self._tenants: dict[str, Tenant] = {}

    async def create_tenant(
        self,
        name: str,
        domain: str,
        tier: TenantTier = TenantTier.FREE,
        admin_email: Optional[str] = None,
        **kwargs,
    ) -> Tenant:
        """Create a new tenant.

        Args:
            name: Display name for the tenant
            domain: Primary domain for the tenant
            tier: Subscription tier
            admin_email: Admin contact email (defaults to admin@domain)
            **kwargs: Additional tenant attributes

        Returns:
            The created tenant
        """
        if admin_email is None:
            admin_email = f"admin@{domain}"

        # Create tenant with tier-appropriate config
        config = TenantConfig.for_tier(tier)

        tenant = Tenant.create(
            name=name,
            owner_email=admin_email,
            tier=tier,
        )
        tenant.config = config

        # Store tenant
        self._tenants[tenant.id] = tenant

        logger.info(f"Created tenant: {tenant.id} ({name}) tier={tier.value}")
        return tenant

    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get a tenant by ID.

        Args:
            tenant_id: The tenant ID

        Returns:
            The tenant, or None if not found
        """
        return self._tenants.get(tenant_id)

    async def upgrade_tier(
        self,
        tenant_id: str,
        new_tier: TenantTier,
    ) -> Tenant:
        """Upgrade a tenant's subscription tier.

        Args:
            tenant_id: The tenant ID
            new_tier: The new subscription tier

        Returns:
            The updated tenant

        Raises:
            KeyError: If tenant not found
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            raise KeyError(f"Tenant not found: {tenant_id}")

        old_tier = tenant.tier
        tenant.tier = new_tier
        tenant.config = TenantConfig.for_tier(new_tier)
        tenant.updated_at = datetime.now()

        logger.info(f"Upgraded tenant {tenant_id}: {old_tier.value} -> {new_tier.value}")
        return tenant

    async def suspend_tenant(
        self,
        tenant_id: str,
        reason: str = "",
    ) -> Tenant:
        """Suspend a tenant account.

        Args:
            tenant_id: The tenant ID
            reason: Reason for suspension

        Returns:
            The updated tenant

        Raises:
            KeyError: If tenant not found
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            raise KeyError(f"Tenant not found: {tenant_id}")

        tenant.status = TenantStatus.SUSPENDED
        tenant.updated_at = datetime.now()

        logger.warning(f"Suspended tenant {tenant_id}: {reason}")
        return tenant

    async def activate_tenant(self, tenant_id: str) -> Tenant:
        """Activate a suspended tenant.

        Args:
            tenant_id: The tenant ID

        Returns:
            The updated tenant

        Raises:
            KeyError: If tenant not found
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            raise KeyError(f"Tenant not found: {tenant_id}")

        tenant.status = TenantStatus.ACTIVE
        tenant.updated_at = datetime.now()

        logger.info(f"Activated tenant {tenant_id}")
        return tenant

    async def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant.

        Args:
            tenant_id: The tenant ID

        Returns:
            True if deleted, False if not found
        """
        if tenant_id in self._tenants:
            del self._tenants[tenant_id]
            logger.info(f"Deleted tenant {tenant_id}")
            return True
        return False

    def list_tenants(
        self,
        status: Optional[TenantStatus] = None,
        tier: Optional[TenantTier] = None,
    ) -> list[Tenant]:
        """List tenants with optional filtering.

        Args:
            status: Filter by status
            tier: Filter by tier

        Returns:
            List of matching tenants
        """
        tenants = list(self._tenants.values())

        if status:
            tenants = [t for t in tenants if t.status == status]
        if tier:
            tenants = [t for t in tenants if t.tier == tier]

        return tenants
