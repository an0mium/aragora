"""
Tenants API resource for the Aragora client.

Provides methods for tenant management in multi-tenant deployments:
- Tenant CRUD operations
- Tenant settings and quotas
- Tenant isolation configuration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..client import AragoraClient

logger = logging.getLogger(__name__)


@dataclass
class TenantQuota:
    """Quota limits for a tenant."""

    debates_per_month: int = 1000
    users_per_org: int = 50
    storage_gb: int = 10
    api_calls_per_minute: int = 100
    concurrent_debates: int = 5
    knowledge_nodes: int = 10000


@dataclass
class TenantSettings:
    """Settings for a tenant."""

    allow_external_agents: bool = True
    enable_knowledge_sharing: bool = False
    data_retention_days: int = 365
    require_mfa: bool = False
    allowed_domains: List[str] = field(default_factory=list)
    custom_branding: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tenant:
    """A tenant in the multi-tenant system."""

    id: str
    name: str
    slug: str
    status: str  # active, suspended, pending
    tier: str  # free, pro, enterprise
    owner_id: str
    quotas: TenantQuota
    settings: TenantSettings
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TenantUsage:
    """Current usage statistics for a tenant."""

    tenant_id: str
    debates_this_month: int = 0
    active_users: int = 0
    storage_used_gb: float = 0.0
    api_calls_today: int = 0
    knowledge_nodes_count: int = 0
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


class TenantsAPI:
    """API interface for tenant management."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    # =========================================================================
    # Tenant CRUD
    # =========================================================================

    def list(
        self,
        status: Optional[str] = None,
        tier: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[Tenant], int]:
        """
        List all tenants (admin only).

        Args:
            status: Filter by status (active, suspended, pending).
            tier: Filter by tier (free, pro, enterprise).
            limit: Maximum number of tenants to return.
            offset: Offset for pagination.

        Returns:
            Tuple of (list of Tenant objects, total count).
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if tier:
            params["tier"] = tier

        response = self._client._get("/api/v1/tenants", params=params)
        tenants = [self._parse_tenant(t) for t in response.get("tenants", [])]
        return tenants, response.get("total", len(tenants))

    async def list_async(
        self,
        status: Optional[str] = None,
        tier: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[Tenant], int]:
        """Async version of list()."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if tier:
            params["tier"] = tier

        response = await self._client._get_async("/api/v1/tenants", params=params)
        tenants = [self._parse_tenant(t) for t in response.get("tenants", [])]
        return tenants, response.get("total", len(tenants))

    def get(self, tenant_id: str) -> Tenant:
        """
        Get tenant details.

        Args:
            tenant_id: The tenant ID.

        Returns:
            Tenant object.
        """
        response = self._client._get(f"/api/v1/tenants/{tenant_id}")
        return self._parse_tenant(response.get("tenant", response))

    async def get_async(self, tenant_id: str) -> Tenant:
        """Async version of get()."""
        response = await self._client._get_async(f"/api/v1/tenants/{tenant_id}")
        return self._parse_tenant(response.get("tenant", response))

    def create(
        self,
        name: str,
        slug: str,
        tier: str = "free",
        owner_id: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        quotas: Optional[Dict[str, Any]] = None,
    ) -> Tenant:
        """
        Create a new tenant.

        Args:
            name: Tenant name.
            slug: URL-friendly slug.
            tier: Pricing tier (free, pro, enterprise).
            owner_id: Owner user ID.
            settings: Initial settings.
            quotas: Custom quota overrides.

        Returns:
            Created Tenant object.
        """
        body: Dict[str, Any] = {
            "name": name,
            "slug": slug,
            "tier": tier,
        }
        if owner_id:
            body["owner_id"] = owner_id
        if settings:
            body["settings"] = settings
        if quotas:
            body["quotas"] = quotas

        response = self._client._post("/api/v1/tenants", body)
        return self._parse_tenant(response.get("tenant", response))

    async def create_async(
        self,
        name: str,
        slug: str,
        tier: str = "free",
        owner_id: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        quotas: Optional[Dict[str, Any]] = None,
    ) -> Tenant:
        """Async version of create()."""
        body: Dict[str, Any] = {
            "name": name,
            "slug": slug,
            "tier": tier,
        }
        if owner_id:
            body["owner_id"] = owner_id
        if settings:
            body["settings"] = settings
        if quotas:
            body["quotas"] = quotas

        response = await self._client._post_async("/api/v1/tenants", body)
        return self._parse_tenant(response.get("tenant", response))

    def update(
        self,
        tenant_id: str,
        name: Optional[str] = None,
        tier: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        quotas: Optional[Dict[str, Any]] = None,
    ) -> Tenant:
        """
        Update tenant details.

        Args:
            tenant_id: The tenant ID.
            name: New tenant name.
            tier: New pricing tier.
            settings: Updated settings (merged with existing).
            quotas: Updated quota overrides.

        Returns:
            Updated Tenant object.
        """
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if tier is not None:
            body["tier"] = tier
        if settings is not None:
            body["settings"] = settings
        if quotas is not None:
            body["quotas"] = quotas

        response = self._client._patch(f"/api/v1/tenants/{tenant_id}", body)
        return self._parse_tenant(response.get("tenant", response))

    async def update_async(
        self,
        tenant_id: str,
        name: Optional[str] = None,
        tier: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        quotas: Optional[Dict[str, Any]] = None,
    ) -> Tenant:
        """Async version of update()."""
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if tier is not None:
            body["tier"] = tier
        if settings is not None:
            body["settings"] = settings
        if quotas is not None:
            body["quotas"] = quotas

        response = await self._client._patch_async(f"/api/v1/tenants/{tenant_id}", body)
        return self._parse_tenant(response.get("tenant", response))

    def delete(self, tenant_id: str) -> bool:
        """
        Delete a tenant (admin only).

        Args:
            tenant_id: The tenant ID.

        Returns:
            True if successful.
        """
        self._client._delete(f"/api/v1/tenants/{tenant_id}")
        return True

    async def delete_async(self, tenant_id: str) -> bool:
        """Async version of delete()."""
        await self._client._delete_async(f"/api/v1/tenants/{tenant_id}")
        return True

    # =========================================================================
    # Tenant Status Management
    # =========================================================================

    def suspend(self, tenant_id: str, reason: Optional[str] = None) -> Tenant:
        """
        Suspend a tenant.

        Args:
            tenant_id: The tenant ID.
            reason: Suspension reason.

        Returns:
            Updated Tenant object.
        """
        body: Dict[str, Any] = {"action": "suspend"}
        if reason:
            body["reason"] = reason

        response = self._client._post(f"/api/v1/tenants/{tenant_id}/status", body)
        return self._parse_tenant(response.get("tenant", response))

    async def suspend_async(self, tenant_id: str, reason: Optional[str] = None) -> Tenant:
        """Async version of suspend()."""
        body: Dict[str, Any] = {"action": "suspend"}
        if reason:
            body["reason"] = reason

        response = await self._client._post_async(f"/api/v1/tenants/{tenant_id}/status", body)
        return self._parse_tenant(response.get("tenant", response))

    def activate(self, tenant_id: str) -> Tenant:
        """
        Activate a suspended tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            Updated Tenant object.
        """
        body = {"action": "activate"}
        response = self._client._post(f"/api/v1/tenants/{tenant_id}/status", body)
        return self._parse_tenant(response.get("tenant", response))

    async def activate_async(self, tenant_id: str) -> Tenant:
        """Async version of activate()."""
        body = {"action": "activate"}
        response = await self._client._post_async(f"/api/v1/tenants/{tenant_id}/status", body)
        return self._parse_tenant(response.get("tenant", response))

    # =========================================================================
    # Usage and Quotas
    # =========================================================================

    def get_usage(self, tenant_id: str) -> TenantUsage:
        """
        Get current usage statistics for a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            TenantUsage object.
        """
        response = self._client._get(f"/api/v1/tenants/{tenant_id}/usage")
        return self._parse_usage(response)

    async def get_usage_async(self, tenant_id: str) -> TenantUsage:
        """Async version of get_usage()."""
        response = await self._client._get_async(f"/api/v1/tenants/{tenant_id}/usage")
        return self._parse_usage(response)

    def update_quotas(self, tenant_id: str, quotas: Dict[str, Any]) -> TenantQuota:
        """
        Update quota limits for a tenant.

        Args:
            tenant_id: The tenant ID.
            quotas: Quota values to update.

        Returns:
            Updated TenantQuota object.
        """
        response = self._client._patch(f"/api/v1/tenants/{tenant_id}/quotas", quotas)
        return self._parse_quota(response.get("quotas", response))

    async def update_quotas_async(self, tenant_id: str, quotas: Dict[str, Any]) -> TenantQuota:
        """Async version of update_quotas()."""
        response = await self._client._patch_async(f"/api/v1/tenants/{tenant_id}/quotas", quotas)
        return self._parse_quota(response.get("quotas", response))

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_tenant(self, data: Dict[str, Any]) -> Tenant:
        """Parse tenant data into Tenant object."""
        created_at = None
        updated_at = None

        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        if data.get("updated_at"):
            try:
                updated_at = datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return Tenant(
            id=data.get("id", ""),
            name=data.get("name", ""),
            slug=data.get("slug", ""),
            status=data.get("status", "active"),
            tier=data.get("tier", "free"),
            owner_id=data.get("owner_id", ""),
            quotas=self._parse_quota(data.get("quotas", {})),
            settings=self._parse_settings(data.get("settings", {})),
            created_at=created_at,
            updated_at=updated_at,
            metadata=data.get("metadata", {}),
        )

    def _parse_quota(self, data: Dict[str, Any]) -> TenantQuota:
        """Parse quota data into TenantQuota object."""
        return TenantQuota(
            debates_per_month=data.get("debates_per_month", 1000),
            users_per_org=data.get("users_per_org", 50),
            storage_gb=data.get("storage_gb", 10),
            api_calls_per_minute=data.get("api_calls_per_minute", 100),
            concurrent_debates=data.get("concurrent_debates", 5),
            knowledge_nodes=data.get("knowledge_nodes", 10000),
        )

    def _parse_settings(self, data: Dict[str, Any]) -> TenantSettings:
        """Parse settings data into TenantSettings object."""
        return TenantSettings(
            allow_external_agents=data.get("allow_external_agents", True),
            enable_knowledge_sharing=data.get("enable_knowledge_sharing", False),
            data_retention_days=data.get("data_retention_days", 365),
            require_mfa=data.get("require_mfa", False),
            allowed_domains=data.get("allowed_domains", []),
            custom_branding=data.get("custom_branding", {}),
        )

    def _parse_usage(self, data: Dict[str, Any]) -> TenantUsage:
        """Parse usage data into TenantUsage object."""
        period_start = None
        period_end = None

        if data.get("period_start"):
            try:
                period_start = datetime.fromisoformat(data["period_start"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        if data.get("period_end"):
            try:
                period_end = datetime.fromisoformat(data["period_end"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return TenantUsage(
            tenant_id=data.get("tenant_id", ""),
            debates_this_month=data.get("debates_this_month", 0),
            active_users=data.get("active_users", 0),
            storage_used_gb=data.get("storage_used_gb", 0.0),
            api_calls_today=data.get("api_calls_today", 0),
            knowledge_nodes_count=data.get("knowledge_nodes_count", 0),
            period_start=period_start,
            period_end=period_end,
        )


__all__ = [
    "TenantsAPI",
    "Tenant",
    "TenantQuota",
    "TenantSettings",
    "TenantUsage",
]
