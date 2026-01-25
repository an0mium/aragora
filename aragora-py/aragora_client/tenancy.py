"""Tenancy API for the Aragora SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from aragora_client.client import AragoraClient


class Tenant(BaseModel):
    """Tenant model."""

    id: str
    name: str
    slug: str | None = None
    owner_id: str | None = None
    settings: dict[str, Any] = {}
    created_at: str | None = None
    updated_at: str | None = None


class CreateTenantRequest(BaseModel):
    """Create tenant request."""

    name: str
    slug: str | None = None
    settings: dict[str, Any] = {}


class UpdateTenantRequest(BaseModel):
    """Update tenant request."""

    name: str | None = None
    settings: dict[str, Any] | None = None


class QuotaStatus(BaseModel):
    """Quota status for a tenant."""

    debates_used: int = 0
    debates_limit: int = 100
    agents_used: int = 0
    agents_limit: int = 10
    storage_used_mb: float = 0.0
    storage_limit_mb: float = 1024.0
    api_calls_today: int = 0
    api_calls_limit: int = 10000


class QuotaUpdate(BaseModel):
    """Quota update request."""

    debates_limit: int | None = None
    agents_limit: int | None = None
    storage_limit_mb: float | None = None
    api_calls_limit: int | None = None


class TenantMember(BaseModel):
    """Tenant member."""

    user_id: str
    tenant_id: str
    role: str
    email: str | None = None
    name: str | None = None
    joined_at: str | None = None


class AddMemberRequest(BaseModel):
    """Add member request."""

    user_id: str | None = None
    email: str | None = None
    role: str = "member"


class TenancyAPI:
    """API for tenant management operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Tenant]:
        """List all tenants the current user has access to."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        data = await self._client._get("/api/v1/tenants", params=params)
        return [Tenant.model_validate(t) for t in data.get("tenants", [])]

    async def get(self, tenant_id: str) -> Tenant:
        """Get a tenant by ID."""
        data = await self._client._get(f"/api/v1/tenants/{tenant_id}")
        return Tenant.model_validate(data)

    async def create(
        self,
        name: str,
        *,
        slug: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> Tenant:
        """Create a new tenant."""
        request = CreateTenantRequest(
            name=name,
            slug=slug,
            settings=settings or {},
        )
        data = await self._client._post("/api/v1/tenants", request.model_dump())
        return Tenant.model_validate(data)

    async def update(
        self,
        tenant_id: str,
        *,
        name: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> Tenant:
        """Update a tenant."""
        request = UpdateTenantRequest(name=name, settings=settings)
        data = await self._client._patch(
            f"/api/v1/tenants/{tenant_id}", request.model_dump(exclude_none=True)
        )
        return Tenant.model_validate(data)

    async def delete(self, tenant_id: str) -> None:
        """Delete a tenant."""
        await self._client._delete(f"/api/v1/tenants/{tenant_id}")

    async def get_quotas(self, tenant_id: str) -> QuotaStatus:
        """Get quota status for a tenant."""
        data = await self._client._get(f"/api/v1/tenants/{tenant_id}/quotas")
        return QuotaStatus.model_validate(data)

    async def update_quotas(
        self,
        tenant_id: str,
        *,
        debates_limit: int | None = None,
        agents_limit: int | None = None,
        storage_limit_mb: float | None = None,
        api_calls_limit: int | None = None,
    ) -> QuotaStatus:
        """Update quota limits for a tenant."""
        request = QuotaUpdate(
            debates_limit=debates_limit,
            agents_limit=agents_limit,
            storage_limit_mb=storage_limit_mb,
            api_calls_limit=api_calls_limit,
        )
        data = await self._client._patch(
            f"/api/v1/tenants/{tenant_id}/quotas", request.model_dump(exclude_none=True)
        )
        return QuotaStatus.model_validate(data)

    async def list_members(
        self,
        tenant_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TenantMember]:
        """List members of a tenant."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        data = await self._client._get(
            f"/api/v1/tenants/{tenant_id}/members", params=params
        )
        return [TenantMember.model_validate(m) for m in data.get("members", [])]

    async def add_member(
        self,
        tenant_id: str,
        *,
        user_id: str | None = None,
        email: str | None = None,
        role: str = "member",
    ) -> TenantMember:
        """Add a member to a tenant."""
        request = AddMemberRequest(user_id=user_id, email=email, role=role)
        data = await self._client._post(
            f"/api/v1/tenants/{tenant_id}/members",
            request.model_dump(exclude_none=True),
        )
        return TenantMember.model_validate(data)

    async def remove_member(self, tenant_id: str, user_id: str) -> None:
        """Remove a member from a tenant."""
        await self._client._delete(f"/api/v1/tenants/{tenant_id}/members/{user_id}")

    async def update_member_role(
        self,
        tenant_id: str,
        user_id: str,
        role: str,
    ) -> TenantMember:
        """Update a member's role in a tenant."""
        data = await self._client._patch(
            f"/api/v1/tenants/{tenant_id}/members/{user_id}",
            {"role": role},
        )
        return TenantMember.model_validate(data)
