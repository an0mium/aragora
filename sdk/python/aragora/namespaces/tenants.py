"""
Tenants namespace for multi-tenancy management.

Provides API access to manage tenants, tenant isolation,
resource quotas, and tenant-level configuration.
"""

from typing import Any


class TenantsAPI:
    """Synchronous tenants API."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def list(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List tenants.

        Args:
            limit: Maximum number of tenants to return
            offset: Number of tenants to skip
            status: Filter by status (active, suspended, pending)

        Returns:
            List of tenant records
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        return self._client._request("GET", "/api/v1/tenants", params=params)

    def get(self, tenant_id: str) -> dict[str, Any]:
        """
        Get tenant details.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Tenant details
        """
        return self._client._request("GET", f"/api/v1/tenants/{tenant_id}")

    def create(
        self,
        name: str,
        slug: str,
        plan: str = "free",
        settings: dict[str, Any] | None = None,
        quotas: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a new tenant.

        Args:
            name: Tenant display name
            slug: Unique tenant slug
            plan: Subscription plan
            settings: Tenant settings
            quotas: Resource quotas

        Returns:
            Created tenant record
        """
        data: dict[str, Any] = {
            "name": name,
            "slug": slug,
            "plan": plan,
        }
        if settings:
            data["settings"] = settings
        if quotas:
            data["quotas"] = quotas

        return self._client._request("POST", "/api/v1/tenants", json=data)

    def update(
        self,
        tenant_id: str,
        name: str | None = None,
        plan: str | None = None,
        settings: dict[str, Any] | None = None,
        quotas: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Update a tenant.

        Args:
            tenant_id: Tenant identifier
            name: New tenant name
            plan: New subscription plan
            settings: Updated settings
            quotas: Updated quotas

        Returns:
            Updated tenant record
        """
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = name
        if plan is not None:
            data["plan"] = plan
        if settings is not None:
            data["settings"] = settings
        if quotas is not None:
            data["quotas"] = quotas

        return self._client._request("PATCH", f"/api/v1/tenants/{tenant_id}", json=data)

    def delete(self, tenant_id: str) -> dict[str, Any]:
        """
        Delete a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Deletion confirmation
        """
        return self._client._request("DELETE", f"/api/v1/tenants/{tenant_id}")

    def suspend(self, tenant_id: str, reason: str | None = None) -> dict[str, Any]:
        """
        Suspend a tenant.

        Args:
            tenant_id: Tenant identifier
            reason: Suspension reason

        Returns:
            Updated tenant
        """
        data: dict[str, Any] = {}
        if reason:
            data["reason"] = reason

        return self._client._request("POST", f"/api/v1/tenants/{tenant_id}/suspend", json=data)

    def reactivate(self, tenant_id: str) -> dict[str, Any]:
        """
        Reactivate a suspended tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Updated tenant
        """
        return self._client._request("POST", f"/api/v1/tenants/{tenant_id}/reactivate")

    def get_usage(self, tenant_id: str) -> dict[str, Any]:
        """
        Get tenant usage statistics.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Usage statistics
        """
        return self._client._request("GET", f"/api/v1/tenants/{tenant_id}/usage")

    def get_quotas(self, tenant_id: str) -> dict[str, Any]:
        """
        Get tenant quotas.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Quota configuration and current usage
        """
        return self._client._request("GET", f"/api/v1/tenants/{tenant_id}/quotas")

    def update_quotas(self, tenant_id: str, quotas: dict[str, Any]) -> dict[str, Any]:
        """
        Update tenant quotas.

        Args:
            tenant_id: Tenant identifier
            quotas: New quota values

        Returns:
            Updated quotas
        """
        return self._client._request("PUT", f"/api/v1/tenants/{tenant_id}/quotas", json=quotas)

    def list_members(self, tenant_id: str) -> list[dict[str, Any]]:
        """
        List tenant members.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of members
        """
        return self._client._request("GET", f"/api/v1/tenants/{tenant_id}/members")

    def invite_member(self, tenant_id: str, email: str, role: str = "member") -> dict[str, Any]:
        """
        Invite a member to the tenant.

        Args:
            tenant_id: Tenant identifier
            email: Email to invite
            role: Member role

        Returns:
            Invitation record
        """
        return self._client._request(
            "POST",
            f"/api/v1/tenants/{tenant_id}/members/invite",
            json={"email": email, "role": role},
        )


class AsyncTenantsAPI:
    """Asynchronous tenants API."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def list(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List tenants."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        return await self._client._request("GET", "/api/v1/tenants", params=params)

    async def get(self, tenant_id: str) -> dict[str, Any]:
        """Get tenant details."""
        return await self._client._request("GET", f"/api/v1/tenants/{tenant_id}")

    async def create(
        self,
        name: str,
        slug: str,
        plan: str = "free",
        settings: dict[str, Any] | None = None,
        quotas: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new tenant."""
        data: dict[str, Any] = {
            "name": name,
            "slug": slug,
            "plan": plan,
        }
        if settings:
            data["settings"] = settings
        if quotas:
            data["quotas"] = quotas

        return await self._client._request("POST", "/api/v1/tenants", json=data)

    async def update(
        self,
        tenant_id: str,
        name: str | None = None,
        plan: str | None = None,
        settings: dict[str, Any] | None = None,
        quotas: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update a tenant."""
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = name
        if plan is not None:
            data["plan"] = plan
        if settings is not None:
            data["settings"] = settings
        if quotas is not None:
            data["quotas"] = quotas

        return await self._client._request("PATCH", f"/api/v1/tenants/{tenant_id}", json=data)

    async def delete(self, tenant_id: str) -> dict[str, Any]:
        """Delete a tenant."""
        return await self._client._request("DELETE", f"/api/v1/tenants/{tenant_id}")

    async def suspend(self, tenant_id: str, reason: str | None = None) -> dict[str, Any]:
        """Suspend a tenant."""
        data: dict[str, Any] = {}
        if reason:
            data["reason"] = reason

        return await self._client._request(
            "POST", f"/api/v1/tenants/{tenant_id}/suspend", json=data
        )

    async def reactivate(self, tenant_id: str) -> dict[str, Any]:
        """Reactivate a suspended tenant."""
        return await self._client._request("POST", f"/api/v1/tenants/{tenant_id}/reactivate")

    async def get_usage(self, tenant_id: str) -> dict[str, Any]:
        """Get tenant usage statistics."""
        return await self._client._request("GET", f"/api/v1/tenants/{tenant_id}/usage")

    async def get_quotas(self, tenant_id: str) -> dict[str, Any]:
        """Get tenant quotas."""
        return await self._client._request("GET", f"/api/v1/tenants/{tenant_id}/quotas")

    async def update_quotas(self, tenant_id: str, quotas: dict[str, Any]) -> dict[str, Any]:
        """Update tenant quotas."""
        return await self._client._request(
            "PUT", f"/api/v1/tenants/{tenant_id}/quotas", json=quotas
        )

    async def list_members(self, tenant_id: str) -> list[dict[str, Any]]:
        """List tenant members."""
        return await self._client._request("GET", f"/api/v1/tenants/{tenant_id}/members")

    async def invite_member(
        self, tenant_id: str, email: str, role: str = "member"
    ) -> dict[str, Any]:
        """Invite a member to the tenant."""
        return await self._client._request(
            "POST",
            f"/api/v1/tenants/{tenant_id}/members/invite",
            json={"email": email, "role": role},
        )
