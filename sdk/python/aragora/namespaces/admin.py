"""
Admin Namespace API

Provides methods for platform administration operations.
Requires admin role for all operations.

Features:
- Organization and user management
- Platform statistics and system metrics
- Nomic loop control
- Credit management
- Security operations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class AdminAPI:
    """
    Synchronous Admin API.

    Provides methods for platform administration:
    - Organization and user management
    - Platform statistics and system metrics
    - Revenue analytics
    - Nomic loop control
    - Credit management
    - Security operations

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="admin-key")
        >>> stats = client.admin.get_stats()
        >>> print(f"{stats['total_organizations']} orgs, {stats['active_debates']} debates")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Organizations and Users
    # ===========================================================================

    def list_organizations(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List all organizations with pagination.

        Args:
            limit: Maximum number of organizations to return
            offset: Number of organizations to skip

        Returns:
            Dict with organizations list, total count, and pagination info
        """
        return self._client.request(
            "GET",
            "/api/v1/admin/organizations",
            params={"limit": limit, "offset": offset},
        )

    def get_organization(self, org_id: str) -> dict[str, Any]:
        """
        Get organization details.

        Args:
            org_id: Organization ID

        Returns:
            Organization details
        """
        return self._client.request("GET", f"/api/v1/admin/organizations/{org_id}")

    def update_organization(
        self,
        org_id: str,
        name: str | None = None,
        status: str | None = None,
        plan: str | None = None,
    ) -> dict[str, Any]:
        """
        Update organization details.

        Args:
            org_id: Organization ID
            name: New organization name
            status: New status (active, suspended, pending)
            plan: New plan

        Returns:
            Updated organization
        """
        data = {}
        if name is not None:
            data["name"] = name
        if status is not None:
            data["status"] = status
        if plan is not None:
            data["plan"] = plan

        return self._client.request("PATCH", f"/api/v1/admin/organizations/{org_id}", json=data)

    def list_users(
        self,
        limit: int = 20,
        offset: int = 0,
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """
        List all users with pagination.

        Args:
            limit: Maximum number of users to return
            offset: Number of users to skip
            org_id: Filter by organization ID

        Returns:
            Dict with users list, total count, and pagination info
        """
        params = {"limit": limit, "offset": offset}
        if org_id:
            params["org_id"] = org_id

        return self._client.request("GET", "/api/v1/admin/users", params=params)

    def get_user(self, user_id: str) -> dict[str, Any]:
        """
        Get user details.

        Args:
            user_id: User ID

        Returns:
            User details
        """
        return self._client.request("GET", f"/api/v1/admin/users/{user_id}")

    def suspend_user(self, user_id: str, reason: str | None = None) -> dict[str, Any]:
        """
        Suspend a user account.

        Args:
            user_id: User ID
            reason: Reason for suspension

        Returns:
            Updated user status
        """
        data = {}
        if reason:
            data["reason"] = reason

        return self._client.request("POST", f"/api/v1/admin/users/{user_id}/suspend", json=data)

    def activate_user(self, user_id: str) -> dict[str, Any]:
        """
        Activate a suspended user account.

        Args:
            user_id: User ID

        Returns:
            Updated user status
        """
        return self._client.request("POST", f"/api/v1/admin/users/{user_id}/activate")

    def impersonate_user(self, user_id: str) -> dict[str, Any]:
        """
        Get an impersonation token for a user (for support/debugging).

        Args:
            user_id: User ID to impersonate

        Returns:
            Dict with token, expires_at, user_id, and user_email
        """
        return self._client.request("POST", f"/api/v1/admin/users/{user_id}/impersonate")

    # ===========================================================================
    # Platform Statistics
    # ===========================================================================

    def get_stats(self) -> dict[str, Any]:
        """
        Get platform-wide statistics.

        Returns:
            Dict with total_organizations, total_users, active_debates, etc.
        """
        return self._client.request("GET", "/api/v1/admin/stats")

    def get_system_metrics(self) -> dict[str, Any]:
        """
        Get real-time system metrics.

        Returns:
            Dict with cpu_usage, memory_usage, disk_usage, etc.
        """
        return self._client.request("GET", "/api/v1/admin/metrics")

    def get_revenue(self) -> dict[str, Any]:
        """
        Get revenue analytics.

        Returns:
            Dict with mrr, arr, revenue_this_month, growth_rate, etc.
        """
        return self._client.request("GET", "/api/v1/admin/revenue")

    # ===========================================================================
    # Nomic Loop Control
    # ===========================================================================

    def get_nomic_status(self) -> dict[str, Any]:
        """
        Get the current Nomic loop status.

        Returns:
            Dict with running, current_phase, current_cycle, health, etc.
        """
        return self._client.request("GET", "/api/v1/admin/nomic/status")

    def get_circuit_breakers(self) -> dict[str, Any]:
        """
        Get circuit breaker states.

        Returns:
            Dict with circuit_breakers list
        """
        return self._client.request("GET", "/api/v1/admin/circuit-breakers")

    def reset_nomic(self) -> dict[str, Any]:
        """
        Reset the Nomic loop to initial state.

        Returns:
            Dict with success status
        """
        return self._client.request("POST", "/api/v1/admin/nomic/reset")

    def pause_nomic(self) -> dict[str, Any]:
        """
        Pause the Nomic loop.

        Returns:
            Dict with success status
        """
        return self._client.request("POST", "/api/v1/admin/nomic/pause")

    def resume_nomic(self) -> dict[str, Any]:
        """
        Resume a paused Nomic loop.

        Returns:
            Dict with success status
        """
        return self._client.request("POST", "/api/v1/admin/nomic/resume")

    def reset_circuit_breakers(self) -> dict[str, Any]:
        """
        Reset all circuit breakers to closed state.

        Returns:
            Dict with success and reset_count
        """
        return self._client.request("POST", "/api/v1/admin/circuit-breakers/reset")

    # ===========================================================================
    # Credit Management
    # ===========================================================================

    def get_credit_account(self, org_id: str) -> dict[str, Any]:
        """
        Get credit account details for an organization.

        Args:
            org_id: Organization ID

        Returns:
            Dict with org_id, balance, lifetime_issued, lifetime_used, expires_at
        """
        return self._client.request("GET", f"/api/v1/admin/credits/{org_id}")

    def issue_credits(
        self,
        org_id: str,
        amount: int,
        reason: str,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        """
        Issue credits to an organization.

        Args:
            org_id: Organization ID
            amount: Number of credits to issue
            reason: Reason for issuing credits
            expires_at: Optional expiration date (ISO 8601 format)

        Returns:
            Updated credit account
        """
        data = {"amount": amount, "reason": reason}
        if expires_at:
            data["expires_at"] = expires_at

        return self._client.request("POST", f"/api/v1/admin/credits/{org_id}/issue", json=data)

    def adjust_credits(
        self,
        org_id: str,
        amount: int,
        reason: str,
    ) -> dict[str, Any]:
        """
        Adjust credit balance for an organization.

        Args:
            org_id: Organization ID
            amount: Amount to adjust (positive or negative)
            reason: Reason for adjustment

        Returns:
            Updated credit account
        """
        return self._client.request(
            "POST",
            f"/api/v1/admin/credits/{org_id}/adjust",
            json={"amount": amount, "reason": reason},
        )

    def list_credit_transactions(
        self,
        org_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List credit transactions for an organization.

        Args:
            org_id: Organization ID
            limit: Maximum number of transactions to return
            offset: Number of transactions to skip

        Returns:
            Dict with transactions list and total count
        """
        return self._client.request(
            "GET",
            f"/api/v1/admin/credits/{org_id}/transactions",
            params={"limit": limit, "offset": offset},
        )

    def get_expiring_credits(self, org_id: str) -> dict[str, Any]:
        """
        Get credits that are expiring soon for an organization.

        Args:
            org_id: Organization ID

        Returns:
            Dict with credits array (amount, expires_at)
        """
        return self._client.request("GET", f"/api/v1/admin/credits/{org_id}/expiring")

    # ===========================================================================
    # Security Operations
    # ===========================================================================

    def get_security_status(self) -> dict[str, Any]:
        """
        Get security status overview.

        Returns:
            Dict with encryption_enabled, mfa_enforcement, audit_logging, etc.
        """
        return self._client.request("GET", "/api/v1/admin/security/status")

    def get_security_health(self) -> dict[str, Any]:
        """
        Get security health check results.

        Returns:
            Dict with healthy status and checks map
        """
        return self._client.request("GET", "/api/v1/admin/security/health")

    def list_security_keys(self) -> dict[str, Any]:
        """
        List all security keys.

        Returns:
            Dict with keys array
        """
        return self._client.request("GET", "/api/v1/admin/security/keys")

    def rotate_security_key(self, key_type: str) -> dict[str, Any]:
        """
        Rotate a security key.

        Args:
            key_type: Type of key to rotate (api, encryption, signing)

        Returns:
            Dict with success and new_key_id
        """
        return self._client.request("POST", f"/api/v1/admin/security/keys/{key_type}/rotate")


class AsyncAdminAPI:
    """
    Asynchronous Admin API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     stats = await client.admin.get_stats()
        ...     print(f"Active debates: {stats['active_debates']}")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Organizations and Users
    # ===========================================================================

    async def list_organizations(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List all organizations with pagination."""
        return await self._client.request(
            "GET",
            "/api/v1/admin/organizations",
            params={"limit": limit, "offset": offset},
        )

    async def get_organization(self, org_id: str) -> dict[str, Any]:
        """Get organization details."""
        return await self._client.request("GET", f"/api/v1/admin/organizations/{org_id}")

    async def update_organization(
        self,
        org_id: str,
        name: str | None = None,
        status: str | None = None,
        plan: str | None = None,
    ) -> dict[str, Any]:
        """Update organization details."""
        data = {}
        if name is not None:
            data["name"] = name
        if status is not None:
            data["status"] = status
        if plan is not None:
            data["plan"] = plan

        return await self._client.request(
            "PATCH", f"/api/v1/admin/organizations/{org_id}", json=data
        )

    async def list_users(
        self,
        limit: int = 20,
        offset: int = 0,
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """List all users with pagination."""
        params = {"limit": limit, "offset": offset}
        if org_id:
            params["org_id"] = org_id

        return await self._client.request("GET", "/api/v1/admin/users", params=params)

    async def get_user(self, user_id: str) -> dict[str, Any]:
        """Get user details."""
        return await self._client.request("GET", f"/api/v1/admin/users/{user_id}")

    async def suspend_user(self, user_id: str, reason: str | None = None) -> dict[str, Any]:
        """Suspend a user account."""
        data = {}
        if reason:
            data["reason"] = reason

        return await self._client.request(
            "POST", f"/api/v1/admin/users/{user_id}/suspend", json=data
        )

    async def activate_user(self, user_id: str) -> dict[str, Any]:
        """Activate a suspended user account."""
        return await self._client.request("POST", f"/api/v1/admin/users/{user_id}/activate")

    async def impersonate_user(self, user_id: str) -> dict[str, Any]:
        """Get an impersonation token for a user."""
        return await self._client.request("POST", f"/api/v1/admin/users/{user_id}/impersonate")

    # ===========================================================================
    # Platform Statistics
    # ===========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get platform-wide statistics."""
        return await self._client.request("GET", "/api/v1/admin/stats")

    async def get_system_metrics(self) -> dict[str, Any]:
        """Get real-time system metrics."""
        return await self._client.request("GET", "/api/v1/admin/metrics")

    async def get_revenue(self) -> dict[str, Any]:
        """Get revenue analytics."""
        return await self._client.request("GET", "/api/v1/admin/revenue")

    # ===========================================================================
    # Nomic Loop Control
    # ===========================================================================

    async def get_nomic_status(self) -> dict[str, Any]:
        """Get the current Nomic loop status."""
        return await self._client.request("GET", "/api/v1/admin/nomic/status")

    async def get_circuit_breakers(self) -> dict[str, Any]:
        """Get circuit breaker states."""
        return await self._client.request("GET", "/api/v1/admin/circuit-breakers")

    async def reset_nomic(self) -> dict[str, Any]:
        """Reset the Nomic loop to initial state."""
        return await self._client.request("POST", "/api/v1/admin/nomic/reset")

    async def pause_nomic(self) -> dict[str, Any]:
        """Pause the Nomic loop."""
        return await self._client.request("POST", "/api/v1/admin/nomic/pause")

    async def resume_nomic(self) -> dict[str, Any]:
        """Resume a paused Nomic loop."""
        return await self._client.request("POST", "/api/v1/admin/nomic/resume")

    async def reset_circuit_breakers(self) -> dict[str, Any]:
        """Reset all circuit breakers to closed state."""
        return await self._client.request("POST", "/api/v1/admin/circuit-breakers/reset")

    # ===========================================================================
    # Credit Management
    # ===========================================================================

    async def get_credit_account(self, org_id: str) -> dict[str, Any]:
        """Get credit account details for an organization."""
        return await self._client.request("GET", f"/api/v1/admin/credits/{org_id}")

    async def issue_credits(
        self,
        org_id: str,
        amount: int,
        reason: str,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        """Issue credits to an organization."""
        data = {"amount": amount, "reason": reason}
        if expires_at:
            data["expires_at"] = expires_at

        return await self._client.request(
            "POST", f"/api/v1/admin/credits/{org_id}/issue", json=data
        )

    async def adjust_credits(
        self,
        org_id: str,
        amount: int,
        reason: str,
    ) -> dict[str, Any]:
        """Adjust credit balance for an organization."""
        return await self._client.request(
            "POST",
            f"/api/v1/admin/credits/{org_id}/adjust",
            json={"amount": amount, "reason": reason},
        )

    async def list_credit_transactions(
        self,
        org_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List credit transactions for an organization."""
        return await self._client.request(
            "GET",
            f"/api/v1/admin/credits/{org_id}/transactions",
            params={"limit": limit, "offset": offset},
        )

    async def get_expiring_credits(self, org_id: str) -> dict[str, Any]:
        """Get credits that are expiring soon for an organization."""
        return await self._client.request("GET", f"/api/v1/admin/credits/{org_id}/expiring")

    # ===========================================================================
    # Security Operations
    # ===========================================================================

    async def get_security_status(self) -> dict[str, Any]:
        """Get security status overview."""
        return await self._client.request("GET", "/api/v1/admin/security/status")

    async def get_security_health(self) -> dict[str, Any]:
        """Get security health check results."""
        return await self._client.request("GET", "/api/v1/admin/security/health")

    async def list_security_keys(self) -> dict[str, Any]:
        """List all security keys."""
        return await self._client.request("GET", "/api/v1/admin/security/keys")

    async def rotate_security_key(self, key_type: str) -> dict[str, Any]:
        """Rotate a security key."""
        return await self._client.request("POST", f"/api/v1/admin/security/keys/{key_type}/rotate")
