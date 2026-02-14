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
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if org_id:
            params["org_id"] = org_id

        return self._client.request("GET", "/api/v1/admin/users", params=params)

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


    # ===========================================================================
    # Diagnostics
    # ===========================================================================

    def get_handler_diagnostics(self) -> dict[str, Any]:
        """
        Get handler diagnostics information.

        GET /api/v1/diagnostics/handlers

        Returns:
            Dict with handler diagnostics
        """
        return self._client.request("GET", "/api/v1/diagnostics/handlers")


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

    async def list_users(
        self,
        limit: int = 20,
        offset: int = 0,
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """List all users with pagination."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if org_id:
            params["org_id"] = org_id

        return await self._client.request("GET", "/api/v1/admin/users", params=params)

    # ===========================================================================
    # Platform Statistics
    # ===========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get platform-wide statistics."""
        return await self._client.request("GET", "/api/v1/admin/stats")

    async def get_revenue(self) -> dict[str, Any]:
        """Get revenue analytics."""
        return await self._client.request("GET", "/api/v1/admin/revenue")

    # ===========================================================================
    # Nomic Loop Control
    # ===========================================================================

    async def get_nomic_status(self) -> dict[str, Any]:
        """Get the current Nomic loop status."""
        return await self._client.request("GET", "/api/v1/admin/nomic/status")

    async def reset_nomic(self) -> dict[str, Any]:
        """Reset the Nomic loop to initial state."""
        return await self._client.request("POST", "/api/v1/admin/nomic/reset")

    async def pause_nomic(self) -> dict[str, Any]:
        """Pause the Nomic loop."""
        return await self._client.request("POST", "/api/v1/admin/nomic/pause")

    async def resume_nomic(self) -> dict[str, Any]:
        """Resume a paused Nomic loop."""
        return await self._client.request("POST", "/api/v1/admin/nomic/resume")

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

    # ===========================================================================
    # Diagnostics
    # ===========================================================================

    async def get_handler_diagnostics(self) -> dict[str, Any]:
        """Get handler diagnostics information. GET /api/v1/diagnostics/handlers"""
        return await self._client.request("GET", "/api/v1/diagnostics/handlers")
