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

    def list_circuit_breakers(self) -> dict[str, Any]:
        """List all circuit breaker states."""
        return self._client.request("GET", "/api/v1/admin/circuit-breakers")

    def reset_circuit_breakers(self) -> dict[str, Any]:
        """Reset all circuit breakers."""
        return self._client.request("POST", "/api/v1/admin/circuit-breakers/reset")

    def get_security_audit(self) -> dict[str, Any]:
        """Get security audit report."""
        return self._client.request("GET", "/api/v1/admin/security/audit")

    def get_security_compliance(self) -> dict[str, Any]:
        """Get security compliance status."""
        return self._client.request("GET", "/api/v1/admin/security/compliance")

    def run_security_scan(self) -> dict[str, Any]:
        """Run a security scan."""
        return self._client.request("POST", "/api/v1/admin/security/scan")

    def list_security_threats(self) -> dict[str, Any]:
        """List detected security threats."""
        return self._client.request("GET", "/api/v1/admin/security/threats")

    # ===========================================================================
    # Emergency Access
    # ===========================================================================

    def activate_emergency(
        self,
        user_id: str,
        reason: str,
        duration_minutes: int = 60,
    ) -> dict[str, Any]:
        """Activate break-glass emergency access.

        Args:
            user_id: User to grant emergency access.
            reason: Reason for emergency access (min 10 chars).
            duration_minutes: Duration in minutes (default 60, max 1440).

        Returns:
            Dict with activation status and session details.
        """
        return self._client.request(
            "POST",
            "/api/v1/admin/emergency/activate",
            json={
                "user_id": user_id,
                "reason": reason,
                "duration_minutes": duration_minutes,
            },
        )

    def deactivate_emergency(self, session_id: str | None = None) -> dict[str, Any]:
        """Deactivate break-glass emergency access.

        Args:
            session_id: Optional session ID to deactivate.

        Returns:
            Dict with deactivation status.
        """
        data: dict[str, Any] = {}
        if session_id:
            data["session_id"] = session_id
        return self._client.request(
            "POST",
            "/api/v1/admin/emergency/deactivate",
            json=data if data else None,
        )

    def get_emergency_status(self) -> dict[str, Any]:
        """Get active emergency access sessions.

        Returns:
            Dict with active emergency sessions and counts.
        """
        return self._client.request("GET", "/api/v1/admin/emergency/status")

    # ===========================================================================
    # Feature Flags
    # ===========================================================================

    def update_feature_flags(self, flags: dict[str, Any]) -> dict[str, Any]:
        """Update admin feature flags.

        Args:
            flags: Feature flag key-value pairs to update.

        Returns:
            Dict with updated flags.
        """
        return self._client.request("PUT", "/api/v1/admin/feature-flags", json=flags)


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

    async def list_circuit_breakers(self) -> dict[str, Any]:
        """List all circuit breaker states."""
        return await self._client.request("GET", "/api/v1/admin/circuit-breakers")

    async def reset_circuit_breakers(self) -> dict[str, Any]:
        """Reset all circuit breakers."""
        return await self._client.request("POST", "/api/v1/admin/circuit-breakers/reset")

    async def get_security_audit(self) -> dict[str, Any]:
        """Get security audit report."""
        return await self._client.request("GET", "/api/v1/admin/security/audit")

    async def get_security_compliance(self) -> dict[str, Any]:
        """Get security compliance status."""
        return await self._client.request("GET", "/api/v1/admin/security/compliance")

    async def run_security_scan(self) -> dict[str, Any]:
        """Run a security scan."""
        return await self._client.request("POST", "/api/v1/admin/security/scan")

    async def list_security_threats(self) -> dict[str, Any]:
        """List detected security threats."""
        return await self._client.request("GET", "/api/v1/admin/security/threats")

    # ===========================================================================
    # Emergency Access
    # ===========================================================================

    async def activate_emergency(
        self,
        user_id: str,
        reason: str,
        duration_minutes: int = 60,
    ) -> dict[str, Any]:
        """Activate break-glass emergency access."""
        return await self._client.request(
            "POST",
            "/api/v1/admin/emergency/activate",
            json={
                "user_id": user_id,
                "reason": reason,
                "duration_minutes": duration_minutes,
            },
        )

    async def deactivate_emergency(self, session_id: str | None = None) -> dict[str, Any]:
        """Deactivate break-glass emergency access."""
        data: dict[str, Any] = {}
        if session_id:
            data["session_id"] = session_id
        return await self._client.request(
            "POST",
            "/api/v1/admin/emergency/deactivate",
            json=data if data else None,
        )

    async def get_emergency_status(self) -> dict[str, Any]:
        """Get active emergency access sessions."""
        return await self._client.request("GET", "/api/v1/admin/emergency/status")

    # ===========================================================================
    # Feature Flags
    # ===========================================================================

    async def update_feature_flags(self, flags: dict[str, Any]) -> dict[str, Any]:
        """Update admin feature flags."""
        return await self._client.request("PUT", "/api/v1/admin/feature-flags", json=flags)
