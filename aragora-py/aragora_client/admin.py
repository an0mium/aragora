"""Admin API for platform administration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora_client.client import AragoraClient


class AdminAPI:
    """API for admin operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Organization Management
    # =========================================================================

    async def list_organizations(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        tier: str | None = None,
    ) -> dict[str, Any]:
        """List all organizations (admin only).

        Args:
            limit: Maximum results (default 50, max 100)
            offset: Pagination offset
            tier: Filter by tier (enterprise, pro, free)

        Returns:
            Organizations list with total count
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if tier:
            params["tier"] = tier
        return await self._client._get("/api/v1/admin/organizations", params=params)

    async def list_all_users(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        org_id: str | None = None,
        role: str | None = None,
        active_only: bool = False,
    ) -> dict[str, Any]:
        """List all users across the platform (admin only).

        Args:
            limit: Maximum results
            offset: Pagination offset
            org_id: Filter by organization
            role: Filter by role
            active_only: Only return active users

        Returns:
            Users list with total count
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if org_id:
            params["org_id"] = org_id
        if role:
            params["role"] = role
        if active_only:
            params["active_only"] = "true"
        return await self._client._get("/api/v1/admin/users", params=params)

    async def get_stats(self) -> dict[str, Any]:
        """Get platform-wide admin statistics.

        Returns:
            User counts, organization counts, usage metrics
        """
        return await self._client._get("/api/v1/admin/stats")

    async def get_system_metrics(self) -> dict[str, Any]:
        """Get system-level metrics (admin only).

        Returns:
            CPU, memory, request rates, error rates
        """
        return await self._client._get("/api/v1/admin/metrics")

    async def get_revenue_stats(self) -> dict[str, Any]:
        """Get revenue and billing statistics (admin only).

        Returns:
            MRR, ARR, churn rates, billing metrics
        """
        return await self._client._get("/api/v1/admin/revenue")

    # =========================================================================
    # User Actions
    # =========================================================================

    async def impersonate_user(self, user_id: str) -> dict[str, Any]:
        """Impersonate a user for support purposes (admin only).

        Args:
            user_id: User ID to impersonate

        Returns:
            Temporary session token for impersonation
        """
        return await self._client._post(
            f"/api/v1/admin/users/{user_id}/impersonate", {}
        )

    async def deactivate_user(self, user_id: str) -> dict[str, Any]:
        """Deactivate a user account (admin only).

        Args:
            user_id: User ID to deactivate

        Returns:
            Updated user status
        """
        return await self._client._post(f"/api/v1/admin/users/{user_id}/deactivate", {})

    async def activate_user(self, user_id: str) -> dict[str, Any]:
        """Activate a user account (admin only).

        Args:
            user_id: User ID to activate

        Returns:
            Updated user status
        """
        return await self._client._post(f"/api/v1/admin/users/{user_id}/activate", {})

    async def unlock_user(self, user_id: str) -> dict[str, Any]:
        """Unlock a locked user account (admin only).

        Args:
            user_id: User ID to unlock

        Returns:
            Updated user status
        """
        return await self._client._post(f"/api/v1/admin/users/{user_id}/unlock", {})

    # =========================================================================
    # Nomic Control
    # =========================================================================

    async def get_nomic_status(self) -> dict[str, Any]:
        """Get current Nomic loop status.

        Returns:
            Phase, progress, active cycles, error counts
        """
        return await self._client._get("/api/v1/admin/nomic/status")

    async def get_nomic_circuit_breakers(self) -> dict[str, Any]:
        """Get Nomic circuit breaker states.

        Returns:
            Circuit breaker statuses for each phase
        """
        return await self._client._get("/api/v1/admin/nomic/circuit-breakers")

    async def reset_nomic(
        self,
        *,
        target_phase: int | None = None,
        clear_errors: bool = False,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Reset the Nomic loop (admin only).

        Args:
            target_phase: Phase to reset to (0-4)
            clear_errors: Clear error counters
            reason: Reason for reset (audit log)

        Returns:
            Reset confirmation and new state
        """
        body: dict[str, Any] = {}
        if target_phase is not None:
            body["target_phase"] = target_phase
        if clear_errors:
            body["clear_errors"] = True
        if reason:
            body["reason"] = reason
        return await self._client._post("/api/v1/admin/nomic/reset", body)

    async def pause_nomic(self, *, reason: str | None = None) -> dict[str, Any]:
        """Pause the Nomic loop (admin only).

        Args:
            reason: Reason for pausing (audit log)

        Returns:
            Pause confirmation
        """
        body: dict[str, Any] = {}
        if reason:
            body["reason"] = reason
        return await self._client._post("/api/v1/admin/nomic/pause", body)

    async def resume_nomic(self, *, target_phase: int | None = None) -> dict[str, Any]:
        """Resume the Nomic loop (admin only).

        Args:
            target_phase: Optional phase to resume from

        Returns:
            Resume confirmation and new state
        """
        body: dict[str, Any] = {}
        if target_phase is not None:
            body["target_phase"] = target_phase
        return await self._client._post("/api/v1/admin/nomic/resume", body)

    async def reset_nomic_circuit_breakers(self) -> dict[str, Any]:
        """Reset all Nomic circuit breakers (admin only).

        Returns:
            Reset confirmation
        """
        return await self._client._post(
            "/api/v1/admin/nomic/circuit-breakers/reset", {}
        )

    # =========================================================================
    # Health Probes
    # =========================================================================

    async def healthz(self) -> dict[str, Any]:
        """Kubernetes liveness probe.

        Returns:
            Simple OK status for liveness check
        """
        return await self._client._get("/healthz")

    async def readyz(self) -> dict[str, Any]:
        """Kubernetes readiness probe.

        Returns:
            Readiness status with dependency checks
        """
        return await self._client._get("/readyz")

    async def get_health(self) -> dict[str, Any]:
        """Get basic health status.

        Returns:
            Service health with uptime
        """
        return await self._client._get("/api/v1/health")

    async def get_detailed_health(self) -> dict[str, Any]:
        """Get detailed health status.

        Returns:
            Health status for all subsystems
        """
        return await self._client._get("/api/v1/health/detailed")

    async def get_deep_health(self) -> dict[str, Any]:
        """Get deep health check with dependency verification.

        Returns:
            Comprehensive health including external dependencies
        """
        return await self._client._get("/api/v1/health/deep")

    async def get_stores_health(self) -> dict[str, Any]:
        """Get health status of all data stores.

        Returns:
            Health status for each store (DB, cache, vector, etc.)
        """
        return await self._client._get("/api/v1/health/stores")

    async def get_sync_health(self) -> dict[str, Any]:
        """Get sync subsystem health.

        Returns:
            Sync queue status, lag metrics
        """
        return await self._client._get("/api/v1/health/sync")

    async def get_circuits_health(self) -> dict[str, Any]:
        """Get circuit breaker health.

        Returns:
            Status of all circuit breakers
        """
        return await self._client._get("/api/v1/health/circuits")

    async def get_slow_debates_health(self) -> dict[str, Any]:
        """Get slow debates health status.

        Returns:
            Debates exceeding time thresholds
        """
        return await self._client._get("/api/v1/health/slow-debates")

    async def get_cross_pollination_health(self) -> dict[str, Any]:
        """Get cross-pollination subsystem health.

        Returns:
            Cross-pollination queue and processing status
        """
        return await self._client._get("/api/v1/health/cross-pollination")

    async def get_knowledge_mound_health(self) -> dict[str, Any]:
        """Get Knowledge Mound health status.

        Returns:
            KM store health, sync status, adapter states
        """
        return await self._client._get("/api/v1/health/knowledge-mound")

    async def get_decay_health(self) -> dict[str, Any]:
        """Get memory decay subsystem health.

        Returns:
            Decay scheduler status, pending decay counts
        """
        return await self._client._get("/api/v1/health/decay")

    async def get_startup_health(self) -> dict[str, Any]:
        """Get startup health status.

        Returns:
            Initialization status of all subsystems
        """
        return await self._client._get("/api/v1/health/startup")

    async def get_database_health(self) -> dict[str, Any]:
        """Get database health status.

        Returns:
            Connection pool status, query latencies
        """
        return await self._client._get("/api/v1/health/database")

    async def get_platform_health(self) -> dict[str, Any]:
        """Get platform-wide health summary.

        Returns:
            Aggregated health across all subsystems
        """
        return await self._client._get("/api/v1/health/platform")

    async def get_diagnostics(self) -> dict[str, Any]:
        """Get system diagnostics (admin only).

        Returns:
            Detailed diagnostic information for troubleshooting
        """
        return await self._client._get("/api/v1/admin/diagnostics")

    async def get_deployment_diagnostics(self) -> dict[str, Any]:
        """Get deployment diagnostics (admin only).

        Returns:
            Deployment info, version, config validation
        """
        return await self._client._get("/api/v1/admin/diagnostics/deployment")

    # =========================================================================
    # Security
    # =========================================================================

    async def get_security_status(self) -> dict[str, Any]:
        """Get security status overview (admin only).

        Returns:
            Security posture, recent events, threat indicators
        """
        return await self._client._get("/api/v1/admin/security/status")

    async def rotate_encryption_key(
        self,
        *,
        dry_run: bool = False,
        stores: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Rotate encryption keys (admin only).

        Args:
            dry_run: Preview changes without applying
            stores: Specific stores to rotate (default: all)
            force: Force rotation even if recent

        Returns:
            Rotation status and affected records count
        """
        body: dict[str, Any] = {"dry_run": dry_run, "force": force}
        if stores:
            body["stores"] = stores
        return await self._client._post("/api/v1/admin/security/rotate-key", body)

    async def get_security_health(self) -> dict[str, Any]:
        """Get security subsystem health.

        Returns:
            Encryption status, key ages, audit log health
        """
        return await self._client._get("/api/v1/health/security")

    async def list_encryption_keys(self) -> dict[str, Any]:
        """List encryption key metadata (admin only).

        Returns:
            Key IDs, creation dates, rotation status (no actual keys)
        """
        return await self._client._get("/api/v1/admin/security/keys")

    # =========================================================================
    # Credits
    # =========================================================================

    async def issue_credits(
        self,
        org_id: str,
        amount_cents: int,
        *,
        credit_type: str = "promotional",
        description: str = "",
        expires_days: int | None = None,
    ) -> dict[str, Any]:
        """Issue credits to an organization (admin only).

        Args:
            org_id: Organization ID
            amount_cents: Amount in cents
            credit_type: Type (promotional, support, refund)
            description: Reason for credits
            expires_days: Days until expiration (None = never)

        Returns:
            Credit transaction details
        """
        body: dict[str, Any] = {
            "org_id": org_id,
            "amount_cents": amount_cents,
            "type": credit_type,
            "description": description,
        }
        if expires_days is not None:
            body["expires_days"] = expires_days
        return await self._client._post("/api/v1/admin/credits/issue", body)

    async def get_credits(self, org_id: str) -> dict[str, Any]:
        """Get organization credit balance (admin only).

        Args:
            org_id: Organization ID

        Returns:
            Current balance, pending, available
        """
        return await self._client._get(f"/api/v1/admin/credits/{org_id}")

    async def list_credit_transactions(
        self,
        org_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List credit transactions for an organization.

        Args:
            org_id: Organization ID
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Transaction history
        """
        return await self._client._get(
            f"/api/v1/admin/credits/{org_id}/transactions",
            params={"limit": limit, "offset": offset},
        )

    async def adjust_credits(
        self,
        org_id: str,
        amount_cents: int,
        description: str,
    ) -> dict[str, Any]:
        """Adjust organization credits (admin only).

        Args:
            org_id: Organization ID
            amount_cents: Amount to adjust (positive or negative)
            description: Reason for adjustment

        Returns:
            Updated balance and transaction details
        """
        return await self._client._post(
            f"/api/v1/admin/credits/{org_id}/adjust",
            {"amount_cents": amount_cents, "description": description},
        )

    # =========================================================================
    # Dashboard
    # =========================================================================

    async def get_debate_dashboard(
        self,
        *,
        domain: str | None = None,
        limit: int = 10,
        hours: int = 24,
    ) -> dict[str, Any]:
        """Get debate dashboard overview (admin only).

        Args:
            domain: Filter by domain
            limit: Max debates to return
            hours: Time window in hours

        Returns:
            Active debates, recent completions, key metrics
        """
        params: dict[str, Any] = {"limit": limit, "hours": hours}
        if domain:
            params["domain"] = domain
        return await self._client._get("/api/v1/admin/dashboard/debates", params=params)

    async def get_dashboard_quality_metrics(self) -> dict[str, Any]:
        """Get quality metrics for admin dashboard.

        Returns:
            Consensus rates, argument quality, agent performance
        """
        return await self._client._get("/api/v1/admin/dashboard/quality")
