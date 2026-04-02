"""
Security Namespace API.

Provides security status, health checks, and key management.

Features:
- Overall security status monitoring
- Security health checks
- Encryption key management
- Key rotation operations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

SecurityLevel = Literal["healthy", "degraded", "unhealthy"]
KeyStatus = Literal["active", "expired", "revoked"]
CheckStatus = Literal["ok", "warning", "error"]


class SecurityAPI:
    """
    Synchronous Security API.

    Provides methods for security management:
    - Get overall security status
    - Run security health checks
    - Manage encryption keys
    - Rotate keys for security maintenance

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> status = client.security.get_status()
        >>> if status.get("rotation_required"):
        ...     print("Active encryption key needs rotation")
        >>> health = client.security.get_health_checks()
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Security Status
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """
        Get encryption status and key rotation metadata.

        Returns:
            Dict with crypto availability, active key metadata,
            rotation recommendations, and total key count.
        """
        return self._client._request("GET", "/api/v1/admin/security/status")

    # =========================================================================
    # Health Checks
    # =========================================================================

    def get_health_checks(self) -> dict[str, Any]:
        """
        Get security health checks.

        Returns:
            Dict with overall status plus keyed checks, issues, and warnings.
        """
        return self._client._request("GET", "/api/v1/admin/security/health")

    def run_security_scan(self) -> dict[str, Any]:
        """
        Trigger a security scan.

        Initiates a comprehensive security scan of the system.

        Returns:
            Dict with scan ID and status.
        """
        return self._client._request("POST", "/api/v1/admin/security/scan")

    def get_scan_status(self, scan_id: str) -> dict[str, Any]:
        """
        Get the status of a security scan.

        Args:
            scan_id: The scan identifier.

        Returns:
            Dict with scan status, progress, and findings.
        """
        return self._client._request("GET", f"/api/v1/admin/security/scan/{scan_id}")

    # =========================================================================
    # Key Management
    # =========================================================================

    def list_keys(self) -> dict[str, Any]:
        """
        List all security keys.

        Returns:
            Dict with key summaries plus active_key_id and total_keys.
        """
        return self._client._request("GET", "/api/v1/admin/security/keys")

    def get_key(self, key_id: str) -> dict[str, Any]:
        """
        Get details of a specific key.

        Args:
            key_id: The key identifier.

        Returns:
            Dict with key details.
        """
        return self._client._request("GET", f"/api/v1/admin/security/keys/{key_id}")

    def create_key(
        self,
        name: str,
        algorithm: str = "AES-256-GCM",
        expires_in_days: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a new encryption key.

        Args:
            name: Name for the key.
            algorithm: Encryption algorithm (default: AES-256-GCM).
            expires_in_days: Optional expiration in days.
            metadata: Optional metadata.

        Returns:
            Dict with created key details.
        """
        data: dict[str, Any] = {
            "name": name,
            "algorithm": algorithm,
        }
        if expires_in_days is not None:
            data["expires_in_days"] = expires_in_days
        if metadata is not None:
            data["metadata"] = metadata

        return self._client._request("POST", "/api/v1/admin/security/keys", json=data)

    def revoke_key(self, key_id: str, reason: str | None = None) -> dict[str, Any]:
        """
        Revoke an encryption key.

        Args:
            key_id: The key identifier.
            reason: Optional reason for revocation.

        Returns:
            Dict confirming revocation.
        """
        data: dict[str, Any] = {}
        if reason is not None:
            data["reason"] = reason

        return self._client._request(
            "POST",
            f"/api/v1/admin/security/keys/{key_id}/revoke",
            json=data if data else None,
        )

    def rotate_key(
        self,
        dry_run: bool | None = None,
        stores: list[str] | None = None,
        force: bool | None = None,
        key_id: str | None = None,
        algorithm: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Rotate an encryption key.

        Args:
            dry_run: Preview changes without executing them.
            stores: Specific stores to re-encrypt.
            force: Force rotation even if the active key is recent.
            key_id: Deprecated compatibility field forwarded as-is.
            algorithm: Deprecated compatibility field forwarded as-is.
            reason: Deprecated compatibility field forwarded as-is.

        Returns:
            Dict with success, dry_run, version metadata, processed stores,
            re-encryption counts, duration, and errors.
        """
        data: dict[str, Any] = {}
        if dry_run is not None:
            data["dry_run"] = dry_run
        if stores is not None:
            data["stores"] = stores
        if force is not None:
            data["force"] = force
        if key_id is not None:
            data["key_id"] = key_id
        if algorithm is not None:
            data["algorithm"] = algorithm
        if reason is not None:
            data["reason"] = reason

        return self._client._request(
            "POST",
            "/api/v1/admin/security/rotate-key",
            json=data if data else None,
        )

    # =========================================================================
    # Audit & Compliance
    # =========================================================================

    def get_audit_log(
        self,
        limit: int = 50,
        offset: int = 0,
        event_type: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """
        Get security audit log entries.

        Args:
            limit: Maximum entries to return.
            offset: Pagination offset.
            event_type: Filter by event type.
            since: Filter events after this timestamp (ISO format).
            until: Filter events before this timestamp (ISO format).

        Returns:
            Dict with audit log entries and total count.
        """
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if event_type is not None:
            params["event_type"] = event_type
        if since is not None:
            params["since"] = since
        if until is not None:
            params["until"] = until

        return self._client._request("GET", "/api/v1/admin/security/audit", params=params)

    def get_compliance_status(self) -> dict[str, Any]:
        """
        Get compliance status for security standards.

        Returns:
            Dict with compliance status for various standards (SOC2, GDPR, etc.)
        """
        return self._client._request("GET", "/api/v1/admin/security/compliance")

    # =========================================================================
    # Threat Detection
    # =========================================================================

    def list_threats(
        self,
        limit: int = 50,
        status: str | None = None,
    ) -> dict[str, Any]:
        """
        List detected threats.

        Args:
            limit: Maximum threats to return.
            status: Filter by status (active/resolved/dismissed).

        Returns:
            Dict with list of threats.
        """
        params: dict[str, Any] = {"limit": limit}
        if status is not None:
            params["status"] = status

        return self._client._request("GET", "/api/v1/admin/security/threats", params=params)

    def resolve_threat(self, threat_id: str, resolution: str) -> dict[str, Any]:
        """
        Mark a threat as resolved.

        Args:
            threat_id: The threat identifier.
            resolution: Description of how the threat was resolved.

        Returns:
            Dict confirming resolution.
        """
        return self._client._request(
            "POST",
            f"/api/v1/admin/security/threats/{threat_id}/resolve",
            json={"resolution": resolution},
        )


class AsyncSecurityAPI:
    """
    Asynchronous Security API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     status = await client.security.get_status()
        ...     print(f"Active key: {status.get('active_key_id')}")
    """

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Security Status
    # =========================================================================

    async def get_status(self) -> dict[str, Any]:
        """Get encryption status and key rotation metadata."""
        return await self._client._request("GET", "/api/v1/admin/security/status")

    # =========================================================================
    # Health Checks
    # =========================================================================

    async def get_health_checks(self) -> dict[str, Any]:
        """Get overall status plus keyed checks, issues, and warnings."""
        return await self._client._request("GET", "/api/v1/admin/security/health")

    async def run_security_scan(self) -> dict[str, Any]:
        """Trigger a security scan."""
        return await self._client._request("POST", "/api/v1/admin/security/scan")

    async def get_scan_status(self, scan_id: str) -> dict[str, Any]:
        """Get the status of a security scan."""
        return await self._client._request("GET", f"/api/v1/admin/security/scan/{scan_id}")

    # =========================================================================
    # Key Management
    # =========================================================================

    async def list_keys(self) -> dict[str, Any]:
        """List key summaries plus active_key_id and total_keys."""
        return await self._client._request("GET", "/api/v1/admin/security/keys")

    async def get_key(self, key_id: str) -> dict[str, Any]:
        """Get details of a specific key."""
        return await self._client._request("GET", f"/api/v1/admin/security/keys/{key_id}")

    async def create_key(
        self,
        name: str,
        algorithm: str = "AES-256-GCM",
        expires_in_days: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new encryption key."""
        data: dict[str, Any] = {
            "name": name,
            "algorithm": algorithm,
        }
        if expires_in_days is not None:
            data["expires_in_days"] = expires_in_days
        if metadata is not None:
            data["metadata"] = metadata

        return await self._client._request("POST", "/api/v1/admin/security/keys", json=data)

    async def revoke_key(self, key_id: str, reason: str | None = None) -> dict[str, Any]:
        """Revoke an encryption key."""
        data: dict[str, Any] = {}
        if reason is not None:
            data["reason"] = reason

        return await self._client._request(
            "POST",
            f"/api/v1/admin/security/keys/{key_id}/revoke",
            json=data if data else None,
        )

    async def rotate_key(
        self,
        dry_run: bool | None = None,
        stores: list[str] | None = None,
        force: bool | None = None,
        key_id: str | None = None,
        algorithm: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Rotate an encryption key using the live handler request shape."""
        data: dict[str, Any] = {}
        if dry_run is not None:
            data["dry_run"] = dry_run
        if stores is not None:
            data["stores"] = stores
        if force is not None:
            data["force"] = force
        if key_id is not None:
            data["key_id"] = key_id
        if algorithm is not None:
            data["algorithm"] = algorithm
        if reason is not None:
            data["reason"] = reason

        return await self._client._request(
            "POST",
            "/api/v1/admin/security/rotate-key",
            json=data if data else None,
        )

    # =========================================================================
    # Audit & Compliance
    # =========================================================================

    async def get_audit_log(
        self,
        limit: int = 50,
        offset: int = 0,
        event_type: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """Get security audit log entries."""
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if event_type is not None:
            params["event_type"] = event_type
        if since is not None:
            params["since"] = since
        if until is not None:
            params["until"] = until

        return await self._client._request("GET", "/api/v1/admin/security/audit", params=params)

    async def get_compliance_status(self) -> dict[str, Any]:
        """Get compliance status for security standards."""
        return await self._client._request("GET", "/api/v1/admin/security/compliance")

    # =========================================================================
    # Threat Detection
    # =========================================================================

    async def list_threats(
        self,
        limit: int = 50,
        status: str | None = None,
    ) -> dict[str, Any]:
        """List detected threats."""
        params: dict[str, Any] = {"limit": limit}
        if status is not None:
            params["status"] = status

        return await self._client._request("GET", "/api/v1/admin/security/threats", params=params)

    async def resolve_threat(self, threat_id: str, resolution: str) -> dict[str, Any]:
        """Mark a threat as resolved."""
        return await self._client._request(
            "POST",
            f"/api/v1/admin/security/threats/{threat_id}/resolve",
            json={"resolution": resolution},
        )
