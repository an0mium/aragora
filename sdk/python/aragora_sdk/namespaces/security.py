"""
Security Namespace API.

Provides security status, health checks, and key rotation management.

Features:
- Overall security status monitoring
- Security health checks
- Encryption key inventory
- Key rotation operations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

SecurityLevel = Literal["healthy", "degraded", "critical"]
KeyStatus = Literal["active", "expired", "revoked"]
CheckStatus = Literal["ok", "warning", "error"]


class SecurityAPI:
    """
    Synchronous Security API.

    Provides methods for security management:
    - Get overall security status
    - Run security health checks
    - Inspect encryption keys
    - Rotate keys for security maintenance

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> status = client.security.get_status()
        >>> if status['overall'] != 'healthy':
        ...     print("Security issues detected!")
        >>> health = client.security.get_health_checks()
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Security Status
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """
        Get overall security status.

        Returns:
            Dict with overall status (healthy/degraded/critical),
            encryption_enabled, audit_logging_enabled, mfa_enabled,
            last_security_scan, active_threats, and metadata.
        """
        return self._client._request("GET", "/api/v1/admin/security/status")

    # =========================================================================
    # Health Checks
    # =========================================================================

    def get_health_checks(self) -> dict[str, Any]:
        """
        Get security health checks.

        Runs checks on all security components and returns their status.

        Returns:
            Dict with list of health checks, each containing component,
            status (ok/warning/error), message, and last_checked timestamp.
        """
        return self._client._request("GET", "/api/v1/admin/security/health")

    # =========================================================================
    # Key Management
    # =========================================================================

    def list_keys(self) -> dict[str, Any]:
        """
        List all security keys.

        Returns:
            Dict with the server's key inventory payload.
        """
        return self._client._request("GET", "/api/v1/admin/security/keys")

    def rotate_key(
        self,
        key_id: str | None = None,
        algorithm: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Rotate an encryption key.

        Creates a new key and deprecates the old one.

        Args:
            key_id: Optional specific key to rotate.
            algorithm: Optional new algorithm to use.
            reason: Optional reason for rotation.

        Returns:
            Dict with the server's rotation result payload.
        """
        data: dict[str, Any] = {}
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


class AsyncSecurityAPI:
    """
    Asynchronous Security API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     status = await client.security.get_status()
        ...     print(f"Security status: {status['overall']}")
    """

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Security Status
    # =========================================================================

    async def get_status(self) -> dict[str, Any]:
        """Get overall security status."""
        return await self._client._request("GET", "/api/v1/admin/security/status")

    # =========================================================================
    # Health Checks
    # =========================================================================

    async def get_health_checks(self) -> dict[str, Any]:
        """Get security health checks."""
        return await self._client._request("GET", "/api/v1/admin/security/health")

    # =========================================================================
    # Key Management
    # =========================================================================

    async def list_keys(self) -> dict[str, Any]:
        """List all security keys."""
        return await self._client._request("GET", "/api/v1/admin/security/keys")

    async def rotate_key(
        self,
        key_id: str | None = None,
        algorithm: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Rotate an encryption key."""
        data: dict[str, Any] = {}
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
