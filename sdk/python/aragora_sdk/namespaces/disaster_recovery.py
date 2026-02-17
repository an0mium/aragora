"""
Disaster Recovery Namespace API

Provides methods for disaster recovery operations:
- Checking DR readiness status
- Running DR drills (tabletop, simulation, full)
- Getting and monitoring RPO/RTO objectives
- Validating DR configuration
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class DisasterRecoveryAPI:
    """
    Synchronous Disaster Recovery API.

    Provides methods for enterprise disaster recovery operations:
    - Backup management (create, list, get, delete)
    - Backup verification and restore testing
    - Cleanup and statistics

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> backups = client.dr.list_backups()
        >>> stats = client.dr.get_stats()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Backups (v2)
    # ===========================================================================

    def list_backups(self) -> dict[str, Any]:
        """List all backups."""
        return self._client.request("GET", "/api/v2/backups")

    def create_backup(self, **kwargs: Any) -> dict[str, Any]:
        """Create a new backup."""
        return self._client.request("POST", "/api/v2/backups", json=kwargs)

    def get_backup(self, backup_id: str) -> dict[str, Any]:
        """Get a backup by ID."""
        return self._client.request("GET", f"/api/v2/backups/{backup_id}")

    def delete_backup(self, backup_id: str) -> dict[str, Any]:
        """Delete a backup."""
        return self._client.request("DELETE", f"/api/v2/backups/{backup_id}")

    # ===========================================================================
    # Backup Operations (v1)
    # ===========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get backup statistics."""
        return self._client.request("GET", "/api/v1/backups/stats")

    def cleanup(self, **kwargs: Any) -> dict[str, Any]:
        """Run backup cleanup."""
        return self._client.request("POST", "/api/v1/backups/cleanup", json=kwargs)

    def verify(self, backup_id: str) -> dict[str, Any]:
        """Verify a backup."""
        return self._client.request("POST", f"/api/v1/backups/{backup_id}/verify")

    def verify_comprehensive(self, backup_id: str) -> dict[str, Any]:
        """Run comprehensive backup verification."""
        return self._client.request("POST", f"/api/v1/backups/{backup_id}/verify-comprehensive")

    def restore_test(self, backup_id: str) -> dict[str, Any]:
        """Run a restore test on a backup."""
        return self._client.request("POST", f"/api/v1/backups/{backup_id}/restore-test")

    # ===========================================================================
    # Convenience Methods
    # ===========================================================================

    def is_ready(self) -> bool:
        """Check if DR is ready for production."""
        status = self.get_status()
        validation = self.validate()
        return bool(status.get("ready")) and bool(validation.get("valid"))

    def get_health_summary(self) -> dict[str, Any]:
        """Get a summary of DR health."""
        status = self.get_status()
        objectives = self.get_objectives()
        return {
            "ready": status.get("ready", False),
            "health": status.get("overall_health", "unknown"),
            "rpo_compliant": objectives.get("rpo_compliant", False),
            "rto_compliant": objectives.get("rto_compliant", False),
            "issues_count": len(status.get("issues", [])),
        }


class AsyncDisasterRecoveryAPI:
    """
    Asynchronous Disaster Recovery API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     backups = await client.dr.list_backups()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Backups (v2)
    # ===========================================================================

    async def list_backups(self) -> dict[str, Any]:
        """List all backups."""
        return await self._client.request("GET", "/api/v2/backups")

    async def create_backup(self, **kwargs: Any) -> dict[str, Any]:
        """Create a new backup."""
        return await self._client.request("POST", "/api/v2/backups", json=kwargs)

    async def get_backup(self, backup_id: str) -> dict[str, Any]:
        """Get a backup by ID."""
        return await self._client.request("GET", f"/api/v2/backups/{backup_id}")

    async def delete_backup(self, backup_id: str) -> dict[str, Any]:
        """Delete a backup."""
        return await self._client.request("DELETE", f"/api/v2/backups/{backup_id}")

    # ===========================================================================
    # Backup Operations (v1)
    # ===========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get backup statistics."""
        return await self._client.request("GET", "/api/v1/backups/stats")

    async def cleanup(self, **kwargs: Any) -> dict[str, Any]:
        """Run backup cleanup."""
        return await self._client.request("POST", "/api/v1/backups/cleanup", json=kwargs)

    async def verify(self, backup_id: str) -> dict[str, Any]:
        """Verify a backup."""
        return await self._client.request("POST", f"/api/v1/backups/{backup_id}/verify")

    async def verify_comprehensive(self, backup_id: str) -> dict[str, Any]:
        """Run comprehensive backup verification."""
        return await self._client.request("POST", f"/api/v1/backups/{backup_id}/verify-comprehensive")

    async def restore_test(self, backup_id: str) -> dict[str, Any]:
        """Run a restore test on a backup."""
        return await self._client.request("POST", f"/api/v1/backups/{backup_id}/restore-test")

    # ===========================================================================
    # Convenience Methods
    # ===========================================================================

    async def is_ready(self) -> bool:
        """Check if DR is ready for production."""
        status = await self.get_status()
        validation = await self.validate()
        return bool(status.get("ready")) and bool(validation.get("valid"))

    async def get_health_summary(self) -> dict[str, Any]:
        """Get a summary of DR health."""
        status = await self.get_status()
        objectives = await self.get_objectives()
        return {
            "ready": status.get("ready", False),
            "health": status.get("overall_health", "unknown"),
            "rpo_compliant": objectives.get("rpo_compliant", False),
            "rto_compliant": objectives.get("rto_compliant", False),
            "issues_count": len(status.get("issues", [])),
        }
