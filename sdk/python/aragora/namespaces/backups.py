"""
Backups namespace for disaster recovery operations.

Provides API access to backup management, restore operations,
and disaster recovery functionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

_List = list  # Preserve builtin list for type annotations


class BackupsAPI:
    """Synchronous backups API."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def list(
        self,
        limit: int = 50,
        offset: int = 0,
        backup_type: str | None = None,
        status: str | None = None,
    ) -> _List[dict[str, Any]]:
        """
        List backups.

        Args:
            limit: Maximum number of backups to return
            offset: Number of backups to skip
            backup_type: Filter by type (full, incremental, snapshot)
            status: Filter by status (completed, in_progress, failed)

        Returns:
            List of backup records
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if backup_type:
            params["backup_type"] = backup_type
        if status:
            params["status"] = status

        return self._client._request("GET", "/api/v1/backups", params=params)

    def get(self, backup_id: str) -> dict[str, Any]:
        """
        Get backup details.

        Args:
            backup_id: Backup identifier

        Returns:
            Backup details
        """
        return self._client._request("GET", f"/api/v1/backups/{backup_id}")

    def create(
        self,
        backup_type: str = "full",
        description: str | None = None,
        include_data: _List[str] | None = None,
        exclude_data: _List[str] | None = None,
    ) -> dict[str, Any]:
        """
        Create a new backup.

        Args:
            backup_type: Type of backup (full, incremental, snapshot)
            description: Optional description
            include_data: Data types to include
            exclude_data: Data types to exclude

        Returns:
            Created backup record
        """
        data: dict[str, Any] = {"backup_type": backup_type}
        if description:
            data["description"] = description
        if include_data:
            data["include_data"] = include_data
        if exclude_data:
            data["exclude_data"] = exclude_data

        return self._client._request("POST", "/api/v1/backups", json=data)

    def delete(self, backup_id: str) -> dict[str, Any]:
        """
        Delete a backup.

        Args:
            backup_id: Backup identifier

        Returns:
            Deletion confirmation
        """
        return self._client._request("DELETE", f"/api/v1/backups/{backup_id}")

    def restore(
        self,
        backup_id: str,
        target_namespace: str | None = None,
        data_types: _List[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Restore from a backup.

        Args:
            backup_id: Backup to restore from
            target_namespace: Namespace to restore to
            data_types: Specific data types to restore
            dry_run: If True, simulate restore without making changes

        Returns:
            Restore operation status
        """
        data: dict[str, Any] = {"dry_run": dry_run}
        if target_namespace:
            data["target_namespace"] = target_namespace
        if data_types:
            data["data_types"] = data_types

        return self._client._request("POST", f"/api/v1/backups/{backup_id}/restore", json=data)

    def get_restore_status(self, restore_id: str) -> dict[str, Any]:
        """
        Get restore operation status.

        Args:
            restore_id: Restore operation identifier

        Returns:
            Restore operation status
        """
        return self._client._request("GET", f"/api/v1/restores/{restore_id}")

    def schedule(
        self,
        schedule: str,
        backup_type: str = "incremental",
        retention_days: int = 30,
        enabled: bool = True,
    ) -> dict[str, Any]:
        """
        Create a backup schedule.

        Args:
            schedule: Cron expression for schedule
            backup_type: Type of scheduled backups
            retention_days: Days to retain backups
            enabled: Whether schedule is active

        Returns:
            Created schedule
        """
        return self._client._request(
            "POST",
            "/api/v1/backups/schedules",
            json={
                "schedule": schedule,
                "backup_type": backup_type,
                "retention_days": retention_days,
                "enabled": enabled,
            },
        )

    def list_schedules(self) -> _List[dict[str, Any]]:
        """
        List backup schedules.

        Returns:
            List of backup schedules
        """
        return self._client._request("GET", "/api/v1/backups/schedules")

    def delete_schedule(self, schedule_id: str) -> dict[str, Any]:
        """
        Delete a backup schedule.

        Args:
            schedule_id: Schedule identifier

        Returns:
            Deletion confirmation
        """
        return self._client._request("DELETE", f"/api/v1/backups/schedules/{schedule_id}")

    def verify(self, backup_id: str) -> dict[str, Any]:
        """
        Verify backup integrity.

        Args:
            backup_id: Backup identifier

        Returns:
            Verification result with integrity status
        """
        return self._client._request("POST", f"/api/v1/backups/{backup_id}/verify")

    def verify_comprehensive(self, backup_id: str) -> dict[str, Any]:
        """
        Run comprehensive backup verification.

        Args:
            backup_id: Backup identifier

        Returns:
            Detailed verification results including checksums and data validation
        """
        return self._client._request("POST", f"/api/v1/backups/{backup_id}/verify-comprehensive")

    def test_restore(
        self,
        backup_id: str,
        target_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Test restore without applying changes.

        Args:
            backup_id: Backup identifier
            target_path: Optional target path for test restore

        Returns:
            Test restore results
        """
        data: dict[str, Any] = {}
        if target_path:
            data["target_path"] = target_path
        return self._client._request(
            "POST",
            f"/api/v1/backups/{backup_id}/restore-test",
            json=data if data else None,
        )

    def cleanup(self, dry_run: bool = True) -> dict[str, Any]:
        """
        Clean up old or expired backups.

        Args:
            dry_run: If True, only report what would be cleaned up

        Returns:
            Cleanup results with affected backups
        """
        return self._client._request("POST", "/api/v1/backups/cleanup", json={"dry_run": dry_run})

    def get_stats(self) -> dict[str, Any]:
        """
        Get backup statistics.

        Returns:
            Backup statistics including counts, sizes, and health metrics
        """
        return self._client._request("GET", "/api/v1/backups/stats")


class AsyncBackupsAPI:
    """Asynchronous backups API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def list(
        self,
        limit: int = 50,
        offset: int = 0,
        backup_type: str | None = None,
        status: str | None = None,
    ) -> _List[dict[str, Any]]:
        """List backups."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if backup_type:
            params["backup_type"] = backup_type
        if status:
            params["status"] = status

        return await self._client._request("GET", "/api/v1/backups", params=params)

    async def get(self, backup_id: str) -> dict[str, Any]:
        """Get backup details."""
        return await self._client._request("GET", f"/api/v1/backups/{backup_id}")

    async def create(
        self,
        backup_type: str = "full",
        description: str | None = None,
        include_data: _List[str] | None = None,
        exclude_data: _List[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new backup."""
        data: dict[str, Any] = {"backup_type": backup_type}
        if description:
            data["description"] = description
        if include_data:
            data["include_data"] = include_data
        if exclude_data:
            data["exclude_data"] = exclude_data

        return await self._client._request("POST", "/api/v1/backups", json=data)

    async def delete(self, backup_id: str) -> dict[str, Any]:
        """Delete a backup."""
        return await self._client._request("DELETE", f"/api/v1/backups/{backup_id}")

    async def restore(
        self,
        backup_id: str,
        target_namespace: str | None = None,
        data_types: _List[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Restore from a backup."""
        data: dict[str, Any] = {"dry_run": dry_run}
        if target_namespace:
            data["target_namespace"] = target_namespace
        if data_types:
            data["data_types"] = data_types

        return await self._client._request(
            "POST", f"/api/v1/backups/{backup_id}/restore", json=data
        )

    async def get_restore_status(self, restore_id: str) -> dict[str, Any]:
        """Get restore operation status."""
        return await self._client._request("GET", f"/api/v1/restores/{restore_id}")

    async def schedule(
        self,
        schedule: str,
        backup_type: str = "incremental",
        retention_days: int = 30,
        enabled: bool = True,
    ) -> dict[str, Any]:
        """Create a backup schedule."""
        return await self._client._request(
            "POST",
            "/api/v1/backups/schedules",
            json={
                "schedule": schedule,
                "backup_type": backup_type,
                "retention_days": retention_days,
                "enabled": enabled,
            },
        )

    async def list_schedules(self) -> _List[dict[str, Any]]:
        """List backup schedules."""
        return await self._client._request("GET", "/api/v1/backups/schedules")

    async def delete_schedule(self, schedule_id: str) -> dict[str, Any]:
        """Delete a backup schedule."""
        return await self._client._request("DELETE", f"/api/v1/backups/schedules/{schedule_id}")

    async def verify(self, backup_id: str) -> dict[str, Any]:
        """Verify backup integrity."""
        return await self._client._request("POST", f"/api/v1/backups/{backup_id}/verify")

    async def verify_comprehensive(self, backup_id: str) -> dict[str, Any]:
        """Run comprehensive backup verification."""
        return await self._client._request(
            "POST", f"/api/v1/backups/{backup_id}/verify-comprehensive"
        )

    async def test_restore(
        self,
        backup_id: str,
        target_path: str | None = None,
    ) -> dict[str, Any]:
        """Test restore without applying changes."""
        data: dict[str, Any] = {}
        if target_path:
            data["target_path"] = target_path
        return await self._client._request(
            "POST",
            f"/api/v1/backups/{backup_id}/restore-test",
            json=data if data else None,
        )

    async def cleanup(self, dry_run: bool = True) -> dict[str, Any]:
        """Clean up old or expired backups."""
        return await self._client._request(
            "POST", "/api/v1/backups/cleanup", json={"dry_run": dry_run}
        )

    async def get_stats(self) -> dict[str, Any]:
        """Get backup statistics."""
        return await self._client._request("GET", "/api/v1/backups/stats")
