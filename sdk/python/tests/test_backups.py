"""Tests for Backups namespace API.

Note: BackupsAPI uses self._client._request() which is an alias for
self.request() that passes params/json/headers explicitly, so all
assert_called_once_with calls must include params= and headers= kwargs.
"""

from __future__ import annotations

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


class TestBackupsList:
    """Tests for listing backups."""

    def test_list_backups_default(self, client: AragoraClient, mock_request) -> None:
        """List backups with default parameters."""
        mock_request.return_value = [{"backup_id": "bk_1", "status": "completed"}]

        result = client.backups.list()

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/backups",
            params={"limit": 50, "offset": 0},
            json=None,
            headers=None,
        )
        assert result[0]["backup_id"] == "bk_1"

    def test_list_backups_filtered(self, client: AragoraClient, mock_request) -> None:
        """List backups filtered by type and status."""
        mock_request.return_value = []

        client.backups.list(backup_type="incremental", status="completed", limit=10)

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["params"]["backup_type"] == "incremental"
        assert call_kwargs["params"]["status"] == "completed"
        assert call_kwargs["params"]["limit"] == 10


class TestBackupsGet:
    """Tests for getting backup details."""

    def test_get_backup(self, client: AragoraClient, mock_request) -> None:
        """Get backup details."""
        mock_request.return_value = {
            "backup_id": "bk_123",
            "backup_type": "full",
            "status": "completed",
            "size_bytes": 1048576,
        }

        result = client.backups.get("bk_123")

        mock_request.assert_called_once_with(
            "GET", "/api/v1/backups/bk_123", params=None, json=None, headers=None
        )
        assert result["backup_type"] == "full"
        assert result["size_bytes"] == 1048576


class TestBackupsCreate:
    """Tests for backup creation."""

    def test_create_backup_default(self, client: AragoraClient, mock_request) -> None:
        """Create a backup with default type (full)."""
        mock_request.return_value = {"backup_id": "bk_new", "status": "in_progress"}

        result = client.backups.create()

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/backups",
            params=None,
            json={"backup_type": "full"},
            headers=None,
        )
        assert result["status"] == "in_progress"

    def test_create_backup_incremental(self, client: AragoraClient, mock_request) -> None:
        """Create an incremental backup."""
        mock_request.return_value = {"backup_id": "bk_inc"}

        client.backups.create(backup_type="incremental")

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["json"]["backup_type"] == "incremental"

    def test_create_backup_full_options(self, client: AragoraClient, mock_request) -> None:
        """Create a backup with all options."""
        mock_request.return_value = {"backup_id": "bk_full"}

        client.backups.create(
            backup_type="snapshot",
            description="Weekly snapshot",
            include_data=["debates", "knowledge"],
            exclude_data=["logs"],
        )

        call_kwargs = mock_request.call_args[1]
        call_json = call_kwargs["json"]
        assert call_json["backup_type"] == "snapshot"
        assert call_json["description"] == "Weekly snapshot"
        assert call_json["include_data"] == ["debates", "knowledge"]
        assert call_json["exclude_data"] == ["logs"]


class TestBackupsDelete:
    """Tests for backup deletion."""

    def test_delete_backup(self, client: AragoraClient, mock_request) -> None:
        """Delete a backup."""
        mock_request.return_value = {"deleted": True}

        result = client.backups.delete("bk_123")

        mock_request.assert_called_once_with(
            "DELETE", "/api/v1/backups/bk_123", params=None, json=None, headers=None
        )
        assert result["deleted"] is True


class TestBackupsRestore:
    """Tests for backup restore operations."""

    def test_restore_default(self, client: AragoraClient, mock_request) -> None:
        """Restore from a backup with defaults."""
        mock_request.return_value = {"restore_id": "rst_1", "status": "in_progress"}

        result = client.backups.restore("bk_123")

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/backups/bk_123/restore",
            params=None,
            json={"dry_run": False},
            headers=None,
        )
        assert result["restore_id"] == "rst_1"

    def test_restore_dry_run(self, client: AragoraClient, mock_request) -> None:
        """Simulate a restore without making changes."""
        mock_request.return_value = {"restore_id": "rst_dry", "changes_preview": []}

        client.backups.restore("bk_123", dry_run=True)

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["json"]["dry_run"] is True

    def test_restore_selective(self, client: AragoraClient, mock_request) -> None:
        """Restore specific data types to a target namespace."""
        mock_request.return_value = {"restore_id": "rst_sel"}

        client.backups.restore(
            "bk_123",
            target_namespace="staging",
            data_types=["debates", "agents"],
        )

        call_kwargs = mock_request.call_args[1]
        call_json = call_kwargs["json"]
        assert call_json["target_namespace"] == "staging"
        assert call_json["data_types"] == ["debates", "agents"]

    def test_get_restore_status(self, client: AragoraClient, mock_request) -> None:
        """Get restore operation status."""
        mock_request.return_value = {
            "restore_id": "rst_1",
            "status": "completed",
            "progress": 100,
        }

        result = client.backups.get_restore_status("rst_1")

        mock_request.assert_called_once_with(
            "GET", "/api/v1/restores/rst_1", params=None, json=None, headers=None
        )
        assert result["status"] == "completed"


class TestBackupsSchedules:
    """Tests for backup schedule management."""

    def test_create_schedule(self, client: AragoraClient, mock_request) -> None:
        """Create a backup schedule."""
        mock_request.return_value = {"schedule_id": "sch_1"}

        result = client.backups.schedule("0 2 * * *")

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/backups/schedules",
            params=None,
            json={
                "schedule": "0 2 * * *",
                "backup_type": "incremental",
                "retention_days": 30,
                "enabled": True,
            },
            headers=None,
        )
        assert result["schedule_id"] == "sch_1"

    def test_create_schedule_custom(self, client: AragoraClient, mock_request) -> None:
        """Create a backup schedule with custom settings."""
        mock_request.return_value = {"schedule_id": "sch_2"}

        client.backups.schedule(
            "0 0 * * 0",
            backup_type="full",
            retention_days=90,
            enabled=False,
        )

        call_kwargs = mock_request.call_args[1]
        call_json = call_kwargs["json"]
        assert call_json["schedule"] == "0 0 * * 0"
        assert call_json["backup_type"] == "full"
        assert call_json["retention_days"] == 90
        assert call_json["enabled"] is False

    def test_list_schedules(self, client: AragoraClient, mock_request) -> None:
        """List backup schedules."""
        mock_request.return_value = [
            {"schedule_id": "sch_1", "schedule": "0 2 * * *"},
        ]

        result = client.backups.list_schedules()

        mock_request.assert_called_once_with(
            "GET", "/api/v1/backups/schedules", params=None, json=None, headers=None
        )
        assert len(result) == 1

    def test_delete_schedule(self, client: AragoraClient, mock_request) -> None:
        """Delete a backup schedule."""
        mock_request.return_value = {"deleted": True}

        result = client.backups.delete_schedule("sch_1")

        mock_request.assert_called_once_with(
            "DELETE", "/api/v1/backups/schedules/sch_1", params=None, json=None, headers=None
        )
        assert result["deleted"] is True


class TestAsyncBackups:
    """Tests for async backups API."""

    @pytest.mark.asyncio
    async def test_async_list_backups(self, mock_async_request) -> None:
        """List backups asynchronously."""
        mock_async_request.return_value = [{"backup_id": "bk_1"}]

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.backups.list()

            assert result[0]["backup_id"] == "bk_1"

    @pytest.mark.asyncio
    async def test_async_create_backup(self, mock_async_request) -> None:
        """Create a backup asynchronously."""
        mock_async_request.return_value = {"backup_id": "bk_async"}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.backups.create(backup_type="snapshot")

            assert result["backup_id"] == "bk_async"

    @pytest.mark.asyncio
    async def test_async_restore(self, mock_async_request) -> None:
        """Restore from a backup asynchronously."""
        mock_async_request.return_value = {"restore_id": "rst_async"}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.backups.restore("bk_123", dry_run=True)

            assert result["restore_id"] == "rst_async"

    @pytest.mark.asyncio
    async def test_async_schedule(self, mock_async_request) -> None:
        """Create a schedule asynchronously."""
        mock_async_request.return_value = {"schedule_id": "sch_async"}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.backups.schedule("0 3 * * *")

            call_kwargs = mock_async_request.call_args[1]
            call_json = call_kwargs["json"]
            assert call_json["schedule"] == "0 3 * * *"
            assert call_json["backup_type"] == "incremental"
            assert call_json["retention_days"] == 30
            assert call_json["enabled"] is True
            assert result["schedule_id"] == "sch_async"
