"""Tests for Disaster Recovery SDK namespace."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock client."""
    return MagicMock()


class TestDisasterRecoveryAPI:
    """Test synchronous DisasterRecoveryAPI."""

    def test_init(self, mock_client: MagicMock) -> None:
        """Test API initialization."""
        from aragora.namespaces.disaster_recovery import DisasterRecoveryAPI

        api = DisasterRecoveryAPI(mock_client)
        assert api._client is mock_client

    def test_get_status(self, mock_client: MagicMock) -> None:
        """Test get_status calls correct endpoint."""
        from aragora.namespaces.disaster_recovery import DisasterRecoveryAPI

        mock_client.request.return_value = {
            "ready": True,
            "overall_health": "healthy",
            "rpo_met": True,
            "rto_met": True,
            "issues": [],
        }

        api = DisasterRecoveryAPI(mock_client)
        result = api.get_status()

        mock_client.request.assert_called_once_with("GET", "/api/v2/dr/status")
        assert result["ready"] is True
        assert result["overall_health"] == "healthy"

    def test_run_drill(self, mock_client: MagicMock) -> None:
        """Test run_drill calls correct endpoint."""
        from aragora.namespaces.disaster_recovery import DisasterRecoveryAPI

        mock_client.request.return_value = {
            "drill_id": "drill-123",
            "type": "tabletop",
            "success": True,
            "recovery_time_seconds": 120,
        }

        api = DisasterRecoveryAPI(mock_client)
        result = api.run_drill(type="tabletop")

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0] == ("POST", "/api/v2/dr/drill")
        assert call_args[1]["json"]["type"] == "tabletop"
        assert result["drill_id"] == "drill-123"
        assert result["success"] is True

    def test_run_drill_with_options(self, mock_client: MagicMock) -> None:
        """Test run_drill with all options."""
        from aragora.namespaces.disaster_recovery import DisasterRecoveryAPI

        mock_client.request.return_value = {"drill_id": "drill-456", "success": True}

        api = DisasterRecoveryAPI(mock_client)
        api.run_drill(
            type="simulation",
            components=["database", "cache", "api"],
            notify_team=True,
            dry_run=True,
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["type"] == "simulation"
        assert json_body["components"] == ["database", "cache", "api"]
        assert json_body["notify_team"] is True
        assert json_body["dry_run"] is True

    def test_run_drill_full_type(self, mock_client: MagicMock) -> None:
        """Test run_drill with full drill type."""
        from aragora.namespaces.disaster_recovery import DisasterRecoveryAPI

        mock_client.request.return_value = {
            "drill_id": "drill-789",
            "type": "full",
            "success": True,
            "duration_seconds": 3600,
        }

        api = DisasterRecoveryAPI(mock_client)
        result = api.run_drill(type="full", notify_team=True)

        call_args = mock_client.request.call_args
        assert call_args[1]["json"]["type"] == "full"
        assert result["duration_seconds"] == 3600

    def test_get_objectives(self, mock_client: MagicMock) -> None:
        """Test get_objectives calls correct endpoint."""
        from aragora.namespaces.disaster_recovery import DisasterRecoveryAPI

        mock_client.request.return_value = {
            "rpo_minutes": 15,
            "rto_minutes": 60,
            "current_rpo_minutes": 10,
            "current_rto_minutes": 45,
            "rpo_compliant": True,
            "rto_compliant": True,
        }

        api = DisasterRecoveryAPI(mock_client)
        result = api.get_objectives()

        mock_client.request.assert_called_once_with("GET", "/api/v2/dr/objectives")
        assert result["rpo_minutes"] == 15
        assert result["rto_minutes"] == 60
        assert result["rpo_compliant"] is True

    def test_validate(self, mock_client: MagicMock) -> None:
        """Test validate calls correct endpoint."""
        from aragora.namespaces.disaster_recovery import DisasterRecoveryAPI

        mock_client.request.return_value = {
            "valid": True,
            "overall_score": 95,
            "checks": [],
        }

        api = DisasterRecoveryAPI(mock_client)
        result = api.validate()

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0] == ("POST", "/api/v2/dr/validate")
        assert result["valid"] is True
        assert result["overall_score"] == 95

    def test_validate_with_options(self, mock_client: MagicMock) -> None:
        """Test validate with all options."""
        from aragora.namespaces.disaster_recovery import DisasterRecoveryAPI

        mock_client.request.return_value = {"valid": True, "checks": []}

        api = DisasterRecoveryAPI(mock_client)
        api.validate(
            check_backups=True,
            check_replication=True,
            check_failover=True,
            check_dns=True,
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["check_backups"] is True
        assert json_body["check_replication"] is True
        assert json_body["check_failover"] is True
        assert json_body["check_dns"] is True

    def test_is_ready(self, mock_client: MagicMock) -> None:
        """Test is_ready convenience method."""
        from aragora.namespaces.disaster_recovery import DisasterRecoveryAPI

        mock_client.request.side_effect = [
            {"ready": True, "overall_health": "healthy"},  # get_status
            {"valid": True, "overall_score": 100},  # validate
        ]

        api = DisasterRecoveryAPI(mock_client)
        result = api.is_ready()

        assert mock_client.request.call_count == 2
        assert result is True

    def test_is_ready_not_ready(self, mock_client: MagicMock) -> None:
        """Test is_ready returns False when not ready."""
        from aragora.namespaces.disaster_recovery import DisasterRecoveryAPI

        mock_client.request.side_effect = [
            {"ready": False, "overall_health": "degraded"},
            {"valid": True},
        ]

        api = DisasterRecoveryAPI(mock_client)
        result = api.is_ready()

        assert result is False

    def test_is_ready_invalid_config(self, mock_client: MagicMock) -> None:
        """Test is_ready returns False when config is invalid."""
        from aragora.namespaces.disaster_recovery import DisasterRecoveryAPI

        mock_client.request.side_effect = [
            {"ready": True},
            {"valid": False},
        ]

        api = DisasterRecoveryAPI(mock_client)
        result = api.is_ready()

        assert result is False

    def test_get_health_summary(self, mock_client: MagicMock) -> None:
        """Test get_health_summary convenience method."""
        from aragora.namespaces.disaster_recovery import DisasterRecoveryAPI

        mock_client.request.side_effect = [
            {
                "ready": True,
                "overall_health": "healthy",
                "issues": ["minor issue 1"],
            },  # get_status
            {
                "rpo_compliant": True,
                "rto_compliant": True,
            },  # get_objectives
        ]

        api = DisasterRecoveryAPI(mock_client)
        result = api.get_health_summary()

        assert mock_client.request.call_count == 2
        assert result["ready"] is True
        assert result["health"] == "healthy"
        assert result["rpo_compliant"] is True
        assert result["rto_compliant"] is True
        assert result["issues_count"] == 1

    def test_get_health_summary_degraded(self, mock_client: MagicMock) -> None:
        """Test get_health_summary with degraded health."""
        from aragora.namespaces.disaster_recovery import DisasterRecoveryAPI

        mock_client.request.side_effect = [
            {
                "ready": False,
                "overall_health": "degraded",
                "issues": ["issue 1", "issue 2", "issue 3"],
            },
            {"rpo_compliant": False, "rto_compliant": True},
        ]

        api = DisasterRecoveryAPI(mock_client)
        result = api.get_health_summary()

        assert result["ready"] is False
        assert result["health"] == "degraded"
        assert result["rpo_compliant"] is False
        assert result["issues_count"] == 3


@pytest.fixture
def mock_async_client() -> MagicMock:
    """Create a mock async client."""
    from unittest.mock import AsyncMock

    client = MagicMock()
    client.request = AsyncMock()
    return client


class TestAsyncDisasterRecoveryAPI:
    """Test asynchronous AsyncDisasterRecoveryAPI."""

    @pytest.mark.asyncio
    async def test_get_status(self, mock_async_client: MagicMock) -> None:
        """Test get_status calls correct endpoint."""
        from aragora.namespaces.disaster_recovery import AsyncDisasterRecoveryAPI

        mock_async_client.request.return_value = {
            "ready": True,
            "overall_health": "healthy",
        }

        api = AsyncDisasterRecoveryAPI(mock_async_client)
        result = await api.get_status()

        mock_async_client.request.assert_called_once_with("GET", "/api/v2/dr/status")
        assert result["ready"] is True

    @pytest.mark.asyncio
    async def test_run_drill(self, mock_async_client: MagicMock) -> None:
        """Test run_drill calls correct endpoint."""
        from aragora.namespaces.disaster_recovery import AsyncDisasterRecoveryAPI

        mock_async_client.request.return_value = {
            "drill_id": "drill-123",
            "success": True,
        }

        api = AsyncDisasterRecoveryAPI(mock_async_client)
        result = await api.run_drill(type="tabletop", notify_team=True)

        mock_async_client.request.assert_called_once()
        call_args = mock_async_client.request.call_args
        assert call_args[0] == ("POST", "/api/v2/dr/drill")
        assert call_args[1]["json"]["type"] == "tabletop"
        assert call_args[1]["json"]["notify_team"] is True
        assert result["drill_id"] == "drill-123"

    @pytest.mark.asyncio
    async def test_run_drill_with_components(self, mock_async_client: MagicMock) -> None:
        """Test run_drill with component selection."""
        from aragora.namespaces.disaster_recovery import AsyncDisasterRecoveryAPI

        mock_async_client.request.return_value = {"drill_id": "drill-456"}

        api = AsyncDisasterRecoveryAPI(mock_async_client)
        await api.run_drill(
            type="simulation",
            components=["redis", "postgres"],
            dry_run=True,
        )

        call_args = mock_async_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["components"] == ["redis", "postgres"]
        assert json_body["dry_run"] is True

    @pytest.mark.asyncio
    async def test_get_objectives(self, mock_async_client: MagicMock) -> None:
        """Test get_objectives calls correct endpoint."""
        from aragora.namespaces.disaster_recovery import AsyncDisasterRecoveryAPI

        mock_async_client.request.return_value = {
            "rpo_minutes": 30,
            "rto_minutes": 120,
            "rpo_compliant": True,
            "rto_compliant": False,
        }

        api = AsyncDisasterRecoveryAPI(mock_async_client)
        result = await api.get_objectives()

        mock_async_client.request.assert_called_once_with("GET", "/api/v2/dr/objectives")
        assert result["rpo_minutes"] == 30
        assert result["rto_compliant"] is False

    @pytest.mark.asyncio
    async def test_validate(self, mock_async_client: MagicMock) -> None:
        """Test validate calls correct endpoint."""
        from aragora.namespaces.disaster_recovery import AsyncDisasterRecoveryAPI

        mock_async_client.request.return_value = {
            "valid": True,
            "overall_score": 85,
        }

        api = AsyncDisasterRecoveryAPI(mock_async_client)
        result = await api.validate(check_backups=True, check_dns=True)

        mock_async_client.request.assert_called_once()
        call_args = mock_async_client.request.call_args
        assert call_args[0] == ("POST", "/api/v2/dr/validate")
        assert call_args[1]["json"]["check_backups"] is True
        assert call_args[1]["json"]["check_dns"] is True
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_is_ready(self, mock_async_client: MagicMock) -> None:
        """Test is_ready convenience method."""
        from aragora.namespaces.disaster_recovery import AsyncDisasterRecoveryAPI

        mock_async_client.request.side_effect = [
            {"ready": True},
            {"valid": True},
        ]

        api = AsyncDisasterRecoveryAPI(mock_async_client)
        result = await api.is_ready()

        assert mock_async_client.request.call_count == 2
        assert result is True

    @pytest.mark.asyncio
    async def test_is_ready_not_ready(self, mock_async_client: MagicMock) -> None:
        """Test is_ready returns False when not ready."""
        from aragora.namespaces.disaster_recovery import AsyncDisasterRecoveryAPI

        mock_async_client.request.side_effect = [
            {"ready": False},
            {"valid": True},
        ]

        api = AsyncDisasterRecoveryAPI(mock_async_client)
        result = await api.is_ready()

        assert result is False

    @pytest.mark.asyncio
    async def test_get_health_summary(self, mock_async_client: MagicMock) -> None:
        """Test get_health_summary convenience method."""
        from aragora.namespaces.disaster_recovery import AsyncDisasterRecoveryAPI

        mock_async_client.request.side_effect = [
            {"ready": True, "overall_health": "healthy", "issues": []},
            {"rpo_compliant": True, "rto_compliant": True},
        ]

        api = AsyncDisasterRecoveryAPI(mock_async_client)
        result = await api.get_health_summary()

        assert mock_async_client.request.call_count == 2
        assert result["ready"] is True
        assert result["health"] == "healthy"
        assert result["rpo_compliant"] is True
        assert result["rto_compliant"] is True
        assert result["issues_count"] == 0

    @pytest.mark.asyncio
    async def test_get_health_summary_critical(self, mock_async_client: MagicMock) -> None:
        """Test get_health_summary with critical health status."""
        from aragora.namespaces.disaster_recovery import AsyncDisasterRecoveryAPI

        mock_async_client.request.side_effect = [
            {
                "ready": False,
                "overall_health": "critical",
                "issues": ["backup failed", "replication lag", "dns misconfigured"],
            },
            {"rpo_compliant": False, "rto_compliant": False},
        ]

        api = AsyncDisasterRecoveryAPI(mock_async_client)
        result = await api.get_health_summary()

        assert result["ready"] is False
        assert result["health"] == "critical"
        assert result["rpo_compliant"] is False
        assert result["rto_compliant"] is False
        assert result["issues_count"] == 3
