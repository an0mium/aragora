"""Tests for SLO (Service Level Objective) SDK namespace."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock client."""
    return MagicMock()


class TestSLOAPI:
    """Test synchronous SLOAPI."""

    def test_init(self, mock_client: MagicMock) -> None:
        """Test API initialization."""
        from aragora.namespaces.slo import SLOAPI

        api = SLOAPI(mock_client)
        assert api._client is mock_client

    def test_get_status(self, mock_client: MagicMock) -> None:
        """Test get_status calls correct endpoint."""
        from aragora.namespaces.slo import SLOAPI

        mock_client.request.return_value = {
            "status": "healthy",
            "timestamp": "2024-01-15T10:00:00Z",
            "slos": [],
            "alerts": [],
            "summary": {"total": 5, "meeting_target": 5},
        }

        api = SLOAPI(mock_client)
        result = api.get_status()

        mock_client.request.assert_called_once_with("GET", "/api/v2/slo/status")
        assert result["status"] == "healthy"
        assert result["summary"]["total"] == 5

    def test_get_slo(self, mock_client: MagicMock) -> None:
        """Test get_slo calls correct endpoint."""
        from aragora.namespaces.slo import SLOAPI

        mock_client.request.return_value = {
            "name": "availability",
            "current_percent": 99.95,
            "target_percent": 99.9,
            "is_meeting": True,
            "total_requests": 100000,
            "successful_requests": 99950,
            "failed_requests": 50,
        }

        api = SLOAPI(mock_client)
        result = api.get_slo("availability")

        mock_client.request.assert_called_once_with("GET", "/api/v2/slo/availability")
        assert result["name"] == "availability"
        assert result["current_percent"] == 99.95
        assert result["is_meeting"] is True

    def test_get_error_budget(self, mock_client: MagicMock) -> None:
        """Test get_error_budget calls correct endpoint."""
        from aragora.namespaces.slo import SLOAPI

        mock_client.request.return_value = {
            "slo_name": "availability",
            "budget_percent": 0.1,
            "consumed_percent": 0.05,
            "remaining_percent": 0.05,
            "is_exhausted": False,
            "burn_rate": 1.2,
            "window_days": 30,
        }

        api = SLOAPI(mock_client)
        result = api.get_error_budget("availability")

        mock_client.request.assert_called_once_with("GET", "/api/v2/slo/availability/error-budget")
        assert result["slo_name"] == "availability"
        assert result["remaining_percent"] == 0.05
        assert result["is_exhausted"] is False

    def test_list_violations(self, mock_client: MagicMock) -> None:
        """Test list_violations calls correct endpoint."""
        from aragora.namespaces.slo import SLOAPI

        mock_client.request.return_value = {
            "violations": [
                {
                    "slo_name": "availability",
                    "timestamp": "2024-01-14T15:00:00Z",
                    "actual_percent": 99.5,
                    "target_percent": 99.9,
                    "severity": "warning",
                    "resolved": True,
                }
            ],
            "total": 1,
        }

        api = SLOAPI(mock_client)
        result = api.list_violations("availability")

        mock_client.request.assert_called_once_with(
            "GET", "/api/v2/slo/availability/violations", params={}
        )
        assert len(result["violations"]) == 1
        assert result["violations"][0]["severity"] == "warning"

    def test_list_violations_with_params(self, mock_client: MagicMock) -> None:
        """Test list_violations with limit and since parameters."""
        from aragora.namespaces.slo import SLOAPI

        mock_client.request.return_value = {"violations": [], "total": 0}

        api = SLOAPI(mock_client)
        api.list_violations("latency", limit=10, since="2024-01-01T00:00:00Z")

        mock_client.request.assert_called_once_with(
            "GET",
            "/api/v2/slo/latency/violations",
            params={"limit": 10, "since": "2024-01-01T00:00:00Z"},
        )

    def test_is_compliant(self, mock_client: MagicMock) -> None:
        """Test is_compliant calls correct endpoint."""
        from aragora.namespaces.slo import SLOAPI

        mock_client.request.return_value = {
            "compliant": True,
            "slo_name": "availability",
            "current_percent": 99.95,
            "target_percent": 99.9,
        }

        api = SLOAPI(mock_client)
        result = api.is_compliant("availability")

        mock_client.request.assert_called_once_with("GET", "/api/v2/slo/availability/compliant")
        assert result["compliant"] is True

    def test_get_alerts(self, mock_client: MagicMock) -> None:
        """Test get_alerts calls correct endpoint."""
        from aragora.namespaces.slo import SLOAPI

        mock_client.request.return_value = {
            "alerts": [
                {
                    "slo_name": "latency",
                    "severity": "critical",
                    "message": "Latency SLO breached",
                    "triggered_at": "2024-01-15T09:00:00Z",
                    "acknowledged": False,
                }
            ],
            "total": 1,
        }

        api = SLOAPI(mock_client)
        result = api.get_alerts("latency")

        mock_client.request.assert_called_once_with("GET", "/api/v2/slo/latency/alerts", params={})
        assert len(result["alerts"]) == 1
        assert result["alerts"][0]["severity"] == "critical"

    def test_get_alerts_active_only(self, mock_client: MagicMock) -> None:
        """Test get_alerts with active_only parameter."""
        from aragora.namespaces.slo import SLOAPI

        mock_client.request.return_value = {"alerts": [], "total": 0}

        api = SLOAPI(mock_client)
        api.get_alerts("latency", active_only=True)

        mock_client.request.assert_called_once_with(
            "GET", "/api/v2/slo/latency/alerts", params={"active_only": True}
        )


@pytest.fixture
def mock_async_client() -> MagicMock:
    """Create a mock async client."""
    from unittest.mock import AsyncMock

    client = MagicMock()
    client.request = AsyncMock()
    return client


class TestAsyncSLOAPI:
    """Test asynchronous AsyncSLOAPI."""

    @pytest.mark.asyncio
    async def test_init(self, mock_async_client: MagicMock) -> None:
        """Test API initialization."""
        from aragora.namespaces.slo import AsyncSLOAPI

        api = AsyncSLOAPI(mock_async_client)
        assert api._client is mock_async_client

    @pytest.mark.asyncio
    async def test_get_status(self, mock_async_client: MagicMock) -> None:
        """Test get_status calls correct endpoint."""
        from aragora.namespaces.slo import AsyncSLOAPI

        mock_async_client.request.return_value = {
            "status": "healthy",
            "timestamp": "2024-01-15T10:00:00Z",
        }

        api = AsyncSLOAPI(mock_async_client)
        result = await api.get_status()

        mock_async_client.request.assert_called_once_with("GET", "/api/v2/slo/status")
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_get_slo(self, mock_async_client: MagicMock) -> None:
        """Test get_slo calls correct endpoint."""
        from aragora.namespaces.slo import AsyncSLOAPI

        mock_async_client.request.return_value = {
            "name": "debate-success",
            "current_percent": 98.5,
            "target_percent": 95.0,
            "is_meeting": True,
        }

        api = AsyncSLOAPI(mock_async_client)
        result = await api.get_slo("debate-success")

        mock_async_client.request.assert_called_once_with("GET", "/api/v2/slo/debate-success")
        assert result["name"] == "debate-success"
        assert result["is_meeting"] is True

    @pytest.mark.asyncio
    async def test_get_error_budget(self, mock_async_client: MagicMock) -> None:
        """Test get_error_budget calls correct endpoint."""
        from aragora.namespaces.slo import AsyncSLOAPI

        mock_async_client.request.return_value = {
            "slo_name": "latency",
            "remaining_percent": 0.02,
            "is_exhausted": False,
        }

        api = AsyncSLOAPI(mock_async_client)
        result = await api.get_error_budget("latency")

        mock_async_client.request.assert_called_once_with("GET", "/api/v2/slo/latency/error-budget")
        assert result["remaining_percent"] == 0.02

    @pytest.mark.asyncio
    async def test_list_violations(self, mock_async_client: MagicMock) -> None:
        """Test list_violations calls correct endpoint."""
        from aragora.namespaces.slo import AsyncSLOAPI

        mock_async_client.request.return_value = {"violations": [], "total": 0}

        api = AsyncSLOAPI(mock_async_client)
        result = await api.list_violations("availability")

        mock_async_client.request.assert_called_once_with(
            "GET", "/api/v2/slo/availability/violations", params={}
        )
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_list_violations_with_params(self, mock_async_client: MagicMock) -> None:
        """Test list_violations with limit and since parameters."""
        from aragora.namespaces.slo import AsyncSLOAPI

        mock_async_client.request.return_value = {"violations": [], "total": 0}

        api = AsyncSLOAPI(mock_async_client)
        await api.list_violations("latency", limit=5, since="2024-01-10T00:00:00Z")

        mock_async_client.request.assert_called_once_with(
            "GET",
            "/api/v2/slo/latency/violations",
            params={"limit": 5, "since": "2024-01-10T00:00:00Z"},
        )

    @pytest.mark.asyncio
    async def test_is_compliant(self, mock_async_client: MagicMock) -> None:
        """Test is_compliant calls correct endpoint."""
        from aragora.namespaces.slo import AsyncSLOAPI

        mock_async_client.request.return_value = {
            "compliant": False,
            "slo_name": "latency",
            "current_percent": 94.5,
            "target_percent": 99.0,
        }

        api = AsyncSLOAPI(mock_async_client)
        result = await api.is_compliant("latency")

        mock_async_client.request.assert_called_once_with("GET", "/api/v2/slo/latency/compliant")
        assert result["compliant"] is False

    @pytest.mark.asyncio
    async def test_get_alerts(self, mock_async_client: MagicMock) -> None:
        """Test get_alerts calls correct endpoint."""
        from aragora.namespaces.slo import AsyncSLOAPI

        mock_async_client.request.return_value = {
            "alerts": [{"severity": "warning", "acknowledged": True}],
            "total": 1,
        }

        api = AsyncSLOAPI(mock_async_client)
        result = await api.get_alerts("availability")

        mock_async_client.request.assert_called_once_with(
            "GET", "/api/v2/slo/availability/alerts", params={}
        )
        assert len(result["alerts"]) == 1

    @pytest.mark.asyncio
    async def test_get_alerts_active_only(self, mock_async_client: MagicMock) -> None:
        """Test get_alerts with active_only parameter."""
        from aragora.namespaces.slo import AsyncSLOAPI

        mock_async_client.request.return_value = {"alerts": [], "total": 0}

        api = AsyncSLOAPI(mock_async_client)
        await api.get_alerts("availability", active_only=True)

        mock_async_client.request.assert_called_once_with(
            "GET", "/api/v2/slo/availability/alerts", params={"active_only": True}
        )
