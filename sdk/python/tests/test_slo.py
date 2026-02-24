"""Tests for SLO (Service Level Objective) namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestSLOStatus:
    """Tests for SLO status methods."""

    def test_get_status(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "status": "healthy",
                "slos": [],
                "alerts": [],
                "summary": {"total": 5, "meeting": 5},
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.slo.get_status()
            mock_request.assert_called_once_with("GET", "/api/v2/slo/status")
            assert result["status"] == "healthy"
            assert result["summary"]["meeting"] == 5
            client.close()

    def test_get_debate_health_default(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": {"status": "healthy"}}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.slo.get_debate_health()
            mock_request.assert_called_once_with(
                "GET",
                "/api/health/slos",
                params={"window": "1h"},
            )
            assert result["data"]["status"] == "healthy"
            client.close()

    def test_get_enforcer_budget(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": {"availability": {"remaining": 99.9}}}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.slo.get_enforcer_budget()
            mock_request.assert_called_once_with("GET", "/api/v1/slo/budget")
            assert "data" in result
            client.close()


class TestAsyncSLO:
    """Tests for async SLO methods."""

    @pytest.mark.asyncio
    async def test_get_status(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"status": "degraded", "slos": [], "alerts": []}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.slo.get_status()
            mock_request.assert_called_once_with("GET", "/api/v2/slo/status")
            assert result["status"] == "degraded"
            await client.close()

    @pytest.mark.asyncio
    async def test_get_debate_health_all_windows(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"data": {"1h": {}, "24h": {}, "7d": {}}}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.slo.get_debate_health(window="24h", all_windows=True)
            mock_request.assert_called_once_with(
                "GET",
                "/api/health/slos",
                params={"window": "24h", "all_windows": "true"},
            )
            assert "data" in result
            await client.close()

    @pytest.mark.asyncio
    async def test_get_enforcer_budget(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"data": {"debate_success": {"remaining": 98.0}}}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.slo.get_enforcer_budget()
            mock_request.assert_called_once_with("GET", "/api/v1/slo/budget")
            assert "data" in result
            await client.close()
