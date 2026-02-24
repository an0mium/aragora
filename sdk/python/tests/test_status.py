"""Tests for Status namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestStatus:
    """Tests for status methods."""

    def test_get_uptime(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": {"current": {"status": "operational"}}}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.status.get_uptime()
            mock_request.assert_called_once_with("GET", "/api/v1/status/uptime")
            assert result["data"]["current"]["status"] == "operational"
            client.close()


class TestAsyncStatus:
    """Tests for async status methods."""

    @pytest.mark.asyncio
    async def test_get_uptime(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"data": {"periods": {"24h": {"uptime_pct": 100.0}}}}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.status.get_uptime()
            mock_request.assert_called_once_with("GET", "/api/v1/status/uptime")
            assert "24h" in result["data"]["periods"]
            await client.close()
