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
