"""Tests for monitoring namespace route mappings."""

from __future__ import annotations

from unittest.mock import AsyncMock, call, patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestObservabilityRoutes:
    """Tests for /api/observability monitoring endpoints."""

    def test_observability_endpoints(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": {}}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            client.monitoring.get_observability_dashboard()
            client.monitoring.get_observability_metrics()
            client.monitoring.list_crashes(limit=25, offset=5)
            client.monitoring.report_crashes([{"message": "boom"}])
            client.monitoring.get_crash_stats()

            expected_calls = [
                call("GET", "/api/observability/dashboard"),
                call("GET", "/api/observability/metrics"),
                call(
                    "GET",
                    "/api/observability/crashes",
                    params={"limit": 25, "offset": 5},
                ),
                call(
                    "POST",
                    "/api/observability/crashes",
                    json={"reports": [{"message": "boom"}]},
                ),
                call("GET", "/api/observability/crashes/stats"),
            ]
            mock_request.assert_has_calls(expected_calls)
            assert mock_request.call_count == len(expected_calls)
            client.close()

    @pytest.mark.asyncio
    async def test_async_observability_endpoints(self) -> None:
        with patch.object(AragoraAsyncClient, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"data": {}}
            async with AragoraAsyncClient(
                base_url="https://api.aragora.ai",
                api_key="test-key",
            ) as client:
                await client.monitoring.get_observability_dashboard()
                await client.monitoring.get_observability_metrics()
                await client.monitoring.list_crashes(limit=25, offset=5)
                await client.monitoring.report_crashes([{"message": "boom"}])
                await client.monitoring.get_crash_stats()

            expected_calls = [
                call("GET", "/api/observability/dashboard"),
                call("GET", "/api/observability/metrics"),
                call(
                    "GET",
                    "/api/observability/crashes",
                    params={"limit": 25, "offset": 5},
                ),
                call(
                    "POST",
                    "/api/observability/crashes",
                    json={"reports": [{"message": "boom"}]},
                ),
                call("GET", "/api/observability/crashes/stats"),
            ]
            mock_request.assert_has_awaits(expected_calls)
            assert mock_request.await_count == len(expected_calls)
