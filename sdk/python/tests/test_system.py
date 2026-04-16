"""Tests for System namespace API."""

from __future__ import annotations

from unittest.mock import AsyncMock, call, patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestSystemIntelligenceDashboard:
    """Tests for /api/v1/system-intelligence dashboard endpoints."""

    def test_system_intelligence_dashboard_endpoints(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": {}}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            client.system.get_system_intelligence_overview()
            client.system.get_system_intelligence_agent_performance()
            client.system.get_system_intelligence_institutional_memory()
            client.system.get_system_intelligence_improvement_queue()
            client.system.get_system_intelligence_anomalies()
            client.system.get_system_intelligence_events(limit=20)
            client.system.get_system_intelligence_km_sync()
            client.system.get_system_intelligence_nomic_status()
            client.system.get_system_intelligence_debate_queue()

            expected_calls = [
                call("GET", "/api/v1/system-intelligence/overview"),
                call("GET", "/api/v1/system-intelligence/agent-performance"),
                call("GET", "/api/v1/system-intelligence/institutional-memory"),
                call("GET", "/api/v1/system-intelligence/improvement-queue"),
                call("GET", "/api/v1/system-intelligence/anomalies"),
                call("GET", "/api/v1/system-intelligence/events", params={"limit": 20}),
                call("GET", "/api/v1/system-intelligence/km-sync"),
                call("GET", "/api/v1/system-intelligence/nomic-status"),
                call("GET", "/api/v1/system-intelligence/debate-queue"),
            ]
            mock_request.assert_has_calls(expected_calls)
            assert mock_request.call_count == 9
            client.close()

    @pytest.mark.asyncio
    async def test_async_system_intelligence_dashboard_endpoints(self) -> None:
        with patch.object(AragoraAsyncClient, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"data": {}}
            async with AragoraAsyncClient(
                base_url="https://api.aragora.ai",
                api_key="test-key",
            ) as client:
                await client.system.get_system_intelligence_overview()
                await client.system.get_system_intelligence_agent_performance()
                await client.system.get_system_intelligence_institutional_memory()
                await client.system.get_system_intelligence_improvement_queue()
                await client.system.get_system_intelligence_anomalies()
                await client.system.get_system_intelligence_events(limit=20)
                await client.system.get_system_intelligence_km_sync()
                await client.system.get_system_intelligence_nomic_status()
                await client.system.get_system_intelligence_debate_queue()

            expected_calls = [
                call("GET", "/api/v1/system-intelligence/overview"),
                call("GET", "/api/v1/system-intelligence/agent-performance"),
                call("GET", "/api/v1/system-intelligence/institutional-memory"),
                call("GET", "/api/v1/system-intelligence/improvement-queue"),
                call("GET", "/api/v1/system-intelligence/anomalies"),
                call("GET", "/api/v1/system-intelligence/events", params={"limit": 20}),
                call("GET", "/api/v1/system-intelligence/km-sync"),
                call("GET", "/api/v1/system-intelligence/nomic-status"),
                call("GET", "/api/v1/system-intelligence/debate-queue"),
            ]
            mock_request.assert_has_awaits(expected_calls)
            assert mock_request.await_count == 9
