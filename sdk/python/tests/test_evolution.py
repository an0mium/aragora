"""Tests for Evolution namespace API."""

from __future__ import annotations

from unittest.mock import AsyncMock, call, patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestAgentEvolutionDashboard:
    """Tests for /api/v1/agent-evolution dashboard endpoints."""

    def test_agent_evolution_dashboard_endpoints(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": {}}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            client.evolution.get_agent_evolution_timeline(limit=10, offset=5)
            client.evolution.get_agent_evolution_elo_trends(period="30d")
            client.evolution.get_agent_evolution_pending()
            client.evolution.approve_agent_evolution_change("change-1")
            client.evolution.reject_agent_evolution_change("change-2")

            expected_calls = [
                call(
                    "GET",
                    "/api/v1/agent-evolution/timeline",
                    params={"limit": 10, "offset": 5},
                ),
                call(
                    "GET",
                    "/api/v1/agent-evolution/elo-trends",
                    params={"period": "30d"},
                ),
                call("GET", "/api/v1/agent-evolution/pending"),
                call("POST", "/api/v1/agent-evolution/pending/change-1/approve"),
                call("POST", "/api/v1/agent-evolution/pending/change-2/reject"),
            ]
            mock_request.assert_has_calls(expected_calls)
            assert mock_request.call_count == 5
            client.close()

    @pytest.mark.asyncio
    async def test_async_agent_evolution_dashboard_endpoints(self) -> None:
        with patch.object(AragoraAsyncClient, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"data": {}}
            async with AragoraAsyncClient(
                base_url="https://api.aragora.ai",
                api_key="test-key",
            ) as client:
                await client.evolution.get_agent_evolution_timeline(limit=10, offset=5)
                await client.evolution.get_agent_evolution_elo_trends(period="30d")
                await client.evolution.get_agent_evolution_pending()
                await client.evolution.approve_agent_evolution_change("change-1")
                await client.evolution.reject_agent_evolution_change("change-2")

            expected_calls = [
                call(
                    "GET",
                    "/api/v1/agent-evolution/timeline",
                    params={"limit": 10, "offset": 5},
                ),
                call(
                    "GET",
                    "/api/v1/agent-evolution/elo-trends",
                    params={"period": "30d"},
                ),
                call("GET", "/api/v1/agent-evolution/pending"),
                call("POST", "/api/v1/agent-evolution/pending/change-1/approve"),
                call("POST", "/api/v1/agent-evolution/pending/change-2/reject"),
            ]
            mock_request.assert_has_awaits(expected_calls)
            assert mock_request.await_count == 5
