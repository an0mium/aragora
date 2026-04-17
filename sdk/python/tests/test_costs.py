"""Tests for costs namespace route mappings."""

from __future__ import annotations

from unittest.mock import AsyncMock, call, patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestDebateSessionCostRoutes:
    """Tests for debate-session cost SDK endpoints."""

    def test_debate_session_cost_endpoints(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": {}}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            client.costs.get_debate_session_costs("debate-1")
            client.costs.list_debate_cost_line_items(
                "debate-1",
                sort_by="cost",
                order="asc",
                limit=25,
                offset=5,
            )
            client.costs.get_debate_cost_performance("debate-1")

            expected_calls = [
                call("GET", "/api/v1/costs/debates/debate-1"),
                call(
                    "GET",
                    "/api/v1/costs/debates/debate-1/line-items",
                    params={"sort_by": "cost", "order": "asc", "limit": 25, "offset": 5},
                ),
                call("GET", "/api/v1/costs/debates/debate-1/performance"),
            ]
            mock_request.assert_has_calls(expected_calls)
            assert mock_request.call_count == len(expected_calls)
            client.close()

    def test_debate_session_cost_endpoints_encode_debate_id(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": {}}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            client.costs.get_debate_session_costs("debate/1")
            client.costs.list_debate_cost_line_items("debate/1")
            client.costs.get_debate_cost_performance("debate/1")

            expected_calls = [
                call("GET", "/api/v1/costs/debates/debate%2F1"),
                call("GET", "/api/v1/costs/debates/debate%2F1/line-items", params={}),
                call("GET", "/api/v1/costs/debates/debate%2F1/performance"),
            ]
            mock_request.assert_has_calls(expected_calls)
            assert mock_request.call_count == len(expected_calls)
            client.close()

    @pytest.mark.asyncio
    async def test_async_debate_session_cost_endpoints(self) -> None:
        with patch.object(AragoraAsyncClient, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"data": {}}
            async with AragoraAsyncClient(
                base_url="https://api.aragora.ai",
                api_key="test-key",
            ) as client:
                await client.costs.get_debate_session_costs("debate-1")
                await client.costs.list_debate_cost_line_items(
                    "debate-1",
                    sort_by="cost",
                    order="asc",
                    limit=25,
                    offset=5,
                )
                await client.costs.get_debate_cost_performance("debate-1")

            expected_calls = [
                call("GET", "/api/v1/costs/debates/debate-1"),
                call(
                    "GET",
                    "/api/v1/costs/debates/debate-1/line-items",
                    params={"sort_by": "cost", "order": "asc", "limit": 25, "offset": 5},
                ),
                call("GET", "/api/v1/costs/debates/debate-1/performance"),
            ]
            mock_request.assert_has_awaits(expected_calls)
            assert mock_request.await_count == len(expected_calls)

    @pytest.mark.asyncio
    async def test_async_debate_session_cost_endpoints_encode_debate_id(self) -> None:
        with patch.object(AragoraAsyncClient, "request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"data": {}}
            async with AragoraAsyncClient(
                base_url="https://api.aragora.ai",
                api_key="test-key",
            ) as client:
                await client.costs.get_debate_session_costs("debate/1")
                await client.costs.list_debate_cost_line_items("debate/1")
                await client.costs.get_debate_cost_performance("debate/1")

            expected_calls = [
                call("GET", "/api/v1/costs/debates/debate%2F1"),
                call("GET", "/api/v1/costs/debates/debate%2F1/line-items", params={}),
                call("GET", "/api/v1/costs/debates/debate%2F1/performance"),
            ]
            mock_request.assert_has_awaits(expected_calls)
            assert mock_request.await_count == len(expected_calls)
