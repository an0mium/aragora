"""Tests for the Python feedback SDK namespace."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestFeedbackHub:
    def test_get_hub_stats_uses_v1_route(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": {"total_routed": 3}}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.feedback.get_hub_stats()

            mock_request.assert_called_once_with("GET", "/api/v1/feedback-hub/stats")
            assert result["data"]["total_routed"] == 3
            client.close()

    def test_list_hub_history_query_encodes_limit(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.feedback.list_hub_history(limit=25)

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/feedback-hub/history",
                params={"limit": 25},
            )
            client.close()

    @pytest.mark.asyncio
    async def test_async_get_hub_stats_uses_v1_route(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"data": {"total_routed": 7}}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.feedback.get_hub_stats()

            mock_request.assert_called_once_with("GET", "/api/v1/feedback-hub/stats")
            assert result["data"]["total_routed"] == 7

    @pytest.mark.asyncio
    async def test_async_list_hub_history_query_encodes_limit(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"data": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.feedback.list_hub_history(limit=10)

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/feedback-hub/history",
                params={"limit": 10},
            )
