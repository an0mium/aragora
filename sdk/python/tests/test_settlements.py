"""Tests for Settlements namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestSettlements:
    """Tests for settlements methods."""

    def test_list_pending(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": {"count": 2}}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.settlements.list_pending(limit=25)
            mock_request.assert_called_once_with("GET", "/api/v1/settlements", params={"limit": 25})
            assert result["data"]["count"] == 2
            client.close()

    def test_settle_batch(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": {"count": 1}}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.settlements.settle_batch(
                settlements=[{"settlement_id": "s-1", "outcome": "correct"}],
                settled_by="sdk-test",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/settlements/batch",
                json={
                    "settlements": [{"settlement_id": "s-1", "outcome": "correct"}],
                    "settled_by": "sdk-test",
                },
            )
            assert result["data"]["count"] == 1
            client.close()


class TestAsyncSettlements:
    """Tests for async settlements methods."""

    @pytest.mark.asyncio
    async def test_get_agent_accuracy(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"data": {"accuracy": 0.9}}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.settlements.get_agent_accuracy("critic-1")
            mock_request.assert_called_once_with(
                "GET", "/api/v1/settlements/agent/critic-1/accuracy"
            )
            assert result["data"]["accuracy"] == 0.9
            await client.close()

    @pytest.mark.asyncio
    async def test_settle(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"data": {"settlement_id": "s-1"}}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.settlements.settle(
                "s-1",
                outcome="correct",
                evidence="verified",
                settled_by="sdk-test",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/settlements/s-1/settle",
                json={
                    "outcome": "correct",
                    "evidence": "verified",
                    "settled_by": "sdk-test",
                },
            )
            assert result["data"]["settlement_id"] == "s-1"
            await client.close()
