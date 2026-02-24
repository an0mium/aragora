"""Tests for Webhooks namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestWebhookCRUD:
    """Tests for webhook create, read, update, delete operations."""

    def test_create_webhook_minimal(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"webhook_id": "wh_1", "secret": "sec_abc"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.webhooks.create(
                url="https://example.com/webhook",
                events=["debate.completed", "receipt.created"],
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/webhooks",
                json={
                    "url": "https://example.com/webhook",
                    "events": ["debate.completed", "receipt.created"],
                },
            )
            assert result["webhook_id"] == "wh_1"
            assert result["secret"] == "sec_abc"
            client.close()

    def test_create_webhook_with_all_options(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"webhook_id": "wh_2"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.webhooks.create(
                url="https://example.com/hook",
                events=["debate.completed"],
                secret="my-secret",
                description="Production webhook",
                headers={"X-Custom": "value"},
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/webhooks",
                json={
                    "url": "https://example.com/hook",
                    "events": ["debate.completed"],
                    "secret": "my-secret",
                    "description": "Production webhook",
                    "headers": {"X-Custom": "value"},
                },
            )
            client.close()

    def test_list_webhooks_defaults(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"webhooks": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.webhooks.list()
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/webhooks",
                params={"active_only": True, "limit": 20, "offset": 0},
            )
            assert result["total"] == 0
            client.close()

    def test_list_webhooks_with_params(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"webhooks": [{"webhook_id": "wh_1"}]}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.webhooks.list(active_only=False, limit=50, offset=10)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/webhooks",
                params={"active_only": False, "limit": 50, "offset": 10},
            )
            client.close()


class TestWebhookActions:
    """Tests for rotate secret, test, and get events."""

    def test_get_events(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "events": [
                    {"type": "debate.completed", "description": "Debate finished"},
                    {"type": "receipt.created", "description": "Receipt generated"},
                ]
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.webhooks.get_events()
            mock_request.assert_called_once_with("GET", "/api/v1/webhooks/events")
            assert len(result["events"]) == 2
            client.close()


class TestAsyncWebhooks:
    """Tests for async webhook methods."""

    @pytest.mark.asyncio
    async def test_async_create_webhook(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"webhook_id": "wh_1", "secret": "sec_abc"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.webhooks.create(
                url="https://example.com/webhook",
                events=["debate.completed"],
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/webhooks",
                json={
                    "url": "https://example.com/webhook",
                    "events": ["debate.completed"],
                },
            )
            assert result["webhook_id"] == "wh_1"
            await client.close()

    @pytest.mark.asyncio
    async def test_async_list_webhooks(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"webhooks": [{"webhook_id": "wh_1"}], "total": 1}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.webhooks.list(active_only=False, limit=10, offset=0)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/webhooks",
                params={"active_only": False, "limit": 10, "offset": 0},
            )
            assert result["total"] == 1
            await client.close()
