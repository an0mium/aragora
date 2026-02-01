"""Tests for Webhooks namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


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

    def test_get_webhook(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "webhook_id": "wh_1",
                "url": "https://example.com/webhook",
                "active": True,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.webhooks.get("wh_1")
            mock_request.assert_called_once_with("GET", "/api/v1/webhooks/wh_1")
            assert result["active"] is True
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

    def test_update_webhook(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"webhook_id": "wh_1", "active": False}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.webhooks.update(
                "wh_1",
                url="https://new-url.com/hook",
                events=["debate.completed"],
                active=False,
                description="Updated",
                headers={"Authorization": "Bearer tok"},
            )
            mock_request.assert_called_once_with(
                "PUT",
                "/api/v1/webhooks/wh_1",
                json={
                    "url": "https://new-url.com/hook",
                    "events": ["debate.completed"],
                    "active": False,
                    "description": "Updated",
                    "headers": {"Authorization": "Bearer tok"},
                },
            )
            assert result["active"] is False
            client.close()

    def test_delete_webhook(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"deleted": True}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.webhooks.delete("wh_1")
            mock_request.assert_called_once_with("DELETE", "/api/v1/webhooks/wh_1")
            assert result["deleted"] is True
            client.close()


class TestWebhookDeliveries:
    """Tests for webhook delivery history and retry."""

    def test_get_deliveries_defaults(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"deliveries": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.webhooks.get_deliveries("wh_1")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/webhooks/wh_1/deliveries",
                params={"limit": 20, "offset": 0},
            )
            assert result["total"] == 0
            client.close()

    def test_get_deliveries_with_status_filter(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"deliveries": [{"id": "del_1"}]}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.webhooks.get_deliveries("wh_1", status="failed", limit=5, offset=0)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/webhooks/wh_1/deliveries",
                params={"limit": 5, "offset": 0, "status": "failed"},
            )
            client.close()

    def test_get_single_delivery(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "delivery_id": "del_1",
                "status": "success",
                "response_code": 200,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.webhooks.get_delivery("wh_1", "del_1")
            mock_request.assert_called_once_with("GET", "/api/v1/webhooks/wh_1/deliveries/del_1")
            assert result["response_code"] == 200
            client.close()

    def test_retry_delivery(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"retried": True, "new_delivery_id": "del_2"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.webhooks.retry_delivery("wh_1", "del_1")
            mock_request.assert_called_once_with(
                "POST", "/api/v1/webhooks/wh_1/deliveries/del_1/retry"
            )
            assert result["retried"] is True
            client.close()


class TestWebhookActions:
    """Tests for rotate secret, test, and get events."""

    def test_rotate_secret(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"new_secret": "sec_new_123"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.webhooks.rotate_secret("wh_1")
            mock_request.assert_called_once_with("POST", "/api/v1/webhooks/wh_1/rotate-secret")
            assert result["new_secret"] == "sec_new_123"
            client.close()

    def test_test_webhook_no_event_type(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"success": True, "response_code": 200}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.webhooks.test("wh_1")
            mock_request.assert_called_once_with("POST", "/api/v1/webhooks/wh_1/test", json={})
            assert result["success"] is True
            client.close()

    def test_test_webhook_with_event_type(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"success": True}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.webhooks.test("wh_1", event_type="debate.completed")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/webhooks/wh_1/test",
                json={"event_type": "debate.completed"},
            )
            client.close()

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

    @pytest.mark.asyncio
    async def test_async_update_webhook(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"webhook_id": "wh_1", "active": False}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.webhooks.update("wh_1", active=False)
            mock_request.assert_called_once_with(
                "PUT",
                "/api/v1/webhooks/wh_1",
                json={"active": False},
            )
            assert result["active"] is False
            await client.close()

    @pytest.mark.asyncio
    async def test_async_delete_webhook(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"deleted": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.webhooks.delete("wh_1")
            mock_request.assert_called_once_with("DELETE", "/api/v1/webhooks/wh_1")
            assert result["deleted"] is True
            await client.close()

    @pytest.mark.asyncio
    async def test_async_get_deliveries_with_status(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"deliveries": [{"id": "del_1", "status": "failed"}]}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.webhooks.get_deliveries("wh_1", status="failed")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/webhooks/wh_1/deliveries",
                params={"limit": 20, "offset": 0, "status": "failed"},
            )
            assert result["deliveries"][0]["status"] == "failed"
            await client.close()

    @pytest.mark.asyncio
    async def test_async_test_webhook(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"success": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.webhooks.test("wh_1", event_type="receipt.created")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/webhooks/wh_1/test",
                json={"event_type": "receipt.created"},
            )
            assert result["success"] is True
            await client.close()
