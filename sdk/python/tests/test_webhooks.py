"""Tests for Webhooks namespace API."""

from __future__ import annotations

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


class TestWebhooksList:
    """Tests for listing webhooks."""

    def test_list_webhooks_default(self, client: AragoraClient, mock_request) -> None:
        """List webhooks with default parameters."""
        mock_request.return_value = {"webhooks": [], "total": 0}

        client.webhooks.list()

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/webhooks",
            params={"active_only": True, "limit": 20, "offset": 0},
        )

    def test_list_webhooks_all(self, client: AragoraClient, mock_request) -> None:
        """List all webhooks including inactive."""
        mock_request.return_value = {"webhooks": []}

        client.webhooks.list(active_only=False, limit=50)

        call_params = mock_request.call_args[1]["params"]
        assert call_params["active_only"] is False
        assert call_params["limit"] == 50


class TestWebhooksGet:
    """Tests for getting webhook details."""

    def test_get_webhook(self, client: AragoraClient, mock_request) -> None:
        """Get a webhook by ID."""
        mock_request.return_value = {
            "webhook_id": "wh_123",
            "url": "https://example.com/hook",
            "events": ["debate.completed"],
        }

        result = client.webhooks.get("wh_123")

        mock_request.assert_called_once_with("GET", "/api/v1/webhooks/wh_123")
        assert result["url"] == "https://example.com/hook"


class TestWebhooksCreate:
    """Tests for webhook creation."""

    def test_create_webhook_minimal(self, client: AragoraClient, mock_request) -> None:
        """Create a webhook with required fields only."""
        mock_request.return_value = {"webhook_id": "wh_new", "secret": "sec_xxx"}

        result = client.webhooks.create(
            url="https://example.com/hook",
            events=["debate.completed"],
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/webhooks",
            json={
                "url": "https://example.com/hook",
                "events": ["debate.completed"],
            },
        )
        assert "secret" in result

    def test_create_webhook_full(self, client: AragoraClient, mock_request) -> None:
        """Create a webhook with all optional fields."""
        mock_request.return_value = {"webhook_id": "wh_new"}

        client.webhooks.create(
            url="https://example.com/hook",
            events=["debate.completed", "receipt.created"],
            secret="my-secret-key",
            description="Production webhook",
            headers={"X-Custom": "value"},
        )

        call_json = mock_request.call_args[1]["json"]
        assert call_json["secret"] == "my-secret-key"
        assert call_json["description"] == "Production webhook"
        assert call_json["headers"] == {"X-Custom": "value"}
        assert len(call_json["events"]) == 2


class TestWebhooksUpdate:
    """Tests for webhook updates."""

    def test_update_webhook_url(self, client: AragoraClient, mock_request) -> None:
        """Update webhook URL."""
        mock_request.return_value = {"webhook_id": "wh_123"}

        client.webhooks.update("wh_123", url="https://new.example.com/hook")

        mock_request.assert_called_once_with(
            "PUT",
            "/api/v1/webhooks/wh_123",
            json={"url": "https://new.example.com/hook"},
        )

    def test_update_webhook_deactivate(self, client: AragoraClient, mock_request) -> None:
        """Deactivate a webhook."""
        mock_request.return_value = {"webhook_id": "wh_123", "active": False}

        client.webhooks.update("wh_123", active=False)

        mock_request.assert_called_once_with(
            "PUT",
            "/api/v1/webhooks/wh_123",
            json={"active": False},
        )

    def test_update_webhook_events(self, client: AragoraClient, mock_request) -> None:
        """Update webhook subscribed events."""
        mock_request.return_value = {"webhook_id": "wh_123"}

        client.webhooks.update("wh_123", events=["debate.completed", "debate.started"])

        call_json = mock_request.call_args[1]["json"]
        assert call_json["events"] == ["debate.completed", "debate.started"]


class TestWebhooksDelete:
    """Tests for webhook deletion."""

    def test_delete_webhook(self, client: AragoraClient, mock_request) -> None:
        """Delete a webhook."""
        mock_request.return_value = {"deleted": True}

        result = client.webhooks.delete("wh_123")

        mock_request.assert_called_once_with("DELETE", "/api/v1/webhooks/wh_123")
        assert result["deleted"] is True


class TestWebhooksSecrets:
    """Tests for webhook secret operations."""

    def test_rotate_secret(self, client: AragoraClient, mock_request) -> None:
        """Rotate webhook signing secret."""
        mock_request.return_value = {"secret": "new_secret_xxx"}

        result = client.webhooks.rotate_secret("wh_123")

        mock_request.assert_called_once_with("POST", "/api/v1/webhooks/wh_123/rotate-secret")
        assert "secret" in result


class TestWebhooksDeliveries:
    """Tests for webhook delivery operations."""

    def test_get_deliveries_default(self, client: AragoraClient, mock_request) -> None:
        """Get delivery history with defaults."""
        mock_request.return_value = {"deliveries": [], "total": 0}

        client.webhooks.get_deliveries("wh_123")

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/webhooks/wh_123/deliveries",
            params={"limit": 20, "offset": 0},
        )

    def test_get_deliveries_filtered(self, client: AragoraClient, mock_request) -> None:
        """Get delivery history filtered by status."""
        mock_request.return_value = {"deliveries": []}

        client.webhooks.get_deliveries("wh_123", status="failed", limit=10)

        call_params = mock_request.call_args[1]["params"]
        assert call_params["status"] == "failed"
        assert call_params["limit"] == 10

    def test_get_specific_delivery(self, client: AragoraClient, mock_request) -> None:
        """Get a specific delivery by ID."""
        mock_request.return_value = {
            "delivery_id": "del_1",
            "status": "success",
            "response_code": 200,
        }

        result = client.webhooks.get_delivery("wh_123", "del_1")

        mock_request.assert_called_once_with("GET", "/api/v1/webhooks/wh_123/deliveries/del_1")
        assert result["status"] == "success"

    def test_retry_delivery(self, client: AragoraClient, mock_request) -> None:
        """Retry a failed delivery."""
        mock_request.return_value = {"retried": True}

        result = client.webhooks.retry_delivery("wh_123", "del_1")

        mock_request.assert_called_once_with(
            "POST", "/api/v1/webhooks/wh_123/deliveries/del_1/retry"
        )
        assert result["retried"] is True


class TestWebhooksTest:
    """Tests for webhook testing."""

    def test_send_test_event(self, client: AragoraClient, mock_request) -> None:
        """Send a test event to webhook."""
        mock_request.return_value = {"success": True, "response_code": 200}

        result = client.webhooks.test("wh_123")

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/webhooks/wh_123/test",
            json={},
        )
        assert result["success"] is True

    def test_send_test_event_specific_type(self, client: AragoraClient, mock_request) -> None:
        """Send a test event with specific event type."""
        mock_request.return_value = {"success": True}

        client.webhooks.test("wh_123", event_type="debate.completed")

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/webhooks/wh_123/test",
            json={"event_type": "debate.completed"},
        )


class TestWebhooksEvents:
    """Tests for webhook event types."""

    def test_get_events(self, client: AragoraClient, mock_request) -> None:
        """Get available webhook event types."""
        mock_request.return_value = {
            "events": [
                {"type": "debate.completed", "description": "Debate finished"},
                {"type": "receipt.created", "description": "Receipt generated"},
            ]
        }

        result = client.webhooks.get_events()

        mock_request.assert_called_once_with("GET", "/api/v1/webhooks/events")
        assert len(result["events"]) == 2


class TestAsyncWebhooks:
    """Tests for async webhooks API."""

    @pytest.mark.asyncio
    async def test_async_list_webhooks(self, mock_async_request) -> None:
        """List webhooks asynchronously."""
        mock_async_request.return_value = {"webhooks": []}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            await client.webhooks.list()

            mock_async_request.assert_called_once_with(
                "GET",
                "/api/v1/webhooks",
                params={"active_only": True, "limit": 20, "offset": 0},
            )

    @pytest.mark.asyncio
    async def test_async_create_webhook(self, mock_async_request) -> None:
        """Create a webhook asynchronously."""
        mock_async_request.return_value = {"webhook_id": "wh_async"}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.webhooks.create(
                url="https://example.com/hook",
                events=["debate.completed"],
            )

            assert result["webhook_id"] == "wh_async"

    @pytest.mark.asyncio
    async def test_async_rotate_secret(self, mock_async_request) -> None:
        """Rotate secret asynchronously."""
        mock_async_request.return_value = {"secret": "new_sec"}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.webhooks.rotate_secret("wh_123")

            mock_async_request.assert_called_once_with(
                "POST", "/api/v1/webhooks/wh_123/rotate-secret"
            )
            assert result["secret"] == "new_sec"

    @pytest.mark.asyncio
    async def test_async_retry_delivery(self, mock_async_request) -> None:
        """Retry a delivery asynchronously."""
        mock_async_request.return_value = {"retried": True}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.webhooks.retry_delivery("wh_123", "del_1")

            assert result["retried"] is True
