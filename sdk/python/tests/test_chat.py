"""Tests for legacy chat integration routes in the Python SDK namespace."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestChatWebhookRoutes:
    def test_get_status_uses_v1_route(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"enabled": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.chat.get_status()

            mock_request.assert_called_once_with("GET", "/api/v1/chat/status", params=None)
            assert result["enabled"] is True
            client.close()

    def test_receive_telegram_webhook_uses_v1_route(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"ok": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            payload = {"message": {"text": "hello"}}
            result = client.chat.receive_telegram_webhook(payload)

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/chat/telegram/webhook",
                json=payload,
                params=None,
            )
            assert result["ok"] is True
            client.close()

    @pytest.mark.asyncio
    async def test_async_receive_generic_webhook_uses_v1_route(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"handled": True}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                payload = {"event": "message"}
                result = await client.chat.receive_webhook(payload)

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/chat/webhook",
                json=payload,
                params=None,
            )
            assert result["handled"] is True
