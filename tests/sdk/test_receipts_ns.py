"""Tests for the Receipts SDK namespace."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora_sdk.namespaces.receipts import AsyncReceiptsAPI, ReceiptsAPI


@pytest.fixture
def sync_client():
    client = MagicMock()
    client.request.return_value = {"ok": True}
    return client


@pytest.fixture
def async_client():
    client = MagicMock()
    client.request = AsyncMock(return_value={"ok": True})
    return client


class TestReceiptsAPI:
    def test_formatted_uses_v2_route(self, sync_client):
        api = ReceiptsAPI(sync_client)

        api.formatted("rcpt-123", "slack", compact=True)

        sync_client.request.assert_called_once_with(
            "GET",
            "/api/v2/receipts/rcpt-123/formatted/slack",
            params={"compact": "true"},
        )

    def test_send_to_channel_uses_v2_route(self, sync_client):
        api = ReceiptsAPI(sync_client)

        api.send_to_channel(
            "rcpt-123",
            "email",
            "user@example.com",
            options={"compact": True},
        )

        sync_client.request.assert_called_once_with(
            "POST",
            "/api/v2/receipts/rcpt-123/send-to-channel",
            json={
                "channel_type": "email",
                "channel_id": "user@example.com",
                "options": {"compact": True},
            },
        )

    def test_verify_signature_uses_v2_route(self, sync_client):
        api = ReceiptsAPI(sync_client)

        api.verify_signature("rcpt-123")

        sync_client.request.assert_called_once_with(
            "POST",
            "/api/v2/receipts/rcpt-123/verify-signature",
        )


class TestAsyncReceiptsAPI:
    @pytest.mark.asyncio
    async def test_formatted_uses_v2_route(self, async_client):
        api = AsyncReceiptsAPI(async_client)

        await api.formatted("rcpt-123", "slack", compact=True)

        async_client.request.assert_awaited_once_with(
            "GET",
            "/api/v2/receipts/rcpt-123/formatted/slack",
            params={"compact": "true"},
        )

    @pytest.mark.asyncio
    async def test_send_to_channel_uses_v2_route(self, async_client):
        api = AsyncReceiptsAPI(async_client)

        await api.send_to_channel(
            "rcpt-123",
            "email",
            "user@example.com",
            options={"compact": True},
        )

        async_client.request.assert_awaited_once_with(
            "POST",
            "/api/v2/receipts/rcpt-123/send-to-channel",
            json={
                "channel_type": "email",
                "channel_id": "user@example.com",
                "options": {"compact": True},
            },
        )

    @pytest.mark.asyncio
    async def test_verify_signature_uses_v2_route(self, async_client):
        api = AsyncReceiptsAPI(async_client)

        await api.verify_signature("rcpt-123")

        async_client.request.assert_awaited_once_with(
            "POST",
            "/api/v2/receipts/rcpt-123/verify-signature",
        )
