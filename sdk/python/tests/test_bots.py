"""Tests for Bots namespace API."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


@pytest.mark.parametrize(
    ("method_name", "path"),
    [
        ("slack_commands", "/api/v1/bots/slack/commands"),
        ("slack_events", "/api/v1/bots/slack/events"),
        ("slack_interactions", "/api/v1/bots/slack/interactions"),
    ],
)
def test_slack_webhook_routes_sync(method_name: str, path: str) -> None:
    """Route Slack bot webhook helpers through the bots namespace."""
    payload = {"team_id": "T123", "event": {"type": "message"}}

    with patch.object(AragoraClient, "request") as mock_request:
        mock_request.return_value = {"ok": True}

        client = AragoraClient(base_url="https://api.aragora.ai")
        getattr(client.bots, method_name)(payload)

        mock_request.assert_called_once_with("POST", path, json=payload)
        client.close()


def test_slack_status_sync() -> None:
    """Get Slack bot status through bots namespace."""
    with patch.object(AragoraClient, "request") as mock_request:
        mock_request.return_value = {"connected": True}

        client = AragoraClient(base_url="https://api.aragora.ai")
        client.bots.slack_status()

        mock_request.assert_called_once_with("GET", "/api/v1/bots/slack/status")
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "path"),
    [
        ("slack_commands", "/api/v1/bots/slack/commands"),
        ("slack_events", "/api/v1/bots/slack/events"),
        ("slack_interactions", "/api/v1/bots/slack/interactions"),
    ],
)
async def test_slack_webhook_routes_async(method_name: str, path: str) -> None:
    """Route async Slack bot webhook helpers through the bots namespace."""
    payload = {"team_id": "T123", "event": {"type": "app_mention"}}

    with patch.object(AragoraAsyncClient, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = {"ok": True}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            await getattr(client.bots, method_name)(payload)

        mock_request.assert_awaited_once_with("POST", path, json=payload)


@pytest.mark.asyncio
async def test_slack_status_async() -> None:
    """Get Slack bot status through async bots namespace."""
    with patch.object(AragoraAsyncClient, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = {"connected": True}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            await client.bots.slack_status()

        mock_request.assert_awaited_once_with("GET", "/api/v1/bots/slack/status")
