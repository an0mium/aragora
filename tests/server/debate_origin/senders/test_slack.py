"""Tests for Slack sender for debate origin result routing.

Tests cover:
1. Result message sending via Slack chat.postMessage
2. Rich Block Kit payload delivery for debated answers
3. Receipt posting with a view button
4. Error message delivery
5. Token and threading behavior
6. Error handling for Slack API failures
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.debate_origin.models import DebateOrigin
from aragora.server.debate_origin.senders.slack import (
    _send_slack_error,
    _send_slack_receipt,
    _send_slack_result,
)


@pytest.fixture
def sample_origin() -> DebateOrigin:
    """Create a sample Slack debate origin for testing."""
    return DebateOrigin(
        debate_id="debate-slack-123",
        platform="slack",
        channel_id="C1234567890",
        user_id="user-slack-456",
        thread_id="1711711717.123456",
        metadata={"topic": "Slack Integration Test"},
    )


@pytest.fixture
def sample_origin_no_thread() -> DebateOrigin:
    """Create a Slack origin without thread_id for threading tests."""
    return DebateOrigin(
        debate_id="debate-slack-456",
        platform="slack",
        channel_id="C0987654321",
        user_id="user-slack-789",
        metadata={"topic": "Slack Integration Test"},
    )


@pytest.fixture
def sample_result() -> dict[str, Any]:
    """Create a sample debate result for testing."""
    return {
        "consensus_reached": True,
        "final_answer": "The team reached agreement on the Slack delivery approach.",
        "confidence": 0.91,
        "participants": ["claude", "gpt-4", "gemini"],
        "task": "Evaluate the Slack proposal",
    }


class TestSendSlackResult:
    """Tests for _send_slack_result function."""

    @pytest.mark.asyncio
    async def test_sends_rich_blocks_with_fallback_text(self, sample_origin, sample_result):
        """Rich formatter payloads are sent as Block Kit with plain-text fallback."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"ok": True}
        rich_message = {
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "*Conclusion:*\nUse Slack blocks."},
                }
            ]
        }

        with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test-token"}):
            with patch(
                "aragora.server.debate_origin.senders.slack._format_result_message",
                return_value=rich_message,
            ):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(return_value=mock_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    result = await _send_slack_result(sample_origin, sample_result)

        assert result is True
        call_args = mock_client.post.call_args
        post_data = call_args.kwargs["json"]
        assert post_data["channel"] == sample_origin.channel_id
        assert post_data["blocks"] == rich_message["blocks"]
        assert post_data["text"] == sample_result["final_answer"]
        assert post_data["thread_ts"] == sample_origin.thread_id

    @pytest.mark.asyncio
    async def test_sends_markdown_result_when_formatter_returns_string(
        self, sample_origin, sample_result
    ):
        """String formatter payloads are sent as classic mrkdwn messages."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"ok": True}

        with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test-token"}):
            with patch(
                "aragora.server.debate_origin.senders.slack._format_result_message",
                return_value="**Debate Complete!**",
            ):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(return_value=mock_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    result = await _send_slack_result(sample_origin, sample_result)

        assert result is True
        post_data = mock_client.post.call_args.kwargs["json"]
        assert post_data["text"] == "**Debate Complete!**"
        assert post_data["mrkdwn"] is True
        assert "blocks" not in post_data

    @pytest.mark.asyncio
    async def test_omits_thread_ts_without_thread_id(self, sample_origin_no_thread, sample_result):
        """thread_ts is not sent when the origin is not threaded."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"ok": True}

        with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                await _send_slack_result(sample_origin_no_thread, sample_result)

        post_data = mock_client.post.call_args.kwargs["json"]
        assert "thread_ts" not in post_data

    @pytest.mark.asyncio
    async def test_returns_false_without_bot_token(self, sample_origin, sample_result):
        """Missing Slack bot token disables result delivery."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SLACK_BOT_TOKEN", None)
            result = await _send_slack_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_slack_api_error(self, sample_origin, sample_result):
        """Slack API ok=false responses are treated as failures."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"ok": False, "error": "invalid_blocks"}

        with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_slack_result(sample_origin, sample_result)

        assert result is False


class TestSendSlackReceipt:
    """Tests for _send_slack_receipt function."""

    @pytest.mark.asyncio
    async def test_posts_receipt_with_button(self, sample_origin):
        """Receipt summaries include a primary button to the full receipt."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"ok": True}

        with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_slack_receipt(
                    sample_origin,
                    "✅ **Decision Receipt**\n• Verdict: APPROVED",
                    "https://aragora.ai/receipt/123",
                )

        assert result is True
        post_data = mock_client.post.call_args.kwargs["json"]
        assert post_data["channel"] == sample_origin.channel_id
        assert post_data["thread_ts"] == sample_origin.thread_id
        attachment = post_data["attachments"][0]
        assert attachment["actions"][0]["text"] == "View Full Receipt"
        assert attachment["actions"][0]["url"] == "https://aragora.ai/receipt/123"

    @pytest.mark.asyncio
    async def test_returns_false_without_bot_token(self, sample_origin):
        """Missing Slack bot token disables receipt delivery."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SLACK_BOT_TOKEN", None)
            result = await _send_slack_receipt(sample_origin, "summary", "https://example.com/r")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_connection_error(self, sample_origin):
        """Connection failures are handled gracefully for receipt delivery."""
        with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(side_effect=OSError("Connection refused"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_slack_receipt(
                    sample_origin, "summary", "https://example.com/r"
                )

        assert result is False


class TestSendSlackError:
    """Tests for _send_slack_error function."""

    @pytest.mark.asyncio
    async def test_sends_error_to_thread(self, sample_origin):
        """Error messages reuse the original Slack thread when available."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"ok": True}

        with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_slack_error(sample_origin, "Please reconnect the Slack app.")

        assert result is True
        post_data = mock_client.post.call_args.kwargs["json"]
        assert post_data["channel"] == sample_origin.channel_id
        assert post_data["text"] == "Please reconnect the Slack app."
        assert post_data["thread_ts"] == sample_origin.thread_id

    @pytest.mark.asyncio
    async def test_returns_false_on_api_error(self, sample_origin):
        """Slack API failures are surfaced as False for error delivery."""
        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.json.return_value = {"ok": False}

        with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_slack_error(sample_origin, "Something went wrong.")

        assert result is False
