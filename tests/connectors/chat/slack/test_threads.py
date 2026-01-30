"""
Tests for SlackThreadManager - Slack thread management module.

Tests cover:
- Thread info retrieval using conversations.replies
- Thread messages retrieval with pagination
- Thread listing in channels
- Reply to thread operations
- Broadcast thread replies
- Thread statistics
- Thread participants
- Platform name property
- Error handling and edge cases
- httpx availability checks
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_connector():
    """Create a mock SlackConnector for thread manager tests."""
    connector = MagicMock()
    connector.bot_token = "xoxb-test-token"
    connector._request_timeout = 30.0
    connector._slack_api_request = AsyncMock()
    connector.send_message = AsyncMock()
    return connector


@pytest.fixture
def thread_manager(mock_connector):
    """Create SlackThreadManager with mock connector."""
    from aragora.connectors.chat.slack import SlackThreadManager

    return SlackThreadManager(mock_connector)


@pytest.fixture
def mock_thread_response():
    """Factory for mock thread API responses."""

    def _make(
        thread_ts: str = "1704067200.000001",
        user_id: str = "U12345",
        text: str = "Thread root message",
        reply_count: int = 5,
        reply_users: list[str] | None = None,
        latest_reply: str | None = "1704153600.000002",
    ):
        return {
            "ok": True,
            "messages": [
                {
                    "ts": thread_ts,
                    "user": user_id,
                    "text": text,
                    "reply_count": reply_count,
                    "reply_users": reply_users or ["U12345", "U67890"],
                    "latest_reply": latest_reply,
                }
            ],
        }

    return _make


@pytest.fixture
def mock_messages_response():
    """Factory for mock thread messages API responses."""

    def _make(
        messages: list[dict[str, Any]] | None = None,
        next_cursor: str | None = None,
    ):
        if messages is None:
            messages = [
                {"ts": "1704067200.000001", "user": "U12345", "text": "Root message"},
                {"ts": "1704067260.000002", "user": "U67890", "text": "Reply 1"},
                {"ts": "1704067320.000003", "user": "U12345", "text": "Reply 2"},
            ]
        return {
            "ok": True,
            "messages": messages,
            "response_metadata": {"next_cursor": next_cursor or ""},
        }

    return _make


# ---------------------------------------------------------------------------
# Platform Name Tests
# ---------------------------------------------------------------------------


class TestPlatformName:
    """Tests for platform_name property."""

    def test_platform_name_is_slack(self, thread_manager):
        """Should return 'slack' as platform name."""
        assert thread_manager.platform_name == "slack"


# ---------------------------------------------------------------------------
# Get Thread Tests
# ---------------------------------------------------------------------------


class TestGetThread:
    """Tests for get_thread method."""

    @pytest.mark.asyncio
    async def test_get_thread_success(self, thread_manager, mock_connector, mock_thread_response):
        """Should get thread info from conversations.replies."""
        mock_connector._slack_api_request.return_value = (
            True,
            mock_thread_response(),
            None,
        )

        thread = await thread_manager.get_thread("1704067200.000001", "C12345")

        assert thread.id == "1704067200.000001"
        assert thread.channel_id == "C12345"
        assert thread.platform == "slack"
        assert thread.message_count == 6  # 5 replies + root
        assert thread.participant_count == 3  # 2 reply_users + OP

    @pytest.mark.asyncio
    async def test_get_thread_extracts_created_by(
        self, thread_manager, mock_connector, mock_thread_response
    ):
        """Should extract created_by from root message user."""
        mock_connector._slack_api_request.return_value = (
            True,
            mock_thread_response(user_id="U_CREATOR"),
            None,
        )

        thread = await thread_manager.get_thread("1704067200.000001", "C12345")

        assert thread.created_by == "U_CREATOR"

    @pytest.mark.asyncio
    async def test_get_thread_extracts_title(
        self, thread_manager, mock_connector, mock_thread_response
    ):
        """Should extract title from root message text (truncated to 100 chars)."""
        long_text = "A" * 200
        mock_connector._slack_api_request.return_value = (
            True,
            mock_thread_response(text=long_text),
            None,
        )

        thread = await thread_manager.get_thread("1704067200.000001", "C12345")

        assert thread.title == "A" * 100

    @pytest.mark.asyncio
    async def test_get_thread_calculates_timestamps(
        self, thread_manager, mock_connector, mock_thread_response
    ):
        """Should calculate created_at from thread_ts and updated_at from latest_reply."""
        mock_connector._slack_api_request.return_value = (
            True,
            mock_thread_response(
                thread_ts="1704067200.000001",  # 2024-01-01T00:00:00
                latest_reply="1704153600.000002",  # 2024-01-02T00:00:00
            ),
            None,
        )

        thread = await thread_manager.get_thread("1704067200.000001", "C12345")

        assert thread.created_at.year == 2024
        assert thread.updated_at > thread.created_at

    @pytest.mark.asyncio
    async def test_get_thread_no_latest_reply(
        self, thread_manager, mock_connector, mock_thread_response
    ):
        """Should use created_at when no latest_reply available."""
        mock_connector._slack_api_request.return_value = (
            True,
            mock_thread_response(latest_reply=None),
            None,
        )

        thread = await thread_manager.get_thread("1704067200.000001", "C12345")

        assert thread.updated_at == thread.created_at

    @pytest.mark.asyncio
    async def test_get_thread_not_found(self, thread_manager, mock_connector):
        """Should raise ThreadNotFoundError when thread doesn't exist."""
        from aragora.connectors.chat.thread_manager import ThreadNotFoundError

        mock_connector._slack_api_request.return_value = (
            False,
            {"ok": False, "error": "thread_not_found"},
            "thread_not_found",
        )

        with pytest.raises(ThreadNotFoundError) as exc_info:
            await thread_manager.get_thread("invalid_ts", "C12345")

        assert exc_info.value.thread_id == "invalid_ts"
        assert exc_info.value.channel_id == "C12345"
        assert exc_info.value.platform == "slack"

    @pytest.mark.asyncio
    async def test_get_thread_message_not_found(self, thread_manager, mock_connector):
        """Should raise ThreadNotFoundError for message_not_found error."""
        from aragora.connectors.chat.thread_manager import ThreadNotFoundError

        mock_connector._slack_api_request.return_value = (
            False,
            {"ok": False, "error": "message_not_found"},
            "message_not_found",
        )

        with pytest.raises(ThreadNotFoundError):
            await thread_manager.get_thread("invalid_ts", "C12345")

    @pytest.mark.asyncio
    async def test_get_thread_empty_messages(self, thread_manager, mock_connector):
        """Should raise ThreadNotFoundError when messages array is empty."""
        from aragora.connectors.chat.thread_manager import ThreadNotFoundError

        mock_connector._slack_api_request.return_value = (
            True,
            {"ok": True, "messages": []},
            None,
        )

        with pytest.raises(ThreadNotFoundError):
            await thread_manager.get_thread("1704067200.000001", "C12345")

    @pytest.mark.asyncio
    async def test_get_thread_api_error(self, thread_manager, mock_connector):
        """Should raise Exception on generic API error."""
        mock_connector._slack_api_request.return_value = (
            False,
            None,
            "internal_error",
        )

        with pytest.raises(Exception) as exc_info:
            await thread_manager.get_thread("1704067200.000001", "C12345")

        assert "internal_error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_thread_httpx_not_available(self, thread_manager):
        """Should raise RuntimeError when httpx not available."""
        with patch("aragora.connectors.chat.slack.threads.HTTPX_AVAILABLE", False):
            with pytest.raises(RuntimeError) as exc_info:
                await thread_manager.get_thread("1704067200.000001", "C12345")

        assert "httpx" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Get Thread Messages Tests
# ---------------------------------------------------------------------------


class TestGetThreadMessages:
    """Tests for get_thread_messages method."""

    @pytest.mark.asyncio
    async def test_get_thread_messages_success(
        self, thread_manager, mock_connector, mock_messages_response
    ):
        """Should get all messages in a thread."""
        mock_connector._slack_api_request.return_value = (
            True,
            mock_messages_response(),
            None,
        )

        messages, cursor = await thread_manager.get_thread_messages(
            "1704067200.000001", "C12345", limit=50
        )

        assert len(messages) == 3
        assert messages[0].content == "Root message"
        assert messages[1].content == "Reply 1"
        assert cursor is None

    @pytest.mark.asyncio
    async def test_get_thread_messages_sets_thread_id(
        self, thread_manager, mock_connector, mock_messages_response
    ):
        """Should set thread_id on all returned messages."""
        mock_connector._slack_api_request.return_value = (
            True,
            mock_messages_response(),
            None,
        )

        messages, _ = await thread_manager.get_thread_messages("1704067200.000001", "C12345")

        for msg in messages:
            assert msg.thread_id == "1704067200.000001"

    @pytest.mark.asyncio
    async def test_get_thread_messages_with_pagination(
        self, thread_manager, mock_connector, mock_messages_response
    ):
        """Should return next cursor for pagination."""
        mock_connector._slack_api_request.return_value = (
            True,
            mock_messages_response(next_cursor="dGVzdF9jdXJzb3I="),
            None,
        )

        messages, cursor = await thread_manager.get_thread_messages(
            "1704067200.000001", "C12345", limit=50
        )

        assert len(messages) == 3
        assert cursor == "dGVzdF9jdXJzb3I="

    @pytest.mark.asyncio
    async def test_get_thread_messages_with_cursor_param(
        self, thread_manager, mock_connector, mock_messages_response
    ):
        """Should pass cursor parameter to API."""
        mock_connector._slack_api_request.return_value = (
            True,
            mock_messages_response(messages=[{"ts": "1.0", "user": "U1", "text": "Next page"}]),
            None,
        )

        await thread_manager.get_thread_messages(
            "1704067200.000001", "C12345", cursor="existing_cursor"
        )

        call_kwargs = mock_connector._slack_api_request.call_args[1]
        assert call_kwargs["params"]["cursor"] == "existing_cursor"

    @pytest.mark.asyncio
    async def test_get_thread_messages_limits_to_1000(
        self, thread_manager, mock_connector, mock_messages_response
    ):
        """Should cap limit at 1000."""
        mock_connector._slack_api_request.return_value = (
            True,
            mock_messages_response(),
            None,
        )

        await thread_manager.get_thread_messages("1704067200.000001", "C12345", limit=5000)

        call_kwargs = mock_connector._slack_api_request.call_args[1]
        assert call_kwargs["params"]["limit"] == 1000

    @pytest.mark.asyncio
    async def test_get_thread_messages_handles_bot_messages(self, thread_manager, mock_connector):
        """Should handle bot messages (bot_id instead of user)."""
        mock_connector._slack_api_request.return_value = (
            True,
            {
                "ok": True,
                "messages": [
                    {"ts": "1.0", "bot_id": "B12345", "text": "Bot message"},
                ],
                "response_metadata": {"next_cursor": ""},
            },
            None,
        )

        messages, _ = await thread_manager.get_thread_messages("1.0", "C12345")

        assert len(messages) == 1
        assert messages[0].author.id == "B12345"
        assert messages[0].author.is_bot is True

    @pytest.mark.asyncio
    async def test_get_thread_messages_api_failure(self, thread_manager, mock_connector):
        """Should return empty list on API failure."""
        mock_connector._slack_api_request.return_value = (
            False,
            None,
            "channel_not_found",
        )

        messages, cursor = await thread_manager.get_thread_messages("1704067200.000001", "C12345")

        assert messages == []
        assert cursor is None

    @pytest.mark.asyncio
    async def test_get_thread_messages_httpx_not_available(self, thread_manager):
        """Should return empty list when httpx not available."""
        with patch("aragora.connectors.chat.slack.threads.HTTPX_AVAILABLE", False):
            messages, cursor = await thread_manager.get_thread_messages(
                "1704067200.000001", "C12345"
            )

        assert messages == []
        assert cursor is None


# ---------------------------------------------------------------------------
# List Threads Tests
# ---------------------------------------------------------------------------


class TestListThreads:
    """Tests for list_threads method."""

    @pytest.mark.asyncio
    async def test_list_threads_success(self, thread_manager, mock_connector):
        """Should list threads in a channel."""
        mock_connector._slack_api_request.return_value = (
            True,
            {
                "ok": True,
                "messages": [
                    {
                        "ts": "1704067200.000001",
                        "user": "U12345",
                        "text": "Thread 1 root",
                        "reply_count": 3,
                        "reply_users": ["U12345"],
                        "latest_reply": "1704153600.000002",
                    },
                    {
                        "ts": "1704067100.000001",
                        "user": "U67890",
                        "text": "Not a thread",
                        "reply_count": 0,
                    },
                    {
                        "ts": "1704067000.000001",
                        "user": "U12345",
                        "text": "Thread 2 root",
                        "reply_count": 1,
                        "reply_users": ["U67890"],
                    },
                ],
                "response_metadata": {"next_cursor": ""},
            },
            None,
        )

        threads, cursor = await thread_manager.list_threads("C12345", limit=10)

        # Only messages with reply_count > 0 should be returned
        assert len(threads) == 2
        assert threads[0].message_count == 4  # 3 replies + root

    @pytest.mark.asyncio
    async def test_list_threads_excludes_non_threaded_messages(
        self, thread_manager, mock_connector
    ):
        """Should exclude messages with no replies."""
        mock_connector._slack_api_request.return_value = (
            True,
            {
                "ok": True,
                "messages": [
                    {"ts": "1.0", "user": "U1", "text": "No replies", "reply_count": 0},
                ],
                "response_metadata": {"next_cursor": ""},
            },
            None,
        )

        threads, _ = await thread_manager.list_threads("C12345")

        assert threads == []

    @pytest.mark.asyncio
    async def test_list_threads_respects_limit(self, thread_manager, mock_connector):
        """Should return at most 'limit' threads."""
        messages = [
            {
                "ts": f"{i}.0",
                "user": "U1",
                "text": f"Thread {i}",
                "reply_count": 1,
                "reply_users": ["U2"],
            }
            for i in range(50)
        ]
        mock_connector._slack_api_request.return_value = (
            True,
            {"ok": True, "messages": messages, "response_metadata": {"next_cursor": ""}},
            None,
        )

        threads, _ = await thread_manager.list_threads("C12345", limit=5)

        assert len(threads) == 5

    @pytest.mark.asyncio
    async def test_list_threads_with_pagination(self, thread_manager, mock_connector):
        """Should return pagination cursor."""
        mock_connector._slack_api_request.return_value = (
            True,
            {
                "ok": True,
                "messages": [
                    {
                        "ts": "1.0",
                        "user": "U1",
                        "text": "Thread",
                        "reply_count": 1,
                        "reply_users": [],
                    },
                ],
                "response_metadata": {"next_cursor": "next_page_cursor"},
            },
            None,
        )

        threads, cursor = await thread_manager.list_threads("C12345")

        assert cursor == "next_page_cursor"

    @pytest.mark.asyncio
    async def test_list_threads_api_failure(self, thread_manager, mock_connector):
        """Should return empty list on API failure."""
        mock_connector._slack_api_request.return_value = (
            False,
            None,
            "channel_not_found",
        )

        threads, cursor = await thread_manager.list_threads("C12345")

        assert threads == []
        assert cursor is None

    @pytest.mark.asyncio
    async def test_list_threads_httpx_not_available(self, thread_manager):
        """Should return empty list when httpx not available."""
        with patch("aragora.connectors.chat.slack.threads.HTTPX_AVAILABLE", False):
            threads, cursor = await thread_manager.list_threads("C12345")

        assert threads == []
        assert cursor is None


# ---------------------------------------------------------------------------
# Reply to Thread Tests
# ---------------------------------------------------------------------------


class TestReplyToThread:
    """Tests for reply_to_thread method."""

    @pytest.mark.asyncio
    async def test_reply_to_thread_success(self, thread_manager, mock_connector):
        """Should reply to an existing thread."""
        mock_connector.send_message.return_value = MagicMock(
            message_id="1704067400.000001",
            author_id="B12345",
        )

        message = await thread_manager.reply_to_thread(
            "1704067200.000001",
            "C12345",
            "This is a reply",
        )

        mock_connector.send_message.assert_called_once()
        call_kwargs = mock_connector.send_message.call_args[1]
        assert call_kwargs["thread_id"] == "1704067200.000001"
        assert call_kwargs["text"] == "This is a reply"
        assert message.id == "1704067400.000001"

    @pytest.mark.asyncio
    async def test_reply_to_thread_with_blocks(self, thread_manager, mock_connector):
        """Should pass blocks to send_message."""
        mock_connector.send_message.return_value = MagicMock(
            message_id="1.0",
            author_id="B1",
        )

        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "Bold"}}]
        await thread_manager.reply_to_thread(
            "1704067200.000001",
            "C12345",
            "Reply text",
            blocks=blocks,
        )

        call_kwargs = mock_connector.send_message.call_args[1]
        assert call_kwargs["blocks"] == blocks

    @pytest.mark.asyncio
    async def test_reply_to_thread_returns_chat_message(self, thread_manager, mock_connector):
        """Should return ChatMessage with correct attributes."""
        mock_connector.send_message.return_value = MagicMock(
            message_id="1704067400.000001",
            author_id="B12345",
        )

        message = await thread_manager.reply_to_thread(
            "1704067200.000001",
            "C12345",
            "Reply content",
        )

        assert message.platform == "slack"
        assert message.channel.id == "C12345"
        assert message.thread_id == "1704067200.000001"
        assert message.content == "Reply content"


# ---------------------------------------------------------------------------
# Broadcast Thread Reply Tests
# ---------------------------------------------------------------------------


class TestBroadcastThreadReply:
    """Tests for broadcast_thread_reply method."""

    @pytest.mark.asyncio
    async def test_broadcast_reply_success(self, thread_manager, mock_connector):
        """Should reply to thread with channel broadcast."""
        mock_connector._slack_api_request.return_value = (
            True,
            {
                "ok": True,
                "ts": "1704067400.000001",
                "message": {"user": "B12345", "text": "Broadcast reply"},
            },
            None,
        )

        message = await thread_manager.broadcast_thread_reply(
            "1704067200.000001",
            "C12345",
            "Important update!",
        )

        assert message.id == "1704067400.000001"
        assert message.thread_id == "1704067200.000001"

    @pytest.mark.asyncio
    async def test_broadcast_reply_includes_broadcast_flag(self, thread_manager, mock_connector):
        """Should include reply_broadcast: true in API payload."""
        mock_connector._slack_api_request.return_value = (
            True,
            {"ok": True, "ts": "1.0", "message": {}},
            None,
        )

        await thread_manager.broadcast_thread_reply(
            "1704067200.000001",
            "C12345",
            "Broadcast message",
        )

        call_kwargs = mock_connector._slack_api_request.call_args[1]
        assert call_kwargs["json_data"]["reply_broadcast"] is True

    @pytest.mark.asyncio
    async def test_broadcast_reply_with_blocks(self, thread_manager, mock_connector):
        """Should include blocks in broadcast payload."""
        mock_connector._slack_api_request.return_value = (
            True,
            {"ok": True, "ts": "1.0", "message": {}},
            None,
        )

        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "Important"}}]
        await thread_manager.broadcast_thread_reply(
            "1704067200.000001",
            "C12345",
            "Broadcast",
            blocks=blocks,
        )

        call_kwargs = mock_connector._slack_api_request.call_args[1]
        assert call_kwargs["json_data"]["blocks"] == blocks

    @pytest.mark.asyncio
    async def test_broadcast_reply_api_failure(self, thread_manager, mock_connector):
        """Should raise Exception on API failure."""
        mock_connector._slack_api_request.return_value = (
            False,
            None,
            "channel_not_found",
        )

        with pytest.raises(Exception) as exc_info:
            await thread_manager.broadcast_thread_reply(
                "1704067200.000001",
                "C12345",
                "Broadcast",
            )

        assert "channel_not_found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_broadcast_reply_httpx_not_available(self, thread_manager):
        """Should raise RuntimeError when httpx not available."""
        with patch("aragora.connectors.chat.slack.threads.HTTPX_AVAILABLE", False):
            with pytest.raises(RuntimeError):
                await thread_manager.broadcast_thread_reply(
                    "1704067200.000001",
                    "C12345",
                    "Broadcast",
                )


# ---------------------------------------------------------------------------
# Get Thread Stats Tests
# ---------------------------------------------------------------------------


class TestGetThreadStats:
    """Tests for get_thread_stats method."""

    @pytest.mark.asyncio
    async def test_get_thread_stats_success(self, thread_manager, mock_connector):
        """Should calculate thread statistics."""
        # First call for messages, then subsequent calls return empty for pagination
        mock_connector._slack_api_request.return_value = (
            True,
            {
                "ok": True,
                "messages": [
                    {"ts": "1.0", "user": "U1", "text": "Root"},
                    {"ts": "2.0", "user": "U2", "text": "Reply 1"},
                    {"ts": "3.0", "user": "U1", "text": "Reply 2"},
                ],
                "response_metadata": {"next_cursor": ""},
            },
            None,
        )

        stats = await thread_manager.get_thread_stats("1.0", "C12345")

        assert stats.thread_id == "1.0"
        assert stats.message_count == 3
        assert stats.participant_count == 2  # U1 and U2

    @pytest.mark.asyncio
    async def test_get_thread_stats_empty_thread(self, thread_manager, mock_connector):
        """Should handle empty thread gracefully."""
        mock_connector._slack_api_request.return_value = (
            True,
            {"ok": True, "messages": [], "response_metadata": {"next_cursor": ""}},
            None,
        )

        stats = await thread_manager.get_thread_stats("1.0", "C12345")

        assert stats.message_count == 0
        assert stats.participant_count == 0


# ---------------------------------------------------------------------------
# Get Thread Participants Tests
# ---------------------------------------------------------------------------


class TestGetThreadParticipants:
    """Tests for get_thread_participants method."""

    @pytest.mark.asyncio
    async def test_get_participants_success(self, thread_manager, mock_connector):
        """Should return list of thread participants."""
        mock_connector._slack_api_request.return_value = (
            True,
            {
                "ok": True,
                "messages": [
                    {"ts": "1.0", "user": "U1", "text": "First"},
                    {"ts": "2.0", "user": "U2", "text": "Second"},
                    {"ts": "3.0", "user": "U1", "text": "Third"},
                ],
                "response_metadata": {"next_cursor": ""},
            },
            None,
        )

        participants = await thread_manager.get_thread_participants("1.0", "C12345")

        assert len(participants) == 2
        # U1 has 2 messages
        u1 = next(p for p in participants if p.user_id == "U1")
        assert u1.message_count == 2

    @pytest.mark.asyncio
    async def test_get_participants_tracks_timestamps(self, thread_manager, mock_connector):
        """Should track first and last message timestamps."""
        mock_connector._slack_api_request.return_value = (
            True,
            {
                "ok": True,
                "messages": [
                    {"ts": "1000.0", "user": "U1", "text": "First"},
                    {"ts": "2000.0", "user": "U1", "text": "Second"},
                ],
                "response_metadata": {"next_cursor": ""},
            },
            None,
        )

        participants = await thread_manager.get_thread_participants("1000.0", "C12345")

        u1 = participants[0]
        assert u1.first_message_at < u1.last_message_at

    @pytest.mark.asyncio
    async def test_get_participants_identifies_bots(self, thread_manager, mock_connector):
        """Should identify bot participants."""
        mock_connector._slack_api_request.return_value = (
            True,
            {
                "ok": True,
                "messages": [
                    {"ts": "1.0", "user": "U1", "text": "Human"},
                    {"ts": "2.0", "bot_id": "B1", "text": "Bot"},
                ],
                "response_metadata": {"next_cursor": ""},
            },
            None,
        )

        participants = await thread_manager.get_thread_participants("1.0", "C12345")

        human = next(p for p in participants if p.user_id == "U1")
        bot = next(p for p in participants if p.user_id == "B1")
        assert human.is_bot is False
        assert bot.is_bot is True
