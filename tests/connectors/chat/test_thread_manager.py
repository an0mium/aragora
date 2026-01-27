"""
Tests for ThreadManager implementations.

Tests cover:
- ThreadManager base class interface
- SlackThreadManager operations
- TeamsThreadManager operations
- Thread context generation for AI prompts
- Error handling and edge cases
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.connectors.chat.thread_manager import (
    ThreadManager,
    ThreadInfo,
    ThreadStats,
    ThreadParticipant,
    ThreadNotFoundError,
)
from aragora.connectors.chat.models import (
    ChatChannel,
    ChatMessage,
    ChatUser,
    MessageType,
)


class TestThreadInfo:
    """Tests for ThreadInfo dataclass."""

    def test_create_thread_info(self):
        """Should create ThreadInfo with required fields."""
        now = datetime.utcnow()
        thread = ThreadInfo(
            id="thread_123",
            channel_id="C12345",
            platform="slack",
            created_by="U12345",
            created_at=now,
            updated_at=now,
        )

        assert thread.id == "thread_123"
        assert thread.channel_id == "C12345"
        assert thread.platform == "slack"
        assert thread.message_count == 0
        assert thread.participant_count == 0
        assert thread.is_archived is False

    def test_thread_info_to_dict(self):
        """Should serialize ThreadInfo to dictionary."""
        now = datetime.utcnow()
        thread = ThreadInfo(
            id="thread_123",
            channel_id="C12345",
            platform="slack",
            created_by="U12345",
            created_at=now,
            updated_at=now,
            message_count=10,
            participant_count=3,
            title="Test thread",
        )

        data = thread.to_dict()

        assert data["id"] == "thread_123"
        assert data["channel_id"] == "C12345"
        assert data["message_count"] == 10
        assert data["title"] == "Test thread"
        assert "created_at" in data


class TestThreadStats:
    """Tests for ThreadStats dataclass."""

    def test_create_thread_stats(self):
        """Should create ThreadStats with required fields."""
        now = datetime.utcnow()
        stats = ThreadStats(
            thread_id="thread_123",
            message_count=25,
            participant_count=5,
            last_activity=now,
        )

        assert stats.thread_id == "thread_123"
        assert stats.message_count == 25
        assert stats.participant_count == 5
        assert stats.avg_response_time_seconds is None

    def test_thread_stats_with_metrics(self):
        """Should handle response time metrics."""
        now = datetime.utcnow()
        stats = ThreadStats(
            thread_id="thread_123",
            message_count=25,
            participant_count=5,
            last_activity=now,
            avg_response_time_seconds=120.5,
            first_response_time_seconds=30.0,
            participants=["U1", "U2", "U3"],
            total_reactions=15,
        )

        data = stats.to_dict()
        assert data["avg_response_time_seconds"] == 120.5
        assert data["first_response_time_seconds"] == 30.0
        assert data["total_reactions"] == 15


class TestThreadNotFoundError:
    """Tests for ThreadNotFoundError exception."""

    def test_error_message(self):
        """Should format error message correctly."""
        error = ThreadNotFoundError("thread_123", "C12345", "slack")

        assert "thread_123" in str(error)
        assert "C12345" in str(error)
        assert "slack" in str(error)

    def test_error_attributes(self):
        """Should store error attributes."""
        error = ThreadNotFoundError("thread_123", "C12345", "teams")

        assert error.thread_id == "thread_123"
        assert error.channel_id == "C12345"
        assert error.platform == "teams"


class TestSlackThreadManager:
    """Tests for SlackThreadManager."""

    @pytest.fixture
    def mock_connector(self):
        """Create mock SlackConnector."""
        connector = MagicMock()
        connector.bot_token = "xoxb-test-token"
        connector._request_timeout = 30.0
        return connector

    @pytest.fixture
    def thread_manager(self, mock_connector):
        """Create SlackThreadManager with mock connector."""
        from aragora.connectors.chat.slack import SlackThreadManager

        return SlackThreadManager(mock_connector)

    def test_platform_name(self, thread_manager):
        """Should return 'slack' as platform name."""
        assert thread_manager.platform_name == "slack"

    @pytest.mark.asyncio
    async def test_get_thread_success(self, thread_manager):
        """Should get thread info from conversations.replies."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ok": True,
            "messages": [
                {
                    "ts": "1704067200.000001",
                    "user": "U12345",
                    "text": "Thread root message",
                    "reply_count": 5,
                    "reply_users": ["U12345", "U67890"],
                    "latest_reply": "1704153600.000002",
                }
            ],
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            thread = await thread_manager.get_thread("1704067200.000001", "C12345")

        assert thread.id == "1704067200.000001"
        assert thread.channel_id == "C12345"
        assert thread.platform == "slack"
        assert thread.message_count == 6  # 5 replies + root
        assert thread.participant_count == 3  # 2 reply_users + OP

    @pytest.mark.asyncio
    async def test_get_thread_not_found(self, thread_manager):
        """Should raise ThreadNotFoundError when thread doesn't exist."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ok": False,
            "error": "thread_not_found",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(ThreadNotFoundError) as exc_info:
                await thread_manager.get_thread("invalid_ts", "C12345")

        assert exc_info.value.thread_id == "invalid_ts"
        assert exc_info.value.platform == "slack"

    @pytest.mark.asyncio
    async def test_get_thread_messages(self, thread_manager):
        """Should get all messages in a thread."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ok": True,
            "messages": [
                {
                    "ts": "1704067200.000001",
                    "user": "U12345",
                    "text": "Root message",
                },
                {
                    "ts": "1704067260.000002",
                    "user": "U67890",
                    "text": "Reply 1",
                },
                {
                    "ts": "1704067320.000003",
                    "user": "U12345",
                    "text": "Reply 2",
                },
            ],
            "response_metadata": {"next_cursor": ""},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            messages, cursor = await thread_manager.get_thread_messages(
                "1704067200.000001", "C12345", limit=50
            )

        assert len(messages) == 3
        assert messages[0].content == "Root message"
        assert messages[1].content == "Reply 1"
        assert cursor is None

    @pytest.mark.asyncio
    async def test_get_thread_messages_with_pagination(self, thread_manager):
        """Should handle pagination in thread messages."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ok": True,
            "messages": [{"ts": "123", "user": "U1", "text": "Msg"}],
            "response_metadata": {"next_cursor": "dGVzdF9jdXJzb3I="},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            messages, cursor = await thread_manager.get_thread_messages("123", "C12345", limit=50)

        assert len(messages) == 1
        assert cursor == "dGVzdF9jdXJzb3I="

    @pytest.mark.asyncio
    async def test_list_threads(self, thread_manager):
        """Should list threads in a channel."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
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
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            threads, cursor = await thread_manager.list_threads("C12345", limit=10)

        # Only messages with reply_count > 0 should be returned
        assert len(threads) == 2
        assert threads[0].message_count == 4  # 3 replies + root

    @pytest.mark.asyncio
    async def test_reply_to_thread(self, thread_manager, mock_connector):
        """Should reply to an existing thread."""
        mock_connector.send_message = AsyncMock(
            return_value=MagicMock(
                message_id="1704067400.000001",
                author_id="B12345",
            )
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

    @pytest.mark.asyncio
    async def test_broadcast_thread_reply(self, thread_manager):
        """Should reply to thread with channel broadcast."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ok": True,
            "ts": "1704067400.000001",
            "message": {"user": "B12345", "text": "Broadcast reply"},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            message = await thread_manager.broadcast_thread_reply(
                "1704067200.000001",
                "C12345",
                "Important update!",
            )

        assert message.id == "1704067400.000001"
        assert message.thread_id == "1704067200.000001"


class TestTeamsThreadManager:
    """Tests for TeamsThreadManager."""

    @pytest.fixture
    def mock_connector(self):
        """Create mock TeamsConnector."""
        connector = MagicMock()
        connector._graph_api_request = AsyncMock()
        return connector

    @pytest.fixture
    def thread_manager(self, mock_connector):
        """Create TeamsThreadManager with mock connector."""
        from aragora.connectors.chat.teams import TeamsThreadManager

        return TeamsThreadManager(mock_connector, team_id="test-team-id")

    def test_platform_name(self, thread_manager):
        """Should return 'teams' as platform name."""
        assert thread_manager.platform_name == "teams"

    def test_team_id_required_at_init(self, mock_connector):
        """Should require team_id at initialization."""
        from aragora.connectors.chat.teams import TeamsThreadManager

        with pytest.raises(TypeError):
            TeamsThreadManager(mock_connector)  # Missing team_id

    @pytest.mark.asyncio
    async def test_get_thread_success(self, thread_manager, mock_connector):
        """Should get thread info via Graph API."""
        mock_connector._graph_api_request.side_effect = [
            # First call: get message
            (
                True,
                {
                    "id": "msg_123",
                    "from": {"user": {"id": "user_456"}},
                    "body": {"content": "Thread root", "contentType": "text"},
                    "createdDateTime": "2024-01-01T00:00:00Z",
                },
                None,
            ),
            # Second call: get replies
            (
                True,
                {
                    "value": [
                        {
                            "id": "reply_1",
                            "from": {"user": {"id": "user_789"}},
                            "createdDateTime": "2024-01-01T01:00:00Z",
                        }
                    ]
                },
                None,
            ),
        ]

        thread = await thread_manager.get_thread("msg_123", "channel_456")

        assert thread.id == "msg_123"
        assert thread.channel_id == "channel_456"
        assert thread.platform == "teams"
        assert thread.message_count == 2  # root + 1 reply

    @pytest.mark.asyncio
    async def test_get_thread_not_found(self, thread_manager, mock_connector):
        """Should raise ThreadNotFoundError when message doesn't exist."""
        mock_connector._graph_api_request.return_value = (False, None, "Not found")

        with pytest.raises(ThreadNotFoundError) as exc_info:
            await thread_manager.get_thread("invalid_msg", "channel_456")

        assert exc_info.value.thread_id == "invalid_msg"
        assert exc_info.value.platform == "teams"

    @pytest.mark.asyncio
    async def test_reply_to_thread(self, thread_manager, mock_connector):
        """Should reply to a Teams thread."""
        mock_connector._graph_api_request.return_value = (
            True,
            {
                "id": "reply_123",
                "from": {"user": {"id": "user_456", "displayName": "Bot"}},
                "body": {"content": "Reply text", "contentType": "text"},
                "createdDateTime": "2024-01-01T02:00:00Z",
            },
            None,
        )

        message = await thread_manager.reply_to_thread(
            "msg_123",
            "channel_456",
            "This is a reply",
        )

        assert message.id == "reply_123"
        assert message.thread_id == "msg_123"

        # Verify API was called correctly
        mock_connector._graph_api_request.assert_called_once()
        call_kwargs = mock_connector._graph_api_request.call_args[1]
        assert "replies" in call_kwargs["endpoint"]
        assert call_kwargs["method"] == "POST"


class TestThreadManagerContext:
    """Tests for thread context generation."""

    @pytest.mark.asyncio
    async def test_get_thread_context(self):
        """Should generate context dict for AI prompts."""
        # Create a concrete implementation for testing
        from aragora.connectors.chat.slack import SlackThreadManager

        mock_connector = MagicMock()
        mock_connector.bot_token = "xoxb-test"
        mock_connector._request_timeout = 30.0

        thread_manager = SlackThreadManager(mock_connector)

        # Mock the methods called by get_thread_context
        now = datetime.utcnow()
        mock_thread = ThreadInfo(
            id="thread_123",
            channel_id="C12345",
            platform="slack",
            created_by="U12345",
            created_at=now,
            updated_at=now,
            message_count=5,
            participant_count=3,
            title="Test discussion",
        )

        mock_messages = [
            ChatMessage(
                id="msg1",
                platform="slack",
                channel=ChatChannel(id="C12345", platform="slack"),
                author=ChatUser(id="U12345", platform="slack", display_name="Alice"),
                content="First message",
                timestamp=now - timedelta(hours=2),
            ),
            ChatMessage(
                id="msg2",
                platform="slack",
                channel=ChatChannel(id="C12345", platform="slack"),
                author=ChatUser(id="U67890", platform="slack", display_name="Bob"),
                content="Reply to first",
                timestamp=now - timedelta(hours=1),
            ),
        ]

        mock_participants = [
            ThreadParticipant(
                user_id="U12345",
                display_name="Alice",
                message_count=3,
            ),
            ThreadParticipant(
                user_id="U67890",
                display_name="Bob",
                message_count=2,
            ),
        ]

        with (
            patch.object(thread_manager, "get_thread", return_value=mock_thread),
            patch.object(
                thread_manager,
                "get_thread_messages",
                return_value=(mock_messages, None),
            ),
            patch.object(
                thread_manager,
                "get_thread_participants",
                return_value=mock_participants,
            ),
        ):
            # Use the base class method
            from aragora.connectors.chat.thread_manager import ThreadManager

            context = await ThreadManager.get_thread_context(thread_manager, "thread_123", "C12345")

        assert context["thread_id"] == "thread_123"
        assert context["channel_id"] == "C12345"
        assert context["platform"] == "slack"
        assert context["message_count"] == 5
        assert len(context["participants"]) == 2
        assert len(context["recent_messages"]) == 2
        assert context["recent_messages"][0]["author"] == "Alice"
