"""
Tests for TeamsThreadManager - Microsoft Teams thread management.

Tests cover:
- Thread retrieval and metadata
- Thread message pagination
- Thread listing
- Thread replies
- Thread statistics
- Thread participants
- Error handling (thread not found, API errors)
- Date/time parsing
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.connectors.chat.models import ChatChannel, ChatMessage, ChatUser
from aragora.connectors.chat.thread_manager import ThreadInfo, ThreadStats, ThreadNotFoundError


class MockTeamsConnector:
    """Mock TeamsConnector for testing thread manager."""

    def __init__(self):
        self._graph_api_request_mock = AsyncMock()

    async def _graph_api_request(
        self,
        endpoint,
        method="GET",
        operation=None,
        json_data=None,
        params=None,
        use_full_url=False,
    ):
        return await self._graph_api_request_mock(
            endpoint=endpoint,
            method=method,
            operation=operation,
            json_data=json_data,
            params=params,
            use_full_url=use_full_url,
        )


class TestTeamsThreadManagerInit:
    """Tests for TeamsThreadManager initialization."""

    def test_init_with_connector_and_team_id(self):
        """Should initialize with connector and team ID."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        assert manager.connector == connector
        assert manager.team_id == "team-123"

    def test_platform_name(self):
        """Should return 'teams' as platform name."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        assert manager.platform_name == "teams"


class TestGetThread:
    """Tests for get_thread method."""

    @pytest.mark.asyncio
    async def test_get_thread_success(self):
        """Should retrieve thread metadata successfully."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        # Mock thread data
        thread_data = {
            "id": "thread-456",
            "from": {"user": {"id": "user-789"}},
            "createdDateTime": "2024-01-15T10:30:00Z",
            "lastModifiedDateTime": "2024-01-15T11:00:00Z",
            "subject": "Thread Title",
            "importance": "normal",
            "messageType": "message",
        }

        # Mock replies
        replies_data = {
            "value": [
                {"from": {"user": {"id": "user-111"}}},
                {"from": {"user": {"id": "user-222"}}},
            ]
        }

        connector._graph_api_request_mock.side_effect = [
            (True, thread_data, None),
            (True, replies_data, None),
        ]

        thread = await manager.get_thread(
            thread_id="thread-456",
            channel_id="channel-abc",
        )

        assert thread.id == "thread-456"
        assert thread.channel_id == "channel-abc"
        assert thread.platform == "teams"
        assert thread.created_by == "user-789"
        assert thread.title == "Thread Title"
        assert thread.message_count == 3  # Root + 2 replies
        assert thread.participant_count == 3  # 3 unique users

    @pytest.mark.asyncio
    async def test_get_thread_not_found(self):
        """Should raise ThreadNotFoundError when thread doesn't exist."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        connector._graph_api_request_mock.return_value = (False, None, "Not found")

        with pytest.raises(ThreadNotFoundError) as exc_info:
            await manager.get_thread(
                thread_id="nonexistent",
                channel_id="channel-abc",
            )

        assert exc_info.value.thread_id == "nonexistent"
        assert exc_info.value.channel_id == "channel-abc"
        assert exc_info.value.platform == "teams"

    @pytest.mark.asyncio
    async def test_get_thread_with_empty_replies(self):
        """Should handle threads with no replies."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        thread_data = {
            "id": "thread-456",
            "from": {"user": {"id": "user-789"}},
            "createdDateTime": "2024-01-15T10:30:00Z",
            "lastModifiedDateTime": "2024-01-15T10:30:00Z",
        }

        connector._graph_api_request_mock.side_effect = [
            (True, thread_data, None),
            (True, {"value": []}, None),
        ]

        thread = await manager.get_thread(
            thread_id="thread-456",
            channel_id="channel-abc",
        )

        assert thread.message_count == 1  # Just root message
        assert thread.participant_count == 1

    @pytest.mark.asyncio
    async def test_get_thread_missing_user_info(self):
        """Should handle missing user information gracefully."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        thread_data = {
            "id": "thread-456",
            "from": {},  # Missing user
            "createdDateTime": "2024-01-15T10:30:00Z",
        }

        connector._graph_api_request_mock.side_effect = [
            (True, thread_data, None),
            (True, {"value": []}, None),
        ]

        thread = await manager.get_thread(
            thread_id="thread-456",
            channel_id="channel-abc",
        )

        assert thread.created_by == "unknown"

    @pytest.mark.asyncio
    async def test_get_thread_metadata_includes_team_id(self):
        """Should include team_id in metadata."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-999")

        thread_data = {
            "id": "thread-456",
            "from": {"user": {"id": "user-789"}},
            "createdDateTime": "2024-01-15T10:30:00Z",
            "importance": "high",
            "messageType": "reply",
        }

        connector._graph_api_request_mock.side_effect = [
            (True, thread_data, None),
            (True, {"value": []}, None),
        ]

        thread = await manager.get_thread(
            thread_id="thread-456",
            channel_id="channel-abc",
        )

        assert thread.metadata["team_id"] == "team-999"
        assert thread.metadata["importance"] == "high"
        assert thread.metadata["message_type"] == "reply"


class TestGetThreadMessages:
    """Tests for get_thread_messages method."""

    @pytest.mark.asyncio
    async def test_get_thread_messages_success(self):
        """Should retrieve thread messages successfully."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        messages_data = {
            "value": [
                {
                    "id": "msg-1",
                    "from": {"user": {"id": "user-111", "displayName": "User One"}},
                    "body": {"content": "First reply", "contentType": "text"},
                    "createdDateTime": "2024-01-15T10:30:00Z",
                    "importance": "normal",
                },
                {
                    "id": "msg-2",
                    "from": {"user": {"id": "user-222", "displayName": "User Two"}},
                    "body": {"content": "Second reply", "contentType": "html"},
                    "createdDateTime": "2024-01-15T10:31:00Z",
                    "importance": "high",
                },
            ],
            "@odata.nextLink": "https://graph.microsoft.com/next?page=2",
        }

        connector._graph_api_request_mock.return_value = (True, messages_data, None)

        messages, cursor = await manager.get_thread_messages(
            thread_id="thread-456",
            channel_id="channel-abc",
            limit=50,
        )

        assert len(messages) == 2
        assert messages[0].id == "msg-1"
        assert messages[0].content == "First reply"
        assert messages[0].author.id == "user-111"
        assert messages[0].author.display_name == "User One"
        assert messages[0].thread_id == "thread-456"
        assert cursor == "https://graph.microsoft.com/next?page=2"

    @pytest.mark.asyncio
    async def test_get_thread_messages_with_cursor(self):
        """Should use cursor for pagination."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        connector._graph_api_request_mock.return_value = (True, {"value": []}, None)

        await manager.get_thread_messages(
            thread_id="thread-456",
            channel_id="channel-abc",
            cursor="https://graph.microsoft.com/next?page=2",
        )

        call_args = connector._graph_api_request_mock.call_args
        assert call_args.kwargs["endpoint"] == "https://graph.microsoft.com/next?page=2"
        assert call_args.kwargs["use_full_url"] is True

    @pytest.mark.asyncio
    async def test_get_thread_messages_empty(self):
        """Should return empty list when no messages."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        connector._graph_api_request_mock.return_value = (True, {"value": []}, None)

        messages, cursor = await manager.get_thread_messages(
            thread_id="thread-456",
            channel_id="channel-abc",
        )

        assert messages == []
        assert cursor is None

    @pytest.mark.asyncio
    async def test_get_thread_messages_api_failure(self):
        """Should return empty list on API failure."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        connector._graph_api_request_mock.return_value = (False, None, "API Error")

        messages, cursor = await manager.get_thread_messages(
            thread_id="thread-456",
            channel_id="channel-abc",
        )

        assert messages == []
        assert cursor is None

    @pytest.mark.asyncio
    async def test_get_thread_messages_limit_parameter(self):
        """Should pass limit as $top parameter."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        connector._graph_api_request_mock.return_value = (True, {"value": []}, None)

        await manager.get_thread_messages(
            thread_id="thread-456",
            channel_id="channel-abc",
            limit=25,
        )

        call_args = connector._graph_api_request_mock.call_args
        assert call_args.kwargs["params"]["$top"] == "25"

    @pytest.mark.asyncio
    async def test_get_thread_messages_missing_timestamp(self):
        """Should handle missing timestamp gracefully."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        messages_data = {
            "value": [
                {
                    "id": "msg-1",
                    "from": {"user": {"id": "user-111"}},
                    "body": {"content": "No timestamp"},
                    # createdDateTime missing
                }
            ]
        }

        connector._graph_api_request_mock.return_value = (True, messages_data, None)

        messages, _ = await manager.get_thread_messages(
            thread_id="thread-456",
            channel_id="channel-abc",
        )

        assert len(messages) == 1
        # Should use current time as fallback
        assert messages[0].timestamp is not None


class TestListThreads:
    """Tests for list_threads method."""

    @pytest.mark.asyncio
    async def test_list_threads_success(self):
        """Should list threads in channel."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        threads_data = {
            "value": [
                {
                    "id": "thread-1",
                    "from": {"user": {"id": "user-111"}},
                    "createdDateTime": "2024-01-15T10:00:00Z",
                    "lastModifiedDateTime": "2024-01-15T12:00:00Z",
                    "subject": "First Thread",
                    "importance": "normal",
                },
                {
                    "id": "thread-2",
                    "from": {"user": {"id": "user-222"}},
                    "createdDateTime": "2024-01-14T09:00:00Z",
                    "lastModifiedDateTime": "2024-01-14T11:00:00Z",
                    "subject": "Second Thread",
                    "importance": "high",
                },
            ]
        }

        connector._graph_api_request_mock.return_value = (True, threads_data, None)

        threads = await manager.list_threads(
            channel_id="channel-abc",
            limit=20,
        )

        assert len(threads) == 2
        assert threads[0].id == "thread-1"
        assert threads[0].title == "First Thread"
        assert threads[0].created_by == "user-111"
        assert threads[1].id == "thread-2"
        assert threads[1].title == "Second Thread"

    @pytest.mark.asyncio
    async def test_list_threads_empty(self):
        """Should return empty list when no threads."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        connector._graph_api_request_mock.return_value = (True, {"value": []}, None)

        threads = await manager.list_threads(channel_id="channel-abc")

        assert threads == []

    @pytest.mark.asyncio
    async def test_list_threads_api_failure(self):
        """Should return empty list on API failure."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        connector._graph_api_request_mock.return_value = (False, None, "API Error")

        threads = await manager.list_threads(channel_id="channel-abc")

        assert threads == []

    @pytest.mark.asyncio
    async def test_list_threads_limit_parameter(self):
        """Should pass limit as $top parameter."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        connector._graph_api_request_mock.return_value = (True, {"value": []}, None)

        await manager.list_threads(channel_id="channel-abc", limit=10)

        call_args = connector._graph_api_request_mock.call_args
        assert call_args.kwargs["params"]["$top"] == "10"


class TestReplyToThread:
    """Tests for reply_to_thread method."""

    @pytest.mark.asyncio
    async def test_reply_to_thread_success(self):
        """Should reply to thread successfully."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        reply_data = {
            "id": "reply-789",
            "from": {"user": {"id": "bot-user", "displayName": "Bot"}},
            "body": {"content": "Reply content"},
            "createdDateTime": "2024-01-15T10:30:00Z",
        }

        connector._graph_api_request_mock.return_value = (True, reply_data, None)

        message = await manager.reply_to_thread(
            thread_id="thread-456",
            channel_id="channel-abc",
            message="Reply content",
        )

        assert message.id == "reply-789"
        assert message.content == "Reply content"
        assert message.thread_id == "thread-456"
        assert message.platform == "teams"

    @pytest.mark.asyncio
    async def test_reply_to_thread_uses_post(self):
        """Should use POST method for replies."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        reply_data = {"id": "reply-789", "from": {}, "body": {"content": "Reply"}}
        connector._graph_api_request_mock.return_value = (True, reply_data, None)

        await manager.reply_to_thread(
            thread_id="thread-456",
            channel_id="channel-abc",
            message="Test reply",
        )

        call_args = connector._graph_api_request_mock.call_args
        assert call_args.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_reply_to_thread_sends_body_content(self):
        """Should send message body with correct format."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        reply_data = {"id": "reply-789", "from": {}, "body": {"content": "Reply"}}
        connector._graph_api_request_mock.return_value = (True, reply_data, None)

        await manager.reply_to_thread(
            thread_id="thread-456",
            channel_id="channel-abc",
            message="Test reply message",
        )

        call_args = connector._graph_api_request_mock.call_args
        json_data = call_args.kwargs["json_data"]
        assert json_data["body"]["content"] == "Test reply message"
        assert json_data["body"]["contentType"] == "text"

    @pytest.mark.asyncio
    async def test_reply_to_thread_failure(self):
        """Should raise RuntimeError on failure."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        connector._graph_api_request_mock.return_value = (False, None, "API Error")

        with pytest.raises(RuntimeError) as exc_info:
            await manager.reply_to_thread(
                thread_id="thread-456",
                channel_id="channel-abc",
                message="Will fail",
            )

        assert "Failed to reply to thread" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_reply_to_thread_missing_user_info(self):
        """Should handle missing user info with defaults."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        reply_data = {"id": "reply-789", "from": {}, "body": {"content": "Reply"}}
        connector._graph_api_request_mock.return_value = (True, reply_data, None)

        message = await manager.reply_to_thread(
            thread_id="thread-456",
            channel_id="channel-abc",
            message="Test reply",
        )

        assert message.author.id == "bot"
        assert message.author.display_name == "Bot"


class TestGetThreadStats:
    """Tests for get_thread_stats method."""

    @pytest.mark.asyncio
    async def test_get_thread_stats_success(self):
        """Should return thread statistics."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        thread_data = {
            "id": "thread-456",
            "from": {"user": {"id": "user-789"}},
            "createdDateTime": "2024-01-15T10:00:00Z",
            "lastModifiedDateTime": "2024-01-15T12:00:00Z",
        }

        replies_data = {
            "value": [
                {"from": {"user": {"id": "user-111"}}},
                {"from": {"user": {"id": "user-222"}}},
            ]
        }

        connector._graph_api_request_mock.side_effect = [
            (True, thread_data, None),
            (True, replies_data, None),
        ]

        stats = await manager.get_thread_stats(
            thread_id="thread-456",
            channel_id="channel-abc",
        )

        assert stats.thread_id == "thread-456"
        assert stats.message_count == 3  # Root + 2 replies
        assert stats.participant_count == 3

    @pytest.mark.asyncio
    async def test_get_thread_stats_includes_last_activity(self):
        """Should include last activity timestamp."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        thread_data = {
            "id": "thread-456",
            "from": {"user": {"id": "user-789"}},
            "createdDateTime": "2024-01-15T10:00:00Z",
            "lastModifiedDateTime": "2024-01-15T14:30:00Z",
        }

        connector._graph_api_request_mock.side_effect = [
            (True, thread_data, None),
            (True, {"value": []}, None),
        ]

        stats = await manager.get_thread_stats(
            thread_id="thread-456",
            channel_id="channel-abc",
        )

        assert stats.last_activity is not None


class TestGetThreadParticipants:
    """Tests for get_thread_participants method."""

    @pytest.mark.asyncio
    async def test_get_thread_participants_success(self):
        """Should return list of participant IDs."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        messages_data = {
            "value": [
                {
                    "id": "msg-1",
                    "from": {"user": {"id": "user-111"}},
                    "body": {"content": "A"},
                    "createdDateTime": "2024-01-15T10:00:00Z",
                },
                {
                    "id": "msg-2",
                    "from": {"user": {"id": "user-222"}},
                    "body": {"content": "B"},
                    "createdDateTime": "2024-01-15T10:01:00Z",
                },
                {
                    "id": "msg-3",
                    "from": {"user": {"id": "user-111"}},
                    "body": {"content": "C"},
                    "createdDateTime": "2024-01-15T10:02:00Z",
                },  # Duplicate user
            ]
        }

        connector._graph_api_request_mock.return_value = (True, messages_data, None)

        participants = await manager.get_thread_participants(
            thread_id="thread-456",
            channel_id="channel-abc",
        )

        assert len(participants) == 2  # Unique users
        assert "user-111" in participants
        assert "user-222" in participants

    @pytest.mark.asyncio
    async def test_get_thread_participants_empty(self):
        """Should return empty list when no messages."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        connector._graph_api_request_mock.return_value = (True, {"value": []}, None)

        participants = await manager.get_thread_participants(
            thread_id="thread-456",
            channel_id="channel-abc",
        )

        assert participants == []


class TestEndpointConstruction:
    """Tests for Graph API endpoint construction."""

    @pytest.mark.asyncio
    async def test_get_thread_endpoint(self):
        """Should construct correct endpoint for get_thread."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-ABC")

        connector._graph_api_request_mock.side_effect = [
            (True, {"id": "t", "from": {}, "createdDateTime": "2024-01-15T10:00:00Z"}, None),
            (True, {"value": []}, None),
        ]

        await manager.get_thread(
            thread_id="thread-XYZ",
            channel_id="channel-123",
        )

        call_args = connector._graph_api_request_mock.call_args_list[0]
        assert (
            "/teams/team-ABC/channels/channel-123/messages/thread-XYZ"
            in call_args.kwargs["endpoint"]
        )

    @pytest.mark.asyncio
    async def test_reply_to_thread_endpoint(self):
        """Should construct correct endpoint for reply."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-ABC")

        reply_data = {"id": "r", "from": {}, "body": {"content": "X"}}
        connector._graph_api_request_mock.return_value = (True, reply_data, None)

        await manager.reply_to_thread(
            thread_id="thread-XYZ",
            channel_id="channel-123",
            message="Reply",
        )

        call_args = connector._graph_api_request_mock.call_args
        expected = "/teams/team-ABC/channels/channel-123/messages/thread-XYZ/replies"
        assert expected in call_args.kwargs["endpoint"]

    @pytest.mark.asyncio
    async def test_list_threads_endpoint(self):
        """Should construct correct endpoint for list_threads."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-ABC")

        connector._graph_api_request_mock.return_value = (True, {"value": []}, None)

        await manager.list_threads(channel_id="channel-123")

        call_args = connector._graph_api_request_mock.call_args
        assert "/teams/team-ABC/channels/channel-123/messages" in call_args.kwargs["endpoint"]


class TestDateTimeParsing:
    """Tests for date/time parsing from Graph API responses."""

    @pytest.mark.asyncio
    async def test_parses_iso8601_with_z_suffix(self):
        """Should parse ISO 8601 timestamps with Z suffix."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        thread_data = {
            "id": "thread-456",
            "from": {"user": {"id": "user-789"}},
            "createdDateTime": "2024-01-15T10:30:00Z",
            "lastModifiedDateTime": "2024-01-15T11:45:00Z",
        }

        connector._graph_api_request_mock.side_effect = [
            (True, thread_data, None),
            (True, {"value": []}, None),
        ]

        thread = await manager.get_thread(
            thread_id="thread-456",
            channel_id="channel-abc",
        )

        assert thread.created_at.year == 2024
        assert thread.created_at.month == 1
        assert thread.created_at.day == 15
        assert thread.created_at.hour == 10
        assert thread.created_at.minute == 30

    @pytest.mark.asyncio
    async def test_handles_missing_created_datetime(self):
        """Should use current time when createdDateTime is missing."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        thread_data = {
            "id": "thread-456",
            "from": {"user": {"id": "user-789"}},
            # createdDateTime missing
        }

        connector._graph_api_request_mock.side_effect = [
            (True, thread_data, None),
            (True, {"value": []}, None),
        ]

        thread = await manager.get_thread(
            thread_id="thread-456",
            channel_id="channel-abc",
        )

        # Should default to current time
        assert thread.created_at is not None
        assert thread.updated_at is not None


class TestChatMessageConstruction:
    """Tests for ChatMessage construction from Graph API data."""

    @pytest.mark.asyncio
    async def test_constructs_chat_message_correctly(self):
        """Should construct ChatMessage with all fields."""
        from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

        connector = MockTeamsConnector()
        manager = TeamsThreadManager(connector, team_id="team-123")

        messages_data = {
            "value": [
                {
                    "id": "msg-unique-123",
                    "from": {"user": {"id": "user-456", "displayName": "Test User"}},
                    "body": {"content": "Message content here", "contentType": "html"},
                    "createdDateTime": "2024-01-15T10:30:00Z",
                    "importance": "high",
                }
            ]
        }

        connector._graph_api_request_mock.return_value = (True, messages_data, None)

        messages, _ = await manager.get_thread_messages(
            thread_id="thread-parent",
            channel_id="channel-abc",
        )

        msg = messages[0]
        assert msg.id == "msg-unique-123"
        assert msg.platform == "teams"
        assert msg.channel.id == "channel-abc"
        assert msg.channel.platform == "teams"
        assert msg.author.id == "user-456"
        assert msg.author.display_name == "Test User"
        assert msg.content == "Message content here"
        assert msg.thread_id == "thread-parent"
        assert msg.metadata["importance"] == "high"
        assert msg.metadata["content_type"] == "html"
