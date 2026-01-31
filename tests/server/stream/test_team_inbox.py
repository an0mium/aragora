"""
Tests for team_inbox.py - WebSocket events for team inbox collaboration.

Tests cover:
- TeamInboxEventType enum values
- TeamMember, Mention, InternalNote dataclasses
- TeamInboxEvent dataclass (serialization, timestamp generation)
- TeamInboxEmitter (subscription management, event emission)
- Team member management (add, get, remove)
- Assignment and status events
- Presence events (viewing, typing)
- Mentions (creation, extraction, acknowledgment)
- Internal notes (creation, retrieval)
- Activity feed events
- Callbacks and error handling
- Concurrent operations and race conditions
- Convenience functions
"""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.server.stream.team_inbox import (
    InternalNote,
    Mention,
    TeamInboxEmitter,
    TeamInboxEvent,
    TeamInboxEventType,
    TeamMember,
    emit_message_assigned,
    emit_message_unassigned,
    emit_status_changed,
    emit_user_typing,
    emit_user_viewing,
    get_team_inbox_emitter,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def emitter():
    """Create a fresh TeamInboxEmitter for each test."""
    return TeamInboxEmitter()


@pytest.fixture
def mock_ws():
    """Create a mock WebSocket connection."""
    ws = MagicMock()
    ws.send = AsyncMock()
    return ws


@pytest.fixture
def mock_ws_closed():
    """Create a mock WebSocket that raises on send (simulating closed connection)."""
    ws = MagicMock()
    ws.send = AsyncMock(side_effect=ConnectionError("Connection closed"))
    return ws


@pytest.fixture
def multiple_mock_ws():
    """Create multiple mock WebSocket connections."""
    return [
        MagicMock(send=AsyncMock()),
        MagicMock(send=AsyncMock()),
        MagicMock(send=AsyncMock()),
    ]


@pytest.fixture(autouse=True)
def reset_global_emitter():
    """Reset the global emitter before each test."""
    import aragora.server.stream.team_inbox as team_inbox_module

    team_inbox_module._team_inbox_emitter = None
    yield
    team_inbox_module._team_inbox_emitter = None


# ===========================================================================
# Test TeamInboxEventType
# ===========================================================================


class TestTeamInboxEventType:
    """Tests for TeamInboxEventType enum."""

    def test_message_assigned_value(self):
        """MESSAGE_ASSIGNED has correct string value."""
        assert TeamInboxEventType.MESSAGE_ASSIGNED.value == "team_inbox_message_assigned"

    def test_message_unassigned_value(self):
        """MESSAGE_UNASSIGNED has correct string value."""
        assert TeamInboxEventType.MESSAGE_UNASSIGNED.value == "team_inbox_message_unassigned"

    def test_message_reassigned_value(self):
        """MESSAGE_REASSIGNED has correct string value."""
        assert TeamInboxEventType.MESSAGE_REASSIGNED.value == "team_inbox_message_reassigned"

    def test_status_changed_value(self):
        """STATUS_CHANGED has correct string value."""
        assert TeamInboxEventType.STATUS_CHANGED.value == "team_inbox_status_changed"

    def test_user_viewing_value(self):
        """USER_VIEWING has correct string value."""
        assert TeamInboxEventType.USER_VIEWING.value == "team_inbox_user_viewing"

    def test_user_typing_value(self):
        """USER_TYPING has correct string value."""
        assert TeamInboxEventType.USER_TYPING.value == "team_inbox_user_typing"

    def test_mention_value(self):
        """MENTION has correct string value."""
        assert TeamInboxEventType.MENTION.value == "team_inbox_mention"

    def test_note_added_value(self):
        """NOTE_ADDED has correct string value."""
        assert TeamInboxEventType.NOTE_ADDED.value == "team_inbox_note_added"

    def test_activity_value(self):
        """ACTIVITY has correct string value."""
        assert TeamInboxEventType.ACTIVITY.value == "team_inbox_activity"

    def test_event_type_is_string_enum(self):
        """TeamInboxEventType is a string enum."""
        assert isinstance(TeamInboxEventType.MESSAGE_ASSIGNED, str)
        assert TeamInboxEventType.MESSAGE_ASSIGNED == "team_inbox_message_assigned"


# ===========================================================================
# Test TeamMember
# ===========================================================================


class TestTeamMember:
    """Tests for TeamMember dataclass."""

    def test_creation_basic(self):
        """TeamMember can be created with basic attributes."""
        member = TeamMember(
            user_id="user-123",
            email="test@example.com",
            name="Test User",
        )
        assert member.user_id == "user-123"
        assert member.email == "test@example.com"
        assert member.name == "Test User"
        assert member.role == "member"

    def test_creation_with_role(self):
        """TeamMember accepts custom role."""
        member = TeamMember(
            user_id="user-123",
            email="admin@example.com",
            name="Admin User",
            role="admin",
        )
        assert member.role == "admin"

    def test_creation_with_avatar(self):
        """TeamMember accepts avatar URL."""
        member = TeamMember(
            user_id="user-123",
            email="test@example.com",
            name="Test User",
            avatar_url="https://example.com/avatar.png",
        )
        assert member.avatar_url == "https://example.com/avatar.png"

    def test_to_dict(self):
        """to_dict returns proper dictionary representation."""
        member = TeamMember(
            user_id="user-123",
            email="test@example.com",
            name="Test User",
            role="admin",
        )
        result = member.to_dict()
        assert result["userId"] == "user-123"
        assert result["email"] == "test@example.com"
        assert result["name"] == "Test User"
        assert result["role"] == "admin"
        assert "joinedAt" in result
        assert result["avatarUrl"] is None

    def test_joined_at_auto_generated(self):
        """joined_at is auto-generated with current time."""
        before = datetime.now(timezone.utc)
        member = TeamMember(
            user_id="user-123",
            email="test@example.com",
            name="Test User",
        )
        after = datetime.now(timezone.utc)
        assert before <= member.joined_at <= after


# ===========================================================================
# Test Mention
# ===========================================================================


class TestMention:
    """Tests for Mention dataclass."""

    def test_creation_basic(self):
        """Mention can be created with required attributes."""
        mention = Mention(
            id="mention-123",
            mentioned_user_id="user-456",
            mentioned_by_user_id="user-789",
            message_id="msg-001",
            inbox_id="inbox-001",
            context="Hey @user-456, check this out",
        )
        assert mention.id == "mention-123"
        assert mention.mentioned_user_id == "user-456"
        assert mention.mentioned_by_user_id == "user-789"
        assert mention.acknowledged is False
        assert mention.acknowledged_at is None

    def test_to_dict(self):
        """to_dict returns proper dictionary representation."""
        mention = Mention(
            id="mention-123",
            mentioned_user_id="user-456",
            mentioned_by_user_id="user-789",
            message_id="msg-001",
            inbox_id="inbox-001",
            context="Check this",
        )
        result = mention.to_dict()
        assert result["id"] == "mention-123"
        assert result["mentionedUserId"] == "user-456"
        assert result["mentionedByUserId"] == "user-789"
        assert result["messageId"] == "msg-001"
        assert result["inboxId"] == "inbox-001"
        assert result["context"] == "Check this"
        assert result["acknowledged"] is False
        assert result["acknowledgedAt"] is None

    def test_to_dict_with_acknowledged(self):
        """to_dict includes acknowledged timestamp when set."""
        ack_time = datetime.now(timezone.utc)
        mention = Mention(
            id="mention-123",
            mentioned_user_id="user-456",
            mentioned_by_user_id="user-789",
            message_id="msg-001",
            inbox_id="inbox-001",
            context="Check this",
            acknowledged=True,
            acknowledged_at=ack_time,
        )
        result = mention.to_dict()
        assert result["acknowledged"] is True
        assert result["acknowledgedAt"] == ack_time.isoformat()


# ===========================================================================
# Test InternalNote
# ===========================================================================


class TestInternalNote:
    """Tests for InternalNote dataclass."""

    def test_creation_basic(self):
        """InternalNote can be created with required attributes."""
        note = InternalNote(
            id="note-123",
            message_id="msg-001",
            inbox_id="inbox-001",
            author_id="user-456",
            author_name="Test Author",
            content="This is a private note",
        )
        assert note.id == "note-123"
        assert note.content == "This is a private note"
        assert note.mentions == []
        assert note.is_pinned is False

    def test_creation_with_mentions(self):
        """InternalNote accepts mentions list."""
        note = InternalNote(
            id="note-123",
            message_id="msg-001",
            inbox_id="inbox-001",
            author_id="user-456",
            author_name="Test Author",
            content="Hey @john, please review",
            mentions=["john", "jane"],
        )
        assert note.mentions == ["john", "jane"]

    def test_to_dict(self):
        """to_dict returns proper dictionary representation."""
        note = InternalNote(
            id="note-123",
            message_id="msg-001",
            inbox_id="inbox-001",
            author_id="user-456",
            author_name="Test Author",
            content="Test note content",
            is_pinned=True,
        )
        result = note.to_dict()
        assert result["id"] == "note-123"
        assert result["messageId"] == "msg-001"
        assert result["inboxId"] == "inbox-001"
        assert result["authorId"] == "user-456"
        assert result["authorName"] == "Test Author"
        assert result["content"] == "Test note content"
        assert result["isPinned"] is True
        assert "createdAt" in result


# ===========================================================================
# Test TeamInboxEvent
# ===========================================================================


class TestTeamInboxEvent:
    """Tests for TeamInboxEvent dataclass."""

    def test_event_creation_basic(self):
        """Event can be created with basic attributes."""
        event = TeamInboxEvent(
            type=TeamInboxEventType.MESSAGE_ASSIGNED,
            inbox_id="inbox-123",
            data={"assignedTo": "user-456"},
        )
        assert event.type == TeamInboxEventType.MESSAGE_ASSIGNED
        assert event.inbox_id == "inbox-123"
        assert event.data["assignedTo"] == "user-456"

    def test_event_auto_generates_timestamp(self):
        """Event auto-generates timestamp if not provided."""
        event = TeamInboxEvent(
            type=TeamInboxEventType.MESSAGE_ASSIGNED,
            inbox_id="inbox-123",
            data={},
        )
        assert event.timestamp is not None
        assert len(event.timestamp) > 0
        # Verify it's a valid ISO timestamp
        datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))

    def test_event_preserves_provided_timestamp(self):
        """Event preserves explicitly provided timestamp."""
        timestamp = "2024-01-15T10:30:00+00:00"
        event = TeamInboxEvent(
            type=TeamInboxEventType.MESSAGE_ASSIGNED,
            inbox_id="inbox-123",
            data={},
            timestamp=timestamp,
        )
        assert event.timestamp == timestamp

    def test_event_to_dict(self):
        """to_dict returns proper dictionary representation."""
        event = TeamInboxEvent(
            type=TeamInboxEventType.STATUS_CHANGED,
            inbox_id="inbox-456",
            user_id="user-789",
            message_id="msg-001",
            data={"oldStatus": "open", "newStatus": "closed"},
            timestamp="2024-01-15T10:30:00+00:00",
        )
        result = event.to_dict()
        assert result["type"] == "team_inbox_status_changed"
        assert result["inboxId"] == "inbox-456"
        assert result["userId"] == "user-789"
        assert result["messageId"] == "msg-001"
        assert result["data"]["oldStatus"] == "open"
        assert result["timestamp"] == "2024-01-15T10:30:00+00:00"

    def test_event_to_json(self):
        """to_json returns valid JSON string."""
        event = TeamInboxEvent(
            type=TeamInboxEventType.NOTE_ADDED,
            inbox_id="inbox-123",
            data={"content": "Test note"},
            timestamp="2024-01-15T10:30:00+00:00",
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["type"] == "team_inbox_note_added"
        assert parsed["inboxId"] == "inbox-123"
        assert parsed["data"]["content"] == "Test note"

    def test_event_with_string_type_in_to_dict(self):
        """to_dict handles string type values correctly."""
        event = TeamInboxEvent(
            type="custom_type",  # type: ignore
            inbox_id="inbox-123",
            data={},
        )
        result = event.to_dict()
        assert result["type"] == "custom_type"


# ===========================================================================
# Test TeamInboxEmitter - Subscription Management
# ===========================================================================


class TestTeamInboxEmitterSubscriptions:
    """Tests for TeamInboxEmitter subscription management."""

    @pytest.mark.asyncio
    async def test_subscribe_new_inbox(self, emitter, mock_ws):
        """Subscribe adds websocket to inbox's subscription set."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)
        assert "inbox-123" in emitter._inbox_subscriptions
        assert mock_ws in emitter._inbox_subscriptions["inbox-123"]

    @pytest.mark.asyncio
    async def test_subscribe_multiple_connections_same_inbox(self, emitter, multiple_mock_ws):
        """Multiple connections can subscribe to same inbox."""
        for ws in multiple_mock_ws:
            await emitter.subscribe_to_inbox("inbox-123", ws)
        assert len(emitter._inbox_subscriptions["inbox-123"]) == 3

    @pytest.mark.asyncio
    async def test_subscribe_different_inboxes(self, emitter, multiple_mock_ws):
        """Different inboxes have separate subscription sets."""
        await emitter.subscribe_to_inbox("inbox-1", multiple_mock_ws[0])
        await emitter.subscribe_to_inbox("inbox-2", multiple_mock_ws[1])
        await emitter.subscribe_to_inbox("inbox-3", multiple_mock_ws[2])

        assert len(emitter._inbox_subscriptions) == 3
        assert multiple_mock_ws[0] in emitter._inbox_subscriptions["inbox-1"]
        assert multiple_mock_ws[1] in emitter._inbox_subscriptions["inbox-2"]

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_connection(self, emitter, mock_ws):
        """Unsubscribe removes the websocket from subscriptions."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)
        await emitter.unsubscribe_from_inbox("inbox-123", mock_ws)
        assert "inbox-123" not in emitter._inbox_subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe_preserves_other_connections(self, emitter, multiple_mock_ws):
        """Unsubscribe only removes specific connection."""
        for ws in multiple_mock_ws:
            await emitter.subscribe_to_inbox("inbox-123", ws)

        await emitter.unsubscribe_from_inbox("inbox-123", multiple_mock_ws[0])

        assert len(emitter._inbox_subscriptions["inbox-123"]) == 2
        assert multiple_mock_ws[0] not in emitter._inbox_subscriptions["inbox-123"]
        assert multiple_mock_ws[1] in emitter._inbox_subscriptions["inbox-123"]

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_inbox(self, emitter, mock_ws):
        """Unsubscribe handles nonexistent inbox gracefully."""
        await emitter.unsubscribe_from_inbox("nonexistent", mock_ws)
        # Should not raise

    @pytest.mark.asyncio
    async def test_unsubscribe_last_connection_removes_inbox_entry(self, emitter, mock_ws):
        """Removing last connection cleans up inbox entry."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)
        await emitter.unsubscribe_from_inbox("inbox-123", mock_ws)
        assert "inbox-123" not in emitter._inbox_subscriptions


# ===========================================================================
# Test TeamInboxEmitter - Event Emission
# ===========================================================================


class TestTeamInboxEmitterEmission:
    """Tests for TeamInboxEmitter event emission."""

    @pytest.mark.asyncio
    async def test_emit_sends_to_subscribed_client(self, emitter, mock_ws):
        """Emit sends event to subscribed client."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)
        event = TeamInboxEvent(
            type=TeamInboxEventType.MESSAGE_ASSIGNED,
            inbox_id="inbox-123",
            data={"assignedTo": "user-456"},
        )

        sent_count = await emitter.emit(event)

        assert sent_count == 1
        mock_ws.send.assert_called_once()
        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["type"] == "team_inbox_message_assigned"

    @pytest.mark.asyncio
    async def test_emit_sends_to_all_inbox_connections(self, emitter, multiple_mock_ws):
        """Emit sends event to all inbox's connections."""
        for ws in multiple_mock_ws:
            await emitter.subscribe_to_inbox("inbox-123", ws)

        event = TeamInboxEvent(
            type=TeamInboxEventType.STATUS_CHANGED,
            inbox_id="inbox-123",
            data={"newStatus": "closed"},
        )

        sent_count = await emitter.emit(event)

        assert sent_count == 3
        for ws in multiple_mock_ws:
            ws.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_returns_zero_for_no_subscribers(self, emitter):
        """Emit returns 0 when no subscribers for inbox."""
        event = TeamInboxEvent(
            type=TeamInboxEventType.MESSAGE_ASSIGNED,
            inbox_id="nonexistent-inbox",
            data={},
        )
        sent_count = await emitter.emit(event)
        assert sent_count == 0

    @pytest.mark.asyncio
    async def test_emit_cleans_up_dead_connections(self, emitter, mock_ws, mock_ws_closed):
        """Emit removes dead connections after failed send."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)
        await emitter.subscribe_to_inbox("inbox-123", mock_ws_closed)

        event = TeamInboxEvent(
            type=TeamInboxEventType.MESSAGE_ASSIGNED,
            inbox_id="inbox-123",
            data={},
        )

        sent_count = await emitter.emit(event)

        assert sent_count == 1
        assert mock_ws_closed not in emitter._inbox_subscriptions["inbox-123"]
        assert mock_ws in emitter._inbox_subscriptions["inbox-123"]


# ===========================================================================
# Test TeamInboxEmitter - Team Member Management
# ===========================================================================


class TestTeamInboxEmitterTeamMembers:
    """Tests for TeamInboxEmitter team member management."""

    @pytest.mark.asyncio
    async def test_add_team_member(self, emitter):
        """add_team_member creates and stores a member."""
        member = await emitter.add_team_member(
            inbox_id="inbox-123",
            user_id="user-456",
            email="test@example.com",
            name="Test User",
        )

        assert member.user_id == "user-456"
        assert member.email == "test@example.com"
        assert member.role == "member"

    @pytest.mark.asyncio
    async def test_add_team_member_with_role(self, emitter):
        """add_team_member accepts custom role."""
        member = await emitter.add_team_member(
            inbox_id="inbox-123",
            user_id="user-456",
            email="admin@example.com",
            name="Admin User",
            role="admin",
        )

        assert member.role == "admin"

    @pytest.mark.asyncio
    async def test_get_team_members(self, emitter):
        """get_team_members returns all members for inbox."""
        await emitter.add_team_member("inbox-123", "user-1", "u1@test.com", "User 1")
        await emitter.add_team_member("inbox-123", "user-2", "u2@test.com", "User 2")
        await emitter.add_team_member("inbox-456", "user-3", "u3@test.com", "User 3")

        members = await emitter.get_team_members("inbox-123")

        assert len(members) == 2
        user_ids = [m.user_id for m in members]
        assert "user-1" in user_ids
        assert "user-2" in user_ids

    @pytest.mark.asyncio
    async def test_get_team_members_empty_inbox(self, emitter):
        """get_team_members returns empty list for unknown inbox."""
        members = await emitter.get_team_members("nonexistent")
        assert members == []

    @pytest.mark.asyncio
    async def test_remove_team_member(self, emitter):
        """remove_team_member removes a member from inbox."""
        await emitter.add_team_member("inbox-123", "user-456", "test@test.com", "Test")

        result = await emitter.remove_team_member("inbox-123", "user-456")

        assert result is True
        members = await emitter.get_team_members("inbox-123")
        assert len(members) == 0

    @pytest.mark.asyncio
    async def test_remove_team_member_nonexistent(self, emitter):
        """remove_team_member returns False for unknown member."""
        result = await emitter.remove_team_member("inbox-123", "nonexistent")
        assert result is False


# ===========================================================================
# Test TeamInboxEmitter - Assignment Events
# ===========================================================================


class TestTeamInboxEmitterAssignmentEvents:
    """Tests for TeamInboxEmitter assignment events."""

    @pytest.mark.asyncio
    async def test_emit_message_assigned(self, emitter, mock_ws):
        """emit_message_assigned sends correct event."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        await emitter.emit_message_assigned(
            inbox_id="inbox-123",
            message_id="msg-001",
            assigned_to_user_id="user-456",
            assigned_by_user_id="user-789",
            assigned_to_name="John Doe",
            assigned_by_name="Jane Smith",
        )

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["type"] == "team_inbox_message_assigned"
        assert sent_message["messageId"] == "msg-001"
        assert sent_message["data"]["assignedToUserId"] == "user-456"
        assert sent_message["data"]["assignedToName"] == "John Doe"
        assert sent_message["data"]["assignedByName"] == "Jane Smith"

    @pytest.mark.asyncio
    async def test_emit_message_unassigned(self, emitter, mock_ws):
        """emit_message_unassigned sends correct event."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        await emitter.emit_message_unassigned(
            inbox_id="inbox-123",
            message_id="msg-001",
            unassigned_by_user_id="user-789",
            unassigned_by_name="Admin User",
            previous_assignee_id="user-456",
        )

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["type"] == "team_inbox_message_unassigned"
        assert sent_message["data"]["unassignedByUserId"] == "user-789"
        assert sent_message["data"]["previousAssigneeId"] == "user-456"


# ===========================================================================
# Test TeamInboxEmitter - Status Events
# ===========================================================================


class TestTeamInboxEmitterStatusEvents:
    """Tests for TeamInboxEmitter status events."""

    @pytest.mark.asyncio
    async def test_emit_status_changed(self, emitter, mock_ws):
        """emit_status_changed sends correct event."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        await emitter.emit_status_changed(
            inbox_id="inbox-123",
            message_id="msg-001",
            old_status="open",
            new_status="resolved",
            changed_by_user_id="user-456",
            changed_by_name="Test User",
        )

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["type"] == "team_inbox_status_changed"
        assert sent_message["data"]["oldStatus"] == "open"
        assert sent_message["data"]["newStatus"] == "resolved"
        assert sent_message["data"]["changedByUserId"] == "user-456"


# ===========================================================================
# Test TeamInboxEmitter - Presence Events
# ===========================================================================


class TestTeamInboxEmitterPresenceEvents:
    """Tests for TeamInboxEmitter presence events."""

    @pytest.mark.asyncio
    async def test_emit_user_viewing(self, emitter, mock_ws):
        """emit_user_viewing sends event and tracks viewer."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        await emitter.emit_user_viewing(
            inbox_id="inbox-123",
            message_id="msg-001",
            user_id="user-456",
            user_name="Test User",
        )

        # Check event was sent
        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["type"] == "team_inbox_user_viewing"
        assert sent_message["data"]["userId"] == "user-456"
        assert "user-456" in sent_message["data"]["viewers"]

        # Check viewer is tracked
        viewers = await emitter.get_message_viewers("msg-001")
        assert "user-456" in viewers

    @pytest.mark.asyncio
    async def test_emit_user_left(self, emitter, mock_ws):
        """emit_user_left removes viewer and typer tracking."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        # First add viewer and typer
        await emitter.emit_user_viewing("inbox-123", "msg-001", "user-456", "Test User")
        await emitter.emit_user_typing("inbox-123", "msg-001", "user-456", "Test User")

        # Now emit user left
        await emitter.emit_user_left("inbox-123", "msg-001", "user-456", "Test User")

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["type"] == "team_inbox_user_left"
        assert "user-456" not in sent_message["data"]["viewers"]

        # Check viewer is removed
        viewers = await emitter.get_message_viewers("msg-001")
        assert "user-456" not in viewers

    @pytest.mark.asyncio
    async def test_emit_user_typing(self, emitter, mock_ws):
        """emit_user_typing tracks typing users."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        await emitter.emit_user_typing(
            inbox_id="inbox-123",
            message_id="msg-001",
            user_id="user-456",
            user_name="Test User",
        )

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["type"] == "team_inbox_user_typing"
        assert "user-456" in sent_message["data"]["typers"]

    @pytest.mark.asyncio
    async def test_emit_user_stopped_typing(self, emitter, mock_ws):
        """emit_user_stopped_typing removes typing tracking."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        # First add typer
        await emitter.emit_user_typing("inbox-123", "msg-001", "user-456", "Test User")

        # Now emit stopped typing
        await emitter.emit_user_stopped_typing("inbox-123", "msg-001", "user-456", "Test User")

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["type"] == "team_inbox_user_stopped_typing"
        assert "user-456" not in sent_message["data"]["typers"]

    @pytest.mark.asyncio
    async def test_get_message_viewers_empty(self, emitter):
        """get_message_viewers returns empty list for unknown message."""
        viewers = await emitter.get_message_viewers("nonexistent")
        assert viewers == []


# ===========================================================================
# Test TeamInboxEmitter - Mentions
# ===========================================================================


class TestTeamInboxEmitterMentions:
    """Tests for TeamInboxEmitter mention functionality."""

    def test_extract_mentions_simple(self, emitter):
        """extract_mentions finds simple @mentions."""
        text = "Hey @john, can you help?"
        mentions = emitter.extract_mentions(text)
        assert mentions == ["john"]

    def test_extract_mentions_multiple(self, emitter):
        """extract_mentions finds multiple @mentions."""
        text = "@alice and @bob should review this"
        mentions = emitter.extract_mentions(text)
        assert set(mentions) == {"alice", "bob"}

    def test_extract_mentions_with_dots(self, emitter):
        """extract_mentions handles mentions with dots."""
        text = "CC @john.smith for approval"
        mentions = emitter.extract_mentions(text)
        assert "john.smith" in mentions

    def test_extract_mentions_with_email_format(self, emitter):
        """extract_mentions handles email-style mentions."""
        text = "Ping @user@example.com please"
        mentions = emitter.extract_mentions(text)
        assert any("user" in m for m in mentions)

    def test_extract_mentions_deduplicates(self, emitter):
        """extract_mentions returns unique mentions."""
        text = "@john @john @john please help"
        mentions = emitter.extract_mentions(text)
        assert mentions == ["john"]

    def test_extract_mentions_empty_text(self, emitter):
        """extract_mentions returns empty list for text without mentions."""
        text = "No mentions here"
        mentions = emitter.extract_mentions(text)
        assert mentions == []

    @pytest.mark.asyncio
    async def test_create_mention(self, emitter, mock_ws):
        """create_mention creates and stores mention, emits event."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        mention = await emitter.create_mention(
            mentioned_user_id="user-456",
            mentioned_by_user_id="user-789",
            message_id="msg-001",
            inbox_id="inbox-123",
            context="Hey @user-456, check this",
        )

        assert mention.mentioned_user_id == "user-456"
        assert mention.mentioned_by_user_id == "user-789"
        assert mention.acknowledged is False

        # Check event was sent
        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["type"] == "team_inbox_mention"

    @pytest.mark.asyncio
    async def test_get_mentions_for_user(self, emitter):
        """get_mentions_for_user retrieves user's mentions."""
        await emitter.create_mention("user-456", "user-1", "msg-1", "inbox-1", "ctx1")
        await emitter.create_mention("user-456", "user-2", "msg-2", "inbox-1", "ctx2")
        await emitter.create_mention("user-789", "user-3", "msg-3", "inbox-1", "ctx3")

        mentions = await emitter.get_mentions_for_user("user-456")

        assert len(mentions) == 2
        assert all(m.mentioned_user_id == "user-456" for m in mentions)

    @pytest.mark.asyncio
    async def test_get_mentions_unacknowledged_only(self, emitter):
        """get_mentions_for_user can filter to unacknowledged only."""
        mention1 = await emitter.create_mention("user-456", "user-1", "msg-1", "inbox-1", "ctx")
        await emitter.create_mention("user-456", "user-2", "msg-2", "inbox-1", "ctx")

        # Acknowledge first mention
        await emitter.acknowledge_mention("user-456", mention1.id)

        mentions = await emitter.get_mentions_for_user("user-456", unacknowledged_only=True)

        assert len(mentions) == 1
        assert mentions[0].acknowledged is False

    @pytest.mark.asyncio
    async def test_acknowledge_mention(self, emitter):
        """acknowledge_mention marks mention as acknowledged."""
        mention = await emitter.create_mention("user-456", "user-789", "msg-1", "inbox-1", "ctx")

        result = await emitter.acknowledge_mention("user-456", mention.id)

        assert result is True
        mentions = await emitter.get_mentions_for_user("user-456")
        assert mentions[0].acknowledged is True
        assert mentions[0].acknowledged_at is not None

    @pytest.mark.asyncio
    async def test_acknowledge_mention_nonexistent(self, emitter):
        """acknowledge_mention returns False for unknown mention."""
        result = await emitter.acknowledge_mention("user-456", "nonexistent")
        assert result is False


# ===========================================================================
# Test TeamInboxEmitter - Internal Notes
# ===========================================================================


class TestTeamInboxEmitterNotes:
    """Tests for TeamInboxEmitter internal notes functionality."""

    @pytest.mark.asyncio
    async def test_add_note(self, emitter, mock_ws):
        """add_note creates and stores note, emits event."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        note = await emitter.add_note(
            inbox_id="inbox-123",
            message_id="msg-001",
            author_id="user-456",
            author_name="Test Author",
            content="This is an internal note",
        )

        assert note.content == "This is an internal note"
        assert note.author_id == "user-456"

        # Check event was sent
        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["type"] == "team_inbox_note_added"

    @pytest.mark.asyncio
    async def test_add_note_extracts_mentions(self, emitter, mock_ws):
        """add_note extracts @mentions from content."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        note = await emitter.add_note(
            inbox_id="inbox-123",
            message_id="msg-001",
            author_id="user-456",
            author_name="Test Author",
            content="Hey @john and @jane, please review",
        )

        assert "john" in note.mentions
        assert "jane" in note.mentions

    @pytest.mark.asyncio
    async def test_get_notes(self, emitter):
        """get_notes retrieves all notes for a message."""
        await emitter.add_note("inbox-1", "msg-001", "user-1", "Author 1", "Note 1")
        await emitter.add_note("inbox-1", "msg-001", "user-2", "Author 2", "Note 2")
        await emitter.add_note("inbox-1", "msg-002", "user-1", "Author 1", "Other message note")

        notes = await emitter.get_notes("msg-001")

        assert len(notes) == 2
        assert all(n.message_id == "msg-001" for n in notes)

    @pytest.mark.asyncio
    async def test_get_notes_empty(self, emitter):
        """get_notes returns empty list for message without notes."""
        notes = await emitter.get_notes("nonexistent")
        assert notes == []


# ===========================================================================
# Test TeamInboxEmitter - Activity Feed
# ===========================================================================


class TestTeamInboxEmitterActivity:
    """Tests for TeamInboxEmitter activity feed functionality."""

    @pytest.mark.asyncio
    async def test_emit_activity(self, emitter, mock_ws):
        """emit_activity sends generic activity event."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        await emitter.emit_activity(
            inbox_id="inbox-123",
            activity_type="tag_added",
            description="Added tag 'urgent'",
            user_id="user-456",
            user_name="Test User",
            message_id="msg-001",
            metadata={"tag": "urgent"},
        )

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["type"] == "team_inbox_activity"
        assert sent_message["data"]["activityType"] == "tag_added"
        assert sent_message["data"]["description"] == "Added tag 'urgent'"
        assert sent_message["data"]["metadata"]["tag"] == "urgent"

    @pytest.mark.asyncio
    async def test_emit_activity_without_metadata(self, emitter, mock_ws):
        """emit_activity works without metadata."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        await emitter.emit_activity(
            inbox_id="inbox-123",
            activity_type="viewed",
            description="User viewed message",
            user_id="user-456",
            user_name="Test User",
        )

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["data"]["metadata"] == {}


# ===========================================================================
# Test TeamInboxEmitter - Callbacks
# ===========================================================================


class TestTeamInboxEmitterCallbacks:
    """Tests for TeamInboxEmitter callback functionality."""

    @pytest.mark.asyncio
    async def test_add_callback_receives_events(self, emitter):
        """Added callback receives emitted events."""
        received_events = []

        def callback(event):
            received_events.append(event)

        emitter.add_callback(callback)

        event = TeamInboxEvent(
            type=TeamInboxEventType.MESSAGE_ASSIGNED,
            inbox_id="inbox-123",
            data={},
        )
        await emitter.emit(event)

        assert len(received_events) == 1
        assert received_events[0].type == TeamInboxEventType.MESSAGE_ASSIGNED

    @pytest.mark.asyncio
    async def test_multiple_callbacks_all_receive_events(self, emitter):
        """Multiple callbacks all receive events."""
        received_1 = []
        received_2 = []

        emitter.add_callback(lambda e: received_1.append(e))
        emitter.add_callback(lambda e: received_2.append(e))

        event = TeamInboxEvent(
            type=TeamInboxEventType.STATUS_CHANGED,
            inbox_id="inbox-123",
            data={},
        )
        await emitter.emit(event)

        assert len(received_1) == 1
        assert len(received_2) == 1

    @pytest.mark.asyncio
    async def test_remove_callback(self, emitter):
        """Removed callback no longer receives events."""
        received_events = []

        def callback(event):
            received_events.append(event)

        emitter.add_callback(callback)
        emitter.remove_callback(callback)

        await emitter.emit(
            TeamInboxEvent(
                type=TeamInboxEventType.MESSAGE_ASSIGNED,
                inbox_id="inbox-123",
                data={},
            )
        )

        assert len(received_events) == 0

    @pytest.mark.asyncio
    async def test_callback_error_does_not_break_emission(self, emitter, mock_ws):
        """Callback error doesn't prevent event emission."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        def failing_callback(event):
            raise RuntimeError("Callback failed")

        emitter.add_callback(failing_callback)

        event = TeamInboxEvent(
            type=TeamInboxEventType.MESSAGE_ASSIGNED,
            inbox_id="inbox-123",
            data={},
        )

        # Should not raise
        sent_count = await emitter.emit(event)
        assert sent_count == 1
        mock_ws.send.assert_called_once()


# ===========================================================================
# Test TeamInboxEmitter - Concurrent Operations
# ===========================================================================


class TestTeamInboxEmitterConcurrency:
    """Tests for TeamInboxEmitter concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_subscriptions(self, emitter):
        """Concurrent subscriptions are handled safely."""
        websockets = [MagicMock(send=AsyncMock()) for _ in range(10)]

        async def subscribe_ws(ws, i):
            await emitter.subscribe_to_inbox(f"inbox-{i % 3}", ws)

        await asyncio.gather(*[subscribe_ws(ws, i) for i, ws in enumerate(websockets)])

        # Verify all subscriptions were added
        total_subscriptions = sum(len(subs) for subs in emitter._inbox_subscriptions.values())
        assert total_subscriptions == 10

    @pytest.mark.asyncio
    async def test_concurrent_emissions(self, emitter, multiple_mock_ws):
        """Concurrent emissions don't interfere with each other."""
        for ws in multiple_mock_ws:
            await emitter.subscribe_to_inbox("inbox-123", ws)

        events = [
            TeamInboxEvent(
                type=TeamInboxEventType.STATUS_CHANGED,
                inbox_id="inbox-123",
                data={"status": f"status-{i}"},
            )
            for i in range(10)
        ]

        results = await asyncio.gather(*[emitter.emit(event) for event in events])

        assert all(count == 3 for count in results)

    @pytest.mark.asyncio
    async def test_concurrent_subscribe_and_emit(self, emitter):
        """Concurrent subscribe and emit operations are safe."""
        websocket = MagicMock(send=AsyncMock())

        async def subscribe_and_emit():
            await emitter.subscribe_to_inbox("inbox-123", websocket)
            event = TeamInboxEvent(
                type=TeamInboxEventType.MESSAGE_ASSIGNED,
                inbox_id="inbox-123",
                data={},
            )
            await emitter.emit(event)

        async def unsubscribe():
            await asyncio.sleep(0.01)
            await emitter.unsubscribe_from_inbox("inbox-123", websocket)

        await asyncio.gather(subscribe_and_emit(), unsubscribe())
        # Should not raise

    @pytest.mark.asyncio
    async def test_concurrent_mentions_creation(self, emitter):
        """Concurrent mention creation is thread-safe."""

        async def create_mentions():
            for i in range(10):
                await emitter.create_mention(
                    f"user-{i % 3}",
                    f"author-{i}",
                    f"msg-{i}",
                    "inbox-1",
                    f"context-{i}",
                )

        await asyncio.gather(*[create_mentions() for _ in range(3)])

        # Should have 30 total mentions across 3 users
        total_mentions = 0
        for user_id in ["user-0", "user-1", "user-2"]:
            mentions = await emitter.get_mentions_for_user(user_id)
            total_mentions += len(mentions)

        assert total_mentions == 30


# ===========================================================================
# Test TeamInboxEmitter - Error Handling
# ===========================================================================


class TestTeamInboxEmitterErrorHandling:
    """Tests for TeamInboxEmitter error handling."""

    @pytest.mark.asyncio
    async def test_handles_oserror_on_send(self, emitter):
        """OSError on send is handled gracefully."""
        mock_ws = MagicMock()
        mock_ws.send = AsyncMock(side_effect=OSError("Network error"))
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        event = TeamInboxEvent(
            type=TeamInboxEventType.MESSAGE_ASSIGNED,
            inbox_id="inbox-123",
            data={},
        )

        sent_count = await emitter.emit(event)
        assert sent_count == 0

    @pytest.mark.asyncio
    async def test_handles_runtime_error_on_send(self, emitter):
        """RuntimeError on send is handled gracefully."""
        mock_ws = MagicMock()
        mock_ws.send = AsyncMock(side_effect=RuntimeError("WebSocket closed"))
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        event = TeamInboxEvent(
            type=TeamInboxEventType.MESSAGE_ASSIGNED,
            inbox_id="inbox-123",
            data={},
        )

        sent_count = await emitter.emit(event)
        assert sent_count == 0
        # Dead connection should be cleaned up
        assert mock_ws not in emitter._inbox_subscriptions.get("inbox-123", set())

    @pytest.mark.asyncio
    async def test_partial_failure_still_sends_to_healthy_clients(self, emitter):
        """Partial failure doesn't prevent sending to healthy clients."""
        healthy_ws = MagicMock(send=AsyncMock())
        failing_ws = MagicMock(send=AsyncMock(side_effect=ConnectionError("Closed")))

        await emitter.subscribe_to_inbox("inbox-123", healthy_ws)
        await emitter.subscribe_to_inbox("inbox-123", failing_ws)

        event = TeamInboxEvent(
            type=TeamInboxEventType.STATUS_CHANGED,
            inbox_id="inbox-123",
            data={},
        )

        sent_count = await emitter.emit(event)
        assert sent_count == 1
        healthy_ws.send.assert_called_once()


# ===========================================================================
# Test Global Emitter and Convenience Functions
# ===========================================================================


class TestGlobalEmitter:
    """Tests for global emitter instance and convenience functions."""

    def test_get_team_inbox_emitter_creates_instance(self):
        """get_team_inbox_emitter creates a singleton instance."""
        emitter = get_team_inbox_emitter()
        assert emitter is not None
        assert isinstance(emitter, TeamInboxEmitter)

    def test_get_team_inbox_emitter_returns_same_instance(self):
        """get_team_inbox_emitter returns the same instance on subsequent calls."""
        emitter1 = get_team_inbox_emitter()
        emitter2 = get_team_inbox_emitter()
        assert emitter1 is emitter2

    @pytest.mark.asyncio
    async def test_emit_message_assigned_convenience(self):
        """emit_message_assigned convenience function works."""
        emitter = get_team_inbox_emitter()
        mock_ws = MagicMock(send=AsyncMock())
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        sent = await emit_message_assigned(
            "inbox-123",
            "msg-001",
            assigned_to_user_id="user-456",
            assigned_by_user_id="user-789",
            assigned_to_name="John",
            assigned_by_name="Jane",
        )
        assert sent == 1

    @pytest.mark.asyncio
    async def test_emit_message_unassigned_convenience(self):
        """emit_message_unassigned convenience function works."""
        emitter = get_team_inbox_emitter()
        mock_ws = MagicMock(send=AsyncMock())
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        sent = await emit_message_unassigned(
            "inbox-123",
            "msg-001",
            unassigned_by_user_id="user-789",
            unassigned_by_name="Admin",
        )
        assert sent == 1

    @pytest.mark.asyncio
    async def test_emit_status_changed_convenience(self):
        """emit_status_changed convenience function works."""
        emitter = get_team_inbox_emitter()
        mock_ws = MagicMock(send=AsyncMock())
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        sent = await emit_status_changed(
            "inbox-123",
            "msg-001",
            old_status="open",
            new_status="closed",
            changed_by_user_id="user-456",
            changed_by_name="Test User",
        )
        assert sent == 1

    @pytest.mark.asyncio
    async def test_emit_user_viewing_convenience(self):
        """emit_user_viewing convenience function works."""
        emitter = get_team_inbox_emitter()
        mock_ws = MagicMock(send=AsyncMock())
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        sent = await emit_user_viewing(
            "inbox-123",
            "msg-001",
            user_id="user-456",
            user_name="Test User",
        )
        assert sent == 1

    @pytest.mark.asyncio
    async def test_emit_user_typing_convenience(self):
        """emit_user_typing convenience function works."""
        emitter = get_team_inbox_emitter()
        mock_ws = MagicMock(send=AsyncMock())
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        sent = await emit_user_typing(
            "inbox-123",
            "msg-001",
            user_id="user-456",
            user_name="Test User",
        )
        assert sent == 1


# ===========================================================================
# Test Edge Cases
# ===========================================================================


class TestTeamInboxEdgeCases:
    """Tests for edge cases in team inbox functionality."""

    @pytest.mark.asyncio
    async def test_emit_to_empty_subscription_set(self, emitter):
        """Emit to inbox with empty subscription set (after all unsubscribe)."""
        mock_ws = MagicMock(send=AsyncMock())
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)
        await emitter.unsubscribe_from_inbox("inbox-123", mock_ws)

        event = TeamInboxEvent(
            type=TeamInboxEventType.MESSAGE_ASSIGNED,
            inbox_id="inbox-123",
            data={},
        )
        sent_count = await emitter.emit(event)
        assert sent_count == 0

    @pytest.mark.asyncio
    async def test_duplicate_subscription_same_websocket(self, emitter, mock_ws):
        """Duplicate subscription of same websocket is idempotent."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        # Should only have one subscription (sets deduplicate)
        assert len(emitter._inbox_subscriptions["inbox-123"]) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_already_unsubscribed(self, emitter, mock_ws):
        """Unsubscribing an already unsubscribed websocket is safe."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)
        await emitter.unsubscribe_from_inbox("inbox-123", mock_ws)
        await emitter.unsubscribe_from_inbox("inbox-123", mock_ws)
        # Should not raise

    def test_event_with_complex_data(self):
        """Event handles complex nested data structures."""
        event = TeamInboxEvent(
            type=TeamInboxEventType.ACTIVITY,
            inbox_id="inbox-123",
            data={
                "activity": {
                    "type": "tag_added",
                    "tags": ["urgent", "customer"],
                    "nested": {"level": 2, "items": [1, 2, 3]},
                },
            },
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["data"]["activity"]["tags"] == ["urgent", "customer"]
        assert parsed["data"]["activity"]["nested"]["items"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_emit_with_unicode_content(self, emitter, mock_ws):
        """Emit handles unicode content correctly."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        await emitter.add_note(
            inbox_id="inbox-123",
            message_id="msg-001",
            author_id="user-456",
            author_name="Test User",
            content="Note with unicode: \u65e5\u672c\u8a9e\u30c6\u30b9\u30c8 \U0001f4dd",
        )

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert "\u65e5\u672c\u8a9e" in sent_message["data"]["content"]

    @pytest.mark.asyncio
    async def test_remove_nonexistent_callback(self, emitter):
        """Removing a non-existent callback is safe."""

        def callback(e):
            pass

        emitter.remove_callback(callback)
        # Should not raise

    @pytest.mark.asyncio
    async def test_multiple_viewers_same_message(self, emitter, mock_ws):
        """Multiple users can view the same message."""
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        await emitter.emit_user_viewing("inbox-123", "msg-001", "user-1", "User 1")
        await emitter.emit_user_viewing("inbox-123", "msg-001", "user-2", "User 2")
        await emitter.emit_user_viewing("inbox-123", "msg-001", "user-3", "User 3")

        viewers = await emitter.get_message_viewers("msg-001")
        assert len(viewers) == 3
        assert set(viewers) == {"user-1", "user-2", "user-3"}

    @pytest.mark.asyncio
    async def test_notes_are_independent_per_message(self, emitter):
        """Notes for different messages are stored independently."""
        await emitter.add_note("inbox-1", "msg-001", "user-1", "Author", "Note for msg 1")
        await emitter.add_note("inbox-1", "msg-002", "user-1", "Author", "Note for msg 2")

        notes_1 = await emitter.get_notes("msg-001")
        notes_2 = await emitter.get_notes("msg-002")

        assert len(notes_1) == 1
        assert len(notes_2) == 1
        assert notes_1[0].content == "Note for msg 1"
        assert notes_2[0].content == "Note for msg 2"


# ===========================================================================
# Test Integration Scenarios
# ===========================================================================


class TestTeamInboxIntegration:
    """Integration tests for team inbox components working together."""

    @pytest.mark.asyncio
    async def test_full_collaboration_workflow(self, emitter):
        """Test a complete collaboration workflow."""
        # Setup
        mock_ws = MagicMock(send=AsyncMock())
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        # Add team members
        await emitter.add_team_member("inbox-123", "alice", "alice@test.com", "Alice", "admin")
        await emitter.add_team_member("inbox-123", "bob", "bob@test.com", "Bob", "member")

        # Alice assigns message to Bob
        await emitter.emit_message_assigned(
            "inbox-123",
            "msg-001",
            assigned_to_user_id="bob",
            assigned_by_user_id="alice",
            assigned_to_name="Bob",
            assigned_by_name="Alice",
        )

        # Bob views the message
        await emitter.emit_user_viewing("inbox-123", "msg-001", "bob", "Bob")

        # Bob starts typing
        await emitter.emit_user_typing("inbox-123", "msg-001", "bob", "Bob")

        # Bob adds a note mentioning Alice
        note = await emitter.add_note(
            "inbox-123",
            "msg-001",
            "bob",
            "Bob",
            "Hey @alice, I think we should escalate this",
        )

        # Bob stops typing
        await emitter.emit_user_stopped_typing("inbox-123", "msg-001", "bob", "Bob")

        # Bob changes status
        await emitter.emit_status_changed(
            "inbox-123",
            "msg-001",
            "open",
            "in_progress",
            "bob",
            "Bob",
        )

        # Verify state
        viewers = await emitter.get_message_viewers("msg-001")
        assert "bob" in viewers

        notes = await emitter.get_notes("msg-001")
        assert len(notes) == 1
        assert "alice" in notes[0].mentions

        team = await emitter.get_team_members("inbox-123")
        assert len(team) == 2

        # Verify events were sent (should be 6: assign, view, type, note, stop_type, status)
        assert mock_ws.send.call_count == 6

    @pytest.mark.asyncio
    async def test_mention_notification_workflow(self, emitter):
        """Test mention notification workflow."""
        mock_ws = MagicMock(send=AsyncMock())
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        # Create mentions for a user
        mention1 = await emitter.create_mention(
            "user-bob", "user-alice", "msg-001", "inbox-123", "@bob urgent issue"
        )
        mention2 = await emitter.create_mention(
            "user-bob", "user-charlie", "msg-002", "inbox-123", "@bob please review"
        )

        # Bob checks unacknowledged mentions
        unacked = await emitter.get_mentions_for_user("user-bob", unacknowledged_only=True)
        assert len(unacked) == 2

        # Bob acknowledges first mention
        await emitter.acknowledge_mention("user-bob", mention1.id)

        # Check remaining unacknowledged
        unacked = await emitter.get_mentions_for_user("user-bob", unacknowledged_only=True)
        assert len(unacked) == 1
        assert unacked[0].id == mention2.id

    @pytest.mark.asyncio
    async def test_presence_tracking_workflow(self, emitter):
        """Test presence tracking with multiple users."""
        mock_ws = MagicMock(send=AsyncMock())
        await emitter.subscribe_to_inbox("inbox-123", mock_ws)

        # Multiple users view same message
        await emitter.emit_user_viewing("inbox-123", "msg-001", "user-1", "User 1")
        await emitter.emit_user_viewing("inbox-123", "msg-001", "user-2", "User 2")
        await emitter.emit_user_viewing("inbox-123", "msg-001", "user-3", "User 3")

        # Check viewers
        viewers = await emitter.get_message_viewers("msg-001")
        assert len(viewers) == 3

        # User 2 starts typing
        await emitter.emit_user_typing("inbox-123", "msg-001", "user-2", "User 2")

        # User 1 leaves
        await emitter.emit_user_left("inbox-123", "msg-001", "user-1", "User 1")

        # Check updated viewers
        viewers = await emitter.get_message_viewers("msg-001")
        assert len(viewers) == 2
        assert "user-1" not in viewers
        assert "user-2" in viewers
        assert "user-3" in viewers
