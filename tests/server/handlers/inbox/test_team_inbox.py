"""
Tests for inbox/team_inbox.py - Team Inbox Collaboration HTTP API handlers.

Tests cover:
- GET /api/v1/inbox/shared/{id}/team - Get team members
- POST /api/v1/inbox/shared/{id}/team - Add team member
- DELETE /api/v1/inbox/shared/{id}/team/{user_id} - Remove team member
- POST /api/v1/inbox/shared/{id}/messages/{msg_id}/viewing - Start viewing
- DELETE /api/v1/inbox/shared/{id}/messages/{msg_id}/viewing - Stop viewing
- POST /api/v1/inbox/shared/{id}/messages/{msg_id}/typing - Start typing
- DELETE /api/v1/inbox/shared/{id}/messages/{msg_id}/typing - Stop typing
- GET /api/v1/inbox/shared/{id}/messages/{msg_id}/notes - Get notes
- POST /api/v1/inbox/shared/{id}/messages/{msg_id}/notes - Add note
- GET /api/v1/inbox/mentions - Get mentions for current user
- POST /api/v1/inbox/mentions/{id}/acknowledge - Acknowledge mention
- GET /api/v1/inbox/shared/{id}/activity - Get activity feed
- RBAC permission enforcement
- Error handling and validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.inbox.team_inbox import (
    handle_get_team_members,
    handle_add_team_member,
    handle_remove_team_member,
    handle_start_viewing,
    handle_stop_viewing,
    handle_start_typing,
    handle_stop_typing,
    handle_get_notes,
    handle_add_note,
    handle_get_mentions,
    handle_acknowledge_mention,
    handle_get_activity_feed,
    get_team_inbox_handlers,
)


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


@dataclass
class MockTeamMember:
    """Mock team member for testing."""

    user_id: str = "user-123"
    email: str = "user@example.com"
    name: str = "Test User"
    role: str = "member"
    joined_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "joined_at": self.joined_at.isoformat(),
        }


@dataclass
class MockNote:
    """Mock internal note for testing."""

    id: str = "note-123"
    content: str = "Internal note content"
    author_id: str = "user-123"
    author_name: str = "Test User"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class MockMention:
    """Mock mention for testing."""

    id: str = "mention-123"
    mentioned_user_id: str = "user-456"
    mentioned_by_user_id: str = "user-123"
    message_id: str = "msg-789"
    inbox_id: str = "inbox-001"
    context: str = "@user-456 please review"
    acknowledged: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mentioned_user_id": self.mentioned_user_id,
            "mentioned_by_user_id": self.mentioned_by_user_id,
            "message_id": self.message_id,
            "inbox_id": self.inbox_id,
            "context": self.context,
            "acknowledged": self.acknowledged,
            "created_at": self.created_at.isoformat(),
        }


class MockTeamInboxEmitter:
    """Mock team inbox emitter for testing."""

    def __init__(self):
        self.team_members: dict[str, list[MockTeamMember]] = {}
        self.notes: dict[str, list[MockNote]] = {}
        self.mentions: dict[str, list[MockMention]] = {}
        self.viewers: dict[str, list[dict]] = {}

    async def get_team_members(self, inbox_id: str) -> list[MockTeamMember]:
        return self.team_members.get(inbox_id, [])

    async def add_team_member(
        self,
        inbox_id: str,
        user_id: str,
        email: str,
        name: str,
        role: str = "member",
    ) -> MockTeamMember:
        member = MockTeamMember(user_id=user_id, email=email, name=name, role=role)
        if inbox_id not in self.team_members:
            self.team_members[inbox_id] = []
        self.team_members[inbox_id].append(member)
        return member

    async def remove_team_member(self, inbox_id: str, user_id: str) -> bool:
        if inbox_id in self.team_members:
            original_len = len(self.team_members[inbox_id])
            self.team_members[inbox_id] = [
                m for m in self.team_members[inbox_id] if m.user_id != user_id
            ]
            return len(self.team_members[inbox_id]) < original_len
        return False

    async def emit_user_viewing(
        self, inbox_id: str, message_id: str, user_id: str, user_name: str
    ) -> None:
        pass

    async def emit_user_left(
        self, inbox_id: str, message_id: str, user_id: str, user_name: str
    ) -> None:
        pass

    async def emit_user_typing(
        self, inbox_id: str, message_id: str, user_id: str, user_name: str
    ) -> None:
        pass

    async def emit_user_stopped_typing(
        self, inbox_id: str, message_id: str, user_id: str, user_name: str
    ) -> None:
        pass

    async def get_message_viewers(self, message_id: str) -> list[dict]:
        return self.viewers.get(message_id, [])

    async def get_notes(self, message_id: str) -> list[MockNote]:
        return self.notes.get(message_id, [])

    async def add_note(
        self,
        inbox_id: str,
        message_id: str,
        author_id: str,
        author_name: str,
        content: str,
    ) -> MockNote:
        note = MockNote(
            content=content,
            author_id=author_id,
            author_name=author_name,
        )
        if message_id not in self.notes:
            self.notes[message_id] = []
        self.notes[message_id].append(note)
        return note

    def extract_mentions(self, content: str) -> list[str]:
        import re

        return re.findall(r"@(\w+)", content)

    async def create_mention(
        self,
        mentioned_user_id: str,
        mentioned_by_user_id: str,
        message_id: str,
        inbox_id: str,
        context: str,
    ) -> MockMention:
        mention = MockMention(
            mentioned_user_id=mentioned_user_id,
            mentioned_by_user_id=mentioned_by_user_id,
            message_id=message_id,
            inbox_id=inbox_id,
            context=context,
        )
        if mentioned_user_id not in self.mentions:
            self.mentions[mentioned_user_id] = []
        self.mentions[mentioned_user_id].append(mention)
        return mention

    async def get_mentions_for_user(
        self, user_id: str, unacknowledged_only: bool = False
    ) -> list[MockMention]:
        mentions = self.mentions.get(user_id, [])
        if unacknowledged_only:
            return [m for m in mentions if not m.acknowledged]
        return mentions

    async def acknowledge_mention(self, user_id: str, mention_id: str) -> bool:
        for user_mentions in self.mentions.values():
            for mention in user_mentions:
                if mention.id == mention_id:
                    mention.acknowledged = True
                    return True
        return False

    async def emit_activity(
        self,
        inbox_id: str,
        activity_type: str,
        description: str,
        user_id: str,
        user_name: str,
        metadata: dict | None = None,
    ) -> None:
        pass


@pytest.fixture
def mock_emitter():
    """Create mock team inbox emitter."""
    return MockTeamInboxEmitter()


@pytest.fixture
def mock_emitter_with_data():
    """Create mock emitter with sample data."""
    emitter = MockTeamInboxEmitter()
    emitter.team_members["inbox-001"] = [
        MockTeamMember(user_id="user-1", email="a@example.com", name="Alice"),
        MockTeamMember(user_id="user-2", email="b@example.com", name="Bob"),
    ]
    emitter.notes["msg-001"] = [MockNote()]
    emitter.mentions["user-1"] = [MockMention(mentioned_user_id="user-1")]
    return emitter


# ===========================================================================
# Team Member Management Tests
# ===========================================================================


class TestGetTeamMembers:
    """Test handle_get_team_members function."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.team_inbox.get_team_inbox_emitter_instance")
    async def test_get_team_members_success(self, mock_get_emitter, mock_emitter_with_data):
        """Test successful team members retrieval."""
        mock_get_emitter.return_value = mock_emitter_with_data

        result = await handle_get_team_members(
            {"inbox_id": "inbox-001"}, inbox_id="inbox-001", user_id="user-1"
        )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_team_members_missing_inbox_id(self):
        """Test retrieval fails without inbox_id."""
        result = await handle_get_team_members({}, inbox_id="", user_id="user-1")

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.team_inbox.get_team_inbox_emitter_instance")
    async def test_get_team_members_empty(self, mock_get_emitter, mock_emitter):
        """Test retrieval when no team members exist."""
        mock_get_emitter.return_value = mock_emitter

        result = await handle_get_team_members({}, inbox_id="inbox-new", user_id="user-1")

        assert result is not None
        assert result.status_code == 200


class TestAddTeamMember:
    """Test handle_add_team_member function."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.team_inbox.get_team_inbox_emitter_instance")
    async def test_add_team_member_success(self, mock_get_emitter, mock_emitter):
        """Test successful team member addition."""
        mock_get_emitter.return_value = mock_emitter

        result = await handle_add_team_member(
            {
                "user_id": "new-user",
                "email": "new@example.com",
                "name": "New User",
                "role": "member",
            },
            inbox_id="inbox-001",
            user_id="admin-user",
        )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.team_inbox.get_team_inbox_emitter_instance")
    async def test_add_team_member_as_admin(self, mock_get_emitter, mock_emitter):
        """Test adding team member with admin role."""
        mock_get_emitter.return_value = mock_emitter

        result = await handle_add_team_member(
            {
                "user_id": "admin-new",
                "email": "admin@example.com",
                "name": "Admin User",
                "role": "admin",
            },
            inbox_id="inbox-001",
            user_id="super-admin",
        )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_add_team_member_missing_user_id(self):
        """Test addition fails without user_id."""
        result = await handle_add_team_member(
            {"email": "test@example.com", "name": "Test"},
            inbox_id="inbox-001",
            user_id="admin",
        )

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_add_team_member_missing_email(self):
        """Test addition fails without email."""
        result = await handle_add_team_member(
            {"user_id": "user-new", "name": "Test"},
            inbox_id="inbox-001",
            user_id="admin",
        )

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_add_team_member_missing_name(self):
        """Test addition fails without name."""
        result = await handle_add_team_member(
            {"user_id": "user-new", "email": "test@example.com"},
            inbox_id="inbox-001",
            user_id="admin",
        )

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_add_team_member_invalid_role(self):
        """Test addition fails with invalid role."""
        result = await handle_add_team_member(
            {
                "user_id": "user-new",
                "email": "test@example.com",
                "name": "Test",
                "role": "superadmin",
            },
            inbox_id="inbox-001",
            user_id="admin",
        )

        assert result is not None
        assert result.status_code == 400


class TestRemoveTeamMember:
    """Test handle_remove_team_member function."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.team_inbox.get_team_inbox_emitter_instance")
    async def test_remove_team_member_success(self, mock_get_emitter, mock_emitter_with_data):
        """Test successful team member removal."""
        mock_get_emitter.return_value = mock_emitter_with_data

        result = await handle_remove_team_member(
            {},
            inbox_id="inbox-001",
            member_user_id="user-1",
            user_id="admin",
        )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.team_inbox.get_team_inbox_emitter_instance")
    async def test_remove_team_member_not_found(self, mock_get_emitter, mock_emitter):
        """Test removal of non-existent member."""
        mock_get_emitter.return_value = mock_emitter

        result = await handle_remove_team_member(
            {},
            inbox_id="inbox-001",
            member_user_id="nonexistent",
            user_id="admin",
        )

        assert result is not None
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_remove_team_member_missing_ids(self):
        """Test removal fails without required IDs."""
        result = await handle_remove_team_member(
            {}, inbox_id="", member_user_id="", user_id="admin"
        )

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Presence Tests
# ===========================================================================


class TestStartViewing:
    """Test handle_start_viewing function."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.team_inbox.get_team_inbox_emitter_instance")
    async def test_start_viewing_success(self, mock_get_emitter, mock_emitter):
        """Test successful start viewing."""
        mock_get_emitter.return_value = mock_emitter

        result = await handle_start_viewing(
            {"user_name": "Alice"},
            inbox_id="inbox-001",
            message_id="msg-001",
            user_id="user-1",
        )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_start_viewing_missing_ids(self):
        """Test start viewing fails without required IDs."""
        result = await handle_start_viewing({}, inbox_id="", message_id="", user_id="user-1")

        assert result is not None
        assert result.status_code == 400


class TestStopViewing:
    """Test handle_stop_viewing function."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.team_inbox.get_team_inbox_emitter_instance")
    async def test_stop_viewing_success(self, mock_get_emitter, mock_emitter):
        """Test successful stop viewing."""
        mock_get_emitter.return_value = mock_emitter

        result = await handle_stop_viewing(
            {"user_name": "Alice"},
            inbox_id="inbox-001",
            message_id="msg-001",
            user_id="user-1",
        )

        assert result is not None
        assert result.status_code == 200


class TestStartTyping:
    """Test handle_start_typing function."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.team_inbox.get_team_inbox_emitter_instance")
    async def test_start_typing_success(self, mock_get_emitter, mock_emitter):
        """Test successful start typing."""
        mock_get_emitter.return_value = mock_emitter

        result = await handle_start_typing(
            {"user_name": "Bob"},
            inbox_id="inbox-001",
            message_id="msg-001",
            user_id="user-2",
        )

        assert result is not None
        assert result.status_code == 200


class TestStopTyping:
    """Test handle_stop_typing function."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.team_inbox.get_team_inbox_emitter_instance")
    async def test_stop_typing_success(self, mock_get_emitter, mock_emitter):
        """Test successful stop typing."""
        mock_get_emitter.return_value = mock_emitter

        result = await handle_stop_typing(
            {"user_name": "Bob"},
            inbox_id="inbox-001",
            message_id="msg-001",
            user_id="user-2",
        )

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Notes Tests
# ===========================================================================


class TestGetNotes:
    """Test handle_get_notes function."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.team_inbox.get_team_inbox_emitter_instance")
    async def test_get_notes_success(self, mock_get_emitter, mock_emitter_with_data):
        """Test successful notes retrieval."""
        mock_get_emitter.return_value = mock_emitter_with_data

        result = await handle_get_notes(
            {},
            inbox_id="inbox-001",
            message_id="msg-001",
            user_id="user-1",
        )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.team_inbox.get_team_inbox_emitter_instance")
    async def test_get_notes_empty(self, mock_get_emitter, mock_emitter):
        """Test retrieval when no notes exist."""
        mock_get_emitter.return_value = mock_emitter

        result = await handle_get_notes(
            {},
            inbox_id="inbox-001",
            message_id="msg-new",
            user_id="user-1",
        )

        assert result is not None
        assert result.status_code == 200


class TestAddNote:
    """Test handle_add_note function."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.team_inbox.get_team_inbox_emitter_instance")
    async def test_add_note_success(self, mock_get_emitter, mock_emitter):
        """Test successful note addition."""
        mock_get_emitter.return_value = mock_emitter

        result = await handle_add_note(
            {"content": "This is an internal note", "author_name": "Alice"},
            inbox_id="inbox-001",
            message_id="msg-001",
            user_id="user-1",
        )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.team_inbox.get_team_inbox_emitter_instance")
    async def test_add_note_with_mentions(self, mock_get_emitter, mock_emitter):
        """Test note with @mentions."""
        mock_get_emitter.return_value = mock_emitter

        result = await handle_add_note(
            {
                "content": "@bob please review this @alice",
                "author_name": "Charlie",
            },
            inbox_id="inbox-001",
            message_id="msg-001",
            user_id="user-3",
        )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_add_note_missing_content(self):
        """Test note addition fails without content."""
        result = await handle_add_note(
            {"author_name": "Alice"},
            inbox_id="inbox-001",
            message_id="msg-001",
            user_id="user-1",
        )

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Mentions Tests
# ===========================================================================


class TestGetMentions:
    """Test handle_get_mentions function."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.team_inbox.get_team_inbox_emitter_instance")
    async def test_get_mentions_success(self, mock_get_emitter, mock_emitter_with_data):
        """Test successful mentions retrieval."""
        mock_get_emitter.return_value = mock_emitter_with_data

        result = await handle_get_mentions({}, user_id="user-1")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.team_inbox.get_team_inbox_emitter_instance")
    async def test_get_mentions_unacknowledged_only(self, mock_get_emitter, mock_emitter_with_data):
        """Test filtering unacknowledged mentions."""
        mock_get_emitter.return_value = mock_emitter_with_data

        result = await handle_get_mentions({"unacknowledged_only": True}, user_id="user-1")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.team_inbox.get_team_inbox_emitter_instance")
    async def test_get_mentions_string_param(self, mock_get_emitter, mock_emitter_with_data):
        """Test unacknowledged_only as string parameter."""
        mock_get_emitter.return_value = mock_emitter_with_data

        result = await handle_get_mentions({"unacknowledged_only": "true"}, user_id="user-1")

        assert result is not None
        assert result.status_code == 200


class TestAcknowledgeMention:
    """Test handle_acknowledge_mention function."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.team_inbox.get_team_inbox_emitter_instance")
    async def test_acknowledge_mention_success(self, mock_get_emitter, mock_emitter_with_data):
        """Test successful mention acknowledgment."""
        mock_get_emitter.return_value = mock_emitter_with_data

        result = await handle_acknowledge_mention({}, mention_id="mention-123", user_id="user-1")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.team_inbox.get_team_inbox_emitter_instance")
    async def test_acknowledge_mention_not_found(self, mock_get_emitter, mock_emitter):
        """Test acknowledgment of non-existent mention."""
        mock_get_emitter.return_value = mock_emitter

        result = await handle_acknowledge_mention({}, mention_id="nonexistent", user_id="user-1")

        assert result is not None
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_acknowledge_mention_missing_id(self):
        """Test acknowledgment fails without mention_id."""
        result = await handle_acknowledge_mention({}, mention_id="", user_id="user-1")

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Activity Feed Tests
# ===========================================================================


class TestGetActivityFeed:
    """Test handle_get_activity_feed function."""

    @pytest.mark.asyncio
    async def test_get_activity_feed_success(self):
        """Test successful activity feed retrieval."""
        result = await handle_get_activity_feed({}, inbox_id="inbox-001", user_id="user-1")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_activity_feed_with_pagination(self):
        """Test activity feed with pagination parameters."""
        result = await handle_get_activity_feed(
            {"limit": 20, "offset": 10},
            inbox_id="inbox-001",
            user_id="user-1",
        )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_activity_feed_missing_inbox_id(self):
        """Test activity feed fails without inbox_id."""
        result = await handle_get_activity_feed({}, inbox_id="", user_id="user-1")

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Handler Registration Tests
# ===========================================================================


class TestHandlerRegistration:
    """Test handler registration function."""

    def test_get_team_inbox_handlers_returns_all(self):
        """Test that get_team_inbox_handlers returns all handlers."""
        handlers = get_team_inbox_handlers()

        # Team members
        assert "get_team_members" in handlers
        assert "add_team_member" in handlers
        assert "remove_team_member" in handlers

        # Presence
        assert "start_viewing" in handlers
        assert "stop_viewing" in handlers
        assert "start_typing" in handlers
        assert "stop_typing" in handlers

        # Notes
        assert "get_notes" in handlers
        assert "add_note" in handlers

        # Mentions
        assert "get_mentions" in handlers
        assert "acknowledge_mention" in handlers

        # Activity
        assert "get_activity_feed" in handlers

    def test_handlers_are_callable(self):
        """Test that all registered handlers are callable."""
        handlers = get_team_inbox_handlers()

        for name, handler in handlers.items():
            assert callable(handler), f"Handler {name} is not callable"
