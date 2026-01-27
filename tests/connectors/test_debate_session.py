"""Tests for debate session management across channels.

Tests the DebateSession dataclass and DebateSessionManager for
multi-channel session tracking and handoff.
"""

import pytest
import time
from unittest.mock import MagicMock, patch


class TestDebateSession:
    """Tests for the DebateSession dataclass."""

    def test_create_session(self):
        """Should create a session with required fields."""
        from aragora.server.session_store import DebateSession

        session = DebateSession(
            session_id="telegram:user123:abc12345",
            channel="telegram",
            user_id="user123",
        )

        assert session.session_id == "telegram:user123:abc12345"
        assert session.channel == "telegram"
        assert session.user_id == "user123"
        assert session.debate_id is None
        assert session.context == {}
        assert session.created_at > 0
        assert session.last_active > 0

    def test_link_debate(self):
        """Should link a debate and update last_active."""
        from aragora.server.session_store import DebateSession

        session = DebateSession(
            session_id="test:user:id",
            channel="test",
            user_id="user",
        )
        initial_time = session.last_active

        # Small delay to ensure time difference
        time.sleep(0.01)

        session.link_debate("debate-123")

        assert session.debate_id == "debate-123"
        assert session.last_active > initial_time

    def test_unlink_debate(self):
        """Should unlink debate and clear debate_id."""
        from aragora.server.session_store import DebateSession

        session = DebateSession(
            session_id="test:user:id",
            channel="test",
            user_id="user",
            debate_id="debate-123",
        )

        session.unlink_debate()

        assert session.debate_id is None

    def test_touch_updates_last_active(self):
        """Should update last_active timestamp."""
        from aragora.server.session_store import DebateSession

        session = DebateSession(
            session_id="test:user:id",
            channel="test",
            user_id="user",
        )
        initial_time = session.last_active

        time.sleep(0.01)
        session.touch()

        assert session.last_active > initial_time

    def test_to_dict_serialization(self):
        """Should serialize all fields to dict."""
        from aragora.server.session_store import DebateSession

        session = DebateSession(
            session_id="telegram:user123:abc12345",
            channel="telegram",
            user_id="user123",
            debate_id="debate-xyz",
            context={"username": "john"},
        )

        data = session.to_dict()

        assert data["session_id"] == "telegram:user123:abc12345"
        assert data["channel"] == "telegram"
        assert data["user_id"] == "user123"
        assert data["debate_id"] == "debate-xyz"
        assert data["context"] == {"username": "john"}
        assert "created_at" in data
        assert "last_active" in data

    def test_from_dict_deserialization(self):
        """Should deserialize from dict."""
        from aragora.server.session_store import DebateSession

        data = {
            "session_id": "slack:user456:def67890",
            "channel": "slack",
            "user_id": "user456",
            "debate_id": "debate-abc",
            "context": {"team": "engineering"},
            "created_at": 1700000000.0,
            "last_active": 1700000100.0,
        }

        session = DebateSession.from_dict(data)

        assert session.session_id == "slack:user456:def67890"
        assert session.channel == "slack"
        assert session.user_id == "user456"
        assert session.debate_id == "debate-abc"
        assert session.context == {"team": "engineering"}
        assert session.created_at == 1700000000.0
        assert session.last_active == 1700000100.0


class TestInMemorySessionStoreDebateSessions:
    """Tests for debate session methods in InMemorySessionStore."""

    @pytest.fixture
    def store(self):
        """Create a fresh InMemorySessionStore."""
        from aragora.server.session_store import InMemorySessionStore

        return InMemorySessionStore()

    def test_get_debate_session_not_found(self, store):
        """Should return None for non-existent session."""
        result = store.get_debate_session("nonexistent")
        assert result is None

    def test_set_and_get_debate_session(self, store):
        """Should store and retrieve a session."""
        from aragora.server.session_store import DebateSession

        session = DebateSession(
            session_id="test:user:id",
            channel="test",
            user_id="user",
        )

        store.set_debate_session(session)
        retrieved = store.get_debate_session("test:user:id")

        assert retrieved is not None
        assert retrieved.session_id == "test:user:id"
        assert retrieved.channel == "test"
        assert retrieved.user_id == "user"

    def test_delete_debate_session(self, store):
        """Should delete a session."""
        from aragora.server.session_store import DebateSession

        session = DebateSession(
            session_id="test:user:id",
            channel="test",
            user_id="user",
        )
        store.set_debate_session(session)

        result = store.delete_debate_session("test:user:id")

        assert result is True
        assert store.get_debate_session("test:user:id") is None

    def test_delete_nonexistent_session(self, store):
        """Should return False when deleting nonexistent session."""
        result = store.delete_debate_session("nonexistent")
        assert result is False

    def test_find_sessions_by_user(self, store):
        """Should find sessions by user ID."""
        from aragora.server.session_store import DebateSession

        session1 = DebateSession(
            session_id="telegram:user123:a",
            channel="telegram",
            user_id="user123",
        )
        session2 = DebateSession(
            session_id="slack:user123:b",
            channel="slack",
            user_id="user123",
        )
        session3 = DebateSession(
            session_id="telegram:user456:c",
            channel="telegram",
            user_id="user456",
        )

        store.set_debate_session(session1)
        store.set_debate_session(session2)
        store.set_debate_session(session3)

        # Find all sessions for user123
        results = store.find_sessions_by_user("user123")
        assert len(results) == 2

        # Find sessions for user123 on telegram only
        results = store.find_sessions_by_user("user123", channel="telegram")
        assert len(results) == 1
        assert results[0].session_id == "telegram:user123:a"

    def test_find_sessions_by_debate(self, store):
        """Should find sessions linked to a debate."""
        from aragora.server.session_store import DebateSession

        session1 = DebateSession(
            session_id="telegram:user123:a",
            channel="telegram",
            user_id="user123",
            debate_id="debate-xyz",
        )
        session2 = DebateSession(
            session_id="slack:user456:b",
            channel="slack",
            user_id="user456",
            debate_id="debate-xyz",
        )
        session3 = DebateSession(
            session_id="telegram:user789:c",
            channel="telegram",
            user_id="user789",
            debate_id="debate-other",
        )

        store.set_debate_session(session1)
        store.set_debate_session(session2)
        store.set_debate_session(session3)

        results = store.find_sessions_by_debate("debate-xyz")
        assert len(results) == 2
        session_ids = {s.session_id for s in results}
        assert "telegram:user123:a" in session_ids
        assert "slack:user456:b" in session_ids


class TestDebateSessionManager:
    """Tests for the DebateSessionManager."""

    @pytest.fixture
    def mock_store(self):
        """Mock the session store singleton."""
        with patch("aragora.connectors.debate_session.get_session_store") as mock:
            from aragora.server.session_store import InMemorySessionStore

            store = InMemorySessionStore()
            mock.return_value = store
            yield store

    @pytest.fixture
    def manager(self, mock_store):
        """Create a DebateSessionManager with mocked store."""
        from aragora.connectors.debate_session import DebateSessionManager

        return DebateSessionManager()

    @pytest.mark.asyncio
    async def test_create_session(self, manager, mock_store):
        """Should create a new session."""
        session = await manager.create_session("telegram", "user123")

        assert session.channel == "telegram"
        assert session.user_id == "user123"
        assert session.session_id.startswith("telegram:user123:")
        assert session.debate_id is None

    @pytest.mark.asyncio
    async def test_create_session_with_context(self, manager, mock_store):
        """Should create session with custom context."""
        context = {"username": "john_doe", "language": "en"}
        session = await manager.create_session("telegram", "user123", context=context)

        assert session.context == context

    @pytest.mark.asyncio
    async def test_get_session(self, manager, mock_store):
        """Should retrieve an existing session."""
        created = await manager.create_session("telegram", "user123")
        retrieved = await manager.get_session(created.session_id)

        assert retrieved is not None
        assert retrieved.session_id == created.session_id

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, manager, mock_store):
        """Should return None for nonexistent session."""
        result = await manager.get_session("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_link_debate(self, manager, mock_store):
        """Should link a debate to a session."""
        session = await manager.create_session("telegram", "user123")
        result = await manager.link_debate(session.session_id, "debate-abc")

        assert result is True

        updated = await manager.get_session(session.session_id)
        assert updated.debate_id == "debate-abc"

    @pytest.mark.asyncio
    async def test_link_debate_session_not_found(self, manager, mock_store):
        """Should return False when linking to nonexistent session."""
        result = await manager.link_debate("nonexistent", "debate-abc")
        assert result is False

    @pytest.mark.asyncio
    async def test_unlink_debate(self, manager, mock_store):
        """Should unlink a debate from a session."""
        session = await manager.create_session("telegram", "user123")
        await manager.link_debate(session.session_id, "debate-abc")

        result = await manager.unlink_debate(session.session_id)

        assert result is True

        updated = await manager.get_session(session.session_id)
        assert updated.debate_id is None

    @pytest.mark.asyncio
    async def test_find_sessions_for_user(self, manager, mock_store):
        """Should find all sessions for a user."""
        await manager.create_session("telegram", "user123")
        await manager.create_session("slack", "user123")
        await manager.create_session("telegram", "user456")

        results = await manager.find_sessions_for_user("user123")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_find_sessions_for_debate(self, manager, mock_store):
        """Should find all sessions linked to a debate."""
        session1 = await manager.create_session("telegram", "user123")
        session2 = await manager.create_session("slack", "user456")
        session3 = await manager.create_session("telegram", "user789")

        await manager.link_debate(session1.session_id, "debate-xyz")
        await manager.link_debate(session2.session_id, "debate-xyz")
        await manager.link_debate(session3.session_id, "debate-other")

        results = await manager.find_sessions_for_debate("debate-xyz")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_or_create_session_existing(self, manager, mock_store):
        """Should return existing session when available."""
        created = await manager.create_session("telegram", "user123")

        result = await manager.get_or_create_session("telegram", "user123")

        assert result.session_id == created.session_id

    @pytest.mark.asyncio
    async def test_get_or_create_session_new(self, manager, mock_store):
        """Should create new session when none exists."""
        result = await manager.get_or_create_session("telegram", "user123")

        assert result.channel == "telegram"
        assert result.user_id == "user123"

    @pytest.mark.asyncio
    async def test_handoff(self, manager, mock_store):
        """Should handoff session to new channel."""
        original = await manager.create_session("telegram", "user123")
        await manager.link_debate(original.session_id, "debate-abc")

        new_session = await manager.handoff(original.session_id, "slack")

        assert new_session is not None
        assert new_session.channel == "slack"
        assert new_session.user_id == "user123"
        assert new_session.debate_id == "debate-abc"  # Preserved
        assert new_session.context["handoff_from"] == original.session_id
        assert new_session.context["handoff_channel"] == "telegram"

    @pytest.mark.asyncio
    async def test_handoff_without_debate(self, manager, mock_store):
        """Should handoff session without debate link."""
        original = await manager.create_session("telegram", "user123")

        new_session = await manager.handoff(original.session_id, "slack", preserve_debate=False)

        assert new_session.debate_id is None

    @pytest.mark.asyncio
    async def test_handoff_session_not_found(self, manager, mock_store):
        """Should return None for nonexistent session handoff."""
        result = await manager.handoff("nonexistent", "slack")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_context(self, manager, mock_store):
        """Should update session context."""
        session = await manager.create_session("telegram", "user123")

        result = await manager.update_context(session.session_id, {"new_key": "new_value"})

        assert result is True

        updated = await manager.get_session(session.session_id)
        assert updated.context["new_key"] == "new_value"

    @pytest.mark.asyncio
    async def test_delete_session(self, manager, mock_store):
        """Should delete a session."""
        session = await manager.create_session("telegram", "user123")

        result = await manager.delete_session(session.session_id)

        assert result is True
        assert await manager.get_session(session.session_id) is None

    @pytest.mark.asyncio
    async def test_get_active_debate(self, manager, mock_store):
        """Should get active debate ID for session."""
        session = await manager.create_session("telegram", "user123")
        await manager.link_debate(session.session_id, "debate-abc")

        debate_id = await manager.get_active_debate(session.session_id)

        assert debate_id == "debate-abc"

    @pytest.mark.asyncio
    async def test_get_active_debate_none(self, manager, mock_store):
        """Should return None when no debate linked."""
        session = await manager.create_session("telegram", "user123")

        debate_id = await manager.get_active_debate(session.session_id)

        assert debate_id is None


class TestDebateSessionSingleton:
    """Tests for the singleton pattern."""

    def test_get_debate_session_manager_singleton(self):
        """Should return the same instance."""
        from aragora.connectors.debate_session import get_debate_session_manager

        # Reset singleton for test
        import aragora.connectors.debate_session as module

        module._manager = None

        manager1 = get_debate_session_manager()
        manager2 = get_debate_session_manager()

        assert manager1 is manager2
