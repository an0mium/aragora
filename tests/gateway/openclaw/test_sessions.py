"""
Tests for OpenClaw session management.

Tests cover:
- Session creation (with/without RBAC, with/without tenant)
- Session retrieval (active, expired, not found)
- Session update (metadata, context, activity)
- Session close (with reason, not found)
- Session listing (with filters)
- Session callbacks
- Session timeout handling
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.gateway.openclaw.adapter import (
    ActionStatus,
    OpenClawAdapter,
    OpenClawChannel,
    OpenClawSession,
    SessionState,
)
from aragora.gateway.openclaw.protocol import AuthorizationContext


# ============================================================================
# Fixtures
# ============================================================================


class MockRBACChecker:
    """Mock RBAC checker for permission tests."""

    def __init__(self, allowed: bool = True):
        self._allowed = allowed
        self.checked_permissions: list[tuple[str, str]] = []

    def check_permission(self, actor_id: str, permission: str, resource_id=None) -> bool:
        self.checked_permissions.append((actor_id, permission))
        return self._allowed

    async def check_permission_async(
        self, actor_id: str, permission: str, resource_id=None
    ) -> bool:
        self.checked_permissions.append((actor_id, permission))
        return self._allowed


@pytest.fixture
def adapter():
    """Create a basic adapter without RBAC."""
    return OpenClawAdapter(
        openclaw_endpoint="http://test:8081",
        session_timeout_seconds=3600,
    )


@pytest.fixture
def rbac_adapter():
    """Create adapter with permissive RBAC."""
    checker = MockRBACChecker(allowed=True)
    return OpenClawAdapter(
        openclaw_endpoint="http://test:8081",
        rbac_checker=checker,
    )


@pytest.fixture
def rbac_denied_adapter():
    """Create adapter with denying RBAC."""
    checker = MockRBACChecker(allowed=False)
    return OpenClawAdapter(
        openclaw_endpoint="http://test:8081",
        rbac_checker=checker,
    )


# ============================================================================
# Session Creation
# ============================================================================


class TestCreateSession:
    """Test session creation."""

    @pytest.mark.asyncio
    async def test_create_basic_session(self, adapter):
        """Test creating a basic session."""
        session = await adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.TELEGRAM,
        )

        assert session.user_id == "user-1"
        assert session.channel == OpenClawChannel.TELEGRAM
        assert session.state == SessionState.ACTIVE
        assert session.session_id.startswith("oc-sess-")
        assert session.tenant_id is None

    @pytest.mark.asyncio
    async def test_create_session_with_tenant(self, adapter):
        """Test creating a session with tenant ID."""
        session = await adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.SLACK,
            tenant_id="tenant-42",
        )

        assert session.tenant_id == "tenant-42"
        assert session.channel == OpenClawChannel.SLACK

    @pytest.mark.asyncio
    async def test_create_session_with_metadata(self, adapter):
        """Test creating a session with custom metadata."""
        session = await adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
            metadata={"lang": "en", "theme": "dark"},
        )

        assert session.metadata["lang"] == "en"
        assert session.metadata["theme"] == "dark"

    @pytest.mark.asyncio
    async def test_create_session_string_channel(self, adapter):
        """Test creating a session with string channel."""
        session = await adapter.create_session(
            user_id="user-1",
            channel="custom_channel",
        )

        assert session.channel == "custom_channel"

    @pytest.mark.asyncio
    async def test_create_session_sets_expiration(self, adapter):
        """Test that session has an expiration time."""
        session = await adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
        )

        assert session.expires_at is not None
        # Expiration should be ~1 hour from now
        expected = datetime.now(timezone.utc) + timedelta(seconds=3600)
        assert abs((session.expires_at - expected).total_seconds()) < 5

    @pytest.mark.asyncio
    async def test_create_session_no_timeout(self):
        """Test creating session when timeout is 0 (no expiry)."""
        adapter = OpenClawAdapter(
            openclaw_endpoint="http://test:8081",
            session_timeout_seconds=0,
        )
        session = await adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
        )

        assert session.expires_at is None

    @pytest.mark.asyncio
    async def test_create_session_unique_ids(self, adapter):
        """Test that each session gets a unique ID."""
        s1 = await adapter.create_session(user_id="user-1", channel=OpenClawChannel.WEB)
        s2 = await adapter.create_session(user_id="user-1", channel=OpenClawChannel.WEB)

        assert s1.session_id != s2.session_id

    @pytest.mark.asyncio
    async def test_create_session_with_rbac_allowed(self, rbac_adapter):
        """Test creating session when RBAC allows it."""
        auth = AuthorizationContext(actor_id="user-1")
        session = await rbac_adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
            auth_context=auth,
        )

        assert session.state == SessionState.ACTIVE

    @pytest.mark.asyncio
    async def test_create_session_with_rbac_denied(self, rbac_denied_adapter):
        """Test creating session when RBAC denies it."""
        auth = AuthorizationContext(actor_id="user-1")
        with pytest.raises(PermissionError, match="openclaw.session.create"):
            await rbac_denied_adapter.create_session(
                user_id="user-1",
                channel=OpenClawChannel.WEB,
                auth_context=auth,
            )

    @pytest.mark.asyncio
    async def test_create_session_without_auth_context_skips_rbac(self, rbac_denied_adapter):
        """Test that RBAC is skipped when no auth context is provided."""
        # Even with a denying checker, no auth_context means no check
        session = await rbac_denied_adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
        )
        assert session.state == SessionState.ACTIVE


# ============================================================================
# Session Retrieval
# ============================================================================


class TestGetSession:
    """Test session retrieval."""

    @pytest.mark.asyncio
    async def test_get_existing_session(self, adapter):
        """Test retrieving an existing session."""
        created = await adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
        )

        retrieved = await adapter.get_session(created.session_id)
        assert retrieved is not None
        assert retrieved.session_id == created.session_id
        assert retrieved.user_id == "user-1"

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, adapter):
        """Test retrieving a session that doesn't exist."""
        result = await adapter.get_session("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_expired_session_marks_state(self):
        """Test that getting an expired session marks it expired."""
        adapter = OpenClawAdapter(
            openclaw_endpoint="http://test:8081",
            session_timeout_seconds=1,
        )
        session = await adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
        )

        # Manually set expiry in the past
        session.expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)

        retrieved = await adapter.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.state == SessionState.EXPIRED


# ============================================================================
# Session Update
# ============================================================================


class TestUpdateSession:
    """Test session update."""

    @pytest.mark.asyncio
    async def test_update_metadata(self, adapter):
        """Test updating session metadata."""
        session = await adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
            metadata={"key1": "val1"},
        )

        updated = await adapter.update_session(
            session.session_id,
            metadata={"key2": "val2"},
        )

        assert updated is not None
        assert updated.metadata["key1"] == "val1"
        assert updated.metadata["key2"] == "val2"

    @pytest.mark.asyncio
    async def test_update_context(self, adapter):
        """Test updating session context."""
        session = await adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
        )

        updated = await adapter.update_session(
            session.session_id,
            context={"conversation_id": "conv-1"},
        )

        assert updated is not None
        assert updated.context["conversation_id"] == "conv-1"

    @pytest.mark.asyncio
    async def test_update_nonexistent_session(self, adapter):
        """Test updating a session that doesn't exist."""
        result = await adapter.update_session(
            "nonexistent",
            metadata={"key": "val"},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_update_refreshes_activity(self, adapter):
        """Test that update refreshes the last_activity timestamp."""
        session = await adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
        )
        original_activity = session.last_activity

        # Small delay to ensure timestamp differs
        await asyncio.sleep(0.01)

        updated = await adapter.update_session(
            session.session_id,
            metadata={"updated": True},
        )

        assert updated.last_activity >= original_activity


# ============================================================================
# Session Close
# ============================================================================


class TestCloseSession:
    """Test session closing."""

    @pytest.mark.asyncio
    async def test_close_session(self, adapter):
        """Test closing a session."""
        session = await adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
        )

        closed = await adapter.close_session(session.session_id)
        assert closed is not None
        assert closed.state == SessionState.TERMINATED
        assert closed.metadata["close_reason"] == "user_closed"

    @pytest.mark.asyncio
    async def test_close_session_with_reason(self, adapter):
        """Test closing a session with custom reason."""
        session = await adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
        )

        closed = await adapter.close_session(session.session_id, reason="idle_timeout")
        assert closed.metadata["close_reason"] == "idle_timeout"

    @pytest.mark.asyncio
    async def test_close_nonexistent_session(self, adapter):
        """Test closing a session that doesn't exist."""
        result = await adapter.close_session("nonexistent")
        assert result is None


# ============================================================================
# Session Listing
# ============================================================================


class TestListSessions:
    """Test session listing with filters."""

    @pytest.mark.asyncio
    async def test_list_all_sessions(self, adapter):
        """Test listing all sessions."""
        await adapter.create_session(user_id="user-1", channel=OpenClawChannel.WEB)
        await adapter.create_session(user_id="user-2", channel=OpenClawChannel.TELEGRAM)
        await adapter.create_session(user_id="user-3", channel=OpenClawChannel.SLACK)

        sessions = await adapter.list_sessions()
        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_list_sessions_filter_by_user(self, adapter):
        """Test filtering sessions by user."""
        await adapter.create_session(user_id="user-1", channel=OpenClawChannel.WEB)
        await adapter.create_session(user_id="user-1", channel=OpenClawChannel.TELEGRAM)
        await adapter.create_session(user_id="user-2", channel=OpenClawChannel.SLACK)

        sessions = await adapter.list_sessions(user_id="user-1")
        assert len(sessions) == 2
        assert all(s.user_id == "user-1" for s in sessions)

    @pytest.mark.asyncio
    async def test_list_sessions_filter_by_tenant(self, adapter):
        """Test filtering sessions by tenant."""
        await adapter.create_session(user_id="user-1", channel=OpenClawChannel.WEB, tenant_id="t1")
        await adapter.create_session(user_id="user-2", channel=OpenClawChannel.WEB, tenant_id="t2")

        sessions = await adapter.list_sessions(tenant_id="t1")
        assert len(sessions) == 1
        assert sessions[0].tenant_id == "t1"

    @pytest.mark.asyncio
    async def test_list_sessions_filter_by_channel(self, adapter):
        """Test filtering sessions by channel."""
        await adapter.create_session(user_id="user-1", channel=OpenClawChannel.WEB)
        await adapter.create_session(user_id="user-2", channel=OpenClawChannel.TELEGRAM)
        await adapter.create_session(user_id="user-3", channel=OpenClawChannel.TELEGRAM)

        sessions = await adapter.list_sessions(channel=OpenClawChannel.TELEGRAM)
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_filter_by_state(self, adapter):
        """Test filtering sessions by state."""
        s1 = await adapter.create_session(user_id="user-1", channel=OpenClawChannel.WEB)
        await adapter.create_session(user_id="user-2", channel=OpenClawChannel.WEB)
        await adapter.close_session(s1.session_id)

        active = await adapter.list_sessions(state=SessionState.ACTIVE)
        terminated = await adapter.list_sessions(state=SessionState.TERMINATED)

        assert len(active) == 1
        assert len(terminated) == 1

    @pytest.mark.asyncio
    async def test_list_sessions_combined_filters(self, adapter):
        """Test combining multiple filters."""
        await adapter.create_session(user_id="user-1", channel=OpenClawChannel.WEB, tenant_id="t1")
        await adapter.create_session(
            user_id="user-1", channel=OpenClawChannel.TELEGRAM, tenant_id="t1"
        )
        await adapter.create_session(user_id="user-2", channel=OpenClawChannel.WEB, tenant_id="t1")

        sessions = await adapter.list_sessions(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
        )
        assert len(sessions) == 1

    @pytest.mark.asyncio
    async def test_list_sessions_no_match(self, adapter):
        """Test listing when no sessions match."""
        await adapter.create_session(user_id="user-1", channel=OpenClawChannel.WEB)

        sessions = await adapter.list_sessions(user_id="nonexistent")
        assert len(sessions) == 0


# ============================================================================
# Session Callbacks
# ============================================================================


class TestSessionCallbacks:
    """Test session lifecycle callbacks."""

    @pytest.mark.asyncio
    async def test_create_triggers_callback(self, adapter):
        """Test that session creation triggers callbacks."""
        events = []

        def on_session(session, event_type):
            events.append((session.session_id, event_type))

        adapter.add_session_callback(on_session)

        session = await adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
        )

        assert len(events) == 1
        assert events[0] == (session.session_id, "created")

    @pytest.mark.asyncio
    async def test_close_triggers_callback(self, adapter):
        """Test that session close triggers callbacks."""
        events = []

        def on_session(session, event_type):
            events.append((session.session_id, event_type))

        adapter.add_session_callback(on_session)

        session = await adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
        )
        await adapter.close_session(session.session_id)

        assert len(events) == 2
        assert events[1] == (session.session_id, "closed")

    @pytest.mark.asyncio
    async def test_async_callback(self, adapter):
        """Test that async callbacks work."""
        events = []

        async def on_session(session, event_type):
            events.append(event_type)

        adapter.add_session_callback(on_session)

        await adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
        )

        assert events == ["created"]

    @pytest.mark.asyncio
    async def test_callback_error_does_not_propagate(self, adapter):
        """Test that callback errors don't break session creation."""

        def bad_callback(session, event_type):
            raise RuntimeError("callback error")

        adapter.add_session_callback(bad_callback)

        # Should not raise
        session = await adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
        )
        assert session.state == SessionState.ACTIVE
