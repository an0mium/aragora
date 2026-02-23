"""
Tests for social collaboration handler.

Tests cover both handler classes:
- CollaborationHandlers: Async methods for session management, participant tracking,
  presence, typing indicators, role changes, approvals, and stats.
- CollaborationHandler: Sync HTTP handler for social collaboration sessions with
  CRUD endpoints, participant management, and messaging.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers.social.collaboration import (
    CollaborationHandler,
    CollaborationHandlers,
    SocialCollaborationSession,
    SocialCollaborationStore,
    _check_permission,
)
from aragora.server.collaboration import (
    CollaborationSession,
    Participant,
    ParticipantRole,
    SessionManager,
    SessionState,
)


# ============================================================================
# Helpers
# ============================================================================


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status", result.get("status_code", 200))
    return result.status_code


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _bypass_collab_rbac(request, monkeypatch):
    """Bypass the inline _check_permission used by CollaborationHandlers.

    The conftest auto-patches RBAC decorators and SecureHandler, but
    CollaborationHandlers calls _check_permission() directly (which calls
    get_permission_checker). We patch _check_permission to return None
    (allowed) by default. Tests marked with no_auto_auth opt out.
    """
    if "no_auto_auth" in [m.name for m in request.node.iter_markers()]:
        yield
        return
    monkeypatch.setattr(
        "aragora.server.handlers.social.collaboration._check_permission",
        lambda user_id, permission, org_id=None, roles=None: None,
    )
    yield


@pytest.fixture
def session_manager():
    """Create a fresh SessionManager for testing."""
    return SessionManager()


@pytest.fixture
def collab_handlers(session_manager):
    """Create CollaborationHandlers with a test SessionManager."""
    return CollaborationHandlers(manager=session_manager)


@pytest.fixture
def social_store():
    """Create a fresh SocialCollaborationStore for testing."""
    return SocialCollaborationStore()


@pytest.fixture
def handler(social_store):
    """Create CollaborationHandler with a test store."""
    return CollaborationHandler(server_context={"collaboration_store": social_store})


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler for the sync CollaborationHandler."""
    mock = MagicMock()
    mock.command = "GET"
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = {"Content-Type": "application/json", "Content-Length": "2"}
    mock.rfile = MagicMock()
    mock.rfile.read.return_value = b"{}"
    return mock


def _set_body(mock_http_handler, data: dict) -> None:
    """Set the JSON body on a mock HTTP handler."""
    body_bytes = json.dumps(data).encode("utf-8")
    mock_http_handler.rfile.read.return_value = body_bytes
    mock_http_handler.headers["Content-Length"] = str(len(body_bytes))


def _create_social_session(store, **overrides) -> SocialCollaborationSession:
    """Create and store a social collaboration session."""
    defaults = {
        "id": "test-session-1",
        "org_id": "",
        "name": "Test Session",
        "description": "A test session",
        "channel_id": "ch-1",
        "platform": "slack",
        "created_by": "user-1",
        "participants": ["user-1"],
    }
    defaults.update(overrides)
    session = SocialCollaborationSession(**defaults)
    store.create(session)
    return session


# ============================================================================
# CollaborationHandlers (async) tests
# ============================================================================


class TestCheckPermission:
    """Tests for the _check_permission helper."""

    @pytest.mark.no_auto_auth
    def test_permission_allowed(self):
        """Permission check returns None when allowed."""
        with patch(
            "aragora.server.handlers.social.collaboration.get_permission_checker"
        ) as mock_checker:
            decision = MagicMock()
            decision.allowed = True
            mock_checker.return_value.check_permission.return_value = decision

            result = _check_permission("user-1", "collaboration:create")
            assert result is None

    @pytest.mark.no_auto_auth
    def test_permission_denied(self):
        """Permission check returns error dict when denied."""
        with patch(
            "aragora.server.handlers.social.collaboration.get_permission_checker"
        ) as mock_checker:
            decision = MagicMock()
            decision.allowed = False
            decision.reason = "Denied"
            mock_checker.return_value.check_permission.return_value = decision

            result = _check_permission("user-1", "collaboration:create")
            assert result is not None
            assert result.get("status") == 403

    @pytest.mark.no_auto_auth
    def test_permission_check_error(self):
        """Permission check returns 500 on exception."""
        with patch(
            "aragora.server.handlers.social.collaboration.get_permission_checker"
        ) as mock_checker:
            mock_checker.side_effect = RuntimeError("RBAC unavailable")
            result = _check_permission("user-1", "collaboration:create")
            assert result is not None
            assert result.get("status") == 500

    @pytest.mark.no_auto_auth
    def test_permission_uses_default_roles(self):
        """Permission check uses member role by default."""
        with patch(
            "aragora.server.handlers.social.collaboration.get_permission_checker"
        ) as mock_checker:
            decision = MagicMock()
            decision.allowed = True
            mock_checker.return_value.check_permission.return_value = decision

            _check_permission("user-1", "collaboration:read")
            call_args = mock_checker.return_value.check_permission.call_args
            context = call_args[0][0]
            assert "member" in context.roles

    @pytest.mark.no_auto_auth
    def test_permission_uses_provided_roles(self):
        """Permission check uses provided roles."""
        with patch(
            "aragora.server.handlers.social.collaboration.get_permission_checker"
        ) as mock_checker:
            decision = MagicMock()
            decision.allowed = True
            mock_checker.return_value.check_permission.return_value = decision

            _check_permission("user-1", "collaboration:admin", roles={"admin", "owner"})
            call_args = mock_checker.return_value.check_permission.call_args
            context = call_args[0][0]
            assert "admin" in context.roles

    @pytest.mark.no_auto_auth
    def test_permission_passes_org_id(self):
        """Permission check passes org_id to context."""
        with patch(
            "aragora.server.handlers.social.collaboration.get_permission_checker"
        ) as mock_checker:
            decision = MagicMock()
            decision.allowed = True
            mock_checker.return_value.check_permission.return_value = decision

            _check_permission("user-1", "collaboration:create", org_id="org-42")
            call_args = mock_checker.return_value.check_permission.call_args
            context = call_args[0][0]
            assert context.org_id == "org-42"


class TestCreateSession:
    """Tests for CollaborationHandlers.create_session."""

    @pytest.mark.asyncio
    async def test_create_session_success(self, collab_handlers):
        """Successfully create a collaboration session."""
        result = await collab_handlers.create_session(
            debate_id="debate-1",
            user_id="user-1",
            title="Test Debate",
            is_public=True,
        )
        assert result["success"] is True
        assert "session" in result
        assert result["session"]["debate_id"] == "debate-1"
        assert result["session"]["is_public"] is True

    @pytest.mark.asyncio
    async def test_create_session_missing_debate_id(self, collab_handlers):
        """Fail when debate_id is missing."""
        result = await collab_handlers.create_session(debate_id="", user_id="user-1")
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    @pytest.mark.asyncio
    async def test_create_session_missing_user_id(self, collab_handlers):
        """Fail when user_id is missing."""
        result = await collab_handlers.create_session(debate_id="debate-1", user_id="")
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    @pytest.mark.asyncio
    async def test_create_session_with_all_options(self, collab_handlers):
        """Create session with all optional parameters."""
        result = await collab_handlers.create_session(
            debate_id="debate-1",
            user_id="user-1",
            title="Full Session",
            description="A full session",
            is_public=True,
            max_participants=10,
            org_id="org-1",
            expires_in=3600.0,
            allow_anonymous=True,
            require_approval=True,
        )
        assert result["success"] is True
        session = result["session"]
        assert session["description"] == "A full session"
        assert session["max_participants"] == 10
        assert session["org_id"] == "org-1"
        assert session["allow_anonymous"] is True
        assert session["require_approval"] is True

    @pytest.mark.asyncio
    async def test_create_session_manager_error(self, collab_handlers):
        """Create session returns error when manager raises."""
        collab_handlers.manager.create_session = MagicMock(side_effect=RuntimeError("DB error"))
        result = await collab_handlers.create_session(debate_id="debate-1", user_id="user-1")
        assert "error" in result
        assert result["code"] == "INTERNAL_ERROR"

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_create_session_rbac_denied(self, collab_handlers):
        """Create session fails when RBAC denies permission."""
        with patch(
            "aragora.server.handlers.social.collaboration.get_permission_checker"
        ) as mock_checker:
            decision = MagicMock()
            decision.allowed = False
            decision.reason = "No permission"
            mock_checker.return_value.check_permission.return_value = decision

            result = await collab_handlers.create_session(debate_id="debate-1", user_id="user-1")
            assert result.get("status") == 403


class TestGetSession:
    """Tests for CollaborationHandlers.get_session."""

    @pytest.mark.asyncio
    async def test_get_session_success(self, collab_handlers, session_manager):
        """Get an existing session by ID."""
        session = session_manager.create_session(debate_id="debate-1", created_by="user-1")
        result = await collab_handlers.get_session(session.session_id)
        assert "session" in result
        assert result["session"]["debate_id"] == "debate-1"

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, collab_handlers):
        """Return 404 when session not found."""
        result = await collab_handlers.get_session("nonexistent-id")
        assert "error" in result
        assert result["status"] == 404

    @pytest.mark.asyncio
    async def test_get_session_empty_id(self, collab_handlers):
        """Fail when session_id is empty."""
        result = await collab_handlers.get_session("")
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    @pytest.mark.asyncio
    async def test_get_session_with_user_id(self, collab_handlers, session_manager):
        """Get session with user_id triggers RBAC check."""
        session = session_manager.create_session(debate_id="debate-1", created_by="user-1")
        result = await collab_handlers.get_session(session.session_id, user_id="user-1")
        assert "session" in result


class TestListSessions:
    """Tests for CollaborationHandlers.list_sessions."""

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, collab_handlers):
        """List sessions returns empty list when no sessions exist."""
        result = await collab_handlers.list_sessions()
        assert result["sessions"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_list_sessions_by_debate(self, collab_handlers, session_manager):
        """List sessions filtered by debate_id."""
        session_manager.create_session(debate_id="debate-1", created_by="user-1")
        session_manager.create_session(debate_id="debate-2", created_by="user-1")
        result = await collab_handlers.list_sessions(debate_id="debate-1")
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_list_sessions_by_user(self, collab_handlers, session_manager):
        """List sessions filtered by user_id."""
        session_manager.create_session(debate_id="debate-1", created_by="user-1")
        result = await collab_handlers.list_sessions(user_id="user-1")
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_list_sessions_excludes_closed(self, collab_handlers, session_manager):
        """Closed sessions excluded by default."""
        session = session_manager.create_session(debate_id="debate-1", created_by="user-1")
        session_manager.close_session(session.session_id, "user-1")
        result = await collab_handlers.list_sessions()
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_list_sessions_include_closed(self, collab_handlers, session_manager):
        """Include closed sessions when requested."""
        session = session_manager.create_session(debate_id="debate-1", created_by="user-1")
        session_manager.close_session(session.session_id, "user-1")
        result = await collab_handlers.list_sessions(include_closed=True)
        assert result["count"] == 1


class TestJoinSession:
    """Tests for CollaborationHandlers.join_session."""

    @pytest.mark.asyncio
    async def test_join_session_success(self, collab_handlers, session_manager):
        """Successfully join a session."""
        session = session_manager.create_session(debate_id="debate-1", created_by="user-1")
        result = await collab_handlers.join_session(
            session_id=session.session_id,
            user_id="user-2",
            role="voter",
            display_name="User 2",
        )
        assert result["success"] is True
        assert result["participant"]["user_id"] == "user-2"

    @pytest.mark.asyncio
    async def test_join_session_missing_session_id(self, collab_handlers):
        """Fail when session_id is empty."""
        result = await collab_handlers.join_session(session_id="", user_id="user-1")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_join_session_missing_user_id(self, collab_handlers):
        """Fail when user_id is empty."""
        result = await collab_handlers.join_session(session_id="s-1", user_id="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_join_session_invalid_role(self, collab_handlers, session_manager):
        """Fail when role is invalid."""
        session = session_manager.create_session(debate_id="debate-1", created_by="user-1")
        result = await collab_handlers.join_session(
            session_id=session.session_id, user_id="user-2", role="invalid_role"
        )
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    @pytest.mark.asyncio
    async def test_join_session_not_found(self, collab_handlers):
        """Fail when session doesn't exist."""
        result = await collab_handlers.join_session(session_id="nonexistent", user_id="user-2")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_join_session_already_joined(self, collab_handlers, session_manager):
        """Rejoin returns success with existing participant."""
        session = session_manager.create_session(debate_id="debate-1", created_by="user-1")
        result = await collab_handlers.join_session(session_id=session.session_id, user_id="user-1")
        assert result["success"] is True
        assert "Already in session" in result["message"]


class TestLeaveSession:
    """Tests for CollaborationHandlers.leave_session."""

    @pytest.mark.asyncio
    async def test_leave_session_success(self, collab_handlers, session_manager):
        """Successfully leave a session."""
        session = session_manager.create_session(debate_id="debate-1", created_by="user-1")
        session_manager.join_session(session.session_id, "user-2")
        result = await collab_handlers.leave_session(session.session_id, "user-2")
        assert result["success"] is True
        assert "Left session" in result["message"]

    @pytest.mark.asyncio
    async def test_leave_session_not_found(self, collab_handlers):
        """Leave fails when session doesn't exist."""
        result = await collab_handlers.leave_session("nonexistent", "user-1")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_leave_session_missing_session_id(self, collab_handlers):
        """Fail when session_id is empty."""
        result = await collab_handlers.leave_session("", "user-1")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_leave_session_missing_user_id(self, collab_handlers):
        """Fail when user_id is empty."""
        result = await collab_handlers.leave_session("s-1", "")
        assert "error" in result


class TestUpdatePresence:
    """Tests for CollaborationHandlers.update_presence."""

    @pytest.mark.asyncio
    async def test_update_presence_online(self, collab_handlers, session_manager):
        """Update presence to online."""
        session = session_manager.create_session(debate_id="debate-1", created_by="user-1")
        result = await collab_handlers.update_presence(session.session_id, "user-1", is_online=True)
        assert result["success"] is True
        assert "Presence updated" in result["message"]

    @pytest.mark.asyncio
    async def test_update_presence_offline(self, collab_handlers, session_manager):
        """Update presence to offline."""
        session = session_manager.create_session(debate_id="debate-1", created_by="user-1")
        result = await collab_handlers.update_presence(
            session.session_id, "user-1", is_online=False
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_update_presence_not_found(self, collab_handlers):
        """Fail when session doesn't exist."""
        result = await collab_handlers.update_presence("nonexistent", "user-1")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_update_presence_missing_session_id(self, collab_handlers):
        """Fail when session_id is empty."""
        result = await collab_handlers.update_presence("", "user-1")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_update_presence_missing_user_id(self, collab_handlers):
        """Fail when user_id is empty."""
        result = await collab_handlers.update_presence("s-1", "")
        assert "error" in result


class TestSetTyping:
    """Tests for CollaborationHandlers.set_typing."""

    @pytest.mark.asyncio
    async def test_set_typing_start(self, collab_handlers, session_manager):
        """Set typing indicator to true."""
        session = session_manager.create_session(debate_id="debate-1", created_by="user-1")
        result = await collab_handlers.set_typing(
            session.session_id, "user-1", is_typing=True, context="vote"
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_set_typing_stop(self, collab_handlers, session_manager):
        """Set typing indicator to false."""
        session = session_manager.create_session(debate_id="debate-1", created_by="user-1")
        result = await collab_handlers.set_typing(session.session_id, "user-1", is_typing=False)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_set_typing_not_found(self, collab_handlers):
        """Fail when session doesn't exist."""
        result = await collab_handlers.set_typing("nonexistent", "user-1")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_set_typing_missing_session_id(self, collab_handlers):
        """Fail when session_id is empty."""
        result = await collab_handlers.set_typing("", "user-1")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_set_typing_missing_user_id(self, collab_handlers):
        """Fail when user_id is empty."""
        result = await collab_handlers.set_typing("s-1", "")
        assert "error" in result


class TestChangeRole:
    """Tests for CollaborationHandlers.change_role."""

    @pytest.mark.asyncio
    async def test_change_role_success(self, collab_handlers, session_manager):
        """Successfully change a participant's role."""
        session = session_manager.create_session(debate_id="debate-1", created_by="user-1")
        session_manager.join_session(session.session_id, "user-2", role=ParticipantRole.VOTER)
        result = await collab_handlers.change_role(
            session_id=session.session_id,
            target_user_id="user-2",
            new_role="contributor",
            changed_by="user-1",
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_change_role_missing_session_id(self, collab_handlers):
        """Fail when session_id is empty."""
        result = await collab_handlers.change_role("", "user-2", "voter", "user-1")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_change_role_missing_target(self, collab_handlers):
        """Fail when target_user_id is empty."""
        result = await collab_handlers.change_role("s-1", "", "voter", "user-1")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_change_role_missing_changed_by(self, collab_handlers):
        """Fail when changed_by is empty."""
        result = await collab_handlers.change_role("s-1", "user-2", "voter", "")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_change_role_invalid_role(self, collab_handlers, session_manager):
        """Fail with invalid role string."""
        session = session_manager.create_session(debate_id="debate-1", created_by="user-1")
        result = await collab_handlers.change_role(
            session.session_id, "user-2", "emperor", "user-1"
        )
        assert "error" in result
        assert result["code"] == "VALIDATION_ERROR"

    @pytest.mark.asyncio
    async def test_change_role_not_moderator(self, collab_handlers, session_manager):
        """Fail when changer is not a moderator."""
        session = session_manager.create_session(debate_id="debate-1", created_by="user-1")
        session_manager.join_session(session.session_id, "user-2", role=ParticipantRole.VOTER)
        session_manager.join_session(session.session_id, "user-3", role=ParticipantRole.VOTER)
        result = await collab_handlers.change_role(
            session.session_id, "user-3", "contributor", "user-2"
        )
        assert result["success"] is False


class TestApproveJoin:
    """Tests for CollaborationHandlers.approve_join."""

    @pytest.mark.asyncio
    async def test_approve_join_calls_manager(self, collab_handlers, session_manager):
        """Approve join delegates to session manager."""
        session = session_manager.create_session(
            debate_id="debate-1", created_by="user-1", require_approval=True
        )
        # Trigger approval request
        session_manager.join_session(session.session_id, "user-2")
        # Mock the manager to test the handler delegates properly
        collab_handlers.manager.approve_join = MagicMock(return_value=(True, "Approved"))
        result = await collab_handlers.approve_join(
            session_id=session.session_id,
            user_id="user-2",
            approved_by="user-1",
            approved=True,
        )
        assert result["success"] is True
        assert result["message"] == "Approved"

    @pytest.mark.asyncio
    async def test_deny_join(self, collab_handlers, session_manager):
        """Successfully deny a join request."""
        session = session_manager.create_session(
            debate_id="debate-1", created_by="user-1", require_approval=True
        )
        session_manager.join_session(session.session_id, "user-2")
        result = await collab_handlers.approve_join(
            session_id=session.session_id,
            user_id="user-2",
            approved_by="user-1",
            approved=False,
        )
        assert result["success"] is True
        assert "denied" in result["message"]

    @pytest.mark.asyncio
    async def test_approve_join_missing_session_id(self, collab_handlers):
        """Fail when session_id is empty."""
        result = await collab_handlers.approve_join("", "user-2", "user-1")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_approve_join_missing_user_id(self, collab_handlers):
        """Fail when user_id is empty."""
        result = await collab_handlers.approve_join("s-1", "", "user-1")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_approve_join_missing_approved_by(self, collab_handlers):
        """Fail when approved_by is empty."""
        result = await collab_handlers.approve_join("s-1", "user-2", "")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_approve_join_invalid_role_defaults_to_voter(
        self, collab_handlers, session_manager
    ):
        """Invalid role string defaults to voter (ParticipantRole.VOTER)."""
        session = session_manager.create_session(
            debate_id="debate-1", created_by="user-1", require_approval=True
        )
        session_manager.join_session(session.session_id, "user-2")
        # Mock the manager to verify role value
        collab_handlers.manager.approve_join = MagicMock(
            return_value=(True, "Approved with default role")
        )
        result = await collab_handlers.approve_join(
            session_id=session.session_id,
            user_id="user-2",
            approved_by="user-1",
            approved=True,
            role="supreme_leader",
        )
        assert result["success"] is True
        # Verify voter was passed as role after invalid role defaulted
        call_kwargs = collab_handlers.manager.approve_join.call_args
        assert call_kwargs.kwargs.get("role") == ParticipantRole.VOTER or (
            len(call_kwargs.args) >= 5 and call_kwargs.args[4] == ParticipantRole.VOTER
        )

    @pytest.mark.asyncio
    async def test_approve_join_session_not_found(self, collab_handlers):
        """Fail when session doesn't exist in manager."""
        collab_handlers.manager.approve_join = MagicMock(return_value=(False, "Session not found"))
        result = await collab_handlers.approve_join(
            session_id="nonexistent",
            user_id="user-2",
            approved_by="user-1",
        )
        assert result["success"] is False


class TestCloseSession:
    """Tests for CollaborationHandlers.close_session."""

    @pytest.mark.asyncio
    async def test_close_session_success(self, collab_handlers, session_manager):
        """Successfully close a session."""
        session = session_manager.create_session(debate_id="debate-1", created_by="user-1")
        result = await collab_handlers.close_session(session.session_id, "user-1")
        assert result["success"] is True
        assert "Session closed" in result["message"]

    @pytest.mark.asyncio
    async def test_close_session_not_found(self, collab_handlers):
        """Fail when session doesn't exist."""
        result = await collab_handlers.close_session("nonexistent", "user-1")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_close_session_missing_session_id(self, collab_handlers):
        """Fail when session_id is empty."""
        result = await collab_handlers.close_session("", "user-1")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_close_session_missing_closed_by(self, collab_handlers):
        """Fail when closed_by is empty."""
        result = await collab_handlers.close_session("s-1", "")
        assert "error" in result


class TestGetStats:
    """Tests for CollaborationHandlers.get_stats."""

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, collab_handlers):
        """Get stats when no sessions exist."""
        result = await collab_handlers.get_stats()
        assert "total_sessions" in result
        assert result["total_sessions"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_sessions(self, collab_handlers, session_manager):
        """Get stats with active sessions."""
        session_manager.create_session(debate_id="debate-1", created_by="user-1")
        session_manager.create_session(debate_id="debate-2", created_by="user-2")
        result = await collab_handlers.get_stats()
        assert result["total_sessions"] == 2
        assert result["active_sessions"] == 2


class TestGetParticipants:
    """Tests for CollaborationHandlers.get_participants."""

    @pytest.mark.asyncio
    async def test_get_participants_success(self, collab_handlers, session_manager):
        """Get participants for a session."""
        session = session_manager.create_session(debate_id="debate-1", created_by="user-1")
        session_manager.join_session(session.session_id, "user-2")
        result = await collab_handlers.get_participants(session.session_id)
        assert result["count"] == 2
        assert len(result["participants"]) == 2

    @pytest.mark.asyncio
    async def test_get_participants_not_found(self, collab_handlers):
        """Return 404 when session doesn't exist."""
        result = await collab_handlers.get_participants("nonexistent")
        assert "error" in result
        assert result["status"] == 404

    @pytest.mark.asyncio
    async def test_get_participants_empty_id(self, collab_handlers):
        """Fail when session_id is empty."""
        result = await collab_handlers.get_participants("")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_participants_with_user_id(self, collab_handlers, session_manager):
        """Get participants with user_id triggers RBAC check."""
        session = session_manager.create_session(debate_id="debate-1", created_by="user-1")
        result = await collab_handlers.get_participants(session.session_id, user_id="user-1")
        assert "participants" in result

    @pytest.mark.asyncio
    async def test_get_participants_online_count(self, collab_handlers, session_manager):
        """Online count reflects presence status."""
        session = session_manager.create_session(debate_id="debate-1", created_by="user-1")
        session_manager.join_session(session.session_id, "user-2")
        session_manager.update_presence(session.session_id, "user-2", is_online=False)
        result = await collab_handlers.get_participants(session.session_id)
        assert result["count"] == 2
        assert result["online_count"] == 1


# ============================================================================
# CollaborationHandler (sync) tests
# ============================================================================


class TestHandlerCanHandle:
    """Tests for CollaborationHandler.can_handle."""

    def test_can_handle_sessions_path(self, handler):
        """Handler can handle the sessions path."""
        assert handler.can_handle("/api/v1/social/collaboration/sessions")

    def test_can_handle_sessions_subpath(self, handler):
        """Handler can handle session subpaths."""
        assert handler.can_handle("/api/v1/social/collaboration/sessions/abc/participants")

    def test_cannot_handle_other_path(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/v1/other/path")


class TestHandlerRouting:
    """Tests for CollaborationHandler.handle routing."""

    def test_not_found_for_unknown_subpath(self, handler, mock_http_handler):
        """Return 404 for unrecognized paths."""
        result = handler.handle(
            "/api/v1/social/collaboration/unknown", {}, mock_http_handler, "GET"
        )
        assert _status(result) == 404

    def test_method_not_allowed_sessions_root(self, handler, mock_http_handler):
        """Return 405 for disallowed method on sessions root."""
        mock_http_handler.command = "DELETE"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions", {}, mock_http_handler, "DELETE"
        )
        assert _status(result) == 405

    def test_method_not_allowed_session_detail(self, handler, mock_http_handler, social_store):
        """Return 405 for disallowed method on session detail."""
        _create_social_session(social_store)
        mock_http_handler.command = "POST"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/test-session-1",
            {},
            mock_http_handler,
            "POST",
        )
        assert _status(result) == 405

    def test_method_not_allowed_participants(self, handler, mock_http_handler):
        """Return 405 for disallowed method on participants."""
        mock_http_handler.command = "PATCH"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/s1/participants",
            {},
            mock_http_handler,
            "PATCH",
        )
        assert _status(result) == 405

    def test_method_not_allowed_participant_detail(self, handler, mock_http_handler):
        """Return 405 for disallowed method on participant detail."""
        mock_http_handler.command = "POST"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/s1/participants/u1",
            {},
            mock_http_handler,
            "POST",
        )
        assert _status(result) == 405

    def test_method_not_allowed_messages(self, handler, mock_http_handler):
        """Return 405 for disallowed method on messages."""
        mock_http_handler.command = "DELETE"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/s1/messages",
            {},
            mock_http_handler,
            "DELETE",
        )
        assert _status(result) == 405

    def test_handler_uses_command_attribute(self, handler, mock_http_handler, social_store):
        """Handler reads method from handler.command when available."""
        _create_social_session(social_store)
        mock_http_handler.command = "GET"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/test-session-1",
            {},
            mock_http_handler,
            "POST",  # This should be overridden by handler.command
        )
        assert _status(result) == 200


class TestHandlerRateLimit:
    """Tests for rate limiting in CollaborationHandler."""

    def test_rate_limit_exceeded(self, handler, mock_http_handler):
        """Return 429 when rate limit is exceeded."""
        with patch("aragora.server.handlers.social.collaboration._collab_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = handler.handle(
                "/api/v1/social/collaboration/sessions",
                {},
                mock_http_handler,
                "GET",
            )
            assert _status(result) == 429


class TestHandlerListSessions:
    """Tests for listing sessions via CollaborationHandler."""

    def test_list_sessions_empty(self, handler, mock_http_handler):
        """List sessions returns empty list."""
        mock_http_handler.command = "GET"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions", {}, mock_http_handler, "GET"
        )
        body = _body(result)
        assert body["sessions"] == []
        assert body["total"] == 0

    def test_list_sessions_with_data(self, handler, mock_http_handler, social_store):
        """List sessions returns stored sessions."""
        _create_social_session(social_store, id="s1", name="Session 1")
        _create_social_session(social_store, id="s2", name="Session 2")
        mock_http_handler.command = "GET"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions", {}, mock_http_handler, "GET"
        )
        body = _body(result)
        assert body["total"] == 2


class TestHandlerCreateSession:
    """Tests for creating sessions via CollaborationHandler."""

    def test_create_session_success(self, handler, mock_http_handler):
        """Successfully create a session."""
        _set_body(
            mock_http_handler,
            {
                "name": "New Session",
                "channel_id": "ch-1",
                "platform": "slack",
                "created_by": "user-1",
            },
        )
        mock_http_handler.command = "POST"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions", {}, mock_http_handler, "POST"
        )
        assert _status(result) == 201
        body = _body(result)
        assert body["session"]["name"] == "New Session"
        assert body["session"]["channel_id"] == "ch-1"

    def test_create_session_missing_name(self, handler, mock_http_handler):
        """Fail when name is missing."""
        _set_body(mock_http_handler, {"channel_id": "ch-1"})
        mock_http_handler.command = "POST"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions", {}, mock_http_handler, "POST"
        )
        assert _status(result) == 400

    def test_create_session_missing_channel_id(self, handler, mock_http_handler):
        """Fail when channel_id is missing."""
        _set_body(mock_http_handler, {"name": "Test"})
        mock_http_handler.command = "POST"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions", {}, mock_http_handler, "POST"
        )
        assert _status(result) == 400

    def test_create_session_invalid_json(self, handler, mock_http_handler):
        """Fail on invalid JSON body."""
        mock_http_handler.rfile.read.return_value = b"not json"
        mock_http_handler.headers["Content-Length"] = "8"
        mock_http_handler.command = "POST"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions", {}, mock_http_handler, "POST"
        )
        assert _status(result) == 400

    def test_create_session_with_all_fields(self, handler, mock_http_handler):
        """Create session with all optional fields."""
        _set_body(
            mock_http_handler,
            {
                "name": "Full Session",
                "channel_id": "ch-2",
                "platform": "teams",
                "org_id": "org-1",
                "description": "Detailed session",
                "created_by": "admin-1",
                "participants": ["user-1", "user-2"],
            },
        )
        mock_http_handler.command = "POST"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions", {}, mock_http_handler, "POST"
        )
        assert _status(result) == 201
        body = _body(result)
        assert body["session"]["org_id"] == "org-1"
        assert body["session"]["description"] == "Detailed session"
        assert len(body["session"]["participants"]) == 2


class TestHandlerGetSession:
    """Tests for getting a session by ID via CollaborationHandler."""

    def test_get_session_success(self, handler, mock_http_handler, social_store):
        """Get existing session by ID."""
        _create_social_session(social_store)
        mock_http_handler.command = "GET"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/test-session-1",
            {},
            mock_http_handler,
            "GET",
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["session"]["id"] == "test-session-1"

    def test_get_session_not_found(self, handler, mock_http_handler):
        """Return 404 for non-existent session."""
        mock_http_handler.command = "GET"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/nonexistent",
            {},
            mock_http_handler,
            "GET",
        )
        assert _status(result) == 404


class TestHandlerUpdateSession:
    """Tests for updating a session via CollaborationHandler."""

    def test_update_session_success(self, handler, mock_http_handler, social_store):
        """Update a session's fields."""
        _create_social_session(social_store)
        _set_body(mock_http_handler, {"name": "Updated Name"})
        mock_http_handler.command = "PATCH"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/test-session-1",
            {},
            mock_http_handler,
            "PATCH",
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["session"]["name"] == "Updated Name"

    def test_update_session_not_found(self, handler, mock_http_handler):
        """Return 404 for non-existent session."""
        _set_body(mock_http_handler, {"name": "Updated"})
        mock_http_handler.command = "PATCH"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/nonexistent",
            {},
            mock_http_handler,
            "PATCH",
        )
        assert _status(result) == 404

    def test_update_session_invalid_json(self, handler, mock_http_handler, social_store):
        """Fail on invalid JSON body for update."""
        _create_social_session(social_store)
        mock_http_handler.rfile.read.return_value = b"{{bad json"
        mock_http_handler.headers["Content-Length"] = "10"
        mock_http_handler.command = "PATCH"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/test-session-1",
            {},
            mock_http_handler,
            "PATCH",
        )
        assert _status(result) == 400


class TestHandlerDeleteSession:
    """Tests for deleting a session via CollaborationHandler."""

    def test_delete_session_success(self, handler, mock_http_handler, social_store):
        """Delete an existing session."""
        _create_social_session(social_store)
        mock_http_handler.command = "DELETE"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/test-session-1",
            {},
            mock_http_handler,
            "DELETE",
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["deleted"] is True

    def test_delete_session_not_found(self, handler, mock_http_handler):
        """Return 404 for non-existent session."""
        mock_http_handler.command = "DELETE"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/nonexistent",
            {},
            mock_http_handler,
            "DELETE",
        )
        assert _status(result) == 404


class TestHandlerParticipants:
    """Tests for participant management via CollaborationHandler."""

    def test_list_participants(self, handler, mock_http_handler, social_store):
        """List participants of a session."""
        _create_social_session(social_store, participants=["user-1", "user-2"])
        mock_http_handler.command = "GET"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/test-session-1/participants",
            {},
            mock_http_handler,
            "GET",
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2

    def test_add_participant(self, handler, mock_http_handler, social_store):
        """Add a participant to a session."""
        _create_social_session(social_store)
        _set_body(mock_http_handler, {"user_id": "user-new"})
        mock_http_handler.command = "POST"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/test-session-1/participants",
            {},
            mock_http_handler,
            "POST",
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["added"] is True
        assert body["user_id"] == "user-new"

    def test_add_participant_missing_user_id(self, handler, mock_http_handler, social_store):
        """Fail when user_id is missing from add participant request."""
        _create_social_session(social_store)
        _set_body(mock_http_handler, {})
        mock_http_handler.command = "POST"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/test-session-1/participants",
            {},
            mock_http_handler,
            "POST",
        )
        assert _status(result) == 400

    def test_add_participant_session_not_found(self, handler, mock_http_handler):
        """Fail when adding to non-existent session."""
        _set_body(mock_http_handler, {"user_id": "user-new"})
        mock_http_handler.command = "POST"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/nonexistent/participants",
            {},
            mock_http_handler,
            "POST",
        )
        assert _status(result) == 404

    def test_remove_participant(self, handler, mock_http_handler, social_store):
        """Remove a participant from a session."""
        _create_social_session(social_store, participants=["user-1", "user-2"])
        mock_http_handler.command = "DELETE"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/test-session-1/participants/user-2",
            {},
            mock_http_handler,
            "DELETE",
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["removed"] is True

    def test_remove_participant_session_not_found(self, handler, mock_http_handler):
        """Fail when removing from non-existent session."""
        mock_http_handler.command = "DELETE"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/nonexistent/participants/user-1",
            {},
            mock_http_handler,
            "DELETE",
        )
        assert _status(result) == 404

    def test_add_participant_invalid_json(self, handler, mock_http_handler, social_store):
        """Fail on invalid JSON for add participant."""
        _create_social_session(social_store)
        mock_http_handler.rfile.read.return_value = b"not valid json"
        mock_http_handler.headers["Content-Length"] = "14"
        mock_http_handler.command = "POST"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/test-session-1/participants",
            {},
            mock_http_handler,
            "POST",
        )
        assert _status(result) == 400


class TestHandlerMessages:
    """Tests for message management via CollaborationHandler."""

    def test_list_messages_empty(self, handler, mock_http_handler, social_store):
        """List messages returns empty list for new session."""
        _create_social_session(social_store)
        mock_http_handler.command = "GET"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/test-session-1/messages",
            {},
            mock_http_handler,
            "GET",
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["messages"] == []
        assert body["total"] == 0

    def test_send_message_success(self, handler, mock_http_handler, social_store):
        """Send a message to a session."""
        _create_social_session(social_store)
        _set_body(mock_http_handler, {"content": "Hello, world!"})
        mock_http_handler.command = "POST"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/test-session-1/messages",
            {},
            mock_http_handler,
            "POST",
        )
        assert _status(result) == 201
        body = _body(result)
        assert body["message"]["content"] == "Hello, world!"
        assert "id" in body["message"]

    def test_send_message_missing_content(self, handler, mock_http_handler, social_store):
        """Fail when content is missing from message."""
        _create_social_session(social_store)
        _set_body(mock_http_handler, {})
        mock_http_handler.command = "POST"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/test-session-1/messages",
            {},
            mock_http_handler,
            "POST",
        )
        assert _status(result) == 400

    def test_send_message_session_not_found(self, handler, mock_http_handler):
        """Fail when sending to non-existent session."""
        _set_body(mock_http_handler, {"content": "Hello"})
        mock_http_handler.command = "POST"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/nonexistent/messages",
            {},
            mock_http_handler,
            "POST",
        )
        assert _status(result) == 404

    def test_list_messages_after_send(self, handler, mock_http_handler, social_store):
        """Messages appear in list after sending."""
        _create_social_session(social_store)
        # Send a message
        _set_body(mock_http_handler, {"content": "First message"})
        mock_http_handler.command = "POST"
        handler.handle(
            "/api/v1/social/collaboration/sessions/test-session-1/messages",
            {},
            mock_http_handler,
            "POST",
        )
        # List messages
        mock_http_handler.command = "GET"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/test-session-1/messages",
            {},
            mock_http_handler,
            "GET",
        )
        body = _body(result)
        assert body["total"] == 1
        assert body["messages"][0]["content"] == "First message"

    def test_send_message_invalid_json(self, handler, mock_http_handler, social_store):
        """Fail on invalid JSON for send message."""
        _create_social_session(social_store)
        mock_http_handler.rfile.read.return_value = b"broken json{{"
        mock_http_handler.headers["Content-Length"] = "13"
        mock_http_handler.command = "POST"
        result = handler.handle(
            "/api/v1/social/collaboration/sessions/test-session-1/messages",
            {},
            mock_http_handler,
            "POST",
        )
        assert _status(result) == 400


# ============================================================================
# SocialCollaborationStore tests
# ============================================================================


class TestSocialCollaborationStore:
    """Tests for the SocialCollaborationStore."""

    def test_create_and_get(self, social_store):
        """Create and retrieve a session."""
        session = _create_social_session(social_store)
        retrieved = social_store.get_by_id("test-session-1")
        assert retrieved is not None
        assert retrieved.name == "Test Session"

    def test_get_by_id_not_found(self, social_store):
        """Return None for non-existent session."""
        assert social_store.get_by_id("nonexistent") is None

    def test_get_by_org(self, social_store):
        """Filter sessions by org_id."""
        _create_social_session(social_store, id="s1", org_id="org-1")
        _create_social_session(social_store, id="s2", org_id="org-2")
        org1_sessions = social_store.get_by_org("org-1")
        assert len(org1_sessions) == 1
        assert org1_sessions[0].org_id == "org-1"

    def test_update_session(self, social_store):
        """Update session fields."""
        _create_social_session(social_store)
        result = social_store.update("test-session-1", {"name": "Updated"})
        assert result is True
        session = social_store.get_by_id("test-session-1")
        assert session.name == "Updated"

    def test_update_session_not_found(self, social_store):
        """Update returns False for non-existent session."""
        result = social_store.update("nonexistent", {"name": "Updated"})
        assert result is False

    def test_delete_session(self, social_store):
        """Delete a session."""
        _create_social_session(social_store)
        result = social_store.delete("test-session-1")
        assert result is True
        assert social_store.get_by_id("test-session-1") is None

    def test_delete_session_not_found(self, social_store):
        """Delete returns False for non-existent session."""
        result = social_store.delete("nonexistent")
        assert result is False

    def test_add_participant(self, social_store):
        """Add a participant to a session."""
        _create_social_session(social_store, participants=[])
        result = social_store.add_participant("test-session-1", "user-new")
        assert result is True
        participants = social_store.list_participants("test-session-1")
        assert "user-new" in participants

    def test_add_participant_duplicate(self, social_store):
        """Adding existing participant is idempotent."""
        _create_social_session(social_store, participants=["user-1"])
        social_store.add_participant("test-session-1", "user-1")
        participants = social_store.list_participants("test-session-1")
        assert participants.count("user-1") == 1

    def test_add_participant_not_found(self, social_store):
        """Add participant returns False for non-existent session."""
        result = social_store.add_participant("nonexistent", "user-1")
        assert result is False

    def test_remove_participant(self, social_store):
        """Remove a participant from a session."""
        _create_social_session(social_store, participants=["user-1", "user-2"])
        result = social_store.remove_participant("test-session-1", "user-2")
        assert result is True
        participants = social_store.list_participants("test-session-1")
        assert "user-2" not in participants

    def test_remove_participant_not_found(self, social_store):
        """Remove participant returns False for non-existent session."""
        result = social_store.remove_participant("nonexistent", "user-1")
        assert result is False

    def test_list_participants_not_found(self, social_store):
        """List participants returns empty list for non-existent session."""
        participants = social_store.list_participants("nonexistent")
        assert participants == []

    def test_add_message(self, social_store):
        """Add a message to a session."""
        _create_social_session(social_store)
        msg = {"content": "Hello", "sender": "user-1"}
        result = social_store.add_message("test-session-1", msg)
        assert result is True
        messages = social_store.list_messages("test-session-1")
        assert len(messages) == 1
        assert messages[0]["content"] == "Hello"

    def test_add_message_not_found(self, social_store):
        """Add message returns False for non-existent session."""
        result = social_store.add_message("nonexistent", {"content": "Hi"})
        assert result is False

    def test_list_messages_not_found(self, social_store):
        """List messages returns empty list for non-existent session."""
        messages = social_store.list_messages("nonexistent")
        assert messages == []

    def test_to_dict(self, social_store):
        """Session to_dict returns all fields."""
        session = _create_social_session(social_store, platform="discord")
        d = session.to_dict()
        assert d["id"] == "test-session-1"
        assert d["platform"] == "discord"
        assert d["status"] == "active"
        assert "created_at" in d


# ============================================================================
# SocialCollaborationSession tests
# ============================================================================


class TestSocialCollaborationSession:
    """Tests for SocialCollaborationSession dataclass."""

    def test_default_status(self):
        """Default status is 'active'."""
        session = SocialCollaborationSession(
            id="s1",
            org_id="",
            name="Test",
            description="",
            channel_id="ch-1",
            platform="slack",
            created_by="user-1",
        )
        assert session.status == "active"

    def test_default_participants(self):
        """Default participants is empty list."""
        session = SocialCollaborationSession(
            id="s1",
            org_id="",
            name="Test",
            description="",
            channel_id="ch-1",
            platform="slack",
            created_by="user-1",
        )
        assert session.participants == []


# ============================================================================
# Handler initialization tests
# ============================================================================


class TestHandlerInit:
    """Tests for handler initialization."""

    def test_default_store(self):
        """Handler creates default store when none provided."""
        h = CollaborationHandler(server_context={})
        assert h._store is not None

    def test_custom_store(self, social_store):
        """Handler uses provided store."""
        h = CollaborationHandler(server_context={"collaboration_store": social_store})
        assert h._store is social_store

    def test_none_context(self):
        """Handler handles None context."""
        h = CollaborationHandler(server_context=None)
        assert h.ctx == {}

    def test_collaboration_handlers_default_manager(self):
        """CollaborationHandlers uses get_session_manager by default."""
        h = CollaborationHandlers()
        assert h.manager is not None

    def test_collaboration_handlers_custom_manager(self, session_manager):
        """CollaborationHandlers uses provided manager."""
        h = CollaborationHandlers(manager=session_manager)
        assert h.manager is session_manager

    def test_routes_list(self):
        """CollaborationHandler has correct ROUTES."""
        assert CollaborationHandler.ROUTES == ["/api/v1/social/collaboration/sessions"]
