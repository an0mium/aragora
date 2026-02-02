"""
Tests for aragora.server.middleware.impersonation - Impersonation Session Enforcement.

Comprehensive tests covering:
- ImpersonationContext dataclass
- validate_impersonation_session function
- log_impersonation_access function
- refresh_impersonation_session function
- impersonation_middleware decorator
- require_valid_impersonation decorator
- get_impersonation_context helper
- Session expiration handling
- Audit logging of impersonation actions
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.auth.impersonation import (
    ImpersonationManager,
    ImpersonationSession,
)
from aragora.server.middleware.impersonation import (
    IMPERSONATION_CONTEXT_KEY,
    IMPERSONATION_SESSION_HEADER,
    ImpersonationContext,
    get_impersonation_context,
    impersonation_middleware,
    log_impersonation_access,
    refresh_impersonation_session,
    require_valid_impersonation,
    validate_impersonation_session,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockUser:
    """Mock user for testing."""

    id: str = "user-123"
    email: str = "user@example.com"
    role: str = "user"
    is_admin: bool = False
    metadata: dict = field(default_factory=dict)


def make_mock_handler(
    ctx: dict = None,
    headers: dict = None,
    path: str = "/api/test",
    method: str = "GET",
):
    """Create mock HTTP handler."""
    handler = MagicMock()
    handler.ctx = ctx or {}
    handler.headers = headers or {}
    handler.path = path
    handler.method = method
    return handler


def make_mock_session(
    session_id: str = "session-abc123",
    admin_user_id: str = "admin-456",
    admin_email: str = "admin@example.com",
    target_user_id: str = "target-789",
    target_email: str = "target@example.com",
    reason: str = "Testing impersonation for support ticket #123",
    started_at: datetime = None,
    expires_at: datetime = None,
    ip_address: str = "192.168.1.1",
    user_agent: str = "TestAgent/1.0",
    actions_performed: int = 0,
) -> ImpersonationSession:
    """Create a mock impersonation session."""
    now = datetime.now(timezone.utc)
    return ImpersonationSession(
        session_id=session_id,
        admin_user_id=admin_user_id,
        admin_email=admin_email,
        target_user_id=target_user_id,
        target_email=target_email,
        reason=reason,
        started_at=started_at or now,
        expires_at=expires_at or (now + timedelta(minutes=30)),
        ip_address=ip_address,
        user_agent=user_agent,
        actions_performed=actions_performed,
    )


def get_status(result) -> int:
    """Extract status code from result."""
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, tuple):
        return result[1]
    return 0


def get_body(result) -> dict:
    """Extract body from result."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, bytes):
            return json.loads(body.decode("utf-8"))
        if isinstance(body, str):
            return json.loads(body)
        return body
    if isinstance(result, tuple):
        body = result[0]
        if isinstance(body, dict):
            return body
        return json.loads(body)
    return {}


# ===========================================================================
# Test ImpersonationContext
# ===========================================================================


class TestImpersonationContext:
    """Tests for ImpersonationContext dataclass."""

    def test_default_values(self):
        ctx = ImpersonationContext()

        assert ctx.is_impersonated is False
        assert ctx.session_id is None
        assert ctx.admin_user_id is None
        assert ctx.admin_email is None
        assert ctx.target_user_id is None
        assert ctx.target_email is None
        assert ctx.reason is None
        assert ctx.started_at is None
        assert ctx.expires_at is None
        assert ctx.actions_performed == 0
        assert ctx.metadata == {}

    def test_from_session(self):
        session = make_mock_session()
        ctx = ImpersonationContext.from_session(session)

        assert ctx.is_impersonated is True
        assert ctx.session_id == session.session_id
        assert ctx.admin_user_id == session.admin_user_id
        assert ctx.admin_email == session.admin_email
        assert ctx.target_user_id == session.target_user_id
        assert ctx.target_email == session.target_email
        assert ctx.reason == session.reason
        assert ctx.started_at == session.started_at
        assert ctx.expires_at == session.expires_at
        assert ctx.actions_performed == session.actions_performed

    def test_to_audit_dict(self):
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=1)
        ctx = ImpersonationContext(
            is_impersonated=True,
            session_id="sess-123",
            admin_user_id="admin-1",
            admin_email="admin@test.com",
            target_user_id="target-1",
            target_email="target@test.com",
            reason="Support ticket",
            started_at=now,
            expires_at=expires,
            actions_performed=5,
        )

        audit_dict = ctx.to_audit_dict()

        assert audit_dict["is_impersonated"] is True
        assert audit_dict["session_id"] == "sess-123"
        assert audit_dict["admin_user_id"] == "admin-1"
        assert audit_dict["admin_email"] == "admin@test.com"
        assert audit_dict["target_user_id"] == "target-1"
        assert audit_dict["target_email"] == "target@test.com"
        assert audit_dict["reason"] == "Support ticket"
        assert audit_dict["started_at"] == now.isoformat()
        assert audit_dict["expires_at"] == expires.isoformat()
        assert audit_dict["actions_performed"] == 5

    def test_to_audit_dict_with_none_dates(self):
        ctx = ImpersonationContext()
        audit_dict = ctx.to_audit_dict()

        assert audit_dict["started_at"] is None
        assert audit_dict["expires_at"] is None


# ===========================================================================
# Test validate_impersonation_session
# ===========================================================================


class TestValidateImpersonationSession:
    """Tests for validate_impersonation_session function."""

    def test_valid_session_passes(self):
        session = make_mock_session(admin_user_id="admin-123")
        manager = MagicMock(spec=ImpersonationManager)
        manager.validate_session.return_value = session

        result_session, error = validate_impersonation_session(
            session_id="session-abc",
            requester_user_id="admin-123",
            ip_address="192.168.1.1",
            user_agent="TestAgent",
            manager=manager,
        )

        assert result_session is session
        assert error is None
        manager.validate_session.assert_called_once_with("session-abc")

    def test_session_not_found_returns_error(self):
        manager = MagicMock(spec=ImpersonationManager)
        manager.validate_session.return_value = None

        with patch("aragora.server.middleware.impersonation.audit_event"):
            result_session, error = validate_impersonation_session(
                session_id="nonexistent-session",
                requester_user_id="admin-123",
                ip_address="192.168.1.1",
                user_agent="TestAgent",
                manager=manager,
            )

        assert result_session is None
        assert "not found or expired" in error

    def test_expired_session_returns_error(self):
        # ImpersonationManager.validate_session returns None for expired sessions
        manager = MagicMock(spec=ImpersonationManager)
        manager.validate_session.return_value = None

        with patch("aragora.server.middleware.impersonation.audit_event"):
            result_session, error = validate_impersonation_session(
                session_id="expired-session",
                requester_user_id="admin-123",
                ip_address="192.168.1.1",
                user_agent="TestAgent",
                manager=manager,
            )

        assert result_session is None
        assert "not found or expired" in error

    def test_wrong_requester_returns_error(self):
        session = make_mock_session(admin_user_id="admin-123")
        manager = MagicMock(spec=ImpersonationManager)
        manager.validate_session.return_value = session

        with patch("aragora.server.middleware.impersonation.audit_event"):
            result_session, error = validate_impersonation_session(
                session_id="session-abc",
                requester_user_id="different-admin-456",  # Not the session admin
                ip_address="192.168.1.1",
                user_agent="TestAgent",
                manager=manager,
            )

        assert result_session is None
        assert "Not authorized" in error

    def test_none_requester_allows_access(self):
        """When requester_user_id is None, skip authorization check."""
        session = make_mock_session(admin_user_id="admin-123")
        manager = MagicMock(spec=ImpersonationManager)
        manager.validate_session.return_value = session

        result_session, error = validate_impersonation_session(
            session_id="session-abc",
            requester_user_id=None,  # No requester info
            ip_address="192.168.1.1",
            user_agent="TestAgent",
            manager=manager,
        )

        assert result_session is session
        assert error is None

    def test_audit_logging_on_invalid_session(self):
        manager = MagicMock(spec=ImpersonationManager)
        manager.validate_session.return_value = None

        with patch("aragora.server.middleware.impersonation.audit_event") as mock_audit:
            validate_impersonation_session(
                session_id="invalid-session",
                requester_user_id="admin-123",
                ip_address="192.168.1.1",
                user_agent="TestAgent",
                manager=manager,
            )

            mock_audit.assert_called_once()
            call_kwargs = mock_audit.call_args[1]
            assert call_kwargs["action"] == "impersonation.session_invalid"
            assert call_kwargs["outcome"] == "denied"

    def test_audit_logging_on_unauthorized_access(self):
        session = make_mock_session(admin_user_id="admin-123")
        manager = MagicMock(spec=ImpersonationManager)
        manager.validate_session.return_value = session

        with patch("aragora.server.middleware.impersonation.audit_event") as mock_audit:
            validate_impersonation_session(
                session_id="session-abc",
                requester_user_id="wrong-admin",
                ip_address="192.168.1.1",
                user_agent="TestAgent",
                manager=manager,
            )

            mock_audit.assert_called_once()
            call_kwargs = mock_audit.call_args[1]
            assert call_kwargs["action"] == "impersonation.unauthorized_access"
            assert call_kwargs["outcome"] == "denied"

    def test_uses_global_manager_when_not_provided(self):
        with patch(
            "aragora.server.middleware.impersonation.get_impersonation_manager"
        ) as mock_get:
            mock_manager = MagicMock(spec=ImpersonationManager)
            mock_manager.validate_session.return_value = None
            mock_get.return_value = mock_manager

            with patch("aragora.server.middleware.impersonation.audit_event"):
                validate_impersonation_session(
                    session_id="session-abc",
                    requester_user_id="admin-123",
                    ip_address="192.168.1.1",
                    user_agent="TestAgent",
                )

            mock_get.assert_called_once()


# ===========================================================================
# Test log_impersonation_access
# ===========================================================================


class TestLogImpersonationAccess:
    """Tests for log_impersonation_access function."""

    def test_logs_to_manager(self):
        session = make_mock_session()
        manager = MagicMock(spec=ImpersonationManager)

        with patch("aragora.server.middleware.impersonation.audit_event"):
            log_impersonation_access(
                session=session,
                action_type="request",
                ip_address="192.168.1.1",
                user_agent="TestAgent",
                manager=manager,
            )

        manager.log_impersonation_action.assert_called_once()
        call_kwargs = manager.log_impersonation_action.call_args[1]
        assert call_kwargs["session_id"] == session.session_id
        assert call_kwargs["action_type"] == "request"
        assert call_kwargs["ip_address"] == "192.168.1.1"
        assert call_kwargs["user_agent"] == "TestAgent"

    def test_logs_to_audit_system(self):
        session = make_mock_session()
        manager = MagicMock(spec=ImpersonationManager)

        with patch("aragora.server.middleware.impersonation.audit_event") as mock_audit:
            log_impersonation_access(
                session=session,
                action_type="data_access",
                ip_address="192.168.1.1",
                user_agent="TestAgent",
                additional_details={"resource": "/api/users"},
                manager=manager,
            )

            mock_audit.assert_called_once()
            call_kwargs = mock_audit.call_args[1]
            assert call_kwargs["action"] == "impersonation.data_access"
            assert call_kwargs["actor"] == session.admin_user_id
            assert call_kwargs["outcome"] == "success"
            assert "resource" in call_kwargs["details"]

    def test_includes_additional_details(self):
        session = make_mock_session()
        manager = MagicMock(spec=ImpersonationManager)

        with patch("aragora.server.middleware.impersonation.audit_event"):
            log_impersonation_access(
                session=session,
                action_type="request",
                ip_address="192.168.1.1",
                user_agent="TestAgent",
                additional_details={"endpoint": "/api/users", "method": "GET"},
                manager=manager,
            )

        call_kwargs = manager.log_impersonation_action.call_args[1]
        details = call_kwargs["action_details"]
        assert details["endpoint"] == "/api/users"
        assert details["method"] == "GET"


# ===========================================================================
# Test refresh_impersonation_session
# ===========================================================================


class TestRefreshImpersonationSession:
    """Tests for refresh_impersonation_session function."""

    def test_refresh_extends_expiration(self):
        now = datetime.now(timezone.utc)
        session = make_mock_session(
            admin_user_id="admin-123",
            started_at=now - timedelta(minutes=10),
            expires_at=now + timedelta(minutes=5),
        )
        manager = MagicMock(spec=ImpersonationManager)
        manager.validate_session.return_value = session
        manager.DEFAULT_SESSION_DURATION = timedelta(minutes=30)
        manager.MAX_SESSION_DURATION = timedelta(hours=1)

        with patch("aragora.server.middleware.impersonation.audit_event"):
            result_session, message = refresh_impersonation_session(
                session_id="session-abc",
                requester_user_id="admin-123",
                ip_address="192.168.1.1",
                user_agent="TestAgent",
                manager=manager,
            )

        assert result_session is session
        assert "refreshed" in message.lower()
        # Expiration should be extended
        assert session.expires_at > now + timedelta(minutes=5)

    def test_refresh_caps_at_max_duration(self):
        now = datetime.now(timezone.utc)
        session = make_mock_session(
            admin_user_id="admin-123",
            started_at=now - timedelta(minutes=50),  # Started 50 mins ago
            expires_at=now + timedelta(minutes=5),
        )
        manager = MagicMock(spec=ImpersonationManager)
        manager.validate_session.return_value = session
        manager.DEFAULT_SESSION_DURATION = timedelta(minutes=30)
        manager.MAX_SESSION_DURATION = timedelta(hours=1)  # 60 min max from start

        with patch("aragora.server.middleware.impersonation.audit_event"):
            result_session, message = refresh_impersonation_session(
                session_id="session-abc",
                requester_user_id="admin-123",
                ip_address="192.168.1.1",
                user_agent="TestAgent",
                manager=manager,
            )

        # Max would be started_at + 1 hour = now - 50min + 60min = now + 10min
        max_expires = session.started_at + manager.MAX_SESSION_DURATION
        assert session.expires_at == max_expires

    def test_refresh_invalid_session_fails(self):
        manager = MagicMock(spec=ImpersonationManager)
        manager.validate_session.return_value = None

        with patch("aragora.server.middleware.impersonation.audit_event"):
            result_session, message = refresh_impersonation_session(
                session_id="invalid-session",
                requester_user_id="admin-123",
                ip_address="192.168.1.1",
                user_agent="TestAgent",
                manager=manager,
            )

        assert result_session is None
        assert "not found or expired" in message or "failed" in message.lower()

    def test_refresh_wrong_user_fails(self):
        session = make_mock_session(admin_user_id="admin-123")
        manager = MagicMock(spec=ImpersonationManager)
        manager.validate_session.return_value = session

        with patch("aragora.server.middleware.impersonation.audit_event"):
            result_session, message = refresh_impersonation_session(
                session_id="session-abc",
                requester_user_id="different-admin",
                ip_address="192.168.1.1",
                user_agent="TestAgent",
                manager=manager,
            )

        assert result_session is None
        assert "Not authorized" in message

    def test_refresh_custom_extension(self):
        now = datetime.now(timezone.utc)
        session = make_mock_session(
            admin_user_id="admin-123",
            started_at=now,
            expires_at=now + timedelta(minutes=5),
        )
        manager = MagicMock(spec=ImpersonationManager)
        manager.validate_session.return_value = session
        manager.MAX_SESSION_DURATION = timedelta(hours=2)

        with patch("aragora.server.middleware.impersonation.audit_event"):
            refresh_impersonation_session(
                session_id="session-abc",
                requester_user_id="admin-123",
                ip_address="192.168.1.1",
                user_agent="TestAgent",
                extension=timedelta(minutes=15),
                manager=manager,
            )

        # Should extend by ~15 minutes from now
        expected_min = now + timedelta(minutes=14)
        expected_max = now + timedelta(minutes=16)
        assert expected_min <= session.expires_at <= expected_max

    def test_refresh_logs_audit_event(self):
        session = make_mock_session(admin_user_id="admin-123")
        manager = MagicMock(spec=ImpersonationManager)
        manager.validate_session.return_value = session
        manager.DEFAULT_SESSION_DURATION = timedelta(minutes=30)
        manager.MAX_SESSION_DURATION = timedelta(hours=1)

        with patch("aragora.server.middleware.impersonation.audit_event") as mock_audit:
            refresh_impersonation_session(
                session_id="session-abc",
                requester_user_id="admin-123",
                ip_address="192.168.1.1",
                user_agent="TestAgent",
                manager=manager,
            )

            # Should log the refresh event
            assert mock_audit.called
            call_kwargs = mock_audit.call_args[1]
            assert call_kwargs["action"] == "impersonation.session_refreshed"


# ===========================================================================
# Test impersonation_middleware Decorator
# ===========================================================================


class TestImpersonationMiddleware:
    """Tests for impersonation_middleware decorator."""

    def test_no_header_proceeds_normally(self):
        @impersonation_middleware
        def endpoint(handler):
            return {"success": True}

        handler = make_mock_handler(headers={})
        result = endpoint(handler)

        assert result["success"] is True

    def test_no_header_sets_empty_context(self):
        @impersonation_middleware
        def endpoint(handler):
            ctx = get_impersonation_context(handler)
            return {"is_impersonated": ctx.is_impersonated}

        handler = make_mock_handler(headers={})
        result = endpoint(handler)

        assert result["is_impersonated"] is False

    @patch("aragora.server.middleware.impersonation.get_current_user")
    @patch("aragora.server.middleware.impersonation.get_impersonation_manager")
    def test_valid_session_sets_context(self, mock_get_manager, mock_get_user):
        session = make_mock_session(admin_user_id="admin-123")
        mock_manager = MagicMock(spec=ImpersonationManager)
        mock_manager.validate_session.return_value = session
        mock_get_manager.return_value = mock_manager

        mock_user = MockUser(id="admin-123")
        mock_get_user.return_value = mock_user

        @impersonation_middleware
        def endpoint(handler):
            ctx = get_impersonation_context(handler)
            return {
                "is_impersonated": ctx.is_impersonated,
                "session_id": ctx.session_id,
            }

        handler = make_mock_handler(headers={IMPERSONATION_SESSION_HEADER: "session-abc"})
        result = endpoint(handler)

        assert result["is_impersonated"] is True
        assert result["session_id"] == session.session_id

    @patch("aragora.server.middleware.impersonation.get_current_user")
    @patch("aragora.server.middleware.impersonation.get_impersonation_manager")
    def test_invalid_session_returns_403(self, mock_get_manager, mock_get_user):
        mock_manager = MagicMock(spec=ImpersonationManager)
        mock_manager.validate_session.return_value = None
        mock_get_manager.return_value = mock_manager

        mock_user = MockUser(id="admin-123")
        mock_get_user.return_value = mock_user

        @impersonation_middleware
        def endpoint(handler):
            return {"success": True}

        handler = make_mock_handler(
            headers={IMPERSONATION_SESSION_HEADER: "invalid-session"}
        )
        result = endpoint(handler)

        assert get_status(result) == 403

    @patch("aragora.server.middleware.impersonation.get_current_user")
    @patch("aragora.server.middleware.impersonation.get_impersonation_manager")
    def test_expired_session_blocked(self, mock_get_manager, mock_get_user):
        mock_manager = MagicMock(spec=ImpersonationManager)
        mock_manager.validate_session.return_value = None  # Returns None for expired
        mock_get_manager.return_value = mock_manager

        mock_user = MockUser(id="admin-123")
        mock_get_user.return_value = mock_user

        @impersonation_middleware
        def endpoint(handler):
            return {"success": True}

        handler = make_mock_handler(
            headers={IMPERSONATION_SESSION_HEADER: "expired-session"}
        )
        result = endpoint(handler)

        assert get_status(result) == 403

    @patch("aragora.server.middleware.impersonation.get_current_user")
    @patch("aragora.server.middleware.impersonation.get_impersonation_manager")
    @patch("aragora.server.middleware.impersonation.log_impersonation_access")
    def test_valid_session_logs_access(self, mock_log, mock_get_manager, mock_get_user):
        session = make_mock_session(admin_user_id="admin-123")
        mock_manager = MagicMock(spec=ImpersonationManager)
        mock_manager.validate_session.return_value = session
        mock_get_manager.return_value = mock_manager

        mock_user = MockUser(id="admin-123")
        mock_get_user.return_value = mock_user

        @impersonation_middleware
        def endpoint(handler):
            return {"success": True}

        handler = make_mock_handler(
            headers={IMPERSONATION_SESSION_HEADER: "session-abc"},
            path="/api/users",
            method="GET",
        )
        endpoint(handler)

        mock_log.assert_called_once()
        call_kwargs = mock_log.call_args[1]
        assert call_kwargs["session"] is session
        assert call_kwargs["action_type"] == "request"

    @patch("aragora.server.middleware.impersonation.get_current_user")
    @patch("aragora.server.middleware.impersonation.get_impersonation_manager")
    def test_wrong_user_blocked(self, mock_get_manager, mock_get_user):
        session = make_mock_session(admin_user_id="admin-123")
        mock_manager = MagicMock(spec=ImpersonationManager)
        mock_manager.validate_session.return_value = session
        mock_get_manager.return_value = mock_manager

        # Different user than session admin
        mock_user = MockUser(id="different-user-456")
        mock_get_user.return_value = mock_user

        @impersonation_middleware
        def endpoint(handler):
            return {"success": True}

        handler = make_mock_handler(
            headers={IMPERSONATION_SESSION_HEADER: "session-abc"}
        )
        result = endpoint(handler)

        assert get_status(result) == 403

    def test_lowercase_header_name_works(self):
        """Header name matching should be case-insensitive."""

        @impersonation_middleware
        def endpoint(handler):
            return {"success": True}

        # Use lowercase header
        handler = make_mock_handler(headers={"x-impersonation-session-id": "session-abc"})

        with patch(
            "aragora.server.middleware.impersonation.get_impersonation_manager"
        ) as mock_get:
            mock_manager = MagicMock(spec=ImpersonationManager)
            mock_manager.validate_session.return_value = None
            mock_get.return_value = mock_manager

            with patch("aragora.server.middleware.impersonation.get_current_user"):
                result = endpoint(handler)

        # Should have tried to validate (returned 403 for invalid)
        assert get_status(result) == 403

    def test_handler_from_kwargs(self):
        @impersonation_middleware
        def endpoint(handler=None):
            return {"success": True}

        handler = make_mock_handler(headers={})
        result = endpoint(handler=handler)

        assert result["success"] is True

    def test_handler_from_args(self):
        @impersonation_middleware
        def endpoint(self_arg, handler):
            return {"success": True}

        handler = make_mock_handler(headers={})
        result = endpoint(object(), handler)

        assert result["success"] is True


# ===========================================================================
# Test require_valid_impersonation Decorator
# ===========================================================================


class TestRequireValidImpersonation:
    """Tests for require_valid_impersonation decorator."""

    def test_no_header_returns_403(self):
        @require_valid_impersonation
        def endpoint(handler):
            return {"success": True}

        handler = make_mock_handler(headers={})
        result = endpoint(handler)

        assert get_status(result) == 403
        body = get_body(result)
        assert "X-Impersonation-Session-ID" in str(body) or "session required" in str(
            body
        ).lower()

    def test_no_handler_returns_500(self):
        @require_valid_impersonation
        def endpoint():
            return {"success": True}

        result = endpoint()

        assert get_status(result) == 500

    @patch("aragora.server.middleware.impersonation.get_current_user")
    @patch("aragora.server.middleware.impersonation.get_impersonation_manager")
    def test_valid_session_allows_access(self, mock_get_manager, mock_get_user):
        session = make_mock_session(admin_user_id="admin-123")
        mock_manager = MagicMock(spec=ImpersonationManager)
        mock_manager.validate_session.return_value = session
        mock_get_manager.return_value = mock_manager

        mock_user = MockUser(id="admin-123")
        mock_get_user.return_value = mock_user

        @require_valid_impersonation
        def endpoint(handler):
            ctx = get_impersonation_context(handler)
            return {"success": True, "impersonating": ctx.target_user_id}

        handler = make_mock_handler(
            headers={IMPERSONATION_SESSION_HEADER: "session-abc"}
        )
        result = endpoint(handler)

        assert result["success"] is True
        assert result["impersonating"] == session.target_user_id

    @patch("aragora.server.middleware.impersonation.get_current_user")
    @patch("aragora.server.middleware.impersonation.get_impersonation_manager")
    def test_invalid_session_blocked(self, mock_get_manager, mock_get_user):
        mock_manager = MagicMock(spec=ImpersonationManager)
        mock_manager.validate_session.return_value = None
        mock_get_manager.return_value = mock_manager

        mock_user = MockUser(id="admin-123")
        mock_get_user.return_value = mock_user

        @require_valid_impersonation
        def endpoint(handler):
            return {"success": True}

        handler = make_mock_handler(
            headers={IMPERSONATION_SESSION_HEADER: "invalid-session"}
        )
        result = endpoint(handler)

        assert get_status(result) == 403

    @patch("aragora.server.middleware.impersonation.get_current_user")
    @patch("aragora.server.middleware.impersonation.get_impersonation_manager")
    @patch("aragora.server.middleware.impersonation.log_impersonation_access")
    def test_logs_privileged_request(self, mock_log, mock_get_manager, mock_get_user):
        session = make_mock_session(admin_user_id="admin-123")
        mock_manager = MagicMock(spec=ImpersonationManager)
        mock_manager.validate_session.return_value = session
        mock_get_manager.return_value = mock_manager

        mock_user = MockUser(id="admin-123")
        mock_get_user.return_value = mock_user

        @require_valid_impersonation
        def endpoint(handler):
            return {"success": True}

        handler = make_mock_handler(
            headers={IMPERSONATION_SESSION_HEADER: "session-abc"}
        )
        endpoint(handler)

        mock_log.assert_called_once()
        call_kwargs = mock_log.call_args[1]
        assert call_kwargs["action_type"] == "privileged_request"


# ===========================================================================
# Test get_impersonation_context
# ===========================================================================


class TestGetImpersonationContext:
    """Tests for get_impersonation_context helper function."""

    def test_returns_empty_context_for_none_handler(self):
        ctx = get_impersonation_context(None)

        assert ctx.is_impersonated is False

    def test_returns_context_from_handler_attribute(self):
        expected_ctx = ImpersonationContext(is_impersonated=True, session_id="sess-1")
        handler = MagicMock()
        handler.impersonation_context = expected_ctx

        ctx = get_impersonation_context(handler)

        assert ctx is expected_ctx

    def test_returns_context_from_handler_ctx_dict(self):
        expected_ctx = ImpersonationContext(is_impersonated=True, session_id="sess-2")
        handler = MagicMock()
        handler.impersonation_context = None
        handler.ctx = {IMPERSONATION_CONTEXT_KEY: expected_ctx}

        ctx = get_impersonation_context(handler)

        assert ctx is expected_ctx

    def test_returns_context_from_handler_ctx_object(self):
        expected_ctx = ImpersonationContext(is_impersonated=True, session_id="sess-3")
        handler = MagicMock()
        handler.impersonation_context = None
        handler.ctx = MagicMock()
        setattr(handler.ctx, IMPERSONATION_CONTEXT_KEY, expected_ctx)

        ctx = get_impersonation_context(handler)

        assert ctx is expected_ctx

    def test_returns_empty_context_when_not_found(self):
        handler = MagicMock()
        handler.impersonation_context = None
        handler.ctx = {}

        ctx = get_impersonation_context(handler)

        assert ctx.is_impersonated is False


# ===========================================================================
# Test Decorator Preserves Function Metadata
# ===========================================================================


class TestDecoratorMetadata:
    """Test that decorators preserve function metadata via functools.wraps."""

    def test_impersonation_middleware_preserves_name(self):
        @impersonation_middleware
        def my_endpoint():
            """My endpoint docstring."""
            pass

        assert my_endpoint.__name__ == "my_endpoint"
        assert my_endpoint.__doc__ == "My endpoint docstring."

    def test_require_valid_impersonation_preserves_name(self):
        @require_valid_impersonation
        def impersonation_endpoint():
            """Impersonation endpoint docstring."""
            pass

        assert impersonation_endpoint.__name__ == "impersonation_endpoint"
        assert impersonation_endpoint.__doc__ == "Impersonation endpoint docstring."


# ===========================================================================
# Integration Test: Full Flow
# ===========================================================================


class TestIntegrationFullFlow:
    """Integration tests for full impersonation middleware flow."""

    @patch("aragora.server.middleware.impersonation.get_current_user")
    @patch("aragora.server.middleware.impersonation.get_impersonation_manager")
    @patch("aragora.server.middleware.impersonation.audit_event")
    def test_full_valid_impersonation_flow(
        self, mock_audit, mock_get_manager, mock_get_user
    ):
        """Test complete flow: valid session, context set, audit logged."""
        session = make_mock_session(
            session_id="session-full-test",
            admin_user_id="admin-123",
            admin_email="admin@company.com",
            target_user_id="user-456",
            target_email="user@company.com",
        )
        mock_manager = MagicMock(spec=ImpersonationManager)
        mock_manager.validate_session.return_value = session
        mock_get_manager.return_value = mock_manager

        mock_user = MockUser(id="admin-123")
        mock_get_user.return_value = mock_user

        @impersonation_middleware
        def api_endpoint(handler):
            ctx = get_impersonation_context(handler)
            return {
                "is_impersonated": ctx.is_impersonated,
                "admin": ctx.admin_email,
                "target": ctx.target_email,
            }

        handler = make_mock_handler(
            headers={IMPERSONATION_SESSION_HEADER: "session-full-test"},
            path="/api/users/profile",
            method="GET",
        )

        result = api_endpoint(handler)

        # Verify successful response
        assert result["is_impersonated"] is True
        assert result["admin"] == "admin@company.com"
        assert result["target"] == "user@company.com"

        # Verify audit was logged
        assert mock_audit.called

        # Verify manager was used
        mock_manager.validate_session.assert_called_once_with("session-full-test")

    @patch("aragora.server.middleware.impersonation.get_current_user")
    @patch("aragora.server.middleware.impersonation.get_impersonation_manager")
    @patch("aragora.server.middleware.impersonation.audit_event")
    def test_full_expired_session_flow(
        self, mock_audit, mock_get_manager, mock_get_user
    ):
        """Test complete flow: expired session rejected with audit."""
        mock_manager = MagicMock(spec=ImpersonationManager)
        mock_manager.validate_session.return_value = None  # Expired/not found
        mock_get_manager.return_value = mock_manager

        mock_user = MockUser(id="admin-123")
        mock_get_user.return_value = mock_user

        @impersonation_middleware
        def api_endpoint(handler):
            return {"success": True}

        handler = make_mock_handler(
            headers={IMPERSONATION_SESSION_HEADER: "expired-session"},
        )

        result = api_endpoint(handler)

        # Verify 403 response
        assert get_status(result) == 403

        # Verify audit was logged for the failed attempt
        assert mock_audit.called
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["outcome"] == "denied"
