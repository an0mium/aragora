"""Tests for admin impersonation controls and audit logging."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from aragora.auth.impersonation import (
    ImpersonationSession,
    ImpersonationAuditEntry,
    ImpersonationManager,
    get_impersonation_manager,
    configure_impersonation_manager,
)


class TestImpersonationSession:
    """Tests for ImpersonationSession dataclass."""

    def test_session_not_expired_when_valid(self):
        """Session should not be expired when expires_at is in the future."""
        session = ImpersonationSession(
            session_id="test123",
            admin_user_id="admin1",
            admin_email="admin@example.com",
            target_user_id="user1",
            target_email="user@example.com",
            reason="Testing impersonation",
            started_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=1),
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
        )
        assert not session.is_expired()

    def test_session_expired_when_past(self):
        """Session should be expired when expires_at is in the past."""
        session = ImpersonationSession(
            session_id="test123",
            admin_user_id="admin1",
            admin_email="admin@example.com",
            target_user_id="user1",
            target_email="user@example.com",
            reason="Testing impersonation",
            started_at=datetime.utcnow() - timedelta(hours=2),
            expires_at=datetime.utcnow() - timedelta(hours=1),
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
        )
        assert session.is_expired()

    def test_to_audit_dict(self):
        """Session should convert to audit dict format."""
        now = datetime.utcnow()
        session = ImpersonationSession(
            session_id="test123",
            admin_user_id="admin1",
            admin_email="admin@example.com",
            target_user_id="user1",
            target_email="user@example.com",
            reason="Testing",
            started_at=now,
            expires_at=now + timedelta(hours=1),
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
            actions_performed=5,
        )
        audit_dict = session.to_audit_dict()
        assert audit_dict["session_id"] == "test123"
        assert audit_dict["admin_user_id"] == "admin1"
        assert audit_dict["actions_performed"] == 5


class TestImpersonationAuditEntry:
    """Tests for ImpersonationAuditEntry dataclass."""

    def test_to_dict_includes_all_fields(self):
        """Audit entry should serialize all fields."""
        entry = ImpersonationAuditEntry(
            timestamp=datetime.utcnow(),
            event_type="start",
            session_id="sess123",
            admin_user_id="admin1",
            target_user_id="user1",
            reason="Testing",
            action_details={"key": "value"},
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
            success=True,
        )
        d = entry.to_dict()
        assert d["event_type"] == "start"
        assert d["success"] is True
        assert d["action_details"] == {"key": "value"}


class TestImpersonationManager:
    """Tests for ImpersonationManager."""

    @pytest.fixture
    def manager(self):
        """Create a fresh manager for each test."""
        return ImpersonationManager(
            require_2fa_for_admin_targets=True,
            max_concurrent_sessions=3,
        )

    def test_start_impersonation_success(self, manager):
        """Should successfully start impersonation with valid inputs."""
        session, msg = manager.start_impersonation(
            admin_user_id="admin1",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user1",
            target_email="user@example.com",
            target_roles=["user"],
            reason="Investigating support ticket #12345",
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
        )
        assert session is not None
        assert session.admin_user_id == "admin1"
        assert session.target_user_id == "user1"
        assert "started" in msg.lower()

    def test_start_impersonation_requires_reason(self, manager):
        """Should reject impersonation without sufficient reason."""
        session, msg = manager.start_impersonation(
            admin_user_id="admin1",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user1",
            target_email="user@example.com",
            target_roles=["user"],
            reason="short",  # Too short
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
        )
        assert session is None
        assert "10 characters" in msg

    def test_cannot_impersonate_self(self, manager):
        """Should reject self-impersonation."""
        session, msg = manager.start_impersonation(
            admin_user_id="admin1",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="admin1",  # Same as admin
            target_email="admin@example.com",
            target_roles=["admin"],
            reason="Testing self-impersonation",
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
        )
        assert session is None
        assert "yourself" in msg.lower()

    def test_requires_2fa_for_admin_targets(self, manager):
        """Should require 2FA when impersonating admin users."""
        session, msg = manager.start_impersonation(
            admin_user_id="admin1",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="admin2",
            target_email="admin2@example.com",
            target_roles=["admin"],  # Target is admin
            reason="Testing admin impersonation",
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
            has_2fa=False,  # No 2FA
        )
        assert session is None
        assert "2fa" in msg.lower()

    def test_2fa_allows_admin_impersonation(self, manager):
        """Should allow admin impersonation with 2FA."""
        session, msg = manager.start_impersonation(
            admin_user_id="admin1",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="admin2",
            target_email="admin2@example.com",
            target_roles=["admin"],
            reason="Testing admin impersonation with 2FA",
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
            has_2fa=True,  # 2FA provided
        )
        assert session is not None

    def test_max_concurrent_sessions_limit(self, manager):
        """Should enforce max concurrent sessions per admin."""
        # Start max sessions
        for i in range(manager._max_concurrent_sessions):
            session, _ = manager.start_impersonation(
                admin_user_id="admin1",
                admin_email="admin@example.com",
                admin_roles=["admin"],
                target_user_id=f"user{i}",
                target_email=f"user{i}@example.com",
                target_roles=["user"],
                reason="Testing concurrent sessions limit",
                ip_address="127.0.0.1",
                user_agent="TestAgent/1.0",
            )
            assert session is not None

        # Try to start one more
        session, msg = manager.start_impersonation(
            admin_user_id="admin1",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="userN",
            target_email="userN@example.com",
            target_roles=["user"],
            reason="This should fail due to limit",
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
        )
        assert session is None
        assert "maximum" in msg.lower()

    def test_session_duration_capped(self, manager):
        """Should cap session duration at MAX_SESSION_DURATION."""
        session, _ = manager.start_impersonation(
            admin_user_id="admin1",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user1",
            target_email="user@example.com",
            target_roles=["user"],
            reason="Testing duration cap",
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
            duration=timedelta(hours=24),  # Request 24 hours
        )
        assert session is not None
        actual_duration = session.expires_at - session.started_at
        assert actual_duration <= manager.MAX_SESSION_DURATION

    def test_end_impersonation(self, manager):
        """Should successfully end impersonation session."""
        session, _ = manager.start_impersonation(
            admin_user_id="admin1",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user1",
            target_email="user@example.com",
            target_roles=["user"],
            reason="Testing end impersonation",
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
        )
        assert session is not None

        success, msg = manager.end_impersonation(
            session_id=session.session_id,
            admin_user_id="admin1",
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
        )
        assert success is True
        assert "ended" in msg.lower()

        # Session should be gone
        assert manager.validate_session(session.session_id) is None

    def test_only_session_owner_can_end(self, manager):
        """Should only allow session owner to end session."""
        session, _ = manager.start_impersonation(
            admin_user_id="admin1",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user1",
            target_email="user@example.com",
            target_roles=["user"],
            reason="Testing ownership check",
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
        )
        assert session is not None

        success, msg = manager.end_impersonation(
            session_id=session.session_id,
            admin_user_id="admin2",  # Different admin
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
        )
        assert success is False
        assert "owner" in msg.lower() or "started" in msg.lower()

    def test_log_impersonation_action(self, manager):
        """Should log actions during impersonation."""
        session, _ = manager.start_impersonation(
            admin_user_id="admin1",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user1",
            target_email="user@example.com",
            target_roles=["user"],
            reason="Testing action logging",
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
        )
        assert session is not None
        assert session.actions_performed == 0

        logged = manager.log_impersonation_action(
            session_id=session.session_id,
            action_type="view_profile",
            action_details={"profile_id": "user1"},
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
        )
        assert logged is True

        # Check counter increased
        updated_session = manager.validate_session(session.session_id)
        assert updated_session.actions_performed == 1

    def test_validate_session_returns_none_for_expired(self, manager):
        """Should return None for expired sessions."""
        session, _ = manager.start_impersonation(
            admin_user_id="admin1",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user1",
            target_email="user@example.com",
            target_roles=["user"],
            reason="Testing expiration",
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
            duration=timedelta(seconds=0),  # Expire immediately
        )
        # Manually expire it
        session.expires_at = datetime.utcnow() - timedelta(seconds=1)
        manager._sessions[session.session_id] = session

        result = manager.validate_session(session.session_id)
        assert result is None

    def test_audit_callback_invoked(self):
        """Should invoke audit callback for each audit entry."""
        audit_entries = []
        manager = ImpersonationManager(
            audit_callback=lambda e: audit_entries.append(e),
        )

        manager.start_impersonation(
            admin_user_id="admin1",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user1",
            target_email="user@example.com",
            target_roles=["user"],
            reason="Testing audit callback",
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
        )

        assert len(audit_entries) == 1
        assert audit_entries[0].event_type == "start"

    def test_notification_callback_invoked(self):
        """Should notify target user of impersonation."""
        notifications = []
        manager = ImpersonationManager(
            notification_callback=lambda uid, email, reason: notifications.append(
                (uid, email, reason)
            ),
        )

        manager.start_impersonation(
            admin_user_id="admin1",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user1",
            target_email="user@example.com",
            target_roles=["user"],
            reason="Testing notification",
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
        )

        assert len(notifications) == 1
        assert notifications[0][0] == "user1"
        assert notifications[0][1] == "admin@example.com"

    def test_get_active_sessions_for_admin(self, manager):
        """Should return all active sessions for an admin."""
        for i in range(2):
            manager.start_impersonation(
                admin_user_id="admin1",
                admin_email="admin@example.com",
                admin_roles=["admin"],
                target_user_id=f"user{i}",
                target_email=f"user{i}@example.com",
                target_roles=["user"],
                reason="Testing active sessions list",
                ip_address="127.0.0.1",
                user_agent="TestAgent/1.0",
            )

        sessions = manager.get_active_sessions_for_admin("admin1")
        assert len(sessions) == 2

    def test_get_audit_log_filtering(self, manager):
        """Should filter audit log by criteria."""
        # Create some audit entries
        manager.start_impersonation(
            admin_user_id="admin1",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user1",
            target_email="user@example.com",
            target_roles=["user"],
            reason="Testing audit log filtering",
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
        )

        # Filter by admin
        entries = manager.get_audit_log(admin_user_id="admin1")
        assert len(entries) >= 1
        assert all(e.admin_user_id == "admin1" for e in entries)

        # Filter by event type
        entries = manager.get_audit_log(event_type="start")
        assert len(entries) >= 1
        assert all(e.event_type == "start" for e in entries)


class TestGlobalManager:
    """Tests for global manager functions."""

    def test_get_impersonation_manager_returns_singleton(self):
        """Should return same instance on repeated calls."""
        manager1 = get_impersonation_manager()
        manager2 = get_impersonation_manager()
        assert manager1 is manager2

    def test_configure_impersonation_manager(self):
        """Should configure new manager with options."""
        callback = MagicMock()
        manager = configure_impersonation_manager(
            audit_callback=callback,
            max_concurrent_sessions=5,
        )
        assert manager._max_concurrent_sessions == 5
        assert manager._audit_callback is callback
