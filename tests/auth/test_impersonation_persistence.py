"""Tests for impersonation session and audit log persistence."""

import os
import tempfile
from datetime import datetime, timedelta

import pytest


@pytest.fixture
def temp_db_path():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    # Cleanup
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def impersonation_store(temp_db_path):
    """Create a fresh impersonation store."""
    from aragora.storage.impersonation_store import ImpersonationStore

    store = ImpersonationStore(db_path=temp_db_path, backend="sqlite")
    yield store
    store.close()


@pytest.fixture
def impersonation_manager(impersonation_store):
    """Create an impersonation manager with persistence."""
    from aragora.auth.impersonation import (
        ImpersonationManager,
        clear_impersonation_sessions,
        reset_session_recovery,
    )

    # Reset state
    reset_session_recovery()
    clear_impersonation_sessions()

    manager = ImpersonationManager()
    manager._store = impersonation_store
    manager._use_persistence = True
    yield manager

    # Cleanup
    reset_session_recovery()
    clear_impersonation_sessions()


class TestImpersonationStore:
    """Tests for ImpersonationStore directly."""

    def test_save_and_get_session(self, impersonation_store):
        """Test saving and retrieving a session."""
        now = datetime.utcnow()
        expires = now + timedelta(hours=1)

        impersonation_store.save_session(
            session_id="test-session-1",
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            target_user_id="user-456",
            target_email="user@example.com",
            reason="Testing impersonation",
            started_at=now,
            expires_at=expires,
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
        )

        session = impersonation_store.get_session("test-session-1")
        assert session is not None
        assert session.session_id == "test-session-1"
        assert session.admin_user_id == "admin-123"
        assert session.target_user_id == "user-456"
        assert session.reason == "Testing impersonation"
        assert session.ended_at is None

    def test_get_active_sessions(self, impersonation_store):
        """Test getting active sessions."""
        now = datetime.utcnow()
        expires = now + timedelta(hours=1)

        # Create two active sessions
        impersonation_store.save_session(
            session_id="active-1",
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            target_user_id="user-1",
            target_email="user1@example.com",
            reason="Active session 1",
            started_at=now,
            expires_at=expires,
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
        )

        impersonation_store.save_session(
            session_id="active-2",
            admin_user_id="admin-456",
            admin_email="admin2@example.com",
            target_user_id="user-2",
            target_email="user2@example.com",
            reason="Active session 2",
            started_at=now,
            expires_at=expires,
            ip_address="192.168.1.2",
            user_agent="TestAgent/1.0",
        )

        active = impersonation_store.get_active_sessions()
        assert len(active) == 2

        # Filter by admin
        admin_sessions = impersonation_store.get_active_sessions(admin_user_id="admin-123")
        assert len(admin_sessions) == 1
        assert admin_sessions[0].session_id == "active-1"

    def test_end_session(self, impersonation_store):
        """Test ending a session."""
        now = datetime.utcnow()
        expires = now + timedelta(hours=1)

        impersonation_store.save_session(
            session_id="to-end",
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            target_user_id="user-456",
            target_email="user@example.com",
            reason="Session to end",
            started_at=now,
            expires_at=expires,
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
            actions_performed=5,
        )

        impersonation_store.end_session(
            session_id="to-end",
            ended_by="admin",
            actions_performed=10,
        )

        session = impersonation_store.get_session("to-end")
        assert session is not None
        assert session.ended_at is not None
        assert session.ended_by == "admin"
        assert session.actions_performed == 10

        # Should not appear in active sessions
        active = impersonation_store.get_active_sessions()
        assert len(active) == 0

    def test_save_and_query_audit_log(self, impersonation_store):
        """Test saving and querying audit entries."""
        now = datetime.utcnow()

        # Save some audit entries
        impersonation_store.save_audit_entry(
            audit_id="audit-1",
            timestamp=now,
            event_type="start",
            admin_user_id="admin-123",
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
            success=True,
            session_id="session-1",
            target_user_id="user-456",
            reason="Starting impersonation",
        )

        impersonation_store.save_audit_entry(
            audit_id="audit-2",
            timestamp=now,
            event_type="action",
            admin_user_id="admin-123",
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
            success=True,
            session_id="session-1",
            target_user_id="user-456",
            action_details={"action_type": "view_profile"},
        )

        impersonation_store.save_audit_entry(
            audit_id="audit-3",
            timestamp=now,
            event_type="denied",
            admin_user_id="admin-789",
            ip_address="192.168.1.2",
            user_agent="TestAgent/1.0",
            success=False,
            target_user_id="user-456",
            error_message="2FA required",
        )

        # Query all
        all_entries = impersonation_store.get_audit_log(limit=100)
        assert len(all_entries) == 3

        # Filter by admin
        admin_entries = impersonation_store.get_audit_log(admin_user_id="admin-123")
        assert len(admin_entries) == 2

        # Filter by event type
        denied_entries = impersonation_store.get_audit_log(event_type="denied")
        assert len(denied_entries) == 1
        assert denied_entries[0].error_message == "2FA required"

        # Filter by session
        session_entries = impersonation_store.get_audit_log(session_id="session-1")
        assert len(session_entries) == 2

    def test_cleanup_expired_sessions(self, impersonation_store):
        """Test cleanup of expired sessions."""
        now = datetime.utcnow()
        expired = now - timedelta(hours=1)  # Already expired
        valid = now + timedelta(hours=1)

        # Create one expired and one valid session
        impersonation_store.save_session(
            session_id="expired-session",
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            target_user_id="user-1",
            target_email="user1@example.com",
            reason="Expired session",
            started_at=expired - timedelta(hours=1),
            expires_at=expired,
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
        )

        impersonation_store.save_session(
            session_id="valid-session",
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            target_user_id="user-2",
            target_email="user2@example.com",
            reason="Valid session",
            started_at=now,
            expires_at=valid,
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
        )

        # Cleanup expired
        count = impersonation_store.cleanup_expired_sessions()
        assert count == 1

        # Check that expired session is marked as ended
        expired_session = impersonation_store.get_session("expired-session")
        assert expired_session.ended_at is not None
        assert expired_session.ended_by == "timeout"

        # Valid session should still be active
        active = impersonation_store.get_active_sessions()
        assert len(active) == 1
        assert active[0].session_id == "valid-session"


class TestImpersonationManagerPersistence:
    """Tests for ImpersonationManager with persistence."""

    def test_session_persists_on_start(self, impersonation_manager, impersonation_store):
        """Test that starting a session persists it."""
        session, message = impersonation_manager.start_impersonation(
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user-456",
            target_email="user@example.com",
            target_roles=["user"],
            reason="Testing persistence of session start",
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
        )

        assert session is not None

        # Check it's in the store
        stored = impersonation_store.get_session(session.session_id)
        assert stored is not None
        assert stored.admin_user_id == "admin-123"
        assert stored.target_user_id == "user-456"

    def test_session_persists_on_end(self, impersonation_manager, impersonation_store):
        """Test that ending a session persists the end state."""
        session, _ = impersonation_manager.start_impersonation(
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user-456",
            target_email="user@example.com",
            target_roles=["user"],
            reason="Testing persistence of session end",
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
        )

        assert session is not None
        session_id = session.session_id

        # End the session
        success, _ = impersonation_manager.end_impersonation(
            session_id=session_id,
            admin_user_id="admin-123",
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
        )
        assert success

        # Check it's ended in the store
        stored = impersonation_store.get_session(session_id)
        assert stored is not None
        assert stored.ended_at is not None
        assert stored.ended_by == "admin"

    def test_audit_entries_persisted(self, impersonation_manager, impersonation_store):
        """Test that audit entries are persisted."""
        session, _ = impersonation_manager.start_impersonation(
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user-456",
            target_email="user@example.com",
            target_roles=["user"],
            reason="Testing audit persistence",
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
        )

        # Log an action
        if session:
            impersonation_manager.log_impersonation_action(
                session_id=session.session_id,
                action_type="view_profile",
                action_details={"page": "/profile"},
                ip_address="192.168.1.1",
                user_agent="TestAgent/1.0",
            )

        # Check audit entries in store
        entries = impersonation_store.get_audit_log(admin_user_id="admin-123")
        assert len(entries) >= 2  # At least start and action

        event_types = [e.event_type for e in entries]
        assert "start" in event_types
        assert "action" in event_types

    def test_action_count_persisted(self, impersonation_manager, impersonation_store):
        """Test that action count is persisted."""
        session, _ = impersonation_manager.start_impersonation(
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user-456",
            target_email="user@example.com",
            target_roles=["user"],
            reason="Testing action count persistence",
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
        )

        assert session is not None

        # Log multiple actions
        for i in range(3):
            impersonation_manager.log_impersonation_action(
                session_id=session.session_id,
                action_type="action",
                action_details={"index": i},
                ip_address="192.168.1.1",
                user_agent="TestAgent/1.0",
            )

        # Check action count in store
        stored = impersonation_store.get_session(session.session_id)
        assert stored is not None
        assert stored.actions_performed == 3


class TestSessionRecovery:
    """Tests for session recovery on startup."""

    def test_recover_sessions_from_store(self, temp_db_path):
        """Test recovering sessions from the store after restart."""
        from aragora.auth.impersonation import (
            ImpersonationManager,
            ImpersonationSession,
            clear_impersonation_sessions,
            recover_impersonation_sessions,
            reset_session_recovery,
        )
        from aragora.storage.impersonation_store import ImpersonationStore

        # Reset state
        reset_session_recovery()
        clear_impersonation_sessions()

        # Create store and manager
        store = ImpersonationStore(db_path=temp_db_path, backend="sqlite")

        # Create a session directly in the store (simulating previous run)
        now = datetime.utcnow()
        expires = now + timedelta(hours=1)

        store.save_session(
            session_id="persisted-session",
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            target_user_id="user-456",
            target_email="user@example.com",
            reason="Session from previous run",
            started_at=now,
            expires_at=expires,
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
            actions_performed=5,
        )

        # Create manager and inject store
        manager = ImpersonationManager()
        manager._store = store
        manager._use_persistence = True

        # Manually trigger recovery by calling the function
        # First, we need to replace the global manager temporarily
        import aragora.auth.impersonation as imp_module

        old_manager = imp_module._impersonation_manager
        imp_module._impersonation_manager = manager

        try:
            # Recover
            recovered = recover_impersonation_sessions()
            assert recovered == 1

            # Check session is in memory
            session = manager.validate_session("persisted-session")
            assert session is not None
            assert session.admin_user_id == "admin-123"
            assert session.target_user_id == "user-456"
            assert session.actions_performed == 5

            # Check admin sessions mapping
            assert "admin-123" in manager._admin_sessions
            assert "persisted-session" in manager._admin_sessions["admin-123"]

        finally:
            imp_module._impersonation_manager = old_manager
            store.close()
            reset_session_recovery()
            clear_impersonation_sessions()

    def test_recovery_is_idempotent(self, temp_db_path):
        """Test that recovery can be called multiple times safely."""
        from aragora.auth.impersonation import (
            ImpersonationManager,
            clear_impersonation_sessions,
            recover_impersonation_sessions,
            reset_session_recovery,
        )
        from aragora.storage.impersonation_store import ImpersonationStore

        # Reset state
        reset_session_recovery()
        clear_impersonation_sessions()

        store = ImpersonationStore(db_path=temp_db_path, backend="sqlite")

        now = datetime.utcnow()
        expires = now + timedelta(hours=1)

        store.save_session(
            session_id="idempotent-session",
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            target_user_id="user-456",
            target_email="user@example.com",
            reason="Testing idempotent recovery",
            started_at=now,
            expires_at=expires,
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
        )

        manager = ImpersonationManager()
        manager._store = store
        manager._use_persistence = True

        import aragora.auth.impersonation as imp_module

        old_manager = imp_module._impersonation_manager
        imp_module._impersonation_manager = manager

        try:
            # First recovery
            recovered1 = recover_impersonation_sessions()
            assert recovered1 == 1

            # Second recovery should return 0 (already recovered)
            recovered2 = recover_impersonation_sessions()
            assert recovered2 == 0

            # Should still have only one session
            assert len(manager._sessions) == 1

        finally:
            imp_module._impersonation_manager = old_manager
            store.close()
            reset_session_recovery()
            clear_impersonation_sessions()

    def test_recovery_skips_expired_sessions(self, temp_db_path):
        """Test that expired sessions are not recovered."""
        from aragora.auth.impersonation import (
            ImpersonationManager,
            clear_impersonation_sessions,
            recover_impersonation_sessions,
            reset_session_recovery,
        )
        from aragora.storage.impersonation_store import ImpersonationStore

        reset_session_recovery()
        clear_impersonation_sessions()

        store = ImpersonationStore(db_path=temp_db_path, backend="sqlite")

        now = datetime.utcnow()
        expired = now - timedelta(hours=1)
        valid = now + timedelta(hours=1)

        # Expired session
        store.save_session(
            session_id="expired-session",
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            target_user_id="user-1",
            target_email="user1@example.com",
            reason="Expired session",
            started_at=expired - timedelta(hours=1),
            expires_at=expired,
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
        )

        # Valid session
        store.save_session(
            session_id="valid-session",
            admin_user_id="admin-456",
            admin_email="admin2@example.com",
            target_user_id="user-2",
            target_email="user2@example.com",
            reason="Valid session",
            started_at=now,
            expires_at=valid,
            ip_address="192.168.1.2",
            user_agent="TestAgent/1.0",
        )

        manager = ImpersonationManager()
        manager._store = store
        manager._use_persistence = True

        import aragora.auth.impersonation as imp_module

        old_manager = imp_module._impersonation_manager
        imp_module._impersonation_manager = manager

        try:
            # Recovery should only get the valid session
            recovered = recover_impersonation_sessions()
            assert recovered == 1

            # Check only valid session is in memory
            assert "valid-session" in manager._sessions
            assert "expired-session" not in manager._sessions

        finally:
            imp_module._impersonation_manager = old_manager
            store.close()
            reset_session_recovery()
            clear_impersonation_sessions()
