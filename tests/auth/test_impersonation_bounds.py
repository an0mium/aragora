"""Tests for bounded audit log in ImpersonationManager."""

import pytest
from collections import deque
from datetime import datetime, timezone, timedelta

from aragora.auth.impersonation import (
    ImpersonationAuditEntry,
    ImpersonationManager,
)


class TestAuditLogBounds:
    """Tests for bounded audit log using deque."""

    @pytest.fixture
    def manager(self):
        """Create a manager with persistence disabled."""
        manager = ImpersonationManager()
        manager._use_persistence = False
        return manager

    def test_audit_log_is_deque_with_maxlen(self, manager: ImpersonationManager):
        """Audit log should be a deque with maxlen set."""
        assert isinstance(manager._audit_log, deque)
        assert manager._audit_log.maxlen == 5000

    def test_audit_log_respects_maxlen(self, manager: ImpersonationManager):
        """Audit log should not exceed maxlen."""
        # Add more entries than maxlen
        for i in range(6000):
            entry = ImpersonationAuditEntry(
                timestamp=datetime.now(timezone.utc),
                event_type="test",
                session_id=f"session_{i}",
                admin_user_id="admin1",
                target_user_id="user1",
                reason="Testing bounds",
                action_details={"index": i},
                ip_address="127.0.0.1",
                user_agent="TestAgent/1.0",
                success=True,
            )
            manager._audit_log.append(entry)

        # Should be capped at maxlen
        assert len(manager._audit_log) == 5000

    def test_old_entries_evicted_when_maxlen_reached(self, manager: ImpersonationManager):
        """Old entries should be evicted when maxlen is reached."""
        # Add exactly maxlen entries
        for i in range(5000):
            entry = ImpersonationAuditEntry(
                timestamp=datetime.now(timezone.utc),
                event_type="test",
                session_id=f"session_{i}",
                admin_user_id="admin1",
                target_user_id="user1",
                reason="Testing bounds",
                action_details={"index": i},
                ip_address="127.0.0.1",
                user_agent="TestAgent/1.0",
                success=True,
            )
            manager._audit_log.append(entry)

        # Verify first entry is session_0
        assert manager._audit_log[0].session_id == "session_0"

        # Add one more entry
        new_entry = ImpersonationAuditEntry(
            timestamp=datetime.now(timezone.utc),
            event_type="test",
            session_id="session_5000",
            admin_user_id="admin1",
            target_user_id="user1",
            reason="Testing bounds",
            action_details={"index": 5000},
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
            success=True,
        )
        manager._audit_log.append(new_entry)

        # Now the first entry should be session_1 (session_0 was evicted)
        assert len(manager._audit_log) == 5000
        assert manager._audit_log[0].session_id == "session_1"
        assert manager._audit_log[-1].session_id == "session_5000"

    def test_audit_logging_works_correctly(self, manager: ImpersonationManager):
        """Audit logging via _log_audit should work correctly."""
        entry = ImpersonationAuditEntry(
            timestamp=datetime.now(timezone.utc),
            event_type="test",
            session_id="test_session",
            admin_user_id="admin1",
            target_user_id="user1",
            reason="Testing audit logging",
            action_details={"test": True},
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
            success=True,
        )

        manager._log_audit(entry)

        assert len(manager._audit_log) == 1
        assert manager._audit_log[0] == entry

    def test_audit_log_via_start_impersonation(self, manager: ImpersonationManager):
        """Audit log entries created via start_impersonation should be stored."""
        session, message = manager.start_impersonation(
            admin_user_id="admin1",
            admin_email="admin@example.com",
            admin_roles=["admin"],
            target_user_id="user1",
            target_email="user@example.com",
            target_roles=["user"],
            reason="Testing impersonation audit",
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
        )

        assert session is not None
        assert len(manager._audit_log) == 1
        assert manager._audit_log[0].event_type == "start"
        assert manager._audit_log[0].admin_user_id == "admin1"
        assert manager._audit_log[0].target_user_id == "user1"

    def test_get_audit_log_works_with_deque(self, manager: ImpersonationManager):
        """get_audit_log should work correctly with deque backend."""
        # Add some entries
        for i in range(10):
            entry = ImpersonationAuditEntry(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=10 - i),
                event_type="test",
                session_id=f"session_{i}",
                admin_user_id="admin1",
                target_user_id="user1",
                reason="Testing",
                action_details={"index": i},
                ip_address="127.0.0.1",
                user_agent="TestAgent/1.0",
                success=True,
            )
            manager._audit_log.append(entry)

        # Get audit log (should return newest first)
        result = manager.get_audit_log(limit=5)

        assert len(result) == 5
        # Newest first (reversed order)
        assert result[0].session_id == "session_9"
        assert result[4].session_id == "session_5"

    def test_get_audit_log_filters_work_with_deque(self, manager: ImpersonationManager):
        """Filtering in get_audit_log should work with deque backend."""
        # Add entries for different admins
        for i in range(5):
            entry = ImpersonationAuditEntry(
                timestamp=datetime.now(timezone.utc),
                event_type="test",
                session_id=f"session_{i}",
                admin_user_id=f"admin{i % 2}",  # Alternates between admin0 and admin1
                target_user_id="user1",
                reason="Testing",
                action_details={"index": i},
                ip_address="127.0.0.1",
                user_agent="TestAgent/1.0",
                success=True,
            )
            manager._audit_log.append(entry)

        # Filter by admin_user_id
        result = manager.get_audit_log(admin_user_id="admin0")

        assert len(result) == 3  # indices 0, 2, 4
        for entry in result:
            assert entry.admin_user_id == "admin0"
