"""
Tests for ImpersonationStore - Database persistence for impersonation sessions and audit logs.

Tests cover:
- ImpersonationStore initialization with SQLite backend
- Session management (save, get, update, end)
- Active session retrieval and expiration handling
- Session revocation
- Audit entry logging for impersonation events
- Cleanup of expired sessions
- Data integrity (save then retrieve)
- Dataclass serialization
- Factory functions (get_impersonation_store, reset_impersonation_store)
"""

from __future__ import annotations

import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.storage.impersonation_store import (
    AuditRecord,
    ImpersonationStore,
    SessionRecord,
    get_impersonation_store,
    reset_impersonation_store,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_impersonation.db"


@pytest.fixture
def impersonation_store(temp_db_path):
    """Create an impersonation store for testing."""
    store = ImpersonationStore(db_path=str(temp_db_path), backend="sqlite")
    yield store
    store.close()


@pytest.fixture
def sample_session_data():
    """Create sample session data."""
    now = datetime.now(timezone.utc)
    return {
        "session_id": "session-001",
        "admin_user_id": "admin-123",
        "admin_email": "admin@example.com",
        "target_user_id": "user-456",
        "target_email": "user@example.com",
        "reason": "Support ticket #12345",
        "started_at": now,
        "expires_at": now + timedelta(hours=1),
        "ip_address": "192.168.1.100",
        "user_agent": "Mozilla/5.0 Chrome/120.0",
        "actions_performed": 0,
    }


@pytest.fixture
def populated_store(impersonation_store, sample_session_data):
    """Impersonation store with sample sessions and audit entries."""
    now = datetime.now(timezone.utc)

    # Active session
    impersonation_store.save_session(**sample_session_data)

    # Session from another admin
    impersonation_store.save_session(
        session_id="session-002",
        admin_user_id="admin-789",
        admin_email="admin2@example.com",
        target_user_id="user-abc",
        target_email="another@example.com",
        reason="Investigation",
        started_at=now - timedelta(minutes=30),
        expires_at=now + timedelta(minutes=30),
        ip_address="10.0.0.1",
        user_agent="Safari/17.0",
    )

    # Audit entries
    impersonation_store.save_audit_entry(
        audit_id="audit-001",
        timestamp=now,
        event_type="start",
        admin_user_id="admin-123",
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0",
        success=True,
        session_id="session-001",
        target_user_id="user-456",
        reason="Support ticket #12345",
    )

    impersonation_store.save_audit_entry(
        audit_id="audit-002",
        timestamp=now + timedelta(seconds=10),
        event_type="action",
        admin_user_id="admin-123",
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0",
        success=True,
        session_id="session-001",
        action_details={"endpoint": "/api/users", "method": "GET"},
    )

    impersonation_store.save_audit_entry(
        audit_id="audit-003",
        timestamp=now - timedelta(minutes=10),
        event_type="denied",
        admin_user_id="admin-999",
        ip_address="10.20.30.40",
        user_agent="curl/7.88.0",
        success=False,
        error_message="Insufficient permissions",
    )

    return impersonation_store


# =============================================================================
# SessionRecord Dataclass Tests
# =============================================================================


class TestSessionRecord:
    """Tests for SessionRecord dataclass."""

    def test_is_expired_false_for_future(self):
        """Should return False for non-expired session."""
        session = SessionRecord(
            session_id="test",
            admin_user_id="admin",
            admin_email="admin@test.com",
            target_user_id="user",
            target_email="user@test.com",
            reason="test",
            started_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            ip_address="127.0.0.1",
            user_agent="test",
        )
        assert session.is_expired() is False

    def test_is_expired_true_for_past(self):
        """Should return True for expired session."""
        session = SessionRecord(
            session_id="test",
            admin_user_id="admin",
            admin_email="admin@test.com",
            target_user_id="user",
            target_email="user@test.com",
            reason="test",
            started_at=datetime.now(timezone.utc) - timedelta(hours=2),
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
            ip_address="127.0.0.1",
            user_agent="test",
        )
        assert session.is_expired() is True

    def test_to_dict_basic_fields(self):
        """Test to_dict includes all basic fields."""
        now = datetime.now(timezone.utc)
        session = SessionRecord(
            session_id="session-001",
            admin_user_id="admin-123",
            admin_email="admin@test.com",
            target_user_id="user-456",
            target_email="user@test.com",
            reason="Support ticket",
            started_at=now,
            expires_at=now + timedelta(hours=1),
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            actions_performed=5,
        )

        result = session.to_dict()

        assert result["session_id"] == "session-001"
        assert result["admin_user_id"] == "admin-123"
        assert result["target_user_id"] == "user-456"
        assert result["actions_performed"] == 5
        assert result["ended_at"] is None
        assert result["ended_by"] is None

    def test_to_dict_with_ended_session(self):
        """Test to_dict includes ended_at and ended_by when present."""
        now = datetime.now(timezone.utc)
        session = SessionRecord(
            session_id="session-001",
            admin_user_id="admin-123",
            admin_email="admin@test.com",
            target_user_id="user-456",
            target_email="user@test.com",
            reason="Support",
            started_at=now - timedelta(hours=1),
            expires_at=now + timedelta(hours=1),
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            ended_at=now,
            ended_by="admin",
        )

        result = session.to_dict()

        assert result["ended_at"] is not None
        assert result["ended_by"] == "admin"


# =============================================================================
# AuditRecord Dataclass Tests
# =============================================================================


class TestAuditRecord:
    """Tests for AuditRecord dataclass."""

    def test_to_dict_basic(self):
        """Test to_dict for basic audit record."""
        now = datetime.now(timezone.utc)
        record = AuditRecord(
            audit_id="audit-001",
            timestamp=now,
            event_type="start",
            session_id="session-001",
            admin_user_id="admin-123",
            target_user_id="user-456",
            reason="Support",
            action_details_json=None,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            success=True,
        )

        result = record.to_dict()

        assert result["audit_id"] == "audit-001"
        assert result["event_type"] == "start"
        assert result["success"] is True
        assert result["action_details"] is None
        assert result["error_message"] is None

    def test_to_dict_with_action_details(self):
        """Test to_dict deserializes action_details JSON."""
        record = AuditRecord(
            audit_id="audit-001",
            timestamp=datetime.now(timezone.utc),
            event_type="action",
            session_id="session-001",
            admin_user_id="admin-123",
            target_user_id=None,
            reason=None,
            action_details_json='{"endpoint": "/api/users", "method": "GET"}',
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            success=True,
        )

        result = record.to_dict()

        assert result["action_details"]["endpoint"] == "/api/users"
        assert result["action_details"]["method"] == "GET"

    def test_to_dict_with_error(self):
        """Test to_dict includes error_message."""
        record = AuditRecord(
            audit_id="audit-001",
            timestamp=datetime.now(timezone.utc),
            event_type="denied",
            session_id=None,
            admin_user_id="admin-123",
            target_user_id=None,
            reason=None,
            action_details_json=None,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            success=False,
            error_message="Insufficient permissions",
        )

        result = record.to_dict()

        assert result["success"] is False
        assert result["error_message"] == "Insufficient permissions"


# =============================================================================
# ImpersonationStore Initialization Tests
# =============================================================================


class TestImpersonationStoreInit:
    """Tests for ImpersonationStore initialization."""

    def test_init_with_sqlite_backend(self, temp_db_path):
        """Should initialize with SQLite backend."""
        store = ImpersonationStore(db_path=str(temp_db_path), backend="sqlite")

        assert store.backend_type == "sqlite"
        assert store._backend is not None

        store.close()

    def test_init_creates_tables(self, temp_db_path):
        """Should create required tables on init."""
        store = ImpersonationStore(db_path=str(temp_db_path), backend="sqlite")

        # Try to query tables to verify they exist
        sessions = store.get_active_sessions()
        assert sessions == []

        audit_log = store.get_audit_log()
        assert audit_log == []

        store.close()

    def test_init_postgresql_requires_url(self, temp_db_path):
        """PostgreSQL backend requires database_url."""
        with pytest.raises(ValueError, match="PostgreSQL backend requires DATABASE_URL"):
            ImpersonationStore(db_path=str(temp_db_path), backend="postgresql")


# =============================================================================
# Session Management Tests
# =============================================================================


class TestSessionManagement:
    """Tests for session CRUD operations."""

    def test_save_and_get_session(self, impersonation_store, sample_session_data):
        """Test save and retrieve a session."""
        session_id = impersonation_store.save_session(**sample_session_data)

        assert session_id == "session-001"

        session = impersonation_store.get_session(session_id)
        assert session is not None
        assert session.session_id == "session-001"
        assert session.admin_user_id == "admin-123"
        assert session.target_user_id == "user-456"
        assert session.reason == "Support ticket #12345"

    def test_get_nonexistent_session(self, impersonation_store):
        """Test get returns None for nonexistent session."""
        result = impersonation_store.get_session("nonexistent-id")
        assert result is None

    def test_save_updates_existing_session(self, impersonation_store, sample_session_data):
        """Test save updates existing session (upsert)."""
        impersonation_store.save_session(**sample_session_data)

        # Update actions count
        sample_session_data["actions_performed"] = 10
        impersonation_store.save_session(**sample_session_data)

        session = impersonation_store.get_session("session-001")
        assert session.actions_performed == 10

    def test_update_session_actions(self, impersonation_store, sample_session_data):
        """Test update_session_actions updates count."""
        impersonation_store.save_session(**sample_session_data)

        result = impersonation_store.update_session_actions("session-001", 5)
        assert result is True

        session = impersonation_store.get_session("session-001")
        assert session.actions_performed == 5

    def test_data_integrity_save_retrieve(self, impersonation_store, sample_session_data):
        """Test data integrity - save then retrieve returns same data."""
        impersonation_store.save_session(**sample_session_data)

        session = impersonation_store.get_session("session-001")

        assert session.session_id == sample_session_data["session_id"]
        assert session.admin_user_id == sample_session_data["admin_user_id"]
        assert session.admin_email == sample_session_data["admin_email"]
        assert session.target_user_id == sample_session_data["target_user_id"]
        assert session.target_email == sample_session_data["target_email"]
        assert session.reason == sample_session_data["reason"]
        assert session.ip_address == sample_session_data["ip_address"]
        assert session.user_agent == sample_session_data["user_agent"]
        # Timestamps may have slight differences due to parsing, but should be close
        assert abs((session.started_at - sample_session_data["started_at"]).total_seconds()) < 1
        assert abs((session.expires_at - sample_session_data["expires_at"]).total_seconds()) < 1


# =============================================================================
# Active Session Tests
# =============================================================================


class TestActiveSessions:
    """Tests for active session retrieval."""

    def test_get_active_sessions(self, populated_store):
        """Should return all active sessions."""
        sessions = populated_store.get_active_sessions()
        assert len(sessions) == 2

    def test_get_active_sessions_filter_by_admin(self, populated_store):
        """Should filter active sessions by admin."""
        sessions = populated_store.get_active_sessions(admin_user_id="admin-123")
        assert len(sessions) == 1
        assert sessions[0].session_id == "session-001"

    def test_expired_sessions_not_returned(self, impersonation_store):
        """Expired sessions should not be returned as active."""
        now = datetime.now(timezone.utc)

        # Expired session
        impersonation_store.save_session(
            session_id="expired-session",
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            target_user_id="user-456",
            target_email="user@example.com",
            reason="Test",
            started_at=now - timedelta(hours=2),
            expires_at=now - timedelta(hours=1),  # Expired 1 hour ago
            ip_address="127.0.0.1",
            user_agent="test",
        )

        # Active session
        impersonation_store.save_session(
            session_id="active-session",
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            target_user_id="user-789",
            target_email="user2@example.com",
            reason="Test",
            started_at=now,
            expires_at=now + timedelta(hours=1),  # Expires in 1 hour
            ip_address="127.0.0.1",
            user_agent="test",
        )

        active = impersonation_store.get_active_sessions()
        assert len(active) == 1
        assert active[0].session_id == "active-session"

    def test_ended_sessions_not_returned_as_active(self, impersonation_store, sample_session_data):
        """Ended sessions should not be returned as active."""
        impersonation_store.save_session(**sample_session_data)
        impersonation_store.end_session("session-001", "admin", 5)

        active = impersonation_store.get_active_sessions()
        assert len(active) == 0


# =============================================================================
# Session Revocation Tests
# =============================================================================


class TestSessionRevocation:
    """Tests for session revocation (ending sessions)."""

    def test_end_session(self, impersonation_store, sample_session_data):
        """Test end_session marks session as ended."""
        impersonation_store.save_session(**sample_session_data)

        result = impersonation_store.end_session(
            session_id="session-001",
            ended_by="admin",
            actions_performed=10,
        )

        assert result is True

        session = impersonation_store.get_session("session-001")
        assert session.ended_at is not None
        assert session.ended_by == "admin"
        assert session.actions_performed == 10

    def test_end_session_by_timeout(self, impersonation_store, sample_session_data):
        """Test end_session with timeout reason."""
        impersonation_store.save_session(**sample_session_data)

        impersonation_store.end_session(
            session_id="session-001",
            ended_by="timeout",
            actions_performed=3,
        )

        session = impersonation_store.get_session("session-001")
        assert session.ended_by == "timeout"

    def test_end_session_by_system(self, impersonation_store, sample_session_data):
        """Test end_session with system reason."""
        impersonation_store.save_session(**sample_session_data)

        impersonation_store.end_session(
            session_id="session-001",
            ended_by="system",
            actions_performed=0,
        )

        session = impersonation_store.get_session("session-001")
        assert session.ended_by == "system"


# =============================================================================
# Sessions for Admin Tests
# =============================================================================


class TestSessionsForAdmin:
    """Tests for get_sessions_for_admin."""

    def test_get_sessions_for_admin(self, populated_store):
        """Should return sessions for specific admin."""
        sessions = populated_store.get_sessions_for_admin("admin-123")
        assert len(sessions) == 1
        assert sessions[0].admin_user_id == "admin-123"

    def test_get_sessions_for_admin_include_ended(self, impersonation_store, sample_session_data):
        """Should include ended sessions when requested."""
        impersonation_store.save_session(**sample_session_data)
        impersonation_store.end_session("session-001", "admin", 5)

        # Without ended
        active = impersonation_store.get_sessions_for_admin("admin-123", include_ended=False)
        assert len(active) == 0

        # With ended
        all_sessions = impersonation_store.get_sessions_for_admin("admin-123", include_ended=True)
        assert len(all_sessions) == 1

    def test_get_sessions_for_admin_respects_limit(self, impersonation_store):
        """Should respect limit parameter."""
        now = datetime.now(timezone.utc)

        for i in range(10):
            impersonation_store.save_session(
                session_id=f"session-{i:03d}",
                admin_user_id="admin-123",
                admin_email="admin@example.com",
                target_user_id=f"user-{i:03d}",
                target_email=f"user{i}@example.com",
                reason=f"Test {i}",
                started_at=now,
                expires_at=now + timedelta(hours=1),
                ip_address="127.0.0.1",
                user_agent="test",
            )

        sessions = impersonation_store.get_sessions_for_admin("admin-123", limit=5)
        assert len(sessions) == 5


# =============================================================================
# Audit Entry Tests
# =============================================================================


class TestAuditEntries:
    """Tests for audit entry logging and retrieval."""

    def test_save_audit_entry(self, impersonation_store):
        """Test save an audit entry."""
        now = datetime.now(timezone.utc)
        audit_id = impersonation_store.save_audit_entry(
            audit_id="audit-001",
            timestamp=now,
            event_type="start",
            admin_user_id="admin-123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            success=True,
            session_id="session-001",
            target_user_id="user-456",
            reason="Support ticket",
        )

        assert audit_id == "audit-001"

    def test_save_audit_entry_with_action_details(self, impersonation_store):
        """Test save audit entry with action details dict."""
        now = datetime.now(timezone.utc)
        impersonation_store.save_audit_entry(
            audit_id="audit-action",
            timestamp=now,
            event_type="action",
            admin_user_id="admin-123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            success=True,
            action_details={"endpoint": "/api/users/123", "method": "PUT"},
        )

        logs = impersonation_store.get_audit_log()
        assert len(logs) == 1
        action_details = logs[0].to_dict()["action_details"]
        assert action_details["endpoint"] == "/api/users/123"

    def test_get_audit_log_all(self, populated_store):
        """Should return all audit entries when no filters."""
        logs = populated_store.get_audit_log()
        assert len(logs) == 3

    def test_get_audit_log_filter_by_admin(self, populated_store):
        """Should filter audit log by admin."""
        logs = populated_store.get_audit_log(admin_user_id="admin-123")
        assert len(logs) == 2
        assert all(log.admin_user_id == "admin-123" for log in logs)

    def test_get_audit_log_filter_by_target(self, populated_store):
        """Should filter audit log by target user."""
        logs = populated_store.get_audit_log(target_user_id="user-456")
        assert len(logs) == 1

    def test_get_audit_log_filter_by_session(self, populated_store):
        """Should filter audit log by session ID."""
        logs = populated_store.get_audit_log(session_id="session-001")
        assert len(logs) == 2

    def test_get_audit_log_filter_by_event_type(self, populated_store):
        """Should filter audit log by event type."""
        logs = populated_store.get_audit_log(event_type="start")
        assert len(logs) == 1
        assert logs[0].event_type == "start"

    def test_get_audit_log_filter_by_since(self, impersonation_store):
        """Should filter audit entries after a timestamp."""
        now = datetime.now(timezone.utc)

        impersonation_store.save_audit_entry(
            audit_id="old-audit",
            timestamp=now - timedelta(hours=2),
            event_type="start",
            admin_user_id="admin",
            ip_address="127.0.0.1",
            user_agent="test",
            success=True,
        )

        impersonation_store.save_audit_entry(
            audit_id="new-audit",
            timestamp=now,
            event_type="end",
            admin_user_id="admin",
            ip_address="127.0.0.1",
            user_agent="test",
            success=True,
        )

        logs = impersonation_store.get_audit_log(since=now - timedelta(hours=1))
        assert len(logs) == 1
        assert logs[0].audit_id == "new-audit"

    def test_get_audit_log_pagination(self, impersonation_store):
        """Should respect limit parameter."""
        now = datetime.now(timezone.utc)

        for i in range(10):
            impersonation_store.save_audit_entry(
                audit_id=f"audit-{i:03d}",
                timestamp=now + timedelta(seconds=i),
                event_type="action",
                admin_user_id="admin",
                ip_address="127.0.0.1",
                user_agent="test",
                success=True,
            )

        logs = impersonation_store.get_audit_log(limit=5)
        assert len(logs) == 5


# =============================================================================
# Cleanup Tests
# =============================================================================


class TestCleanup:
    """Tests for session cleanup operations."""

    def test_cleanup_expired_sessions(self, impersonation_store):
        """Test cleanup_expired_sessions marks expired sessions as ended."""
        now = datetime.now(timezone.utc)

        # Expired session
        impersonation_store.save_session(
            session_id="expired-session",
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            target_user_id="user-456",
            target_email="user@example.com",
            reason="Test",
            started_at=now - timedelta(hours=2),
            expires_at=now - timedelta(hours=1),
            ip_address="127.0.0.1",
            user_agent="test",
        )

        # Active session
        impersonation_store.save_session(
            session_id="active-session",
            admin_user_id="admin-123",
            admin_email="admin@example.com",
            target_user_id="user-789",
            target_email="user2@example.com",
            reason="Test",
            started_at=now,
            expires_at=now + timedelta(hours=1),
            ip_address="127.0.0.1",
            user_agent="test",
        )

        count = impersonation_store.cleanup_expired_sessions()
        assert count == 1

        expired_session = impersonation_store.get_session("expired-session")
        assert expired_session.ended_at is not None
        assert expired_session.ended_by == "timeout"

        active_session = impersonation_store.get_session("active-session")
        assert active_session.ended_at is None

    def test_cleanup_expired_returns_zero_when_none_expired(
        self, impersonation_store, sample_session_data
    ):
        """Should return 0 when no sessions expired."""
        impersonation_store.save_session(**sample_session_data)

        count = impersonation_store.cleanup_expired_sessions()
        assert count == 0

    def test_cleanup_old_records_sessions(self, temp_db_path):
        """Should delete old ended sessions."""
        import sqlite3

        store = ImpersonationStore(db_path=str(temp_db_path), backend="sqlite")

        now = datetime.now(timezone.utc)
        old_date = (now - timedelta(days=100)).isoformat()

        # Insert old ended session directly
        conn = sqlite3.connect(str(temp_db_path))
        conn.execute(
            """
            INSERT INTO impersonation_sessions
            (session_id, admin_user_id, admin_email, target_user_id, target_email,
             reason, started_at, expires_at, ip_address, user_agent, ended_at, ended_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "old-session",
                "admin",
                "admin@test.com",
                "user",
                "user@test.com",
                "Old test",
                old_date,
                old_date,
                "127.0.0.1",
                "test",
                old_date,
                "timeout",
            ),
        )
        conn.commit()
        conn.close()

        counts = store.cleanup_old_records(sessions_days=50, audit_days=365)
        assert counts["sessions"] == 1

        store.close()

    def test_cleanup_old_records_audit(self, temp_db_path):
        """Should delete old audit entries."""
        import sqlite3

        store = ImpersonationStore(db_path=str(temp_db_path), backend="sqlite")

        now = datetime.now(timezone.utc)
        old_date = (now - timedelta(days=400)).isoformat()

        # Insert old audit entry directly
        conn = sqlite3.connect(str(temp_db_path))
        conn.execute(
            """
            INSERT INTO impersonation_audit
            (audit_id, timestamp, event_type, admin_user_id, ip_address, user_agent, success)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("old-audit", old_date, "start", "admin", "127.0.0.1", "test", 1),
        )
        conn.commit()
        conn.close()

        counts = store.cleanup_old_records(sessions_days=90, audit_days=365)
        assert counts["audit"] == 1

        store.close()


# =============================================================================
# Concurrent Session Limits Tests
# =============================================================================


class TestConcurrentSessions:
    """Tests for concurrent session handling."""

    def test_multiple_active_sessions_for_same_admin(self, impersonation_store):
        """Should allow multiple active sessions for same admin."""
        now = datetime.now(timezone.utc)

        for i in range(3):
            impersonation_store.save_session(
                session_id=f"session-{i}",
                admin_user_id="admin-123",
                admin_email="admin@example.com",
                target_user_id=f"user-{i}",
                target_email=f"user{i}@example.com",
                reason=f"Reason {i}",
                started_at=now,
                expires_at=now + timedelta(hours=1),
                ip_address="127.0.0.1",
                user_agent="test",
            )

        sessions = impersonation_store.get_active_sessions(admin_user_id="admin-123")
        assert len(sessions) == 3

    def test_track_sessions_per_admin(self, impersonation_store):
        """Should track session count per admin correctly."""
        now = datetime.now(timezone.utc)

        # Admin 1: 2 sessions
        for i in range(2):
            impersonation_store.save_session(
                session_id=f"admin1-session-{i}",
                admin_user_id="admin-001",
                admin_email="admin1@example.com",
                target_user_id=f"user-a{i}",
                target_email=f"usera{i}@example.com",
                reason="Test",
                started_at=now,
                expires_at=now + timedelta(hours=1),
                ip_address="127.0.0.1",
                user_agent="test",
            )

        # Admin 2: 3 sessions
        for i in range(3):
            impersonation_store.save_session(
                session_id=f"admin2-session-{i}",
                admin_user_id="admin-002",
                admin_email="admin2@example.com",
                target_user_id=f"user-b{i}",
                target_email=f"userb{i}@example.com",
                reason="Test",
                started_at=now,
                expires_at=now + timedelta(hours=1),
                ip_address="127.0.0.1",
                user_agent="test",
            )

        admin1_sessions = impersonation_store.get_active_sessions(admin_user_id="admin-001")
        admin2_sessions = impersonation_store.get_active_sessions(admin_user_id="admin-002")

        assert len(admin1_sessions) == 2
        assert len(admin2_sessions) == 3


# =============================================================================
# Persistence Tests
# =============================================================================


class TestPersistence:
    """Tests for data persistence across store instances."""

    def test_data_persists_across_instances(self, temp_db_path, sample_session_data):
        """Data should persist after store is recreated."""
        # Create and populate first store
        store1 = ImpersonationStore(db_path=str(temp_db_path), backend="sqlite")
        store1.save_session(**sample_session_data)
        store1.close()

        # Create new store instance
        store2 = ImpersonationStore(db_path=str(temp_db_path), backend="sqlite")

        # Verify data persists
        session = store2.get_session("session-001")
        assert session is not None
        assert session.admin_user_id == "admin-123"

        store2.close()


# =============================================================================
# Factory Functions Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_reset_impersonation_store(self, temp_db_path):
        """Should reset the default store instance."""
        reset_impersonation_store()

        # Mock dependencies to avoid production guards
        with patch.dict(
            "os.environ",
            {
                "ARAGORA_DATA_DIR": str(temp_db_path.parent),
                "ARAGORA_ENVIRONMENT": "development",
                "ARAGORA_IMPERSONATION_STORE_BACKEND": "sqlite",
            },
        ):
            with patch(
                "aragora.storage.connection_factory.resolve_database_config"
            ) as mock_resolve:
                from aragora.storage.connection_factory import StorageBackendType

                mock_config = MagicMock()
                mock_config.backend_type = StorageBackendType.SQLITE
                mock_config.dsn = None
                mock_resolve.return_value = mock_config

                with patch("aragora.storage.production_guards.require_distributed_store"):
                    store1 = get_impersonation_store(db_path=str(temp_db_path))
                    reset_impersonation_store()
                    store2 = get_impersonation_store(db_path=str(temp_db_path))

                    # Should be different instances
                    assert store1 is not store2

        reset_impersonation_store()


# =============================================================================
# Close Tests
# =============================================================================


class TestClose:
    """Tests for close method."""

    def test_close_backend(self, temp_db_path):
        """Should close backend connection."""
        store = ImpersonationStore(db_path=str(temp_db_path), backend="sqlite")
        store.save_session(
            session_id="test",
            admin_user_id="admin",
            admin_email="admin@test.com",
            target_user_id="user",
            target_email="user@test.com",
            reason="test",
            started_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            ip_address="127.0.0.1",
            user_agent="test",
        )

        store.close()
        # After close, operations may fail but store should be in closed state


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_importable(self):
        """All items in __all__ should be importable."""
        import aragora.storage.impersonation_store as module

        for name in module.__all__:
            assert hasattr(module, name), f"Missing export: {name}"

    def test_key_exports(self):
        """Key exports should be available."""
        from aragora.storage.impersonation_store import (
            AuditRecord,
            ImpersonationStore,
            SessionRecord,
            get_impersonation_store,
            reset_impersonation_store,
        )

        assert ImpersonationStore is not None
        assert SessionRecord is not None
        assert AuditRecord is not None
        assert callable(get_impersonation_store)
        assert callable(reset_impersonation_store)
