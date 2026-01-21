"""
Tests for the SQLite-backed AuditStore.

Covers:
- Audit event logging
- Query operations with filters
- Pagination
- Cleanup operations
- Recent activity retrieval
- Security event filtering
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from aragora.storage.audit_store import AuditStore


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_audit.db"


@pytest.fixture
def db_with_schema(temp_db):
    """Create database with required schema."""
    conn = sqlite3.connect(str(temp_db))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_id TEXT,
            org_id TEXT,
            action TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            resource_id TEXT,
            old_value TEXT,
            new_value TEXT,
            metadata TEXT DEFAULT '{}',
            ip_address TEXT,
            user_agent TEXT
        )
    """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_log(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_org_id ON audit_log(org_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action)")
    conn.commit()
    conn.close()
    yield temp_db


@pytest.fixture
def store(db_with_schema):
    """Create an AuditStore instance."""
    return AuditStore(db_with_schema)


# =============================================================================
# Event Logging Tests
# =============================================================================


class TestEventLogging:
    """Tests for audit event logging."""

    def test_log_event_basic(self, store):
        """Test logging a basic audit event."""
        entry_id = store.log_event(
            action="user.login",
            resource_type="user",
            resource_id="user_123",
        )

        assert entry_id is not None
        assert entry_id > 0

    def test_log_event_full(self, store):
        """Test logging an event with all fields."""
        entry_id = store.log_event(
            action="subscription.created",
            resource_type="subscription",
            resource_id="sub_123",
            user_id="user_456",
            org_id="org_789",
            old_value=None,
            new_value={"plan": "professional", "price": 99},
            metadata={"source": "stripe_webhook"},
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0",
        )

        assert entry_id is not None

    def test_log_event_with_old_new_values(self, store):
        """Test logging an event with old and new values."""
        entry_id = store.log_event(
            action="tier.changed",
            resource_type="organization",
            resource_id="org_123",
            old_value={"tier": "free"},
            new_value={"tier": "professional"},
        )

        assert entry_id is not None

        # Retrieve and verify
        logs = store.get_log(action="tier.changed")
        assert len(logs) == 1
        assert logs[0]["old_value"] == {"tier": "free"}
        assert logs[0]["new_value"] == {"tier": "professional"}


class TestLogRetrieval:
    """Tests for audit log retrieval."""

    @pytest.fixture(autouse=True)
    def setup_logs(self, store):
        """Set up test logs."""
        # Create various log entries
        for i in range(5):
            store.log_event(
                action="user.login",
                resource_type="user",
                resource_id=f"user_{i}",
                user_id=f"user_{i}",
                org_id="org_1",
            )
        for i in range(3):
            store.log_event(
                action="subscription.updated",
                resource_type="subscription",
                resource_id=f"sub_{i}",
                user_id="user_0",
                org_id="org_1",
            )
        for i in range(2):
            store.log_event(
                action="user.login",
                resource_type="user",
                resource_id=f"user_org2_{i}",
                user_id=f"user_org2_{i}",
                org_id="org_2",
            )

    def test_get_log_all(self, store):
        """Test retrieving all logs."""
        logs = store.get_log()

        assert len(logs) == 10

    def test_get_log_by_org_id(self, store):
        """Test filtering logs by organization."""
        logs = store.get_log(org_id="org_1")

        assert len(logs) == 8

    def test_get_log_by_user_id(self, store):
        """Test filtering logs by user."""
        logs = store.get_log(user_id="user_0")

        assert len(logs) == 4  # 1 login + 3 subscription updates

    def test_get_log_by_action(self, store):
        """Test filtering logs by action."""
        logs = store.get_log(action="user.login")

        assert len(logs) == 7

    def test_get_log_by_action_prefix(self, store):
        """Test filtering logs by action prefix (wildcard)."""
        logs = store.get_log(action="subscription.*")

        assert len(logs) == 3

    def test_get_log_by_resource_type(self, store):
        """Test filtering logs by resource type."""
        logs = store.get_log(resource_type="subscription")

        assert len(logs) == 3

    def test_get_log_combined_filters(self, store):
        """Test combining multiple filters."""
        logs = store.get_log(
            org_id="org_1",
            user_id="user_0",
            action="subscription.*",
        )

        assert len(logs) == 3


class TestPagination:
    """Tests for log pagination."""

    @pytest.fixture(autouse=True)
    def setup_logs(self, store):
        """Set up many log entries for pagination tests."""
        for i in range(50):
            store.log_event(
                action=f"action_{i}",
                resource_type="test",
                resource_id=f"res_{i}",
            )

    def test_get_log_limit(self, store):
        """Test limiting results."""
        logs = store.get_log(limit=10)

        assert len(logs) == 10

    def test_get_log_offset(self, store):
        """Test offset pagination."""
        all_logs = store.get_log(limit=50)
        offset_logs = store.get_log(limit=10, offset=10)

        # Offset logs should be different from first 10
        first_10_ids = {log["id"] for log in all_logs[:10]}
        offset_ids = {log["id"] for log in offset_logs}

        assert first_10_ids.isdisjoint(offset_ids)

    def test_get_log_pagination(self, store):
        """Test paginating through all results."""
        page1 = store.get_log(limit=20, offset=0)
        page2 = store.get_log(limit=20, offset=20)
        page3 = store.get_log(limit=20, offset=40)

        # Should have all unique IDs
        all_ids = (
            {log["id"] for log in page1}
            | {log["id"] for log in page2}
            | {log["id"] for log in page3}
        )

        assert len(all_ids) == 50


class TestTimeFilters:
    """Tests for time-based filtering."""

    @pytest.fixture
    def store_with_old_logs(self, store):
        """Create store with logs at different times."""
        # We need to manually insert old logs since log_event uses utcnow()
        conn = store._get_connection()

        # Insert old log
        old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        conn.execute(
            """
            INSERT INTO audit_log
            (timestamp, user_id, org_id, action, resource_type, resource_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, '{}')
            """,
            (old_time, "user_1", "org_1", "old.action", "test", "old_1"),
        )

        # Insert recent log
        recent_time = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO audit_log
            (timestamp, user_id, org_id, action, resource_type, resource_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, '{}')
            """,
            (recent_time, "user_1", "org_1", "recent.action", "test", "recent_1"),
        )
        conn.commit()

        return store

    def test_get_log_since(self, store_with_old_logs):
        """Test filtering logs since a timestamp."""
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        logs = store_with_old_logs.get_log(since=since)

        # Should only get recent log
        assert len(logs) == 1
        assert logs[0]["action"] == "recent.action"

    def test_get_log_until(self, store_with_old_logs):
        """Test filtering logs until a timestamp."""
        until = datetime.now(timezone.utc) - timedelta(hours=24)
        logs = store_with_old_logs.get_log(until=until)

        # Should only get old log
        assert len(logs) == 1
        assert logs[0]["action"] == "old.action"

    def test_get_log_time_range(self, store_with_old_logs):
        """Test filtering logs within a time range."""
        since = datetime.now(timezone.utc) - timedelta(hours=72)
        until = datetime.now(timezone.utc) - timedelta(hours=24)
        logs = store_with_old_logs.get_log(since=since, until=until)

        # Should only get old log (between 72h and 24h ago)
        assert len(logs) == 1
        assert logs[0]["action"] == "old.action"


class TestLogCount:
    """Tests for log counting."""

    @pytest.fixture(autouse=True)
    def setup_logs(self, store):
        """Set up test logs."""
        for i in range(10):
            store.log_event(
                action="user.login",
                resource_type="user",
                resource_id=f"user_{i}",
                user_id=f"user_{i % 3}",  # 3 unique users
                org_id=f"org_{i % 2}",  # 2 unique orgs
            )

    def test_get_log_count_all(self, store):
        """Test counting all logs."""
        count = store.get_log_count()

        assert count == 10

    def test_get_log_count_by_org(self, store):
        """Test counting logs by organization."""
        count = store.get_log_count(org_id="org_0")

        assert count == 5

    def test_get_log_count_by_user(self, store):
        """Test counting logs by user."""
        count = store.get_log_count(user_id="user_0")

        # Users are assigned mod 3, so user_0 gets entries 0, 3, 6, 9
        assert count == 4

    def test_get_log_count_by_action(self, store):
        """Test counting logs by action."""
        count = store.get_log_count(action="user.login")

        assert count == 10

    def test_get_log_count_combined(self, store):
        """Test counting with combined filters."""
        count = store.get_log_count(
            org_id="org_0",
            user_id="user_0",
        )

        # Entries where i % 2 == 0 (org_0) AND i % 3 == 0 (user_0): 0, 6
        assert count == 2


class TestCleanup:
    """Tests for log cleanup operations."""

    @pytest.fixture
    def store_with_old_logs(self, store):
        """Create store with old logs."""
        conn = store._get_connection()

        # Insert old log (100 days ago)
        old_time = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        conn.execute(
            """
            INSERT INTO audit_log
            (timestamp, user_id, org_id, action, resource_type, resource_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, '{}')
            """,
            (old_time, "user_1", "org_1", "old.action", "test", "old_1"),
        )

        # Insert recent log
        recent_time = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO audit_log
            (timestamp, user_id, org_id, action, resource_type, resource_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, '{}')
            """,
            (recent_time, "user_1", "org_1", "recent.action", "test", "recent_1"),
        )
        conn.commit()

        return store

    def test_cleanup_old_entries(self, store_with_old_logs):
        """Test cleaning up old entries."""
        count = store_with_old_logs.cleanup_old_entries(days=90)

        assert count == 1

        # Verify old entry is gone
        logs = store_with_old_logs.get_log()
        assert len(logs) == 1
        assert logs[0]["action"] == "recent.action"

    def test_cleanup_preserves_recent(self, store_with_old_logs):
        """Test that cleanup preserves recent entries."""
        store_with_old_logs.cleanup_old_entries(days=90)

        logs = store_with_old_logs.get_log(action="recent.action")
        assert len(logs) == 1


class TestRecentActivity:
    """Tests for recent activity retrieval."""

    @pytest.fixture
    def store_with_activity(self, store):
        """Create store with recent activity."""
        # Add recent activity
        for i in range(5):
            store.log_event(
                action=f"action_{i}",
                resource_type="test",
                user_id="user_1",
                org_id="org_1",
            )
        return store

    def test_get_recent_activity(self, store_with_activity):
        """Test getting recent activity."""
        activity = store_with_activity.get_recent_activity(
            user_id="user_1",
            hours=24,
        )

        assert len(activity) == 5

    def test_get_recent_activity_by_org(self, store_with_activity):
        """Test getting recent activity by organization."""
        activity = store_with_activity.get_recent_activity(
            org_id="org_1",
            hours=24,
        )

        assert len(activity) == 5


class TestSecurityEvents:
    """Tests for security event retrieval."""

    @pytest.fixture
    def store_with_security_events(self, store):
        """Create store with security events."""
        # Add security events
        security_actions = [
            "login.success",
            "login.failed",
            "password.changed",
            "api_key.created",
            "permission.granted",
            "auth.token_refresh",
            "lockout.triggered",
        ]

        for action in security_actions:
            store.log_event(
                action=action,
                resource_type="security",
                user_id="user_1",
                org_id="org_1",
            )

        # Add non-security events
        store.log_event(
            action="subscription.created",
            resource_type="subscription",
            user_id="user_1",
        )

        return store

    def test_get_security_events(self, store_with_security_events):
        """Test getting security events."""
        events = store_with_security_events.get_security_events(user_id="user_1")

        # Should only get security-related events
        assert len(events) == 7
        for event in events:
            assert any(
                event["action"].startswith(prefix)
                for prefix in [
                    "login.",
                    "password.",
                    "api_key.",
                    "permission.",
                    "auth.",
                    "lockout.",
                ]
            )

    def test_get_security_events_limit(self, store_with_security_events):
        """Test security events with limit."""
        events = store_with_security_events.get_security_events(
            user_id="user_1",
            limit=3,
        )

        assert len(events) <= 3


class TestConnectionManagement:
    """Tests for database connection management."""

    def test_close(self, store):
        """Test closing the store."""
        # Should not raise
        store.close()

    def test_external_connection(self, db_with_schema):
        """Test using external connection factory."""

        def get_conn():
            conn = sqlite3.connect(str(db_with_schema), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn

        store = AuditStore(db_with_schema, get_connection=get_conn)
        entry_id = store.log_event(
            action="external.test",
            resource_type="test",
        )

        assert entry_id is not None
        logs = store.get_log()
        assert len(logs) == 1
