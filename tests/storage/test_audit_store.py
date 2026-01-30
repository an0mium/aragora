"""
Tests for AuditStore - SQLite backend for audit logging and compliance.

Tests cover:
- AuditStore initialization with SQLite backend
- log_event() - Audit event logging
- get_log() - Query with various filters
- get_log_count() - Entry counting
- cleanup_old_entries() - Retention policy
- get_recent_activity() - Recent activity queries
- get_security_events() - Security event filtering
- Factory functions (get_audit_store, reset_audit_store)
"""

from __future__ import annotations

import json
import pytest
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import aragora.storage.audit_store as module
from aragora.storage.audit_store import (
    AuditStore,
    get_audit_store,
    reset_audit_store,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def audit_store(tmp_path: Path) -> AuditStore:
    """Create a fresh AuditStore for each test."""
    db_path = tmp_path / "audit_test.db"
    return AuditStore(db_path=str(db_path), backend="sqlite")


@pytest.fixture
def populated_store(audit_store: AuditStore) -> AuditStore:
    """AuditStore with sample audit events."""
    # Login events
    audit_store.log_event(
        action="login.success",
        resource_type="session",
        user_id="user-1",
        org_id="org-1",
        ip_address="192.168.1.1",
    )
    audit_store.log_event(
        action="login.failed",
        resource_type="session",
        user_id="user-2",
        org_id="org-1",
        metadata={"reason": "invalid_password"},
    )

    # Data events
    audit_store.log_event(
        action="debate.created",
        resource_type="debate",
        resource_id="debate-123",
        user_id="user-1",
        org_id="org-1",
        new_value={"title": "Test Debate"},
    )
    audit_store.log_event(
        action="debate.updated",
        resource_type="debate",
        resource_id="debate-123",
        user_id="user-1",
        org_id="org-1",
        old_value={"title": "Test Debate"},
        new_value={"title": "Updated Debate"},
    )

    # Different org
    audit_store.log_event(
        action="subscription.created",
        resource_type="subscription",
        user_id="user-3",
        org_id="org-2",
    )

    return audit_store


# =============================================================================
# AuditStore Initialization Tests
# =============================================================================


class TestAuditStoreInit:
    """Tests for AuditStore initialization."""

    def test_init_with_sqlite_backend(self, tmp_path: Path):
        """Should initialize with SQLite backend."""
        db_path = tmp_path / "test_audit.db"
        store = AuditStore(db_path=str(db_path), backend="sqlite")

        assert store.backend_type == "sqlite"
        assert store._backend is not None

    def test_init_creates_db_file(self, tmp_path: Path):
        """Should create database file."""
        db_path = tmp_path / "test_audit.db"
        AuditStore(db_path=str(db_path), backend="sqlite")

        assert db_path.exists()

    def test_init_with_external_connection(self, tmp_path: Path):
        """Should accept external connection factory."""
        import sqlite3

        db_path = tmp_path / "shared.db"
        conn = sqlite3.connect(str(db_path))

        def get_conn():
            return conn

        store = AuditStore(db_path=str(db_path), get_connection=get_conn)
        assert store._external_get_connection is not None

    def test_init_postgresql_requires_url(self, tmp_path: Path):
        """PostgreSQL backend requires database_url."""
        with pytest.raises(ValueError, match="PostgreSQL backend requires database_url"):
            AuditStore(db_path=str(tmp_path / "test.db"), backend="postgresql")


# =============================================================================
# log_event Tests
# =============================================================================


class TestLogEvent:
    """Tests for log_event method."""

    def test_log_basic_event(self, audit_store: AuditStore):
        """Should log a basic audit event."""
        rowid = audit_store.log_event(
            action="login.success",
            resource_type="session",
        )

        # Event was logged
        logs = audit_store.get_log()
        assert len(logs) == 1
        assert logs[0]["action"] == "login.success"
        assert logs[0]["resource_type"] == "session"

    def test_log_event_with_user_and_org(self, audit_store: AuditStore):
        """Should log event with user and org context."""
        audit_store.log_event(
            action="debate.created",
            resource_type="debate",
            user_id="user-123",
            org_id="org-456",
        )

        logs = audit_store.get_log()
        assert logs[0]["user_id"] == "user-123"
        assert logs[0]["org_id"] == "org-456"

    def test_log_event_with_resource_id(self, audit_store: AuditStore):
        """Should log event with resource ID."""
        audit_store.log_event(
            action="debate.updated",
            resource_type="debate",
            resource_id="debate-789",
        )

        logs = audit_store.get_log()
        assert logs[0]["resource_id"] == "debate-789"

    def test_log_event_with_old_and_new_values(self, audit_store: AuditStore):
        """Should log old and new values for changes."""
        audit_store.log_event(
            action="settings.changed",
            resource_type="user",
            old_value={"theme": "light"},
            new_value={"theme": "dark"},
        )

        logs = audit_store.get_log()
        assert logs[0]["old_value"] == {"theme": "light"}
        assert logs[0]["new_value"] == {"theme": "dark"}

    def test_log_event_with_metadata(self, audit_store: AuditStore):
        """Should log additional metadata."""
        audit_store.log_event(
            action="login.failed",
            resource_type="session",
            metadata={"reason": "invalid_password", "attempts": 3},
        )

        logs = audit_store.get_log()
        assert logs[0]["metadata"]["reason"] == "invalid_password"
        assert logs[0]["metadata"]["attempts"] == 3

    def test_log_event_with_ip_and_user_agent(self, audit_store: AuditStore):
        """Should log IP address and user agent."""
        audit_store.log_event(
            action="api.request",
            resource_type="api",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0",
        )

        logs = audit_store.get_log()
        assert logs[0]["ip_address"] == "192.168.1.100"
        assert logs[0]["user_agent"] == "Mozilla/5.0"

    def test_log_event_sets_timestamp(self, audit_store: AuditStore):
        """Should set timestamp automatically."""
        before = datetime.now(timezone.utc)
        audit_store.log_event(action="test.event", resource_type="test")
        after = datetime.now(timezone.utc)

        logs = audit_store.get_log()
        timestamp = datetime.fromisoformat(logs[0]["timestamp"])
        assert before <= timestamp <= after


# =============================================================================
# get_log Tests
# =============================================================================


class TestGetLog:
    """Tests for get_log method."""

    def test_get_all_logs(self, populated_store: AuditStore):
        """Should return all logs when no filters."""
        logs = populated_store.get_log()
        assert len(logs) == 5

    def test_filter_by_org_id(self, populated_store: AuditStore):
        """Should filter by organization."""
        logs = populated_store.get_log(org_id="org-1")
        assert len(logs) == 4
        assert all(log["org_id"] == "org-1" for log in logs)

    def test_filter_by_user_id(self, populated_store: AuditStore):
        """Should filter by user."""
        logs = populated_store.get_log(user_id="user-1")
        assert len(logs) == 3
        assert all(log["user_id"] == "user-1" for log in logs)

    def test_filter_by_action_exact(self, populated_store: AuditStore):
        """Should filter by exact action."""
        logs = populated_store.get_log(action="login.success")
        assert len(logs) == 1
        assert logs[0]["action"] == "login.success"

    def test_filter_by_action_prefix(self, populated_store: AuditStore):
        """Should filter by action prefix with wildcard."""
        logs = populated_store.get_log(action="login.*")
        assert len(logs) == 2
        assert all(log["action"].startswith("login.") for log in logs)

    def test_filter_by_resource_type(self, populated_store: AuditStore):
        """Should filter by resource type."""
        logs = populated_store.get_log(resource_type="debate")
        assert len(logs) == 2
        assert all(log["resource_type"] == "debate" for log in logs)

    def test_filter_by_since(self, audit_store: AuditStore):
        """Should filter entries after a timestamp."""
        # Create events with delay
        audit_store.log_event(action="event.old", resource_type="test")
        time.sleep(0.1)
        cutoff = datetime.now(timezone.utc)
        time.sleep(0.1)
        audit_store.log_event(action="event.new", resource_type="test")

        logs = audit_store.get_log(since=cutoff)
        assert len(logs) == 1
        assert logs[0]["action"] == "event.new"

    def test_filter_by_until(self, audit_store: AuditStore):
        """Should filter entries before a timestamp."""
        audit_store.log_event(action="event.old", resource_type="test")
        time.sleep(0.1)
        cutoff = datetime.now(timezone.utc)
        time.sleep(0.1)
        audit_store.log_event(action="event.new", resource_type="test")

        logs = audit_store.get_log(until=cutoff)
        assert len(logs) == 1
        assert logs[0]["action"] == "event.old"

    def test_pagination_limit(self, populated_store: AuditStore):
        """Should respect limit parameter."""
        logs = populated_store.get_log(limit=2)
        assert len(logs) == 2

    def test_pagination_offset(self, populated_store: AuditStore):
        """Should respect offset parameter."""
        all_logs = populated_store.get_log()
        offset_logs = populated_store.get_log(offset=2, limit=10)
        assert len(offset_logs) == len(all_logs) - 2

    def test_combined_filters(self, populated_store: AuditStore):
        """Should combine multiple filters."""
        logs = populated_store.get_log(
            org_id="org-1",
            user_id="user-1",
            resource_type="debate",
        )
        assert len(logs) == 2
        assert all(
            log["org_id"] == "org-1"
            and log["user_id"] == "user-1"
            and log["resource_type"] == "debate"
            for log in logs
        )

    def test_returns_ordered_by_timestamp_desc(self, audit_store: AuditStore):
        """Should return logs ordered by timestamp descending."""
        audit_store.log_event(action="first", resource_type="test")
        time.sleep(0.05)
        audit_store.log_event(action="second", resource_type="test")
        time.sleep(0.05)
        audit_store.log_event(action="third", resource_type="test")

        logs = audit_store.get_log()
        assert logs[0]["action"] == "third"
        assert logs[1]["action"] == "second"
        assert logs[2]["action"] == "first"


# =============================================================================
# get_log_count Tests
# =============================================================================


class TestGetLogCount:
    """Tests for get_log_count method."""

    def test_count_all(self, populated_store: AuditStore):
        """Should count all entries."""
        count = populated_store.get_log_count()
        assert count == 5

    def test_count_by_org(self, populated_store: AuditStore):
        """Should count entries by org."""
        count = populated_store.get_log_count(org_id="org-1")
        assert count == 4

    def test_count_by_user(self, populated_store: AuditStore):
        """Should count entries by user."""
        count = populated_store.get_log_count(user_id="user-1")
        assert count == 3

    def test_count_by_action_prefix(self, populated_store: AuditStore):
        """Should count entries by action prefix."""
        count = populated_store.get_log_count(action="login.*")
        assert count == 2

    def test_count_by_resource_type(self, populated_store: AuditStore):
        """Should count entries by resource type."""
        count = populated_store.get_log_count(resource_type="debate")
        assert count == 2

    def test_count_with_combined_filters(self, populated_store: AuditStore):
        """Should count with combined filters."""
        count = populated_store.get_log_count(
            org_id="org-1",
            resource_type="debate",
        )
        assert count == 2


# =============================================================================
# cleanup_old_entries Tests
# =============================================================================


class TestCleanupOldEntries:
    """Tests for cleanup_old_entries method."""

    def test_cleanup_deletes_old_entries(self, audit_store: AuditStore):
        """Should delete entries older than specified days."""
        # Create an old entry by manipulating the database directly
        import sqlite3

        conn = sqlite3.connect(str(audit_store.db_path))
        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        conn.execute(
            """
            INSERT INTO audit_log (timestamp, action, resource_type)
            VALUES (?, ?, ?)
            """,
            (old_timestamp, "old.event", "test"),
        )
        conn.commit()
        conn.close()

        # Add a recent entry
        audit_store.log_event(action="recent.event", resource_type="test")

        # Cleanup with 90-day retention
        deleted = audit_store.cleanup_old_entries(days=90)

        assert deleted == 1
        logs = audit_store.get_log()
        assert len(logs) == 1
        assert logs[0]["action"] == "recent.event"

    def test_cleanup_returns_zero_when_nothing_to_delete(self, audit_store: AuditStore):
        """Should return 0 when no old entries."""
        audit_store.log_event(action="recent.event", resource_type="test")

        deleted = audit_store.cleanup_old_entries(days=90)
        assert deleted == 0

    def test_cleanup_respects_days_parameter(self, audit_store: AuditStore):
        """Should respect custom retention period."""
        import sqlite3

        conn = sqlite3.connect(str(audit_store.db_path))

        # Entry from 10 days ago
        ten_days_ago = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        conn.execute(
            "INSERT INTO audit_log (timestamp, action, resource_type) VALUES (?, ?, ?)",
            (ten_days_ago, "ten.days.old", "test"),
        )
        conn.commit()
        conn.close()

        # Cleanup with 30-day retention - should not delete
        deleted = audit_store.cleanup_old_entries(days=30)
        assert deleted == 0

        # Cleanup with 5-day retention - should delete
        deleted = audit_store.cleanup_old_entries(days=5)
        assert deleted == 1


# =============================================================================
# get_recent_activity Tests
# =============================================================================


class TestGetRecentActivity:
    """Tests for get_recent_activity method."""

    def test_returns_recent_events(self, populated_store: AuditStore):
        """Should return recent events."""
        activity = populated_store.get_recent_activity(hours=24)
        assert len(activity) == 5

    def test_filter_by_user(self, populated_store: AuditStore):
        """Should filter by user."""
        activity = populated_store.get_recent_activity(user_id="user-1")
        assert len(activity) == 3

    def test_filter_by_org(self, populated_store: AuditStore):
        """Should filter by organization."""
        activity = populated_store.get_recent_activity(org_id="org-2")
        assert len(activity) == 1

    def test_respects_limit(self, populated_store: AuditStore):
        """Should respect limit parameter."""
        activity = populated_store.get_recent_activity(limit=2)
        assert len(activity) == 2


# =============================================================================
# get_security_events Tests
# =============================================================================


class TestGetSecurityEvents:
    """Tests for get_security_events method."""

    def test_returns_login_events(self, populated_store: AuditStore):
        """Should return login events."""
        events = populated_store.get_security_events()
        login_events = [e for e in events if e["action"].startswith("login.")]
        assert len(login_events) == 2

    def test_filter_by_user(self, audit_store: AuditStore):
        """Should filter security events by user."""
        audit_store.log_event(
            action="login.success", resource_type="session", user_id="user-1"
        )
        audit_store.log_event(
            action="password.changed", resource_type="user", user_id="user-1"
        )
        audit_store.log_event(
            action="login.success", resource_type="session", user_id="user-2"
        )

        events = audit_store.get_security_events(user_id="user-1")
        assert len(events) == 2
        assert all(e["user_id"] == "user-1" for e in events)

    def test_filter_by_org(self, audit_store: AuditStore):
        """Should filter security events by org."""
        audit_store.log_event(
            action="api_key.created", resource_type="api_key", org_id="org-1"
        )
        audit_store.log_event(
            action="api_key.created", resource_type="api_key", org_id="org-2"
        )

        events = audit_store.get_security_events(org_id="org-1")
        assert len(events) == 1
        assert events[0]["org_id"] == "org-1"

    def test_sorted_by_timestamp(self, audit_store: AuditStore):
        """Should return events sorted by timestamp descending."""
        audit_store.log_event(action="login.first", resource_type="session")
        time.sleep(0.05)
        audit_store.log_event(action="login.second", resource_type="session")

        events = audit_store.get_security_events()
        assert events[0]["action"] == "login.second"
        assert events[1]["action"] == "login.first"

    def test_respects_limit(self, audit_store: AuditStore):
        """Should respect limit parameter."""
        for i in range(10):
            audit_store.log_event(action=f"login.event{i}", resource_type="session")

        events = audit_store.get_security_events(limit=5)
        assert len(events) == 5


# =============================================================================
# close Tests
# =============================================================================


class TestClose:
    """Tests for close method."""

    def test_close_backend(self, tmp_path: Path):
        """Should close backend connection."""
        store = AuditStore(db_path=str(tmp_path / "test.db"), backend="sqlite")
        store.log_event(action="test", resource_type="test")

        store.close()
        assert store._backend is None

    def test_close_is_idempotent(self, tmp_path: Path):
        """Should handle multiple close calls."""
        store = AuditStore(db_path=str(tmp_path / "test.db"), backend="sqlite")
        store.close()
        store.close()  # Should not raise


# =============================================================================
# Factory Functions Tests
# =============================================================================


class TestGetAuditStore:
    """Tests for get_audit_store factory function."""

    def test_returns_audit_store(self, tmp_path: Path):
        """Should return an AuditStore instance."""
        # Reset any existing singleton
        reset_audit_store()

        with patch.dict(
            "os.environ",
            {"ARAGORA_DATA_DIR": str(tmp_path), "ARAGORA_ENVIRONMENT": "development"},
        ):
            store = get_audit_store()
            assert isinstance(store, AuditStore)

        reset_audit_store()

    def test_returns_singleton(self, tmp_path: Path):
        """Should return same instance on multiple calls."""
        reset_audit_store()

        with patch.dict(
            "os.environ",
            {"ARAGORA_DATA_DIR": str(tmp_path), "ARAGORA_ENVIRONMENT": "development"},
        ):
            store1 = get_audit_store()
            store2 = get_audit_store()
            assert store1 is store2

        reset_audit_store()

    def test_uses_data_dir(self, tmp_path: Path):
        """Should use ARAGORA_DATA_DIR for db_path."""
        reset_audit_store()

        with patch.dict(
            "os.environ",
            {"ARAGORA_DATA_DIR": str(tmp_path), "ARAGORA_ENVIRONMENT": "development"},
        ):
            store = get_audit_store()
            assert "audit.db" in str(store.db_path)

        reset_audit_store()


class TestResetAuditStore:
    """Tests for reset_audit_store function."""

    def test_resets_singleton(self, tmp_path: Path):
        """Should reset the singleton instance."""
        reset_audit_store()

        with patch.dict(
            "os.environ",
            {"ARAGORA_DATA_DIR": str(tmp_path), "ARAGORA_ENVIRONMENT": "development"},
        ):
            store1 = get_audit_store()
            reset_audit_store()
            store2 = get_audit_store()

            # Should be different instances
            assert store1 is not store2

        reset_audit_store()


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_importable(self):
        """All items in __all__ should be importable."""
        for name in module.__all__:
            assert hasattr(module, name), f"Missing export: {name}"

    def test_key_exports(self):
        """Key exports should be available."""
        from aragora.storage.audit_store import (
            AuditStore,
            get_audit_store,
            reset_audit_store,
        )

        assert AuditStore is not None
        assert callable(get_audit_store)
        assert callable(reset_audit_store)
