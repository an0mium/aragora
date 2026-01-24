"""Tests for InboxActivityStore."""

import pytest
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

from aragora.storage.inbox_activity_store import (
    InboxActivity,
    InboxActivityAction,
    InboxActivityStore,
    reset_inbox_activity_store,
)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_inbox_activities.db"


@pytest.fixture
def store(temp_db_path):
    """Create a fresh InboxActivityStore for testing."""
    reset_inbox_activity_store()
    store = InboxActivityStore(db_path=temp_db_path, backend="sqlite")
    yield store
    store.close()


class TestInboxActivity:
    """Tests for InboxActivity dataclass."""

    def test_create_activity(self):
        """Test creating an activity."""
        activity = InboxActivity(
            inbox_id="inbox_123",
            org_id="org_456",
            actor_id="user_789",
            action=InboxActivityAction.ASSIGNED,
            target_id="message_abc",
            metadata={"assignee_id": "user_def"},
        )

        assert activity.inbox_id == "inbox_123"
        assert activity.org_id == "org_456"
        assert activity.actor_id == "user_789"
        assert activity.action == "assigned"
        assert activity.target_id == "message_abc"
        assert activity.metadata["assignee_id"] == "user_def"
        assert activity.id is not None
        assert activity.created_at is not None

    def test_to_dict(self):
        """Test converting activity to dict."""
        activity = InboxActivity(
            id="act_123",
            inbox_id="inbox_123",
            org_id="org_456",
            actor_id="user_789",
            action=InboxActivityAction.STATUS_CHANGED,
            metadata={"from": "open", "to": "resolved"},
        )

        d = activity.to_dict()
        assert d["id"] == "act_123"
        assert d["inbox_id"] == "inbox_123"
        assert d["action"] == "status_changed"
        assert d["metadata"]["from"] == "open"
        assert "created_at" in d

    def test_from_dict(self):
        """Test creating activity from dict."""
        data = {
            "id": "act_123",
            "inbox_id": "inbox_123",
            "org_id": "org_456",
            "actor_id": "user_789",
            "action": "note_added",
            "target_id": "message_abc",
            "metadata": {"note_id": "note_123"},
            "created_at": "2024-01-15T10:30:00+00:00",
        }

        activity = InboxActivity.from_dict(data)
        assert activity.id == "act_123"
        assert activity.action == "note_added"
        assert activity.metadata["note_id"] == "note_123"
        assert activity.created_at.year == 2024


class TestInboxActivityStore:
    """Tests for InboxActivityStore."""

    def test_log_and_retrieve_activity(self, store):
        """Test logging and retrieving an activity."""
        activity = InboxActivity(
            inbox_id="inbox_123",
            org_id="org_456",
            actor_id="user_789",
            action=InboxActivityAction.ASSIGNED,
            target_id="message_abc",
            metadata={"assignee_id": "user_def"},
        )

        store.log_activity(activity)

        retrieved = store.get_activity(activity.id)
        assert retrieved is not None
        assert retrieved.id == activity.id
        assert retrieved.inbox_id == "inbox_123"
        assert retrieved.action == "assigned"
        assert retrieved.metadata["assignee_id"] == "user_def"

    def test_get_activities_by_inbox(self, store):
        """Test getting activities for a specific inbox."""
        # Log activities for different inboxes
        for i in range(5):
            store.log_activity(
                InboxActivity(
                    inbox_id="inbox_A",
                    org_id="org_1",
                    actor_id="user_1",
                    action=InboxActivityAction.MESSAGE_RECEIVED,
                    target_id=f"msg_{i}",
                )
            )

        for i in range(3):
            store.log_activity(
                InboxActivity(
                    inbox_id="inbox_B",
                    org_id="org_1",
                    actor_id="user_2",
                    action=InboxActivityAction.MESSAGE_RECEIVED,
                    target_id=f"msg_b_{i}",
                )
            )

        inbox_a_activities = store.get_activities("inbox_A")
        assert len(inbox_a_activities) == 5

        inbox_b_activities = store.get_activities("inbox_B")
        assert len(inbox_b_activities) == 3

    def test_get_activities_with_limit_offset(self, store):
        """Test pagination of activities."""
        for i in range(10):
            store.log_activity(
                InboxActivity(
                    inbox_id="inbox_123",
                    org_id="org_456",
                    actor_id="user_789",
                    action=InboxActivityAction.MESSAGE_RECEIVED,
                    target_id=f"msg_{i}",
                )
            )

        # Get first page
        page1 = store.get_activities("inbox_123", limit=3, offset=0)
        assert len(page1) == 3

        # Get second page
        page2 = store.get_activities("inbox_123", limit=3, offset=3)
        assert len(page2) == 3

        # Pages should be different
        assert page1[0].id != page2[0].id

    def test_get_activities_filter_by_action(self, store):
        """Test filtering activities by action type."""
        store.log_activity(
            InboxActivity(
                inbox_id="inbox_123",
                org_id="org_456",
                actor_id="user_1",
                action=InboxActivityAction.ASSIGNED,
            )
        )
        store.log_activity(
            InboxActivity(
                inbox_id="inbox_123",
                org_id="org_456",
                actor_id="user_1",
                action=InboxActivityAction.STATUS_CHANGED,
            )
        )
        store.log_activity(
            InboxActivity(
                inbox_id="inbox_123",
                org_id="org_456",
                actor_id="user_1",
                action=InboxActivityAction.ASSIGNED,
            )
        )

        assigned = store.get_activities("inbox_123", action=InboxActivityAction.ASSIGNED)
        assert len(assigned) == 2

        status_changed = store.get_activities(
            "inbox_123", action=InboxActivityAction.STATUS_CHANGED
        )
        assert len(status_changed) == 1

    def test_get_message_history(self, store):
        """Test getting activity history for a specific message."""
        message_id = "msg_123"

        # Log various activities for the same message
        store.log_activity(
            InboxActivity(
                inbox_id="inbox_1",
                org_id="org_1",
                actor_id="user_1",
                action=InboxActivityAction.MESSAGE_RECEIVED,
                target_id=message_id,
            )
        )
        store.log_activity(
            InboxActivity(
                inbox_id="inbox_1",
                org_id="org_1",
                actor_id="user_2",
                action=InboxActivityAction.ASSIGNED,
                target_id=message_id,
                metadata={"assignee_id": "user_3"},
            )
        )
        store.log_activity(
            InboxActivity(
                inbox_id="inbox_1",
                org_id="org_1",
                actor_id="user_3",
                action=InboxActivityAction.STATUS_CHANGED,
                target_id=message_id,
                metadata={"from": "open", "to": "in_progress"},
            )
        )

        # Log activity for different message
        store.log_activity(
            InboxActivity(
                inbox_id="inbox_1",
                org_id="org_1",
                actor_id="user_1",
                action=InboxActivityAction.MESSAGE_RECEIVED,
                target_id="other_msg",
            )
        )

        history = store.get_message_history(message_id)
        assert len(history) == 3

        other_history = store.get_message_history("other_msg")
        assert len(other_history) == 1

    def test_get_org_activities(self, store):
        """Test getting all activities for an organization."""
        # Log activities for different orgs
        for i in range(4):
            store.log_activity(
                InboxActivity(
                    inbox_id=f"inbox_{i}",
                    org_id="org_A",
                    actor_id="user_1",
                    action=InboxActivityAction.MESSAGE_RECEIVED,
                )
            )

        for i in range(2):
            store.log_activity(
                InboxActivity(
                    inbox_id=f"inbox_{i}",
                    org_id="org_B",
                    actor_id="user_2",
                    action=InboxActivityAction.MESSAGE_RECEIVED,
                )
            )

        org_a_activities = store.get_org_activities("org_A")
        assert len(org_a_activities) == 4

        org_b_activities = store.get_org_activities("org_B")
        assert len(org_b_activities) == 2

    def test_get_actor_activities(self, store):
        """Test getting all activities by a specific actor."""
        # Log activities by different actors
        for i in range(3):
            store.log_activity(
                InboxActivity(
                    inbox_id="inbox_1",
                    org_id="org_1",
                    actor_id="user_A",
                    action=InboxActivityAction.MESSAGE_SENT,
                )
            )

        for i in range(5):
            store.log_activity(
                InboxActivity(
                    inbox_id="inbox_1",
                    org_id="org_1",
                    actor_id="user_B",
                    action=InboxActivityAction.MESSAGE_SENT,
                )
            )

        user_a_activities = store.get_actor_activities("user_A")
        assert len(user_a_activities) == 3

        user_b_activities = store.get_actor_activities("user_B")
        assert len(user_b_activities) == 5

    def test_count_activities(self, store):
        """Test counting activities with filters."""
        # Log various activities
        for i in range(5):
            store.log_activity(
                InboxActivity(
                    inbox_id="inbox_1",
                    org_id="org_1",
                    actor_id="user_1",
                    action=InboxActivityAction.ASSIGNED,
                )
            )

        for i in range(3):
            store.log_activity(
                InboxActivity(
                    inbox_id="inbox_2",
                    org_id="org_1",
                    actor_id="user_1",
                    action=InboxActivityAction.STATUS_CHANGED,
                )
            )

        # Count all
        total = store.count_activities()
        assert total == 8

        # Count by inbox
        inbox1_count = store.count_activities(inbox_id="inbox_1")
        assert inbox1_count == 5

        # Count by action
        assigned_count = store.count_activities(action=InboxActivityAction.ASSIGNED)
        assert assigned_count == 5

    def test_activities_ordered_by_created_at_desc(self, store):
        """Test that activities are returned in descending order by created_at."""
        base_time = datetime.now(timezone.utc)

        # Log activities with different timestamps
        for i in range(5):
            activity = InboxActivity(
                inbox_id="inbox_123",
                org_id="org_456",
                actor_id="user_789",
                action=InboxActivityAction.MESSAGE_RECEIVED,
                target_id=f"msg_{i}",
                created_at=base_time - timedelta(hours=i),
            )
            store.log_activity(activity)

        activities = store.get_activities("inbox_123")

        # Most recent should be first
        for i in range(len(activities) - 1):
            assert activities[i].created_at >= activities[i + 1].created_at


class TestInboxActivityStoreCleanup:
    """Tests for activity cleanup functionality."""

    def test_cleanup_expired(self, store):
        """Test cleaning up expired activities."""
        # Temporarily set short retention
        store.retention_days = 1

        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=2)

        # Log old activity
        old_activity = InboxActivity(
            inbox_id="inbox_123",
            org_id="org_456",
            actor_id="user_789",
            action=InboxActivityAction.MESSAGE_RECEIVED,
            created_at=old_time,
        )
        store.log_activity(old_activity)

        # Log recent activity
        recent_activity = InboxActivity(
            inbox_id="inbox_123",
            org_id="org_456",
            actor_id="user_789",
            action=InboxActivityAction.MESSAGE_RECEIVED,
            created_at=now,
        )
        store.log_activity(recent_activity)

        # Verify both exist
        assert store.count_activities() == 2

        # Run cleanup
        removed = store.cleanup_expired()
        assert removed == 1

        # Only recent activity should remain
        assert store.count_activities() == 1
        remaining = store.get_activities("inbox_123")
        assert len(remaining) == 1
        assert remaining[0].id == recent_activity.id


class TestInboxActivityActions:
    """Tests for activity action constants."""

    def test_all_actions_defined(self):
        """Test that all expected action types are defined."""
        expected_actions = [
            "assigned",
            "reassigned",
            "status_changed",
            "note_added",
            "sla_breached",
            "sla_warning",
            "member_added",
            "member_removed",
            "member_role_changed",
            "message_received",
            "message_sent",
            "tag_added",
            "tag_removed",
            "priority_changed",
            "merged",
            "split",
        ]

        for action in expected_actions:
            assert hasattr(InboxActivityAction, action.upper())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
