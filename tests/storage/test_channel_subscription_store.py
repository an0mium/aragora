"""Tests for Channel Subscription Store."""

import os
import tempfile
import pytest

from aragora.storage.channel_subscription_store import (
    ChannelSubscription,
    ChannelSubscriptionStore,
    ChannelType,
    EventType,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def store(temp_db):
    """Create a store with temporary database."""
    return ChannelSubscriptionStore(db_path=temp_db)


@pytest.fixture
def sample_subscription():
    """Create a sample subscription for testing."""
    return ChannelSubscription(
        id="sub-123",
        org_id="org-456",
        channel_type=ChannelType.SLACK,
        channel_id="C12345678",
        workspace_id="T12345678",
        channel_name="#general",
        event_types=[EventType.RECEIPT, EventType.BUDGET_ALERT],
        created_at=1700000000.0,
        created_by="user-789",
        is_active=True,
        config={"mention_users": ["U123", "U456"]},
    )


class TestChannelSubscription:
    """Tests for ChannelSubscription dataclass."""

    def test_to_dict(self, sample_subscription):
        """Test to_dict includes all fields."""
        result = sample_subscription.to_dict()

        assert result["id"] == "sub-123"
        assert result["org_id"] == "org-456"
        assert result["channel_type"] == "slack"
        assert result["channel_id"] == "C12345678"
        assert result["workspace_id"] == "T12345678"
        assert result["channel_name"] == "#general"
        assert "receipt" in result["event_types"]
        assert "budget_alert" in result["event_types"]
        assert result["created_at"] == 1700000000.0
        assert "created_at_iso" in result
        assert result["created_by"] == "user-789"
        assert result["is_active"] is True
        assert result["config"]["mention_users"] == ["U123", "U456"]

    def test_to_dict_handles_string_enums(self):
        """Test to_dict handles string event types."""
        sub = ChannelSubscription(
            id="sub-1",
            org_id="org-1",
            channel_type="teams",  # String instead of enum
            channel_id="channel-1",
            event_types=["custom_event"],  # String instead of enum
            created_at=1700000000.0,
        )
        result = sub.to_dict()
        assert result["channel_type"] == "teams"
        assert result["event_types"] == ["custom_event"]


class TestChannelSubscriptionStore:
    """Tests for ChannelSubscriptionStore."""

    def test_create_subscription(self, store, sample_subscription):
        """Test creating a subscription."""
        result = store.create(sample_subscription)

        assert result.id == "sub-123"
        assert result.org_id == "org-456"
        assert result.channel_type == ChannelType.SLACK

    def test_create_generates_id_if_missing(self, store):
        """Test that create generates ID if not provided."""
        sub = ChannelSubscription(
            id="",
            org_id="org-1",
            channel_type=ChannelType.TEAMS,
            channel_id="channel-1",
            event_types=[EventType.RECEIPT],
            created_at=0,
        )
        result = store.create(sub)
        assert result.id  # Should have generated ID
        assert len(result.id) > 0

    def test_create_duplicate_raises_error(self, store, sample_subscription):
        """Test that creating duplicate subscription raises ValueError."""
        store.create(sample_subscription)

        duplicate = ChannelSubscription(
            id="sub-different",
            org_id=sample_subscription.org_id,
            channel_type=sample_subscription.channel_type,
            channel_id=sample_subscription.channel_id,
            event_types=[EventType.RECEIPT],
            created_at=1700000000.0,
        )

        with pytest.raises(ValueError, match="already exists"):
            store.create(duplicate)

    def test_get_subscription(self, store, sample_subscription):
        """Test getting a subscription by ID."""
        store.create(sample_subscription)

        result = store.get("sub-123")

        assert result is not None
        assert result.id == "sub-123"
        assert result.org_id == "org-456"
        assert result.channel_name == "#general"

    def test_get_nonexistent_returns_none(self, store):
        """Test that getting nonexistent subscription returns None."""
        result = store.get("nonexistent")
        assert result is None

    def test_get_by_org(self, store):
        """Test getting subscriptions by organization."""
        # Create multiple subscriptions
        for i in range(3):
            sub = ChannelSubscription(
                id=f"sub-{i}",
                org_id="org-1",
                channel_type=ChannelType.SLACK,
                channel_id=f"C{i}",
                event_types=[EventType.RECEIPT],
                created_at=1700000000.0,
            )
            store.create(sub)

        # Create one for different org
        other_sub = ChannelSubscription(
            id="sub-other",
            org_id="org-2",
            channel_type=ChannelType.SLACK,
            channel_id="C-other",
            event_types=[EventType.RECEIPT],
            created_at=1700000000.0,
        )
        store.create(other_sub)

        result = store.get_by_org("org-1")
        assert len(result) == 3

    def test_get_by_org_filters_by_channel_type(self, store):
        """Test filtering by channel type."""
        # Slack subscription
        slack_sub = ChannelSubscription(
            id="sub-slack",
            org_id="org-1",
            channel_type=ChannelType.SLACK,
            channel_id="C123",
            event_types=[EventType.RECEIPT],
            created_at=1700000000.0,
        )
        store.create(slack_sub)

        # Teams subscription
        teams_sub = ChannelSubscription(
            id="sub-teams",
            org_id="org-1",
            channel_type=ChannelType.TEAMS,
            channel_id="teams-channel",
            event_types=[EventType.RECEIPT],
            created_at=1700000000.0,
        )
        store.create(teams_sub)

        result = store.get_by_org("org-1", channel_type=ChannelType.SLACK)
        assert len(result) == 1
        assert result[0].channel_type == ChannelType.SLACK

    def test_get_by_org_filters_by_event_type(self, store):
        """Test filtering by event type."""
        # Receipt subscription
        receipt_sub = ChannelSubscription(
            id="sub-receipt",
            org_id="org-1",
            channel_type=ChannelType.SLACK,
            channel_id="C-receipt",
            event_types=[EventType.RECEIPT],
            created_at=1700000000.0,
        )
        store.create(receipt_sub)

        # Budget alert subscription
        budget_sub = ChannelSubscription(
            id="sub-budget",
            org_id="org-1",
            channel_type=ChannelType.SLACK,
            channel_id="C-budget",
            event_types=[EventType.BUDGET_ALERT],
            created_at=1700000000.0,
        )
        store.create(budget_sub)

        result = store.get_by_org("org-1", event_type=EventType.RECEIPT)
        assert len(result) == 1
        assert EventType.RECEIPT in result[0].event_types

    def test_get_by_org_active_only(self, store):
        """Test filtering by active status."""
        # Active subscription
        active_sub = ChannelSubscription(
            id="sub-active",
            org_id="org-1",
            channel_type=ChannelType.SLACK,
            channel_id="C-active",
            event_types=[EventType.RECEIPT],
            created_at=1700000000.0,
            is_active=True,
        )
        store.create(active_sub)

        # Inactive subscription
        inactive_sub = ChannelSubscription(
            id="sub-inactive",
            org_id="org-1",
            channel_type=ChannelType.SLACK,
            channel_id="C-inactive",
            event_types=[EventType.RECEIPT],
            created_at=1700000000.0,
            is_active=False,
        )
        store.create(inactive_sub)

        # Active only (default)
        result = store.get_by_org("org-1", active_only=True)
        assert len(result) == 1
        assert result[0].is_active is True

        # Include inactive
        result = store.get_by_org("org-1", active_only=False)
        assert len(result) == 2

    def test_get_for_event(self, store):
        """Test getting subscriptions for a specific event."""
        # Multiple event types
        multi_sub = ChannelSubscription(
            id="sub-multi",
            org_id="org-1",
            channel_type=ChannelType.SLACK,
            channel_id="C-multi",
            event_types=[EventType.RECEIPT, EventType.BUDGET_ALERT],
            created_at=1700000000.0,
        )
        store.create(multi_sub)

        # Receipt only
        receipt_sub = ChannelSubscription(
            id="sub-receipt",
            org_id="org-1",
            channel_type=ChannelType.SLACK,
            channel_id="C-receipt",
            event_types=[EventType.RECEIPT],
            created_at=1700000000.0,
        )
        store.create(receipt_sub)

        result = store.get_for_event("org-1", EventType.BUDGET_ALERT)
        assert len(result) == 1
        assert result[0].id == "sub-multi"

    def test_update_subscription(self, store, sample_subscription):
        """Test updating a subscription."""
        store.create(sample_subscription)

        result = store.update(
            "sub-123",
            event_types=[EventType.DEBATE_COMPLETE],
            channel_name="#updated",
        )

        assert result is not None
        assert result.channel_name == "#updated"
        assert EventType.DEBATE_COMPLETE in result.event_types

        # Verify persistence
        fetched = store.get("sub-123")
        assert fetched.channel_name == "#updated"

    def test_update_nonexistent_returns_none(self, store):
        """Test updating nonexistent subscription returns None."""
        result = store.update("nonexistent", is_active=False)
        assert result is None

    def test_update_is_active(self, store, sample_subscription):
        """Test updating active status."""
        store.create(sample_subscription)

        result = store.update("sub-123", is_active=False)

        assert result is not None
        assert result.is_active is False

    def test_delete_subscription(self, store, sample_subscription):
        """Test deleting a subscription."""
        store.create(sample_subscription)

        result = store.delete("sub-123")

        assert result is True
        assert store.get("sub-123") is None

    def test_delete_nonexistent_returns_false(self, store):
        """Test deleting nonexistent subscription returns False."""
        result = store.delete("nonexistent")
        assert result is False

    def test_deactivate_subscription(self, store, sample_subscription):
        """Test deactivating (soft delete) a subscription."""
        store.create(sample_subscription)

        result = store.deactivate("sub-123")

        assert result is True
        fetched = store.get("sub-123")
        assert fetched is not None
        assert fetched.is_active is False

    def test_count_by_org(self, store):
        """Test counting subscriptions by organization."""
        for i in range(5):
            sub = ChannelSubscription(
                id=f"sub-{i}",
                org_id="org-1",
                channel_type=ChannelType.SLACK,
                channel_id=f"C{i}",
                event_types=[EventType.RECEIPT],
                created_at=1700000000.0,
                is_active=i < 3,  # First 3 active, last 2 inactive
            )
            store.create(sub)

        assert store.count_by_org("org-1", active_only=True) == 3
        assert store.count_by_org("org-1", active_only=False) == 5

    def test_clear(self, store, sample_subscription):
        """Test clearing all subscriptions."""
        store.create(sample_subscription)
        assert store.get("sub-123") is not None

        store.clear()

        assert store.get("sub-123") is None
        assert store.count_by_org("org-456", active_only=False) == 0


class TestChannelType:
    """Tests for ChannelType enum."""

    def test_all_types_exist(self):
        """Test all expected channel types exist."""
        assert ChannelType.SLACK.value == "slack"
        assert ChannelType.TEAMS.value == "teams"
        assert ChannelType.EMAIL.value == "email"
        assert ChannelType.WEBHOOK.value == "webhook"


class TestEventType:
    """Tests for EventType enum."""

    def test_all_types_exist(self):
        """Test all expected event types exist."""
        assert EventType.RECEIPT.value == "receipt"
        assert EventType.BUDGET_ALERT.value == "budget_alert"
        assert EventType.DEBATE_COMPLETE.value == "debate_complete"
        assert EventType.CONSENSUS_REACHED.value == "consensus_reached"
