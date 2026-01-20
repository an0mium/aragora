"""
Tests for Knowledge Mound Sharing Notifications.

Tests the notification system for knowledge sharing events:
- In-app notification storage
- Notification preferences
- Email/webhook integration
- API endpoints
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.notifications import (
    NotificationType,
    NotificationStatus,
    NotificationChannel,
    SharingNotification,
    NotificationPreferences,
    InAppNotificationStore,
    SharingNotifier,
    get_notification_store,
    notify_item_shared,
    notify_share_revoked,
    get_notifications_for_user,
    get_unread_count,
    mark_notification_read,
    mark_all_notifications_read,
    get_notification_preferences,
    set_notification_preferences,
)


class TestSharingNotification:
    """Test SharingNotification dataclass."""

    def test_notification_creation(self):
        """Test creating a notification."""
        notification = SharingNotification(
            id="notif_123",
            user_id="user_456",
            notification_type=NotificationType.ITEM_SHARED,
            title="Knowledge Shared",
            message="Alice shared 'Important Doc' with you",
            item_id="km_789",
            item_title="Important Doc",
            from_user_id="user_alice",
            from_user_name="Alice",
        )

        assert notification.id == "notif_123"
        assert notification.user_id == "user_456"
        assert notification.notification_type == NotificationType.ITEM_SHARED
        assert notification.status == NotificationStatus.UNREAD
        assert notification.item_id == "km_789"

    def test_notification_to_dict(self):
        """Test serializing notification to dict."""
        notification = SharingNotification(
            id="notif_123",
            user_id="user_456",
            notification_type=NotificationType.ITEM_SHARED,
            title="Test",
            message="Test message",
        )

        data = notification.to_dict()
        assert data["id"] == "notif_123"
        assert data["user_id"] == "user_456"
        assert data["notification_type"] == "item_shared"
        assert data["status"] == "unread"
        assert "created_at" in data

    def test_notification_from_dict(self):
        """Test deserializing notification from dict."""
        data = {
            "id": "notif_abc",
            "user_id": "user_xyz",
            "notification_type": "item_unshared",
            "title": "Share Revoked",
            "message": "Access revoked",
            "status": "read",
            "created_at": "2024-01-15T10:30:00",
        }

        notification = SharingNotification.from_dict(data)
        assert notification.id == "notif_abc"
        assert notification.notification_type == NotificationType.ITEM_UNSHARED
        assert notification.status == NotificationStatus.READ


class TestInAppNotificationStore:
    """Test the in-memory notification store."""

    def test_add_notification(self):
        """Test adding a notification."""
        store = InAppNotificationStore()

        notification = SharingNotification(
            id="notif_1",
            user_id="user_1",
            notification_type=NotificationType.ITEM_SHARED,
            title="Test",
            message="Test",
        )

        store.add_notification(notification)
        notifications = store.get_notifications("user_1")
        assert len(notifications) == 1
        assert notifications[0].id == "notif_1"

    def test_get_notifications_with_status_filter(self):
        """Test filtering notifications by status."""
        store = InAppNotificationStore()

        # Add unread notification
        store.add_notification(
            SharingNotification(
                id="notif_1",
                user_id="user_1",
                notification_type=NotificationType.ITEM_SHARED,
                title="Test 1",
                message="Test 1",
                status=NotificationStatus.UNREAD,
            )
        )

        # Add read notification
        store.add_notification(
            SharingNotification(
                id="notif_2",
                user_id="user_1",
                notification_type=NotificationType.ITEM_SHARED,
                title="Test 2",
                message="Test 2",
                status=NotificationStatus.READ,
            )
        )

        unread = store.get_notifications("user_1", status=NotificationStatus.UNREAD)
        assert len(unread) == 1
        assert unread[0].id == "notif_1"

        read = store.get_notifications("user_1", status=NotificationStatus.READ)
        assert len(read) == 1
        assert read[0].id == "notif_2"

    def test_get_unread_count(self):
        """Test getting unread notification count."""
        store = InAppNotificationStore()

        for i in range(5):
            store.add_notification(
                SharingNotification(
                    id=f"notif_{i}",
                    user_id="user_1",
                    notification_type=NotificationType.ITEM_SHARED,
                    title=f"Test {i}",
                    message=f"Test {i}",
                )
            )

        assert store.get_unread_count("user_1") == 5
        assert store.get_unread_count("user_2") == 0

    def test_mark_as_read(self):
        """Test marking notification as read."""
        store = InAppNotificationStore()

        store.add_notification(
            SharingNotification(
                id="notif_1",
                user_id="user_1",
                notification_type=NotificationType.ITEM_SHARED,
                title="Test",
                message="Test",
            )
        )

        assert store.get_unread_count("user_1") == 1

        result = store.mark_as_read("notif_1", "user_1")
        assert result is True
        assert store.get_unread_count("user_1") == 0

        notifications = store.get_notifications("user_1")
        assert notifications[0].status == NotificationStatus.READ
        assert notifications[0].read_at is not None

    def test_mark_as_read_not_found(self):
        """Test marking non-existent notification as read."""
        store = InAppNotificationStore()
        result = store.mark_as_read("notif_999", "user_1")
        assert result is False

    def test_mark_all_as_read(self):
        """Test marking all notifications as read."""
        store = InAppNotificationStore()

        for i in range(3):
            store.add_notification(
                SharingNotification(
                    id=f"notif_{i}",
                    user_id="user_1",
                    notification_type=NotificationType.ITEM_SHARED,
                    title=f"Test {i}",
                    message=f"Test {i}",
                )
            )

        count = store.mark_all_as_read("user_1")
        assert count == 3
        assert store.get_unread_count("user_1") == 0

    def test_dismiss_notification(self):
        """Test dismissing a notification."""
        store = InAppNotificationStore()

        store.add_notification(
            SharingNotification(
                id="notif_1",
                user_id="user_1",
                notification_type=NotificationType.ITEM_SHARED,
                title="Test",
                message="Test",
            )
        )

        result = store.dismiss_notification("notif_1", "user_1")
        assert result is True

        notifications = store.get_notifications("user_1")
        assert notifications[0].status == NotificationStatus.DISMISSED

    def test_notification_limit_per_user(self):
        """Test that notifications are limited per user."""
        store = InAppNotificationStore()

        # Add 110 notifications (limit is 100)
        for i in range(110):
            store.add_notification(
                SharingNotification(
                    id=f"notif_{i}",
                    user_id="user_1",
                    notification_type=NotificationType.ITEM_SHARED,
                    title=f"Test {i}",
                    message=f"Test {i}",
                )
            )

        # Should only keep 100
        notifications = store.get_notifications("user_1", limit=200)
        assert len(notifications) == 100

        # Most recent should be first
        assert notifications[0].id == "notif_109"

    def test_preferences_default(self):
        """Test default notification preferences."""
        store = InAppNotificationStore()
        prefs = store.get_preferences("user_1")

        assert prefs.user_id == "user_1"
        assert prefs.email_on_share is True
        assert prefs.in_app_enabled is True
        assert prefs.telegram_enabled is False

    def test_preferences_set_and_get(self):
        """Test setting and getting preferences."""
        store = InAppNotificationStore()

        new_prefs = NotificationPreferences(
            user_id="user_1",
            email_on_share=False,
            email_on_unshare=True,
            in_app_enabled=True,
            webhook_url="https://example.com/hook",
        )

        store.set_preferences(new_prefs)
        prefs = store.get_preferences("user_1")

        assert prefs.email_on_share is False
        assert prefs.email_on_unshare is True
        assert prefs.webhook_url == "https://example.com/hook"


class TestSharingNotifier:
    """Test the SharingNotifier class."""

    @pytest.mark.asyncio
    async def test_notify_item_shared_in_app(self):
        """Test sending in-app notification for item shared."""
        store = InAppNotificationStore()
        notifier = SharingNotifier(store=store)

        results = await notifier.notify_item_shared(
            item_id="km_123",
            item_title="Important Doc",
            from_user_id="user_alice",
            from_user_name="Alice",
            to_user_id="user_bob",
            permissions=["read"],
        )

        assert results["in_app"] is True

        notifications = store.get_notifications("user_bob")
        assert len(notifications) == 1
        assert notifications[0].notification_type == NotificationType.ITEM_SHARED
        assert "Alice" in notifications[0].message
        assert "Important Doc" in notifications[0].message

    @pytest.mark.asyncio
    async def test_notify_item_shared_respects_preferences(self):
        """Test that notifications respect user preferences."""
        store = InAppNotificationStore()

        # Disable in-app notifications
        store.set_preferences(
            NotificationPreferences(
                user_id="user_bob",
                in_app_enabled=False,
            )
        )

        notifier = SharingNotifier(store=store)

        results = await notifier.notify_item_shared(
            item_id="km_123",
            item_title="Test",
            from_user_id="user_alice",
            from_user_name="Alice",
            to_user_id="user_bob",
        )

        assert "in_app" not in results

        notifications = store.get_notifications("user_bob")
        assert len(notifications) == 0

    @pytest.mark.asyncio
    async def test_notify_share_revoked(self):
        """Test notification for share revocation."""
        store = InAppNotificationStore()
        notifier = SharingNotifier(store=store)

        results = await notifier.notify_share_revoked(
            item_id="km_123",
            item_title="Secret Doc",
            from_user_id="user_alice",
            from_user_name="Alice",
            to_user_id="user_bob",
        )

        assert results["in_app"] is True

        notifications = store.get_notifications("user_bob")
        assert len(notifications) == 1
        assert notifications[0].notification_type == NotificationType.ITEM_UNSHARED

    @pytest.mark.asyncio
    async def test_notify_permission_changed(self):
        """Test notification for permission changes."""
        store = InAppNotificationStore()
        notifier = SharingNotifier(store=store)

        results = await notifier.notify_permission_changed(
            item_id="km_123",
            item_title="Shared Doc",
            from_user_id="user_alice",
            from_user_name="Alice",
            to_user_id="user_bob",
            old_permissions=["read"],
            new_permissions=["read", "write"],
        )

        assert results["in_app"] is True

        notifications = store.get_notifications("user_bob")
        assert len(notifications) == 1
        assert notifications[0].notification_type == NotificationType.PERMISSION_CHANGED
        assert notifications[0].metadata["old_permissions"] == ["read"]
        assert notifications[0].metadata["new_permissions"] == ["read", "write"]

    @pytest.mark.asyncio
    async def test_notify_share_expiring(self):
        """Test notification for expiring shares."""
        store = InAppNotificationStore()
        notifier = SharingNotifier(store=store)

        results = await notifier.notify_share_expiring(
            item_id="km_123",
            item_title="Temp Doc",
            to_user_id="user_bob",
            expires_at=datetime.now() + timedelta(days=3),
            days_remaining=3,
        )

        assert results["in_app"] is True

        notifications = store.get_notifications("user_bob")
        assert len(notifications) == 1
        assert notifications[0].notification_type == NotificationType.SHARE_EXPIRING
        assert "3 days" in notifications[0].message

    @pytest.mark.asyncio
    async def test_webhook_notification(self):
        """Test sending webhook notification."""
        store = InAppNotificationStore()
        store.set_preferences(
            NotificationPreferences(
                user_id="user_bob",
                webhook_url="https://example.com/hook",
            )
        )

        notifier = SharingNotifier(store=store)

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"{}"
            mock_urlopen.return_value.__enter__.return_value = mock_response

            results = await notifier.notify_item_shared(
                item_id="km_123",
                item_title="Test",
                from_user_id="user_alice",
                from_user_name="Alice",
                to_user_id="user_bob",
            )

            assert results["webhook"] is True
            mock_urlopen.assert_called_once()


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_notify_item_shared_function(self):
        """Test the convenience function for item sharing notification."""
        # Reset global store
        import aragora.knowledge.mound.notifications as notif_module

        notif_module._notification_store = None

        results = await notify_item_shared(
            item_id="km_test",
            item_title="Test Item",
            from_user_id="alice",
            from_user_name="Alice",
            to_user_id="bob",
        )

        assert results["in_app"] is True

    def test_get_notifications_for_user_function(self):
        """Test the convenience function for getting notifications."""
        import aragora.knowledge.mound.notifications as notif_module

        notif_module._notification_store = None
        store = get_notification_store()

        store.add_notification(
            SharingNotification(
                id="test_1",
                user_id="bob",
                notification_type=NotificationType.ITEM_SHARED,
                title="Test",
                message="Test",
            )
        )

        notifications = get_notifications_for_user("bob")
        assert len(notifications) == 1

    def test_get_unread_count_function(self):
        """Test the convenience function for unread count."""
        import aragora.knowledge.mound.notifications as notif_module

        notif_module._notification_store = None
        store = get_notification_store()

        for i in range(3):
            store.add_notification(
                SharingNotification(
                    id=f"test_{i}",
                    user_id="charlie",
                    notification_type=NotificationType.ITEM_SHARED,
                    title=f"Test {i}",
                    message=f"Test {i}",
                )
            )

        count = get_unread_count("charlie")
        assert count == 3

    def test_mark_notification_read_function(self):
        """Test the convenience function for marking read."""
        import aragora.knowledge.mound.notifications as notif_module

        notif_module._notification_store = None
        store = get_notification_store()

        store.add_notification(
            SharingNotification(
                id="mark_test",
                user_id="dave",
                notification_type=NotificationType.ITEM_SHARED,
                title="Test",
                message="Test",
            )
        )

        result = mark_notification_read("mark_test", "dave")
        assert result is True
        assert get_unread_count("dave") == 0

    def test_mark_all_notifications_read_function(self):
        """Test the convenience function for marking all read."""
        import aragora.knowledge.mound.notifications as notif_module

        notif_module._notification_store = None
        store = get_notification_store()

        for i in range(5):
            store.add_notification(
                SharingNotification(
                    id=f"all_test_{i}",
                    user_id="eve",
                    notification_type=NotificationType.ITEM_SHARED,
                    title=f"Test {i}",
                    message=f"Test {i}",
                )
            )

        count = mark_all_notifications_read("eve")
        assert count == 5
        assert get_unread_count("eve") == 0

    def test_preferences_functions(self):
        """Test the preference convenience functions."""
        import aragora.knowledge.mound.notifications as notif_module

        notif_module._notification_store = None

        prefs = get_notification_preferences("frank")
        assert prefs.user_id == "frank"
        assert prefs.email_on_share is True

        new_prefs = NotificationPreferences(
            user_id="frank",
            email_on_share=False,
        )
        set_notification_preferences(new_prefs)

        updated = get_notification_preferences("frank")
        assert updated.email_on_share is False


class TestNotificationTypes:
    """Test notification type enum."""

    def test_all_notification_types(self):
        """Test that all expected notification types exist."""
        assert NotificationType.ITEM_SHARED.value == "item_shared"
        assert NotificationType.ITEM_UNSHARED.value == "item_unshared"
        assert NotificationType.PERMISSION_CHANGED.value == "permission_changed"
        assert NotificationType.SHARE_EXPIRING.value == "share_expiring"
        assert NotificationType.SHARE_EXPIRED.value == "share_expired"
        assert NotificationType.FEDERATION_SYNC.value == "federation_sync"


class TestNotificationChannels:
    """Test notification channel enum."""

    def test_all_channels(self):
        """Test that all expected channels exist."""
        assert NotificationChannel.IN_APP.value == "in_app"
        assert NotificationChannel.EMAIL.value == "email"
        assert NotificationChannel.WEBHOOK.value == "webhook"
        assert NotificationChannel.TELEGRAM.value == "telegram"


class TestNotificationStatus:
    """Test notification status enum."""

    def test_all_statuses(self):
        """Test that all expected statuses exist."""
        assert NotificationStatus.UNREAD.value == "unread"
        assert NotificationStatus.READ.value == "read"
        assert NotificationStatus.DISMISSED.value == "dismissed"
