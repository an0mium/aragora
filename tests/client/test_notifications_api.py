"""Tests for NotificationsAPI client resource."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.client.client import AragoraClient
from aragora.client.resources.notifications import (
    Notification,
    NotificationChannel,
    NotificationPreference,
    NotificationStats,
    NotificationsAPI,
)


@pytest.fixture
def mock_client() -> AragoraClient:
    client = MagicMock(spec=AragoraClient)
    return client


@pytest.fixture
def api(mock_client: AragoraClient) -> NotificationsAPI:
    return NotificationsAPI(mock_client)


SAMPLE_PREFERENCE = {
    "event_type": "debate_completed",
    "enabled": True,
    "channels": ["email", "slack"],
    "frequency": "immediate",
}

SAMPLE_CHANNEL = {
    "type": "slack",
    "enabled": True,
    "target": "#decisions",
    "settings": {"mention_users": True},
}

SAMPLE_NOTIFICATION = {
    "id": "notif-001",
    "type": "debate_completed",
    "title": "Debate Finished",
    "message": "The debate on Redis vs Memcached has concluded.",
    "status": "delivered",
    "priority": "normal",
    "created_at": "2026-01-15T10:00:00Z",
    "sent_at": "2026-01-15T10:00:05Z",
    "read_at": None,
    "metadata": {"debate_id": "deb-42"},
}

SAMPLE_STATS = {
    "total_sent": 100,
    "total_delivered": 95,
    "total_failed": 5,
    "total_read": 60,
    "delivery_rate": 0.95,
    "read_rate": 0.63,
}


# =========================================================================
# Preferences
# =========================================================================


class TestGetPreferences:
    def test_get_preferences(self, api: NotificationsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"preferences": [SAMPLE_PREFERENCE]}
        prefs = api.get_preferences()
        assert len(prefs) == 1
        assert isinstance(prefs[0], NotificationPreference)
        assert prefs[0].event_type == "debate_completed"
        assert prefs[0].channels == ["email", "slack"]
        mock_client._get.assert_called_once_with("/api/v1/notifications/preferences")

    def test_get_preferences_empty(self, api: NotificationsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"preferences": []}
        prefs = api.get_preferences()
        assert prefs == []

    def test_get_preferences_missing_key(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {}
        prefs = api.get_preferences()
        assert prefs == []

    @pytest.mark.asyncio
    async def test_get_preferences_async(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"preferences": [SAMPLE_PREFERENCE]})
        prefs = await api.get_preferences_async()
        assert len(prefs) == 1
        assert prefs[0].frequency == "immediate"


class TestUpdatePreference:
    def test_update_preference_all_fields(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._patch.return_value = {"preference": SAMPLE_PREFERENCE}
        result = api.update_preference(
            event_type="debate_completed",
            enabled=True,
            channels=["email", "slack"],
            frequency="immediate",
        )
        assert isinstance(result, NotificationPreference)
        assert result.event_type == "debate_completed"
        body = mock_client._patch.call_args[0][1]
        assert body["event_type"] == "debate_completed"
        assert body["enabled"] is True
        assert body["channels"] == ["email", "slack"]
        assert body["frequency"] == "immediate"

    def test_update_preference_minimal(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._patch.return_value = {"preference": SAMPLE_PREFERENCE}
        api.update_preference(event_type="debate_completed")
        body = mock_client._patch.call_args[0][1]
        assert body == {"event_type": "debate_completed"}
        assert "enabled" not in body
        assert "channels" not in body
        assert "frequency" not in body

    def test_update_preference_partial(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._patch.return_value = {"preference": SAMPLE_PREFERENCE}
        api.update_preference(event_type="debate_completed", frequency="daily")
        body = mock_client._patch.call_args[0][1]
        assert body["event_type"] == "debate_completed"
        assert body["frequency"] == "daily"
        assert "enabled" not in body
        assert "channels" not in body

    def test_update_preference_fallback_parse(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        """When response has no 'preference' key, parse the response itself."""
        mock_client._patch.return_value = SAMPLE_PREFERENCE
        result = api.update_preference(event_type="debate_completed")
        assert result.event_type == "debate_completed"

    @pytest.mark.asyncio
    async def test_update_preference_async(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._patch_async = AsyncMock(return_value={"preference": SAMPLE_PREFERENCE})
        result = await api.update_preference_async(
            event_type="debate_completed",
            enabled=False,
        )
        assert isinstance(result, NotificationPreference)
        body = mock_client._patch_async.call_args[0][1]
        assert body["enabled"] is False


class TestMuteUnmute:
    def test_mute_all_indefinite(self, api: NotificationsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {}
        result = api.mute_all()
        assert result is True
        body = mock_client._post.call_args[0][1]
        assert body["action"] == "mute"
        assert "duration_hours" not in body

    def test_mute_all_with_duration(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = {}
        result = api.mute_all(duration_hours=4)
        assert result is True
        body = mock_client._post.call_args[0][1]
        assert body["duration_hours"] == 4

    def test_unmute_all(self, api: NotificationsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {}
        result = api.unmute_all()
        assert result is True
        body = mock_client._post.call_args[0][1]
        assert body["action"] == "unmute"

    @pytest.mark.asyncio
    async def test_mute_all_async(self, api: NotificationsAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value={})
        result = await api.mute_all_async(duration_hours=8)
        assert result is True
        body = mock_client._post_async.call_args[0][1]
        assert body["duration_hours"] == 8

    @pytest.mark.asyncio
    async def test_unmute_all_async(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value={})
        result = await api.unmute_all_async()
        assert result is True


# =========================================================================
# Channels
# =========================================================================


class TestListChannels:
    def test_list_channels(self, api: NotificationsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"channels": [SAMPLE_CHANNEL]}
        channels = api.list_channels()
        assert len(channels) == 1
        assert isinstance(channels[0], NotificationChannel)
        assert channels[0].type == "slack"
        assert channels[0].target == "#decisions"
        assert channels[0].settings == {"mention_users": True}
        mock_client._get.assert_called_once_with("/api/v1/notifications/channels")

    def test_list_channels_empty(self, api: NotificationsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"channels": []}
        channels = api.list_channels()
        assert channels == []

    def test_list_channels_missing_key(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {}
        channels = api.list_channels()
        assert channels == []

    @pytest.mark.asyncio
    async def test_list_channels_async(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"channels": [SAMPLE_CHANNEL]})
        channels = await api.list_channels_async()
        assert len(channels) == 1
        assert channels[0].enabled is True


class TestConfigureChannel:
    def test_configure_channel_full(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = {"channel": SAMPLE_CHANNEL}
        result = api.configure_channel(
            channel_type="slack",
            enabled=True,
            target="#decisions",
            settings={"mention_users": True},
        )
        assert isinstance(result, NotificationChannel)
        assert result.type == "slack"
        body = mock_client._post.call_args[0][1]
        assert body["type"] == "slack"
        assert body["enabled"] is True
        assert body["target"] == "#decisions"
        assert body["settings"] == {"mention_users": True}

    def test_configure_channel_minimal(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = {"channel": {"type": "email", "enabled": True}}
        api.configure_channel(channel_type="email")
        body = mock_client._post.call_args[0][1]
        assert body == {"type": "email", "enabled": True}
        assert "target" not in body
        assert "settings" not in body

    def test_configure_channel_disabled(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = {"channel": {"type": "webhook", "enabled": False}}
        result = api.configure_channel(channel_type="webhook", enabled=False)
        assert result.enabled is False
        body = mock_client._post.call_args[0][1]
        assert body["enabled"] is False

    def test_configure_channel_fallback_parse(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        """When response has no 'channel' key, parse the response itself."""
        mock_client._post.return_value = SAMPLE_CHANNEL
        result = api.configure_channel(channel_type="slack", target="#decisions")
        assert result.type == "slack"

    @pytest.mark.asyncio
    async def test_configure_channel_async(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value={"channel": SAMPLE_CHANNEL})
        result = await api.configure_channel_async(
            channel_type="slack",
            target="#decisions",
        )
        assert result.type == "slack"


class TestTestChannel:
    def test_test_channel(self, api: NotificationsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {}
        result = api.test_channel("email")
        assert result is True
        body = mock_client._post.call_args[0][1]
        assert body == {"type": "email"}
        mock_client._post.assert_called_once_with(
            "/api/v1/notifications/channels/test", {"type": "email"}
        )

    @pytest.mark.asyncio
    async def test_test_channel_async(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value={})
        result = await api.test_channel_async("slack")
        assert result is True


# =========================================================================
# Notification History
# =========================================================================


class TestListNotifications:
    def test_list_default(self, api: NotificationsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"notifications": [SAMPLE_NOTIFICATION], "total": 1}
        notifications, total = api.list()
        assert len(notifications) == 1
        assert total == 1
        assert isinstance(notifications[0], Notification)
        assert notifications[0].id == "notif-001"
        assert notifications[0].title == "Debate Finished"
        params = mock_client._get.call_args[1]["params"]
        assert params["limit"] == 50
        assert params["offset"] == 0

    def test_list_with_filters(self, api: NotificationsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"notifications": [], "total": 0}
        api.list(status="delivered", type_filter="debate_completed", limit=10, offset=5)
        params = mock_client._get.call_args[1]["params"]
        assert params["status"] == "delivered"
        assert params["type"] == "debate_completed"
        assert params["limit"] == 10
        assert params["offset"] == 5

    def test_list_no_filters(self, api: NotificationsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"notifications": [], "total": 0}
        api.list()
        params = mock_client._get.call_args[1]["params"]
        assert "status" not in params
        assert "type" not in params

    def test_list_total_fallback(self, api: NotificationsAPI, mock_client: AragoraClient) -> None:
        """When 'total' is missing, fall back to len(notifications)."""
        mock_client._get.return_value = {"notifications": [SAMPLE_NOTIFICATION]}
        _, total = api.list()
        assert total == 1

    def test_list_empty(self, api: NotificationsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"notifications": [], "total": 0}
        notifications, total = api.list()
        assert notifications == []
        assert total == 0

    @pytest.mark.asyncio
    async def test_list_async(self, api: NotificationsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(
            return_value={"notifications": [SAMPLE_NOTIFICATION], "total": 1}
        )
        notifications, total = await api.list_async()
        assert total == 1
        assert notifications[0].type == "debate_completed"

    @pytest.mark.asyncio
    async def test_list_async_with_filters(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"notifications": [], "total": 0})
        await api.list_async(status="read", type_filter="alert", limit=25, offset=10)
        params = mock_client._get_async.call_args[1]["params"]
        assert params["status"] == "read"
        assert params["type"] == "alert"


class TestMarkAsRead:
    def test_mark_as_read(self, api: NotificationsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {}
        result = api.mark_as_read("notif-001")
        assert result is True
        mock_client._post.assert_called_once_with("/api/v1/notifications/notif-001/read", {})

    @pytest.mark.asyncio
    async def test_mark_as_read_async(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value={})
        result = await api.mark_as_read_async("notif-002")
        assert result is True
        mock_client._post_async.assert_called_once_with("/api/v1/notifications/notif-002/read", {})


class TestMarkAllAsRead:
    def test_mark_all_as_read(self, api: NotificationsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"count": 15}
        count = api.mark_all_as_read()
        assert count == 15
        mock_client._post.assert_called_once_with("/api/v1/notifications/read-all", {})

    def test_mark_all_as_read_zero(self, api: NotificationsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"count": 0}
        count = api.mark_all_as_read()
        assert count == 0

    def test_mark_all_as_read_missing_count(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = {}
        count = api.mark_all_as_read()
        assert count == 0

    @pytest.mark.asyncio
    async def test_mark_all_as_read_async(
        self, api: NotificationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value={"count": 7})
        count = await api.mark_all_as_read_async()
        assert count == 7


# =========================================================================
# Stats
# =========================================================================


class TestGetStats:
    def test_get_stats(self, api: NotificationsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_STATS
        stats = api.get_stats()
        assert isinstance(stats, NotificationStats)
        assert stats.total_sent == 100
        assert stats.total_delivered == 95
        assert stats.total_failed == 5
        assert stats.total_read == 60
        assert stats.delivery_rate == 0.95
        assert stats.read_rate == 0.63
        mock_client._get.assert_called_once_with("/api/v1/notifications/stats")

    def test_get_stats_defaults(self, api: NotificationsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {}
        stats = api.get_stats()
        assert stats.total_sent == 0
        assert stats.total_delivered == 0
        assert stats.total_failed == 0
        assert stats.total_read == 0
        assert stats.delivery_rate == 0.0
        assert stats.read_rate == 0.0

    @pytest.mark.asyncio
    async def test_get_stats_async(self, api: NotificationsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_STATS)
        stats = await api.get_stats_async()
        assert stats.total_sent == 100
        assert stats.delivery_rate == 0.95


# =========================================================================
# Parser / Edge Cases
# =========================================================================


class TestParseNotification:
    def test_parse_datetime_iso(self, api: NotificationsAPI) -> None:
        result = api._parse_notification(SAMPLE_NOTIFICATION)
        assert result.created_at is not None
        assert result.created_at.year == 2026
        assert result.sent_at is not None
        assert result.sent_at.year == 2026

    def test_parse_missing_datetimes(self, api: NotificationsAPI) -> None:
        data = {
            "id": "n1",
            "type": "alert",
            "title": "t",
            "message": "m",
            "status": "pending",
            "priority": "low",
        }
        result = api._parse_notification(data)
        assert result.created_at is None
        assert result.sent_at is None
        assert result.read_at is None

    def test_parse_invalid_datetime(self, api: NotificationsAPI) -> None:
        data = {
            **SAMPLE_NOTIFICATION,
            "created_at": "not-a-date",
            "sent_at": "bad",
            "read_at": "invalid",
        }
        result = api._parse_notification(data)
        assert result.created_at is None
        assert result.sent_at is None
        assert result.read_at is None

    def test_parse_read_at_datetime(self, api: NotificationsAPI) -> None:
        data = {**SAMPLE_NOTIFICATION, "read_at": "2026-01-15T11:00:00Z"}
        result = api._parse_notification(data)
        assert result.read_at is not None
        assert result.read_at.hour == 11

    def test_parse_defaults(self, api: NotificationsAPI) -> None:
        result = api._parse_notification({})
        assert result.id == ""
        assert result.type == ""
        assert result.title == ""
        assert result.message == ""
        assert result.status == "pending"
        assert result.priority == "normal"
        assert result.metadata == {}

    def test_parse_metadata(self, api: NotificationsAPI) -> None:
        result = api._parse_notification(SAMPLE_NOTIFICATION)
        assert result.metadata == {"debate_id": "deb-42"}


class TestParsePreference:
    def test_parse_preference_full(self, api: NotificationsAPI) -> None:
        result = api._parse_preference(SAMPLE_PREFERENCE)
        assert result.event_type == "debate_completed"
        assert result.enabled is True
        assert result.channels == ["email", "slack"]
        assert result.frequency == "immediate"

    def test_parse_preference_defaults(self, api: NotificationsAPI) -> None:
        result = api._parse_preference({})
        assert result.event_type == ""
        assert result.enabled is True
        assert result.channels == []
        assert result.frequency == "immediate"


class TestParseChannel:
    def test_parse_channel_full(self, api: NotificationsAPI) -> None:
        result = api._parse_channel(SAMPLE_CHANNEL)
        assert result.type == "slack"
        assert result.enabled is True
        assert result.target == "#decisions"
        assert result.settings == {"mention_users": True}

    def test_parse_channel_defaults(self, api: NotificationsAPI) -> None:
        result = api._parse_channel({})
        assert result.type == ""
        assert result.enabled is True
        assert result.target is None
        assert result.settings == {}


# =========================================================================
# Dataclasses
# =========================================================================


class TestDataclasses:
    def test_notification_channel_defaults(self) -> None:
        ch = NotificationChannel(type="email")
        assert ch.type == "email"
        assert ch.enabled is True
        assert ch.target is None
        assert ch.settings == {}

    def test_notification_channel_full(self) -> None:
        ch = NotificationChannel(
            type="webhook",
            enabled=False,
            target="https://example.com/hook",
            settings={"retry": True},
        )
        assert ch.type == "webhook"
        assert ch.enabled is False
        assert ch.target == "https://example.com/hook"
        assert ch.settings == {"retry": True}

    def test_notification_preference_defaults(self) -> None:
        pref = NotificationPreference(event_type="alert")
        assert pref.event_type == "alert"
        assert pref.enabled is True
        assert pref.channels == []
        assert pref.frequency == "immediate"

    def test_notification_preference_full(self) -> None:
        pref = NotificationPreference(
            event_type="debate_completed",
            enabled=False,
            channels=["email"],
            frequency="weekly",
        )
        assert pref.enabled is False
        assert pref.frequency == "weekly"

    def test_notification_defaults(self) -> None:
        n = Notification(
            id="n1", type="alert", title="T", message="M", status="pending", priority="low"
        )
        assert n.created_at is None
        assert n.sent_at is None
        assert n.read_at is None
        assert n.metadata == {}

    def test_notification_stats_defaults(self) -> None:
        stats = NotificationStats()
        assert stats.total_sent == 0
        assert stats.total_delivered == 0
        assert stats.total_failed == 0
        assert stats.total_read == 0
        assert stats.delivery_rate == 0.0
        assert stats.read_rate == 0.0

    def test_notification_stats_custom(self) -> None:
        stats = NotificationStats(
            total_sent=50,
            total_delivered=48,
            total_failed=2,
            total_read=30,
            delivery_rate=0.96,
            read_rate=0.625,
        )
        assert stats.total_sent == 50
        assert stats.delivery_rate == 0.96
