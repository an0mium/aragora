"""
Tests for chat metadata enrichment.

Tests cover:
- ChatUser enrichment fields
- ChatChannel enrichment fields
- MetadataCache TTL functionality
- Context building for debate prompts
"""

from datetime import datetime, timedelta

import pytest


class TestChatUserEnrichment:
    """Tests for ChatUser enrichment fields."""

    def test_user_default_role_is_unknown(self):
        """User role defaults to UNKNOWN."""
        from aragora.connectors.chat.models import ChatUser, UserRole

        user = ChatUser(id="u1", platform="slack")

        assert user.role == UserRole.UNKNOWN

    def test_user_with_enrichment_fields(self):
        """User can have enrichment fields."""
        from aragora.connectors.chat.models import ChatUser, UserRole

        user = ChatUser(
            id="u1",
            platform="slack",
            timezone="America/New_York",
            language="en",
            locale="en-US",
            role=UserRole.ADMIN,
            status="online",
        )

        assert user.timezone == "America/New_York"
        assert user.language == "en"
        assert user.locale == "en-US"
        assert user.role == UserRole.ADMIN
        assert user.status == "online"

    def test_user_is_enriched_property(self):
        """is_enriched returns True when enrichment data present."""
        from aragora.connectors.chat.models import ChatUser

        user_plain = ChatUser(id="u1", platform="slack")
        user_enriched = ChatUser(id="u2", platform="slack", timezone="UTC")

        assert not user_plain.is_enriched
        assert user_enriched.is_enriched

    def test_user_to_context_dict(self):
        """User context dict contains enrichment fields."""
        from aragora.connectors.chat.models import ChatUser, UserRole

        user = ChatUser(
            id="u1",
            platform="slack",
            username="johndoe",
            display_name="John Doe",
            timezone="Europe/London",
            language="en",
            role=UserRole.MODERATOR,
        )

        ctx = user.to_context_dict()

        assert ctx["user_id"] == "u1"
        assert ctx["username"] == "johndoe"
        assert ctx["timezone"] == "Europe/London"
        assert ctx["language"] == "en"
        assert ctx["role"] == "moderator"


class TestChatChannelEnrichment:
    """Tests for ChatChannel enrichment fields."""

    def test_channel_default_type_is_unknown(self):
        """Channel type defaults to UNKNOWN."""
        from aragora.connectors.chat.models import ChannelType, ChatChannel

        channel = ChatChannel(id="c1", platform="slack")

        assert channel.channel_type == ChannelType.UNKNOWN

    def test_channel_with_enrichment_fields(self):
        """Channel can have enrichment fields."""
        from aragora.connectors.chat.models import ChannelType, ChatChannel

        channel = ChatChannel(
            id="c1",
            platform="slack",
            name="general",
            channel_type=ChannelType.PUBLIC,
            topic="General discussions",
            description="Main channel for team updates",
            member_count=100,
        )

        assert channel.topic == "General discussions"
        assert channel.description == "Main channel for team updates"
        assert channel.member_count == 100
        assert channel.channel_type == ChannelType.PUBLIC

    def test_channel_is_enriched_property(self):
        """is_enriched returns True when enrichment data present."""
        from aragora.connectors.chat.models import ChatChannel

        channel_plain = ChatChannel(id="c1", platform="slack")
        channel_enriched = ChatChannel(id="c2", platform="slack", topic="Test")

        assert not channel_plain.is_enriched
        assert channel_enriched.is_enriched

    def test_channel_to_context_dict(self):
        """Channel context dict contains enrichment fields."""
        from aragora.connectors.chat.models import ChannelType, ChatChannel

        channel = ChatChannel(
            id="c1",
            platform="slack",
            name="engineering",
            channel_type=ChannelType.PRIVATE,
            topic="Engineering discussions",
            member_count=25,
        )

        ctx = channel.to_context_dict()

        assert ctx["channel_id"] == "c1"
        assert ctx["channel_name"] == "engineering"
        assert ctx["channel_type"] == "private"
        assert ctx["topic"] == "Engineering discussions"
        assert ctx["member_count"] == 25


class TestMetadataCache:
    """Tests for MetadataCache TTL functionality."""

    def test_cache_set_and_get_user(self):
        """Cache can store and retrieve user metadata."""
        from aragora.connectors.chat.models import MetadataCache

        cache = MetadataCache()
        cache.set_user("u1", "slack", {"timezone": "UTC"})

        result = cache.get_user("u1", "slack")

        assert result == {"timezone": "UTC"}

    def test_cache_set_and_get_channel(self):
        """Cache can store and retrieve channel metadata."""
        from aragora.connectors.chat.models import MetadataCache

        cache = MetadataCache()
        cache.set_channel("c1", "slack", {"topic": "Test"})

        result = cache.get_channel("c1", "slack")

        assert result == {"topic": "Test"}

    def test_cache_returns_none_for_missing(self):
        """Cache returns None for missing entries."""
        from aragora.connectors.chat.models import MetadataCache

        cache = MetadataCache()

        assert cache.get_user("nonexistent", "slack") is None
        assert cache.get_channel("nonexistent", "slack") is None

    def test_cache_invalidate_user(self):
        """Cache can invalidate user entries."""
        from aragora.connectors.chat.models import MetadataCache

        cache = MetadataCache()
        cache.set_user("u1", "slack", {"timezone": "UTC"})
        cache.invalidate_user("u1", "slack")

        assert cache.get_user("u1", "slack") is None

    def test_cache_invalidate_channel(self):
        """Cache can invalidate channel entries."""
        from aragora.connectors.chat.models import MetadataCache

        cache = MetadataCache()
        cache.set_channel("c1", "slack", {"topic": "Test"})
        cache.invalidate_channel("c1", "slack")

        assert cache.get_channel("c1", "slack") is None

    def test_cache_clear(self):
        """Cache can clear all entries."""
        from aragora.connectors.chat.models import MetadataCache

        cache = MetadataCache()
        cache.set_user("u1", "slack", {"timezone": "UTC"})
        cache.set_channel("c1", "slack", {"topic": "Test"})
        cache.clear()

        assert cache.get_user("u1", "slack") is None
        assert cache.get_channel("c1", "slack") is None

    def test_cache_stats(self):
        """Cache returns statistics."""
        from aragora.connectors.chat.models import MetadataCache

        cache = MetadataCache(default_ttl=7200)
        cache.set_user("u1", "slack", {})
        cache.set_user("u2", "slack", {})
        cache.set_channel("c1", "slack", {})

        stats = cache.stats()

        assert stats["user_entries"] == 2
        assert stats["channel_entries"] == 1
        assert stats["default_ttl_seconds"] == 7200


class TestBuildChatContext:
    """Tests for build_chat_context function."""

    def test_build_context_with_user_and_channel(self):
        """Context includes both user and channel."""
        from aragora.connectors.chat.models import (
            ChatChannel,
            ChatUser,
            build_chat_context,
        )

        user = ChatUser(id="u1", platform="slack", timezone="UTC")
        channel = ChatChannel(id="c1", platform="slack", topic="Test")

        ctx = build_chat_context(user=user, channel=channel)

        assert "user" in ctx
        assert "channel" in ctx
        assert ctx["user"]["timezone"] == "UTC"
        assert ctx["channel"]["topic"] == "Test"

    def test_build_context_user_only(self):
        """Context can include only user."""
        from aragora.connectors.chat.models import ChatUser, build_chat_context

        user = ChatUser(id="u1", platform="slack", timezone="UTC")

        ctx = build_chat_context(user=user, include_channel=False)

        assert "user" in ctx
        assert "channel" not in ctx

    def test_build_context_channel_only(self):
        """Context can include only channel."""
        from aragora.connectors.chat.models import ChatChannel, build_chat_context

        channel = ChatChannel(id="c1", platform="slack", topic="Test")

        ctx = build_chat_context(channel=channel, include_user=False)

        assert "user" not in ctx
        assert "channel" in ctx

    def test_build_context_filters_none_values(self):
        """Context filters out None values."""
        from aragora.connectors.chat.models import ChatUser, build_chat_context

        user = ChatUser(
            id="u1",
            platform="slack",
            timezone="UTC",
            # language is None
        )

        ctx = build_chat_context(user=user)

        # None values should not be in the context
        assert "language" not in ctx["user"] or ctx["user"]["language"] is not None


class TestGlobalMetadataCache:
    """Tests for global metadata cache singleton."""

    def test_get_metadata_cache_returns_same_instance(self):
        """get_metadata_cache returns singleton instance."""
        from aragora.connectors.chat.models import get_metadata_cache

        cache1 = get_metadata_cache()
        cache2 = get_metadata_cache()

        assert cache1 is cache2


class TestUserRoleEnum:
    """Tests for UserRole enum."""

    def test_user_role_values(self):
        """UserRole has expected values."""
        from aragora.connectors.chat.models import UserRole

        assert UserRole.OWNER.value == "owner"
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.MODERATOR.value == "moderator"
        assert UserRole.MEMBER.value == "member"
        assert UserRole.GUEST.value == "guest"
        assert UserRole.UNKNOWN.value == "unknown"


class TestChannelTypeEnum:
    """Tests for ChannelType enum."""

    def test_channel_type_values(self):
        """ChannelType has expected values."""
        from aragora.connectors.chat.models import ChannelType

        assert ChannelType.PUBLIC.value == "public"
        assert ChannelType.PRIVATE.value == "private"
        assert ChannelType.DM.value == "dm"
        assert ChannelType.GROUP_DM.value == "group_dm"
        assert ChannelType.THREAD.value == "thread"
        assert ChannelType.UNKNOWN.value == "unknown"
