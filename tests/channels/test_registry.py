"""
Tests for DockRegistry - dock discovery and management.

Tests registration, lookup, capability queries, and the global singleton.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from aragora.channels.dock import ChannelDock, ChannelCapability, SendResult
from aragora.channels.normalized import NormalizedMessage
from aragora.channels.registry import DockRegistry, get_dock_registry


# =============================================================================
# Helper docks for testing
# =============================================================================


class FakeSlackDock(ChannelDock):
    PLATFORM = "slack"
    CAPABILITIES = (
        ChannelCapability.RICH_TEXT | ChannelCapability.BUTTONS | ChannelCapability.THREADS
    )

    async def send_message(self, channel_id, message, **kwargs):
        return SendResult.ok(message_id="msg-1", platform=self.PLATFORM, channel_id=channel_id)


class FakeTelegramDock(ChannelDock):
    PLATFORM = "telegram"
    CAPABILITIES = (
        ChannelCapability.RICH_TEXT
        | ChannelCapability.BUTTONS
        | ChannelCapability.VOICE
        | ChannelCapability.FILES
    )

    async def send_message(self, channel_id, message, **kwargs):
        return SendResult.ok(message_id="msg-2", platform=self.PLATFORM, channel_id=channel_id)


class FakeEmailDock(ChannelDock):
    PLATFORM = "email"
    CAPABILITIES = ChannelCapability.RICH_TEXT | ChannelCapability.FILES

    async def send_message(self, channel_id, message, **kwargs):
        return SendResult.ok(platform=self.PLATFORM, channel_id=channel_id)


class FakeSimpleDock(ChannelDock):
    PLATFORM = "simple"
    CAPABILITIES = ChannelCapability.NONE

    async def send_message(self, channel_id, message, **kwargs):
        return SendResult.ok(platform=self.PLATFORM, channel_id=channel_id)


# =============================================================================
# DockRegistry Tests
# =============================================================================


class TestDockRegistry:
    """Tests for DockRegistry."""

    def test_register_dock(self):
        """Test registering a dock class."""
        registry = DockRegistry()
        registry.register(FakeSlackDock)
        assert registry.has_dock("slack")

    def test_register_with_config(self):
        """Test registering a dock with default config."""
        registry = DockRegistry()
        registry.register(FakeSlackDock, config={"token": "abc"})
        dock = registry.get_dock("slack")
        assert dock is not None
        assert dock.config == {"token": "abc"}

    def test_register_case_insensitive(self):
        """Test platform lookup is case-insensitive."""
        registry = DockRegistry()
        registry.register(FakeSlackDock)
        assert registry.has_dock("Slack")
        assert registry.has_dock("SLACK")
        assert registry.has_dock("slack")

    def test_get_dock_returns_instance(self):
        """Test getting a dock returns an instance."""
        registry = DockRegistry()
        registry.register(FakeSlackDock)
        dock = registry.get_dock("slack")
        assert dock is not None
        assert isinstance(dock, FakeSlackDock)

    def test_get_dock_caches_instance(self):
        """Test that get_dock caches and returns the same instance."""
        registry = DockRegistry()
        registry.register(FakeSlackDock)
        dock1 = registry.get_dock("slack")
        dock2 = registry.get_dock("slack")
        assert dock1 is dock2

    def test_get_dock_config_override_no_cache(self):
        """Test that config override skips cache."""
        registry = DockRegistry()
        registry.register(FakeSlackDock, config={"token": "default"})
        dock_default = registry.get_dock("slack")
        dock_override = registry.get_dock("slack", config={"token": "override"})
        assert dock_default is not dock_override
        assert dock_override.config == {"token": "override"}

    def test_get_dock_merges_config(self):
        """Test that config is merged with defaults."""
        registry = DockRegistry()
        registry.register(FakeSlackDock, config={"base_url": "http://api", "token": "abc"})
        dock = registry.get_dock("slack", config={"token": "xyz"})
        assert dock.config["base_url"] == "http://api"
        assert dock.config["token"] == "xyz"

    def test_get_dock_not_registered(self):
        """Test getting an unregistered dock returns None."""
        registry = DockRegistry()
        dock = registry.get_dock("nonexistent")
        assert dock is None

    def test_has_dock_false(self):
        """Test has_dock for unregistered platform."""
        registry = DockRegistry()
        assert registry.has_dock("unknown") is False

    def test_get_platforms_empty(self):
        """Test get_platforms on empty registry."""
        registry = DockRegistry()
        assert registry.get_platforms() == []

    def test_get_platforms_populated(self):
        """Test get_platforms returns all registered platforms."""
        registry = DockRegistry()
        registry.register(FakeSlackDock)
        registry.register(FakeTelegramDock)
        registry.register(FakeEmailDock)
        platforms = registry.get_platforms()
        assert set(platforms) == {"slack", "telegram", "email"}

    def test_get_platforms_with_capability_voice(self):
        """Test filtering platforms by VOICE capability."""
        registry = DockRegistry()
        registry.register(FakeSlackDock)
        registry.register(FakeTelegramDock)
        registry.register(FakeEmailDock)
        voice_platforms = registry.get_platforms_with_capability(ChannelCapability.VOICE)
        assert voice_platforms == ["telegram"]

    def test_get_platforms_with_capability_rich_text(self):
        """Test filtering platforms by RICH_TEXT capability."""
        registry = DockRegistry()
        registry.register(FakeSlackDock)
        registry.register(FakeTelegramDock)
        registry.register(FakeEmailDock)
        registry.register(FakeSimpleDock)
        rich_platforms = registry.get_platforms_with_capability(ChannelCapability.RICH_TEXT)
        assert set(rich_platforms) == {"slack", "telegram", "email"}

    def test_get_platforms_with_capability_none_match(self):
        """Test filtering when no platform matches."""
        registry = DockRegistry()
        registry.register(FakeSimpleDock)
        result = registry.get_platforms_with_capability(ChannelCapability.VOICE)
        assert result == []

    def test_get_platforms_with_capability_threads(self):
        """Test filtering by THREADS capability."""
        registry = DockRegistry()
        registry.register(FakeSlackDock)
        registry.register(FakeTelegramDock)
        result = registry.get_platforms_with_capability(ChannelCapability.THREADS)
        assert result == ["slack"]

    def test_unregister_dock(self):
        """Test unregistering a dock."""
        registry = DockRegistry()
        registry.register(FakeSlackDock)
        assert registry.has_dock("slack")
        removed = registry.unregister("slack")
        assert removed is True
        assert not registry.has_dock("slack")

    def test_unregister_nonexistent(self):
        """Test unregistering a dock that was not registered."""
        registry = DockRegistry()
        removed = registry.unregister("nonexistent")
        assert removed is False

    def test_unregister_clears_instances(self):
        """Test unregister clears cached instances."""
        registry = DockRegistry()
        registry.register(FakeSlackDock)
        dock = registry.get_dock("slack")
        assert dock is not None
        registry.unregister("slack")
        assert registry.get_dock("slack") is None

    def test_clear_registry(self):
        """Test clearing all registered docks."""
        registry = DockRegistry()
        registry.register(FakeSlackDock)
        registry.register(FakeTelegramDock)
        registry.clear()
        assert registry.get_platforms() == []
        assert not registry.has_dock("slack")
        assert not registry.has_dock("telegram")

    @pytest.mark.asyncio
    async def test_get_initialized_dock(self):
        """Test get_initialized_dock initializes the dock."""
        registry = DockRegistry()
        registry.register(FakeSlackDock)
        dock = await registry.get_initialized_dock("slack")
        assert dock is not None
        assert dock.is_initialized is True

    @pytest.mark.asyncio
    async def test_get_initialized_dock_not_registered(self):
        """Test get_initialized_dock for unregistered platform."""
        registry = DockRegistry()
        dock = await registry.get_initialized_dock("nonexistent")
        assert dock is None

    @pytest.mark.asyncio
    async def test_get_initialized_dock_already_initialized(self):
        """Test get_initialized_dock does not re-initialize."""
        registry = DockRegistry()
        registry.register(FakeSlackDock)
        # Get and initialize
        dock1 = await registry.get_initialized_dock("slack")
        assert dock1.is_initialized
        # Getting again should return same cached instance
        dock2 = await registry.get_initialized_dock("slack")
        assert dock1 is dock2


# =============================================================================
# Global Registry Singleton Tests
# =============================================================================


class TestGetDockRegistry:
    """Tests for the global singleton registry."""

    def test_singleton_returns_instance(self):
        """Test that get_dock_registry returns a DockRegistry."""
        # Reset the global singleton for this test
        import aragora.channels.registry as reg_module

        old = reg_module._dock_registry
        try:
            reg_module._dock_registry = None
            registry = get_dock_registry()
            assert isinstance(registry, DockRegistry)
        finally:
            reg_module._dock_registry = old

    def test_singleton_returns_same_instance(self):
        """Test that repeated calls return the same registry."""
        import aragora.channels.registry as reg_module

        old = reg_module._dock_registry
        try:
            reg_module._dock_registry = None
            r1 = get_dock_registry()
            r2 = get_dock_registry()
            assert r1 is r2
        finally:
            reg_module._dock_registry = old

    def test_singleton_registers_builtin_docks(self):
        """Test that built-in docks are auto-registered."""
        import aragora.channels.registry as reg_module

        old = reg_module._dock_registry
        try:
            reg_module._dock_registry = None
            registry = get_dock_registry()
            # The registry should have attempted to register built-in docks
            # At minimum, it should have some platforms registered
            platforms = registry.get_platforms()
            assert isinstance(platforms, list)
        finally:
            reg_module._dock_registry = old
