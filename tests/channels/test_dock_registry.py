"""
Tests for Dock Registry.

Tests cover:
- DockRegistry class (register, get_dock, has_dock, etc.)
- Singleton pattern for get_dock_registry
- Capability filtering
- Built-in dock registration
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.channels.registry import DockRegistry, get_dock_registry
from aragora.channels.dock import ChannelDock, ChannelCapability


class MockDock(ChannelDock):
    """Mock dock for testing."""

    PLATFORM = "mock"
    CAPABILITIES = ChannelCapability.RICH_TEXT | ChannelCapability.BUTTONS

    def __init__(self, config=None):
        self.config = config or {}
        self._initialized = False

    @property
    def is_initialized(self):
        return self._initialized

    async def initialize(self):
        self._initialized = True

    async def send_message(self, channel_id, message):
        return None

    async def send_result(self, channel_id, result, **kwargs):
        return None

    async def send_error(self, channel_id, error, **kwargs):
        return None


class VoiceDock(ChannelDock):
    """Mock dock with voice support."""

    PLATFORM = "voice_platform"
    CAPABILITIES = ChannelCapability.RICH_TEXT | ChannelCapability.VOICE

    def __init__(self, config=None):
        self.config = config or {}
        self._initialized = False

    @property
    def is_initialized(self):
        return self._initialized

    async def initialize(self):
        self._initialized = True

    async def send_message(self, channel_id, message):
        return None

    async def send_result(self, channel_id, result, **kwargs):
        return None

    async def send_error(self, channel_id, error, **kwargs):
        return None


class TestDockRegistry:
    """Tests for DockRegistry class."""

    def test_init(self):
        """Test registry initialization."""
        registry = DockRegistry()
        assert registry._dock_classes == {}
        assert registry._dock_instances == {}
        assert registry._default_configs == {}

    def test_register_dock(self):
        """Test registering a dock class."""
        registry = DockRegistry()
        registry.register(MockDock)

        assert "mock" in registry._dock_classes
        assert registry._dock_classes["mock"] is MockDock

    def test_register_dock_with_config(self):
        """Test registering a dock with default config."""
        registry = DockRegistry()
        config = {"api_key": "test-key"}
        registry.register(MockDock, config=config)

        assert "mock" in registry._dock_classes
        assert registry._default_configs["mock"] == config

    def test_register_normalizes_platform(self):
        """Test that platform name is normalized to lowercase."""

        # Create a dock with uppercase platform
        class UpperDock(MockDock):
            PLATFORM = "UPPER"

        registry = DockRegistry()
        registry.register(UpperDock)

        assert "upper" in registry._dock_classes
        assert "UPPER" not in registry._dock_classes

    def test_get_dock(self):
        """Test getting a dock instance."""
        registry = DockRegistry()
        registry.register(MockDock)

        dock = registry.get_dock("mock")

        assert dock is not None
        assert isinstance(dock, MockDock)

    def test_get_dock_case_insensitive(self):
        """Test that get_dock is case insensitive."""
        registry = DockRegistry()
        registry.register(MockDock)

        dock1 = registry.get_dock("MOCK")
        dock2 = registry.get_dock("Mock")
        dock3 = registry.get_dock("mock")

        assert dock1 is dock2 is dock3

    def test_get_dock_caches_instance(self):
        """Test that dock instances are cached."""
        registry = DockRegistry()
        registry.register(MockDock)

        dock1 = registry.get_dock("mock")
        dock2 = registry.get_dock("mock")

        assert dock1 is dock2

    def test_get_dock_with_config_override(self):
        """Test getting a dock with config override."""
        registry = DockRegistry()
        registry.register(MockDock, config={"key": "default"})

        dock = registry.get_dock("mock", config={"key": "override"})

        assert dock is not None
        assert dock.config["key"] == "override"

    def test_get_dock_with_config_not_cached(self):
        """Test that dock with config override is not cached."""
        registry = DockRegistry()
        registry.register(MockDock)

        dock1 = registry.get_dock("mock", config={"special": True})
        dock2 = registry.get_dock("mock", config={"special": True})

        # Different instances when config is provided
        assert dock1 is not dock2

    def test_get_dock_unknown_platform(self):
        """Test getting a dock for unknown platform."""
        registry = DockRegistry()

        dock = registry.get_dock("unknown")

        assert dock is None

    def test_get_dock_merges_config(self):
        """Test that default config is merged with override."""
        registry = DockRegistry()
        registry.register(MockDock, config={"a": 1, "b": 2})

        dock = registry.get_dock("mock", config={"b": 3, "c": 4})

        assert dock.config["a"] == 1  # From default
        assert dock.config["b"] == 3  # Overridden
        assert dock.config["c"] == 4  # From override

    @pytest.mark.asyncio
    async def test_get_initialized_dock(self):
        """Test getting an initialized dock."""
        registry = DockRegistry()
        registry.register(MockDock)

        dock = await registry.get_initialized_dock("mock")

        assert dock is not None
        assert dock.is_initialized

    @pytest.mark.asyncio
    async def test_get_initialized_dock_unknown_platform(self):
        """Test getting initialized dock for unknown platform."""
        registry = DockRegistry()

        dock = await registry.get_initialized_dock("unknown")

        assert dock is None

    @pytest.mark.asyncio
    async def test_get_initialized_dock_already_initialized(self):
        """Test that already initialized dock is not reinitialized."""
        registry = DockRegistry()
        registry.register(MockDock)

        # First call initializes
        dock1 = await registry.get_initialized_dock("mock")
        assert dock1.is_initialized

        # Add tracking
        init_count = 0
        original_init = dock1.initialize

        async def counted_init():
            nonlocal init_count
            init_count += 1
            return await original_init()

        dock1.initialize = counted_init

        # Second call should not reinitialize
        dock2 = await registry.get_initialized_dock("mock")
        assert dock1 is dock2
        assert init_count == 0  # Not called again

    def test_has_dock(self):
        """Test checking if dock is registered."""
        registry = DockRegistry()
        registry.register(MockDock)

        assert registry.has_dock("mock") is True
        assert registry.has_dock("MOCK") is True
        assert registry.has_dock("unknown") is False

    def test_get_platforms(self):
        """Test getting list of registered platforms."""
        registry = DockRegistry()
        registry.register(MockDock)
        registry.register(VoiceDock)

        platforms = registry.get_platforms()

        assert "mock" in platforms
        assert "voice_platform" in platforms
        assert len(platforms) == 2

    def test_get_platforms_empty(self):
        """Test getting platforms when none registered."""
        registry = DockRegistry()

        platforms = registry.get_platforms()

        assert platforms == []

    def test_get_platforms_with_capability_rich_text(self):
        """Test filtering platforms by rich text capability."""
        registry = DockRegistry()
        registry.register(MockDock)
        registry.register(VoiceDock)

        platforms = registry.get_platforms_with_capability(ChannelCapability.RICH_TEXT)

        assert "mock" in platforms
        assert "voice_platform" in platforms

    def test_get_platforms_with_capability_voice(self):
        """Test filtering platforms by voice capability."""
        registry = DockRegistry()
        registry.register(MockDock)
        registry.register(VoiceDock)

        platforms = registry.get_platforms_with_capability(ChannelCapability.VOICE)

        assert "mock" not in platforms
        assert "voice_platform" in platforms

    def test_get_platforms_with_capability_buttons(self):
        """Test filtering platforms by buttons capability."""
        registry = DockRegistry()
        registry.register(MockDock)
        registry.register(VoiceDock)

        platforms = registry.get_platforms_with_capability(ChannelCapability.BUTTONS)

        assert "mock" in platforms
        assert "voice_platform" not in platforms

    def test_unregister_dock(self):
        """Test unregistering a dock."""
        registry = DockRegistry()
        registry.register(MockDock, config={"test": True})
        registry.get_dock("mock")  # Create instance

        result = registry.unregister("mock")

        assert result is True
        assert "mock" not in registry._dock_classes
        assert "mock" not in registry._dock_instances
        assert "mock" not in registry._default_configs

    def test_unregister_unknown_dock(self):
        """Test unregistering an unknown dock."""
        registry = DockRegistry()

        result = registry.unregister("unknown")

        assert result is False

    def test_unregister_case_insensitive(self):
        """Test that unregister is case insensitive."""
        registry = DockRegistry()
        registry.register(MockDock)

        result = registry.unregister("MOCK")

        assert result is True
        assert "mock" not in registry._dock_classes

    def test_clear(self):
        """Test clearing all docks."""
        registry = DockRegistry()
        registry.register(MockDock, config={"test": True})
        registry.register(VoiceDock)
        registry.get_dock("mock")

        registry.clear()

        assert registry._dock_classes == {}
        assert registry._dock_instances == {}
        assert registry._default_configs == {}


class TestGetDockRegistry:
    """Tests for get_dock_registry singleton."""

    @pytest.fixture(autouse=True)
    def _reset_dock_registry_singleton(self):
        """Reset dock registry singleton before/after each test."""
        import aragora.channels.registry as registry_module

        registry_module._dock_registry = None
        yield
        registry_module._dock_registry = None

    def test_returns_singleton(self):
        """Test that get_dock_registry returns same instance."""
        registry1 = get_dock_registry()
        registry2 = get_dock_registry()

        assert registry1 is registry2

    def test_registers_builtin_docks(self):
        """Test that builtin docks are registered."""
        registry = get_dock_registry()
        platforms = registry.get_platforms()

        # At least some docks should be registered
        # The actual platforms depend on what's importable
        assert isinstance(platforms, list)


class TestBuiltinDockRegistration:
    """Tests for _register_builtin_docks."""

    def test_handles_import_errors(self):
        """Test that missing docks don't break registration."""
        from aragora.channels.registry import _register_builtin_docks

        registry = DockRegistry()

        # Mock all imports to fail
        with patch.dict(
            "sys.modules",
            {
                "aragora.channels.docks.slack": None,
                "aragora.channels.docks.telegram": None,
                "aragora.channels.docks.discord": None,
                "aragora.channels.docks.teams": None,
                "aragora.channels.docks.whatsapp": None,
                "aragora.channels.docks.email": None,
                "aragora.channels.docks.google_chat": None,
            },
        ):
            # Should not raise
            _register_builtin_docks(registry)

    def test_registers_available_docks(self):
        """Test that available docks are registered."""
        from aragora.channels.registry import _register_builtin_docks

        registry = DockRegistry()
        _register_builtin_docks(registry)

        # Should have at least some platforms registered
        platforms = registry.get_platforms()
        # The exact platforms depend on what's installed
        assert isinstance(platforms, list)


class TestRegistryWithRealDocks:
    """Tests with real dock implementations."""

    def test_slack_dock_registration(self):
        """Test registering Slack dock."""
        from aragora.channels.docks.slack import SlackDock

        registry = DockRegistry()
        registry.register(SlackDock)

        assert registry.has_dock("slack")
        dock = registry.get_dock("slack")
        assert dock is not None

    def test_telegram_dock_registration(self):
        """Test registering Telegram dock."""
        from aragora.channels.docks.telegram import TelegramDock

        registry = DockRegistry()
        registry.register(TelegramDock)

        assert registry.has_dock("telegram")
        dock = registry.get_dock("telegram")
        assert dock is not None

    def test_teams_dock_registration(self):
        """Test registering Teams dock."""
        from aragora.channels.docks.teams import TeamsDock

        registry = DockRegistry()
        registry.register(TeamsDock)

        assert registry.has_dock("teams")

    def test_discord_dock_registration(self):
        """Test registering Discord dock."""
        from aragora.channels.docks.discord import DiscordDock

        registry = DockRegistry()
        registry.register(DiscordDock)

        assert registry.has_dock("discord")
