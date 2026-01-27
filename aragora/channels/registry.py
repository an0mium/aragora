"""
Dock Registry - Discovery and management of channel docks.

Provides centralized registration and lookup of platform-specific docks.

Example:
    from aragora.channels.registry import DockRegistry

    registry = DockRegistry()
    registry.register(SlackDock)
    registry.register(TelegramDock)

    dock = registry.get_dock("slack")
    await dock.send_message(channel_id, message)
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type

from aragora.channels.dock import ChannelDock, ChannelCapability

logger = logging.getLogger(__name__)

__all__ = [
    "DockRegistry",
    "get_dock_registry",
]


class DockRegistry:
    """
    Registry for channel docks.

    Manages registration and lookup of platform-specific docks,
    with support for lazy initialization and capability queries.
    """

    def __init__(self):
        """Initialize the registry."""
        self._dock_classes: dict[str, Type[ChannelDock]] = {}
        self._dock_instances: dict[str, ChannelDock] = {}
        self._default_configs: dict[str, dict[str, Any]] = {}

    def register(
        self,
        dock_class: Type[ChannelDock],
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Register a dock class.

        Args:
            dock_class: The ChannelDock subclass to register
            config: Optional default configuration for this dock
        """
        platform = dock_class.PLATFORM.lower()
        self._dock_classes[platform] = dock_class
        if config:
            self._default_configs[platform] = config
        logger.debug(f"Registered dock for platform: {platform}")

    def get_dock(
        self,
        platform: str,
        config: Optional[dict[str, Any]] = None,
        auto_initialize: bool = False,
    ) -> Optional[ChannelDock]:
        """
        Get a dock instance for a platform.

        Creates a new instance if not already cached, or returns
        the cached instance.

        Args:
            platform: Platform identifier (e.g., "slack", "telegram")
            config: Optional configuration override
            auto_initialize: If True, initialize the dock before returning

        Returns:
            ChannelDock instance or None if platform not registered
        """
        platform = platform.lower()

        # Return cached instance if no config override
        if platform in self._dock_instances and config is None:
            return self._dock_instances[platform]

        # Get dock class
        dock_class = self._dock_classes.get(platform)
        if dock_class is None:
            logger.warning(f"No dock registered for platform: {platform}")
            return None

        # Create instance with merged config
        merged_config = {**self._default_configs.get(platform, {})}
        if config:
            merged_config.update(config)

        dock = dock_class(merged_config)

        # Cache if no config override
        if config is None:
            self._dock_instances[platform] = dock

        return dock

    async def get_initialized_dock(
        self,
        platform: str,
        config: Optional[dict[str, Any]] = None,
    ) -> Optional[ChannelDock]:
        """
        Get an initialized dock instance.

        Args:
            platform: Platform identifier
            config: Optional configuration override

        Returns:
            Initialized ChannelDock instance or None
        """
        dock = self.get_dock(platform, config)
        if dock and not dock.is_initialized:
            await dock.initialize()
        return dock

    def has_dock(self, platform: str) -> bool:
        """
        Check if a dock is registered for a platform.

        Args:
            platform: Platform identifier

        Returns:
            True if a dock is registered
        """
        return platform.lower() in self._dock_classes

    def get_platforms(self) -> list[str]:
        """
        Get list of registered platforms.

        Returns:
            List of platform identifiers
        """
        return list(self._dock_classes.keys())

    def get_platforms_with_capability(
        self,
        capability: ChannelCapability,
    ) -> list[str]:
        """
        Get platforms that support a specific capability.

        Args:
            capability: The capability to check

        Returns:
            List of platform identifiers that support the capability
        """
        platforms = []
        for platform, dock_class in self._dock_classes.items():
            if bool(dock_class.CAPABILITIES & capability):
                platforms.append(platform)
        return platforms

    def unregister(self, platform: str) -> bool:
        """
        Unregister a dock.

        Args:
            platform: Platform identifier

        Returns:
            True if the dock was unregistered
        """
        platform = platform.lower()
        removed = platform in self._dock_classes
        self._dock_classes.pop(platform, None)
        self._dock_instances.pop(platform, None)
        self._default_configs.pop(platform, None)
        return removed

    def clear(self) -> None:
        """Clear all registered docks."""
        self._dock_classes.clear()
        self._dock_instances.clear()
        self._default_configs.clear()


# Global registry singleton
_dock_registry: Optional[DockRegistry] = None


def get_dock_registry() -> DockRegistry:
    """
    Get the global dock registry singleton.

    The registry is automatically populated with available docks
    on first access.

    Returns:
        DockRegistry instance
    """
    global _dock_registry

    if _dock_registry is None:
        _dock_registry = DockRegistry()
        _register_builtin_docks(_dock_registry)

    return _dock_registry


def _register_builtin_docks(registry: DockRegistry) -> None:
    """
    Register built-in dock implementations.

    Called automatically when the registry is first accessed.
    """
    # Import and register available docks
    try:
        from aragora.channels.docks.slack import SlackDock

        registry.register(SlackDock)
    except ImportError:
        pass

    try:
        from aragora.channels.docks.telegram import TelegramDock

        registry.register(TelegramDock)
    except ImportError:
        pass

    try:
        from aragora.channels.docks.discord import DiscordDock

        registry.register(DiscordDock)
    except ImportError:
        pass

    try:
        from aragora.channels.docks.teams import TeamsDock

        registry.register(TeamsDock)
    except ImportError:
        pass

    try:
        from aragora.channels.docks.whatsapp import WhatsAppDock

        registry.register(WhatsAppDock)
    except ImportError:
        pass

    try:
        from aragora.channels.docks.email import EmailDock

        registry.register(EmailDock)
    except ImportError:
        pass

    try:
        from aragora.channels.docks.google_chat import GoogleChatDock

        registry.register(GoogleChatDock)
    except ImportError:
        pass
