"""
Device Connector Registry.

Manages device connector instances and provides factory methods
for creating and accessing connectors by platform name.

Usage:
    from aragora.connectors.devices.registry import DeviceConnectorRegistry

    registry = DeviceConnectorRegistry()

    # Get or create a connector
    fcm = registry.get("fcm")

    # Register a custom connector
    registry.register("custom", my_connector)

    # List all connectors
    for platform, connector in registry.all().items():
        print(f"{platform}: {connector.platform_display_name}")
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Type

from .base import DeviceConnector, DeviceConnectorConfig
from .models import DeviceType

logger = logging.getLogger(__name__)


class DeviceConnectorRegistry:
    """
    Registry for device connector instances.

    Manages connector lifecycle and provides lookup by platform name.
    Uses lazy initialization for connectors.
    """

    # Known connector types
    CONNECTOR_CLASSES: Dict[str, Type[DeviceConnector]] = {}

    def __init__(self):
        self._connectors: Dict[str, DeviceConnector] = {}
        self._lock = threading.Lock()
        self._initialized = False

    def _load_connector_classes(self) -> None:
        """Load connector classes on first access."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            try:
                from .push import FCMConnector, APNsConnector, WebPushConnector

                self.CONNECTOR_CLASSES["fcm"] = FCMConnector
                self.CONNECTOR_CLASSES["apns"] = APNsConnector
                self.CONNECTOR_CLASSES["web_push"] = WebPushConnector
            except ImportError as e:
                logger.debug(f"Could not load push connectors: {e}")

            try:
                from .alexa import AlexaConnector

                self.CONNECTOR_CLASSES["alexa"] = AlexaConnector
            except ImportError as e:
                logger.debug(f"Could not load Alexa connector: {e}")

            try:
                from .google_home import GoogleHomeConnector

                self.CONNECTOR_CLASSES["google_home"] = GoogleHomeConnector
            except ImportError as e:
                logger.debug(f"Could not load Google Home connector: {e}")

            self._initialized = True

    def get(
        self,
        platform: str,
        config: Optional[DeviceConnectorConfig] = None,
        auto_initialize: bool = True,
    ) -> DeviceConnector:
        """
        Get or create a connector for the specified platform.

        Args:
            platform: Platform name (e.g., "fcm", "apns", "web_push")
            config: Optional configuration for new connector
            auto_initialize: Whether to initialize the connector automatically

        Returns:
            DeviceConnector instance

        Raises:
            KeyError: If platform is not supported
        """
        self._load_connector_classes()

        with self._lock:
            if platform in self._connectors:
                return self._connectors[platform]

            if platform not in self.CONNECTOR_CLASSES:
                raise KeyError(f"Unknown device connector platform: {platform}")

            connector_class = self.CONNECTOR_CLASSES[platform]
            connector = connector_class(config)

            self._connectors[platform] = connector

        # Initialize outside of lock
        if auto_initialize:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule initialization
                    asyncio.create_task(connector.initialize())
                else:
                    loop.run_until_complete(connector.initialize())
            except RuntimeError:
                # No event loop, create one
                asyncio.run(connector.initialize())

        return connector

    def register(
        self,
        platform: str,
        connector: DeviceConnector,
    ) -> None:
        """
        Register a connector instance.

        Args:
            platform: Platform name
            connector: Connector instance to register
        """
        with self._lock:
            self._connectors[platform] = connector
            logger.info(f"Registered device connector: {platform}")

    def unregister(self, platform: str) -> Optional[DeviceConnector]:
        """
        Unregister a connector.

        Args:
            platform: Platform name to unregister

        Returns:
            The unregistered connector, or None if not found
        """
        with self._lock:
            return self._connectors.pop(platform, None)

    def all(self) -> Dict[str, DeviceConnector]:
        """Get all registered connectors."""
        with self._lock:
            return dict(self._connectors)

    def list_platforms(self) -> List[str]:
        """List all registered platform names."""
        with self._lock:
            return list(self._connectors.keys())

    def list_available_platforms(self) -> List[str]:
        """List all available (loadable) platform names."""
        self._load_connector_classes()
        return list(self.CONNECTOR_CLASSES.keys())

    def get_for_device_type(self, device_type: DeviceType) -> List[DeviceConnector]:
        """
        Get all connectors that support a device type.

        Args:
            device_type: Device type to find connectors for

        Returns:
            List of connectors supporting the device type
        """
        result = []
        for connector in self._connectors.values():
            if device_type in connector.supported_device_types:
                result.append(connector)
        return result

    def get_configured_platforms(self) -> List[str]:
        """
        Get list of platforms that have required environment variables configured.

        Returns:
            List of platform names with credentials configured
        """
        self._load_connector_classes()
        configured = []

        # Check FCM
        if os.environ.get("FCM_PROJECT_ID") and (
            os.environ.get("FCM_PRIVATE_KEY") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        ):
            configured.append("fcm")

        # Check APNs
        if all(
            os.environ.get(key)
            for key in ["APNS_KEY_ID", "APNS_TEAM_ID", "APNS_BUNDLE_ID", "APNS_PRIVATE_KEY"]
        ):
            configured.append("apns")

        # Check Web Push
        if all(
            os.environ.get(key)
            for key in ["VAPID_PUBLIC_KEY", "VAPID_PRIVATE_KEY", "VAPID_SUBJECT"]
        ):
            configured.append("web_push")

        # Check Alexa
        if all(
            os.environ.get(key)
            for key in ["ALEXA_CLIENT_ID", "ALEXA_CLIENT_SECRET", "ALEXA_SKILL_ID"]
        ):
            configured.append("alexa")

        # Check Google Home
        if os.environ.get("GOOGLE_HOME_PROJECT_ID") and (
            os.environ.get("GOOGLE_HOME_CREDENTIALS")
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        ):
            configured.append("google_home")

        return configured

    async def initialize_all(self) -> Dict[str, bool]:
        """
        Initialize all known connectors.

        Returns:
            Dict mapping platform name to initialization success
        """
        self._load_connector_classes()
        results = {}

        for platform, connector_class in self.CONNECTOR_CLASSES.items():
            try:
                connector = connector_class()
                success = await connector.initialize()
                results[platform] = success

                if success:
                    with self._lock:
                        self._connectors[platform] = connector
                    logger.info(f"Initialized device connector: {platform}")
                else:
                    logger.debug(f"Device connector not configured: {platform}")

            except Exception as e:
                logger.warning(f"Failed to initialize {platform}: {e}")
                results[platform] = False

        return results

    async def shutdown_all(self) -> None:
        """Shutdown all connectors."""
        with self._lock:
            connectors = list(self._connectors.values())

        for connector in connectors:
            try:
                await connector.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down {connector.platform_name}: {e}")

        with self._lock:
            self._connectors.clear()

    async def get_health(self) -> Dict[str, Any]:
        """
        Get health status for all connectors.

        Returns:
            Dict with health information for each connector
        """
        health: Dict[str, Any] = {
            "status": "healthy",
            "connectors": {},
            "configured_platforms": self.get_configured_platforms(),
            "available_platforms": self.list_available_platforms(),
        }

        unhealthy_count = 0

        for platform, connector in self._connectors.items():
            try:
                connector_health = await connector.get_health()
                health["connectors"][platform] = connector_health

                if connector_health.get("status") == "unhealthy":
                    unhealthy_count += 1
            except Exception as e:
                health["connectors"][platform] = {
                    "status": "error",
                    "error": str(e),
                }
                unhealthy_count += 1

        if unhealthy_count > 0:
            health["status"] = (
                "degraded" if unhealthy_count < len(self._connectors) else "unhealthy"
            )

        return health


# Global registry instance
_registry: Optional[DeviceConnectorRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> DeviceConnectorRegistry:
    """Get the global device connector registry."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = DeviceConnectorRegistry()
    return _registry


def get_connector(platform: str) -> DeviceConnector:
    """Get a device connector by platform name."""
    return get_registry().get(platform)


def register_connector(platform: str, connector: DeviceConnector) -> None:
    """Register a device connector."""
    get_registry().register(platform, connector)


def list_available_platforms() -> List[str]:
    """List all available device connector platforms."""
    return get_registry().list_available_platforms()


def get_configured_platforms() -> List[str]:
    """Get list of configured device connector platforms."""
    return get_registry().get_configured_platforms()
