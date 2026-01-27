"""
Device Connectors - Unified interface for push notification integrations.

This module provides a consistent API for sending push notifications
to various platforms including iOS (APNs), Android (FCM), Web Push,
Alexa, and Google Home.

Usage:
    from aragora.connectors.devices import get_connector, get_registry

    # Get a specific connector
    fcm = get_connector("fcm")
    result = await fcm.send_notification(device, message)

    # Get all configured connectors
    registry = get_registry()
    for platform, connector in registry.all().items():
        await connector.send_notification(device, message)

    # Send to all devices for a user
    results = await fcm.send_to_user("user-123", message)
"""

from .base import DeviceConnector, DeviceConnectorConfig
from .models import (
    BatchSendResult,
    DeliveryStatus,
    DeviceMessage,
    DeviceRegistration,
    DeviceToken,
    DeviceType,
    NotificationPriority,
    SendResult,
    VoiceDeviceRequest,
    VoiceDeviceResponse,
)

__all__ = [
    # Base class
    "DeviceConnector",
    "DeviceConnectorConfig",
    # Models
    "BatchSendResult",
    "DeliveryStatus",
    "DeviceMessage",
    "DeviceRegistration",
    "DeviceToken",
    "DeviceType",
    "NotificationPriority",
    "SendResult",
    "VoiceDeviceRequest",
    "VoiceDeviceResponse",
    # Registry (lazy-loaded)
    "get_connector",
    "get_registry",
    "register_connector",
]


# Lazy-loaded registry
_registry = None


def get_registry():
    """Get the device connector registry."""
    global _registry
    if _registry is None:
        from .registry import DeviceConnectorRegistry

        _registry = DeviceConnectorRegistry()
    return _registry


def get_connector(platform: str) -> DeviceConnector:
    """Get a specific device connector by platform name."""
    return get_registry().get(platform)


def register_connector(platform: str, connector: DeviceConnector) -> None:
    """Register a device connector."""
    get_registry().register(platform, connector)


def __getattr__(name: str):
    """Lazy-load connector classes."""
    if name == "FCMConnector":
        from .push import FCMConnector

        return FCMConnector
    elif name == "APNsConnector":
        from .push import APNsConnector

        return APNsConnector
    elif name == "WebPushConnector":
        from .push import WebPushConnector

        return WebPushConnector
    elif name == "AlexaConnector":
        from .alexa import AlexaConnector

        return AlexaConnector
    elif name == "GoogleHomeConnector":
        from .google_home import GoogleHomeConnector

        return GoogleHomeConnector

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
