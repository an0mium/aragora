"""
Devices Namespace API.

Provides a namespaced interface for device registration and push notifications:
- Android (FCM), iOS (APNS), Web Push
- Alexa and Google Home voice assistants
- Health monitoring
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

DeviceType = Literal["android", "ios", "web", "alexa", "google_home"]
NotificationStatus = Literal["sent", "delivered", "failed", "pending"]


class Device(TypedDict, total=False):
    """Registered device."""

    device_id: str
    user_id: str
    device_type: str
    device_name: str | None
    app_version: str | None
    os_version: str | None
    last_active: str
    notification_count: int
    created_at: str


class NotificationMessage(TypedDict, total=False):
    """Push notification message."""

    title: str
    body: str
    data: dict[str, Any] | None
    image_url: str | None
    action_url: str | None
    badge: int | None
    sound: str | None


class NotificationResult(TypedDict, total=False):
    """Notification delivery result."""

    success: bool
    device_id: str
    message_id: str | None
    status: str
    error: str | None


class ConnectorHealth(TypedDict, total=False):
    """Device connector health status."""

    status: str
    fcm: dict[str, Any] | None
    apns: dict[str, Any] | None
    web_push: dict[str, Any] | None
    alexa: dict[str, Any] | None
    google_home: dict[str, Any] | None
    error: str | None


class DevicesAPI:
    """
    Synchronous Devices API.

    Provides comprehensive device management:
    - Device registration for multiple platforms
    - Push notification delivery
    - Voice assistant webhook handling
    - Health monitoring

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> # Register a device
        >>> device = client.devices.register(
        ...     device_type="android",
        ...     push_token="fcm_token_here",
        ...     device_name="My Phone"
        ... )
        >>> # Send notification
        >>> result = client.devices.notify(
        ...     device["device_id"],
        ...     title="New Decision",
        ...     body="The team has reached consensus"
        ... )
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Device Registration
    # =========================================================================

    def register(
        self,
        device_type: DeviceType,
        push_token: str,
        user_id: str | None = None,
        device_name: str | None = None,
        app_version: str | None = None,
        os_version: str | None = None,
        device_model: str | None = None,
        timezone: str | None = None,
        locale: str | None = None,
        app_bundle_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Register a device for push notifications.

        Supports Android (FCM), iOS (APNS), Web Push, Alexa, and Google Home.

        Args:
            device_type: Type of device (android, ios, web, alexa, google_home).
            push_token: Push notification token from the platform.
            user_id: Optional user ID to associate with device.
            device_name: Human-readable device name.
            app_version: Application version.
            os_version: Operating system version.
            device_model: Device model identifier.
            timezone: Device timezone.
            locale: Device locale.
            app_bundle_id: Application bundle identifier.

        Returns:
            Registration result with device_id.
        """
        data: dict[str, Any] = {
            "device_type": device_type,
            "push_token": push_token,
        }
        if user_id:
            data["user_id"] = user_id
        if device_name:
            data["device_name"] = device_name
        if app_version:
            data["app_version"] = app_version
        if os_version:
            data["os_version"] = os_version
        if device_model:
            data["device_model"] = device_model
        if timezone:
            data["timezone"] = timezone
        if locale:
            data["locale"] = locale
        if app_bundle_id:
            data["app_bundle_id"] = app_bundle_id
        return self._client.request("POST", "/api/v1/devices/register", json=data)

    def unregister(self, device_id: str) -> dict[str, Any]:
        """
        Unregister a device.

        Args:
            device_id: Device ID to unregister.

        Returns:
            Confirmation with deleted_at timestamp.
        """
        return self._client.request("DELETE", f"/api/v1/devices/{device_id}")

    def get(self, device_id: str) -> Device:
        """
        Get device information.

        Args:
            device_id: Device ID.

        Returns:
            Device details.
        """
        return self._client.request("GET", f"/api/v1/devices/{device_id}")

    def list_by_user(self, user_id: str) -> dict[str, Any]:
        """
        List all devices for a user.

        Args:
            user_id: User ID.

        Returns:
            Dict with device_count and devices list.
        """
        return self._client.request("GET", f"/api/v1/devices/user/{user_id}")

    # =========================================================================
    # Push Notifications
    # =========================================================================

    def notify(
        self,
        device_id: str,
        title: str,
        body: str,
        data: dict[str, Any] | None = None,
        image_url: str | None = None,
        action_url: str | None = None,
        badge: int | None = None,
        sound: str | None = None,
    ) -> NotificationResult:
        """
        Send notification to a specific device.

        Args:
            device_id: Target device ID.
            title: Notification title.
            body: Notification body text.
            data: Custom data payload.
            image_url: Image to display.
            action_url: URL to open on tap.
            badge: Badge count (iOS).
            sound: Sound to play.

        Returns:
            Delivery result with status.
        """
        message: dict[str, Any] = {"title": title, "body": body}
        if data:
            message["data"] = data
        if image_url:
            message["image_url"] = image_url
        if action_url:
            message["action_url"] = action_url
        if badge is not None:
            message["badge"] = badge
        if sound:
            message["sound"] = sound
        return self._client.request("POST", f"/api/v1/devices/{device_id}/notify", json=message)

    def notify_user(
        self,
        user_id: str,
        title: str,
        body: str,
        data: dict[str, Any] | None = None,
        image_url: str | None = None,
        action_url: str | None = None,
        badge: int | None = None,
        sound: str | None = None,
    ) -> dict[str, Any]:
        """
        Send notification to all devices for a user.

        Args:
            user_id: Target user ID.
            title: Notification title.
            body: Notification body text.
            data: Custom data payload.
            image_url: Image to display.
            action_url: URL to open on tap.
            badge: Badge count (iOS).
            sound: Sound to play.

        Returns:
            Delivery results for all devices.
        """
        message: dict[str, Any] = {"title": title, "body": body}
        if data:
            message["data"] = data
        if image_url:
            message["image_url"] = image_url
        if action_url:
            message["action_url"] = action_url
        if badge is not None:
            message["badge"] = badge
        if sound:
            message["sound"] = sound
        return self._client.request("POST", f"/api/v1/devices/user/{user_id}/notify", json=message)

    # =========================================================================
    # Health Monitoring
    # =========================================================================

    def get_health(self) -> ConnectorHealth:
        """
        Get device connector health status.

        Shows availability of FCM, APNS, Web Push, and voice connectors.

        Returns:
            Connector health status.
        """
        return self._client.request("GET", "/api/v1/devices/health")

    # =========================================================================
    # Voice Assistant Webhooks
    # =========================================================================

    def handle_alexa_webhook(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Handle Alexa skill webhook request.

        Processes voice commands and returns Alexa-formatted responses.
        Note: This is typically called by Alexa's servers, not directly.

        Args:
            request: Alexa skill request.

        Returns:
            Alexa skill response.
        """
        return self._client.request("POST", "/api/v1/devices/alexa/webhook", json=request)

    def handle_google_webhook(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Handle Google Actions webhook request.

        Processes voice commands and Smart Home intents.
        Note: This is typically called by Google's servers, not directly.

        Args:
            request: Google Actions request.

        Returns:
            Google Actions response.
        """
        return self._client.request("POST", "/api/v1/devices/google/webhook", json=request)


class AsyncDevicesAPI:
    """Asynchronous Devices API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def register(
        self,
        device_type: DeviceType,
        push_token: str,
        user_id: str | None = None,
        device_name: str | None = None,
        app_version: str | None = None,
        os_version: str | None = None,
        device_model: str | None = None,
        timezone: str | None = None,
        locale: str | None = None,
        app_bundle_id: str | None = None,
    ) -> dict[str, Any]:
        """Register a device for push notifications."""
        data: dict[str, Any] = {
            "device_type": device_type,
            "push_token": push_token,
        }
        if user_id:
            data["user_id"] = user_id
        if device_name:
            data["device_name"] = device_name
        if app_version:
            data["app_version"] = app_version
        if os_version:
            data["os_version"] = os_version
        if device_model:
            data["device_model"] = device_model
        if timezone:
            data["timezone"] = timezone
        if locale:
            data["locale"] = locale
        if app_bundle_id:
            data["app_bundle_id"] = app_bundle_id
        return await self._client.request("POST", "/api/v1/devices/register", json=data)

    async def unregister(self, device_id: str) -> dict[str, Any]:
        """Unregister a device."""
        return await self._client.request("DELETE", f"/api/v1/devices/{device_id}")

    async def get(self, device_id: str) -> Device:
        """Get device information."""
        return await self._client.request("GET", f"/api/v1/devices/{device_id}")

    async def list_by_user(self, user_id: str) -> dict[str, Any]:
        """List all devices for a user."""
        return await self._client.request("GET", f"/api/v1/devices/user/{user_id}")

    async def notify(
        self,
        device_id: str,
        title: str,
        body: str,
        data: dict[str, Any] | None = None,
        image_url: str | None = None,
        action_url: str | None = None,
        badge: int | None = None,
        sound: str | None = None,
    ) -> NotificationResult:
        """Send notification to a specific device."""
        message: dict[str, Any] = {"title": title, "body": body}
        if data:
            message["data"] = data
        if image_url:
            message["image_url"] = image_url
        if action_url:
            message["action_url"] = action_url
        if badge is not None:
            message["badge"] = badge
        if sound:
            message["sound"] = sound
        return await self._client.request(
            "POST", f"/api/v1/devices/{device_id}/notify", json=message
        )

    async def notify_user(
        self,
        user_id: str,
        title: str,
        body: str,
        data: dict[str, Any] | None = None,
        image_url: str | None = None,
        action_url: str | None = None,
        badge: int | None = None,
        sound: str | None = None,
    ) -> dict[str, Any]:
        """Send notification to all devices for a user."""
        message: dict[str, Any] = {"title": title, "body": body}
        if data:
            message["data"] = data
        if image_url:
            message["image_url"] = image_url
        if action_url:
            message["action_url"] = action_url
        if badge is not None:
            message["badge"] = badge
        if sound:
            message["sound"] = sound
        return await self._client.request(
            "POST", f"/api/v1/devices/user/{user_id}/notify", json=message
        )

    async def get_health(self) -> ConnectorHealth:
        """Get device connector health status."""
        return await self._client.request("GET", "/api/v1/devices/health")

    async def handle_alexa_webhook(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle Alexa skill webhook request."""
        return await self._client.request("POST", "/api/v1/devices/alexa/webhook", json=request)

    async def handle_google_webhook(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle Google Actions webhook request."""
        return await self._client.request("POST", "/api/v1/devices/google/webhook", json=request)
