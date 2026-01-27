"""
Device Connector Models.

Data models for push notification devices across platforms:
- iOS (APNs)
- Android (FCM)
- Web Push (VAPID)

These models define the structure for device registration, notifications,
and delivery results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class DeviceType(Enum):
    """Supported device types."""

    IOS = "ios"
    ANDROID = "android"
    WEB = "web"
    ALEXA = "alexa"
    GOOGLE_HOME = "google_home"


class NotificationPriority(Enum):
    """Notification priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class DeliveryStatus(Enum):
    """Status of notification delivery."""

    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    EXPIRED = "expired"
    INVALID_TOKEN = "invalid_token"


@dataclass
class DeviceToken:
    """
    Device registration token.

    Represents a registered device that can receive push notifications.

    Attributes:
        device_id: Unique server-generated device identifier
        user_id: Associated user identifier
        device_type: Platform type (ios, android, web)
        push_token: Platform-specific push token
        device_name: Optional human-readable device name
        app_version: Application version for compatibility
        created_at: Registration timestamp
        last_active: Last activity timestamp
        metadata: Additional device metadata
    """

    device_id: str
    user_id: str
    device_type: DeviceType
    push_token: str
    device_name: Optional[str] = None
    app_version: Optional[str] = None
    created_at: Optional[datetime] = None
    last_active: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "device_id": self.device_id,
            "user_id": self.user_id,
            "device_type": self.device_type.value,
            "push_token": self.push_token,
            "device_name": self.device_name,
            "app_version": self.app_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeviceToken:
        """Deserialize from dictionary."""
        return cls(
            device_id=data["device_id"],
            user_id=data["user_id"],
            device_type=DeviceType(data["device_type"]),
            push_token=data["push_token"],
            device_name=data.get("device_name"),
            app_version=data.get("app_version"),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None,
            last_active=datetime.fromisoformat(data["last_active"])
            if data.get("last_active")
            else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class DeviceMessage:
    """
    Push notification message.

    Defines the content and behavior of a push notification.

    Attributes:
        title: Notification title
        body: Notification body text
        data: Custom data payload
        image_url: Optional image URL for rich notifications
        action_url: URL to open when notification is tapped
        badge: Badge count (iOS/Android)
        sound: Sound to play (platform-specific)
        priority: Notification priority
        ttl_seconds: Time-to-live in seconds
        collapse_key: Key for notification collapsing
        topic: Topic/category for the notification
        channel_id: Android notification channel ID
        thread_id: iOS thread identifier for grouping
        mutable_content: iOS mutable content flag for extensions
    """

    title: str
    body: str
    data: Dict[str, Any] = field(default_factory=dict)
    image_url: Optional[str] = None
    action_url: Optional[str] = None
    badge: Optional[int] = None
    sound: Optional[str] = "default"
    priority: NotificationPriority = NotificationPriority.NORMAL
    ttl_seconds: int = 3600  # 1 hour default
    collapse_key: Optional[str] = None
    topic: Optional[str] = None
    channel_id: Optional[str] = None
    thread_id: Optional[str] = None
    mutable_content: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "title": self.title,
            "body": self.body,
            "data": self.data,
            "image_url": self.image_url,
            "action_url": self.action_url,
            "badge": self.badge,
            "sound": self.sound,
            "priority": self.priority.value,
            "ttl_seconds": self.ttl_seconds,
            "collapse_key": self.collapse_key,
            "topic": self.topic,
            "channel_id": self.channel_id,
            "thread_id": self.thread_id,
            "mutable_content": self.mutable_content,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeviceMessage:
        """Deserialize from dictionary."""
        return cls(
            title=data["title"],
            body=data["body"],
            data=data.get("data", {}),
            image_url=data.get("image_url"),
            action_url=data.get("action_url"),
            badge=data.get("badge"),
            sound=data.get("sound", "default"),
            priority=NotificationPriority(data.get("priority", "normal")),
            ttl_seconds=data.get("ttl_seconds", 3600),
            collapse_key=data.get("collapse_key"),
            topic=data.get("topic"),
            channel_id=data.get("channel_id"),
            thread_id=data.get("thread_id"),
            mutable_content=data.get("mutable_content", False),
        )


@dataclass
class SendResult:
    """
    Result of sending a push notification.

    Attributes:
        success: Whether the notification was sent successfully
        device_id: Device ID that was targeted
        message_id: Platform-specific message ID (if available)
        status: Delivery status
        error: Error message if failed
        error_code: Platform-specific error code
        should_unregister: Whether the device token should be removed
        timestamp: When the result was generated
    """

    success: bool
    device_id: str
    message_id: Optional[str] = None
    status: DeliveryStatus = DeliveryStatus.PENDING
    error: Optional[str] = None
    error_code: Optional[str] = None
    should_unregister: bool = False
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "device_id": self.device_id,
            "message_id": self.message_id,
            "status": self.status.value,
            "error": self.error,
            "error_code": self.error_code,
            "should_unregister": self.should_unregister,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class BatchSendResult:
    """
    Result of sending notifications to multiple devices.

    Attributes:
        total_sent: Total notifications attempted
        success_count: Number of successful deliveries
        failure_count: Number of failed deliveries
        results: Individual results per device
        tokens_to_remove: Device IDs with invalid tokens
    """

    total_sent: int
    success_count: int
    failure_count: int
    results: List[SendResult] = field(default_factory=list)
    tokens_to_remove: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_sent == 0:
            return 0.0
        return (self.success_count / self.total_sent) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_sent": self.total_sent,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(self.success_rate, 1),
            "results": [r.to_dict() for r in self.results],
            "tokens_to_remove": self.tokens_to_remove,
        }


@dataclass
class DeviceRegistration:
    """
    Device registration request.

    Used when registering a new device for push notifications.

    Attributes:
        user_id: User identifier
        device_type: Platform type
        push_token: Platform-specific push token
        device_name: Optional device name
        app_version: Application version
        os_version: Operating system version
        device_model: Device model (e.g., "iPhone 15", "Pixel 8")
        timezone: Device timezone
        locale: Device locale
        app_bundle_id: Application bundle identifier
    """

    user_id: str
    device_type: DeviceType
    push_token: str
    device_name: Optional[str] = None
    app_version: Optional[str] = None
    os_version: Optional[str] = None
    device_model: Optional[str] = None
    timezone: Optional[str] = None
    locale: Optional[str] = None
    app_bundle_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "user_id": self.user_id,
            "device_type": self.device_type.value,
            "push_token": self.push_token,
            "device_name": self.device_name,
            "app_version": self.app_version,
            "os_version": self.os_version,
            "device_model": self.device_model,
            "timezone": self.timezone,
            "locale": self.locale,
            "app_bundle_id": self.app_bundle_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeviceRegistration:
        """Deserialize from dictionary."""
        return cls(
            user_id=data["user_id"],
            device_type=DeviceType(data["device_type"]),
            push_token=data["push_token"],
            device_name=data.get("device_name"),
            app_version=data.get("app_version"),
            os_version=data.get("os_version"),
            device_model=data.get("device_model"),
            timezone=data.get("timezone"),
            locale=data.get("locale"),
            app_bundle_id=data.get("app_bundle_id"),
        )


@dataclass
class VoiceDeviceRequest:
    """
    Request from a voice device (Alexa, Google Home).

    Attributes:
        request_id: Unique request identifier
        device_type: Voice device platform
        user_id: User identifier (from account linking)
        intent: Voice command intent
        slots: Intent slot values
        raw_input: Raw voice input text
        session_id: Voice session identifier
        is_new_session: Whether this starts a new session
        timestamp: Request timestamp
        locale: Device locale
        metadata: Additional platform-specific metadata
    """

    request_id: str
    device_type: DeviceType
    user_id: str
    intent: str
    slots: Dict[str, Any] = field(default_factory=dict)
    raw_input: Optional[str] = None
    session_id: Optional[str] = None
    is_new_session: bool = False
    timestamp: Optional[datetime] = None
    locale: str = "en-US"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "request_id": self.request_id,
            "device_type": self.device_type.value,
            "user_id": self.user_id,
            "intent": self.intent,
            "slots": self.slots,
            "raw_input": self.raw_input,
            "session_id": self.session_id,
            "is_new_session": self.is_new_session,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "locale": self.locale,
            "metadata": self.metadata,
        }


@dataclass
class VoiceDeviceResponse:
    """
    Response to send to a voice device.

    Attributes:
        text: Response text to speak
        should_end_session: Whether to end the voice session
        reprompt: Text to speak if user doesn't respond
        card_title: Title for companion app card
        card_content: Content for companion app card
        card_image_url: Image URL for card
        directives: Platform-specific directives
    """

    text: str
    should_end_session: bool = True
    reprompt: Optional[str] = None
    card_title: Optional[str] = None
    card_content: Optional[str] = None
    card_image_url: Optional[str] = None
    directives: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "text": self.text,
            "should_end_session": self.should_end_session,
            "reprompt": self.reprompt,
            "card_title": self.card_title,
            "card_content": self.card_content,
            "card_image_url": self.card_image_url,
            "directives": self.directives,
        }
