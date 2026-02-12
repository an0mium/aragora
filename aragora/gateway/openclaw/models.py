"""
OpenClaw Gateway Data Models.

Contains enums and dataclasses for the OpenClaw gateway integration:
- Channel and action type enums
- Message, action, and session dataclasses
- Channel mapping and result types
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4
import logging
logger = logging.getLogger(__name__)



# ============================================================================
# Enums
# ============================================================================


class OpenClawChannel(str, Enum):
    """Supported OpenClaw communication channels."""

    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    SLACK = "slack"
    DISCORD = "discord"
    SMS = "sms"
    EMAIL = "email"
    WEB = "web"
    VOICE = "voice"
    TEAMS = "teams"
    MATRIX = "matrix"


class OpenClawActionType(str, Enum):
    """Supported OpenClaw action types."""

    # Browser control actions
    BROWSER_NAVIGATE = "browser_navigate"
    BROWSER_CLICK = "browser_click"
    BROWSER_TYPE = "browser_type"
    BROWSER_SCREENSHOT = "browser_screenshot"
    BROWSER_SCROLL = "browser_scroll"
    BROWSER_WAIT = "browser_wait"
    BROWSER_EXTRACT = "browser_extract"

    # Canvas/drawing actions
    CANVAS_CREATE = "canvas_create"
    CANVAS_DRAW = "canvas_draw"
    CANVAS_RENDER = "canvas_render"
    CANVAS_EXPORT = "canvas_export"

    # Cron/scheduling actions
    CRON_CREATE = "cron_create"
    CRON_UPDATE = "cron_update"
    CRON_DELETE = "cron_delete"
    CRON_LIST = "cron_list"
    CRON_TRIGGER = "cron_trigger"

    # File operations
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    FILE_LIST = "file_list"

    # API/HTTP actions
    HTTP_REQUEST = "http_request"
    API_CALL = "api_call"

    # Code execution
    CODE_RUN = "code_run"
    CODE_EVAL = "code_eval"

    # Database actions
    DB_QUERY = "db_query"
    DB_EXECUTE = "db_execute"

    # Notification actions
    NOTIFY_SEND = "notify_send"
    NOTIFY_BROADCAST = "notify_broadcast"

    # Custom action
    CUSTOM = "custom"


class SessionState(str, Enum):
    """OpenClaw session states."""

    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    EXPIRED = "expired"
    TERMINATED = "terminated"


class ActionStatus(str, Enum):
    """Action execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


# ============================================================================
# Core Dataclasses
# ============================================================================


@dataclass
class OpenClawMessage:
    """
    Message in OpenClaw format.

    Represents a message that can be sent to or received from OpenClaw,
    with channel-specific metadata and formatting.

    Attributes:
        message_id: Unique message identifier.
        type: Message type (text, image, audio, video, file, action, system).
        content: Message content (text or structured data).
        channel: Source/destination channel.
        metadata: Channel-specific metadata (e.g., reply_to, thread_id).
        sender_id: Sender identifier.
        timestamp: Message timestamp.
        attachments: List of attachment URLs or data.
        reply_to: ID of message being replied to.
        thread_id: Thread identifier for threaded conversations.
    """

    message_id: str
    type: str  # text, image, audio, video, file, action, system
    content: Any
    channel: OpenClawChannel | str
    metadata: dict[str, Any] = field(default_factory=dict)
    sender_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attachments: list[dict[str, Any]] = field(default_factory=list)
    reply_to: str | None = None
    thread_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "message_id": self.message_id,
            "type": self.type,
            "content": self.content,
            "channel": self.channel.value
            if isinstance(self.channel, OpenClawChannel)
            else self.channel,
            "metadata": self.metadata,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp.isoformat(),
            "attachments": self.attachments,
            "reply_to": self.reply_to,
            "thread_id": self.thread_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenClawMessage:
        """Create message from dictionary."""
        channel = data.get("channel", "web")
        try:
            channel = OpenClawChannel(channel)
        except ValueError:
            pass  # Keep as string for unknown channels

        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return cls(
            message_id=data.get("message_id", str(uuid4())),
            type=data.get("type", "text"),
            content=data.get("content"),
            channel=channel,
            metadata=data.get("metadata", {}),
            sender_id=data.get("sender_id"),
            timestamp=timestamp,
            attachments=data.get("attachments", []),
            reply_to=data.get("reply_to"),
            thread_id=data.get("thread_id"),
        )


@dataclass
class OpenClawAction:
    """
    Action to be executed by OpenClaw.

    Represents an operation that OpenClaw should perform, such as
    browser automation, file operations, or scheduled tasks.

    Attributes:
        action_type: Type of action to execute.
        parameters: Action-specific parameters.
        timeout: Maximum execution time in seconds.
        retry_count: Number of retry attempts on failure.
        callback_url: URL to call when action completes.
        metadata: Additional action metadata.
    """

    action_type: OpenClawActionType | str
    parameters: dict[str, Any] = field(default_factory=dict)
    timeout: int = 60  # seconds
    retry_count: int = 0
    callback_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert action to dictionary format."""
        return {
            "action_type": self.action_type.value
            if isinstance(self.action_type, OpenClawActionType)
            else self.action_type,
            "parameters": self.parameters,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "callback_url": self.callback_url,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenClawAction:
        """Create action from dictionary."""
        action_type = data.get("action_type", "custom")
        try:
            action_type = OpenClawActionType(action_type)
        except ValueError:
            pass  # Keep as string for custom actions

        return cls(
            action_type=action_type,
            parameters=data.get("parameters", {}),
            timeout=data.get("timeout", 60),
            retry_count=data.get("retry_count", 0),
            callback_url=data.get("callback_url"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class OpenClawSession:
    """
    OpenClaw session for tracking user interactions.

    Manages the lifecycle of a user's interaction with OpenClaw,
    including channel binding, tenant context, and state tracking.

    Attributes:
        session_id: Unique session identifier.
        user_id: Associated user identifier.
        channel: Communication channel for this session.
        state: Current session state.
        tenant_id: Tenant identifier for multi-tenancy.
        created_at: Session creation timestamp.
        last_activity: Last activity timestamp.
        expires_at: Session expiration timestamp.
        metadata: Session metadata and preferences.
        context: Conversation context and history.
    """

    session_id: str
    user_id: str
    channel: OpenClawChannel | str
    state: SessionState = SessionState.CREATED
    tenant_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    def is_active(self) -> bool:
        """Check if session is currently active."""
        if self.state != SessionState.ACTIVE:
            return False
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return True

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary format."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "channel": self.channel.value
            if isinstance(self.channel, OpenClawChannel)
            else self.channel,
            "state": self.state.value,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenClawSession:
        """Create session from dictionary."""
        channel = data.get("channel", "web")
        try:
            channel = OpenClawChannel(channel)
        except ValueError as e:
            logger.debug("from dict encountered an error: %s", e)

        state = data.get("state", "created")
        try:
            state = SessionState(state)
        except ValueError:
            state = SessionState.CREATED

        def parse_datetime(value: str | datetime | None) -> datetime | None:
            if value is None:
                return None
            if isinstance(value, datetime):
                return value
            return datetime.fromisoformat(value.replace("Z", "+00:00"))

        return cls(
            session_id=data.get("session_id", str(uuid4())),
            user_id=data["user_id"],
            channel=channel,
            state=state,
            tenant_id=data.get("tenant_id"),
            created_at=parse_datetime(data.get("created_at")) or datetime.now(timezone.utc),
            last_activity=parse_datetime(data.get("last_activity")) or datetime.now(timezone.utc),
            expires_at=parse_datetime(data.get("expires_at")),
            metadata=data.get("metadata", {}),
            context=data.get("context", {}),
        )


@dataclass
class ChannelMapping:
    """
    Mapping configuration between Aragora and OpenClaw channels.

    Defines how to translate messages and route responses between
    different channel representations.

    Attributes:
        aragora_channel: Aragora channel identifier.
        openclaw_channel: Corresponding OpenClaw channel.
        formatter: Optional custom formatter function name.
        response_routing: How to route responses back.
        metadata: Additional mapping configuration.
    """

    aragora_channel: str
    openclaw_channel: OpenClawChannel | str
    formatter: str | None = None
    response_routing: str = "direct"  # direct, async, webhook
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert mapping to dictionary format."""
        return {
            "aragora_channel": self.aragora_channel,
            "openclaw_channel": self.openclaw_channel.value
            if isinstance(self.openclaw_channel, OpenClawChannel)
            else self.openclaw_channel,
            "formatter": self.formatter,
            "response_routing": self.response_routing,
            "metadata": self.metadata,
        }


@dataclass
class ActionResult:
    """Result of an OpenClaw action execution."""

    action_id: str
    status: ActionStatus
    result: Any | None = None
    error: str | None = None
    execution_time_ms: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "action_id": self.action_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }


# ============================================================================
# Legacy Gateway Result Types (for backward compatibility)
# ============================================================================


@dataclass
class GatewayResult:
    """Result from gateway operation."""

    success: bool
    request_id: str
    response: Any | None = None  # AragoraResponse when imported from protocol
    error: str | None = None
    blocked_reason: str | None = None
    audit_event_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceRegistration:
    """Device registration request."""

    device_id: str
    device_name: str
    device_type: str  # desktop, mobile, server, iot
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceHandle:
    """Handle to a registered device."""

    device_id: str
    registration_id: str
    registered_at: datetime
    status: str = "active"


@dataclass
class PluginInstallRequest:
    """Plugin installation request."""

    plugin_id: str
    plugin_name: str
    version: str
    source: str  # marketplace, local, url
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    # Enums
    "OpenClawChannel",
    "OpenClawActionType",
    "SessionState",
    "ActionStatus",
    # Core dataclasses
    "OpenClawMessage",
    "OpenClawAction",
    "OpenClawSession",
    "ChannelMapping",
    "ActionResult",
    # Legacy types
    "GatewayResult",
    "DeviceRegistration",
    "DeviceHandle",
    "PluginInstallRequest",
]
