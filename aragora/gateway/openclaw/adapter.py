"""
OpenClaw Gateway Adapter.

Provides comprehensive integration between Aragora and OpenClaw systems:
- Protocol translation (Aragora <-> OpenClaw message formats)
- Channel mapping (WhatsApp, Telegram, Slack, Discord, etc.)
- Action execution (browser control, canvas, cron, etc.)
- Session management with tenant/user context

Usage:
    from aragora.gateway.openclaw import (
        OpenClawAdapter,
        OpenClawMessage,
        OpenClawAction,
        OpenClawSession,
        ChannelMapping,
    )

    adapter = OpenClawAdapter(
        openclaw_endpoint="http://localhost:8081",
        rbac_checker=checker,
    )

    # Create session
    session = await adapter.create_session(
        user_id="user-123",
        channel="telegram",
        tenant_id="tenant-456",
    )

    # Execute action
    result = await adapter.execute_action(
        session_id=session.session_id,
        action=OpenClawAction(
            action_type="browser_navigate",
            parameters={"url": "https://example.com"},
        ),
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Protocol
from uuid import uuid4

from .audit import OpenClawAuditEvents, get_event_severity
from .capabilities import CapabilityCheckResult, CapabilityFilter
from .protocol import (
    AragoraRequest,
    AragoraResponse,
    AuthorizationContext,
    OpenClawProtocolTranslator,
    TenantContext,
)
from .sandbox import OpenClawSandbox, OpenClawTask, SandboxConfig, SandboxStatus

logger = logging.getLogger(__name__)


# ============================================================================
# Protocol definitions for dependency injection
# ============================================================================


class RBACCheckerProtocol(Protocol):
    """Protocol for RBAC permission checking."""

    def check_permission(
        self,
        actor_id: str,
        permission: str,
        resource_id: str | None = None,
    ) -> bool:
        """Check if actor has permission."""
        ...

    async def check_permission_async(
        self,
        actor_id: str,
        permission: str,
        resource_id: str | None = None,
    ) -> bool:
        """Async check if actor has permission."""
        ...


class AuditLoggerProtocol(Protocol):
    """Protocol for audit logging."""

    def log(
        self,
        event_type: str,
        actor_id: str,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        severity: str = "info",
    ) -> None:
        """Log an audit event."""
        ...

    async def log_async(
        self,
        event_type: str,
        actor_id: str,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        severity: str = "info",
    ) -> None:
        """Async log an audit event."""
        ...


class ApprovalGateProtocol(Protocol):
    """Protocol for approval gate checking."""

    async def check_approval(
        self,
        gate: str,
        actor_id: str,
        resource_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Check if actor has approval for gate. Returns (approved, reason)."""
        ...


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
    def from_dict(cls, data: dict[str, Any]) -> "OpenClawMessage":
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
    def from_dict(cls, data: dict[str, Any]) -> "OpenClawAction":
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
    def from_dict(cls, data: dict[str, Any]) -> "OpenClawSession":
        """Create session from dictionary."""
        channel = data.get("channel", "web")
        try:
            channel = OpenClawChannel(channel)
        except ValueError:
            pass

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
    response: AragoraResponse | None = None
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


# ============================================================================
# Channel Formatters
# ============================================================================


class ChannelFormatter:
    """Base class for channel-specific message formatting."""

    def format_outgoing(
        self,
        message: OpenClawMessage,
        session: OpenClawSession,
    ) -> dict[str, Any]:
        """Format message for sending to channel."""
        return message.to_dict()

    def parse_incoming(
        self,
        raw_message: dict[str, Any],
        session: OpenClawSession,
    ) -> OpenClawMessage:
        """Parse incoming channel message to OpenClaw format."""
        return OpenClawMessage.from_dict(raw_message)


class WhatsAppFormatter(ChannelFormatter):
    """Formatter for WhatsApp messages."""

    def format_outgoing(
        self,
        message: OpenClawMessage,
        session: OpenClawSession,
    ) -> dict[str, Any]:
        """Format message for WhatsApp."""
        result: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "to": session.metadata.get("phone_number"),
            "type": self._map_message_type(message.type),
        }

        if message.type == "text":
            result["text"] = {"body": message.content}
        elif message.type == "image":
            result["image"] = {"link": message.content}
        elif message.type == "audio":
            result["audio"] = {"link": message.content}
        elif message.type == "video":
            result["video"] = {"link": message.content}
        elif message.type == "file":
            result["document"] = {"link": message.content}

        return result

    def _map_message_type(self, msg_type: str) -> str:
        """Map OpenClaw message type to WhatsApp type."""
        type_map = {
            "text": "text",
            "image": "image",
            "audio": "audio",
            "video": "video",
            "file": "document",
        }
        return type_map.get(msg_type, "text")


class TelegramFormatter(ChannelFormatter):
    """Formatter for Telegram messages."""

    def format_outgoing(
        self,
        message: OpenClawMessage,
        session: OpenClawSession,
    ) -> dict[str, Any]:
        """Format message for Telegram."""
        chat_id = session.metadata.get("chat_id")
        result: dict[str, Any] = {"chat_id": chat_id}

        if message.type == "text":
            result["text"] = message.content
            result["parse_mode"] = "HTML"
        elif message.type == "image":
            result["photo"] = message.content
        elif message.type == "audio":
            result["audio"] = message.content
        elif message.type == "video":
            result["video"] = message.content
        elif message.type == "file":
            result["document"] = message.content

        if message.reply_to:
            result["reply_to_message_id"] = message.reply_to

        return result


class SlackFormatter(ChannelFormatter):
    """Formatter for Slack messages."""

    def format_outgoing(
        self,
        message: OpenClawMessage,
        session: OpenClawSession,
    ) -> dict[str, Any]:
        """Format message for Slack."""
        result: dict[str, Any] = {
            "channel": session.metadata.get("channel_id"),
        }

        if message.type == "text":
            result["text"] = message.content
            # Support Slack blocks for rich formatting
            if message.metadata.get("blocks"):
                result["blocks"] = message.metadata["blocks"]
        elif message.type in ("image", "file"):
            result["text"] = message.metadata.get("alt_text", "")
            result["attachments"] = [
                {
                    "fallback": message.metadata.get("alt_text", "attachment"),
                    "image_url" if message.type == "image" else "title_link": message.content,
                }
            ]

        if message.thread_id:
            result["thread_ts"] = message.thread_id

        return result


class DiscordFormatter(ChannelFormatter):
    """Formatter for Discord messages."""

    def format_outgoing(
        self,
        message: OpenClawMessage,
        session: OpenClawSession,
    ) -> dict[str, Any]:
        """Format message for Discord."""
        result: dict[str, Any] = {}

        if message.type == "text":
            result["content"] = message.content
            if message.metadata.get("embeds"):
                result["embeds"] = message.metadata["embeds"]
        elif message.type == "image":
            result["embeds"] = [{"image": {"url": message.content}}]
        elif message.type == "file":
            result["content"] = message.metadata.get("description", "")
            result["attachments"] = [{"url": message.content}]

        return result


# ============================================================================
# OpenClaw Adapter
# ============================================================================


class OpenClawAdapter:
    """
    Comprehensive adapter for Aragora-OpenClaw integration.

    Provides:
    - Protocol translation between Aragora and OpenClaw formats
    - Channel mapping with custom formatters
    - Action execution with timeout and retry handling
    - Session management with tenant/user context

    Usage:
        adapter = OpenClawAdapter(
            openclaw_endpoint="http://localhost:8081",
            rbac_checker=checker,
        )

        # Create session
        session = await adapter.create_session(
            user_id="user-123",
            channel="telegram",
            tenant_id="tenant-456",
        )

        # Send message
        message = OpenClawMessage(
            message_id=str(uuid4()),
            type="text",
            content="Hello from Aragora!",
            channel=OpenClawChannel.TELEGRAM,
        )
        result = await adapter.send_message(session.session_id, message)
    """

    # Default channel mappings
    DEFAULT_CHANNEL_MAPPINGS: dict[str, ChannelMapping] = {
        "whatsapp": ChannelMapping("whatsapp", OpenClawChannel.WHATSAPP, "whatsapp"),
        "telegram": ChannelMapping("telegram", OpenClawChannel.TELEGRAM, "telegram"),
        "slack": ChannelMapping("slack", OpenClawChannel.SLACK, "slack"),
        "discord": ChannelMapping("discord", OpenClawChannel.DISCORD, "discord"),
        "sms": ChannelMapping("sms", OpenClawChannel.SMS),
        "email": ChannelMapping("email", OpenClawChannel.EMAIL),
        "web": ChannelMapping("web", OpenClawChannel.WEB),
        "voice": ChannelMapping("voice", OpenClawChannel.VOICE),
        "teams": ChannelMapping("teams", OpenClawChannel.TEAMS),
        "matrix": ChannelMapping("matrix", OpenClawChannel.MATRIX),
    }

    # Default formatters
    DEFAULT_FORMATTERS: dict[str, type[ChannelFormatter]] = {
        "whatsapp": WhatsAppFormatter,
        "telegram": TelegramFormatter,
        "slack": SlackFormatter,
        "discord": DiscordFormatter,
    }

    def __init__(
        self,
        openclaw_endpoint: str = "http://localhost:8081",
        rbac_checker: RBACCheckerProtocol | None = None,
        audit_logger: AuditLoggerProtocol | None = None,
        approval_gate: ApprovalGateProtocol | None = None,
        sandbox_config: SandboxConfig | None = None,
        capability_filter: CapabilityFilter | None = None,
        protocol_translator: OpenClawProtocolTranslator | None = None,
        session_timeout_seconds: int = 3600,
        action_default_timeout: int = 60,
    ) -> None:
        """
        Initialize OpenClaw adapter.

        Args:
            openclaw_endpoint: OpenClaw runtime endpoint URL.
            rbac_checker: RBAC permission checker for authorization.
            audit_logger: Audit logging service.
            approval_gate: Approval gate service for sensitive operations.
            sandbox_config: Default sandbox configuration.
            capability_filter: Capability filter with policy rules.
            protocol_translator: Protocol translator instance.
            session_timeout_seconds: Default session timeout.
            action_default_timeout: Default action timeout in seconds.
        """
        self.openclaw_endpoint = openclaw_endpoint
        self.rbac_checker = rbac_checker
        self.audit_logger = audit_logger
        self.approval_gate = approval_gate
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.capability_filter = capability_filter or CapabilityFilter()
        self.protocol_translator = protocol_translator or OpenClawProtocolTranslator()
        self.session_timeout_seconds = session_timeout_seconds
        self.action_default_timeout = action_default_timeout

        # Sandbox for task execution
        self.sandbox = OpenClawSandbox(
            config=self.sandbox_config,
            openclaw_endpoint=openclaw_endpoint,
        )

        # Session storage
        self._sessions: dict[str, OpenClawSession] = {}
        self._sessions_lock = asyncio.Lock()

        # Channel mappings and formatters
        self._channel_mappings: dict[str, ChannelMapping] = dict(self.DEFAULT_CHANNEL_MAPPINGS)
        self._formatters: dict[str, ChannelFormatter] = {
            name: cls() for name, cls in self.DEFAULT_FORMATTERS.items()
        }

        # Action handlers
        self._action_handlers: dict[str, Callable[..., Any]] = {}
        self._register_default_action_handlers()

        # Event callbacks
        self._session_callbacks: list[Callable[[OpenClawSession, str], Any]] = []
        self._action_callbacks: list[Callable[[ActionResult], Any]] = []

    # ========================================================================
    # Session Management
    # ========================================================================

    async def create_session(
        self,
        user_id: str,
        channel: OpenClawChannel | str,
        tenant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        auth_context: AuthorizationContext | None = None,
    ) -> OpenClawSession:
        """
        Create a new OpenClaw session.

        Args:
            user_id: User identifier.
            channel: Communication channel.
            tenant_id: Optional tenant identifier.
            metadata: Session metadata.
            auth_context: Authorization context for permission checking.

        Returns:
            Created OpenClawSession.

        Raises:
            PermissionError: If user lacks session creation permission.
        """
        # Check permission if RBAC is enabled
        if self.rbac_checker and auth_context:
            has_permission = await self._check_permission(
                auth_context.actor_id,
                "openclaw.session.create",
            )
            if not has_permission:
                await self._log_audit(
                    OpenClawAuditEvents.TASK_BLOCKED,
                    auth_context.actor_id,
                    details={"reason": "permission_denied", "operation": "session.create"},
                )
                raise PermissionError("Permission denied: openclaw.session.create required")

        session_id = f"oc-sess-{uuid4().hex[:16]}"
        expires_at: datetime | None = None
        if self.session_timeout_seconds > 0:
            expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=self.session_timeout_seconds
            )

        session = OpenClawSession(
            session_id=session_id,
            user_id=user_id,
            channel=channel,
            state=SessionState.ACTIVE,
            tenant_id=tenant_id,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        async with self._sessions_lock:
            self._sessions[session_id] = session

        # Log session creation
        await self._log_audit(
            OpenClawAuditEvents.DEVICE_REGISTERED,
            user_id,
            session_id,
            {
                "channel": channel.value if isinstance(channel, OpenClawChannel) else channel,
                "tenant_id": tenant_id,
            },
        )

        # Notify callbacks
        for callback in self._session_callbacks:
            try:
                result = callback(session, "created")
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Session callback error: {e}")

        return session

    async def get_session(self, session_id: str) -> OpenClawSession | None:
        """
        Retrieve a session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            OpenClawSession or None if not found.
        """
        async with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session and session.expires_at:
                if datetime.now(timezone.utc) > session.expires_at:
                    session.state = SessionState.EXPIRED
            return session

    async def update_session(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> OpenClawSession | None:
        """
        Update session metadata and context.

        Args:
            session_id: Session identifier.
            metadata: New metadata to merge.
            context: New context to merge.

        Returns:
            Updated session or None if not found.
        """
        async with self._sessions_lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            session.update_activity()

            if metadata:
                session.metadata.update(metadata)
            if context:
                session.context.update(context)

            return session

    async def close_session(
        self,
        session_id: str,
        reason: str = "user_closed",
    ) -> OpenClawSession | None:
        """
        Close and terminate a session.

        Args:
            session_id: Session identifier.
            reason: Reason for closing.

        Returns:
            Closed session or None if not found.
        """
        async with self._sessions_lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            session.state = SessionState.TERMINATED
            session.metadata["close_reason"] = reason

        # Log session closure
        await self._log_audit(
            OpenClawAuditEvents.DEVICE_UNREGISTERED,
            session.user_id,
            session_id,
            {"reason": reason},
        )

        # Notify callbacks
        for callback in self._session_callbacks:
            try:
                result = callback(session, "closed")
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Session callback error: {e}")

        return session

    async def list_sessions(
        self,
        user_id: str | None = None,
        tenant_id: str | None = None,
        channel: OpenClawChannel | str | None = None,
        state: SessionState | None = None,
    ) -> list[OpenClawSession]:
        """
        List sessions with optional filters.

        Args:
            user_id: Filter by user.
            tenant_id: Filter by tenant.
            channel: Filter by channel.
            state: Filter by state.

        Returns:
            List of matching sessions.
        """
        async with self._sessions_lock:
            sessions = list(self._sessions.values())

        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        if tenant_id:
            sessions = [s for s in sessions if s.tenant_id == tenant_id]
        if channel:
            channel_value = channel.value if isinstance(channel, OpenClawChannel) else channel
            sessions = [
                s
                for s in sessions
                if (s.channel.value if isinstance(s.channel, OpenClawChannel) else s.channel)
                == channel_value
            ]
        if state:
            sessions = [s for s in sessions if s.state == state]

        return sessions

    def add_session_callback(
        self,
        callback: Callable[[OpenClawSession, str], Any],
    ) -> None:
        """Register a callback for session lifecycle events."""
        self._session_callbacks.append(callback)

    # ========================================================================
    # Channel Mapping
    # ========================================================================

    def register_channel_mapping(self, mapping: ChannelMapping) -> None:
        """
        Register a custom channel mapping.

        Args:
            mapping: Channel mapping configuration.
        """
        self._channel_mappings[mapping.aragora_channel] = mapping

    def get_channel_mapping(self, aragora_channel: str) -> ChannelMapping | None:
        """
        Get channel mapping for an Aragora channel.

        Args:
            aragora_channel: Aragora channel identifier.

        Returns:
            ChannelMapping or None if not found.
        """
        return self._channel_mappings.get(aragora_channel)

    def register_formatter(
        self,
        channel: str,
        formatter: ChannelFormatter,
    ) -> None:
        """
        Register a custom channel formatter.

        Args:
            channel: Channel identifier.
            formatter: Formatter instance.
        """
        self._formatters[channel] = formatter

    def get_formatter(self, channel: str) -> ChannelFormatter:
        """
        Get formatter for a channel.

        Args:
            channel: Channel identifier.

        Returns:
            ChannelFormatter (default if not found).
        """
        return self._formatters.get(channel, ChannelFormatter())

    def map_aragora_to_openclaw_channel(
        self,
        aragora_channel: str,
    ) -> OpenClawChannel | str:
        """
        Map Aragora channel to OpenClaw channel.

        Args:
            aragora_channel: Aragora channel identifier.

        Returns:
            OpenClaw channel.
        """
        mapping = self._channel_mappings.get(aragora_channel)
        if mapping:
            return mapping.openclaw_channel
        return aragora_channel

    def map_openclaw_to_aragora_channel(
        self,
        openclaw_channel: OpenClawChannel | str,
    ) -> str:
        """
        Map OpenClaw channel to Aragora channel.

        Args:
            openclaw_channel: OpenClaw channel.

        Returns:
            Aragora channel identifier.
        """
        openclaw_value = (
            openclaw_channel.value
            if isinstance(openclaw_channel, OpenClawChannel)
            else openclaw_channel
        )
        for aragora_ch, mapping in self._channel_mappings.items():
            mapping_value = (
                mapping.openclaw_channel.value
                if isinstance(mapping.openclaw_channel, OpenClawChannel)
                else mapping.openclaw_channel
            )
            if mapping_value == openclaw_value:
                return aragora_ch
        return openclaw_value

    # ========================================================================
    # Protocol Translation
    # ========================================================================

    def convert_aragora_request(
        self,
        request: AragoraRequest,
        auth_context: AuthorizationContext | None = None,
        tenant_context: TenantContext | None = None,
    ) -> OpenClawTask:
        """
        Convert Aragora request to OpenClaw task.

        Args:
            request: Aragora-format request.
            auth_context: Authorization context.
            tenant_context: Tenant context.

        Returns:
            OpenClawTask for execution.
        """
        return self.protocol_translator.aragora_to_openclaw(
            request,
            auth_context=auth_context,
            tenant_context=tenant_context,
        )

    def convert_openclaw_response(
        self,
        task: OpenClawTask,
        openclaw_result: dict[str, Any],
    ) -> AragoraResponse:
        """
        Convert OpenClaw result to Aragora response.

        Args:
            task: Original OpenClaw task.
            openclaw_result: Result from OpenClaw.

        Returns:
            AragoraResponse.
        """
        return self.protocol_translator.openclaw_to_aragora(task, openclaw_result)

    def convert_aragora_message(
        self,
        aragora_msg: dict[str, Any],
        session: OpenClawSession,
    ) -> OpenClawMessage:
        """
        Convert Aragora message format to OpenClaw message.

        Args:
            aragora_msg: Aragora message dictionary.
            session: Associated session.

        Returns:
            OpenClawMessage.
        """
        return OpenClawMessage(
            message_id=aragora_msg.get("message_id", str(uuid4())),
            type=aragora_msg.get("type", "text"),
            content=aragora_msg.get("content"),
            channel=session.channel,
            metadata=aragora_msg.get("metadata", {}),
            sender_id=aragora_msg.get("sender_id", session.user_id),
            attachments=aragora_msg.get("attachments", []),
            reply_to=aragora_msg.get("reply_to"),
            thread_id=aragora_msg.get("thread_id"),
        )

    def convert_openclaw_message(
        self,
        openclaw_msg: OpenClawMessage,
    ) -> dict[str, Any]:
        """
        Convert OpenClaw message to Aragora format.

        Args:
            openclaw_msg: OpenClaw message.

        Returns:
            Aragora message dictionary.
        """
        return {
            "message_id": openclaw_msg.message_id,
            "type": openclaw_msg.type,
            "content": openclaw_msg.content,
            "channel": self.map_openclaw_to_aragora_channel(openclaw_msg.channel),
            "metadata": openclaw_msg.metadata,
            "sender_id": openclaw_msg.sender_id,
            "timestamp": openclaw_msg.timestamp.isoformat(),
            "attachments": openclaw_msg.attachments,
            "reply_to": openclaw_msg.reply_to,
            "thread_id": openclaw_msg.thread_id,
        }

    # ========================================================================
    # Action Execution
    # ========================================================================

    async def execute_action(
        self,
        session_id: str,
        action: OpenClawAction,
        auth_context: AuthorizationContext | None = None,
    ) -> ActionResult:
        """
        Execute an OpenClaw action.

        Args:
            session_id: Session identifier.
            action: Action to execute.
            auth_context: Authorization context.

        Returns:
            ActionResult with execution details.
        """
        action_id = str(uuid4())
        started_at = datetime.now(timezone.utc)
        start_time = time.monotonic()

        # Validate session
        session = await self.get_session(session_id)
        if not session:
            return ActionResult(
                action_id=action_id,
                status=ActionStatus.FAILED,
                error="Session not found",
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )

        if not session.is_active():
            return ActionResult(
                action_id=action_id,
                status=ActionStatus.FAILED,
                error=f"Session is {session.state.value}",
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )

        # Check permission if RBAC is enabled
        if self.rbac_checker and auth_context:
            action_type_value = (
                action.action_type.value
                if isinstance(action.action_type, OpenClawActionType)
                else action.action_type
            )
            permission = f"openclaw.action.{action_type_value}"
            has_permission = await self._check_permission(
                auth_context.actor_id,
                permission,
            )
            if not has_permission:
                await self._log_audit(
                    OpenClawAuditEvents.TASK_BLOCKED,
                    auth_context.actor_id,
                    action_id,
                    {"reason": "permission_denied", "action_type": action_type_value},
                )
                return ActionResult(
                    action_id=action_id,
                    status=ActionStatus.FAILED,
                    error=f"Permission denied: {permission} required",
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                )

        # Check capability filter
        action_capability = self._action_to_capability(action.action_type)
        if action_capability:
            check_result = self.capability_filter.check(action_capability)
            if not check_result.allowed and not check_result.requires_approval:
                await self._log_audit(
                    OpenClawAuditEvents.CAPABILITY_DENIED,
                    session.user_id,
                    action_id,
                    {"capability": action_capability},
                )
                return ActionResult(
                    action_id=action_id,
                    status=ActionStatus.FAILED,
                    error=f"Capability blocked: {action_capability}",
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                )

        # Log action start
        await self._log_audit(
            OpenClawAuditEvents.TASK_SUBMITTED,
            session.user_id,
            action_id,
            {
                "action_type": action.action_type.value
                if isinstance(action.action_type, OpenClawActionType)
                else action.action_type,
                "session_id": session_id,
            },
        )

        # Execute action
        try:
            timeout = action.timeout or self.action_default_timeout
            result = await asyncio.wait_for(
                self._execute_action_impl(action, session),
                timeout=timeout,
            )
            execution_time_ms = int((time.monotonic() - start_time) * 1000)

            action_result = ActionResult(
                action_id=action_id,
                status=ActionStatus.COMPLETED,
                result=result,
                execution_time_ms=execution_time_ms,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )

            # Log completion
            await self._log_audit(
                OpenClawAuditEvents.TASK_COMPLETED,
                session.user_id,
                action_id,
                {"execution_time_ms": execution_time_ms},
            )

        except asyncio.TimeoutError:
            action_result = ActionResult(
                action_id=action_id,
                status=ActionStatus.TIMEOUT,
                error=f"Action timed out after {action.timeout}s",
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )

            await self._log_audit(
                OpenClawAuditEvents.TASK_TIMEOUT,
                session.user_id,
                action_id,
            )

        except Exception as e:
            logger.exception(f"Action {action_id} failed")
            action_result = ActionResult(
                action_id=action_id,
                status=ActionStatus.FAILED,
                error=str(e),
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )

            await self._log_audit(
                OpenClawAuditEvents.TASK_FAILED,
                session.user_id,
                action_id,
                {"error": str(e)},
            )

        # Update session activity
        session.update_activity()

        # Notify callbacks
        for callback in self._action_callbacks:
            try:
                cb_result = callback(action_result)
                if asyncio.iscoroutine(cb_result):
                    await cb_result
            except Exception as e:
                logger.error(f"Action callback error: {e}")

        return action_result

    async def _execute_action_impl(
        self,
        action: OpenClawAction,
        session: OpenClawSession,
    ) -> Any:
        """Internal action execution implementation."""
        action_type_value = (
            action.action_type.value
            if isinstance(action.action_type, OpenClawActionType)
            else action.action_type
        )

        # Check for registered handler
        handler = self._action_handlers.get(action_type_value)
        if handler:
            result = handler(action, session)
            if asyncio.iscoroutine(result):
                return await result
            return result

        # Forward to OpenClaw runtime
        return await self._forward_action_to_openclaw(action, session)

    async def _forward_action_to_openclaw(
        self,
        action: OpenClawAction,
        session: OpenClawSession,
    ) -> Any:
        """Forward action to OpenClaw runtime via HTTP."""
        try:
            import aiohttp
        except ImportError:
            logger.warning("aiohttp not available, returning mock result")
            return {"mock": True, "action_type": action.action_type}

        try:
            async with aiohttp.ClientSession() as http_session:
                async with http_session.post(
                    f"{self.openclaw_endpoint}/api/v1/actions",
                    json={
                        "action": action.to_dict(),
                        "session_id": session.session_id,
                        "user_id": session.user_id,
                        "tenant_id": session.tenant_id,
                    },
                    timeout=aiohttp.ClientTimeout(total=action.timeout),
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"OpenClaw returned {response.status}: {error_text}")

        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to connect to OpenClaw: {e}")

    def _action_to_capability(
        self,
        action_type: OpenClawActionType | str,
    ) -> str | None:
        """Map action type to capability for filtering."""
        action_value = (
            action_type.value if isinstance(action_type, OpenClawActionType) else action_type
        )

        capability_map = {
            "browser_navigate": "browser_automation",
            "browser_click": "browser_automation",
            "browser_type": "browser_automation",
            "browser_screenshot": "browser_automation",
            "browser_scroll": "browser_automation",
            "browser_wait": "browser_automation",
            "browser_extract": "browser_automation",
            "file_read": "file_system_read",
            "file_write": "file_system_write",
            "file_delete": "file_system_write",
            "file_list": "file_system_read",
            "http_request": "network_external",
            "api_call": "network_external",
            "code_run": "code_execution",
            "code_eval": "code_execution",
            "db_query": "database_read",
            "db_execute": "database_write",
            "cron_create": "scheduler_manage",
            "cron_update": "scheduler_manage",
            "cron_delete": "scheduler_manage",
            "cron_list": "scheduler_read",
            "cron_trigger": "scheduler_manage",
        }

        return capability_map.get(action_value)

    def _register_default_action_handlers(self) -> None:
        """Register default action handlers."""
        # These are placeholder handlers - in production, they would
        # integrate with actual browser automation, file systems, etc.
        pass

    def register_action_handler(
        self,
        action_type: OpenClawActionType | str,
        handler: Callable[..., Any],
    ) -> None:
        """
        Register a custom action handler.

        Args:
            action_type: Action type to handle.
            handler: Handler function (sync or async).
        """
        action_value = (
            action_type.value if isinstance(action_type, OpenClawActionType) else action_type
        )
        self._action_handlers[action_value] = handler

    def add_action_callback(
        self,
        callback: Callable[[ActionResult], Any],
    ) -> None:
        """Register a callback for action completion events."""
        self._action_callbacks.append(callback)

    # ========================================================================
    # Message Sending
    # ========================================================================

    async def send_message(
        self,
        session_id: str,
        message: OpenClawMessage,
    ) -> dict[str, Any]:
        """
        Send a message through the appropriate channel.

        Args:
            session_id: Session identifier.
            message: Message to send.

        Returns:
            Sending result from channel.
        """
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if not session.is_active():
            raise ValueError(f"Session is {session.state.value}")

        # Get formatter for channel
        channel_value = (
            session.channel.value
            if isinstance(session.channel, OpenClawChannel)
            else session.channel
        )
        formatter = self.get_formatter(channel_value)

        # Format message for channel
        formatted = formatter.format_outgoing(message, session)

        # Send via OpenClaw (in production, this would call the actual channel API)
        result = await self._send_to_openclaw(session, formatted)

        # Update session activity
        session.update_activity()

        return result

    async def _send_to_openclaw(
        self,
        session: OpenClawSession,
        formatted_message: dict[str, Any],
    ) -> dict[str, Any]:
        """Send formatted message via OpenClaw."""
        try:
            import aiohttp
        except ImportError:
            logger.warning("aiohttp not available, returning mock result")
            return {"mock": True, "sent": True}

        channel_value = (
            session.channel.value
            if isinstance(session.channel, OpenClawChannel)
            else session.channel
        )

        try:
            async with aiohttp.ClientSession() as http_session:
                async with http_session.post(
                    f"{self.openclaw_endpoint}/api/v1/channels/{channel_value}/send",
                    json={
                        "message": formatted_message,
                        "session_id": session.session_id,
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"Send failed: {response.status}: {error_text}")

        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to send message: {e}")

    # ========================================================================
    # Internal Helpers
    # ========================================================================

    async def _check_permission(
        self,
        actor_id: str,
        permission: str,
    ) -> bool:
        """Check if actor has permission."""
        if not self.rbac_checker:
            return True

        try:
            return await self.rbac_checker.check_permission_async(
                actor_id,
                permission,
            )
        except Exception:
            # Fallback to sync check
            return self.rbac_checker.check_permission(actor_id, permission)

    async def _log_audit(
        self,
        event: OpenClawAuditEvents,
        actor_id: str,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log an audit event."""
        if not self.audit_logger:
            logger.info(f"Audit: {event.value} actor={actor_id} resource={resource_id}")
            return

        severity = get_event_severity(event)

        try:
            await self.audit_logger.log_async(
                event.value,
                actor_id,
                resource_id,
                details,
                severity,
            )
        except Exception:
            # Fallback to sync log
            self.audit_logger.log(
                event.value,
                actor_id,
                resource_id,
                details,
                severity,
            )


# ============================================================================
# Legacy OpenClawGatewayAdapter (for backward compatibility)
# ============================================================================


class OpenClawGatewayAdapter(OpenClawAdapter):
    """
    Legacy gateway adapter for backward compatibility.

    This class maintains the original OpenClawGatewayAdapter API while
    inheriting from the new OpenClawAdapter implementation.

    Usage:
        adapter = OpenClawGatewayAdapter(
            openclaw_endpoint="http://localhost:8081",
            rbac_checker=checker,
            audit_logger=logger,
        )

        result = await adapter.execute_task(
            request=AragoraRequest(content="Generate a summary"),
            auth_context=ctx,
        )
    """

    async def execute_task(
        self,
        request: AragoraRequest,
        auth_context: AuthorizationContext,
        tenant_context: TenantContext | None = None,
        sandbox_override: SandboxConfig | None = None,
    ) -> GatewayResult:
        """
        Execute task via OpenClaw with full security enforcement.

        Args:
            request: Aragora-format request
            auth_context: Authorization context
            tenant_context: Optional tenant context for isolation
            sandbox_override: Optional sandbox config override

        Returns:
            GatewayResult with execution details
        """
        request_id = str(uuid4())

        # Check RBAC permission
        if self.rbac_checker:
            has_permission = await self._check_permission(
                auth_context.actor_id,
                "gateway.execute",
            )
            if not has_permission:
                await self._log_audit(
                    OpenClawAuditEvents.TASK_BLOCKED,
                    auth_context.actor_id,
                    request_id,
                    {"reason": "permission_denied"},
                )
                return GatewayResult(
                    success=False,
                    request_id=request_id,
                    error="Permission denied: gateway.execute required",
                    blocked_reason="permission_denied",
                )

        # Check capabilities
        blocked_capabilities: list[str] = []
        approval_needed: list[CapabilityCheckResult] = []

        for capability in request.capabilities:
            check_result = self.capability_filter.check(capability)
            if not check_result.allowed:
                if check_result.requires_approval:
                    approval_needed.append(check_result)
                else:
                    blocked_capabilities.append(capability)

        # Block if any capabilities are not allowed
        if blocked_capabilities:
            await self._log_audit(
                OpenClawAuditEvents.CAPABILITY_DENIED,
                auth_context.actor_id,
                request_id,
                {"blocked_capabilities": blocked_capabilities},
            )
            return GatewayResult(
                success=False,
                request_id=request_id,
                error=f"Capabilities blocked: {blocked_capabilities}",
                blocked_reason="capability_blocked",
            )

        # Check approval gates for capabilities that require them
        if approval_needed and self.approval_gate:
            for check_result in approval_needed:
                if check_result.approval_gate:
                    approved, reason = await self.approval_gate.check_approval(
                        check_result.approval_gate,
                        auth_context.actor_id,
                        request_id,
                    )
                    if not approved:
                        await self._log_audit(
                            OpenClawAuditEvents.CAPABILITY_DENIED,
                            auth_context.actor_id,
                            request_id,
                            {
                                "capability": check_result.capability,
                                "approval_gate": check_result.approval_gate,
                                "reason": reason,
                            },
                        )
                        return GatewayResult(
                            success=False,
                            request_id=request_id,
                            error=f"Approval required for {check_result.capability}: {reason}",
                            blocked_reason="approval_required",
                        )

        # Log task submission
        await self._log_audit(
            OpenClawAuditEvents.TASK_SUBMITTED,
            auth_context.actor_id,
            request_id,
            {
                "request_type": request.request_type,
                "capabilities": request.capabilities,
                "plugins": request.plugins,
            },
        )

        # Translate to OpenClaw format
        task = self.protocol_translator.aragora_to_openclaw(
            request,
            auth_context=auth_context,
            tenant_context=tenant_context,
        )

        # Override task ID with our request ID for tracing
        task.id = request_id

        # Execute in sandbox
        sandbox_result = await self.sandbox.execute(
            task,
            config_override=sandbox_override,
        )

        # Log completion
        if sandbox_result.status == SandboxStatus.COMPLETED:
            await self._log_audit(
                OpenClawAuditEvents.TASK_COMPLETED,
                auth_context.actor_id,
                request_id,
                {
                    "execution_time_ms": sandbox_result.execution_time_ms,
                    "memory_used_mb": sandbox_result.memory_used_mb,
                },
            )
        elif sandbox_result.status == SandboxStatus.TIMEOUT:
            await self._log_audit(
                OpenClawAuditEvents.TASK_TIMEOUT,
                auth_context.actor_id,
                request_id,
                {"error": sandbox_result.error},
            )
        elif sandbox_result.status == SandboxStatus.POLICY_VIOLATION:
            await self._log_audit(
                OpenClawAuditEvents.SANDBOX_VIOLATION,
                auth_context.actor_id,
                request_id,
                {"error": sandbox_result.error},
            )
        else:
            await self._log_audit(
                OpenClawAuditEvents.TASK_FAILED,
                auth_context.actor_id,
                request_id,
                {"error": sandbox_result.error},
            )

        # Convert to Aragora response
        response = self.protocol_translator.openclaw_to_aragora(
            task,
            {
                "status": sandbox_result.status.value,
                "result": sandbox_result.output,
                "error": sandbox_result.error,
                "execution_time_ms": sandbox_result.execution_time_ms,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        return GatewayResult(
            success=sandbox_result.status == SandboxStatus.COMPLETED,
            request_id=request_id,
            response=response,
            error=sandbox_result.error,
            metadata={
                "sandbox_status": sandbox_result.status.value,
                "execution_time_ms": sandbox_result.execution_time_ms,
            },
        )

    async def register_device(
        self,
        device: DeviceRegistration,
        auth_context: AuthorizationContext,
    ) -> GatewayResult:
        """
        Register a device with the OpenClaw gateway.

        Args:
            device: Device registration details
            auth_context: Authorization context

        Returns:
            GatewayResult with device handle
        """
        # Check RBAC permission
        if self.rbac_checker:
            has_permission = await self._check_permission(
                auth_context.actor_id,
                "gateway.device.register",
            )
            if not has_permission:
                return GatewayResult(
                    success=False,
                    request_id=device.device_id,
                    error="Permission denied: gateway.device.register required",
                    blocked_reason="permission_denied",
                )

        # Log registration
        await self._log_audit(
            OpenClawAuditEvents.DEVICE_REGISTERED,
            auth_context.actor_id,
            device.device_id,
            {
                "device_name": device.device_name,
                "device_type": device.device_type,
                "capabilities": device.capabilities,
            },
        )

        # Create device handle
        handle = DeviceHandle(
            device_id=device.device_id,
            registration_id=str(uuid4()),
            registered_at=datetime.now(timezone.utc),
        )

        return GatewayResult(
            success=True,
            request_id=device.device_id,
            metadata={"device_handle": handle.__dict__},
        )

    async def unregister_device(
        self,
        device_id: str,
        auth_context: AuthorizationContext,
    ) -> GatewayResult:
        """Unregister a device from the gateway."""
        if self.rbac_checker:
            has_permission = await self._check_permission(
                auth_context.actor_id,
                "gateway.device.unregister",
            )
            if not has_permission:
                return GatewayResult(
                    success=False,
                    request_id=device_id,
                    error="Permission denied",
                    blocked_reason="permission_denied",
                )

        await self._log_audit(
            OpenClawAuditEvents.DEVICE_UNREGISTERED,
            auth_context.actor_id,
            device_id,
        )

        return GatewayResult(
            success=True,
            request_id=device_id,
        )

    async def install_plugin(
        self,
        plugin: PluginInstallRequest,
        auth_context: AuthorizationContext,
        tenant_context: TenantContext | None = None,
    ) -> GatewayResult:
        """
        Install a plugin via the OpenClaw gateway.

        Args:
            plugin: Plugin installation request
            auth_context: Authorization context
            tenant_context: Optional tenant context

        Returns:
            GatewayResult with installation status
        """
        # Check RBAC permission
        if self.rbac_checker:
            has_permission = await self._check_permission(
                auth_context.actor_id,
                "gateway.plugin.install",
            )
            if not has_permission:
                await self._log_audit(
                    OpenClawAuditEvents.PLUGIN_BLOCKED,
                    auth_context.actor_id,
                    plugin.plugin_id,
                    {"reason": "permission_denied"},
                )
                return GatewayResult(
                    success=False,
                    request_id=plugin.plugin_id,
                    error="Permission denied: gateway.plugin.install required",
                    blocked_reason="permission_denied",
                )

        # Check plugin allowlist if enabled
        if self.sandbox_config.plugin_allowlist_mode:
            if plugin.plugin_id not in self.sandbox_config.allowed_plugins:
                await self._log_audit(
                    OpenClawAuditEvents.PLUGIN_BLOCKED,
                    auth_context.actor_id,
                    plugin.plugin_id,
                    {"reason": "not_in_allowlist"},
                )
                return GatewayResult(
                    success=False,
                    request_id=plugin.plugin_id,
                    error=f"Plugin '{plugin.plugin_id}' not in allowlist",
                    blocked_reason="plugin_not_allowed",
                )

        # Log installation
        await self._log_audit(
            OpenClawAuditEvents.PLUGIN_INSTALLED,
            auth_context.actor_id,
            plugin.plugin_id,
            {
                "plugin_name": plugin.plugin_name,
                "version": plugin.version,
                "source": plugin.source,
            },
        )

        return GatewayResult(
            success=True,
            request_id=plugin.plugin_id,
            metadata={"installed_at": datetime.now(timezone.utc).isoformat()},
        )

    async def uninstall_plugin(
        self,
        plugin_id: str,
        auth_context: AuthorizationContext,
    ) -> GatewayResult:
        """Uninstall a plugin from the gateway."""
        if self.rbac_checker:
            has_permission = await self._check_permission(
                auth_context.actor_id,
                "gateway.plugin.uninstall",
            )
            if not has_permission:
                return GatewayResult(
                    success=False,
                    request_id=plugin_id,
                    error="Permission denied",
                    blocked_reason="permission_denied",
                )

        await self._log_audit(
            OpenClawAuditEvents.PLUGIN_UNINSTALLED,
            auth_context.actor_id,
            plugin_id,
        )

        return GatewayResult(
            success=True,
            request_id=plugin_id,
        )

    def update_sandbox_config(self, config: SandboxConfig) -> None:
        """Update default sandbox configuration."""
        self.sandbox_config = config
        self.sandbox = OpenClawSandbox(
            config=config,
            openclaw_endpoint=self.openclaw_endpoint,
        )

    def update_capability_filter(self, filter_config: CapabilityFilter) -> None:
        """Update capability filter configuration."""
        self.capability_filter = filter_config

    def enable_tenant_capability(
        self,
        capability: str,
        tenant_context: TenantContext,
    ) -> None:
        """Enable a capability for a specific tenant."""
        self.capability_filter.enable_for_tenant(capability)
        tenant_context.enabled_capabilities.add(capability)

    def add_plugin_to_allowlist(self, plugin_id: str) -> None:
        """Add a plugin to the allowed plugins list."""
        if plugin_id not in self.sandbox_config.allowed_plugins:
            self.sandbox_config.allowed_plugins.append(plugin_id)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # New adapter classes
    "OpenClawAdapter",
    "OpenClawGatewayAdapter",
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
    # Legacy dataclasses
    "GatewayResult",
    "DeviceRegistration",
    "DeviceHandle",
    "PluginInstallRequest",
    # Formatters
    "ChannelFormatter",
    "WhatsAppFormatter",
    "TelegramFormatter",
    "SlackFormatter",
    "DiscordFormatter",
    # Protocol interfaces
    "RBACCheckerProtocol",
    "AuditLoggerProtocol",
    "ApprovalGateProtocol",
]
