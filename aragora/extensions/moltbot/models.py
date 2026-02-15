"""
Moltbot Extension Data Models.

Defines the core data structures for consumer/device interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal


class ChannelType(Enum):
    """Supported communication channel types."""

    SMS = "sms"
    EMAIL = "email"
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    SLACK = "slack"
    DISCORD = "discord"
    WEB = "web"
    VOICE = "voice"
    PUSH = "push"


class InboxMessageStatus(Enum):
    """Message processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    DELIVERED = "delivered"
    READ = "read"
    RESPONDED = "responded"
    FAILED = "failed"
    ARCHIVED = "archived"


@dataclass
class ChannelConfig:
    """Configuration for a communication channel."""

    type: ChannelType
    name: str
    credentials: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 0  # Higher = preferred
    rate_limit: int = 100  # Messages per minute
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Channel:
    """A communication channel instance."""

    id: str
    config: ChannelConfig
    user_id: str
    tenant_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: Literal["active", "paused", "disconnected"] = "active"
    last_message_at: datetime | None = None
    message_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InboxMessage:
    """A message in the unified inbox."""

    id: str
    channel_id: str
    user_id: str
    direction: Literal["inbound", "outbound"]
    content: str
    content_type: str = "text"  # text, image, audio, video, file
    status: InboxMessageStatus = InboxMessageStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    delivered_at: datetime | None = None
    read_at: datetime | None = None

    # Threading
    thread_id: str | None = None
    reply_to: str | None = None

    # Processing
    intent: str | None = None
    entities: dict[str, Any] = field(default_factory=dict)
    sentiment: float | None = None  # -1 to 1

    # Response tracking
    response_id: str | None = None
    response_time_ms: int | None = None

    # Metadata
    external_id: str | None = None  # Provider's message ID
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceNodeConfig:
    """Configuration for a device node."""

    name: str
    device_type: str  # "iot", "mobile", "desktop", "embedded"
    capabilities: list[str] = field(default_factory=list)
    connection_type: str = "mqtt"  # mqtt, websocket, http
    heartbeat_interval: int = 60  # seconds
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceNode:
    """A connected device in the network."""

    id: str
    config: DeviceNodeConfig
    user_id: str
    gateway_id: str
    tenant_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: Literal["online", "offline", "error"] = "offline"
    last_seen: datetime | None = None
    last_heartbeat: datetime | None = None

    # State
    state: dict[str, Any] = field(default_factory=dict)
    firmware_version: str = ""
    battery_level: float | None = None
    signal_strength: float | None = None

    # Metrics
    messages_sent: int = 0
    messages_received: int = 0
    errors: int = 0
    uptime_seconds: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceSessionConfig:
    """Configuration for a voice session."""

    language: str = "en-US"
    voice_id: str = "default"
    sample_rate: int = 16000
    encoding: str = "pcm"
    enable_stt: bool = True
    enable_tts: bool = True
    enable_vad: bool = True  # Voice activity detection
    silence_timeout: float = 2.0  # seconds
    max_duration: float = 300.0  # 5 minutes
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceSession:
    """An active voice interaction session."""

    id: str
    config: VoiceSessionConfig
    user_id: str
    channel_id: str
    tenant_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: Literal["active", "paused", "ended"] = "active"
    started_at: datetime | None = None
    ended_at: datetime | None = None

    # Transcript
    transcripts: list[dict[str, Any]] = field(default_factory=list)
    current_transcript: str = ""

    # Processing
    intent_history: list[str] = field(default_factory=list)
    entities_extracted: dict[str, Any] = field(default_factory=dict)

    # Metrics
    duration_seconds: float = 0.0
    turns: int = 0
    words_spoken: int = 0
    words_heard: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OnboardingStep:
    """A step in an onboarding flow."""

    id: str
    name: str
    type: Literal["info", "input", "verification", "action", "decision"]
    content: dict[str, Any] = field(default_factory=dict)
    required: bool = True
    order: int = 0
    timeout_seconds: int | None = None
    retry_limit: int = 3
    validation: dict[str, Any] | None = None
    next_step: str | None = None
    branch_conditions: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OnboardingFlow:
    """An onboarding flow definition."""

    id: str
    name: str
    description: str = ""
    steps: list[OnboardingStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: Literal["draft", "active", "archived"] = "draft"

    # Targeting
    target_segment: str | None = None
    channels: list[ChannelType] = field(default_factory=list)

    # Metrics
    started_count: int = 0
    completed_count: int = 0
    abandoned_count: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OnboardingSession:
    """A user's progress through an onboarding flow."""

    id: str
    flow_id: str
    user_id: str
    channel_id: str
    tenant_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: Literal["in_progress", "completed", "abandoned", "paused"] = "in_progress"
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Progress
    current_step: str | None = None
    completed_steps: list[str] = field(default_factory=list)
    step_data: dict[str, Any] = field(default_factory=dict)
    retries: dict[str, int] = field(default_factory=dict)

    # Collected data
    collected_data: dict[str, Any] = field(default_factory=dict)
    verification_status: dict[str, bool] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)
