"""
Type definitions for Moltbot HTTP handlers.

This module provides TypedDicts for request and response bodies,
ensuring type safety across all moltbot API endpoints.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, TypedDict

from typing_extensions import NotRequired


# =============================================================================
# Helper Functions
# =============================================================================


def serialize_enum(value: Any, default: str = "") -> str:
    """Type-safe enum serialization.

    Args:
        value: Enum value or string to serialize
        default: Default value if input is None

    Returns:
        String value of the enum or the original string
    """
    if value is None:
        return default
    if isinstance(value, Enum):
        return value.value
    return str(value)


def serialize_datetime(dt: Any) -> str | None:
    """Type-safe datetime serialization.

    Args:
        dt: Datetime object or None

    Returns:
        ISO format string or None
    """
    if dt is None:
        return None
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    return str(dt)


# =============================================================================
# Gateway Types
# =============================================================================


class RegisterDeviceRequest(TypedDict, total=False):
    """Request body for POST /api/v1/moltbot/devices."""

    name: str
    device_type: str
    capabilities: list[str]
    metadata: dict[str, Any]
    tenant_id: str


class SendCommandRequest(TypedDict):
    """Request body for POST /api/v1/moltbot/devices/{id}/command."""

    command: str
    payload: NotRequired[dict[str, Any]]
    timeout: NotRequired[float]


class HeartbeatRequest(TypedDict, total=False):
    """Request body for POST /api/v1/moltbot/devices/{id}/heartbeat."""

    state: dict[str, Any]
    metrics: dict[str, Any]


class DeviceSummaryResponse(TypedDict):
    """Device summary in list responses."""

    id: str
    name: str
    device_type: str
    status: str
    user_id: str
    last_seen: str | None
    battery_level: float | None
    signal_strength: float | None


class DeviceDetailResponse(TypedDict):
    """Device detail in GET /api/v1/moltbot/devices/{id}."""

    id: str
    name: str
    device_type: str
    status: str
    user_id: str
    gateway_id: str
    tenant_id: str | None
    state: dict[str, Any]
    capabilities: list[str]
    last_heartbeat: str | None
    last_seen: str | None
    battery_level: float | None
    signal_strength: float | None
    firmware_version: str
    messages_sent: int
    messages_received: int
    errors: int
    created_at: str | None
    updated_at: str | None


class DeviceListResponse(TypedDict):
    """Response for GET /api/v1/moltbot/devices."""

    devices: list[DeviceSummaryResponse]
    total: int


class HeartbeatResponse(TypedDict):
    """Response for POST /api/v1/moltbot/devices/{id}/heartbeat."""

    success: bool
    device_id: str
    status: str
    last_heartbeat: str | None


# =============================================================================
# Inbox Types
# =============================================================================


class RegisterChannelRequest(TypedDict):
    """Request body for POST /api/v1/moltbot/channels."""

    name: str
    type: str
    provider_config: NotRequired[dict[str, Any]]
    metadata: NotRequired[dict[str, Any]]
    tenant_id: NotRequired[str]


class SendMessageRequest(TypedDict):
    """Request body for POST /api/v1/moltbot/messages."""

    channel_id: str
    recipient_user_id: str
    content: str
    content_type: NotRequired[str]
    thread_id: NotRequired[str]
    reply_to: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]


class ChannelSummaryResponse(TypedDict):
    """Channel summary in list responses."""

    id: str
    name: str
    type: str
    status: str
    user_id: str
    message_count: int
    last_message_at: str | None


class ChannelDetailResponse(TypedDict):
    """Channel detail in GET /api/v1/moltbot/channels/{id}."""

    id: str
    name: str
    type: str
    status: str
    user_id: str
    tenant_id: str | None
    message_count: int
    last_message_at: str | None
    created_at: str | None
    updated_at: str | None


class MessageSummaryResponse(TypedDict):
    """Message summary in list responses."""

    id: str
    channel_id: str
    user_id: str
    direction: str
    content: str
    content_type: str
    status: str
    thread_id: str | None
    reply_to: str | None
    intent: str | None
    created_at: str | None


class MessageDetailResponse(TypedDict):
    """Message detail in GET /api/v1/moltbot/messages/{id}."""

    id: str
    channel_id: str
    user_id: str
    direction: str
    content: str
    content_type: str
    status: str
    thread_id: str | None
    reply_to: str | None
    external_id: str | None
    intent: str | None
    metadata: dict[str, Any]
    created_at: str | None
    delivered_at: str | None
    read_at: str | None


class MessageListResponse(TypedDict):
    """Response for GET /api/v1/moltbot/messages."""

    messages: list[MessageSummaryResponse]
    total: int
    limit: int
    offset: int


# =============================================================================
# Voice Types
# =============================================================================


class VoiceConfigRequest(TypedDict, total=False):
    """Voice session configuration in requests."""

    sample_rate: int
    channels: int
    encoding: str
    language: str
    enable_stt: bool
    enable_tts: bool


class CreateVoiceSessionRequest(TypedDict):
    """Request body for POST /api/v1/moltbot/voice/sessions."""

    channel_id: str
    config: NotRequired[VoiceConfigRequest]
    tenant_id: NotRequired[str]


class ProcessAudioRequest(TypedDict):
    """Request body for POST /api/v1/moltbot/voice/sessions/{id}/audio."""

    audio: str  # base64 encoded


class SynthesizeSpeechRequest(TypedDict):
    """Request body for POST /api/v1/moltbot/voice/sessions/{id}/speak."""

    text: str


class VoiceConfigResponse(TypedDict):
    """Voice configuration in responses."""

    sample_rate: int
    encoding: str
    language: str
    enable_stt: NotRequired[bool]
    enable_tts: NotRequired[bool]


class VoiceSessionSummaryResponse(TypedDict):
    """Voice session summary in list responses."""

    id: str
    user_id: str
    channel_id: str
    status: str
    turns: int
    duration_seconds: float
    started_at: str | None


class VoiceSessionDetailResponse(TypedDict):
    """Voice session detail in GET /api/v1/moltbot/voice/sessions/{id}."""

    id: str
    user_id: str
    channel_id: str
    tenant_id: str | None
    status: str
    turns: int
    words_spoken: int
    words_heard: int
    duration_seconds: float
    current_transcript: str
    intent_history: list[str]
    config: VoiceConfigResponse
    started_at: str | None
    ended_at: str | None


# =============================================================================
# Canvas Types
# =============================================================================


class CreateCanvasRequest(TypedDict):
    """Request body for POST /api/v1/moltbot/canvas."""

    name: str
    width: NotRequired[int]
    height: NotRequired[int]
    background: NotRequired[str]
    owner_id: NotRequired[str]
    tenant_id: NotRequired[str]


class AddElementRequest(TypedDict):
    """Request body for POST /api/v1/moltbot/canvas/{id}/elements."""

    type: str
    x: NotRequired[float]
    y: NotRequired[float]
    width: NotRequired[float]
    height: NotRequired[float]
    properties: NotRequired[dict[str, Any]]
    layer_id: NotRequired[str]


class UpdateElementRequest(TypedDict, total=False):
    """Request body for PUT /api/v1/moltbot/canvas/{id}/elements/{eid}."""

    x: float
    y: float
    width: float
    height: float
    rotation: float
    z_index: int
    properties: dict[str, Any]


class AddCollaboratorRequest(TypedDict):
    """Request body for POST /api/v1/moltbot/canvas/{id}/collaborators."""

    user_id: str
    permission: NotRequired[str]


class CanvasSummaryResponse(TypedDict):
    """Canvas summary in list responses."""

    id: str
    name: str
    owner_id: str
    width: int
    height: int
    background: str
    created_at: str | None
    updated_at: str | None
    element_count: int
    layer_count: int


class ElementResponse(TypedDict):
    """Canvas element in responses."""

    id: str
    type: str
    x: float
    y: float
    width: float
    height: float
    rotation: float
    z_index: int
    properties: dict[str, Any]
    layer_id: str | None


class LayerResponse(TypedDict):
    """Canvas layer in responses."""

    id: str
    name: str
    visible: bool
    locked: bool


class CanvasDetailResponse(TypedDict):
    """Canvas detail in GET /api/v1/moltbot/canvas/{id}."""

    id: str
    name: str
    owner_id: str
    width: int
    height: int
    background: str
    created_at: str | None
    updated_at: str | None
    element_count: int
    layer_count: int
    elements: list[ElementResponse]
    layers: list[LayerResponse]


# =============================================================================
# Onboarding Types
# =============================================================================


class CreateFlowRequest(TypedDict):
    """Request body for POST /api/v1/moltbot/flows."""

    name: str
    description: NotRequired[str]
    is_active: NotRequired[bool]
    tenant_id: NotRequired[str]


class UpdateFlowRequest(TypedDict, total=False):
    """Request body for PUT /api/v1/moltbot/flows/{id}."""

    name: str
    description: str
    is_active: bool


class AddStepRequest(TypedDict):
    """Request body for POST /api/v1/moltbot/flows/{id}/steps."""

    type: str
    title: str
    content: NotRequired[str]
    is_required: NotRequired[bool]
    timeout_seconds: NotRequired[int]
    metadata: NotRequired[dict[str, Any]]


class StartSessionRequest(TypedDict, total=False):
    """Request body for POST /api/v1/moltbot/flows/{id}/sessions."""

    user_id: str
    device_id: str


class AdvanceSessionRequest(TypedDict, total=False):
    """Request body for POST /api/v1/moltbot/sessions/{id}/advance."""

    response: dict[str, Any]


class FlowSummaryResponse(TypedDict):
    """Flow summary in list responses."""

    id: str
    name: str
    description: str
    is_active: bool
    step_count: int
    created_at: str | None
    updated_at: str | None


class StepResponse(TypedDict):
    """Onboarding step in responses."""

    id: str
    type: str
    title: str
    content: str
    order: int
    is_required: bool
    timeout_seconds: int | None
    metadata: dict[str, Any]


class FlowDetailResponse(TypedDict):
    """Flow detail in GET /api/v1/moltbot/flows/{id}."""

    id: str
    name: str
    description: str
    is_active: bool
    step_count: int
    created_at: str | None
    updated_at: str | None
    steps: list[StepResponse]


class SessionResponse(TypedDict):
    """Onboarding session in responses."""

    id: str
    flow_id: str
    user_id: str
    device_id: str | None
    status: str
    current_step_index: int
    completed_steps: list[str]
    skipped_steps: list[str]
    responses: dict[str, Any]
    started_at: str | None
    completed_at: str | None


class SessionDetailResponse(TypedDict):
    """Session detail with current step info."""

    id: str
    flow_id: str
    user_id: str
    device_id: str | None
    status: str
    current_step_index: int
    completed_steps: list[str]
    skipped_steps: list[str]
    responses: dict[str, Any]
    started_at: str | None
    completed_at: str | None
    current_step: NotRequired[StepResponse]


# =============================================================================
# Capabilities Types
# =============================================================================


class CheckCapabilityRequest(TypedDict):
    """Request body for POST /api/v1/moltbot/devices/{id}/capabilities/check."""

    capability: str


class CapabilityResponse(TypedDict):
    """Capability in responses."""

    name: str
    category: str
    description: str
    version: str
    is_required: bool
    metadata: dict[str, Any]


class CapabilityDetailResponse(TypedDict):
    """Capability detail with dependents."""

    name: str
    category: str
    description: str
    version: str
    is_required: bool
    metadata: dict[str, Any]
    dependents: list[str]


class CapabilityGroupResponse(TypedDict, total=False):
    """Capability group (display, audio, etc.) in responses."""

    supported: bool
    details: dict[str, Any]


class DeviceCapabilitiesResponse(TypedDict):
    """Device capabilities in responses."""

    device_id: str
    device_type: str
    capabilities: list[str]
    display: dict[str, Any] | None
    audio: dict[str, Any] | None
    video: dict[str, Any] | None
    input: dict[str, Any] | None
    network: dict[str, Any] | None
    compute: dict[str, Any] | None
    sensor: dict[str, Any] | None
    actuator: dict[str, Any] | None


class CapabilityCheckResponse(TypedDict):
    """Response for capability check."""

    device_id: str
    capability: str
    supported: bool
    details: dict[str, Any] | None


class CapabilityMatrixResponse(TypedDict):
    """Response for GET /api/v1/moltbot/capabilities/matrix."""

    matrix: dict[str, Any]
    devices: list[str]
    capabilities: list[str]


class CategoryResponse(TypedDict):
    """Capability category in responses."""

    name: str
    description: str


# =============================================================================
# Generic Response Types
# =============================================================================


class SuccessResponse(TypedDict):
    """Generic success response."""

    success: bool
    message: NotRequired[str]


class DeleteResponse(TypedDict):
    """Generic delete response."""

    success: bool
    deleted: str


class StatsResponse(TypedDict, total=False):
    """Generic stats response (structure varies by endpoint)."""

    total: int
    active: int
    # Additional fields depend on specific endpoint


__all__ = [
    # Helpers
    "serialize_enum",
    "serialize_datetime",
    # Gateway
    "RegisterDeviceRequest",
    "SendCommandRequest",
    "HeartbeatRequest",
    "DeviceSummaryResponse",
    "DeviceDetailResponse",
    "DeviceListResponse",
    "HeartbeatResponse",
    # Inbox
    "RegisterChannelRequest",
    "SendMessageRequest",
    "ChannelSummaryResponse",
    "ChannelDetailResponse",
    "MessageSummaryResponse",
    "MessageDetailResponse",
    "MessageListResponse",
    # Voice
    "VoiceConfigRequest",
    "CreateVoiceSessionRequest",
    "ProcessAudioRequest",
    "SynthesizeSpeechRequest",
    "VoiceConfigResponse",
    "VoiceSessionSummaryResponse",
    "VoiceSessionDetailResponse",
    # Canvas
    "CreateCanvasRequest",
    "AddElementRequest",
    "UpdateElementRequest",
    "AddCollaboratorRequest",
    "CanvasSummaryResponse",
    "ElementResponse",
    "LayerResponse",
    "CanvasDetailResponse",
    # Onboarding
    "CreateFlowRequest",
    "UpdateFlowRequest",
    "AddStepRequest",
    "StartSessionRequest",
    "AdvanceSessionRequest",
    "FlowSummaryResponse",
    "StepResponse",
    "FlowDetailResponse",
    "SessionResponse",
    "SessionDetailResponse",
    # Capabilities
    "CheckCapabilityRequest",
    "CapabilityResponse",
    "CapabilityDetailResponse",
    "CapabilityGroupResponse",
    "DeviceCapabilitiesResponse",
    "CapabilityCheckResponse",
    "CapabilityMatrixResponse",
    "CategoryResponse",
    # Generic
    "SuccessResponse",
    "DeleteResponse",
    "StatsResponse",
]
