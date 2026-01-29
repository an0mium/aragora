"""
Moltbot Extension - Consumer/Device Interface Layer.

Provides multi-channel consumer engagement:
- Gateway: Local/edge orchestration for device networks
- Inbox: Unified multi-channel message aggregation
- Voice: Speech-to-text and TTS integration
- Canvas: Rich media rendering and interaction
- Voice Wake: Wake word detection and voice commands
- Capabilities: Device capability models and matching
- Onboarding: User journey and activation flows

Inspired by moltbot/moltbot, adapted for Aragora's enterprise semantics.
"""

from .models import (
    Channel,
    ChannelConfig,
    ChannelType,
    DeviceNode,
    DeviceNodeConfig,
    InboxMessage,
    InboxMessageStatus,
    OnboardingFlow,
    OnboardingStep,
    VoiceSession,
    VoiceSessionConfig,
)
from .inbox import InboxManager
from .gateway import LocalGateway
from .voice import VoiceProcessor
from .onboarding import OnboardingOrchestrator
from .canvas import (
    Canvas,
    CanvasConfig,
    CanvasElement,
    CanvasLayer,
    CanvasManager,
    ElementType,
)
from .voice_wake import (
    VoiceActivityState,
    VoiceCommand,
    VoiceSession as VoiceWakeSession,
    VoiceWakeManager,
    WakeWordConfig,
    WakeWordEngine,
    WakeWordEvent,
)
from .capabilities import (
    ActuatorCapability,
    AudioCapability,
    CapabilityCategory,
    CapabilityMatcher,
    CapabilityRequirement,
    CapabilitySpec,
    ComputeCapability,
    DeviceCapabilities,
    DisplayCapability,
    InputCapability,
    NetworkCapability,
    SensorCapability,
    VideoCapability,
    edge_compute_capabilities,
    iot_hub_capabilities,
    mobile_app_capabilities,
    smart_display_capabilities,
    smart_speaker_capabilities,
)
from .adapter import MoltbotGatewayAdapter

# HTTP Handlers - import separately to avoid circular imports
from .handlers import (
    MoltbotCanvasHandler,
    MoltbotCapabilitiesHandler,
    MoltbotGatewayHandler,
    MoltbotInboxHandler,
    MoltbotOnboardingHandler,
    MoltbotVoiceHandler,
    get_all_handlers,
)

__all__ = [
    # Models
    "Channel",
    "ChannelConfig",
    "ChannelType",
    "DeviceNode",
    "DeviceNodeConfig",
    "InboxMessage",
    "InboxMessageStatus",
    "OnboardingFlow",
    "OnboardingStep",
    "VoiceSession",
    "VoiceSessionConfig",
    # Managers
    "InboxManager",
    "LocalGateway",
    "VoiceProcessor",
    "OnboardingOrchestrator",
    # Canvas
    "Canvas",
    "CanvasConfig",
    "CanvasElement",
    "CanvasLayer",
    "CanvasManager",
    "ElementType",
    # Voice Wake
    "VoiceActivityState",
    "VoiceCommand",
    "VoiceWakeManager",
    "VoiceWakeSession",
    "WakeWordConfig",
    "WakeWordEngine",
    "WakeWordEvent",
    # Capabilities
    "ActuatorCapability",
    "AudioCapability",
    "CapabilityCategory",
    "CapabilityMatcher",
    "CapabilityRequirement",
    "CapabilitySpec",
    "ComputeCapability",
    "DeviceCapabilities",
    "DisplayCapability",
    "InputCapability",
    "NetworkCapability",
    "SensorCapability",
    "VideoCapability",
    "edge_compute_capabilities",
    "iot_hub_capabilities",
    "mobile_app_capabilities",
    "smart_display_capabilities",
    "smart_speaker_capabilities",
    # HTTP Handlers
    "MoltbotCanvasHandler",
    "MoltbotCapabilitiesHandler",
    "MoltbotGatewayHandler",
    "MoltbotInboxHandler",
    "MoltbotOnboardingHandler",
    "MoltbotVoiceHandler",
    "get_all_handlers",
    "MoltbotGatewayAdapter",
]
