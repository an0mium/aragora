"""
Moltbot Extension - Consumer/Device Interface Layer.

Provides multi-channel consumer engagement:
- Gateway: Local/edge orchestration for device networks
- Inbox: Unified multi-channel message aggregation
- Voice: Speech-to-text and TTS integration
- Canvas: Rich media rendering and interaction
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
]
