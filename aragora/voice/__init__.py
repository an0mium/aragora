"""
Voice Wake Module - On-device always-listening voice trigger.

Provides hands-free activation for Aragora via configurable wake words.
Integrates with the existing voice streaming infrastructure for STT/TTS.

Key concepts:
- WakeWordDetector: Listens for configurable wake phrases
- VoiceConfig: Configuration for wake words, sensitivity, audio devices
- VoiceSession: Active voice session after wake word detected

Usage:
    from aragora.voice import WakeWordDetector, VoiceConfig

    config = VoiceConfig(wake_phrases=["hey aragora", "ok aragora"])
    detector = WakeWordDetector(config)

    async def on_wake(phrase: str):
        print(f"Wake word detected: {phrase}")
        # Start voice streaming session

    await detector.start(on_wake_callback=on_wake)
"""

from aragora.voice.config import VoiceConfig, AudioDevice
from aragora.voice.wake import (
    WakeWordDetector,
    WakeEvent,
    DetectorStatus,
)

__all__ = [
    "VoiceConfig",
    "AudioDevice",
    "WakeWordDetector",
    "WakeEvent",
    "DetectorStatus",
]
