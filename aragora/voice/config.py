"""
Voice configuration for wake word detection.

Provides configuration dataclasses for:
- Wake phrases and sensitivity
- Audio device selection
- Detection thresholds
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class DetectionBackend(Enum):
    """Wake word detection backend."""

    KEYWORD = "keyword"  # Simple keyword spotting (default, no external deps)
    PORCUPINE = "porcupine"  # Picovoice Porcupine (requires license)
    VOSK = "vosk"  # Vosk offline recognition


@dataclass
class AudioDevice:
    """Audio input device configuration."""

    device_id: int | str | None = None  # None = system default
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024  # Samples per chunk
    format: str = "int16"

    @classmethod
    def from_env(cls) -> AudioDevice:
        """Create from environment variables."""
        return cls(
            device_id=os.getenv("ARAGORA_VOICE_DEVICE_ID"),
            sample_rate=int(os.getenv("ARAGORA_VOICE_SAMPLE_RATE", "16000")),
            channels=int(os.getenv("ARAGORA_VOICE_CHANNELS", "1")),
            chunk_size=int(os.getenv("ARAGORA_VOICE_CHUNK_SIZE", "1024")),
        )


@dataclass
class VoiceConfig:
    """Configuration for voice wake detection."""

    # Wake phrases (case-insensitive)
    wake_phrases: list[str] = field(
        default_factory=lambda: ["hey aragora", "ok aragora", "aragora"]
    )

    # Detection sensitivity (0.0 = strict, 1.0 = loose)
    sensitivity: float = 0.5

    # Detection backend
    backend: DetectionBackend = DetectionBackend.KEYWORD

    # Audio device configuration
    audio_device: AudioDevice = field(default_factory=AudioDevice)

    # Minimum confidence for keyword detection (0.0 - 1.0)
    min_confidence: float = 0.6

    # Cooldown between wake detections (seconds)
    cooldown_seconds: float = 2.0

    # Maximum recording duration after wake (seconds)
    max_listen_seconds: float = 30.0

    # Silence threshold for end-of-speech detection (seconds)
    silence_threshold_seconds: float = 1.5

    # Enable debug logging
    debug: bool = False

    # Porcupine-specific settings
    porcupine_access_key: str | None = None
    porcupine_model_path: str | None = None

    # Vosk-specific settings
    vosk_model_path: str | None = None

    @classmethod
    def from_env(cls) -> VoiceConfig:
        """Create configuration from environment variables."""
        wake_phrases_str = os.getenv("ARAGORA_WAKE_PHRASES", "hey aragora,ok aragora")
        wake_phrases = [p.strip() for p in wake_phrases_str.split(",") if p.strip()]

        backend_str = os.getenv("ARAGORA_WAKE_BACKEND", "keyword").lower()
        try:
            backend = DetectionBackend(backend_str)
        except ValueError:
            backend = DetectionBackend.KEYWORD

        return cls(
            wake_phrases=wake_phrases,
            sensitivity=float(os.getenv("ARAGORA_WAKE_SENSITIVITY", "0.5")),
            backend=backend,
            audio_device=AudioDevice.from_env(),
            min_confidence=float(os.getenv("ARAGORA_WAKE_MIN_CONFIDENCE", "0.6")),
            cooldown_seconds=float(os.getenv("ARAGORA_WAKE_COOLDOWN", "2.0")),
            max_listen_seconds=float(os.getenv("ARAGORA_WAKE_MAX_LISTEN", "30.0")),
            silence_threshold_seconds=float(os.getenv("ARAGORA_WAKE_SILENCE_THRESHOLD", "1.5")),
            debug=os.getenv("ARAGORA_WAKE_DEBUG", "").lower() in ("1", "true", "yes"),
            porcupine_access_key=os.getenv("PICOVOICE_ACCESS_KEY"),
            porcupine_model_path=os.getenv("PORCUPINE_MODEL_PATH"),
            vosk_model_path=os.getenv("VOSK_MODEL_PATH"),
        )

    def validate(self) -> list[str]:
        """Validate configuration, return list of errors."""
        errors = []

        if not self.wake_phrases:
            errors.append("At least one wake phrase is required")

        if not 0.0 <= self.sensitivity <= 1.0:
            errors.append("Sensitivity must be between 0.0 and 1.0")

        if not 0.0 <= self.min_confidence <= 1.0:
            errors.append("min_confidence must be between 0.0 and 1.0")

        if self.cooldown_seconds < 0:
            errors.append("cooldown_seconds must be non-negative")

        if self.backend == DetectionBackend.PORCUPINE and not self.porcupine_access_key:
            errors.append("Porcupine backend requires PICOVOICE_ACCESS_KEY")

        if self.backend == DetectionBackend.VOSK and not self.vosk_model_path:
            errors.append("Vosk backend requires VOSK_MODEL_PATH")

        return errors


# Type alias for wake callback
WakeCallback = Callable[[str, float], Any]  # (phrase, confidence) -> Any
