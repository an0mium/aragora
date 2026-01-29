"""
Wake word detector for hands-free voice activation.

Provides always-on listening with configurable wake phrases.
Supports multiple backends: simple keyword spotting, Porcupine, Vosk.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aragora.voice.config import DetectionBackend, VoiceConfig, WakeCallback

logger = logging.getLogger(__name__)


class DetectorStatus(Enum):
    """Wake word detector status."""

    STOPPED = "stopped"
    STARTING = "starting"
    LISTENING = "listening"
    PROCESSING = "processing"
    COOLDOWN = "cooldown"
    ERROR = "error"


@dataclass
class WakeEvent:
    """Event emitted when wake word is detected."""

    phrase: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    audio_data: bytes | None = None


class WakeWordDetector:
    """
    Always-on wake word detector.

    Listens for configurable wake phrases and triggers callbacks when detected.
    Supports multiple detection backends with graceful fallback.

    Usage:
        config = VoiceConfig(wake_phrases=["hey aragora"])
        detector = WakeWordDetector(config)

        async def on_wake(phrase: str, confidence: float):
            print(f"Detected: {phrase} ({confidence:.2f})")

        await detector.start(on_wake)
        # ... runs until stopped
        await detector.stop()
    """

    def __init__(self, config: VoiceConfig | None = None):
        """Initialize wake word detector.

        Args:
            config: Voice configuration. If None, loads from environment.
        """
        self.config = config or VoiceConfig.from_env()
        self._status = DetectorStatus.STOPPED
        self._callback: WakeCallback | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._last_wake_time: float = 0.0
        self._lock = asyncio.Lock()

        # Compile wake phrase patterns for keyword matching
        self._wake_patterns = [
            re.compile(re.escape(phrase.lower()), re.IGNORECASE)
            for phrase in self.config.wake_phrases
        ]

        # Backend-specific initialization
        self._backend_instance: Any = None

    @property
    def status(self) -> DetectorStatus:
        """Current detector status."""
        return self._status

    @property
    def is_running(self) -> bool:
        """Whether the detector is actively listening."""
        return self._status in (
            DetectorStatus.LISTENING,
            DetectorStatus.PROCESSING,
            DetectorStatus.COOLDOWN,
        )

    async def start(self, on_wake_callback: WakeCallback) -> None:
        """Start listening for wake words.

        Args:
            on_wake_callback: Called when wake word detected. Receives (phrase, confidence).
        """
        if self.is_running:
            logger.warning("Wake detector already running")
            return

        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid voice config: {'; '.join(errors)}")

        self._callback = on_wake_callback
        self._stop_event.clear()
        self._status = DetectorStatus.STARTING

        try:
            # Initialize backend
            await self._init_backend()

            # Start listening loop
            self._task = asyncio.create_task(self._listen_loop())
            self._status = DetectorStatus.LISTENING

            logger.info(
                f"Wake detector started with {len(self.config.wake_phrases)} phrase(s), "
                f"backend={self.config.backend.value}"
            )

        except Exception as e:
            self._status = DetectorStatus.ERROR
            logger.error(f"Failed to start wake detector: {e}")
            raise

    async def stop(self) -> None:
        """Stop listening for wake words."""
        if not self.is_running:
            return

        logger.info("Stopping wake detector")
        self._stop_event.set()

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        await self._cleanup_backend()
        self._status = DetectorStatus.STOPPED

    async def _init_backend(self) -> None:
        """Initialize the detection backend."""
        backend = self.config.backend

        if backend == DetectionBackend.KEYWORD:
            # Simple keyword spotting - no external deps
            logger.debug("Using keyword spotting backend")

        elif backend == DetectionBackend.PORCUPINE:
            await self._init_porcupine()

        elif backend == DetectionBackend.VOSK:
            await self._init_vosk()

    async def _init_porcupine(self) -> None:
        """Initialize Porcupine wake word engine."""
        try:
            import pvporcupine

            self._backend_instance = pvporcupine.create(
                access_key=self.config.porcupine_access_key,
                keywords=self.config.wake_phrases,
                sensitivities=[self.config.sensitivity] * len(self.config.wake_phrases),
            )
            logger.info("Porcupine backend initialized")
        except ImportError:
            logger.warning("pvporcupine not installed, falling back to keyword backend")
            self.config.backend = DetectionBackend.KEYWORD
        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {e}")
            self.config.backend = DetectionBackend.KEYWORD

    async def _init_vosk(self) -> None:
        """Initialize Vosk speech recognition."""
        try:
            from vosk import Model, KaldiRecognizer

            model = Model(self.config.vosk_model_path)
            self._backend_instance = KaldiRecognizer(model, self.config.audio_device.sample_rate)
            logger.info("Vosk backend initialized")
        except ImportError:
            logger.warning("vosk not installed, falling back to keyword backend")
            self.config.backend = DetectionBackend.KEYWORD
        except Exception as e:
            logger.error(f"Failed to initialize Vosk: {e}")
            self.config.backend = DetectionBackend.KEYWORD

    async def _cleanup_backend(self) -> None:
        """Clean up backend resources."""
        if self._backend_instance is not None:
            if hasattr(self._backend_instance, "delete"):
                self._backend_instance.delete()
            self._backend_instance = None

    async def _listen_loop(self) -> None:
        """Main listening loop."""
        while not self._stop_event.is_set():
            try:
                # Simulate audio input for testing
                # In production, this would read from actual audio device
                await asyncio.sleep(0.1)

                # Check for wake word (mock for now)
                # Real implementation would process audio chunks
                if self._status == DetectorStatus.COOLDOWN:
                    if time.time() - self._last_wake_time >= self.config.cooldown_seconds:
                        self._status = DetectorStatus.LISTENING

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in listen loop: {e}")
                self._status = DetectorStatus.ERROR
                await asyncio.sleep(1.0)  # Brief pause before retry

    async def process_audio_chunk(self, audio_data: bytes) -> WakeEvent | None:
        """Process an audio chunk and check for wake word.

        This method can be called externally to feed audio data.

        Args:
            audio_data: Raw audio bytes (PCM 16-bit, mono)

        Returns:
            WakeEvent if wake word detected, None otherwise
        """
        if not self.is_running:
            return None

        # Check cooldown
        if self._status == DetectorStatus.COOLDOWN:
            return None

        async with self._lock:
            self._status = DetectorStatus.PROCESSING

            try:
                result = await self._detect_wake_word(audio_data)

                if result:
                    phrase, confidence = result

                    # Check confidence threshold
                    if confidence >= self.config.min_confidence:
                        self._last_wake_time = time.time()
                        self._status = DetectorStatus.COOLDOWN

                        event = WakeEvent(
                            phrase=phrase,
                            confidence=confidence,
                            audio_data=audio_data,
                        )

                        # Invoke callback
                        if self._callback:
                            try:
                                result = self._callback(phrase, confidence)
                                if asyncio.iscoroutine(result):
                                    await result
                            except Exception as e:
                                logger.error(f"Wake callback error: {e}")

                        return event

            finally:
                if self._status == DetectorStatus.PROCESSING:
                    self._status = DetectorStatus.LISTENING

        return None

    async def _detect_wake_word(self, audio_data: bytes) -> tuple[str, float] | None:
        """Detect wake word in audio data.

        Args:
            audio_data: Raw audio bytes

        Returns:
            (phrase, confidence) tuple if detected, None otherwise
        """
        backend = self.config.backend

        if backend == DetectionBackend.KEYWORD:
            return await self._keyword_detect(audio_data)
        elif backend == DetectionBackend.PORCUPINE:
            return await self._porcupine_detect(audio_data)
        elif backend == DetectionBackend.VOSK:
            return await self._vosk_detect(audio_data)

        return None

    async def _keyword_detect(self, audio_data: bytes) -> tuple[str, float] | None:
        """Simple keyword spotting (placeholder for real implementation).

        In a real implementation, this would use a lightweight local STT
        or pattern matching on audio features.
        """
        # This is a placeholder - real implementation would:
        # 1. Convert audio to text using lightweight local STT
        # 2. Match against wake phrases
        # For now, return None (no detection)
        return None

    async def _porcupine_detect(self, audio_data: bytes) -> tuple[str, float] | None:
        """Porcupine wake word detection."""
        if self._backend_instance is None:
            return None

        try:
            import struct

            # Convert bytes to int16 array
            pcm = struct.unpack_from(f"{len(audio_data) // 2}h", audio_data)

            # Process frame
            keyword_index = self._backend_instance.process(pcm)

            if keyword_index >= 0:
                phrase = self.config.wake_phrases[keyword_index]
                # Porcupine doesn't provide confidence, use sensitivity as proxy
                confidence = 1.0 - self.config.sensitivity * 0.5
                return (phrase, confidence)

        except Exception as e:
            logger.error(f"Porcupine detection error: {e}")

        return None

    async def _vosk_detect(self, audio_data: bytes) -> tuple[str, float] | None:
        """Vosk speech recognition for wake word detection."""
        if self._backend_instance is None:
            return None

        try:
            import json

            if self._backend_instance.AcceptWaveform(audio_data):
                result = json.loads(self._backend_instance.Result())
                text = result.get("text", "").lower()

                # Check if any wake phrase is in the recognized text
                for i, pattern in enumerate(self._wake_patterns):
                    if pattern.search(text):
                        phrase = self.config.wake_phrases[i]
                        # Vosk provides confidence in result
                        confidence = result.get("confidence", 0.8)
                        return (phrase, confidence)

        except Exception as e:
            logger.error(f"Vosk detection error: {e}")

        return None

    def simulate_wake(self, phrase: str | None = None, confidence: float = 0.95) -> None:
        """Simulate wake word detection for testing.

        Args:
            phrase: Wake phrase to simulate (uses first configured if None)
            confidence: Confidence score
        """
        if not self._callback:
            return

        phrase = phrase or (self.config.wake_phrases[0] if self.config.wake_phrases else "aragora")

        logger.info(f"Simulating wake word: {phrase} ({confidence:.2f})")

        # Call callback with error handling
        try:
            result = self._callback(phrase, confidence)
            if asyncio.iscoroutine(result):
                # Schedule coroutine
                asyncio.create_task(result)
        except Exception as e:
            logger.error(f"Wake callback error in simulate_wake: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get detector statistics."""
        return {
            "status": self._status.value,
            "backend": self.config.backend.value,
            "wake_phrases": self.config.wake_phrases,
            "sensitivity": self.config.sensitivity,
            "last_wake_time": self._last_wake_time,
            "is_running": self.is_running,
        }
