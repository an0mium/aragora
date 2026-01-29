"""
Voice Wake Word Detection and Processing.

Provides wake word detection, voice activity detection (VAD),
and voice command processing for hands-free device interaction.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class WakeWordEngine(Enum):
    """Supported wake word detection engines."""

    PORCUPINE = "porcupine"  # Picovoice Porcupine
    SNOWBOY = "snowboy"  # Snowboy (deprecated but still used)
    VOSK = "vosk"  # Vosk offline
    CUSTOM = "custom"  # Custom trained model


class VoiceActivityState(Enum):
    """Voice activity detection states."""

    IDLE = "idle"
    LISTENING = "listening"
    DETECTING = "detecting"
    PROCESSING = "processing"
    RESPONDING = "responding"


@dataclass
class WakeWordConfig:
    """Configuration for wake word detection."""

    wake_words: list[str] = field(default_factory=lambda: ["hey aragora", "aragora"])
    engine: WakeWordEngine = WakeWordEngine.PORCUPINE
    sensitivity: float = 0.5  # 0.0 to 1.0
    audio_gain: float = 1.0
    sample_rate: int = 16000
    frame_length: int = 512
    timeout_seconds: float = 10.0  # Max listening time after wake
    vad_aggressiveness: int = 2  # 0-3 (0 = least aggressive)
    enable_noise_suppression: bool = True
    enable_echo_cancellation: bool = True
    model_path: str | None = None  # Custom model path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WakeWordEvent:
    """A detected wake word event."""

    id: str
    wake_word: str
    confidence: float
    timestamp: datetime
    device_id: str
    audio_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceCommand:
    """A processed voice command."""

    id: str
    transcript: str
    confidence: float
    intent: str | None = None
    entities: dict[str, Any] = field(default_factory=dict)
    wake_event_id: str | None = None
    device_id: str = ""
    user_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceSession:
    """An active voice interaction session."""

    id: str
    device_id: str
    state: VoiceActivityState
    started_at: datetime
    wake_event: WakeWordEvent | None = None
    commands: list[str] = field(default_factory=list)  # Command IDs
    audio_buffer: bytes = b""
    last_activity: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class VoiceWakeManager:
    """
    Manages wake word detection and voice command processing.

    Provides continuous listening for wake words, voice activity detection,
    and command processing with support for multiple devices.
    """

    def __init__(
        self,
        config: WakeWordConfig | None = None,
        storage_path: str | Path | None = None,
    ) -> None:
        """
        Initialize the voice wake manager.

        Args:
            config: Wake word configuration
            storage_path: Path for session storage
        """
        self._config = config or WakeWordConfig()
        self._storage_path = Path(storage_path) if storage_path else None

        self._sessions: dict[str, VoiceSession] = {}
        self._wake_events: dict[str, WakeWordEvent] = {}
        self._commands: dict[str, VoiceCommand] = {}
        self._device_states: dict[str, VoiceActivityState] = {}

        # Event callbacks
        self._wake_callbacks: list[Callable] = []
        self._command_callbacks: list[Callable] = []
        self._state_callbacks: list[Callable] = []

        self._lock = asyncio.Lock()
        self._running = False
        self._listener_tasks: dict[str, asyncio.Task] = {}

        if self._storage_path:
            self._storage_path.mkdir(parents=True, exist_ok=True)

    # ========== Lifecycle ==========

    async def start(self) -> None:
        """Start the voice wake manager."""
        if self._running:
            return

        self._running = True
        logger.info(f"VoiceWakeManager started with wake words: {self._config.wake_words}")

    async def stop(self) -> None:
        """Stop the voice wake manager and all listeners."""
        self._running = False

        # Cancel all listener tasks
        for task in self._listener_tasks.values():
            task.cancel()

        self._listener_tasks.clear()
        logger.info("VoiceWakeManager stopped")

    # ========== Device Listening ==========

    async def start_listening(self, device_id: str) -> bool:
        """
        Start listening for wake words on a device.

        Args:
            device_id: Device to listen on

        Returns:
            True if listening started
        """
        if device_id in self._listener_tasks:
            return True  # Already listening

        async with self._lock:
            self._device_states[device_id] = VoiceActivityState.LISTENING

            # Create listener task
            task = asyncio.create_task(self._listen_loop(device_id))
            self._listener_tasks[device_id] = task

            await self._notify_state_change(device_id, VoiceActivityState.LISTENING)
            logger.info(f"Started listening on device {device_id}")
            return True

    async def stop_listening(self, device_id: str) -> bool:
        """Stop listening on a device."""
        task = self._listener_tasks.pop(device_id, None)
        if task:
            task.cancel()
            self._device_states[device_id] = VoiceActivityState.IDLE
            await self._notify_state_change(device_id, VoiceActivityState.IDLE)
            logger.info(f"Stopped listening on device {device_id}")
            return True
        return False

    async def _listen_loop(self, device_id: str) -> None:
        """Internal listening loop for a device."""
        while self._running and device_id in self._listener_tasks:
            try:
                # In a real implementation, this would:
                # 1. Read audio from device
                # 2. Run wake word detection
                # 3. Trigger events on detection

                # Simulate listening with periodic checks
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in listen loop for {device_id}: {e}")
                await asyncio.sleep(1.0)

    # ========== Wake Word Detection ==========

    async def process_audio_frame(
        self,
        device_id: str,
        audio_data: bytes,
    ) -> WakeWordEvent | None:
        """
        Process an audio frame for wake word detection.

        Args:
            device_id: Source device
            audio_data: Audio frame data

        Returns:
            WakeWordEvent if wake word detected
        """
        async with self._lock:
            # In a real implementation, this would run the audio through
            # the wake word detection engine

            # Placeholder: simulate detection based on audio length
            # (real implementation would use Porcupine/Vosk/etc.)

            state = self._device_states.get(device_id, VoiceActivityState.IDLE)
            if state != VoiceActivityState.LISTENING:
                return None

            # Simulated detection (would be real ML inference)
            detected = False  # Placeholder

            if detected:
                event = await self._create_wake_event(device_id)
                return event

            return None

    async def simulate_wake_detection(
        self,
        device_id: str,
        wake_word: str | None = None,
        confidence: float = 0.95,
    ) -> WakeWordEvent:
        """
        Simulate a wake word detection (for testing).

        Args:
            device_id: Device that detected wake word
            wake_word: Detected wake word
            confidence: Detection confidence

        Returns:
            Created wake event
        """
        word = wake_word or self._config.wake_words[0]
        return await self._create_wake_event(device_id, word, confidence)

    async def _create_wake_event(
        self,
        device_id: str,
        wake_word: str | None = None,
        confidence: float = 0.95,
    ) -> WakeWordEvent:
        """Create a wake word event and start a session."""
        event_id = str(uuid.uuid4())
        word = wake_word or self._config.wake_words[0]

        event = WakeWordEvent(
            id=event_id,
            wake_word=word,
            confidence=confidence,
            timestamp=datetime.utcnow(),
            device_id=device_id,
        )

        self._wake_events[event_id] = event

        # Start a voice session
        await self._start_session(device_id, event)

        # Notify callbacks
        await self._notify_wake_detected(event)

        logger.info(f"Wake word '{word}' detected on {device_id} (conf: {confidence})")
        return event

    # ========== Session Management ==========

    async def _start_session(
        self,
        device_id: str,
        wake_event: WakeWordEvent,
    ) -> VoiceSession:
        """Start a voice session after wake detection."""
        session_id = str(uuid.uuid4())

        session = VoiceSession(
            id=session_id,
            device_id=device_id,
            state=VoiceActivityState.DETECTING,
            started_at=datetime.utcnow(),
            wake_event=wake_event,
        )

        self._sessions[session_id] = session
        self._device_states[device_id] = VoiceActivityState.DETECTING

        await self._notify_state_change(device_id, VoiceActivityState.DETECTING)

        # Set timeout for command detection
        asyncio.create_task(self._session_timeout(session_id))

        return session

    async def _session_timeout(self, session_id: str) -> None:
        """Handle session timeout."""
        await asyncio.sleep(self._config.timeout_seconds)

        session = self._sessions.get(session_id)
        if session and session.state in (
            VoiceActivityState.DETECTING,
            VoiceActivityState.PROCESSING,
        ):
            await self.end_session(session_id, reason="timeout")

    async def end_session(
        self,
        session_id: str,
        reason: str = "completed",
    ) -> bool:
        """
        End a voice session.

        Args:
            session_id: Session to end
            reason: Reason for ending

        Returns:
            True if session ended
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            self._device_states[session.device_id] = VoiceActivityState.LISTENING

            await self._notify_state_change(session.device_id, VoiceActivityState.LISTENING)

            logger.debug(f"Session {session_id} ended: {reason}")
            return True

    async def get_session(self, session_id: str) -> VoiceSession | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    async def get_device_session(self, device_id: str) -> VoiceSession | None:
        """Get the active session for a device."""
        for session in self._sessions.values():
            if session.device_id == device_id and session.state != VoiceActivityState.IDLE:
                return session
        return None

    # ========== Command Processing ==========

    async def process_command(
        self,
        session_id: str,
        transcript: str,
        confidence: float = 1.0,
    ) -> VoiceCommand:
        """
        Process a voice command transcript.

        Args:
            session_id: Active session
            transcript: Command transcript
            confidence: Transcription confidence

        Returns:
            Processed command
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        async with self._lock:
            session.state = VoiceActivityState.PROCESSING
            self._device_states[session.device_id] = VoiceActivityState.PROCESSING
            await self._notify_state_change(session.device_id, VoiceActivityState.PROCESSING)

        start_time = datetime.utcnow()

        # Parse intent and entities
        intent, entities = await self._parse_command(transcript)

        command = VoiceCommand(
            id=str(uuid.uuid4()),
            transcript=transcript,
            confidence=confidence,
            intent=intent,
            entities=entities,
            wake_event_id=session.wake_event.id if session.wake_event else None,
            device_id=session.device_id,
            processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
        )

        self._commands[command.id] = command
        session.commands.append(command.id)
        session.last_activity = datetime.utcnow()

        await self._notify_command_processed(command)

        logger.info(f"Processed command: '{transcript}' -> intent={intent}")
        return command

    async def _parse_command(
        self,
        transcript: str,
    ) -> tuple[str | None, dict[str, Any]]:
        """
        Parse a command transcript for intent and entities.

        Args:
            transcript: Command transcript

        Returns:
            Tuple of (intent, entities)
        """
        # Simple rule-based parsing (would use NLU in production)
        transcript_lower = transcript.lower()

        # Common intents - check stop/pause before play to handle "stop the music"
        if any(w in transcript_lower for w in ["stop", "pause"]):
            return "media.pause", {}
        if any(w in transcript_lower for w in ["play", "music", "song"]):
            return "media.play", {"type": "music"}
        if any(w in transcript_lower for w in ["weather", "forecast"]):
            return "query.weather", {}
        if any(w in transcript_lower for w in ["time", "clock"]):
            return "query.time", {}
        if any(w in transcript_lower for w in ["light", "lights"]):
            action = "on" if "on" in transcript_lower else "off"
            return "home.lights", {"action": action}
        if any(w in transcript_lower for w in ["remind", "reminder"]):
            return "reminder.set", {}
        if any(w in transcript_lower for w in ["search", "find", "look up"]):
            return "search.web", {"query": transcript}
        if any(w in transcript_lower for w in ["debate", "discuss", "analyze"]):
            return "aragora.debate", {"topic": transcript}

        return None, {}

    async def get_command(self, command_id: str) -> VoiceCommand | None:
        """Get a command by ID."""
        return self._commands.get(command_id)

    async def list_commands(
        self,
        device_id: str | None = None,
        limit: int = 100,
    ) -> list[VoiceCommand]:
        """List commands with optional device filter."""
        commands = list(self._commands.values())

        if device_id:
            commands = [c for c in commands if c.device_id == device_id]

        commands.sort(key=lambda c: c.timestamp, reverse=True)
        return commands[:limit]

    # ========== Event Callbacks ==========

    def on_wake_detected(self, callback: Callable) -> None:
        """Register callback for wake word detection."""
        self._wake_callbacks.append(callback)

    def on_command_processed(self, callback: Callable) -> None:
        """Register callback for command processing."""
        self._command_callbacks.append(callback)

    def on_state_change(self, callback: Callable) -> None:
        """Register callback for state changes."""
        self._state_callbacks.append(callback)

    async def _notify_wake_detected(self, event: WakeWordEvent) -> None:
        """Notify wake detection callbacks."""
        for callback in self._wake_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Wake callback error: {e}")

    async def _notify_command_processed(self, command: VoiceCommand) -> None:
        """Notify command processed callbacks."""
        for callback in self._command_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(command)
                else:
                    callback(command)
            except Exception as e:
                logger.error(f"Command callback error: {e}")

    async def _notify_state_change(
        self,
        device_id: str,
        state: VoiceActivityState,
    ) -> None:
        """Notify state change callbacks."""
        for callback in self._state_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(device_id, state)
                else:
                    callback(device_id, state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    # ========== Device State ==========

    def get_device_state(self, device_id: str) -> VoiceActivityState:
        """Get the current state of a device."""
        return self._device_states.get(device_id, VoiceActivityState.IDLE)

    async def list_active_devices(self) -> list[str]:
        """List devices that are actively listening."""
        return [
            device_id
            for device_id, state in self._device_states.items()
            if state != VoiceActivityState.IDLE
        ]

    # ========== Statistics ==========

    async def get_stats(self) -> dict[str, Any]:
        """Get voice wake manager statistics."""
        async with self._lock:
            commands_by_intent: dict[str, int] = {}
            for cmd in self._commands.values():
                intent = cmd.intent or "unknown"
                commands_by_intent[intent] = commands_by_intent.get(intent, 0) + 1

            states_count = {}
            for state in self._device_states.values():
                states_count[state.value] = states_count.get(state.value, 0) + 1

            return {
                "running": self._running,
                "wake_words": self._config.wake_words,
                "engine": self._config.engine.value,
                "active_listeners": len(self._listener_tasks),
                "active_sessions": len(
                    [s for s in self._sessions.values() if s.state != VoiceActivityState.IDLE]
                ),
                "total_wake_events": len(self._wake_events),
                "total_commands": len(self._commands),
                "commands_by_intent": commands_by_intent,
                "device_states": states_count,
            }
