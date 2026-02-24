"""
TTS Event Bridge for Live Debate Streaming.

Subscribes to ``agent_message`` events from the debate :class:`EventBus` and
automatically queues them for TTS synthesis when a voice session is active.
Text is accumulated until a sentence boundary is detected, then each complete
sentence is synthesized via :class:`TTSIntegration` and dispatched through the
:class:`VoiceStreamHandler`.

Usage::

    from aragora.server.stream.tts_event_bridge import TTSEventBridge

    bridge = TTSEventBridge(tts_integration, voice_handler)
    bridge.connect(event_bus)

    # When done:
    await bridge.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.debate.event_bus import DebateEvent, EventBus
    from aragora.server.stream.tts_integration import TTSIntegration
    from aragora.server.stream.voice_stream import VoiceStreamHandler

logger = logging.getLogger(__name__)


def _is_oracle_voice_enabled() -> bool:
    """Check whether the ``enable_oracle_voice`` feature flag is active."""
    try:
        from aragora.config.feature_flags import is_enabled

        return is_enabled("enable_oracle_voice")
    except ImportError:
        return False


# Sentence-ending punctuation pattern.
# Matches '.', '!', '?' optionally followed by a closing quote or paren,
# then whitespace or end-of-string.  This avoids splitting on abbreviations
# like "e.g." or "Dr." in most common cases.
_SENTENCE_END_RE = re.compile(r'[.!?]["\')]*(?:\s|$)')

# Maximum time (seconds) to hold a partial sentence before flushing anyway.
_PARTIAL_FLUSH_TIMEOUT = 5.0

# Maximum characters to accumulate before forcing a flush.
_MAX_ACCUMULATOR_LEN = 2000


class AudioPlaybackState(Enum):
    """Playback state for a voice session's TTS audio."""

    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"


@dataclass
class _SessionAudioState:
    """Tracks audio state for a single voice session."""

    current_agent: str = ""
    playback_state: AudioPlaybackState = AudioPlaybackState.IDLE
    queued_frames: int = 0
    total_frames_sent: int = 0


@dataclass
class TTSEventBridge:
    """
    Bridges the debate :class:`EventBus` to TTS synthesis and voice streaming.

    When connected, the bridge subscribes to ``agent_message`` events.  Incoming
    text tokens are accumulated in a per-debate buffer until a sentence boundary
    is detected.  Complete sentences are enqueued for asynchronous TTS synthesis
    and the resulting audio chunks are dispatched via the voice stream handler.

    Attributes:
        tts: The TTSIntegration instance used for synthesis.
        voice_handler: The VoiceStreamHandler that manages voice sessions.
    """

    tts: TTSIntegration
    voice_handler: VoiceStreamHandler
    _text_buffers: dict[str, str] = field(default_factory=dict)
    _buffer_timestamps: dict[str, float] = field(default_factory=dict)
    _queues: dict[str, asyncio.Queue[tuple[str, str] | None]] = field(
        default_factory=dict
    )
    _workers: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    _audio_states: dict[str, _SessionAudioState] = field(default_factory=dict)
    _connected: bool = False
    _event_bus: EventBus | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self, event_bus: EventBus) -> None:
        """Subscribe to the event bus and start processing.

        Subscribes to ``agent_message`` events.  When the
        ``enable_oracle_voice`` feature flag is active, also subscribes
        to ``critique`` events so Oracle debates can synthesize critiques
        as voice output.

        Args:
            event_bus: The debate EventBus to subscribe to.
        """
        if self._connected:
            logger.warning("[TTS Bridge] Already connected to event bus")
            return
        event_bus.subscribe("agent_message", self._on_agent_message)

        if _is_oracle_voice_enabled():
            event_bus.subscribe("critique", self._on_agent_message)
            logger.info("[TTS Bridge] Oracle voice enabled — subscribed to critique events")

        self._event_bus = event_bus
        self._connected = True
        logger.info("[TTS Bridge] Connected to event bus")

    async def shutdown(self) -> None:
        """Gracefully shut down: flush buffers, stop workers, unsubscribe."""
        logger.info("[TTS Bridge] Shutting down")
        self._connected = False

        # Flush any remaining partial text
        for debate_id in list(self._text_buffers):
            await self._flush_buffer(debate_id)

        # Signal workers to stop via sentinel
        for debate_id, q in self._queues.items():
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass

        # Wait for workers to drain
        for task in list(self._workers.values()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._workers.clear()
        self._queues.clear()
        self._text_buffers.clear()
        self._buffer_timestamps.clear()
        self._audio_states.clear()

        # Unsubscribe from event bus
        if self._event_bus is not None:
            self._event_bus.unsubscribe("agent_message", self._on_agent_message)
            self._event_bus.unsubscribe("critique", self._on_agent_message)
            self._event_bus = None

        logger.info("[TTS Bridge] Shutdown complete")

    def get_audio_state(self, debate_id: str) -> dict[str, Any]:
        """Return the current audio state for a debate session.

        Args:
            debate_id: The debate to query.

        Returns:
            Dictionary with ``current_agent``, ``playback_state``,
            ``queued_frames``, and ``total_frames_sent``.
        """
        state = self._audio_states.get(debate_id)
        if state is None:
            return {
                "current_agent": "",
                "playback_state": AudioPlaybackState.IDLE.value,
                "queued_frames": 0,
                "total_frames_sent": 0,
            }
        return {
            "current_agent": state.current_agent,
            "playback_state": state.playback_state.value,
            "queued_frames": state.queued_frames,
            "total_frames_sent": state.total_frames_sent,
        }

    def pause_playback(self, debate_id: str) -> bool:
        """Pause audio playback state for a debate.

        Returns:
            True if state changed, False if already paused or no session.
        """
        state = self._audio_states.get(debate_id)
        if state is None or state.playback_state != AudioPlaybackState.PLAYING:
            return False
        state.playback_state = AudioPlaybackState.PAUSED
        logger.debug("[TTS Bridge] Paused playback for %s", debate_id)
        return True

    def resume_playback(self, debate_id: str) -> bool:
        """Resume audio playback state for a debate.

        Returns:
            True if state changed, False if not paused or no session.
        """
        state = self._audio_states.get(debate_id)
        if state is None or state.playback_state != AudioPlaybackState.PAUSED:
            return False
        state.playback_state = AudioPlaybackState.PLAYING
        logger.debug("[TTS Bridge] Resumed playback for %s", debate_id)
        return True

    # ------------------------------------------------------------------
    # Voice stream audio injection
    # ------------------------------------------------------------------

    def enqueue_audio_frame(self, debate_id: str, frame: bytes, agent: str = "") -> None:
        """Receive a synthesized audio frame and queue it for the voice stream.

        This method is called by the synthesis worker after TTSIntegration
        produces audio.  It updates the per-session audio state tracking.

        Args:
            debate_id: Target debate.
            frame: Raw audio bytes.
            agent: Agent that produced the text.
        """
        state = self._audio_states.setdefault(debate_id, _SessionAudioState())
        state.current_agent = agent
        state.queued_frames += 1
        state.total_frames_sent += 1
        if state.playback_state == AudioPlaybackState.IDLE:
            state.playback_state = AudioPlaybackState.PLAYING

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def _on_agent_message(self, event: DebateEvent) -> None:
        """Handle an ``agent_message`` event from the debate EventBus."""
        if not self._connected:
            return

        debate_id = event.debate_id
        content = event.data.get("content", "")
        agent = event.data.get("agent", "")

        if not content or not debate_id:
            return

        # Only process if there is an active voice session for this debate
        if not self.voice_handler.has_voice_session(debate_id):
            return

        # Accumulate text and detect sentence boundaries
        self._text_buffers.setdefault(debate_id, "")
        self._text_buffers[debate_id] += content

        if debate_id not in self._buffer_timestamps:
            self._buffer_timestamps[debate_id] = time.monotonic()

        # Check for complete sentences
        await self._process_buffer(debate_id, agent)

    async def _process_buffer(self, debate_id: str, agent: str) -> None:
        """Extract complete sentences from the buffer and enqueue for synthesis."""
        buf = self._text_buffers.get(debate_id, "")
        if not buf:
            return

        elapsed = time.monotonic() - self._buffer_timestamps.get(debate_id, time.monotonic())
        force_flush = (
            elapsed >= _PARTIAL_FLUSH_TIMEOUT or len(buf) >= _MAX_ACCUMULATOR_LEN
        )

        sentences: list[str] = []
        remaining = buf

        # Extract all complete sentences
        while True:
            match = _SENTENCE_END_RE.search(remaining)
            if match is None:
                break
            end = match.end()
            sentence = remaining[:end].strip()
            if sentence:
                sentences.append(sentence)
            remaining = remaining[end:]

        if sentences:
            self._text_buffers[debate_id] = remaining
            self._buffer_timestamps[debate_id] = time.monotonic()
        elif force_flush and remaining.strip():
            # Force-flush the partial buffer
            sentences.append(remaining.strip())
            self._text_buffers[debate_id] = ""
            self._buffer_timestamps[debate_id] = time.monotonic()

        # Enqueue complete sentences for synthesis
        for sentence in sentences:
            await self._enqueue_sentence(debate_id, agent, sentence)

    async def _flush_buffer(self, debate_id: str) -> None:
        """Flush any remaining text in the buffer for a debate."""
        buf = self._text_buffers.pop(debate_id, "").strip()
        self._buffer_timestamps.pop(debate_id, None)
        if buf:
            await self._enqueue_sentence(debate_id, "", buf)

    async def _enqueue_sentence(self, debate_id: str, agent: str, text: str) -> None:
        """Put a sentence into the per-debate synthesis queue.

        Lazily creates the queue and worker task on first use.
        """
        if debate_id not in self._queues:
            self._queues[debate_id] = asyncio.Queue(maxsize=100)
        if debate_id not in self._workers:
            self._workers[debate_id] = asyncio.create_task(
                self._synthesis_worker(debate_id)
            )

        try:
            self._queues[debate_id].put_nowait((agent, text))
        except asyncio.QueueFull:
            logger.warning(
                "[TTS Bridge] Synthesis queue full for %s, dropping sentence", debate_id
            )

    # ------------------------------------------------------------------
    # Synthesis worker
    # ------------------------------------------------------------------

    async def _synthesis_worker(self, debate_id: str) -> None:
        """Consume sentences from the queue and synthesize them."""
        q = self._queues[debate_id]
        while True:
            try:
                item = await q.get()
            except asyncio.CancelledError:
                break

            if item is None:
                break  # Shutdown sentinel

            agent, text = item
            try:
                sessions_sent = await self.tts.synthesize_for_debate(
                    debate_id=debate_id,
                    agent_name=agent,
                    text=text,
                )
                if sessions_sent and sessions_sent > 0:
                    self.enqueue_audio_frame(debate_id, b"", agent)
                    logger.debug(
                        "[TTS Bridge] Synthesized '%s...' for %s (%d sessions)",
                        text[:40],
                        debate_id,
                        sessions_sent,
                    )
            except (OSError, RuntimeError, ValueError, TimeoutError) as e:
                logger.warning("[TTS Bridge] Synthesis failed for %s: %s", debate_id, e)


# ---------------------------------------------------------------------------
# Convenience helpers for TTSIntegration
# ---------------------------------------------------------------------------

def _add_synthesize_for_debate(tts_cls: type) -> None:
    """Monkey-patch ``synthesize_for_debate`` onto TTSIntegration if absent.

    The existing ``_handle_agent_message`` in TTSIntegration calls
    ``synthesize_agent_message`` on the voice handler.  The bridge needs a
    simpler interface that doesn't require looking up WebSocket connections
    internally, so we add a thin wrapper that delegates appropriately.
    """
    if hasattr(tts_cls, "synthesize_for_debate"):
        return

    async def synthesize_for_debate(
        self: Any,
        debate_id: str,
        agent_name: str,
        text: str,
    ) -> int:
        """Synthesize text for all voice sessions of a debate.

        This is a convenience wrapper around the voice handler's
        ``synthesize_agent_message``.
        """
        if self._voice_handler is None:
            return 0
        return await self._voice_handler.synthesize_agent_message(
            debate_id=debate_id,
            agent_name=agent_name,
            message=text,
        )

    tts_cls.synthesize_for_debate = synthesize_for_debate


# Apply the patch at import time so the bridge can always call it.
try:
    from aragora.server.stream.tts_integration import TTSIntegration as _TTSCls

    _add_synthesize_for_debate(_TTSCls)
except ImportError:
    pass


__all__ = [
    "TTSEventBridge",
    "AudioPlaybackState",
]
