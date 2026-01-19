"""
WebSocket handler for live voice streaming and transcription.

Provides real-time speech-to-text via OpenAI Whisper API for:
- Live voice input during debates
- Recording and transcribing spoken arguments
- Voice-controlled debate participation

Architecture:
    Browser -> WebSocket -> VoiceStreamHandler -> WhisperConnector -> Transcription
                                    |
                                    v
                              StreamEvent (VOICE_TRANSCRIPT)
                                    |
                                    v
                              Debate Context / Arena

Usage:
    # Register in unified server routes
    server.add_websocket_handler("/ws/voice/{debate_id}", VoiceStreamHandler(server))
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

from aragora.connectors.whisper import (
    WhisperConnector,
    TranscriptionResult,
    TranscriptionSegment,
    MAX_FILE_SIZE_BYTES,
    is_supported_media,
)
from aragora.connectors.exceptions import ConnectorConfigError, ConnectorRateLimitError
from aragora.server.stream.events import StreamEvent, StreamEventType

if TYPE_CHECKING:
    from aiohttp import web
    from aragora.server.stream.server_base import ServerBase

logger = logging.getLogger(__name__)

# Voice stream configuration
VOICE_CHUNK_SIZE_BYTES = int(os.getenv("ARAGORA_VOICE_CHUNK_SIZE", str(16000 * 2 * 3)))  # 3 seconds at 16kHz 16-bit
VOICE_MAX_SESSION_SECONDS = int(os.getenv("ARAGORA_VOICE_MAX_SESSION", "300"))  # 5 minutes max
VOICE_MAX_BUFFER_BYTES = int(os.getenv("ARAGORA_VOICE_MAX_BUFFER", str(25 * 1024 * 1024)))  # 25MB (Whisper limit)
VOICE_TRANSCRIBE_INTERVAL_MS = int(os.getenv("ARAGORA_VOICE_INTERVAL", "3000"))  # 3 seconds

# Rate limiting
VOICE_MAX_SESSIONS_PER_IP = int(os.getenv("ARAGORA_VOICE_MAX_SESSIONS_IP", "3"))
VOICE_MAX_BYTES_PER_MINUTE = int(os.getenv("ARAGORA_VOICE_RATE_BYTES", str(5 * 1024 * 1024)))  # 5MB/min


@dataclass
class VoiceSession:
    """Tracks state for an active voice streaming session."""

    session_id: str
    debate_id: str
    client_ip: str
    started_at: float = field(default_factory=time.time)
    last_chunk_at: float = field(default_factory=time.time)
    audio_buffer: bytes = b""
    total_bytes_received: int = 0
    transcription_count: int = 0
    accumulated_text: str = ""
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str = ""
    is_active: bool = True

    def add_chunk(self, chunk: bytes) -> bool:
        """Add audio chunk to buffer, return False if buffer overflow."""
        if len(self.audio_buffer) + len(chunk) > VOICE_MAX_BUFFER_BYTES:
            return False
        self.audio_buffer += chunk
        self.total_bytes_received += len(chunk)
        self.last_chunk_at = time.time()
        return True

    def clear_buffer(self) -> bytes:
        """Get and clear the audio buffer."""
        buffer = self.audio_buffer
        self.audio_buffer = b""
        return buffer

    def elapsed_seconds(self) -> float:
        """Get session duration in seconds."""
        return time.time() - self.started_at


class VoiceStreamHandler:
    """
    WebSocket handler for live voice streaming.

    Receives audio chunks via WebSocket, buffers them, and periodically
    sends them to Whisper for transcription. Emits VOICE_TRANSCRIPT events
    that can be consumed by debates.

    Protocol:
        Client -> Server (binary): Raw audio chunks (PCM 16kHz 16-bit mono recommended)
        Client -> Server (JSON): Control messages {"type": "start"|"end"|"config", ...}
        Server -> Client (JSON): Events {"type": "transcript"|"error"|"ready", ...}

    Example client usage:
        ws = new WebSocket("/ws/voice/debate_123");
        ws.send(JSON.stringify({type: "start", format: "pcm", sample_rate: 16000}));

        // Send audio chunks from MediaRecorder/AudioWorklet
        ws.send(audioChunkArrayBuffer);

        // Receive transcripts
        ws.onmessage = (e) => {
            const event = JSON.parse(e.data);
            if (event.type === "transcript") {
                console.log(event.text);
            }
        };

        // End session
        ws.send(JSON.stringify({type: "end"}));
    """

    def __init__(
        self,
        server: "ServerBase",
        whisper: Optional[WhisperConnector] = None,
    ):
        """
        Initialize VoiceStreamHandler.

        Args:
            server: Parent server for emitting events
            whisper: WhisperConnector instance (creates new one if not provided)
        """
        self.server = server
        self.whisper = whisper or WhisperConnector()

        # Active voice sessions
        self._sessions: Dict[str, VoiceSession] = {}
        self._sessions_lock = asyncio.Lock()

        # Rate limiting by IP
        self._ip_sessions: Dict[str, Set[str]] = {}  # ip -> set of session_ids
        self._ip_bytes_minute: Dict[str, list[tuple[float, int]]] = {}  # ip -> [(timestamp, bytes)]

    @property
    def is_available(self) -> bool:
        """Check if voice streaming is available."""
        return self.whisper.is_available

    async def handle_websocket(
        self,
        request: "web.Request",
        ws: "web.WebSocketResponse",
        debate_id: str,
    ) -> None:
        """
        Handle a voice streaming WebSocket connection.

        Args:
            request: The aiohttp request
            ws: The WebSocket response
            debate_id: The debate ID from URL path
        """
        # Extract client IP
        client_ip = self._get_client_ip(request)
        ws_id = id(ws)

        # Check availability
        if not self.is_available:
            await ws.send_json({
                "type": "error",
                "code": "voice_unavailable",
                "message": "Voice transcription is not available. Check OPENAI_API_KEY.",
            })
            await ws.close()
            return

        # Check per-IP session limit
        async with self._sessions_lock:
            ip_sessions = self._ip_sessions.get(client_ip, set())
            if len(ip_sessions) >= VOICE_MAX_SESSIONS_PER_IP:
                await ws.send_json({
                    "type": "error",
                    "code": "rate_limited",
                    "message": f"Maximum {VOICE_MAX_SESSIONS_PER_IP} voice sessions per IP.",
                })
                await ws.close()
                return

        # Create session
        session_id = f"voice_{uuid.uuid4().hex[:12]}"
        session = VoiceSession(
            session_id=session_id,
            debate_id=debate_id,
            client_ip=client_ip,
        )

        async with self._sessions_lock:
            self._sessions[session_id] = session
            if client_ip not in self._ip_sessions:
                self._ip_sessions[client_ip] = set()
            self._ip_sessions[client_ip].add(session_id)

        logger.info(f"[Voice] Session {session_id} started for debate {debate_id} from {client_ip}")

        # Send ready message
        await ws.send_json({
            "type": "ready",
            "session_id": session_id,
            "debate_id": debate_id,
            "config": {
                "max_buffer_bytes": VOICE_MAX_BUFFER_BYTES,
                "transcribe_interval_ms": VOICE_TRANSCRIBE_INTERVAL_MS,
                "max_session_seconds": VOICE_MAX_SESSION_SECONDS,
            },
        })

        # Emit voice start event
        self._emit_event(StreamEventType.VOICE_START, {
            "session_id": session_id,
            "debate_id": debate_id,
        }, debate_id)

        # Background task for periodic transcription
        transcribe_task = asyncio.create_task(
            self._periodic_transcribe(session, ws)
        )

        try:
            async for msg in ws:
                if msg.type == 1:  # aiohttp.WSMsgType.TEXT
                    await self._handle_text_message(session, ws, msg.data)
                elif msg.type == 2:  # aiohttp.WSMsgType.BINARY
                    await self._handle_binary_chunk(session, ws, msg.data)
                elif msg.type == 8:  # aiohttp.WSMsgType.ERROR
                    logger.error(f"[Voice] WebSocket error: {ws.exception()}")
                    break

                # Check session limits
                if session.elapsed_seconds() > VOICE_MAX_SESSION_SECONDS:
                    await ws.send_json({
                        "type": "error",
                        "code": "session_timeout",
                        "message": f"Maximum session duration ({VOICE_MAX_SESSION_SECONDS}s) exceeded.",
                    })
                    break

        except Exception as e:
            logger.error(f"[Voice] Session {session_id} error: {e}")

        finally:
            # Clean up
            session.is_active = False
            transcribe_task.cancel()

            # Final transcription of remaining buffer
            if session.audio_buffer:
                try:
                    await self._transcribe_buffer(session, ws, final=True)
                except Exception as e:
                    logger.warning(f"[Voice] Final transcription failed: {e}")

            # Remove session
            async with self._sessions_lock:
                self._sessions.pop(session_id, None)
                if client_ip in self._ip_sessions:
                    self._ip_sessions[client_ip].discard(session_id)
                    if not self._ip_sessions[client_ip]:
                        del self._ip_sessions[client_ip]

            # Emit voice end event
            self._emit_event(StreamEventType.VOICE_END, {
                "session_id": session_id,
                "debate_id": debate_id,
                "total_bytes": session.total_bytes_received,
                "transcription_count": session.transcription_count,
                "total_text": session.accumulated_text,
                "duration_seconds": session.elapsed_seconds(),
            }, debate_id)

            logger.info(
                f"[Voice] Session {session_id} ended: "
                f"{session.total_bytes_received} bytes, "
                f"{session.transcription_count} transcriptions, "
                f"{session.elapsed_seconds():.1f}s"
            )

    async def _handle_text_message(
        self,
        session: VoiceSession,
        ws: "web.WebSocketResponse",
        data: str,
    ) -> None:
        """Handle JSON control message from client."""
        try:
            msg = json.loads(data)
            msg_type = msg.get("type", "")

            if msg_type == "config":
                # Client sending audio format configuration
                session.language = msg.get("language", "")
                await ws.send_json({"type": "config_ack"})

            elif msg_type == "end":
                # Client requesting end of session
                session.is_active = False
                # Final transcription will happen in cleanup

            elif msg_type == "ping":
                await ws.send_json({"type": "pong", "timestamp": time.time()})

        except json.JSONDecodeError:
            logger.warning(f"[Voice] Invalid JSON message: {data[:100]}")

    async def _handle_binary_chunk(
        self,
        session: VoiceSession,
        ws: "web.WebSocketResponse",
        chunk: bytes,
    ) -> None:
        """Handle binary audio chunk from client."""
        # Check rate limit
        if not self._check_rate_limit(session.client_ip, len(chunk)):
            await ws.send_json({
                "type": "error",
                "code": "rate_limited",
                "message": "Audio upload rate limit exceeded.",
            })
            return

        # Add to buffer
        if not session.add_chunk(chunk):
            await ws.send_json({
                "type": "error",
                "code": "buffer_overflow",
                "message": "Audio buffer full. End session or wait for transcription.",
            })
            return

        # Emit chunk event (for progress tracking)
        self._emit_event(StreamEventType.VOICE_CHUNK, {
            "session_id": session.session_id,
            "chunk_size": len(chunk),
            "buffer_size": len(session.audio_buffer),
        }, session.debate_id)

    async def _periodic_transcribe(
        self,
        session: VoiceSession,
        ws: "web.WebSocketResponse",
    ) -> None:
        """Periodically transcribe accumulated audio."""
        interval = VOICE_TRANSCRIBE_INTERVAL_MS / 1000.0

        while session.is_active:
            await asyncio.sleep(interval)

            if session.audio_buffer and len(session.audio_buffer) >= VOICE_CHUNK_SIZE_BYTES:
                try:
                    await self._transcribe_buffer(session, ws)
                except Exception as e:
                    logger.warning(f"[Voice] Periodic transcription failed: {e}")

    async def _transcribe_buffer(
        self,
        session: VoiceSession,
        ws: "web.WebSocketResponse",
        final: bool = False,
    ) -> None:
        """Transcribe the current audio buffer."""
        buffer = session.clear_buffer()
        if not buffer:
            return

        try:
            # Create WAV header for raw PCM data (assumed 16kHz 16-bit mono)
            wav_buffer = self._create_wav_header(buffer, 16000, 1, 16) + buffer
            filename = f"voice_{session.session_id}_{session.transcription_count}.wav"

            # Transcribe
            result = await self.whisper.transcribe(
                wav_buffer,
                filename,
                prompt=session.accumulated_text[-500:] if session.accumulated_text else None,
            )

            session.transcription_count += 1
            session.accumulated_text += " " + result.text
            session.segments.extend(result.segments)
            if result.language:
                session.language = result.language

            # Send transcript to client
            await ws.send_json({
                "type": "transcript",
                "session_id": session.session_id,
                "text": result.text,
                "language": result.language,
                "duration_seconds": result.duration_seconds,
                "word_count": result.word_count,
                "is_final": final,
                "segments": [s.to_dict() for s in result.segments],
            })

            # Emit transcript event
            self._emit_event(StreamEventType.VOICE_TRANSCRIPT, {
                "session_id": session.session_id,
                "debate_id": session.debate_id,
                "text": result.text,
                "language": result.language,
                "is_final": final,
                "accumulated_text": session.accumulated_text.strip(),
            }, session.debate_id)

        except ConnectorRateLimitError as e:
            logger.warning(f"[Voice] Whisper rate limit: {e}")
            # Re-add buffer to try again later
            session.audio_buffer = buffer + session.audio_buffer
            await ws.send_json({
                "type": "warning",
                "code": "rate_limited",
                "message": "Transcription API rate limited. Buffering audio.",
            })

        except ConnectorConfigError as e:
            logger.error(f"[Voice] Configuration error: {e}")
            await ws.send_json({
                "type": "error",
                "code": "config_error",
                "message": str(e),
            })

        except Exception as e:
            logger.error(f"[Voice] Transcription error: {e}")
            await ws.send_json({
                "type": "error",
                "code": "transcription_failed",
                "message": f"Transcription failed: {e}",
            })

    def _create_wav_header(
        self,
        pcm_data: bytes,
        sample_rate: int,
        channels: int,
        bits_per_sample: int,
    ) -> bytes:
        """Create WAV file header for raw PCM data."""
        import struct

        data_size = len(pcm_data)
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,  # File size - 8
            b"WAVE",
            b"fmt ",
            16,  # Subchunk1 size
            1,  # Audio format (PCM)
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size,
        )
        return header

    def _check_rate_limit(self, client_ip: str, chunk_size: int) -> bool:
        """Check if client is within byte rate limit."""
        now = time.time()
        minute_ago = now - 60.0

        if client_ip not in self._ip_bytes_minute:
            self._ip_bytes_minute[client_ip] = []

        # Clean old entries
        self._ip_bytes_minute[client_ip] = [
            (ts, size) for ts, size in self._ip_bytes_minute[client_ip]
            if ts > minute_ago
        ]

        # Calculate bytes in last minute
        total_bytes = sum(size for _, size in self._ip_bytes_minute[client_ip])

        if total_bytes + chunk_size > VOICE_MAX_BYTES_PER_MINUTE:
            return False

        self._ip_bytes_minute[client_ip].append((now, chunk_size))
        return True

    def _get_client_ip(self, request: "web.Request") -> str:
        """Extract client IP from request."""
        # Check X-Forwarded-For for proxied requests
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            # Take the first IP (original client)
            return forwarded.split(",")[0].strip()
        # Fall back to direct connection
        peername = request.transport.get_extra_info("peername")
        if peername:
            return peername[0]
        return "unknown"

    def _emit_event(
        self,
        event_type: StreamEventType,
        data: dict,
        debate_id: str,
    ) -> None:
        """Emit a stream event."""
        event = StreamEvent(
            type=event_type,
            data=data,
            loop_id=debate_id,
        )
        self.server.emitter.emit(event)

    async def get_session_info(self, session_id: str) -> Optional[dict]:
        """Get information about an active voice session."""
        async with self._sessions_lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            return {
                "session_id": session.session_id,
                "debate_id": session.debate_id,
                "started_at": session.started_at,
                "elapsed_seconds": session.elapsed_seconds(),
                "total_bytes_received": session.total_bytes_received,
                "transcription_count": session.transcription_count,
                "language": session.language,
                "is_active": session.is_active,
            }

    async def get_active_sessions(self) -> list[dict]:
        """Get all active voice sessions."""
        async with self._sessions_lock:
            return [
                {
                    "session_id": s.session_id,
                    "debate_id": s.debate_id,
                    "elapsed_seconds": s.elapsed_seconds(),
                    "total_bytes": s.total_bytes_received,
                }
                for s in self._sessions.values()
                if s.is_active
            ]
