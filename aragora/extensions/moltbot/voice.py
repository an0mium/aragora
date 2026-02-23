"""
Voice Processor - Speech-to-Text and Text-to-Speech Integration.

Provides voice session management with STT/TTS integration,
voice activity detection, and conversation transcription.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Callable

from .models import (
    VoiceSession,
    VoiceSessionConfig,
)

logger = logging.getLogger(__name__)


class VoiceProcessor:
    """
    Voice processing for speech interactions.

    Manages voice sessions with speech-to-text, text-to-speech,
    and voice activity detection capabilities.
    """

    def __init__(
        self,
        storage_path: str | Path | None = None,
        stt_provider: str = "mock",
        tts_provider: str = "mock",
    ) -> None:
        """
        Initialize the voice processor.

        Args:
            storage_path: Path for session and transcript storage
            stt_provider: Speech-to-text provider name
            tts_provider: Text-to-speech provider name
        """
        self._storage_path = Path(storage_path) if storage_path else None
        self._stt_provider = stt_provider
        self._tts_provider = tts_provider

        self._sessions: dict[str, VoiceSession] = {}
        self._stt_handlers: dict[str, Callable] = {}
        self._tts_handlers: dict[str, Callable] = {}
        self._lock = asyncio.Lock()

        if self._storage_path:
            self._storage_path.mkdir(parents=True, exist_ok=True)

        # Register mock providers
        self._register_mock_providers()

    def _register_mock_providers(self) -> None:
        """Register mock STT/TTS providers for testing."""

        async def mock_stt(audio_data: bytes, config: VoiceSessionConfig) -> str:
            # Simulate STT by returning placeholder
            return "[mock transcription]"

        async def mock_tts(text: str, config: VoiceSessionConfig) -> bytes:
            # Simulate TTS by returning placeholder audio
            return b"mock_audio_data"

        self._stt_handlers["mock"] = mock_stt
        self._tts_handlers["mock"] = mock_tts

    # ========== Session Management ==========

    async def create_session(
        self,
        config: VoiceSessionConfig,
        user_id: str,
        channel_id: str,
        tenant_id: str | None = None,
    ) -> VoiceSession:
        """
        Create a new voice session.

        Args:
            config: Session configuration
            user_id: User ID
            channel_id: Source channel ID
            tenant_id: Tenant ID for multi-tenancy

        Returns:
            Created voice session
        """
        async with self._lock:
            session_id = str(uuid.uuid4())

            session = VoiceSession(
                id=session_id,
                config=config,
                user_id=user_id,
                channel_id=channel_id,
                tenant_id=tenant_id,
                started_at=datetime.now(timezone.utc),
            )

            self._sessions[session_id] = session
            logger.info("Created voice session %s", session_id)

            return session

    async def get_session(self, session_id: str) -> VoiceSession | None:
        """Get a voice session by ID."""
        return self._sessions.get(session_id)

    async def list_sessions(
        self,
        user_id: str | None = None,
        channel_id: str | None = None,
        status: str | None = None,
        tenant_id: str | None = None,
    ) -> list[VoiceSession]:
        """List voice sessions with optional filters."""
        sessions = list(self._sessions.values())

        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        if channel_id:
            sessions = [s for s in sessions if s.channel_id == channel_id]
        if status:
            sessions = [s for s in sessions if s.status == status]
        if tenant_id:
            sessions = [s for s in sessions if s.tenant_id == tenant_id]

        return sessions

    async def end_session(
        self,
        session_id: str,
        reason: str = "completed",
    ) -> VoiceSession | None:
        """
        End a voice session.

        Args:
            session_id: Session to end
            reason: Reason for ending

        Returns:
            Ended session
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            session.status = "ended"
            session.ended_at = datetime.now(timezone.utc)
            session.updated_at = datetime.now(timezone.utc)

            if session.started_at and session.ended_at:
                session.duration_seconds = (session.ended_at - session.started_at).total_seconds()

            session.metadata["end_reason"] = reason

            logger.info("Ended voice session %s: %s", session_id, reason)

            # Persist transcript if storage configured
            if self._storage_path:
                await self._persist_transcript(session)

            return session

    async def pause_session(self, session_id: str) -> VoiceSession | None:
        """Pause a voice session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            session.status = "paused"
            session.updated_at = datetime.now(timezone.utc)

            logger.debug("Paused voice session %s", session_id)
            return session

    async def resume_session(self, session_id: str) -> VoiceSession | None:
        """Resume a paused voice session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            if session.status != "paused":
                return session

            session.status = "active"
            session.updated_at = datetime.now(timezone.utc)

            logger.debug("Resumed voice session %s", session_id)
            return session

    # ========== Speech-to-Text ==========

    async def process_audio(
        self,
        session_id: str,
        audio_data: bytes,
    ) -> dict[str, Any]:
        """
        Process audio data and return transcription.

        Args:
            session_id: Active session ID
            audio_data: Raw audio bytes

        Returns:
            Transcription result
        """
        session = self._sessions.get(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}

        if session.status != "active":
            return {"success": False, "error": f"Session is {session.status}"}

        if not session.config.enable_stt:
            return {"success": False, "error": "STT not enabled for this session"}

        # Get STT handler
        handler = self._stt_handlers.get(self._stt_provider)
        if not handler:
            return {"success": False, "error": f"Unknown STT provider: {self._stt_provider}"}

        try:
            # Transcribe audio
            transcript = await handler(audio_data, session.config)

            # Update session
            async with self._lock:
                session.current_transcript = transcript
                session.transcripts.append(
                    {
                        "type": "user",
                        "text": transcript,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                session.words_heard += len(transcript.split())
                session.turns += 1
                session.updated_at = datetime.now(timezone.utc)

            # Simple intent detection
            intent = self._detect_intent(transcript)
            if intent:
                session.intent_history.append(intent)

            return {
                "success": True,
                "transcript": transcript,
                "intent": intent,
                "is_final": True,
            }

        except (RuntimeError, OSError, ValueError) as e:
            logger.error("STT error: %s", e)
            return {"success": False, "error": "Speech-to-text processing failed"}

    def _detect_intent(self, text: str) -> str | None:
        """Simple intent detection from transcript."""
        text_lower = text.lower()

        if any(word in text_lower for word in ["stop", "end", "goodbye", "bye"]):
            return "end_conversation"
        if any(word in text_lower for word in ["help", "assist", "support"]):
            return "help_request"
        if any(word in text_lower for word in ["repeat", "again", "what"]):
            return "clarification"
        if "?" in text:
            return "question"

        return None

    # ========== Text-to-Speech ==========

    async def synthesize_speech(
        self,
        session_id: str,
        text: str,
    ) -> dict[str, Any]:
        """
        Synthesize speech from text.

        Args:
            session_id: Active session ID
            text: Text to synthesize

        Returns:
            Synthesis result with audio data
        """
        session = self._sessions.get(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}

        if not session.config.enable_tts:
            return {"success": False, "error": "TTS not enabled for this session"}

        # Get TTS handler
        handler = self._tts_handlers.get(self._tts_provider)
        if not handler:
            return {"success": False, "error": f"Unknown TTS provider: {self._tts_provider}"}

        try:
            # Synthesize speech
            audio_data = await handler(text, session.config)

            # Update session
            async with self._lock:
                session.transcripts.append(
                    {
                        "type": "system",
                        "text": text,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                session.words_spoken += len(text.split())
                session.updated_at = datetime.now(timezone.utc)

            return {
                "success": True,
                "audio": audio_data,
                "format": session.config.encoding,
                "sample_rate": session.config.sample_rate,
            }

        except (RuntimeError, OSError, ValueError) as e:
            logger.error("TTS error: %s", e)
            return {"success": False, "error": "Text-to-speech synthesis failed"}

    # ========== Provider Management ==========

    def register_stt_provider(
        self,
        name: str,
        handler: Callable,
    ) -> None:
        """Register a speech-to-text provider."""
        self._stt_handlers[name] = handler
        logger.info("Registered STT provider: %s", name)

    def register_tts_provider(
        self,
        name: str,
        handler: Callable,
    ) -> None:
        """Register a text-to-speech provider."""
        self._tts_handlers[name] = handler
        logger.info("Registered TTS provider: %s", name)

    def set_stt_provider(self, name: str) -> bool:
        """Set the active STT provider."""
        if name not in self._stt_handlers:
            return False
        self._stt_provider = name
        return True

    def set_tts_provider(self, name: str) -> bool:
        """Set the active TTS provider."""
        if name not in self._tts_handlers:
            return False
        self._tts_provider = name
        return True

    # ========== Transcript Management ==========

    async def get_transcript(self, session_id: str) -> list[dict[str, Any]]:
        """Get the full transcript for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return []
        return session.transcripts.copy()

    async def _persist_transcript(self, session: VoiceSession) -> None:
        """Persist session transcript to storage."""
        if not self._storage_path:
            return

        import json

        transcript_path = self._storage_path / f"{session.id}.json"
        transcript_data = {
            "session_id": session.id,
            "user_id": session.user_id,
            "channel_id": session.channel_id,
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
            "duration_seconds": session.duration_seconds,
            "turns": session.turns,
            "transcripts": session.transcripts,
            "intents": session.intent_history,
        }

        transcript_path.write_text(json.dumps(transcript_data, indent=2))
        logger.debug("Persisted transcript for session %s", session.id)

    # ========== Statistics ==========

    async def get_stats(self) -> dict[str, Any]:
        """Get voice processor statistics."""
        async with self._lock:
            active = sum(1 for s in self._sessions.values() if s.status == "active")
            ended = sum(1 for s in self._sessions.values() if s.status == "ended")

            total_turns = sum(s.turns for s in self._sessions.values())
            total_words_spoken = sum(s.words_spoken for s in self._sessions.values())
            total_words_heard = sum(s.words_heard for s in self._sessions.values())
            total_duration = sum(s.duration_seconds for s in self._sessions.values())

            return {
                "sessions_total": len(self._sessions),
                "sessions_active": active,
                "sessions_ended": ended,
                "total_turns": total_turns,
                "total_words_spoken": total_words_spoken,
                "total_words_heard": total_words_heard,
                "total_duration_seconds": total_duration,
                "stt_provider": self._stt_provider,
                "tts_provider": self._tts_provider,
                "stt_providers_available": list(self._stt_handlers.keys()),
                "tts_providers_available": list(self._tts_handlers.keys()),
            }
