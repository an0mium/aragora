"""
Moltbot Voice Handler - Voice Session Management REST API.

Endpoints:
- GET  /api/v1/moltbot/voice/sessions         - List sessions
- POST /api/v1/moltbot/voice/sessions         - Create session
- GET  /api/v1/moltbot/voice/sessions/{id}    - Get session
- DELETE /api/v1/moltbot/voice/sessions/{id}  - End session
- POST /api/v1/moltbot/voice/sessions/{id}/pause  - Pause session
- POST /api/v1/moltbot/voice/sessions/{id}/resume - Resume session
- POST /api/v1/moltbot/voice/sessions/{id}/audio  - Process audio (STT)
- POST /api/v1/moltbot/voice/sessions/{id}/speak  - Synthesize speech (TTS)
- GET  /api/v1/moltbot/voice/sessions/{id}/transcript - Get transcript
- GET  /api/v1/moltbot/voice/stats            - Voice statistics
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)

if TYPE_CHECKING:
    from aragora.extensions.moltbot import VoiceProcessor

logger = logging.getLogger(__name__)

# Global voice processor instance (lazily initialized)
_voice_processor: Optional["VoiceProcessor"] = None


def get_voice_processor() -> "VoiceProcessor":
    """Get or create the voice processor instance."""
    global _voice_processor
    if _voice_processor is None:
        from aragora.extensions.moltbot import VoiceProcessor

        _voice_processor = VoiceProcessor()
    return _voice_processor


class MoltbotVoiceHandler(BaseHandler):
    """HTTP handler for Moltbot voice operations."""

    routes = [
        ("GET", "/api/v1/moltbot/voice/sessions"),
        ("POST", "/api/v1/moltbot/voice/sessions"),
        ("GET", "/api/v1/moltbot/voice/sessions/"),
        ("DELETE", "/api/v1/moltbot/voice/sessions/"),
        ("POST", "/api/v1/moltbot/voice/sessions/*/pause"),
        ("POST", "/api/v1/moltbot/voice/sessions/*/resume"),
        ("POST", "/api/v1/moltbot/voice/sessions/*/audio"),
        ("POST", "/api/v1/moltbot/voice/sessions/*/speak"),
        ("GET", "/api/v1/moltbot/voice/sessions/*/transcript"),
        ("GET", "/api/v1/moltbot/voice/stats"),
    ]

    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle GET requests."""
        if path == "/api/v1/moltbot/voice/sessions":
            return await self._handle_list_sessions(query_params, handler)
        elif path == "/api/v1/moltbot/voice/stats":
            return await self._handle_voice_stats(handler)
        elif path.startswith("/api/v1/moltbot/voice/sessions/"):
            parts = path.split("/")
            if len(parts) >= 6:
                session_id = parts[5]
                if len(parts) == 7 and parts[6] == "transcript":
                    return await self._handle_get_transcript(session_id, handler)
                elif len(parts) == 6:
                    return await self._handle_get_session(session_id, handler)
        return None

    async def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle POST requests."""
        if path == "/api/v1/moltbot/voice/sessions":
            return await self._handle_create_session(handler)

        if path.startswith("/api/v1/moltbot/voice/sessions/"):
            parts = path.split("/")
            if len(parts) >= 6:
                session_id = parts[5]
                if len(parts) == 7:
                    action = parts[6]
                    if action == "pause":
                        return await self._handle_pause_session(session_id, handler)
                    elif action == "resume":
                        return await self._handle_resume_session(session_id, handler)
                    elif action == "audio":
                        return await self._handle_process_audio(session_id, handler)
                    elif action == "speak":
                        return await self._handle_synthesize_speech(session_id, handler)
        return None

    async def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle DELETE requests."""
        if path.startswith("/api/v1/moltbot/voice/sessions/"):
            parts = path.split("/")
            if len(parts) >= 6:
                session_id = parts[5]
                return await self._handle_end_session(session_id, query_params, handler)
        return None

    # ========== Session Handlers ==========

    async def _handle_list_sessions(
        self, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """
        List voice sessions with optional filters.

        GET /api/v1/moltbot/voice/sessions?user_id=...&status=...
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        processor = get_voice_processor()

        # Extract filters
        user_id = query_params.get("user_id")
        channel_id = query_params.get("channel_id")
        status = query_params.get("status")
        tenant_id = query_params.get("tenant_id")

        sessions = await processor.list_sessions(
            user_id=user_id,
            channel_id=channel_id,
            status=status,
            tenant_id=tenant_id,
        )

        return json_response(
            {
                "sessions": [
                    {
                        "id": s.id,
                        "user_id": s.user_id,
                        "channel_id": s.channel_id,
                        "status": s.status,
                        "turns": s.turns,
                        "duration_seconds": s.duration_seconds,
                        "started_at": s.started_at.isoformat() if s.started_at else None,
                    }
                    for s in sessions
                ],
                "total": len(sessions),
            }
        )

    async def _handle_create_session(self, handler: Any) -> HandlerResult:
        """
        Create a new voice session.

        POST /api/v1/moltbot/voice/sessions
        Body: {channel_id, config?}
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if not body:
            return error_response("Request body required", 400)

        channel_id = body.get("channel_id")
        if not channel_id:
            return error_response("channel_id is required", 400)

        from aragora.extensions.moltbot import VoiceSessionConfig

        config_data = body.get("config", {})
        config = VoiceSessionConfig(
            sample_rate=config_data.get("sample_rate", 16000),
            encoding=config_data.get("encoding", "pcm"),
            language=config_data.get("language", "en-US"),
            enable_stt=config_data.get("enable_stt", True),
            enable_tts=config_data.get("enable_tts", True),
            metadata={"channels": config_data.get("channels", 1)},
        )

        processor = get_voice_processor()
        session = await processor.create_session(
            config=config,
            user_id=user.user_id,
            channel_id=channel_id,
            tenant_id=body.get("tenant_id"),
        )

        return json_response(
            {
                "success": True,
                "session": {
                    "id": session.id,
                    "status": session.status,
                    "config": {
                        "sample_rate": session.config.sample_rate,
                        "encoding": session.config.encoding,
                        "language": session.config.language,
                    },
                },
            },
            status=201,
        )

    async def _handle_get_session(self, session_id: str, handler: Any) -> HandlerResult:
        """
        Get a voice session by ID.

        GET /api/v1/moltbot/voice/sessions/{session_id}
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        processor = get_voice_processor()
        session = await processor.get_session(session_id)

        if not session:
            return error_response("Session not found", 404)

        return json_response(
            {
                "session": {
                    "id": session.id,
                    "user_id": session.user_id,
                    "channel_id": session.channel_id,
                    "tenant_id": session.tenant_id,
                    "status": session.status,
                    "turns": session.turns,
                    "words_spoken": session.words_spoken,
                    "words_heard": session.words_heard,
                    "duration_seconds": session.duration_seconds,
                    "current_transcript": session.current_transcript,
                    "intent_history": session.intent_history,
                    "config": {
                        "sample_rate": session.config.sample_rate,
                        "encoding": session.config.encoding,
                        "language": session.config.language,
                        "enable_stt": session.config.enable_stt,
                        "enable_tts": session.config.enable_tts,
                    },
                    "started_at": session.started_at.isoformat() if session.started_at else None,
                    "ended_at": session.ended_at.isoformat() if session.ended_at else None,
                },
            }
        )

    async def _handle_end_session(
        self, session_id: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """
        End a voice session.

        DELETE /api/v1/moltbot/voice/sessions/{session_id}?reason=...
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        reason = query_params.get("reason", "completed")

        processor = get_voice_processor()
        session = await processor.end_session(session_id, reason=reason)

        if not session:
            return error_response("Session not found", 404)

        return json_response(
            {
                "success": True,
                "session_id": session.id,
                "status": session.status,
                "duration_seconds": session.duration_seconds,
                "turns": session.turns,
            }
        )

    async def _handle_pause_session(self, session_id: str, handler: Any) -> HandlerResult:
        """
        Pause a voice session.

        POST /api/v1/moltbot/voice/sessions/{session_id}/pause
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        processor = get_voice_processor()
        session = await processor.pause_session(session_id)

        if not session:
            return error_response("Session not found", 404)

        return json_response(
            {
                "success": True,
                "session_id": session.id,
                "status": session.status,
            }
        )

    async def _handle_resume_session(self, session_id: str, handler: Any) -> HandlerResult:
        """
        Resume a paused voice session.

        POST /api/v1/moltbot/voice/sessions/{session_id}/resume
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        processor = get_voice_processor()
        session = await processor.resume_session(session_id)

        if not session:
            return error_response("Session not found", 404)

        return json_response(
            {
                "success": True,
                "session_id": session.id,
                "status": session.status,
            }
        )

    # ========== Audio Processing ==========

    async def _handle_process_audio(self, session_id: str, handler: Any) -> HandlerResult:
        """
        Process audio data (Speech-to-Text).

        POST /api/v1/moltbot/voice/sessions/{session_id}/audio
        Body: {audio: base64-encoded audio data}
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if not body:
            return error_response("Request body required", 400)

        audio_b64 = body.get("audio")
        if not audio_b64:
            return error_response("audio (base64) is required", 400)

        try:
            audio_data = base64.b64decode(audio_b64)
        except Exception as e:
            logger.debug(f"Base64 audio decode failed: {type(e).__name__}: {e}")
            return error_response("Invalid base64 audio data", 400)

        processor = get_voice_processor()
        result = await processor.process_audio(session_id, audio_data)

        status = 200 if result.get("success") else 400
        return json_response(result, status=status)

    async def _handle_synthesize_speech(self, session_id: str, handler: Any) -> HandlerResult:
        """
        Synthesize speech from text (Text-to-Speech).

        POST /api/v1/moltbot/voice/sessions/{session_id}/speak
        Body: {text: string}
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if not body:
            return error_response("Request body required", 400)

        text = body.get("text")
        if not text:
            return error_response("text is required", 400)

        processor = get_voice_processor()
        result = await processor.synthesize_speech(session_id, text)

        if not result.get("success"):
            return json_response(result, status=400)

        # Encode audio as base64 for JSON response
        audio = result.get("audio")
        if audio:
            result["audio"] = base64.b64encode(audio).decode("utf-8")

        return json_response(result)

    async def _handle_get_transcript(self, session_id: str, handler: Any) -> HandlerResult:
        """
        Get the full transcript for a session.

        GET /api/v1/moltbot/voice/sessions/{session_id}/transcript
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        processor = get_voice_processor()
        transcript = await processor.get_transcript(session_id)

        return json_response(
            {
                "session_id": session_id,
                "transcript": transcript,
                "total_entries": len(transcript),
            }
        )

    async def _handle_voice_stats(self, handler: Any) -> HandlerResult:
        """
        Get voice processor statistics.

        GET /api/v1/moltbot/voice/stats
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        processor = get_voice_processor()
        stats = await processor.get_stats()

        return json_response(stats)
