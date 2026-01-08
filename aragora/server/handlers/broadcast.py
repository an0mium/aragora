"""
Broadcast generation handler.

Endpoints:
- POST /api/debates/{id}/broadcast - Generate podcast audio from debate trace
"""

import logging
import os
from typing import Optional

from aragora.server.http_utils import run_async
from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    SAFE_SLUG_PATTERN,
)

logger = logging.getLogger(__name__)

# Optional imports for broadcast functionality
try:
    from aragora.broadcast import broadcast_debate
    BROADCAST_AVAILABLE = True
except ImportError:
    BROADCAST_AVAILABLE = False
    broadcast_debate = None

try:
    from mutagen.mp3 import MP3
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    MP3 = None


def _safe_error_message(e: Exception, context: str) -> str:
    """Generate safe error message without exposing internals."""
    error_type = type(e).__name__
    if os.environ.get("ARAGORA_DEBUG"):
        return f"{context}: {error_type}: {str(e)}"
    return f"{context} failed: {error_type}"


def _run_async(coro):
    """Run async coroutine in sync context."""
    return run_async(coro)


class BroadcastHandler(BaseHandler):
    """Handler for broadcast generation endpoint."""

    ROUTES = [
        "/api/debates/*/broadcast",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the request."""
        if path.startswith('/api/debates/') and path.endswith('/broadcast'):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler=None) -> Optional[HandlerResult]:
        """Handle GET requests (none for this handler)."""
        return None

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Handle POST requests."""
        # Broadcast generation
        if path.startswith('/api/debates/') and path.endswith('/broadcast'):
            debate_id, err = self.extract_path_param(path, 2, "debate_id", SAFE_SLUG_PATTERN)
            if err:
                return err
            return self._generate_broadcast(debate_id, handler)

        return None

    def _generate_broadcast(self, debate_id: str, handler) -> HandlerResult:
        """Generate podcast audio from a debate trace."""
        if not BROADCAST_AVAILABLE:
            return error_response("Broadcast module not available", status=503)

        storage = self.get_storage()
        if not storage:
            return error_response("Storage not available", status=503)

        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", status=503)

        audio_store = self.ctx.get("audio_store")

        # Look up debate
        actual_debate_id = debate_id
        debate = storage.get_debate(debate_id)
        if not debate:
            debate = storage.get_debate_by_slug(debate_id)
            if debate:
                actual_debate_id = debate.get("id", debate_id)

        if not debate:
            return error_response("Debate not found", status=404)

        # Check if audio already exists
        if audio_store and audio_store.exists(actual_debate_id):
            existing = audio_store.get_metadata(actual_debate_id)
            audio_path = audio_store.get_path(actual_debate_id)
            return json_response({
                "debate_id": actual_debate_id,
                "status": "exists",
                "audio_url": f"/audio/{actual_debate_id}.mp3",
                "audio_path": str(audio_path) if audio_path else None,
                "generated_at": existing.get("generated_at") if existing else None,
            })

        # Load trace
        from aragora.debate.traces import DebateTrace

        trace_path = nomic_dir / "traces" / f"{actual_debate_id}.json"
        if not trace_path.exists():
            trace_path = nomic_dir / "traces" / f"{debate_id}.json"

        if not trace_path.exists():
            return error_response("Debate trace not found", status=404)

        try:
            trace = DebateTrace.load(trace_path)
        except Exception as e:
            return error_response(f"Failed to load trace: {e}", status=500)

        try:
            # Generate broadcast asynchronously
            temp_output_path = _run_async(broadcast_debate(trace))

            if not temp_output_path:
                return error_response("Failed to generate audio", status=500)

            # Persist audio to storage
            if audio_store:
                try:
                    duration_seconds = None
                    if MUTAGEN_AVAILABLE:
                        try:
                            audio = MP3(temp_output_path)
                            duration_seconds = int(audio.info.length)
                        except Exception as e:
                            logger.warning(f"Failed to extract audio metadata from {temp_output_path}: {e}")

                    stored_path = audio_store.save(
                        debate_id=actual_debate_id,
                        audio_path=temp_output_path,
                        duration_seconds=duration_seconds,
                    )

                    # Update database with audio info
                    storage.update_audio(
                        debate_id=actual_debate_id,
                        audio_path=str(stored_path),
                    )

                    return json_response({
                        "debate_id": actual_debate_id,
                        "status": "generated",
                        "audio_url": f"/audio/{actual_debate_id}.mp3",
                        "audio_path": str(stored_path),
                        "duration_seconds": duration_seconds,
                    })

                except Exception as e:
                    logger.warning(f"Failed to persist audio: {e}")
                    return json_response({
                        "debate_id": actual_debate_id,
                        "status": "generated",
                        "audio_path": str(temp_output_path),
                        "duration_seconds": None,
                        "warning": "Audio generated but not persisted",
                    })
            else:
                return json_response({
                    "debate_id": actual_debate_id,
                    "status": "generated",
                    "audio_path": str(temp_output_path),
                })

        except Exception as e:
            return error_response(_safe_error_message(e, "broadcast_generation"), status=500)
