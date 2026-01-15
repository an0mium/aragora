"""
Broadcast generation handler.

Endpoints:
- POST /api/debates/{id}/broadcast - Generate podcast audio from debate trace
- POST /api/debates/{id}/broadcast/full - Run full broadcast pipeline
- GET /api/podcast/feed.xml - Get RSS podcast feed
"""

from __future__ import annotations

import logging
from typing import Any, Coroutine, Optional, TypeVar

T = TypeVar("T")

from aragora.server.error_utils import safe_error_message as _safe_error_message
from aragora.server.http_utils import run_async
from aragora.server.middleware.rate_limit import rate_limit

from ..base import (
    SAFE_SLUG_PATTERN,
    BaseHandler,
    HandlerResult,
    error_response,
    get_bool_param,
    get_int_param,
    get_string_param,
    json_response,
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
    from aragora.broadcast.pipeline import BroadcastOptions, BroadcastPipeline

    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    BroadcastPipeline = None  # type: ignore[misc, assignment]
    BroadcastOptions = None  # type: ignore[misc, assignment]

try:
    from mutagen.mp3 import MP3

    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    MP3 = None


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run async coroutine in sync context."""
    return run_async(coro)


class BroadcastHandler(BaseHandler):
    """Handler for broadcast generation endpoints."""

    ROUTES = [
        "/api/debates/*/broadcast",
        "/api/debates/*/broadcast/full",
        "/api/podcast/feed.xml",
    ]

    # Cached pipeline instance
    _pipeline: Optional["BroadcastPipeline"] = None

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the request."""
        if path.startswith("/api/debates/") and "/broadcast" in path:
            return True
        if path == "/api/podcast/feed.xml":
            return True
        return False

    def handle(self, path: str, query_params: dict, handler=None) -> Optional[HandlerResult]:
        """Handle GET requests."""
        if path == "/api/podcast/feed.xml":
            return self._get_rss_feed()
        return None

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Handle POST requests."""
        # Full pipeline
        if path.startswith("/api/debates/") and path.endswith("/broadcast/full"):
            debate_id, err = self.extract_path_param(path, 2, "debate_id", SAFE_SLUG_PATTERN)
            if err:
                return err
            return self._run_full_pipeline(debate_id, query_params, handler)

        # Basic broadcast generation
        if path.startswith("/api/debates/") and path.endswith("/broadcast"):
            debate_id, err = self.extract_path_param(path, 2, "debate_id", SAFE_SLUG_PATTERN)
            if err:
                return err
            return self._generate_broadcast(debate_id, handler)

        return None

    def _get_pipeline(self) -> Optional["BroadcastPipeline"]:
        """Get or create the broadcast pipeline."""
        if not PIPELINE_AVAILABLE:
            return None

        if self._pipeline is None:
            nomic_dir = self.get_nomic_dir()
            if nomic_dir:
                audio_store = self.ctx.get("audio_store")
                self._pipeline = BroadcastPipeline(
                    nomic_dir=nomic_dir,
                    audio_store=audio_store,
                )
        return self._pipeline

    @rate_limit(requests_per_minute=2, burst=1, limiter_name="broadcast_full_pipeline")
    def _run_full_pipeline(self, debate_id: str, query_params: dict, handler) -> HandlerResult:
        """Run the full broadcast pipeline with all options.

        Query params:
            video: bool - Generate video (default: false)
            title: str - Custom title
            description: str - Custom description
            episode_number: int - Episode number for RSS
        """
        pipeline = self._get_pipeline()
        if not pipeline:
            return error_response("Broadcast pipeline not available", status=503)

        # Parse options from query params
        options = BroadcastOptions(
            audio_enabled=True,
            video_enabled=get_bool_param(query_params, "video", False),
            generate_rss_episode=get_bool_param(query_params, "rss", True),
            custom_title=get_string_param(query_params, "title"),
            custom_description=get_string_param(query_params, "description"),
            episode_number=get_int_param(query_params, "episode_number"),
        )

        try:
            result = _run_async(pipeline.run(debate_id, options))

            return json_response(
                {
                    "debate_id": result.debate_id,
                    "success": result.success,
                    "audio_path": str(result.audio_path) if result.audio_path else None,
                    "audio_url": f"/audio/{debate_id}.mp3" if result.audio_path else None,
                    "video_path": str(result.video_path) if result.video_path else None,
                    "video_url": f"/video/{debate_id}.mp4" if result.video_path else None,
                    "rss_episode_guid": result.rss_episode_guid,
                    "duration_seconds": result.duration_seconds,
                    "steps_completed": result.steps_completed,
                    "generated_at": result.generated_at,
                    "error": result.error_message,
                }
            )
        except Exception as e:
            logger.error(f"Pipeline failed for {debate_id}: {e}", exc_info=True)
            return error_response(_safe_error_message(e, "broadcast_pipeline"), status=500)

    def _get_rss_feed(self) -> HandlerResult:
        """Get the RSS podcast feed."""
        pipeline = self._get_pipeline()
        if not pipeline:
            return error_response("Broadcast pipeline not available", status=503)

        feed_xml = pipeline.get_rss_feed()
        if not feed_xml:
            # Return empty feed if no episodes yet
            try:
                from aragora.broadcast.rss_gen import PodcastConfig, PodcastFeedGenerator

                config = PodcastConfig()
                generator = PodcastFeedGenerator(config)
                feed_xml = generator.generate_feed([])
            except ImportError:
                return error_response("RSS generator not available", status=503)

        # Return XML with correct content type
        return HandlerResult(
            status_code=200,
            content_type="application/rss+xml; charset=utf-8",
            body=feed_xml.encode("utf-8"),
        )

    @rate_limit(requests_per_minute=3, burst=2, limiter_name="broadcast_generation")
    def _generate_broadcast(self, debate_id: str, handler) -> HandlerResult:
        """Generate podcast audio from a debate trace.

        Rate limited to 3 requests/min due to high CPU usage for TTS.
        """
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
            return json_response(
                {
                    "debate_id": actual_debate_id,
                    "status": "exists",
                    "audio_url": f"/audio/{actual_debate_id}.mp3",
                    "audio_path": str(audio_path) if audio_path else None,
                    "generated_at": existing.get("generated_at") if existing else None,
                }
            )

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
            return error_response(_safe_error_message(e, "load trace"), status=500)

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
                            logger.warning(
                                f"Failed to extract audio metadata from {temp_output_path}: {e}"
                            )

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

                    return json_response(
                        {
                            "debate_id": actual_debate_id,
                            "status": "generated",
                            "audio_url": f"/audio/{actual_debate_id}.mp3",
                            "audio_path": str(stored_path),
                            "duration_seconds": duration_seconds,
                        }
                    )

                except Exception as e:
                    logger.warning(f"Failed to persist audio: {e}")
                    return json_response(
                        {
                            "debate_id": actual_debate_id,
                            "status": "generated",
                            "audio_path": str(temp_output_path),
                            "duration_seconds": None,
                            "warning": "Audio generated but not persisted",
                        }
                    )
            else:
                return json_response(
                    {
                        "debate_id": actual_debate_id,
                        "status": "generated",
                        "audio_path": str(temp_output_path),
                    }
                )

        except Exception as e:
            return error_response(_safe_error_message(e, "broadcast_generation"), status=500)
