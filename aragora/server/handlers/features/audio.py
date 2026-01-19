"""
Audio and Podcast endpoint handlers.

Endpoints:
- GET /audio/{id}.mp3 - Serve audio file
- GET /api/podcast/feed.xml - iTunes-compatible RSS feed
- GET /api/podcast/episodes - JSON episode listing
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from aragora.server.errors import safe_error_message as _safe_error_message

from aragora.server.validation import validate_debate_id

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_host_header,
    get_int_param,
    json_response,
)
from ..utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for audio endpoints (10 requests per minute - resource-intensive TTS)
_audio_limiter = RateLimiter(requests_per_minute=10)

# Podcast feed limits
MAX_PODCAST_EPISODES = 200  # Prevent unbounded feed generation

# Type-only imports for type checking
if TYPE_CHECKING:
    from aragora.broadcast.rss_gen import (
        PodcastConfig as PodcastConfigType,
        PodcastEpisode as PodcastEpisodeType,
        PodcastFeedGenerator as PodcastFeedGeneratorType,
    )

# Optional imports for broadcast functionality
try:
    from aragora.broadcast.rss_gen import PodcastConfig, PodcastEpisode, PodcastFeedGenerator

    PODCAST_AVAILABLE: bool = True
except ImportError:
    PODCAST_AVAILABLE = False
    PodcastFeedGenerator: type[PodcastFeedGeneratorType] | None = None
    PodcastConfig: type[PodcastConfigType] | None = None
    PodcastEpisode: type[PodcastEpisodeType] | None = None


class AudioHandler(BaseHandler):
    """Handler for audio file serving and podcast endpoints."""

    ROUTES = [
        "/audio/*",
        "/api/podcast/feed.xml",
        "/api/podcast/episodes",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the request."""
        if path.startswith("/audio/") and path.endswith(".mp3"):
            return True
        if path in ("/api/podcast/feed.xml", "/api/podcast/episodes"):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler=None) -> Optional[HandlerResult]:
        """Handle GET requests."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _audio_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for audio endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Audio file serving
        if path.startswith("/audio/") and path.endswith(".mp3"):
            debate_id = path[7:-4]  # Extract ID from /audio/{id}.mp3
            return self._serve_audio(debate_id)

        # Podcast endpoints
        if path == "/api/podcast/feed.xml":
            return self._get_podcast_feed(handler)
        if path == "/api/podcast/episodes":
            limit = get_int_param(query_params, "limit", 50)
            return self._get_podcast_episodes(limit, handler)

        return None

    def _serve_audio(self, debate_id: str) -> HandlerResult:
        """Serve audio file for a debate with security checks."""
        audio_store = self.ctx.get("audio_store")
        if not audio_store:
            return error_response("Audio storage not configured", status=404)

        # Validate debate ID to prevent path traversal
        valid, error = validate_debate_id(debate_id)
        if not valid:
            return error_response(error, status=400)

        # Get audio file path
        audio_path = audio_store.get_path(debate_id)
        if not audio_path or not audio_path.exists():
            return error_response("Audio not found", status=404)

        # Security: Ensure file is within audio storage directory
        try:
            audio_path_resolved = audio_path.resolve()
            storage_dir_resolved = audio_store.storage_dir.resolve()
            if not str(audio_path_resolved).startswith(str(storage_dir_resolved)):
                logger.warning(f"Audio path traversal attempt: {debate_id}")
                return error_response("Invalid path", status=400)
        except (OSError, ValueError) as e:
            logger.warning(f"Path validation error for {debate_id}: {e}")
            return error_response("Path validation failed", status=400)

        try:
            content = audio_path.read_bytes()
            return HandlerResult(
                status_code=200,
                content_type="audio/mpeg",
                body=content,
                headers={
                    "Content-Length": str(len(content)),
                    "Accept-Ranges": "bytes",
                    "Cache-Control": "public, max-age=86400",
                },
            )
        except Exception as e:
            logger.error(f"Failed to serve audio {debate_id}: {e}")
            return error_response("Failed to read audio file", status=500)

    def _get_podcast_feed(self, handler) -> HandlerResult:
        """Generate iTunes-compatible podcast RSS feed."""
        audio_store = self.ctx.get("audio_store")
        if not audio_store:
            return error_response("Audio storage not configured", status=503)

        if not PODCAST_AVAILABLE:
            return error_response("Podcast module not available", status=503)

        try:
            storage = self.get_storage()
            debates_with_audio = []
            episode_count = 0

            for audio_meta in audio_store.list_all():
                # Limit podcast feed size to prevent unbounded growth
                if episode_count >= MAX_PODCAST_EPISODES:
                    break

                debate_id = audio_meta.get("debate_id")
                if not debate_id:
                    continue

                debate = storage.get_debate(debate_id) if storage else None
                if not debate:
                    continue

                audio_path = audio_store.get_path(debate_id)
                if not audio_path or not audio_path.exists():
                    continue

                debates_with_audio.append(
                    {
                        "debate_id": debate_id,
                        "task": debate.get("task", "Untitled Debate"),
                        "agents": debate.get("agents", []),
                        "verdict": debate.get("verdict"),
                        "created_at": debate.get("created_at", audio_meta.get("generated_at")),
                        "duration_seconds": audio_meta.get("duration_seconds", 0),
                        "file_size_bytes": audio_meta.get("file_size_bytes", 0),
                    }
                )
                episode_count += 1

            # Generate RSS feed
            # Assertions for type narrowing - we already checked PODCAST_AVAILABLE above
            assert PodcastConfig is not None
            assert PodcastFeedGenerator is not None
            assert PodcastEpisode is not None
            config = PodcastConfig()
            generator = PodcastFeedGenerator(config)

            # Get host for URLs
            host = get_host_header(handler)
            scheme = (
                "https"
                if handler and handler.headers.get("X-Forwarded-Proto") == "https"
                else "http"
            )

            episodes = []
            for i, debate in enumerate(debates_with_audio):
                audio_url = f"{scheme}://{host}/audio/{debate['debate_id']}.mp3"

                episode = PodcastEpisode(
                    guid=debate["debate_id"],
                    title=debate["task"],
                    description=f"AI debate: {debate['task']}",
                    content=f"AI debate: {debate['task']}",
                    audio_url=audio_url,
                    pub_date=debate["created_at"],
                    duration_seconds=debate["duration_seconds"],
                    file_size_bytes=debate["file_size_bytes"],
                    episode_number=len(debates_with_audio) - i,
                )
                episodes.append(episode)

            feed_xml = generator.generate_feed(episodes)

            return HandlerResult(
                status_code=200,
                content_type="application/rss+xml; charset=utf-8",
                body=feed_xml.encode("utf-8"),
                headers={
                    "Cache-Control": "public, max-age=300",
                },
            )

        except Exception as e:
            logger.error(f"Failed to generate podcast feed: {e}")
            return error_response(_safe_error_message(e, "podcast_feed"), status=500)

    def _get_podcast_episodes(self, limit: int, handler) -> HandlerResult:
        """Get podcast episodes as JSON."""
        audio_store = self.ctx.get("audio_store")
        if not audio_store:
            return error_response("Audio storage not configured", status=503)

        try:
            storage = self.get_storage()
            episodes = []

            host = get_host_header(handler)
            scheme = (
                "https"
                if handler and handler.headers.get("X-Forwarded-Proto") == "https"
                else "http"
            )

            for audio_meta in audio_store.list_all()[:limit]:
                debate_id = audio_meta.get("debate_id")
                if not debate_id:
                    continue

                debate = storage.get_debate(debate_id) if storage else None

                episodes.append(
                    {
                        "debate_id": debate_id,
                        "task": debate.get("task", "Untitled") if debate else "Unknown",
                        "agents": debate.get("agents", []) if debate else [],
                        "audio_url": f"{scheme}://{host}/audio/{debate_id}.mp3",
                        "duration_seconds": audio_meta.get("duration_seconds"),
                        "file_size_bytes": audio_meta.get("file_size_bytes"),
                        "generated_at": audio_meta.get("generated_at"),
                    }
                )

            return json_response(
                {
                    "episodes": episodes,
                    "count": len(episodes),
                    "feed_url": "/api/podcast/feed.xml",
                }
            )

        except Exception as e:
            logger.error(f"Failed to get podcast episodes: {e}")
            return error_response(_safe_error_message(e, "podcast_episodes"), status=500)
