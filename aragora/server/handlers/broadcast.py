"""
Broadcast handler for audio, podcast, and social media publishing endpoints.

Handles:
- Audio file serving (GET /audio/{id}.mp3)
- Podcast RSS feed (GET /api/podcast/feed.xml)
- Podcast episodes listing (GET /api/podcast/episodes)
- YouTube OAuth (GET /api/youtube/auth, callback, status)
- Broadcast generation (POST /api/debates/{id}/broadcast)
- Social publishing (POST /api/debates/{id}/publish/twitter, /publish/youtube)
"""

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    validate_debate_id,
)

logger = logging.getLogger(__name__)

# OAuth state storage for CSRF protection (with TTL cleanup)
_oauth_states: dict[str, float] = {}  # state -> expiry_time
_oauth_states_lock = threading.Lock()
_OAUTH_STATE_TTL = 600  # 10 minutes

# Allowed hosts for OAuth redirect URI (prevent open redirect)
ALLOWED_OAUTH_HOSTS = frozenset(
    h.strip() for h in os.getenv(
        'ARAGORA_ALLOWED_OAUTH_HOSTS',
        'localhost:8080,127.0.0.1:8080'
    ).split(',')
)


def _store_oauth_state(state: str) -> None:
    """Store OAuth state for later validation."""
    with _oauth_states_lock:
        # Cleanup expired states
        now = time.time()
        expired = [s for s, exp in _oauth_states.items() if exp < now]
        for s in expired:
            del _oauth_states[s]
        # Store new state
        _oauth_states[state] = now + _OAUTH_STATE_TTL


def _validate_oauth_state(state: str) -> bool:
    """Validate and consume OAuth state (one-time use)."""
    with _oauth_states_lock:
        if state in _oauth_states:
            expiry = _oauth_states.pop(state)
            return time.time() < expiry
        return False

# Optional imports for broadcast functionality
try:
    from aragora.broadcast import broadcast_debate
    from aragora.broadcast.storage import AudioFileStore
    from aragora.broadcast.rss_gen import PodcastFeedGenerator, PodcastConfig, PodcastEpisode
    BROADCAST_AVAILABLE = True
except ImportError:
    BROADCAST_AVAILABLE = False
    broadcast_debate = None
    AudioFileStore = None
    PodcastFeedGenerator = None
    PodcastConfig = None
    PodcastEpisode = None

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
    """Run async coroutine in sync context.

    Uses asyncio.run() which properly creates and closes an event loop,
    avoiding resource leaks and deprecation warnings.
    """
    return asyncio.run(coro)


class BroadcastHandler(BaseHandler):
    """Handler for broadcast, podcast, and social media endpoints."""

    # Routes handled by this handler
    ROUTES = [
        "/audio/*",
        "/api/podcast/*",
        "/api/youtube/*",
        "/api/debates/*/broadcast",
        "/api/debates/*/publish/*",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the request."""
        if path.startswith('/audio/') and path.endswith('.mp3'):
            return True
        if path.startswith('/api/podcast/'):
            return True
        if path.startswith('/api/youtube/'):
            return True
        if path.startswith('/api/debates/') and (
            path.endswith('/broadcast') or '/publish/' in path
        ):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler=None) -> Optional[HandlerResult]:
        """Handle GET requests."""
        # Audio file serving
        if path.startswith('/audio/') and path.endswith('.mp3'):
            debate_id = path[7:-4]  # Extract ID from /audio/{id}.mp3
            return self._serve_audio(debate_id)

        # Podcast endpoints
        if path == '/api/podcast/feed.xml':
            return self._get_podcast_feed(handler)
        if path == '/api/podcast/episodes':
            limit = get_int_param(query_params, 'limit', 50)
            return self._get_podcast_episodes(limit, handler)

        # YouTube OAuth endpoints
        if path == '/api/youtube/auth':
            return self._get_youtube_auth_url(handler)
        if path == '/api/youtube/callback':
            code = query_params.get('code')
            state = query_params.get('state')
            return self._handle_youtube_callback(code, state, handler)
        if path == '/api/youtube/status':
            return self._get_youtube_status()

        return None

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Handle POST requests."""
        # Broadcast generation
        if path.startswith('/api/debates/') and path.endswith('/broadcast'):
            parts = path.split('/')
            if len(parts) >= 4:
                debate_id = parts[3]
                return self._generate_broadcast(debate_id, handler)

        # Twitter publishing
        if path.startswith('/api/debates/') and path.endswith('/publish/twitter'):
            parts = path.split('/')
            if len(parts) >= 5:
                debate_id = parts[3]
                return self._publish_to_twitter(debate_id, handler)

        # YouTube publishing
        if path.startswith('/api/debates/') and path.endswith('/publish/youtube'):
            parts = path.split('/')
            if len(parts) >= 5:
                debate_id = parts[3]
                return self._publish_to_youtube(debate_id, handler)

        return None

    # === Audio Endpoints ===

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
        except Exception:
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

    # === Podcast Endpoints ===

    def _get_podcast_feed(self, handler) -> HandlerResult:
        """Generate iTunes-compatible podcast RSS feed."""
        audio_store = self.ctx.get("audio_store")
        if not audio_store:
            return error_response("Audio storage not configured", status=503)

        if not BROADCAST_AVAILABLE:
            return error_response("Broadcast module not available", status=503)

        try:
            storage = self.get_storage()
            debates_with_audio = []

            for audio_meta in audio_store.list_all():
                debate_id = audio_meta.get("debate_id")
                if not debate_id:
                    continue

                debate = storage.get_debate(debate_id) if storage else None
                if not debate:
                    continue

                audio_path = audio_store.get_path(debate_id)
                if not audio_path or not audio_path.exists():
                    continue

                debates_with_audio.append({
                    "debate_id": debate_id,
                    "task": debate.get("task", "Untitled Debate"),
                    "agents": debate.get("agents", []),
                    "verdict": debate.get("verdict"),
                    "created_at": debate.get("created_at", audio_meta.get("generated_at")),
                    "duration_seconds": audio_meta.get("duration_seconds", 0),
                    "file_size_bytes": audio_meta.get("file_size_bytes", 0),
                })

            # Generate RSS feed
            config = PodcastConfig()
            generator = PodcastFeedGenerator(config)

            # Get host for URLs
            host = handler.headers.get('Host', 'localhost:8080') if handler else 'localhost:8080'
            scheme = 'https' if handler and handler.headers.get('X-Forwarded-Proto') == 'https' else 'http'

            for i, debate in enumerate(debates_with_audio):
                audio_url = f"{scheme}://{host}/audio/{debate['debate_id']}.mp3"

                episode = PodcastEpisode(
                    guid=debate["debate_id"],
                    title=debate["task"],
                    description=f"AI debate: {debate['task']}",
                    audio_url=audio_url,
                    pub_date=debate["created_at"],
                    duration_seconds=debate["duration_seconds"],
                    file_size_bytes=debate["file_size_bytes"],
                    episode_number=len(debates_with_audio) - i,
                )
                generator.add_episode(episode)

            feed_xml = generator.generate(f"{scheme}://{host}")

            return HandlerResult(
                status_code=200,
                content_type="application/rss+xml; charset=utf-8",
                body=feed_xml.encode('utf-8'),
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

            host = handler.headers.get('Host', 'localhost:8080') if handler else 'localhost:8080'
            scheme = 'https' if handler and handler.headers.get('X-Forwarded-Proto') == 'https' else 'http'

            for audio_meta in audio_store.list_all()[:limit]:
                debate_id = audio_meta.get("debate_id")
                if not debate_id:
                    continue

                debate = storage.get_debate(debate_id) if storage else None

                episodes.append({
                    "debate_id": debate_id,
                    "task": debate.get("task", "Untitled") if debate else "Unknown",
                    "agents": debate.get("agents", []) if debate else [],
                    "audio_url": f"{scheme}://{host}/audio/{debate_id}.mp3",
                    "duration_seconds": audio_meta.get("duration_seconds"),
                    "file_size_bytes": audio_meta.get("file_size_bytes"),
                    "generated_at": audio_meta.get("generated_at"),
                })

            return json_response({
                "episodes": episodes,
                "count": len(episodes),
                "feed_url": "/api/podcast/feed.xml",
            })

        except Exception as e:
            logger.error(f"Failed to get podcast episodes: {e}")
            return error_response(_safe_error_message(e, "podcast_episodes"), status=500)

    # === YouTube Endpoints ===

    def _get_youtube_auth_url(self, handler) -> HandlerResult:
        """Get YouTube OAuth authorization URL."""
        youtube = self.ctx.get("youtube_connector")
        if not youtube:
            return error_response("YouTube connector not initialized", status=500)

        if not youtube.client_id:
            return json_response({
                "error": "YouTube client ID not configured",
                "hint": "Set YOUTUBE_CLIENT_ID environment variable"
            }, status=400)

        # Validate Host header against whitelist (prevent open redirect)
        host = handler.headers.get('Host', 'localhost:8080') if handler else 'localhost:8080'
        if host not in ALLOWED_OAUTH_HOSTS:
            logger.warning(f"OAuth auth request with untrusted host: {host}")
            return error_response("Untrusted host for OAuth redirect", status=400)

        scheme = 'https' if handler and handler.headers.get('X-Forwarded-Proto') == 'https' else 'http'
        redirect_uri = f"{scheme}://{host}/api/youtube/callback"

        import secrets
        state = secrets.token_urlsafe(32)

        # Store state for CSRF validation in callback
        _store_oauth_state(state)

        auth_url = youtube.get_auth_url(redirect_uri, state)
        return json_response({
            "auth_url": auth_url,
            "state": state,
        })

    def _handle_youtube_callback(self, code: Optional[str], state: Optional[str], handler) -> HandlerResult:
        """Handle YouTube OAuth callback."""
        if not code:
            return error_response("Missing authorization code", status=400)
        if not state:
            return error_response("Missing state parameter", status=400)

        # Validate state parameter (CSRF protection)
        if not _validate_oauth_state(state):
            logger.warning(f"OAuth callback with invalid/expired state")
            return error_response("Invalid or expired state parameter", status=400)

        youtube = self.ctx.get("youtube_connector")
        if not youtube:
            return error_response("YouTube connector not initialized", status=500)

        # Validate Host header against whitelist (prevent open redirect)
        host = handler.headers.get('Host', 'localhost:8080') if handler else 'localhost:8080'
        if host not in ALLOWED_OAUTH_HOSTS:
            logger.warning(f"OAuth callback with untrusted host: {host}")
            return error_response("Untrusted host for OAuth redirect", status=400)

        scheme = 'https' if handler and handler.headers.get('X-Forwarded-Proto') == 'https' else 'http'
        redirect_uri = f"{scheme}://{host}/api/youtube/callback"

        try:
            result = _run_async(youtube.exchange_code(code, redirect_uri))

            if result.get("success"):
                return json_response({
                    "success": True,
                    "message": "YouTube authentication successful",
                })
            else:
                return json_response({
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                }, status=400)

        except Exception as e:
            logger.error(f"YouTube OAuth callback failed: {e}")
            return error_response(_safe_error_message(e, "youtube_callback"), status=500)

    def _get_youtube_status(self) -> HandlerResult:
        """Get YouTube connector status."""
        youtube = self.ctx.get("youtube_connector")
        if not youtube:
            return json_response({
                "configured": False,
                "error": "YouTube connector not initialized",
            })

        return json_response({
            "configured": youtube.is_configured,
            "has_client_id": bool(youtube.client_id),
            "has_client_secret": bool(youtube.client_secret),
            "has_refresh_token": bool(youtube.refresh_token),
            "quota_remaining": youtube.rate_limiter.remaining_quota,
            "circuit_breaker_open": youtube.circuit_breaker.is_open,
        })

    # === Broadcast Generation ===

    def _generate_broadcast(self, debate_id: str, handler) -> HandlerResult:
        """Generate podcast audio from a debate trace."""
        if not BROADCAST_AVAILABLE:
            return error_response("Broadcast module not available", status=503)

        valid, error = validate_debate_id(debate_id)
        if not valid:
            return error_response(error, status=400)

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

    # === Social Publishing ===

    def _publish_to_twitter(self, debate_id: str, handler) -> HandlerResult:
        """Publish debate to Twitter/X."""
        twitter = self.ctx.get("twitter_connector")
        if not twitter:
            return error_response("Twitter connector not initialized", status=500)

        if not twitter.is_configured:
            return json_response({
                "error": "Twitter API credentials not configured",
                "hint": "Set TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET"
            }, status=400)

        valid, error = validate_debate_id(debate_id)
        if not valid:
            return error_response(error, status=400)

        storage = self.get_storage()
        if not storage:
            return error_response("Storage not available", status=503)

        debate = storage.get_debate(debate_id)
        if not debate:
            debate = storage.get_debate_by_slug(debate_id)
        if not debate:
            return error_response("Debate not found", status=404)

        # Read POST body for options
        body = self.read_json_body(handler) if handler else {}
        if body is None:
            return error_response("Invalid JSON body", status=400)

        options = body or {}
        include_audio = options.get("include_audio_link", True)
        thread_mode = options.get("thread_mode", False)

        host = handler.headers.get('Host', 'localhost:8080') if handler else 'localhost:8080'
        scheme = 'https' if handler and handler.headers.get('X-Forwarded-Proto') == 'https' else 'http'
        debate_url = f"{scheme}://{host}/debates/{debate_id}"

        audio_store = self.ctx.get("audio_store")
        audio_url = None
        if include_audio and audio_store and audio_store.exists(debate_id):
            audio_url = f"{scheme}://{host}/audio/{debate_id}.mp3"

        try:
            from aragora.connectors.twitter_poster import DebateContentFormatter

            formatter = DebateContentFormatter()

            if thread_mode:
                tweets = formatter.format_as_thread(
                    task=debate.get("task", ""),
                    agents=debate.get("agents", []),
                    verdict=debate.get("verdict"),
                    debate_url=audio_url or debate_url,
                )
                result = _run_async(twitter.post_thread(tweets))
            else:
                tweet_text = formatter.format_single_tweet(
                    task=debate.get("task", ""),
                    agents=debate.get("agents", []),
                    verdict=debate.get("verdict"),
                    audio_url=audio_url,
                    debate_url=debate_url if not audio_url else None,
                )
                result = _run_async(twitter.post_tweet(tweet_text))

            if result.get("success"):
                return json_response({
                    "success": True,
                    "debate_id": debate_id,
                    "tweet_id": result.get("tweet_id"),
                    "thread_ids": result.get("thread_ids"),
                    "url": result.get("url"),
                })
            else:
                return json_response({
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                }, status=500)

        except Exception as e:
            logger.error(f"Failed to publish to Twitter: {e}")
            return error_response(_safe_error_message(e, "twitter_publish"), status=500)

    def _publish_to_youtube(self, debate_id: str, handler) -> HandlerResult:
        """Publish debate to YouTube."""
        youtube = self.ctx.get("youtube_connector")
        if not youtube:
            return error_response("YouTube connector not initialized", status=500)

        if not youtube.is_configured:
            return json_response({
                "error": "YouTube API credentials not configured",
                "hint": "Set YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN"
            }, status=400)

        valid, error = validate_debate_id(debate_id)
        if not valid:
            return error_response(error, status=400)

        storage = self.get_storage()
        if not storage:
            return error_response("Storage not available", status=503)

        debate = storage.get_debate(debate_id)
        if not debate:
            debate = storage.get_debate_by_slug(debate_id)
        if not debate:
            return error_response("Debate not found", status=404)

        # Check quota
        if not youtube.rate_limiter.can_upload():
            return json_response({
                "error": "YouTube daily quota exceeded",
                "quota_remaining": youtube.rate_limiter.remaining_quota,
            }, status=429)

        audio_store = self.ctx.get("audio_store")
        if not audio_store or not audio_store.exists(debate_id):
            return json_response({
                "error": "No audio found for this debate",
                "hint": "Generate audio first using POST /api/debates/{id}/broadcast"
            }, status=400)

        # Read POST body for options
        body = self.read_json_body(handler) if handler else {}
        if body is None:
            return error_response("Invalid JSON body", status=400)

        options = body or {}
        custom_title = options.get("title")
        custom_description = options.get("description")
        tags = options.get("tags", [])
        privacy = options.get("privacy", "public")

        task = debate.get("task", "AI Debate")
        agents = debate.get("agents", [])

        audio_path = audio_store.get_path(debate_id)
        if not audio_path:
            return error_response("Audio file not found", status=404)

        try:
            video_generator = self.ctx.get("video_generator")
            if video_generator:
                try:
                    video_path = video_generator.generate_waveform_video(audio_path)
                except Exception as e:
                    logger.debug(f"Waveform video generation failed, using static fallback: {e}")
                    video_path = video_generator.generate_static_video(
                        audio_path, task, agents
                    )
            else:
                return error_response("Video generator not available", status=503)

            from aragora.connectors.youtube_uploader import YouTubeVideoMetadata

            metadata = YouTubeVideoMetadata(
                title=custom_title or f"AI Debate: {task[:80]}",
                description=custom_description or f"Multi-agent AI debate on: {task}\n\nAgents: {', '.join(agents)}",
                tags=tags or ["AI", "debate", "agents", "artificial intelligence"],
                privacy_status=privacy,
            )

            logger.info(f"Uploading video to YouTube: {metadata.title}")
            result = _run_async(youtube.upload(video_path, metadata))

            if result.get("success"):
                return json_response({
                    "success": True,
                    "debate_id": debate_id,
                    "video_id": result.get("video_id"),
                    "url": result.get("url"),
                    "quota_remaining": youtube.rate_limiter.remaining_quota,
                })
            else:
                return json_response({
                    "success": False,
                    "error": result.get("error", "Upload failed"),
                }, status=500)

        except Exception as e:
            logger.error(f"Failed to publish to YouTube: {e}")
            return error_response(_safe_error_message(e, "youtube_publish"), status=500)
