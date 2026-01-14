"""
Unified server combining HTTP API and WebSocket streaming.

Provides a single entry point for:
- HTTP API at /api/* endpoints
- WebSocket streaming at ws://host:port/ws
- Static file serving for the live dashboard
"""

import asyncio
import json
import re
import sqlite3
from collections import deque
from html import escape as html_escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from aragora.agents.grounded import (
        MomentDetector,  # type: ignore[attr-defined]
        PositionLedger,  # type: ignore[attr-defined]
    )
    from aragora.agents.personas import PersonaManager
    from aragora.agents.truth_grounding import PositionTracker  # type: ignore[attr-defined]
    from aragora.billing.usage import UsageTracker
    from aragora.broadcast.storage import AudioFileStore
    from aragora.broadcast.video_gen import VideoGenerator
    from aragora.connectors.twitter_poster import TwitterPosterConnector
    from aragora.connectors.youtube_uploader import YouTubeUploaderConnector
    from aragora.debate.embeddings import DebateEmbeddingsDatabase
    from aragora.insights.flip_detector import FlipDetector
    from aragora.insights.store import InsightStore
    from aragora.memory.consensus import (  # type: ignore[attr-defined]
        ConsensusMemory,
        DissentRetriever,
    )
    from aragora.persistence.supabase import SupabaseClient
    from aragora.ranking.elo import EloSystem
    from aragora.server.documents import DocumentStore
    from aragora.server.middleware.rate_limit import RateLimitResult
    from aragora.storage import UserStore
import logging

# For ad-hoc debates
import threading
import time
import uuid
from urllib.parse import parse_qs, urlparse

from .auth import auth_config, check_auth
from .cors_config import cors_config
from .middleware.tracing import TRACE_ID_HEADER, TracingMiddleware, get_trace_id
from .prometheus import record_http_request
from .storage import DebateStorage
from .stream import (
    DebateStreamServer,
    SyncEventEmitter,
)

# Configure module logger
logger = logging.getLogger(__name__)

# Import centralized config and error utilities
from aragora.config import ALLOWED_AGENT_TYPES
from aragora.server.debate_controller import DebateController
from aragora.server.debate_factory import DebateFactory
from aragora.server.debate_utils import get_active_debates

# Import utilities from extracted modules
from aragora.server.http_utils import validate_query_params as _validate_query_params
from aragora.server.validation import safe_query_float, safe_query_int

# DoS protection limits
MAX_MULTIPART_PARTS = 10
# Maximum content length for POST requests (100MB - DoS protection)
MAX_CONTENT_LENGTH = 100 * 1024 * 1024
# Maximum content length for JSON API requests (10MB)
MAX_JSON_CONTENT_LENGTH = 10 * 1024 * 1024

# Note: SAFE_ID_PATTERN imported from validation.py (prevent path traversal)

# Trusted proxies for X-Forwarded-For header validation
# Only trust X-Forwarded-For if request comes from these IPs
import os

TRUSTED_PROXIES = frozenset(
    p.strip() for p in os.getenv("ARAGORA_TRUSTED_PROXIES", "127.0.0.1,::1,localhost").split(",")
)


# Import from initialization module (subsystem init functions)
# Modular HTTP handlers via registry mixin
from aragora.server.handler_registry import HandlerRegistryMixin
from aragora.server.initialization import (
    ROUTING_AVAILABLE,
    AgentProfile,
    AgentSelector,
    TaskRequirements,
    init_consensus_memory,
    init_debate_embeddings,
    init_elo_system,
    init_flip_detector,
    init_insight_store,
    init_moment_detector,
    init_persistence,
    init_persona_manager,
    init_position_ledger,
)

# Server startup time for uptime tracking
_server_start_time: float = time.time()


class UnifiedHandler(HandlerRegistryMixin, BaseHTTPRequestHandler):  # type: ignore[misc]
    """HTTP handler with API endpoints and static file serving.

    Handler routing is provided by HandlerRegistryMixin from handler_registry.py.
    """

    storage: Optional[DebateStorage] = None
    static_dir: Optional[Path] = None
    stream_emitter: Optional[SyncEventEmitter] = None
    tracing: TracingMiddleware = TracingMiddleware(service_name="aragora-api")
    nomic_state_file: Optional[Path] = None
    persistence: Optional["SupabaseClient"] = None
    insight_store: Optional["InsightStore"] = None
    elo_system: Optional["EloSystem"] = None
    document_store: Optional["DocumentStore"] = None
    audio_store: Optional["AudioFileStore"] = None
    twitter_connector: Optional["TwitterPosterConnector"] = None
    youtube_connector: Optional["YouTubeUploaderConnector"] = None
    video_generator: Optional["VideoGenerator"] = None
    flip_detector: Optional["FlipDetector"] = None
    persona_manager: Optional["PersonaManager"] = None
    debate_embeddings: Optional["DebateEmbeddingsDatabase"] = None
    position_tracker: Optional["PositionTracker"] = None
    position_ledger: Optional["PositionLedger"] = None
    consensus_memory: Optional["ConsensusMemory"] = None
    dissent_retriever: Optional["DissentRetriever"] = None
    moment_detector: Optional["MomentDetector"] = None
    user_store: Optional["UserStore"] = None
    usage_tracker: Optional["UsageTracker"] = None

    # Note: Modular HTTP handlers are provided by HandlerRegistryMixin
    # Handler instance variables (_system_handler, etc.) and _handlers_initialized
    # are inherited from the mixin along with _init_handlers() and _try_modular_handler()

    # Debate controller and factory (initialized lazily)
    # Note: DebateController manages its own ThreadPoolExecutor with proper locking
    _debate_controller: Optional[DebateController] = None
    _debate_factory: Optional[DebateFactory] = None

    # Upload rate limiting (IP-based, independent of auth)
    # Uses deque with maxlen to prevent unbounded memory growth
    _upload_counts: Dict[str, deque] = {}  # IP -> deque of upload timestamps
    _upload_counts_lock = threading.Lock()
    MAX_UPLOADS_PER_MINUTE = 5  # Maximum uploads per IP per minute
    MAX_UPLOADS_PER_HOUR = 30  # Maximum uploads per IP per hour
    _MAX_UPLOAD_TIMESTAMPS = 30  # Max timestamps to keep per IP (matches hourly limit)

    # Request logging for observability
    _request_log_enabled = True  # Can be disabled via environment
    _slow_request_threshold_ms = 1000  # Log warning for requests slower than this

    # Per-request rate limit result (set by _check_tier_rate_limit)
    # Used to include X-RateLimit-* headers in all responses
    _rate_limit_result: Optional["RateLimitResult"] = None

    def send_error(
        self, code: int, message: str | None = None, explain: str | None = None
    ) -> None:
        """Override send_error to include CORS headers.

        The default BaseHTTPRequestHandler.send_error() doesn't include CORS headers,
        which causes browsers to block error responses for cross-origin requests.
        """
        # Send response headers first
        self.send_response(code, message)
        self.send_header("Content-Type", self.error_content_type)
        # Add CORS headers so browser can read error responses
        self._add_cors_headers()
        self._add_security_headers()
        self.end_headers()

        # Format error body (same as BaseHTTPRequestHandler)
        if code in self.responses:
            short, long_msg = self.responses[code]
        else:
            short, long_msg = "Unknown", "Unknown error"

        if message is None:
            message = short
        if explain is None:
            explain = long_msg

        body = self.error_message_format % {
            "code": code,
            "message": html_escape(message, quote=False),
            "explain": html_escape(explain, quote=False),
        }
        self.wfile.write(body.encode("utf-8", "replace"))

    def _log_request(
        self, method: str, path: str, status: int, duration_ms: float, extra: dict = None
    ) -> None:
        """Log request details for observability.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path
            status: HTTP status code
            duration_ms: Request duration in milliseconds
            extra: Additional context to log
        """
        if not self._request_log_enabled:
            return

        # Determine log level based on status and duration
        if status >= 500:
            log_fn = logger.error
        elif status >= 400:
            log_fn = logger.warning
        elif duration_ms > self._slow_request_threshold_ms:
            log_fn = logger.warning
        else:
            log_fn = logger.info

        # Build log message
        client_ip = self._get_client_ip()
        msg_parts = [
            f"{method} {path}",
            f"status={status}",
            f"duration={duration_ms:.1f}ms",
            f"ip={client_ip}",
        ]

        if extra:
            for k, v in extra.items():
                msg_parts.append(f"{k}={v}")

        if duration_ms > self._slow_request_threshold_ms:
            msg_parts.append("SLOW")

        log_fn(f"[request] {' '.join(msg_parts)}")

    def _normalize_endpoint(self, path: str) -> str:
        """Normalize API endpoint path for metrics by replacing dynamic IDs.

        Replaces UUIDs, numeric IDs, and other dynamic segments with placeholders
        to avoid high cardinality in Prometheus metrics.

        Args:
            path: Raw request path (e.g., "/api/debates/abc123/messages")

        Returns:
            Normalized path (e.g., "/api/debates/{id}/messages")
        """
        # UUID pattern (e.g., 550e8400-e29b-41d4-a716-446655440000)
        uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
        # Short ID pattern (alphanumeric, 8-32 chars, likely an ID)
        short_id_pattern = r"/[a-zA-Z0-9]{8,32}(?=/|$)"
        # Numeric ID pattern
        numeric_pattern = r"/\d+(?=/|$)"

        normalized = path
        # Replace UUIDs first (most specific)
        normalized = re.sub(uuid_pattern, "{id}", normalized)
        # Replace numeric IDs
        normalized = re.sub(numeric_pattern, "/{id}", normalized)
        # Replace remaining short alphanumeric IDs in path segments
        # Only if they're surrounded by slashes or at end
        normalized = re.sub(short_id_pattern, "/{id}", normalized)

        return normalized

    def _get_client_ip(self) -> str:
        """Get client IP address, respecting trusted proxy headers."""
        remote_ip = self.client_address[0] if hasattr(self, "client_address") else "unknown"
        client_ip = remote_ip
        if remote_ip in TRUSTED_PROXIES:
            forwarded = self.headers.get("X-Forwarded-For", "")
            if forwarded:
                first_ip = forwarded.split(",")[0].strip()
                if first_ip:
                    client_ip = first_ip
        return client_ip

    def _safe_int(self, query: dict, key: str, default: int, max_val: int = 100) -> int:
        """Safely parse integer query param with bounds checking.

        Delegates to shared safe_query_int from validation module.
        """
        return safe_query_int(query, key, default, min_val=1, max_val=max_val)

    def _safe_float(
        self, query: dict, key: str, default: float, min_val: float = 0.0, max_val: float = 1.0
    ) -> float:
        """Safely parse float query param with bounds checking.

        Delegates to shared safe_query_float from validation module.
        """
        return safe_query_float(query, key, default, min_val=min_val, max_val=max_val)

    def _safe_string(
        self, value: str, max_len: int = 500, pattern: Optional[str] = None
    ) -> Optional[str]:
        """Safely validate string parameter with length and pattern checks.

        Args:
            value: The string to validate
            max_len: Maximum allowed length (default 500)
            pattern: Optional regex pattern to match (e.g., r'^[a-zA-Z0-9_-]+$')

        Returns:
            Validated string or None if invalid
        """
        import re

        if not value or not isinstance(value, str):
            return None
        # Truncate to max length
        value = value[:max_len]
        # Validate pattern if provided
        if pattern and not re.match(pattern, value):
            return None
        return value

    def _extract_path_segment(
        self, path: str, index: int, segment_name: str = "id"
    ) -> Optional[str]:
        """Safely extract path segment with bounds checking.

        Returns None and sends 400 error if segment is missing.
        """
        parts = path.split("/")
        if len(parts) <= index or not parts[index]:
            self._send_json({"error": f"Missing {segment_name} in path"}, status=400)
            return None
        return parts[index]

    # Note: _init_handlers(), _log_resource_availability(), and _try_modular_handler()
    # are inherited from HandlerRegistryMixin (handler_registry.py)

    def _validate_content_length(self, max_size: int | None = None) -> Optional[int]:
        """Validate Content-Length header for DoS protection.

        Returns content length if valid, None if invalid (error already sent).
        """
        max_size = max_size or MAX_JSON_CONTENT_LENGTH

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "Invalid Content-Length header"}, status=400)
            return None

        if content_length < 0:
            self._send_json({"error": "Invalid Content-Length: negative value"}, status=400)
            return None

        if content_length > max_size:
            size_mb = max_size / (1024 * 1024)
            self._send_json({"error": f"Content too large. Max: {size_mb:.0f}MB"}, status=413)
            return None

        return content_length

    def _check_rate_limit(self) -> bool:
        """Check auth and rate limit. Returns True if allowed, False if blocked.

        Sends appropriate error response if blocked.
        """
        if not auth_config.enabled:
            return True

        # Convert headers to dict
        headers = {k: v for k, v in self.headers.items()}
        parsed = urlparse(self.path)

        authenticated, remaining = check_auth(headers, parsed.query)

        if not authenticated:
            if remaining == 0:
                # Rate limited
                self._send_json({"error": "Rate limit exceeded. Try again later."}, status=429)
            else:
                # Auth failed
                self._send_json({"error": "Authentication required"}, status=401)
            return False

        # Note: Rate limit headers are now added by individual handlers
        # that need to include them in their responses
        return True

    def _check_tier_rate_limit(self) -> bool:
        """Check tier-aware rate limit based on user's subscription.

        Returns True if allowed, False if blocked.
        Sends 429 error response if rate limited.
        Also stores the result for inclusion in response headers.
        """
        from aragora.server.middleware.rate_limit import check_tier_rate_limit

        result = check_tier_rate_limit(self, UnifiedHandler.user_store)

        # Store result for response headers (used by _add_rate_limit_headers)
        self._rate_limit_result = result

        if not result.allowed:
            self._send_json(
                {
                    "error": "Rate limit exceeded for your subscription tier",
                    "code": "tier_rate_limit",
                    "limit": result.limit,
                    "retry_after": int(result.retry_after) + 1,
                    "upgrade_url": "/pricing",
                },
                status=429,
            )
            return False

        return True

    def _check_upload_rate_limit(self) -> bool:
        """Check IP-based upload rate limit. Returns True if allowed, False if blocked.

        Uses sliding window rate limiting per IP address.
        Deques with maxlen prevent unbounded memory growth.
        """
        import time

        # Get client IP (validate proxy headers for security)
        remote_ip = self.client_address[0] if hasattr(self, "client_address") else "unknown"
        client_ip = remote_ip  # Default to direct connection IP
        if remote_ip in TRUSTED_PROXIES:
            # Only trust X-Forwarded-For from trusted proxies
            forwarded = self.headers.get("X-Forwarded-For", "")
            if forwarded:
                first_ip = forwarded.split(",")[0].strip()
                if first_ip:
                    client_ip = first_ip

        now = time.time()
        one_minute_ago = now - 60
        one_hour_ago = now - 3600

        with UnifiedHandler._upload_counts_lock:
            # Get or create upload history for this IP (bounded deque)
            if client_ip not in UnifiedHandler._upload_counts:
                UnifiedHandler._upload_counts[client_ip] = deque(
                    maxlen=UnifiedHandler._MAX_UPLOAD_TIMESTAMPS
                )

            timestamps = UnifiedHandler._upload_counts[client_ip]

            # Clean up old entries (rebuild deque with only recent timestamps)
            recent_timestamps = [ts for ts in timestamps if ts > one_hour_ago]
            timestamps.clear()
            timestamps.extend(recent_timestamps)

            # Periodically clean up stale IPs (those with no recent uploads)
            # Do this occasionally to avoid overhead on every request
            if len(UnifiedHandler._upload_counts) > 100:
                stale_ips = [
                    ip
                    for ip, ts_deque in UnifiedHandler._upload_counts.items()
                    if not ts_deque or max(ts_deque) < one_hour_ago
                ]
                for ip in stale_ips:
                    del UnifiedHandler._upload_counts[ip]

            # Check per-minute limit
            recent_minute = sum(1 for ts in timestamps if ts > one_minute_ago)
            if recent_minute >= UnifiedHandler.MAX_UPLOADS_PER_MINUTE:
                self._send_json(
                    {
                        "error": f"Upload rate limit exceeded. Max {UnifiedHandler.MAX_UPLOADS_PER_MINUTE} uploads per minute.",
                        "retry_after": 60,
                    },
                    status=429,
                )
                return False

            # Check per-hour limit
            if len(timestamps) >= UnifiedHandler.MAX_UPLOADS_PER_HOUR:
                self._send_json(
                    {
                        "error": f"Upload rate limit exceeded. Max {UnifiedHandler.MAX_UPLOADS_PER_HOUR} uploads per hour.",
                        "retry_after": 3600,
                    },
                    status=429,
                )
                return False

            # Record this upload
            timestamps.append(now)

        return True

    def _get_debate_controller(self) -> DebateController:
        """Get or create the debate controller (lazy initialization).

        Returns:
            DebateController instance
        """
        if UnifiedHandler._debate_controller is None:
            # Create factory with all subsystems
            factory = DebateFactory(
                elo_system=self.elo_system,
                persona_manager=self.persona_manager,
                debate_embeddings=self.debate_embeddings,
                position_tracker=self.position_tracker,
                position_ledger=self.position_ledger,
                flip_detector=self.flip_detector,
                dissent_retriever=self.dissent_retriever,
                moment_detector=self.moment_detector,
                stream_emitter=self.stream_emitter,
            )
            UnifiedHandler._debate_factory = factory

            # Create controller with storage for debate persistence
            UnifiedHandler._debate_controller = DebateController(
                factory=factory,
                emitter=self.stream_emitter,
                elo_system=self.elo_system,
                auto_select_fn=self._auto_select_agents,
                storage=self.storage,
            )
        return UnifiedHandler._debate_controller

    def _auto_select_agents(self, question: str, config: dict) -> str:
        """Select optimal agents using AgentSelector.

        Args:
            question: The debate question/topic
            config: Optional configuration with:
                - primary_domain: Main domain (default: 'general')
                - secondary_domains: Additional domains
                - min_agents: Minimum team size (default: 2)
                - max_agents: Maximum team size (default: 4)
                - quality_priority: 0-1 scale (default: 0.7)
                - diversity_preference: 0-1 scale (default: 0.5)

        Returns:
            Comma-separated string of agent types with optional roles
        """
        if not ROUTING_AVAILABLE:
            return "gemini,anthropic-api"  # Fallback

        try:
            # Build task requirements from question and config
            requirements = TaskRequirements(
                task_id=f"debate-{uuid.uuid4().hex[:8]}",
                description=question[:500],  # Truncate for safety
                primary_domain=config.get("primary_domain", "general"),
                secondary_domains=config.get("secondary_domains", []),
                required_traits=config.get("required_traits", []),
                min_agents=min(max(config.get("min_agents", 2), 2), 5),
                max_agents=min(max(config.get("max_agents", 4), 2), 8),
                quality_priority=min(max(config.get("quality_priority", 0.7), 0), 1),
                diversity_preference=min(max(config.get("diversity_preference", 0.5), 0), 1),
            )

            # Create selector with ELO system and persona manager
            selector = AgentSelector(
                elo_system=self.elo_system,
                persona_manager=self.persona_manager,
            )

            # Populate agent pool from allowed types
            for agent_type in ALLOWED_AGENT_TYPES:
                selector.register_agent(
                    AgentProfile(
                        name=agent_type,
                        agent_type=agent_type,
                    )
                )

            # Select optimal team
            team = selector.select_team(requirements)

            # Build agent string with roles if available
            agent_specs = []
            for agent in team.agents:
                role = team.roles.get(agent.name, "")
                if role:
                    agent_specs.append(f"{agent.agent_type}:{role}")
                else:
                    agent_specs.append(agent.agent_type)

            logger.info(
                f"[auto_select] Selected team: {agent_specs} (rationale: {team.rationale[:100]})"
            )
            return ",".join(agent_specs)

        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.warning(f"[auto_select] Failed: {e}, using fallback")
            return "gemini,anthropic-api"  # Fallback on error

    def do_GET(self) -> None:
        """Handle GET requests."""
        self._rate_limit_result = None  # Reset per-request state
        self._response_status = 200  # Track response status for metrics
        start_time = time.time()
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # Start tracing span for API requests
        span = None
        if path.startswith("/api/"):
            span = self.tracing.start_request_span("GET", path, dict(self.headers))

        try:
            self._do_GET_internal(path, query)
        except Exception as e:
            # Top-level safety net for GET handlers
            self._response_status = 500
            logger.exception(f"[request] Unhandled exception in GET {path}: {e}")
            if span:
                span.set_error(e)
            try:
                self._send_json({"error": "Internal server error"}, status=500)
            except Exception as send_err:
                logger.debug(f"Could not send error response (already sent?): {send_err}")
        finally:
            status_code = getattr(self, "_response_status", 200)
            if span:
                self.tracing.finish_request_span(span, status_code)
            duration_seconds = time.time() - start_time
            # Record Prometheus metrics for API requests
            if path.startswith("/api/"):
                # Normalize endpoint path for metrics (strip IDs)
                endpoint = self._normalize_endpoint(path)
                record_http_request("GET", endpoint, status_code, duration_seconds)
                self._log_request("GET", path, status_code, duration_seconds * 1000)

    def _do_GET_internal(self, path: str, query: dict) -> None:
        """Internal GET handler with actual routing logic."""
        # Validate query parameters against whitelist (security)
        if query and path.startswith("/api/"):
            is_valid, error_msg = _validate_query_params(query)
            if not is_valid:
                self._send_json({"error": error_msg}, status=400)
                return

        # Rate limit all API GET requests (DoS protection)
        if path.startswith("/api/"):
            if not self._check_rate_limit():
                return

        # Route all /api/* requests through modular handlers
        if path.startswith("/api/"):
            if self._try_modular_handler(path, query):
                return

        # Static file serving (non-API routes)
        if path in ("/", "/index.html"):
            self._serve_file("index.html")
        elif path.endswith((".html", ".css", ".js", ".json", ".ico", ".svg", ".png")):
            self._serve_file(path.lstrip("/"))
        else:
            # Try serving as a static file
            self._serve_file(path.lstrip("/"))

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(200)
        self._add_cors_headers()
        self.end_headers()

    def do_POST(self) -> None:
        """Handle POST requests."""
        self._rate_limit_result = None  # Reset per-request state
        self._response_status = 200  # Track response status for metrics
        start_time = time.time()
        parsed = urlparse(self.path)
        path = parsed.path

        # Start tracing span for API requests
        span = None
        if path.startswith("/api/"):
            span = self.tracing.start_request_span("POST", path, dict(self.headers))

        try:
            self._do_POST_internal(path)
        except Exception as e:
            # Top-level safety net for POST handlers
            self._response_status = 500
            logger.exception(f"[request] Unhandled exception in POST {path}: {e}")
            if span:
                span.set_error(e)
            try:
                self._send_json({"error": "Internal server error"}, status=500)
            except Exception as send_err:
                logger.debug(f"Could not send error response (already sent?): {send_err}")
        finally:
            status_code = getattr(self, "_response_status", 200)
            if span:
                self.tracing.finish_request_span(span, status_code)
            duration_seconds = time.time() - start_time
            # Record Prometheus metrics for API requests
            if path.startswith("/api/"):
                endpoint = self._normalize_endpoint(path)
                record_http_request("POST", endpoint, status_code, duration_seconds)
            self._log_request("POST", path, status_code, duration_seconds * 1000)

    def _do_POST_internal(self, path: str) -> None:
        """Internal POST handler with actual routing logic."""
        # Route all /api/* requests through modular handlers
        if path.startswith("/api/"):
            try:
                if self._try_modular_handler(path, {}):
                    return
            except (TypeError, ValueError, AttributeError, KeyError, RuntimeError, OSError) as e:
                logger.exception(f"Modular handler failed for {path}: {e}")
                # Fall through to 404

        # Debug endpoint for POST testing
        if path == "/api/debug/post-test":
            self._send_json({"status": "ok", "message": "POST handling works"})
        else:
            self.send_error(404, f"Unknown POST endpoint: {path}")

    def do_DELETE(self) -> None:
        """Handle DELETE requests."""
        self._rate_limit_result = None  # Reset per-request state
        start_time = time.time()
        status_code = 200  # Default, updated by handlers
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            self._do_DELETE_internal(path)
        except Exception as e:
            # Top-level safety net for DELETE handlers
            status_code = 500
            logger.exception(f"[request] Unhandled exception in DELETE {path}: {e}")
            try:
                self._send_json({"error": "Internal server error"}, status=500)
            except Exception as send_err:
                logger.debug(f"Could not send error response (already sent?): {send_err}")
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self._log_request("DELETE", path, status_code, duration_ms)

    def _do_DELETE_internal(self, path: str) -> None:
        """Internal DELETE handler with actual routing logic."""
        # Try modular handlers first
        if path.startswith("/api/"):
            if self._try_modular_handler(path, {}):
                return

        self.send_error(404, f"Unknown DELETE endpoint: {path}")

    def do_PATCH(self) -> None:
        """Handle PATCH requests."""
        start_time = time.time()
        status_code = 200  # Default, updated by handlers
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            self._do_PATCH_internal(path)
        except Exception as e:
            # Top-level safety net for PATCH handlers
            status_code = 500
            logger.exception(f"[request] Unhandled exception in PATCH {path}: {e}")
            try:
                self._send_json({"error": "Internal server error"}, status=500)
            except Exception as send_err:
                logger.debug(f"Could not send error response (already sent?): {send_err}")
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self._log_request("PATCH", path, status_code, duration_ms)

    def _do_PATCH_internal(self, path: str) -> None:
        """Internal PATCH handler with actual routing logic."""
        # Try modular handlers first
        if path.startswith("/api/"):
            if self._try_modular_handler(path, {}):
                return

        self.send_error(404, f"Unknown PATCH endpoint: {path}")

    def do_PUT(self) -> None:
        """Handle PUT requests."""
        start_time = time.time()
        status_code = 200  # Default, updated by handlers
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            self._do_PUT_internal(path)
        except Exception as e:
            # Top-level safety net for PUT handlers
            status_code = 500
            logger.exception(f"[request] Unhandled exception in PUT {path}: {e}")
            try:
                self._send_json({"error": "Internal server error"}, status=500)
            except Exception as send_err:
                logger.debug(f"Could not send error response (already sent?): {send_err}")
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self._log_request("PUT", path, status_code, duration_ms)

    def _do_PUT_internal(self, path: str) -> None:
        """Internal PUT handler with actual routing logic."""
        # Try modular handlers first
        if path.startswith("/api/"):
            if self._try_modular_handler(path, {}):
                return

        self.send_error(404, f"Unknown PUT endpoint: {path}")

    def _serve_file(self, filename: str) -> None:
        """Serve a static file with path traversal protection."""
        if not self.static_dir:
            self.send_error(404, "Static directory not configured")
            return

        # Security: Resolve paths and prevent directory traversal
        try:
            filepath = (self.static_dir / filename).resolve()
            static_dir_resolved = self.static_dir.resolve()

            # Ensure resolved path is within static directory
            if not str(filepath).startswith(str(static_dir_resolved)):
                self.send_error(403, "Access denied")
                return

            # Security: Reject symlinks to prevent escape attacks
            # Check the original path (before resolve) for symlink
            original_path = self.static_dir / filename
            if original_path.is_symlink():
                logger.warning(f"Symlink access denied: {filename}")
                self.send_error(403, "Symlinks not allowed")
                return
        except (ValueError, OSError):
            self.send_error(400, "Invalid path")
            return

        if not filepath.exists():
            # Try index.html for SPA routing
            filepath = self.static_dir / "index.html"
            if not filepath.exists():
                self.send_error(404, "File not found")
                return

        # Determine content type
        content_type = "text/html"
        if filename.endswith(".css"):
            content_type = "text/css"
        elif filename.endswith(".js"):
            content_type = "application/javascript"
        elif filename.endswith(".json"):
            content_type = "application/json"
        elif filename.endswith(".ico"):
            content_type = "image/x-icon"
        elif filename.endswith(".svg"):
            content_type = "image/svg+xml"
        elif filename.endswith(".png"):
            content_type = "image/png"

        try:
            content = filepath.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self._add_cors_headers()
            self._add_security_headers()
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404, "File not found")
        except PermissionError:
            self.send_error(403, "Permission denied")
        except (IOError, OSError) as e:
            logger.error(f"File read error: {e}")
            self.send_error(500, "Failed to read file")
        except (BrokenPipeError, ConnectionResetError) as e:
            # Client disconnected before response could be sent
            logger.debug(f"Client disconnected during file serve: {type(e).__name__}")

    def _send_json(self, data, status: int = 200) -> None:
        """Send JSON response."""
        self._response_status = status  # Track for metrics
        content = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self._add_cors_headers()
        self._add_security_headers()
        self._add_rate_limit_headers()
        self._add_trace_headers()
        self.end_headers()
        self.wfile.write(content)

    def _add_trace_headers(self) -> None:
        """Add trace ID header to response for correlation."""
        trace_id = get_trace_id()
        if trace_id:
            self.send_header(TRACE_ID_HEADER, trace_id)

    def _add_rate_limit_headers(self) -> None:
        """Add rate limit headers to response.

        Includes X-RateLimit-Limit, X-RateLimit-Remaining, and X-RateLimit-Reset
        headers if a rate limit check was performed for this request.
        """
        result = getattr(self, "_rate_limit_result", None)
        if result is None:
            return

        from aragora.server.middleware.rate_limit import rate_limit_headers

        headers = rate_limit_headers(result)
        for name, value in headers.items():
            self.send_header(name, value)

    def _add_security_headers(self) -> None:
        """Add security headers to prevent common attacks."""
        # Prevent clickjacking
        self.send_header("X-Frame-Options", "DENY")
        # Prevent MIME type sniffing
        self.send_header("X-Content-Type-Options", "nosniff")
        # Enable XSS filter
        self.send_header("X-XSS-Protection", "1; mode=block")
        # Referrer policy - don't leak internal URLs
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        # Content Security Policy - prevent XSS and data injection
        # Note: 'unsafe-inline' for styles needed by CSS-in-JS frameworks
        # 'unsafe-eval' removed for security - blocks eval()/new Function()
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "connect-src 'self' wss: https:; "
            "font-src 'self' data:; "
            "frame-ancestors 'none'",
        )
        # HTTP Strict Transport Security - enforce HTTPS
        self.send_header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")

    def _add_cors_headers(self) -> None:
        """Add CORS headers with origin validation."""
        # Security: Validate origin against centralized allowlist
        request_origin = self.headers.get("Origin", "")

        if cors_config.is_origin_allowed(request_origin):
            self.send_header("Access-Control-Allow-Origin", request_origin)
        elif not request_origin:
            # Same-origin requests don't have Origin header
            pass
        # else: no CORS header = browser blocks cross-origin request

        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename, Authorization")

    def log_message(self, format: str, *args) -> None:
        """Suppress default logging."""
        pass


class UnifiedServer:
    """
    Combined HTTP + WebSocket server for the nomic loop dashboard.

    Usage:
        server = UnifiedServer(
            http_port=8080,
            ws_port=8765,
            static_dir=Path("aragora/live/out"),
            nomic_dir=Path("/path/to/aragora/.nomic"),
        )
        await server.start()  # Starts both servers
    """

    def __init__(
        self,
        http_port: int = 8080,
        ws_port: int = 8765,
        ws_host: str = "0.0.0.0",
        http_host: str = "",
        static_dir: Optional[Path] = None,
        nomic_dir: Optional[Path] = None,
        storage: Optional[DebateStorage] = None,
        enable_persistence: bool = True,
        ssl_cert: Optional[str] = None,
        ssl_key: Optional[str] = None,
    ):
        """Initialize the unified HTTP/WebSocket server with all subsystems.

        Args:
            http_port: Port for HTTP API server (default 8080)
            ws_port: Port for WebSocket streaming (default 8765)
            ws_host: WebSocket bind address (default "0.0.0.0")
            http_host: HTTP bind address (default "" for all interfaces)
            static_dir: Optional path to static files for serving UI
            nomic_dir: Optional path to nomic state directory (enables many features)
            storage: Optional DebateStorage for debate persistence
            enable_persistence: Enable Supabase persistence if configured
            ssl_cert: Path to SSL certificate for HTTPS
            ssl_key: Path to SSL key for HTTPS

        Subsystems initialized when nomic_dir is provided:
            - InsightStore: Extract learnings from debates
            - EloSystem: Agent skill ratings
            - FlipDetector: Position reversal detection
            - DocumentStore: File upload handling
            - AudioFileStore: Broadcast audio storage
            - TwitterPosterConnector: Social media posting
            - YouTubeUploaderConnector: Video uploads
            - VideoGenerator: Video creation
            - PersonaManager: Agent specialization
            - PositionLedger: Truth-grounded positions
            - DebateEmbeddingsDatabase: Historical debate vectors
            - ConsensusMemory/DissentRetriever: Minority view tracking
            - MomentDetector: Narrative moment detection
        """
        self.http_port = http_port
        self.ws_port = ws_port
        self.ws_host = ws_host
        self.http_host = http_host
        self.static_dir = static_dir
        self.nomic_dir = nomic_dir
        self.storage = storage
        self.ssl_cert = ssl_cert
        self.ssl_key = ssl_key
        self.ssl_enabled = bool(ssl_cert and ssl_key)

        # Create WebSocket server
        self.stream_server = DebateStreamServer(host=ws_host, port=ws_port)

        # Initialize Supabase persistence if available
        self.persistence = init_persistence(enable_persistence)

        # Setup HTTP handler with base resources
        UnifiedHandler.storage = storage
        UnifiedHandler.static_dir = static_dir
        UnifiedHandler.stream_emitter = self.stream_server.emitter
        UnifiedHandler.persistence = self.persistence

        # Initialize nomic-dependent subsystems
        if nomic_dir:
            self._init_subsystems(nomic_dir)

    def _init_subsystems(self, nomic_dir: Path) -> None:
        """Initialize all nomic directory dependent subsystems.

        Configures the UnifiedHandler class with all required subsystems
        for full API functionality.
        """
        from aragora.server.initialization import init_handler_stores

        UnifiedHandler.nomic_state_file = nomic_dir / "nomic_state.json"

        # Database-backed subsystems (from initialization.py)
        UnifiedHandler.insight_store = init_insight_store(nomic_dir)
        UnifiedHandler.elo_system = init_elo_system(nomic_dir)
        UnifiedHandler.flip_detector = init_flip_detector(nomic_dir)
        UnifiedHandler.persona_manager = init_persona_manager(nomic_dir)
        UnifiedHandler.position_ledger = init_position_ledger(nomic_dir)
        UnifiedHandler.debate_embeddings = init_debate_embeddings(nomic_dir)
        UnifiedHandler.consensus_memory, UnifiedHandler.dissent_retriever = init_consensus_memory()
        UnifiedHandler.moment_detector = init_moment_detector(
            elo_system=UnifiedHandler.elo_system,
            position_ledger=UnifiedHandler.position_ledger,
        )

        # Non-database stores and connectors (from initialization.py)
        stores = init_handler_stores(nomic_dir)
        UnifiedHandler.document_store = stores["document_store"]
        UnifiedHandler.audio_store = stores["audio_store"]
        UnifiedHandler.video_generator = stores["video_generator"]
        UnifiedHandler.twitter_connector = stores["twitter_connector"]
        UnifiedHandler.youtube_connector = stores["youtube_connector"]
        UnifiedHandler.user_store = stores["user_store"]
        UnifiedHandler.usage_tracker = stores["usage_tracker"]

    @property
    def emitter(self) -> SyncEventEmitter:
        """Get the event emitter for nomic loop integration."""
        return self.stream_server.emitter

    def _run_http_server(self) -> None:
        """Run HTTP server in a thread, optionally with SSL/TLS."""
        import ssl

        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                server = HTTPServer((self.http_host, self.http_port), UnifiedHandler)

                # Configure SSL if cert and key are provided
                if self.ssl_enabled:
                    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                    ssl_context.load_cert_chain(
                        certfile=self.ssl_cert,
                        keyfile=self.ssl_key,
                    )
                    # Use secure defaults
                    ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
                    ssl_context.set_ciphers("ECDHE+AESGCM:DHE+AESGCM:ECDHE+CHACHA20:DHE+CHACHA20")
                    server.socket = ssl_context.wrap_socket(
                        server.socket,
                        server_side=True,
                    )
                    protocol = "HTTPS"
                else:
                    protocol = "HTTP"

                logger.info(f"{protocol} server listening on {self.http_host}:{self.http_port}")
                server.serve_forever()
                break  # Normal exit
            except ssl.SSLError as e:
                logger.error(f"SSL configuration error: {e}")
                break
            except OSError as e:
                if e.errno == 98 or "Address already in use" in str(e):  # EADDRINUSE
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Port {self.http_port} in use, retrying in {retry_delay}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(
                            f"Failed to bind HTTP server to port {self.http_port} "
                            f"after {max_retries} attempts: {e}"
                        )
                else:
                    logger.error(f"HTTP server failed to start: {e}")
                    break
            except (RuntimeError, SystemError, KeyboardInterrupt) as e:
                logger.error(f"HTTP server unexpected error: {e}")
                break

    async def start(self) -> None:
        """Start both HTTP and WebSocket servers."""
        # Initialize error monitoring (no-op if SENTRY_DSN not set)
        try:
            from aragora.server.error_monitoring import init_monitoring

            if init_monitoring():
                logger.info("Error monitoring enabled (Sentry)")
        except ImportError:
            pass  # sentry-sdk not installed

        # Initialize OpenTelemetry tracing (if OTEL_ENABLED=true)
        try:
            from aragora.observability.config import is_tracing_enabled
            from aragora.observability.tracing import get_tracer

            if is_tracing_enabled():
                get_tracer()  # Initialize tracer singleton
                logger.info("OpenTelemetry tracing enabled")
            else:
                logger.debug("OpenTelemetry tracing disabled (set OTEL_ENABLED=true to enable)")
        except ImportError as e:
            logger.debug(f"OpenTelemetry not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenTelemetry: {e}")

        # Initialize Prometheus metrics (if METRICS_ENABLED=true)
        try:
            from aragora.observability.config import is_metrics_enabled
            from aragora.observability.metrics import start_metrics_server

            if is_metrics_enabled():
                # Note: start_metrics_server starts a separate HTTP server on METRICS_PORT (default 9090)
                # The /metrics endpoint at /api path uses aragora.server.metrics instead
                start_metrics_server()
                logger.info("Prometheus metrics server started")
        except ImportError as e:
            logger.debug(f"Prometheus metrics not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")

        # Initialize circuit breaker persistence
        try:
            from aragora.resilience import (
                init_circuit_breaker_persistence,
                load_circuit_breakers,
            )

            data_dir = self.nomic_dir or Path(".data")
            db_path = str(data_dir / "circuit_breaker.db")
            init_circuit_breaker_persistence(db_path)
            loaded = load_circuit_breakers()
            if loaded > 0:
                logger.info(f"Restored {loaded} circuit breaker states from disk")
        except (ImportError, OSError, RuntimeError) as e:
            logger.debug(f"Circuit breaker persistence not available: {e}")

        # Initialize background tasks for maintenance
        try:
            from aragora.server.background import get_background_manager, setup_default_tasks

            nomic_path = str(self.nomic_dir) if self.nomic_dir else None
            # Pass the shared continuum_memory instance for efficient cleanup
            # Note: continuum_memory is initialized lazily via get_continuum_memory()
            setup_default_tasks(
                nomic_dir=nomic_path,
                memory_instance=None,  # Will use shared instance from get_continuum_memory()
            )
            background_mgr = get_background_manager()
            background_mgr.start()
            logger.info("Background task manager started")
        except (ImportError, RuntimeError, OSError) as e:
            logger.warning("Failed to start background tasks: %s", e)

        # Auto-start pulse scheduler if configured
        try:
            from aragora.config.legacy import (
                PULSE_SCHEDULER_AUTOSTART,
                PULSE_SCHEDULER_MAX_PER_HOUR,
                PULSE_SCHEDULER_POLL_INTERVAL,
            )

            if PULSE_SCHEDULER_AUTOSTART:
                from aragora.server.handlers.pulse import get_pulse_scheduler

                scheduler = get_pulse_scheduler()
                if scheduler:
                    # Update config from environment
                    scheduler.update_config(
                        {
                            "poll_interval_seconds": PULSE_SCHEDULER_POLL_INTERVAL,
                            "max_debates_per_hour": PULSE_SCHEDULER_MAX_PER_HOUR,
                        }
                    )

                    # Set up debate creator callback
                    async def auto_create_debate(topic_text: str, rounds: int, threshold: float):
                        try:
                            from aragora import Arena, DebateProtocol, Environment
                            from aragora.agents import get_agents_by_names

                            env = Environment(task=topic_text)
                            agents = get_agents_by_names(["anthropic-api", "openai-api"])
                            protocol = DebateProtocol(rounds=rounds, consensus="majority")
                            if not agents:
                                return None
                            arena = Arena.from_env(env, agents, protocol)
                            result = await arena.run()
                            return {
                                "debate_id": result.id,
                                "consensus_reached": result.consensus_reached,
                                "confidence": result.confidence,
                                "rounds_used": result.rounds_used,
                            }
                        except Exception as e:
                            logger.error(f"Auto-scheduled debate failed: {e}")
                            return None

                    scheduler.set_debate_creator(auto_create_debate)
                    asyncio.create_task(scheduler.start())
                    logger.info("Pulse scheduler auto-started (PULSE_SCHEDULER_AUTOSTART=true)")
                else:
                    logger.warning("Pulse scheduler not available for autostart")
        except ImportError as e:
            logger.debug(f"Pulse scheduler autostart not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to auto-start pulse scheduler: {e}")

        # Start periodic state cleanup task (prevents memory leaks from stale entries)
        try:
            from aragora.server.stream.state_manager import (
                get_state_manager,
                start_cleanup_task,
            )

            state_manager = get_state_manager()
            start_cleanup_task(state_manager, interval_seconds=300)
            logger.debug("State cleanup task started (5 min interval)")
        except (ImportError, RuntimeError) as e:
            logger.debug(f"State cleanup task not started: {e}")

        logger.info("Starting unified server...")
        protocol = "https" if self.ssl_enabled else "http"
        logger.info(f"  HTTP API:   {protocol}://localhost:{self.http_port}")
        logger.info(f"  WebSocket:  ws://localhost:{self.ws_port}")
        if self.ssl_enabled:
            logger.info(f"  SSL:        enabled (cert: {self.ssl_cert})")
        if self.static_dir:
            logger.info(f"  Static dir: {self.static_dir}")
        if self.nomic_dir:
            logger.info(f"  Nomic dir:  {self.nomic_dir}")

        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()

        # Start HTTP server in background thread
        self._http_thread = Thread(target=self._run_http_server, daemon=True)
        self._http_thread.start()

        # Start WebSocket server in foreground
        await self.stream_server.start()

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        import signal

        def signal_handler(signum, frame):
            signame = signal.Signals(signum).name
            logger.info(f"Received {signame}, initiating graceful shutdown...")
            asyncio.create_task(self.graceful_shutdown())

        # Register handlers for common termination signals
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            logger.debug("Signal handlers registered for graceful shutdown")
        except (ValueError, OSError) as e:
            # Signal handling may not work in all contexts (e.g., non-main thread)
            logger.debug(f"Could not register signal handlers: {e}")

    async def graceful_shutdown(self, timeout: float = 30.0) -> None:
        """Gracefully shut down the server.

        Steps:
        1. Stop accepting new debates
        2. Wait for in-flight debates to complete (with timeout)
        3. Persist circuit breaker states
        4. Stop background tasks
        5. Close WebSocket connections
        6. Close shared HTTP connector
        7. Close database connections (connection pool cleanup)

        Args:
            timeout: Maximum seconds to wait for in-flight debates
        """
        logger.info("Starting graceful shutdown...")
        shutdown_start = time.time()

        # 1. Stop accepting new debates by setting flag
        # This is checked by debate creation endpoints
        self._shutting_down = True

        # 2. Wait for in-flight debates to complete
        logger.info("Waiting for in-flight debates to complete...")
        active_debates = get_active_debates()
        if active_debates:
            in_progress = [
                d_id for d_id, d in active_debates.items() if d.get("status") == "in_progress"
            ]
            if in_progress:
                logger.info(f"Waiting for {len(in_progress)} in-flight debate(s)")
                wait_start = time.time()
                while time.time() - wait_start < timeout:
                    # Check if debates are still running
                    still_running = sum(
                        1
                        for d_id in in_progress
                        if d_id in active_debates
                        and active_debates.get(d_id, {}).get("status") == "in_progress"
                    )
                    if still_running == 0:
                        logger.info("All in-flight debates completed")
                        break
                    await asyncio.sleep(1)
                else:
                    logger.warning(
                        f"Shutdown timeout reached with {still_running} debate(s) still running"
                    )

        # 3. Persist circuit breaker states
        try:
            from aragora.resilience import persist_all_circuit_breakers

            count = persist_all_circuit_breakers()
            if count > 0:
                logger.info(f"Persisted {count} circuit breaker state(s)")
        except (ImportError, OSError, RuntimeError) as e:
            logger.warning(f"Failed to persist circuit breaker states: {e}")

        # 3.5 Shutdown OpenTelemetry tracer (flushes pending spans)
        try:
            from aragora.observability.tracing import shutdown as shutdown_tracing

            shutdown_tracing()
            logger.info("OpenTelemetry tracer shutdown complete")
        except (ImportError, RuntimeError) as e:
            logger.debug(f"Tracer shutdown: {e}")

        # 4. Stop background tasks
        try:
            from aragora.server.background import get_background_manager

            background_mgr = get_background_manager()
            background_mgr.stop()
            logger.info("Background tasks stopped")
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.debug(f"Background task shutdown: {e}")

        # 4.5. Stop pulse scheduler if running
        try:
            from aragora.server.handlers.pulse import get_pulse_scheduler

            scheduler = get_pulse_scheduler()
            if scheduler and scheduler.state.value != "stopped":
                await scheduler.stop(graceful=True)
                logger.info("Pulse scheduler stopped")
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.debug(f"Pulse scheduler shutdown: {e}")

        # 4.6. Stop state cleanup task
        try:
            from aragora.server.stream.state_manager import stop_cleanup_task

            stop_cleanup_task()
            logger.debug("State cleanup task stopped")
        except (ImportError, RuntimeError) as e:
            logger.debug(f"State cleanup shutdown: {e}")

        # 5. Close WebSocket connections
        if hasattr(self, "stream_server") and self.stream_server:
            try:
                await self.stream_server.graceful_shutdown()
                logger.info("WebSocket connections closed")
            except (OSError, RuntimeError, asyncio.CancelledError) as e:
                logger.warning(f"WebSocket shutdown error: {e}")

        # 6. Close shared HTTP connector (prevents connection leaks)
        try:
            from aragora.agents.api_agents.common import close_shared_connector

            await close_shared_connector()
            logger.info("Shared HTTP connector closed")
        except (ImportError, OSError, RuntimeError) as e:
            logger.debug(f"Connector shutdown: {e}")

        # 6.5. Close Redis connection pool
        try:
            from aragora.server.redis_config import close_redis_pool

            close_redis_pool()
            logger.debug("Redis connection pool closed")
        except (ImportError, RuntimeError) as e:
            logger.debug(f"Redis shutdown: {e}")

        # 6.6. Stop auth cleanup thread
        try:
            auth_config.stop_cleanup_thread()
            logger.debug("Auth cleanup thread stopped")
        except (RuntimeError, AttributeError) as e:
            logger.debug(f"Auth cleanup shutdown: {e}")

        # 7. Close database connections (connection pool cleanup)
        try:
            from aragora.storage.schema import DatabaseManager

            DatabaseManager.clear_instances()
            logger.info("Database connections closed")
        except (ImportError, sqlite3.Error) as e:
            logger.debug(f"Database shutdown: {e}")

        elapsed = time.time() - shutdown_start
        logger.info(f"Graceful shutdown completed in {elapsed:.1f}s")

    @property
    def is_shutting_down(self) -> bool:
        """Check if server is in shutdown mode."""
        return getattr(self, "_shutting_down", False)


async def run_unified_server(
    http_port: int = 8080,
    ws_port: int = 8765,
    static_dir: Optional[Path] = None,
    nomic_dir: Optional[Path] = None,
    ssl_cert: Optional[str] = None,
    ssl_key: Optional[str] = None,
) -> None:
    """
    Convenience function to run the unified server.

    Args:
        http_port: Port for HTTP API (default 8080)
        ws_port: Port for WebSocket streaming (default 8765)
        static_dir: Directory containing static files (dashboard build)
        nomic_dir: Path to .nomic directory for state access
        ssl_cert: Path to SSL certificate file (optional)
        ssl_key: Path to SSL private key file (optional)

    Environment variables:
        ARAGORA_SSL_ENABLED: Set to 'true' to enable SSL
        ARAGORA_SSL_CERT: Path to SSL certificate
        ARAGORA_SSL_KEY: Path to SSL private key

    Example:
        # Without SSL
        await run_unified_server()

        # With SSL
        await run_unified_server(
            ssl_cert="/path/to/cert.pem",
            ssl_key="/path/to/key.pem",
        )
    """
    # Check environment variables for SSL config
    from aragora.config import SSL_CERT_PATH, SSL_ENABLED, SSL_KEY_PATH

    if ssl_cert is None and SSL_ENABLED:
        ssl_cert = SSL_CERT_PATH
        ssl_key = SSL_KEY_PATH

    # Initialize storage from nomic directory
    storage = None
    if nomic_dir:
        db_path = nomic_dir / "debates.db"
        try:
            storage = DebateStorage(str(db_path))
            logger.info(f"[server] DebateStorage initialized at {db_path}")
        except (OSError, RuntimeError) as e:
            logger.warning(f"[server] Failed to initialize DebateStorage: {e}")

    # Ensure demo data is loaded for search functionality
    try:
        from aragora.fixtures import ensure_demo_data

        logger.info("[server] Checking demo data initialization...")
        ensure_demo_data()
    except (ImportError, OSError, RuntimeError) as e:
        logger.warning(f"[server] Demo data initialization failed: {e}")

    server = UnifiedServer(
        http_port=http_port,
        ws_port=ws_port,
        static_dir=static_dir,
        nomic_dir=nomic_dir,
        storage=storage,
        ssl_cert=ssl_cert,
        ssl_key=ssl_key,
    )
    await server.start()
