"""
Unified server combining HTTP API and WebSocket streaming.

Provides a single entry point for:
- HTTP API at /api/* endpoints
- WebSocket streaming at ws://host:port/ws
- Static file serving for the live dashboard
"""

import asyncio
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.agents.grounded import MomentDetector, PositionLedger
    from aragora.agents.personas import PersonaManager
    from aragora.agents.truth_grounding import PositionTracker
    from aragora.billing.usage import UsageTracker
    from aragora.broadcast.storage import AudioFileStore
    from aragora.broadcast.video_gen import VideoGenerator
    from aragora.connectors.twitter_poster import TwitterPosterConnector
    from aragora.connectors.youtube_uploader import YouTubeUploaderConnector
    from aragora.core.decision import DecisionRouter
    from aragora.debate.embeddings import DebateEmbeddingsDatabase
    from aragora.insights.flip_detector import FlipDetector
    from aragora.insights.store import InsightStore
    from aragora.memory.consensus import ConsensusMemory, DissentRetriever
    from aragora.persistence.supabase import SupabaseClient
    from aragora.ranking.elo import EloSystem
    from aragora.server.documents import DocumentStore
    from aragora.server.middleware.rate_limit import RateLimitResult
    from aragora.storage import UserStore
import logging
import time
from urllib.parse import urlparse

from .auth import auth_config, check_auth
from .middleware.tracing import TracingMiddleware
from aragora.rbac.middleware import RBACMiddleware, RBACMiddlewareConfig, DEFAULT_ROUTE_PERMISSIONS
from .storage import DebateStorage
from .stream import (
    ControlPlaneStreamServer,
    DebateStreamServer,
    NomicLoopStreamServer,
    SyncEventEmitter,
)

# Configure module logger
logger = logging.getLogger(__name__)

# Import centralized config and error utilities
from aragora.server.debate_controller import DebateController
from aragora.server.debate_factory import DebateFactory

# Import utilities from extracted modules
from aragora.server.http_utils import validate_query_params as _validate_query_params
from aragora.server.validation import safe_query_float, safe_query_int

# Import extracted modules
from aragora.server.agent_selection import auto_select_agents
from aragora.server.request_lifecycle import create_lifecycle_manager
from aragora.server.response_utils import ResponseHelpersMixin
from aragora.server.shutdown_sequence import create_server_shutdown_sequence
from aragora.server.static_file_handler import StaticFileHandler
from aragora.server.upload_rate_limit import get_upload_limiter

# DoS protection limits
MAX_MULTIPART_PARTS = 10
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB for uploads
MAX_JSON_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB for JSON API

# Trusted proxies for X-Forwarded-For header validation
TRUSTED_PROXIES = frozenset(
    p.strip() for p in os.getenv("ARAGORA_TRUSTED_PROXIES", "127.0.0.1,::1,localhost").split(",")
)

# Import from initialization module
from aragora.server.handler_registry import HandlerRegistryMixin
from aragora.server.initialization import init_persistence

# Server startup time for uptime tracking
_server_start_time: float = time.time()


class UnifiedHandler(ResponseHelpersMixin, HandlerRegistryMixin, BaseHTTPRequestHandler):  # type: ignore[misc]
    """HTTP handler with API endpoints and static file serving.

    Handler routing is provided by HandlerRegistryMixin from handler_registry.py.
    Response helpers are provided by ResponseHelpersMixin from response_utils.py.
    """

    storage: Optional[DebateStorage] = None
    static_dir: Optional[Path] = None
    stream_emitter: Optional[SyncEventEmitter] = None
    control_plane_stream: Optional["ControlPlaneStreamServer"] = None
    nomic_loop_stream: Optional["NomicLoopStreamServer"] = None
    tracing: TracingMiddleware = TracingMiddleware(service_name="aragora-api")
    rbac: RBACMiddleware = RBACMiddleware(
        RBACMiddlewareConfig(
            route_permissions=DEFAULT_ROUTE_PERMISSIONS,
            bypass_paths={
                # Health checks (required by load balancers/orchestrators)
                "/health",
                "/healthz",
                "/ready",
                "/readyz",
                # Observability
                "/metrics",
                # API documentation
                "/api/docs",
                "/api/docs/",
                "/api/redoc",
                "/api/redoc/",
                "/openapi.json",
                "/api/openapi",
                "/api/openapi.json",
                "/api/openapi.yaml",
                "/api/postman.json",
                "/api/v1/docs",
                "/api/v1/docs/",
                "/api/v1/openapi",
                "/api/v1/openapi.json",
                # Auth endpoints (must be accessible before authentication)
                "/api/v1/auth/register",
                "/api/v1/auth/login",
                "/api/v1/auth/refresh",
                "/api/v1/auth/signup",
                "/api/v1/auth/verify-email",
                "/api/v1/auth/resend-verification",
                "/api/v1/auth/accept-invite",
                "/api/v1/auth/check-invite",
                # OAuth endpoints (callbacks must be public)
                "/api/v1/auth/oauth/",
                "/api/auth/oauth/",
                # SSO endpoints
                "/api/v1/auth/sso/",
                "/auth/sso/",
                # Explicit public endpoints
                "/api/public/",
            },
            bypass_methods={"OPTIONS"},
            default_authenticated=True,  # SECURITY: Require auth by default for unmatched routes
        )
    )
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
    decision_router: Optional["DecisionRouter"] = None

    # Debate controller and factory (initialized lazily)
    _debate_controller: Optional[DebateController] = None
    _debate_factory: Optional[DebateFactory] = None

    # Request logging for observability
    _request_log_enabled = True
    _slow_request_threshold_ms = 1000

    # Per-request rate limit result (set by _check_tier_rate_limit)
    _rate_limit_result: Optional["RateLimitResult"] = None

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        """Override send_error to return JSON instead of HTML.

        This ensures API clients always receive JSON error responses that can be
        properly parsed, rather than HTML error pages that cause JSON parse errors.
        """
        # Get default message if not provided
        if code in self.responses:
            short, _ = self.responses[code]
        else:
            short = "Unknown"

        if message is None:
            message = short

        # Build JSON error response
        error_body: dict[str, Any] = {
            "error": message,
            "code": code,
        }
        if explain:
            error_body["explain"] = explain

        # Use _send_json which handles CORS headers
        self._send_json(error_body, status=code)

    def _log_request(
        self, method: str, path: str, status: int, duration_ms: float, extra: Optional[dict] = None
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

    # Paths exempt from authentication (health checks, probes, OAuth flow, public read-only)
    AUTH_EXEMPT_PATHS = frozenset(
        [
            # Health checks (needed for load balancers, monitoring)
            "/healthz",
            "/readyz",
            "/api/health",
            "/api/health/detailed",
            "/api/health/deep",
            "/api/health/stores",
            "/api/v1/health",
            "/api/v1/health/detailed",
            "/api/v1/health/deep",
            "/api/v1/health/stores",
            # OAuth
            "/api/auth/oauth/providers",  # Login page needs to show available providers
            "/api/v1/auth/oauth/providers",  # v1 route
            # API documentation (public)
            "/api/openapi",
            "/api/openapi.json",
            "/api/openapi.yaml",
            "/api/postman.json",
            "/api/docs",
            "/api/docs/",
            "/api/redoc",
            "/api/redoc/",
            "/api/v1/openapi",
            "/api/v1/openapi.json",
            "/api/v1/docs",
            "/api/v1/docs/",
            # Read-only public endpoints
            "/api/insights/recent",
            "/api/flips/recent",
            "/api/evidence",
            "/api/evidence/statistics",
            "/api/verification/status",
            "/api/v1/insights/recent",
            "/api/v1/flips/recent",
            "/api/v1/evidence",
            "/api/v1/evidence/statistics",
            "/api/v1/verification/status",
            # Agent/ranking public data
            "/api/leaderboard",
            "/api/leaderboard-view",
            "/api/agents",
            "/api/v1/leaderboard",
            "/api/v1/leaderboard-view",
            "/api/v1/agents",
        ]
    )

    # Path prefixes exempt from authentication (OAuth callbacks, read-only data)
    AUTH_EXEMPT_PREFIXES = (
        "/api/auth/oauth/",  # OAuth flow (login, callback)
        "/api/v1/auth/oauth/",  # OAuth flow v1 routes
        "/api/agent/",  # Agent profiles (read-only)
        "/api/v1/agent/",  # Agent profiles v1 routes
        "/api/routing/",  # Domain detection and routing (read-only)
        "/api/v1/routing/",  # Domain routing v1 routes
    )

    # Path prefixes exempt ONLY for GET requests (read-only access)
    AUTH_EXEMPT_GET_PREFIXES = ("/api/evidence/",)  # Evidence read-only access

    def _check_rate_limit(self) -> bool:
        """Check auth and rate limit. Returns True if allowed, False if blocked.

        Sends appropriate error response if blocked.
        """
        if not auth_config.enabled:
            return True

        # Skip auth for health endpoints (needed for load balancers, monitoring)
        # and OAuth flow (login/callback need to work before user is authenticated)
        parsed = urlparse(self.path)
        if parsed.path in self.AUTH_EXEMPT_PATHS:
            return True
        if any(parsed.path.startswith(prefix) for prefix in self.AUTH_EXEMPT_PREFIXES):
            return True
        # For GET-only exempt paths, check method
        if self.command == "GET" and any(
            parsed.path.startswith(prefix) for prefix in self.AUTH_EXEMPT_GET_PREFIXES
        ):
            return True

        # Convert headers to dict
        headers = {k: v for k, v in self.headers.items()}

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

    def _check_rbac(self, path: str, method: str) -> bool:
        """Check RBAC permission for the request.

        Returns True if allowed, False if blocked.
        Sends 401/403 error response if denied.
        """
        from aragora.billing.auth import extract_user_from_request
        from aragora.rbac import AuthorizationContext, get_role_permissions

        logger.info(f"[RBAC_DEBUG] Checking auth for {method} {path}")

        # Get Authorization header for debugging
        auth_header = self.headers.get("Authorization", "")
        if auth_header:
            header_type = auth_header.split(" ")[0] if " " in auth_header else auth_header
            token_prefix = auth_header[7:27] + "..." if len(auth_header) > 27 else auth_header[7:]
            logger.info(
                f"[RBAC_DEBUG] Authorization header present: type={header_type}, token_prefix={token_prefix}"
            )
        else:
            logger.info("[RBAC_DEBUG] No Authorization header")

        # Build authorization context from JWT
        auth_ctx = None
        try:
            user_ctx = extract_user_from_request(self, UnifiedHandler.user_store)
            logger.info(
                f"[RBAC_DEBUG] extract_user_from_request result: authenticated={user_ctx.authenticated}, user_id={user_ctx.user_id}, error_reason={user_ctx.error_reason}"
            )
            if user_ctx.authenticated and user_ctx.user_id:
                roles = {user_ctx.role} if user_ctx.role else {"member"}
                permissions: set[str] = set()
                for role in roles:
                    permissions |= get_role_permissions(role, include_inherited=True)

                auth_ctx = AuthorizationContext(
                    user_id=user_ctx.user_id,
                    org_id=user_ctx.org_id,
                    roles=roles,
                    permissions=permissions,
                    ip_address=user_ctx.client_ip,
                )
                logger.info(f"[RBAC_DEBUG] Auth context created for user {user_ctx.user_id}")
            else:
                logger.info("[RBAC_DEBUG] User not authenticated, auth_ctx will be None")
        except Exception as e:
            logger.warning(f"[RBAC_DEBUG] RBAC context extraction failed: {e}")

        # Check permission
        allowed, reason, permission_key = self.rbac.check_request(path, method, auth_ctx)

        if not allowed:
            if auth_ctx is None:
                self._send_json(
                    {"error": "Authentication required", "code": "auth_required"},
                    status=401,
                )
            else:
                self._send_json(
                    {
                        "error": f"Permission denied: {reason}",
                        "code": "permission_denied",
                        "required_permission": permission_key,
                    },
                    status=403,
                )
            return False

        return True

    def _check_upload_rate_limit(self) -> bool:
        """Check IP-based upload rate limit. Returns True if allowed, False if blocked."""
        limiter = get_upload_limiter()
        client_ip = limiter.get_client_ip(self)
        allowed, error_info = limiter.check_allowed(client_ip)

        if not allowed and error_info:
            self._send_json(
                {"error": error_info["message"], "retry_after": error_info["retry_after"]},
                status=429,
            )
            return False

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
        """Select optimal agents using question classification and AgentSelector."""
        return auto_select_agents(
            question=question,
            config=config,
            elo_system=self.elo_system,
            persona_manager=self.persona_manager,
        )

    def do_GET(self) -> None:
        """Handle GET requests."""
        lifecycle = create_lifecycle_manager(self)
        lifecycle.handle_request("GET", self._do_GET_internal, with_query=True)

    def _do_GET_internal(self, path: str, query: dict) -> None:
        """Internal GET handler with actual routing logic."""
        # Validate query parameters against whitelist (security)
        if query and path.startswith("/api/"):
            is_valid, error_msg = _validate_query_params(query)
            if not is_valid:
                self._send_json({"error": error_msg}, status=400)
                return

        # RBAC check for all API requests (authorization)
        if path.startswith("/api/"):
            if not self._check_rbac(path, "GET"):
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
        lifecycle = create_lifecycle_manager(self)
        lifecycle.handle_request("POST", self._do_POST_internal)

    def _do_POST_internal(self, path: str) -> None:
        """Internal POST handler with actual routing logic."""
        # RBAC check for all API requests (authorization)
        if path.startswith("/api/"):
            if not self._check_rbac(path, "POST"):
                return

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
        lifecycle = create_lifecycle_manager(self)
        lifecycle.handle_request("DELETE", self._do_DELETE_internal)

    def _do_DELETE_internal(self, path: str) -> None:
        """Internal DELETE handler with actual routing logic."""
        # RBAC check for all API requests (authorization)
        if path.startswith("/api/"):
            if not self._check_rbac(path, "DELETE"):
                return

        # Try modular handlers first
        if path.startswith("/api/"):
            if self._try_modular_handler(path, {}):
                return

        self.send_error(404, f"Unknown DELETE endpoint: {path}")

    def do_PATCH(self) -> None:
        """Handle PATCH requests."""
        lifecycle = create_lifecycle_manager(self)
        lifecycle.handle_request("PATCH", self._do_PATCH_internal)

    def _do_PATCH_internal(self, path: str) -> None:
        """Internal PATCH handler with actual routing logic."""
        # RBAC check for all API requests (authorization)
        if path.startswith("/api/"):
            if not self._check_rbac(path, "PATCH"):
                return

        # Try modular handlers first
        if path.startswith("/api/"):
            if self._try_modular_handler(path, {}):
                return

        self.send_error(404, f"Unknown PATCH endpoint: {path}")

    def do_PUT(self) -> None:
        """Handle PUT requests."""
        lifecycle = create_lifecycle_manager(self)
        lifecycle.handle_request("PUT", self._do_PUT_internal)

    def _do_PUT_internal(self, path: str) -> None:
        """Internal PUT handler with actual routing logic."""
        # RBAC check for all API requests (authorization)
        if path.startswith("/api/"):
            if not self._check_rbac(path, "PUT"):
                return

        # Try modular handlers first
        if path.startswith("/api/"):
            if self._try_modular_handler(path, {}):
                return

        self.send_error(404, f"Unknown PUT endpoint: {path}")

    def _serve_file(self, filename: str) -> None:
        """Serve a static file with path traversal protection.

        Delegates to StaticFileHandler for implementation.
        """
        file_handler = StaticFileHandler(static_dir=self.static_dir)

        # Validate path first for better error messages
        is_valid, filepath, error = file_handler.validate_path(filename)
        if not is_valid:
            if error == "Access denied":
                self.send_error(403, error)
            elif error == "Symlinks not allowed":
                self.send_error(403, error)
            elif error == "Invalid path":
                self.send_error(400, error)
            else:
                self.send_error(404, error)
            return

        # Serve the file
        result = file_handler.serve_file(filename)
        if result is None:
            self.send_error(404, "File not found")
            return

        status, headers, content = result

        try:
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self._add_cors_headers()
            self._add_security_headers()
            self.end_headers()
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError) as e:
            logger.debug(f"Client disconnected during file serve: {type(e).__name__}")

    # Note: _send_json, _add_cors_headers, _add_security_headers, _add_rate_limit_headers,
    # and _add_trace_headers are inherited from ResponseHelpersMixin

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
        control_plane_port: int = 8766,
        nomic_loop_port: int = 8767,
        ws_host: str = os.environ.get("ARAGORA_BIND_HOST", "127.0.0.1"),
        http_host: str = os.environ.get("ARAGORA_BIND_HOST", "127.0.0.1"),
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
            control_plane_port: Port for control plane WebSocket (default 8766)
            nomic_loop_port: Port for nomic loop WebSocket (default 8767)
            ws_host: WebSocket bind address (default 127.0.0.1, use ARAGORA_BIND_HOST env)
            http_host: HTTP bind address (default 127.0.0.1, use ARAGORA_BIND_HOST env)
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
        self.control_plane_port = control_plane_port
        self.nomic_loop_port = nomic_loop_port
        self.ws_host = ws_host
        self.http_host = http_host
        self.static_dir = static_dir
        self.nomic_dir = nomic_dir
        self.storage = storage
        self.ssl_cert = ssl_cert
        self.ssl_key = ssl_key
        self.ssl_enabled = bool(ssl_cert and ssl_key)

        # Create WebSocket servers
        self.stream_server = DebateStreamServer(host=ws_host, port=ws_port)
        self.control_plane_stream = ControlPlaneStreamServer(host=ws_host, port=control_plane_port)
        self.nomic_loop_stream = NomicLoopStreamServer(host=ws_host, port=nomic_loop_port)

        # Initialize Supabase persistence if available
        self.persistence = init_persistence(enable_persistence)

        # Setup HTTP handler with base resources
        UnifiedHandler.storage = storage
        UnifiedHandler.static_dir = static_dir
        UnifiedHandler.stream_emitter = self.stream_server.emitter
        UnifiedHandler.control_plane_stream = self.control_plane_stream
        UnifiedHandler.nomic_loop_stream = self.nomic_loop_stream
        UnifiedHandler.persistence = self.persistence

        # Initialize nomic-dependent subsystems
        if nomic_dir:
            self._init_subsystems(nomic_dir)

    def _init_subsystems(self, nomic_dir: Path) -> None:
        """Initialize all nomic directory dependent subsystems.

        Uses SubsystemRegistry for centralized initialization of database-backed
        subsystems, then initializes non-database stores separately.

        Configures the UnifiedHandler class with all required subsystems
        for full API functionality.
        """
        from aragora.server.initialization import (
            init_handler_stores,
            initialize_subsystems,
        )

        UnifiedHandler.nomic_state_file = nomic_dir / "nomic_state.json"

        # Use SubsystemRegistry for batch initialization of database-backed subsystems
        # This centralizes initialization and enables future async/parallel init
        registry = initialize_subsystems(nomic_dir=nomic_dir, enable_persistence=False)

        # Wire registry subsystems to UnifiedHandler
        UnifiedHandler.insight_store = registry.insight_store
        UnifiedHandler.elo_system = registry.elo_system
        UnifiedHandler.flip_detector = registry.flip_detector
        UnifiedHandler.persona_manager = registry.persona_manager
        UnifiedHandler.position_ledger = registry.position_ledger
        UnifiedHandler.debate_embeddings = registry.debate_embeddings
        UnifiedHandler.consensus_memory = registry.consensus_memory
        UnifiedHandler.dissent_retriever = registry.dissent_retriever
        UnifiedHandler.moment_detector = registry.moment_detector

        # Non-database stores and connectors (not yet in registry)
        stores = init_handler_stores(nomic_dir)
        UnifiedHandler.document_store = stores["document_store"]
        UnifiedHandler.audio_store = stores["audio_store"]
        UnifiedHandler.video_generator = stores["video_generator"]
        UnifiedHandler.twitter_connector = stores["twitter_connector"]
        UnifiedHandler.youtube_connector = stores["youtube_connector"]
        UnifiedHandler.user_store = stores["user_store"]
        UnifiedHandler.usage_tracker = stores["usage_tracker"]

        # Initialize DecisionRouter for unified decision routing
        self._init_decision_router()

    def _init_decision_router(self) -> None:
        """Initialize DecisionRouter for unified decision routing.

        The DecisionRouter provides:
        - Unified entry point for debates, workflows, and gauntlets
        - Request caching and deduplication
        - RBAC enforcement
        - Response delivery across channels
        """
        try:
            from aragora.core.decision import DecisionRouter

            UnifiedHandler.decision_router = DecisionRouter(
                enable_caching=True,
                enable_deduplication=True,
                cache_ttl_seconds=3600.0,
            )
            logger.info("DecisionRouter initialized for unified routing")
        except ImportError as e:
            logger.debug(f"DecisionRouter not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize DecisionRouter: {e}")

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
                # Use ThreadingHTTPServer for concurrent request handling
                server = ThreadingHTTPServer((self.http_host, self.http_port), UnifiedHandler)

                # Configure SSL if cert and key are provided
                if self.ssl_enabled:
                    # ssl_enabled is True only when both ssl_cert and ssl_key are set
                    assert self.ssl_cert is not None
                    assert self.ssl_key is not None
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
        """Start HTTP and WebSocket servers."""
        # Run startup sequence (monitoring, tracing, metrics, background tasks, etc.)
        from aragora.server.startup import run_startup_sequence

        startup_status = await run_startup_sequence(
            nomic_dir=self.nomic_dir,
            stream_emitter=self.stream_server.emitter,
        )
        self._watchdog_task = startup_status.get("watchdog_task")

        # Wire Control Plane coordinator to handler
        self._control_plane_coordinator = startup_status.get("control_plane_coordinator")
        if self._control_plane_coordinator:
            from aragora.server.handlers.control_plane import ControlPlaneHandler

            ControlPlaneHandler.coordinator = self._control_plane_coordinator
            logger.info("Control Plane coordinator wired to handler")

        logger.info("Starting unified server...")
        protocol = "https" if self.ssl_enabled else "http"
        logger.info(f"  HTTP API:   {protocol}://localhost:{self.http_port}")
        logger.info(f"  WebSocket:  ws://localhost:{self.ws_port}")
        logger.info(f"  Control Plane WS: ws://localhost:{self.control_plane_port}")
        logger.info(f"  Nomic Loop WS: ws://localhost:{self.nomic_loop_port}")
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

        # Start all WebSocket servers concurrently
        await asyncio.gather(
            self.stream_server.start(),
            self.control_plane_stream.start(),
            self.nomic_loop_stream.start(),
        )

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

        Delegates to ShutdownSequence for structured, phase-based shutdown.

        Args:
            timeout: Maximum seconds to wait for all shutdown phases
        """
        sequence = create_server_shutdown_sequence(self)
        await sequence.execute_all(overall_timeout=timeout)

    @property
    def is_shutting_down(self) -> bool:
        """Check if server is in shutdown mode."""
        return getattr(self, "_shutting_down", False)


async def run_unified_server(
    http_port: int = 8080,
    ws_port: int = 8765,
    http_host: Optional[str] = None,
    ws_host: Optional[str] = None,
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
        http_host: Bind address for HTTP (default: from ARAGORA_BIND_HOST or 127.0.0.1)
        ws_host: Bind address for WebSocket (default: from ARAGORA_BIND_HOST or 127.0.0.1)
        static_dir: Directory containing static files (dashboard build)
        nomic_dir: Path to .nomic directory for state access
        ssl_cert: Path to SSL certificate file (optional)
        ssl_key: Path to SSL private key file (optional)

    Environment variables:
        ARAGORA_BIND_HOST: Default bind address (default: 127.0.0.1)
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

    # Validate configuration at startup (comprehensive validator with security checks)
    from aragora.config import validate_all, ValidatorConfigurationError

    try:
        validation_result = validate_all(strict=False)
        if validation_result.get("errors"):
            for error in validation_result["errors"]:
                logger.error(f"[server] Config error: {error}")
            raise ValidatorConfigurationError(
                f"Configuration validation failed with {len(validation_result['errors'])} errors"
            )
        if validation_result.get("warnings"):
            for warning in validation_result["warnings"]:
                logger.warning(f"[server] Config warning: {warning}")
        logger.info("[server] Configuration validated successfully")
    except ValidatorConfigurationError:
        raise
    except Exception as e:
        logger.warning(f"[server] Config validation skipped: {e}")

    # Initialize storage from nomic directory
    storage = None
    if nomic_dir:
        # Ensure nomic_dir exists - critical for debate persistence
        try:
            nomic_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[server] Nomic directory ready: {nomic_dir}")
        except (OSError, PermissionError) as e:
            logger.error(f"[server] CRITICAL: Cannot create nomic directory {nomic_dir}: {e}")
            raise RuntimeError(f"Cannot create nomic directory: {e}") from e

        db_path = nomic_dir / "debates.db"
        try:
            storage = DebateStorage(str(db_path))
            logger.info(f"[server] DebateStorage initialized at {db_path}")
        except (OSError, RuntimeError) as e:
            logger.error(f"[server] CRITICAL: Cannot initialize DebateStorage at {db_path}: {e}")
            raise RuntimeError(f"Cannot initialize debate storage: {e}") from e

    # Ensure demo data is loaded for search functionality
    try:
        from aragora.fixtures import ensure_demo_data

        logger.info("[server] Checking demo data initialization...")
        ensure_demo_data()
    except (ImportError, OSError, RuntimeError) as e:
        logger.warning(f"[server] Demo data initialization failed: {e}")

    # Build server kwargs, only passing host params if explicitly provided
    server_kwargs: dict = {
        "http_port": http_port,
        "ws_port": ws_port,
        "static_dir": static_dir,
        "nomic_dir": nomic_dir,
        "storage": storage,
        "ssl_cert": ssl_cert,
        "ssl_key": ssl_key,
    }
    if http_host is not None:
        server_kwargs["http_host"] = http_host
    if ws_host is not None:
        server_kwargs["ws_host"] = ws_host

    server = UnifiedServer(**server_kwargs)
    await server.start()
