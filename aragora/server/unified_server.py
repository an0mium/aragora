"""
Unified server combining HTTP API and WebSocket streaming.

Provides a single entry point for:
- HTTP API at /api/* endpoints
- WebSocket streaming at ws://host:port/ws
- Static file serving for the live dashboard
"""

import asyncio
import os
import signal
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from types import FrameType

    import uvicorn

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
    from aragora.server.stream.canvas_stream import CanvasStreamServer
    from aragora.storage import UserStore
import logging
import time

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

# Import utilities from extracted modules
from aragora.server.http_utils import validate_query_params as _validate_query_params

# Import extracted modules
from aragora.server.request_lifecycle import create_lifecycle_manager
from aragora.server.response_utils import ResponseHelpersMixin
from aragora.server.shutdown_sequence import create_server_shutdown_sequence
from aragora.server.static_file_handler import StaticFileHandler

# Import extracted mixins for focused handler responsibilities
from aragora.server.auth_checks import AuthChecksMixin
from aragora.server.request_utils import MAX_JSON_CONTENT_LENGTH, RequestUtilsMixin
from aragora.server.request_logging import TRUSTED_PROXIES as REQUEST_TRUSTED_PROXIES
from aragora.server.request_logging import RequestLoggingMixin
from aragora.server.debate_controller_mixin import DebateControllerMixin

# DoS protection limits
MAX_MULTIPART_PARTS: int = 10
MAX_CONTENT_LENGTH: int = 100 * 1024 * 1024  # 100MB for uploads
# Note: MAX_JSON_CONTENT_LENGTH is imported from request_utils

# Note: TRUSTED_PROXIES is imported from request_logging
TRUSTED_PROXIES = REQUEST_TRUSTED_PROXIES

# Import from initialization module
from aragora.server.handler_registry import HandlerRegistryMixin
from aragora.server.initialization import init_persistence
from aragora.config import MAX_CONCURRENT_DEBATES as CONFIG_MAX_CONCURRENT_DEBATES

try:
    from aragora.server.handlers.features.documents import DocumentHandler

    MAX_UPLOADS_PER_MINUTE = DocumentHandler.MAX_UPLOADS_PER_MINUTE
    MAX_UPLOADS_PER_HOUR = DocumentHandler.MAX_UPLOADS_PER_HOUR
    _UPLOAD_COUNTS = DocumentHandler._upload_counts
except Exception:
    MAX_UPLOADS_PER_MINUTE = 5
    MAX_UPLOADS_PER_HOUR = 30
    _UPLOAD_COUNTS = OrderedDict()

# Server startup time for uptime tracking
_server_start_time: float = time.time()

MAX_CONCURRENT_DEBATES: int = CONFIG_MAX_CONCURRENT_DEBATES


class _UnifiedHandlerBase(BaseHTTPRequestHandler):
    """Base class to establish proper MRO for mixins.

    This intermediate class ensures mypy understands that UnifiedHandler
    inherits from BaseHTTPRequestHandler, allowing the mixins to properly
    reference methods like send_response, send_header, etc.
    """

    pass


class UnifiedHandler(
    ResponseHelpersMixin,
    HandlerRegistryMixin,
    AuthChecksMixin,
    RequestUtilsMixin,
    RequestLoggingMixin,
    DebateControllerMixin,
    _UnifiedHandlerBase,
):
    """HTTP handler with API endpoints and static file serving.

    Responsibilities are split across focused mixins:
    - ResponseHelpersMixin: JSON responses, CORS, security headers
    - HandlerRegistryMixin: Modular handler routing and dispatch
    - AuthChecksMixin: Rate limiting and RBAC permission checks
    - RequestUtilsMixin: Parameter parsing and content validation
    - RequestLoggingMixin: Request logging and metrics
    - DebateControllerMixin: Debate controller lifecycle management
    """

    storage: DebateStorage | None = None
    MAX_UPLOADS_PER_MINUTE: int = MAX_UPLOADS_PER_MINUTE
    MAX_UPLOADS_PER_HOUR: int = MAX_UPLOADS_PER_HOUR
    MAX_JSON_CONTENT_LENGTH: int = MAX_JSON_CONTENT_LENGTH
    _upload_counts: OrderedDict[str, list] = _UPLOAD_COUNTS
    static_dir: Path | None = None
    stream_emitter: SyncEventEmitter | None = None
    control_plane_stream: Optional["ControlPlaneStreamServer"] = None
    nomic_loop_stream: Optional["NomicLoopStreamServer"] = None
    canvas_stream: Optional["CanvasStreamServer"] = None
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
                # GraphQL endpoints (GraphiQL playground in dev mode)
                "/graphql",
                "/graphiql",
                "/api/graphql",
                "/api/v1/graphql",
                "/graphql/schema",
                "/api/graphql/schema",
                "/api/v1/graphql/schema",
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
    nomic_state_file: Path | None = None
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

    # Note: The following class attributes are inherited from mixins:
    # - _debate_controller, _debate_factory (from DebateControllerMixin)
    # - _request_log_enabled, _slow_request_threshold_ms (from RequestLoggingMixin)
    # - _rate_limit_result (from AuthChecksMixin)

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

    # Note: The following methods are inherited from mixins:
    # - _log_request, _normalize_endpoint, _get_client_ip (from RequestLoggingMixin)
    # - _safe_int, _safe_float, _safe_string, _extract_path_segment, _validate_content_length (from RequestUtilsMixin)
    # - AUTH_EXEMPT_PATHS, AUTH_EXEMPT_PREFIXES, AUTH_EXEMPT_GET_PREFIXES (from AuthChecksMixin)
    # - _check_rate_limit, _check_tier_rate_limit, _check_rbac, _check_upload_rate_limit (from AuthChecksMixin)
    # - _get_debate_controller, _auto_select_agents (from DebateControllerMixin)
    # - _init_handlers(), _log_resource_availability(), _try_modular_handler() (from HandlerRegistryMixin)

    def do_GET(self) -> None:
        """Handle HTTP GET requests.

        Processes GET requests through the request lifecycle with:
        - Query parameter validation for /api/* routes
        - RBAC authorization checks
        - Rate limiting (DoS protection)
        - Modular handler routing for API endpoints
        - Static file serving for non-API routes

        Routes:
            /api/*: Routed through modular handlers (see handler_registry.py)
            /: Serves index.html
            *.html, *.css, *.js, etc.: Static file serving

        Security:
            - Query params validated against whitelist
            - RBAC checks before handler invocation
            - Rate limiting applied to API routes
        """
        lifecycle = create_lifecycle_manager(self)
        lifecycle.handle_request("GET", self._do_GET_internal, with_query=True)

    def _do_GET_internal(self, path: str, query: dict[str, Any]) -> None:
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
            # Fallback for auth/me - return 401 instead of 404 when handler unavailable
            if path in ("/api/auth/me", "/api/v1/auth/me"):
                self._send_json(
                    {"error": "Authentication required", "code": "auth_required"},
                    status=401,
                )
                return
            # Return 404 for unhandled API endpoints (don't fall through to static files)
            self._send_json(
                {"error": f"API endpoint not found: {path}", "code": "not_found"},
                status=404,
            )
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
        """Handle HTTP POST requests.

        Processes POST requests for creating resources through:
        - Request body parsing (JSON, multipart/form-data)
        - RBAC authorization checks
        - Modular handler routing

        Common POST endpoints:
            /api/debates: Create new debate
            /api/auth/login: User authentication
            /api/documents: Upload documents
            /api/webhooks/*: Webhook receivers

        Request body is parsed from Content-Type header:
            application/json: Parsed as JSON
            multipart/form-data: Parsed for file uploads
        """
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
        """Handle HTTP DELETE requests.

        Processes DELETE requests for removing resources through:
        - RBAC authorization checks (requires appropriate permissions)
        - Modular handler routing

        DELETE operations are idempotent - multiple requests for the
        same resource return success even if already deleted.

        Common DELETE endpoints:
            /api/debates/{id}: Delete debate
            /api/documents/{id}: Delete document
            /api/users/{id}: Delete user (admin only)
        """
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
        """Handle HTTP PATCH requests.

        Processes PATCH requests for partial resource updates through:
        - Request body parsing (JSON with partial fields)
        - RBAC authorization checks
        - Modular handler routing

        PATCH applies partial modifications to a resource. Only fields
        included in the request body are updated; omitted fields remain
        unchanged.

        Common PATCH endpoints:
            /api/debates/{id}: Update debate metadata
            /api/users/{id}: Update user profile
            /api/settings: Update settings
        """
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
        """Handle HTTP PUT requests.

        Processes PUT requests for full resource replacement through:
        - Request body parsing (complete resource representation)
        - RBAC authorization checks
        - Modular handler routing

        PUT replaces the entire resource with the provided representation.
        All fields must be included; omitted fields are set to defaults
        or null.

        Common PUT endpoints:
            /api/debates/{id}: Replace debate configuration
            /api/documents/{id}: Replace document content
        """
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

    def log_message(self, format: str, *args: Any) -> None:
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
        canvas_port: int = 8768,
        ws_host: str = os.environ.get("ARAGORA_BIND_HOST", "127.0.0.1"),
        http_host: str = os.environ.get("ARAGORA_BIND_HOST", "127.0.0.1"),
        static_dir: Path | None = None,
        nomic_dir: Path | None = None,
        storage: DebateStorage | None = None,
        enable_persistence: bool = True,
        ssl_cert: str | None = None,
        ssl_key: str | None = None,
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
        self.http_port: int = http_port
        self.ws_port: int = ws_port
        self.control_plane_port: int = control_plane_port
        self.nomic_loop_port: int = nomic_loop_port
        self.canvas_port: int = canvas_port
        self.ws_host: str = ws_host
        self.http_host: str = http_host
        self.static_dir: Path | None = static_dir
        self.nomic_dir: Path | None = nomic_dir
        self.storage: DebateStorage | None = storage
        self.ssl_cert: str | None = ssl_cert
        self.ssl_key: str | None = ssl_key
        self.ssl_enabled: bool = bool(ssl_cert and ssl_key)

        # HTTP server reference for graceful shutdown
        self._http_server: ThreadingHTTPServer | None = None
        # Uvicorn server reference for graceful shutdown
        self._uvicorn_server: "uvicorn.Server | None" = None

        # Create WebSocket servers
        self.stream_server: DebateStreamServer = DebateStreamServer(host=ws_host, port=ws_port)
        self.control_plane_stream: ControlPlaneStreamServer = ControlPlaneStreamServer(
            host=ws_host, port=control_plane_port
        )
        self.nomic_loop_stream: NomicLoopStreamServer = NomicLoopStreamServer(
            host=ws_host, port=nomic_loop_port
        )

        # Create Canvas WebSocket server
        self.canvas_stream: Optional["CanvasStreamServer"] = None
        try:
            from aragora.server.stream.canvas_stream import CanvasStreamServer

            self.canvas_stream = CanvasStreamServer(host=ws_host, port=canvas_port)
        except ImportError:
            logger.warning("Canvas stream server not available")

        # Initialize Supabase persistence if available
        self.persistence: Optional["SupabaseClient"] = init_persistence(enable_persistence)

        # Setup HTTP handler with base resources
        UnifiedHandler.storage = storage
        UnifiedHandler.static_dir = static_dir
        UnifiedHandler.stream_emitter = self.stream_server.emitter
        UnifiedHandler.control_plane_stream = self.control_plane_stream
        UnifiedHandler.nomic_loop_stream = self.nomic_loop_stream
        UnifiedHandler.canvas_stream = self.canvas_stream
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
        except (TypeError, ValueError, RuntimeError, OSError) as e:
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
                self._http_server = server

                # Configure SSL if cert and key are provided
                if self.ssl_enabled:
                    # ssl_enabled is True only when both ssl_cert and ssl_key are set
                    if self.ssl_cert is None:
                        raise ValueError("ssl_cert required when ssl_enabled is True")
                    if self.ssl_key is None:
                        raise ValueError("ssl_key required when ssl_enabled is True")
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

    async def start(self, use_parallel_init: bool | None = None) -> None:
        """Start HTTP and WebSocket servers.

        Args:
            use_parallel_init: Use parallel initialization for faster startup.
                If None, determined by ARAGORA_PARALLEL_INIT env var (default: True).
        """
        import time as time_mod

        startup_start = time_mod.perf_counter()

        # Determine whether to use parallel initialization
        if use_parallel_init is None:
            use_parallel_init = os.environ.get("ARAGORA_PARALLEL_INIT", "true").lower() in (
                "1",
                "true",
                "yes",
            )

        # Run startup sequence (monitoring, tracing, metrics, background tasks, etc.)
        if use_parallel_init:
            from aragora.server.startup import parallel_init

            logger.info("[startup] Using parallel initialization")
            startup_status = await parallel_init(
                nomic_dir=self.nomic_dir,
                stream_emitter=self.stream_server.emitter,
            )

            # Log parallel init timing
            if startup_status.get("_parallel_init_duration_ms"):
                duration_ms = startup_status["_parallel_init_duration_ms"]
                logger.info(f"[startup] Parallel init completed in {duration_ms:.0f}ms")

            # Extract watchdog task from results
            self._watchdog_task = startup_status.get("watchdog_task")
        else:
            from aragora.server.startup import run_startup_sequence

            logger.info("[startup] Using sequential initialization")
            startup_status = await run_startup_sequence(
                nomic_dir=self.nomic_dir,
                stream_emitter=self.stream_server.emitter,
            )
            self._watchdog_task = startup_status.get("watchdog_task")

        # Upgrade stores from SQLite to PostgreSQL now that the pool exists.
        # init_handler_stores() ran in __init__() before the pool was available,
        # so stores initially fell back to SQLite.  Now we can replace them.
        if startup_status.get("postgres_pool", {}).get("enabled"):
            try:
                from aragora.server.initialization import upgrade_handler_stores

                upgrade_results = await upgrade_handler_stores(self.nomic_dir)
                if upgrade_results:
                    logger.info(f"Store upgrades: {upgrade_results}")
            except (ImportError, OSError, RuntimeError, ValueError) as e:
                logger.warning(f"Store upgrade failed (continuing with SQLite): {e}")

        # Wire Control Plane coordinator to handler
        self._control_plane_coordinator = startup_status.get("control_plane_coordinator")
        if self._control_plane_coordinator:
            from aragora.server.handlers.control_plane import ControlPlaneHandler

            ControlPlaneHandler.coordinator = self._control_plane_coordinator
            logger.info("Control Plane coordinator wired to handler")

        # Log GraphQL status
        if startup_status.get("graphql"):
            logger.info("GraphQL API enabled at /graphql")

        # Initialize handlers eagerly at startup to avoid first-request latency
        # and ensure route index is built before accepting requests
        UnifiedHandler._init_handlers()

        # Log startup timing
        startup_elapsed_ms = (time_mod.perf_counter() - startup_start) * 1000
        init_mode = "parallel" if use_parallel_init else "sequential"

        logger.info("Starting unified server...")
        logger.info(f"  Init mode:  {init_mode} ({startup_elapsed_ms:.0f}ms)")
        protocol = "https" if self.ssl_enabled else "http"
        logger.info(f"  HTTP API:   {protocol}://localhost:{self.http_port}")
        logger.info(f"  WebSocket:  ws://localhost:{self.ws_port}")
        logger.info(f"  Control Plane WS: ws://localhost:{self.control_plane_port}")
        logger.info(f"  Nomic Loop WS: ws://localhost:{self.nomic_loop_port}")
        if self.canvas_stream:
            logger.info(f"  Canvas WS: ws://localhost:{self.canvas_port}")
        if self.ssl_enabled:
            logger.info(f"  SSL:        enabled (cert: {self.ssl_cert})")
        if self.static_dir:
            logger.info(f"  Static dir: {self.static_dir}")
        if self.nomic_dir:
            logger.info(f"  Nomic dir:  {self.nomic_dir}")

        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()

        # Start HTTP server
        use_fastapi = os.environ.get("ARAGORA_USE_FASTAPI", "").lower() in ("1", "true", "yes")
        fastapi_port = int(os.environ.get("ARAGORA_FASTAPI_PORT", str(self.http_port)))

        if use_fastapi:
            # Start FastAPI (async-native, high concurrency)
            logger.info(f"  FastAPI:    http://localhost:{fastapi_port} (async mode)")
            fastapi_task = asyncio.create_task(self._start_fastapi_server(fastapi_port))
            stream_tasks = [
                fastapi_task,
                self.stream_server.start(),
                self.control_plane_stream.start(),
                self.nomic_loop_stream.start(),
            ]
        else:
            # Start legacy ThreadingHTTPServer in background thread
            self._http_thread = Thread(target=self._run_http_server, daemon=True)
            self._http_thread.start()
            stream_tasks = [
                self.stream_server.start(),
                self.control_plane_stream.start(),
                self.nomic_loop_stream.start(),
            ]

        # Start all WebSocket servers concurrently
        if self.canvas_stream:
            stream_tasks.append(self.canvas_stream.start())
        await asyncio.gather(*stream_tasks)

    async def _start_fastapi_server(self, port: int) -> None:
        """Start FastAPI server using uvicorn.

        This replaces ThreadingHTTPServer with an async-native server
        for better concurrency (10,000+ vs ~500 concurrent connections).

        Args:
            port: Port to bind the FastAPI server to.
        """
        try:
            import uvicorn
            from aragora.server.fastapi import create_app

            app = create_app(nomic_dir=self.nomic_dir)
            config = uvicorn.Config(
                app,
                host=self.http_host,
                port=port,
                log_level="info",
                access_log=True,
            )
            server = uvicorn.Server(config)
            self._uvicorn_server = server  # Store reference for graceful shutdown
            await server.serve()
        except ImportError:
            logger.error(
                "[server] FastAPI/uvicorn not installed. "
                "Install with: pip install 'aragora[fastapi]' or pip install fastapi uvicorn"
            )
            raise

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(signum: int, frame: "FrameType | None") -> None:
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
    http_host: str | None = None,
    ws_host: str | None = None,
    static_dir: Path | None = None,
    nomic_dir: Path | None = None,
    ssl_cert: str | None = None,
    ssl_key: str | None = None,
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
    # Check for minimal mode (SQLite + in-memory, no Redis/Postgres)
    from aragora.config.minimal import is_minimal_mode, apply_minimal_mode

    if is_minimal_mode():
        applied = apply_minimal_mode()
        logger.info(f"[server] Running in minimal mode (SQLite + in-memory): {applied}")

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
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as e:
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
    server_kwargs: dict[str, Any] = {
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
