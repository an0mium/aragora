"""
Unified server combining HTTP API and WebSocket streaming.

Provides a single entry point for:
- HTTP API at /api/* endpoints
- WebSocket streaming at ws://host:port/ws
- Static file serving for the live dashboard
"""

import asyncio
import atexit
import json
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import Any, Coroutine, Optional, Dict
from urllib.parse import urlparse, parse_qs

from .stream import DebateStreamServer, SyncEventEmitter, StreamEvent, StreamEventType, create_arena_hooks
from .storage import DebateStorage
from .documents import DocumentStore, parse_document, get_supported_formats, SUPPORTED_EXTENSIONS
from ..broadcast.storage import AudioFileStore
from ..broadcast.rss_gen import PodcastFeedGenerator, PodcastConfig, PodcastEpisode
from ..connectors.twitter_poster import TwitterPosterConnector, DebateContentFormatter
from ..connectors.youtube_uploader import YouTubeUploaderConnector, YouTubeVideoMetadata, create_video_metadata_from_debate
from ..broadcast.video_gen import VideoGenerator
from .auth import auth_config, check_auth
from .cors_config import cors_config

# For ad-hoc debates
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import uuid
import logging

# Configure module logger
logger = logging.getLogger(__name__)

# Import centralized config and error utilities
from aragora.config import (
    DB_INSIGHTS_PATH,
    DB_PERSONAS_PATH,
    DB_TIMEOUT_SECONDS,
    MAX_AGENTS_PER_DEBATE,
    MAX_CONCURRENT_DEBATES,
    ALLOWED_AGENT_TYPES,
)
from aragora.server.error_utils import safe_error_message as _safe_error_message
from aragora.server.validation import SAFE_ID_PATTERN, validate_id

# Import utilities from extracted modules
from aragora.server.http_utils import (
    ALLOWED_QUERY_PARAMS,
    validate_query_params as _validate_query_params,
    safe_float as _safe_float,
    safe_int as _safe_int,
    run_async as _run_async,
)
from aragora.server.debate_utils import (
    get_active_debates,
    get_active_debates_lock,
    update_debate_status as _update_debate_status,
    cleanup_stale_debates as _cleanup_stale_debates,
    increment_cleanup_counter,
    wrap_agent_for_streaming as _wrap_agent_for_streaming,
    _DEBATE_TTL_SECONDS,
    _active_debates,
    _active_debates_lock,
)
from aragora.server.debate_controller import (
    DebateController,
    DebateRequest,
    DebateResponse,
)
from aragora.server.debate_factory import DebateFactory

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
    p.strip() for p in os.getenv('ARAGORA_TRUSTED_PROXIES', '127.0.0.1,::1,localhost').split(',')
)


# Import optional subsystems from centralized initialization module
from aragora.server.initialization import (
    # Availability flags
    PERSISTENCE_AVAILABLE,
    INSIGHTS_AVAILABLE,
    RANKING_AVAILABLE,
    FLIP_DETECTOR_AVAILABLE,
    DEBATE_AVAILABLE,
    PERSONAS_AVAILABLE,
    EMBEDDINGS_AVAILABLE,
    CONSENSUS_MEMORY_AVAILABLE,
    CALIBRATION_AVAILABLE,
    PULSE_AVAILABLE,
    VERIFICATION_AVAILABLE,
    CONTINUUM_AVAILABLE,
    POSITION_LEDGER_AVAILABLE,
    MOMENT_DETECTOR_AVAILABLE,
    POSITION_TRACKER_AVAILABLE,
    BROADCAST_AVAILABLE,
    RELATIONSHIP_TRACKER_AVAILABLE,
    CRITIQUE_STORE_AVAILABLE,
    EXPORT_AVAILABLE,
    PROBER_AVAILABLE,
    REDTEAM_AVAILABLE,
    LABORATORY_AVAILABLE,
    BELIEF_NETWORK_AVAILABLE,
    PROVENANCE_AVAILABLE,
    FORMAL_VERIFICATION_AVAILABLE,
    IMPASSE_DETECTOR_AVAILABLE,
    CONVERGENCE_DETECTOR_AVAILABLE,
    ROUTING_AVAILABLE,
    TOURNAMENT_AVAILABLE,
    EVOLUTION_AVAILABLE,
    INSIGHT_EXTRACTOR_AVAILABLE,
    # Classes (for type hints and direct use)
    Arena,
    DebateProtocol,
    create_agent,
    Environment,
    format_flip_for_ui,
    format_consistency_for_ui,
    PositionLedger,
    # Broadcast module
    broadcast_debate,
    DebateTrace,
    # RelationshipTracker
    RelationshipTracker,
    # CritiqueStore
    CritiqueStore,
    # Export module
    DebateArtifact,
    CSVExporter,
    DOTExporter,
    StaticHTMLExporter,
    # Prober and RedTeam
    CapabilityProber,
    RedTeamMode,
    # Laboratory
    PersonaLaboratory,
    # Belief network
    BeliefNetwork,
    BeliefPropagationAnalyzer,
    # Provenance
    ProvenanceTracker,
    # Verification
    FormalVerificationManager,
    get_formal_verification_manager,
    # Impasse and Convergence
    ImpasseDetector,
    ConvergenceDetector,
    # Routing
    AgentSelector,
    AgentProfile,
    TaskRequirements,
    # Tournament
    TournamentManager,
    # Evolution
    PromptEvolver,
    # Memory
    ContinuumMemory,
    MemoryTier,
    # Insights
    InsightExtractor,
    # Initialization functions
    init_persistence,
    init_insight_store,
    init_elo_system,
    init_flip_detector,
    init_persona_manager,
    init_position_ledger,
    init_debate_embeddings,
    init_consensus_memory,
    init_moment_detector,
    initialize_subsystems,
    SubsystemRegistry,
    # Classes that need explicit checks
    ConsensusMemory,
    DissentRetriever,
    MomentDetector,
)

# Import static file serving utilities
from aragora.server.static_server import (
    serve_static_file,
    serve_audio_file,
    get_content_type,
)

# Modular HTTP handlers via registry mixin
from aragora.server.handler_registry import (
    HandlerRegistryMixin,
    HANDLERS_AVAILABLE,
)

# Server startup time for uptime tracking
_server_start_time: float = time.time()


class UnifiedHandler(HandlerRegistryMixin, BaseHTTPRequestHandler):
    """HTTP handler with API endpoints and static file serving.

    Handler routing is provided by HandlerRegistryMixin from handler_registry.py.
    """

    storage: Optional[DebateStorage] = None
    static_dir: Optional[Path] = None
    stream_emitter: Optional[SyncEventEmitter] = None
    nomic_state_file: Optional[Path] = None
    persistence: Optional["SupabaseClient"] = None  # Supabase client for history
    insight_store: Optional["InsightStore"] = None  # InsightStore for debate insights
    elo_system: Optional["EloSystem"] = None  # EloSystem for agent rankings
    document_store: Optional[DocumentStore] = None  # Document store for uploads
    audio_store: Optional[AudioFileStore] = None  # Audio store for broadcasts
    twitter_connector: Optional[TwitterPosterConnector] = None  # Twitter posting connector
    youtube_connector: Optional[YouTubeUploaderConnector] = None  # YouTube upload connector
    video_generator: Optional[VideoGenerator] = None  # Video generator for YouTube
    flip_detector: Optional["FlipDetector"] = None  # FlipDetector for position reversals
    persona_manager: Optional["PersonaManager"] = None  # PersonaManager for agent specialization
    debate_embeddings: Optional["DebateEmbeddingsDatabase"] = None  # Historical memory
    position_tracker: Optional["PositionTracker"] = None  # PositionTracker for truth-grounded personas
    position_ledger: Optional["PositionLedger"] = None  # PositionLedger for grounded positions
    consensus_memory: Optional["ConsensusMemory"] = None  # ConsensusMemory for historical positions
    dissent_retriever: Optional["DissentRetriever"] = None  # DissentRetriever for minority views
    moment_detector: Optional["MomentDetector"] = None  # MomentDetector for significant moments

    # Note: Modular HTTP handlers are provided by HandlerRegistryMixin
    # Handler instance variables (_system_handler, etc.) and _handlers_initialized
    # are inherited from the mixin along with _init_handlers() and _try_modular_handler()

    # Debate controller and factory (initialized lazily)
    # Note: DebateController manages its own ThreadPoolExecutor with proper locking
    _debate_controller: Optional[DebateController] = None
    _debate_factory: Optional[DebateFactory] = None

    # Upload rate limiting (IP-based, independent of auth)
    _upload_counts: Dict[str, list] = {}  # IP -> list of upload timestamps
    _upload_counts_lock = threading.Lock()
    MAX_UPLOADS_PER_MINUTE = 5  # Maximum uploads per IP per minute
    MAX_UPLOADS_PER_HOUR = 30  # Maximum uploads per IP per hour

    # Request logging for observability
    _request_log_enabled = True  # Can be disabled via environment
    _slow_request_threshold_ms = 1000  # Log warning for requests slower than this

    def _log_request(self, method: str, path: str, status: int, duration_ms: float, extra: dict = None) -> None:
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

    def _safe_int(self, query: dict, key: str, default: int, max_val: int = 100) -> int:
        """Safely parse integer query param with bounds checking."""
        try:
            val = int(query.get(key, [default])[0])
            return min(max(val, 1), max_val)
        except (ValueError, IndexError, TypeError):
            return default

    def _safe_float(self, query: dict, key: str, default: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Safely parse float query param with bounds checking."""
        try:
            val = float(query.get(key, [default])[0])
            return min(max(val, min_val), max_val)
        except (ValueError, IndexError, TypeError):
            return default

    def _safe_string(self, value: str, max_len: int = 500, pattern: Optional[str] = None) -> Optional[str]:
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

    def _extract_path_segment(self, path: str, index: int, segment_name: str = "id") -> Optional[str]:
        """Safely extract path segment with bounds checking.

        Returns None and sends 400 error if segment is missing.
        """
        parts = path.split('/')
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
            content_length = int(self.headers.get('Content-Length', '0'))
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
                self._send_json(
                    {"error": "Rate limit exceeded. Try again later."},
                    status=429
                )
            else:
                # Auth failed
                self._send_json(
                    {"error": "Authentication required"},
                    status=401
                )
            return False

        # Add rate limit headers
        if remaining >= 0:
            self.send_response(200)
            self.send_header("X-RateLimit-Remaining", str(remaining))
            self.send_header("X-RateLimit-Limit", str(auth_config.rate_limit_per_minute))

        return True

    def _check_upload_rate_limit(self) -> bool:
        """Check IP-based upload rate limit. Returns True if allowed, False if blocked.

        Uses sliding window rate limiting per IP address.
        """
        import time

        # Get client IP (validate proxy headers for security)
        remote_ip = self.client_address[0] if hasattr(self, 'client_address') else 'unknown'
        client_ip = remote_ip  # Default to direct connection IP
        if remote_ip in TRUSTED_PROXIES:
            # Only trust X-Forwarded-For from trusted proxies
            forwarded = self.headers.get('X-Forwarded-For', '')
            if forwarded:
                first_ip = forwarded.split(',')[0].strip()
                if first_ip:
                    client_ip = first_ip

        now = time.time()
        one_minute_ago = now - 60
        one_hour_ago = now - 3600

        with UnifiedHandler._upload_counts_lock:
            # Get or create upload history for this IP
            if client_ip not in UnifiedHandler._upload_counts:
                UnifiedHandler._upload_counts[client_ip] = []

            # Clean up old entries
            UnifiedHandler._upload_counts[client_ip] = [
                ts for ts in UnifiedHandler._upload_counts[client_ip]
                if ts > one_hour_ago
            ]

            timestamps = UnifiedHandler._upload_counts[client_ip]

            # Check per-minute limit
            recent_minute = sum(1 for ts in timestamps if ts > one_minute_ago)
            if recent_minute >= UnifiedHandler.MAX_UPLOADS_PER_MINUTE:
                self._send_json({
                    "error": f"Upload rate limit exceeded. Max {UnifiedHandler.MAX_UPLOADS_PER_MINUTE} uploads per minute.",
                    "retry_after": 60
                }, status=429)
                return False

            # Check per-hour limit
            if len(timestamps) >= UnifiedHandler.MAX_UPLOADS_PER_HOUR:
                self._send_json({
                    "error": f"Upload rate limit exceeded. Max {UnifiedHandler.MAX_UPLOADS_PER_HOUR} uploads per hour.",
                    "retry_after": 3600
                }, status=429)
                return False

            # Record this upload
            UnifiedHandler._upload_counts[client_ip].append(now)

        return True

    def _revoke_token(self) -> None:
        """Handle token revocation request.

        Revokes a token so it can no longer be used for authentication.
        Requires the token to be revoked in the request body.
        """
        import json

        # Only allow authenticated requests to revoke tokens
        if not self._check_rate_limit():
            return

        content_length = self._validate_content_length()
        if content_length is None:
            return  # Error already sent

        try:
            if content_length > 0:
                body = self.rfile.read(content_length).decode('utf-8')
                data = json.loads(body)
            else:
                data = {}
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=400)
            return

        token_to_revoke = data.get('token')
        reason = data.get('reason', '')

        if not token_to_revoke:
            self._send_json({"error": "Missing 'token' field"}, status=400)
            return

        # Revoke the token
        revoked = auth_config.revoke_token(token_to_revoke, reason)

        if revoked:
            self._send_json({
                "status": "revoked",
                "message": "Token has been revoked and can no longer be used",
                "revoked_tokens_count": auth_config.get_revocation_count(),
            })
        else:
            self._send_json({"error": "Failed to revoke token"}, status=500)

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

            # Create controller
            UnifiedHandler._debate_controller = DebateController(
                factory=factory,
                emitter=self.stream_emitter,
                elo_system=self.elo_system,
                auto_select_fn=self._auto_select_agents,
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
            return 'gemini,anthropic-api'  # Fallback

        try:
            # Build task requirements from question and config
            requirements = TaskRequirements(
                task_id=f"debate-{uuid.uuid4().hex[:8]}",
                description=question[:500],  # Truncate for safety
                primary_domain=config.get('primary_domain', 'general'),
                secondary_domains=config.get('secondary_domains', []),
                required_traits=config.get('required_traits', []),
                min_agents=min(max(config.get('min_agents', 2), 2), 5),
                max_agents=min(max(config.get('max_agents', 4), 2), 8),
                quality_priority=min(max(config.get('quality_priority', 0.7), 0), 1),
                diversity_preference=min(max(config.get('diversity_preference', 0.5), 0), 1),
            )

            # Create selector with ELO system and persona manager
            selector = AgentSelector(
                elo_system=self.elo_system,
                persona_manager=self.persona_manager,
            )

            # Populate agent pool from allowed types
            for agent_type in ALLOWED_AGENT_TYPES:
                selector.register_agent(AgentProfile(
                    name=agent_type,
                    agent_type=agent_type,
                ))

            # Select optimal team
            team = selector.select_team(requirements)

            # Build agent string with roles if available
            agent_specs = []
            for agent in team.agents:
                role = team.roles.get(agent.name, '')
                if role:
                    agent_specs.append(f"{agent.agent_type}:{role}")
                else:
                    agent_specs.append(agent.agent_type)

            logger.info(f"[auto_select] Selected team: {agent_specs} (rationale: {team.rationale[:100]})")
            return ','.join(agent_specs)

        except Exception as e:
            logger.warning(f"[auto_select] Failed: {e}, using fallback")
            return 'gemini,anthropic-api'  # Fallback on error

    def do_GET(self) -> None:
        """Handle GET requests."""
        start_time = time.time()
        status_code = 200  # Default, updated by handlers
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            self._do_GET_internal(path, query)
        except Exception as e:
            status_code = 500
            logger.exception(f"[request] Unhandled exception in GET {path}: {e}")
            try:
                self._send_json({"error": "Internal server error"}, status=500)
            except Exception as send_err:
                logger.debug(f"Could not send error response (already sent?): {send_err}")
        finally:
            duration_ms = (time.time() - start_time) * 1000
            # Log API requests (skip static file logging for noise reduction)
            if path.startswith('/api/'):
                self._log_request("GET", path, status_code, duration_ms)

    def _do_GET_internal(self, path: str, query: dict) -> None:
        """Internal GET handler with actual routing logic."""
        # Validate query parameters against whitelist (security)
        if query and path.startswith('/api/'):
            is_valid, error_msg = _validate_query_params(query)
            if not is_valid:
                self._send_json({"error": error_msg}, status=400)
                return

        # Rate limit all API GET requests (DoS protection)
        if path.startswith('/api/'):
            if not self._check_rate_limit():
                return

        # Try modular handlers first (gradual migration)
        if path.startswith('/api/'):
            if self._try_modular_handler(path, query):
                return

        # Insights API - NOW HANDLED BY InsightsHandler
        # Note: /api/debates/*, /api/health, /api/nomic/*, /api/history/*
        # are now handled by modular handlers (DebatesHandler, SystemHandler)

        # Note: /api/leaderboard, /api/matches/recent, /api/agent/*/history,
        # /api/calibration/leaderboard, /api/agent/*/calibration
        # are now handled by AgentsHandler

        # Pulse API - NOW HANDLED BY PulseHandler
        # Document API - NOW HANDLED BY DocumentHandler
        # Replay API - NOW HANDLED BY ReplaysHandler
        # Flip Detection API - NOW HANDLED BY AgentsHandler
        # Persona API - NOW HANDLED BY PersonaHandler

        # Consensus Memory API - NOW HANDLED BY ConsensusHandler
        # Combined Agent Profile API - NOW HANDLED BY AgentsHandler

        # Debate Analytics API - NOW HANDLED BY AnalyticsHandler
        # Modes API - NOW HANDLED BY SystemHandler
        # Agent Position Tracking API - NOW HANDLED BY AgentsHandler
        # Agent Relationship Network API - NOW HANDLED BY AgentsHandler
        # Agent Moments API - NOW HANDLED BY AgentsHandler

        # System Statistics API - NOW HANDLED BY AnalyticsHandler
        # Critiques/Reputation API - NOW HANDLED BY CritiqueHandler

        # Agent Comparison API - NOW HANDLED BY AgentsHandler
        # Head-to-Head API - NOW HANDLED BY AgentsHandler
        # Opponent Briefing API - NOW HANDLED BY AgentsHandler
        # Introspection API - NOW HANDLED BY IntrospectionHandler
        # Calibration Curve API - NOW HANDLED BY CalibrationHandler
        # Meta-Critique API - NOW HANDLED BY DebatesHandler
        # Debate Graph Stats API - NOW HANDLED BY DebatesHandler

        # Laboratory/Belief Network APIs - NOW HANDLED BY BeliefHandler
        # Tournament API - NOW HANDLED BY TournamentHandler
        # Best Team Combinations API - NOW HANDLED BY RoutingHandler
        # Evolution History API - NOW HANDLED BY EvolutionHandler
        # Load-Bearing Claims API - NOW HANDLED BY BeliefHandler
        # Calibration Summary API - NOW HANDLED BY CalibrationHandler
        # Continuum Memory API - NOW HANDLED BY MemoryHandler
        # Formal Verification Status API - NOW HANDLED BY VerificationHandler
        # Plugins API (GET) - NOW HANDLED BY PluginsHandler
        # Genesis API - NOW HANDLED BY GenesisHandler

        # Audio file serving (for podcast broadcasts)
        # NOTE: Audio, podcast, and YouTube routes are NOW HANDLED BY BroadcastHandler

        # Static file serving
        elif path in ('/', '/index.html'):
            self._serve_file('index.html')
        elif path.endswith(('.html', '.css', '.js', '.json', '.ico', '.svg', '.png')):
            self._serve_file(path.lstrip('/'))
        else:
            # Try serving as a static file
            self._serve_file(path.lstrip('/'))

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(200)
        self._add_cors_headers()
        self.end_headers()

    def do_POST(self) -> None:
        """Handle POST requests."""
        start_time = time.time()
        status_code = 200  # Default, updated by handlers
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            self._do_POST_internal(path)
        except Exception as e:
            status_code = 500
            logger.exception(f"[request] Unhandled exception in POST {path}: {e}")
            try:
                self._send_json({"error": "Internal server error"}, status=500)
            except Exception as send_err:
                logger.debug(f"Could not send error response (already sent?): {send_err}")
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self._log_request("POST", path, status_code, duration_ms)

    def _do_POST_internal(self, path: str) -> None:
        """Internal POST handler with actual routing logic."""
        # Try modular handlers first (gradual migration)
        if path.startswith('/api/'):
            if self._try_modular_handler(path, {}):
                return

        if path == '/api/documents/upload':
            self._upload_document()
        elif path == '/api/debate':
            self._start_debate()
        # NOTE: Broadcast, publishing, laboratory, routing, verification, probes,
        # plugins, insights routes are NOW HANDLED BY modular handlers (BroadcastHandler,
        # LaboratoryHandler, RoutingHandler, VerificationHandler, ProbesHandler,
        # PluginsHandler, AuditingHandler, InsightsHandler)
        elif path.startswith('/api/debates/') and path.endswith('/verify'):
            debate_id = self._extract_path_segment(path, 3, "debate_id")
            if debate_id:
                self._verify_debate_outcome(debate_id)
        elif path == '/api/auth/revoke':
            self._revoke_token()
        else:
            self.send_error(404, f"Unknown POST endpoint: {path}")

    def _upload_document(self) -> None:
        """Handle document upload. Rate limited by auth and IP-based limits."""
        # Rate limit uploads (auth-based when enabled)
        if not self._check_rate_limit():
            return

        # IP-based upload rate limiting (always active, prevents DoS)
        if not self._check_upload_rate_limit():
            return

        if not self.document_store:
            self._send_json({"error": "Document storage not configured"}, status=500)
            return

        # Get content length with validation
        try:
            content_length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            self._send_json({"error": "Invalid Content-Length header"}, status=400)
            return

        if content_length == 0:
            self._send_json({"error": "No content provided"}, status=400)
            return

        # Check max size (10MB)
        max_size = 10 * 1024 * 1024
        if content_length > max_size:
            self._send_json({"error": "File too large. Max size: 10MB"}, status=413)
            return

        content_type = self.headers.get('Content-Type', '')

        # Handle multipart form data
        if 'multipart/form-data' in content_type:
            # Parse boundary
            boundary = None
            for part in content_type.split(';'):
                if 'boundary=' in part:
                    # Use maxsplit=1 to handle boundaries containing '='
                    parts = part.split('=', 1)
                    if len(parts) == 2 and parts[1].strip():
                        boundary = parts[1].strip()
                    break

            if not boundary:
                self._send_json({"error": "No boundary in multipart data"}, status=400)
                return

            # Read body
            body = self.rfile.read(content_length)

            # Parse multipart - simple implementation with DoS protection
            boundary_bytes = f'--{boundary}'.encode()
            parts = body.split(boundary_bytes)

            # Enforce MAX_MULTIPART_PARTS to prevent DoS
            if len(parts) > MAX_MULTIPART_PARTS:
                self._send_json({"error": f"Too many multipart parts. Max: {MAX_MULTIPART_PARTS}"}, status=400)
                return

            file_content = None
            filename = None

            for part in parts:
                if b'Content-Disposition' not in part:
                    continue

                # Extract filename
                try:
                    header_end = part.index(b'\r\n\r\n')
                    headers_raw = part[:header_end].decode('utf-8', errors='ignore')
                    file_data = part[header_end + 4:]

                    # Remove trailing boundary markers
                    if file_data.endswith(b'--\r\n'):
                        file_data = file_data[:-4]
                    elif file_data.endswith(b'\r\n'):
                        file_data = file_data[:-2]

                    # Extract filename from headers with path traversal protection
                    if 'filename="' in headers_raw:
                        start = headers_raw.index('filename="') + 10
                        end = headers_raw.index('"', start)
                        raw_filename = headers_raw[start:end]
                        # Sanitize: extract basename only, reject path traversal attempts
                        import os
                        filename = os.path.basename(raw_filename)
                        # Reject null bytes, control chars, and suspicious patterns
                        if not filename or '\x00' in filename or '..' in filename:
                            continue
                        # Reject filenames that are just dots or whitespace
                        if filename.strip('.').strip() == '':
                            continue
                        file_content = file_data
                        break
                except (ValueError, IndexError):
                    continue

            if not file_content or not filename:
                self._send_json({"error": "No file found in upload"}, status=400)
                return

        else:
            # Raw file upload - get filename from header with path traversal protection
            import os
            raw_filename = self.headers.get('X-Filename', 'document.txt')
            filename = os.path.basename(raw_filename)
            # Reject null bytes, control chars, and suspicious patterns
            if not filename or '\x00' in filename or '..' in filename:
                self._send_json({"error": "Invalid filename"}, status=400)
                return
            file_content = self.rfile.read(content_length)

        # Validate file extension
        ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
        if ext not in SUPPORTED_EXTENSIONS:
            self._send_json({
                "error": f"Unsupported file type: {ext}",
                "supported": list(SUPPORTED_EXTENSIONS)
            }, status=400)
            return

        # Parse document
        try:
            doc = parse_document(file_content, filename)
            doc_id = self.document_store.add(doc)

            self._send_json({
                "success": True,
                "document": {
                    "id": doc_id,
                    "filename": doc.filename,
                    "word_count": doc.word_count,
                    "page_count": doc.page_count,
                    "preview": doc.preview,
                }
            })
        except ImportError as e:
            self._send_json({"error": _safe_error_message(e, "document_import")}, status=400)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "document_parsing")}, status=500)

    def _start_debate(self) -> None:
        """Start an ad-hoc debate with specified question.

        Accepts JSON body with:
            question: The topic/question to debate (required)
            agents: Comma-separated agent list (optional, default varies)
            rounds: Number of debate rounds (optional, default: 3)
            consensus: Consensus method (optional, default: "majority")
            auto_select: Whether to auto-select agents (optional, default: False)
            use_trending: Whether to use trending topic (optional, default: False)

        Rate limited: requires auth when enabled.
        """
        # Rate limit expensive debate creation
        if not self._check_rate_limit():
            return

        if not DEBATE_AVAILABLE:
            self._send_json({"error": "Debate orchestrator not available"}, status=500)
            return

        if not self.stream_emitter:
            self._send_json({"error": "Event streaming not configured"}, status=500)
            return

        # Parse JSON body with size validation
        try:
            content_length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            self._send_json({"error": "Invalid Content-Length header"}, status=400)
            return

        if content_length == 0:
            self._send_json({"error": "No content provided"}, status=400)
            return

        # Limit debate request size to 1MB
        if content_length > 1024 * 1024:
            self._send_json({"error": "Request too large"}, status=413)
            return

        try:
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, status=400)
            return

        # Parse and validate request using DebateRequest
        try:
            request = DebateRequest.from_dict(data)
        except ValueError as e:
            self._send_json({"error": str(e)}, status=400)
            return

        # Get or create debate controller and start debate
        controller = self._get_debate_controller()
        response = controller.start_debate(request)

        # Send response
        self._send_json(response.to_dict(), status=response.status_code)

    def _list_documents(self) -> None:
        """List all uploaded documents."""
        if not self.document_store:
            self._send_json({"documents": [], "error": "Document storage not configured"})
            return

        docs = self.document_store.list_all()
        self._send_json({"documents": docs, "count": len(docs)})

    # NOTE: Insights methods moved to handlers/insights.py (InsightsHandler)
    # NOTE: _run_capability_probe moved to handlers/probes.py (ProbesHandler)

    # =========================================================================
    # Deep Audit Helper Methods (extracted from _run_deep_audit for clarity)
    # =========================================================================

    def _parse_audit_config(self, config_data: dict) -> "DeepAuditConfig":
        """Parse audit configuration from request data.

        Returns:
            DeepAuditConfig instance
        """
        from aragora.modes.deep_audit import (
            DeepAuditConfig,
            STRATEGY_AUDIT,
            CONTRACT_AUDIT,
            CODE_ARCHITECTURE_AUDIT,
        )

        audit_type = config_data.get('audit_type', '')
        if audit_type == 'strategy':
            return STRATEGY_AUDIT
        elif audit_type == 'contract':
            return CONTRACT_AUDIT
        elif audit_type == 'code_architecture':
            return CODE_ARCHITECTURE_AUDIT
        else:
            return DeepAuditConfig(
                rounds=min(_safe_int(config_data.get('rounds', 6), 6), 10),
                enable_research=config_data.get('enable_research', True),
                cross_examination_depth=min(_safe_int(config_data.get('cross_examination_depth', 3), 3), 10),
                risk_threshold=_safe_float(config_data.get('risk_threshold', 0.7), 0.7),
            )

    def _create_audit_agents(self, agent_names: list, model_type: str) -> list:
        """Create agents for deep audit.

        Args:
            agent_names: List of agent names
            model_type: Agent model type

        Returns:
            List of created agents
        """
        if not agent_names:
            agent_names = ['Claude-Analyst', 'Claude-Skeptic', 'Claude-Synthesizer']

        agents = []
        for name in agent_names[:5]:  # Limit to 5 agents
            is_valid, _ = validate_id(name, "agent name")
            if not is_valid:
                continue
            try:
                agent = create_agent(model_type, name=name, role="proposer")
                agents.append(agent)
            except Exception as e:
                logger.debug(f"Failed to create audit agent {name}: {e}")
        return agents

    def _calculate_audit_elo(self, verdict, audit_id: str) -> dict:
        """Calculate ELO adjustments based on audit findings.

        Args:
            verdict: Audit verdict with findings
            audit_id: Audit session ID

        Returns:
            Dict of agent_name -> ELO adjustment
        """
        elo_adjustments = {}
        if not self.elo_system:
            return elo_adjustments

        # Agents who identified issues get ELO boost
        for finding in verdict.findings:
            for agent_name in finding.agents_agree:
                elo_adjustments[agent_name] = elo_adjustments.get(agent_name, 0) + 2
            for agent_name in finding.agents_disagree:
                elo_adjustments[agent_name] = elo_adjustments.get(agent_name, 0) - 1

        # Record adjustments
        for agent_name, adjustment in elo_adjustments.items():
            try:
                if adjustment > 0:
                    self.elo_system.record_redteam_result(
                        agent_name=agent_name,
                        robustness_score=1.0,
                        successful_attacks=0,
                        total_attacks=1,
                        critical_vulnerabilities=0,
                        session_id=audit_id
                    )
            except Exception as e:
                logger.warning(f"Failed to record audit ELO result for {agent_name}: {e}")

        return elo_adjustments

    def _save_audit_to_disk(
        self, audit_id: str, task: str, context: str, agents: list,
        verdict, config, duration_ms: float, elo_adjustments: dict
    ) -> None:
        """Save audit results to .nomic/audits/ directory.

        Args:
            audit_id: Unique audit ID
            task: Audit task
            context: Task context
            agents: List of agents
            verdict: Audit verdict
            config: Audit configuration
            duration_ms: Duration in milliseconds
            elo_adjustments: ELO adjustments dict
        """
        if not self.nomic_dir:
            return

        try:
            from datetime import datetime
            audits_dir = self.nomic_dir / "audits"
            audits_dir.mkdir(parents=True, exist_ok=True)
            date_str = datetime.now().strftime("%Y-%m-%d")
            audit_file = audits_dir / f"{date_str}_{audit_id}.json"
            audit_file.write_text(json.dumps({
                "audit_id": audit_id,
                "task": task,
                "context": context[:1000],
                "agents": [a.name for a in agents],
                "recommendation": verdict.recommendation,
                "confidence": verdict.confidence,
                "unanimous_issues": verdict.unanimous_issues,
                "split_opinions": verdict.split_opinions,
                "risk_areas": verdict.risk_areas,
                "findings": [
                    {
                        "category": f.category,
                        "summary": f.summary,
                        "details": f.details,
                        "agents_agree": f.agents_agree,
                        "agents_disagree": f.agents_disagree,
                        "confidence": f.confidence,
                        "severity": f.severity,
                        "citations": f.citations,
                    }
                    for f in verdict.findings
                ],
                "cross_examination_notes": verdict.cross_examination_notes,
                "citations": verdict.citations,
                "config": {
                    "rounds": config.rounds,
                    "enable_research": config.enable_research,
                    "cross_examination_depth": config.cross_examination_depth,
                    "risk_threshold": config.risk_threshold,
                },
                "duration_ms": duration_ms,
                "elo_adjustments": elo_adjustments,
                "created_at": datetime.now().isoformat(),
            }, indent=2, default=str))
        except Exception as e:
            logger.warning("Audit storage failed for %s (non-fatal): %s: %s", audit_id, type(e).__name__, e)

    def _build_audit_response(
        self, audit_id: str, task: str, agents: list, verdict, config,
        duration_ms: float, elo_adjustments: dict
    ) -> dict:
        """Build audit response dictionary.

        Args:
            audit_id: Unique audit ID
            task: Audit task
            agents: List of agents
            verdict: Audit verdict
            config: Audit configuration
            duration_ms: Duration in milliseconds
            elo_adjustments: ELO adjustments dict

        Returns:
            Response dictionary
        """
        return {
            "audit_id": audit_id,
            "task": task,
            "recommendation": verdict.recommendation,
            "confidence": verdict.confidence,
            "unanimous_issues": verdict.unanimous_issues,
            "split_opinions": verdict.split_opinions,
            "risk_areas": verdict.risk_areas,
            "findings": [
                {
                    "category": f.category,
                    "summary": f.summary,
                    "details": f.details[:500],
                    "agents_agree": f.agents_agree,
                    "agents_disagree": f.agents_disagree,
                    "confidence": f.confidence,
                    "severity": f.severity,
                }
                for f in verdict.findings
            ],
            "cross_examination_notes": verdict.cross_examination_notes[:2000],
            "citations": verdict.citations[:20],
            "rounds_completed": config.rounds,
            "duration_ms": round(duration_ms, 1),
            "agents": [a.name for a in agents],
            "elo_adjustments": elo_adjustments,
            "summary": {
                "unanimous_count": len(verdict.unanimous_issues),
                "split_count": len(verdict.split_opinions),
                "risk_count": len(verdict.risk_areas),
                "findings_count": len(verdict.findings),
                "high_severity_count": sum(1 for f in verdict.findings if f.severity >= 0.7),
            }
        }

    def _run_deep_audit(self) -> None:
        """Run a deep audit (Heavy3-inspired intensive multi-round debate protocol).

        Uses helper methods for config parsing, agent creation, ELO calculation,
        storage, and response building.
        """
        if not self._check_rate_limit():
            return

        try:
            from aragora.modes.deep_audit import DeepAuditOrchestrator
        except ImportError:
            self._send_json({
                "error": "Deep audit module not available",
                "hint": "aragora.modes.deep_audit failed to import"
            }, status=503)
            return

        if not DEBATE_AVAILABLE or create_agent is None:
            self._send_json({
                "error": "Agent system not available",
                "hint": "Debate module or create_agent failed to import"
            }, status=503)
            return

        content_length = self._validate_content_length()
        if content_length is None:
            return

        try:
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}

            task = data.get('task', '').strip()
            if not task:
                self._send_json({"error": "Missing required field: task"}, status=400)
                return

            context = data.get('context', '')
            config = self._parse_audit_config(data.get('config', {}))
            agents = self._create_audit_agents(
                data.get('agent_names', []),
                data.get('model_type', 'anthropic-api')
            )

            if len(agents) < 2:
                self._send_json({
                    "error": "Need at least 2 agents for deep audit",
                    "hint": f"Only created {len(agents)} agent(s)"
                }, status=400)
                return

            # Setup and emit start event
            audit_hooks = None
            if hasattr(self.server, 'stream_server') and self.server.stream_server:
                from .nomic_stream import create_nomic_hooks
                audit_hooks = create_nomic_hooks(self.server.stream_server.emitter)

            audit_id = f"audit-{uuid.uuid4().hex[:8]}"
            start_time = time.time()

            if audit_hooks and 'on_audit_start' in audit_hooks:
                audit_hooks['on_audit_start'](
                    audit_id=audit_id, task=task, agents=[a.name for a in agents],
                    config={
                        "rounds": config.rounds,
                        "enable_research": config.enable_research,
                        "cross_examination_depth": config.cross_examination_depth,
                        "risk_threshold": config.risk_threshold,
                    }
                )

            # Execute audit
            orchestrator = DeepAuditOrchestrator(agents, config)
            try:
                verdict = asyncio.run(orchestrator.run(task, context))
            except Exception as e:
                self._send_json({"error": f"Deep audit execution failed: {str(e)}"}, status=500)
                return

            duration_ms = (time.time() - start_time) * 1000
            elo_adjustments = self._calculate_audit_elo(verdict, audit_id)

            # Emit verdict event
            if audit_hooks and 'on_audit_verdict' in audit_hooks:
                audit_hooks['on_audit_verdict'](
                    audit_id=audit_id, task=task,
                    recommendation=verdict.recommendation[:2000],
                    confidence=verdict.confidence,
                    unanimous_issues=verdict.unanimous_issues[:10],
                    split_opinions=verdict.split_opinions[:10],
                    risk_areas=verdict.risk_areas[:10],
                    rounds_completed=config.rounds,
                    total_duration_ms=duration_ms,
                    agents=[a.name for a in agents],
                    elo_adjustments=elo_adjustments,
                )

            # Save and respond
            self._save_audit_to_disk(
                audit_id, task, context, agents, verdict, config, duration_ms, elo_adjustments
            )
            self._send_json(self._build_audit_response(
                audit_id, task, agents, verdict, config, duration_ms, elo_adjustments
            ))

        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=400)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "deep_audit")}, status=500)

    def _analyze_proposal_for_redteam(
        self, proposal: str, attack_types: list, debate_data: dict
    ) -> list:
        """Analyze a proposal for potential vulnerabilities.

        Returns findings with severity based on text analysis.
        """
        from aragora.modes.redteam import AttackType

        findings = []
        proposal_lower = proposal.lower() if proposal else ""

        # Keyword-based vulnerability detection
        vulnerability_patterns = {
            'logical_fallacy': {
                'keywords': ['always', 'never', 'all', 'none', 'obviously', 'clearly'],
                'description': 'Absolute language suggests potential logical fallacy',
                'base_severity': 0.4,
            },
            'edge_case': {
                'keywords': ['usually', 'most', 'typical', 'normal', 'standard'],
                'description': 'Generalization may miss edge cases',
                'base_severity': 0.5,
            },
            'unstated_assumption': {
                'keywords': ['should', 'must', 'need', 'require'],
                'description': 'Prescriptive language may hide unstated assumptions',
                'base_severity': 0.45,
            },
            'counterexample': {
                'keywords': ['best', 'optimal', 'superior', 'only'],
                'description': 'Strong claims may be vulnerable to counterexamples',
                'base_severity': 0.55,
            },
            'scalability': {
                'keywords': ['scale', 'growth', 'expand', 'distributed'],
                'description': 'Scalability claims require validation',
                'base_severity': 0.5,
            },
            'security': {
                'keywords': ['secure', 'safe', 'protected', 'auth', 'encrypt'],
                'description': 'Security claims need rigorous testing',
                'base_severity': 0.6,
            },
        }

        for attack_type in attack_types:
            try:
                AttackType(attack_type)
            except ValueError:
                continue

            pattern = vulnerability_patterns.get(attack_type, {})
            keywords = pattern.get('keywords', [])
            base_severity = pattern.get('base_severity', 0.5)

            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in proposal_lower)
            severity = min(0.9, base_severity + (matches * 0.1))

            if matches > 0:
                findings.append({
                    "attack_type": attack_type,
                    "description": pattern.get('description', f"Potential {attack_type} issue"),
                    "severity": round(severity, 2),
                    "exploitability": round(severity * 0.8, 2),
                    "keyword_matches": matches,
                    "requires_manual_review": severity > 0.6,
                })
            else:
                # Still include attack type but with lower severity
                findings.append({
                    "attack_type": attack_type,
                    "description": f"No obvious {attack_type.replace('_', ' ')} patterns detected",
                    "severity": round(base_severity * 0.5, 2),
                    "exploitability": round(base_severity * 0.3, 2),
                    "keyword_matches": 0,
                    "requires_manual_review": False,
                })

        return findings

    def _run_red_team_analysis(self, debate_id: str) -> None:
        """Run adversarial red-team analysis on a debate.

        POST body:
            attack_types: List of attack types (optional)
            max_rounds: Maximum attack/defend rounds (default: 3, max: 5)
            focus_proposal: Optional specific proposal to analyze

        Returns:
            session_id: Red team session ID
            findings: List of potential vulnerabilities
            robustness_score: 0-1 score
        """
        if not self._check_rate_limit():
            return

        if not REDTEAM_AVAILABLE:
            self._send_json({
                "error": "Red team mode not available",
                "hint": "RedTeam module failed to import"
            }, status=503)
            return

        content_length = self._validate_content_length()
        if content_length is None:
            return  # Error already sent

        try:
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}

            if not self.storage:
                self._send_json({"error": "Storage not configured"}, status=500)
                return

            debate_data = self.storage.get_by_slug(debate_id) or self.storage.get_by_id(debate_id)
            if not debate_data:
                self._send_json({"error": "Debate not found"}, status=404)
                return

            from aragora.modes.redteam import AttackType
            from datetime import datetime

            attack_type_names = data.get('attack_types', [
                'logical_fallacy', 'edge_case', 'unstated_assumption',
                'counterexample', 'scalability', 'security'
            ])
            max_rounds = min(_safe_int(data.get('max_rounds', 3), 3), 5)

            focus_proposal = data.get('focus_proposal') or (
                debate_data.get('consensus_answer') or
                debate_data.get('final_answer') or
                debate_data.get('task', '')
            )

            session_id = f"redteam-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"

            # Analyze proposal for potential weaknesses
            findings = self._analyze_proposal_for_redteam(
                focus_proposal, attack_type_names, debate_data
            )

            # Calculate robustness based on finding severity
            avg_severity = sum(f.get('severity', 0.5) for f in findings) / max(len(findings), 1)
            robustness_score = max(0.0, 1.0 - avg_severity)

            self._send_json({
                "session_id": session_id,
                "debate_id": debate_id,
                "target_proposal": focus_proposal[:500] if focus_proposal else "",
                "attack_types": attack_type_names,
                "max_rounds": max_rounds,
                "findings": findings,
                "robustness_score": round(robustness_score, 2),
                "status": "analysis_complete",
                "created_at": datetime.now().isoformat(),
            })

        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=400)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "red_team_analysis")}, status=500)

    def _verify_debate_outcome(self, debate_id: str) -> None:
        """Record verification of whether a debate's winning position was correct.

        POST body:
            correct: Boolean - whether the winning position was actually correct
            source: String - verification source (default: "manual")

        Completes the truth-grounding feedback loop by linking positions to outcomes.
        """
        if not self._check_rate_limit():
            return

        content_length = self._validate_content_length()
        if content_length is None:
            return  # Error already sent

        try:
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}

            correct = data.get('correct', False)
            source = data.get('source', 'manual')

            # Use position_tracker if available
            if hasattr(self, 'position_tracker') and self.position_tracker:
                self.position_tracker.record_verification(debate_id, correct, source)
                self._send_json({
                    "status": "verified",
                    "debate_id": debate_id,
                    "correct": correct,
                    "source": source,
                })
            else:
                # Try to create a temporary tracker
                try:
                    from aragora.agents.truth_grounding import PositionTracker
                    db_path = self.nomic_dir / "aragora_positions.db" if self.nomic_dir else None
                    if db_path and db_path.exists():
                        tracker = PositionTracker(db_path=str(db_path))
                        tracker.record_verification(debate_id, correct, source)
                        self._send_json({
                            "status": "verified",
                            "debate_id": debate_id,
                            "correct": correct,
                            "source": source,
                        })
                    else:
                        self._send_json({"error": "Position tracking not configured"}, status=503)
                except ImportError:
                    self._send_json({"error": "PositionTracker module not available"}, status=503)

        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=400)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "verify_debate")}, status=500)

    # Tournament methods moved to TournamentHandler
    # Best team combinations moved to RoutingHandler
    # Evolution history moved to EvolutionHandler

    def _serve_audio(self, debate_id: str) -> None:
        """Serve audio file for a debate with security checks.

        GET /audio/{debate_id}.mp3
        """
        if not self.audio_store:
            self.send_error(404, "Audio storage not configured")
            return

        # Validate debate_id format (prevent path traversal)
        if not debate_id or '..' in debate_id or '/' in debate_id or '\\' in debate_id:
            self.send_error(400, "Invalid debate ID")
            return

        # Get audio file path
        audio_path = self.audio_store.get_path(debate_id)
        if not audio_path or not audio_path.exists():
            self.send_error(404, "Audio not found")
            return

        # Security: Ensure file is within audio storage directory
        try:
            audio_path_resolved = audio_path.resolve()
            storage_dir_resolved = self.audio_store.storage_dir.resolve()
            if not str(audio_path_resolved).startswith(str(storage_dir_resolved)):
                logger.warning(f"Audio path traversal attempt: {debate_id}")
                self.send_error(403, "Access denied")
                return
        except (ValueError, OSError):
            self.send_error(400, "Invalid path")
            return

        try:
            content = audio_path.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'audio/mpeg')
            self.send_header('Content-Length', len(content))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Cache-Control', 'public, max-age=86400')  # Cache for 1 day
            self._add_cors_headers()
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            logger.error(f"Failed to serve audio {debate_id}: {e}")
            self.send_error(500, "Failed to read audio file")

    # NOTE: Podcast and social publishing methods moved to handlers/broadcast.py (BroadcastHandler)
    # _get_podcast_feed, _get_podcast_episodes, _publish_to_twitter
    # _get_youtube_auth_url, _handle_youtube_callback, _get_youtube_status, _publish_to_youtube

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
        content_type = 'text/html'
        if filename.endswith('.css'):
            content_type = 'text/css'
        elif filename.endswith('.js'):
            content_type = 'application/javascript'
        elif filename.endswith('.json'):
            content_type = 'application/json'
        elif filename.endswith('.ico'):
            content_type = 'image/x-icon'
        elif filename.endswith('.svg'):
            content_type = 'image/svg+xml'
        elif filename.endswith('.png'):
            content_type = 'image/png'

        try:
            content = filepath.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', len(content))
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
        content = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(content))
        self._add_cors_headers()
        self._add_security_headers()
        self.end_headers()
        self.wfile.write(content)

    def _add_security_headers(self) -> None:
        """Add security headers to prevent common attacks."""
        # Prevent clickjacking
        self.send_header('X-Frame-Options', 'DENY')
        # Prevent MIME type sniffing
        self.send_header('X-Content-Type-Options', 'nosniff')
        # Enable XSS filter
        self.send_header('X-XSS-Protection', '1; mode=block')
        # Referrer policy - don't leak internal URLs
        self.send_header('Referrer-Policy', 'strict-origin-when-cross-origin')
        # Content Security Policy - prevent XSS and data injection
        # Note: 'unsafe-inline' for styles needed by CSS-in-JS frameworks
        # 'unsafe-eval' removed for security - blocks eval()/new Function()
        self.send_header('Content-Security-Policy',
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "connect-src 'self' wss: https:; "
            "font-src 'self' data:; "
            "frame-ancestors 'none'")
        # HTTP Strict Transport Security - enforce HTTPS
        self.send_header('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')

    def _add_cors_headers(self) -> None:
        """Add CORS headers with origin validation."""
        # Security: Validate origin against centralized allowlist
        request_origin = self.headers.get('Origin', '')

        if cors_config.is_origin_allowed(request_origin):
            self.send_header('Access-Control-Allow-Origin', request_origin)
        elif not request_origin:
            # Same-origin requests don't have Origin header
            pass
        # else: no CORS header = browser blocks cross-origin request

        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Filename, Authorization')

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

        # Setup HTTP handler
        UnifiedHandler.storage = storage
        UnifiedHandler.static_dir = static_dir
        UnifiedHandler.stream_emitter = self.stream_server.emitter
        UnifiedHandler.persistence = self.persistence
        if nomic_dir:
            UnifiedHandler.nomic_state_file = nomic_dir / "nomic_state.json"
            # Initialize InsightStore from nomic directory
            UnifiedHandler.insight_store = init_insight_store(nomic_dir)
            # Initialize EloSystem from nomic directory
            UnifiedHandler.elo_system = init_elo_system(nomic_dir)

            # Initialize FlipDetector from nomic directory
            UnifiedHandler.flip_detector = init_flip_detector(nomic_dir)

            # Initialize DocumentStore for file uploads
            doc_dir = nomic_dir / "documents"
            UnifiedHandler.document_store = DocumentStore(doc_dir)
            logger.info(f"[server] DocumentStore initialized at {doc_dir}")

            # Initialize AudioFileStore for broadcast audio
            audio_dir = nomic_dir / "audio"
            UnifiedHandler.audio_store = AudioFileStore(audio_dir)
            logger.info(f"[server] AudioFileStore initialized at {audio_dir}")

            # Initialize Twitter connector for social posting
            UnifiedHandler.twitter_connector = TwitterPosterConnector()
            if UnifiedHandler.twitter_connector.is_configured:
                logger.info("[server] TwitterPosterConnector initialized")
            else:
                logger.info("[server] TwitterPosterConnector created (credentials not configured)")

            # Initialize YouTube connector for video uploads
            UnifiedHandler.youtube_connector = YouTubeUploaderConnector()
            if UnifiedHandler.youtube_connector.is_configured:
                logger.info("[server] YouTubeUploaderConnector initialized")
            else:
                logger.info("[server] YouTubeUploaderConnector created (credentials not configured)")

            # Initialize video generator for YouTube
            video_dir = nomic_dir / "videos"
            UnifiedHandler.video_generator = VideoGenerator(video_dir)
            logger.info(f"[server] VideoGenerator initialized at {video_dir}")

            # Initialize PersonaManager for agent specialization
            UnifiedHandler.persona_manager = init_persona_manager(nomic_dir)

            # Initialize PositionLedger for truth-grounded personas
            UnifiedHandler.position_ledger = init_position_ledger(nomic_dir)

            # Initialize DebateEmbeddingsDatabase for historical memory
            UnifiedHandler.debate_embeddings = init_debate_embeddings(nomic_dir)

            # Initialize ConsensusMemory and DissentRetriever for historical minority views
            if CONSENSUS_MEMORY_AVAILABLE and DissentRetriever is not None:
                try:
                    UnifiedHandler.consensus_memory = ConsensusMemory()
                    UnifiedHandler.dissent_retriever = DissentRetriever(UnifiedHandler.consensus_memory)
                    logger.info("[server] DissentRetriever loaded for historical minority views")
                except Exception as e:
                    logger.warning(f"[server] DissentRetriever initialization failed: {e}")

            # Initialize MomentDetector for significant agent moments (narrative storytelling)
            if MOMENT_DETECTOR_AVAILABLE and MomentDetector is not None:
                try:
                    UnifiedHandler.moment_detector = MomentDetector(
                        elo_system=UnifiedHandler.elo_system,
                        position_ledger=UnifiedHandler.position_ledger,
                    )
                    logger.info("[server] MomentDetector loaded for agent moments API")
                except Exception as e:
                    logger.warning(f"[server] MomentDetector initialization failed: {e}")

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
                    ssl_context.set_ciphers('ECDHE+AESGCM:DHE+AESGCM:ECDHE+CHACHA20:DHE+CHACHA20')
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
            except Exception as e:
                logger.error(f"HTTP server unexpected error: {e}")
                break

    async def start(self) -> None:
        """Start both HTTP and WebSocket servers."""
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

        # Start HTTP server in background thread
        http_thread = Thread(target=self._run_http_server, daemon=True)
        http_thread.start()

        # Start WebSocket server in foreground
        await self.stream_server.start()


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
    from aragora.config import SSL_ENABLED, SSL_CERT_PATH, SSL_KEY_PATH

    if ssl_cert is None and SSL_ENABLED:
        ssl_cert = SSL_CERT_PATH
        ssl_key = SSL_KEY_PATH

    server = UnifiedServer(
        http_port=http_port,
        ws_port=ws_port,
        static_dir=static_dir,
        nomic_dir=nomic_dir,
        ssl_cert=ssl_cert,
        ssl_key=ssl_key,
    )
    await server.start()
