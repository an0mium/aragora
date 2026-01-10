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
from typing import TYPE_CHECKING, Any, Coroutine, Optional, Dict

if TYPE_CHECKING:
    from aragora.persistence.supabase import SupabaseClient
    from aragora.insights.store import InsightStore
    from aragora.ranking.elo import EloSystem
    from aragora.insights.flip_detector import FlipDetector
    from aragora.agents.personas import PersonaManager
    from aragora.debate.embeddings import DebateEmbeddingsDatabase
    from aragora.agents.truth_grounding import PositionTracker  # type: ignore[attr-defined]
    from aragora.agents.grounded import PositionLedger  # type: ignore[attr-defined]
    from aragora.memory.consensus import ConsensusMemory, DissentRetriever  # type: ignore[attr-defined]
    from aragora.agents.grounded import MomentDetector  # type: ignore[attr-defined]
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
from aragora.server.validation import (
    SAFE_ID_PATTERN,
    validate_id,
    safe_query_int,
    safe_query_float,
)
from aragora.server.handlers.base import invalidate_leaderboard_cache

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


class UnifiedHandler(HandlerRegistryMixin, BaseHTTPRequestHandler):  # type: ignore[misc]
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
    user_store: Optional["UserStore"] = None  # UserStore for user/org persistence

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

    def _get_client_ip(self) -> str:
        """Get client IP address, respecting trusted proxy headers."""
        remote_ip = self.client_address[0] if hasattr(self, 'client_address') else 'unknown'
        client_ip = remote_ip
        if remote_ip in TRUSTED_PROXIES:
            forwarded = self.headers.get('X-Forwarded-For', '')
            if forwarded:
                first_ip = forwarded.split(',')[0].strip()
                if first_ip:
                    client_ip = first_ip
        return client_ip

    def _safe_int(self, query: dict, key: str, default: int, max_val: int = 100) -> int:
        """Safely parse integer query param with bounds checking.

        Delegates to shared safe_query_int from validation module.
        """
        return safe_query_int(query, key, default, min_val=1, max_val=max_val)

    def _safe_float(self, query: dict, key: str, default: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Safely parse float query param with bounds checking.

        Delegates to shared safe_query_float from validation module.
        """
        return safe_query_float(query, key, default, min_val=min_val, max_val=max_val)

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

        # NOTE: /api/documents/upload is now handled by DocumentHandler
        if path == '/api/debate':
            self._start_debate()
        # NOTE: Broadcast, publishing, laboratory, routing, verification, probes,
        # plugins, insights routes are NOW HANDLED BY modular handlers (BroadcastHandler,
        # LaboratoryHandler, RoutingHandler, VerificationHandler, ProbesHandler,
        # PluginsHandler, AuditingHandler, InsightsHandler)
        # NOTE: /api/debates/{id}/verify is now handled by DebatesHandler
        elif path == '/api/auth/revoke':
            self._revoke_token()
        else:
            self.send_error(404, f"Unknown POST endpoint: {path}")

    def do_DELETE(self) -> None:
        """Handle DELETE requests."""
        start_time = time.time()
        status_code = 200  # Default, updated by handlers
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            self._do_DELETE_internal(path)
        except Exception as e:
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
        if path.startswith('/api/'):
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
        if path.startswith('/api/'):
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
        if path.startswith('/api/'):
            if self._try_modular_handler(path, {}):
                return

        self.send_error(404, f"Unknown PUT endpoint: {path}")

    # NOTE: _upload_document moved to handlers/documents.py (DocumentHandler)

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

        # Quota enforcement - check org usage limits
        if UnifiedHandler.user_store:
            from aragora.billing.jwt_auth import extract_user_from_request
            auth_ctx = extract_user_from_request(self, UnifiedHandler.user_store)
            if auth_ctx.is_authenticated and auth_ctx.org_id:
                org = UnifiedHandler.user_store.get_organization_by_id(auth_ctx.org_id)
                if org and org.is_at_limit:
                    self._send_json({
                        "error": "Monthly debate quota exceeded",
                        "code": "quota_exceeded",
                        "limit": org.limits.debates_per_month,
                        "used": org.debates_used_this_month,
                        "upgrade_url": "/pricing",
                    }, status=429)
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

        # Increment usage on successful debate creation
        if response.status_code < 400 and UnifiedHandler.user_store:
            from aragora.billing.jwt_auth import extract_user_from_request
            auth_ctx = extract_user_from_request(self, UnifiedHandler.user_store)
            if auth_ctx.is_authenticated and auth_ctx.org_id:
                UnifiedHandler.user_store.increment_usage(auth_ctx.org_id)
                logger.info(f"Incremented debate usage for org {auth_ctx.org_id}")

        # Send response
        self._send_json(response.to_dict(), status=response.status_code)

    # NOTE: _list_documents moved to handlers/documents.py (DocumentHandler)
    # NOTE: Insights methods moved to handlers/insights.py (InsightsHandler)
    # NOTE: _run_capability_probe moved to handlers/probes.py (ProbesHandler)
    # NOTE: Deep audit methods moved to handlers/auditing.py (AuditingHandler)
    # NOTE: Red team methods moved to handlers/auditing.py (AuditingHandler)
    # NOTE: _verify_debate_outcome moved to handlers/debates.py (DebatesHandler)
    # NOTE: Tournament methods moved to TournamentHandler
    # NOTE: Best team combinations moved to RoutingHandler
    # NOTE: Evolution history moved to EvolutionHandler
    # NOTE: _serve_audio moved to handlers/audio.py (AudioHandler)
    # NOTE: Podcast and social publishing methods moved to handlers/broadcast.py (BroadcastHandler)

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
            self.send_header('Content-Length', str(len(content)))
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
        self.send_header('Content-Length', str(len(content)))
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
            UnifiedHandler.consensus_memory, UnifiedHandler.dissent_retriever = init_consensus_memory()

            # Initialize MomentDetector for significant agent moments (narrative storytelling)
            UnifiedHandler.moment_detector = init_moment_detector(
                elo_system=UnifiedHandler.elo_system,
                position_ledger=UnifiedHandler.position_ledger,
            )

            # Initialize UserStore for user/organization persistence
            from aragora.storage import UserStore
            user_db_path = nomic_dir / "users.db"
            UnifiedHandler.user_store = UserStore(user_db_path)
            logger.info(f"[server] UserStore initialized at {user_db_path}")

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
        # Initialize error monitoring (no-op if SENTRY_DSN not set)
        try:
            from aragora.server.error_monitoring import init_monitoring
            if init_monitoring():
                logger.info("Error monitoring enabled (Sentry)")
        except ImportError:
            pass  # sentry-sdk not installed

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

    # Initialize storage from nomic directory
    storage = None
    if nomic_dir:
        db_path = nomic_dir / "debates.db"
        try:
            storage = DebateStorage(str(db_path))
            logger.info(f"[server] DebateStorage initialized at {db_path}")
        except Exception as e:
            logger.warning(f"[server] Failed to initialize DebateStorage: {e}")

    # Ensure demo data is loaded for search functionality
    try:
        from aragora.fixtures import ensure_demo_data
        logger.info("[server] Checking demo data initialization...")
        ensure_demo_data()
    except Exception as e:
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
