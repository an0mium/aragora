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
from aragora.config import DB_INSIGHTS_PATH, DB_PERSONAS_PATH, DB_TIMEOUT_SECONDS, MAX_AGENTS_PER_DEBATE, MAX_CONCURRENT_DEBATES
from aragora.server.error_utils import safe_error_message as _safe_error_message
from aragora.server.validation import SAFE_ID_PATTERN

# Valid agent types (allowlist for security)
ALLOWED_AGENT_TYPES = frozenset({
    # CLI-based
    "codex", "claude", "openai", "gemini-cli", "grok-cli", "qwen-cli", "deepseek-cli", "kilocode",
    # API-based (direct)
    "gemini", "ollama", "anthropic-api", "openai-api", "grok",
    # API-based (via OpenRouter)
    "deepseek", "deepseek-r1", "llama", "mistral", "openrouter",
})

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

# Query parameter whitelist (security: reject unknown params to prevent injection)
# Maps param name -> allowed values (None means any string allowed, set means restricted)
ALLOWED_QUERY_PARAMS = {
    # Pagination
    "limit": None,
    "offset": None,
    # Filtering
    "domain": None,
    "loop_id": None,
    "topic": None,
    "query": None,
    # Export
    "table": {"summary", "debates", "proposals", "votes", "critiques", "messages"},
    # Agent queries
    "agent": None,
    "agent_a": None,
    "agent_b": None,
    "sections": {"identity", "performance", "relationships", "all"},
    # Calibration
    "buckets": None,
    # Memory
    "tiers": None,
    "min_importance": None,
    # Genesis
    "event_type": {"mutation", "crossover", "selection", "extinction", "speciation"},
    # Logs
    "lines": None,
}


def _validate_query_params(query: dict) -> tuple[bool, str]:
    """Validate query parameters against whitelist.

    Returns (is_valid, error_message).
    """
    for param, values in query.items():
        if param not in ALLOWED_QUERY_PARAMS:
            return False, f"Unknown query parameter: {param}"

        allowed = ALLOWED_QUERY_PARAMS[param]
        if allowed is not None:
            # Check if value is in the allowed set
            for val in values:
                if val not in allowed:
                    return False, f"Invalid value for {param}: {val}"

    return True, ""


# Optional imports using utility for consistent handling
from aragora.utils.optional_imports import try_import

# Optional Supabase persistence
_imp, PERSISTENCE_AVAILABLE = try_import("aragora.persistence", "SupabaseClient")
SupabaseClient = _imp["SupabaseClient"]

# Optional InsightStore for debate insights
_imp, INSIGHTS_AVAILABLE = try_import("aragora.insights.store", "InsightStore")
InsightStore = _imp["InsightStore"]

# Optional EloSystem for agent rankings
_imp, RANKING_AVAILABLE = try_import("aragora.ranking.elo", "EloSystem")
EloSystem = _imp["EloSystem"]

# Optional FlipDetector for position reversal detection
_imp, FLIP_DETECTOR_AVAILABLE = try_import(
    "aragora.insights.flip_detector",
    "FlipDetector", "format_flip_for_ui", "format_consistency_for_ui"
)
FlipDetector = _imp["FlipDetector"]
format_flip_for_ui = _imp.get("format_flip_for_ui")
format_consistency_for_ui = _imp.get("format_consistency_for_ui")

# Optional debate orchestrator for ad-hoc debates
_imp1, _avail1 = try_import("aragora.debate.orchestrator", "Arena", "DebateProtocol")
_imp2, _avail2 = try_import("aragora.agents.base", "create_agent")
_imp3, _avail3 = try_import("aragora.core", "Environment")
DEBATE_AVAILABLE = _avail1 and _avail2 and _avail3
Arena = _imp1["Arena"]
DebateProtocol = _imp1["DebateProtocol"]
create_agent = _imp2["create_agent"]
Environment = _imp3["Environment"]

# Optional PersonaManager for agent specialization
_imp, PERSONAS_AVAILABLE = try_import("aragora.agents.personas", "PersonaManager")
PersonaManager = _imp["PersonaManager"]

# Optional DebateEmbeddingsDatabase for historical memory
_imp, EMBEDDINGS_AVAILABLE = try_import("aragora.debate.embeddings", "DebateEmbeddingsDatabase")
DebateEmbeddingsDatabase = _imp["DebateEmbeddingsDatabase"]

# Optional ConsensusMemory for historical consensus data
_imp, CONSENSUS_MEMORY_AVAILABLE = try_import(
    "aragora.memory.consensus", "ConsensusMemory", "DissentRetriever"
)
ConsensusMemory = _imp["ConsensusMemory"]
DissentRetriever = _imp["DissentRetriever"]

# Optional CalibrationTracker for agent calibration
_imp, CALIBRATION_AVAILABLE = try_import("aragora.agents.calibration", "CalibrationTracker")
CalibrationTracker = _imp["CalibrationTracker"]

# Optional PulseManager for trending topics
_imp, PULSE_AVAILABLE = try_import(
    "aragora.pulse.ingestor", "PulseManager", "TrendingTopic", "TwitterIngestor"
)
PulseManager = _imp["PulseManager"]
TrendingTopic = _imp["TrendingTopic"]

# Optional FormalVerificationManager for theorem proving
_imp, FORMAL_VERIFICATION_AVAILABLE = try_import(
    "aragora.verification.formal",
    "FormalVerificationManager", "get_formal_verification_manager"
)
FormalVerificationManager = _imp["FormalVerificationManager"]
get_formal_verification_manager = _imp["get_formal_verification_manager"]

# Optional Broadcast module for podcast generation
_imp1, _avail1 = try_import("aragora.broadcast", "broadcast_debate")
_imp2, _avail2 = try_import("aragora.debate.traces", "DebateTrace")
BROADCAST_AVAILABLE = _avail1 and _avail2
broadcast_debate = _imp1["broadcast_debate"]
DebateTrace = _imp2["DebateTrace"]

# Optional RelationshipTracker for agent network analysis
_imp, RELATIONSHIP_TRACKER_AVAILABLE = try_import("aragora.agents.grounded", "RelationshipTracker")
RelationshipTracker = _imp["RelationshipTracker"]

# Optional PositionLedger for truth-grounded personas
_imp, POSITION_LEDGER_AVAILABLE = try_import("aragora.agents.grounded", "PositionLedger")
PositionLedger = _imp["PositionLedger"]

# Optional CritiqueStore for pattern retrieval
_imp, CRITIQUE_STORE_AVAILABLE = try_import("aragora.memory.store", "CritiqueStore")
CritiqueStore = _imp["CritiqueStore"]

# Optional export module for debate artifact export
_imp, EXPORT_AVAILABLE = try_import(
    "aragora.export", "DebateArtifact", "CSVExporter", "DOTExporter", "StaticHTMLExporter"
)
DebateArtifact = _imp["DebateArtifact"]
CSVExporter = _imp["CSVExporter"]
DOTExporter = _imp["DOTExporter"]
StaticHTMLExporter = _imp["StaticHTMLExporter"]

# Optional CapabilityProber for vulnerability detection
_imp, PROBER_AVAILABLE = try_import("aragora.modes.prober", "CapabilityProber")
CapabilityProber = _imp["CapabilityProber"]

# Optional RedTeamMode for adversarial testing
_imp, REDTEAM_AVAILABLE = try_import("aragora.modes.redteam", "RedTeamMode")
RedTeamMode = _imp["RedTeamMode"]

# Optional PersonaLaboratory for emergent traits
_imp, LABORATORY_AVAILABLE = try_import("aragora.agents.laboratory", "PersonaLaboratory")
PersonaLaboratory = _imp["PersonaLaboratory"]

# Optional BeliefNetwork for debate cruxes
_imp, BELIEF_NETWORK_AVAILABLE = try_import(
    "aragora.reasoning.belief", "BeliefNetwork", "BeliefPropagationAnalyzer"
)
BeliefNetwork = _imp["BeliefNetwork"]
BeliefPropagationAnalyzer = _imp["BeliefPropagationAnalyzer"]

# Optional ProvenanceTracker for claim support
_imp, PROVENANCE_AVAILABLE = try_import("aragora.reasoning.provenance", "ProvenanceTracker")
ProvenanceTracker = _imp["ProvenanceTracker"]

# Optional MomentDetector for significant agent moments
_imp, MOMENT_DETECTOR_AVAILABLE = try_import("aragora.agents.grounded", "MomentDetector")
MomentDetector = _imp["MomentDetector"]

# Optional ImpasseDetector for debate deadlock detection
_imp, IMPASSE_DETECTOR_AVAILABLE = try_import("aragora.debate.counterfactual", "ImpasseDetector")
ImpasseDetector = _imp["ImpasseDetector"]

# Optional ConvergenceDetector for semantic position convergence
_imp, CONVERGENCE_DETECTOR_AVAILABLE = try_import("aragora.debate.convergence", "ConvergenceDetector")
ConvergenceDetector = _imp["ConvergenceDetector"]

# Optional AgentSelector for routing recommendations and auto team selection
_imp, ROUTING_AVAILABLE = try_import(
    "aragora.routing.selection", "AgentSelector", "AgentProfile", "TaskRequirements"
)
AgentSelector = _imp["AgentSelector"]
AgentProfile = _imp["AgentProfile"]
TaskRequirements = _imp["TaskRequirements"]

# Optional TournamentManager for tournament standings
_imp, TOURNAMENT_AVAILABLE = try_import("aragora.tournaments.tournament", "TournamentManager")
TournamentManager = _imp["TournamentManager"]

# Optional PromptEvolver for evolution history
_imp, EVOLUTION_AVAILABLE = try_import("aragora.evolution.evolver", "PromptEvolver")
PromptEvolver = _imp["PromptEvolver"]

# Optional ContinuumMemory for multi-timescale memory
_imp, CONTINUUM_AVAILABLE = try_import("aragora.memory.continuum", "ContinuumMemory", "MemoryTier")
ContinuumMemory = _imp["ContinuumMemory"]
MemoryTier = _imp["MemoryTier"]

# Optional InsightExtractor for debate insights
_imp, INSIGHT_EXTRACTOR_AVAILABLE = try_import("aragora.insights.extractor", "InsightExtractor")
InsightExtractor = _imp["InsightExtractor"]

# Modular HTTP handlers for endpoint routing
try:
    from aragora.server.handlers import (
        SystemHandler,
        DebatesHandler,
        AgentsHandler,
        PulseHandler,
        AnalyticsHandler,
        MetricsHandler,
        ConsensusHandler,
        BeliefHandler,
        CritiqueHandler,
        GenesisHandler,
        ReplaysHandler,
        TournamentHandler,
        MemoryHandler,
        LeaderboardViewHandler,
        DocumentHandler,
        VerificationHandler,
        AuditingHandler,
        RelationshipHandler,
        MomentsHandler,
        PersonaHandler,
        DashboardHandler,
        IntrospectionHandler,
        CalibrationHandler,
        RoutingHandler,
        EvolutionHandler,
        PluginsHandler,
        BroadcastHandler,
        LaboratoryHandler,
        ProbesHandler,
        HandlerResult,
    )
    HANDLERS_AVAILABLE = True
except ImportError:
    HANDLERS_AVAILABLE = False
    MetricsHandler = None
    SystemHandler = None
    DebatesHandler = None
    AgentsHandler = None
    PulseHandler = None
    AnalyticsHandler = None
    ConsensusHandler = None
    BeliefHandler = None
    CritiqueHandler = None
    GenesisHandler = None
    ReplaysHandler = None
    TournamentHandler = None
    MemoryHandler = None
    LeaderboardViewHandler = None
    DocumentHandler = None
    VerificationHandler = None
    AuditingHandler = None
    RelationshipHandler = None
    MomentsHandler = None
    PersonaHandler = None
    DashboardHandler = None
    IntrospectionHandler = None
    CalibrationHandler = None
    RoutingHandler = None
    EvolutionHandler = None
    PluginsHandler = None
    BroadcastHandler = None
    LaboratoryHandler = None
    ProbesHandler = None
    HandlerResult = None

# Track active ad-hoc debates
_active_debates: dict[str, dict] = {}
_active_debates_lock = threading.Lock()  # Thread-safe access to _active_debates
_debate_cleanup_counter = 0  # Counter for periodic cleanup

# TTL for completed debates (24 hours)
_DEBATE_TTL_SECONDS = 86400


def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert value to float, returning default on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value, default: int = 0) -> int:
    """Safely convert value to int, returning default on failure."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _update_debate_status(debate_id: str, status: str, **kwargs) -> None:
    """Atomic debate status update with consistent locking."""
    with _active_debates_lock:
        if debate_id in _active_debates:
            _active_debates[debate_id]["status"] = status
            # Record completion time for TTL cleanup
            if status in ("completed", "error"):
                _active_debates[debate_id]["completed_at"] = time.time()
            for key, value in kwargs.items():
                _active_debates[debate_id][key] = value


def _cleanup_stale_debates() -> None:
    """Remove completed/errored debates older than TTL."""
    now = time.time()
    with _active_debates_lock:
        stale_ids = [
            debate_id for debate_id, debate in _active_debates.items()
            if debate.get("status") in ("completed", "error")
            and now - debate.get("completed_at", now) > _DEBATE_TTL_SECONDS
        ]
        for debate_id in stale_ids:
            _active_debates.pop(debate_id, None)
    if stale_ids:
        logger.debug(f"Cleaned up {len(stale_ids)} stale debate entries")


# Server startup time for uptime tracking
_server_start_time: float = time.time()


def _wrap_agent_for_streaming(agent: Any, emitter: SyncEventEmitter, debate_id: str) -> Any:
    """Wrap an agent to emit token streaming events.

    If the agent has a generate_stream() method, we override its generate()
    to call generate_stream() and emit TOKEN_* events.

    Args:
        agent: Agent instance (duck-typed, must have generate method)
        emitter: Event emitter for streaming events
        debate_id: ID of the current debate

    Returns:
        The agent with wrapped generate method (or unchanged if no streaming support)
    """
    from datetime import datetime

    # Check if agent supports streaming
    if not hasattr(agent, 'generate_stream'):
        return agent

    # Store original generate method
    original_generate = agent.generate

    async def streaming_generate(prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Streaming wrapper that emits TOKEN_* events."""
        # Emit start event
        emitter.emit(StreamEvent(
            type=StreamEventType.TOKEN_START,
            data={
                "debate_id": debate_id,
                "agent": agent.name,
                "timestamp": datetime.now().isoformat(),
            },
            agent=agent.name,
        ))

        full_response = ""
        try:
            # Stream tokens from the agent
            async for token in agent.generate_stream(prompt, context):
                full_response += token
                # Emit token delta event
                emitter.emit(StreamEvent(
                    type=StreamEventType.TOKEN_DELTA,
                    data={
                        "debate_id": debate_id,
                        "agent": agent.name,
                        "token": token,
                    },
                    agent=agent.name,
                ))

            # Emit end event
            emitter.emit(StreamEvent(
                type=StreamEventType.TOKEN_END,
                data={
                    "debate_id": debate_id,
                    "agent": agent.name,
                    "full_response": full_response,
                },
                agent=agent.name,
            ))

            return full_response

        except Exception as e:
            # Emit error as end event
            emitter.emit(StreamEvent(
                type=StreamEventType.TOKEN_END,
                data={
                    "debate_id": debate_id,
                    "agent": agent.name,
                    "error": _safe_error_message(e, f"token streaming for {agent.name}"),
                    "full_response": full_response,
                },
                agent=agent.name,
            ))
            # Fall back to non-streaming
            if full_response:
                return full_response
            return await original_generate(prompt, context)

    # Replace the generate method
    agent.generate = streaming_generate
    return agent


def _run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run async coroutine in HTTP handler thread (which may not have an event loop)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Can't use run_until_complete if loop is running
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=30)
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop in this thread, create one
        return asyncio.run(coro)


class UnifiedHandler(BaseHTTPRequestHandler):
    """HTTP handler with API endpoints and static file serving."""

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

    # Modular HTTP handlers (initialized lazily)
    _system_handler: Optional["SystemHandler"] = None
    _debates_handler: Optional["DebatesHandler"] = None
    _agents_handler: Optional["AgentsHandler"] = None
    _pulse_handler: Optional["PulseHandler"] = None
    _analytics_handler: Optional["AnalyticsHandler"] = None
    _metrics_handler: Optional["MetricsHandler"] = None
    _consensus_handler: Optional["ConsensusHandler"] = None
    _belief_handler: Optional["BeliefHandler"] = None
    _critique_handler: Optional["CritiqueHandler"] = None
    _genesis_handler: Optional["GenesisHandler"] = None
    _replays_handler: Optional["ReplaysHandler"] = None
    _tournament_handler: Optional["TournamentHandler"] = None
    _memory_handler: Optional["MemoryHandler"] = None
    _leaderboard_handler: Optional["LeaderboardViewHandler"] = None
    _document_handler: Optional["DocumentHandler"] = None
    _verification_handler: Optional["VerificationHandler"] = None
    _auditing_handler: Optional["AuditingHandler"] = None
    _relationship_handler: Optional["RelationshipHandler"] = None
    _moments_handler: Optional["MomentsHandler"] = None
    _persona_handler: Optional["PersonaHandler"] = None
    _dashboard_handler: Optional["DashboardHandler"] = None
    _introspection_handler: Optional["IntrospectionHandler"] = None
    _calibration_handler: Optional["CalibrationHandler"] = None
    _routing_handler: Optional["RoutingHandler"] = None
    _evolution_handler: Optional["EvolutionHandler"] = None
    _plugins_handler: Optional["PluginsHandler"] = None
    _broadcast_handler: Optional["BroadcastHandler"] = None
    _handlers_initialized: bool = False

    # Thread pool for debate execution (prevents unbounded thread creation)
    _debate_executor: Optional["ThreadPoolExecutor"] = None
    _debate_executor_lock = threading.Lock()  # Lock for thread-safe executor creation
    # MAX_CONCURRENT_DEBATES imported from aragora.config

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

    @classmethod
    def _init_handlers(cls) -> None:
        """Initialize modular HTTP handlers with server context.

        Called lazily on first request. Creates handler instances with
        references to storage, ELO system, and other shared resources.
        """
        if cls._handlers_initialized or not HANDLERS_AVAILABLE:
            return

        # Build server context for handlers
        nomic_dir = None
        if cls.nomic_state_file:
            nomic_dir = cls.nomic_state_file.parent

        ctx = {
            "storage": cls.storage,
            "elo_system": cls.elo_system,
            "nomic_dir": nomic_dir,
            "debate_embeddings": cls.debate_embeddings,
            "critique_store": getattr(cls, 'critique_store', None),
            "document_store": cls.document_store,
            "persona_manager": getattr(cls, 'persona_manager', None),
            "position_ledger": getattr(cls, 'position_ledger', None),
        }

        # Initialize handlers
        cls._system_handler = SystemHandler(ctx)
        cls._debates_handler = DebatesHandler(ctx)
        cls._agents_handler = AgentsHandler(ctx)
        cls._pulse_handler = PulseHandler(ctx)
        cls._analytics_handler = AnalyticsHandler(ctx)
        cls._metrics_handler = MetricsHandler(ctx)
        cls._consensus_handler = ConsensusHandler(ctx)
        cls._belief_handler = BeliefHandler(ctx)
        cls._critique_handler = CritiqueHandler(ctx)
        cls._genesis_handler = GenesisHandler(ctx)
        cls._replays_handler = ReplaysHandler(ctx)
        cls._tournament_handler = TournamentHandler(ctx)
        cls._memory_handler = MemoryHandler(ctx)
        cls._leaderboard_handler = LeaderboardViewHandler(ctx)
        cls._document_handler = DocumentHandler(ctx)
        cls._verification_handler = VerificationHandler(ctx)
        cls._auditing_handler = AuditingHandler(ctx)
        cls._relationship_handler = RelationshipHandler(ctx)
        cls._moments_handler = MomentsHandler(ctx)
        cls._persona_handler = PersonaHandler(ctx)
        cls._dashboard_handler = DashboardHandler(ctx)
        cls._introspection_handler = IntrospectionHandler(ctx)
        cls._calibration_handler = CalibrationHandler(ctx)
        cls._routing_handler = RoutingHandler(ctx)
        cls._evolution_handler = EvolutionHandler(ctx)
        cls._plugins_handler = PluginsHandler(ctx)
        cls._broadcast_handler = BroadcastHandler(ctx)
        cls._laboratory_handler = LaboratoryHandler(ctx)
        cls._probes_handler = ProbesHandler(ctx)
        cls._handlers_initialized = True
        logger.info("[handlers] Modular handlers initialized (29 handlers)")

        # Log resource availability for observability
        cls._log_resource_availability(nomic_dir)

    @classmethod
    def _log_resource_availability(cls, nomic_dir) -> None:
        """Log which optional resources are available at startup."""
        resources = {
            "storage": cls.storage is not None,
            "elo_system": cls.elo_system is not None,
            "debate_embeddings": cls.debate_embeddings is not None,
            "document_store": cls.document_store is not None,
            "nomic_dir": nomic_dir is not None,
        }

        # Check database files if nomic_dir exists
        if nomic_dir:
            db_files = [
                ("positions_db", "aragora_positions.db"),
                ("personas_db", DB_PERSONAS_PATH),
                ("grounded_db", "grounded_positions.db"),
                ("insights_db", "insights.db"),
                ("calibration_db", "agent_calibration.db"),
                ("embeddings_db", "debate_embeddings.db"),
            ]
            for name, filename in db_files:
                resources[name] = (nomic_dir / filename).exists()

        available = [k for k, v in resources.items() if v]
        unavailable = [k for k, v in resources.items() if not v]

        if unavailable:
            logger.info(f"[resources] Available: {', '.join(available)}")
            logger.warning(f"[resources] Unavailable: {', '.join(unavailable)}")
        else:
            logger.info(f"[resources] All resources available: {', '.join(available)}")

    def _try_modular_handler(self, path: str, query: dict) -> bool:
        """Try to handle request via modular handlers.

        Returns True if handled, False if should fall through to legacy routes.
        """
        if not HANDLERS_AVAILABLE:
            return False

        # Ensure handlers are initialized
        self._init_handlers()

        # Convert query params from {key: [val]} to {key: val}
        query_dict = {k: v[0] if len(v) == 1 else v for k, v in query.items()}

        # Try each handler in order
        handlers = [
            self._system_handler,
            self._debates_handler,
            self._agents_handler,
            self._pulse_handler,
            self._analytics_handler,
            self._metrics_handler,
            self._consensus_handler,
            self._belief_handler,
            self._critique_handler,
            self._genesis_handler,
            self._replays_handler,
            self._tournament_handler,
            self._memory_handler,
            self._leaderboard_handler,
            self._document_handler,
            self._verification_handler,
            self._auditing_handler,
            self._relationship_handler,
            self._moments_handler,
            self._persona_handler,
            self._dashboard_handler,
            self._introspection_handler,
            self._calibration_handler,
            self._routing_handler,
            self._evolution_handler,
            self._plugins_handler,
            self._broadcast_handler,
            self._laboratory_handler,
            self._probes_handler,
        ]

        # Determine HTTP method for routing
        method = getattr(self, 'command', 'GET')

        for handler in handlers:
            if handler and handler.can_handle(path):
                try:
                    # Call handle() for GET, handle_post() for POST if available
                    if method == 'POST' and hasattr(handler, 'handle_post'):
                        result = handler.handle_post(path, query_dict, self)
                    else:
                        result = handler.handle(path, query_dict, self)

                    if result:
                        self.send_response(result.status_code)
                        self.send_header('Content-Type', result.content_type)
                        for h_name, h_val in result.headers.items():
                            self.send_header(h_name, h_val)
                        # Add CORS and security headers for modular handlers
                        self._add_cors_headers()
                        self._add_security_headers()
                        self.end_headers()
                        self.wfile.write(result.body)
                        return True
                except Exception as e:
                    logger.error(f"[handlers] Error in {handler.__class__.__name__}: {e}")
                    # Fall through to legacy handler on error
                    return False

        return False

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

        # Insights API (debate consensus feature)
        # Note: /api/debates/*, /api/health, /api/nomic/*, /api/history/*
        # are now handled by modular handlers (DebatesHandler, SystemHandler)
        if path == '/api/insights/recent':
            limit = self._safe_int(query, 'limit', 20, 100)
            self._get_recent_insights(limit)

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
        # plugins routes are NOW HANDLED BY modular handlers (BroadcastHandler,
        # LaboratoryHandler, RoutingHandler, VerificationHandler, ProbesHandler,
        # PluginsHandler, AuditingHandler)
        elif path == '/api/insights/extract-detailed':
            self._extract_detailed_insights()
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
            agents: Comma-separated agent list (optional, default: "claude,openai")
            rounds: Number of debate rounds (optional, default: 3)
            consensus: Consensus method (optional, default: "majority")

        Rate limited: requires auth when enabled.
        """
        global _active_debates

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

        # Validate required fields with length limits
        question = data.get('question', '').strip()
        if not question:
            self._send_json({"error": "question field is required"}, status=400)
            return
        if len(question) > 10000:
            self._send_json({"error": "question must be under 10,000 characters"}, status=400)
            return

        # Parse optional fields with validation
        auto_select = data.get('auto_select', False)
        auto_select_config = data.get('auto_select_config', {})

        # Auto-select agents or use provided list
        if auto_select and ROUTING_AVAILABLE:
            agents_str = self._auto_select_agents(question, auto_select_config)
        else:
            agents_str = data.get('agents', 'anthropic-api,openai-api,gemini,grok')

        try:
            rounds = min(max(int(data.get('rounds', 3)), 1), 10)  # Clamp to 1-10
        except (ValueError, TypeError):
            rounds = 3  # Default on invalid input
        consensus = data.get('consensus', 'majority')

        # Parse optional trending topic parameter
        trending_topic = None
        use_trending = data.get('use_trending', False)
        trending_category = data.get('trending_category', None)

        if use_trending:
            try:
                from aragora.pulse.ingestor import (
                    PulseManager,
                    TwitterIngestor,
                    HackerNewsIngestor,
                    RedditIngestor,
                )
                import asyncio as _async

                async def _fetch_topic():
                    manager = PulseManager()
                    manager.add_ingestor("twitter", TwitterIngestor())
                    manager.add_ingestor("hackernews", HackerNewsIngestor())
                    manager.add_ingestor("reddit", RedditIngestor())

                    filters = {}
                    if trending_category:
                        filters["categories"] = [trending_category]

                    topics = await manager.get_trending_topics(
                        limit_per_platform=3, filters=filters if filters else None
                    )
                    return manager.select_topic_for_debate(topics)

                # Run async in the current thread
                loop = _async.new_event_loop()
                try:
                    trending_topic = loop.run_until_complete(_fetch_topic())
                finally:
                    loop.close()

                if trending_topic:
                    logger.info(f"Selected trending topic: {trending_topic.topic}")
            except Exception as e:
                logger.warning(f"Trending topic fetch failed (non-fatal): {e}")

        # Generate debate ID
        debate_id = f"adhoc_{uuid.uuid4().hex[:8]}"

        # Track this debate (thread-safe)
        with _active_debates_lock:
            _active_debates[debate_id] = {
                "id": debate_id,
                "question": question,
                "status": "starting",
                "agents": agents_str,
                "rounds": rounds,
            }

        # Periodic cleanup of stale debates (every 100 debates)
        global _debate_cleanup_counter
        _debate_cleanup_counter += 1
        if _debate_cleanup_counter >= 100:
            _debate_cleanup_counter = 0
            _cleanup_stale_debates()

        # Set loop_id on emitter so events are tagged
        self.stream_emitter.set_loop_id(debate_id)

        # Start debate in background thread
        def run_debate():
            import asyncio as _asyncio

            try:
                # Parse agents with bounds check
                agent_list = [s.strip() for s in agents_str.split(",") if s.strip()]
                if len(agent_list) > MAX_AGENTS_PER_DEBATE:
                    _update_debate_status(debate_id, "error", error=f"Too many agents. Maximum: {MAX_AGENTS_PER_DEBATE}")
                    return
                if len(agent_list) < 2:
                    _update_debate_status(debate_id, "error", error="At least 2 agents required for a debate")
                    return

                agent_specs = []
                for spec in agent_list:
                    spec = spec.strip()
                    if ":" in spec:
                        agent_type, role = spec.split(":", 1)
                    else:
                        agent_type = spec
                        role = None
                    # Validate agent type against allowlist
                    if agent_type.lower() not in ALLOWED_AGENT_TYPES:
                        raise ValueError(f"Invalid agent type: {agent_type}. Allowed: {', '.join(sorted(ALLOWED_AGENT_TYPES))}")
                    agent_specs.append((agent_type, role))

                # Create agents with streaming support
                # All agents are proposers for full participation in all rounds
                agents = []
                failed_agents = []
                for i, (agent_type, role) in enumerate(agent_specs):
                    if role is None:
                        role = "proposer"  # All agents propose and participate fully
                    try:
                        agent = create_agent(
                            model_type=agent_type,
                            name=f"{agent_type}_{role}",
                            role=role,
                        )
                        # Check if API key is missing (for API-based agents)
                        if hasattr(agent, 'api_key') and not agent.api_key:
                            raise ValueError(f"Missing API key for {agent_type}")
                        # Wrap agent for token streaming if supported
                        agent = _wrap_agent_for_streaming(agent, self.stream_emitter, debate_id)
                        agents.append(agent)
                        logger.debug(f"Created agent {agent_type} successfully")
                    except Exception as e:
                        error_msg = f"Failed to create agent {agent_type}: {e}"
                        logger.error(error_msg)
                        failed_agents.append((agent_type, str(e)))
                        # Emit error event so frontend knows which agent failed
                        self.stream_emitter.emit(StreamEvent(
                            type=StreamEventType.ERROR,
                            data={"agent": agent_type, "error": str(e), "phase": "initialization"},
                            debate_id=debate_id,
                        ))

                # Check if enough agents were created
                if len(agents) < 2:
                    error_msg = f"Only {len(agents)} agents initialized (need at least 2). Failed: {', '.join(a for a, _ in failed_agents)}"
                    logger.error(error_msg)
                    _update_debate_status(debate_id, "error", error=error_msg)
                    self.stream_emitter.emit(StreamEvent(
                        type=StreamEventType.ERROR,
                        data={"error": error_msg, "failed_agents": [a for a, _ in failed_agents]},
                        debate_id=debate_id,
                    ))
                    return

                # Create environment and protocol
                env = Environment(task=question, context="", max_rounds=rounds)
                protocol = DebateProtocol(
                    rounds=rounds,
                    consensus=consensus,
                    proposer_count=len(agents),  # All agents propose initially
                    topology="all-to-all",  # Everyone critiques everyone
                )

                # Create arena with hooks and all available context systems
                hooks = create_arena_hooks(self.stream_emitter)
                arena = Arena(
                    env, agents, protocol,
                    event_hooks=hooks,
                    event_emitter=self.stream_emitter,
                    persona_manager=self.persona_manager,
                    debate_embeddings=self.debate_embeddings,
                    elo_system=self.elo_system,
                    position_tracker=self.position_tracker,
                    position_ledger=self.position_ledger,
                    flip_detector=self.flip_detector,
                    dissent_retriever=self.dissent_retriever,
                    moment_detector=self.moment_detector,
                    loop_id=debate_id,
                    trending_topic=trending_topic,
                )

                # Log and optionally reset circuit breaker for fresh debates
                cb_status = arena.circuit_breaker.get_all_status()
                if cb_status:
                    logger.debug(f"Agent status before debate: {cb_status}")
                    # Reset all circuits for ad-hoc debates to ensure full participation
                    open_circuits = [name for name, status in cb_status.items() if status["status"] == "open"]
                    if open_circuits:
                        logger.debug(f"Resetting open circuits for: {open_circuits}")
                        arena.circuit_breaker.reset_all()

                # Run debate with timeout protection (10 minutes max)
                _update_debate_status(debate_id, "running")
                async def run_with_timeout():
                    return await _asyncio.wait_for(arena.run(), timeout=600)
                result = _asyncio.run(run_with_timeout())
                _update_debate_status(debate_id, "completed", result={
                    "final_answer": result.final_answer,
                    "consensus_reached": result.consensus_reached,
                    "confidence": result.confidence,
                    "grounded_verdict": result.grounded_verdict.to_dict() if result.grounded_verdict else None,
                })

                # Emit LEADERBOARD_UPDATE after debate completes (if ELO system available)
                if self.elo_system:
                    try:
                        top_agents = self.elo_system.get_leaderboard(limit=10)
                        self.stream_emitter.emit(StreamEvent(
                            type=StreamEventType.LEADERBOARD_UPDATE,
                            data={
                                "debate_id": debate_id,
                                "leaderboard": [
                                    {"agent": a.agent_name, "elo": a.elo_rating, "wins": a.wins, "debates": a.total_debates}
                                    for a in top_agents
                                ],
                            }
                        ))
                    except Exception as e:
                        logger.debug(f"Leaderboard emission failed: {e}")

            except Exception as e:
                import traceback
                # Use safe error message for client, keep full trace server-side
                safe_msg = _safe_error_message(e, "debate_execution")
                error_trace = traceback.format_exc()
                _update_debate_status(debate_id, "error", error=safe_msg)
                # Log full traceback so thread failures aren't silent
                logger.error(f"[debate] Thread error in {debate_id}: {str(e)}\n{error_trace}")
                # Emit sanitized error event to client
                self.stream_emitter.emit(StreamEvent(
                    type=StreamEventType.ERROR,
                    data={"error": safe_msg, "debate_id": debate_id},
                ))

        # Use thread pool to prevent unbounded thread creation
        # Capture executor reference under lock to prevent race with shutdown
        with UnifiedHandler._debate_executor_lock:
            if UnifiedHandler._debate_executor is None:
                UnifiedHandler._debate_executor = ThreadPoolExecutor(
                    max_workers=MAX_CONCURRENT_DEBATES,
                    thread_name_prefix="debate-"
                )
            executor = UnifiedHandler._debate_executor

        try:
            executor.submit(run_debate)
        except RuntimeError as e:
            # Thread pool full or shut down
            logger.warning(f"Cannot submit debate: {e}")
            self._send_json({
                "success": False,
                "error": "Server at capacity. Please try again later.",
            }, status=503)
            return
        except (AttributeError, TypeError) as e:
            # Executor not initialized or invalid state
            logger.error(f"Failed to submit debate: {type(e).__name__}: {e}")
            self._send_json({
                "success": False,
                "error": "Internal server error",
            }, status=500)
            return

        # Return immediately with debate ID
        self._send_json({
            "success": True,
            "debate_id": debate_id,
            "question": question,
            "agents": agents_str.split(","),
            "rounds": rounds,
            "status": "starting",
            "message": "Debate started. Connect to WebSocket to receive events.",
        })

    def _list_documents(self) -> None:
        """List all uploaded documents."""
        if not self.document_store:
            self._send_json({"documents": [], "error": "Document storage not configured"})
            return

        docs = self.document_store.list_all()
        self._send_json({"documents": docs, "count": len(docs)})

    def _get_recent_insights(self, limit: int) -> None:
        """Get recent insights from InsightStore (debate consensus feature)."""
        if not self.insight_store:
            self._send_json({"error": "Insights not configured", "insights": []})
            return

        try:
            insights = _run_async(
                self.insight_store.get_recent_insights(limit=limit)
            )
            self._send_json({
                "insights": [
                    {
                        "id": i.id,
                        "type": i.type.value,
                        "title": i.title,
                        "description": i.description,
                        "confidence": i.confidence,
                        "agents_involved": i.agents_involved,
                        "evidence": i.evidence[:3] if i.evidence else [],
                    }
                    for i in insights
                ],
                "count": len(insights),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "insights"), "insights": []})

    def _extract_detailed_insights(self) -> None:
        """Extract detailed insights from debate content.

        POST body:
            content: The debate content to analyze (required)
            debate_id: Optional debate ID for context
            extract_claims: Whether to extract claims (default: True)
            extract_evidence: Whether to extract evidence chains (default: True)
            extract_patterns: Whether to extract argumentation patterns (default: True)

        Returns detailed analysis of the debate content.
        """
        if not self._check_rate_limit():
            return

        content_length = self._validate_content_length()
        if content_length is None:
            return  # Error already sent

        try:
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}

            content = data.get('content', '').strip()
            if not content:
                self._send_json({"error": "Missing required field: content"}, status=400)
                return

            debate_id = data.get('debate_id', '')
            extract_claims = data.get('extract_claims', True)
            extract_evidence = data.get('extract_evidence', True)
            extract_patterns = data.get('extract_patterns', True)

            result = {
                "debate_id": debate_id,
                "content_length": len(content),
            }

            # Extract claims if requested
            if extract_claims:
                claims = self._extract_claims_from_content(content)
                result["claims"] = claims

            # Extract evidence chains if requested
            if extract_evidence:
                evidence = self._extract_evidence_from_content(content)
                result["evidence_chains"] = evidence

            # Extract patterns if requested
            if extract_patterns:
                patterns = self._extract_patterns_from_content(content)
                result["patterns"] = patterns

            self._send_json(result)

        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=400)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "insight_extraction")}, status=500)

    def _extract_claims_from_content(self, content: str) -> list:
        """Extract claims from content using simple heuristics."""
        import re

        claims = []
        sentences = re.split(r'[.!?]+', content)

        # Claim indicators
        claim_patterns = [
            r'\b(therefore|thus|hence|consequently|as a result)\b',
            r'\b(I believe|we argue|it is clear|evidence shows)\b',
            r'\b(should|must|need to|ought to)\b',
            r'\b(is better|is worse|is more|is less)\b',
        ]

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            for pattern in claim_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    claims.append({
                        "text": sentence[:500],
                        "position": i,
                        "type": "argument" if "should" in sentence.lower() else "assertion",
                    })
                    break

        return claims[:20]  # Limit to 20 claims

    def _extract_evidence_from_content(self, content: str) -> list:
        """Extract evidence chains from content."""
        import re

        evidence = []

        # Evidence indicators
        evidence_patterns = [
            (r'according to ([^,.]+)', 'citation'),
            (r'research shows ([^.]+)', 'research'),
            (r'data indicates ([^.]+)', 'data'),
            (r'for example,? ([^.]+)', 'example'),
            (r'studies have shown ([^.]+)', 'study'),
        ]

        for pattern, etype in evidence_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                evidence.append({
                    "text": match.group(0)[:300],
                    "type": etype,
                    "source": match.group(1)[:100] if match.groups() else None,
                })

        return evidence[:15]  # Limit to 15 evidence items

    def _extract_patterns_from_content(self, content: str) -> list:
        """Extract argumentation patterns from content."""
        patterns = []

        content_lower = content.lower()

        # Pattern detection
        if 'on one hand' in content_lower and 'on the other hand' in content_lower:
            patterns.append({"type": "balanced_comparison", "strength": "strong"})

        if 'while' in content_lower and 'however' in content_lower:
            patterns.append({"type": "concession_rebuttal", "strength": "medium"})

        if content_lower.count('first') > 0 and content_lower.count('second') > 0:
            patterns.append({"type": "enumerated_argument", "strength": "medium"})

        if 'if' in content_lower and 'then' in content_lower:
            patterns.append({"type": "conditional_reasoning", "strength": "medium"})

        if 'because' in content_lower:
            count = content_lower.count('because')
            patterns.append({
                "type": "causal_reasoning",
                "strength": "strong" if count > 2 else "medium",
                "instances": count,
            })

        return patterns

    # NOTE: _run_capability_probe moved to handlers/probes.py (ProbesHandler)

    def _run_deep_audit(self) -> None:
        """Run a deep audit (Heavy3-inspired intensive multi-round debate protocol).

        POST body:
            task: The question/decision to audit (required)
            context: Additional context/documents (optional)
            agent_names: List of agent names to participate (optional, default: all available)
            model_type: Agent model type (optional, default: anthropic-api)
            config: Optional configuration object:
                rounds: Number of rounds (default: 6)
                enable_research: Enable web research (default: True)
                cross_examination_depth: Questions per finding (default: 3)
                risk_threshold: Severity threshold for findings (default: 0.7)
                audit_type: Pre-configured type: strategy, contract, code_architecture (optional)

        Returns:
            audit_id: Unique audit report ID
            task: The audited question
            recommendation: Final verdict recommendation
            confidence: Confidence in the recommendation
            unanimous_issues: Issues all agents agreed on
            split_opinions: Issues with disagreement
            risk_areas: Identified risk areas
            findings: Detailed findings list
            cross_examination_notes: Synthesizer cross-examination notes
            rounds_completed: Number of rounds completed
            duration_ms: Total audit duration
            agents: Participating agents
            elo_adjustments: ELO changes per agent
        """
        if not self._check_rate_limit():
            return

        try:
            from aragora.modes.deep_audit import (
                DeepAuditOrchestrator,
                DeepAuditConfig,
                STRATEGY_AUDIT,
                CONTRACT_AUDIT,
                CODE_ARCHITECTURE_AUDIT,
            )
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
            return  # Error already sent

        try:
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}

            task = data.get('task', '').strip()
            if not task:
                self._send_json({"error": "Missing required field: task"}, status=400)
                return

            context = data.get('context', '')
            agent_names = data.get('agent_names', [])
            model_type = data.get('model_type', 'anthropic-api')
            config_data = data.get('config', {})

            # Use pre-configured audit type if specified
            audit_type = config_data.get('audit_type', '')
            if audit_type == 'strategy':
                config = STRATEGY_AUDIT
            elif audit_type == 'contract':
                config = CONTRACT_AUDIT
            elif audit_type == 'code_architecture':
                config = CODE_ARCHITECTURE_AUDIT
            else:
                # Build custom config
                config = DeepAuditConfig(
                    rounds=min(_safe_int(config_data.get('rounds', 6), 6), 10),
                    enable_research=config_data.get('enable_research', True),
                    cross_examination_depth=min(_safe_int(config_data.get('cross_examination_depth', 3), 3), 10),
                    risk_threshold=_safe_float(config_data.get('risk_threshold', 0.7), 0.7),
                )

            # Create agents for the audit
            if not agent_names:
                # Default to 3 agents with different models
                agent_names = ['Claude-Analyst', 'Claude-Skeptic', 'Claude-Synthesizer']

            agents = []
            for name in agent_names[:5]:  # Limit to 5 agents
                if not re.match(SAFE_ID_PATTERN, name):
                    continue
                try:
                    agent = create_agent(model_type, name=name, role="proposer")
                    agents.append(agent)
                except Exception as e:
                    logger.debug(f"Failed to create audit agent {name}: {e}")

            if len(agents) < 2:
                self._send_json({
                    "error": "Need at least 2 agents for deep audit",
                    "hint": f"Only created {len(agents)} agent(s)"
                }, status=400)
                return

            # Get stream hooks for real-time updates
            audit_hooks = None
            if hasattr(self.server, 'stream_server') and self.server.stream_server:
                from .nomic_stream import create_nomic_hooks
                audit_hooks = create_nomic_hooks(self.server.stream_server.emitter)

            audit_id = f"audit-{uuid.uuid4().hex[:8]}"
            import time
            start_time = time.time()

            # Emit audit start event
            if audit_hooks and 'on_audit_start' in audit_hooks:
                audit_hooks['on_audit_start'](
                    audit_id=audit_id,
                    task=task,
                    agents=[a.name for a in agents],
                    config={
                        "rounds": config.rounds,
                        "enable_research": config.enable_research,
                        "cross_examination_depth": config.cross_examination_depth,
                        "risk_threshold": config.risk_threshold,
                    }
                )

            # Create orchestrator and run audit
            orchestrator = DeepAuditOrchestrator(agents, config)

            import asyncio

            async def run_audit():
                return await orchestrator.run(task, context)

            # Execute in event loop
            # Use asyncio.run() for proper event loop lifecycle management
            try:
                verdict = asyncio.run(run_audit())
            except Exception as e:
                self._send_json({
                    "error": f"Deep audit execution failed: {str(e)}"
                }, status=500)
                return

            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Calculate ELO adjustments based on findings contribution
            elo_adjustments = {}
            if self.elo_system:
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

            # Emit audit verdict event
            if audit_hooks and 'on_audit_verdict' in audit_hooks:
                audit_hooks['on_audit_verdict'](
                    audit_id=audit_id,
                    task=task,
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

            # Save results to .nomic/audits/
            if self.nomic_dir:
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

            # Build response
            self._send_json({
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
            })

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
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected, no response needed
            pass

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


def _shutdown_debate_executor() -> None:
    """Shutdown debate executor on process exit."""
    with UnifiedHandler._debate_executor_lock:
        if UnifiedHandler._debate_executor:
            logger.info("Shutting down debate executor...")
            UnifiedHandler._debate_executor.shutdown(wait=True, cancel_futures=False)
            UnifiedHandler._debate_executor = None


atexit.register(_shutdown_debate_executor)


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
    ):
        self.http_port = http_port
        self.ws_port = ws_port
        self.ws_host = ws_host
        self.http_host = http_host
        self.static_dir = static_dir
        self.nomic_dir = nomic_dir
        self.storage = storage

        # Create WebSocket server
        self.stream_server = DebateStreamServer(host=ws_host, port=ws_port)

        # Initialize Supabase persistence if available
        self.persistence = None
        if enable_persistence and PERSISTENCE_AVAILABLE:
            self.persistence = SupabaseClient()
            if self.persistence.is_configured:
                logger.info("[server] Supabase persistence enabled")
            else:
                self.persistence = None

        # Setup HTTP handler
        UnifiedHandler.storage = storage
        UnifiedHandler.static_dir = static_dir
        UnifiedHandler.stream_emitter = self.stream_server.emitter
        UnifiedHandler.persistence = self.persistence
        if nomic_dir:
            UnifiedHandler.nomic_state_file = nomic_dir / "nomic_state.json"
            # Initialize InsightStore from nomic directory
            if INSIGHTS_AVAILABLE:
                insights_path = nomic_dir / DB_INSIGHTS_PATH
                if insights_path.exists():
                    UnifiedHandler.insight_store = InsightStore(str(insights_path))
                    logger.info("[server] InsightStore loaded for API access")
            # Initialize EloSystem from nomic directory
            if RANKING_AVAILABLE:
                elo_path = nomic_dir / "agent_elo.db"
                if elo_path.exists():
                    UnifiedHandler.elo_system = EloSystem(str(elo_path))
                    logger.info("[server] EloSystem loaded for leaderboard API")

            # Initialize FlipDetector from nomic directory
            if FLIP_DETECTOR_AVAILABLE:
                positions_path = nomic_dir / "aragora_positions.db"
                if positions_path.exists():
                    UnifiedHandler.flip_detector = FlipDetector(str(positions_path))
                    logger.info("[server] FlipDetector loaded for position reversal API")

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
            if PERSONAS_AVAILABLE:
                personas_path = nomic_dir / "personas.db"
                UnifiedHandler.persona_manager = PersonaManager(str(personas_path))
                logger.info("[server] PersonaManager loaded for agent specialization")

            # Initialize PositionLedger for truth-grounded personas
            if POSITION_LEDGER_AVAILABLE:
                ledger_path = nomic_dir / "position_ledger.db"
                try:
                    UnifiedHandler.position_ledger = PositionLedger(db_path=str(ledger_path))
                    logger.info("[server] PositionLedger loaded for truth-grounded personas")
                except Exception as e:
                    logger.warning(f"[server] PositionLedger initialization failed: {e}")

            # Initialize DebateEmbeddingsDatabase for historical memory
            if EMBEDDINGS_AVAILABLE:
                embeddings_path = nomic_dir / "debate_embeddings.db"
                try:
                    UnifiedHandler.debate_embeddings = DebateEmbeddingsDatabase(str(embeddings_path))
                    logger.info("[server] DebateEmbeddings loaded for historical memory")
                except Exception as e:
                    logger.warning(f"[server] DebateEmbeddings initialization failed: {e}")

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
        """Run HTTP server in a thread."""
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                server = HTTPServer((self.http_host, self.http_port), UnifiedHandler)
                logger.info(f"HTTP server listening on {self.http_host}:{self.http_port}")
                server.serve_forever()
                break  # Normal exit
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
        logger.info(f"  HTTP API:   http://localhost:{self.http_port}")
        logger.info(f"  WebSocket:  ws://localhost:{self.ws_port}")
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
) -> None:
    """
    Convenience function to run the unified server.

    Args:
        http_port: Port for HTTP API (default 8080)
        ws_port: Port for WebSocket streaming (default 8765)
        static_dir: Directory containing static files (dashboard build)
        nomic_dir: Path to .nomic directory for state access
    """
    server = UnifiedServer(
        http_port=http_port,
        ws_port=ws_port,
        static_dir=static_dir,
        nomic_dir=nomic_dir,
    )
    await server.start()
