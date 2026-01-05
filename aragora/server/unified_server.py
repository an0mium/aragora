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
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import Optional, Dict
from urllib.parse import urlparse, parse_qs

from .stream import DebateStreamServer, SyncEventEmitter, StreamEvent, StreamEventType, create_arena_hooks
from .storage import DebateStorage
from .documents import DocumentStore, parse_document, get_supported_formats, SUPPORTED_EXTENSIONS
from .auth import auth_config, check_auth
from .cors_config import cors_config

# For ad-hoc debates
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid
import logging

# Configure module logger
logger = logging.getLogger(__name__)


def _safe_error_message(e: Exception, context: str = "") -> str:
    """Return a sanitized error message for client responses.

    Logs the full error server-side while returning a generic message to clients.
    This prevents information disclosure of internal details like file paths,
    stack traces, or sensitive configuration.
    """
    # Log full details server-side for debugging
    logger.error(f"Error in {context}: {type(e).__name__}: {e}", exc_info=True)

    # Map common exceptions to user-friendly messages
    error_type = type(e).__name__
    if error_type in ("FileNotFoundError", "OSError"):
        return "Resource not found"
    elif error_type in ("json.JSONDecodeError", "ValueError"):
        return "Invalid data format"
    elif error_type in ("PermissionError",):
        return "Access denied"
    elif error_type in ("TimeoutError", "asyncio.TimeoutError"):
        return "Operation timed out"
    else:
        return "An error occurred"


# Valid agent types (allowlist for security)
ALLOWED_AGENT_TYPES = frozenset({
    # CLI-based
    "codex", "claude", "openai", "gemini-cli", "grok-cli", "qwen-cli", "deepseek-cli", "kilocode",
    # API-based (direct)
    "gemini", "ollama", "anthropic-api", "openai-api", "grok",
    # API-based (via OpenRouter)
    "deepseek", "deepseek-r1", "llama", "mistral", "openrouter",
})

# Maximum number of agents per debate (DoS protection)
MAX_AGENTS_PER_DEBATE = 10
MAX_MULTIPART_PARTS = 10
# Maximum content length for POST requests (100MB - DoS protection)
MAX_CONTENT_LENGTH = 100 * 1024 * 1024
# Maximum content length for JSON API requests (10MB)
MAX_JSON_CONTENT_LENGTH = 10 * 1024 * 1024

# Safe ID pattern for path segments (prevent path traversal)
SAFE_ID_PATTERN = r'^[a-zA-Z0-9_-]+$'

# Trusted proxies for X-Forwarded-For header validation
# Only trust X-Forwarded-For if request comes from these IPs
import os
TRUSTED_PROXIES = frozenset(
    os.getenv('ARAGORA_TRUSTED_PROXIES', '127.0.0.1,::1,localhost').split(',')
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


# Optional Supabase persistence
try:
    from aragora.persistence import SupabaseClient
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False
    SupabaseClient = None

# Optional InsightStore for debate insights
try:
    from aragora.insights.store import InsightStore
    INSIGHTS_AVAILABLE = True
except ImportError:
    INSIGHTS_AVAILABLE = False
    InsightStore = None

# Optional EloSystem for agent rankings
try:
    from aragora.ranking.elo import EloSystem
    RANKING_AVAILABLE = True
except ImportError:
    RANKING_AVAILABLE = False
    EloSystem = None

# Optional FlipDetector for position reversal detection
try:
    from aragora.insights.flip_detector import (
        FlipDetector,
        format_flip_for_ui,
        format_consistency_for_ui,
    )
    FLIP_DETECTOR_AVAILABLE = True
except ImportError:
    FLIP_DETECTOR_AVAILABLE = False
    FlipDetector = None

# Optional debate orchestrator for ad-hoc debates
try:
    from aragora.debate.orchestrator import Arena, DebateProtocol
    from aragora.agents.base import create_agent
    from aragora.core import Environment
    DEBATE_AVAILABLE = True
except ImportError:
    DEBATE_AVAILABLE = False
    Arena = None
    DebateProtocol = None
    create_agent = None
    Environment = None

# Optional PersonaManager for agent specialization
try:
    from aragora.agents.personas import PersonaManager
    PERSONAS_AVAILABLE = True
except ImportError:
    PERSONAS_AVAILABLE = False
    PersonaManager = None

# Optional DebateEmbeddingsDatabase for historical memory
try:
    from aragora.debate.embeddings import DebateEmbeddingsDatabase
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    DebateEmbeddingsDatabase = None

# Optional ConsensusMemory for historical consensus data
try:
    from aragora.memory.consensus import ConsensusMemory, DissentRetriever
    CONSENSUS_MEMORY_AVAILABLE = True
except ImportError:
    CONSENSUS_MEMORY_AVAILABLE = False
    ConsensusMemory = None
    DissentRetriever = None

# Optional CalibrationTracker for agent calibration
try:
    from aragora.agents.calibration import CalibrationTracker
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    CalibrationTracker = None

# Optional PulseManager for trending topics
try:
    from aragora.pulse.ingestor import PulseManager, TrendingTopic, TwitterIngestor
    PULSE_AVAILABLE = True
except ImportError:
    PULSE_AVAILABLE = False
    PulseManager = None
    TrendingTopic = None

# Optional FormalVerificationManager for theorem proving
try:
    from aragora.verification.formal import (
        FormalVerificationManager,
        get_formal_verification_manager,
    )
    FORMAL_VERIFICATION_AVAILABLE = True
except ImportError:
    FORMAL_VERIFICATION_AVAILABLE = False
    FormalVerificationManager = None
    get_formal_verification_manager = None

# Optional Broadcast module for podcast generation
try:
    from aragora.broadcast import broadcast_debate
    from aragora.debate.traces import DebateTrace
    BROADCAST_AVAILABLE = True
except ImportError:
    BROADCAST_AVAILABLE = False
    broadcast_debate = None
    DebateTrace = None

# Optional RelationshipTracker for agent network analysis
try:
    from aragora.agents.grounded import RelationshipTracker
    RELATIONSHIP_TRACKER_AVAILABLE = True
except ImportError:
    RELATIONSHIP_TRACKER_AVAILABLE = False
    RelationshipTracker = None

# Optional PositionLedger for truth-grounded personas
try:
    from aragora.agents.grounded import PositionLedger
    POSITION_LEDGER_AVAILABLE = True
except ImportError:
    POSITION_LEDGER_AVAILABLE = False
    PositionLedger = None

# Optional CritiqueStore for pattern retrieval
try:
    from aragora.memory.store import CritiqueStore
    CRITIQUE_STORE_AVAILABLE = True
except ImportError:
    CRITIQUE_STORE_AVAILABLE = False
    CritiqueStore = None

# Optional export module for debate artifact export
try:
    from aragora.export import DebateArtifact, CSVExporter, DOTExporter, StaticHTMLExporter
    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False
    DebateArtifact = None
    CSVExporter = None
    DOTExporter = None
    StaticHTMLExporter = None

# Optional CapabilityProber for vulnerability detection
try:
    from aragora.modes.prober import CapabilityProber
    PROBER_AVAILABLE = True
except ImportError:
    PROBER_AVAILABLE = False
    CapabilityProber = None

# Optional RedTeamMode for adversarial testing
try:
    from aragora.modes.redteam import RedTeamMode
    REDTEAM_AVAILABLE = True
except ImportError:
    REDTEAM_AVAILABLE = False
    RedTeamMode = None

# Optional PersonaLaboratory for emergent traits
try:
    from aragora.agents.laboratory import PersonaLaboratory
    LABORATORY_AVAILABLE = True
except ImportError:
    LABORATORY_AVAILABLE = False
    PersonaLaboratory = None

# Optional BeliefNetwork for debate cruxes
try:
    from aragora.reasoning.belief import BeliefNetwork, BeliefPropagationAnalyzer
    BELIEF_NETWORK_AVAILABLE = True
except ImportError:
    BELIEF_NETWORK_AVAILABLE = False
    BeliefNetwork = None
    BeliefPropagationAnalyzer = None

# Optional ProvenanceTracker for claim support
try:
    from aragora.reasoning.provenance import ProvenanceTracker
    PROVENANCE_AVAILABLE = True
except ImportError:
    PROVENANCE_AVAILABLE = False
    ProvenanceTracker = None

# Optional MomentDetector for significant agent moments
try:
    from aragora.agents.grounded import MomentDetector
    MOMENT_DETECTOR_AVAILABLE = True
except ImportError:
    MOMENT_DETECTOR_AVAILABLE = False
    MomentDetector = None

# Optional ImpasseDetector for debate deadlock detection
try:
    from aragora.debate.counterfactual import ImpasseDetector
    IMPASSE_DETECTOR_AVAILABLE = True
except ImportError:
    IMPASSE_DETECTOR_AVAILABLE = False
    ImpasseDetector = None

# Optional ConvergenceDetector for semantic position convergence
try:
    from aragora.debate.convergence import ConvergenceDetector
    CONVERGENCE_DETECTOR_AVAILABLE = True
except ImportError:
    CONVERGENCE_DETECTOR_AVAILABLE = False
    ConvergenceDetector = None

# Optional AgentSelector for routing recommendations and auto team selection
try:
    from aragora.routing.selection import AgentSelector, AgentProfile, TaskRequirements
    ROUTING_AVAILABLE = True
except ImportError:
    ROUTING_AVAILABLE = False
    AgentSelector = None
    AgentProfile = None
    TaskRequirements = None

# Optional TournamentManager for tournament standings
try:
    from aragora.tournaments.tournament import TournamentManager
    TOURNAMENT_AVAILABLE = True
except ImportError:
    TOURNAMENT_AVAILABLE = False
    TournamentManager = None

# Optional PromptEvolver for evolution history
try:
    from aragora.evolution.evolver import PromptEvolver
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False
    PromptEvolver = None

# Optional ContinuumMemory for multi-timescale memory
try:
    from aragora.memory.continuum import ContinuumMemory, MemoryTier
    CONTINUUM_AVAILABLE = True
except ImportError:
    CONTINUUM_AVAILABLE = False
    ContinuumMemory = None
    MemoryTier = None

# Optional InsightExtractor for debate insights
try:
    from aragora.insights.extractor import InsightExtractor
    INSIGHT_EXTRACTOR_AVAILABLE = True
except ImportError:
    INSIGHT_EXTRACTOR_AVAILABLE = False
    InsightExtractor = None

# Modular HTTP handlers for endpoint routing
try:
    from aragora.server.handlers import (
        SystemHandler,
        DebatesHandler,
        AgentsHandler,
        PulseHandler,
        AnalyticsHandler,
        HandlerResult,
    )
    HANDLERS_AVAILABLE = True
except ImportError:
    HANDLERS_AVAILABLE = False
    SystemHandler = None
    DebatesHandler = None
    AgentsHandler = None
    PulseHandler = None
    AnalyticsHandler = None
    HandlerResult = None

# Track active ad-hoc debates
_active_debates: dict[str, dict] = {}
_active_debates_lock = threading.Lock()  # Thread-safe access to _active_debates


def _wrap_agent_for_streaming(agent, emitter: SyncEventEmitter, debate_id: str):
    """Wrap an agent to emit token streaming events.

    If the agent has a generate_stream() method, we override its generate()
    to call generate_stream() and emit TOKEN_* events.
    """
    from datetime import datetime

    # Check if agent supports streaming
    if not hasattr(agent, 'generate_stream'):
        return agent

    # Store original generate method
    original_generate = agent.generate

    async def streaming_generate(prompt: str, context=None):
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


def _run_async(coro):
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
    flip_detector: Optional["FlipDetector"] = None  # FlipDetector for position reversals
    persona_manager: Optional["PersonaManager"] = None  # PersonaManager for agent specialization
    debate_embeddings: Optional["DebateEmbeddingsDatabase"] = None  # Historical memory
    position_tracker: Optional["PositionTracker"] = None  # PositionTracker for truth-grounded personas
    position_ledger: Optional["PositionLedger"] = None  # PositionLedger for grounded positions
    consensus_memory: Optional["ConsensusMemory"] = None  # ConsensusMemory for historical positions
    dissent_retriever: Optional["DissentRetriever"] = None  # DissentRetriever for minority views

    # Modular HTTP handlers (initialized lazily)
    _system_handler: Optional["SystemHandler"] = None
    _debates_handler: Optional["DebatesHandler"] = None
    _agents_handler: Optional["AgentsHandler"] = None
    _pulse_handler: Optional["PulseHandler"] = None
    _analytics_handler: Optional["AnalyticsHandler"] = None
    _handlers_initialized: bool = False

    # Thread pool for debate execution (prevents unbounded thread creation)
    _debate_executor: Optional["ThreadPoolExecutor"] = None
    _debate_executor_lock = threading.Lock()  # Lock for thread-safe executor creation
    MAX_CONCURRENT_DEBATES = 10  # Limit concurrent debates to prevent resource exhaustion

    # Upload rate limiting (IP-based, independent of auth)
    _upload_counts: Dict[str, list] = {}  # IP -> list of upload timestamps
    _upload_counts_lock = threading.Lock()
    MAX_UPLOADS_PER_MINUTE = 5  # Maximum uploads per IP per minute
    MAX_UPLOADS_PER_HOUR = 30  # Maximum uploads per IP per hour

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
        }

        # Initialize handlers
        cls._system_handler = SystemHandler(ctx)
        cls._debates_handler = DebatesHandler(ctx)
        cls._agents_handler = AgentsHandler(ctx)
        cls._pulse_handler = PulseHandler(ctx)
        cls._analytics_handler = AnalyticsHandler(ctx)
        cls._handlers_initialized = True
        logger.info("[handlers] Modular handlers initialized (5 handlers)")

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
        ]

        for handler in handlers:
            if handler and handler.can_handle(path):
                try:
                    result = handler.handle(path, query_dict, self)
                    if result:
                        self.send_response(result.status_code)
                        self.send_header('Content-Type', result.content_type)
                        for h_name, h_val in result.headers.items():
                            self.send_header(h_name, h_val)
                        self.end_headers()
                        self.wfile.write(result.body)
                        return True
                except Exception as e:
                    logger.error(f"[handlers] Error in {handler.__class__.__name__}: {e}")
                    # Fall through to legacy handler on error
                    return False

        return False

    def _validate_content_length(self, max_size: int = None) -> Optional[int]:
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
        if remote_ip in TRUSTED_PROXIES:
            # Only trust X-Forwarded-For from trusted proxies
            forwarded = self.headers.get('X-Forwarded-For', '')
            client_ip = forwarded.split(',')[0].strip() if forwarded else remote_ip
        else:
            # Untrusted source - use direct connection IP
            client_ip = remote_ip

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
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

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

        # Pulse API (trending topics)
        elif path == '/api/pulse/trending':
            limit = self._safe_int(query, 'limit', 10, 50)
            self._get_trending_topics(limit)

        # Document API
        elif path == '/api/documents':
            self._list_documents()
        elif path == '/api/documents/formats':
            self._send_json(get_supported_formats())
        elif path.startswith('/api/documents/'):
            doc_id = self._extract_path_segment(path, 3, "document_id")
            if doc_id is None:
                return
            self._get_document(doc_id)

        # Replay API
        elif path == '/api/replays':
            self._list_replays()
        elif path.startswith('/api/replays/') and not path.endswith('/fork'):
            replay_id = self._extract_path_segment(path, 3, "replay_id")
            if replay_id is None:
                return
            self._get_replay(replay_id)

        # Learning Evolution API
        elif path == '/api/learning/evolution':
            self._get_learning_evolution()

        # Flip Detection API
        # Note: /api/agent/*/consistency is handled by AgentsHandler
        elif path == '/api/flips/recent':
            limit = self._safe_int(query, 'limit', 20, 100)
            self._get_recent_flips(limit)
        elif path == '/api/flips/summary':
            self._get_flip_summary()
        elif path.startswith('/api/agent/') and path.endswith('/flips'):
            agent = self._extract_path_segment(path, 3, "agent")
            if agent is None:
                return
            limit = self._safe_int(query, 'limit', 20, 100)
            self._get_agent_flips(agent, limit)

        # Persona API
        elif path == '/api/personas':
            self._get_all_personas()
        elif path.startswith('/api/agent/') and path.endswith('/persona'):
            agent = self._extract_path_segment(path, 3, "agent")
            if agent is None:
                return
            self._get_agent_persona(agent)
        elif path.startswith('/api/agent/') and path.endswith('/performance'):
            agent = self._extract_path_segment(path, 3, "agent")
            if agent is None:
                return
            self._get_agent_performance(agent)
        elif path.startswith('/api/agent/') and path.endswith('/domains'):
            agent = self._extract_path_segment(path, 3, "agent")
            if agent is None:
                return
            limit = self._safe_int(query, 'limit', 5, 20)
            self._get_agent_domains(agent, limit)
        elif path.startswith('/api/agent/') and path.endswith('/grounded-persona'):
            agent = self._extract_path_segment(path, 3, "agent")
            if agent is None:
                return
            self._get_grounded_persona(agent)
        elif path.startswith('/api/agent/') and path.endswith('/identity-prompt'):
            agent = self._extract_path_segment(path, 3, "agent")
            if agent is None:
                return
            sections = query.get('sections', [None])[0]
            self._get_identity_prompt(agent, sections)

        # Agent Position Accuracy API (PositionTracker integration)
        elif path.startswith('/api/agent/') and path.endswith('/accuracy'):
            agent = self._extract_path_segment(path, 3, "agent")
            if agent is None:
                return
            self._get_agent_accuracy(agent)

        # Consensus Memory API (expose underutilized databases)
        elif path == '/api/consensus/similar':
            topic = query.get('topic', [''])[0]
            # Validate topic parameter
            if not topic or len(topic) > 500:
                self._send_json({"error": "Topic required (max 500 chars)"}, status=400)
                return
            topic = topic.strip()[:500]  # Sanitize
            limit = self._safe_int(query, 'limit', 5, 20)
            self._get_similar_debates(topic, limit)
        elif path == '/api/consensus/settled':
            min_confidence = self._safe_float(query, 'min_confidence', 0.8, 0.0, 1.0)
            limit = self._safe_int(query, 'limit', 20, 100)
            self._get_settled_topics(min_confidence, limit)
        elif path == '/api/consensus/stats':
            self._get_consensus_stats()
        elif path == '/api/consensus/dissents':
            # Topic is optional - if not provided, get recent dissents globally
            topic = query.get('topic', [''])[0]
            if topic and len(topic) > 500:
                topic = topic[:500]
            domain = query.get('domain', [None])[0]
            limit = self._safe_int(query, 'limit', 10, 50)
            self._get_recent_dissents(topic.strip() if topic else None, domain, limit)
        elif path == '/api/consensus/contrarian-views':
            # Topic is optional - if not provided, get recent contrarian views globally
            topic = query.get('topic', [''])[0]
            if topic and len(topic) > 500:
                topic = topic[:500]
            domain = query.get('domain', [None])[0]
            limit = self._safe_int(query, 'limit', 10, 50)
            self._get_contrarian_views(topic.strip() if topic else None, domain, limit)
        elif path == '/api/consensus/risk-warnings':
            # Topic is optional - if not provided, get recent risk warnings globally
            topic = query.get('topic', [''])[0]
            if topic and len(topic) > 500:
                topic = topic[:500]
            domain = query.get('domain', [None])[0]
            limit = self._safe_int(query, 'limit', 10, 50)
            self._get_risk_warnings(topic.strip() if topic else None, domain, limit)
        elif path.startswith('/api/consensus/domain/'):
            domain = self._extract_path_segment(path, 4, "domain")
            if domain is None:
                return
            limit = self._safe_int(query, 'limit', 50, 200)
            self._get_domain_history(domain, limit)

        # Combined Agent Profile API
        elif path.startswith('/api/agent/') and path.endswith('/profile'):
            agent = self._extract_path_segment(path, 3, "agent")
            if agent is None:
                return
            self._get_agent_full_profile(agent)

        # Debate Analytics API
        elif path == '/api/analytics/disagreements':
            limit = self._safe_int(query, 'limit', 20, 100)
            self._get_disagreement_report(limit)
        elif path == '/api/analytics/role-rotation':
            limit = self._safe_int(query, 'limit', 50, 200)
            self._get_role_rotation_report(limit)
        elif path == '/api/analytics/early-stops':
            limit = self._safe_int(query, 'limit', 20, 100)
            self._get_early_stop_signals(limit)

        # Modes API
        elif path == '/api/modes':
            self._list_available_modes()

        # Agent Position Tracking API
        elif path.startswith('/api/agent/') and path.endswith('/positions'):
            agent = self._extract_path_segment(path, 3, "agent")
            if agent is None:
                return
            limit = self._safe_int(query, 'limit', 50, 200)
            self._get_agent_positions(agent, limit)

        # Agent Relationship Network API
        elif path.startswith('/api/agent/') and path.endswith('/network'):
            agent = self._extract_path_segment(path, 3, "agent")
            if agent is None:
                return
            self._get_agent_network(agent)
        elif path.startswith('/api/agent/') and path.endswith('/rivals'):
            agent = self._extract_path_segment(path, 3, "agent")
            if agent is None:
                return
            limit = self._safe_int(query, 'limit', 5, 20)
            self._get_agent_rivals(agent, limit)
        elif path.startswith('/api/agent/') and path.endswith('/allies'):
            agent = self._extract_path_segment(path, 3, "agent")
            if agent is None:
                return
            limit = self._safe_int(query, 'limit', 5, 20)
            self._get_agent_allies(agent, limit)

        # Agent Moments API (significant achievements timeline)
        elif path.startswith('/api/agent/') and path.endswith('/moments'):
            agent = self._extract_path_segment(path, 3, "agent")
            if agent is None:
                return
            limit = self._safe_int(query, 'limit', 10, 50)
            self._get_agent_moments(agent, limit)

        # System Statistics API
        elif path == '/api/ranking/stats':
            self._get_ranking_stats()
        elif path == '/api/memory/stats' or path == '/api/memory/tier-stats':
            self._get_memory_stats()
        elif path == '/api/critiques/patterns':
            limit = self._safe_int(query, 'limit', 10, 50)
            min_success = self._safe_float(query, 'min_success', 0.5, 0.0, 1.0)
            self._get_critique_patterns(limit, min_success)
        elif path == '/api/critiques/archive':
            self._get_archive_stats()
        elif path == '/api/reputation/all':
            self._get_all_reputations()
        elif path.startswith('/api/agent/') and path.endswith('/reputation'):
            agent = self._extract_path_segment(path, 3, "agent")
            if agent is None:
                return
            self._get_agent_reputation(agent)

        # Agent Comparison API
        elif path == '/api/agent/compare':
            agent_a = query.get('agent_a', [None])[0]
            agent_b = query.get('agent_b', [None])[0]
            if not agent_a or not agent_b:
                self._send_json({"error": "agent_a and agent_b query params required"}, status=400)
            else:
                self._get_agent_comparison(agent_a, agent_b)

        # Head-to-Head & Opponent Briefing API
        elif '/head-to-head/' in path and path.startswith('/api/agent/'):
            # Pattern: /api/agent/{agent}/head-to-head/{opponent}
            parts = path.split('/')
            if len(parts) >= 6:
                agent = parts[3]
                opponent = parts[5]
                self._get_head_to_head(agent, opponent)
            else:
                self._send_json({"error": "Invalid path format"}, status=400)
        elif '/opponent-briefing/' in path and path.startswith('/api/agent/'):
            # Pattern: /api/agent/{agent}/opponent-briefing/{opponent}
            parts = path.split('/')
            if len(parts) >= 6:
                agent = parts[3]
                opponent = parts[5]
                self._get_opponent_briefing(agent, opponent)
            else:
                self._send_json({"error": "Invalid path format"}, status=400)

        # Introspection API (Agent Self-Awareness)
        elif path == '/api/introspection/all':
            self._get_all_introspection()
        elif path == '/api/introspection/leaderboard':
            limit = self._safe_int(query, 'limit', 10, 50)
            self._get_introspection_leaderboard(limit)
        elif path.startswith('/api/introspection/agents/'):
            agent = path.split('/')[-1]
            if not agent or not re.match(SAFE_ID_PATTERN, agent):
                self._send_json({"error": "Invalid agent name"}, status=400)
            else:
                self._get_agent_introspection(agent)

        # Calibration Curve API
        elif path.startswith('/api/agent/') and path.endswith('/calibration-curve'):
            agent = self._extract_path_segment(path, 3, "agent")
            if agent is None:
                return
            buckets = self._safe_int(query, 'buckets', 10, 20)
            domain = query.get('domain', [None])[0]
            self._get_calibration_curve(agent, buckets, domain)

        # Meta-Critique API
        elif path.startswith('/api/debate/') and path.endswith('/meta-critique'):
            debate_id = self._extract_path_segment(path, 3, "debate_id")
            if debate_id is None:
                return
            self._get_meta_critique(debate_id)

        # Debate Graph Stats API
        elif path.startswith('/api/debate/') and path.endswith('/graph/stats'):
            debate_id = self._extract_path_segment(path, 3, "debate_id")
            if debate_id is None:
                return
            self._get_debate_graph_stats(debate_id)

        # Laboratory API - Emergent Traits
        elif path == '/api/laboratory/emergent-traits':
            min_confidence = self._safe_float(query, 'min_confidence', 0.5, 0.0, 1.0)
            limit = self._safe_int(query, 'limit', 10, 50)
            self._get_emergent_traits(min_confidence, limit)

        # Belief Network API - Debate Cruxes
        elif path.startswith('/api/belief-network/') and path.endswith('/cruxes'):
            debate_id = self._extract_path_segment(path, 3, "debate_id")
            if debate_id is None:
                return
            top_k = self._safe_int(query, 'top_k', 3, 10)
            self._get_debate_cruxes(debate_id, top_k)

        # Provenance API - Claim Support
        elif '/claims/' in path and path.endswith('/support'):
            # Pattern: /api/provenance/:debate_id/claims/:claim_id/support
            parts = path.split('/')
            if len(parts) >= 6:
                debate_id = parts[3]
                claim_id = parts[5]
                self._get_claim_support(debate_id, claim_id)
            else:
                self._send_json({"error": "Invalid path format"}, status=400)

        # Tournament API
        elif path == '/api/tournaments':
            self._list_tournaments()
        elif path.startswith('/api/tournaments/') and path.endswith('/standings'):
            tournament_id = self._extract_path_segment(path, 3, "tournament_id")
            if tournament_id is None:
                return
            self._get_tournament_standings(tournament_id)

        # Best Team Combinations API
        elif path == '/api/routing/best-teams':
            min_debates = self._safe_int(query, 'min_debates', 3, 20)
            limit = self._safe_int(query, 'limit', 10, 50)
            self._get_best_team_combinations(min_debates, limit)

        # Evolution History API
        elif path.startswith('/api/evolution/') and path.endswith('/history'):
            agent = self._extract_path_segment(path, 3, "agent")
            if agent is None:
                return
            limit = self._safe_int(query, 'limit', 10, 50)
            self._get_evolution_history(agent, limit)

        # Load-Bearing Claims API
        elif path.startswith('/api/belief-network/') and path.endswith('/load-bearing-claims'):
            debate_id = self._extract_path_segment(path, 3, "debate_id")
            if debate_id is None:
                return
            limit = self._safe_int(query, 'limit', 5, 20)
            self._get_load_bearing_claims(debate_id, limit)

        # Calibration Summary API
        elif path.startswith('/api/agent/') and path.endswith('/calibration-summary'):
            agent = self._extract_path_segment(path, 3, "agent")
            if agent is None:
                return
            domain = query.get('domain', [None])[0]
            self._get_calibration_summary(agent, domain)

        # Continuum Memory API
        elif path == '/api/memory/continuum/retrieve':
            query_str = query.get('query', [''])[0]
            tiers = query.get('tiers', ['fast,medium'])[0]
            limit = self._safe_int(query, 'limit', 10, 50)
            min_importance = self._safe_float(query, 'min_importance', 0.0, 0.0, 1.0)
            self._get_continuum_memories(query_str, tiers, limit, min_importance)
        elif path == '/api/memory/continuum/consolidate':
            self._get_continuum_consolidation()

        # Formal Verification Status API
        elif path == '/api/verification/status':
            self._formal_verification_status()

        # Plugins API
        elif path == '/api/plugins':
            self._list_plugins()
        elif path.startswith('/api/plugins/') and not path.endswith('/run'):
            plugin_name = path.split('/')[-1]
            if not plugin_name or not re.match(SAFE_ID_PATTERN, plugin_name):
                self._send_json({"error": "Invalid plugin name"}, status=400)
            else:
                self._get_plugin(plugin_name)

        # Genesis API (Evolution Visibility)
        elif path == '/api/genesis/stats':
            self._get_genesis_stats()
        elif path == '/api/genesis/events':
            limit = self._safe_int(query, 'limit', 20, 100)
            event_type = query.get('event_type', [None])[0]
            self._get_genesis_events(limit, event_type)
        elif path.startswith('/api/genesis/lineage/'):
            genome_id = path.split('/')[-1]
            if not genome_id or not re.match(SAFE_ID_PATTERN, genome_id):
                self._send_json({"error": "Invalid genome ID"}, status=400)
            else:
                self._get_genome_lineage(genome_id)
        elif path.startswith('/api/genesis/tree/'):
            debate_id = path.split('/')[-1]
            if not debate_id or not re.match(SAFE_ID_PATTERN, debate_id):
                self._send_json({"error": "Invalid debate ID"}, status=400)
            else:
                self._get_debate_tree(debate_id)

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
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/api/documents/upload':
            self._upload_document()
        elif path == '/api/debate':
            self._start_debate()
        elif path.startswith('/api/debates/') and path.endswith('/broadcast'):
            debate_id = self._extract_path_segment(path, 3, "debate_id")
            if debate_id is None:
                return
            self._generate_broadcast(debate_id)
        elif path == '/api/laboratory/cross-pollinations/suggest':
            self._suggest_cross_pollinations()
        elif path == '/api/routing/recommendations':
            self._get_routing_recommendations()
        elif path == '/api/verification/formal-verify':
            self._formal_verify_claim()
        elif path == '/api/verification/status':
            self._formal_verification_status()
        elif path == '/api/insights/extract-detailed':
            self._extract_detailed_insights()
        elif path == '/api/probes/run':
            self._run_capability_probe()
        elif path == '/api/debates/deep-audit':
            self._run_deep_audit()
        elif path.startswith('/api/debates/') and path.endswith('/red-team'):
            debate_id = self._extract_path_segment(path, 3, "debate_id")
            if debate_id:
                self._run_red_team_analysis(debate_id)
        elif path.startswith('/api/debates/') and path.endswith('/verify'):
            debate_id = self._extract_path_segment(path, 3, "debate_id")
            if debate_id:
                self._verify_debate_outcome(debate_id)
        elif path.startswith('/api/plugins/') and path.endswith('/run'):
            # Pattern: /api/plugins/{name}/run
            parts = path.split('/')
            if len(parts) >= 4:
                plugin_name = parts[3]
                if not re.match(SAFE_ID_PATTERN, plugin_name):
                    self._send_json({"error": "Invalid plugin name"}, status=400)
                else:
                    self._run_plugin(plugin_name)
            else:
                self._send_json({"error": "Invalid path format"}, status=400)
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
                    boundary = part.split('=')[1].strip()
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

        # Set loop_id on emitter so events are tagged
        self.stream_emitter.set_loop_id(debate_id)

        # Start debate in background thread
        def run_debate():
            import asyncio as _asyncio

            try:
                # Parse agents with bounds check
                agent_list = [s.strip() for s in agents_str.split(",") if s.strip()]
                if len(agent_list) > MAX_AGENTS_PER_DEBATE:
                    with _active_debates_lock:
                        _active_debates[debate_id]["status"] = "error"
                        _active_debates[debate_id]["error"] = f"Too many agents. Maximum: {MAX_AGENTS_PER_DEBATE}"
                    return
                if len(agent_list) < 2:
                    with _active_debates_lock:
                        _active_debates[debate_id]["status"] = "error"
                        _active_debates[debate_id]["error"] = "At least 2 agents required for a debate"
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
                for i, (agent_type, role) in enumerate(agent_specs):
                    if role is None:
                        role = "proposer"  # All agents propose and participate fully
                    agent = create_agent(
                        model_type=agent_type,
                        name=f"{agent_type}_{role}",
                        role=role,
                    )
                    # Wrap agent for token streaming if supported
                    agent = _wrap_agent_for_streaming(agent, self.stream_emitter, debate_id)
                    agents.append(agent)

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
                    loop_id=debate_id,
                )

                # Run debate with timeout protection (10 minutes max)
                with _active_debates_lock:
                    _active_debates[debate_id]["status"] = "running"
                async def run_with_timeout():
                    return await _asyncio.wait_for(arena.run(), timeout=600)
                result = _asyncio.run(run_with_timeout())
                with _active_debates_lock:
                    _active_debates[debate_id]["status"] = "completed"
                    _active_debates[debate_id]["result"] = {
                        "final_answer": result.final_answer,
                        "consensus_reached": result.consensus_reached,
                        "confidence": result.confidence,
                    }

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
                    except Exception:
                        pass  # Don't break on leaderboard emission errors

            except Exception as e:
                import traceback
                # Use safe error message for client, keep full trace server-side
                safe_msg = _safe_error_message(e, "debate_execution")
                error_trace = traceback.format_exc()
                with _active_debates_lock:
                    _active_debates[debate_id]["status"] = "error"
                    _active_debates[debate_id]["error"] = safe_msg
                # Log full traceback so thread failures aren't silent
                logger.error(f"[debate] Thread error in {debate_id}: {str(e)}\n{error_trace}")
                # Emit sanitized error event to client
                self.stream_emitter.emit(StreamEvent(
                    type=StreamEventType.ERROR,
                    data={"error": safe_msg, "debate_id": debate_id},
                ))

        # Use thread pool to prevent unbounded thread creation
        # Double-checked locking for thread-safe executor creation
        if UnifiedHandler._debate_executor is None:
            with UnifiedHandler._debate_executor_lock:
                if UnifiedHandler._debate_executor is None:
                    UnifiedHandler._debate_executor = ThreadPoolExecutor(
                        max_workers=UnifiedHandler.MAX_CONCURRENT_DEBATES,
                        thread_name_prefix="debate-"
                    )

        try:
            UnifiedHandler._debate_executor.submit(run_debate)
        except RuntimeError as e:
            # Thread pool full or shut down
            self._send_json({
                "success": False,
                "error": "Server at capacity. Please try again later.",
            }, status=503)
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

    def _get_document(self, doc_id: str) -> None:
        """Get a document by ID."""
        if not self.document_store:
            self.send_error(500, "Document storage not configured")
            return

        doc = self.document_store.get(doc_id)
        if doc:
            self._send_json(doc.to_dict())
        else:
            self.send_error(404, f"Document not found: {doc_id}")

    def _get_debate_by_slug(self, slug: str) -> None:
        """Get debate metadata by human-readable slug for permalinks."""
        if not self.storage:
            self._send_json({"error": "Storage not configured"}, status=500)
            return

        debate = self.storage.get_by_slug(slug)
        if not debate:
            self._send_json({"error": "Debate not found"}, status=404)
            return

        # Return structured metadata for frontend lookup (including consensus metrics)
        self._send_json({
            "slug": slug,
            "debate_id": debate.get("id", slug),
            "task": debate.get("task", ""),
            "agents": debate.get("agents", []),
            "consensus_reached": debate.get("consensus_reached", False),
            "confidence": debate.get("confidence", 0.0),
            "consensus_strength": debate.get("consensus_strength", "none"),
            "consensus_variance": debate.get("consensus_variance"),
            "convergence_status": debate.get("convergence_status"),
            "winner": debate.get("winner"),
            "created_at": debate.get("created_at", ""),
        })

    def _get_debate(self, slug: str) -> None:
        """Get a single debate by slug."""
        if not self.storage:
            self.send_error(500, "Storage not configured")
            return

        debate = self.storage.get_by_slug(slug)
        if debate:
            self._send_json(debate)
        else:
            self.send_error(404, f"Debate not found: {slug}")

    def _list_debates(self, limit: int = 20) -> None:
        """List recent debates."""
        if not self.storage:
            self._send_json([])
            return

        debates = self.storage.list_recent(limit)
        self._send_json([{
            "slug": d.slug,
            "task": d.task[:100] + "..." if len(d.task) > 100 else d.task,
            "agents": d.agents,
            "consensus": d.consensus_reached,
            "confidence": d.confidence,
            "consensus_strength": getattr(d, 'consensus_strength', 'none'),
            "winner": getattr(d, 'winner', None),
            "views": d.view_count,
            "created": d.created_at.isoformat(),
        } for d in debates])

    def _export_debate(self, debate_id: str, export_format: str, table: str = "summary") -> None:
        """Export a debate in the specified format (json, csv, dot, html)."""
        if not self._check_rate_limit():
            return

        if not EXPORT_AVAILABLE:
            self._send_json({"error": "Export module not available"}, status=503)
            return

        # Validate format
        valid_formats = {"json", "csv", "dot", "html"}
        if export_format not in valid_formats:
            self._send_json({"error": f"Invalid format. Use: {', '.join(valid_formats)}"}, status=400)
            return

        # Validate table (for CSV)
        valid_tables = {"summary", "messages", "critiques", "votes", "verifications"}
        if table not in valid_tables:
            table = "summary"

        # Load debate data
        if not self.storage:
            self._send_json({"error": "Storage not configured"}, status=500)
            return

        debate = self.storage.get_by_slug(debate_id)
        if not debate:
            self._send_json({"error": f"Debate not found: {debate_id}"}, status=404)
            return

        try:
            # Build artifact from debate data
            from aragora.export.artifact import ConsensusProof

            artifact = DebateArtifact(
                debate_id=debate.slug,
                task=debate.task,
                agents=debate.agents,
                rounds=getattr(debate, 'rounds', 0),
                message_count=len(debate.messages) if hasattr(debate, 'messages') else 0,
                critique_count=len(debate.critiques) if hasattr(debate, 'critiques') else 0,
                consensus_proof=ConsensusProof(
                    reached=debate.consensus_reached,
                    confidence=debate.confidence,
                    vote_breakdown={v.agent: v.choice == debate.final_answer[:20]
                                    for v in debate.votes} if hasattr(debate, 'votes') else {},
                    final_answer=debate.final_answer,
                    rounds_used=getattr(debate, 'rounds', 0),
                ) if debate.consensus_reached else None,
            )

            # Add trace data if available
            if hasattr(debate, 'messages'):
                artifact.trace_data = {
                    "events": [
                        {
                            "event_type": "message",
                            "agent": msg.agent,
                            "content": msg.content,
                            "round": getattr(msg, 'round', i // len(debate.agents) + 1),
                        }
                        for i, msg in enumerate(debate.messages)
                    ]
                }

            # Export based on format
            if export_format == "json":
                self._send_json(artifact.to_dict())
            elif export_format == "csv":
                exporter = CSVExporter(artifact)
                if table == "messages":
                    content = exporter.export_messages()
                elif table == "critiques":
                    content = exporter.export_critiques()
                elif table == "votes":
                    content = exporter.export_votes()
                elif table == "verifications":
                    content = exporter.export_verifications()
                else:
                    content = exporter.export_summary()
                self._send_text(content, content_type="text/csv")
            elif export_format == "dot":
                exporter = DOTExporter(artifact)
                content = exporter.export_flow()  # Default to flow view
                self._send_text(content, content_type="text/vnd.graphviz")
            elif export_format == "html":
                exporter = StaticHTMLExporter(artifact)
                content = exporter.generate()
                self._send_text(content, content_type="text/html")

        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "export_debate")}, status=500)

    def _send_text(self, content: str, content_type: str = "text/plain") -> None:
        """Send plain text response."""
        data = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _health_check(self) -> None:
        """Health check endpoint."""
        self._send_json({
            "status": "ok",
            "storage": self.storage is not None,
            "streaming": self.stream_emitter is not None,
            "static_dir": str(self.static_dir) if self.static_dir else None,
        })

    def _get_nomic_state(self) -> None:
        """Get current nomic loop state."""
        if not self.nomic_state_file or not self.nomic_state_file.exists():
            self._send_json({"status": "idle", "message": "No active nomic loop"})
            return

        try:
            with open(self.nomic_state_file) as f:
                state = json.load(f)
            self._send_json(state)
        except Exception as e:
            self._send_json({"status": "error", "message": _safe_error_message(e, "nomic_state")})

    def _get_nomic_log(self, lines: int = 100) -> None:
        """Get last N lines of nomic loop log."""
        if not self._check_rate_limit():
            return
        if not self.nomic_state_file:
            self._send_json({"lines": []})
            return

        log_file = self.nomic_state_file.parent / "nomic_loop.log"
        if not log_file.exists():
            self._send_json({"lines": []})
            return

        try:
            # Security: limit file read to prevent memory exhaustion
            MAX_LOG_BYTES = 100 * 1024  # 100KB max
            with open(log_file) as f:
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()
                start_pos = max(0, file_size - MAX_LOG_BYTES)
                f.seek(start_pos)
                if start_pos > 0:
                    f.readline()  # Skip partial line
                all_lines = f.readlines()
            self._send_json({"lines": all_lines[-lines:]})
        except Exception as e:
            logger.error(f"Log read error: {type(e).__name__}: {e}")
            self._send_json({"lines": []})

    def _get_history_cycles(self, loop_id: Optional[str], limit: int) -> None:
        """Get nomic cycles from Supabase."""
        if not self.persistence:
            self._send_json({"error": "Persistence not configured", "cycles": []})
            return

        try:
            cycles = _run_async(
                self.persistence.list_cycles(loop_id=loop_id, limit=limit)
            )
            self._send_json({
                "cycles": [c.to_dict() for c in cycles],
                "count": len(cycles),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "history_cycles"), "cycles": []})

    def _get_history_events(self, loop_id: Optional[str], limit: int) -> None:
        """Get stream events from Supabase."""
        if not self.persistence:
            self._send_json({"error": "Persistence not configured", "events": []})
            return

        if not loop_id:
            self._send_json({"error": "loop_id required", "events": []})
            return

        try:
            events = _run_async(
                self.persistence.get_events(loop_id=loop_id, limit=limit)
            )
            self._send_json({
                "events": [e.to_dict() for e in events],
                "count": len(events),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "history_events"), "events": []})

    def _get_history_debates(self, loop_id: Optional[str], limit: int) -> None:
        """Get debate artifacts from Supabase."""
        if not self.persistence:
            self._send_json({"error": "Persistence not configured", "debates": []})
            return

        try:
            debates = _run_async(
                self.persistence.list_debates(loop_id=loop_id, limit=limit)
            )
            self._send_json({
                "debates": [d.to_dict() for d in debates],
                "count": len(debates),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "history_debates"), "debates": []})

    def _get_history_summary(self, loop_id: Optional[str]) -> None:
        """Get summary statistics for a loop."""
        if not self.persistence:
            self._send_json({"error": "Persistence not configured"})
            return

        if not loop_id:
            self._send_json({"error": "loop_id required"})
            return

        try:
            summary = _run_async(
                self.persistence.get_loop_summary(loop_id)
            )
            self._send_json(summary)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "history_summary")})

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

    def _get_leaderboard(self, limit: int, domain: Optional[str]) -> None:
        """Get agent leaderboard by ELO ranking (debate consensus feature)."""
        if not self.elo_system:
            self._send_json({"error": "Rankings not configured", "agents": []})
            return

        try:
            agents = self.elo_system.get_leaderboard(limit=limit, domain=domain)
            self._send_json({
                "agents": [
                    {
                        "name": a.agent_name,
                        "elo": round(a.elo),
                        "wins": a.wins,
                        "losses": a.losses,
                        "draws": a.draws,
                        "win_rate": round(a.win_rate * 100, 1),
                        "games": a.games_played,
                    }
                    for a in agents
                ],
                "count": len(agents),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "agents"), "agents": []})

    def _get_recent_matches(self, limit: int, loop_id: Optional[str] = None) -> None:
        """Get recent match results (debate consensus feature).

        Args:
            limit: Maximum number of matches to return
            loop_id: Optional loop ID to filter matches by (multi-loop support)
        """
        if not self.elo_system:
            self._send_json({"error": "Rankings not configured", "matches": []})
            return

        try:
            # Use EloSystem's encapsulated method instead of raw SQL
            matches = self.elo_system.get_recent_matches(limit=limit)
            # Filter by loop_id if provided (multi-loop support)
            if loop_id:
                matches = [m for m in matches if m.get('loop_id') == loop_id]
            self._send_json({"matches": matches, "count": len(matches)})
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "matches"), "matches": []})

    def _get_agent_history(self, agent: str, limit: int) -> None:
        """Get ELO history for an agent (debate consensus feature)."""
        if not self.elo_system:
            self._send_json({"error": "Rankings not configured", "history": []})
            return

        try:
            history = self.elo_system.get_elo_history(agent, limit=limit)
            self._send_json({
                "agent": agent,
                "history": [{"debate_id": h[0], "elo": h[1]} for h in history],
                "count": len(history),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "elo_history"), "history": []})

    def _get_calibration_leaderboard(self, limit: int) -> None:
        """Get agents ranked by calibration score (accuracy vs confidence)."""
        if not self.elo_system:
            self._send_json({"error": "Rankings not configured", "agents": []})
            return

        try:
            agents = self.elo_system.get_calibration_leaderboard(limit=limit)
            self._send_json({
                "agents": [
                    {
                        "name": a.agent_name,
                        "elo": round(a.elo),
                        "calibration_score": round(a.calibration_score, 3),
                        "brier_score": round(a.calibration_brier_score, 3),
                        "accuracy": round(a.calibration_accuracy, 3),
                        "games": a.games_played,
                    }
                    for a in agents
                ],
                "count": len(agents),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "agents"), "agents": []})

    def _get_agent_calibration(self, agent: str, domain: Optional[str] = None) -> None:
        """Get detailed calibration metrics for an agent."""
        if not self.elo_system:
            self._send_json({"error": "Rankings not configured"})
            return

        try:
            # Get ECE (Expected Calibration Error)
            ece = self.elo_system.get_expected_calibration_error(agent)

            # Get confidence buckets
            buckets = self.elo_system.get_calibration_by_bucket(agent, domain)

            # Get domain-specific calibration if available
            domain_calibration = self.elo_system.get_domain_calibration(agent, domain)

            self._send_json({
                "agent": agent,
                "ece": round(ece, 3),
                "buckets": buckets,
                "domain_calibration": domain_calibration,
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "persona_update")})

    def _get_trending_topics(self, limit: int) -> None:
        """Get trending topics from pulse ingestors."""
        if not PULSE_AVAILABLE:
            self._send_json({"error": "Pulse ingestor not available", "topics": []}, status=503)
            return

        try:
            # Create manager with default ingestors
            manager = PulseManager()
            manager.add_ingestor("twitter", TwitterIngestor())

            # Fetch trending topics asynchronously
            topics = _run_async(manager.get_trending_topics(limit_per_platform=limit))

            self._send_json({
                "topics": [
                    {
                        "topic": t.topic,
                        "platform": t.platform,
                        "volume": t.volume,
                        "category": t.category,
                    }
                    for t in topics
                ],
                "count": len(topics),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "trending_topics"), "topics": []})

    def _generate_broadcast(self, debate_id: str) -> None:
        """Generate podcast audio from a debate trace.

        POST /api/debates/:id/broadcast

        Rate limited. Returns path to generated MP3 or error.
        """
        if not self._check_rate_limit():
            return

        if not BROADCAST_AVAILABLE:
            self._send_json({"error": "Broadcast module not available"}, status=503)
            return

        if not self.storage:
            self._send_json({"error": "Storage not configured"}, status=500)
            return

        try:
            # Load debate from storage
            debate_data = self.storage.get_by_slug(debate_id) or self.storage.get_by_id(debate_id)
            if not debate_data:
                self._send_json({"error": "Debate not found"}, status=404)
                return

            # Convert to DebateTrace format
            # Try to extract trace data from artifact_json
            trace_data = debate_data.get("trace") or debate_data
            trace = DebateTrace(
                trace_id=debate_id,
                debate_id=debate_data.get("id", debate_id),
                task=debate_data.get("task", ""),
                agents=debate_data.get("agents", []),
                random_seed=debate_data.get("random_seed", 0),
                events=[],  # Events will be extracted from messages
                metadata={"source": "storage"},
            )

            # Extract messages as trace events if available
            messages = debate_data.get("messages", [])
            if messages:
                from aragora.debate.traces import TraceEvent, EventType
                for i, msg in enumerate(messages):
                    trace.events.append(TraceEvent(
                        event_type=EventType.MESSAGE,
                        timestamp=msg.get("timestamp", ""),
                        agent=msg.get("agent", "unknown"),
                        round_num=msg.get("round", i // 3),
                        data={"content": msg.get("content", "")},
                    ))

            # Generate broadcast asynchronously
            from pathlib import Path
            output_path = _run_async(broadcast_debate(trace))

            if output_path:
                self._send_json({
                    "success": True,
                    "debate_id": debate_id,
                    "audio_path": str(output_path),
                    "format": "mp3",
                })
            else:
                self._send_json({"error": "Failed to generate audio"}, status=500)

        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "broadcast_generation")}, status=500)

    def _list_replays(self) -> None:
        """List available replay directories."""
        if not self.nomic_state_file:
            self._send_json([])
            return

        try:
            replays_dir = self.nomic_state_file.parent / "replays"
            if not replays_dir.exists():
                self._send_json([])
                return

            replays = []
            for replay_path in replays_dir.iterdir():
                if replay_path.is_dir():
                    meta_file = replay_path / "meta.json"
                    if meta_file.exists():
                        meta = json.loads(meta_file.read_text())
                        replays.append({
                            "id": replay_path.name,
                            "topic": meta.get("topic", replay_path.name),
                            "agents": [a.get("name") for a in meta.get("agents", [])],
                            "schema_version": meta.get("schema_version", "1.0"),
                        })
            self._send_json(sorted(replays, key=lambda x: x["id"], reverse=True))
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "list_replays")})

    def _get_replay(self, replay_id: str) -> None:
        """Get a specific replay with events."""
        if not self.nomic_state_file:
            self._send_json({"error": "Replays not configured"})
            return

        # Validate replay_id to prevent path traversal
        if not re.match(SAFE_ID_PATTERN, replay_id):
            self._send_json({"error": "Invalid replay ID format"}, status=400)
            return

        try:
            replay_dir = self.nomic_state_file.parent / "replays" / replay_id
            if not replay_dir.exists():
                self._send_json({"error": f"Replay not found: {replay_id}"})
                return

            # Load meta
            meta_file = replay_dir / "meta.json"
            meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}

            # Load events
            events_file = replay_dir / "events.jsonl"
            events = []
            if events_file.exists():
                for line in events_file.read_text().strip().split("\n"):
                    if line:
                        events.append(json.loads(line))

            self._send_json({
                "id": replay_id,
                "meta": meta,
                "events": events,
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "get_replay")})

    def _get_learning_evolution(self) -> None:
        """Get learning/evolution data from meta_learning.db."""
        if not self.nomic_state_file:
            self._send_json({"error": "Learning data not configured", "patterns": []})
            return

        try:
            import sqlite3
            db_path = self.nomic_state_file.parent / "meta_learning.db"
            if not db_path.exists():
                self._send_json({"patterns": [], "count": 0})
                return

            with sqlite3.connect(str(db_path), timeout=30.0) as conn:
                conn.row_factory = sqlite3.Row

                # Get recent patterns
                cursor = conn.execute("""
                    SELECT * FROM meta_patterns
                    ORDER BY created_at DESC
                    LIMIT 20
                """)
                patterns = [dict(row) for row in cursor.fetchall()]

            self._send_json({
                "patterns": patterns,
                "count": len(patterns),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "patterns"), "patterns": []})

    def _get_recent_flips(self, limit: int) -> None:
        """Get recent position flips across all agents."""
        if not self.flip_detector:
            self._send_json({"error": "Flip detection not configured", "flips": []})
            return

        try:
            flips = self.flip_detector.get_recent_flips(limit=limit)
            self._send_json({
                "flips": [format_flip_for_ui(f) for f in flips],
                "count": len(flips),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "flips"), "flips": []})

    def _get_flip_summary(self) -> None:
        """Get summary of all flips for dashboard display."""
        if not self.flip_detector:
            self._send_json({"error": "Flip detection not configured", "summary": {}})
            return

        try:
            summary = self.flip_detector.get_flip_summary()
            self._send_json({"summary": summary})
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "flip_summary"), "summary": {}})

    def _get_agent_consistency(self, agent: str) -> None:
        """Get consistency score for an agent."""
        if not self.flip_detector:
            self._send_json({"error": "Flip detection not configured", "consistency": {}})
            return

        try:
            score = self.flip_detector.get_agent_consistency(agent)
            self._send_json({"consistency": format_consistency_for_ui(score)})
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "agent_consistency"), "consistency": {}})

    def _get_agent_flips(self, agent: str, limit: int) -> None:
        """Get flips for a specific agent."""
        if not self.flip_detector:
            self._send_json({"error": "Flip detection not configured", "flips": []})
            return

        try:
            flips = self.flip_detector.detect_flips_for_agent(agent, lookback_positions=limit)
            self._send_json({
                "agent": agent,
                "flips": [format_flip_for_ui(f) for f in flips],
                "count": len(flips),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "flips"), "flips": []})

    def _get_all_personas(self) -> None:
        """Get all agent personas."""
        if not self.persona_manager:
            self._send_json({"error": "Persona management not configured", "personas": []})
            return

        try:
            personas = self.persona_manager.get_all_personas()
            self._send_json({
                "personas": [
                    {
                        "agent_name": p.agent_name,
                        "description": p.description,
                        "traits": p.traits,
                        "expertise": p.expertise,
                        "created_at": p.created_at,
                        "updated_at": p.updated_at,
                    }
                    for p in personas
                ],
                "count": len(personas),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "personas"), "personas": []})

    def _get_agent_persona(self, agent: str) -> None:
        """Get persona for a specific agent."""
        if not self.persona_manager:
            self._send_json({"error": "Persona management not configured"})
            return

        try:
            persona = self.persona_manager.get_persona(agent)
            if persona:
                self._send_json({
                    "persona": {
                        "agent_name": persona.agent_name,
                        "description": persona.description,
                        "traits": persona.traits,
                        "expertise": persona.expertise,
                        "created_at": persona.created_at,
                        "updated_at": persona.updated_at,
                    }
                })
            else:
                self._send_json({"error": f"No persona found for agent '{agent}'", "persona": None})
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "get_persona")})

    def _get_agent_performance(self, agent: str) -> None:
        """Get performance summary for an agent."""
        if not self.persona_manager:
            self._send_json({"error": "Persona management not configured"})
            return

        try:
            summary = self.persona_manager.get_performance_summary(agent)
            self._send_json({
                "agent": agent,
                "performance": summary,
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "agent_performance")})

    def _get_agent_domains(self, agent: str, limit: int) -> None:
        """Get agent's best expertise domains by calibration."""
        if not self._check_rate_limit():
            return

        if not RANKING_AVAILABLE or not self.elo_system:
            self._send_json({"error": "Ranking system not available"}, status=503)
            return

        try:
            domains = self.elo_system.get_best_domains(agent, limit=limit)
            self._send_json({
                "agent": agent,
                "domains": [
                    {"domain": d[0], "calibration_score": d[1]}
                    for d in domains
                ],
                "count": len(domains),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "agent_domains")}, status=500)

    def _get_grounded_persona(self, agent: str) -> None:
        """Get truth-grounded persona synthesized from performance data."""
        if not self._check_rate_limit():
            return

        try:
            from aragora.agents.grounded import PersonaSynthesizer

            db_path = self.nomic_dir / "aragora_personas.db" if self.nomic_dir else None
            synthesizer = PersonaSynthesizer(
                persona_manager=self.persona_manager,
                elo_system=self.elo_system,
                position_ledger=getattr(self, 'position_ledger', None),
                relationship_tracker=None,
            )
            persona = synthesizer.get_grounded_persona(agent)
            if persona:
                self._send_json({
                    "agent": agent,
                    "elo": persona.elo,
                    "domain_elos": persona.domain_elos,
                    "games_played": persona.games_played,
                    "win_rate": persona.win_rate,
                    "calibration_score": persona.calibration_score,
                    "position_accuracy": persona.position_accuracy,
                    "positions_taken": persona.positions_taken,
                    "reversals": persona.reversals,
                })
            else:
                self._send_json({"agent": agent, "message": "No grounded persona data"})
        except ImportError:
            self._send_json({"error": "Grounded personas module not available"}, status=503)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "grounded_persona")}, status=500)

    def _get_identity_prompt(self, agent: str, sections: str = None) -> None:
        """Get evidence-grounded identity prompt for agent initialization."""
        if not self._check_rate_limit():
            return

        try:
            from aragora.agents.grounded import PersonaSynthesizer

            synthesizer = PersonaSynthesizer(
                persona_manager=self.persona_manager,
                elo_system=self.elo_system,
                position_ledger=getattr(self, 'position_ledger', None),
                relationship_tracker=None,
            )
            include_sections = sections.split(',') if sections else None
            prompt = synthesizer.synthesize_identity_prompt(agent, include_sections=include_sections)
            self._send_json({
                "agent": agent,
                "identity_prompt": prompt,
                "sections": include_sections,
            })
        except ImportError:
            self._send_json({"error": "Grounded personas module not available"}, status=503)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "identity_prompt")}, status=500)

    def _get_agent_accuracy(self, agent: str) -> None:
        """Get position accuracy stats for an agent from PositionTracker."""
        if not self._check_rate_limit():
            return

        try:
            from aragora.agents.truth_grounding import PositionTracker

            # Use existing position_tracker or create one
            if hasattr(self, 'position_tracker') and self.position_tracker:
                tracker = self.position_tracker
            else:
                db_path = self.nomic_dir / "aragora_positions.db" if self.nomic_dir else None
                if not db_path or not db_path.exists():
                    self._send_json({"error": "Position tracking not configured"}, status=503)
                    return
                tracker = PositionTracker(db_path=str(db_path))

            accuracy = tracker.get_agent_position_accuracy(agent)
            if accuracy:
                self._send_json({
                    "agent": agent,
                    "total_positions": accuracy.get("total_positions", 0),
                    "verified_positions": accuracy.get("verified_positions", 0),
                    "correct_positions": accuracy.get("correct_positions", 0),
                    "accuracy_rate": accuracy.get("accuracy_rate", 0.0),
                    "by_type": accuracy.get("by_type", {}),
                })
            else:
                self._send_json({
                    "agent": agent,
                    "total_positions": 0,
                    "verified_positions": 0,
                    "accuracy_rate": 0.0,
                    "message": "No position accuracy data available",
                })
        except ImportError:
            self._send_json({"error": "PositionTracker module not available"}, status=503)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "agent_accuracy")}, status=500)

    # === Consensus Memory API ===

    def _get_similar_debates(self, topic: str, limit: int) -> None:
        """Find debates similar to a topic."""
        if not CONSENSUS_MEMORY_AVAILABLE:
            self._send_json({"error": "Consensus memory not available"}, status=503)
            return

        if not topic:
            self._send_json({"error": "topic parameter required"}, status=400)
            return

        try:
            memory = ConsensusMemory()
            similar = memory.find_similar_debates(topic, limit=limit)
            self._send_json({
                "query": topic,
                "similar": [
                    {
                        "topic": s.consensus.topic,
                        "conclusion": s.consensus.conclusion,
                        "strength": s.consensus.strength.value,
                        "confidence": s.consensus.confidence,
                        "similarity": s.similarity_score,
                        "agents": s.consensus.participating_agents,
                        "dissent_count": len(s.dissents),
                        "timestamp": s.consensus.timestamp.isoformat(),
                    }
                    for s in similar
                ],
                "count": len(similar),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "similar_topics")}, status=500)

    def _get_settled_topics(self, min_confidence: float, limit: int) -> None:
        """Get high-confidence settled topics."""
        if not CONSENSUS_MEMORY_AVAILABLE:
            self._send_json({"error": "Consensus memory not available"}, status=503)
            return

        try:
            memory = ConsensusMemory()
            # Query for high-confidence topics
            import sqlite3
            with sqlite3.connect(memory.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT topic, conclusion, confidence, strength, timestamp
                    FROM consensus
                    WHERE confidence >= ?
                    ORDER BY confidence DESC, timestamp DESC
                    LIMIT ?
                """, (min_confidence, limit))
                rows = cursor.fetchall()

            self._send_json({
                "min_confidence": min_confidence,
                "topics": [
                    {
                        "topic": row[0],
                        "conclusion": row[1],
                        "confidence": row[2],
                        "strength": row[3],
                        "timestamp": row[4],
                    }
                    for row in rows
                ],
                "count": len(rows),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "settled_topics")}, status=500)

    def _get_consensus_stats(self) -> None:
        """Get consensus memory statistics."""
        if not CONSENSUS_MEMORY_AVAILABLE:
            self._send_json({"error": "Consensus memory not available"}, status=503)
            return

        try:
            memory = ConsensusMemory()
            raw_stats = memory.get_statistics()

            # Transform to match frontend ConsensusStats interface
            # Count high confidence topics (>= 0.7)
            import sqlite3
            with sqlite3.connect(memory.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM consensus WHERE confidence >= 0.7")
                high_confidence_count = cursor.fetchone()[0]
                cursor.execute("SELECT AVG(confidence) FROM consensus")
                avg_row = cursor.fetchone()
                avg_confidence = avg_row[0] if avg_row[0] else 0.0

            self._send_json({
                "total_topics": raw_stats.get("total_consensus", 0),
                "high_confidence_count": high_confidence_count,
                "domains": list(raw_stats.get("by_domain", {}).keys()),
                "avg_confidence": round(avg_confidence, 3),
                # Include original stats for backwards compatibility
                "total_dissents": raw_stats.get("total_dissents", 0),
                "by_strength": raw_stats.get("by_strength", {}),
                "by_domain": raw_stats.get("by_domain", {}),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "consensus_stats")}, status=500)

    def _get_dissents_for_topic(self, topic: str, domain: Optional[str] = None) -> None:
        """Get dissenting views relevant to a topic."""
        if not CONSENSUS_MEMORY_AVAILABLE or DissentRetriever is None:
            self._send_json({"error": "Dissent retriever not available"}, status=503)
            return

        try:
            memory = ConsensusMemory()
            retriever = DissentRetriever(memory)
            context = retriever.retrieve_for_new_debate(topic, domain=domain)
            self._send_json({
                "topic": topic,
                "domain": domain,
                "similar_debates": context.get("similar_debates", []),
                "dissents_by_type": context.get("dissent_by_type", {}),
                "unacknowledged_dissents": len(context.get("unacknowledged", [])),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "dissent_retrieval")}, status=500)

    def _get_recent_dissents(self, topic: Optional[str], domain: Optional[str], limit: int) -> None:
        """Get recent dissents, optionally filtered by topic."""
        if not CONSENSUS_MEMORY_AVAILABLE:
            self._send_json({"error": "Consensus memory not available"}, status=503)
            return

        try:
            memory = ConsensusMemory()
            import sqlite3
            import json

            # Query recent dissents with their associated consensus topics
            with sqlite3.connect(memory.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                # Join dissent with consensus to get topic and majority view
                query = """
                    SELECT d.data, c.topic, c.conclusion
                    FROM dissent d
                    LEFT JOIN consensus c ON d.debate_id = c.id
                    ORDER BY d.timestamp DESC
                    LIMIT ?
                """
                cursor.execute(query, (limit,))
                rows = cursor.fetchall()

            # Transform to match frontend DissentView interface
            dissents = []
            for row in rows:
                try:
                    from aragora.memory.consensus import DissentRecord
                    record = DissentRecord.from_dict(json.loads(row[0]))
                    topic_name = row[1] or "Unknown topic"
                    majority_view = row[2] or "No consensus recorded"

                    dissents.append({
                        "topic": topic_name,
                        "majority_view": majority_view,
                        "dissenting_view": record.content,
                        "dissenting_agent": record.agent_id,
                        "confidence": record.confidence,
                        "reasoning": record.reasoning if record.reasoning else None,
                    })
                except Exception:
                    pass

            self._send_json({"dissents": dissents})
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "recent_dissents")}, status=500)

    def _get_contrarian_views(self, topic: Optional[str], domain: Optional[str], limit: int) -> None:
        """Get historical contrarian/dissenting views, optionally filtered by topic."""
        if not CONSENSUS_MEMORY_AVAILABLE:
            self._send_json({"error": "Consensus memory not available"}, status=503)
            return

        try:
            memory = ConsensusMemory()
            import sqlite3

            # If topic provided, use DissentRetriever for similarity search
            if topic and DissentRetriever is not None:
                retriever = DissentRetriever(memory)
                records = retriever.find_contrarian_views(topic, domain=domain, limit=limit)
            else:
                # Get recent contrarian views globally (FUNDAMENTAL_DISAGREEMENT, ALTERNATIVE_APPROACH)
                with sqlite3.connect(memory.db_path, timeout=30.0) as conn:
                    cursor = conn.cursor()
                    query = """
                        SELECT data FROM dissent
                        WHERE dissent_type IN ('fundamental_disagreement', 'alternative_approach')
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """
                    cursor.execute(query, (limit,))
                    rows = cursor.fetchall()

                import json
                records = []
                for row in rows:
                    try:
                        from aragora.memory.consensus import DissentRecord
                        records.append(DissentRecord.from_dict(json.loads(row[0])))
                    except Exception:
                        pass

            # Transform to match frontend ContraryView interface
            self._send_json({
                "views": [
                    {
                        "agent": r.agent_id,
                        "position": r.content,
                        "confidence": r.confidence,
                        "reasoning": r.reasoning,
                        "debate_id": r.debate_id,
                    }
                    for r in records
                ],
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "contrarian_views")}, status=500)

    def _get_risk_warnings(self, topic: Optional[str], domain: Optional[str], limit: int) -> None:
        """Get risk warnings and edge case concerns, optionally filtered by topic."""
        if not CONSENSUS_MEMORY_AVAILABLE:
            self._send_json({"error": "Consensus memory not available"}, status=503)
            return

        try:
            memory = ConsensusMemory()
            import sqlite3

            # If topic provided, use DissentRetriever for similarity search
            if topic and DissentRetriever is not None:
                retriever = DissentRetriever(memory)
                records = retriever.find_risk_warnings(topic, domain=domain, limit=limit)
            else:
                # Get recent risk warnings globally (RISK_WARNING, EDGE_CASE_CONCERN)
                with sqlite3.connect(memory.db_path, timeout=30.0) as conn:
                    cursor = conn.cursor()
                    query = """
                        SELECT data FROM dissent
                        WHERE dissent_type IN ('risk_warning', 'edge_case_concern')
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """
                    cursor.execute(query, (limit,))
                    rows = cursor.fetchall()

                import json
                records = []
                for row in rows:
                    try:
                        from aragora.memory.consensus import DissentRecord
                        records.append(DissentRecord.from_dict(json.loads(row[0])))
                    except Exception:
                        pass

            # Transform to match frontend RiskWarning interface
            # Map dissent_type to risk_type and infer severity from confidence
            def infer_severity(confidence: float, dissent_type: str) -> str:
                if dissent_type == "risk_warning":
                    if confidence >= 0.8:
                        return "critical"
                    elif confidence >= 0.6:
                        return "high"
                    elif confidence >= 0.4:
                        return "medium"
                return "low"

            self._send_json({
                "warnings": [
                    {
                        "domain": r.metadata.get("domain", "general"),
                        "risk_type": r.dissent_type.value.replace("_", " ").title(),
                        "severity": infer_severity(r.confidence, r.dissent_type.value),
                        "description": r.content,
                        "mitigation": r.rebuttal if r.rebuttal else None,
                        "detected_at": r.timestamp.isoformat(),
                    }
                    for r in records
                ],
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "risk_warnings")}, status=500)

    def _get_domain_history(self, domain: str, limit: int) -> None:
        """Get consensus history for a domain."""
        if not CONSENSUS_MEMORY_AVAILABLE:
            self._send_json({"error": "Consensus memory not available"}, status=503)
            return

        try:
            memory = ConsensusMemory()
            records = memory.get_domain_consensus_history(domain, limit=limit)
            self._send_json({
                "domain": domain,
                "history": [r.to_dict() for r in records],
                "count": len(records),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "domain_history")}, status=500)

    def _get_agent_full_profile(self, agent: str) -> None:
        """Get combined profile for an agent (ELO + Persona + Flips + Calibration)."""
        profile = {"agent": agent}

        # ELO ranking
        if self.elo_system:
            try:
                rating = self.elo_system.get_rating(agent)
                history = self.elo_system.get_match_history(agent, limit=10)
                profile["ranking"] = {
                    "rating": rating,
                    "recent_matches": len(history),
                }
            except Exception:
                profile["ranking"] = None
        else:
            profile["ranking"] = None

        # Persona
        if self.persona_manager:
            try:
                persona = self.persona_manager.get_persona(agent)
                if persona:
                    profile["persona"] = {
                        "type": persona.persona_type.value,
                        "primary_stance": persona.primary_stance,
                        "specializations": persona.specializations[:3],
                        "debate_count": persona.debate_count,
                    }
                else:
                    profile["persona"] = None
            except Exception:
                profile["persona"] = None
        else:
            profile["persona"] = None

        # Flip/consistency
        if self.flip_detector:
            try:
                consistency = self.flip_detector.get_consistency_score(agent)
                flips = self.flip_detector.get_agent_flips(agent, limit=5)
                profile["consistency"] = {
                    "score": consistency,
                    "recent_flips": len(flips),
                }
            except Exception:
                profile["consistency"] = None
        else:
            profile["consistency"] = None

        # Calibration
        if CALIBRATION_AVAILABLE:
            try:
                tracker = CalibrationTracker()
                cal = tracker.get_agent_calibration(agent)
                if cal:
                    profile["calibration"] = {
                        "brier_score": cal.get("brier_score"),
                        "prediction_count": cal.get("prediction_count", 0),
                    }
                else:
                    profile["calibration"] = None
            except Exception:
                profile["calibration"] = None
        else:
            profile["calibration"] = None

        self._send_json(profile)

    def _get_disagreement_report(self, limit: int) -> None:
        """Get report of debates with significant disagreements."""
        # Rate limit analytics queries
        if not self._check_rate_limit():
            return
        if not self.storage:
            self._send_json({"error": "Storage not configured", "disagreements": []})
            return

        try:
            debates = self.storage.list_debates(limit=limit * 2)  # Fetch extra to filter
            disagreements = []
            for debate in debates:
                # Check for low consensus or explicit dissent
                if not debate.get("consensus_reached", True) or debate.get("dissent_count", 0) > 0:
                    disagreements.append({
                        "debate_id": debate.get("id", debate.get("slug", "")),
                        "topic": debate.get("task", debate.get("topic", "")),
                        "agents": debate.get("agents", []),
                        "dissent_count": debate.get("dissent_count", 0),
                        "consensus_reached": debate.get("consensus_reached", False),
                        "confidence": debate.get("confidence", 0.0),
                        "timestamp": debate.get("created_at", ""),
                    })
                if len(disagreements) >= limit:
                    break
            self._send_json({"disagreements": disagreements, "count": len(disagreements)})
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "disagreement_report"), "disagreements": []})

    def _get_role_rotation_report(self, limit: int) -> None:
        """Get report of agent role assignments across debates."""
        # Rate limit analytics queries
        if not self._check_rate_limit():
            return
        if not self.storage:
            self._send_json({"error": "Storage not configured", "rotations": []})
            return

        try:
            debates = self.storage.list_debates(limit=limit)
            role_counts: dict = {}  # agent -> role -> count
            rotations = []
            for debate in debates:
                agents = debate.get("agents", [])
                roles = debate.get("roles", {})  # {agent: role} if available
                for agent in agents:
                    role = roles.get(agent, "participant")
                    if agent not in role_counts:
                        role_counts[agent] = {}
                    role_counts[agent][role] = role_counts[agent].get(role, 0) + 1
                    rotations.append({
                        "debate_id": debate.get("id", debate.get("slug", "")),
                        "agent": agent,
                        "role": role,
                        "timestamp": debate.get("created_at", ""),
                    })
            self._send_json({
                "rotations": rotations[:limit],
                "summary": role_counts,
                "count": len(rotations),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "role_rotation"), "rotations": []})

    def _get_early_stop_signals(self, limit: int) -> None:
        """Get debates that were terminated early (before all rounds completed)."""
        # Rate limit analytics queries
        if not self._check_rate_limit():
            return
        if not self.storage:
            self._send_json({"error": "Storage not configured", "early_stops": []})
            return

        try:
            debates = self.storage.list_debates(limit=limit * 2)
            early_stops = []
            for debate in debates:
                rounds_completed = debate.get("rounds_completed", 0)
                rounds_planned = debate.get("rounds_planned", debate.get("rounds", 3))
                early_stopped = debate.get("early_stopped", False)
                # Detect early termination
                if early_stopped or (rounds_planned > 0 and rounds_completed < rounds_planned):
                    early_stops.append({
                        "debate_id": debate.get("id", debate.get("slug", "")),
                        "topic": debate.get("task", debate.get("topic", "")),
                        "rounds_completed": rounds_completed,
                        "rounds_planned": rounds_planned,
                        "reason": debate.get("stop_reason", "unknown"),
                        "consensus_early": debate.get("consensus_reached", False),
                        "timestamp": debate.get("created_at", ""),
                    })
                if len(early_stops) >= limit:
                    break
            self._send_json({"early_stops": early_stops, "count": len(early_stops)})
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "early_stops"), "early_stops": []})

    def _list_available_modes(self) -> None:
        """List available debate and operational modes."""
        try:
            from aragora.modes import ModeRegistry
            from aragora.modes.builtin import register_all_builtins

            # Ensure builtins are registered
            register_all_builtins()

            modes = []
            for name, mode_cls in ModeRegistry.list_modes().items():
                mode_info = {
                    "name": name,
                    "description": (getattr(mode_cls, '__doc__', '') or '').strip().split('\n')[0],
                    "category": "operational",
                }
                # Add tool groups if available
                if hasattr(mode_cls, 'tool_groups'):
                    mode_info["tool_groups"] = [g.value for g in mode_cls.tool_groups]
                modes.append(mode_info)

            # Add debate modes
            debate_modes = [
                {"name": "redteam", "description": "Adversarial red-teaming for security analysis", "category": "debate"},
                {"name": "deep_audit", "description": "Heavy3-inspired intensive debate protocol", "category": "debate"},
                {"name": "capability_probe", "description": "Agent capability and vulnerability probing", "category": "debate"},
            ]
            modes.extend(debate_modes)

            self._send_json({"modes": modes, "count": len(modes)})
        except ImportError:
            # Modes module not available, return empty list
            self._send_json({"modes": [], "count": 0, "warning": "Modes module not available"})
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "modes"), "modes": []})

    def _get_agent_positions(self, agent: str, limit: int) -> None:
        """Get position history for an agent from truth grounding system."""
        if not self._check_rate_limit():
            return

        try:
            from aragora.agents.truth_grounding import PositionLedger

            # Try to use existing position ledger or create one
            if hasattr(self, 'position_ledger') and self.position_ledger:
                ledger = self.position_ledger
            else:
                # Create a temporary ledger instance
                db_path = self.nomic_dir / "aragora_personas.db" if self.nomic_dir else None
                if not db_path or not db_path.exists():
                    self._send_json({"error": "Position tracking not configured"}, status=503)
                    return
                ledger = PositionLedger(db_path=str(db_path))

            # Get agent stats
            stats = ledger.get_agent_stats(agent)
            if stats:
                self._send_json({
                    "agent": agent,
                    "total_positions": stats.get("total_positions", 0),
                    "avg_confidence": stats.get("avg_confidence", 0.0),
                    "reversal_count": stats.get("reversal_count", 0),
                    "consistency_score": stats.get("consistency_score", 1.0),
                    "positions_by_debate": stats.get("positions_by_debate", {}),
                })
            else:
                self._send_json({"agent": agent, "total_positions": 0, "message": "No position data"})

        except ImportError:
            self._send_json({"error": "Position tracking module not available"}, status=503)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "agent_positions")}, status=500)

    def _get_agent_network(self, agent: str) -> None:
        """Get complete influence/relationship network for an agent."""
        if not self._check_rate_limit():
            return

        if not RELATIONSHIP_TRACKER_AVAILABLE:
            self._send_json({"error": "Relationship tracking not available"}, status=503)
            return

        try:
            db_path = self.nomic_dir / "grounded_positions.db" if self.nomic_dir else None
            if not db_path or not db_path.exists():
                self._send_json({"agent": agent, "message": "No relationship data"})
                return

            tracker = RelationshipTracker(str(db_path))
            network = tracker.get_influence_network(agent)
            rivals = tracker.get_rivals(agent, limit=5)
            allies = tracker.get_allies(agent, limit=5)

            self._send_json({
                "agent": agent,
                "influences": network.get("influences", []),
                "influenced_by": network.get("influenced_by", []),
                "rivals": rivals,
                "allies": allies,
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "agent_network")}, status=500)

    def _get_agent_rivals(self, agent: str, limit: int) -> None:
        """Get top rivals for an agent."""
        if not self._check_rate_limit():
            return

        if not RELATIONSHIP_TRACKER_AVAILABLE:
            self._send_json({"error": "Relationship tracking not available"}, status=503)
            return

        try:
            db_path = self.nomic_dir / "grounded_positions.db" if self.nomic_dir else None
            if not db_path or not db_path.exists():
                self._send_json({"agent": agent, "rivals": []})
                return

            tracker = RelationshipTracker(str(db_path))
            rivals = tracker.get_rivals(agent, limit=limit)
            self._send_json({"agent": agent, "rivals": rivals, "count": len(rivals)})
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "agent_rivals")}, status=500)

    def _get_agent_allies(self, agent: str, limit: int) -> None:
        """Get top allies for an agent."""
        if not self._check_rate_limit():
            return

        if not RELATIONSHIP_TRACKER_AVAILABLE:
            self._send_json({"error": "Relationship tracking not available"}, status=503)
            return

        try:
            db_path = self.nomic_dir / "grounded_positions.db" if self.nomic_dir else None
            if not db_path or not db_path.exists():
                self._send_json({"agent": agent, "allies": []})
                return

            tracker = RelationshipTracker(str(db_path))
            allies = tracker.get_allies(agent, limit=limit)
            self._send_json({"agent": agent, "allies": allies, "count": len(allies)})
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "agent_allies")}, status=500)

    def _get_agent_moments(self, agent: str, limit: int) -> None:
        """Get significant moments timeline for an agent."""
        if not self._check_rate_limit():
            return

        if not MOMENT_DETECTOR_AVAILABLE:
            self._send_json({"error": "MomentDetector not available"}, status=503)
            return

        try:
            db_path = self.nomic_dir / "grounded_positions.db" if self.nomic_dir else None
            if not db_path or not db_path.exists():
                self._send_json({"agent": agent, "moments": [], "narrative": ""})
                return

            detector = MomentDetector(str(db_path))
            moments = detector.get_agent_moments(agent, limit=limit)
            narrative = detector.get_narrative_summary(agent) if moments else ""

            self._send_json({
                "agent": agent,
                "moments": [
                    {
                        "id": m.get("id", ""),
                        "type": m.get("type", "unknown"),
                        "description": m.get("description", ""),
                        "significance_score": m.get("significance_score", 0.0),
                        "created_at": m.get("created_at", ""),
                    }
                    for m in moments
                ],
                "narrative": narrative,
                "count": len(moments),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "agent_moments")}, status=500)

    def _get_debate_impasse(self, debate_id: str) -> None:
        """Analyze debate for impasse/deadlock detection."""
        if not self._check_rate_limit():
            return

        if not IMPASSE_DETECTOR_AVAILABLE:
            self._send_json({"error": "ImpasseDetector not available"}, status=503)
            return

        try:
            # Load debate from storage
            if not self.storage:
                self._send_json({"error": "Storage not configured"}, status=503)
                return

            debate = self.storage.get_debate(debate_id)
            if not debate:
                self._send_json({"error": "Debate not found"}, status=404)
                return

            # Extract messages for impasse analysis
            transcript = debate.get("transcript", [])
            if not transcript:
                self._send_json({
                    "debate_id": debate_id,
                    "has_impasse": False,
                    "pivot_claim": None,
                    "should_branch": False,
                })
                return

            # Run impasse detection
            detector = ImpasseDetector()
            pivot = detector.detect_impasse(transcript)

            self._send_json({
                "debate_id": debate_id,
                "has_impasse": pivot is not None,
                "pivot_claim": {
                    "claim_id": pivot.claim_id if pivot else None,
                    "statement": pivot.statement if pivot else None,
                    "disagreement_score": pivot.disagreement_score if pivot else 0.0,
                    "importance_score": pivot.importance_score if pivot else 0.0,
                } if pivot else None,
                "should_branch": pivot is not None and pivot.importance_score > 0.5,
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "debate_impasse")}, status=500)

    def _get_debate_convergence(self, debate_id: str) -> None:
        """Check semantic convergence of agent positions in a debate."""
        if not self._check_rate_limit():
            return

        if not CONVERGENCE_DETECTOR_AVAILABLE:
            self._send_json({"error": "ConvergenceDetector not available"}, status=503)
            return

        try:
            # Load debate from storage
            if not self.storage:
                self._send_json({"error": "Storage not configured"}, status=503)
                return

            debate = self.storage.get_debate(debate_id)
            if not debate:
                self._send_json({"error": "Debate not found"}, status=404)
                return

            # Extract agent positions from transcript
            transcript = debate.get("transcript", [])
            positions = {}
            for msg in transcript:
                agent = msg.get("agent") or msg.get("speaker", "unknown")
                content = msg.get("content", "")
                if content:
                    positions[agent] = content  # Use latest position

            if len(positions) < 2:
                self._send_json({
                    "debate_id": debate_id,
                    "convergence_score": 0.0,
                    "is_converged": False,
                    "recommendation": "Not enough positions to analyze",
                })
                return

            # Run convergence detection
            detector = ConvergenceDetector(threshold=0.85)
            result = detector.check_convergence(list(positions.values()))

            self._send_json({
                "debate_id": debate_id,
                "convergence_score": round(result.similarity, 3),
                "is_converged": result.converged,
                "threshold": 0.85,
                "recommendation": (
                    "Positions have converged; further rounds unlikely to produce new insights"
                    if result.converged
                    else "Positions still divergent; debate may continue productively"
                ),
                "positions_analyzed": len(positions),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "debate_convergence")}, status=500)

    def _get_critique_patterns(self, limit: int, min_success: float) -> None:
        """Get high-impact critique patterns for learning."""
        if not self._check_rate_limit():
            return

        if not CRITIQUE_STORE_AVAILABLE:
            self._send_json({"error": "Critique store not available"}, status=503)
            return

        try:
            db_path = self.nomic_dir / "debates.db" if self.nomic_dir else None
            if not db_path or not db_path.exists():
                self._send_json({"patterns": [], "count": 0})
                return

            store = CritiqueStore(str(db_path))
            patterns = store.retrieve_patterns(min_success_rate=min_success, limit=limit)
            stats = store.get_stats()

            self._send_json({
                "patterns": [
                    {
                        "issue_type": p.issue_type,
                        "pattern": p.pattern_text,
                        "success_rate": p.success_rate,
                        "usage_count": p.usage_count,
                    }
                    for p in patterns
                ],
                "count": len(patterns),
                "stats": stats,
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "critique_patterns")}, status=500)

    def _get_archive_stats(self) -> None:
        """Get archive statistics from critique store."""
        if not self._check_rate_limit():
            return

        if not CRITIQUE_STORE_AVAILABLE:
            self._send_json({"error": "Critique store not available"}, status=503)
            return

        try:
            db_path = self.nomic_dir / "debates.db" if self.nomic_dir else None
            if not db_path or not db_path.exists():
                self._send_json({"archived": 0, "by_type": {}})
                return

            store = CritiqueStore(str(db_path))
            stats = store.get_archive_stats()
            self._send_json(stats)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "archive_stats")}, status=500)

    def _get_all_reputations(self) -> None:
        """Get all agent reputations ranked by score."""
        if not self._check_rate_limit():
            return

        if not CRITIQUE_STORE_AVAILABLE:
            self._send_json({"error": "Critique store not available"}, status=503)
            return

        try:
            db_path = self.nomic_dir / "debates.db" if self.nomic_dir else None
            if not db_path or not db_path.exists():
                self._send_json({"reputations": [], "count": 0})
                return

            store = CritiqueStore(str(db_path))
            reputations = store.get_all_reputations()
            self._send_json({
                "reputations": [
                    {
                        "agent": r.agent_name,
                        "score": r.reputation_score,
                        "vote_weight": r.vote_weight,  # Use property directly (avoids N+1 query)
                        "proposal_acceptance_rate": r.proposal_acceptance_rate,
                        "critique_value": r.critique_value,
                        "debates_participated": r.debates_participated,
                    }
                    for r in reputations
                ],
                "count": len(reputations),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "reputations")}, status=500)

    def _get_agent_reputation(self, agent: str) -> None:
        """Get reputation for a specific agent."""
        if not self._check_rate_limit():
            return

        if not CRITIQUE_STORE_AVAILABLE:
            self._send_json({"error": "Critique store not available"}, status=503)
            return

        try:
            db_path = self.nomic_dir / "debates.db" if self.nomic_dir else None
            if not db_path or not db_path.exists():
                self._send_json({"agent": agent, "message": "No reputation data"})
                return

            store = CritiqueStore(str(db_path))
            rep = store.get_reputation(agent)
            if rep:
                self._send_json({
                    "agent": agent,
                    "score": rep.reputation_score,
                    "vote_weight": rep.vote_weight,  # Use property directly (avoids extra query)
                    "proposal_acceptance_rate": rep.proposal_acceptance_rate,
                    "critique_value": rep.critique_value,
                    "debates_participated": rep.debates_participated,
                })
            else:
                self._send_json({"agent": agent, "score": 0.5, "message": "No reputation data"})
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "agent_reputation")}, status=500)

    def _get_agent_introspection(self, agent: str) -> None:
        """Get introspection data for a specific agent."""
        if not self._check_rate_limit():
            return

        try:
            from aragora.introspection import get_agent_introspection, IntrospectionSnapshot
        except ImportError:
            self._send_json({"error": "Introspection module not available"}, status=503)
            return

        try:
            # Get critique store if available
            memory = None
            if CRITIQUE_STORE_AVAILABLE:
                db_path = self.nomic_dir / "debates.db" if self.nomic_dir else None
                if db_path and db_path.exists():
                    memory = CritiqueStore(str(db_path))

            # Get persona manager if available
            persona_manager = None
            if PERSONA_MANAGER_AVAILABLE:
                persona_db = self.nomic_dir / "personas.db" if self.nomic_dir else None
                if persona_db and persona_db.exists():
                    from aragora.agents.personas import PersonaManager
                    persona_manager = PersonaManager(str(persona_db))

            snapshot = get_agent_introspection(agent, memory=memory, persona_manager=persona_manager)
            self._send_json(snapshot.to_dict())
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "introspection")}, status=500)

    def _get_all_introspection(self) -> None:
        """Get introspection data for all known agents."""
        if not self._check_rate_limit():
            return

        try:
            from aragora.introspection import get_agent_introspection
        except ImportError:
            self._send_json({"error": "Introspection module not available"}, status=503)
            return

        try:
            # Get all known agents from reputation store
            agents = []
            if CRITIQUE_STORE_AVAILABLE:
                db_path = self.nomic_dir / "debates.db" if self.nomic_dir else None
                if db_path and db_path.exists():
                    store = CritiqueStore(str(db_path))
                    reputations = store.get_all_reputations()
                    agents = [r.agent_name for r in reputations]

            if not agents:
                # Default agents
                agents = ["gemini", "claude", "codex", "grok", "deepseek"]

            # Get critique store and persona manager
            memory = None
            persona_manager = None
            if CRITIQUE_STORE_AVAILABLE:
                db_path = self.nomic_dir / "debates.db" if self.nomic_dir else None
                if db_path and db_path.exists():
                    memory = CritiqueStore(str(db_path))
            if PERSONA_MANAGER_AVAILABLE:
                persona_db = self.nomic_dir / "personas.db" if self.nomic_dir else None
                if persona_db and persona_db.exists():
                    from aragora.agents.personas import PersonaManager
                    persona_manager = PersonaManager(str(persona_db))

            snapshots = {}
            for agent in agents:
                snapshot = get_agent_introspection(agent, memory=memory, persona_manager=persona_manager)
                snapshots[agent] = snapshot.to_dict()

            self._send_json({"agents": snapshots, "count": len(snapshots)})
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "all_introspection")}, status=500)

    def _get_introspection_leaderboard(self, limit: int) -> None:
        """Get agents ranked by reputation score."""
        if not self._check_rate_limit():
            return

        try:
            from aragora.introspection import get_agent_introspection
        except ImportError:
            self._send_json({"error": "Introspection module not available"}, status=503)
            return

        try:
            # Get all known agents from reputation store
            agents = []
            memory = None
            if CRITIQUE_STORE_AVAILABLE:
                db_path = self.nomic_dir / "debates.db" if self.nomic_dir else None
                if db_path and db_path.exists():
                    memory = CritiqueStore(str(db_path))
                    reputations = memory.get_all_reputations()
                    agents = [r.agent_name for r in reputations]

            if not agents:
                agents = ["gemini", "claude", "codex", "grok", "deepseek"]

            persona_manager = None
            if PERSONA_MANAGER_AVAILABLE:
                persona_db = self.nomic_dir / "personas.db" if self.nomic_dir else None
                if persona_db and persona_db.exists():
                    from aragora.agents.personas import PersonaManager
                    persona_manager = PersonaManager(str(persona_db))

            snapshots = []
            for agent in agents:
                snapshot = get_agent_introspection(agent, memory=memory, persona_manager=persona_manager)
                snapshots.append(snapshot.to_dict())

            # Sort by reputation score descending
            snapshots.sort(key=lambda x: x.get("reputation_score", 0), reverse=True)

            self._send_json({
                "leaderboard": snapshots[:limit],
                "total_agents": len(snapshots),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "introspection_leaderboard")}, status=500)

    # === Plugins API Implementation ===

    def _list_plugins(self) -> None:
        """List all available plugins."""
        if not self._check_rate_limit():
            return

        try:
            from aragora.plugins.runner import get_registry
            registry = get_registry()
            plugins = registry.list_plugins()
            self._send_json({
                "plugins": [p.to_dict() for p in plugins],
                "count": len(plugins),
            })
        except ImportError:
            self._send_json({"error": "Plugins module not available"}, status=503)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "list_plugins")}, status=500)

    def _get_plugin(self, plugin_name: str) -> None:
        """Get details for a specific plugin."""
        if not self._check_rate_limit():
            return

        try:
            from aragora.plugins.runner import get_registry
            registry = get_registry()
            manifest = registry.get(plugin_name)
            if manifest:
                # Also check if requirements are satisfied
                runner = registry.get_runner(plugin_name)
                if runner:
                    valid, missing = runner._validate_requirements()
                    self._send_json({
                        **manifest.to_dict(),
                        "requirements_satisfied": valid,
                        "missing_requirements": missing,
                    })
                else:
                    self._send_json(manifest.to_dict())
            else:
                self._send_json({"error": f"Plugin not found: {plugin_name}"}, status=404)
        except ImportError:
            self._send_json({"error": "Plugins module not available"}, status=503)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "get_plugin")}, status=500)

    def _run_plugin(self, plugin_name: str) -> None:
        """Run a plugin with provided input."""
        if not self._check_rate_limit():
            return

        # Rate limit plugin execution more strictly
        if not self._check_upload_rate_limit():
            return

        body = self._read_json_body()
        if body is None:
            return

        input_data = body.get("input", {})
        config = body.get("config", {})
        working_dir = body.get("working_dir", ".")

        # Validate working_dir (must be under current directory for security)
        try:
            from pathlib import Path
            cwd = Path.cwd().resolve()
            work_path = Path(working_dir).resolve()
            if not str(work_path).startswith(str(cwd)):
                self._send_json({"error": "Working directory must be under current directory"}, status=400)
                return
        except Exception:
            self._send_json({"error": "Invalid working directory"}, status=400)
            return

        try:
            from aragora.plugins.runner import get_registry
            import asyncio

            registry = get_registry()
            manifest = registry.get(plugin_name)
            if not manifest:
                self._send_json({"error": f"Plugin not found: {plugin_name}"}, status=404)
                return

            # Run plugin with timeout
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    registry.run_plugin(plugin_name, input_data, config, working_dir)
                )
                self._send_json(result.to_dict())
            finally:
                loop.close()

        except ImportError:
            self._send_json({"error": "Plugins module not available"}, status=503)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "run_plugin")}, status=500)

    # === Genesis API Implementation ===

    def _get_genesis_stats(self) -> None:
        """Get overall genesis statistics for evolution visibility."""
        if not self._check_rate_limit():
            return

        try:
            from aragora.genesis.ledger import GenesisLedger, GenesisEventType

            ledger_path = ".nomic/genesis.db"
            if self.nomic_dir:
                ledger_path = str(self.nomic_dir / "genesis.db")

            ledger = GenesisLedger(ledger_path)

            # Count events by type
            event_counts = {}
            for event_type in GenesisEventType:
                events = ledger.get_events_by_type(event_type)
                event_counts[event_type.value] = len(events)

            # Get recent births and deaths
            births = ledger.get_events_by_type(GenesisEventType.AGENT_BIRTH)
            deaths = ledger.get_events_by_type(GenesisEventType.AGENT_DEATH)

            # Get fitness updates for trend
            fitness_updates = ledger.get_events_by_type(GenesisEventType.FITNESS_UPDATE)
            avg_fitness_change = 0.0
            if fitness_updates:
                changes = [e.data.get("change", 0) for e in fitness_updates[-50:]]
                avg_fitness_change = sum(changes) / len(changes) if changes else 0.0

            self._send_json({
                "event_counts": event_counts,
                "total_events": sum(event_counts.values()),
                "total_births": len(births),
                "total_deaths": len(deaths),
                "net_population_change": len(births) - len(deaths),
                "avg_fitness_change_recent": round(avg_fitness_change, 4),
                "integrity_verified": ledger.verify_integrity(),
                "merkle_root": ledger.get_merkle_root()[:32] + "...",
            })
        except ImportError:
            self._send_json({"error": "Genesis module not available"}, status=503)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "genesis_stats")}, status=500)

    def _get_genesis_events(self, limit: int, event_type: str = None) -> None:
        """Get recent genesis events."""
        if not self._check_rate_limit():
            return

        try:
            from aragora.genesis.ledger import GenesisLedger, GenesisEventType
            import sqlite3
            import json

            ledger_path = ".nomic/genesis.db"
            if self.nomic_dir:
                ledger_path = str(self.nomic_dir / "genesis.db")

            # Filter by type if specified
            if event_type:
                try:
                    etype = GenesisEventType(event_type)
                    ledger = GenesisLedger(ledger_path)
                    events = ledger.get_events_by_type(etype)[-limit:]
                    self._send_json({
                        "events": [e.to_dict() for e in events],
                        "count": len(events),
                        "filter": event_type,
                    })
                    return
                except ValueError:
                    self._send_json({"error": f"Unknown event type: {event_type}"}, status=400)
                    return

            # Get all recent events
            conn = sqlite3.connect(ledger_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT event_id, event_type, timestamp, parent_event_id, content_hash, data
                FROM genesis_events
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            events = []
            for row in cursor.fetchall():
                events.append({
                    "event_id": row[0],
                    "event_type": row[1],
                    "timestamp": row[2],
                    "parent_event_id": row[3],
                    "content_hash": row[4][:16] + "...",
                    "data": json.loads(row[5]) if row[5] else {},
                })

            conn.close()

            self._send_json({
                "events": events,
                "count": len(events),
            })
        except ImportError:
            self._send_json({"error": "Genesis module not available"}, status=503)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "genesis_events")}, status=500)

    def _get_genome_lineage(self, genome_id: str) -> None:
        """Get the lineage (ancestry) of a genome."""
        if not self._check_rate_limit():
            return

        try:
            from aragora.genesis.ledger import GenesisLedger

            ledger_path = ".nomic/genesis.db"
            if self.nomic_dir:
                ledger_path = str(self.nomic_dir / "genesis.db")

            ledger = GenesisLedger(ledger_path)
            lineage = ledger.get_lineage(genome_id)

            if lineage:
                self._send_json({
                    "genome_id": genome_id,
                    "lineage": lineage,
                    "generations": len(lineage),
                })
            else:
                self._send_json({"error": f"Genome not found: {genome_id}"}, status=404)

        except ImportError:
            self._send_json({"error": "Genesis module not available"}, status=503)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "genome_lineage")}, status=500)

    def _get_debate_tree(self, debate_id: str) -> None:
        """Get the fractal tree structure for a debate."""
        if not self._check_rate_limit():
            return

        try:
            from aragora.genesis.ledger import GenesisLedger

            ledger_path = ".nomic/genesis.db"
            if self.nomic_dir:
                ledger_path = str(self.nomic_dir / "genesis.db")

            ledger = GenesisLedger(ledger_path)
            tree = ledger.get_debate_tree(debate_id)

            self._send_json({
                "debate_id": debate_id,
                "tree": tree.to_dict(),
                "total_nodes": len(tree.nodes),
            })

        except ImportError:
            self._send_json({"error": "Genesis module not available"}, status=503)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "debate_tree")}, status=500)

    def _get_ranking_stats(self) -> None:
        """Get ELO ranking system statistics."""
        if not self._check_rate_limit():
            return

        if not RANKING_AVAILABLE or not self.elo_system:
            self._send_json({"error": "Ranking system not available"}, status=503)
            return

        try:
            stats = self.elo_system.get_stats()
            self._send_json(stats)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "ranking_stats")}, status=500)

    def _get_memory_stats(self) -> None:
        """Get memory tier statistics from continuum memory."""
        if not self._check_rate_limit():
            return

        try:
            from aragora.memory.continuum import ContinuumMemory

            db_path = self.nomic_dir / "continuum_memory.db" if self.nomic_dir else None
            if not db_path or not db_path.exists():
                self._send_json({"message": "No memory data available", "tiers": {}})
                return

            memory = ContinuumMemory(db_path=str(db_path))
            stats = memory.get_stats()
            self._send_json(stats)
        except ImportError:
            self._send_json({"error": "Continuum memory module not available"}, status=503)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "memory_stats")}, status=500)

    def _get_agent_comparison(self, agent_a: str, agent_b: str) -> None:
        """Get head-to-head comparison between two agents."""
        if not self._check_rate_limit():
            return

        if not RANKING_AVAILABLE or not self.elo_system:
            self._send_json({"error": "Ranking system not available"}, status=503)
            return

        try:
            comparison = self.elo_system.get_head_to_head(agent_a, agent_b)
            if comparison:
                self._send_json({
                    "agent_a": agent_a,
                    "agent_b": agent_b,
                    **comparison
                })
            else:
                self._send_json({
                    "agent_a": agent_a,
                    "agent_b": agent_b,
                    "matches": 0,
                    "message": "No head-to-head data available"
                })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "agent_comparison")}, status=500)

    def _get_head_to_head(self, agent: str, opponent: str) -> None:
        """Get detailed head-to-head statistics between two agents."""
        if not self._check_rate_limit():
            return

        if not RANKING_AVAILABLE or not self.elo_system:
            self._send_json({"error": "Ranking system not available"}, status=503)
            return

        try:
            h2h = self.elo_system.get_head_to_head(agent, opponent)
            if h2h:
                self._send_json({
                    "agent": agent,
                    "opponent": opponent,
                    **h2h,
                })
            else:
                self._send_json({
                    "agent": agent,
                    "opponent": opponent,
                    "matches": 0,
                    "message": "No head-to-head data available"
                })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "head_to_head")}, status=500)

    def _get_opponent_briefing(self, agent: str, opponent: str) -> None:
        """Get strategic briefing about an opponent for an agent."""
        if not self._check_rate_limit():
            return

        if not RELATIONSHIP_TRACKER_AVAILABLE:
            self._send_json({"error": "Relationship tracker not available"}, status=503)
            return

        try:
            from aragora.agents.grounded import PersonaSynthesizer
            synthesizer = PersonaSynthesizer(
                elo_system=self.elo_system,
                calibration_tracker=CalibrationTracker() if CALIBRATION_AVAILABLE else None,
                position_ledger=self.position_ledger,
            )
            briefing = synthesizer.get_opponent_briefing(agent, opponent)
            if briefing:
                self._send_json({
                    "agent": agent,
                    "opponent": opponent,
                    "briefing": briefing,
                })
            else:
                self._send_json({
                    "agent": agent,
                    "opponent": opponent,
                    "briefing": None,
                    "message": "No opponent data available"
                })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "opponent_briefing")}, status=500)

    def _get_calibration_curve(self, agent: str, buckets: int, domain: str = None) -> None:
        """Get calibration curve (expected vs actual accuracy per bucket)."""
        if not self._check_rate_limit():
            return

        if not CALIBRATION_AVAILABLE:
            self._send_json({"error": "Calibration tracker not available"}, status=503)
            return

        try:
            tracker = CalibrationTracker()
            curve = tracker.get_calibration_curve(agent, num_buckets=buckets, domain=domain)
            self._send_json({
                "agent": agent,
                "domain": domain,
                "buckets": [
                    {
                        "range_start": b.range_start,
                        "range_end": b.range_end,
                        "total_predictions": b.total_predictions,
                        "correct_predictions": b.correct_predictions,
                        "accuracy": b.accuracy,
                        "expected_accuracy": (b.range_start + b.range_end) / 2,
                        "brier_score": b.brier_score,
                    }
                    for b in curve
                ],
                "count": len(curve),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "calibration_curve")}, status=500)

    def _get_meta_critique(self, debate_id: str) -> None:
        """Get meta-level analysis of a debate (repetition, circular arguments, etc)."""
        if not self._check_rate_limit():
            return

        try:
            from aragora.debate.meta import MetaCritiqueAnalyzer
            from aragora.debate.traces import DebateTrace

            # Load debate trace
            if not self.nomic_dir:
                self._send_json({"error": "Nomic directory not configured"}, status=503)
                return

            trace_path = self.nomic_dir / "traces" / f"{debate_id}.json"
            if not trace_path.exists():
                self._send_json({"error": "Debate trace not found"}, status=404)
                return

            trace = DebateTrace.load(trace_path)
            result = trace.to_debate_result()

            analyzer = MetaCritiqueAnalyzer()
            critique = analyzer.analyze(result)

            self._send_json({
                "debate_id": debate_id,
                "overall_quality": critique.overall_quality,
                "productive_rounds": critique.productive_rounds,
                "unproductive_rounds": critique.unproductive_rounds,
                "observations": [
                    {
                        "type": o.observation_type,
                        "severity": o.severity,
                        "agent": o.agent,
                        "round": o.round_num,
                        "description": o.description,
                    }
                    for o in critique.observations
                ],
                "recommendations": critique.recommendations,
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "meta_critique")}, status=500)

    def _get_debate_graph_stats(self, debate_id: str) -> None:
        """Get argument graph statistics for a debate.

        Returns:
            node_count, edge_count: Basic counts
            max_depth: Maximum argument chain length
            avg_branching: Average outgoing edges per node
            complexity_score: 0-1 normalized complexity metric
            claim_count, rebuttal_count: Type-specific counts
        """
        if not self._check_rate_limit():
            return

        try:
            from aragora.visualization.mapper import ArgumentCartographer
            from aragora.debate.traces import DebateTrace

            # Load debate trace
            if not self.nomic_dir:
                self._send_json({"error": "Nomic directory not configured"}, status=503)
                return

            trace_path = self.nomic_dir / "traces" / f"{debate_id}.json"
            if not trace_path.exists():
                # Try replays directory as fallback
                replay_path = self.nomic_dir / "replays" / debate_id / "events.jsonl"
                if replay_path.exists():
                    # Build cartographer from replay events
                    cartographer = ArgumentCartographer()
                    cartographer.set_debate_context(debate_id, "")
                    with replay_path.open() as f:
                        for line in f:
                            if line.strip():
                                event = json.loads(line)
                                if event.get("type") == "agent_message":
                                    cartographer.update_from_message(
                                        agent=event.get("agent", "unknown"),
                                        content=event.get("data", {}).get("content", ""),
                                        role=event.get("data", {}).get("role", "proposer"),
                                        round_num=event.get("round", 1),
                                    )
                                elif event.get("type") == "critique":
                                    cartographer.update_from_critique(
                                        critic_agent=event.get("agent", "unknown"),
                                        target_agent=event.get("data", {}).get("target", "unknown"),
                                        severity=event.get("data", {}).get("severity", 0.5),
                                        round_num=event.get("round", 1),
                                        critique_text=event.get("data", {}).get("content", ""),
                                    )
                    stats = cartographer.get_statistics()
                    self._send_json(stats)
                    return
                else:
                    self._send_json({"error": "Debate not found"}, status=404)
                    return

            # Load from trace file
            trace = DebateTrace.load(trace_path)
            result = trace.to_debate_result()

            # Build cartographer from debate result
            cartographer = ArgumentCartographer()
            cartographer.set_debate_context(debate_id, result.task or "")

            # Process messages from the debate
            for msg in result.messages:
                cartographer.update_from_message(
                    agent=msg.agent,
                    content=msg.content,
                    role=msg.role,
                    round_num=msg.round,
                )

            # Process critiques
            for critique in result.critiques:
                cartographer.update_from_critique(
                    critic_agent=critique.agent,
                    target_agent=critique.target or "",
                    severity=critique.severity,
                    round_num=getattr(critique, 'round', 1),
                    critique_text=critique.reasoning,
                )

            stats = cartographer.get_statistics()
            self._send_json(stats)

        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "debate_graph_stats")}, status=500)

    def _get_emergent_traits(self, min_confidence: float, limit: int) -> None:
        """Get emergent traits detected from agent performance patterns."""
        if not self._check_rate_limit():
            return

        if not LABORATORY_AVAILABLE:
            self._send_json({"error": "Persona laboratory not available"}, status=503)
            return

        try:
            lab = PersonaLaboratory(
                db_path=str(self.nomic_dir / "laboratory.db") if self.nomic_dir else None,
                persona_manager=self.persona_manager,
            )
            traits = lab.detect_emergent_traits()
            # Filter by confidence and limit
            filtered = [t for t in traits if t.confidence >= min_confidence][:limit]
            self._send_json({
                "emergent_traits": [
                    {
                        "agent": t.agent_name,
                        "trait": t.trait_name,
                        "domain": t.domain,
                        "confidence": t.confidence,
                        "evidence": t.evidence,
                        "detected_at": t.detected_at,
                    }
                    for t in filtered
                ],
                "count": len(filtered),
                "min_confidence": min_confidence,
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "emergent_traits")}, status=500)

    def _suggest_cross_pollinations(self) -> None:
        """Suggest beneficial trait transfers for a target agent (POST)."""
        if not self._check_rate_limit():
            return

        if not LABORATORY_AVAILABLE:
            self._send_json({"error": "Persona laboratory not available"}, status=503)
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}

            target_agent = data.get('target_agent')
            if not target_agent:
                self._send_json({"error": "target_agent required"}, status=400)
                return

            lab = PersonaLaboratory(
                db_path=str(self.nomic_dir / "laboratory.db") if self.nomic_dir else None,
                persona_manager=self.persona_manager,
            )
            suggestions = lab.suggest_cross_pollinations(target_agent)
            self._send_json({
                "target_agent": target_agent,
                "suggestions": [
                    {
                        "source_agent": s[0],
                        "trait_or_domain": s[1],
                        "reason": s[2],
                    }
                    for s in suggestions
                ],
                "count": len(suggestions),
            })
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=400)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "cross_pollinations")}, status=500)

    def _get_debate_cruxes(self, debate_id: str, top_k: int) -> None:
        """Get key claims that would most impact the debate outcome."""
        if not self._check_rate_limit():
            return

        if not BELIEF_NETWORK_AVAILABLE:
            self._send_json({"error": "Belief network not available"}, status=503)
            return

        try:
            from aragora.debate.traces import DebateTrace

            if not self.nomic_dir:
                self._send_json({"error": "Nomic directory not configured"}, status=503)
                return

            trace_path = self.nomic_dir / "traces" / f"{debate_id}.json"
            if not trace_path.exists():
                self._send_json({"error": "Debate trace not found"}, status=404)
                return

            trace = DebateTrace.load(trace_path)
            result = trace.to_debate_result()

            # Build belief network from debate
            network = BeliefNetwork(debate_id=debate_id)
            for msg in result.messages:
                network.add_claim(msg.agent, msg.content[:200], confidence=0.7)

            analyzer = BeliefPropagationAnalyzer(network)
            cruxes = analyzer.identify_debate_cruxes(top_k=top_k)

            self._send_json({
                "debate_id": debate_id,
                "cruxes": cruxes,
                "count": len(cruxes),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "debate_cruxes")}, status=500)

    def _get_claim_support(self, debate_id: str, claim_id: str) -> None:
        """Get verification status of all evidence supporting a claim."""
        if not self._check_rate_limit():
            return

        if not PROVENANCE_AVAILABLE:
            self._send_json({"error": "Provenance tracker not available"}, status=503)
            return

        try:
            if not self.nomic_dir:
                self._send_json({"error": "Nomic directory not configured"}, status=503)
                return

            provenance_path = self.nomic_dir / "provenance" / f"{debate_id}.json"
            if not provenance_path.exists():
                self._send_json({
                    "debate_id": debate_id,
                    "claim_id": claim_id,
                    "support": None,
                    "message": "No provenance data for this debate"
                })
                return

            tracker = ProvenanceTracker.load(provenance_path)
            support = tracker.get_claim_support(claim_id)

            self._send_json({
                "debate_id": debate_id,
                "claim_id": claim_id,
                "support": support,
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "claim_support")}, status=500)

    def _get_routing_recommendations(self) -> None:
        """Get agent recommendations for a task (POST)."""
        if not self._check_rate_limit():
            return

        if not ROUTING_AVAILABLE:
            self._send_json({"error": "Agent selector not available"}, status=503)
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}

            primary_domain = data.get('primary_domain', 'general')
            secondary_domains = data.get('secondary_domains', [])
            required_traits = data.get('required_traits', [])
            limit = min(data.get('limit', 5), 20)

            requirements = TaskRequirements(
                task_id=data.get('task_id', 'ad-hoc'),
                primary_domain=primary_domain,
                secondary_domains=secondary_domains,
                required_traits=required_traits,
            )

            selector = AgentSelector(
                elo_system=self.elo_system,
                persona_manager=self.persona_manager,
            )
            recommendations = selector.get_recommendations(requirements, limit=limit)

            self._send_json({
                "task_id": requirements.task_id,
                "primary_domain": primary_domain,
                "recommendations": recommendations,
                "count": len(recommendations),
            })
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=400)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "routing_recommendations")}, status=500)

    def _formal_verify_claim(self) -> None:
        """Attempt formal verification of a claim using Z3 SMT solver.

        POST body:
            claim: The claim to verify (required)
            claim_type: Optional hint (assertion, logical, arithmetic, etc.)
            context: Optional additional context
            timeout: Timeout in seconds (default: 30, max: 120)

        Returns:
            status: proof_found, proof_failed, translation_failed, etc.
            is_verified: True if claim was formally proven
            formal_statement: The SMT-LIB2 translation (if successful)
            proof_hash: Hash of the proof (if found)
        """
        if not self._check_rate_limit():
            return

        if not FORMAL_VERIFICATION_AVAILABLE:
            self._send_json({
                "error": "Formal verification not available",
                "hint": "Install z3-solver: pip install z3-solver"
            }, status=503)
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}

            claim = data.get('claim', '').strip()
            if not claim:
                self._send_json({"error": "Missing required field: claim"}, status=400)
                return

            claim_type = data.get('claim_type')
            context = data.get('context', '')
            timeout = min(float(data.get('timeout', 30)), 120)

            # Get the formal verification manager
            manager = get_formal_verification_manager()

            # Check backend availability
            status_report = manager.status_report()
            if not status_report.get("any_available"):
                self._send_json({
                    "error": "No formal verification backends available",
                    "backends": status_report.get("backends", []),
                }, status=503)
                return

            # Run verification asynchronously
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    manager.attempt_formal_verification(
                        claim=claim,
                        claim_type=claim_type,
                        context=context,
                        timeout_seconds=timeout,
                    )
                )
            finally:
                loop.close()

            # Build response
            response = result.to_dict()
            response["claim"] = claim
            if claim_type:
                response["claim_type"] = claim_type

            self._send_json(response)

        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=400)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "formal_verification")}, status=500)

    def _formal_verification_status(self) -> None:
        """Get status of formal verification backends.

        Returns availability of Z3 and Lean backends.
        """
        if not FORMAL_VERIFICATION_AVAILABLE:
            self._send_json({
                "available": False,
                "hint": "Install z3-solver: pip install z3-solver",
                "backends": [],
            })
            return

        try:
            manager = get_formal_verification_manager()
            status = manager.status_report()
            self._send_json({
                "available": status.get("any_available", False),
                "backends": status.get("backends", []),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "formal_status")}, status=500)

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

        try:
            content_length = int(self.headers.get('Content-Length', 0))
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

    def _run_capability_probe(self) -> None:
        """Run capability probes on an agent to find vulnerabilities.

        POST body:
            agent_name: Name of agent to probe (required)
            probe_types: List of probe types (optional, default: all)
                Options: contradiction, hallucination, sycophancy, persistence,
                         confidence_calibration, reasoning_depth, edge_case
            probes_per_type: Number of probes per type (default: 3, max: 10)
            model_type: Agent model type (optional, default: anthropic-api)

        Returns:
            report_id: Unique report ID
            target_agent: Name of probed agent
            probes_run: Total probes executed
            vulnerabilities_found: Count of vulnerabilities detected
            vulnerability_rate: Fraction of probes that found vulnerabilities
            elo_penalty: ELO penalty applied
            by_type: Results grouped by probe type with passed/failed status
            summary: Quick stats for UI display
        """
        if not self._check_rate_limit():
            return

        if not PROBER_AVAILABLE:
            self._send_json({
                "error": "Capability prober not available",
                "hint": "Prober module failed to import"
            }, status=503)
            return

        if not DEBATE_AVAILABLE or create_agent is None:
            self._send_json({
                "error": "Agent system not available",
                "hint": "Debate module or create_agent failed to import"
            }, status=503)
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}

            agent_name = data.get('agent_name', '').strip()
            if not agent_name:
                self._send_json({"error": "Missing required field: agent_name"}, status=400)
                return

            # Validate agent name
            if not re.match(SAFE_ID_PATTERN, agent_name):
                self._send_json({"error": "Invalid agent_name format"}, status=400)
                return

            probe_type_strs = data.get('probe_types', [
                'contradiction', 'hallucination', 'sycophancy', 'persistence'
            ])
            probes_per_type = min(int(data.get('probes_per_type', 3)), 10)
            model_type = data.get('model_type', 'anthropic-api')

            from aragora.modes.prober import ProbeType, CapabilityProber
            from datetime import datetime
            import asyncio

            # Convert string probe types to enum
            probe_types = []
            for pt_str in probe_type_strs:
                try:
                    probe_types.append(ProbeType(pt_str))
                except ValueError:
                    pass  # Skip invalid probe types

            if not probe_types:
                self._send_json({"error": "No valid probe types specified"}, status=400)
                return

            # Create agent for probing
            try:
                agent = create_agent(model_type, name=agent_name, role="proposer")
            except Exception as e:
                self._send_json({
                    "error": f"Failed to create agent: {str(e)}",
                    "hint": f"model_type '{model_type}' may not be available"
                }, status=400)
                return

            # Create prober with optional ELO integration
            prober = CapabilityProber(
                elo_system=self.elo_system,
                elo_penalty_multiplier=5.0
            )

            # Get stream hooks if available for real-time updates
            probe_hooks = None
            if hasattr(self.server, 'stream_server') and self.server.stream_server:
                from .nomic_stream import create_nomic_hooks
                probe_hooks = create_nomic_hooks(self.server.stream_server.emitter)

            report_id = f"probe-report-{uuid.uuid4().hex[:8]}"

            # Emit probe start event
            if probe_hooks and 'on_probe_start' in probe_hooks:
                probe_hooks['on_probe_start'](
                    probe_id=report_id,
                    target_agent=agent_name,
                    probe_types=[pt.value for pt in probe_types],
                    probes_per_type=probes_per_type
                )

            # Define run_agent_fn callback for prober
            async def run_agent_fn(target_agent, prompt: str) -> str:
                """Execute agent with probe prompt."""
                try:
                    if asyncio.iscoroutinefunction(target_agent.generate):
                        return await target_agent.generate(prompt)
                    else:
                        return target_agent.generate(prompt)
                except Exception as e:
                    return f"[Agent Error: {str(e)}]"

            # Run probes asynchronously
            async def run_probes():
                return await prober.probe_agent(
                    target_agent=agent,
                    run_agent_fn=run_agent_fn,
                    probe_types=probe_types,
                    probes_per_type=probes_per_type,
                )

            # Execute in event loop
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                report = loop.run_until_complete(run_probes())
                loop.close()
            except Exception as e:
                self._send_json({
                    "error": f"Probe execution failed: {str(e)}"
                }, status=500)
                return

            # Transform results for frontend (vulnerability_found -> passed)
            by_type_transformed = {}
            for probe_type_key, results in report.by_type.items():
                transformed_results = []
                for r in results:
                    result_dict = r.to_dict() if hasattr(r, 'to_dict') else r
                    # Invert vulnerability_found to get passed
                    passed = not result_dict.get('vulnerability_found', False)
                    transformed_results.append({
                        "probe_id": result_dict.get('probe_id', ''),
                        "type": result_dict.get('probe_type', probe_type_key),
                        "passed": passed,
                        "severity": result_dict.get('severity', '').lower() if result_dict.get('severity') else None,
                        "description": result_dict.get('vulnerability_description', ''),
                        "details": result_dict.get('evidence', ''),
                        "response_time_ms": result_dict.get('response_time_ms', 0),
                    })

                    # Emit individual probe result event
                    if probe_hooks and 'on_probe_result' in probe_hooks:
                        probe_hooks['on_probe_result'](
                            probe_id=result_dict.get('probe_id', ''),
                            probe_type=probe_type_key,
                            passed=passed,
                            severity=result_dict.get('severity', '').lower() if result_dict.get('severity') else None,
                            description=result_dict.get('vulnerability_description', ''),
                            response_time_ms=result_dict.get('response_time_ms', 0)
                        )

                by_type_transformed[probe_type_key] = transformed_results

            # Record red team result in ELO system
            if self.elo_system and report.probes_run > 0:
                robustness_score = 1.0 - report.vulnerability_rate
                try:
                    self.elo_system.record_redteam_result(
                        agent_name=agent_name,
                        robustness_score=robustness_score,
                        successful_attacks=report.vulnerabilities_found,
                        total_attacks=report.probes_run,
                        critical_vulnerabilities=report.critical_count,
                        session_id=report_id
                    )
                except Exception:
                    pass  # Don't fail probe if ELO update fails

            # Save results to .nomic/probes/
            if self.nomic_dir:
                try:
                    probes_dir = self.nomic_dir / "probes" / agent_name
                    probes_dir.mkdir(parents=True, exist_ok=True)
                    date_str = datetime.now().strftime("%Y-%m-%d")
                    probe_file = probes_dir / f"{date_str}_{report.report_id}.json"
                    probe_file.write_text(json.dumps(report.to_dict(), indent=2, default=str))
                except Exception:
                    pass  # Don't fail if storage fails

            # Emit probe complete event
            if probe_hooks and 'on_probe_complete' in probe_hooks:
                probe_hooks['on_probe_complete'](
                    report_id=report.report_id,
                    target_agent=agent_name,
                    probes_run=report.probes_run,
                    vulnerabilities_found=report.vulnerabilities_found,
                    vulnerability_rate=report.vulnerability_rate,
                    elo_penalty=report.elo_penalty,
                    by_severity={
                        "critical": report.critical_count,
                        "high": report.high_count,
                        "medium": report.medium_count,
                        "low": report.low_count,
                    }
                )

            # Calculate summary for frontend
            passed_count = report.probes_run - report.vulnerabilities_found
            pass_rate = passed_count / report.probes_run if report.probes_run > 0 else 1.0

            self._send_json({
                "report_id": report.report_id,
                "target_agent": agent_name,
                "probes_run": report.probes_run,
                "vulnerabilities_found": report.vulnerabilities_found,
                "vulnerability_rate": round(report.vulnerability_rate, 3),
                "elo_penalty": round(report.elo_penalty, 1),
                "by_type": by_type_transformed,
                "summary": {
                    "total": report.probes_run,
                    "passed": passed_count,
                    "failed": report.vulnerabilities_found,
                    "pass_rate": round(pass_rate, 3),
                    "critical": report.critical_count,
                    "high": report.high_count,
                    "medium": report.medium_count,
                    "low": report.low_count,
                },
                "recommendations": report.recommendations,
                "created_at": report.created_at,
            })

        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=400)
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "capability_probe")}, status=500)

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

        try:
            content_length = int(self.headers.get('Content-Length', 0))
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
                    rounds=min(int(config_data.get('rounds', 6)), 10),
                    enable_research=config_data.get('enable_research', True),
                    cross_examination_depth=min(int(config_data.get('cross_examination_depth', 3)), 10),
                    risk_threshold=float(config_data.get('risk_threshold', 0.7)),
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
                except Exception:
                    pass

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
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                verdict = loop.run_until_complete(run_audit())
                loop.close()
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
                    except Exception:
                        pass

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
                except Exception:
                    pass  # Don't fail if storage fails

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

        try:
            content_length = int(self.headers.get('Content-Length', 0))
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
            max_rounds = min(int(data.get('max_rounds', 3)), 5)

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

        try:
            content_length = int(self.headers.get('Content-Length', 0))
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

    def _get_tournament_standings(self, tournament_id: str) -> None:
        """Get current tournament standings."""
        if not self._check_rate_limit():
            return

        if not TOURNAMENT_AVAILABLE:
            self._send_json({"error": "Tournament system not available"}, status=503)
            return

        # Validate tournament_id to prevent path traversal
        if not re.match(SAFE_ID_PATTERN, tournament_id):
            self._send_json({"error": "Invalid tournament ID format"}, status=400)
            return

        try:
            if not self.nomic_dir:
                self._send_json({"error": "Nomic directory not configured"}, status=503)
                return

            tournament_path = self.nomic_dir / "tournaments" / f"{tournament_id}.db"
            if not tournament_path.exists():
                self._send_json({"error": "Tournament not found"}, status=404)
                return

            manager = TournamentManager(db_path=str(tournament_path))
            standings = manager.get_current_standings()

            self._send_json({
                "tournament_id": tournament_id,
                "standings": [
                    {
                        "agent": s.agent,
                        "wins": s.wins,
                        "losses": s.losses,
                        "draws": s.draws,
                        "points": s.points,
                        "total_score": s.total_score,
                        "win_rate": s.win_rate,
                    }
                    for s in standings
                ],
                "count": len(standings),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "tournament_standings")}, status=500)

    def _list_tournaments(self) -> None:
        """List all available tournaments."""
        if not self._check_rate_limit():
            return

        if not TOURNAMENT_AVAILABLE:
            self._send_json({"error": "Tournament system not available"}, status=503)
            return

        try:
            if not self.nomic_dir:
                self._send_json({"error": "Nomic directory not configured"}, status=503)
                return

            tournaments_dir = self.nomic_dir / "tournaments"
            tournaments = []

            if tournaments_dir.exists():
                for db_file in tournaments_dir.glob("*.db"):
                    tournament_id = db_file.stem
                    try:
                        manager = TournamentManager(db_path=str(db_file))
                        standings = manager.get_current_standings()
                        tournaments.append({
                            "tournament_id": tournament_id,
                            "participants": len(standings),
                            "total_matches": sum(s.wins + s.losses + s.draws for s in standings) // 2,
                            "top_agent": standings[0].agent if standings else None,
                        })
                    except Exception:
                        # Skip corrupted or invalid tournament files
                        continue

            self._send_json({
                "tournaments": tournaments,
                "count": len(tournaments),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "list_tournaments")}, status=500)

    def _get_best_team_combinations(self, min_debates: int, limit: int) -> None:
        """Get best-performing team combinations from history."""
        if not self._check_rate_limit():
            return

        if not ROUTING_AVAILABLE:
            self._send_json({"error": "Agent selector not available"}, status=503)
            return

        try:
            selector = AgentSelector(
                elo_system=self.elo_system,
                persona_manager=self.persona_manager,
            )
            combinations = selector.get_best_team_combinations(min_debates=min_debates)[:limit]

            self._send_json({
                "min_debates": min_debates,
                "combinations": combinations,
                "count": len(combinations),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "best_teams")}, status=500)

    def _get_evolution_history(self, agent: str, limit: int) -> None:
        """Get prompt evolution history for an agent."""
        if not self._check_rate_limit():
            return

        if not EVOLUTION_AVAILABLE:
            self._send_json({"error": "Prompt evolution not available"}, status=503)
            return

        try:
            if not self.nomic_dir:
                self._send_json({"error": "Nomic directory not configured"}, status=503)
                return

            evolver = PromptEvolver(db_path=str(self.nomic_dir / "prompt_evolution.db"))
            history = evolver.get_evolution_history(agent, limit=limit)

            self._send_json({
                "agent": agent,
                "history": history,
                "count": len(history),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "evolution_history")}, status=500)

    def _get_load_bearing_claims(self, debate_id: str, limit: int) -> None:
        """Get claims with highest centrality (most load-bearing)."""
        if not self._check_rate_limit():
            return

        if not BELIEF_NETWORK_AVAILABLE:
            self._send_json({"error": "Belief network not available"}, status=503)
            return

        try:
            from aragora.debate.traces import DebateTrace

            if not self.nomic_dir:
                self._send_json({"error": "Nomic directory not configured"}, status=503)
                return

            trace_path = self.nomic_dir / "traces" / f"{debate_id}.json"
            if not trace_path.exists():
                self._send_json({"error": "Debate trace not found"}, status=404)
                return

            trace = DebateTrace.load(trace_path)
            result = trace.to_debate_result()

            # Build belief network from debate
            network = BeliefNetwork(debate_id=debate_id)
            for msg in result.messages:
                network.add_claim(msg.agent, msg.content[:200], confidence=0.7)

            load_bearing = network.get_load_bearing_claims(limit=limit)

            self._send_json({
                "debate_id": debate_id,
                "load_bearing_claims": [
                    {
                        "claim_id": node.claim_id,
                        "statement": node.claim_statement,
                        "author": node.author,
                        "centrality": centrality,
                    }
                    for node, centrality in load_bearing
                ],
                "count": len(load_bearing),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "load_bearing_claims")}, status=500)

    def _get_calibration_summary(self, agent: str, domain: str = None) -> None:
        """Get comprehensive calibration summary for an agent."""
        if not self._check_rate_limit():
            return

        if not CALIBRATION_AVAILABLE:
            self._send_json({"error": "Calibration tracker not available"}, status=503)
            return

        try:
            tracker = CalibrationTracker()
            summary = tracker.get_calibration_summary(agent, domain=domain)

            self._send_json({
                "agent": summary.agent,
                "domain": domain,
                "total_predictions": summary.total_predictions,
                "total_correct": summary.total_correct,
                "accuracy": summary.accuracy,
                "brier_score": summary.brier_score,
                "ece": summary.ece,
                "is_overconfident": summary.is_overconfident,
                "is_underconfident": summary.is_underconfident,
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "calibration_summary")}, status=500)

    def _get_continuum_memories(self, query: str, tiers_str: str, limit: int, min_importance: float) -> None:
        """Retrieve memories from the continuum memory system."""
        if not self._check_rate_limit():
            return

        if not CONTINUUM_AVAILABLE:
            self._send_json({"error": "Continuum memory not available"}, status=503)
            return

        try:
            if not self.nomic_dir:
                self._send_json({"error": "Nomic directory not configured"}, status=503)
                return

            memory = ContinuumMemory(db_path=str(self.nomic_dir / "continuum.db"))

            # Parse tier filter
            tier_names = [t.strip().upper() for t in tiers_str.split(',') if t.strip()]
            tiers = []
            for name in tier_names:
                try:
                    tiers.append(MemoryTier[name])
                except KeyError:
                    pass  # Ignore invalid tier names

            if not tiers:
                tiers = [MemoryTier.FAST, MemoryTier.MEDIUM]  # Default

            entries = memory.retrieve(
                query=query if query else None,
                tiers=tiers,
                limit=limit,
                min_importance=min_importance,
            )

            self._send_json({
                "query": query,
                "tiers": [t.name for t in tiers],
                "memories": [
                    {
                        "id": e.id,
                        "tier": e.tier.name,
                        "content": e.content,
                        "importance": e.importance,
                        "surprise_score": e.surprise_score,
                        "consolidation_score": e.consolidation_score,
                        "success_rate": e.success_rate,
                        "update_count": e.update_count,
                        "created_at": e.created_at,
                        "updated_at": e.updated_at,
                    }
                    for e in entries
                ],
                "count": len(entries),
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "continuum_memories")}, status=500)

    def _get_continuum_consolidation(self) -> None:
        """Get memory consolidation status and run consolidation."""
        if not self._check_rate_limit():
            return

        if not CONTINUUM_AVAILABLE:
            self._send_json({"error": "Continuum memory not available"}, status=503)
            return

        try:
            if not self.nomic_dir:
                self._send_json({"error": "Nomic directory not configured"}, status=503)
                return

            memory = ContinuumMemory(db_path=str(self.nomic_dir / "continuum.db"))
            stats = memory.consolidate()

            self._send_json({
                "consolidation": stats,
                "message": "Memory consolidation complete",
            })
        except Exception as e:
            self._send_json({"error": _safe_error_message(e, "continuum_consolidation")}, status=500)

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
        except Exception:
            self.send_error(500, "Failed to read file")

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
                print("[server] Supabase persistence enabled")
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
                insights_path = nomic_dir / "aragora_insights.db"
                if insights_path.exists():
                    UnifiedHandler.insight_store = InsightStore(str(insights_path))
                    print("[server] InsightStore loaded for API access")
            # Initialize EloSystem from nomic directory
            if RANKING_AVAILABLE:
                elo_path = nomic_dir / "agent_elo.db"
                if elo_path.exists():
                    UnifiedHandler.elo_system = EloSystem(str(elo_path))
                    print("[server] EloSystem loaded for leaderboard API")

            # Initialize FlipDetector from nomic directory
            if FLIP_DETECTOR_AVAILABLE:
                positions_path = nomic_dir / "aragora_positions.db"
                if positions_path.exists():
                    UnifiedHandler.flip_detector = FlipDetector(str(positions_path))
                    print("[server] FlipDetector loaded for position reversal API")

            # Initialize DocumentStore for file uploads
            doc_dir = nomic_dir / "documents"
            UnifiedHandler.document_store = DocumentStore(doc_dir)
            print(f"[server] DocumentStore initialized at {doc_dir}")

            # Initialize PersonaManager for agent specialization
            if PERSONAS_AVAILABLE:
                personas_path = nomic_dir / "personas.db"
                UnifiedHandler.persona_manager = PersonaManager(str(personas_path))
                print("[server] PersonaManager loaded for agent specialization")

            # Initialize PositionLedger for truth-grounded personas
            if POSITION_LEDGER_AVAILABLE:
                ledger_path = nomic_dir / "position_ledger.db"
                try:
                    UnifiedHandler.position_ledger = PositionLedger(db_path=str(ledger_path))
                    print("[server] PositionLedger loaded for truth-grounded personas")
                except Exception as e:
                    print(f"[server] PositionLedger initialization failed: {e}")

            # Initialize DebateEmbeddingsDatabase for historical memory
            if EMBEDDINGS_AVAILABLE:
                embeddings_path = nomic_dir / "debate_embeddings.db"
                try:
                    UnifiedHandler.debate_embeddings = DebateEmbeddingsDatabase(str(embeddings_path))
                    print("[server] DebateEmbeddings loaded for historical memory")
                except Exception as e:
                    print(f"[server] DebateEmbeddings initialization failed: {e}")

            # Initialize ConsensusMemory and DissentRetriever for historical minority views
            if CONSENSUS_MEMORY_AVAILABLE and DissentRetriever is not None:
                try:
                    UnifiedHandler.consensus_memory = ConsensusMemory()
                    UnifiedHandler.dissent_retriever = DissentRetriever(UnifiedHandler.consensus_memory)
                    print("[server] DissentRetriever loaded for historical minority views")
                except Exception as e:
                    print(f"[server] DissentRetriever initialization failed: {e}")

    @property
    def emitter(self) -> SyncEventEmitter:
        """Get the event emitter for nomic loop integration."""
        return self.stream_server.emitter

    def _run_http_server(self) -> None:
        """Run HTTP server in a thread."""
        server = HTTPServer((self.http_host, self.http_port), UnifiedHandler)
        server.serve_forever()

    async def start(self) -> None:
        """Start both HTTP and WebSocket servers."""
        print(f"Starting unified server...")
        print(f"  HTTP API:   http://localhost:{self.http_port}")
        print(f"  WebSocket:  ws://localhost:{self.ws_port}")
        if self.static_dir:
            print(f"  Static dir: {self.static_dir}")
        if self.nomic_dir:
            print(f"  Nomic dir:  {self.nomic_dir}")

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
