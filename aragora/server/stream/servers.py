"""
Real-time debate streaming via WebSocket.

The SyncEventEmitter bridges synchronous Arena code with async WebSocket broadcasts.
Events are queued synchronously and consumed by an async drain loop.

This module also supports unified HTTP+WebSocket serving on a single port via aiohttp.

Note: Core components are now in submodules for better organization:
- aragora.server.stream.events - StreamEventType, StreamEvent, AudienceMessage
- aragora.server.stream.emitter - SyncEventEmitter, TokenBucket, AudienceInbox
- aragora.server.stream.state_manager - DebateStateManager, BoundedDebateDict
- aragora.server.stream.arena_hooks - create_arena_hooks, wrap_agent_for_streaming
"""

import asyncio
import json
import logging
import os
import queue
import secrets
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Any, Dict, cast
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    import aiohttp.web
    from aragora.core import Agent
from concurrent.futures import ThreadPoolExecutor
import uuid

# Configure module logger
logger = logging.getLogger(__name__)

# Import from sibling modules (core streaming components)
from .events import (
    StreamEventType,
    StreamEvent,
    AudienceMessage,
)
from .emitter import (
    TokenBucket,
    AudienceInbox,
    SyncEventEmitter,
    normalize_intensity,
)
from .state_manager import (
    BoundedDebateDict,
    LoopInstance,
    DebateStateManager,
    get_active_debates,
    get_active_debates_lock,
    get_debate_executor,
    set_debate_executor,
    get_debate_executor_lock,
    cleanup_stale_debates,
    increment_cleanup_counter,
)
from .arena_hooks import (
    create_arena_hooks,
    wrap_agent_for_streaming,
)
from .stream_handlers import StreamAPIHandlersMixin
from .server_base import ServerBase, ServerConfig

# Import debate components (lazy-loaded for optional functionality)
try:
    from aragora.debate.orchestrator import Arena, DebateProtocol
    from aragora.agents.base import create_agent
    from aragora.core import Environment

    DEBATE_AVAILABLE = True
except ImportError:
    DEBATE_AVAILABLE = False
    Arena = None  # type: ignore[misc, assignment]
    DebateProtocol = None  # type: ignore[misc, assignment]
    create_agent = None  # type: ignore[misc, assignment]
    Environment = None  # type: ignore[misc, assignment]

# Import centralized config and error utilities
from aragora.config import (
    ALLOWED_AGENT_TYPES,
    DB_INSIGHTS_PATH,
    DB_PERSONAS_PATH,
    DEBATE_TIMEOUT_SECONDS,
    MAX_AGENTS_PER_DEBATE,
    MAX_CONCURRENT_DEBATES,
)
from aragora.server.error_utils import safe_error_message as _safe_error_message

# Backward compatibility aliases
_active_debates = get_active_debates()
_active_debates_lock = get_active_debates_lock()
_debate_executor_lock = get_debate_executor_lock()

# TTL for completed debates (24 hours)
_DEBATE_TTL_SECONDS = 86400


def _cleanup_stale_debates_stream() -> None:
    """Remove completed/errored debates older than TTL."""
    cleanup_stale_debates()


# Backward compatibility alias - use wrap_agent_for_streaming from arena_hooks
_wrap_agent_for_streaming = wrap_agent_for_streaming


# Centralized CORS configuration
from aragora.server.cors_config import WS_ALLOWED_ORIGINS

# Import WebSocket config from centralized location
from aragora.config import WS_MAX_MESSAGE_SIZE

# Import auth for WebSocket authentication
from aragora.server.auth import auth_config

# Trusted proxies for X-Forwarded-For header validation
# Only trust X-Forwarded-For if request comes from these IPs
TRUSTED_PROXIES = frozenset(
    p.strip() for p in os.getenv("ARAGORA_TRUSTED_PROXIES", "127.0.0.1,::1,localhost").split(",")
)

# =============================================================================
# WebSocket Security Configuration
# =============================================================================

# Connection rate limiting per IP
WS_CONNECTIONS_PER_IP_PER_MINUTE = int(os.getenv("ARAGORA_WS_CONN_RATE", "30"))

# Token revalidation interval for long-lived connections (5 minutes)
WS_TOKEN_REVALIDATION_INTERVAL = 300.0

# Maximum connections per IP (concurrent)
WS_MAX_CONNECTIONS_PER_IP = int(os.getenv("ARAGORA_WS_MAX_PER_IP", "10"))


# =============================================================================
# NOTE: Core streaming classes are now in submodules for better organization:
# - StreamEventType, StreamEvent, AudienceMessage -> aragora.server.stream.events
# - TokenBucket, AudienceInbox, SyncEventEmitter -> aragora.server.stream.emitter
# - BoundedDebateDict, LoopInstance, DebateStateManager -> aragora.server.stream.state_manager
# - create_arena_hooks, wrap_agent_for_streaming -> aragora.server.stream.arena_hooks
# - DebateStreamServer -> aragora.server.stream.debate_stream_server
#
# The classes are imported at the top of this file for backward compatibility.
# =============================================================================

# Import DebateStreamServer from its dedicated module for backward compatibility
from .debate_stream_server import DebateStreamServer


# =============================================================================
# Unified HTTP + WebSocket Server (aiohttp-based)
# =============================================================================


class AiohttpUnifiedServer(ServerBase, StreamAPIHandlersMixin):  # type: ignore[misc]
    """
    Unified server using aiohttp to handle both HTTP API and WebSocket on a single port.

    This is the recommended server for production as it avoids CORS issues with
    separate ports for HTTP and WebSocket.

    Inherits common functionality from ServerBase (rate limiting, state caching)
    and HTTP API handlers from StreamAPIHandlersMixin.

    Usage:
        server = AiohttpUnifiedServer(port=8080, nomic_dir=Path(".nomic"))
        await server.start()
    """

    def __init__(
        self,
        port: int = 8080,
        host: str = "0.0.0.0",
        nomic_dir: Optional[Path] = None,
    ):
        # Initialize base class with common functionality
        super().__init__()

        self.port = port
        self.host = host
        self.nomic_dir = nomic_dir

        # ArgumentCartographer registry - Lock hierarchy level 4 (acquire last)
        self.cartographers: Dict[str, Any] = {}
        self._cartographers_lock = threading.Lock()

        # Optional stores (initialized from nomic_dir)
        self.elo_system = None
        self.insight_store = None
        self.flip_detector = None
        self.persona_manager = None
        self.debate_embeddings = None

        # Initialize stores from nomic_dir
        if nomic_dir:
            self._init_stores(nomic_dir)

    def _init_stores(self, nomic_dir: Path) -> None:
        """Initialize optional stores from nomic directory."""
        # EloSystem for leaderboard
        try:
            from aragora.ranking.elo import EloSystem

            elo_path = nomic_dir / "agent_elo.db"
            if elo_path.exists():
                self.elo_system = EloSystem(str(elo_path))
                logger.info("[server] EloSystem loaded")
        except ImportError:
            logger.debug("[server] EloSystem not available (optional dependency)")

        # InsightStore for insights
        try:
            from aragora.insights.store import InsightStore

            insights_path = nomic_dir / DB_INSIGHTS_PATH
            if insights_path.exists():
                self.insight_store = InsightStore(str(insights_path))
                logger.info("[server] InsightStore loaded")
        except ImportError:
            logger.debug("[server] InsightStore not available (optional dependency)")

        # FlipDetector for position reversals
        try:
            from aragora.insights.flip_detector import FlipDetector

            positions_path = nomic_dir / "aragora_positions.db"
            if positions_path.exists():
                self.flip_detector = FlipDetector(str(positions_path))
                logger.info("[server] FlipDetector loaded")
        except ImportError:
            logger.debug("[server] FlipDetector not available (optional dependency)")

        # PersonaManager for agent specialization
        try:
            from aragora.personas.manager import PersonaManager

            personas_path = nomic_dir / DB_PERSONAS_PATH
            if personas_path.exists():
                self.persona_manager = PersonaManager(str(personas_path))
                logger.info("[server] PersonaManager loaded")
        except ImportError:
            logger.debug("[server] PersonaManager not available (optional dependency)")

        # DebateEmbeddingsDatabase for memory
        try:
            from aragora.debate.embeddings import DebateEmbeddingsDatabase

            embeddings_path = nomic_dir / "debate_embeddings.db"
            if embeddings_path.exists():
                self.debate_embeddings = DebateEmbeddingsDatabase(str(embeddings_path))
                logger.info("[server] DebateEmbeddings loaded")
        except ImportError:
            logger.debug("[server] DebateEmbeddings not available (optional dependency)")

    def _cleanup_stale_entries(self) -> None:
        """Remove stale entries from all tracking dicts.

        Delegates to ServerBase.cleanup_all().
        """
        results = self.cleanup_all()
        total = sum(results.values())
        if total > 0:
            logger.debug(f"Cleaned up {total} stale entries")

    def _update_debate_state(self, event: StreamEvent) -> None:  # type: ignore[override]
        """Update cached debate state based on emitted events.

        Overrides ServerBase._update_debate_state with StreamEvent-specific handling.
        """
        loop_id = event.loop_id
        with self._debate_states_lock:
            if event.type == StreamEventType.DEBATE_START:
                # Enforce max size with LRU eviction (only evict ended debates)
                if len(self.debate_states) >= self.config.max_debate_states:
                    ended_states = [
                        (k, self._debate_states_last_access.get(k, 0))
                        for k, v in self.debate_states.items()
                        if v.get("ended")
                    ]
                    if ended_states:
                        oldest = min(ended_states, key=lambda x: x[1])[0]
                        self.debate_states.pop(oldest, None)
                        self._debate_states_last_access.pop(oldest, None)
                self.debate_states[loop_id] = {
                    "id": loop_id,
                    "task": event.data.get("task"),
                    "agents": event.data.get("agents"),
                    "started_at": event.timestamp,
                    "ended": False,
                }
                self._debate_states_last_access[loop_id] = time.time()
            elif event.type == StreamEventType.DEBATE_END:
                if loop_id in self.debate_states:
                    self.debate_states[loop_id]["ended"] = True
                    self._debate_states_last_access[loop_id] = time.time()
            elif event.type == StreamEventType.LOOP_UNREGISTER:
                self.debate_states.pop(loop_id, None)
                self._debate_states_last_access.pop(loop_id, None)

    def register_loop(self, loop_id: str, name: str, path: str = "") -> None:
        """Register a new nomic loop instance."""
        # Trigger periodic cleanup using base class config
        self._rate_limiter_cleanup_counter += 1
        if self._rate_limiter_cleanup_counter >= self.config.rate_limiter_cleanup_interval:
            self._rate_limiter_cleanup_counter = 0
            self._cleanup_stale_entries()

        instance = LoopInstance(
            loop_id=loop_id,
            name=name,
            started_at=time.time(),
            path=path,
        )
        # Use base class method for active loop management
        self.set_active_loop(loop_id, instance)
        # Broadcast loop registration
        self._emitter.emit(
            StreamEvent(
                type=StreamEventType.LOOP_REGISTER,
                data={
                    "loop_id": loop_id,
                    "name": name,
                    "started_at": instance.started_at,
                    "path": path,
                },
                loop_id=loop_id,
            )
        )

    def unregister_loop(self, loop_id: str) -> None:
        """Unregister a nomic loop instance."""
        self.remove_active_loop(loop_id)
        # Also cleanup associated cartographer to prevent memory leak
        self.unregister_cartographer(loop_id)
        # Broadcast loop unregistration
        self._emitter.emit(
            StreamEvent(
                type=StreamEventType.LOOP_UNREGISTER,
                data={"loop_id": loop_id},
                loop_id=loop_id,
            )
        )

    def update_loop_state(
        self, loop_id: str, cycle: Optional[int] = None, phase: Optional[str] = None
    ) -> None:
        """Update loop state (cycle/phase)."""
        with self._active_loops_lock:
            if loop_id in self.active_loops:
                if cycle is not None:
                    self.active_loops[loop_id].cycle = cycle
                if phase is not None:
                    self.active_loops[loop_id].phase = phase

    def register_cartographer(self, loop_id: str, cartographer: Any) -> None:
        """Register an ArgumentCartographer instance for a loop."""
        with self._cartographers_lock:
            self.cartographers[loop_id] = cartographer

    def unregister_cartographer(self, loop_id: str) -> None:
        """Unregister an ArgumentCartographer instance."""
        with self._cartographers_lock:
            self.cartographers.pop(loop_id, None)

    def _get_loops_data(self) -> list[dict]:
        """Get serializable list of active loops. Thread-safe."""
        with self._active_loops_lock:
            return [
                {
                    "loop_id": loop.loop_id,
                    "name": loop.name,
                    "started_at": loop.started_at,
                    "cycle": loop.cycle,
                    "phase": loop.phase,
                    "path": loop.path,
                }
                for loop in self.active_loops.values()
            ]

    def _cors_headers(self, origin: Optional[str] = None) -> dict:
        """Generate CORS headers with proper origin validation.

        Only allows origins in the whitelist. Does NOT fallback to first
        origin for unauthorized requests (that would be a security issue).
        """
        headers = {
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400",
        }
        # Only add Allow-Origin for whitelisted origins or same-origin requests
        if origin and origin in WS_ALLOWED_ORIGINS:
            headers["Access-Control-Allow-Origin"] = origin
        elif not origin:
            # Same-origin request - allow with wildcard
            headers["Access-Control-Allow-Origin"] = "*"
        # For unauthorized origins, don't add Allow-Origin (browser will block)
        return headers

    async def _check_usage_limit(self, headers: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Check if user has remaining debate quota.

        Returns None if within limits, or error dict if limit exceeded.
        """
        try:
            from aragora.billing.jwt_auth import validate_access_token
            from aragora.billing.usage import UsageTracker
            from aragora.storage import UserStore

            # Extract JWT from Authorization header
            auth_header = headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return None  # No JWT, skip check

            token = auth_header[7:]
            if token.startswith("ara_"):
                return None  # API key, skip JWT-based check

            # Validate JWT and get payload (returns JWTPayload dataclass)
            payload = validate_access_token(token)
            if not payload:
                return None  # Invalid token, skip check

            org_id = payload.org_id
            if not org_id:
                return None  # No org in token, skip check

            # Require nomic_dir for UserStore initialization
            if not self.nomic_dir:
                return None

            user_store = UserStore(self.nomic_dir / "users.db")
            org = user_store.get_organization_by_id(org_id)
            if not org:
                return None

            # Get usage for current period
            tracker = UsageTracker()
            usage = tracker.get_summary(org.id)

            # Check tier limits
            tier_limits = {
                "free": 10,
                "starter": 50,
                "professional": 200,
                "enterprise": 999999,
            }
            tier_value = org.tier.value if hasattr(org.tier, "value") else str(org.tier)
            limit = tier_limits.get(tier_value, 10)
            debates_used = usage.total_debates

            if debates_used >= limit:
                return {
                    "error": "Debate limit reached for this billing period",
                    "debates_used": debates_used,
                    "debates_limit": limit,
                    "tier": tier_value,
                    "upgrade_url": "/pricing",
                }

            return None

        except ImportError:
            # Billing module not available, skip check
            return None
        except Exception as e:
            logger.debug(f"Usage limit check failed: {e}")
            return None  # Fail open to not block debates

    # NOTE: HTTP API handlers (_handle_options, _handle_leaderboard, etc.)
    # are provided by StreamAPIHandlersMixin from stream_handlers.py

    def _parse_debate_request(self, data: dict) -> tuple[Optional[dict], Optional[str]]:
        """Parse and validate debate request data.

        Returns:
            Tuple of (parsed_config, error_message). If error_message is set,
            parsed_config will be None.
        """
        # Validate required fields with length limits
        question = data.get("question", "").strip()
        if not question:
            return None, "question field is required"
        if len(question) > 10000:
            return None, "question must be under 10,000 characters"

        # Parse optional fields with validation
        agents_str = data.get("agents", "anthropic-api,openai-api,gemini,grok")
        try:
            rounds = min(max(int(data.get("rounds", 3)), 1), 10)  # Clamp to 1-10
        except (ValueError, TypeError):
            rounds = 3
        consensus = data.get("consensus", "majority")

        return {
            "question": question,
            "agents_str": agents_str,
            "rounds": rounds,
            "consensus": consensus,
            "use_trending": data.get("use_trending", False),
            "trending_category": data.get("trending_category", None),
        }, None

    async def _fetch_trending_topic_async(self, category: Optional[str] = None) -> Optional[Any]:
        """Fetch a trending topic for the debate.

        Returns:
            A TrendingTopic object or None if unavailable.
        """
        try:
            from aragora.pulse.ingestor import (
                PulseManager,
                TwitterIngestor,
                HackerNewsIngestor,
                RedditIngestor,
            )

            manager = PulseManager()
            manager.add_ingestor("twitter", TwitterIngestor())
            manager.add_ingestor("hackernews", HackerNewsIngestor())
            manager.add_ingestor("reddit", RedditIngestor())

            filters = {}
            if category:
                filters["categories"] = [category]

            topics = await manager.get_trending_topics(
                limit_per_platform=3, filters=filters if filters else None
            )
            topic = manager.select_topic_for_debate(topics)

            if topic:
                logger.info(f"Selected trending topic: {topic.topic}")
            return topic
        except Exception as e:
            logger.warning(f"Trending topic fetch failed (non-fatal): {e}")
            return None

    def _execute_debate_thread(
        self,
        debate_id: str,
        question: str,
        agents_str: str,
        rounds: int,
        consensus: str,
        trending_topic: Optional[Any],
        user_id: str = "",
        org_id: str = "",
    ) -> None:
        """Execute a debate in a background thread.

        This method is run in a ThreadPoolExecutor to avoid blocking the event loop.
        """
        import asyncio as _asyncio

        try:
            # Parse agents with bounds check
            agent_list = [s.strip() for s in agents_str.split(",") if s.strip()]
            if len(agent_list) > MAX_AGENTS_PER_DEBATE:
                with _active_debates_lock:
                    _active_debates[debate_id]["status"] = "error"
                    _active_debates[debate_id][
                        "error"
                    ] = f"Too many agents. Maximum: {MAX_AGENTS_PER_DEBATE}"
                    _active_debates[debate_id]["completed_at"] = time.time()
                return
            if len(agent_list) < 2:
                with _active_debates_lock:
                    _active_debates[debate_id]["status"] = "error"
                    _active_debates[debate_id]["error"] = "At least 2 agents required for a debate"
                    _active_debates[debate_id]["completed_at"] = time.time()
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
                    raise ValueError(
                        f"Invalid agent type: {agent_type}. Allowed: {', '.join(sorted(ALLOWED_AGENT_TYPES))}"
                    )
                agent_specs.append((agent_type, role))

            # Create agents with streaming support
            # All agents are proposers for full participation in all rounds
            agents: list[Agent] = []
            for i, (agent_type, role) in enumerate(agent_specs):
                if role is None:
                    role = "proposer"  # All agents propose and participate fully
                agent = create_agent(
                    model_type=agent_type,  # type: ignore[arg-type]
                    name=f"{agent_type}_{role}",
                    role=role,
                )
                # Wrap agent for token streaming if supported
                agent = _wrap_agent_for_streaming(agent, self.emitter, debate_id)
                agents.append(agent)

            # Create environment and protocol
            env = Environment(task=question, context="", max_rounds=rounds)
            protocol = DebateProtocol(
                rounds=rounds,
                consensus=consensus,  # type: ignore[arg-type]
                proposer_count=len(agents),  # All agents propose initially
                topology="all-to-all",  # Everyone critiques everyone
            )

            # Create arena with hooks and available context systems
            hooks = create_arena_hooks(self.emitter)

            # Initialize usage tracking if user/org context is available
            usage_tracker = None
            if user_id or org_id:
                try:
                    from aragora.billing.usage import UsageTracker

                    usage_tracker = UsageTracker()
                except ImportError:
                    pass

            arena = Arena(
                env,
                agents,
                protocol,  # type: ignore[arg-type]
                event_hooks=hooks,
                event_emitter=self.emitter,
                loop_id=debate_id,
                trending_topic=trending_topic,
                user_id=user_id,
                org_id=org_id,
                usage_tracker=usage_tracker,
            )

            # Run debate with timeout protection
            # Use protocol timeout if configured, otherwise use global default
            protocol_timeout = getattr(arena.protocol, "timeout_seconds", 0)
            timeout = (
                protocol_timeout
                if isinstance(protocol_timeout, (int, float)) and protocol_timeout > 0
                else DEBATE_TIMEOUT_SECONDS
            )
            with _active_debates_lock:
                _active_debates[debate_id]["status"] = "running"

            async def run_with_timeout():
                return await _asyncio.wait_for(arena.run(), timeout=timeout)

            result = _asyncio.run(run_with_timeout())
            with _active_debates_lock:
                _active_debates[debate_id]["status"] = "completed"
                _active_debates[debate_id]["completed_at"] = time.time()
                _active_debates[debate_id]["result"] = {
                    "final_answer": result.final_answer,
                    "consensus_reached": result.consensus_reached,
                    "confidence": result.confidence,
                }

        except Exception as e:
            import traceback

            safe_msg = _safe_error_message(e, "debate_execution")
            error_trace = traceback.format_exc()
            with _active_debates_lock:
                _active_debates[debate_id]["status"] = "error"
                _active_debates[debate_id]["completed_at"] = time.time()
                _active_debates[debate_id]["error"] = safe_msg
            logger.error(f"[debate] Thread error in {debate_id}: {str(e)}\n{error_trace}")
            # Emit error event to client
            self.emitter.emit(
                StreamEvent(
                    type=StreamEventType.ERROR,
                    data={"error": safe_msg, "debate_id": debate_id},
                )
            )

    async def _handle_start_debate(self, request) -> "aiohttp.web.Response":
        """POST /api/debate - Start an ad-hoc debate with specified question.

        Accepts JSON body with:
            question: The topic/question to debate (required)
            agents: Comma-separated agent list (optional, default: "anthropic-api,openai-api,gemini,grok")
            rounds: Number of debate rounds (optional, default: 3)
            consensus: Consensus method (optional, default: "majority")
            use_trending: If true, fetch a trending topic to seed the debate (optional)
            trending_category: Filter trending topics by category (optional)

        All agents participate as proposers for full participation in all rounds.

        Requires authentication when ARAGORA_API_TOKEN is set.
        """
        global _active_debates, _debate_executor
        import aiohttp.web as web
        from aragora.server.auth import auth_config, check_auth

        origin = request.headers.get("Origin")

        # Authenticate if auth is enabled (starting debates uses compute resources)
        if auth_config.enabled:
            headers = dict(request.headers)
            client_ip = request.remote or ""
            if client_ip in TRUSTED_PROXIES:
                forwarded = request.headers.get("X-Forwarded-For", "")
                if forwarded:
                    client_ip = forwarded.split(",")[0].strip()

            authenticated, remaining = check_auth(headers, "", loop_id="", ip_address=client_ip)
            if not authenticated:
                status = 429 if remaining == 0 else 401
                msg = (
                    "Rate limit exceeded"
                    if remaining == 0
                    else "Authentication required to start debates"
                )
                return web.json_response(
                    {"error": msg}, status=status, headers=self._cors_headers(origin)
                )

            # Check usage limits for authenticated users
            usage_error = await self._check_usage_limit(headers)
            if usage_error:
                return web.json_response(
                    usage_error, status=402, headers=self._cors_headers(origin)
                )

        if not DEBATE_AVAILABLE:
            return web.json_response(
                {"error": "Debate orchestrator not available"},
                status=500,
                headers=self._cors_headers(origin),
            )

        # Parse JSON body
        try:
            data = await request.json()
        except Exception as e:
            logger.debug(f"Invalid JSON in request: {e}")
            return web.json_response(
                {"error": "Invalid JSON"}, status=400, headers=self._cors_headers(origin)
            )

        # Parse and validate request
        config, error = self._parse_debate_request(data)
        if error:
            return web.json_response(
                {"error": error}, status=400, headers=self._cors_headers(origin)
            )

        question = config["question"]
        agents_str = config["agents_str"]
        rounds = config["rounds"]
        consensus = config["consensus"]

        # Fetch trending topic if requested
        trending_topic = None
        if config["use_trending"]:
            trending_topic = await self._fetch_trending_topic_async(config["trending_category"])

        # Extract user/org context from JWT for usage tracking
        user_id = ""
        org_id = ""
        try:
            from aragora.billing.jwt_auth import validate_access_token

            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer ") and not auth_header[7:].startswith("ara_"):
                payload = validate_access_token(auth_header[7:])
                if payload:
                    # JWTPayload is a dataclass, use attribute access not dict
                    user_id = payload.sub or ""
                    org_id = payload.org_id or ""
        except ImportError:
            pass

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
        if increment_cleanup_counter():
            _cleanup_stale_debates_stream()

        # Set loop_id on emitter so events are tagged
        self.emitter.set_loop_id(debate_id)

        # Use thread pool to prevent unbounded thread creation
        executor = get_debate_executor()
        with _debate_executor_lock:
            if executor is None:
                executor = ThreadPoolExecutor(
                    max_workers=MAX_CONCURRENT_DEBATES, thread_name_prefix="debate-"
                )
                set_debate_executor(executor)

        try:
            executor.submit(
                self._execute_debate_thread,
                debate_id,
                question,
                agents_str,
                rounds,
                consensus,
                trending_topic,
                user_id,
                org_id,
            )
        except RuntimeError:
            return web.json_response(
                {
                    "success": False,
                    "error": "Server at capacity. Please try again later.",
                },
                status=503,
                headers=self._cors_headers(origin),
            )

        # Return immediately with debate ID
        return web.json_response(
            {
                "success": True,
                "debate_id": debate_id,
                "question": question,
                "agents": agents_str.split(","),
                "rounds": rounds,
                "status": "starting",
                "message": "Debate started. Connect to WebSocket to receive events.",
            },
            headers=self._cors_headers(origin),
        )

    def _validate_audience_payload(self, data: dict) -> tuple[Optional[dict], Optional[str]]:
        """Validate audience message payload.

        Returns:
            Tuple of (validated_payload, error_message). If error, payload is None.
        """
        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            return None, "Invalid payload format"

        # Limit payload size to 10KB (DoS protection)
        try:
            payload_str = json.dumps(payload)
            if len(payload_str) > 10240:
                return None, "Payload too large (max 10KB)"
        except (TypeError, ValueError):
            return None, "Invalid payload structure"

        return payload, None

    async def _validate_ws_auth_for_write(
        self,
        ws_id: int,
        ws: Any,
    ) -> tuple[bool, Optional[dict]]:
        """Validate WebSocket authentication for write operations.

        Returns:
            Tuple of (is_authorized, error_response). If not authorized, error_response
            contains the JSON response to send to the client.
        """
        try:
            from aragora.server.auth import auth_config

            if not auth_config.enabled:
                return True, None

            # Check basic authentication
            if not self.is_ws_authenticated(ws_id):
                return False, {
                    "type": "error",
                    "data": {
                        "message": "Authentication required for voting/suggestions",
                        "code": 401,
                    },
                }

            # Periodic token revalidation for long-lived connections
            if self.should_revalidate_ws_token(ws_id):
                stored_token = self.get_ws_token(ws_id)
                if stored_token and not auth_config.validate_token(stored_token):
                    self.revoke_ws_auth(ws_id, "Token expired or revoked")
                    return False, {
                        "type": "auth_revoked",
                        "data": {"message": "Token has been revoked or expired", "code": 401},
                    }
                self.mark_ws_token_validated(ws_id)

            return True, None
        except ImportError:
            return True, None  # Auth module not available

    def _validate_loop_id_access(
        self,
        ws_id: int,
        loop_id: str,
    ) -> tuple[bool, Optional[dict]]:
        """Validate loop_id exists and client has access.

        Returns:
            Tuple of (is_valid, error_response). If not valid, error_response
            contains the JSON response to send to the client.
        """
        # Validate loop_id exists and is active
        with self._active_loops_lock:
            loop_valid = loop_id and loop_id in self.active_loops

        if not loop_valid:
            return False, {
                "type": "error",
                "data": {"message": f"Invalid or inactive loop_id: {loop_id}"},
            }

        # Validate token is authorized for this specific loop_id
        try:
            from aragora.server.auth import auth_config

            if auth_config.enabled:
                stored_token = self.get_ws_token(ws_id)
                if stored_token:
                    is_valid, err_msg = auth_config.validate_token_for_loop(stored_token, loop_id)
                    if not is_valid:
                        return False, {"type": "error", "data": {"message": err_msg, "code": 403}}
        except ImportError:
            pass

        return True, None

    def _check_audience_rate_limit(
        self,
        client_id: str,
    ) -> tuple[bool, Optional[dict]]:
        """Check rate limit for audience messages.

        Returns:
            Tuple of (is_allowed, error_response). If not allowed, error_response
            contains the JSON response to send to the client.
        """
        with self._rate_limiters_lock:
            self._rate_limiter_last_access[client_id] = time.time()
            rate_limiter = self._rate_limiters.get(client_id)

        if rate_limiter is None or not rate_limiter.consume(1):
            return False, {
                "type": "error",
                "data": {"message": "Rate limit exceeded, try again later"},
            }

        return True, None

    def _process_audience_message(
        self,
        msg_type: str,
        loop_id: str,
        payload: dict,
        client_id: str,
    ) -> None:
        """Process validated audience vote/suggestion message."""
        audience_msg = AudienceMessage(
            type="vote" if msg_type == "user_vote" else "suggestion",
            loop_id=loop_id,
            payload=payload,
            user_id=client_id,
        )
        self.audience_inbox.put(audience_msg)

        # Emit event for dashboard visibility
        event_type = (
            StreamEventType.USER_VOTE
            if msg_type == "user_vote"
            else StreamEventType.USER_SUGGESTION
        )
        self._emitter.emit(
            StreamEvent(
                type=event_type,
                data=audience_msg.payload,
                loop_id=loop_id,
            )
        )

        # Emit updated audience metrics after each vote
        if msg_type == "user_vote":
            metrics = self.audience_inbox.get_summary(loop_id=loop_id)
            self._emitter.emit(
                StreamEvent(
                    type=StreamEventType.AUDIENCE_METRICS,
                    data=metrics,
                    loop_id=loop_id,
                )
            )

    async def _websocket_handler(self, request) -> "aiohttp.web.StreamResponse":
        """Handle WebSocket connections with security validation and optional auth."""
        import aiohttp
        import aiohttp.web as web

        # Validate origin for security (match websockets handler behavior)
        origin = request.headers.get("Origin", "")
        if origin and origin not in WS_ALLOWED_ORIGINS:
            # Reject connection from unauthorized origin
            return web.Response(status=403, text="Origin not allowed")

        # Extract client IP (validate proxy headers for security)
        remote_ip = request.remote or ""
        client_ip = remote_ip  # Default to direct connection IP
        if remote_ip in TRUSTED_PROXIES:
            # Only trust X-Forwarded-For from trusted proxies
            forwarded = request.headers.get("X-Forwarded-For", "")
            if forwarded:
                first_ip = forwarded.split(",")[0].strip()
                if first_ip:
                    client_ip = first_ip

        # Extract token for authentication tracking
        is_authenticated = False
        ws_token = None

        # Optional authentication (controlled by ARAGORA_API_TOKEN env var)
        try:
            from aragora.server.auth import auth_config, check_auth

            # Convert headers to dict for check_auth
            headers = dict(request.headers)

            # Extract token from Authorization header for tracking
            auth_header = headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                ws_token = auth_header[7:]

            if auth_config.enabled:
                authenticated, remaining = check_auth(headers, "", loop_id="", ip_address=client_ip)

                if not authenticated:
                    status = 429 if remaining == 0 else 401
                    error_msg = (
                        "Rate limit exceeded" if remaining == 0 else "Authentication required"
                    )
                    return web.Response(status=status, text=error_msg)

                is_authenticated = True
            else:
                # Auth disabled - still track token if provided for optional validation
                is_authenticated = True  # Everyone is "authenticated" when auth is disabled
        except ImportError:
            # Log warning if auth is required but module unavailable
            if os.getenv("ARAGORA_AUTH_REQUIRED"):
                logger.warning("[ws] Auth required but module unavailable - rejecting connection")
                return web.Response(status=500, text="Authentication system unavailable")
            is_authenticated = True  # Auth module not available, allow connection

        # Enable permessage-deflate compression for reduced bandwidth
        # compress=15 uses 15-bit window (32KB) for good compression ratio
        ws = web.WebSocketResponse(
            max_msg_size=WS_MAX_MESSAGE_SIZE,
            compress=True,  # Enable permessage-deflate compression
        )
        await ws.prepare(request)

        # Initialize tracking variables before any operations that could fail
        ws_id = id(ws)
        client_id = secrets.token_hex(16)
        self.clients.add(ws)
        # Enforce max size with LRU eviction
        if len(self._client_ids) >= self.config.max_client_ids:
            self._client_ids.popitem(last=False)  # Remove oldest
        self._client_ids[ws_id] = client_id

        # Track authentication state using ServerBase method
        self.set_ws_auth_state(
            ws_id=ws_id,
            authenticated=is_authenticated,
            token=ws_token,
            ip_address=client_ip,
        )

        # Initialize rate limiter for this client (thread-safe)
        with self._rate_limiters_lock:
            self._rate_limiters[client_id] = TokenBucket(
                rate_per_minute=10.0, burst_size=5  # 10 messages per minute  # Allow burst of 5
            )
            self._rate_limiter_last_access[client_id] = time.time()

        logger.info(
            f"[ws] Client {client_id[:8]}... connected from {client_ip} "
            f"(authenticated={is_authenticated}, total_clients={len(self.clients)})"
        )

        # Send connection info including auth status
        try:
            from aragora.server.auth import auth_config as _auth_config

            write_access = is_authenticated or not _auth_config.enabled
        except ImportError:
            write_access = True

        await ws.send_json(
            {
                "type": "connection_info",
                "data": {
                    "authenticated": is_authenticated,
                    "client_id": client_id[:8] + "...",  # Partial for privacy
                    "write_access": write_access,
                },
            }
        )

        # Send initial loop list
        loops_data = self._get_loops_data()
        await ws.send_json(
            {
                "type": "loop_list",
                "data": {"loops": loops_data, "count": len(loops_data)},
            }
        )

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:  # type: ignore[union-attr]
                    # Defense-in-depth: check message size before parsing
                    msg_data = msg.data  # type: ignore[union-attr]
                    if len(msg_data) > WS_MAX_MESSAGE_SIZE:
                        logger.warning(
                            f"[ws] Message too large: {len(msg_data)} bytes (max {WS_MAX_MESSAGE_SIZE})"
                        )
                        await ws.send_json(
                            {
                                "type": "error",
                                "data": {
                                    "code": "MESSAGE_TOO_LARGE",
                                    "message": f"Message exceeds {WS_MAX_MESSAGE_SIZE} byte limit",
                                },
                            }
                        )
                        continue

                    try:
                        data = json.loads(msg_data)
                        msg_type = data.get("type")

                        if msg_type == "get_loops":
                            loops_data = self._get_loops_data()
                            await ws.send_json(
                                {
                                    "type": "loop_list",
                                    "data": {"loops": loops_data, "count": len(loops_data)},
                                }
                            )

                        elif msg_type in ("user_vote", "user_suggestion"):
                            # Validate authentication for write operations
                            is_auth, auth_error = await self._validate_ws_auth_for_write(ws_id, ws)
                            if not is_auth:
                                await ws.send_json(auth_error)
                                continue

                            # Get loop_id (use ws-bound as fallback for proprioceptive socket)
                            loop_id = data.get("loop_id") or getattr(ws, "_bound_loop_id", "")

                            # Optional per-message token validation
                            msg_token = data.get("token")
                            if msg_token:
                                try:
                                    from aragora.server.auth import auth_config

                                    if not auth_config.validate_token(msg_token, loop_id):
                                        await ws.send_json(
                                            {
                                                "type": "error",
                                                "data": {
                                                    "code": "AUTH_FAILED",
                                                    "message": "Invalid or revoked token",
                                                },
                                            }
                                        )
                                        continue
                                except ImportError:
                                    pass

                            # Validate loop_id and access
                            is_valid, loop_error = self._validate_loop_id_access(ws_id, loop_id)
                            if not is_valid:
                                await ws.send_json(loop_error)
                                continue

                            # Bind loop_id to WebSocket for future reference (proprioceptive socket)
                            ws._bound_loop_id = loop_id  # type: ignore[attr-defined]

                            # Validate payload
                            payload, error = self._validate_audience_payload(data)
                            if error:
                                await ws.send_json({"type": "error", "data": {"message": error}})
                                continue

                            # Check rate limit
                            is_allowed, rate_error = self._check_audience_rate_limit(client_id)
                            if not is_allowed:
                                await ws.send_json(rate_error)
                                continue

                            # Process the message
                            self._process_audience_message(msg_type, loop_id, payload, client_id)
                            await ws.send_json(
                                {
                                    "type": "ack",
                                    "data": {"message": "Message received", "msg_type": msg_type},
                                }
                            )

                    except json.JSONDecodeError as e:
                        logger.warning(f"[ws] Invalid JSON: {e.msg} at pos {e.pos}")
                        await ws.send_json(
                            {
                                "type": "error",
                                "data": {
                                    "code": "INVALID_JSON",
                                    "message": f"JSON parse error: {e.msg}",
                                },
                            }
                        )

                elif msg.type == aiohttp.WSMsgType.ERROR:  # type: ignore[union-attr]
                    logger.error(f"[ws] Error: {ws.exception()}")
                    break

                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):  # type: ignore[union-attr]
                    logger.debug(f"[ws] Client {client_id[:8]}... closed connection")
                    break

                elif msg.type == aiohttp.WSMsgType.BINARY:  # type: ignore[union-attr]
                    logger.warning(f"[ws] Binary message rejected from {client_id[:8]}...")
                    await ws.send_json(
                        {
                            "type": "error",
                            "data": {
                                "code": "BINARY_NOT_SUPPORTED",
                                "message": "Binary messages not supported",
                            },
                        }
                    )

                # PING/PONG handled automatically by aiohttp, but log if we see them
                elif msg.type in (aiohttp.WSMsgType.PING, aiohttp.WSMsgType.PONG):  # type: ignore[union-attr]
                    pass  # Handled by aiohttp automatically

                else:
                    logger.warning(f"[ws] Unhandled message type: {msg.type}")  # type: ignore[union-attr]

        finally:
            self.clients.discard(ws)
            self._client_ids.pop(ws_id, None)
            # Clean up rate limiter for this client (thread-safe)
            with self._rate_limiters_lock:
                self._rate_limiters.pop(client_id, None)
                self._rate_limiter_last_access.pop(client_id, None)
            # Clean up auth state
            self.remove_ws_auth_state(ws_id)
            logger.info(
                f"[ws] Client {client_id[:8]}... disconnected from {client_ip} "
                f"(remaining_clients={len(self.clients)})"
            )

        return ws

    async def _drain_loop(self) -> None:
        """Drain events from the sync emitter and broadcast to WebSocket clients."""
        import aiohttp

        while self._running:
            try:
                event = self._emitter._queue.get(timeout=0.1)

                # Update loop state for cycle/phase events
                if event.type == StreamEventType.CYCLE_START:
                    self.update_loop_state(event.loop_id, cycle=event.data.get("cycle"))
                elif event.type == StreamEventType.PHASE_START:
                    self.update_loop_state(event.loop_id, phase=event.data.get("phase"))

                # Serialize event
                event_dict = {
                    "type": event.type.value,
                    "data": event.data,
                    "timestamp": event.timestamp,
                    "round": event.round,
                    "agent": event.agent,
                    "loop_id": event.loop_id,
                }
                message = json.dumps(event_dict)

                # Broadcast to all clients
                dead_clients = []
                for client in list(self.clients):
                    try:
                        await client.send_str(message)
                    except Exception as e:
                        logger.debug(
                            "WebSocket client disconnected during broadcast: %s", type(e).__name__
                        )
                        dead_clients.append(client)

                if dead_clients:
                    logger.info("Removed %d dead WebSocket client(s)", len(dead_clients))
                    for client in dead_clients:
                        self.clients.discard(client)

            except queue.Empty:
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"[ws] Drain loop error: {e}")
                await asyncio.sleep(0.1)

    def _add_versioned_routes(self, app, routes: list) -> None:
        """Add both versioned (/api/v1/) and legacy (/api/) routes.

        This enables API versioning while maintaining backwards compatibility.
        Routes registered:
        - /api/v1/{path} - Versioned (preferred)
        - /api/{path}    - Legacy (deprecated, for backwards compatibility)

        Args:
            app: aiohttp Application
            routes: List of (method, path, handler) tuples where path is without prefix
        """
        for method, path, handler in routes:
            # Add versioned route (preferred)
            v1_path = f"/api/v1{path}"
            # Add legacy route (backwards compatible)
            legacy_path = f"/api{path}"

            if method == "GET":
                app.router.add_get(v1_path, handler)
                app.router.add_get(legacy_path, handler)
            elif method == "POST":
                app.router.add_post(v1_path, handler)
                app.router.add_post(legacy_path, handler)
            elif method == "PUT":
                app.router.add_put(v1_path, handler)
                app.router.add_put(legacy_path, handler)
            elif method == "DELETE":
                app.router.add_delete(v1_path, handler)
                app.router.add_delete(legacy_path, handler)

    async def start(self) -> None:
        """Start the unified HTTP+WebSocket server."""
        import aiohttp.web as web

        # Initialize error monitoring (no-op if SENTRY_DSN not set)
        try:
            from aragora.server.error_monitoring import init_monitoring

            if init_monitoring():
                logger.info("Error monitoring enabled (Sentry)")
        except ImportError:
            pass  # sentry-sdk not installed

        self._running = True

        # Create aiohttp app
        app = web.Application()

        # Add OPTIONS handler for CORS preflight
        app.router.add_route("OPTIONS", "/{path:.*}", self._handle_options)

        # Define API routes (path suffix after /api or /api/v1)
        api_routes = [
            ("GET", "/leaderboard", self._handle_leaderboard),
            ("GET", "/matches/recent", self._handle_matches_recent),
            ("GET", "/insights/recent", self._handle_insights_recent),
            ("GET", "/flips/summary", self._handle_flips_summary),
            ("GET", "/flips/recent", self._handle_flips_recent),
            ("GET", "/tournaments", self._handle_tournaments),
            ("GET", "/tournaments/{tournament_id}", self._handle_tournament_details),
            ("GET", "/agent/{name}/consistency", self._handle_agent_consistency),
            ("GET", "/agent/{name}/network", self._handle_agent_network),
            ("GET", "/memory/tier-stats", self._handle_memory_tier_stats),
            ("GET", "/laboratory/emergent-traits", self._handle_laboratory_emergent_traits),
            (
                "GET",
                "/laboratory/cross-pollinations/suggest",
                self._handle_laboratory_cross_pollinations,
            ),
            ("GET", "/health", self._handle_health),
            ("GET", "/nomic/state", self._handle_nomic_state),
            ("GET", "/debate/{loop_id}/graph", self._handle_graph_json),
            ("GET", "/debate/{loop_id}/graph/mermaid", self._handle_graph_mermaid),
            ("GET", "/debate/{loop_id}/graph/stats", self._handle_graph_stats),
            ("GET", "/debate/{loop_id}/audience/clusters", self._handle_audience_clusters),
            ("GET", "/replays", self._handle_replays),
            ("GET", "/replays/{replay_id}/html", self._handle_replay_html),
            ("POST", "/debate", self._handle_start_debate),
        ]

        # Add routes with both versioned and legacy paths
        self._add_versioned_routes(app, api_routes)

        # WebSocket handlers (not versioned)
        app.router.add_get("/", self._websocket_handler)
        app.router.add_get("/ws", self._websocket_handler)

        # Prometheus metrics endpoint (not under /api/)
        app.router.add_get("/metrics", self._handle_metrics)

        # Start drain loop
        asyncio.create_task(self._drain_loop())

        # Run server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)

        logger.info(f"Unified server (HTTP+WS) running on http://{self.host}:{self.port}")
        logger.info(f"  WebSocket: ws://{self.host}:{self.port}/")
        logger.info(f"  HTTP API:  http://{self.host}:{self.port}/api/v1/* (preferred)")
        logger.info(f"  Legacy:    http://{self.host}:{self.port}/api/* (deprecated)")

        await site.start()

        # Keep running
        try:
            await asyncio.Future()
        finally:
            self._running = False
            await runner.cleanup()

    def stop(self) -> None:
        """Stop the server."""
        self._running = False
