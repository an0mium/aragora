"""
Controller for ad-hoc debate execution.

Handles debate lifecycle orchestration using DebateFactory for creation
and debate_utils for state management. Extracted from unified_server.py
for better modularity and testability.
"""

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from aragora.config import (
    DEBATE_TIMEOUT_SECONDS,
    MAX_CONCURRENT_DEBATES,
)
from aragora.server.debate_factory import DebateConfig, DebateFactory
from aragora.server.debate_utils import (
    _active_debates,
    _active_debates_lock,
    cleanup_stale_debates,
    update_debate_status,
)
from aragora.server.errors import safe_error_message
from aragora.server.http_utils import run_async
from aragora.server.state import get_state_manager
from aragora.server.stream import (
    StreamEvent,
    StreamEventType,
    create_arena_hooks,
    wrap_agent_for_streaming,
)

if TYPE_CHECKING:
    from aragora.server.stream import SyncEventEmitter

logger = logging.getLogger(__name__)


@dataclass
class DebateRequest:
    """Parsed debate request from HTTP body."""

    question: str
    agents_str: str = "anthropic-api,openai-api,gemini,grok"
    rounds: int = 8  # 9-round format (0-8), default for all debates
    consensus: str = "judge"  # Judge-based consensus for final decisions
    debate_format: str = "full"  # "light" (~5 min) or "full" (~30 min)
    auto_select: bool = False
    auto_select_config: dict = None
    use_trending: bool = False
    trending_category: Optional[str] = None

    def __post_init__(self):
        if self.auto_select_config is None:
            self.auto_select_config = {}
        # Normalize debate_format
        if self.debate_format not in ("light", "full"):
            self.debate_format = "full"

    @classmethod
    def from_dict(cls, data: dict) -> "DebateRequest":
        """Create request from parsed JSON data.

        Args:
            data: Parsed JSON dictionary

        Returns:
            DebateRequest instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        question = data.get("question") or data.get("task") or ""
        question = str(question).strip()
        if not question:
            raise ValueError("question or task field is required")
        if len(question) > 10000:
            raise ValueError("question must be under 10,000 characters")

        try:
            rounds = min(max(int(data.get("rounds", 8)), 1), 10)
        except (ValueError, TypeError):
            rounds = 8

        return cls(
            question=question,
            agents_str=data.get("agents", "anthropic-api,openai-api,gemini,grok"),
            rounds=rounds,
            consensus=data.get("consensus", "judge"),
            debate_format=data.get("debate_format", "full"),
            auto_select=data.get("auto_select", False),
            auto_select_config=data.get("auto_select_config", {}),
            use_trending=data.get("use_trending", False),
            trending_category=data.get("trending_category"),
        )


@dataclass
class DebateResponse:
    """Response from debate controller."""

    success: bool
    debate_id: Optional[str] = None
    status: Optional[str] = None
    task: Optional[str] = None
    error: Optional[str] = None
    status_code: int = 200

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {"success": self.success}
        if self.debate_id:
            result["debate_id"] = str(self.debate_id)
        if self.status:
            result["status"] = str(self.status)
        if self.task:
            result["task"] = str(self.task)
        if self.error:
            result["error"] = str(self.error)
        return result


class DebateController:
    """
    Controls debate execution lifecycle.

    Responsibilities:
    - Validates and processes debate requests
    - Coordinates with StateManager for thread pool access
    - Coordinates with DebateFactory for arena creation
    - Handles trending topic integration
    - Manages debate state through debate_utils

    Thread Safety:
        The thread pool is managed by StateManager which handles
        its own locking. All debate state is also managed through
        StateManager.

    Usage:
        controller = DebateController(
            factory=debate_factory,
            emitter=stream_emitter,
            elo_system=elo_system,
        )

        request = DebateRequest.from_dict(json_data)
        response = controller.start_debate(request)
    """

    def __init__(
        self,
        factory: DebateFactory,
        emitter: "SyncEventEmitter",
        elo_system: Optional[Any] = None,
        auto_select_fn: Optional[Callable[..., str]] = None,
        storage: Optional[Any] = None,
    ):
        """Initialize the debate controller.

        Args:
            factory: DebateFactory for creating arenas
            emitter: Event emitter for streaming
            elo_system: Optional ELO system for leaderboard updates
            auto_select_fn: Optional function for auto-selecting agents
            storage: Optional DebateStorage instance for persisting debates
        """
        self.factory = factory
        self.emitter = emitter
        self.elo_system = elo_system
        self.auto_select_fn = auto_select_fn
        self.storage = storage

    def start_debate(self, request: DebateRequest) -> DebateResponse:
        """Start a new debate asynchronously.

        Args:
            request: Validated debate request

        Returns:
            DebateResponse with debate_id on success
        """
        # Validate storage is available for persistence
        if not self.storage:
            logger.error("[debate] Cannot start debate: storage not configured")
            return DebateResponse(
                success=False,
                error="Server storage not configured. Debates cannot be persisted.",
                status_code=503,
            )

        # Generate debate ID
        debate_id = f"adhoc_{uuid.uuid4().hex[:8]}"

        # Resolve agents (auto-select if requested)
        agents_str = request.agents_str
        if request.auto_select and self.auto_select_fn:
            try:
                agents_str = self.auto_select_fn(request.question, request.auto_select_config)
            except Exception as e:
                logger.warning(f"Auto-select failed, using defaults: {e}")

        # Track debate state (use "task" not "question" for StateManager compatibility)
        with _active_debates_lock:
            _active_debates[debate_id] = {
                "id": debate_id,
                "task": request.question,
                "status": "starting",
                "agents": agents_str,
                "rounds": request.rounds,
                "total_rounds": request.rounds,
            }

        # Periodic cleanup
        cleanup_stale_debates()

        # Set loop_id on emitter
        self.emitter.set_loop_id(debate_id)

        # Parse agent names for immediate event (handle both string and list)
        if isinstance(agents_str, str):
            agent_names = [a.strip() for a in agents_str.split(",") if a.strip()]
        else:
            agent_names = list(agents_str) if agents_str else []

        # Emit immediate DEBATE_START event so clients see progress within seconds
        # (The debate phases will emit more detailed events as they execute)
        self.emitter.emit(
            StreamEvent(
                type=StreamEventType.DEBATE_START,
                data={"task": request.question, "agents": agent_names},
                loop_id=debate_id,
            )
        )

        # Emit PHASE_PROGRESS to show research is starting
        # This gives users immediate feedback that something is happening
        self.emitter.emit(
            StreamEvent(
                type=StreamEventType.PHASE_PROGRESS,
                data={
                    "phase": "research",
                    "status": "starting",
                    "message": "Gathering context and researching topic...",
                },
                loop_id=debate_id,
            )
        )

        # Fetch trending topic if requested
        trending_topic = None
        if request.use_trending:
            trending_topic = self._fetch_trending_topic(request.trending_category)

        # Create config for factory
        config = DebateConfig(
            question=request.question,
            agents_str=agents_str,
            rounds=request.rounds,
            consensus=request.consensus,
            debate_format=request.debate_format,
            debate_id=debate_id,
            trending_topic=trending_topic,
        )

        # Submit to thread pool
        try:
            executor = self._get_executor()
            executor.submit(self._run_debate, config, debate_id)
        except RuntimeError as e:
            logger.warning(f"Cannot submit debate: {e}")
            return DebateResponse(
                success=False,
                error="Server at capacity. Please try again later.",
                status_code=503,
            )

        return DebateResponse(
            success=True,
            debate_id=debate_id,
            status="created",
            task=request.question,
            status_code=200,
        )

    def _get_executor(self) -> ThreadPoolExecutor:
        """Get the shared thread pool executor from StateManager."""
        return get_state_manager().get_executor(max_workers=MAX_CONCURRENT_DEBATES)

    def _run_debate(self, config: DebateConfig, debate_id: str) -> None:
        """Execute debate in background thread.

        Args:
            config: Debate configuration
            debate_id: Unique debate identifier
        """
        import time

        start_time = time.time()
        try:
            # Create event hooks for streaming with explicit loop_id
            # (prevents race condition when multiple debates run concurrently)
            hooks = create_arena_hooks(self.emitter, loop_id=debate_id)

            # Create arena using factory with streaming wrapper
            arena = self.factory.create_arena(
                config,
                event_hooks=hooks,
                stream_wrapper=wrap_agent_for_streaming,
            )

            # Reset circuit breakers for fresh start
            self.factory.reset_circuit_breakers(arena)

            # Run debate with timeout
            # Use protocol timeout if configured, otherwise use global default
            protocol_timeout = getattr(arena.protocol, "timeout_seconds", 0)
            timeout = (
                protocol_timeout
                if isinstance(protocol_timeout, (int, float)) and protocol_timeout > 0
                else DEBATE_TIMEOUT_SECONDS
            )
            update_debate_status(debate_id, "running")

            async def run_with_timeout():
                return await asyncio.wait_for(arena.run(), timeout=timeout)

            result = run_async(run_with_timeout())

            # Update status with result
            update_debate_status(
                debate_id,
                "completed",
                result={
                    "final_answer": result.final_answer,
                    "consensus_reached": result.consensus_reached,
                    "confidence": result.confidence,
                    "grounded_verdict": (
                        result.grounded_verdict.to_dict() if result.grounded_verdict else None
                    ),
                },
            )

            # Persist debate to SQLite storage
            try:
                if self.storage:
                    # Parse agents string to list
                    agents_list = (
                        config.agents_str.split(",")
                        if isinstance(config.agents_str, str)
                        else config.agents_str
                    )
                    # Serialize messages from result
                    messages_data = []
                    if hasattr(result, "messages") and result.messages:
                        for msg in result.messages:
                            messages_data.append(
                                {
                                    "role": msg.role,
                                    "agent": msg.agent,
                                    "content": msg.content,
                                    "round": msg.round,
                                    "timestamp": (
                                        msg.timestamp.isoformat()
                                        if hasattr(msg.timestamp, "isoformat")
                                        else str(msg.timestamp)
                                    ),
                                }
                            )

                    debate_data = {
                        "id": debate_id,
                        "task": config.question,
                        "agents": agents_list,
                        "rounds": config.rounds,
                        "final_answer": result.final_answer,
                        "consensus_reached": result.consensus_reached,
                        "confidence": result.confidence,
                        "grounded_verdict": (
                            result.grounded_verdict.to_dict() if result.grounded_verdict else None
                        ),
                        "messages": messages_data,
                    }
                    self.storage.save_dict(debate_data)
                    logger.info(f"[debate] Persisted debate {debate_id} to storage")
            except Exception as e:
                logger.error(f"[debate] Failed to persist debate {debate_id}: {e}")

            # Emit leaderboard update
            self._emit_leaderboard_update(debate_id)

        except ValueError as e:
            # Validation errors (not enough agents, etc.)
            safe_msg = str(e)
            update_debate_status(debate_id, "error", error=safe_msg)
            self.emitter.emit(
                StreamEvent(
                    type=StreamEventType.ERROR,
                    data={"error": safe_msg, "debate_id": debate_id},
                )
            )
            # Emit DEBATE_END so frontend knows debate is finished
            self.emitter.emit(
                StreamEvent(
                    type=StreamEventType.DEBATE_END,
                    data={
                        "debate_id": debate_id,
                        "duration": time.time() - start_time,
                        "rounds": 0,
                        "error": safe_msg,
                    },
                    loop_id=debate_id,
                )
            )
            logger.error(f"[debate] Validation error in {debate_id}: {e}")

        except Exception as e:
            import traceback

            safe_msg = safe_error_message(e, "debate_execution")
            error_trace = traceback.format_exc()
            update_debate_status(debate_id, "error", error=safe_msg)
            logger.error(f"[debate] Thread error in {debate_id}: {e}\n{error_trace}")
            self.emitter.emit(
                StreamEvent(
                    type=StreamEventType.ERROR,
                    data={"error": safe_msg, "debate_id": debate_id},
                )
            )
            # Emit DEBATE_END so frontend knows debate is finished
            self.emitter.emit(
                StreamEvent(
                    type=StreamEventType.DEBATE_END,
                    data={
                        "debate_id": debate_id,
                        "duration": time.time() - start_time,
                        "rounds": 0,
                        "error": safe_msg,
                    },
                    loop_id=debate_id,
                )
            )

    def _fetch_trending_topic(self, category: Optional[str]) -> Optional[Any]:
        """Fetch a trending topic for the debate.

        Args:
            category: Optional category filter

        Returns:
            TrendingTopic or None
        """
        try:
            from aragora.pulse.ingestor import (
                HackerNewsIngestor,
                PulseManager,
                RedditIngestor,
                TwitterIngestor,
            )

            async def _fetch():
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
                return manager.select_topic_for_debate(topics)

            loop = asyncio.new_event_loop()
            try:
                topic = loop.run_until_complete(_fetch())
                if topic:
                    logger.info(f"Selected trending topic: {topic.topic}")
                return topic
            finally:
                loop.close()

        except Exception as e:
            logger.warning(f"Trending topic fetch failed (non-fatal): {e}")
            return None

    def _emit_leaderboard_update(self, debate_id: str) -> None:
        """Emit leaderboard update event after debate completion."""
        if not self.elo_system:
            return

        try:
            top_agents = self.elo_system.get_leaderboard(limit=10)
            self.emitter.emit(
                StreamEvent(
                    type=StreamEventType.LEADERBOARD_UPDATE,
                    data={
                        "debate_id": debate_id,
                        "leaderboard": [
                            {
                                "agent": a.agent_name,
                                "elo": a.elo_rating,
                                "wins": a.wins,
                                "debates": a.total_debates,
                            }
                            for a in top_agents
                        ],
                    },
                )
            )
        except Exception as e:
            logger.debug(f"Leaderboard emission failed: {e}")

    @classmethod
    def shutdown(cls) -> None:
        """Shutdown the thread pool executor via StateManager."""
        get_state_manager().shutdown_executor()
