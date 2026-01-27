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
    DEFAULT_AGENTS,
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

# Default classification when Haiku call fails or times out
_DEFAULT_CLASSIFICATION = {
    "type": "general",
    "domain": "other",
    "complexity": "moderate",
    "aspects": [],
    "approach": "Agents will analyze this topic from multiple perspectives.",
}

if TYPE_CHECKING:
    from aragora.server.stream import SyncEventEmitter

logger = logging.getLogger(__name__)


@dataclass
class DebateRequest:
    """Parsed debate request from HTTP body."""

    question: str
    agents_str: str = DEFAULT_AGENTS
    rounds: int = 8  # 9-round format (0-8), default for all debates
    consensus: str = "judge"  # Judge-based consensus for final decisions
    debate_format: str = "full"  # "light" (~5 min) or "full" (~30 min)
    auto_select: bool = False
    auto_select_config: dict = None
    use_trending: bool = False
    trending_category: Optional[str] = None
    metadata: dict = None  # Custom metadata (e.g., is_onboarding)

    def __post_init__(self):
        if self.auto_select_config is None:
            self.auto_select_config = {}
        if self.metadata is None:
            self.metadata = {}
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
            agents_str=data.get("agents", DEFAULT_AGENTS),
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

    def _preflight_agents(self, agents_str: Any) -> Optional[str]:
        """Validate agent availability before starting a debate.

        Returns:
            Error message if agents are missing/unavailable, otherwise None.
        """
        try:
            from aragora.agents import filter_available_agents
            from aragora.agents.spec import AgentSpec
        except ImportError:
            logger.debug("Agent preflight skipped: credential validator unavailable")
            return None

        try:
            # Handle list or string formats
            if isinstance(agents_str, list):
                agents_str = ",".join(
                    s.strip() if isinstance(s, str) else str(s) for s in agents_str if s
                )
            specs = AgentSpec.parse_list(str(agents_str))
        except Exception as e:
            return f"Invalid agent specification: {e}"

        try:
            requested_count = len(specs)
            available_specs, filtered = filter_available_agents(
                specs,
                log_filtered=False,
                min_agents=requested_count,
            )
        except Exception as e:
            return str(e)

        if filtered:
            missing_detail = "; ".join(f"{agent}: {reason}" for agent, reason in filtered)
            available_names = ", ".join(s.provider for s in available_specs) or "none"
            return (
                "Missing credentials for requested agents. "
                f"Missing: {missing_detail}. Available: {available_names}. "
                "Configure API keys in AWS Secrets Manager or environment variables."
            )

        if requested_count < 2:
            return "At least 2 agents are required to start a debate."

        if len(available_specs) < requested_count:
            available_names = ", ".join(s.provider for s in available_specs) or "none"
            requested_names = ", ".join(s.provider for s in specs) or "none"
            return (
                f"Only {len(available_specs)}/{requested_count} requested agents are available. "
                f"Requested: {requested_names}. Available: {available_names}."
            )

        return None

    async def _quick_classify_async(self, question: str) -> dict:
        """Fast Haiku classification of question type and domain.

        This provides immediate context to users while the debate initializes.
        Uses Claude 3.5 Haiku for speed (~100-200ms typical latency).

        Args:
            question: The debate question to classify

        Returns:
            Dict with type, domain, complexity, aspects, and approach
        """
        import asyncio
        import json
        import os

        # Check for API key first
        if not os.getenv("ANTHROPIC_API_KEY"):
            logger.error("[quick_classify] ANTHROPIC_API_KEY not set - skipping classification")
            return _DEFAULT_CLASSIFICATION

        logger.info("[quick_classify] Starting Haiku classification")

        try:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic()
            # Wrap API call with 5 second timeout
            response = await asyncio.wait_for(
                client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=300,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Classify this debate question. Return ONLY valid JSON, no other text.

Question: {question[:500]}

Return JSON with these exact fields:
- type: one of [factual, ethical, technical, creative, policy, comparative]
- domain: one of [science, technology, philosophy, politics, society, economics, other]
- complexity: one of [simple, moderate, complex]
- aspects: array of 3-4 key focus areas as short phrases
- approach: one sentence on how AI agents will analyze this""",
                        }
                    ],
                ),
                timeout=5.0,
            )
            # Parse JSON from response
            content = response.content[0].text.strip()  # type: ignore[union-attr]
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content)
            logger.info(
                f"[quick_classify] Success: type={result.get('type')}, domain={result.get('domain')}"
            )
            return result
        except asyncio.TimeoutError:
            logger.error("[quick_classify] Haiku API timeout after 5s")
            return _DEFAULT_CLASSIFICATION
        except json.JSONDecodeError as e:
            logger.error(f"[quick_classify] JSON parse error: {e}")
            return _DEFAULT_CLASSIFICATION
        except Exception as e:
            logger.error(f"[quick_classify] Failed: {type(e).__name__}: {e}")
            return _DEFAULT_CLASSIFICATION

    def _quick_classify(self, question: str, debate_id: str) -> None:
        """Run quick classification and emit event (sync wrapper)."""
        try:
            classification = run_async(self._quick_classify_async(question))
            self.emitter.emit(
                StreamEvent(
                    type=StreamEventType.QUICK_CLASSIFICATION,
                    data={
                        "question_type": classification.get("type", "general"),
                        "domain": classification.get("domain", "other"),
                        "complexity": classification.get("complexity", "moderate"),
                        "key_aspects": classification.get("aspects", []),
                        "suggested_approach": classification.get("approach", ""),
                    },
                    loop_id=debate_id,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to emit quick classification: {e}")

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

        preflight_error = self._preflight_agents(agents_str)
        if preflight_error:
            logger.warning(f"[debate] Agent preflight failed: {preflight_error}")
            return DebateResponse(
                success=False,
                error=preflight_error,
                status_code=400,
            )

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

        # Quick classification with Haiku (~100-200ms) - shows immediately in UI
        # This runs while the rest of initialization continues
        self._quick_classify(request.question, debate_id)

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
            metadata=request.metadata,
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
            # Update status to initializing immediately to prevent stuck "starting" state
            update_debate_status(debate_id, "initializing")

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
                    "status": result.status,
                    "agent_failures": result.agent_failures,
                    "participants": result.participants,
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

            # Auto-generate receipt for onboarding debates
            if config.metadata and config.metadata.get("is_onboarding"):
                self._generate_onboarding_receipt(
                    debate_id=debate_id,
                    config=config,
                    result=result,
                    duration_seconds=time.time() - start_time,
                )

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

    def _generate_onboarding_receipt(
        self,
        debate_id: str,
        config: DebateConfig,
        result: Any,
        duration_seconds: float,
    ) -> None:
        """Generate and save a receipt for onboarding debates.

        Args:
            debate_id: Unique debate identifier
            config: Debate configuration
            result: Debate result with final_answer, consensus, etc.
            duration_seconds: Total debate duration
        """
        import hashlib
        import uuid
        from datetime import datetime, timezone

        try:
            from aragora.storage.receipt_store import get_receipt_store

            receipt_store = get_receipt_store()

            # Parse agents
            agents_list = (
                config.agents_str.split(",")
                if isinstance(config.agents_str, str)
                else config.agents_str
            )

            # Build receipt dict
            receipt_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()

            # Determine verdict based on consensus
            if result.consensus_reached and result.confidence >= 0.7:
                verdict = "APPROVED"
                risk_level = "LOW"
            elif result.consensus_reached:
                verdict = "APPROVED_WITH_CONDITIONS"
                risk_level = "MEDIUM"
            else:
                verdict = "NEEDS_REVIEW"
                risk_level = "MEDIUM"

            # Calculate input hash
            input_content = f"{config.question}|{config.agents_str}|{config.rounds}"
            input_hash = hashlib.sha256(input_content.encode()).hexdigest()

            receipt_dict = {
                "receipt_id": receipt_id,
                "gauntlet_id": f"onboarding-{debate_id}",
                "debate_id": debate_id,
                "timestamp": timestamp,
                "input_summary": config.question[:200],
                "input_hash": input_hash,
                "verdict": verdict,
                "confidence": result.confidence if hasattr(result, "confidence") else 0.5,
                "risk_level": risk_level,
                "risk_score": 1.0 - (result.confidence if hasattr(result, "confidence") else 0.5),
                "robustness_score": result.confidence if hasattr(result, "confidence") else 0.5,
                "agents_involved": agents_list,
                "rounds_completed": config.rounds,
                "duration_seconds": duration_seconds,
                "final_answer": result.final_answer if hasattr(result, "final_answer") else "",
                "consensus_reached": (
                    result.consensus_reached if hasattr(result, "consensus_reached") else False
                ),
                "is_onboarding": True,
            }

            # Calculate checksum
            checksum_content = f"{receipt_id}|{debate_id}|{input_hash}|{verdict}"
            receipt_dict["checksum"] = hashlib.sha256(checksum_content.encode()).hexdigest()

            # Save receipt
            receipt_store.save(receipt_dict)
            logger.info(f"[onboarding] Generated receipt {receipt_id} for debate {debate_id}")

            # Update onboarding flow with receipt ID if user_id is available
            user_id = config.metadata.get("user_id") if config.metadata else None
            org_id = config.metadata.get("organization_id") if config.metadata else None
            if user_id:
                try:
                    from aragora.storage.repositories.onboarding import get_onboarding_repository

                    repo = get_onboarding_repository()
                    flow = repo.get_flow(user_id, org_id)
                    if flow:
                        repo.update_flow(
                            flow["id"],
                            {"metadata": {**flow.get("metadata", {}), "receipt_id": receipt_id}},
                        )
                except Exception as e:
                    logger.debug(f"Could not update onboarding flow with receipt: {e}")

        except Exception as e:
            logger.warning(f"[onboarding] Failed to generate receipt for {debate_id}: {e}")

    @classmethod
    def shutdown(cls) -> None:
        """Shutdown the thread pool executor via StateManager."""
        get_state_manager().shutdown_executor()
