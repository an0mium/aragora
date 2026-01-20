"""
Debate execution logic for the streaming server.

This module handles the background execution of ad-hoc debates started via the
HTTP API. It runs debates in a ThreadPoolExecutor to avoid blocking the event loop.

Key components:
- _parse_debate_request: Validate and parse debate request JSON
- _fetch_trending_topic_async: Fetch trending topics for debate seeding
- _execute_debate_thread: Run a debate in a background thread
"""

import logging
import time
from typing import TYPE_CHECKING, Any, Optional, Union, cast

if TYPE_CHECKING:
    from typing import Literal

    from aragora.agents.base import AgentType
    from aragora.core import Agent
    from aragora.debate.orchestrator import Arena as ArenaClass
    from aragora.debate.protocol import DebateProtocol as DebateProtocolClass
    from aragora.core import Environment as EnvironmentClass
    from aragora.server.stream.emitter import SyncEventEmitter

    # Consensus type from DebateProtocol
    ConsensusType = Literal[
        "majority", "unanimous", "judge", "none", "weighted", "supermajority", "any", "byzantine"
    ]

from aragora.config import (
    ALLOWED_AGENT_TYPES,
    DEBATE_TIMEOUT_SECONDS,
    MAX_AGENTS_PER_DEBATE,
)
from aragora.server.errors import safe_error_message as _safe_error_message
from aragora.server.stream.arena_hooks import (
    create_arena_hooks,
    wrap_agent_for_streaming,
)
from aragora.server.stream.events import StreamEvent, StreamEventType
from aragora.server.stream.state_manager import (
    get_active_debates,
    get_active_debates_lock,
)

logger = logging.getLogger(__name__)

# Backward compatibility aliases
_active_debates = get_active_debates()
_active_debates_lock = get_active_debates_lock()

# Check if debate orchestrator is available
# Type aliases for optional debate components
_ArenaType = Union[type["ArenaClass"], None]
_DebateProtocolType = Union[type["DebateProtocolClass"], None]
_EnvironmentType = Union[type["EnvironmentClass"], None]
_CreateAgentType = Union[Any, None]  # Callable type is complex, use Any

try:
    from aragora.agents.base import create_agent as _create_agent
    from aragora.core import Environment as _Environment
    from aragora.debate.orchestrator import Arena as _Arena, DebateProtocol as _DebateProtocol

    DEBATE_AVAILABLE = True
    Arena: _ArenaType = _Arena
    DebateProtocol: _DebateProtocolType = _DebateProtocol
    create_agent: _CreateAgentType = _create_agent
    Environment: _EnvironmentType = _Environment
except ImportError:
    DEBATE_AVAILABLE = False
    Arena = None
    DebateProtocol = None
    create_agent = None
    Environment = None


def parse_debate_request(data: dict) -> tuple[Optional[dict], Optional[str]]:
    """Parse and validate debate request data.

    Args:
        data: JSON request body from the HTTP API

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


async def fetch_trending_topic_async(category: Optional[str] = None) -> Optional[Any]:
    """Fetch a trending topic for the debate.

    Args:
        category: Optional category to filter trending topics

    Returns:
        A TrendingTopic object or None if unavailable.
    """
    try:
        from aragora.pulse.ingestor import (
            HackerNewsIngestor,
            PulseManager,
            RedditIngestor,
            TwitterIngestor,
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


def execute_debate_thread(
    debate_id: str,
    question: str,
    agents_str: str,
    rounds: int,
    consensus: str,
    trending_topic: Optional[Any],
    emitter: "SyncEventEmitter",
    user_id: str = "",
    org_id: str = "",
) -> None:
    """Execute a debate in a background thread.

    This method is run in a ThreadPoolExecutor to avoid blocking the event loop.

    Args:
        debate_id: Unique identifier for this debate
        question: The debate topic/question
        agents_str: Comma-separated list of agent types
        rounds: Number of debate rounds
        consensus: Consensus method to use
        trending_topic: Optional trending topic to seed the debate
        emitter: Event emitter for streaming updates
        user_id: Optional user ID for usage tracking
        org_id: Optional organization ID for usage tracking
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

        # Parse agent specs using unified AgentSpec (validates provider against allowlist)
        from aragora.agents.spec import AgentSpec

        agent_specs = AgentSpec.parse_list(agents_str)

        # Create agents with streaming support
        # All agents are proposers for full participation in all rounds
        agents: list[Agent] = []
        for spec in agent_specs:
            role = spec.role or "proposer"  # All agents propose and participate fully
            agent = create_agent(
                model_type=cast("AgentType", spec.provider),
                name=spec.name,
                role=role,
            )
            # Wrap agent for token streaming if supported
            agent = wrap_agent_for_streaming(agent, emitter, debate_id)
            agents.append(agent)

        # Create environment and protocol
        env = Environment(task=question, context="", max_rounds=rounds)
        protocol = DebateProtocol(
            rounds=rounds,
            consensus=cast("ConsensusType", consensus),
            proposer_count=len(agents),  # All agents propose initially
            topology="all-to-all",  # Everyone critiques everyone
            # Disable early termination to ensure full rounds with all phases
            early_stopping=False,
            convergence_detection=False,
            min_rounds_before_early_stop=rounds,
        )

        # Create arena with hooks and available context systems
        # Pass loop_id explicitly to prevent race conditions with concurrent debates
        hooks = create_arena_hooks(emitter, loop_id=debate_id)

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
            cast("DebateProtocolClass", protocol),
            event_hooks=hooks,
            event_emitter=emitter,
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
        emitter.emit(
            StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": safe_msg, "debate_id": debate_id},
            )
        )


__all__ = [
    "DEBATE_AVAILABLE",
    "execute_debate_thread",
    "fetch_trending_topic_async",
    "parse_debate_request",
]
