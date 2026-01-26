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
import os
import re
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
    DEBATE_TIMEOUT_SECONDS,
    DEFAULT_AGENTS,
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

_ENV_VAR_RE = re.compile(r"[A-Z][A-Z0-9_]+")
_OPENROUTER_FALLBACK_MODELS = {
    "anthropic-api": "anthropic/claude-3.5-sonnet",
    "openai-api": "openai/gpt-4o-mini",
    "gemini": "google/gemini-2.0-flash-exp:free",
    "grok": "x-ai/grok-2-1212",
    "mistral-api": "mistralai/mistral-large-2411",
}


def _missing_required_env_vars(env_vars: str) -> list[str]:
    """Return missing required env vars for a provider spec."""
    if not env_vars:
        return []
    if "optional" in env_vars.lower():
        return []
    candidates = _ENV_VAR_RE.findall(env_vars)
    if not candidates:
        return []
    if any(os.getenv(var) for var in candidates):
        return []
    return candidates


def _openrouter_key_available() -> bool:
    """Return True if OpenRouter key is configured via secrets or env."""
    try:
        from aragora.config.secrets import get_secret

        value = get_secret("OPENROUTER_API_KEY")
        if value and value.strip():
            return True
    except Exception:
        pass
    env_value = os.getenv("OPENROUTER_API_KEY")
    return bool(env_value and env_value.strip())


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
    agents_value = data.get("agents")
    if isinstance(agents_value, list):
        agents_str = ",".join(str(agent).strip() for agent in agents_value if str(agent).strip())
    elif isinstance(agents_value, str):
        agents_str = agents_value.strip()
    else:
        agents_str = DEFAULT_AGENTS
    if not agents_str:
        agents_str = DEFAULT_AGENTS
    # Validate agent providers early to avoid starting invalid debates
    try:
        from aragora.agents.spec import AgentSpec

        AgentSpec.parse_list(agents_str, _warn=False)
    except Exception as e:
        return None, str(e)
    agent_count = len([s for s in agents_str.split(",") if s.strip()])
    if agent_count < 2:
        return None, "At least 2 agents required for a debate"
    if agent_count > MAX_AGENTS_PER_DEBATE:
        return None, f"Too many agents. Maximum: {MAX_AGENTS_PER_DEBATE}"
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

    # Debug: Log thread start
    logger.info(
        f"[debate] Thread started for {debate_id}: "
        f"question={question[:50]}..., agents={agents_str}, rounds={rounds}"
    )
    thread_start_time = time.time()

    try:
        # Parse agents with bounds check
        agent_list = [s.strip() for s in agents_str.split(",") if s.strip()]
        if len(agent_list) > MAX_AGENTS_PER_DEBATE:
            with _active_debates_lock:
                _active_debates[debate_id]["status"] = "error"
                _active_debates[debate_id]["error"] = (
                    f"Too many agents. Maximum: {MAX_AGENTS_PER_DEBATE}"
                )
                _active_debates[debate_id]["completed_at"] = time.time()
            return
        if len(agent_list) < 2:
            with _active_debates_lock:
                _active_debates[debate_id]["status"] = "error"
                _active_debates[debate_id]["error"] = "At least 2 agents required for a debate"
                _active_debates[debate_id]["completed_at"] = time.time()
            return

        # Parse agent specs using unified AgentSpec (validates provider against allowlist)
        from aragora.agents.registry import AgentRegistry
        from aragora.agents.spec import AgentSpec

        agent_specs = AgentSpec.parse_list(agents_str)
        filtered_specs = []
        openrouter_available = _openrouter_key_available()
        for spec in agent_specs:
            registry_spec = AgentRegistry.get_spec(spec.provider)
            missing_env = []
            if registry_spec and registry_spec.env_vars:
                missing_env = _missing_required_env_vars(registry_spec.env_vars)
            if missing_env:
                fallback_model = _OPENROUTER_FALLBACK_MODELS.get(spec.provider)
                if openrouter_available and fallback_model:
                    fallback_spec = AgentSpec(
                        provider="openrouter",
                        model=fallback_model,
                        persona=spec.persona,
                        role=spec.role,
                        name=spec.name or spec.provider,
                    )
                    emitter.emit(
                        StreamEvent(
                            type=StreamEventType.AGENT_ERROR,
                            data={
                                "error_type": "missing_env_fallback",
                                "message": (
                                    f"Missing {spec.provider} key(s); using OpenRouter model "
                                    f"{fallback_model}"
                                ),
                                "recoverable": True,
                                "phase": "setup",
                            },
                            agent=spec.name or spec.provider,
                            loop_id=debate_id,
                        )
                    )
                    logger.warning(
                        f"[debate] {debate_id}: {spec.provider} missing key(s), "
                        f"fallback to openrouter:{fallback_model}"
                    )
                    filtered_specs.append(fallback_spec)
                    continue
                message = (
                    f"Missing required API key(s) for {spec.provider}: {', '.join(missing_env)}"
                )
                emitter.emit(
                    StreamEvent(
                        type=StreamEventType.AGENT_ERROR,
                        data={
                            "error_type": "missing_env",
                            "message": message,
                            "recoverable": False,
                            "phase": "setup",
                        },
                        agent=spec.name or spec.provider,
                        loop_id=debate_id,
                    )
                )
                logger.warning(f"[debate] {debate_id}: {message}")
                continue
            filtered_specs.append(spec)
        agent_specs = filtered_specs
        if len(agent_specs) < 2:
            error_msg = "Not enough configured agents available to start the debate"
            with _active_debates_lock:
                _active_debates[debate_id]["status"] = "error"
                _active_debates[debate_id]["error"] = error_msg
                _active_debates[debate_id]["completed_at"] = time.time()
            emitter.emit(
                StreamEvent(
                    type=StreamEventType.ERROR,
                    data={"error": error_msg, "debate_id": debate_id},
                    loop_id=debate_id,
                )
            )
            return

        # Create agents with streaming support
        # Assign roles based on position for diverse debate dynamics
        agents: list[Agent] = []
        for i, spec in enumerate(agent_specs):
            # Assign role based on position if not explicitly specified
            role = spec.role
            if role is None:
                if i == 0:
                    role = "proposer"
                elif i == len(agent_specs) - 1 and len(agent_specs) > 1:
                    role = "synthesizer"
                else:
                    role = "critic"
            try:
                agent = create_agent(
                    model_type=cast("AgentType", spec.provider),
                    name=spec.name,
                    role=role,
                    model=spec.model,  # Pass model from spec
                )
            except Exception as e:
                msg = _safe_error_message(e, "agent_init")
                emitter.emit(
                    StreamEvent(
                        type=StreamEventType.AGENT_ERROR,
                        data={
                            "error_type": "init",
                            "message": f"{spec.provider} init failed: {msg}",
                            "recoverable": False,
                            "phase": "setup",
                        },
                        agent=spec.name or spec.provider,
                        loop_id=debate_id,
                    )
                )
                logger.warning(f"[debate] {debate_id}: {spec.provider} init failed: {e}")
                continue

            # Apply persona as system prompt modifier if specified
            if spec.persona:
                try:
                    from aragora.agents.personas import apply_persona_to_agent

                    apply_persona_to_agent(agent, spec.persona)
                except ImportError:
                    pass  # Personas module not available

            # Wrap agent for token streaming if supported
            agent = wrap_agent_for_streaming(agent, emitter, debate_id)
            agents.append(agent)

        if len(agents) < 2:
            error_msg = "Not enough agents could be initialized to start the debate"
            with _active_debates_lock:
                _active_debates[debate_id]["status"] = "error"
                _active_debates[debate_id]["error"] = error_msg
                _active_debates[debate_id]["completed_at"] = time.time()
            emitter.emit(
                StreamEvent(
                    type=StreamEventType.ERROR,
                    data={"error": error_msg, "debate_id": debate_id},
                    loop_id=debate_id,
                )
            )
            return

        # Debug: Log agent creation complete
        agent_names = [a.name for a in agents]
        logger.info(f"[debate] {debate_id}: Created {len(agents)} agents: {agent_names}")

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
            protocol,
            event_hooks=hooks,
            event_emitter=emitter,  # type: ignore[arg-type]
            loop_id=debate_id,
            trending_topic=trending_topic,
            user_id=user_id,
            org_id=org_id,
            usage_tracker=usage_tracker,
        )

        # Debug: Log arena creation
        setup_time = time.time() - thread_start_time
        logger.info(
            f"[debate] {debate_id}: Arena created in {setup_time:.2f}s, starting execution..."
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

        # Debug: Log successful completion
        total_time = time.time() - thread_start_time
        logger.info(
            f"[debate] {debate_id}: Completed in {total_time:.2f}s, "
            f"consensus={result.consensus_reached}, confidence={result.confidence:.2f}"
        )

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
