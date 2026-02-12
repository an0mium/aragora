"""Debate execution runner extracted from Arena.

Contains _DebateExecutionState and the _run_inner helper methods that
coordinate debate initialization, infrastructure setup, phase execution,
metrics recording, completion handling, and resource cleanup.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aragora.core import DebateResult
from aragora.debate.complexity_governor import (
    classify_task_complexity,
    get_complexity_governor,
)
from aragora.debate.context import DebateContext
from aragora.logging_config import LogContext, get_logger as get_structured_logger
from aragora.observability.tracing import add_span_attributes
from aragora.server.metrics import (
    ACTIVE_DEBATES,
    track_debate_outcome,
)
from aragora.observability.metrics.debate_slo import (
    record_debate_completion_slo,
    update_debate_success_rate,
)

if TYPE_CHECKING:
    from aragora.debate.orchestrator import Arena

logger = get_structured_logger(__name__)

# ThinkPRM integration -- availability flag and helper functions

try:
    from aragora.verification.think_prm import (
        ProcessVerificationResult,
        ThinkPRMConfig,
        ThinkPRMVerifier,
    )

    THINK_PRM_AVAILABLE = True
except ImportError:
    THINK_PRM_AVAILABLE = False


def _convert_messages_to_think_prm_rounds(
    messages: list,
) -> list[dict]:
    """Convert debate Messages into ThinkPRM round format.

    Groups messages by round number and formats them as contribution dicts
    expected by ThinkPRMVerifier.verify_debate_process().

    Args:
        messages: List of aragora.core.Message objects with round attribute.

    Returns:
        List of round dicts, each with a 'contributions' list.
    """
    if not messages:
        return []

    # Group by round number
    rounds_map: dict[int, list[dict]] = {}
    for msg in messages:
        round_num = getattr(msg, "round", 0) or 0
        if round_num not in rounds_map:
            rounds_map[round_num] = []
        rounds_map[round_num].append(
            {
                "content": getattr(msg, "content", ""),
                "agent_id": getattr(msg, "agent", "unknown"),
                "dependencies": [],
            }
        )

    # Sort by round number and return
    return [
        {"contributions": rounds_map[r]}
        for r in sorted(rounds_map.keys())
    ]


async def _run_think_prm_verification(
    arena: Arena,
    ctx: DebateContext,
) -> ProcessVerificationResult | None:
    """Run ThinkPRM verification on completed debate rounds.

    Args:
        arena: The Arena instance with agents and protocol config.
        ctx: DebateContext with context_messages and debate_id.

    Returns:
        ProcessVerificationResult or None if verification cannot run.
    """
    if not THINK_PRM_AVAILABLE:
        return None

    agents = getattr(arena, "agents", [])
    if not agents:
        return None

    messages = getattr(ctx, "context_messages", [])
    if not messages:
        return None

    # Convert messages to ThinkPRM round format
    rounds = _convert_messages_to_think_prm_rounds(messages)
    if not rounds:
        return None

    # Find the verifier agent
    protocol = getattr(arena, "protocol", None)
    verifier_agent_id = getattr(protocol, "think_prm_verifier_agent", "claude")
    parallel = getattr(protocol, "think_prm_parallel", True)
    max_parallel = getattr(protocol, "think_prm_max_parallel", 3)

    # Use the autonomic executor's generate method as the query function
    autonomic = getattr(arena, "autonomic", None)
    if autonomic is None:
        return None

    # Find the agent to use for verification
    verifier = None
    for agent in agents:
        if getattr(agent, "name", None) == verifier_agent_id:
            verifier = agent
            break
    if verifier is None and agents:
        verifier = agents[0]  # Fallback to first agent

    async def query_fn(agent_id: str, prompt: str, max_tokens: int = 1000) -> str:
        return await autonomic.generate(verifier, prompt, [])

    # Set the debate_id on round data for result tracking
    if rounds:
        rounds[0]["debate_id"] = getattr(ctx, "debate_id", "unknown")

    # Configure and run verifier
    config = ThinkPRMConfig(
        verifier_agent_id=verifier_agent_id,
        parallel_verification=parallel,
        max_parallel=max_parallel,
    )
    prm_verifier = ThinkPRMVerifier(config)

    try:
        result = await prm_verifier.verify_debate_process(rounds, query_fn)
        # Override debate_id from context
        result.debate_id = getattr(ctx, "debate_id", "unknown")
        return result
    except Exception as e:
        logger.warning(f"think_prm_verification_failed: {e}")
        return None


@dataclass
class _DebateExecutionState:
    """Internal state for debate execution passed between _run_inner helper methods."""

    debate_id: str
    correlation_id: str
    domain: str
    task_complexity: Any  # TaskComplexity enum
    ctx: DebateContext
    gupp_bead_id: str | None = None
    gupp_hook_entries: dict[str, str] = field(default_factory=dict)
    debate_status: str = "completed"
    debate_start_time: float = 0.0


async def _populate_result_cost(
    result: DebateResult,
    debate_id: str,
    extensions: Any,
) -> None:
    """Populate DebateResult cost fields from cost tracker data.

    Called after extensions.on_debate_complete() to ensure the result
    object carries accurate cost information for downstream consumers
    (DecisionPlanFactory, budget coordinator, etc.).
    """
    try:
        # Aggregate per-agent costs from cost tracker
        cost_tracker = getattr(extensions, "cost_tracker", None)
        if cost_tracker is not None:
            debate_costs = await cost_tracker.get_debate_cost(debate_id)
            if debate_costs:
                total = float(debate_costs.get("total_cost_usd", 0))
                if total > 0:
                    result.total_cost_usd = total

                cost_by_agent = debate_costs.get("cost_by_agent", {})
                if cost_by_agent:
                    result.per_agent_cost = {str(k): float(v) for k, v in cost_by_agent.items()}

        # Carry budget limit through to result
        budget_limit = getattr(extensions, "debate_budget_limit_usd", None)
        if budget_limit is not None:
            result.budget_limit_usd = budget_limit

    except Exception as e:
        logger.debug(f"cost_population_failed (non-critical): {e}")


async def initialize_debate_context(
    arena: Arena,
    correlation_id: str,
) -> _DebateExecutionState:
    """Initialize debate context and return execution state.

    Sets up:
    - Debate ID and correlation ID
    - Convergence detector (debate-scoped cache)
    - Knowledge Mound context
    - Culture hints
    - DebateContext with all dependencies
    - BeliefNetwork (if enabled)
    - Task complexity classification
    - Question domain classification
    - Agent selection and hierarchy roles
    - Agent-to-agent channels
    """
    import uuid

    debate_id = str(uuid.uuid4())
    if not correlation_id:
        correlation_id = f"corr-{debate_id[:8]}"

    # Reinitialize convergence detector with debate-scoped cache
    arena._reinit_convergence_for_debate(debate_id)

    # Extract domain early for metrics
    domain = arena._extract_debate_domain()

    # Initialize Knowledge Mound context for bidirectional integration
    await arena._init_km_context(debate_id, domain)

    # Apply culture-informed protocol adjustments
    culture_hints = arena._get_culture_hints(debate_id)
    if culture_hints:
        arena._apply_culture_hints(culture_hints)

    # Create shared context for all phases
    ctx = DebateContext(
        env=arena.env,
        agents=arena.agents,
        start_time=time.time(),
        debate_id=debate_id,
        correlation_id=correlation_id,
        domain=domain,
        hook_manager=arena.hook_manager,
        org_id=arena.org_id,
        auth_context=getattr(arena, "auth_context", None),
        budget_check_callback=lambda round_num: arena._budget_coordinator.check_budget_mid_debate(
            debate_id, round_num
        ),
    )
    ctx.molecule_orchestrator = arena.molecule_orchestrator
    ctx.checkpoint_bridge = arena.checkpoint_bridge

    # Initialize BeliefNetwork with KM seeding if enabled
    if getattr(arena.protocol, "enable_km_belief_sync", False):
        ctx.belief_network = arena._setup_belief_network(
            debate_id=debate_id,
            topic=arena.env.task,
            seed_from_km=True,
        )

    # Classify task complexity and configure adaptive timeouts
    task_complexity = classify_task_complexity(arena.env.task)
    governor = get_complexity_governor()
    governor.set_task_complexity(task_complexity)

    # Wire governor to API agents for per-agent adaptive timeout management
    from aragora.agents.api_agents.base import APIAgent

    for agent in arena.agents:
        if isinstance(agent, APIAgent):
            agent.set_complexity_governor(governor)

    # Classify question domain using LLM for accurate persona selection
    if arena.prompt_builder:
        try:
            from aragora.utils.env import is_offline_mode

            use_llm = bool(
                getattr(arena.protocol, "enable_llm_question_classification", True)
            )
            if is_offline_mode():
                use_llm = False

            await arena.prompt_builder.classify_question_async(use_llm=use_llm)
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            logger.warning(f"Question classification timed out: {e}")
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Question classification failed with data error: {e}")
        except Exception as e:
            logger.exception(f"Unexpected question classification error: {e}")

    # Apply performance-based agent selection if enabled
    if arena.use_performance_selection:
        arena.agents = arena._select_debate_team(arena.agents)
        ctx.agents = arena.agents

    # Assign hierarchy roles to agents (Gastown pattern)
    arena._assign_hierarchy_roles(ctx, task_type=domain)

    # Initialize agent-to-agent channels
    await arena._setup_agent_channels(ctx, debate_id)

    return _DebateExecutionState(
        debate_id=debate_id,
        correlation_id=correlation_id,
        domain=domain,
        task_complexity=task_complexity,
        ctx=ctx,
    )


async def setup_debate_infrastructure(
    arena: Arena,
    state: _DebateExecutionState,
) -> None:
    """Set up debate infrastructure before execution.

    Handles:
    - Structured logging for debate start
    - Trackers notification
    - Agent preview emission
    - Budget validation
    - GUPP hook tracking initialization
    - Initial result creation
    """
    ctx = state.ctx

    # Structured logging for debate lifecycle
    with LogContext(trace_id=state.correlation_id):
        logger.info(
            "debate_start",
            debate_id=state.debate_id,
            complexity=state.task_complexity.value,
            agent_count=len(arena.agents),
            agents=[a.name for a in arena.agents],
            domain=state.domain,
            task_length=len(arena.env.task),
        )

    # Notify subsystem coordinator of debate start
    arena._trackers.on_debate_start(ctx)

    # Emit agent preview for quick UI feedback
    arena._emit_agent_preview()

    # Check budget before starting debate (may raise BudgetExceededError)
    arena._budget_coordinator.check_budget_before_debate(state.debate_id)

    # Initialize per-debate budget tracking in extensions
    arena.extensions.setup_debate_budget(state.debate_id)

    # Initialize GUPP hook tracking for crash recovery
    if getattr(arena.protocol, "enable_hook_tracking", False):
        try:
            state.gupp_bead_id = await arena._create_pending_debate_bead(
                state.debate_id, arena.env.task
            )
            if state.gupp_bead_id:
                state.gupp_hook_entries = await arena._init_hook_tracking(
                    state.debate_id, state.gupp_bead_id
                )
        except (OSError, RuntimeError, ValueError, TypeError) as e:
            logger.debug(f"GUPP initialization failed (non-critical): {e}")

    # Initialize result early for timeout recovery
    ctx.result = DebateResult(
        task=arena.env.task,
        consensus_reached=False,
        confidence=0.0,
        messages=[],
        critiques=[],
        votes=[],
        rounds_used=0,
        final_answer="",
    )

    # Record start time for metrics
    state.debate_start_time = time.perf_counter()


async def execute_debate_phases(
    arena: Arena,
    state: _DebateExecutionState,
    span: Any,
) -> None:
    """Execute all debate phases with tracing and error handling.

    Args:
        arena: The Arena instance
        state: The debate execution state
        span: OpenTelemetry span for tracing
    """
    from aragora.exceptions import EarlyStopError

    ctx = state.ctx

    try:
        # Execute all phases via PhaseExecutor with OpenTelemetry tracing
        execution_result = await arena.phase_executor.execute(
            ctx,
            debate_id=state.debate_id,
        )
        arena._log_phase_failures(execution_result)

    except asyncio.TimeoutError:
        # Timeout recovery - use partial results from context
        ctx.result.messages = ctx.partial_messages
        ctx.result.critiques = ctx.partial_critiques
        ctx.result.rounds_used = ctx.partial_rounds
        state.debate_status = "timeout"
        span.set_attribute("debate.status", "timeout")
        logger.warning("Debate timed out, returning partial results")

    except EarlyStopError:
        state.debate_status = "aborted"
        span.set_attribute("debate.status", "aborted")
        raise

    except Exception as e:
        state.debate_status = "error"
        span.set_attribute("debate.status", "error")
        span.record_exception(e)
        raise


def record_debate_metrics(
    arena: Arena,
    state: _DebateExecutionState,
    span: Any,
) -> None:
    """Record debate metrics in the finally block.

    Args:
        arena: The Arena instance
        state: The debate execution state
        span: OpenTelemetry span for tracing
    """
    ACTIVE_DEBATES.dec()
    duration = time.perf_counter() - state.debate_start_time
    ctx = state.ctx

    # Get consensus info from result
    consensus_reached = getattr(ctx.result, "consensus_reached", False)
    confidence = getattr(ctx.result, "confidence", 0.0)

    # Add final attributes to span
    add_span_attributes(
        span,
        {
            "debate.status": state.debate_status,
            "debate.duration_seconds": duration,
            "debate.consensus_reached": consensus_reached,
            "debate.confidence": confidence,
            "debate.message_count": len(ctx.result.messages) if ctx.result else 0,
        },
    )

    track_debate_outcome(
        status=state.debate_status,
        domain=state.domain,
        duration_seconds=duration,
        consensus_reached=consensus_reached,
        confidence=confidence,
    )

    # Record SLO-specific metrics for percentile tracking (p50/p95/p99)
    if state.debate_status == "completed":
        outcome = "consensus" if consensus_reached else "no_consensus"
    elif state.debate_status == "timeout":
        outcome = "timeout"
    else:
        outcome = "error"
    record_debate_completion_slo(duration, outcome)
    update_debate_success_rate(consensus_reached)

    # Structured logging for debate completion
    logger.info(
        "debate_end",
        debate_id=state.debate_id,
        status=state.debate_status,
        duration_seconds=round(duration, 3),
        consensus_reached=consensus_reached,
        confidence=round(confidence, 3),
        rounds_used=ctx.result.rounds_used if ctx.result else 0,
        message_count=len(ctx.result.messages) if ctx.result else 0,
        domain=state.domain,
    )

    arena._track_circuit_breaker_metrics()


async def handle_debate_completion(
    arena: Arena,
    state: _DebateExecutionState,
) -> None:
    """Handle post-debate completion tasks.

    Includes:
    - Trackers notification
    - Extensions triggering (billing, training export)
    - Budget recording
    - Knowledge Mound ingestion
    - GUPP hook completion
    - Bead creation
    - Supabase sync queuing
    """
    ctx = state.ctx

    # Notify subsystem coordinator of debate completion
    if ctx.result:
        arena._trackers.on_debate_complete(ctx, ctx.result)

    # Trigger extensions (billing, training export)
    arena.extensions.on_debate_complete(ctx, ctx.result, arena.agents)

    # Populate DebateResult cost fields from cost tracker
    if ctx.result:
        await _populate_result_cost(ctx.result, state.debate_id, arena.extensions)

    # Record debate cost against organization budget
    if ctx.result:
        arena._budget_coordinator.record_debate_cost(
            state.debate_id, ctx.result, extensions=arena.extensions
        )

    # Ingest high-confidence consensus into Knowledge Mound
    if ctx.result:
        try:
            await arena._ingest_debate_outcome(ctx.result)
        except (ConnectionError, OSError, ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Knowledge Mound ingestion failed (non-critical): {e}")

    # Complete GUPP hook tracking for crash recovery
    if state.gupp_bead_id and state.gupp_hook_entries:
        try:
            success = state.debate_status == "completed"
            await arena._update_debate_bead(state.gupp_bead_id, ctx.result, success)
            await arena._complete_hook_tracking(
                state.gupp_bead_id,
                state.gupp_hook_entries,
                success,
                error_msg="" if success else f"Debate {state.debate_status}",
            )
            if success:
                ctx.result.bead_id = state.gupp_bead_id
        except (ConnectionError, OSError, ValueError, TypeError, AttributeError) as e:
            logger.debug(f"GUPP completion failed (non-critical): {e}")
    # Create a Bead if GUPP didn't already create one
    elif ctx.result and not state.gupp_bead_id:
        try:
            bead_id = await arena._create_debate_bead(ctx.result)
            if bead_id:
                ctx.result.bead_id = bead_id
        except (OSError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.debug(f"Bead creation failed (non-critical): {e}")

    # Queue for Supabase background sync
    arena._queue_for_supabase_sync(ctx, ctx.result)


async def cleanup_debate_resources(
    arena: Arena,
    state: _DebateExecutionState,
) -> DebateResult:
    """Clean up debate resources and finalize result.

    Handles:
    - Checkpoint cleanup (on success)
    - Convergence cache cleanup
    - Agent channel teardown
    - Result finalization
    - Translation (if enabled)

    Returns:
        The finalized DebateResult
    """
    ctx = state.ctx

    # Clean up checkpoints after successful completion
    if state.debate_status == "completed" and getattr(
        arena.protocol, "checkpoint_cleanup_on_success", True
    ):
        try:
            keep_count = getattr(arena.protocol, "checkpoint_keep_on_success", 0)
            deleted = await arena.cleanup_checkpoints(state.debate_id, keep_latest=keep_count)
            if deleted > 0:
                logger.debug(f"[checkpoint] Cleaned up {deleted} checkpoints for completed debate")
        except Exception as e:
            logger.debug(f"[checkpoint] Cleanup failed (non-critical): {e}")

    # Cleanup debate-scoped embedding cache to free memory
    arena._cleanup_convergence_cache()
    await arena._teardown_agent_channels()

    # Finalize the result
    result = ctx.finalize_result()

    # Translate conclusions if multi-language support is enabled
    if result and getattr(arena.protocol, "enable_translation", False):
        await arena._translate_conclusions(result)

    return result
