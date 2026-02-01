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

if TYPE_CHECKING:
    from aragora.debate.orchestrator import Arena

logger = get_structured_logger(__name__)


@dataclass
class _DebateExecutionState:
    """Internal state for debate execution passed between _run_inner helper methods."""

    debate_id: str
    correlation_id: str
    domain: str
    task_complexity: Any  # TaskComplexity enum
    ctx: "DebateContext"
    gupp_bead_id: str | None = None
    gupp_hook_entries: dict[str, str] = field(default_factory=dict)
    debate_status: str = "completed"
    debate_start_time: float = 0.0


async def initialize_debate_context(
    arena: "Arena",
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

    # Classify question domain using LLM for accurate persona selection
    if arena.prompt_builder:
        try:
            await arena.prompt_builder.classify_question_async(use_llm=True)
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
    arena: "Arena",
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
    arena: "Arena",
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
    arena: "Arena",
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
    arena: "Arena",
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
    arena: "Arena",
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
