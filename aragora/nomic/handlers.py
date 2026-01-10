"""
Nomic Loop State Handlers.

Provides handler factories that bridge the event-driven state machine
to existing phase implementations. Each handler wraps a phase's execute()
method and translates results into state machine transitions.

Usage:
    from aragora.nomic.handlers import create_handlers

    handlers = create_handlers(
        aragora_path=Path("."),
        agents=agent_list,
        ...
    )

    for state, handler in handlers.items():
        machine.register_handler(state, handler)
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .states import NomicState, StateContext
from .events import Event

logger = logging.getLogger(__name__)


# Type for state handlers
StateHandler = Callable[[StateContext, Event], "asyncio.Future[Tuple[NomicState, Dict[str, Any]]]"]


async def context_handler(
    context: StateContext,
    event: Event,
    *,
    context_phase: Any,
) -> Tuple[NomicState, Dict[str, Any]]:
    """
    Handler for the CONTEXT state.

    Gathers codebase understanding from multiple agents.

    Args:
        context: Current state context
        event: Trigger event
        context_phase: ContextPhase instance

    Returns:
        (next_state, result_data)
    """
    try:
        result = await context_phase.execute()

        if result.get("success"):
            return NomicState.DEBATE, {
                "codebase_summary": result.get("codebase_summary", ""),
                "recent_changes": result.get("recent_changes", ""),
                "open_issues": result.get("open_issues", []),
                "duration_seconds": result.get("duration_seconds", 0),
                "agents_succeeded": result.get("data", {}).get("agents_succeeded", 0),
            }
        else:
            # Context gathering failed - still try to continue with limited context
            logger.warning("Context gathering incomplete, proceeding with limited context")
            return NomicState.DEBATE, {
                "codebase_summary": result.get("codebase_summary", ""),
                "partial": True,
                "error": result.get("error"),
            }

    except Exception as e:
        logger.error(f"Context phase error: {e}")
        raise


async def debate_handler(
    context: StateContext,
    event: Event,
    *,
    debate_phase: Any,
    learning_context_builder: Optional[Callable[[], Any]] = None,
    post_debate_hooks: Optional[Any] = None,
) -> Tuple[NomicState, Dict[str, Any]]:
    """
    Handler for the DEBATE state.

    Runs multi-agent debate to determine improvement.

    Args:
        context: Current state context
        event: Trigger event
        debate_phase: DebatePhase instance
        learning_context_builder: Optional function to build learning context
        post_debate_hooks: Optional PostDebateHooks instance

    Returns:
        (next_state, result_data)
    """
    try:
        # Get context from previous state
        context_result = context.context_result or {}
        codebase_context = context_result.get("codebase_summary", "")
        recent_changes = context_result.get("recent_changes", "")

        # Build learning context if available
        learning = None
        if learning_context_builder:
            try:
                learning = learning_context_builder()
            except Exception as e:
                logger.warning(f"Failed to build learning context: {e}")

        result = await debate_phase.execute(
            codebase_context=codebase_context,
            recent_changes=recent_changes,
            learning_context=learning,
            hooks=post_debate_hooks,
        )

        if result.get("consensus_reached") and result.get("improvement"):
            return NomicState.DESIGN, {
                "improvement": result.get("improvement", ""),
                "confidence": result.get("confidence", 0),
                "votes": result.get("votes", []),
                "duration_seconds": result.get("duration_seconds", 0),
            }
        elif result.get("improvement"):
            # No consensus but have improvement - proceed with lower confidence
            logger.info("No consensus reached, proceeding with best proposal")
            return NomicState.DESIGN, {
                "improvement": result.get("improvement", ""),
                "confidence": result.get("confidence", 0) * 0.5,
                "no_consensus": True,
                "votes": result.get("votes", []),
            }
        else:
            # No viable improvement - skip to completed
            logger.warning("Debate produced no viable improvement")
            return NomicState.COMPLETED, {
                "skipped": True,
                "reason": "no_improvement_proposed",
            }

    except Exception as e:
        logger.error(f"Debate phase error: {e}")
        raise


async def design_handler(
    context: StateContext,
    event: Event,
    *,
    design_phase: Any,
    belief_context_builder: Optional[Callable[[], Any]] = None,
) -> Tuple[NomicState, Dict[str, Any]]:
    """
    Handler for the DESIGN state.

    Creates implementation design from debate improvement.

    Args:
        context: Current state context
        event: Trigger event
        design_phase: DesignPhase instance
        belief_context_builder: Optional function to build belief context

    Returns:
        (next_state, result_data)
    """
    try:
        # Get improvement from debate
        debate_result = context.debate_result or {}
        improvement = debate_result.get("improvement", "")

        if not improvement:
            logger.warning("No improvement to design")
            return NomicState.COMPLETED, {
                "skipped": True,
                "reason": "no_improvement",
            }

        # Build belief context if available
        belief = None
        if belief_context_builder:
            try:
                belief = belief_context_builder()
            except Exception as e:
                logger.warning(f"Failed to build belief context: {e}")

        result = await design_phase.execute(
            improvement=improvement,
            belief_context=belief,
        )

        if result.get("success") and result.get("design"):
            return NomicState.IMPLEMENT, {
                "design": result.get("design", ""),
                "files_affected": result.get("files_affected", []),
                "complexity_estimate": result.get("complexity_estimate", ""),
                "duration_seconds": result.get("duration_seconds", 0),
            }
        else:
            # Design failed - attempt recovery
            logger.warning(f"Design phase failed: {result.get('error')}")
            return NomicState.RECOVERY, {
                "error": result.get("error"),
                "phase": "design",
            }

    except Exception as e:
        logger.error(f"Design phase error: {e}")
        raise


async def implement_handler(
    context: StateContext,
    event: Event,
    *,
    implement_phase: Any,
) -> Tuple[NomicState, Dict[str, Any]]:
    """
    Handler for the IMPLEMENT state.

    Generates code from design specification.

    Args:
        context: Current state context
        event: Trigger event
        implement_phase: ImplementPhase instance

    Returns:
        (next_state, result_data)
    """
    try:
        # Get design from previous state
        design_result = context.design_result or {}
        design = design_result.get("design", "")

        if not design:
            logger.warning("No design to implement")
            return NomicState.COMPLETED, {
                "skipped": True,
                "reason": "no_design",
            }

        result = await implement_phase.execute(design=design)

        if result.get("success"):
            return NomicState.VERIFY, {
                "files_modified": result.get("files_modified", []),
                "diff_summary": result.get("diff_summary", ""),
                "duration_seconds": result.get("duration_seconds", 0),
            }
        else:
            # Implementation failed
            error = result.get("error", "Unknown error")
            logger.warning(f"Implementation failed: {error}")

            # Check if it's a scope issue
            if "scope" in str(error).lower():
                return NomicState.RECOVERY, {
                    "error": error,
                    "phase": "implement",
                    "recoverable": True,
                    "suggestion": "simplify_design",
                }
            else:
                return NomicState.RECOVERY, {
                    "error": error,
                    "phase": "implement",
                }

    except Exception as e:
        logger.error(f"Implement phase error: {e}")
        raise


async def verify_handler(
    context: StateContext,
    event: Event,
    *,
    verify_phase: Any,
) -> Tuple[NomicState, Dict[str, Any]]:
    """
    Handler for the VERIFY state.

    Runs verification checks on implemented changes.

    Args:
        context: Current state context
        event: Trigger event
        verify_phase: VerifyPhase instance

    Returns:
        (next_state, result_data)
    """
    try:
        result = await verify_phase.execute()

        if result.get("tests_passed") and result.get("syntax_valid"):
            return NomicState.COMMIT, {
                "tests_passed": True,
                "syntax_valid": True,
                "test_output": result.get("test_output", ""),
                "duration_seconds": result.get("duration_seconds", 0),
            }
        else:
            # Verification failed - rollback and recover
            checks = result.get("data", {}).get("checks", [])
            failed_checks = [c for c in checks if not c.get("passed")]

            logger.warning(f"Verification failed: {len(failed_checks)} checks failed")
            return NomicState.RECOVERY, {
                "error": "verification_failed",
                "phase": "verify",
                "failed_checks": failed_checks,
                "test_output": result.get("test_output", ""),
            }

    except Exception as e:
        logger.error(f"Verify phase error: {e}")
        raise


async def commit_handler(
    context: StateContext,
    event: Event,
    *,
    commit_phase: Any,
) -> Tuple[NomicState, Dict[str, Any]]:
    """
    Handler for the COMMIT state.

    Commits verified changes to git.

    Args:
        context: Current state context
        event: Trigger event
        commit_phase: CommitPhase instance

    Returns:
        (next_state, result_data)
    """
    try:
        # Get improvement description for commit message
        debate_result = context.debate_result or {}
        improvement = debate_result.get("improvement", "nomic improvement")

        result = await commit_phase.execute(improvement=improvement)

        if result.get("committed"):
            return NomicState.COMPLETED, {
                "commit_hash": result.get("commit_hash"),
                "committed": True,
                "message": improvement,
                "duration_seconds": result.get("duration_seconds", 0),
            }
        else:
            # Commit failed or declined
            reason = result.get("data", {}).get("reason", "unknown")
            logger.info(f"Commit not made: {reason}")
            return NomicState.COMPLETED, {
                "committed": False,
                "reason": reason,
            }

    except Exception as e:
        logger.error(f"Commit phase error: {e}")
        raise


def create_context_handler(
    context_phase: Any,
) -> StateHandler:
    """
    Create a context handler bound to a ContextPhase instance.

    Args:
        context_phase: ContextPhase instance

    Returns:
        Bound handler function
    """
    async def handler(context: StateContext, event: Event) -> Tuple[NomicState, Dict[str, Any]]:
        return await context_handler(context, event, context_phase=context_phase)
    return handler


def create_debate_handler(
    debate_phase: Any,
    learning_context_builder: Optional[Callable[[], Any]] = None,
    post_debate_hooks: Optional[Any] = None,
) -> StateHandler:
    """
    Create a debate handler bound to a DebatePhase instance.

    Args:
        debate_phase: DebatePhase instance
        learning_context_builder: Optional function to build learning context
        post_debate_hooks: Optional PostDebateHooks

    Returns:
        Bound handler function
    """
    async def handler(context: StateContext, event: Event) -> Tuple[NomicState, Dict[str, Any]]:
        return await debate_handler(
            context, event,
            debate_phase=debate_phase,
            learning_context_builder=learning_context_builder,
            post_debate_hooks=post_debate_hooks,
        )
    return handler


def create_design_handler(
    design_phase: Any,
    belief_context_builder: Optional[Callable[[], Any]] = None,
) -> StateHandler:
    """
    Create a design handler bound to a DesignPhase instance.

    Args:
        design_phase: DesignPhase instance
        belief_context_builder: Optional function to build belief context

    Returns:
        Bound handler function
    """
    async def handler(context: StateContext, event: Event) -> Tuple[NomicState, Dict[str, Any]]:
        return await design_handler(
            context, event,
            design_phase=design_phase,
            belief_context_builder=belief_context_builder,
        )
    return handler


def create_implement_handler(
    implement_phase: Any,
) -> StateHandler:
    """
    Create an implement handler bound to an ImplementPhase instance.

    Args:
        implement_phase: ImplementPhase instance

    Returns:
        Bound handler function
    """
    async def handler(context: StateContext, event: Event) -> Tuple[NomicState, Dict[str, Any]]:
        return await implement_handler(context, event, implement_phase=implement_phase)
    return handler


def create_verify_handler(
    verify_phase: Any,
) -> StateHandler:
    """
    Create a verify handler bound to a VerifyPhase instance.

    Args:
        verify_phase: VerifyPhase instance

    Returns:
        Bound handler function
    """
    async def handler(context: StateContext, event: Event) -> Tuple[NomicState, Dict[str, Any]]:
        return await verify_handler(context, event, verify_phase=verify_phase)
    return handler


def create_commit_handler(
    commit_phase: Any,
) -> StateHandler:
    """
    Create a commit handler bound to a CommitPhase instance.

    Args:
        commit_phase: CommitPhase instance

    Returns:
        Bound handler function
    """
    async def handler(context: StateContext, event: Event) -> Tuple[NomicState, Dict[str, Any]]:
        return await commit_handler(context, event, commit_phase=commit_phase)
    return handler


def create_handlers(
    aragora_path: Path,
    agents: List[Any],
    claude_agent: Any,
    codex_agent: Any,
    arena_factory: Callable,
    environment_factory: Callable,
    protocol_factory: Callable,
    nomic_integration: Optional[Any] = None,
    kilocode_available: bool = False,
    kilocode_agent_factory: Optional[Callable] = None,
    cycle_count: int = 0,
    log_fn: Optional[Callable[[str], None]] = None,
    stream_emit_fn: Optional[Callable] = None,
    record_replay_fn: Optional[Callable] = None,
    auto_commit: bool = False,
) -> Dict[NomicState, StateHandler]:
    """
    Create all handlers for a nomic loop cycle.

    This is the main factory function that creates bound handlers
    for all states in the nomic loop.

    Args:
        aragora_path: Path to aragora project root
        agents: List of agents for debate/design
        claude_agent: Claude agent for context
        codex_agent: Codex agent for context/verify
        arena_factory: Factory for Arena instances
        environment_factory: Factory for Environment instances
        protocol_factory: Factory for DebateProtocol instances
        nomic_integration: Optional NomicIntegration
        kilocode_available: Whether KiloCode is available
        kilocode_agent_factory: Factory for KiloCode agents
        cycle_count: Current cycle number
        log_fn: Logging function
        stream_emit_fn: Stream event function
        record_replay_fn: Replay recording function
        auto_commit: Whether to auto-commit

    Returns:
        Dict mapping states to handlers
    """
    # Import phase implementations lazily to avoid circular imports
    from scripts.nomic.phases import (
        ContextPhase,
        DebatePhase,
        DesignPhase,
        ImplementPhase,
        VerifyPhase,
        CommitPhase,
    )

    # Create phase instances
    context_phase = ContextPhase(
        aragora_path=aragora_path,
        claude_agent=claude_agent,
        codex_agent=codex_agent,
        kilocode_available=kilocode_available,
        kilocode_agent_factory=kilocode_agent_factory,
        cycle_count=cycle_count,
        log_fn=log_fn,
        stream_emit_fn=stream_emit_fn,
    )

    debate_phase = DebatePhase(
        aragora_path=aragora_path,
        agents=agents,
        arena_factory=arena_factory,
        environment_factory=environment_factory,
        protocol_factory=protocol_factory,
        nomic_integration=nomic_integration,
        cycle_count=cycle_count,
        log_fn=log_fn,
        stream_emit_fn=stream_emit_fn,
        record_replay_fn=record_replay_fn,
    )

    design_phase = DesignPhase(
        aragora_path=aragora_path,
        agents=agents,
        arena_factory=arena_factory,
        environment_factory=environment_factory,
        protocol_factory=protocol_factory,
        nomic_integration=nomic_integration,
        cycle_count=cycle_count,
        log_fn=log_fn,
        stream_emit_fn=stream_emit_fn,
        record_replay_fn=record_replay_fn,
    )

    implement_phase = ImplementPhase(
        aragora_path=aragora_path,
        cycle_count=cycle_count,
        log_fn=log_fn,
        stream_emit_fn=stream_emit_fn,
        record_replay_fn=record_replay_fn,
    )

    verify_phase = VerifyPhase(
        aragora_path=aragora_path,
        codex=codex_agent,
        nomic_integration=nomic_integration,
        cycle_count=cycle_count,
        log_fn=log_fn,
        stream_emit_fn=stream_emit_fn,
        record_replay_fn=record_replay_fn,
    )

    commit_phase = CommitPhase(
        aragora_path=aragora_path,
        require_human_approval=not auto_commit,
        auto_commit=auto_commit,
        cycle_count=cycle_count,
        log_fn=log_fn,
        stream_emit_fn=stream_emit_fn,
    )

    # Create and return bound handlers
    return {
        NomicState.CONTEXT: create_context_handler(context_phase),
        NomicState.DEBATE: create_debate_handler(debate_phase),
        NomicState.DESIGN: create_design_handler(design_phase),
        NomicState.IMPLEMENT: create_implement_handler(implement_phase),
        NomicState.VERIFY: create_verify_handler(verify_phase),
        NomicState.COMMIT: create_commit_handler(commit_phase),
    }


__all__ = [
    # Individual handler factories
    "create_context_handler",
    "create_debate_handler",
    "create_design_handler",
    "create_implement_handler",
    "create_verify_handler",
    "create_commit_handler",
    # Main factory
    "create_handlers",
    # Raw handlers (for custom phase instances)
    "context_handler",
    "debate_handler",
    "design_handler",
    "implement_handler",
    "verify_handler",
    "commit_handler",
]
