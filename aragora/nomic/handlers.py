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

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any
from collections.abc import Callable, Coroutine

from aragora.nomic.sica_settings import load_sica_settings

from .events import Event
from .recovery import RecoveryManager, recovery_handler as core_recovery_handler
from .states import NomicState, StateContext

logger = logging.getLogger(__name__)

# Type for state handlers (async functions returning state and data)
StateHandler = Callable[
    [StateContext, Event], "Coroutine[Any, Any, tuple[NomicState, dict[str, Any]]]"
]


def _select_sica_agent(agents: list[Any], model_name: str | None) -> Any | None:
    """Select an agent for SICA prompts based on model name."""
    model = (model_name or "").lower()
    candidates = [
        ("codex",),
        ("openai",),
        ("gpt",),
        ("claude",),
        ("gemini",),
        ("grok",),
    ]
    for key_tuple in candidates:
        key = key_tuple[0]
        if key in model:
            for agent in agents:
                if agent and key in getattr(agent, "name", "").lower():
                    return agent
    for agent in agents:
        if agent:
            return agent
    return None


async def _run_sica_cycle(
    repo_path: Path,
    agents: list[Any],
    log_fn: Callable[[str], None],
) -> dict[str, Any]:
    """Run the SICA improvement cycle if enabled via env."""
    settings = load_sica_settings()
    if not settings.enabled:
        return {"status": "disabled"}

    try:
        from aragora.nomic.sica_improver import (
            ImprovementType,
            SICAConfig,
            SICAImprover,
        )
    except (ImportError, RuntimeError) as exc:
        log_fn(f"[sica] Unavailable: {exc}")
        return {"status": "unavailable", "error": str(exc)}

    raw_types = [t.strip() for t in settings.improvement_types_csv.split(",") if t.strip()]
    improvement_types: list[ImprovementType] = []
    for raw in raw_types:
        try:
            improvement_types.append(ImprovementType(raw))
        except (ValueError, KeyError):
            log_fn(f"[sica] Unknown improvement type '{raw}', skipping")

    agent = _select_sica_agent(agents, settings.generator_model)

    async def query_fn(model: str, prompt: str, max_tokens: int) -> str:
        if not agent:
            raise RuntimeError("No agent available for SICA")
        return await agent.generate(prompt, context=[])

    config = SICAConfig(
        improvement_types=improvement_types or None,
        generator_model=settings.generator_model,
        require_human_approval=settings.require_approval,
        run_tests=settings.run_tests,
        run_typecheck=settings.run_typecheck,
        run_lint=settings.run_lint,
        test_command=settings.test_command,
        typecheck_command=settings.typecheck_command,
        lint_command=settings.lint_command,
        validation_timeout_seconds=settings.validation_timeout,
        max_opportunities_per_cycle=settings.max_opportunities,
        max_rollbacks_per_cycle=settings.max_rollbacks,
    )

    if settings.require_approval:

        async def approve(patch):
            if not sys.stdin.isatty():
                log_fn("[sica] Approval required but no TTY; rejecting.")
                return False
            log_fn(f"[sica] Proposed patch: {patch.description}")
            if patch.diff:
                print("\n" + patch.diff)  # noqa: T201 — CLI output
            response = input("Apply this patch? [y/N]: ").strip().lower()
            return response in ("y", "yes")

        config.approval_callback = approve

    improver = SICAImprover(
        repo_path=repo_path,
        config=config,
        query_fn=query_fn if agent else None,
    )
    if not agent:
        log_fn("[sica] No agent available; running heuristic-only cycle")

    result = await improver.run_improvement_cycle()
    log_fn(f"[sica] {result.summary()}")
    return {
        "status": "success" if result.patches_successful else "no_changes",
        "summary": result.summary(),
        "result": result.to_dict(),
    }


async def context_handler(
    context: StateContext,
    event: Event,
    *,
    context_phase: Any,
) -> tuple[NomicState, dict[str, Any]]:
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

    except (RuntimeError, OSError, ValueError) as e:
        logger.error("Context phase error: %s", e)
        raise


async def debate_handler(
    context: StateContext,
    event: Event,
    *,
    debate_phase: Any,
    learning_context_builder: Callable[[], Any] | None = None,
    post_debate_hooks: Any | None = None,
) -> tuple[NomicState, dict[str, Any]]:
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
            except (RuntimeError, ValueError, KeyError) as e:
                logger.warning("Failed to build learning context: %s", e)

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

    except (RuntimeError, OSError, ValueError) as e:
        logger.error("Debate phase error: %s", e)
        raise


async def design_handler(
    context: StateContext,
    event: Event,
    *,
    design_phase: Any,
    belief_context_builder: Callable[[], Any] | None = None,
) -> tuple[NomicState, dict[str, Any]]:
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
            except (RuntimeError, ValueError, KeyError) as e:
                logger.warning("Failed to build belief context: %s", e)

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
            logger.warning("Design phase failed: %s", result.get("error"))
            return NomicState.RECOVERY, {
                "error": result.get("error"),
                "phase": "design",
            }

    except (RuntimeError, OSError, ValueError) as e:
        logger.error("Design phase error: %s", e)
        raise


async def implement_handler(
    context: StateContext,
    event: Event,
    *,
    implement_phase: Any,
) -> tuple[NomicState, dict[str, Any]]:
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
            logger.warning("Implementation failed: %s", error)

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

    except (RuntimeError, OSError, ValueError) as e:
        logger.error("Implement phase error: %s", e)
        raise


async def verify_handler(
    context: StateContext,
    event: Event,
    *,
    verify_phase: Any,
    repo_path: Path,
    sica_agents: list[Any] | None = None,
) -> tuple[NomicState, dict[str, Any]]:
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

        sica_result = None
        if sica_agents is None:
            sica_agents = []

        if load_sica_settings().enabled:
            sica_result = await _run_sica_cycle(repo_path, sica_agents, logger.info)

            if sica_result.get("status") == "success":
                reverify = await verify_phase.execute()
                if reverify.get("tests_passed") and reverify.get("syntax_valid"):
                    return NomicState.COMMIT, {
                        "tests_passed": True,
                        "syntax_valid": True,
                        "test_output": reverify.get("test_output", ""),
                        "duration_seconds": reverify.get("duration_seconds", 0),
                        "sica": sica_result,
                    }
                result = reverify

        # Verification failed - rollback and recover
        checks = result.get("data", {}).get("checks", [])
        failed_checks = [c for c in checks if not c.get("passed")]

        logger.warning("Verification failed: %s checks failed", len(failed_checks))
        return NomicState.RECOVERY, {
            "error": "verification_failed",
            "phase": "verify",
            "failed_checks": failed_checks,
            "test_output": result.get("test_output", ""),
            "sica": sica_result,
        }

    except (RuntimeError, OSError, ValueError) as e:
        logger.error("Verify phase error: %s", e)
        raise


async def recovery_state_handler(
    context: StateContext,
    event: Event,
    *,
    recovery_manager: RecoveryManager,
    repo_path: Path,
    sica_agents: list[Any] | None = None,
) -> tuple[NomicState, dict[str, Any]]:
    """
    Handler for the RECOVERY state.

    Uses standard recovery strategies and can optionally run a SICA
    improvement cycle before deciding the recovery transition.
    """
    if sica_agents is None:
        sica_agents = []

    failed_state = context.previous_state
    sica_result: dict[str, Any] | None = None
    if failed_state in (NomicState.IMPLEMENT, NomicState.VERIFY) and load_sica_settings().enabled:
        sica_result = await _run_sica_cycle(repo_path, sica_agents, logger.info)
        if sica_result.get("status") == "success":
            # SICA may have changed code directly; re-run verification next.
            return NomicState.VERIFY, {
                "decision": {
                    "strategy": "SICA_REPAIR",
                    "target_state": NomicState.VERIFY.name,
                    "reason": f"SICA recovered from {failed_state.name if failed_state else 'unknown'}",
                },
                "recovered_from": failed_state.name if failed_state else "UNKNOWN",
                "sica": sica_result,
            }

    next_state, data = await core_recovery_handler(
        context=context,
        event=event,
        recovery_manager=recovery_manager,
    )
    if sica_result is not None:
        data["sica"] = sica_result
    return next_state, data


async def commit_handler(
    context: StateContext,
    event: Event,
    *,
    commit_phase: Any,
) -> tuple[NomicState, dict[str, Any]]:
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
            logger.info("Commit not made: %s", reason)
            return NomicState.COMPLETED, {
                "committed": False,
                "reason": reason,
            }

    except (RuntimeError, OSError, ValueError) as e:
        logger.error("Commit phase error: %s", e)
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

    async def handler(context: StateContext, event: Event) -> tuple[NomicState, dict[str, Any]]:
        return await context_handler(context, event, context_phase=context_phase)

    return handler


def create_debate_handler(
    debate_phase: Any,
    learning_context_builder: Callable[[], Any] | None = None,
    post_debate_hooks: Any | None = None,
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

    async def handler(context: StateContext, event: Event) -> tuple[NomicState, dict[str, Any]]:
        return await debate_handler(
            context,
            event,
            debate_phase=debate_phase,
            learning_context_builder=learning_context_builder,
            post_debate_hooks=post_debate_hooks,
        )

    return handler


def create_design_handler(
    design_phase: Any,
    belief_context_builder: Callable[[], Any] | None = None,
) -> StateHandler:
    """
    Create a design handler bound to a DesignPhase instance.

    Args:
        design_phase: DesignPhase instance
        belief_context_builder: Optional function to build belief context

    Returns:
        Bound handler function
    """

    async def handler(context: StateContext, event: Event) -> tuple[NomicState, dict[str, Any]]:
        return await design_handler(
            context,
            event,
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

    async def handler(context: StateContext, event: Event) -> tuple[NomicState, dict[str, Any]]:
        return await implement_handler(context, event, implement_phase=implement_phase)

    return handler


def create_verify_handler(
    verify_phase: Any,
    *,
    repo_path: Path,
    sica_agents: list[Any] | None = None,
) -> StateHandler:
    """
    Create a verify handler bound to a VerifyPhase instance.

    Args:
        verify_phase: VerifyPhase instance

    Returns:
        Bound handler function
    """

    async def handler(context: StateContext, event: Event) -> tuple[NomicState, dict[str, Any]]:
        return await verify_handler(
            context,
            event,
            verify_phase=verify_phase,
            repo_path=repo_path,
            sica_agents=sica_agents,
        )

    return handler


def create_recovery_handler(
    recovery_manager: RecoveryManager,
    *,
    repo_path: Path,
    sica_agents: list[Any] | None = None,
) -> StateHandler:
    """
    Create a recovery handler bound to a RecoveryManager instance.

    Args:
        recovery_manager: Recovery manager instance
        repo_path: Repository path for optional SICA recovery
        sica_agents: Candidate agents for SICA prompts

    Returns:
        Bound handler function
    """

    async def handler(context: StateContext, event: Event) -> tuple[NomicState, dict[str, Any]]:
        return await recovery_state_handler(
            context,
            event,
            recovery_manager=recovery_manager,
            repo_path=repo_path,
            sica_agents=sica_agents,
        )

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

    async def handler(context: StateContext, event: Event) -> tuple[NomicState, dict[str, Any]]:
        return await commit_handler(context, event, commit_phase=commit_phase)

    return handler


def create_handlers(
    aragora_path: Path,
    agents: list[Any],
    claude_agent: Any,
    codex_agent: Any,
    arena_factory: Callable,
    environment_factory: Callable,
    protocol_factory: Callable,
    nomic_integration: Any | None = None,
    kilocode_available: bool = False,
    kilocode_agent_factory: Callable | None = None,
    cycle_count: int = 0,
    log_fn: Callable[[str], None] | None = None,
    stream_emit_fn: Callable | None = None,
    record_replay_fn: Callable | None = None,
    auto_commit: bool = False,
) -> dict[NomicState, StateHandler]:
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
    from aragora.nomic.phases import (
        CommitPhase,
        ContextPhase,
        DebatePhase,
        DesignPhase,
        ImplementPhase,
        VerifyPhase,
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

    executor: Any = None
    if os.environ.get("NOMIC_USE_GASTOWN_EXECUTOR", "1") == "1":
        try:
            from aragora.nomic.convoy_executor import GastownConvoyExecutor

            implementers = [a for a in agents if a is not None]
            for extra in (claude_agent, codex_agent):
                if extra and extra not in implementers:
                    implementers.append(extra)

            executor = GastownConvoyExecutor(
                repo_path=aragora_path,
                implementers=implementers,
                reviewers=implementers,
                log_fn=log_fn,
                stream_emit_fn=stream_emit_fn,
            )
            if log_fn:
                log_fn(
                    f"  [implement] GastownConvoyExecutor created with {len(implementers)} agents"
                )
        except (ImportError, RuntimeError, TypeError) as exc:
            logger.warning("Failed to initialize GastownConvoyExecutor: %s", exc)
            executor = None

    if executor is None:
        try:
            from aragora.nomic.implement_executor import ConvoyImplementExecutor

            implementer_names = [getattr(a, "name", "") for a in agents if a is not None]

            def _agent_factory(name: str):
                for agent in agents:
                    if getattr(agent, "name", "") == name:
                        return agent
                return agents[0] if agents else None

            executor = ConvoyImplementExecutor(
                aragora_path=aragora_path,
                agents=[n for n in implementer_names if n],
                agent_factory=_agent_factory if agents else None,
                max_parallel=4,
                enable_cross_check=True,
                log_fn=log_fn,
            )
            if log_fn:
                log_fn(
                    f"  [implement] ConvoyImplementExecutor created with "
                    f"{len(implementer_names)} agents"
                )
        except (ImportError, RuntimeError, TypeError) as exc:
            logger.warning("Failed to initialize ConvoyImplementExecutor: %s", exc)
            executor = None

    async def _generate_implement_plan(design: str, repo_path: Path):
        from aragora.implement import create_single_task_plan, generate_implement_plan

        try:
            return await generate_implement_plan(design, repo_path)
        except (RuntimeError, OSError, ValueError) as exc:
            if log_fn:
                log_fn(f"  [implement] Plan generation failed, using fallback: {exc}")
            return create_single_task_plan(design, repo_path)

    # Create approval gates when not auto-committing
    design_gate = None
    test_quality_gate = None
    commit_gate = None
    if not auto_commit:
        try:
            from aragora.nomic.gates import (
                DesignGate,
                TestQualityGate,
                CommitGate,
            )

            design_gate = DesignGate(
                enabled=True,
                auto_approve_dev=os.environ.get("ARAGORA_DEV_MODE", "0") == "1",
            )
            test_quality_gate = TestQualityGate(
                enabled=True,
                require_all_tests_pass=True,
            )
            commit_gate = CommitGate(
                enabled=True,
                aragora_path=aragora_path,
            )
        except ImportError:
            pass

    implement_phase = ImplementPhase(
        aragora_path=aragora_path,
        plan_generator=_generate_implement_plan,
        executor=executor,
        cycle_count=cycle_count,
        log_fn=log_fn,
        stream_emit_fn=stream_emit_fn,
        record_replay_fn=record_replay_fn,
        design_gate=design_gate,
    )

    verify_phase = VerifyPhase(
        aragora_path=aragora_path,
        codex=codex_agent,
        nomic_integration=nomic_integration,
        cycle_count=cycle_count,
        log_fn=log_fn,
        stream_emit_fn=stream_emit_fn,
        record_replay_fn=record_replay_fn,
        test_quality_gate=test_quality_gate,
    )

    commit_phase = CommitPhase(
        aragora_path=aragora_path,
        require_human_approval=not auto_commit,
        auto_commit=auto_commit,
        cycle_count=cycle_count,
        log_fn=log_fn,
        stream_emit_fn=stream_emit_fn,
        commit_gate=commit_gate,
    )

    sica_agents: list[Any] = []
    for agent in (codex_agent, claude_agent, *agents):
        if agent and agent not in sica_agents:
            sica_agents.append(agent)
    recovery_manager = RecoveryManager()

    # Create and return bound handlers
    return {
        NomicState.CONTEXT: create_context_handler(context_phase),
        NomicState.DEBATE: create_debate_handler(debate_phase),
        NomicState.DESIGN: create_design_handler(design_phase),
        NomicState.IMPLEMENT: create_implement_handler(implement_phase),
        NomicState.VERIFY: create_verify_handler(
            verify_phase,
            repo_path=aragora_path,
            sica_agents=sica_agents,
        ),
        NomicState.COMMIT: create_commit_handler(commit_phase),
        NomicState.RECOVERY: create_recovery_handler(
            recovery_manager,
            repo_path=aragora_path,
            sica_agents=sica_agents,
        ),
    }


__all__ = [
    # Individual handler factories
    "create_context_handler",
    "create_debate_handler",
    "create_design_handler",
    "create_implement_handler",
    "create_verify_handler",
    "create_recovery_handler",
    "create_commit_handler",
    # Main factory
    "create_handlers",
    # Raw handlers (for custom phase instances)
    "context_handler",
    "debate_handler",
    "design_handler",
    "implement_handler",
    "verify_handler",
    "recovery_state_handler",
    "commit_handler",
]
