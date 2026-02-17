#!/usr/bin/env python3
"""
Self-Development CLI - Invoke HardenedOrchestrator with a high-level goal.

This script provides a command-line interface to Aragora's autonomous
development pipeline, which can:
- Decompose high-level goals into subtasks (heuristic or debate-based)
- Route tasks to appropriate agents by domain
- Execute improvements across multiple tracks in parallel
- Handle failures with retry and escalation
- Optionally use MetaPlanner for debate-driven goal prioritization
- Enforce mode constraints (architect/coder/reviewer) per phase
- Scan for prompt injection before execution
- Track budget and reconcile cross-agent file overlaps
- Route through the DecisionPlan pipeline for risk registers, receipts, and KM ingestion

Usage:
    # Dry run with heuristic decomposition (fast, needs concrete goals)
    python scripts/self_develop.py --goal "Refactor auth.py" --dry-run

    # Dry run with debate decomposition (slower, works with abstract goals)
    python scripts/self_develop.py --goal "Maximize utility for SME" --dry-run --debate

    # Run with approval gates (hardened mode is the default)
    python scripts/self_develop.py --goal "Improve error handling" --require-approval

    # Full autonomous run with worktree isolation
    python scripts/self_develop.py --goal "Enhance SME experience" --tracks sme developer --worktree

    # Use MetaPlanner for debate-driven prioritization before execution
    python scripts/self_develop.py --goal "Maximize utility" --meta-plan --debate

    # Route through the DecisionPlan pipeline (risk registers, receipts, KM)
    python scripts/self_develop.py --goal "Improve error handling" --use-pipeline

    # Use pipeline with hybrid execution mode (Claude + Codex)
    python scripts/self_develop.py --goal "Refactor auth" --use-pipeline --pipeline-mode hybrid

    # Fall back to base orchestrator (no hardening)
    python scripts/self_develop.py --goal "Simple fix" --standard
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from aragora.nomic.autonomous_orchestrator import (
    OrchestrationResult,
    Track,
)
from aragora.nomic.hardened_orchestrator import HardenedOrchestrator
from aragora.nomic.task_decomposer import TaskDecomposer, TaskDecomposition

logger = logging.getLogger(__name__)


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(text)
    print("=" * 60)


def print_decomposition(result: TaskDecomposition) -> None:
    """Print goal decomposition analysis."""
    print_header("GOAL ANALYSIS")
    print(f"Goal: {result.original_task}")
    print(f"Complexity: {result.complexity_level} ({result.complexity_score}/10)")
    print(f"Should decompose: {result.should_decompose}")
    print(
        f"Rationale: {result.rationale[:200]}..."
        if len(result.rationale) > 200
        else f"Rationale: {result.rationale}"
    )

    if result.subtasks:
        print(f"\nSubtasks ({len(result.subtasks)}):")
        for i, st in enumerate(result.subtasks, 1):
            print(f"  {i}. [{st.estimated_complexity}] {st.title}")
            print(f"     {st.description}")
            if st.file_scope:
                print(f"     Files: {', '.join(st.file_scope)}")
            if st.dependencies:
                print(f"     Depends on: {', '.join(st.dependencies)}")
    else:
        print("\nNo subtasks generated (goal may be simple enough to handle directly)")


async def run_debate_decomposition(goal: str) -> TaskDecomposition:
    """Run debate-based decomposition for abstract goals."""
    print_header("DEBATE DECOMPOSITION")
    print("Using multi-agent debate to decompose goal...")
    print("(This may take a minute as agents discuss what improvements would best serve the goal)")
    print()

    decomposer = TaskDecomposer()
    return await decomposer.analyze_with_debate(goal)


def run_heuristic_decomposition(goal: str) -> TaskDecomposition:
    """Run fast heuristic decomposition."""
    decomposer = TaskDecomposer()
    return decomposer.analyze(goal)


async def run_pipeline_execution(
    goal: str,
    use_debate: bool = False,
    pipeline_mode: str = "hybrid",
    budget_limit: float | None = None,
) -> Any:
    """Decompose goal and execute via the DecisionPlan pipeline.

    This routes through NomicPipelineBridge -> DecisionPlanFactory ->
    PlanExecutor, giving self-improvement access to risk registers,
    verification plans, execution receipts, and KM ingestion.

    Args:
        goal: The high-level goal.
        use_debate: Use debate-based decomposition.
        pipeline_mode: Execution mode for PlanExecutor.
        budget_limit: Optional budget cap in USD.

    Returns:
        A PlanOutcome from PlanExecutor.
    """
    from pathlib import Path

    from aragora.nomic.pipeline_bridge import NomicPipelineBridge

    # Step 1: Decompose
    if use_debate:
        decomposition = await run_debate_decomposition(goal)
    else:
        decomposition = run_heuristic_decomposition(goal)

    print_decomposition(decomposition)

    if not decomposition.subtasks:
        print("\nNo subtasks to execute via pipeline.")
        return None

    # Step 2: Route through the pipeline
    bridge = NomicPipelineBridge(
        repo_path=Path.cwd(),
        budget_limit_usd=budget_limit,
        execution_mode=pipeline_mode,
    )

    print_header("PIPELINE EXECUTION")
    print(f"Execution mode: {pipeline_mode}")
    print(f"Subtasks: {len(decomposition.subtasks)}")
    if budget_limit:
        print(f"Budget limit: ${budget_limit:.2f}")

    plan = bridge.build_decision_plan(
        goal=goal,
        subtasks=decomposition.subtasks,
    )

    print(f"\nDecisionPlan: {plan.id}")
    print(f"Status: {plan.status.value}")
    if plan.risk_register:
        print(f"Risks: {len(plan.risk_register.risks)}")
        for risk in plan.risk_register.risks[:5]:
            print(f"  [{risk.level.value}] {risk.title}")
    if plan.verification_plan:
        test_count = len(plan.verification_plan.test_cases)
        print(f"Verification cases: {test_count}")
    if plan.implement_plan:
        print(f"Implementation tasks: {len(plan.implement_plan.tasks)}")

    print("\nExecuting plan...")

    outcome = await bridge.execute_via_pipeline(
        goal=goal,
        subtasks=decomposition.subtasks,
        execution_mode=pipeline_mode,
    )

    return outcome


def print_pipeline_outcome(outcome: Any) -> None:
    """Print pipeline execution outcome."""
    print_header("PIPELINE EXECUTION COMPLETE")
    print(f"Status: {'SUCCESS' if outcome.success else 'FAILED'}")
    print(f"Tasks completed: {outcome.tasks_completed}/{outcome.tasks_total}")
    if outcome.verification_passed or outcome.verification_total:
        print(
            f"Verification: {outcome.verification_passed}/{outcome.verification_total} passed"
        )
    if outcome.total_cost_usd > 0:
        print(f"Cost: ${outcome.total_cost_usd:.4f}")
    print(f"Duration: {outcome.duration_seconds:.1f}s")
    if outcome.receipt_id:
        print(f"Receipt: {outcome.receipt_id}")
    if outcome.lessons:
        print("\nLessons learned:")
        for lesson in outcome.lessons:
            print(f"  - {lesson}")
    if outcome.error:
        print(f"\nError: {outcome.error}")


def print_result(result: OrchestrationResult) -> None:
    """Print orchestration result summary."""
    print_header("ORCHESTRATION COMPLETE")
    print(result.summary)
    print(f"\nStatus: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Completed: {result.completed_subtasks}/{result.total_subtasks}")
    print(f"Failed: {result.failed_subtasks}")
    print(f"Skipped: {result.skipped_subtasks}")
    print(f"Duration: {result.duration_seconds:.1f}s")

    if result.error:
        print(f"\nError: {result.error}")


def create_checkpoint_handler(require_approval: bool):
    """Create a checkpoint callback handler."""

    def on_checkpoint(phase: str, data: dict[str, Any]) -> None:
        print_header(f"CHECKPOINT: {phase.upper()}")

        # Print checkpoint data
        for key, value in data.items():
            if key in ("orchestration_id", "timestamp"):
                print(f"  {key}: {value}")
            elif key == "subtask_count":
                print(f"  Subtasks: {value}")
            elif key == "assignment_count":
                print(f"  Assignments: {value}")
            elif key == "result":
                print(f"  Result: {value}")
            else:
                print(f"  {key}: {value}")

        if require_approval:
            print("\nThis checkpoint requires your approval.")
            while True:
                response = input("Approve and continue? [y/n]: ").strip().lower()
                if response == "y":
                    print("Approved. Continuing...")
                    return
                elif response == "n":
                    print("Rejected. Aborting orchestration.")
                    raise KeyboardInterrupt("User rejected checkpoint")
                else:
                    print("Please enter 'y' or 'n'")

    return on_checkpoint


async def run_orchestration(
    goal: str,
    tracks: list[str] | None,
    max_cycles: int,
    max_parallel: int,
    require_approval: bool,
    use_debate: bool = False,
    use_worktree: bool = False,
    use_standard: bool = False,
    use_parallel: bool = False,
    enable_gauntlet: bool = True,
    enable_meta_plan: bool = False,
    budget_limit: float | None = None,
    repo_path: Path | None = None,
) -> OrchestrationResult:
    """Run the autonomous orchestration.

    Default mode is HARDENED (mode enforcement, prompt defense, gauntlet,
    audit reconciliation). Use --standard to fall back to the base
    AutonomousOrchestrator.
    """
    common_kwargs: dict[str, Any] = {
        "require_human_approval": require_approval,
        "max_parallel_tasks": max_parallel,
        "on_checkpoint": create_checkpoint_handler(require_approval),
        "use_debate_decomposition": use_debate,
    }
    if repo_path is not None:
        common_kwargs["aragora_path"] = repo_path

    if use_parallel:
        from aragora.nomic.parallel_orchestrator import ParallelOrchestrator

        orchestrator = ParallelOrchestrator(
            use_worktrees=use_worktree,
            enable_gauntlet=enable_gauntlet,
            enable_convoy_tracking=True,
            budget_limit_usd=budget_limit,
            **common_kwargs,
        )
        mode_label = "PARALLEL"
    elif use_standard:
        from aragora.nomic.autonomous_orchestrator import AutonomousOrchestrator

        orchestrator = AutonomousOrchestrator(**common_kwargs)
        mode_label = "STANDARD"
    else:
        # Default: HardenedOrchestrator with production features
        orchestrator = HardenedOrchestrator(
            use_worktree_isolation=use_worktree,
            enable_meta_planning=enable_meta_plan,
            budget_limit_usd=budget_limit,
            **common_kwargs,
        )
        mode_label = "HARDENED"

    print_header(f"STARTING ORCHESTRATION ({mode_label})")
    print(f"Goal: {goal}")
    if repo_path:
        print(f"Repository: {repo_path}")
    print(f"Tracks: {tracks if tracks else 'all'}")
    print(f"Max cycles per subtask: {max_cycles}")
    print(f"Max parallel tasks: {max_parallel}")
    print(f"Require approval: {require_approval}")
    if use_parallel:
        print(f"Worktree isolation: {use_worktree}")
        print(f"Gauntlet gate: {enable_gauntlet}")
        print(f"Convoy tracking: enabled")
    elif not use_standard:
        print(f"Worktree isolation: {use_worktree}")
        print(f"Meta-planning: {enable_meta_plan}")
        if budget_limit:
            print(f"Budget limit: ${budget_limit:.2f}")

    result = await orchestrator.execute_goal(
        goal=goal,
        tracks=tracks,
        max_cycles=max_cycles,
    )

    # Clean up worktrees if using parallel orchestrator
    if use_parallel and hasattr(orchestrator, "cleanup"):
        await orchestrator.cleanup()

    return result


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Aragora self-development with a high-level goal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview with heuristic decomposition (fast, concrete goals)
  %(prog)s --goal "Refactor dashboard.tsx and api.py" --dry-run

  # Preview with debate decomposition (slower, abstract goals)
  %(prog)s --goal "Maximize utility for SME businesses" --dry-run --debate

  # Run with human approval at each checkpoint
  %(prog)s --goal "Improve test coverage" --require-approval

  # Focus on specific tracks
  %(prog)s --goal "Enhance SDK" --tracks developer qa

  # Full autonomous run
  %(prog)s --goal "Improve SME experience" --tracks sme developer --max-parallel 2
        """,
    )

    parser.add_argument(
        "--goal",
        required=True,
        help="High-level goal to achieve (e.g., 'Improve error handling')",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Path to the target repository (default: current directory). "
        "Enables running the Nomic Loop on external codebases.",
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=[t.value for t in Track],
        help=f"Tracks to focus on. Choices: {', '.join(t.value for t in Track)}",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=5,
        help="Max improvement cycles per subtask (default: 5)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help="Max parallel tasks across all tracks (default: 4)",
    )
    parser.add_argument(
        "--require-approval",
        action="store_true",
        help="Require human approval at checkpoints",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show goal decomposition without executing",
    )
    parser.add_argument(
        "--debate",
        action="store_true",
        help="Use multi-agent debate for goal decomposition (slower but works with abstract goals)",
    )
    parser.add_argument(
        "--worktree",
        action="store_true",
        help="Use git worktree isolation for parallel agent execution",
    )
    parser.add_argument(
        "--standard",
        action="store_true",
        help="Use base AutonomousOrchestrator without hardening (no mode enforcement, no prompt defense)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use ParallelOrchestrator with worktree isolation, gauntlet gate, and convoy tracking",
    )
    parser.add_argument(
        "--meta-plan",
        action="store_true",
        help="Use MetaPlanner for debate-driven goal prioritization before decomposition",
    )
    parser.add_argument(
        "--gauntlet",
        action="store_true",
        default=True,
        help="Enable adversarial gauntlet gate between design and implement (default: on, use --no-gauntlet to disable)",
    )
    parser.add_argument(
        "--no-gauntlet",
        action="store_true",
        help="Disable adversarial gauntlet gate",
    )
    parser.add_argument(
        "--budget-limit",
        type=float,
        default=None,
        help="Maximum cost in USD for the entire run (requires --hardened or --parallel)",
    )
    parser.add_argument(
        "--use-pipeline",
        action="store_true",
        help="Route subtasks through the DecisionPlan pipeline (risk registers, receipts, KM ingestion)",
    )
    parser.add_argument(
        "--pipeline-mode",
        type=str,
        default="hybrid",
        choices=["workflow", "hybrid", "fabric", "computer_use"],
        help="Execution mode when --use-pipeline is enabled (default: hybrid)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging — only enable DEBUG for aragora loggers, NOT third-party
    # libraries like botocore which dump secrets in HTTP response bodies at DEBUG level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Set aragora loggers to requested level
    logging.getLogger("aragora").setLevel(log_level)
    # Suppress noisy/sensitive third-party loggers
    for noisy in ("botocore", "boto3", "urllib3", "asyncio", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Dry run: just show decomposition
    if args.dry_run:
        if args.debate:
            # Use debate-based decomposition (async)
            try:
                result = asyncio.run(run_debate_decomposition(args.goal))
            except RuntimeError as e:
                if "No API agents available" in str(e):
                    print(f"[!] Debate mode requires API keys: {e}")
                    print("[!] Falling back to heuristic decomposition...\n")
                    result = run_heuristic_decomposition(args.goal)
                else:
                    raise
        else:
            # Use fast heuristic decomposition
            result = run_heuristic_decomposition(args.goal)

        print_decomposition(result)
        return 0

    # Pipeline mode: decompose then execute via DecisionPlan pipeline
    if args.use_pipeline:
        try:
            outcome = asyncio.run(
                run_pipeline_execution(
                    goal=args.goal,
                    use_debate=args.debate,
                    pipeline_mode=args.pipeline_mode,
                    budget_limit=args.budget_limit,
                )
            )
            if outcome is None:
                print("\nNo outcome (no subtasks generated).")
                return 0
            print_pipeline_outcome(outcome)
            return 0 if outcome.success else 1

        except KeyboardInterrupt:
            print("\n\nPipeline execution cancelled by user.")
            return 130

        except Exception as e:
            logger.exception("Pipeline execution failed with error")
            print(f"\nError: {e}")
            return 1

    # Resolve gauntlet flag (--no-gauntlet overrides --gauntlet)
    enable_gauntlet = not args.no_gauntlet

    # --parallel implies --worktree unless explicitly disabled
    use_worktree = args.worktree or args.parallel

    # Resolve repo path
    resolved_repo_path = Path(args.repo).resolve() if args.repo else None

    # Full run
    try:
        result = asyncio.run(
            run_orchestration(
                goal=args.goal,
                tracks=args.tracks,
                max_cycles=args.max_cycles,
                max_parallel=args.max_parallel,
                require_approval=args.require_approval,
                use_debate=args.debate,
                use_worktree=use_worktree,
                use_standard=args.standard,
                use_parallel=args.parallel,
                enable_gauntlet=enable_gauntlet,
                enable_meta_plan=args.meta_plan,
                budget_limit=args.budget_limit,
                repo_path=resolved_repo_path,
            )
        )
        print_result(result)
        return 0 if result.success else 1

    except KeyboardInterrupt:
        print("\n\nOrchestration cancelled by user.")
        return 130

    except Exception as e:
        logger.exception("Orchestration failed with error")
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
