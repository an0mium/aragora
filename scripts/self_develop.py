#!/usr/bin/env python3
"""
Self-Development CLI - Invoke AutonomousOrchestrator with a high-level goal.

This script provides a command-line interface to Aragora's autonomous
development pipeline, which can:
- Decompose high-level goals into subtasks (heuristic or debate-based)
- Route tasks to appropriate agents by domain
- Execute improvements across multiple tracks in parallel
- Handle failures with retry and escalation

Usage:
    # Dry run with heuristic decomposition (fast, needs concrete goals)
    python scripts/self_develop.py --goal "Refactor auth.py" --dry-run

    # Dry run with debate decomposition (slower, works with abstract goals)
    python scripts/self_develop.py --goal "Maximize utility for SME" --dry-run --debate

    # Run with approval gates
    python scripts/self_develop.py --goal "Improve error handling" --require-approval

    # Full autonomous run
    python scripts/self_develop.py --goal "Enhance SME experience" --tracks sme developer
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any, Dict

from aragora.nomic.autonomous_orchestrator import (
    AutonomousOrchestrator,
    OrchestrationResult,
    Track,
)
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

    def on_checkpoint(phase: str, data: Dict[str, Any]) -> None:
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
) -> OrchestrationResult:
    """Run the autonomous orchestration."""
    orchestrator = AutonomousOrchestrator(
        require_human_approval=require_approval,
        max_parallel_tasks=max_parallel,
        on_checkpoint=create_checkpoint_handler(require_approval),
    )

    print_header("STARTING ORCHESTRATION")
    print(f"Goal: {goal}")
    print(f"Tracks: {tracks if tracks else 'all'}")
    print(f"Max cycles per subtask: {max_cycles}")
    print(f"Max parallel tasks: {max_parallel}")
    print(f"Require approval: {require_approval}")

    return await orchestrator.execute_goal(
        goal=goal,
        tracks=tracks,
        max_cycles=max_cycles,
    )


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
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Dry run: just show decomposition
    if args.dry_run:
        if args.debate:
            # Use debate-based decomposition (async)
            result = asyncio.run(run_debate_decomposition(args.goal))
        else:
            # Use fast heuristic decomposition
            result = run_heuristic_decomposition(args.goal)

        print_decomposition(result)
        return 0

    # Full run
    try:
        result = asyncio.run(
            run_orchestration(
                goal=args.goal,
                tracks=args.tracks,
                max_cycles=args.max_cycles,
                max_parallel=args.max_parallel,
                require_approval=args.require_approval,
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
