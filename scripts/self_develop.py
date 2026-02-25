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

    # Run on an external codebase (customer repo)
    python scripts/self_develop.py --goal "Improve test coverage" --repo /path/to/customer/repo --dry-run

    # Use MetaPlanner for debate-driven prioritization before execution
    python scripts/self_develop.py --goal "Maximize utility" --meta-plan --debate

    # Route through the DecisionPlan pipeline (risk registers, receipts, KM)
    python scripts/self_develop.py --goal "Improve error handling" --use-pipeline

    # Use pipeline with hybrid execution mode (Claude + Codex)
    python scripts/self_develop.py --goal "Refactor auth" --use-pipeline --pipeline-mode hybrid

    # Auto mode: low-risk auto-execute, budget capped at 10 files
    python scripts/self_develop.py --goal "Fix lint and test issues" --auto

    # Fall back to base orchestrator (no hardening)
    python scripts/self_develop.py --goal "Simple fix" --standard
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from aragora.nomic.autonomous_orchestrator import (
    OrchestrationResult,
    Track,
)
from aragora.nomic.cli_stream_bridge import CLIStreamBridge
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


def _validate_pipeline(goal: str) -> int:
    """Validate all pipeline components without executing anything.

    Probes imports, decomposition, bridge, indexer, evaluator, and debug loop.
    Prints a structured report with pass/fail for each component.

    Returns 0 if all checks pass, 1 if any fail.
    """
    print_header("PIPELINE VALIDATION")
    checks: list[tuple[str, bool, str]] = []

    # 1. TaskDecomposer
    try:
        decomposer = TaskDecomposer()
        result = decomposer.analyze(goal)
        checks.append(
            (
                "TaskDecomposer",
                True,
                f"score={result.complexity_score}, subtasks={len(result.subtasks)}",
            )
        )
    except Exception as e:
        checks.append(("TaskDecomposer", False, str(e)[:80]))

    # 2. ExecutionBridge
    try:
        from aragora.nomic.execution_bridge import ExecutionBridge
        from aragora.nomic.task_decomposer import SubTask

        bridge = ExecutionBridge()
        dummy = SubTask(id="val_1", title="Validate", description="Pipeline check", file_scope=[])
        instr = bridge.create_instruction(dummy)
        prompt = instr.to_agent_prompt()
        checks.append(("ExecutionBridge", True, f"prompt={len(prompt)} chars"))
    except Exception as e:
        checks.append(("ExecutionBridge", False, str(e)[:80]))

    # 3. SelfImprovePipeline
    try:
        from aragora.nomic.self_improve import SelfImprovePipeline, SelfImproveConfig

        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                enable_codebase_indexing=False,
                capture_metrics=False,
                persist_outcomes=False,
            )
        )
        checks.append(("SelfImprovePipeline", True, "importable + configurable"))
    except Exception as e:
        checks.append(("SelfImprovePipeline", False, str(e)[:80]))

    # 4. GoalEvaluator
    try:
        from aragora.nomic.goal_evaluator import GoalEvaluator

        ev = GoalEvaluator()
        score = ev.evaluate(goal=goal, files_changed=["test.py"])
        checks.append(("GoalEvaluator", True, f"score={score.achievement_score:.2f}"))
    except Exception as e:
        checks.append(("GoalEvaluator", False, str(e)[:80]))

    # 5. DebugLoop
    try:
        from aragora.nomic.debug_loop import DebugLoop, DebugLoopConfig

        dl = DebugLoop(DebugLoopConfig(max_retries=1))
        checks.append(("DebugLoop", True, f"max_retries={dl.config.max_retries}"))
    except Exception as e:
        checks.append(("DebugLoop", False, str(e)[:80]))

    # 6. MetaPlanner
    try:
        from aragora.nomic.meta_planner import MetaPlanner, MetaPlannerConfig

        mp = MetaPlanner(MetaPlannerConfig(scan_mode=True))
        checks.append(("MetaPlanner", True, "scan_mode ready"))
    except Exception as e:
        checks.append(("MetaPlanner", False, str(e)[:80]))

    # 7. BranchCoordinator
    try:
        from aragora.nomic.branch_coordinator import BranchCoordinator, BranchCoordinatorConfig

        bc = BranchCoordinator(config=BranchCoordinatorConfig(use_worktrees=True))
        checks.append(("BranchCoordinator", True, f"worktrees={bc.config.use_worktrees}"))
    except Exception as e:
        checks.append(("BranchCoordinator", False, str(e)[:80]))

    # 8. Claude Code CLI
    import shutil

    claude_path = shutil.which("claude")
    if claude_path:
        checks.append(("Claude CLI", True, claude_path))
    else:
        checks.append(("Claude CLI", False, "not found in PATH"))

    # Print report
    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print(f"\nGoal: {goal[:80]}")
    print(f"Components: {passed}/{total} passed\n")

    for name, ok, detail in checks:
        icon = "PASS" if ok else "FAIL"
        print(f"  [{icon}] {name}: {detail}")

    print()
    if passed == total:
        print("All pipeline components validated successfully.")
    else:
        failed_names = [name for name, ok, _ in checks if not ok]
        print(f"Failed: {', '.join(failed_names)}")
        print("Fix these components before running --self-improve --autonomous")

    return 0 if passed == total else 1


async def run_debate_decomposition(goal: str) -> TaskDecomposition:
    """Run debate-based decomposition for abstract goals."""
    print_header("DEBATE DECOMPOSITION")
    print("Using multi-agent debate to decompose goal...")
    print("(This may take a minute as agents discuss what improvements would best serve the goal)")
    print()

    decomposer = TaskDecomposer()
    return await decomposer.analyze_with_debate(goal)


def run_heuristic_decomposition(goal: str) -> TaskDecomposition:
    """Run fast heuristic decomposition.

    If the goal is abstract (high complexity, no file hints) and
    ``auto_debate_abstract`` is enabled, this annotates the result
    with ``recommend_debate=True`` so callers can decide to re-run
    with debate mode.
    """
    decomposer = TaskDecomposer()
    return decomposer.analyze(goal)


async def run_pipeline_execution(
    goal: str,
    use_debate: bool = False,
    pipeline_mode: str = "hybrid",
    budget_limit: float | None = None,
    repo_path: Path | None = None,
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
        repo_path=repo_path or Path.cwd(),
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

    # Run outcome feedback cycle to detect systematic errors and queue
    # improvement goals for future Nomic Loop iterations
    try:
        from aragora.nomic.outcome_feedback import OutcomeFeedbackBridge

        feedback = OutcomeFeedbackBridge()
        cycle = feedback.run_feedback_cycle()
        goals_generated = cycle.get("goals_generated", 0)
        if goals_generated > 0:
            print(f"\nOutcome feedback: {goals_generated} improvement goals queued")
            for domain in cycle.get("domains_flagged", []):
                print(f"  Domain flagged: {domain}")
    except ImportError:
        pass
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
        logger.debug("Outcome feedback cycle failed (non-critical): %s", exc)

    return outcome


def print_pipeline_outcome(outcome: Any) -> None:
    """Print pipeline execution outcome."""
    print_header("PIPELINE EXECUTION COMPLETE")
    print(f"Status: {'SUCCESS' if outcome.success else 'FAILED'}")
    print(f"Tasks completed: {outcome.tasks_completed}/{outcome.tasks_total}")
    if outcome.verification_passed or outcome.verification_total:
        print(f"Verification: {outcome.verification_passed}/{outcome.verification_total} passed")
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

    if result.metrics_delta:
        print_header("METRICS COMPARISON")
        delta = result.metrics_delta
        print(f"  Improvement score: {delta.get('improvement_score', 0):.2f}")
        print(f"  Improved: {delta.get('improved', False)}")
        print(f"  Summary: {delta.get('summary', 'N/A')}")
        if delta.get("tests_passed_delta", 0) != 0:
            print(f"  Tests passed delta: {delta['tests_passed_delta']:+d}")
        if delta.get("tests_failed_delta", 0) != 0:
            print(f"  Tests failed delta: {delta['tests_failed_delta']:+d}")
        if delta.get("lint_errors_delta", 0) != 0:
            print(f"  Lint errors delta: {delta['lint_errors_delta']:+d}")


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


async def _enrich_abstract_goal(goal: str, tracks: list[str] | None) -> str:
    """Enrich an abstract goal with real codebase signals from MetaPlanner scan.

    When self_develop.py receives an abstract goal (no file mentions, broad scope),
    this function runs MetaPlanner._scan_prioritize() to gather signals from
    git log, pytest cache, ruff violations, and TODOs, then prepends a summary
    to the goal string so the decomposer and orchestrator have concrete context.

    Args:
        goal: The original abstract goal string.
        tracks: Optional track names to filter signals.

    Returns:
        Enriched goal string with codebase signal summary prepended,
        or the original goal if scanning fails.
    """
    try:
        from aragora.nomic.meta_planner import MetaPlanner, MetaPlannerConfig, Track

        available_tracks = [Track(t) for t in tracks] if tracks else list(Track)

        planner = MetaPlanner(MetaPlannerConfig(scan_mode=True))
        scan_goals = await planner._scan_prioritize(goal, available_tracks)

        if not scan_goals:
            return goal

        # Build context summary from scan signals
        signal_lines: list[str] = []
        for sg in scan_goals[:3]:
            # Extract the signal summary (first line of description)
            desc_first_line = sg.description.split("\n")[0]
            signal_lines.append(f"- [{sg.track.value}] {desc_first_line}")

        if signal_lines:
            context_block = (
                "CODEBASE SIGNALS (from scan):\n"
                + "\n".join(signal_lines)
                + "\n\nORIGINAL GOAL: "
                + goal
            )
            logger.info(
                "abstract_goal_enriched signals=%d tracks=%s",
                len(scan_goals),
                [sg.track.value for sg in scan_goals[:3]],
            )
            return context_block

    except (ImportError, RuntimeError, ValueError, OSError) as e:
        logger.debug("Abstract goal enrichment failed (non-critical): %s", e)

    return goal


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
    enable_metrics: bool = False,
    enable_preflight: bool = False,
    enable_stuck_detection: bool = False,
    enable_watchdog: bool = False,
    auto_execute_low_risk: bool = False,
) -> OrchestrationResult:
    """Run the autonomous orchestration.

    Default mode is HARDENED (mode enforcement, prompt defense, gauntlet,
    audit reconciliation). Use --standard to fall back to the base
    AutonomousOrchestrator.

    For abstract goals, automatically enriches the goal context with
    codebase signals from MetaPlanner scan mode before execution.
    """
    # Enrich abstract goals with codebase signals before execution
    decomposer = TaskDecomposer()
    preliminary = decomposer.analyze(goal)
    if preliminary.recommend_debate and not use_debate:
        # Goal is abstract -- gather codebase signals to ground it
        print("[*] Abstract goal detected, scanning codebase for context...")
        goal = await _enrich_abstract_goal(goal, tracks)
        # Also enable debate mode so the orchestrator uses debate decomposition
        use_debate = True

    common_kwargs: dict[str, Any] = {
        "require_human_approval": require_approval,
        "max_parallel_tasks": max_parallel,
        "on_checkpoint": create_checkpoint_handler(require_approval),
        "use_debate_decomposition": use_debate,
        "enable_metrics": enable_metrics,
        "enable_preflight": enable_preflight,
        "enable_stuck_detection": enable_stuck_detection,
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
            enable_watchdog=enable_watchdog,
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
    if auto_execute_low_risk:
        print("Auto-execute low-risk: enabled (test fixes, doc updates, lint)")
    if use_parallel:
        print(f"Worktree isolation: {use_worktree}")
        print(f"Gauntlet gate: {enable_gauntlet}")
        print("Convoy tracking: enabled")
    elif not use_standard:
        print(f"Worktree isolation: {use_worktree}")
        print(f"Meta-planning: {enable_meta_plan}")
        if enable_watchdog:
            print("Stall watchdog: enabled")
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

  # Auto mode: low-risk goals execute without approval, budget capped at 10 files
  %(prog)s --goal "Fix lint and test issues" --auto
        """,
    )

    parser.add_argument(
        "--goal",
        required=False,
        default=None,
        help="High-level goal to achieve (e.g., 'Improve error handling'). "
        "Optional when --scan is used (MetaPlanner generates goals from codebase signals).",
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
        "--watchdog",
        action="store_true",
        help="Enable worktree stall watchdog to detect and recover stuck sessions",
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
        "--metrics",
        action="store_true",
        help="Collect test/lint/size metrics before and after to objectively measure improvement",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run preflight health checks (API keys, circuit breakers) before execution",
    )
    parser.add_argument(
        "--stuck-detection",
        action="store_true",
        help="Monitor running tasks for stalls and auto-recover stuck work",
    )
    parser.add_argument(
        "--feedback",
        action="store_true",
        default=True,
        help="Run outcome feedback cycle after execution (on by default; use --no-feedback to disable)",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        default=False,
        help="Disable the post-execution outcome feedback cycle",
    )
    parser.add_argument(
        "--self-improve",
        action="store_true",
        help="Use the unified SelfImprovePipeline (Plan→Execute→Verify→Learn) "
        "with Claude Code dispatch and worktree isolation",
    )
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Skip approval gates for fully autonomous execution (requires --self-improve)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Autonomous mode: sets require-approval=False and defaults budget-limit to 10 "
        "(max files changed per cycle). Low-risk goals (test fixes, doc updates, lint) "
        "auto-execute without approval; medium/high-risk goals still require review. "
        "Safer than --autonomous (which executes everything).",
    )
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        default=True,
        help="Enable risk-scored execution gating (default: on). "
        "Scores each goal before execution: auto/review/block based on risk threshold. "
        "Use --no-safe-mode to disable.",
    )
    parser.add_argument(
        "--no-safe-mode",
        action="store_true",
        default=False,
        help="Disable risk-scored safe mode execution gating.",
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=0.5,
        help="Risk score threshold for auto-execution (default: 0.5). "
        "Goals scoring below this are auto-executed; above require review or are blocked.",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Use scan mode: prioritize from codebase signals (git log, untested modules, "
        "past regressions) without LLM calls (requires --self-improve or --meta-plan)",
    )
    parser.add_argument(
        "--visual-pipeline",
        action="store_true",
        help="Generate a visual Idea-to-Execution pipeline from MetaPlanner goals "
        "(viewable at /pipeline in the web UI). Works with --meta-plan or --self-improve.",
    )
    parser.add_argument(
        "--validate-pipeline",
        action="store_true",
        help="Validate that all pipeline components can be imported and initialized. "
        "Runs a non-executing probe: decomposition, bridge, indexer, and evaluator.",
    )
    parser.add_argument(
        "--assess",
        action="store_true",
        help="Run autonomous codebase assessment and print health report. "
        "No LLM calls — pure static analysis. Exits after assessment.",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Start the continuous self-improvement daemon. "
        "Runs assess→generate→execute→measure cycles on an interval.",
    )
    parser.add_argument(
        "--daemon-interval",
        type=float,
        default=3600.0,
        help="Seconds between daemon cycles (default: 3600 = 1 hour)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # --auto implies require_approval=False and a conservative budget-limit default
    if args.auto:
        args.require_approval = False
        if args.budget_limit is None:
            args.budget_limit = 10.0

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

    # Validate: --goal is required unless --scan, --assess, or --daemon is set
    if args.goal is None and not args.scan and not args.assess and not args.daemon:
        parser.error(
            "--goal is required unless --scan, --assess, or --daemon is used. "
            "Use --scan to let MetaPlanner generate goals from codebase signals, "
            "--assess for health assessment, --daemon for continuous improvement, "
            "or provide --goal 'your objective'."
        )

    # Validate pipeline: probe all components without execution
    if args.validate_pipeline:
        return _validate_pipeline(args.goal or "validate pipeline components")

    # Assess mode: run autonomous assessment and print health report
    if args.assess:
        try:
            from aragora.nomic.assessment_engine import AutonomousAssessmentEngine
            from aragora.nomic.goal_generator import GoalGenerator

            engine = AutonomousAssessmentEngine()
            report = asyncio.run(engine.assess())

            print_header("CODEBASE HEALTH ASSESSMENT")
            print(f"Health Score: {report.health_score:.2f}")
            print(f"Assessment Duration: {report.assessment_duration_seconds:.1f}s")
            print(f"\nSignal Sources ({len(report.signal_sources)}):")
            for src in report.signal_sources:
                status = f"{len(src.findings)} findings" if not src.error else f"ERROR: {src.error}"
                print(f"  [{src.name}] weight={src.weight:.2f}  {status}")

            print(f"\nImprovement Candidates ({len(report.improvement_candidates)}):")
            for i, c in enumerate(report.improvement_candidates[:10], 1):
                files_str = f" ({', '.join(c.files[:2])})" if c.files else ""
                print(f"  {i}. [{c.category}] {c.description}{files_str}")
                print(f"     priority={c.priority:.2f}  source={c.source}")

            if report.improvement_candidates:
                generator = GoalGenerator()
                goals = generator.generate_goals(report)
                if goals:
                    print(f"\nGenerated Goals ({len(goals)}):")
                    for g in goals:
                        print(f"  [{g.track.value}] {g.description}")

            return 0

        except (ImportError, RuntimeError, ValueError) as e:
            print(f"\nAssessment failed: {e}")
            return 1

    # Daemon mode: start continuous self-improvement loop
    if args.daemon:
        try:
            from aragora.nomic.daemon import DaemonConfig, SelfImprovementDaemon

            config = DaemonConfig(
                interval_seconds=args.daemon_interval,
                dry_run=args.dry_run,
                require_approval=args.require_approval,
                autonomous=args.autonomous,
                budget_limit_per_cycle_usd=args.budget_limit or 5.0,
                use_worktrees=args.worktree or args.parallel,
                run_tests=True,
            )
            daemon = SelfImprovementDaemon(config)

            print_header("SELF-IMPROVEMENT DAEMON")
            print(f"Interval: {config.interval_seconds:.0f}s")
            print(f"Health threshold: {config.health_threshold:.2f}")
            print(f"Budget per cycle: ${config.budget_limit_per_cycle_usd:.2f}")
            print(f"Budget cumulative: ${config.budget_limit_cumulative_usd:.2f}")
            print(f"Dry run: {config.dry_run}")
            print(f"Max consecutive failures: {config.max_consecutive_failures}")
            print("\nStarting daemon... (Ctrl+C to stop)")

            try:
                asyncio.run(daemon.start())
            except KeyboardInterrupt:
                print("\n\nDaemon stopped by user.")

            status = daemon.get_status()
            print(f"\nFinal status: {status.state}")
            print(f"Cycles completed: {status.cycles_completed}")
            print(f"Cycles failed: {status.cycles_failed}")
            return 0

        except (ImportError, RuntimeError, ValueError) as e:
            print(f"\nDaemon failed: {e}")
            return 1

    # Dry run: just show decomposition (unless --self-improve handles its own dry-run)
    if args.dry_run and not args.self_improve:
        # goal is guaranteed non-None here (validated above: --scan requires --self-improve for None goal)
        assert args.goal is not None, "--goal is required for dry-run without --self-improve"

        # Enrich goal with StrategicScanner findings when --scan is set
        enriched_goal = args.goal
        if args.scan:
            try:
                from aragora.nomic.strategic_scanner import StrategicScanner

                scanner = StrategicScanner()
                assessment = scanner.scan(objective=args.goal)
                if assessment.findings:
                    top_findings = assessment.findings[:5]
                    signals = "\n".join(
                        f"- [{f.category}] {f.file_path}: {f.description[:80]}"
                        for f in top_findings
                    )
                    enriched_goal = (
                        f"CODEBASE SIGNALS (from StrategicScanner):\n{signals}\n\n"
                        f"OBJECTIVE: {args.goal}"
                    )
                    print(
                        f"[scan] Found {len(assessment.findings)} findings, "
                        f"top areas: {', '.join(assessment.focus_areas[:3])}"
                    )
            except (ImportError, RuntimeError, ValueError, OSError) as exc:
                logger.debug("StrategicScanner not available for dry-run: %s", exc)

        use_debate = args.debate
        if not use_debate:
            # Run heuristic first; if the goal is abstract, auto-switch to debate
            result = run_heuristic_decomposition(enriched_goal)
            if result.recommend_debate:
                print(
                    "[*] Abstract goal detected (score={}/10, no file hints)".format(
                        result.complexity_score
                    )
                )
                print("[*] Auto-switching to debate-based decomposition...\n")
                use_debate = True

        if use_debate:
            # Use debate-based decomposition (async)
            try:
                result = asyncio.run(run_debate_decomposition(enriched_goal))
            except RuntimeError as e:
                if "No API agents available" in str(e):
                    print(f"[!] Debate mode requires API keys: {e}")
                    print("[!] Falling back to heuristic decomposition...\n")
                    result = run_heuristic_decomposition(enriched_goal)
                else:
                    raise

        print_decomposition(result)
        return 0

    # Resolve repo path (used by both pipeline and orchestration modes)
    resolved_repo = Path(args.repo).resolve() if args.repo else None

    # Ingest repo knowledge when --repo is provided
    if resolved_repo and not args.dry_run:
        try:
            from aragora.memory.codebase_builder import CodebaseKnowledgeBuilder
            from aragora.memory.fabric import MemoryFabric

            fabric = MemoryFabric()
            builder = CodebaseKnowledgeBuilder(fabric=fabric, repo_path=resolved_repo)
            ingested = asyncio.run(builder.ingest_structure())
            imports = asyncio.run(builder.ingest_imports())
            print(f"[repo] Ingested {ingested} structure entries, {imports} import relationships")
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            logger.debug("Codebase knowledge ingestion skipped: %s", e)

    # Progress callback for self-improve pipeline -- use CLIStreamBridge
    # to emit events to both stdout and WebSocket stream servers.
    stream_bridge = CLIStreamBridge(
        nomic_port=8767,
        pipeline_id=f"self-develop-{uuid.uuid4().hex[:8]}",
        print_to_stdout=True,
    )
    _print_progress = stream_bridge.as_progress_callback()

    # Self-improve mode: unified pipeline with Claude Code dispatch
    if args.self_improve:
        try:
            from aragora.nomic.self_improve import SelfImprovePipeline, SelfImproveConfig

            safe_mode_enabled = args.safe_mode and not args.no_safe_mode
            config = SelfImproveConfig(
                use_meta_planner=args.meta_plan or args.debate or args.scan,
                quick_mode=not args.debate and not args.scan,
                scan_mode=args.scan,
                use_worktrees=args.worktree or args.parallel,
                max_parallel=args.max_parallel,
                budget_limit_usd=args.budget_limit or 10.0,
                require_approval=args.require_approval,
                autonomous=args.autonomous,
                auto_mode=args.auto,
                safe_mode=safe_mode_enabled,
                risk_threshold=args.risk_threshold,
                progress_callback=_print_progress,
                run_tests=True,
                run_review=not args.no_gauntlet,
                capture_metrics=args.metrics,
                persist_outcomes=True,
                auto_revert_on_regression=True,
            )
            pipeline = SelfImprovePipeline(config)

            # objective may be None when --scan is used (self-directing mode)
            objective = args.goal

            async def _run_self_improve_with_bridge(dry_run: bool):
                """Run self-improve pipeline with stream bridge lifecycle."""
                await stream_bridge.start()
                try:
                    if dry_run:
                        return await pipeline.dry_run(objective)
                    return await pipeline.run(objective)
                finally:
                    await stream_bridge.stop()

            if args.dry_run:
                plan = asyncio.run(_run_self_improve_with_bridge(dry_run=True))
                print_header("SELF-IMPROVE DRY RUN")
                print(f"Objective: {plan['objective']}")
                print(f"\nGoals ({len(plan['goals'])}):")
                for i, g in enumerate(plan["goals"]):
                    print(f"  [{i + 1}] {g['description'][:80]}")
                    print(f"      Track: {g['track']}  Priority: {g['priority']}")
                print(f"\nSubtasks ({len(plan['subtasks'])}):")
                for i, s in enumerate(plan["subtasks"]):
                    title = s.get("title", s.get("description", "???"))
                    print(f"  [{i + 1}] {str(title)[:80]}")
                    if s.get("file_hints"):
                        print(f"      Files: {', '.join(s['file_hints'][:3])}")
                print(
                    f"\nConfig: worktrees={config.use_worktrees} parallel={config.max_parallel} budget=${config.budget_limit_usd}"
                )
                print(f"  safe_mode={config.safe_mode} risk_threshold={config.risk_threshold}")

                # Display risk assessments if available
                risk_assessments = plan.get("risk_assessments", [])
                if risk_assessments:
                    print(f"\nRisk Assessments ({len(risk_assessments)}):")
                    auto_count = sum(1 for r in risk_assessments if r["recommendation"] == "auto")
                    review_count = sum(
                        1 for r in risk_assessments if r["recommendation"] == "review"
                    )
                    block_count = sum(1 for r in risk_assessments if r["recommendation"] == "block")
                    print(
                        f"  Auto-approve: {auto_count}  Needs review: {review_count}  Blocked: {block_count}"
                    )
                    for i, ra in enumerate(risk_assessments):
                        category_label = ra["category"].upper()
                        rec_label = ra["recommendation"].upper()
                        goal_preview = ra.get("goal", "")[:60]
                        print(
                            f"  [{i + 1}] score={ra['score']:.2f} [{category_label}] -> {rec_label}"
                        )
                        if goal_preview:
                            print(f"      {goal_preview}")
                        for factor in ra.get("factors", [])[:2]:
                            print(f"      - {factor['name']}: {factor['detail'][:60]}")

                # Generate visual pipeline from dry-run goals
                if args.visual_pipeline and plan.get("goals"):
                    try:
                        from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

                        ideas = [g["description"][:200] for g in plan["goals"]]
                        pipe = IdeaToExecutionPipeline()
                        pipe_result = pipe.from_ideas(ideas, auto_advance=True)
                        print(f"\n  Visual Pipeline: /pipeline?id={pipe_result.pipeline_id}")
                    except (ImportError, RuntimeError, ValueError) as exc:
                        logger.debug("Visual pipeline generation failed: %s", exc)

                return 0

            result = asyncio.run(_run_self_improve_with_bridge(dry_run=False))
            print_header("SELF-IMPROVE RESULT")
            print(f"Cycle: {result.cycle_id}")
            print(f"Objective: {result.objective}")
            print(f"Goals planned: {result.goals_planned}")
            print(f"Subtasks: {result.subtasks_completed}/{result.subtasks_total} completed")
            print(f"Files changed: {len(result.files_changed)}")
            print(f"Tests: {result.tests_passed} passed, {result.tests_failed} failed")
            print(f"Regressions: {'YES' if result.regressions_detected else 'none'}")
            print(f"Duration: {result.duration_seconds:.1f}s")
            return 0 if result.subtasks_failed == 0 else 1

        except KeyboardInterrupt:
            print("\n\nSelf-improve cancelled by user.")
            return 130
        except (ImportError, RuntimeError, ValueError) as e:
            logger.exception("Self-improve pipeline failed")
            print(f"\nError: {e}")
            return 1

    # Pipeline mode: decompose then execute via DecisionPlan pipeline
    if args.use_pipeline:
        try:
            outcome = asyncio.run(
                run_pipeline_execution(
                    goal=args.goal,
                    use_debate=args.debate,
                    pipeline_mode=args.pipeline_mode,
                    budget_limit=args.budget_limit,
                    repo_path=resolved_repo,
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

    # Full run
    async def _run_orchestration_with_bridge():
        """Run orchestration with stream bridge lifecycle."""
        await stream_bridge.start()
        try:
            return await run_orchestration(
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
                repo_path=resolved_repo,
                enable_metrics=args.metrics,
                enable_preflight=args.preflight,
                enable_stuck_detection=args.stuck_detection,
                enable_watchdog=getattr(args, "watchdog", False),
                auto_execute_low_risk=args.auto,
            )
        finally:
            await stream_bridge.stop()

    try:
        result = asyncio.run(_run_orchestration_with_bridge())
        print_result(result)

        # Generate visual pipeline from orchestration goals if requested
        if args.visual_pipeline:
            try:
                from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

                # Convert orchestration results into ideas for pipeline
                ideas: list[str] = []
                if hasattr(result, "subtask_results"):
                    for sr in result.subtask_results:
                        title = getattr(sr, "title", getattr(sr, "subtask_id", "task"))
                        status = "completed" if getattr(sr, "success", False) else "failed"
                        ideas.append(f"[{status}] {title}")
                elif hasattr(result, "goals"):
                    for g in result.goals:
                        desc = getattr(g, "description", str(g))
                        ideas.append(desc)

                if ideas:
                    pipeline = IdeaToExecutionPipeline()
                    pipe_result = pipeline.from_ideas(ideas, auto_advance=True)
                    print(f"\n  Visual Pipeline: /pipeline?id={pipe_result.pipeline_id}")
                    stages = [s for s, v in pipe_result.stage_status.items() if v == "complete"]
                    print(f"  Stages: {' -> '.join(stages)}")
                else:
                    print("\n  Visual pipeline: no goals available to visualize")
            except (ImportError, RuntimeError, ValueError) as exc:
                logger.debug("Visual pipeline generation failed: %s", exc)

        # Run outcome feedback if requested
        if args.feedback and not args.no_feedback:
            try:
                from aragora.nomic.outcome_feedback import OutcomeFeedbackBridge

                feedback = OutcomeFeedbackBridge()
                cycle = feedback.run_feedback_cycle()
                goals_generated = cycle.get("goals_generated", 0)
                if goals_generated > 0:
                    print(f"\nOutcome feedback: {goals_generated} improvement goals queued")
                else:
                    print("\nOutcome feedback: no systematic errors detected")
            except ImportError:
                print("\nOutcome feedback: OutcomeFeedbackBridge not available")
            except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
                logger.debug("Outcome feedback failed: %s", exc)

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
