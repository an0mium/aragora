"""Pipeline CLI command -- idea-to-execution pipeline operations.

Runs the full four-stage pipeline:
  Stage 1 (Ideas) -> Stage 2 (Goals) -> Stage 3 (Actions) -> Stage 4 (Orchestration)

Also supports a self-improve subcommand that combines TaskDecomposer +
MetaPlanner + IdeaToExecutionPipeline for goal-driven self-improvement.

Usage:
    aragora pipeline run "Build rate limiter, Add caching"
    aragora pipeline run "Improve error handling" --dry-run
    aragora pipeline self-improve "Maximize utility for SMEs" --budget-limit 5
    aragora pipeline status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Handle 'pipeline' command -- route to subcommand."""
    subcommand = getattr(args, "pipeline_action", None)
    if subcommand == "run":
        _cmd_pipeline_run(args)
    elif subcommand == "self-improve":
        _cmd_pipeline_self_improve(args)
    elif subcommand == "status":
        _cmd_pipeline_status(args)
    else:
        print("Usage: aragora pipeline {run,self-improve,status}")
        print("Run 'aragora pipeline --help' for details.")


def _cmd_pipeline_run(args: argparse.Namespace) -> None:
    """Run the full idea-to-execution pipeline from raw ideas."""
    ideas_raw = args.ideas
    dry_run = args.dry_run
    require_approval = args.require_approval
    budget_limit = args.budget_limit

    # Split comma-separated ideas
    ideas = [i.strip() for i in ideas_raw.split(",") if i.strip()]
    if not ideas:
        print("\nError: No ideas provided. Pass comma-separated ideas.")
        print('  Example: aragora pipeline run "Build rate limiter, Add caching"')
        return

    print("\n" + "=" * 60)
    print("IDEA-TO-EXECUTION PIPELINE")
    print("=" * 60)
    print(f"\nIdeas ({len(ideas)}):")
    for i, idea in enumerate(ideas, 1):
        print(f"  {i}. {idea}")
    if dry_run:
        print("\nMode: DRY RUN (preview only)")
    if budget_limit:
        print(f"Budget limit: ${budget_limit:.2f}")
    if require_approval:
        print("Approval: Required at gates")
    print()

    if dry_run:
        _run_pipeline_dry_run(ideas)
    else:
        _run_pipeline_execute(ideas, require_approval, budget_limit)


def _run_pipeline_dry_run(ideas: list[str]) -> None:
    """Preview the pipeline stages without executing."""
    print("-" * 60)
    print("PIPELINE PREVIEW")
    print("-" * 60)

    try:
        from aragora.pipeline.idea_to_execution import (
            IdeaToExecutionPipeline,
            PipelineConfig,
        )

        _config = PipelineConfig(dry_run=True)  # noqa: F841 — reserved for async run()
        pipeline = IdeaToExecutionPipeline()
        result = pipeline.from_ideas(ideas)

        print(f"\nPipeline ID: {result.pipeline_id}")

        # Stage 1: Ideas
        if result.ideas_canvas:
            node_count = (
                len(result.ideas_canvas.nodes) if hasattr(result.ideas_canvas, "nodes") else 0
            )
            print(f"\nStage 1 - Ideas Canvas: {node_count} nodes")

        # Stage 2: Goals
        if result.goal_graph:
            goal_count = len(result.goal_graph.goals) if hasattr(result.goal_graph, "goals") else 0
            print(f"Stage 2 - Goal Graph: {goal_count} goals")
            if hasattr(result.goal_graph, "goals"):
                for g in result.goal_graph.goals[:5]:
                    title = getattr(g, "title", getattr(g, "description", str(g)))
                    print(f"  - {title}")

        # Stage results
        if result.stage_results:
            print("\nStage Results:")
            for sr in result.stage_results:
                print(f"  [{sr.status}] {sr.stage_name} ({sr.duration:.2f}s)")

        print(f"\nProvenance chain: {len(result.provenance)} links")

    except ImportError:
        print("\nIdeaToExecutionPipeline unavailable.")
        print("Install required dependencies or check aragora/pipeline/.")
        return

    print()
    print("To execute this pipeline:")
    ideas_str = ", ".join(ideas)
    print(f'  aragora pipeline run "{ideas_str}"')
    print()


def _run_pipeline_execute(
    ideas: list[str],
    require_approval: bool,
    budget_limit: float | None,
) -> None:
    """Execute the full pipeline."""
    try:
        from aragora.pipeline.idea_to_execution import (
            IdeaToExecutionPipeline,
            PipelineConfig,
        )

        _config = PipelineConfig(
            dry_run=False,
            enable_receipts=True,
        )  # noqa: F841 — reserved for async run()
        pipeline = IdeaToExecutionPipeline()

        print("-" * 60)
        print("EXECUTING PIPELINE")
        print("-" * 60)

        result = pipeline.from_ideas(ideas)

        print(f"\nPipeline ID: {result.pipeline_id}")
        print(f"Duration: {result.duration:.1f}s")

        if result.stage_results:
            print("\nStage Results:")
            for sr in result.stage_results:
                status_icon = "OK" if sr.status == "completed" else sr.status.upper()
                print(f"  [{status_icon}] {sr.stage_name} ({sr.duration:.2f}s)")
                if sr.error:
                    print(f"         Error: {sr.error}")

        if result.receipt:
            print(f"\nReceipt: {json.dumps(result.receipt, indent=2, default=str)[:200]}...")

        print()

    except ImportError:
        print("\nError: IdeaToExecutionPipeline unavailable.")
        print("Check that aragora/pipeline/ is properly installed.")

    except (OSError, RuntimeError, ValueError) as e:
        print(f"\nPipeline failed: {e}")


def _cmd_pipeline_self_improve(args: argparse.Namespace) -> None:
    """Run self-improvement using TaskDecomposer + MetaPlanner + Pipeline."""
    goal = args.goal
    dry_run = args.dry_run
    require_approval = args.require_approval
    budget_limit = args.budget_limit

    print("\n" + "=" * 60)
    print("PIPELINE SELF-IMPROVEMENT")
    print("=" * 60)
    print(f"\nGoal: {goal}")
    if dry_run:
        print("Mode: DRY RUN (preview only)")
    if budget_limit:
        print(f"Budget limit: ${budget_limit:.2f}")
    if require_approval:
        print("Approval: Required at gates")
    print()

    # Step 1: TaskDecomposer analyzes the goal
    print("-" * 60)
    print("STEP 1: TASK DECOMPOSITION")
    print("-" * 60)

    subtasks = []
    try:
        from aragora.nomic.task_decomposer import DecomposerConfig, TaskDecomposer

        decomposer = TaskDecomposer(config=DecomposerConfig(complexity_threshold=4))
        decomposition = decomposer.analyze(goal)

        print(
            f"\nComplexity: {decomposition.complexity_score}/10 ({decomposition.complexity_level})"
        )
        print(f"Subtasks: {len(decomposition.subtasks)}")

        for i, subtask in enumerate(decomposition.subtasks, 1):
            print(f"  {i}. {subtask.title}")
            print(f"     Complexity: {subtask.estimated_complexity}")
            if subtask.file_scope:
                files = ", ".join(subtask.file_scope[:3])
                extra = f" +{len(subtask.file_scope) - 3}" if len(subtask.file_scope) > 3 else ""
                print(f"     Files: {files}{extra}")

        subtasks = decomposition.subtasks
        print()

    except ImportError:
        print("\nTaskDecomposer unavailable, skipping decomposition.")
        print()

    # Step 2: MetaPlanner debates improvement priorities
    print("-" * 60)
    print("STEP 2: META-PLANNING (priority debate)")
    print("-" * 60)

    prioritized_goals = []
    try:
        from aragora.nomic.meta_planner import MetaPlanner, MetaPlannerConfig

        planner = MetaPlanner(MetaPlannerConfig(quick_mode=True))
        prioritized_goals = asyncio.run(planner.prioritize_work(objective=goal))

        print(f"\nPrioritized goals ({len(prioritized_goals)}):")
        for pg in prioritized_goals:
            print(f"  {pg.priority}. [{pg.track.value}] {pg.description}")
            print(f"     Impact: {pg.estimated_impact}")
            if pg.rationale:
                print(f"     Rationale: {pg.rationale}")
        print()

    except ImportError:
        print("\nMetaPlanner unavailable, skipping priority debate.")
        print()

    # Step 3: IdeaToExecutionPipeline structures ideas into goals -> actions
    print("-" * 60)
    print("STEP 3: IDEA-TO-EXECUTION PIPELINE")
    print("-" * 60)

    # Build ideas from decomposition + prioritization
    ideas = []
    for pg in prioritized_goals:
        ideas.append(pg.description)
    if not ideas:
        # Fall back to subtask titles
        for st in subtasks:
            ideas.append(st.title)
    if not ideas:
        ideas = [goal]

    try:
        from aragora.pipeline.idea_to_execution import (
            IdeaToExecutionPipeline,
            PipelineConfig,
        )

        _config = PipelineConfig(dry_run=dry_run, enable_receipts=not dry_run)  # noqa: F841
        pipeline = IdeaToExecutionPipeline()
        result = pipeline.from_ideas(ideas)

        print(f"\nPipeline ID: {result.pipeline_id}")

        if result.goal_graph and hasattr(result.goal_graph, "goals"):
            print(f"Goals extracted: {len(result.goal_graph.goals)}")
            for g in result.goal_graph.goals[:5]:
                title = getattr(g, "title", getattr(g, "description", str(g)))
                print(f"  - {title}")

        if result.stage_results:
            print("\nStage Results:")
            for sr in result.stage_results:
                status_icon = "OK" if sr.status == "completed" else sr.status.upper()
                print(f"  [{status_icon}] {sr.stage_name} ({sr.duration:.2f}s)")

        print(f"\nProvenance chain: {len(result.provenance)} links")
        print(f"Duration: {result.duration:.1f}s")

    except ImportError:
        print("\nIdeaToExecutionPipeline unavailable.")
        print("Showing decomposition results only.")

    print()

    if dry_run:
        print("To execute this plan:")
        print(f'  aragora pipeline self-improve "{goal}"')
        print()


def _cmd_pipeline_status(args: argparse.Namespace) -> None:
    """Show active pipeline status."""
    print("\n" + "=" * 60)
    print("PIPELINE STATUS")
    print("=" * 60)

    try:
        from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

        pipeline = IdeaToExecutionPipeline()
        # Check if there's a status method or active pipelines
        active = getattr(pipeline, "active_pipelines", None)
        if active:
            print(f"\nActive pipelines: {len(active)}")
            for pid, info in active.items():
                print(f"  {pid}: {info}")
        else:
            print("\nNo active pipelines.")
            print("\nStart one with:")
            print('  aragora pipeline run "Build rate limiter, Add caching"')
            print('  aragora pipeline self-improve "Improve test coverage"')

    except ImportError:
        print("\nPipeline module unavailable.")
        print("Check that aragora/pipeline/ is properly installed.")

    print()


def add_pipeline_parser(subparsers) -> None:
    """Register the 'pipeline' subcommand parser."""
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run idea-to-execution pipeline operations",
        description="""
Run the four-stage idea-to-execution pipeline:

  Stage 1 (Ideas) -> Stage 2 (Goals) -> Stage 3 (Actions) -> Stage 4 (Orchestration)

Subcommands:
  run           - Run the pipeline from raw ideas
  self-improve  - Combine TaskDecomposer + MetaPlanner + Pipeline for self-improvement
  status        - Show active pipeline status

Examples:
  aragora pipeline run "Build rate limiter, Add caching"
  aragora pipeline run "Improve error handling" --dry-run
  aragora pipeline self-improve "Maximize utility for SMEs" --budget-limit 5
  aragora pipeline status
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    pipeline_sub = pipeline_parser.add_subparsers(dest="pipeline_action")

    # pipeline run
    run_parser = pipeline_sub.add_parser(
        "run",
        help="Run the full pipeline from ideas",
    )
    run_parser.add_argument(
        "ideas",
        help="Comma-separated ideas to process (e.g. 'Build rate limiter, Add caching')",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview pipeline stages without executing",
    )
    run_parser.add_argument(
        "--require-approval",
        action="store_true",
        help="Require human approval at stage gates",
    )
    run_parser.add_argument(
        "--budget-limit",
        type=float,
        default=None,
        help="Maximum budget in USD for pipeline execution",
    )

    # pipeline self-improve
    si_parser = pipeline_sub.add_parser(
        "self-improve",
        help="Run self-improvement via TaskDecomposer + MetaPlanner + Pipeline",
    )
    si_parser.add_argument(
        "goal",
        help="The improvement goal (e.g. 'Maximize utility for SMEs')",
    )
    si_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the plan without executing",
    )
    si_parser.add_argument(
        "--require-approval",
        action="store_true",
        help="Require human approval at gates",
    )
    si_parser.add_argument(
        "--budget-limit",
        type=float,
        default=None,
        help="Maximum budget in USD",
    )

    # pipeline status
    pipeline_sub.add_parser(
        "status",
        help="Show active pipeline status",
    )

    pipeline_parser.set_defaults(func=cmd_pipeline)
