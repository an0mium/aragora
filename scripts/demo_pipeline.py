#!/usr/bin/env python3
"""End-to-end demo of the Aragora Idea-to-Execution Pipeline.

Takes a set of ideas and transforms them through four stages:
  1. Ideas -> Goals (via MetaPlanner or keyword clustering)
  2. Goals -> Tasks (via TaskDecomposer or heuristic decomposition)
  3. Tasks -> Workflow (via WorkflowEngine or DAG generation)
  4. Workflow -> Execution (optional, requires running server)

Usage:
    # From command line arguments
    python scripts/demo_pipeline.py "Build rate limiter" "Add caching" "Improve API docs"

    # From a file (one idea per line)
    python scripts/demo_pipeline.py --file ideas.txt

    # JSON output
    python scripts/demo_pipeline.py --json "Build rate limiter" "Add caching"

    # Dry run (no execution)
    python scripts/demo_pipeline.py --dry-run "Build rate limiter" "Add caching"

    # With built-in example ideas
    python scripts/demo_pipeline.py --demo
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import os
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in demo ideas
# ---------------------------------------------------------------------------

DEMO_IDEAS = [
    "Build a rate limiter for API endpoints",
    "Add Redis caching layer for frequently accessed data",
    "Improve API documentation with OpenAPI examples",
    "Set up health check monitoring with Prometheus",
    "Add request validation middleware",
    "Implement audit logging for sensitive operations",
    "Create onboarding wizard for new users",
    "Add dark mode support to dashboard",
]


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_stage(title: str, items: list[str]) -> None:
    """Print a stage header with its input items."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    for item in items[:10]:
        print(f"  * {item}")
    if len(items) > 10:
        print(f"  ... and {len(items) - 10} more")


def print_result(result: Any, json_output: bool) -> None:
    """Print a TransitionResult in human-readable or JSON form."""
    if json_output:
        print(json.dumps(
            {
                "nodes": [
                    {
                        "id": n.id,
                        "stage": n.stage,
                        "label": n.label,
                        "hash": n.hash,
                    }
                    for n in result.nodes
                ],
                "edges": [
                    {
                        "source": e.source,
                        "target": e.target,
                        "type": e.edge_type,
                    }
                    for e in result.edges
                ],
                "provenance": result.provenance,
            },
            indent=2,
        ))
    else:
        print(f"\n  Generated {len(result.nodes)} nodes, {len(result.edges)} edges")
        method = result.provenance.get("method", "unknown")
        print(f"  Method: {method}")
        for node in result.nodes:
            hash_prefix = node.hash[:8] if node.hash else "n/a"
            print(f"    [{node.stage}] {node.label} (#{hash_prefix})")


def print_provenance(
    ideas: list[str],
    goals_result: Any,
    tasks_result: Any,
    workflow_result: Any,
) -> None:
    """Print the full provenance chain summary."""
    goal_nodes = [n for n in goals_result.nodes if n.stage == "goal"]
    task_nodes = tasks_result.nodes
    orch_nodes = workflow_result.nodes

    print(f"\n{'=' * 60}")
    print("  Provenance Chain")
    print(f"{'=' * 60}")

    total_nodes = len(ideas) + len(goal_nodes) + len(task_nodes) + len(orch_nodes)
    total_edges = (
        len(goals_result.edges)
        + len(tasks_result.edges)
        + len(workflow_result.edges)
    )
    print(
        f"  {len(ideas)} ideas -> {len(goal_nodes)} goals "
        f"-> {len(task_nodes)} tasks -> {len(orch_nodes)} orchestration nodes"
    )
    print(f"  Total: {total_nodes} nodes, {total_edges} edges")
    print(
        f"  Methods: {goals_result.provenance.get('method', '?')} "
        f"-> {tasks_result.provenance.get('method', '?')} "
        f"-> {workflow_result.provenance.get('method', '?')}"
    )

    # Walk one sample chain from orchestration back to ideas
    if orch_nodes:
        orch_node = orch_nodes[0]
        print(f"\n  Sample chain for '{orch_node.label}':")
        hash_prefix = orch_node.hash[:8] if orch_node.hash else "n/a"
        print(f"    Orchestration: {orch_node.label} (#{hash_prefix})")

        for task_node in task_nodes:
            if task_node.id in orch_node.derived_from:
                task_hash = task_node.hash[:8] if task_node.hash else "n/a"
                print(f"      <- Task: {task_node.label} (#{task_hash})")
                for goal_node in goal_nodes:
                    if goal_node.id in task_node.derived_from:
                        goal_hash = goal_node.hash[:8] if goal_node.hash else "n/a"
                        print(f"          <- Goal: {goal_node.label} (#{goal_hash})")
                break


def format_full_json(
    ideas: list[str],
    goals_result: Any,
    tasks_result: Any,
    workflow_result: Any,
    dry_run: bool,
) -> dict[str, Any]:
    """Build a complete JSON summary of the pipeline run."""

    def serialize_nodes(result: Any) -> list[dict[str, Any]]:
        return [
            {"id": n.id, "stage": n.stage, "label": n.label, "hash": n.hash}
            for n in result.nodes
        ]

    def serialize_edges(result: Any) -> list[dict[str, Any]]:
        return [
            {"source": e.source, "target": e.target, "type": e.edge_type}
            for e in result.edges
        ]

    goal_nodes = [n for n in goals_result.nodes if n.stage == "goal"]
    return {
        "input_ideas": ideas,
        "stages": {
            "ideas_to_goals": {
                "nodes": serialize_nodes(goals_result),
                "edges": serialize_edges(goals_result),
                "provenance": goals_result.provenance,
            },
            "goals_to_tasks": {
                "nodes": serialize_nodes(tasks_result),
                "edges": serialize_edges(tasks_result),
                "provenance": tasks_result.provenance,
            },
            "tasks_to_workflow": {
                "nodes": serialize_nodes(workflow_result),
                "edges": serialize_edges(workflow_result),
                "provenance": workflow_result.provenance,
            },
        },
        "summary": {
            "ideas": len(ideas),
            "goals": len(goal_nodes),
            "tasks": len(tasks_result.nodes),
            "orchestration_nodes": len(workflow_result.nodes),
            "total_edges": (
                len(goals_result.edges)
                + len(tasks_result.edges)
                + len(workflow_result.edges)
            ),
            "methods": [
                goals_result.provenance.get("method", "unknown"),
                tasks_result.provenance.get("method", "unknown"),
                workflow_result.provenance.get("method", "unknown"),
            ],
        },
        "dry_run": dry_run,
    }


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def run_pipeline(
    ideas: list[str],
    *,
    dry_run: bool = False,
    json_output: bool = False,
) -> int:
    """Run the full idea-to-execution pipeline.

    Returns 0 on success, 1 on error.
    """
    try:
        from aragora.server.handlers.pipeline.transitions import (
            _ideas_to_goals_logic,
            _goals_to_tasks_logic,
            _tasks_to_workflow_logic,
            PipelineNode,  # noqa: F401 — verify import
            get_node_store,
        )
    except ImportError as exc:
        print(f"Error: Could not import pipeline transition logic: {exc}")
        print("Make sure you are running from the aragora project root.")
        return 1

    # Clear the in-memory node store so repeated runs start fresh
    get_node_store().clear()

    # -- Stage 1: Ideas -> Goals ----------------------------------------
    if not json_output:
        print_stage("Stage 1: Ideas -> Goals", ideas)

    idea_dicts = [{"id": f"idea-{i}", "label": idea} for i, idea in enumerate(ideas)]
    goals_result = _ideas_to_goals_logic(idea_dicts)

    if not json_output:
        print_result(goals_result, json_output=False)

    # -- Stage 2: Goals -> Tasks ----------------------------------------
    goal_nodes = [n for n in goals_result.nodes if n.stage == "goal"]
    if not json_output:
        print_stage("Stage 2: Goals -> Tasks", [n.label for n in goal_nodes])

    goal_dicts = [
        {"id": n.id, "label": n.label, "metadata": n.metadata}
        for n in goal_nodes
    ]
    tasks_result = _goals_to_tasks_logic(goal_dicts)

    if not json_output:
        print_result(tasks_result, json_output=False)

    # -- Stage 3: Tasks -> Workflow -------------------------------------
    if not json_output:
        print_stage("Stage 3: Tasks -> Workflow", [n.label for n in tasks_result.nodes])

    task_dicts = [
        {"id": n.id, "label": n.label, "metadata": n.metadata}
        for n in tasks_result.nodes
    ]
    workflow_result = _tasks_to_workflow_logic(task_dicts)

    if not json_output:
        print_result(workflow_result, json_output=False)

    # -- Stage 4: Execution (or dry run) --------------------------------
    if not json_output:
        if dry_run:
            print_stage("Stage 4: Execution (DRY RUN)", [])
            print("  Skipping execution -- use without --dry-run to execute")
        else:
            print_stage("Stage 4: Execution", [])
            print("  Execution requires a running Aragora server")
            print("  Start with: aragora serve --api-port 8080")

    # -- Output ---------------------------------------------------------
    if json_output:
        full = format_full_json(ideas, goals_result, tasks_result, workflow_result, dry_run)
        print(json.dumps(full, indent=2))
    else:
        print_provenance(ideas, goals_result, tasks_result, workflow_result)

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (exposed for testing)."""
    parser = argparse.ArgumentParser(
        description="Aragora Idea-to-Execution Pipeline Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Built-in demo ideas
  %(prog)s --demo

  # Dry run with custom ideas
  %(prog)s --dry-run "Build rate limiter" "Add caching"

  # JSON output
  %(prog)s --json "Build rate limiter" "Add caching"

  # From a file (one idea per line)
  %(prog)s --file ideas.txt
        """,
    )
    parser.add_argument("ideas", nargs="*", help="Ideas to process")
    parser.add_argument(
        "--file", "-f",
        help="Read ideas from file (one per line)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use built-in demo ideas",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip execution stage",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Resolve ideas
    if args.demo:
        ideas = DEMO_IDEAS
    elif args.file:
        try:
            with open(args.file) as f:
                ideas = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}")
            return 1
    elif args.ideas:
        ideas = args.ideas
    else:
        parser.print_help()
        return 1

    if not ideas:
        print("Error: No ideas provided.")
        return 1

    print("Aragora Idea-to-Execution Pipeline")
    print(f"Processing {len(ideas)} ideas...")

    return run_pipeline(ideas, dry_run=args.dry_run, json_output=args.json)


if __name__ == "__main__":
    sys.exit(main())
