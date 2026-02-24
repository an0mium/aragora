"""
Workflow Automation Example
===========================

Defines a 3-step workflow using Aragora's WorkflowEngine:
  1. Gather -- collect context and requirements
  2. Debate -- multi-agent analysis of the gathered data
  3. Report -- generate a final summary report

Demonstrates:
  - WorkflowDefinition with StepDefinition and TransitionRule
  - WorkflowEngine execution with event callbacks
  - Step chaining via next_steps and conditional transitions

Requirements:
    - pip install aragora
    - ANTHROPIC_API_KEY or OPENAI_API_KEY set in environment

Usage:
    python examples/workflow_automation/main.py --topic "API rate limiting strategy"
    python examples/workflow_automation/main.py --topic "Database migration plan" --rounds 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

# --- API key check -------------------------------------------------------

def _check_api_keys() -> None:
    """Exit early with a helpful message if no API keys are configured."""
    keys = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
    if not any(os.environ.get(k) for k in keys):
        print(
            "ERROR: No API key found. Set at least one of:\n"
            "  ANTHROPIC_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY\n"
            "See the README for setup instructions.",
            file=sys.stderr,
        )
        sys.exit(1)


# --- Workflow definition --------------------------------------------------

def build_workflow(topic: str, rounds: int = 3):
    """Build a 3-step gather-debate-report workflow definition.

    Returns a WorkflowDefinition that the engine can execute.
    """
    from aragora.workflow.types import (
        StepDefinition,
        TransitionRule,
        WorkflowDefinition,
    )

    # Step 1: Gather context and requirements
    gather_step = StepDefinition(
        id="gather",
        name="Gather Context",
        step_type="agent",
        description="Collect background information and requirements for the topic.",
        config={
            "task": (
                f"Research and gather key context for the following topic: {topic}\n\n"
                "Provide:\n"
                "- Background information\n"
                "- Key constraints and requirements\n"
                "- Relevant prior decisions or precedents\n"
                "- Stakeholder concerns"
            ),
            "agent_type": "default",
        },
        timeout_seconds=120.0,
        next_steps=["debate"],
    )

    # Step 2: Multi-agent debate on the gathered context
    debate_step = StepDefinition(
        id="debate",
        name="Multi-Agent Debate",
        step_type="debate",
        description="Agents debate the best approach based on gathered context.",
        config={
            "task": (
                f"Based on the gathered context, debate the best approach for: {topic}\n\n"
                "Consider trade-offs, risks, and implementation feasibility. "
                "Arrive at a concrete recommendation."
            ),
            "rounds": rounds,
            "consensus": "majority",
        },
        timeout_seconds=300.0,
        next_steps=["report"],
    )

    # Step 3: Generate final report
    report_step = StepDefinition(
        id="report",
        name="Generate Report",
        step_type="agent",
        description="Compile the debate outcome into a structured report.",
        config={
            "task": (
                "Based on the debate results, generate a structured report with:\n"
                "1. Executive Summary\n"
                "2. Recommended Approach\n"
                "3. Key Trade-offs Considered\n"
                "4. Implementation Steps\n"
                "5. Risk Mitigation Plan"
            ),
            "agent_type": "default",
        },
        timeout_seconds=120.0,
    )

    # Conditional transition: skip report if debate failed to reach consensus
    consensus_transition = TransitionRule(
        id="tr_debate_report",
        from_step="debate",
        to_step="report",
        condition="True",  # Always transition; could add consensus check
        label="Debate complete",
    )

    # Assemble the workflow definition
    workflow = WorkflowDefinition(
        id="gather_debate_report",
        name=f"Analysis: {topic[:50]}",
        description=(
            "Three-step workflow that gathers context, runs a multi-agent "
            "debate, and produces a final report."
        ),
        steps=[gather_step, debate_step, report_step],
        transitions=[consensus_transition],
        entry_step="gather",
        metadata={"topic": topic, "rounds": rounds},
    )

    return workflow


# --- Event callback -------------------------------------------------------

def on_workflow_event(event_type: str, payload: dict) -> None:
    """Log workflow events to stderr for visibility."""
    step_name = payload.get("step_name", "")
    status = payload.get("status", "")
    duration = payload.get("duration_ms", 0)

    if "step_start" in event_type:
        print(f"  [START] {step_name}", file=sys.stderr)
    elif "step_complete" in event_type:
        print(f"  [DONE]  {step_name} ({duration:.0f}ms) - {status}", file=sys.stderr)
    elif "step_failed" in event_type:
        error = payload.get("error", "unknown")
        print(f"  [FAIL]  {step_name} - {error}", file=sys.stderr)


# --- Execution ------------------------------------------------------------

async def run_workflow(topic: str, rounds: int = 3) -> dict:
    """Build and execute the gather-debate-report workflow."""
    from aragora.workflow.engine import WorkflowEngine
    from aragora.workflow.types import WorkflowConfig

    # Build the workflow definition
    workflow_def = build_workflow(topic, rounds)

    # Validate the definition before execution
    is_valid, errors = workflow_def.validate()
    if not is_valid:
        print(f"ERROR: Invalid workflow: {errors}", file=sys.stderr)
        sys.exit(1)

    # Create the engine with a reasonable timeout
    config = WorkflowConfig(
        total_timeout_seconds=600.0,  # 10 minutes max
        stop_on_failure=True,
        enable_checkpointing=False,  # Keep it simple for the example
    )
    engine = WorkflowEngine(config=config)

    print(f"Executing workflow: {workflow_def.name}", file=sys.stderr)
    print(f"Steps: {' -> '.join(s.name for s in workflow_def.steps)}", file=sys.stderr)

    # Execute the workflow
    result = await engine.execute(
        definition=workflow_def,
        inputs={"topic": topic},
        event_callback=on_workflow_event,
    )

    # Format the output
    output = {
        "workflow_id": result.workflow_id,
        "topic": topic,
        "success": result.success,
        "total_duration_ms": result.total_duration_ms,
        "steps": [
            {
                "name": step.step_name,
                "status": step.status.value,
                "duration_ms": step.duration_ms,
                "output_preview": (
                    str(step.output)[:300] if step.output else None
                ),
            }
            for step in result.steps
        ],
        "final_output": str(result.final_output)[:2000] if result.final_output else None,
        "error": result.error,
    }

    return output


# --- CLI entry point ------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3-step workflow automation: gather, debate, report"
    )
    parser.add_argument(
        "--topic",
        required=True,
        help="The topic or question to analyze",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of debate rounds (default: 3)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full results as JSON",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    _check_api_keys()

    result = await run_workflow(args.topic, args.rounds)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = "SUCCESS" if result["success"] else "FAILED"
        duration = result["total_duration_ms"] / 1000
        print(f"\n{'='*60}")
        print(f"  Workflow: {status} ({duration:.1f}s)")
        print(f"  Topic: {args.topic}")
        print(f"{'='*60}")

        for step in result["steps"]:
            icon = "[OK]" if step["status"] == "completed" else "[!!]"
            print(f"  {icon} {step['name']}: {step['status']} ({step['duration_ms']:.0f}ms)")

        if result["final_output"]:
            print(f"\n--- Final Report ---\n{result['final_output']}")

        if result["error"]:
            print(f"\nError: {result['error']}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
