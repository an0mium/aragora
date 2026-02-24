#!/usr/bin/env python3
"""
Workflow Automation Example
===========================

Uses Aragora's WorkflowEngine to automate a business process with
DAG-based task orchestration. Demonstrates sequential, parallel, and
conditional step execution with checkpointing and error handling.

This example implements an automated **content publishing pipeline**:

    1. Draft content (agent step)
    2. Run parallel reviews (security review + editorial review)
    3. Conditional: if reviews pass, publish; otherwise send back for revision
    4. Notify stakeholders

The workflow engine supports:
    - Sequential and parallel step execution
    - Conditional branching based on step outputs
    - Checkpointing for long-running workflows
    - Configurable timeouts and retries per step

Requirements:
    - ANTHROPIC_API_KEY or OPENAI_API_KEY (at least one, for live mode)

Usage:
    # Run in demo mode (no API keys needed)
    python examples/workflow-automation/main.py --demo

    # Run with live agents
    python examples/workflow-automation/main.py --topic "AI governance best practices"

    # Custom workflow with more steps
    python examples/workflow-automation/main.py --topic "API security guide" --rounds 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure aragora is importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from aragora.workflow import (
    WorkflowEngine,
    WorkflowDefinition,
    WorkflowConfig,
    StepDefinition,
    TransitionRule,
    WorkflowContext,
    WorkflowResult,
)
from aragora.workflow.types import ExecutionPattern
from aragora.workflow.step import BaseStep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("workflow-automation")


# ---------------------------------------------------------------------------
# Custom Step Implementations
# ---------------------------------------------------------------------------


class DraftContentStep(BaseStep):
    """Step that drafts content on a given topic.

    In live mode, this would call an Aragora agent to generate content.
    In demo mode, it produces a sample draft.
    """

    @property
    def name(self) -> str:
        return "draft_content"

    async def execute(self, context: WorkflowContext) -> Any:
        topic = context.get_input("topic", "AI governance")
        demo_mode = context.get_input("demo", True)

        logger.info("Drafting content on: %s", topic)

        if demo_mode:
            draft = {
                "title": f"Guide: {topic}",
                "content": (
                    f"# {topic}\n\n"
                    "## Introduction\n"
                    f"This guide covers best practices for {topic.lower()}, "
                    "drawing on industry standards and expert consensus.\n\n"
                    "## Key Principles\n"
                    "1. **Transparency** -- All AI decisions should be explainable\n"
                    "2. **Accountability** -- Clear ownership for AI system outcomes\n"
                    "3. **Fairness** -- Regular bias audits across protected classes\n"
                    "4. **Security** -- Defense in depth for AI model endpoints\n\n"
                    "## Implementation Checklist\n"
                    "- [ ] Document model training data provenance\n"
                    "- [ ] Implement output monitoring and alerting\n"
                    "- [ ] Establish human-in-the-loop escalation paths\n"
                    "- [ ] Schedule quarterly bias and drift audits\n\n"
                    "## Conclusion\n"
                    f"Effective {topic.lower()} requires ongoing commitment "
                    "from engineering, legal, and leadership teams."
                ),
                "word_count": 150,
                "status": "drafted",
            }
        else:
            # In live mode, use Aragora agent
            try:
                from aragora.agents.base import create_agent

                create_agent(
                    model_type="anthropic-api",
                    name="content_drafter",
                    role="content_writer",
                )
                # Simple agent call for draft generation
                draft = {
                    "title": f"Guide: {topic}",
                    "content": f"Draft content about {topic} (agent would generate this)",
                    "word_count": 0,
                    "status": "drafted",
                }
            except Exception as exc:
                logger.warning("Agent unavailable, using demo content: %s", exc)
                draft = {
                    "title": f"Guide: {topic}",
                    "content": f"Draft about {topic}",
                    "word_count": 50,
                    "status": "drafted",
                }

        context.emit_event("content_drafted", {"title": draft["title"]})
        return draft


class SecurityReviewStep(BaseStep):
    """Step that reviews content for security issues.

    Checks for sensitive data exposure, insecure recommendations,
    and compliance issues.
    """

    @property
    def name(self) -> str:
        return "security_review"

    async def execute(self, context: WorkflowContext) -> Any:
        draft = context.get_step_output("draft", {})
        content = draft.get("content", "")

        logger.info("Running security review on: %s", draft.get("title", ""))

        # Simulate review delay
        await asyncio.sleep(0.1)

        # Simple keyword-based security check (demo)
        issues = []
        lower_content = content.lower()

        sensitive_patterns = [
            ("password", "Potential credential exposure"),
            ("api_key", "API key reference detected"),
            ("secret", "Secret reference detected"),
            ("token", "Token reference -- ensure not a literal value"),
        ]

        for pattern, description in sensitive_patterns:
            if pattern in lower_content:
                issues.append(
                    {
                        "type": "security",
                        "severity": "medium",
                        "description": description,
                        "pattern": pattern,
                    }
                )

        passed = len([i for i in issues if i["severity"] in ("critical", "high")]) == 0

        result = {
            "passed": passed,
            "issues_found": len(issues),
            "issues": issues,
            "reviewer": "security_review_agent",
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }

        context.emit_event(
            "security_review_complete",
            {
                "passed": passed,
                "issues": len(issues),
            },
        )

        return result


class EditorialReviewStep(BaseStep):
    """Step that reviews content for editorial quality.

    Checks structure, readability, and completeness.
    """

    @property
    def name(self) -> str:
        return "editorial_review"

    async def execute(self, context: WorkflowContext) -> Any:
        draft = context.get_step_output("draft", {})
        content = draft.get("content", "")

        logger.info("Running editorial review on: %s", draft.get("title", ""))

        # Simulate review delay
        await asyncio.sleep(0.1)

        # Simple quality checks (demo)
        checks = {
            "has_title": content.startswith("#"),
            "has_sections": "##" in content,
            "has_introduction": "introduction" in content.lower() or "intro" in content.lower(),
            "has_conclusion": "conclusion" in content.lower() or "summary" in content.lower(),
            "min_length": len(content) > 200,
            "has_actionable_items": "- [" in content or "1." in content,
        }

        passed_checks = sum(1 for v in checks.values() if v)
        total_checks = len(checks)
        score = passed_checks / total_checks if total_checks > 0 else 0

        suggestions = []
        if not checks["has_introduction"]:
            suggestions.append("Add an introduction section")
        if not checks["has_conclusion"]:
            suggestions.append("Add a conclusion or summary")
        if not checks["min_length"]:
            suggestions.append("Content is too short; expand key sections")
        if not checks["has_actionable_items"]:
            suggestions.append("Add actionable items or a checklist")

        result = {
            "passed": score >= 0.7,
            "score": round(score, 2),
            "checks": checks,
            "suggestions": suggestions,
            "reviewer": "editorial_review_agent",
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }

        context.emit_event(
            "editorial_review_complete",
            {
                "passed": result["passed"],
                "score": score,
            },
        )

        return result


class PublishStep(BaseStep):
    """Step that publishes approved content.

    In production, this would push to a CMS, documentation site,
    or internal knowledge base.
    """

    @property
    def name(self) -> str:
        return "publish"

    async def execute(self, context: WorkflowContext) -> Any:
        draft = context.get_step_output("draft", {})
        security = context.get_step_output("security_review", {})
        editorial = context.get_step_output("editorial_review", {})

        title = draft.get("title", "Untitled")

        logger.info("Publishing: %s", title)

        result = {
            "published": True,
            "title": title,
            "url": f"https://docs.example.com/guides/{title.lower().replace(' ', '-').replace(':', '')}",
            "security_cleared": security.get("passed", False),
            "editorial_score": editorial.get("score", 0),
            "published_at": datetime.now(timezone.utc).isoformat(),
        }

        context.emit_event("content_published", {"url": result["url"]})
        return result


class RevisionStep(BaseStep):
    """Step that sends content back for revision when reviews fail."""

    @property
    def name(self) -> str:
        return "request_revision"

    async def execute(self, context: WorkflowContext) -> Any:
        security = context.get_step_output("security_review", {})
        editorial = context.get_step_output("editorial_review", {})

        reasons = []
        if not security.get("passed", True):
            reasons.append(f"Security issues: {security.get('issues_found', 0)} found")
            for issue in security.get("issues", []):
                reasons.append(f"  - {issue['description']}")

        if not editorial.get("passed", True):
            reasons.append(f"Editorial score too low: {editorial.get('score', 0):.0%}")
            for suggestion in editorial.get("suggestions", []):
                reasons.append(f"  - {suggestion}")

        result = {
            "revision_requested": True,
            "reasons": reasons,
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }

        context.emit_event("revision_requested", {"reasons": len(reasons)})
        return result


class NotifyStep(BaseStep):
    """Step that notifies stakeholders of the workflow outcome."""

    @property
    def name(self) -> str:
        return "notify"

    async def execute(self, context: WorkflowContext) -> Any:
        draft = context.get_step_output("draft", {})
        publish = context.get_step_output("publish", {})
        revision = context.get_step_output("request_revision", {})

        if publish and publish.get("published"):
            message = (
                f"Content published: {draft.get('title', 'Unknown')}\n"
                f"URL: {publish.get('url', 'N/A')}"
            )
            status = "published"
        elif revision and revision.get("revision_requested"):
            reasons = "\n".join(revision.get("reasons", []))
            message = f"Revision needed for: {draft.get('title', 'Unknown')}\nReasons:\n{reasons}"
            status = "revision_needed"
        else:
            message = f"Workflow complete for: {draft.get('title', 'Unknown')}"
            status = "complete"

        logger.info("Notification: %s", message)

        result = {
            "notified": True,
            "status": status,
            "message": message,
            "notified_at": datetime.now(timezone.utc).isoformat(),
        }

        context.emit_event("stakeholders_notified", {"status": status})
        return result


# ---------------------------------------------------------------------------
# Workflow Definition Builder
# ---------------------------------------------------------------------------


def build_content_publishing_workflow(
    topic: str = "AI governance",
    demo: bool = True,
) -> tuple[WorkflowDefinition, dict[str, Any]]:
    """Build the content publishing workflow definition.

    The workflow DAG:

        [Draft] --> [Security Review] -+-> [Publish]  --> [Notify]
                --> [Editorial Review] -+
                                         +-> [Revision] --> [Notify]
                            (if any review fails)

    Returns the workflow definition and input parameters.
    """
    steps = [
        StepDefinition(
            id="draft",
            name="Draft Content",
            step_type="draft_content",
            description="Generate initial content draft on the given topic",
            config={"topic": topic},
            next_steps=["security_review", "editorial_review"],
            timeout_seconds=60.0,
        ),
        StepDefinition(
            id="security_review",
            name="Security Review",
            step_type="security_review",
            description="Check content for security issues and sensitive data",
            execution_pattern=ExecutionPattern.PARALLEL,
            timeout_seconds=30.0,
            retries=1,
        ),
        StepDefinition(
            id="editorial_review",
            name="Editorial Review",
            step_type="editorial_review",
            description="Review content structure, readability, and completeness",
            execution_pattern=ExecutionPattern.PARALLEL,
            timeout_seconds=30.0,
            retries=1,
        ),
        StepDefinition(
            id="publish",
            name="Publish Content",
            step_type="publish",
            description="Publish approved content to documentation site",
            timeout_seconds=30.0,
        ),
        StepDefinition(
            id="request_revision",
            name="Request Revision",
            step_type="request_revision",
            description="Send content back for revision with feedback",
            timeout_seconds=15.0,
        ),
        StepDefinition(
            id="notify",
            name="Notify Stakeholders",
            step_type="notify",
            description="Send notifications about workflow outcome",
            timeout_seconds=15.0,
            optional=True,
        ),
    ]

    transitions = [
        TransitionRule(
            id="draft_to_reviews",
            from_step="draft",
            to_step="security_review",
            condition="True",
            label="Start parallel reviews",
        ),
        TransitionRule(
            id="draft_to_editorial",
            from_step="draft",
            to_step="editorial_review",
            condition="True",
            label="Start editorial review",
        ),
        TransitionRule(
            id="reviews_to_publish",
            from_step="security_review",
            to_step="publish",
            condition="output.get('passed', False)",
            label="Security passed",
            priority=1,
        ),
        TransitionRule(
            id="reviews_to_revision",
            from_step="security_review",
            to_step="request_revision",
            condition="not output.get('passed', True)",
            label="Security failed",
            priority=0,
        ),
        TransitionRule(
            id="editorial_to_publish",
            from_step="editorial_review",
            to_step="publish",
            condition="output.get('passed', False)",
            label="Editorial passed",
            priority=1,
        ),
        TransitionRule(
            id="editorial_to_revision",
            from_step="editorial_review",
            to_step="request_revision",
            condition="not output.get('passed', True)",
            label="Editorial failed",
            priority=0,
        ),
        TransitionRule(
            id="publish_to_notify",
            from_step="publish",
            to_step="notify",
            condition="True",
            label="Notify on publish",
        ),
        TransitionRule(
            id="revision_to_notify",
            from_step="request_revision",
            to_step="notify",
            condition="True",
            label="Notify on revision",
        ),
    ]

    definition = WorkflowDefinition(
        id="content-publishing-pipeline",
        name="Content Publishing Pipeline",
        description=(
            "Automated content publishing with parallel security and "
            "editorial reviews, conditional branching, and stakeholder "
            "notification."
        ),
        version="1.0.0",
        steps=steps,
        transitions=transitions,
        entry_step="draft",
        metadata={
            "author": "aragora-examples",
            "created": datetime.now(timezone.utc).isoformat(),
        },
    )

    inputs = {
        "topic": topic,
        "demo": demo,
    }

    return definition, inputs


# ---------------------------------------------------------------------------
# Workflow Execution
# ---------------------------------------------------------------------------


async def run_workflow(
    topic: str = "AI governance best practices",
    demo: bool = True,
) -> dict[str, Any]:
    """Build and execute the content publishing workflow.

    Registers custom step implementations, builds the workflow definition,
    and executes it through Aragora's WorkflowEngine.
    """
    # Build the workflow
    definition, inputs = build_content_publishing_workflow(
        topic=topic,
        demo=demo,
    )

    # Configure the engine
    config = WorkflowConfig(
        total_timeout_seconds=300.0,
        step_timeout_seconds=60.0,
        stop_on_failure=False,  # Continue to notification even on failure
        enable_checkpointing=True,
    )

    # Create engine with custom step registry
    engine = WorkflowEngine(
        config=config,
        step_registry={
            "draft_content": DraftContentStep,
            "security_review": SecurityReviewStep,
            "editorial_review": EditorialReviewStep,
            "publish": PublishStep,
            "request_revision": RevisionStep,
            "notify": NotifyStep,
        },
    )

    # Track events for reporting
    events: list[dict[str, Any]] = []

    def event_handler(event_type: str, payload: dict[str, Any]) -> None:
        events.append(
            {
                "type": event_type,
                "payload": payload,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        logger.info("Event: %s -- %s", event_type, json.dumps(payload))

    logger.info("Executing workflow: %s", definition.name)
    logger.info("Topic: %s", topic)
    logger.info("Steps: %s", " -> ".join(s.id for s in definition.steps))

    start_time = time.monotonic()
    result: WorkflowResult = await engine.execute(
        definition=definition,
        inputs=inputs,
    )
    elapsed_ms = (time.monotonic() - start_time) * 1000

    # Build summary
    summary = {
        "workflow_id": result.workflow_id,
        "workflow_name": definition.name,
        "success": result.success,
        "total_steps": len(result.steps),
        "completed_steps": sum(1 for s in result.steps if s.status.value == "completed"),
        "failed_steps": sum(1 for s in result.steps if s.status.value == "failed"),
        "skipped_steps": sum(1 for s in result.steps if s.status.value == "skipped"),
        "elapsed_ms": elapsed_ms,
        "step_results": {},
        "events": events,
        "final_output": result.final_output,
        "error": result.error,
    }

    for step_result in result.steps:
        summary["step_results"][step_result.step_id] = {
            "name": step_result.step_name,
            "status": step_result.status.value,
            "duration_ms": step_result.duration_ms,
            "output": step_result.output,
            "error": step_result.error,
        }

    return summary


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------


def print_workflow_result(summary: dict[str, Any]) -> None:
    """Print a formatted workflow execution report."""
    border = "=" * 70

    print(f"\n{border}")
    print(f"  Workflow: {summary['workflow_name']}")
    print(f"  ID: {summary['workflow_id']}")
    print(border)

    # Overall status
    status = "SUCCESS" if summary["success"] else "FAILED"
    print(f"\n  Status: {status}")
    print(
        f"  Steps: {summary['completed_steps']}/{summary['total_steps']} completed, "
        f"{summary['failed_steps']} failed, "
        f"{summary['skipped_steps']} skipped"
    )
    print(f"  Duration: {summary['elapsed_ms']:.0f}ms")

    # Step-by-step breakdown
    print("\n  --- Step Results ---\n")

    step_icons = {
        "completed": "[OK]",
        "failed": "[FAIL]",
        "skipped": "[SKIP]",
        "pending": "[..]",
        "running": "[>>]",
    }

    for step_id, step_data in summary["step_results"].items():
        icon = step_icons.get(step_data["status"], "[??]")
        print(f"  {icon} {step_data['name']} ({step_id}) -- {step_data['duration_ms']:.0f}ms")

        output = step_data.get("output")
        if output and isinstance(output, dict):
            # Print key output fields
            for key in (
                "passed",
                "score",
                "published",
                "revision_requested",
                "status",
                "url",
                "title",
            ):
                if key in output:
                    print(f"       {key}: {output[key]}")
            # Print issues if any
            if output.get("issues"):
                for issue in output["issues"][:3]:
                    print(f"       - {issue.get('description', issue)}")
            # Print suggestions if any
            if output.get("suggestions"):
                for suggestion in output["suggestions"][:3]:
                    print(f"       - {suggestion}")

        if step_data.get("error"):
            print(f"       Error: {step_data['error']}")

    # Events timeline
    events = summary.get("events", [])
    if events:
        print(f"\n  --- Events ({len(events)}) ---\n")
        for evt in events:
            print(f"  [{evt['timestamp'][11:19]}] {evt['type']}: {json.dumps(evt['payload'])}")

    # Final output
    if summary.get("error"):
        print(f"\n  Error: {summary['error']}")

    print(border)


def print_workflow_dag(definition: WorkflowDefinition) -> None:
    """Print an ASCII representation of the workflow DAG."""
    print("\n  Workflow DAG:")
    print("  " + "-" * 50)

    # Build adjacency from transitions
    edges: dict[str, list[str]] = {}
    for tr in definition.transitions:
        edges.setdefault(tr.from_step, []).append(tr.to_step)

    # Simple topological print
    visited: set[str] = set()

    def print_node(step_id: str, indent: int = 0) -> None:
        if step_id in visited:
            return
        visited.add(step_id)

        step = definition.get_step(step_id)
        name = step.name if step else step_id
        prefix = "  " + "  " * indent

        children = edges.get(step_id, [])
        if children:
            child_str = " -> " + ", ".join(children)
        else:
            child_str = " (end)"

        print(f"{prefix}[{name}]{child_str}")

        for child in children:
            print_node(child, indent + 1)

    if definition.entry_step:
        print_node(definition.entry_step)

    print("  " + "-" * 50)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the workflow automation example."""
    parser = argparse.ArgumentParser(
        description="Aragora Workflow Automation Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo mode (no API keys needed)
  python examples/workflow-automation/main.py --demo

  # Custom topic with live agents
  python examples/workflow-automation/main.py --topic "API security guidelines"

  # Show workflow DAG only
  python examples/workflow-automation/main.py --show-dag
        """,
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Run in demo mode (no API calls)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="AI governance best practices",
        help="Content topic for the publishing pipeline",
    )
    parser.add_argument(
        "--show-dag",
        action="store_true",
        default=False,
        help="Show the workflow DAG and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results as JSON instead of formatted text",
    )

    args = parser.parse_args()

    # Build workflow definition for display
    definition, inputs = build_content_publishing_workflow(
        topic=args.topic,
        demo=args.demo or True,  # Always demo for now unless explicitly live
    )

    # Show DAG (skip in JSON mode for clean output)
    if not args.json:
        print_workflow_dag(definition)

    if args.show_dag:
        return

    # Execute the workflow
    use_demo = args.demo or not any(
        os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
    )

    if use_demo:
        logger.info("Running in demo mode")

    summary = await run_workflow(
        topic=args.topic,
        demo=use_demo,
    )

    # Print results
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print_workflow_result(summary)


if __name__ == "__main__":
    asyncio.run(main())
