"""
Workflow Chains Example

Demonstrates chaining multiple debates and operations
into complex workflows with conditional logic.

Usage:
    python examples/workflow_chains.py

Environment:
    ARAGORA_API_KEY - Your API key
    ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from aragora_sdk import AragoraAsyncClient

# =============================================================================
# Workflow Types
# =============================================================================


class WorkflowStatus(Enum):
    """Workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """A single step in a workflow chain."""

    name: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None


# =============================================================================
# Simple Chain: Sequential Debates
# =============================================================================


async def simple_chain(client: AragoraAsyncClient) -> list[WorkflowStep]:
    """Run debates in sequence, passing context forward."""
    print("=== Simple Chain: Sequential Debates ===\n")

    steps = []

    # Step 1: Initial analysis
    print("Step 1: Initial problem analysis...")
    step1 = WorkflowStep(name="initial_analysis", status=WorkflowStatus.RUNNING)

    debate1 = await client.debates.create(
        task="What are the key factors to consider when choosing a cloud provider?",
        agents=["claude", "gpt-4"],
        rounds=2,
    )
    await wait_for_debate(client, debate1["debate_id"])
    debate1 = await client.debates.get(debate1["debate_id"])

    step1.status = WorkflowStatus.COMPLETED
    step1.result = debate1
    steps.append(step1)
    print(f"  Completed: {debate1.get('consensus', {}).get('final_answer', 'N/A')[:80]}...")

    # Step 2: Deep dive on top factor
    print("\nStep 2: Deep dive on primary factor...")
    step2 = WorkflowStep(name="deep_dive", status=WorkflowStatus.RUNNING)

    # Extract key factor from previous debate
    key_factor = debate1.get("consensus", {}).get("final_answer", "cost")[:100]

    debate2 = await client.debates.create(
        task=f"Given that '{key_factor}' is important, how should we evaluate it specifically?",
        agents=["claude", "gpt-4", "gemini"],
        rounds=2,
        context={"previous_debate": debate1["debate_id"]},
    )
    await wait_for_debate(client, debate2["debate_id"])
    debate2 = await client.debates.get(debate2["debate_id"])

    step2.status = WorkflowStatus.COMPLETED
    step2.result = debate2
    steps.append(step2)
    print(f"  Completed: {debate2.get('consensus', {}).get('final_answer', 'N/A')[:80]}...")

    # Step 3: Final recommendation
    print("\nStep 3: Final recommendation...")
    step3 = WorkflowStep(name="recommendation", status=WorkflowStatus.RUNNING)

    debate3 = await client.debates.create(
        task="Based on our analysis, what is the final recommendation?",
        agents=["claude", "gpt-4"],
        rounds=1,
        context={
            "analysis_debate": debate1["debate_id"],
            "deep_dive_debate": debate2["debate_id"],
        },
    )
    await wait_for_debate(client, debate3["debate_id"])
    debate3 = await client.debates.get(debate3["debate_id"])

    step3.status = WorkflowStatus.COMPLETED
    step3.result = debate3
    steps.append(step3)
    print(f"  Completed: {debate3.get('consensus', {}).get('final_answer', 'N/A')[:80]}...")

    return steps


# =============================================================================
# Conditional Chain: Branch Based on Results
# =============================================================================


async def conditional_chain(client: AragoraAsyncClient) -> list[WorkflowStep]:
    """Branch workflow based on debate results."""
    print("\n=== Conditional Chain: Branching Logic ===\n")

    steps = []

    # Step 1: Classification debate
    print("Step 1: Classify the problem type...")
    step1 = WorkflowStep(name="classification", status=WorkflowStatus.RUNNING)

    debate = await client.debates.create(
        task="Is implementing a caching layer primarily a performance problem or a data consistency problem?",
        agents=["claude", "gpt-4"],
        rounds=2,
    )
    await wait_for_debate(client, debate["debate_id"])
    debate = await client.debates.get(debate["debate_id"])

    step1.status = WorkflowStatus.COMPLETED
    step1.result = debate
    steps.append(step1)

    # Determine branch based on result
    answer = debate.get("consensus", {}).get("final_answer", "").lower()
    is_performance = "performance" in answer

    print(f"  Classification: {'Performance' if is_performance else 'Consistency'} problem")

    # Step 2: Branch-specific debate
    if is_performance:
        print("\nStep 2a: Performance optimization strategies...")
        step2 = WorkflowStep(name="performance_analysis", status=WorkflowStatus.RUNNING)

        debate2 = await client.debates.create(
            task="What caching strategies best optimize for performance (latency, throughput)?",
            agents=["claude", "gpt-4"],
            rounds=2,
        )
    else:
        print("\nStep 2b: Consistency strategies...")
        step2 = WorkflowStep(name="consistency_analysis", status=WorkflowStatus.RUNNING)

        debate2 = await client.debates.create(
            task="What caching strategies best maintain data consistency?",
            agents=["claude", "gpt-4"],
            rounds=2,
        )

    await wait_for_debate(client, debate2["debate_id"])
    debate2 = await client.debates.get(debate2["debate_id"])

    step2.status = WorkflowStatus.COMPLETED
    step2.result = debate2
    steps.append(step2)
    print(f"  Completed: {debate2.get('consensus', {}).get('final_answer', 'N/A')[:80]}...")

    return steps


# =============================================================================
# Parallel Chain: Concurrent Debates
# =============================================================================


async def parallel_chain(client: AragoraAsyncClient) -> list[WorkflowStep]:
    """Run multiple debates in parallel, then synthesize."""
    print("\n=== Parallel Chain: Concurrent Analysis ===\n")

    steps = []

    # Step 1: Parallel analysis from different angles
    print("Step 1: Running 3 debates in parallel...")

    tasks = [
        client.debates.create(
            task="What are the security considerations for a REST API?",
            agents=["claude", "gpt-4"],
            rounds=2,
        ),
        client.debates.create(
            task="What are the performance considerations for a REST API?",
            agents=["claude", "gpt-4"],
            rounds=2,
        ),
        client.debates.create(
            task="What are the usability considerations for a REST API?",
            agents=["claude", "gpt-4"],
            rounds=2,
        ),
    ]

    debates = await asyncio.gather(*tasks)
    debate_ids = [d["debate_id"] for d in debates]
    print(f"  Created {len(debate_ids)} debates")

    # Wait for all to complete
    print("  Waiting for all debates to complete...")
    await asyncio.gather(*[wait_for_debate(client, did) for did in debate_ids])

    for name, did in [
        ("security", debate_ids[0]),
        ("performance", debate_ids[1]),
        ("usability", debate_ids[2]),
    ]:
        debate = await client.debates.get(did)
        step = WorkflowStep(
            name=f"parallel_{name}",
            status=WorkflowStatus.COMPLETED,
            result=debate,
        )
        steps.append(step)
        print(
            f"  {name.capitalize()}: {debate.get('consensus', {}).get('final_answer', 'N/A')[:50]}..."
        )

    # Step 2: Synthesis
    print("\nStep 2: Synthesizing parallel results...")
    step_synthesis = WorkflowStep(name="synthesis", status=WorkflowStatus.RUNNING)

    synthesis = await client.debates.create(
        task="Given security, performance, and usability analyses, what is the balanced API design approach?",
        agents=["claude", "gpt-4", "gemini"],
        rounds=2,
        context={
            "security_debate": debate_ids[0],
            "performance_debate": debate_ids[1],
            "usability_debate": debate_ids[2],
        },
    )
    await wait_for_debate(client, synthesis["debate_id"])
    synthesis = await client.debates.get(synthesis["debate_id"])

    step_synthesis.status = WorkflowStatus.COMPLETED
    step_synthesis.result = synthesis
    steps.append(step_synthesis)
    print(f"  Synthesis: {synthesis.get('consensus', {}).get('final_answer', 'N/A')[:80]}...")

    return steps


# =============================================================================
# Template-Based Workflow
# =============================================================================


async def template_workflow(client: AragoraAsyncClient) -> dict[str, Any]:
    """Use pre-built workflow templates."""
    print("\n=== Template-Based Workflow ===\n")

    # List available templates
    print("Available workflow templates:")
    templates = await client.workflows.list_templates()
    for tmpl in templates.get("templates", [])[:5]:
        print(f"  - {tmpl['name']}: {tmpl.get('description', 'N/A')[:50]}...")

    # Use a template
    print("\nExecuting 'code_review' template...")
    workflow = await client.workflows.create_from_template(
        template="code_review",
        parameters={
            "code": "def add(a, b): return a + b",
            "language": "python",
            "focus_areas": ["security", "performance", "readability"],
        },
    )

    workflow_id = workflow["workflow_id"]
    print(f"Workflow created: {workflow_id}")

    # Wait for completion
    while workflow.get("status") in ("running", "pending"):
        await asyncio.sleep(2)
        workflow = await client.workflows.get(workflow_id)
        print(f"  Status: {workflow['status']}")

    # Show results
    if workflow.get("status") == "completed":
        print("\n--- Workflow Results ---")
        for step_result in workflow.get("step_results", []):
            print(f"\n{step_result['step_name']}:")
            print(f"  {step_result.get('summary', 'N/A')[:100]}...")

    return workflow


# =============================================================================
# Error Handling in Chains
# =============================================================================


async def error_handling_chain(client: AragoraAsyncClient) -> list[WorkflowStep]:
    """Demonstrate error handling in workflow chains."""
    print("\n=== Error Handling in Chains ===\n")

    steps = []

    # Step 1: Normal step
    print("Step 1: Initial analysis...")
    step1 = WorkflowStep(name="step1", status=WorkflowStatus.RUNNING)

    try:
        debate = await client.debates.create(
            task="What is 2+2?",
            agents=["claude"],
            rounds=1,
        )
        await wait_for_debate(client, debate["debate_id"])
        debate = await client.debates.get(debate["debate_id"])
        step1.status = WorkflowStatus.COMPLETED
        step1.result = debate
        print("  Completed successfully")
    except Exception as e:
        step1.status = WorkflowStatus.FAILED
        step1.error = str(e)
        print(f"  Failed: {e}")

    steps.append(step1)

    # Step 2: Conditional on step 1 success
    if step1.status == WorkflowStatus.COMPLETED:
        print("\nStep 2: Follow-up (step 1 succeeded)...")
        step2 = WorkflowStep(name="step2", status=WorkflowStatus.RUNNING)

        try:
            debate = await client.debates.create(
                task="What is 3+3?",
                agents=["claude"],
                rounds=1,
            )
            await wait_for_debate(client, debate["debate_id"])
            debate = await client.debates.get(debate["debate_id"])
            step2.status = WorkflowStatus.COMPLETED
            step2.result = debate
            print("  Completed successfully")
        except Exception as e:
            step2.status = WorkflowStatus.FAILED
            step2.error = str(e)
            print(f"  Failed: {e}")

        steps.append(step2)
    else:
        print("\nStep 2: Skipped (step 1 failed)")
        steps.append(WorkflowStep(name="step2", status=WorkflowStatus.SKIPPED))

    return steps


# =============================================================================
# Helpers
# =============================================================================


async def wait_for_debate(
    client: AragoraAsyncClient,
    debate_id: str,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Wait for a debate to complete."""
    elapsed = 0.0
    interval = 2.0

    while elapsed < timeout:
        debate = await client.debates.get(debate_id)
        if debate.get("status") not in ("running", "pending"):
            return debate
        await asyncio.sleep(interval)
        elapsed += interval

    raise TimeoutError(f"Debate {debate_id} did not complete within {timeout}s")


def print_workflow_summary(steps: list[WorkflowStep]) -> None:
    """Print a summary of workflow steps."""
    print("\n--- Workflow Summary ---")
    for i, step in enumerate(steps, 1):
        status_icon = {
            WorkflowStatus.COMPLETED: "[OK]",
            WorkflowStatus.FAILED: "[X]",
            WorkflowStatus.SKIPPED: "[-]",
            WorkflowStatus.RUNNING: "[...]",
            WorkflowStatus.PENDING: "[ ]",
        }.get(step.status, "[?]")

        print(f"{i}. {status_icon} {step.name}")
        if step.error:
            print(f"      Error: {step.error}")


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run workflow chain demonstrations."""
    print("Aragora SDK Workflow Chains Example")
    print("=" * 60)

    # Check if we should run actual examples
    run_examples = os.environ.get("RUN_EXAMPLES", "false").lower() == "true"

    if not run_examples:
        print("\nWorkflow patterns demonstrated:")
        print("  1. Simple Chain: Sequential debates passing context")
        print("  2. Conditional Chain: Branch based on results")
        print("  3. Parallel Chain: Concurrent debates + synthesis")
        print("  4. Template Workflow: Pre-built workflow templates")
        print("  5. Error Handling: Graceful failure handling")
        print("\nSet RUN_EXAMPLES=true to run actual API examples.")
        return

    async with AragoraAsyncClient(
        base_url=os.environ.get("ARAGORA_API_URL", "https://api.aragora.ai"),
        api_key=os.environ.get("ARAGORA_API_KEY"),
    ) as client:
        # Run demonstrations
        steps1 = await simple_chain(client)
        print_workflow_summary(steps1)

        steps2 = await conditional_chain(client)
        print_workflow_summary(steps2)

        steps3 = await parallel_chain(client)
        print_workflow_summary(steps3)

        await template_workflow(client)

        steps4 = await error_handling_chain(client)
        print_workflow_summary(steps4)

    print("\n" + "=" * 60)
    print("Workflow chains example complete!")


if __name__ == "__main__":
    asyncio.run(main())
