#!/usr/bin/env python3
"""
13_pipeline_execution.py - Create and execute idea-to-execution pipelines.

The pipeline system transforms ideas into running code through 4 stages:
  Ideas → Goals → Actions → Orchestration

This example shows how to create a pipeline, add stages, and trigger
autonomous execution via the self-improvement system.

Usage:
    python 13_pipeline_execution.py                    # Run pipeline
    python 13_pipeline_execution.py --dry-run          # Preview only
"""

import argparse
import asyncio
from aragora_sdk import AragoraClient


async def run_pipeline(dry_run: bool = False) -> dict:
    """Create and execute a full pipeline."""

    client = AragoraClient()

    if dry_run:
        print("[DRY RUN] Would create and execute pipeline")
        return {"status": "dry_run"}

    # Stage 1: Create ideas
    pipeline = await client.pipeline.create(name="SDK Enhancement Pipeline")
    pipeline_id = pipeline["id"]
    print(f"Created pipeline: {pipeline_id}")

    ideas = [
        {"title": "Add retry middleware to SDK", "description": "Auto-retry on transient failures"},
        {
            "title": "Add response caching",
            "description": "Cache debate results for repeated queries",
        },
    ]

    for idea in ideas:
        node = await client.pipeline.add_node(
            pipeline_id,
            stage="ideas",
            data=idea,
        )
        print(f"  Added idea: {idea['title']} ({node.get('id', '?')})")

    # Stage 2: Derive goals from ideas
    goals = await client.pipeline.advance(pipeline_id, from_stage="ideas", to_stage="goals")
    print(f"\nDerived {len(goals.get('nodes', []))} goals")

    # Stage 3: Plan actions
    actions = await client.pipeline.advance(pipeline_id, from_stage="goals", to_stage="actions")
    print(f"Planned {len(actions.get('nodes', []))} actions")

    # Stage 4: Execute via self-improvement
    execution = await client.pipeline.execute(
        pipeline_id,
        require_approval=True,
        budget_limit_usd=10.0,
    )

    cycle_id = execution["cycle_id"]
    print(f"\nExecution started: {cycle_id}")

    # Monitor progress
    while True:
        status = await client.pipeline.execution_status(pipeline_id)
        print(f"  Status: {status.get('phase', '?')} ({status.get('progress', 0)}%)")

        if status.get("completed"):
            break
        await asyncio.sleep(5)

    # Get the receipt
    receipt = await client.pipeline.receipt(pipeline_id)
    print(f"\nReceipt: {receipt.get('receipt_id', '?')}")
    print("  Ideas → Goals → Actions → Results")
    print(f"  Files changed: {receipt.get('files_changed', 0)}")
    print(f"  Content hash: {receipt.get('content_hash', '?')[:16]}")

    return receipt


def main():
    parser = argparse.ArgumentParser(description="Pipeline execution via SDK")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = asyncio.run(run_pipeline(args.dry_run))
    print(f"\nDone: {result.get('status', 'ok')}")


if __name__ == "__main__":
    main()
