#!/usr/bin/env python3
"""
06_batch_debates.py - Run multiple debates in parallel.

This example demonstrates how to:
- Execute multiple debates concurrently
- Aggregate results across debates
- Handle errors in individual debates without failing the batch

Usage:
    python 06_batch_debates.py --dry-run
    python 06_batch_debates.py --max-concurrent 3
"""

import argparse
import asyncio
from typing import Any
from aragora_sdk import ArenaClient, DebateConfig, Agent
from aragora_sdk.batch import BatchResult


async def run_single_debate(
    client: ArenaClient, topic: str, debate_id: int, dry_run: bool = False
) -> dict[str, Any]:
    """Run a single debate with error handling."""
    try:
        if dry_run:
            # Simulate varying outcomes
            await asyncio.sleep(0.1 * debate_id)  # Simulate work
            return {
                "debate_id": debate_id,
                "topic": topic,
                "status": "success",
                "consensus_reached": debate_id % 2 == 0,
                "confidence": 0.7 + (debate_id * 0.05),
            }

        agents = [
            Agent(name="claude", model="claude-sonnet-4-20250514"),
            Agent(name="gpt", model="gpt-4o"),
        ]

        config = DebateConfig(topic=topic, agents=agents, rounds=2)
        result = await client.run_debate(config)

        return {
            "debate_id": debate_id,
            "topic": topic,
            "status": "success",
            "consensus_reached": result.consensus_reached,
            "decision": result.decision,
            "confidence": result.confidence,
        }

    except Exception as e:
        # Return error info instead of raising - allows batch to continue
        return {
            "debate_id": debate_id,
            "topic": topic,
            "status": "error",
            "error": str(e),
        }


async def run_batch_debates(
    topics: list[str], max_concurrent: int = 5, dry_run: bool = False
) -> BatchResult:
    """Run multiple debates in parallel with controlled concurrency."""

    client = ArenaClient()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_debate(topic: str, debate_id: int) -> dict[str, Any]:
        """Run debate with semaphore to limit concurrency."""
        async with semaphore:
            print(f"[{debate_id}] Starting: {topic[:50]}...")
            result = await run_single_debate(client, topic, debate_id, dry_run)
            status = "OK" if result["status"] == "success" else "FAILED"
            print(f"[{debate_id}] {status}")
            return result

    # Run all debates concurrently (semaphore limits actual parallelism)
    tasks = [limited_debate(topic, i) for i, topic in enumerate(topics)]
    results = await asyncio.gather(*tasks)

    # Aggregate results
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    consensus_reached = [r for r in successful if r.get("consensus_reached")]

    print("\n=== Batch Summary ===")
    print(f"Total debates: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Consensus reached: {len(consensus_reached)}/{len(successful)}")

    if successful:
        avg_confidence = sum(r.get("confidence", 0) for r in successful) / len(successful)
        print(f"Average confidence: {avg_confidence:.2%}")

    return BatchResult(
        total=len(results),
        successful=len(successful),
        failed=len(failed),
        results=results,
        consensus_rate=len(consensus_reached) / len(successful) if successful else 0,
    )


def main():
    parser = argparse.ArgumentParser(description="Run multiple debates in parallel")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Maximum concurrent debates")
    parser.add_argument("--dry-run", action="store_true", help="Test without API calls")
    args = parser.parse_args()

    # Example topics for batch processing
    topics = [
        "Should we use microservices or monolith for our new project?",
        "What testing strategy should we adopt: TDD or BDD?",
        "Should we migrate to Kubernetes or stay with VMs?",
        "Which database: PostgreSQL, MongoDB, or DynamoDB?",
        "Should we build or buy our analytics solution?",
        "What CI/CD tool should we standardize on?",
    ]

    print(f"Running {len(topics)} debates with max {args.max_concurrent} concurrent...")

    result = asyncio.run(run_batch_debates(topics, args.max_concurrent, args.dry_run))
    return result


if __name__ == "__main__":
    main()
