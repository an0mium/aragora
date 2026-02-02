"""
Batch Operations Example

Demonstrates bulk debate operations for high-throughput scenarios.
Shows how to submit multiple debates, track progress, and handle
partial failures efficiently.

Usage:
    python examples/batch_operations.py

Environment:
    ARAGORA_API_KEY - Your API key
    ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aragora_sdk import AragoraAsyncClient
from aragora_sdk.exceptions import AragoraError, RateLimitError

# =============================================================================
# Data Structures
# =============================================================================


class BatchItemStatus(Enum):
    """Status of a batch item."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchItem:
    """A single item in a batch operation."""

    id: str
    task: str
    status: BatchItemStatus = BatchItemStatus.PENDING
    debate_id: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class BatchResult:
    """Result of a batch operation."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    items: list[BatchItem] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total == 0:
            return 0.0
        return self.completed / self.total


# =============================================================================
# Simple Batch Submission
# =============================================================================


async def simple_batch(client: AragoraAsyncClient) -> BatchResult:
    """Submit multiple debates concurrently."""
    print("=== Simple Batch Submission ===\n")

    tasks = [
        "What is the best database for a startup?",
        "Should we use Kubernetes or serverless?",
        "What CI/CD tool should we adopt?",
        "How should we handle authentication?",
        "What monitoring solution is best?",
    ]

    print(f"Submitting {len(tasks)} debates concurrently...")

    # Submit all debates concurrently
    async def submit_debate(task: str, index: int) -> BatchItem:
        item = BatchItem(id=f"batch-{index}", task=task)
        try:
            debate = await client.debates.create(
                task=task,
                agents=["claude", "gpt-4"],
                rounds=2,
            )
            item.debate_id = debate["debate_id"]
            item.status = BatchItemStatus.RUNNING
            print(f"  [{index + 1}/{len(tasks)}] Submitted: {debate['debate_id']}")
        except AragoraError as e:
            item.status = BatchItemStatus.FAILED
            item.error = str(e)
            print(f"  [{index + 1}/{len(tasks)}] Failed: {e.message}")
        return item

    items = await asyncio.gather(*[submit_debate(task, i) for i, task in enumerate(tasks)])

    # Wait for all to complete
    print("\nWaiting for debates to complete...")
    running = [item for item in items if item.status == BatchItemStatus.RUNNING]

    while running:
        await asyncio.sleep(3)
        for item in running[:]:
            if item.debate_id:
                debate = await client.debates.get(item.debate_id)
                if debate.get("status") not in ("running", "pending"):
                    item.status = BatchItemStatus.COMPLETED
                    item.result = debate
                    running.remove(item)
                    print(f"  Completed: {item.debate_id}")

    # Build result
    result = BatchResult(
        total=len(items),
        completed=sum(1 for i in items if i.status == BatchItemStatus.COMPLETED),
        failed=sum(1 for i in items if i.status == BatchItemStatus.FAILED),
        items=list(items),
    )

    print(f"\nBatch complete: {result.completed}/{result.total} succeeded")
    return result


# =============================================================================
# Rate-Limited Batch
# =============================================================================


async def rate_limited_batch(
    client: AragoraAsyncClient,
    max_concurrent: int = 3,
) -> BatchResult:
    """Submit debates with rate limiting and concurrency control."""
    print("\n=== Rate-Limited Batch ===\n")

    tasks = [f"Question {i}: What is the best approach?" for i in range(10)]

    print(f"Submitting {len(tasks)} debates (max {max_concurrent} concurrent)...")

    semaphore = asyncio.Semaphore(max_concurrent)
    items: list[BatchItem] = []

    async def submit_with_limit(task: str, index: int) -> BatchItem:
        async with semaphore:
            item = BatchItem(id=f"rate-{index}", task=task)
            try:
                # Retry on rate limit
                for attempt in range(3):
                    try:
                        debate = await client.debates.create(
                            task=task,
                            agents=["claude"],
                            rounds=1,
                        )
                        item.debate_id = debate["debate_id"]
                        item.status = BatchItemStatus.RUNNING
                        print(f"  [{index + 1}] Submitted")
                        break
                    except RateLimitError as e:
                        wait_time = e.retry_after or (2**attempt)
                        print(f"  [{index + 1}] Rate limited, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                else:
                    item.status = BatchItemStatus.FAILED
                    item.error = "Max retries exceeded"
            except AragoraError as e:
                item.status = BatchItemStatus.FAILED
                item.error = str(e)
            return item

    items = await asyncio.gather(*[submit_with_limit(task, i) for i, task in enumerate(tasks)])

    # Count results
    result = BatchResult(
        total=len(items),
        completed=sum(1 for i in items if i.status == BatchItemStatus.RUNNING),
        failed=sum(1 for i in items if i.status == BatchItemStatus.FAILED),
        items=list(items),
    )

    print(f"\nSubmission complete: {result.completed} submitted, {result.failed} failed")
    return result


# =============================================================================
# Using the Batch API
# =============================================================================


async def batch_api(client: AragoraAsyncClient) -> dict[str, Any]:
    """Use the built-in batch API for server-side batching."""
    print("\n=== Batch API ===\n")

    tasks = [
        {"task": "What is the best web framework?", "agents": ["claude", "gpt-4"]},
        {"task": "What database should we use?", "agents": ["claude", "gpt-4"]},
        {"task": "How should we structure our API?", "agents": ["claude", "gpt-4"]},
    ]

    print(f"Creating batch with {len(tasks)} debates...")

    # Submit batch
    batch = await client.batch.create(
        debates=tasks,
        options={
            "rounds": 2,
            "consensus": "weighted",
            "parallel": True,
        },
    )

    batch_id = batch["batch_id"]
    print(f"Batch created: {batch_id}")

    # Poll for completion
    print("Waiting for batch to complete...")
    while batch.get("status") in ("running", "pending"):
        await asyncio.sleep(3)
        batch = await client.batch.get(batch_id)
        progress = batch.get("progress", {})
        completed = progress.get("completed", 0)
        total = progress.get("total", 0)
        print(f"  Progress: {completed}/{total}")

    # Get results
    print("\n--- Batch Results ---")
    print(f"Status: {batch.get('status')}")
    print(f"Duration: {batch.get('duration_seconds', 0):.1f}s")

    results = batch.get("results", [])
    for i, result in enumerate(results, 1):
        status = result.get("status", "unknown")
        debate_id = result.get("debate_id", "N/A")
        consensus = result.get("consensus", {})
        answer = consensus.get("final_answer", "N/A")[:50]
        print(f"\n{i}. [{status}] {debate_id}")
        print(f"   Answer: {answer}...")

    return batch


# =============================================================================
# Error Handling in Batches
# =============================================================================


async def batch_with_error_handling(
    client: AragoraAsyncClient,
) -> BatchResult:
    """Demonstrate error handling in batch operations."""
    print("\n=== Batch Error Handling ===\n")

    # Mix of valid and invalid tasks
    tasks = [
        {"task": "Valid question 1?", "valid": True},
        {"task": "", "valid": False},  # Invalid: empty task
        {"task": "Valid question 2?", "valid": True},
        {"task": "x" * 10000, "valid": False},  # Invalid: too long
        {"task": "Valid question 3?", "valid": True},
    ]

    items: list[BatchItem] = []

    for i, task_data in enumerate(tasks):
        item = BatchItem(id=f"err-{i}", task=task_data["task"][:50])

        try:
            debate = await client.debates.create(
                task=task_data["task"],
                agents=["claude"],
                rounds=1,
            )
            item.debate_id = debate["debate_id"]
            item.status = BatchItemStatus.COMPLETED
            print(f"  [{i + 1}] OK: {debate['debate_id']}")
        except AragoraError as e:
            item.status = BatchItemStatus.FAILED
            item.error = e.message
            print(f"  [{i + 1}] FAILED: {e.message}")

        items.append(item)

    # Summary
    result = BatchResult(
        total=len(items),
        completed=sum(1 for i in items if i.status == BatchItemStatus.COMPLETED),
        failed=sum(1 for i in items if i.status == BatchItemStatus.FAILED),
        items=items,
    )

    print(f"\nBatch complete: {result.success_rate:.0%} success rate")
    print(f"  Completed: {result.completed}")
    print(f"  Failed: {result.failed}")

    # Show failures
    failures = [i for i in items if i.status == BatchItemStatus.FAILED]
    if failures:
        print("\nFailures:")
        for item in failures:
            print(f"  - {item.id}: {item.error}")

    return result


# =============================================================================
# Batch Progress Callback
# =============================================================================


async def batch_with_progress(client: AragoraAsyncClient) -> None:
    """Track batch progress with callbacks."""
    print("\n=== Batch with Progress Tracking ===\n")

    tasks = [f"Question {i}?" for i in range(5)]
    total = len(tasks)
    completed = 0

    def on_progress(current: int, total: int, item: BatchItem) -> None:
        """Progress callback."""
        pct = (current / total) * 100
        bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
        status = "OK" if item.status == BatchItemStatus.COMPLETED else "FAIL"
        print(f"\r  [{bar}] {current}/{total} ({pct:.0f}%) - Last: {status}", end="")

    print(f"Processing {total} items...")

    for i, task in enumerate(tasks):
        item = BatchItem(id=f"prog-{i}", task=task)

        try:
            debate = await client.debates.create(
                task=task,
                agents=["claude"],
                rounds=1,
            )
            item.debate_id = debate["debate_id"]
            item.status = BatchItemStatus.COMPLETED
        except AragoraError as e:
            item.status = BatchItemStatus.FAILED
            item.error = str(e)

        completed += 1
        on_progress(completed, total, item)

    print("\n\nBatch complete!")


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run batch operations demonstration."""
    print("Aragora SDK Batch Operations")
    print("=" * 60)

    # Check if we should run actual examples
    run_examples = os.environ.get("RUN_EXAMPLES", "false").lower() == "true"

    if not run_examples:
        print("\nBatch operation patterns:")
        print("  1. Simple Batch: Submit multiple debates concurrently")
        print("  2. Rate-Limited: Control concurrency with semaphores")
        print("  3. Batch API: Server-side batching for efficiency")
        print("  4. Error Handling: Handle partial failures gracefully")
        print("  5. Progress Tracking: Monitor batch progress")
        print("\nSet RUN_EXAMPLES=true to run actual API examples.")
        return

    async with AragoraAsyncClient(
        base_url=os.environ.get("ARAGORA_API_URL", "https://api.aragora.ai"),
        api_key=os.environ.get("ARAGORA_API_KEY"),
    ) as client:
        # Simple batch
        await simple_batch(client)

        # Rate-limited batch
        await rate_limited_batch(client, max_concurrent=3)

        # Batch API
        await batch_api(client)

        # Error handling
        await batch_with_error_handling(client)

        # Progress tracking
        await batch_with_progress(client)

    print("\n" + "=" * 60)
    print("Batch operations complete!")
    print("\nBest Practices:")
    print("  - Use semaphores to control concurrency")
    print("  - Implement retry logic for rate limits")
    print("  - Handle partial failures gracefully")
    print("  - Use server-side batch API for large batches")
    print("  - Track progress for long-running batches")


if __name__ == "__main__":
    asyncio.run(main())
