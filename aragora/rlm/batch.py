"""
Batch parallelism utilities for RLM operations.

Implements the llm_batch() pattern from Prime Intellect's RLM paper
(arXiv:2512.24601) - parallel delegation to multiple sub-LLMs with
optional early stopping.

Key insight: The main LLM orchestrates, sub-LLMs execute in parallel.
Each sub-LLM gets a fresh context window, avoiding context pollution.

Usage:
    from aragora.rlm.batch import llm_batch, BatchConfig

    # Basic parallel execution
    results = await llm_batch(
        items=[proposal1, proposal2, proposal3],
        process_fn=generate_critique,
        max_concurrent=3,
    )

    # With early stopping (e.g., majority vote)
    results = await llm_batch(
        items=agents,
        process_fn=lambda agent: agent.vote(proposal),
        early_stop=lambda results: has_majority(results),
    )

    # With configuration
    config = BatchConfig(
        max_concurrent=5,
        timeout_per_item=30.0,
        retry_on_error=True,
        max_retries=2,
    )
    results = await llm_batch(items, process_fn, config=config)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Generic,
    Optional,
    TypeVar,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type


class BatchItemStatus(Enum):
    """Status of an individual batch item."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    max_concurrent: int = 5
    """Maximum number of concurrent LLM calls."""

    timeout_per_item: Optional[float] = 60.0
    """Timeout in seconds per item. None for no timeout."""

    retry_on_error: bool = False
    """Whether to retry failed items."""

    max_retries: int = 2
    """Maximum retry attempts per item."""

    retry_delay: float = 1.0
    """Delay between retries in seconds."""

    fail_fast: bool = False
    """Stop all processing on first failure."""

    preserve_order: bool = True
    """Return results in original input order."""


@dataclass
class BatchItemResult(Generic[T, R]):
    """Result for a single batch item."""

    index: int
    """Original index of the item."""

    item: T
    """The input item."""

    result: Optional[R] = None
    """The result if successful."""

    error: Optional[Exception] = None
    """The error if failed."""

    status: BatchItemStatus = BatchItemStatus.PENDING
    """Current status."""

    duration_seconds: float = 0.0
    """Time taken to process."""

    attempts: int = 0
    """Number of processing attempts."""


@dataclass
class BatchResult(Generic[T, R]):
    """Result of a batch operation."""

    items: list[BatchItemResult[T, R]] = field(default_factory=list)
    """Results for each item."""

    total_duration_seconds: float = 0.0
    """Total time for the batch operation."""

    early_stopped: bool = False
    """Whether early stopping was triggered."""

    early_stop_at_index: Optional[int] = None
    """Index where early stopping was triggered."""

    @property
    def results(self) -> list[R]:
        """Get successful results only (in order if preserve_order was True)."""
        return [
            item.result
            for item in self.items
            if item.status == BatchItemStatus.COMPLETED and item.result is not None
        ]

    @property
    def all_succeeded(self) -> bool:
        """Check if all items succeeded."""
        return all(
            item.status == BatchItemStatus.COMPLETED for item in self.items
        )

    @property
    def success_count(self) -> int:
        """Count of successful items."""
        return sum(1 for item in self.items if item.status == BatchItemStatus.COMPLETED)

    @property
    def failure_count(self) -> int:
        """Count of failed items."""
        return sum(1 for item in self.items if item.status == BatchItemStatus.FAILED)

    @property
    def errors(self) -> list[tuple[int, Exception]]:
        """Get all errors with their indices."""
        return [
            (item.index, item.error)
            for item in self.items
            if item.error is not None
        ]


async def llm_batch(
    items: list[T],
    process_fn: Callable[[T], Awaitable[R]],
    max_concurrent: int = 5,
    early_stop: Optional[Callable[[list[R]], bool]] = None,
    config: Optional[BatchConfig] = None,
) -> list[R]:
    """
    Execute LLM calls in parallel with optional early stopping.

    Based on the RLM paper's llm_batch() pattern - delegates work to
    multiple sub-LLMs in parallel, each with a fresh context window.

    Args:
        items: Items to process
        process_fn: Async function to call for each item
        max_concurrent: Max parallel calls (overridden by config if provided)
        early_stop: Optional function to check if we can stop early.
                   Called after each completed item with list of results so far.
        config: Full configuration (overrides max_concurrent if provided)

    Returns:
        List of results (may be partial if early_stop triggered or errors occurred).
        Results are in the same order as input items if preserve_order is True.

    Example:
        # Parallel critique generation
        critiques = await llm_batch(
            items=proposals,
            process_fn=generate_critique,
            max_concurrent=3,
        )

        # Voting with early majority detection
        votes = await llm_batch(
            items=voters,
            process_fn=lambda v: v.cast_vote(proposal),
            early_stop=lambda votes: count_votes(votes) > len(voters) // 2,
        )
    """
    if not items:
        return []

    # Use config or create from max_concurrent
    cfg = config or BatchConfig(max_concurrent=max_concurrent)

    batch_result = await llm_batch_detailed(
        items=items,
        process_fn=process_fn,
        early_stop=early_stop,
        config=cfg,
    )

    return batch_result.results


async def llm_batch_detailed(
    items: list[T],
    process_fn: Callable[[T], Awaitable[R]],
    early_stop: Optional[Callable[[list[R]], bool]] = None,
    config: Optional[BatchConfig] = None,
) -> BatchResult[T, R]:
    """
    Execute LLM calls in parallel with detailed result tracking.

    Like llm_batch(), but returns BatchResult with full metadata including
    timing, errors, and status for each item.

    Args:
        items: Items to process
        process_fn: Async function to call for each item
        early_stop: Optional function to check if we can stop early
        config: Configuration options

    Returns:
        BatchResult with detailed information about each item's processing.
    """
    if not items:
        return BatchResult()

    cfg = config or BatchConfig()
    batch_result = BatchResult[T, R]()
    start_time = time.time()

    # Initialize item results
    item_results: dict[int, BatchItemResult[T, R]] = {
        i: BatchItemResult(index=i, item=item)
        for i, item in enumerate(items)
    }

    # Semaphore for concurrency control
    semaphore = asyncio.Semaphore(cfg.max_concurrent)

    # Track completed results for early stopping
    completed_results: list[R] = []
    early_stop_triggered = asyncio.Event()
    early_stop_index: Optional[int] = None

    async def process_item(index: int, item: T) -> None:
        """Process a single item with retries and timeout."""
        nonlocal early_stop_index

        item_result = item_results[index]
        item_result.status = BatchItemStatus.RUNNING

        for attempt in range(cfg.max_retries + 1):
            if early_stop_triggered.is_set():
                item_result.status = BatchItemStatus.CANCELLED
                return

            item_result.attempts = attempt + 1
            item_start = time.time()

            try:
                async with semaphore:
                    if early_stop_triggered.is_set():
                        item_result.status = BatchItemStatus.CANCELLED
                        return

                    # Apply timeout if configured
                    if cfg.timeout_per_item:
                        result = await asyncio.wait_for(
                            process_fn(item),
                            timeout=cfg.timeout_per_item,
                        )
                    else:
                        result = await process_fn(item)

                item_result.result = result
                item_result.status = BatchItemStatus.COMPLETED
                item_result.duration_seconds = time.time() - item_start

                # Add to completed results for early stopping
                completed_results.append(result)

                # Check early stopping condition
                if early_stop and early_stop(completed_results):
                    early_stop_index = index
                    early_stop_triggered.set()

                return

            except asyncio.TimeoutError:
                item_result.status = BatchItemStatus.TIMEOUT
                item_result.error = TimeoutError(
                    f"Item {index} timed out after {cfg.timeout_per_item}s"
                )
                item_result.duration_seconds = time.time() - item_start
                logger.warning(f"Batch item {index} timed out")

                if cfg.fail_fast:
                    early_stop_triggered.set()
                    return

            except Exception as e:
                item_result.error = e
                item_result.status = BatchItemStatus.FAILED
                item_result.duration_seconds = time.time() - item_start
                logger.warning(f"Batch item {index} failed (attempt {attempt + 1}): {e}")

                if cfg.fail_fast:
                    early_stop_triggered.set()
                    return

                if cfg.retry_on_error and attempt < cfg.max_retries:
                    await asyncio.sleep(cfg.retry_delay)
                    continue

                # No more retries
                return

    # Create all tasks
    tasks = [
        asyncio.create_task(process_item(i, item))
        for i, item in enumerate(items)
    ]

    # Wait for all tasks to complete (or be cancelled)
    await asyncio.gather(*tasks, return_exceptions=True)

    # Build final result
    batch_result.total_duration_seconds = time.time() - start_time
    batch_result.early_stopped = early_stop_triggered.is_set()
    batch_result.early_stop_at_index = early_stop_index

    # Add item results in order
    if cfg.preserve_order:
        batch_result.items = [item_results[i] for i in range(len(items))]
    else:
        # Order by completion
        batch_result.items = sorted(
            item_results.values(),
            key=lambda x: (x.status != BatchItemStatus.COMPLETED, x.index),
        )

    return batch_result


async def llm_batch_with_progress(
    items: list[T],
    process_fn: Callable[[T], Awaitable[R]],
    on_progress: Callable[[int, int, BatchItemResult[T, R]], None],
    config: Optional[BatchConfig] = None,
) -> BatchResult[T, R]:
    """
    Execute batch with progress callback for each completed item.

    Args:
        items: Items to process
        process_fn: Async function to call for each item
        on_progress: Callback(completed_count, total_count, item_result)
        config: Configuration options

    Returns:
        BatchResult with detailed information.
    """
    cfg = config or BatchConfig()
    completed_count = 0
    total = len(items)

    async def wrapped_process(item: T) -> R:
        nonlocal completed_count
        result = await process_fn(item)
        completed_count += 1
        return result

    # Run with detailed tracking
    batch_result = await llm_batch_detailed(
        items=items,
        process_fn=wrapped_process,
        config=cfg,
    )

    # Call progress for each completed item
    for item_result in batch_result.items:
        if item_result.status == BatchItemStatus.COMPLETED:
            on_progress(completed_count, total, item_result)

    return batch_result


# Convenience functions for common patterns


async def batch_map(
    items: list[T],
    fn: Callable[[T], Awaitable[R]],
    concurrency: int = 5,
) -> list[R]:
    """
    Map a function over items in parallel.

    Simple wrapper around llm_batch for map-style operations.
    """
    return await llm_batch(items, fn, max_concurrent=concurrency)


async def batch_filter(
    items: list[T],
    predicate: Callable[[T], Awaitable[bool]],
    concurrency: int = 5,
) -> list[T]:
    """
    Filter items in parallel based on async predicate.
    """
    async def check_item(item: T) -> tuple[T, bool]:
        result = await predicate(item)
        return (item, result)

    results = await llm_batch(
        items,
        check_item,
        max_concurrent=concurrency,
    )

    return [item for item, passed in results if passed]


async def batch_first(
    items: list[T],
    predicate: Callable[[T], Awaitable[bool]],
    concurrency: int = 3,
) -> Optional[T]:
    """
    Find first item matching predicate in parallel.

    Uses early stopping to cancel remaining checks once found.
    """
    found: list[T] = []

    async def check_and_collect(item: T) -> Optional[T]:
        if await predicate(item):
            found.append(item)
            return item
        return None

    await llm_batch(
        items,
        check_and_collect,
        max_concurrent=concurrency,
        early_stop=lambda results: any(r is not None for r in results),
    )

    return found[0] if found else None


async def batch_race(
    callables: list[Callable[[], Awaitable[R]]],
    winner_predicate: Optional[Callable[[R], bool]] = None,
) -> Optional[R]:
    """
    Race multiple async operations, return first valid result.

    Useful for trying multiple strategies in parallel.

    Args:
        callables: List of async functions to race
        winner_predicate: Optional function to validate a result.
                         If None, first completed result wins.

    Returns:
        First valid result, or None if all failed.
    """
    winners: list[R] = []

    async def run_and_check(fn: Callable[[], Awaitable[R]]) -> Optional[R]:
        result = await fn()
        if winner_predicate is None or winner_predicate(result):
            winners.append(result)
            return result
        return None

    await llm_batch(
        callables,
        run_and_check,
        max_concurrent=len(callables),  # Race all at once
        early_stop=lambda results: any(r is not None for r in results),
    )

    return winners[0] if winners else None


__all__ = [
    # Core types
    "BatchConfig",
    "BatchItemStatus",
    "BatchItemResult",
    "BatchResult",
    # Main functions
    "llm_batch",
    "llm_batch_detailed",
    "llm_batch_with_progress",
    # Convenience functions
    "batch_map",
    "batch_filter",
    "batch_first",
    "batch_race",
]
