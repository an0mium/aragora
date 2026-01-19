"""
Tests for RLM batch parallelism utilities.

Tests the llm_batch() pattern from the Prime Intellect RLM paper.
"""

import asyncio
import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from aragora.rlm.batch import (
    BatchConfig,
    BatchItemStatus,
    BatchItemResult,
    BatchResult,
    llm_batch,
    llm_batch_detailed,
    llm_batch_with_progress,
    batch_map,
    batch_filter,
    batch_first,
    batch_race,
)


class TestBatchConfig:
    """Tests for BatchConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BatchConfig()

        assert config.max_concurrent == 5
        assert config.timeout_per_item == 60.0
        assert config.retry_on_error is False
        assert config.max_retries == 2
        assert config.retry_delay == 1.0
        assert config.fail_fast is False
        assert config.preserve_order is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BatchConfig(
            max_concurrent=10,
            timeout_per_item=30.0,
            retry_on_error=True,
            max_retries=3,
            fail_fast=True,
        )

        assert config.max_concurrent == 10
        assert config.timeout_per_item == 30.0
        assert config.retry_on_error is True
        assert config.max_retries == 3
        assert config.fail_fast is True


class TestBatchItemResult:
    """Tests for BatchItemResult dataclass."""

    def test_item_result_defaults(self):
        """Test default values for item result."""
        result = BatchItemResult(index=0, item="test")

        assert result.index == 0
        assert result.item == "test"
        assert result.result is None
        assert result.error is None
        assert result.status == BatchItemStatus.PENDING
        assert result.duration_seconds == 0.0
        assert result.attempts == 0

    def test_item_result_completed(self):
        """Test completed item result."""
        result = BatchItemResult(
            index=1,
            item="input",
            result="output",
            status=BatchItemStatus.COMPLETED,
            duration_seconds=0.5,
            attempts=1,
        )

        assert result.result == "output"
        assert result.status == BatchItemStatus.COMPLETED

    def test_item_result_failed(self):
        """Test failed item result."""
        error = ValueError("Test error")
        result = BatchItemResult(
            index=2,
            item="bad_input",
            error=error,
            status=BatchItemStatus.FAILED,
            attempts=3,
        )

        assert result.error == error
        assert result.status == BatchItemStatus.FAILED
        assert result.attempts == 3


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_empty_batch_result(self):
        """Test empty batch result."""
        result = BatchResult()

        assert result.items == []
        assert result.results == []
        assert result.all_succeeded is True  # Vacuously true
        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.errors == []

    def test_batch_result_with_items(self):
        """Test batch result with mixed items."""
        items = [
            BatchItemResult(
                index=0,
                item="a",
                result="A",
                status=BatchItemStatus.COMPLETED,
            ),
            BatchItemResult(
                index=1,
                item="b",
                error=ValueError("failed"),
                status=BatchItemStatus.FAILED,
            ),
            BatchItemResult(
                index=2,
                item="c",
                result="C",
                status=BatchItemStatus.COMPLETED,
            ),
        ]
        result = BatchResult(items=items)

        assert result.results == ["A", "C"]
        assert result.all_succeeded is False
        assert result.success_count == 2
        assert result.failure_count == 1
        assert len(result.errors) == 1
        assert result.errors[0][0] == 1

    def test_batch_result_early_stop(self):
        """Test batch result with early stopping."""
        items = [
            BatchItemResult(index=0, item="a", result="A", status=BatchItemStatus.COMPLETED),
            BatchItemResult(index=1, item="b", status=BatchItemStatus.CANCELLED),
        ]
        result = BatchResult(
            items=items,
            early_stopped=True,
            early_stop_at_index=0,
        )

        assert result.early_stopped is True
        assert result.early_stop_at_index == 0


class TestLlmBatch:
    """Tests for the main llm_batch function."""

    @pytest.mark.asyncio
    async def test_empty_items(self):
        """Test with empty input list."""
        async def process(x):
            return x * 2

        results = await llm_batch([], process)
        assert results == []

    @pytest.mark.asyncio
    async def test_single_item(self):
        """Test with single item."""
        async def process(x):
            return x * 2

        results = await llm_batch([5], process)
        assert results == [10]

    @pytest.mark.asyncio
    async def test_multiple_items(self):
        """Test with multiple items."""
        async def process(x):
            await asyncio.sleep(0.01)  # Simulate async work
            return x * 2

        results = await llm_batch([1, 2, 3, 4, 5], process)
        assert results == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_preserves_order(self):
        """Test that results are in input order."""
        call_order = []

        async def process(x):
            call_order.append(x)
            # Varying delays to test order preservation
            await asyncio.sleep(0.1 - x * 0.01)
            return x * 10

        results = await llm_batch([1, 2, 3, 4, 5], process, max_concurrent=5)

        # Results should be in original order
        assert results == [10, 20, 30, 40, 50]

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Test that concurrency is properly limited."""
        concurrent_count = 0
        max_seen = 0

        async def process(x):
            nonlocal concurrent_count, max_seen
            concurrent_count += 1
            max_seen = max(max_seen, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return x

        await llm_batch(
            list(range(10)),
            process,
            max_concurrent=3,
        )

        assert max_seen <= 3

    @pytest.mark.asyncio
    async def test_early_stop_triggered(self):
        """Test early stopping when condition is met."""
        processed = []

        async def process(x):
            processed.append(x)
            await asyncio.sleep(0.05)
            return x

        def should_stop(results):
            return len(results) >= 3

        results = await llm_batch(
            list(range(10)),
            process,
            max_concurrent=2,
            early_stop=should_stop,
        )

        # Should have at least 3 results (early stop condition)
        assert len(results) >= 3
        # Should have stopped early
        assert len(results) < 10

    @pytest.mark.asyncio
    async def test_early_stop_majority_vote(self):
        """Test early stopping for majority vote pattern."""
        votes = []

        async def cast_vote(voter_id):
            await asyncio.sleep(0.02)
            # First 3 voters vote "yes"
            vote = "yes" if voter_id < 3 else "no"
            votes.append(vote)
            return vote

        def has_majority(results):
            yes_count = sum(1 for v in results if v == "yes")
            return yes_count > 2  # Majority of 5

        results = await llm_batch(
            list(range(5)),
            cast_vote,
            max_concurrent=3,
            early_stop=has_majority,
        )

        # Should have enough "yes" votes for majority
        yes_count = sum(1 for v in results if v == "yes")
        assert yes_count >= 3

    @pytest.mark.asyncio
    async def test_with_errors(self):
        """Test handling of errors during processing."""
        async def process(x):
            if x == 2:
                raise ValueError("Error on 2")
            return x * 2

        # Default: errors don't stop other processing
        results = await llm_batch([1, 2, 3], process)

        # Should have results for non-failing items
        assert 2 in results  # 1 * 2
        assert 6 in results  # 3 * 2
        # Item 2 failed, so 4 is not in results

    @pytest.mark.asyncio
    async def test_with_config(self):
        """Test using BatchConfig."""
        async def process(x):
            await asyncio.sleep(0.01)
            return x + 100

        config = BatchConfig(max_concurrent=2)
        results = await llm_batch(
            [1, 2, 3],
            process,
            config=config,
        )

        assert results == [101, 102, 103]


class TestLlmBatchDetailed:
    """Tests for llm_batch_detailed function."""

    @pytest.mark.asyncio
    async def test_detailed_result(self):
        """Test detailed result structure."""
        async def process(x):
            await asyncio.sleep(0.01)
            return x * 2

        result = await llm_batch_detailed([1, 2, 3], process)

        assert isinstance(result, BatchResult)
        assert len(result.items) == 3
        assert result.all_succeeded
        assert result.total_duration_seconds > 0

    @pytest.mark.asyncio
    async def test_detailed_timing(self):
        """Test that timing information is captured."""
        async def process(x):
            await asyncio.sleep(0.05)
            return x

        result = await llm_batch_detailed([1], process)

        item = result.items[0]
        assert item.duration_seconds >= 0.04  # Allow some tolerance
        assert item.attempts == 1

    @pytest.mark.asyncio
    async def test_detailed_with_failure(self):
        """Test detailed result with failures."""
        async def process(x):
            if x == 2:
                raise ValueError("Failed")
            return x

        result = await llm_batch_detailed([1, 2, 3], process)

        assert result.success_count == 2
        assert result.failure_count == 1
        assert result.items[1].status == BatchItemStatus.FAILED
        assert result.items[1].error is not None

    @pytest.mark.asyncio
    async def test_detailed_early_stop_info(self):
        """Test early stop information in detailed result."""
        processed = []

        async def process(x):
            processed.append(x)
            await asyncio.sleep(0.02)
            return x

        result = await llm_batch_detailed(
            list(range(10)),
            process,
            early_stop=lambda r: len(r) >= 2,
            config=BatchConfig(max_concurrent=2),
        )

        assert result.early_stopped is True
        assert result.early_stop_at_index is not None

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Test timeout handling."""
        async def slow_process(x):
            await asyncio.sleep(1.0)  # Very slow
            return x

        config = BatchConfig(timeout_per_item=0.1)
        result = await llm_batch_detailed([1], slow_process, config=config)

        assert result.items[0].status == BatchItemStatus.TIMEOUT
        assert isinstance(result.items[0].error, TimeoutError)

    @pytest.mark.asyncio
    async def test_retry_on_error(self):
        """Test retry functionality."""
        attempts = 0

        async def flaky_process(x):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("Retry me")
            return x

        config = BatchConfig(
            retry_on_error=True,
            max_retries=3,
            retry_delay=0.01,
        )
        result = await llm_batch_detailed([1], flaky_process, config=config)

        assert result.items[0].status == BatchItemStatus.COMPLETED
        assert result.items[0].attempts >= 3

    @pytest.mark.asyncio
    async def test_fail_fast(self):
        """Test fail-fast mode."""
        processed = []

        async def process(x):
            processed.append(x)
            await asyncio.sleep(0.05)
            if x == 0:
                raise ValueError("First failed")
            return x

        config = BatchConfig(fail_fast=True, max_concurrent=1)
        result = await llm_batch_detailed(
            [0, 1, 2],
            process,
            config=config,
        )

        # First item failed, others should be cancelled
        assert result.items[0].status == BatchItemStatus.FAILED


class TestLlmBatchWithProgress:
    """Tests for llm_batch_with_progress function."""

    @pytest.mark.asyncio
    async def test_progress_callback(self):
        """Test progress callback is called."""
        progress_calls = []

        def on_progress(completed, total, item_result):
            progress_calls.append((completed, total, item_result.index))

        async def process(x):
            await asyncio.sleep(0.01)
            return x

        result = await llm_batch_with_progress(
            [1, 2, 3],
            process,
            on_progress,
        )

        assert len(progress_calls) == 3
        assert all(total == 3 for _, total, _ in progress_calls)


class TestBatchMap:
    """Tests for batch_map convenience function."""

    @pytest.mark.asyncio
    async def test_batch_map(self):
        """Test batch_map for simple mapping."""
        async def double(x):
            return x * 2

        results = await batch_map([1, 2, 3], double, concurrency=2)
        assert results == [2, 4, 6]


class TestBatchFilter:
    """Tests for batch_filter convenience function."""

    @pytest.mark.asyncio
    async def test_batch_filter(self):
        """Test batch_filter for filtering."""
        async def is_even(x):
            return x % 2 == 0

        results = await batch_filter([1, 2, 3, 4, 5], is_even, concurrency=3)
        assert results == [2, 4]


class TestBatchFirst:
    """Tests for batch_first convenience function."""

    @pytest.mark.asyncio
    async def test_batch_first_found(self):
        """Test batch_first when match is found."""
        async def is_greater_than_five(x):
            await asyncio.sleep(0.01)
            return x > 5

        result = await batch_first([1, 2, 7, 3, 8], is_greater_than_five)
        assert result in [7, 8]  # Either could be found first

    @pytest.mark.asyncio
    async def test_batch_first_not_found(self):
        """Test batch_first when no match is found."""
        async def is_negative(x):
            return x < 0

        result = await batch_first([1, 2, 3], is_negative)
        assert result is None


class TestBatchRace:
    """Tests for batch_race convenience function."""

    @pytest.mark.asyncio
    async def test_batch_race_first_wins(self):
        """Test batch_race returns first completed result."""
        async def fast():
            await asyncio.sleep(0.01)
            return "fast"

        async def slow():
            await asyncio.sleep(0.5)
            return "slow"

        result = await batch_race([fast, slow])
        assert result == "fast"

    @pytest.mark.asyncio
    async def test_batch_race_with_predicate(self):
        """Test batch_race with winner predicate."""
        async def returns_none():
            await asyncio.sleep(0.01)
            return None

        async def returns_value():
            await asyncio.sleep(0.02)
            return "value"

        result = await batch_race(
            [returns_none, returns_value],
            winner_predicate=lambda r: r is not None,
        )
        assert result == "value"


class TestRealWorldPatterns:
    """Tests for real-world usage patterns."""

    @pytest.mark.asyncio
    async def test_parallel_critique_generation(self):
        """Test pattern: generating critiques for multiple proposals."""
        proposals = ["Proposal A", "Proposal B", "Proposal C"]

        async def generate_critique(proposal):
            await asyncio.sleep(0.02)
            return f"Critique of {proposal}"

        critiques = await llm_batch(
            proposals,
            generate_critique,
            max_concurrent=3,
        )

        assert len(critiques) == 3
        assert all("Critique of" in c for c in critiques)

    @pytest.mark.asyncio
    async def test_voting_with_early_majority(self):
        """Test pattern: voting with early majority detection."""
        voters = list(range(7))  # 7 voters, need 4 for majority
        vote_results = []

        async def cast_vote(voter_id):
            await asyncio.sleep(0.02)
            vote = "yes" if voter_id % 2 == 0 else "no"  # 4 yes, 3 no
            vote_results.append((voter_id, vote))
            return vote

        def majority_reached(votes):
            yes_count = sum(1 for v in votes if v == "yes")
            no_count = sum(1 for v in votes if v == "no")
            return yes_count >= 4 or no_count >= 4

        votes = await llm_batch(
            voters,
            cast_vote,
            max_concurrent=3,
            early_stop=majority_reached,
        )

        # Should have reached majority
        yes_count = sum(1 for v in votes if v == "yes")
        no_count = sum(1 for v in votes if v == "no")
        assert yes_count >= 4 or no_count >= 4

    @pytest.mark.asyncio
    async def test_strategy_racing(self):
        """Test pattern: racing multiple strategies."""
        async def strategy_a():
            await asyncio.sleep(0.05)
            return {"confidence": 0.7, "answer": "A"}

        async def strategy_b():
            await asyncio.sleep(0.03)
            return {"confidence": 0.9, "answer": "B"}

        async def strategy_c():
            await asyncio.sleep(0.04)
            return {"confidence": 0.6, "answer": "C"}

        result = await batch_race(
            [strategy_a, strategy_b, strategy_c],
            winner_predicate=lambda r: r["confidence"] > 0.8,
        )

        # Strategy B should win (high confidence)
        assert result is not None
        assert result["confidence"] > 0.8

    @pytest.mark.asyncio
    async def test_parallel_agent_health_check(self):
        """Test pattern: checking multiple agents in parallel."""
        agents = ["agent_1", "agent_2", "agent_3", "agent_4"]

        async def check_health(agent_id):
            await asyncio.sleep(0.02)
            # Simulate some agents being unhealthy
            healthy = agent_id != "agent_2"
            return {"agent": agent_id, "healthy": healthy}

        results = await batch_map(agents, check_health, concurrency=4)

        healthy_agents = [r["agent"] for r in results if r["healthy"]]
        assert len(healthy_agents) == 3
        assert "agent_2" not in healthy_agents


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_all_items_fail(self):
        """Test when all items fail."""
        async def always_fail(x):
            raise ValueError(f"Failed on {x}")

        results = await llm_batch([1, 2, 3], always_fail)
        assert results == []

    @pytest.mark.asyncio
    async def test_none_results(self):
        """Test handling of None results."""
        async def maybe_none(x):
            return None if x % 2 == 0 else x

        results = await llm_batch([1, 2, 3], maybe_none)
        # None results are filtered out by the results property
        assert results == [1, 3]
        # Use detailed to see all results including None
        detailed = await llm_batch_detailed([1, 2, 3], maybe_none)
        assert detailed.items[1].result is None
        assert detailed.items[1].status == BatchItemStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_large_batch(self):
        """Test with larger batch size."""
        async def process(x):
            await asyncio.sleep(0.001)
            return x * 2

        items = list(range(100))
        results = await llm_batch(items, process, max_concurrent=10)

        assert len(results) == 100
        assert results == [x * 2 for x in items]

    @pytest.mark.asyncio
    async def test_zero_timeout(self):
        """Test with very short timeout."""
        async def slow(x):
            await asyncio.sleep(0.5)
            return x

        config = BatchConfig(timeout_per_item=0.001)
        result = await llm_batch_detailed([1], slow, config=config)

        assert result.items[0].status in [
            BatchItemStatus.TIMEOUT,
            BatchItemStatus.FAILED,
        ]

    @pytest.mark.asyncio
    async def test_immediate_early_stop(self):
        """Test early stop on first result."""
        processed = []

        async def process(x):
            processed.append(x)
            await asyncio.sleep(0.05)
            return x

        results = await llm_batch(
            list(range(10)),
            process,
            max_concurrent=1,
            early_stop=lambda r: len(r) >= 1,  # Stop after first
        )

        assert len(results) >= 1
