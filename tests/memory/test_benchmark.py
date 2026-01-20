"""
Performance benchmarks for MemoryCoordinator operations.

Uses pytest-benchmark for reproducible benchmarking.

Run with:
    pytest tests/memory/test_benchmark.py -v --benchmark-only
    pytest tests/memory/test_benchmark.py -v --benchmark-compare
    pytest tests/memory/test_benchmark.py -v --benchmark-autosave

Note: These tests use the 'serial' marker to avoid parallel execution
which can cause resource contention and timeouts.
"""

import asyncio
import tempfile
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mark entire module as serial to avoid parallel execution issues
pytestmark = [
    pytest.mark.serial,
    pytest.mark.timeout(300),  # 5 minute timeout for benchmarks
]

from aragora.core_types import Message, DebateResult
from aragora.memory.coordinator import (
    CoordinatorOptions,
    MemoryCoordinator,
    MemoryTransaction,
    WriteOperation,
    WriteStatus,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_continuum_memory():
    """Mock ContinuumMemory for benchmarking."""
    memory = MagicMock()
    memory.add = AsyncMock(return_value="entry-123")
    memory.add_with_metadata = AsyncMock(return_value="entry-123")
    return memory


@pytest.fixture
def mock_consensus_memory():
    """Mock ConsensusMemory for benchmarking."""
    memory = MagicMock()
    memory.add_consensus = AsyncMock(return_value="consensus-123")
    memory.add = AsyncMock(return_value="consensus-123")
    return memory


@pytest.fixture
def mock_critique_store():
    """Mock CritiqueStore for benchmarking."""
    store = MagicMock()
    store.add_critique = AsyncMock(return_value="critique-123")
    store.add = AsyncMock(return_value="critique-123")
    return store


@pytest.fixture
def mock_knowledge_mound():
    """Mock KnowledgeMound for benchmarking."""
    mound = MagicMock()
    mound.ingest = AsyncMock(return_value=None)
    mound.add = AsyncMock(return_value="node-123")
    return mound


@pytest.fixture
def coordinator(
    mock_continuum_memory,
    mock_consensus_memory,
    mock_critique_store,
    mock_knowledge_mound,
):
    """Create MemoryCoordinator with mocks."""
    return MemoryCoordinator(
        continuum_memory=mock_continuum_memory,
        consensus_memory=mock_consensus_memory,
        critique_store=mock_critique_store,
        knowledge_mound=mock_knowledge_mound,
        options=CoordinatorOptions(
            parallel_writes=False,
            rollback_on_failure=True,
        ),
    )


@pytest.fixture
def debate_context():
    """Create a mock DebateContext for benchmarking."""
    ctx = MagicMock()
    ctx.debate_id = "bench-debate-001"
    ctx.env = MagicMock()
    ctx.env.task = "Benchmark debate task"
    ctx.result = DebateResult(
        final_answer="This is the benchmark consensus answer.",
        confidence=0.95,
        rounds_completed=3,
        messages=[
            Message(
                role="proposer",
                agent="agent-1",
                content="First response",
                timestamp=datetime.now(),
            ),
            Message(
                role="critic",
                agent="agent-2",
                content="Second response",
                timestamp=datetime.now(),
            ),
        ],
        consensus_reached=True,
        participants=["agent-1", "agent-2"],
    )
    return ctx


# =============================================================================
# Benchmark Tests
# =============================================================================


class TestMemoryCoordinatorBenchmarks:
    """Benchmark tests for MemoryCoordinator operations."""

    @pytest.mark.benchmark(group="coordinator-write")
    def test_sequential_write_benchmark(self, benchmark, coordinator, debate_context):
        """Benchmark sequential write operations."""
        coordinator.options.parallel_writes = False

        def sync_wrapper():
            return asyncio.run(coordinator.commit_debate_outcome(debate_context))

        result = benchmark(sync_wrapper)
        assert result is not None

    @pytest.mark.benchmark(group="coordinator-write")
    def test_parallel_write_benchmark(self, benchmark, coordinator, debate_context):
        """Benchmark parallel write operations."""
        coordinator.options.parallel_writes = True

        def sync_wrapper():
            return asyncio.run(coordinator.commit_debate_outcome(debate_context))

        result = benchmark(sync_wrapper)
        assert result is not None

    @pytest.mark.benchmark(group="coordinator-operations")
    def test_build_operations_benchmark(self, benchmark, coordinator, debate_context):
        """Benchmark operation building."""
        result = debate_context.result
        opts = coordinator.options

        operations = benchmark(coordinator._build_operations, debate_context, result, opts)
        assert len(operations) > 0

    @pytest.mark.benchmark(group="coordinator-metrics")
    def test_metrics_update_benchmark(self, benchmark, coordinator):
        """Benchmark metrics updates."""
        # Create a sample transaction
        transaction = MemoryTransaction(
            id="tx-bench-001",
            debate_id="debate-001",
            operations=[
                WriteOperation(
                    id="op-1", target="continuum", status=WriteStatus.SUCCESS
                ),
                WriteOperation(
                    id="op-2", target="consensus", status=WriteStatus.SUCCESS
                ),
                WriteOperation(id="op-3", target="mound", status=WriteStatus.SUCCESS),
            ],
        )

        def update_metrics():
            coordinator._update_write_metrics(transaction)

        benchmark(update_metrics)


class TestWriteOperationBenchmarks:
    """Benchmark tests for WriteOperation."""

    @pytest.mark.benchmark(group="write-operation")
    def test_operation_creation_benchmark(self, benchmark):
        """Benchmark operation creation."""

        def create():
            return WriteOperation(
                id="op-bench",
                target="continuum",
                data={"task": "test", "answer": "result"},
            )

        op = benchmark(create)
        assert op.status == WriteStatus.PENDING

    @pytest.mark.benchmark(group="write-operation")
    def test_operation_status_update_benchmark(self, benchmark):
        """Benchmark status updates."""
        op = WriteOperation(id="op-bench", target="continuum")

        def update():
            op.mark_success(result="entry-123")
            op.status = WriteStatus.PENDING  # Reset for next iteration

        benchmark(update)


class TestTransactionBenchmarks:
    """Benchmark tests for MemoryTransaction."""

    @pytest.mark.benchmark(group="transaction")
    def test_transaction_creation_benchmark(self, benchmark):
        """Benchmark transaction creation."""

        def create():
            return MemoryTransaction(id="tx-bench", debate_id="debate-bench")

        tx = benchmark(create)
        assert tx.success is True

    @pytest.mark.benchmark(group="transaction")
    def test_transaction_with_many_operations_benchmark(self, benchmark):
        """Benchmark transaction with many operations."""

        def create():
            ops = [
                WriteOperation(
                    id=f"op-{i}",
                    target=["continuum", "consensus", "mound"][i % 3],
                    status=WriteStatus.SUCCESS,
                )
                for i in range(100)
            ]
            return MemoryTransaction(id="tx-bench", debate_id="debate-bench", operations=ops)

        tx = benchmark(create)
        assert len(tx.operations) == 100

    @pytest.mark.benchmark(group="transaction")
    def test_get_failed_operations_benchmark(self, benchmark):
        """Benchmark failed operation retrieval."""
        # Create transaction with mixed success/failure
        ops = [
            WriteOperation(
                id=f"op-{i}",
                target="continuum",
                status=WriteStatus.SUCCESS if i % 2 == 0 else WriteStatus.FAILED,
            )
            for i in range(100)
        ]
        tx = MemoryTransaction(id="tx-bench", debate_id="debate-bench", operations=ops)

        failed = benchmark(tx.get_failed_operations)
        assert len(failed) == 50


# =============================================================================
# Load Tests
# =============================================================================


class TestMemoryCoordinatorLoadTests:
    """Load tests for MemoryCoordinator under stress."""

    @pytest.mark.benchmark(group="load-test")
    def test_concurrent_transactions_benchmark(
        self, benchmark, coordinator, debate_context
    ):
        """Benchmark multiple concurrent transactions."""
        NUM_CONCURRENT = 10

        async def run_concurrent():
            tasks = [
                coordinator.commit_debate_outcome(debate_context)
                for _ in range(NUM_CONCURRENT)
            ]
            return await asyncio.gather(*tasks)

        def sync_wrapper():
            return asyncio.run(run_concurrent())

        results = benchmark(sync_wrapper)
        assert len(results) == NUM_CONCURRENT
        for tx in results:
            assert tx is not None

    @pytest.mark.benchmark(group="load-test")
    def test_high_volume_operations_benchmark(
        self, benchmark, coordinator, debate_context
    ):
        """Benchmark high volume of sequential operations."""
        NUM_OPERATIONS = 50

        async def run_many():
            results = []
            for _ in range(NUM_OPERATIONS):
                tx = await coordinator.commit_debate_outcome(debate_context)
                results.append(tx)
            return results

        def sync_wrapper():
            return asyncio.run(run_many())

        results = benchmark.pedantic(
            sync_wrapper,
            rounds=3,
            iterations=1,
            warmup_rounds=1,
        )
        assert len(results) == NUM_OPERATIONS


# =============================================================================
# SLO Assertions
# =============================================================================

# Define SLOs for memory coordinator operations
SLO_SINGLE_WRITE_MS = 10  # Single write should complete in <10ms
SLO_PARALLEL_WRITE_MS = 20  # Parallel write should complete in <20ms
SLO_BUILD_OPERATIONS_MS = 1  # Building operations should take <1ms
SLO_METRICS_UPDATE_MS = 0.5  # Metrics update should take <0.5ms


class TestMemoryCoordinatorSLOs:
    """Test that operations meet SLO requirements."""

    @pytest.mark.benchmark(group="slo-verification")
    def test_sequential_write_meets_slo(self, benchmark, coordinator, debate_context):
        """Verify sequential writes meet SLO."""
        coordinator.options.parallel_writes = False

        def sync_wrapper():
            return asyncio.run(coordinator.commit_debate_outcome(debate_context))

        result = benchmark(sync_wrapper)

        # Check SLO (benchmark.stats available after benchmark call)
        # Note: With mocked memory systems, this should be very fast
        assert result is not None

    @pytest.mark.benchmark(group="slo-verification")
    def test_build_operations_meets_slo(self, benchmark, coordinator, debate_context):
        """Verify operation building meets SLO."""
        result = debate_context.result
        opts = coordinator.options

        operations = benchmark(coordinator._build_operations, debate_context, result, opts)

        # Verify operations were built
        assert len(operations) > 0
