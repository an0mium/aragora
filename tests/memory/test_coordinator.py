"""Tests for MemoryCoordinator - cross-system atomic writes."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.memory.coordinator import (
    CoordinatorMetrics,
    CoordinatorOptions,
    MemoryCoordinator,
    MemoryTransaction,
    WriteOperation,
    WriteStatus,
)


class TestWriteOperation:
    """Tests for WriteOperation dataclass."""

    def test_initial_status(self):
        """Test initial status is pending."""
        op = WriteOperation(id="op-1", target="continuum")
        assert op.status == WriteStatus.PENDING
        assert op.result is None
        assert op.error is None

    def test_mark_success(self):
        """Test marking operation as successful."""
        op = WriteOperation(id="op-1", target="continuum")
        op.mark_success(result="entry-123")

        assert op.status == WriteStatus.SUCCESS
        assert op.result == "entry-123"
        assert op.error is None

    def test_mark_failed(self):
        """Test marking operation as failed."""
        op = WriteOperation(id="op-1", target="continuum")
        op.mark_failed("Connection timeout")

        assert op.status == WriteStatus.FAILED
        assert op.result is None
        assert op.error == "Connection timeout"


class TestMemoryTransaction:
    """Tests for MemoryTransaction dataclass."""

    def test_empty_transaction_success(self):
        """Test empty transaction is considered successful."""
        tx = MemoryTransaction(id="tx-1", debate_id="debate-1")
        assert tx.success is True
        assert tx.partial_failure is False

    def test_all_successful(self):
        """Test transaction with all successful ops."""
        tx = MemoryTransaction(
            id="tx-1",
            debate_id="debate-1",
            operations=[
                WriteOperation(id="op-1", target="continuum", status=WriteStatus.SUCCESS),
                WriteOperation(id="op-2", target="consensus", status=WriteStatus.SUCCESS),
            ],
        )
        assert tx.success is True
        assert tx.partial_failure is False

    def test_partial_failure(self):
        """Test transaction with partial failure."""
        tx = MemoryTransaction(
            id="tx-1",
            debate_id="debate-1",
            operations=[
                WriteOperation(id="op-1", target="continuum", status=WriteStatus.SUCCESS),
                WriteOperation(id="op-2", target="consensus", status=WriteStatus.FAILED),
            ],
        )
        assert tx.success is False
        assert tx.partial_failure is True

    def test_get_failed_operations(self):
        """Test getting failed operations."""
        tx = MemoryTransaction(
            id="tx-1",
            debate_id="debate-1",
            operations=[
                WriteOperation(
                    id="op-1", target="continuum", status=WriteStatus.SUCCESS
                ),
                WriteOperation(
                    id="op-2", target="consensus", status=WriteStatus.FAILED, error="timeout"
                ),
                WriteOperation(
                    id="op-3", target="mound", status=WriteStatus.FAILED, error="permission"
                ),
            ],
        )
        failed = tx.get_failed_operations()
        assert len(failed) == 2
        assert failed[0].target == "consensus"
        assert failed[1].target == "mound"


class TestCoordinatorOptions:
    """Tests for CoordinatorOptions configuration."""

    def test_defaults(self):
        """Test default configuration values."""
        opts = CoordinatorOptions()

        assert opts.write_continuum is True
        assert opts.write_consensus is True
        assert opts.write_critique is True
        assert opts.write_mound is True
        assert opts.rollback_on_failure is True
        assert opts.parallel_writes is False
        assert opts.min_confidence_for_mound == 0.7

    def test_custom_options(self):
        """Test custom configuration."""
        opts = CoordinatorOptions(
            write_mound=False,
            parallel_writes=True,
            min_confidence_for_mound=0.9,
        )

        assert opts.write_mound is False
        assert opts.parallel_writes is True
        assert opts.min_confidence_for_mound == 0.9


class TestMemoryCoordinator:
    """Tests for MemoryCoordinator class."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock DebateContext."""
        ctx = MagicMock()
        ctx.debate_id = "debate-123"
        ctx.domain = "security"
        ctx.env.task = "Analyze security vulnerabilities"
        ctx.agents = [
            MagicMock(name="claude"),
            MagicMock(name="gpt-4"),
        ]

        # Mock result
        ctx.result = MagicMock()
        ctx.result.final_answer = "Security analysis complete"
        ctx.result.confidence = 0.85
        ctx.result.consensus_reached = True
        ctx.result.winner = "claude"
        ctx.result.rounds_used = 3

        return ctx

    @pytest.fixture
    def mock_memory_systems(self):
        """Create mock memory system instances."""
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="continuum-entry-1")

        consensus = MagicMock()
        consensus.store_consensus = MagicMock(return_value="consensus-entry-1")

        critique = MagicMock()
        critique.store_result = MagicMock()

        mound = AsyncMock()
        mound.ingest_debate_outcome = AsyncMock(return_value="mound-item-1")

        return continuum, consensus, critique, mound

    @pytest.mark.asyncio
    async def test_commit_without_result(self, mock_context):
        """Test commit returns empty transaction without result."""
        mock_context.result = None
        coordinator = MemoryCoordinator()

        tx = await coordinator.commit_debate_outcome(mock_context)

        assert len(tx.operations) == 0

    @pytest.mark.asyncio
    async def test_commit_sequential_writes(self, mock_context, mock_memory_systems):
        """Test sequential write execution."""
        continuum, consensus, critique, mound = mock_memory_systems

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
            critique_store=critique,
            knowledge_mound=mound,
        )

        tx = await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(parallel_writes=False),
        )

        assert tx.success is True
        assert len(tx.operations) == 4
        assert coordinator.metrics.successful_transactions == 1

    @pytest.mark.asyncio
    async def test_commit_parallel_writes(self, mock_context, mock_memory_systems):
        """Test parallel write execution."""
        continuum, consensus, critique, mound = mock_memory_systems

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
            critique_store=critique,
            knowledge_mound=mound,
        )

        tx = await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(parallel_writes=True),
        )

        assert tx.success is True
        assert len(tx.operations) == 4

    @pytest.mark.asyncio
    async def test_skip_mound_below_confidence(self, mock_context, mock_memory_systems):
        """Test mound write is skipped for low confidence."""
        mock_context.result.confidence = 0.5  # Below default 0.7 threshold
        continuum, consensus, critique, mound = mock_memory_systems

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
            critique_store=critique,
            knowledge_mound=mound,
        )

        tx = await coordinator.commit_debate_outcome(mock_context)

        # Should have 3 operations (no mound)
        targets = [op.target for op in tx.operations]
        assert "mound" not in targets
        assert len(tx.operations) == 3

    @pytest.mark.asyncio
    async def test_partial_failure_metrics(self, mock_context):
        """Test metrics are updated on partial failure."""
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="ok")

        consensus = MagicMock()
        consensus.store_consensus = MagicMock(side_effect=Exception("DB error"))

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
            options=CoordinatorOptions(rollback_on_failure=False),
        )

        tx = await coordinator.commit_debate_outcome(mock_context)

        assert tx.partial_failure is True
        assert coordinator.metrics.partial_failures == 1

    @pytest.mark.asyncio
    async def test_selective_writes(self, mock_context):
        """Test selective writes via options."""
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="ok")

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
        )

        tx = await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(
                write_continuum=True,
                write_consensus=False,
                write_critique=False,
                write_mound=False,
            ),
        )

        assert len(tx.operations) == 1
        assert tx.operations[0].target == "continuum"

    def test_get_metrics(self):
        """Test getting coordinator metrics."""
        coordinator = MemoryCoordinator()
        coordinator.metrics.total_transactions = 10
        coordinator.metrics.successful_transactions = 8
        coordinator.metrics.partial_failures = 2

        metrics = coordinator.get_metrics()

        assert metrics["total_transactions"] == 10
        assert metrics["success_rate"] == 0.8


class TestCoordinatorRollback:
    """Tests for rollback functionality."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock DebateContext."""
        ctx = MagicMock()
        ctx.debate_id = "debate-123"
        ctx.domain = "testing"
        ctx.env.task = "Test task"
        ctx.agents = [MagicMock(name="agent1")]
        ctx.result = MagicMock()
        ctx.result.final_answer = "Answer"
        ctx.result.confidence = 0.9
        ctx.result.consensus_reached = True
        ctx.result.winner = "agent1"
        ctx.result.rounds_used = 2
        return ctx

    @pytest.mark.asyncio
    async def test_rollback_on_partial_failure(self, mock_context):
        """Test rollback is called on partial failure."""
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="ok")

        consensus = MagicMock()
        consensus.store_consensus = MagicMock(side_effect=Exception("Error"))

        rollback_called = []

        async def rollback_handler(op):
            rollback_called.append(op.target)
            return True

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
        )
        coordinator.register_rollback_handler("continuum", rollback_handler)

        tx = await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(
                rollback_on_failure=True,
                write_critique=False,
                write_mound=False,
            ),
        )

        assert tx.rolled_back is True
        assert "continuum" in rollback_called
        assert coordinator.metrics.rollbacks_performed == 1
