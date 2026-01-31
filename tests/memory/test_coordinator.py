"""Tests for MemoryCoordinator - cross-system atomic writes.

Comprehensive test coverage for atomic write coordination across multiple
memory systems (ContinuumMemory, ConsensusMemory, CritiqueStore, KnowledgeMound).

Tests include:
- Initialization and configuration
- Single-target writes
- Multi-target atomic writes
- Rollback on partial failure
- Parallel vs sequential execution
- Transaction semantics
- State consistency after failures
- Event emission and callbacks
- Retry behavior
- Metrics tracking
"""

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


# =============================================================================
# Test WriteOperation
# =============================================================================


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

    def test_mark_success_without_result(self):
        """Test marking operation as successful without explicit result."""
        op = WriteOperation(id="op-1", target="critique")
        op.mark_success()

        assert op.status == WriteStatus.SUCCESS
        assert op.result is None

    def test_mark_failed(self):
        """Test marking operation as failed."""
        op = WriteOperation(id="op-1", target="continuum")
        op.mark_failed("Connection timeout")

        assert op.status == WriteStatus.FAILED
        assert op.result is None
        assert op.error == "Connection timeout"

    def test_operation_with_data(self):
        """Test operation initialized with data payload."""
        data = {"debate_id": "d123", "task": "Test task"}
        op = WriteOperation(id="op-1", target="continuum", data=data)

        assert op.data == data
        assert op.data["debate_id"] == "d123"

    def test_timestamp_is_set(self):
        """Test timestamp is automatically set on creation."""
        op = WriteOperation(id="op-1", target="consensus")
        assert isinstance(op.timestamp, datetime)


# =============================================================================
# Test MemoryTransaction
# =============================================================================


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

    def test_all_failed(self):
        """Test transaction where all operations failed."""
        tx = MemoryTransaction(
            id="tx-1",
            debate_id="debate-1",
            operations=[
                WriteOperation(id="op-1", target="continuum", status=WriteStatus.FAILED),
                WriteOperation(id="op-2", target="consensus", status=WriteStatus.FAILED),
            ],
        )
        assert tx.success is False
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
                WriteOperation(id="op-1", target="continuum", status=WriteStatus.SUCCESS),
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

    def test_get_successful_operations(self):
        """Test getting successful operations."""
        tx = MemoryTransaction(
            id="tx-1",
            debate_id="debate-1",
            operations=[
                WriteOperation(id="op-1", target="continuum", status=WriteStatus.SUCCESS),
                WriteOperation(id="op-2", target="consensus", status=WriteStatus.FAILED),
                WriteOperation(id="op-3", target="critique", status=WriteStatus.SUCCESS),
            ],
        )
        successful = tx.get_successful_operations()
        assert len(successful) == 2
        assert successful[0].target == "continuum"
        assert successful[1].target == "critique"

    def test_transaction_rolled_back_flag(self):
        """Test rolled_back flag is initially False."""
        tx = MemoryTransaction(id="tx-1", debate_id="debate-1")
        assert tx.rolled_back is False

    def test_transaction_timestamps(self):
        """Test transaction timestamps are tracked."""
        tx = MemoryTransaction(id="tx-1", debate_id="debate-1")
        assert isinstance(tx.started_at, datetime)
        assert tx.completed_at is None


# =============================================================================
# Test CoordinatorOptions
# =============================================================================


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
        assert opts.timeout_seconds == 30.0
        assert opts.max_retries == 2
        assert opts.retry_delay_seconds == 0.5

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

    def test_disable_rollback(self):
        """Test disabling rollback on failure."""
        opts = CoordinatorOptions(rollback_on_failure=False)
        assert opts.rollback_on_failure is False

    def test_custom_retry_settings(self):
        """Test custom retry configuration."""
        opts = CoordinatorOptions(
            max_retries=5,
            retry_delay_seconds=1.0,
        )
        assert opts.max_retries == 5
        assert opts.retry_delay_seconds == 1.0


# =============================================================================
# Test CoordinatorMetrics
# =============================================================================


class TestCoordinatorMetrics:
    """Tests for CoordinatorMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = CoordinatorMetrics()

        assert metrics.total_transactions == 0
        assert metrics.successful_transactions == 0
        assert metrics.partial_failures == 0
        assert metrics.full_failures == 0
        assert metrics.rollbacks_performed == 0
        assert metrics.total_writes == 0
        assert metrics.writes_per_target == {}

    def test_metrics_increment(self):
        """Test metrics can be incremented."""
        metrics = CoordinatorMetrics()
        metrics.total_transactions += 1
        metrics.successful_transactions += 1
        metrics.writes_per_target["continuum"] = 1

        assert metrics.total_transactions == 1
        assert metrics.successful_transactions == 1
        assert metrics.writes_per_target["continuum"] == 1


# =============================================================================
# Test MemoryCoordinator Initialization
# =============================================================================


class TestMemoryCoordinatorInitialization:
    """Tests for MemoryCoordinator initialization."""

    def test_init_without_memory_systems(self):
        """Test initialization without any memory systems."""
        coordinator = MemoryCoordinator()

        assert coordinator.continuum_memory is None
        assert coordinator.consensus_memory is None
        assert coordinator.critique_store is None
        assert coordinator.knowledge_mound is None
        assert isinstance(coordinator.options, CoordinatorOptions)
        assert isinstance(coordinator.metrics, CoordinatorMetrics)

    def test_init_with_custom_options(self):
        """Test initialization with custom options."""
        opts = CoordinatorOptions(parallel_writes=True)
        coordinator = MemoryCoordinator(options=opts)

        assert coordinator.options.parallel_writes is True

    def test_init_with_single_memory_system(self):
        """Test initialization with only one memory system."""
        continuum = MagicMock()
        coordinator = MemoryCoordinator(continuum_memory=continuum)

        assert coordinator.continuum_memory is continuum
        assert coordinator.consensus_memory is None
        assert "continuum" in coordinator._rollback_handlers

    def test_init_registers_rollback_handlers(self):
        """Test that rollback handlers are registered for available systems."""
        continuum = MagicMock()
        consensus = MagicMock()
        critique = MagicMock()
        mound = AsyncMock()

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
            critique_store=critique,
            knowledge_mound=mound,
        )

        assert "continuum" in coordinator._rollback_handlers
        assert "consensus" in coordinator._rollback_handlers
        assert "critique" in coordinator._rollback_handlers
        assert "mound" in coordinator._rollback_handlers

    def test_init_no_handlers_for_missing_systems(self):
        """Test no rollback handlers registered for missing systems."""
        continuum = MagicMock()
        coordinator = MemoryCoordinator(continuum_memory=continuum)

        assert "continuum" in coordinator._rollback_handlers
        assert "consensus" not in coordinator._rollback_handlers
        assert "critique" not in coordinator._rollback_handlers
        assert "mound" not in coordinator._rollback_handlers


# =============================================================================
# Test Single-Target Writes
# =============================================================================


class TestSingleTargetWrites:
    """Tests for single-target write operations."""

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
        ctx.result = MagicMock()
        ctx.result.final_answer = "Security analysis complete"
        ctx.result.confidence = 0.85
        ctx.result.consensus_reached = True
        ctx.result.winner = "claude"
        ctx.result.rounds_used = 3
        return ctx

    @pytest.mark.asyncio
    async def test_write_to_continuum_only(self, mock_context):
        """Test writing only to continuum memory."""
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="entry-1")

        coordinator = MemoryCoordinator(continuum_memory=continuum)

        tx = await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(
                write_continuum=True,
                write_consensus=False,
                write_critique=False,
                write_mound=False,
            ),
        )

        assert tx.success is True
        assert len(tx.operations) == 1
        assert tx.operations[0].target == "continuum"
        continuum.store_pattern.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_to_consensus_only(self, mock_context):
        """Test writing only to consensus memory."""
        consensus = MagicMock()
        mock_record = MagicMock()
        mock_record.id = "consensus-1"
        consensus.store_consensus = MagicMock(return_value=mock_record)

        coordinator = MemoryCoordinator(consensus_memory=consensus)

        tx = await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(
                write_continuum=False,
                write_consensus=True,
                write_critique=False,
                write_mound=False,
            ),
        )

        assert tx.success is True
        assert len(tx.operations) == 1
        assert tx.operations[0].target == "consensus"
        consensus.store_consensus.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_to_critique_only(self, mock_context):
        """Test writing only to critique store."""
        critique = MagicMock()
        critique.store_result = MagicMock()

        coordinator = MemoryCoordinator(critique_store=critique)

        tx = await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(
                write_continuum=False,
                write_consensus=False,
                write_critique=True,
                write_mound=False,
            ),
        )

        assert tx.success is True
        assert len(tx.operations) == 1
        assert tx.operations[0].target == "critique"
        critique.store_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_to_mound_only(self, mock_context):
        """Test writing only to knowledge mound."""
        mound = AsyncMock()
        mound.ingest_debate_outcome = AsyncMock(return_value="mound-1")

        coordinator = MemoryCoordinator(knowledge_mound=mound)

        tx = await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(
                write_continuum=False,
                write_consensus=False,
                write_critique=False,
                write_mound=True,
            ),
        )

        assert tx.success is True
        assert len(tx.operations) == 1
        assert tx.operations[0].target == "mound"
        mound.ingest_debate_outcome.assert_called_once()


# =============================================================================
# Test Multi-Target Atomic Writes
# =============================================================================


class TestMultiTargetWrites:
    """Tests for multi-target atomic write operations."""

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
        mock_consensus_record = MagicMock()
        mock_consensus_record.id = "consensus-entry-1"
        consensus.store_consensus = MagicMock(return_value=mock_consensus_record)

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
    async def test_mound_write_at_threshold(self, mock_context, mock_memory_systems):
        """Test mound write at exactly the confidence threshold."""
        mock_context.result.confidence = 0.7  # Exactly at threshold
        continuum, consensus, critique, mound = mock_memory_systems

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
            critique_store=critique,
            knowledge_mound=mound,
        )

        tx = await coordinator.commit_debate_outcome(mock_context)

        targets = [op.target for op in tx.operations]
        assert "mound" in targets
        assert len(tx.operations) == 4

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


# =============================================================================
# Test Rollback Functionality
# =============================================================================


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

    @pytest.mark.asyncio
    async def test_no_rollback_when_disabled(self, mock_context):
        """Test no rollback when rollback_on_failure is False."""
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
                rollback_on_failure=False,
                write_critique=False,
                write_mound=False,
            ),
        )

        assert tx.rolled_back is False
        assert "continuum" not in rollback_called

    @pytest.mark.asyncio
    async def test_sequential_stops_on_failure_with_rollback(self, mock_context):
        """Test sequential execution stops on failure when rollback enabled."""
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="ok")
        continuum.delete = MagicMock(return_value={"deleted": True})

        consensus = MagicMock()
        consensus.store_consensus = MagicMock(side_effect=Exception("Error"))

        critique = MagicMock()
        critique.store_result = MagicMock()

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
            critique_store=critique,
        )

        tx = await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(
                rollback_on_failure=True,
                write_mound=False,
            ),
        )

        # Critique should not be called because consensus failed first
        critique.store_result.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_rollback_handler(self, mock_context):
        """Test registering and using a custom rollback handler."""
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="entry-123")

        consensus = MagicMock()
        consensus.store_consensus = MagicMock(side_effect=Exception("Error"))

        custom_rollback_data = []

        async def custom_rollback(op):
            custom_rollback_data.append(
                {
                    "target": op.target,
                    "result": op.result,
                }
            )
            return True

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
        )
        coordinator.register_rollback_handler("continuum", custom_rollback)

        tx = await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(
                rollback_on_failure=True,
                write_critique=False,
                write_mound=False,
            ),
        )

        assert len(custom_rollback_data) == 1
        assert custom_rollback_data[0]["target"] == "continuum"
        assert custom_rollback_data[0]["result"] == "entry-123"


# =============================================================================
# Test Default Rollback Handlers
# =============================================================================


class TestDefaultRollbackHandlers:
    """Tests for default rollback handlers with real delete methods."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock debate context for testing."""
        ctx = MagicMock()
        ctx.debate_id = "rollback-test-123"
        ctx.env = MagicMock()
        ctx.env.task = "Test rollback handlers"
        ctx.result = MagicMock()
        ctx.result.id = "rollback-test-123"
        ctx.result.task = "Test rollback handlers"
        ctx.result.final_answer = "Test answer"
        ctx.result.confidence = 0.9
        ctx.result.domain = "testing"
        ctx.result.consensus_reached = True
        ctx.result.winner = "agent1"
        ctx.result.rounds_used = 2
        ctx.result.critiques = []
        ctx.result.key_claims = []
        return ctx

    @pytest.mark.asyncio
    async def test_default_handlers_registered_for_continuum(self, mock_context):
        """Default rollback handler is registered for continuum memory."""
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="entry-123")
        continuum.delete = MagicMock(return_value={"deleted": True})

        coordinator = MemoryCoordinator(continuum_memory=continuum)

        # Verify handler is registered
        assert "continuum" in coordinator._rollback_handlers

    @pytest.mark.asyncio
    async def test_default_handlers_registered_for_consensus(self, mock_context):
        """Default rollback handler is registered for consensus memory."""
        consensus = MagicMock()
        mock_record = MagicMock()
        mock_record.id = "consensus-123"
        consensus.store_consensus = MagicMock(return_value=mock_record)
        consensus.delete_consensus = MagicMock(return_value=True)

        coordinator = MemoryCoordinator(consensus_memory=consensus)

        # Verify handler is registered
        assert "consensus" in coordinator._rollback_handlers

    @pytest.mark.asyncio
    async def test_default_handlers_registered_for_critique(self, mock_context):
        """Default rollback handler is registered for critique store."""
        critique = MagicMock()
        critique.store_result = MagicMock()
        critique.delete_debate = MagicMock(return_value=True)

        coordinator = MemoryCoordinator(critique_store=critique)

        # Verify handler is registered
        assert "critique" in coordinator._rollback_handlers

    @pytest.mark.asyncio
    async def test_default_handlers_registered_for_mound(self, mock_context):
        """Default rollback handler is registered for knowledge mound."""
        mound = AsyncMock()
        mound.ingest_debate_outcome = AsyncMock(return_value="mound-123")
        mound.delete_entry = AsyncMock(return_value=True)

        coordinator = MemoryCoordinator(knowledge_mound=mound)

        # Verify handler is registered
        assert "mound" in coordinator._rollback_handlers

    @pytest.mark.asyncio
    async def test_continuum_rollback_calls_delete(self, mock_context):
        """Continuum rollback handler calls delete method."""
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="entry-to-rollback")
        continuum.delete = MagicMock(return_value={"deleted": True})

        # Consensus will fail to trigger rollback
        consensus = MagicMock()
        consensus.store_consensus = MagicMock(side_effect=Exception("Trigger rollback"))

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
        )

        tx = await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(
                rollback_on_failure=True,
                write_critique=False,
                write_mound=False,
            ),
        )

        # Verify rollback occurred
        assert tx.rolled_back is True
        # Verify delete was called with the entry ID
        continuum.delete.assert_called_once()
        call_kwargs = continuum.delete.call_args[1]
        assert call_kwargs["memory_id"] == "entry-to-rollback"
        assert call_kwargs["archive"] is True
        assert call_kwargs["reason"] == "transaction_rollback"

    @pytest.mark.asyncio
    async def test_consensus_rollback_calls_delete_consensus(self, mock_context):
        """Consensus rollback handler calls delete_consensus method."""
        # Continuum succeeds
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="continuum-entry")
        continuum.delete = MagicMock(return_value={"deleted": True})

        # Consensus succeeds
        consensus = MagicMock()
        mock_record = MagicMock()
        mock_record.id = "consensus-to-rollback"
        consensus.store_consensus = MagicMock(return_value=mock_record)
        consensus.delete_consensus = MagicMock(return_value=True)

        # Critique fails to trigger rollback
        critique = MagicMock()
        critique.store_result = MagicMock(side_effect=Exception("Trigger rollback"))

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
            critique_store=critique,
        )

        tx = await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(
                rollback_on_failure=True,
                write_mound=False,
            ),
        )

        # Verify rollback occurred
        assert tx.rolled_back is True
        # Verify delete_consensus was called
        consensus.delete_consensus.assert_called_once_with(
            consensus_id="consensus-to-rollback",
            cascade_dissents=True,
        )

    @pytest.mark.asyncio
    async def test_critique_rollback_calls_delete_debate(self, mock_context):
        """Critique rollback handler calls delete_debate method."""
        # All succeed except mound
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="continuum-entry")
        continuum.delete = MagicMock(return_value={"deleted": True})

        consensus = MagicMock()
        mock_record = MagicMock()
        mock_record.id = "consensus-entry"
        consensus.store_consensus = MagicMock(return_value=mock_record)
        consensus.delete_consensus = MagicMock(return_value=True)

        critique = MagicMock()
        critique.store_result = MagicMock()
        critique.delete_debate = MagicMock(return_value=True)

        # Mound fails to trigger rollback
        mound = AsyncMock()
        mound.ingest_debate_outcome = AsyncMock(side_effect=Exception("Trigger rollback"))

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
            critique_store=critique,
            knowledge_mound=mound,
        )

        tx = await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(rollback_on_failure=True),
        )

        # Verify rollback occurred
        assert tx.rolled_back is True
        # Verify delete_debate was called
        critique.delete_debate.assert_called_once_with(
            debate_id="rollback-test-123",
            cascade_critiques=True,
        )

    @pytest.mark.asyncio
    async def test_mound_rollback_calls_delete_entry(self, mock_context):
        """Mound rollback handler calls delete_entry method."""
        # All succeed
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="continuum-entry")
        continuum.delete = MagicMock(return_value={"deleted": True})

        consensus = MagicMock()
        mock_record = MagicMock()
        mock_record.id = "consensus-entry"
        consensus.store_consensus = MagicMock(return_value=mock_record)
        consensus.delete_consensus = MagicMock(return_value=True)

        critique = MagicMock()
        critique.store_result = MagicMock()
        critique.delete_debate = MagicMock(return_value=True)

        mound = AsyncMock()
        mound.ingest_debate_outcome = AsyncMock(return_value="mound-entry-to-rollback")
        mound.delete_entry = AsyncMock(return_value=True)

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
            critique_store=critique,
            knowledge_mound=mound,
        )

        # Inject a failure after all writes succeed by calling rollback manually
        # First, create a transaction with successful mound write
        tx = await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(rollback_on_failure=False),
        )

        # Now manually trigger rollback on the successful operations
        await coordinator._rollback_successful(tx)

        # Verify mound delete_entry was called
        mound.delete_entry.assert_called_once()
        call_kwargs = mound.delete_entry.call_args[1]
        assert call_kwargs["km_id"] == "mound-entry-to-rollback"
        assert call_kwargs["archive"] is True

    @pytest.mark.asyncio
    async def test_mound_rollback_uses_delete_node_async_fallback(self, mock_context):
        """Mound rollback uses delete_node_async if delete_entry unavailable."""
        mound = AsyncMock()
        mound.ingest_debate_outcome = AsyncMock(return_value="mound-entry")
        # Only has delete_node_async, not delete_entry
        del mound.delete_entry
        mound.delete_node_async = AsyncMock(return_value=True)

        # Create a successful write operation to test rollback
        op = WriteOperation(
            id="op-1",
            target="mound",
            status=WriteStatus.SUCCESS,
            result="mound-entry",
        )

        coordinator = MemoryCoordinator(knowledge_mound=mound)

        # Manually call the rollback handler
        result = await coordinator._rollback_mound(op)

        assert result is True
        mound.delete_node_async.assert_called_once_with("mound-entry")


# =============================================================================
# Test Metrics and State Tracking
# =============================================================================


class TestMetricsTracking:
    """Tests for metrics tracking."""

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

    def test_get_metrics(self):
        """Test getting coordinator metrics."""
        coordinator = MemoryCoordinator()
        coordinator.metrics.total_transactions = 10
        coordinator.metrics.successful_transactions = 8
        coordinator.metrics.partial_failures = 2

        metrics = coordinator.get_metrics()

        assert metrics["total_transactions"] == 10
        assert metrics["success_rate"] == 0.8

    def test_success_rate_zero_transactions(self):
        """Test success rate with zero transactions."""
        coordinator = MemoryCoordinator()
        metrics = coordinator.get_metrics()

        assert metrics["success_rate"] == 0.0

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
    async def test_full_failure_metrics(self, mock_context):
        """Test metrics updated on full failure."""
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(side_effect=Exception("DB error"))

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            options=CoordinatorOptions(
                write_consensus=False,
                write_critique=False,
                write_mound=False,
            ),
        )

        tx = await coordinator.commit_debate_outcome(mock_context)

        assert tx.success is False
        assert coordinator.metrics.full_failures == 1

    @pytest.mark.asyncio
    async def test_writes_per_target_metrics(self, mock_context):
        """Test writes per target are tracked."""
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="ok")

        consensus = MagicMock()
        mock_record = MagicMock()
        mock_record.id = "consensus-1"
        consensus.store_consensus = MagicMock(return_value=mock_record)

        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus,
        )

        await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(
                write_critique=False,
                write_mound=False,
            ),
        )

        metrics = coordinator.get_metrics()
        assert metrics["writes_per_target"]["continuum"] == 1
        assert metrics["writes_per_target"]["consensus"] == 1
        assert metrics["total_writes"] == 2


# =============================================================================
# Test Retry Behavior
# =============================================================================


class TestRetryBehavior:
    """Tests for retry behavior on transient failures."""

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
    async def test_retry_on_transient_failure(self, mock_context):
        """Test operation is retried on transient failure."""
        call_count = 0

        def store_with_retry(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Transient error")
            return "entry-1"

        continuum = MagicMock()
        continuum.store_pattern = MagicMock(side_effect=store_with_retry)

        coordinator = MemoryCoordinator(continuum_memory=continuum)

        tx = await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(
                max_retries=2,
                retry_delay_seconds=0.01,
                write_consensus=False,
                write_critique=False,
                write_mound=False,
            ),
        )

        assert tx.success is True
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_fail_after_max_retries(self, mock_context):
        """Test operation fails after max retries exhausted."""
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(side_effect=ValueError("Persistent error"))

        coordinator = MemoryCoordinator(continuum_memory=continuum)

        tx = await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(
                max_retries=2,
                retry_delay_seconds=0.01,
                rollback_on_failure=False,
                write_consensus=False,
                write_critique=False,
                write_mound=False,
            ),
        )

        assert tx.success is False
        # Should have been called 3 times (1 initial + 2 retries)
        assert continuum.store_pattern.call_count == 3


# =============================================================================
# Test Transaction Timeout
# =============================================================================


class TestTransactionTimeout:
    """Tests for transaction timeout handling."""

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
    async def test_parallel_timeout_marks_pending_as_failed(self, mock_context):
        """Test timeout marks pending operations as failed."""

        async def slow_operation(*args, **kwargs):
            await asyncio.sleep(2)  # Longer than timeout
            return "entry-1"

        mound = AsyncMock()
        mound.ingest_debate_outcome = slow_operation

        coordinator = MemoryCoordinator(knowledge_mound=mound)

        tx = await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(
                timeout_seconds=0.1,
                parallel_writes=True,
                write_continuum=False,
                write_consensus=False,
                write_critique=False,
                write_mound=True,
            ),
        )

        # The operation should fail due to timeout
        assert tx.success is False
        failed = tx.get_failed_operations()
        assert len(failed) == 1
        assert failed[0].error == "timeout"


# =============================================================================
# Test Unknown Target Handling
# =============================================================================


class TestUnknownTargetHandling:
    """Tests for handling unknown write targets."""

    def test_unknown_target_operation_fails(self):
        """Test that unknown target operations are marked as failed."""
        op = WriteOperation(id="op-1", target="unknown_system")

        coordinator = MemoryCoordinator()

        # Manually call execute operation
        import asyncio

        asyncio.run(coordinator._execute_operation(op, CoordinatorOptions()))

        assert op.status == WriteStatus.FAILED
        assert "Unknown target" in op.error


# =============================================================================
# Test Consensus Strength Mapping
# =============================================================================


class TestConsensusStrengthMapping:
    """Tests for consensus strength mapping based on confidence."""

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
        ctx.result.consensus_reached = True
        ctx.result.winner = "agent1"
        ctx.result.rounds_used = 2
        return ctx

    @pytest.mark.asyncio
    async def test_unanimous_strength_high_confidence(self, mock_context):
        """Test unanimous strength for high confidence."""
        mock_context.result.confidence = 0.95

        consensus = MagicMock()
        mock_record = MagicMock()
        mock_record.id = "consensus-1"
        consensus.store_consensus = MagicMock(return_value=mock_record)

        coordinator = MemoryCoordinator(consensus_memory=consensus)

        await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(
                write_continuum=False,
                write_critique=False,
                write_mound=False,
            ),
        )

        # Check that store_consensus was called with UNANIMOUS strength
        call_kwargs = consensus.store_consensus.call_args[1]
        from aragora.memory.consensus import ConsensusStrength

        assert call_kwargs["strength"] == ConsensusStrength.UNANIMOUS

    @pytest.mark.asyncio
    async def test_split_strength_low_confidence(self, mock_context):
        """Test split strength for low confidence."""
        mock_context.result.confidence = 0.4

        consensus = MagicMock()
        mock_record = MagicMock()
        mock_record.id = "consensus-1"
        consensus.store_consensus = MagicMock(return_value=mock_record)

        coordinator = MemoryCoordinator(consensus_memory=consensus)

        await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(
                write_continuum=False,
                write_critique=False,
                write_mound=False,
            ),
        )

        # Check that store_consensus was called with SPLIT strength
        call_kwargs = consensus.store_consensus.call_args[1]
        from aragora.memory.consensus import ConsensusStrength

        assert call_kwargs["strength"] == ConsensusStrength.SPLIT


# =============================================================================
# Test Mound Fallback Methods
# =============================================================================


class TestMoundFallbackMethods:
    """Tests for KnowledgeMound fallback methods."""

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
    async def test_mound_uses_store_knowledge_fallback(self, mock_context):
        """Test mound write uses store_knowledge when ingest_debate_outcome unavailable."""
        mound = AsyncMock()
        # Remove ingest_debate_outcome to test fallback
        del mound.ingest_debate_outcome
        mound.store_knowledge = AsyncMock(return_value="knowledge-1")

        coordinator = MemoryCoordinator(knowledge_mound=mound)

        tx = await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(
                write_continuum=False,
                write_consensus=False,
                write_critique=False,
                write_mound=True,
            ),
        )

        assert tx.success is True
        mound.store_knowledge.assert_called_once()

    @pytest.mark.asyncio
    async def test_mound_fails_without_any_method(self, mock_context):
        """Test mound write fails when no suitable method available."""
        mound = AsyncMock()
        # Remove both methods
        del mound.ingest_debate_outcome
        del mound.store_knowledge

        coordinator = MemoryCoordinator(knowledge_mound=mound)

        tx = await coordinator.commit_debate_outcome(
            mock_context,
            options=CoordinatorOptions(
                write_continuum=False,
                write_consensus=False,
                write_critique=False,
                write_mound=True,
                rollback_on_failure=False,
            ),
        )

        assert tx.success is False
        failed = tx.get_failed_operations()
        assert len(failed) == 1
        assert "neither ingest_debate_outcome nor store_knowledge" in failed[0].error
