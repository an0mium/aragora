"""
Tests for memory coordinator edge cases.

Covers:
- Skipped operations tracking (confidence thresholds)
- Supermemory rollback markers
- Tenant isolation enforcement

Run with:
    pytest tests/memory/test_coordinator_edge_cases.py -v --timeout=30
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    """Verify the coordinator module and edge case types can be imported."""

    def test_import_write_status_skipped(self):
        from aragora.memory.coordinator import WriteStatus

        assert WriteStatus.SKIPPED.value == "skipped"

    def test_import_skipped_operation(self):
        from aragora.memory.coordinator import SkippedOperation

        op = SkippedOperation(
            target="mound",
            reason="Low confidence",
            threshold=0.7,
            actual_value=0.5,
        )
        assert op.target == "mound"
        assert op.threshold == 0.7
        assert op.actual_value == 0.5

    def test_import_supermemory_rollback_marker(self):
        from aragora.memory.coordinator import SupermemoryRollbackMarker

        marker = SupermemoryRollbackMarker(
            memory_id="test-123",
            transaction_id="tx-456",
            reason="transaction_rollback",
        )
        assert marker.memory_id == "test-123"
        assert marker.transaction_id == "tx-456"
        assert marker.retry_count == 0
        assert marker.max_retries == 3


# ---------------------------------------------------------------------------
# SkippedOperation tests
# ---------------------------------------------------------------------------


class TestSkippedOperation:
    """Tests for SkippedOperation tracking."""

    def test_skipped_operation_records_threshold_details(self):
        from aragora.memory.coordinator import SkippedOperation

        op = SkippedOperation(
            target="mound",
            reason="Confidence 0.50 below threshold 0.70",
            threshold=0.7,
            actual_value=0.5,
        )

        assert "0.50" in op.reason
        assert "0.70" in op.reason
        assert op.threshold == 0.7
        assert op.actual_value == 0.5

    def test_skipped_operation_has_timestamp(self):
        from aragora.memory.coordinator import SkippedOperation

        before = datetime.now()
        op = SkippedOperation(target="supermemory", reason="test")
        after = datetime.now()

        assert before <= op.timestamp <= after


# ---------------------------------------------------------------------------
# SupermemoryRollbackMarker tests
# ---------------------------------------------------------------------------


class TestSupermemoryRollbackMarker:
    """Tests for SupermemoryRollbackMarker."""

    def test_marker_has_default_retry_count(self):
        from aragora.memory.coordinator import SupermemoryRollbackMarker

        marker = SupermemoryRollbackMarker(
            memory_id="test",
            transaction_id="tx-1",
        )
        assert marker.retry_count == 0
        assert marker.max_retries == 3

    def test_marker_records_timestamp(self):
        from aragora.memory.coordinator import SupermemoryRollbackMarker

        before = datetime.now()
        marker = SupermemoryRollbackMarker(
            memory_id="test",
            transaction_id="tx-1",
        )
        after = datetime.now()

        assert before <= marker.marked_at <= after


# ---------------------------------------------------------------------------
# MemoryTransaction tests
# ---------------------------------------------------------------------------


class TestMemoryTransactionSkipped:
    """Tests for MemoryTransaction with skipped operations."""

    def test_transaction_has_skipped_operations_field(self):
        from aragora.memory.coordinator import MemoryTransaction

        tx = MemoryTransaction(id="test-1", debate_id="debate-1")
        assert hasattr(tx, "skipped_operations")
        assert tx.skipped_operations == []

    def test_transaction_has_skipped_property(self):
        from aragora.memory.coordinator import MemoryTransaction, SkippedOperation

        tx = MemoryTransaction(id="test-1", debate_id="debate-1")
        assert tx.has_skipped is False

        tx.skipped_operations.append(SkippedOperation(target="mound", reason="test"))
        assert tx.has_skipped is True

    def test_get_skipped_operations_returns_list(self):
        from aragora.memory.coordinator import MemoryTransaction, SkippedOperation

        tx = MemoryTransaction(id="test-1", debate_id="debate-1")
        skip1 = SkippedOperation(target="mound", reason="low confidence")
        skip2 = SkippedOperation(target="supermemory", reason="disabled")

        tx.skipped_operations = [skip1, skip2]
        skipped = tx.get_skipped_operations()

        assert len(skipped) == 2
        assert skip1 in skipped
        assert skip2 in skipped

    def test_success_ignores_skipped(self):
        from aragora.memory.coordinator import (
            MemoryTransaction,
            WriteOperation,
            WriteStatus,
        )

        tx = MemoryTransaction(id="test-1", debate_id="debate-1")

        # Add a successful operation
        op1 = WriteOperation(id="op-1", target="continuum")
        op1.mark_success("result-1")
        tx.operations.append(op1)

        # Add a skipped operation
        op2 = WriteOperation(id="op-2", target="mound")
        op2.mark_skipped("Low confidence")
        tx.operations.append(op2)

        # Transaction should still be considered successful
        assert tx.success is True


# ---------------------------------------------------------------------------
# MemoryCoordinator._build_operations tests
# ---------------------------------------------------------------------------


class TestBuildOperationsSkipping:
    """Tests for _build_operations skipping logic."""

    @pytest.fixture
    def mock_coordinator(self):
        """Create a coordinator with mock memory systems."""
        from aragora.memory.coordinator import CoordinatorOptions, MemoryCoordinator

        # Create mock memory systems
        mock_continuum = MagicMock()
        mock_mound = MagicMock()
        mock_supermemory = MagicMock()

        coordinator = MemoryCoordinator(
            continuum_memory=mock_continuum,
            knowledge_mound=mock_mound,
            supermemory_adapter=mock_supermemory,
            options=CoordinatorOptions(
                write_continuum=True,
                write_consensus=False,
                write_critique=False,
                write_mound=True,
                write_supermemory=True,
                min_confidence_for_mound=0.7,
                min_confidence_for_supermemory=0.7,
            ),
        )
        return coordinator

    @pytest.fixture
    def mock_debate_context(self):
        """Create a mock debate context."""
        ctx = MagicMock()
        ctx.debate_id = "debate-123"
        ctx.env = MagicMock()
        ctx.env.task = "Test task"
        ctx.domain = "test"
        ctx.agents = []
        return ctx

    @pytest.fixture
    def mock_result(self):
        """Create a mock debate result."""
        result = MagicMock()
        result.final_answer = "Test answer"
        result.confidence = 0.5  # Below default thresholds
        result.consensus_reached = False
        result.winner = None
        return result

    def test_low_confidence_skips_mound(self, mock_coordinator, mock_debate_context, mock_result):
        """Test that low confidence skips mound write and records it."""
        mock_result.confidence = 0.5  # Below 0.7 threshold

        operations, skipped = mock_coordinator._build_operations(
            mock_debate_context,
            mock_result,
            mock_coordinator.options,
        )

        # Should have continuum but not mound
        targets = [op.target for op in operations]
        assert "continuum" in targets
        assert "mound" not in targets

        # Should have skipped mound
        skipped_targets = [s.target for s in skipped]
        assert "mound" in skipped_targets

        # Skipped should have threshold info
        mound_skip = next(s for s in skipped if s.target == "mound")
        assert mound_skip.threshold == 0.7
        assert mound_skip.actual_value == 0.5

    def test_low_confidence_skips_supermemory(
        self, mock_coordinator, mock_debate_context, mock_result
    ):
        """Test that low confidence skips supermemory write and records it."""
        mock_result.confidence = 0.5  # Below 0.7 threshold

        operations, skipped = mock_coordinator._build_operations(
            mock_debate_context,
            mock_result,
            mock_coordinator.options,
        )

        # Should have skipped supermemory
        skipped_targets = [s.target for s in skipped]
        assert "supermemory" in skipped_targets

        sm_skip = next(s for s in skipped if s.target == "supermemory")
        assert sm_skip.threshold == 0.7

    def test_high_confidence_includes_all(self, mock_coordinator, mock_debate_context, mock_result):
        """Test that high confidence includes mound and supermemory."""
        mock_result.confidence = 0.9  # Above thresholds

        operations, skipped = mock_coordinator._build_operations(
            mock_debate_context,
            mock_result,
            mock_coordinator.options,
        )

        targets = [op.target for op in operations]
        assert "mound" in targets
        assert "supermemory" in targets
        assert len(skipped) == 0


# ---------------------------------------------------------------------------
# Supermemory rollback tests
# ---------------------------------------------------------------------------


class TestSupermemoryRollback:
    """Tests for supermemory rollback marker functionality."""

    @pytest.fixture
    def coordinator_with_supermemory(self):
        """Create a coordinator with mock supermemory adapter."""
        from aragora.memory.coordinator import CoordinatorOptions, MemoryCoordinator

        mock_adapter = MagicMock()
        mock_adapter.delete_memory = AsyncMock()

        coordinator = MemoryCoordinator(
            supermemory_adapter=mock_adapter,
            options=CoordinatorOptions(write_supermemory=True),
        )
        return coordinator

    @pytest.mark.asyncio
    async def test_rollback_creates_marker(self, coordinator_with_supermemory):
        """Test that _rollback_supermemory creates a marker."""
        from aragora.memory.coordinator import WriteOperation

        coordinator = coordinator_with_supermemory
        op = WriteOperation(id="op-1", target="supermemory", result="sm-memory-123")

        result = await coordinator._rollback_supermemory(op)

        assert result is True
        assert len(coordinator._supermemory_rollback_markers) == 1

        marker = coordinator._supermemory_rollback_markers[0]
        assert marker.memory_id == "sm-memory-123"
        assert marker.transaction_id == "op-1"

    def test_get_pending_rollbacks(self, coordinator_with_supermemory):
        """Test getting pending rollback markers."""
        from aragora.memory.coordinator import SupermemoryRollbackMarker

        coordinator = coordinator_with_supermemory

        marker1 = SupermemoryRollbackMarker(memory_id="m1", transaction_id="t1")
        marker2 = SupermemoryRollbackMarker(memory_id="m2", transaction_id="t2")
        coordinator._supermemory_rollback_markers = [marker1, marker2]

        pending = coordinator.get_pending_supermemory_rollbacks()

        assert len(pending) == 2
        assert marker1 in pending
        assert marker2 in pending

    @pytest.mark.asyncio
    async def test_process_rollbacks_success(self, coordinator_with_supermemory):
        """Test successful processing of rollback markers."""
        from aragora.memory.coordinator import SupermemoryRollbackMarker

        coordinator = coordinator_with_supermemory
        coordinator.supermemory_adapter.delete_memory = AsyncMock()

        marker = SupermemoryRollbackMarker(memory_id="m1", transaction_id="t1")
        coordinator._supermemory_rollback_markers = [marker]

        successful, failed = await coordinator.process_supermemory_rollbacks()

        assert successful == 1
        assert failed == 0
        assert len(coordinator._supermemory_rollback_markers) == 0

    @pytest.mark.asyncio
    async def test_process_rollbacks_with_network_error(self, coordinator_with_supermemory):
        """Test rollback processing with network errors (should retry)."""
        from aragora.memory.coordinator import SupermemoryRollbackMarker

        coordinator = coordinator_with_supermemory
        coordinator.supermemory_adapter.delete_memory = AsyncMock(
            side_effect=ConnectionError("Network error")
        )

        marker = SupermemoryRollbackMarker(memory_id="m1", transaction_id="t1")
        coordinator._supermemory_rollback_markers = [marker]

        successful, failed = await coordinator.process_supermemory_rollbacks()

        assert successful == 0
        assert failed == 1
        # Marker should be kept for retry
        assert len(coordinator._supermemory_rollback_markers) == 1
        assert coordinator._supermemory_rollback_markers[0].retry_count == 1


# ---------------------------------------------------------------------------
# TenantRequiredError tests
# ---------------------------------------------------------------------------


class TestTenantRequiredError:
    """Tests for TenantRequiredError exception."""

    def test_import_tenant_required_error(self):
        from aragora.memory.continuum.retrieval import TenantRequiredError

        assert TenantRequiredError is not None

    def test_error_message_includes_operation(self):
        from aragora.memory.continuum.retrieval import TenantRequiredError

        error = TenantRequiredError("search")
        assert "search" in str(error)
        assert "tenant_id" in str(error)

    def test_error_default_operation(self):
        from aragora.memory.continuum.retrieval import TenantRequiredError

        error = TenantRequiredError()
        assert "retrieve" in str(error)


# ---------------------------------------------------------------------------
# Tenant isolation enforcement tests
# ---------------------------------------------------------------------------


class TestTenantIsolationEnforcement:
    """Tests for tenant isolation enforcement in retrieval."""

    @pytest.fixture
    def mock_continuum_memory(self, tmp_path):
        """Create a mock continuum memory."""
        from aragora.memory.continuum import ContinuumMemory

        db_path = tmp_path / "test_continuum.db"
        return ContinuumMemory(db_path=str(db_path))

    def test_retrieve_without_enforcement_allows_none_tenant(self, mock_continuum_memory):
        """Test that retrieve without enforcement allows None tenant_id."""
        # Should not raise
        results = mock_continuum_memory.retrieve(
            query="test",
            tenant_id=None,
            enforce_tenant_isolation=False,
        )
        assert isinstance(results, list)

    def test_retrieve_with_enforcement_requires_tenant(self, mock_continuum_memory):
        """Test that retrieve with enforcement requires tenant_id."""
        from aragora.memory.continuum.retrieval import TenantRequiredError

        with pytest.raises(TenantRequiredError) as exc_info:
            mock_continuum_memory.retrieve(
                query="test",
                tenant_id=None,
                enforce_tenant_isolation=True,
            )

        assert "tenant_id" in str(exc_info.value)

    def test_retrieve_with_enforcement_passes_with_tenant(self, mock_continuum_memory):
        """Test that retrieve with enforcement passes when tenant_id provided."""
        # Should not raise
        results = mock_continuum_memory.retrieve(
            query="test",
            tenant_id="tenant-123",
            enforce_tenant_isolation=True,
        )
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# WriteOperation.mark_skipped tests
# ---------------------------------------------------------------------------


class TestWriteOperationMarkSkipped:
    """Tests for WriteOperation.mark_skipped method."""

    def test_mark_skipped_sets_status(self):
        from aragora.memory.coordinator import WriteOperation, WriteStatus

        op = WriteOperation(id="test-1", target="mound")
        op.mark_skipped("Low confidence")

        assert op.status == WriteStatus.SKIPPED
        assert op.error == "Low confidence"
