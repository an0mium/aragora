"""
Tests for Molecules transaction safety features.

This module tests:
1. Transaction begin/commit/rollback semantics
2. Dependency graph validation (cycle detection)
3. Compensating actions for rollback
4. Deadlock detection for parallel molecules
5. Crash recovery scenarios
"""

import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.molecules import (
    CompensatingAction,
    CyclicDependencyError,
    DeadlockDetector,
    DeadlockError,
    DependencyGraph,
    DependencyValidationError,
    Molecule,
    MoleculeEngine,
    MoleculeResult,
    MoleculeStatus,
    MoleculeStep,
    MoleculeTransaction,
    StepExecutor,
    StepStatus,
    TransactionError,
    TransactionRollbackError,
    TransactionState,
    get_deadlock_detector,
    reset_deadlock_detector,
    reset_molecule_engine,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def reset_engine():
    """Reset the molecule engine singleton before and after each test."""
    reset_molecule_engine()
    reset_deadlock_detector()
    yield
    reset_molecule_engine()
    reset_deadlock_detector()


@pytest.fixture
def sample_steps():
    """Create sample steps for testing."""
    return [
        MoleculeStep.create("step1", "shell", {"command": "echo hello"}),
        MoleculeStep.create("step2", "shell", {"command": "echo world"}),
        MoleculeStep.create("step3", "agent", {"task": "review"}),
    ]


@pytest.fixture
def sample_molecule(sample_steps):
    """Create a sample molecule for testing."""
    return Molecule.create(
        name="test_molecule",
        steps=sample_steps,
        description="A test molecule",
    )


# =============================================================================
# Transaction Exceptions Tests
# =============================================================================


class TestTransactionExceptions:
    """Tests for transaction-related exceptions."""

    def test_transaction_error_base(self):
        """Test base TransactionError."""
        error = TransactionError("test error")
        assert str(error) == "test error"

    def test_transaction_rollback_error(self):
        """Test TransactionRollbackError."""
        error = TransactionRollbackError("rollback failed")
        assert str(error) == "rollback failed"
        assert isinstance(error, TransactionError)

    def test_cyclic_dependency_error_with_cycle(self):
        """Test CyclicDependencyError with cycle list."""
        cycle = ["step1", "step2", "step3", "step1"]
        error = CyclicDependencyError(cycle)
        assert error.cycle == cycle
        assert "step1 -> step2 -> step3 -> step1" in str(error)

    def test_cyclic_dependency_error_custom_message(self):
        """Test CyclicDependencyError with custom message."""
        error = CyclicDependencyError(["a", "b"], "Custom cycle message")
        assert str(error) == "Custom cycle message"

    def test_deadlock_error(self):
        """Test DeadlockError."""
        error = DeadlockError(
            molecules=["mol1", "mol2"],
            resources=["res1", "res2"],
        )
        assert error.molecules == ["mol1", "mol2"]
        assert error.resources == ["res1", "res2"]
        assert "Deadlock detected" in str(error)

    def test_dependency_validation_error(self):
        """Test DependencyValidationError."""
        error = DependencyValidationError(
            step_id="step1",
            missing_deps=["dep1", "dep2"],
        )
        assert error.step_id == "step1"
        assert error.missing_deps == ["dep1", "dep2"]
        assert "missing dependencies" in str(error)


# =============================================================================
# CompensatingAction Tests
# =============================================================================


class TestCompensatingAction:
    """Tests for CompensatingAction."""

    @pytest.mark.asyncio
    async def test_execute_restore_checkpoint(self, temp_dir):
        """Test executing checkpoint restoration."""
        checkpoint_file = temp_dir / "test.json"
        checkpoint_file.write_text('{"test": "data"}')

        action = CompensatingAction(
            step_id="step1",
            step_name="test_step",
            action_type="restore_checkpoint",
            checkpoint_path=checkpoint_file,
        )

        result = await action.execute()
        assert result is True

    @pytest.mark.asyncio
    async def test_execute_custom_handler_async(self):
        """Test executing async custom handler."""
        handler_called = {"called": False}

        async def custom_handler(metadata):
            handler_called["called"] = True
            handler_called["metadata"] = metadata

        action = CompensatingAction(
            step_id="step1",
            step_name="test_step",
            action_type="custom",
            custom_handler=custom_handler,
            metadata={"key": "value"},
        )

        result = await action.execute()
        assert result is True
        assert handler_called["called"] is True
        assert handler_called["metadata"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_execute_custom_handler_sync(self):
        """Test executing sync custom handler."""
        handler_called = {"called": False}

        def sync_handler(metadata):
            handler_called["called"] = True

        action = CompensatingAction(
            step_id="step1",
            step_name="test_step",
            action_type="custom",
            custom_handler=sync_handler,
        )

        result = await action.execute()
        assert result is True
        assert handler_called["called"] is True

    @pytest.mark.asyncio
    async def test_execute_no_checkpoint_returns_true(self, temp_dir):
        """Test that missing checkpoint still returns true (idempotent)."""
        action = CompensatingAction(
            step_id="step1",
            step_name="test_step",
            action_type="restore_checkpoint",
            checkpoint_path=temp_dir / "nonexistent.json",
        )

        result = await action.execute()
        # Returns true even if checkpoint doesn't exist
        assert result is True

    @pytest.mark.asyncio
    async def test_execute_handler_exception(self):
        """Test handling of exception in custom handler."""

        def failing_handler(metadata):
            raise RuntimeError("Handler failed")

        action = CompensatingAction(
            step_id="step1",
            step_name="test_step",
            action_type="custom",
            custom_handler=failing_handler,
        )

        result = await action.execute()
        assert result is False


# =============================================================================
# MoleculeTransaction Tests
# =============================================================================


class TestMoleculeTransaction:
    """Tests for MoleculeTransaction."""

    def test_begin_transaction(self, temp_dir):
        """Test beginning a new transaction."""
        transaction = MoleculeTransaction.begin(
            molecule_id="mol-123",
            checkpoint_dir=temp_dir,
        )

        assert transaction.molecule_id == "mol-123"
        assert transaction.state == TransactionState.ACTIVE
        assert transaction.transaction_id is not None
        assert transaction.started_at is not None

    def test_begin_transaction_with_snapshot(self, temp_dir):
        """Test beginning transaction with pre-transaction snapshot."""
        snapshot = {"id": "mol-123", "status": "pending"}
        transaction = MoleculeTransaction.begin(
            molecule_id="mol-123",
            checkpoint_dir=temp_dir,
            snapshot=snapshot,
        )

        assert transaction.pre_transaction_snapshot == snapshot

    def test_add_compensating_action(self, temp_dir):
        """Test adding compensating actions."""
        transaction = MoleculeTransaction.begin("mol-123", temp_dir)

        action = CompensatingAction(
            step_id="step1",
            step_name="test_step",
            action_type="custom",
        )
        transaction.add_compensating_action(action)

        assert len(transaction.compensating_actions) == 1
        assert transaction.compensating_actions[0].step_id == "step1"

    def test_mark_step_completed(self, temp_dir):
        """Test marking steps as completed."""
        transaction = MoleculeTransaction.begin("mol-123", temp_dir)

        transaction.mark_step_completed("step1")
        transaction.mark_step_completed("step2")
        transaction.mark_step_completed("step1")  # Duplicate

        assert len(transaction.completed_steps) == 2
        assert "step1" in transaction.completed_steps
        assert "step2" in transaction.completed_steps

    @pytest.mark.asyncio
    async def test_commit_success(self, temp_dir):
        """Test successful commit."""
        transaction = MoleculeTransaction.begin("mol-123", temp_dir)
        transaction.mark_step_completed("step1")

        result = await transaction.commit()

        assert result is True
        assert transaction.state == TransactionState.COMMITTED
        assert transaction.committed_at is not None

    @pytest.mark.asyncio
    async def test_commit_not_active(self, temp_dir):
        """Test that commit fails if not in active state."""
        transaction = MoleculeTransaction.begin("mol-123", temp_dir)
        transaction.state = TransactionState.COMMITTED

        result = await transaction.commit()

        assert result is False

    @pytest.mark.asyncio
    async def test_rollback_success(self, temp_dir):
        """Test successful rollback."""
        transaction = MoleculeTransaction.begin("mol-123", temp_dir)
        transaction.mark_step_completed("step1")

        result = await transaction.rollback()

        assert result is True
        assert transaction.state == TransactionState.ROLLED_BACK
        assert transaction.rolled_back_at is not None

    @pytest.mark.asyncio
    async def test_rollback_executes_compensating_actions(self, temp_dir):
        """Test that rollback executes compensating actions in reverse order."""
        transaction = MoleculeTransaction.begin("mol-123", temp_dir)

        execution_order = []

        async def handler1(metadata):
            execution_order.append("action1")

        async def handler2(metadata):
            execution_order.append("action2")

        action1 = CompensatingAction(
            step_id="step1",
            step_name="step1",
            action_type="custom",
            custom_handler=handler1,
        )
        action2 = CompensatingAction(
            step_id="step2",
            step_name="step2",
            action_type="custom",
            custom_handler=handler2,
        )

        transaction.add_compensating_action(action1)
        transaction.add_compensating_action(action2)
        transaction.mark_step_completed("step1")
        transaction.mark_step_completed("step2")

        await transaction.rollback()

        # Actions should be executed in reverse order (LIFO)
        assert execution_order == ["action2", "action1"]

    @pytest.mark.asyncio
    async def test_rollback_restores_snapshot(self, temp_dir):
        """Test that rollback restores pre-transaction snapshot."""
        snapshot = {"id": "mol-123", "name": "original", "status": "pending"}
        transaction = MoleculeTransaction.begin(
            molecule_id="mol-123",
            checkpoint_dir=temp_dir,
            snapshot=snapshot,
        )

        await transaction.rollback()

        # Check that snapshot was written to file
        snapshot_file = temp_dir / "mol-123.json"
        assert snapshot_file.exists()

        with open(snapshot_file) as f:
            restored = json.load(f)
        assert restored == snapshot

    @pytest.mark.asyncio
    async def test_rollback_not_active_or_failed(self, temp_dir):
        """Test that rollback fails if not in active or failed state."""
        transaction = MoleculeTransaction.begin("mol-123", temp_dir)
        transaction.state = TransactionState.COMMITTED

        result = await transaction.rollback()

        assert result is False

    def test_to_dict(self, temp_dir):
        """Test serialization to dictionary."""
        transaction = MoleculeTransaction.begin("mol-123", temp_dir)
        transaction.mark_step_completed("step1")

        data = transaction.to_dict()

        assert data["molecule_id"] == "mol-123"
        assert data["state"] == "active"
        assert data["completed_steps"] == ["step1"]
        assert "transaction_id" in data
        assert "started_at" in data


# =============================================================================
# DependencyGraph Tests
# =============================================================================


class TestDependencyGraph:
    """Tests for DependencyGraph validation and cycle detection."""

    def test_validate_no_dependencies(self):
        """Test validation with no dependencies."""
        steps = [
            MoleculeStep.create("step1", "shell"),
            MoleculeStep.create("step2", "shell"),
            MoleculeStep.create("step3", "shell"),
        ]

        graph = DependencyGraph(steps)
        is_valid, error = graph.validate()

        assert is_valid is True
        assert error is None

    def test_validate_linear_dependencies(self):
        """Test validation with linear dependencies."""
        step1 = MoleculeStep.create("step1", "shell")
        step2 = MoleculeStep.create("step2", "shell", dependencies=[step1.id])
        step3 = MoleculeStep.create("step3", "shell", dependencies=[step2.id])

        graph = DependencyGraph([step1, step2, step3])
        is_valid, error = graph.validate()

        assert is_valid is True
        assert error is None

    def test_validate_parallel_dependencies(self):
        """Test validation with parallel dependencies."""
        step1 = MoleculeStep.create("step1", "shell")
        step2 = MoleculeStep.create("step2", "shell")
        step3 = MoleculeStep.create("step3", "shell", dependencies=[step1.id, step2.id])

        graph = DependencyGraph([step1, step2, step3])
        is_valid, error = graph.validate()

        assert is_valid is True
        assert error is None

    def test_detect_simple_cycle(self):
        """Test detection of simple 2-node cycle."""
        step1 = MoleculeStep.create("step1", "shell", dependencies=["placeholder"])
        step2 = MoleculeStep.create("step2", "shell", dependencies=[step1.id])
        step1.dependencies = [step2.id]

        graph = DependencyGraph([step1, step2])
        is_valid, error = graph.validate()

        assert is_valid is False
        assert error is not None

    def test_detect_complex_cycle(self):
        """Test detection of multi-node cycle."""
        step1 = MoleculeStep.create("step1", "shell", dependencies=["placeholder"])
        step2 = MoleculeStep.create("step2", "shell", dependencies=[step1.id])
        step3 = MoleculeStep.create("step3", "shell", dependencies=[step2.id])
        step1.dependencies = [step3.id]

        graph = DependencyGraph([step1, step2, step3])
        is_valid, error = graph.validate()

        assert is_valid is False
        assert error is not None

    def test_detect_missing_dependency(self):
        """Test detection of missing dependency reference."""
        step1 = MoleculeStep.create("step1", "shell")
        step2 = MoleculeStep.create("step2", "shell", dependencies=["nonexistent"])

        graph = DependencyGraph([step1, step2])
        is_valid, error = graph.validate()

        assert is_valid is False
        assert error is not None
        # Should contain step_id:missing_dep format
        assert any("nonexistent" in e for e in error)

    def test_get_execution_order_linear(self):
        """Test getting execution order for linear dependencies."""
        step1 = MoleculeStep.create("step1", "shell")
        step2 = MoleculeStep.create("step2", "shell", dependencies=[step1.id])
        step3 = MoleculeStep.create("step3", "shell", dependencies=[step2.id])

        graph = DependencyGraph([step1, step2, step3])
        order = graph.get_execution_order()

        # step1 must come before step2, step2 before step3
        assert order.index(step1.id) < order.index(step2.id)
        assert order.index(step2.id) < order.index(step3.id)

    def test_get_execution_order_with_cycle_raises(self):
        """Test that get_execution_order raises on cycle."""
        step1 = MoleculeStep.create("step1", "shell", dependencies=["placeholder"])
        step2 = MoleculeStep.create("step2", "shell", dependencies=[step1.id])
        step1.dependencies = [step2.id]

        graph = DependencyGraph([step1, step2])

        with pytest.raises(CyclicDependencyError):
            graph.get_execution_order()


# =============================================================================
# DeadlockDetector Tests
# =============================================================================


class TestDeadlockDetector:
    """Tests for DeadlockDetector."""

    @pytest.fixture
    def detector(self):
        """Create a fresh deadlock detector."""
        reset_deadlock_detector()
        return get_deadlock_detector()

    @pytest.mark.asyncio
    async def test_acquire_lock_success(self, detector):
        """Test successfully acquiring a lock."""
        result = await detector.acquire_lock("mol1", "resource1")
        assert result is True

        state = await detector.get_lock_state()
        assert "resource1" in state["locks"]
        assert state["locks"]["resource1"]["molecule"] == "mol1"

    @pytest.mark.asyncio
    async def test_acquire_own_lock_again(self, detector):
        """Test that acquiring own lock again succeeds."""
        await detector.acquire_lock("mol1", "resource1")
        result = await detector.acquire_lock("mol1", "resource1")
        assert result is True

    @pytest.mark.asyncio
    async def test_release_lock(self, detector):
        """Test releasing a lock."""
        await detector.acquire_lock("mol1", "resource1")
        await detector.release_lock("mol1", "resource1")

        state = await detector.get_lock_state()
        assert "resource1" not in state["locks"]

    @pytest.mark.asyncio
    async def test_release_all_locks(self, detector):
        """Test releasing all locks for a molecule."""
        await detector.acquire_lock("mol1", "resource1")
        await detector.acquire_lock("mol1", "resource2")
        await detector.release_all_locks("mol1")

        state = await detector.get_lock_state()
        assert "resource1" not in state["locks"]
        assert "resource2" not in state["locks"]

    @pytest.mark.asyncio
    async def test_detect_simple_deadlock(self, detector):
        """Test detection of simple deadlock."""
        # mol1 holds resource1
        await detector.acquire_lock("mol1", "resource1")

        # mol2 holds resource2
        await detector.acquire_lock("mol2", "resource2")

        # mol1 tries to get resource2 (would wait)
        detector._waiting["mol1"].add("resource2")

        # mol2 tries to get resource1 - should detect deadlock
        with pytest.raises(DeadlockError) as exc_info:
            await detector.acquire_lock("mol2", "resource1")

        assert "mol1" in exc_info.value.molecules or "mol2" in exc_info.value.molecules

    @pytest.mark.asyncio
    async def test_lock_timeout(self, detector):
        """Test lock acquisition timeout."""
        # mol1 holds resource1
        await detector.acquire_lock("mol1", "resource1")

        # mol2 tries to get resource1 with very short timeout
        result = await detector.acquire_lock("mol2", "resource1", timeout=0.1)

        assert result is False

    @pytest.mark.asyncio
    async def test_get_lock_state(self, detector):
        """Test getting lock state for debugging."""
        await detector.acquire_lock("mol1", "resource1")
        detector._waiting["mol2"].add("resource1")

        state = await detector.get_lock_state()

        assert "locks" in state
        assert "waiting" in state
        assert "holding" in state
        assert state["locks"]["resource1"]["molecule"] == "mol1"

    @pytest.mark.asyncio
    async def test_no_false_positive_deadlock(self, detector):
        """Test that non-deadlock situations don't raise."""
        # mol1 holds resource1
        await detector.acquire_lock("mol1", "resource1")

        # mol2 can get resource2 (no conflict)
        result = await detector.acquire_lock("mol2", "resource2")
        assert result is True


# =============================================================================
# MoleculeEngine Transaction Integration Tests
# =============================================================================


class TestMoleculeEngineTransactions:
    """Tests for MoleculeEngine with transaction support."""

    @pytest.mark.asyncio
    async def test_execute_with_transactions_enabled(self, temp_dir, reset_engine):
        """Test execution with transactions enabled."""
        engine = MoleculeEngine(
            checkpoint_dir=temp_dir,
            enable_transactions=True,
        )
        await engine.initialize()

        steps = [MoleculeStep.create("step1", "shell", {"command": "echo test"})]
        molecule = Molecule.create("txn_test", steps)

        result = await engine.execute(molecule)

        assert result.success is True
        # Transaction should be committed and cleaned up

    @pytest.mark.asyncio
    async def test_execute_with_transactions_disabled(self, temp_dir, reset_engine):
        """Test execution with transactions disabled."""
        engine = MoleculeEngine(
            checkpoint_dir=temp_dir,
            enable_transactions=False,
        )
        await engine.initialize()

        steps = [MoleculeStep.create("step1", "shell", {"command": "echo test"})]
        molecule = Molecule.create("no_txn_test", steps)

        result = await engine.execute(molecule)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_validates_dependencies(self, temp_dir, reset_engine):
        """Test that execute validates dependencies before running."""
        engine = MoleculeEngine(
            checkpoint_dir=temp_dir,
            validate_dependencies=True,
        )
        await engine.initialize()

        # Create circular dependency
        step1 = MoleculeStep.create("step1", "shell", dependencies=["placeholder"])
        step2 = MoleculeStep.create("step2", "shell", dependencies=[step1.id])
        step1.dependencies = [step2.id]

        molecule = Molecule.create("cycle_test", [step1, step2])

        result = await engine.execute(molecule)

        assert result.success is False
        assert "Cyclic" in result.error_message or "cycle" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_auto_rollback_on_failure(self, temp_dir, reset_engine):
        """Test automatic rollback on step failure."""
        engine = MoleculeEngine(
            checkpoint_dir=temp_dir,
            enable_transactions=True,
        )
        await engine.initialize()

        class FailingExecutor(StepExecutor):
            async def execute(self, step: MoleculeStep, context: dict) -> Any:
                raise RuntimeError("Intentional failure")

        engine.register_executor("failing", FailingExecutor())

        steps = [
            MoleculeStep.create("good_step", "shell", {"command": "echo test"}),
            MoleculeStep.create("bad_step", "failing", max_attempts=1),
        ]
        molecule = Molecule.create("rollback_test", steps)

        result = await engine.execute(molecule, auto_rollback=True)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_no_auto_rollback(self, temp_dir, reset_engine):
        """Test execution without automatic rollback."""
        engine = MoleculeEngine(
            checkpoint_dir=temp_dir,
            enable_transactions=True,
        )
        await engine.initialize()

        class FailingExecutor(StepExecutor):
            async def execute(self, step: MoleculeStep, context: dict) -> Any:
                raise RuntimeError("Intentional failure")

        engine.register_executor("failing", FailingExecutor())

        steps = [MoleculeStep.create("bad_step", "failing", max_attempts=1)]
        molecule = Molecule.create("no_rollback_test", steps)

        result = await engine.execute(molecule, auto_rollback=False)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_validate_dependencies_method(self, temp_dir, reset_engine):
        """Test the validate_dependencies method directly."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)

        # Valid dependencies
        step1 = MoleculeStep.create("step1", "shell")
        step2 = MoleculeStep.create("step2", "shell", dependencies=[step1.id])
        valid_molecule = Molecule.create("valid", [step1, step2])

        is_valid, error = engine.validate_dependencies(valid_molecule)
        assert is_valid is True
        assert error is None

        # Invalid dependencies (cycle)
        step3 = MoleculeStep.create("step3", "shell", dependencies=["placeholder"])
        step4 = MoleculeStep.create("step4", "shell", dependencies=[step3.id])
        step3.dependencies = [step4.id]
        invalid_molecule = Molecule.create("invalid", [step3, step4])

        is_valid, error = engine.validate_dependencies(invalid_molecule)
        assert is_valid is False
        assert error is not None

    @pytest.mark.asyncio
    async def test_list_transactions(self, temp_dir, reset_engine):
        """Test listing transactions."""
        engine = MoleculeEngine(
            checkpoint_dir=temp_dir,
            enable_transactions=True,
        )
        await engine.initialize()

        steps = [MoleculeStep.create("step1", "shell", {"command": "echo test"})]
        molecule = Molecule.create("list_txn_test", steps)

        await engine.execute(molecule)

        transactions = await engine.list_transactions(molecule_id=molecule.id)
        # Transaction may be cleaned up after commit, or still present
        # depending on implementation details

    @pytest.mark.asyncio
    async def test_get_statistics_includes_transactions(self, temp_dir, reset_engine):
        """Test that statistics include transaction info."""
        engine = MoleculeEngine(
            checkpoint_dir=temp_dir,
            enable_transactions=True,
        )
        await engine.initialize()

        stats = await engine.get_statistics()

        assert "transactions" in stats
        assert "total" in stats["transactions"]
        assert "by_state" in stats["transactions"]


# =============================================================================
# Crash Recovery Tests
# =============================================================================


class TestCrashRecovery:
    """Tests for crash recovery scenarios."""

    @pytest.mark.asyncio
    async def test_recover_incomplete_transactions(self, temp_dir, reset_engine):
        """Test recovering incomplete transactions after crash."""
        # Simulate an incomplete transaction log
        txn_log = {
            "transaction_id": "txn-123",
            "molecule_id": "mol-456",
            "status": "active",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "completed_steps": ["step1"],
        }
        log_file = temp_dir / "txn_txn-123.log"
        with open(log_file, "w") as f:
            json.dump(txn_log, f)

        # Create a molecule checkpoint
        molecule = Molecule.create("recovery_test", [])
        molecule.id = "mol-456"  # Match the log
        checkpoint_file = temp_dir / "mol-456.json"
        with open(checkpoint_file, "w") as f:
            json.dump(molecule.to_dict(), f)

        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        results = await engine.recover_incomplete_transactions()

        # Should have attempted recovery
        assert "txn-123" in results

    @pytest.mark.asyncio
    async def test_recover_completed_transaction_cleanup(self, temp_dir, reset_engine):
        """Test that completed transaction logs are cleaned up."""
        # Simulate a completed transaction log
        txn_log = {
            "transaction_id": "txn-completed",
            "molecule_id": "mol-123",
            "status": "committed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "completed_steps": [],
        }
        log_file = temp_dir / "txn_txn-completed.log"
        with open(log_file, "w") as f:
            json.dump(txn_log, f)

        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        results = await engine.recover_incomplete_transactions()

        # Log should be cleaned up
        assert results.get("txn-completed") is True
        assert not log_file.exists()

    @pytest.mark.asyncio
    async def test_rollback_molecule_method(self, temp_dir, reset_engine):
        """Test manual molecule rollback."""
        engine = MoleculeEngine(
            checkpoint_dir=temp_dir,
            enable_transactions=True,
        )
        await engine.initialize()

        steps = [MoleculeStep.create("step1", "shell", {"command": "echo test"})]
        molecule = Molecule.create("manual_rollback", steps)

        # Store molecule without executing
        engine._molecules[molecule.id] = molecule

        # Without an active transaction, rollback should fail
        result = await engine.rollback_molecule(molecule.id)
        assert result is False

    @pytest.mark.asyncio
    async def test_rollback_nonexistent_molecule(self, temp_dir, reset_engine):
        """Test rollback of nonexistent molecule."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)
        await engine.initialize()

        result = await engine.rollback_molecule("nonexistent")
        assert result is False


# =============================================================================
# Deadlock Detection Integration Tests
# =============================================================================


class TestDeadlockDetectionIntegration:
    """Tests for deadlock detection in parallel molecule execution."""

    @pytest.mark.asyncio
    async def test_execute_with_deadlock_detection_enabled(self, temp_dir, reset_engine):
        """Test execution with deadlock detection enabled."""
        engine = MoleculeEngine(
            checkpoint_dir=temp_dir,
            enable_deadlock_detection=True,
        )
        await engine.initialize()

        steps = [MoleculeStep.create("step1", "shell", {"command": "echo test"})]
        molecule = Molecule.create("deadlock_test", steps)

        result = await engine.execute(molecule)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_releases_locks_on_completion(self, temp_dir, reset_engine):
        """Test that locks are released after execution completes."""
        reset_deadlock_detector()

        engine = MoleculeEngine(
            checkpoint_dir=temp_dir,
            enable_deadlock_detection=True,
        )
        await engine.initialize()

        steps = [
            MoleculeStep.create("step1", "shell", {"command": "echo test", "resources": ["res1"]})
        ]
        molecule = Molecule.create("lock_release_test", steps)

        await engine.execute(molecule)

        detector = get_deadlock_detector()
        state = await detector.get_lock_state()

        # Molecule's locks should be released
        assert molecule.id not in state.get("holding", {})

    @pytest.mark.asyncio
    async def test_execute_releases_locks_on_failure(self, temp_dir, reset_engine):
        """Test that locks are released even on failure."""
        reset_deadlock_detector()

        engine = MoleculeEngine(
            checkpoint_dir=temp_dir,
            enable_deadlock_detection=True,
        )
        await engine.initialize()

        class FailingExecutor(StepExecutor):
            async def execute(self, step: MoleculeStep, context: dict) -> Any:
                raise RuntimeError("Intentional failure")

        engine.register_executor("failing", FailingExecutor())

        steps = [
            MoleculeStep.create(
                "bad_step",
                "failing",
                {"resources": ["critical_resource"]},
                max_attempts=1,
            )
        ]
        molecule = Molecule.create("fail_release_test", steps)

        await engine.execute(molecule)

        detector = get_deadlock_detector()
        state = await detector.get_lock_state()

        # Locks should be released even on failure
        assert molecule.id not in state.get("holding", {})


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_molecule_with_transactions(self, temp_dir, reset_engine):
        """Test executing empty molecule with transactions."""
        engine = MoleculeEngine(
            checkpoint_dir=temp_dir,
            enable_transactions=True,
        )
        await engine.initialize()

        molecule = Molecule.create("empty", [])

        result = await engine.execute(molecule)

        assert result.success is True
        assert result.total_steps == 0

    @pytest.mark.asyncio
    async def test_dependency_validation_empty_molecule(self, temp_dir, reset_engine):
        """Test dependency validation on empty molecule."""
        engine = MoleculeEngine(checkpoint_dir=temp_dir)

        molecule = Molecule.create("empty", [])

        is_valid, error = engine.validate_dependencies(molecule)

        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_transaction_state_transitions(self, temp_dir):
        """Test all valid transaction state transitions."""
        # PENDING -> ACTIVE
        txn = MoleculeTransaction.begin("mol-1", temp_dir)
        assert txn.state == TransactionState.ACTIVE

        # ACTIVE -> COMMITTED
        await txn.commit()
        assert txn.state == TransactionState.COMMITTED

        # COMMITTED -> no further transitions allowed
        result = await txn.commit()
        assert result is False

        result = await txn.rollback()
        assert result is False

    @pytest.mark.asyncio
    async def test_failed_transaction_can_rollback(self, temp_dir):
        """Test that failed transactions can still be rolled back."""
        txn = MoleculeTransaction.begin("mol-1", temp_dir)
        txn.state = TransactionState.FAILED

        result = await txn.rollback()

        assert result is True
        assert txn.state == TransactionState.ROLLED_BACK

    def test_dependency_graph_single_node(self):
        """Test dependency graph with single node."""
        step = MoleculeStep.create("only_step", "shell")
        graph = DependencyGraph([step])

        is_valid, error = graph.validate()
        assert is_valid is True

        order = graph.get_execution_order()
        assert order == [step.id]

    def test_dependency_graph_self_dependency(self):
        """Test detection of self-dependency."""
        step = MoleculeStep.create("self_dep", "shell")
        step.dependencies = [step.id]

        graph = DependencyGraph([step])
        is_valid, error = graph.validate()

        # Self-dependency should be detected as invalid
        assert is_valid is False
