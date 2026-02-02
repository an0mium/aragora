"""
Comprehensive tests for aragora.persistence.transaction module.

Tests cover:
1. Transaction context manager (begin, commit, rollback)
2. ACID guarantees (Atomicity, Consistency, Isolation, Durability)
3. Nested transactions (savepoints)
4. Error handling and automatic rollback
5. Cross-system atomic writes
6. Concurrent transaction handling
7. Statistics and metrics tracking
8. Factory functions and configuration

This test file provides comprehensive coverage complementing
test_transaction_handling.py with additional scenarios for
concurrent operations, cross-system writes, and edge cases.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from aragora.persistence.transaction import (
    DeadlockError,
    NestedTransactionManager,
    SavepointError,
    TransactionConfig,
    TransactionContext,
    TransactionError,
    TransactionIsolation,
    TransactionManager,
    TransactionState,
    TransactionStateError,
    TransactionStats,
    create_transaction_manager,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


def create_mock_connection(conn_id: Optional[int] = None) -> MagicMock:
    """Create a properly configured mock asyncpg connection.

    Args:
        conn_id: Optional connection identifier for tracking in tests
    """
    conn = MagicMock()
    conn.execute = AsyncMock(return_value="OK")
    conn.fetch = AsyncMock(return_value=[{"id": 1}])
    conn.fetchrow = AsyncMock(return_value={"id": 1})
    conn.fetchval = AsyncMock(return_value=1)
    conn._test_id = conn_id or id(conn)
    # Remove auto-generated 'connection' attribute
    del conn.connection
    return conn


def create_mock_pool(connection: Optional[MagicMock] = None) -> MagicMock:
    """Create a mock connection pool."""
    mock_conn = connection or create_mock_connection()
    pool = MagicMock()

    @asynccontextmanager
    async def mock_acquire(readonly: bool = False):
        yield mock_conn

    pool.acquire = mock_acquire
    pool._mock_connection = mock_conn
    return pool


@pytest.fixture
def mock_connection():
    """Create a mock asyncpg connection."""
    return create_mock_connection()


@pytest.fixture
def mock_pool(mock_connection):
    """Create a mock connection pool."""
    return create_mock_pool(mock_connection)


@pytest.fixture
def manager(mock_pool):
    """Create a TransactionManager with mock pool."""
    config = TransactionConfig(validate_connection_state=False)
    return TransactionManager(mock_pool, config)


@pytest.fixture
def config():
    """Create a TransactionConfig for testing."""
    return TransactionConfig(
        isolation=TransactionIsolation.READ_COMMITTED,
        timeout_seconds=30.0,
        deadlock_retries=3,
        deadlock_base_delay=0.01,
        deadlock_max_delay=0.1,
        savepoint_on_nested=True,
        validate_connection_state=False,
    )


# ===========================================================================
# Test Transaction Context Manager
# ===========================================================================


class TestTransactionContextManager:
    """Tests for transaction context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager_begins_transaction(self, manager, mock_connection):
        """Context manager properly begins a transaction."""
        async with manager.transaction() as conn:
            pass

        calls = [str(c) for c in mock_connection.execute.call_args_list]
        assert any("BEGIN" in c for c in calls)

    @pytest.mark.asyncio
    async def test_context_manager_commits_on_success(self, manager, mock_connection):
        """Context manager commits transaction on successful completion."""
        async with manager.transaction() as conn:
            await conn.execute("INSERT INTO test VALUES (1)")

        calls = [str(c) for c in mock_connection.execute.call_args_list]
        assert any("COMMIT" in c for c in calls)

    @pytest.mark.asyncio
    async def test_context_manager_rollback_on_exception(self, manager, mock_connection):
        """Context manager rolls back on exception."""
        with pytest.raises(RuntimeError):
            async with manager.transaction() as conn:
                await conn.execute("INSERT INTO test VALUES (1)")
                raise RuntimeError("Test error")

        calls = [str(c) for c in mock_connection.execute.call_args_list]
        assert any("ROLLBACK" in c for c in calls)

    @pytest.mark.asyncio
    async def test_context_manager_sets_isolation_level(self, manager, mock_connection):
        """Context manager sets the specified isolation level."""
        async with manager.transaction(isolation=TransactionIsolation.SERIALIZABLE) as conn:
            pass

        calls = [str(c) for c in mock_connection.execute.call_args_list]
        assert any("SERIALIZABLE" in c for c in calls)

    @pytest.mark.asyncio
    async def test_context_manager_returns_connection(self, manager, mock_connection):
        """Context manager yields the database connection."""
        async with manager.transaction() as conn:
            assert conn is mock_connection

    @pytest.mark.asyncio
    async def test_context_manager_updates_stats(self, manager, mock_connection):
        """Context manager updates transaction statistics."""
        assert manager.get_stats().transactions_started == 0
        assert manager.get_stats().transactions_committed == 0

        async with manager.transaction() as conn:
            pass

        assert manager.get_stats().transactions_started == 1
        assert manager.get_stats().transactions_committed == 1


# ===========================================================================
# Test ACID Guarantees
# ===========================================================================


class TestACIDGuarantees:
    """Tests verifying ACID guarantee behaviors."""

    @pytest.mark.asyncio
    async def test_atomicity_all_or_nothing(self, manager, mock_connection):
        """Atomicity: All operations succeed or all are rolled back."""
        operations_executed = []

        async def tracking_execute(query, *args, **kwargs):
            operations_executed.append(query)
            if "ROLLBACK" in query:
                return "OK"
            if len(operations_executed) >= 4:  # Fail on 4th operation
                raise ValueError("Simulated failure")
            return "OK"

        mock_connection.execute = AsyncMock(side_effect=tracking_execute)

        with pytest.raises(ValueError):
            async with manager.transaction() as conn:
                await conn.execute("INSERT INTO users VALUES (1)")
                await conn.execute("INSERT INTO orders VALUES (1)")
                await conn.execute("INSERT INTO payments VALUES (1)")  # This fails

        # Verify rollback was called after failure
        assert any("ROLLBACK" in op for op in operations_executed)

    @pytest.mark.asyncio
    async def test_isolation_level_read_committed(self, mock_connection):
        """Isolation: READ COMMITTED level is set correctly."""
        pool = create_mock_pool(mock_connection)
        config = TransactionConfig(
            isolation=TransactionIsolation.READ_COMMITTED,
            validate_connection_state=False,
        )
        manager = TransactionManager(pool, config)

        async with manager.transaction() as conn:
            pass

        calls = [str(c) for c in mock_connection.execute.call_args_list]
        assert any("READ COMMITTED" in c for c in calls)

    @pytest.mark.asyncio
    async def test_isolation_level_repeatable_read(self, mock_connection):
        """Isolation: REPEATABLE READ level is set correctly."""
        pool = create_mock_pool(mock_connection)
        config = TransactionConfig(
            isolation=TransactionIsolation.REPEATABLE_READ,
            validate_connection_state=False,
        )
        manager = TransactionManager(pool, config)

        async with manager.transaction() as conn:
            pass

        calls = [str(c) for c in mock_connection.execute.call_args_list]
        assert any("REPEATABLE READ" in c for c in calls)

    @pytest.mark.asyncio
    async def test_isolation_level_serializable(self, mock_connection):
        """Isolation: SERIALIZABLE level is set correctly."""
        pool = create_mock_pool(mock_connection)
        config = TransactionConfig(
            isolation=TransactionIsolation.SERIALIZABLE,
            validate_connection_state=False,
        )
        manager = TransactionManager(pool, config)

        async with manager.transaction() as conn:
            pass

        calls = [str(c) for c in mock_connection.execute.call_args_list]
        assert any("SERIALIZABLE" in c for c in calls)

    @pytest.mark.asyncio
    async def test_isolation_per_transaction_override(self, manager, mock_connection):
        """Isolation level can be overridden per transaction."""
        async with manager.transaction(isolation=TransactionIsolation.SERIALIZABLE) as conn:
            pass

        async with manager.transaction(isolation=TransactionIsolation.READ_COMMITTED) as conn:
            pass

        calls = [str(c) for c in mock_connection.execute.call_args_list]
        assert any("SERIALIZABLE" in c for c in calls)
        assert any("READ COMMITTED" in c for c in calls)

    @pytest.mark.asyncio
    async def test_durability_commit_persists(self, manager, mock_connection):
        """Durability: COMMIT is issued to persist changes."""
        async with manager.transaction() as conn:
            await conn.execute("INSERT INTO test VALUES (1)")

        # Verify COMMIT was the last significant command
        calls = [str(c) for c in mock_connection.execute.call_args_list]
        commit_indices = [i for i, c in enumerate(calls) if "COMMIT" in c]
        assert len(commit_indices) > 0


# ===========================================================================
# Test Nested Transactions (Savepoints)
# ===========================================================================


class TestNestedTransactions:
    """Tests for nested transactions using savepoints."""

    @pytest.mark.asyncio
    async def test_savepoint_creation(self, manager, mock_connection):
        """Savepoints are created within transactions."""
        async with manager.transaction() as conn:
            async with manager.savepoint(conn, "sp1"):
                await conn.execute("INSERT INTO test VALUES (1)")

        calls = [str(c) for c in mock_connection.execute.call_args_list]
        assert any("SAVEPOINT sp1" in c for c in calls)

    @pytest.mark.asyncio
    async def test_savepoint_release_on_success(self, manager, mock_connection):
        """Savepoints are released on successful completion."""
        async with manager.transaction() as conn:
            async with manager.savepoint(conn, "sp1"):
                await conn.execute("INSERT INTO test VALUES (1)")

        calls = [str(c) for c in mock_connection.execute.call_args_list]
        assert any("RELEASE SAVEPOINT sp1" in c for c in calls)

    @pytest.mark.asyncio
    async def test_savepoint_rollback_on_error(self, manager, mock_connection):
        """Savepoints are rolled back on error, outer transaction continues."""
        async with manager.transaction() as conn:
            await conn.execute("INSERT INTO outer VALUES (1)")

            with pytest.raises(ValueError):
                async with manager.savepoint(conn, "sp1"):
                    await conn.execute("INSERT INTO inner VALUES (1)")
                    raise ValueError("Inner error")

            # Outer transaction continues
            await conn.execute("INSERT INTO outer VALUES (2)")

        calls = [str(c) for c in mock_connection.execute.call_args_list]
        assert any("ROLLBACK TO SAVEPOINT sp1" in c for c in calls)
        assert any("COMMIT" in c for c in calls)

    @pytest.mark.asyncio
    async def test_nested_savepoints_multiple_levels(self, manager, mock_connection):
        """Multiple levels of nested savepoints work correctly."""
        async with manager.transaction() as conn:
            await conn.execute("Level 0")

            async with manager.savepoint(conn, "sp_level_1"):
                await conn.execute("Level 1")

                async with manager.savepoint(conn, "sp_level_2"):
                    await conn.execute("Level 2")

                    async with manager.savepoint(conn, "sp_level_3"):
                        await conn.execute("Level 3")

        calls = [str(c) for c in mock_connection.execute.call_args_list]
        assert any("SAVEPOINT sp_level_1" in c for c in calls)
        assert any("SAVEPOINT sp_level_2" in c for c in calls)
        assert any("SAVEPOINT sp_level_3" in c for c in calls)
        assert any("RELEASE SAVEPOINT sp_level_3" in c for c in calls)
        assert any("RELEASE SAVEPOINT sp_level_2" in c for c in calls)
        assert any("RELEASE SAVEPOINT sp_level_1" in c for c in calls)

    @pytest.mark.asyncio
    async def test_partial_rollback_with_savepoints(self, manager, mock_connection):
        """Partial rollback using savepoints preserves outer work."""
        async with manager.transaction() as conn:
            await conn.execute("INSERT INTO users VALUES (1)")

            with pytest.raises(ValueError):
                async with manager.savepoint(conn, "optional_work"):
                    await conn.execute("INSERT INTO optional VALUES (1)")
                    raise ValueError("Optional work failed")

            # Continue with other work
            await conn.execute("INSERT INTO users VALUES (2)")

        # Transaction should commit despite inner failure
        stats = manager.get_stats()
        assert stats.transactions_committed == 1
        assert stats.savepoints_rolled_back == 1

    @pytest.mark.asyncio
    async def test_invalid_savepoint_name_rejected(self, manager, mock_connection):
        """Invalid savepoint names are rejected."""
        async with manager.transaction() as conn:
            with pytest.raises(SavepointError, match="Invalid savepoint name"):
                async with manager.savepoint(conn, "invalid;name"):
                    pass

    @pytest.mark.asyncio
    async def test_savepoint_name_with_numbers_and_underscores(self, manager, mock_connection):
        """Savepoint names with numbers and underscores are valid."""
        async with manager.transaction() as conn:
            async with manager.savepoint(conn, "save_point_123"):
                pass

        stats = manager.get_stats()
        assert stats.savepoints_released == 1


# ===========================================================================
# Test Error Handling and Automatic Rollback
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling and automatic rollback behavior."""

    @pytest.mark.asyncio
    async def test_automatic_rollback_on_exception(self, manager, mock_connection):
        """Transactions automatically roll back on exceptions."""
        with pytest.raises(ValueError):
            async with manager.transaction() as conn:
                await conn.execute("INSERT INTO test VALUES (1)")
                raise ValueError("Test error")

        calls = [str(c) for c in mock_connection.execute.call_args_list]
        assert any("ROLLBACK" in c for c in calls)
        assert manager.get_stats().transactions_failed == 1

    @pytest.mark.asyncio
    async def test_rollback_on_asyncio_cancelled_error(self, manager, mock_connection):
        """Transactions roll back on asyncio.CancelledError."""
        with pytest.raises(asyncio.CancelledError):
            async with manager.transaction() as conn:
                raise asyncio.CancelledError()

        calls = [str(c) for c in mock_connection.execute.call_args_list]
        assert any("ROLLBACK" in c for c in calls)
        assert manager.get_stats().transactions_rolled_back == 1

    @pytest.mark.asyncio
    async def test_rollback_on_timeout(self):
        """Transactions roll back on timeout."""
        mock_conn = create_mock_connection()

        async def slow_operation(query, *args, **kwargs):
            if "SLOW" in query:
                await asyncio.sleep(5.0)
            return "OK"

        mock_conn.execute = AsyncMock(side_effect=slow_operation)

        pool = create_mock_pool(mock_conn)
        config = TransactionConfig(timeout_seconds=0.1, validate_connection_state=False)
        manager = TransactionManager(pool, config)

        with pytest.raises(asyncio.TimeoutError):
            async with manager.transaction() as conn:
                await conn.execute("SLOW QUERY")

        assert manager.get_stats().transactions_rolled_back == 1

    @pytest.mark.asyncio
    async def test_rollback_failure_handled_gracefully(self, mock_connection):
        """Rollback failure is handled gracefully."""
        call_count = [0]

        async def failing_rollback(query, *args, **kwargs):
            call_count[0] += 1
            if "ROLLBACK" in query:
                raise ConnectionError("Connection lost")
            if call_count[0] >= 4:  # Fail on 4th operation
                raise ValueError("Original error")
            return "OK"

        mock_connection.execute = AsyncMock(side_effect=failing_rollback)

        pool = create_mock_pool(mock_connection)
        config = TransactionConfig(validate_connection_state=False)
        manager = TransactionManager(pool, config)

        # Should raise the original error, not the rollback error
        with pytest.raises(ValueError, match="Original error"):
            async with manager.transaction() as conn:
                await conn.execute("INSERT 1")
                await conn.execute("INSERT 2")
                await conn.execute("INSERT 3")  # This triggers failure

    @pytest.mark.asyncio
    async def test_deadlock_error_detection(self):
        """Deadlock errors are detected and wrapped."""
        mock_conn = create_mock_connection()

        async def deadlock_error(query, *args, **kwargs):
            if "INSERT" in query:
                raise Exception("ERROR: deadlock detected")
            return "OK"

        mock_conn.execute = AsyncMock(side_effect=deadlock_error)

        pool = create_mock_pool(mock_conn)
        config = TransactionConfig(validate_connection_state=False)
        manager = TransactionManager(pool, config)

        with pytest.raises(DeadlockError):
            async with manager.transaction() as conn:
                await conn.execute("INSERT INTO test VALUES (1)")

        assert manager.get_stats().deadlocks_detected == 1

    @pytest.mark.asyncio
    async def test_serialization_failure_detected_as_deadlock(self):
        """Serialization failures are treated as deadlocks."""
        mock_conn = create_mock_connection()

        async def serialization_error(query, *args, **kwargs):
            if "UPDATE" in query:
                raise Exception("ERROR 40001: could not serialize access")
            return "OK"

        mock_conn.execute = AsyncMock(side_effect=serialization_error)

        pool = create_mock_pool(mock_conn)
        config = TransactionConfig(validate_connection_state=False)
        manager = TransactionManager(pool, config)

        with pytest.raises(DeadlockError):
            async with manager.transaction() as conn:
                await conn.execute("UPDATE test SET x = 1")

        assert manager.get_stats().deadlocks_detected == 1

    @pytest.mark.asyncio
    async def test_connection_validation_failure(self):
        """Connection validation failure raises TransactionError."""
        mock_conn = create_mock_connection()
        mock_conn.fetchval = AsyncMock(side_effect=ConnectionError("No connection"))

        pool = create_mock_pool(mock_conn)
        config = TransactionConfig(validate_connection_state=True)
        manager = TransactionManager(pool, config)

        with pytest.raises(TransactionError, match="validation failed"):
            async with manager.transaction() as conn:
                pass


# ===========================================================================
# Test Cross-System Atomic Writes
# ===========================================================================


class TestCrossSystemAtomicWrites:
    """Tests for cross-system atomic write patterns."""

    @pytest.mark.asyncio
    async def test_multiple_table_atomic_write(self):
        """Multiple table operations are atomic within a transaction."""
        operations = []

        mock_conn = create_mock_connection()

        async def tracking_execute(query, *args, **kwargs):
            operations.append(query)
            return "OK"

        mock_conn.execute = AsyncMock(side_effect=tracking_execute)

        pool = create_mock_pool(mock_conn)
        config = TransactionConfig(validate_connection_state=False)
        manager = TransactionManager(pool, config)

        async with manager.transaction() as conn:
            await conn.execute("INSERT INTO users VALUES (1)")
            await conn.execute("INSERT INTO profiles VALUES (1)")
            await conn.execute("INSERT INTO settings VALUES (1)")

        # All operations should be between BEGIN and COMMIT
        # Note: SET TRANSACTION ISOLATION LEVEL comes before BEGIN
        begin_idx = next(i for i, op in enumerate(operations) if op == "BEGIN")
        commit_idx = next(i for i, op in enumerate(operations) if op == "COMMIT")

        user_idx = next(i for i, op in enumerate(operations) if "users" in op)
        profile_idx = next(i for i, op in enumerate(operations) if "profiles" in op)
        settings_idx = next(i for i, op in enumerate(operations) if "settings" in op)

        assert begin_idx < user_idx < commit_idx
        assert begin_idx < profile_idx < commit_idx
        assert begin_idx < settings_idx < commit_idx

    @pytest.mark.asyncio
    async def test_cross_system_rollback_on_partial_failure(self, manager, mock_connection):
        """Cross-system writes roll back completely on partial failure."""
        call_count = [0]
        operations = []

        async def partial_failure(query, *args, **kwargs):
            call_count[0] += 1
            operations.append(query)
            if "system_c" in query:
                raise ValueError("System C failure")
            return "OK"

        mock_connection.execute = AsyncMock(side_effect=partial_failure)

        with pytest.raises(ValueError):
            async with manager.transaction() as conn:
                await conn.execute("INSERT INTO system_a VALUES (1)")
                await conn.execute("INSERT INTO system_b VALUES (1)")
                await conn.execute("INSERT INTO system_c VALUES (1)")  # Fails

        # Verify rollback was called
        assert any("ROLLBACK" in op for op in operations)

    @pytest.mark.asyncio
    async def test_savepoint_for_optional_systems(self, manager, mock_connection):
        """Use savepoints for optional cross-system writes."""
        async with manager.transaction() as conn:
            # Required write
            await conn.execute("INSERT INTO required_system VALUES (1)")

            # Optional write with savepoint
            try:
                async with manager.savepoint(conn, "optional_systems"):
                    await conn.execute("INSERT INTO optional_system VALUES (1)")
                    raise ValueError("Optional system unavailable")
            except ValueError:
                pass  # Continue without optional system

            # Another required write
            await conn.execute("INSERT INTO required_system VALUES (2)")

        # Transaction should succeed
        stats = manager.get_stats()
        assert stats.transactions_committed == 1
        assert stats.savepoints_rolled_back == 1


# ===========================================================================
# Test Concurrent Transaction Handling
# ===========================================================================


class TestConcurrentTransactions:
    """Tests for concurrent transaction handling."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_transactions(self):
        """Multiple concurrent transactions operate independently."""
        mock_conn = create_mock_connection()
        pool = create_mock_pool(mock_conn)
        config = TransactionConfig(validate_connection_state=False)
        manager = TransactionManager(pool, config)

        results = []

        async def transaction_task(task_id: int):
            async with manager.transaction() as conn:
                await conn.execute(f"INSERT INTO test VALUES ({task_id})")
                await asyncio.sleep(0.01)  # Simulate some work
                results.append(task_id)
                return task_id

        # Run 10 concurrent transactions
        task_results = await asyncio.gather(*[transaction_task(i) for i in range(10)])

        assert len(task_results) == 10
        assert manager.get_stats().transactions_committed == 10

    @pytest.mark.asyncio
    async def test_concurrent_transactions_with_failures(self):
        """Concurrent transactions handle individual failures correctly."""
        mock_conn = create_mock_connection()

        async def sometimes_fails(query, *args, **kwargs):
            if "FAIL_5" in query:
                raise ValueError("Intentional failure for task 5")
            return "OK"

        mock_conn.execute = AsyncMock(side_effect=sometimes_fails)

        pool = create_mock_pool(mock_conn)
        config = TransactionConfig(validate_connection_state=False)
        manager = TransactionManager(pool, config)

        async def transaction_task(task_id: int):
            try:
                async with manager.transaction() as conn:
                    query = f"INSERT FAIL_{task_id}" if task_id == 5 else f"INSERT {task_id}"
                    await conn.execute(query)
                    return ("success", task_id)
            except ValueError:
                return ("failed", task_id)

        results = await asyncio.gather(*[transaction_task(i) for i in range(10)])

        success_count = sum(1 for r in results if r[0] == "success")
        failure_count = sum(1 for r in results if r[0] == "failed")

        assert success_count == 9
        assert failure_count == 1

    @pytest.mark.asyncio
    async def test_active_transaction_count_tracking(self):
        """Active transaction count is tracked correctly during concurrency."""
        mock_conn = create_mock_connection()
        pool = create_mock_pool(mock_conn)
        config = TransactionConfig(validate_connection_state=False)
        manager = TransactionManager(pool, config)

        max_active = [0]
        barrier = asyncio.Event()

        async def transaction_task():
            async with manager.transaction() as conn:
                current_active = manager.active_transaction_count
                max_active[0] = max(max_active[0], current_active)
                # Wait until all transactions are active
                if current_active < 5:
                    await asyncio.sleep(0.01)
                return True

        tasks = [transaction_task() for _ in range(5)]
        await asyncio.gather(*tasks)

        # After all complete, active count should be 0
        assert manager.active_transaction_count == 0

    @pytest.mark.asyncio
    async def test_concurrent_savepoints(self):
        """Concurrent transactions with savepoints work correctly."""
        mock_conn = create_mock_connection()
        pool = create_mock_pool(mock_conn)
        config = TransactionConfig(validate_connection_state=False)
        manager = TransactionManager(pool, config)

        async def transaction_with_savepoint(task_id: int):
            async with manager.transaction() as conn:
                await conn.execute(f"INSERT outer {task_id}")

                async with manager.savepoint(conn, f"sp_{task_id}"):
                    await conn.execute(f"INSERT inner {task_id}")

                return task_id

        results = await asyncio.gather(*[transaction_with_savepoint(i) for i in range(5)])

        assert len(results) == 5
        stats = manager.get_stats()
        assert stats.transactions_committed == 5
        assert stats.savepoints_released == 5


# ===========================================================================
# Test Manual Transaction Control
# ===========================================================================


class TestManualTransactionControl:
    """Tests for manual begin/commit/rollback methods."""

    @pytest.mark.asyncio
    async def test_manual_begin_commit(self, manager, mock_connection):
        """Manual begin and commit work correctly."""
        await manager.begin(mock_connection)
        await mock_connection.execute("INSERT INTO test VALUES (1)")
        await manager.commit(mock_connection)

        calls = [str(c) for c in mock_connection.execute.call_args_list]
        assert any("BEGIN" in c for c in calls)
        assert any("COMMIT" in c for c in calls)

        stats = manager.get_stats()
        assert stats.transactions_started == 1
        assert stats.transactions_committed == 1

    @pytest.mark.asyncio
    async def test_manual_begin_rollback(self, manager, mock_connection):
        """Manual begin and rollback work correctly."""
        await manager.begin(mock_connection)
        await mock_connection.execute("INSERT INTO test VALUES (1)")
        await manager.rollback(mock_connection)

        calls = [str(c) for c in mock_connection.execute.call_args_list]
        assert any("ROLLBACK" in c for c in calls)

        stats = manager.get_stats()
        assert stats.transactions_started == 1
        assert stats.transactions_rolled_back == 1

    @pytest.mark.asyncio
    async def test_manual_begin_with_custom_isolation(self, manager, mock_connection):
        """Manual begin respects custom isolation level."""
        await manager.begin(mock_connection, isolation=TransactionIsolation.SERIALIZABLE)

        calls = [str(c) for c in mock_connection.execute.call_args_list]
        assert any("SERIALIZABLE" in c for c in calls)


# ===========================================================================
# Test Statistics and Metrics
# ===========================================================================


class TestStatisticsAndMetrics:
    """Tests for statistics and metrics tracking."""

    def test_stats_default_values(self):
        """TransactionStats has correct default values."""
        stats = TransactionStats()

        assert stats.transactions_started == 0
        assert stats.transactions_committed == 0
        assert stats.transactions_rolled_back == 0
        assert stats.transactions_failed == 0
        assert stats.savepoints_created == 0
        assert stats.savepoints_released == 0
        assert stats.savepoints_rolled_back == 0
        assert stats.deadlocks_detected == 0
        assert stats.deadlocks_recovered == 0
        assert stats.active_transactions == 0
        assert stats.total_transaction_time_ms == 0.0

    def test_stats_to_dict(self):
        """TransactionStats to_dict includes all fields."""
        stats = TransactionStats(
            transactions_started=10,
            transactions_committed=8,
            transactions_rolled_back=1,
            transactions_failed=1,
            total_transaction_time_ms=5000.0,
        )

        result = stats.to_dict()

        assert result["transactions_started"] == 10
        assert result["transactions_committed"] == 8
        assert result["transactions_rolled_back"] == 1
        assert result["transactions_failed"] == 1
        assert result["total_transaction_time_ms"] == 5000.0
        assert "average_transaction_time_ms" in result

    def test_stats_average_calculation(self):
        """Average transaction time is calculated correctly."""
        stats = TransactionStats(
            transactions_committed=9,
            transactions_rolled_back=1,
            total_transaction_time_ms=10000.0,
        )

        result = stats.to_dict()

        # Average = 10000 / (9 + 1) = 1000
        assert result["average_transaction_time_ms"] == 1000.0

    def test_stats_average_with_zero_transactions(self):
        """Average calculation handles zero transactions."""
        stats = TransactionStats()

        result = stats.to_dict()

        assert result["average_transaction_time_ms"] == 0.0

    @pytest.mark.asyncio
    async def test_transaction_time_tracking(self, manager, mock_connection):
        """Transaction time is tracked in statistics."""
        async with manager.transaction() as conn:
            await asyncio.sleep(0.05)  # 50ms

        stats = manager.get_stats()
        assert stats.total_transaction_time_ms >= 50.0

    def test_reset_stats(self, manager):
        """Statistics can be reset."""
        manager._stats.transactions_started = 100
        manager._stats.transactions_committed = 99

        manager.reset_stats()

        stats = manager.get_stats()
        assert stats.transactions_started == 0
        assert stats.transactions_committed == 0


# ===========================================================================
# Test Transaction Context
# ===========================================================================


class TestTransactionContext:
    """Tests for TransactionContext dataclass."""

    def test_context_is_active(self):
        """is_active returns correct state."""
        ctx = TransactionContext(
            id="txn_1",
            connection=MagicMock(),
            isolation=TransactionIsolation.READ_COMMITTED,
            state=TransactionState.ACTIVE,
        )

        assert ctx.is_active is True

        ctx.state = TransactionState.COMMITTED
        assert ctx.is_active is False

        ctx.state = TransactionState.ROLLED_BACK
        assert ctx.is_active is False

        ctx.state = TransactionState.FAILED
        assert ctx.is_active is False

    def test_context_duration_calculation(self):
        """duration_ms calculates elapsed time correctly."""
        ctx = TransactionContext(
            id="txn_1",
            connection=MagicMock(),
            isolation=TransactionIsolation.READ_COMMITTED,
            started_at=time.time() - 1.0,  # 1 second ago
        )

        duration = ctx.duration_ms
        assert 1000.0 <= duration < 2000.0

    def test_context_savepoint_stack(self):
        """Savepoint stack is properly managed."""
        ctx = TransactionContext(
            id="txn_1",
            connection=MagicMock(),
            isolation=TransactionIsolation.READ_COMMITTED,
        )

        assert ctx.savepoint_stack == []

        ctx.savepoint_stack.append("sp1")
        ctx.savepoint_stack.append("sp2")

        assert len(ctx.savepoint_stack) == 2
        assert ctx.savepoint_stack == ["sp1", "sp2"]


# ===========================================================================
# Test Nested Transaction Manager
# ===========================================================================


class TestNestedTransactionManager:
    """Tests for NestedTransactionManager."""

    @pytest.mark.asyncio
    async def test_nested_manager_initialization(self):
        """NestedTransactionManager initializes correctly."""
        pool = create_mock_pool()
        manager = NestedTransactionManager(pool)

        assert manager._connection_depth == {}

    @pytest.mark.asyncio
    async def test_nested_manager_tracks_depth(self):
        """NestedTransactionManager tracks connection depth."""
        mock_conn = create_mock_connection()
        pool = create_mock_pool(mock_conn)
        config = TransactionConfig(validate_connection_state=False)
        manager = NestedTransactionManager(pool, config)

        # The depth tracking happens inside the transaction
        async with manager.transaction() as conn:
            pass

        # After transaction completes, depth should be reset
        assert manager._connection_depth.get(id(mock_conn), 0) == 0


# ===========================================================================
# Test Factory Function
# ===========================================================================


class TestFactoryFunction:
    """Tests for create_transaction_manager factory."""

    def test_creates_standard_manager(self):
        """Factory creates TransactionManager with defaults."""
        pool = create_mock_pool()
        manager = create_transaction_manager(pool)

        assert isinstance(manager, TransactionManager)
        assert not isinstance(manager, NestedTransactionManager)

    def test_creates_manager_with_custom_config(self):
        """Factory creates manager with custom configuration."""
        pool = create_mock_pool()
        manager = create_transaction_manager(
            pool,
            isolation=TransactionIsolation.SERIALIZABLE,
            timeout_seconds=60.0,
            deadlock_retries=5,
        )

        assert manager.config.isolation == TransactionIsolation.SERIALIZABLE
        assert manager.config.timeout_seconds == 60.0
        assert manager.config.deadlock_retries == 5

    def test_creates_nested_manager(self):
        """Factory creates NestedTransactionManager when requested."""
        pool = create_mock_pool()
        manager = create_transaction_manager(pool, nested_support=True)

        assert isinstance(manager, NestedTransactionManager)


# ===========================================================================
# Test Exception Classes
# ===========================================================================


class TestExceptionClasses:
    """Tests for custom exception classes."""

    def test_deadlock_error_attributes(self):
        """DeadlockError stores retry count and message."""
        error = DeadlockError("Deadlock detected", retry_count=3)

        assert error.retry_count == 3
        assert "Deadlock detected" in str(error)

    def test_transaction_state_error_attributes(self):
        """TransactionStateError stores state information."""
        error = TransactionStateError(
            expected_state="active",
            actual_state="committed",
            operation="execute",
        )

        assert error.expected_state == "active"
        assert error.actual_state == "committed"
        assert error.operation == "execute"
        assert "execute" in str(error)
        assert "active" in str(error)
        assert "committed" in str(error)

    def test_savepoint_error_inheritance(self):
        """SavepointError inherits from TransactionError."""
        error = SavepointError("Invalid savepoint")

        assert isinstance(error, TransactionError)
        assert isinstance(error, Exception)

    def test_transaction_error_base_class(self):
        """TransactionError is a proper base exception."""
        error = TransactionError("Generic transaction error")

        assert isinstance(error, Exception)
        assert "Generic transaction error" in str(error)


# ===========================================================================
# Test Configuration
# ===========================================================================


class TestConfiguration:
    """Tests for TransactionConfig."""

    def test_default_configuration(self):
        """TransactionConfig has sensible defaults."""
        config = TransactionConfig()

        assert config.isolation == TransactionIsolation.READ_COMMITTED
        assert config.timeout_seconds == 30.0
        assert config.deadlock_retries == 3
        assert config.deadlock_base_delay == 0.1
        assert config.deadlock_max_delay == 2.0
        assert config.savepoint_on_nested is True
        assert config.validate_connection_state is True

    def test_custom_configuration(self):
        """TransactionConfig accepts custom values."""
        config = TransactionConfig(
            isolation=TransactionIsolation.SERIALIZABLE,
            timeout_seconds=120.0,
            deadlock_retries=10,
            deadlock_base_delay=0.5,
            deadlock_max_delay=5.0,
            savepoint_on_nested=False,
            validate_connection_state=False,
        )

        assert config.isolation == TransactionIsolation.SERIALIZABLE
        assert config.timeout_seconds == 120.0
        assert config.deadlock_retries == 10
        assert config.deadlock_base_delay == 0.5
        assert config.deadlock_max_delay == 5.0
        assert config.savepoint_on_nested is False
        assert config.validate_connection_state is False

    def test_config_property_access(self, manager):
        """Manager config property returns configuration."""
        config = manager.config

        assert isinstance(config, TransactionConfig)


# ===========================================================================
# Test Isolation Levels Enum
# ===========================================================================


class TestIsolationLevelsEnum:
    """Tests for TransactionIsolation enum."""

    def test_isolation_values(self):
        """TransactionIsolation has correct SQL values."""
        assert TransactionIsolation.READ_COMMITTED.value == "READ COMMITTED"
        assert TransactionIsolation.REPEATABLE_READ.value == "REPEATABLE READ"
        assert TransactionIsolation.SERIALIZABLE.value == "SERIALIZABLE"

    def test_isolation_is_string_enum(self):
        """TransactionIsolation can be compared to strings."""
        assert TransactionIsolation.READ_COMMITTED == "READ COMMITTED"
        assert TransactionIsolation.SERIALIZABLE == "SERIALIZABLE"


# ===========================================================================
# Test Transaction State Enum
# ===========================================================================


class TestTransactionStateEnum:
    """Tests for TransactionState enum."""

    def test_all_states_exist(self):
        """All expected transaction states are defined."""
        assert TransactionState.INACTIVE == "inactive"
        assert TransactionState.ACTIVE == "active"
        assert TransactionState.COMMITTED == "committed"
        assert TransactionState.ROLLED_BACK == "rolled_back"
        assert TransactionState.FAILED == "failed"


# ===========================================================================
# Test Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_pool_without_acquire_method(self):
        """Pool without acquire method raises TransactionError."""
        pool = MagicMock(spec=[])  # No acquire method

        manager = TransactionManager(pool)

        with pytest.raises(TransactionError, match="does not support acquire"):
            async with manager.transaction() as conn:
                pass

    @pytest.mark.asyncio
    async def test_wrapped_connection_unwrapped(self):
        """Wrapped connections are properly unwrapped."""
        inner_conn = create_mock_connection()
        wrapper = MagicMock()
        wrapper.connection = inner_conn

        @asynccontextmanager
        async def mock_acquire(readonly: bool = False):
            yield wrapper

        pool = MagicMock()
        pool.acquire = mock_acquire

        config = TransactionConfig(validate_connection_state=False)
        manager = TransactionManager(pool, config)

        async with manager.transaction() as conn:
            await conn.execute("SELECT 1")

        # The inner connection should be used
        inner_conn.execute.assert_called()

    @pytest.mark.asyncio
    async def test_empty_transaction_commits(self, manager, mock_connection):
        """Empty transaction (no operations) still commits."""
        async with manager.transaction() as conn:
            pass  # No operations

        calls = [str(c) for c in mock_connection.execute.call_args_list]
        assert any("COMMIT" in c for c in calls)

    @pytest.mark.asyncio
    async def test_transaction_id_generation(self, manager, mock_connection):
        """Transaction IDs are unique and incrementing."""
        initial_counter = manager._transaction_counter

        for _ in range(5):
            async with manager.transaction() as conn:
                pass

        assert manager._transaction_counter == initial_counter + 5

    @pytest.mark.asyncio
    async def test_readonly_flag_passed_to_pool(self):
        """readonly flag is passed to pool.acquire."""
        mock_conn = create_mock_connection()
        acquire_calls = []

        @asynccontextmanager
        async def tracking_acquire(readonly: bool = False):
            acquire_calls.append(readonly)
            yield mock_conn

        pool = MagicMock()
        pool.acquire = tracking_acquire

        config = TransactionConfig(validate_connection_state=False)
        manager = TransactionManager(pool, config)

        async with manager.transaction(readonly=True) as conn:
            pass

        async with manager.transaction(readonly=False) as conn:
            pass

        assert acquire_calls == [True, False]

    @pytest.mark.asyncio
    async def test_zero_timeout_means_no_timeout(self):
        """Timeout of 0 disables timeout."""
        mock_conn = create_mock_connection()
        pool = create_mock_pool(mock_conn)
        config = TransactionConfig(timeout_seconds=0, validate_connection_state=False)
        manager = TransactionManager(pool, config)

        async with manager.transaction() as conn:
            await conn.execute("SELECT 1")

        assert manager.get_stats().transactions_committed == 1

    @pytest.mark.asyncio
    async def test_timeout_override_per_transaction(self):
        """Timeout can be overridden per transaction."""
        mock_conn = create_mock_connection()
        pool = create_mock_pool(mock_conn)
        config = TransactionConfig(timeout_seconds=1.0, validate_connection_state=False)
        manager = TransactionManager(pool, config)

        # Use a longer timeout for this transaction
        async with manager.transaction(timeout=60.0) as conn:
            await conn.execute("SELECT 1")

        assert manager.get_stats().transactions_committed == 1


# ===========================================================================
# Test Retry Delay Calculation
# ===========================================================================


class TestRetryDelayCalculation:
    """Tests for deadlock retry delay calculation."""

    def test_exponential_backoff(self, manager):
        """Retry delay uses exponential backoff."""
        delays = []
        for attempt in range(5):
            delay = manager._calculate_retry_delay(attempt)
            delays.append(delay)

        # Each delay should generally be larger than the previous
        # (accounting for jitter)
        assert delays[4] > delays[0]  # Last delay > first delay

    def test_delay_respects_max(self):
        """Retry delay is capped at max delay."""
        pool = create_mock_pool()
        config = TransactionConfig(
            deadlock_base_delay=0.1,
            deadlock_max_delay=1.0,
            validate_connection_state=False,
        )
        manager = TransactionManager(pool, config)

        # With high attempt, delay should be capped
        delay = manager._calculate_retry_delay(100)

        # Max delay is 1.0, with jitter it's between 0.75 and 1.25
        assert delay <= 1.25

    def test_delay_has_jitter(self, manager):
        """Retry delay includes jitter."""
        delays = set()
        for _ in range(10):
            delay = manager._calculate_retry_delay(2)
            delays.add(round(delay, 6))

        # With jitter, we should have some variation
        assert len(delays) > 1


# ===========================================================================
# Test Deadlock Detection
# ===========================================================================


class TestDeadlockDetection:
    """Tests for deadlock error detection logic."""

    def test_detects_deadlock_keyword(self, manager):
        """Detects 'deadlock' keyword in error."""
        error = Exception("ERROR: deadlock detected on relation users")
        assert manager._is_deadlock_error(error) is True

    def test_detects_40001_code(self, manager):
        """Detects PostgreSQL error code 40001."""
        error = Exception("ERROR 40001: could not serialize access")
        assert manager._is_deadlock_error(error) is True

    def test_detects_40p01_code(self, manager):
        """Detects PostgreSQL error code 40P01."""
        error = Exception("ERROR 40P01: deadlock detected")
        assert manager._is_deadlock_error(error) is True

    def test_detects_serialization_keyword(self, manager):
        """Detects 'serialization' keyword in error."""
        error = Exception("serialization failure during transaction")
        assert manager._is_deadlock_error(error) is True

    def test_non_deadlock_error(self, manager):
        """Does not detect non-deadlock errors."""
        error = Exception("Constraint violation")
        assert manager._is_deadlock_error(error) is False

        error = Exception("Connection refused")
        assert manager._is_deadlock_error(error) is False
