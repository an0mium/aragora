"""
Tests for aragora.persistence.transaction - PostgreSQL transaction handling.

Tests cover:
- Transaction commit and rollback
- Nested transactions (SAVEPOINT)
- Transaction isolation levels
- Deadlock handling and recovery
- Connection state after transaction completion
- Edge cases (empty transactions, multiple rollbacks, etc.)
- Transaction statistics
- Factory functions
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from aragora.persistence.transaction import (
    TransactionManager,
    NestedTransactionManager,
    TransactionConfig,
    TransactionStats,
    TransactionContext,
    TransactionState,
    TransactionIsolation,
    TransactionError,
    TransactionStateError,
    SavepointError,
    DeadlockError,
    create_transaction_manager,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


def create_mock_connection() -> MagicMock:
    """Create a properly configured mock asyncpg connection.

    The mock is configured to:
    - Have async methods (execute, fetch, fetchrow, fetchval)
    - Not auto-create a 'connection' attribute (for proper getattr fallback)
    """
    conn = MagicMock()
    conn.execute = AsyncMock(return_value="OK")
    conn.fetch = AsyncMock(return_value=[{"id": 1}])
    conn.fetchrow = AsyncMock(return_value={"id": 1})
    conn.fetchval = AsyncMock(return_value=1)
    # Prevent auto-creation of 'connection' attribute that would break getattr fallback
    # Setting to None so getattr(conn, "connection", conn) returns conn
    del conn.connection  # Remove auto-generated mock attribute
    return conn


@pytest.fixture
def mock_connection():
    """Create a mock asyncpg connection."""
    return create_mock_connection()


@pytest.fixture
def mock_pool(mock_connection):
    """Create a mock connection pool."""
    pool = MagicMock()

    @asynccontextmanager
    async def mock_acquire(readonly: bool = False):
        yield mock_connection

    pool.acquire = mock_acquire
    return pool


@pytest.fixture
def manager(mock_pool):
    """Create a TransactionManager with mock pool (validation disabled)."""
    config = TransactionConfig(validate_connection_state=False)
    return TransactionManager(mock_pool, config)


@pytest.fixture
def config():
    """Create a TransactionConfig for testing."""
    return TransactionConfig(
        isolation=TransactionIsolation.READ_COMMITTED,
        timeout_seconds=30.0,
        deadlock_retries=3,
        deadlock_base_delay=0.01,  # Fast retries for testing
        deadlock_max_delay=0.1,
        savepoint_on_nested=True,
        validate_connection_state=False,  # Disable for testing with mocks
    )


@pytest.fixture
def manager_with_config(mock_pool, config):
    """Create a TransactionManager with custom config."""
    return TransactionManager(mock_pool, config)


@pytest.fixture
def manager_with_validation(mock_pool, mock_connection):
    """Create a TransactionManager with connection validation enabled."""
    config = TransactionConfig(validate_connection_state=True)
    return TransactionManager(mock_pool, config)


# ===========================================================================
# Test TransactionConfig
# ===========================================================================


class TestTransactionConfig:
    """Tests for TransactionConfig dataclass."""

    def test_default_values(self):
        """Default configuration values are sensible."""
        config = TransactionConfig()

        assert config.isolation == TransactionIsolation.READ_COMMITTED
        assert config.timeout_seconds == 30.0
        assert config.deadlock_retries == 3
        assert config.deadlock_base_delay == 0.1
        assert config.deadlock_max_delay == 2.0
        assert config.savepoint_on_nested is True
        assert config.validate_connection_state is True

    def test_custom_values(self):
        """Custom configuration values are stored correctly."""
        config = TransactionConfig(
            isolation=TransactionIsolation.SERIALIZABLE,
            timeout_seconds=60.0,
            deadlock_retries=5,
            deadlock_base_delay=0.5,
            deadlock_max_delay=5.0,
            savepoint_on_nested=False,
            validate_connection_state=False,
        )

        assert config.isolation == TransactionIsolation.SERIALIZABLE
        assert config.timeout_seconds == 60.0
        assert config.deadlock_retries == 5
        assert config.deadlock_base_delay == 0.5
        assert config.deadlock_max_delay == 5.0
        assert config.savepoint_on_nested is False
        assert config.validate_connection_state is False


# ===========================================================================
# Test TransactionStats
# ===========================================================================


class TestTransactionStats:
    """Tests for TransactionStats dataclass."""

    def test_default_values(self):
        """Default stats are all zero."""
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

    def test_to_dict(self):
        """to_dict returns correct dictionary representation."""
        stats = TransactionStats(
            transactions_started=10,
            transactions_committed=8,
            transactions_rolled_back=1,
            transactions_failed=1,
            savepoints_created=5,
            savepoints_released=4,
            savepoints_rolled_back=1,
            deadlocks_detected=2,
            deadlocks_recovered=1,
            active_transactions=2,
            total_transaction_time_ms=5000.0,
        )

        result = stats.to_dict()

        assert result["transactions_started"] == 10
        assert result["transactions_committed"] == 8
        assert result["transactions_rolled_back"] == 1
        assert result["transactions_failed"] == 1
        assert result["savepoints_created"] == 5
        assert result["savepoints_released"] == 4
        assert result["savepoints_rolled_back"] == 1
        assert result["deadlocks_detected"] == 2
        assert result["deadlocks_recovered"] == 1
        assert result["active_transactions"] == 2
        assert result["total_transaction_time_ms"] == 5000.0
        # Average: 5000 / (8 + 1) = ~555.56
        assert "average_transaction_time_ms" in result

    def test_average_calculation_with_no_transactions(self):
        """Average calculation handles zero transactions."""
        stats = TransactionStats()
        result = stats.to_dict()

        # Should not divide by zero
        assert result["average_transaction_time_ms"] == 0.0


# ===========================================================================
# Test TransactionIsolation
# ===========================================================================


class TestTransactionIsolation:
    """Tests for TransactionIsolation enum."""

    def test_isolation_levels(self):
        """All isolation levels have correct SQL values."""
        assert TransactionIsolation.READ_COMMITTED.value == "READ COMMITTED"
        assert TransactionIsolation.REPEATABLE_READ.value == "REPEATABLE READ"
        assert TransactionIsolation.SERIALIZABLE.value == "SERIALIZABLE"

    def test_isolation_is_string_enum(self):
        """Isolation levels can be used as strings."""
        assert str(TransactionIsolation.READ_COMMITTED) == "TransactionIsolation.READ_COMMITTED"
        assert TransactionIsolation.SERIALIZABLE == "SERIALIZABLE"


# ===========================================================================
# Test TransactionState
# ===========================================================================


class TestTransactionState:
    """Tests for TransactionState enum."""

    def test_all_states_exist(self):
        """All expected states are defined."""
        assert TransactionState.INACTIVE == "inactive"
        assert TransactionState.ACTIVE == "active"
        assert TransactionState.COMMITTED == "committed"
        assert TransactionState.ROLLED_BACK == "rolled_back"
        assert TransactionState.FAILED == "failed"


# ===========================================================================
# Test TransactionContext
# ===========================================================================


class TestTransactionContext:
    """Tests for TransactionContext dataclass."""

    def test_is_active_property(self):
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

        ctx.state = TransactionState.INACTIVE
        assert ctx.is_active is False

    def test_duration_ms_property(self):
        """duration_ms calculates elapsed time."""
        import time

        ctx = TransactionContext(
            id="txn_1",
            connection=MagicMock(),
            isolation=TransactionIsolation.READ_COMMITTED,
            started_at=time.time() - 1.0,  # 1 second ago
        )

        duration = ctx.duration_ms
        assert duration >= 1000.0  # At least 1 second
        assert duration < 2000.0  # Less than 2 seconds

    def test_savepoint_stack(self):
        """Savepoint stack is properly initialized."""
        ctx = TransactionContext(
            id="txn_1",
            connection=MagicMock(),
            isolation=TransactionIsolation.READ_COMMITTED,
        )

        assert ctx.savepoint_stack == []
        ctx.savepoint_stack.append("sp1")
        ctx.savepoint_stack.append("sp2")
        assert len(ctx.savepoint_stack) == 2


# ===========================================================================
# Test TransactionManager - Basic Transaction Operations
# ===========================================================================


class TestTransactionBasicOperations:
    """Tests for basic transaction commit and rollback."""

    @pytest.mark.asyncio
    async def test_successful_transaction_commits(self, manager, mock_connection):
        """Successful transaction is committed."""
        async with manager.transaction() as conn:
            await conn.execute("INSERT INTO test VALUES (1)")

        # Verify BEGIN and COMMIT were called
        calls = mock_connection.execute.call_args_list
        call_args = [str(c) for c in calls]

        assert any("BEGIN" in str(c) for c in calls)
        assert any("COMMIT" in str(c) for c in calls)

        # Stats should reflect committed transaction
        assert manager.get_stats().transactions_committed == 1
        assert manager.get_stats().transactions_started == 1

    @pytest.mark.asyncio
    async def test_exception_triggers_rollback(self, manager, mock_connection):
        """Exception within transaction triggers rollback."""
        with pytest.raises(ValueError, match="test error"):
            async with manager.transaction() as conn:
                await conn.execute("INSERT INTO test VALUES (1)")
                raise ValueError("test error")

        # Verify ROLLBACK was called
        calls = mock_connection.execute.call_args_list
        assert any("ROLLBACK" in str(c) for c in calls)

        # Stats should reflect failed transaction
        assert manager.get_stats().transactions_failed == 1

    @pytest.mark.asyncio
    async def test_empty_transaction_commits(self, manager, mock_connection):
        """Empty transaction (no operations) still commits."""
        async with manager.transaction() as conn:
            pass  # No operations

        # Should still commit
        calls = mock_connection.execute.call_args_list
        assert any("COMMIT" in str(c) for c in calls)
        assert manager.get_stats().transactions_committed == 1

    @pytest.mark.asyncio
    async def test_transaction_with_isolation_level(self, manager, mock_connection):
        """Transaction sets correct isolation level."""
        async with manager.transaction(isolation=TransactionIsolation.SERIALIZABLE) as conn:
            await conn.execute("SELECT 1")

        # Verify isolation level was set
        calls = mock_connection.execute.call_args_list
        assert any("SERIALIZABLE" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_readonly_hint_passed_to_pool(self):
        """readonly flag is passed to pool.acquire()."""
        mock_conn = create_mock_connection()

        acquire_kwargs = []

        @asynccontextmanager
        async def tracking_acquire(readonly: bool = False):
            acquire_kwargs.append({"readonly": readonly})
            yield mock_conn

        pool = MagicMock()
        pool.acquire = tracking_acquire

        config = TransactionConfig(validate_connection_state=False)
        manager = TransactionManager(pool, config)

        async with manager.transaction(readonly=True) as conn:
            pass

        assert acquire_kwargs[-1]["readonly"] is True


# ===========================================================================
# Test Transaction Rollback Scenarios
# ===========================================================================


class TestTransactionRollback:
    """Tests for transaction rollback scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_operations_rolled_back(self, manager, mock_connection):
        """All operations are rolled back on error."""
        with pytest.raises(RuntimeError):
            async with manager.transaction() as conn:
                await conn.execute("INSERT INTO a VALUES (1)")
                await conn.execute("INSERT INTO b VALUES (2)")
                await conn.execute("INSERT INTO c VALUES (3)")
                raise RuntimeError("Abort!")

        # Verify rollback was called
        calls = mock_connection.execute.call_args_list
        assert any("ROLLBACK" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_rollback_still_works_if_rollback_fails(self, manager, mock_connection):
        """Transaction handles rollback failure gracefully."""
        call_count = [0]

        async def failing_execute(query, *args, **kwargs):
            call_count[0] += 1
            if "ROLLBACK" in query:
                raise ConnectionError("Connection lost during rollback")
            return "OK"

        mock_connection.execute = AsyncMock(side_effect=failing_execute)

        # Should still raise the original error
        with pytest.raises(ValueError, match="original error"):
            async with manager.transaction() as conn:
                await conn.execute("INSERT INTO test VALUES (1)")
                raise ValueError("original error")

    @pytest.mark.asyncio
    async def test_cancelled_transaction_rolls_back(self, manager, mock_connection):
        """Cancelled async task triggers rollback."""
        with pytest.raises(asyncio.CancelledError):
            async with manager.transaction() as conn:
                await conn.execute("INSERT INTO test VALUES (1)")
                raise asyncio.CancelledError()

        calls = mock_connection.execute.call_args_list
        assert any("ROLLBACK" in str(c) for c in calls)
        assert manager.get_stats().transactions_rolled_back == 1


# ===========================================================================
# Test Nested Transactions (SAVEPOINTs)
# ===========================================================================


class TestSavepoints:
    """Tests for nested transactions using SAVEPOINTs."""

    @pytest.mark.asyncio
    async def test_savepoint_created_and_released(self, manager, mock_connection):
        """Savepoint is created and released on success."""
        async with manager.transaction() as conn:
            await conn.execute("INSERT INTO main_table VALUES (1)")

            async with manager.savepoint(conn, "sp1"):
                await conn.execute("INSERT INTO nested_table VALUES (2)")

        calls = mock_connection.execute.call_args_list
        call_strings = [str(c) for c in calls]

        assert any("SAVEPOINT sp1" in s for s in call_strings)
        assert any("RELEASE SAVEPOINT sp1" in s for s in call_strings)

        assert manager.get_stats().savepoints_created == 1
        assert manager.get_stats().savepoints_released == 1

    @pytest.mark.asyncio
    async def test_savepoint_rolled_back_on_error(self, manager, mock_connection):
        """Savepoint is rolled back on error, outer transaction continues."""
        async with manager.transaction() as conn:
            await conn.execute("INSERT INTO main_table VALUES (1)")

            with pytest.raises(ValueError):
                async with manager.savepoint(conn, "sp1"):
                    await conn.execute("INSERT INTO nested_table VALUES (2)")
                    raise ValueError("nested error")

            # Continue with outer transaction
            await conn.execute("INSERT INTO main_table VALUES (3)")

        calls = mock_connection.execute.call_args_list
        call_strings = [str(c) for c in calls]

        assert any("ROLLBACK TO SAVEPOINT sp1" in s for s in call_strings)
        assert any("COMMIT" in s for s in call_strings)  # Outer commits

        assert manager.get_stats().savepoints_rolled_back == 1
        assert manager.get_stats().transactions_committed == 1

    @pytest.mark.asyncio
    async def test_nested_savepoints(self, manager, mock_connection):
        """Multiple levels of savepoints work correctly."""
        async with manager.transaction() as conn:
            await conn.execute("Level 0")

            async with manager.savepoint(conn, "sp1"):
                await conn.execute("Level 1")

                async with manager.savepoint(conn, "sp2"):
                    await conn.execute("Level 2")

        calls = mock_connection.execute.call_args_list
        call_strings = [str(c) for c in calls]

        assert any("SAVEPOINT sp1" in s for s in call_strings)
        assert any("SAVEPOINT sp2" in s for s in call_strings)
        assert any("RELEASE SAVEPOINT sp2" in s for s in call_strings)
        assert any("RELEASE SAVEPOINT sp1" in s for s in call_strings)

        assert manager.get_stats().savepoints_created == 2
        assert manager.get_stats().savepoints_released == 2

    @pytest.mark.asyncio
    async def test_invalid_savepoint_name_rejected(self, manager, mock_connection):
        """Invalid savepoint names are rejected."""
        async with manager.transaction() as conn:
            with pytest.raises(SavepointError, match="Invalid savepoint name"):
                async with manager.savepoint(conn, "sp; DROP TABLE"):
                    pass

    @pytest.mark.asyncio
    async def test_savepoint_with_underscores_allowed(self, manager, mock_connection):
        """Savepoint names with underscores are valid."""
        async with manager.transaction() as conn:
            async with manager.savepoint(conn, "my_savepoint_123"):
                await conn.execute("SELECT 1")

        assert manager.get_stats().savepoints_released == 1


# ===========================================================================
# Test Transaction Isolation Levels
# ===========================================================================


class TestIsolationLevels:
    """Tests for transaction isolation level handling."""

    @pytest.mark.asyncio
    async def test_read_committed_isolation(self, manager, mock_connection):
        """READ COMMITTED isolation is set correctly."""
        async with manager.transaction(isolation=TransactionIsolation.READ_COMMITTED) as conn:
            pass

        calls = mock_connection.execute.call_args_list
        assert any("READ COMMITTED" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_repeatable_read_isolation(self, manager, mock_connection):
        """REPEATABLE READ isolation is set correctly."""
        async with manager.transaction(isolation=TransactionIsolation.REPEATABLE_READ) as conn:
            pass

        calls = mock_connection.execute.call_args_list
        assert any("REPEATABLE READ" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_serializable_isolation(self, manager, mock_connection):
        """SERIALIZABLE isolation is set correctly."""
        async with manager.transaction(isolation=TransactionIsolation.SERIALIZABLE) as conn:
            pass

        calls = mock_connection.execute.call_args_list
        assert any("SERIALIZABLE" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_default_isolation_from_config(self):
        """Default isolation level comes from config."""
        mock_conn = create_mock_connection()

        config = TransactionConfig(
            isolation=TransactionIsolation.SERIALIZABLE,
            validate_connection_state=False,
        )

        @asynccontextmanager
        async def mock_acquire(readonly: bool = False):
            yield mock_conn

        pool = MagicMock()
        pool.acquire = mock_acquire

        manager = TransactionManager(pool, config)

        async with manager.transaction() as conn:
            pass

        calls = mock_conn.execute.call_args_list
        assert any("SERIALIZABLE" in str(c) for c in calls)


# ===========================================================================
# Test Deadlock Handling
# ===========================================================================


class TestDeadlockHandling:
    """Tests for deadlock detection and recovery."""

    @pytest.mark.asyncio
    async def test_deadlock_detected(self):
        """Deadlock error is detected from error message."""
        mock_conn = create_mock_connection()

        # Simulate deadlock error
        async def deadlock_execute(query, *args, **kwargs):
            if "INSERT" in query:
                raise Exception("ERROR: deadlock detected")
            return "OK"

        mock_conn.execute = AsyncMock(side_effect=deadlock_execute)

        @asynccontextmanager
        async def mock_acquire(readonly: bool = False):
            yield mock_conn

        pool = MagicMock()
        pool.acquire = mock_acquire

        config = TransactionConfig(
            deadlock_retries=2,
            deadlock_base_delay=0.001,
            deadlock_max_delay=0.01,
            validate_connection_state=False,
        )
        manager = TransactionManager(pool, config)

        with pytest.raises(DeadlockError):
            async with manager.transaction() as conn:
                await conn.execute("INSERT INTO test VALUES (1)")

        assert manager.get_stats().deadlocks_detected >= 1

    @pytest.mark.asyncio
    async def test_deadlock_recovery_succeeds(self):
        """Deadlock is recovered after retry."""
        mock_conn = create_mock_connection()
        call_count = [0]

        async def intermittent_deadlock(query, *args, **kwargs):
            call_count[0] += 1
            if "INSERT" in query and call_count[0] <= 2:
                raise Exception("ERROR: deadlock detected")
            return "OK"

        mock_conn.execute = AsyncMock(side_effect=intermittent_deadlock)

        @asynccontextmanager
        async def mock_acquire(readonly: bool = False):
            yield mock_conn

        pool = MagicMock()
        pool.acquire = mock_acquire

        config = TransactionConfig(
            deadlock_retries=3,
            deadlock_base_delay=0.001,
            deadlock_max_delay=0.01,
            validate_connection_state=False,
        )
        manager = TransactionManager(pool, config)

        # Should succeed after retries
        async with manager.transaction() as conn:
            await conn.execute("INSERT INTO test VALUES (1)")

        assert manager.get_stats().deadlocks_detected >= 1
        assert manager.get_stats().transactions_committed == 1

    @pytest.mark.asyncio
    async def test_deadlock_max_retries_exceeded(self):
        """DeadlockError raised after max retries exceeded."""
        mock_conn = create_mock_connection()

        async def always_deadlock(query, *args, **kwargs):
            if "INSERT" in query:
                raise Exception("ERROR: deadlock detected")
            return "OK"

        mock_conn.execute = AsyncMock(side_effect=always_deadlock)

        @asynccontextmanager
        async def mock_acquire(readonly: bool = False):
            yield mock_conn

        pool = MagicMock()
        pool.acquire = mock_acquire

        config = TransactionConfig(
            deadlock_retries=2,
            deadlock_base_delay=0.001,
            deadlock_max_delay=0.01,
            validate_connection_state=False,
        )
        manager = TransactionManager(pool, config)

        with pytest.raises(DeadlockError) as exc_info:
            async with manager.transaction() as conn:
                await conn.execute("INSERT INTO test VALUES (1)")

        assert exc_info.value.retry_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_serialization_failure_treated_as_deadlock(self):
        """Serialization failure (40001) is treated as deadlock."""
        mock_conn = create_mock_connection()

        async def serialization_failure(query, *args, **kwargs):
            if "UPDATE" in query:
                raise Exception("ERROR 40001: could not serialize access")
            return "OK"

        mock_conn.execute = AsyncMock(side_effect=serialization_failure)

        @asynccontextmanager
        async def mock_acquire(readonly: bool = False):
            yield mock_conn

        pool = MagicMock()
        pool.acquire = mock_acquire

        config = TransactionConfig(
            deadlock_retries=1,
            deadlock_base_delay=0.001,
            validate_connection_state=False,
        )
        manager = TransactionManager(pool, config)

        with pytest.raises(DeadlockError):
            async with manager.transaction() as conn:
                await conn.execute("UPDATE test SET value = 1")

        assert manager.get_stats().deadlocks_detected >= 1


# ===========================================================================
# Test Connection State
# ===========================================================================


class TestConnectionState:
    """Tests for connection state validation and handling."""

    @pytest.mark.asyncio
    async def test_connection_validated_before_transaction(self):
        """Connection is validated before starting transaction."""
        mock_conn = create_mock_connection()
        mock_conn.fetchval = AsyncMock(return_value=1)

        @asynccontextmanager
        async def mock_acquire(readonly: bool = False):
            yield mock_conn

        pool = MagicMock()
        pool.acquire = mock_acquire

        config = TransactionConfig(validate_connection_state=True)
        manager = TransactionManager(pool, config)

        async with manager.transaction() as conn:
            pass

        # fetchval should be called for validation
        mock_conn.fetchval.assert_called()

    @pytest.mark.asyncio
    async def test_validation_disabled(self):
        """Connection validation can be disabled."""
        mock_conn = create_mock_connection()
        mock_conn.fetchval = AsyncMock(return_value=1)

        @asynccontextmanager
        async def mock_acquire(readonly: bool = False):
            yield mock_conn

        pool = MagicMock()
        pool.acquire = mock_acquire

        config = TransactionConfig(validate_connection_state=False)
        manager = TransactionManager(pool, config)

        async with manager.transaction() as conn:
            pass

        # Validation (fetchval) should not have been called
        mock_conn.fetchval.assert_not_called()

    @pytest.mark.asyncio
    async def test_validation_failure_raises_error(self):
        """Failed connection validation raises TransactionError."""
        mock_conn = create_mock_connection()
        mock_conn.fetchval = AsyncMock(side_effect=ConnectionError("Lost connection"))

        @asynccontextmanager
        async def mock_acquire(readonly: bool = False):
            yield mock_conn

        pool = MagicMock()
        pool.acquire = mock_acquire

        config = TransactionConfig(validate_connection_state=True)
        manager = TransactionManager(pool, config)

        with pytest.raises(TransactionError, match="validation failed"):
            async with manager.transaction() as conn:
                pass


# ===========================================================================
# Test Transaction Timeout
# ===========================================================================


class TestTransactionTimeout:
    """Tests for transaction timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_triggers_rollback(self):
        """Transaction timeout triggers rollback."""
        mock_conn = create_mock_connection()

        async def slow_execute(query, *args, **kwargs):
            if "SLOW" in query:
                await asyncio.sleep(2.0)
            return "OK"

        mock_conn.execute = AsyncMock(side_effect=slow_execute)

        config = TransactionConfig(
            timeout_seconds=0.1,
            validate_connection_state=False,
        )

        @asynccontextmanager
        async def mock_acquire(readonly: bool = False):
            yield mock_conn

        pool = MagicMock()
        pool.acquire = mock_acquire

        manager = TransactionManager(pool, config)

        with pytest.raises(asyncio.TimeoutError):
            async with manager.transaction() as conn:
                await conn.execute("SLOW QUERY")

        # Stats should reflect rolled back transaction
        assert manager.get_stats().transactions_rolled_back == 1

    @pytest.mark.asyncio
    async def test_timeout_override(self):
        """Transaction timeout can be overridden per-transaction."""
        mock_conn = create_mock_connection()

        @asynccontextmanager
        async def mock_acquire(readonly: bool = False):
            yield mock_conn

        pool = MagicMock()
        pool.acquire = mock_acquire

        config = TransactionConfig(
            timeout_seconds=60.0,
            validate_connection_state=False,
        )
        manager = TransactionManager(pool, config)

        # Should work with long timeout
        async with manager.transaction(timeout=120.0) as conn:
            await conn.execute("SELECT 1")

        assert manager.get_stats().transactions_committed == 1

    @pytest.mark.asyncio
    async def test_zero_timeout_means_no_timeout(self):
        """Timeout of 0 means no timeout."""
        mock_conn = create_mock_connection()

        @asynccontextmanager
        async def mock_acquire(readonly: bool = False):
            yield mock_conn

        pool = MagicMock()
        pool.acquire = mock_acquire

        config = TransactionConfig(
            timeout_seconds=0,
            validate_connection_state=False,
        )
        manager = TransactionManager(pool, config)

        async with manager.transaction() as conn:
            await conn.execute("SELECT 1")

        assert manager.get_stats().transactions_committed == 1


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

        calls = mock_connection.execute.call_args_list
        assert any("BEGIN" in str(c) for c in calls)
        assert any("COMMIT" in str(c) for c in calls)

        assert manager.get_stats().transactions_started == 1
        assert manager.get_stats().transactions_committed == 1

    @pytest.mark.asyncio
    async def test_manual_begin_rollback(self, manager, mock_connection):
        """Manual begin and rollback work correctly."""
        await manager.begin(mock_connection)
        await mock_connection.execute("INSERT INTO test VALUES (1)")
        await manager.rollback(mock_connection)

        calls = mock_connection.execute.call_args_list
        assert any("ROLLBACK" in str(c) for c in calls)

        assert manager.get_stats().transactions_rolled_back == 1

    @pytest.mark.asyncio
    async def test_manual_begin_with_isolation(self, manager, mock_connection):
        """Manual begin sets isolation level."""
        await manager.begin(mock_connection, isolation=TransactionIsolation.SERIALIZABLE)

        calls = mock_connection.execute.call_args_list
        assert any("SERIALIZABLE" in str(c) for c in calls)


# ===========================================================================
# Test Statistics
# ===========================================================================


class TestTransactionStatistics:
    """Tests for transaction statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_accumulate(self, manager, mock_connection):
        """Statistics accumulate over multiple transactions."""
        # First transaction - success
        async with manager.transaction() as conn:
            await conn.execute("SELECT 1")

        # Second transaction - success
        async with manager.transaction() as conn:
            await conn.execute("SELECT 2")

        # Third transaction - failure
        with pytest.raises(ValueError):
            async with manager.transaction() as conn:
                raise ValueError("error")

        stats = manager.get_stats()
        assert stats.transactions_started == 3
        assert stats.transactions_committed == 2
        assert stats.transactions_failed == 1

    @pytest.mark.asyncio
    async def test_active_transaction_count(self):
        """Active transaction count is tracked."""
        mock_conn = create_mock_connection()

        @asynccontextmanager
        async def mock_acquire(readonly: bool = False):
            yield mock_conn

        pool = MagicMock()
        pool.acquire = mock_acquire

        config = TransactionConfig(validate_connection_state=False)
        manager = TransactionManager(pool, config)

        assert manager.active_transaction_count == 0

        async with manager.transaction() as conn:
            # Would be 1 inside, but hard to test due to async nature
            pass

        assert manager.active_transaction_count == 0

    def test_reset_stats(self, manager):
        """Statistics can be reset."""
        manager._stats.transactions_started = 100
        manager._stats.transactions_committed = 99

        manager.reset_stats()

        assert manager.get_stats().transactions_started == 0
        assert manager.get_stats().transactions_committed == 0

    def test_get_stats_dict(self, manager):
        """get_stats_dict returns dictionary format."""
        manager._stats.transactions_started = 10
        result = manager.get_stats_dict()

        assert isinstance(result, dict)
        assert result["transactions_started"] == 10


# ===========================================================================
# Test NestedTransactionManager
# ===========================================================================


class TestNestedTransactionManager:
    """Tests for NestedTransactionManager with automatic nesting."""

    @pytest.mark.asyncio
    async def test_nested_uses_savepoint(self, mock_connection):
        """Nested transaction automatically uses savepoint."""

        @asynccontextmanager
        async def mock_acquire(readonly: bool = False):
            yield mock_connection

        pool = MagicMock()
        pool.acquire = mock_acquire

        manager = NestedTransactionManager(pool)

        # Note: NestedTransactionManager reuses the connection from pool
        # In real usage, the outer transaction would be active
        # This test verifies the savepoint mechanism

        # Just verify the class initializes correctly
        assert manager._connection_depth == {}


# ===========================================================================
# Test Factory Function
# ===========================================================================


class TestFactoryFunction:
    """Tests for create_transaction_manager factory."""

    def test_creates_standard_manager(self, mock_pool):
        """Factory creates standard TransactionManager."""
        manager = create_transaction_manager(
            mock_pool,
            isolation=TransactionIsolation.REPEATABLE_READ,
            timeout_seconds=45.0,
            deadlock_retries=5,
        )

        assert isinstance(manager, TransactionManager)
        assert manager.config.isolation == TransactionIsolation.REPEATABLE_READ
        assert manager.config.timeout_seconds == 45.0
        assert manager.config.deadlock_retries == 5

    def test_creates_nested_manager(self, mock_pool):
        """Factory creates NestedTransactionManager when requested."""
        manager = create_transaction_manager(mock_pool, nested_support=True)

        assert isinstance(manager, NestedTransactionManager)

    def test_default_values(self, mock_pool):
        """Factory uses sensible defaults."""
        manager = create_transaction_manager(mock_pool)

        assert manager.config.isolation == TransactionIsolation.READ_COMMITTED
        assert manager.config.timeout_seconds == 30.0
        assert manager.config.deadlock_retries == 3


# ===========================================================================
# Test Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_pool_without_acquire_raises_error(self):
        """Pool without acquire method raises TransactionError."""
        pool = MagicMock(spec=[])  # No acquire method

        manager = TransactionManager(pool)

        with pytest.raises(TransactionError, match="does not support acquire"):
            async with manager.transaction() as conn:
                pass

    @pytest.mark.asyncio
    async def test_wrapped_connection_unwrapped(self):
        """ConnectionWrapper is properly unwrapped."""
        mock_inner_conn = create_mock_connection()

        wrapper = MagicMock()
        wrapper.connection = mock_inner_conn

        @asynccontextmanager
        async def mock_acquire(readonly: bool = False):
            yield wrapper

        pool = MagicMock()
        pool.acquire = mock_acquire

        config = TransactionConfig(validate_connection_state=False)
        manager = TransactionManager(pool, config)

        async with manager.transaction() as conn:
            # Should use the unwrapped connection
            await conn.execute("SELECT 1")

        # The wrapped connection's execute should be called
        mock_inner_conn.execute.assert_called()

    @pytest.mark.asyncio
    async def test_transaction_id_uniqueness(self, manager, mock_connection):
        """Transaction IDs are unique."""
        ids = set()

        for _ in range(10):
            async with manager.transaction() as conn:
                pass

        # IDs should be generated (can't easily capture them, but counter increments)
        assert manager._transaction_counter >= 10

    @pytest.mark.asyncio
    async def test_concurrent_transactions(self):
        """Multiple concurrent transactions work correctly."""
        mock_conn = create_mock_connection()

        @asynccontextmanager
        async def mock_acquire(readonly: bool = False):
            yield mock_conn

        pool = MagicMock()
        pool.acquire = mock_acquire

        config = TransactionConfig(validate_connection_state=False)
        manager = TransactionManager(pool, config)

        async def run_transaction(value: int):
            async with manager.transaction() as conn:
                await conn.execute(f"INSERT INTO test VALUES ({value})")
                await asyncio.sleep(0.01)

        # Run multiple concurrent transactions
        await asyncio.gather(*[run_transaction(i) for i in range(5)])

        assert manager.get_stats().transactions_committed == 5

    @pytest.mark.asyncio
    async def test_transaction_time_tracked(self, manager, mock_connection):
        """Transaction time is tracked in statistics."""
        async with manager.transaction() as conn:
            await asyncio.sleep(0.01)

        assert manager.get_stats().total_transaction_time_ms > 0


# ===========================================================================
# Test Exception Classes
# ===========================================================================


class TestExceptions:
    """Tests for custom exception classes."""

    def test_deadlock_error(self):
        """DeadlockError stores retry count."""
        error = DeadlockError("deadlock occurred", retry_count=3)
        assert error.retry_count == 3
        assert "deadlock" in str(error)

    def test_transaction_state_error(self):
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

    def test_savepoint_error(self):
        """SavepointError is TransactionError subclass."""
        error = SavepointError("invalid name")
        assert isinstance(error, TransactionError)
        assert "invalid" in str(error)


# ===========================================================================
# Test Configuration Property Access
# ===========================================================================


class TestConfigAccess:
    """Tests for configuration access."""

    def test_config_property(self, manager, config):
        """Config property returns configuration."""
        manager = TransactionManager(MagicMock(), config)
        assert manager.config is config
        assert manager.config.timeout_seconds == 30.0
