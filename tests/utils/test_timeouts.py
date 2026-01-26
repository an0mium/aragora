"""
Tests for timeout utilities.

Tests cover:
- timed_lock context manager
- timed_rlock context manager
- async_timeout function
- DEFAULT_LOCK_TIMEOUT constant
"""

import asyncio
import threading
import time
import pytest

from aragora.utils.timeouts import (
    DEFAULT_LOCK_TIMEOUT,
    timed_lock,
    timed_rlock,
    async_timeout,
)


class TestTimedLock:
    """Tests for timed_lock context manager."""

    def test_acquires_available_lock(self):
        """Acquires an available lock."""
        lock = threading.Lock()
        with timed_lock(lock, timeout=1.0):
            assert True  # Lock was acquired

    def test_releases_lock_on_exit(self):
        """Releases lock after context exits."""
        lock = threading.Lock()
        with timed_lock(lock, timeout=1.0):
            pass
        # Lock should be released, can acquire again
        assert lock.acquire(blocking=False)
        lock.release()

    def test_releases_lock_on_exception(self):
        """Releases lock even when exception raised."""
        lock = threading.Lock()
        with pytest.raises(ValueError):
            with timed_lock(lock, timeout=1.0):
                raise ValueError("test error")
        # Lock should be released
        assert lock.acquire(blocking=False)
        lock.release()

    def test_timeout_on_held_lock(self):
        """Raises TimeoutError when lock is held."""
        lock = threading.Lock()
        lock.acquire()  # Hold the lock

        try:
            with pytest.raises(TimeoutError, match="Failed to acquire lock"):
                with timed_lock(lock, timeout=0.1):
                    pass
        finally:
            lock.release()

    def test_timeout_error_message_includes_name(self):
        """Error message includes lock name when provided."""
        lock = threading.Lock()
        lock.acquire()

        try:
            with pytest.raises(TimeoutError, match="'database'"):
                with timed_lock(lock, timeout=0.1, name="database"):
                    pass
        finally:
            lock.release()

    def test_timeout_error_message_no_name(self):
        """Error message works without lock name."""
        lock = threading.Lock()
        lock.acquire()

        try:
            with pytest.raises(TimeoutError) as exc_info:
                with timed_lock(lock, timeout=0.1):
                    pass
            # Should not have the single quote that indicates a name
            assert "'" not in str(exc_info.value) or "deadlock" in str(exc_info.value)
        finally:
            lock.release()

    def test_concurrent_access(self):
        """Handles concurrent access correctly."""
        lock = threading.Lock()
        results = []

        def worker(n):
            try:
                with timed_lock(lock, timeout=5.0):
                    time.sleep(0.05)
                    results.append(n)
            except TimeoutError:
                results.append(f"timeout-{n}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should complete (no timeouts)
        assert len(results) == 3
        assert all(isinstance(r, int) for r in results)


class TestTimedRLock:
    """Tests for timed_rlock context manager."""

    def test_acquires_available_rlock(self):
        """Acquires an available RLock."""
        lock = threading.RLock()
        with timed_rlock(lock, timeout=1.0):
            assert True  # Lock was acquired

    def test_releases_rlock_on_exit(self):
        """Releases RLock after context exits."""
        lock = threading.RLock()
        with timed_rlock(lock, timeout=1.0):
            pass
        # Lock should be released, can acquire again
        assert lock.acquire(blocking=False)
        lock.release()

    def test_allows_reentrant_acquisition(self):
        """RLock allows reentrant acquisition in same thread."""
        lock = threading.RLock()
        with timed_rlock(lock, timeout=1.0):
            # Should be able to acquire again (reentrant)
            with timed_rlock(lock, timeout=1.0):
                assert True

    def test_timeout_on_held_rlock(self):
        """Raises TimeoutError when RLock is held by another thread."""
        lock = threading.RLock()
        acquired_event = threading.Event()
        release_event = threading.Event()

        def holder():
            lock.acquire()
            acquired_event.set()
            release_event.wait()  # Wait until test signals release
            lock.release()

        holder_thread = threading.Thread(target=holder)
        holder_thread.start()
        acquired_event.wait()  # Wait until holder has the lock

        try:
            with pytest.raises(TimeoutError, match="Failed to acquire RLock"):
                with timed_rlock(lock, timeout=0.1):
                    pass
        finally:
            release_event.set()
            holder_thread.join()

    def test_error_message_mentions_recursive(self):
        """Error message mentions recursive locking."""
        lock = threading.RLock()
        acquired_event = threading.Event()
        release_event = threading.Event()

        def holder():
            lock.acquire()
            acquired_event.set()
            release_event.wait()
            lock.release()

        holder_thread = threading.Thread(target=holder)
        holder_thread.start()
        acquired_event.wait()

        try:
            with pytest.raises(TimeoutError, match="recursive"):
                with timed_rlock(lock, timeout=0.1):
                    pass
        finally:
            release_event.set()
            holder_thread.join()


class TestAsyncTimeout:
    """Tests for async_timeout function."""

    @pytest.mark.asyncio
    async def test_completes_within_timeout(self):
        """Returns result when operation completes within timeout."""

        async def fast_op():
            await asyncio.sleep(0.01)
            return "success"

        result = await async_timeout(fast_op(), timeout=1.0)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_timeout_on_slow_operation(self):
        """Raises TimeoutError when operation exceeds timeout."""

        async def slow_op():
            await asyncio.sleep(10)
            return "never"

        with pytest.raises(TimeoutError, match="timed out after"):
            await async_timeout(slow_op(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_error_includes_operation_name(self):
        """Error message includes operation name when provided."""

        async def slow_op():
            await asyncio.sleep(10)

        with pytest.raises(TimeoutError, match="database query"):
            await async_timeout(slow_op(), timeout=0.1, operation_name="database query")

    @pytest.mark.asyncio
    async def test_error_without_operation_name(self):
        """Error message works without operation name."""

        async def slow_op():
            await asyncio.sleep(10)

        with pytest.raises(TimeoutError) as exc_info:
            await async_timeout(slow_op(), timeout=0.1)
        # Should not have parentheses for operation name
        assert "()" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_propagates_exceptions(self):
        """Propagates exceptions from the coroutine."""

        async def failing_op():
            raise ValueError("operation failed")

        with pytest.raises(ValueError, match="operation failed"):
            await async_timeout(failing_op(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_returns_various_types(self):
        """Works with coroutines returning various types."""

        async def return_dict():
            return {"key": "value"}

        async def return_list():
            return [1, 2, 3]

        async def return_none():
            return None

        assert await async_timeout(return_dict(), timeout=1.0) == {"key": "value"}
        assert await async_timeout(return_list(), timeout=1.0) == [1, 2, 3]
        assert await async_timeout(return_none(), timeout=1.0) is None


class TestDefaultLockTimeout:
    """Tests for DEFAULT_LOCK_TIMEOUT constant."""

    def test_default_timeout_value(self):
        """Default timeout is 30 seconds."""
        assert DEFAULT_LOCK_TIMEOUT == 30.0

    def test_used_as_default_in_timed_lock(self):
        """timed_lock uses DEFAULT_LOCK_TIMEOUT as default."""
        # Can't easily test this without reflection, but we can verify
        # that timed_lock works without explicit timeout
        lock = threading.Lock()
        with timed_lock(lock):  # No timeout argument
            pass


class TestIntegration:
    """Integration tests."""

    def test_multiple_locks_same_thread(self):
        """Multiple locks can be acquired in same thread."""
        lock1 = threading.Lock()
        lock2 = threading.Lock()

        with timed_lock(lock1, timeout=1.0):
            with timed_lock(lock2, timeout=1.0):
                # Both locks held
                assert True

    @pytest.mark.asyncio
    async def test_async_timeout_with_gather(self):
        """async_timeout works with asyncio.gather."""

        async def op1():
            await asyncio.sleep(0.01)
            return 1

        async def op2():
            await asyncio.sleep(0.01)
            return 2

        results = await asyncio.gather(
            async_timeout(op1(), timeout=1.0),
            async_timeout(op2(), timeout=1.0),
        )
        assert results == [1, 2]

    @pytest.mark.asyncio
    async def test_async_timeout_partial_failure(self):
        """One timeout doesn't affect other operations in gather."""

        async def fast_op():
            await asyncio.sleep(0.01)
            return "fast"

        async def slow_op():
            await asyncio.sleep(10)
            return "slow"

        # Use gather with return_exceptions to see results
        results = await asyncio.gather(
            async_timeout(fast_op(), timeout=1.0),
            async_timeout(slow_op(), timeout=0.1),
            return_exceptions=True,
        )

        assert results[0] == "fast"
        assert isinstance(results[1], TimeoutError)
