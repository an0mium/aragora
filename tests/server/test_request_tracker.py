"""
Tests for Request Tracker Module.

Tests cover:
- Basic request tracking (increment/decrement)
- Drain mode behavior (rejecting new requests)
- Graceful drain completion
- Drain timeout handling
- Concurrent request tracking
- Reset functionality
"""

import asyncio
from unittest.mock import patch

import pytest

from aragora.server.request_tracker import (
    RequestTracker,
    ServiceUnavailable,
    get_request_tracker,
    request_tracker,
)


class TestRequestTrackerBasics:
    """Tests for basic RequestTracker functionality."""

    def test_initial_state(self):
        """Test tracker starts with zero active requests."""
        tracker = RequestTracker()
        assert tracker.active_count == 0
        assert tracker.is_draining is False

    @pytest.mark.asyncio
    async def test_track_request_increments_count(self):
        """Test that tracking a request increments the count."""
        tracker = RequestTracker()

        async with tracker.track_request():
            assert tracker.active_count == 1

        assert tracker.active_count == 0

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self):
        """Test tracking multiple concurrent requests."""
        tracker = RequestTracker()

        async def simulate_request(delay: float):
            async with tracker.track_request():
                await asyncio.sleep(delay)

        # Start multiple concurrent requests
        tasks = [
            asyncio.create_task(simulate_request(0.1)),
            asyncio.create_task(simulate_request(0.1)),
            asyncio.create_task(simulate_request(0.1)),
        ]

        # Give time for all tasks to start
        await asyncio.sleep(0.01)
        assert tracker.active_count == 3

        # Wait for all to complete
        await asyncio.gather(*tasks)
        assert tracker.active_count == 0

    @pytest.mark.asyncio
    async def test_request_exception_decrements_count(self):
        """Test that exceptions in requests still decrement the count."""
        tracker = RequestTracker()

        with pytest.raises(ValueError, match="test error"):
            async with tracker.track_request():
                assert tracker.active_count == 1
                raise ValueError("test error")

        assert tracker.active_count == 0


class TestDrainMode:
    """Tests for drain mode behavior."""

    @pytest.mark.asyncio
    async def test_draining_rejects_new_requests(self):
        """Test that new requests are rejected when draining."""
        tracker = RequestTracker()
        tracker._draining = True

        with pytest.raises(ServiceUnavailable, match="shutting down"):
            async with tracker.track_request():
                pass

    @pytest.mark.asyncio
    async def test_start_drain_sets_flag(self):
        """Test that start_drain sets the draining flag."""
        tracker = RequestTracker()
        assert tracker.is_draining is False

        await tracker.start_drain(timeout=0.1)
        assert tracker.is_draining is True

    @pytest.mark.asyncio
    async def test_drain_completes_immediately_with_no_requests(self):
        """Test that drain completes immediately with no active requests."""
        tracker = RequestTracker()

        success = await tracker.start_drain(timeout=1.0)
        assert success is True
        assert tracker.is_draining is True

    @pytest.mark.asyncio
    async def test_drain_waits_for_active_requests(self):
        """Test that drain waits for active requests to complete."""
        tracker = RequestTracker()

        async def long_request():
            async with tracker.track_request():
                await asyncio.sleep(0.2)

        # Start a request
        task = asyncio.create_task(long_request())
        await asyncio.sleep(0.01)  # Let request start

        # Start drain
        success = await tracker.start_drain(timeout=1.0)
        assert success is True

        # Ensure task completed
        await task

    @pytest.mark.asyncio
    async def test_drain_timeout_returns_false(self):
        """Test that drain returns False when timeout is reached."""
        tracker = RequestTracker()

        async def long_request():
            async with tracker.track_request():
                await asyncio.sleep(10.0)  # Very long request

        # Start a request that won't complete
        task = asyncio.create_task(long_request())
        await asyncio.sleep(0.01)  # Let request start

        # Start drain with short timeout
        success = await tracker.start_drain(timeout=0.1)
        assert success is False
        assert tracker.active_count == 1

        # Cancel the long task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class TestReset:
    """Tests for reset functionality."""

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        """Test that reset clears all state."""
        tracker = RequestTracker()

        # Set up some state
        tracker._draining = True
        tracker._active_count = 5
        tracker._drain_complete.set()

        # Reset
        tracker.reset()

        assert tracker.is_draining is False
        assert tracker.active_count == 0
        assert not tracker._drain_complete.is_set()


class TestGlobalSingleton:
    """Tests for global singleton access."""

    def test_get_request_tracker_returns_singleton(self):
        """Test that get_request_tracker returns the global singleton."""
        tracker = get_request_tracker()
        assert tracker is request_tracker

    def test_singleton_is_request_tracker_instance(self):
        """Test that the singleton is a RequestTracker instance."""
        tracker = get_request_tracker()
        assert isinstance(tracker, RequestTracker)


class TestServiceUnavailable:
    """Tests for ServiceUnavailable exception."""

    def test_exception_has_message(self):
        """Test that ServiceUnavailable exception has a message."""
        exc = ServiceUnavailable("test message")
        assert str(exc) == "test message"

    def test_exception_is_exception(self):
        """Test that ServiceUnavailable is an Exception subclass."""
        assert issubclass(ServiceUnavailable, Exception)


class TestIntegrationWithShutdown:
    """Integration tests with shutdown sequence."""

    @pytest.mark.asyncio
    async def test_drain_phase_integration(self):
        """Test that drain phase works correctly."""
        from aragora.server.shutdown_sequence import ShutdownPhase, ShutdownSequence

        tracker = RequestTracker()

        async def drain_requests():
            active = tracker.active_count
            success = await tracker.start_drain(timeout=1.0)
            return success

        sequence = ShutdownSequence()
        sequence.add_phase(
            ShutdownPhase(
                name="Drain requests",
                execute=drain_requests,
                timeout=2.0,
                critical=True,
            )
        )

        result = await sequence.execute_all(overall_timeout=5.0)
        assert "Drain requests" in result["completed"]

    @pytest.mark.asyncio
    async def test_drain_with_active_requests(self):
        """Test drain completes after active requests finish."""
        tracker = RequestTracker()
        completed_requests = []

        async def request_handler(request_id: int):
            async with tracker.track_request():
                await asyncio.sleep(0.1)
                completed_requests.append(request_id)

        # Start requests
        tasks = [
            asyncio.create_task(request_handler(1)),
            asyncio.create_task(request_handler(2)),
        ]
        await asyncio.sleep(0.01)  # Let requests start

        # Drain
        success = await tracker.start_drain(timeout=2.0)
        assert success is True
        assert len(completed_requests) == 2

        await asyncio.gather(*tasks)
