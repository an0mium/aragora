"""
Tests for the cancellation module.

Tests cover:
- CancellationReason enum
- DebateCancelled exception
- CancellationToken class (cancel, check, callbacks, linked tokens)
- CancellationScope context manager
- create_linked_token function
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.cancellation import (
    CancellationReason,
    CancellationScope,
    CancellationToken,
    DebateCancelled,
    create_linked_token,
)


class TestCancellationReason:
    """Tests for CancellationReason enum."""

    def test_enum_values(self):
        """Test all enum values exist and have correct string values."""
        assert CancellationReason.USER_REQUESTED.value == "user_requested"
        assert CancellationReason.TIMEOUT.value == "timeout"
        assert CancellationReason.RESOURCE_LIMIT.value == "resource_limit"
        assert CancellationReason.PARENT_CANCELLED.value == "parent_cancelled"
        assert CancellationReason.ERROR.value == "error"
        assert CancellationReason.SHUTDOWN.value == "shutdown"

    def test_enum_members(self):
        """Test all expected enum members exist."""
        members = list(CancellationReason)
        assert len(members) == 6
        assert CancellationReason.USER_REQUESTED in members
        assert CancellationReason.TIMEOUT in members
        assert CancellationReason.RESOURCE_LIMIT in members
        assert CancellationReason.PARENT_CANCELLED in members
        assert CancellationReason.ERROR in members
        assert CancellationReason.SHUTDOWN in members


class TestDebateCancelled:
    """Tests for DebateCancelled exception."""

    def test_exception_creation_with_defaults(self):
        """Test creating exception with default reason type."""
        exc = DebateCancelled(reason="User stopped the debate")

        assert exc.reason == "User stopped the debate"
        assert exc.reason_type == CancellationReason.USER_REQUESTED
        assert exc.partial_result is None
        assert "User stopped the debate" in str(exc)

    def test_exception_creation_with_reason_type(self):
        """Test creating exception with specific reason type."""
        exc = DebateCancelled(
            reason="Operation timed out",
            reason_type=CancellationReason.TIMEOUT,
        )

        assert exc.reason == "Operation timed out"
        assert exc.reason_type == CancellationReason.TIMEOUT

    def test_exception_creation_with_partial_result(self):
        """Test creating exception with partial result."""
        partial = {"rounds_completed": 3, "last_response": "Partial answer"}
        exc = DebateCancelled(
            reason="Cancelled with partial work",
            partial_result=partial,
        )

        assert exc.partial_result == partial
        assert exc.partial_result["rounds_completed"] == 3

    def test_exception_message_format(self):
        """Test exception message format."""
        exc = DebateCancelled(reason="Test cancellation")
        assert str(exc) == "Debate cancelled: Test cancellation"

    def test_exception_can_be_raised_and_caught(self):
        """Test that exception can be raised and caught properly."""
        with pytest.raises(DebateCancelled) as exc_info:
            raise DebateCancelled(
                reason="Test error",
                reason_type=CancellationReason.ERROR,
            )

        assert exc_info.value.reason == "Test error"
        assert exc_info.value.reason_type == CancellationReason.ERROR


class TestCancellationToken:
    """Tests for CancellationToken class."""

    def test_token_creation(self):
        """Test creating a new token."""
        token = CancellationToken()

        assert token.is_cancelled is False
        assert token.reason is None
        assert token.reason_type == CancellationReason.USER_REQUESTED
        assert token.cancelled_at is None

    def test_cancel_sets_state(self):
        """Test that cancel() sets the cancelled state."""
        token = CancellationToken()
        token.cancel(reason="User requested stop")

        assert token.is_cancelled is True
        assert token.reason == "User requested stop"
        assert token.reason_type == CancellationReason.USER_REQUESTED
        assert token.cancelled_at is not None

    def test_cancel_with_reason_type(self):
        """Test cancel with specific reason type."""
        token = CancellationToken()
        token.cancel(
            reason="Resource exhausted",
            reason_type=CancellationReason.RESOURCE_LIMIT,
        )

        assert token.reason_type == CancellationReason.RESOURCE_LIMIT

    def test_cancel_sets_timestamp(self):
        """Test that cancel sets a timestamp."""
        token = CancellationToken()
        before = datetime.now(timezone.utc)
        token.cancel()
        after = datetime.now(timezone.utc)

        assert token.cancelled_at is not None
        assert before <= token.cancelled_at <= after

    def test_cancel_idempotent(self):
        """Test that calling cancel twice is idempotent."""
        token = CancellationToken()
        token.cancel(reason="First cancel")
        first_time = token.cancelled_at
        first_reason = token.reason

        token.cancel(reason="Second cancel")

        # Should keep first cancel's state
        assert token.reason == first_reason
        assert token.cancelled_at == first_time

    def test_check_not_cancelled(self):
        """Test check() does nothing when not cancelled."""
        token = CancellationToken()

        # Should not raise
        token.check()

    def test_check_raises_when_cancelled(self):
        """Test check() raises DebateCancelled when cancelled."""
        token = CancellationToken()
        token.cancel(
            reason="Stopped by user",
            reason_type=CancellationReason.USER_REQUESTED,
        )

        with pytest.raises(DebateCancelled) as exc_info:
            token.check()

        assert exc_info.value.reason == "Stopped by user"
        assert exc_info.value.reason_type == CancellationReason.USER_REQUESTED

    def test_bool_true_when_not_cancelled(self):
        """Test token is truthy when not cancelled."""
        token = CancellationToken()

        assert bool(token) is True
        assert token  # In boolean context

    def test_bool_false_when_cancelled(self):
        """Test token is falsy when cancelled."""
        token = CancellationToken()
        token.cancel()

        assert bool(token) is False
        assert not token  # In boolean context

    @pytest.mark.asyncio
    async def test_wait_for_cancellation(self):
        """Test waiting for cancellation."""
        token = CancellationToken()

        async def cancel_later():
            await asyncio.sleep(0.01)
            token.cancel(reason="Delayed cancel")

        task = asyncio.create_task(cancel_later())
        reason = await token.wait_for_cancellation()
        await task

        assert reason == "Delayed cancel"

    @pytest.mark.asyncio
    async def test_wait_for_cancellation_timeout(self):
        """Test wait_for_cancellation with timeout."""
        token = CancellationToken()

        with pytest.raises(asyncio.TimeoutError):
            await token.wait_for_cancellation(timeout=0.01)

    @pytest.mark.asyncio
    async def test_wait_for_cancellation_already_cancelled(self):
        """Test wait_for_cancellation when already cancelled."""
        token = CancellationToken()
        token.cancel(reason="Already done")

        reason = await token.wait_for_cancellation()

        assert reason == "Already done"

    @pytest.mark.asyncio
    async def test_wait_for_cancellation_no_reason(self):
        """Test wait_for_cancellation returns default reason if none set."""
        token = CancellationToken()
        token._cancelled.set()  # Set without reason

        reason = await token.wait_for_cancellation()

        assert reason == "Unknown reason"


class TestCancellationTokenCallbacks:
    """Tests for CancellationToken callback functionality."""

    def test_register_callback(self):
        """Test registering a callback."""
        token = CancellationToken()
        callback_called = []

        def callback(t):
            callback_called.append(t)

        unregister = token.register_callback(callback)

        assert callable(unregister)
        token.cancel()
        assert len(callback_called) == 1
        assert callback_called[0] is token

    def test_callback_called_on_cancel(self):
        """Test callback is called when token is cancelled."""
        token = CancellationToken()
        results = []

        token.register_callback(lambda t: results.append("callback1"))
        token.register_callback(lambda t: results.append("callback2"))

        token.cancel()

        assert "callback1" in results
        assert "callback2" in results

    def test_callback_called_immediately_if_already_cancelled(self):
        """Test callback is called immediately if token already cancelled."""
        token = CancellationToken()
        token.cancel()

        called = []
        token.register_callback(lambda t: called.append(True))

        assert len(called) == 1

    def test_unregister_callback(self):
        """Test unregistering a callback."""
        token = CancellationToken()
        called = []

        def callback(t):
            called.append(True)

        unregister = token.register_callback(callback)
        unregister()

        token.cancel()

        assert len(called) == 0

    def test_unregister_twice_is_safe(self):
        """Test calling unregister twice doesn't error."""
        token = CancellationToken()
        unregister = token.register_callback(lambda t: None)

        unregister()
        unregister()  # Should not raise

    def test_callback_exception_logged_not_raised(self):
        """Test callback exceptions are logged but don't stop other callbacks."""
        token = CancellationToken()
        results = []

        def bad_callback(t):
            raise ValueError("Callback error")

        def good_callback(t):
            results.append("good")

        token.register_callback(bad_callback)
        token.register_callback(good_callback)

        # Should not raise
        with patch("aragora.debate.cancellation.logger") as mock_logger:
            token.cancel()

        assert "good" in results
        mock_logger.warning.assert_called()


class TestCancellationTokenLinkedTokens:
    """Tests for linked parent-child token functionality."""

    def test_create_child(self):
        """Test creating a child token."""
        parent = CancellationToken()
        child = parent.create_child()

        assert child is not parent
        assert child._parent is parent
        assert child in parent._children

    def test_link_child(self):
        """Test linking an existing token as child."""
        parent = CancellationToken()
        child = CancellationToken()

        parent.link_child(child)

        assert child._parent is parent
        assert child in parent._children

    def test_child_cancelled_when_parent_cancelled(self):
        """Test child is cancelled when parent is cancelled."""
        parent = CancellationToken()
        child = parent.create_child()

        parent.cancel(reason="Parent stopped")

        assert child.is_cancelled is True
        assert "Parent cancelled" in child.reason
        assert child.reason_type == CancellationReason.PARENT_CANCELLED

    def test_multiple_children_cancelled(self):
        """Test all children are cancelled when parent is cancelled."""
        parent = CancellationToken()
        child1 = parent.create_child()
        child2 = parent.create_child()
        child3 = parent.create_child()

        parent.cancel()

        assert child1.is_cancelled is True
        assert child2.is_cancelled is True
        assert child3.is_cancelled is True

    def test_child_linked_to_already_cancelled_parent(self):
        """Test linking child to already cancelled parent cancels child."""
        parent = CancellationToken()
        parent.cancel(reason="Already stopped")

        child = parent.create_child()

        assert child.is_cancelled is True
        assert child.reason_type == CancellationReason.PARENT_CANCELLED

    def test_parent_cancellation_preserves_already_cancelled_child(self):
        """Test parent cancellation doesn't change already cancelled child."""
        parent = CancellationToken()
        child = parent.create_child()

        # Cancel child first
        child.cancel(reason="Child specific reason", reason_type=CancellationReason.ERROR)
        child_reason = child.reason

        # Now cancel parent
        parent.cancel(reason="Parent reason")

        # Child should keep its original reason
        assert child.reason == child_reason
        assert child.reason_type == CancellationReason.ERROR

    def test_grandchild_propagation(self):
        """Test cancellation propagates to grandchildren."""
        grandparent = CancellationToken()
        parent = grandparent.create_child()
        child = parent.create_child()

        grandparent.cancel()

        assert parent.is_cancelled is True
        assert child.is_cancelled is True


class TestCreateLinkedToken:
    """Tests for create_linked_token function."""

    def test_create_without_parent(self):
        """Test creating token without parent."""
        token = create_linked_token()

        assert token is not None
        assert token._parent is None
        assert token.is_cancelled is False

    def test_create_with_parent(self):
        """Test creating token linked to parent."""
        parent = CancellationToken()
        child = create_linked_token(parent)

        assert child._parent is parent
        assert child in parent._children

    def test_create_with_none_parent(self):
        """Test creating token with explicit None parent."""
        token = create_linked_token(parent=None)

        assert token._parent is None


class TestCancellationScope:
    """Tests for CancellationScope context manager."""

    @pytest.mark.asyncio
    async def test_scope_creates_token(self):
        """Test scope creates a token on entry."""
        async with CancellationScope() as token:
            assert token is not None
            assert isinstance(token, CancellationToken)
            assert token.is_cancelled is False

    @pytest.mark.asyncio
    async def test_scope_with_parent(self):
        """Test scope creates child of parent."""
        parent = CancellationToken()

        async with CancellationScope(parent=parent) as token:
            assert token._parent is parent
            assert token in parent._children

    @pytest.mark.asyncio
    async def test_scope_cleanup_on_exit(self):
        """Test scope cleans up on exit."""
        parent = CancellationToken()

        async with CancellationScope(parent=parent) as token:
            assert token in parent._children

        # After exit, child should be removed from parent
        assert token not in parent._children

    @pytest.mark.asyncio
    async def test_scope_cleanup_on_exception(self):
        """Test scope cleans up even when exception raised."""
        parent = CancellationToken()
        captured_token = None

        with pytest.raises(ValueError):
            async with CancellationScope(parent=parent) as token:
                captured_token = token
                raise ValueError("Test error")

        # Should still clean up
        assert captured_token not in parent._children

    @pytest.mark.asyncio
    async def test_scope_does_not_suppress_exceptions(self):
        """Test scope doesn't suppress exceptions."""
        with pytest.raises(RuntimeError):
            async with CancellationScope() as token:
                raise RuntimeError("Should propagate")

    @pytest.mark.asyncio
    async def test_scope_with_timeout(self):
        """Test scope with timeout auto-cancels."""
        async with CancellationScope(timeout=0.05) as token:
            await asyncio.sleep(0.1)  # Wait longer than timeout

        assert token.is_cancelled is True
        assert token.reason_type == CancellationReason.TIMEOUT
        assert "Timeout after" in token.reason

    @pytest.mark.asyncio
    async def test_scope_timeout_cancelled_on_early_exit(self):
        """Test timeout is cancelled when scope exits before timeout."""
        async with CancellationScope(timeout=10.0) as token:
            pass  # Exit immediately

        # Token should not be cancelled (we exited before timeout)
        assert token.is_cancelled is False

    @pytest.mark.asyncio
    async def test_scope_token_property(self):
        """Test token property during and after scope."""
        scope = CancellationScope()

        assert scope.token is None

        async with scope as token:
            assert scope.token is token

    @pytest.mark.asyncio
    async def test_scope_parent_cancelled_propagates(self):
        """Test parent cancellation propagates to scope token."""
        parent = CancellationToken()

        async def cancel_parent():
            await asyncio.sleep(0.01)
            parent.cancel(reason="Parent cancelled")

        task = asyncio.create_task(cancel_parent())

        async with CancellationScope(parent=parent) as token:
            await asyncio.sleep(0.05)

        await task

        assert token.is_cancelled is True
        assert token.reason_type == CancellationReason.PARENT_CANCELLED

    @pytest.mark.asyncio
    async def test_scope_no_parent_cleanup_if_no_parent(self):
        """Test scope without parent doesn't error on cleanup."""
        async with CancellationScope() as token:
            pass

        # Should complete without error


class TestCancellationIntegration:
    """Integration tests for cancellation patterns."""

    @pytest.mark.asyncio
    async def test_cooperative_cancellation_pattern(self):
        """Test typical cooperative cancellation pattern."""
        token = CancellationToken()
        iterations_completed = 0

        async def long_running_task():
            nonlocal iterations_completed
            for i in range(100):
                if token.is_cancelled:
                    break
                iterations_completed += 1
                await asyncio.sleep(0.01)

        task = asyncio.create_task(long_running_task())

        await asyncio.sleep(0.05)
        token.cancel(reason="Enough iterations")
        await task

        assert iterations_completed < 100
        assert iterations_completed > 0

    @pytest.mark.asyncio
    async def test_check_pattern_raises(self):
        """Test using check() at cancellation points."""
        token = CancellationToken()
        completed_phases = []

        async def phased_task():
            for phase in ["init", "process", "finalize"]:
                token.check()  # Check at start of each phase
                completed_phases.append(phase)
                if phase == "process":
                    token.cancel(reason="Stop after process")

        with pytest.raises(DebateCancelled):
            await phased_task()

        assert "init" in completed_phases
        assert "process" in completed_phases
        assert "finalize" not in completed_phases

    @pytest.mark.asyncio
    async def test_nested_scopes(self):
        """Test nested cancellation scopes."""
        root = CancellationToken()

        async with CancellationScope(parent=root) as outer:
            async with CancellationScope(parent=outer) as inner:
                assert inner._parent is outer
                assert outer._parent is root

                root.cancel()

                assert outer.is_cancelled is True
                assert inner.is_cancelled is True

    @pytest.mark.asyncio
    async def test_parallel_tasks_with_shared_token(self):
        """Test multiple parallel tasks sharing a cancellation token."""
        token = CancellationToken()
        task_results = {}

        async def worker(name: str):
            count = 0
            while not token.is_cancelled:
                count += 1
                await asyncio.sleep(0.01)
            task_results[name] = count

        tasks = [
            asyncio.create_task(worker("worker1")),
            asyncio.create_task(worker("worker2")),
            asyncio.create_task(worker("worker3")),
        ]

        await asyncio.sleep(0.05)
        token.cancel()
        await asyncio.gather(*tasks)

        # All workers should have run some iterations
        assert all(count > 0 for count in task_results.values())
        # All should have stopped
        assert len(task_results) == 3
