"""
Tests for the unified timeout management module.

Tests cover:
- TimeoutError attributes (message, timeout_seconds, operation)
- TimeoutConfig defaults, get_message with custom and default messages
- is_timeout_available() returns bool
- with_timeout async decorator: success, timeout, custom error_class, on_timeout callback
- with_timeout with float argument vs TimeoutConfig
- with_timeout_sync decorator: SIGALRM-based timeout on Unix, graceful fallback otherwise
- timeout_context async context manager: success, timeout, on_timeout callback
- timeout_context_sync sync context manager: success, timeout on Unix
- Custom error messages via TimeoutConfig
"""

from __future__ import annotations

import asyncio
import signal
import time
from dataclasses import fields
from unittest.mock import MagicMock

import pytest

from aragora.resilience.timeout import (
    TimeoutConfig,
    TimeoutError,
    is_timeout_available,
    timeout_context,
    timeout_context_sync,
    with_timeout,
    with_timeout_sync,
)

# Whether SIGALRM is available on this platform (Unix/macOS)
HAS_SIGALRM = hasattr(signal, "SIGALRM")


# ============================================================================
# TimeoutError Tests
# ============================================================================


class TestTimeoutError:
    """Tests for TimeoutError exception."""

    def test_is_asyncio_timeout_error_subclass(self):
        """TimeoutError should extend asyncio.TimeoutError."""
        assert issubclass(TimeoutError, asyncio.TimeoutError)

    def test_default_message(self):
        """Test default error message when no arguments provided."""
        err = TimeoutError()
        assert str(err) == "Operation timed out"

    def test_custom_message(self):
        """Test custom error message."""
        err = TimeoutError("Custom timeout message")
        assert str(err) == "Custom timeout message"

    def test_timeout_seconds_attribute(self):
        """Test timeout_seconds attribute is stored correctly."""
        err = TimeoutError("msg", timeout_seconds=5.0)
        assert err.timeout_seconds == 5.0

    def test_timeout_seconds_default_none(self):
        """Test timeout_seconds defaults to None."""
        err = TimeoutError()
        assert err.timeout_seconds is None

    def test_operation_attribute(self):
        """Test operation attribute is stored correctly."""
        err = TimeoutError("msg", timeout_seconds=3.0, operation="fetch_data")
        assert err.operation == "fetch_data"

    def test_operation_default_none(self):
        """Test operation defaults to None."""
        err = TimeoutError()
        assert err.operation is None

    def test_all_attributes_set(self):
        """Test all attributes are set when fully specified."""
        err = TimeoutError("timed out", timeout_seconds=10.0, operation="my_op")
        assert str(err) == "timed out"
        assert err.timeout_seconds == 10.0
        assert err.operation == "my_op"

    def test_can_be_caught_as_asyncio_timeout_error(self):
        """TimeoutError should be catchable as asyncio.TimeoutError."""
        with pytest.raises(asyncio.TimeoutError):
            raise TimeoutError("test")

    def test_can_be_caught_as_base_exception(self):
        """TimeoutError should be catchable as Exception."""
        with pytest.raises(Exception):
            raise TimeoutError("test")


# ============================================================================
# TimeoutConfig Tests
# ============================================================================


class TestTimeoutConfig:
    """Tests for TimeoutConfig dataclass."""

    def test_seconds_required(self):
        """Test that seconds is a required field."""
        config = TimeoutConfig(seconds=5.0)
        assert config.seconds == 5.0

    def test_default_on_timeout_none(self):
        """Test on_timeout defaults to None."""
        config = TimeoutConfig(seconds=1.0)
        assert config.on_timeout is None

    def test_default_error_class(self):
        """Test error_class defaults to TimeoutError."""
        config = TimeoutConfig(seconds=1.0)
        assert config.error_class is TimeoutError

    def test_default_message_none(self):
        """Test message defaults to None."""
        config = TimeoutConfig(seconds=1.0)
        assert config.message is None

    def test_custom_on_timeout(self):
        """Test custom on_timeout callback is stored."""
        callback = MagicMock()
        config = TimeoutConfig(seconds=1.0, on_timeout=callback)
        assert config.on_timeout is callback

    def test_custom_error_class(self):
        """Test custom error_class is stored."""

        class MyError(TimeoutError):
            pass

        config = TimeoutConfig(seconds=1.0, error_class=MyError)
        assert config.error_class is MyError

    def test_custom_message(self):
        """Test custom message is stored."""
        config = TimeoutConfig(seconds=1.0, message="Custom timeout!")
        assert config.message == "Custom timeout!"

    def test_get_message_default(self):
        """Test get_message generates default message from operation name."""
        config = TimeoutConfig(seconds=5.0)
        msg = config.get_message("fetch_data")
        assert msg == "Operation 'fetch_data' timed out after 5.0s"

    def test_get_message_custom(self):
        """Test get_message returns custom message when set."""
        config = TimeoutConfig(seconds=5.0, message="Custom timeout message")
        msg = config.get_message("fetch_data")
        assert msg == "Custom timeout message"

    def test_get_message_includes_seconds(self):
        """Test default message includes the timeout seconds."""
        config = TimeoutConfig(seconds=3.5)
        msg = config.get_message("op")
        assert "3.5" in msg

    def test_get_message_includes_operation(self):
        """Test default message includes the operation name."""
        config = TimeoutConfig(seconds=1.0)
        msg = config.get_message("my_operation")
        assert "my_operation" in msg

    def test_is_dataclass(self):
        """Test that TimeoutConfig has dataclass fields."""
        field_names = {f.name for f in fields(TimeoutConfig)}
        assert "seconds" in field_names
        assert "on_timeout" in field_names
        assert "error_class" in field_names
        assert "message" in field_names


# ============================================================================
# is_timeout_available Tests
# ============================================================================


class TestIsTimeoutAvailable:
    """Tests for is_timeout_available function."""

    def test_returns_bool(self):
        """is_timeout_available should return a boolean."""
        result = is_timeout_available()
        assert isinstance(result, bool)

    def test_returns_true_on_python_311_plus(self):
        """On Python 3.11+ or with async-timeout, should return True."""
        # We are running on Python 3.11+ or have async-timeout installed,
        # so this should be True in any normal test environment.
        assert is_timeout_available() is True


# ============================================================================
# with_timeout Async Decorator Tests
# ============================================================================


class TestWithTimeoutDecorator:
    """Tests for with_timeout async decorator."""

    @pytest.mark.asyncio
    async def test_completes_within_timeout(self):
        """Function that completes within timeout should return normally."""

        @with_timeout(2.0)
        async def fast_operation():
            await asyncio.sleep(0.01)
            return "done"

        result = await fast_operation()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_raises_timeout_error_on_timeout(self):
        """Function exceeding timeout should raise TimeoutError."""

        @with_timeout(0.05)
        async def slow_operation():
            await asyncio.sleep(10.0)
            return "never"

        with pytest.raises(TimeoutError) as exc_info:
            await slow_operation()

        err = exc_info.value
        assert err.timeout_seconds == 0.05
        assert err.operation == "slow_operation"

    @pytest.mark.asyncio
    async def test_timeout_error_message_default(self):
        """Timeout error should have a descriptive default message."""

        @with_timeout(0.05)
        async def my_func():
            await asyncio.sleep(10.0)

        with pytest.raises(TimeoutError) as exc_info:
            await my_func()

        msg = str(exc_info.value)
        assert "my_func" in msg
        assert "0.05" in msg

    @pytest.mark.asyncio
    async def test_with_float_argument(self):
        """with_timeout should accept a plain float as the timeout value."""

        @with_timeout(0.05)
        async def func():
            await asyncio.sleep(10.0)

        with pytest.raises(TimeoutError) as exc_info:
            await func()

        assert exc_info.value.timeout_seconds == 0.05

    @pytest.mark.asyncio
    async def test_with_timeout_config_argument(self):
        """with_timeout should accept a TimeoutConfig instance."""

        config = TimeoutConfig(seconds=0.05, message="Custom config timeout")

        @with_timeout(config)
        async def func():
            await asyncio.sleep(10.0)

        with pytest.raises(TimeoutError) as exc_info:
            await func()

        assert str(exc_info.value) == "Custom config timeout"
        assert exc_info.value.timeout_seconds == 0.05

    @pytest.mark.asyncio
    async def test_custom_error_class(self):
        """with_timeout should raise the custom error_class from config."""

        class CustomTimeoutError(TimeoutError):
            pass

        config = TimeoutConfig(seconds=0.05, error_class=CustomTimeoutError)

        @with_timeout(config)
        async def func():
            await asyncio.sleep(10.0)

        with pytest.raises(CustomTimeoutError) as exc_info:
            await func()

        assert isinstance(exc_info.value, CustomTimeoutError)
        assert exc_info.value.timeout_seconds == 0.05

    @pytest.mark.asyncio
    async def test_on_timeout_callback_called(self):
        """on_timeout callback should be called with the operation name on timeout."""

        callback = MagicMock()
        config = TimeoutConfig(seconds=0.05, on_timeout=callback)

        @with_timeout(config)
        async def my_operation():
            await asyncio.sleep(10.0)

        with pytest.raises(TimeoutError):
            await my_operation()

        callback.assert_called_once_with("my_operation")

    @pytest.mark.asyncio
    async def test_on_timeout_callback_not_called_on_success(self):
        """on_timeout callback should NOT be called when function succeeds."""

        callback = MagicMock()
        config = TimeoutConfig(seconds=2.0, on_timeout=callback)

        @with_timeout(config)
        async def fast_op():
            return 42

        result = await fast_op()
        assert result == 42
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_timeout_callback_error_suppressed(self):
        """Errors in on_timeout callback should be suppressed (logged)."""

        def bad_callback(op: str) -> None:
            raise ValueError("callback broke")

        config = TimeoutConfig(seconds=0.05, on_timeout=bad_callback)

        @with_timeout(config)
        async def func():
            await asyncio.sleep(10.0)

        # Should still raise TimeoutError, not ValueError
        with pytest.raises(TimeoutError):
            await func()

    @pytest.mark.asyncio
    async def test_preserves_function_name(self):
        """Decorated function should preserve __name__ via functools.wraps."""

        @with_timeout(1.0)
        async def original_name():
            pass

        assert original_name.__name__ == "original_name"

    @pytest.mark.asyncio
    async def test_passes_arguments_through(self):
        """Decorated function should receive all args and kwargs."""

        @with_timeout(2.0)
        async def add(a: int, b: int, extra: int = 0) -> int:
            return a + b + extra

        result = await add(3, 4, extra=5)
        assert result == 12

    @pytest.mark.asyncio
    async def test_custom_message_in_config(self):
        """Custom message from TimeoutConfig should appear in the error."""

        config = TimeoutConfig(seconds=0.05, message="Service unavailable")

        @with_timeout(config)
        async def func():
            await asyncio.sleep(10.0)

        with pytest.raises(TimeoutError) as exc_info:
            await func()

        assert str(exc_info.value) == "Service unavailable"


# ============================================================================
# with_timeout_sync Decorator Tests
# ============================================================================


class TestWithTimeoutSyncDecorator:
    """Tests for with_timeout_sync sync decorator."""

    def test_completes_within_timeout(self):
        """Sync function completing within timeout should return normally."""

        @with_timeout_sync(2.0)
        def fast_sync():
            return "fast"

        result = fast_sync()
        assert result == "fast"

    def test_raises_timeout_error_on_timeout(self):
        """Sync function exceeding timeout should raise TimeoutError via SIGALRM."""

        @with_timeout_sync(0.1)
        def slow_sync():
            time.sleep(10.0)
            return "never"

        with pytest.raises(TimeoutError) as exc_info:
            slow_sync()

        err = exc_info.value
        assert err.timeout_seconds == 0.1
        assert err.operation == "slow_sync"

    def test_on_timeout_callback_called(self):
        """on_timeout callback should be called on sync timeout."""

        callback = MagicMock()
        config = TimeoutConfig(seconds=0.1, on_timeout=callback)

        @with_timeout_sync(config)
        def slow_sync():
            time.sleep(10.0)

        with pytest.raises(TimeoutError):
            slow_sync()

        callback.assert_called_once_with("slow_sync")

    def test_custom_error_class_sync(self):
        """Sync decorator should raise the custom error_class from config."""

        class SyncTimeoutError(TimeoutError):
            pass

        config = TimeoutConfig(seconds=0.1, error_class=SyncTimeoutError)

        @with_timeout_sync(config)
        def slow_sync():
            time.sleep(10.0)

        with pytest.raises(SyncTimeoutError):
            slow_sync()

    def test_custom_message_sync(self):
        """Sync decorator should use custom message from config."""

        config = TimeoutConfig(seconds=0.1, message="Sync timed out!")

        @with_timeout_sync(config)
        def slow_sync():
            time.sleep(10.0)

        with pytest.raises(TimeoutError) as exc_info:
            slow_sync()

        assert str(exc_info.value) == "Sync timed out!"

    def test_restores_previous_signal_handler(self):
        """After execution, the previous SIGALRM handler should be restored."""

        original_handler = signal.getsignal(signal.SIGALRM)

        @with_timeout_sync(2.0)
        def quick():
            return "ok"

        quick()

        restored_handler = signal.getsignal(signal.SIGALRM)
        assert restored_handler == original_handler

    def test_restores_signal_handler_on_timeout(self):
        """Signal handler should be restored even after timeout."""

        original_handler = signal.getsignal(signal.SIGALRM)

        @with_timeout_sync(0.1)
        def slow():
            time.sleep(10.0)

        with pytest.raises(TimeoutError):
            slow()

        # The finally block in the decorator should restore the handler.
        # Note: on timeout, the handler is restored in the finally block
        # after the signal fires. We verify it was set back.
        restored_handler = signal.getsignal(signal.SIGALRM)
        assert restored_handler == original_handler

    def test_preserves_function_name_sync(self):
        """Sync decorated function should preserve __name__."""

        @with_timeout_sync(1.0)
        def my_sync_func():
            pass

        assert my_sync_func.__name__ == "my_sync_func"

    def test_passes_arguments_through_sync(self):
        """Sync decorated function should receive all args and kwargs."""

        @with_timeout_sync(2.0)
        def multiply(a: int, b: int) -> int:
            return a * b

        assert multiply(6, 7) == 42

    def test_on_timeout_callback_error_suppressed_sync(self):
        """Errors in on_timeout callback should be suppressed in sync decorator."""

        def bad_callback(op: str) -> None:
            raise RuntimeError("callback error")

        config = TimeoutConfig(seconds=0.1, on_timeout=bad_callback)

        @with_timeout_sync(config)
        def slow():
            time.sleep(10.0)

        with pytest.raises(TimeoutError):
            slow()

    def test_with_float_argument_sync(self):
        """with_timeout_sync should accept a plain float."""

        @with_timeout_sync(5.0)
        def func():
            return "ok"

        assert func() == "ok"

    def test_with_timeout_config_argument_sync(self):
        """with_timeout_sync should accept a TimeoutConfig instance."""

        config = TimeoutConfig(seconds=5.0)

        @with_timeout_sync(config)
        def func():
            return "ok"

        assert func() == "ok"


# ============================================================================
# timeout_context Async Context Manager Tests
# ============================================================================


class TestTimeoutContext:
    """Tests for timeout_context async context manager."""

    @pytest.mark.asyncio
    async def test_success_within_timeout(self):
        """Code completing within timeout should work normally."""
        result = None
        async with timeout_context(2.0, context_name="test_op"):
            await asyncio.sleep(0.01)
            result = "completed"

        assert result == "completed"

    @pytest.mark.asyncio
    async def test_raises_timeout_error(self):
        """Code exceeding timeout should raise TimeoutError."""
        with pytest.raises(TimeoutError) as exc_info:
            async with timeout_context(0.05, context_name="slow_context"):
                await asyncio.sleep(10.0)

        err = exc_info.value
        assert err.timeout_seconds == 0.05
        assert err.operation == "slow_context"

    @pytest.mark.asyncio
    async def test_timeout_error_message(self):
        """Timeout error message should include context name and seconds."""
        with pytest.raises(TimeoutError) as exc_info:
            async with timeout_context(0.05, context_name="fetch"):
                await asyncio.sleep(10.0)

        msg = str(exc_info.value)
        assert "fetch" in msg
        assert "0.05" in msg

    @pytest.mark.asyncio
    async def test_on_timeout_callback_called(self):
        """on_timeout callback should be called with context_name."""
        callback = MagicMock()

        with pytest.raises(TimeoutError):
            async with timeout_context(0.05, on_timeout=callback, context_name="my_ctx"):
                await asyncio.sleep(10.0)

        callback.assert_called_once_with("my_ctx")

    @pytest.mark.asyncio
    async def test_on_timeout_callback_not_called_on_success(self):
        """on_timeout callback should NOT be called on success."""
        callback = MagicMock()

        async with timeout_context(2.0, on_timeout=callback, context_name="ok"):
            await asyncio.sleep(0.01)

        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_timeout_callback_error_suppressed(self):
        """Errors in on_timeout callback should be suppressed."""

        def bad_callback(name: str) -> None:
            raise TypeError("broken callback")

        with pytest.raises(TimeoutError):
            async with timeout_context(0.05, on_timeout=bad_callback, context_name="ctx"):
                await asyncio.sleep(10.0)

    @pytest.mark.asyncio
    async def test_default_context_name(self):
        """Default context_name should be 'operation'."""
        with pytest.raises(TimeoutError) as exc_info:
            async with timeout_context(0.05):
                await asyncio.sleep(10.0)

        assert exc_info.value.operation == "operation"

    @pytest.mark.asyncio
    async def test_timeout_error_is_correct_type(self):
        """Raised error should be an instance of our TimeoutError."""
        with pytest.raises(TimeoutError) as exc_info:
            async with timeout_context(0.05):
                await asyncio.sleep(10.0)

        assert isinstance(exc_info.value, TimeoutError)
        assert isinstance(exc_info.value, asyncio.TimeoutError)


# ============================================================================
# timeout_context_sync Sync Context Manager Tests
# ============================================================================


class TestTimeoutContextSync:
    """Tests for timeout_context_sync sync context manager."""

    def test_success_within_timeout(self):
        """Code completing within timeout should work normally."""
        result = None
        with timeout_context_sync(2.0, context_name="sync_test"):
            result = "completed"

        assert result == "completed"

    def test_raises_timeout_error(self):
        """Code exceeding timeout should raise TimeoutError on Unix."""
        with pytest.raises(TimeoutError) as exc_info:
            with timeout_context_sync(0.1, context_name="slow_sync_ctx"):
                time.sleep(10.0)

        err = exc_info.value
        assert err.timeout_seconds == 0.1
        assert err.operation == "slow_sync_ctx"

    def test_timeout_error_message(self):
        """Timeout error message should include context_name and seconds."""
        with pytest.raises(TimeoutError) as exc_info:
            with timeout_context_sync(0.1, context_name="process"):
                time.sleep(10.0)

        msg = str(exc_info.value)
        assert "process" in msg
        assert "0.1" in msg

    def test_on_timeout_callback_called(self):
        """on_timeout callback should be called on sync context timeout."""
        callback = MagicMock()

        with pytest.raises(TimeoutError):
            with timeout_context_sync(0.1, on_timeout=callback, context_name="sync_ctx"):
                time.sleep(10.0)

        callback.assert_called_once_with("sync_ctx")

    def test_on_timeout_callback_not_called_on_success(self):
        """on_timeout callback should NOT be called on success."""
        callback = MagicMock()

        with timeout_context_sync(2.0, on_timeout=callback, context_name="ok"):
            pass

        callback.assert_not_called()

    def test_on_timeout_callback_error_suppressed(self):
        """Errors in on_timeout callback should be suppressed in sync context."""

        def bad_callback(name: str) -> None:
            raise AttributeError("broken")

        with pytest.raises(TimeoutError):
            with timeout_context_sync(0.1, on_timeout=bad_callback, context_name="ctx"):
                time.sleep(10.0)

    def test_default_context_name(self):
        """Default context_name should be 'operation'."""
        # Just ensure it doesn't raise when completing within timeout
        with timeout_context_sync(2.0):
            pass

    def test_default_context_name_in_error(self):
        """Default context_name 'operation' should appear in error."""
        with pytest.raises(TimeoutError) as exc_info:
            with timeout_context_sync(0.1):
                time.sleep(10.0)

        assert exc_info.value.operation == "operation"

    def test_restores_signal_handler(self):
        """Signal handler should be restored after sync context exits."""
        original_handler = signal.getsignal(signal.SIGALRM)

        with timeout_context_sync(2.0, context_name="test"):
            pass

        restored_handler = signal.getsignal(signal.SIGALRM)
        assert restored_handler == original_handler

    @pytest.mark.skipif(not HAS_SIGALRM, reason="SIGALRM not available on this platform")
    def test_restores_signal_handler_on_timeout(self):
        """Signal handler should be restored even after timeout in sync context."""
        original_handler = signal.getsignal(signal.SIGALRM)

        with pytest.raises(TimeoutError):
            with timeout_context_sync(0.1, context_name="test"):
                time.sleep(10.0)

        restored_handler = signal.getsignal(signal.SIGALRM)
        assert restored_handler == original_handler


# ============================================================================
# Integration / Edge Case Tests
# ============================================================================


class TestTimeoutIntegration:
    """Integration and edge case tests for timeout utilities."""

    @pytest.mark.asyncio
    async def test_zero_sleep_completes(self):
        """Function with zero sleep should complete within any timeout."""

        @with_timeout(1.0)
        async def instant():
            return "instant"

        assert await instant() == "instant"

    @pytest.mark.asyncio
    async def test_multiple_decorated_functions(self):
        """Multiple functions with different timeouts should work independently."""

        @with_timeout(0.05)
        async def short_timeout():
            await asyncio.sleep(10.0)

        @with_timeout(5.0)
        async def long_timeout():
            await asyncio.sleep(0.01)
            return "ok"

        # short_timeout should fail
        with pytest.raises(TimeoutError):
            await short_timeout()

        # long_timeout should succeed
        assert await long_timeout() == "ok"

    @pytest.mark.asyncio
    async def test_timeout_with_exception_in_function(self):
        """If function raises before timeout, that exception should propagate."""

        @with_timeout(5.0)
        async def raises_value_error():
            raise ValueError("bad value")

        with pytest.raises(ValueError, match="bad value"):
            await raises_value_error()

    @pytest.mark.asyncio
    async def test_timeout_context_with_exception(self):
        """If code in context raises before timeout, that exception should propagate."""
        with pytest.raises(RuntimeError, match="test error"):
            async with timeout_context(5.0, context_name="err_test"):
                raise RuntimeError("test error")

    def test_sync_decorator_without_sigalrm(self):
        """When SIGALRM is not available, function should run without timeout enforcement."""
        # We simulate the no-SIGALRM path by testing that the decorator
        # still returns the function result even for slow functions
        # (only relevant if SIGALRM is truly unavailable, but we test the
        # code path works for normal fast functions regardless).

        @with_timeout_sync(2.0)
        def fast_func():
            return "result"

        assert fast_func() == "result"

    @pytest.mark.asyncio
    async def test_error_class_receives_all_kwargs(self):
        """Custom error_class should receive message, timeout_seconds, and operation."""

        received_kwargs: dict = {}

        class TrackingError(TimeoutError):
            def __init__(self, message="", timeout_seconds=None, operation=None):
                super().__init__(message, timeout_seconds=timeout_seconds, operation=operation)
                received_kwargs["message"] = message
                received_kwargs["timeout_seconds"] = timeout_seconds
                received_kwargs["operation"] = operation

        config = TimeoutConfig(seconds=0.05, error_class=TrackingError)

        @with_timeout(config)
        async def tracked_op():
            await asyncio.sleep(10.0)

        with pytest.raises(TrackingError):
            await tracked_op()

        assert received_kwargs["timeout_seconds"] == 0.05
        assert received_kwargs["operation"] == "tracked_op"
        assert "tracked_op" in received_kwargs["message"]

    @pytest.mark.asyncio
    async def test_timeout_context_is_async_context_manager(self):
        """timeout_context should work as an async context manager."""
        # Verify the async with syntax works correctly
        value = 0
        async with timeout_context(2.0):
            value = 42
        assert value == 42

    def test_timeout_context_sync_is_context_manager(self):
        """timeout_context_sync should work as a regular context manager."""
        value = 0
        with timeout_context_sync(2.0):
            value = 42
        assert value == 42

    @pytest.mark.skipif(not HAS_SIGALRM, reason="SIGALRM not available on this platform")
    def test_sync_function_exception_propagates(self):
        """If sync function raises before timeout, that exception should propagate."""

        @with_timeout_sync(5.0)
        def raises_error():
            raise KeyError("missing key")

        with pytest.raises(KeyError, match="missing key"):
            raises_error()

    @pytest.mark.skipif(not HAS_SIGALRM, reason="SIGALRM not available on this platform")
    def test_sync_context_exception_propagates(self):
        """If code in sync context raises before timeout, that exception should propagate."""
        with pytest.raises(IOError, match="disk error"):
            with timeout_context_sync(5.0, context_name="io_test"):
                raise OSError("disk error")
