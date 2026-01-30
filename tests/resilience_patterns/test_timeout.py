"""
Tests for Timeout Pattern.

Tests the timeout implementation including:
- TimeoutConfig dataclass
- TimeoutError exception
- Async and sync decorators
- Context managers
- Callback invocation
"""

from __future__ import annotations

import asyncio
import signal
import sys
from unittest.mock import MagicMock

import pytest

from aragora.resilience.timeout import (
    TimeoutConfig,
    TimeoutError,
    timeout_context,
    timeout_context_sync,
    with_timeout,
    with_timeout_sync,
)


# =============================================================================
# TimeoutError Tests
# =============================================================================


class TestTimeoutError:
    """Test TimeoutError exception."""

    def test_default_message(self):
        """Test default error message."""
        error = TimeoutError()
        assert str(error) == "Operation timed out"
        assert error.timeout_seconds is None
        assert error.operation is None

    def test_custom_message(self):
        """Test custom error message."""
        error = TimeoutError("Custom timeout message")
        assert str(error) == "Custom timeout message"

    def test_with_timeout_seconds(self):
        """Test error with timeout_seconds attribute."""
        error = TimeoutError("Timed out", timeout_seconds=5.0)
        assert error.timeout_seconds == 5.0

    def test_with_operation(self):
        """Test error with operation attribute."""
        error = TimeoutError("Timed out", operation="fetch_data")
        assert error.operation == "fetch_data"

    def test_full_attributes(self):
        """Test error with all attributes."""
        error = TimeoutError(
            "Operation 'test_op' timed out",
            timeout_seconds=10.0,
            operation="test_op",
        )
        assert error.timeout_seconds == 10.0
        assert error.operation == "test_op"

    def test_is_asyncio_timeout_error(self):
        """Test that TimeoutError inherits from asyncio.TimeoutError."""
        error = TimeoutError()
        assert isinstance(error, asyncio.TimeoutError)


# =============================================================================
# TimeoutConfig Tests
# =============================================================================


class TestTimeoutConfig:
    """Test TimeoutConfig dataclass."""

    def test_minimal_config(self):
        """Test minimal configuration with only seconds."""
        config = TimeoutConfig(seconds=5.0)
        assert config.seconds == 5.0
        assert config.on_timeout is None
        assert config.error_class == TimeoutError
        assert config.message is None

    def test_full_config(self):
        """Test full configuration."""
        callback = MagicMock()

        class CustomError(Exception):
            pass

        config = TimeoutConfig(
            seconds=10.0,
            on_timeout=callback,
            error_class=CustomError,
            message="Custom message",
        )
        assert config.seconds == 10.0
        assert config.on_timeout is callback
        assert config.error_class == CustomError
        assert config.message == "Custom message"

    def test_get_message_default(self):
        """Test get_message with default message."""
        config = TimeoutConfig(seconds=5.0)
        message = config.get_message("test_operation")
        assert message == "Operation 'test_operation' timed out after 5.0s"

    def test_get_message_custom(self):
        """Test get_message with custom message."""
        config = TimeoutConfig(seconds=5.0, message="Custom timeout error")
        message = config.get_message("test_operation")
        assert message == "Custom timeout error"


# =============================================================================
# with_timeout (Async Decorator) Tests
# =============================================================================


class TestWithTimeoutAsync:
    """Test with_timeout async decorator."""

    @pytest.mark.asyncio
    async def test_success_passthrough(self):
        """Test that successful operations pass through."""

        @with_timeout(5.0)
        async def fast_operation():
            return "success"

        result = await fast_operation()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_timeout_triggers(self):
        """Test that timeout triggers correctly."""

        @with_timeout(0.1)
        async def slow_operation():
            await asyncio.sleep(1.0)
            return "never reached"

        with pytest.raises(TimeoutError) as exc_info:
            await slow_operation()

        assert exc_info.value.timeout_seconds == 0.1
        assert exc_info.value.operation == "slow_operation"

    @pytest.mark.asyncio
    async def test_float_parameter(self):
        """Test timeout with float parameter."""

        @with_timeout(0.5)
        async def operation():
            return "done"

        result = await operation()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_config_parameter(self):
        """Test timeout with TimeoutConfig parameter."""
        config = TimeoutConfig(seconds=0.5)

        @with_timeout(config)
        async def operation():
            return "done"

        result = await operation()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_callback_invoked_on_timeout(self):
        """Test that callback is invoked on timeout."""
        callback = MagicMock()
        config = TimeoutConfig(seconds=0.1, on_timeout=callback)

        @with_timeout(config)
        async def slow_operation():
            await asyncio.sleep(1.0)

        with pytest.raises(TimeoutError):
            await slow_operation()

        callback.assert_called_once_with("slow_operation")

    @pytest.mark.asyncio
    async def test_callback_error_handled(self):
        """Test that callback errors are handled gracefully."""

        def bad_callback(operation: str):
            raise ValueError("Callback error")

        config = TimeoutConfig(seconds=0.1, on_timeout=bad_callback)

        @with_timeout(config)
        async def slow_operation():
            await asyncio.sleep(1.0)

        # Should still raise TimeoutError, not ValueError
        with pytest.raises(TimeoutError):
            await slow_operation()

    @pytest.mark.asyncio
    async def test_custom_error_class(self):
        """Test timeout with custom error class."""

        class CustomTimeoutError(TimeoutError):
            pass

        config = TimeoutConfig(seconds=0.1, error_class=CustomTimeoutError)

        @with_timeout(config)
        async def slow_operation():
            await asyncio.sleep(1.0)

        with pytest.raises(CustomTimeoutError):
            await slow_operation()

    @pytest.mark.asyncio
    async def test_custom_message(self):
        """Test timeout with custom message."""
        config = TimeoutConfig(seconds=0.1, message="API request timed out")

        @with_timeout(config)
        async def slow_operation():
            await asyncio.sleep(1.0)

        with pytest.raises(TimeoutError) as exc_info:
            await slow_operation()

        assert str(exc_info.value) == "API request timed out"

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""

        @with_timeout(5.0)
        async def documented_function():
            """This is the docstring."""
            return "result"

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is the docstring."


# =============================================================================
# with_timeout_sync (Sync Decorator) Tests
# =============================================================================


class TestWithTimeoutSync:
    """Test with_timeout_sync decorator."""

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM not available on this platform",
    )
    def test_success_passthrough(self):
        """Test that successful operations pass through."""

        @with_timeout_sync(5.0)
        def fast_operation():
            return "success"

        result = fast_operation()
        assert result == "success"

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM not available on this platform",
    )
    def test_timeout_triggers(self):
        """Test that timeout triggers correctly."""
        import time

        @with_timeout_sync(0.1)
        def slow_operation():
            time.sleep(1.0)
            return "never reached"

        with pytest.raises(TimeoutError) as exc_info:
            slow_operation()

        assert exc_info.value.timeout_seconds == 0.1
        assert exc_info.value.operation == "slow_operation"

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM not available on this platform",
    )
    def test_callback_invoked_on_timeout(self):
        """Test that callback is invoked on timeout."""
        import time

        callback = MagicMock()
        config = TimeoutConfig(seconds=0.1, on_timeout=callback)

        @with_timeout_sync(config)
        def slow_operation():
            time.sleep(1.0)

        with pytest.raises(TimeoutError):
            slow_operation()

        callback.assert_called_once_with("slow_operation")

    def test_non_unix_graceful_degradation(self):
        """Test graceful degradation on non-Unix systems."""
        # This test verifies the behavior when SIGALRM is not available
        # On Unix, the decorated function still works without timeout enforcement

        @with_timeout_sync(5.0)
        def operation():
            return "completed"

        # Should complete without error regardless of platform
        result = operation()
        assert result == "completed"

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM not available on this platform",
    )
    def test_signal_handler_cleanup(self):
        """Test that signal handler is properly cleaned up after success."""
        import time

        original_handler = signal.getsignal(signal.SIGALRM)

        @with_timeout_sync(5.0)
        def fast_operation():
            return "done"

        result = fast_operation()
        assert result == "done"

        # Signal handler should be restored
        current_handler = signal.getsignal(signal.SIGALRM)
        assert current_handler == original_handler


# =============================================================================
# timeout_context (Async Context Manager) Tests
# =============================================================================


class TestTimeoutContextAsync:
    """Test timeout_context async context manager."""

    @pytest.mark.asyncio
    async def test_success_case(self):
        """Test successful operation within context."""
        async with timeout_context(5.0, context_name="test"):
            result = "success"
        assert result == "success"

    @pytest.mark.asyncio
    async def test_timeout_case(self):
        """Test timeout within context."""
        with pytest.raises(TimeoutError) as exc_info:
            async with timeout_context(0.1, context_name="slow_context"):
                await asyncio.sleep(1.0)

        assert exc_info.value.timeout_seconds == 0.1
        assert exc_info.value.operation == "slow_context"

    @pytest.mark.asyncio
    async def test_callback_invoked(self):
        """Test that callback is invoked on timeout."""
        callback = MagicMock()

        with pytest.raises(TimeoutError):
            async with timeout_context(0.1, on_timeout=callback, context_name="test"):
                await asyncio.sleep(1.0)

        callback.assert_called_once_with("test")

    @pytest.mark.asyncio
    async def test_callback_error_handled(self):
        """Test that callback errors are handled gracefully."""

        def bad_callback(name: str):
            raise ValueError("Callback error")

        with pytest.raises(TimeoutError):
            async with timeout_context(0.1, on_timeout=bad_callback, context_name="test"):
                await asyncio.sleep(1.0)

    @pytest.mark.asyncio
    async def test_default_context_name(self):
        """Test default context name."""
        with pytest.raises(TimeoutError) as exc_info:
            async with timeout_context(0.1):
                await asyncio.sleep(1.0)

        assert exc_info.value.operation == "operation"


# =============================================================================
# timeout_context_sync (Sync Context Manager) Tests
# =============================================================================


class TestTimeoutContextSync:
    """Test timeout_context_sync context manager."""

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM not available on this platform",
    )
    def test_success_case(self):
        """Test successful operation within context."""
        with timeout_context_sync(5.0, context_name="test"):
            result = "success"
        assert result == "success"

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM not available on this platform",
    )
    def test_timeout_case(self):
        """Test timeout within context."""
        import time

        with pytest.raises(TimeoutError) as exc_info:
            with timeout_context_sync(0.1, context_name="slow_context"):
                time.sleep(1.0)

        assert exc_info.value.timeout_seconds == 0.1
        assert exc_info.value.operation == "slow_context"

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM not available on this platform",
    )
    def test_callback_invoked(self):
        """Test that callback is invoked on timeout."""
        import time

        callback = MagicMock()

        with pytest.raises(TimeoutError):
            with timeout_context_sync(0.1, on_timeout=callback, context_name="test"):
                time.sleep(1.0)

        callback.assert_called_once_with("test")

    @pytest.mark.skipif(
        not hasattr(signal, "SIGALRM"),
        reason="SIGALRM not available on this platform",
    )
    def test_signal_handler_cleanup(self):
        """Test that signal handler is properly cleaned up."""
        original_handler = signal.getsignal(signal.SIGALRM)

        with timeout_context_sync(5.0, context_name="test"):
            pass

        current_handler = signal.getsignal(signal.SIGALRM)
        assert current_handler == original_handler

    def test_non_unix_graceful_degradation(self):
        """Test graceful degradation on non-Unix systems."""
        # Context should complete without error
        with timeout_context_sync(5.0, context_name="test"):
            result = "completed"
        assert result == "completed"
