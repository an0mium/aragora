"""
Tests for resilience_patterns.timeout module.

Tests cover:
- Async timeout decorator
- Sync timeout decorator
- Timeout configuration
- Callback invocation
- Context managers
"""

import asyncio
import pytest
from unittest.mock import MagicMock

from aragora.resilience_patterns import (
    TimeoutConfig,
    with_timeout,
    with_timeout_sync,
)
from aragora.resilience_patterns.timeout import TimeoutError as PatternTimeoutError


class TestTimeoutConfig:
    """Tests for TimeoutConfig dataclass."""

    def test_required_seconds(self):
        """Test that seconds is required."""
        config = TimeoutConfig(seconds=5.0)
        assert config.seconds == 5.0
        assert config.on_timeout is None
        assert config.message is None

    def test_custom_config(self):
        """Test custom configuration."""
        callback = MagicMock()
        config = TimeoutConfig(
            seconds=5.0,
            message="Custom timeout message",
            on_timeout=callback,
        )
        assert config.seconds == 5.0
        assert config.message == "Custom timeout message"
        assert config.on_timeout == callback


class TestWithTimeoutAsync:
    """Tests for async with_timeout decorator."""

    @pytest.mark.asyncio
    async def test_fast_operation_completes(self):
        """Test that fast operations complete successfully."""

        @with_timeout(1.0)
        async def fast_op():
            await asyncio.sleep(0.01)
            return "success"

        result = await fast_op()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_slow_operation_times_out(self):
        """Test that slow operations time out."""

        @with_timeout(0.1)
        async def slow_op():
            await asyncio.sleep(1.0)
            return "should not reach"

        with pytest.raises((asyncio.TimeoutError, PatternTimeoutError)):
            await slow_op()

    @pytest.mark.asyncio
    async def test_timeout_with_config(self):
        """Test timeout with TimeoutConfig."""
        config = TimeoutConfig(seconds=0.1)

        @with_timeout(config)
        async def slow_op():
            await asyncio.sleep(1.0)

        with pytest.raises((asyncio.TimeoutError, PatternTimeoutError)):
            await slow_op()

    @pytest.mark.asyncio
    async def test_timeout_callback(self):
        """Test timeout callback invocation."""
        callback_called = False
        operation_name = None

        def on_timeout(op_name):
            nonlocal callback_called, operation_name
            callback_called = True
            operation_name = op_name

        config = TimeoutConfig(
            seconds=0.1,
            on_timeout=on_timeout,
        )

        @with_timeout(config)
        async def slow_op():
            await asyncio.sleep(1.0)

        with pytest.raises((asyncio.TimeoutError, PatternTimeoutError)):
            await slow_op()

        assert callback_called is True
        assert operation_name == "slow_op"

    @pytest.mark.asyncio
    async def test_exception_propagation(self):
        """Test that non-timeout exceptions propagate correctly."""

        @with_timeout(1.0)
        async def error_op():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await error_op()

    @pytest.mark.asyncio
    async def test_return_value_preserved(self):
        """Test that return values are preserved."""

        @with_timeout(1.0)
        async def return_dict():
            return {"key": "value", "number": 42}

        result = await return_dict()
        assert result == {"key": "value", "number": 42}

    @pytest.mark.asyncio
    async def test_zero_timeout(self):
        """Test behavior with zero timeout."""

        @with_timeout(0.0)
        async def instant_op():
            return "instant"

        # Zero timeout should still allow instant completion
        # or immediately timeout - both are acceptable behaviors
        try:
            result = await instant_op()
            assert result == "instant"
        except (asyncio.TimeoutError, PatternTimeoutError):
            pass  # Also acceptable


class TestWithTimeoutSync:
    """Tests for sync with_timeout_sync decorator."""

    def test_fast_operation_completes(self):
        """Test that fast operations complete successfully."""

        @with_timeout_sync(1.0)
        def fast_op():
            return "success"

        result = fast_op()
        assert result == "success"

    def test_exception_propagation(self):
        """Test that exceptions propagate correctly."""

        @with_timeout_sync(1.0)
        def error_op():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            error_op()


class TestPatternTimeoutError:
    """Tests for custom TimeoutError exception."""

    def test_error_message(self):
        """Test error message format."""
        error = PatternTimeoutError(
            message="Operation 'test_op' timed out after 5.0s",
            timeout_seconds=5.0,
            operation="test_operation",
        )
        assert "timed out" in str(error)

    def test_error_attributes(self):
        """Test error attributes."""
        error = PatternTimeoutError(
            message="Timeout",
            timeout_seconds=10.0,
            operation="my_operation",
        )
        assert error.timeout_seconds == 10.0
        assert error.operation == "my_operation"


class TestTimeoutEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_nested_timeouts(self):
        """Test nested timeout decorators."""

        @with_timeout(0.5)
        async def outer():
            @with_timeout(0.1)
            async def inner():
                await asyncio.sleep(0.05)
                return "inner_success"

            return await inner()

        result = await outer()
        assert result == "inner_success"

    @pytest.mark.asyncio
    async def test_timeout_with_multiple_awaits(self):
        """Test timeout across multiple await points."""

        @with_timeout(0.5)
        async def multiple_awaits():
            await asyncio.sleep(0.1)
            await asyncio.sleep(0.1)
            await asyncio.sleep(0.1)
            return "success"

        result = await multiple_awaits()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_timeout_cancellation_cleanup(self):
        """Test that cancellation is handled cleanly."""
        cleanup_called = False

        @with_timeout(0.1)
        async def with_cleanup():
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                nonlocal cleanup_called
                cleanup_called = True
                raise

        with pytest.raises((asyncio.TimeoutError, PatternTimeoutError, asyncio.CancelledError)):
            await with_cleanup()


class TestTimeoutContextManagers:
    """Tests for timeout context managers."""

    @pytest.mark.asyncio
    async def test_timeout_context_success(self):
        """Test async context manager with successful operation."""
        from aragora.resilience_patterns.timeout import timeout_context

        async with timeout_context(1.0, context_name="test"):
            await asyncio.sleep(0.01)
        # Should complete without exception

    @pytest.mark.asyncio
    async def test_timeout_context_timeout(self):
        """Test async context manager with timeout."""
        from aragora.resilience_patterns.timeout import timeout_context

        with pytest.raises((asyncio.TimeoutError, PatternTimeoutError)):
            async with timeout_context(0.1, context_name="slow_op"):
                await asyncio.sleep(1.0)

    @pytest.mark.asyncio
    async def test_timeout_context_callback(self):
        """Test callback invocation on timeout."""
        from aragora.resilience_patterns.timeout import timeout_context

        callback_called = False

        def on_timeout(name):
            nonlocal callback_called
            callback_called = True

        with pytest.raises((asyncio.TimeoutError, PatternTimeoutError)):
            async with timeout_context(0.1, on_timeout=on_timeout, context_name="test_callback"):
                await asyncio.sleep(1.0)

        assert callback_called is True

    @pytest.mark.asyncio
    async def test_timeout_context_preserves_return(self):
        """Test that context manager preserves operations correctly."""
        from aragora.resilience_patterns.timeout import timeout_context

        result = None
        async with timeout_context(1.0):
            result = "success"
            await asyncio.sleep(0.01)

        assert result == "success"
