"""
Tests for error handling decorators in aragora.agents.errors.decorators.

Covers:
- Retry delay calculation with jitter
- Error handler functions
- handle_agent_errors decorator (retry, circuit breaker)
- with_error_handling decorator (sync/async)
- handle_stream_errors decorator
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import aiohttp

from aragora.agents.errors.decorators import (
    calculate_retry_delay_with_jitter,
    _build_error_action,
    _handle_timeout_error,
    _handle_connection_error,
    _handle_payload_error,
    _handle_response_error,
    _handle_agent_error,
    _handle_json_error,
    _handle_unexpected_error,
    handle_agent_errors,
    with_error_handling,
    handle_stream_errors,
)
from aragora.agents.errors.classifier import ErrorContext, ErrorAction
from aragora.agents.errors.exceptions import (
    AgentError,
    AgentAPIError,
    AgentConnectionError,
    AgentRateLimitError,
    AgentResponseError,
    AgentStreamError,
    AgentTimeoutError,
    AgentCircuitOpenError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def error_context():
    """Create a standard error context for testing."""
    return ErrorContext(
        agent_name="test_agent",
        attempt=1,
        max_retries=3,
        retry_delay=1.0,
        max_delay=30.0,
        timeout=60.0,
    )


@pytest.fixture
def mock_circuit_breaker():
    """Create a mock circuit breaker."""
    cb = MagicMock()
    cb.can_proceed.return_value = True
    cb.cooldown_seconds = 30
    return cb


# =============================================================================
# calculate_retry_delay_with_jitter Tests
# =============================================================================


class TestCalculateRetryDelayWithJitter:
    """Tests for exponential backoff delay calculation."""

    def test_initial_delay_is_base_delay(self):
        """First attempt should use approximately base delay."""
        delay = calculate_retry_delay_with_jitter(
            attempt=0, base_delay=1.0, max_delay=30.0, jitter_factor=0.0
        )
        assert delay == 1.0

    def test_exponential_increase(self):
        """Delay should double with each attempt (no jitter)."""
        delays = [
            calculate_retry_delay_with_jitter(
                attempt=i, base_delay=1.0, max_delay=100.0, jitter_factor=0.0
            )
            for i in range(5)
        ]
        assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]

    def test_max_delay_cap(self):
        """Delay should not exceed max_delay."""
        delay = calculate_retry_delay_with_jitter(
            attempt=10, base_delay=1.0, max_delay=30.0, jitter_factor=0.0
        )
        assert delay == 30.0

    def test_jitter_adds_variance(self):
        """Jitter should add random variance to delay."""
        delays = [
            calculate_retry_delay_with_jitter(
                attempt=0, base_delay=10.0, max_delay=100.0, jitter_factor=0.3
            )
            for _ in range(50)
        ]

        # With 30% jitter, delays should be in range [7.0, 13.0]
        assert all(7.0 <= d <= 13.0 for d in delays)

        # Should have variance (very unlikely all same with jitter)
        assert len(set(round(d, 2) for d in delays)) > 1

    def test_minimum_delay_is_0_1(self):
        """Delay should never be less than 0.1 seconds."""
        # Use negative jitter to try to push below minimum
        delay = calculate_retry_delay_with_jitter(
            attempt=0, base_delay=0.05, max_delay=30.0, jitter_factor=0.9
        )
        assert delay >= 0.1

    def test_zero_jitter_is_deterministic(self):
        """Zero jitter should produce deterministic delays."""
        delays = [
            calculate_retry_delay_with_jitter(
                attempt=2, base_delay=1.0, max_delay=100.0, jitter_factor=0.0
            )
            for _ in range(10)
        ]
        assert len(set(delays)) == 1


# =============================================================================
# _build_error_action Tests
# =============================================================================


class TestBuildErrorAction:
    """Tests for error action building helper."""

    def test_retryable_error_should_retry(self, error_context):
        """Should retry for retryable exception within max_retries."""
        error = AgentConnectionError("Connection failed")
        retryable = (AgentConnectionError, AgentTimeoutError)

        action = _build_error_action(error, error_context, retryable)

        assert action.should_retry is True
        assert action.delay_seconds > 0

    def test_non_retryable_error_should_not_retry(self, error_context):
        """Should not retry for non-retryable exception type."""
        error = AgentAPIError("Bad request", status_code=400)
        retryable = (AgentConnectionError, AgentTimeoutError)

        action = _build_error_action(error, error_context, retryable)

        assert action.should_retry is False
        assert action.delay_seconds == 0

    def test_exceeds_max_retries_should_not_retry(self, error_context):
        """Should not retry when attempt exceeds max_retries."""
        error_context.attempt = 5  # Exceeds max_retries=3
        error = AgentConnectionError("Connection failed")
        retryable = (AgentConnectionError,)

        action = _build_error_action(error, error_context, retryable)

        assert action.should_retry is False

    def test_zero_max_retries_never_retries(self, error_context):
        """Should never retry when max_retries is 0."""
        error_context.max_retries = 0
        error = AgentConnectionError("Connection failed")
        retryable = (AgentConnectionError,)

        action = _build_error_action(error, error_context, retryable)

        assert action.should_retry is False

    def test_override_delay_used_when_provided(self, error_context):
        """Should use override_delay when provided."""
        error = AgentRateLimitError("Rate limited", retry_after=30.0)
        retryable = (AgentRateLimitError,)

        action = _build_error_action(error, error_context, retryable, override_delay=30.0)

        assert action.should_retry is True
        assert action.delay_seconds == 30.0


# =============================================================================
# Error Handler Function Tests
# =============================================================================


class TestHandleTimeoutError:
    """Tests for _handle_timeout_error function."""

    def test_wraps_in_agent_timeout_error(self, error_context):
        """Should wrap asyncio.TimeoutError in AgentTimeoutError."""
        original = asyncio.TimeoutError()
        retryable = (AgentTimeoutError,)

        action = _handle_timeout_error(original, error_context, retryable)

        assert isinstance(action.error, AgentTimeoutError)
        assert action.error.agent_name == "test_agent"
        assert "60" in str(action.error)  # timeout from context

    def test_timeout_is_retryable_by_default(self, error_context):
        """Timeout errors should be retryable by default."""
        original = asyncio.TimeoutError()
        retryable = (AgentTimeoutError,)

        action = _handle_timeout_error(original, error_context, retryable)

        assert action.should_retry is True


class TestHandleConnectionError:
    """Tests for _handle_connection_error function."""

    def test_handles_client_connector_error(self, error_context):
        """Should handle aiohttp.ClientConnectorError."""
        # Create a mock connector error
        original = aiohttp.ClientConnectorError(
            MagicMock(host="api.example.com", port=443, ssl=True),
            OSError("Connection refused"),
        )
        retryable = (AgentConnectionError,)

        action = _handle_connection_error(original, error_context, retryable)

        assert isinstance(action.error, AgentConnectionError)
        assert "Connection failed" in str(action.error)

    def test_handles_server_disconnected_error(self, error_context):
        """Should handle aiohttp.ServerDisconnectedError."""
        original = aiohttp.ServerDisconnectedError()
        retryable = (AgentConnectionError,)

        action = _handle_connection_error(original, error_context, retryable)

        assert isinstance(action.error, AgentConnectionError)
        assert "Server disconnected" in str(action.error)


class TestHandlePayloadError:
    """Tests for _handle_payload_error function."""

    def test_wraps_in_stream_error(self, error_context):
        """Should wrap ClientPayloadError in AgentStreamError."""
        original = aiohttp.ClientPayloadError("Payload error")
        retryable = (AgentStreamError,)

        action = _handle_payload_error(original, error_context, retryable)

        assert isinstance(action.error, AgentStreamError)
        assert "Payload error" in str(action.error)


class TestHandleResponseError:
    """Tests for _handle_response_error function."""

    def test_429_creates_rate_limit_error(self, error_context):
        """HTTP 429 should create AgentRateLimitError."""
        original = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=429,
            message="Too Many Requests",
        )
        retryable = (AgentRateLimitError,)

        action = _handle_response_error(original, error_context, retryable)

        assert isinstance(action.error, AgentRateLimitError)
        assert action.should_retry is True

    def test_429_with_retry_after_header(self, error_context):
        """HTTP 429 with Retry-After header should use that value."""
        original = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=429,
            message="Too Many Requests",
            headers={"Retry-After": "30"},
        )
        retryable = (AgentRateLimitError,)

        action = _handle_response_error(original, error_context, retryable)

        assert isinstance(action.error, AgentRateLimitError)
        assert action.error.retry_after == 30.0
        # Delay should be approximately 30 (with some jitter)
        assert 30.0 <= action.delay_seconds <= 33.0

    def test_5xx_creates_connection_error(self, error_context):
        """HTTP 5xx should create AgentConnectionError."""
        original = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=503,
            message="Service Unavailable",
        )
        retryable = (AgentConnectionError,)

        action = _handle_response_error(original, error_context, retryable)

        assert isinstance(action.error, AgentConnectionError)
        assert action.should_retry is True

    def test_4xx_creates_api_error_not_retryable(self, error_context):
        """HTTP 4xx (except 429) should create non-retryable AgentAPIError."""
        original = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=400,
            message="Bad Request",
        )
        retryable = (AgentConnectionError, AgentRateLimitError)

        action = _handle_response_error(original, error_context, retryable)

        assert isinstance(action.error, AgentAPIError)
        assert action.should_retry is False
        assert action.log_level == "error"


class TestHandleAgentError:
    """Tests for _handle_agent_error function."""

    def test_sets_agent_name_if_missing(self, error_context):
        """Should set agent_name if not already set."""
        error = AgentConnectionError("Connection failed")
        assert error.agent_name is None

        action = _handle_agent_error(error, error_context, (AgentConnectionError,))

        assert error.agent_name == "test_agent"

    def test_preserves_existing_agent_name(self, error_context):
        """Should preserve agent_name if already set."""
        error = AgentConnectionError("Connection failed", agent_name="original_agent")

        action = _handle_agent_error(error, error_context, (AgentConnectionError,))

        assert error.agent_name == "original_agent"

    def test_non_recoverable_error_not_retried(self, error_context):
        """Non-recoverable errors should not be retried."""
        error = AgentError("Fatal error", recoverable=False)

        action = _handle_agent_error(error, error_context, (AgentError,))

        assert action.should_retry is False


class TestHandleJsonError:
    """Tests for _handle_json_error function."""

    def test_creates_response_error(self, error_context):
        """Should create AgentResponseError for JSON decode errors."""
        original = ValueError("Invalid JSON: unexpected token")

        action = _handle_json_error(original, error_context)

        assert isinstance(action.error, AgentResponseError)
        assert action.should_retry is False
        assert action.log_level == "error"


class TestHandleUnexpectedError:
    """Tests for _handle_unexpected_error function."""

    def test_wraps_in_agent_error(self, error_context):
        """Should wrap unexpected exceptions in AgentError."""
        original = RuntimeError("Something unexpected happened")

        action = _handle_unexpected_error(original, error_context)

        assert isinstance(action.error, AgentError)
        assert not action.error.recoverable
        assert action.should_retry is False


# =============================================================================
# handle_agent_errors Decorator Tests
# =============================================================================


class TestHandleAgentErrorsDecorator:
    """Tests for handle_agent_errors decorator."""

    def test_success_path_returns_result(self):
        """Successful execution should return result."""

        class MockAgent:
            name = "test_agent"

            @handle_agent_errors()
            async def generate(self, prompt: str) -> str:
                return f"Response to: {prompt}"

        agent = MockAgent()
        result = asyncio.get_event_loop().run_until_complete(agent.generate("Hello"))
        assert result == "Response to: Hello"

    def test_records_success_to_circuit_breaker(self, mock_circuit_breaker):
        """Success should record to circuit breaker."""

        class MockAgent:
            name = "test_agent"
            _circuit_breaker = mock_circuit_breaker

            @handle_agent_errors()
            async def generate(self, prompt: str) -> str:
                return "OK"

        agent = MockAgent()
        asyncio.get_event_loop().run_until_complete(agent.generate("test"))

        mock_circuit_breaker.record_success.assert_called_once()

    def test_circuit_breaker_open_raises_error(self, mock_circuit_breaker):
        """Open circuit breaker should raise AgentCircuitOpenError."""
        mock_circuit_breaker.can_proceed.return_value = False

        class MockAgent:
            name = "test_agent"
            _circuit_breaker = mock_circuit_breaker

            @handle_agent_errors()
            async def generate(self, prompt: str) -> str:
                return "OK"

        agent = MockAgent()
        with pytest.raises(AgentCircuitOpenError):
            asyncio.get_event_loop().run_until_complete(agent.generate("test"))

    def test_timeout_error_wrapped_correctly(self):
        """TimeoutError should be wrapped in AgentTimeoutError."""

        class MockAgent:
            name = "test_agent"
            timeout = 30

            @handle_agent_errors(max_retries=0)
            async def generate(self, prompt: str) -> str:
                raise asyncio.TimeoutError()

        agent = MockAgent()
        with pytest.raises(AgentTimeoutError) as exc_info:
            asyncio.get_event_loop().run_until_complete(agent.generate("test"))

        assert exc_info.value.agent_name == "test_agent"

    @pytest.mark.asyncio
    async def test_retry_on_recoverable_error(self):
        """Should retry on recoverable errors up to max_retries."""
        call_count = 0

        class MockAgent:
            name = "test_agent"

            @handle_agent_errors(max_retries=2, retry_delay=0.01, max_delay=0.1)
            async def generate(self, prompt: str) -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise asyncio.TimeoutError()
                return "Success on 3rd try"

        agent = MockAgent()
        result = await agent.generate("test")

        assert result == "Success on 3rd try"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        """Should raise after exhausting max_retries."""

        class MockAgent:
            name = "test_agent"

            @handle_agent_errors(max_retries=2, retry_delay=0.01, max_delay=0.1)
            async def generate(self, prompt: str) -> str:
                raise asyncio.TimeoutError()

        agent = MockAgent()
        with pytest.raises(AgentTimeoutError):
            await agent.generate("test")

    @pytest.mark.asyncio
    async def test_records_failure_to_circuit_breaker(self, mock_circuit_breaker):
        """Failures should be recorded to circuit breaker."""

        class MockAgent:
            name = "test_agent"
            _circuit_breaker = mock_circuit_breaker

            @handle_agent_errors(max_retries=0)
            async def generate(self, prompt: str) -> str:
                raise asyncio.TimeoutError()

        agent = MockAgent()
        with pytest.raises(AgentTimeoutError):
            await agent.generate("test")

        mock_circuit_breaker.record_failure.assert_called()

    @pytest.mark.asyncio
    async def test_custom_agent_name_attribute(self):
        """Should use custom agent_name_attr."""

        class MockAgent:
            agent_id = "custom_agent"

            @handle_agent_errors(agent_name_attr="agent_id", max_retries=0)
            async def generate(self, prompt: str) -> str:
                raise asyncio.TimeoutError()

        agent = MockAgent()
        with pytest.raises(AgentTimeoutError) as exc_info:
            await agent.generate("test")

        assert exc_info.value.agent_name == "custom_agent"

    @pytest.mark.asyncio
    async def test_json_error_not_retried(self):
        """JSON decode errors should not be retried."""

        class MockAgent:
            name = "test_agent"

            @handle_agent_errors(max_retries=3)
            async def generate(self, prompt: str) -> str:
                raise ValueError("Invalid JSON: decode error")

        agent = MockAgent()
        with pytest.raises(AgentResponseError):
            await agent.generate("test")


# =============================================================================
# with_error_handling Decorator Tests
# =============================================================================


class TestWithErrorHandlingDecorator:
    """Tests for with_error_handling decorator."""

    def test_sync_function_success(self):
        """Sync function should return result on success."""

        @with_error_handling()
        def risky_function() -> str:
            return "success"

        assert risky_function() == "success"

    @pytest.mark.asyncio
    async def test_async_function_success(self):
        """Async function should return result on success."""

        @with_error_handling()
        async def risky_function() -> str:
            return "success"

        result = await risky_function()
        assert result == "success"

    def test_sync_function_returns_fallback_on_error(self):
        """Sync function should return fallback on error."""

        @with_error_handling(fallback="default")
        def risky_function() -> str:
            raise ValueError("Something went wrong")

        assert risky_function() == "default"

    @pytest.mark.asyncio
    async def test_async_function_returns_fallback_on_error(self):
        """Async function should return fallback on error."""

        @with_error_handling(fallback="default")
        async def risky_function() -> str:
            raise ValueError("Something went wrong")

        result = await risky_function()
        assert result == "default"

    def test_catches_specific_error_types(self):
        """Should only catch specified error types."""

        @with_error_handling(error_types=(ValueError,), fallback="caught")
        def risky_function(raise_type: bool) -> str:
            if raise_type:
                raise ValueError("Value error")
            raise TypeError("Type error")

        # ValueError should be caught
        assert risky_function(True) == "caught"

        # TypeError should not be caught
        with pytest.raises(TypeError):
            risky_function(False)

    def test_reraise_option(self):
        """Should re-raise after logging when reraise=True."""

        @with_error_handling(reraise=True)
        def risky_function() -> str:
            raise ValueError("Something went wrong")

        with pytest.raises(ValueError):
            risky_function()

    def test_custom_log_level(self):
        """Should use custom log level."""
        with patch("aragora.agents.errors.decorators.logger") as mock_logger:

            @with_error_handling(log_level="debug")
            def risky_function() -> str:
                raise ValueError("Debug error")

            risky_function()
            mock_logger.debug.assert_called()

    def test_custom_message_template(self):
        """Should use custom message template."""
        with patch("aragora.agents.errors.decorators.logger") as mock_logger:

            @with_error_handling(message_template="Function {func} failed: {error}")
            def my_function() -> str:
                raise ValueError("Oops")

            my_function()

            call_args = str(mock_logger.warning.call_args)
            assert "my_function" in call_args
            assert "Oops" in call_args

    def test_preserves_function_metadata(self):
        """Should preserve function name and docstring."""

        @with_error_handling()
        def documented_function() -> str:
            """This is the docstring."""
            return "result"

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is the docstring."


# =============================================================================
# handle_stream_errors Decorator Tests
# =============================================================================


class TestHandleStreamErrorsDecorator:
    """Tests for handle_stream_errors decorator."""

    @pytest.mark.asyncio
    async def test_successful_streaming(self):
        """Should yield all chunks on success."""

        class MockAgent:
            name = "test_agent"

            @handle_stream_errors()
            async def stream(self, prompt: str):
                for chunk in ["Hello", " ", "World"]:
                    yield chunk

        agent = MockAgent()
        chunks = [chunk async for chunk in agent.stream("test")]
        assert chunks == ["Hello", " ", "World"]

    @pytest.mark.asyncio
    async def test_timeout_preserves_partial_content(self):
        """Timeout should preserve partial content."""

        class MockAgent:
            name = "test_agent"
            timeout = 30

            @handle_stream_errors()
            async def stream(self, prompt: str):
                yield "First"
                yield "Second"
                raise asyncio.TimeoutError()

        agent = MockAgent()
        with pytest.raises(AgentTimeoutError) as exc_info:
            async for _ in agent.stream("test"):
                pass

        assert exc_info.value.partial_content == "FirstSecond"

    @pytest.mark.asyncio
    async def test_payload_error_preserves_partial_content(self):
        """Payload error should preserve partial content."""

        class MockAgent:
            name = "test_agent"

            @handle_stream_errors()
            async def stream(self, prompt: str):
                yield "Chunk1"
                raise aiohttp.ClientPayloadError("Payload interrupted")

        agent = MockAgent()
        with pytest.raises(AgentStreamError) as exc_info:
            async for _ in agent.stream("test"):
                pass

        assert exc_info.value.partial_content == "Chunk1"

    @pytest.mark.asyncio
    async def test_server_disconnected_preserves_partial_content(self):
        """Server disconnected should preserve partial content."""

        class MockAgent:
            name = "test_agent"

            @handle_stream_errors()
            async def stream(self, prompt: str):
                yield "Data"
                raise aiohttp.ServerDisconnectedError()

        agent = MockAgent()
        with pytest.raises(AgentStreamError) as exc_info:
            async for _ in agent.stream("test"):
                pass

        assert exc_info.value.partial_content == "Data"

    @pytest.mark.asyncio
    async def test_unexpected_error_wrapped_in_stream_error(self):
        """Unexpected errors should be wrapped in AgentStreamError."""

        class MockAgent:
            name = "test_agent"

            @handle_stream_errors()
            async def stream(self, prompt: str):
                yield "Some content"
                raise RuntimeError("Unexpected error")

        agent = MockAgent()
        with pytest.raises(AgentStreamError) as exc_info:
            async for _ in agent.stream("test"):
                pass

        assert "Unexpected stream error" in str(exc_info.value)
        assert exc_info.value.partial_content == "Some content"

    @pytest.mark.asyncio
    async def test_agent_error_re_raised_unchanged(self):
        """AgentError subclasses should be re-raised unchanged."""

        class MockAgent:
            name = "test_agent"

            @handle_stream_errors()
            async def stream(self, prompt: str):
                yield "Content"
                raise AgentAPIError("API Error", status_code=400)

        agent = MockAgent()
        with pytest.raises(AgentAPIError):
            async for _ in agent.stream("test"):
                pass

    @pytest.mark.asyncio
    async def test_custom_agent_name_attribute(self):
        """Should use custom agent_name_attr."""

        class MockAgent:
            agent_id = "custom_streamer"
            timeout = 30

            @handle_stream_errors(agent_name_attr="agent_id")
            async def stream(self, prompt: str):
                yield "Starting"
                raise asyncio.TimeoutError()

        agent = MockAgent()
        with pytest.raises(AgentTimeoutError) as exc_info:
            async for _ in agent.stream("test"):
                pass

        assert exc_info.value.agent_name == "custom_streamer"

    @pytest.mark.asyncio
    async def test_non_string_chunks_not_collected(self):
        """Non-string chunks should not be added to partial content."""

        class MockAgent:
            name = "test_agent"

            @handle_stream_errors()
            async def stream(self, prompt: str):
                yield {"type": "start"}
                yield "Text content"
                yield {"type": "end"}
                raise asyncio.TimeoutError()

        agent = MockAgent()
        with pytest.raises(AgentTimeoutError) as exc_info:
            async for _ in agent.stream("test"):
                pass

        # Only string chunks should be in partial_content
        assert exc_info.value.partial_content == "Text content"
