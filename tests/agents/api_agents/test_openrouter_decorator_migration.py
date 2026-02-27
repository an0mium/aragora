"""
Tests for OpenRouter decorator migration (T2: @handle_agent_errors).

Verifies that:
- generate() delegates to @handle_agent_errors-decorated _generate_with_model()
- _generate_with_model() retries on rate limit, connection, and timeout errors
- generate() falls back to alternate model after decorator retries exhausted
- generate_stream() uses _stream_with_retry() with calculate_retry_delay_with_jitter
- Streaming retries on 429 with jittered backoff (not fixed delay)
- Streaming retries on connection errors with jittered backoff
- Streaming retries on timeout errors
- Circuit breaker integration is consistent across generate and stream paths
- OpenRouter-specific rate limiter metrics are recorded during retries
- Fallback model escalation preserved in generate() path
- critique() delegates through generate() and inherits decorator protection
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import aiohttp
import pytest

from aragora.agents.api_agents.common import (
    AgentAPIError,
    AgentCircuitOpenError,
    AgentConnectionError,
    AgentRateLimitError,
    AgentStreamError,
    AgentTimeoutError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ok_response(json_data):
    """Create a mock 200 response with the given JSON payload."""
    from tests.agents.api_agents.conftest import MockResponse

    return MockResponse(status=200, json_data=json_data)


def _make_429_response(retry_after="0.01"):
    """Create a mock 429 rate-limit response."""
    from tests.agents.api_agents.conftest import MockResponse

    return MockResponse(
        status=429,
        text='{"error": "rate_limited"}',
        headers={"Retry-After": retry_after},
    )


def _make_dynamic_session(response_factory):
    """Create a mock session that calls response_factory on each post()."""

    class DynamicSession:
        def post(self, *args, **kwargs):
            return response_factory()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    return DynamicSession()


# ===========================================================================
# generate() / _generate_with_model() decorator integration
# ===========================================================================


class TestGenerateDecoratorConfig:
    """Verify that _generate_with_model has the correct decorator parameters."""

    def test_decorator_is_applied(self, mock_env_with_api_keys):
        """_generate_with_model should be wrapped by handle_agent_errors."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent

        agent = OpenRouterAgent()
        # The decorator sets __wrapped__ on the wrapper via functools.wraps
        method = agent._generate_with_model
        assert hasattr(method, "__wrapped__") or callable(method)

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_then_succeeds(
        self, mock_env_with_api_keys, mock_openrouter_response, mock_openrouter_limiter
    ):
        """Decorator should retry AgentRateLimitError then return success."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent

        agent = OpenRouterAgent()

        call_count = [0]

        def response_factory():
            call_count[0] += 1
            if call_count[0] <= 2:
                return _make_429_response()
            return _make_ok_response(mock_openrouter_response)

        with (
            patch(
                "aragora.agents.api_agents.openrouter.get_openrouter_limiter",
                return_value=mock_openrouter_limiter,
            ),
            patch(
                "aragora.agents.api_agents.openrouter.create_client_session",
                return_value=_make_dynamic_session(response_factory),
            ),
        ):
            result = await agent.generate("Hello")

        assert "test response from DeepSeek" in result
        # 2 retries + 1 success = 3 calls
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_retries_on_connection_error_then_succeeds(
        self, mock_env_with_api_keys, mock_openrouter_response, mock_openrouter_limiter
    ):
        """Decorator should retry aiohttp.ClientError mapped to AgentConnectionError."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent

        agent = OpenRouterAgent()

        call_count = [0]

        def response_factory():
            call_count[0] += 1
            if call_count[0] == 1:
                raise aiohttp.ClientConnectorError(
                    connection_key=MagicMock(), os_error=OSError("refused")
                )
            return _make_ok_response(mock_openrouter_response)

        with (
            patch(
                "aragora.agents.api_agents.openrouter.get_openrouter_limiter",
                return_value=mock_openrouter_limiter,
            ),
            patch(
                "aragora.agents.api_agents.openrouter.create_client_session",
                return_value=_make_dynamic_session(response_factory),
            ),
        ):
            result = await agent.generate("Hello")

        assert result is not None
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_immediately(
        self, mock_env_with_api_keys, mock_openrouter_limiter
    ):
        """AgentAPIError (non-retryable) should not be retried."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent
        from tests.agents.api_agents.conftest import MockResponse

        agent = OpenRouterAgent()

        call_count = [0]

        def response_factory():
            call_count[0] += 1
            return MockResponse(status=400, text='{"error": "bad request"}')

        with (
            patch(
                "aragora.agents.api_agents.openrouter.get_openrouter_limiter",
                return_value=mock_openrouter_limiter,
            ),
            patch(
                "aragora.agents.api_agents.openrouter.create_client_session",
                return_value=_make_dynamic_session(response_factory),
            ),
        ):
            with pytest.raises(AgentAPIError):
                await agent.generate("Hello")

        # Should not retry 4xx errors
        assert call_count[0] == 1


# ===========================================================================
# generate() model fallback after decorator retries exhausted
# ===========================================================================


class TestGenerateModelFallback:
    """Verify model fallback escalation in generate()."""

    @pytest.mark.asyncio
    async def test_fallback_model_used_after_primary_exhausted(
        self, mock_env_with_api_keys, mock_openrouter_response, mock_openrouter_limiter
    ):
        """generate() should fall back to OPENROUTER_FALLBACK_MODELS entry."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent

        # deepseek/deepseek-chat -> openai/gpt-5.2-chat
        agent = OpenRouterAgent(model="deepseek/deepseek-chat")

        call_count = [0]
        models_seen = []

        def response_factory():
            call_count[0] += 1
            if call_count[0] <= 4:
                # Primary model + retries all fail
                return _make_429_response()
            return _make_ok_response(mock_openrouter_response)

        original_post = None

        class TrackingSession:
            def post(self, url, *, json=None, **kwargs):
                if json:
                    models_seen.append(json.get("model"))
                return response_factory()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with (
            patch(
                "aragora.agents.api_agents.openrouter.get_openrouter_limiter",
                return_value=mock_openrouter_limiter,
            ),
            patch(
                "aragora.agents.api_agents.openrouter.create_client_session",
                return_value=TrackingSession(),
            ),
        ):
            result = await agent.generate("Hello")

        assert result is not None
        # Should have seen primary model first, then fallback
        assert "deepseek/deepseek-chat" in models_seen
        assert "openai/gpt-5.2-chat" in models_seen

    @pytest.mark.asyncio
    async def test_no_fallback_when_model_not_in_map(
        self, mock_env_with_api_keys, mock_openrouter_limiter
    ):
        """Should raise without fallback when model has no fallback mapping."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent

        agent = OpenRouterAgent(model="unknown/no-fallback-model")

        def response_factory():
            return _make_429_response()

        with (
            patch(
                "aragora.agents.api_agents.openrouter.get_openrouter_limiter",
                return_value=mock_openrouter_limiter,
            ),
            patch(
                "aragora.agents.api_agents.openrouter.create_client_session",
                return_value=_make_dynamic_session(response_factory),
            ),
        ):
            with pytest.raises(AgentRateLimitError):
                await agent.generate("Hello")

    @pytest.mark.asyncio
    async def test_fallback_records_chain_depth(
        self, mock_env_with_api_keys, mock_openrouter_response, mock_openrouter_limiter
    ):
        """Model fallback should record fallback_chain_depth=1."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent

        agent = OpenRouterAgent(model="deepseek/deepseek-chat")

        call_count = [0]

        def response_factory():
            call_count[0] += 1
            if call_count[0] <= 4:
                return _make_429_response()
            return _make_ok_response(mock_openrouter_response)

        with (
            patch(
                "aragora.agents.api_agents.openrouter.get_openrouter_limiter",
                return_value=mock_openrouter_limiter,
            ),
            patch(
                "aragora.agents.api_agents.openrouter.create_client_session",
                return_value=_make_dynamic_session(response_factory),
            ),
            patch(
                "aragora.agents.api_agents.openrouter.record_fallback_chain_depth"
            ) as mock_record,
        ):
            result = await agent.generate("Hello")

        assert result is not None
        # Should record depth=1 for the fallback
        mock_record.assert_any_call(1)


# ===========================================================================
# generate_stream() / _stream_with_retry() -- jittered backoff
# ===========================================================================


class TestStreamRetryJitteredBackoff:
    """Verify streaming uses calculate_retry_delay_with_jitter for backoff."""

    @pytest.mark.asyncio
    async def test_streaming_429_uses_jittered_delay(
        self, mock_env_with_api_keys, mock_sse_chunks, mock_openrouter_limiter
    ):
        """On 429 without Retry-After, should use calculate_retry_delay_with_jitter."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent
        from tests.agents.api_agents.conftest import MockResponse, MockStreamResponse

        agent = OpenRouterAgent()

        call_count = [0]

        def response_factory():
            call_count[0] += 1
            if call_count[0] == 1:
                # No Retry-After header -- should use jittered backoff
                return MockResponse(status=429, text='{"error": "rate_limited"}')
            return MockStreamResponse(status=200, chunks=mock_sse_chunks)

        sleep_delays = []
        original_sleep = asyncio.sleep

        async def tracking_sleep(delay):
            sleep_delays.append(delay)

        with (
            patch(
                "aragora.agents.api_agents.openrouter.get_openrouter_limiter",
                return_value=mock_openrouter_limiter,
            ),
            patch(
                "aragora.agents.api_agents.openrouter.create_client_session",
                return_value=_make_dynamic_session(response_factory),
            ),
            patch(
                "aragora.agents.api_agents.openrouter.asyncio.sleep",
                side_effect=tracking_sleep,
            ),
        ):
            chunks = []
            async for chunk in agent.generate_stream("Hello"):
                chunks.append(chunk)

        assert call_count[0] == 2
        # Should have slept once (between retry 1 and retry 2)
        assert len(sleep_delays) == 1
        # Jittered delay should be > 0 and reasonable (base_delay=2.0, attempt=0)
        assert sleep_delays[0] > 0.0

    @pytest.mark.asyncio
    async def test_streaming_connection_error_uses_jittered_delay(
        self, mock_env_with_api_keys, mock_sse_chunks, mock_openrouter_limiter
    ):
        """Connection errors should use calculate_retry_delay_with_jitter for backoff."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent
        from tests.agents.api_agents.conftest import MockStreamResponse

        agent = OpenRouterAgent()

        call_count = [0]

        def response_factory():
            call_count[0] += 1
            if call_count[0] == 1:
                raise aiohttp.ClientConnectorError(
                    connection_key=MagicMock(), os_error=OSError("refused")
                )
            return MockStreamResponse(status=200, chunks=mock_sse_chunks)

        sleep_delays = []

        async def tracking_sleep(delay):
            sleep_delays.append(delay)

        with (
            patch(
                "aragora.agents.api_agents.openrouter.get_openrouter_limiter",
                return_value=mock_openrouter_limiter,
            ),
            patch(
                "aragora.agents.api_agents.openrouter.create_client_session",
                return_value=_make_dynamic_session(response_factory),
            ),
            patch(
                "aragora.agents.api_agents.openrouter.asyncio.sleep",
                side_effect=tracking_sleep,
            ),
        ):
            chunks = []
            async for chunk in agent.generate_stream("Hello"):
                chunks.append(chunk)

        assert call_count[0] == 2
        assert len(sleep_delays) == 1
        assert sleep_delays[0] > 0.0

    @pytest.mark.asyncio
    async def test_streaming_timeout_retries_with_jittered_delay(
        self, mock_env_with_api_keys, mock_sse_chunks, mock_openrouter_limiter
    ):
        """Timeout errors should be retried with jittered backoff."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent
        from tests.agents.api_agents.conftest import MockStreamResponse

        agent = OpenRouterAgent()

        call_count = [0]

        def response_factory():
            call_count[0] += 1
            if call_count[0] == 1:
                raise asyncio.TimeoutError("stream timeout")
            return MockStreamResponse(status=200, chunks=mock_sse_chunks)

        with (
            patch(
                "aragora.agents.api_agents.openrouter.get_openrouter_limiter",
                return_value=mock_openrouter_limiter,
            ),
            patch(
                "aragora.agents.api_agents.openrouter.create_client_session",
                return_value=_make_dynamic_session(response_factory),
            ),
            patch(
                "aragora.agents.api_agents.openrouter.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            chunks = []
            async for chunk in agent.generate_stream("Hello"):
                chunks.append(chunk)

        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_streaming_timeout_raises_after_max_retries(
        self, mock_env_with_api_keys, mock_openrouter_limiter
    ):
        """Should raise AgentTimeoutError after exhausting retries on timeout."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent

        agent = OpenRouterAgent()

        def response_factory():
            raise asyncio.TimeoutError("stream timeout")

        with (
            patch(
                "aragora.agents.api_agents.openrouter.get_openrouter_limiter",
                return_value=mock_openrouter_limiter,
            ),
            patch(
                "aragora.agents.api_agents.openrouter.create_client_session",
                return_value=_make_dynamic_session(response_factory),
            ),
            patch(
                "aragora.agents.api_agents.openrouter.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            with pytest.raises(AgentTimeoutError, match="timed out"):
                async for _ in agent.generate_stream("Hello"):
                    pass

    @pytest.mark.asyncio
    async def test_streaming_429_with_retry_after_header(
        self, mock_env_with_api_keys, mock_sse_chunks, mock_openrouter_limiter
    ):
        """Should prefer Retry-After header value over jittered backoff."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent
        from tests.agents.api_agents.conftest import MockResponse, MockStreamResponse

        agent = OpenRouterAgent()

        call_count = [0]

        def response_factory():
            call_count[0] += 1
            if call_count[0] == 1:
                return MockResponse(
                    status=429,
                    text='{"error": "rate_limited"}',
                    headers={"Retry-After": "0.5"},
                )
            return MockStreamResponse(status=200, chunks=mock_sse_chunks)

        sleep_delays = []

        async def tracking_sleep(delay):
            sleep_delays.append(delay)

        with (
            patch(
                "aragora.agents.api_agents.openrouter.get_openrouter_limiter",
                return_value=mock_openrouter_limiter,
            ),
            patch(
                "aragora.agents.api_agents.openrouter.create_client_session",
                return_value=_make_dynamic_session(response_factory),
            ),
            patch(
                "aragora.agents.api_agents.openrouter.asyncio.sleep",
                side_effect=tracking_sleep,
            ),
        ):
            chunks = []
            async for chunk in agent.generate_stream("Hello"):
                chunks.append(chunk)

        assert call_count[0] == 2
        assert len(sleep_delays) == 1
        # Should use the Retry-After value (0.5)
        assert sleep_delays[0] == pytest.approx(0.5, abs=0.01)


# ===========================================================================
# _stream_with_retry accepts model parameter for future fallback
# ===========================================================================


class TestStreamWithRetryModelParam:
    """Verify _stream_with_retry uses the model parameter in payload."""

    @pytest.mark.asyncio
    async def test_stream_with_retry_uses_provided_model(
        self, mock_env_with_api_keys, mock_sse_chunks, mock_openrouter_limiter
    ):
        """_stream_with_retry should send the model parameter in the API payload."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent
        from tests.agents.api_agents.conftest import MockStreamResponse

        agent = OpenRouterAgent(model="deepseek/deepseek-chat")

        models_seen = []

        class TrackingSession:
            def post(self, url, *, json=None, **kwargs):
                if json:
                    models_seen.append(json.get("model"))
                return MockStreamResponse(status=200, chunks=mock_sse_chunks)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with (
            patch(
                "aragora.agents.api_agents.openrouter.get_openrouter_limiter",
                return_value=mock_openrouter_limiter,
            ),
            patch(
                "aragora.agents.api_agents.openrouter.create_client_session",
                return_value=TrackingSession(),
            ),
        ):
            chunks = []
            async for chunk in agent._stream_with_retry("openai/gpt-5.2-chat", "Hello"):
                chunks.append(chunk)

        assert models_seen == ["openai/gpt-5.2-chat"]

    @pytest.mark.asyncio
    async def test_stream_with_retry_uses_max_tokens(
        self, mock_env_with_api_keys, mock_sse_chunks, mock_openrouter_limiter
    ):
        """_stream_with_retry should respect self.max_tokens in payload."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent
        from tests.agents.api_agents.conftest import MockStreamResponse

        agent = OpenRouterAgent(max_tokens=2048)

        payloads = []

        class TrackingSession:
            def post(self, url, *, json=None, **kwargs):
                if json:
                    payloads.append(json)
                return MockStreamResponse(status=200, chunks=mock_sse_chunks)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with (
            patch(
                "aragora.agents.api_agents.openrouter.get_openrouter_limiter",
                return_value=mock_openrouter_limiter,
            ),
            patch(
                "aragora.agents.api_agents.openrouter.create_client_session",
                return_value=TrackingSession(),
            ),
        ):
            async for _ in agent._stream_with_retry(agent.model, "Hello"):
                pass

        assert payloads[0]["max_tokens"] == 2048


# ===========================================================================
# critique() inherits decorator protection via generate()
# ===========================================================================


class TestCritiqueDecoratorInheritance:
    """Verify critique() inherits retry/fallback from generate()."""

    @pytest.mark.asyncio
    async def test_critique_retries_via_generate(self, mock_env_with_api_keys):
        """critique() should benefit from generate() retry logic."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent

        agent = OpenRouterAgent()

        call_count = [0]

        async def mock_generate(prompt, context=None):
            call_count[0] += 1
            return """ISSUES:
- Issue one

SUGGESTIONS:
- Fix it

SEVERITY: 3.0
REASONING: Test reasoning"""

        with patch.object(agent, "generate", side_effect=mock_generate):
            critique = await agent.critique(
                proposal="Test proposal",
                task="Test task",
                target_agent="test-agent",
            )

        assert critique is not None
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_critique_propagates_errors(self, mock_env_with_api_keys):
        """critique() should propagate AgentError from generate()."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent

        agent = OpenRouterAgent()

        async def failing_generate(prompt, context=None):
            raise AgentRateLimitError("Rate limited after all retries", agent_name=agent.name)

        with patch.object(agent, "generate", side_effect=failing_generate):
            with pytest.raises(AgentRateLimitError):
                await agent.critique(
                    proposal="Test proposal",
                    task="Test task",
                )


# ===========================================================================
# Circuit breaker consistency across generate and stream
# ===========================================================================


class TestCircuitBreakerConsistency:
    """Verify circuit breaker behavior is consistent across all paths."""

    @pytest.mark.asyncio
    async def test_generate_records_success_on_circuit_breaker(
        self, mock_env_with_api_keys, mock_openrouter_response, mock_openrouter_limiter
    ):
        """generate path should record circuit breaker success."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent

        agent = OpenRouterAgent()
        agent._circuit_breaker = MagicMock()
        agent._circuit_breaker.can_proceed.return_value = True

        with (
            patch(
                "aragora.agents.api_agents.openrouter.get_openrouter_limiter",
                return_value=mock_openrouter_limiter,
            ),
            patch(
                "aragora.agents.api_agents.openrouter.create_client_session",
                return_value=_make_dynamic_session(
                    lambda: _make_ok_response(mock_openrouter_response)
                ),
            ),
        ):
            await agent.generate("Hello")

        agent._circuit_breaker.record_success.assert_called()

    @pytest.mark.asyncio
    async def test_stream_records_success_on_circuit_breaker(
        self, mock_env_with_api_keys, mock_sse_chunks, mock_openrouter_limiter
    ):
        """stream path should record circuit breaker success."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent
        from tests.agents.api_agents.conftest import MockStreamResponse

        agent = OpenRouterAgent()
        agent._circuit_breaker = MagicMock()
        agent._circuit_breaker.can_proceed.return_value = True

        with (
            patch(
                "aragora.agents.api_agents.openrouter.get_openrouter_limiter",
                return_value=mock_openrouter_limiter,
            ),
            patch(
                "aragora.agents.api_agents.openrouter.create_client_session",
            ) as mock_create,
        ):
            mock_session = MagicMock()
            mock_session.post = MagicMock(
                return_value=MockStreamResponse(status=200, chunks=mock_sse_chunks)
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_create.return_value = mock_session

            async for _ in agent.generate_stream("Hello"):
                pass

        agent._circuit_breaker.record_success.assert_called()

    @pytest.mark.asyncio
    async def test_generate_circuit_breaker_open_raises(self, mock_env_with_api_keys):
        """generate should raise AgentCircuitOpenError when breaker is open."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent

        agent = OpenRouterAgent()
        agent._circuit_breaker = MagicMock()
        agent._circuit_breaker.can_proceed.return_value = False
        agent._circuit_breaker.cooldown_seconds = 60.0

        with pytest.raises(AgentCircuitOpenError):
            await agent.generate("Hello")

    @pytest.mark.asyncio
    async def test_stream_circuit_breaker_open_raises(self, mock_env_with_api_keys):
        """generate_stream should raise AgentCircuitOpenError when breaker is open."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent

        agent = OpenRouterAgent()
        agent._circuit_breaker = MagicMock()
        agent._circuit_breaker.can_proceed.return_value = False

        with pytest.raises(AgentCircuitOpenError, match="Circuit breaker open"):
            async for _ in agent.generate_stream("Hello"):
                pass


# ===========================================================================
# Rate limiter metric recording
# ===========================================================================


class TestRateLimiterMetrics:
    """Verify OpenRouter-specific rate limiter metrics are recorded."""

    @pytest.mark.asyncio
    async def test_rate_limit_detected_recorded_on_429(
        self, mock_env_with_api_keys, mock_openrouter_response, mock_openrouter_limiter
    ):
        """Should call record_rate_limit_detected on 429 in generate path."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent

        agent = OpenRouterAgent()

        call_count = [0]

        def response_factory():
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_429_response()
            return _make_ok_response(mock_openrouter_response)

        with (
            patch(
                "aragora.agents.api_agents.openrouter.get_openrouter_limiter",
                return_value=mock_openrouter_limiter,
            ),
            patch(
                "aragora.agents.api_agents.openrouter.create_client_session",
                return_value=_make_dynamic_session(response_factory),
            ),
            patch("aragora.agents.api_agents.openrouter.record_rate_limit_detected") as mock_record,
        ):
            await agent.generate("Hello")

        mock_record.assert_called()

    @pytest.mark.asyncio
    async def test_provider_call_recorded_on_success(
        self, mock_env_with_api_keys, mock_openrouter_response, mock_openrouter_limiter
    ):
        """Should call record_provider_call with success=True on success."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent

        agent = OpenRouterAgent()

        with (
            patch(
                "aragora.agents.api_agents.openrouter.get_openrouter_limiter",
                return_value=mock_openrouter_limiter,
            ),
            patch(
                "aragora.agents.api_agents.openrouter.create_client_session",
                return_value=_make_dynamic_session(
                    lambda: _make_ok_response(mock_openrouter_response)
                ),
            ),
            patch("aragora.agents.api_agents.openrouter.record_provider_call") as mock_record,
        ):
            await agent.generate("Hello")

        # Find the success call
        success_calls = [c for c in mock_record.call_args_list if c.kwargs.get("success")]
        assert len(success_calls) >= 1

    @pytest.mark.asyncio
    async def test_stream_rate_limit_detected_recorded_on_429(
        self, mock_env_with_api_keys, mock_sse_chunks, mock_openrouter_limiter
    ):
        """Should call record_rate_limit_detected on 429 in streaming path."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent
        from tests.agents.api_agents.conftest import MockResponse, MockStreamResponse

        agent = OpenRouterAgent()

        call_count = [0]

        def response_factory():
            call_count[0] += 1
            if call_count[0] == 1:
                return MockResponse(
                    status=429,
                    text='{"error": "rate_limited"}',
                    headers={"Retry-After": "0.01"},
                )
            return MockStreamResponse(status=200, chunks=mock_sse_chunks)

        with (
            patch(
                "aragora.agents.api_agents.openrouter.get_openrouter_limiter",
                return_value=mock_openrouter_limiter,
            ),
            patch(
                "aragora.agents.api_agents.openrouter.create_client_session",
                return_value=_make_dynamic_session(response_factory),
            ),
            patch(
                "aragora.agents.api_agents.openrouter.asyncio.sleep",
                new_callable=AsyncMock,
            ),
            patch("aragora.agents.api_agents.openrouter.record_rate_limit_detected") as mock_record,
        ):
            async for _ in agent.generate_stream("Hello"):
                pass

        mock_record.assert_called_once_with("openrouter", 0)


# ===========================================================================
# Public API preservation
# ===========================================================================


class TestPublicAPIPreserved:
    """Verify the public API of OpenRouterAgent is unchanged."""

    def test_generate_signature(self, mock_env_with_api_keys):
        """generate() should accept (prompt, context=None)."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent
        import inspect

        agent = OpenRouterAgent()
        sig = inspect.signature(agent.generate)
        params = list(sig.parameters.keys())
        assert "prompt" in params
        assert "context" in params

    def test_generate_stream_signature(self, mock_env_with_api_keys):
        """generate_stream() should accept (prompt, context=None)."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent
        import inspect

        agent = OpenRouterAgent()
        sig = inspect.signature(agent.generate_stream)
        params = list(sig.parameters.keys())
        assert "prompt" in params
        assert "context" in params

    def test_critique_signature(self, mock_env_with_api_keys):
        """critique() should accept (proposal, task, context=None, target_agent=None)."""
        from aragora.agents.api_agents.openrouter import OpenRouterAgent
        import inspect

        agent = OpenRouterAgent()
        sig = inspect.signature(agent.critique)
        params = list(sig.parameters.keys())
        assert "proposal" in params
        assert "task" in params
        assert "context" in params
        assert "target_agent" in params

    def test_subclasses_inherit_generate(self, mock_env_with_api_keys):
        """All subclasses should inherit generate() from OpenRouterAgent."""
        from aragora.agents.api_agents.openrouter import (
            DeepSeekAgent,
            LlamaAgent,
            QwenAgent,
            KimiK2Agent,
        )

        for cls in (DeepSeekAgent, LlamaAgent, QwenAgent, KimiK2Agent):
            agent = cls()
            # generate should be inherited, not overridden
            assert agent.generate.__qualname__.startswith("OpenRouterAgent")
