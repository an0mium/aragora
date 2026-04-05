"""Tests for fallback agent quota error handling.

Verifies that when a primary agent raises a quota/rate-limit error,
the fallback chain correctly routes the request to the next provider.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from aragora.agents.fallback import (
    AgentFallbackChain,
    AllProvidersExhaustedError,
    FallbackMetrics,
)


def _make_agent(name: str, response: str | None = None, error: Exception | None = None):
    """Create a mock agent with an async generate method."""
    agent = MagicMock()
    agent.name = name
    if error:
        agent.generate = AsyncMock(side_effect=error)
    else:
        agent.generate = AsyncMock(return_value=response or f"response-from-{name}")
    return agent


class TestFallbackOnQuotaError:
    """Primary agent quota errors trigger fallback to next provider."""

    @pytest.mark.asyncio
    async def test_fallback_receives_request_on_quota_error(self):
        """When the primary agent raises a rate-limit error, the fallback agent gets called."""
        primary = _make_agent("openai", error=RuntimeError("429 rate limit exceeded"))
        fallback = _make_agent("openrouter", response="fallback-ok")

        chain = AgentFallbackChain(providers=[primary, fallback])
        result = await chain.generate("What is 2+2?")

        assert result == "fallback-ok"
        primary.generate.assert_awaited_once_with("What is 2+2?", None)
        fallback.generate.assert_awaited_once_with("What is 2+2?", None)

    @pytest.mark.asyncio
    async def test_fallback_metrics_recorded(self):
        """Metrics reflect one primary failure and one fallback success."""
        primary = _make_agent("openai", error=RuntimeError("quota exceeded"))
        fallback = _make_agent("openrouter", response="ok")

        chain = AgentFallbackChain(providers=[primary, fallback])
        await chain.generate("prompt")

        assert chain.metrics.total_failures >= 1
        assert chain.metrics.fallback_successes >= 1

    @pytest.mark.asyncio
    async def test_all_providers_exhausted(self):
        """AllProvidersExhaustedError raised when every provider fails."""
        p1 = _make_agent("openai", error=RuntimeError("429 rate limit"))
        p2 = _make_agent("openrouter", error=RuntimeError("quota exceeded"))

        chain = AgentFallbackChain(providers=[p1, p2])
        with pytest.raises(AllProvidersExhaustedError):
            await chain.generate("prompt")

    @pytest.mark.asyncio
    async def test_primary_succeeds_no_fallback(self):
        """When the primary succeeds, fallback is never called."""
        primary = _make_agent("openai", response="primary-ok")
        fallback = _make_agent("openrouter", response="fallback-ok")

        chain = AgentFallbackChain(providers=[primary, fallback])
        result = await chain.generate("prompt")

        assert result == "primary-ok"
        fallback.generate.assert_not_awaited()
        assert chain.metrics.total_failures == 0

    @pytest.mark.asyncio
    async def test_context_forwarded_to_fallback(self):
        """Conversation context is forwarded to the fallback agent."""
        ctx = [{"role": "user", "content": "prior message"}]
        primary = _make_agent("openai", error=RuntimeError("rate_limit"))
        fallback = _make_agent("openrouter", response="ok")

        chain = AgentFallbackChain(providers=[primary, fallback])
        await chain.generate("prompt", context=ctx)

        fallback.generate.assert_awaited_once_with("prompt", ctx)

    @pytest.mark.asyncio
    async def test_skips_to_third_provider(self):
        """When both primary and second provider fail, third gets the request."""
        p1 = _make_agent("openai", error=RuntimeError("429"))
        p2 = _make_agent("anthropic", error=RuntimeError("billing error"))
        p3 = _make_agent("openrouter", response="third-ok")

        chain = AgentFallbackChain(providers=[p1, p2, p3])
        result = await chain.generate("prompt")

        assert result == "third-ok"
        p3.generate.assert_awaited_once()
