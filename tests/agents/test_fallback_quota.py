"""Tests that quota/rate-limit errors on a primary agent trigger fallback."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from aragora.agents.fallback import (
    AgentFallbackChain,
    AllProvidersExhaustedError,
)


def _make_agent(name: str, side_effect=None, return_value=None):
    """Create a mock agent with a generate method."""
    agent = AsyncMock()
    agent.name = name
    if side_effect is not None:
        agent.generate.side_effect = side_effect
    elif return_value is not None:
        agent.generate.return_value = return_value
    return agent


class TestFallbackOnQuotaError:
    """Verify that AgentFallbackChain routes to fallback on quota errors."""

    @pytest.mark.asyncio
    async def test_fallback_on_rate_limit(self):
        """Primary raises 429-style error; fallback returns successfully."""
        primary = _make_agent("primary", side_effect=Exception("429 Too Many Requests"))
        fallback = _make_agent("fallback", return_value="fallback response")

        chain = AgentFallbackChain(providers=[primary, fallback])
        result = await chain.generate("test prompt")

        assert result == "fallback response"
        primary.generate.assert_called_once()
        fallback.generate.assert_called_once()
        assert chain.metrics.primary_attempts == 1
        assert chain.metrics.primary_successes == 0
        assert chain.metrics.fallback_attempts == 1
        assert chain.metrics.fallback_successes == 1

    @pytest.mark.asyncio
    async def test_fallback_on_quota_exceeded(self):
        """Primary raises quota-exceeded error; fallback handles it."""
        primary = _make_agent("primary", side_effect=Exception("quota exceeded"))
        fallback = _make_agent("fallback", return_value="ok")

        chain = AgentFallbackChain(providers=[primary, fallback])
        result = await chain.generate("prompt")

        assert result == "ok"
        assert chain.metrics.fallback_rate > 0

    @pytest.mark.asyncio
    async def test_all_providers_exhausted(self):
        """Both primary and fallback fail; AllProvidersExhaustedError raised."""
        primary = _make_agent("primary", side_effect=Exception("rate limit"))
        fallback = _make_agent("fallback", side_effect=Exception("also rate limited"))

        chain = AgentFallbackChain(providers=[primary, fallback])
        with pytest.raises(AllProvidersExhaustedError):
            await chain.generate("prompt")

        assert chain.metrics.total_failures == 2
