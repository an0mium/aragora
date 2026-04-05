"""Regression tests for quota-triggered agent fallback."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_message",
    ["429 rate limit exceeded", "quota exceeded for this API key"],
)
async def test_quota_error_routes_request_to_fallback_agent(error_message):
    """A quota/rate-limit failure on the primary agent should use the fallback."""
    from aragora.agents.fallback import AgentFallbackChain

    context = [{"role": "user", "content": "Hello"}]

    primary = MagicMock()
    primary.name = "openai"
    primary.generate = AsyncMock(side_effect=RuntimeError(error_message))

    fallback = MagicMock()
    fallback.name = "openrouter"
    fallback.generate = AsyncMock(return_value="response from fallback")

    chain = AgentFallbackChain(providers=[primary, fallback])

    result = await chain.generate("Test prompt", context=context)

    assert result == "response from fallback"
    primary.generate.assert_called_once_with("Test prompt", context)
    fallback.generate.assert_called_once_with("Test prompt", context)
    assert chain.metrics.primary_attempts == 1
    assert chain.metrics.fallback_successes == 1
