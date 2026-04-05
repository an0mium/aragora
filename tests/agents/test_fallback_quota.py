"""Regression tests for quota-triggered fallback handoff."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_quota_error_routes_request_to_fallback_agent():
    """A quota-style primary failure should hand the request to the fallback agent."""
    from aragora.agents.fallback import AgentFallbackChain

    prompt = "Test prompt"
    context = [{"role": "user", "content": "hello"}]

    primary = MagicMock()
    primary.name = "anthropic"
    primary.generate = AsyncMock(side_effect=RuntimeError("Quota exceeded for this API key"))

    fallback = MagicMock()
    fallback.name = "openrouter"
    fallback.generate = AsyncMock(return_value="fallback response")

    chain = AgentFallbackChain(providers=[primary, fallback])

    with (
        patch("aragora.agents.fallback.record_fallback_activation") as activation,
        patch("aragora.agents.fallback.record_fallback_success"),
    ):
        result = await chain.generate(prompt, context)

    assert result == "fallback response"
    primary.generate.assert_awaited_once_with(prompt, context)
    fallback.generate.assert_awaited_once_with(prompt, context)
    activation.assert_called_once_with(
        primary_agent="anthropic",
        fallback_provider="openrouter",
        error_type="rate_limit",
    )
    assert chain.metrics.primary_attempts == 1
    assert chain.metrics.fallback_successes == 1
