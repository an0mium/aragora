"""Tests for fallback agent quota handling.

Verifies that when a primary agent raises a quota/rate-limit error,
the fallback mechanism routes the request to a fallback provider.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.agents.fallback import QuotaFallbackMixin


class FakePrimaryAgent(QuotaFallbackMixin):
    """Minimal agent using QuotaFallbackMixin for testing."""

    OPENROUTER_MODEL_MAP = {"test-model": "openrouter/test-model"}
    DEFAULT_FALLBACK_MODEL = "openrouter/default"

    def __init__(self):
        self.name = "fake-primary"
        self.model = "test-model"
        self.enable_fallback = True
        self.role = "proposer"
        self.timeout = 30
        self.system_prompt = None
        self._fallback_agent = None


@pytest.mark.asyncio
async def test_fallback_receives_request_on_quota_error():
    """Primary agent quota error triggers fallback_generate which calls fallback provider."""
    agent = FakePrimaryAgent()

    fallback_agent = AsyncMock()
    fallback_agent.generate = AsyncMock(return_value="fallback response")

    with patch.object(
        agent,
        "_build_fallback_providers",
        return_value=[("openrouter", fallback_agent)],
    ):
        result = await agent.fallback_generate("test prompt", context=None, status_code=429)

    assert result == "fallback response"
    fallback_agent.generate.assert_awaited_once_with("test prompt", None)


@pytest.mark.asyncio
async def test_fallback_skipped_when_disabled():
    """When enable_fallback is False, fallback_generate returns None immediately."""
    agent = FakePrimaryAgent()
    agent.enable_fallback = False

    result = await agent.fallback_generate("test prompt", status_code=429)
    assert result is None


@pytest.mark.asyncio
async def test_fallback_returns_none_when_all_providers_fail():
    """When all fallback providers fail, fallback_generate returns None."""
    agent = FakePrimaryAgent()

    failing_agent = AsyncMock()
    failing_agent.generate = AsyncMock(side_effect=RuntimeError("provider down"))

    with patch.object(
        agent,
        "_build_fallback_providers",
        return_value=[("openrouter", failing_agent)],
    ):
        result = await agent.fallback_generate("test prompt", status_code=429)

    assert result is None
