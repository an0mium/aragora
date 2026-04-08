"""Tests quota-triggered fallback from a primary agent."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from aragora.agents.fallback import QuotaFallbackMixin


class _PrimaryAgent(QuotaFallbackMixin):
    def __init__(self):
        self.name = "primary"
        self.model = "gpt-4o"
        self.role = "proposer"
        self.timeout = 30
        self.enable_fallback = True
        self._fallback_agent = None

    async def _primary_generate(self, prompt, context=None):
        return "primary"

    async def generate(self, prompt, context=None):
        try:
            return await self._primary_generate(prompt, context)
        except RuntimeError as exc:
            if self.is_quota_error(429, str(exc)):
                return await self.fallback_generate(prompt, context, status_code=429)
            raise


@pytest.mark.asyncio
async def test_primary_quota_error_routes_request_to_fallback():
    agent = _PrimaryAgent()
    context = [{"role": "user", "content": "hi"}]
    agent._primary_generate = AsyncMock(side_effect=RuntimeError("HTTP 429: rate limit exceeded"))
    fallback = SimpleNamespace(
        generate=AsyncMock(return_value="fallback response"),
        model="openai/gpt-4o",
    )
    agent._fallback_agent = fallback

    with (
        patch("aragora.agents.fallback._get_session_cb", return_value=None),
        patch("aragora.agents.fallback.record_fallback_activation"),
        patch("aragora.agents.fallback.record_fallback_success"),
    ):
        assert await agent.generate("Test prompt", context) == "fallback response"

    fallback.generate.assert_awaited_once_with("Test prompt", context)
