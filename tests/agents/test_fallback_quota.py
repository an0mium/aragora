"""Quota fallback test with a mocked primary agent."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from aragora.agents.fallback import QuotaFallbackMixin


class _QuotaPrimaryAgent(QuotaFallbackMixin):
    def __init__(self):
        self.name = "anthropic-primary"
        self.model = "claude-opus-4-6"
        self.role = "proposer"
        self.timeout = 30
        self.enable_fallback = True
        self.primary = AsyncMock(side_effect=RuntimeError("rate limit exceeded"))
        self._fallback_agent = AsyncMock()
        self._fallback_agent.generate = AsyncMock(return_value="fallback response")

    async def generate(self, prompt: str, context=None):
        try:
            return await self.primary(prompt, context)
        except RuntimeError as exc:
            if self.is_quota_error(429, str(exc)):
                return await self.fallback_generate(prompt, context, status_code=429)
            raise


def test_quota_error_routes_request_to_fallback():
    agent = _QuotaPrimaryAgent()
    context = [{"role": "user", "content": "hello"}]

    with (
        patch("aragora.agents.fallback._get_session_cb", return_value=None),
        patch("aragora.agents.fallback.record_fallback_activation"),
        patch("aragora.agents.fallback.record_fallback_success"),
    ):
        result = asyncio.run(agent.generate("test prompt", context))

    assert result == "fallback response"
    agent.primary.assert_awaited_once_with("test prompt", context)
    agent._fallback_agent.generate.assert_awaited_once_with("test prompt", context)
