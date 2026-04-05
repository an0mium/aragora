"""Tests for quota-triggered fallback delegation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aragora.agents.fallback import QuotaFallbackMixin


class _QuotaPrimaryAgent(QuotaFallbackMixin):
    def __init__(self, status_code: int, error_text: str):
        self.name = "primary"
        self.model = "gpt-4o"
        self.role = "proposer"
        self.timeout = 30
        self.enable_fallback = True
        self._status_code = status_code
        self._error_text = error_text
        self._fallback_agent = AsyncMock()

    async def generate(self, prompt: str, context: list | None = None) -> str | None:
        if self.is_quota_error(self._status_code, self._error_text):
            return await self.fallback_generate(prompt, context, status_code=self._status_code)
        raise RuntimeError(self._error_text)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "error_text"),
    [
        (429, "rate limit exceeded"),
        (403, "quota exceeded for this project"),
    ],
)
async def test_quota_error_calls_fallback_with_original_request(status_code: int, error_text: str):
    prompt = "Explain the fallback path"
    context = [{"role": "user", "content": "hello"}]
    primary = _QuotaPrimaryAgent(status_code, error_text)
    primary._fallback_agent.generate.return_value = "fallback response"

    with (
        patch("aragora.agents.fallback._get_session_cb", return_value=None),
        patch("aragora.agents.fallback.record_fallback_activation"),
        patch("aragora.agents.fallback.record_fallback_success"),
    ):
        result = await primary.generate(prompt, context)

    assert result == "fallback response"
    primary._fallback_agent.generate.assert_awaited_once_with(prompt, context)
