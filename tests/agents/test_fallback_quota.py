"""Focused tests for quota-triggered fallback handoff."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.agents.fallback import QuotaFallbackMixin


class MockPrimaryAgent(QuotaFallbackMixin):
    """Minimal primary agent that falls back on quota-style failures."""

    def __init__(self, status_code: int, error_text: str):
        self.name = "primary-agent"
        self.model = "test-model"
        self.enable_fallback = True
        self.role = "proposer"
        self.timeout = 30
        self.system_prompt = None
        self._fallback_agent = None
        self._status_code = status_code
        self._primary_generate = AsyncMock(side_effect=RuntimeError(error_text))

    async def generate(self, prompt: str, context: list | None = None) -> str:
        try:
            return await self._primary_generate(prompt, context)
        except RuntimeError as exc:
            if self.is_quota_error(self._status_code, str(exc)):
                result = await self.fallback_generate(
                    prompt, context=context, status_code=self._status_code
                )
                if result is not None:
                    return result
            raise


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "error_text"),
    [
        (429, "Rate limit exceeded"),
        (400, "Credit balance is too low"),
    ],
)
async def test_quota_failure_hands_request_to_fallback(status_code: int, error_text: str):
    """Fallback agent receives the original request when primary hits quota errors."""
    agent = MockPrimaryAgent(status_code, error_text)
    fallback_agent = MagicMock()
    fallback_agent.generate = AsyncMock(return_value="fallback response")
    agent._fallback_agent = fallback_agent
    context = [{"role": "user", "content": "Hello"}]

    with (
        patch.object(type(agent), "_get_available_fallback_providers", return_value=[]),
        patch("aragora.agents.fallback.record_fallback_activation"),
        patch("aragora.agents.fallback.record_fallback_success"),
    ):
        result = await agent.generate("Test prompt", context)

    assert result == "fallback response"
    agent._primary_generate.assert_called_once_with("Test prompt", context)
    fallback_agent.generate.assert_called_once_with("Test prompt", context)
