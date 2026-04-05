"""Tests for fallback quota verification.

Verifies that when a primary agent raises a quota/rate error,
the fallback agent receives the request.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.agents.fallback import QuotaFallbackMixin


class FakePrimaryAgent(QuotaFallbackMixin):
    """Minimal agent stub that uses QuotaFallbackMixin."""

    OPENROUTER_MODEL_MAP = {"test-model": "openrouter/test-model"}
    DEFAULT_FALLBACK_MODEL = "openrouter/test-model"

    def __init__(self):
        self.name = "fake-primary"
        self.model = "test-model"
        self.timeout = 30
        self.enable_fallback = True
        self.role = "proposer"
        self.system_prompt = "You are a test agent."
        self._fallback_agent = None


class TestFallbackQuotaVerification:
    """Verify fallback agent receives request on quota/rate errors."""

    def test_429_detected_as_quota_error(self):
        agent = FakePrimaryAgent()
        assert agent.is_quota_error(429, "rate limit exceeded")

    def test_401_detected_as_quota_error(self):
        agent = FakePrimaryAgent()
        assert agent.is_quota_error(401, "invalid api key")

    def test_200_not_quota_error(self):
        agent = FakePrimaryAgent()
        assert not agent.is_quota_error(200, "ok")

    @pytest.mark.asyncio
    async def test_fallback_called_on_quota_error(self):
        """When primary hits quota, fallback_generate delegates to a fallback provider."""
        agent = FakePrimaryAgent()

        mock_fallback = MagicMock()
        mock_fallback.model = "openrouter/test-model"
        mock_fallback.generate = AsyncMock(return_value="fallback response")

        with patch.object(
            agent, "_build_fallback_providers", return_value=[("openrouter", mock_fallback)]
        ):
            result = await agent.fallback_generate("test prompt", status_code=429)

        assert result == "fallback response"
        mock_fallback.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fallback_not_called_when_disabled(self):
        """When enable_fallback is False, fallback_generate returns None."""
        agent = FakePrimaryAgent()
        agent.enable_fallback = False

        result = await agent.fallback_generate("test prompt", status_code=429)
        assert result is None

    @pytest.mark.asyncio
    async def test_fallback_returns_none_without_providers(self):
        """When no fallback providers available, returns None."""
        agent = FakePrimaryAgent()

        with patch.object(agent, "_build_fallback_providers", return_value=[]):
            result = await agent.fallback_generate("test prompt", status_code=429)

        assert result is None
