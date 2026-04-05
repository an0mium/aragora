"""Tests for fallback agent quota error handling.

Verifies that when a primary agent raises quota/rate-limit errors,
the fallback agent receives the request and produces a response.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.agents.fallback import QuotaFallbackMixin


class FakePrimaryAgent(QuotaFallbackMixin):
    """Simulated primary agent that can trigger quota errors."""

    OPENROUTER_MODEL_MAP = {"test-model": "openrouter/test-model"}
    DEFAULT_FALLBACK_MODEL = "openrouter/fallback-model"

    def __init__(self, *, enable_fallback: bool = True):
        self.name = "fake-primary"
        self.model = "test-model"
        self.enable_fallback = enable_fallback
        self.role = "proposer"
        self.timeout = 60
        self.system_prompt = None
        self._fallback_agent = None


class TestIsQuotaError:
    """Verify quota/rate-limit detection across status codes."""

    @pytest.fixture()
    def agent(self):
        return FakePrimaryAgent()

    @pytest.mark.parametrize("status", [401, 429, 408, 504, 524])
    def test_quota_status_codes(self, agent, status):
        assert agent.is_quota_error(status, "") is True

    def test_403_with_quota_keyword(self, agent):
        assert agent.is_quota_error(403, "Resource exhausted") is True

    def test_403_without_keyword(self, agent):
        assert agent.is_quota_error(403, "Forbidden") is False

    def test_200_not_quota(self, agent):
        assert agent.is_quota_error(200, "") is False


class TestFallbackGenerate:
    """Verify fallback_generate delegates to fallback agent on quota errors."""

    @pytest.fixture()
    def agent(self):
        return FakePrimaryAgent()

    @pytest.mark.asyncio
    async def test_fallback_called_on_quota_error(self, agent):
        """Primary quota error -> fallback agent receives the prompt."""
        mock_fallback = AsyncMock()
        mock_fallback.generate = AsyncMock(return_value="fallback response")
        mock_fallback.name = "openrouter-fallback"

        agent._fallback_agent = mock_fallback
        # Patch _build_fallback_providers to return our mock
        agent._build_fallback_providers = lambda: [("openrouter", mock_fallback)]

        result = await agent.fallback_generate("test prompt", status_code=429)

        assert result == "fallback response"
        mock_fallback.generate.assert_called_once_with("test prompt", None)

    @pytest.mark.asyncio
    async def test_fallback_disabled_returns_none(self):
        """When enable_fallback is False, no fallback is attempted."""
        agent = FakePrimaryAgent(enable_fallback=False)
        result = await agent.fallback_generate("test prompt", status_code=429)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_providers_returns_none(self, agent):
        """When no fallback providers available, returns None."""
        agent._build_fallback_providers = lambda: []
        result = await agent.fallback_generate("test prompt", status_code=429)
        assert result is None

    @pytest.mark.asyncio
    async def test_fallback_tries_next_on_failure(self, agent):
        """If first fallback raises, second fallback is tried."""
        first = AsyncMock()
        first.generate = AsyncMock(side_effect=Exception("also down"))
        first.name = "provider-a"

        second = AsyncMock()
        second.generate = AsyncMock(return_value="second ok")
        second.name = "provider-b"

        agent._build_fallback_providers = lambda: [
            ("provider-a", first),
            ("provider-b", second),
        ]

        result = await agent.fallback_generate("prompt", status_code=429)

        assert result == "second ok"
        first.generate.assert_called_once()
        second.generate.assert_called_once()
