"""
Tests for fallback behavior under quota/rate-limit errors.

Mocks a primary agent that raises quota errors and verifies that
the fallback agent receives the request.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.agents.fallback import QuotaFallbackMixin


class FakePrimaryAgent(QuotaFallbackMixin):
    """A mock primary agent that uses the QuotaFallbackMixin."""

    OPENROUTER_MODEL_MAP = {"test-model": "openrouter/test-model"}
    DEFAULT_FALLBACK_MODEL = "openrouter/fallback-model"

    def __init__(self):
        self.name = "fake-primary"
        self.model = "test-model"
        self.enable_fallback = True
        self.timeout = 30
        self.role = "proposer"
        self.system_prompt = "You are a test agent."
        self._fallback_agent = None


class TestQuotaErrorDetection:
    """Verify is_quota_error identifies quota/rate-limit status codes."""

    @pytest.fixture()
    def agent(self):
        return FakePrimaryAgent()

    @pytest.mark.parametrize(
        "status_code",
        [401, 429, 403, 408, 504, 524],
    )
    def test_quota_status_codes_detected(self, agent, status_code):
        assert agent.is_quota_error(status_code, "rate limit exceeded")

    def test_normal_error_not_detected(self, agent):
        assert not agent.is_quota_error(200, "ok")

    def test_keyword_match_on_500(self, agent):
        assert agent.is_quota_error(500, "quota exceeded, try later")

    def test_no_keyword_on_500(self, agent):
        assert not agent.is_quota_error(500, "internal server error")


class TestFallbackGenerate:
    """Verify fallback_generate delegates to the fallback agent on quota errors."""

    @pytest.fixture()
    def agent(self):
        return FakePrimaryAgent()

    @pytest.mark.asyncio
    async def test_fallback_receives_request(self, agent):
        """Primary quota error triggers fallback agent generation."""
        mock_fallback = AsyncMock()
        mock_fallback.generate = AsyncMock(return_value="fallback response")

        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-key"}),
            patch.object(
                agent, "_build_fallback_providers", return_value=[("openrouter", mock_fallback)]
            ),
        ):
            result = await agent.fallback_generate("test prompt", status_code=429)

        assert result == "fallback response"
        mock_fallback.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fallback_disabled_returns_none(self, agent):
        """When fallback is disabled, returns None immediately."""
        agent.enable_fallback = False
        result = await agent.fallback_generate("test prompt", status_code=429)
        assert result is None

    @pytest.mark.asyncio
    async def test_fallback_no_providers_returns_none(self, agent):
        """When no fallback providers are available, returns None."""
        with patch.object(agent, "_build_fallback_providers", return_value=[]):
            result = await agent.fallback_generate("test prompt", status_code=429)
        assert result is None

    @pytest.mark.asyncio
    async def test_fallback_provider_also_fails(self, agent):
        """When fallback provider also fails, returns None."""
        mock_fallback = AsyncMock()
        mock_fallback.generate = AsyncMock(side_effect=RuntimeError("also down"))

        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-key"}),
            patch.object(
                agent, "_build_fallback_providers", return_value=[("openrouter", mock_fallback)]
            ),
        ):
            result = await agent.fallback_generate("test prompt", status_code=429)

        assert result is None
