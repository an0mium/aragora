"""
Tests verifying that fallback agents receive requests on quota/rate-limit errors.

Covers the end-to-end flow: primary agent hits a quota error -> QuotaFallbackMixin
detects it -> fallback_generate delegates to the fallback agent.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.agents.fallback import QuotaFallbackMixin


class FakePrimaryAgent(QuotaFallbackMixin):
    """Minimal agent stub that uses QuotaFallbackMixin."""

    OPENROUTER_MODEL_MAP = {"test-model": "openrouter/test-model"}
    DEFAULT_FALLBACK_MODEL = "openrouter/default"

    def __init__(self):
        self.name = "fake-primary"
        self.model = "test-model"
        self.enable_fallback = True
        self.role = "proposer"
        self.timeout = 60
        self.system_prompt = None
        self._fallback_agent = None


class TestFallbackOnQuotaError:
    """Verify that quota errors cause the request to be routed to a fallback."""

    def test_429_detected_as_quota_error(self):
        agent = FakePrimaryAgent()
        assert agent.is_quota_error(429, "Too many requests")

    def test_401_detected_as_quota_error(self):
        agent = FakePrimaryAgent()
        assert agent.is_quota_error(401, "Unauthorized")

    def test_403_with_quota_keyword_detected(self):
        agent = FakePrimaryAgent()
        assert agent.is_quota_error(403, "Resource exhausted: quota exceeded")

    def test_400_billing_detected(self):
        agent = FakePrimaryAgent()
        assert agent.is_quota_error(400, "Your credit balance is too low")

    def test_normal_400_not_quota(self):
        agent = FakePrimaryAgent()
        assert not agent.is_quota_error(400, "Invalid request body")

    def test_200_not_quota(self):
        agent = FakePrimaryAgent()
        assert not agent.is_quota_error(200, "OK")

    @pytest.mark.asyncio
    async def test_fallback_generate_calls_fallback_agent_on_quota(self):
        """When primary raises quota error, fallback_generate uses the fallback agent."""
        agent = FakePrimaryAgent()

        mock_fallback = AsyncMock()
        mock_fallback.generate = AsyncMock(return_value="fallback response")
        mock_fallback.model = "openrouter/test-model"

        agent._fallback_agent = mock_fallback

        providers = [("openrouter", mock_fallback)]
        with patch.object(agent, "_build_fallback_providers", return_value=providers):
            result = await agent.fallback_generate("test prompt", status_code=429)

        assert result == "fallback response"
        mock_fallback.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fallback_generate_returns_none_when_disabled(self):
        """When enable_fallback is False, fallback_generate returns None."""
        agent = FakePrimaryAgent()
        agent.enable_fallback = False

        result = await agent.fallback_generate("test prompt", status_code=429)
        assert result is None

    @pytest.mark.asyncio
    async def test_fallback_generate_returns_none_when_no_providers(self):
        """When no fallback providers are available, returns None."""
        agent = FakePrimaryAgent()

        with patch.object(agent, "_build_fallback_providers", return_value=[]):
            result = await agent.fallback_generate("test prompt", status_code=429)

        assert result is None

    def test_timeout_status_codes_trigger_quota(self):
        """408, 504, 524 timeout codes should also be detected as quota errors."""
        agent = FakePrimaryAgent()
        for code in (408, 504, 524):
            assert agent.is_quota_error(code, ""), f"Status {code} should be quota error"
