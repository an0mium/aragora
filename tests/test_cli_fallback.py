"""
Tests for CLI agent fallback behavior.

Tests that CLI agents correctly detect errors that should trigger
fallback to OpenRouter, and that fallback is properly activated.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import test targets
from aragora.agents.cli_agents import (
    RATE_LIMIT_PATTERNS,
    CLIAgent,
    ClaudeAgent,
    GeminiCLIAgent,
    CodexAgent,
)


class TestRateLimitPatterns:
    """Test the RATE_LIMIT_PATTERNS constant."""

    def test_patterns_include_rate_limit(self) -> None:
        """Patterns include basic rate limit text."""
        assert any("rate limit" in p.lower() for p in RATE_LIMIT_PATTERNS)
        assert any("rate_limit" in p.lower() for p in RATE_LIMIT_PATTERNS)

    def test_patterns_include_429(self) -> None:
        """Patterns include HTTP 429."""
        assert "429" in RATE_LIMIT_PATTERNS

    def test_patterns_include_quota(self) -> None:
        """Patterns include quota errors."""
        assert any("quota" in p.lower() for p in RATE_LIMIT_PATTERNS)

    def test_patterns_include_throttle(self) -> None:
        """Patterns include throttle errors."""
        assert any("throttl" in p.lower() for p in RATE_LIMIT_PATTERNS)

    def test_patterns_include_billing(self) -> None:
        """Patterns include billing/credit errors."""
        assert any("billing" in p.lower() or "credit" in p.lower() for p in RATE_LIMIT_PATTERNS)


class TestIsFallbackError:
    """Test the _is_fallback_error method."""

    def setup_method(self) -> None:
        """Create a test agent instance."""
        self.agent = ClaudeAgent(name="test-claude", model="claude-3", role="test")

    def test_rate_limit_triggers_fallback(self) -> None:
        """Rate limit error triggers fallback."""
        error = Exception("Error: rate limit exceeded")
        assert self.agent._is_fallback_error(error) is True

    def test_429_triggers_fallback(self) -> None:
        """HTTP 429 triggers fallback."""
        error = Exception("HTTP Error 429: Too Many Requests")
        assert self.agent._is_fallback_error(error) is True

    def test_quota_triggers_fallback(self) -> None:
        """Quota exceeded triggers fallback."""
        error = Exception("Error: quota exceeded for this model")
        assert self.agent._is_fallback_error(error) is True

    def test_timeout_triggers_fallback(self) -> None:
        """TimeoutError triggers fallback."""
        error = TimeoutError("Request timed out")
        assert self.agent._is_fallback_error(error) is True

    def test_asyncio_timeout_triggers_fallback(self) -> None:
        """asyncio.TimeoutError triggers fallback."""
        error = asyncio.TimeoutError()
        assert self.agent._is_fallback_error(error) is True

    def test_connection_error_triggers_fallback(self) -> None:
        """ConnectionError triggers fallback."""
        error = ConnectionError("Connection refused")
        assert self.agent._is_fallback_error(error) is True

    def test_connection_refused_triggers_fallback(self) -> None:
        """ConnectionRefusedError triggers fallback."""
        error = ConnectionRefusedError("Connection refused")
        assert self.agent._is_fallback_error(error) is True

    def test_503_triggers_fallback(self) -> None:
        """Service unavailable (503) triggers fallback."""
        error = Exception("HTTP Error 503: Service Unavailable")
        assert self.agent._is_fallback_error(error) is True

    def test_resource_exhausted_triggers_fallback(self) -> None:
        """Resource exhausted error triggers fallback."""
        error = Exception("RESOURCE_EXHAUSTED: API quota exceeded")
        assert self.agent._is_fallback_error(error) is True

    def test_regular_error_does_not_trigger_fallback(self) -> None:
        """Regular errors don't trigger fallback."""
        error = ValueError("Invalid input")
        assert self.agent._is_fallback_error(error) is False

    def test_file_not_found_triggers_fallback(self) -> None:
        """FileNotFoundError triggers fallback (CLI not installed)."""
        error = FileNotFoundError("claude: command not found")
        assert self.agent._is_fallback_error(error) is True


class TestGetFallbackAgent:
    """Test the _get_fallback_agent method."""

    def setup_method(self) -> None:
        """Create test agent."""
        self.agent = ClaudeAgent(name="test-claude", model="claude-3", role="test")

    def test_returns_none_without_api_key(self) -> None:
        """Returns None if OPENROUTER_API_KEY not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure key is not set
            os.environ.pop("OPENROUTER_API_KEY", None)
            assert self.agent._get_fallback_agent() is None

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    def test_creates_fallback_with_api_key(self) -> None:
        """Creates fallback agent when API key is available."""
        fallback = self.agent._get_fallback_agent()
        assert fallback is not None
        assert "fallback" in fallback.name

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    def test_caches_fallback_agent(self) -> None:
        """Fallback agent is cached on first creation."""
        fallback1 = self.agent._get_fallback_agent()
        fallback2 = self.agent._get_fallback_agent()
        assert fallback1 is fallback2

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    def test_fallback_copies_system_prompt(self) -> None:
        """Fallback agent inherits system prompt."""
        self.agent.system_prompt = "Be helpful."
        fallback = self.agent._get_fallback_agent()
        assert fallback.system_prompt == "Be helpful."


class TestFallbackActivation:
    """Test that fallback is actually activated on errors."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    async def test_fallback_used_on_rate_limit(self) -> None:
        """Fallback is used when CLI returns rate limit error."""
        agent = ClaudeAgent(name="test-claude", model="claude-3", role="test")

        # Mock _run_cli to raise rate limit error
        async def mock_run_cli(*args, **kwargs):
            raise Exception("Error: rate limit exceeded")

        agent._run_cli = mock_run_cli

        # Mock the fallback agent's generate
        mock_fallback = AsyncMock()
        mock_fallback.generate = AsyncMock(return_value="Fallback response")
        agent._fallback_agent = mock_fallback

        result = await agent.generate("Test prompt")

        assert result == "Fallback response"
        assert agent._fallback_used is True
        mock_fallback.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_fallback_on_regular_error(self) -> None:
        """No fallback on regular errors (re-raises)."""
        agent = ClaudeAgent(name="test-claude", model="claude-3", role="test")

        async def mock_run_cli(*args, **kwargs):
            raise ValueError("Invalid input")

        agent._run_cli = mock_run_cli

        with pytest.raises(ValueError):
            await agent.generate("Test prompt")

        assert agent._fallback_used is False


class TestFallbackUsedTracking:
    """Test tracking of fallback usage."""

    def test_fallback_used_initially_false(self) -> None:
        """_fallback_used starts as False."""
        agent = ClaudeAgent(name="test", model="claude-3", role="test")
        assert agent._fallback_used is False

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    async def test_fallback_used_set_on_fallback(self) -> None:
        """_fallback_used is set to True when fallback is used."""
        agent = ClaudeAgent(name="test", model="claude-3", role="test")

        async def mock_run_cli(*args, **kwargs):
            raise Exception("429 rate limit")

        agent._run_cli = mock_run_cli

        mock_fallback = AsyncMock()
        mock_fallback.generate = AsyncMock(return_value="response")
        agent._fallback_agent = mock_fallback

        await agent.generate("test")

        assert agent._fallback_used is True


class TestMultipleCLIAgents:
    """Test fallback behavior across different CLI agent types."""

    @pytest.mark.parametrize("agent_class,agent_name,model", [
        (ClaudeAgent, "claude", "claude-3"),
        (GeminiCLIAgent, "gemini", "gemini-pro"),
        (CodexAgent, "codex", "codex"),
    ])
    def test_all_agents_have_fallback_method(self, agent_class, agent_name, model) -> None:
        """All CLI agents have _is_fallback_error method."""
        agent = agent_class(name=agent_name, model=model, role="test")
        assert hasattr(agent, "_is_fallback_error")
        assert callable(agent._is_fallback_error)

    @pytest.mark.parametrize("agent_class,agent_name,model", [
        (ClaudeAgent, "claude", "claude-3"),
        (GeminiCLIAgent, "gemini", "gemini-pro"),
        (CodexAgent, "codex", "codex"),
    ])
    def test_all_agents_have_get_fallback(self, agent_class, agent_name, model) -> None:
        """All CLI agents have _get_fallback_agent method."""
        agent = agent_class(name=agent_name, model=model, role="test")
        assert hasattr(agent, "_get_fallback_agent")
        assert callable(agent._get_fallback_agent)


class TestEdgeCases:
    """Test edge cases in fallback behavior."""

    def test_case_insensitive_pattern_matching(self) -> None:
        """Pattern matching is case insensitive."""
        agent = ClaudeAgent(name="test", model="claude-3", role="test")

        # Should match regardless of case
        assert agent._is_fallback_error(Exception("RATE LIMIT EXCEEDED"))
        assert agent._is_fallback_error(Exception("Rate Limit Exceeded"))
        assert agent._is_fallback_error(Exception("rate limit exceeded"))

    def test_partial_pattern_matching(self) -> None:
        """Patterns match as substrings."""
        agent = ClaudeAgent(name="test", model="claude-3", role="test")

        # Should match within longer message
        assert agent._is_fallback_error(
            Exception("The API returned: rate limit exceeded, please retry later")
        )

    def test_empty_error_message(self) -> None:
        """Empty error message doesn't cause crash."""
        agent = ClaudeAgent(name="test", model="claude-3", role="test")

        # Should not crash and should not trigger fallback
        result = agent._is_fallback_error(Exception(""))
        assert result is False

    def test_none_like_error_handling(self) -> None:
        """Handles errors with None-like string representation."""
        agent = ClaudeAgent(name="test", model="claude-3", role="test")

        class WeirdError(Exception):
            def __str__(self):
                return ""

        result = agent._is_fallback_error(WeirdError())
        assert result is False
