"""Tests for TL;DR synthesis in PlaygroundHandler.

Covers:
- _synthesize_tldr returns a short result when the model call succeeds
- _synthesize_tldr returns the first sentence of fallback_text when the model call times out
- _call_frontier_model falls back to OpenRouter when Anthropic is unavailable
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.agents.errors import AgentCircuitOpenError
from aragora.server.handlers.playground import PlaygroundHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def handler() -> PlaygroundHandler:
    """Create a PlaygroundHandler instance for testing."""
    return PlaygroundHandler()


# ---------------------------------------------------------------------------
# _synthesize_tldr tests
# ---------------------------------------------------------------------------


class TestSynthesizeTldr:
    """Tests for PlaygroundHandler._synthesize_tldr."""

    def test_returns_model_response_on_success(self, handler: PlaygroundHandler) -> None:
        """When the frontier model call succeeds, _synthesize_tldr returns its output."""
        expected = "Use a token bucket algorithm with Redis for distributed rate limiting."
        with patch.object(handler, "_call_frontier_model", return_value=expected) as mock_call:
            result = handler._synthesize_tldr(
                question="How should I implement rate limiting?",
                proposals={
                    "analyst": "A token bucket algorithm provides the best balance of simplicity and flexibility.",
                    "critic": "Consider using a sliding window approach to avoid burst issues.",
                },
            )
        assert result == expected
        mock_call.assert_called_once()

    def test_falls_back_to_first_sentence_on_timeout(self, handler: PlaygroundHandler) -> None:
        """When the model call raises TimeoutError, extract the first sentence from fallback_text."""
        with patch.object(handler, "_call_frontier_model", side_effect=TimeoutError("timed out")):
            result = handler._synthesize_tldr(
                question="How should I implement rate limiting?",
                proposals={"analyst": "Use tokens."},
                fallback_text="Use a token bucket with Redis for distributed systems. This ensures fair rate limiting across all nodes.",
            )
        assert result == "Use a token bucket with Redis for distributed systems."

    def test_falls_back_to_truncated_text_when_no_sentence_boundary(
        self, handler: PlaygroundHandler
    ) -> None:
        """When fallback_text has no '. ' boundary, return first 200 chars."""
        with patch.object(
            handler, "_call_frontier_model", side_effect=OSError("connection failed")
        ):
            result = handler._synthesize_tldr(
                question="What is the meaning of life?",
                proposals={"analyst": "42"},
                fallback_text="A single long run-on sentence with no period boundary",
            )
        assert result == "A single long run-on sentence with no period boundary"
        assert len(result) <= 200

    def test_falls_back_on_connection_error(self, handler: PlaygroundHandler) -> None:
        """ConnectionError (network issue) should trigger fallback, not crash."""
        with patch.object(handler, "_call_frontier_model", side_effect=ConnectionError("refused")):
            result = handler._synthesize_tldr(
                question="Test?",
                proposals={"a": "Text"},
                fallback_text="Fallback sentence here. More text follows.",
            )
        assert result == "Fallback sentence here."

    def test_falls_back_on_runtime_error(self, handler: PlaygroundHandler) -> None:
        """RuntimeError (no agent available) should trigger fallback."""
        with patch.object(
            handler, "_call_frontier_model", side_effect=RuntimeError("No agent available")
        ):
            result = handler._synthesize_tldr(
                question="Test?",
                proposals={"a": "Text"},
                fallback_text="First sentence here. Second sentence.",
            )
        assert result == "First sentence here."

    def test_falls_back_on_agent_circuit_open_error(self, handler: PlaygroundHandler) -> None:
        """Agent circuit breaker failures should not break the public fallback path."""
        with patch.object(
            handler,
            "_call_frontier_model",
            side_effect=AgentCircuitOpenError("Circuit open", agent_name="tldr-synth"),
        ):
            result = handler._synthesize_tldr(
                question="Test?",
                proposals={"a": "Text"},
                fallback_text="Fallback sentence here. More text follows.",
            )
        assert result == "Fallback sentence here."

    def test_empty_fallback_text_returns_empty_string(self, handler: PlaygroundHandler) -> None:
        """When fallback_text is empty and model fails, return empty string."""
        with patch.object(handler, "_call_frontier_model", side_effect=TimeoutError):
            result = handler._synthesize_tldr(
                question="Test?",
                proposals={"a": "Text"},
                fallback_text="",
            )
        assert result == ""

    def test_none_fallback_text_returns_empty_string(self, handler: PlaygroundHandler) -> None:
        """When fallback_text is None and model fails, return empty string."""
        with patch.object(handler, "_call_frontier_model", side_effect=TimeoutError):
            result = handler._synthesize_tldr(
                question="Test?",
                proposals={"a": "Text"},
                fallback_text=None,
            )
        assert result == ""

    def test_prompt_truncates_long_proposals(self, handler: PlaygroundHandler) -> None:
        """Proposals should be truncated to 500 chars in the prompt."""
        long_text = "x" * 1000
        captured_prompt = None

        def capture_prompt(prompt: str, timeout: float = 5.0) -> str:
            nonlocal captured_prompt
            captured_prompt = prompt
            return "Short answer."

        with patch.object(handler, "_call_frontier_model", side_effect=capture_prompt):
            handler._synthesize_tldr(
                question="Test?",
                proposals={"verbose_agent": long_text},
            )
        assert captured_prompt is not None
        # The agent text in the prompt should be truncated to 500
        assert "x" * 501 not in captured_prompt
        assert "x" * 500 in captured_prompt


# ---------------------------------------------------------------------------
# _call_frontier_model tests
# ---------------------------------------------------------------------------


class TestCallFrontierModel:
    """Tests for PlaygroundHandler._call_frontier_model."""

    def test_uses_anthropic_agent_first(self, handler: PlaygroundHandler) -> None:
        """When Anthropic agent is available, use it."""
        mock_agent = MagicMock()
        mock_agent.generate = AsyncMock(return_value="Anthropic response")

        with patch(
            "aragora.agents.api_agents.anthropic.AnthropicAPIAgent",
            return_value=mock_agent,
        ) as mock_cls:
            result = handler._call_frontier_model("test prompt", timeout=5.0)

        assert result == "Anthropic response"
        mock_cls.assert_called_once()
        mock_agent.generate.assert_awaited_once_with("test prompt")

    def test_falls_back_to_openrouter_when_anthropic_unavailable(
        self, handler: PlaygroundHandler
    ) -> None:
        """When Anthropic import fails, fall back to OpenRouter."""
        mock_agent = MagicMock()
        mock_agent.generate = AsyncMock(return_value="OpenRouter response")

        with (
            patch(
                "aragora.agents.api_agents.anthropic.AnthropicAPIAgent",
                side_effect=ImportError("no anthropic"),
            ),
            patch(
                "aragora.agents.api_agents.openrouter.OpenRouterAgent",
                return_value=mock_agent,
            ) as mock_or_cls,
        ):
            result = handler._call_frontier_model("test prompt", timeout=5.0)

        assert result == "OpenRouter response"
        mock_or_cls.assert_called_once()
        mock_agent.generate.assert_awaited_once_with("test prompt")

    def test_falls_back_to_openrouter_on_anthropic_runtime_error(
        self, handler: PlaygroundHandler
    ) -> None:
        """When Anthropic agent raises RuntimeError (e.g. missing API key), fall back."""
        mock_agent = MagicMock()
        mock_agent.generate = AsyncMock(return_value="OpenRouter fallback")

        with (
            patch(
                "aragora.agents.api_agents.anthropic.AnthropicAPIAgent",
                side_effect=RuntimeError("No API key"),
            ),
            patch(
                "aragora.agents.api_agents.openrouter.OpenRouterAgent",
                return_value=mock_agent,
            ),
        ):
            result = handler._call_frontier_model("test prompt")

        assert result == "OpenRouter fallback"

    def test_raises_on_timeout(self, handler: PlaygroundHandler) -> None:
        """When the agent call exceeds timeout, TimeoutError is raised."""
        mock_agent = MagicMock()

        async def slow_generate(prompt: str) -> str:
            await asyncio.sleep(10)
            return "too slow"

        mock_agent.generate = slow_generate

        with patch(
            "aragora.agents.api_agents.anthropic.AnthropicAPIAgent",
            return_value=mock_agent,
        ):
            with pytest.raises(TimeoutError):
                handler._call_frontier_model("test prompt", timeout=0.1)

    def test_raises_when_no_agent_available(self, handler: PlaygroundHandler) -> None:
        """When neither Anthropic nor OpenRouter is available, raise RuntimeError."""
        with (
            patch(
                "aragora.agents.api_agents.anthropic.AnthropicAPIAgent",
                side_effect=ImportError("no module"),
            ),
            patch(
                "aragora.agents.api_agents.openrouter.OpenRouterAgent",
                side_effect=ImportError("no module"),
            ),
        ):
            with pytest.raises(RuntimeError, match="No frontier model available"):
                handler._call_frontier_model("test prompt")
