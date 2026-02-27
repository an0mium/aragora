"""
Tests for Anthropic API Agent extended thinking support.

Tests cover:
- Thinking budget configuration
- Thinking disabled by default
- Extracting thinking blocks from API responses
- Handling responses with no thinking blocks
- Concatenating multiple thinking blocks
- Exposing thinking in get_metadata()
"""

from unittest.mock import patch

import pytest


class TestAnthropicThinking:
    """Tests for extended thinking support in AnthropicAPIAgent."""

    def test_thinking_budget_in_config(self, mock_env_with_api_keys):
        """Should accept thinking_budget in constructor and store it."""
        from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

        agent = AnthropicAPIAgent(thinking_budget=10_000)

        assert agent.thinking_budget == 10_000

    def test_thinking_disabled_by_default(self, mock_env_with_api_keys):
        """Should have thinking disabled (None) by default."""
        from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

        agent = AnthropicAPIAgent()

        assert agent.thinking_budget is None
        assert agent._last_thinking_trace is None

    @pytest.mark.asyncio
    async def test_extracts_thinking_from_response(self, mock_env_with_api_keys):
        """Should extract thinking blocks from API response and store them."""
        from aragora.agents.api_agents.anthropic import AnthropicAPIAgent
        from tests.agents.api_agents.conftest import MockClientSession, MockResponse

        agent = AnthropicAPIAgent(thinking_budget=8_000)

        response_data = {
            "id": "msg_thinking_01",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Let me reason about this step by step...",
                },
                {
                    "type": "text",
                    "text": "Here is my answer.",
                },
            ],
            "model": "claude-opus-4-6",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 200, "output_tokens": 100},
        }

        mock_response = MockResponse(status=200, json_data=response_data)
        mock_session = MockClientSession([mock_response])

        with patch(
            "aragora.agents.api_agents.anthropic.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.generate("Explain quantum computing")

        assert result == "Here is my answer."
        assert agent._last_thinking_trace == "Let me reason about this step by step..."
        assert agent.last_thinking_trace == "Let me reason about this step by step..."

    @pytest.mark.asyncio
    async def test_no_thinking_blocks_returns_none(self, mock_env_with_api_keys):
        """Should set _last_thinking_trace to None when no thinking blocks present."""
        from aragora.agents.api_agents.anthropic import AnthropicAPIAgent
        from tests.agents.api_agents.conftest import MockClientSession, MockResponse

        agent = AnthropicAPIAgent()

        response_data = {
            "id": "msg_no_thinking_01",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "A simple response without thinking.",
                },
            ],
            "model": "claude-opus-4-6",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 30},
        }

        mock_response = MockResponse(status=200, json_data=response_data)
        mock_session = MockClientSession([mock_response])

        with patch(
            "aragora.agents.api_agents.anthropic.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.generate("Hello")

        assert result == "A simple response without thinking."
        assert agent._last_thinking_trace is None

    @pytest.mark.asyncio
    async def test_multiple_thinking_blocks_concatenated(self, mock_env_with_api_keys):
        """Should concatenate multiple thinking blocks with double newlines."""
        from aragora.agents.api_agents.anthropic import AnthropicAPIAgent
        from tests.agents.api_agents.conftest import MockClientSession, MockResponse

        agent = AnthropicAPIAgent(thinking_budget=16_000)

        response_data = {
            "id": "msg_multi_thinking_01",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "First, let me consider the problem.",
                },
                {
                    "type": "thinking",
                    "thinking": "Now let me refine my approach.",
                },
                {
                    "type": "text",
                    "text": "My final answer.",
                },
            ],
            "model": "claude-opus-4-6",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 300, "output_tokens": 150},
        }

        mock_response = MockResponse(status=200, json_data=response_data)
        mock_session = MockClientSession([mock_response])

        with patch(
            "aragora.agents.api_agents.anthropic.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.generate("Complex reasoning task")

        # Verify via _parse_content_blocks directly
        text, thinking = agent._parse_content_blocks(response_data["content"])
        assert "First, let me consider the problem." in thinking
        assert "Now let me refine my approach." in thinking
        # Multiple thinking blocks joined with \n\n
        assert thinking == "First, let me consider the problem.\n\nNow let me refine my approach."
        assert result == "My final answer."

    def test_thinking_stored_in_metadata(self, mock_env_with_api_keys):
        """Should expose thinking trace and budget via get_metadata()."""
        from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

        agent = AnthropicAPIAgent(thinking_budget=12_000)
        # Simulate having received a thinking trace from a previous generation
        agent._last_thinking_trace = "This is my internal reasoning."

        metadata = agent.get_metadata()

        assert "thinking" in metadata
        assert metadata["thinking"] == "This is my internal reasoning."
        assert "thinking_budget" in metadata
        assert metadata["thinking_budget"] == 12_000

    def test_metadata_thinking_none_when_not_set(self, mock_env_with_api_keys):
        """Should return None for thinking in metadata when no thinking trace exists."""
        from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

        agent = AnthropicAPIAgent()

        metadata = agent.get_metadata()

        assert metadata["thinking"] is None
        assert metadata["thinking_budget"] is None
