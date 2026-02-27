"""End-to-end test for thinking trace flow from API response to receipt metadata.

Verifies the full pipeline:
1. AnthropicAPIAgent._parse_content_blocks() separates thinking from text
2. Thinking is stored as _last_thinking_trace
3. get_metadata() returns thinking and budget
4. Text output is clean (no thinking leaked)
"""

from __future__ import annotations

from typing import Any

import pytest

from aragora.agents.api_agents.anthropic import AnthropicAPIAgent


class TestThinkingFlowsToReceiptMetadata:
    """Integration test: thinking flows from API response to receipt metadata."""

    def test_thinking_flows_to_receipt_metadata(self):
        """Full pipeline: parse blocks -> store trace -> get_metadata -> clean output."""
        # 1. Create agent with thinking enabled
        agent = AnthropicAPIAgent(
            name="claude-thinking-test",
            model="claude-opus-4-6",
            thinking_budget=5000,
            api_key="test-key-not-used",
            enable_fallback=False,
        )

        # 2. Simulate API response content blocks with thinking + text
        content_blocks: list[dict[str, Any]] = [
            {
                "type": "thinking",
                "thinking": (
                    "Let me analyze this step by step. The user wants a rate limiter. "
                    "I should consider token bucket vs sliding window approaches. "
                    "Token bucket is simpler but sliding window provides smoother rate limiting."
                ),
            },
            {
                "type": "text",
                "text": (
                    "I recommend a sliding window rate limiter for your use case. "
                    "It provides smoother rate limiting compared to token bucket."
                ),
            },
        ]

        # 3. Parse the content blocks using the static method
        text_output, thinking_trace = AnthropicAPIAgent._parse_content_blocks(content_blocks)

        # 4. Verify thinking was captured
        assert thinking_trace is not None, "Thinking trace should not be None"
        assert "step by step" in thinking_trace
        assert "token bucket" in thinking_trace
        assert "sliding window" in thinking_trace

        # 5. Store as agent._last_thinking_trace (mirrors what generate() does)
        agent._last_thinking_trace = thinking_trace

        # 6. Verify get_metadata() returns the thinking and budget
        metadata = agent.get_metadata()
        assert metadata["thinking"] is not None
        assert metadata["thinking"] == thinking_trace
        assert metadata["thinking_budget"] == 5000
        assert "step by step" in metadata["thinking"]

        # 7. Verify text output is clean (no thinking leaked)
        assert "sliding window rate limiter" in text_output
        assert "Let me analyze this step by step" not in text_output
        assert text_output.startswith("I recommend")

    def test_no_thinking_blocks_returns_none(self):
        """When API response has no thinking blocks, trace is None."""
        content_blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Here is my response without extended thinking.",
            },
        ]

        text_output, thinking_trace = AnthropicAPIAgent._parse_content_blocks(content_blocks)

        assert thinking_trace is None
        assert "without extended thinking" in text_output

    def test_multiple_thinking_blocks_joined(self):
        """Multiple thinking blocks are joined with double newlines."""
        content_blocks: list[dict[str, Any]] = [
            {"type": "thinking", "thinking": "First chain of thought."},
            {"type": "thinking", "thinking": "Second chain of thought."},
            {"type": "text", "text": "Final answer."},
        ]

        text_output, thinking_trace = AnthropicAPIAgent._parse_content_blocks(content_blocks)

        assert thinking_trace is not None
        assert "First chain of thought." in thinking_trace
        assert "Second chain of thought." in thinking_trace
        # Multiple thinking blocks joined with \n\n
        assert "\n\n" in thinking_trace
        assert text_output == "Final answer."
