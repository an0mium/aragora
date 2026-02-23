"""Comprehensive tests for the BlocksMixin in _slack_impl/blocks.py.

Covers every public/internal method of BlocksMixin:
- _build_starting_blocks (basic, with agents, with rounds, with both, empty topic)
- _post_round_update (Web API path, response_url fallback, all phases, unknown phase)
- _post_agent_response (Web API path, response_url fallback, all agent emojis, truncation)
- _build_result_blocks (consensus/no-consensus, winner, confidence levels, receipt URL,
  participant truncation, long answers, action buttons)
- Edge cases: empty strings, None values, special characters, XSS-like input
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _make_debate_result(
    debate_id: str = "debate-001",
    consensus_reached: bool = True,
    confidence: float = 0.8,
    participants: list[str] | None = None,
    rounds_used: int = 3,
    final_answer: str = "The best approach is X.",
    winner: str | None = None,
) -> SimpleNamespace:
    """Create a mock debate result object."""
    return SimpleNamespace(
        id=debate_id,
        consensus_reached=consensus_reached,
        confidence=confidence,
        participants=participants or ["claude", "gpt-4", "gemini"],
        rounds_used=rounds_used,
        final_answer=final_answer,
        winner=winner,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler_module():
    """Import the handler module lazily (after conftest patches)."""
    from aragora.server.handlers.social._slack_impl import handler as mod

    return mod


@pytest.fixture
def blocks_module():
    """Import the blocks module lazily."""
    from aragora.server.handlers.social._slack_impl import blocks as mod

    return mod


@pytest.fixture
def config_module():
    """Import the config module lazily."""
    from aragora.server.handlers.social._slack_impl import config as mod

    return mod


@pytest.fixture
def slack_handler(handler_module):
    """Create a SlackHandler with empty context."""
    return handler_module.SlackHandler(ctx={})


@pytest.fixture(autouse=True)
def _reset_config_singletons(config_module, monkeypatch):
    """Reset module-level singletons between tests."""
    monkeypatch.setattr(config_module, "_slack_audit", None)
    monkeypatch.setattr(config_module, "_slack_user_limiter", None)
    monkeypatch.setattr(config_module, "_slack_workspace_limiter", None)
    monkeypatch.setattr(config_module, "_slack_integration", None)
    yield


@pytest.fixture(autouse=True)
def _disable_rate_limit_decorator(monkeypatch):
    """Disable the @rate_limit decorator so it does not interfere with tests."""
    try:
        from aragora.server.handlers.utils import rate_limit as rl_mod

        monkeypatch.setattr(rl_mod, "_RATE_LIMIT_DISABLED", True, raising=False)
    except (ImportError, AttributeError):
        pass


# ===========================================================================
# _build_starting_blocks
# ===========================================================================


class TestBuildStartingBlocks:
    """Tests for BlocksMixin._build_starting_blocks."""

    def test_basic_starting_blocks(self, slack_handler):
        """Basic call returns header, section with topic, and context."""
        blocks = slack_handler._build_starting_blocks(
            topic="Should we adopt Rust?",
            user_id="U123",
            debate_id="d-001",
        )
        assert isinstance(blocks, list)
        assert len(blocks) == 3

    def test_header_block(self, slack_handler):
        """First block is a header with 'Debate Starting...'."""
        blocks = slack_handler._build_starting_blocks(
            topic="topic", user_id="U1", debate_id="d-1"
        )
        header = blocks[0]
        assert header["type"] == "header"
        assert header["text"]["type"] == "plain_text"
        assert header["text"]["text"] == "Debate Starting..."
        assert header["text"]["emoji"] is True

    def test_topic_section(self, slack_handler):
        """Second block contains the topic in mrkdwn."""
        blocks = slack_handler._build_starting_blocks(
            topic="AI Safety", user_id="U1", debate_id="d-1"
        )
        section = blocks[1]
        assert section["type"] == "section"
        assert "*Topic:* AI Safety" in section["text"]["text"]

    def test_context_block_user_and_id(self, slack_handler):
        """Context block includes user mention and debate ID."""
        blocks = slack_handler._build_starting_blocks(
            topic="topic", user_id="U42", debate_id="d-99"
        )
        context = blocks[2]
        assert context["type"] == "context"
        text = context["elements"][0]["text"]
        assert "<@U42>" in text
        assert "`d-99`" in text

    def test_with_agents(self, slack_handler):
        """Agents are appended to context when provided."""
        blocks = slack_handler._build_starting_blocks(
            topic="topic",
            user_id="U1",
            debate_id="d-1",
            agents=["claude", "gpt-4"],
        )
        text = blocks[2]["elements"][0]["text"]
        assert "claude" in text
        assert "gpt-4" in text
        assert "Agents:" in text

    def test_with_expected_rounds(self, slack_handler):
        """Expected rounds appear in context when provided."""
        blocks = slack_handler._build_starting_blocks(
            topic="topic",
            user_id="U1",
            debate_id="d-1",
            expected_rounds=5,
        )
        text = blocks[2]["elements"][0]["text"]
        assert "Rounds: 5" in text

    def test_with_agents_and_rounds(self, slack_handler):
        """Both agents and rounds appear in context together."""
        blocks = slack_handler._build_starting_blocks(
            topic="topic",
            user_id="U1",
            debate_id="d-1",
            agents=["claude"],
            expected_rounds=3,
        )
        text = blocks[2]["elements"][0]["text"]
        assert "Agents: claude" in text
        assert "Rounds: 3" in text

    def test_no_agents_no_rounds_context(self, slack_handler):
        """Without agents or rounds, context only has user and ID."""
        blocks = slack_handler._build_starting_blocks(
            topic="topic", user_id="U1", debate_id="d-1"
        )
        text = blocks[2]["elements"][0]["text"]
        assert "Agents:" not in text
        assert "Rounds:" not in text

    def test_empty_agents_list(self, slack_handler):
        """Empty agents list is treated as no agents."""
        blocks = slack_handler._build_starting_blocks(
            topic="topic",
            user_id="U1",
            debate_id="d-1",
            agents=[],
        )
        text = blocks[2]["elements"][0]["text"]
        assert "Agents:" not in text

    def test_zero_rounds_not_shown(self, slack_handler):
        """expected_rounds=0 is falsy, should not appear."""
        blocks = slack_handler._build_starting_blocks(
            topic="topic",
            user_id="U1",
            debate_id="d-1",
            expected_rounds=0,
        )
        text = blocks[2]["elements"][0]["text"]
        assert "Rounds:" not in text

    def test_special_characters_in_topic(self, slack_handler):
        """Special characters in topic are preserved (not escaped)."""
        blocks = slack_handler._build_starting_blocks(
            topic="<script>alert('xss')</script>",
            user_id="U1",
            debate_id="d-1",
        )
        assert "<script>" in blocks[1]["text"]["text"]

    def test_long_topic(self, slack_handler):
        """A very long topic is passed through without truncation at this layer."""
        topic = "A" * 5000
        blocks = slack_handler._build_starting_blocks(
            topic=topic, user_id="U1", debate_id="d-1"
        )
        assert topic in blocks[1]["text"]["text"]

    def test_many_agents(self, slack_handler):
        """Many agents are comma-joined in context."""
        agents = [f"agent-{i}" for i in range(20)]
        blocks = slack_handler._build_starting_blocks(
            topic="topic",
            user_id="U1",
            debate_id="d-1",
            agents=agents,
        )
        text = blocks[2]["elements"][0]["text"]
        assert "agent-0" in text
        assert "agent-19" in text

    def test_context_parts_joined_by_pipe(self, slack_handler):
        """Context parts are joined with ' | ' separator."""
        blocks = slack_handler._build_starting_blocks(
            topic="topic",
            user_id="U1",
            debate_id="d-1",
            agents=["claude"],
            expected_rounds=5,
        )
        text = blocks[2]["elements"][0]["text"]
        # The first part contains "Requested by <@U1> | ID: `d-1`"
        # then agents, then rounds => 4 pipe-separated segments
        parts = text.split(" | ")
        assert len(parts) == 4

    def test_single_agent(self, slack_handler):
        """A single agent in the list."""
        blocks = slack_handler._build_starting_blocks(
            topic="topic",
            user_id="U1",
            debate_id="d-1",
            agents=["claude"],
        )
        text = blocks[2]["elements"][0]["text"]
        assert "Agents: claude" in text

    def test_returns_list_of_dicts(self, slack_handler):
        """Return type is a list of dicts."""
        blocks = slack_handler._build_starting_blocks(
            topic="x", user_id="U", debate_id="d"
        )
        assert all(isinstance(b, dict) for b in blocks)


# ===========================================================================
# _post_round_update
# ===========================================================================


class TestPostRoundUpdate:
    """Tests for BlocksMixin._post_round_update."""

    @pytest.mark.asyncio
    async def test_uses_web_api_when_bot_token_and_channel(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """When SLACK_BOT_TOKEN, channel_id, and thread_ts are set, uses Web API."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", "xoxb-fake-token")
        slack_handler._post_message_async = AsyncMock()
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_round_update(
            response_url="https://hooks.slack.com/resp/1",
            topic="Topic",
            round_num=2,
            total_rounds=5,
            agent="claude",
            channel_id="C123",
            thread_ts="1234567890.123456",
        )

        slack_handler._post_message_async.assert_awaited_once()
        slack_handler._post_to_response_url.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_falls_back_to_response_url_no_token(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Falls back to response_url when no SLACK_BOT_TOKEN."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_message_async = AsyncMock()
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_round_update(
            response_url="https://hooks.slack.com/resp/1",
            topic="Topic",
            round_num=1,
            total_rounds=3,
            agent="gpt-4",
        )

        slack_handler._post_to_response_url.assert_awaited_once()
        slack_handler._post_message_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_falls_back_no_channel_id(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Falls back to response_url when no channel_id."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", "xoxb-token")
        slack_handler._post_message_async = AsyncMock()
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_round_update(
            response_url="https://hooks.slack.com/resp/1",
            topic="Topic",
            round_num=1,
            total_rounds=3,
            agent="gpt-4",
            channel_id=None,
            thread_ts="12345.67890",
        )

        slack_handler._post_to_response_url.assert_awaited_once()
        slack_handler._post_message_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_falls_back_no_thread_ts(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Falls back to response_url when no thread_ts."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", "xoxb-token")
        slack_handler._post_message_async = AsyncMock()
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_round_update(
            response_url="https://hooks.slack.com/resp/1",
            topic="Topic",
            round_num=1,
            total_rounds=3,
            agent="gpt-4",
            channel_id="C123",
            thread_ts=None,
        )

        slack_handler._post_to_response_url.assert_awaited_once()
        slack_handler._post_message_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_progress_bar_visual(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Progress bar uses filled/empty squares proportionally."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_round_update(
            response_url="https://hooks.slack.com/resp/1",
            topic="Topic",
            round_num=3,
            total_rounds=5,
            agent="claude",
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        block_text = payload["blocks"][0]["text"]["text"]
        assert block_text.count(":black_large_square:") == 3
        assert block_text.count(":white_large_square:") == 2

    @pytest.mark.asyncio
    async def test_progress_bar_complete(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """When round_num == total_rounds, all squares are filled."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_round_update(
            response_url="https://hooks.slack.com/resp/1",
            topic="Topic",
            round_num=5,
            total_rounds=5,
            agent="claude",
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        block_text = payload["blocks"][0]["text"]["text"]
        assert block_text.count(":black_large_square:") == 5
        assert ":white_large_square:" not in block_text

    @pytest.mark.asyncio
    async def test_progress_bar_start(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """At round 0, all squares are empty."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_round_update(
            response_url="https://hooks.slack.com/resp/1",
            topic="Topic",
            round_num=0,
            total_rounds=5,
            agent="claude",
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        block_text = payload["blocks"][0]["text"]["text"]
        assert ":black_large_square:" not in block_text
        assert block_text.count(":white_large_square:") == 5

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "phase,expected_emoji",
        [
            ("analyzing", ":mag:"),
            ("critique", ":speech_balloon:"),
            ("voting", ":ballot_box:"),
            ("complete", ":white_check_mark:"),
        ],
    )
    async def test_phase_emojis(
        self, slack_handler, blocks_module, monkeypatch, phase, expected_emoji
    ):
        """Each known phase gets its designated emoji."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_round_update(
            response_url="https://hooks.slack.com/resp/1",
            topic="Topic",
            round_num=1,
            total_rounds=3,
            agent="claude",
            phase=phase,
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        block_text = payload["blocks"][0]["text"]["text"]
        assert expected_emoji in block_text

    @pytest.mark.asyncio
    async def test_unknown_phase_fallback_emoji(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Unknown phase uses hourglass emoji."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_round_update(
            response_url="https://hooks.slack.com/resp/1",
            topic="Topic",
            round_num=1,
            total_rounds=3,
            agent="claude",
            phase="unknown_phase",
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        block_text = payload["blocks"][0]["text"]["text"]
        assert ":hourglass_flowing_sand:" in block_text

    @pytest.mark.asyncio
    async def test_default_phase_is_analyzing(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Default phase is 'analyzing'."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_round_update(
            response_url="https://hooks.slack.com/resp/1",
            topic="Topic",
            round_num=1,
            total_rounds=3,
            agent="claude",
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        block_text = payload["blocks"][0]["text"]["text"]
        assert ":mag:" in block_text

    @pytest.mark.asyncio
    async def test_agent_name_in_block(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Agent name appears in the block text."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_round_update(
            response_url="https://hooks.slack.com/resp/1",
            topic="Topic",
            round_num=2,
            total_rounds=4,
            agent="gemini-pro",
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        block_text = payload["blocks"][0]["text"]["text"]
        assert "gemini-pro responded" in block_text

    @pytest.mark.asyncio
    async def test_round_info_in_text(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Round info appears in both text and block."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_round_update(
            response_url="https://hooks.slack.com/resp/1",
            topic="Topic",
            round_num=3,
            total_rounds=7,
            agent="claude",
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        assert "Round 3/7 complete" in payload["text"]
        block_text = payload["blocks"][0]["text"]["text"]
        assert "Round 3/7" in block_text

    @pytest.mark.asyncio
    async def test_response_url_payload_structure(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Response URL payload has correct structure."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_round_update(
            response_url="https://hooks.slack.com/resp/1",
            topic="Topic",
            round_num=1,
            total_rounds=3,
            agent="claude",
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        assert payload["response_type"] == "in_channel"
        assert payload["replace_original"] is False
        assert "blocks" in payload
        assert "text" in payload

    @pytest.mark.asyncio
    async def test_web_api_passes_thread_ts(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Web API call includes thread_ts for threading."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", "xoxb-token")
        slack_handler._post_message_async = AsyncMock()

        await slack_handler._post_round_update(
            response_url="https://hooks.slack.com/resp/1",
            topic="Topic",
            round_num=1,
            total_rounds=3,
            agent="claude",
            channel_id="C123",
            thread_ts="1234.5678",
        )

        call_kwargs = slack_handler._post_message_async.call_args[1]
        assert call_kwargs["thread_ts"] == "1234.5678"
        assert call_kwargs["channel"] == "C123"


# ===========================================================================
# _post_agent_response
# ===========================================================================


class TestPostAgentResponse:
    """Tests for BlocksMixin._post_agent_response."""

    @pytest.mark.asyncio
    async def test_uses_web_api_when_bot_token_and_channel(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Uses Web API when bot token, channel, and thread_ts are present."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", "xoxb-token")
        slack_handler._post_message_async = AsyncMock()
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_agent_response(
            response_url="https://hooks.slack.com/resp/1",
            agent="claude",
            response="Hello world",
            round_num=1,
            channel_id="C123",
            thread_ts="1234.5678",
        )

        slack_handler._post_message_async.assert_awaited_once()
        slack_handler._post_to_response_url.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_falls_back_to_response_url(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Falls back to response_url without token."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()
        slack_handler._post_message_async = AsyncMock()

        await slack_handler._post_agent_response(
            response_url="https://hooks.slack.com/resp/1",
            agent="claude",
            response="Hello world",
            round_num=1,
        )

        slack_handler._post_to_response_url.assert_awaited_once()
        slack_handler._post_message_async.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "agent_name,expected_emoji",
        [
            ("anthropic-api", ":robot_face:"),
            ("openai-api", ":brain:"),
            ("gemini", ":gem:"),
            ("grok", ":zap:"),
            ("mistral", ":wind_face:"),
            ("deepseek", ":mag:"),
        ],
    )
    async def test_agent_emoji_mapping(
        self,
        slack_handler,
        blocks_module,
        monkeypatch,
        agent_name,
        expected_emoji,
    ):
        """Each known agent gets its designated emoji."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_agent_response(
            response_url="https://hooks.slack.com/resp/1",
            agent=agent_name,
            response="response text",
            round_num=1,
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        context_text = payload["blocks"][0]["elements"][0]["text"]
        assert expected_emoji in context_text

    @pytest.mark.asyncio
    async def test_unknown_agent_default_emoji(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Unknown agent gets the default speech_balloon emoji."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_agent_response(
            response_url="https://hooks.slack.com/resp/1",
            agent="unknown-agent",
            response="response text",
            round_num=1,
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        context_text = payload["blocks"][0]["elements"][0]["text"]
        assert ":speech_balloon:" in context_text

    @pytest.mark.asyncio
    async def test_agent_name_case_insensitive(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Agent name matching is case-insensitive."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_agent_response(
            response_url="https://hooks.slack.com/resp/1",
            agent="ANTHROPIC-API",
            response="text",
            round_num=1,
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        context_text = payload["blocks"][0]["elements"][0]["text"]
        assert ":robot_face:" in context_text

    @pytest.mark.asyncio
    async def test_response_truncation_long(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Responses longer than 2800 chars are truncated with '...'."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        long_response = "A" * 3000

        await slack_handler._post_agent_response(
            response_url="https://hooks.slack.com/resp/1",
            agent="claude",
            response=long_response,
            round_num=1,
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        section_text = payload["blocks"][1]["text"]["text"]
        assert len(section_text) == 2803  # 2800 + "..."
        assert section_text.endswith("...")

    @pytest.mark.asyncio
    async def test_response_not_truncated_short(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Responses under 2800 chars are not truncated."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        short_response = "A" * 2800

        await slack_handler._post_agent_response(
            response_url="https://hooks.slack.com/resp/1",
            agent="claude",
            response=short_response,
            round_num=1,
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        section_text = payload["blocks"][1]["text"]["text"]
        assert section_text == short_response
        assert not section_text.endswith("...")

    @pytest.mark.asyncio
    async def test_response_exactly_2800(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Responses of exactly 2800 chars are not truncated."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        response = "B" * 2800

        await slack_handler._post_agent_response(
            response_url="https://hooks.slack.com/resp/1",
            agent="claude",
            response=response,
            round_num=1,
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        section_text = payload["blocks"][1]["text"]["text"]
        assert section_text == response

    @pytest.mark.asyncio
    async def test_response_exactly_2801(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Responses of exactly 2801 chars are truncated."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        response = "C" * 2801

        await slack_handler._post_agent_response(
            response_url="https://hooks.slack.com/resp/1",
            agent="claude",
            response=response,
            round_num=1,
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        section_text = payload["blocks"][1]["text"]["text"]
        assert len(section_text) == 2803
        assert section_text.endswith("...")

    @pytest.mark.asyncio
    async def test_blocks_structure(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Blocks contain context, section, and divider in order."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_agent_response(
            response_url="https://hooks.slack.com/resp/1",
            agent="claude",
            response="Hello",
            round_num=2,
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        blocks = payload["blocks"]
        assert len(blocks) == 3
        assert blocks[0]["type"] == "context"
        assert blocks[1]["type"] == "section"
        assert blocks[2]["type"] == "divider"

    @pytest.mark.asyncio
    async def test_text_field(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Fallback text field contains agent and round info."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_agent_response(
            response_url="https://hooks.slack.com/resp/1",
            agent="claude",
            response="text",
            round_num=3,
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        assert payload["text"] == "claude (Round 3)"

    @pytest.mark.asyncio
    async def test_response_url_payload_structure(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Response URL payload has correct in_channel structure."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_agent_response(
            response_url="https://hooks.slack.com/resp/1",
            agent="claude",
            response="text",
            round_num=1,
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        assert payload["response_type"] == "in_channel"
        assert payload["replace_original"] is False

    @pytest.mark.asyncio
    async def test_web_api_passes_thread_ts(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Web API call passes thread_ts."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", "xoxb-token")
        slack_handler._post_message_async = AsyncMock()

        await slack_handler._post_agent_response(
            response_url="https://hooks.slack.com/resp/1",
            agent="claude",
            response="text",
            round_num=1,
            channel_id="C123",
            thread_ts="9999.8888",
        )

        call_kwargs = slack_handler._post_message_async.call_args[1]
        assert call_kwargs["thread_ts"] == "9999.8888"
        assert call_kwargs["channel"] == "C123"

    @pytest.mark.asyncio
    async def test_context_block_agent_and_round(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Context block shows agent name and round number."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_agent_response(
            response_url="https://hooks.slack.com/resp/1",
            agent="gemini",
            response="text",
            round_num=4,
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        context_text = payload["blocks"][0]["elements"][0]["text"]
        assert "*gemini*" in context_text
        assert "Round 4" in context_text


# ===========================================================================
# _build_result_blocks
# ===========================================================================


class TestBuildResultBlocks:
    """Tests for BlocksMixin._build_result_blocks."""

    def test_consensus_reached(self, slack_handler):
        """Consensus reached uses check mark emoji."""
        result = _make_debate_result(consensus_reached=True)
        blocks = slack_handler._build_result_blocks(
            topic="Topic", result=result, user_id="U1"
        )
        header = blocks[1]
        assert ":white_check_mark:" in header["text"]["text"]

    def test_no_consensus(self, slack_handler):
        """No consensus uses warning emoji."""
        result = _make_debate_result(consensus_reached=False)
        blocks = slack_handler._build_result_blocks(
            topic="Topic", result=result, user_id="U1"
        )
        header = blocks[1]
        assert ":warning:" in header["text"]["text"]

    def test_consensus_status_text(self, slack_handler):
        """Status text matches consensus state."""
        result_yes = _make_debate_result(consensus_reached=True)
        result_no = _make_debate_result(consensus_reached=False)

        blocks_yes = slack_handler._build_result_blocks(
            topic="T", result=result_yes, user_id="U1"
        )
        blocks_no = slack_handler._build_result_blocks(
            topic="T", result=result_no, user_id="U1"
        )

        fields_yes = blocks_yes[2]["fields"]
        fields_no = blocks_no[2]["fields"]

        assert "Consensus Reached" in fields_yes[0]["text"]
        assert "No Consensus" in fields_no[0]["text"]

    @pytest.mark.parametrize(
        "confidence,expected_filled",
        [
            (0.0, 0),
            (0.2, 1),
            (0.5, 2),
            (0.8, 4),
            (1.0, 5),
        ],
    )
    def test_confidence_visualization(
        self, slack_handler, confidence, expected_filled
    ):
        """Confidence bar maps correctly to filled circles."""
        result = _make_debate_result(confidence=confidence)
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        confidence_field = blocks[2]["fields"][1]["text"]
        assert confidence_field.count(":large_blue_circle:") == expected_filled
        assert confidence_field.count(":white_circle:") == 5 - expected_filled

    def test_confidence_percentage_display(self, slack_handler):
        """Confidence percentage is displayed."""
        result = _make_debate_result(confidence=0.75)
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        confidence_field = blocks[2]["fields"][1]["text"]
        assert "75%" in confidence_field

    def test_rounds_used_field(self, slack_handler):
        """Rounds field shows correct number."""
        result = _make_debate_result(rounds_used=7)
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        rounds_field = blocks[2]["fields"][2]["text"]
        assert "7" in rounds_field

    def test_participants_up_to_four(self, slack_handler):
        """Shows up to 4 participant names."""
        result = _make_debate_result(
            participants=["a", "b", "c", "d"]
        )
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        participants_field = blocks[2]["fields"][3]["text"]
        assert "a, b, c, d" in participants_field

    def test_participants_more_than_four(self, slack_handler):
        """Shows 4 names and +N for extras."""
        result = _make_debate_result(
            participants=["a", "b", "c", "d", "e", "f"]
        )
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        participants_field = blocks[2]["fields"][3]["text"]
        assert "a, b, c, d" in participants_field
        assert "+2" in participants_field

    def test_participants_fewer_than_four(self, slack_handler):
        """Fewer than 4 participants shown without +N."""
        result = _make_debate_result(participants=["claude", "gpt"])
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        participants_field = blocks[2]["fields"][3]["text"]
        assert "claude, gpt" in participants_field
        assert "+" not in participants_field

    def test_winner_present(self, slack_handler):
        """Winner block added when result has a winner."""
        result = _make_debate_result(winner="claude")
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        # Find the winner block
        winner_blocks = [
            b
            for b in blocks
            if b.get("type") == "section"
            and ":trophy:" in b.get("text", {}).get("text", "")
        ]
        assert len(winner_blocks) == 1
        assert "claude" in winner_blocks[0]["text"]["text"]

    def test_no_winner(self, slack_handler):
        """No winner block when winner is None."""
        result = _make_debate_result(winner=None)
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        winner_blocks = [
            b
            for b in blocks
            if b.get("type") == "section"
            and ":trophy:" in b.get("text", {}).get("text", "")
        ]
        assert len(winner_blocks) == 0

    def test_final_answer_displayed(self, slack_handler):
        """Final answer appears in result blocks."""
        result = _make_debate_result(final_answer="Use Kubernetes for orchestration.")
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        answer_blocks = [
            b
            for b in blocks
            if b.get("type") == "section"
            and "Answer:" in b.get("text", {}).get("text", "")
        ]
        assert len(answer_blocks) == 1
        assert "Use Kubernetes" in answer_blocks[0]["text"]["text"]

    def test_final_answer_truncated_at_500(self, slack_handler):
        """Long final answers are truncated at 500 characters."""
        long_answer = "Z" * 600
        result = _make_debate_result(final_answer=long_answer)
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        answer_blocks = [
            b
            for b in blocks
            if b.get("type") == "section"
            and "Answer:" in b.get("text", {}).get("text", "")
        ]
        answer_text = answer_blocks[0]["text"]["text"]
        # The answer part: "*Answer:*\n" + truncated content
        # The content part is long_answer[:500]
        assert "Z" * 500 in answer_text
        assert "Z" * 501 not in answer_text

    def test_no_final_answer(self, slack_handler):
        """None final_answer shows 'No conclusion reached'."""
        result = _make_debate_result(final_answer=None)
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        answer_blocks = [
            b
            for b in blocks
            if b.get("type") == "section"
            and "Answer:" in b.get("text", {}).get("text", "")
        ]
        assert "No conclusion reached" in answer_blocks[0]["text"]["text"]

    def test_empty_final_answer(self, slack_handler):
        """Empty string final_answer shows 'No conclusion reached'."""
        result = _make_debate_result(final_answer="")
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        answer_blocks = [
            b
            for b in blocks
            if b.get("type") == "section"
            and "Answer:" in b.get("text", {}).get("text", "")
        ]
        assert "No conclusion reached" in answer_blocks[0]["text"]["text"]

    def test_action_buttons_agree_disagree_details(self, slack_handler):
        """Actions block has Agree, Disagree, Details buttons."""
        result = _make_debate_result(debate_id="d-42")
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        actions_blocks = [b for b in blocks if b.get("type") == "actions"]
        assert len(actions_blocks) == 1
        elements = actions_blocks[0]["elements"]
        assert len(elements) == 3  # No receipt URL -> no 4th button

        button_texts = [e["text"]["text"] for e in elements]
        assert "Agree" in button_texts
        assert "Disagree" in button_texts
        assert "Details" in button_texts

    def test_agree_button_action_id(self, slack_handler):
        """Agree button has correct action_id pattern."""
        result = _make_debate_result(debate_id="d-42")
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        actions = [b for b in blocks if b.get("type") == "actions"][0]
        agree_btn = [e for e in actions["elements"] if e["text"]["text"] == "Agree"][0]
        assert agree_btn["action_id"] == "vote_d-42_agree"
        assert agree_btn["value"] == "d-42"
        assert agree_btn["style"] == "primary"

    def test_disagree_button_action_id(self, slack_handler):
        """Disagree button has correct action_id pattern."""
        result = _make_debate_result(debate_id="d-42")
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        actions = [b for b in blocks if b.get("type") == "actions"][0]
        disagree_btn = [
            e for e in actions["elements"] if e["text"]["text"] == "Disagree"
        ][0]
        assert disagree_btn["action_id"] == "vote_d-42_disagree"
        assert disagree_btn["value"] == "d-42"
        assert "style" not in disagree_btn  # No style for disagree

    def test_details_button_action_id(self, slack_handler):
        """Details button has view_details action_id."""
        result = _make_debate_result(debate_id="d-42")
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        actions = [b for b in blocks if b.get("type") == "actions"][0]
        details_btn = [
            e for e in actions["elements"] if e["text"]["text"] == "Details"
        ][0]
        assert details_btn["action_id"] == "view_details"
        assert details_btn["value"] == "d-42"

    def test_receipt_url_button(self, slack_handler):
        """Receipt URL button appears when receipt_url is provided."""
        result = _make_debate_result(debate_id="d-42")
        blocks = slack_handler._build_result_blocks(
            topic="T",
            result=result,
            user_id="U1",
            receipt_url="https://example.com/receipt/d-42",
        )
        actions = [b for b in blocks if b.get("type") == "actions"][0]
        assert len(actions["elements"]) == 4
        receipt_btn = actions["elements"][3]
        assert receipt_btn["text"]["text"] == "View Receipt"
        assert receipt_btn["url"] == "https://example.com/receipt/d-42"
        assert receipt_btn["action_id"] == "receipt_d-42"

    def test_no_receipt_url_button(self, slack_handler):
        """No receipt button when receipt_url is None."""
        result = _make_debate_result(debate_id="d-42")
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1", receipt_url=None
        )
        actions = [b for b in blocks if b.get("type") == "actions"][0]
        assert len(actions["elements"]) == 3

    def test_context_block_at_end(self, slack_handler):
        """Context block with debate ID and user is last block."""
        result = _make_debate_result(debate_id="d-99")
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U42"
        )
        last_block = blocks[-1]
        assert last_block["type"] == "context"
        text = last_block["elements"][0]["text"]
        assert "`d-99`" in text
        assert "<@U42>" in text

    def test_divider_at_start(self, slack_handler):
        """First block is a divider."""
        result = _make_debate_result()
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        assert blocks[0]["type"] == "divider"

    def test_header_debate_complete(self, slack_handler):
        """Header says 'Debate Complete'."""
        result = _make_debate_result()
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        header = blocks[1]
        assert "Debate Complete" in header["text"]["text"]

    def test_fields_section_structure(self, slack_handler):
        """Fields section has exactly 4 fields."""
        result = _make_debate_result()
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        fields_section = blocks[2]
        assert fields_section["type"] == "section"
        assert len(fields_section["fields"]) == 4

    def test_total_block_count_without_winner(self, slack_handler):
        """Without winner: divider, header, fields, answer, actions, context = 6."""
        result = _make_debate_result(winner=None)
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        assert len(blocks) == 6

    def test_total_block_count_with_winner(self, slack_handler):
        """With winner: divider, header, fields, winner, answer, actions, context = 7."""
        result = _make_debate_result(winner="claude")
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        assert len(blocks) == 7

    def test_confidence_zero(self, slack_handler):
        """Zero confidence = 0 filled circles, 5 empty."""
        result = _make_debate_result(confidence=0.0)
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        conf_field = blocks[2]["fields"][1]["text"]
        assert conf_field.count(":large_blue_circle:") == 0
        assert conf_field.count(":white_circle:") == 5

    def test_confidence_full(self, slack_handler):
        """Full confidence = 5 filled circles, 0 empty."""
        result = _make_debate_result(confidence=1.0)
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        conf_field = blocks[2]["fields"][1]["text"]
        assert conf_field.count(":large_blue_circle:") == 5
        assert conf_field.count(":white_circle:") == 0

    def test_single_participant(self, slack_handler):
        """Single participant shown without +N."""
        result = _make_debate_result(participants=["solo"])
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        pf = blocks[2]["fields"][3]["text"]
        assert "solo" in pf
        assert "+" not in pf

    def test_empty_participants_list(self, slack_handler):
        """Empty participants list doesn't crash."""
        result = _make_debate_result(participants=[])
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        pf = blocks[2]["fields"][3]["text"]
        assert "*Participants:*" in pf

    def test_special_chars_in_answer(self, slack_handler):
        """Special characters in answer are preserved."""
        result = _make_debate_result(final_answer="<b>bold</b> & 'quotes' \"double\"")
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        answer_blocks = [
            b
            for b in blocks
            if b.get("type") == "section"
            and "Answer:" in b.get("text", {}).get("text", "")
        ]
        assert "<b>bold</b>" in answer_blocks[0]["text"]["text"]

    def test_winner_with_special_chars(self, slack_handler):
        """Winner name with special characters is preserved."""
        result = _make_debate_result(winner="<script>alert(1)</script>")
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        winner_blocks = [
            b
            for b in blocks
            if b.get("type") == "section"
            and ":trophy:" in b.get("text", {}).get("text", "")
        ]
        assert "<script>" in winner_blocks[0]["text"]["text"]


# ===========================================================================
# Integration-style tests
# ===========================================================================


class TestBlocksIntegration:
    """Integration-style tests combining multiple BlocksMixin methods."""

    def test_starting_then_result_blocks(self, slack_handler):
        """Can build starting blocks then result blocks sequentially."""
        starting = slack_handler._build_starting_blocks(
            topic="Python vs Rust",
            user_id="U1",
            debate_id="d-1",
            agents=["claude", "gpt-4"],
            expected_rounds=5,
        )
        assert len(starting) == 3

        result = _make_debate_result(
            debate_id="d-1",
            consensus_reached=True,
            confidence=0.9,
            participants=["claude", "gpt-4"],
            rounds_used=5,
            final_answer="Use Rust for performance-critical paths.",
            winner="claude",
        )
        result_blocks = slack_handler._build_result_blocks(
            topic="Python vs Rust",
            result=result,
            user_id="U1",
            receipt_url="https://example.com/receipt/d-1",
        )
        assert len(result_blocks) == 7  # With winner

    @pytest.mark.asyncio
    async def test_post_round_then_agent_response(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Can post a round update then agent response in sequence."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_round_update(
            response_url="https://hooks.slack.com/resp/1",
            topic="Topic",
            round_num=1,
            total_rounds=3,
            agent="claude",
        )
        await slack_handler._post_agent_response(
            response_url="https://hooks.slack.com/resp/1",
            agent="claude",
            response="My analysis...",
            round_num=1,
        )

        assert slack_handler._post_to_response_url.await_count == 2

    @pytest.mark.asyncio
    async def test_multiple_round_updates(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Multiple round updates with increasing round numbers."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        for i in range(1, 4):
            await slack_handler._post_round_update(
                response_url="https://hooks.slack.com/resp/1",
                topic="Topic",
                round_num=i,
                total_rounds=3,
                agent=f"agent-{i}",
                phase="analyzing" if i < 3 else "complete",
            )

        assert slack_handler._post_to_response_url.await_count == 3

        # Check last call has all squares filled
        last_payload = slack_handler._post_to_response_url.call_args_list[-1][0][1]
        block_text = last_payload["blocks"][0]["text"]["text"]
        assert ":white_check_mark:" in block_text
        assert block_text.count(":black_large_square:") == 3

    def test_result_blocks_all_fields_mrkdwn(self, slack_handler):
        """All field blocks use mrkdwn type."""
        result = _make_debate_result()
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        fields = blocks[2]["fields"]
        for field in fields:
            assert field["type"] == "mrkdwn"

    def test_result_blocks_five_exactly_participants(self, slack_handler):
        """5 participants: show 4 + '+1'."""
        result = _make_debate_result(participants=["a", "b", "c", "d", "e"])
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        pf = blocks[2]["fields"][3]["text"]
        assert "a, b, c, d" in pf
        assert "+1" in pf

    def test_confidence_percentage_format(self, slack_handler):
        """Confidence shown as whole percentage."""
        result = _make_debate_result(confidence=0.333)
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        conf_field = blocks[2]["fields"][1]["text"]
        # 0.333 formatted as 33%
        assert "33%" in conf_field

    def test_confidence_100_percent(self, slack_handler):
        """100% confidence shows correctly."""
        result = _make_debate_result(confidence=1.0)
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        conf_field = blocks[2]["fields"][1]["text"]
        assert "100%" in conf_field

    def test_confidence_0_percent(self, slack_handler):
        """0% confidence shows correctly."""
        result = _make_debate_result(confidence=0.0)
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        conf_field = blocks[2]["fields"][1]["text"]
        assert "0%" in conf_field


# ===========================================================================
# Edge cases and security
# ===========================================================================


class TestEdgeCasesAndSecurity:
    """Edge cases and security-related tests."""

    def test_unicode_in_topic(self, slack_handler):
        """Unicode characters in topic are preserved."""
        blocks = slack_handler._build_starting_blocks(
            topic="Should we use emojis? \U0001F914\U0001F4A1",
            user_id="U1",
            debate_id="d-1",
        )
        assert "\U0001F914" in blocks[1]["text"]["text"]

    def test_newlines_in_topic(self, slack_handler):
        """Newlines in topic are preserved."""
        blocks = slack_handler._build_starting_blocks(
            topic="Line 1\nLine 2\nLine 3",
            user_id="U1",
            debate_id="d-1",
        )
        assert "\n" in blocks[1]["text"]["text"]

    def test_mrkdwn_injection_in_user_id(self, slack_handler):
        """mrkdwn in user_id is passed through (Slack handles escaping)."""
        blocks = slack_handler._build_starting_blocks(
            topic="t",
            user_id="*bold*_italic_~strike~",
            debate_id="d-1",
        )
        text = blocks[2]["elements"][0]["text"]
        assert "*bold*" in text

    def test_empty_topic(self, slack_handler):
        """Empty topic string does not crash."""
        blocks = slack_handler._build_starting_blocks(
            topic="", user_id="U1", debate_id="d-1"
        )
        assert blocks[1]["text"]["text"] == "*Topic:* "

    def test_empty_debate_id(self, slack_handler):
        """Empty debate_id does not crash."""
        blocks = slack_handler._build_starting_blocks(
            topic="t", user_id="U1", debate_id=""
        )
        assert "``" in blocks[2]["elements"][0]["text"]

    @pytest.mark.asyncio
    async def test_empty_response_text(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Empty response text does not crash agent response."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_agent_response(
            response_url="https://hooks.slack.com/resp/1",
            agent="claude",
            response="",
            round_num=1,
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        assert payload["blocks"][1]["text"]["text"] == ""

    def test_winner_empty_string(self, slack_handler):
        """Empty string winner is falsy, so no winner block."""
        result = _make_debate_result(winner="")
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        winner_blocks = [
            b
            for b in blocks
            if b.get("type") == "section"
            and ":trophy:" in b.get("text", {}).get("text", "")
        ]
        assert len(winner_blocks) == 0

    def test_all_blocks_are_valid_types(self, slack_handler):
        """All blocks have valid Slack block types."""
        valid_types = {"header", "section", "context", "divider", "actions", "image"}
        result = _make_debate_result(winner="claude")
        blocks = slack_handler._build_result_blocks(
            topic="T",
            result=result,
            user_id="U1",
            receipt_url="https://example.com/receipt",
        )
        for block in blocks:
            assert block["type"] in valid_types, f"Invalid block type: {block['type']}"

    def test_agent_response_preserves_markdown(self, slack_handler, blocks_module, monkeypatch):
        """Markdown formatting in agent responses is preserved."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        import asyncio

        asyncio.run(
            slack_handler._post_agent_response(
                response_url="https://hooks.slack.com/resp/1",
                agent="claude",
                response="*bold* _italic_ `code`",
                round_num=1,
            )
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        assert "*bold*" in payload["blocks"][1]["text"]["text"]
        assert "`code`" in payload["blocks"][1]["text"]["text"]

    def test_result_blocks_answer_exactly_500_chars(self, slack_handler):
        """Answer of exactly 500 chars is NOT truncated."""
        answer = "X" * 500
        result = _make_debate_result(final_answer=answer)
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        answer_blocks = [
            b
            for b in blocks
            if b.get("type") == "section"
            and "Answer:" in b.get("text", {}).get("text", "")
        ]
        answer_text = answer_blocks[0]["text"]["text"]
        assert "X" * 500 in answer_text

    def test_result_blocks_answer_501_chars_truncated(self, slack_handler):
        """Answer of 501 chars IS truncated."""
        answer = "Y" * 501
        result = _make_debate_result(final_answer=answer)
        blocks = slack_handler._build_result_blocks(
            topic="T", result=result, user_id="U1"
        )
        answer_blocks = [
            b
            for b in blocks
            if b.get("type") == "section"
            and "Answer:" in b.get("text", {}).get("text", "")
        ]
        answer_text = answer_blocks[0]["text"]["text"]
        # Only first 500 Y's in the answer portion
        assert "Y" * 500 in answer_text
        assert "Y" * 501 not in answer_text

    @pytest.mark.asyncio
    async def test_round_update_single_round(
        self, slack_handler, blocks_module, monkeypatch
    ):
        """Single round debate (1/1) works correctly."""
        monkeypatch.setattr(blocks_module, "SLACK_BOT_TOKEN", None)
        slack_handler._post_to_response_url = AsyncMock()

        await slack_handler._post_round_update(
            response_url="https://hooks.slack.com/resp/1",
            topic="T",
            round_num=1,
            total_rounds=1,
            agent="claude",
            phase="complete",
        )

        payload = slack_handler._post_to_response_url.call_args[0][1]
        block_text = payload["blocks"][0]["text"]["text"]
        assert "Round 1/1" in block_text
        assert ":black_large_square:" in block_text
