"""Comprehensive tests for Slack Block Kit builders.

Covers all public functions in aragora.server.handlers.social.slack.blocks.builders:
- build_starting_blocks: Debate start announcement blocks
- build_round_update_blocks: Round progress updates with visual progress bar
- build_agent_response_blocks: Individual agent response blocks with emoji mapping
- build_result_blocks: Final debate result blocks with actions
- build_gauntlet_result_blocks: Gauntlet stress-test result blocks
- build_search_result_blocks: Search results display blocks
- AGENT_EMOJIS: Agent-to-emoji mapping constant
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from aragora.server.handlers.social.slack.blocks.builders import (
    AGENT_EMOJIS,
    build_agent_response_blocks,
    build_gauntlet_result_blocks,
    build_result_blocks,
    build_round_update_blocks,
    build_search_result_blocks,
    build_starting_blocks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_debate_result(
    consensus_reached: bool = True,
    confidence: float = 0.8,
    participants: list[str] | None = None,
    rounds_used: int = 3,
    conclusion: str | None = "The consensus was reached.",
) -> SimpleNamespace:
    """Create a mock debate result object."""
    return SimpleNamespace(
        consensus_reached=consensus_reached,
        confidence=confidence,
        participants=participants or ["agent-a", "agent-b", "agent-c"],
        rounds_used=rounds_used,
        conclusion=conclusion,
    )


def _make_gauntlet_result(
    verdict: str = "valid",
    confidence: float = 0.9,
    findings: list[str] | None = None,
) -> SimpleNamespace:
    """Create a mock gauntlet result object."""
    return SimpleNamespace(
        verdict=verdict,
        confidence=confidence,
        findings=findings or [],
    )


def _find_block(blocks: list[dict[str, Any]], block_type: str) -> dict[str, Any] | None:
    """Find the first block of a given type."""
    for block in blocks:
        if block.get("type") == block_type:
            return block
    return None


def _find_all_blocks(blocks: list[dict[str, Any]], block_type: str) -> list[dict[str, Any]]:
    """Find all blocks of a given type."""
    return [b for b in blocks if b.get("type") == block_type]


# ===========================================================================
# build_starting_blocks
# ===========================================================================


class TestBuildStartingBlocks:
    """Tests for build_starting_blocks()."""

    def test_basic_structure(self):
        """Blocks contain header, topic section, and context."""
        blocks = build_starting_blocks(
            topic="Rate limiter design",
            user_id="U001",
            debate_id="d-123",
        )
        assert isinstance(blocks, list)
        assert len(blocks) >= 3

        # Header
        header = blocks[0]
        assert header["type"] == "header"
        assert header["text"]["type"] == "plain_text"
        assert "Debate Starting" in header["text"]["text"]

        # Topic section
        section = blocks[1]
        assert section["type"] == "section"
        assert "Rate limiter design" in section["text"]["text"]

        # Context
        context = blocks[2]
        assert context["type"] == "context"

    def test_user_id_in_context(self):
        """Context references the requesting user via Slack mention."""
        blocks = build_starting_blocks(
            topic="Test", user_id="U999", debate_id="d-abc"
        )
        context = blocks[-1]
        context_text = context["elements"][0]["text"]
        assert "<@U999>" in context_text

    def test_debate_id_in_context(self):
        """Context includes the debate ID in code format."""
        blocks = build_starting_blocks(
            topic="Test", user_id="U001", debate_id="d-xyz-789"
        )
        context = blocks[-1]
        context_text = context["elements"][0]["text"]
        assert "`d-xyz-789`" in context_text

    def test_agents_included(self):
        """Agents list is appended to context when provided."""
        blocks = build_starting_blocks(
            topic="Test",
            user_id="U001",
            debate_id="d-1",
            agents=["claude", "gpt-4", "gemini"],
        )
        context_text = blocks[-1]["elements"][0]["text"]
        assert "claude" in context_text
        assert "gpt-4" in context_text
        assert "gemini" in context_text

    def test_no_agents(self):
        """No agent info when agents=None."""
        blocks = build_starting_blocks(
            topic="Test", user_id="U001", debate_id="d-1", agents=None
        )
        context_text = blocks[-1]["elements"][0]["text"]
        assert "Agents:" not in context_text

    def test_empty_agents_list(self):
        """Empty agents list is treated as falsy, omitted from context."""
        blocks = build_starting_blocks(
            topic="Test", user_id="U001", debate_id="d-1", agents=[]
        )
        context_text = blocks[-1]["elements"][0]["text"]
        assert "Agents:" not in context_text

    def test_expected_rounds_included(self):
        """Rounds info is included when expected_rounds is provided."""
        blocks = build_starting_blocks(
            topic="Test", user_id="U001", debate_id="d-1", expected_rounds=5
        )
        context_text = blocks[-1]["elements"][0]["text"]
        assert "Rounds: 5" in context_text

    def test_no_expected_rounds(self):
        """Rounds info omitted when expected_rounds is None."""
        blocks = build_starting_blocks(
            topic="Test", user_id="U001", debate_id="d-1", expected_rounds=None
        )
        context_text = blocks[-1]["elements"][0]["text"]
        assert "Rounds:" not in context_text

    def test_zero_expected_rounds_omitted(self):
        """Zero rounds is falsy, omitted from context."""
        blocks = build_starting_blocks(
            topic="Test", user_id="U001", debate_id="d-1", expected_rounds=0
        )
        context_text = blocks[-1]["elements"][0]["text"]
        assert "Rounds:" not in context_text

    def test_all_optional_params(self):
        """All optional params appear in context when provided."""
        blocks = build_starting_blocks(
            topic="Full test",
            user_id="U111",
            debate_id="d-full",
            agents=["a1", "a2"],
            expected_rounds=3,
        )
        context_text = blocks[-1]["elements"][0]["text"]
        assert "<@U111>" in context_text
        assert "`d-full`" in context_text
        assert "a1" in context_text
        assert "Rounds: 3" in context_text

    def test_context_parts_joined_by_pipe(self):
        """Multiple context parts are separated by ' | '."""
        blocks = build_starting_blocks(
            topic="Test",
            user_id="U001",
            debate_id="d-1",
            agents=["a1"],
            expected_rounds=2,
        )
        context_text = blocks[-1]["elements"][0]["text"]
        # Should have at least 3 pipe separators (user, agents, rounds)
        assert context_text.count(" | ") >= 2

    def test_special_characters_in_topic(self):
        """Special characters in topic are preserved (no escaping)."""
        blocks = build_starting_blocks(
            topic="Should we use <script> tags & 'quotes'?",
            user_id="U001",
            debate_id="d-1",
        )
        assert "<script>" in blocks[1]["text"]["text"]
        assert "&" in blocks[1]["text"]["text"]


# ===========================================================================
# build_round_update_blocks
# ===========================================================================


class TestBuildRoundUpdateBlocks:
    """Tests for build_round_update_blocks()."""

    def test_basic_structure(self):
        """Returns a list with a single section block."""
        blocks = build_round_update_blocks(
            round_num=2, total_rounds=5, agent="claude"
        )
        assert isinstance(blocks, list)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "section"

    def test_round_numbers_in_text(self):
        """Round number and total are displayed in the block."""
        blocks = build_round_update_blocks(
            round_num=3, total_rounds=7, agent="gpt-4"
        )
        text = blocks[0]["text"]["text"]
        assert "Round 3/7" in text

    def test_agent_name_in_text(self):
        """Agent name appears in the response text."""
        blocks = build_round_update_blocks(
            round_num=1, total_rounds=3, agent="gemini-pro"
        )
        text = blocks[0]["text"]["text"]
        assert "gemini-pro responded" in text

    def test_progress_bar_visualization(self):
        """Progress bar shows correct filled/empty squares."""
        blocks = build_round_update_blocks(
            round_num=2, total_rounds=5, agent="test"
        )
        text = blocks[0]["text"]["text"]
        assert ":black_large_square:" * 2 in text
        assert ":white_large_square:" * 3 in text

    def test_progress_bar_complete(self):
        """Full progress bar when round_num == total_rounds."""
        blocks = build_round_update_blocks(
            round_num=4, total_rounds=4, agent="test"
        )
        text = blocks[0]["text"]["text"]
        assert ":black_large_square:" * 4 in text
        assert ":white_large_square:" not in text

    def test_progress_bar_start(self):
        """Mostly empty progress bar at round 1."""
        blocks = build_round_update_blocks(
            round_num=1, total_rounds=10, agent="test"
        )
        text = blocks[0]["text"]["text"]
        assert ":black_large_square:" in text
        assert text.count(":white_large_square:") == 9

    def test_phase_analyzing(self):
        """Analyzing phase uses magnifying glass emoji."""
        blocks = build_round_update_blocks(
            round_num=1, total_rounds=3, agent="test", phase="analyzing"
        )
        text = blocks[0]["text"]["text"]
        assert ":mag:" in text

    def test_phase_critique(self):
        """Critique phase uses speech balloon emoji."""
        blocks = build_round_update_blocks(
            round_num=1, total_rounds=3, agent="test", phase="critique"
        )
        text = blocks[0]["text"]["text"]
        assert ":speech_balloon:" in text

    def test_phase_voting(self):
        """Voting phase uses ballot box emoji."""
        blocks = build_round_update_blocks(
            round_num=1, total_rounds=3, agent="test", phase="voting"
        )
        text = blocks[0]["text"]["text"]
        assert ":ballot_box:" in text

    def test_phase_complete(self):
        """Complete phase uses check mark emoji."""
        blocks = build_round_update_blocks(
            round_num=3, total_rounds=3, agent="test", phase="complete"
        )
        text = blocks[0]["text"]["text"]
        assert ":white_check_mark:" in text

    def test_unknown_phase_default_emoji(self):
        """Unknown phase falls back to hourglass emoji."""
        blocks = build_round_update_blocks(
            round_num=1, total_rounds=3, agent="test", phase="custom_phase"
        )
        text = blocks[0]["text"]["text"]
        assert ":hourglass_flowing_sand:" in text

    def test_default_phase_is_analyzing(self):
        """Default phase parameter is 'analyzing'."""
        blocks = build_round_update_blocks(
            round_num=1, total_rounds=3, agent="test"
        )
        text = blocks[0]["text"]["text"]
        assert ":mag:" in text


# ===========================================================================
# build_agent_response_blocks
# ===========================================================================


class TestBuildAgentResponseBlocks:
    """Tests for build_agent_response_blocks()."""

    def test_basic_structure(self):
        """Returns context, section, and divider blocks."""
        blocks = build_agent_response_blocks(
            agent="claude", response="Hello world", round_num=1
        )
        assert len(blocks) == 3
        assert blocks[0]["type"] == "context"
        assert blocks[1]["type"] == "section"
        assert blocks[2]["type"] == "divider"

    def test_agent_name_in_context(self):
        """Agent name appears in the context header."""
        blocks = build_agent_response_blocks(
            agent="gpt-4", response="test", round_num=2
        )
        context_text = blocks[0]["elements"][0]["text"]
        assert "gpt-4" in context_text

    def test_round_num_in_context(self):
        """Round number appears in the context header."""
        blocks = build_agent_response_blocks(
            agent="claude", response="test", round_num=5
        )
        context_text = blocks[0]["elements"][0]["text"]
        assert "Round 5" in context_text

    def test_response_in_section(self):
        """Response text appears in the section body."""
        blocks = build_agent_response_blocks(
            agent="claude", response="This is my analysis.", round_num=1
        )
        section_text = blocks[1]["text"]["text"]
        assert "This is my analysis." in section_text

    def test_known_agent_emoji_anthropic(self):
        """Anthropic API agent gets robot face emoji."""
        blocks = build_agent_response_blocks(
            agent="anthropic-api", response="test", round_num=1
        )
        context_text = blocks[0]["elements"][0]["text"]
        assert ":robot_face:" in context_text

    def test_known_agent_emoji_openai(self):
        """OpenAI API agent gets brain emoji."""
        blocks = build_agent_response_blocks(
            agent="openai-api", response="test", round_num=1
        )
        context_text = blocks[0]["elements"][0]["text"]
        assert ":brain:" in context_text

    def test_known_agent_emoji_gemini(self):
        """Gemini agent gets gem emoji."""
        blocks = build_agent_response_blocks(
            agent="gemini", response="test", round_num=1
        )
        context_text = blocks[0]["elements"][0]["text"]
        assert ":gem:" in context_text

    def test_known_agent_emoji_grok(self):
        """Grok agent gets zap emoji."""
        blocks = build_agent_response_blocks(
            agent="grok", response="test", round_num=1
        )
        context_text = blocks[0]["elements"][0]["text"]
        assert ":zap:" in context_text

    def test_known_agent_emoji_mistral(self):
        """Mistral agent gets wind face emoji."""
        blocks = build_agent_response_blocks(
            agent="mistral", response="test", round_num=1
        )
        context_text = blocks[0]["elements"][0]["text"]
        assert ":wind_face:" in context_text

    def test_known_agent_emoji_deepseek(self):
        """DeepSeek agent gets magnifying glass emoji."""
        blocks = build_agent_response_blocks(
            agent="deepseek", response="test", round_num=1
        )
        context_text = blocks[0]["elements"][0]["text"]
        assert ":mag:" in context_text

    def test_unknown_agent_fallback_emoji(self):
        """Unknown agent gets speech balloon fallback emoji."""
        blocks = build_agent_response_blocks(
            agent="unknown-agent", response="test", round_num=1
        )
        context_text = blocks[0]["elements"][0]["text"]
        assert ":speech_balloon:" in context_text

    def test_agent_name_case_insensitive(self):
        """Agent emoji lookup is case-insensitive."""
        blocks = build_agent_response_blocks(
            agent="GEMINI", response="test", round_num=1
        )
        context_text = blocks[0]["elements"][0]["text"]
        assert ":gem:" in context_text

    def test_response_truncation_at_2800(self):
        """Long responses are truncated at 2800 characters with ellipsis."""
        long_response = "A" * 3000
        blocks = build_agent_response_blocks(
            agent="claude", response=long_response, round_num=1
        )
        section_text = blocks[1]["text"]["text"]
        assert len(section_text) == 2803  # 2800 + "..."
        assert section_text.endswith("...")

    def test_response_exactly_2800_not_truncated(self):
        """Response of exactly 2800 chars is not truncated."""
        exact_response = "B" * 2800
        blocks = build_agent_response_blocks(
            agent="claude", response=exact_response, round_num=1
        )
        section_text = blocks[1]["text"]["text"]
        assert section_text == exact_response
        assert not section_text.endswith("...")

    def test_response_2801_truncated(self):
        """Response of 2801 chars triggers truncation."""
        response = "C" * 2801
        blocks = build_agent_response_blocks(
            agent="claude", response=response, round_num=1
        )
        section_text = blocks[1]["text"]["text"]
        assert section_text.endswith("...")
        assert len(section_text) == 2803

    def test_short_response_not_truncated(self):
        """Short response is passed through as-is."""
        blocks = build_agent_response_blocks(
            agent="claude", response="Short.", round_num=1
        )
        assert blocks[1]["text"]["text"] == "Short."

    def test_empty_response(self):
        """Empty response string is handled."""
        blocks = build_agent_response_blocks(
            agent="claude", response="", round_num=1
        )
        assert blocks[1]["text"]["text"] == ""


# ===========================================================================
# AGENT_EMOJIS constant
# ===========================================================================


class TestAgentEmojis:
    """Tests for the AGENT_EMOJIS constant."""

    def test_all_known_agents_present(self):
        """All expected agents are in the mapping."""
        expected = {"anthropic-api", "openai-api", "gemini", "grok", "mistral", "deepseek"}
        assert set(AGENT_EMOJIS.keys()) == expected

    def test_all_emojis_are_strings(self):
        """All emoji values are non-empty strings."""
        for agent, emoji in AGENT_EMOJIS.items():
            assert isinstance(emoji, str), f"Emoji for {agent} is not a string"
            assert len(emoji) > 0, f"Emoji for {agent} is empty"

    def test_all_emojis_are_slack_format(self):
        """All emojis use Slack :emoji_name: format."""
        for agent, emoji in AGENT_EMOJIS.items():
            assert emoji.startswith(":"), f"Emoji for {agent} doesn't start with :"
            assert emoji.endswith(":"), f"Emoji for {agent} doesn't end with :"


# ===========================================================================
# build_result_blocks
# ===========================================================================


class TestBuildResultBlocks:
    """Tests for build_result_blocks()."""

    def test_basic_structure_with_consensus(self):
        """Consensus result has divider, header, fields section, and context."""
        result = _make_debate_result(consensus_reached=True)
        blocks = build_result_blocks(
            topic="Test topic", result=result, user_id="U001"
        )
        assert blocks[0]["type"] == "divider"
        assert blocks[1]["type"] == "header"
        assert "Debate Complete" in blocks[1]["text"]["text"]

    def test_consensus_reached_status(self):
        """Consensus reached shows check mark and 'Consensus Reached' text."""
        result = _make_debate_result(consensus_reached=True)
        blocks = build_result_blocks(
            topic="Test", result=result, user_id="U001"
        )
        header_text = blocks[1]["text"]["text"]
        assert ":white_check_mark:" in header_text

        fields = blocks[2]["fields"]
        status_field = fields[0]["text"]
        assert "Consensus Reached" in status_field

    def test_no_consensus_status(self):
        """No consensus shows warning emoji and 'No Consensus' text."""
        result = _make_debate_result(consensus_reached=False)
        blocks = build_result_blocks(
            topic="Test", result=result, user_id="U001"
        )
        header_text = blocks[1]["text"]["text"]
        assert ":warning:" in header_text

        fields = blocks[2]["fields"]
        status_field = fields[0]["text"]
        assert "No Consensus" in status_field

    def test_confidence_bar_full(self):
        """100% confidence fills all 5 circles."""
        result = _make_debate_result(confidence=1.0)
        blocks = build_result_blocks(
            topic="Test", result=result, user_id="U001"
        )
        fields = blocks[2]["fields"]
        confidence_text = fields[1]["text"]
        assert ":large_blue_circle:" * 5 in confidence_text
        assert ":white_circle:" not in confidence_text
        assert "100%" in confidence_text

    def test_confidence_bar_zero(self):
        """0% confidence shows all empty circles."""
        result = _make_debate_result(confidence=0.0)
        blocks = build_result_blocks(
            topic="Test", result=result, user_id="U001"
        )
        fields = blocks[2]["fields"]
        confidence_text = fields[1]["text"]
        assert ":white_circle:" * 5 in confidence_text
        assert ":large_blue_circle:" not in confidence_text
        assert "0%" in confidence_text

    def test_confidence_bar_partial(self):
        """60% confidence shows 3 filled, 2 empty circles."""
        result = _make_debate_result(confidence=0.6)
        blocks = build_result_blocks(
            topic="Test", result=result, user_id="U001"
        )
        fields = blocks[2]["fields"]
        confidence_text = fields[1]["text"]
        assert ":large_blue_circle:" * 3 in confidence_text
        assert ":white_circle:" * 2 in confidence_text

    def test_rounds_used_displayed(self):
        """Rounds used is shown in the fields."""
        result = _make_debate_result(rounds_used=7)
        blocks = build_result_blocks(
            topic="Test", result=result, user_id="U001"
        )
        fields = blocks[2]["fields"]
        rounds_text = fields[2]["text"]
        assert "7" in rounds_text

    def test_participants_up_to_four(self):
        """Up to 4 participants are listed by name."""
        result = _make_debate_result(participants=["a1", "a2", "a3", "a4"])
        blocks = build_result_blocks(
            topic="Test", result=result, user_id="U001"
        )
        fields = blocks[2]["fields"]
        participants_text = fields[3]["text"]
        assert "a1" in participants_text
        assert "a4" in participants_text

    def test_participants_overflow(self):
        """More than 4 participants shows +N suffix."""
        result = _make_debate_result(
            participants=["a1", "a2", "a3", "a4", "a5", "a6"]
        )
        blocks = build_result_blocks(
            topic="Test", result=result, user_id="U001"
        )
        fields = blocks[2]["fields"]
        participants_text = fields[3]["text"]
        assert "a1" in participants_text
        assert "a4" in participants_text
        assert "+2" in participants_text
        # a5 and a6 should NOT appear by name
        assert "a5" not in participants_text

    def test_empty_participants(self):
        """Empty participants list produces empty text."""
        result = _make_debate_result(participants=[])
        blocks = build_result_blocks(
            topic="Test", result=result, user_id="U001"
        )
        fields = blocks[2]["fields"]
        participants_text = fields[3]["text"]
        assert "Participants" in participants_text

    def test_conclusion_section_present(self):
        """Conclusion text is rendered as its own section."""
        result = _make_debate_result(conclusion="Final answer is 42.")
        blocks = build_result_blocks(
            topic="Test", result=result, user_id="U001"
        )
        # Find the conclusion section
        conclusion_blocks = [
            b for b in blocks
            if b["type"] == "section" and "Conclusion" in b.get("text", {}).get("text", "")
        ]
        assert len(conclusion_blocks) == 1
        assert "Final answer is 42." in conclusion_blocks[0]["text"]["text"]

    def test_conclusion_truncated_at_2800(self):
        """Long conclusion is truncated at 2800 characters."""
        long_conclusion = "X" * 3000
        result = _make_debate_result(conclusion=long_conclusion)
        blocks = build_result_blocks(
            topic="Test", result=result, user_id="U001"
        )
        conclusion_blocks = [
            b for b in blocks
            if b["type"] == "section" and "Conclusion" in b.get("text", {}).get("text", "")
        ]
        assert len(conclusion_blocks) == 1
        text = conclusion_blocks[0]["text"]["text"]
        # The text includes "*Conclusion:*\n" prefix, so check the conclusion portion
        assert "..." in text

    def test_no_conclusion(self):
        """No conclusion section when conclusion is None."""
        result = _make_debate_result(conclusion=None)
        blocks = build_result_blocks(
            topic="Test", result=result, user_id="U001"
        )
        conclusion_blocks = [
            b for b in blocks
            if b["type"] == "section" and "Conclusion" in b.get("text", {}).get("text", "")
        ]
        assert len(conclusion_blocks) == 0

    def test_empty_conclusion(self):
        """Empty string conclusion is falsy, omitted."""
        result = _make_debate_result(conclusion="")
        blocks = build_result_blocks(
            topic="Test", result=result, user_id="U001"
        )
        conclusion_blocks = [
            b for b in blocks
            if b["type"] == "section" and "Conclusion" in b.get("text", {}).get("text", "")
        ]
        assert len(conclusion_blocks) == 0

    def test_actions_with_debate_id(self):
        """Actions block with buttons appears when debate_id is provided."""
        result = _make_debate_result()
        blocks = build_result_blocks(
            topic="Test",
            result=result,
            user_id="U001",
            debate_id="d-42",
        )
        actions = _find_block(blocks, "actions")
        assert actions is not None
        elements = actions["elements"]
        # Should have view_details, vote_agree, vote_disagree (at least 3)
        assert len(elements) >= 3

    def test_view_details_button(self):
        """View Details button has correct action_id and value."""
        result = _make_debate_result()
        blocks = build_result_blocks(
            topic="Test",
            result=result,
            user_id="U001",
            debate_id="d-42",
        )
        actions = _find_block(blocks, "actions")
        details_btn = next(
            (e for e in actions["elements"] if "view_details" in e.get("action_id", "")),
            None,
        )
        assert details_btn is not None
        assert details_btn["action_id"] == "view_details_d-42"
        assert details_btn["value"] == "d-42"

    def test_receipt_button_with_url(self):
        """Receipt button appears when receipt_url is provided."""
        result = _make_debate_result()
        blocks = build_result_blocks(
            topic="Test",
            result=result,
            user_id="U001",
            debate_id="d-42",
            receipt_url="https://example.com/receipt/42",
        )
        actions = _find_block(blocks, "actions")
        receipt_btn = next(
            (e for e in actions["elements"] if "view_receipt" in e.get("action_id", "")),
            None,
        )
        assert receipt_btn is not None
        assert receipt_btn["url"] == "https://example.com/receipt/42"

    def test_no_receipt_button_without_url(self):
        """Receipt button is absent when receipt_url is None."""
        result = _make_debate_result()
        blocks = build_result_blocks(
            topic="Test",
            result=result,
            user_id="U001",
            debate_id="d-42",
            receipt_url=None,
        )
        actions = _find_block(blocks, "actions")
        receipt_btn = next(
            (e for e in actions["elements"] if "view_receipt" in e.get("action_id", "")),
            None,
        )
        assert receipt_btn is None

    def test_vote_agree_button(self):
        """Agree vote button has primary style."""
        result = _make_debate_result()
        blocks = build_result_blocks(
            topic="Test",
            result=result,
            user_id="U001",
            debate_id="d-42",
        )
        actions = _find_block(blocks, "actions")
        agree_btn = next(
            (e for e in actions["elements"] if "vote_agree" in e.get("action_id", "")),
            None,
        )
        assert agree_btn is not None
        assert agree_btn["style"] == "primary"
        assert agree_btn["value"] == "d-42"

    def test_vote_disagree_button(self):
        """Disagree vote button has danger style."""
        result = _make_debate_result()
        blocks = build_result_blocks(
            topic="Test",
            result=result,
            user_id="U001",
            debate_id="d-42",
        )
        actions = _find_block(blocks, "actions")
        disagree_btn = next(
            (e for e in actions["elements"] if "vote_disagree" in e.get("action_id", "")),
            None,
        )
        assert disagree_btn is not None
        assert disagree_btn["style"] == "danger"
        assert disagree_btn["value"] == "d-42"

    def test_no_actions_without_debate_id(self):
        """No actions block when debate_id is None."""
        result = _make_debate_result()
        blocks = build_result_blocks(
            topic="Test",
            result=result,
            user_id="U001",
            debate_id=None,
        )
        actions = _find_block(blocks, "actions")
        assert actions is None

    def test_context_footer_with_debate_id(self):
        """Context footer includes user and debate ID."""
        result = _make_debate_result()
        blocks = build_result_blocks(
            topic="Test",
            result=result,
            user_id="U001",
            debate_id="d-42",
        )
        context = blocks[-1]
        assert context["type"] == "context"
        text = context["elements"][0]["text"]
        assert "<@U001>" in text
        assert "`d-42`" in text

    def test_context_footer_without_debate_id(self):
        """Context footer has user but no debate ID when omitted."""
        result = _make_debate_result()
        blocks = build_result_blocks(
            topic="Test",
            result=result,
            user_id="U001",
            debate_id=None,
        )
        context = blocks[-1]
        text = context["elements"][0]["text"]
        assert "<@U001>" in text
        assert "ID:" not in text

    def test_fields_section_has_four_fields(self):
        """Fields section always has 4 fields: status, confidence, rounds, participants."""
        result = _make_debate_result()
        blocks = build_result_blocks(
            topic="Test", result=result, user_id="U001"
        )
        fields_section = blocks[2]
        assert fields_section["type"] == "section"
        assert len(fields_section["fields"]) == 4


# ===========================================================================
# build_gauntlet_result_blocks
# ===========================================================================


class TestBuildGauntletResultBlocks:
    """Tests for build_gauntlet_result_blocks()."""

    def test_basic_structure(self):
        """Returns divider, header, statement, verdict fields, and context."""
        result = _make_gauntlet_result()
        blocks = build_gauntlet_result_blocks(
            statement="Test statement",
            result=result,
            user_id="U001",
        )
        assert blocks[0]["type"] == "divider"
        assert blocks[1]["type"] == "header"
        assert "Gauntlet Complete" in blocks[1]["text"]["text"]

    def test_valid_verdict(self):
        """Valid verdict shows check mark and 'Statement Validated'."""
        result = _make_gauntlet_result(verdict="valid")
        blocks = build_gauntlet_result_blocks(
            statement="True statement",
            result=result,
            user_id="U001",
        )
        header_text = blocks[1]["text"]["text"]
        assert ":white_check_mark:" in header_text

        fields = blocks[3]["fields"]
        verdict_text = fields[0]["text"]
        assert "Statement Validated" in verdict_text

    def test_invalid_verdict(self):
        """Invalid verdict shows X and 'Statement Refuted'."""
        result = _make_gauntlet_result(verdict="invalid")
        blocks = build_gauntlet_result_blocks(
            statement="False statement",
            result=result,
            user_id="U001",
        )
        header_text = blocks[1]["text"]["text"]
        assert ":x:" in header_text

        fields = blocks[3]["fields"]
        verdict_text = fields[0]["text"]
        assert "Statement Refuted" in verdict_text

    def test_unknown_verdict(self):
        """Unknown verdict shows warning and 'Inconclusive'."""
        result = _make_gauntlet_result(verdict="unknown")
        blocks = build_gauntlet_result_blocks(
            statement="Ambiguous statement",
            result=result,
            user_id="U001",
        )
        header_text = blocks[1]["text"]["text"]
        assert ":warning:" in header_text

        fields = blocks[3]["fields"]
        verdict_text = fields[0]["text"]
        assert "Inconclusive" in verdict_text

    def test_missing_verdict_attribute(self):
        """Missing verdict attribute defaults to 'unknown' / 'Inconclusive'."""
        result = SimpleNamespace(confidence=0.5)  # No verdict attribute
        blocks = build_gauntlet_result_blocks(
            statement="Test",
            result=result,
            user_id="U001",
        )
        header_text = blocks[1]["text"]["text"]
        assert ":warning:" in header_text

    def test_statement_in_section(self):
        """Statement text appears in its own section."""
        result = _make_gauntlet_result()
        blocks = build_gauntlet_result_blocks(
            statement="The earth is round",
            result=result,
            user_id="U001",
        )
        statement_section = blocks[2]
        assert statement_section["type"] == "section"
        assert "The earth is round" in statement_section["text"]["text"]

    def test_confidence_displayed(self):
        """Confidence is shown as a percentage in fields."""
        result = _make_gauntlet_result(confidence=0.85)
        blocks = build_gauntlet_result_blocks(
            statement="Test",
            result=result,
            user_id="U001",
        )
        fields = blocks[3]["fields"]
        confidence_text = fields[1]["text"]
        assert "85%" in confidence_text

    def test_findings_displayed(self):
        """Findings are listed as bullet points."""
        result = _make_gauntlet_result(
            findings=["Finding 1", "Finding 2", "Finding 3"]
        )
        blocks = build_gauntlet_result_blocks(
            statement="Test",
            result=result,
            user_id="U001",
        )
        findings_blocks = [
            b for b in blocks
            if b["type"] == "section" and "Key Findings" in b.get("text", {}).get("text", "")
        ]
        assert len(findings_blocks) == 1
        text = findings_blocks[0]["text"]["text"]
        assert "Finding 1" in text
        assert "Finding 2" in text
        assert "Finding 3" in text

    def test_findings_truncated_at_five(self):
        """More than 5 findings shows first 5 plus '...and N more'."""
        result = _make_gauntlet_result(
            findings=[f"Finding {i}" for i in range(8)]
        )
        blocks = build_gauntlet_result_blocks(
            statement="Test",
            result=result,
            user_id="U001",
        )
        findings_blocks = [
            b for b in blocks
            if b["type"] == "section" and "Key Findings" in b.get("text", {}).get("text", "")
        ]
        text = findings_blocks[0]["text"]["text"]
        assert "Finding 0" in text
        assert "Finding 4" in text
        assert "Finding 5" not in text
        assert "3 more" in text

    def test_no_findings_section_when_empty(self):
        """No findings section when findings list is empty."""
        result = _make_gauntlet_result(findings=[])
        blocks = build_gauntlet_result_blocks(
            statement="Test",
            result=result,
            user_id="U001",
        )
        findings_blocks = [
            b for b in blocks
            if b["type"] == "section" and "Key Findings" in b.get("text", {}).get("text", "")
        ]
        assert len(findings_blocks) == 0

    def test_no_findings_attribute(self):
        """Missing findings attribute produces no findings section."""
        result = SimpleNamespace(verdict="valid", confidence=0.9)
        blocks = build_gauntlet_result_blocks(
            statement="Test",
            result=result,
            user_id="U001",
        )
        findings_blocks = [
            b for b in blocks
            if b["type"] == "section" and "Key Findings" in b.get("text", {}).get("text", "")
        ]
        assert len(findings_blocks) == 0

    def test_receipt_url_button(self):
        """Receipt URL produces a 'View Full Report' button."""
        result = _make_gauntlet_result()
        blocks = build_gauntlet_result_blocks(
            statement="Test",
            result=result,
            user_id="U001",
            receipt_url="https://example.com/report/1",
        )
        actions = _find_block(blocks, "actions")
        assert actions is not None
        btn = actions["elements"][0]
        assert "View Full Report" in btn["text"]["text"]
        assert btn["url"] == "https://example.com/report/1"

    def test_no_receipt_url(self):
        """No actions block when receipt_url is None."""
        result = _make_gauntlet_result()
        blocks = build_gauntlet_result_blocks(
            statement="Test",
            result=result,
            user_id="U001",
            receipt_url=None,
        )
        actions = _find_block(blocks, "actions")
        assert actions is None

    def test_context_footer(self):
        """Context footer references the requesting user."""
        result = _make_gauntlet_result()
        blocks = build_gauntlet_result_blocks(
            statement="Test",
            result=result,
            user_id="U999",
        )
        context = blocks[-1]
        assert context["type"] == "context"
        assert "<@U999>" in context["elements"][0]["text"]

    def test_zero_confidence(self):
        """Zero confidence is displayed as 0%."""
        result = _make_gauntlet_result(confidence=0.0)
        blocks = build_gauntlet_result_blocks(
            statement="Test",
            result=result,
            user_id="U001",
        )
        fields = blocks[3]["fields"]
        confidence_text = fields[1]["text"]
        assert "0%" in confidence_text

    def test_missing_confidence_attribute(self):
        """Missing confidence attribute defaults to 0."""
        result = SimpleNamespace(verdict="valid")
        blocks = build_gauntlet_result_blocks(
            statement="Test",
            result=result,
            user_id="U001",
        )
        fields = blocks[3]["fields"]
        confidence_text = fields[1]["text"]
        assert "0%" in confidence_text


# ===========================================================================
# build_search_result_blocks
# ===========================================================================


class TestBuildSearchResultBlocks:
    """Tests for build_search_result_blocks()."""

    def test_basic_structure_with_results(self):
        """Header followed by result sections."""
        results = [{"title": "Result 1", "snippet": "Snippet 1"}]
        blocks = build_search_result_blocks(
            query="test query", results=results, total=1
        )
        assert blocks[0]["type"] == "header"
        assert "test query" in blocks[0]["text"]["text"]

    def test_empty_results(self):
        """Empty results show 'No results found' message."""
        blocks = build_search_result_blocks(
            query="nothing", results=[], total=0
        )
        assert len(blocks) == 2
        assert blocks[0]["type"] == "header"
        assert blocks[1]["type"] == "section"
        assert "No results found" in blocks[1]["text"]["text"]

    def test_query_in_header(self):
        """Search query appears in the header."""
        blocks = build_search_result_blocks(
            query="rate limiting", results=[], total=0
        )
        assert "rate limiting" in blocks[0]["text"]["text"]

    def test_result_title_from_title_field(self):
        """Result title is extracted from 'title' field."""
        results = [{"title": "My Great Debate", "snippet": "A snippet"}]
        blocks = build_search_result_blocks(
            query="test", results=results, total=1
        )
        section = blocks[1]
        assert "My Great Debate" in section["text"]["text"]

    def test_result_title_from_topic_field(self):
        """Falls back to 'topic' field when 'title' is missing."""
        results = [{"topic": "Fallback Topic", "snippet": "A snippet"}]
        blocks = build_search_result_blocks(
            query="test", results=results, total=1
        )
        section = blocks[1]
        assert "Fallback Topic" in section["text"]["text"]

    def test_result_title_untitled_fallback(self):
        """Falls back to 'Untitled' when both title and topic are missing."""
        results = [{"snippet": "A snippet"}]
        blocks = build_search_result_blocks(
            query="test", results=results, total=1
        )
        section = blocks[1]
        assert "Untitled" in section["text"]["text"]

    def test_result_snippet_from_snippet_field(self):
        """Snippet is extracted from 'snippet' field."""
        results = [{"title": "T", "snippet": "This is the snippet"}]
        blocks = build_search_result_blocks(
            query="test", results=results, total=1
        )
        section = blocks[1]
        assert "This is the snippet" in section["text"]["text"]

    def test_result_snippet_from_conclusion_field(self):
        """Falls back to 'conclusion' field when 'snippet' is missing."""
        results = [{"title": "T", "conclusion": "The conclusion text"}]
        blocks = build_search_result_blocks(
            query="test", results=results, total=1
        )
        section = blocks[1]
        assert "The conclusion text" in section["text"]["text"]

    def test_snippet_truncated_at_200(self):
        """Long snippets are truncated at 200 characters."""
        results = [{"title": "T", "snippet": "A" * 300}]
        blocks = build_search_result_blocks(
            query="test", results=results, total=1
        )
        section = blocks[1]
        text = section["text"]["text"]
        # The snippet portion should be 200 chars followed by "..."
        # The full text includes "emoji *1. T*\nAAA...AAA..."
        assert "A" * 200 in text
        assert "A" * 201 not in text

    def test_result_numbering(self):
        """Results are numbered starting from 1."""
        results = [
            {"title": "First", "snippet": "s1"},
            {"title": "Second", "snippet": "s2"},
            {"title": "Third", "snippet": "s3"},
        ]
        blocks = build_search_result_blocks(
            query="test", results=results, total=3
        )
        assert "1. First" in blocks[1]["text"]["text"]
        assert "2. Second" in blocks[2]["text"]["text"]
        assert "3. Third" in blocks[3]["text"]["text"]

    def test_max_ten_results(self):
        """Only first 10 results are displayed even if more provided."""
        results = [{"title": f"R{i}", "snippet": f"S{i}"} for i in range(15)]
        blocks = build_search_result_blocks(
            query="test", results=results, total=15
        )
        # Header + 10 result sections + "showing X of Y" context
        result_sections = _find_all_blocks(blocks, "section")
        assert len(result_sections) == 10

    def test_showing_count_when_total_exceeds_ten(self):
        """Context shows 'Showing 10 of N results' when total > 10."""
        results = [{"title": f"R{i}", "snippet": f"S{i}"} for i in range(10)]
        blocks = build_search_result_blocks(
            query="test", results=results, total=25
        )
        context = blocks[-1]
        assert context["type"] == "context"
        assert "Showing 10 of 25 results" in context["elements"][0]["text"]

    def test_no_showing_count_when_ten_or_fewer(self):
        """No 'showing' context when total <= 10."""
        results = [{"title": f"R{i}", "snippet": f"S{i}"} for i in range(5)]
        blocks = build_search_result_blocks(
            query="test", results=results, total=5
        )
        # Last block should be a section (result), not a context
        assert blocks[-1]["type"] == "section"

    def test_type_emoji_debate(self):
        """Debate type results get speech balloon emoji."""
        results = [{"title": "T", "snippet": "S", "type": "debate"}]
        blocks = build_search_result_blocks(
            query="test", results=results, total=1
        )
        assert ":speech_balloon:" in blocks[1]["text"]["text"]

    def test_type_emoji_evidence(self):
        """Evidence type results get page emoji."""
        results = [{"title": "T", "snippet": "S", "type": "evidence"}]
        blocks = build_search_result_blocks(
            query="test", results=results, total=1
        )
        assert ":page_facing_up:" in blocks[1]["text"]["text"]

    def test_type_emoji_consensus(self):
        """Consensus type results get handshake emoji."""
        results = [{"title": "T", "snippet": "S", "type": "consensus"}]
        blocks = build_search_result_blocks(
            query="test", results=results, total=1
        )
        assert ":handshake:" in blocks[1]["text"]["text"]

    def test_type_emoji_unknown(self):
        """Unknown type results get magnifying glass emoji."""
        results = [{"title": "T", "snippet": "S", "type": "something_else"}]
        blocks = build_search_result_blocks(
            query="test", results=results, total=1
        )
        assert ":mag:" in blocks[1]["text"]["text"]

    def test_type_emoji_default(self):
        """Missing type defaults to 'debate' emoji."""
        results = [{"title": "T", "snippet": "S"}]
        blocks = build_search_result_blocks(
            query="test", results=results, total=1
        )
        assert ":speech_balloon:" in blocks[1]["text"]["text"]

    def test_total_defaults_to_zero(self):
        """Default total=0 does not add a 'showing' context."""
        results = [{"title": "T", "snippet": "S"}]
        blocks = build_search_result_blocks(query="test", results=results)
        # No 'showing' context since total=0 <= 10
        contexts = _find_all_blocks(blocks, "context")
        assert len(contexts) == 0

    def test_mixed_result_types(self):
        """Multiple results with different types get correct emojis."""
        results = [
            {"title": "D", "snippet": "S", "type": "debate"},
            {"title": "E", "snippet": "S", "type": "evidence"},
            {"title": "C", "snippet": "S", "type": "consensus"},
        ]
        blocks = build_search_result_blocks(
            query="test", results=results, total=3
        )
        assert ":speech_balloon:" in blocks[1]["text"]["text"]
        assert ":page_facing_up:" in blocks[2]["text"]["text"]
        assert ":handshake:" in blocks[3]["text"]["text"]


# ===========================================================================
# Integration / cross-cutting tests
# ===========================================================================


class TestBlockStructureIntegrity:
    """Cross-cutting tests ensuring all builders produce valid Slack block structures."""

    def test_all_blocks_have_type(self):
        """Every block from every builder has a 'type' key."""
        all_blocks = []
        all_blocks.extend(
            build_starting_blocks("T", "U1", "d1", ["a"], 3)
        )
        all_blocks.extend(
            build_round_update_blocks(1, 3, "agent")
        )
        all_blocks.extend(
            build_agent_response_blocks("claude", "response", 1)
        )
        all_blocks.extend(
            build_result_blocks(
                "T",
                _make_debate_result(),
                "U1",
                "https://receipt.url",
                "d1",
            )
        )
        all_blocks.extend(
            build_gauntlet_result_blocks(
                "Statement",
                _make_gauntlet_result(findings=["f1"]),
                "U1",
                "https://report.url",
            )
        )
        all_blocks.extend(
            build_search_result_blocks("q", [{"title": "T"}], 1)
        )

        for i, block in enumerate(all_blocks):
            assert "type" in block, f"Block at index {i} is missing 'type' key: {block}"

    def test_mrkdwn_text_objects_have_correct_type(self):
        """All mrkdwn text objects use type 'mrkdwn'."""
        blocks = build_starting_blocks("T", "U1", "d1")
        for block in blocks:
            if "text" in block and isinstance(block["text"], dict):
                text_type = block["text"].get("type")
                assert text_type in ("plain_text", "mrkdwn"), (
                    f"Unexpected text type: {text_type}"
                )

    def test_header_blocks_use_plain_text(self):
        """Header blocks always use plain_text type."""
        blocks = build_starting_blocks("T", "U1", "d1")
        headers = _find_all_blocks(blocks, "header")
        for header in headers:
            assert header["text"]["type"] == "plain_text"

    def test_result_blocks_return_list(self):
        """All builder functions return lists (not dicts or other types)."""
        assert isinstance(build_starting_blocks("T", "U1", "d1"), list)
        assert isinstance(build_round_update_blocks(1, 3, "a"), list)
        assert isinstance(build_agent_response_blocks("a", "r", 1), list)
        assert isinstance(
            build_result_blocks("T", _make_debate_result(), "U1"), list
        )
        assert isinstance(
            build_gauntlet_result_blocks("S", _make_gauntlet_result(), "U1"), list
        )
        assert isinstance(
            build_search_result_blocks("q", []), list
        )
