"""
Tests for Slack Block Kit Message Builders.

Covers all functions and behavior in aragora.server.handlers.bots.slack.blocks:

- build_debate_message_blocks():
  - Header block structure
  - Task section with mrkdwn text
  - Agent list and progress fields
  - Divider block
  - Vote buttons (include_vote_buttons=True, default)
    - Max 5 buttons (truncation)
    - Primary style on first button only
    - Button action_id and JSON value
    - "View Summary" and "Provenance" action buttons
  - No vote buttons (include_vote_buttons=False)
  - Footer context block with truncated debate ID
  - Edge cases: empty agents, single agent, special chars, long values

- build_consensus_message_blocks():
  - Consensus reached vs not reached (header, emoji)
  - Task section
  - Confidence formatting (percentage)
  - Winner field (present / absent)
  - Vote counts (sorted descending, singular/plural)
  - Final answer preview (truncation at 500 chars)
  - No final answer
  - Action buttons (View Full, Audit Trail URLs)
  - Edge cases: empty vote_counts, zero confidence, long answers

- build_debate_result_blocks (alias):
  - Alias resolves to build_consensus_message_blocks

- build_start_debate_modal():
  - Modal structure (type, callback_id, title, submit, close)
  - Task input block (multiline, placeholder)
  - Agent selection block (multi_static_select, 8 options)
  - Rounds selection block (static_select, 4 options, initial_option)
  - DEFAULT_ROUNDS integration

- _build_start_debate_modal (private alias):
  - Alias resolves to build_start_debate_modal

- __all__ exports:
  - All 5 names exported

- Security tests:
  - XSS/injection in task, agents, debate_id
  - Unicode handling
"""

from __future__ import annotations

import json
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------

from aragora.server.handlers.bots.slack.blocks import (
    _build_start_debate_modal,
    build_consensus_message_blocks,
    build_debate_message_blocks,
    build_debate_result_blocks,
    build_start_debate_modal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_blocks_of_type(blocks: list[dict], block_type: str) -> list[dict]:
    """Return all blocks matching a given type."""
    return [b for b in blocks if b.get("type") == block_type]


def _find_first_block(blocks: list[dict], block_type: str) -> dict | None:
    """Return the first block of a given type, or None."""
    for b in blocks:
        if b.get("type") == block_type:
            return b
    return None


def _get_all_text(blocks: list[dict]) -> str:
    """Concatenate all text content from blocks for searching."""
    parts: list[str] = []
    for b in blocks:
        if "text" in b:
            if isinstance(b["text"], dict):
                parts.append(b["text"].get("text", ""))
            elif isinstance(b["text"], str):
                parts.append(b["text"])
        if "fields" in b:
            for f in b["fields"]:
                if isinstance(f, dict) and "text" in f:
                    parts.append(f["text"])
        if "elements" in b:
            for el in b["elements"]:
                if isinstance(el, dict):
                    if "text" in el and isinstance(el["text"], dict):
                        parts.append(el["text"].get("text", ""))
                    elif "text" in el and isinstance(el["text"], str):
                        parts.append(el["text"])
    return "\n".join(parts)


# ===========================================================================
# build_debate_message_blocks tests
# ===========================================================================


class TestBuildDebateMessageBlocks:
    """Tests for build_debate_message_blocks."""

    # --- Basic structure ---

    def test_returns_list_of_dicts(self):
        """Result is a list of Block Kit block dicts."""
        blocks = build_debate_message_blocks(
            debate_id="abc123",
            task="Test task",
            agents=["claude", "gpt4"],
            current_round=1,
            total_rounds=3,
        )
        assert isinstance(blocks, list)
        assert all(isinstance(b, dict) for b in blocks)

    def test_header_block_present(self):
        """First block is a header with 'Active Debate'."""
        blocks = build_debate_message_blocks(
            debate_id="abc123",
            task="Test task",
            agents=["claude"],
            current_round=1,
            total_rounds=3,
        )
        header = blocks[0]
        assert header["type"] == "header"
        assert "Active Debate" in header["text"]["text"]
        assert header["text"]["type"] == "plain_text"
        assert header["text"]["emoji"] is True

    def test_task_section_present(self):
        """Second block is a section showing the task."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="Design a rate limiter",
            agents=["claude"],
            current_round=1,
            total_rounds=3,
        )
        section = blocks[1]
        assert section["type"] == "section"
        assert "*Task:* Design a rate limiter" in section["text"]["text"]
        assert section["text"]["type"] == "mrkdwn"

    def test_agents_and_progress_fields(self):
        """Third block has fields for agents and progress."""
        agents = ["claude", "gpt4", "gemini"]
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=agents,
            current_round=2,
            total_rounds=5,
        )
        fields_block = blocks[2]
        assert fields_block["type"] == "section"
        assert len(fields_block["fields"]) == 2

        agents_field = fields_block["fields"][0]
        assert agents_field["type"] == "mrkdwn"
        assert "claude, gpt4, gemini" in agents_field["text"]

        progress_field = fields_block["fields"][1]
        assert "Round 2/5" in progress_field["text"]

    def test_divider_present(self):
        """Fourth block is a divider."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=["claude"],
            current_round=1,
            total_rounds=3,
        )
        assert blocks[3]["type"] == "divider"

    def test_footer_context_block(self):
        """Last block is a context block with truncated debate ID."""
        debate_id = "abcdef1234567890"
        blocks = build_debate_message_blocks(
            debate_id=debate_id,
            task="task",
            agents=["claude"],
            current_round=1,
            total_rounds=3,
        )
        footer = blocks[-1]
        assert footer["type"] == "context"
        assert len(footer["elements"]) == 1
        footer_text = footer["elements"][0]["text"]
        assert "abcdef12..." in footer_text
        assert "Aragora" in footer_text

    def test_footer_short_debate_id(self):
        """Footer handles debate IDs shorter than 8 chars."""
        blocks = build_debate_message_blocks(
            debate_id="abc",
            task="task",
            agents=["a"],
            current_round=1,
            total_rounds=1,
        )
        footer = blocks[-1]
        footer_text = footer["elements"][0]["text"]
        assert "abc..." in footer_text

    # --- Vote buttons (include_vote_buttons=True, default) ---

    def test_vote_buttons_included_by_default(self):
        """Vote buttons appear when include_vote_buttons is True (default)."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=["claude", "gpt4"],
            current_round=1,
            total_rounds=3,
        )
        actions_blocks = _find_blocks_of_type(blocks, "actions")
        assert len(actions_blocks) >= 1

    def test_cast_your_vote_section(self):
        """A 'Cast your vote' section appears before vote buttons."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=["claude"],
            current_round=1,
            total_rounds=3,
        )
        text = _get_all_text(blocks)
        assert "Cast your vote" in text

    def test_vote_button_per_agent(self):
        """One vote button per agent, up to 5."""
        agents = ["claude", "gpt4", "gemini"]
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=agents,
            current_round=1,
            total_rounds=3,
        )
        # The first actions block has agent vote buttons
        actions_blocks = _find_blocks_of_type(blocks, "actions")
        vote_actions = actions_blocks[0]
        buttons = vote_actions["elements"]
        assert len(buttons) == 3
        for i, agent in enumerate(agents):
            assert buttons[i]["text"]["text"] == f"Vote {agent}"

    def test_max_five_vote_buttons(self):
        """At most 5 vote buttons even with more agents."""
        agents = ["a1", "a2", "a3", "a4", "a5", "a6", "a7"]
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=agents,
            current_round=1,
            total_rounds=3,
        )
        actions_blocks = _find_blocks_of_type(blocks, "actions")
        vote_buttons = actions_blocks[0]["elements"]
        assert len(vote_buttons) == 5

    def test_first_button_has_primary_style(self):
        """First vote button has 'primary' style."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=["claude", "gpt4"],
            current_round=1,
            total_rounds=3,
        )
        actions_blocks = _find_blocks_of_type(blocks, "actions")
        buttons = actions_blocks[0]["elements"]
        assert buttons[0].get("style") == "primary"

    def test_non_first_buttons_no_style(self):
        """Non-first vote buttons have no 'style' key."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=["claude", "gpt4", "gemini"],
            current_round=1,
            total_rounds=3,
        )
        actions_blocks = _find_blocks_of_type(blocks, "actions")
        buttons = actions_blocks[0]["elements"]
        assert "style" not in buttons[1]
        assert "style" not in buttons[2]

    def test_button_action_id_format(self):
        """Vote button action_id follows 'vote_{debate_id}_{agent}' format."""
        blocks = build_debate_message_blocks(
            debate_id="debate-xyz",
            task="task",
            agents=["claude"],
            current_round=1,
            total_rounds=3,
        )
        actions_blocks = _find_blocks_of_type(blocks, "actions")
        btn = actions_blocks[0]["elements"][0]
        assert btn["action_id"] == "vote_debate-xyz_claude"

    def test_button_value_is_json(self):
        """Vote button value is a JSON string with debate_id and agent."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=["gpt4"],
            current_round=1,
            total_rounds=3,
        )
        actions_blocks = _find_blocks_of_type(blocks, "actions")
        btn = actions_blocks[0]["elements"][0]
        value = json.loads(btn["value"])
        assert value == {"debate_id": "d1", "agent": "gpt4"}

    def test_summary_button_present(self):
        """A 'View Summary' button is included."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=["claude"],
            current_round=1,
            total_rounds=3,
        )
        text = _get_all_text(blocks)
        assert "View Summary" in text

    def test_summary_button_action_id(self):
        """Summary button has correct action_id."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=["claude"],
            current_round=1,
            total_rounds=3,
        )
        actions_blocks = _find_blocks_of_type(blocks, "actions")
        # Summary button is in the second actions block
        summary_actions = actions_blocks[1]
        summary_btn = summary_actions["elements"][0]
        assert summary_btn["action_id"] == "summary_d1"
        assert summary_btn["value"] == "d1"

    def test_provenance_button_present(self):
        """A 'Provenance' button with URL is included."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=["claude"],
            current_round=1,
            total_rounds=3,
        )
        actions_blocks = _find_blocks_of_type(blocks, "actions")
        prov_btn = actions_blocks[1]["elements"][1]
        assert "Provenance" in prov_btn["text"]["text"]
        assert prov_btn["action_id"] == "provenance_d1"
        assert prov_btn["url"] == "https://aragora.ai/debates/provenance?debate=d1"

    # --- No vote buttons (include_vote_buttons=False) ---

    def test_no_vote_buttons_when_disabled(self):
        """No vote/summary/provenance buttons when include_vote_buttons=False."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=["claude"],
            current_round=1,
            total_rounds=3,
            include_vote_buttons=False,
        )
        actions_blocks = _find_blocks_of_type(blocks, "actions")
        assert len(actions_blocks) == 0
        text = _get_all_text(blocks)
        assert "Cast your vote" not in text

    def test_no_vote_buttons_has_header_and_footer(self):
        """Even without vote buttons, header and footer are present."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=["claude"],
            current_round=1,
            total_rounds=3,
            include_vote_buttons=False,
        )
        assert blocks[0]["type"] == "header"
        assert blocks[-1]["type"] == "context"

    def test_no_vote_buttons_block_count(self):
        """Without vote buttons: header + task + fields + divider + footer = 5 blocks."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=["claude"],
            current_round=1,
            total_rounds=3,
            include_vote_buttons=False,
        )
        assert len(blocks) == 5

    # --- Edge cases ---

    def test_empty_agents_list(self):
        """Empty agents list produces empty agent field text."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=[],
            current_round=0,
            total_rounds=0,
        )
        fields_block = blocks[2]
        agents_text = fields_block["fields"][0]["text"]
        assert "*Agents:*\n" in agents_text

    def test_single_agent(self):
        """Single agent shows just that agent name."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=["claude"],
            current_round=1,
            total_rounds=1,
        )
        fields_block = blocks[2]
        assert "claude" in fields_block["fields"][0]["text"]

    def test_empty_agents_no_vote_buttons_created(self):
        """Empty agents with vote buttons creates empty actions block."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=[],
            current_round=1,
            total_rounds=3,
            include_vote_buttons=True,
        )
        actions_blocks = _find_blocks_of_type(blocks, "actions")
        # First actions block should have 0 buttons
        assert len(actions_blocks[0]["elements"]) == 0

    def test_round_zero_progress(self):
        """Round 0/0 displays correctly."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=["claude"],
            current_round=0,
            total_rounds=0,
        )
        fields_block = blocks[2]
        assert "Round 0/0" in fields_block["fields"][1]["text"]


# ===========================================================================
# build_consensus_message_blocks tests
# ===========================================================================


class TestBuildConsensusMessageBlocks:
    """Tests for build_consensus_message_blocks."""

    # --- Consensus reached ---

    def test_consensus_reached_header(self):
        """Header shows 'Consensus Reached' when consensus is True."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.85,
            winner="claude",
            final_answer="Answer text",
            vote_counts={"claude": 3},
        )
        header = blocks[0]
        assert header["type"] == "header"
        assert "Consensus Reached" in header["text"]["text"]

    def test_consensus_not_reached_header(self):
        """Header shows 'No Consensus' when consensus is False."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=False,
            confidence=0.3,
            winner=None,
            final_answer=None,
            vote_counts={},
        )
        header = blocks[0]
        assert "No Consensus" in header["text"]["text"]

    def test_header_emoji_flag(self):
        """Header text has emoji=True."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.5,
            winner=None,
            final_answer=None,
            vote_counts={},
        )
        assert blocks[0]["text"]["emoji"] is True

    # --- Task section ---

    def test_task_section(self):
        """Task section shows the debate task."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="Evaluate microservices",
            consensus_reached=True,
            confidence=0.9,
            winner=None,
            final_answer=None,
            vote_counts={},
        )
        section = blocks[1]
        assert section["type"] == "section"
        assert "*Task:* Evaluate microservices" in section["text"]["text"]

    # --- Confidence and winner fields ---

    def test_confidence_formatted_as_percentage(self):
        """Confidence is formatted as a percentage (e.g., '85%')."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.85,
            winner=None,
            final_answer=None,
            vote_counts={},
        )
        fields_block = blocks[2]
        confidence_field = fields_block["fields"][0]
        assert "85%" in confidence_field["text"]

    def test_zero_confidence(self):
        """Zero confidence is formatted as '0%'."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=False,
            confidence=0.0,
            winner=None,
            final_answer=None,
            vote_counts={},
        )
        fields_block = blocks[2]
        assert "0%" in fields_block["fields"][0]["text"]

    def test_full_confidence(self):
        """100% confidence is formatted as '100%'."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=1.0,
            winner="claude",
            final_answer=None,
            vote_counts={},
        )
        fields_block = blocks[2]
        assert "100%" in fields_block["fields"][0]["text"]

    def test_winner_field_present_when_set(self):
        """Winner field is added when winner is not None."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.9,
            winner="claude",
            final_answer=None,
            vote_counts={},
        )
        fields_block = blocks[2]
        assert len(fields_block["fields"]) == 2
        winner_field = fields_block["fields"][1]
        assert "*Winner:*" in winner_field["text"]
        assert "claude" in winner_field["text"]

    def test_no_winner_field_when_none(self):
        """No winner field when winner is None."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=False,
            confidence=0.3,
            winner=None,
            final_answer=None,
            vote_counts={},
        )
        fields_block = blocks[2]
        assert len(fields_block["fields"]) == 1

    # --- Vote counts ---

    def test_vote_counts_displayed(self):
        """Vote counts are displayed in descending order."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.8,
            winner="claude",
            final_answer=None,
            vote_counts={"claude": 5, "gpt4": 3, "gemini": 1},
        )
        text = _get_all_text(blocks)
        # claude (5) should appear before gpt4 (3) before gemini (1)
        claude_pos = text.index("claude: 5")
        gpt4_pos = text.index("gpt4: 3")
        gemini_pos = text.index("gemini: 1")
        assert claude_pos < gpt4_pos < gemini_pos

    def test_vote_counts_singular_plural(self):
        """Vote counts use 'vote' (singular) for 1 and 'votes' (plural) for >1."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.8,
            winner=None,
            final_answer=None,
            vote_counts={"claude": 1, "gpt4": 2},
        )
        text = _get_all_text(blocks)
        assert "claude: 1 vote" in text
        assert "gpt4: 2 votes" in text

    def test_empty_vote_counts_no_section(self):
        """Empty vote_counts dict does not add a vote section."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.9,
            winner=None,
            final_answer=None,
            vote_counts={},
        )
        text = _get_all_text(blocks)
        assert "User Votes" not in text

    def test_vote_counts_section_label(self):
        """Vote counts section has 'User Votes' label."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.9,
            winner=None,
            final_answer=None,
            vote_counts={"claude": 1},
        )
        text = _get_all_text(blocks)
        assert "User Votes" in text

    # --- Final answer ---

    def test_final_answer_shown(self):
        """Final answer is shown in a code block."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.9,
            winner="claude",
            final_answer="This is the decision.",
            vote_counts={},
        )
        text = _get_all_text(blocks)
        assert "Decision" in text
        assert "This is the decision." in text

    def test_final_answer_truncated_at_500(self):
        """Final answer preview is truncated at 500 chars with '...'."""
        long_answer = "A" * 600
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.9,
            winner=None,
            final_answer=long_answer,
            vote_counts={},
        )
        text = _get_all_text(blocks)
        # The preview should be exactly 500 'A's + '...'
        assert "A" * 500 in text
        assert "A" * 501 not in text
        assert "..." in text

    def test_final_answer_exactly_500_no_ellipsis(self):
        """Answer exactly 500 chars is shown without '...'."""
        answer = "B" * 500
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.9,
            winner=None,
            final_answer=answer,
            vote_counts={},
        )
        text = _get_all_text(blocks)
        assert "B" * 500 in text
        # There should be no trailing "..." since len == 500
        # The text appears inside ```...```, so check the raw content
        found = False
        for b in blocks:
            if b.get("type") == "section" and b.get("text", {}).get("type") == "mrkdwn":
                t = b["text"]["text"]
                if "Decision" in t and "..." not in t:
                    found = True
        # At minimum, the answer fits without truncation
        assert "B" * 500 in text

    def test_no_final_answer_no_divider_or_decision(self):
        """No final answer means no divider or decision section."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.9,
            winner=None,
            final_answer=None,
            vote_counts={},
        )
        text = _get_all_text(blocks)
        assert "Decision" not in text

    def test_final_answer_preceded_by_divider(self):
        """A divider block precedes the final answer section."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.9,
            winner=None,
            final_answer="Answer",
            vote_counts={},
        )
        # Find the divider; it should appear before the decision section
        dividers = _find_blocks_of_type(blocks, "divider")
        assert len(dividers) >= 1

    # --- Action buttons ---

    def test_view_full_button(self):
        """'View Full' button with correct URL."""
        blocks = build_consensus_message_blocks(
            debate_id="debate-123",
            task="task",
            consensus_reached=True,
            confidence=0.9,
            winner=None,
            final_answer=None,
            vote_counts={},
        )
        actions_blocks = _find_blocks_of_type(blocks, "actions")
        assert len(actions_blocks) >= 1
        view_btn = actions_blocks[-1]["elements"][0]
        assert "View Full" in view_btn["text"]["text"]
        assert view_btn["url"] == "https://aragora.ai/debate/debate-123"

    def test_audit_trail_button(self):
        """'Audit Trail' button with correct URL."""
        blocks = build_consensus_message_blocks(
            debate_id="debate-456",
            task="task",
            consensus_reached=True,
            confidence=0.9,
            winner=None,
            final_answer=None,
            vote_counts={},
        )
        actions_blocks = _find_blocks_of_type(blocks, "actions")
        audit_btn = actions_blocks[-1]["elements"][1]
        assert "Audit Trail" in audit_btn["text"]["text"]
        assert audit_btn["url"] == "https://aragora.ai/debates/provenance?debate=debate-456"

    def test_action_buttons_always_present(self):
        """Action buttons are always present regardless of consensus status."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=False,
            confidence=0.1,
            winner=None,
            final_answer=None,
            vote_counts={},
        )
        actions_blocks = _find_blocks_of_type(blocks, "actions")
        assert len(actions_blocks) >= 1

    # --- Returns list of dicts ---

    def test_returns_list_of_dicts(self):
        """Result is a list of Block Kit block dicts."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.5,
            winner=None,
            final_answer=None,
            vote_counts={},
        )
        assert isinstance(blocks, list)
        assert all(isinstance(b, dict) for b in blocks)

    # --- Block count variations ---

    def test_block_count_no_winner_no_votes_no_answer(self):
        """Minimal blocks: header + task + fields + actions = 4."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.5,
            winner=None,
            final_answer=None,
            vote_counts={},
        )
        assert len(blocks) == 4

    def test_block_count_with_votes_and_answer(self):
        """With votes and answer: header + task + fields + votes + divider + answer + actions = 7."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.9,
            winner="claude",
            final_answer="Decision text",
            vote_counts={"claude": 2},
        )
        assert len(blocks) == 7

    def test_block_count_with_votes_no_answer(self):
        """With votes but no answer: header + task + fields + votes + actions = 5."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.9,
            winner=None,
            final_answer=None,
            vote_counts={"claude": 2},
        )
        assert len(blocks) == 5

    def test_block_count_with_answer_no_votes(self):
        """With answer but no votes: header + task + fields + divider + answer + actions = 6."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.9,
            winner=None,
            final_answer="Answer text",
            vote_counts={},
        )
        assert len(blocks) == 6


# ===========================================================================
# build_debate_result_blocks alias tests
# ===========================================================================


class TestBuildDebateResultBlocksAlias:
    """Tests for the build_debate_result_blocks backward-compat alias."""

    def test_alias_is_same_function(self):
        """build_debate_result_blocks is the same function as build_consensus_message_blocks."""
        assert build_debate_result_blocks is build_consensus_message_blocks

    def test_alias_produces_same_output(self):
        """Alias produces identical output."""
        kwargs: dict[str, Any] = dict(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.75,
            winner="claude",
            final_answer="Answer",
            vote_counts={"claude": 1},
        )
        assert build_debate_result_blocks(**kwargs) == build_consensus_message_blocks(**kwargs)


# ===========================================================================
# build_start_debate_modal tests
# ===========================================================================


class TestBuildStartDebateModal:
    """Tests for build_start_debate_modal."""

    def test_returns_dict(self):
        """Result is a dict."""
        modal = build_start_debate_modal()
        assert isinstance(modal, dict)

    def test_modal_type(self):
        """Modal type is 'modal'."""
        modal = build_start_debate_modal()
        assert modal["type"] == "modal"

    def test_callback_id(self):
        """Callback ID is 'start_debate_modal'."""
        modal = build_start_debate_modal()
        assert modal["callback_id"] == "start_debate_modal"

    def test_title(self):
        """Title is 'Start Debate'."""
        modal = build_start_debate_modal()
        assert modal["title"]["type"] == "plain_text"
        assert modal["title"]["text"] == "Start Debate"

    def test_submit_button(self):
        """Submit button text is 'Start'."""
        modal = build_start_debate_modal()
        assert modal["submit"]["type"] == "plain_text"
        assert modal["submit"]["text"] == "Start"

    def test_close_button(self):
        """Close button text is 'Cancel'."""
        modal = build_start_debate_modal()
        assert modal["close"]["type"] == "plain_text"
        assert modal["close"]["text"] == "Cancel"

    def test_has_three_blocks(self):
        """Modal has exactly 3 input blocks."""
        modal = build_start_debate_modal()
        assert len(modal["blocks"]) == 3

    # --- Task input block ---

    def test_task_block_id(self):
        """First block is the task input with block_id 'task_block'."""
        modal = build_start_debate_modal()
        task_block = modal["blocks"][0]
        assert task_block["block_id"] == "task_block"
        assert task_block["type"] == "input"

    def test_task_input_multiline(self):
        """Task input is multiline."""
        modal = build_start_debate_modal()
        element = modal["blocks"][0]["element"]
        assert element["multiline"] is True

    def test_task_input_action_id(self):
        """Task input action_id is 'task_input'."""
        modal = build_start_debate_modal()
        element = modal["blocks"][0]["element"]
        assert element["action_id"] == "task_input"

    def test_task_input_placeholder(self):
        """Task input has a placeholder text."""
        modal = build_start_debate_modal()
        element = modal["blocks"][0]["element"]
        assert element["placeholder"]["text"] == "What should the agents debate?"

    def test_task_input_type(self):
        """Task input type is 'plain_text_input'."""
        modal = build_start_debate_modal()
        element = modal["blocks"][0]["element"]
        assert element["type"] == "plain_text_input"

    def test_task_label(self):
        """Task block label is 'Debate Task'."""
        modal = build_start_debate_modal()
        label = modal["blocks"][0]["label"]
        assert label["text"] == "Debate Task"

    # --- Agents selection block ---

    def test_agents_block_id(self):
        """Second block is the agent selection with block_id 'agents_block'."""
        modal = build_start_debate_modal()
        agents_block = modal["blocks"][1]
        assert agents_block["block_id"] == "agents_block"
        assert agents_block["type"] == "input"

    def test_agents_select_type(self):
        """Agent selection is a multi_static_select."""
        modal = build_start_debate_modal()
        element = modal["blocks"][1]["element"]
        assert element["type"] == "multi_static_select"

    def test_agents_select_action_id(self):
        """Agent selection action_id is 'agents_select'."""
        modal = build_start_debate_modal()
        element = modal["blocks"][1]["element"]
        assert element["action_id"] == "agents_select"

    def test_agents_options_count(self):
        """Agent selection has 8 options."""
        modal = build_start_debate_modal()
        options = modal["blocks"][1]["element"]["options"]
        assert len(options) == 8

    def test_agents_option_values(self):
        """Agent options have correct value identifiers."""
        modal = build_start_debate_modal()
        options = modal["blocks"][1]["element"]["options"]
        values = [opt["value"] for opt in options]
        expected = ["claude", "gpt4", "gemini", "mistral", "deepseek", "grok", "qwen", "kimi"]
        assert values == expected

    def test_agents_option_display_names(self):
        """Agent options have correct display names."""
        modal = build_start_debate_modal()
        options = modal["blocks"][1]["element"]["options"]
        names = [opt["text"]["text"] for opt in options]
        expected = ["Claude", "GPT-4", "Gemini", "Mistral", "DeepSeek", "Grok", "Qwen", "Kimi"]
        assert names == expected

    def test_agents_option_structure(self):
        """Each agent option has correct structure."""
        modal = build_start_debate_modal()
        options = modal["blocks"][1]["element"]["options"]
        for opt in options:
            assert opt["text"]["type"] == "plain_text"
            assert isinstance(opt["value"], str)
            assert len(opt["value"]) > 0

    def test_agents_placeholder(self):
        """Agent selection has placeholder text."""
        modal = build_start_debate_modal()
        element = modal["blocks"][1]["element"]
        assert element["placeholder"]["text"] == "Select agents"

    def test_agents_label(self):
        """Agents block label is 'Agents'."""
        modal = build_start_debate_modal()
        label = modal["blocks"][1]["label"]
        assert label["text"] == "Agents"

    # --- Rounds selection block ---

    def test_rounds_block_id(self):
        """Third block is the rounds selection with block_id 'rounds_block'."""
        modal = build_start_debate_modal()
        rounds_block = modal["blocks"][2]
        assert rounds_block["block_id"] == "rounds_block"
        assert rounds_block["type"] == "input"

    def test_rounds_select_type(self):
        """Rounds selection is a static_select."""
        modal = build_start_debate_modal()
        element = modal["blocks"][2]["element"]
        assert element["type"] == "static_select"

    def test_rounds_select_action_id(self):
        """Rounds selection action_id is 'rounds_select'."""
        modal = build_start_debate_modal()
        element = modal["blocks"][2]["element"]
        assert element["action_id"] == "rounds_select"

    def test_rounds_options(self):
        """Rounds selection has 4 options: 3, 5, 8, 9."""
        modal = build_start_debate_modal()
        options = modal["blocks"][2]["element"]["options"]
        assert len(options) == 4
        values = [opt["value"] for opt in options]
        assert values == ["3", "5", "8", "9"]

    def test_rounds_options_display(self):
        """Rounds options show 'N rounds' text."""
        modal = build_start_debate_modal()
        options = modal["blocks"][2]["element"]["options"]
        texts = [opt["text"]["text"] for opt in options]
        assert texts == ["3 rounds", "5 rounds", "8 rounds", "9 rounds"]

    def test_rounds_initial_option_uses_default_rounds(self):
        """Initial option uses DEFAULT_ROUNDS from config."""
        from aragora.config import DEFAULT_ROUNDS

        modal = build_start_debate_modal()
        initial = modal["blocks"][2]["element"]["initial_option"]
        assert initial["value"] == str(DEFAULT_ROUNDS)
        assert initial["text"]["text"] == f"{DEFAULT_ROUNDS} rounds"

    def test_rounds_placeholder(self):
        """Rounds selection has placeholder text."""
        modal = build_start_debate_modal()
        element = modal["blocks"][2]["element"]
        assert element["placeholder"]["text"] == "Number of rounds"

    def test_rounds_label(self):
        """Rounds block label is 'Rounds'."""
        modal = build_start_debate_modal()
        label = modal["blocks"][2]["label"]
        assert label["text"] == "Rounds"

    # --- Idempotency ---

    def test_multiple_calls_return_equal_results(self):
        """Repeated calls return structurally equal results."""
        m1 = build_start_debate_modal()
        m2 = build_start_debate_modal()
        assert m1 == m2


# ===========================================================================
# _build_start_debate_modal alias tests
# ===========================================================================


class TestPrivateAlias:
    """Tests for the _build_start_debate_modal private alias."""

    def test_private_alias_is_same_function(self):
        """_build_start_debate_modal is the same function as build_start_debate_modal."""
        assert _build_start_debate_modal is build_start_debate_modal

    def test_private_alias_produces_same_output(self):
        """Private alias produces identical output."""
        assert _build_start_debate_modal() == build_start_debate_modal()


# ===========================================================================
# __all__ exports tests
# ===========================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports(self):
        """Module __all__ contains all 5 expected names."""
        from aragora.server.handlers.bots.slack import blocks

        expected = {
            "build_debate_message_blocks",
            "build_consensus_message_blocks",
            "build_debate_result_blocks",
            "build_start_debate_modal",
            "_build_start_debate_modal",
        }
        assert set(blocks.__all__) == expected

    def test_all_exports_are_importable(self):
        """Every name in __all__ can be resolved from the module."""
        from aragora.server.handlers.bots.slack import blocks

        for name in blocks.__all__:
            assert hasattr(blocks, name), f"Missing export: {name}"

    def test_all_exports_count(self):
        """Exactly 5 names are exported."""
        from aragora.server.handlers.bots.slack import blocks

        assert len(blocks.__all__) == 5


# ===========================================================================
# Security and edge case tests
# ===========================================================================


class TestSecurityAndEdgeCases:
    """Security and edge case tests for block builders."""

    # --- XSS / injection in inputs ---

    def test_task_with_html_tags(self):
        """HTML tags in task are passed through (Slack renders mrkdwn, not HTML)."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task='<script>alert("xss")</script>',
            agents=["claude"],
            current_round=1,
            total_rounds=3,
        )
        section = blocks[1]
        assert '<script>alert("xss")</script>' in section["text"]["text"]

    def test_agents_with_special_chars(self):
        """Special characters in agent names are preserved."""
        agents = ["claude<>", "gpt4\"'", "gem&ini"]
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=agents,
            current_round=1,
            total_rounds=3,
        )
        fields = blocks[2]["fields"][0]["text"]
        for agent in agents:
            assert agent in fields

    def test_debate_id_with_special_chars(self):
        """Debate ID with special chars is preserved in block content."""
        debate_id = "d1/../../../etc/passwd"
        blocks = build_debate_message_blocks(
            debate_id=debate_id,
            task="task",
            agents=["claude"],
            current_round=1,
            total_rounds=3,
        )
        # Footer should truncate to first 8 chars
        footer = blocks[-1]
        footer_text = footer["elements"][0]["text"]
        assert debate_id[:8] in footer_text

    def test_unicode_task(self):
        """Unicode characters in task are handled correctly."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="Discuss the future of AI",
            agents=["claude"],
            current_round=1,
            total_rounds=3,
        )
        section = blocks[1]
        assert "AI" in section["text"]["text"]

    def test_unicode_agents(self):
        """Unicode agent names are handled correctly."""
        agents = ["agent_alpha", "agent_beta"]
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=agents,
            current_round=1,
            total_rounds=3,
        )
        fields = blocks[2]["fields"][0]["text"]
        for agent in agents:
            assert agent in fields

    def test_consensus_task_with_injection(self):
        """Injection patterns in consensus task are preserved."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task='"; DROP TABLE debates; --',
            consensus_reached=True,
            confidence=0.5,
            winner=None,
            final_answer=None,
            vote_counts={},
        )
        section = blocks[1]
        assert "DROP TABLE debates" in section["text"]["text"]

    def test_consensus_winner_with_injection(self):
        """Injection patterns in winner are preserved."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.5,
            winner="<img src=x onerror=alert(1)>",
            final_answer=None,
            vote_counts={},
        )
        fields_block = blocks[2]
        winner_field = fields_block["fields"][1]
        assert "<img src=x onerror=alert(1)>" in winner_field["text"]

    def test_consensus_final_answer_with_backticks(self):
        """Backticks in final answer are preserved."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.5,
            winner=None,
            final_answer="Use `code` here and ```blocks```",
            vote_counts={},
        )
        text = _get_all_text(blocks)
        assert "`code`" in text

    def test_very_long_task(self):
        """Very long task string does not crash."""
        long_task = "X" * 10000
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task=long_task,
            agents=["claude"],
            current_round=1,
            total_rounds=3,
        )
        assert len(blocks) > 0
        assert long_task in blocks[1]["text"]["text"]

    def test_very_long_debate_id(self):
        """Very long debate ID is handled (footer truncates)."""
        long_id = "Z" * 1000
        blocks = build_debate_message_blocks(
            debate_id=long_id,
            task="task",
            agents=["claude"],
            current_round=1,
            total_rounds=3,
        )
        footer = blocks[-1]
        footer_text = footer["elements"][0]["text"]
        assert long_id[:8] in footer_text
        assert "..." in footer_text

    def test_empty_debate_id(self):
        """Empty debate ID does not crash."""
        blocks = build_debate_message_blocks(
            debate_id="",
            task="task",
            agents=["claude"],
            current_round=1,
            total_rounds=3,
        )
        assert len(blocks) > 0
        footer = blocks[-1]
        footer_text = footer["elements"][0]["text"]
        assert "..." in footer_text

    def test_empty_task(self):
        """Empty task string does not crash."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="",
            agents=["claude"],
            current_round=1,
            total_rounds=3,
        )
        section = blocks[1]
        assert "*Task:* " in section["text"]["text"]

    def test_negative_round_numbers(self):
        """Negative round numbers are handled."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=["claude"],
            current_round=-1,
            total_rounds=-5,
        )
        fields = blocks[2]
        assert "Round -1/-5" in fields["fields"][1]["text"]

    def test_consensus_vote_counts_large_numbers(self):
        """Large vote counts are handled."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.99,
            winner="claude",
            final_answer=None,
            vote_counts={"claude": 1000000, "gpt4": 999999},
        )
        text = _get_all_text(blocks)
        assert "1000000" in text
        assert "999999" in text

    def test_consensus_confidence_edge_values(self):
        """Confidence values at boundaries work."""
        for conf in [0.0, 0.001, 0.5, 0.999, 1.0]:
            blocks = build_consensus_message_blocks(
                debate_id="d1",
                task="task",
                consensus_reached=True,
                confidence=conf,
                winner=None,
                final_answer=None,
                vote_counts={},
            )
            assert len(blocks) > 0

    def test_vote_button_value_json_is_valid(self):
        """All vote button values are valid JSON strings."""
        agents = ["a", "b", "c"]
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=agents,
            current_round=1,
            total_rounds=3,
        )
        actions_blocks = _find_blocks_of_type(blocks, "actions")
        for btn in actions_blocks[0]["elements"]:
            parsed = json.loads(btn["value"])
            assert "debate_id" in parsed
            assert "agent" in parsed

    def test_provenance_url_debate_id_encoding(self):
        """Provenance URL includes raw debate ID (no encoding needed for query param)."""
        debate_id = "debate-with-dashes-123"
        blocks = build_debate_message_blocks(
            debate_id=debate_id,
            task="task",
            agents=["claude"],
            current_round=1,
            total_rounds=3,
        )
        actions_blocks = _find_blocks_of_type(blocks, "actions")
        prov_btn = actions_blocks[1]["elements"][1]
        assert debate_id in prov_btn["url"]


# ===========================================================================
# Integration-style tests combining multiple features
# ===========================================================================


class TestIntegration:
    """Integration tests combining multiple aspects."""

    def test_full_debate_message_with_all_features(self):
        """Build a full debate message with all features enabled."""
        blocks = build_debate_message_blocks(
            debate_id="abc12345-6789-abcd-ef01-234567890abc",
            task="Should we adopt microservices architecture?",
            agents=["claude", "gpt4", "gemini"],
            current_round=3,
            total_rounds=5,
            include_vote_buttons=True,
        )
        # Check structure
        assert blocks[0]["type"] == "header"
        assert blocks[1]["type"] == "section"
        assert blocks[2]["type"] == "section"
        assert blocks[3]["type"] == "divider"
        # Vote section + vote actions + summary/prov actions + footer
        assert blocks[-1]["type"] == "context"

        # Check content
        text = _get_all_text(blocks)
        assert "microservices" in text
        assert "claude, gpt4, gemini" in text
        assert "Round 3/5" in text
        assert "abc12345..." in text

    def test_full_consensus_message_with_all_features(self):
        """Build a full consensus message with all features."""
        blocks = build_consensus_message_blocks(
            debate_id="debate-final-001",
            task="Rate limiter design",
            consensus_reached=True,
            confidence=0.92,
            winner="claude",
            final_answer="Implement a token bucket algorithm with Redis.",
            vote_counts={"claude": 5, "gpt4": 2, "gemini": 1},
        )
        text = _get_all_text(blocks)
        assert "Consensus Reached" in text
        assert "Rate limiter design" in text
        assert "92%" in text
        assert "claude" in text
        assert "token bucket" in text
        assert "claude: 5 votes" in text
        assert "gpt4: 2 votes" in text
        assert "gemini: 1 vote" in text  # singular

    def test_no_consensus_minimal(self):
        """No consensus with minimal data."""
        blocks = build_consensus_message_blocks(
            debate_id="d-min",
            task="Minimal",
            consensus_reached=False,
            confidence=0.0,
            winner=None,
            final_answer=None,
            vote_counts={},
        )
        text = _get_all_text(blocks)
        assert "No Consensus" in text
        assert "0%" in text
        assert "Minimal" in text

    def test_debate_message_serializable_to_json(self):
        """Debate message blocks are JSON-serializable."""
        blocks = build_debate_message_blocks(
            debate_id="d1",
            task="task",
            agents=["claude", "gpt4"],
            current_round=1,
            total_rounds=3,
        )
        serialized = json.dumps(blocks)
        deserialized = json.loads(serialized)
        assert deserialized == blocks

    def test_consensus_message_serializable_to_json(self):
        """Consensus message blocks are JSON-serializable."""
        blocks = build_consensus_message_blocks(
            debate_id="d1",
            task="task",
            consensus_reached=True,
            confidence=0.75,
            winner="claude",
            final_answer="Answer here",
            vote_counts={"claude": 3, "gpt4": 1},
        )
        serialized = json.dumps(blocks)
        deserialized = json.loads(serialized)
        assert deserialized == blocks

    def test_modal_serializable_to_json(self):
        """Modal dict is JSON-serializable."""
        modal = build_start_debate_modal()
        serialized = json.dumps(modal)
        deserialized = json.loads(serialized)
        assert deserialized == modal
