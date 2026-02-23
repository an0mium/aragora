"""
Tests for Microsoft Teams Adaptive Card templates.

Covers all public and internal functions in aragora.server.handlers.bots.teams_cards:
- _base_card, _header, _text, _fact_set, _submit_action, _column_set, _column
- _progress_bar
- create_debate_card (statuses, progress, truncation, actions)
- create_voting_card (default/custom options, deadline)
- create_consensus_card (confidence thresholds, agents, key points, votes)
- create_leaderboard_card (standings, period, top-10 limit, medals)
- create_debate_progress_card (progress calc, messages, timestamp, truncation)
- create_error_card (error code, retry)
- create_help_card (default/custom commands)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from aragora.server.handlers.bots.teams_cards import (
    SCHEMA,
    VERSION,
    _base_card,
    _column,
    _column_set,
    _fact_set,
    _header,
    _progress_bar,
    _submit_action,
    _text,
    create_consensus_card,
    create_debate_card,
    create_debate_progress_card,
    create_error_card,
    create_help_card,
    create_leaderboard_card,
    create_voting_card,
)


# ===========================================================================
# Constants
# ===========================================================================


class TestConstants:
    """Test module-level constants."""

    def test_schema_url(self):
        assert SCHEMA == "http://adaptivecards.io/schemas/adaptive-card.json"

    def test_version(self):
        assert VERSION == "1.4"


# ===========================================================================
# _base_card
# ===========================================================================


class TestBaseCard:
    """Tests for _base_card helper."""

    def test_returns_dict(self):
        card = _base_card()
        assert isinstance(card, dict)

    def test_schema_field(self):
        card = _base_card()
        assert card["$schema"] == SCHEMA

    def test_type_field(self):
        card = _base_card()
        assert card["type"] == "AdaptiveCard"

    def test_version_field(self):
        card = _base_card()
        assert card["version"] == VERSION

    def test_body_is_empty_list(self):
        card = _base_card()
        assert card["body"] == []

    def test_actions_is_empty_list(self):
        card = _base_card()
        assert card["actions"] == []

    def test_returns_new_instance_each_call(self):
        c1 = _base_card()
        c2 = _base_card()
        assert c1 is not c2
        c1["body"].append({"test": True})
        assert len(c2["body"]) == 0


# ===========================================================================
# _header
# ===========================================================================


class TestHeader:
    """Tests for _header helper."""

    def test_text_field(self):
        h = _header("Hello")
        assert h["text"] == "Hello"

    def test_type_is_textblock(self):
        h = _header("X")
        assert h["type"] == "TextBlock"

    def test_weight_is_bolder(self):
        h = _header("X")
        assert h["weight"] == "Bolder"

    def test_default_size_is_large(self):
        h = _header("X")
        assert h["size"] == "Large"

    def test_custom_size(self):
        h = _header("X", size="Medium")
        assert h["size"] == "Medium"

    def test_wrap_is_true(self):
        h = _header("X")
        assert h["wrap"] is True

    def test_no_color_by_default(self):
        h = _header("X")
        assert "color" not in h

    def test_with_color(self):
        h = _header("X", color="Good")
        assert h["color"] == "Good"


# ===========================================================================
# _text
# ===========================================================================


class TestText:
    """Tests for _text helper."""

    def test_text_field(self):
        t = _text("content")
        assert t["text"] == "content"

    def test_type(self):
        t = _text("X")
        assert t["type"] == "TextBlock"

    def test_default_wrap(self):
        t = _text("X")
        assert t["wrap"] is True

    def test_wrap_false(self):
        t = _text("X", wrap=False)
        assert t["wrap"] is False

    def test_default_size(self):
        t = _text("X")
        assert t["size"] == "Default"

    def test_custom_size(self):
        t = _text("X", size="Small")
        assert t["size"] == "Small"


# ===========================================================================
# _fact_set
# ===========================================================================


class TestFactSet:
    """Tests for _fact_set helper."""

    def test_type(self):
        fs = _fact_set([("A", "1")])
        assert fs["type"] == "FactSet"

    def test_single_fact(self):
        fs = _fact_set([("Key", "Value")])
        assert fs["facts"] == [{"title": "Key", "value": "Value"}]

    def test_multiple_facts(self):
        fs = _fact_set([("A", "1"), ("B", "2"), ("C", "3")])
        assert len(fs["facts"]) == 3
        assert fs["facts"][1] == {"title": "B", "value": "2"}

    def test_empty_facts(self):
        fs = _fact_set([])
        assert fs["facts"] == []


# ===========================================================================
# _submit_action
# ===========================================================================


class TestSubmitAction:
    """Tests for _submit_action helper."""

    def test_type(self):
        a = _submit_action("Click", {"key": "val"})
        assert a["type"] == "Action.Submit"

    def test_title(self):
        a = _submit_action("Click Me", {})
        assert a["title"] == "Click Me"

    def test_data(self):
        data = {"action": "vote", "value": "yes"}
        a = _submit_action("Vote", data)
        assert a["data"] == data

    def test_no_style_by_default(self):
        a = _submit_action("X", {})
        assert "style" not in a

    def test_with_style(self):
        a = _submit_action("X", {}, style="positive")
        assert a["style"] == "positive"


# ===========================================================================
# _column_set / _column
# ===========================================================================


class TestColumnSet:
    """Tests for _column_set helper."""

    def test_type(self):
        cs = _column_set([])
        assert cs["type"] == "ColumnSet"

    def test_columns(self):
        cols = [{"type": "Column"}, {"type": "Column"}]
        cs = _column_set(cols)
        assert cs["columns"] == cols


class TestColumn:
    """Tests for _column helper."""

    def test_type(self):
        c = _column([])
        assert c["type"] == "Column"

    def test_default_width(self):
        c = _column([])
        assert c["width"] == "auto"

    def test_custom_width(self):
        c = _column([], width="stretch")
        assert c["width"] == "stretch"

    def test_items(self):
        items = [_text("hello")]
        c = _column(items)
        assert c["items"] == items


# ===========================================================================
# _progress_bar
# ===========================================================================


class TestProgressBar:
    """Tests for _progress_bar helper."""

    def test_returns_list(self):
        result = _progress_bar(50)
        assert isinstance(result, list)

    def test_without_label_has_column_set_and_percentage(self):
        result = _progress_bar(50)
        # Should have: ColumnSet + percentage text
        assert len(result) == 2
        assert result[0]["type"] == "ColumnSet"
        assert result[1]["text"] == "50%"

    def test_with_label_has_three_elements(self):
        result = _progress_bar(75, label="Progress")
        assert len(result) == 3
        assert result[0]["text"] == "Progress"
        assert result[1]["type"] == "ColumnSet"
        assert result[2]["text"] == "75%"

    def test_filled_and_empty_widths(self):
        result = _progress_bar(60)
        cs = result[0]
        filled = cs["columns"][0]
        empty = cs["columns"][1]
        assert filled["width"] == "60"
        assert empty["width"] == "40"

    def test_zero_progress_uses_min_1(self):
        result = _progress_bar(0)
        cs = result[0]
        filled = cs["columns"][0]
        empty = cs["columns"][1]
        assert filled["width"] == "1"
        assert empty["width"] == "100"

    def test_100_progress(self):
        result = _progress_bar(100)
        cs = result[0]
        filled = cs["columns"][0]
        empty = cs["columns"][1]
        assert filled["width"] == "100"
        assert empty["width"] == "1"

    def test_column_styles(self):
        result = _progress_bar(50)
        cs = result[0]
        filled_container = cs["columns"][0]["items"][0]
        empty_container = cs["columns"][1]["items"][0]
        assert filled_container["style"] == "good"
        assert empty_container["style"] == "default"

    def test_spacing_none(self):
        result = _progress_bar(50)
        cs = result[0]
        assert cs["spacing"] == "None"

    def test_percentage_text_format(self):
        result = _progress_bar(42)
        assert result[-1]["text"] == "42%"


# ===========================================================================
# create_debate_card
# ===========================================================================


class TestCreateDebateCard:
    """Tests for create_debate_card."""

    def test_returns_valid_adaptive_card(self):
        card = create_debate_card("d1", "Topic", ["claude"])
        assert card["type"] == "AdaptiveCard"
        assert card["$schema"] == SCHEMA
        assert card["version"] == VERSION

    def test_topic_in_header(self):
        card = create_debate_card("d1", "My Topic", ["claude"])
        header = card["body"][0]
        assert "My Topic" in header["text"]

    def test_status_in_progress_default(self):
        card = create_debate_card("d1", "T", ["a"])
        status_block = card["body"][1]
        assert "In Progress" in status_block["text"]
        assert status_block["color"] == "Accent"

    def test_status_pending(self):
        card = create_debate_card("d1", "T", ["a"], status="pending")
        status_block = card["body"][1]
        assert "Pending" in status_block["text"]
        assert status_block["color"] == "Warning"

    def test_status_completed(self):
        card = create_debate_card("d1", "T", ["a"], status="completed")
        status_block = card["body"][1]
        assert "Completed" in status_block["text"]
        assert status_block["color"] == "Good"

    def test_status_failed(self):
        card = create_debate_card("d1", "T", ["a"], status="failed")
        status_block = card["body"][1]
        assert "Failed" in status_block["text"]
        assert status_block["color"] == "Attention"

    def test_unknown_status_uses_defaults(self):
        card = create_debate_card("d1", "T", ["a"], status="unknown_thing")
        status_block = card["body"][1]
        assert "Unknown_Thing" in status_block["text"]  # .title()
        assert status_block["color"] == "Default"

    def test_in_progress_shows_progress_bar(self):
        card = create_debate_card("d1", "T", ["a"], progress=50, status="in_progress")
        # body: header, status, progress elements (label + columnset + percent), fact_set
        body_types = [b.get("type") for b in card["body"]]
        assert "ColumnSet" in body_types  # progress bar

    def test_pending_no_progress_bar(self):
        card = create_debate_card("d1", "T", ["a"], status="pending")
        body_types = [b.get("type") for b in card["body"]]
        # Only header + status + fact_set
        assert body_types.count("ColumnSet") == 0

    def test_debate_id_truncation_long(self):
        long_id = "abcdefghijklmnop"
        card = create_debate_card(long_id, "T", ["a"])
        fact_set = [b for b in card["body"] if b.get("type") == "FactSet"][0]
        debate_id_fact = fact_set["facts"][0]
        assert debate_id_fact["value"] == "abcdefghijkl..."

    def test_debate_id_no_truncation_short(self):
        short_id = "abc123"
        card = create_debate_card(short_id, "T", ["a"])
        fact_set = [b for b in card["body"] if b.get("type") == "FactSet"][0]
        debate_id_fact = fact_set["facts"][0]
        assert debate_id_fact["value"] == "abc123"

    def test_debate_id_exactly_12_chars(self):
        exact_id = "abcdefghijkl"  # exactly 12
        card = create_debate_card(exact_id, "T", ["a"])
        fact_set = [b for b in card["body"] if b.get("type") == "FactSet"][0]
        debate_id_fact = fact_set["facts"][0]
        assert debate_id_fact["value"] == "abcdefghijkl"

    def test_agents_in_fact_set(self):
        card = create_debate_card("d1", "T", ["claude", "gpt-4", "gemini"])
        fact_set = [b for b in card["body"] if b.get("type") == "FactSet"][0]
        agents_fact = fact_set["facts"][1]
        assert agents_fact["value"] == "claude, gpt-4, gemini"

    def test_rounds_in_fact_set(self):
        card = create_debate_card("d1", "T", ["a"], current_round=2, total_rounds=5)
        fact_set = [b for b in card["body"] if b.get("type") == "FactSet"][0]
        rounds_fact = fact_set["facts"][2]
        assert rounds_fact["value"] == "2/5"

    def test_in_progress_has_vote_and_view_actions(self):
        card = create_debate_card("d1", "T", ["a"], status="in_progress")
        titles = [a["title"] for a in card["actions"]]
        assert "Vote Approve" in titles
        assert "Vote Reject" in titles
        assert "Abstain" in titles
        assert "View Details" in titles

    def test_completed_has_vote_but_no_view_details(self):
        card = create_debate_card("d1", "T", ["a"], status="completed")
        titles = [a["title"] for a in card["actions"]]
        assert "Vote Approve" in titles
        assert "Vote Reject" in titles
        assert "Abstain" in titles
        assert "View Details" not in titles

    def test_pending_no_actions(self):
        card = create_debate_card("d1", "T", ["a"], status="pending")
        assert len(card["actions"]) == 0

    def test_failed_no_actions(self):
        card = create_debate_card("d1", "T", ["a"], status="failed")
        assert len(card["actions"]) == 0

    def test_vote_action_data_includes_debate_id(self):
        card = create_debate_card("debate-xyz", "T", ["a"], status="in_progress")
        approve_action = card["actions"][0]
        assert approve_action["data"]["debate_id"] == "debate-xyz"
        assert approve_action["data"]["action"] == "vote"
        assert approve_action["data"]["value"] == "approve"

    def test_vote_reject_style_destructive(self):
        card = create_debate_card("d1", "T", ["a"], status="in_progress")
        reject_action = [a for a in card["actions"] if a["title"] == "Vote Reject"][0]
        assert reject_action["style"] == "destructive"

    def test_vote_approve_style_positive(self):
        card = create_debate_card("d1", "T", ["a"], status="in_progress")
        approve_action = [a for a in card["actions"] if a["title"] == "Vote Approve"][0]
        assert approve_action["style"] == "positive"

    def test_abstain_no_style(self):
        card = create_debate_card("d1", "T", ["a"], status="in_progress")
        abstain_action = [a for a in card["actions"] if a["title"] == "Abstain"][0]
        assert "style" not in abstain_action

    def test_progress_bar_label_shows_round(self):
        card = create_debate_card(
            "d1", "T", ["a"], current_round=2, total_rounds=5, status="in_progress"
        )
        # Find the progress label text block
        texts = [b["text"] for b in card["body"] if b.get("type") == "TextBlock"]
        assert any("Round 2/5" in t for t in texts)

    def test_card_is_json_serializable(self):
        card = create_debate_card("d1", "Topic", ["claude", "gpt-4"])
        serialized = json.dumps(card)
        assert isinstance(serialized, str)


# ===========================================================================
# create_voting_card
# ===========================================================================


class TestCreateVotingCard:
    """Tests for create_voting_card."""

    def test_returns_valid_adaptive_card(self):
        card = create_voting_card("d1", "Should we?")
        assert card["type"] == "AdaptiveCard"

    def test_header_says_cast_your_vote(self):
        card = create_voting_card("d1", "Topic")
        header = card["body"][0]
        assert header["text"] == "Cast Your Vote"
        assert header["size"] == "Medium"

    def test_topic_shown(self):
        card = create_voting_card("d1", "My question?")
        topic_block = card["body"][1]
        assert topic_block["text"] == "My question?"

    def test_default_options_three_buttons(self):
        card = create_voting_card("d1", "T")
        assert len(card["actions"]) == 3
        labels = [a["title"] for a in card["actions"]]
        assert labels == ["Approve", "Reject", "Abstain"]

    def test_default_options_styles(self):
        card = create_voting_card("d1", "T")
        assert card["actions"][0]["style"] == "positive"
        assert card["actions"][1]["style"] == "destructive"
        assert "style" not in card["actions"][2]

    def test_custom_options(self):
        opts = [
            {"value": "yes", "label": "Yes Please", "style": "positive"},
            {"value": "no", "label": "No Thanks"},
        ]
        card = create_voting_card("d1", "T", options=opts)
        assert len(card["actions"]) == 2
        assert card["actions"][0]["title"] == "Yes Please"
        assert card["actions"][1]["title"] == "No Thanks"

    def test_custom_option_without_label_uses_title_case(self):
        opts = [{"value": "maybe"}]
        card = create_voting_card("d1", "T", options=opts)
        assert card["actions"][0]["title"] == "Maybe"

    def test_action_data_includes_debate_id(self):
        card = create_voting_card("debate-abc", "T")
        for action in card["actions"]:
            assert action["data"]["debate_id"] == "debate-abc"
            assert action["data"]["action"] == "vote"

    def test_no_deadline_no_deadline_block(self):
        card = create_voting_card("d1", "T")
        texts = [b.get("text", "") for b in card["body"] if b.get("type") == "TextBlock"]
        assert not any("Deadline" in t for t in texts)

    def test_with_deadline(self):
        card = create_voting_card("d1", "T", deadline="2026-03-01 17:00")
        deadline_blocks = [
            b
            for b in card["body"]
            if b.get("type") == "TextBlock" and "Deadline" in b.get("text", "")
        ]
        assert len(deadline_blocks) == 1
        assert deadline_blocks[0]["color"] == "Warning"
        assert deadline_blocks[0]["isSubtle"] is True

    def test_container_separator_present(self):
        card = create_voting_card("d1", "T")
        containers = [b for b in card["body"] if b.get("type") == "Container"]
        assert len(containers) == 1
        assert containers[0]["separator"] is True

    def test_json_serializable(self):
        card = create_voting_card("d1", "Topic")
        assert json.dumps(card)


# ===========================================================================
# create_consensus_card
# ===========================================================================


class TestCreateConsensusCard:
    """Tests for create_consensus_card."""

    def _make(self, **overrides) -> dict[str, Any]:
        defaults = {
            "debate_id": "d1",
            "topic": "My topic",
            "consensus_type": "unanimous",
            "final_answer": "Yes, we should.",
            "confidence": 0.85,
            "supporting_agents": ["claude", "gpt-4"],
        }
        defaults.update(overrides)
        return create_consensus_card(**defaults)

    def test_returns_valid_card(self):
        card = self._make()
        assert card["type"] == "AdaptiveCard"

    def test_header_consensus_reached(self):
        card = self._make()
        header = card["body"][0]
        assert header["text"] == "Consensus Reached"
        assert header["color"] == "Good"

    def test_topic_shown(self):
        card = self._make(topic="Adopt microservices?")
        topic_block = card["body"][1]
        assert topic_block["text"] == "Adopt microservices?"
        assert topic_block["isSubtle"] is True

    def test_consensus_type_formatted(self):
        card = self._make(consensus_type="super_majority")
        container = card["body"][2]
        type_text = container["items"][0]["text"]
        assert type_text == "Super Majority"

    def test_high_confidence_color_good(self):
        card = self._make(confidence=0.9)
        container = card["body"][2]
        confidence_block = container["items"][1]
        assert confidence_block["color"] == "Good"

    def test_medium_confidence_color_warning(self):
        card = self._make(confidence=0.5)
        container = card["body"][2]
        confidence_block = container["items"][1]
        assert confidence_block["color"] == "Warning"

    def test_low_confidence_color_attention(self):
        card = self._make(confidence=0.2)
        container = card["body"][2]
        confidence_block = container["items"][1]
        assert confidence_block["color"] == "Attention"

    def test_confidence_boundary_070(self):
        card = self._make(confidence=0.7)
        container = card["body"][2]
        assert container["items"][1]["color"] == "Good"

    def test_confidence_boundary_040(self):
        card = self._make(confidence=0.4)
        container = card["body"][2]
        assert container["items"][1]["color"] == "Warning"

    def test_confidence_boundary_039(self):
        card = self._make(confidence=0.39)
        container = card["body"][2]
        assert container["items"][1]["color"] == "Attention"

    def test_confidence_format_percentage(self):
        card = self._make(confidence=0.85)
        container = card["body"][2]
        confidence_text = container["items"][1]["text"]
        assert "85%" in confidence_text

    def test_final_answer_shown(self):
        card = self._make(final_answer="We should proceed.")
        decision_container = card["body"][3]
        answer_block = decision_container["items"][1]
        assert answer_block["text"] == "We should proceed."

    def test_supporting_agents_in_facts(self):
        card = self._make(supporting_agents=["claude", "gpt-4"])
        fact_set = [b for b in card["body"] if b.get("type") == "FactSet"][0]
        supporting = fact_set["facts"][0]
        assert supporting["title"] == "Supporting"
        assert supporting["value"] == "claude, gpt-4"

    def test_no_dissenting_agents(self):
        card = self._make(dissenting_agents=None)
        fact_set = [b for b in card["body"] if b.get("type") == "FactSet"][0]
        assert len(fact_set["facts"]) == 1

    def test_with_dissenting_agents(self):
        card = self._make(dissenting_agents=["gemini"])
        fact_set = [b for b in card["body"] if b.get("type") == "FactSet"][0]
        assert len(fact_set["facts"]) == 2
        dissenting = fact_set["facts"][1]
        assert dissenting["title"] == "Dissenting"
        assert dissenting["value"] == "gemini"

    def test_no_key_points(self):
        card = self._make(key_points=None)
        containers_with_key_points = [
            b
            for b in card["body"]
            if b.get("type") == "Container"
            and any("Key Points" in str(i.get("text", "")) for i in b.get("items", []))
        ]
        assert len(containers_with_key_points) == 0

    def test_with_key_points(self):
        card = self._make(key_points=["Point A", "Point B"])
        containers = [
            b
            for b in card["body"]
            if b.get("type") == "Container"
            and any("Key Points" in str(i.get("text", "")) for i in b.get("items", []))
        ]
        assert len(containers) == 1
        items = containers[0]["items"]
        # header + 2 points
        assert len(items) == 3

    def test_key_points_limited_to_5(self):
        points = [f"Point {i}" for i in range(10)]
        card = self._make(key_points=points)
        containers = [
            b
            for b in card["body"]
            if b.get("type") == "Container"
            and any("Key Points" in str(i.get("text", "")) for i in b.get("items", []))
        ]
        items = containers[0]["items"]
        # header + 5 points (capped)
        assert len(items) == 6

    def test_no_vote_summary(self):
        card = self._make(vote_summary=None)
        vote_texts = [
            b
            for b in card["body"]
            if b.get("type") == "TextBlock" and "Votes:" in b.get("text", "")
        ]
        assert len(vote_texts) == 0

    def test_with_vote_summary(self):
        card = self._make(vote_summary={"approve": 3, "reject": 1})
        vote_texts = [
            b
            for b in card["body"]
            if b.get("type") == "TextBlock" and "Votes:" in b.get("text", "")
        ]
        assert len(vote_texts) == 1
        text = vote_texts[0]["text"]
        assert "approve: 3" in text
        assert "reject: 1" in text

    def test_actions_include_view_report_and_share(self):
        card = self._make()
        titles = [a["title"] for a in card["actions"]]
        assert "View Full Report" in titles
        assert "Share Result" in titles

    def test_view_report_action_data(self):
        card = self._make(debate_id="d-42")
        report_action = [a for a in card["actions"] if a["title"] == "View Full Report"][0]
        assert report_action["data"]["action"] == "view_report"
        assert report_action["data"]["debate_id"] == "d-42"

    def test_share_action_data(self):
        card = self._make(debate_id="d-42")
        share_action = [a for a in card["actions"] if a["title"] == "Share Result"][0]
        assert share_action["data"]["action"] == "share"
        assert share_action["data"]["debate_id"] == "d-42"

    def test_json_serializable(self):
        card = self._make(
            key_points=["A", "B"],
            vote_summary={"yes": 2},
            dissenting_agents=["grok"],
        )
        assert json.dumps(card)


# ===========================================================================
# create_leaderboard_card
# ===========================================================================


class TestCreateLeaderboardCard:
    """Tests for create_leaderboard_card."""

    def _standings(self, n: int = 5) -> list[dict[str, Any]]:
        return [
            {"name": f"agent-{i}", "score": 1500 - i * 50, "wins": 10 - i, "debates": 20}
            for i in range(n)
        ]

    def test_returns_valid_card(self):
        card = create_leaderboard_card(self._standings())
        assert card["type"] == "AdaptiveCard"

    def test_default_title(self):
        card = create_leaderboard_card(self._standings())
        assert card["body"][0]["text"] == "Agent Leaderboard"

    def test_custom_title(self):
        card = create_leaderboard_card(self._standings(), title="Weekly Standings")
        assert card["body"][0]["text"] == "Weekly Standings"

    def test_default_period(self):
        card = create_leaderboard_card(self._standings())
        period_block = card["body"][1]
        assert "All Time" in period_block["text"]

    def test_custom_period(self):
        card = create_leaderboard_card(self._standings(), period="week")
        period_block = card["body"][1]
        assert "Week" in period_block["text"]

    def test_period_underscore_replacement(self):
        card = create_leaderboard_card(self._standings(), period="this_month")
        period_block = card["body"][1]
        assert "This Month" in period_block["text"]

    def test_standings_limited_to_10(self):
        card = create_leaderboard_card(self._standings(15))
        column_sets = [b for b in card["body"] if b.get("type") == "ColumnSet"]
        assert len(column_sets) == 10

    def test_medals_for_top_3(self):
        card = create_leaderboard_card(self._standings())
        column_sets = [b for b in card["body"] if b.get("type") == "ColumnSet"]
        # First column of each ColumnSet is the medal
        medal_1 = column_sets[0]["columns"][0]["items"][0]["text"]
        medal_2 = column_sets[1]["columns"][0]["items"][0]["text"]
        medal_3 = column_sets[2]["columns"][0]["items"][0]["text"]
        assert medal_1 == "1st"
        assert medal_2 == "2nd"
        assert medal_3 == "3rd"

    def test_fourth_place_uses_th_suffix(self):
        card = create_leaderboard_card(self._standings())
        column_sets = [b for b in card["body"] if b.get("type") == "ColumnSet"]
        medal_4 = column_sets[3]["columns"][0]["items"][0]["text"]
        assert medal_4 == "4th"

    def test_top_3_bolder_and_good_color(self):
        card = create_leaderboard_card(self._standings())
        column_sets = [b for b in card["body"] if b.get("type") == "ColumnSet"]
        for i in range(3):
            name_block = column_sets[i]["columns"][1]["items"][0]
            assert name_block["weight"] == "Bolder"
            assert name_block["color"] == "Good"

    def test_beyond_top_3_default_style(self):
        card = create_leaderboard_card(self._standings())
        column_sets = [b for b in card["body"] if b.get("type") == "ColumnSet"]
        name_block = column_sets[3]["columns"][1]["items"][0]
        assert name_block["weight"] == "Default"
        assert name_block["color"] == "Default"

    def test_score_displayed(self):
        standings = [{"name": "claude", "score": 1523.7, "wins": 5, "debates": 10}]
        card = create_leaderboard_card(standings)
        column_sets = [b for b in card["body"] if b.get("type") == "ColumnSet"]
        score_text = column_sets[0]["columns"][2]["items"][0]["text"]
        assert score_text == "1524"  # formatted with .0f

    def test_elo_fallback_for_score(self):
        standings = [{"name": "claude", "elo": 1600, "wins": 3, "debates": 8}]
        card = create_leaderboard_card(standings)
        column_sets = [b for b in card["body"] if b.get("type") == "ColumnSet"]
        score_text = column_sets[0]["columns"][2]["items"][0]["text"]
        assert score_text == "1600"

    def test_total_debates_fallback(self):
        standings = [{"name": "claude", "score": 1500, "wins": 5, "total_debates": 15}]
        card = create_leaderboard_card(standings)
        column_sets = [b for b in card["body"] if b.get("type") == "ColumnSet"]
        wd_text = column_sets[0]["columns"][3]["items"][0]["text"]
        assert wd_text == "5W/15D"

    def test_missing_fields_use_defaults(self):
        standings = [{"name": "unknown-agent"}]
        card = create_leaderboard_card(standings)
        column_sets = [b for b in card["body"] if b.get("type") == "ColumnSet"]
        score_text = column_sets[0]["columns"][2]["items"][0]["text"]
        wd_text = column_sets[0]["columns"][3]["items"][0]["text"]
        assert score_text == "0"
        assert wd_text == "0W/0D"

    def test_unknown_name_default(self):
        standings = [{"score": 1500, "wins": 1, "debates": 2}]
        card = create_leaderboard_card(standings)
        column_sets = [b for b in card["body"] if b.get("type") == "ColumnSet"]
        name_block = column_sets[0]["columns"][1]["items"][0]
        assert name_block["text"] == "Unknown"

    def test_full_rankings_action(self):
        card = create_leaderboard_card(self._standings(), period="month")
        assert len(card["actions"]) == 1
        action = card["actions"][0]
        assert action["title"] == "Full Rankings"
        assert action["data"]["action"] == "view_rankings"
        assert action["data"]["period"] == "month"

    def test_first_entry_has_separator(self):
        card = create_leaderboard_card(self._standings(3))
        column_sets = [b for b in card["body"] if b.get("type") == "ColumnSet"]
        assert column_sets[0]["separator"] is True
        assert column_sets[1]["separator"] is False

    def test_empty_standings(self):
        card = create_leaderboard_card([])
        column_sets = [b for b in card["body"] if b.get("type") == "ColumnSet"]
        assert len(column_sets) == 0

    def test_json_serializable(self):
        card = create_leaderboard_card(self._standings(12))
        assert json.dumps(card)


# ===========================================================================
# create_debate_progress_card
# ===========================================================================


class TestCreateDebateProgressCard:
    """Tests for create_debate_progress_card."""

    def test_returns_valid_card(self):
        card = create_debate_progress_card("d1", "Topic", 2, 4, "critique")
        assert card["type"] == "AdaptiveCard"

    def test_header(self):
        card = create_debate_progress_card("d1", "T", 1, 3, "proposal")
        assert card["body"][0]["text"] == "Debate Update"
        assert card["body"][0]["size"] == "Medium"

    def test_topic_shown_subtle(self):
        card = create_debate_progress_card("d1", "My topic", 1, 3, "proposal")
        topic_block = card["body"][1]
        assert topic_block["text"] == "My topic"
        assert topic_block["isSubtle"] is True

    def test_progress_calculation(self):
        card = create_debate_progress_card("d1", "T", 2, 4, "critique")
        # Progress = 2/4 * 100 = 50
        percentage_blocks = [
            b
            for b in card["body"]
            if b.get("type") == "TextBlock" and b.get("text", "").endswith("%")
        ]
        assert any("50%" in b["text"] for b in percentage_blocks)

    def test_progress_100_percent(self):
        card = create_debate_progress_card("d1", "T", 3, 3, "done")
        percentage_blocks = [
            b
            for b in card["body"]
            if b.get("type") == "TextBlock" and b.get("text", "").endswith("%")
        ]
        assert any("100%" in b["text"] for b in percentage_blocks)

    def test_fact_set_round_and_phase(self):
        card = create_debate_progress_card("d1", "T", 2, 5, "voting_phase")
        fact_sets = [b for b in card["body"] if b.get("type") == "FactSet"]
        assert len(fact_sets) == 1
        facts = fact_sets[0]["facts"]
        assert facts[0]["value"] == "2/5"
        assert facts[1]["value"] == "Voting Phase"

    def test_phase_formatting(self):
        card = create_debate_progress_card("d1", "T", 1, 3, "initial_proposal")
        fact_sets = [b for b in card["body"] if b.get("type") == "FactSet"]
        phase_fact = fact_sets[0]["facts"][1]
        assert phase_fact["value"] == "Initial Proposal"

    def test_no_agent_messages(self):
        card = create_debate_progress_card("d1", "T", 1, 3, "p", agent_messages=None)
        containers = [
            b
            for b in card["body"]
            if b.get("type") == "Container"
            and any("Recent Activity" in str(i.get("text", "")) for i in b.get("items", []))
        ]
        assert len(containers) == 0

    def test_with_agent_messages(self):
        msgs = [
            {"agent": "claude", "preview": "I propose we..."},
            {"agent": "gpt-4", "preview": "I disagree because..."},
        ]
        card = create_debate_progress_card("d1", "T", 1, 3, "p", agent_messages=msgs)
        msg_blocks = [
            b for b in card["body"] if b.get("type") == "TextBlock" and "**" in b.get("text", "")
        ]
        assert len(msg_blocks) == 2
        assert "**claude**" in msg_blocks[0]["text"]

    def test_agent_messages_limited_to_3(self):
        msgs = [{"agent": f"agent-{i}", "preview": f"Message {i}"} for i in range(5)]
        card = create_debate_progress_card("d1", "T", 1, 3, "p", agent_messages=msgs)
        msg_blocks = [
            b
            for b in card["body"]
            if b.get("type") == "TextBlock"
            and "**" in b.get("text", "")
            and "agent-" in b.get("text", "")
        ]
        assert len(msg_blocks) == 3

    def test_long_message_preview_truncated(self):
        long_preview = "x" * 100
        msgs = [{"agent": "claude", "preview": long_preview}]
        card = create_debate_progress_card("d1", "T", 1, 3, "p", agent_messages=msgs)
        msg_blocks = [
            b
            for b in card["body"]
            if b.get("type") == "TextBlock" and "**claude**" in b.get("text", "")
        ]
        assert len(msg_blocks) == 1
        # Should be truncated to 80 chars + "..."
        preview_part = msg_blocks[0]["text"].split(": ", 1)[1]
        assert preview_part == "x" * 80 + "..."

    def test_short_message_not_truncated(self):
        msgs = [{"agent": "claude", "preview": "Short"}]
        card = create_debate_progress_card("d1", "T", 1, 3, "p", agent_messages=msgs)
        msg_blocks = [
            b
            for b in card["body"]
            if b.get("type") == "TextBlock" and "**claude**" in b.get("text", "")
        ]
        assert "Short" in msg_blocks[0]["text"]
        assert "..." not in msg_blocks[0]["text"]

    def test_message_exactly_80_chars_not_truncated(self):
        preview = "x" * 80
        msgs = [{"agent": "a", "preview": preview}]
        card = create_debate_progress_card("d1", "T", 1, 3, "p", agent_messages=msgs)
        msg_blocks = [
            b for b in card["body"] if b.get("type") == "TextBlock" and "**a**" in b.get("text", "")
        ]
        assert "..." not in msg_blocks[0]["text"]

    def test_message_default_agent_name(self):
        msgs = [{"preview": "text"}]
        card = create_debate_progress_card("d1", "T", 1, 3, "p", agent_messages=msgs)
        msg_blocks = [
            b
            for b in card["body"]
            if b.get("type") == "TextBlock" and "**Agent**" in b.get("text", "")
        ]
        assert len(msg_blocks) == 1

    def test_message_default_preview(self):
        msgs = [{"agent": "claude"}]
        card = create_debate_progress_card("d1", "T", 1, 3, "p", agent_messages=msgs)
        msg_blocks = [
            b
            for b in card["body"]
            if b.get("type") == "TextBlock" and "**claude**" in b.get("text", "")
        ]
        assert "..." in msg_blocks[0]["text"]

    def test_no_timestamp(self):
        card = create_debate_progress_card("d1", "T", 1, 3, "p", timestamp=None)
        updated_blocks = [
            b
            for b in card["body"]
            if b.get("type") == "TextBlock" and "Updated:" in b.get("text", "")
        ]
        assert len(updated_blocks) == 0

    def test_with_timestamp(self):
        ts = datetime(2026, 2, 23, 14, 30, 45, tzinfo=timezone.utc)
        card = create_debate_progress_card("d1", "T", 1, 3, "p", timestamp=ts)
        updated_blocks = [
            b
            for b in card["body"]
            if b.get("type") == "TextBlock" and "Updated:" in b.get("text", "")
        ]
        assert len(updated_blocks) == 1
        assert "14:30:45" in updated_blocks[0]["text"]
        assert updated_blocks[0]["isSubtle"] is True

    def test_watch_live_action(self):
        card = create_debate_progress_card("debate-99", "T", 1, 3, "p")
        assert len(card["actions"]) == 1
        action = card["actions"][0]
        assert action["title"] == "Watch Live"
        assert action["data"]["action"] == "watch"
        assert action["data"]["debate_id"] == "debate-99"

    def test_json_serializable(self):
        ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        card = create_debate_progress_card(
            "d1",
            "Topic",
            2,
            4,
            "critique",
            agent_messages=[{"agent": "claude", "preview": "msg"}],
            timestamp=ts,
        )
        assert json.dumps(card)


# ===========================================================================
# create_error_card
# ===========================================================================


class TestCreateErrorCard:
    """Tests for create_error_card."""

    def test_returns_valid_card(self):
        card = create_error_card("Error", "Something went wrong")
        assert card["type"] == "AdaptiveCard"

    def test_header_with_attention_color(self):
        card = create_error_card("API Error", "Timeout")
        header = card["body"][0]
        assert header["text"] == "API Error"
        assert header["color"] == "Attention"

    def test_message_shown(self):
        card = create_error_card("E", "Connection refused")
        msg_block = card["body"][1]
        assert msg_block["text"] == "Connection refused"

    def test_no_error_code(self):
        card = create_error_card("E", "msg", error_code=None)
        code_blocks = [
            b
            for b in card["body"]
            if b.get("type") == "TextBlock" and "Error Code" in b.get("text", "")
        ]
        assert len(code_blocks) == 0

    def test_with_error_code(self):
        card = create_error_card("E", "msg", error_code="ERR_429")
        code_blocks = [
            b
            for b in card["body"]
            if b.get("type") == "TextBlock" and "Error Code" in b.get("text", "")
        ]
        assert len(code_blocks) == 1
        assert "ERR_429" in code_blocks[0]["text"]
        assert code_blocks[0]["isSubtle"] is True

    def test_no_retry_action(self):
        card = create_error_card("E", "msg", retry_action=None)
        titles = [a["title"] for a in card["actions"]]
        assert "Retry" not in titles
        assert "Get Help" in titles

    def test_with_retry_action(self):
        retry_data = {"action": "retry", "debate_id": "d1"}
        card = create_error_card("E", "msg", retry_action=retry_data)
        titles = [a["title"] for a in card["actions"]]
        assert "Retry" in titles
        retry_action = [a for a in card["actions"] if a["title"] == "Retry"][0]
        assert retry_action["data"] == retry_data
        assert retry_action["style"] == "positive"

    def test_get_help_action_always_present(self):
        card = create_error_card("E", "msg")
        help_action = [a for a in card["actions"] if a["title"] == "Get Help"][0]
        assert help_action["data"] == {"action": "help"}

    def test_retry_before_help(self):
        card = create_error_card("E", "msg", retry_action={"action": "retry"})
        assert card["actions"][0]["title"] == "Retry"
        assert card["actions"][1]["title"] == "Get Help"

    def test_json_serializable(self):
        card = create_error_card("Error", "msg", error_code="E1", retry_action={"r": 1})
        assert json.dumps(card)


# ===========================================================================
# create_help_card
# ===========================================================================


class TestCreateHelpCard:
    """Tests for create_help_card."""

    def test_returns_valid_card(self):
        card = create_help_card()
        assert card["type"] == "AdaptiveCard"

    def test_header(self):
        card = create_help_card()
        assert card["body"][0]["text"] == "Aragora Commands"

    def test_description_text(self):
        card = create_help_card()
        desc = card["body"][1]
        assert "Multi-agent debate" in desc["text"]

    def test_default_commands_count(self):
        card = create_help_card()
        container = [b for b in card["body"] if b.get("type") == "Container"][0]
        assert len(container["items"]) == 8  # 8 default commands

    def test_default_commands_include_debate(self):
        card = create_help_card()
        container = [b for b in card["body"] if b.get("type") == "Container"][0]
        texts = [item["text"] for item in container["items"]]
        assert any("/aragora debate" in t for t in texts)

    def test_default_commands_include_help(self):
        card = create_help_card()
        container = [b for b in card["body"] if b.get("type") == "Container"][0]
        texts = [item["text"] for item in container["items"]]
        assert any("/aragora help" in t for t in texts)

    def test_custom_commands(self):
        custom = [
            {"name": "/test cmd", "desc": "A test command"},
            {"name": "/another", "desc": "Another one"},
        ]
        card = create_help_card(commands=custom)
        container = [b for b in card["body"] if b.get("type") == "Container"][0]
        assert len(container["items"]) == 2
        assert "/test cmd" in container["items"][0]["text"]

    def test_documentation_action(self):
        card = create_help_card()
        assert len(card["actions"]) == 1
        action = card["actions"][0]
        assert action["type"] == "Action.OpenUrl"
        assert action["title"] == "Documentation"
        assert action["url"] == "https://docs.aragora.ai"

    def test_container_has_separator(self):
        card = create_help_card()
        container = [b for b in card["body"] if b.get("type") == "Container"][0]
        assert container["separator"] is True

    def test_command_items_wrap(self):
        card = create_help_card()
        container = [b for b in card["body"] if b.get("type") == "Container"][0]
        for item in container["items"]:
            assert item["wrap"] is True

    def test_json_serializable(self):
        card = create_help_card()
        assert json.dumps(card)


# ===========================================================================
# __all__ exports
# ===========================================================================


class TestExports:
    """Test that __all__ exports are correct."""

    def test_all_exports(self):
        from aragora.server.handlers.bots import teams_cards

        expected = [
            "create_debate_card",
            "create_voting_card",
            "create_consensus_card",
            "create_leaderboard_card",
            "create_debate_progress_card",
            "create_error_card",
            "create_help_card",
        ]
        assert set(teams_cards.__all__) == set(expected)

    def test_all_exports_are_callable(self):
        from aragora.server.handlers.bots import teams_cards

        for name in teams_cards.__all__:
            assert callable(getattr(teams_cards, name))
