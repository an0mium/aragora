"""
Tests for Microsoft Teams Adaptive Cards Templates.

Tests cover:
- _base_card() - Base card structure
- _header() - Header text blocks
- _text() - Text blocks
- _fact_set() - Fact sets
- _submit_action() - Submit action buttons
- _column_set() / _column() - Column layouts
- _progress_bar() - Progress visualization
- create_debate_card() - Debate information card
- create_voting_card() - Interactive voting card
- create_consensus_card() - Consensus result card
- create_leaderboard_card() - Agent leaderboard card
- create_debate_progress_card() - Progress update card
- create_error_card() - Error notification card
- create_help_card() - Help/commands card
"""

import pytest
from datetime import datetime, timezone

from aragora.server.handlers.bots.teams_cards import (
    SCHEMA,
    VERSION,
    _base_card,
    _header,
    _text,
    _fact_set,
    _submit_action,
    _column_set,
    _column,
    _progress_bar,
    create_debate_card,
    create_voting_card,
    create_consensus_card,
    create_leaderboard_card,
    create_debate_progress_card,
    create_error_card,
    create_help_card,
)


# =============================================================================
# Tests: Constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_schema_url(self):
        assert SCHEMA == "http://adaptivecards.io/schemas/adaptive-card.json"

    def test_version(self):
        assert VERSION == "1.4"


# =============================================================================
# Tests: _base_card
# =============================================================================


class TestBaseCard:
    """Tests for _base_card function."""

    def test_base_card_structure(self):
        """Base card has required structure."""
        card = _base_card()

        assert card["$schema"] == SCHEMA
        assert card["type"] == "AdaptiveCard"
        assert card["version"] == VERSION
        assert isinstance(card["body"], list)
        assert isinstance(card["actions"], list)

    def test_base_card_empty_body_and_actions(self):
        """Base card has empty body and actions by default."""
        card = _base_card()

        assert len(card["body"]) == 0
        assert len(card["actions"]) == 0


# =============================================================================
# Tests: _header
# =============================================================================


class TestHeader:
    """Tests for _header function."""

    def test_header_default(self):
        """Header with default options."""
        block = _header("Test Header")

        assert block["type"] == "TextBlock"
        assert block["text"] == "Test Header"
        assert block["weight"] == "Bolder"
        assert block["size"] == "Large"
        assert block["wrap"] is True

    def test_header_custom_size(self):
        """Header with custom size."""
        block = _header("Test", size="Medium")

        assert block["size"] == "Medium"

    def test_header_with_color(self):
        """Header with color."""
        block = _header("Test", color="Good")

        assert block["color"] == "Good"

    def test_header_without_color(self):
        """Header without color omits the property."""
        block = _header("Test")

        assert "color" not in block


# =============================================================================
# Tests: _text
# =============================================================================


class TestText:
    """Tests for _text function."""

    def test_text_default(self):
        """Text block with default options."""
        block = _text("Sample text")

        assert block["type"] == "TextBlock"
        assert block["text"] == "Sample text"
        assert block["wrap"] is True
        assert block["size"] == "Default"

    def test_text_no_wrap(self):
        """Text block without wrap."""
        block = _text("No wrap", wrap=False)

        assert block["wrap"] is False

    def test_text_custom_size(self):
        """Text block with custom size."""
        block = _text("Small text", size="Small")

        assert block["size"] == "Small"


# =============================================================================
# Tests: _fact_set
# =============================================================================


class TestFactSet:
    """Tests for _fact_set function."""

    def test_fact_set_structure(self):
        """Fact set with facts."""
        facts = [("Key1", "Value1"), ("Key2", "Value2")]
        block = _fact_set(facts)

        assert block["type"] == "FactSet"
        assert len(block["facts"]) == 2
        assert block["facts"][0]["title"] == "Key1"
        assert block["facts"][0]["value"] == "Value1"

    def test_fact_set_empty(self):
        """Fact set with no facts."""
        block = _fact_set([])

        assert block["type"] == "FactSet"
        assert len(block["facts"]) == 0


# =============================================================================
# Tests: _submit_action
# =============================================================================


class TestSubmitAction:
    """Tests for _submit_action function."""

    def test_submit_action_basic(self):
        """Submit action with basic options."""
        action = _submit_action("Click Me", {"action": "click", "id": "123"})

        assert action["type"] == "Action.Submit"
        assert action["title"] == "Click Me"
        assert action["data"]["action"] == "click"
        assert action["data"]["id"] == "123"

    def test_submit_action_with_style(self):
        """Submit action with style."""
        action = _submit_action("Delete", {"action": "delete"}, style="destructive")

        assert action["style"] == "destructive"

    def test_submit_action_without_style(self):
        """Submit action without style omits the property."""
        action = _submit_action("OK", {"action": "ok"})

        assert "style" not in action


# =============================================================================
# Tests: _column_set and _column
# =============================================================================


class TestColumnLayout:
    """Tests for column layout functions."""

    def test_column_set(self):
        """Column set structure."""
        columns = [{"type": "Column"}, {"type": "Column"}]
        block = _column_set(columns)

        assert block["type"] == "ColumnSet"
        assert len(block["columns"]) == 2

    def test_column_default(self):
        """Column with default width."""
        items = [{"type": "TextBlock", "text": "Item"}]
        col = _column(items)

        assert col["type"] == "Column"
        assert col["width"] == "auto"
        assert len(col["items"]) == 1

    def test_column_custom_width(self):
        """Column with custom width."""
        col = _column([], width="stretch")

        assert col["width"] == "stretch"


# =============================================================================
# Tests: _progress_bar
# =============================================================================


class TestProgressBar:
    """Tests for _progress_bar function."""

    def test_progress_bar_structure(self):
        """Progress bar generates expected elements."""
        elements = _progress_bar(50)

        # Should have column set and percentage text
        assert len(elements) >= 2
        # Find the column set
        column_set = next(e for e in elements if e.get("type") == "ColumnSet")
        assert len(column_set["columns"]) == 2

    def test_progress_bar_with_label(self):
        """Progress bar with label."""
        elements = _progress_bar(75, label="Progress")

        # First element should be label
        assert elements[0]["type"] == "TextBlock"
        assert elements[0]["text"] == "Progress"

    def test_progress_bar_percentage_text(self):
        """Progress bar shows percentage."""
        elements = _progress_bar(30)

        # Last element should show percentage
        percentage_block = elements[-1]
        assert "30%" in percentage_block["text"]


# =============================================================================
# Tests: create_debate_card
# =============================================================================


class TestCreateDebateCard:
    """Tests for create_debate_card function."""

    def test_debate_card_structure(self):
        """Debate card has expected structure."""
        card = create_debate_card(
            debate_id="debate-123",
            topic="Should we adopt microservices?",
            agents=["Claude", "GPT-4", "Gemini"],
            progress=50,
            current_round=2,
            total_rounds=5,
        )

        assert card["type"] == "AdaptiveCard"
        assert len(card["body"]) > 0
        assert len(card["actions"]) > 0

    def test_debate_card_topic_in_header(self):
        """Debate card shows topic in header."""
        card = create_debate_card(
            debate_id="debate-123",
            topic="Important Question",
            agents=["Claude"],
        )

        header = card["body"][0]
        assert "Important Question" in header["text"]

    def test_debate_card_status_pending(self):
        """Debate card shows pending status."""
        card = create_debate_card(
            debate_id="debate-123",
            topic="Test",
            agents=["Claude"],
            status="pending",
        )

        # Find status block
        status_block = card["body"][1]
        assert "Pending" in status_block["text"]
        assert status_block["color"] == "Warning"

    def test_debate_card_status_in_progress(self):
        """Debate card shows in_progress status with progress bar."""
        card = create_debate_card(
            debate_id="debate-123",
            topic="Test",
            agents=["Claude"],
            status="in_progress",
            progress=75,
        )

        # Should include progress bar elements
        body_types = [b.get("type") for b in card["body"]]
        assert "ColumnSet" in body_types  # Progress bar uses ColumnSet

    def test_debate_card_status_completed(self):
        """Debate card shows completed status."""
        card = create_debate_card(
            debate_id="debate-123",
            topic="Test",
            agents=["Claude"],
            status="completed",
        )

        status_block = card["body"][1]
        assert status_block["color"] == "Good"

    def test_debate_card_voting_actions(self):
        """Debate card has voting actions for in_progress/completed."""
        card = create_debate_card(
            debate_id="debate-123",
            topic="Test",
            agents=["Claude"],
            status="in_progress",
        )

        action_titles = [a["title"] for a in card["actions"]]
        assert "Vote Approve" in action_titles
        assert "Vote Reject" in action_titles
        assert "Abstain" in action_titles

    def test_debate_card_agents_in_facts(self):
        """Debate card lists agents in fact set."""
        card = create_debate_card(
            debate_id="debate-123",
            topic="Test",
            agents=["Claude", "GPT-4"],
        )

        # Find fact set
        fact_set = next((b for b in card["body"] if b.get("type") == "FactSet"), None)
        assert fact_set is not None
        facts_dict = {f["title"]: f["value"] for f in fact_set["facts"]}
        assert "Claude, GPT-4" in facts_dict.get("Agents", "")


# =============================================================================
# Tests: create_voting_card
# =============================================================================


class TestCreateVotingCard:
    """Tests for create_voting_card function."""

    def test_voting_card_structure(self):
        """Voting card has expected structure."""
        card = create_voting_card(
            debate_id="debate-123",
            topic="Cast your vote on the proposal",
        )

        assert card["type"] == "AdaptiveCard"
        assert len(card["body"]) > 0
        assert len(card["actions"]) >= 3  # Default: approve, reject, abstain

    def test_voting_card_default_options(self):
        """Voting card has default vote options."""
        card = create_voting_card(
            debate_id="debate-123",
            topic="Test vote",
        )

        action_titles = [a["title"] for a in card["actions"]]
        assert "Approve" in action_titles
        assert "Reject" in action_titles
        assert "Abstain" in action_titles

    def test_voting_card_custom_options(self):
        """Voting card with custom options."""
        options = [
            {"value": "option_a", "label": "Option A"},
            {"value": "option_b", "label": "Option B"},
        ]
        card = create_voting_card(
            debate_id="debate-123",
            topic="Choose an option",
            options=options,
        )

        action_titles = [a["title"] for a in card["actions"]]
        assert "Option A" in action_titles
        assert "Option B" in action_titles

    def test_voting_card_with_deadline(self):
        """Voting card shows deadline."""
        card = create_voting_card(
            debate_id="debate-123",
            topic="Time-limited vote",
            deadline="2025-01-31 23:59",
        )

        # Find deadline text
        deadline_block = next((b for b in card["body"] if "Deadline" in b.get("text", "")), None)
        assert deadline_block is not None

    def test_voting_card_action_data(self):
        """Voting card actions include debate_id."""
        card = create_voting_card(
            debate_id="debate-xyz",
            topic="Test",
        )

        for action in card["actions"]:
            assert action["data"]["debate_id"] == "debate-xyz"
            assert action["data"]["action"] == "vote"


# =============================================================================
# Tests: create_consensus_card
# =============================================================================


class TestCreateConsensusCard:
    """Tests for create_consensus_card function."""

    def test_consensus_card_structure(self):
        """Consensus card has expected structure."""
        card = create_consensus_card(
            debate_id="debate-123",
            topic="Original topic",
            consensus_type="unanimous",
            final_answer="The recommendation is to proceed.",
            confidence=0.85,
            supporting_agents=["Claude", "GPT-4"],
        )

        assert card["type"] == "AdaptiveCard"
        assert len(card["body"]) > 0
        assert len(card["actions"]) >= 2  # View report, share

    def test_consensus_card_confidence_color_high(self):
        """Consensus card shows Good color for high confidence."""
        card = create_consensus_card(
            debate_id="debate-123",
            topic="Test",
            consensus_type="majority",
            final_answer="Answer",
            confidence=0.75,
            supporting_agents=["Claude"],
        )

        # Find confidence block in container
        body_texts = []
        for block in card["body"]:
            if block.get("type") == "Container":
                for item in block.get("items", []):
                    if "Confidence" in item.get("text", ""):
                        assert item["color"] == "Good"

    def test_consensus_card_confidence_color_medium(self):
        """Consensus card shows Warning color for medium confidence."""
        card = create_consensus_card(
            debate_id="debate-123",
            topic="Test",
            consensus_type="majority",
            final_answer="Answer",
            confidence=0.5,
            supporting_agents=["Claude"],
        )

        for block in card["body"]:
            if block.get("type") == "Container":
                for item in block.get("items", []):
                    if "Confidence" in item.get("text", ""):
                        assert item["color"] == "Warning"

    def test_consensus_card_confidence_color_low(self):
        """Consensus card shows Attention color for low confidence."""
        card = create_consensus_card(
            debate_id="debate-123",
            topic="Test",
            consensus_type="weak",
            final_answer="Answer",
            confidence=0.3,
            supporting_agents=["Claude"],
        )

        for block in card["body"]:
            if block.get("type") == "Container":
                for item in block.get("items", []):
                    if "Confidence" in item.get("text", ""):
                        assert item["color"] == "Attention"

    def test_consensus_card_with_dissenting(self):
        """Consensus card shows dissenting agents."""
        card = create_consensus_card(
            debate_id="debate-123",
            topic="Test",
            consensus_type="majority",
            final_answer="Answer",
            confidence=0.7,
            supporting_agents=["Claude", "GPT-4"],
            dissenting_agents=["Gemini"],
        )

        # Find fact set with dissenting
        fact_set = next((b for b in card["body"] if b.get("type") == "FactSet"), None)
        assert fact_set is not None
        facts_dict = {f["title"]: f["value"] for f in fact_set["facts"]}
        assert "Dissenting" in facts_dict

    def test_consensus_card_with_key_points(self):
        """Consensus card shows key points."""
        card = create_consensus_card(
            debate_id="debate-123",
            topic="Test",
            consensus_type="unanimous",
            final_answer="Answer",
            confidence=0.9,
            supporting_agents=["Claude"],
            key_points=["Point 1", "Point 2", "Point 3"],
        )

        # Find container with key points
        has_key_points = any("Key Points" in str(block) for block in card["body"])
        assert has_key_points

    def test_consensus_card_with_vote_summary(self):
        """Consensus card shows vote summary."""
        card = create_consensus_card(
            debate_id="debate-123",
            topic="Test",
            consensus_type="majority",
            final_answer="Answer",
            confidence=0.75,
            supporting_agents=["Claude"],
            vote_summary={"approve": 5, "reject": 2},
        )

        # Find vote text
        has_vote_summary = any("Votes:" in str(block) for block in card["body"])
        assert has_vote_summary


# =============================================================================
# Tests: create_leaderboard_card
# =============================================================================


class TestCreateLeaderboardCard:
    """Tests for create_leaderboard_card function."""

    def test_leaderboard_card_structure(self):
        """Leaderboard card has expected structure."""
        standings = [
            {"name": "Claude", "score": 1250, "wins": 10, "debates": 15},
            {"name": "GPT-4", "score": 1180, "wins": 8, "debates": 14},
        ]
        card = create_leaderboard_card(standings)

        assert card["type"] == "AdaptiveCard"
        assert len(card["body"]) > 0

    def test_leaderboard_card_shows_period(self):
        """Leaderboard card shows time period."""
        card = create_leaderboard_card([], period="week")

        # Find period text
        has_period = any("Week" in str(block) for block in card["body"])
        assert has_period

    def test_leaderboard_card_custom_title(self):
        """Leaderboard card with custom title."""
        card = create_leaderboard_card([], title="Weekly Rankings")

        header = card["body"][0]
        assert "Weekly Rankings" in header["text"]

    def test_leaderboard_card_rankings(self):
        """Leaderboard card shows rankings."""
        standings = [
            {"name": "Claude", "score": 1300, "wins": 12, "total_debates": 15},
            {"name": "GPT-4", "elo": 1250, "wins": 10, "debates": 14},  # Alternative fields
            {"name": "Gemini", "score": 1200, "wins": 8, "debates": 12},
        ]
        card = create_leaderboard_card(standings)

        # Should have column sets for each agent
        column_sets = [b for b in card["body"] if b.get("type") == "ColumnSet"]
        assert len(column_sets) == 3

    def test_leaderboard_card_top_3_highlighted(self):
        """Top 3 entries are highlighted."""
        standings = [
            {"name": "First", "score": 100, "wins": 10, "debates": 10},
            {"name": "Second", "score": 90, "wins": 9, "debates": 10},
            {"name": "Third", "score": 80, "wins": 8, "debates": 10},
            {"name": "Fourth", "score": 70, "wins": 7, "debates": 10},
        ]
        card = create_leaderboard_card(standings)

        # Check first 3 have "Good" color
        column_sets = [b for b in card["body"] if b.get("type") == "ColumnSet"]
        for i, cs in enumerate(column_sets[:3]):
            # Name column should have color "Good"
            name_col = cs["columns"][1]
            name_block = name_col["items"][0]
            assert name_block.get("color") == "Good"

    def test_leaderboard_card_full_rankings_action(self):
        """Leaderboard card has full rankings action."""
        card = create_leaderboard_card([])

        action_titles = [a["title"] for a in card["actions"]]
        assert "Full Rankings" in action_titles


# =============================================================================
# Tests: create_debate_progress_card
# =============================================================================


class TestCreateDebateProgressCard:
    """Tests for create_debate_progress_card function."""

    def test_progress_card_structure(self):
        """Progress card has expected structure."""
        card = create_debate_progress_card(
            debate_id="debate-123",
            topic="Test debate",
            current_round=2,
            total_rounds=5,
            current_phase="proposal",
        )

        assert card["type"] == "AdaptiveCard"
        assert len(card["body"]) > 0

    def test_progress_card_shows_progress(self):
        """Progress card shows progress percentage."""
        card = create_debate_progress_card(
            debate_id="debate-123",
            topic="Test",
            current_round=3,
            total_rounds=6,
            current_phase="critique",
        )

        # Progress should be 50%
        has_progress = any("50%" in str(block) for block in card["body"])
        assert has_progress

    def test_progress_card_shows_phase(self):
        """Progress card shows current phase."""
        card = create_debate_progress_card(
            debate_id="debate-123",
            topic="Test",
            current_round=1,
            total_rounds=3,
            current_phase="voting_round",
        )

        # Find fact set with phase
        fact_set = next((b for b in card["body"] if b.get("type") == "FactSet"), None)
        assert fact_set is not None
        facts_dict = {f["title"]: f["value"] for f in fact_set["facts"]}
        assert "Voting Round" in facts_dict.get("Phase", "")

    def test_progress_card_with_messages(self):
        """Progress card shows recent agent messages."""
        messages = [
            {"agent": "Claude", "preview": "I believe we should..."},
            {"agent": "GPT-4", "preview": "On the other hand..."},
        ]
        card = create_debate_progress_card(
            debate_id="debate-123",
            topic="Test",
            current_round=2,
            total_rounds=3,
            current_phase="proposal",
            agent_messages=messages,
        )

        # Should have recent activity
        has_activity = any(
            "Recent Activity" in str(block) or "Claude" in str(block) for block in card["body"]
        )
        assert has_activity

    def test_progress_card_truncates_long_messages(self):
        """Progress card truncates long messages."""
        long_message = "A" * 100  # 100 characters
        messages = [{"agent": "Claude", "preview": long_message}]

        card = create_debate_progress_card(
            debate_id="debate-123",
            topic="Test",
            current_round=1,
            total_rounds=3,
            current_phase="proposal",
            agent_messages=messages,
        )

        # Should be truncated to ~80 chars + "..."
        card_str = str(card)
        assert "A" * 81 not in card_str  # Should not have 81 consecutive A's

    def test_progress_card_with_timestamp(self):
        """Progress card shows timestamp."""
        timestamp = datetime(2025, 1, 15, 14, 30, 45, tzinfo=timezone.utc)
        card = create_debate_progress_card(
            debate_id="debate-123",
            topic="Test",
            current_round=1,
            total_rounds=3,
            current_phase="proposal",
            timestamp=timestamp,
        )

        has_timestamp = any("14:30:45" in str(block) for block in card["body"])
        assert has_timestamp

    def test_progress_card_watch_action(self):
        """Progress card has watch live action."""
        card = create_debate_progress_card(
            debate_id="debate-123",
            topic="Test",
            current_round=1,
            total_rounds=3,
            current_phase="proposal",
        )

        action_titles = [a["title"] for a in card["actions"]]
        assert "Watch Live" in action_titles


# =============================================================================
# Tests: create_error_card
# =============================================================================


class TestCreateErrorCard:
    """Tests for create_error_card function."""

    def test_error_card_structure(self):
        """Error card has expected structure."""
        card = create_error_card(
            title="Error Occurred",
            message="Something went wrong. Please try again.",
        )

        assert card["type"] == "AdaptiveCard"
        assert len(card["body"]) > 0

    def test_error_card_header_color(self):
        """Error card header has Attention color."""
        card = create_error_card(
            title="Error",
            message="Details",
        )

        header = card["body"][0]
        assert header["color"] == "Attention"

    def test_error_card_with_error_code(self):
        """Error card shows error code."""
        card = create_error_card(
            title="Error",
            message="Details",
            error_code="ERR_001",
        )

        has_error_code = any("ERR_001" in str(block) for block in card["body"])
        assert has_error_code

    def test_error_card_with_retry_action(self):
        """Error card has retry action when provided."""
        card = create_error_card(
            title="Error",
            message="Details",
            retry_action={"action": "retry", "id": "123"},
        )

        action_titles = [a["title"] for a in card["actions"]]
        assert "Retry" in action_titles

    def test_error_card_has_help_action(self):
        """Error card always has help action."""
        card = create_error_card(
            title="Error",
            message="Details",
        )

        action_titles = [a["title"] for a in card["actions"]]
        assert "Get Help" in action_titles


# =============================================================================
# Tests: create_help_card
# =============================================================================


class TestCreateHelpCard:
    """Tests for create_help_card function."""

    def test_help_card_structure(self):
        """Help card has expected structure."""
        card = create_help_card()

        assert card["type"] == "AdaptiveCard"
        assert len(card["body"]) > 0

    def test_help_card_default_commands(self):
        """Help card shows default commands."""
        card = create_help_card()

        card_str = str(card)
        assert "/aragora debate" in card_str
        assert "/aragora status" in card_str
        assert "/aragora vote" in card_str
        assert "/aragora help" in card_str

    def test_help_card_custom_commands(self):
        """Help card with custom commands."""
        commands = [
            {"name": "/custom", "desc": "A custom command"},
            {"name": "/another", "desc": "Another command"},
        ]
        card = create_help_card(commands=commands)

        card_str = str(card)
        assert "/custom" in card_str
        assert "A custom command" in card_str

    def test_help_card_documentation_link(self):
        """Help card has documentation link."""
        card = create_help_card()

        doc_action = next((a for a in card["actions"] if a.get("type") == "Action.OpenUrl"), None)
        assert doc_action is not None
        assert "docs.aragora.ai" in doc_action["url"]


# =============================================================================
# Tests: Module Exports
# =============================================================================


class TestExports:
    """Tests for module __all__ exports."""

    def test_all_exports(self):
        """All public functions are exported."""
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

        for name in expected:
            assert name in teams_cards.__all__
            assert hasattr(teams_cards, name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
