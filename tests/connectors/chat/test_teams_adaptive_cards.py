"""
Comprehensive tests for Teams Adaptive Cards Builder.

Tests cover:
- AgentContribution and RoundProgress dataclasses
- TeamsAdaptiveCards.get_agent_icon()
- TeamsAdaptiveCards.wrap_as_card()
- TeamsAdaptiveCards.starting_card()
- TeamsAdaptiveCards.progress_card()
- TeamsAdaptiveCards.voting_card()
- TeamsAdaptiveCards.verdict_card()
- TeamsAdaptiveCards.error_card()
- TeamsAdaptiveCards.receipt_card()
"""

import pytest

from aragora.connectors.chat.teams_adaptive_cards import (
    TeamsAdaptiveCards,
    AgentContribution,
    RoundProgress,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def sample_agents():
    """Create sample agent contributions."""
    return [
        AgentContribution(
            name="Claude",
            position="for",
            key_point="Better scalability",
            confidence=0.85,
        ),
        AgentContribution(
            name="GPT-4",
            position="for",
            key_point="Team autonomy",
            confidence=0.78,
        ),
        AgentContribution(
            name="Gemini",
            position="against",
            key_point="Complexity cost",
            confidence=0.72,
        ),
    ]


@pytest.fixture
def sample_round_progress():
    """Create sample round progress."""
    return RoundProgress(
        round_number=2,
        total_rounds=5,
        agent_messages=[
            {"agent": "Claude", "summary": "We should consider the long-term benefits"},
            {"agent": "GPT-4", "summary": "I agree with the previous points"},
        ],
        current_consensus="Leaning towards microservices",
    )


# ===========================================================================
# AgentContribution Tests
# ===========================================================================


class TestAgentContribution:
    """Tests for AgentContribution dataclass."""

    def test_agent_contribution_creation(self):
        """Test AgentContribution with all fields."""
        agent = AgentContribution(
            name="Claude",
            position="for",
            key_point="Good performance",
            confidence=0.9,
            icon_url="https://example.com/claude.png",
        )

        assert agent.name == "Claude"
        assert agent.position == "for"
        assert agent.key_point == "Good performance"
        assert agent.confidence == 0.9
        assert agent.icon_url == "https://example.com/claude.png"

    def test_agent_contribution_defaults(self):
        """Test AgentContribution default values."""
        agent = AgentContribution(
            name="Test",
            position="against",
            key_point="Key point",
        )

        assert agent.confidence == 0.5
        assert agent.icon_url is None

    def test_agent_contribution_positions(self):
        """Test different position values."""
        for_agent = AgentContribution(name="A", position="for", key_point="P")
        against_agent = AgentContribution(name="B", position="against", key_point="P")
        neutral_agent = AgentContribution(name="C", position="neutral", key_point="P")

        assert for_agent.position == "for"
        assert against_agent.position == "against"
        assert neutral_agent.position == "neutral"


# ===========================================================================
# RoundProgress Tests
# ===========================================================================


class TestRoundProgress:
    """Tests for RoundProgress dataclass."""

    def test_round_progress_creation(self, sample_round_progress):
        """Test RoundProgress with all fields."""
        assert sample_round_progress.round_number == 2
        assert sample_round_progress.total_rounds == 5
        assert len(sample_round_progress.agent_messages) == 2
        assert sample_round_progress.current_consensus == "Leaning towards microservices"

    def test_round_progress_defaults(self):
        """Test RoundProgress default values."""
        progress = RoundProgress(round_number=1, total_rounds=3)

        assert progress.agent_messages == []
        assert progress.current_consensus is None

    def test_round_progress_empty_messages(self):
        """Test RoundProgress with no messages."""
        progress = RoundProgress(
            round_number=1,
            total_rounds=3,
            agent_messages=[],
        )

        assert len(progress.agent_messages) == 0


# ===========================================================================
# TeamsAdaptiveCards.get_agent_icon Tests
# ===========================================================================


class TestGetAgentIcon:
    """Tests for TeamsAdaptiveCards.get_agent_icon()."""

    def test_get_claude_icon(self):
        """Test getting Claude icon."""
        url = TeamsAdaptiveCards.get_agent_icon("Claude")
        assert "claude" in url.lower()

    def test_get_anthropic_icon(self):
        """Test getting Anthropic icon (should return Claude icon)."""
        url = TeamsAdaptiveCards.get_agent_icon("Anthropic")
        assert "claude" in url.lower()

    def test_get_gpt_icon(self):
        """Test getting GPT icon."""
        url = TeamsAdaptiveCards.get_agent_icon("GPT-4")
        assert "openai" in url.lower()

    def test_get_openai_icon(self):
        """Test getting OpenAI icon."""
        url = TeamsAdaptiveCards.get_agent_icon("OpenAI")
        assert "openai" in url.lower()

    def test_get_gemini_icon(self):
        """Test getting Gemini icon."""
        url = TeamsAdaptiveCards.get_agent_icon("Gemini")
        assert "gemini" in url.lower()

    def test_get_grok_icon(self):
        """Test getting Grok icon."""
        url = TeamsAdaptiveCards.get_agent_icon("Grok")
        assert "grok" in url.lower()

    def test_get_mistral_icon(self):
        """Test getting Mistral icon."""
        url = TeamsAdaptiveCards.get_agent_icon("Mistral-Large")
        assert "mistral" in url.lower()

    def test_get_unknown_agent_icon(self):
        """Test getting icon for unknown agent returns default."""
        url = TeamsAdaptiveCards.get_agent_icon("UnknownAgent")
        assert "agent" in url.lower() or "default" in url.lower()

    def test_case_insensitive_matching(self):
        """Test icon matching is case-insensitive."""
        url1 = TeamsAdaptiveCards.get_agent_icon("CLAUDE")
        url2 = TeamsAdaptiveCards.get_agent_icon("claude")
        url3 = TeamsAdaptiveCards.get_agent_icon("Claude")

        assert url1 == url2 == url3


# ===========================================================================
# TeamsAdaptiveCards.wrap_as_card Tests
# ===========================================================================


class TestWrapAsCard:
    """Tests for TeamsAdaptiveCards.wrap_as_card()."""

    def test_wrap_basic_body(self):
        """Test wrapping body elements as card."""
        body = [{"type": "TextBlock", "text": "Hello"}]
        card = TeamsAdaptiveCards.wrap_as_card(body)

        assert card["type"] == "AdaptiveCard"
        assert "adaptivecards.io" in card["$schema"]
        assert card["version"] == "1.5"
        assert card["body"] == body
        assert "actions" not in card

    def test_wrap_with_actions(self):
        """Test wrapping body with actions."""
        body = [{"type": "TextBlock", "text": "Hello"}]
        actions = [{"type": "Action.Submit", "title": "OK"}]
        card = TeamsAdaptiveCards.wrap_as_card(body, actions)

        assert card["actions"] == actions

    def test_wrap_empty_body(self):
        """Test wrapping empty body."""
        card = TeamsAdaptiveCards.wrap_as_card([])

        assert card["body"] == []

    def test_wrap_none_actions(self):
        """Test wrapping with None actions."""
        body = [{"type": "TextBlock", "text": "Hello"}]
        card = TeamsAdaptiveCards.wrap_as_card(body, None)

        assert "actions" not in card


# ===========================================================================
# TeamsAdaptiveCards.starting_card Tests
# ===========================================================================


class TestStartingCard:
    """Tests for TeamsAdaptiveCards.starting_card()."""

    def test_starting_card_structure(self):
        """Test starting card has required structure."""
        card = TeamsAdaptiveCards.starting_card(
            topic="Should we use microservices?",
            initiated_by="User123",
            agents=["Claude", "GPT-4", "Gemini"],
        )

        assert card["type"] == "AdaptiveCard"
        assert len(card["body"]) > 0
        assert len(card["actions"]) > 0

    def test_starting_card_shows_topic(self):
        """Test starting card displays topic."""
        card = TeamsAdaptiveCards.starting_card(
            topic="Important Question",
            initiated_by="User",
            agents=["Claude"],
        )

        card_str = str(card)
        assert "Important Question" in card_str

    def test_starting_card_shows_initiated_by(self):
        """Test starting card shows who initiated."""
        card = TeamsAdaptiveCards.starting_card(
            topic="Topic",
            initiated_by="TestUser",
            agents=["Claude"],
        )

        card_str = str(card)
        assert "TestUser" in card_str

    def test_starting_card_shows_agents(self):
        """Test starting card lists agents."""
        agents = ["Claude", "GPT-4", "Gemini"]
        card = TeamsAdaptiveCards.starting_card(
            topic="Topic",
            initiated_by="User",
            agents=agents,
        )

        card_str = str(card)
        assert "Claude" in card_str
        assert "GPT-4" in card_str
        assert "Gemini" in card_str

    def test_starting_card_limits_agents_to_4(self):
        """Test starting card limits agent icons to 4."""
        agents = ["Agent1", "Agent2", "Agent3", "Agent4", "Agent5", "Agent6"]
        card = TeamsAdaptiveCards.starting_card(
            topic="Topic",
            initiated_by="User",
            agents=agents,
        )

        # Find the column with agent icons
        # The card should limit icons but still list all agents in text
        card_str = str(card)
        # All agents should appear in the text list
        assert "Agent5" in card_str

    def test_starting_card_has_cancel_action(self):
        """Test starting card has cancel action."""
        card = TeamsAdaptiveCards.starting_card(
            topic="Topic",
            initiated_by="User",
            agents=["Claude"],
            debate_id="debate-123",
        )

        actions = card["actions"]
        assert any("Cancel" in a.get("title", "") for a in actions)

    def test_starting_card_cancel_action_includes_debate_id(self):
        """Test cancel action includes debate_id."""
        card = TeamsAdaptiveCards.starting_card(
            topic="Topic",
            initiated_by="User",
            agents=["Claude"],
            debate_id="debate-xyz",
        )

        cancel_action = next(a for a in card["actions"] if "Cancel" in a.get("title", ""))
        assert cancel_action["data"]["debate_id"] == "debate-xyz"

    def test_starting_card_shows_deliberating_message(self):
        """Test starting card shows agents are deliberating."""
        card = TeamsAdaptiveCards.starting_card(
            topic="Topic",
            initiated_by="User",
            agents=["Claude"],
        )

        card_str = str(card)
        assert "deliberating" in card_str.lower()


# ===========================================================================
# TeamsAdaptiveCards.progress_card Tests
# ===========================================================================


class TestProgressCard:
    """Tests for TeamsAdaptiveCards.progress_card()."""

    def test_progress_card_structure(self, sample_round_progress):
        """Test progress card has required structure."""
        card = TeamsAdaptiveCards.progress_card(
            topic="Should we use microservices?",
            progress=sample_round_progress,
        )

        assert card["type"] == "AdaptiveCard"
        assert len(card["body"]) > 0

    def test_progress_card_shows_round_info(self, sample_round_progress):
        """Test progress card shows round information."""
        card = TeamsAdaptiveCards.progress_card(
            topic="Topic",
            progress=sample_round_progress,
        )

        card_str = str(card)
        assert "2" in card_str  # round_number
        assert "5" in card_str  # total_rounds

    def test_progress_card_shows_topic(self, sample_round_progress):
        """Test progress card displays topic."""
        card = TeamsAdaptiveCards.progress_card(
            topic="My Important Topic",
            progress=sample_round_progress,
        )

        card_str = str(card)
        assert "My Important Topic" in card_str

    def test_progress_card_shows_agent_messages(self, sample_round_progress):
        """Test progress card shows agent messages."""
        card = TeamsAdaptiveCards.progress_card(
            topic="Topic",
            progress=sample_round_progress,
        )

        card_str = str(card)
        assert "Claude" in card_str
        assert "GPT-4" in card_str

    def test_progress_card_limits_messages_to_3(self):
        """Test progress card limits to last 3 messages."""
        progress = RoundProgress(
            round_number=3,
            total_rounds=5,
            agent_messages=[
                {"agent": "Agent1", "summary": "Message 1"},
                {"agent": "Agent2", "summary": "Message 2"},
                {"agent": "Agent3", "summary": "Message 3"},
                {"agent": "Agent4", "summary": "Message 4"},
                {"agent": "Agent5", "summary": "Message 5"},
            ],
        )

        card = TeamsAdaptiveCards.progress_card(topic="Topic", progress=progress)

        # Only last 3 messages should be shown
        card_str = str(card)
        # Agent5 should definitely be there (most recent)
        assert "Agent5" in card_str

    def test_progress_card_shows_consensus(self):
        """Test progress card shows current consensus when available."""
        progress = RoundProgress(
            round_number=2,
            total_rounds=3,
            current_consensus="The consensus is emerging",
        )

        card = TeamsAdaptiveCards.progress_card(topic="Topic", progress=progress)

        card_str = str(card)
        assert "consensus" in card_str.lower()

    def test_progress_card_no_messages(self):
        """Test progress card handles no messages."""
        progress = RoundProgress(
            round_number=1,
            total_rounds=3,
            agent_messages=[],
        )

        card = TeamsAdaptiveCards.progress_card(topic="Topic", progress=progress)

        # Should still create valid card
        assert card["type"] == "AdaptiveCard"

    def test_progress_card_truncates_long_summaries(self):
        """Test progress card truncates long message summaries."""
        long_summary = "A" * 300
        progress = RoundProgress(
            round_number=1,
            total_rounds=3,
            agent_messages=[{"agent": "Claude", "summary": long_summary}],
        )

        card = TeamsAdaptiveCards.progress_card(topic="Topic", progress=progress)

        card_str = str(card)
        # Should not contain full 300 character string
        assert "A" * 250 not in card_str


# ===========================================================================
# TeamsAdaptiveCards.voting_card Tests
# ===========================================================================


class TestVotingCard:
    """Tests for TeamsAdaptiveCards.voting_card()."""

    def test_voting_card_structure(self):
        """Test voting card has required structure."""
        card = TeamsAdaptiveCards.voting_card(
            topic="Should we proceed?",
            verdict="Yes, we should proceed",
            debate_id="debate-123",
        )

        assert card["type"] == "AdaptiveCard"
        assert len(card["body"]) > 0
        assert len(card["actions"]) >= 3

    def test_voting_card_shows_topic(self):
        """Test voting card displays topic."""
        card = TeamsAdaptiveCards.voting_card(
            topic="Important Decision",
            verdict="Verdict",
            debate_id="debate-123",
        )

        card_str = str(card)
        assert "Important Decision" in card_str

    def test_voting_card_shows_verdict(self):
        """Test voting card displays AI verdict."""
        card = TeamsAdaptiveCards.voting_card(
            topic="Topic",
            verdict="The AI recommends proceeding",
            debate_id="debate-123",
        )

        card_str = str(card)
        assert "AI recommends proceeding" in card_str

    def test_voting_card_default_options(self):
        """Test voting card has default vote options."""
        card = TeamsAdaptiveCards.voting_card(
            topic="Topic",
            verdict="Verdict",
            debate_id="debate-123",
        )

        actions = card["actions"]
        action_titles = [a["title"] for a in actions]

        assert "Agree" in action_titles
        assert "Disagree" in action_titles
        assert "Abstain" in action_titles

    def test_voting_card_custom_options(self):
        """Test voting card with custom options."""
        card = TeamsAdaptiveCards.voting_card(
            topic="Topic",
            verdict="Verdict",
            debate_id="debate-123",
            options=["Option A", "Option B", "Option C"],
        )

        actions = card["actions"]
        action_titles = [a["title"] for a in actions]

        assert "Option A" in action_titles
        assert "Option B" in action_titles
        assert "Option C" in action_titles

    def test_voting_card_action_styles(self):
        """Test voting card actions have appropriate styles."""
        card = TeamsAdaptiveCards.voting_card(
            topic="Topic",
            verdict="Verdict",
            debate_id="debate-123",
        )

        actions = card["actions"]

        agree_action = next(a for a in actions if a["title"] == "Agree")
        disagree_action = next(a for a in actions if a["title"] == "Disagree")

        assert agree_action.get("style") == "positive"
        assert disagree_action.get("style") == "destructive"

    def test_voting_card_action_data(self):
        """Test voting card actions include proper data."""
        card = TeamsAdaptiveCards.voting_card(
            topic="Topic",
            verdict="Verdict",
            debate_id="debate-xyz",
        )

        for action in card["actions"]:
            assert action["data"]["action"] == "vote"
            assert action["data"]["debate_id"] == "debate-xyz"
            assert "vote" in action["data"]


# ===========================================================================
# TeamsAdaptiveCards.verdict_card Tests
# ===========================================================================


class TestVerdictCard:
    """Tests for TeamsAdaptiveCards.verdict_card()."""

    def test_verdict_card_structure(self, sample_agents):
        """Test verdict card has required structure."""
        card = TeamsAdaptiveCards.verdict_card(
            topic="Should we use microservices?",
            verdict="Yes, for services needing independent scaling",
            confidence=0.85,
            agents=sample_agents,
        )

        assert card["type"] == "AdaptiveCard"
        assert len(card["body"]) > 0

    def test_verdict_card_shows_topic(self, sample_agents):
        """Test verdict card displays topic."""
        card = TeamsAdaptiveCards.verdict_card(
            topic="My Important Topic",
            verdict="Verdict",
            confidence=0.8,
            agents=sample_agents,
        )

        card_str = str(card)
        assert "My Important Topic" in card_str

    def test_verdict_card_shows_verdict(self, sample_agents):
        """Test verdict card displays verdict."""
        card = TeamsAdaptiveCards.verdict_card(
            topic="Topic",
            verdict="The final verdict is clear",
            confidence=0.9,
            agents=sample_agents,
        )

        card_str = str(card)
        assert "final verdict is clear" in card_str

    def test_verdict_card_shows_confidence(self, sample_agents):
        """Test verdict card displays confidence."""
        card = TeamsAdaptiveCards.verdict_card(
            topic="Topic",
            verdict="Verdict",
            confidence=0.85,
            agents=sample_agents,
        )

        card_str = str(card)
        assert "85%" in card_str

    def test_verdict_card_confidence_color_high(self, sample_agents):
        """Test high confidence shows Good color."""
        card = TeamsAdaptiveCards.verdict_card(
            topic="Topic",
            verdict="Verdict",
            confidence=0.75,
            agents=sample_agents,
        )

        card_str = str(card)
        assert "Good" in card_str

    def test_verdict_card_confidence_color_medium(self, sample_agents):
        """Test medium confidence shows Warning color."""
        card = TeamsAdaptiveCards.verdict_card(
            topic="Topic",
            verdict="Verdict",
            confidence=0.55,
            agents=sample_agents,
        )

        card_str = str(card)
        assert "Warning" in card_str

    def test_verdict_card_confidence_color_low(self, sample_agents):
        """Test low confidence shows Attention color."""
        card = TeamsAdaptiveCards.verdict_card(
            topic="Topic",
            verdict="Verdict",
            confidence=0.3,
            agents=sample_agents,
        )

        card_str = str(card)
        assert "Attention" in card_str

    def test_verdict_card_shows_rounds(self, sample_agents):
        """Test verdict card shows rounds completed."""
        card = TeamsAdaptiveCards.verdict_card(
            topic="Topic",
            verdict="Verdict",
            confidence=0.8,
            agents=sample_agents,
            rounds_completed=5,
        )

        card_str = str(card)
        assert "5" in card_str

    def test_verdict_card_shows_agent_count(self, sample_agents):
        """Test verdict card shows agent count."""
        card = TeamsAdaptiveCards.verdict_card(
            topic="Topic",
            verdict="Verdict",
            confidence=0.8,
            agents=sample_agents,
        )

        card_str = str(card)
        assert "3" in card_str  # 3 agents

    def test_verdict_card_shows_agent_breakdown(self, sample_agents):
        """Test verdict card shows for/against breakdown."""
        card = TeamsAdaptiveCards.verdict_card(
            topic="Topic",
            verdict="Verdict",
            confidence=0.8,
            agents=sample_agents,
        )

        card_str = str(card)
        assert "FOR" in card_str
        assert "AGAINST" in card_str
        assert "Claude" in card_str
        assert "Gemini" in card_str

    def test_verdict_card_shows_key_points(self, sample_agents):
        """Test verdict card shows agent key points."""
        card = TeamsAdaptiveCards.verdict_card(
            topic="Topic",
            verdict="Verdict",
            confidence=0.8,
            agents=sample_agents,
        )

        card_str = str(card)
        assert "Better scalability" in card_str
        assert "Complexity cost" in card_str

    def test_verdict_card_with_receipt_id(self, sample_agents):
        """Test verdict card has receipt action when receipt_id provided."""
        card = TeamsAdaptiveCards.verdict_card(
            topic="Topic",
            verdict="Verdict",
            confidence=0.8,
            agents=sample_agents,
            receipt_id="rec_abc123",
        )

        actions = card.get("actions", [])
        receipt_action = next((a for a in actions if "Receipt" in a.get("title", "")), None)
        assert receipt_action is not None
        assert "rec_abc123" in receipt_action["url"]

    def test_verdict_card_with_debate_id(self, sample_agents):
        """Test verdict card has voting actions when debate_id provided."""
        card = TeamsAdaptiveCards.verdict_card(
            topic="Topic",
            verdict="Verdict",
            confidence=0.8,
            agents=sample_agents,
            debate_id="debate-123",
        )

        actions = card.get("actions", [])
        agree_action = next((a for a in actions if a.get("title") == "Agree"), None)
        disagree_action = next((a for a in actions if a.get("title") == "Disagree"), None)

        assert agree_action is not None
        assert disagree_action is not None


# ===========================================================================
# TeamsAdaptiveCards.error_card Tests
# ===========================================================================


class TestErrorCard:
    """Tests for TeamsAdaptiveCards.error_card()."""

    def test_error_card_structure(self):
        """Test error card has required structure."""
        card = TeamsAdaptiveCards.error_card(
            title="Error Occurred",
            message="Something went wrong.",
        )

        assert card["type"] == "AdaptiveCard"
        assert len(card["body"]) > 0
        assert len(card["actions"]) > 0

    def test_error_card_shows_title(self):
        """Test error card displays title."""
        card = TeamsAdaptiveCards.error_card(
            title="Connection Failed",
            message="Message",
        )

        card_str = str(card)
        assert "Connection Failed" in card_str

    def test_error_card_shows_message(self):
        """Test error card displays message."""
        card = TeamsAdaptiveCards.error_card(
            title="Error",
            message="Please try again later.",
        )

        card_str = str(card)
        assert "Please try again later" in card_str

    def test_error_card_with_suggestions(self):
        """Test error card shows troubleshooting suggestions."""
        suggestions = [
            "Check your network connection",
            "Try refreshing the page",
            "Contact support if the issue persists",
        ]
        card = TeamsAdaptiveCards.error_card(
            title="Error",
            message="Error message",
            suggestions=suggestions,
        )

        card_str = str(card)
        assert "Check your network" in card_str
        assert "Try refreshing" in card_str

    def test_error_card_with_retry_action(self):
        """Test error card has retry action when provided."""
        card = TeamsAdaptiveCards.error_card(
            title="Error",
            message="Error message",
            retry_action={"action": "retry", "debate_id": "123"},
        )

        actions = card["actions"]
        retry_action = next((a for a in actions if a.get("title") == "Retry"), None)
        assert retry_action is not None

    def test_error_card_always_has_help_action(self):
        """Test error card always has Get Help action."""
        card = TeamsAdaptiveCards.error_card(
            title="Error",
            message="Error message",
        )

        actions = card["actions"]
        help_action = next((a for a in actions if "Help" in a.get("title", "")), None)
        assert help_action is not None


# ===========================================================================
# TeamsAdaptiveCards.receipt_card Tests
# ===========================================================================


class TestReceiptCard:
    """Tests for TeamsAdaptiveCards.receipt_card()."""

    def test_receipt_card_structure(self):
        """Test receipt card has required structure."""
        card = TeamsAdaptiveCards.receipt_card(
            receipt_id="rec_abc123def456",
            topic="Should we use microservices?",
            verdict="Yes, proceed with microservices",
            timestamp="2025-01-15T14:30:00Z",
            hash_preview="abc123def456789",
        )

        assert card["type"] == "AdaptiveCard"
        assert len(card["body"]) > 0
        assert len(card["actions"]) >= 1

    def test_receipt_card_shows_receipt_id(self):
        """Test receipt card displays receipt ID (truncated)."""
        card = TeamsAdaptiveCards.receipt_card(
            receipt_id="rec_abc123def456ghi789",
            topic="Topic",
            verdict="Verdict",
            timestamp="2025-01-15T14:30:00Z",
            hash_preview="hash123",
        )

        card_str = str(card)
        # Should show truncated version
        assert "rec_abc123def456" in card_str

    def test_receipt_card_shows_topic(self):
        """Test receipt card displays topic (truncated if long)."""
        card = TeamsAdaptiveCards.receipt_card(
            receipt_id="rec_123",
            topic="This is a very long topic that should be truncated after fifty characters",
            verdict="Verdict",
            timestamp="2025-01-15T14:30:00Z",
            hash_preview="hash123",
        )

        card_str = str(card)
        assert "This is a very long" in card_str

    def test_receipt_card_shows_timestamp(self):
        """Test receipt card displays timestamp."""
        card = TeamsAdaptiveCards.receipt_card(
            receipt_id="rec_123",
            topic="Topic",
            verdict="Verdict",
            timestamp="2025-01-15T14:30:00Z",
            hash_preview="hash123",
        )

        card_str = str(card)
        assert "2025-01-15" in card_str

    def test_receipt_card_shows_hash(self):
        """Test receipt card displays hash (truncated)."""
        card = TeamsAdaptiveCards.receipt_card(
            receipt_id="rec_123",
            topic="Topic",
            verdict="Verdict",
            timestamp="2025-01-15T14:30:00Z",
            hash_preview="abc123def456ghi789jkl",
        )

        card_str = str(card)
        assert "abc123def456ghi7" in card_str

    def test_receipt_card_shows_verdict(self):
        """Test receipt card displays verdict."""
        card = TeamsAdaptiveCards.receipt_card(
            receipt_id="rec_123",
            topic="Topic",
            verdict="The final decision was to proceed",
            timestamp="2025-01-15T14:30:00Z",
            hash_preview="hash123",
        )

        card_str = str(card)
        assert "final decision was to proceed" in card_str

    def test_receipt_card_has_view_receipt_action(self):
        """Test receipt card has View Full Receipt action."""
        card = TeamsAdaptiveCards.receipt_card(
            receipt_id="rec_xyz789",
            topic="Topic",
            verdict="Verdict",
            timestamp="2025-01-15T14:30:00Z",
            hash_preview="hash123",
        )

        actions = card["actions"]
        view_action = next((a for a in actions if "Receipt" in a.get("title", "")), None)
        assert view_action is not None
        assert "rec_xyz789" in view_action["url"]

    def test_receipt_card_with_verification_url(self):
        """Test receipt card has Verify Integrity action when URL provided."""
        card = TeamsAdaptiveCards.receipt_card(
            receipt_id="rec_123",
            topic="Topic",
            verdict="Verdict",
            timestamp="2025-01-15T14:30:00Z",
            hash_preview="hash123",
            verification_url="https://verify.example.com/abc123",
        )

        actions = card["actions"]
        verify_action = next((a for a in actions if "Verify" in a.get("title", "")), None)
        assert verify_action is not None
        assert verify_action["url"] == "https://verify.example.com/abc123"


# ===========================================================================
# Module Exports Tests
# ===========================================================================


class TestExports:
    """Tests for module __all__ exports."""

    def test_all_exports(self):
        """Test all public classes are exported."""
        from aragora.connectors.chat import teams_adaptive_cards

        expected = ["TeamsAdaptiveCards", "AgentContribution", "RoundProgress"]

        for name in expected:
            assert name in teams_adaptive_cards.__all__
            assert hasattr(teams_adaptive_cards, name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
