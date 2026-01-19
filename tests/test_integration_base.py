"""
Tests for integration base class utilities.

Covers:
- Text truncation
- Agent list formatting
- Confidence formatting
- Debate data extraction
- HTML escaping
"""

import pytest

from aragora.core import DebateResult
from aragora.integrations.base import (
    BaseIntegration,
    FormattedConsensusData,
    FormattedDebateData,
    FormattedErrorData,
    FormattedLeaderboardData,
)


# =============================================================================
# Concrete Implementation for Testing
# =============================================================================


class ConcreteIntegration(BaseIntegration):
    """Concrete implementation for testing abstract base class."""

    def __init__(self, configured: bool = True):
        super().__init__()
        self._configured = configured

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def send_message(self, content: str, **kwargs) -> bool:
        return self._configured


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def integration():
    """Create a concrete integration instance."""
    return ConcreteIntegration()


@pytest.fixture
def sample_debate_result():
    """Sample DebateResult for testing."""
    result = DebateResult(
        task="What is the best programming language?",
        final_answer="Python is widely considered excellent.",
        consensus_reached=True,
        rounds_used=3,
        winner="claude",
        confidence=0.85,
    )
    result.debate_id = "test-debate-123"
    result.question = "What is the best programming language?"
    result.answer = "Python is widely considered excellent."
    result.total_rounds = 3
    result.consensus_confidence = 0.85
    result.participating_agents = ["claude", "gpt-4", "gemini"]
    return result


# =============================================================================
# Text Truncation Tests
# =============================================================================


class TestTruncateText:
    """Tests for truncate_text utility."""

    def test_short_text_unchanged(self, integration):
        """Test that short text is not truncated."""
        result = integration.truncate_text("Hello", 10)
        assert result == "Hello"

    def test_exact_length_unchanged(self, integration):
        """Test that text at exact limit is not truncated."""
        result = integration.truncate_text("Hello", 5)
        assert result == "Hello"

    def test_long_text_truncated(self, integration):
        """Test that long text is truncated with suffix."""
        result = integration.truncate_text("Hello World", 8)
        assert result == "Hello..."
        assert len(result) == 8

    def test_custom_suffix(self, integration):
        """Test truncation with custom suffix."""
        result = integration.truncate_text("Hello World", 9, suffix="…")
        assert result == "Hello Wo…"

    def test_empty_string(self, integration):
        """Test truncation of empty string."""
        result = integration.truncate_text("", 10)
        assert result == ""


# =============================================================================
# Agent List Formatting Tests
# =============================================================================


class TestFormatAgentsList:
    """Tests for format_agents_list utility."""

    def test_empty_list(self, integration):
        """Test formatting empty agent list."""
        result = integration.format_agents_list([])
        assert result == ""

    def test_single_agent(self, integration):
        """Test formatting single agent."""
        result = integration.format_agents_list(["claude"])
        assert result == "claude"

    def test_few_agents(self, integration):
        """Test formatting few agents under limit."""
        result = integration.format_agents_list(["claude", "gpt-4", "gemini"])
        assert result == "claude, gpt-4, gemini"

    def test_at_limit(self, integration):
        """Test formatting agents at exact limit."""
        agents = ["a1", "a2", "a3", "a4", "a5"]
        result = integration.format_agents_list(agents, limit=5)
        assert result == "a1, a2, a3, a4, a5"
        assert "+0 more" not in result

    def test_over_limit(self, integration):
        """Test formatting agents over limit."""
        agents = ["a1", "a2", "a3", "a4", "a5", "a6", "a7"]
        result = integration.format_agents_list(agents, limit=5)
        assert result == "a1, a2, a3, a4, a5 +2 more"

    def test_custom_separator(self, integration):
        """Test formatting with custom separator."""
        result = integration.format_agents_list(["a", "b", "c"], separator=" | ")
        assert result == "a | b | c"

    def test_custom_limit(self, integration):
        """Test formatting with custom limit."""
        agents = ["a", "b", "c", "d"]
        result = integration.format_agents_list(agents, limit=2)
        assert result == "a, b +2 more"


# =============================================================================
# Confidence Formatting Tests
# =============================================================================


class TestFormatConfidence:
    """Tests for format_confidence utility."""

    def test_high_confidence(self, integration):
        """Test formatting high confidence."""
        result = integration.format_confidence(0.95)
        assert result == "95%"

    def test_medium_confidence(self, integration):
        """Test formatting medium confidence."""
        result = integration.format_confidence(0.75)
        assert result == "75%"

    def test_low_confidence(self, integration):
        """Test formatting low confidence."""
        result = integration.format_confidence(0.5)
        assert result == "50%"

    def test_zero_confidence(self, integration):
        """Test formatting zero confidence."""
        result = integration.format_confidence(0.0)
        assert result == "0%"

    def test_full_confidence(self, integration):
        """Test formatting 100% confidence."""
        result = integration.format_confidence(1.0)
        assert result == "100%"


class TestGetConfidenceColor:
    """Tests for get_confidence_color utility."""

    def test_high_confidence_green(self, integration):
        """Test high confidence returns green."""
        result = integration.get_confidence_color(0.9)
        assert result == "green"

    def test_at_threshold_green(self, integration):
        """Test confidence at threshold returns green."""
        result = integration.get_confidence_color(0.8)
        assert result == "green"

    def test_below_threshold_orange(self, integration):
        """Test confidence below threshold returns orange."""
        result = integration.get_confidence_color(0.79)
        assert result == "orange"

    def test_low_confidence_orange(self, integration):
        """Test low confidence returns orange."""
        result = integration.get_confidence_color(0.5)
        assert result == "orange"

    def test_custom_threshold(self, integration):
        """Test custom threshold."""
        result = integration.get_confidence_color(0.85, threshold=0.9)
        assert result == "orange"


# =============================================================================
# URL Generation Tests
# =============================================================================


class TestURLGeneration:
    """Tests for URL generation utilities."""

    def test_debate_url(self, integration):
        """Test debate URL generation."""
        result = integration.get_debate_url("abc123")
        assert result == "https://aragora.ai/debate/abc123"

    def test_leaderboard_url(self, integration):
        """Test leaderboard URL generation."""
        result = integration.get_leaderboard_url()
        assert result == "https://aragora.ai/leaderboard"


# =============================================================================
# Debate Data Formatting Tests
# =============================================================================


class TestFormatDebateData:
    """Tests for format_debate_data method."""

    def test_basic_formatting(self, integration, sample_debate_result):
        """Test basic debate data formatting."""
        data = integration.format_debate_data(sample_debate_result)

        assert isinstance(data, FormattedDebateData)
        assert data.debate_id == "test-debate-123"
        assert data.question == sample_debate_result.question
        assert data.total_rounds == 3
        assert data.confidence == 0.85
        assert data.confidence_percent == "85%"
        assert data.agent_count == 3

    def test_question_truncation(self, integration, sample_debate_result):
        """Test question is truncated to limit."""
        sample_debate_result.question = "A" * 300
        data = integration.format_debate_data(sample_debate_result, question_limit=100)

        assert len(data.question_truncated) == 100
        assert data.question_truncated.endswith("...")

    def test_answer_truncation(self, integration, sample_debate_result):
        """Test answer is truncated to limit."""
        sample_debate_result.answer = "B" * 600
        data = integration.format_debate_data(sample_debate_result, answer_limit=200)

        assert len(data.answer_truncated) == 200
        assert data.answer_truncated.endswith("...")

    def test_agents_display(self, integration, sample_debate_result):
        """Test agents are formatted correctly."""
        data = integration.format_debate_data(sample_debate_result)
        assert "claude" in data.agents_display
        assert "gpt-4" in data.agents_display
        assert "gemini" in data.agents_display

    def test_agents_truncation(self, integration, sample_debate_result):
        """Test agents list is truncated."""
        sample_debate_result.participating_agents = ["a1", "a2", "a3", "a4", "a5", "a6"]
        data = integration.format_debate_data(sample_debate_result, agents_limit=3)

        assert "+3 more" in data.agents_display

    def test_stats_line(self, integration, sample_debate_result):
        """Test stats line formatting."""
        data = integration.format_debate_data(sample_debate_result)

        assert "Rounds: 3" in data.stats_line
        assert "85%" in data.stats_line
        assert "Agents: 3" in data.stats_line

    def test_debate_url(self, integration, sample_debate_result):
        """Test debate URL is generated."""
        data = integration.format_debate_data(sample_debate_result)
        assert data.debate_url == "https://aragora.ai/debate/test-debate-123"

    def test_no_answer(self, integration, sample_debate_result):
        """Test handling when answer is None."""
        sample_debate_result.answer = None
        data = integration.format_debate_data(sample_debate_result)

        assert data.answer is None
        assert data.answer_truncated is None

    def test_no_confidence(self, integration, sample_debate_result):
        """Test handling when confidence is None."""
        sample_debate_result.consensus_confidence = None
        data = integration.format_debate_data(sample_debate_result)

        assert data.confidence is None
        assert data.confidence_percent is None


# =============================================================================
# Consensus Data Formatting Tests
# =============================================================================


class TestFormatConsensusData:
    """Tests for format_consensus_data method."""

    def test_basic_formatting(self, integration):
        """Test basic consensus data formatting."""
        data = integration.format_consensus_data(
            debate_id="test-123",
            answer="The answer is clear",
            confidence=0.85,
            agents=["claude", "gpt-4"],
        )

        assert isinstance(data, FormattedConsensusData)
        assert data.debate_id == "test-123"
        assert data.answer == "The answer is clear"
        assert data.confidence == 0.85
        assert data.confidence_percent == "85%"
        assert data.confidence_color == "green"

    def test_answer_truncation(self, integration):
        """Test answer truncation."""
        long_answer = "A" * 600
        data = integration.format_consensus_data(
            debate_id="test",
            answer=long_answer,
            confidence=0.8,
            answer_limit=100,
        )

        assert len(data.answer_truncated) == 100

    def test_agents_display(self, integration):
        """Test agents display formatting."""
        data = integration.format_consensus_data(
            debate_id="test",
            answer="Answer",
            confidence=0.8,
            agents=["a1", "a2", "a3", "a4", "a5", "a6"],
            agents_limit=3,
        )

        assert "+3 more" in data.agents_display

    def test_low_confidence_color(self, integration):
        """Test low confidence gets orange color."""
        data = integration.format_consensus_data(
            debate_id="test",
            answer="Answer",
            confidence=0.7,
        )

        assert data.confidence_color == "orange"


# =============================================================================
# Error Data Formatting Tests
# =============================================================================


class TestFormatErrorData:
    """Tests for format_error_data method."""

    def test_basic_formatting(self, integration):
        """Test basic error data formatting."""
        data = integration.format_error_data(
            debate_id="test-123",
            error="Something went wrong",
            phase="consensus",
        )

        assert isinstance(data, FormattedErrorData)
        assert data.debate_id == "test-123"
        assert data.error == "Something went wrong"
        assert data.phase == "consensus"

    def test_error_truncation(self, integration):
        """Test error message truncation."""
        long_error = "E" * 600
        data = integration.format_error_data(
            debate_id="test",
            error=long_error,
            error_limit=100,
        )

        assert len(data.error_truncated) == 100

    def test_no_phase(self, integration):
        """Test error without phase."""
        data = integration.format_error_data(
            debate_id="test",
            error="Error",
        )

        assert data.phase is None


# =============================================================================
# Leaderboard Data Formatting Tests
# =============================================================================


class TestFormatLeaderboardData:
    """Tests for format_leaderboard_data method."""

    def test_basic_formatting(self, integration):
        """Test basic leaderboard data formatting."""
        rankings = [
            {"name": "claude", "elo": 1650, "wins": 10, "losses": 5},
            {"name": "gpt-4", "elo": 1600, "wins": 8, "losses": 7},
        ]
        data = integration.format_leaderboard_data(rankings)

        assert isinstance(data, FormattedLeaderboardData)
        assert data.title == "LEADERBOARD UPDATE"
        assert data.domain is None
        assert len(data.rankings) == 2

    def test_with_domain(self, integration):
        """Test leaderboard with domain filter."""
        data = integration.format_leaderboard_data([], domain="math")

        assert "(math)" in data.title

    def test_ranking_limit(self, integration):
        """Test ranking limit."""
        rankings = [{"name": f"a{i}", "elo": 1500} for i in range(15)]
        data = integration.format_leaderboard_data(rankings, limit=10)

        assert len(data.rankings) == 10

    def test_leaderboard_url(self, integration):
        """Test leaderboard URL."""
        data = integration.format_leaderboard_data([])
        assert data.leaderboard_url == "https://aragora.ai/leaderboard"


# =============================================================================
# HTML Escaping Tests
# =============================================================================


class TestEscapeHTML:
    """Tests for escape_html utility."""

    def test_escape_ampersand(self, integration):
        """Test ampersand escaping."""
        result = integration.escape_html("A & B")
        assert result == "A &amp; B"

    def test_escape_less_than(self, integration):
        """Test less than escaping."""
        result = integration.escape_html("A < B")
        assert result == "A &lt; B"

    def test_escape_greater_than(self, integration):
        """Test greater than escaping."""
        result = integration.escape_html("A > B")
        assert result == "A &gt; B"

    def test_escape_quotes(self, integration):
        """Test quote escaping."""
        result = integration.escape_html('"test"')
        assert result == "&quot;test&quot;"

    def test_escape_script_tag(self, integration):
        """Test XSS prevention with script tag."""
        result = integration.escape_html('<script>alert("xss")</script>')
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    def test_allows_within_limit(self, integration):
        """Test messages allowed within limit."""
        for _ in range(5):
            result = integration._check_rate_limit(max_per_minute=10)
            assert result is True

    def test_blocks_over_limit(self, integration):
        """Test messages blocked over limit."""
        for _ in range(5):
            integration._check_rate_limit(max_per_minute=5)

        result = integration._check_rate_limit(max_per_minute=5)
        assert result is False


# =============================================================================
# Session Management Tests
# =============================================================================


class TestSessionManagement:
    """Tests for session management."""

    @pytest.mark.asyncio
    async def test_get_session_creates_new(self):
        """Test session is created when none exists."""
        integration = ConcreteIntegration()
        assert integration._session is None

        session = await integration._get_session()
        assert session is not None

        await integration.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with ConcreteIntegration() as integration:
            assert integration is not None
            session = await integration._get_session()
            assert session is not None
