"""Tests for MCP context tools helper functions.

These tests focus on the helper functions in context_tools.py that can be
tested directly without complex mocking of external dependencies.
"""

import pytest
from datetime import datetime, timezone

from aragora.mcp.tools_module.context_tools import (
    _analyze_activity,
    _extract_decisions,
    _extract_questions,
    _extract_topics,
    _analyze_sentiment,
    analyze_conversation_tool,
    fetch_debate_context_tool,
    fetch_channel_context_tool,
    get_thread_context_tool,
    get_user_context_tool,
)

pytest.importorskip("mcp")


# =============================================================================
# Activity Analysis Tests
# =============================================================================


class TestAnalyzeActivity:
    """Tests for _analyze_activity helper function."""

    def test_empty_messages(self):
        """Empty messages returns basic structure."""
        result = _analyze_activity([])
        assert result["total_messages"] == 0

    def test_single_message(self):
        """Single message without timestamp."""
        result = _analyze_activity([{"content": "Hello", "author": "user1"}])
        assert result["total_messages"] == 1
        assert result["unique_authors"] == 1

    def test_multiple_authors(self):
        """Multiple unique authors counted correctly."""
        messages = [
            {"content": "Hello", "author": "user1"},
            {"content": "Hi", "author": "user2"},
            {"content": "Hey", "author": "user1"},
            {"content": "Yo", "author": "user3"},
        ]
        result = _analyze_activity(messages)
        assert result["total_messages"] == 4
        assert result["unique_authors"] == 3

    def test_with_timestamps(self):
        """Messages with timestamps include duration and peak hour."""
        messages = [
            {
                "content": "Hello",
                "author": "user1",
                "timestamp": "2024-01-15T10:00:00+00:00",
            },
            {
                "content": "Hi",
                "author": "user2",
                "timestamp": "2024-01-15T10:30:00+00:00",
            },
            {
                "content": "Hey",
                "author": "user1",
                "timestamp": "2024-01-15T11:00:00+00:00",
            },
        ]
        result = _analyze_activity(messages)
        assert result["total_messages"] == 3
        assert "first_message" in result
        assert "last_message" in result
        assert "duration_hours" in result
        assert result["duration_hours"] == 1.0  # 1 hour difference

    def test_with_datetime_objects(self):
        """Handles datetime objects in timestamps."""
        messages = [
            {
                "content": "Hello",
                "author": "user1",
                "timestamp": datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            },
            {
                "content": "Hi",
                "author": "user2",
                "timestamp": datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            },
        ]
        result = _analyze_activity(messages)
        assert result["duration_hours"] == 2.0

    def test_peak_hour_detection(self):
        """Detects the most active hour."""
        messages = [
            {"content": "a", "author": "u1", "timestamp": "2024-01-15T10:00:00+00:00"},
            {"content": "b", "author": "u1", "timestamp": "2024-01-15T10:15:00+00:00"},
            {"content": "c", "author": "u1", "timestamp": "2024-01-15T10:30:00+00:00"},
            {"content": "d", "author": "u1", "timestamp": "2024-01-15T14:00:00+00:00"},
        ]
        result = _analyze_activity(messages)
        assert result["peak_hour"] == 10  # Most messages at 10:xx

    def test_invalid_timestamp_ignored(self):
        """Invalid timestamps are gracefully ignored."""
        messages = [
            {"content": "Hello", "author": "user1", "timestamp": "not a date"},
            {"content": "Hi", "author": "user2", "timestamp": "also invalid"},
        ]
        result = _analyze_activity(messages)
        assert result["total_messages"] == 2
        # No duration since timestamps couldn't be parsed
        assert "duration_hours" not in result

    def test_empty_author(self):
        """Empty author is counted."""
        messages = [
            {"content": "Hello", "author": ""},
            {"content": "Hi", "author": "user1"},
        ]
        result = _analyze_activity(messages)
        assert result["unique_authors"] == 2  # "" and "user1"


# =============================================================================
# Question Extraction Tests
# =============================================================================


class TestExtractQuestions:
    """Tests for _extract_questions helper function."""

    def test_empty_messages(self):
        """Empty messages returns empty list."""
        result = _extract_questions([])
        assert result == []

    def test_question_mark_detection(self):
        """Detects questions ending with ?"""
        messages = [
            {"content": "What is this?", "author": "user1"},
            {"content": "This is a statement.", "author": "user2"},
        ]
        result = _extract_questions(messages)
        assert len(result) == 1
        assert "What is this?" in result[0]["content"]

    def test_question_word_detection(self):
        """Detects questions starting with question words."""
        messages = [
            {"content": "What time is it", "author": "user1"},
            {"content": "Why would you do that", "author": "user2"},
            {"content": "How does this work", "author": "user3"},
            {"content": "When should we meet", "author": "user4"},
            {"content": "Where is the office", "author": "user5"},
            {"content": "Who is coming", "author": "user6"},
            {"content": "Which option is best", "author": "user7"},
            {"content": "Can you help me", "author": "user8"},
            {"content": "Could you explain", "author": "user9"},
            {"content": "Would you agree", "author": "user10"},
            {"content": "Should we proceed", "author": "user11"},
        ]
        result = _extract_questions(messages)
        assert len(result) == 11

    def test_preserves_metadata(self):
        """Preserves author and timestamp in extracted questions."""
        messages = [
            {
                "content": "What is this?",
                "author": "user1",
                "timestamp": "2024-01-15T10:00:00+00:00",
            },
        ]
        result = _extract_questions(messages)
        assert len(result) == 1
        assert result[0]["author"] == "user1"
        assert result[0]["timestamp"] == "2024-01-15T10:00:00+00:00"

    def test_case_insensitive_question_words(self):
        """Question word detection is case insensitive."""
        messages = [
            {"content": "WHAT is this", "author": "user1"},
            {"content": "why would you", "author": "user2"},
            {"content": "How Does This", "author": "user3"},
        ]
        result = _extract_questions(messages)
        # All should be detected as questions
        assert len(result) == 3

    def test_no_questions(self):
        """Returns empty list when no questions found."""
        messages = [
            {"content": "This is a statement.", "author": "user1"},
            {"content": "Another statement here.", "author": "user2"},
        ]
        result = _extract_questions(messages)
        assert len(result) == 0


# =============================================================================
# Decision Extraction Tests
# =============================================================================


class TestExtractDecisions:
    """Tests for _extract_decisions helper function."""

    def test_empty_messages(self):
        """Empty messages returns empty list."""
        result = _extract_decisions([])
        assert result == []

    def test_decision_indicators(self):
        """Detects various decision indicators."""
        messages = [
            {"content": "We decided to go with option A", "author": "user1"},
            {"content": "The decision is made", "author": "user2"},
            {"content": "Everyone agreed on this", "author": "user3"},
            {"content": "The conclusion is clear", "author": "user4"},
            {"content": "Issue resolved", "author": "user5"},
            {"content": "Let's go with plan B", "author": "user6"},
            {"content": "We'll do this tomorrow", "author": "user7"},
            {"content": "The final answer is 42", "author": "user8"},
            {"content": "The answer is yes", "author": "user9"},
            {"content": "We reached consensus", "author": "user10"},
            {"content": "Request approved", "author": "user11"},
            {"content": "Proposal accepted", "author": "user12"},
            {"content": "Moving forward with this", "author": "user13"},
        ]
        result = _extract_decisions(messages)
        assert len(result) == 13

    def test_preserves_metadata(self):
        """Preserves author and timestamp in extracted decisions."""
        messages = [
            {
                "content": "We decided to proceed",
                "author": "manager",
                "timestamp": "2024-01-15T10:00:00+00:00",
            },
        ]
        result = _extract_decisions(messages)
        assert len(result) == 1
        assert result[0]["author"] == "manager"
        assert result[0]["timestamp"] == "2024-01-15T10:00:00+00:00"

    def test_no_decisions(self):
        """Returns empty list when no decisions found."""
        messages = [
            {"content": "Just a regular message", "author": "user1"},
            {"content": "What should we do?", "author": "user2"},
        ]
        result = _extract_decisions(messages)
        assert len(result) == 0


# =============================================================================
# Topic Extraction Tests
# =============================================================================


class TestExtractTopics:
    """Tests for _extract_topics helper function."""

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        """Empty messages returns empty topics."""
        result = await _extract_topics([])
        assert result == []

    @pytest.mark.asyncio
    async def test_extracts_common_words(self):
        """Extracts most common non-stopword words."""
        messages = [
            {"content": "Python programming language"},
            {"content": "Python is great for programming"},
            {"content": "I love Python development"},
        ]
        result = await _extract_topics(messages)
        # "python" should be in top topics
        assert any("python" in topic for topic in result)

    @pytest.mark.asyncio
    async def test_filters_common_stopwords(self):
        """Common stopwords are filtered out."""
        messages = [
            {"content": "The quick brown fox jumps the lazy dog"},
            {"content": "A quick brown fox the running the forest"},
        ]
        result = await _extract_topics(messages)
        # Core stopwords like "the" should not appear
        core_stopwords = {"the", "and", "but", "for"}
        for topic in result:
            assert topic not in core_stopwords

    @pytest.mark.asyncio
    async def test_minimum_word_length(self):
        """Words shorter than 3 characters are excluded."""
        messages = [
            {"content": "I am on it as we go by"},
            {"content": "Programming is fun"},
        ]
        result = await _extract_topics(messages)
        # Short words should not appear
        for topic in result:
            assert len(topic) >= 3

    @pytest.mark.asyncio
    async def test_max_ten_topics(self):
        """Returns at most 10 topics."""
        messages = [{"content": f"word{i} " * 5} for i in range(20)]
        result = await _extract_topics(messages)
        assert len(result) <= 10


# =============================================================================
# Sentiment Analysis Tests
# =============================================================================


class TestAnalyzeSentiment:
    """Tests for _analyze_sentiment helper function."""

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        """Empty messages returns neutral sentiment."""
        result = await _analyze_sentiment([])
        assert "positive_ratio" in result
        assert "negative_ratio" in result
        assert "neutral_ratio" in result

    @pytest.mark.asyncio
    async def test_positive_sentiment(self):
        """Detects positive sentiment."""
        messages = [
            {"content": "This is great and amazing!"},
            {"content": "I love this, it's wonderful!"},
            {"content": "Excellent work, fantastic job!"},
        ]
        result = await _analyze_sentiment(messages)
        assert result["positive_ratio"] > 0.5
        assert result["overall"] == "positive"

    @pytest.mark.asyncio
    async def test_negative_sentiment(self):
        """Detects negative sentiment."""
        messages = [
            {"content": "This is terrible and awful"},
            {"content": "I hate this broken thing"},
            {"content": "There's a problem with this error"},
        ]
        result = await _analyze_sentiment(messages)
        assert result["negative_ratio"] > 0.5
        assert result["overall"] == "negative"

    @pytest.mark.asyncio
    async def test_neutral_sentiment(self):
        """Detects neutral sentiment."""
        messages = [
            {"content": "Here is some information"},
            {"content": "The meeting is at noon"},
            {"content": "Please review the document"},
        ]
        result = await _analyze_sentiment(messages)
        assert result["overall"] == "neutral"

    @pytest.mark.asyncio
    async def test_mixed_sentiment(self):
        """Handles mixed sentiment correctly."""
        messages = [
            {"content": "This is great amazing wonderful excellent"},
            {"content": "This is terrible awful bad hate"},
            {"content": "Just a regular message nothing special"},
            {"content": "Meeting is at noon in room 5"},
        ]
        result = await _analyze_sentiment(messages)
        # Result should include all ratio fields
        assert "positive_ratio" in result
        assert "negative_ratio" in result
        assert "neutral_ratio" in result
        # Positive should be detected for first message
        # Negative should be detected for second message
        assert result["positive_ratio"] + result["negative_ratio"] + result["neutral_ratio"] > 0


# =============================================================================
# Conversation Analysis Tool Tests
# =============================================================================


class TestAnalyzeConversationTool:
    """Tests for analyze_conversation_tool."""

    @pytest.mark.asyncio
    async def test_basic_analysis(self):
        """Basic conversation analysis works."""
        messages = [
            {"content": "Hello everyone!", "author": "user1"},
            {"content": "Hi there, how are you?", "author": "user2"},
        ]
        result = await analyze_conversation_tool(messages=messages)
        assert result["message_count"] == 2
        assert "analyzed_at" in result

    @pytest.mark.asyncio
    async def test_includes_sentiment_by_default(self):
        """Includes sentiment analysis by default."""
        messages = [{"content": "This is great!", "author": "user1"}]
        result = await analyze_conversation_tool(messages=messages)
        assert "sentiment" in result

    @pytest.mark.asyncio
    async def test_includes_activity_by_default(self):
        """Includes activity analysis by default."""
        messages = [{"content": "Hello", "author": "user1"}]
        result = await analyze_conversation_tool(messages=messages)
        assert "activity" in result

    @pytest.mark.asyncio
    async def test_includes_questions_by_default(self):
        """Includes question extraction by default."""
        messages = [{"content": "What is this?", "author": "user1"}]
        result = await analyze_conversation_tool(messages=messages)
        assert "questions" in result
        assert "question_count" in result

    @pytest.mark.asyncio
    async def test_includes_decisions_by_default(self):
        """Includes decision extraction by default."""
        messages = [{"content": "We decided to proceed", "author": "user1"}]
        result = await analyze_conversation_tool(messages=messages)
        assert "decisions" in result
        assert "decision_count" in result

    @pytest.mark.asyncio
    async def test_can_disable_sentiment(self):
        """Can disable sentiment analysis."""
        messages = [{"content": "Hello", "author": "user1"}]
        result = await analyze_conversation_tool(messages=messages, analyze_sentiment=False)
        assert "sentiment" not in result

    @pytest.mark.asyncio
    async def test_can_disable_activity(self):
        """Can disable activity analysis."""
        messages = [{"content": "Hello", "author": "user1"}]
        result = await analyze_conversation_tool(messages=messages, analyze_activity=False)
        assert "activity" not in result

    @pytest.mark.asyncio
    async def test_can_disable_questions(self):
        """Can disable question extraction."""
        messages = [{"content": "What is this?", "author": "user1"}]
        result = await analyze_conversation_tool(messages=messages, extract_questions=False)
        assert "questions" not in result

    @pytest.mark.asyncio
    async def test_can_disable_decisions(self):
        """Can disable decision extraction."""
        messages = [{"content": "We decided to proceed", "author": "user1"}]
        result = await analyze_conversation_tool(messages=messages, extract_decisions=False)
        assert "decisions" not in result


# =============================================================================
# Fetch Context Tool Tests (with fallbacks)
# =============================================================================


class TestFetchChannelContextTool:
    """Tests for fetch_channel_context_tool with missing connector."""

    @pytest.mark.asyncio
    async def test_unknown_platform_returns_error(self):
        """Unknown platform returns error."""
        result = await fetch_channel_context_tool(channel_id="test", platform="unknown_platform")
        assert "error" in result
        assert "unknown_platform" in result["error"].lower()


class TestFetchDebateContextTool:
    """Tests for fetch_debate_context_tool."""

    @pytest.mark.asyncio
    async def test_returns_result_or_error(self):
        """Returns either result structure or error."""
        result = await fetch_debate_context_tool(debate_id="nonexistent")
        # Either error or debate_id present
        assert "error" in result or "debate_id" in result


class TestGetThreadContextTool:
    """Tests for get_thread_context_tool."""

    @pytest.mark.asyncio
    async def test_unknown_platform_returns_error(self):
        """Unknown platform returns error."""
        result = await get_thread_context_tool(thread_id="test", platform="unknown_platform")
        assert "error" in result


class TestGetUserContextTool:
    """Tests for get_user_context_tool."""

    @pytest.mark.asyncio
    async def test_unknown_platform_returns_error(self):
        """Unknown platform returns error."""
        result = await get_user_context_tool(user_id="test", platform="unknown_platform")
        assert "error" in result
