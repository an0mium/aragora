"""Tests for MCP context tools execution logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.mcp.tools_module.context_tools import (
    analyze_conversation_tool,
    fetch_channel_context_tool,
    fetch_debate_context_tool,
    get_thread_context_tool,
    get_user_context_tool,
)

pytest.importorskip("mcp")


class TestFetchChannelContextTool:
    """Tests for fetch_channel_context_tool."""

    @pytest.mark.asyncio
    async def test_fetch_success(self):
        """Test successful channel context fetch."""
        mock_messages = [
            {"content": "Hello", "author": "user1", "timestamp": "2024-01-01T10:00:00Z"},
            {"content": "Hi there", "author": "user2", "timestamp": "2024-01-01T10:01:00Z"},
        ]

        mock_connector = MagicMock()

        with (
            patch(
                "aragora.mcp.tools_module.context_tools._get_chat_connector",
                return_value=mock_connector,
            ),
            patch(
                "aragora.mcp.tools_module.context_tools._fetch_channel_messages",
                return_value=mock_messages,
            ),
            patch(
                "aragora.mcp.tools_module.context_tools._extract_topics",
                return_value=["topic1", "topic2"],
            ),
        ):
            result = await fetch_channel_context_tool(
                channel_id="channel-1",
                platform="slack",
            )

        assert result["channel_id"] == "channel-1"
        assert result["message_count"] == 2
        assert result["participant_count"] == 2
        assert "user1" in result["participants"]
        assert "topics" in result

    @pytest.mark.asyncio
    async def test_fetch_no_connector(self):
        """Test fetch when no connector available."""
        with patch(
            "aragora.mcp.tools_module.context_tools._get_chat_connector",
            return_value=None,
        ):
            result = await fetch_channel_context_tool(
                channel_id="channel-1",
                platform="unknown",
            )

        assert "error" in result
        assert "no connector" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_fetch_without_participants(self):
        """Test fetch without participant extraction."""
        mock_messages = [{"content": "Test", "author": "user1"}]

        mock_connector = MagicMock()

        with (
            patch(
                "aragora.mcp.tools_module.context_tools._get_chat_connector",
                return_value=mock_connector,
            ),
            patch(
                "aragora.mcp.tools_module.context_tools._fetch_channel_messages",
                return_value=mock_messages,
            ),
        ):
            result = await fetch_channel_context_tool(
                channel_id="channel-1",
                include_participants=False,
                include_topics=False,
            )

        assert "participants" not in result
        assert "topics" not in result


class TestFetchDebateContextTool:
    """Tests for fetch_debate_context_tool."""

    @pytest.mark.asyncio
    async def test_fetch_debate_success(self):
        """Test successful debate context fetch."""
        mock_consensus = MagicMock()
        mock_consensus.topic = "Which database to use?"
        mock_consensus.conclusion = "PostgreSQL"
        mock_consensus.timestamp = MagicMock()
        mock_consensus.timestamp.isoformat.return_value = "2024-01-01T12:00:00"
        mock_consensus.rounds = 3
        mock_consensus.strength = MagicMock(value="strong")
        mock_consensus.confidence = 0.85
        mock_consensus.participating_agents = ["claude", "gpt4"]
        mock_consensus.debate_duration_seconds = 120
        mock_consensus.metadata = {"total_tokens": 5000}

        mock_memory = MagicMock()
        mock_memory.get_consensus.return_value = mock_consensus

        with patch(
            "aragora.memory.consensus.ConsensusMemory",
            return_value=mock_memory,
        ):
            result = await fetch_debate_context_tool(
                debate_id="debate-123",
                include_history=True,
                include_consensus=True,
                include_metrics=True,
            )

        assert result["debate_id"] == "debate-123"
        assert result["task"] == "Which database to use?"
        assert result["final_answer"] == "PostgreSQL"
        assert result["rounds"] == 3
        assert result["confidence"] == 0.85
        assert "metrics" in result

    @pytest.mark.asyncio
    async def test_fetch_debate_not_found(self):
        """Test fetch when debate not found."""
        mock_memory = MagicMock()
        mock_memory.get_consensus.return_value = None

        with patch(
            "aragora.memory.consensus.ConsensusMemory",
            return_value=mock_memory,
        ):
            result = await fetch_debate_context_tool(debate_id="nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()


class TestAnalyzeConversationTool:
    """Tests for analyze_conversation_tool."""

    @pytest.mark.asyncio
    async def test_analyze_empty_messages(self):
        """Test analysis with empty messages."""
        result = await analyze_conversation_tool(messages=[])

        assert "error" in result
        assert result["message_count"] == 0

    @pytest.mark.asyncio
    async def test_analyze_success(self):
        """Test successful conversation analysis."""
        messages = [
            {"content": "What should we do?", "author": "user1", "timestamp": "2024-01-01T10:00:00Z"},
            {"content": "I think we decided to go with option A", "author": "user2", "timestamp": "2024-01-01T10:05:00Z"},
            {"content": "Great idea!", "author": "user1", "timestamp": "2024-01-01T10:06:00Z"},
        ]

        result = await analyze_conversation_tool(
            messages=messages,
            analyze_sentiment=True,
            analyze_activity=True,
            extract_questions=True,
            extract_decisions=True,
        )

        assert result["message_count"] == 3
        assert "sentiment" in result
        assert "activity" in result
        assert "questions" in result
        assert "decisions" in result

    @pytest.mark.asyncio
    async def test_analyze_extracts_questions(self):
        """Test question extraction."""
        messages = [
            {"content": "What database should we use?", "author": "user1"},
            {"content": "How about PostgreSQL?", "author": "user2"},
            {"content": "Sounds good.", "author": "user1"},
        ]

        result = await analyze_conversation_tool(
            messages=messages,
            extract_questions=True,
            analyze_sentiment=False,
            analyze_activity=False,
            extract_decisions=False,
        )

        assert result["question_count"] == 2
        assert len(result["questions"]) == 2

    @pytest.mark.asyncio
    async def test_analyze_extracts_decisions(self):
        """Test decision extraction."""
        messages = [
            {"content": "We decided to go with PostgreSQL", "author": "user1"},
            {"content": "The conclusion is clear", "author": "user2"},
            {"content": "Let me think about it", "author": "user3"},
        ]

        result = await analyze_conversation_tool(
            messages=messages,
            extract_decisions=True,
            analyze_sentiment=False,
            analyze_activity=False,
            extract_questions=False,
        )

        assert result["decision_count"] == 2

    @pytest.mark.asyncio
    async def test_analyze_sentiment(self):
        """Test sentiment analysis."""
        messages = [
            {"content": "This is great! I love it!", "author": "user1"},
            {"content": "Excellent work, amazing!", "author": "user2"},
            {"content": "Thanks for the helpful response", "author": "user3"},
        ]

        result = await analyze_conversation_tool(
            messages=messages,
            analyze_sentiment=True,
            analyze_activity=False,
            extract_questions=False,
            extract_decisions=False,
        )

        assert result["sentiment"]["overall"] == "positive"
        assert result["sentiment"]["positive_ratio"] > 0


class TestGetThreadContextTool:
    """Tests for get_thread_context_tool."""

    @pytest.mark.asyncio
    async def test_get_thread_success(self):
        """Test successful thread context fetch."""
        mock_messages = [
            {"content": "Parent message", "id": "msg-1"},
            {"content": "Reply 1", "id": "msg-2"},
        ]

        mock_connector = AsyncMock()
        mock_connector.get_thread_messages.return_value = mock_messages

        with patch(
            "aragora.mcp.tools_module.context_tools._get_chat_connector",
            return_value=mock_connector,
        ):
            result = await get_thread_context_tool(
                thread_id="thread-123",
                platform="slack",
            )

        assert result["thread_id"] == "thread-123"
        assert result["message_count"] == 2
        assert result["parent_message"] == mock_messages[0]

    @pytest.mark.asyncio
    async def test_get_thread_no_connector(self):
        """Test thread fetch when no connector available."""
        with patch(
            "aragora.mcp.tools_module.context_tools._get_chat_connector",
            return_value=None,
        ):
            result = await get_thread_context_tool(
                thread_id="thread-123",
                platform="unknown",
            )

        assert "error" in result


class TestGetUserContextTool:
    """Tests for get_user_context_tool."""

    @pytest.mark.asyncio
    async def test_get_user_success(self):
        """Test successful user context fetch."""
        mock_user_info = {"name": "Test User", "email": "test@example.com"}
        mock_messages = [{"content": "Hello", "id": "msg-1"}]

        mock_connector = AsyncMock()
        mock_connector.get_user_info.return_value = mock_user_info
        mock_connector.get_user_messages.return_value = mock_messages

        with patch(
            "aragora.mcp.tools_module.context_tools._get_chat_connector",
            return_value=mock_connector,
        ):
            result = await get_user_context_tool(
                user_id="user-123",
                platform="slack",
            )

        assert result["user_id"] == "user-123"
        assert result["user_info"] == mock_user_info
        assert result["message_count"] == 1

    @pytest.mark.asyncio
    async def test_get_user_no_connector(self):
        """Test user fetch when no connector available."""
        with patch(
            "aragora.mcp.tools_module.context_tools._get_chat_connector",
            return_value=None,
        ):
            result = await get_user_context_tool(
                user_id="user-123",
                platform="unknown",
            )

        assert "error" in result
