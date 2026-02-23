"""Tests for MCP chat action tools execution logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.mcp.tools_module.chat_actions import (
    add_reaction_tool,
    create_poll_tool,
    send_message_tool,
    update_message_tool,
)


class TestSendMessageTool:
    """Tests for send_message_tool."""

    @pytest.mark.asyncio
    async def test_send_success(self):
        """Test successful message send."""
        mock_connector = AsyncMock()
        mock_connector.send_message.return_value = {"message_id": "msg-123"}

        with patch(
            "aragora.mcp.tools_module.chat_actions._get_chat_connector",
            return_value=mock_connector,
        ):
            result = await send_message_tool(
                channel_id="channel-1",
                content="Hello, world!",
                platform="slack",
            )

        assert result["success"] is True
        assert result["message_id"] == "msg-123"
        assert result["channel_id"] == "channel-1"
        assert result["platform"] == "slack"

    @pytest.mark.asyncio
    async def test_send_no_connector(self):
        """Test send when no connector available."""
        with patch(
            "aragora.mcp.tools_module.chat_actions._get_chat_connector",
            return_value=None,
        ):
            result = await send_message_tool(
                channel_id="channel-1",
                content="Hello",
                platform="unknown",
            )

        assert "error" in result
        assert "no connector" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_send_with_thread(self):
        """Test send with thread ID."""
        mock_connector = AsyncMock()
        mock_connector.send_message.return_value = {"message_id": "msg-456"}

        with patch(
            "aragora.mcp.tools_module.chat_actions._get_chat_connector",
            return_value=mock_connector,
        ):
            result = await send_message_tool(
                channel_id="channel-1",
                content="Thread reply",
                thread_id="thread-123",
            )

        assert result["success"] is True
        mock_connector.send_message.assert_called_once()
        call_kwargs = mock_connector.send_message.call_args.kwargs
        assert call_kwargs["thread_id"] == "thread-123"

    @pytest.mark.asyncio
    async def test_send_exception(self):
        """Test send exception handling."""
        mock_connector = AsyncMock()
        mock_connector.send_message.side_effect = RuntimeError("API error")

        with patch(
            "aragora.mcp.tools_module.chat_actions._get_chat_connector",
            return_value=mock_connector,
        ):
            result = await send_message_tool(
                channel_id="channel-1",
                content="Hello",
            )

        assert "error" in result
        assert "failed" in result["error"].lower()


class TestCreatePollTool:
    """Tests for create_poll_tool."""

    @pytest.mark.asyncio
    async def test_create_poll_success(self):
        """Test successful poll creation."""
        mock_connector = AsyncMock()
        mock_connector.create_poll.return_value = {"message_id": "poll-msg-1"}

        with patch(
            "aragora.mcp.tools_module.chat_actions._get_chat_connector",
            return_value=mock_connector,
        ):
            result = await create_poll_tool(
                channel_id="channel-1",
                question="What database to use?",
                options=["PostgreSQL", "MySQL", "MongoDB"],
            )

        assert result["success"] is True
        assert "poll_id" in result
        assert result["question"] == "What database to use?"
        assert result["options"] == ["PostgreSQL", "MySQL", "MongoDB"]

    @pytest.mark.asyncio
    async def test_create_poll_too_few_options(self):
        """Test poll with less than 2 options."""
        result = await create_poll_tool(
            channel_id="channel-1",
            question="Question?",
            options=["Only one"],
        )

        assert "error" in result
        assert "at least 2" in result["error"]

    @pytest.mark.asyncio
    async def test_create_poll_too_many_options(self):
        """Test poll with more than 10 options."""
        result = await create_poll_tool(
            channel_id="channel-1",
            question="Question?",
            options=[f"Option {i}" for i in range(12)],
        )

        assert "error" in result
        assert "at most 10" in result["error"]

    @pytest.mark.asyncio
    async def test_create_poll_fallback_to_message(self):
        """Test poll falls back to send_message when create_poll not available."""
        mock_connector = AsyncMock()
        # Remove create_poll method
        del mock_connector.create_poll
        mock_connector.send_message.return_value = {"message_id": "fallback-msg"}

        with patch(
            "aragora.mcp.tools_module.chat_actions._get_chat_connector",
            return_value=mock_connector,
        ):
            result = await create_poll_tool(
                channel_id="channel-1",
                question="Question?",
                options=["A", "B"],
            )

        assert result["success"] is True
        mock_connector.send_message.assert_called_once()


class TestUpdateMessageTool:
    """Tests for update_message_tool."""

    @pytest.mark.asyncio
    async def test_update_success(self):
        """Test successful message update."""
        mock_connector = AsyncMock()
        mock_connector.update_message = AsyncMock()

        with patch(
            "aragora.mcp.tools_module.chat_actions._get_chat_connector",
            return_value=mock_connector,
        ):
            result = await update_message_tool(
                message_id="msg-123",
                channel_id="channel-1",
                content="Updated content",
            )

        assert result["success"] is True
        assert result["message_id"] == "msg-123"
        assert "updated_at" in result

    @pytest.mark.asyncio
    async def test_update_no_connector(self):
        """Test update when no connector available."""
        with patch(
            "aragora.mcp.tools_module.chat_actions._get_chat_connector",
            return_value=None,
        ):
            result = await update_message_tool(
                message_id="msg-123",
                channel_id="channel-1",
                content="Updated",
            )

        assert "error" in result


class TestAddReactionTool:
    """Tests for add_reaction_tool."""

    @pytest.mark.asyncio
    async def test_add_reaction_success(self):
        """Test successful reaction add."""
        mock_connector = AsyncMock()
        mock_connector.add_reaction = AsyncMock()

        with patch(
            "aragora.mcp.tools_module.chat_actions._get_chat_connector",
            return_value=mock_connector,
        ):
            result = await add_reaction_tool(
                message_id="msg-123",
                channel_id="channel-1",
                emoji="thumbsup",
            )

        assert result["success"] is True
        assert result["emoji"] == "thumbsup"

    @pytest.mark.asyncio
    async def test_add_reaction_no_connector(self):
        """Test reaction when no connector available."""
        with patch(
            "aragora.mcp.tools_module.chat_actions._get_chat_connector",
            return_value=None,
        ):
            result = await add_reaction_tool(
                message_id="msg-123",
                channel_id="channel-1",
                emoji="thumbsup",
            )

        assert "error" in result
