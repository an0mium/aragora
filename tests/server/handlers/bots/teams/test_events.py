"""
Tests for Teams events handler.

Tests cover:
- Activity type routing
- Message handling
- Conversation update handling
- Installation update handling
- Mention pattern parsing
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.bots.teams.events import TeamsEventProcessor


@pytest.fixture
def mock_bot():
    """Create a mock TeamsBot instance."""
    bot = MagicMock()
    bot.send_activity = AsyncMock()
    return bot


@pytest.fixture
def processor(mock_bot):
    """Create a TeamsEventProcessor instance."""
    return TeamsEventProcessor(mock_bot)


class TestActivityRouting:
    """Tests for activity type routing."""

    @pytest.mark.asyncio
    async def test_message_activity_routed(self, processor):
        """Test message activity is routed correctly."""
        activity = {
            "type": "message",
            "text": "Hello",
            "from": {"id": "user123"},
            "conversation": {"id": "conv123"},
        }

        with patch.object(processor, "_handle_message", new_callable=AsyncMock, return_value={}):
            result = await processor.process_activity(activity)

            processor._handle_message.assert_called_once_with(activity)

    @pytest.mark.asyncio
    async def test_invoke_activity_routed(self, processor):
        """Test invoke activity is routed correctly."""
        activity = {
            "type": "invoke",
            "name": "adaptiveCard/action",
            "value": {"action": "vote"},
            "from": {"id": "user123"},
            "conversation": {"id": "conv123"},
        }

        with patch.object(processor, "_handle_invoke", new_callable=AsyncMock, return_value={}):
            result = await processor.process_activity(activity)

            processor._handle_invoke.assert_called_once_with(activity)

    @pytest.mark.asyncio
    async def test_conversation_update_routed(self, processor):
        """Test conversationUpdate activity is routed correctly."""
        activity = {
            "type": "conversationUpdate",
            "membersAdded": [{"id": "bot123"}],
            "conversation": {"id": "conv123"},
        }

        with patch.object(
            processor,
            "_handle_conversation_update",
            new_callable=AsyncMock,
            return_value={},
        ):
            result = await processor.process_activity(activity)

            processor._handle_conversation_update.assert_called_once_with(activity)

    @pytest.mark.asyncio
    async def test_installation_update_routed(self, processor):
        """Test installationUpdate activity is routed correctly."""
        activity = {
            "type": "installationUpdate",
            "action": "add",
            "conversation": {"id": "conv123"},
        }

        with patch.object(
            processor,
            "_handle_installation_update",
            new_callable=AsyncMock,
            return_value={},
        ):
            result = await processor.process_activity(activity)

            processor._handle_installation_update.assert_called_once_with(activity)

    @pytest.mark.asyncio
    async def test_unknown_activity_handled_gracefully(self, processor):
        """Test unknown activity types are handled gracefully."""
        activity = {
            "type": "unknownActivityType",
            "data": "test",
        }

        result = await processor.process_activity(activity)

        # Should return empty dict for unknown types
        assert result == {}


class TestMentionPattern:
    """Tests for mention pattern parsing."""

    def test_mention_pattern_exists(self):
        """Test MENTION_PATTERN is defined."""
        from aragora.server.handlers.bots.teams.events import MENTION_PATTERN

        assert MENTION_PATTERN is not None

    def test_mention_pattern_matches_at_tags(self):
        """Test MENTION_PATTERN matches <at> tags."""
        from aragora.server.handlers.bots.teams.events import MENTION_PATTERN

        text = "<at>Bot Name</at> help"
        result = MENTION_PATTERN.sub("", text)

        assert result.strip() == "help"

    def test_mention_pattern_handles_multiple_mentions(self):
        """Test MENTION_PATTERN handles multiple mentions."""
        from aragora.server.handlers.bots.teams.events import MENTION_PATTERN

        text = "<at>Bot</at> <at>User</at> help me"
        result = MENTION_PATTERN.sub("", text)

        assert "help me" in result
        assert "<at>" not in result


class TestAgentDisplayNames:
    """Tests for agent display names in events module."""

    def test_agent_display_names_defined(self):
        """Test agent display names are defined."""
        from aragora.server.handlers.bots.teams.events import AGENT_DISPLAY_NAMES

        assert AGENT_DISPLAY_NAMES is not None
        assert isinstance(AGENT_DISPLAY_NAMES, dict)

    def test_display_names_for_common_agents(self):
        """Test display names exist for common agents."""
        from aragora.server.handlers.bots.teams.events import AGENT_DISPLAY_NAMES

        assert "claude" in AGENT_DISPLAY_NAMES
        assert "gpt4" in AGENT_DISPLAY_NAMES
        assert "gemini" in AGENT_DISPLAY_NAMES


class TestPermissionConstants:
    """Tests for permission constants in events module."""

    def test_permission_constants_defined(self):
        """Test permission constants are defined."""
        from aragora.server.handlers.bots.teams.events import (
            PERM_TEAMS_DEBATES_CREATE,
            PERM_TEAMS_MESSAGES_READ,
        )

        assert PERM_TEAMS_MESSAGES_READ is not None
        assert PERM_TEAMS_DEBATES_CREATE is not None


class TestEventProcessorInit:
    """Tests for TeamsEventProcessor initialization."""

    def test_processor_stores_bot_reference(self, mock_bot, processor):
        """Test processor stores bot reference."""
        assert processor.bot is mock_bot

    def test_processor_creates_without_error(self, mock_bot):
        """Test processor can be created without error."""
        processor = TeamsEventProcessor(mock_bot)

        assert processor is not None


class TestMessageReaction:
    """Tests for message reaction handling."""

    @pytest.mark.asyncio
    async def test_message_reaction_routed(self, processor):
        """Test messageReaction activity is routed correctly."""
        activity = {
            "type": "messageReaction",
            "reactionsAdded": [{"type": "like"}],
            "replyToId": "msg123",
            "from": {"id": "user123"},
            "conversation": {"id": "conv123"},
        }

        with patch.object(
            processor,
            "_handle_message_reaction",
            new_callable=AsyncMock,
            return_value={},
        ):
            result = await processor.process_activity(activity)

            processor._handle_message_reaction.assert_called_once_with(activity)
