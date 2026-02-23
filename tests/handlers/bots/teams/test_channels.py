"""
Tests for Teams Bot channel and team operations.

Covers all behavior of the TeamsChannelManager class:

- __init__()
  - With bot / without bot

- _get_connector()
  - Returns bot connector when bot is set
  - Creates TeamsConnector when no bot
  - Returns None on ImportError
  - Caches connector on second call

- Conversation Reference Management
  - get_all_conversation_references() with empty / populated store
  - get_reference() existing / missing
  - remove_reference() existing / missing
  - clear_references() with entries / empty store

- Team/Channel Operations
  - get_team_details() success / no connector / error
  - get_team_channels() success / no connector / error / missing key
  - get_conversation_members() success (list) / success (non-list) / no connector / error
  - get_member() success / no connector / error

- Proactive Messaging
  - send_to_conversation() with bot / without bot + ref / no ref / no connector / error
  - send_to_conversation() text only / card only / card + fallback text
  - send_to_channel() success text / success card / no connector / error
  - send_to_channel() uses TEAMS_APP_ID in conversation params

- Message Management
  - update_activity() success text / success card / no connector / error
  - delete_activity() success / no connector / error

- Error types
  - RuntimeError, OSError, ValueError, KeyError all caught
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_conversation_references():
    """Ensure conversation references are clean for every test."""
    from aragora.server.handlers.bots.teams_utils import _conversation_references

    _conversation_references.clear()
    yield
    _conversation_references.clear()


@pytest.fixture()
def refs():
    """Direct access to the conversation references dict."""
    from aragora.server.handlers.bots.teams_utils import _conversation_references

    return _conversation_references


@pytest.fixture()
def mock_connector():
    """Create a mock Teams connector."""
    connector = AsyncMock()
    connector._get_access_token = AsyncMock(return_value="test-token-123")
    connector._http_request = AsyncMock(return_value={})
    return connector


@pytest.fixture()
def mock_bot(mock_connector):
    """Create a mock TeamsBot."""
    bot = AsyncMock()
    bot._get_connector = AsyncMock(return_value=mock_connector)
    bot.send_proactive_message = AsyncMock(return_value=True)
    return bot


@pytest.fixture()
def manager():
    """Create a TeamsChannelManager without a bot."""
    from aragora.server.handlers.bots.teams.channels import TeamsChannelManager

    return TeamsChannelManager(bot=None)


@pytest.fixture()
def manager_with_bot(mock_bot):
    """Create a TeamsChannelManager with a bot."""
    from aragora.server.handlers.bots.teams.channels import TeamsChannelManager

    return TeamsChannelManager(bot=mock_bot)


SERVICE_URL = "https://smba.trafficmanager.net/teams"


# ===========================================================================
# __init__
# ===========================================================================


class TestInit:
    def test_init_without_bot(self, manager):
        assert manager.bot is None
        assert manager._connector is None

    def test_init_with_bot(self, manager_with_bot, mock_bot):
        assert manager_with_bot.bot is mock_bot
        assert manager_with_bot._connector is None


# ===========================================================================
# _get_connector
# ===========================================================================


class TestGetConnector:
    @pytest.mark.asyncio
    async def test_returns_bot_connector_when_bot_set(self, manager_with_bot, mock_connector):
        result = await manager_with_bot._get_connector()
        assert result is mock_connector

    @pytest.mark.asyncio
    async def test_creates_teams_connector_when_no_bot(self, manager):
        fake_connector = MagicMock()
        with patch(
            "aragora.server.handlers.bots.teams.channels.TeamsConnector",
            return_value=fake_connector,
            create=True,
        ):
            # Patch the import inside the method
            import aragora.server.handlers.bots.teams.channels as mod

            original = mod.__dict__.get("TeamsConnector")
            try:
                # We need to mock the dynamic import
                with patch.dict("sys.modules", {
                    "aragora.connectors.chat.teams": MagicMock(
                        TeamsConnector=MagicMock(return_value=fake_connector)
                    ),
                }):
                    manager._connector = None
                    result = await manager._get_connector()
                    assert result is fake_connector
            finally:
                if original is not None:
                    mod.__dict__["TeamsConnector"] = original

    @pytest.mark.asyncio
    async def test_returns_none_on_import_error(self, manager):
        with patch.dict("sys.modules", {"aragora.connectors.chat.teams": None}):
            manager._connector = None
            result = await manager._get_connector()
            assert result is None

    @pytest.mark.asyncio
    async def test_caches_connector_on_second_call(self, manager):
        fake_connector = MagicMock()
        manager._connector = fake_connector
        result = await manager._get_connector()
        assert result is fake_connector


# ===========================================================================
# Conversation Reference Management
# ===========================================================================


class TestGetAllConversationReferences:
    def test_empty_store(self, manager):
        result = manager.get_all_conversation_references()
        assert result == {}

    def test_populated_store(self, manager, refs):
        refs["conv-1"] = {"service_url": "https://a.com"}
        refs["conv-2"] = {"service_url": "https://b.com"}
        result = manager.get_all_conversation_references()
        assert len(result) == 2
        assert "conv-1" in result
        assert "conv-2" in result

    def test_returns_copy_not_original(self, manager, refs):
        refs["conv-1"] = {"service_url": "https://a.com"}
        result = manager.get_all_conversation_references()
        # Modifying the result should not affect the store
        result["conv-new"] = {}
        assert "conv-new" not in refs


class TestGetReference:
    def test_existing_reference(self, manager, refs):
        refs["conv-abc"] = {"service_url": "https://x.com", "tenant_id": "t-1"}
        result = manager.get_reference("conv-abc")
        assert result is not None
        assert result["service_url"] == "https://x.com"

    def test_missing_reference(self, manager):
        result = manager.get_reference("nonexistent")
        assert result is None


class TestRemoveReference:
    def test_remove_existing(self, manager, refs):
        refs["conv-abc"] = {"service_url": "https://x.com"}
        result = manager.remove_reference("conv-abc")
        assert result is True
        assert "conv-abc" not in refs

    def test_remove_missing(self, manager):
        result = manager.remove_reference("nonexistent")
        assert result is False


class TestClearReferences:
    def test_clear_with_entries(self, manager, refs):
        refs["conv-1"] = {"service_url": "https://a.com"}
        refs["conv-2"] = {"service_url": "https://b.com"}
        refs["conv-3"] = {"service_url": "https://c.com"}
        count = manager.clear_references()
        assert count == 3
        assert len(refs) == 0

    def test_clear_empty_store(self, manager, refs):
        count = manager.clear_references()
        assert count == 0


# ===========================================================================
# Team/Channel Operations
# ===========================================================================


class TestGetTeamDetails:
    @pytest.mark.asyncio
    async def test_success(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.return_value = {
            "id": "team-1",
            "displayName": "Engineering",
        }
        result = await manager.get_team_details("team-1", SERVICE_URL)
        assert result is not None
        assert result["id"] == "team-1"
        assert result["displayName"] == "Engineering"
        mock_connector._get_access_token.assert_awaited_once()
        mock_connector._http_request.assert_awaited_once()
        call_kwargs = mock_connector._http_request.call_args
        assert call_kwargs.kwargs["method"] == "GET"
        assert "team-1" in call_kwargs.kwargs["url"]

    @pytest.mark.asyncio
    async def test_no_connector(self, manager):
        manager._connector = None
        with patch.dict("sys.modules", {"aragora.connectors.chat.teams": None}):
            result = await manager.get_team_details("team-1", SERVICE_URL)
        assert result is None

    @pytest.mark.asyncio
    async def test_runtime_error(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.side_effect = RuntimeError("connection failed")
        result = await manager.get_team_details("team-1", SERVICE_URL)
        assert result is None

    @pytest.mark.asyncio
    async def test_os_error(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.side_effect = OSError("network error")
        result = await manager.get_team_details("team-1", SERVICE_URL)
        assert result is None

    @pytest.mark.asyncio
    async def test_value_error(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._get_access_token.side_effect = ValueError("bad token")
        result = await manager.get_team_details("team-1", SERVICE_URL)
        assert result is None

    @pytest.mark.asyncio
    async def test_key_error(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._get_access_token.side_effect = KeyError("missing")
        result = await manager.get_team_details("team-1", SERVICE_URL)
        assert result is None


class TestGetTeamChannels:
    @pytest.mark.asyncio
    async def test_success(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.return_value = {
            "conversations": [
                {"id": "chan-1", "name": "General"},
                {"id": "chan-2", "name": "Random"},
            ]
        }
        result = await manager.get_team_channels("team-1", SERVICE_URL)
        assert len(result) == 2
        assert result[0]["id"] == "chan-1"

    @pytest.mark.asyncio
    async def test_no_connector(self, manager):
        manager._connector = None
        with patch.dict("sys.modules", {"aragora.connectors.chat.teams": None}):
            result = await manager.get_team_channels("team-1", SERVICE_URL)
        assert result == []

    @pytest.mark.asyncio
    async def test_error_returns_empty(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.side_effect = RuntimeError("fail")
        result = await manager.get_team_channels("team-1", SERVICE_URL)
        assert result == []

    @pytest.mark.asyncio
    async def test_missing_conversations_key(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.return_value = {}
        result = await manager.get_team_channels("team-1", SERVICE_URL)
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_conversations(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.return_value = {"conversations": []}
        result = await manager.get_team_channels("team-1", SERVICE_URL)
        assert result == []

    @pytest.mark.asyncio
    async def test_url_includes_team_id(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.return_value = {"conversations": []}
        await manager.get_team_channels("my-team-id", SERVICE_URL)
        call_kwargs = mock_connector._http_request.call_args.kwargs
        assert "my-team-id" in call_kwargs["url"]
        assert call_kwargs["url"].endswith("/conversations")


class TestGetConversationMembers:
    @pytest.mark.asyncio
    async def test_success_list_response(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.return_value = [
            {"id": "user-1", "name": "Alice"},
            {"id": "user-2", "name": "Bob"},
        ]
        result = await manager.get_conversation_members("conv-1", SERVICE_URL)
        assert len(result) == 2
        assert result[0]["id"] == "user-1"

    @pytest.mark.asyncio
    async def test_non_list_response_returns_empty(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.return_value = {"members": []}
        result = await manager.get_conversation_members("conv-1", SERVICE_URL)
        assert result == []

    @pytest.mark.asyncio
    async def test_no_connector(self, manager):
        manager._connector = None
        with patch.dict("sys.modules", {"aragora.connectors.chat.teams": None}):
            result = await manager.get_conversation_members("conv-1", SERVICE_URL)
        assert result == []

    @pytest.mark.asyncio
    async def test_error_returns_empty(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.side_effect = OSError("timeout")
        result = await manager.get_conversation_members("conv-1", SERVICE_URL)
        assert result == []

    @pytest.mark.asyncio
    async def test_url_includes_conversation_id(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.return_value = []
        await manager.get_conversation_members("conv-xyz", SERVICE_URL)
        url = mock_connector._http_request.call_args.kwargs["url"]
        assert "/conversations/conv-xyz/members" in url


class TestGetMember:
    @pytest.mark.asyncio
    async def test_success(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.return_value = {
            "id": "user-1",
            "name": "Alice",
            "email": "alice@example.com",
        }
        result = await manager.get_member("conv-1", "user-1", SERVICE_URL)
        assert result is not None
        assert result["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_no_connector(self, manager):
        manager._connector = None
        with patch.dict("sys.modules", {"aragora.connectors.chat.teams": None}):
            result = await manager.get_member("conv-1", "user-1", SERVICE_URL)
        assert result is None

    @pytest.mark.asyncio
    async def test_error_returns_none(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.side_effect = ValueError("bad response")
        result = await manager.get_member("conv-1", "user-1", SERVICE_URL)
        assert result is None

    @pytest.mark.asyncio
    async def test_url_includes_member_id(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.return_value = {"id": "user-42"}
        await manager.get_member("conv-1", "user-42", SERVICE_URL)
        url = mock_connector._http_request.call_args.kwargs["url"]
        assert "/members/user-42" in url

    @pytest.mark.asyncio
    async def test_authorization_header(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.return_value = {"id": "user-1"}
        await manager.get_member("conv-1", "user-1", SERVICE_URL)
        headers = mock_connector._http_request.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-token-123"


# ===========================================================================
# Proactive Messaging - send_to_conversation
# ===========================================================================


class TestSendToConversation:
    @pytest.mark.asyncio
    async def test_delegates_to_bot(self, manager_with_bot, mock_bot):
        result = await manager_with_bot.send_to_conversation(
            conversation_id="conv-1",
            text="Hello from bot",
        )
        assert result is True
        mock_bot.send_proactive_message.assert_awaited_once_with(
            conversation_id="conv-1",
            text="Hello from bot",
            card=None,
            fallback_text="",
        )

    @pytest.mark.asyncio
    async def test_delegates_to_bot_with_card(self, manager_with_bot, mock_bot):
        card = {"type": "AdaptiveCard", "body": []}
        result = await manager_with_bot.send_to_conversation(
            conversation_id="conv-1",
            card=card,
            fallback_text="Fallback",
        )
        assert result is True
        mock_bot.send_proactive_message.assert_awaited_once_with(
            conversation_id="conv-1",
            text=None,
            card=card,
            fallback_text="Fallback",
        )

    @pytest.mark.asyncio
    async def test_no_ref_returns_false(self, manager):
        result = await manager.send_to_conversation(
            conversation_id="nonexistent",
            text="Hello",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_with_ref_text_only(self, manager, mock_connector, refs):
        manager._connector = mock_connector
        refs["conv-1"] = {"service_url": SERVICE_URL}
        result = await manager.send_to_conversation(
            conversation_id="conv-1",
            text="Hello world",
        )
        assert result is True
        call_kwargs = mock_connector._http_request.call_args.kwargs
        assert call_kwargs["method"] == "POST"
        assert "/conv-1/activities" in call_kwargs["url"]
        activity = call_kwargs["json"]
        assert activity["type"] == "message"
        assert activity["text"] == "Hello world"
        assert "attachments" not in activity

    @pytest.mark.asyncio
    async def test_with_ref_card(self, manager, mock_connector, refs):
        manager._connector = mock_connector
        refs["conv-1"] = {"service_url": SERVICE_URL}
        card = {"type": "AdaptiveCard", "version": "1.4", "body": []}
        result = await manager.send_to_conversation(
            conversation_id="conv-1",
            card=card,
            fallback_text="Card fallback",
        )
        assert result is True
        activity = mock_connector._http_request.call_args.kwargs["json"]
        assert activity["text"] == "Card fallback"
        assert len(activity["attachments"]) == 1
        assert activity["attachments"][0]["contentType"] == "application/vnd.microsoft.card.adaptive"
        assert activity["attachments"][0]["content"] is card

    @pytest.mark.asyncio
    async def test_with_ref_no_connector(self, manager, refs):
        refs["conv-1"] = {"service_url": SERVICE_URL}
        manager._connector = None
        with patch.dict("sys.modules", {"aragora.connectors.chat.teams": None}):
            result = await manager.send_to_conversation(
                conversation_id="conv-1",
                text="Hello",
            )
        assert result is False

    @pytest.mark.asyncio
    async def test_error_returns_false(self, manager, mock_connector, refs):
        manager._connector = mock_connector
        refs["conv-1"] = {"service_url": SERVICE_URL}
        mock_connector._http_request.side_effect = RuntimeError("send failed")
        result = await manager.send_to_conversation(
            conversation_id="conv-1",
            text="Hello",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_uses_fallback_text_when_no_text(self, manager, mock_connector, refs):
        manager._connector = mock_connector
        refs["conv-1"] = {"service_url": SERVICE_URL}
        await manager.send_to_conversation(
            conversation_id="conv-1",
            fallback_text="Fallback message",
        )
        activity = mock_connector._http_request.call_args.kwargs["json"]
        assert activity["text"] == "Fallback message"

    @pytest.mark.asyncio
    async def test_empty_text_when_no_text_or_fallback(self, manager, mock_connector, refs):
        manager._connector = mock_connector
        refs["conv-1"] = {"service_url": SERVICE_URL}
        await manager.send_to_conversation(conversation_id="conv-1")
        activity = mock_connector._http_request.call_args.kwargs["json"]
        assert activity["text"] == ""

    @pytest.mark.asyncio
    async def test_ref_service_url_used(self, manager, mock_connector, refs):
        manager._connector = mock_connector
        refs["conv-1"] = {"service_url": "https://custom.service.url/api"}
        await manager.send_to_conversation(
            conversation_id="conv-1",
            text="Test",
        )
        url = mock_connector._http_request.call_args.kwargs["url"]
        assert url.startswith("https://custom.service.url/api")

    @pytest.mark.asyncio
    async def test_ref_empty_dict_treated_as_missing(self, manager, mock_connector, refs):
        """An empty ref dict is falsy, so code treats it as 'no reference found'."""
        manager._connector = mock_connector
        refs["conv-1"] = {}  # empty dict is falsy
        result = await manager.send_to_conversation(
            conversation_id="conv-1",
            text="Test",
        )
        assert result is False  # treated as missing reference

    @pytest.mark.asyncio
    async def test_ref_with_empty_service_url(self, manager, mock_connector, refs):
        """A ref with at least one key is truthy; service_url defaults to ''."""
        manager._connector = mock_connector
        refs["conv-1"] = {"tenant_id": "t-1"}  # truthy, but no service_url
        await manager.send_to_conversation(
            conversation_id="conv-1",
            text="Test",
        )
        url = mock_connector._http_request.call_args.kwargs["url"]
        # service_url defaults to ""
        assert url.startswith("/v3/conversations/conv-1/activities")


# ===========================================================================
# Proactive Messaging - send_to_channel
# ===========================================================================


class TestSendToChannel:
    @pytest.mark.asyncio
    async def test_success_text(self, manager, mock_connector):
        manager._connector = mock_connector
        result = await manager.send_to_channel(
            team_id="team-1",
            channel_id="chan-1",
            service_url=SERVICE_URL,
            text="Hello channel",
        )
        assert result is True
        call_kwargs = mock_connector._http_request.call_args.kwargs
        assert call_kwargs["method"] == "POST"
        assert call_kwargs["url"] == f"{SERVICE_URL}/v3/conversations"
        params = call_kwargs["json"]
        assert params["isGroup"] is True
        assert params["channelData"]["teamsChannelId"] == "chan-1"
        assert params["channelData"]["teamsTeamId"] == "team-1"
        assert params["activity"]["text"] == "Hello channel"

    @pytest.mark.asyncio
    async def test_success_card(self, manager, mock_connector):
        manager._connector = mock_connector
        card = {"type": "AdaptiveCard", "body": [{"type": "TextBlock", "text": "Hi"}]}
        result = await manager.send_to_channel(
            team_id="team-1",
            channel_id="chan-1",
            service_url=SERVICE_URL,
            card=card,
            fallback_text="Card fallback",
        )
        assert result is True
        params = mock_connector._http_request.call_args.kwargs["json"]
        assert params["activity"]["text"] == "Card fallback"
        assert len(params["activity"]["attachments"]) == 1
        assert params["activity"]["attachments"][0]["content"] is card

    @pytest.mark.asyncio
    async def test_no_connector(self, manager):
        manager._connector = None
        with patch.dict("sys.modules", {"aragora.connectors.chat.teams": None}):
            result = await manager.send_to_channel(
                team_id="team-1",
                channel_id="chan-1",
                service_url=SERVICE_URL,
                text="Hello",
            )
        assert result is False

    @pytest.mark.asyncio
    async def test_error_returns_false(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.side_effect = OSError("network down")
        result = await manager.send_to_channel(
            team_id="team-1",
            channel_id="chan-1",
            service_url=SERVICE_URL,
            text="Hello",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_no_text_no_card_uses_fallback(self, manager, mock_connector):
        manager._connector = mock_connector
        await manager.send_to_channel(
            team_id="team-1",
            channel_id="chan-1",
            service_url=SERVICE_URL,
            fallback_text="Fallback only",
        )
        params = mock_connector._http_request.call_args.kwargs["json"]
        assert params["activity"]["text"] == "Fallback only"

    @pytest.mark.asyncio
    async def test_no_text_no_fallback(self, manager, mock_connector):
        manager._connector = mock_connector
        await manager.send_to_channel(
            team_id="team-1",
            channel_id="chan-1",
            service_url=SERVICE_URL,
        )
        params = mock_connector._http_request.call_args.kwargs["json"]
        assert params["activity"]["text"] == ""

    @pytest.mark.asyncio
    async def test_app_id_in_conversation_params(self, manager, mock_connector):
        manager._connector = mock_connector
        with patch(
            "aragora.server.handlers.bots.teams.channels.TEAMS_APP_ID", "app-id-123"
        ):
            await manager.send_to_channel(
                team_id="team-1",
                channel_id="chan-1",
                service_url=SERVICE_URL,
                text="Hi",
            )
        params = mock_connector._http_request.call_args.kwargs["json"]
        assert params["bot"]["id"] == "app-id-123"

    @pytest.mark.asyncio
    async def test_content_type_header(self, manager, mock_connector):
        manager._connector = mock_connector
        await manager.send_to_channel(
            team_id="team-1",
            channel_id="chan-1",
            service_url=SERVICE_URL,
            text="Hi",
        )
        headers = mock_connector._http_request.call_args.kwargs["headers"]
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-token-123"


# ===========================================================================
# Create Personal Conversation
# ===========================================================================


class TestCreatePersonalConversation:
    @pytest.mark.asyncio
    async def test_success(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.return_value = {"id": "new-conv-id"}
        result = await manager.create_personal_conversation(
            user_id="user-1",
            tenant_id="tenant-abc",
            service_url=SERVICE_URL,
        )
        assert result == "new-conv-id"
        call_kwargs = mock_connector._http_request.call_args.kwargs
        assert call_kwargs["method"] == "POST"
        assert call_kwargs["url"] == f"{SERVICE_URL}/v3/conversations"
        params = call_kwargs["json"]
        assert params["members"] == [{"id": "user-1"}]
        assert params["channelData"]["tenant"]["id"] == "tenant-abc"

    @pytest.mark.asyncio
    async def test_no_connector(self, manager):
        manager._connector = None
        with patch.dict("sys.modules", {"aragora.connectors.chat.teams": None}):
            result = await manager.create_personal_conversation(
                user_id="user-1",
                tenant_id="tenant-abc",
                service_url=SERVICE_URL,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_error_returns_none(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.side_effect = RuntimeError("create failed")
        result = await manager.create_personal_conversation(
            user_id="user-1",
            tenant_id="tenant-abc",
            service_url=SERVICE_URL,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_response_missing_id(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.return_value = {}
        result = await manager.create_personal_conversation(
            user_id="user-1",
            tenant_id="tenant-abc",
            service_url=SERVICE_URL,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_app_id_in_params(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.return_value = {"id": "c-1"}
        with patch(
            "aragora.server.handlers.bots.teams.channels.TEAMS_APP_ID", "my-app"
        ):
            await manager.create_personal_conversation(
                user_id="user-1",
                tenant_id="tenant-abc",
                service_url=SERVICE_URL,
            )
        params = mock_connector._http_request.call_args.kwargs["json"]
        assert params["bot"]["id"] == "my-app"

    @pytest.mark.asyncio
    async def test_authorization_header(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.return_value = {"id": "c-1"}
        await manager.create_personal_conversation(
            user_id="user-1",
            tenant_id="tenant-abc",
            service_url=SERVICE_URL,
        )
        headers = mock_connector._http_request.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-token-123"


# ===========================================================================
# Message Management - update_activity
# ===========================================================================


class TestUpdateActivity:
    @pytest.mark.asyncio
    async def test_success_text(self, manager, mock_connector):
        manager._connector = mock_connector
        result = await manager.update_activity(
            conversation_id="conv-1",
            activity_id="act-1",
            service_url=SERVICE_URL,
            text="Updated text",
        )
        assert result is True
        call_kwargs = mock_connector._http_request.call_args.kwargs
        assert call_kwargs["method"] == "PUT"
        assert "/conv-1/activities/act-1" in call_kwargs["url"]
        activity = call_kwargs["json"]
        assert activity["type"] == "message"
        assert activity["id"] == "act-1"
        assert activity["text"] == "Updated text"
        assert "attachments" not in activity

    @pytest.mark.asyncio
    async def test_success_card(self, manager, mock_connector):
        manager._connector = mock_connector
        card = {"type": "AdaptiveCard", "body": []}
        result = await manager.update_activity(
            conversation_id="conv-1",
            activity_id="act-1",
            service_url=SERVICE_URL,
            card=card,
        )
        assert result is True
        activity = mock_connector._http_request.call_args.kwargs["json"]
        assert len(activity["attachments"]) == 1
        assert activity["attachments"][0]["content"] is card

    @pytest.mark.asyncio
    async def test_no_connector(self, manager):
        manager._connector = None
        with patch.dict("sys.modules", {"aragora.connectors.chat.teams": None}):
            result = await manager.update_activity(
                conversation_id="conv-1",
                activity_id="act-1",
                service_url=SERVICE_URL,
                text="Updated",
            )
        assert result is False

    @pytest.mark.asyncio
    async def test_error_returns_false(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.side_effect = KeyError("missing field")
        result = await manager.update_activity(
            conversation_id="conv-1",
            activity_id="act-1",
            service_url=SERVICE_URL,
            text="Updated",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_no_text_no_card(self, manager, mock_connector):
        manager._connector = mock_connector
        await manager.update_activity(
            conversation_id="conv-1",
            activity_id="act-1",
            service_url=SERVICE_URL,
        )
        activity = mock_connector._http_request.call_args.kwargs["json"]
        assert activity["text"] == ""
        assert "attachments" not in activity

    @pytest.mark.asyncio
    async def test_text_and_card(self, manager, mock_connector):
        manager._connector = mock_connector
        card = {"type": "AdaptiveCard"}
        await manager.update_activity(
            conversation_id="conv-1",
            activity_id="act-1",
            service_url=SERVICE_URL,
            text="Both text and card",
            card=card,
        )
        activity = mock_connector._http_request.call_args.kwargs["json"]
        assert activity["text"] == "Both text and card"
        assert len(activity["attachments"]) == 1


# ===========================================================================
# Message Management - delete_activity
# ===========================================================================


class TestDeleteActivity:
    @pytest.mark.asyncio
    async def test_success(self, manager, mock_connector):
        manager._connector = mock_connector
        result = await manager.delete_activity(
            conversation_id="conv-1",
            activity_id="act-1",
            service_url=SERVICE_URL,
        )
        assert result is True
        call_kwargs = mock_connector._http_request.call_args.kwargs
        assert call_kwargs["method"] == "DELETE"
        assert "/conv-1/activities/act-1" in call_kwargs["url"]

    @pytest.mark.asyncio
    async def test_no_connector(self, manager):
        manager._connector = None
        with patch.dict("sys.modules", {"aragora.connectors.chat.teams": None}):
            result = await manager.delete_activity(
                conversation_id="conv-1",
                activity_id="act-1",
                service_url=SERVICE_URL,
            )
        assert result is False

    @pytest.mark.asyncio
    async def test_error_returns_false(self, manager, mock_connector):
        manager._connector = mock_connector
        mock_connector._http_request.side_effect = RuntimeError("delete failed")
        result = await manager.delete_activity(
            conversation_id="conv-1",
            activity_id="act-1",
            service_url=SERVICE_URL,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_authorization_header(self, manager, mock_connector):
        manager._connector = mock_connector
        await manager.delete_activity(
            conversation_id="conv-1",
            activity_id="act-1",
            service_url=SERVICE_URL,
        )
        headers = mock_connector._http_request.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-token-123"

    @pytest.mark.asyncio
    async def test_no_content_type_header(self, manager, mock_connector):
        manager._connector = mock_connector
        await manager.delete_activity(
            conversation_id="conv-1",
            activity_id="act-1",
            service_url=SERVICE_URL,
        )
        headers = mock_connector._http_request.call_args.kwargs["headers"]
        # DELETE requests should not have Content-Type
        assert "Content-Type" not in headers

    @pytest.mark.asyncio
    async def test_no_json_body(self, manager, mock_connector):
        manager._connector = mock_connector
        await manager.delete_activity(
            conversation_id="conv-1",
            activity_id="act-1",
            service_url=SERVICE_URL,
        )
        call_kwargs = mock_connector._http_request.call_args.kwargs
        assert "json" not in call_kwargs


# ===========================================================================
# Error Handling - All Caught Exception Types
# ===========================================================================


class TestErrorHandling:
    """Verify all four exception types are caught for each endpoint."""

    EXCEPTION_TYPES = [
        RuntimeError("runtime"),
        OSError("os"),
        ValueError("value"),
        KeyError("key"),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exc", EXCEPTION_TYPES, ids=lambda e: type(e).__name__)
    async def test_get_team_details_catches(self, manager, mock_connector, exc):
        manager._connector = mock_connector
        mock_connector._http_request.side_effect = exc
        result = await manager.get_team_details("t-1", SERVICE_URL)
        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exc", EXCEPTION_TYPES, ids=lambda e: type(e).__name__)
    async def test_get_team_channels_catches(self, manager, mock_connector, exc):
        manager._connector = mock_connector
        mock_connector._http_request.side_effect = exc
        result = await manager.get_team_channels("t-1", SERVICE_URL)
        assert result == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exc", EXCEPTION_TYPES, ids=lambda e: type(e).__name__)
    async def test_get_conversation_members_catches(self, manager, mock_connector, exc):
        manager._connector = mock_connector
        mock_connector._http_request.side_effect = exc
        result = await manager.get_conversation_members("c-1", SERVICE_URL)
        assert result == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exc", EXCEPTION_TYPES, ids=lambda e: type(e).__name__)
    async def test_get_member_catches(self, manager, mock_connector, exc):
        manager._connector = mock_connector
        mock_connector._http_request.side_effect = exc
        result = await manager.get_member("c-1", "u-1", SERVICE_URL)
        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exc", EXCEPTION_TYPES, ids=lambda e: type(e).__name__)
    async def test_send_to_conversation_catches(self, manager, mock_connector, refs, exc):
        manager._connector = mock_connector
        refs["conv-1"] = {"service_url": SERVICE_URL}
        mock_connector._http_request.side_effect = exc
        result = await manager.send_to_conversation("conv-1", text="Hi")
        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exc", EXCEPTION_TYPES, ids=lambda e: type(e).__name__)
    async def test_send_to_channel_catches(self, manager, mock_connector, exc):
        manager._connector = mock_connector
        mock_connector._http_request.side_effect = exc
        result = await manager.send_to_channel("t-1", "c-1", SERVICE_URL, text="Hi")
        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exc", EXCEPTION_TYPES, ids=lambda e: type(e).__name__)
    async def test_create_personal_conversation_catches(self, manager, mock_connector, exc):
        manager._connector = mock_connector
        mock_connector._http_request.side_effect = exc
        result = await manager.create_personal_conversation("u-1", "t-1", SERVICE_URL)
        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exc", EXCEPTION_TYPES, ids=lambda e: type(e).__name__)
    async def test_update_activity_catches(self, manager, mock_connector, exc):
        manager._connector = mock_connector
        mock_connector._http_request.side_effect = exc
        result = await manager.update_activity("c-1", "a-1", SERVICE_URL, text="x")
        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exc", EXCEPTION_TYPES, ids=lambda e: type(e).__name__)
    async def test_delete_activity_catches(self, manager, mock_connector, exc):
        manager._connector = mock_connector
        mock_connector._http_request.side_effect = exc
        result = await manager.delete_activity("c-1", "a-1", SERVICE_URL)
        assert result is False


# ===========================================================================
# Module Exports
# ===========================================================================


class TestModuleExports:
    def test_all_exports(self):
        from aragora.server.handlers.bots.teams import channels

        assert "TeamsChannelManager" in channels.__all__

    def test_class_importable(self):
        from aragora.server.handlers.bots.teams.channels import TeamsChannelManager

        assert TeamsChannelManager is not None
