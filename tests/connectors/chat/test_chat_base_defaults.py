"""
Tests for ChatPlatformConnector default/graceful-degradation methods.

Validates that base class methods return sensible defaults when subclasses
do not override them, and that all operations degrade gracefully rather
than raising NotImplementedError.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.chat.base import ChatPlatformConnector
from aragora.connectors.chat.models import (
    BotCommand,
    ChatChannel,
    ChatUser,
    FileAttachment,
    SendMessageResponse,
    WebhookEvent,
)


# ============================================================================
# Minimal Connector — only implements abstract properties
# ============================================================================


class MinimalConnector(ChatPlatformConnector):
    """Connector that only implements required abstract properties.

    Tests that ALL default method implementations degrade gracefully.
    """

    @property
    def platform_name(self) -> str:
        return "minimal"

    @property
    def platform_display_name(self) -> str:
        return "Minimal Platform"


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    """Clear circuit breakers between tests."""
    from aragora.resilience import _circuit_breakers, _circuit_breakers_lock

    with _circuit_breakers_lock:
        _circuit_breakers.clear()
    yield
    with _circuit_breakers_lock:
        _circuit_breakers.clear()


@pytest.fixture
def connector():
    """MinimalConnector with circuit breaker disabled for speed."""
    return MinimalConnector(bot_token="tok", enable_circuit_breaker=False)


@pytest.fixture
def connector_with_webhook():
    """MinimalConnector with webhook_url configured."""
    return MinimalConnector(
        webhook_url="https://hook.example.com",
        enable_circuit_breaker=False,
    )


# ============================================================================
# Connection Lifecycle
# ============================================================================


class TestConnectionLifecycle:
    """connect(), disconnect(), is_connected."""

    @pytest.mark.asyncio
    async def test_connect_succeeds_when_configured(self, connector):
        result = await connector.connect()
        assert result is True
        assert connector.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_fails_when_unconfigured(self):
        c = MinimalConnector(enable_circuit_breaker=False)
        result = await c.connect()
        assert result is False
        assert c.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self, connector):
        await connector.connect()
        assert connector.is_connected is True
        await connector.disconnect()
        assert connector.is_connected is False

    @pytest.mark.asyncio
    async def test_is_connected_requires_both_config_and_init(self):
        c = MinimalConnector(bot_token="tok", enable_circuit_breaker=False)
        assert c.is_connected is False  # configured but not initialized
        await c.connect()
        assert c.is_connected is True


# ============================================================================
# Generic send() / receive() Wrappers
# ============================================================================


class TestGenericSendReceive:
    """send() and receive() wrapper methods."""

    @pytest.mark.asyncio
    async def test_send_delegates_to_send_message(self, connector_with_webhook):
        """send() calls send_message with the right args."""
        connector_with_webhook._http_request = AsyncMock(
            return_value=(True, {"message_id": "m1"}, None)
        )
        result = await connector_with_webhook.send(
            {"channel_id": "ch1", "text": "hello", "thread_id": "t1"}
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_send_missing_channel_id(self, connector):
        result = await connector.send({"text": "hello"})
        assert result["success"] is False
        assert "channel_id required" in result["error"]

    @pytest.mark.asyncio
    async def test_send_uses_channel_fallback_key(self, connector_with_webhook):
        """send() accepts 'channel' as alternative to 'channel_id'."""
        connector_with_webhook._http_request = AsyncMock(
            return_value=(True, {"message_id": "m1"}, None)
        )
        result = await connector_with_webhook.send({"channel": "ch1", "text": "hi"})
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_send_uses_message_fallback_key(self, connector_with_webhook):
        """send() accepts 'message' as alternative to 'text'."""
        connector_with_webhook._http_request = AsyncMock(
            return_value=(True, {"message_id": "m1"}, None)
        )
        result = await connector_with_webhook.send(
            {"channel_id": "ch1", "message": "hello"}
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_receive_returns_none(self, connector):
        """Chat connectors use webhooks; receive() returns None."""
        result = await connector.receive()
        assert result is None


# ============================================================================
# Capabilities
# ============================================================================


class TestCapabilities:
    """capabilities() method."""

    def test_capabilities_returns_correct_defaults(self, connector):
        caps = connector.capabilities()
        assert caps.can_send is True
        assert caps.can_receive is True
        assert caps.can_search is False
        assert caps.supports_webhooks is True
        assert caps.supports_files is True
        assert caps.supports_rich_text is True
        assert caps.supports_reactions is True
        assert caps.supports_threads is True
        assert caps.supports_voice is True
        assert caps.requires_auth is True

    def test_capabilities_circuit_breaker_flag(self):
        c_on = MinimalConnector(bot_token="tok")
        c_off = MinimalConnector(bot_token="tok", enable_circuit_breaker=False)
        assert c_on.capabilities().has_circuit_breaker is True
        assert c_off.capabilities().has_circuit_breaker is False


# ============================================================================
# Messaging Defaults
# ============================================================================


class TestMessagingDefaults:
    """Default messaging methods that degrade gracefully."""

    @pytest.mark.asyncio
    async def test_send_message_without_webhook_returns_failure(self, connector):
        result = await connector.send_message("ch1", "hello")
        assert result.success is False
        assert "not implemented" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_message_with_webhook_posts(self, connector_with_webhook):
        connector_with_webhook._http_request = AsyncMock(
            return_value=(True, {"message_id": "m1", "timestamp": "123"}, None)
        )
        result = await connector_with_webhook.send_message("ch1", "hello")
        assert result.success is True
        assert result.message_id == "m1"

    @pytest.mark.asyncio
    async def test_update_message_returns_failure(self, connector):
        result = await connector.update_message("ch1", "msg1", "new text")
        assert result.success is False
        assert "not support" in result.error.lower()

    @pytest.mark.asyncio
    async def test_delete_message_returns_false(self, connector):
        result = await connector.delete_message("ch1", "msg1")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_ephemeral_falls_back_to_send_message(self, connector_with_webhook):
        connector_with_webhook._http_request = AsyncMock(
            return_value=(True, {"message_id": "m1"}, None)
        )
        result = await connector_with_webhook.send_ephemeral("ch1", "user1", "psst")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_typing_indicator_returns_false(self, connector):
        result = await connector.send_typing_indicator("ch1")
        assert result is False


# ============================================================================
# File Operations Graceful Degradation
# ============================================================================


class TestFileOpsDefaults:
    """File upload/download default behaviour — RuntimeError, not NotImplementedError."""

    @pytest.mark.asyncio
    async def test_upload_file_raises_runtime_error_without_webhook(self, connector):
        """upload_file raises RuntimeError (not NotImplementedError) when unconfigured."""
        with pytest.raises(RuntimeError, match="file upload not available"):
            await connector.upload_file("ch1", b"data", "test.txt")

    @pytest.mark.asyncio
    async def test_download_file_raises_runtime_error_without_url(self, connector):
        """download_file raises RuntimeError (not NotImplementedError) when no url given."""
        with pytest.raises(RuntimeError, match="file download not available"):
            await connector.download_file("file-1")

    @pytest.mark.asyncio
    async def test_upload_file_with_webhook_posts_file(self, connector_with_webhook):
        connector_with_webhook._http_request = AsyncMock(
            return_value=(True, {"file_id": "f1", "filename": "test.txt"}, None)
        )
        result = await connector_with_webhook.upload_file("ch1", b"data", "test.txt")
        assert isinstance(result, FileAttachment)
        assert result.id == "f1"
        assert result.size == 4

    @pytest.mark.asyncio
    async def test_download_file_with_url_kwarg(self, connector):
        connector._http_request = AsyncMock(
            return_value=(True, b"file-content", None)
        )
        result = await connector.download_file(
            "file-1", url="https://example.com/file.txt"
        )
        assert isinstance(result, FileAttachment)
        assert result.content == b"file-content"
        assert result.size == 12

    @pytest.mark.asyncio
    async def test_send_voice_message_catches_runtime_error(self, connector):
        """send_voice_message catches RuntimeError from upload_file."""
        result = await connector.send_voice_message("ch1", b"audio")
        assert result.success is False
        assert "not available" in result.error

    @pytest.mark.asyncio
    async def test_upload_file_webhook_failure_raises_runtime(self, connector_with_webhook):
        connector_with_webhook._http_request = AsyncMock(
            return_value=(False, None, "Server error")
        )
        with pytest.raises(RuntimeError, match="Server error"):
            await connector_with_webhook.upload_file("ch1", b"data", "test.txt")

    @pytest.mark.asyncio
    async def test_download_file_url_failure_raises_runtime(self, connector):
        connector._http_request = AsyncMock(
            return_value=(False, None, "Network error")
        )
        with pytest.raises(RuntimeError, match="Network error"):
            await connector.download_file(
                "file-1", url="https://example.com/file.txt"
            )


# ============================================================================
# Channel / User Defaults
# ============================================================================


class TestChannelUserDefaults:
    """Default channel/user operations that return sensible defaults."""

    @pytest.mark.asyncio
    async def test_get_channel_info_returns_none(self, connector):
        result = await connector.get_channel_info("ch1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_info_returns_none(self, connector):
        result = await connector.get_user_info("u1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_profile_delegates_to_get_user_info(self, connector):
        result = await connector.get_user_profile("u1")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_users_returns_empty(self, connector):
        users, cursor = await connector.list_users("ch1")
        assert users == []
        assert cursor is None

    @pytest.mark.asyncio
    async def test_list_users_without_channel(self, connector):
        users, cursor = await connector.list_users()
        assert users == []
        assert cursor is None

    @pytest.mark.asyncio
    async def test_create_channel_returns_none(self, connector):
        result = await connector.create_channel("new-channel")
        assert result is None

    @pytest.mark.asyncio
    async def test_send_dm_delegates_to_send_message(self, connector_with_webhook):
        connector_with_webhook._http_request = AsyncMock(
            return_value=(True, {"message_id": "dm1"}, None)
        )
        result = await connector_with_webhook.send_dm("user1", "hello")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_get_voice_message_returns_none(self, connector):
        result = await connector.get_voice_message("file1")
        assert result is None


# ============================================================================
# Reactions / Pinning / Threading Defaults
# ============================================================================


class TestReactionsPinningThreading:
    """Default reactions, pinning, and threading operations."""

    @pytest.mark.asyncio
    async def test_react_to_message_returns_false(self, connector):
        result = await connector.react_to_message("ch1", "msg1", "thumbsup")
        assert result is False

    @pytest.mark.asyncio
    async def test_remove_reaction_returns_false(self, connector):
        result = await connector.remove_reaction("ch1", "msg1", "thumbsup")
        assert result is False

    @pytest.mark.asyncio
    async def test_pin_message_returns_false(self, connector):
        result = await connector.pin_message("ch1", "msg1")
        assert result is False

    @pytest.mark.asyncio
    async def test_unpin_message_returns_false(self, connector):
        result = await connector.unpin_message("ch1", "msg1")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_pinned_messages_returns_empty(self, connector):
        result = await connector.get_pinned_messages("ch1")
        assert result == []

    @pytest.mark.asyncio
    async def test_create_thread_delegates_to_send_message(self, connector_with_webhook):
        connector_with_webhook._http_request = AsyncMock(
            return_value=(True, {"message_id": "thread1"}, None)
        )
        result = await connector_with_webhook.create_thread("ch1", "msg1", "reply")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_receive_messages_yields_nothing(self, connector):
        """receive_messages() is an empty async generator by default."""
        messages = []
        async for msg in connector.receive_messages("ch1"):
            messages.append(msg)
        assert messages == []


# ============================================================================
# Slash Command Handling
# ============================================================================


class TestSlashCommandHandling:
    """handle_slash_command() builds BotCommand from raw input."""

    @pytest.mark.asyncio
    async def test_handle_slash_command_parses_args(self, connector):
        cmd = await connector.handle_slash_command(
            command_name="debate",
            channel_id="ch1",
            user_id="u1",
            text="Should we use React?",
            response_url="https://response.url",
        )
        assert isinstance(cmd, BotCommand)
        assert cmd.name == "debate"
        assert cmd.args == ["Should", "we", "use", "React?"]
        assert cmd.platform == "minimal"
        assert cmd.response_url == "https://response.url"
        assert cmd.user.id == "u1"
        assert cmd.channel.id == "ch1"

    @pytest.mark.asyncio
    async def test_handle_slash_command_empty_text(self, connector):
        cmd = await connector.handle_slash_command(
            command_name="status",
            channel_id="ch1",
            user_id="u1",
        )
        assert cmd.args == []
        assert cmd.text == "/status"

    @pytest.mark.asyncio
    async def test_respond_to_command_via_response_url(self, connector):
        connector._http_request = AsyncMock(
            return_value=(True, {}, None)
        )
        cmd = BotCommand(
            name="test",
            text="/test",
            args=[],
            user=ChatUser(id="u1", platform="minimal"),
            channel=ChatChannel(id="ch1", platform="minimal"),
            platform="minimal",
            response_url="https://response.example.com",
        )
        result = await connector.respond_to_command(cmd, "response text")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_respond_to_command_falls_back_to_send(self, connector_with_webhook):
        connector_with_webhook._http_request = AsyncMock(
            return_value=(True, {"message_id": "m1"}, None)
        )
        cmd = BotCommand(
            name="test",
            text="/test",
            args=[],
            user=ChatUser(id="u1", platform="minimal"),
            channel=ChatChannel(id="ch1", platform="minimal"),
            platform="minimal",
        )
        result = await connector_with_webhook.respond_to_command(cmd, "response")
        assert result.success is True


# ============================================================================
# Interaction Response
# ============================================================================


class TestInteractionResponse:
    """respond_to_interaction() default implementation."""

    @pytest.mark.asyncio
    async def test_respond_via_response_url(self, connector):
        from aragora.connectors.chat.models import UserInteraction

        connector._http_request = AsyncMock(
            return_value=(True, {}, None)
        )
        interaction = UserInteraction(
            id="int1",
            platform="minimal",
            interaction_type="button_click",
            action_id="act1",
            response_url="https://response.example.com",
        )
        result = await connector.respond_to_interaction(interaction, "clicked!")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_respond_replace_original_via_update(self, connector):
        from aragora.connectors.chat.models import UserInteraction

        interaction = UserInteraction(
            id="int1",
            platform="minimal",
            interaction_type="button_click",
            action_id="act1",
            channel=ChatChannel(id="ch1", platform="minimal"),
            message_id="msg1",
        )
        # update_message returns failure by default
        result = await connector.respond_to_interaction(
            interaction, "updated!", replace_original=True
        )
        assert result.success is False  # default update_message fails

    @pytest.mark.asyncio
    async def test_respond_no_channel_returns_failure(self, connector):
        from aragora.connectors.chat.models import UserInteraction

        interaction = UserInteraction(
            id="int1",
            platform="minimal",
            interaction_type="button_click",
            action_id="act1",
        )
        result = await connector.respond_to_interaction(interaction, "text")
        assert result.success is False
        assert "No channel" in result.error


# ============================================================================
# Webhook Defaults
# ============================================================================


class TestWebhookDefaults:
    """Default webhook verification and parsing."""

    def test_verify_webhook_without_secret_in_dev(self, connector):
        """In dev/test mode, missing signing_secret returns True with warning."""
        c = MinimalConnector(enable_circuit_breaker=False)
        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}):
            assert c.verify_webhook({}, b"body") is True

    def test_verify_webhook_without_secret_in_production(self):
        """In production, missing signing_secret fails closed."""
        c = MinimalConnector(enable_circuit_breaker=False)
        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
            assert c.verify_webhook({}, b"body") is False

    def test_verify_webhook_valid_signature(self):
        import hashlib
        import hmac

        secret = "test-secret"
        body = b'{"event": "message"}'
        sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

        c = MinimalConnector(signing_secret=secret, enable_circuit_breaker=False)
        assert c.verify_webhook({"X-Signature-256": f"sha256={sig}"}, body) is True

    def test_verify_webhook_invalid_signature(self):
        c = MinimalConnector(signing_secret="secret", enable_circuit_breaker=False)
        assert c.verify_webhook({"X-Signature-256": "sha256=bad"}, b"body") is False

    def test_verify_webhook_missing_header(self):
        c = MinimalConnector(signing_secret="secret", enable_circuit_breaker=False)
        assert c.verify_webhook({}, b"body") is False

    def test_parse_webhook_event_valid_json(self, connector):
        event = connector.parse_webhook_event(
            {}, b'{"type": "message", "text": "hello"}'
        )
        assert isinstance(event, WebhookEvent)
        assert event.platform == "minimal"
        assert event.event_type == "message"
        assert event.raw_payload["text"] == "hello"

    def test_parse_webhook_event_invalid_json(self, connector):
        event = connector.parse_webhook_event({}, b"not json!")
        assert event.event_type == "error"
        assert "invalid_json" in event.metadata.get("error", "")

    def test_parse_webhook_event_empty_body(self, connector):
        event = connector.parse_webhook_event({}, b"")
        assert event.event_type == "unknown"

    def test_parse_webhook_event_nested_type(self, connector):
        """Extracts event_type from nested 'event.type' field."""
        event = connector.parse_webhook_event(
            {}, b'{"event": {"type": "app_mention"}}'
        )
        assert event.event_type == "app_mention"


# ============================================================================
# Rich Content Formatting Defaults
# ============================================================================


class TestRichContentDefaults:
    """Default format_blocks() and format_button() implementations."""

    def test_format_blocks_with_all_parts(self, connector):
        from aragora.connectors.chat.models import MessageButton

        blocks = connector.format_blocks(
            title="Report",
            body="Summary text",
            fields=[("Status", "Active"), ("Priority", "High")],
            actions=[
                MessageButton(text="Approve", action_id="approve", style="primary"),
            ],
        )
        assert len(blocks) == 4
        assert blocks[0] == {"type": "header", "text": "Report"}
        assert blocks[1] == {"type": "section", "text": "Summary text"}
        assert blocks[2]["type"] == "fields"
        assert len(blocks[2]["items"]) == 2
        assert blocks[3]["type"] == "actions"

    def test_format_blocks_empty(self, connector):
        blocks = connector.format_blocks()
        assert blocks == []

    def test_format_blocks_filters_none_fields(self, connector):
        blocks = connector.format_blocks(
            fields=[("Status", "Active"), (None, "ignored"), ("Key", None)]
        )
        assert len(blocks) == 1
        assert len(blocks[0]["items"]) == 1  # Only ("Status", "Active") passes

    def test_format_button_defaults(self, connector):
        btn = connector.format_button("Click", "action-1")
        assert btn == {"type": "button", "text": "Click", "action_id": "action-1"}

    def test_format_button_with_all_options(self, connector):
        btn = connector.format_button(
            "Delete", "delete-1", value="item-42", style="danger", url="https://x.com"
        )
        assert btn["value"] == "item-42"
        assert btn["style"] == "danger"
        assert btn["url"] == "https://x.com"


# ============================================================================
# Evidence Collection Defaults
# ============================================================================


class TestEvidenceDefaults:
    """Default evidence collection methods."""

    @pytest.mark.asyncio
    async def test_collect_evidence_returns_empty(self, connector):
        result = await connector.collect_evidence("ch1", query="test")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_channel_history_returns_empty(self, connector):
        result = await connector.get_channel_history("ch1", limit=10)
        assert result == []


# ============================================================================
# Session Integration Defaults
# ============================================================================


class TestSessionDefaults:
    """Session management integration defaults."""

    @pytest.mark.asyncio
    async def test_get_or_create_session_without_manager(self, connector):
        """Returns None when session manager not available."""
        with patch.object(
            connector, "_get_session_manager", return_value=None
        ):
            result = await connector.get_or_create_session("user1")
            assert result is None

    @pytest.mark.asyncio
    async def test_link_debate_to_session_without_manager(self, connector):
        with patch.object(
            connector, "_get_session_manager", return_value=None
        ):
            result = await connector.link_debate_to_session("user1", "debate1")
            assert result is None

    @pytest.mark.asyncio
    async def test_find_sessions_for_debate_without_manager(self, connector):
        with patch.object(
            connector, "_get_session_manager", return_value=None
        ):
            result = await connector.find_sessions_for_debate("debate1")
            assert result == []

    @pytest.mark.asyncio
    async def test_route_debate_result_without_manager(self, connector):
        with patch.object(
            connector, "_get_session_manager", return_value=None
        ):
            result = await connector.route_debate_result("debate1", "consensus text")
            assert result == []

    @pytest.mark.asyncio
    async def test_get_or_create_session_with_manager(self, connector):
        mock_session = MagicMock(session_id="sess-1")
        mock_manager = AsyncMock()
        mock_manager.get_or_create_session = AsyncMock(return_value=mock_session)

        with patch.object(
            connector, "_get_session_manager", return_value=mock_manager
        ):
            result = await connector.get_or_create_session("user1")
            assert result.session_id == "sess-1"
            mock_manager.get_or_create_session.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_link_debate_and_find_sessions(self, connector):
        mock_session = MagicMock(session_id="sess-1", channel="minimal")
        mock_manager = AsyncMock()
        mock_manager.get_or_create_session = AsyncMock(return_value=mock_session)
        mock_manager.link_debate = AsyncMock()
        mock_manager.find_sessions_for_debate = AsyncMock(return_value=[mock_session])

        with patch.object(
            connector, "_get_session_manager", return_value=mock_manager
        ):
            session_id = await connector.link_debate_to_session("user1", "debate1")
            assert session_id == "sess-1"
            mock_manager.link_debate.assert_awaited_once_with("sess-1", "debate1")

            sessions = await connector.find_sessions_for_debate("debate1")
            assert len(sessions) == 1


# ============================================================================
# NotImplementedError Elimination Verification
# ============================================================================


class TestNoNotImplementedError:
    """Verify that NO base class method raises NotImplementedError.

    Only abstract properties (platform_name, platform_display_name) should
    raise NotImplementedError, and those are tested implicitly by MinimalConnector
    implementing them.
    """

    @pytest.mark.asyncio
    async def test_all_async_defaults_return_without_error(self, connector):
        """Call every async default method — none should raise NotImplementedError."""
        # Messaging
        await connector.send_typing_indicator("ch1")
        await connector.update_message("ch1", "msg1", "text")
        await connector.delete_message("ch1", "msg1")

        # Channel/User
        await connector.get_channel_info("ch1")
        await connector.get_user_info("u1")
        await connector.get_user_profile("u1")
        await connector.list_users("ch1")
        await connector.create_channel("test")
        await connector.get_voice_message("f1")

        # Reactions/Pinning
        await connector.react_to_message("ch1", "msg1", "thumbsup")
        await connector.remove_reaction("ch1", "msg1", "thumbsup")
        await connector.pin_message("ch1", "msg1")
        await connector.unpin_message("ch1", "msg1")
        await connector.get_pinned_messages("ch1")

        # Evidence
        await connector.collect_evidence("ch1")
        await connector.get_channel_history("ch1")

        # Connection
        await connector.connect()
        await connector.disconnect()
        await connector.test_connection()
        await connector.get_health()
        await connector.receive()

    def test_all_sync_defaults_return_without_error(self, connector):
        """Call every sync default method — none should raise NotImplementedError."""
        connector.capabilities()
        connector.format_blocks()
        connector.format_button("btn", "act")
        connector._format_timestamp_for_api(None)

    @pytest.mark.asyncio
    async def test_file_ops_raise_runtime_not_not_implemented(self, connector):
        """upload_file and download_file raise RuntimeError, not NotImplementedError."""
        with pytest.raises(RuntimeError):
            await connector.upload_file("ch1", b"data", "f.txt")

        with pytest.raises(RuntimeError):
            await connector.download_file("f1")
