"""
Tests for OpenClaw action execution.

Tests cover:
- Action execution (success, session not found, session inactive)
- RBAC permission checks on actions
- Capability filtering
- Custom action handlers (sync and async)
- Action timeout handling
- Action callbacks
- Channel mapping and formatting
- Protocol translation
- Message sending
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.gateway.openclaw.adapter import (
    ActionResult,
    ActionStatus,
    ChannelFormatter,
    ChannelMapping,
    OpenClawAction,
    OpenClawActionType,
    OpenClawAdapter,
    OpenClawChannel,
    OpenClawMessage,
    OpenClawSession,
    SessionState,
)
from aragora.gateway.openclaw.capabilities import (
    CapabilityCategory,
    CapabilityCheckResult,
    CapabilityFilter,
    CapabilityRule,
)
from aragora.gateway.openclaw.protocol import AuthorizationContext


# ============================================================================
# Fixtures
# ============================================================================


class MockRBACChecker:
    """Mock RBAC checker."""

    def __init__(self, allowed: bool = True):
        self._allowed = allowed

    def check_permission(self, actor_id, permission, resource_id=None):
        return self._allowed

    async def check_permission_async(self, actor_id, permission, resource_id=None):
        return self._allowed


def _permissive_filter() -> CapabilityFilter:
    """Create a capability filter that allows all mapped capabilities."""
    extra_rules = {
        cap: CapabilityRule(
            name=cap,
            category=CapabilityCategory.ALWAYS_ALLOWED,
            description=f"Test rule for {cap}",
        )
        for cap in [
            "browser_automation",
            "file_system_read",
            "file_system_write",
            "network_external",
            "code_execution",
            "database_read",
            "database_write",
            "scheduler_manage",
            "scheduler_read",
        ]
    }
    from aragora.gateway.openclaw.capabilities import DEFAULT_CAPABILITY_RULES

    rules = {**DEFAULT_CAPABILITY_RULES, **extra_rules}
    return CapabilityFilter(rules=rules)


@pytest.fixture
def adapter():
    """Create a basic adapter with permissive capability filter."""
    return OpenClawAdapter(
        openclaw_endpoint="http://test:8081",
        session_timeout_seconds=3600,
        action_default_timeout=30,
        capability_filter=_permissive_filter(),
    )


@pytest.fixture
async def session(adapter):
    """Create a session and return (adapter, session) pair."""
    s = await adapter.create_session(
        user_id="user-1",
        channel=OpenClawChannel.TELEGRAM,
        tenant_id="tenant-1",
    )
    return s


# ============================================================================
# Action Execution - Basic
# ============================================================================


class TestExecuteAction:
    """Test action execution basics."""

    @pytest.mark.asyncio
    async def test_execute_action_session_not_found(self, adapter):
        """Test action fails when session doesn't exist."""
        action = OpenClawAction(
            action_type=OpenClawActionType.BROWSER_NAVIGATE,
            parameters={"url": "https://example.com"},
        )

        result = await adapter.execute_action("nonexistent", action)
        assert result.status == ActionStatus.FAILED
        assert "Session not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_action_session_terminated(self, adapter, session):
        """Test action fails when session is terminated."""
        await adapter.close_session(session.session_id)

        action = OpenClawAction(
            action_type=OpenClawActionType.BROWSER_NAVIGATE,
            parameters={"url": "https://example.com"},
        )

        result = await adapter.execute_action(session.session_id, action)
        assert result.status == ActionStatus.FAILED
        assert "terminated" in result.error

    @pytest.mark.asyncio
    async def test_execute_action_with_handler(self, adapter, session):
        """Test action with registered handler."""

        def my_handler(action, sess):
            return {"navigated": True, "url": action.parameters.get("url")}

        adapter.register_action_handler(OpenClawActionType.BROWSER_NAVIGATE, my_handler)

        action = OpenClawAction(
            action_type=OpenClawActionType.BROWSER_NAVIGATE,
            parameters={"url": "https://example.com"},
        )

        result = await adapter.execute_action(session.session_id, action)
        assert result.status == ActionStatus.COMPLETED
        assert result.result["navigated"] is True
        assert result.result["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_execute_action_with_async_handler(self, adapter, session):
        """Test action with async handler."""

        async def my_handler(action, sess):
            await asyncio.sleep(0.01)
            return {"screenshot_taken": True}

        adapter.register_action_handler(OpenClawActionType.BROWSER_SCREENSHOT, my_handler)

        action = OpenClawAction(
            action_type=OpenClawActionType.BROWSER_SCREENSHOT,
            parameters={},
        )

        result = await adapter.execute_action(session.session_id, action)
        assert result.status == ActionStatus.COMPLETED
        assert result.result["screenshot_taken"] is True

    @pytest.mark.asyncio
    async def test_execute_action_records_timing(self, adapter, session):
        """Test that execution timing is recorded."""

        def my_handler(action, sess):
            return {}

        adapter.register_action_handler(OpenClawActionType.BROWSER_CLICK, my_handler)

        action = OpenClawAction(
            action_type=OpenClawActionType.BROWSER_CLICK,
            parameters={"selector": "#btn"},
        )

        result = await adapter.execute_action(session.session_id, action)
        assert result.execution_time_ms >= 0
        assert result.started_at is not None
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_execute_action_generates_unique_ids(self, adapter, session):
        """Test that each action gets a unique ID."""

        def my_handler(action, sess):
            return {}

        adapter.register_action_handler(OpenClawActionType.BROWSER_CLICK, my_handler)

        action = OpenClawAction(
            action_type=OpenClawActionType.BROWSER_CLICK,
            parameters={},
        )

        r1 = await adapter.execute_action(session.session_id, action)
        r2 = await adapter.execute_action(session.session_id, action)
        assert r1.action_id != r2.action_id


# ============================================================================
# Action Execution - RBAC
# ============================================================================


class TestActionRBAC:
    """Test RBAC enforcement on actions."""

    @pytest.mark.asyncio
    async def test_action_rbac_denied(self):
        """Test action is blocked when RBAC denies permission."""
        adapter = OpenClawAdapter(
            openclaw_endpoint="http://test:8081",
            rbac_checker=MockRBACChecker(allowed=False),
            capability_filter=_permissive_filter(),
        )

        session = await adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
        )

        auth = AuthorizationContext(actor_id="user-1")
        action = OpenClawAction(
            action_type=OpenClawActionType.CODE_RUN,
            parameters={"code": "print('hi')"},
        )

        result = await adapter.execute_action(
            session.session_id,
            action,
            auth_context=auth,
        )

        assert result.status == ActionStatus.FAILED
        assert "Permission denied" in result.error

    @pytest.mark.asyncio
    async def test_action_rbac_allowed(self):
        """Test action proceeds when RBAC allows it."""
        adapter = OpenClawAdapter(
            openclaw_endpoint="http://test:8081",
            rbac_checker=MockRBACChecker(allowed=True),
            capability_filter=_permissive_filter(),
        )

        # Register handler so action can complete
        adapter.register_action_handler(
            OpenClawActionType.CODE_RUN,
            lambda a, s: {"output": "hi"},
        )

        session = await adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
        )

        auth = AuthorizationContext(actor_id="user-1")
        action = OpenClawAction(
            action_type=OpenClawActionType.CODE_RUN,
            parameters={"code": "print('hi')"},
        )

        result = await adapter.execute_action(
            session.session_id,
            action,
            auth_context=auth,
        )

        assert result.status == ActionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_action_no_auth_skips_rbac(self):
        """Test that RBAC is skipped without auth context."""
        adapter = OpenClawAdapter(
            openclaw_endpoint="http://test:8081",
            rbac_checker=MockRBACChecker(allowed=False),
            capability_filter=_permissive_filter(),
        )

        adapter.register_action_handler(
            OpenClawActionType.BROWSER_NAVIGATE,
            lambda a, s: {"ok": True},
        )

        session = await adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
        )

        action = OpenClawAction(
            action_type=OpenClawActionType.BROWSER_NAVIGATE,
            parameters={"url": "https://example.com"},
        )

        # No auth_context = no RBAC check
        result = await adapter.execute_action(session.session_id, action)
        assert result.status == ActionStatus.COMPLETED


# ============================================================================
# Action Execution - Capability Filter
# ============================================================================


class TestActionCapabilityFilter:
    """Test capability filtering on actions."""

    @pytest.mark.asyncio
    async def test_blocked_capability(self):
        """Test that blocked capabilities prevent action execution."""
        cap_filter = CapabilityFilter(blocked_override={"browser_automation"})
        adapter = OpenClawAdapter(
            openclaw_endpoint="http://test:8081",
            capability_filter=cap_filter,
        )

        session = await adapter.create_session(
            user_id="user-1",
            channel=OpenClawChannel.WEB,
        )

        action = OpenClawAction(
            action_type=OpenClawActionType.BROWSER_NAVIGATE,
            parameters={"url": "https://example.com"},
        )

        result = await adapter.execute_action(session.session_id, action)
        assert result.status == ActionStatus.FAILED
        assert "Capability blocked" in result.error

    @pytest.mark.asyncio
    async def test_custom_action_no_capability_mapping(self, adapter, session):
        """Test that custom actions without capability mapping are not blocked by filter."""
        # Custom actions don't map to a capability, so filter doesn't block
        adapter.register_action_handler("custom", lambda a, s: {"custom": True})

        action = OpenClawAction(
            action_type=OpenClawActionType.CUSTOM,
            parameters={"data": "test"},
        )

        result = await adapter.execute_action(session.session_id, action)
        assert result.status == ActionStatus.COMPLETED


# ============================================================================
# Action Execution - Error Handling
# ============================================================================


class TestActionErrorHandling:
    """Test action error handling."""

    @pytest.mark.asyncio
    async def test_handler_exception_returns_failed(self, adapter, session):
        """Test that handler exceptions result in FAILED status."""

        def failing_handler(action, sess):
            raise ValueError("something went wrong")

        adapter.register_action_handler(OpenClawActionType.NOTIFY_SEND, failing_handler)

        action = OpenClawAction(
            action_type=OpenClawActionType.NOTIFY_SEND,
            parameters={"to": "user-2"},
        )

        result = await adapter.execute_action(session.session_id, action)
        assert result.status == ActionStatus.FAILED
        assert "something went wrong" in result.error

    @pytest.mark.asyncio
    async def test_action_timeout(self, adapter, session):
        """Test that actions timeout correctly."""

        async def slow_handler(action, sess):
            await asyncio.sleep(10)
            return {}

        adapter.register_action_handler(OpenClawActionType.HTTP_REQUEST, slow_handler)

        action = OpenClawAction(
            action_type=OpenClawActionType.HTTP_REQUEST,
            parameters={"url": "https://slow.example.com"},
            timeout=0.1,  # Very short timeout
        )

        result = await adapter.execute_action(session.session_id, action)
        assert result.status == ActionStatus.TIMEOUT


# ============================================================================
# Action Callbacks
# ============================================================================


class TestActionCallbacks:
    """Test action completion callbacks."""

    @pytest.mark.asyncio
    async def test_action_triggers_callback(self, adapter, session):
        """Test that action completion triggers callbacks."""
        results = []

        def on_action(result):
            results.append(result)

        adapter.register_action_handler(
            OpenClawActionType.NOTIFY_SEND,
            lambda a, s: {"sent": True},
        )
        adapter.add_action_callback(on_action)

        action = OpenClawAction(
            action_type=OpenClawActionType.NOTIFY_SEND,
            parameters={"to": "user-2", "message": "hello"},
        )

        await adapter.execute_action(session.session_id, action)

        assert len(results) == 1
        assert results[0].status == ActionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_action_callback_error_does_not_propagate(self, adapter, session):
        """Test that callback errors don't break action execution."""

        def bad_callback(result):
            raise RuntimeError("callback error")

        adapter.register_action_handler(
            OpenClawActionType.NOTIFY_SEND,
            lambda a, s: {"sent": True},
        )
        adapter.add_action_callback(bad_callback)

        action = OpenClawAction(
            action_type=OpenClawActionType.NOTIFY_SEND,
            parameters={},
        )

        # Should not raise
        result = await adapter.execute_action(session.session_id, action)
        assert result.status == ActionStatus.COMPLETED


# ============================================================================
# Channel Mapping
# ============================================================================


class TestChannelMapping:
    """Test channel mapping operations."""

    def test_default_channel_mappings(self, adapter):
        """Test that default channel mappings are registered."""
        mapping = adapter.get_channel_mapping("telegram")
        assert mapping is not None
        assert mapping.openclaw_channel == OpenClawChannel.TELEGRAM

    def test_all_default_channels_mapped(self, adapter):
        """Test all default channels have mappings."""
        for channel in OpenClawChannel:
            mapping = adapter.get_channel_mapping(channel.value)
            assert mapping is not None, f"Missing mapping for {channel.value}"

    def test_register_custom_mapping(self, adapter):
        """Test registering a custom channel mapping."""
        custom = ChannelMapping("my_channel", "my_openclaw_channel", "custom")
        adapter.register_channel_mapping(custom)

        mapping = adapter.get_channel_mapping("my_channel")
        assert mapping is not None
        assert mapping.openclaw_channel == "my_openclaw_channel"

    def test_map_aragora_to_openclaw(self, adapter):
        """Test mapping Aragora channel to OpenClaw."""
        oc = adapter.map_aragora_to_openclaw_channel("slack")
        assert oc == OpenClawChannel.SLACK

    def test_map_openclaw_to_aragora(self, adapter):
        """Test mapping OpenClaw channel to Aragora."""
        aragora = adapter.map_openclaw_to_aragora_channel(OpenClawChannel.DISCORD)
        assert aragora == "discord"

    def test_map_unknown_channel_returns_same(self, adapter):
        """Test that unknown channels pass through."""
        oc = adapter.map_aragora_to_openclaw_channel("unknown_channel")
        assert oc == "unknown_channel"

    def test_get_channel_mapping_none_for_unknown(self, adapter):
        """Test that unknown channels return None."""
        mapping = adapter.get_channel_mapping("nonexistent")
        assert mapping is None


# ============================================================================
# Formatters
# ============================================================================


class TestFormatters:
    """Test formatter management."""

    def test_default_formatters_registered(self, adapter):
        """Test that default formatters are registered."""
        from aragora.gateway.openclaw.adapter import (
            WhatsAppFormatter,
            TelegramFormatter,
            SlackFormatter,
            DiscordFormatter,
        )

        expected = {
            "whatsapp": WhatsAppFormatter,
            "telegram": TelegramFormatter,
            "slack": SlackFormatter,
            "discord": DiscordFormatter,
        }
        for channel, cls in expected.items():
            formatter = adapter.get_formatter(channel)
            assert isinstance(formatter, cls), f"{channel} formatter should be {cls.__name__}"

    def test_register_custom_formatter(self, adapter):
        """Test registering a custom formatter."""

        class CustomFormatter(ChannelFormatter):
            def format_outgoing(self, message, session):
                return {"custom": True, "content": message.content}

        adapter.register_formatter("custom_chan", CustomFormatter())
        formatter = adapter.get_formatter("custom_chan")
        assert isinstance(formatter, CustomFormatter)

    def test_unknown_channel_gets_default_formatter(self, adapter):
        """Test that unknown channels get the base ChannelFormatter."""
        formatter = adapter.get_formatter("totally_unknown")
        assert isinstance(formatter, ChannelFormatter)


# ============================================================================
# Protocol Translation
# ============================================================================


class TestProtocolTranslation:
    """Test message conversion between formats."""

    @pytest.mark.asyncio
    async def test_convert_aragora_message(self, adapter, session):
        """Test converting Aragora message to OpenClaw format."""
        aragora_msg = {
            "message_id": "msg-1",
            "type": "text",
            "content": "Hello!",
            "metadata": {"key": "val"},
        }

        openclaw_msg = adapter.convert_aragora_message(aragora_msg, session)
        assert openclaw_msg.message_id == "msg-1"
        assert openclaw_msg.type == "text"
        assert openclaw_msg.content == "Hello!"
        assert openclaw_msg.channel == session.channel
        assert openclaw_msg.sender_id == session.user_id

    @pytest.mark.asyncio
    async def test_convert_openclaw_message(self, adapter):
        """Test converting OpenClaw message to Aragora format."""
        msg = OpenClawMessage(
            message_id="msg-2",
            type="image",
            content="https://example.com/image.png",
            channel=OpenClawChannel.SLACK,
            sender_id="bot-1",
        )

        aragora_msg = adapter.convert_openclaw_message(msg)
        assert aragora_msg["message_id"] == "msg-2"
        assert aragora_msg["type"] == "image"
        assert aragora_msg["channel"] == "slack"
        assert aragora_msg["sender_id"] == "bot-1"

    @pytest.mark.asyncio
    async def test_convert_aragora_message_defaults(self, adapter, session):
        """Test defaults when converting partial Aragora message."""
        aragora_msg = {"content": "Hello!"}

        openclaw_msg = adapter.convert_aragora_message(aragora_msg, session)
        assert openclaw_msg.type == "text"
        assert openclaw_msg.sender_id == session.user_id
        assert openclaw_msg.message_id  # Should have auto-generated ID


# ============================================================================
# Message Sending
# ============================================================================


class TestSendMessage:
    """Test message sending."""

    @pytest.mark.asyncio
    async def test_send_message_session_not_found(self, adapter):
        """Test sending message to nonexistent session raises."""
        msg = OpenClawMessage(
            message_id="msg-1",
            type="text",
            content="Hello",
            channel=OpenClawChannel.WEB,
        )

        with pytest.raises(ValueError, match="Session not found"):
            await adapter.send_message("nonexistent", msg)

    @pytest.mark.asyncio
    async def test_send_message_session_terminated(self, adapter, session):
        """Test sending message to terminated session raises."""
        await adapter.close_session(session.session_id)

        msg = OpenClawMessage(
            message_id="msg-1",
            type="text",
            content="Hello",
            channel=OpenClawChannel.TELEGRAM,
        )

        with pytest.raises(ValueError, match="terminated"):
            await adapter.send_message(session.session_id, msg)

    @pytest.mark.asyncio
    async def test_send_message_formats_and_sends(self, adapter, session):
        """Test that send_message formats and forwards the message."""
        from unittest.mock import patch

        msg = OpenClawMessage(
            message_id="msg-1",
            type="text",
            content="Hello from test",
            channel=OpenClawChannel.TELEGRAM,
        )

        mock_result = {"sent": True, "message_id": "msg-1"}
        with patch.object(adapter, "_send_to_openclaw", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = mock_result
            result = await adapter.send_message(session.session_id, msg)

        assert result == mock_result
        mock_send.assert_called_once()


# ============================================================================
# Action Handler Registration
# ============================================================================


class TestActionHandlerRegistration:
    """Test action handler registration."""

    def test_register_handler_with_enum(self, adapter):
        """Test registering handler with enum action type."""

        def handler(action, session):
            return {}

        adapter.register_action_handler(OpenClawActionType.BROWSER_NAVIGATE, handler)
        assert "browser_navigate" in adapter._action_handlers

    def test_register_handler_with_string(self, adapter):
        """Test registering handler with string action type."""

        def handler(action, session):
            return {}

        adapter.register_action_handler("custom_action", handler)
        assert "custom_action" in adapter._action_handlers

    def test_overwrite_handler(self, adapter):
        """Test that registering a handler overwrites the previous one."""

        def handler1(a, s):
            return {"v": 1}

        def handler2(a, s):
            return {"v": 2}

        adapter.register_action_handler(OpenClawActionType.BROWSER_CLICK, handler1)
        adapter.register_action_handler(OpenClawActionType.BROWSER_CLICK, handler2)

        assert adapter._action_handlers["browser_click"] is handler2


# ============================================================================
# Action-to-Capability Mapping
# ============================================================================


class TestActionToCapability:
    """Test action type to capability mapping."""

    def test_browser_actions_map_to_browser_automation(self, adapter):
        """Test browser actions map to browser_automation capability."""
        for action_type in [
            OpenClawActionType.BROWSER_NAVIGATE,
            OpenClawActionType.BROWSER_CLICK,
            OpenClawActionType.BROWSER_TYPE,
            OpenClawActionType.BROWSER_SCREENSHOT,
        ]:
            cap = adapter._action_to_capability(action_type)
            assert cap == "browser_automation", f"{action_type} should map to browser_automation"

    def test_file_read_maps_to_fs_read(self, adapter):
        """Test file read maps to file_system_read."""
        assert adapter._action_to_capability(OpenClawActionType.FILE_READ) == "file_system_read"
        assert adapter._action_to_capability(OpenClawActionType.FILE_LIST) == "file_system_read"

    def test_file_write_maps_to_fs_write(self, adapter):
        """Test file write/delete maps to file_system_write."""
        assert adapter._action_to_capability(OpenClawActionType.FILE_WRITE) == "file_system_write"
        assert adapter._action_to_capability(OpenClawActionType.FILE_DELETE) == "file_system_write"

    def test_code_actions_map_to_code_execution(self, adapter):
        """Test code actions map to code_execution."""
        assert adapter._action_to_capability(OpenClawActionType.CODE_RUN) == "code_execution"
        assert adapter._action_to_capability(OpenClawActionType.CODE_EVAL) == "code_execution"

    def test_db_actions_map_correctly(self, adapter):
        """Test database actions map correctly."""
        assert adapter._action_to_capability(OpenClawActionType.DB_QUERY) == "database_read"
        assert adapter._action_to_capability(OpenClawActionType.DB_EXECUTE) == "database_write"

    def test_notify_actions_have_no_capability(self, adapter):
        """Test notification actions don't map to a capability."""
        assert adapter._action_to_capability(OpenClawActionType.NOTIFY_SEND) is None

    def test_custom_action_has_no_capability(self, adapter):
        """Test custom actions don't map to a capability."""
        assert adapter._action_to_capability(OpenClawActionType.CUSTOM) is None

    def test_string_action_type(self, adapter):
        """Test capability mapping with string action type."""
        assert adapter._action_to_capability("browser_navigate") == "browser_automation"
        assert adapter._action_to_capability("unknown_action") is None
