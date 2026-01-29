"""
Tests for Local Gateway modules (Moltbot parity).

Tests the consumer/device interface layer including:
- InboxAggregator (unified inbox)
- DeviceRegistry (device capabilities)
- AgentRouter (message routing)
- LocalGateway (server integration)
"""

from __future__ import annotations

import pytest

from aragora.gateway.inbox import (
    InboxAggregator,
    InboxMessage,
    InboxThread,
    MessagePriority,
)
from aragora.gateway.device_registry import DeviceNode, DeviceRegistry, DeviceStatus
from aragora.gateway.router import AgentRouter, RoutingRule
from aragora.gateway.server import AgentResponse, GatewayConfig, LocalGateway


# =============================================================================
# InboxMessage Tests
# =============================================================================


class TestInboxMessage:
    """Test InboxMessage dataclass."""

    def test_message_creation(self):
        msg = InboxMessage(
            message_id="msg-1",
            channel="slack",
            sender="user@example.com",
            content="Hello world",
        )
        assert msg.message_id == "msg-1"
        assert msg.channel == "slack"
        assert msg.is_read is False
        assert msg.priority == MessagePriority.NORMAL


class TestInboxAggregator:
    """Test InboxAggregator."""

    @pytest.fixture
    def inbox(self):
        return InboxAggregator(max_size=100)

    @pytest.mark.asyncio
    async def test_add_message(self, inbox):
        msg = InboxMessage(message_id="m1", channel="slack", sender="alice", content="hi")
        await inbox.add_message(msg)
        assert await inbox.get_size() == 1

    @pytest.mark.asyncio
    async def test_get_messages(self, inbox):
        await inbox.add_message(
            InboxMessage(message_id="m1", channel="slack", sender="a", content="1")
        )
        await inbox.add_message(
            InboxMessage(message_id="m2", channel="teams", sender="b", content="2")
        )
        msgs = await inbox.get_messages()
        assert len(msgs) == 2

    @pytest.mark.asyncio
    async def test_get_messages_by_channel(self, inbox):
        await inbox.add_message(
            InboxMessage(message_id="m1", channel="slack", sender="a", content="1")
        )
        await inbox.add_message(
            InboxMessage(message_id="m2", channel="teams", sender="b", content="2")
        )
        msgs = await inbox.get_messages(channel="slack")
        assert len(msgs) == 1
        assert msgs[0].channel == "slack"

    @pytest.mark.asyncio
    async def test_get_messages_unread_only(self, inbox):
        await inbox.add_message(
            InboxMessage(message_id="m1", channel="slack", sender="a", content="1")
        )
        await inbox.add_message(
            InboxMessage(message_id="m2", channel="slack", sender="b", content="2")
        )
        await inbox.mark_read(["m1"])
        msgs = await inbox.get_messages(is_read=False)
        assert len(msgs) == 1
        assert msgs[0].message_id == "m2"

    @pytest.mark.asyncio
    async def test_mark_read(self, inbox):
        await inbox.add_message(
            InboxMessage(message_id="m1", channel="slack", sender="a", content="1")
        )
        count = await inbox.mark_read(["m1"])
        assert count == 1
        msg = await inbox.get_message("m1")
        assert msg.is_read is True

    @pytest.mark.asyncio
    async def test_mark_replied(self, inbox):
        await inbox.add_message(
            InboxMessage(message_id="m1", channel="slack", sender="a", content="1")
        )
        result = await inbox.mark_replied("m1")
        assert result is True
        msg = await inbox.get_message("m1")
        assert msg.is_replied is True

    @pytest.mark.asyncio
    async def test_threading(self, inbox):
        await inbox.add_message(
            InboxMessage(
                message_id="m1",
                channel="slack",
                sender="alice",
                content="Thread start",
                thread_id="t1",
            )
        )
        await inbox.add_message(
            InboxMessage(
                message_id="m2",
                channel="slack",
                sender="bob",
                content="Reply",
                thread_id="t1",
            )
        )
        threads = await inbox.get_threads()
        assert len(threads) == 1
        assert len(threads[0].messages) == 2
        assert "alice" in threads[0].participants
        assert "bob" in threads[0].participants

    @pytest.mark.asyncio
    async def test_unread_count(self, inbox):
        await inbox.add_message(
            InboxMessage(message_id="m1", channel="slack", sender="a", content="1")
        )
        await inbox.add_message(
            InboxMessage(message_id="m2", channel="slack", sender="b", content="2")
        )
        await inbox.mark_read(["m1"])
        count = await inbox.get_unread_count()
        assert count == 1

    @pytest.mark.asyncio
    async def test_clear_all(self, inbox):
        await inbox.add_message(
            InboxMessage(message_id="m1", channel="slack", sender="a", content="1")
        )
        await inbox.add_message(
            InboxMessage(message_id="m2", channel="teams", sender="b", content="2")
        )
        removed = await inbox.clear()
        assert removed == 2
        assert await inbox.get_size() == 0

    @pytest.mark.asyncio
    async def test_clear_by_channel(self, inbox):
        await inbox.add_message(
            InboxMessage(message_id="m1", channel="slack", sender="a", content="1")
        )
        await inbox.add_message(
            InboxMessage(message_id="m2", channel="teams", sender="b", content="2")
        )
        removed = await inbox.clear(channel="slack")
        assert removed == 1
        assert await inbox.get_size() == 1

    @pytest.mark.asyncio
    async def test_max_size_eviction(self):
        inbox = InboxAggregator(max_size=2)
        await inbox.add_message(InboxMessage(message_id="m1", channel="s", sender="a", content="1"))
        await inbox.add_message(InboxMessage(message_id="m2", channel="s", sender="b", content="2"))
        await inbox.add_message(InboxMessage(message_id="m3", channel="s", sender="c", content="3"))
        assert await inbox.get_size() == 2  # m1 evicted


# =============================================================================
# DeviceRegistry Tests
# =============================================================================


class TestDeviceNode:
    """Test DeviceNode dataclass."""

    def test_device_creation(self):
        device = DeviceNode(
            name="My Laptop", device_type="laptop", capabilities=["browser", "shell"]
        )
        assert device.name == "My Laptop"
        assert device.status == DeviceStatus.OFFLINE


class TestDeviceRegistry:
    """Test DeviceRegistry."""

    @pytest.fixture
    def registry(self):
        return DeviceRegistry()

    @pytest.mark.asyncio
    async def test_register_device(self, registry):
        device = DeviceNode(name="laptop", device_type="laptop")
        device_id = await registry.register(device)
        assert device_id.startswith("dev-")
        assert await registry.count() == 1

    @pytest.mark.asyncio
    async def test_register_with_custom_id(self, registry):
        device = DeviceNode(device_id="dev-custom", name="laptop", device_type="laptop")
        device_id = await registry.register(device)
        assert device_id == "dev-custom"

    @pytest.mark.asyncio
    async def test_get_device(self, registry):
        device = DeviceNode(device_id="dev-get", name="laptop", device_type="laptop")
        await registry.register(device)
        result = await registry.get("dev-get")
        assert result is not None
        assert result.name == "laptop"
        assert result.status == DeviceStatus.PAIRED

    @pytest.mark.asyncio
    async def test_unregister(self, registry):
        device = DeviceNode(device_id="dev-rm", name="laptop", device_type="laptop")
        await registry.register(device)
        assert await registry.unregister("dev-rm") is True
        assert await registry.count() == 0

    @pytest.mark.asyncio
    async def test_heartbeat(self, registry):
        device = DeviceNode(device_id="dev-hb", name="laptop", device_type="laptop")
        await registry.register(device)
        assert await registry.heartbeat("dev-hb") is True
        result = await registry.get("dev-hb")
        assert result.status == DeviceStatus.ONLINE

    @pytest.mark.asyncio
    async def test_block_device(self, registry):
        device = DeviceNode(device_id="dev-block", name="laptop", device_type="laptop")
        await registry.register(device)
        await registry.block("dev-block")
        result = await registry.get("dev-block")
        assert result.status == DeviceStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_list_devices(self, registry):
        await registry.register(DeviceNode(device_id="d1", name="laptop", device_type="laptop"))
        await registry.register(DeviceNode(device_id="d2", name="phone", device_type="phone"))
        devices = await registry.list_devices()
        assert len(devices) == 2

    @pytest.mark.asyncio
    async def test_list_devices_by_type(self, registry):
        await registry.register(DeviceNode(device_id="d1", name="laptop", device_type="laptop"))
        await registry.register(DeviceNode(device_id="d2", name="phone", device_type="phone"))
        devices = await registry.list_devices(device_type="laptop")
        assert len(devices) == 1

    @pytest.mark.asyncio
    async def test_has_capability(self, registry):
        device = DeviceNode(
            device_id="d1", name="laptop", device_type="laptop", capabilities=["browser", "shell"]
        )
        await registry.register(device)
        assert await registry.has_capability("d1", "browser") is True
        assert await registry.has_capability("d1", "camera") is False


# =============================================================================
# AgentRouter Tests
# =============================================================================


class TestRoutingRule:
    """Test RoutingRule dataclass."""

    def test_rule_creation(self):
        rule = RoutingRule(rule_id="r1", agent_id="claude", channel_pattern="slack")
        assert rule.rule_id == "r1"
        assert rule.enabled is True


class TestAgentRouter:
    """Test AgentRouter."""

    @pytest.fixture
    def router(self):
        return AgentRouter(default_agent="fallback")

    @pytest.mark.asyncio
    async def test_default_routing(self, router):
        msg = InboxMessage(message_id="m1", channel="slack", sender="alice", content="hi")
        agent = await router.route("slack", msg)
        assert agent == "fallback"

    @pytest.mark.asyncio
    async def test_channel_routing(self, router):
        await router.add_rule(
            RoutingRule(
                rule_id="r1",
                agent_id="claude",
                channel_pattern="slack",
            )
        )
        msg = InboxMessage(message_id="m1", channel="slack", sender="alice", content="hi")
        agent = await router.route("slack", msg)
        assert agent == "claude"

    @pytest.mark.asyncio
    async def test_sender_routing(self, router):
        await router.add_rule(
            RoutingRule(
                rule_id="r1",
                agent_id="gpt",
                sender_pattern="boss*",
            )
        )
        msg = InboxMessage(message_id="m1", channel="slack", sender="boss@co.com", content="hi")
        agent = await router.route("slack", msg)
        assert agent == "gpt"

    @pytest.mark.asyncio
    async def test_content_routing(self, router):
        await router.add_rule(
            RoutingRule(
                rule_id="r1",
                agent_id="code-agent",
                content_pattern="code review",
            )
        )
        msg = InboxMessage(
            message_id="m1", channel="slack", sender="a", content="Please do a code review"
        )
        agent = await router.route("slack", msg)
        assert agent == "code-agent"

    @pytest.mark.asyncio
    async def test_priority_ordering(self, router):
        await router.add_rule(
            RoutingRule(
                rule_id="r1",
                agent_id="low-priority",
                channel_pattern="*",
                priority=0,
            )
        )
        await router.add_rule(
            RoutingRule(
                rule_id="r2",
                agent_id="high-priority",
                channel_pattern="*",
                priority=10,
            )
        )
        msg = InboxMessage(message_id="m1", channel="slack", sender="a", content="hi")
        agent = await router.route("slack", msg)
        assert agent == "high-priority"

    @pytest.mark.asyncio
    async def test_remove_rule(self, router):
        await router.add_rule(RoutingRule(rule_id="r1", agent_id="claude", channel_pattern="*"))
        assert await router.remove_rule("r1") is True
        assert await router.count_rules() == 0

    @pytest.mark.asyncio
    async def test_list_rules(self, router):
        await router.add_rule(RoutingRule(rule_id="r1", agent_id="a", priority=5))
        await router.add_rule(RoutingRule(rule_id="r2", agent_id="b", priority=10))
        rules = await router.list_rules()
        assert len(rules) == 2
        assert rules[0].priority == 10  # Higher first

    @pytest.mark.asyncio
    async def test_disabled_rule_skipped(self, router):
        await router.add_rule(
            RoutingRule(
                rule_id="r1",
                agent_id="claude",
                channel_pattern="*",
                enabled=False,
            )
        )
        msg = InboxMessage(message_id="m1", channel="slack", sender="a", content="hi")
        agent = await router.route("slack", msg)
        assert agent == "fallback"  # Disabled rule not matched


# =============================================================================
# LocalGateway Tests
# =============================================================================


class TestGatewayConfig:
    """Test GatewayConfig."""

    def test_defaults(self):
        config = GatewayConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 8090
        assert config.enable_auth is True


class TestLocalGateway:
    """Test LocalGateway."""

    @pytest.fixture
    def gw(self):
        return LocalGateway(config=GatewayConfig(enable_auth=False))

    @pytest.mark.asyncio
    async def test_start_stop(self, gw):
        assert gw.is_running is False
        await gw.start()
        assert gw.is_running is True
        await gw.stop()
        assert gw.is_running is False

    @pytest.mark.asyncio
    async def test_route_message(self, gw):
        await gw.start()
        msg = InboxMessage(message_id="m1", channel="slack", sender="alice", content="hello")
        response = await gw.route_message("slack", msg)
        assert response.success is True
        assert response.agent_id == "default"
        await gw.stop()

    @pytest.mark.asyncio
    async def test_route_message_not_running(self, gw):
        msg = InboxMessage(message_id="m1", channel="slack", sender="a", content="hi")
        response = await gw.route_message("slack", msg)
        assert response.success is False
        assert "not running" in response.error

    @pytest.mark.asyncio
    async def test_route_message_auth_failure(self):
        gw = LocalGateway(config=GatewayConfig(enable_auth=True, api_key="secret"))
        await gw.start()
        msg = InboxMessage(
            message_id="m1",
            channel="slack",
            sender="a",
            content="hi",
            metadata={"api_key": "wrong"},
        )
        response = await gw.route_message("slack", msg)
        assert response.success is False
        assert "Authentication" in response.error
        await gw.stop()

    @pytest.mark.asyncio
    async def test_register_device(self, gw):
        device = DeviceNode(name="laptop", device_type="laptop")
        device_id = await gw.register_device(device)
        assert device_id.startswith("dev-")

    @pytest.mark.asyncio
    async def test_get_inbox(self, gw):
        await gw.start()
        msg = InboxMessage(message_id="m1", channel="slack", sender="a", content="hi")
        await gw.route_message("slack", msg)
        inbox = await gw.get_inbox()
        assert len(inbox) == 1
        await gw.stop()

    @pytest.mark.asyncio
    async def test_get_stats(self, gw):
        await gw.start()
        msg = InboxMessage(message_id="m1", channel="slack", sender="a", content="hi")
        await gw.route_message("slack", msg)
        stats = await gw.get_stats()
        assert stats["running"] is True
        assert stats["messages_routed"] == 1
        await gw.stop()
