"""Tests for capability-aware routing."""

from __future__ import annotations

import pytest

from aragora.gateway.device_registry import DeviceNode, DeviceRegistry, DeviceStatus
from aragora.gateway.inbox import InboxMessage
from aragora.gateway.router import RoutingRule
from aragora.gateway.capability_router import CapabilityRouter, CapabilityRule, RoutingResult


@pytest.fixture
def registry():
    return DeviceRegistry()


@pytest.fixture
def router(registry):
    return CapabilityRouter(default_agent="default-agent", device_registry=registry)


@pytest.fixture
def message():
    return InboxMessage(
        message_id="msg-1",
        sender="user@example.com",
        content="Hello, I need help with video",
        channel="slack",
    )


class TestCapabilityRule:
    def test_default_values(self):
        rule = CapabilityRule(rule_id="r1", agent_id="agent1")
        assert rule.channel_pattern == "*"
        assert rule.sender_pattern == "*"
        assert rule.required_capabilities == []
        assert rule.fallback_capabilities == []
        assert rule.fallback_agent_id is None

    def test_with_capabilities(self):
        rule = CapabilityRule(
            rule_id="r1",
            agent_id="video-agent",
            required_capabilities=["camera", "mic"],
            fallback_capabilities=["mic"],
            fallback_agent_id="audio-agent",
        )
        assert rule.required_capabilities == ["camera", "mic"]
        assert rule.fallback_agent_id == "audio-agent"


class TestRoutingResult:
    def test_default_values(self):
        result = RoutingResult(agent_id="agent1")
        assert result.rule_id is None
        assert result.used_fallback is False
        assert result.missing_capabilities == []


class TestCapabilityRouterBasic:
    @pytest.mark.asyncio
    async def test_add_and_get_rule(self, router):
        rule = CapabilityRule(rule_id="r1", agent_id="agent1")
        await router.add_capability_rule(rule)

        retrieved = await router.get_rule("r1")
        assert retrieved is not None
        assert retrieved.agent_id == "agent1"

    @pytest.mark.asyncio
    async def test_remove_rule(self, router):
        rule = CapabilityRule(rule_id="r1", agent_id="agent1")
        await router.add_capability_rule(rule)

        result = await router.remove_capability_rule("r1")
        assert result is True

        retrieved = await router.get_rule("r1")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_route_default(self, router, message):
        agent = await router.route_with_capabilities("slack", message)
        assert agent == "default-agent"

    @pytest.mark.asyncio
    async def test_route_matching_rule(self, router, message):
        await router.add_capability_rule(
            CapabilityRule(
                rule_id="slack-rule",
                agent_id="slack-agent",
                channel_pattern="slack",
            )
        )

        agent = await router.route_with_capabilities("slack", message)
        assert agent == "slack-agent"

    @pytest.mark.asyncio
    async def test_route_with_details(self, router, message):
        await router.add_capability_rule(
            CapabilityRule(
                rule_id="slack-rule",
                agent_id="slack-agent",
                channel_pattern="slack",
            )
        )

        result = await router.route_with_details("slack", message)
        assert result.agent_id == "slack-agent"
        assert result.rule_id == "slack-rule"
        assert result.used_fallback is False


class TestCapabilityAwareRouting:
    @pytest.mark.asyncio
    async def test_route_with_capability_match(self, router, registry, message):
        device = DeviceNode(
            device_id="dev-1",
            name="Laptop",
            device_type="laptop",
            capabilities=["camera", "mic", "browser"],
            status=DeviceStatus.ONLINE,
        )
        await registry.register(device)

        await router.add_capability_rule(
            CapabilityRule(
                rule_id="video-rule",
                agent_id="video-agent",
                channel_pattern="*",
                required_capabilities=["camera", "mic"],
            )
        )

        result = await router.route_with_details("slack", message, device_id="dev-1")
        assert result.agent_id == "video-agent"
        assert result.used_fallback is False

    @pytest.mark.asyncio
    async def test_route_with_missing_capability_skips_rule(self, router, registry, message):
        device = DeviceNode(
            device_id="dev-1",
            name="Phone",
            device_type="phone",
            capabilities=["mic"],
            status=DeviceStatus.ONLINE,
        )
        await registry.register(device)

        await router.add_capability_rule(
            CapabilityRule(
                rule_id="video-rule",
                agent_id="video-agent",
                channel_pattern="*",
                required_capabilities=["camera", "mic"],
                priority=10,
            )
        )
        await router.add_capability_rule(
            CapabilityRule(
                rule_id="text-rule",
                agent_id="text-agent",
                channel_pattern="*",
                priority=5,
            )
        )

        result = await router.route_with_details("slack", message, device_id="dev-1")
        assert result.agent_id == "text-agent"
        assert result.rule_id == "text-rule"

    @pytest.mark.asyncio
    async def test_route_with_fallback_agent(self, router, registry, message):
        device = DeviceNode(
            device_id="dev-1",
            name="Phone",
            device_type="phone",
            capabilities=["mic"],
            status=DeviceStatus.ONLINE,
        )
        await registry.register(device)

        await router.add_capability_rule(
            CapabilityRule(
                rule_id="video-rule",
                agent_id="video-agent",
                channel_pattern="*",
                required_capabilities=["camera", "mic"],
                fallback_capabilities=["mic"],
                fallback_agent_id="audio-agent",
            )
        )

        result = await router.route_with_details("slack", message, device_id="dev-1")
        assert result.agent_id == "audio-agent"
        assert result.used_fallback is True
        assert "camera" in result.missing_capabilities

    @pytest.mark.asyncio
    async def test_route_without_device_id(self, router, message):
        await router.add_capability_rule(
            CapabilityRule(
                rule_id="video-rule",
                agent_id="video-agent",
                channel_pattern="*",
                required_capabilities=["camera"],
            )
        )

        result = await router.route_with_details("slack", message, device_id=None)
        assert result.agent_id == "video-agent"


class TestFindCapableDevice:
    @pytest.mark.asyncio
    async def test_find_device_with_capabilities(self, router, registry):
        device1 = DeviceNode(
            device_id="dev-1",
            name="Laptop",
            device_type="laptop",
            capabilities=["browser", "shell"],
            status=DeviceStatus.ONLINE,
        )
        device2 = DeviceNode(
            device_id="dev-2",
            name="Phone",
            device_type="phone",
            capabilities=["camera", "mic"],
            status=DeviceStatus.ONLINE,
        )
        await registry.register(device1)
        await registry.register(device2)

        found = await router.find_capable_device(["camera", "mic"])
        assert found == "dev-2"

    @pytest.mark.asyncio
    async def test_find_device_prefers_online(self, router, registry):
        device1 = DeviceNode(
            device_id="dev-1",
            name="Laptop",
            device_type="laptop",
            capabilities=["camera"],
        )
        device2 = DeviceNode(
            device_id="dev-2",
            name="Phone",
            device_type="phone",
            capabilities=["camera"],
        )
        await registry.register(device1)
        await registry.register(device2)
        # Use heartbeat to set device2 to ONLINE (register sets PAIRED)
        await registry.heartbeat("dev-2")

        found = await router.find_capable_device(["camera"], prefer_online=True)
        assert found == "dev-2"

    @pytest.mark.asyncio
    async def test_find_device_no_match(self, router, registry):
        device = DeviceNode(
            device_id="dev-1",
            name="Laptop",
            device_type="laptop",
            capabilities=["browser"],
            status=DeviceStatus.ONLINE,
        )
        await registry.register(device)

        found = await router.find_capable_device(["camera", "mic"])
        assert found is None

    @pytest.mark.asyncio
    async def test_find_device_no_registry(self, message):
        router = CapabilityRouter(default_agent="default")
        found = await router.find_capable_device(["camera"])
        assert found is None
