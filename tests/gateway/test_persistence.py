"""
Tests for Gateway Persistence Layer.

Tests the storage backends for gateway state:
- InMemoryGatewayStore (testing)
- FileGatewayStore (local-first)
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from aragora.gateway.inbox import InboxMessage, MessagePriority
from aragora.gateway.device_registry import DeviceNode, DeviceStatus
from aragora.gateway.router import RoutingRule
from aragora.gateway.persistence import (
    InMemoryGatewayStore,
    FileGatewayStore,
    get_gateway_store,
    _message_to_dict,
    _dict_to_message,
    _device_to_dict,
    _dict_to_device,
    _rule_to_dict,
    _dict_to_rule,
)


# =============================================================================
# Serialization Helper Tests
# =============================================================================


class TestMessageSerialization:
    """Test InboxMessage serialization."""

    def test_message_round_trip(self):
        msg = InboxMessage(
            message_id="m1",
            channel="slack",
            sender="alice",
            content="Hello world",
            is_read=True,
            is_replied=False,
            priority=MessagePriority.HIGH,
            thread_id="t1",
            metadata={"key": "value"},
        )
        d = _message_to_dict(msg)
        restored = _dict_to_message(d)

        assert restored.message_id == msg.message_id
        assert restored.channel == msg.channel
        assert restored.sender == msg.sender
        assert restored.content == msg.content
        assert restored.is_read == msg.is_read
        assert restored.is_replied == msg.is_replied
        assert restored.priority == msg.priority
        assert restored.thread_id == msg.thread_id
        assert restored.metadata == msg.metadata

    def test_message_defaults(self):
        d = {
            "message_id": "m1",
            "channel": "slack",
            "sender": "alice",
            "content": "test",
        }
        msg = _dict_to_message(d)
        assert msg.is_read is False
        assert msg.is_replied is False
        assert msg.priority == MessagePriority.NORMAL


class TestDeviceSerialization:
    """Test DeviceNode serialization."""

    def test_device_round_trip(self):
        device = DeviceNode(
            device_id="dev-1",
            name="My Laptop",
            device_type="laptop",
            status=DeviceStatus.ONLINE,
            capabilities=["browser", "shell"],
            metadata={"os": "macOS"},
        )
        d = _device_to_dict(device)
        restored = _dict_to_device(d)

        assert restored.device_id == device.device_id
        assert restored.name == device.name
        assert restored.device_type == device.device_type
        assert restored.status == device.status
        assert restored.capabilities == device.capabilities
        assert restored.metadata == device.metadata


class TestRuleSerialization:
    """Test RoutingRule serialization."""

    def test_rule_round_trip(self):
        rule = RoutingRule(
            rule_id="r1",
            agent_id="claude",
            channel_pattern="slack",
            sender_pattern="boss*",
            content_pattern="urgent",
            priority=10,
            enabled=False,
        )
        d = _rule_to_dict(rule)
        restored = _dict_to_rule(d)

        assert restored.rule_id == rule.rule_id
        assert restored.agent_id == rule.agent_id
        assert restored.channel_pattern == rule.channel_pattern
        assert restored.sender_pattern == rule.sender_pattern
        assert restored.content_pattern == rule.content_pattern
        assert restored.priority == rule.priority
        assert restored.enabled == rule.enabled


# =============================================================================
# InMemoryGatewayStore Tests
# =============================================================================


class TestInMemoryGatewayStore:
    """Test InMemoryGatewayStore."""

    @pytest.fixture
    def store(self):
        return InMemoryGatewayStore()

    # Message tests

    @pytest.mark.asyncio
    async def test_save_and_load_message(self, store):
        msg = InboxMessage(message_id="m1", channel="slack", sender="a", content="hi")
        await store.save_message(msg)
        messages = await store.load_messages()
        assert len(messages) == 1
        assert messages[0].message_id == "m1"

    @pytest.mark.asyncio
    async def test_load_messages_ordered_by_timestamp(self, store):
        msg1 = InboxMessage(message_id="m1", channel="s", sender="a", content="1", timestamp=100)
        msg2 = InboxMessage(message_id="m2", channel="s", sender="a", content="2", timestamp=200)
        msg3 = InboxMessage(message_id="m3", channel="s", sender="a", content="3", timestamp=150)

        await store.save_message(msg1)
        await store.save_message(msg2)
        await store.save_message(msg3)

        messages = await store.load_messages()
        assert [m.message_id for m in messages] == ["m2", "m3", "m1"]

    @pytest.mark.asyncio
    async def test_load_messages_limit(self, store):
        for i in range(10):
            await store.save_message(
                InboxMessage(message_id=f"m{i}", channel="s", sender="a", content=str(i))
            )
        messages = await store.load_messages(limit=3)
        assert len(messages) == 3

    @pytest.mark.asyncio
    async def test_delete_message(self, store):
        msg = InboxMessage(message_id="m1", channel="s", sender="a", content="hi")
        await store.save_message(msg)
        assert await store.delete_message("m1") is True
        assert await store.delete_message("m1") is False
        assert len(await store.load_messages()) == 0

    @pytest.mark.asyncio
    async def test_clear_messages_all(self, store):
        for i in range(5):
            await store.save_message(
                InboxMessage(message_id=f"m{i}", channel="s", sender="a", content=str(i))
            )
        count = await store.clear_messages()
        assert count == 5
        assert len(await store.load_messages()) == 0

    @pytest.mark.asyncio
    async def test_clear_messages_older_than(self, store):
        import time

        now = time.time()
        old_msg = InboxMessage(
            message_id="old", channel="s", sender="a", content="old", timestamp=now - 100
        )
        new_msg = InboxMessage(
            message_id="new", channel="s", sender="a", content="new", timestamp=now
        )

        await store.save_message(old_msg)
        await store.save_message(new_msg)

        count = await store.clear_messages(older_than_seconds=50)
        assert count == 1
        messages = await store.load_messages()
        assert len(messages) == 1
        assert messages[0].message_id == "new"

    # Device tests

    @pytest.mark.asyncio
    async def test_save_and_load_device(self, store):
        device = DeviceNode(device_id="d1", name="Laptop", device_type="laptop")
        await store.save_device(device)
        devices = await store.load_devices()
        assert len(devices) == 1
        assert devices[0].name == "Laptop"

    @pytest.mark.asyncio
    async def test_delete_device(self, store):
        device = DeviceNode(device_id="d1", name="Laptop", device_type="laptop")
        await store.save_device(device)
        assert await store.delete_device("d1") is True
        assert await store.delete_device("d1") is False
        assert len(await store.load_devices()) == 0

    # Rule tests

    @pytest.mark.asyncio
    async def test_save_and_load_rule(self, store):
        rule = RoutingRule(rule_id="r1", agent_id="claude", priority=5)
        await store.save_rule(rule)
        rules = await store.load_rules()
        assert len(rules) == 1
        assert rules[0].agent_id == "claude"

    @pytest.mark.asyncio
    async def test_load_rules_ordered_by_priority(self, store):
        await store.save_rule(RoutingRule(rule_id="r1", agent_id="low", priority=1))
        await store.save_rule(RoutingRule(rule_id="r2", agent_id="high", priority=10))
        await store.save_rule(RoutingRule(rule_id="r3", agent_id="mid", priority=5))

        rules = await store.load_rules()
        assert [r.agent_id for r in rules] == ["high", "mid", "low"]

    @pytest.mark.asyncio
    async def test_delete_rule(self, store):
        rule = RoutingRule(rule_id="r1", agent_id="claude")
        await store.save_rule(rule)
        assert await store.delete_rule("r1") is True
        assert await store.delete_rule("r1") is False

    @pytest.mark.asyncio
    async def test_close(self, store):
        # Should be a no-op
        await store.close()


# =============================================================================
# FileGatewayStore Tests
# =============================================================================


class TestFileGatewayStore:
    """Test FileGatewayStore."""

    @pytest.fixture
    def temp_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "gateway.json"

    @pytest.fixture
    def store(self, temp_path):
        return FileGatewayStore(path=temp_path, auto_save=True, auto_save_interval=0)

    # Message tests

    @pytest.mark.asyncio
    async def test_save_and_load_message(self, store):
        msg = InboxMessage(message_id="m1", channel="slack", sender="a", content="hi")
        await store.save_message(msg)
        messages = await store.load_messages()
        assert len(messages) == 1
        assert messages[0].message_id == "m1"

    @pytest.mark.asyncio
    async def test_persistence_across_instances(self, temp_path):
        # Save with first instance
        store1 = FileGatewayStore(path=temp_path, auto_save=True, auto_save_interval=0)
        msg = InboxMessage(message_id="m1", channel="slack", sender="a", content="hi")
        await store1.save_message(msg)
        await store1.close()

        # Load with second instance
        store2 = FileGatewayStore(path=temp_path)
        messages = await store2.load_messages()
        assert len(messages) == 1
        assert messages[0].message_id == "m1"
        await store2.close()

    @pytest.mark.asyncio
    async def test_delete_message(self, store):
        msg = InboxMessage(message_id="m1", channel="s", sender="a", content="hi")
        await store.save_message(msg)
        assert await store.delete_message("m1") is True
        assert len(await store.load_messages()) == 0

    @pytest.mark.asyncio
    async def test_clear_messages(self, store):
        for i in range(3):
            await store.save_message(
                InboxMessage(message_id=f"m{i}", channel="s", sender="a", content=str(i))
            )
        count = await store.clear_messages()
        assert count == 3

    # Device tests

    @pytest.mark.asyncio
    async def test_save_and_load_device(self, store):
        device = DeviceNode(device_id="d1", name="Laptop", device_type="laptop")
        await store.save_device(device)
        devices = await store.load_devices()
        assert len(devices) == 1

    @pytest.mark.asyncio
    async def test_device_persistence(self, temp_path):
        store1 = FileGatewayStore(path=temp_path, auto_save=True, auto_save_interval=0)
        device = DeviceNode(device_id="d1", name="Laptop", device_type="laptop")
        await store1.save_device(device)
        await store1.close()

        store2 = FileGatewayStore(path=temp_path)
        devices = await store2.load_devices()
        assert len(devices) == 1
        assert devices[0].name == "Laptop"
        await store2.close()

    # Rule tests

    @pytest.mark.asyncio
    async def test_save_and_load_rule(self, store):
        rule = RoutingRule(rule_id="r1", agent_id="claude")
        await store.save_rule(rule)
        rules = await store.load_rules()
        assert len(rules) == 1

    @pytest.mark.asyncio
    async def test_rule_persistence(self, temp_path):
        store1 = FileGatewayStore(path=temp_path, auto_save=True, auto_save_interval=0)
        rule = RoutingRule(rule_id="r1", agent_id="claude", priority=5)
        await store1.save_rule(rule)
        await store1.close()

        store2 = FileGatewayStore(path=temp_path)
        rules = await store2.load_rules()
        assert len(rules) == 1
        assert rules[0].priority == 5
        await store2.close()

    # Edge cases

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self, temp_path):
        store = FileGatewayStore(path=temp_path)
        messages = await store.load_messages()
        assert messages == []
        await store.close()

    @pytest.mark.asyncio
    async def test_corrupt_file_handling(self, temp_path):
        # Create a corrupt JSON file
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "w") as f:
            f.write("not valid json {{{")

        store = FileGatewayStore(path=temp_path)
        # Should not raise, just log warning and return empty
        messages = await store.load_messages()
        assert messages == []
        await store.close()


# =============================================================================
# Factory Tests
# =============================================================================


class TestGetGatewayStore:
    """Test get_gateway_store factory."""

    def test_memory_backend(self):
        store = get_gateway_store("memory")
        assert isinstance(store, InMemoryGatewayStore)

    def test_file_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = get_gateway_store("file", path=Path(tmpdir) / "test.json")
            assert isinstance(store, FileGatewayStore)

    def test_auto_backend_no_redis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = get_gateway_store("auto", path=Path(tmpdir) / "test.json")
            # Without REDIS_URL, should fall back to file
            assert isinstance(store, FileGatewayStore)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_gateway_store("unknown")
