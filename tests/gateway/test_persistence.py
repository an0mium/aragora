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
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from aragora.gateway.persistence import (
    InMemoryGatewayStore,
    FileGatewayStore,
    RedisGatewayStore,
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

    def test_message_priority_from_int(self):
        """Test that priority can be deserialized from integer."""
        d = {
            "message_id": "m1",
            "channel": "slack",
            "sender": "alice",
            "content": "test",
            "priority": 1,  # HIGH
        }
        msg = _dict_to_message(d)
        assert msg.priority == MessagePriority.HIGH

    def test_message_priority_from_string(self):
        """Test that priority can be deserialized from string."""
        d = {
            "message_id": "m1",
            "channel": "slack",
            "sender": "alice",
            "content": "test",
            "priority": "HIGH",
        }
        msg = _dict_to_message(d)
        assert msg.priority == MessagePriority.HIGH

        # Test lowercase
        d["priority"] = "urgent"
        msg = _dict_to_message(d)
        assert msg.priority == MessagePriority.URGENT

    def test_message_with_unicode_content(self):
        """Test serialization of message with unicode content."""
        msg = InboxMessage(
            message_id="m1",
            channel="slack",
            sender="alice",
            content="Hello World with unicode chars: \u4e2d\u6587 \u0410\u0411\u0412 \ud83d\ude00",
            metadata={"key": "value with unicode: \u00e9\u00e8\u00ea"},
        )
        d = _message_to_dict(msg)
        restored = _dict_to_message(d)

        assert restored.content == msg.content
        assert restored.metadata == msg.metadata

    def test_message_with_empty_metadata(self):
        """Test serialization of message with empty metadata."""
        msg = InboxMessage(
            message_id="m1",
            channel="slack",
            sender="alice",
            content="test",
            metadata={},
        )
        d = _message_to_dict(msg)
        restored = _dict_to_message(d)

        assert restored.metadata == {}

    def test_message_with_none_thread_id(self):
        """Test serialization of message with None thread_id."""
        msg = InboxMessage(
            message_id="m1",
            channel="slack",
            sender="alice",
            content="test",
            thread_id=None,
        )
        d = _message_to_dict(msg)
        restored = _dict_to_message(d)

        assert restored.thread_id is None

    def test_message_timestamp_default(self):
        """Test that timestamp uses default when not provided."""
        import time

        before = time.time()
        d = {
            "message_id": "m1",
            "channel": "slack",
            "sender": "alice",
            "content": "test",
        }
        msg = _dict_to_message(d)
        after = time.time()

        assert before <= msg.timestamp <= after

    def test_message_all_priorities(self):
        """Test serialization with all priority levels."""
        priorities = [
            (MessagePriority.LOW, 3),
            (MessagePriority.NORMAL, 2),
            (MessagePriority.HIGH, 1),
            (MessagePriority.URGENT, 0),
        ]

        for priority_enum, priority_int in priorities:
            msg = InboxMessage(
                message_id="m1",
                channel="slack",
                sender="alice",
                content="test",
                priority=priority_enum,
            )
            d = _message_to_dict(msg)
            assert d["priority"] == priority_int

            restored = _dict_to_message(d)
            assert restored.priority == priority_enum


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

    def test_device_status_from_string(self):
        """Test that device status can be deserialized from string."""
        d = {
            "device_id": "dev-1",
            "name": "Laptop",
            "device_type": "laptop",
            "status": "online",
        }
        device = _dict_to_device(d)
        assert device.status == DeviceStatus.ONLINE

        d["status"] = "blocked"
        device = _dict_to_device(d)
        assert device.status == DeviceStatus.BLOCKED

    def test_device_defaults(self):
        """Test device defaults for type and capabilities."""
        d = {
            "name": "My Device",
        }
        device = _dict_to_device(d)

        assert device.device_type == "unknown"
        assert device.capabilities == []
        assert device.metadata == {}

    def test_device_with_empty_capabilities(self):
        """Test device with empty capabilities list."""
        device = DeviceNode(
            device_id="dev-1",
            name="Laptop",
            device_type="laptop",
            capabilities=[],
        )
        d = _device_to_dict(device)
        restored = _dict_to_device(d)

        assert restored.capabilities == []

    def test_device_with_all_statuses(self):
        """Test serialization with all device statuses."""
        statuses = [
            DeviceStatus.ONLINE,
            DeviceStatus.OFFLINE,
            DeviceStatus.PAIRED,
            DeviceStatus.BLOCKED,
        ]

        for status in statuses:
            device = DeviceNode(
                device_id="dev-1",
                name="Laptop",
                device_type="laptop",
                status=status,
            )
            d = _device_to_dict(device)
            assert d["status"] == status.value

            restored = _dict_to_device(d)
            assert restored.status == status

    def test_device_auto_generated_id(self):
        """Test device with missing device_id uses None (auto-gen happens at registration)."""
        d = {
            "name": "My Device",
            "device_type": "laptop",
        }
        device = _dict_to_device(d)
        # device_id should be None when not provided (auto-generated on registration)
        assert device.device_id is None

    def test_device_with_none_last_seen(self):
        """Test device with None last_seen timestamp."""
        device = DeviceNode(
            device_id="dev-1",
            name="Laptop",
            device_type="laptop",
            last_seen=None,
        )
        d = _device_to_dict(device)
        restored = _dict_to_device(d)

        assert restored.last_seen is None


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

    def test_rule_defaults(self):
        """Test rule defaults for patterns, enabled, and priority."""
        d = {
            "rule_id": "r1",
            "agent_id": "claude",
        }
        rule = _dict_to_rule(d)

        assert rule.channel_pattern is None  # _dict_to_rule uses .get() which returns None
        assert rule.sender_pattern is None
        assert rule.content_pattern is None
        assert rule.enabled is True
        assert rule.priority == 0

    def test_rule_with_none_patterns(self):
        """Test rule with None patterns."""
        rule = RoutingRule(
            rule_id="r1",
            agent_id="claude",
            channel_pattern=None,
            sender_pattern=None,
            content_pattern=None,
        )
        d = _rule_to_dict(rule)
        restored = _dict_to_rule(d)

        assert restored.channel_pattern is None
        assert restored.sender_pattern is None
        assert restored.content_pattern is None

    def test_rule_enabled_default_true(self):
        """Test that rule.enabled defaults to True."""
        d = {
            "rule_id": "r1",
            "agent_id": "claude",
            # enabled not specified
        }
        rule = _dict_to_rule(d)
        assert rule.enabled is True

    def test_rule_priority_default_zero(self):
        """Test that rule.priority defaults to 0."""
        d = {
            "rule_id": "r1",
            "agent_id": "claude",
            # priority not specified
        }
        rule = _dict_to_rule(d)
        assert rule.priority == 0

    def test_rule_with_all_patterns(self):
        """Test rule with all patterns specified."""
        rule = RoutingRule(
            rule_id="r1",
            agent_id="claude",
            channel_pattern="slack*",
            sender_pattern="boss@*",
            content_pattern="urgent",
            priority=99,
            enabled=True,
        )
        d = _rule_to_dict(rule)
        restored = _dict_to_rule(d)

        assert restored.channel_pattern == "slack*"
        assert restored.sender_pattern == "boss@*"
        assert restored.content_pattern == "urgent"
        assert restored.priority == 99
        assert restored.enabled is True


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

    # Session tests

    @pytest.mark.asyncio
    async def test_save_and_load_session(self, store):
        session = {
            "session_id": "s1",
            "user_id": "u1",
            "device_id": "d1",
            "status": "active",
            "created_at": 100.0,
            "last_seen": 150.0,
            "metadata": {"role": "assistant"},
        }
        await store.save_session(session)
        sessions = await store.load_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "s1"

    @pytest.mark.asyncio
    async def test_load_sessions_ordered_by_last_seen(self, store):
        """Sessions should be ordered by last_seen descending (most recent first)."""
        session1 = {"session_id": "s1", "user_id": "u1", "last_seen": 100.0}
        session2 = {"session_id": "s2", "user_id": "u2", "last_seen": 300.0}
        session3 = {"session_id": "s3", "user_id": "u3", "last_seen": 200.0}

        await store.save_session(session1)
        await store.save_session(session2)
        await store.save_session(session3)

        sessions = await store.load_sessions()
        assert [s["session_id"] for s in sessions] == ["s2", "s3", "s1"]

    @pytest.mark.asyncio
    async def test_load_sessions_limit(self, store):
        """Load sessions should respect the limit parameter."""
        for i in range(10):
            await store.save_session(
                {"session_id": f"s{i}", "user_id": f"u{i}", "last_seen": float(i)}
            )
        sessions = await store.load_sessions(limit=3)
        assert len(sessions) == 3
        # Should get the 3 most recent (highest last_seen)
        assert sessions[0]["session_id"] == "s9"

    @pytest.mark.asyncio
    async def test_delete_session(self, store):
        """Test deleting an existing session."""
        session = {"session_id": "s1", "user_id": "u1", "last_seen": 100.0}
        await store.save_session(session)

        result = await store.delete_session("s1")
        assert result is True
        sessions = await store.load_sessions()
        assert len(sessions) == 0

    @pytest.mark.asyncio
    async def test_delete_session_nonexistent(self, store):
        """Deleting a nonexistent session should return False."""
        result = await store.delete_session("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear_sessions_all(self, store):
        """Clear all sessions when no threshold is provided."""
        for i in range(5):
            await store.save_session(
                {"session_id": f"s{i}", "user_id": f"u{i}", "last_seen": float(i)}
            )
        count = await store.clear_sessions()
        assert count == 5
        sessions = await store.load_sessions()
        assert len(sessions) == 0

    @pytest.mark.asyncio
    async def test_clear_sessions_older_than(self, store):
        """Clear sessions older than a threshold based on last_seen."""
        import time

        now = time.time()
        old_session = {"session_id": "old", "user_id": "u1", "last_seen": now - 100}
        new_session = {"session_id": "new", "user_id": "u2", "last_seen": now}

        await store.save_session(old_session)
        await store.save_session(new_session)

        count = await store.clear_sessions(older_than_seconds=50)
        assert count == 1
        sessions = await store.load_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "new"

    @pytest.mark.asyncio
    async def test_save_session_no_session_id(self, store):
        """Saving a session without session_id should be a no-op."""
        session = {"user_id": "u1", "last_seen": 100.0}
        await store.save_session(session)
        sessions = await store.load_sessions()
        assert len(sessions) == 0

    @pytest.mark.asyncio
    async def test_save_session_updates_existing(self, store):
        """Saving a session with the same ID should update it."""
        session1 = {"session_id": "s1", "user_id": "u1", "status": "active"}
        await store.save_session(session1)

        session2 = {"session_id": "s1", "user_id": "u1", "status": "inactive"}
        await store.save_session(session2)

        sessions = await store.load_sessions()
        assert len(sessions) == 1
        assert sessions[0]["status"] == "inactive"

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

    # Session tests

    @pytest.mark.asyncio
    async def test_save_and_load_session(self, store):
        session = {
            "session_id": "s1",
            "user_id": "u1",
            "device_id": "d1",
            "status": "active",
            "created_at": 100.0,
            "last_seen": 150.0,
            "metadata": {"role": "assistant"},
        }
        await store.save_session(session)
        sessions = await store.load_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "s1"

    @pytest.mark.asyncio
    async def test_session_persistence(self, temp_path):
        store1 = FileGatewayStore(path=temp_path, auto_save=True, auto_save_interval=0)
        session = {
            "session_id": "s2",
            "user_id": "u2",
            "device_id": "d2",
            "status": "active",
            "created_at": 200.0,
            "last_seen": 250.0,
        }
        await store1.save_session(session)
        await store1.close()

        store2 = FileGatewayStore(path=temp_path)
        sessions = await store2.load_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "s2"
        await store2.close()

    @pytest.mark.asyncio
    async def test_load_sessions_ordered_by_last_seen(self, store):
        """Sessions should be ordered by last_seen descending (most recent first)."""
        session1 = {"session_id": "s1", "user_id": "u1", "last_seen": 100.0}
        session2 = {"session_id": "s2", "user_id": "u2", "last_seen": 300.0}
        session3 = {"session_id": "s3", "user_id": "u3", "last_seen": 200.0}

        await store.save_session(session1)
        await store.save_session(session2)
        await store.save_session(session3)

        sessions = await store.load_sessions()
        assert [s["session_id"] for s in sessions] == ["s2", "s3", "s1"]

    @pytest.mark.asyncio
    async def test_load_sessions_limit(self, store):
        """Load sessions should respect the limit parameter."""
        for i in range(10):
            await store.save_session(
                {"session_id": f"s{i}", "user_id": f"u{i}", "last_seen": float(i)}
            )
        sessions = await store.load_sessions(limit=3)
        assert len(sessions) == 3
        # Should get the 3 most recent (highest last_seen)
        assert sessions[0]["session_id"] == "s9"

    @pytest.mark.asyncio
    async def test_delete_session(self, store):
        """Test deleting an existing session."""
        session = {"session_id": "s1", "user_id": "u1", "last_seen": 100.0}
        await store.save_session(session)

        result = await store.delete_session("s1")
        assert result is True
        sessions = await store.load_sessions()
        assert len(sessions) == 0

    @pytest.mark.asyncio
    async def test_delete_session_nonexistent(self, store):
        """Deleting a nonexistent session should return False."""
        result = await store.delete_session("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear_sessions_all(self, store):
        """Clear all sessions when no threshold is provided."""
        for i in range(5):
            await store.save_session(
                {"session_id": f"s{i}", "user_id": f"u{i}", "last_seen": float(i)}
            )
        count = await store.clear_sessions()
        assert count == 5
        sessions = await store.load_sessions()
        assert len(sessions) == 0

    @pytest.mark.asyncio
    async def test_clear_sessions_older_than(self, store):
        """Clear sessions older than a threshold based on last_seen."""
        import time

        now = time.time()
        old_session = {"session_id": "old", "user_id": "u1", "last_seen": now - 100}
        new_session = {"session_id": "new", "user_id": "u2", "last_seen": now}

        await store.save_session(old_session)
        await store.save_session(new_session)

        count = await store.clear_sessions(older_than_seconds=50)
        assert count == 1
        sessions = await store.load_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "new"

    @pytest.mark.asyncio
    async def test_clear_sessions_partial_match(self, store):
        """Clear sessions should only remove sessions matching the threshold."""
        import time

        now = time.time()
        # Create sessions at different times
        sessions_data = [
            {"session_id": "s1", "user_id": "u1", "last_seen": now - 200},  # Old
            {"session_id": "s2", "user_id": "u2", "last_seen": now - 150},  # Old
            {"session_id": "s3", "user_id": "u3", "last_seen": now - 50},  # Recent
            {"session_id": "s4", "user_id": "u4", "last_seen": now},  # Current
        ]
        for sess in sessions_data:
            await store.save_session(sess)

        # Clear sessions older than 100 seconds
        count = await store.clear_sessions(older_than_seconds=100)
        assert count == 2  # s1 and s2

        sessions = await store.load_sessions()
        assert len(sessions) == 2
        session_ids = [s["session_id"] for s in sessions]
        assert "s3" in session_ids
        assert "s4" in session_ids

    @pytest.mark.asyncio
    async def test_session_without_session_id(self, store):
        """Saving a session without session_id should be a no-op."""
        session = {"user_id": "u1", "last_seen": 100.0}
        await store.save_session(session)
        sessions = await store.load_sessions()
        assert len(sessions) == 0

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


# =============================================================================
# Session Timeout/Expiration Pattern Tests
# =============================================================================


class TestSessionExpiration:
    """Test session expiration patterns for both store implementations."""

    @pytest.fixture
    def memory_store(self):
        return InMemoryGatewayStore()

    @pytest.fixture
    def file_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield FileGatewayStore(
                path=Path(tmpdir) / "gateway.json",
                auto_save=True,
                auto_save_interval=0,
            )

    @pytest.mark.asyncio
    async def test_session_expiration_by_last_seen_memory(self, memory_store):
        """Sessions should expire based on last_seen timestamp (memory store)."""
        import time

        now = time.time()

        # Create sessions with different last_seen times
        old_session = {"session_id": "old", "user_id": "u1", "last_seen": now - 3600}  # 1 hour ago
        active_session = {"session_id": "active", "user_id": "u2", "last_seen": now - 60}  # 1 minute ago

        await memory_store.save_session(old_session)
        await memory_store.save_session(active_session)

        # Clear sessions older than 30 minutes (1800 seconds)
        count = await memory_store.clear_sessions(older_than_seconds=1800)
        assert count == 1  # Only the old session

        sessions = await memory_store.load_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "active"

    @pytest.mark.asyncio
    async def test_session_expiration_by_last_seen_file(self, file_store):
        """Sessions should expire based on last_seen timestamp (file store)."""
        import time

        now = time.time()

        # Create sessions with different last_seen times
        old_session = {"session_id": "old", "user_id": "u1", "last_seen": now - 3600}
        active_session = {"session_id": "active", "user_id": "u2", "last_seen": now - 60}

        await file_store.save_session(old_session)
        await file_store.save_session(active_session)

        # Clear sessions older than 30 minutes
        count = await file_store.clear_sessions(older_than_seconds=1800)
        assert count == 1

        sessions = await file_store.load_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "active"

    @pytest.mark.asyncio
    async def test_session_expiration_by_created_at_fallback_memory(self, memory_store):
        """Sessions should fall back to created_at if last_seen is missing (memory store)."""
        import time

        now = time.time()

        # Session without last_seen but with created_at
        session_no_last_seen = {
            "session_id": "s1",
            "user_id": "u1",
            "created_at": now - 3600,  # 1 hour ago, no last_seen
        }
        session_with_last_seen = {
            "session_id": "s2",
            "user_id": "u2",
            "created_at": now - 3600,  # Created 1 hour ago
            "last_seen": now - 60,     # But seen 1 minute ago
        }

        await memory_store.save_session(session_no_last_seen)
        await memory_store.save_session(session_with_last_seen)

        # Clear sessions older than 30 minutes
        count = await memory_store.clear_sessions(older_than_seconds=1800)
        assert count == 1  # s1 (using created_at fallback)

        sessions = await memory_store.load_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "s2"

    @pytest.mark.asyncio
    async def test_session_expiration_by_created_at_fallback_file(self, file_store):
        """Sessions should fall back to created_at if last_seen is missing (file store)."""
        import time

        now = time.time()

        session_no_last_seen = {
            "session_id": "s1",
            "user_id": "u1",
            "created_at": now - 3600,
        }
        session_with_last_seen = {
            "session_id": "s2",
            "user_id": "u2",
            "created_at": now - 3600,
            "last_seen": now - 60,
        }

        await file_store.save_session(session_no_last_seen)
        await file_store.save_session(session_with_last_seen)

        count = await file_store.clear_sessions(older_than_seconds=1800)
        assert count == 1

        sessions = await file_store.load_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "s2"

    @pytest.mark.asyncio
    async def test_session_cleanup_preserves_active_memory(self, memory_store):
        """Cleanup should preserve all active sessions (memory store)."""
        import time

        now = time.time()

        # Create multiple active sessions
        for i in range(5):
            await memory_store.save_session({
                "session_id": f"active_{i}",
                "user_id": f"u{i}",
                "last_seen": now - (i * 60),  # 0 to 4 minutes ago
            })

        # Create some old sessions
        for i in range(3):
            await memory_store.save_session({
                "session_id": f"old_{i}",
                "user_id": f"old_u{i}",
                "last_seen": now - 7200 - (i * 60),  # 2+ hours ago
            })

        # Clear sessions older than 1 hour
        count = await memory_store.clear_sessions(older_than_seconds=3600)
        assert count == 3

        sessions = await memory_store.load_sessions()
        assert len(sessions) == 5
        # All remaining sessions should be "active_*"
        for sess in sessions:
            assert sess["session_id"].startswith("active_")

    @pytest.mark.asyncio
    async def test_session_cleanup_preserves_active_file(self, file_store):
        """Cleanup should preserve all active sessions (file store)."""
        import time

        now = time.time()

        for i in range(5):
            await file_store.save_session({
                "session_id": f"active_{i}",
                "user_id": f"u{i}",
                "last_seen": now - (i * 60),
            })

        for i in range(3):
            await file_store.save_session({
                "session_id": f"old_{i}",
                "user_id": f"old_u{i}",
                "last_seen": now - 7200 - (i * 60),
            })

        count = await file_store.clear_sessions(older_than_seconds=3600)
        assert count == 3

        sessions = await file_store.load_sessions()
        assert len(sessions) == 5
        for sess in sessions:
            assert sess["session_id"].startswith("active_")


# =============================================================================
# Session Metadata Handling Tests
# =============================================================================


class TestSessionMetadata:
    """Test session metadata handling for both store implementations."""

    @pytest.fixture
    def memory_store(self):
        return InMemoryGatewayStore()

    @pytest.fixture
    def file_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield FileGatewayStore(
                path=Path(tmpdir) / "gateway.json",
                auto_save=True,
                auto_save_interval=0,
            )

    @pytest.mark.asyncio
    async def test_session_with_complex_metadata_memory(self, memory_store):
        """Test session with complex nested metadata (memory store)."""
        session = {
            "session_id": "s1",
            "user_id": "u1",
            "last_seen": 100.0,
            "metadata": {
                "role": "admin",
                "permissions": ["read", "write", "delete"],
                "settings": {
                    "theme": "dark",
                    "notifications": True,
                    "nested": {
                        "deep": {"value": 42}
                    }
                },
                "tags": ["important", "vip"],
                "numeric": 12345,
                "float_val": 3.14159,
                "boolean": False,
                "null_val": None,
            }
        }

        await memory_store.save_session(session)
        sessions = await memory_store.load_sessions()

        assert len(sessions) == 1
        loaded = sessions[0]
        assert loaded["metadata"]["role"] == "admin"
        assert loaded["metadata"]["permissions"] == ["read", "write", "delete"]
        assert loaded["metadata"]["settings"]["theme"] == "dark"
        assert loaded["metadata"]["settings"]["nested"]["deep"]["value"] == 42
        assert loaded["metadata"]["numeric"] == 12345
        assert loaded["metadata"]["float_val"] == 3.14159
        assert loaded["metadata"]["boolean"] is False
        assert loaded["metadata"]["null_val"] is None

    @pytest.mark.asyncio
    async def test_session_with_complex_metadata_file(self, file_store):
        """Test session with complex nested metadata (file store)."""
        session = {
            "session_id": "s1",
            "user_id": "u1",
            "last_seen": 100.0,
            "metadata": {
                "role": "admin",
                "permissions": ["read", "write", "delete"],
                "settings": {
                    "theme": "dark",
                    "notifications": True,
                    "nested": {
                        "deep": {"value": 42}
                    }
                },
                "tags": ["important", "vip"],
                "numeric": 12345,
                "float_val": 3.14159,
                "boolean": False,
                "null_val": None,
            }
        }

        await file_store.save_session(session)
        sessions = await file_store.load_sessions()

        assert len(sessions) == 1
        loaded = sessions[0]
        assert loaded["metadata"]["role"] == "admin"
        assert loaded["metadata"]["permissions"] == ["read", "write", "delete"]
        assert loaded["metadata"]["settings"]["theme"] == "dark"
        assert loaded["metadata"]["settings"]["nested"]["deep"]["value"] == 42

    @pytest.mark.asyncio
    async def test_session_metadata_persistence(self):
        """Test that session metadata persists across file store instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gateway.json"

            # Save with first instance
            store1 = FileGatewayStore(path=path, auto_save=True, auto_save_interval=0)
            session = {
                "session_id": "s1",
                "user_id": "u1",
                "last_seen": 100.0,
                "metadata": {
                    "complex": {"nested": {"data": [1, 2, 3]}},
                    "unicode": "Hello World",
                }
            }
            await store1.save_session(session)
            await store1.close()

            # Load with second instance
            store2 = FileGatewayStore(path=path)
            sessions = await store2.load_sessions()
            await store2.close()

            assert len(sessions) == 1
            loaded = sessions[0]
            assert loaded["metadata"]["complex"]["nested"]["data"] == [1, 2, 3]
            assert loaded["metadata"]["unicode"] == "Hello World"


# =============================================================================
# Session Edge Cases Tests
# =============================================================================


class TestSessionEdgeCases:
    """Test session edge cases for both store implementations."""

    @pytest.fixture
    def memory_store(self):
        return InMemoryGatewayStore()

    @pytest.fixture
    def file_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield FileGatewayStore(
                path=Path(tmpdir) / "gateway.json",
                auto_save=True,
                auto_save_interval=0,
            )

    @pytest.mark.asyncio
    async def test_save_session_empty_dict_memory(self, memory_store):
        """Saving an empty session dict should be a no-op (memory store)."""
        await memory_store.save_session({})
        sessions = await memory_store.load_sessions()
        assert len(sessions) == 0

    @pytest.mark.asyncio
    async def test_save_session_empty_dict_file(self, file_store):
        """Saving an empty session dict should be a no-op (file store)."""
        await file_store.save_session({})
        sessions = await file_store.load_sessions()
        assert len(sessions) == 0

    @pytest.mark.asyncio
    async def test_load_sessions_empty_store_memory(self, memory_store):
        """Loading sessions from empty store should return empty list (memory store)."""
        sessions = await memory_store.load_sessions()
        assert sessions == []

    @pytest.mark.asyncio
    async def test_load_sessions_empty_store_file(self, file_store):
        """Loading sessions from empty store should return empty list (file store)."""
        sessions = await file_store.load_sessions()
        assert sessions == []

    @pytest.mark.asyncio
    async def test_session_ordering_with_ties_memory(self, memory_store):
        """Sessions with same last_seen should maintain stable ordering (memory store)."""
        same_time = 1000.0

        # Save multiple sessions with identical last_seen
        for i in range(5):
            await memory_store.save_session({
                "session_id": f"s{i}",
                "user_id": f"u{i}",
                "last_seen": same_time,
            })

        sessions = await memory_store.load_sessions()
        assert len(sessions) == 5
        # All should be present (order may vary but all with same timestamp)
        session_ids = {s["session_id"] for s in sessions}
        assert session_ids == {"s0", "s1", "s2", "s3", "s4"}

    @pytest.mark.asyncio
    async def test_session_ordering_with_ties_file(self, file_store):
        """Sessions with same last_seen should maintain stable ordering (file store)."""
        same_time = 1000.0

        for i in range(5):
            await file_store.save_session({
                "session_id": f"s{i}",
                "user_id": f"u{i}",
                "last_seen": same_time,
            })

        sessions = await file_store.load_sessions()
        assert len(sessions) == 5
        session_ids = {s["session_id"] for s in sessions}
        assert session_ids == {"s0", "s1", "s2", "s3", "s4"}

    @pytest.mark.asyncio
    async def test_session_with_unicode_values_memory(self, memory_store):
        """Test session with unicode values in various fields (memory store)."""
        session = {
            "session_id": "s1",
            "user_id": "user_hello_world",
            "last_seen": 100.0,
            "name": "Hello World",
            "metadata": {"greeting": "world"}
        }

        await memory_store.save_session(session)
        sessions = await memory_store.load_sessions()

        assert len(sessions) == 1
        assert sessions[0]["name"] == "Hello World"

    @pytest.mark.asyncio
    async def test_session_with_unicode_values_file(self, file_store):
        """Test session with unicode values in various fields (file store)."""
        session = {
            "session_id": "s1",
            "user_id": "user_hello",
            "last_seen": 100.0,
            "name": "Hello World",
            "metadata": {"greeting": "world"}
        }

        await file_store.save_session(session)
        sessions = await file_store.load_sessions()

        assert len(sessions) == 1
        assert sessions[0]["name"] == "Hello World"

    @pytest.mark.asyncio
    async def test_session_with_very_long_id_memory(self, memory_store):
        """Test session with very long session_id (memory store)."""
        long_id = "s" * 1000
        session = {
            "session_id": long_id,
            "user_id": "u1",
            "last_seen": 100.0,
        }

        await memory_store.save_session(session)
        sessions = await memory_store.load_sessions()

        assert len(sessions) == 1
        assert sessions[0]["session_id"] == long_id
        assert len(sessions[0]["session_id"]) == 1000

    @pytest.mark.asyncio
    async def test_session_with_very_long_id_file(self, file_store):
        """Test session with very long session_id (file store)."""
        long_id = "s" * 1000
        session = {
            "session_id": long_id,
            "user_id": "u1",
            "last_seen": 100.0,
        }

        await file_store.save_session(session)
        sessions = await file_store.load_sessions()

        assert len(sessions) == 1
        assert sessions[0]["session_id"] == long_id

    @pytest.mark.asyncio
    async def test_clear_sessions_returns_zero_when_empty_memory(self, memory_store):
        """Clear sessions should return 0 when store is empty (memory store)."""
        count = await memory_store.clear_sessions()
        assert count == 0

    @pytest.mark.asyncio
    async def test_clear_sessions_returns_zero_when_empty_file(self, file_store):
        """Clear sessions should return 0 when store is empty (file store)."""
        count = await file_store.clear_sessions()
        assert count == 0

    @pytest.mark.asyncio
    async def test_clear_sessions_with_threshold_returns_zero_when_all_recent_memory(self, memory_store):
        """Clear sessions with threshold should return 0 when all sessions are recent (memory store)."""
        import time

        now = time.time()
        for i in range(3):
            await memory_store.save_session({
                "session_id": f"s{i}",
                "user_id": f"u{i}",
                "last_seen": now - (i * 10),  # All within last 30 seconds
            })

        count = await memory_store.clear_sessions(older_than_seconds=60)
        assert count == 0
        sessions = await memory_store.load_sessions()
        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_clear_sessions_with_threshold_returns_zero_when_all_recent_file(self, file_store):
        """Clear sessions with threshold should return 0 when all sessions are recent (file store)."""
        import time

        now = time.time()
        for i in range(3):
            await file_store.save_session({
                "session_id": f"s{i}",
                "user_id": f"u{i}",
                "last_seen": now - (i * 10),
            })

        count = await file_store.clear_sessions(older_than_seconds=60)
        assert count == 0
        sessions = await file_store.load_sessions()
        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_session_with_zero_last_seen_memory(self, memory_store):
        """Test session with last_seen of 0 (epoch) (memory store)."""
        session = {"session_id": "s1", "user_id": "u1", "last_seen": 0}
        await memory_store.save_session(session)

        sessions = await memory_store.load_sessions()
        assert len(sessions) == 1
        assert sessions[0]["last_seen"] == 0

    @pytest.mark.asyncio
    async def test_session_with_zero_last_seen_file(self, file_store):
        """Test session with last_seen of 0 (epoch) (file store)."""
        session = {"session_id": "s1", "user_id": "u1", "last_seen": 0}
        await file_store.save_session(session)

        sessions = await file_store.load_sessions()
        assert len(sessions) == 1
        assert sessions[0]["last_seen"] == 0

    @pytest.mark.asyncio
    async def test_session_without_timestamps_uses_zero_for_ordering_memory(self, memory_store):
        """Sessions without timestamps should use 0 for ordering (memory store)."""
        # Session without any timestamp
        session_no_ts = {"session_id": "s1", "user_id": "u1"}
        # Session with last_seen
        session_with_ts = {"session_id": "s2", "user_id": "u2", "last_seen": 100.0}

        await memory_store.save_session(session_no_ts)
        await memory_store.save_session(session_with_ts)

        sessions = await memory_store.load_sessions()
        # Session with timestamp should come first (higher = more recent)
        assert sessions[0]["session_id"] == "s2"
        assert sessions[1]["session_id"] == "s1"

    @pytest.mark.asyncio
    async def test_session_without_timestamps_uses_zero_for_ordering_file(self, file_store):
        """Sessions without timestamps should use 0 for ordering (file store)."""
        session_no_ts = {"session_id": "s1", "user_id": "u1"}
        session_with_ts = {"session_id": "s2", "user_id": "u2", "last_seen": 100.0}

        await file_store.save_session(session_no_ts)
        await file_store.save_session(session_with_ts)

        sessions = await file_store.load_sessions()
        assert sessions[0]["session_id"] == "s2"
        assert sessions[1]["session_id"] == "s1"
