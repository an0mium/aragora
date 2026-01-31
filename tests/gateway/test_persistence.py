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

    def test_redis_backend_explicit(self):
        """Test explicit redis backend."""
        with patch("aragora.gateway.persistence.RedisGatewayStore") as MockRedisStore:
            mock_instance = MagicMock()
            MockRedisStore.return_value = mock_instance
            store = get_gateway_store("redis", redis_url="redis://localhost:6379")
            assert store is mock_instance
            MockRedisStore.assert_called_once_with(redis_url="redis://localhost:6379")

    def test_auto_backend_with_redis_url(self):
        """Test auto backend with REDIS_URL env var."""
        with patch.dict(os.environ, {"REDIS_URL": "redis://redis.example.com:6379"}):
            with patch("aragora.gateway.persistence.RedisGatewayStore") as MockRedisStore:
                mock_instance = MagicMock()
                MockRedisStore.return_value = mock_instance
                # Mock the import check
                with patch.dict("sys.modules", {"redis.asyncio": MagicMock()}):
                    store = get_gateway_store("auto")
                    assert store is mock_instance
                    MockRedisStore.assert_called_once_with(
                        redis_url="redis://redis.example.com:6379"
                    )

    def test_auto_backend_redis_import_error_fallback(self):
        """Test auto backend falls back to file when redis import fails."""
        with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}):
            # Simulate ImportError during the import check in get_gateway_store
            original_import = __builtins__["__import__"]

            def mock_import(name, *args, **kwargs):
                if name == "redis.asyncio":
                    raise ImportError("No module named 'redis'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                with tempfile.TemporaryDirectory() as tmpdir:
                    store = get_gateway_store("auto", path=Path(tmpdir) / "test.json")
                    assert isinstance(store, FileGatewayStore)


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
        active_session = {
            "session_id": "active",
            "user_id": "u2",
            "last_seen": now - 60,
        }  # 1 minute ago

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
            "last_seen": now - 60,  # But seen 1 minute ago
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
            await memory_store.save_session(
                {
                    "session_id": f"active_{i}",
                    "user_id": f"u{i}",
                    "last_seen": now - (i * 60),  # 0 to 4 minutes ago
                }
            )

        # Create some old sessions
        for i in range(3):
            await memory_store.save_session(
                {
                    "session_id": f"old_{i}",
                    "user_id": f"old_u{i}",
                    "last_seen": now - 7200 - (i * 60),  # 2+ hours ago
                }
            )

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
            await file_store.save_session(
                {
                    "session_id": f"active_{i}",
                    "user_id": f"u{i}",
                    "last_seen": now - (i * 60),
                }
            )

        for i in range(3):
            await file_store.save_session(
                {
                    "session_id": f"old_{i}",
                    "user_id": f"old_u{i}",
                    "last_seen": now - 7200 - (i * 60),
                }
            )

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
                    "nested": {"deep": {"value": 42}},
                },
                "tags": ["important", "vip"],
                "numeric": 12345,
                "float_val": 3.14159,
                "boolean": False,
                "null_val": None,
            },
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
                    "nested": {"deep": {"value": 42}},
                },
                "tags": ["important", "vip"],
                "numeric": 12345,
                "float_val": 3.14159,
                "boolean": False,
                "null_val": None,
            },
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
                },
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
            await memory_store.save_session(
                {
                    "session_id": f"s{i}",
                    "user_id": f"u{i}",
                    "last_seen": same_time,
                }
            )

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
            await file_store.save_session(
                {
                    "session_id": f"s{i}",
                    "user_id": f"u{i}",
                    "last_seen": same_time,
                }
            )

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
            "metadata": {"greeting": "world"},
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
            "metadata": {"greeting": "world"},
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
    async def test_clear_sessions_with_threshold_returns_zero_when_all_recent_memory(
        self, memory_store
    ):
        """Clear sessions with threshold should return 0 when all sessions are recent (memory store)."""
        import time

        now = time.time()
        for i in range(3):
            await memory_store.save_session(
                {
                    "session_id": f"s{i}",
                    "user_id": f"u{i}",
                    "last_seen": now - (i * 10),  # All within last 30 seconds
                }
            )

        count = await memory_store.clear_sessions(older_than_seconds=60)
        assert count == 0
        sessions = await memory_store.load_sessions()
        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_clear_sessions_with_threshold_returns_zero_when_all_recent_file(
        self, file_store
    ):
        """Clear sessions with threshold should return 0 when all sessions are recent (file store)."""
        import time

        now = time.time()
        for i in range(3):
            await file_store.save_session(
                {
                    "session_id": f"s{i}",
                    "user_id": f"u{i}",
                    "last_seen": now - (i * 10),
                }
            )

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


# =============================================================================
# Factory Edge Case Tests
# =============================================================================


class TestGetGatewayStoreEdgeCases:
    """Edge case tests for get_gateway_store factory."""

    def test_auto_backend_with_redis_env_var(self):
        """Test that auto backend attempts Redis when REDIS_URL is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save original env
            original_redis_url = os.environ.get("REDIS_URL")
            try:
                os.environ["REDIS_URL"] = "redis://localhost:6379"
                # Since redis may not be installed, it should fall back to file
                # We patch to simulate redis import failure
                import builtins

                original_import = builtins.__import__

                def mock_import(name, *args, **kwargs):
                    if "redis" in name:
                        raise ImportError("No module named 'redis'")
                    return original_import(name, *args, **kwargs)

                with patch.object(builtins, "__import__", mock_import):
                    store = get_gateway_store("auto", path=Path(tmpdir) / "test.json")
                    assert isinstance(store, FileGatewayStore)
            finally:
                if original_redis_url is None:
                    os.environ.pop("REDIS_URL", None)
                else:
                    os.environ["REDIS_URL"] = original_redis_url

    def test_auto_backend_redis_import_error(self):
        """Test fallback to file store when redis is not importable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_redis_url = os.environ.get("REDIS_URL")
            try:
                os.environ["REDIS_URL"] = "redis://localhost:6379"
                import builtins

                original_import = builtins.__import__

                def mock_import(name, *args, **kwargs):
                    if "redis" in name:
                        raise ImportError("No module named 'redis'")
                    return original_import(name, *args, **kwargs)

                with patch.object(builtins, "__import__", mock_import):
                    store = get_gateway_store("auto", path=Path(tmpdir) / "test.json")
                    # Should fall back to FileGatewayStore
                    assert isinstance(store, FileGatewayStore)
            finally:
                if original_redis_url is None:
                    os.environ.pop("REDIS_URL", None)
                else:
                    os.environ["REDIS_URL"] = original_redis_url

    def test_file_backend_creates_directory(self):
        """Test that file backend creates parent directories on save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "dir" / "gateway.json"
            store = get_gateway_store("file", path=nested_path)
            assert isinstance(store, FileGatewayStore)
            assert store._path == nested_path

    def test_file_backend_default_path(self):
        """Test that file backend uses default path when none provided."""
        store = get_gateway_store("file")
        assert isinstance(store, FileGatewayStore)
        expected_default = Path.home() / ".aragora" / "gateway.json"
        assert store._path == expected_default


# =============================================================================
# FileGatewayStore Edge Case Tests
# =============================================================================


class TestFileGatewayStoreEdgeCases:
    """Edge case tests for FileGatewayStore."""

    @pytest.fixture
    def temp_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "gateway.json"

    @pytest.mark.asyncio
    async def test_malformed_message_in_file(self, temp_path):
        """Test that malformed messages are skipped during load."""
        import json

        temp_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "saved_at": 100.0,
            "messages": [
                {
                    "message_id": "m1",
                    "channel": "slack",
                    "sender": "alice",
                    "content": "valid message",
                },
                {
                    # Malformed: missing required fields
                    "message_id": "m2",
                    # missing channel, sender, content
                },
                {
                    "message_id": "m3",
                    "channel": "slack",
                    "sender": "bob",
                    "content": "another valid message",
                },
            ],
            "devices": [],
            "rules": [],
            "sessions": [],
        }
        with open(temp_path, "w") as f:
            json.dump(data, f)

        store = FileGatewayStore(path=temp_path)
        messages = await store.load_messages()

        # Should have loaded 2 valid messages, skipped the malformed one
        assert len(messages) == 2
        message_ids = [m.message_id for m in messages]
        assert "m1" in message_ids
        assert "m3" in message_ids
        assert "m2" not in message_ids
        await store.close()

    @pytest.mark.asyncio
    async def test_malformed_device_in_file(self, temp_path):
        """Test that malformed devices are skipped during load."""
        import json

        temp_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "saved_at": 100.0,
            "messages": [],
            "devices": [
                {
                    "device_id": "d1",
                    "name": "Laptop",
                    "device_type": "laptop",
                },
                {
                    # Malformed: status is invalid enum value
                    "device_id": "d2",
                    "name": "Phone",
                    "status": "invalid_status",  # This will cause error
                },
                {
                    "device_id": "d3",
                    "name": "Server",
                    "device_type": "server",
                },
            ],
            "rules": [],
            "sessions": [],
        }
        with open(temp_path, "w") as f:
            json.dump(data, f)

        store = FileGatewayStore(path=temp_path)
        devices = await store.load_devices()

        # Should have loaded 2 valid devices, skipped the malformed one
        assert len(devices) == 2
        device_ids = [d.device_id for d in devices]
        assert "d1" in device_ids
        assert "d3" in device_ids
        await store.close()

    @pytest.mark.asyncio
    async def test_malformed_rule_in_file(self, temp_path):
        """Test that malformed rules are skipped during load."""
        import json

        temp_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "saved_at": 100.0,
            "messages": [],
            "devices": [],
            "rules": [
                {
                    "rule_id": "r1",
                    "agent_id": "claude",
                },
                {
                    # Malformed: missing required fields
                    # missing rule_id and agent_id
                },
                {
                    "rule_id": "r3",
                    "agent_id": "gemini",
                },
            ],
            "sessions": [],
        }
        with open(temp_path, "w") as f:
            json.dump(data, f)

        store = FileGatewayStore(path=temp_path)
        rules = await store.load_rules()

        # Should have loaded 2 valid rules, skipped the malformed one
        assert len(rules) == 2
        rule_ids = [r.rule_id for r in rules]
        assert "r1" in rule_ids
        assert "r3" in rule_ids
        await store.close()

    @pytest.mark.asyncio
    async def test_malformed_session_in_file(self, temp_path):
        """Test that malformed sessions are skipped during load."""
        import json

        temp_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "saved_at": 100.0,
            "messages": [],
            "devices": [],
            "rules": [],
            "sessions": [
                {
                    "session_id": "s1",
                    "user_id": "u1",
                    "status": "active",
                },
                {
                    # Missing session_id - should be skipped
                    "user_id": "u2",
                    "status": "active",
                },
                {
                    "session_id": "s3",
                    "user_id": "u3",
                    "status": "active",
                },
            ],
        }
        with open(temp_path, "w") as f:
            json.dump(data, f)

        store = FileGatewayStore(path=temp_path)
        sessions = await store.load_sessions()

        # Should have loaded 2 valid sessions, skipped the one without session_id
        assert len(sessions) == 2
        session_ids = [s["session_id"] for s in sessions]
        assert "s1" in session_ids
        assert "s3" in session_ids
        await store.close()

    @pytest.mark.asyncio
    async def test_auto_save_interval_throttling(self, temp_path):
        """Test that auto-save is throttled by interval."""
        store = FileGatewayStore(
            path=temp_path,
            auto_save=True,
            auto_save_interval=10.0,  # 10 second interval
        )

        # First save should work
        msg1 = InboxMessage(message_id="m1", channel="s", sender="a", content="1")
        await store.save_message(msg1)

        # Second save should be throttled (interval not elapsed)
        msg2 = InboxMessage(message_id="m2", channel="s", sender="a", content="2")
        await store.save_message(msg2)

        # _dirty should still be True because interval hasn't elapsed
        assert store._dirty is True

        await store.close()

    @pytest.mark.asyncio
    async def test_dirty_flag_behavior(self, temp_path):
        """Test that dirty flag is set on modifications."""
        store = FileGatewayStore(path=temp_path, auto_save=False)

        # Initially not dirty
        assert store._dirty is False

        # Save message should set dirty
        msg = InboxMessage(message_id="m1", channel="s", sender="a", content="hi")
        await store.save_message(msg)
        assert store._dirty is True

        # Force save clears dirty
        await store._save(force=True)
        assert store._dirty is False

        # Delete should set dirty
        await store.delete_message("m1")
        assert store._dirty is True

        await store.close()

    @pytest.mark.asyncio
    async def test_force_save_ignores_interval(self, temp_path):
        """Test that force save ignores the auto-save interval."""
        store = FileGatewayStore(
            path=temp_path,
            auto_save=False,
            auto_save_interval=1000.0,  # Very long interval
        )

        # Save a message
        msg = InboxMessage(message_id="m1", channel="s", sender="a", content="hi")
        await store.save_message(msg)

        # Force save should work regardless of interval
        await store._save(force=True)
        assert store._dirty is False
        assert temp_path.exists()

        await store.close()


# =============================================================================
# Data Integrity Edge Cases
# =============================================================================


class TestDataIntegrityEdgeCases:
    """Tests for data integrity edge cases."""

    def test_large_metadata_handling(self):
        """Test serialization with large metadata."""
        large_metadata = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}
        msg = InboxMessage(
            message_id="m1",
            channel="slack",
            sender="alice",
            content="test",
            metadata=large_metadata,
        )
        d = _message_to_dict(msg)
        restored = _dict_to_message(d)

        assert restored.metadata == large_metadata
        assert len(restored.metadata) == 100

    def test_special_characters_in_strings(self):
        """Test serialization with special characters."""
        special_content = (
            "Test with \"quotes\", 'apostrophes', \n newlines, \t tabs, and \\ backslashes"
        )
        special_sender = "user@domain.com"

        msg = InboxMessage(
            message_id="m1",
            channel="slack",
            sender=special_sender,
            content=special_content,
            metadata={"special": "value with <html> & entities"},
        )
        d = _message_to_dict(msg)
        restored = _dict_to_message(d)

        assert restored.sender == special_sender
        assert restored.content == special_content
        assert restored.metadata["special"] == "value with <html> & entities"

    def test_empty_string_fields(self):
        """Test serialization with empty string fields."""
        msg = InboxMessage(
            message_id="m1",
            channel="",
            sender="",
            content="",
            thread_id="",
        )
        d = _message_to_dict(msg)
        restored = _dict_to_message(d)

        assert restored.channel == ""
        assert restored.sender == ""
        assert restored.content == ""
        assert restored.thread_id == ""

        # Test device with empty strings
        device = DeviceNode(
            device_id="",
            name="",
            device_type="",
        )
        d = _device_to_dict(device)
        restored_device = _dict_to_device(d)

        assert restored_device.name == ""
        assert restored_device.device_type == ""

        # Test rule with empty patterns
        rule = RoutingRule(
            rule_id="r1",
            agent_id="",
            channel_pattern="",
            sender_pattern="",
            content_pattern="",
        )
        d = _rule_to_dict(rule)
        restored_rule = _dict_to_rule(d)

        assert restored_rule.agent_id == ""
        assert restored_rule.channel_pattern == ""
        assert restored_rule.sender_pattern == ""
        assert restored_rule.content_pattern == ""


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestPersistenceConcurrency:
    """
    Test concurrent operations for gateway persistence stores.

    Verifies thread/async safety for both InMemoryGatewayStore and FileGatewayStore.
    """

    @pytest.fixture
    def memory_store(self):
        return InMemoryGatewayStore()

    @pytest.fixture
    def temp_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "gateway.json"

    @pytest.fixture
    def file_store(self, temp_path):
        return FileGatewayStore(path=temp_path, auto_save=True, auto_save_interval=0)

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _create_message(self, i: int) -> InboxMessage:
        """Create a test message with given index."""
        return InboxMessage(
            message_id=f"msg-{i}",
            channel="test-channel",
            sender=f"sender-{i}",
            content=f"Content for message {i}",
            priority=MessagePriority.NORMAL,
        )

    def _create_device(self, i: int) -> DeviceNode:
        """Create a test device with given index."""
        return DeviceNode(
            device_id=f"device-{i}",
            name=f"Device {i}",
            device_type="test",
            status=DeviceStatus.ONLINE,
        )

    def _create_rule(self, i: int, priority: int | None = None) -> RoutingRule:
        """Create a test routing rule with given index."""
        return RoutingRule(
            rule_id=f"rule-{i}",
            agent_id=f"agent-{i}",
            priority=priority if priority is not None else i,
        )

    def _create_session(self, i: int) -> dict:
        """Create a test session with given index."""
        import time

        return {
            "session_id": f"session-{i}",
            "user_id": f"user-{i}",
            "device_id": f"device-{i}",
            "status": "active",
            "created_at": time.time(),
            "last_seen": time.time(),
        }

    # -------------------------------------------------------------------------
    # Concurrent Message Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_concurrent_message_saves_memory(self, memory_store):
        """Test multiple concurrent save_message calls on InMemoryGatewayStore."""
        num_messages = 50
        messages = [self._create_message(i) for i in range(num_messages)]

        # Save all messages concurrently
        await asyncio.gather(*[memory_store.save_message(msg) for msg in messages])

        # Verify all messages are saved without data loss
        loaded = await memory_store.load_messages(limit=num_messages + 10)
        assert len(loaded) == num_messages

        loaded_ids = {m.message_id for m in loaded}
        expected_ids = {f"msg-{i}" for i in range(num_messages)}
        assert loaded_ids == expected_ids

    @pytest.mark.asyncio
    async def test_concurrent_message_saves_file(self, file_store):
        """Test multiple concurrent save_message calls on FileGatewayStore."""
        num_messages = 30
        messages = [self._create_message(i) for i in range(num_messages)]

        # Save all messages concurrently
        await asyncio.gather(*[file_store.save_message(msg) for msg in messages])

        # Verify all messages are saved without data loss
        loaded = await file_store.load_messages(limit=num_messages + 10)
        assert len(loaded) == num_messages

        loaded_ids = {m.message_id for m in loaded}
        expected_ids = {f"msg-{i}" for i in range(num_messages)}
        assert loaded_ids == expected_ids

    @pytest.mark.asyncio
    async def test_concurrent_save_then_load_memory(self, memory_store):
        """Test concurrent save followed by load on InMemoryGatewayStore."""
        num_messages = 20

        async def save_and_load(i: int):
            msg = self._create_message(i)
            await memory_store.save_message(msg)
            # Small delay to allow other saves to happen
            await asyncio.sleep(0.001)
            return await memory_store.load_messages(limit=100)

        results = await asyncio.gather(*[save_and_load(i) for i in range(num_messages)])

        # Final check: all messages should be present
        final_loaded = await memory_store.load_messages(limit=100)
        assert len(final_loaded) == num_messages

    @pytest.mark.asyncio
    async def test_concurrent_save_then_load_file(self, file_store):
        """Test concurrent save followed by load on FileGatewayStore."""
        num_messages = 15

        async def save_and_load(i: int):
            msg = self._create_message(i)
            await file_store.save_message(msg)
            await asyncio.sleep(0.001)
            return await file_store.load_messages(limit=100)

        await asyncio.gather(*[save_and_load(i) for i in range(num_messages)])

        # Final check: all messages should be present
        final_loaded = await file_store.load_messages(limit=100)
        assert len(final_loaded) == num_messages

    # -------------------------------------------------------------------------
    # Concurrent Device Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_concurrent_device_saves_memory(self, memory_store):
        """Test multiple concurrent save_device calls on InMemoryGatewayStore."""
        num_devices = 50
        devices = [self._create_device(i) for i in range(num_devices)]

        await asyncio.gather(*[memory_store.save_device(dev) for dev in devices])

        loaded = await memory_store.load_devices()
        assert len(loaded) == num_devices

        loaded_ids = {d.device_id for d in loaded}
        expected_ids = {f"device-{i}" for i in range(num_devices)}
        assert loaded_ids == expected_ids

    @pytest.mark.asyncio
    async def test_concurrent_device_saves_file(self, file_store):
        """Test multiple concurrent save_device calls on FileGatewayStore."""
        num_devices = 30
        devices = [self._create_device(i) for i in range(num_devices)]

        await asyncio.gather(*[file_store.save_device(dev) for dev in devices])

        loaded = await file_store.load_devices()
        assert len(loaded) == num_devices

    @pytest.mark.asyncio
    async def test_concurrent_save_delete_device_memory(self, memory_store):
        """Test concurrent save_device and delete_device on InMemoryGatewayStore."""
        num_devices = 40

        # First, save all devices
        devices = [self._create_device(i) for i in range(num_devices)]
        await asyncio.gather(*[memory_store.save_device(dev) for dev in devices])

        # Concurrently delete even-numbered devices while saving new ones
        async def delete_device(i: int):
            await memory_store.delete_device(f"device-{i}")

        async def save_new_device(i: int):
            new_dev = self._create_device(num_devices + i)
            await memory_store.save_device(new_dev)

        # Delete even, save new
        tasks = []
        for i in range(0, num_devices, 2):
            tasks.append(delete_device(i))
        for i in range(10):
            tasks.append(save_new_device(i))

        await asyncio.gather(*tasks)

        # Verify registry integrity
        loaded = await memory_store.load_devices()

        # We deleted 20 even-numbered devices (0,2,4,...,38)
        # We added 10 new devices (40-49)
        # Expected: 20 odd devices + 10 new = 30
        assert len(loaded) == 30

        loaded_ids = {d.device_id for d in loaded}
        # Odd devices from 0-39
        expected_odd = {f"device-{i}" for i in range(1, num_devices, 2)}
        # New devices 40-49
        expected_new = {f"device-{i}" for i in range(num_devices, num_devices + 10)}
        assert loaded_ids == expected_odd | expected_new

    @pytest.mark.asyncio
    async def test_concurrent_save_delete_device_file(self, file_store):
        """Test concurrent save_device and delete_device on FileGatewayStore."""
        num_devices = 20

        # First, save all devices
        devices = [self._create_device(i) for i in range(num_devices)]
        await asyncio.gather(*[file_store.save_device(dev) for dev in devices])

        # Concurrently delete even-numbered and save new
        tasks = []
        for i in range(0, num_devices, 2):
            tasks.append(file_store.delete_device(f"device-{i}"))
        for i in range(5):
            new_dev = self._create_device(num_devices + i)
            tasks.append(file_store.save_device(new_dev))

        await asyncio.gather(*tasks)

        loaded = await file_store.load_devices()
        # 10 odd + 5 new = 15
        assert len(loaded) == 15

    # -------------------------------------------------------------------------
    # Concurrent Rule Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_concurrent_rule_saves_memory(self, memory_store):
        """Test multiple concurrent save_rule calls on InMemoryGatewayStore."""
        num_rules = 50
        rules = [self._create_rule(i) for i in range(num_rules)]

        await asyncio.gather(*[memory_store.save_rule(r) for r in rules])

        loaded = await memory_store.load_rules()
        assert len(loaded) == num_rules

    @pytest.mark.asyncio
    async def test_concurrent_rule_saves_file(self, file_store):
        """Test multiple concurrent save_rule calls on FileGatewayStore."""
        num_rules = 30
        rules = [self._create_rule(i) for i in range(num_rules)]

        await asyncio.gather(*[file_store.save_rule(r) for r in rules])

        loaded = await file_store.load_rules()
        assert len(loaded) == num_rules

    @pytest.mark.asyncio
    async def test_concurrent_rules_priority_ordering_memory(self, memory_store):
        """Verify priority ordering is maintained after concurrent saves on InMemoryGatewayStore."""
        num_rules = 30
        # Create rules with varied priorities
        rules = [self._create_rule(i, priority=i * 10) for i in range(num_rules)]

        # Shuffle and save concurrently
        import random

        shuffled = rules.copy()
        random.shuffle(shuffled)

        await asyncio.gather(*[memory_store.save_rule(r) for r in shuffled])

        loaded = await memory_store.load_rules()
        assert len(loaded) == num_rules

        # Verify ordering: highest priority first
        priorities = [r.priority for r in loaded]
        assert priorities == sorted(priorities, reverse=True)

    @pytest.mark.asyncio
    async def test_concurrent_rules_priority_ordering_file(self, file_store):
        """Verify priority ordering is maintained after concurrent saves on FileGatewayStore."""
        num_rules = 20
        rules = [self._create_rule(i, priority=i * 5) for i in range(num_rules)]

        import random

        shuffled = rules.copy()
        random.shuffle(shuffled)

        await asyncio.gather(*[file_store.save_rule(r) for r in shuffled])

        loaded = await file_store.load_rules()
        assert len(loaded) == num_rules

        priorities = [r.priority for r in loaded]
        assert priorities == sorted(priorities, reverse=True)

    # -------------------------------------------------------------------------
    # Concurrent Session Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_concurrent_session_saves_memory(self, memory_store):
        """Test multiple concurrent save_session calls on InMemoryGatewayStore."""
        num_sessions = 50
        sessions = [self._create_session(i) for i in range(num_sessions)]

        await asyncio.gather(*[memory_store.save_session(s) for s in sessions])

        loaded = await memory_store.load_sessions(limit=100)
        assert len(loaded) == num_sessions

        loaded_ids = {s["session_id"] for s in loaded}
        expected_ids = {f"session-{i}" for i in range(num_sessions)}
        assert loaded_ids == expected_ids

    @pytest.mark.asyncio
    async def test_concurrent_session_saves_file(self, file_store):
        """Test multiple concurrent save_session calls on FileGatewayStore."""
        num_sessions = 30
        sessions = [self._create_session(i) for i in range(num_sessions)]

        await asyncio.gather(*[file_store.save_session(s) for s in sessions])

        loaded = await file_store.load_sessions(limit=100)
        assert len(loaded) == num_sessions

    @pytest.mark.asyncio
    async def test_concurrent_clear_sessions_while_saving_memory(self, memory_store):
        """Test concurrent clear_sessions while saving on InMemoryGatewayStore."""
        # First, populate with some sessions
        initial_sessions = [self._create_session(i) for i in range(20)]
        await asyncio.gather(*[memory_store.save_session(s) for s in initial_sessions])

        # Now run concurrent clears and saves
        new_sessions = [self._create_session(100 + i) for i in range(10)]

        async def clear_old():
            await memory_store.clear_sessions()

        async def save_new(s):
            await asyncio.sleep(0.002)  # slight delay to interleave
            await memory_store.save_session(s)

        tasks = [clear_old()]
        tasks.extend([save_new(s) for s in new_sessions])

        await asyncio.gather(*tasks)

        # The final state depends on timing, but we should not have any corruption
        loaded = await memory_store.load_sessions(limit=100)
        # All loaded sessions should be valid dicts with session_id
        for sess in loaded:
            assert "session_id" in sess
            assert isinstance(sess["session_id"], str)

    @pytest.mark.asyncio
    async def test_concurrent_clear_sessions_while_saving_file(self, file_store):
        """Test concurrent clear_sessions while saving on FileGatewayStore."""
        initial_sessions = [self._create_session(i) for i in range(15)]
        await asyncio.gather(*[file_store.save_session(s) for s in initial_sessions])

        new_sessions = [self._create_session(100 + i) for i in range(5)]

        async def clear_old():
            await file_store.clear_sessions()

        async def save_new(s):
            await asyncio.sleep(0.002)
            await file_store.save_session(s)

        tasks = [clear_old()]
        tasks.extend([save_new(s) for s in new_sessions])

        await asyncio.gather(*tasks)

        # Verify no corruption
        loaded = await file_store.load_sessions(limit=100)
        for sess in loaded:
            assert "session_id" in sess

    # -------------------------------------------------------------------------
    # Mixed Operations Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_mixed_operations_memory(self, memory_store):
        """Test concurrent operations across different entity types on InMemoryGatewayStore."""
        num_each = 15

        messages = [self._create_message(i) for i in range(num_each)]
        devices = [self._create_device(i) for i in range(num_each)]
        rules = [self._create_rule(i) for i in range(num_each)]
        sessions = [self._create_session(i) for i in range(num_each)]

        # Mix all operations
        tasks = []
        tasks.extend([memory_store.save_message(m) for m in messages])
        tasks.extend([memory_store.save_device(d) for d in devices])
        tasks.extend([memory_store.save_rule(r) for r in rules])
        tasks.extend([memory_store.save_session(s) for s in sessions])

        await asyncio.gather(*tasks)

        # Verify no cross-contamination
        loaded_messages = await memory_store.load_messages(limit=100)
        loaded_devices = await memory_store.load_devices()
        loaded_rules = await memory_store.load_rules()
        loaded_sessions = await memory_store.load_sessions(limit=100)

        assert len(loaded_messages) == num_each
        assert len(loaded_devices) == num_each
        assert len(loaded_rules) == num_each
        assert len(loaded_sessions) == num_each

        # Verify each type has correct data
        assert all(m.message_id.startswith("msg-") for m in loaded_messages)
        assert all(d.device_id.startswith("device-") for d in loaded_devices)
        assert all(r.rule_id.startswith("rule-") for r in loaded_rules)
        assert all(s["session_id"].startswith("session-") for s in loaded_sessions)

    @pytest.mark.asyncio
    async def test_mixed_operations_file(self, file_store):
        """Test concurrent operations across different entity types on FileGatewayStore."""
        num_each = 10

        messages = [self._create_message(i) for i in range(num_each)]
        devices = [self._create_device(i) for i in range(num_each)]
        rules = [self._create_rule(i) for i in range(num_each)]
        sessions = [self._create_session(i) for i in range(num_each)]

        tasks = []
        tasks.extend([file_store.save_message(m) for m in messages])
        tasks.extend([file_store.save_device(d) for d in devices])
        tasks.extend([file_store.save_rule(r) for r in rules])
        tasks.extend([file_store.save_session(s) for s in sessions])

        await asyncio.gather(*tasks)

        loaded_messages = await file_store.load_messages(limit=100)
        loaded_devices = await file_store.load_devices()
        loaded_rules = await file_store.load_rules()
        loaded_sessions = await file_store.load_sessions(limit=100)

        assert len(loaded_messages) == num_each
        assert len(loaded_devices) == num_each
        assert len(loaded_rules) == num_each
        assert len(loaded_sessions) == num_each

    @pytest.mark.asyncio
    async def test_mixed_save_delete_operations_memory(self, memory_store):
        """Test mixed save and delete operations concurrently on InMemoryGatewayStore."""
        # Setup initial data
        for i in range(20):
            await memory_store.save_message(self._create_message(i))
            await memory_store.save_device(self._create_device(i))

        # Run concurrent saves and deletes
        tasks = []
        # Delete some messages
        for i in range(0, 10, 2):
            tasks.append(memory_store.delete_message(f"msg-{i}"))
        # Save new messages
        for i in range(20, 30):
            tasks.append(memory_store.save_message(self._create_message(i)))
        # Delete some devices
        for i in range(0, 10, 2):
            tasks.append(memory_store.delete_device(f"device-{i}"))
        # Save new devices
        for i in range(20, 30):
            tasks.append(memory_store.save_device(self._create_device(i)))

        await asyncio.gather(*tasks)

        # Verify final state
        messages = await memory_store.load_messages(limit=100)
        devices = await memory_store.load_devices()

        # 20 - 5 deleted + 10 new = 25 messages
        # 20 - 5 deleted + 10 new = 25 devices
        assert len(messages) == 25
        assert len(devices) == 25

    # -------------------------------------------------------------------------
    # FileGatewayStore Specific Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_file_auto_save_interval_throttling(self, temp_path):
        """Test concurrent writes with auto_save_interval throttling."""
        # Use a longer interval to test throttling
        store = FileGatewayStore(path=temp_path, auto_save=True, auto_save_interval=1.0)

        num_messages = 20
        messages = [self._create_message(i) for i in range(num_messages)]

        # Save all concurrently - most should be throttled
        await asyncio.gather(*[store.save_message(m) for m in messages])

        # Force final save
        await store.close()

        # Reload and verify
        store2 = FileGatewayStore(path=temp_path)
        loaded = await store2.load_messages(limit=100)
        assert len(loaded) == num_messages
        await store2.close()

    @pytest.mark.asyncio
    async def test_file_write_atomicity(self, temp_path):
        """Test file write atomicity (temp file rename)."""
        store = FileGatewayStore(path=temp_path, auto_save=True, auto_save_interval=0)

        # Do several concurrent writes
        num_messages = 25
        messages = [self._create_message(i) for i in range(num_messages)]
        await asyncio.gather(*[store.save_message(m) for m in messages])

        # The temp file should not exist (it gets renamed)
        temp_file = temp_path.with_suffix(".tmp")
        assert not temp_file.exists()

        # The main file should exist and be valid JSON
        assert temp_path.exists()

        import json

        with open(temp_path) as f:
            data = json.load(f)
        assert "messages" in data
        assert len(data["messages"]) == num_messages

        await store.close()

    @pytest.mark.asyncio
    async def test_concurrent_load_and_save_file(self, temp_path):
        """Test concurrent load and save operations on FileGatewayStore."""
        store = FileGatewayStore(path=temp_path, auto_save=True, auto_save_interval=0)

        # Pre-populate
        for i in range(10):
            await store.save_message(self._create_message(i))

        # Concurrent loads and saves
        async def load_op():
            return await store.load_messages(limit=100)

        async def save_op(i: int):
            await store.save_message(self._create_message(100 + i))

        tasks = []
        for i in range(10):
            tasks.append(load_op())
            tasks.append(save_op(i))

        results = await asyncio.gather(*tasks)

        # Verify final state
        final = await store.load_messages(limit=100)
        assert len(final) == 20  # 10 original + 10 new

        await store.close()

    @pytest.mark.asyncio
    async def test_file_persistence_after_concurrent_ops(self, temp_path):
        """Test data persists correctly after concurrent operations."""
        store1 = FileGatewayStore(path=temp_path, auto_save=True, auto_save_interval=0)

        # Mixed concurrent operations
        tasks = []
        for i in range(15):
            tasks.append(store1.save_message(self._create_message(i)))
            tasks.append(store1.save_device(self._create_device(i)))
            tasks.append(store1.save_rule(self._create_rule(i)))
            tasks.append(store1.save_session(self._create_session(i)))

        await asyncio.gather(*tasks)
        await store1.close()

        # Verify with new instance
        store2 = FileGatewayStore(path=temp_path)

        messages = await store2.load_messages(limit=100)
        devices = await store2.load_devices()
        rules = await store2.load_rules()
        sessions = await store2.load_sessions(limit=100)

        assert len(messages) == 15
        assert len(devices) == 15
        assert len(rules) == 15
        assert len(sessions) == 15

        await store2.close()

    @pytest.mark.asyncio
    async def test_rapid_concurrent_updates_same_entity(self, memory_store):
        """Test rapid concurrent updates to the same entity."""
        device_id = "device-shared"
        num_updates = 30

        async def update_device(i: int):
            device = DeviceNode(
                device_id=device_id,
                name=f"Device Update {i}",
                device_type="test",
                status=DeviceStatus.ONLINE if i % 2 == 0 else DeviceStatus.OFFLINE,
            )
            await memory_store.save_device(device)

        await asyncio.gather(*[update_device(i) for i in range(num_updates)])

        # Should have exactly one device (all updates to same ID)
        devices = await memory_store.load_devices()
        assert len(devices) == 1
        assert devices[0].device_id == device_id

    @pytest.mark.asyncio
    async def test_rapid_concurrent_updates_same_entity_file(self, file_store):
        """Test rapid concurrent updates to the same entity on FileGatewayStore."""
        device_id = "device-shared"
        num_updates = 20

        async def update_device(i: int):
            device = DeviceNode(
                device_id=device_id,
                name=f"Device Update {i}",
                device_type="test",
                status=DeviceStatus.ONLINE if i % 2 == 0 else DeviceStatus.OFFLINE,
            )
            await file_store.save_device(device)

        await asyncio.gather(*[update_device(i) for i in range(num_updates)])

        devices = await file_store.load_devices()
        assert len(devices) == 1
        assert devices[0].device_id == device_id

    @pytest.mark.asyncio
    async def test_concurrent_delete_nonexistent_memory(self, memory_store):
        """Test concurrent deletes of non-existent entities don't cause issues."""
        # Try to delete entities that don't exist
        tasks = []
        for i in range(20):
            tasks.append(memory_store.delete_message(f"nonexistent-msg-{i}"))
            tasks.append(memory_store.delete_device(f"nonexistent-dev-{i}"))
            tasks.append(memory_store.delete_rule(f"nonexistent-rule-{i}"))
            tasks.append(memory_store.delete_session(f"nonexistent-sess-{i}"))

        results = await asyncio.gather(*tasks)

        # All deletes should return False
        assert all(r is False for r in results)

        # Store should remain empty
        assert len(await memory_store.load_messages()) == 0
        assert len(await memory_store.load_devices()) == 0
        assert len(await memory_store.load_rules()) == 0
        assert len(await memory_store.load_sessions()) == 0

    @pytest.mark.asyncio
    async def test_concurrent_delete_nonexistent_file(self, file_store):
        """Test concurrent deletes of non-existent entities on FileGatewayStore."""
        tasks = []
        for i in range(15):
            tasks.append(file_store.delete_message(f"nonexistent-msg-{i}"))
            tasks.append(file_store.delete_device(f"nonexistent-dev-{i}"))
            tasks.append(file_store.delete_rule(f"nonexistent-rule-{i}"))
            tasks.append(file_store.delete_session(f"nonexistent-sess-{i}"))

        results = await asyncio.gather(*tasks)
        assert all(r is False for r in results)


# =============================================================================
# RedisGatewayStore Tests
# =============================================================================


class TestRedisGatewayStore:
    """Test RedisGatewayStore with mocked Redis client."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = AsyncMock()
        mock.pipeline.return_value = AsyncMock()
        mock.pipeline.return_value.execute = AsyncMock(return_value=[1, 1])
        return mock

    @pytest.fixture
    def store(self, mock_redis):
        """Create a RedisGatewayStore with mocked Redis."""
        store = RedisGatewayStore(
            redis_url="redis://localhost:6379",
            key_prefix="test:gateway:",
            message_ttl_seconds=3600,
            device_ttl_seconds=7200,
            session_ttl_seconds=1800,
        )
        store._redis = mock_redis
        return store

    # -------------------------------------------------------------------------
    # Import Error Handling
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_redis_import_error(self):
        """Test ImportError when redis module not installed."""
        import builtins

        store = RedisGatewayStore()
        store._redis = None

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "redis.asyncio":
                raise ImportError("No module named 'redis'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", mock_import):
            with pytest.raises(ImportError, match="redis-py with async support required"):
                await store._get_redis()

    @pytest.mark.asyncio
    async def test_get_redis_reuses_existing_client(self, store, mock_redis):
        """Test that _get_redis returns existing client."""
        result = await store._get_redis()
        assert result is mock_redis

    # -------------------------------------------------------------------------
    # Message Operations
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_save_message_creates_key_and_sorted_set(self, store, mock_redis):
        """Test save_message creates key and adds to sorted set."""
        msg = InboxMessage(
            message_id="m1",
            channel="slack",
            sender="alice",
            content="Hello",
            timestamp=1000.0,
        )

        await store.save_message(msg)

        pipe = mock_redis.pipeline.return_value
        pipe.set.assert_called_once()
        # Verify the key
        call_args = pipe.set.call_args
        assert call_args[0][0] == "test:gateway:msg:m1"
        # Verify TTL
        assert call_args[1]["ex"] == 3600

        pipe.zadd.assert_called_once()
        zadd_call = pipe.zadd.call_args
        assert zadd_call[0][0] == "test:gateway:msg:index"
        assert zadd_call[0][1] == {"m1": 1000.0}

        pipe.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_messages_retrieves_from_sorted_set(self, store, mock_redis):
        """Test load_messages retrieves from sorted set with proper ordering."""
        import json

        # Mock the sorted set response (newest first)
        mock_redis.zrevrange.return_value = [b"m2", b"m1"]

        # Mock the message data
        msg1_data = _message_to_dict(
            InboxMessage(
                message_id="m1",
                channel="slack",
                sender="alice",
                content="First",
                timestamp=100.0,
            )
        )
        msg2_data = _message_to_dict(
            InboxMessage(
                message_id="m2",
                channel="slack",
                sender="bob",
                content="Second",
                timestamp=200.0,
            )
        )
        mock_redis.mget.return_value = [json.dumps(msg2_data), json.dumps(msg1_data)]

        messages = await store.load_messages(limit=10)

        mock_redis.zrevrange.assert_called_once_with("test:gateway:msg:index", 0, 9)
        mock_redis.mget.assert_called_once()
        assert len(messages) == 2
        assert messages[0].message_id == "m2"
        assert messages[1].message_id == "m1"

    @pytest.mark.asyncio
    async def test_load_messages_empty_result(self, store, mock_redis):
        """Test load_messages returns empty list when no messages."""
        mock_redis.zrevrange.return_value = []

        messages = await store.load_messages()

        assert messages == []
        mock_redis.mget.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_messages_handles_bytes_keys(self, store, mock_redis):
        """Test load_messages handles bytes vs string keys from Redis."""
        import json

        # Redis returns bytes for keys
        mock_redis.zrevrange.return_value = [b"m1"]
        msg_data = _message_to_dict(
            InboxMessage(message_id="m1", channel="s", sender="a", content="test")
        )
        mock_redis.mget.return_value = [json.dumps(msg_data)]

        messages = await store.load_messages()

        # Verify key was decoded properly
        mget_call = mock_redis.mget.call_args[0][0]
        assert mget_call[0] == "test:gateway:msg:m1"
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_load_messages_skips_invalid_data(self, store, mock_redis):
        """Test load_messages skips entries with invalid JSON."""
        import json

        mock_redis.zrevrange.return_value = [b"m1", b"m2"]
        valid_msg = _message_to_dict(
            InboxMessage(message_id="m2", channel="s", sender="a", content="test")
        )
        mock_redis.mget.return_value = ["invalid json{{{", json.dumps(valid_msg)]

        messages = await store.load_messages()

        assert len(messages) == 1
        assert messages[0].message_id == "m2"

    @pytest.mark.asyncio
    async def test_delete_message_removes_key_and_from_set(self, store, mock_redis):
        """Test delete_message removes key and from sorted set."""
        pipe = mock_redis.pipeline.return_value
        pipe.execute.return_value = [1, 1]

        result = await store.delete_message("m1")

        assert result is True
        pipe.delete.assert_called_once_with("test:gateway:msg:m1")
        pipe.zrem.assert_called_once_with("test:gateway:msg:index", "m1")
        pipe.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_message_returns_false_when_not_found(self, store, mock_redis):
        """Test delete_message returns False when message not found."""
        pipe = mock_redis.pipeline.return_value
        pipe.execute.return_value = [0, 0]

        result = await store.delete_message("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_clear_messages_all(self, store, mock_redis):
        """Test clear_messages clears all messages."""
        mock_redis.zrange.return_value = [b"m1", b"m2", b"m3"]
        pipe = mock_redis.pipeline.return_value

        count = await store.clear_messages()

        assert count == 3
        mock_redis.zrange.assert_called_once_with("test:gateway:msg:index", 0, -1)
        pipe.delete.assert_called()
        # Verify all keys and index are deleted
        delete_calls = pipe.delete.call_args_list
        assert len(delete_calls) == 2  # One for keys, one for index

    @pytest.mark.asyncio
    async def test_clear_messages_empty(self, store, mock_redis):
        """Test clear_messages with no messages returns 0."""
        mock_redis.zrange.return_value = []

        count = await store.clear_messages()

        assert count == 0

    @pytest.mark.asyncio
    async def test_clear_messages_older_than(self, store, mock_redis):
        """Test clear_messages with older_than_seconds."""
        mock_redis.zrangebyscore.return_value = [b"m1", b"m2"]
        pipe = mock_redis.pipeline.return_value

        with patch("time.time", return_value=1000.0):
            count = await store.clear_messages(older_than_seconds=100)

        assert count == 2
        mock_redis.zrangebyscore.assert_called_once_with("test:gateway:msg:index", "-inf", 900.0)
        pipe.delete.assert_called()
        pipe.zremrangebyscore.assert_called_once_with("test:gateway:msg:index", "-inf", 900.0)

    @pytest.mark.asyncio
    async def test_clear_messages_older_than_none_found(self, store, mock_redis):
        """Test clear_messages with older_than returns 0 when none found."""
        mock_redis.zrangebyscore.return_value = []

        count = await store.clear_messages(older_than_seconds=100)

        assert count == 0

    # -------------------------------------------------------------------------
    # Device Operations
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_save_device_creates_key_and_set(self, store, mock_redis):
        """Test save_device creates key and adds to set index."""
        device = DeviceNode(
            device_id="d1",
            name="Laptop",
            device_type="laptop",
            status=DeviceStatus.ONLINE,
        )

        await store.save_device(device)

        pipe = mock_redis.pipeline.return_value
        pipe.set.assert_called_once()
        call_args = pipe.set.call_args
        assert call_args[0][0] == "test:gateway:dev:d1"
        assert call_args[1]["ex"] == 7200  # device_ttl_seconds

        pipe.sadd.assert_called_once_with("test:gateway:dev:index", "d1")
        pipe.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_devices_retrieves_all_from_set(self, store, mock_redis):
        """Test load_devices retrieves all from set."""
        import json

        mock_redis.smembers.return_value = {b"d1", b"d2"}
        dev1 = _device_to_dict(DeviceNode(device_id="d1", name="Laptop", device_type="laptop"))
        dev2 = _device_to_dict(DeviceNode(device_id="d2", name="Phone", device_type="mobile"))
        mock_redis.mget.return_value = [json.dumps(dev1), json.dumps(dev2)]

        devices = await store.load_devices()

        mock_redis.smembers.assert_called_once_with("test:gateway:dev:index")
        assert len(devices) == 2

    @pytest.mark.asyncio
    async def test_load_devices_empty(self, store, mock_redis):
        """Test load_devices returns empty list when no devices."""
        mock_redis.smembers.return_value = set()

        devices = await store.load_devices()

        assert devices == []
        mock_redis.mget.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_devices_handles_bytes(self, store, mock_redis):
        """Test load_devices handles bytes from Redis."""
        import json

        mock_redis.smembers.return_value = {b"d1"}
        dev = _device_to_dict(DeviceNode(device_id="d1", name="Laptop", device_type="laptop"))
        mock_redis.mget.return_value = [json.dumps(dev)]

        devices = await store.load_devices()

        mget_call = mock_redis.mget.call_args[0][0]
        assert mget_call[0] == "test:gateway:dev:d1"
        assert len(devices) == 1

    @pytest.mark.asyncio
    async def test_load_devices_skips_invalid(self, store, mock_redis):
        """Test load_devices skips invalid data."""
        import json

        mock_redis.smembers.return_value = {b"d1", b"d2"}
        valid_dev = _device_to_dict(DeviceNode(device_id="d2", name="Phone", device_type="mobile"))
        mock_redis.mget.return_value = ["not json", json.dumps(valid_dev)]

        devices = await store.load_devices()

        assert len(devices) == 1

    @pytest.mark.asyncio
    async def test_delete_device_removes_key_and_from_set(self, store, mock_redis):
        """Test delete_device removes key and from set."""
        pipe = mock_redis.pipeline.return_value
        pipe.execute.return_value = [1, 1]

        result = await store.delete_device("d1")

        assert result is True
        pipe.delete.assert_called_once_with("test:gateway:dev:d1")
        pipe.srem.assert_called_once_with("test:gateway:dev:index", "d1")

    @pytest.mark.asyncio
    async def test_delete_device_returns_false(self, store, mock_redis):
        """Test delete_device returns False when not found."""
        pipe = mock_redis.pipeline.return_value
        pipe.execute.return_value = [0, 0]

        result = await store.delete_device("nonexistent")

        assert result is False

    # -------------------------------------------------------------------------
    # Rule Operations
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_save_rule_creates_key_and_sorted_set_by_priority(self, store, mock_redis):
        """Test save_rule creates key and adds to sorted set by priority."""
        rule = RoutingRule(
            rule_id="r1",
            agent_id="claude",
            priority=10,
        )

        await store.save_rule(rule)

        pipe = mock_redis.pipeline.return_value
        pipe.set.assert_called_once()
        call_args = pipe.set.call_args
        assert call_args[0][0] == "test:gateway:rule:r1"
        # Rules don't have TTL set explicitly

        pipe.zadd.assert_called_once_with("test:gateway:rule:index", {"r1": 10})
        pipe.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_rules_sorted_by_priority_descending(self, store, mock_redis):
        """Test load_rules retrieves sorted by priority (descending)."""
        import json

        mock_redis.zrevrange.return_value = [b"r_high", b"r_low"]
        rule_high = _rule_to_dict(RoutingRule(rule_id="r_high", agent_id="gpt", priority=100))
        rule_low = _rule_to_dict(RoutingRule(rule_id="r_low", agent_id="claude", priority=1))
        mock_redis.mget.return_value = [json.dumps(rule_high), json.dumps(rule_low)]

        rules = await store.load_rules()

        mock_redis.zrevrange.assert_called_once_with("test:gateway:rule:index", 0, -1)
        assert len(rules) == 2
        assert rules[0].rule_id == "r_high"
        assert rules[0].priority == 100
        assert rules[1].rule_id == "r_low"

    @pytest.mark.asyncio
    async def test_load_rules_empty(self, store, mock_redis):
        """Test load_rules returns empty list when no rules."""
        mock_redis.zrevrange.return_value = []

        rules = await store.load_rules()

        assert rules == []

    @pytest.mark.asyncio
    async def test_load_rules_skips_invalid(self, store, mock_redis):
        """Test load_rules skips invalid data."""
        import json

        mock_redis.zrevrange.return_value = [b"r1", b"r2"]
        valid = _rule_to_dict(RoutingRule(rule_id="r2", agent_id="claude"))
        mock_redis.mget.return_value = ["{invalid", json.dumps(valid)]

        rules = await store.load_rules()

        assert len(rules) == 1
        assert rules[0].rule_id == "r2"

    @pytest.mark.asyncio
    async def test_delete_rule_removes_key_and_from_set(self, store, mock_redis):
        """Test delete_rule removes key and from sorted set."""
        pipe = mock_redis.pipeline.return_value
        pipe.execute.return_value = [1, 1]

        result = await store.delete_rule("r1")

        assert result is True
        pipe.delete.assert_called_once_with("test:gateway:rule:r1")
        pipe.zrem.assert_called_once_with("test:gateway:rule:index", "r1")

    @pytest.mark.asyncio
    async def test_delete_rule_returns_false(self, store, mock_redis):
        """Test delete_rule returns False when not found."""
        pipe = mock_redis.pipeline.return_value
        pipe.execute.return_value = [0, 0]

        result = await store.delete_rule("nonexistent")

        assert result is False

    # -------------------------------------------------------------------------
    # Session Operations
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_save_session_creates_key_with_ttl(self, store, mock_redis):
        """Test save_session creates key with TTL."""
        session = {
            "session_id": "s1",
            "user_id": "u1",
            "device_id": "d1",
            "status": "active",
            "last_seen": 1000.0,
        }

        await store.save_session(session)

        pipe = mock_redis.pipeline.return_value
        pipe.set.assert_called_once()
        call_args = pipe.set.call_args
        assert call_args[0][0] == "test:gateway:sess:s1"
        assert call_args[1]["ex"] == 1800  # session_ttl_seconds

        pipe.zadd.assert_called_once_with("test:gateway:sess:index", {"s1": 1000.0})

    @pytest.mark.asyncio
    async def test_save_session_without_session_id_no_op(self, store, mock_redis):
        """Test save_session does nothing without session_id."""
        session = {"user_id": "u1"}  # Missing session_id

        await store.save_session(session)

        mock_redis.pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_session_uses_created_at_fallback(self, store, mock_redis):
        """Test save_session uses created_at if last_seen missing."""
        session = {
            "session_id": "s1",
            "created_at": 500.0,
        }

        await store.save_session(session)

        pipe = mock_redis.pipeline.return_value
        pipe.zadd.assert_called_once_with("test:gateway:sess:index", {"s1": 500.0})

    @pytest.mark.asyncio
    async def test_load_sessions_sorted_by_last_seen(self, store, mock_redis):
        """Test load_sessions retrieves sorted by last_seen."""
        import json

        mock_redis.zrevrange.return_value = [b"s2", b"s1"]
        sess1 = {"session_id": "s1", "last_seen": 100.0}
        sess2 = {"session_id": "s2", "last_seen": 200.0}
        mock_redis.mget.return_value = [json.dumps(sess2), json.dumps(sess1)]

        sessions = await store.load_sessions(limit=10)

        mock_redis.zrevrange.assert_called_once_with("test:gateway:sess:index", 0, 9)
        assert len(sessions) == 2
        assert sessions[0]["session_id"] == "s2"

    @pytest.mark.asyncio
    async def test_load_sessions_empty(self, store, mock_redis):
        """Test load_sessions returns empty list when no sessions."""
        mock_redis.zrevrange.return_value = []

        sessions = await store.load_sessions()

        assert sessions == []

    @pytest.mark.asyncio
    async def test_load_sessions_handles_bytes(self, store, mock_redis):
        """Test load_sessions handles bytes from Redis."""
        import json

        mock_redis.zrevrange.return_value = [b"s1"]
        sess = {"session_id": "s1", "status": "active"}
        mock_redis.mget.return_value = [json.dumps(sess)]

        sessions = await store.load_sessions()

        mget_call = mock_redis.mget.call_args[0][0]
        assert mget_call[0] == "test:gateway:sess:s1"
        assert len(sessions) == 1

    @pytest.mark.asyncio
    async def test_load_sessions_skips_invalid(self, store, mock_redis):
        """Test load_sessions skips invalid data."""
        import json

        mock_redis.zrevrange.return_value = [b"s1", b"s2"]
        valid = {"session_id": "s2", "status": "active"}
        mock_redis.mget.return_value = ["not json{{", json.dumps(valid)]

        sessions = await store.load_sessions()

        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "s2"

    @pytest.mark.asyncio
    async def test_load_sessions_skips_non_dict(self, store, mock_redis):
        """Test load_sessions skips non-dict JSON."""
        import json

        mock_redis.zrevrange.return_value = [b"s1", b"s2"]
        valid = {"session_id": "s2"}
        mock_redis.mget.return_value = [json.dumps([1, 2, 3]), json.dumps(valid)]

        sessions = await store.load_sessions()

        assert len(sessions) == 1

    @pytest.mark.asyncio
    async def test_delete_session_removes_key_and_from_set(self, store, mock_redis):
        """Test delete_session removes key and from sorted set."""
        pipe = mock_redis.pipeline.return_value
        pipe.execute.return_value = [1, 1]

        result = await store.delete_session("s1")

        assert result is True
        pipe.delete.assert_called_once_with("test:gateway:sess:s1")
        pipe.zrem.assert_called_once_with("test:gateway:sess:index", "s1")

    @pytest.mark.asyncio
    async def test_delete_session_returns_false(self, store, mock_redis):
        """Test delete_session returns False when not found."""
        pipe = mock_redis.pipeline.return_value
        pipe.execute.return_value = [0, 0]

        result = await store.delete_session("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_clear_sessions_all(self, store, mock_redis):
        """Test clear_sessions clears all sessions."""
        mock_redis.zrange.return_value = [b"s1", b"s2"]
        pipe = mock_redis.pipeline.return_value

        count = await store.clear_sessions()

        assert count == 2
        mock_redis.zrange.assert_called_once_with("test:gateway:sess:index", 0, -1)
        pipe.delete.assert_called()

    @pytest.mark.asyncio
    async def test_clear_sessions_empty(self, store, mock_redis):
        """Test clear_sessions with no sessions."""
        mock_redis.zrange.return_value = []

        count = await store.clear_sessions()

        assert count == 0

    @pytest.mark.asyncio
    async def test_clear_sessions_older_than(self, store, mock_redis):
        """Test clear_sessions with older_than_seconds."""
        mock_redis.zrangebyscore.return_value = [b"s1"]
        pipe = mock_redis.pipeline.return_value

        with patch("time.time", return_value=1000.0):
            count = await store.clear_sessions(older_than_seconds=200)

        assert count == 1
        mock_redis.zrangebyscore.assert_called_once_with("test:gateway:sess:index", "-inf", 800.0)
        pipe.zremrangebyscore.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_sessions_older_than_none_found(self, store, mock_redis):
        """Test clear_sessions older_than returns 0 when none found."""
        mock_redis.zrangebyscore.return_value = []

        count = await store.clear_sessions(older_than_seconds=100)

        assert count == 0

    # -------------------------------------------------------------------------
    # Pipeline Execution and TTL Configuration
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_pipeline_execution(self, store, mock_redis):
        """Test that pipeline is used for atomic operations."""
        msg = InboxMessage(message_id="m1", channel="s", sender="a", content="test")

        await store.save_message(msg)

        mock_redis.pipeline.assert_called_once()
        pipe = mock_redis.pipeline.return_value
        pipe.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_ttl_configuration(self):
        """Test TTL configuration is applied correctly."""
        store = RedisGatewayStore(
            redis_url="redis://localhost",
            message_ttl_seconds=100,
            device_ttl_seconds=200,
            session_ttl_seconds=300,
        )

        assert store._message_ttl == 100
        assert store._device_ttl == 200
        assert store._session_ttl == 300

    @pytest.mark.asyncio
    async def test_key_prefix_configuration(self):
        """Test key prefix configuration."""
        store = RedisGatewayStore(
            redis_url="redis://localhost",
            key_prefix="custom:prefix:",
        )

        assert store._msg_key("m1") == "custom:prefix:msg:m1"
        assert store._dev_key("d1") == "custom:prefix:dev:d1"
        assert store._rule_key("r1") == "custom:prefix:rule:r1"
        assert store._session_key("s1") == "custom:prefix:sess:s1"
        assert store._msg_index_key() == "custom:prefix:msg:index"
        assert store._dev_index_key() == "custom:prefix:dev:index"
        assert store._rule_index_key() == "custom:prefix:rule:index"
        assert store._session_index_key() == "custom:prefix:sess:index"

    # -------------------------------------------------------------------------
    # Close / Cleanup
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_close_closes_redis_connection(self, store, mock_redis):
        """Test close properly closes Redis connection."""
        await store.close()

        mock_redis.close.assert_called_once()
        assert store._redis is None

    @pytest.mark.asyncio
    async def test_close_no_op_when_no_connection(self):
        """Test close is safe when no connection exists."""
        store = RedisGatewayStore()
        store._redis = None

        # Should not raise
        await store.close()

    # -------------------------------------------------------------------------
    # Edge Cases
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_load_messages_with_null_data(self, store, mock_redis):
        """Test load_messages handles null data in mget response."""
        import json

        mock_redis.zrevrange.return_value = [b"m1", b"m2"]
        valid_msg = _message_to_dict(
            InboxMessage(message_id="m2", channel="s", sender="a", content="test")
        )
        mock_redis.mget.return_value = [None, json.dumps(valid_msg)]

        messages = await store.load_messages()

        assert len(messages) == 1
        assert messages[0].message_id == "m2"

    @pytest.mark.asyncio
    async def test_string_keys_from_redis(self, store, mock_redis):
        """Test handling when Redis returns strings instead of bytes."""
        import json

        # Some Redis clients may return strings
        mock_redis.zrevrange.return_value = ["m1", "m2"]
        msg1 = _message_to_dict(
            InboxMessage(message_id="m1", channel="s", sender="a", content="test1")
        )
        msg2 = _message_to_dict(
            InboxMessage(message_id="m2", channel="s", sender="a", content="test2")
        )
        mock_redis.mget.return_value = [json.dumps(msg1), json.dumps(msg2)]

        messages = await store.load_messages()

        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_default_redis_url(self):
        """Test default Redis URL."""
        store = RedisGatewayStore()
        assert store._redis_url == "redis://localhost:6379"

    @pytest.mark.asyncio
    async def test_default_key_prefix(self):
        """Test default key prefix."""
        store = RedisGatewayStore()
        assert store._key_prefix == "aragora:gateway:"

    @pytest.mark.asyncio
    async def test_default_ttl_values(self):
        """Test default TTL values."""
        store = RedisGatewayStore()
        assert store._message_ttl == 86400 * 7  # 7 days
        assert store._device_ttl == 86400 * 30  # 30 days
        assert store._session_ttl == 86400  # 1 day

    @pytest.mark.asyncio
    async def test_load_devices_handles_none_data(self, store, mock_redis):
        """Test load_devices handles None data in mget response."""
        import json

        mock_redis.smembers.return_value = {b"d1", b"d2"}
        valid_dev = _device_to_dict(DeviceNode(device_id="d2", name="Phone", device_type="mobile"))
        mock_redis.mget.return_value = [None, json.dumps(valid_dev)]

        devices = await store.load_devices()

        assert len(devices) == 1
        assert devices[0].device_id == "d2"

    @pytest.mark.asyncio
    async def test_load_rules_handles_none_data(self, store, mock_redis):
        """Test load_rules handles None data in mget response."""
        import json

        mock_redis.zrevrange.return_value = [b"r1", b"r2"]
        valid_rule = _rule_to_dict(RoutingRule(rule_id="r2", agent_id="claude"))
        mock_redis.mget.return_value = [None, json.dumps(valid_rule)]

        rules = await store.load_rules()

        assert len(rules) == 1
        assert rules[0].rule_id == "r2"

    @pytest.mark.asyncio
    async def test_load_sessions_handles_none_data(self, store, mock_redis):
        """Test load_sessions handles None data in mget response."""
        import json

        mock_redis.zrevrange.return_value = [b"s1", b"s2"]
        valid_sess = {"session_id": "s2", "status": "active"}
        mock_redis.mget.return_value = [None, json.dumps(valid_sess)]

        sessions = await store.load_sessions()

        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "s2"

    @pytest.mark.asyncio
    async def test_save_session_uses_time_fallback(self, store, mock_redis):
        """Test save_session uses current time if no timestamps."""
        session = {
            "session_id": "s1",
            "user_id": "u1",
        }

        with patch("time.time", return_value=999.0):
            await store.save_session(session)

        pipe = mock_redis.pipeline.return_value
        pipe.zadd.assert_called_once_with("test:gateway:sess:index", {"s1": 999.0})
