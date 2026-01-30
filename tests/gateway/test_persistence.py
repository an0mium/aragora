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
# RedisGatewayStore Tests (Mocked)
# =============================================================================


class TestRedisGatewayStore:
    """Test RedisGatewayStore with mocked Redis client."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client with async support."""
        mock = MagicMock()
        mock.pipeline = MagicMock(return_value=MagicMock())
        mock.pipeline.return_value.execute = AsyncMock(return_value=[1, 1])
        mock.set = AsyncMock()
        mock.get = AsyncMock()
        mock.delete = AsyncMock(return_value=1)
        mock.mget = AsyncMock(return_value=[])
        mock.zadd = AsyncMock()
        mock.zrevrange = AsyncMock(return_value=[])
        mock.zrange = AsyncMock(return_value=[])
        mock.zrangebyscore = AsyncMock(return_value=[])
        mock.zrem = AsyncMock()
        mock.zremrangebyscore = AsyncMock()
        mock.sadd = AsyncMock()
        mock.smembers = AsyncMock(return_value=set())
        mock.srem = AsyncMock()
        mock.close = AsyncMock()
        return mock

    @pytest.fixture
    def store(self, mock_redis):
        """Create RedisGatewayStore with mocked Redis."""
        store = RedisGatewayStore(
            redis_url="redis://localhost:6379",
            key_prefix="test:gateway:",
            message_ttl_seconds=3600,
            device_ttl_seconds=7200,
            session_ttl_seconds=1800,
        )
        store._redis = mock_redis
        return store

    # Message tests

    @pytest.mark.asyncio
    async def test_save_message(self, store, mock_redis):
        """Test save_message creates key and adds to sorted set."""
        msg = InboxMessage(
            message_id="m1",
            channel="slack",
            sender="alice",
            content="hello",
            timestamp=1000.0,
        )

        # Setup pipeline mock
        pipeline = MagicMock()
        pipeline.set = MagicMock()
        pipeline.zadd = MagicMock()
        pipeline.execute = AsyncMock(return_value=[True, 1])
        mock_redis.pipeline.return_value = pipeline

        await store.save_message(msg)

        # Verify pipeline operations
        pipeline.set.assert_called_once()
        call_args = pipeline.set.call_args
        assert "test:gateway:msg:m1" in call_args[0]
        assert call_args[1]["ex"] == 3600  # message_ttl_seconds

        pipeline.zadd.assert_called_once()
        zadd_call = pipeline.zadd.call_args
        assert "test:gateway:msg:index" in zadd_call[0]
        assert zadd_call[0][1] == {"m1": 1000.0}

    @pytest.mark.asyncio
    async def test_load_messages(self, store, mock_redis):
        """Test load_messages retrieves from sorted set."""
        import json

        # Setup mock returns
        mock_redis.zrevrange = AsyncMock(return_value=[b"m1", b"m2"])
        mock_redis.mget = AsyncMock(
            return_value=[
                json.dumps(
                    {
                        "message_id": "m1",
                        "channel": "slack",
                        "sender": "alice",
                        "content": "hello",
                        "timestamp": 2000.0,
                    }
                ),
                json.dumps(
                    {
                        "message_id": "m2",
                        "channel": "slack",
                        "sender": "bob",
                        "content": "hi",
                        "timestamp": 1000.0,
                    }
                ),
            ]
        )

        messages = await store.load_messages(limit=10)

        assert len(messages) == 2
        assert messages[0].message_id == "m1"
        assert messages[1].message_id == "m2"
        mock_redis.zrevrange.assert_called_once_with("test:gateway:msg:index", 0, 9)

    @pytest.mark.asyncio
    async def test_load_messages_empty(self, store, mock_redis):
        """Test load_messages with empty result set."""
        mock_redis.zrevrange = AsyncMock(return_value=[])

        messages = await store.load_messages()

        assert messages == []

    @pytest.mark.asyncio
    async def test_load_messages_with_bytes_keys(self, store, mock_redis):
        """Test load_messages handles bytes keys from Redis."""
        import json

        mock_redis.zrevrange = AsyncMock(return_value=[b"msg1", "msg2"])  # Mixed bytes and str
        mock_redis.mget = AsyncMock(
            return_value=[
                json.dumps(
                    {
                        "message_id": "msg1",
                        "channel": "slack",
                        "sender": "a",
                        "content": "test1",
                    }
                ),
                json.dumps(
                    {
                        "message_id": "msg2",
                        "channel": "slack",
                        "sender": "b",
                        "content": "test2",
                    }
                ),
            ]
        )

        messages = await store.load_messages()

        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_delete_message(self, store, mock_redis):
        """Test delete_message removes key and from sorted set."""
        pipeline = MagicMock()
        pipeline.delete = MagicMock()
        pipeline.zrem = MagicMock()
        pipeline.execute = AsyncMock(return_value=[1, 1])
        mock_redis.pipeline.return_value = pipeline

        result = await store.delete_message("m1")

        assert result is True
        pipeline.delete.assert_called_once_with("test:gateway:msg:m1")
        pipeline.zrem.assert_called_once_with("test:gateway:msg:index", "m1")

    @pytest.mark.asyncio
    async def test_delete_message_not_found(self, store, mock_redis):
        """Test delete_message returns False when message doesn't exist."""
        pipeline = MagicMock()
        pipeline.delete = MagicMock()
        pipeline.zrem = MagicMock()
        pipeline.execute = AsyncMock(return_value=[0, 0])
        mock_redis.pipeline.return_value = pipeline

        result = await store.delete_message("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_clear_messages_all(self, store, mock_redis):
        """Test clear_messages removes all messages."""
        mock_redis.zrange = AsyncMock(return_value=[b"m1", b"m2", b"m3"])

        pipeline = MagicMock()
        pipeline.delete = MagicMock()
        pipeline.execute = AsyncMock(return_value=[3, 1])
        mock_redis.pipeline.return_value = pipeline

        count = await store.clear_messages()

        assert count == 3

    @pytest.mark.asyncio
    async def test_clear_messages_older_than(self, store, mock_redis):
        """Test clear_messages with age threshold."""
        import time

        mock_redis.zrangebyscore = AsyncMock(return_value=[b"old1", b"old2"])

        pipeline = MagicMock()
        pipeline.delete = MagicMock()
        pipeline.zremrangebyscore = MagicMock()
        pipeline.execute = AsyncMock(return_value=[2, 2])
        mock_redis.pipeline.return_value = pipeline

        count = await store.clear_messages(older_than_seconds=3600)

        assert count == 2
        # Verify zrangebyscore was called with cutoff time
        call_args = mock_redis.zrangebyscore.call_args[0]
        assert call_args[0] == "test:gateway:msg:index"
        assert call_args[1] == "-inf"

    # Device tests

    @pytest.mark.asyncio
    async def test_save_device(self, store, mock_redis):
        """Test save_device creates key and adds to set index."""
        device = DeviceNode(
            device_id="d1",
            name="Laptop",
            device_type="laptop",
            status=DeviceStatus.ONLINE,
        )

        pipeline = MagicMock()
        pipeline.set = MagicMock()
        pipeline.sadd = MagicMock()
        pipeline.execute = AsyncMock(return_value=[True, 1])
        mock_redis.pipeline.return_value = pipeline

        await store.save_device(device)

        pipeline.set.assert_called_once()
        call_args = pipeline.set.call_args
        assert "test:gateway:dev:d1" in call_args[0]
        assert call_args[1]["ex"] == 7200  # device_ttl_seconds

        pipeline.sadd.assert_called_once()
        sadd_call = pipeline.sadd.call_args
        assert "test:gateway:dev:index" in sadd_call[0]
        assert "d1" in sadd_call[0]

    @pytest.mark.asyncio
    async def test_load_devices(self, store, mock_redis):
        """Test load_devices retrieves from set."""
        import json

        mock_redis.smembers = AsyncMock(return_value={b"d1", b"d2"})
        mock_redis.mget = AsyncMock(
            return_value=[
                json.dumps(
                    {
                        "device_id": "d1",
                        "name": "Laptop",
                        "device_type": "laptop",
                        "status": "online",
                    }
                ),
                json.dumps(
                    {
                        "device_id": "d2",
                        "name": "Phone",
                        "device_type": "phone",
                        "status": "offline",
                    }
                ),
            ]
        )

        devices = await store.load_devices()

        assert len(devices) == 2

    @pytest.mark.asyncio
    async def test_load_devices_empty(self, store, mock_redis):
        """Test load_devices with empty result."""
        mock_redis.smembers = AsyncMock(return_value=set())

        devices = await store.load_devices()

        assert devices == []

    @pytest.mark.asyncio
    async def test_delete_device(self, store, mock_redis):
        """Test delete_device removes key and from set."""
        pipeline = MagicMock()
        pipeline.delete = MagicMock()
        pipeline.srem = MagicMock()
        pipeline.execute = AsyncMock(return_value=[1, 1])
        mock_redis.pipeline.return_value = pipeline

        result = await store.delete_device("d1")

        assert result is True
        pipeline.delete.assert_called_once_with("test:gateway:dev:d1")
        pipeline.srem.assert_called_once_with("test:gateway:dev:index", "d1")

    # Rule tests

    @pytest.mark.asyncio
    async def test_save_rule(self, store, mock_redis):
        """Test save_rule creates key and adds to sorted set by priority."""
        rule = RoutingRule(
            rule_id="r1",
            agent_id="claude",
            priority=10,
        )

        pipeline = MagicMock()
        pipeline.set = MagicMock()
        pipeline.zadd = MagicMock()
        pipeline.execute = AsyncMock(return_value=[True, 1])
        mock_redis.pipeline.return_value = pipeline

        await store.save_rule(rule)

        pipeline.set.assert_called_once()
        pipeline.zadd.assert_called_once()
        zadd_call = pipeline.zadd.call_args
        assert "test:gateway:rule:index" in zadd_call[0]
        assert zadd_call[0][1] == {"r1": 10}  # priority as score

    @pytest.mark.asyncio
    async def test_load_rules(self, store, mock_redis):
        """Test load_rules retrieves sorted by priority (descending)."""
        import json

        mock_redis.zrevrange = AsyncMock(return_value=[b"r1", b"r2"])
        mock_redis.mget = AsyncMock(
            return_value=[
                json.dumps(
                    {
                        "rule_id": "r1",
                        "agent_id": "claude",
                        "priority": 10,
                    }
                ),
                json.dumps(
                    {
                        "rule_id": "r2",
                        "agent_id": "gpt",
                        "priority": 5,
                    }
                ),
            ]
        )

        rules = await store.load_rules()

        assert len(rules) == 2
        assert rules[0].rule_id == "r1"
        assert rules[0].priority == 10
        mock_redis.zrevrange.assert_called_once_with("test:gateway:rule:index", 0, -1)

    @pytest.mark.asyncio
    async def test_delete_rule(self, store, mock_redis):
        """Test delete_rule removes key and from sorted set."""
        pipeline = MagicMock()
        pipeline.delete = MagicMock()
        pipeline.zrem = MagicMock()
        pipeline.execute = AsyncMock(return_value=[1, 1])
        mock_redis.pipeline.return_value = pipeline

        result = await store.delete_rule("r1")

        assert result is True
        pipeline.delete.assert_called_once_with("test:gateway:rule:r1")
        pipeline.zrem.assert_called_once_with("test:gateway:rule:index", "r1")

    # Session tests

    @pytest.mark.asyncio
    async def test_save_session(self, store, mock_redis):
        """Test save_session creates key with TTL."""
        session = {
            "session_id": "s1",
            "user_id": "u1",
            "last_seen": 1000.0,
        }

        pipeline = MagicMock()
        pipeline.set = MagicMock()
        pipeline.zadd = MagicMock()
        pipeline.execute = AsyncMock(return_value=[True, 1])
        mock_redis.pipeline.return_value = pipeline

        await store.save_session(session)

        pipeline.set.assert_called_once()
        call_args = pipeline.set.call_args
        assert "test:gateway:sess:s1" in call_args[0]
        assert call_args[1]["ex"] == 1800  # session_ttl_seconds

        pipeline.zadd.assert_called_once()
        zadd_call = pipeline.zadd.call_args
        assert zadd_call[0][1] == {"s1": 1000.0}  # last_seen as score

    @pytest.mark.asyncio
    async def test_save_session_no_session_id(self, store, mock_redis):
        """Test save_session with missing session_id is a no-op."""
        session = {"user_id": "u1", "last_seen": 100.0}

        await store.save_session(session)

        # Pipeline should not be created
        mock_redis.pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_sessions(self, store, mock_redis):
        """Test load_sessions retrieves sorted by last_seen."""
        import json

        mock_redis.zrevrange = AsyncMock(return_value=[b"s1", b"s2"])
        mock_redis.mget = AsyncMock(
            return_value=[
                json.dumps(
                    {
                        "session_id": "s1",
                        "user_id": "u1",
                        "last_seen": 2000.0,
                    }
                ),
                json.dumps(
                    {
                        "session_id": "s2",
                        "user_id": "u2",
                        "last_seen": 1000.0,
                    }
                ),
            ]
        )

        sessions = await store.load_sessions(limit=10)

        assert len(sessions) == 2
        assert sessions[0]["session_id"] == "s1"
        mock_redis.zrevrange.assert_called_once_with("test:gateway:sess:index", 0, 9)

    @pytest.mark.asyncio
    async def test_delete_session(self, store, mock_redis):
        """Test delete_session removes key and from sorted set."""
        pipeline = MagicMock()
        pipeline.delete = MagicMock()
        pipeline.zrem = MagicMock()
        pipeline.execute = AsyncMock(return_value=[1, 1])
        mock_redis.pipeline.return_value = pipeline

        result = await store.delete_session("s1")

        assert result is True
        pipeline.delete.assert_called_once_with("test:gateway:sess:s1")
        pipeline.zrem.assert_called_once_with("test:gateway:sess:index", "s1")

    @pytest.mark.asyncio
    async def test_clear_sessions_all(self, store, mock_redis):
        """Test clear_sessions removes all sessions."""
        mock_redis.zrange = AsyncMock(return_value=[b"s1", b"s2"])

        pipeline = MagicMock()
        pipeline.delete = MagicMock()
        pipeline.execute = AsyncMock(return_value=[2, 1])
        mock_redis.pipeline.return_value = pipeline

        count = await store.clear_sessions()

        assert count == 2

    @pytest.mark.asyncio
    async def test_clear_sessions_older_than(self, store, mock_redis):
        """Test clear_sessions with age threshold."""
        mock_redis.zrangebyscore = AsyncMock(return_value=[b"old_s1"])

        pipeline = MagicMock()
        pipeline.delete = MagicMock()
        pipeline.zremrangebyscore = MagicMock()
        pipeline.execute = AsyncMock(return_value=[1, 1])
        mock_redis.pipeline.return_value = pipeline

        count = await store.clear_sessions(older_than_seconds=3600)

        assert count == 1

    @pytest.mark.asyncio
    async def test_close(self, store, mock_redis):
        """Test close releases Redis connection."""
        await store.close()

        mock_redis.close.assert_called_once()
        assert store._redis is None

    # Edge cases

    @pytest.mark.asyncio
    async def test_malformed_message_in_redis(self, store, mock_redis):
        """Test load_messages handles malformed data gracefully."""
        mock_redis.zrevrange = AsyncMock(return_value=[b"m1", b"m2"])
        mock_redis.mget = AsyncMock(
            return_value=[
                "not valid json",
                '{"message_id": "m2", "channel": "s", "sender": "a", "content": "hi"}',
            ]
        )

        messages = await store.load_messages()

        assert len(messages) == 1
        assert messages[0].message_id == "m2"

    @pytest.mark.asyncio
    async def test_none_values_in_mget(self, store, mock_redis):
        """Test load handles None values from mget (expired keys)."""
        import json

        mock_redis.zrevrange = AsyncMock(return_value=[b"m1", b"m2", b"m3"])
        mock_redis.mget = AsyncMock(
            return_value=[
                json.dumps({"message_id": "m1", "channel": "s", "sender": "a", "content": "1"}),
                None,  # Expired or deleted
                json.dumps({"message_id": "m3", "channel": "s", "sender": "a", "content": "3"}),
            ]
        )

        messages = await store.load_messages()

        assert len(messages) == 2
        assert messages[0].message_id == "m1"
        assert messages[1].message_id == "m3"


class TestRedisGatewayStoreImport:
    """Test Redis import handling."""

    @pytest.mark.asyncio
    async def test_redis_import_error(self):
        """Test that ImportError is raised when redis is not installed."""
        store = RedisGatewayStore()
        store._redis = None

        with patch.dict("sys.modules", {"redis.asyncio": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'redis'")):
                with pytest.raises(ImportError, match="redis-py with async support required"):
                    await store._get_redis()


class TestGetGatewayStoreEdgeCases:
    """Test get_gateway_store factory edge cases."""

    def test_auto_backend_with_redis_env_var(self):
        """Test auto backend uses Redis when REDIS_URL env var is set."""
        with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}):
            # Mock the redis import to succeed
            with patch("aragora.gateway.persistence.aioredis", create=True):
                # Note: Don't pass path as it's not compatible with RedisGatewayStore
                store = get_gateway_store("auto")
                assert isinstance(store, RedisGatewayStore)

    def test_auto_backend_redis_import_error_fallback(self):
        """Test auto backend falls back to file when redis import fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}):
                # Mock the redis import to fail
                with patch.dict("sys.modules", {"redis.asyncio": None}):
                    store = get_gateway_store("auto", path=Path(tmpdir) / "test.json")
                    assert isinstance(store, FileGatewayStore)

    def test_redis_backend_explicit(self):
        """Test explicit redis backend creates RedisGatewayStore."""
        store = get_gateway_store(
            "redis",
            redis_url="redis://localhost:6379",
            key_prefix="custom:",
        )
        assert isinstance(store, RedisGatewayStore)
        assert store._key_prefix == "custom:"

    def test_file_backend_creates_directory(self):
        """Test file backend handles nested paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "deep" / "nested" / "gateway.json"
            store = get_gateway_store("file", path=nested_path)
            assert isinstance(store, FileGatewayStore)
            assert store._path == nested_path


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestPersistenceConcurrency:
    """Test concurrent operations for gateway stores."""

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

    # Concurrent message tests

    @pytest.mark.asyncio
    async def test_concurrent_message_saves_memory(self, memory_store):
        """Test multiple concurrent save_message calls (memory store)."""

        async def save_msg(i: int):
            msg = InboxMessage(
                message_id=f"m{i}",
                channel="slack",
                sender=f"user{i}",
                content=f"content{i}",
            )
            await memory_store.save_message(msg)

        # Run 50 concurrent saves
        await asyncio.gather(*[save_msg(i) for i in range(50)])

        messages = await memory_store.load_messages(limit=100)
        assert len(messages) == 50

    @pytest.mark.asyncio
    async def test_concurrent_message_saves_file(self, file_store):
        """Test multiple concurrent save_message calls (file store)."""

        async def save_msg(i: int):
            msg = InboxMessage(
                message_id=f"m{i}",
                channel="slack",
                sender=f"user{i}",
                content=f"content{i}",
            )
            await file_store.save_message(msg)

        # Run 30 concurrent saves
        await asyncio.gather(*[save_msg(i) for i in range(30)])

        messages = await file_store.load_messages(limit=100)
        assert len(messages) == 30

    # Concurrent device tests

    @pytest.mark.asyncio
    async def test_concurrent_device_saves_memory(self, memory_store):
        """Test multiple concurrent save_device calls (memory store)."""

        async def save_dev(i: int):
            device = DeviceNode(
                device_id=f"d{i}",
                name=f"Device{i}",
                device_type="laptop",
            )
            await memory_store.save_device(device)

        await asyncio.gather(*[save_dev(i) for i in range(50)])

        devices = await memory_store.load_devices()
        assert len(devices) == 50

    @pytest.mark.asyncio
    async def test_concurrent_save_delete_device_memory(self, memory_store):
        """Test concurrent save and delete operations (memory store)."""
        # First save some devices
        for i in range(20):
            device = DeviceNode(device_id=f"d{i}", name=f"Device{i}", device_type="laptop")
            await memory_store.save_device(device)

        # Concurrently delete odd devices and save new ones
        async def delete_odd(i: int):
            if i % 2 == 1:
                await memory_store.delete_device(f"d{i}")

        async def save_new(i: int):
            device = DeviceNode(device_id=f"new{i}", name=f"New{i}", device_type="phone")
            await memory_store.save_device(device)

        await asyncio.gather(
            *[delete_odd(i) for i in range(20)],
            *[save_new(i) for i in range(10)],
        )

        devices = await memory_store.load_devices()
        # 10 original even devices + 10 new devices = 20
        assert len(devices) == 20

    # Concurrent rule tests

    @pytest.mark.asyncio
    async def test_concurrent_rule_saves_memory(self, memory_store):
        """Test multiple concurrent save_rule calls (memory store)."""

        async def save_rule(i: int):
            rule = RoutingRule(
                rule_id=f"r{i}",
                agent_id=f"agent{i}",
                priority=i,
            )
            await memory_store.save_rule(rule)

        await asyncio.gather(*[save_rule(i) for i in range(50)])

        rules = await memory_store.load_rules()
        assert len(rules) == 50
        # Verify priority ordering (highest first)
        assert rules[0].priority == 49

    # Concurrent session tests

    @pytest.mark.asyncio
    async def test_concurrent_session_saves_memory(self, memory_store):
        """Test multiple concurrent save_session calls (memory store)."""

        async def save_session(i: int):
            session = {
                "session_id": f"s{i}",
                "user_id": f"u{i}",
                "last_seen": float(i),
            }
            await memory_store.save_session(session)

        await asyncio.gather(*[save_session(i) for i in range(50)])

        sessions = await memory_store.load_sessions(limit=100)
        assert len(sessions) == 50

    @pytest.mark.asyncio
    async def test_concurrent_clear_while_saving_memory(self, memory_store):
        """Test concurrent clear_sessions while saving new sessions."""
        import time

        now = time.time()

        # Pre-populate with old sessions
        for i in range(10):
            await memory_store.save_session(
                {
                    "session_id": f"old{i}",
                    "user_id": f"old_u{i}",
                    "last_seen": now - 7200,  # 2 hours ago
                }
            )

        # Concurrently clear old sessions and add new ones
        async def clear_old():
            await memory_store.clear_sessions(older_than_seconds=3600)

        async def save_new(i: int):
            await memory_store.save_session(
                {
                    "session_id": f"new{i}",
                    "user_id": f"new_u{i}",
                    "last_seen": now,
                }
            )

        await asyncio.gather(
            clear_old(),
            *[save_new(i) for i in range(10)],
        )

        sessions = await memory_store.load_sessions()
        # All sessions should be new ones (old ones cleared)
        for sess in sessions:
            assert sess["session_id"].startswith("new")

    # Mixed operations

    @pytest.mark.asyncio
    async def test_mixed_operations_memory(self, memory_store):
        """Test concurrent operations across different entity types."""

        async def save_msg(i: int):
            msg = InboxMessage(message_id=f"m{i}", channel="s", sender="a", content=str(i))
            await memory_store.save_message(msg)

        async def save_dev(i: int):
            device = DeviceNode(device_id=f"d{i}", name=f"D{i}", device_type="laptop")
            await memory_store.save_device(device)

        async def save_rule(i: int):
            rule = RoutingRule(rule_id=f"r{i}", agent_id=f"a{i}", priority=i)
            await memory_store.save_rule(rule)

        async def save_sess(i: int):
            session = {"session_id": f"s{i}", "user_id": f"u{i}", "last_seen": float(i)}
            await memory_store.save_session(session)

        # Run all operations concurrently
        await asyncio.gather(
            *[save_msg(i) for i in range(20)],
            *[save_dev(i) for i in range(20)],
            *[save_rule(i) for i in range(20)],
            *[save_sess(i) for i in range(20)],
        )

        # Verify all entities saved correctly
        assert len(await memory_store.load_messages(limit=100)) == 20
        assert len(await memory_store.load_devices()) == 20
        assert len(await memory_store.load_rules()) == 20
        assert len(await memory_store.load_sessions(limit=100)) == 20

    @pytest.mark.asyncio
    async def test_rapid_updates_same_entity_memory(self, memory_store):
        """Test rapid concurrent updates to the same entity."""

        # Rapidly update the same device 50 times
        async def update_device(version: int):
            device = DeviceNode(
                device_id="same_device",
                name=f"Version{version}",
                device_type="laptop",
            )
            await memory_store.save_device(device)

        await asyncio.gather(*[update_device(i) for i in range(50)])

        devices = await memory_store.load_devices()
        # Should only have one device (last update wins)
        assert len(devices) == 1
        assert devices[0].device_id == "same_device"


# =============================================================================
# FileGatewayStore Edge Cases
# =============================================================================


class TestFileGatewayStoreEdgeCases:
    """Test FileGatewayStore edge cases."""

    @pytest.fixture
    def temp_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "gateway.json"

    @pytest.mark.asyncio
    async def test_malformed_message_in_file(self, temp_path):
        """Test partial load continues with malformed messages."""
        import json

        # Create file with mixed valid/invalid data
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "messages": [
                {"message_id": "m1", "channel": "s", "sender": "a", "content": "valid"},
                {"invalid": "message"},  # Missing required fields
                {"message_id": "m3", "channel": "s", "sender": "a", "content": "also_valid"},
            ],
            "devices": [],
            "rules": [],
            "sessions": [],
        }
        with open(temp_path, "w") as f:
            json.dump(data, f)

        store = FileGatewayStore(path=temp_path)
        messages = await store.load_messages()
        await store.close()

        # Should load the 2 valid messages
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_malformed_device_in_file(self, temp_path):
        """Test partial load continues with malformed devices."""
        import json

        temp_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "messages": [],
            "devices": [
                {"device_id": "d1", "name": "Valid", "device_type": "laptop"},
                {},  # Empty device
                {"device_id": "d3", "name": "Also Valid", "device_type": "phone"},
            ],
            "rules": [],
            "sessions": [],
        }
        with open(temp_path, "w") as f:
            json.dump(data, f)

        store = FileGatewayStore(path=temp_path)
        devices = await store.load_devices()
        await store.close()

        assert len(devices) == 2

    @pytest.mark.asyncio
    async def test_auto_save_interval_throttling(self, temp_path):
        """Test auto-save interval throttling behavior."""
        store = FileGatewayStore(
            path=temp_path,
            auto_save=True,
            auto_save_interval=10.0,  # 10 seconds
        )

        # First save should write
        msg1 = InboxMessage(message_id="m1", channel="s", sender="a", content="1")
        await store.save_message(msg1)

        # Check file was created
        assert temp_path.exists()

        # Second save should be throttled (no immediate write)
        store._last_save = store._last_save  # Record the last save time
        msg2 = InboxMessage(message_id="m2", channel="s", sender="a", content="2")
        await store.save_message(msg2)

        # Force close to ensure write
        await store.close()

        # Verify both messages persisted
        store2 = FileGatewayStore(path=temp_path)
        messages = await store2.load_messages()
        await store2.close()

        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_dirty_flag_behavior(self, temp_path):
        """Test dirty flag lifecycle."""
        store = FileGatewayStore(
            path=temp_path,
            auto_save=False,  # Disable auto-save to test dirty flag
            auto_save_interval=0,
        )

        # Initially not dirty
        assert store._dirty is False

        # Save message sets dirty flag
        msg = InboxMessage(message_id="m1", channel="s", sender="a", content="1")
        await store.save_message(msg)
        assert store._dirty is True

        # Force save clears dirty flag
        await store._save(force=True)
        assert store._dirty is False

        await store.close()

    @pytest.mark.asyncio
    async def test_force_save_ignores_interval(self, temp_path):
        """Test force save bypasses interval throttling."""
        import time

        store = FileGatewayStore(
            path=temp_path,
            auto_save=True,
            auto_save_interval=3600,  # Very long interval
        )

        msg = InboxMessage(message_id="m1", channel="s", sender="a", content="1")
        await store.save_message(msg)

        # Set last_save to recent time
        store._last_save = time.time()
        store._dirty = True

        # Force save should still work
        await store._save(force=True)
        assert store._dirty is False

        await store.close()


# =============================================================================
# Data Integrity Tests
# =============================================================================


class TestDataIntegrityEdgeCases:
    """Test data integrity edge cases."""

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
    async def test_large_metadata_handling(self, file_store):
        """Test serialization with large metadata."""
        large_metadata = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}

        msg = InboxMessage(
            message_id="m1",
            channel="slack",
            sender="alice",
            content="test",
            metadata=large_metadata,
        )
        await file_store.save_message(msg)

        messages = await file_store.load_messages()
        assert len(messages) == 1
        assert len(messages[0].metadata) == 100

    @pytest.mark.asyncio
    async def test_special_characters_in_strings(self, file_store):
        """Test handling of special characters."""
        msg = InboxMessage(
            message_id="m1",
            channel="slack",
            sender="alice",
            content='Content with "quotes", \\backslash, \n\tnewlines\ttabs, and <html>',
        )
        await file_store.save_message(msg)

        messages = await file_store.load_messages()
        assert len(messages) == 1
        assert '"quotes"' in messages[0].content
        assert "\\backslash" in messages[0].content

    @pytest.mark.asyncio
    async def test_empty_string_fields(self, file_store):
        """Test handling of empty string fields."""
        msg = InboxMessage(
            message_id="m1",
            channel="",
            sender="",
            content="",
        )
        await file_store.save_message(msg)

        messages = await file_store.load_messages()
        assert len(messages) == 1
        assert messages[0].channel == ""
        assert messages[0].sender == ""
        assert messages[0].content == ""
