"""Tests for Slack message delivery queue."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.chat.slack.queue import (
    MessageStatus,
    QueuedMessage,
    SlackMessageQueue,
    SlackMessageQueueStore,
)


class TestQueuedMessage:
    """Tests for QueuedMessage dataclass."""

    def test_create_message(self):
        """Test creating a queued message."""
        msg = QueuedMessage(
            id="msg-123",
            workspace_id="T12345",
            channel_id="C67890",
            text="Hello, world!",
        )

        assert msg.id == "msg-123"
        assert msg.workspace_id == "T12345"
        assert msg.channel_id == "C67890"
        assert msg.text == "Hello, world!"
        assert msg.status == MessageStatus.PENDING
        assert msg.retries == 0

    def test_message_to_dict(self):
        """Test message serialization."""
        msg = QueuedMessage(
            id="msg-1",
            workspace_id="T1",
            channel_id="C1",
            text="test",
            blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": "test"}}],
        )

        data = msg.to_dict()
        assert data["id"] == "msg-1"
        assert data["status"] == "pending"
        assert len(data["blocks"]) == 1

    def test_message_from_dict(self):
        """Test message deserialization."""
        data = {
            "id": "msg-2",
            "workspace_id": "T2",
            "channel_id": "C2",
            "text": "hello",
            "status": "delivered",
            "retries": 3,
        }

        msg = QueuedMessage.from_dict(data)
        assert msg.id == "msg-2"
        assert msg.status == MessageStatus.DELIVERED
        assert msg.retries == 3


class TestSlackMessageQueueStore:
    """Tests for SQLite-backed queue store."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a temporary queue store."""
        return SlackMessageQueueStore(db_path=str(tmp_path / "test_queue.db"))

    @pytest.fixture
    def sample_message(self):
        """Create a sample message."""
        return QueuedMessage(
            id="msg-test-1",
            workspace_id="T12345",
            channel_id="C67890",
            text="Test message",
            created_at=time.time(),
        )

    def test_insert_and_get(self, store, sample_message):
        """Test inserting and retrieving a message."""
        result = store.insert(sample_message)
        assert result is True

        retrieved = store.get(sample_message.id)
        assert retrieved is not None
        assert retrieved.id == sample_message.id
        assert retrieved.text == "Test message"
        assert retrieved.workspace_id == "T12345"

    def test_get_nonexistent(self, store):
        """Test getting a nonexistent message."""
        assert store.get("nonexistent") is None

    def test_get_pending(self, store):
        """Test retrieving pending messages."""
        for i in range(5):
            msg = QueuedMessage(
                id=f"msg-{i}",
                workspace_id="T1",
                channel_id="C1",
                text=f"Message {i}",
                created_at=time.time(),
            )
            store.insert(msg)

        pending = store.get_pending(limit=10)
        assert len(pending) == 5

    def test_get_pending_respects_retry_time(self, store):
        """Test that pending messages respect next_retry_at."""
        # Message ready for retry
        ready = QueuedMessage(
            id="msg-ready",
            workspace_id="T1",
            channel_id="C1",
            text="Ready",
            next_retry_at=time.time() - 60,
            created_at=time.time() - 120,
        )
        store.insert(ready)

        # Message not yet ready
        not_ready = QueuedMessage(
            id="msg-not-ready",
            workspace_id="T1",
            channel_id="C1",
            text="Not ready",
            next_retry_at=time.time() + 3600,
            created_at=time.time() - 60,
        )
        store.insert(not_ready)

        pending = store.get_pending()
        assert len(pending) == 1
        assert pending[0].id == "msg-ready"

    def test_mark_delivered(self, store, sample_message):
        """Test marking message as delivered."""
        store.insert(sample_message)
        store.mark_delivered(sample_message.id)

        msg = store.get(sample_message.id)
        assert msg.status == MessageStatus.DELIVERED
        assert msg.delivered_at is not None

    def test_mark_failed(self, store, sample_message):
        """Test marking message as failed with retry."""
        store.insert(sample_message)
        next_retry = time.time() + 60
        store.mark_failed(sample_message.id, "API error", next_retry)

        msg = store.get(sample_message.id)
        assert msg.status == MessageStatus.PENDING
        assert msg.retries == 1
        assert msg.last_error == "API error"
        assert msg.next_retry_at == pytest.approx(next_retry, abs=1)

    def test_mark_dead(self, store, sample_message):
        """Test moving message to dead letter queue."""
        store.insert(sample_message)
        store.mark_dead(sample_message.id, "Max retries exceeded")

        msg = store.get(sample_message.id)
        assert msg.status == MessageStatus.DEAD
        assert msg.last_error == "Max retries exceeded"

    def test_get_stats(self, store):
        """Test queue statistics."""
        for i, status in enumerate(["pending", "pending", "delivered", "dead"]):
            msg = QueuedMessage(
                id=f"msg-{i}",
                workspace_id="T1",
                channel_id="C1",
                text=f"msg {i}",
                status=MessageStatus(status),
                created_at=time.time(),
                delivered_at=time.time() if status == "delivered" else None,
            )
            store.insert(msg)

        stats = store.get_stats()
        assert stats["pending"] == 2
        assert stats["delivered"] == 1
        assert stats["dead"] == 1
        assert stats["total"] == 4

    def test_get_dead_letters(self, store):
        """Test retrieving dead letters."""
        for i in range(3):
            msg = QueuedMessage(
                id=f"dead-{i}",
                workspace_id="T1",
                channel_id="C1",
                text=f"Dead {i}",
                status=MessageStatus.DEAD,
                last_error="Failed permanently",
                created_at=time.time(),
            )
            store.insert(msg)

        dead = store.get_dead_letters()
        assert len(dead) == 3

    def test_retry_dead_letter(self, store):
        """Test retrying a dead letter message."""
        msg = QueuedMessage(
            id="dead-retry",
            workspace_id="T1",
            channel_id="C1",
            text="Retry me",
            status=MessageStatus.DEAD,
            retries=10,
            last_error="Some error",
            created_at=time.time(),
        )
        store.insert(msg)

        store.retry_dead_letter("dead-retry")

        retried = store.get("dead-retry")
        assert retried.status == MessageStatus.PENDING
        assert retried.retries == 0

    def test_cleanup_delivered(self, store):
        """Test cleaning up old delivered messages."""
        old_msg = QueuedMessage(
            id="old-delivered",
            workspace_id="T1",
            channel_id="C1",
            text="Old",
            status=MessageStatus.DELIVERED,
            delivered_at=time.time() - 48 * 3600,  # 48 hours ago
            created_at=time.time() - 49 * 3600,
        )
        store.insert(old_msg)

        recent_msg = QueuedMessage(
            id="recent-delivered",
            workspace_id="T1",
            channel_id="C1",
            text="Recent",
            status=MessageStatus.DELIVERED,
            delivered_at=time.time() - 1 * 3600,  # 1 hour ago
            created_at=time.time() - 2 * 3600,
        )
        store.insert(recent_msg)

        removed = store.cleanup_delivered(older_than_hours=24)
        assert removed == 1

        # Recent should still exist
        assert store.get("recent-delivered") is not None
        # Old should be gone
        assert store.get("old-delivered") is None

    def test_insert_with_blocks(self, store):
        """Test inserting message with Block Kit blocks."""
        msg = QueuedMessage(
            id="msg-blocks",
            workspace_id="T1",
            channel_id="C1",
            text="With blocks",
            blocks=[
                {"type": "section", "text": {"type": "mrkdwn", "text": "Hello"}},
                {"type": "divider"},
            ],
            created_at=time.time(),
        )
        store.insert(msg)

        retrieved = store.get("msg-blocks")
        assert retrieved.blocks is not None
        assert len(retrieved.blocks) == 2
        assert retrieved.blocks[0]["type"] == "section"


class TestSlackMessageQueue:
    """Tests for the high-level message queue."""

    @pytest.fixture
    def queue(self, tmp_path):
        """Create a test queue."""
        store = SlackMessageQueueStore(db_path=str(tmp_path / "test_queue.db"))
        return SlackMessageQueue(
            store=store,
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0,
            process_interval=0.1,
        )

    @pytest.mark.asyncio
    async def test_enqueue(self, queue):
        """Test enqueueing a message."""
        msg_id = await queue.enqueue(
            workspace_id="T12345",
            channel_id="C67890",
            text="Test message",
        )

        assert msg_id is not None
        msg = queue._store.get(msg_id)
        assert msg is not None
        assert msg.workspace_id == "T12345"
        assert msg.text == "Test message"

    @pytest.mark.asyncio
    async def test_enqueue_with_blocks(self, queue):
        """Test enqueueing a message with blocks."""
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "test"}}]
        msg_id = await queue.enqueue(
            workspace_id="T1",
            channel_id="C1",
            text="With blocks",
            blocks=blocks,
        )

        msg = queue._store.get(msg_id)
        assert msg.blocks is not None
        assert len(msg.blocks) == 1

    @pytest.mark.asyncio
    async def test_process_pending_delivers(self, queue):
        """Test processing delivers messages successfully."""
        msg_id = await queue.enqueue(workspace_id="T1", channel_id="C1", text="Deliver me")

        with patch.object(queue, "_send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = True
            stats = await queue.process_pending()

        assert stats["delivered"] == 1
        msg = queue._store.get(msg_id)
        assert msg.status == MessageStatus.DELIVERED

    @pytest.mark.asyncio
    async def test_process_pending_retries_on_failure(self, queue):
        """Test processing retries failed messages."""
        msg_id = await queue.enqueue(workspace_id="T1", channel_id="C1", text="Will fail")

        with patch.object(queue, "_send_message", new_callable=AsyncMock) as mock_send:
            mock_send.side_effect = ConnectionError("API error")
            stats = await queue.process_pending()

        assert stats["failed"] == 1
        msg = queue._store.get(msg_id)
        assert msg.status == MessageStatus.PENDING
        assert msg.retries == 1
        assert msg.last_error  # Sanitized error message present
        assert msg.next_retry_at is not None

    @pytest.mark.asyncio
    async def test_process_moves_to_dead_after_max_retries(self, queue):
        """Test messages move to dead letter after max retries."""
        msg_id = await queue.enqueue(workspace_id="T1", channel_id="C1", text="Die eventually")

        # Manually set retries to max
        msg = queue._store.get(msg_id)
        conn = queue._store._get_connection()
        conn.execute(
            "UPDATE queued_messages SET retries = ? WHERE id = ?",
            (queue._max_retries, msg_id),
        )
        conn.commit()

        with patch.object(queue, "_send_message", new_callable=AsyncMock) as mock_send:
            mock_send.side_effect = ConnectionError("Permanent failure")
            stats = await queue.process_pending()

        assert stats["dead"] == 1
        msg = queue._store.get(msg_id)
        assert msg.status == MessageStatus.DEAD

    def test_calculate_retry_delay(self, queue):
        """Test exponential backoff calculation."""
        delay0 = queue._calculate_retry_delay(0)
        delay1 = queue._calculate_retry_delay(1)
        delay2 = queue._calculate_retry_delay(2)

        # Base delay increases exponentially
        assert delay0 >= queue._base_delay
        assert delay1 >= queue._base_delay * 2 * 0.9  # Allow jitter margin
        assert delay2 >= queue._base_delay * 4 * 0.9

        # Should not exceed max_delay (+ jitter)
        delay10 = queue._calculate_retry_delay(10)
        assert delay10 <= queue._max_delay * 1.1

    def test_get_stats(self, queue):
        """Test queue stats."""
        stats = queue.get_stats()
        assert "pending" in stats
        assert "delivered" in stats
        assert "dead" in stats
        assert "processor_running" in stats
        assert stats["processor_running"] is False

    @pytest.mark.asyncio
    async def test_start_stop_processor(self, queue):
        """Test starting and stopping the background processor."""
        task = await queue.start_processor()
        assert task is not None
        assert queue._running is True

        await queue.stop_processor()
        assert queue._running is False

    @pytest.mark.asyncio
    async def test_get_and_retry_dead_letters(self, queue):
        """Test retrieving and retrying dead letters."""
        msg_id = await queue.enqueue(workspace_id="T1", channel_id="C1", text="Dead letter test")

        # Move to dead
        queue._store.mark_dead(msg_id, "Test failure")

        dead = queue.get_dead_letters()
        assert len(dead) == 1
        assert dead[0].id == msg_id

        # Retry
        queue.retry_dead_letter(msg_id)
        msg = queue._store.get(msg_id)
        assert msg.status == MessageStatus.PENDING
        assert msg.retries == 0
