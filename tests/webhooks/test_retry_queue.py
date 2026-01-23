"""
Tests for the WebhookRetryQueue and related components.

Tests cover:
- WebhookDelivery dataclass and serialization
- DeliveryStatus enum
- InMemoryDeliveryStore operations
- WebhookRetryQueue lifecycle and delivery processing
- Exponential backoff calculation
- Dead-letter handling
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.webhooks.retry_queue import (
    DEFAULT_MAX_ATTEMPTS,
    DeliveryStatus,
    InMemoryDeliveryStore,
    WebhookDelivery,
    WebhookRetryQueue,
    get_retry_queue,
    reset_retry_queue,
    set_retry_queue,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def delivery():
    """Create a sample webhook delivery."""
    return WebhookDelivery(
        id="test-delivery-123",
        url="https://example.com/webhook",
        payload={"event": "test", "data": {"message": "Hello"}},
        headers={"X-Custom-Header": "value"},
        correlation_id="corr-456",
        webhook_id="webhook-789",
        metadata={"source": "test"},
    )


@pytest.fixture
def store():
    """Create an in-memory delivery store."""
    return InMemoryDeliveryStore()


@pytest.fixture
async def queue(store):
    """Create a webhook retry queue with in-memory store."""
    q = WebhookRetryQueue(store=store, max_concurrent=5)
    yield q
    if q.is_running:
        await q.stop(wait=False)


@pytest.fixture(autouse=True)
async def reset_global_queue():
    """Reset global queue before and after each test."""
    await reset_retry_queue()
    yield
    await reset_retry_queue()


# =============================================================================
# WebhookDelivery Tests
# =============================================================================


class TestWebhookDelivery:
    """Tests for the WebhookDelivery dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        delivery = WebhookDelivery(
            id="test-1",
            url="https://example.com",
            payload={"event": "test"},
        )

        assert delivery.id == "test-1"
        assert delivery.url == "https://example.com"
        assert delivery.status == DeliveryStatus.PENDING
        assert delivery.attempts == 0
        assert delivery.max_attempts == DEFAULT_MAX_ATTEMPTS
        assert delivery.next_retry_at is None
        assert delivery.last_error is None
        assert delivery.headers == {}
        assert delivery.metadata == {}

    def test_calculate_next_retry_exponential_backoff(self):
        """Test exponential backoff calculation."""
        delivery = WebhookDelivery(
            id="test-1",
            url="https://example.com",
            payload={},
        )

        # First retry: 2^0 = 1 second
        delivery.attempts = 0
        next_retry = delivery.calculate_next_retry()
        expected_min = datetime.now(timezone.utc) + timedelta(seconds=0.9)
        expected_max = datetime.now(timezone.utc) + timedelta(seconds=1.1)
        assert expected_min <= next_retry <= expected_max

        # Second retry: 2^1 = 2 seconds
        delivery.attempts = 1
        next_retry = delivery.calculate_next_retry()
        expected_min = datetime.now(timezone.utc) + timedelta(seconds=1.9)
        expected_max = datetime.now(timezone.utc) + timedelta(seconds=2.1)
        assert expected_min <= next_retry <= expected_max

        # Fifth retry: 2^4 = 16 seconds
        delivery.attempts = 4
        next_retry = delivery.calculate_next_retry()
        expected_min = datetime.now(timezone.utc) + timedelta(seconds=15.9)
        expected_max = datetime.now(timezone.utc) + timedelta(seconds=16.1)
        assert expected_min <= next_retry <= expected_max

    def test_calculate_next_retry_max_cap(self):
        """Test that backoff is capped at 300 seconds."""
        delivery = WebhookDelivery(
            id="test-1",
            url="https://example.com",
            payload={},
        )

        # Very high attempt count - should cap at 300 seconds
        delivery.attempts = 20
        next_retry = delivery.calculate_next_retry()
        expected_max = datetime.now(timezone.utc) + timedelta(seconds=301)
        expected_min = datetime.now(timezone.utc) + timedelta(seconds=299)
        assert expected_min <= next_retry <= expected_max

    def test_is_ready_for_retry_pending_no_next_retry(self):
        """Test is_ready_for_retry with no next_retry_at set."""
        delivery = WebhookDelivery(
            id="test-1",
            url="https://example.com",
            payload={},
            status=DeliveryStatus.PENDING,
            next_retry_at=None,
        )
        assert delivery.is_ready_for_retry() is True

    def test_is_ready_for_retry_pending_past_time(self):
        """Test is_ready_for_retry when next_retry_at is in the past."""
        delivery = WebhookDelivery(
            id="test-1",
            url="https://example.com",
            payload={},
            status=DeliveryStatus.PENDING,
            next_retry_at=datetime.now(timezone.utc) - timedelta(seconds=10),
        )
        assert delivery.is_ready_for_retry() is True

    def test_is_ready_for_retry_pending_future_time(self):
        """Test is_ready_for_retry when next_retry_at is in the future."""
        delivery = WebhookDelivery(
            id="test-1",
            url="https://example.com",
            payload={},
            status=DeliveryStatus.PENDING,
            next_retry_at=datetime.now(timezone.utc) + timedelta(seconds=10),
        )
        assert delivery.is_ready_for_retry() is False

    def test_is_ready_for_retry_non_pending_status(self):
        """Test is_ready_for_retry with non-PENDING status."""
        for status in [
            DeliveryStatus.IN_FLIGHT,
            DeliveryStatus.DELIVERED,
            DeliveryStatus.FAILED,
            DeliveryStatus.DEAD_LETTER,
        ]:
            delivery = WebhookDelivery(
                id="test-1",
                url="https://example.com",
                payload={},
                status=status,
            )
            assert delivery.is_ready_for_retry() is False

    def test_should_dead_letter(self):
        """Test dead-letter detection based on attempt count."""
        delivery = WebhookDelivery(
            id="test-1",
            url="https://example.com",
            payload={},
            max_attempts=3,
        )

        delivery.attempts = 0
        assert delivery.should_dead_letter() is False

        delivery.attempts = 2
        assert delivery.should_dead_letter() is False

        delivery.attempts = 3
        assert delivery.should_dead_letter() is True

        delivery.attempts = 5
        assert delivery.should_dead_letter() is True

    def test_to_dict(self, delivery):
        """Test serialization to dictionary."""
        data = delivery.to_dict()

        assert data["id"] == "test-delivery-123"
        assert data["url"] == "https://example.com/webhook"
        assert data["payload"] == {"event": "test", "data": {"message": "Hello"}}
        assert data["headers"] == {"X-Custom-Header": "value"}
        assert data["status"] == "pending"
        assert data["attempts"] == 0
        assert data["correlation_id"] == "corr-456"
        assert data["webhook_id"] == "webhook-789"
        assert data["metadata"] == {"source": "test"}

    def test_from_dict(self, delivery):
        """Test deserialization from dictionary."""
        data = delivery.to_dict()
        restored = WebhookDelivery.from_dict(data)

        assert restored.id == delivery.id
        assert restored.url == delivery.url
        assert restored.payload == delivery.payload
        assert restored.headers == delivery.headers
        assert restored.status == delivery.status
        assert restored.attempts == delivery.attempts
        assert restored.correlation_id == delivery.correlation_id
        assert restored.webhook_id == delivery.webhook_id
        assert restored.metadata == delivery.metadata

    def test_to_json_and_from_json(self, delivery):
        """Test JSON serialization round-trip."""
        json_str = delivery.to_json()
        restored = WebhookDelivery.from_json(json_str)

        assert restored.id == delivery.id
        assert restored.url == delivery.url
        assert restored.payload == delivery.payload
        assert restored.status == delivery.status


# =============================================================================
# DeliveryStatus Tests
# =============================================================================


class TestDeliveryStatus:
    """Tests for the DeliveryStatus enum."""

    def test_status_values(self):
        """Test that all expected status values exist."""
        assert DeliveryStatus.PENDING.value == "pending"
        assert DeliveryStatus.IN_FLIGHT.value == "in_flight"
        assert DeliveryStatus.DELIVERED.value == "delivered"
        assert DeliveryStatus.FAILED.value == "failed"
        assert DeliveryStatus.DEAD_LETTER.value == "dead_letter"

    def test_status_string_comparison(self):
        """Test that status can be compared with strings."""
        assert DeliveryStatus.PENDING == "pending"
        assert DeliveryStatus.DEAD_LETTER == "dead_letter"


# =============================================================================
# InMemoryDeliveryStore Tests
# =============================================================================


class TestInMemoryDeliveryStore:
    """Tests for the InMemoryDeliveryStore."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, store, delivery):
        """Test saving and retrieving a delivery."""
        await store.save(delivery)
        retrieved = await store.get(delivery.id)

        assert retrieved is not None
        assert retrieved.id == delivery.id
        assert retrieved.url == delivery.url

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Test getting a non-existent delivery returns None."""
        result = await store.get("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, store, delivery):
        """Test deleting a delivery."""
        await store.save(delivery)
        assert await store.get(delivery.id) is not None

        result = await store.delete(delivery.id)
        assert result is True
        assert await store.get(delivery.id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        """Test deleting a non-existent delivery returns False."""
        result = await store.delete("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_ready_for_retry(self, store):
        """Test getting deliveries ready for retry."""
        # Create deliveries with different states
        ready_delivery = WebhookDelivery(
            id="ready-1",
            url="https://example.com",
            payload={},
            status=DeliveryStatus.PENDING,
            next_retry_at=datetime.now(timezone.utc) - timedelta(seconds=10),
        )

        future_delivery = WebhookDelivery(
            id="future-1",
            url="https://example.com",
            payload={},
            status=DeliveryStatus.PENDING,
            next_retry_at=datetime.now(timezone.utc) + timedelta(seconds=60),
        )

        in_flight_delivery = WebhookDelivery(
            id="inflight-1",
            url="https://example.com",
            payload={},
            status=DeliveryStatus.IN_FLIGHT,
        )

        await store.save(ready_delivery)
        await store.save(future_delivery)
        await store.save(in_flight_delivery)

        ready = await store.get_ready_for_retry()

        assert len(ready) == 1
        assert ready[0].id == "ready-1"

    @pytest.mark.asyncio
    async def test_get_ready_for_retry_with_limit(self, store):
        """Test get_ready_for_retry respects limit."""
        for i in range(10):
            delivery = WebhookDelivery(
                id=f"delivery-{i}",
                url="https://example.com",
                payload={},
                status=DeliveryStatus.PENDING,
            )
            await store.save(delivery)

        ready = await store.get_ready_for_retry(limit=3)
        assert len(ready) == 3

    @pytest.mark.asyncio
    async def test_get_by_status(self, store):
        """Test getting deliveries by status."""
        pending = WebhookDelivery(
            id="pending-1",
            url="https://example.com",
            payload={},
            status=DeliveryStatus.PENDING,
        )
        dead_letter = WebhookDelivery(
            id="dead-1",
            url="https://example.com",
            payload={},
            status=DeliveryStatus.DEAD_LETTER,
        )

        await store.save(pending)
        await store.save(dead_letter)

        pending_list = await store.get_by_status(DeliveryStatus.PENDING)
        dead_list = await store.get_by_status(DeliveryStatus.DEAD_LETTER)

        assert len(pending_list) == 1
        assert pending_list[0].id == "pending-1"
        assert len(dead_list) == 1
        assert dead_list[0].id == "dead-1"

    @pytest.mark.asyncio
    async def test_get_dead_letters(self, store):
        """Test getting dead-letter deliveries."""
        dead_letter = WebhookDelivery(
            id="dead-1",
            url="https://example.com",
            payload={},
            status=DeliveryStatus.DEAD_LETTER,
        )
        await store.save(dead_letter)

        dead_letters = await store.get_dead_letters()
        assert len(dead_letters) == 1
        assert dead_letters[0].id == "dead-1"

    @pytest.mark.asyncio
    async def test_count_by_status(self, store):
        """Test counting deliveries by status."""
        for i in range(3):
            await store.save(
                WebhookDelivery(
                    id=f"pending-{i}",
                    url="https://example.com",
                    payload={},
                    status=DeliveryStatus.PENDING,
                )
            )

        for i in range(2):
            await store.save(
                WebhookDelivery(
                    id=f"dead-{i}",
                    url="https://example.com",
                    payload={},
                    status=DeliveryStatus.DEAD_LETTER,
                )
            )

        counts = await store.count_by_status()

        assert counts[DeliveryStatus.PENDING] == 3
        assert counts[DeliveryStatus.DEAD_LETTER] == 2
        assert counts[DeliveryStatus.DELIVERED] == 0

    @pytest.mark.asyncio
    async def test_clear(self, store):
        """Test clearing all deliveries."""
        for i in range(5):
            await store.save(
                WebhookDelivery(
                    id=f"delivery-{i}",
                    url="https://example.com",
                    payload={},
                )
            )

        count = await store.clear()
        assert count == 5

        counts = await store.count_by_status()
        assert all(c == 0 for c in counts.values())


# =============================================================================
# WebhookRetryQueue Tests
# =============================================================================


class TestWebhookRetryQueue:
    """Tests for the WebhookRetryQueue."""

    @pytest.mark.asyncio
    async def test_enqueue(self, queue, delivery):
        """Test enqueueing a delivery."""
        delivery_id = await queue.enqueue(delivery)

        assert delivery_id == delivery.id
        assert delivery.status == DeliveryStatus.PENDING

        # Verify in store
        stored = await queue.store.get(delivery_id)
        assert stored is not None
        assert stored.url == delivery.url

    @pytest.mark.asyncio
    async def test_enqueue_generates_id(self, queue):
        """Test that enqueue generates ID if not set."""
        delivery = WebhookDelivery(
            id="",  # Empty ID
            url="https://example.com",
            payload={},
        )

        delivery_id = await queue.enqueue(delivery)
        assert delivery_id != ""
        assert delivery.id == delivery_id

    @pytest.mark.asyncio
    async def test_start_and_stop(self, queue):
        """Test starting and stopping the queue."""
        assert queue.is_running is False

        await queue.start()
        assert queue.is_running is True

        await queue.stop()
        assert queue.is_running is False

    @pytest.mark.asyncio
    async def test_double_start(self, queue):
        """Test that starting twice doesn't cause issues."""
        await queue.start()
        await queue.start()  # Should log warning but not error
        assert queue.is_running is True

        await queue.stop()

    @pytest.mark.asyncio
    async def test_successful_delivery(self, queue):
        """Test successful webhook delivery."""
        delivery = WebhookDelivery(
            id="test-1",
            url="https://example.com/webhook",
            payload={"event": "test"},
        )

        # Mock the _send_webhook method directly
        async def mock_send(d):
            return True, 200, None

        queue._send_webhook = mock_send

        await queue.enqueue(delivery)
        await queue.start()

        # Wait for processing
        await asyncio.sleep(2)

        await queue.stop()

        # Check delivery was successful
        stored = await queue.store.get(delivery.id)
        assert stored is not None
        assert stored.status == DeliveryStatus.DELIVERED
        assert stored.attempts == 1

    @pytest.mark.asyncio
    async def test_failed_delivery_retry(self, queue):
        """Test that failed deliveries are retried."""
        delivery = WebhookDelivery(
            id="test-1",
            url="https://example.com/webhook",
            payload={"event": "test"},
            max_attempts=3,
        )

        # Mock the _send_webhook method directly
        async def mock_send(d):
            return False, 500, "HTTP 500"

        queue._send_webhook = mock_send

        await queue.enqueue(delivery)
        await queue._attempt_delivery(delivery)

        # Check delivery is pending retry
        stored = await queue.store.get(delivery.id)
        assert stored is not None
        assert stored.status == DeliveryStatus.PENDING
        assert stored.attempts == 1
        assert stored.next_retry_at is not None
        assert stored.last_error == "HTTP 500"

    @pytest.mark.asyncio
    async def test_dead_letter_after_max_attempts(self, queue):
        """Test that deliveries move to dead-letter after max attempts."""
        delivery = WebhookDelivery(
            id="test-1",
            url="https://example.com/webhook",
            payload={"event": "test"},
            max_attempts=2,
            attempts=1,  # Already attempted once
        )

        # Mock the _send_webhook method directly
        async def mock_send(d):
            return False, 500, "HTTP 500"

        queue._send_webhook = mock_send

        await queue.store.save(delivery)
        await queue._attempt_delivery(delivery)

        # Check delivery is dead-lettered
        stored = await queue.store.get(delivery.id)
        assert stored is not None
        assert stored.status == DeliveryStatus.DEAD_LETTER
        assert stored.attempts == 2

    @pytest.mark.asyncio
    async def test_dead_letter_callback(self, queue):
        """Test that dead-letter callback is called."""
        callback_called = []

        async def dead_letter_callback(delivery):
            callback_called.append(delivery.id)

        queue._dead_letter_callback = dead_letter_callback

        delivery = WebhookDelivery(
            id="test-1",
            url="https://example.com/webhook",
            payload={"event": "test"},
            max_attempts=1,
        )

        # Mock the _send_webhook method directly
        async def mock_send(d):
            return False, 500, "HTTP 500"

        queue._send_webhook = mock_send

        await queue.store.save(delivery)
        await queue._attempt_delivery(delivery)

        assert "test-1" in callback_called

    @pytest.mark.asyncio
    async def test_delivery_callback(self, queue):
        """Test that delivery callback is called on success."""
        callback_called = []

        async def delivery_callback(delivery):
            callback_called.append(delivery.id)

        queue._delivery_callback = delivery_callback

        delivery = WebhookDelivery(
            id="test-1",
            url="https://example.com/webhook",
            payload={"event": "test"},
        )

        # Mock the _send_webhook method directly
        async def mock_send(d):
            return True, 200, None

        queue._send_webhook = mock_send

        await queue.store.save(delivery)
        await queue._attempt_delivery(delivery)

        assert "test-1" in callback_called

    @pytest.mark.asyncio
    async def test_retry_dead_letter(self, queue):
        """Test retrying a dead-letter delivery."""
        delivery = WebhookDelivery(
            id="test-1",
            url="https://example.com/webhook",
            payload={"event": "test"},
            status=DeliveryStatus.DEAD_LETTER,
            attempts=5,
            last_error="Previous error",
        )
        await queue.store.save(delivery)

        result = await queue.retry_dead_letter("test-1")
        assert result is True

        stored = await queue.store.get("test-1")
        assert stored.status == DeliveryStatus.PENDING
        assert stored.attempts == 0
        assert stored.last_error is None
        assert stored.next_retry_at is None

    @pytest.mark.asyncio
    async def test_retry_dead_letter_nonexistent(self, queue):
        """Test retrying a non-existent delivery."""
        result = await queue.retry_dead_letter("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_retry_dead_letter_wrong_status(self, queue):
        """Test retrying a non-dead-letter delivery."""
        delivery = WebhookDelivery(
            id="test-1",
            url="https://example.com/webhook",
            payload={"event": "test"},
            status=DeliveryStatus.PENDING,
        )
        await queue.store.save(delivery)

        result = await queue.retry_dead_letter("test-1")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_delivery(self, queue):
        """Test cancelling a pending delivery."""
        delivery = WebhookDelivery(
            id="test-1",
            url="https://example.com/webhook",
            payload={"event": "test"},
            status=DeliveryStatus.PENDING,
        )
        await queue.store.save(delivery)

        result = await queue.cancel_delivery("test-1")
        assert result is True

        stored = await queue.store.get("test-1")
        assert stored is None

    @pytest.mark.asyncio
    async def test_cancel_delivery_in_flight(self, queue):
        """Test that in-flight deliveries cannot be cancelled."""
        delivery = WebhookDelivery(
            id="test-1",
            url="https://example.com/webhook",
            payload={"event": "test"},
            status=DeliveryStatus.IN_FLIGHT,
        )
        await queue.store.save(delivery)

        result = await queue.cancel_delivery("test-1")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_stats(self, queue):
        """Test getting queue statistics."""
        # Enqueue some deliveries
        for i in range(3):
            await queue.enqueue(
                WebhookDelivery(
                    id=f"test-{i}",
                    url="https://example.com/webhook",
                    payload={"event": "test"},
                )
            )

        stats = await queue.get_stats()

        assert stats["enqueued"] == 3
        assert stats["pending"] == 3
        assert stats["running"] is False

    @pytest.mark.asyncio
    async def test_get_dead_letters(self, queue):
        """Test getting dead-letter deliveries through queue."""
        delivery = WebhookDelivery(
            id="test-1",
            url="https://example.com/webhook",
            payload={"event": "test"},
            status=DeliveryStatus.DEAD_LETTER,
        )
        await queue.store.save(delivery)

        dead_letters = await queue.get_dead_letters()
        assert len(dead_letters) == 1
        assert dead_letters[0].id == "test-1"

    @pytest.mark.asyncio
    async def test_get_delivery(self, queue, delivery):
        """Test getting a delivery by ID."""
        await queue.enqueue(delivery)

        retrieved = await queue.get_delivery(delivery.id)
        assert retrieved is not None
        assert retrieved.id == delivery.id


# =============================================================================
# Global Queue Factory Tests
# =============================================================================


class TestGlobalQueueFactory:
    """Tests for the global queue factory functions."""

    @pytest.mark.asyncio
    async def test_get_retry_queue(self):
        """Test getting the global retry queue."""
        queue = get_retry_queue()
        assert queue is not None
        assert isinstance(queue, WebhookRetryQueue)

    @pytest.mark.asyncio
    async def test_get_retry_queue_returns_same_instance(self):
        """Test that get_retry_queue returns the same instance."""
        queue1 = get_retry_queue()
        queue2 = get_retry_queue()
        assert queue1 is queue2

    @pytest.mark.asyncio
    async def test_set_retry_queue(self):
        """Test setting a custom retry queue."""
        custom_queue = WebhookRetryQueue()
        set_retry_queue(custom_queue)

        retrieved = get_retry_queue()
        assert retrieved is custom_queue

    @pytest.mark.asyncio
    async def test_reset_retry_queue(self):
        """Test resetting the global retry queue."""
        queue1 = get_retry_queue()
        await queue1.start()

        await reset_retry_queue()

        queue2 = get_retry_queue()
        assert queue2 is not queue1


# =============================================================================
# HTTP Request Tests
# =============================================================================


class TestHTTPRequests:
    """Tests for HTTP request handling."""

    @pytest.mark.asyncio
    async def test_connection_error(self, queue):
        """Test handling of connection errors."""
        delivery = WebhookDelivery(
            id="test-1",
            url="https://nonexistent.invalid/webhook",
            payload={"event": "test"},
            max_attempts=1,
        )

        # Mock the _send_webhook method to simulate connection error
        async def mock_send(d):
            return False, 0, "Connection error: Connection refused"

        queue._send_webhook = mock_send

        await queue.store.save(delivery)
        await queue._attempt_delivery(delivery)

        stored = await queue.store.get(delivery.id)
        assert stored.status == DeliveryStatus.DEAD_LETTER
        assert "Connection error" in stored.last_error

    @pytest.mark.asyncio
    async def test_timeout_error(self, queue):
        """Test handling of timeout errors."""
        delivery = WebhookDelivery(
            id="test-1",
            url="https://example.com/webhook",
            payload={"event": "test"},
            max_attempts=1,
        )

        # Mock the _send_webhook method to simulate timeout
        async def mock_send(d):
            return False, 0, "Request timed out"

        queue._send_webhook = mock_send

        await queue.store.save(delivery)
        await queue._attempt_delivery(delivery)

        stored = await queue.store.get(delivery.id)
        assert stored.status == DeliveryStatus.DEAD_LETTER
        assert "timed out" in stored.last_error

    @pytest.mark.asyncio
    async def test_4xx_errors_retry_behavior(self, queue):
        """Test 4xx error handling (still gets retried in current implementation)."""
        delivery = WebhookDelivery(
            id="test-1",
            url="https://example.com/webhook",
            payload={"event": "test"},
            max_attempts=5,  # High retry count
        )

        # Mock the _send_webhook method to simulate 400 error
        async def mock_send(d):
            return False, 400, "HTTP 400"

        queue._send_webhook = mock_send

        await queue.store.save(delivery)
        await queue._attempt_delivery(delivery)

        stored = await queue.store.get(delivery.id)
        # With current implementation, 4xx still gets retried
        # This test documents current behavior
        assert stored.attempts == 1


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestConcurrency:
    """Tests for concurrent delivery handling."""

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Test that semaphore limits concurrent deliveries."""
        store = InMemoryDeliveryStore()
        queue = WebhookRetryQueue(store=store, max_concurrent=2)

        concurrent_count = []
        max_concurrent = [0]
        lock = asyncio.Lock()

        original_send = queue._send_webhook

        async def mock_send(delivery):
            async with lock:
                concurrent_count.append(1)
                current = len(concurrent_count)
                if current > max_concurrent[0]:
                    max_concurrent[0] = current

            await asyncio.sleep(0.1)  # Simulate network delay

            async with lock:
                concurrent_count.pop()

            return True, 200, None

        queue._send_webhook = mock_send

        # Enqueue 5 deliveries
        for i in range(5):
            await queue.enqueue(
                WebhookDelivery(
                    id=f"test-{i}",
                    url="https://example.com/webhook",
                    payload={"event": "test"},
                )
            )

        await queue.start()
        await asyncio.sleep(1)  # Let processing happen
        await queue.stop()

        # Max concurrent should not exceed 2
        assert max_concurrent[0] <= 2
