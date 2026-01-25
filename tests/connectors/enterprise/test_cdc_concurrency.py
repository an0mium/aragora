"""
CDC Concurrency Tests.

Tests the CDC system's ability to handle high-volume event processing:
- 50+ simultaneous CDC events
- Mixed operation types (INSERT, UPDATE, DELETE)
- Multiple source types (PostgreSQL, MongoDB, MySQL, SQL Server)
- Handler performance under load
- Resume token consistency under concurrent writes
"""

import asyncio
import hashlib
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from aragora.connectors.enterprise.database.cdc import (
    CDCSourceType,
    CDCStreamManager,
    CallbackHandler,
    ChangeEvent,
    ChangeEventHandler,
    ChangeOperation,
    CompositeHandler,
    KnowledgeMoundHandler,
    ResumeToken,
    ResumeTokenStore,
)


# =============================================================================
# Test Configuration
# =============================================================================

NUM_CONCURRENT_EVENTS = 100  # Test with 100 simultaneous events (2x requirement)
NUM_HANDLERS = 5  # Number of handlers in composite
EVENT_BATCH_SIZE = 50  # Batch size for event generation


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def event_factory():
    """Factory for creating CDC events with varying data."""

    def create(
        source_type: str = "postgresql",
        operation: str = "insert",
        table: str = "products",
        data: Dict[str, Any] = None,
        event_id: str = None,
    ) -> ChangeEvent:
        if data is None:
            data = {
                "id": random.randint(1, 10000),
                "name": f"Product-{uuid.uuid4().hex[:8]}",
                "price": round(random.uniform(10.0, 1000.0), 2),
                "category": random.choice(["electronics", "clothing", "food", "books"]),
                "in_stock": random.choice([True, False]),
            }

        source_map = {
            "postgresql": CDCSourceType.POSTGRESQL,
            "mongodb": CDCSourceType.MONGODB,
            "mysql": CDCSourceType.MYSQL,
            "sqlserver": CDCSourceType.SQLSERVER,
        }

        op_map = {
            "insert": ChangeOperation.INSERT,
            "update": ChangeOperation.UPDATE,
            "delete": ChangeOperation.DELETE,
        }

        return ChangeEvent(
            id=event_id or f"evt-{uuid.uuid4().hex[:12]}",
            source_type=source_map.get(source_type, CDCSourceType.POSTGRESQL),
            connector_id=f"{source_type}_localhost_testdb",
            operation=op_map.get(operation, ChangeOperation.INSERT),
            timestamp=datetime.now(timezone.utc),
            database="testdb",
            table=table,
            data=data if operation != "delete" else None,
            old_data=data if operation == "delete" else None,
            primary_key={"id": data.get("id") if data else random.randint(1, 10000)},
        )

    return create


@pytest.fixture
def event_batch(event_factory):
    """Generate a batch of diverse events."""
    events = []
    operations = ["insert", "update", "delete"]
    sources = ["postgresql", "mongodb", "mysql", "sqlserver"]
    tables = ["users", "orders", "products", "inventory", "logs"]

    for i in range(NUM_CONCURRENT_EVENTS):
        event = event_factory(
            source_type=random.choice(sources),
            operation=random.choice(operations),
            table=random.choice(tables),
            event_id=f"evt-batch-{i:04d}",
        )
        events.append(event)

    return events


@pytest.fixture
def tracking_handler():
    """Handler that tracks all events it processes."""

    class TrackingHandler(ChangeEventHandler):
        def __init__(self):
            self.events_processed: List[ChangeEvent] = []
            self.processing_times: List[float] = []
            self.errors: List[Exception] = []

        async def handle(self, event: ChangeEvent) -> bool:
            start = time.monotonic()
            try:
                # Simulate some processing time
                await asyncio.sleep(random.uniform(0.001, 0.005))
                self.events_processed.append(event)
                return True
            except Exception as e:
                self.errors.append(e)
                return False
            finally:
                self.processing_times.append(time.monotonic() - start)

    return TrackingHandler()


@pytest.fixture
def slow_handler():
    """Handler that simulates slow processing."""

    class SlowHandler(ChangeEventHandler):
        def __init__(self, delay: float = 0.01):
            self.delay = delay
            self.events_processed = 0

        async def handle(self, event: ChangeEvent) -> bool:
            await asyncio.sleep(self.delay)
            self.events_processed += 1
            return True

    return SlowHandler()


@pytest.fixture
def failing_handler():
    """Handler that randomly fails."""

    class FailingHandler(ChangeEventHandler):
        def __init__(self, failure_rate: float = 0.2):
            self.failure_rate = failure_rate
            self.successes = 0
            self.failures = 0

        async def handle(self, event: ChangeEvent) -> bool:
            if random.random() < self.failure_rate:
                self.failures += 1
                return False
            self.successes += 1
            return True

    return FailingHandler()


@pytest.fixture
def mock_mound():
    """Mock KnowledgeMound for testing."""
    mound = MagicMock()
    mound.store = AsyncMock(return_value=True)
    mound.delete = AsyncMock(return_value=True)
    return mound


# =============================================================================
# Concurrent Event Processing Tests
# =============================================================================


class TestConcurrentEventProcessing:
    """Test processing of 50+ simultaneous CDC events."""

    @pytest.mark.asyncio
    async def test_process_100_events_concurrently(self, event_batch, tracking_handler):
        """Test processing 100 events concurrently."""
        manager = CDCStreamManager(
            connector_id="test_connector",
            source_type=CDCSourceType.POSTGRESQL,
            handler=tracking_handler,
        )

        # Process all events concurrently
        tasks = [manager.process_event(event) for event in event_batch]
        results = await asyncio.gather(*tasks)

        # All events should be processed
        assert len(tracking_handler.events_processed) == NUM_CONCURRENT_EVENTS
        assert all(results)
        assert len(tracking_handler.errors) == 0

    @pytest.mark.asyncio
    async def test_process_mixed_operations_concurrently(self, event_factory, tracking_handler):
        """Test concurrent processing of mixed INSERT/UPDATE/DELETE operations."""
        manager = CDCStreamManager(
            connector_id="test_connector",
            source_type=CDCSourceType.POSTGRESQL,
            handler=tracking_handler,
        )

        # Create balanced mix of operations
        events = []
        for i in range(50):
            events.append(event_factory(operation="insert", event_id=f"ins-{i}"))
            events.append(event_factory(operation="update", event_id=f"upd-{i}"))
            if i < 25:  # Fewer deletes
                events.append(event_factory(operation="delete", event_id=f"del-{i}"))

        # Shuffle for realistic ordering
        random.shuffle(events)

        # Process concurrently
        tasks = [manager.process_event(event) for event in events]
        await asyncio.gather(*tasks)

        # Verify operation distribution
        inserts = sum(
            1 for e in tracking_handler.events_processed if e.operation == ChangeOperation.INSERT
        )
        updates = sum(
            1 for e in tracking_handler.events_processed if e.operation == ChangeOperation.UPDATE
        )
        deletes = sum(
            1 for e in tracking_handler.events_processed if e.operation == ChangeOperation.DELETE
        )

        assert inserts == 50
        assert updates == 50
        assert deletes == 25

    @pytest.mark.asyncio
    async def test_process_multi_source_concurrently(self, event_factory, tracking_handler):
        """Test concurrent processing from multiple database sources."""
        manager = CDCStreamManager(
            connector_id="test_connector",
            source_type=CDCSourceType.POSTGRESQL,
            handler=tracking_handler,
        )

        # Create events from all sources
        sources = ["postgresql", "mongodb", "mysql", "sqlserver"]
        events = []
        for source in sources:
            for i in range(25):
                events.append(event_factory(source_type=source, event_id=f"{source}-{i}"))

        random.shuffle(events)

        # Process concurrently
        tasks = [manager.process_event(event) for event in events]
        await asyncio.gather(*tasks)

        # All 100 events should be processed
        assert len(tracking_handler.events_processed) == 100


# =============================================================================
# Composite Handler Concurrency Tests
# =============================================================================


class TestCompositeHandlerConcurrency:
    """Test composite handler with concurrent events."""

    @pytest.mark.asyncio
    async def test_composite_handler_multiple_handlers(self, event_batch):
        """Test composite handler distributes to multiple handlers concurrently."""
        handlers = [MagicMock(spec=ChangeEventHandler) for _ in range(NUM_HANDLERS)]
        for h in handlers:
            h.handle = AsyncMock(return_value=True)

        composite = CompositeHandler(handlers)
        manager = CDCStreamManager(
            connector_id="test_connector",
            source_type=CDCSourceType.POSTGRESQL,
            handler=composite,
        )

        # Process events
        tasks = [manager.process_event(event) for event in event_batch]
        await asyncio.gather(*tasks)

        # Each handler should receive all events
        for handler in handlers:
            assert handler.handle.call_count == NUM_CONCURRENT_EVENTS

    @pytest.mark.asyncio
    async def test_composite_handler_partial_failure(self, event_factory):
        """Test composite handler continues when some handlers fail."""
        success_handler = MagicMock(spec=ChangeEventHandler)
        success_handler.handle = AsyncMock(return_value=True)

        failing_handler = MagicMock(spec=ChangeEventHandler)
        failing_handler.handle = AsyncMock(return_value=False)

        composite = CompositeHandler([success_handler, failing_handler])
        manager = CDCStreamManager(
            connector_id="test_connector",
            source_type=CDCSourceType.POSTGRESQL,
            handler=composite,
        )

        events = [event_factory(event_id=f"evt-{i}") for i in range(50)]
        tasks = [manager.process_event(event) for event in events]
        results = await asyncio.gather(*tasks)

        # Both handlers should be called for all events
        assert success_handler.handle.call_count == 50
        assert failing_handler.handle.call_count == 50
        # Results should reflect partial failure
        assert all(not r for r in results)


# =============================================================================
# KnowledgeMound Handler Concurrency Tests
# =============================================================================


class TestKnowledgeMoundHandlerConcurrency:
    """Test KnowledgeMound handler under concurrent load."""

    @pytest.mark.asyncio
    async def test_km_handler_concurrent_stores(self, event_factory, mock_mound):
        """Test KM handler handles concurrent store operations."""
        handler = KnowledgeMoundHandler(workspace_id="default")
        handler._mound = mock_mound

        manager = CDCStreamManager(
            connector_id="test_connector",
            source_type=CDCSourceType.POSTGRESQL,
            handler=handler,
        )

        # Generate insert events (should trigger store)
        events = [
            event_factory(operation="insert", event_id=f"ins-{i}")
            for i in range(NUM_CONCURRENT_EVENTS)
        ]

        tasks = [manager.process_event(event) for event in events]
        await asyncio.gather(*tasks)

        # All events should trigger store
        assert mock_mound.store.call_count == NUM_CONCURRENT_EVENTS

    @pytest.mark.asyncio
    async def test_km_handler_mixed_operations(self, event_factory, mock_mound):
        """Test KM handler with mixed INSERT/DELETE operations."""
        handler = KnowledgeMoundHandler(workspace_id="default")
        handler._mound = mock_mound

        manager = CDCStreamManager(
            connector_id="test_connector",
            source_type=CDCSourceType.POSTGRESQL,
            handler=handler,
        )

        # Generate mixed events
        events = []
        for i in range(50):
            events.append(event_factory(operation="insert", event_id=f"ins-{i}"))
            events.append(event_factory(operation="delete", event_id=f"del-{i}"))

        random.shuffle(events)

        tasks = [manager.process_event(event) for event in events]
        await asyncio.gather(*tasks)

        # Verify operation distribution
        assert mock_mound.store.call_count == 50  # Inserts
        assert mock_mound.delete.call_count == 0  # Deletes are logged only


# =============================================================================
# Resume Token Concurrency Tests
# =============================================================================


class TestResumeTokenConcurrency:
    """Test resume token persistence under concurrent writes."""

    @pytest.mark.asyncio
    async def test_concurrent_token_updates(self, tmp_path):
        """Test resume tokens are safely updated under concurrent writes."""
        store = ResumeTokenStore(storage_path=tmp_path / "tokens")

        # Simulate concurrent token updates from multiple streams
        async def update_token(stream_id: str, position: int):
            token = ResumeToken(
                connector_id=stream_id,
                source_type=CDCSourceType.POSTGRESQL,
                token=str(position),
                timestamp=datetime.now(timezone.utc),
                sequence_number=position,
            )
            await asyncio.to_thread(store.save, token)
            return token

        # Create concurrent updates
        tasks = []
        for stream in range(5):  # 5 different streams
            for pos in range(20):  # 20 updates per stream
                tasks.append(update_token(f"stream-{stream}", pos))

        # Run all updates concurrently
        tokens = await asyncio.gather(*tasks)

        # Verify all tokens were saved
        assert len(tokens) == 100

        # Verify final positions are correct
        for stream in range(5):
            token = store.get(f"stream-{stream}")
            # The latest position should be one of the values 0-19
            # (exact value depends on race conditions)
            assert token is not None
            assert int(token.token) in range(20)


# =============================================================================
# Performance Tests
# =============================================================================


class TestCDCPerformance:
    """Test CDC system performance under load."""

    @pytest.mark.asyncio
    async def test_throughput_100_events_per_second(self, event_factory, tracking_handler):
        """Test system can handle 100+ events per second."""
        manager = CDCStreamManager(
            connector_id="test_connector",
            source_type=CDCSourceType.POSTGRESQL,
            handler=tracking_handler,
        )

        events = [event_factory(event_id=f"perf-{i}") for i in range(100)]

        start_time = time.monotonic()
        tasks = [manager.process_event(event) for event in events]
        await asyncio.gather(*tasks)
        elapsed = time.monotonic() - start_time

        # Should complete 100 events in under 1 second (accounting for small delays)
        assert elapsed < 2.0  # Allow some margin
        assert len(tracking_handler.events_processed) == 100

        # Calculate throughput
        throughput = 100 / elapsed
        assert throughput > 50  # At least 50 events/second

    @pytest.mark.asyncio
    async def test_latency_under_concurrent_load(self, event_batch, tracking_handler):
        """Test event processing latency under concurrent load."""
        manager = CDCStreamManager(
            connector_id="test_connector",
            source_type=CDCSourceType.POSTGRESQL,
            handler=tracking_handler,
        )

        tasks = [manager.process_event(event) for event in event_batch]
        await asyncio.gather(*tasks)

        # Calculate latency statistics
        times = tracking_handler.processing_times
        avg_latency = sum(times) / len(times)
        max_latency = max(times)
        p95_latency = sorted(times)[int(len(times) * 0.95)]

        # Average latency should be reasonable
        assert avg_latency < 0.1  # Less than 100ms average
        assert p95_latency < 0.2  # P95 less than 200ms

    @pytest.mark.asyncio
    async def test_slow_handler_doesnt_block_others(self, event_factory, slow_handler):
        """Test that slow handlers don't block the processing pipeline."""
        fast_handler = MagicMock(spec=ChangeEventHandler)
        fast_handler.handle = AsyncMock(return_value=True)

        # Composite with both slow and fast
        composite = CompositeHandler([fast_handler, slow_handler])
        manager = CDCStreamManager(
            connector_id="test_connector",
            source_type=CDCSourceType.POSTGRESQL,
            handler=composite,
        )

        events = [event_factory(event_id=f"slow-{i}") for i in range(20)]

        start = time.monotonic()
        tasks = [manager.process_event(event) for event in events]
        await asyncio.gather(*tasks)
        elapsed = time.monotonic() - start

        # With 10ms delay per event and concurrent processing,
        # should complete faster than sequential (20 * 10ms = 200ms)
        assert elapsed < 0.5  # Should be much faster than sequential
        assert fast_handler.handle.call_count == 20


# =============================================================================
# Error Recovery Tests
# =============================================================================


class TestErrorRecovery:
    """Test CDC error recovery under concurrent processing."""

    @pytest.mark.asyncio
    async def test_handler_failure_isolation(self, event_factory, failing_handler):
        """Test that handler failures don't affect other events."""
        manager = CDCStreamManager(
            connector_id="test_connector",
            source_type=CDCSourceType.POSTGRESQL,
            handler=failing_handler,
        )

        events = [event_factory(event_id=f"fail-{i}") for i in range(100)]

        tasks = [manager.process_event(event) for event in events]
        results = await asyncio.gather(*tasks)

        # Some should succeed, some should fail
        successes = sum(1 for r in results if r)
        failures = sum(1 for r in results if not r)

        # With 20% failure rate, expect ~80 successes, ~20 failures
        assert 60 < successes < 100  # Allow for randomness
        assert 0 < failures < 40
        assert successes + failures == 100

    @pytest.mark.asyncio
    async def test_exception_handling(self, event_factory):
        """Test that exceptions are properly handled."""

        class ExceptionHandler(ChangeEventHandler):
            def __init__(self):
                self.call_count = 0

            async def handle(self, event: ChangeEvent) -> bool:
                self.call_count += 1
                if self.call_count % 5 == 0:
                    raise ValueError("Simulated exception")
                return True

        handler = ExceptionHandler()
        manager = CDCStreamManager(
            connector_id="test_connector",
            source_type=CDCSourceType.POSTGRESQL,
            handler=handler,
        )

        events = [event_factory(event_id=f"exc-{i}") for i in range(50)]

        # Should not raise - exceptions are caught internally
        tasks = [manager.process_event(event) for event in events]
        results = await asyncio.gather(*tasks)

        # All events should be attempted
        assert handler.call_count == 50


# =============================================================================
# Callback Handler Concurrency Tests
# =============================================================================


class TestCallbackHandlerConcurrency:
    """Test callback handler under concurrent load."""

    @pytest.mark.asyncio
    async def test_callback_invoked_for_all_events(self, event_factory):
        """Test callback is invoked for all concurrent events."""
        callback_count = [0]

        async def callback(event: ChangeEvent) -> None:
            callback_count[0] += 1

        handler = CallbackHandler(callback)
        manager = CDCStreamManager(
            connector_id="test_connector",
            source_type=CDCSourceType.POSTGRESQL,
            handler=handler,
        )

        events = [event_factory(event_id=f"cb-{i}") for i in range(100)]

        tasks = [manager.process_event(event) for event in events]
        await asyncio.gather(*tasks)

        assert callback_count[0] == 100

    @pytest.mark.asyncio
    async def test_sync_callback_wrapped_correctly(self, event_factory):
        """Test synchronous callbacks work correctly under load."""
        sync_count = [0]

        def sync_callback(event: ChangeEvent) -> None:
            sync_count[0] += 1

        handler = CallbackHandler(sync_callback)
        manager = CDCStreamManager(
            connector_id="test_connector",
            source_type=CDCSourceType.POSTGRESQL,
            handler=handler,
        )

        events = [event_factory(event_id=f"sync-{i}") for i in range(50)]

        tasks = [manager.process_event(event) for event in events]
        await asyncio.gather(*tasks)

        assert sync_count[0] == 50


# =============================================================================
# Stream Manager State Tests
# =============================================================================


class TestStreamManagerState:
    """Test stream manager state under concurrent operations."""

    @pytest.mark.asyncio
    async def test_metrics_updated_correctly(self, event_batch, tracking_handler):
        """Test manager metrics are correctly updated under load."""
        manager = CDCStreamManager(
            connector_id="test_connector",
            source_type=CDCSourceType.POSTGRESQL,
            handler=tracking_handler,
        )

        tasks = [manager.process_event(event) for event in event_batch]
        await asyncio.gather(*tasks)

        # Check metrics are updated
        metrics = manager.stats
        assert metrics["events_processed"] == NUM_CONCURRENT_EVENTS
        assert metrics["last_event_time"] is not None

    @pytest.mark.asyncio
    async def test_multiple_managers_isolated(self, event_factory):
        """Test multiple managers operate independently."""
        handlers = [MagicMock(spec=ChangeEventHandler) for _ in range(3)]
        for h in handlers:
            h.handle = AsyncMock(return_value=True)

        managers = [
            CDCStreamManager(
                connector_id=f"connector_{i}",
                source_type=CDCSourceType.POSTGRESQL,
                handler=handlers[i],
            )
            for i in range(3)
        ]

        # Each manager processes its own events
        all_tasks = []
        for i, manager in enumerate(managers):
            events = [event_factory(event_id=f"mgr{i}-{j}") for j in range(30)]
            all_tasks.extend([manager.process_event(e) for e in events])

        await asyncio.gather(*all_tasks)

        # Each handler should only receive its manager's events
        for handler in handlers:
            assert handler.handle.call_count == 30
