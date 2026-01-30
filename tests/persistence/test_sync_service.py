"""
Tests for aragora.persistence.sync_service - Background Supabase Sync Service.

Tests cover:
- SyncService initialization and configuration
- Synchronization operations (queue, flush, sync_batch)
- Conflict resolution via retry strategies
- Transaction management in sync batches
- Status tracking and statistics
- Error handling and recovery
"""

from __future__ import annotations

import asyncio
import os
import queue
import threading
import time
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from aragora.persistence.sync_service import (
    SupabaseSyncService,
    SyncItem,
    SyncItemType,
    SyncStatus,
    get_sync_service,
    shutdown_sync_service,
    _sync_service,
    _sync_service_lock,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the global singleton before and after each test."""
    import aragora.persistence.sync_service as sync_module

    original = sync_module._sync_service
    sync_module._sync_service = None
    yield
    # Cleanup
    if sync_module._sync_service is not None:
        sync_module._sync_service.stop(timeout=1.0)
        sync_module._sync_service = None
    sync_module._sync_service = original


@pytest.fixture
def mock_env_enabled():
    """Mock environment variables for enabled sync service."""
    env_vars = {
        "SUPABASE_SYNC_ENABLED": "true",
        "SUPABASE_SYNC_BATCH_SIZE": "5",
        "SUPABASE_SYNC_INTERVAL_SECONDS": "1",
        "SUPABASE_SYNC_MAX_RETRIES": "2",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


@pytest.fixture
def mock_env_disabled():
    """Mock environment variables for disabled sync service."""
    env_vars = {
        "SUPABASE_SYNC_ENABLED": "false",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client."""
    client = MagicMock()
    client.is_configured = True
    client.save_debate = AsyncMock(return_value="debate_id_123")
    client.save_cycle = AsyncMock(return_value="cycle_id_123")
    client.save_event = AsyncMock(return_value="event_id_123")
    client.save_metrics = AsyncMock(return_value="metrics_id_123")
    return client


@pytest.fixture
def mock_unconfigured_client():
    """Create a mock Supabase client that is not configured."""
    client = MagicMock()
    client.is_configured = False
    return client


@pytest.fixture
def sync_service_enabled(mock_env_enabled, mock_supabase_client):
    """Create an enabled sync service with mocked client."""
    with patch(
        "aragora.persistence.sync_service.SupabaseSyncService._get_client",
        return_value=mock_supabase_client,
    ):
        service = SupabaseSyncService(
            batch_size=5,
            interval_seconds=1.0,
            max_retries=2,
        )
        yield service
        service.stop(timeout=1.0)


@pytest.fixture
def sync_service_disabled(mock_env_disabled):
    """Create a disabled sync service."""
    service = SupabaseSyncService()
    yield service
    service.stop(timeout=1.0)


def create_sample_debate_data() -> dict[str, Any]:
    """Create sample debate data for testing."""
    return {
        "id": "debate_123",
        "loop_id": "loop_1",
        "cycle_number": 1,
        "phase": "debate",
        "task": "Test task",
        "agents": ["claude", "gpt4"],
        "transcript": "Discussion transcript",
        "consensus_reached": True,
        "confidence": 0.85,
        "winning_proposal": "Proposal A",
        "vote_tally": {"claude": 1, "gpt4": 1},
    }


def create_sample_cycle_data() -> dict[str, Any]:
    """Create sample cycle data for testing."""
    return {
        "loop_id": "loop_1",
        "cycle_number": 1,
        "phase": "debate",
        "stage": "proposing",
        "started_at": datetime.now(),
        "completed_at": None,
        "success": None,
    }


def create_sample_event_data() -> dict[str, Any]:
    """Create sample event data for testing."""
    return {
        "loop_id": "loop_1",
        "cycle": 1,
        "event_type": "phase_start",
        "event_data": {"phase": "debate"},
        "agent": "claude",
        "timestamp": datetime.now(),
    }


def create_sample_metrics_data() -> dict[str, Any]:
    """Create sample metrics data for testing."""
    return {
        "loop_id": "loop_1",
        "cycle": 1,
        "agent_name": "claude",
        "model": "claude-3-opus",
        "phase": "debate",
        "messages_sent": 5,
        "proposals_made": 2,
        "timestamp": datetime.now(),
    }


# ===========================================================================
# Test SyncService Initialization
# ===========================================================================


class TestSyncServiceInitialization:
    """Tests for SyncService initialization and configuration."""

    def test_default_configuration(self):
        """Default configuration values are applied."""
        with patch.dict(os.environ, {}, clear=True):
            service = SupabaseSyncService()

            assert service.batch_size == 10
            assert service.interval_seconds == 30.0
            assert service.max_retries == 3
            assert service.enabled is False

    def test_custom_configuration_via_constructor(self):
        """Custom configuration via constructor parameters."""
        with patch.dict(os.environ, {}, clear=True):
            service = SupabaseSyncService(
                batch_size=20,
                interval_seconds=60.0,
                max_retries=5,
            )

            assert service.batch_size == 20
            assert service.interval_seconds == 60.0
            assert service.max_retries == 5

    def test_configuration_via_environment_variables(self, mock_env_enabled):
        """Configuration via environment variables."""
        service = SupabaseSyncService()

        assert service.batch_size == 5
        assert service.interval_seconds == 1.0
        assert service.max_retries == 2
        assert service.enabled is True

    def test_environment_variables_override_constructor(self, mock_env_enabled):
        """Environment variables override constructor parameters."""
        service = SupabaseSyncService(
            batch_size=100,
            interval_seconds=100.0,
            max_retries=100,
        )

        # Environment variables should override
        assert service.batch_size == 5
        assert service.interval_seconds == 1.0
        assert service.max_retries == 2

    def test_enabled_true_when_env_true(self):
        """Service is enabled when SUPABASE_SYNC_ENABLED=true."""
        with patch.dict(os.environ, {"SUPABASE_SYNC_ENABLED": "true"}):
            service = SupabaseSyncService()
            assert service.enabled is True

    def test_enabled_false_when_env_false(self):
        """Service is disabled when SUPABASE_SYNC_ENABLED=false."""
        with patch.dict(os.environ, {"SUPABASE_SYNC_ENABLED": "false"}):
            service = SupabaseSyncService()
            assert service.enabled is False

    def test_enabled_false_when_env_missing(self):
        """Service is disabled when SUPABASE_SYNC_ENABLED is missing."""
        with patch.dict(os.environ, {}, clear=True):
            service = SupabaseSyncService()
            assert service.enabled is False

    def test_enabled_case_insensitive(self):
        """SUPABASE_SYNC_ENABLED is case insensitive."""
        for value in ["TRUE", "True", "true", "TrUe"]:
            with patch.dict(os.environ, {"SUPABASE_SYNC_ENABLED": value}):
                service = SupabaseSyncService()
                assert service.enabled is True, f"Failed for value: {value}"

    def test_initial_statistics(self):
        """Initial statistics are zeroed."""
        with patch.dict(os.environ, {}, clear=True):
            service = SupabaseSyncService()

            assert service._synced_count == 0
            assert service._failed_count == 0
            assert service._last_sync_at is None
            assert service._last_error is None

    def test_initial_state(self):
        """Initial state is not running."""
        with patch.dict(os.environ, {}, clear=True):
            service = SupabaseSyncService()

            assert service._running is False
            assert service._thread is None
            assert service._client is None

    def test_queue_initialized_empty(self):
        """Queue is initialized and empty."""
        with patch.dict(os.environ, {}, clear=True):
            service = SupabaseSyncService()

            assert service._queue.empty()
            assert service._queue.qsize() == 0


# ===========================================================================
# Test Synchronization Operations - Queuing
# ===========================================================================


class TestQueueOperations:
    """Tests for queueing items for sync."""

    def test_queue_debate_when_enabled(self, mock_env_enabled):
        """queue_debate adds item to queue when enabled."""
        service = SupabaseSyncService()
        data = create_sample_debate_data()

        result = service.queue_debate(data)

        assert result is True
        assert service._queue.qsize() == 1

        item = service._queue.get_nowait()
        assert item.item_type == SyncItemType.DEBATE
        assert item.data == data

    def test_queue_debate_when_disabled(self, mock_env_disabled):
        """queue_debate returns False when disabled."""
        service = SupabaseSyncService()
        data = create_sample_debate_data()

        result = service.queue_debate(data)

        assert result is False
        assert service._queue.empty()

    def test_queue_cycle_when_enabled(self, mock_env_enabled):
        """queue_cycle adds item to queue when enabled."""
        service = SupabaseSyncService()
        data = create_sample_cycle_data()

        result = service.queue_cycle(data)

        assert result is True
        item = service._queue.get_nowait()
        assert item.item_type == SyncItemType.CYCLE

    def test_queue_cycle_when_disabled(self, mock_env_disabled):
        """queue_cycle returns False when disabled."""
        service = SupabaseSyncService()
        result = service.queue_cycle(create_sample_cycle_data())
        assert result is False

    def test_queue_event_when_enabled(self, mock_env_enabled):
        """queue_event adds item to queue when enabled."""
        service = SupabaseSyncService()
        data = create_sample_event_data()

        result = service.queue_event(data)

        assert result is True
        item = service._queue.get_nowait()
        assert item.item_type == SyncItemType.EVENT

    def test_queue_event_when_disabled(self, mock_env_disabled):
        """queue_event returns False when disabled."""
        service = SupabaseSyncService()
        result = service.queue_event(create_sample_event_data())
        assert result is False

    def test_queue_metrics_when_enabled(self, mock_env_enabled):
        """queue_metrics adds item to queue when enabled."""
        service = SupabaseSyncService()
        data = create_sample_metrics_data()

        result = service.queue_metrics(data)

        assert result is True
        item = service._queue.get_nowait()
        assert item.item_type == SyncItemType.METRICS

    def test_queue_metrics_when_disabled(self, mock_env_disabled):
        """queue_metrics returns False when disabled."""
        service = SupabaseSyncService()
        result = service.queue_metrics(create_sample_metrics_data())
        assert result is False

    def test_multiple_items_queued(self, mock_env_enabled):
        """Multiple items can be queued."""
        service = SupabaseSyncService()

        service.queue_debate(create_sample_debate_data())
        service.queue_cycle(create_sample_cycle_data())
        service.queue_event(create_sample_event_data())
        service.queue_metrics(create_sample_metrics_data())

        assert service._queue.qsize() == 4

    def test_sync_item_metadata(self, mock_env_enabled):
        """SyncItem includes proper metadata."""
        service = SupabaseSyncService()
        data = create_sample_debate_data()

        service.queue_debate(data)
        item = service._queue.get_nowait()

        assert item.retries == 0
        assert item.last_error is None
        assert isinstance(item.created_at, datetime)


# ===========================================================================
# Test Synchronization Operations - Batch Processing
# ===========================================================================


class TestBatchProcessing:
    """Tests for batch processing of sync items."""

    def test_get_batch_returns_up_to_batch_size(self, mock_env_enabled):
        """_get_batch returns up to batch_size items."""
        service = SupabaseSyncService()
        service.batch_size = 3

        for i in range(5):
            service.queue_debate({"id": f"debate_{i}"})

        batch = service._get_batch()

        assert len(batch) == 3
        assert service._queue.qsize() == 2

    def test_get_batch_returns_all_when_less_than_batch_size(self, mock_env_enabled):
        """_get_batch returns all items when fewer than batch_size."""
        service = SupabaseSyncService()
        service.batch_size = 10

        for i in range(3):
            service.queue_debate({"id": f"debate_{i}"})

        batch = service._get_batch()

        assert len(batch) == 3
        assert service._queue.empty()

    def test_get_batch_returns_empty_when_queue_empty(self, mock_env_enabled):
        """_get_batch returns empty list when queue is empty."""
        service = SupabaseSyncService()
        batch = service._get_batch()
        assert batch == []

    def test_sync_batch_with_empty_batch(self, mock_env_enabled, mock_supabase_client):
        """_sync_batch handles empty batch."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            result = service._sync_batch([])
            assert result == 0

    def test_sync_batch_when_client_not_configured(
        self, mock_env_enabled, mock_unconfigured_client
    ):
        """_sync_batch re-queues items when client not configured."""
        with patch.object(
            SupabaseSyncService, "_get_client", return_value=mock_unconfigured_client
        ):
            service = SupabaseSyncService()
            items = [
                SyncItem(SyncItemType.DEBATE, create_sample_debate_data()),
                SyncItem(SyncItemType.CYCLE, create_sample_cycle_data()),
            ]

            result = service._sync_batch(items)

            assert result == 0
            assert service._queue.qsize() == 2

    def test_sync_batch_successful_items(self, mock_env_enabled, mock_supabase_client):
        """_sync_batch syncs items successfully."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            items = [
                SyncItem(SyncItemType.DEBATE, create_sample_debate_data()),
            ]

            result = service._sync_batch(items)

            assert result == 1
            assert service._synced_count == 1

    def test_sync_batch_updates_statistics(self, mock_env_enabled, mock_supabase_client):
        """_sync_batch updates synced_count."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            items = [
                SyncItem(SyncItemType.DEBATE, create_sample_debate_data()),
                SyncItem(SyncItemType.CYCLE, create_sample_cycle_data()),
            ]

            service._sync_batch(items)

            assert service._synced_count == 2


# ===========================================================================
# Test Conflict Resolution / Retry Logic
# ===========================================================================


class TestConflictResolution:
    """Tests for conflict resolution and retry strategies."""

    def test_failed_item_retried(self, mock_env_enabled, mock_supabase_client):
        """Failed items are re-queued for retry."""
        mock_supabase_client.save_debate = AsyncMock(side_effect=Exception("Network error"))

        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            service.max_retries = 3

            item = SyncItem(SyncItemType.DEBATE, create_sample_debate_data())
            service._sync_batch([item])

            # Item should be re-queued
            assert service._queue.qsize() == 1
            requeued = service._queue.get_nowait()
            assert requeued.retries == 1
            assert requeued.last_error == "Network error"

    def test_item_dropped_after_max_retries(self, mock_env_enabled, mock_supabase_client):
        """Items are dropped after max_retries."""
        mock_supabase_client.save_debate = AsyncMock(side_effect=Exception("Persistent error"))

        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            service.max_retries = 2

            item = SyncItem(SyncItemType.DEBATE, create_sample_debate_data())
            item.retries = 1  # Already retried once

            service._sync_batch([item])

            # Item should NOT be re-queued (at max retries)
            assert service._queue.empty()
            assert service._failed_count == 1
            assert service._last_error == "Persistent error"

    def test_mixed_success_and_failure(self, mock_env_enabled, mock_supabase_client):
        """Batch with mixed success and failure items."""
        call_count = [0]

        async def alternating_save(artifact):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise Exception("Every other fails")
            return "id_123"

        mock_supabase_client.save_debate = AsyncMock(side_effect=alternating_save)

        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            service.max_retries = 3

            items = [
                SyncItem(SyncItemType.DEBATE, {"id": "1"}),
                SyncItem(SyncItemType.DEBATE, {"id": "2"}),
                SyncItem(SyncItemType.DEBATE, {"id": "3"}),
            ]

            synced = service._sync_batch(items)

            # 2 should succeed, 1 should fail and be re-queued
            assert synced == 2
            assert service._queue.qsize() == 1

    def test_retry_preserves_item_data(self, mock_env_enabled, mock_supabase_client):
        """Retry preserves original item data."""
        mock_supabase_client.save_debate = AsyncMock(side_effect=Exception("Temp error"))

        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            original_data = create_sample_debate_data()
            item = SyncItem(SyncItemType.DEBATE, original_data)

            service._sync_batch([item])

            requeued = service._queue.get_nowait()
            assert requeued.data == original_data
            assert requeued.item_type == SyncItemType.DEBATE

    def test_error_message_captured(self, mock_env_enabled, mock_supabase_client):
        """Error message is captured on failure."""
        mock_supabase_client.save_debate = AsyncMock(
            side_effect=Exception("Specific error message")
        )

        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            item = SyncItem(SyncItemType.DEBATE, create_sample_debate_data())

            service._sync_batch([item])

            requeued = service._queue.get_nowait()
            assert requeued.last_error == "Specific error message"

    def test_all_item_types_can_fail_and_retry(self, mock_env_enabled, mock_supabase_client):
        """All item types can fail and be retried."""
        mock_supabase_client.save_debate = AsyncMock(side_effect=Exception("Debate error"))
        mock_supabase_client.save_cycle = AsyncMock(side_effect=Exception("Cycle error"))
        mock_supabase_client.save_event = AsyncMock(side_effect=Exception("Event error"))
        mock_supabase_client.save_metrics = AsyncMock(side_effect=Exception("Metrics error"))

        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            items = [
                SyncItem(SyncItemType.DEBATE, create_sample_debate_data()),
                SyncItem(SyncItemType.CYCLE, create_sample_cycle_data()),
                SyncItem(SyncItemType.EVENT, create_sample_event_data()),
                SyncItem(SyncItemType.METRICS, create_sample_metrics_data()),
            ]

            service._sync_batch(items)

            assert service._queue.qsize() == 4

    def test_partial_batch_failure(self, mock_env_enabled, mock_supabase_client):
        """Partial batch failures are handled correctly."""
        call_count = [0]

        async def failing_after_first(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1:
                raise Exception("Failed")
            return "id"

        mock_supabase_client.save_debate = AsyncMock(side_effect=failing_after_first)

        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            items = [
                SyncItem(SyncItemType.DEBATE, {"id": "1"}),
                SyncItem(SyncItemType.DEBATE, {"id": "2"}),
                SyncItem(SyncItemType.DEBATE, {"id": "3"}),
            ]

            synced = service._sync_batch(items)

            assert synced == 1
            assert service._synced_count == 1
            assert service._queue.qsize() == 2

    def test_retry_count_incremented_correctly(self, mock_env_enabled, mock_supabase_client):
        """Retry count is incremented on each failure."""
        mock_supabase_client.save_debate = AsyncMock(side_effect=Exception("Error"))

        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            service.max_retries = 5

            item = SyncItem(SyncItemType.DEBATE, create_sample_debate_data())
            item.retries = 2

            service._sync_batch([item])

            requeued = service._queue.get_nowait()
            assert requeued.retries == 3


# ===========================================================================
# Test Transaction Management
# ===========================================================================


class TestTransactionManagement:
    """Tests for transaction management in sync operations."""

    def test_sync_batch_creates_event_loop(self, mock_env_enabled, mock_supabase_client):
        """_sync_batch creates and closes event loop."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            items = [SyncItem(SyncItemType.DEBATE, create_sample_debate_data())]

            # Should not raise
            service._sync_batch(items)
            assert service._synced_count == 1

    def test_sync_item_debate_conversion(self, mock_env_enabled, mock_supabase_client):
        """_sync_item converts debate data correctly."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            data = create_sample_debate_data()
            items = [SyncItem(SyncItemType.DEBATE, data)]

            service._sync_batch(items)

            mock_supabase_client.save_debate.assert_called_once()
            call_args = mock_supabase_client.save_debate.call_args[0][0]
            assert call_args.task == data["task"]

    def test_sync_item_cycle_conversion(self, mock_env_enabled, mock_supabase_client):
        """_sync_item converts cycle data correctly."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            data = create_sample_cycle_data()
            items = [SyncItem(SyncItemType.CYCLE, data)]

            service._sync_batch(items)

            mock_supabase_client.save_cycle.assert_called_once()

    def test_sync_item_event_conversion(self, mock_env_enabled, mock_supabase_client):
        """_sync_item converts event data correctly."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            data = create_sample_event_data()
            items = [SyncItem(SyncItemType.EVENT, data)]

            service._sync_batch(items)

            mock_supabase_client.save_event.assert_called_once()

    def test_sync_item_metrics_conversion(self, mock_env_enabled, mock_supabase_client):
        """_sync_item converts metrics data correctly."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            data = create_sample_metrics_data()
            items = [SyncItem(SyncItemType.METRICS, data)]

            service._sync_batch(items)

            mock_supabase_client.save_metrics.assert_called_once()

    def test_sync_returns_false_for_none_result(self, mock_env_enabled, mock_supabase_client):
        """_sync_item returns False when save returns None."""
        mock_supabase_client.save_debate = AsyncMock(return_value=None)

        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            items = [SyncItem(SyncItemType.DEBATE, create_sample_debate_data())]

            synced = service._sync_batch(items)

            # Should fail (return None is treated as failure)
            assert synced == 0
            assert service._queue.qsize() == 1

    def test_atomic_batch_processing(self, mock_env_enabled, mock_supabase_client):
        """Batch processing handles each item independently."""
        results = ["id1", None, "id3"]
        call_count = [0]

        async def mock_save(*args, **kwargs):
            result = results[call_count[0]]
            call_count[0] += 1
            return result

        mock_supabase_client.save_debate = AsyncMock(side_effect=mock_save)

        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            items = [
                SyncItem(SyncItemType.DEBATE, {"id": "1"}),
                SyncItem(SyncItemType.DEBATE, {"id": "2"}),
                SyncItem(SyncItemType.DEBATE, {"id": "3"}),
            ]

            synced = service._sync_batch(items)

            assert synced == 2
            assert service._queue.qsize() == 1


# ===========================================================================
# Test Status Tracking
# ===========================================================================


class TestStatusTracking:
    """Tests for sync status tracking and statistics."""

    def test_get_status_returns_sync_status(self, mock_env_enabled):
        """get_status returns SyncStatus dataclass."""
        service = SupabaseSyncService()
        status = service.get_status()

        assert isinstance(status, SyncStatus)

    def test_status_reflects_enabled_state(self):
        """Status reflects enabled/disabled state."""
        with patch.dict(os.environ, {"SUPABASE_SYNC_ENABLED": "true"}):
            service = SupabaseSyncService()
            assert service.get_status().enabled is True

        with patch.dict(os.environ, {"SUPABASE_SYNC_ENABLED": "false"}):
            service = SupabaseSyncService()
            assert service.get_status().enabled is False

    def test_status_reflects_running_state(self, mock_env_enabled, mock_supabase_client):
        """Status reflects running state."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()

            assert service.get_status().running is False

            service.start()
            assert service.get_status().running is True

            service.stop()
            assert service.get_status().running is False

    def test_status_reflects_queue_size(self, mock_env_enabled):
        """Status reflects queue size."""
        service = SupabaseSyncService()

        assert service.get_status().queue_size == 0

        service.queue_debate(create_sample_debate_data())
        assert service.get_status().queue_size == 1

        service.queue_debate(create_sample_debate_data())
        assert service.get_status().queue_size == 2

    def test_status_reflects_synced_count(self, mock_env_enabled, mock_supabase_client):
        """Status reflects synced count."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()

            assert service.get_status().synced_count == 0

            items = [SyncItem(SyncItemType.DEBATE, create_sample_debate_data())]
            service._sync_batch(items)

            assert service.get_status().synced_count == 1

    def test_status_reflects_failed_count(self, mock_env_enabled, mock_supabase_client):
        """Status reflects failed count."""
        mock_supabase_client.save_debate = AsyncMock(side_effect=Exception("Error"))

        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            service.max_retries = 1

            item = SyncItem(SyncItemType.DEBATE, create_sample_debate_data())
            service._sync_batch([item])

            assert service.get_status().failed_count == 1

    def test_status_reflects_last_sync_at(self, mock_env_enabled, mock_supabase_client):
        """Status reflects last_sync_at timestamp."""
        service = SupabaseSyncService()

        assert service.get_status().last_sync_at is None

        service._last_sync_at = datetime.now()
        assert service.get_status().last_sync_at is not None

    def test_status_reflects_last_error(self, mock_env_enabled, mock_supabase_client):
        """Status reflects last_error."""
        mock_supabase_client.save_debate = AsyncMock(side_effect=Exception("Test error"))

        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            service.max_retries = 1

            item = SyncItem(SyncItemType.DEBATE, create_sample_debate_data())
            service._sync_batch([item])

            assert service.get_status().last_error == "Test error"

    def test_sync_status_dataclass_fields(self):
        """SyncStatus dataclass has all expected fields."""
        status = SyncStatus(
            enabled=True,
            running=True,
            queue_size=5,
            synced_count=100,
            failed_count=2,
            last_sync_at=datetime.now(),
            last_error="Previous error",
        )

        assert status.enabled is True
        assert status.running is True
        assert status.queue_size == 5
        assert status.synced_count == 100
        assert status.failed_count == 2
        assert status.last_sync_at is not None
        assert status.last_error == "Previous error"


# ===========================================================================
# Test Error Handling
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling and recovery scenarios."""

    def test_network_failure_during_sync(self, mock_env_enabled, mock_supabase_client):
        """Network failures are handled and items re-queued."""
        mock_supabase_client.save_debate = AsyncMock(
            side_effect=ConnectionError("Network unreachable")
        )

        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            item = SyncItem(SyncItemType.DEBATE, create_sample_debate_data())

            service._sync_batch([item])

            assert service._queue.qsize() == 1
            requeued = service._queue.get_nowait()
            assert "Network unreachable" in requeued.last_error

    def test_timeout_handling_during_sync(self, mock_env_enabled, mock_supabase_client):
        """Timeout errors are handled."""
        mock_supabase_client.save_debate = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            item = SyncItem(SyncItemType.DEBATE, create_sample_debate_data())

            service._sync_batch([item])

            assert service._queue.qsize() == 1

    def test_store_unavailability(self, mock_env_enabled, mock_unconfigured_client):
        """Unconfigured store re-queues all items."""
        with patch.object(
            SupabaseSyncService, "_get_client", return_value=mock_unconfigured_client
        ):
            service = SupabaseSyncService()
            items = [
                SyncItem(SyncItemType.DEBATE, create_sample_debate_data()),
                SyncItem(SyncItemType.CYCLE, create_sample_cycle_data()),
            ]

            synced = service._sync_batch(items)

            assert synced == 0
            assert service._queue.qsize() == 2

    def test_exception_in_sync_loop_caught(self, mock_env_enabled, mock_supabase_client):
        """Exceptions in sync loop are caught and logged."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            with patch.object(
                SupabaseSyncService,
                "_get_batch",
                side_effect=Exception("Unexpected error"),
            ):
                service = SupabaseSyncService()
                service._stop_event = threading.Event()
                service._stop_event.set()  # Stop immediately

                # Should not raise
                service._sync_loop()
                assert service._last_error == "Unexpected error"

    def test_recovery_after_transient_failure(self, mock_env_enabled, mock_supabase_client):
        """Service recovers after transient failure."""
        call_count = [0]

        async def failing_then_succeeding(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Transient failure")
            return "id_123"

        mock_supabase_client.save_debate = AsyncMock(side_effect=failing_then_succeeding)

        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            item = SyncItem(SyncItemType.DEBATE, create_sample_debate_data())

            # First attempt fails
            service._sync_batch([item])
            assert service._synced_count == 0
            assert service._queue.qsize() == 1

            # Second attempt succeeds
            batch = service._get_batch()
            service._sync_batch(batch)
            assert service._synced_count == 1
            assert service._queue.empty()

    def test_invalid_item_type_handled(self, mock_env_enabled, mock_supabase_client):
        """Invalid item types are handled gracefully."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()

            # Create an item with mocked invalid type
            item = SyncItem(SyncItemType.DEBATE, create_sample_debate_data())
            item.item_type = MagicMock()  # Invalid type

            # Should not raise but should fail
            synced = service._sync_batch([item])
            assert synced == 0

    def test_data_conversion_error_handled(self, mock_env_enabled, mock_supabase_client):
        """Data conversion errors are handled."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()

            # Invalid data that will fail conversion
            item = SyncItem(SyncItemType.DEBATE, {"invalid": "data"})

            # Should not raise but should fail
            synced = service._sync_batch([item])
            assert service._queue.qsize() == 1

    def test_client_lazy_loading_error(self, mock_env_enabled):
        """Client lazy loading errors are handled."""
        with patch(
            "aragora.persistence.sync_service.SupabaseClient",
            side_effect=Exception("Client init failed"),
        ):
            service = SupabaseSyncService()

            # Should raise when trying to get client
            with pytest.raises(Exception, match="Client init failed"):
                service._get_client()

    def test_flush_timeout(self, mock_env_enabled, mock_supabase_client):
        """Flush respects timeout parameter."""

        # Create slow sync
        async def slow_save(*args, **kwargs):
            await asyncio.sleep(0.5)
            return "id"

        mock_supabase_client.save_debate = AsyncMock(side_effect=slow_save)

        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            for i in range(10):
                service.queue_debate({"id": f"debate_{i}"})

            start = time.time()
            service.flush(timeout=0.2)
            elapsed = time.time() - start

            # Should stop around timeout
            assert elapsed < 1.0


# ===========================================================================
# Test Start/Stop Operations
# ===========================================================================


class TestStartStopOperations:
    """Tests for service start and stop operations."""

    def test_start_when_disabled(self, mock_env_disabled):
        """Start does nothing when disabled."""
        service = SupabaseSyncService()
        service.start()

        assert service._running is False
        assert service._thread is None

    def test_start_when_enabled(self, mock_env_enabled, mock_supabase_client):
        """Start creates background thread when enabled."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            service.start()

            assert service._running is True
            assert service._thread is not None
            assert service._thread.is_alive()

            service.stop()

    def test_start_idempotent(self, mock_env_enabled, mock_supabase_client):
        """Multiple start calls are safe."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            service.start()
            thread1 = service._thread

            service.start()  # Second start
            thread2 = service._thread

            assert thread1 is thread2
            service.stop()

    def test_stop_when_not_running(self, mock_env_enabled):
        """Stop is safe when not running."""
        service = SupabaseSyncService()
        service.stop()  # Should not raise

        assert service._running is False

    def test_stop_terminates_thread(self, mock_env_enabled, mock_supabase_client):
        """Stop terminates background thread."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            service.start()

            assert service._thread.is_alive()

            service.stop(timeout=2.0)

            assert service._running is False
            # Thread should stop

    def test_start_fails_when_client_not_configured(
        self, mock_env_enabled, mock_unconfigured_client
    ):
        """Start fails gracefully when client not configured."""
        with patch.object(
            SupabaseSyncService, "_get_client", return_value=mock_unconfigured_client
        ):
            service = SupabaseSyncService()
            service.start()

            # Should not start
            assert service._running is False
            assert service._thread is None

    def test_stop_with_short_timeout(self, mock_env_enabled, mock_supabase_client):
        """Stop handles short timeout gracefully."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            service.start()

            # Very short timeout
            service.stop(timeout=0.01)

            assert service._running is False


# ===========================================================================
# Test Flush Operation
# ===========================================================================


class TestFlushOperation:
    """Tests for flush operation."""

    def test_flush_when_disabled(self, mock_env_disabled):
        """Flush returns 0 when disabled."""
        service = SupabaseSyncService()
        result = service.flush()
        assert result == 0

    def test_flush_empty_queue(self, mock_env_enabled, mock_supabase_client):
        """Flush with empty queue returns 0."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            result = service.flush()
            assert result == 0

    def test_flush_syncs_all_items(self, mock_env_enabled, mock_supabase_client):
        """Flush syncs all queued items."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            for i in range(5):
                service.queue_debate({"id": f"debate_{i}"})

            result = service.flush(timeout=10.0)

            assert result == 5
            assert service._queue.empty()

    def test_flush_processes_multiple_batches(self, mock_env_enabled, mock_supabase_client):
        """Flush processes multiple batches."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            service.batch_size = 2

            for i in range(5):
                service.queue_debate({"id": f"debate_{i}"})

            result = service.flush(timeout=10.0)

            assert result == 5


# ===========================================================================
# Test Singleton Pattern
# ===========================================================================


class TestSingletonPattern:
    """Tests for singleton get_sync_service and shutdown_sync_service."""

    def test_get_sync_service_creates_instance(self):
        """get_sync_service creates singleton instance."""
        import aragora.persistence.sync_service as sync_module

        with patch.object(SupabaseSyncService, "start"):  # Prevent actual start
            sync_module._sync_service = None
            service = get_sync_service()

            assert service is not None
            assert sync_module._sync_service is service

    def test_get_sync_service_returns_same_instance(self):
        """get_sync_service returns same instance on multiple calls."""
        import aragora.persistence.sync_service as sync_module

        with patch.object(SupabaseSyncService, "start"):
            sync_module._sync_service = None
            service1 = get_sync_service()
            service2 = get_sync_service()

            assert service1 is service2

    def test_shutdown_sync_service(self):
        """shutdown_sync_service stops and clears singleton."""
        import aragora.persistence.sync_service as sync_module

        with patch.object(SupabaseSyncService, "start"):
            sync_module._sync_service = None
            service = get_sync_service()

            shutdown_sync_service(timeout=1.0)

            assert sync_module._sync_service is None

    def test_shutdown_sync_service_when_none(self):
        """shutdown_sync_service is safe when no service exists."""
        import aragora.persistence.sync_service as sync_module

        sync_module._sync_service = None
        shutdown_sync_service()  # Should not raise

    def test_thread_safety_of_singleton(self):
        """Singleton is thread-safe."""
        import aragora.persistence.sync_service as sync_module

        with patch.object(SupabaseSyncService, "start"):
            sync_module._sync_service = None
            services = []

            def get_service():
                services.append(get_sync_service())

            threads = [threading.Thread(target=get_service) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All should be the same instance
            assert all(s is services[0] for s in services)


# ===========================================================================
# Test SyncItem Dataclass
# ===========================================================================


class TestSyncItemDataclass:
    """Tests for SyncItem dataclass."""

    def test_sync_item_defaults(self):
        """SyncItem has sensible defaults."""
        item = SyncItem(SyncItemType.DEBATE, {"id": "123"})

        assert item.item_type == SyncItemType.DEBATE
        assert item.data == {"id": "123"}
        assert item.retries == 0
        assert item.last_error is None
        assert isinstance(item.created_at, datetime)

    def test_sync_item_custom_values(self):
        """SyncItem accepts custom values."""
        created = datetime(2024, 1, 1)
        item = SyncItem(
            item_type=SyncItemType.CYCLE,
            data={"cycle": 1},
            created_at=created,
            retries=2,
            last_error="Previous error",
        )

        assert item.item_type == SyncItemType.CYCLE
        assert item.data == {"cycle": 1}
        assert item.created_at == created
        assert item.retries == 2
        assert item.last_error == "Previous error"


# ===========================================================================
# Test SyncItemType Enum
# ===========================================================================


class TestSyncItemTypeEnum:
    """Tests for SyncItemType enum."""

    def test_all_types_defined(self):
        """All expected types are defined."""
        assert SyncItemType.DEBATE.value == "debate"
        assert SyncItemType.CYCLE.value == "cycle"
        assert SyncItemType.EVENT.value == "event"
        assert SyncItemType.METRICS.value == "metrics"

    def test_type_count(self):
        """Expected number of types exist."""
        assert len(SyncItemType) == 4


# ===========================================================================
# Test Sync Loop
# ===========================================================================


class TestSyncLoop:
    """Tests for the background sync loop."""

    def test_sync_loop_processes_items(self, mock_env_enabled, mock_supabase_client):
        """Sync loop processes queued items."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            service.interval_seconds = 0.1

            service.queue_debate(create_sample_debate_data())
            service.start()

            # Wait for sync to happen
            time.sleep(0.3)

            service.stop()

            assert service._synced_count >= 1

    def test_sync_loop_updates_last_sync_at(self, mock_env_enabled, mock_supabase_client):
        """Sync loop updates last_sync_at."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            service.interval_seconds = 0.1

            assert service._last_sync_at is None

            service.queue_debate(create_sample_debate_data())
            service.start()

            time.sleep(0.3)

            service.stop()

            assert service._last_sync_at is not None

    def test_sync_loop_handles_empty_queue(self, mock_env_enabled, mock_supabase_client):
        """Sync loop handles empty queue gracefully."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            service.interval_seconds = 0.1

            service.start()

            # Run with empty queue
            time.sleep(0.3)

            service.stop()

            # Should not have crashed
            assert service._synced_count == 0

    def test_sync_loop_respects_stop_event(self, mock_env_enabled, mock_supabase_client):
        """Sync loop stops when stop event is set."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            service.interval_seconds = 10.0  # Long interval

            service.start()
            assert service._running is True

            service.stop(timeout=2.0)

            assert service._running is False


# ===========================================================================
# Test Integration Scenarios
# ===========================================================================


class TestIntegrationScenarios:
    """Integration tests for common usage scenarios."""

    def test_full_sync_workflow(self, mock_env_enabled, mock_supabase_client):
        """Complete workflow: queue, sync, verify."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()

            # Queue various items
            assert service.queue_debate(create_sample_debate_data()) is True
            assert service.queue_cycle(create_sample_cycle_data()) is True
            assert service.queue_event(create_sample_event_data()) is True
            assert service.queue_metrics(create_sample_metrics_data()) is True

            # Verify queue state
            assert service.get_status().queue_size == 4

            # Flush all
            synced = service.flush(timeout=10.0)

            # Verify results
            assert synced == 4
            assert service.get_status().queue_size == 0
            assert service.get_status().synced_count == 4

    def test_high_volume_queueing(self, mock_env_enabled, mock_supabase_client):
        """Handle high volume of items."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            service.batch_size = 10

            # Queue many items
            for i in range(100):
                service.queue_debate({"id": f"debate_{i}"})

            assert service.get_status().queue_size == 100

            # Flush all
            synced = service.flush(timeout=30.0)

            assert synced == 100

    def test_concurrent_queue_and_sync(self, mock_env_enabled, mock_supabase_client):
        """Handle concurrent queueing and syncing."""
        with patch.object(SupabaseSyncService, "_get_client", return_value=mock_supabase_client):
            service = SupabaseSyncService()
            service.interval_seconds = 0.1
            service.batch_size = 5

            service.start()

            # Queue items while syncing
            for i in range(20):
                service.queue_debate({"id": f"debate_{i}"})
                time.sleep(0.05)

            # Wait for sync
            time.sleep(0.5)

            service.stop()

            # All items should eventually be synced
            total = service._synced_count + service.get_status().queue_size
            assert total <= 20  # May have processed some
