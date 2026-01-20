"""
Background Supabase Sync Service.

Provides non-blocking background replication from SQLite to Supabase.
SQLite remains the primary storage; Supabase acts as a cloud read replica.

Usage:
    from aragora.persistence.sync_service import get_sync_service

    # Queue a debate for sync (non-blocking)
    sync = get_sync_service()
    sync.queue_debate(debate_result)

    # Check sync status
    status = sync.get_status()

Environment Variables:
    SUPABASE_SYNC_ENABLED: Set to "true" to enable sync (default: false)
    SUPABASE_SYNC_BATCH_SIZE: Items per batch (default: 10)
    SUPABASE_SYNC_INTERVAL_SECONDS: Sync interval (default: 30)
    SUPABASE_SYNC_MAX_RETRIES: Max retries per item (default: 3)
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SyncItemType(Enum):
    """Types of items that can be synced."""

    DEBATE = "debate"
    CYCLE = "cycle"
    EVENT = "event"
    METRICS = "metrics"


@dataclass
class SyncItem:
    """An item queued for sync to Supabase."""

    item_type: SyncItemType
    data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    retries: int = 0
    last_error: Optional[str] = None


@dataclass
class SyncStatus:
    """Current status of the sync service."""

    enabled: bool
    running: bool
    queue_size: int
    synced_count: int
    failed_count: int
    last_sync_at: Optional[datetime]
    last_error: Optional[str]


class SupabaseSyncService:
    """
    Background service for replicating data to Supabase.

    Runs in a background thread, periodically flushing queued items
    to Supabase. Failures are retried with exponential backoff.

    Thread-safe: Can be called from any thread.
    """

    def __init__(
        self,
        batch_size: int = 10,
        interval_seconds: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Initialize the sync service.

        Args:
            batch_size: Number of items to sync per batch
            interval_seconds: Seconds between sync cycles
            max_retries: Max retry attempts per item before dropping
        """
        self.batch_size = int(
            os.getenv("SUPABASE_SYNC_BATCH_SIZE", str(batch_size))
        )
        self.interval_seconds = float(
            os.getenv("SUPABASE_SYNC_INTERVAL_SECONDS", str(interval_seconds))
        )
        self.max_retries = int(
            os.getenv("SUPABASE_SYNC_MAX_RETRIES", str(max_retries))
        )

        # Check if sync is enabled
        self.enabled = os.getenv("SUPABASE_SYNC_ENABLED", "false").lower() == "true"

        # Thread-safe queue
        self._queue: queue.Queue[SyncItem] = queue.Queue()

        # Statistics
        self._synced_count = 0
        self._failed_count = 0
        self._last_sync_at: Optional[datetime] = None
        self._last_error: Optional[str] = None

        # Background thread
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # Supabase client (lazy loaded)
        self._client = None

        if self.enabled:
            logger.info(
                f"SupabaseSyncService enabled: batch_size={self.batch_size}, "
                f"interval={self.interval_seconds}s, max_retries={self.max_retries}"
            )
        else:
            logger.debug("SupabaseSyncService disabled (SUPABASE_SYNC_ENABLED != true)")

    def _get_client(self):
        """Lazy load the Supabase client."""
        if self._client is None:
            from aragora.persistence.supabase_client import SupabaseClient

            self._client = SupabaseClient()
        return self._client

    def start(self) -> None:
        """Start the background sync thread."""
        if not self.enabled:
            logger.debug("Sync service not enabled, skipping start")
            return

        if self._running:
            logger.warning("Sync service already running")
            return

        # Verify Supabase is configured
        client = self._get_client()
        if not client.is_configured:
            logger.warning(
                "Supabase not configured, sync service will not start. "
                "Set SUPABASE_URL and SUPABASE_KEY environment variables."
            )
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sync_loop,
            name="SupabaseSyncService",
            daemon=True,
        )
        self._thread.start()
        self._running = True
        logger.info("SupabaseSyncService started")

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the background sync thread.

        Args:
            timeout: Seconds to wait for thread to stop
        """
        if not self._running:
            return

        logger.info("Stopping SupabaseSyncService...")
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Sync thread did not stop gracefully")

        self._running = False
        logger.info("SupabaseSyncService stopped")

    def queue_debate(self, debate_data: Dict[str, Any]) -> bool:
        """
        Queue a debate result for sync.

        Args:
            debate_data: Dict with debate fields (id, task, consensus_reached, etc.)

        Returns:
            True if queued, False if sync is disabled
        """
        if not self.enabled:
            return False

        item = SyncItem(item_type=SyncItemType.DEBATE, data=debate_data)
        self._queue.put(item)
        logger.debug(f"Queued debate for sync: {debate_data.get('id', 'unknown')}")
        return True

    def queue_cycle(self, cycle_data: Dict[str, Any]) -> bool:
        """
        Queue a nomic cycle for sync.

        Args:
            cycle_data: Dict with cycle fields

        Returns:
            True if queued, False if sync is disabled
        """
        if not self.enabled:
            return False

        item = SyncItem(item_type=SyncItemType.CYCLE, data=cycle_data)
        self._queue.put(item)
        return True

    def queue_event(self, event_data: Dict[str, Any]) -> bool:
        """
        Queue a stream event for sync.

        Args:
            event_data: Dict with event fields

        Returns:
            True if queued, False if sync is disabled
        """
        if not self.enabled:
            return False

        item = SyncItem(item_type=SyncItemType.EVENT, data=event_data)
        self._queue.put(item)
        return True

    def queue_metrics(self, metrics_data: Dict[str, Any]) -> bool:
        """
        Queue agent metrics for sync.

        Args:
            metrics_data: Dict with metrics fields

        Returns:
            True if queued, False if sync is disabled
        """
        if not self.enabled:
            return False

        item = SyncItem(item_type=SyncItemType.METRICS, data=metrics_data)
        self._queue.put(item)
        return True

    def get_status(self) -> SyncStatus:
        """Get current sync service status."""
        return SyncStatus(
            enabled=self.enabled,
            running=self._running,
            queue_size=self._queue.qsize(),
            synced_count=self._synced_count,
            failed_count=self._failed_count,
            last_sync_at=self._last_sync_at,
            last_error=self._last_error,
        )

    def flush(self, timeout: float = 30.0) -> int:
        """
        Synchronously flush all queued items.

        Useful for shutdown or testing. Blocks until queue is empty or timeout.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            Number of items synced
        """
        if not self.enabled:
            return 0

        start = time.time()
        synced = 0

        while not self._queue.empty() and (time.time() - start) < timeout:
            batch = self._get_batch()
            if batch:
                synced += self._sync_batch(batch)

        return synced

    def _sync_loop(self) -> None:
        """Background thread main loop."""
        logger.debug("Sync loop started")

        while not self._stop_event.is_set():
            try:
                batch = self._get_batch()
                if batch:
                    self._sync_batch(batch)
                    self._last_sync_at = datetime.now()

            except Exception as e:
                logger.exception(f"Error in sync loop: {e}")
                self._last_error = str(e)

            # Wait for next interval or stop signal
            self._stop_event.wait(timeout=self.interval_seconds)

        logger.debug("Sync loop stopped")

    def _get_batch(self) -> List[SyncItem]:
        """Get a batch of items from the queue."""
        batch = []
        try:
            while len(batch) < self.batch_size:
                item = self._queue.get_nowait()
                batch.append(item)
        except queue.Empty:
            pass
        return batch

    def _sync_batch(self, batch: List[SyncItem]) -> int:
        """
        Sync a batch of items to Supabase.

        Returns:
            Number of successfully synced items
        """
        if not batch:
            return 0

        client = self._get_client()
        if not client.is_configured:
            # Re-queue items for later
            for item in batch:
                self._queue.put(item)
            return 0

        synced = 0
        failed_items = []

        # Create event loop for async calls
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            for item in batch:
                try:
                    success = loop.run_until_complete(self._sync_item(client, item))
                    if success:
                        synced += 1
                        self._synced_count += 1
                    else:
                        failed_items.append(item)
                except Exception as e:
                    logger.warning(f"Failed to sync {item.item_type.value}: {e}")
                    item.last_error = str(e)
                    failed_items.append(item)
        finally:
            loop.close()

        # Handle failures with retry
        for item in failed_items:
            item.retries += 1
            if item.retries < self.max_retries:
                self._queue.put(item)
                logger.debug(
                    f"Re-queued {item.item_type.value} for retry "
                    f"({item.retries}/{self.max_retries})"
                )
            else:
                self._failed_count += 1
                self._last_error = item.last_error
                logger.error(
                    f"Dropped {item.item_type.value} after {self.max_retries} retries: "
                    f"{item.last_error}"
                )

        if synced > 0:
            logger.debug(f"Synced {synced}/{len(batch)} items to Supabase")

        return synced

    async def _sync_item(self, client, item: SyncItem) -> bool:
        """Sync a single item to Supabase."""
        from aragora.persistence.models import (
            AgentMetrics,
            DebateArtifact,
            NomicCycle,
            StreamEvent,
        )

        try:
            if item.item_type == SyncItemType.DEBATE:
                # Convert dict to DebateArtifact
                artifact = DebateArtifact(
                    id=item.data.get("id"),
                    loop_id=item.data.get("loop_id", "default"),
                    cycle_number=item.data.get("cycle_number", 0),
                    phase=item.data.get("phase", "debate"),
                    task=item.data.get("task", ""),
                    agents=item.data.get("agents", []),
                    transcript=item.data.get("transcript", ""),
                    consensus_reached=item.data.get("consensus_reached", False),
                    confidence=item.data.get("confidence", 0.0),
                    winning_proposal=item.data.get("winning_proposal"),
                    vote_tally=item.data.get("vote_tally"),
                )
                result = await client.save_debate(artifact)
                return result is not None

            elif item.item_type == SyncItemType.CYCLE:
                cycle = NomicCycle(**item.data)
                result = await client.save_cycle(cycle)
                return result is not None

            elif item.item_type == SyncItemType.EVENT:
                event = StreamEvent(**item.data)
                result = await client.save_event(event)
                return result is not None

            elif item.item_type == SyncItemType.METRICS:
                metrics = AgentMetrics(**item.data)
                result = await client.save_metrics(metrics)
                return result is not None

            return False

        except Exception as e:
            logger.debug(f"Sync item error: {e}")
            raise


# Singleton instance
_sync_service: Optional[SupabaseSyncService] = None
_sync_service_lock = threading.Lock()


def get_sync_service() -> SupabaseSyncService:
    """
    Get the global sync service instance.

    Creates and starts the service if needed.
    Thread-safe.

    Returns:
        The singleton SupabaseSyncService
    """
    global _sync_service

    if _sync_service is None:
        with _sync_service_lock:
            if _sync_service is None:
                _sync_service = SupabaseSyncService()
                _sync_service.start()

    return _sync_service


def shutdown_sync_service(timeout: float = 5.0) -> None:
    """
    Shutdown the global sync service.

    Flushes remaining items and stops the background thread.

    Args:
        timeout: Seconds to wait for flush and stop
    """
    global _sync_service

    if _sync_service is not None:
        with _sync_service_lock:
            if _sync_service is not None:
                # Try to flush remaining items
                try:
                    _sync_service.flush(timeout=timeout / 2)
                except Exception as e:
                    logger.warning(f"Error flushing sync service: {e}")

                _sync_service.stop(timeout=timeout / 2)
                _sync_service = None
