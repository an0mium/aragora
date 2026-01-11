"""
Debate queue management for batch processing.

Provides a queue-based approach for processing multiple debates with:
- Priority ordering (higher priority debates run first)
- Concurrency limits (configurable max parallel debates)
- Progress tracking and status monitoring
- Webhook callbacks on completion

Usage:
    from aragora.server.debate_queue import DebateQueue, BatchRequest, BatchItem

    queue = DebateQueue(max_concurrent=3)

    batch = BatchRequest(
        items=[
            BatchItem(question="Question 1", agents="anthropic-api,openai-api"),
            BatchItem(question="Question 2", priority=10),  # Higher priority
        ],
        webhook_url="https://example.com/callback",
    )

    batch_id = await queue.submit_batch(batch)
    status = queue.get_batch_status(batch_id)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class BatchStatus(str, Enum):
    """Status of a batch request."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"  # Some items failed
    FAILED = "failed"
    CANCELLED = "cancelled"


class ItemStatus(str, Enum):
    """Status of an individual batch item."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchItem:
    """A single debate request within a batch."""
    question: str
    agents: str = "anthropic-api,openai-api,gemini"
    rounds: int = 3
    consensus: str = "majority"
    priority: int = 0  # Higher = runs first
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Populated during execution
    item_id: str = field(default_factory=lambda: f"item_{uuid.uuid4().hex[:8]}")
    status: ItemStatus = ItemStatus.QUEUED
    debate_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "item_id": self.item_id,
            "question": self.question,
            "agents": self.agents,
            "rounds": self.rounds,
            "consensus": self.consensus,
            "priority": self.priority,
            "metadata": self.metadata,
            "status": self.status.value,
            "debate_id": self.debate_id,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": (
                self.completed_at - self.started_at
                if self.completed_at and self.started_at
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchItem":
        """Create from dictionary (e.g., parsed JSON)."""
        return cls(
            question=data["question"],
            agents=data.get("agents", "anthropic-api,openai-api,gemini"),
            rounds=data.get("rounds", 3),
            consensus=data.get("consensus", "majority"),
            priority=data.get("priority", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BatchRequest:
    """A batch of debate requests to process."""
    items: List[BatchItem]
    webhook_url: Optional[str] = None  # Called when batch completes
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    max_parallel: Optional[int] = None  # Override queue's default

    # Populated during execution
    batch_id: str = field(default_factory=lambda: f"batch_{uuid.uuid4().hex[:12]}")
    status: BatchStatus = BatchStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        completed = sum(1 for i in self.items if i.status == ItemStatus.COMPLETED)
        failed = sum(1 for i in self.items if i.status == ItemStatus.FAILED)
        running = sum(1 for i in self.items if i.status == ItemStatus.RUNNING)
        queued = sum(1 for i in self.items if i.status == ItemStatus.QUEUED)

        return {
            "batch_id": self.batch_id,
            "status": self.status.value,
            "total_items": len(self.items),
            "completed": completed,
            "failed": failed,
            "running": running,
            "queued": queued,
            "progress_percent": round(100 * (completed + failed) / len(self.items), 1) if self.items else 0,
            "webhook_url": self.webhook_url,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": (
                self.completed_at - self.started_at
                if self.completed_at and self.started_at
                else None
            ),
            "items": [item.to_dict() for item in self.items],
        }

    def summary(self) -> Dict[str, Any]:
        """Get summary without individual items (for list endpoints)."""
        result = self.to_dict()
        del result["items"]
        return result


class DebateQueue:
    """
    Queue manager for batch debate processing.

    Handles:
    - Priority-based ordering of debates
    - Concurrency limits
    - Progress tracking
    - Webhook notifications

    Thread Safety:
        Uses asyncio primitives for coordination. The actual debate
        execution uses DebateController's thread pool.
    """

    def __init__(
        self,
        max_concurrent: int = 3,
        debate_executor: Optional[Callable] = None,
    ):
        """
        Initialize the debate queue.

        Args:
            max_concurrent: Maximum debates to run in parallel
            debate_executor: Callable that runs a single debate.
                            Signature: async (item: BatchItem) -> Dict[str, Any]
        """
        self.max_concurrent = max_concurrent
        self.debate_executor = debate_executor

        # Active batches by batch_id
        self._batches: Dict[str, BatchRequest] = {}

        # Processing state
        self._processing_lock = asyncio.Lock()
        self._active_count = 0

        # Background processing task
        self._processor_task: Optional[asyncio.Task] = None
        self._shutdown = False

    async def submit_batch(self, batch: BatchRequest) -> str:
        """
        Submit a batch of debates for processing.

        Args:
            batch: BatchRequest with items to process

        Returns:
            batch_id for tracking
        """
        if not batch.items:
            raise ValueError("Batch must contain at least one item")

        if len(batch.items) > 1000:
            raise ValueError("Batch cannot exceed 1000 items")

        # Sort items by priority (highest first)
        batch.items.sort(key=lambda x: x.priority, reverse=True)

        # Register batch
        self._batches[batch.batch_id] = batch

        logger.info(
            f"Batch {batch.batch_id} submitted with {len(batch.items)} items"
        )

        # Start processing if not already running
        if self._processor_task is None or self._processor_task.done():
            self._processor_task = asyncio.create_task(self._process_batches())

        return batch.batch_id

    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a batch."""
        batch = self._batches.get(batch_id)
        if batch:
            return batch.to_dict()
        return None

    def get_batch_summary(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a batch (without individual items)."""
        batch = self._batches.get(batch_id)
        if batch:
            return batch.summary()
        return None

    def list_batches(
        self,
        status: Optional[BatchStatus] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List batches, optionally filtered by status."""
        batches = list(self._batches.values())

        if status:
            batches = [b for b in batches if b.status == status]

        # Sort by creation time, newest first
        batches.sort(key=lambda x: x.created_at, reverse=True)

        return [b.summary() for b in batches[:limit]]

    async def cancel_batch(self, batch_id: str) -> bool:
        """
        Cancel a pending or processing batch.

        Debates already running will complete, but queued items
        will be cancelled.
        """
        batch = self._batches.get(batch_id)
        if not batch:
            return False

        if batch.status in (BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED):
            return False  # Already terminal

        batch.status = BatchStatus.CANCELLED

        # Cancel queued items
        for item in batch.items:
            if item.status == ItemStatus.QUEUED:
                item.status = ItemStatus.CANCELLED

        logger.info(f"Batch {batch_id} cancelled")
        return True

    async def _process_batches(self) -> None:
        """Background task that processes batches."""
        while not self._shutdown:
            # Find work to do
            work = await self._get_next_work()

            if not work:
                # No work available, wait a bit
                await asyncio.sleep(0.1)
                continue

            batch, item = work

            # Process item
            await self._process_item(batch, item)

    async def _get_next_work(self) -> Optional[tuple[BatchRequest, BatchItem]]:
        """Get the next item to process."""
        async with self._processing_lock:
            if self._active_count >= self.max_concurrent:
                return None

            # Find a batch with pending items
            for batch in self._batches.values():
                if batch.status in (BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED):
                    continue

                # Start batch if not started
                if batch.status == BatchStatus.PENDING:
                    batch.status = BatchStatus.PROCESSING
                    batch.started_at = time.time()

                # Find next queued item
                for item in batch.items:
                    if item.status == ItemStatus.QUEUED:
                        item.status = ItemStatus.RUNNING
                        item.started_at = time.time()
                        self._active_count += 1
                        return batch, item

            return None

    async def _process_item(self, batch: BatchRequest, item: BatchItem) -> None:
        """Process a single batch item."""
        try:
            if self.debate_executor:
                result = await self.debate_executor(item)
                item.result = result
                item.debate_id = result.get("debate_id")
                item.status = ItemStatus.COMPLETED
            else:
                # No executor configured, simulate for testing
                item.status = ItemStatus.FAILED
                item.error = "No debate executor configured"

        except Exception as e:
            logger.error(f"Failed to process item {item.item_id}: {e}")
            item.status = ItemStatus.FAILED
            item.error = str(e)
        finally:
            item.completed_at = time.time()

            async with self._processing_lock:
                self._active_count -= 1

            # Check if batch is complete
            await self._check_batch_completion(batch)

    async def _check_batch_completion(self, batch: BatchRequest) -> None:
        """Check if batch is complete and trigger webhook if so."""
        pending = sum(
            1 for item in batch.items
            if item.status in (ItemStatus.QUEUED, ItemStatus.RUNNING)
        )

        if pending > 0:
            return  # Still processing

        # Batch complete
        batch.completed_at = time.time()

        failed = sum(1 for item in batch.items if item.status == ItemStatus.FAILED)
        cancelled = sum(1 for item in batch.items if item.status == ItemStatus.CANCELLED)

        if cancelled == len(batch.items):
            batch.status = BatchStatus.CANCELLED
        elif failed == len(batch.items):
            batch.status = BatchStatus.FAILED
        elif failed > 0 or cancelled > 0:
            batch.status = BatchStatus.PARTIAL
        else:
            batch.status = BatchStatus.COMPLETED

        logger.info(
            f"Batch {batch.batch_id} completed: "
            f"{len(batch.items) - failed - cancelled}/{len(batch.items)} succeeded"
        )

        # Trigger webhook if configured
        if batch.webhook_url:
            await self._send_webhook(batch)

    async def _send_webhook(self, batch: BatchRequest) -> None:
        """Send webhook notification for completed batch."""
        try:
            import aiohttp

            payload = batch.to_dict()
            headers = {"Content-Type": "application/json"}
            headers.update(batch.webhook_headers)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    batch.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status >= 400:
                        logger.warning(
                            f"Webhook failed for batch {batch.batch_id}: "
                            f"status={response.status}"
                        )
                    else:
                        logger.info(f"Webhook sent for batch {batch.batch_id}")
        except ImportError:
            logger.warning("aiohttp not available for webhook")
        except Exception as e:
            logger.error(f"Webhook error for batch {batch.batch_id}: {e}")

    def cleanup_old_batches(self, max_age_hours: int = 24) -> int:
        """Remove batches older than max_age_hours."""
        cutoff = time.time() - (max_age_hours * 3600)
        to_remove = [
            batch_id
            for batch_id, batch in self._batches.items()
            if batch.created_at < cutoff
            and batch.status in (
                BatchStatus.COMPLETED,
                BatchStatus.FAILED,
                BatchStatus.CANCELLED,
            )
        ]

        for batch_id in to_remove:
            del self._batches[batch_id]

        return len(to_remove)

    async def shutdown(self) -> None:
        """Shutdown the queue processor."""
        self._shutdown = True
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass


# Global queue instance
_queue: Optional[DebateQueue] = None
_queue_lock = asyncio.Lock()


async def get_debate_queue() -> DebateQueue:
    """Get the global debate queue instance."""
    global _queue

    async with _queue_lock:
        if _queue is None:
            from aragora.config import MAX_CONCURRENT_DEBATES
            _queue = DebateQueue(max_concurrent=MAX_CONCURRENT_DEBATES)
        return _queue


def get_debate_queue_sync() -> Optional[DebateQueue]:
    """Get the global debate queue instance (sync version)."""
    return _queue


__all__ = [
    "DebateQueue",
    "BatchRequest",
    "BatchItem",
    "BatchStatus",
    "ItemStatus",
    "get_debate_queue",
    "get_debate_queue_sync",
]
