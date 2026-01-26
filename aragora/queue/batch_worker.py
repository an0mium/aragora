"""
Batch Explainability Worker.

Background worker for processing batch explainability jobs:
- Processes multiple debates in a batch
- Generates explanations for each debate
- Tracks progress and handles failures
- Emits events for monitoring
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

from aragora.queue.base import Job, JobQueue, JobStatus
from aragora.queue.config import get_queue_config

logger = logging.getLogger(__name__)

# Type alias for explanation generator
ExplainGenerator = Callable[[str, Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]


@dataclass
class BatchJobProgress:
    """Progress tracking for a batch job."""

    job_id: str
    total: int
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    started_at: float = field(default_factory=time.time)
    results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def completion_percent(self) -> float:
        """Get completion percentage."""
        if self.total == 0:
            return 100.0
        return (self.processed / self.total) * 100

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.started_at


class BatchExplainabilityWorker:
    """
    Worker for processing batch explainability jobs.

    Features:
    - Concurrent processing of debates within a batch
    - Progress tracking and partial results
    - Graceful shutdown with checkpoint support
    - Metrics emission for monitoring
    """

    QUEUE_NAME = "batch_explainability"

    def __init__(
        self,
        queue: JobQueue,
        worker_id: str,
        explain_generator: ExplainGenerator,
        max_concurrent_debates: int = 5,
        max_concurrent_batches: int = 2,
    ) -> None:
        """
        Initialize the batch worker.

        Args:
            queue: The job queue to process from
            worker_id: Unique identifier for this worker
            explain_generator: Function to generate explanations
            max_concurrent_debates: Max debates to process in parallel per batch
            max_concurrent_batches: Max batches to process in parallel
        """
        self._queue = queue
        self._worker_id = worker_id
        self._explain_generator = explain_generator
        self._max_concurrent_debates = max_concurrent_debates
        self._max_concurrent_batches = max_concurrent_batches
        self._config = get_queue_config()

        self._running = False
        self._tasks: set[asyncio.Task[None]] = set()
        self._batch_semaphore = asyncio.Semaphore(max_concurrent_batches)
        self._shutdown_event = asyncio.Event()

        # Progress tracking
        self._active_batches: Dict[str, BatchJobProgress] = {}

        # Metrics
        self._batches_processed = 0
        self._debates_processed = 0
        self._debates_failed = 0
        self._start_time: Optional[float] = None

    @property
    def worker_id(self) -> str:
        """Get the worker ID."""
        return self._worker_id

    @property
    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._running

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "worker_id": self._worker_id,
            "is_running": self._running,
            "uptime_seconds": uptime,
            "batches_processed": self._batches_processed,
            "debates_processed": self._debates_processed,
            "debates_failed": self._debates_failed,
            "active_batches": len(self._active_batches),
            "queue_name": self.QUEUE_NAME,
        }

    def get_batch_progress(self, job_id: str) -> Optional[BatchJobProgress]:
        """Get progress for a specific batch job."""
        return self._active_batches.get(job_id)

    async def start(self) -> None:
        """Start the worker."""
        if self._running:
            logger.warning(f"Worker {self._worker_id} already running")
            return

        logger.info(f"Starting batch explainability worker {self._worker_id}")
        self._running = True
        self._start_time = time.time()
        self._shutdown_event.clear()

        # Start the main processing loop
        asyncio.create_task(self._process_loop())

    async def stop(self, timeout: float = 30.0) -> None:
        """
        Stop the worker gracefully.

        Args:
            timeout: Maximum seconds to wait for active jobs to complete
        """
        if not self._running:
            return

        logger.info(f"Stopping batch worker {self._worker_id}")
        self._running = False
        self._shutdown_event.set()

        # Wait for active tasks with timeout
        if self._tasks:
            logger.info(f"Waiting for {len(self._tasks)} active batch jobs")
            done, pending = await asyncio.wait(
                self._tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED,
            )

            if pending:
                logger.warning(f"Cancelling {len(pending)} pending batch jobs")
                for task in pending:
                    task.cancel()

        logger.info(f"Batch worker {self._worker_id} stopped")

    async def _process_loop(self) -> None:
        """Main processing loop."""
        poll_interval = self._config.poll_interval_seconds

        while self._running:
            try:
                # Check for available capacity
                if self._batch_semaphore.locked():
                    await asyncio.sleep(poll_interval)
                    continue

                # Try to get a job from the queue
                job = await self._queue.dequeue(
                    queue_name=self.QUEUE_NAME,
                    worker_id=self._worker_id,
                )

                if job is None:
                    await asyncio.sleep(poll_interval)
                    continue

                # Process the batch job
                await self._batch_semaphore.acquire()
                task = asyncio.create_task(self._process_batch(job))
                self._tasks.add(task)
                task.add_done_callback(self._on_task_done)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch worker loop: {e}", exc_info=True)
                await asyncio.sleep(poll_interval)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        """Callback when a task completes."""
        self._tasks.discard(task)
        self._batch_semaphore.release()

        if task.exception():
            logger.error(f"Batch task failed: {task.exception()}")

    async def _process_batch(self, job: Job) -> None:
        """Process a single batch job."""
        job_id = job.id
        payload = job.payload

        debate_ids: List[str] = payload.get("debate_ids", [])
        options: Dict[str, Any] = payload.get("options", {})

        logger.info(f"Processing batch job {job_id} with {len(debate_ids)} debates")

        # Initialize progress tracking
        progress = BatchJobProgress(
            job_id=job_id,
            total=len(debate_ids),
        )
        self._active_batches[job_id] = progress

        try:
            # Process debates concurrently with rate limiting
            semaphore = asyncio.Semaphore(self._max_concurrent_debates)
            tasks = []

            for debate_id in debate_ids:
                task = asyncio.create_task(
                    self._process_debate(debate_id, options, progress, semaphore)
                )
                tasks.append(task)

            # Wait for all debates to complete
            await asyncio.gather(*tasks, return_exceptions=True)

            # Mark job as complete
            final_status = (
                JobStatus.COMPLETED
                if progress.failed == 0
                else JobStatus.FAILED
                if progress.succeeded == 0
                else JobStatus.COMPLETED
            )

            await self._queue.complete(
                job_id=job_id,
                result={
                    "total": progress.total,
                    "succeeded": progress.succeeded,
                    "failed": progress.failed,
                    "results": progress.results,
                    "errors": progress.errors,
                    "elapsed_seconds": progress.elapsed_seconds,
                },
                status=final_status,
            )

            self._batches_processed += 1
            logger.info(
                f"Batch job {job_id} completed: {progress.succeeded}/{progress.total} succeeded"
            )

        except Exception as e:
            logger.error(f"Batch job {job_id} failed: {e}", exc_info=True)
            await self._queue.fail(job_id=job_id, error=str(e))
        finally:
            del self._active_batches[job_id]

    async def _process_debate(
        self,
        debate_id: str,
        options: Dict[str, Any],
        progress: BatchJobProgress,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Process a single debate within a batch."""
        async with semaphore:
            start_time = time.time()

            try:
                result = await self._explain_generator(debate_id, options)

                progress.results.append(
                    {
                        "debate_id": debate_id,
                        "status": "success",
                        "explanation": result,
                        "processing_time_ms": (time.time() - start_time) * 1000,
                    }
                )
                progress.succeeded += 1
                self._debates_processed += 1

            except Exception as e:
                progress.errors.append(
                    {
                        "debate_id": debate_id,
                        "status": "error",
                        "error": str(e),
                        "processing_time_ms": (time.time() - start_time) * 1000,
                    }
                )
                progress.failed += 1
                self._debates_failed += 1
                logger.warning(f"Failed to explain debate {debate_id}: {e}")

            progress.processed += 1


async def create_batch_job(
    queue: JobQueue,
    debate_ids: List[str],
    options: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    priority: int = 0,
) -> Job:
    """
    Create a batch explainability job.

    Args:
        queue: The job queue
        debate_ids: List of debate IDs to process
        options: Processing options
        user_id: User requesting the batch
        priority: Job priority

    Returns:
        The created job
    """
    job = Job(
        payload={
            "debate_ids": debate_ids,
            "options": options or {},
            "user_id": user_id,
        },
        priority=priority,
        metadata={
            "type": "batch_explainability",
            "debate_count": len(debate_ids),
        },
    )

    await queue.enqueue(
        job=job,
        queue_name=BatchExplainabilityWorker.QUEUE_NAME,
    )

    return job


__all__ = [
    "BatchExplainabilityWorker",
    "BatchJobProgress",
    "create_batch_job",
]
