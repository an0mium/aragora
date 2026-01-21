"""
Routing Job Queue Worker.

Processes routing jobs to deliver debate results back to their
originating channels, enabling:
- Restart recovery after server crashes
- Retry logic on transient failures (platform API errors)
- Priority-based delivery scheduling

Usage:
    from aragora.queue.workers.routing_worker import RoutingWorker

    worker = RoutingWorker()
    await worker.start()  # Starts processing loop
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

from aragora.storage.job_queue_store import (
    QueuedJob,
    get_job_store,
)

logger = logging.getLogger(__name__)

# Job type constants
JOB_TYPE_ROUTING = "routing"
JOB_TYPE_ROUTING_DEBATE = "routing_debate"
JOB_TYPE_ROUTING_EMAIL = "routing_email"


class RoutingWorker:
    """
    Worker that processes routing jobs to deliver results to originating channels.

    Features:
    - Polls job queue for pending routing jobs
    - Routes results back via debate_origin.route_debate_result
    - Handles failures with automatic retry
    - Supports graceful shutdown
    """

    def __init__(
        self,
        worker_id: Optional[str] = None,
        poll_interval: float = 2.0,
        max_concurrent: int = 5,
        retry_delay_seconds: float = 30.0,
    ):
        """
        Initialize routing worker.

        Args:
            worker_id: Unique worker identifier (auto-generated if not provided)
            poll_interval: Seconds between queue polls when idle
            max_concurrent: Maximum concurrent routing jobs
            retry_delay_seconds: Delay before retrying failed delivery
        """
        self.worker_id = worker_id or f"routing-worker-{os.getpid()}"
        self.poll_interval = poll_interval
        self.max_concurrent = max_concurrent
        self.retry_delay_seconds = retry_delay_seconds

        self._running = False
        self._active_jobs: dict[str, asyncio.Task] = {}
        self._store = get_job_store()

    async def start(self) -> None:
        """Start the worker processing loop."""
        self._running = True
        logger.info(f"[{self.worker_id}] Starting routing worker")

        job_types = [
            JOB_TYPE_ROUTING,
            JOB_TYPE_ROUTING_DEBATE,
            JOB_TYPE_ROUTING_EMAIL,
        ]

        while self._running:
            try:
                # Clean up completed tasks
                self._cleanup_completed_tasks()

                # Check if we can take more jobs
                if len(self._active_jobs) >= self.max_concurrent:
                    await asyncio.sleep(0.5)
                    continue

                # Dequeue next job
                job = await self._store.dequeue(
                    worker_id=self.worker_id,
                    job_types=job_types,
                )

                if job:
                    # Start processing job
                    task = asyncio.create_task(
                        self._process_job(job),
                        name=f"routing-{job.id}",
                    )
                    self._active_jobs[job.id] = task
                    logger.info(f"[{self.worker_id}] Started routing job {job.id}")
                else:
                    # No jobs available, wait before polling again
                    await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                logger.info(f"[{self.worker_id}] Worker cancelled")
                break
            except Exception as e:
                logger.error(f"[{self.worker_id}] Worker error: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)

        # Wait for active jobs to complete on shutdown
        if self._active_jobs:
            logger.info(
                f"[{self.worker_id}] Waiting for {len(self._active_jobs)} active jobs"
            )
            await asyncio.gather(*self._active_jobs.values(), return_exceptions=True)

        logger.info(f"[{self.worker_id}] Worker stopped")

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info(f"[{self.worker_id}] Stopping worker")
        self._running = False

    def _cleanup_completed_tasks(self) -> None:
        """Remove completed tasks from tracking dict."""
        completed = [
            job_id for job_id, task in self._active_jobs.items() if task.done()
        ]
        for job_id in completed:
            task = self._active_jobs.pop(job_id)
            if task.exception():
                logger.warning(
                    f"[{self.worker_id}] Job {job_id} failed: {task.exception()}"
                )

    async def _process_job(self, job: QueuedJob) -> None:
        """Process a single routing job."""
        start_time = time.time()

        try:
            job_type = job.job_type
            logger.info(f"[{self.worker_id}] Processing {job_type} job {job.id}")

            # Route to appropriate handler
            if job_type in (JOB_TYPE_ROUTING_DEBATE, JOB_TYPE_ROUTING):
                success = await self._route_debate_result(job)
            elif job_type == JOB_TYPE_ROUTING_EMAIL:
                success = await self._route_email_result(job)
            else:
                raise ValueError(f"Unknown job type: {job_type}")

            # Mark job as completed or failed
            duration = time.time() - start_time
            if success:
                await self._store.complete(
                    job.id,
                    result={
                        "status": "delivered",
                        "duration_seconds": duration,
                    },
                )
                logger.info(
                    f"[{self.worker_id}] Delivered job {job.id} in {duration:.1f}s"
                )
            else:
                # Delivery failed, retry if attempts remain
                should_retry = job.attempts < job.max_attempts
                await self._store.fail(
                    job.id,
                    error="Delivery failed to platform",
                    should_retry=should_retry,
                )
                if should_retry:
                    logger.info(
                        f"[{self.worker_id}] Job {job.id} will retry "
                        f"(attempt {job.attempts}/{job.max_attempts})"
                    )

        except Exception as e:
            logger.error(
                f"[{self.worker_id}] Job {job.id} failed: {e}",
                exc_info=True,
            )

            # Check if we should retry
            should_retry = job.attempts < job.max_attempts
            await self._store.fail(
                job.id,
                error=str(e),
                should_retry=should_retry,
            )

            if should_retry:
                logger.info(
                    f"[{self.worker_id}] Job {job.id} will retry "
                    f"(attempt {job.attempts}/{job.max_attempts})"
                )

    async def _route_debate_result(self, job: QueuedJob) -> bool:
        """Route a debate result back to its originating platform."""
        from aragora.server.debate_origin import route_debate_result

        payload = job.payload
        debate_id = payload.get("debate_id")
        result = payload.get("result", {})
        include_voice = payload.get("include_voice", False)

        if not debate_id:
            raise ValueError("Missing debate_id in payload")

        return await route_debate_result(
            debate_id=debate_id,
            result=result,
            include_voice=include_voice,
        )

    async def _route_email_result(self, job: QueuedJob) -> bool:
        """Route a result via email reply."""
        from aragora.integrations.email_reply_loop import send_debate_result_email

        payload = job.payload
        debate_id = payload.get("debate_id")
        result = payload.get("result", {})
        recipient_email = payload.get("recipient_email")

        if not debate_id or not recipient_email:
            raise ValueError("Missing debate_id or recipient_email in payload")

        return await send_debate_result_email(
            debate_id=debate_id,
            result=result,
            recipient_email=recipient_email,
        )


async def enqueue_routing_job(
    job_id: str,
    debate_id: str,
    result: Dict[str, Any],
    job_type: str = JOB_TYPE_ROUTING_DEBATE,
    include_voice: bool = False,
    recipient_email: Optional[str] = None,
    user_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    priority: int = 0,
) -> QueuedJob:
    """
    Enqueue a routing job for durable delivery.

    Args:
        job_id: Unique job identifier
        debate_id: Debate ID to route
        result: Debate result dict to deliver
        job_type: Type of routing job
        include_voice: Whether to include TTS voice message
        recipient_email: Email for email routing jobs
        user_id: Optional user ID
        workspace_id: Optional workspace ID
        priority: Job priority (higher = more urgent)

    Returns:
        The queued job
    """
    payload = {
        "debate_id": debate_id,
        "result": result,
        "include_voice": include_voice,
    }
    if recipient_email:
        payload["recipient_email"] = recipient_email

    job = QueuedJob(
        id=job_id,
        job_type=job_type,
        payload=payload,
        priority=priority,
        user_id=user_id,
        workspace_id=workspace_id,
        max_attempts=5,  # More retries for transient platform failures
    )

    store = get_job_store()
    await store.enqueue(job)

    logger.info(f"Enqueued routing job: {job_id} ({job_type})")
    return job


async def recover_interrupted_routing() -> int:
    """
    Recover interrupted routing jobs after server restart.

    Returns:
        Number of jobs recovered
    """
    store = get_job_store()
    recovered = 0

    try:
        job_types = [
            JOB_TYPE_ROUTING,
            JOB_TYPE_ROUTING_DEBATE,
            JOB_TYPE_ROUTING_EMAIL,
        ]

        for job_type in job_types:
            stale_recovered = await store.recover_stale_jobs(
                stale_threshold_seconds=300.0,
                job_types=[job_type],
            )
            if stale_recovered:
                logger.info(f"Recovered {stale_recovered} stale {job_type} jobs")
                recovered += stale_recovered

    except Exception as e:
        logger.error(f"Error recovering routing jobs: {e}", exc_info=True)

    if recovered:
        logger.info(f"Recovered {recovered} interrupted routing jobs")

    return recovered


__all__ = [
    "RoutingWorker",
    "enqueue_routing_job",
    "recover_interrupted_routing",
    "JOB_TYPE_ROUTING",
    "JOB_TYPE_ROUTING_DEBATE",
    "JOB_TYPE_ROUTING_EMAIL",
]
