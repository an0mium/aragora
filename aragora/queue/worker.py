"""
Debate worker for processing jobs from the queue.

Provides a worker pattern for horizontal scaling with:
- Concurrent job processing
- Graceful shutdown
- Health reporting
- Stale job recovery
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import Any, Callable, Coroutine, Dict, Optional

from aragora.exceptions import InfrastructureError
from aragora.queue.base import Job, JobQueue, JobStatus
from aragora.queue.config import get_queue_config
from aragora.queue.retry import RetryPolicy, is_retryable_error

logger = logging.getLogger(__name__)

# Type alias for the debate executor function
DebateExecutor = Callable[[Job], Coroutine[Any, Any, Dict[str, Any]]]


class DebateWorker:
    """
    Worker that processes debate jobs from the queue.

    Designed for horizontal scaling:
    - Each worker has a unique ID
    - Uses consumer groups for work distribution
    - Handles graceful shutdown
    - Reports health via heartbeat
    - Recovers stale jobs from dead workers
    """

    def __init__(
        self,
        queue: JobQueue,
        worker_id: str,
        executor: DebateExecutor,
        max_concurrent: int = 3,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> None:
        """
        Initialize the worker.

        Args:
            queue: The job queue to process from
            worker_id: Unique identifier for this worker
            executor: Async function to execute debates
            max_concurrent: Maximum concurrent jobs to process
            retry_policy: Policy for retrying failed jobs
        """
        self._queue = queue
        self._worker_id = worker_id
        self._executor = executor
        self._max_concurrent = max_concurrent
        self._retry_policy = retry_policy or RetryPolicy.from_config()
        self._config = get_queue_config()

        self._running = False
        self._tasks: set[asyncio.Task[None]] = set()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._shutdown_event = asyncio.Event()

        # Metrics
        self._jobs_processed = 0
        self._jobs_failed = 0
        self._start_time: Optional[float] = None

    @property
    def worker_id(self) -> str:
        """Get the worker ID."""
        return self._worker_id

    @property
    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._running

    @property
    def active_jobs(self) -> int:
        """Get number of currently processing jobs."""
        return self._max_concurrent - self._semaphore._value

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "worker_id": self._worker_id,
            "running": self._running,
            "active_jobs": self.active_jobs,
            "max_concurrent": self._max_concurrent,
            "jobs_processed": self._jobs_processed,
            "jobs_failed": self._jobs_failed,
            "uptime_seconds": uptime,
        }

    async def start(self) -> None:
        """
        Start the worker.

        Runs until stop() is called or a signal is received.
        """
        if self._running:
            logger.warning(f"Worker {self._worker_id} is already running")
            return

        self._running = True
        self._start_time = time.time()
        self._shutdown_event.clear()

        # Set up signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_signal)

        logger.info(
            f"Worker {self._worker_id} started (max_concurrent={self._max_concurrent})"
        )

        try:
            # Start background tasks
            claim_task = asyncio.create_task(self._claim_stale_jobs_loop())
            self._tasks.add(claim_task)

            # Main processing loop
            while self._running:
                try:
                    # Wait for a slot
                    await self._semaphore.acquire()

                    if not self._running:
                        self._semaphore.release()
                        break

                    # Try to get a job
                    job = await self._queue.dequeue(
                        self._worker_id,
                        timeout_ms=self._config.worker_block_ms,
                    )

                    if job is None:
                        self._semaphore.release()
                        continue

                    # Process the job in background
                    task = asyncio.create_task(self._process_job(job))
                    self._tasks.add(task)
                    task.add_done_callback(lambda t: self._tasks.discard(t))

                except asyncio.CancelledError:
                    self._semaphore.release()
                    break
                except Exception as e:
                    self._semaphore.release()
                    logger.error(f"Error in worker loop: {e}", exc_info=True)
                    await asyncio.sleep(1)  # Brief pause before retrying

        finally:
            # Clean up signal handlers
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.remove_signal_handler(sig)

            self._running = False
            logger.info(f"Worker {self._worker_id} stopped")

    async def stop(self, timeout: float = 30.0) -> None:
        """
        Stop the worker gracefully.

        Waits for current jobs to complete up to the timeout.

        Args:
            timeout: Maximum time to wait for jobs to complete
        """
        if not self._running:
            return

        logger.info(f"Worker {self._worker_id} stopping (timeout={timeout}s)")
        self._running = False
        self._shutdown_event.set()

        # Wait for active tasks with timeout
        if self._tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for {len(self._tasks)} tasks")
                # Cancel remaining tasks
                for task in self._tasks:
                    task.cancel()

        await self._queue.close()

    def _handle_signal(self) -> None:
        """Handle shutdown signal."""
        logger.info(f"Worker {self._worker_id} received shutdown signal")
        self._running = False
        self._shutdown_event.set()

    async def _process_job(self, job: Job) -> None:
        """
        Process a single job.

        Args:
            job: The job to process
        """
        try:
            logger.info(f"Processing job {job.id} (attempt {job.attempts})")
            start_time = time.time()

            # Execute the debate
            result = await self._executor(job)

            # Mark completed
            job.mark_completed(result)
            await self._queue.ack(job.id)

            duration = time.time() - start_time
            self._jobs_processed += 1

            logger.info(f"Job {job.id} completed in {duration:.2f}s")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Job {job.id} failed: {error_msg}", exc_info=True)

            job.mark_retrying(error_msg)
            self._jobs_failed += 1

            # Determine if we should retry
            should_retry = (
                is_retryable_error(e) and
                self._retry_policy.should_retry(job.attempts, e)
            )

            if should_retry:
                # Calculate retry delay
                delay = self._retry_policy.get_delay(job.attempts - 1)
                logger.info(
                    f"Job {job.id} will retry in {delay:.1f}s "
                    f"(attempt {job.attempts}/{job.max_attempts})"
                )
                # Leave in pending list for retry
                await self._queue.nack(job.id, requeue=True)
            else:
                # Mark as permanently failed
                job.mark_failed(error_msg)
                await self._queue.nack(job.id, requeue=False)
                logger.warning(f"Job {job.id} permanently failed after {job.attempts} attempts")

        finally:
            self._semaphore.release()

    async def _claim_stale_jobs_loop(self) -> None:
        """
        Periodically claim stale jobs from dead workers.
        """
        while self._running:
            try:
                # Wait between claim attempts
                await asyncio.sleep(self._config.claim_idle_ms / 1000 / 2)

                if not self._running:
                    break

                # Claim stale jobs
                claimed = await self._queue.claim_stale_jobs(self._config.claim_idle_ms)
                if claimed > 0:
                    logger.info(f"Claimed {claimed} stale jobs")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error claiming stale jobs: {e}")


async def create_default_executor() -> DebateExecutor:
    """
    Create a default debate executor.

    This imports the debate infrastructure and creates an executor
    that runs debates using the Arena.

    Returns:
        An async function that executes debate jobs
    """

    async def execute_debate(job: Job) -> Dict[str, Any]:
        """Execute a debate from a job."""
        # Import here to avoid circular imports
        from aragora.queue.job import get_debate_payload, DebateResult

        payload = get_debate_payload(job)

        # Import debate infrastructure
        try:
            from aragora.debate.orchestrator import Arena
            from aragora.core import Environment, DebateProtocol
        except ImportError as e:
            raise InfrastructureError(f"Debate infrastructure not available: {e}")

        # Create environment and protocol
        env = Environment(task=payload.question)
        protocol = DebateProtocol(
            rounds=payload.rounds,
            consensus=payload.consensus,
        )

        # Run debate
        start_time = time.time()
        arena = Arena(env, agents=payload.agents, protocol=protocol)
        result = await arena.run()

        duration = time.time() - start_time

        # Build result
        debate_result = DebateResult(
            debate_id=result.debate_id if hasattr(result, "debate_id") else job.id,
            consensus_reached=result.consensus_reached if hasattr(result, "consensus_reached") else False,
            final_answer=result.final_answer if hasattr(result, "final_answer") else None,
            confidence=result.confidence if hasattr(result, "confidence") else 0.0,
            rounds_used=result.rounds_used if hasattr(result, "rounds_used") else payload.rounds,
            participants=payload.agents,
            duration_seconds=duration,
        )

        return debate_result.to_dict()

    return execute_debate
