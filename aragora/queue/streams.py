"""
Redis Streams implementation of the job queue.

Uses Redis Streams for reliable, scalable job processing with:
- XADD for enqueueing with automatic ID generation
- XREADGROUP for consumer groups (enables horizontal scaling)
- XPENDING for tracking unacknowledged jobs
- XCLAIM for handling stale jobs from dead workers
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from aragora.queue.base import Job, JobQueue, JobStatus
from aragora.queue.config import QueueConfig, get_queue_config
from aragora.queue.status import JobStatusTracker

logger = logging.getLogger(__name__)


class RedisStreamsQueue(JobQueue):
    """
    Redis Streams-based job queue.

    Implements the JobQueue interface using Redis Streams for reliable
    message delivery with consumer groups for horizontal scaling.

    Key Features:
    - At-least-once delivery guarantee
    - Consumer groups for work distribution
    - Automatic recovery of stale jobs from dead workers
    - Job status tracking in Redis hashes
    """

    def __init__(
        self,
        redis_client: Any,
        consumer_name: str,
        config: Optional[QueueConfig] = None,
    ) -> None:
        """
        Initialize the Redis Streams queue.

        Args:
            redis_client: An async Redis client (redis.asyncio.Redis)
            consumer_name: Unique name for this consumer (worker ID)
            config: Optional queue configuration
        """
        self._redis = redis_client
        self._consumer_name = consumer_name
        self._config = config or get_queue_config()
        self._status_tracker = JobStatusTracker(redis_client)
        self._initialized = False

    @property
    def stream_key(self) -> str:
        """Get the stream key."""
        return self._config.stream_key

    @property
    def group_name(self) -> str:
        """Get the consumer group name."""
        return self._config.consumer_group

    async def _ensure_initialized(self) -> None:
        """Ensure the stream and consumer group exist."""
        if self._initialized:
            return

        try:
            # Create consumer group (creates stream if needed)
            await self._redis.xgroup_create(
                self.stream_key,
                self.group_name,
                id="0",
                mkstream=True,
            )
            logger.info(f"Created consumer group {self.group_name} on {self.stream_key}")
        except Exception as e:
            # Group already exists (BUSYGROUP error)
            if "BUSYGROUP" not in str(e):
                raise
            logger.debug(f"Consumer group {self.group_name} already exists")

        self._initialized = True

    async def enqueue(self, job: Job, priority: int = 0) -> str:
        """
        Add a job to the queue.

        Args:
            job: The job to enqueue
            priority: Job priority (stored in job, not used for ordering in streams)

        Returns:
            The job ID
        """
        await self._ensure_initialized()

        job.priority = priority

        # Serialize job to stream entry
        entry = {
            "job_id": job.id,
            "payload": json.dumps(job.payload),
            "priority": str(priority),
            "max_attempts": str(job.max_attempts),
            "created_at": str(job.created_at),
            "metadata": json.dumps(job.metadata),
        }

        # Add to stream
        await self._redis.xadd(self.stream_key, entry)

        # Track job status
        await self._status_tracker.create(job)

        logger.debug(f"Enqueued job {job.id}")
        return job.id

    async def dequeue(self, worker_id: str, timeout_ms: int = 5000) -> Optional[Job]:
        """
        Get the next job from the queue.

        Uses XREADGROUP to get work from the consumer group.

        Args:
            worker_id: ID of the worker requesting work
            timeout_ms: How long to block waiting for a job

        Returns:
            A job if available, None otherwise
        """
        await self._ensure_initialized()

        try:
            # Read from consumer group
            messages = await self._redis.xreadgroup(
                groupname=self.group_name,
                consumername=worker_id,
                streams={self.stream_key: ">"},  # Only new messages
                count=1,
                block=timeout_ms,
            )

            if not messages:
                return None

            # Parse the message
            stream_name, entries = messages[0]
            if not entries:
                return None

            message_id, data = entries[0]

            # Decode bytes if needed
            if isinstance(message_id, bytes):
                message_id = message_id.decode()
            if isinstance(next(iter(data.keys()), ""), bytes):
                data = {k.decode(): v.decode() for k, v in data.items()}

            # Reconstruct job
            job = Job(
                id=data["job_id"],
                payload=json.loads(data["payload"]),
                priority=int(data.get("priority", 0)),
                max_attempts=int(data.get("max_attempts", 3)),
                created_at=float(data.get("created_at", datetime.now().timestamp())),
                metadata=json.loads(data.get("metadata", "{}")),
            )

            # Store message ID for acknowledgment
            job.metadata["_stream_message_id"] = message_id

            # Update status to processing
            job.mark_processing(worker_id)
            await self._status_tracker.update_status(
                job.id,
                JobStatus.PROCESSING,
                worker_id=worker_id,
                attempts=job.attempts,
            )

            logger.debug(f"Dequeued job {job.id} for worker {worker_id}")
            return job

        except Exception as e:
            logger.error(f"Error dequeuing job: {e}")
            return None

    async def ack(self, job_id: str) -> bool:
        """
        Acknowledge successful processing of a job.

        Args:
            job_id: The job ID to acknowledge

        Returns:
            True if acknowledged, False if job not found
        """
        # Get job to find message ID
        job = await self._status_tracker.get_job(job_id)
        if job is None:
            return False

        message_id = job.metadata.get("_stream_message_id")
        if message_id:
            await self._redis.xack(self.stream_key, self.group_name, message_id)

        # Update status
        await self._status_tracker.update_status(job_id, JobStatus.COMPLETED)

        logger.debug(f"Acknowledged job {job_id}")
        return True

    async def nack(self, job_id: str, requeue: bool = True) -> bool:
        """
        Negative acknowledge - job processing failed.

        Args:
            job_id: The job ID
            requeue: Whether to requeue for retry

        Returns:
            True if processed, False if job not found
        """
        job = await self._status_tracker.get_job(job_id)
        if job is None:
            return False

        message_id = job.metadata.get("_stream_message_id")

        if requeue and job.should_retry():
            # Mark as retrying - the job stays in the pending list
            await self._status_tracker.update_status(
                job_id,
                JobStatus.RETRYING,
                error=job.error,
            )
            logger.debug(f"Job {job_id} marked for retry (attempt {job.attempts})")
        else:
            # Mark as failed and acknowledge to remove from pending
            await self._status_tracker.update_status(
                job_id,
                JobStatus.FAILED,
                error=job.error,
            )
            if message_id:
                await self._redis.xack(self.stream_key, self.group_name, message_id)
            logger.debug(f"Job {job_id} marked as failed")

        return True

    async def get_status(self, job_id: str) -> Optional[Job]:
        """
        Get the current status of a job.

        Args:
            job_id: The job ID

        Returns:
            The job if found, None otherwise
        """
        return await self._status_tracker.get_job(job_id)

    async def cancel(self, job_id: str) -> bool:
        """
        Cancel a pending job.

        Args:
            job_id: The job ID to cancel

        Returns:
            True if cancelled, False if not found or already processing
        """
        job = await self._status_tracker.get_job(job_id)
        if job is None:
            return False

        # Can only cancel pending/retrying jobs
        if job.status not in (JobStatus.PENDING, JobStatus.RETRYING):
            return False

        # Mark as cancelled
        await self._status_tracker.update_status(job_id, JobStatus.CANCELLED)

        # Acknowledge to remove from stream
        message_id = job.metadata.get("_stream_message_id")
        if message_id:
            await self._redis.xack(self.stream_key, self.group_name, message_id)

        logger.info(f"Cancelled job {job_id}")
        return True

    async def get_queue_stats(self) -> Dict[str, int]:
        """
        Get queue statistics.

        Returns:
            Dict with counts for pending, processing, completed, failed jobs
        """
        # Get counts from status tracker
        counts = await self._status_tracker.get_counts_by_status()

        # Get stream length
        stream_length = await self._redis.xlen(self.stream_key)

        # Get pending count from consumer group
        try:
            pending_info = await self._redis.xpending(self.stream_key, self.group_name)
            pending_count = pending_info.get("pending", 0) if pending_info else 0
        except Exception as e:
            logger.debug(f"Could not get pending count for {self.stream_key}: {e}")
            pending_count = 0

        return {
            "stream_length": stream_length,
            "pending_in_group": pending_count,
            **counts,
        }

    async def claim_stale_jobs(self, idle_ms: int) -> int:
        """
        Claim jobs from dead workers.

        Uses XCLAIM to take over jobs that have been idle for too long,
        indicating the original worker has died.

        Args:
            idle_ms: Minimum idle time to consider a job stale

        Returns:
            Number of jobs claimed
        """
        await self._ensure_initialized()

        try:
            # Get pending messages info
            pending_info = await self._redis.xpending_range(
                self.stream_key,
                self.group_name,
                min="-",
                max="+",
                count=100,
            )

            if not pending_info:
                return 0

            claimed = 0
            for entry in pending_info:
                # Extract idle time
                if isinstance(entry, dict):
                    entry_idle = entry.get("idle", 0)
                    message_id = entry.get("message_id")
                else:
                    # Tuple format: (message_id, consumer, idle, delivery_count)
                    message_id, _, entry_idle, _ = entry

                if entry_idle >= idle_ms:
                    # Claim the message
                    try:
                        await self._redis.xclaim(
                            self.stream_key,
                            self.group_name,
                            self._consumer_name,
                            min_idle_time=idle_ms,
                            message_ids=[message_id],
                        )
                        claimed += 1
                        logger.info(f"Claimed stale message {message_id}")
                    except Exception as e:
                        logger.warning(f"Failed to claim message {message_id}: {e}")

            return claimed

        except Exception as e:
            logger.error(f"Error claiming stale jobs: {e}")
            return 0

    async def close(self) -> None:
        """Close the queue connection."""
        # The redis client is shared, don't close it here
        self._initialized = False
        logger.debug("Queue closed")


async def create_redis_queue(
    redis_url: Optional[str] = None,
    consumer_name: Optional[str] = None,
) -> RedisStreamsQueue:
    """
    Create a Redis Streams queue instance.

    Args:
        redis_url: Redis connection URL (defaults to config)
        consumer_name: Unique consumer name (defaults to hostname-based)

    Returns:
        A RedisStreamsQueue instance
    """
    import socket

    try:
        import redis.asyncio as redis
    except ImportError:
        raise ImportError(
            "redis package required for queue. Install with: pip install redis"
        )

    config = get_queue_config()
    url = redis_url or config.redis_url
    name = consumer_name or f"worker-{socket.gethostname()}-{id(asyncio.get_event_loop())}"

    client = redis.from_url(url, decode_responses=False)

    return RedisStreamsQueue(
        redis_client=client,
        consumer_name=name,
        config=config,
    )
