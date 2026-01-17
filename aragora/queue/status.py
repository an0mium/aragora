"""
Job status tracking using Redis hashes.

Provides fast status updates and queries for job state.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from aragora.queue.base import Job, JobStatus
from aragora.queue.config import get_queue_config

logger = logging.getLogger(__name__)


class JobStatusTracker:
    """
    Tracks job status in Redis for fast queries and updates.

    Uses Redis hashes for atomic status updates:
    - HSET for atomic status updates
    - HGETALL for full job state
    - TTL for automatic cleanup of old jobs
    """

    def __init__(self, redis_client: Any) -> None:
        """
        Initialize the status tracker.

        Args:
            redis_client: An async Redis client instance
        """
        self._redis = redis_client
        self._config = get_queue_config()

    def _job_key(self, job_id: str) -> str:
        """Get the Redis key for a job's status."""
        return f"{self._config.status_key_prefix}{job_id}"

    async def create(self, job: Job) -> None:
        """
        Create a new job status entry.

        Args:
            job: The job to track
        """
        key = self._job_key(job.id)
        data = {
            "id": job.id,
            "status": job.status.value,
            "created_at": str(job.created_at),
            "attempts": str(job.attempts),
            "max_attempts": str(job.max_attempts),
            "priority": str(job.priority),
            "payload": json.dumps(job.payload),
            "metadata": json.dumps(job.metadata),
        }

        await self._redis.hset(key, mapping=data)
        await self._redis.expire(key, self._config.job_ttl_seconds)

    async def update_status(
        self,
        job_id: str,
        status: JobStatus,
        **kwargs: Any,
    ) -> bool:
        """
        Update job status atomically.

        Args:
            job_id: The job ID
            status: New status
            **kwargs: Additional fields to update (error, worker_id, result, etc.)

        Returns:
            True if updated, False if job not found
        """
        key = self._job_key(job_id)

        # Check if job exists
        if not await self._redis.exists(key):
            return False

        # Prepare update data
        update = {"status": status.value}

        # Add timestamp based on status
        now = datetime.now().timestamp()
        if status == JobStatus.PROCESSING:
            update["started_at"] = str(now)
        elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            update["completed_at"] = str(now)

        # Add optional fields
        for field in ("error", "worker_id", "attempts"):
            if field in kwargs:
                update[field] = str(kwargs[field])

        if "result" in kwargs:
            update["result"] = json.dumps(kwargs["result"])

        await self._redis.hset(key, mapping=update)

        # Refresh TTL
        await self._redis.expire(key, self._config.job_ttl_seconds)

        return True

    async def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get job state from Redis.

        Args:
            job_id: The job ID

        Returns:
            The Job if found, None otherwise
        """
        key = self._job_key(job_id)
        data = await self._redis.hgetall(key)

        if not data:
            return None

        # Decode bytes if needed
        if isinstance(next(iter(data.keys()), ""), bytes):
            data = {k.decode(): v.decode() for k, v in data.items()}

        try:
            return Job(
                id=data["id"],
                payload=json.loads(data.get("payload", "{}")),
                status=JobStatus(data["status"]),
                created_at=float(data.get("created_at", 0)),
                started_at=float(data["started_at"]) if data.get("started_at") else None,
                completed_at=float(data["completed_at"]) if data.get("completed_at") else None,
                attempts=int(data.get("attempts", 0)),
                max_attempts=int(data.get("max_attempts", 3)),
                error=data.get("error"),
                worker_id=data.get("worker_id"),
                priority=int(data.get("priority", 0)),
                metadata=json.loads(data.get("metadata", "{}")),
            )
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to parse job {job_id}: {e}")
            return None

    async def get_status(self, job_id: str) -> Optional[JobStatus]:
        """
        Get just the status of a job.

        Args:
            job_id: The job ID

        Returns:
            The JobStatus if found, None otherwise
        """
        key = self._job_key(job_id)
        status = await self._redis.hget(key, "status")

        if status is None:
            return None

        if isinstance(status, bytes):
            status = status.decode()

        return JobStatus(status)

    async def delete(self, job_id: str) -> bool:
        """
        Delete a job status entry.

        Args:
            job_id: The job ID

        Returns:
            True if deleted, False if not found
        """
        key = self._job_key(job_id)
        result: int = await self._redis.delete(key)
        return result > 0

    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 50,
    ) -> List[Job]:
        """
        List jobs, optionally filtered by status.

        Note: This is a potentially expensive operation that scans keys.
        Use with caution in production.

        Args:
            status: Optional status filter
            limit: Maximum number of jobs to return

        Returns:
            List of matching jobs
        """
        pattern = f"{self._config.status_key_prefix}*"
        jobs: list[Job] = []

        async for key in self._redis.scan_iter(match=pattern, count=100):
            if len(jobs) >= limit:
                break

            if isinstance(key, bytes):
                key = key.decode()

            job_id = key.replace(self._config.status_key_prefix, "")
            job = await self.get_job(job_id)

            if job is None:
                continue

            if status is None or job.status == status:
                jobs.append(job)

        return jobs

    async def get_counts_by_status(self) -> Dict[str, int]:
        """
        Get counts of jobs by status.

        Returns:
            Dict mapping status names to counts
        """
        counts: Dict[str, int] = {status.value: 0 for status in JobStatus}
        pattern = f"{self._config.status_key_prefix}*"

        async for key in self._redis.scan_iter(match=pattern, count=100):
            if isinstance(key, bytes):
                key = key.decode()

            status = await self._redis.hget(key, "status")
            if status:
                if isinstance(status, bytes):
                    status = status.decode()
                counts[status] = counts.get(status, 0) + 1

        return counts
