"""
Queue management endpoint handlers.

Exposes the job queue system for monitoring and management.

Endpoints:
- POST /api/queue/jobs - Submit new job
- GET /api/queue/jobs - List jobs with filters
- GET /api/queue/jobs/:id - Get job status
- POST /api/queue/jobs/:id/retry - Retry failed job
- DELETE /api/queue/jobs/:id - Cancel job
- GET /api/queue/stats - Queue statistics
- GET /api/queue/workers - Worker status
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from aragora.server.validation import validate_path_segment, SAFE_ID_PATTERN

from .base import (
    BaseHandler,
    HandlerResult,
    PaginatedHandlerMixin,
    error_response,
    get_string_param,
    json_response,
    safe_error_message,
)
from .utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)


# Module-level cache for queue instance
_queue_instance: Optional[Any] = None
_queue_lock = asyncio.Lock()


async def _get_queue() -> Optional[Any]:
    """Get or create the queue instance.

    Returns None if Redis is not available.
    """
    global _queue_instance

    if _queue_instance is not None:
        return _queue_instance

    async with _queue_lock:
        if _queue_instance is not None:
            return _queue_instance

        try:
            from aragora.queue import create_redis_queue

            _queue_instance = await create_redis_queue()
            return _queue_instance
        except ImportError:
            logger.warning("Redis package not available for queue")
            return None
        except (ConnectionError, OSError) as e:
            logger.warning(f"Failed to connect to Redis for queue: {e}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error creating queue: {e}")
            return None


class QueueHandler(BaseHandler, PaginatedHandlerMixin):
    """Handler for job queue management endpoints."""

    ROUTES = [
        "/api/queue/jobs",
        "/api/queue/jobs/*",
        "/api/queue/jobs/*/retry",
        "/api/queue/stats",
        "/api/queue/workers",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the request."""
        if path.startswith("/api/queue/"):
            return True
        return False

    @rate_limit(rpm=60)
    async def handle(  # type: ignore[override]
        self, path: str, method: str, handler: Any = None
    ) -> Optional[HandlerResult]:
        """Route request to appropriate handler method."""
        query_params: Dict[str, Any] = {}
        if handler:
            query_str = handler.path.split("?", 1)[1] if "?" in handler.path else ""
            from urllib.parse import parse_qs

            query_params = parse_qs(query_str)

        # GET /api/queue/stats
        if path == "/api/queue/stats" and method == "GET":
            return await self._get_stats()

        # GET /api/queue/workers
        if path == "/api/queue/workers" and method == "GET":
            return await self._get_workers()

        # POST /api/queue/jobs (submit new job)
        if path == "/api/queue/jobs" and method == "POST":
            return await self._submit_job(handler)

        # GET /api/queue/jobs (list jobs)
        if path == "/api/queue/jobs" and method == "GET":
            return await self._list_jobs(query_params)

        # Handle job-specific endpoints
        if path.startswith("/api/queue/jobs/"):
            parts = path.split("/")

            # POST /api/queue/jobs/:id/retry
            if len(parts) == 6 and parts[5] == "retry" and method == "POST":
                job_id = parts[4]
                is_valid, err = validate_path_segment(job_id, "job_id", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
                return await self._retry_job(job_id)

            # GET /api/queue/jobs/:id
            if len(parts) == 5 and method == "GET":
                job_id = parts[4]
                is_valid, err = validate_path_segment(job_id, "job_id", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
                return await self._get_job(job_id)

            # DELETE /api/queue/jobs/:id
            if len(parts) == 5 and method == "DELETE":
                job_id = parts[4]
                is_valid, err = validate_path_segment(job_id, "job_id", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
                return await self._cancel_job(job_id)

        return None

    async def _get_stats(self) -> HandlerResult:
        """Get queue statistics."""
        queue = await _get_queue()
        if queue is None:
            return json_response(
                {
                    "error": "Queue not available",
                    "message": "Redis queue is not configured or unavailable",
                    "stats": {
                        "pending": 0,
                        "processing": 0,
                        "completed": 0,
                        "failed": 0,
                        "cancelled": 0,
                        "retrying": 0,
                        "stream_length": 0,
                        "pending_in_group": 0,
                    },
                },
                status=503,
            )

        try:
            stats = await queue.get_queue_stats()
            return json_response(
                {
                    "stats": stats,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except (ConnectionError, OSError, TimeoutError) as e:
            logger.error(f"Failed to get queue stats due to connection error: {e}")
            return error_response(safe_error_message(e, "get queue stats"), 503)
        except Exception as e:
            logger.exception(f"Unexpected error getting queue stats: {e}")
            return error_response(safe_error_message(e, "get queue stats"), 500)

    async def _get_workers(self) -> HandlerResult:
        """Get worker status.

        Note: This provides limited info since workers are separate processes.
        Full worker monitoring requires a dedicated worker registry.
        """
        queue = await _get_queue()
        if queue is None:
            return json_response(
                {
                    "workers": [],
                    "total": 0,
                    "message": "Queue not available",
                },
                status=503,
            )

        try:
            # Workers are tracked via consumer group in Redis
            # This gives us consumer info from xinfo groups
            redis_client = queue._redis
            groups_info = await redis_client.xinfo_groups(queue.stream_key)

            workers: List[Dict[str, Any]] = []
            for group in groups_info:
                if isinstance(group, dict):
                    # Get consumers in this group
                    try:
                        consumers = await redis_client.xinfo_consumers(
                            queue.stream_key, group.get("name", "")
                        )
                        for consumer in consumers:
                            if isinstance(consumer, dict):
                                workers.append(
                                    {
                                        "worker_id": consumer.get("name", "unknown"),
                                        "group": group.get("name", "unknown"),
                                        "pending": consumer.get("pending", 0),
                                        "idle_ms": consumer.get("idle", 0),
                                    }
                                )
                    except (ConnectionError, OSError, TimeoutError) as ce:
                        logger.debug(f"Could not get consumers due to connection error: {ce}")
                    except Exception as ce:
                        logger.warning(f"Unexpected error getting consumers: {ce}")

            return json_response(
                {
                    "workers": workers,
                    "total": len(workers),
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except (ConnectionError, OSError, TimeoutError) as e:
            logger.error(f"Failed to get worker status due to connection error: {e}")
            return json_response(
                {
                    "workers": [],
                    "total": 0,
                    "error": str(e),
                }
            )
        except Exception as e:
            logger.exception(f"Unexpected error getting worker status: {e}")
            return json_response(
                {
                    "workers": [],
                    "total": 0,
                    "error": "Internal error",
                }
            )

    async def _submit_job(self, handler: Any) -> HandlerResult:
        """Submit a new job to the queue."""
        queue = await _get_queue()
        if queue is None:
            return error_response("Queue not available", 503)

        # Parse request body
        data = self.read_json_body(handler)
        if data is None:
            return error_response("Invalid or too large request body", 400)

        # Validate required fields
        question = data.get("question")
        if not question:
            return error_response("Missing required field: question", 400)

        try:
            from aragora.queue import create_debate_job

            # Create job from request
            job = create_debate_job(
                question=question,
                agents=data.get("agents"),
                rounds=data.get("rounds", 3),
                consensus=data.get("consensus", "majority"),
                protocol=data.get("protocol", "standard"),
                priority=data.get("priority", 0),
                max_attempts=data.get("max_attempts", 3),
                timeout_seconds=data.get("timeout_seconds"),
                webhook_url=data.get("webhook_url"),
                user_id=data.get("user_id"),
                organization_id=data.get("organization_id"),
                metadata=data.get("metadata"),
            )

            # Enqueue
            job_id = await queue.enqueue(job, priority=data.get("priority", 0))

            return json_response(
                {
                    "job_id": job_id,
                    "status": "pending",
                    "message": "Job submitted successfully",
                },
                status=202,
            )

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Invalid job submission data: {e}")
            return error_response(safe_error_message(e, "submit job"), 400)
        except (ConnectionError, OSError, TimeoutError) as e:
            logger.error(f"Failed to submit job due to connection error: {e}")
            return error_response(safe_error_message(e, "submit job"), 503)
        except Exception as e:
            logger.exception(f"Unexpected error submitting job: {e}")
            return error_response(safe_error_message(e, "submit job"), 500)

    async def _list_jobs(self, query_params: Dict[str, Any]) -> HandlerResult:
        """List jobs with optional filtering."""
        queue = await _get_queue()
        if queue is None:
            return json_response(
                {
                    "jobs": [],
                    "total": 0,
                    "message": "Queue not available",
                },
                status=503,
            )

        try:
            from aragora.queue import JobStatus

            # Parse filters
            limit, offset = self.get_pagination(query_params)
            status_filter = get_string_param(query_params, "status", None)

            # Convert status string to enum
            status_enum = None
            if status_filter:
                try:
                    status_enum = JobStatus(status_filter)
                except ValueError:
                    return error_response(
                        f"Invalid status: {status_filter}. "
                        f"Valid values: {[s.value for s in JobStatus]}",
                        400,
                    )

            # Get jobs from status tracker
            jobs = await queue._status_tracker.list_jobs(
                status=status_enum,
                limit=limit + offset,  # Get extra for pagination
            )

            # Apply offset
            jobs = jobs[offset : offset + limit]

            # Format response
            jobs_data = [
                {
                    "job_id": job.id,
                    "status": job.status.value,
                    "created_at": datetime.fromtimestamp(job.created_at).isoformat(),
                    "started_at": (
                        datetime.fromtimestamp(job.started_at).isoformat()
                        if job.started_at
                        else None
                    ),
                    "completed_at": (
                        datetime.fromtimestamp(job.completed_at).isoformat()
                        if job.completed_at
                        else None
                    ),
                    "attempts": job.attempts,
                    "max_attempts": job.max_attempts,
                    "priority": job.priority,
                    "error": job.error,
                    "worker_id": job.worker_id,
                    "metadata": {
                        k: v
                        for k, v in job.metadata.items()
                        if not k.startswith("_")  # Hide internal fields
                    },
                }
                for job in jobs
            ]

            # Get total count
            counts = await queue._status_tracker.get_counts_by_status()
            total = sum(counts.values())

            return json_response(
                {
                    "jobs": jobs_data,
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "counts_by_status": counts,
                }
            )

        except (ConnectionError, OSError, TimeoutError) as e:
            logger.error(f"Failed to list jobs due to connection error: {e}")
            return error_response(safe_error_message(e, "list jobs"), 503)
        except Exception as e:
            logger.exception(f"Unexpected error listing jobs: {e}")
            return error_response(safe_error_message(e, "list jobs"), 500)

    async def _get_job(self, job_id: str) -> HandlerResult:
        """Get a specific job's status."""
        queue = await _get_queue()
        if queue is None:
            return error_response("Queue not available", 503)

        try:
            job = await queue.get_status(job_id)
            if job is None:
                return error_response(f"Job not found: {job_id}", 404)

            return json_response(
                {
                    "job_id": job.id,
                    "status": job.status.value,
                    "created_at": datetime.fromtimestamp(job.created_at).isoformat(),
                    "started_at": (
                        datetime.fromtimestamp(job.started_at).isoformat()
                        if job.started_at
                        else None
                    ),
                    "completed_at": (
                        datetime.fromtimestamp(job.completed_at).isoformat()
                        if job.completed_at
                        else None
                    ),
                    "attempts": job.attempts,
                    "max_attempts": job.max_attempts,
                    "priority": job.priority,
                    "error": job.error,
                    "worker_id": job.worker_id,
                    "payload": job.payload,
                    "metadata": {k: v for k, v in job.metadata.items() if not k.startswith("_")},
                    "result": job.metadata.get("result"),
                }
            )

        except (ConnectionError, OSError, TimeoutError) as e:
            logger.error(f"Failed to get job {job_id} due to connection error: {e}")
            return error_response(safe_error_message(e, "get job"), 503)
        except Exception as e:
            logger.exception(f"Unexpected error getting job {job_id}: {e}")
            return error_response(safe_error_message(e, "get job"), 500)

    async def _retry_job(self, job_id: str) -> HandlerResult:
        """Retry a failed job."""
        queue = await _get_queue()
        if queue is None:
            return error_response("Queue not available", 503)

        try:
            from aragora.queue import JobStatus

            # Get job status
            job = await queue.get_status(job_id)
            if job is None:
                return error_response(f"Job not found: {job_id}", 404)

            # Can only retry failed or cancelled jobs
            if job.status not in (JobStatus.FAILED, JobStatus.CANCELLED):
                return error_response(
                    f"Cannot retry job with status: {job.status.value}. "
                    "Only failed or cancelled jobs can be retried.",
                    400,
                )

            # Reset job for retry
            job.status = JobStatus.PENDING
            job.attempts = 0
            job.error = None
            job.started_at = None
            job.completed_at = None
            job.worker_id = None

            # Re-enqueue
            await queue.enqueue(job, priority=job.priority)

            return json_response(
                {
                    "job_id": job_id,
                    "status": "pending",
                    "message": "Job queued for retry",
                }
            )

        except (ConnectionError, OSError, TimeoutError) as e:
            logger.error(f"Failed to retry job {job_id} due to connection error: {e}")
            return error_response(safe_error_message(e, "retry job"), 503)
        except Exception as e:
            logger.exception(f"Unexpected error retrying job {job_id}: {e}")
            return error_response(safe_error_message(e, "retry job"), 500)

    async def _cancel_job(self, job_id: str) -> HandlerResult:
        """Cancel a pending job."""
        queue = await _get_queue()
        if queue is None:
            return error_response("Queue not available", 503)

        try:
            cancelled = await queue.cancel(job_id)
            if not cancelled:
                # Check if job exists
                job = await queue.get_status(job_id)
                if job is None:
                    return error_response(f"Job not found: {job_id}", 404)
                else:
                    return error_response(
                        f"Cannot cancel job with status: {job.status.value}. "
                        "Only pending or retrying jobs can be cancelled.",
                        400,
                    )

            return json_response(
                {
                    "job_id": job_id,
                    "status": "cancelled",
                    "message": "Job cancelled successfully",
                }
            )

        except (ConnectionError, OSError, TimeoutError) as e:
            logger.error(f"Failed to cancel job {job_id} due to connection error: {e}")
            return error_response(safe_error_message(e, "cancel job"), 503)
        except Exception as e:
            logger.exception(f"Unexpected error cancelling job {job_id}: {e}")
            return error_response(safe_error_message(e, "cancel job"), 500)
