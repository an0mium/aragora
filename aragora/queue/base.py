"""
Base interfaces and data structures for the queue system.

Provides abstract base classes and dataclasses that define the queue contract.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class JobStatus(Enum):
    """Status of a job in the queue."""

    PENDING = "pending"  # Job is waiting to be processed
    PROCESSING = "processing"  # Job is being processed by a worker
    COMPLETED = "completed"  # Job completed successfully
    FAILED = "failed"  # Job failed after all retries
    CANCELLED = "cancelled"  # Job was cancelled
    RETRYING = "retrying"  # Job is waiting to be retried


@dataclass
class Job:
    """
    Represents a job in the queue.

    Attributes:
        id: Unique job identifier
        payload: Job-specific data
        status: Current job status
        created_at: Unix timestamp when job was created
        started_at: Unix timestamp when processing started
        completed_at: Unix timestamp when job completed/failed
        attempts: Number of processing attempts
        max_attempts: Maximum allowed attempts
        error: Last error message if failed
        worker_id: ID of worker processing this job
        priority: Job priority (higher = more important)
        metadata: Additional metadata for tracking
    """

    payload: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    attempts: int = 0
    max_attempts: int = 3
    error: Optional[str] = None
    worker_id: Optional[str] = None
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization."""
        return {
            "id": self.id,
            "payload": self.payload,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "error": self.error,
            "worker_id": self.worker_id,
            "priority": self.priority,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """Create job from dictionary."""
        return cls(
            id=data["id"],
            payload=data["payload"],
            status=JobStatus(data["status"]),
            created_at=data["created_at"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 3),
            error=data.get("error"),
            worker_id=data.get("worker_id"),
            priority=data.get("priority", 0),
            metadata=data.get("metadata", {}),
        )

    def mark_processing(self, worker_id: str) -> None:
        """Mark job as being processed."""
        self.status = JobStatus.PROCESSING
        self.started_at = datetime.now().timestamp()
        self.worker_id = worker_id
        self.attempts += 1

    def mark_completed(self, result: Optional[Dict[str, Any]] = None) -> None:
        """Mark job as completed."""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.now().timestamp()
        if result:
            self.metadata["result"] = result

    def mark_failed(self, error: str) -> None:
        """Mark job as failed."""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.now().timestamp()
        self.error = error

    def mark_retrying(self, error: str) -> None:
        """Mark job as waiting for retry."""
        self.status = JobStatus.RETRYING
        self.error = error

    def should_retry(self) -> bool:
        """Check if job should be retried."""
        return self.attempts < self.max_attempts


class JobQueue(ABC):
    """
    Abstract base class for job queues.

    Implementations must provide methods for enqueueing, dequeueing,
    acknowledging, and managing job lifecycle.
    """

    @abstractmethod
    async def enqueue(self, job: Job, priority: int = 0) -> str:
        """
        Add a job to the queue.

        Args:
            job: The job to enqueue
            priority: Job priority (higher = more important)

        Returns:
            The job ID
        """
        pass

    @abstractmethod
    async def dequeue(self, worker_id: str, timeout_ms: int = 5000) -> Optional[Job]:
        """
        Get the next job from the queue.

        Args:
            worker_id: ID of the worker requesting work
            timeout_ms: How long to block waiting for a job

        Returns:
            A job if available, None otherwise
        """
        pass

    @abstractmethod
    async def ack(self, job_id: str) -> bool:
        """
        Acknowledge successful processing of a job.

        Args:
            job_id: The job ID to acknowledge

        Returns:
            True if acknowledged, False if job not found
        """
        pass

    @abstractmethod
    async def nack(self, job_id: str, requeue: bool = True) -> bool:
        """
        Negative acknowledge - job processing failed.

        Args:
            job_id: The job ID
            requeue: Whether to requeue for retry

        Returns:
            True if processed, False if job not found
        """
        pass

    @abstractmethod
    async def get_status(self, job_id: str) -> Optional[Job]:
        """
        Get the current status of a job.

        Args:
            job_id: The job ID

        Returns:
            The job if found, None otherwise
        """
        pass

    @abstractmethod
    async def cancel(self, job_id: str) -> bool:
        """
        Cancel a pending job.

        Args:
            job_id: The job ID to cancel

        Returns:
            True if cancelled, False if not found or already processing
        """
        pass

    @abstractmethod
    async def get_queue_stats(self) -> Dict[str, int]:
        """
        Get queue statistics.

        Returns:
            Dict with counts for pending, processing, completed, failed jobs
        """
        pass

    @abstractmethod
    async def claim_stale_jobs(self, idle_ms: int) -> int:
        """
        Claim jobs from dead workers.

        Args:
            idle_ms: Minimum idle time to consider a job stale

        Returns:
            Number of jobs claimed
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the queue connection."""
        pass
