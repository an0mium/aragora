"""
Message queue system for Aragora.

Provides Redis Streams-based job queue for async debate processing
with horizontal scaling support.

Usage:
    from aragora.queue import create_redis_queue, create_debate_job, DebateWorker

    # Enqueue a debate
    queue = await create_redis_queue()
    job = create_debate_job(
        question="Should we use microservices?",
        agents=["claude", "gpt"],
        rounds=3,
    )
    job_id = await queue.enqueue(job)

    # Check status
    job = await queue.get_status(job_id)
    print(f"Job status: {job.status}")

    # Process jobs with a worker
    executor = await create_default_executor()
    worker = DebateWorker(queue, "worker-1", executor)
    await worker.start()

Environment Variables:
    REDIS_URL: Redis connection URL (default: redis://localhost:6379)
    ARAGORA_QUEUE_PREFIX: Key prefix (default: aragora:queue:)

See docs/QUEUE.md for full documentation.
"""

from aragora.queue.base import (
    Job,
    JobQueue,
    JobStatus,
)
from aragora.queue.config import (
    QueueConfig,
    get_queue_config,
    reset_queue_config,
    set_queue_config,
)
from aragora.queue.job import (
    DebateJobPayload,
    DebateResult,
    create_debate_job,
    get_debate_payload,
)
from aragora.queue.retry import (
    RetryPolicy,
    is_retryable_error,
)
from aragora.queue.status import JobStatusTracker
from aragora.queue.streams import (
    RedisStreamsQueue,
    create_redis_queue,
)
from aragora.queue.worker import (
    DebateExecutor,
    DebateWorker,
    create_default_executor,
)

__all__ = [
    # Base types
    "Job",
    "JobQueue",
    "JobStatus",
    # Configuration
    "QueueConfig",
    "get_queue_config",
    "set_queue_config",
    "reset_queue_config",
    # Job types
    "DebateJobPayload",
    "DebateResult",
    "create_debate_job",
    "get_debate_payload",
    # Retry
    "RetryPolicy",
    "is_retryable_error",
    # Status tracking
    "JobStatusTracker",
    # Redis Streams implementation
    "RedisStreamsQueue",
    "create_redis_queue",
    # Worker
    "DebateWorker",
    "DebateExecutor",
    "create_default_executor",
]
