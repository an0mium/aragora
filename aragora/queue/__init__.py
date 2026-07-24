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
        rounds=9,
    )
    job_id = await queue.enqueue(job)

    # Check status
    job = await queue.get_status(job_id)
    print(f"Job status: {job.status}")

    # Process jobs with a worker (executor factory lives in aragora.debate -
    # it imports agents/debate, so it is not part of this domain-free package)
    from aragora.debate.queue_executor import create_default_executor

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
    get_job_handler,
    get_registered_job_handlers,
    get_registered_workers,
    register_job_handler,
    register_worker,
    registered_job_handler_names,
    registered_worker_names,
    reset_registry,
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
    # Job-handler registry (domain-free; P4a queue inversion Q1)
    "register_worker",
    "register_job_handler",
    "get_registered_workers",
    "get_registered_job_handlers",
    "get_job_handler",
    "registered_worker_names",
    "registered_job_handler_names",
    "reset_registry",
]
