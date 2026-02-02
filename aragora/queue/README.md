# Queue Module

Redis Streams-based job queue for asynchronous task processing with horizontal scaling, fault tolerance, and comprehensive observability.

## Overview

The `aragora.queue` module provides a production-ready job queue system built on Redis Streams. It enables Aragora to process debates, transcriptions, webhook deliveries, and other background tasks with:

- **Horizontal Scaling**: Multiple workers process jobs in parallel using consumer groups
- **Fault Tolerance**: Automatic retry with exponential backoff and jitter
- **Stale Job Recovery**: Dead worker detection via XCLAIM reclaims abandoned jobs
- **Status Tracking**: Real-time job status via Redis hashes with TTL-based cleanup
- **Distributed Tracing**: Trace context propagation from HTTP requests to background jobs

## Architecture

```
                                   +-----------------+
                                   |   HTTP Handler  |
                                   +--------+--------+
                                            |
                                            v
+------------------+              +------------------+              +------------------+
|     Client       |              |   Redis Streams  |              |     Worker       |
|   (enqueue)      |------------->|   (job queue)    |------------->|   (process)      |
+------------------+              +--------+---------+              +------------------+
                                           |                                 |
                                           v                                 v
                                  +------------------+              +------------------+
                                  |   Redis Hashes   |              |   Job Executor   |
                                  |  (job status)    |              |  (Arena, etc.)   |
                                  +------------------+              +------------------+
```

### Job Lifecycle

```
PENDING --> PROCESSING --> COMPLETED
                 |
                 +--> RETRYING --> PROCESSING --> ...
                 |         |
                 |         +--> FAILED (max attempts)
                 |
                 +--> CANCELLED
```

## Module Structure

| File | Purpose |
|------|---------|
| `__init__.py` | Public API exports |
| `base.py` | Abstract `JobQueue` interface, `Job` dataclass, `JobStatus` enum |
| `streams.py` | `RedisStreamsQueue` implementation using consumer groups |
| `worker.py` | `DebateWorker` for horizontal scaling |
| `job.py` | `DebateJobPayload` and `DebateResult` types |
| `retry.py` | `RetryPolicy` with exponential backoff + jitter |
| `status.py` | `JobStatusTracker` for Redis hash-based status |
| `config.py` | `QueueConfig` with environment variable support |
| `tracing.py` | Distributed trace context propagation |
| `batch_worker.py` | Batch explainability job processing |
| `webhook_worker.py` | Reliable webhook delivery with circuit breakers |
| `workers/` | Specialized workers (gauntlet, transcription, routing, consensus healing) |

## Quick Start

### Enqueueing a Debate

```python
from aragora.queue import create_redis_queue, create_debate_job

# Create queue connection
queue = await create_redis_queue()

# Create a debate job
job = create_debate_job(
    question="Should we use microservices?",
    agents=["anthropic-api", "openai-api"],
    rounds=3,
    consensus="majority",
)

# Enqueue the job
job_id = await queue.enqueue(job)
print(f"Job enqueued: {job_id}")

# Check status
status = await queue.get_status(job_id)
print(f"Status: {status.status.value}")
```

### Running a Worker

```python
from aragora.queue import (
    create_redis_queue,
    DebateWorker,
    create_default_executor,
)

# Create queue and executor
queue = await create_redis_queue(consumer_name="worker-1")
executor = await create_default_executor()

# Create and start worker
worker = DebateWorker(
    queue=queue,
    worker_id="worker-1",
    executor=executor,
    max_concurrent=3,
)

await worker.start()
```

### Command Line Worker

```bash
# Start a worker
python -m scripts.queue_worker --worker-id worker-1 --concurrency 3

# With custom Redis URL
REDIS_URL=redis://redis.example.com:6379 python -m scripts.queue_worker

# Run multiple workers for scaling
python -m scripts.queue_worker --worker-id worker-1 &
python -m scripts.queue_worker --worker-id worker-2 &
```

## Worker Patterns

### DebateWorker

The core worker for processing debate jobs with full Arena integration.

```python
from aragora.queue import DebateWorker, create_redis_queue, create_default_executor

queue = await create_redis_queue(consumer_name="debate-worker-1")
executor = await create_default_executor()

worker = DebateWorker(
    queue=queue,
    worker_id="debate-worker-1",
    executor=executor,
    max_concurrent=3,  # Process up to 3 debates concurrently
)

# Worker runs until stop() is called or signal received
await worker.start()
```

Features:
- Concurrent job processing with semaphore-based rate limiting
- Graceful shutdown with configurable timeout
- Automatic stale job claiming from dead workers
- Health reporting and statistics

### BatchExplainabilityWorker

Processes batches of debates for explanation generation.

```python
from aragora.queue.batch_worker import BatchExplainabilityWorker, create_batch_job

async def explain_debate(debate_id: str, options: dict) -> dict:
    # Generate explanation for a single debate
    ...

worker = BatchExplainabilityWorker(
    queue=queue,
    worker_id="batch-worker-1",
    explain_generator=explain_debate,
    max_concurrent_debates=5,  # Per batch
    max_concurrent_batches=2,
)

# Enqueue a batch job
job = await create_batch_job(
    queue=queue,
    debate_ids=["d1", "d2", "d3"],
    options={"include_counterfactuals": True},
)
```

Features:
- Progress tracking per batch
- Partial results on failures
- Concurrent debate processing within batches

### WebhookDeliveryWorker

Reliable webhook delivery with circuit breakers.

```python
from aragora.queue.webhook_worker import WebhookDeliveryWorker, enqueue_webhook_delivery

worker = WebhookDeliveryWorker(
    queue=queue,
    worker_id="webhook-worker-1",
    max_concurrent=10,
    request_timeout=10.0,
)

# Enqueue a webhook delivery
job = await enqueue_webhook_delivery(
    queue=queue,
    webhook_id="wh-123",
    url="https://example.com/webhook",
    secret="signing-secret",
    event_type="debate.completed",
    event_data={"debate_id": "d-456", "result": {...}},
)
```

Features:
- Per-endpoint circuit breakers
- HMAC-SHA256 signature generation
- Exponential backoff retry (up to 5 attempts)
- Endpoint health tracking

### Specialized Workers

Located in `aragora/queue/workers/`:

| Worker | Purpose |
|--------|---------|
| `GauntletWorker` | Stress-testing and validation jobs |
| `TranscriptionWorker` | Audio/video transcription (Whisper integration) |
| `RoutingWorker` | Debate result delivery to originating channels |
| `ConsensusHealingWorker` | Monitors and heals failed consensus attempts |

## Job Definitions

### DebateJobPayload

```python
@dataclass
class DebateJobPayload:
    question: str
    agents: list[str]           # ["anthropic-api", "openai-api"]
    rounds: int = 9
    consensus: str = "majority"
    protocol: str = "standard"
    timeout_seconds: int | None = None
    webhook_url: str | None = None
    user_id: str | None = None
    organization_id: str | None = None
```

### DebateResult

```python
@dataclass
class DebateResult:
    debate_id: str
    consensus_reached: bool
    final_answer: str | None
    confidence: float
    rounds_used: int
    participants: list[str]
    duration_seconds: float
    token_usage: dict[str, int]
    error: str | None
```

## Failure Handling and Retries

### RetryPolicy

```python
from aragora.queue import RetryPolicy

policy = RetryPolicy(
    max_attempts=3,           # Total attempts before permanent failure
    base_delay_seconds=1.0,   # Initial delay
    max_delay_seconds=300.0,  # Maximum delay cap
    exponential_base=2.0,     # Delay multiplier per attempt
    jitter=True,              # Add randomness to prevent thundering herd
)

# Delay calculation: base * (exponential_base ^ attempt) +/- 20% jitter
# Attempt 0: 1.0s (+/- 20%)
# Attempt 1: 2.0s (+/- 20%)
# Attempt 2: 4.0s (+/- 20%)
```

### Retryable vs Non-Retryable Errors

**Retryable** (will be retried):
- `ConnectionError`, `TimeoutError`
- `RuntimeError`, generic `Exception`
- Temporary API failures (rate limits, server errors)

**Non-Retryable** (immediate failure):
- `ValueError`, `TypeError`, `KeyError`, `AttributeError`
- `ImportError`, `SyntaxError`
- Errors with "invalid", "not found", "unauthorized", "forbidden", "bad request", "validation" in message

```python
from aragora.queue import is_retryable_error

if is_retryable_error(error):
    # Schedule for retry
else:
    # Mark as permanently failed
```

## Consumer Groups

Redis Streams consumer groups enable horizontal scaling with work distribution.

### How It Works

1. All workers join the same consumer group (`debate-workers` by default)
2. Each job is delivered to exactly one worker in the group
3. Workers acknowledge jobs after successful processing
4. Unacknowledged jobs can be claimed by other workers

### Key Operations

| Operation | Redis Command | Purpose |
|-----------|---------------|---------|
| Enqueue | `XADD` | Add job to stream |
| Dequeue | `XREADGROUP` | Read job for consumer |
| Acknowledge | `XACK` | Mark job as processed |
| Claim stale | `XCLAIM` | Take over from dead worker |
| Get pending | `XPENDING` | List unacknowledged jobs |
| Stream length | `XLEN` | Count total jobs |

### Stale Job Recovery

Workers automatically claim jobs from dead workers:

```python
# Configured via QueueConfig
claim_idle_ms = 60000  # Claim jobs idle for 60+ seconds

# Workers run a background loop that periodically claims stale jobs
claimed = await queue.claim_stale_jobs(idle_ms=claim_idle_ms)
```

## Monitoring and Observability

### Queue Statistics

```python
stats = await queue.get_queue_stats()
print(f"Stream length: {stats['stream_length']}")
print(f"Pending in group: {stats['pending_in_group']}")
print(f"Status counts: {stats}")
# {
#   'stream_length': 150,
#   'pending_in_group': 5,
#   'pending': 10,
#   'processing': 3,
#   'completed': 130,
#   'failed': 7,
#   'cancelled': 0,
#   'retrying': 0
# }
```

### Worker Statistics

```python
stats = worker.get_stats()
print(f"Worker ID: {stats['worker_id']}")
print(f"Active jobs: {stats['active_jobs']}")
print(f"Jobs processed: {stats['jobs_processed']}")
print(f"Jobs failed: {stats['jobs_failed']}")
print(f"Uptime: {stats['uptime_seconds']}s")
```

### Job Status Tracking

```python
from aragora.queue import JobStatusTracker

tracker = JobStatusTracker(redis_client)

# Get single job
job = await tracker.get_job("job-123")
if job:
    print(f"Status: {job.status.value}")
    print(f"Attempts: {job.attempts}/{job.max_attempts}")
    print(f"Error: {job.error}")

# Get status counts
counts = await tracker.get_counts_by_status()
# {'pending': 10, 'processing': 2, 'completed': 100, ...}

# List jobs by status
failed_jobs = await tracker.list_jobs(status=JobStatus.FAILED, limit=50)
```

### Distributed Tracing

Trace context propagates from HTTP requests to background jobs:

```python
from aragora.queue.tracing import inject_trace_context, traced_job

# Enqueue side: inject current trace context
payload = {"debate_id": "d-123"}
payload = inject_trace_context(payload)
await queue.enqueue(Job(payload=payload))

# Worker side: trace context is restored
@traced_job("debate.execute")
async def execute_job(job: Job) -> dict:
    # Trace context from original request is active
    # New spans will be children of the enqueuing span
    ...
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `ARAGORA_QUEUE_PREFIX` | `aragora:queue:` | Key prefix for queue data |
| `ARAGORA_QUEUE_RETRY_MAX` | `3` | Maximum retry attempts |
| `ARAGORA_QUEUE_RETRY_BASE_DELAY` | `1.0` | Base delay for retries (seconds) |
| `ARAGORA_QUEUE_RETRY_MAX_DELAY` | `300.0` | Max delay for retries (seconds) |
| `ARAGORA_QUEUE_CLAIM_IDLE_MS` | `60000` | Claim stale jobs after (ms) |
| `ARAGORA_QUEUE_MAX_TTL_DAYS` | `7` | Job data retention (days) |
| `ARAGORA_QUEUE_WORKER_BLOCK_MS` | `5000` | Block time for XREADGROUP (ms) |
| `ARAGORA_QUEUE_CONSUMER_GROUP` | `debate-workers` | Consumer group name |

### QueueConfig

```python
from aragora.queue import QueueConfig, set_queue_config

config = QueueConfig(
    redis_url="redis://localhost:6379",
    key_prefix="aragora:queue:",
    retry_max_attempts=3,
    retry_base_delay=1.0,
    retry_max_delay=300.0,
    claim_idle_ms=60000,
    max_job_ttl_days=7,
    worker_block_ms=5000,
    consumer_group="debate-workers",
)
set_queue_config(config)
```

### Configuration Validation

The `QueueConfig` validates values on initialization:

| Parameter | Valid Range |
|-----------|-------------|
| `max_job_ttl_days` | 1-30 |
| `claim_idle_ms` | 10000-600000 |
| `retry_max_attempts` | 1-10 |
| `retry_base_delay` | 0.1-60.0 |
| `retry_max_delay` | 1.0-3600.0 |
| `worker_block_ms` | 1000-30000 |

## API Reference

### Core Classes

#### Job

```python
@dataclass
class Job:
    payload: dict[str, Any]
    id: str                          # Auto-generated UUID
    status: JobStatus = PENDING
    created_at: float                # Unix timestamp
    started_at: float | None = None
    completed_at: float | None = None
    attempts: int = 0
    max_attempts: int = 3
    error: str | None = None
    worker_id: str | None = None
    priority: int = 0
    metadata: dict[str, Any]

    def mark_processing(worker_id: str) -> None
    def mark_completed(result: dict | None) -> None
    def mark_failed(error: str) -> None
    def mark_retrying(error: str) -> None
    def should_retry() -> bool
    def to_dict() -> dict
    @classmethod from_dict(data: dict) -> Job
```

#### JobQueue (Abstract Base)

```python
class JobQueue(ABC):
    async def enqueue(job, priority=0, queue_name=None, delay_seconds=0) -> str
    async def dequeue(worker_id, timeout_ms=5000, queue_name=None) -> Job | None
    async def ack(job_id: str) -> bool
    async def nack(job_id: str, requeue=True) -> bool
    async def get_status(job_id: str) -> Job | None
    async def cancel(job_id: str) -> bool
    async def get_queue_stats() -> dict[str, int]
    async def claim_stale_jobs(idle_ms: int) -> int
    async def close() -> None
    async def complete(job_id, result=None, status=None) -> bool
    async def fail(job_id, error, requeue=False) -> bool
```

#### RedisStreamsQueue

```python
class RedisStreamsQueue(JobQueue):
    def __init__(redis_client, consumer_name, config=None)

    @property stream_key: str
    @property group_name: str

# Factory function
async def create_redis_queue(redis_url=None, consumer_name=None) -> RedisStreamsQueue
```

#### DebateWorker

```python
class DebateWorker:
    def __init__(queue, worker_id, executor, max_concurrent=3, retry_policy=None)

    @property worker_id: str
    @property is_running: bool
    @property active_jobs: int

    async def start() -> None
    async def stop(timeout=30.0) -> None
    def get_stats() -> dict[str, Any]

# Factory function
async def create_default_executor() -> DebateExecutor
```

### Helper Functions

```python
# Job creation
def create_debate_job(
    question: str,
    agents: list[str] | None = None,
    rounds: int = 9,
    consensus: str = "majority",
    protocol: str = "standard",
    priority: int = 0,
    max_attempts: int = 3,
    timeout_seconds: int | None = None,
    webhook_url: str | None = None,
    user_id: str | None = None,
    organization_id: str | None = None,
    metadata: dict | None = None,
) -> Job

# Payload extraction
def get_debate_payload(job: Job) -> DebateJobPayload

# Error classification
def is_retryable_error(error: Exception) -> bool
```

## Docker Deployment

### docker-compose.yml

```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

  aragora:
    build: .
    ports:
      - "8080:8080"
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

  debate-worker:
    build: .
    command: python -m scripts.queue_worker --concurrency 3
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - aragora
    deploy:
      replicas: 2  # Scale horizontally
    profiles:
      - with-workers

volumes:
  redis-data:
```

### Running with Workers

```bash
# Start with workers
docker compose --profile with-workers up

# Scale workers
docker compose --profile with-workers up --scale debate-worker=5
```

## Testing

```bash
# Run queue tests
pytest tests/test_queue.py -v

# With coverage
pytest tests/test_queue.py --cov=aragora.queue

# Specific worker tests
pytest tests/test_queue_worker.py -v
```

## Troubleshooting

### Jobs Stuck in PENDING

1. Check Redis connection: `redis-cli ping`
2. Verify workers are running: `docker compose ps`
3. Check worker logs: `docker compose logs debate-worker`
4. Verify consumer group exists:
   ```bash
   redis-cli XINFO GROUPS aragora:queue:debates:stream
   ```

### Jobs Not Being Processed

1. Verify consumer group exists:
   ```bash
   redis-cli XINFO GROUPS aragora:queue:debates:stream
   ```
2. Check for pending jobs:
   ```bash
   redis-cli XPENDING aragora:queue:debates:stream debate-workers
   ```
3. Manually claim stale jobs if needed

### High Failure Rate

1. Check retry policy configuration
2. Review error logs for patterns
3. Verify API keys and credentials
4. Consider increasing `max_attempts` for transient issues
5. Check circuit breaker states for webhook endpoints

### Memory Issues

1. Monitor Redis memory: `redis-cli INFO memory`
2. Verify job TTL is appropriate
3. Consider trimming old stream entries:
   ```bash
   redis-cli XTRIM aragora:queue:debates:stream MAXLEN ~ 10000
   ```

## See Also

- [docs/QUEUE.md](/docs/QUEUE.md) - High-level queue documentation
- [aragora/resilience.py](/aragora/resilience.py) - CircuitBreaker implementation
- [aragora/storage/job_queue_store.py](/aragora/storage/job_queue_store.py) - Persistent job storage
