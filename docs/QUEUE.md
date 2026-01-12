# Message Queue System

Redis Streams-based job queue for async debate processing with horizontal scaling support.

## Overview

The queue system enables Aragora to process debates asynchronously with:
- **Horizontal scaling**: Multiple workers process jobs in parallel
- **Fault tolerance**: Automatic retry with exponential backoff
- **Stale job recovery**: Dead worker detection via XCLAIM
- **Status tracking**: Real-time job status via Redis hashes

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Client    │────▶│  Redis Streams   │────▶│   Worker    │
│  (enqueue)  │     │  (job queue)     │     │  (process)  │
└─────────────┘     └──────────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────────┐
                    │   Redis Hashes   │
                    │  (job status)    │
                    └──────────────────┘
```

### Job Lifecycle

```
PENDING → PROCESSING → COMPLETED
                   ↓
              RETRYING → PROCESSING → ...
                   ↓
                FAILED (max attempts)
```

### Components

| Module | Purpose |
|--------|---------|
| `base.py` | Abstract `JobQueue` interface, `Job` dataclass, `JobStatus` enum |
| `streams.py` | `RedisStreamsQueue` implementation using consumer groups |
| `worker.py` | `DebateWorker` for horizontal scaling |
| `job.py` | `DebateJobPayload` and `DebateResult` types |
| `retry.py` | `RetryPolicy` with exponential backoff + jitter |
| `status.py` | `JobStatusTracker` for Redis hash-based status |
| `config.py` | `QueueConfig` with environment variable support |

## Usage

### Enqueueing a Debate

```python
from aragora.queue import create_redis_queue, create_debate_job

# Create queue connection
queue = await create_redis_queue()

# Create a debate job
job = create_debate_job(
    question="Should we use microservices?",
    agents=["claude", "gpt"],
    rounds=3,
    consensus="majority",
)

# Enqueue the job
job_id = await queue.enqueue(job)
print(f"Job enqueued: {job_id}")

# Check status
status = await queue.get_status(job_id)
print(f"Status: {status}")
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

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `ARAGORA_QUEUE_PREFIX` | `aragora:queue:` | Key prefix for queue data |
| `ARAGORA_QUEUE_RETRY_MAX` | `3` | Maximum retry attempts |
| `ARAGORA_QUEUE_CLAIM_IDLE_MS` | `60000` | Claim stale jobs after (ms) |
| `ARAGORA_QUEUE_MAX_JOB_TTL_DAYS` | `7` | Job data retention (days) |

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
)
set_queue_config(config)
```

## Retry Policy

Jobs are retried with exponential backoff and jitter:

```python
from aragora.queue import RetryPolicy

policy = RetryPolicy(
    max_attempts=3,           # Total attempts before permanent failure
    base_delay_seconds=1.0,   # Initial delay
    max_delay_seconds=300.0,  # Maximum delay cap
    exponential_base=2.0,     # Delay multiplier per attempt
    jitter=True,              # Add randomness to prevent thundering herd
)

# Delay calculation: base * (exponential_base ^ attempt) ± 20% jitter
# Attempt 0: 1.0s (± 20%)
# Attempt 1: 2.0s (± 20%)
# Attempt 2: 4.0s (± 20%)
```

### Retryable vs Non-Retryable Errors

**Retryable** (will be retried):
- `ConnectionError`, `TimeoutError`
- `RuntimeError`, generic `Exception`
- Temporary API failures (rate limits, server errors)

**Non-Retryable** (immediate failure):
- `ValueError`, `TypeError`, `KeyError`
- Errors with "invalid", "not found", "validation" in message

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

## Monitoring

### Queue Statistics

```python
stats = await queue.get_queue_stats()
print(f"Stream length: {stats['stream_length']}")
print(f"Pending jobs: {stats['pending_count']}")
print(f"Consumer groups: {stats['groups']}")
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

## Redis Streams Commands

The queue uses these Redis Streams operations:

| Operation | Redis Command | Purpose |
|-----------|---------------|---------|
| Enqueue | `XADD` | Add job to stream |
| Dequeue | `XREADGROUP` | Read job for consumer |
| Acknowledge | `XACK` | Mark job as processed |
| Claim stale | `XCLAIM` | Take over from dead worker |
| Get pending | `XPENDING` | List unacknowledged jobs |
| Stream length | `XLEN` | Count total jobs |

## Testing

Run queue tests:

```bash
# Unit tests
pytest tests/test_queue.py -v

# With coverage
pytest tests/test_queue.py --cov=aragora.queue
```

## Troubleshooting

### Jobs Stuck in PENDING

1. Check Redis connection: `redis-cli ping`
2. Verify workers are running: `docker compose ps`
3. Check worker logs: `docker compose logs debate-worker`

### Jobs Not Being Processed

1. Verify consumer group exists: `redis-cli XINFO GROUPS aragora:queue:debates:stream`
2. Check for pending jobs: `redis-cli XPENDING aragora:queue:debates:stream aragora-workers`
3. Manually claim stale jobs if needed

### High Failure Rate

1. Check retry policy configuration
2. Review error logs for patterns
3. Verify API keys and credentials
4. Consider increasing `max_attempts` for transient issues
