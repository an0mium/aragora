#!/usr/bin/env python3
"""
Debate queue worker process.

Processes debate jobs from the Redis Streams queue with horizontal scaling support.

Usage:
    python -m scripts.queue_worker --worker-id worker-1 --concurrency 3

    # With custom Redis URL
    REDIS_URL=redis://localhost:6379 python -m scripts.queue_worker

    # Run multiple workers for scaling
    python -m scripts.queue_worker --worker-id worker-1 &
    python -m scripts.queue_worker --worker-id worker-2 &

Environment Variables:
    REDIS_URL: Redis connection URL (default: redis://localhost:6379)
    ARAGORA_QUEUE_*: Queue configuration (see aragora.queue.config)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import socket
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the worker."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from some loggers
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.WARNING)


async def main(
    worker_id: str,
    concurrency: int,
    redis_url: str | None,
) -> None:
    """
    Main worker entry point.

    Args:
        worker_id: Unique worker identifier
        concurrency: Maximum concurrent jobs
        redis_url: Optional Redis URL override
    """
    from aragora.queue import (
        create_redis_queue,
        DebateWorker,
        create_default_executor,
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Starting worker {worker_id} with concurrency={concurrency}")

    # Create queue
    queue = await create_redis_queue(redis_url=redis_url, consumer_name=worker_id)

    # Create executor
    executor = await create_default_executor()

    # Create and start worker
    worker = DebateWorker(
        queue=queue,
        worker_id=worker_id,
        executor=executor,
        max_concurrent=concurrency,
    )

    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
    finally:
        await worker.stop(timeout=30.0)
        logger.info("Worker shutdown complete")


def generate_worker_id() -> str:
    """Generate a unique worker ID based on hostname and PID."""
    hostname = socket.gethostname()
    pid = os.getpid()
    return f"{hostname}-{pid}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debate queue worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--worker-id",
        default=None,
        help="Unique worker ID (default: auto-generated from hostname-pid)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Maximum concurrent jobs (default: 3)",
    )
    parser.add_argument(
        "--redis-url",
        default=None,
        help="Redis URL (default: from REDIS_URL env or redis://localhost:6379)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)

    # Generate worker ID if not provided
    worker_id = args.worker_id or generate_worker_id()

    # Run the worker
    asyncio.run(
        main(
            worker_id=worker_id,
            concurrency=args.concurrency,
            redis_url=args.redis_url,
        )
    )
