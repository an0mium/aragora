"""
Queue configuration settings.

Environment Variables:
    REDIS_URL: Redis connection URL (default: redis://localhost:6379)
    ARAGORA_QUEUE_PREFIX: Key prefix for queue keys (default: aragora:queue:)
    ARAGORA_QUEUE_MAX_TTL_DAYS: Max job TTL in days (default: 7)
    ARAGORA_QUEUE_CLAIM_IDLE_MS: Idle time before claiming stale jobs (default: 60000)
    ARAGORA_QUEUE_RETRY_MAX: Max retry attempts (default: 3)
    ARAGORA_QUEUE_RETRY_BASE_DELAY: Base delay for retries in seconds (default: 1.0)
    ARAGORA_QUEUE_RETRY_MAX_DELAY: Max delay for retries in seconds (default: 300.0)
    ARAGORA_QUEUE_WORKER_BLOCK_MS: Block time for XREADGROUP in ms (default: 5000)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QueueConfig:
    """Configuration for the Redis Streams queue system."""

    # Redis connection
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))

    # Key prefix for all queue-related Redis keys
    key_prefix: str = field(
        default_factory=lambda: os.getenv("ARAGORA_QUEUE_PREFIX", "aragora:queue:")
    )

    # Job retention
    max_job_ttl_days: int = field(
        default_factory=lambda: int(os.getenv("ARAGORA_QUEUE_MAX_TTL_DAYS", "7"))
    )

    # Stale job claiming (for dead worker recovery)
    claim_idle_ms: int = field(
        default_factory=lambda: int(os.getenv("ARAGORA_QUEUE_CLAIM_IDLE_MS", "60000"))
    )

    # Retry configuration
    retry_max_attempts: int = field(
        default_factory=lambda: int(os.getenv("ARAGORA_QUEUE_RETRY_MAX", "3"))
    )
    retry_base_delay: float = field(
        default_factory=lambda: float(os.getenv("ARAGORA_QUEUE_RETRY_BASE_DELAY", "1.0"))
    )
    retry_max_delay: float = field(
        default_factory=lambda: float(os.getenv("ARAGORA_QUEUE_RETRY_MAX_DELAY", "300.0"))
    )

    # Worker settings
    worker_block_ms: int = field(
        default_factory=lambda: int(os.getenv("ARAGORA_QUEUE_WORKER_BLOCK_MS", "5000"))
    )

    # Consumer group settings
    consumer_group: str = field(
        default_factory=lambda: os.getenv("ARAGORA_QUEUE_CONSUMER_GROUP", "debate-workers")
    )

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_job_ttl_days < 1 or self.max_job_ttl_days > 30:
            raise ValueError(f"max_job_ttl_days must be 1-30, got {self.max_job_ttl_days}")
        if self.claim_idle_ms < 10000 or self.claim_idle_ms > 600000:
            raise ValueError(f"claim_idle_ms must be 10000-600000, got {self.claim_idle_ms}")
        if self.retry_max_attempts < 1 or self.retry_max_attempts > 10:
            raise ValueError(f"retry_max_attempts must be 1-10, got {self.retry_max_attempts}")
        if self.retry_base_delay < 0.1 or self.retry_base_delay > 60.0:
            raise ValueError(f"retry_base_delay must be 0.1-60.0, got {self.retry_base_delay}")
        if self.retry_max_delay < 1.0 or self.retry_max_delay > 3600.0:
            raise ValueError(f"retry_max_delay must be 1.0-3600.0, got {self.retry_max_delay}")
        if self.worker_block_ms < 1000 or self.worker_block_ms > 30000:
            raise ValueError(f"worker_block_ms must be 1000-30000, got {self.worker_block_ms}")

    @property
    def stream_key(self) -> str:
        """Get the Redis Stream key for debates."""
        return f"{self.key_prefix}debates:stream"

    @property
    def status_key_prefix(self) -> str:
        """Get the prefix for job status hash keys."""
        return f"{self.key_prefix}job:"

    @property
    def job_ttl_seconds(self) -> int:
        """Get job TTL in seconds."""
        return self.max_job_ttl_days * 86400


# Global config instance (lazy loaded)
_config: Optional[QueueConfig] = None


def get_queue_config() -> QueueConfig:
    """Get the global queue configuration."""
    global _config
    if _config is None:
        _config = QueueConfig()
    return _config


def set_queue_config(config: QueueConfig) -> None:
    """Set a custom queue configuration (for testing)."""
    global _config
    _config = config


def reset_queue_config() -> None:
    """Reset to default configuration (for testing)."""
    global _config
    _config = None
