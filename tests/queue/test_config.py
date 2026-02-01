"""
Tests for queue configuration module.

Tests cover:
- QueueConfig dataclass defaults and env var loading
- Validation rules for all bounded fields
- Computed properties (stream_key, status_key_prefix, job_ttl_seconds)
- Singleton pattern (get_queue_config, set_queue_config, reset_queue_config)
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from aragora.queue.config import (
    QueueConfig,
    get_queue_config,
    reset_queue_config,
    set_queue_config,
)


class TestQueueConfigDefaults:
    """Tests for QueueConfig default values."""

    def test_default_redis_url(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REDIS_URL", None)
            config = QueueConfig()
        assert config.redis_url == "redis://localhost:6379"

    def test_default_key_prefix(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARAGORA_QUEUE_PREFIX", None)
            config = QueueConfig()
        assert config.key_prefix == "aragora:queue:"

    def test_default_max_job_ttl_days(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARAGORA_QUEUE_MAX_TTL_DAYS", None)
            config = QueueConfig()
        assert config.max_job_ttl_days == 7

    def test_default_claim_idle_ms(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARAGORA_QUEUE_CLAIM_IDLE_MS", None)
            config = QueueConfig()
        assert config.claim_idle_ms == 60000

    def test_default_retry_settings(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARAGORA_QUEUE_RETRY_MAX", None)
            os.environ.pop("ARAGORA_QUEUE_RETRY_BASE_DELAY", None)
            os.environ.pop("ARAGORA_QUEUE_RETRY_MAX_DELAY", None)
            config = QueueConfig()
        assert config.retry_max_attempts == 3
        assert config.retry_base_delay == 1.0
        assert config.retry_max_delay == 300.0

    def test_default_worker_block_ms(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARAGORA_QUEUE_WORKER_BLOCK_MS", None)
            config = QueueConfig()
        assert config.worker_block_ms == 5000

    def test_default_consumer_group(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARAGORA_QUEUE_CONSUMER_GROUP", None)
            config = QueueConfig()
        assert config.consumer_group == "debate-workers"


class TestQueueConfigEnvVars:
    """Tests for QueueConfig loading from environment variables."""

    def test_redis_url_from_env(self):
        with patch.dict(os.environ, {"REDIS_URL": "redis://custom:6380"}, clear=False):
            config = QueueConfig()
        assert config.redis_url == "redis://custom:6380"

    def test_key_prefix_from_env(self):
        with patch.dict(os.environ, {"ARAGORA_QUEUE_PREFIX": "test:queue:"}, clear=False):
            config = QueueConfig()
        assert config.key_prefix == "test:queue:"

    def test_max_ttl_from_env(self):
        with patch.dict(os.environ, {"ARAGORA_QUEUE_MAX_TTL_DAYS": "14"}, clear=False):
            config = QueueConfig()
        assert config.max_job_ttl_days == 14

    def test_claim_idle_from_env(self):
        with patch.dict(os.environ, {"ARAGORA_QUEUE_CLAIM_IDLE_MS": "120000"}, clear=False):
            config = QueueConfig()
        assert config.claim_idle_ms == 120000

    def test_retry_max_from_env(self):
        with patch.dict(os.environ, {"ARAGORA_QUEUE_RETRY_MAX": "5"}, clear=False):
            config = QueueConfig()
        assert config.retry_max_attempts == 5

    def test_worker_block_from_env(self):
        with patch.dict(os.environ, {"ARAGORA_QUEUE_WORKER_BLOCK_MS": "10000"}, clear=False):
            config = QueueConfig()
        assert config.worker_block_ms == 10000


class TestQueueConfigValidation:
    """Tests for QueueConfig validation rules in __post_init__."""

    def test_max_job_ttl_too_low(self):
        with pytest.raises(ValueError, match="max_job_ttl_days must be 1-30"):
            QueueConfig(max_job_ttl_days=0)

    def test_max_job_ttl_too_high(self):
        with pytest.raises(ValueError, match="max_job_ttl_days must be 1-30"):
            QueueConfig(max_job_ttl_days=31)

    def test_max_job_ttl_boundary_low(self):
        config = QueueConfig(max_job_ttl_days=1)
        assert config.max_job_ttl_days == 1

    def test_max_job_ttl_boundary_high(self):
        config = QueueConfig(max_job_ttl_days=30)
        assert config.max_job_ttl_days == 30

    def test_claim_idle_too_low(self):
        with pytest.raises(ValueError, match="claim_idle_ms must be 10000-600000"):
            QueueConfig(claim_idle_ms=9999)

    def test_claim_idle_too_high(self):
        with pytest.raises(ValueError, match="claim_idle_ms must be 10000-600000"):
            QueueConfig(claim_idle_ms=600001)

    def test_claim_idle_boundary_low(self):
        config = QueueConfig(claim_idle_ms=10000)
        assert config.claim_idle_ms == 10000

    def test_claim_idle_boundary_high(self):
        config = QueueConfig(claim_idle_ms=600000)
        assert config.claim_idle_ms == 600000

    def test_retry_max_attempts_too_low(self):
        with pytest.raises(ValueError, match="retry_max_attempts must be 1-10"):
            QueueConfig(retry_max_attempts=0)

    def test_retry_max_attempts_too_high(self):
        with pytest.raises(ValueError, match="retry_max_attempts must be 1-10"):
            QueueConfig(retry_max_attempts=11)

    def test_retry_base_delay_too_low(self):
        with pytest.raises(ValueError, match="retry_base_delay must be 0.1-60.0"):
            QueueConfig(retry_base_delay=0.05)

    def test_retry_base_delay_too_high(self):
        with pytest.raises(ValueError, match="retry_base_delay must be 0.1-60.0"):
            QueueConfig(retry_base_delay=61.0)

    def test_retry_max_delay_too_low(self):
        with pytest.raises(ValueError, match="retry_max_delay must be 1.0-3600.0"):
            QueueConfig(retry_max_delay=0.5)

    def test_retry_max_delay_too_high(self):
        with pytest.raises(ValueError, match="retry_max_delay must be 1.0-3600.0"):
            QueueConfig(retry_max_delay=3601.0)

    def test_worker_block_too_low(self):
        with pytest.raises(ValueError, match="worker_block_ms must be 1000-30000"):
            QueueConfig(worker_block_ms=999)

    def test_worker_block_too_high(self):
        with pytest.raises(ValueError, match="worker_block_ms must be 1000-30000"):
            QueueConfig(worker_block_ms=30001)


class TestQueueConfigProperties:
    """Tests for QueueConfig computed properties."""

    def test_stream_key(self):
        config = QueueConfig(key_prefix="test:")
        assert config.stream_key == "test:debates:stream"

    def test_stream_key_default(self):
        config = QueueConfig()
        assert config.stream_key == "aragora:queue:debates:stream"

    def test_status_key_prefix(self):
        config = QueueConfig(key_prefix="test:")
        assert config.status_key_prefix == "test:job:"

    def test_status_key_prefix_default(self):
        config = QueueConfig()
        assert config.status_key_prefix == "aragora:queue:job:"

    def test_job_ttl_seconds_default(self):
        config = QueueConfig(max_job_ttl_days=7)
        assert config.job_ttl_seconds == 7 * 86400

    def test_job_ttl_seconds_one_day(self):
        config = QueueConfig(max_job_ttl_days=1)
        assert config.job_ttl_seconds == 86400

    def test_job_ttl_seconds_thirty_days(self):
        config = QueueConfig(max_job_ttl_days=30)
        assert config.job_ttl_seconds == 30 * 86400


class TestQueueConfigSingleton:
    """Tests for the module-level singleton functions."""

    def setup_method(self):
        reset_queue_config()

    def teardown_method(self):
        reset_queue_config()

    def test_get_queue_config_creates_default(self):
        config = get_queue_config()
        assert isinstance(config, QueueConfig)

    def test_get_queue_config_returns_same_instance(self):
        config1 = get_queue_config()
        config2 = get_queue_config()
        assert config1 is config2

    def test_set_queue_config(self):
        custom = QueueConfig(max_job_ttl_days=14)
        set_queue_config(custom)
        config = get_queue_config()
        assert config is custom
        assert config.max_job_ttl_days == 14

    def test_reset_queue_config(self):
        config1 = get_queue_config()
        reset_queue_config()
        config2 = get_queue_config()
        assert config1 is not config2

    def test_set_then_reset(self):
        custom = QueueConfig(max_job_ttl_days=20)
        set_queue_config(custom)
        assert get_queue_config().max_job_ttl_days == 20

        reset_queue_config()
        assert get_queue_config().max_job_ttl_days == 7  # back to default
