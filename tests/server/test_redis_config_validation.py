"""
Tests for Redis connection pool configuration validation.

Tests cover:
- Valid max_connections values accepted without clamping
- max_connections < 1 is clamped to 1
- max_connections > 10000 is capped to 10000
- Default value of 50 passes validation
- Warning logs are emitted when values are clamped
"""

from __future__ import annotations

import importlib
import logging
from unittest import mock

import pytest


def _reset_and_get_pool(env_overrides: dict[str, str] | None = None):
    """Reset Redis module state and initialize pool with optional env var overrides.

    The Redis module uses module-level state that must be reset between tests.
    Returns the pool and availability status after initialization.
    """
    import aragora.server.redis_config as redis_mod

    # Reset module state
    redis_mod.reset_redis_state()

    # Build environment with required URL
    env = {
        "ARAGORA_REDIS_URL": "redis://localhost:6379",
    }
    if env_overrides:
        env.update(env_overrides)

    with mock.patch.dict("os.environ", env, clear=False):
        # Mock redis to avoid actual connection
        mock_pool = mock.MagicMock()
        mock_redis_class = mock.MagicMock()
        mock_redis_class.return_value.ping.return_value = True
        mock_connection_pool = mock.MagicMock()
        mock_connection_pool.from_url.return_value = mock_pool

        with mock.patch.object(redis_mod, "_redis_pool", None):
            with mock.patch.object(redis_mod, "_redis_available", None):
                with mock.patch("redis.ConnectionPool", mock_connection_pool):
                    with mock.patch("redis.Redis", mock_redis_class):
                        pool = redis_mod.get_redis_pool()

        # Return the call args to check what max_connections was used
        if mock_connection_pool.from_url.called:
            call_kwargs = mock_connection_pool.from_url.call_args[1]
            return call_kwargs.get("max_connections")
        return None


class TestMaxConnectionsDefaults:
    """Test that default max_connections value is valid."""

    def test_default_value_is_50(self):
        """Default max_connections is 50 when env var is not set."""
        max_connections = _reset_and_get_pool()
        assert max_connections == 50

    def test_default_value_is_within_bounds(self):
        """Default max_connections is within valid bounds."""
        max_connections = _reset_and_get_pool()
        assert max_connections is not None
        assert max_connections >= 1
        assert max_connections <= 10000


class TestMaxConnectionsValidation:
    """Test max_connections validation and clamping."""

    def test_valid_max_connections_accepted(self):
        """A valid max_connections (e.g. 100) is accepted without clamping."""
        max_connections = _reset_and_get_pool({"ARAGORA_REDIS_MAX_CONNECTIONS": "100"})
        assert max_connections == 100

    def test_max_connections_of_1_accepted(self):
        """max_connections of exactly 1 is accepted (boundary)."""
        max_connections = _reset_and_get_pool({"ARAGORA_REDIS_MAX_CONNECTIONS": "1"})
        assert max_connections == 1

    def test_max_connections_of_10000_accepted(self):
        """max_connections of exactly 10000 is accepted (boundary)."""
        max_connections = _reset_and_get_pool({"ARAGORA_REDIS_MAX_CONNECTIONS": "10000"})
        assert max_connections == 10000

    def test_max_connections_zero_clamped_to_1(self, caplog):
        """max_connections of 0 is clamped to 1."""
        with caplog.at_level(logging.WARNING, logger="aragora.server.redis_config"):
            max_connections = _reset_and_get_pool({"ARAGORA_REDIS_MAX_CONNECTIONS": "0"})
        assert max_connections == 1

    def test_max_connections_negative_clamped_to_1(self, caplog):
        """max_connections of -5 is clamped to 1."""
        with caplog.at_level(logging.WARNING, logger="aragora.server.redis_config"):
            max_connections = _reset_and_get_pool({"ARAGORA_REDIS_MAX_CONNECTIONS": "-5"})
        assert max_connections == 1

    def test_max_connections_exceeding_10000_capped(self, caplog):
        """max_connections > 10000 is capped to 10000."""
        with caplog.at_level(logging.WARNING, logger="aragora.server.redis_config"):
            max_connections = _reset_and_get_pool({"ARAGORA_REDIS_MAX_CONNECTIONS": "15000"})
        assert max_connections == 10000


class TestMaxConnectionsWarningLogs:
    """Test that warning logs are emitted when values are clamped."""

    def test_clamping_to_1_logs_warning(self, caplog):
        """Clamping max_connections to 1 logs a warning."""
        with caplog.at_level(logging.WARNING, logger="aragora.server.redis_config"):
            _reset_and_get_pool({"ARAGORA_REDIS_MAX_CONNECTIONS": "0"})
        assert any("clamping to 1" in record.message for record in caplog.records)

    def test_capping_at_10000_logs_warning(self, caplog):
        """Capping max_connections to 10000 logs a warning."""
        with caplog.at_level(logging.WARNING, logger="aragora.server.redis_config"):
            _reset_and_get_pool({"ARAGORA_REDIS_MAX_CONNECTIONS": "20000"})
        assert any("capping at 10000" in record.message for record in caplog.records)
