"""
Performance tests for health check endpoints.

These tests verify that health endpoints meet latency SLOs
and properly utilize caching for repeated requests.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin.health import (
    HealthHandler,
    _get_cached_health,
    _set_cached_health,
    _HEALTH_CACHE,
    _HEALTH_CACHE_TIMESTAMPS,
)


def create_mock_context():
    """Create a mock ServerContext for testing."""
    ctx = MagicMock()
    ctx.storage = None
    ctx.elo = None
    ctx.debate_store = None
    return ctx


class TestHealthCaching:
    """Tests for health check result caching."""

    def setup_method(self):
        """Clear cache before each test."""
        _HEALTH_CACHE.clear()
        _HEALTH_CACHE_TIMESTAMPS.clear()

    def test_cache_set_and_get(self):
        """Test basic cache set and get."""
        result = {"status": "ready", "checks": {}}
        _set_cached_health("test_key", result)

        cached = _get_cached_health("test_key")
        assert cached == result

    def test_cache_expires_after_ttl(self):
        """Test cache expires after TTL."""
        result = {"status": "ready", "checks": {}}
        _set_cached_health("test_key", result)

        # Manually set timestamp to expired (TTL is 5 seconds, use 6 to ensure expiry)
        _HEALTH_CACHE_TIMESTAMPS["test_key"] = time.time() - 6.0  # 6 seconds ago

        cached = _get_cached_health("test_key")
        assert cached is None

    def test_cache_returns_none_for_missing_key(self):
        """Test cache returns None for non-existent key."""
        cached = _get_cached_health("nonexistent")
        assert cached is None


class TestLivenessProbePerformance:
    """Tests for liveness probe (/healthz) performance."""

    @pytest.fixture
    def handler(self):
        """Create a HealthHandler instance."""
        return HealthHandler(create_mock_context())

    def test_liveness_probe_fast(self, handler):
        """Liveness probe should complete in under 10ms."""
        start = time.perf_counter()
        result = handler._liveness_probe()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 10, f"Liveness probe took {elapsed_ms:.2f}ms (should be <10ms)"
        assert result.status_code == 200

    def test_liveness_probe_returns_ok(self, handler):
        """Liveness probe should return ok status."""
        result = handler._liveness_probe()
        assert result.status_code == 200

        import json

        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "ok"


class TestReadinessProbePerformance:
    """Tests for readiness probe (/readyz) performance."""

    @pytest.fixture
    def handler(self):
        """Create a HealthHandler instance."""
        handler = HealthHandler(create_mock_context())
        # Mock storage and ELO to avoid actual initialization
        handler.get_storage = MagicMock(return_value=None)
        handler.get_elo_system = MagicMock(return_value=None)
        return handler

    def setup_method(self):
        """Clear cache before each test."""
        _HEALTH_CACHE.clear()
        _HEALTH_CACHE_TIMESTAMPS.clear()

    def test_readiness_returns_latency(self, handler):
        """Readiness probe should include latency_ms in response."""
        with patch.dict("os.environ", {}, clear=False):
            result = handler._readiness_probe_fast()

        import json

        body = json.loads(result.body.decode("utf-8"))
        assert "latency_ms" in body
        assert isinstance(body["latency_ms"], (int, float))

    def test_readiness_uses_cache(self, handler):
        """Second readiness check should use cache."""
        with patch.dict("os.environ", {}, clear=False):
            # First call - should compute
            result1 = handler._readiness_probe_fast()

            # Second call - should use cache
            start = time.perf_counter()
            result2 = handler._readiness_probe_fast()
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Cached response should be very fast
            assert elapsed_ms < 1, f"Cached readiness took {elapsed_ms:.2f}ms (should be <1ms)"

        import json

        body1 = json.loads(result1.body.decode("utf-8"))
        body2 = json.loads(result2.body.decode("utf-8"))

        # Both should have same status
        assert body1["status"] == body2["status"]


class TestHealthEndpointRouting:
    """Tests for health endpoint routing."""

    @pytest.fixture
    def handler(self):
        """Create a HealthHandler instance."""
        return HealthHandler(create_mock_context())

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("/healthz", True),
            ("/readyz", True),
            ("/api/health", True),
            ("/api/v1/health", True),
            ("/api/health/detailed", True),
            ("/api/v1/health/detailed", True),
            ("/unknown", False),
            ("/api/other", False),
        ],
    )
    def test_can_handle(self, handler, path, expected):
        """Test handler routing for health endpoints."""
        assert handler.can_handle(path) == expected


class TestHealthEndpointSLO:
    """Tests for health endpoint SLO compliance."""

    @pytest.fixture
    def handler(self):
        """Create a HealthHandler instance."""
        handler = HealthHandler(create_mock_context())
        handler.get_storage = MagicMock(return_value=None)
        handler.get_elo_system = MagicMock(return_value=None)
        return handler

    def test_liveness_slo_10ms(self, handler):
        """Liveness endpoint must complete in <10ms."""
        times = []
        for _ in range(10):
            start = time.perf_counter()
            handler._liveness_probe()
            times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)
        max_time = max(times)

        assert avg_time < 5, f"Average liveness time {avg_time:.2f}ms exceeds 5ms"
        assert max_time < 10, f"Max liveness time {max_time:.2f}ms exceeds 10ms"

    def test_cached_readiness_slo_1ms(self, handler):
        """Cached readiness endpoint must complete in <1ms."""
        _HEALTH_CACHE.clear()
        _HEALTH_CACHE_TIMESTAMPS.clear()

        with patch.dict("os.environ", {}, clear=False):
            # Prime the cache
            handler._readiness_probe_fast()

            # Measure cached responses
            times = []
            for _ in range(10):
                start = time.perf_counter()
                handler._readiness_probe_fast()
                times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)
        max_time = max(times)

        assert avg_time < 0.5, f"Average cached readiness {avg_time:.2f}ms exceeds 0.5ms"
        assert max_time < 1, f"Max cached readiness {max_time:.2f}ms exceeds 1ms"
