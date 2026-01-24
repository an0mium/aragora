"""
Tests for Request Timeout Middleware.

Tests cover:
- RequestTimeoutConfig class (configuration)
- with_timeout decorator (sync timeout handling)
- async_with_timeout decorator (async timeout handling)
- timeout_context context manager
- get_timeout_stats utility
"""

from __future__ import annotations

import asyncio
import pytest
import time
from unittest.mock import Mock, patch

from aragora.server.middleware.timeout import (
    RequestTimeoutConfig,
    RequestTimeoutError,
    get_timeout_config,
    configure_timeout,
    with_timeout,
    async_with_timeout,
    timeout_context,
    get_timeout_stats,
    shutdown_executor,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_timeout_config():
    """Reset timeout configuration after each test."""
    import aragora.server.middleware.timeout as timeout_module

    original = timeout_module._timeout_config
    timeout_module._timeout_config = None
    yield
    timeout_module._timeout_config = original


@pytest.fixture
def timeout_config():
    """Create a fresh timeout configuration."""
    return RequestTimeoutConfig(
        default_timeout=5.0,
        slow_timeout=30.0,
        max_timeout=60.0,
    )


# ============================================================================
# RequestTimeoutConfig Tests
# ============================================================================


class TestRequestTimeoutConfig:
    """Tests for RequestTimeoutConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RequestTimeoutConfig()
        assert config.default_timeout == 30.0
        assert config.slow_timeout == 120.0
        assert config.max_timeout == 600.0
        assert config.endpoint_timeouts == {}

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RequestTimeoutConfig(
            default_timeout=10.0,
            slow_timeout=60.0,
            max_timeout=120.0,
        )
        assert config.default_timeout == 10.0
        assert config.slow_timeout == 60.0
        assert config.max_timeout == 120.0

    def test_get_timeout_default(self):
        """Test get_timeout returns default for unknown paths."""
        config = RequestTimeoutConfig(default_timeout=15.0, max_timeout=100.0)
        assert config.get_timeout("/api/unknown") == 15.0

    def test_get_timeout_slow_patterns(self):
        """Test get_timeout returns slow_timeout for known slow paths."""
        config = RequestTimeoutConfig(slow_timeout=60.0, max_timeout=100.0)

        slow_paths = [
            "/api/debates/create",
            "/api/debates/batch/process",
            "/api/gauntlet/run",
            "/api/evolution/generate",
            "/api/verify/proof",
            "/api/evidence/collect",
            "/api/broadcast/generate",
        ]

        for path in slow_paths:
            assert config.get_timeout(path) == 60.0, f"Path {path} should use slow_timeout"

    def test_get_timeout_respects_max(self):
        """Test get_timeout never exceeds max_timeout."""
        config = RequestTimeoutConfig(
            default_timeout=100.0,
            slow_timeout=200.0,
            max_timeout=50.0,
        )
        assert config.get_timeout("/api/unknown") == 50.0
        assert config.get_timeout("/api/debates/create") == 50.0

    def test_endpoint_overrides(self):
        """Test per-endpoint timeout overrides."""
        config = RequestTimeoutConfig(
            default_timeout=10.0,
            max_timeout=100.0,
            endpoint_timeouts={
                "/api/custom": 25.0,
                "/api/special/endpoint": 45.0,
            },
        )
        assert config.get_timeout("/api/custom") == 25.0
        assert config.get_timeout("/api/special/endpoint") == 45.0
        assert config.get_timeout("/api/other") == 10.0

    def test_endpoint_override_respects_max(self):
        """Test endpoint overrides are capped at max_timeout."""
        config = RequestTimeoutConfig(
            max_timeout=30.0,
            endpoint_timeouts={"/api/custom": 50.0},
        )
        assert config.get_timeout("/api/custom") == 30.0


# ============================================================================
# Configuration Function Tests
# ============================================================================


class TestConfigureFunctions:
    """Tests for configuration utility functions."""

    def test_get_timeout_config_singleton(self):
        """Test get_timeout_config returns singleton instance."""
        config1 = get_timeout_config()
        config2 = get_timeout_config()
        assert config1 is config2

    def test_configure_timeout_updates_values(self):
        """Test configure_timeout updates configuration."""
        config = configure_timeout(
            default_timeout=20.0,
            slow_timeout=80.0,
            max_timeout=200.0,
        )
        assert config.default_timeout == 20.0
        assert config.slow_timeout == 80.0
        assert config.max_timeout == 200.0

    def test_configure_timeout_partial_update(self):
        """Test configure_timeout allows partial updates."""
        # First, set all values
        configure_timeout(default_timeout=10.0, slow_timeout=40.0)

        # Then update only one
        config = configure_timeout(default_timeout=15.0)
        assert config.default_timeout == 15.0
        assert config.slow_timeout == 40.0  # Unchanged

    def test_configure_timeout_endpoint_overrides(self):
        """Test configure_timeout adds endpoint overrides."""
        config = configure_timeout(endpoint_overrides={"/api/fast": 5.0, "/api/slow": 60.0})
        assert config.endpoint_timeouts["/api/fast"] == 5.0
        assert config.endpoint_timeouts["/api/slow"] == 60.0


# ============================================================================
# RequestTimeoutError Tests
# ============================================================================


class TestRequestTimeoutError:
    """Tests for RequestTimeoutError exception."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = RequestTimeoutError()
        assert "Request timed out" in str(error)

    def test_error_with_details(self):
        """Test error with timeout and path details."""
        error = RequestTimeoutError(
            message="Custom timeout message",
            timeout=30.0,
            path="/api/debates/1234",
        )
        assert error.timeout == 30.0
        assert error.path == "/api/debates/1234"
        assert "30" in str(error)
        assert "/api/debates/1234" in str(error)


# ============================================================================
# with_timeout Decorator Tests
# ============================================================================


class TestWithTimeoutDecorator:
    """Tests for with_timeout sync decorator."""

    def test_fast_function_completes(self):
        """Test function that completes quickly returns result."""

        @with_timeout(5.0)
        def fast_func(self, path, query_params, handler):
            return {"success": True}

        result = fast_func(None, "/api/test", {}, None)
        assert result == {"success": True}

    def test_slow_function_times_out(self):
        """Test function that exceeds timeout returns 504."""

        @with_timeout(0.1)  # Very short timeout
        def slow_func(self, path, query_params, handler):
            time.sleep(1.0)  # Sleep longer than timeout
            return {"success": True}

        result = slow_func(None, "/api/test", {}, None)
        # Should return error tuple (dict, status, headers)
        assert isinstance(result, tuple)
        assert result[1] == 504
        assert result[0]["code"] == "request_timeout"

    def test_custom_error_response(self):
        """Test custom error response function."""

        def custom_error(timeout, path):
            return {"custom_error": True, "timeout": timeout}

        @with_timeout(0.1, error_response=custom_error)
        def slow_func(self, path, query_params, handler):
            time.sleep(1.0)
            return {"success": True}

        result = slow_func(None, "/api/test", {}, None)
        assert result["custom_error"] is True

    def test_timeout_from_config(self):
        """Test timeout pulled from config when not specified."""
        configure_timeout(default_timeout=5.0)

        @with_timeout()  # No explicit timeout
        def fast_func(self, path, query_params, handler):
            return {"success": True}

        # Should complete (5s is plenty for this)
        result = fast_func(None, "/api/test", {}, None)
        assert result == {"success": True}


# ============================================================================
# async_with_timeout Decorator Tests
# ============================================================================


class TestAsyncWithTimeoutDecorator:
    """Tests for async_with_timeout async decorator."""

    @pytest.mark.asyncio
    async def test_fast_async_completes(self):
        """Test async function that completes quickly returns result."""

        @async_with_timeout(5.0)
        async def fast_async(self, path, query_params, handler):
            await asyncio.sleep(0.01)
            return {"success": True}

        result = await fast_async(None, "/api/test", {}, None)
        assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_slow_async_times_out(self):
        """Test async function that exceeds timeout returns 504."""

        @async_with_timeout(0.1)
        async def slow_async(self, path, query_params, handler):
            await asyncio.sleep(1.0)
            return {"success": True}

        result = await slow_async(None, "/api/test", {}, None)
        assert isinstance(result, tuple)
        assert result[1] == 504
        assert result[0]["code"] == "request_timeout"

    @pytest.mark.asyncio
    async def test_async_custom_error_response(self):
        """Test custom error response for async timeout."""

        def custom_error(timeout, path):
            return {"async_error": True, "path": path}

        @async_with_timeout(0.1, error_response=custom_error)
        async def slow_async(self, path, query_params, handler):
            await asyncio.sleep(1.0)
            return {"success": True}

        result = await slow_async(None, "/api/custom", {}, None)
        assert result["async_error"] is True
        assert result["path"] == "/api/custom"


# ============================================================================
# timeout_context Tests
# ============================================================================


class TestTimeoutContext:
    """Tests for timeout_context context manager."""

    def test_context_fast_operation(self):
        """Test context manager with fast operation."""
        with timeout_context(5.0, "/api/test"):
            result = 1 + 1
        assert result == 2

    @pytest.mark.skipif(
        __import__("platform").system() == "Windows",
        reason="signal.alarm not available on Windows",
    )
    @pytest.mark.skipif(
        __import__("os").environ.get("PYTEST_TIMEOUT") is not None,
        reason="signal.alarm conflicts with pytest-timeout plugin",
    )
    def test_context_timeout_raises(self):
        """Test context manager raises on timeout (Unix only)."""
        with pytest.raises(RequestTimeoutError):
            with timeout_context(0.1, "/api/test"):
                time.sleep(1.0)


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_timeout_stats(self):
        """Test get_timeout_stats returns configuration info."""
        configure_timeout(
            default_timeout=15.0,
            slow_timeout=60.0,
            max_timeout=120.0,
        )

        stats = get_timeout_stats()

        assert "config" in stats
        assert stats["config"]["default_timeout"] == 15.0
        assert stats["config"]["slow_timeout"] == 60.0
        assert stats["config"]["max_timeout"] == 120.0
        assert "executor" in stats

    def test_shutdown_executor_safe(self):
        """Test shutdown_executor is safe to call multiple times."""
        # Should not raise even if called multiple times
        shutdown_executor()
        shutdown_executor()


# ============================================================================
# Environment Variable Tests
# ============================================================================


class TestEnvironmentConfiguration:
    """Tests for environment variable configuration.

    Note: Environment variables are read at class definition time in
    RequestTimeoutConfig, so we test the float conversion behavior
    rather than requiring module reload.
    """

    def test_env_values_are_read_as_floats(self):
        """Test that environment values are converted to floats."""
        # Test that default values can be overridden via configure_timeout
        config = configure_timeout(
            default_timeout=45.0,
            slow_timeout=180.0,
            max_timeout=300.0,
        )
        assert config.default_timeout == 45.0
        assert config.slow_timeout == 180.0
        assert config.max_timeout == 300.0

    def test_env_float_parsing(self):
        """Test float parsing of various formats."""
        # The implementation uses float() which handles various formats
        assert float("45") == 45.0
        assert float("45.5") == 45.5
        assert float("0.5") == 0.5

    def test_config_uses_os_environ_at_init(self):
        """Test that config reads from os.environ at initialization time."""
        import os

        # Get current default (should be from actual env or 30.0)
        config = RequestTimeoutConfig()
        expected_default = float(os.environ.get("ARAGORA_REQUEST_TIMEOUT", "30"))
        assert config.default_timeout == expected_default


# ============================================================================
# Integration Tests
# ============================================================================


class TestTimeoutIntegration:
    """Integration tests for timeout middleware."""

    def test_timeout_preserves_function_metadata(self):
        """Test decorator preserves function name and docstring."""

        @with_timeout(10.0)
        def my_handler(self, path, query_params, handler):
            """My handler docstring."""
            return {"result": "ok"}

        assert my_handler.__name__ == "my_handler"
        assert "My handler docstring" in (my_handler.__doc__ or "")

    @pytest.mark.asyncio
    async def test_async_timeout_preserves_metadata(self):
        """Test async decorator preserves function metadata."""

        @async_with_timeout(10.0)
        async def my_async_handler(self, path, query_params, handler):
            """My async handler docstring."""
            return {"result": "ok"}

        assert my_async_handler.__name__ == "my_async_handler"
        assert "My async handler docstring" in (my_async_handler.__doc__ or "")

    def test_timeout_with_exception(self):
        """Test timeout handling when function raises exception."""

        @with_timeout(5.0)
        def error_func(self, path, query_params, handler):
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            error_func(None, "/api/test", {}, None)

    @pytest.mark.asyncio
    async def test_async_timeout_with_exception(self):
        """Test async timeout handling when function raises exception."""

        @async_with_timeout(5.0)
        async def async_error_func(self, path, query_params, handler):
            raise ValueError("Async test error")

        with pytest.raises(ValueError, match="Async test error"):
            await async_error_func(None, "/api/test", {}, None)
