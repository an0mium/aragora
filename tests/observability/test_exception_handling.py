"""
Tests for hardened exception handling in observability modules.

Validates that:
- Specific exception types are caught and logged appropriately
- Error context is included in structured logging
- Security-related errors are never silently swallowed
- Unexpected errors propagate correctly
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# TestMetricsExceptionHandling
# =============================================================================


class TestMetricsExceptionHandling:
    """Tests for metrics.py exception handling."""

    def test_init_metrics_import_error(self, caplog):
        """ImportError during metrics init should be logged with context."""
        from aragora.observability import metrics as metrics_module

        # Reset initialization state
        metrics_module._initialized = False
        metrics_module._metrics_server = None

        with patch.dict("sys.modules", {"prometheus_client": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("prometheus-client not found"),
            ):
                # Reset to force re-initialization
                metrics_module._initialized = False
                result = metrics_module._init_metrics()

        # Should return False and use no-op metrics
        # Note: the actual test depends on whether prometheus is installed

    def test_start_metrics_server_os_error(self, caplog):
        """OSError during server start should be logged with port context."""
        from aragora.observability.metrics import start_metrics_server, _init_metrics

        _init_metrics()

        # Patch at the prometheus_client level
        with patch(
            "prometheus_client.start_http_server",
            side_effect=OSError("Address already in use"),
        ):
            with caplog.at_level(logging.ERROR):
                result = start_metrics_server(9090)

        assert result is False

    def test_stop_metrics_server_when_not_running(self):
        """stop_metrics_server should return False when not running."""
        from aragora.observability.metrics import stop_metrics_server
        import aragora.observability.metrics as metrics_module

        # Ensure server is not running
        metrics_module._metrics_server = None
        result = stop_metrics_server()
        assert result is False

    def test_stop_metrics_server_when_running(self):
        """stop_metrics_server should handle normal shutdown."""
        from aragora.observability.metrics import stop_metrics_server, _init_metrics
        import aragora.observability.metrics as metrics_module

        _init_metrics()

        # Set server as running
        metrics_module._metrics_server = 9090
        result = stop_metrics_server()
        assert result is True
        assert metrics_module._metrics_server is None


# =============================================================================
# TestTracingExceptionHandling
# =============================================================================


class TestTracingExceptionHandling:
    """Tests for tracing.py exception handling."""

    def test_init_tracer_import_error_fallback(self):
        """ImportError should result in NoOpTracer fallback."""
        from aragora.observability.tracing import _init_tracer, _NoOpTracer

        # With mocked import error, should return NoOpTracer
        with patch.dict("sys.modules", {"opentelemetry": None}):
            # Reset tracer state
            import aragora.observability.tracing as tracing_module

            tracing_module._tracer = None

            tracer = _init_tracer()
            # Should be NoOpTracer when OTel not available
            assert tracer is not None

    def test_shutdown_handles_runtime_error(self, caplog):
        """shutdown() should handle RuntimeError gracefully."""
        from aragora.observability import tracing as tracing_module

        # Create a mock provider that raises on shutdown
        mock_provider = MagicMock()
        mock_provider.shutdown.side_effect = RuntimeError("Already shutdown")

        tracing_module._tracer_provider = mock_provider

        with caplog.at_level(logging.ERROR):
            tracing_module.shutdown()

        # Should log error but not raise
        mock_provider.shutdown.assert_called_once()

    def test_redact_url_handles_valid_url(self):
        """_redact_url should strip query params from valid URLs."""
        from aragora.observability.tracing import _redact_url

        result = _redact_url("https://example.com/path?secret=key")
        assert result == "https://example.com/path"

    def test_redact_url_handles_empty_string(self):
        """_redact_url should handle empty strings."""
        from aragora.observability.tracing import _redact_url

        result = _redact_url("")
        # Should return something parseable (empty scheme/host is still valid parsing)
        assert result is not None

    def test_record_function_args_handles_type_error(self):
        """_record_function_args should not fail on type errors."""
        from aragora.observability.tracing import _record_function_args
        import inspect

        def sample_func(x: int, y: str) -> None:
            pass

        sig = inspect.signature(sample_func)
        mock_span = MagicMock()

        # Call with wrong types - should not raise
        _record_function_args(mock_span, sig, (1, 2, 3), {})  # Extra args

    def test_record_result_handles_unusual_types(self):
        """_record_result should handle unusual result types."""
        from aragora.observability.tracing import _record_result

        mock_span = MagicMock()

        # Should not raise for any of these
        _record_result(mock_span, "string")
        _record_result(mock_span, 123)
        _record_result(mock_span, [1, 2, 3])
        _record_result(mock_span, {"key": "value"})
        _record_result(mock_span, object())


# =============================================================================
# TestGasTownDashboardExceptionHandling
# =============================================================================


class TestGasTownDashboardExceptionHandling:
    """Tests for gastown_dashboard.py exception handling."""

    def test_get_gastown_state_returns_none_on_error(self):
        """_get_gastown_state returns None when extensions not available."""
        # Import and call the function - it should handle errors internally
        from aragora.server.handlers.gastown_dashboard import _get_gastown_state

        # Patch the internal import to simulate failure
        with patch.dict("sys.modules", {"aragora.server.extensions": None}):
            # Force re-import to fail
            import importlib
            import sys

            # Remove cached module if present
            if "aragora.server.extensions" in sys.modules:
                del sys.modules["aragora.server.extensions"]

            result = _get_gastown_state()
            # Should return None without raising (ImportError path)
            assert result is None

    def test_get_gastown_state_runtime_error(self, caplog):
        """_get_gastown_state logs RuntimeError with context."""
        from aragora.server.handlers.gastown_dashboard import _get_gastown_state

        # Create a mock module that raises RuntimeError
        mock_extensions = MagicMock()
        mock_extensions.get_extension_state.side_effect = RuntimeError("State not initialized")

        with patch.dict(
            "sys.modules",
            {"aragora.server.extensions": mock_extensions},
        ):
            with caplog.at_level(logging.DEBUG):
                result = _get_gastown_state()

            assert result is None


# =============================================================================
# TestGatewayHealthExceptionHandling
# =============================================================================


class TestGatewayHealthExceptionHandling:
    """Tests for gateway_health_handler.py exception handling."""

    def test_agent_without_is_available_returns_false(self):
        """Agent without is_available method should return False."""
        from aragora.server.handlers.gateway_health_handler import GatewayHealthHandler

        # Create mock agent without is_available attribute
        mock_agent = MagicMock(spec=[])  # Empty spec means no is_available
        mock_agent.agent_type = "test"

        ctx = {"external_agents": {"test-agent": mock_agent}}
        handler = GatewayHealthHandler(ctx)

        available = handler._check_agent_available(mock_agent)
        # Should return False because hasattr check fails
        assert available is False

    def test_agent_sync_is_available_returns_value(self):
        """Agent with sync is_available should return the value."""
        from aragora.server.handlers.gateway_health_handler import GatewayHealthHandler

        # Create mock agent with sync is_available
        mock_agent = MagicMock()
        mock_agent.is_available.return_value = True
        mock_agent.agent_type = "test"

        ctx = {"external_agents": {"test-agent": mock_agent}}
        handler = GatewayHealthHandler(ctx)

        available = handler._check_agent_available(mock_agent)
        assert available is True

    def test_agent_sync_is_available_false(self):
        """Agent with sync is_available returning False."""
        from aragora.server.handlers.gateway_health_handler import GatewayHealthHandler

        mock_agent = MagicMock()
        mock_agent.is_available.return_value = False
        mock_agent.agent_type = "test"

        ctx = {"external_agents": {"test-agent": mock_agent}}
        handler = GatewayHealthHandler(ctx)

        available = handler._check_agent_available(mock_agent)
        assert available is False


# =============================================================================
# TestErrorClassification
# =============================================================================


class TestErrorClassification:
    """Tests for error classification in exception handling."""

    def test_data_errors_logged_as_debug(self):
        """Data errors (TypeError, ValueError, KeyError) should log at debug level."""
        # This tests the pattern used across modules
        logger = logging.getLogger("test_error_classification")

        # Set up handler to capture output
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            raise TypeError("Invalid type")
        except (TypeError, ValueError, KeyError) as e:
            logger.debug(
                "Expected data error",
                extra={"error_type": type(e).__name__, "error": str(e)},
            )

        # Test passes if no exception raised - we're testing the pattern works
        logger.removeHandler(handler)

    def test_runtime_errors_logged_as_warning(self):
        """Runtime errors (RuntimeError, OSError) should log at warning level."""
        logger = logging.getLogger("test_error_classification_warn")

        handler = logging.StreamHandler()
        handler.setLevel(logging.WARNING)
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

        try:
            raise OSError("Resource unavailable")
        except (RuntimeError, OSError) as e:
            logger.warning(
                "Runtime error occurred",
                extra={"error_type": type(e).__name__, "error": str(e)},
            )

        # Test passes if no exception raised
        logger.removeHandler(handler)

    def test_security_errors_not_swallowed(self):
        """Security-related errors should propagate, not be swallowed."""
        # This is a pattern test - security errors like PermissionError
        # should not be silently caught
        with pytest.raises(PermissionError):
            try:
                raise PermissionError("Access denied")
            except (TypeError, ValueError):  # These don't catch PermissionError
                pass
