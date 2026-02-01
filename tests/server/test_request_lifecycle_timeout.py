"""
Tests for Request Lifecycle Timeout Integration.

Tests cover:
- Handler timeout enforcement (504 response for slow handlers)
- Normal handler completion (within timeout)
- Path-dependent timeout values
- Timeout response headers (Retry-After, X-Timeout-Seconds)
- Metrics and logging still execute on timeout
- Graceful degradation when timeout middleware unavailable
"""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, path: str = "/api/test"):
        self.path = path
        self.headers = {"Content-Type": "application/json"}
        self._rate_limit_result = None
        self._response_status = 200
        self._sent_json: dict[str, Any] | None = None
        self._sent_status: int | None = None
        self._sent_headers: dict[str, str] | None = None

    def _send_json(
        self,
        data: dict[str, Any],
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Mock send_json method."""
        self._sent_json = data
        self._sent_status = status
        self._sent_headers = headers or {}


class TestTimeoutEnforcement:
    """Tests for timeout enforcement on slow handlers."""

    def test_handler_timeout_returns_504(self):
        """Test that a handler exceeding timeout returns 504 response."""
        from aragora.server.request_lifecycle import RequestLifecycleManager

        handler = MockHandler("/api/slow")

        # Create a slow handler that sleeps longer than timeout
        def slow_handler(path: str) -> None:
            time.sleep(1.0)  # Sleep longer than timeout

        lifecycle = RequestLifecycleManager(handler=handler)

        # Mock timeout config to return short timeout
        mock_config = MagicMock()
        mock_config.get_timeout.return_value = 0.1  # 100ms timeout

        # Create a real executor for the test
        test_executor = ThreadPoolExecutor(max_workers=2)

        with patch(
            "aragora.server.request_lifecycle._timeout_config_factory",
            return_value=mock_config,
        ):
            with patch(
                "aragora.server.request_lifecycle._timeout_executor_factory",
                return_value=test_executor,
            ):
                lifecycle.handle_request(method="GET", internal_handler=slow_handler)

        # Verify 504 response
        assert handler._response_status == 504
        assert handler._sent_status == 504
        assert handler._sent_json is not None
        assert handler._sent_json["error"] == "Gateway Timeout"
        assert handler._sent_json["code"] == "request_timeout"
        assert "timeout_seconds" in handler._sent_json

        test_executor.shutdown(wait=False)

    def test_handler_completes_within_timeout(self):
        """Test that a fast handler returns normally."""
        from aragora.server.request_lifecycle import RequestLifecycleManager

        handler = MockHandler("/api/fast")

        # Create a fast handler
        def fast_handler(path: str) -> None:
            handler._send_json({"success": True})
            handler._response_status = 200

        lifecycle = RequestLifecycleManager(handler=handler)

        # Mock timeout config to return generous timeout
        mock_config = MagicMock()
        mock_config.get_timeout.return_value = 10.0  # 10s timeout

        # Create a real executor for the test
        test_executor = ThreadPoolExecutor(max_workers=2)

        with patch(
            "aragora.server.request_lifecycle._timeout_config_factory",
            return_value=mock_config,
        ):
            with patch(
                "aragora.server.request_lifecycle._timeout_executor_factory",
                return_value=test_executor,
            ):
                lifecycle.handle_request(method="GET", internal_handler=fast_handler)

        # Verify normal response
        assert handler._response_status == 200
        assert handler._sent_json == {"success": True}

        test_executor.shutdown(wait=False)


class TestPathDependentTimeouts:
    """Tests for path-based timeout configuration."""

    def test_slow_endpoint_gets_longer_timeout(self):
        """Test that slow endpoints get appropriate longer timeouts."""
        from aragora.server.request_lifecycle import RequestLifecycleManager

        # Test with a debates endpoint (should get slow_timeout)
        handler = MockHandler("/api/debates/create")
        received_timeout = None

        def capture_handler(path: str) -> None:
            pass

        lifecycle = RequestLifecycleManager(handler=handler)

        # Mock config that returns different timeouts for different paths
        mock_config = MagicMock()

        def get_timeout_for_path(path: str) -> float:
            nonlocal received_timeout
            if "debates" in path:
                received_timeout = 120.0
                return 120.0
            received_timeout = 30.0
            return 30.0

        mock_config.get_timeout.side_effect = get_timeout_for_path

        test_executor = ThreadPoolExecutor(max_workers=2)

        with patch(
            "aragora.server.request_lifecycle._timeout_config_factory",
            return_value=mock_config,
        ):
            with patch(
                "aragora.server.request_lifecycle._timeout_executor_factory",
                return_value=test_executor,
            ):
                lifecycle.handle_request(method="POST", internal_handler=capture_handler)

        # Verify the config was queried for the correct path
        mock_config.get_timeout.assert_called_once_with("/api/debates/create")
        assert received_timeout == 120.0

        test_executor.shutdown(wait=False)

    def test_fast_endpoint_gets_shorter_timeout(self):
        """Test that fast endpoints get default shorter timeouts."""
        from aragora.server.request_lifecycle import RequestLifecycleManager

        handler = MockHandler("/api/health")
        received_timeout = None

        def capture_handler(path: str) -> None:
            pass

        lifecycle = RequestLifecycleManager(handler=handler)

        mock_config = MagicMock()

        def get_timeout_for_path(path: str) -> float:
            nonlocal received_timeout
            if "health" in path:
                received_timeout = 5.0
                return 5.0
            received_timeout = 30.0
            return 30.0

        mock_config.get_timeout.side_effect = get_timeout_for_path

        test_executor = ThreadPoolExecutor(max_workers=2)

        with patch(
            "aragora.server.request_lifecycle._timeout_config_factory",
            return_value=mock_config,
        ):
            with patch(
                "aragora.server.request_lifecycle._timeout_executor_factory",
                return_value=test_executor,
            ):
                lifecycle.handle_request(method="GET", internal_handler=capture_handler)

        mock_config.get_timeout.assert_called_once_with("/api/health")
        assert received_timeout == 5.0

        test_executor.shutdown(wait=False)


class TestTimeoutResponseHeaders:
    """Tests for timeout response headers."""

    def test_timeout_includes_retry_after_header(self):
        """Test that 504 response includes Retry-After header."""
        from aragora.server.request_lifecycle import RequestLifecycleManager

        handler = MockHandler("/api/slow")

        def slow_handler(path: str) -> None:
            time.sleep(1.0)

        lifecycle = RequestLifecycleManager(handler=handler)

        mock_config = MagicMock()
        mock_config.get_timeout.return_value = 0.1

        test_executor = ThreadPoolExecutor(max_workers=2)

        with patch(
            "aragora.server.request_lifecycle._timeout_config_factory",
            return_value=mock_config,
        ):
            with patch(
                "aragora.server.request_lifecycle._timeout_executor_factory",
                return_value=test_executor,
            ):
                lifecycle.handle_request(method="GET", internal_handler=slow_handler)

        # Verify headers
        assert handler._sent_headers is not None
        assert "Retry-After" in handler._sent_headers
        # Retry-After should be positive integer
        assert int(handler._sent_headers["Retry-After"]) >= 0

        test_executor.shutdown(wait=False)

    def test_timeout_includes_x_timeout_seconds_header(self):
        """Test that 504 response includes X-Timeout-Seconds header."""
        from aragora.server.request_lifecycle import RequestLifecycleManager

        handler = MockHandler("/api/slow")

        def slow_handler(path: str) -> None:
            time.sleep(1.0)

        lifecycle = RequestLifecycleManager(handler=handler)

        mock_config = MagicMock()
        mock_config.get_timeout.return_value = 0.1

        test_executor = ThreadPoolExecutor(max_workers=2)

        with patch(
            "aragora.server.request_lifecycle._timeout_config_factory",
            return_value=mock_config,
        ):
            with patch(
                "aragora.server.request_lifecycle._timeout_executor_factory",
                return_value=test_executor,
            ):
                lifecycle.handle_request(method="GET", internal_handler=slow_handler)

        # Verify X-Timeout-Seconds header
        assert handler._sent_headers is not None
        assert "X-Timeout-Seconds" in handler._sent_headers
        assert handler._sent_headers["X-Timeout-Seconds"] == "0.1"

        test_executor.shutdown(wait=False)


class TestMetricsAndLoggingOnTimeout:
    """Tests for metrics and logging execution on timeout."""

    def test_metrics_recorded_on_timeout(self):
        """Test that metrics are still recorded when timeout occurs."""
        from aragora.server.request_lifecycle import RequestLifecycleManager

        handler = MockHandler("/api/slow")
        metrics_calls: list[tuple[str, str, int, float]] = []

        def record_metrics(method: str, endpoint: str, status: int, duration: float) -> None:
            metrics_calls.append((method, endpoint, status, duration))

        def slow_handler(path: str) -> None:
            time.sleep(1.0)

        lifecycle = RequestLifecycleManager(
            handler=handler,
            record_metrics_fn=record_metrics,
        )

        mock_config = MagicMock()
        mock_config.get_timeout.return_value = 0.1

        test_executor = ThreadPoolExecutor(max_workers=2)

        with patch(
            "aragora.server.request_lifecycle._timeout_config_factory",
            return_value=mock_config,
        ):
            with patch(
                "aragora.server.request_lifecycle._timeout_executor_factory",
                return_value=test_executor,
            ):
                lifecycle.handle_request(
                    method="GET",
                    internal_handler=slow_handler,
                    record_api_metrics_only=False,
                )

        # Verify metrics were recorded with 504 status
        assert len(metrics_calls) == 1
        method, endpoint, status, duration = metrics_calls[0]
        assert method == "GET"
        assert endpoint == "/api/slow"
        assert status == 504
        assert duration >= 0.1  # At least the timeout duration

        test_executor.shutdown(wait=False)

    def test_logging_executed_on_timeout(self):
        """Test that request logging still executes when timeout occurs."""
        from aragora.server.request_lifecycle import RequestLifecycleManager

        handler = MockHandler("/api/slow")
        log_calls: list[tuple[str, str, int, float]] = []

        def log_request(method: str, path: str, status: int, duration_ms: float) -> None:
            log_calls.append((method, path, status, duration_ms))

        def slow_handler(path: str) -> None:
            time.sleep(1.0)

        lifecycle = RequestLifecycleManager(
            handler=handler,
            log_request_fn=log_request,
        )

        mock_config = MagicMock()
        mock_config.get_timeout.return_value = 0.1

        test_executor = ThreadPoolExecutor(max_workers=2)

        with patch(
            "aragora.server.request_lifecycle._timeout_config_factory",
            return_value=mock_config,
        ):
            with patch(
                "aragora.server.request_lifecycle._timeout_executor_factory",
                return_value=test_executor,
            ):
                lifecycle.handle_request(method="GET", internal_handler=slow_handler)

        # Verify logging was called with 504 status
        assert len(log_calls) == 1
        method, path, status, duration_ms = log_calls[0]
        assert method == "GET"
        assert path == "/api/slow"
        assert status == 504
        assert duration_ms >= 100  # At least 100ms

        test_executor.shutdown(wait=False)


class TestGracefulDegradation:
    """Tests for graceful degradation when timeout middleware unavailable."""

    def test_no_timeout_config_proceeds_normally(self):
        """Test that requests proceed without timeout when config unavailable."""
        from aragora.server.request_lifecycle import RequestLifecycleManager

        handler = MockHandler("/api/test")
        handler_called = False

        def simple_handler(path: str) -> None:
            nonlocal handler_called
            handler_called = True
            handler._send_json({"ok": True})

        lifecycle = RequestLifecycleManager(handler=handler)

        # Simulate timeout config being unavailable
        with patch(
            "aragora.server.request_lifecycle._timeout_config_factory",
            None,
        ):
            lifecycle.handle_request(method="GET", internal_handler=simple_handler)

        # Handler should still be called and complete normally
        assert handler_called
        assert handler._sent_json == {"ok": True}

    def test_no_executor_proceeds_normally(self):
        """Test that requests proceed without timeout when executor unavailable."""
        from aragora.server.request_lifecycle import RequestLifecycleManager

        handler = MockHandler("/api/test")
        handler_called = False

        def simple_handler(path: str) -> None:
            nonlocal handler_called
            handler_called = True
            handler._send_json({"ok": True})

        lifecycle = RequestLifecycleManager(handler=handler)

        mock_config = MagicMock()
        mock_config.get_timeout.return_value = 30.0

        # Config available but executor returns None
        with patch(
            "aragora.server.request_lifecycle._timeout_config_factory",
            return_value=mock_config,
        ):
            with patch(
                "aragora.server.request_lifecycle._timeout_executor_factory",
                return_value=None,
            ):
                lifecycle.handle_request(method="GET", internal_handler=simple_handler)

        # Handler should still be called and complete normally
        assert handler_called
        assert handler._sent_json == {"ok": True}

    def test_config_exception_proceeds_normally(self):
        """Test that requests proceed when config raises exception."""
        from aragora.server.request_lifecycle import RequestLifecycleManager

        handler = MockHandler("/api/test")
        handler_called = False

        def simple_handler(path: str) -> None:
            nonlocal handler_called
            handler_called = True
            handler._send_json({"ok": True})

        lifecycle = RequestLifecycleManager(handler=handler)

        # Config factory raises exception
        def raise_error() -> None:
            raise RuntimeError("Config error")

        with patch(
            "aragora.server.request_lifecycle._timeout_config_factory",
            raise_error,
        ):
            lifecycle.handle_request(method="GET", internal_handler=simple_handler)

        # Handler should still be called
        assert handler_called
        assert handler._sent_json == {"ok": True}


class TestQueryParameterHandling:
    """Tests for timeout with query parameter handlers."""

    def test_timeout_with_query_parameters(self):
        """Test timeout works correctly with handlers that receive query params."""
        from aragora.server.request_lifecycle import RequestLifecycleManager

        handler = MockHandler("/api/search?q=test")
        received_query = None

        def search_handler(path: str, query: dict[str, list[str]]) -> None:
            nonlocal received_query
            received_query = query
            handler._send_json({"results": []})

        lifecycle = RequestLifecycleManager(handler=handler)

        mock_config = MagicMock()
        mock_config.get_timeout.return_value = 10.0

        test_executor = ThreadPoolExecutor(max_workers=2)

        with patch(
            "aragora.server.request_lifecycle._timeout_config_factory",
            return_value=mock_config,
        ):
            with patch(
                "aragora.server.request_lifecycle._timeout_executor_factory",
                return_value=test_executor,
            ):
                lifecycle.handle_request(
                    method="GET",
                    internal_handler=search_handler,
                    with_query=True,
                )

        # Handler should receive query parameters
        assert received_query is not None
        assert "q" in received_query
        assert received_query["q"] == ["test"]

        test_executor.shutdown(wait=False)


class TestTracingIntegration:
    """Tests for tracing span handling with timeouts."""

    def test_tracing_span_finished_on_timeout(self):
        """Test that tracing span is properly finished when timeout occurs."""
        from aragora.server.request_lifecycle import RequestLifecycleManager

        handler = MockHandler("/api/slow")
        mock_tracing = MagicMock()
        mock_span = MagicMock()
        mock_tracing.start_request_span.return_value = mock_span

        def slow_handler(path: str) -> None:
            time.sleep(1.0)

        lifecycle = RequestLifecycleManager(
            handler=handler,
            tracing=mock_tracing,
        )

        mock_config = MagicMock()
        mock_config.get_timeout.return_value = 0.1

        test_executor = ThreadPoolExecutor(max_workers=2)

        with patch(
            "aragora.server.request_lifecycle._timeout_config_factory",
            return_value=mock_config,
        ):
            with patch(
                "aragora.server.request_lifecycle._timeout_executor_factory",
                return_value=test_executor,
            ):
                lifecycle.handle_request(
                    method="GET",
                    internal_handler=slow_handler,
                    trace_api_only=False,
                )

        # Verify span was started and finished
        mock_tracing.start_request_span.assert_called_once()
        mock_tracing.finish_request_span.assert_called_once_with(mock_span, 504)

        test_executor.shutdown(wait=False)
