"""Tests for handler instrumentation module."""

import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.observability.handler_instrumentation import (
    MetricsMiddleware,
    _categorize_endpoint,
    instrument_handler,
    record_agent_registration,
    record_control_plane_operation,
    record_deliberation_complete,
    record_deliberation_start,
    record_task_submission,
    track_request,
)


class TestEndpointCategorization:
    """Test endpoint categorization for metrics labels."""

    def test_control_plane_agents(self):
        """Test control plane agent endpoints."""
        assert _categorize_endpoint("/api/control-plane/agents") == "control_plane.agents"
        assert _categorize_endpoint("/api/control-plane/agents/123") == "control_plane.agents"

    def test_control_plane_tasks(self):
        """Test control plane task endpoints."""
        assert _categorize_endpoint("/api/control-plane/tasks") == "control_plane.tasks"
        assert (
            _categorize_endpoint("/api/control-plane/tasks/456/complete") == "control_plane.tasks"
        )

    def test_control_plane_deliberations(self):
        """Test control plane deliberation endpoints."""
        assert (
            _categorize_endpoint("/api/control-plane/deliberations")
            == "control_plane.deliberations"
        )
        assert (
            _categorize_endpoint("/api/control-plane/deliberations/789/status")
            == "control_plane.deliberations"
        )

    def test_debates_endpoints(self):
        """Test debate endpoints."""
        assert _categorize_endpoint("/api/debates") == "debates"
        assert _categorize_endpoint("/api/debates/123") == "debates"

    def test_unknown_endpoints(self):
        """Test fallback categorization for unknown endpoints."""
        result = _categorize_endpoint("/api/v2/new-feature")
        assert result == "api.v2"

    def test_root_endpoint(self):
        """Test root endpoint categorization."""
        result = _categorize_endpoint("/health")
        assert result == "other" or "health" in result


class TestTrackRequest:
    """Test track_request context manager."""

    def test_basic_tracking(self):
        """Test basic request tracking."""
        with track_request("GET", "/api/debates", include_tracing=False) as ctx:
            ctx["status"] = 200
        # Should not raise

    def test_error_tracking(self):
        """Test tracking with error."""
        with pytest.raises(ValueError):
            with track_request("POST", "/api/debates", include_tracing=False) as ctx:
                ctx["status"] = 500
                raise ValueError("Test error")

    def test_default_status(self):
        """Test default status is 200."""
        with track_request("GET", "/api/debates", include_tracing=False) as ctx:
            pass
        assert ctx["status"] == 200


class TestInstrumentHandler:
    """Test instrument_handler decorator."""

    def test_decorator_preserves_function(self):
        """Test decorator preserves function metadata."""

        @instrument_handler("test.handler", include_tracing=False)
        def my_handler(self, query_params, handler):
            return {"result": "success"}

        assert my_handler.__name__ == "my_handler"

    def test_decorator_calls_function(self):
        """Test decorator calls underlying function."""
        called = {"value": False}

        @instrument_handler("test.handler", include_tracing=False)
        def my_handler(self, query_params, handler):
            called["value"] = True
            return {"result": "success"}

        result = my_handler(None, {}, None)
        assert called["value"] is True
        assert result == {"result": "success"}

    def test_decorator_extracts_handler_info(self):
        """Test decorator extracts handler information."""
        handler = MagicMock()
        handler.path = "/api/test"
        handler.command = "POST"

        @instrument_handler("test.handler", include_tracing=False)
        def my_handler(self, query_params, req_handler):
            return {"result": "success"}

        result = my_handler(None, {}, handler)
        assert result == {"result": "success"}

    def test_decorator_handles_exception(self):
        """Test decorator handles exceptions properly."""

        @instrument_handler("test.handler", include_tracing=False)
        def failing_handler(self, query_params, handler):
            raise RuntimeError("Test failure")

        with pytest.raises(RuntimeError, match="Test failure"):
            failing_handler(None, {}, None)


class TestMetricsMiddleware:
    """Test MetricsMiddleware class."""

    def test_default_configuration(self):
        """Test default middleware configuration."""
        middleware = MetricsMiddleware()
        assert middleware.enabled is True
        assert middleware.include_tracing is True
        assert "/health" in middleware.exclude_paths

    def test_should_instrument_normal_path(self):
        """Test instrumentation for normal paths."""
        middleware = MetricsMiddleware()
        assert middleware.should_instrument("/api/debates") is True
        assert middleware.should_instrument("/api/control-plane/agents") is True

    def test_should_not_instrument_excluded_paths(self):
        """Test exclusion of health/metrics paths."""
        middleware = MetricsMiddleware()
        assert middleware.should_instrument("/health") is False
        assert middleware.should_instrument("/metrics") is False
        assert middleware.should_instrument("/favicon.ico") is False

    def test_disabled_middleware(self):
        """Test disabled middleware doesn't instrument."""
        middleware = MetricsMiddleware(enabled=False)
        assert middleware.should_instrument("/api/debates") is False

    def test_wrap_success(self):
        """Test wrapping successful handler."""
        middleware = MetricsMiddleware(include_tracing=False)

        def handler():
            return {"status": "ok"}

        result = middleware.wrap(handler, "/api/test", "GET")
        assert result == {"status": "ok"}

    def test_wrap_with_result_tuple(self):
        """Test wrapping handler returning tuple (body, status)."""
        middleware = MetricsMiddleware(include_tracing=False)

        def handler():
            return ({"data": "test"}, 201)

        result = middleware.wrap(handler, "/api/test", "POST")
        assert result == ({"data": "test"}, 201)

    def test_wrap_handles_exception(self):
        """Test wrapping handler that raises exception."""
        middleware = MetricsMiddleware(include_tracing=False)

        def failing_handler():
            raise ValueError("Handler error")

        with pytest.raises(ValueError, match="Handler error"):
            middleware.wrap(failing_handler, "/api/test", "POST")

    def test_wrap_skips_excluded_paths(self):
        """Test wrap bypasses instrumentation for excluded paths."""
        middleware = MetricsMiddleware(include_tracing=False)

        def handler():
            return "health check"

        result = middleware.wrap(handler, "/health", "GET")
        assert result == "health check"


class TestControlPlaneRecordingFunctions:
    """Test control plane specific recording functions."""

    @patch("aragora.observability.handler_instrumentation.record_control_plane_operation")
    def test_record_agent_registration_success(self, mock_record):
        """Test recording successful agent registration."""
        record_agent_registration("agent-001", success=True)
        # Function should not raise

    @patch("aragora.observability.handler_instrumentation.record_control_plane_operation")
    def test_record_agent_registration_failure(self, mock_record):
        """Test recording failed agent registration."""
        record_agent_registration("agent-001", success=False)
        # Function should not raise

    @patch("aragora.observability.handler_instrumentation.record_control_plane_operation")
    def test_record_task_submission(self, mock_record):
        """Test recording task submission."""
        record_task_submission("deliberation", success=True)
        # Function should not raise

    @patch("aragora.observability.handler_instrumentation.record_control_plane_operation")
    def test_record_deliberation_start(self, mock_record):
        """Test recording deliberation start."""
        record_deliberation_start("request-001", agent_count=3)
        # Function should not raise

    @patch("aragora.observability.handler_instrumentation.record_control_plane_operation")
    def test_record_deliberation_complete_success(self, mock_record):
        """Test recording successful deliberation completion."""
        record_deliberation_complete(
            request_id="request-001",
            success=True,
            consensus_reached=True,
            duration_seconds=45.5,
            sla_compliant=True,
        )
        # Function should not raise

    @patch("aragora.observability.handler_instrumentation.record_control_plane_operation")
    def test_record_deliberation_complete_no_consensus(self, mock_record):
        """Test recording deliberation with no consensus."""
        record_deliberation_complete(
            request_id="request-001",
            success=True,
            consensus_reached=False,
            duration_seconds=120.0,
            sla_compliant=False,
        )
        # Function should not raise


class TestRecordControlPlaneOperation:
    """Test the core record_control_plane_operation function."""

    def test_record_without_metrics(self):
        """Test recording when metrics module unavailable."""
        # Should not raise even without prometheus
        record_control_plane_operation("test_operation", "success")

    def test_record_with_latency(self):
        """Test recording with latency value."""
        record_control_plane_operation("test_operation", "success", latency=0.5)
        # Should not raise


class TestIntegrationWithTracing:
    """Integration tests with tracing (mocked)."""

    def test_track_request_with_tracing_disabled(self):
        """Test request tracking with tracing disabled."""
        with track_request("GET", "/api/test", include_tracing=False) as ctx:
            time.sleep(0.01)  # Small delay
            ctx["status"] = 200

    def test_instrument_handler_with_tracing_disabled(self):
        """Test handler instrumentation with tracing disabled."""

        @instrument_handler("test.handler", include_tracing=False)
        def my_handler(self, params, handler):
            return ({"ok": True}, 200)

        result = my_handler(None, {}, None)
        assert result == ({"ok": True}, 200)
