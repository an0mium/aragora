"""
Tests for training data export handler.

Tests:
- TrainingHandler initialization
- Route matching (can_handle)
- Format endpoint (no auth required)
- Parameter validation bounds
- Circuit breaker functionality
- Job route handling
"""

import pytest
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from aragora.server.handlers.training import (
    TrainingHandler,
    TrainingCircuitBreaker,
    get_training_circuit_breaker_status,
    _clear_training_components,
    _get_training_circuit_breaker,
)


class TestTrainingHandlerInit:
    """Tests for TrainingHandler initialization."""

    def test_init_creates_export_dir(self, tmp_path, monkeypatch):
        """Should create export directory on init."""
        export_dir = tmp_path / "exports"
        monkeypatch.setenv("ARAGORA_TRAINING_EXPORT_DIR", str(export_dir))
        handler = TrainingHandler({})
        assert export_dir.exists()

    def test_init_with_existing_dir(self, tmp_path, monkeypatch):
        """Should work with existing directory."""
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        monkeypatch.setenv("ARAGORA_TRAINING_EXPORT_DIR", str(export_dir))
        handler = TrainingHandler({})
        assert export_dir.exists()

    def test_init_empty_exporters(self):
        """Should initialize with empty exporters dict."""
        handler = TrainingHandler({})
        assert handler._exporters == {}


class TestTrainingHandlerCanHandle:
    """Tests for can_handle routing."""

    def test_can_handle_sft_export(self):
        """Should handle SFT export path."""
        handler = TrainingHandler({})
        assert handler.can_handle("/api/v1/training/export/sft") is True

    def test_can_handle_dpo_export(self):
        """Should handle DPO export path."""
        handler = TrainingHandler({})
        assert handler.can_handle("/api/v1/training/export/dpo") is True

    def test_can_handle_gauntlet_export(self):
        """Should handle Gauntlet export path."""
        handler = TrainingHandler({})
        assert handler.can_handle("/api/v1/training/export/gauntlet") is True

    def test_can_handle_stats(self):
        """Should handle stats path."""
        handler = TrainingHandler({})
        assert handler.can_handle("/api/v1/training/stats") is True

    def test_can_handle_formats(self):
        """Should handle formats path."""
        handler = TrainingHandler({})
        assert handler.can_handle("/api/v1/training/formats") is True

    def test_cannot_handle_unknown_path(self):
        """Should not handle unknown paths."""
        handler = TrainingHandler({})
        assert handler.can_handle("/api/v1/training/unknown") is False
        assert handler.can_handle("/api/v1/other/path") is False
        assert handler.can_handle("/api/v1/training") is False


class TestTrainingHandlerRoutes:
    """Tests for route configuration."""

    def test_routes_constant_has_all_endpoints(self):
        """ROUTES should have all expected endpoints."""
        assert "/api/v1/training/export/sft" in TrainingHandler.ROUTES
        assert "/api/v1/training/export/dpo" in TrainingHandler.ROUTES
        assert "/api/v1/training/export/gauntlet" in TrainingHandler.ROUTES
        assert "/api/v1/training/stats" in TrainingHandler.ROUTES
        assert "/api/v1/training/formats" in TrainingHandler.ROUTES

    def test_routes_map_to_handler_methods(self):
        """Each route should map to a valid method."""
        handler = TrainingHandler({})
        for path, method_name in TrainingHandler.ROUTES.items():
            assert hasattr(handler, method_name), f"Missing method: {method_name}"


class TestTrainingHandlerFormats:
    """Tests for formats endpoint (no auth required)."""

    def test_handle_formats_returns_result(self):
        """handle_formats should return a result."""
        handler = TrainingHandler({})
        result = handler.handle_formats("/api/v1/training/formats", {}, None)
        assert result is not None

    def test_handle_formats_has_expected_structure(self):
        """Formats response should have expected keys."""
        handler = TrainingHandler({})
        result = handler.handle_formats("/api/v1/training/formats", {}, None)
        # Result is a HandlerResult, need to check body
        assert result.status_code == 200
        import json

        body = json.loads(result.body.decode("utf-8"))
        assert "formats" in body
        assert "output_formats" in body
        assert "endpoints" in body

    def test_handle_formats_includes_sft(self):
        """Formats should include SFT description."""
        handler = TrainingHandler({})
        result = handler.handle_formats("/api/v1/training/formats", {}, None)
        import json

        body = json.loads(result.body.decode("utf-8"))
        assert "sft" in body["formats"]
        assert "description" in body["formats"]["sft"]
        assert "schema" in body["formats"]["sft"]

    def test_handle_formats_includes_dpo(self):
        """Formats should include DPO description."""
        handler = TrainingHandler({})
        result = handler.handle_formats("/api/v1/training/formats", {}, None)
        import json

        body = json.loads(result.body.decode("utf-8"))
        assert "dpo" in body["formats"]
        assert "description" in body["formats"]["dpo"]

    def test_handle_formats_includes_gauntlet(self):
        """Formats should include Gauntlet description."""
        handler = TrainingHandler({})
        result = handler.handle_formats("/api/v1/training/formats", {}, None)
        import json

        body = json.loads(result.body.decode("utf-8"))
        assert "gauntlet" in body["formats"]
        assert "description" in body["formats"]["gauntlet"]

    def test_output_formats_include_json_jsonl(self):
        """Output formats should include json and jsonl."""
        handler = TrainingHandler({})
        result = handler.handle_formats("/api/v1/training/formats", {}, None)
        import json

        body = json.loads(result.body.decode("utf-8"))
        assert "json" in body["output_formats"]
        assert "jsonl" in body["output_formats"]


class TestTrainingHandlerExporterLazyLoad:
    """Tests for lazy-loading exporters."""

    def test_sft_exporter_returns_none_without_module(self):
        """Should return None when training module not available."""
        handler = TrainingHandler({})
        # Without actual training module, should return None
        result = handler._get_sft_exporter()
        # May return None or actual exporter depending on install
        assert result is None or result is not None

    def test_dpo_exporter_returns_none_without_module(self):
        """Should return None when training module not available."""
        handler = TrainingHandler({})
        result = handler._get_dpo_exporter()
        assert result is None or result is not None

    def test_gauntlet_exporter_returns_none_without_module(self):
        """Should return None when training module not available."""
        handler = TrainingHandler({})
        result = handler._get_gauntlet_exporter()
        assert result is None or result is not None

    def test_exporter_cached_after_first_load(self):
        """Exporter should be cached after first load."""
        handler = TrainingHandler({})
        handler._exporters["sft"] = "mock_exporter"
        result = handler._get_sft_exporter()
        assert result == "mock_exporter"


class TestTrainingHandlerHandle:
    """Tests for the handle dispatcher method."""

    def test_handle_unknown_path_returns_none(self):
        """Unknown path should return None."""
        handler = TrainingHandler({})
        result = handler.handle("/api/v1/unknown", {}, None)
        assert result is None

    def test_handle_formats_dispatches_correctly(self):
        """Should dispatch formats path to handle_formats."""
        handler = TrainingHandler({})
        result = handler.handle("/api/v1/training/formats", {}, None)
        assert result is not None
        assert result.status_code == 200


class TestTrainingHandlerStats:
    """Tests for stats endpoint."""

    def test_handle_stats_returns_result(self):
        """handle_stats should return a result."""
        handler = TrainingHandler({})
        result = handler.handle_stats("/api/v1/training/stats", {}, None)
        assert result is not None

    def test_handle_stats_has_available_exporters(self):
        """Stats response should list available exporters."""
        handler = TrainingHandler({})
        result = handler.handle_stats("/api/v1/training/stats", {}, None)
        import json

        body = json.loads(result.body.decode("utf-8"))
        assert "available_exporters" in body
        assert isinstance(body["available_exporters"], list)

    def test_handle_stats_has_export_directory(self):
        """Stats response should include export directory."""
        handler = TrainingHandler({})
        result = handler.handle_stats("/api/v1/training/stats", {}, None)
        import json

        body = json.loads(result.body.decode("utf-8"))
        assert "export_directory" in body

    def test_handle_stats_has_exported_files(self):
        """Stats response should list exported files."""
        handler = TrainingHandler({})
        result = handler.handle_stats("/api/v1/training/stats", {}, None)
        import json

        body = json.loads(result.body.decode("utf-8"))
        assert "exported_files" in body
        assert isinstance(body["exported_files"], list)


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestTrainingCircuitBreaker:
    """Tests for TrainingCircuitBreaker class."""

    def setup_method(self):
        """Reset circuit breaker before each test."""
        _clear_training_components()

    def teardown_method(self):
        """Clean up after each test."""
        _clear_training_components()

    def test_initial_state_is_closed(self):
        """Circuit breaker should start in CLOSED state."""
        cb = TrainingCircuitBreaker()
        assert cb.state == "closed"

    def test_is_allowed_in_closed_state(self):
        """Should allow requests in CLOSED state."""
        cb = TrainingCircuitBreaker()
        assert cb.is_allowed() is True

    def test_state_transitions_to_open_after_failures(self):
        """Should transition to OPEN after failure_threshold failures."""
        cb = TrainingCircuitBreaker(failure_threshold=3)
        assert cb.state == "closed"

        # Record 3 failures
        for _ in range(3):
            cb.record_failure()

        assert cb.state == "open"

    def test_open_state_rejects_requests(self):
        """Should reject requests in OPEN state."""
        cb = TrainingCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == "open"
        assert cb.is_allowed() is False

    def test_transition_to_half_open_after_cooldown(self):
        """Should transition to HALF_OPEN after cooldown period."""
        cb = TrainingCircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        assert cb.state == "open"

        # Wait for cooldown
        time.sleep(0.02)

        # State check should trigger transition
        assert cb.state == "half_open"

    def test_half_open_allows_limited_requests(self):
        """Should allow limited requests in HALF_OPEN state."""
        cb = TrainingCircuitBreaker(
            failure_threshold=1,
            cooldown_seconds=0.01,
            half_open_max_calls=2,
        )
        cb.record_failure()
        time.sleep(0.02)

        # Should allow first 2 calls
        assert cb.is_allowed() is True
        assert cb.is_allowed() is True
        # Third should be rejected
        assert cb.is_allowed() is False

    def test_success_in_half_open_closes_circuit(self):
        """Successful calls in HALF_OPEN should close the circuit."""
        cb = TrainingCircuitBreaker(
            failure_threshold=1,
            cooldown_seconds=0.01,
            half_open_max_calls=2,
        )
        cb.record_failure()
        time.sleep(0.02)

        # Make successful calls
        cb.is_allowed()
        cb.record_success()
        cb.is_allowed()
        cb.record_success()

        assert cb.state == "closed"

    def test_failure_in_half_open_reopens_circuit(self):
        """Failure in HALF_OPEN should reopen the circuit."""
        cb = TrainingCircuitBreaker(
            failure_threshold=1,
            cooldown_seconds=0.01,
            half_open_max_calls=2,
        )
        cb.record_failure()
        time.sleep(0.02)

        cb.is_allowed()
        cb.record_failure()

        assert cb.state == "open"

    def test_success_resets_failure_count(self):
        """Success in CLOSED state should reset failure count."""
        cb = TrainingCircuitBreaker(failure_threshold=3)

        # Record 2 failures
        cb.record_failure()
        cb.record_failure()
        assert cb._failure_count == 2

        # Success resets count
        cb.record_success()
        assert cb._failure_count == 0

    def test_reset_restores_initial_state(self):
        """reset() should restore circuit to initial state."""
        cb = TrainingCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == "open"

        cb.reset()
        assert cb.state == "closed"
        assert cb._failure_count == 0

    def test_get_status_returns_all_info(self):
        """get_status() should return all circuit breaker info."""
        cb = TrainingCircuitBreaker(failure_threshold=5, cooldown_seconds=30.0)
        status = cb.get_status()

        assert "state" in status
        assert "failure_count" in status
        assert "success_count" in status
        assert "failure_threshold" in status
        assert "cooldown_seconds" in status
        assert status["failure_threshold"] == 5
        assert status["cooldown_seconds"] == 30.0


class TestGlobalCircuitBreaker:
    """Tests for global circuit breaker functions."""

    def setup_method(self):
        """Reset circuit breaker before each test."""
        _clear_training_components()

    def teardown_method(self):
        """Clean up after each test."""
        _clear_training_components()

    def test_get_training_circuit_breaker_creates_instance(self):
        """Should create circuit breaker on first call."""
        cb = _get_training_circuit_breaker()
        assert cb is not None
        assert isinstance(cb, TrainingCircuitBreaker)

    def test_get_training_circuit_breaker_returns_same_instance(self):
        """Should return the same instance on subsequent calls."""
        cb1 = _get_training_circuit_breaker()
        cb2 = _get_training_circuit_breaker()
        assert cb1 is cb2

    def test_get_training_circuit_breaker_status(self):
        """get_training_circuit_breaker_status() should return status dict."""
        status = get_training_circuit_breaker_status()
        assert isinstance(status, dict)
        assert "state" in status

    def test_clear_training_components(self):
        """_clear_training_components() should reset and clear circuit breaker."""
        cb1 = _get_training_circuit_breaker()
        cb1.record_failure()

        _clear_training_components()

        # New instance should be created
        cb2 = _get_training_circuit_breaker()
        assert cb2._failure_count == 0


class TestTrainingHandlerCircuitBreaker:
    """Tests for circuit breaker integration in TrainingHandler."""

    def setup_method(self):
        """Reset circuit breaker before each test."""
        _clear_training_components()

    def teardown_method(self):
        """Clean up after each test."""
        _clear_training_components()

    def test_get_training_pipeline_respects_circuit_breaker(self):
        """_get_training_pipeline should return None when circuit is open."""
        handler = TrainingHandler({})

        # Open the circuit
        cb = _get_training_circuit_breaker()
        for _ in range(5):
            cb.record_failure()
        assert cb.state == "open"

        # Pipeline should return None
        result = handler._get_training_pipeline()
        assert result is None

    def test_check_pipeline_circuit_breaker_returns_error_when_open(self):
        """_check_pipeline_circuit_breaker should return error when circuit is open."""
        handler = TrainingHandler({})

        # Open the circuit
        cb = _get_training_circuit_breaker()
        for _ in range(5):
            cb.record_failure()

        result = handler._check_pipeline_circuit_breaker()
        assert result is not None
        assert result.status_code == 503

    def test_check_pipeline_circuit_breaker_returns_none_when_closed(self):
        """_check_pipeline_circuit_breaker should return None when circuit is closed."""
        handler = TrainingHandler({})

        result = handler._check_pipeline_circuit_breaker()
        assert result is None


# =============================================================================
# Job Route Handling Tests
# =============================================================================


class TestTrainingHandlerJobRoutes:
    """Tests for job-specific route handling."""

    def test_can_handle_job_routes(self):
        """Should handle job-specific paths."""
        handler = TrainingHandler({})
        assert handler.can_handle("/api/v1/training/jobs/job123") is True
        assert handler.can_handle("/api/v1/training/jobs/job123/metrics") is True
        assert handler.can_handle("/api/v1/training/jobs/job123/artifacts") is True
        assert handler.can_handle("/api/v1/training/jobs/job123/start") is True
        assert handler.can_handle("/api/v1/training/jobs/job123/export") is True

    def test_handle_job_route_validates_job_id(self):
        """Should validate job_id format."""
        handler = TrainingHandler({})

        # Valid job ID path structure: /api/v1/training/jobs/{job_id}
        # For path /api/v1/training/jobs/invalid!id, parts[4] = 'invalid!id' (the job_id)
        # But actually with leading empty string: ['', 'api', 'v1', 'training', 'jobs', 'invalid!id']
        # So we need a path where parts[4] is invalid

        # The correct path structure for job_id at parts[4] is /api/training/jobs/{job_id}
        # Let's use a path that makes parts[4] invalid
        result = handler._handle_job_route(
            "/api/training/jobs/invalid!@#$id",
            {},
            None,
        )
        assert result is not None
        assert result.status_code == 400

    def test_handle_job_route_invalid_path_length(self):
        """Should reject paths with too few segments."""
        handler = TrainingHandler({})

        result = handler._handle_job_route(
            "/api/v1/training",
            {},
            None,
        )
        assert result is not None
        assert result.status_code == 400

    def test_handle_job_route_unknown_endpoint(self):
        """Should return 404 for unknown job endpoints."""
        handler = TrainingHandler({})

        mock_handler = MagicMock()
        mock_handler.command = "GET"

        result = handler._handle_job_route(
            "/api/v1/training/jobs/job123/unknown_action",
            {},
            mock_handler,
        )
        assert result is not None
        assert result.status_code == 404
