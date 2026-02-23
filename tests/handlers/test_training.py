"""Tests for training data export handler (aragora/server/handlers/training.py).

Covers all routes and behavior of the TrainingHandler class:
- can_handle() routing for all static and dynamic routes
- GET    /api/v1/training/stats              - Training statistics
- GET    /api/v1/training/formats            - Export format schemas
- POST   /api/v1/training/export/sft         - SFT export
- POST   /api/v1/training/export/dpo         - DPO export
- POST   /api/v1/training/export/gauntlet    - Gauntlet export
- GET    /api/v1/training/jobs               - List training jobs
- GET    /api/v1/training/jobs/{id}          - Get job details
- DELETE /api/v1/training/jobs/{id}          - Cancel job
- POST   /api/v1/training/jobs/{id}/export   - Export job data
- POST   /api/v1/training/jobs/{id}/start    - Start training job
- POST   /api/v1/training/jobs/{id}/complete - Complete training job
- GET    /api/v1/training/jobs/{id}/metrics  - Get job metrics
- GET    /api/v1/training/jobs/{id}/artifacts - Get job artifacts
- Circuit breaker pattern
- Error handling, validation, edge cases
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.training import (
    TrainingCircuitBreaker,
    TrainingHandler,
    _clear_training_components,
    _get_training_circuit_breaker,
    get_training_circuit_breaker_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, method: str = "GET", body: dict | None = None):
        self.command = method
        if body:
            body_bytes = json.dumps(body).encode()
            self.headers = {"Content-Length": str(len(body_bytes))}
            self.rfile = BytesIO(body_bytes)
        else:
            self.headers = {"Content-Length": "0"}
            self.rfile = BytesIO(b"")


# ---------------------------------------------------------------------------
# Mock training components
# ---------------------------------------------------------------------------


class MockTrainingStatus(Enum):
    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MockVertical(Enum):
    HEALTHCARE = "healthcare"
    LEGAL = "legal"
    FINANCIAL = "financial"


@dataclass
class MockTrainingConfig:
    vertical: MockVertical = MockVertical.HEALTHCARE


@dataclass
class MockSpecialistModel:
    id: str = "model-001"
    vertical: MockVertical = MockVertical.HEALTHCARE
    status: MockTrainingStatus = MockTrainingStatus.PENDING
    base_model: str = "gpt-4"
    adapter_name: str = "healthcare-adapter"
    created_at: datetime = field(default_factory=datetime.now)
    training_data_examples: int = 100
    training_data_debates: int = 10
    final_loss: float = 0.05
    elo_rating: float = 1200.0
    win_rate: float = 0.65
    vertical_accuracy: float = 0.85
    checkpoint_path: str = "/tmp/checkpoints/model-001"
    training_config: MockTrainingConfig | None = None

    def __post_init__(self):
        if self.training_config is None:
            self.training_config = MockTrainingConfig(vertical=self.vertical)


class MockRegistry:
    """Mock SpecialistModelRegistry."""

    def __init__(self, models: list[MockSpecialistModel] | None = None):
        self._models: dict[str, MockSpecialistModel] = {}
        if models:
            for m in models:
                self._models[m.id] = m

    def get(self, model_id: str) -> MockSpecialistModel | None:
        return self._models.get(model_id)

    def update_status(self, model_id: str, status: Any) -> None:
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not found")
        self._models[model_id].status = status


class MockPipeline:
    """Mock SpecialistTrainingPipeline."""

    def __init__(self, registry: MockRegistry):
        self._registry = registry

    async def get_training_status(self, job_id: str) -> dict:
        model = self._registry.get(job_id)
        if model is None:
            raise ValueError(f"Job {job_id} not found")
        return {
            "job_id": model.id,
            "status": model.status.value,
            "vertical": model.vertical.value,
        }

    async def export_training_data(self, job_id: str) -> int:
        model = self._registry.get(job_id)
        if model is None:
            raise ValueError(f"Job {job_id} not found")
        return model.training_data_examples

    async def start_training(self, job_id: str) -> str:
        model = self._registry.get(job_id)
        if model is None:
            raise ValueError(f"Job {job_id} not found")
        model.status = MockTrainingStatus.TRAINING
        return f"training-{job_id}"

    async def complete_training(
        self, job_id: str, final_loss: float, checkpoint_path: str
    ) -> None:
        model = self._registry.get(job_id)
        if model is None:
            raise ValueError(f"Job {job_id} not found")
        model.status = MockTrainingStatus.COMPLETED
        model.final_loss = final_loss
        model.checkpoint_path = checkpoint_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler(tmp_path):
    """Create a TrainingHandler with mocked export dir."""
    with patch(
        "aragora.persistence.db_config.get_nomic_dir",
        return_value=tmp_path,
    ):
        h = TrainingHandler(ctx={})
    return h


@pytest.fixture(autouse=True)
def _clear_state():
    """Reset global training components and rate limiters between tests."""
    _clear_training_components()
    yield
    _clear_training_components()


@pytest.fixture(autouse=True)
def _disable_rate_limiting():
    """Disable rate limiting entirely during tests to avoid order-dependent failures."""
    import sys

    _rl_mod = sys.modules["aragora.server.handlers.utils.rate_limit"]
    original = _rl_mod.RATE_LIMITING_DISABLED
    _rl_mod.RATE_LIMITING_DISABLED = True
    yield
    _rl_mod.RATE_LIMITING_DISABLED = original


def _make_mock_exporter(records: list[dict] | None = None):
    """Create a mock exporter that returns given records."""
    mock = MagicMock()
    mock.export.return_value = records or []
    return mock


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify that can_handle correctly accepts or rejects paths."""

    def test_sft_export_path(self, handler):
        assert handler.can_handle("/api/v1/training/export/sft")

    def test_dpo_export_path(self, handler):
        assert handler.can_handle("/api/v1/training/export/dpo")

    def test_gauntlet_export_path(self, handler):
        assert handler.can_handle("/api/v1/training/export/gauntlet")

    def test_stats_path(self, handler):
        assert handler.can_handle("/api/v1/training/stats")

    def test_formats_path(self, handler):
        assert handler.can_handle("/api/v1/training/formats")

    def test_jobs_path(self, handler):
        assert handler.can_handle("/api/v1/training/jobs")

    def test_job_detail_path(self, handler):
        assert handler.can_handle("/api/v1/training/jobs/job-123")

    def test_job_export_path(self, handler):
        assert handler.can_handle("/api/v1/training/jobs/job-123/export")

    def test_job_start_path(self, handler):
        assert handler.can_handle("/api/v1/training/jobs/job-123/start")

    def test_job_complete_path(self, handler):
        assert handler.can_handle("/api/v1/training/jobs/job-123/complete")

    def test_job_metrics_path(self, handler):
        assert handler.can_handle("/api/v1/training/jobs/job-123/metrics")

    def test_job_artifacts_path(self, handler):
        assert handler.can_handle("/api/v1/training/jobs/job-123/artifacts")

    def test_rejects_unrelated_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_rejects_partial_prefix(self, handler):
        assert not handler.can_handle("/api/v1/training")

    def test_rejects_empty_path(self, handler):
        assert not handler.can_handle("")

    def test_rejects_root(self, handler):
        assert not handler.can_handle("/")

    def test_rejects_different_api_version(self, handler):
        assert not handler.can_handle("/api/v2/training/stats")


# ============================================================================
# Initialization
# ============================================================================


class TestHandlerInit:
    """Test handler initialization."""

    def test_init_with_empty_context(self, tmp_path):
        with patch(
            "aragora.persistence.db_config.get_nomic_dir",
            return_value=tmp_path,
        ):
            h = TrainingHandler(ctx={})
        assert h.ctx == {}

    def test_init_creates_export_dir(self, tmp_path):
        with patch(
            "aragora.persistence.db_config.get_nomic_dir",
            return_value=tmp_path,
        ):
            h = TrainingHandler(ctx={})
        assert h._export_dir.exists()

    def test_init_with_custom_export_dir(self, tmp_path):
        custom = tmp_path / "custom_exports"
        with patch.dict(
            "os.environ", {"ARAGORA_TRAINING_EXPORT_DIR": str(custom)}
        ):
            with patch(
                "aragora.persistence.db_config.get_nomic_dir",
                return_value=tmp_path,
            ):
                h = TrainingHandler(ctx={})
        assert h._export_dir == custom
        assert custom.exists()

    def test_routes_defined(self, handler):
        assert len(handler.ROUTES) == 6
        assert "/api/v1/training/export/sft" in handler.ROUTES
        assert "/api/v1/training/export/dpo" in handler.ROUTES
        assert "/api/v1/training/export/gauntlet" in handler.ROUTES

    def test_route_map_defined(self, handler):
        assert len(handler._ROUTE_MAP) == 6

    def test_job_routes_defined(self, handler):
        assert len(handler.JOB_ROUTES) >= 6


# ============================================================================
# handle() routing
# ============================================================================


class TestHandleRouting:
    """Test that handle() dispatches to the correct method."""

    def test_handle_returns_none_for_unknown_path(self, handler):
        result = handler.handle("/api/v1/unknown", {}, None)
        assert result is None

    def test_handle_routes_to_stats(self, handler):
        result = handler.handle("/api/v1/training/stats", {}, None)
        assert result is not None
        assert _status(result) == 200

    def test_handle_routes_to_formats(self, handler):
        result = handler.handle("/api/v1/training/formats", {}, None)
        assert result is not None
        assert _status(result) == 200

    def test_handle_routes_to_sft_export(self, handler):
        handler._exporters["sft"] = _make_mock_exporter([{"record": 1}])
        result = handler.handle(
            "/api/v1/training/export/sft", {}, MockHTTPHandler()
        )
        assert result is not None
        assert _status(result) == 200

    def test_handle_routes_to_dpo_export(self, handler):
        handler._exporters["dpo"] = _make_mock_exporter([{"pair": 1}])
        result = handler.handle(
            "/api/v1/training/export/dpo", {}, MockHTTPHandler()
        )
        assert result is not None
        assert _status(result) == 200

    def test_handle_routes_to_gauntlet_export(self, handler):
        handler._exporters["gauntlet"] = _make_mock_exporter([{"vuln": 1}])
        result = handler.handle(
            "/api/v1/training/export/gauntlet", {}, MockHTTPHandler()
        )
        assert result is not None
        assert _status(result) == 200

    def test_handle_routes_to_job_route(self, handler):
        # Without pipeline, should return 503
        result = handler.handle(
            "/api/v1/training/jobs/job-123",
            {},
            MockHTTPHandler(method="GET"),
        )
        assert result is not None
        assert _status(result) in (400, 503, 404)


# ============================================================================
# GET /api/v1/training/formats
# ============================================================================


class TestFormats:
    """Test supported training data formats endpoint."""

    def test_returns_all_formats(self, handler):
        result = handler.handle_formats(
            "/api/v1/training/formats", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert "formats" in body
        assert "sft" in body["formats"]
        assert "dpo" in body["formats"]
        assert "gauntlet" in body["formats"]

    def test_sft_format_has_schema(self, handler):
        result = handler.handle_formats(
            "/api/v1/training/formats", {}, MockHTTPHandler()
        )
        body = _body(result)
        sft = body["formats"]["sft"]
        assert "schema" in sft
        assert "description" in sft
        assert "use_case" in sft

    def test_dpo_format_has_schema(self, handler):
        result = handler.handle_formats(
            "/api/v1/training/formats", {}, MockHTTPHandler()
        )
        body = _body(result)
        dpo = body["formats"]["dpo"]
        assert "prompt" in dpo["schema"]
        assert "chosen" in dpo["schema"]
        assert "rejected" in dpo["schema"]

    def test_gauntlet_format_has_schema(self, handler):
        result = handler.handle_formats(
            "/api/v1/training/formats", {}, MockHTTPHandler()
        )
        body = _body(result)
        gauntlet = body["formats"]["gauntlet"]
        assert "instruction" in gauntlet["schema"]
        assert "response" in gauntlet["schema"]

    def test_output_formats_listed(self, handler):
        result = handler.handle_formats(
            "/api/v1/training/formats", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert "json" in body["output_formats"]
        assert "jsonl" in body["output_formats"]

    def test_endpoints_listed(self, handler):
        result = handler.handle_formats(
            "/api/v1/training/formats", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert body["endpoints"]["sft"] == "/api/v1/training/export/sft"
        assert body["endpoints"]["dpo"] == "/api/v1/training/export/dpo"
        assert body["endpoints"]["gauntlet"] == "/api/v1/training/export/gauntlet"


# ============================================================================
# GET /api/v1/training/stats
# ============================================================================


class TestStats:
    """Test training data statistics endpoint."""

    def test_stats_with_no_exporters(self, handler):
        result = handler.handle_stats(
            "/api/v1/training/stats", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert "available_exporters" in body
        assert "export_directory" in body
        assert "exported_files" in body

    def test_stats_with_sft_exporter(self, handler):
        mock_sft = _make_mock_exporter([{"record": 1}])
        handler._exporters["sft"] = mock_sft
        result = handler.handle_stats(
            "/api/v1/training/stats", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert "sft" in body["available_exporters"]
        assert body["sft_available"] is True

    def test_stats_sft_check_failure(self, handler):
        mock_sft = MagicMock()
        mock_sft.export.side_effect = RuntimeError("DB error")
        handler._exporters["sft"] = mock_sft
        result = handler.handle_stats(
            "/api/v1/training/stats", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert "sft" in body["available_exporters"]
        assert body["sft_available"] is False

    def test_stats_lists_exported_files(self, handler):
        # Create a fake .jsonl file in the export dir
        test_file = handler._export_dir / "test_export.jsonl"
        test_file.write_text('{"record": 1}\n')

        result = handler.handle_stats(
            "/api/v1/training/stats", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert len(body["exported_files"]) == 1
        assert body["exported_files"][0]["name"] == "test_export.jsonl"
        assert "size_bytes" in body["exported_files"][0]
        assert "created_at" in body["exported_files"][0]
        assert "modified_at" in body["exported_files"][0]

    def test_stats_no_export_dir(self, handler, tmp_path):
        handler._export_dir = tmp_path / "nonexistent"
        result = handler.handle_stats(
            "/api/v1/training/stats", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert body["exported_files"] == []

    def test_stats_with_multiple_exporters(self, handler):
        handler._exporters["sft"] = _make_mock_exporter([{"r": 1}])
        handler._exporters["dpo"] = _make_mock_exporter([{"r": 1}])
        handler._exporters["gauntlet"] = _make_mock_exporter([{"r": 1}])

        result = handler.handle_stats(
            "/api/v1/training/stats", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert "sft" in body["available_exporters"]
        assert "dpo" in body["available_exporters"]
        assert "gauntlet" in body["available_exporters"]


# ============================================================================
# POST /api/v1/training/export/sft
# ============================================================================


class TestExportSFT:
    """Test SFT export endpoint."""

    def test_sft_export_success_json(self, handler):
        records = [
            {"instruction": "Q1", "response": "A1"},
            {"instruction": "Q2", "response": "A2"},
        ]
        handler._exporters["sft"] = _make_mock_exporter(records)

        result = handler.handle_export_sft(
            "/api/v1/training/export/sft", {}, MockHTTPHandler()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["export_type"] == "sft"
        assert body["total_records"] == 2
        assert body["format"] == "json"
        assert body["records"] == records

    def test_sft_export_jsonl_format(self, handler):
        records = [{"instruction": "Q1", "response": "A1"}]
        handler._exporters["sft"] = _make_mock_exporter(records)

        result = handler.handle_export_sft(
            "/api/v1/training/export/sft",
            {"format": "jsonl"},
            MockHTTPHandler(),
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["format"] == "jsonl"
        assert "data" in body
        assert json.loads(body["data"]) == records[0]

    def test_sft_export_default_parameters(self, handler):
        mock = _make_mock_exporter([])
        handler._exporters["sft"] = mock

        handler.handle_export_sft(
            "/api/v1/training/export/sft", {}, MockHTTPHandler()
        )
        mock.export.assert_called_once_with(
            min_confidence=0.7,
            min_success_rate=0.6,
            limit=1000,
            offset=0,
            include_critiques=True,
            include_patterns=True,
            include_debates=True,
        )

    def test_sft_export_custom_parameters(self, handler):
        mock = _make_mock_exporter([])
        handler._exporters["sft"] = mock

        handler.handle_export_sft(
            "/api/v1/training/export/sft",
            {
                "min_confidence": "0.9",
                "min_success_rate": "0.8",
                "limit": "500",
                "offset": "10",
                "include_critiques": "false",
                "include_patterns": "false",
                "include_debates": "false",
            },
            MockHTTPHandler(),
        )
        mock.export.assert_called_once_with(
            min_confidence=0.9,
            min_success_rate=0.8,
            limit=500,
            offset=10,
            include_critiques=False,
            include_patterns=False,
            include_debates=False,
        )

    def test_sft_export_clamps_confidence(self, handler):
        mock = _make_mock_exporter([])
        handler._exporters["sft"] = mock

        handler.handle_export_sft(
            "/api/v1/training/export/sft",
            {"min_confidence": "2.0"},
            MockHTTPHandler(),
        )
        call_kwargs = mock.export.call_args.kwargs
        assert call_kwargs["min_confidence"] == 1.0

    def test_sft_export_clamps_negative_confidence(self, handler):
        mock = _make_mock_exporter([])
        handler._exporters["sft"] = mock

        handler.handle_export_sft(
            "/api/v1/training/export/sft",
            {"min_confidence": "-0.5"},
            MockHTTPHandler(),
        )
        call_kwargs = mock.export.call_args.kwargs
        assert call_kwargs["min_confidence"] == 0.0

    def test_sft_export_clamps_limit(self, handler):
        mock = _make_mock_exporter([])
        handler._exporters["sft"] = mock

        handler.handle_export_sft(
            "/api/v1/training/export/sft",
            {"limit": "50000"},
            MockHTTPHandler(),
        )
        call_kwargs = mock.export.call_args.kwargs
        assert call_kwargs["limit"] == 10000

    def test_sft_exporter_not_available(self, handler):
        # Ensure no sft exporter exists
        handler._exporters.pop("sft", None)
        # Patch _get_sft_exporter to return None (simulating ImportError)
        with patch.object(handler, "_get_sft_exporter", return_value=None):
            result = handler.handle_export_sft(
                "/api/v1/training/export/sft", {}, MockHTTPHandler()
            )
        assert _status(result) == 500
        body = _body(result)
        err = body.get("error", body.get("message", ""))
        err_str = err.get("message", "") if isinstance(err, dict) else str(err)
        assert "not available" in err_str

    def test_sft_export_value_error(self, handler):
        mock = MagicMock()
        mock.export.side_effect = ValueError("bad param")
        handler._exporters["sft"] = mock

        result = handler.handle_export_sft(
            "/api/v1/training/export/sft", {}, MockHTTPHandler()
        )
        assert _status(result) == 400

    def test_sft_export_runtime_error(self, handler):
        mock = MagicMock()
        mock.export.side_effect = RuntimeError("pipeline broken")
        handler._exporters["sft"] = mock

        result = handler.handle_export_sft(
            "/api/v1/training/export/sft", {}, MockHTTPHandler()
        )
        assert _status(result) == 500

    def test_sft_export_includes_timestamp(self, handler):
        handler._exporters["sft"] = _make_mock_exporter([])
        result = handler.handle_export_sft(
            "/api/v1/training/export/sft", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert "exported_at" in body

    def test_sft_export_includes_parameters(self, handler):
        handler._exporters["sft"] = _make_mock_exporter([])
        result = handler.handle_export_sft(
            "/api/v1/training/export/sft", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert "parameters" in body
        params = body["parameters"]
        assert "min_confidence" in params
        assert "min_success_rate" in params
        assert "limit" in params
        assert "offset" in params


# ============================================================================
# POST /api/v1/training/export/dpo
# ============================================================================


class TestExportDPO:
    """Test DPO export endpoint."""

    def test_dpo_export_success_json(self, handler):
        records = [{"prompt": "Q1", "chosen": "A1", "rejected": "A2"}]
        handler._exporters["dpo"] = _make_mock_exporter(records)

        result = handler.handle_export_dpo(
            "/api/v1/training/export/dpo", {}, MockHTTPHandler()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["export_type"] == "dpo"
        assert body["total_records"] == 1
        assert body["format"] == "json"
        assert body["records"] == records

    def test_dpo_export_jsonl_format(self, handler):
        records = [{"prompt": "Q1", "chosen": "A1", "rejected": "A2"}]
        handler._exporters["dpo"] = _make_mock_exporter(records)

        result = handler.handle_export_dpo(
            "/api/v1/training/export/dpo",
            {"format": "jsonl"},
            MockHTTPHandler(),
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["format"] == "jsonl"
        assert "data" in body

    def test_dpo_export_default_parameters(self, handler):
        mock = _make_mock_exporter([])
        handler._exporters["dpo"] = mock

        handler.handle_export_dpo(
            "/api/v1/training/export/dpo", {}, MockHTTPHandler()
        )
        mock.export.assert_called_once_with(
            min_confidence_diff=0.1,
            limit=500,
        )

    def test_dpo_export_custom_parameters(self, handler):
        mock = _make_mock_exporter([])
        handler._exporters["dpo"] = mock

        handler.handle_export_dpo(
            "/api/v1/training/export/dpo",
            {"min_confidence_diff": "0.3", "limit": "200"},
            MockHTTPHandler(),
        )
        mock.export.assert_called_once_with(
            min_confidence_diff=0.3,
            limit=200,
        )

    def test_dpo_export_clamps_confidence_diff(self, handler):
        mock = _make_mock_exporter([])
        handler._exporters["dpo"] = mock

        handler.handle_export_dpo(
            "/api/v1/training/export/dpo",
            {"min_confidence_diff": "5.0"},
            MockHTTPHandler(),
        )
        call_kwargs = mock.export.call_args.kwargs
        assert call_kwargs["min_confidence_diff"] == 1.0

    def test_dpo_export_clamps_limit(self, handler):
        mock = _make_mock_exporter([])
        handler._exporters["dpo"] = mock

        handler.handle_export_dpo(
            "/api/v1/training/export/dpo",
            {"limit": "99999"},
            MockHTTPHandler(),
        )
        call_kwargs = mock.export.call_args.kwargs
        assert call_kwargs["limit"] == 5000

    def test_dpo_exporter_not_available(self, handler):
        handler._exporters.pop("dpo", None)
        with patch.object(handler, "_get_dpo_exporter", return_value=None):
            result = handler.handle_export_dpo(
                "/api/v1/training/export/dpo", {}, MockHTTPHandler()
            )
        assert _status(result) == 500
        body = _body(result)
        err = body.get("error", body.get("message", ""))
        err_str = err.get("message", "") if isinstance(err, dict) else str(err)
        assert "not available" in err_str

    def test_dpo_export_value_error(self, handler):
        mock = MagicMock()
        mock.export.side_effect = ValueError("bad param")
        handler._exporters["dpo"] = mock

        result = handler.handle_export_dpo(
            "/api/v1/training/export/dpo", {}, MockHTTPHandler()
        )
        assert _status(result) == 400

    def test_dpo_export_runtime_error(self, handler):
        mock = MagicMock()
        mock.export.side_effect = RuntimeError("crash")
        handler._exporters["dpo"] = mock

        result = handler.handle_export_dpo(
            "/api/v1/training/export/dpo", {}, MockHTTPHandler()
        )
        assert _status(result) == 500

    def test_dpo_export_includes_timestamp(self, handler):
        handler._exporters["dpo"] = _make_mock_exporter([])
        result = handler.handle_export_dpo(
            "/api/v1/training/export/dpo", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert "exported_at" in body

    def test_dpo_export_includes_parameters(self, handler):
        handler._exporters["dpo"] = _make_mock_exporter([])
        result = handler.handle_export_dpo(
            "/api/v1/training/export/dpo", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert "parameters" in body
        params = body["parameters"]
        assert "min_confidence_diff" in params
        assert "limit" in params


# ============================================================================
# POST /api/v1/training/export/gauntlet
# ============================================================================


class TestExportGauntlet:
    """Test Gauntlet adversarial export endpoint."""

    def test_gauntlet_export_success_json(self, handler):
        records = [{"instruction": "Attack", "response": "Defense"}]
        handler._exporters["gauntlet"] = _make_mock_exporter(records)

        result = handler.handle_export_gauntlet(
            "/api/v1/training/export/gauntlet", {}, MockHTTPHandler()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["export_type"] == "gauntlet"
        assert body["total_records"] == 1
        assert body["format"] == "json"

    def test_gauntlet_export_jsonl_format(self, handler):
        records = [{"instruction": "Attack", "response": "Defense"}]
        handler._exporters["gauntlet"] = _make_mock_exporter(records)

        result = handler.handle_export_gauntlet(
            "/api/v1/training/export/gauntlet",
            {"format": "jsonl"},
            MockHTTPHandler(),
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["format"] == "jsonl"
        assert "data" in body

    def test_gauntlet_export_default_persona_all(self, handler):
        mock = _make_mock_exporter([])
        handler._exporters["gauntlet"] = mock

        handler.handle_export_gauntlet(
            "/api/v1/training/export/gauntlet", {}, MockHTTPHandler()
        )
        # When persona is "all", it should not pass persona kwarg
        call_kwargs = mock.export.call_args.kwargs
        assert "persona" not in call_kwargs
        assert call_kwargs["min_severity"] == 0.5
        assert call_kwargs["limit"] == 500

    def test_gauntlet_export_specific_persona(self, handler):
        mock = _make_mock_exporter([])
        handler._exporters["gauntlet"] = mock

        handler.handle_export_gauntlet(
            "/api/v1/training/export/gauntlet",
            {"persona": "gdpr"},
            MockHTTPHandler(),
        )
        call_kwargs = mock.export.call_args.kwargs
        assert call_kwargs["persona"] == "gdpr"

    def test_gauntlet_export_hipaa_persona(self, handler):
        mock = _make_mock_exporter([])
        handler._exporters["gauntlet"] = mock

        handler.handle_export_gauntlet(
            "/api/v1/training/export/gauntlet",
            {"persona": "hipaa"},
            MockHTTPHandler(),
        )
        call_kwargs = mock.export.call_args.kwargs
        assert call_kwargs["persona"] == "hipaa"

    def test_gauntlet_export_custom_severity(self, handler):
        mock = _make_mock_exporter([])
        handler._exporters["gauntlet"] = mock

        handler.handle_export_gauntlet(
            "/api/v1/training/export/gauntlet",
            {"min_severity": "0.8"},
            MockHTTPHandler(),
        )
        call_kwargs = mock.export.call_args.kwargs
        assert call_kwargs["min_severity"] == 0.8

    def test_gauntlet_export_clamps_severity(self, handler):
        mock = _make_mock_exporter([])
        handler._exporters["gauntlet"] = mock

        handler.handle_export_gauntlet(
            "/api/v1/training/export/gauntlet",
            {"min_severity": "3.0"},
            MockHTTPHandler(),
        )
        call_kwargs = mock.export.call_args.kwargs
        assert call_kwargs["min_severity"] == 1.0

    def test_gauntlet_exporter_not_available(self, handler):
        handler._exporters.pop("gauntlet", None)
        with patch.object(handler, "_get_gauntlet_exporter", return_value=None):
            result = handler.handle_export_gauntlet(
                "/api/v1/training/export/gauntlet", {}, MockHTTPHandler()
            )
        assert _status(result) == 500
        body = _body(result)
        err = body.get("error", body.get("message", ""))
        err_str = err.get("message", "") if isinstance(err, dict) else str(err)
        assert "not available" in err_str

    def test_gauntlet_export_value_error(self, handler):
        mock = MagicMock()
        mock.export.side_effect = TypeError("bad type")
        handler._exporters["gauntlet"] = mock

        result = handler.handle_export_gauntlet(
            "/api/v1/training/export/gauntlet", {}, MockHTTPHandler()
        )
        assert _status(result) == 400

    def test_gauntlet_export_runtime_error(self, handler):
        mock = MagicMock()
        mock.export.side_effect = AttributeError("attr missing")
        handler._exporters["gauntlet"] = mock

        result = handler.handle_export_gauntlet(
            "/api/v1/training/export/gauntlet", {}, MockHTTPHandler()
        )
        assert _status(result) == 500

    def test_gauntlet_export_includes_parameters(self, handler):
        handler._exporters["gauntlet"] = _make_mock_exporter([])
        result = handler.handle_export_gauntlet(
            "/api/v1/training/export/gauntlet", {}, MockHTTPHandler()
        )
        body = _body(result)
        params = body["parameters"]
        assert params["persona"] == "all"
        assert params["min_severity"] == 0.5
        assert params["limit"] == 500


# ============================================================================
# Job route validation and dispatching
# ============================================================================


class TestJobRouteValidation:
    """Test job route parsing, validation, and dispatching."""

    def test_invalid_short_path(self, handler):
        # Path too short: /api/v1/training
        result = handler._handle_job_route("/api/v1/training", {}, None)
        assert _status(result) == 400
        assert "Invalid" in _body(result)["error"]

    def test_invalid_job_id_special_chars(self, handler):
        # parts[4] = ".." which fails SAFE_ID_PATTERN validation
        result = handler._handle_job_route(
            "/api/training/jobs/../etc/passwd", {}, MockHTTPHandler()
        )
        assert _status(result) == 400

    def test_invalid_job_id_too_long(self, handler):
        long_id = "a" * 100
        result = handler._handle_job_route(
            f"/api/training/jobs/{long_id}", {}, MockHTTPHandler()
        )
        assert _status(result) == 400

    def test_valid_job_id_format(self, handler):
        # Valid ID but pipeline not available -> 503
        with patch.object(handler, "_get_training_pipeline", return_value=None):
            result = handler._handle_job_route(
                "/api/training/jobs/job-abc123",
                {},
                MockHTTPHandler(method="GET"),
            )
        # Should get 503 (pipeline not available) not 400 (bad ID)
        assert _status(result) == 503

    def test_unknown_job_sub_action(self, handler):
        # Unknown sub-action should return 404
        # Set up pipeline so we get past the pipeline check
        registry = MockRegistry([MockSpecialistModel(id="job-123")])
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        mock_handler = MockHTTPHandler(method="GET")
        result = handler._handle_job_route(
            "/api/training/jobs/job-123/unknown",
            {},
            mock_handler,
        )
        assert _status(result) == 404

    def test_wrong_method_for_export(self, handler):
        registry = MockRegistry([MockSpecialistModel(id="job-123")])
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        mock_handler = MockHTTPHandler(method="GET")
        result = handler._handle_job_route(
            "/api/training/jobs/job-123/export",
            {},
            mock_handler,
        )
        # GET on /export doesn't match POST expected
        assert _status(result) == 404

    def test_wrong_method_for_start(self, handler):
        registry = MockRegistry([MockSpecialistModel(id="job-123")])
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        mock_handler = MockHTTPHandler(method="GET")
        result = handler._handle_job_route(
            "/api/training/jobs/job-123/start",
            {},
            mock_handler,
        )
        assert _status(result) == 404


# ============================================================================
# GET /api/v1/training/jobs
# ============================================================================


class TestListJobs:
    """Test listing training jobs."""

    def test_pipeline_not_available(self, handler):
        with patch.object(handler, "_get_training_pipeline", return_value=None):
            result = handler.handle_list_jobs(
                "/api/v1/training/jobs", {}, MockHTTPHandler()
            )
        assert _status(result) == 503

    def test_list_empty_jobs(self, handler):
        registry = MockRegistry([])
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        result = handler.handle_list_jobs(
            "/api/v1/training/jobs", {}, MockHTTPHandler()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["jobs"] == []
        assert body["total"] == 0

    def test_list_jobs_with_models(self, handler):
        models = [
            MockSpecialistModel(id="m1", vertical=MockVertical.HEALTHCARE),
            MockSpecialistModel(id="m2", vertical=MockVertical.LEGAL),
        ]
        registry = MockRegistry(models)
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        result = handler.handle_list_jobs(
            "/api/v1/training/jobs", {}, MockHTTPHandler()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert len(body["jobs"]) == 2
        job_ids = {j["id"] for j in body["jobs"]}
        assert "m1" in job_ids
        assert "m2" in job_ids

    def test_list_jobs_filter_by_status(self, handler):
        models = [
            MockSpecialistModel(
                id="m1", status=MockTrainingStatus.PENDING
            ),
            MockSpecialistModel(
                id="m2", status=MockTrainingStatus.COMPLETED
            ),
        ]
        registry = MockRegistry(models)
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        result = handler.handle_list_jobs(
            "/api/v1/training/jobs",
            {"status": "completed"},
            MockHTTPHandler(),
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["jobs"][0]["status"] == "completed"

    def test_list_jobs_filter_by_vertical(self, handler):
        models = [
            MockSpecialistModel(
                id="m1", vertical=MockVertical.HEALTHCARE
            ),
            MockSpecialistModel(
                id="m2", vertical=MockVertical.LEGAL
            ),
        ]
        registry = MockRegistry(models)
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        result = handler.handle_list_jobs(
            "/api/v1/training/jobs",
            {"vertical": "legal"},
            MockHTTPHandler(),
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["jobs"][0]["vertical"] == "legal"

    def test_list_jobs_pagination(self, handler):
        models = [
            MockSpecialistModel(id=f"m{i}") for i in range(10)
        ]
        registry = MockRegistry(models)
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        result = handler.handle_list_jobs(
            "/api/v1/training/jobs",
            {"limit": "3", "offset": "2"},
            MockHTTPHandler(),
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 10
        assert len(body["jobs"]) == 3
        assert body["limit"] == 3
        assert body["offset"] == 2

    def test_list_jobs_attribute_error(self, handler):
        # Create a pipeline whose registry._models.values() raises AttributeError
        mock_pipeline = MagicMock()
        mock_pipeline._registry._models.values.side_effect = AttributeError(
            "no _models"
        )
        handler._exporters["pipeline"] = mock_pipeline

        result = handler.handle_list_jobs(
            "/api/v1/training/jobs", {}, MockHTTPHandler()
        )
        assert _status(result) == 500


# ============================================================================
# GET /api/v1/training/jobs/{id}
# ============================================================================


class TestGetJob:
    """Test getting a specific training job."""

    def test_pipeline_not_available(self, handler):
        with patch.object(handler, "_get_training_pipeline", return_value=None):
            result = handler._get_job("job-123", {}, MockHTTPHandler())
        assert _status(result) == 503

    def test_get_existing_job(self, handler):
        models = [MockSpecialistModel(id="job-123")]
        registry = MockRegistry(models)
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        result = handler._get_job("job-123", {}, MockHTTPHandler())
        assert _status(result) == 200
        body = _body(result)
        assert body["job_id"] == "job-123"

    def test_get_nonexistent_job(self, handler):
        registry = MockRegistry([])
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        result = handler._get_job("nonexistent", {}, MockHTTPHandler())
        assert _status(result) == 404

    def test_get_job_key_error(self, handler):
        mock_pipeline = MagicMock()
        mock_pipeline.get_training_status = MagicMock(
            side_effect=KeyError("missing key")
        )
        handler._exporters["pipeline"] = mock_pipeline

        with patch(
            "aragora.server.handlers.training.run_async",
            side_effect=KeyError("missing key"),
        ):
            result = handler._get_job("job-123", {}, MockHTTPHandler())
        assert _status(result) == 500

    def test_get_job_runtime_error(self, handler):
        handler._exporters["pipeline"] = MagicMock()
        with patch(
            "aragora.server.handlers.training.run_async",
            side_effect=RuntimeError("failed"),
        ):
            result = handler._get_job("job-123", {}, MockHTTPHandler())
        assert _status(result) == 500


# ============================================================================
# DELETE /api/v1/training/jobs/{id}
# ============================================================================


class TestCancelJob:
    """Test cancelling a training job."""

    def test_pipeline_not_available(self, handler):
        with patch.object(handler, "_get_training_pipeline", return_value=None):
            result = handler._cancel_job("job-123", {}, MockHTTPHandler())
        assert _status(result) == 503

    def test_cancel_existing_job(self, handler):
        models = [MockSpecialistModel(id="job-123")]
        registry = MockRegistry(models)
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        with patch(
            "aragora.training.specialist_models.TrainingStatus",
            MockTrainingStatus,
        ):
            result = handler._cancel_job("job-123", {}, MockHTTPHandler())
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["job_id"] == "job-123"
        assert body["status"] == "cancelled"

    def test_cancel_nonexistent_job(self, handler):
        registry = MockRegistry([])
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        with patch(
            "aragora.training.specialist_models.TrainingStatus",
            MockTrainingStatus,
        ):
            result = handler._cancel_job("nonexistent", {}, MockHTTPHandler())
        assert _status(result) == 404

    def test_cancel_runtime_error(self, handler):
        handler._exporters["pipeline"] = MagicMock()
        handler._exporters["pipeline"]._registry = MagicMock()
        handler._exporters["pipeline"]._registry.update_status.side_effect = (
            RuntimeError("fail")
        )

        with patch(
            "aragora.training.specialist_models.TrainingStatus",
            MockTrainingStatus,
        ):
            result = handler._cancel_job("job-123", {}, MockHTTPHandler())
        assert _status(result) == 500


# ============================================================================
# POST /api/v1/training/jobs/{id}/export
# ============================================================================


class TestExportJobData:
    """Test exporting training data for a specific job."""

    def test_pipeline_not_available(self, handler):
        with patch.object(handler, "_get_training_pipeline", return_value=None):
            result = handler._export_job_data("job-123", {}, MockHTTPHandler())
        assert _status(result) == 503

    def test_export_existing_job(self, handler):
        models = [MockSpecialistModel(id="job-123", training_data_examples=50)]
        registry = MockRegistry(models)
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        result = handler._export_job_data("job-123", {}, MockHTTPHandler())
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["job_id"] == "job-123"
        assert body["examples_exported"] == 50

    def test_export_nonexistent_job(self, handler):
        registry = MockRegistry([])
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        result = handler._export_job_data("nonexistent", {}, MockHTTPHandler())
        assert _status(result) == 404

    def test_export_os_error(self, handler):
        handler._exporters["pipeline"] = MagicMock()
        with patch(
            "aragora.server.handlers.training.run_async",
            side_effect=OSError("disk full"),
        ):
            result = handler._export_job_data("job-123", {}, MockHTTPHandler())
        assert _status(result) == 500


# ============================================================================
# POST /api/v1/training/jobs/{id}/start
# ============================================================================


class TestStartJob:
    """Test starting training for a job."""

    def test_pipeline_not_available(self, handler):
        with patch.object(handler, "_get_training_pipeline", return_value=None):
            result = handler._start_job("job-123", {}, MockHTTPHandler())
        assert _status(result) == 503

    def test_start_existing_job(self, handler):
        models = [MockSpecialistModel(id="job-123")]
        registry = MockRegistry(models)
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        result = handler._start_job("job-123", {}, MockHTTPHandler())
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["status"] == "training"
        assert body["training_job_id"] == "training-job-123"

    def test_start_nonexistent_job(self, handler):
        registry = MockRegistry([])
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        result = handler._start_job("nonexistent", {}, MockHTTPHandler())
        assert _status(result) == 400

    def test_start_runtime_error(self, handler):
        handler._exporters["pipeline"] = MagicMock()
        with patch(
            "aragora.server.handlers.training.run_async",
            side_effect=RuntimeError("fail"),
        ):
            result = handler._start_job("job-123", {}, MockHTTPHandler())
        assert _status(result) == 500


# ============================================================================
# POST /api/v1/training/jobs/{id}/complete
# ============================================================================


class TestCompleteJob:
    """Test completing a training job."""

    def test_pipeline_not_available(self, handler):
        with patch.object(handler, "_get_training_pipeline", return_value=None):
            result = handler._complete_job("job-123", {}, MockHTTPHandler())
        assert _status(result) == 503

    def test_complete_existing_job(self, handler):
        models = [MockSpecialistModel(id="job-123")]
        registry = MockRegistry(models)
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        mock_handler = MockHTTPHandler(
            method="POST",
            body={"final_loss": 0.02, "checkpoint_path": "/tmp/ckpt"},
        )
        result = handler._complete_job("job-123", {}, mock_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["status"] == "completed"
        assert body["final_loss"] == 0.02

    def test_complete_with_empty_body(self, handler):
        models = [MockSpecialistModel(id="job-123")]
        registry = MockRegistry(models)
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        mock_handler = MockHTTPHandler(method="POST")
        result = handler._complete_job("job-123", {}, mock_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["final_loss"] == 0.0

    def test_complete_nonexistent_job(self, handler):
        registry = MockRegistry([])
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        mock_handler = MockHTTPHandler(method="POST")
        result = handler._complete_job("nonexistent", {}, mock_handler)
        assert _status(result) == 404

    def test_complete_invalid_body_json(self, handler):
        models = [MockSpecialistModel(id="job-123")]
        registry = MockRegistry(models)
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        mock_handler = MockHTTPHandler(method="POST")
        mock_handler.headers = {"Content-Length": "10"}
        mock_handler.rfile = BytesIO(b"not json!!")

        result = handler._complete_job("job-123", {}, mock_handler)
        # Should default to 0.0 final_loss and proceed
        assert _status(result) == 200
        body = _body(result)
        assert body["final_loss"] == 0.0

    def test_complete_runtime_error(self, handler):
        handler._exporters["pipeline"] = MagicMock()
        with patch(
            "aragora.server.handlers.training.run_async",
            side_effect=RuntimeError("fail"),
        ):
            result = handler._complete_job(
                "job-123", {}, MockHTTPHandler(method="POST")
            )
        assert _status(result) == 500

    def test_complete_with_null_handler(self, handler):
        models = [MockSpecialistModel(id="job-123")]
        registry = MockRegistry(models)
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        result = handler._complete_job("job-123", {}, None)
        assert _status(result) == 200
        body = _body(result)
        assert body["final_loss"] == 0.0


# ============================================================================
# GET /api/v1/training/jobs/{id}/metrics
# ============================================================================


class TestGetJobMetrics:
    """Test getting training metrics for a job."""

    def test_pipeline_not_available(self, handler):
        with patch.object(handler, "_get_training_pipeline", return_value=None):
            result = handler._get_job_metrics("job-123", {}, MockHTTPHandler())
        assert _status(result) == 503

    def test_get_metrics_existing_job(self, handler):
        model = MockSpecialistModel(
            id="job-123",
            training_data_examples=100,
            training_data_debates=10,
            final_loss=0.05,
            elo_rating=1200.0,
            win_rate=0.65,
            vertical_accuracy=0.85,
        )
        registry = MockRegistry([model])
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        result = handler._get_job_metrics("job-123", {}, MockHTTPHandler())
        assert _status(result) == 200
        body = _body(result)
        assert body["job_id"] == "job-123"
        assert body["training_data_examples"] == 100
        assert body["training_data_debates"] == 10
        assert body["final_loss"] == 0.05
        assert body["elo_rating"] == 1200.0
        assert body["win_rate"] == 0.65
        assert body["vertical_accuracy"] == 0.85

    def test_get_metrics_nonexistent_job(self, handler):
        registry = MockRegistry([])
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        result = handler._get_job_metrics("nonexistent", {}, MockHTTPHandler())
        assert _status(result) == 404

    def test_get_metrics_attribute_error(self, handler):
        mock_pipeline = MagicMock()
        mock_pipeline._registry.get.side_effect = AttributeError("no attr")
        handler._exporters["pipeline"] = mock_pipeline

        result = handler._get_job_metrics("job-123", {}, MockHTTPHandler())
        assert _status(result) == 500


# ============================================================================
# GET /api/v1/training/jobs/{id}/artifacts
# ============================================================================


class TestGetJobArtifacts:
    """Test getting artifact information for a job."""

    def test_pipeline_not_available(self, handler):
        with patch.object(handler, "_get_training_pipeline", return_value=None):
            result = handler._get_job_artifacts("job-123", {}, MockHTTPHandler())
        assert _status(result) == 503

    def test_get_artifacts_existing_job(self, handler):
        model = MockSpecialistModel(
            id="job-123",
            checkpoint_path="/tmp/ckpt/model-001",
        )
        registry = MockRegistry([model])
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        result = handler._get_job_artifacts("job-123", {}, MockHTTPHandler())
        assert _status(result) == 200
        body = _body(result)
        assert body["job_id"] == "job-123"
        assert body["checkpoint_path"] == "/tmp/ckpt/model-001"

    def test_get_artifacts_nonexistent_job(self, handler):
        registry = MockRegistry([])
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        result = handler._get_job_artifacts("nonexistent", {}, MockHTTPHandler())
        assert _status(result) == 404

    def test_get_artifacts_with_data_directory(self, handler, tmp_path):
        model = MockSpecialistModel(
            id="job-123",
            vertical=MockVertical.HEALTHCARE,
            checkpoint_path="/tmp/ckpt",
        )
        registry = MockRegistry([model])
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        # Create a data directory
        data_dir = Path(f"data/training/healthcare/job-123")
        data_dir.mkdir(parents=True, exist_ok=True)
        try:
            (data_dir / "sft_data.jsonl").write_text('{"r": 1}\n')
            (data_dir / "dpo_data.jsonl").write_text('{"r": 2}\n')

            result = handler._get_job_artifacts(
                "job-123", {}, MockHTTPHandler()
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["data_directory"] is not None
            assert len(body["files"]) == 2
            types = {f["type"] for f in body["files"]}
            assert "sft" in types
            assert "dpo" in types
        finally:
            import shutil
            if data_dir.exists():
                shutil.rmtree("data/training", ignore_errors=True)

    def test_get_artifacts_no_data_directory(self, handler):
        model = MockSpecialistModel(id="job-123")
        registry = MockRegistry([model])
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        result = handler._get_job_artifacts("job-123", {}, MockHTTPHandler())
        assert _status(result) == 200
        body = _body(result)
        assert body["data_directory"] is None
        assert body["files"] == []

    def test_get_artifacts_os_error(self, handler):
        mock_pipeline = MagicMock()
        mock_model = MagicMock()
        mock_model.checkpoint_path = "/tmp/ckpt"
        mock_model.training_config.vertical.value = "healthcare"
        mock_pipeline._registry.get.return_value = mock_model

        handler._exporters["pipeline"] = mock_pipeline

        with patch("aragora.server.handlers.training.Path") as mock_path_cls:
            mock_path_cls.side_effect = OSError("disk error")
            result = handler._get_job_artifacts(
                "job-123", {}, MockHTTPHandler()
            )
        assert _status(result) == 500


# ============================================================================
# Exporter lazy loading
# ============================================================================


class TestExporterLoading:
    """Test exporter lazy loading with ImportError handling."""

    def test_sft_exporter_import_error(self, handler):
        with patch(
            "aragora.server.handlers.training.TrainingHandler._get_sft_exporter",
            return_value=None,
        ):
            result = handler.handle_export_sft(
                "/api/v1/training/export/sft", {}, MockHTTPHandler()
            )
        assert _status(result) == 500

    def test_dpo_exporter_import_error(self, handler):
        with patch(
            "aragora.server.handlers.training.TrainingHandler._get_dpo_exporter",
            return_value=None,
        ):
            result = handler.handle_export_dpo(
                "/api/v1/training/export/dpo", {}, MockHTTPHandler()
            )
        assert _status(result) == 500

    def test_gauntlet_exporter_import_error(self, handler):
        with patch(
            "aragora.server.handlers.training.TrainingHandler._get_gauntlet_exporter",
            return_value=None,
        ):
            result = handler.handle_export_gauntlet(
                "/api/v1/training/export/gauntlet", {}, MockHTTPHandler()
            )
        assert _status(result) == 500

    def test_sft_exporter_caches_instance(self, handler):
        mock = _make_mock_exporter([])
        handler._exporters["sft"] = mock
        # Second call should return same cached instance
        assert handler._get_sft_exporter() is mock

    def test_dpo_exporter_caches_instance(self, handler):
        mock = _make_mock_exporter([])
        handler._exporters["dpo"] = mock
        assert handler._get_dpo_exporter() is mock

    def test_gauntlet_exporter_caches_instance(self, handler):
        mock = _make_mock_exporter([])
        handler._exporters["gauntlet"] = mock
        assert handler._get_gauntlet_exporter() is mock


# ============================================================================
# Circuit breaker
# ============================================================================


class TestCircuitBreaker:
    """Test training circuit breaker integration."""

    def test_get_circuit_breaker_status(self):
        status = get_training_circuit_breaker_status()
        assert "state" in status
        assert "failure_count" in status
        assert "success_count" in status

    def test_circuit_breaker_initial_state_closed(self):
        status = get_training_circuit_breaker_status()
        assert status["state"] == "closed"

    def test_clear_training_components(self):
        # Get circuit breaker to create it
        _ = get_training_circuit_breaker_status()
        _clear_training_components()
        # Should recreate fresh one
        status = get_training_circuit_breaker_status()
        assert status["state"] == "closed"
        assert status["failure_count"] == 0

    def test_circuit_breaker_is_simple_circuit_breaker(self):
        cb = _get_training_circuit_breaker()
        assert isinstance(cb, TrainingCircuitBreaker)

    def test_circuit_breaker_blocks_when_open(self, handler):
        cb = _get_training_circuit_breaker()
        # Trip the circuit breaker
        for _ in range(cb.failure_threshold + 1):
            cb.record_failure()
        assert cb.state == "open"

        # _get_training_pipeline should return None when circuit is open
        pipeline = handler._get_training_pipeline()
        assert pipeline is None

    def test_check_pipeline_circuit_breaker_returns_503(self, handler):
        cb = _get_training_circuit_breaker()
        for _ in range(cb.failure_threshold + 1):
            cb.record_failure()

        result = handler._check_pipeline_circuit_breaker()
        assert result is not None
        assert _status(result) == 503
        body = _body(result)
        err = body.get("error", body.get("message", ""))
        err_str = err.get("message", "") if isinstance(err, dict) else str(err)
        assert "temporarily unavailable" in err_str

    def test_check_pipeline_circuit_breaker_allows_when_closed(self, handler):
        result = handler._check_pipeline_circuit_breaker()
        assert result is None

    def test_pipeline_import_error_records_failure(self, handler):
        cb = _get_training_circuit_breaker()
        initial_failures = cb.get_status()["failure_count"]

        # Force pipeline import to fail
        with patch(
            "builtins.__import__",
            side_effect=ImportError("no training module"),
        ):
            pipeline = handler._get_training_pipeline()

        assert pipeline is None
        assert cb.get_status()["failure_count"] > initial_failures


# ============================================================================
# Edge cases and error handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_sft_export_empty_records(self, handler):
        handler._exporters["sft"] = _make_mock_exporter([])
        result = handler.handle_export_sft(
            "/api/v1/training/export/sft", {}, MockHTTPHandler()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total_records"] == 0
        assert body["records"] == []

    def test_dpo_export_empty_records(self, handler):
        handler._exporters["dpo"] = _make_mock_exporter([])
        result = handler.handle_export_dpo(
            "/api/v1/training/export/dpo", {}, MockHTTPHandler()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total_records"] == 0

    def test_gauntlet_export_empty_records(self, handler):
        handler._exporters["gauntlet"] = _make_mock_exporter([])
        result = handler.handle_export_gauntlet(
            "/api/v1/training/export/gauntlet", {}, MockHTTPHandler()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total_records"] == 0

    def test_sft_export_many_records_jsonl(self, handler):
        records = [{"q": f"Q{i}", "a": f"A{i}"} for i in range(100)]
        handler._exporters["sft"] = _make_mock_exporter(records)

        result = handler.handle_export_sft(
            "/api/v1/training/export/sft",
            {"format": "jsonl"},
            MockHTTPHandler(),
        )
        assert _status(result) == 200
        body = _body(result)
        lines = body["data"].split("\n")
        assert len(lines) == 100

    def test_handler_with_no_command_attribute(self, handler):
        # Test handler without command defaults to "GET"
        mock_handler = MagicMock(spec=[])
        with patch.object(handler, "_get_training_pipeline", return_value=None):
            result = handler._handle_job_route(
                "/api/training/jobs/job-123",
                {},
                mock_handler,
            )
        # Should try GET path (pipeline not available -> 503)
        assert _status(result) == 503

    def test_handler_with_none(self, handler):
        # Test with None handler (defaults to "GET")
        with patch.object(handler, "_get_training_pipeline", return_value=None):
            result = handler._handle_job_route(
                "/api/training/jobs/job-123",
                {},
                None,
            )
        # None handler defaults to "GET", pipeline not available -> 503
        assert _status(result) == 503

    def test_job_route_delete_method(self, handler):
        models = [MockSpecialistModel(id="job-123")]
        registry = MockRegistry(models)
        pipeline = MockPipeline(registry)
        handler._exporters["pipeline"] = pipeline

        mock_handler = MockHTTPHandler(method="DELETE")

        with patch(
            "aragora.training.specialist_models.TrainingStatus",
            MockTrainingStatus,
        ):
            result = handler._handle_job_route(
                "/api/training/jobs/job-123",
                {},
                mock_handler,
            )
        # /api/training/jobs/job-123 -> parts[4] = "job-123", len == 5
        # method == DELETE -> _cancel_job
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["status"] == "cancelled"

    def test_stats_returns_200(self, handler):
        result = handler.handle_stats(
            "/api/v1/training/stats", {}, MockHTTPHandler()
        )
        assert _status(result) == 200

    def test_sft_export_attribute_error(self, handler):
        mock = MagicMock()
        mock.export.side_effect = AttributeError("missing attr")
        handler._exporters["sft"] = mock

        result = handler.handle_export_sft(
            "/api/v1/training/export/sft", {}, MockHTTPHandler()
        )
        assert _status(result) == 500

    def test_dpo_export_type_error(self, handler):
        mock = MagicMock()
        mock.export.side_effect = TypeError("wrong type")
        handler._exporters["dpo"] = mock

        result = handler.handle_export_dpo(
            "/api/v1/training/export/dpo", {}, MockHTTPHandler()
        )
        assert _status(result) == 400

    def test_gauntlet_export_clamps_negative_severity(self, handler):
        mock = _make_mock_exporter([])
        handler._exporters["gauntlet"] = mock

        handler.handle_export_gauntlet(
            "/api/v1/training/export/gauntlet",
            {"min_severity": "-1.0"},
            MockHTTPHandler(),
        )
        call_kwargs = mock.export.call_args.kwargs
        assert call_kwargs["min_severity"] == 0.0

    def test_sft_export_min_limit_clamped_to_1(self, handler):
        mock = _make_mock_exporter([])
        handler._exporters["sft"] = mock

        handler.handle_export_sft(
            "/api/v1/training/export/sft",
            {"limit": "0"},
            MockHTTPHandler(),
        )
        call_kwargs = mock.export.call_args.kwargs
        assert call_kwargs["limit"] >= 1

    def test_dpo_export_min_limit_clamped_to_1(self, handler):
        mock = _make_mock_exporter([])
        handler._exporters["dpo"] = mock

        handler.handle_export_dpo(
            "/api/v1/training/export/dpo",
            {"limit": "-5"},
            MockHTTPHandler(),
        )
        call_kwargs = mock.export.call_args.kwargs
        assert call_kwargs["limit"] >= 1
