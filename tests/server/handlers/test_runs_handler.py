"""
Tests for aragora.server.handlers.runs - Run Ledger Handlers.

Tests cover:
- RunsHandler: instantiation, can_handle routing
- GET /api/runs: list runs with pagination and filters
- GET /api/runs/{run_id}: get run by ID
- Formatting: _format_run_summary, _format_run_detail
- Error paths: run not found, missing run_id
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.runs import (
    RunsHandler,
    _format_run_summary,
    _format_run_detail,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


@dataclass
class FakeStageEvent:
    stage: str = "intake"
    status: str = "completed"
    created_at: str = "2026-04-01T00:00:00Z"

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "status": self.status, "created_at": self.created_at}


@dataclass
class FakeRunLedger:
    run_id: str = "run-001"
    entrypoint: str = "cli"
    status: str = "completed"
    execution_id: str = "exec-001"
    receipt_id: str = "rcpt-001"
    plan_id: str = "plan-001"
    debate_id: str = "debate-001"
    taint_flags: list[str] = field(default_factory=list)
    stage_events: list[FakeStageEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = "2026-04-01T00:00:00Z"
    updated_at: str = "2026-04-01T01:00:00Z"


def _make_run(**overrides: Any) -> FakeRunLedger:
    """Create a FakeRunLedger with optional overrides."""
    kwargs: dict[str, Any] = {}
    kwargs.update(overrides)
    return FakeRunLedger(**kwargs)


def _make_handler_ctx() -> dict[str, Any]:
    return {}


def _make_mock_http_handler() -> MagicMock:
    handler = MagicMock()
    handler.command = "GET"
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {"Content-Type": "application/json"}
    return handler


# ===========================================================================
# Format function tests
# ===========================================================================


class TestFormatRunSummary:
    def test_basic_fields(self):
        run = _make_run()
        result = _format_run_summary(run)
        assert result["run_id"] == "run-001"
        assert result["status"] == "completed"
        assert result["execution_id"] == "exec-001"
        assert result["receipt_id"] == "rcpt-001"
        assert result["created_at"] == "2026-04-01T00:00:00Z"
        assert result["updated_at"] == "2026-04-01T01:00:00Z"

    def test_safety_mode_from_metadata(self):
        run = _make_run(metadata={"safety_mode": "supervised"})
        result = _format_run_summary(run)
        assert result["safety_mode"] == "supervised"

    def test_safety_mode_defaults_empty(self):
        run = _make_run(metadata={})
        result = _format_run_summary(run)
        assert result["safety_mode"] == ""

    def test_stages_formatting(self):
        events = [
            FakeStageEvent(stage="intake", status="completed"),
            FakeStageEvent(stage="spec", status="running"),
        ]
        run = _make_run(stage_events=events)
        result = _format_run_summary(run)
        assert len(result["stages"]) == 2
        assert result["stages"][0]["stage"] == "intake"
        assert result["stages"][0]["status"] == "completed"
        assert result["stages"][1]["stage"] == "spec"
        assert result["stages"][1]["status"] == "running"

    def test_empty_stages(self):
        run = _make_run(stage_events=[])
        result = _format_run_summary(run)
        assert result["stages"] == []


class TestFormatRunDetail:
    def test_includes_summary_fields(self):
        run = _make_run()
        result = _format_run_detail(run)
        assert result["run_id"] == "run-001"
        assert result["status"] == "completed"

    def test_includes_extra_detail_fields(self):
        run = _make_run(
            entrypoint="api",
            plan_id="plan-002",
            debate_id="debate-002",
            taint_flags=["flag1"],
            metadata={"key": "value"},
        )
        result = _format_run_detail(run)
        assert result["entrypoint"] == "api"
        assert result["plan_id"] == "plan-002"
        assert result["debate_id"] == "debate-002"
        assert result["taint_flags"] == ["flag1"]
        assert result["metadata"] == {"key": "value"}


# ===========================================================================
# RunsHandler tests
# ===========================================================================


class TestRunsHandlerCanHandle:
    def test_handles_runs_path(self):
        h = RunsHandler(_make_handler_ctx())
        assert h.can_handle("/api/runs") is True

    def test_handles_run_id_path(self):
        h = RunsHandler(_make_handler_ctx())
        assert h.can_handle("/api/runs/run-001") is True

    def test_rejects_unrelated_path(self):
        h = RunsHandler(_make_handler_ctx())
        assert h.can_handle("/api/plans") is False

    def test_rejects_partial_prefix(self):
        h = RunsHandler(_make_handler_ctx())
        assert h.can_handle("/api/running") is False


class TestRunsHandlerListRuns:
    @patch("aragora.server.handlers.runs._get_plan_store")
    def test_list_returns_runs(self, mock_store_fn):
        store = MagicMock()
        store.list_runs.return_value = [_make_run(), _make_run(run_id="run-002")]
        mock_store_fn.return_value = store

        h = RunsHandler(_make_handler_ctx())
        result = h.handle("/api/runs", {}, _make_mock_http_handler())

        assert result is not None
        body = _parse_body(result)
        assert len(body["runs"]) == 2
        assert body["runs"][0]["run_id"] == "run-001"
        assert body["runs"][1]["run_id"] == "run-002"
        assert body["limit"] == 50
        assert body["offset"] == 0

    @patch("aragora.server.handlers.runs._get_plan_store")
    def test_list_passes_filters(self, mock_store_fn):
        store = MagicMock()
        store.list_runs.return_value = []
        mock_store_fn.return_value = store

        h = RunsHandler(_make_handler_ctx())
        h.handle(
            "/api/runs",
            {"status": "completed", "plan_id": "plan-1", "limit": "10", "offset": "5"},
            _make_mock_http_handler(),
        )

        store.list_runs.assert_called_once_with(
            status="completed",
            plan_id="plan-1",
            debate_id=None,
            execution_id=None,
            limit=10,
            offset=5,
        )

    @patch("aragora.server.handlers.runs._get_plan_store")
    def test_list_empty(self, mock_store_fn):
        store = MagicMock()
        store.list_runs.return_value = []
        mock_store_fn.return_value = store

        h = RunsHandler(_make_handler_ctx())
        result = h.handle("/api/runs", {}, _make_mock_http_handler())

        body = _parse_body(result)
        assert body["runs"] == []


class TestRunsHandlerGetRun:
    @patch("aragora.server.handlers.runs._get_plan_store")
    def test_get_existing_run(self, mock_store_fn):
        store = MagicMock()
        store.get_run.return_value = _make_run(metadata={"safety_mode": "autonomous"})
        mock_store_fn.return_value = store

        h = RunsHandler(_make_handler_ctx())
        result = h.handle("/api/runs/run-001", {}, _make_mock_http_handler())

        assert result is not None
        body = _parse_body(result)
        assert body["run_id"] == "run-001"
        assert body["safety_mode"] == "autonomous"
        assert body["entrypoint"] == "cli"
        assert body["plan_id"] == "plan-001"

    @patch("aragora.server.handlers.runs._get_plan_store")
    def test_get_missing_run_returns_404(self, mock_store_fn):
        store = MagicMock()
        store.get_run.return_value = None
        mock_store_fn.return_value = store

        h = RunsHandler(_make_handler_ctx())
        result = h.handle("/api/runs/nonexistent", {}, _make_mock_http_handler())

        assert result is not None
        assert result.status == 404

    @patch("aragora.server.handlers.runs._get_plan_store")
    def test_get_run_with_stages(self, mock_store_fn):
        events = [
            FakeStageEvent(stage="intake", status="completed"),
            FakeStageEvent(stage="deliberation", status="running"),
        ]
        store = MagicMock()
        store.get_run.return_value = _make_run(stage_events=events)
        mock_store_fn.return_value = store

        h = RunsHandler(_make_handler_ctx())
        result = h.handle("/api/runs/run-001", {}, _make_mock_http_handler())

        body = _parse_body(result)
        assert len(body["stages"]) == 2
        assert body["stages"][0]["stage"] == "intake"

    @patch("aragora.server.handlers.runs._get_plan_store")
    def test_get_run_taint_flags(self, mock_store_fn):
        store = MagicMock()
        store.get_run.return_value = _make_run(taint_flags=["budget_exceeded", "timeout"])
        mock_store_fn.return_value = store

        h = RunsHandler(_make_handler_ctx())
        result = h.handle("/api/runs/run-001", {}, _make_mock_http_handler())

        body = _parse_body(result)
        assert body["taint_flags"] == ["budget_exceeded", "timeout"]


class TestRunsHandlerEdgeCases:
    def test_unrelated_path_returns_none(self):
        h = RunsHandler(_make_handler_ctx())
        result = h.handle("/api/plans", {}, _make_mock_http_handler())
        assert result is None

    @patch("aragora.server.handlers.runs._get_plan_store")
    def test_nested_path_returns_none(self, mock_store_fn):
        h = RunsHandler(_make_handler_ctx())
        result = h.handle("/api/runs/run-001/extra/path", {}, _make_mock_http_handler())
        assert result is None
