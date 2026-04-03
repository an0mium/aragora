"""Tests for backbone run ledger handlers."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.pipeline.backbone_contracts import RunLedger, RunStageEvent
from aragora.server.handlers.runs import RunsHandler
from aragora.server.handlers.utils.responses import HandlerResult


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    return json.loads(result.body)


def _make_mock_handler() -> MagicMock:
    handler = MagicMock()
    handler.command = "GET"
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {}
    return handler


def _make_run(
    *,
    run_id: str,
    status: str = "received",
    execution_id: str = "",
    receipt_id: str = "",
    safety_mode: str = "",
    stage_details_mode: str = "",
) -> RunLedger:
    run = RunLedger(
        run_id=run_id,
        entrypoint="prompt_engine.run",
        status=status,
        execution_id=execution_id,
        receipt_id=receipt_id,
        metadata={"safety_mode": safety_mode} if safety_mode else {},
    )
    run.stage_events = [
        RunStageEvent.create("intake", status="completed"),
        RunStageEvent.create(
            "execution",
            status="queued",
            details={"safety_mode": stage_details_mode} if stage_details_mode else {},
        ),
    ]
    return run


@pytest.fixture
def handler() -> RunsHandler:
    return RunsHandler(ctx={})


class TestRunsHandlerBasics:
    def test_can_handle_list_routes(self, handler: RunsHandler) -> None:
        assert handler.can_handle("/api/runs") is True
        assert handler.can_handle("/api/v1/runs") is True

    def test_can_handle_detail_routes(self, handler: RunsHandler) -> None:
        assert handler.can_handle("/api/runs/run-123") is True
        assert handler.can_handle("/api/v1/runs/run-123") is True

    def test_rejects_unrelated_routes(self, handler: RunsHandler) -> None:
        assert handler.can_handle("/api/plans") is False


class TestListRuns:
    def test_list_runs_formats_run_ledger_fields(self, handler: RunsHandler) -> None:
        store = MagicMock()
        store.list_runs.return_value = [
            _make_run(
                run_id="run-1",
                status="execution_queued",
                execution_id="exec-1",
                receipt_id="receipt-1",
                safety_mode="interactive",
            ),
            _make_run(
                run_id="run-2",
                status="receipt_ready",
                execution_id="exec-2",
                receipt_id="receipt-2",
                stage_details_mode="autonomous",
            ),
        ]

        with patch("aragora.server.handlers.runs._get_plan_store", return_value=store):
            result = handler.handle(
                "/api/runs", {"limit": "2", "offset": "0"}, _make_mock_handler()
            )

        assert result is not None
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["total"] == 2
        assert data["limit"] == 2
        assert data["offset"] == 0
        assert data["runs"][0] == {
            "run_id": "run-1",
            "status": "execution_queued",
            "stages": [
                {"stage": "intake", "status": "completed"},
                {"stage": "execution", "status": "queued"},
            ],
            "execution_id": "exec-1",
            "receipt_id": "receipt-1",
            "safety_mode": "interactive",
        }
        assert data["runs"][1]["safety_mode"] == "autonomous"
        store.list_runs.assert_called_once_with(
            status=None,
            execution_id=None,
            limit=2,
            offset=0,
        )


class TestGetRun:
    def test_get_run_prefers_get_backbone_run(self, handler: RunsHandler) -> None:
        preferred = _make_run(
            run_id="run-preferred",
            status="execution_queued",
            execution_id="exec-1",
            receipt_id="receipt-1",
            safety_mode="interactive",
        )
        fallback = _make_run(run_id="run-fallback", safety_mode="autonomous")
        store = MagicMock()
        store.get_backbone_run.return_value = preferred
        store.get_run.return_value = fallback

        with patch("aragora.server.handlers.runs._get_plan_store", return_value=store):
            result = handler.handle(
                "/api/runs/run-preferred",
                {},
                _make_mock_handler(),
            )

        assert result is not None
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["run_id"] == "run-preferred"
        assert data["status"] == "execution_queued"
        assert data["execution_id"] == "exec-1"
        assert data["receipt_id"] == "receipt-1"
        assert data["safety_mode"] == "interactive"
        assert data["stages"] == [
            {"stage": "intake", "status": "completed"},
            {"stage": "execution", "status": "queued"},
        ]
        store.get_backbone_run.assert_called_once_with("run-preferred")
        store.get_run.assert_not_called()

    def test_get_run_returns_not_found(self, handler: RunsHandler) -> None:
        store = MagicMock()
        store.get_backbone_run.return_value = None

        with patch("aragora.server.handlers.runs._get_plan_store", return_value=store):
            result = handler.handle("/api/runs/missing-run", {}, _make_mock_handler())

        assert result is not None
        assert result.status_code == 404
        assert _parse_body(result) == {"error": "Run not found: missing-run"}
