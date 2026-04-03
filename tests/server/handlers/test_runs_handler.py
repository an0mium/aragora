"""Tests for backbone run ledger handlers."""

from __future__ import annotations

from typing import Any

import pytest

from aragora.pipeline.backbone_contracts import BackboneStage, RunLedger, RunStageEvent
from aragora.pipeline.execution_mode import ExecutionMode
from aragora.pipeline.plan_store import PlanStore
from aragora.server.handlers.runs import RunsHandler, handle_run_detail, handle_runs_list


def _parse(result: Any) -> dict[str, Any]:
    """Normalize a HandlerResult into a simple dict."""
    if hasattr(result, "to_dict"):
        return result.to_dict()
    raise AssertionError("Expected HandlerResult-compatible response")


def _make_run(
    run_id: str,
    *,
    status: str,
    execution_id: str = "",
    receipt_id: str = "",
    safety_mode: str | None = None,
    stage_events: list[RunStageEvent] | None = None,
) -> RunLedger:
    """Build a test RunLedger."""
    run = RunLedger(
        run_id=run_id,
        entrypoint="prompt_engine.run",
        status=status,
        execution_id=execution_id,
        receipt_id=receipt_id,
        metadata={"safety_mode": safety_mode} if safety_mode else {},
    )
    for event in stage_events or []:
        run.add_event(event)
    return run


@pytest.fixture(autouse=True)
def isolated_plan_store(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> PlanStore:
    store = PlanStore(db_path=str(tmp_path / "runs_handler.db"))
    monkeypatch.setattr("aragora.pipeline.plan_store.get_plan_store", lambda: store)
    return store


def test_handle_runs_list_returns_compact_backbone_payload(
    isolated_plan_store: PlanStore,
) -> None:
    run = _make_run(
        "run-001",
        status="plan_ready",
        execution_id="exec-001",
        receipt_id="receipt-001",
        safety_mode=ExecutionMode.INTERACTIVE.value,
        stage_events=[
            RunStageEvent.create(BackboneStage.INTAKE, status="received"),
            RunStageEvent.create(BackboneStage.PLAN, status="completed"),
        ],
    )
    isolated_plan_store.create_run(run)

    result = handle_runs_list({"status": "plan_ready", "limit": "10", "offset": "0"})
    parsed = _parse(result)

    assert parsed["status"] == 200
    assert parsed["body"] == {
        "runs": [
            {
                "run_id": "run-001",
                "status": "plan_ready",
                "stages": [
                    {"stage": BackboneStage.INTAKE.value, "status": "received"},
                    {"stage": BackboneStage.PLAN.value, "status": "completed"},
                ],
                "execution_id": "exec-001",
                "receipt_id": "receipt-001",
                "safety_mode": ExecutionMode.INTERACTIVE.value,
            }
        ]
    }


def test_handle_runs_list_prefers_backbone_lister() -> None:
    run = _make_run(
        "run-compat",
        status="receipt_ready",
        stage_events=[RunStageEvent.create(BackboneStage.RECEIPT, status="completed")],
    )

    class _CompatStore:
        def __init__(self) -> None:
            self.calls: list[tuple[str | None, int, int]] = []

        def list_backbone_runs(
            self,
            *,
            status: str | None = None,
            limit: int = 50,
            offset: int = 0,
        ) -> list[RunLedger]:
            self.calls.append((status, limit, offset))
            return [run]

    store = _CompatStore()

    result = handle_runs_list({"status": "receipt_ready", "limit": "5", "offset": "2"}, store=store)
    parsed = _parse(result)

    assert parsed["status"] == 200
    assert store.calls == [("receipt_ready", 5, 2)]
    assert parsed["body"]["runs"][0]["run_id"] == "run-compat"
    assert parsed["body"]["runs"][0]["stages"] == [
        {"stage": BackboneStage.RECEIPT.value, "status": "completed"}
    ]


def test_handle_run_detail_prefers_get_backbone_run() -> None:
    run = _make_run(
        "run-detail",
        status="execution_started",
        execution_id="exec-detail",
        safety_mode=ExecutionMode.AUTONOMOUS.value,
        stage_events=[RunStageEvent.create(BackboneStage.EXECUTION, status="running")],
    )

    class _CompatStore:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def get_backbone_run(self, run_id: str) -> RunLedger | None:
            self.seen.append(run_id)
            return run if run_id == "run-detail" else None

    store = _CompatStore()
    result = handle_run_detail("run-detail", store=store)
    parsed = _parse(result)

    assert parsed["status"] == 200
    assert store.seen == ["run-detail"]
    assert parsed["body"] == {
        "run": {
            "run_id": "run-detail",
            "status": "execution_started",
            "stages": [
                {"stage": BackboneStage.EXECUTION.value, "status": "running"},
            ],
            "execution_id": "exec-detail",
            "receipt_id": None,
            "safety_mode": ExecutionMode.AUTONOMOUS.value,
        }
    }


def test_handle_run_detail_returns_404_when_missing() -> None:
    class _MissingStore:
        def get_backbone_run(self, run_id: str) -> None:
            return None

    result = handle_run_detail("missing-run", store=_MissingStore())
    parsed = _parse(result)

    assert parsed["status"] == 404
    assert parsed["body"] == {"error": "Run not found"}


def test_runs_handler_routes_list_requests(
    isolated_plan_store: PlanStore,
) -> None:
    run = _make_run(
        "run-handler-list",
        status="plan_ready",
        stage_events=[RunStageEvent.create(BackboneStage.PLAN, status="completed")],
    )
    isolated_plan_store.create_run(run)

    result = RunsHandler({"plan_store": isolated_plan_store}).handle("/api/runs", {}, None)
    parsed = _parse(result)

    assert parsed["status"] == 200
    assert parsed["body"]["runs"][0]["run_id"] == "run-handler-list"


def test_runs_handler_routes_detail_requests(
    isolated_plan_store: PlanStore,
) -> None:
    run = _make_run(
        "run-handler-detail",
        status="execution_started",
        stage_events=[RunStageEvent.create(BackboneStage.EXECUTION, status="running")],
    )
    isolated_plan_store.create_run(run)

    result = RunsHandler({"plan_store": isolated_plan_store}).handle(
        "/api/runs/run-handler-detail",
        {},
        None,
    )
    parsed = _parse(result)

    assert parsed["status"] == 200
    assert parsed["body"]["run"]["run_id"] == "run-handler-detail"
