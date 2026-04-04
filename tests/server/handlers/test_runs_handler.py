"""Tests for backbone run ledger handlers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.pipeline.backbone_contracts import (
    BackboneStage,
    ReceiptEnvelope,
    RunLedger,
    RunStageEvent,
)
from aragora.pipeline.execution_mode import ExecutionMode
from aragora.pipeline.plan_store import PlanStore
from aragora.server.handlers.runs import RunsHandler, handle_run_detail, handle_runs_list
from aragora.utils.public_urls import public_receipt_url


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
    debate_id: str = "",
    plan_id: str = "",
    receipt_envelope: ReceiptEnvelope | None = None,
    stage_events: list[RunStageEvent] | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> RunLedger:
    """Build a test RunLedger."""
    run = RunLedger(
        run_id=run_id,
        entrypoint="prompt_engine.run",
        status=status,
        plan_id=plan_id,
        debate_id=debate_id,
        execution_id=execution_id,
        receipt_id=receipt_id,
        receipt_envelope=receipt_envelope,
        metadata={"safety_mode": safety_mode} if safety_mode else {},
    )
    for event in stage_events or []:
        run.add_event(event)
    if created_at is not None:
        run.created_at = created_at
    if updated_at is not None:
        run.updated_at = updated_at
    return run


def _make_http_handler() -> Any:
    handler = MagicMock()
    handler.command = "GET"
    handler.headers = {}
    handler.user_store = None
    return handler


@pytest.fixture(autouse=True)
def isolated_plan_store(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> PlanStore:
    store = PlanStore(db_path=str(tmp_path / "runs_handler.db"))
    monkeypatch.setattr("aragora.pipeline.plan_store.get_plan_store", lambda: store)
    return store


@pytest.fixture
def authorized_http_handler(monkeypatch: pytest.MonkeyPatch) -> Any:
    from aragora.server.handlers.utils import decorators as handler_decorators

    auth_ctx = MagicMock()
    auth_ctx.is_authenticated = True
    auth_ctx.user_id = "runs-reader"
    auth_ctx.role = "admin"
    auth_ctx.error_reason = None

    monkeypatch.setattr(
        "aragora.billing.jwt_auth.extract_user_from_request",
        lambda handler, user_store=None: auth_ctx,
    )
    monkeypatch.setattr(
        handler_decorators,
        "has_permission",
        lambda role, permission: permission == "orchestration:read",
    )
    return _make_http_handler()


def test_handle_runs_list_returns_compact_backbone_payload(
    isolated_plan_store: PlanStore,
) -> None:
    stage_events = [
        RunStageEvent(
            event_id="evt-intake",
            stage=BackboneStage.INTAKE.value,
            status="received",
            created_at="2026-04-04T00:00:00+00:00",
        ),
        RunStageEvent(
            event_id="evt-plan",
            stage=BackboneStage.PLAN.value,
            status="completed",
            created_at="2026-04-04T00:02:00+00:00",
        ),
    ]
    run = _make_run(
        "run-001",
        status="plan_ready",
        plan_id="plan-001",
        debate_id="debate-001",
        execution_id="exec-001",
        receipt_id="receipt-001",
        receipt_envelope=ReceiptEnvelope(
            receipt_id="receipt-001",
            artifact_hash="hash-001",
            verdict="pass",
            confidence=0.97,
        ),
        safety_mode=ExecutionMode.INTERACTIVE.value,
        stage_events=stage_events,
        created_at="2026-04-04T00:00:00+00:00",
        updated_at="2026-04-04T00:02:00+00:00",
    )
    isolated_plan_store.create_run(run)
    isolated_plan_store.create_execution_record(
        plan_id="plan-001",
        debate_id="debate-001",
        status="running",
        correlation_id="corr-001",
        execution_id="exec-001",
    )

    result = handle_runs_list({"status": "plan_ready", "limit": "10", "offset": "0"})
    parsed = _parse(result)

    assert parsed["status"] == 200
    assert parsed["body"] == {
        "runs": [
            {
                "run_id": "run-001",
                "entrypoint": "prompt_engine.run",
                "status": "plan_ready",
                "stages": [
                    {
                        "stage": BackboneStage.INTAKE.value,
                        "status": "received",
                        "created_at": "2026-04-04T00:00:00+00:00",
                    },
                    {
                        "stage": BackboneStage.PLAN.value,
                        "status": "completed",
                        "created_at": "2026-04-04T00:02:00+00:00",
                    },
                ],
                "execution_id": "exec-001",
                "correlation_id": "corr-001",
                "debate_id": "debate-001",
                "receipt_id": "receipt-001",
                "receipt_url": public_receipt_url("receipt-001"),
                "safety_mode": ExecutionMode.INTERACTIVE.value,
                "created_at": "2026-04-04T00:00:00+00:00",
                "updated_at": "2026-04-04T00:02:00+00:00",
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
        {
            "stage": BackboneStage.RECEIPT.value,
            "status": "completed",
            "created_at": run.stage_events[0].created_at,
        }
    ]


def test_handle_run_detail_prefers_get_backbone_run() -> None:
    stage_events = [
        RunStageEvent(
            event_id="evt-execution",
            stage=BackboneStage.EXECUTION.value,
            status="running",
            artifact_ref="artifact://execution",
            details={"attempt": 1},
            created_at="2026-04-04T00:03:00+00:00",
        )
    ]
    run = _make_run(
        "run-detail",
        status="execution_started",
        plan_id="plan-detail",
        debate_id="debate-detail",
        execution_id="exec-detail",
        receipt_id="receipt-detail",
        receipt_envelope=ReceiptEnvelope(
            receipt_id="receipt-detail",
            artifact_hash="hash-detail",
            verdict="pass",
            confidence=0.88,
        ),
        safety_mode=ExecutionMode.AUTONOMOUS.value,
        stage_events=stage_events,
        created_at="2026-04-04T00:00:00+00:00",
        updated_at="2026-04-04T00:03:00+00:00",
    )

    class _CompatStore:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def get_backbone_run(self, run_id: str) -> RunLedger | None:
            self.seen.append(run_id)
            return run if run_id == "run-detail" else None

        def get_execution_record(self, execution_id: str) -> dict[str, Any] | None:
            if execution_id != "exec-detail":
                return None
            return {
                "execution_id": execution_id,
                "correlation_id": "corr-detail",
            }

    store = _CompatStore()
    result = handle_run_detail("run-detail", store=store)
    parsed = _parse(result)

    assert parsed["status"] == 200
    assert store.seen == ["run-detail"]
    assert parsed["body"] == {
        "run": {
            "run_id": "run-detail",
            "entrypoint": "prompt_engine.run",
            "status": "execution_started",
            "stages": [
                {
                    "stage": BackboneStage.EXECUTION.value,
                    "status": "running",
                    "created_at": "2026-04-04T00:03:00+00:00",
                },
            ],
            "stage_events": [
                {
                    "event_id": "evt-execution",
                    "stage": BackboneStage.EXECUTION.value,
                    "status": "running",
                    "artifact_ref": "artifact://execution",
                    "details": {"attempt": 1},
                    "created_at": "2026-04-04T00:03:00+00:00",
                }
            ],
            "execution_id": "exec-detail",
            "correlation_id": "corr-detail",
            "debate_id": "debate-detail",
            "plan_id": "plan-detail",
            "receipt_id": "receipt-detail",
            "receipt_url": public_receipt_url("receipt-detail"),
            "receipt_envelope": {
                "receipt_id": "receipt-detail",
                "artifact_hash": "hash-detail",
                "verdict": "pass",
                "confidence": 0.88,
                "signature": "",
                "policy_gate_result": {},
                "provenance_chain": [],
                "taint_summary": {},
                "extras": {},
            },
            "safety_mode": ExecutionMode.AUTONOMOUS.value,
            "created_at": "2026-04-04T00:00:00+00:00",
            "updated_at": "2026-04-04T00:03:00+00:00",
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
    authorized_http_handler: Any,
) -> None:
    run = _make_run(
        "run-handler-list",
        status="plan_ready",
        stage_events=[RunStageEvent.create(BackboneStage.PLAN, status="completed")],
    )
    isolated_plan_store.create_run(run)

    result = RunsHandler({"plan_store": isolated_plan_store}).handle(
        "/api/runs",
        {},
        authorized_http_handler,
    )
    parsed = _parse(result)

    assert parsed["status"] == 200
    assert parsed["body"]["runs"][0]["run_id"] == "run-handler-list"


def test_runs_handler_routes_detail_requests(
    isolated_plan_store: PlanStore,
    authorized_http_handler: Any,
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
        authorized_http_handler,
    )
    parsed = _parse(result)

    assert parsed["status"] == 200
    assert parsed["body"]["run"]["run_id"] == "run-handler-detail"


def test_runs_handler_requires_auth(
    isolated_plan_store: PlanStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aragora.server.handlers.utils import decorators as handler_decorators

    unauthenticated = MagicMock()
    unauthenticated.is_authenticated = False
    unauthenticated.error_reason = "Authentication required"

    monkeypatch.setattr(handler_decorators, "_test_user_context_override", None)
    monkeypatch.setattr(
        "aragora.billing.jwt_auth.extract_user_from_request",
        lambda handler, user_store=None: unauthenticated,
    )

    result = RunsHandler({"plan_store": isolated_plan_store}).handle(
        "/api/runs",
        {},
        _make_http_handler(),
    )
    parsed = _parse(result)

    assert parsed["status"] == 401
    assert parsed["body"] == {"error": "Authentication required"}
