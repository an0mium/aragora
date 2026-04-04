"""FastAPI tests for the backbone runs routes."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from aragora.pipeline.backbone_contracts import (
    BackboneStage,
    ReceiptEnvelope,
    RunLedger,
    RunStageEvent,
)
from aragora.pipeline.execution_mode import ExecutionMode
from aragora.pipeline.plan_store import PlanStore
from aragora.server.fastapi import create_app
from aragora.server.fastapi.dependencies.auth import require_authenticated
from aragora.utils.public_urls import public_receipt_url


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


@pytest.fixture
def app(tmp_path: Any):
    app = create_app()
    app.state.context = {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
        "user_store": None,
        "rbac_checker": MagicMock(),
        "decision_service": MagicMock(),
        "plan_store": PlanStore(db_path=str(tmp_path / "runs_routes.db")),
    }
    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def _override_auth(client: TestClient, permissions: set[str]) -> None:
    from aragora.rbac.models import AuthorizationContext

    auth_ctx = AuthorizationContext(
        user_id="user-1",
        org_id="org-1",
        workspace_id="ws-1",
        roles={"admin"},
        permissions=permissions,
    )
    client.app.dependency_overrides[require_authenticated] = lambda: auth_ctx


def test_list_runs_route_requires_auth(client) -> None:
    response = client.get("/api/v2/runs")

    assert response.status_code == 401


def test_list_runs_route_is_registered(client) -> None:
    plan_store = client.app.state.context["plan_store"]
    stage_event = RunStageEvent(
        event_id="evt-plan",
        stage=BackboneStage.PLAN.value,
        status="completed",
        created_at="2026-04-04T00:05:00+00:00",
    )
    plan_store.create_run(
        _make_run(
            "run-fastapi-list",
            status="plan_ready",
            plan_id="plan-fastapi",
            debate_id="debate-fastapi",
            execution_id="exec-fastapi",
            receipt_id="receipt-fastapi",
            receipt_envelope=ReceiptEnvelope(
                receipt_id="receipt-fastapi",
                artifact_hash="hash-fastapi",
                verdict="pass",
                confidence=0.91,
            ),
            safety_mode=ExecutionMode.INTERACTIVE.value,
            stage_events=[stage_event],
            created_at="2026-04-04T00:00:00+00:00",
            updated_at="2026-04-04T00:05:00+00:00",
        )
    )
    plan_store.create_execution_record(
        plan_id="plan-fastapi",
        debate_id="debate-fastapi",
        status="running",
        correlation_id="corr-fastapi",
        execution_id="exec-fastapi",
    )

    _override_auth(client, {"orchestration:read"})
    response = client.get("/api/v2/runs")
    client.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "runs": [
            {
                "run_id": "run-fastapi-list",
                "entrypoint": "prompt_engine.run",
                "status": "plan_ready",
                "stages": [
                    {
                        "created_at": "2026-04-04T00:05:00+00:00",
                        "stage": BackboneStage.PLAN.value,
                        "status": "completed",
                    }
                ],
                "execution_id": "exec-fastapi",
                "correlation_id": "corr-fastapi",
                "debate_id": "debate-fastapi",
                "receipt_id": "receipt-fastapi",
                "receipt_url": public_receipt_url("receipt-fastapi"),
                "safety_mode": ExecutionMode.INTERACTIVE.value,
                "created_at": "2026-04-04T00:00:00+00:00",
                "updated_at": "2026-04-04T00:05:00+00:00",
            }
        ]
    }


def test_get_run_route_requires_auth(client) -> None:
    response = client.get("/api/v2/runs/run-fastapi-detail")

    assert response.status_code == 401


def test_get_run_route_is_registered(client) -> None:
    plan_store = client.app.state.context["plan_store"]
    stage_event = RunStageEvent(
        event_id="evt-execution",
        stage=BackboneStage.EXECUTION.value,
        status="running",
        artifact_ref="artifact://exec-fastapi",
        details={"attempt": 2},
        created_at="2026-04-04T00:07:00+00:00",
    )
    plan_store.create_run(
        _make_run(
            "run-fastapi-detail",
            status="execution_started",
            plan_id="plan-fastapi-detail",
            debate_id="debate-fastapi-detail",
            execution_id="exec-fastapi-detail",
            receipt_id="receipt-fastapi-detail",
            receipt_envelope=ReceiptEnvelope(
                receipt_id="receipt-fastapi-detail",
                artifact_hash="hash-fastapi-detail",
                verdict="pass",
                confidence=0.84,
            ),
            stage_events=[stage_event],
            created_at="2026-04-04T00:00:00+00:00",
            updated_at="2026-04-04T00:07:00+00:00",
        )
    )
    plan_store.create_execution_record(
        plan_id="plan-fastapi-detail",
        debate_id="debate-fastapi-detail",
        status="running",
        correlation_id="corr-fastapi-detail",
        execution_id="exec-fastapi-detail",
    )

    _override_auth(client, {"orchestration:read"})
    response = client.get("/api/v2/runs/run-fastapi-detail")
    client.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "run": {
            "run_id": "run-fastapi-detail",
            "entrypoint": "prompt_engine.run",
            "status": "execution_started",
            "stages": [
                {
                    "created_at": "2026-04-04T00:07:00+00:00",
                    "stage": BackboneStage.EXECUTION.value,
                    "status": "running",
                }
            ],
            "stage_events": [
                {
                    "event_id": "evt-execution",
                    "stage": BackboneStage.EXECUTION.value,
                    "status": "running",
                    "artifact_ref": "artifact://exec-fastapi",
                    "details": {"attempt": 2},
                    "created_at": "2026-04-04T00:07:00+00:00",
                }
            ],
            "execution_id": "exec-fastapi-detail",
            "correlation_id": "corr-fastapi-detail",
            "debate_id": "debate-fastapi-detail",
            "plan_id": "plan-fastapi-detail",
            "receipt_id": "receipt-fastapi-detail",
            "receipt_url": public_receipt_url("receipt-fastapi-detail"),
            "receipt_envelope": {
                "receipt_id": "receipt-fastapi-detail",
                "artifact_hash": "hash-fastapi-detail",
                "verdict": "pass",
                "confidence": 0.84,
                "signature": "",
                "policy_gate_result": {},
                "provenance_chain": [],
                "taint_summary": {},
                "extras": {},
            },
            "safety_mode": None,
            "created_at": "2026-04-04T00:00:00+00:00",
            "updated_at": "2026-04-04T00:07:00+00:00",
        }
    }


def test_runs_routes_are_exposed_in_openapi(client) -> None:
    spec = client.app.openapi()

    assert "/api/runs" in spec["paths"]
    assert "/api/runs/{run_id}" in spec["paths"]
    assert "/api/v2/runs" in spec["paths"]
    assert "/api/v2/runs/{run_id}" in spec["paths"]
