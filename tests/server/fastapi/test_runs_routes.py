"""FastAPI tests for the backbone runs routes."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from aragora.pipeline.backbone_contracts import BackboneStage, RunLedger, RunStageEvent
from aragora.pipeline.execution_mode import ExecutionMode
from aragora.pipeline.plan_store import PlanStore
from aragora.server.fastapi import create_app
from aragora.server.fastapi.dependencies.auth import require_authenticated


def _make_run(
    run_id: str,
    *,
    status: str,
    execution_id: str = "",
    receipt_id: str = "",
    safety_mode: str | None = None,
    stage_events: list[RunStageEvent] | None = None,
) -> RunLedger:
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
    plan_store.create_run(
        _make_run(
            "run-fastapi-list",
            status="plan_ready",
            execution_id="exec-fastapi",
            receipt_id="receipt-fastapi",
            safety_mode=ExecutionMode.INTERACTIVE.value,
            stage_events=[RunStageEvent.create(BackboneStage.PLAN, status="completed")],
        )
    )

    _override_auth(client, {"orchestration:read"})
    response = client.get("/api/v2/runs")
    client.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "runs": [
            {
                "run_id": "run-fastapi-list",
                "status": "plan_ready",
                "stages": [
                    {
                        "stage": BackboneStage.PLAN.value,
                        "status": "completed",
                        "created_at": None,
                    }
                ],
                "execution_id": "exec-fastapi",
                "receipt_id": "receipt-fastapi",
                "safety_mode": ExecutionMode.INTERACTIVE.value,
                "created_at": None,
            }
        ]
    }


def test_get_run_route_requires_auth(client) -> None:
    response = client.get("/api/v2/runs/run-fastapi-detail")

    assert response.status_code == 401


def test_get_run_route_is_registered(client) -> None:
    plan_store = client.app.state.context["plan_store"]
    plan_store.create_run(
        _make_run(
            "run-fastapi-detail",
            status="execution_started",
            stage_events=[RunStageEvent.create(BackboneStage.EXECUTION, status="running")],
        )
    )

    _override_auth(client, {"orchestration:read"})
    response = client.get("/api/v2/runs/run-fastapi-detail")
    client.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "run": {
            "run_id": "run-fastapi-detail",
            "status": "execution_started",
            "stages": [
                {
                    "stage": BackboneStage.EXECUTION.value,
                    "status": "running",
                    "created_at": None,
                }
            ],
            "execution_id": None,
            "receipt_id": None,
            "safety_mode": None,
            "created_at": None,
        }
    }


def test_runs_routes_are_exposed_in_openapi(client) -> None:
    spec = client.app.openapi()

    assert "/api/v2/runs" in spec["paths"]
    assert "/api/v2/runs/{run_id}" in spec["paths"]
