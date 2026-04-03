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

    response = client.get("/api/v2/runs")

    assert response.status_code == 200
    assert response.json() == {
        "runs": [
            {
                "run_id": "run-fastapi-list",
                "status": "plan_ready",
                "stages": [{"stage": BackboneStage.PLAN.value, "status": "completed"}],
                "execution_id": "exec-fastapi",
                "receipt_id": "receipt-fastapi",
                "safety_mode": ExecutionMode.INTERACTIVE.value,
            }
        ]
    }


def test_get_run_route_is_registered(client) -> None:
    plan_store = client.app.state.context["plan_store"]
    plan_store.create_run(
        _make_run(
            "run-fastapi-detail",
            status="execution_started",
            stage_events=[RunStageEvent.create(BackboneStage.EXECUTION, status="running")],
        )
    )

    response = client.get("/api/v2/runs/run-fastapi-detail")

    assert response.status_code == 200
    assert response.json() == {
        "run": {
            "run_id": "run-fastapi-detail",
            "status": "execution_started",
            "stages": [{"stage": BackboneStage.EXECUTION.value, "status": "running"}],
            "execution_id": None,
            "receipt_id": None,
            "safety_mode": None,
        }
    }


def test_runs_routes_are_exposed_in_openapi(client) -> None:
    spec = client.app.openapi()

    assert "/api/v2/runs" in spec["paths"]
    assert "/api/v2/runs/{run_id}" in spec["paths"]
