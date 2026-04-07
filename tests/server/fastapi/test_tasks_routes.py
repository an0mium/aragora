"""Tests for FastAPI v2 task routes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aragora.control_plane.scheduler import TaskPriority
from aragora.server.fastapi import create_app


@pytest.fixture
def app():
    app = create_app()
    app.state.context = {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
        "user_store": None,
        "rbac_checker": MagicMock(),
        "decision_service": MagicMock(),
    }
    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def test_submit_task_returns_created_task_id(client):
    control_plane = MagicMock()
    control_plane.submit_task = AsyncMock(return_value="task-123")

    with patch(
        "aragora.control_plane.integration.get_integrated_control_plane",
        return_value=control_plane,
    ):
        response = client.post(
            "/api/v2/tasks",
            json={
                "task_type": "analysis",
                "payload": {"topic": "queue health"},
                "required_capabilities": ["deliberation"],
                "priority": "high",
                "timeout_seconds": 30,
                "metadata": {"request_id": "req-1"},
            },
        )

    assert response.status_code == 201
    assert response.json() == {"data": {"task_id": "task-123"}}
    control_plane.submit_task.assert_awaited_once_with(
        task_type="analysis",
        payload={"topic": "queue health"},
        required_capabilities=["deliberation"],
        priority=TaskPriority.HIGH,
        timeout_seconds=30,
        metadata={"request_id": "req-1"},
    )


def test_submit_task_rejects_invalid_priority(client):
    with patch(
        "aragora.control_plane.integration.get_integrated_control_plane",
        return_value=MagicMock(),
    ):
        response = client.post(
            "/api/v2/tasks",
            json={"task_type": "analysis", "priority": "not-a-priority"},
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid priority: not-a-priority"}


def test_claim_task_returns_serialized_task(client):
    task = MagicMock()
    task.to_dict.return_value = {"id": "task-123", "task_type": "analysis"}

    control_plane = MagicMock()
    control_plane.coordinator.claim_task = AsyncMock(return_value=task)

    with patch(
        "aragora.control_plane.integration.get_integrated_control_plane",
        return_value=control_plane,
    ):
        response = client.post(
            "/api/v2/tasks/claim",
            json={"agent_id": "agent-1", "capabilities": ["deliberation"], "block_ms": 10},
        )

    assert response.status_code == 200
    assert response.json() == {"data": {"task": {"id": "task-123", "task_type": "analysis"}}}


def test_task_history_route_is_not_shadowed_by_task_lookup_route(client):
    scheduler = MagicMock()
    scheduler.list_by_status = AsyncMock(return_value=[])

    control_plane = MagicMock()
    control_plane.coordinator = SimpleNamespace(
        _scheduler_bridge=SimpleNamespace(_scheduler=scheduler)
    )
    control_plane.get_task = AsyncMock(return_value=None)

    with patch(
        "aragora.control_plane.integration.get_integrated_control_plane",
        return_value=control_plane,
    ):
        response = client.get("/api/v2/tasks/history")

    assert response.status_code == 200
    assert response.json()["data"]["history"] == []
    assert control_plane.get_task.await_count == 0


def test_queue_metrics_falls_back_to_zeroes_without_async_stats(client):
    with patch(
        "aragora.control_plane.integration.get_integrated_control_plane",
        return_value=MagicMock(),
    ):
        response = client.get("/api/v2/queue/metrics")

    assert response.status_code == 200
    assert response.json() == {
        "data": {
            "pending": 0,
            "running": 0,
            "completed_today": 0,
            "failed_today": 0,
            "avg_wait_time_ms": 0,
            "avg_execution_time_ms": 0,
            "throughput_per_minute": 0,
        }
    }


def test_queue_metrics_maps_control_plane_stats(client):
    control_plane = MagicMock()
    control_plane.get_stats = AsyncMock(
        return_value={
            "pending_tasks": 3,
            "running_tasks": 1,
            "completed_tasks": 7,
            "failed_tasks": 2,
            "avg_wait_time_ms": 125,
            "avg_execution_time_ms": 450,
            "throughput_per_minute": 4,
        }
    )

    with patch(
        "aragora.control_plane.integration.get_integrated_control_plane",
        return_value=control_plane,
    ):
        response = client.get("/api/v2/queue/metrics")

    assert response.status_code == 200
    assert response.json() == {
        "data": {
            "pending": 3,
            "running": 1,
            "completed_today": 7,
            "failed_today": 2,
            "avg_wait_time_ms": 125,
            "avg_execution_time_ms": 450,
            "throughput_per_minute": 4,
        }
    }


def test_tasks_routes_are_exposed_in_openapi(client):
    spec = client.app.openapi()

    assert "/api/v2/tasks" in spec["paths"]
    assert "/api/v2/tasks/claim" in spec["paths"]
    assert "/api/v2/tasks/{task_id}" in spec["paths"]
    assert "/api/v2/tasks/history" in spec["paths"]
    assert "/api/v2/queue/metrics" in spec["paths"]
