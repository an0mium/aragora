from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from aragora.server.fastapi import create_app


@pytest.fixture
def decision_service() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(fastapi_context_builder, decision_service: AsyncMock) -> TestClient:
    app = create_app()
    app.state.context = fastapi_context_builder(decision_service=decision_service)
    return TestClient(app, raise_server_exceptions=False)


def test_start_decision_requires_auth(client: TestClient) -> None:
    response = client.post("/api/v2/decisions", json={"task": "Ship it?"})

    assert response.status_code == 401


def test_list_decisions_forwards_filters(
    client: TestClient,
    decision_service: AsyncMock,
) -> None:
    decision_service.list_debates.return_value = []

    response = client.get("/api/v2/decisions?status=completed&limit=7")

    assert response.status_code == 200
    assert response.json() == []
    kwargs = decision_service.list_debates.await_args.kwargs
    assert kwargs["limit"] == 7
    assert kwargs["status"].value == "completed"


def test_list_decisions_rejects_malformed_status(client: TestClient) -> None:
    response = client.get("/api/v2/decisions?status=not-a-state")

    assert response.status_code == 400
    assert "Invalid status" in response.json()["detail"]


def test_get_decision_returns_404_when_missing(
    client: TestClient,
    decision_service: AsyncMock,
    override_fastapi_auth,
) -> None:
    decision_service.get_debate.return_value = None
    override_fastapi_auth(client, roles={"admin"}, permissions={"debates:create"})

    response = client.get("/api/v2/decisions/missing-debate")

    assert response.status_code == 404
    assert response.json()["detail"] == "Debate missing-debate not found"
