from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from aragora.rbac.models import AuthorizationContext
from aragora.server.fastapi import create_app
from aragora.server.fastapi.dependencies.auth import require_authenticated


def _make_client() -> tuple[TestClient, AsyncMock]:
    app = create_app()
    service = AsyncMock()
    app.state.context = {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
        "user_store": None,
        "rbac_checker": MagicMock(),
        "decision_service": service,
    }
    return TestClient(app, raise_server_exceptions=False), service


def _override_auth(client: TestClient, *permissions: str) -> None:
    auth_ctx = AuthorizationContext(
        user_id="user-1",
        org_id="org-1",
        workspace_id="ws-1",
        roles={"admin"},
        permissions=set(permissions),
    )
    client.app.dependency_overrides[require_authenticated] = lambda: auth_ctx


def test_start_decision_requires_auth() -> None:
    client, _service = _make_client()

    response = client.post("/api/v2/decisions", json={"task": "Ship it?"})

    assert response.status_code == 401


def test_list_decisions_forwards_filters() -> None:
    client, service = _make_client()
    service.list_debates.return_value = []

    response = client.get("/api/v2/decisions?status=completed&limit=7")

    assert response.status_code == 200
    assert response.json() == []
    kwargs = service.list_debates.await_args.kwargs
    assert kwargs["limit"] == 7
    assert kwargs["status"].value == "completed"


def test_list_decisions_rejects_malformed_status() -> None:
    client, _service = _make_client()

    response = client.get("/api/v2/decisions?status=not-a-state")

    assert response.status_code == 400
    assert "Invalid status" in response.json()["detail"]


def test_get_decision_returns_404_when_missing() -> None:
    client, service = _make_client()
    service.get_debate.return_value = None
    _override_auth(client, "debates:create")

    response = client.get("/api/v2/decisions/missing-debate")

    assert response.status_code == 404
    assert response.json()["detail"] == "Debate missing-debate not found"
