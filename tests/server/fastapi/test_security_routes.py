from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from aragora.rbac.models import AuthorizationContext
from aragora.server.fastapi.dependencies.auth import require_authenticated
from aragora.server.fastapi.routes import security


def test_security_routes_require_auth(fastapi_client):
    response = fastapi_client.get("/api/v2/security/rbac-coverage")
    assert response.status_code == 401


def test_security_routes_reject_unexpected_query_params(fastapi_client, override_auth):
    override_auth(fastapi_client)
    response = fastapi_client.get("/api/v2/security/encryption-status?scope=invalid")
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid query"}


def test_rbac_coverage_returns_summary_shape(fastapi_client, fastapi_context, override_auth):
    checker = MagicMock()
    checker.list_assignments.return_value = ["one", "two"]
    fastapi_context["rbac_checker"] = checker

    override_auth(fastapi_client)
    response = fastapi_client.get("/api/v2/security/rbac-coverage")

    assert response.status_code == 200
    data = response.json()["data"]
    assert {
        "roles_defined",
        "permissions_defined",
        "assignments_active",
        "unprotected_endpoints",
        "total_endpoints",
        "coverage_percent",
    } <= data.keys()
    assert data["assignments_active"] == 2


def test_rbac_coverage_maps_assignment_failures_to_safe_summary(
    fastapi_client, fastapi_context, override_auth
):
    checker = MagicMock()
    checker.list_assignments.side_effect = OSError("backend unavailable")
    fastapi_context["rbac_checker"] = checker

    override_auth(fastapi_client)
    response = fastapi_client.get("/api/v2/security/rbac-coverage")

    assert response.status_code == 200
    assert response.json()["data"]["assignments_active"] == 0


def test_rbac_coverage_counts_openapi_operations_for_included_router_tree():
    app = FastAPI()
    parent = APIRouter(prefix="/api/v2")
    nested = APIRouter(prefix="/protected")

    @nested.get("/covered")
    async def covered_route() -> dict[str, bool]:
        return {"ok": True}

    parent.include_router(nested)
    app.include_router(parent)
    app.include_router(security.router)
    checker = MagicMock()
    checker.list_assignments.return_value = []
    app.state.context = {"rbac_checker": checker}
    app.dependency_overrides[require_authenticated] = lambda: AuthorizationContext(
        user_id="user-1",
        org_id="org-1",
        workspace_id="ws-1",
        roles={"admin"},
        permissions={"*"},
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v2/security/rbac-coverage")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total_endpoints"] == 3
    assert data["unprotected_endpoints"] == 0
    assert data["coverage_percent"] == 100.0


def test_encryption_status_maps_tls_failures_to_degraded(
    fastapi_client, override_auth, monkeypatch
):
    override_auth(fastapi_client)
    monkeypatch.setenv("ARAGORA_TLS_CERT_PATH", "/tmp/missing-cert.pem")

    response = fastapi_client.get("/api/v2/security/encryption-status")

    assert response.status_code == 200
    data = response.json()["data"]
    assert {"at_rest", "in_transit"} <= data.keys()
    assert data["in_transit"]["status"] == "degraded"
