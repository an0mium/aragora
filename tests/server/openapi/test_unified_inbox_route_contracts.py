from __future__ import annotations

from aragora.server.auth_requirements import AuthLevel, get_requirement
from aragora.server.openapi import generate_openapi_schema


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}


def _operation_methods(path_spec: dict[str, object]) -> set[str]:
    return {method for method in path_spec if method in HTTP_METHODS}


def _operation(path: str, method: str) -> dict[str, object]:
    return generate_openapi_schema()["paths"][path][method]


def test_unified_inbox_routes_export_exact_methods() -> None:
    paths = generate_openapi_schema()["paths"]

    expected_methods = {
        "/api/v1/inbox/oauth/gmail": {"get"},
        "/api/v1/inbox/oauth/outlook": {"get"},
        "/api/v1/inbox/connect": {"post"},
        "/api/v1/inbox/accounts": {"get"},
        "/api/v1/inbox/accounts/{account_id}": {"delete"},
        "/api/v1/inbox/messages": {"get"},
        "/api/v1/inbox/messages/{message_id}": {"get"},
        "/api/v1/inbox/messages/{message_id}/debate": {"post"},
        "/api/v1/inbox/triage": {"post"},
        "/api/v1/inbox/bulk-action": {"post"},
        "/api/v1/inbox/stats": {"get"},
        "/api/v1/inbox/trends": {"get"},
    }

    for path, methods in expected_methods.items():
        assert path in paths
        assert _operation_methods(paths[path]) == methods


def test_unified_inbox_routes_export_security_and_request_bodies() -> None:
    protected_routes = [
        ("get", "/api/v1/inbox/oauth/gmail"),
        ("get", "/api/v1/inbox/oauth/outlook"),
        ("post", "/api/v1/inbox/connect"),
        ("get", "/api/v1/inbox/accounts"),
        ("delete", "/api/v1/inbox/accounts/{account_id}"),
        ("get", "/api/v1/inbox/messages"),
        ("get", "/api/v1/inbox/messages/{message_id}"),
        ("post", "/api/v1/inbox/messages/{message_id}/debate"),
        ("post", "/api/v1/inbox/triage"),
        ("post", "/api/v1/inbox/bulk-action"),
        ("get", "/api/v1/inbox/stats"),
        ("get", "/api/v1/inbox/trends"),
    ]

    for method, path in protected_routes:
        assert _operation(path, method)["security"] == [{"bearerAuth": []}]

    for path in [
        "/api/v1/inbox/connect",
        "/api/v1/inbox/messages/{message_id}/debate",
        "/api/v1/inbox/triage",
        "/api/v1/inbox/bulk-action",
    ]:
        request_body = _operation(path, "post")["requestBody"]
        assert request_body["content"]["application/json"]["schema"]["type"] == "object"

    connect_schema = _operation("/api/v1/inbox/connect", "post")["requestBody"]["content"][
        "application/json"
    ]["schema"]
    assert connect_schema["required"] == ["provider", "auth_code"]

    triage_schema = _operation("/api/v1/inbox/triage", "post")["requestBody"]["content"][
        "application/json"
    ]["schema"]
    assert triage_schema["required"] == ["message_ids"]


def test_unified_inbox_auth_manifest_matches_live_contracts() -> None:
    required_permissions = {
        ("GET", "/api/v1/inbox/oauth/gmail"): "inbox:read",
        ("GET", "/api/v1/inbox/oauth/outlook"): "inbox:read",
        ("POST", "/api/v1/inbox/connect"): "inbox:update",
        ("GET", "/api/v1/inbox/accounts"): "inbox:read",
        ("DELETE", "/api/v1/inbox/accounts/{account_id}"): "inbox:update",
        ("GET", "/api/v1/inbox/messages"): "inbox:read",
        ("GET", "/api/v1/inbox/messages/{message_id}"): "inbox:read",
        ("POST", "/api/v1/inbox/messages/{message_id}/debate"): "inbox:update",
        ("POST", "/api/v1/inbox/triage"): "inbox:update",
        ("POST", "/api/v1/inbox/bulk-action"): "inbox:update",
        ("GET", "/api/v1/inbox/stats"): "inbox:read",
        ("GET", "/api/v1/inbox/trends"): "inbox:read",
    }

    for (method, path), permission in required_permissions.items():
        requirement = get_requirement(path, method)
        assert requirement is not None
        assert requirement.level == AuthLevel.PERMISSION
        assert requirement.permission == permission
