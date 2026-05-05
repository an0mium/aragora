from __future__ import annotations

from typing import Any, cast

from aragora.server.auth_requirements import AuthLevel, get_requirement
from aragora.server.handlers import ALL_HANDLERS
from aragora.server.handlers.coordination import CoordinationHandler
from aragora.server.handlers.features.unified_inbox.handler import UnifiedInboxHandler
from aragora.server.handlers.inbox_command import InboxCommandHandler
from aragora.server.openapi import generate_openapi_schema


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}


def _operation_methods(path_spec: dict[str, Any]) -> set[str]:
    return {method for method in path_spec if method in HTTP_METHODS}


def _operation(path: str, method: str) -> dict[str, Any]:
    return generate_openapi_schema()["paths"][path][method]


def test_inbox_and_coordination_handlers_participate_in_all_handlers() -> None:
    handlers = set(ALL_HANDLERS)

    assert CoordinationHandler in handlers
    assert UnifiedInboxHandler in handlers
    assert InboxCommandHandler in handlers


def test_coordination_routes_export_exact_methods() -> None:
    paths = generate_openapi_schema()["paths"]

    expected_methods = {
        "/api/v1/coordination/workspaces": {"get", "post"},
        "/api/v1/coordination/workspaces/{workspace_id}": {"delete"},
        "/api/v1/coordination/federation": {"get", "post"},
        "/api/v1/coordination/execute": {"post"},
        "/api/v1/coordination/executions": {"get"},
        "/api/v1/coordination/consent": {"get", "post"},
        "/api/v1/coordination/consent/{consent_id}": {"delete"},
        "/api/v1/coordination/approve/{request_id}": {"post"},
        "/api/v1/coordination/stats": {"get"},
        "/api/v1/coordination/health": {"get"},
    }

    for path, methods in expected_methods.items():
        assert path in paths
        assert _operation_methods(paths[path]) == methods


def test_inbox_and_coordination_routes_export_security_contracts() -> None:
    protected_routes = [
        ("post", "/api/v1/inbox/connect"),
        ("post", "/api/v1/inbox/triage"),
        ("post", "/api/v1/inbox/bulk-action"),
        ("get", "/api/v1/inbox/command"),
        ("post", "/api/v1/inbox/actions"),
        ("post", "/api/v1/inbox/bulk-actions"),
        ("post", "/api/v1/inbox/reprioritize"),
        ("get", "/api/v1/coordination/workspaces"),
        ("post", "/api/v1/coordination/workspaces"),
        ("post", "/api/v1/coordination/execute"),
        ("post", "/api/v1/coordination/approve/{request_id}"),
    ]

    for method, path in protected_routes:
        operation = _operation(path, method)
        assert operation["security"] == [{"bearerAuth": []}]

    assert "security" not in _operation("/api/v1/coordination/health", "get")


def test_inbox_and_coordination_routes_export_request_bodies() -> None:
    request_body_routes = [
        "/api/v1/inbox/connect",
        "/api/v1/inbox/messages/{message_id}/debate",
        "/api/v1/inbox/triage",
        "/api/v1/inbox/bulk-action",
        "/api/v1/inbox/actions",
        "/api/v1/inbox/bulk-actions",
        "/api/v1/inbox/reprioritize",
        "/api/v1/coordination/workspaces",
        "/api/v1/coordination/federation",
        "/api/v1/coordination/execute",
        "/api/v1/coordination/consent",
        "/api/v1/coordination/approve/{request_id}",
    ]

    for path in request_body_routes:
        operation = _operation(path, "post")
        assert "requestBody" in operation
        request_body = cast(dict[str, Any], operation["requestBody"])
        assert request_body["content"]["application/json"]["schema"]["type"] == "object"

    connect_request_body = cast(
        dict[str, Any], _operation("/api/v1/inbox/connect", "post")["requestBody"]
    )
    connect_schema = connect_request_body["content"]["application/json"]["schema"]
    assert connect_schema["required"] == ["provider", "auth_code"]

    execute_request_body = cast(
        dict[str, Any], _operation("/api/v1/coordination/execute", "post")["requestBody"]
    )
    execute_schema = execute_request_body["content"]["application/json"]["schema"]
    assert execute_schema["required"] == [
        "operation",
        "source_workspace_id",
        "target_workspace_id",
    ]


def test_inbox_and_coordination_auth_manifest_matches_live_contracts() -> None:
    required_permissions = {
        ("POST", "/api/v1/inbox/connect"): "inbox:update",
        ("POST", "/api/v1/inbox/actions"): "inbox:write",
        ("GET", "/api/v1/coordination/workspaces"): "coordination:read",
        ("POST", "/api/v1/coordination/workspaces"): "coordination:write",
        ("POST", "/api/v1/coordination/approve/{request_id}"): "coordination:admin",
    }

    for (method, path), permission in required_permissions.items():
        requirement = get_requirement(path, method)
        assert requirement is not None
        assert requirement.level == AuthLevel.PERMISSION
        assert requirement.permission == permission

    health_requirement = get_requirement("/api/v1/coordination/health", "GET")
    assert health_requirement is not None
    assert health_requirement.level == AuthLevel.PUBLIC


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


def test_inbox_command_routes_export_exact_methods() -> None:
    paths = generate_openapi_schema()["paths"]

    expected_methods = {
        "/api/v1/inbox/command": {"get"},
        "/api/v1/inbox/actions": {"post"},
        "/api/v1/inbox/bulk-actions": {"post"},
        "/api/v1/inbox/sender-profile": {"get"},
        "/api/v1/inbox/daily-digest": {"get"},
        "/api/v1/inbox/reprioritize": {"post"},
    }

    for path, methods in expected_methods.items():
        assert path in paths
        assert _operation_methods(paths[path]) == methods
