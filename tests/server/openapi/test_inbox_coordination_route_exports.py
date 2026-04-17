from __future__ import annotations

from aragora.server.handlers import ALL_HANDLERS
from aragora.server.handlers.coordination import CoordinationHandler
from aragora.server.handlers.features.unified_inbox.handler import UnifiedInboxHandler
from aragora.server.handlers.inbox_command import InboxCommandHandler
from aragora.server.openapi import generate_openapi_schema


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}


def _operation_methods(path_spec: dict[str, object]) -> set[str]:
    return {method for method in path_spec if method in HTTP_METHODS}


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
