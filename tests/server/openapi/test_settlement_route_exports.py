from __future__ import annotations

from aragora.server.handlers import ALL_HANDLERS
from aragora.server.handlers.settlements import SettlementHandler
from aragora.server.openapi import generate_openapi_schema


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}


def _operation_methods(path_spec: dict[str, object]) -> set[str]:
    return {method for method in path_spec if method in HTTP_METHODS}


def test_settlement_handler_participates_in_all_handlers() -> None:
    assert SettlementHandler in set(ALL_HANDLERS)


def test_settlement_routes_appear_in_generated_openapi() -> None:
    schema = generate_openapi_schema()
    paths = schema["paths"]

    required = {
        "/api/settlements",
        "/api/settlements/history",
        "/api/settlements/summary",
        "/api/settlements/{id}",
        "/api/settlements/{id}/settle",
        "/api/settlements/batch",
        "/api/settlements/agent/{agent}/accuracy",
        "/api/v1/settlements",
        "/api/v1/settlements/history",
        "/api/v1/settlements/summary",
        "/api/v1/settlements/{id}",
        "/api/v1/settlements/{id}/settle",
        "/api/v1/settlements/batch",
        "/api/v1/settlements/agent/{agent}/accuracy",
    }

    assert required <= set(paths)


def test_settlement_routes_use_declared_methods_without_generic_placeholder_paths() -> None:
    schema = generate_openapi_schema()
    paths = schema["paths"]

    expected_methods = {
        "/api/settlements": {"get"},
        "/api/settlements/history": {"get"},
        "/api/settlements/summary": {"get"},
        "/api/settlements/{id}": {"get"},
        "/api/settlements/{id}/settle": {"post"},
        "/api/settlements/batch": {"post"},
        "/api/settlements/agent/{agent}/accuracy": {"get"},
        "/api/v1/settlements": {"get"},
        "/api/v1/settlements/history": {"get"},
        "/api/v1/settlements/summary": {"get"},
        "/api/v1/settlements/{id}": {"get"},
        "/api/v1/settlements/{id}/settle": {"post"},
        "/api/v1/settlements/batch": {"post"},
        "/api/v1/settlements/agent/{agent}/accuracy": {"get"},
    }

    for path, methods in expected_methods.items():
        assert _operation_methods(paths[path]) == methods

    assert "/api/settlements/{param}" not in paths
    assert "/api/v1/settlements/{param}" not in paths
