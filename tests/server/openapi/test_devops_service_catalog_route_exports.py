from __future__ import annotations

from aragora.server.handlers import ALL_HANDLERS
from aragora.server.handlers.features.devops import DevOpsHandler
from aragora.server.openapi import generate_openapi_schema


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}


def _operation_methods(path_spec: dict[str, object]) -> set[str]:
    return {method for method in path_spec if method in HTTP_METHODS}


def test_devops_handler_participates_in_all_handlers() -> None:
    assert DevOpsHandler in set(ALL_HANDLERS)


def test_service_catalog_routes_match_live_devops_handler_contract() -> None:
    paths = generate_openapi_schema()["paths"]

    assert "/api/v1/services" in paths
    assert "/api/v1/services/{service_id}" in paths
    assert _operation_methods(paths["/api/v1/services"]) == {"get"}
    assert _operation_methods(paths["/api/v1/services/{service_id}"]) == {"get"}
    assert "/api/v1/services/{service_id}/health" not in paths
    assert "/api/v1/services/{service_id}/metrics" not in paths
