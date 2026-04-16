from __future__ import annotations

from aragora.server.handlers import ALL_HANDLERS
from aragora.server.handlers.self_improve_details import SelfImproveDetailsHandler
from aragora.server.openapi import generate_openapi_schema


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}


def _operation_methods(paths: dict[str, dict[str, object]], path: str) -> set[str]:
    return {method for method in paths[path] if method in HTTP_METHODS}


def test_self_improve_details_handler_participates_in_all_handlers() -> None:
    assert SelfImproveDetailsHandler in set(ALL_HANDLERS)


def test_self_improve_details_routes_appear_in_generated_openapi() -> None:
    schema = generate_openapi_schema()
    paths = schema["paths"]

    required = {
        "/api/self-improve/meta-planner/goals",
        "/api/self-improve/execution/timeline",
        "/api/self-improve/learning/insights",
        "/api/self-improve/metrics/comparison",
        "/api/self-improve/trends/cycles",
        "/api/self-improve/improvement-queue",
        "/api/self-improve/improvement-queue/{id}/priority",
        "/api/self-improve/improvement-queue/{id}",
        "/api/v1/self-improve/meta-planner/goals",
        "/api/v1/self-improve/improvement-queue",
        "/api/v1/self-improve/improvement-queue/{id}/priority",
        "/api/v1/self-improve/improvement-queue/{id}",
    }

    assert required <= set(paths)


def test_self_improve_details_route_methods_are_preserved() -> None:
    schema = generate_openapi_schema()
    paths = schema["paths"]

    for path in (
        "/api/self-improve/meta-planner/goals",
        "/api/self-improve/execution/timeline",
        "/api/self-improve/learning/insights",
        "/api/self-improve/metrics/comparison",
        "/api/self-improve/trends/cycles",
        "/api/v1/self-improve/meta-planner/goals",
    ):
        assert _operation_methods(paths, path) == {"get"}

    for path in (
        "/api/self-improve/improvement-queue",
        "/api/v1/self-improve/improvement-queue",
    ):
        assert _operation_methods(paths, path) == {"post"}

    for path in (
        "/api/self-improve/improvement-queue/{id}/priority",
        "/api/v1/self-improve/improvement-queue/{id}/priority",
    ):
        assert _operation_methods(paths, path) == {"put"}

    for path in (
        "/api/self-improve/improvement-queue/{id}",
        "/api/v1/self-improve/improvement-queue/{id}",
    ):
        assert _operation_methods(paths, path) == {"delete"}
