from __future__ import annotations

from aragora.server.handlers import ALL_HANDLERS
from aragora.server.handlers.observability.crashes import CrashTelemetryHandler
from aragora.server.handlers.observability.dashboard import ObservabilityDashboardHandler
from aragora.server.openapi import generate_openapi_schema


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}


def _operation_methods(path_spec: dict[str, object]) -> set[str]:
    return {method for method in path_spec if method in HTTP_METHODS}


def test_observability_handlers_participate_in_all_handlers() -> None:
    handlers = set(ALL_HANDLERS)

    assert ObservabilityDashboardHandler in handlers
    assert CrashTelemetryHandler in handlers


def test_observability_routes_appear_in_generated_openapi() -> None:
    paths = generate_openapi_schema()["paths"]

    assert _operation_methods(paths["/api/v1/observability/dashboard"]) == {"get"}
    assert _operation_methods(paths["/api/v1/observability/metrics"]) == {"get"}
    assert _operation_methods(paths["/api/v1/observability/crashes"]) == {"get", "post"}
    assert _operation_methods(paths["/api/v1/observability/crashes/stats"]) == {"get"}
