from __future__ import annotations

from aragora.server.openapi import generate_openapi_schema


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}

EXPECTED_PIPELINE_GRAPH_METHODS = {
    "/api/v1/pipeline/graph": {"get", "post"},
    "/api/v1/pipeline/graph/{id}": {"get", "delete"},
    "/api/v1/pipeline/graph/{id}/react-flow": {"get"},
    "/api/v1/pipeline/graph/{id}/provenance/{node_id}": {"get"},
    "/api/v1/pipeline/graph/{id}/suggestions": {"get"},
    "/api/v1/pipeline/graph/{id}/node/{nodeId}/reassign": {"post"},
}


def _operation_methods(path_spec: dict[str, object]) -> set[str]:
    return {method for method in path_spec if method in HTTP_METHODS}


def test_pipeline_graph_routes_export_declared_methods() -> None:
    paths = generate_openapi_schema()["paths"]

    for path, expected_methods in EXPECTED_PIPELINE_GRAPH_METHODS.items():
        assert _operation_methods(paths[path]) == expected_methods


def test_pipeline_graph_get_only_routes_do_not_export_mutating_placeholders() -> None:
    paths = generate_openapi_schema()["paths"]

    for path in (
        "/api/v1/pipeline/graph/{id}/react-flow",
        "/api/v1/pipeline/graph/{id}/provenance/{node_id}",
        "/api/v1/pipeline/graph/{id}/suggestions",
    ):
        assert "delete" not in paths[path]
        assert "post" not in paths[path]
