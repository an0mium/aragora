"""
Core API contract tests for OpenAPI schema generation.
"""

from aragora.server.openapi import generate_openapi_schema


CORE_ENDPOINTS = {
    "/api/v1/debates": {"get", "post"},
    "/api/v1/debates/{id}": {"get"},
    "/api/v1/agents": {"get"},
    "/api/v1/plugins": {"get"},
    "/api/v1/plugins/{name}/run": {"post"},
    "/api/v1/auth/oauth/providers": {"get"},
    "/api/v1/knowledge/mound/governance/roles": {"post"},
}

PIPELINE_CANVAS_ENDPOINTS = {
    "/api/v1/canvas/pipeline": {"get"},
    "/api/v1/canvas/pipeline/from-braindump": {"post"},
    "/api/v1/canvas/pipeline/demo": {"post"},
    "/api/v1/canvas/pipeline/approve-transition": {"post"},
    "/api/v1/canvas/pipeline/auto-run": {"post"},
    "/api/v1/canvas/pipeline/extract-principles": {"post"},
    "/api/v1/canvas/pipeline/from-system-metrics": {"post"},
    "/api/v1/canvas/pipeline/{id}/execute": {"post"},
    "/api/v1/canvas/pipeline/{id}/intelligence": {"get"},
    "/api/v1/canvas/pipeline/{id}/beliefs": {"get"},
    "/api/v1/canvas/pipeline/{id}/explanations": {"get"},
    "/api/v1/canvas/pipeline/{id}/precedents": {"get"},
    "/api/v1/canvas/pipeline/{id}/self-improve": {"post"},
    "/api/v1/debates/{id}/to-pipeline": {"post"},
    "/api/v1/pipeline/{id}/agents": {"get"},
    "/api/v1/pipeline/{id}/agents/{agent_id}/approve": {"post"},
    "/api/v1/pipeline/{id}/agents/{agent_id}/reject": {"post"},
}


def test_core_endpoints_present() -> None:
    """Core endpoints must exist in the OpenAPI schema."""
    schema = generate_openapi_schema()
    paths = schema["paths"]
    missing = [path for path in CORE_ENDPOINTS if path not in paths]
    assert not missing, f"Missing core endpoints: {missing}"


def test_core_endpoints_methods() -> None:
    """Core endpoints must expose expected HTTP methods."""
    schema = generate_openapi_schema()
    paths = schema["paths"]
    for path, expected_methods in CORE_ENDPOINTS.items():
        methods = {m for m in paths[path].keys() if m not in ("parameters", "servers")}
        assert expected_methods.issubset(methods), (
            f"{path} missing methods: {expected_methods - methods}"
        )


def test_pipeline_canvas_endpoints_present() -> None:
    """Canvas pipeline endpoints used by the live pipeline UI must exist in the schema."""
    schema = generate_openapi_schema()
    paths = schema["paths"]
    missing = [path for path in PIPELINE_CANVAS_ENDPOINTS if path not in paths]
    assert not missing, f"Missing canvas pipeline endpoints: {missing}"


def test_pipeline_canvas_endpoint_methods() -> None:
    """Canvas pipeline endpoints must expose the handler-backed methods they support."""
    schema = generate_openapi_schema()
    paths = schema["paths"]
    for path, expected_methods in PIPELINE_CANVAS_ENDPOINTS.items():
        methods = {m for m in paths[path].keys() if m not in ("parameters", "servers")}
        assert expected_methods.issubset(methods), (
            f"{path} missing methods: {expected_methods - methods}"
        )
