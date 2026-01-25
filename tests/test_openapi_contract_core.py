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
        assert expected_methods.issubset(
            methods
        ), f"{path} missing methods: {expected_methods - methods}"
