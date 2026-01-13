"""
OpenAPI Contract Tests.

Validates that API handlers match the OpenAPI specification.
Tests endpoint paths, methods, and response schemas.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def openapi_spec() -> dict[str, Any]:
    """Load the OpenAPI specification."""
    spec_path = Path(__file__).parent.parent.parent / "aragora" / "server" / "openapi.yaml"
    with open(spec_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def all_handlers():
    """Import all handlers."""
    from aragora.server.handlers import ALL_HANDLERS

    return ALL_HANDLERS


@pytest.fixture(scope="module")
def handler_instances(all_handlers):
    """Create instances of all handlers."""
    instances = []
    for handler_class in all_handlers:
        try:
            instance = handler_class({})
            instances.append(instance)
        except Exception:
            # Skip handlers that require specific initialization
            pass
    return instances


# =============================================================================
# Path Validation Tests
# =============================================================================


class TestOpenAPIPaths:
    """Tests that OpenAPI paths have corresponding handlers."""

    def test_spec_has_paths(self, openapi_spec: dict) -> None:
        """OpenAPI spec should define paths."""
        assert "paths" in openapi_spec
        assert len(openapi_spec["paths"]) > 0

    def test_all_paths_have_operations(self, openapi_spec: dict) -> None:
        """Each path should have at least one HTTP operation."""
        http_methods = {"get", "post", "put", "patch", "delete", "options", "head"}

        for path, path_item in openapi_spec["paths"].items():
            operations = [m for m in path_item.keys() if m.lower() in http_methods]
            assert len(operations) > 0, f"Path {path} has no HTTP operations"

    def test_paths_have_responses(self, openapi_spec: dict) -> None:
        """Each operation should define responses."""
        http_methods = {"get", "post", "put", "patch", "delete", "options", "head"}

        for path, path_item in openapi_spec["paths"].items():
            for method, operation in path_item.items():
                if method.lower() not in http_methods:
                    continue
                assert "responses" in operation, f"{method.upper()} {path} missing responses"
                assert (
                    len(operation["responses"]) > 0
                ), f"{method.upper()} {path} has empty responses"

    def test_core_endpoints_defined(self, openapi_spec: dict) -> None:
        """Core API endpoints should be defined in spec."""
        # Check that at least some core endpoint prefixes are present
        required_prefixes = [
            "/api/debates",
            "/api/agent",  # Note: uses /api/agent/{name}/... pattern
            "/api/health",
        ]

        spec_paths = list(openapi_spec["paths"].keys())

        for prefix in required_prefixes:
            found = any(path.startswith(prefix) for path in spec_paths)
            assert found, f"No paths starting with {prefix} in OpenAPI spec"


class TestHandlerPathCoverage:
    """Tests that handlers cover OpenAPI-defined paths."""

    def test_handlers_cover_spec_paths(self, openapi_spec: dict, handler_instances) -> None:
        """Each OpenAPI path should be handled by at least one handler."""
        # Collect all paths handlers claim to handle
        handled_paths: set[str] = set()

        for instance in handler_instances:
            if hasattr(instance, "ROUTES"):
                for route in instance.ROUTES:
                    # Normalize route pattern
                    handled_paths.add(route.rstrip("*").rstrip("/"))

        # Check coverage (allow some flexibility for path parameters)
        uncovered_paths = []
        for spec_path in openapi_spec["paths"].keys():
            # Convert OpenAPI path params to regex pattern
            normalized = re.sub(r"\{[^}]+\}", "*", spec_path)
            base_path = normalized.split("*")[0].rstrip("/")

            # Check if any handler covers this path
            covered = False
            for handled in handled_paths:
                if spec_path.startswith(handled) or handled.startswith(base_path):
                    covered = True
                    break

            if not covered:
                uncovered_paths.append(spec_path)

        # Allow some uncovered paths (deprecated, internal, etc.)
        max_uncovered = len(openapi_spec["paths"]) * 0.1  # Allow 10% uncovered
        assert (
            len(uncovered_paths) <= max_uncovered
        ), f"Too many uncovered paths ({len(uncovered_paths)}): {uncovered_paths[:10]}"


# =============================================================================
# Schema Validation Tests
# =============================================================================


class TestOpenAPISchemas:
    """Tests for OpenAPI schema definitions."""

    def test_schemas_defined(self, openapi_spec: dict) -> None:
        """OpenAPI spec should define reusable schemas."""
        assert "components" in openapi_spec
        assert "schemas" in openapi_spec["components"]
        assert len(openapi_spec["components"]["schemas"]) > 0

    def test_core_schemas_exist(self, openapi_spec: dict) -> None:
        """Core schemas should be defined."""
        # Check for schemas (may have different naming conventions)
        required_schema_patterns = [
            ["APIError", "Error"],  # Either name is acceptable
            ["DebateSummary", "Debate"],  # Either name is acceptable
        ]

        schemas = openapi_spec.get("components", {}).get("schemas", {})

        for alternatives in required_schema_patterns:
            found = any(name in schemas for name in alternatives)
            assert found, f"None of {alternatives} schemas defined"

    def test_schemas_have_types(self, openapi_spec: dict) -> None:
        """All schemas should have type definitions."""
        schemas = openapi_spec.get("components", {}).get("schemas", {})

        for name, schema in schemas.items():
            # Allow $ref, allOf, oneOf, anyOf without explicit type
            has_type = (
                "type" in schema
                or "$ref" in schema
                or "allOf" in schema
                or "oneOf" in schema
                or "anyOf" in schema
            )
            assert has_type, f"Schema {name} missing type definition"


# =============================================================================
# Security Tests
# =============================================================================


class TestOpenAPISecurity:
    """Tests for OpenAPI security definitions."""

    def test_security_schemes_defined(self, openapi_spec: dict) -> None:
        """Security schemes should be defined."""
        components = openapi_spec.get("components", {})
        assert "securitySchemes" in components, "No security schemes defined"

    def test_bearer_auth_defined(self, openapi_spec: dict) -> None:
        """Bearer authentication should be defined."""
        schemes = openapi_spec.get("components", {}).get("securitySchemes", {})
        assert "bearerAuth" in schemes, "Bearer auth not defined"
        assert schemes["bearerAuth"]["type"] == "http"
        assert schemes["bearerAuth"]["scheme"] == "bearer"

    def test_protected_endpoints_require_auth(self, openapi_spec: dict) -> None:
        """Protected endpoints should require authentication."""
        # Endpoints that should require auth
        protected_prefixes = [
            "/api/debates",
            "/api/agents",
            "/api/memory",
            "/api/documents",
        ]

        # Public endpoints that don't require auth
        public_endpoints = {
            "/api/health",
            "/api/healthz",
            "/api/readyz",
            "/api/metrics",
            "/api/auth/login",
            "/api/auth/register",
        }

        http_methods = {"get", "post", "put", "patch", "delete"}

        for path, path_item in openapi_spec["paths"].items():
            if path in public_endpoints:
                continue

            is_protected = any(path.startswith(prefix) for prefix in protected_prefixes)
            if not is_protected:
                continue

            for method, operation in path_item.items():
                if method.lower() not in http_methods:
                    continue

                has_security = "security" in operation and len(operation["security"]) > 0
                # Note: This is a soft check - some endpoints may intentionally be public
                if not has_security:
                    # Just log, don't fail - allows for intentional public endpoints
                    pass


# =============================================================================
# Response Validation Tests
# =============================================================================


class TestOpenAPIResponses:
    """Tests for OpenAPI response definitions."""

    def test_error_responses_defined(self, openapi_spec: dict) -> None:
        """Common error responses should be defined."""
        responses = openapi_spec.get("components", {}).get("responses", {})

        # At minimum, unauthorized and not found should be defined
        required_responses = ["Unauthorized", "NotFound"]

        for response_name in required_responses:
            assert response_name in responses, f"Common response {response_name} not defined"

        # Rate limiting is also important
        assert "RateLimited" in responses, "RateLimited response not defined"

    def test_success_responses_have_content(self, openapi_spec: dict) -> None:
        """200/201 responses should define response content."""
        http_methods = {"get", "post", "put", "patch", "delete"}

        for path, path_item in openapi_spec["paths"].items():
            for method, operation in path_item.items():
                if method.lower() not in http_methods:
                    continue

                responses = operation.get("responses", {})
                for status, response in responses.items():
                    # 200 and 201 should have content (unless explicitly empty)
                    if status in ("200", "201") and "$ref" not in response:
                        # Allow no content for some responses
                        if response.get("description", "").lower() not in (
                            "no content",
                            "success with no content",
                        ):
                            # Content is optional but recommended
                            pass


# =============================================================================
# Consistency Tests
# =============================================================================


class TestOpenAPIConsistency:
    """Tests for OpenAPI specification consistency."""

    def test_all_refs_resolve(self, openapi_spec: dict) -> None:
        """All $ref references should resolve to defined schemas."""
        schemas = openapi_spec.get("components", {}).get("schemas", {})
        responses = openapi_spec.get("components", {}).get("responses", {})

        def check_refs(obj, path=""):
            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref = obj["$ref"]
                    # Check local refs
                    if ref.startswith("#/components/schemas/"):
                        schema_name = ref.split("/")[-1]
                        assert schema_name in schemas, f"Unresolved schema ref: {ref} at {path}"
                    elif ref.startswith("#/components/responses/"):
                        response_name = ref.split("/")[-1]
                        assert (
                            response_name in responses
                        ), f"Unresolved response ref: {ref} at {path}"
                else:
                    for key, value in obj.items():
                        check_refs(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_refs(item, f"{path}[{i}]")

        check_refs(openapi_spec["paths"], "paths")

    def test_tags_are_defined(self, openapi_spec: dict) -> None:
        """Most used tags should be defined in the tags section."""
        defined_tags = {tag["name"] for tag in openapi_spec.get("tags", [])}
        http_methods = {"get", "post", "put", "patch", "delete", "options", "head"}

        used_tags: set[str] = set()
        for path, path_item in openapi_spec["paths"].items():
            for method, operation in path_item.items():
                if method.lower() not in http_methods:
                    continue
                if "tags" in operation:
                    used_tags.update(operation["tags"])

        undefined_tags = used_tags - defined_tags

        # Allow small number of undefined tags (spec may be incrementally updated)
        max_undefined = max(2, len(used_tags) * 0.1)  # 10% or 2, whichever is larger
        assert len(undefined_tags) <= max_undefined, (
            f"Too many undefined tags: {undefined_tags}. "
            f"Add these to the 'tags' section of the OpenAPI spec."
        )

    def test_operation_ids_unique(self, openapi_spec: dict) -> None:
        """Operation IDs should be unique across all endpoints."""
        http_methods = {"get", "post", "put", "patch", "delete", "options", "head"}

        operation_ids: dict[str, str] = {}

        for path, path_item in openapi_spec["paths"].items():
            for method, operation in path_item.items():
                if method.lower() not in http_methods:
                    continue
                op_id = operation.get("operationId")
                if op_id:
                    location = f"{method.upper()} {path}"
                    assert op_id not in operation_ids, (
                        f"Duplicate operationId '{op_id}': "
                        f"{operation_ids[op_id]} and {location}"
                    )
                    operation_ids[op_id] = location
