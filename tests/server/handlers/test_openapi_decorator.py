"""
Tests for OpenAPI auto-generation decorator module.

Tests cover:
- OpenAPIEndpoint dataclass and to_openapi_spec()
- api_endpoint decorator registration
- Global registry management (get, clear, register)
- Endpoint dict generation
- Helper functions (path_param, query_param, json_body, ok_response, error_response)
- Pydantic model schema extraction
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.server.handlers.openapi_decorator import (
    OpenAPIEndpoint,
    api_endpoint,
    clear_registry,
    error_response,
    get_registered_endpoints,
    get_registered_endpoints_dict,
    json_body,
    ok_response,
    path_param,
    query_param,
    register_endpoint,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear endpoint registry between tests."""
    clear_registry()
    yield
    clear_registry()


# ===========================================================================
# OpenAPIEndpoint Tests
# ===========================================================================


class TestOpenAPIEndpoint:
    """Tests for OpenAPIEndpoint dataclass."""

    def test_basic_creation(self):
        endpoint = OpenAPIEndpoint(
            path="/api/v1/test",
            method="GET",
            summary="Test endpoint",
            tags=["Test"],
        )
        assert endpoint.path == "/api/v1/test"
        assert endpoint.method == "GET"
        assert endpoint.summary == "Test endpoint"
        assert endpoint.tags == ["Test"]

    def test_default_values(self):
        endpoint = OpenAPIEndpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=[],
        )
        assert endpoint.description == ""
        assert endpoint.parameters == []
        assert endpoint.request_body is None
        assert endpoint.responses == {}
        assert endpoint.security == []
        assert endpoint.operation_id is None
        assert endpoint.deprecated is False

    def test_to_openapi_spec_minimal(self):
        endpoint = OpenAPIEndpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=["Test"],
        )
        spec = endpoint.to_openapi_spec()
        assert spec["summary"] == "Test"
        assert spec["tags"] == ["Test"]
        # Should have default 200 response
        assert "200" in spec["responses"]

    def test_to_openapi_spec_with_description(self):
        endpoint = OpenAPIEndpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=[],
            description="Detailed description",
        )
        spec = endpoint.to_openapi_spec()
        assert spec["description"] == "Detailed description"

    def test_to_openapi_spec_with_operation_id(self):
        endpoint = OpenAPIEndpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=[],
            operation_id="getTest",
        )
        spec = endpoint.to_openapi_spec()
        assert spec["operationId"] == "getTest"

    def test_to_openapi_spec_with_parameters(self):
        params = [{"name": "id", "in": "path", "required": True}]
        endpoint = OpenAPIEndpoint(
            path="/api/test/{id}",
            method="GET",
            summary="Test",
            tags=[],
            parameters=params,
        )
        spec = endpoint.to_openapi_spec()
        assert spec["parameters"] == params

    def test_to_openapi_spec_with_request_body(self):
        body = {"description": "Request", "content": {"application/json": {}}}
        endpoint = OpenAPIEndpoint(
            path="/api/test",
            method="POST",
            summary="Test",
            tags=[],
            request_body=body,
        )
        spec = endpoint.to_openapi_spec()
        assert spec["requestBody"] == body

    def test_to_openapi_spec_with_custom_responses(self):
        responses = {"200": {"description": "OK"}, "404": {"description": "Not Found"}}
        endpoint = OpenAPIEndpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=[],
            responses=responses,
        )
        spec = endpoint.to_openapi_spec()
        assert spec["responses"] == responses

    def test_to_openapi_spec_with_security(self):
        security = [{"bearerAuth": []}]
        endpoint = OpenAPIEndpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=[],
            security=security,
        )
        spec = endpoint.to_openapi_spec()
        assert spec["security"] == security

    def test_to_openapi_spec_deprecated(self):
        endpoint = OpenAPIEndpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=[],
            deprecated=True,
        )
        spec = endpoint.to_openapi_spec()
        assert spec["deprecated"] is True

    def test_to_openapi_spec_not_deprecated_excludes_key(self):
        endpoint = OpenAPIEndpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=[],
            deprecated=False,
        )
        spec = endpoint.to_openapi_spec()
        assert "deprecated" not in spec


# ===========================================================================
# api_endpoint Decorator Tests
# ===========================================================================


class TestApiEndpointDecorator:
    """Tests for the api_endpoint decorator."""

    def test_registers_endpoint(self):
        @api_endpoint(path="/api/v1/items", method="GET", summary="List items", tags=["Items"])
        def list_items():
            return []

        endpoints = get_registered_endpoints()
        assert len(endpoints) == 1
        assert endpoints[0].path == "/api/v1/items"
        assert endpoints[0].method == "GET"

    def test_decorated_function_has_openapi_attr(self):
        @api_endpoint(path="/api/v1/items", method="GET", summary="List items", tags=["Items"])
        def list_items():
            return []

        assert hasattr(list_items, "_openapi")
        assert list_items._openapi.path == "/api/v1/items"

    def test_decorated_function_remains_callable(self):
        @api_endpoint(path="/api/v1/echo", method="POST", summary="Echo", tags=["Test"])
        def echo(value):
            return value

        assert echo("hello") == "hello"

    def test_preserves_function_name(self):
        @api_endpoint(path="/api/test", summary="Test", tags=[])
        def my_handler():
            pass

        assert my_handler.__name__ == "my_handler"

    def test_method_defaults_to_get(self):
        @api_endpoint(path="/api/test", summary="Test", tags=[])
        def handler():
            pass

        endpoints = get_registered_endpoints()
        assert endpoints[0].method == "GET"

    def test_method_uppercased(self):
        @api_endpoint(path="/api/test", method="post", summary="Test", tags=[])
        def handler():
            pass

        endpoints = get_registered_endpoints()
        assert endpoints[0].method == "POST"

    def test_auto_summary_from_function_name(self):
        @api_endpoint(path="/api/test", tags=[])
        def get_all_items():
            pass

        endpoints = get_registered_endpoints()
        assert "Get All Items" in endpoints[0].summary

    def test_uses_docstring_as_description(self):
        @api_endpoint(path="/api/test", summary="Test", tags=[])
        def handler():
            """My detailed description."""
            pass

        endpoints = get_registered_endpoints()
        assert endpoints[0].description == "My detailed description."

    def test_auth_required_adds_security(self):
        @api_endpoint(path="/api/test", summary="Test", tags=[], auth_required=True)
        def handler():
            pass

        endpoints = get_registered_endpoints()
        assert len(endpoints[0].security) > 0
        assert "bearerAuth" in endpoints[0].security[0]

    def test_auth_not_required_no_security(self):
        @api_endpoint(path="/api/test", summary="Test", tags=[], auth_required=False)
        def handler():
            pass

        endpoints = get_registered_endpoints()
        assert len(endpoints[0].security) == 0

    def test_deprecated_flag(self):
        @api_endpoint(path="/api/test", summary="Test", tags=[], deprecated=True)
        def handler():
            pass

        endpoints = get_registered_endpoints()
        assert endpoints[0].deprecated is True

    def test_custom_operation_id(self):
        @api_endpoint(path="/api/test", summary="Test", tags=[], operation_id="customOp")
        def handler():
            pass

        endpoints = get_registered_endpoints()
        assert endpoints[0].operation_id == "customOp"


# ===========================================================================
# Registry Management
# ===========================================================================


class TestRegistryManagement:
    """Tests for registry get, clear, register functions."""

    def test_get_registered_endpoints_returns_copy(self):
        register_endpoint(OpenAPIEndpoint(path="/api/a", method="GET", summary="A", tags=[]))
        eps1 = get_registered_endpoints()
        eps2 = get_registered_endpoints()
        assert eps1 is not eps2

    def test_clear_registry_empties_list(self):
        register_endpoint(OpenAPIEndpoint(path="/api/a", method="GET", summary="A", tags=[]))
        clear_registry()
        assert len(get_registered_endpoints()) == 0

    def test_register_endpoint_manually(self):
        ep = OpenAPIEndpoint(path="/api/manual", method="POST", summary="Manual", tags=["Manual"])
        register_endpoint(ep)
        eps = get_registered_endpoints()
        assert len(eps) == 1
        assert eps[0].path == "/api/manual"

    def test_get_registered_endpoints_dict_format(self):
        register_endpoint(
            OpenAPIEndpoint(path="/api/items", method="GET", summary="List", tags=["Items"])
        )
        register_endpoint(
            OpenAPIEndpoint(path="/api/items", method="POST", summary="Create", tags=["Items"])
        )
        d = get_registered_endpoints_dict()
        assert "/api/items" in d
        assert "get" in d["/api/items"]
        assert "post" in d["/api/items"]

    def test_endpoints_dict_multiple_paths(self):
        register_endpoint(OpenAPIEndpoint(path="/api/a", method="GET", summary="A", tags=[]))
        register_endpoint(OpenAPIEndpoint(path="/api/b", method="GET", summary="B", tags=[]))
        d = get_registered_endpoints_dict()
        assert "/api/a" in d
        assert "/api/b" in d


# ===========================================================================
# Helper Functions
# ===========================================================================


class TestHelperFunctions:
    """Tests for parameter and response helper functions."""

    def test_path_param(self):
        param = path_param("id", "Resource ID", "string")
        assert param["name"] == "id"
        assert param["in"] == "path"
        assert param["required"] is True
        assert param["schema"]["type"] == "string"

    def test_path_param_integer(self):
        param = path_param("page", "Page number", "integer")
        assert param["schema"]["type"] == "integer"

    def test_query_param_basic(self):
        param = query_param("search", "Search query")
        assert param["name"] == "search"
        assert param["in"] == "query"
        assert param["schema"]["type"] == "string"

    def test_query_param_required(self):
        param = query_param("q", required=True)
        assert param["required"] is True

    def test_query_param_with_default(self):
        param = query_param("limit", default=10, schema_type="integer")
        assert param["schema"]["default"] == 10

    def test_query_param_with_enum(self):
        param = query_param("status", enum=["active", "inactive"])
        assert param["schema"]["enum"] == ["active", "inactive"]

    def test_json_body_with_dict_schema(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        body = json_body(schema, "Create item")
        assert body["description"] == "Create item"
        assert body["required"] is True
        assert body["content"]["application/json"]["schema"] == schema

    def test_json_body_not_required(self):
        body = json_body({"type": "object"}, required=False)
        assert body["required"] is False

    def test_ok_response_basic(self):
        resp = ok_response("Success")
        assert "200" in resp
        assert resp["200"]["description"] == "Success"

    def test_ok_response_custom_status(self):
        resp = ok_response("Created", status_code="201")
        assert "201" in resp

    def test_ok_response_with_schema(self):
        schema = {"type": "object", "properties": {"id": {"type": "string"}}}
        resp = ok_response("Success", schema=schema)
        assert resp["200"]["content"]["application/json"]["schema"] == schema

    def test_error_response_format(self):
        resp = error_response("404", "Not found")
        assert "404" in resp
        assert resp["404"]["description"] == "Not found"
        assert "error" in resp["404"]["content"]["application/json"]["schema"]["properties"]


# ===========================================================================
# Pydantic Model Integration
# ===========================================================================


class TestPydanticIntegration:
    """Tests for Pydantic model schema extraction."""

    def test_pydantic_request_model(self):
        """Test that Pydantic models generate request body schema."""
        try:
            from pydantic import BaseModel

            class TestRequest(BaseModel):
                name: str
                value: int

            @api_endpoint(
                path="/api/test",
                method="POST",
                summary="Test",
                tags=[],
                request_model=TestRequest,
            )
            def handler():
                pass

            endpoints = get_registered_endpoints()
            assert endpoints[0].request_body is not None
            assert "content" in endpoints[0].request_body
        except ImportError:
            pytest.skip("pydantic not installed")

    def test_pydantic_response_model(self):
        """Test that Pydantic models generate response schema."""
        try:
            from pydantic import BaseModel

            class TestResponse(BaseModel):
                id: str

            @api_endpoint(
                path="/api/test",
                method="GET",
                summary="Test",
                tags=[],
                response_model=TestResponse,
            )
            def handler():
                pass

            endpoints = get_registered_endpoints()
            assert "200" in endpoints[0].responses
        except ImportError:
            pytest.skip("pydantic not installed")

    def test_json_body_with_pydantic_model(self):
        """Test json_body helper with Pydantic model."""
        try:
            from pydantic import BaseModel

            class MyModel(BaseModel):
                field: str

            body = json_body(MyModel)
            assert "content" in body
            assert "MyModel request" in body["description"]
        except ImportError:
            pytest.skip("pydantic not installed")
