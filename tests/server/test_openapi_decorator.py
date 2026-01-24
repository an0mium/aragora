"""Tests for OpenAPI auto-generation decorator."""

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


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


class TestOpenAPIEndpoint:
    """Tests for OpenAPIEndpoint dataclass."""

    def test_basic_endpoint(self):
        """Test creating a basic endpoint."""
        endpoint = OpenAPIEndpoint(
            path="/api/test",
            method="GET",
            summary="Test endpoint",
            tags=["Test"],
        )

        assert endpoint.path == "/api/test"
        assert endpoint.method == "GET"
        assert endpoint.summary == "Test endpoint"
        assert endpoint.tags == ["Test"]
        assert endpoint.parameters == []
        assert endpoint.request_body is None
        assert endpoint.deprecated is False

    def test_to_openapi_spec_basic(self):
        """Test converting endpoint to OpenAPI spec."""
        endpoint = OpenAPIEndpoint(
            path="/api/test",
            method="GET",
            summary="Test endpoint",
            tags=["Test"],
        )

        spec = endpoint.to_openapi_spec()

        assert spec["summary"] == "Test endpoint"
        assert spec["tags"] == ["Test"]
        assert "responses" in spec
        assert "200" in spec["responses"]

    def test_to_openapi_spec_full(self):
        """Test converting endpoint with all fields."""
        endpoint = OpenAPIEndpoint(
            path="/api/test/{id}",
            method="POST",
            summary="Create test",
            tags=["Test"],
            description="Creates a new test resource",
            parameters=[{"name": "id", "in": "path", "required": True}],
            request_body={"content": {"application/json": {}}},
            responses={"201": {"description": "Created"}},
            security=[{"bearerAuth": []}],
            operation_id="createTest",
            deprecated=True,
        )

        spec = endpoint.to_openapi_spec()

        assert spec["summary"] == "Create test"
        assert spec["description"] == "Creates a new test resource"
        assert spec["operationId"] == "createTest"
        assert spec["parameters"] == [{"name": "id", "in": "path", "required": True}]
        assert spec["requestBody"] == {"content": {"application/json": {}}}
        assert spec["responses"] == {"201": {"description": "Created"}}
        assert spec["security"] == [{"bearerAuth": []}]
        assert spec["deprecated"] is True


class TestApiEndpointDecorator:
    """Tests for @api_endpoint decorator."""

    def test_basic_decoration(self):
        """Test decorating a simple function."""

        @api_endpoint(
            path="/api/test",
            method="GET",
            summary="Test endpoint",
            tags=["Test"],
        )
        def test_handler():
            return "result"

        # Function should still work
        assert test_handler() == "result"

        # Should have _openapi attribute
        assert hasattr(test_handler, "_openapi")
        assert test_handler._openapi.path == "/api/test"
        assert test_handler._openapi.method == "GET"

    def test_registers_endpoint(self):
        """Test that decorator registers endpoint in global registry."""

        @api_endpoint(
            path="/api/registered",
            method="POST",
            summary="Registered endpoint",
            tags=["Test"],
        )
        def registered_handler():
            pass

        endpoints = get_registered_endpoints()
        assert len(endpoints) == 1
        assert endpoints[0].path == "/api/registered"
        assert endpoints[0].method == "POST"

    def test_multiple_decorators(self):
        """Test registering multiple endpoints."""

        @api_endpoint(path="/api/first", method="GET", summary="First", tags=["A"])
        def first():
            pass

        @api_endpoint(path="/api/second", method="POST", summary="Second", tags=["B"])
        def second():
            pass

        endpoints = get_registered_endpoints()
        assert len(endpoints) == 2

        paths = [e.path for e in endpoints]
        assert "/api/first" in paths
        assert "/api/second" in paths

    def test_auth_required_default(self):
        """Test auth_required defaults to True."""

        @api_endpoint(path="/api/protected", method="GET", summary="Protected", tags=["Test"])
        def protected():
            pass

        endpoint = protected._openapi
        assert endpoint.security == [{"bearerAuth": []}]

    def test_auth_not_required(self):
        """Test auth_required=False removes security."""

        @api_endpoint(
            path="/api/public",
            method="GET",
            summary="Public",
            tags=["Test"],
            auth_required=False,
        )
        def public():
            pass

        endpoint = public._openapi
        assert endpoint.security == []

    def test_operation_id_from_function_name(self):
        """Test operation_id defaults to function name."""

        @api_endpoint(path="/api/test", method="GET", summary="Test", tags=["Test"])
        def my_custom_function():
            pass

        endpoint = my_custom_function._openapi
        assert endpoint.operation_id == "my_custom_function"

    def test_custom_operation_id(self):
        """Test custom operation_id."""

        @api_endpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=["Test"],
            operation_id="customOperationId",
        )
        def handler():
            pass

        endpoint = handler._openapi
        assert endpoint.operation_id == "customOperationId"

    def test_description_from_docstring(self):
        """Test description extracted from docstring."""

        @api_endpoint(path="/api/test", method="GET", summary="Test", tags=["Test"])
        def documented_handler():
            """This is the detailed description from docstring."""
            pass

        endpoint = documented_handler._openapi
        assert endpoint.description == "This is the detailed description from docstring."

    def test_explicit_description_overrides_docstring(self):
        """Test explicit description takes precedence over docstring."""

        @api_endpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=["Test"],
            description="Explicit description",
        )
        def documented_handler():
            """Docstring description."""
            pass

        endpoint = documented_handler._openapi
        assert endpoint.description == "Explicit description"

    def test_deprecated_flag(self):
        """Test deprecated flag."""

        @api_endpoint(
            path="/api/old",
            method="GET",
            summary="Old",
            tags=["Test"],
            deprecated=True,
        )
        def old_handler():
            pass

        endpoint = old_handler._openapi
        assert endpoint.deprecated is True

    def test_with_parameters(self):
        """Test with parameter definitions."""
        params = [
            {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
            {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 10}},
        ]

        @api_endpoint(
            path="/api/items/{id}",
            method="GET",
            summary="Get items",
            tags=["Items"],
            parameters=params,
        )
        def get_items():
            pass

        endpoint = get_items._openapi
        assert endpoint.parameters == params

    def test_with_request_body(self):
        """Test with request body definition."""
        body = {
            "content": {
                "application/json": {
                    "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                }
            }
        }

        @api_endpoint(
            path="/api/create",
            method="POST",
            summary="Create",
            tags=["Test"],
            request_body=body,
        )
        def create():
            pass

        endpoint = create._openapi
        assert endpoint.request_body == body

    def test_with_responses(self):
        """Test with custom responses."""
        responses = {
            "201": {"description": "Created"},
            "400": {"description": "Bad request"},
        }

        @api_endpoint(
            path="/api/create",
            method="POST",
            summary="Create",
            tags=["Test"],
            responses=responses,
        )
        def create():
            pass

        endpoint = create._openapi
        assert endpoint.responses == responses


class TestGetRegisteredEndpointsDict:
    """Tests for get_registered_endpoints_dict."""

    def test_empty_registry(self):
        """Test empty registry returns empty dict."""
        result = get_registered_endpoints_dict()
        assert result == {}

    def test_single_endpoint(self):
        """Test single endpoint converted to dict."""

        @api_endpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=["Test"],
        )
        def handler():
            pass

        result = get_registered_endpoints_dict()

        assert "/api/test" in result
        assert "get" in result["/api/test"]
        assert result["/api/test"]["get"]["summary"] == "Test"

    def test_multiple_methods_same_path(self):
        """Test multiple methods on same path."""

        @api_endpoint(path="/api/resource", method="GET", summary="Get", tags=["Test"])
        def get_handler():
            pass

        @api_endpoint(path="/api/resource", method="POST", summary="Create", tags=["Test"])
        def post_handler():
            pass

        result = get_registered_endpoints_dict()

        assert "/api/resource" in result
        assert "get" in result["/api/resource"]
        assert "post" in result["/api/resource"]

    def test_method_lowercase(self):
        """Test method keys are lowercase."""

        @api_endpoint(path="/api/test", method="DELETE", summary="Delete", tags=["Test"])
        def handler():
            pass

        result = get_registered_endpoints_dict()
        assert "delete" in result["/api/test"]


class TestRegisterEndpoint:
    """Tests for manual register_endpoint function."""

    def test_manual_registration(self):
        """Test manually registering an endpoint."""
        endpoint = OpenAPIEndpoint(
            path="/api/manual",
            method="GET",
            summary="Manual endpoint",
            tags=["Manual"],
        )

        register_endpoint(endpoint)

        endpoints = get_registered_endpoints()
        assert len(endpoints) == 1
        assert endpoints[0].path == "/api/manual"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_path_param(self):
        """Test path_param helper."""
        param = path_param("id", "Resource ID", "string")

        assert param["name"] == "id"
        assert param["in"] == "path"
        assert param["required"] is True
        assert param["description"] == "Resource ID"
        assert param["schema"]["type"] == "string"

    def test_query_param_basic(self):
        """Test query_param helper basic usage."""
        param = query_param("search", "Search term")

        assert param["name"] == "search"
        assert param["in"] == "query"
        assert "required" not in param
        assert param["schema"]["type"] == "string"

    def test_query_param_full(self):
        """Test query_param helper with all options."""
        param = query_param(
            "status",
            "Filter by status",
            schema_type="string",
            required=True,
            default="active",
            enum=["active", "inactive", "pending"],
        )

        assert param["name"] == "status"
        assert param["required"] is True
        assert param["schema"]["default"] == "active"
        assert param["schema"]["enum"] == ["active", "inactive", "pending"]

    def test_json_body(self):
        """Test json_body helper."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        body = json_body(schema, "Create resource")

        assert body["description"] == "Create resource"
        assert body["required"] is True
        assert body["content"]["application/json"]["schema"] == schema

    def test_ok_response(self):
        """Test ok_response helper."""
        response = ok_response("Success")

        assert "200" in response
        assert response["200"]["description"] == "Success"
        assert "application/json" in response["200"]["content"]

    def test_ok_response_with_schema(self):
        """Test ok_response with custom schema."""
        schema = {"type": "array", "items": {"type": "string"}}
        response = ok_response("List of items", schema)

        assert response["200"]["content"]["application/json"]["schema"] == schema

    def test_ok_response_custom_status(self):
        """Test ok_response with custom status code."""
        response = ok_response("Created", status_code="201")

        assert "201" in response
        assert response["201"]["description"] == "Created"

    def test_error_response(self):
        """Test error_response helper."""
        response = error_response("404", "Not found")

        assert "404" in response
        assert response["404"]["description"] == "Not found"


class TestIntegration:
    """Integration tests for OpenAPI decorator system."""

    def test_full_endpoint_example(self):
        """Test a realistic full endpoint definition."""

        @api_endpoint(
            path="/api/v1/debates/{debate_id}/export",
            method="GET",
            summary="Export debate",
            tags=["Debates", "Export"],
            description="Export a debate in various formats",
            parameters=[
                path_param("debate_id", "The debate ID"),
                query_param("format", "Export format", default="json", enum=["json", "csv", "pdf"]),
            ],
            responses={
                **ok_response("Exported debate"),
                **error_response("404", "Debate not found"),
            },
            auth_required=True,
        )
        async def export_debate(debate_id: str, format: str = "json"):
            pass

        # Verify endpoint metadata
        endpoint = export_debate._openapi
        assert endpoint.path == "/api/v1/debates/{debate_id}/export"
        assert "Debates" in endpoint.tags
        assert len(endpoint.parameters) == 2
        assert endpoint.security == [{"bearerAuth": []}]

        # Verify it's registered
        endpoints = get_registered_endpoints()
        assert len(endpoints) == 1

        # Verify dict conversion
        paths = get_registered_endpoints_dict()
        assert "/api/v1/debates/{debate_id}/export" in paths
        spec = paths["/api/v1/debates/{debate_id}/export"]["get"]
        assert spec["summary"] == "Export debate"
        assert "200" in spec["responses"]
        assert "404" in spec["responses"]

    def test_preserves_async_function(self):
        """Test that async functions remain async."""
        import asyncio

        @api_endpoint(path="/api/async", method="GET", summary="Async", tags=["Test"])
        async def async_handler():
            return "async result"

        # Should be able to run as async
        result = asyncio.get_event_loop().run_until_complete(async_handler())
        assert result == "async result"
