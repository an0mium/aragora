"""
Tests for aragora.server.handlers.openapi_decorator module.

Covers:
1. OpenAPIEndpoint dataclass and to_openapi_spec()
2. @api_endpoint decorator functionality
3. Registry management (get_registered_endpoints, clear_registry)
4. Parameter helper functions (path_param, query_param)
5. Request body helpers (json_body)
6. Response helpers (ok_response, error_response)
7. Pydantic model schema extraction
8. Edge cases and error handling
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.openapi_decorator import (
    OpenAPIEndpoint,
    _extract_pydantic_schema,
    _is_pydantic_model,
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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_registry_before_each():
    """Clear registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


# =============================================================================
# TestOpenAPIEndpoint
# =============================================================================


class TestOpenAPIEndpoint:
    """Tests for the OpenAPIEndpoint dataclass."""

    def test_basic_instantiation(self):
        """Should create endpoint with required fields."""
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
        """Should have correct default values."""
        endpoint = OpenAPIEndpoint(
            path="/api/v1/test",
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

    def test_full_instantiation(self):
        """Should create endpoint with all fields."""
        endpoint = OpenAPIEndpoint(
            path="/api/v1/users/{id}",
            method="PUT",
            summary="Update user",
            tags=["Users", "Admin"],
            description="Update a user by ID",
            parameters=[{"name": "id", "in": "path", "required": True}],
            request_body={"content": {"application/json": {"schema": {}}}},
            responses={"200": {"description": "Success"}},
            security=[{"bearerAuth": []}],
            operation_id="updateUser",
            deprecated=True,
        )
        assert endpoint.description == "Update a user by ID"
        assert len(endpoint.parameters) == 1
        assert endpoint.request_body is not None
        assert "200" in endpoint.responses
        assert endpoint.security == [{"bearerAuth": []}]
        assert endpoint.operation_id == "updateUser"
        assert endpoint.deprecated is True


class TestOpenAPIEndpointToSpec:
    """Tests for OpenAPIEndpoint.to_openapi_spec()."""

    def test_minimal_spec(self):
        """Should generate minimal spec with required fields."""
        endpoint = OpenAPIEndpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=["Test"],
        )
        spec = endpoint.to_openapi_spec()
        assert spec["summary"] == "Test"
        assert spec["tags"] == ["Test"]
        # Default response added
        assert "200" in spec["responses"]

    def test_includes_description(self):
        """Should include description when provided."""
        endpoint = OpenAPIEndpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=[],
            description="Detailed description",
        )
        spec = endpoint.to_openapi_spec()
        assert spec["description"] == "Detailed description"

    def test_excludes_empty_description(self):
        """Should not include description when empty."""
        endpoint = OpenAPIEndpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=[],
            description="",
        )
        spec = endpoint.to_openapi_spec()
        assert "description" not in spec

    def test_includes_operation_id(self):
        """Should include operationId when provided."""
        endpoint = OpenAPIEndpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=[],
            operation_id="getTest",
        )
        spec = endpoint.to_openapi_spec()
        assert spec["operationId"] == "getTest"

    def test_includes_parameters(self):
        """Should include parameters when provided."""
        endpoint = OpenAPIEndpoint(
            path="/api/users/{id}",
            method="GET",
            summary="Get user",
            tags=[],
            parameters=[{"name": "id", "in": "path", "required": True}],
        )
        spec = endpoint.to_openapi_spec()
        assert "parameters" in spec
        assert len(spec["parameters"]) == 1
        assert spec["parameters"][0]["name"] == "id"

    def test_includes_request_body(self):
        """Should include requestBody when provided."""
        request_body = {
            "content": {"application/json": {"schema": {"type": "object"}}},
            "required": True,
        }
        endpoint = OpenAPIEndpoint(
            path="/api/users",
            method="POST",
            summary="Create user",
            tags=[],
            request_body=request_body,
        )
        spec = endpoint.to_openapi_spec()
        assert "requestBody" in spec
        assert spec["requestBody"]["required"] is True

    def test_includes_custom_responses(self):
        """Should include custom responses instead of default."""
        endpoint = OpenAPIEndpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=[],
            responses={
                "200": {"description": "Custom success"},
                "404": {"description": "Not found"},
            },
        )
        spec = endpoint.to_openapi_spec()
        assert spec["responses"]["200"]["description"] == "Custom success"
        assert "404" in spec["responses"]

    def test_default_200_response(self):
        """Should add default 200 response when none provided."""
        endpoint = OpenAPIEndpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=[],
        )
        spec = endpoint.to_openapi_spec()
        assert "200" in spec["responses"]
        assert spec["responses"]["200"]["description"] == "Success"
        assert "application/json" in spec["responses"]["200"]["content"]

    def test_includes_security(self):
        """Should include security requirements when provided."""
        endpoint = OpenAPIEndpoint(
            path="/api/secure",
            method="GET",
            summary="Secure endpoint",
            tags=[],
            security=[{"bearerAuth": []}],
        )
        spec = endpoint.to_openapi_spec()
        assert "security" in spec
        assert spec["security"] == [{"bearerAuth": []}]

    def test_includes_deprecated(self):
        """Should include deprecated flag when true."""
        endpoint = OpenAPIEndpoint(
            path="/api/old",
            method="GET",
            summary="Old endpoint",
            tags=[],
            deprecated=True,
        )
        spec = endpoint.to_openapi_spec()
        assert spec["deprecated"] is True

    def test_excludes_deprecated_when_false(self):
        """Should not include deprecated when false."""
        endpoint = OpenAPIEndpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=[],
            deprecated=False,
        )
        spec = endpoint.to_openapi_spec()
        assert "deprecated" not in spec


# =============================================================================
# TestApiEndpointDecorator
# =============================================================================


class TestApiEndpointDecorator:
    """Tests for @api_endpoint decorator."""

    def test_basic_decoration(self):
        """Should decorate function and register endpoint."""

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

        # Endpoint should be registered
        endpoints = get_registered_endpoints()
        assert len(endpoints) == 1
        assert endpoints[0].path == "/api/test"

    def test_attaches_openapi_attribute(self):
        """Should attach _openapi attribute to decorated function."""

        @api_endpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=[],
        )
        def handler():
            pass

        assert hasattr(handler, "_openapi")
        assert isinstance(handler._openapi, OpenAPIEndpoint)
        assert handler._openapi.path == "/api/test"

    def test_preserves_function_metadata(self):
        """Should preserve original function name and docstring."""

        @api_endpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=[],
        )
        def my_handler_function():
            """This is my docstring."""
            pass

        assert my_handler_function.__name__ == "my_handler_function"
        assert "my docstring" in my_handler_function.__doc__

    def test_uses_function_name_as_operation_id(self):
        """Should use function name as default operation_id."""

        @api_endpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=[],
        )
        def get_test_resource():
            pass

        endpoint = get_test_resource._openapi
        assert endpoint.operation_id == "get_test_resource"

    def test_custom_operation_id(self):
        """Should use custom operation_id when provided."""

        @api_endpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=[],
            operation_id="customOperationId",
        )
        def handler():
            pass

        endpoint = handler._openapi
        assert endpoint.operation_id == "customOperationId"

    def test_uses_docstring_as_description(self):
        """Should use function docstring as description."""

        @api_endpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=[],
        )
        def handler():
            """This is the detailed description."""
            pass

        endpoint = handler._openapi
        assert endpoint.description == "This is the detailed description."

    def test_explicit_description_overrides_docstring(self):
        """Should use explicit description over docstring."""

        @api_endpoint(
            path="/api/test",
            method="GET",
            summary="Test",
            tags=[],
            description="Explicit description",
        )
        def handler():
            """Docstring description."""
            pass

        endpoint = handler._openapi
        assert endpoint.description == "Explicit description"

    def test_method_uppercase(self):
        """Should convert method to uppercase."""

        @api_endpoint(
            path="/api/test",
            method="post",
            summary="Test",
            tags=[],
        )
        def handler():
            pass

        endpoint = handler._openapi
        assert endpoint.method == "POST"

    def test_auth_required_adds_security(self):
        """Should add bearerAuth security when auth_required=True."""

        @api_endpoint(
            path="/api/secure",
            method="GET",
            summary="Secure",
            tags=[],
            auth_required=True,
        )
        def handler():
            pass

        endpoint = handler._openapi
        assert endpoint.security == [{"bearerAuth": []}]

    def test_auth_not_required_empty_security(self):
        """Should have empty security when auth_required=False."""

        @api_endpoint(
            path="/api/public",
            method="GET",
            summary="Public",
            tags=[],
            auth_required=False,
        )
        def handler():
            pass

        endpoint = handler._openapi
        assert endpoint.security == []

    def test_default_summary_from_function_name(self):
        """Should generate summary from function name if not provided."""

        @api_endpoint(
            path="/api/test",
            method="GET",
            tags=[],
        )
        def get_user_profile():
            pass

        endpoint = get_user_profile._openapi
        assert endpoint.summary == "Get User Profile"

    def test_deprecated_flag(self):
        """Should set deprecated flag."""

        @api_endpoint(
            path="/api/old",
            method="GET",
            summary="Old",
            tags=[],
            deprecated=True,
        )
        def handler():
            pass

        endpoint = handler._openapi
        assert endpoint.deprecated is True

    def test_parameters_passed_through(self):
        """Should pass parameters to endpoint."""
        params = [
            {"name": "id", "in": "path", "required": True},
            {"name": "limit", "in": "query", "schema": {"type": "integer"}},
        ]

        @api_endpoint(
            path="/api/users/{id}",
            method="GET",
            summary="Get user",
            tags=[],
            parameters=params,
        )
        def handler():
            pass

        endpoint = handler._openapi
        assert len(endpoint.parameters) == 2

    def test_request_body_passed_through(self):
        """Should pass request_body to endpoint."""
        body = {"content": {"application/json": {"schema": {"type": "object"}}}}

        @api_endpoint(
            path="/api/users",
            method="POST",
            summary="Create user",
            tags=[],
            request_body=body,
        )
        def handler():
            pass

        endpoint = handler._openapi
        assert endpoint.request_body is not None

    def test_responses_passed_through(self):
        """Should pass responses to endpoint."""
        responses = {
            "200": {"description": "Success"},
            "400": {"description": "Bad request"},
        }

        @api_endpoint(
            path="/api/test",
            method="POST",
            summary="Test",
            tags=[],
            responses=responses,
        )
        def handler():
            pass

        endpoint = handler._openapi
        assert "200" in endpoint.responses
        assert "400" in endpoint.responses

    def test_async_function_support(self):
        """Should work with async functions."""

        @api_endpoint(
            path="/api/async",
            method="GET",
            summary="Async endpoint",
            tags=[],
        )
        async def async_handler():
            return "async result"

        endpoints = get_registered_endpoints()
        assert len(endpoints) == 1
        assert endpoints[0].path == "/api/async"


# =============================================================================
# TestPydanticModelSupport
# =============================================================================


class TestPydanticModelSupport:
    """Tests for Pydantic model schema extraction."""

    def test_is_pydantic_model_with_basemodel(self):
        """Should identify Pydantic BaseModel."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            name: str

        assert _is_pydantic_model(MyModel) is True

    def test_is_pydantic_model_with_non_model(self):
        """Should return False for non-Pydantic types."""
        assert _is_pydantic_model(str) is False
        assert _is_pydantic_model(dict) is False
        assert _is_pydantic_model(object) is False
        assert _is_pydantic_model(None) is False

    def test_is_pydantic_model_with_instance(self):
        """Should return False for model instances (not classes)."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            name: str

        instance = MyModel(name="test")
        assert _is_pydantic_model(instance) is False

    def test_extract_pydantic_schema_v2(self):
        """Should extract schema from Pydantic v2 model."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            count: int

        schema = _extract_pydantic_schema(TestModel)
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "count" in schema["properties"]

    def test_extract_pydantic_schema_fallback(self):
        """Should return fallback schema for non-Pydantic types."""

        class NotAModel:
            pass

        schema = _extract_pydantic_schema(NotAModel)
        assert schema == {"type": "object"}

    def test_request_model_auto_generates_body(self):
        """Should auto-generate request body from request_model."""
        from pydantic import BaseModel

        class CreateUserRequest(BaseModel):
            name: str
            email: str

        @api_endpoint(
            path="/api/users",
            method="POST",
            summary="Create user",
            tags=[],
            request_model=CreateUserRequest,
        )
        def handler():
            pass

        endpoint = handler._openapi
        assert endpoint.request_body is not None
        assert "CreateUserRequest" in endpoint.request_body["description"]
        content = endpoint.request_body["content"]["application/json"]
        assert "schema" in content
        assert "properties" in content["schema"]

    def test_response_model_auto_generates_response(self):
        """Should auto-generate 200 response from response_model."""
        from pydantic import BaseModel

        class UserResponse(BaseModel):
            id: str
            name: str

        @api_endpoint(
            path="/api/users/{id}",
            method="GET",
            summary="Get user",
            tags=[],
            response_model=UserResponse,
        )
        def handler():
            pass

        endpoint = handler._openapi
        assert "200" in endpoint.responses
        content = endpoint.responses["200"]["content"]["application/json"]
        assert "schema" in content
        assert "properties" in content["schema"]

    def test_explicit_body_overrides_request_model(self):
        """Should use explicit request_body over request_model."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            field: str

        explicit_body = {"description": "Explicit body", "required": True, "content": {}}

        @api_endpoint(
            path="/api/test",
            method="POST",
            summary="Test",
            tags=[],
            request_body=explicit_body,
            request_model=MyModel,
        )
        def handler():
            pass

        endpoint = handler._openapi
        assert endpoint.request_body["description"] == "Explicit body"

    def test_explicit_responses_overrides_response_model(self):
        """Should use explicit responses over response_model."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            field: str

        explicit_responses = {"201": {"description": "Created"}}

        @api_endpoint(
            path="/api/test",
            method="POST",
            summary="Test",
            tags=[],
            responses=explicit_responses,
            response_model=MyModel,
        )
        def handler():
            pass

        endpoint = handler._openapi
        assert "201" in endpoint.responses
        assert "200" not in endpoint.responses


# =============================================================================
# TestRegistryManagement
# =============================================================================


class TestRegistryManagement:
    """Tests for endpoint registry functions."""

    def test_get_registered_endpoints_empty(self):
        """Should return empty list when no endpoints registered."""
        endpoints = get_registered_endpoints()
        assert endpoints == []

    def test_get_registered_endpoints_returns_copy(self):
        """Should return a copy of the registry."""

        @api_endpoint(path="/api/test", method="GET", summary="Test", tags=[])
        def handler():
            pass

        endpoints1 = get_registered_endpoints()
        endpoints2 = get_registered_endpoints()
        assert endpoints1 == endpoints2
        assert endpoints1 is not endpoints2

    def test_clear_registry(self):
        """Should clear all registered endpoints."""

        @api_endpoint(path="/api/test1", method="GET", summary="Test 1", tags=[])
        def handler1():
            pass

        @api_endpoint(path="/api/test2", method="GET", summary="Test 2", tags=[])
        def handler2():
            pass

        assert len(get_registered_endpoints()) == 2
        clear_registry()
        assert len(get_registered_endpoints()) == 0

    def test_register_endpoint_manually(self):
        """Should allow manual endpoint registration."""
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

    def test_get_registered_endpoints_dict(self):
        """Should return endpoints as OpenAPI paths dict."""

        @api_endpoint(path="/api/users", method="GET", summary="List users", tags=["Users"])
        def list_users():
            pass

        @api_endpoint(path="/api/users", method="POST", summary="Create user", tags=["Users"])
        def create_user():
            pass

        @api_endpoint(path="/api/debates", method="GET", summary="List debates", tags=["Debates"])
        def list_debates():
            pass

        paths = get_registered_endpoints_dict()

        assert "/api/users" in paths
        assert "/api/debates" in paths
        assert "get" in paths["/api/users"]
        assert "post" in paths["/api/users"]
        assert paths["/api/users"]["get"]["summary"] == "List users"

    def test_multiple_methods_same_path(self):
        """Should handle multiple methods on same path."""

        @api_endpoint(path="/api/resource", method="GET", summary="Get", tags=[])
        def get_resource():
            pass

        @api_endpoint(path="/api/resource", method="PUT", summary="Update", tags=[])
        def update_resource():
            pass

        @api_endpoint(path="/api/resource", method="DELETE", summary="Delete", tags=[])
        def delete_resource():
            pass

        paths = get_registered_endpoints_dict()
        assert len(paths) == 1
        assert "get" in paths["/api/resource"]
        assert "put" in paths["/api/resource"]
        assert "delete" in paths["/api/resource"]


# =============================================================================
# TestPathParamHelper
# =============================================================================


class TestPathParamHelper:
    """Tests for path_param helper function."""

    def test_basic_path_param(self):
        """Should create basic path parameter."""
        param = path_param("id")
        assert param["name"] == "id"
        assert param["in"] == "path"
        assert param["required"] is True
        assert param["schema"]["type"] == "string"

    def test_path_param_with_description(self):
        """Should include description."""
        param = path_param("user_id", description="The user identifier")
        assert param["description"] == "The user identifier"

    def test_path_param_with_integer_type(self):
        """Should support integer schema type."""
        param = path_param("id", schema_type="integer")
        assert param["schema"]["type"] == "integer"

    def test_path_param_always_required(self):
        """Path params should always be required."""
        param = path_param("id")
        assert param["required"] is True


# =============================================================================
# TestQueryParamHelper
# =============================================================================


class TestQueryParamHelper:
    """Tests for query_param helper function."""

    def test_basic_query_param(self):
        """Should create basic query parameter."""
        param = query_param("search")
        assert param["name"] == "search"
        assert param["in"] == "query"
        assert param["schema"]["type"] == "string"

    def test_query_param_not_required_by_default(self):
        """Should not be required by default."""
        param = query_param("filter")
        assert "required" not in param

    def test_query_param_required(self):
        """Should support required flag."""
        param = query_param("filter", required=True)
        assert param["required"] is True

    def test_query_param_with_description(self):
        """Should include description."""
        param = query_param("limit", description="Maximum number of results")
        assert param["description"] == "Maximum number of results"

    def test_query_param_with_default(self):
        """Should include default value in schema."""
        param = query_param("page", default=1)
        assert param["schema"]["default"] == 1

    def test_query_param_with_enum(self):
        """Should include enum values in schema."""
        param = query_param("status", enum=["active", "inactive", "pending"])
        assert param["schema"]["enum"] == ["active", "inactive", "pending"]

    def test_query_param_with_integer_type(self):
        """Should support integer type."""
        param = query_param("limit", schema_type="integer", default=10)
        assert param["schema"]["type"] == "integer"
        assert param["schema"]["default"] == 10


# =============================================================================
# TestJsonBodyHelper
# =============================================================================


class TestJsonBodyHelper:
    """Tests for json_body helper function."""

    def test_basic_json_body(self):
        """Should create JSON request body."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        body = json_body(schema)
        assert body["required"] is True
        assert "application/json" in body["content"]
        assert body["content"]["application/json"]["schema"] == schema

    def test_json_body_with_description(self):
        """Should include description."""
        schema = {"type": "object"}
        body = json_body(schema, description="User data")
        assert body["description"] == "User data"

    def test_json_body_not_required(self):
        """Should support optional body."""
        schema = {"type": "object"}
        body = json_body(schema, required=False)
        assert body["required"] is False

    def test_json_body_with_pydantic_model(self):
        """Should extract schema from Pydantic model."""
        from pydantic import BaseModel

        class CreateRequest(BaseModel):
            name: str
            value: int

        body = json_body(CreateRequest)
        schema = body["content"]["application/json"]["schema"]
        assert "properties" in schema
        assert "name" in schema["properties"]

    def test_json_body_pydantic_auto_description(self):
        """Should auto-generate description from Pydantic model name."""
        from pydantic import BaseModel

        class MyRequestModel(BaseModel):
            field: str

        body = json_body(MyRequestModel)
        assert "MyRequestModel" in body["description"]


# =============================================================================
# TestOkResponseHelper
# =============================================================================


class TestOkResponseHelper:
    """Tests for ok_response helper function."""

    def test_basic_ok_response(self):
        """Should create basic OK response."""
        resp = ok_response("Success")
        assert "200" in resp
        assert resp["200"]["description"] == "Success"

    def test_ok_response_default_schema(self):
        """Should have default object schema."""
        resp = ok_response("Success")
        schema = resp["200"]["content"]["application/json"]["schema"]
        assert schema["type"] == "object"

    def test_ok_response_with_schema(self):
        """Should include custom schema."""
        custom_schema = {"type": "array", "items": {"type": "string"}}
        resp = ok_response("List of strings", custom_schema)
        schema = resp["200"]["content"]["application/json"]["schema"]
        assert schema["type"] == "array"

    def test_ok_response_custom_status_code(self):
        """Should support custom status code."""
        resp = ok_response("Created", status_code="201")
        assert "201" in resp
        assert "200" not in resp

    def test_ok_response_with_pydantic_model(self):
        """Should extract schema from Pydantic model."""
        from pydantic import BaseModel

        class UserResponse(BaseModel):
            id: str
            name: str

        resp = ok_response("User details", UserResponse)
        schema = resp["200"]["content"]["application/json"]["schema"]
        assert "properties" in schema
        assert "id" in schema["properties"]


# =============================================================================
# TestErrorResponseHelper
# =============================================================================


class TestErrorResponseHelper:
    """Tests for error_response helper function."""

    def test_basic_error_response(self):
        """Should create error response."""
        resp = error_response("404", "Not found")
        assert "404" in resp
        assert resp["404"]["description"] == "Not found"

    def test_error_response_schema(self):
        """Should have error schema structure."""
        resp = error_response("400", "Bad request")
        schema = resp["400"]["content"]["application/json"]["schema"]
        assert "error" in schema["properties"]
        assert "details" in schema["properties"]

    def test_multiple_error_responses(self):
        """Should be combinable for multiple errors."""
        responses = {}
        responses.update(error_response("400", "Bad request"))
        responses.update(error_response("401", "Unauthorized"))
        responses.update(error_response("404", "Not found"))

        assert "400" in responses
        assert "401" in responses
        assert "404" in responses


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_tags(self):
        """Should handle empty tags list."""

        @api_endpoint(path="/api/test", method="GET", summary="Test", tags=[])
        def handler():
            pass

        endpoint = handler._openapi
        assert endpoint.tags == []

    def test_none_tags(self):
        """Should handle None tags."""

        @api_endpoint(path="/api/test", method="GET", summary="Test", tags=None)
        def handler():
            pass

        endpoint = handler._openapi
        assert endpoint.tags == []

    def test_empty_parameters(self):
        """Should handle empty parameters list."""

        @api_endpoint(path="/api/test", method="GET", summary="Test", tags=[], parameters=[])
        def handler():
            pass

        endpoint = handler._openapi
        assert endpoint.parameters == []

    def test_none_parameters(self):
        """Should handle None parameters."""

        @api_endpoint(path="/api/test", method="GET", summary="Test", tags=[], parameters=None)
        def handler():
            pass

        endpoint = handler._openapi
        assert endpoint.parameters == []

    def test_function_with_no_docstring(self):
        """Should handle function without docstring."""

        @api_endpoint(path="/api/test", method="GET", summary="Test", tags=[])
        def handler_no_doc():
            pass

        endpoint = handler_no_doc._openapi
        assert endpoint.description == ""

    def test_special_characters_in_path(self):
        """Should handle special characters in path."""

        @api_endpoint(
            path="/api/users/{user_id}/posts/{post_id}", method="GET", summary="Get post", tags=[]
        )
        def handler():
            pass

        endpoint = handler._openapi
        assert endpoint.path == "/api/users/{user_id}/posts/{post_id}"

    def test_method_case_insensitivity(self):
        """Should handle various method casings."""
        for method in ["get", "GET", "Get", "GeT"]:

            @api_endpoint(path=f"/api/test-{method}", method=method, summary="Test", tags=[])
            def handler():
                pass

            endpoint = handler._openapi
            assert endpoint.method == "GET"

    def test_all_http_methods(self):
        """Should support all standard HTTP methods."""
        methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

        for method in methods:

            @api_endpoint(
                path=f"/api/test-{method.lower()}",
                method=method,
                summary=f"{method} endpoint",
                tags=[],
            )
            def handler():
                pass

            endpoint = handler._openapi
            assert endpoint.method == method

    def test_decorator_with_other_decorators(self):
        """Should work alongside other decorators."""

        def other_decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            wrapper.__name__ = func.__name__
            return wrapper

        @other_decorator
        @api_endpoint(path="/api/test", method="GET", summary="Test", tags=[])
        def handler():
            return "result"

        # Should still work
        assert handler() == "result"

    def test_class_method_decoration(self):
        """Should work with class methods."""

        class MyHandler:
            @api_endpoint(path="/api/test", method="GET", summary="Test", tags=[])
            def handle(self):
                return "result"

        handler = MyHandler()
        assert handler.handle() == "result"
        assert hasattr(MyHandler.handle, "_openapi")


# =============================================================================
# TestSchemaExtractionEdgeCases
# =============================================================================


class TestSchemaExtractionEdgeCases:
    """Tests for schema extraction edge cases."""

    def test_extract_schema_exception_handling(self):
        """Should handle exceptions during schema extraction."""
        # Create a mock that raises an exception
        mock_model = MagicMock()
        mock_model.model_json_schema.side_effect = Exception("Schema error")

        with patch(
            "aragora.server.handlers.openapi_decorator._is_pydantic_model",
            return_value=True,
        ):
            schema = _extract_pydantic_schema(mock_model)
            assert schema == {"type": "object"}

    def test_pydantic_import_error(self):
        """Should handle Pydantic import error gracefully."""
        with patch.dict("sys.modules", {"pydantic": None}):
            # When pydantic is not importable
            result = _is_pydantic_model(dict)
            assert result is False


__all__ = [
    "TestOpenAPIEndpoint",
    "TestOpenAPIEndpointToSpec",
    "TestApiEndpointDecorator",
    "TestPydanticModelSupport",
    "TestRegistryManagement",
    "TestPathParamHelper",
    "TestQueryParamHelper",
    "TestJsonBodyHelper",
    "TestOkResponseHelper",
    "TestErrorResponseHelper",
    "TestEdgeCases",
    "TestSchemaExtractionEdgeCases",
]
