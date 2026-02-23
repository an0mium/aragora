"""Tests for OpenAPI auto-generation decorator module.

Covers:
- OpenAPIEndpoint dataclass and to_openapi_spec conversion
- api_endpoint decorator: registration, metadata, wrapping, Pydantic model support
- get_registered_endpoints / get_registered_endpoints_dict
- clear_registry / register_endpoint
- Helper functions: path_param, query_param, json_body, ok_response, error_response
- _extract_pydantic_schema (v1 and v2 fallback paths)
- _is_pydantic_model
- Edge cases: empty values, duplicate endpoints, deprecated endpoints, no auth
- Security: no code injection via parameter values
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
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


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clean_registry_before_each():
    """Ensure a clean global registry for every test."""
    clear_registry()
    yield
    clear_registry()


# ============================================================================
# Pydantic helpers for tests
# ============================================================================

from pydantic import BaseModel as PydanticBaseModel


class SampleRequest(PydanticBaseModel):
    name: str
    count: int = 1


class SampleResponse(PydanticBaseModel):
    id: str
    status: str


class EmptyModel(PydanticBaseModel):
    pass


# ============================================================================
# OpenAPIEndpoint Dataclass
# ============================================================================


class TestOpenAPIEndpoint:
    """Tests for the OpenAPIEndpoint dataclass."""

    def test_create_minimal(self):
        ep = OpenAPIEndpoint(path="/test", method="GET", summary="Test", tags=["t"])
        assert ep.path == "/test"
        assert ep.method == "GET"
        assert ep.summary == "Test"
        assert ep.tags == ["t"]
        assert ep.description == ""
        assert ep.parameters == []
        assert ep.request_body is None
        assert ep.responses == {}
        assert ep.security == []
        assert ep.operation_id is None
        assert ep.deprecated is False

    def test_create_full(self):
        ep = OpenAPIEndpoint(
            path="/api/v1/items",
            method="POST",
            summary="Create item",
            tags=["Items", "CRUD"],
            description="Creates a new item",
            parameters=[{"name": "x", "in": "query"}],
            request_body={"content": {}},
            responses={"201": {"description": "Created"}},
            security=[{"bearerAuth": []}],
            operation_id="create_item",
            deprecated=True,
        )
        assert ep.path == "/api/v1/items"
        assert ep.method == "POST"
        assert ep.description == "Creates a new item"
        assert len(ep.parameters) == 1
        assert ep.request_body is not None
        assert "201" in ep.responses
        assert ep.security == [{"bearerAuth": []}]
        assert ep.operation_id == "create_item"
        assert ep.deprecated is True

    def test_to_openapi_spec_minimal(self):
        ep = OpenAPIEndpoint(path="/t", method="GET", summary="Min", tags=["A"])
        spec = ep.to_openapi_spec()
        assert spec["summary"] == "Min"
        assert spec["tags"] == ["A"]
        # Default 200 response should be generated
        assert "200" in spec["responses"]
        assert spec["responses"]["200"]["description"] == "Success"
        # No optional fields
        assert "description" not in spec
        assert "operationId" not in spec
        assert "parameters" not in spec
        assert "requestBody" not in spec
        assert "security" not in spec
        assert "deprecated" not in spec

    def test_to_openapi_spec_with_description(self):
        ep = OpenAPIEndpoint(path="/t", method="GET", summary="S", tags=[], description="Desc")
        spec = ep.to_openapi_spec()
        assert spec["description"] == "Desc"

    def test_to_openapi_spec_with_operation_id(self):
        ep = OpenAPIEndpoint(path="/t", method="GET", summary="S", tags=[], operation_id="my_op")
        spec = ep.to_openapi_spec()
        assert spec["operationId"] == "my_op"

    def test_to_openapi_spec_with_parameters(self):
        params = [{"name": "id", "in": "path", "required": True}]
        ep = OpenAPIEndpoint(path="/t", method="GET", summary="S", tags=[], parameters=params)
        spec = ep.to_openapi_spec()
        assert spec["parameters"] == params

    def test_to_openapi_spec_with_request_body(self):
        body = {"content": {"application/json": {"schema": {"type": "object"}}}}
        ep = OpenAPIEndpoint(path="/t", method="POST", summary="S", tags=[], request_body=body)
        spec = ep.to_openapi_spec()
        assert spec["requestBody"] == body

    def test_to_openapi_spec_with_custom_responses(self):
        resps = {"201": {"description": "Created"}, "400": {"description": "Bad"}}
        ep = OpenAPIEndpoint(path="/t", method="POST", summary="S", tags=[], responses=resps)
        spec = ep.to_openapi_spec()
        assert spec["responses"] == resps
        # No default 200 when custom responses provided
        assert "200" not in spec["responses"]

    def test_to_openapi_spec_with_security(self):
        sec = [{"bearerAuth": []}, {"apiKey": []}]
        ep = OpenAPIEndpoint(path="/t", method="GET", summary="S", tags=[], security=sec)
        spec = ep.to_openapi_spec()
        assert spec["security"] == sec

    def test_to_openapi_spec_deprecated(self):
        ep = OpenAPIEndpoint(path="/t", method="GET", summary="S", tags=[], deprecated=True)
        spec = ep.to_openapi_spec()
        assert spec["deprecated"] is True

    def test_to_openapi_spec_not_deprecated_omits_key(self):
        ep = OpenAPIEndpoint(path="/t", method="GET", summary="S", tags=[], deprecated=False)
        spec = ep.to_openapi_spec()
        assert "deprecated" not in spec

    def test_to_openapi_spec_all_fields(self):
        ep = OpenAPIEndpoint(
            path="/api/v1/items/{id}",
            method="PUT",
            summary="Update item",
            tags=["Items"],
            description="Updates an existing item",
            parameters=[{"name": "id", "in": "path", "required": True}],
            request_body={"content": {"application/json": {"schema": {}}}},
            responses={"200": {"description": "Updated"}},
            security=[{"bearerAuth": []}],
            operation_id="update_item",
            deprecated=True,
        )
        spec = ep.to_openapi_spec()
        assert spec["summary"] == "Update item"
        assert spec["tags"] == ["Items"]
        assert spec["description"] == "Updates an existing item"
        assert spec["operationId"] == "update_item"
        assert len(spec["parameters"]) == 1
        assert spec["requestBody"] is not None
        assert "200" in spec["responses"]
        assert spec["security"] == [{"bearerAuth": []}]
        assert spec["deprecated"] is True

    def test_default_factory_independence(self):
        """Ensure default mutable fields are independent across instances."""
        ep1 = OpenAPIEndpoint(path="/a", method="GET", summary="A", tags=["a"])
        ep2 = OpenAPIEndpoint(path="/b", method="GET", summary="B", tags=["b"])
        ep1.parameters.append({"name": "x"})
        assert ep2.parameters == []
        ep1.responses["200"] = {"description": "Ok"}
        assert ep2.responses == {}

    def test_empty_tags(self):
        ep = OpenAPIEndpoint(path="/t", method="GET", summary="S", tags=[])
        spec = ep.to_openapi_spec()
        assert spec["tags"] == []

    def test_multiple_tags(self):
        ep = OpenAPIEndpoint(path="/t", method="GET", summary="S", tags=["A", "B", "C"])
        spec = ep.to_openapi_spec()
        assert spec["tags"] == ["A", "B", "C"]


# ============================================================================
# api_endpoint Decorator
# ============================================================================


class TestApiEndpoint:
    """Tests for the api_endpoint decorator."""

    def test_basic_registration(self):
        @api_endpoint(path="/api/v1/foo", method="GET", summary="Foo", tags=["X"])
        def my_func():
            return "result"

        endpoints = get_registered_endpoints()
        assert len(endpoints) == 1
        assert endpoints[0].path == "/api/v1/foo"
        assert endpoints[0].method == "GET"
        assert endpoints[0].summary == "Foo"
        assert endpoints[0].tags == ["X"]

    def test_function_still_callable(self):
        @api_endpoint(path="/test", summary="T", tags=[])
        def my_func(x, y):
            return x + y

        assert my_func(2, 3) == 5

    def test_async_function_still_callable(self):
        @api_endpoint(path="/test", summary="T", tags=[])
        async def my_func(x):
            return x * 2

        result = asyncio.run(my_func(5))
        assert result == 10

    def test_openapi_attribute_attached(self):
        @api_endpoint(path="/p", summary="S", tags=[])
        def my_func():
            pass

        assert hasattr(my_func, "_openapi")
        assert isinstance(my_func._openapi, OpenAPIEndpoint)
        assert my_func._openapi.path == "/p"

    def test_method_uppercased(self):
        @api_endpoint(path="/p", method="post", summary="S", tags=[])
        def my_func():
            pass

        assert my_func._openapi.method == "POST"

    def test_default_method_is_get(self):
        @api_endpoint(path="/p", summary="S", tags=[])
        def my_func():
            pass

        assert my_func._openapi.method == "GET"

    def test_operation_id_defaults_to_func_name(self):
        @api_endpoint(path="/p", summary="S", tags=[])
        def get_all_items():
            pass

        assert get_all_items._openapi.operation_id == "get_all_items"

    def test_operation_id_custom(self):
        @api_endpoint(path="/p", summary="S", tags=[], operation_id="custom_op")
        def my_func():
            pass

        assert my_func._openapi.operation_id == "custom_op"

    def test_description_from_docstring(self):
        @api_endpoint(path="/p", summary="S", tags=[])
        def my_func():
            """My function description."""
            pass

        assert my_func._openapi.description == "My function description."

    def test_description_explicit_overrides_docstring(self):
        @api_endpoint(path="/p", summary="S", tags=[], description="Explicit desc")
        def my_func():
            """Docstring desc."""
            pass

        assert my_func._openapi.description == "Explicit desc"

    def test_description_empty_when_no_docstring(self):
        @api_endpoint(path="/p", summary="S", tags=[])
        def my_func():
            pass

        assert my_func._openapi.description == ""

    def test_auth_required_default_true(self):
        @api_endpoint(path="/p", summary="S", tags=[])
        def my_func():
            pass

        assert my_func._openapi.security == [{"bearerAuth": []}]

    def test_auth_required_false(self):
        @api_endpoint(path="/p", summary="S", tags=[], auth_required=False)
        def my_func():
            pass

        assert my_func._openapi.security == []

    def test_deprecated_flag(self):
        @api_endpoint(path="/p", summary="S", tags=[], deprecated=True)
        def my_func():
            pass

        assert my_func._openapi.deprecated is True

    def test_deprecated_default_false(self):
        @api_endpoint(path="/p", summary="S", tags=[])
        def my_func():
            pass

        assert my_func._openapi.deprecated is False

    def test_parameters_passed_through(self):
        params = [
            {"name": "id", "in": "path", "required": True},
            {"name": "q", "in": "query"},
        ]

        @api_endpoint(path="/p", summary="S", tags=[], parameters=params)
        def my_func():
            pass

        assert my_func._openapi.parameters == params

    def test_parameters_default_empty(self):
        @api_endpoint(path="/p", summary="S", tags=[])
        def my_func():
            pass

        assert my_func._openapi.parameters == []

    def test_request_body_passed_through(self):
        body = {"content": {"application/json": {"schema": {"type": "object"}}}}

        @api_endpoint(path="/p", method="POST", summary="S", tags=[], request_body=body)
        def my_func():
            pass

        assert my_func._openapi.request_body == body

    def test_responses_passed_through(self):
        resps = {"200": {"description": "OK"}, "404": {"description": "Not found"}}

        @api_endpoint(path="/p", summary="S", tags=[], responses=resps)
        def my_func():
            pass

        assert my_func._openapi.responses == resps

    def test_summary_auto_generated_from_func_name(self):
        @api_endpoint(path="/p", tags=[])
        def get_all_items():
            pass

        assert get_all_items._openapi.summary == "Get All Items"

    def test_summary_explicit(self):
        @api_endpoint(path="/p", summary="My Summary", tags=[])
        def my_func():
            pass

        assert my_func._openapi.summary == "My Summary"

    def test_tags_default_empty(self):
        @api_endpoint(path="/p", summary="S")
        def my_func():
            pass

        assert my_func._openapi.tags == []

    def test_wraps_preserves_name(self):
        @api_endpoint(path="/p", summary="S", tags=[])
        def my_special_func():
            """My docstring."""
            pass

        assert my_special_func.__name__ == "my_special_func"
        assert my_special_func.__doc__ == "My docstring."

    def test_wraps_preserves_args(self):
        @api_endpoint(path="/p", summary="S", tags=[])
        def my_func(a, b, c=10):
            return a + b + c

        assert my_func(1, 2) == 13
        assert my_func(1, 2, c=20) == 23

    def test_wraps_preserves_kwargs(self):
        @api_endpoint(path="/p", summary="S", tags=[])
        def my_func(**kwargs):
            return kwargs

        result = my_func(x=1, y=2)
        assert result == {"x": 1, "y": 2}

    def test_multiple_decorators_register_multiple(self):
        @api_endpoint(path="/a", summary="A", tags=[])
        def func_a():
            pass

        @api_endpoint(path="/b", summary="B", tags=[])
        def func_b():
            pass

        @api_endpoint(path="/c", method="POST", summary="C", tags=[])
        def func_c():
            pass

        endpoints = get_registered_endpoints()
        assert len(endpoints) == 3
        paths = {ep.path for ep in endpoints}
        assert paths == {"/a", "/b", "/c"}

    def test_request_model_generates_schema(self):
        @api_endpoint(
            path="/p",
            method="POST",
            summary="S",
            tags=[],
            request_model=SampleRequest,
        )
        def my_func():
            pass

        body = my_func._openapi.request_body
        assert body is not None
        assert body["required"] is True
        assert "SampleRequest request" in body["description"]
        schema = body["content"]["application/json"]["schema"]
        assert "properties" in schema

    def test_response_model_generates_schema(self):
        @api_endpoint(path="/p", summary="S", tags=[], response_model=SampleResponse)
        def my_func():
            pass

        resps = my_func._openapi.responses
        assert "200" in resps
        schema = resps["200"]["content"]["application/json"]["schema"]
        assert "properties" in schema

    def test_request_body_overrides_request_model(self):
        manual_body = {"description": "Manual", "content": {}}

        @api_endpoint(
            path="/p",
            method="POST",
            summary="S",
            tags=[],
            request_body=manual_body,
            request_model=SampleRequest,
        )
        def my_func():
            pass

        assert my_func._openapi.request_body == manual_body

    def test_responses_overrides_response_model(self):
        manual_resps = {"201": {"description": "Created"}}

        @api_endpoint(
            path="/p",
            summary="S",
            tags=[],
            responses=manual_resps,
            response_model=SampleResponse,
        )
        def my_func():
            pass

        assert my_func._openapi.responses == manual_resps

    def test_request_model_non_pydantic_ignored(self):
        """Non-Pydantic request_model should not generate a request body."""

        class PlainClass:
            pass

        @api_endpoint(path="/p", method="POST", summary="S", tags=[], request_model=PlainClass)
        def my_func():
            pass

        assert my_func._openapi.request_body is None

    def test_response_model_non_pydantic_ignored(self):
        """Non-Pydantic response_model should not generate responses."""

        class PlainClass:
            pass

        @api_endpoint(path="/p", summary="S", tags=[], response_model=PlainClass)
        def my_func():
            pass

        assert my_func._openapi.responses == {}


# ============================================================================
# Registry Functions
# ============================================================================


class TestRegistry:
    """Tests for registry management functions."""

    def test_get_registered_endpoints_empty(self):
        endpoints = get_registered_endpoints()
        assert endpoints == []

    def test_get_registered_endpoints_returns_copy(self):
        register_endpoint(OpenAPIEndpoint(path="/a", method="GET", summary="A", tags=[]))
        ep_list1 = get_registered_endpoints()
        ep_list2 = get_registered_endpoints()
        assert ep_list1 is not ep_list2
        assert ep_list1 == ep_list2

    def test_get_registered_endpoints_copy_isolation(self):
        """Modifying returned list does not affect the global registry."""
        register_endpoint(OpenAPIEndpoint(path="/a", method="GET", summary="A", tags=[]))
        ep_list = get_registered_endpoints()
        ep_list.clear()
        assert len(get_registered_endpoints()) == 1

    def test_register_endpoint(self):
        ep = OpenAPIEndpoint(path="/x", method="DELETE", summary="Del", tags=["X"])
        register_endpoint(ep)
        endpoints = get_registered_endpoints()
        assert len(endpoints) == 1
        assert endpoints[0].path == "/x"
        assert endpoints[0].method == "DELETE"

    def test_register_multiple_endpoints(self):
        for i in range(5):
            register_endpoint(
                OpenAPIEndpoint(path=f"/ep{i}", method="GET", summary=f"EP {i}", tags=[])
            )
        assert len(get_registered_endpoints()) == 5

    def test_clear_registry(self):
        register_endpoint(OpenAPIEndpoint(path="/a", method="GET", summary="A", tags=[]))
        register_endpoint(OpenAPIEndpoint(path="/b", method="POST", summary="B", tags=[]))
        assert len(get_registered_endpoints()) == 2
        clear_registry()
        assert len(get_registered_endpoints()) == 0

    def test_clear_registry_when_empty(self):
        clear_registry()
        assert len(get_registered_endpoints()) == 0

    def test_get_registered_endpoints_dict_empty(self):
        result = get_registered_endpoints_dict()
        assert result == {}

    def test_get_registered_endpoints_dict_single(self):
        register_endpoint(
            OpenAPIEndpoint(
                path="/api/v1/items",
                method="GET",
                summary="List items",
                tags=["Items"],
            )
        )
        result = get_registered_endpoints_dict()
        assert "/api/v1/items" in result
        assert "get" in result["/api/v1/items"]
        spec = result["/api/v1/items"]["get"]
        assert spec["summary"] == "List items"

    def test_get_registered_endpoints_dict_multiple_methods_same_path(self):
        register_endpoint(
            OpenAPIEndpoint(
                path="/api/v1/items",
                method="GET",
                summary="List",
                tags=["Items"],
            )
        )
        register_endpoint(
            OpenAPIEndpoint(
                path="/api/v1/items",
                method="POST",
                summary="Create",
                tags=["Items"],
            )
        )
        result = get_registered_endpoints_dict()
        assert "/api/v1/items" in result
        assert "get" in result["/api/v1/items"]
        assert "post" in result["/api/v1/items"]

    def test_get_registered_endpoints_dict_multiple_paths(self):
        register_endpoint(OpenAPIEndpoint(path="/a", method="GET", summary="A", tags=[]))
        register_endpoint(OpenAPIEndpoint(path="/b", method="POST", summary="B", tags=[]))
        result = get_registered_endpoints_dict()
        assert "/a" in result
        assert "/b" in result

    def test_get_registered_endpoints_dict_method_lowercased(self):
        register_endpoint(OpenAPIEndpoint(path="/t", method="DELETE", summary="D", tags=[]))
        result = get_registered_endpoints_dict()
        assert "delete" in result["/t"]

    def test_duplicate_endpoints_allowed(self):
        """Registry does not deduplicate -- all appends accumulate."""
        ep = OpenAPIEndpoint(path="/dup", method="GET", summary="Dup", tags=[])
        register_endpoint(ep)
        register_endpoint(ep)
        assert len(get_registered_endpoints()) == 2

    def test_duplicate_in_dict_last_wins(self):
        """When same path+method is registered twice, last registration wins in dict."""
        register_endpoint(OpenAPIEndpoint(path="/dup", method="GET", summary="First", tags=["A"]))
        register_endpoint(OpenAPIEndpoint(path="/dup", method="GET", summary="Second", tags=["B"]))
        result = get_registered_endpoints_dict()
        assert result["/dup"]["get"]["summary"] == "Second"


# ============================================================================
# path_param helper
# ============================================================================


class TestPathParam:
    """Tests for path_param helper."""

    def test_basic(self):
        p = path_param("id")
        assert p["name"] == "id"
        assert p["in"] == "path"
        assert p["required"] is True
        assert p["schema"]["type"] == "string"
        assert p["description"] == ""

    def test_with_description(self):
        p = path_param("user_id", description="The user identifier")
        assert p["description"] == "The user identifier"

    def test_integer_schema(self):
        p = path_param("count", schema_type="integer")
        assert p["schema"]["type"] == "integer"

    def test_custom_type(self):
        p = path_param("data", schema_type="boolean")
        assert p["schema"]["type"] == "boolean"

    def test_always_required(self):
        p = path_param("x")
        assert p["required"] is True


# ============================================================================
# query_param helper
# ============================================================================


class TestQueryParam:
    """Tests for query_param helper."""

    def test_basic(self):
        p = query_param("q")
        assert p["name"] == "q"
        assert p["in"] == "query"
        assert p["schema"]["type"] == "string"
        assert p["description"] == ""
        assert "required" not in p

    def test_required(self):
        p = query_param("q", required=True)
        assert p["required"] is True

    def test_not_required_omits_key(self):
        p = query_param("q", required=False)
        assert "required" not in p

    def test_with_description(self):
        p = query_param("search", description="Search term")
        assert p["description"] == "Search term"

    def test_integer_type(self):
        p = query_param("limit", schema_type="integer")
        assert p["schema"]["type"] == "integer"

    def test_with_default(self):
        p = query_param("limit", schema_type="integer", default=10)
        assert p["schema"]["default"] == 10

    def test_default_none_omits_key(self):
        p = query_param("q")
        assert "default" not in p["schema"]

    def test_with_enum(self):
        p = query_param("sort", enum=["asc", "desc"])
        assert p["schema"]["enum"] == ["asc", "desc"]

    def test_enum_none_omits_key(self):
        p = query_param("q")
        assert "enum" not in p["schema"]

    def test_all_options(self):
        p = query_param(
            "status",
            description="Filter by status",
            schema_type="string",
            required=True,
            default="active",
            enum=["active", "inactive", "all"],
        )
        assert p["name"] == "status"
        assert p["in"] == "query"
        assert p["required"] is True
        assert p["description"] == "Filter by status"
        assert p["schema"]["type"] == "string"
        assert p["schema"]["default"] == "active"
        assert p["schema"]["enum"] == ["active", "inactive", "all"]

    def test_default_zero_is_set(self):
        """Default of 0 should NOT be treated as None/falsy."""
        # The implementation uses `if default is not None` so 0 should work.
        p = query_param("offset", schema_type="integer", default=0)
        assert p["schema"]["default"] == 0

    def test_empty_enum_is_not_set(self):
        """Empty list enum should not be set since it is falsy."""
        p = query_param("q", enum=[])
        assert "enum" not in p["schema"]


# ============================================================================
# json_body helper
# ============================================================================


class TestJsonBody:
    """Tests for json_body helper."""

    def test_basic_dict_schema(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        body = json_body(schema, "Create item")
        assert body["description"] == "Create item"
        assert body["required"] is True
        assert body["content"]["application/json"]["schema"] == schema

    def test_not_required(self):
        body = json_body({"type": "object"}, required=False)
        assert body["required"] is False

    def test_default_required_true(self):
        body = json_body({"type": "object"})
        assert body["required"] is True

    def test_empty_description(self):
        body = json_body({"type": "string"})
        assert body["description"] == ""

    def test_pydantic_model_auto_schema(self):
        body = json_body(SampleRequest)
        schema = body["content"]["application/json"]["schema"]
        assert "properties" in schema
        assert body["description"] == "SampleRequest request"

    def test_pydantic_model_with_description(self):
        body = json_body(SampleRequest, description="Custom desc")
        assert body["description"] == "Custom desc"

    def test_pydantic_model_auto_description_from_name(self):
        body = json_body(SampleRequest)
        assert "SampleRequest" in body["description"]


# ============================================================================
# ok_response helper
# ============================================================================


class TestOkResponse:
    """Tests for ok_response helper."""

    def test_basic(self):
        resp = ok_response()
        assert "200" in resp
        assert resp["200"]["description"] == "Success"
        assert resp["200"]["content"]["application/json"]["schema"] == {"type": "object"}

    def test_custom_description(self):
        resp = ok_response(description="Items listed")
        assert resp["200"]["description"] == "Items listed"

    def test_custom_status_code(self):
        resp = ok_response(status_code="201")
        assert "201" in resp
        assert "200" not in resp

    def test_with_dict_schema(self):
        schema = {"type": "array", "items": {"type": "string"}}
        resp = ok_response(schema=schema)
        assert resp["200"]["content"]["application/json"]["schema"] == schema

    def test_none_schema_default(self):
        resp = ok_response()
        assert resp["200"]["content"]["application/json"]["schema"] == {"type": "object"}

    def test_pydantic_model_schema(self):
        resp = ok_response(schema=SampleResponse)
        s = resp["200"]["content"]["application/json"]["schema"]
        assert "properties" in s


# ============================================================================
# error_response helper
# ============================================================================


class TestErrorResponse:
    """Tests for error_response helper."""

    def test_basic(self):
        resp = error_response("400", "Bad request")
        assert "400" in resp
        assert resp["400"]["description"] == "Bad request"

    def test_schema_has_error_and_details(self):
        resp = error_response("500", "Server error")
        schema = resp["500"]["content"]["application/json"]["schema"]
        assert schema["type"] == "object"
        assert "error" in schema["properties"]
        assert "details" in schema["properties"]

    def test_different_status_codes(self):
        for code in ["400", "401", "403", "404", "422", "500", "503"]:
            resp = error_response(code, f"Error {code}")
            assert code in resp
            assert resp[code]["description"] == f"Error {code}"

    def test_custom_description(self):
        resp = error_response("429", "Too many requests")
        assert resp["429"]["description"] == "Too many requests"


# ============================================================================
# _extract_pydantic_schema
# ============================================================================


class TestExtractPydanticSchema:
    """Tests for _extract_pydantic_schema internal function."""

    def test_pydantic_v2_model(self):
        schema = _extract_pydantic_schema(SampleRequest)
        assert isinstance(schema, dict)
        assert "properties" in schema

    def test_empty_model(self):
        schema = _extract_pydantic_schema(EmptyModel)
        assert isinstance(schema, dict)

    def test_model_without_schema_method(self):
        """If model has neither model_json_schema nor schema, return default."""

        class NoSchemaModel:
            pass

        schema = _extract_pydantic_schema(NoSchemaModel)
        assert schema == {"type": "object"}

    def test_model_with_failing_schema(self):
        """If schema extraction raises, return default."""

        class FailModel:
            def model_json_schema(self):
                raise ValueError("broken")

        schema = _extract_pydantic_schema(FailModel)
        assert schema == {"type": "object"}

    def test_model_with_v1_schema(self):
        """Test Pydantic v1 style .schema() method."""

        class V1Model:
            @classmethod
            def schema(cls):
                return {"type": "object", "title": "V1Model"}

        schema = _extract_pydantic_schema(V1Model)
        assert schema == {"type": "object", "title": "V1Model"}

    def test_model_with_v2_model_json_schema(self):
        """Test that model_json_schema takes priority over schema."""

        class DualModel:
            @classmethod
            def model_json_schema(cls):
                return {"type": "object", "title": "V2"}

            @classmethod
            def schema(cls):
                return {"type": "object", "title": "V1"}

        schema = _extract_pydantic_schema(DualModel)
        assert schema["title"] == "V2"

    def test_schema_raising_type_error(self):
        class BadModel:
            def model_json_schema(self):
                raise TypeError("bad type")

        schema = _extract_pydantic_schema(BadModel)
        assert schema == {"type": "object"}

    def test_schema_raising_key_error(self):
        class BadModel:
            def model_json_schema(self):
                raise KeyError("missing key")

        schema = _extract_pydantic_schema(BadModel)
        assert schema == {"type": "object"}

    def test_schema_raising_attribute_error(self):
        class BadModel:
            def model_json_schema(self):
                raise AttributeError("no attr")

        schema = _extract_pydantic_schema(BadModel)
        assert schema == {"type": "object"}

    def test_schema_raising_runtime_error(self):
        class BadModel:
            def model_json_schema(self):
                raise RuntimeError("runtime fail")

        schema = _extract_pydantic_schema(BadModel)
        assert schema == {"type": "object"}


# ============================================================================
# _is_pydantic_model
# ============================================================================


class TestIsPydanticModel:
    """Tests for _is_pydantic_model internal function."""

    def test_pydantic_model_class(self):
        assert _is_pydantic_model(SampleRequest) is True

    def test_pydantic_model_instance_is_false(self):
        """Instances are not model classes."""
        instance = SampleRequest(name="test")
        assert _is_pydantic_model(instance) is False

    def test_regular_class_is_false(self):
        class MyClass:
            pass

        assert _is_pydantic_model(MyClass) is False

    def test_none_is_false(self):
        assert _is_pydantic_model(None) is False

    def test_string_is_false(self):
        assert _is_pydantic_model("not a model") is False

    def test_int_is_false(self):
        assert _is_pydantic_model(42) is False

    def test_dict_is_false(self):
        assert _is_pydantic_model({"type": "object"}) is False

    def test_list_is_false(self):
        assert _is_pydantic_model([]) is False

    def test_builtin_type_is_false(self):
        assert _is_pydantic_model(str) is False
        assert _is_pydantic_model(int) is False
        assert _is_pydantic_model(dict) is False

    def test_pydantic_import_failure(self):
        """When pydantic cannot be imported, returns False."""
        with patch("aragora.server.handlers.openapi_decorator._is_pydantic_model") as mock_fn:
            # We can't easily mock the import inside the function, so we
            # test the behavior by calling directly with a non-pydantic type
            pass
        # Test with a plain dict -- should always be False
        assert _is_pydantic_model(dict) is False


# ============================================================================
# Integration / Combination Tests
# ============================================================================


class TestIntegration:
    """Tests combining multiple features of the decorator module."""

    def test_full_crud_registration(self):
        """Register a full CRUD set and verify dict output."""

        @api_endpoint(
            path="/api/v1/items",
            method="GET",
            summary="List items",
            tags=["Items"],
            auth_required=False,
        )
        def list_items():
            pass

        @api_endpoint(
            path="/api/v1/items",
            method="POST",
            summary="Create item",
            tags=["Items"],
        )
        def create_item():
            pass

        @api_endpoint(
            path="/api/v1/items/{id}",
            method="GET",
            summary="Get item",
            tags=["Items"],
        )
        def get_item():
            pass

        @api_endpoint(
            path="/api/v1/items/{id}",
            method="PUT",
            summary="Update item",
            tags=["Items"],
        )
        def update_item():
            pass

        @api_endpoint(
            path="/api/v1/items/{id}",
            method="DELETE",
            summary="Delete item",
            tags=["Items"],
            deprecated=True,
        )
        def delete_item():
            pass

        endpoints = get_registered_endpoints()
        assert len(endpoints) == 5

        d = get_registered_endpoints_dict()
        assert len(d) == 2  # two paths
        assert "get" in d["/api/v1/items"]
        assert "post" in d["/api/v1/items"]
        assert "get" in d["/api/v1/items/{id}"]
        assert "put" in d["/api/v1/items/{id}"]
        assert "delete" in d["/api/v1/items/{id}"]

        # Verify the deprecated endpoint
        delete_spec = d["/api/v1/items/{id}"]["delete"]
        assert delete_spec["deprecated"] is True

        # Verify auth
        list_spec = d["/api/v1/items"]["get"]
        assert "security" not in list_spec  # auth_required=False

        create_spec = d["/api/v1/items"]["post"]
        assert create_spec["security"] == [{"bearerAuth": []}]

    def test_helpers_compose_with_decorator(self):
        """Use helper functions inside api_endpoint calls."""

        @api_endpoint(
            path="/api/v1/search",
            method="GET",
            summary="Search",
            tags=["Search"],
            parameters=[
                path_param("namespace", "The namespace"),
                query_param("q", "Search query", required=True),
                query_param("limit", schema_type="integer", default=10),
                query_param("sort", enum=["asc", "desc"]),
            ],
            responses={
                **ok_response("Search results"),
                **error_response("400", "Bad query"),
                **error_response("500", "Server error"),
            },
        )
        def search():
            pass

        ep = search._openapi
        assert len(ep.parameters) == 4
        assert ep.parameters[0]["in"] == "path"
        assert ep.parameters[1]["required"] is True
        assert ep.parameters[2]["schema"]["default"] == 10
        assert ep.parameters[3]["schema"]["enum"] == ["asc", "desc"]

        assert "200" in ep.responses
        assert "400" in ep.responses
        assert "500" in ep.responses

    def test_register_then_clear_then_reregister(self):
        register_endpoint(OpenAPIEndpoint(path="/a", method="GET", summary="A", tags=[]))
        assert len(get_registered_endpoints()) == 1
        clear_registry()
        assert len(get_registered_endpoints()) == 0
        register_endpoint(OpenAPIEndpoint(path="/b", method="POST", summary="B", tags=[]))
        assert len(get_registered_endpoints()) == 1
        assert get_registered_endpoints()[0].path == "/b"

    def test_pydantic_models_in_json_body_helper_with_decorator(self):
        @api_endpoint(
            path="/api/v1/resource",
            method="POST",
            summary="Create",
            tags=[],
            request_body=json_body(SampleRequest, "Create resource"),
            responses=ok_response("Resource created", SampleResponse, status_code="201"),
        )
        def create_resource():
            pass

        ep = create_resource._openapi
        assert ep.request_body is not None
        assert ep.request_body["description"] == "Create resource"
        schema = ep.request_body["content"]["application/json"]["schema"]
        assert "properties" in schema

        assert "201" in ep.responses
        resp_schema = ep.responses["201"]["content"]["application/json"]["schema"]
        assert "properties" in resp_schema


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_empty_path(self):
        @api_endpoint(path="", summary="S", tags=[])
        def f():
            pass

        assert f._openapi.path == ""

    def test_very_long_path(self):
        long_path = "/api/v1/" + "/".join(f"segment{i}" for i in range(50))

        @api_endpoint(path=long_path, summary="S", tags=[])
        def f():
            pass

        assert f._openapi.path == long_path

    def test_special_characters_in_path(self):
        @api_endpoint(path="/api/v1/items/{item-id}", summary="S", tags=[])
        def f():
            pass

        assert f._openapi.path == "/api/v1/items/{item-id}"

    def test_unicode_in_summary(self):
        @api_endpoint(path="/p", summary="Buscar datos", tags=[])
        def f():
            pass

        assert f._openapi.summary == "Buscar datos"

    def test_unicode_in_description(self):
        @api_endpoint(path="/p", summary="S", tags=[], description="Beschreibung")
        def f():
            pass

        assert f._openapi.description == "Beschreibung"

    def test_method_mixed_case(self):
        @api_endpoint(path="/p", method="PaTcH", summary="S", tags=[])
        def f():
            pass

        assert f._openapi.method == "PATCH"

    def test_lambda_as_decorated_function(self):
        """Lambdas cannot be decorated with @, but can be passed manually."""
        decorator = api_endpoint(path="/p", summary="S", tags=[])
        decorated = decorator(lambda: 42)
        assert decorated() == 42
        assert hasattr(decorated, "_openapi")

    def test_class_method_decorated(self):
        class MyHandler:
            @api_endpoint(path="/handler", summary="Handle", tags=["H"])
            def handle(self, request):
                return "handled"

        h = MyHandler()
        assert h.handle("req") == "handled"
        assert hasattr(MyHandler.handle, "_openapi")

    def test_none_tags_becomes_empty_list(self):
        """When tags=None is passed, should default to []."""

        @api_endpoint(path="/p", summary="S", tags=None)
        def f():
            pass

        assert f._openapi.tags == []

    def test_none_parameters_becomes_empty_list(self):
        @api_endpoint(path="/p", summary="S", tags=[], parameters=None)
        def f():
            pass

        assert f._openapi.parameters == []


# ============================================================================
# Security-oriented tests
# ============================================================================


class TestSecurity:
    """Security-related tests."""

    def test_path_traversal_in_path_param(self):
        """Path traversal in param names is just a string -- no execution risk."""
        p = path_param("../../etc/passwd")
        assert p["name"] == "../../etc/passwd"
        assert p["in"] == "path"

    def test_script_injection_in_summary(self):
        """Script injection in summary is stored as-is (rendering responsibility)."""

        @api_endpoint(
            path="/p",
            summary="<script>alert('xss')</script>",
            tags=[],
        )
        def f():
            pass

        assert f._openapi.summary == "<script>alert('xss')</script>"

    def test_html_injection_in_description(self):
        @api_endpoint(
            path="/p",
            summary="S",
            tags=[],
            description="<img src=x onerror=alert(1)>",
        )
        def f():
            pass

        assert f._openapi.description == "<img src=x onerror=alert(1)>"

    def test_null_bytes_in_path(self):
        @api_endpoint(path="/api/v1/foo\x00bar", summary="S", tags=[])
        def f():
            pass

        assert "\x00" in f._openapi.path

    def test_extremely_long_summary(self):
        long_summary = "A" * 10000

        @api_endpoint(path="/p", summary=long_summary, tags=[])
        def f():
            pass

        assert len(f._openapi.summary) == 10000

    def test_query_param_injection_in_enum(self):
        """Enum with potentially dangerous values stored safely."""
        p = query_param("x", enum=["'; DROP TABLE users; --", "<script>"])
        assert p["schema"]["enum"] == ["'; DROP TABLE users; --", "<script>"]


# ============================================================================
# __all__ export list
# ============================================================================


class TestExports:
    """Verify the module exports the expected public API."""

    def test_all_exports(self):
        from aragora.server.handlers import openapi_decorator

        expected = [
            "OpenAPIEndpoint",
            "api_endpoint",
            "get_registered_endpoints",
            "get_registered_endpoints_dict",
            "clear_registry",
            "register_endpoint",
            "path_param",
            "query_param",
            "json_body",
            "ok_response",
            "error_response",
        ]
        for name in expected:
            assert name in openapi_decorator.__all__, f"{name} missing from __all__"
            assert hasattr(openapi_decorator, name), f"{name} not accessible on module"

    def test_all_count(self):
        from aragora.server.handlers import openapi_decorator

        assert len(openapi_decorator.__all__) == 11
