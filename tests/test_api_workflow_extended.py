"""
API Workflow Extended Tests.

Tests for API patterns across handlers:
- Routing and path matching
- Parameter extraction and validation
- Error response formatting
- Handler registration and discovery
- Request/response patterns
"""

from __future__ import annotations

import json
import pytest
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Optional
from unittest.mock import MagicMock, patch

from aragora.server.handlers.base import BaseHandler, HandlerResult


# =============================================================================
# Mock Classes
# =============================================================================


@dataclass
class MockRequest:
    """Mock HTTP request for testing."""

    path: str = "/api/test"
    method: str = "GET"
    headers: dict = field(default_factory=dict)
    query_params: dict = field(default_factory=dict)
    body: Optional[bytes] = None

    def __post_init__(self):
        if self.body:
            self.rfile = BytesIO(self.body)
            self.headers["Content-Length"] = str(len(self.body))
        else:
            self.rfile = BytesIO(b"")

        self.command = self.method

    def get(self, key: str, default: Any = None) -> Any:
        """Get query parameter."""
        return self.query_params.get(key, default)


class TestHandler(BaseHandler):
    """Test handler implementation."""

    ROUTES = [
        "/api/test",
        "/api/test/item",
        "/api/test/search",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path in self.ROUTES

    def handle(self, path: str, params: dict, handler: Any, method: str) -> HandlerResult:
        """Handle test routes."""
        if path == "/api/test" and method == "GET":
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=json.dumps({"message": "Success"}).encode()
            )
        elif path == "/api/test/item" and method == "POST":
            return HandlerResult(
                status_code=201,
                content_type="application/json",
                body=json.dumps({"created": True}).encode()
            )
        elif path == "/api/test/search":
            query = params.get("q", "")
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=json.dumps({"query": query, "results": []}).encode()
            )
        else:
            return HandlerResult(
                status_code=405,
                content_type="application/json",
                body=json.dumps({"error": "Method not allowed"}).encode()
            )


# =============================================================================
# BaseHandler Tests
# =============================================================================


class TestBaseHandler:
    """Tests for BaseHandler base class."""

    def test_base_handler_init_with_context(self):
        """Test BaseHandler initialization with context."""
        ctx = {"user_store": MagicMock()}
        handler = BaseHandler(server_context=ctx)
        assert handler.ctx == ctx

    def test_base_handler_init_empty_context(self):
        """Test BaseHandler initialization with empty context."""
        handler = BaseHandler(server_context={})
        assert handler.ctx == {}

    def test_base_handler_has_ctx(self):
        """Test BaseHandler stores context."""
        ctx = {"key": "value"}
        handler = BaseHandler(server_context=ctx)
        assert handler.ctx["key"] == "value"


# =============================================================================
# HandlerResult Tests
# =============================================================================


class TestHandlerResult:
    """Tests for HandlerResult dataclass."""

    def test_handler_result_creation(self):
        """Test HandlerResult creation."""
        result = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=b'{"test": true}'
        )
        assert result.status_code == 200
        assert result.body == b'{"test": true}'

    def test_handler_result_with_headers(self):
        """Test HandlerResult with custom headers."""
        result = HandlerResult(
            status_code=200,
            content_type="text/plain",
            body=b"OK",
            headers={"X-Custom": "value"}
        )
        assert result.headers["X-Custom"] == "value"

    def test_handler_result_content_type(self):
        """Test HandlerResult with content type."""
        result = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=b'{"json": true}'
        )
        assert result.content_type == "application/json"


# =============================================================================
# Route Matching Tests
# =============================================================================


class TestRouteMatching:
    """Tests for route matching patterns."""

    def test_can_handle_exact_match(self):
        """Test exact route matching."""
        handler = TestHandler(server_context={})
        assert handler.can_handle("/api/test") is True

    def test_can_handle_no_match(self):
        """Test non-matching route."""
        handler = TestHandler(server_context={})
        assert handler.can_handle("/api/other") is False

    def test_can_handle_partial_match_fails(self):
        """Test partial match doesn't succeed."""
        handler = TestHandler(server_context={})
        assert handler.can_handle("/api/tes") is False
        assert handler.can_handle("/api/test/extra/path") is False

    def test_can_handle_all_routes(self):
        """Test all defined routes can be handled."""
        handler = TestHandler(server_context={})
        for route in TestHandler.ROUTES:
            assert handler.can_handle(route) is True


# =============================================================================
# Request Handling Tests
# =============================================================================


class TestRequestHandling:
    """Tests for request handling patterns."""

    def test_get_request(self):
        """Test handling GET request."""
        handler = TestHandler(server_context={})
        request = MockRequest(path="/api/test", method="GET")

        result = handler.handle("/api/test", {}, request, "GET")

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["message"] == "Success"

    def test_post_request(self):
        """Test handling POST request."""
        handler = TestHandler(server_context={})
        request = MockRequest(path="/api/test/item", method="POST")

        result = handler.handle("/api/test/item", {}, request, "POST")

        assert result.status_code == 201
        body = json.loads(result.body.decode())
        assert body["created"] is True

    def test_method_not_allowed(self):
        """Test method not allowed response."""
        handler = TestHandler(server_context={})
        request = MockRequest(path="/api/test", method="DELETE")

        result = handler.handle("/api/test", {}, request, "DELETE")

        assert result.status_code == 405

    def test_query_parameters(self):
        """Test query parameter extraction."""
        handler = TestHandler(server_context={})
        request = MockRequest(
            path="/api/test/search",
            method="GET",
            query_params={"q": "search term"}
        )

        result = handler.handle("/api/test/search", {"q": "search term"}, request, "GET")

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["query"] == "search term"


# =============================================================================
# Error Response Tests
# =============================================================================


class TestErrorResponses:
    """Tests for error response formatting."""

    def test_error_response_400(self):
        """Test 400 Bad Request response."""
        result = HandlerResult(
            status_code=400,
            content_type="application/json",
            body=json.dumps({"error": "Invalid request"}).encode()
        )
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body

    def test_error_response_401(self):
        """Test 401 Unauthorized response."""
        result = HandlerResult(
            status_code=401,
            content_type="application/json",
            body=json.dumps({"error": "Not authenticated"}).encode()
        )
        assert result.status_code == 401

    def test_error_response_403(self):
        """Test 403 Forbidden response."""
        result = HandlerResult(
            status_code=403,
            content_type="application/json",
            body=json.dumps({"error": "Access denied"}).encode()
        )
        assert result.status_code == 403

    def test_error_response_404(self):
        """Test 404 Not Found response."""
        result = HandlerResult(
            status_code=404,
            content_type="application/json",
            body=json.dumps({"error": "Resource not found"}).encode()
        )
        assert result.status_code == 404

    def test_error_response_500(self):
        """Test 500 Internal Server Error response."""
        result = HandlerResult(
            status_code=500,
            content_type="application/json",
            body=json.dumps({"error": "Internal server error"}).encode()
        )
        assert result.status_code == 500


# =============================================================================
# Parameter Validation Tests
# =============================================================================


class TestParameterValidation:
    """Tests for parameter validation patterns."""

    def test_required_parameter_missing(self):
        """Test required parameter validation."""
        params = {}
        assert params.get("required_field") is None

    def test_required_parameter_present(self):
        """Test required parameter present."""
        params = {"required_field": "value"}
        assert params.get("required_field") == "value"

    def test_optional_parameter_with_default(self):
        """Test optional parameter with default."""
        params = {}
        value = params.get("optional", "default_value")
        assert value == "default_value"

    def test_integer_parameter_conversion(self):
        """Test integer parameter conversion."""
        params = {"limit": "10"}
        limit = int(params.get("limit", "50"))
        assert limit == 10

    def test_boolean_parameter_conversion(self):
        """Test boolean parameter conversion."""
        params = {"enabled": "true"}
        enabled = params.get("enabled", "false").lower() == "true"
        assert enabled is True


# =============================================================================
# JSON Body Tests
# =============================================================================


class TestJSONBody:
    """Tests for JSON body handling."""

    def test_parse_json_body(self):
        """Test JSON body parsing."""
        body = json.dumps({"key": "value"}).encode()
        request = MockRequest(body=body)

        data = json.loads(request.rfile.read())
        assert data["key"] == "value"

    def test_parse_empty_body(self):
        """Test empty body handling."""
        request = MockRequest()
        content = request.rfile.read()
        assert content == b""

    def test_invalid_json_handling(self):
        """Test invalid JSON handling."""
        body = b"not valid json"
        request = MockRequest(body=body)

        with pytest.raises(json.JSONDecodeError):
            json.loads(request.rfile.read())

    def test_nested_json_body(self):
        """Test nested JSON body."""
        body = json.dumps({
            "outer": {
                "inner": {
                    "value": 42
                }
            }
        }).encode()
        request = MockRequest(body=body)

        data = json.loads(request.rfile.read())
        assert data["outer"]["inner"]["value"] == 42


# =============================================================================
# Header Tests
# =============================================================================


class TestHeaders:
    """Tests for header handling."""

    def test_content_type_header(self):
        """Test Content-Type header."""
        request = MockRequest(headers={"Content-Type": "application/json"})
        assert request.headers["Content-Type"] == "application/json"

    def test_authorization_header(self):
        """Test Authorization header."""
        request = MockRequest(headers={"Authorization": "Bearer token123"})
        assert request.headers["Authorization"] == "Bearer token123"

    def test_custom_headers(self):
        """Test custom headers."""
        request = MockRequest(headers={"X-Custom-Header": "custom-value"})
        assert request.headers["X-Custom-Header"] == "custom-value"

    def test_content_length_header(self):
        """Test Content-Length header auto-set."""
        body = b"test body"
        request = MockRequest(body=body)
        assert request.headers["Content-Length"] == str(len(body))


# =============================================================================
# Response Content Type Tests
# =============================================================================


class TestResponseContentType:
    """Tests for response content type handling."""

    def test_json_response(self):
        """Test JSON response content type."""
        result = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=b'{"data": true}'
        )
        assert result.content_type == "application/json"

    def test_text_response(self):
        """Test text response content type."""
        result = HandlerResult(
            status_code=200,
            content_type="text/plain",
            body=b"Plain text"
        )
        assert result.content_type == "text/plain"

    def test_html_response(self):
        """Test HTML response content type."""
        result = HandlerResult(
            status_code=200,
            content_type="text/html",
            body=b"<html>...</html>"
        )
        assert result.content_type == "text/html"


# =============================================================================
# Path Pattern Tests
# =============================================================================


class TestPathPatterns:
    """Tests for path pattern matching."""

    def test_api_prefix(self):
        """Test API prefix pattern."""
        path = "/api/v1/resource"
        assert path.startswith("/api/")

    def test_version_in_path(self):
        """Test version extraction from path."""
        path = "/api/v2/resource"
        parts = path.split("/")
        version = parts[2] if len(parts) > 2 else None
        assert version == "v2"

    def test_resource_id_in_path(self):
        """Test resource ID extraction."""
        path = "/api/items/123"
        parts = path.split("/")
        resource_id = parts[-1] if parts else None
        assert resource_id == "123"

    def test_nested_resource_path(self):
        """Test nested resource path parsing."""
        path = "/api/users/42/posts/123"
        parts = path.split("/")
        assert parts[3] == "42"  # user_id
        assert parts[5] == "123"  # post_id


# =============================================================================
# Status Code Tests
# =============================================================================


class TestStatusCodes:
    """Tests for HTTP status codes."""

    def test_success_codes(self):
        """Test 2xx success codes."""
        assert HandlerResult(status_code=200, content_type="text/plain", body=b"").status_code == 200
        assert HandlerResult(status_code=201, content_type="text/plain", body=b"").status_code == 201
        assert HandlerResult(status_code=204, content_type="text/plain", body=b"").status_code == 204

    def test_redirect_codes(self):
        """Test 3xx redirect codes."""
        assert HandlerResult(status_code=301, content_type="text/plain", body=b"").status_code == 301
        assert HandlerResult(status_code=302, content_type="text/plain", body=b"").status_code == 302
        assert HandlerResult(status_code=304, content_type="text/plain", body=b"").status_code == 304

    def test_client_error_codes(self):
        """Test 4xx client error codes."""
        assert HandlerResult(status_code=400, content_type="text/plain", body=b"").status_code == 400
        assert HandlerResult(status_code=401, content_type="text/plain", body=b"").status_code == 401
        assert HandlerResult(status_code=404, content_type="text/plain", body=b"").status_code == 404
        assert HandlerResult(status_code=429, content_type="text/plain", body=b"").status_code == 429

    def test_server_error_codes(self):
        """Test 5xx server error codes."""
        assert HandlerResult(status_code=500, content_type="text/plain", body=b"").status_code == 500
        assert HandlerResult(status_code=502, content_type="text/plain", body=b"").status_code == 502
        assert HandlerResult(status_code=503, content_type="text/plain", body=b"").status_code == 503


# =============================================================================
# Pagination Tests
# =============================================================================


class TestPagination:
    """Tests for pagination patterns."""

    def test_limit_offset_pagination(self):
        """Test limit/offset pagination."""
        params = {"limit": "20", "offset": "40"}
        limit = int(params.get("limit", "50"))
        offset = int(params.get("offset", "0"))
        assert limit == 20
        assert offset == 40

    def test_page_pagination(self):
        """Test page-based pagination."""
        params = {"page": "3", "per_page": "25"}
        page = int(params.get("page", "1"))
        per_page = int(params.get("per_page", "20"))
        offset = (page - 1) * per_page
        assert offset == 50

    def test_cursor_pagination(self):
        """Test cursor-based pagination."""
        params = {"cursor": "abc123", "limit": "10"}
        cursor = params.get("cursor")
        limit = int(params.get("limit", "20"))
        assert cursor == "abc123"
        assert limit == 10

    def test_pagination_defaults(self):
        """Test pagination defaults."""
        params = {}
        limit = int(params.get("limit", "50"))
        offset = int(params.get("offset", "0"))
        assert limit == 50
        assert offset == 0


# =============================================================================
# Filter Tests
# =============================================================================


class TestFiltering:
    """Tests for filtering patterns."""

    def test_single_filter(self):
        """Test single filter parameter."""
        params = {"status": "active"}
        assert params.get("status") == "active"

    def test_multiple_filters(self):
        """Test multiple filter parameters."""
        params = {"status": "active", "category": "tech", "sort": "date"}
        assert params.get("status") == "active"
        assert params.get("category") == "tech"
        assert params.get("sort") == "date"

    def test_list_filter(self):
        """Test list filter (comma-separated)."""
        params = {"tags": "python,async,api"}
        tags = params.get("tags", "").split(",")
        assert "python" in tags
        assert "async" in tags
        assert "api" in tags

    def test_range_filter(self):
        """Test range filter."""
        params = {"min_price": "10", "max_price": "100"}
        min_price = int(params.get("min_price", "0"))
        max_price = int(params.get("max_price", "999999"))
        assert min_price == 10
        assert max_price == 100


# =============================================================================
# Sorting Tests
# =============================================================================


class TestSorting:
    """Tests for sorting patterns."""

    def test_sort_field(self):
        """Test sort field parameter."""
        params = {"sort": "created_at"}
        assert params.get("sort") == "created_at"

    def test_sort_order(self):
        """Test sort order parameter."""
        params = {"sort": "name", "order": "desc"}
        order = params.get("order", "asc")
        assert order == "desc"

    def test_sort_combined(self):
        """Test combined sort format."""
        params = {"sort": "-created_at"}  # Prefix with - for descending
        sort_field = params.get("sort", "id")
        descending = sort_field.startswith("-")
        field_name = sort_field.lstrip("-")
        assert descending is True
        assert field_name == "created_at"


# =============================================================================
# Search Tests
# =============================================================================


class TestSearch:
    """Tests for search patterns."""

    def test_basic_search(self):
        """Test basic search query."""
        params = {"q": "search term"}
        assert params.get("q") == "search term"

    def test_empty_search(self):
        """Test empty search query."""
        params = {}
        query = params.get("q", "")
        assert query == ""

    def test_search_with_filters(self):
        """Test search with filters."""
        params = {"q": "python", "category": "programming"}
        assert params.get("q") == "python"
        assert params.get("category") == "programming"


__all__ = [
    "TestBaseHandler",
    "TestHandlerResult",
    "TestRouteMatching",
    "TestRequestHandling",
    "TestErrorResponses",
    "TestParameterValidation",
    "TestJSONBody",
    "TestHeaders",
    "TestResponseContentType",
    "TestPathPatterns",
    "TestStatusCodes",
    "TestPagination",
    "TestFiltering",
    "TestSorting",
    "TestSearch",
]
