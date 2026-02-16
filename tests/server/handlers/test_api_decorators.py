"""
Tests for aragora.server.handlers.api_decorators - API decorator utilities.

Tests cover:
- api_endpoint: Attaching API metadata to handlers
- rate_limit: Async-friendly rate limiting wrapper
- validate_body: JSON request body validation
- require_quota: Organization debate quota enforcement
- Decorator composition/stacking

Run with: pytest tests/server/handlers/test_api_decorators.py -v
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers.api_decorators import (
    api_endpoint,
    rate_limit,
    validate_body,
    require_quota,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockRequest:
    """Mock HTTP request object for testing."""

    _json_data: dict[str, Any] | None = None
    _json_error: Exception | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> dict[str, Any]:
        """Synchronous JSON body retrieval."""
        if self._json_error:
            raise self._json_error
        return self._json_data or {}

    async def async_json(self) -> dict[str, Any]:
        """Async JSON body retrieval."""
        if self._json_error:
            raise self._json_error
        return self._json_data or {}


@dataclass
class MockAsyncRequest:
    """Mock HTTP request with async json() method."""

    _json_data: dict[str, Any] | None = None
    _json_error: Exception | None = None
    headers: dict[str, str] = field(default_factory=dict)

    async def json(self) -> dict[str, Any]:
        """Async JSON body retrieval."""
        if self._json_error:
            raise self._json_error
        return self._json_data or {}


@dataclass
class MockUserAuthContext:
    """Mock user authentication context."""

    is_authenticated: bool = True
    user_id: str = "user-123"
    email: str = "test@example.com"
    org_id: str | None = "org-123"
    role: str = "user"
    error_reason: str | None = None


@dataclass
class MockOrganizationLimits:
    """Mock organization limits."""

    debates_per_month: int = 100


@dataclass
class MockOrganization:
    """Mock organization for quota testing."""

    id: str = "org-123"
    name: str = "Test Org"
    tier: Any = field(default_factory=lambda: MagicMock(value="pro"))
    limits: MockOrganizationLimits = field(default_factory=MockOrganizationLimits)
    debates_used_this_month: int = 50
    is_at_limit: bool = False


@dataclass
class MockHandler:
    """Mock handler class with error_response method."""

    headers: dict[str, str] = field(default_factory=dict)
    user_store: Any = None

    def error_response(self, message: str, status: int = 400) -> HandlerResult:
        """Create error response."""
        body = json.dumps({"error": message}).encode("utf-8")
        return HandlerResult(
            status_code=status,
            content_type="application/json",
            body=body,
        )


@dataclass
class MockUserStore:
    """Mock user store for quota testing."""

    organizations: dict[str, MockOrganization] = field(default_factory=dict)
    usage_increments: list[tuple[str, int]] = field(default_factory=list)

    def get_organization_by_id(self, org_id: str) -> MockOrganization | None:
        return self.organizations.get(org_id)

    def increment_usage(self, org_id: str, count: int) -> None:
        self.usage_increments.append((org_id, count))


# ===========================================================================
# api_endpoint Tests
# ===========================================================================


class TestApiEndpoint:
    """Tests for the api_endpoint decorator."""

    def test_basic_metadata_attachment(self):
        """Test that api_endpoint attaches metadata to function."""

        @api_endpoint(method="GET", path="/api/items")
        def get_items():
            return []

        assert hasattr(get_items, "_api_metadata")
        metadata = get_items._api_metadata
        assert metadata["method"] == "GET"
        assert metadata["path"] == "/api/items"

    def test_all_metadata_fields(self):
        """Test all metadata fields are attached correctly."""

        @api_endpoint(
            method="POST",
            path="/api/debates/{id}",
            summary="Create a debate",
            description="Creates a new multi-agent debate session",
        )
        def create_debate():
            return {"id": "123"}

        metadata = create_debate._api_metadata
        assert metadata["method"] == "POST"
        assert metadata["path"] == "/api/debates/{id}"
        assert metadata["summary"] == "Create a debate"
        assert metadata["description"] == "Creates a new multi-agent debate session"

    def test_default_summary_and_description(self):
        """Test default values for summary and description."""

        @api_endpoint(method="DELETE", path="/api/items/{id}")
        def delete_item():
            pass

        metadata = delete_item._api_metadata
        assert metadata["summary"] == ""
        assert metadata["description"] == ""

    def test_preserves_function_behavior(self):
        """Test that decorated function still works correctly."""

        @api_endpoint(method="GET", path="/api/test")
        def test_func(x: int, y: int) -> int:
            return x + y

        assert test_func(2, 3) == 5

    def test_preserves_function_name(self):
        """Test that function name is preserved."""

        @api_endpoint(method="GET", path="/api/test")
        def my_handler():
            pass

        # The function object itself should still be the same
        assert my_handler.__name__ == "my_handler"

    def test_all_http_methods(self):
        """Test decorator works with all HTTP methods."""
        methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]

        for method in methods:

            @api_endpoint(method=method, path="/api/test")
            def handler():
                pass

            assert handler._api_metadata["method"] == method

    def test_path_with_multiple_params(self):
        """Test path patterns with multiple parameters."""

        @api_endpoint(method="GET", path="/api/orgs/{org_id}/debates/{debate_id}/messages")
        def get_messages():
            pass

        assert (
            get_messages._api_metadata["path"] == "/api/orgs/{org_id}/debates/{debate_id}/messages"
        )


# ===========================================================================
# rate_limit Tests
# ===========================================================================


class TestRateLimit:
    """Tests for the rate_limit decorator wrapper."""

    @patch("aragora.server.handlers.api_decorators._rate_limit")
    def test_sync_function_decoration(self, mock_rate_limit):
        """Test rate_limit decorator on synchronous function."""
        # Setup mock
        mock_decorator = MagicMock(side_effect=lambda f: f)
        mock_rate_limit.return_value = mock_decorator

        @rate_limit(requests_per_minute=30)
        def sync_handler():
            return "result"

        result = sync_handler()
        assert result == "result"
        mock_rate_limit.assert_called_once_with(requests_per_minute=30)

    @patch("aragora.server.handlers.api_decorators._rate_limit")
    def test_async_function_decoration(self, mock_rate_limit):
        """Test rate_limit decorator on async function."""

        # Setup mock that returns an awaitable
        async def async_result(*args, **kwargs):
            return "async_result"

        mock_decorated = MagicMock(return_value=async_result())
        mock_decorator = MagicMock(return_value=mock_decorated)
        mock_rate_limit.return_value = mock_decorator

        @rate_limit(requests_per_minute=60)
        async def async_handler():
            return "async_result"

        result = asyncio.run(async_handler())
        assert result == "async_result"
        mock_rate_limit.assert_called_once_with(requests_per_minute=60)

    @patch("aragora.server.handlers.api_decorators._rate_limit")
    def test_rate_limit_with_burst(self, mock_rate_limit):
        """Test rate_limit with burst parameter."""
        mock_decorator = MagicMock(side_effect=lambda f: f)
        mock_rate_limit.return_value = mock_decorator

        @rate_limit(requests_per_minute=10, burst=5)
        def handler():
            return "ok"

        handler()
        mock_rate_limit.assert_called_once_with(requests_per_minute=10, burst=5)

    @patch("aragora.server.handlers.api_decorators._rate_limit")
    def test_rate_limit_passes_all_kwargs(self, mock_rate_limit):
        """Test that all kwargs are passed to underlying rate_limit."""
        mock_decorator = MagicMock(side_effect=lambda f: f)
        mock_rate_limit.return_value = mock_decorator

        @rate_limit(
            requests_per_minute=100,
            burst=20,
            limiter_name="test_limiter",
            key_type="combined",
        )
        def handler():
            pass

        handler()
        mock_rate_limit.assert_called_once_with(
            requests_per_minute=100,
            burst=20,
            limiter_name="test_limiter",
            key_type="combined",
        )


# ===========================================================================
# validate_body Tests
# ===========================================================================


class TestValidateBody:
    """Tests for the validate_body decorator."""

    # ---- Async Handler Tests ----

    def test_async_valid_body(self):
        """Test async handler with valid request body."""

        class Handler:
            @validate_body(["name", "email"])
            async def create_user(self, request):
                body = await request.json()
                return {"created": True, "name": body["name"]}

        handler = Handler()
        request = MockAsyncRequest(_json_data={"name": "Test", "email": "test@example.com"})

        result = asyncio.run(handler.create_user(request))
        assert result["created"] is True
        assert result["name"] == "Test"

    def test_async_missing_required_fields(self):
        """Test async handler with missing required fields."""

        class Handler:
            @validate_body(["name", "email", "password"])
            async def register(self, request):
                return {"success": True}

        handler = Handler()
        request = MockAsyncRequest(_json_data={"name": "Test"})

        result = asyncio.run(handler.register(request))
        # Should return error response
        assert isinstance(result, HandlerResult)
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "Missing required fields" in body["error"]
        assert "email" in body["error"]
        assert "password" in body["error"]

    def test_async_invalid_json(self):
        """Test async handler with invalid JSON body."""

        class Handler:
            @validate_body(["field"])
            async def process(self, request):
                return {"success": True}

        handler = Handler()
        request = MockAsyncRequest(_json_error=json.JSONDecodeError("Test", "", 0))

        result = asyncio.run(handler.process(request))
        assert isinstance(result, HandlerResult)
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "Invalid JSON body" in body["error"]

    def test_async_uses_handler_error_response(self):
        """Test that async handler uses self.error_response if available."""

        class HandlerWithErrorMethod:
            def error_response(self, message: str, status: int = 400) -> dict:
                return {"custom_error": message, "status": status}

            @validate_body(["field"])
            async def process(self, request):
                return {"success": True}

        handler = HandlerWithErrorMethod()
        request = MockAsyncRequest(_json_data={})

        result = asyncio.run(handler.process(request))
        assert result["custom_error"] == "Missing required fields: field"
        assert result["status"] == 400

    # ---- Sync Handler Tests ----

    def test_sync_valid_body(self):
        """Test sync handler with valid request body."""

        class Handler:
            @validate_body(["task"])
            def create_debate(self, request):
                body = request.json()
                return {"task": body["task"]}

        handler = Handler()
        request = MockRequest(_json_data={"task": "Analyze data"})

        result = handler.create_debate(request)
        assert result["task"] == "Analyze data"

    def test_sync_missing_required_fields(self):
        """Test sync handler with missing required fields."""

        class Handler:
            @validate_body(["field1", "field2"])
            def process(self, request):
                return {"success": True}

        handler = Handler()
        request = MockRequest(_json_data={"field1": "value"})

        result = handler.process(request)
        assert isinstance(result, HandlerResult)
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "field2" in body["error"]

    def test_sync_invalid_json(self):
        """Test sync handler with invalid JSON body."""

        class Handler:
            @validate_body(["field"])
            def process(self, request):
                return {"success": True}

        handler = Handler()
        request = MockRequest(_json_error=ValueError("Invalid JSON"))

        result = handler.process(request)
        assert isinstance(result, HandlerResult)
        assert result.status_code == 400

    def test_sync_uses_handler_error_response(self):
        """Test that sync handler uses self.error_response if available."""

        class HandlerWithErrorMethod:
            def error_response(self, message: str, status: int = 400) -> dict:
                return {"custom_error": message, "code": status}

            @validate_body(["required_field"])
            def process(self, request):
                return {"success": True}

        handler = HandlerWithErrorMethod()
        request = MockRequest(_json_data={})

        result = handler.process(request)
        assert result["custom_error"] == "Missing required fields: required_field"

    def test_empty_required_fields(self):
        """Test with empty required fields list."""

        class Handler:
            @validate_body([])
            async def process(self, request):
                return {"success": True}

        handler = Handler()
        request = MockAsyncRequest(_json_data={})

        result = asyncio.run(handler.process(request))
        assert result["success"] is True

    def test_preserves_additional_args_kwargs(self):
        """Test that additional arguments are preserved."""

        class Handler:
            @validate_body(["field"])
            async def process(self, request, extra_arg, kwarg1=None):
                body = await request.json()
                return {
                    "field": body["field"],
                    "extra": extra_arg,
                    "kwarg": kwarg1,
                }

        handler = Handler()
        request = MockAsyncRequest(_json_data={"field": "value"})

        result = asyncio.run(handler.process(request, "extra_value", kwarg1="kwarg_value"))
        assert result["field"] == "value"
        assert result["extra"] == "extra_value"
        assert result["kwarg"] == "kwarg_value"

    def test_multiple_missing_fields_listed(self):
        """Test that all missing fields are listed in error."""

        class Handler:
            @validate_body(["a", "b", "c", "d"])
            async def process(self, request):
                return {"success": True}

        handler = Handler()
        request = MockAsyncRequest(_json_data={"a": 1})

        result = asyncio.run(handler.process(request))
        assert isinstance(result, HandlerResult)
        body = json.loads(result.body)
        # All missing fields should be mentioned
        assert "b" in body["error"]
        assert "c" in body["error"]
        assert "d" in body["error"]

    def test_type_error_in_json_parse(self):
        """Test handling of TypeError during JSON parsing."""

        class Handler:
            @validate_body(["field"])
            async def process(self, request):
                return {"success": True}

        handler = Handler()
        request = MockAsyncRequest(_json_error=TypeError("Cannot parse"))

        result = asyncio.run(handler.process(request))
        assert isinstance(result, HandlerResult)
        assert result.status_code == 400

    def test_sync_none_body(self):
        """Test sync handler when json() returns None."""

        class Handler:
            @validate_body(["field"])
            def process(self, request):
                return {"success": True}

        handler = Handler()
        request = MockRequest(_json_data=None)

        result = handler.process(request)
        assert isinstance(result, HandlerResult)
        assert result.status_code == 400


# ===========================================================================
# require_quota Tests
# ===========================================================================


class TestRequireQuota:
    """Tests for the require_quota decorator."""

    def test_quota_check_passes(self):
        """Test that quota check passes when under limit."""
        org = MockOrganization(
            id="org-123",
            debates_used_this_month=50,
            is_at_limit=False,
        )
        org.limits.debates_per_month = 100

        user_store = MockUserStore(organizations={"org-123": org})
        user_ctx = MockUserAuthContext(org_id="org-123")

        class Handler:
            def __init__(self):
                self.user_store = user_store

            @require_quota()
            def _create_debate(self, handler, user):
                return {"created": True}

        handler_instance = Handler()
        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.user_store = user_store

        result = handler_instance._create_debate(mock_handler, user=user_ctx)
        assert result == {"created": True}

    def test_quota_exceeded_returns_429(self):
        """Test that exceeding quota returns 429 error."""
        org = MockOrganization(
            id="org-123",
            debates_used_this_month=100,
            is_at_limit=True,
        )
        org.limits.debates_per_month = 100

        user_store = MockUserStore(organizations={"org-123": org})
        user_ctx = MockUserAuthContext(org_id="org-123")

        class Handler:
            def __init__(self):
                self.user_store = user_store

            @require_quota()
            def _create_debate(self, handler, user):
                return {"created": True}

        handler_instance = Handler()
        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.user_store = user_store

        result = handler_instance._create_debate(mock_handler, user=user_ctx)
        assert isinstance(result, HandlerResult)
        assert result.status_code == 429
        body = json.loads(result.body)
        assert body["code"] == "quota_exceeded"
        assert "upgrade_url" in body

    def test_quota_insufficient_for_batch(self):
        """Test quota check for batch operations."""
        org = MockOrganization(
            id="org-123",
            debates_used_this_month=95,
            is_at_limit=False,
        )
        org.limits.debates_per_month = 100

        user_store = MockUserStore(organizations={"org-123": org})
        user_ctx = MockUserAuthContext(org_id="org-123")

        class Handler:
            def __init__(self):
                self.user_store = user_store

            @require_quota(debate_count=10)
            def _batch_create(self, handler, user):
                return {"created": 10}

        handler_instance = Handler()
        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.user_store = user_store

        result = handler_instance._batch_create(mock_handler, user=user_ctx)
        assert isinstance(result, HandlerResult)
        assert result.status_code == 429
        body = json.loads(result.body)
        assert body["code"] == "quota_insufficient"
        assert body["remaining"] == 5
        assert body["requested"] == 10

    def test_usage_incremented_on_success(self):
        """Test that usage is incremented on successful operation."""
        org = MockOrganization(
            id="org-123",
            debates_used_this_month=50,
            is_at_limit=False,
        )
        org.limits.debates_per_month = 100

        user_store = MockUserStore(organizations={"org-123": org})
        user_ctx = MockUserAuthContext(org_id="org-123")

        class Handler:
            def __init__(self):
                self.user_store = user_store

            @require_quota()
            def _create_debate(self, handler, user):
                return MagicMock(status_code=201)

        handler_instance = Handler()
        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.user_store = user_store

        handler_instance._create_debate(mock_handler, user=user_ctx)
        assert ("org-123", 1) in user_store.usage_increments

    def test_usage_not_incremented_on_failure(self):
        """Test that usage is NOT incremented on failed operation."""
        org = MockOrganization(
            id="org-123",
            debates_used_this_month=50,
            is_at_limit=False,
        )
        org.limits.debates_per_month = 100

        user_store = MockUserStore(organizations={"org-123": org})
        user_ctx = MockUserAuthContext(org_id="org-123")

        class Handler:
            def __init__(self):
                self.user_store = user_store

            @require_quota()
            def _create_debate(self, handler, user):
                return MagicMock(status_code=400)

        handler_instance = Handler()
        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.user_store = user_store

        handler_instance._create_debate(mock_handler, user=user_ctx)
        assert len(user_store.usage_increments) == 0

    def test_no_org_id_skips_quota_check(self):
        """Test that quota check is skipped when no org_id."""
        user_ctx = MockUserAuthContext(org_id=None)

        class Handler:
            @require_quota()
            def _create_debate(self, handler, user):
                return {"created": True}

        handler_instance = Handler()
        mock_handler = MagicMock()
        mock_handler.headers = {}

        result = handler_instance._create_debate(mock_handler, user=user_ctx)
        assert result == {"created": True}

    @patch("aragora.server.handlers.api_decorators.extract_user_from_request")
    def test_authentication_required_when_no_user(self, mock_extract):
        """Test authentication is required when no user context provided."""
        mock_extract.return_value = MockUserAuthContext(
            is_authenticated=False, error_reason="Token expired"
        )

        class Handler:
            user_store = None

            @require_quota()
            def _create_debate(self, handler):
                return {"created": True}

        handler_instance = Handler()
        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.user_store = None

        result = handler_instance._create_debate(mock_handler)
        assert isinstance(result, HandlerResult)
        assert result.status_code == 401

    def test_no_handler_returns_401(self):
        """Test that missing handler returns 401 error."""

        class Handler:
            @require_quota()
            def _create_debate(self):
                return {"created": True}

        handler_instance = Handler()
        result = handler_instance._create_debate()
        assert isinstance(result, HandlerResult)
        assert result.status_code == 401

    def test_quota_check_failure_does_not_block(self):
        """Test that quota check failure (exception) does not block the request."""
        org = MockOrganization(id="org-123")

        user_store = MagicMock()
        user_store.get_organization_by_id = MagicMock(side_effect=OSError("Database error"))
        user_ctx = MockUserAuthContext(org_id="org-123")

        class Handler:
            def __init__(self):
                self.user_store = user_store

            @require_quota()
            def _create_debate(self, handler, user):
                return {"created": True}

        handler_instance = Handler()
        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.user_store = user_store

        # Should still succeed even though quota check failed
        result = handler_instance._create_debate(mock_handler, user=user_ctx)
        assert result == {"created": True}

    def test_custom_debate_count(self):
        """Test require_quota with custom debate_count."""
        org = MockOrganization(
            id="org-123",
            debates_used_this_month=80,
            is_at_limit=False,
        )
        org.limits.debates_per_month = 100

        user_store = MockUserStore(organizations={"org-123": org})
        user_ctx = MockUserAuthContext(org_id="org-123")

        class Handler:
            def __init__(self):
                self.user_store = user_store

            @require_quota(debate_count=5)
            def _batch_create(self, handler, user):
                return MagicMock(status_code=200)

        handler_instance = Handler()
        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.user_store = user_store

        handler_instance._batch_create(mock_handler, user=user_ctx)
        # Should increment by 5
        assert ("org-123", 5) in user_store.usage_increments


# ===========================================================================
# Decorator Composition Tests
# ===========================================================================


class TestDecoratorComposition:
    """Tests for stacking multiple decorators."""

    def test_api_endpoint_with_validate_body(self):
        """Test api_endpoint stacked with validate_body."""

        class Handler:
            @api_endpoint(method="POST", path="/api/items", summary="Create item")
            @validate_body(["name"])
            async def create_item(self, request):
                body = await request.json()
                return {"name": body["name"]}

        handler = Handler()

        # Check metadata is preserved
        assert hasattr(handler.create_item, "_api_metadata")
        assert handler.create_item._api_metadata["method"] == "POST"

        # Check validation still works
        valid_request = MockAsyncRequest(_json_data={"name": "Test Item"})
        result = asyncio.run(handler.create_item(valid_request))
        assert result["name"] == "Test Item"

        # Check validation fails correctly
        invalid_request = MockAsyncRequest(_json_data={})
        result = asyncio.run(handler.create_item(invalid_request))
        assert isinstance(result, HandlerResult)
        assert result.status_code == 400

    @patch("aragora.server.handlers.api_decorators._rate_limit")
    def test_all_decorators_stacked(self, mock_rate_limit):
        """Test all decorators stacked together."""
        mock_decorator = MagicMock(side_effect=lambda f: f)
        mock_rate_limit.return_value = mock_decorator

        class Handler:
            @api_endpoint(method="POST", path="/api/debates", summary="Create debate")
            @rate_limit(requests_per_minute=30)
            @validate_body(["task", "agents"])
            async def create_debate(self, request, user=None):
                body = await request.json()
                return {"task": body["task"]}

        handler = Handler()

        # Verify metadata
        # Note: With multiple decorators, the outermost decorator's result has the metadata
        assert hasattr(handler.create_debate, "_api_metadata")

        # Test valid request
        request = MockAsyncRequest(_json_data={"task": "Test", "agents": ["claude"]})
        result = asyncio.run(handler.create_debate(request))
        assert result["task"] == "Test"

    def test_decorator_order_matters_for_validation(self):
        """Test that decorator order affects behavior."""

        class Handler:
            @validate_body(["field1"])
            @api_endpoint(method="POST", path="/test")
            async def handler1(self, request):
                return {"ok": True}

            @api_endpoint(method="POST", path="/test")
            @validate_body(["field2"])
            async def handler2(self, request):
                return {"ok": True}

        handler = Handler()

        # Both should have metadata (api_endpoint preserves it)
        # handler1: api_endpoint is inner, validate_body is outer
        # handler2: validate_body is inner, api_endpoint is outer

        # Test that handler2 still validates
        invalid_request = MockAsyncRequest(_json_data={})
        result = asyncio.run(handler.handler2(invalid_request))
        assert isinstance(result, HandlerResult)
        assert result.status_code == 400


# ===========================================================================
# Edge Cases and Error Handling
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_validate_body_with_nested_fields(self):
        """Test validate_body with nested field names (dotted)."""

        # The decorator only checks top-level fields
        class Handler:
            @validate_body(["user.name"])
            async def process(self, request):
                return {"ok": True}

        handler = Handler()
        request = MockAsyncRequest(_json_data={"user": {"name": "Test"}})

        result = asyncio.run(handler.process(request))
        # Should fail because "user.name" is not a top-level key
        assert isinstance(result, HandlerResult)
        assert result.status_code == 400

    def test_validate_body_with_null_value(self):
        """Test that field with null value is considered present."""

        class Handler:
            @validate_body(["field"])
            async def process(self, request):
                body = await request.json()
                return {"field": body["field"]}

        handler = Handler()
        request = MockAsyncRequest(_json_data={"field": None})

        result = asyncio.run(handler.process(request))
        assert result["field"] is None

    def test_validate_body_with_empty_string(self):
        """Test that field with empty string is considered present."""

        class Handler:
            @validate_body(["field"])
            async def process(self, request):
                body = await request.json()
                return {"field": body["field"]}

        handler = Handler()
        request = MockAsyncRequest(_json_data={"field": ""})

        result = asyncio.run(handler.process(request))
        assert result["field"] == ""

    def test_validate_body_with_false_value(self):
        """Test that field with False value is considered present."""

        class Handler:
            @validate_body(["enabled"])
            async def process(self, request):
                body = await request.json()
                return {"enabled": body["enabled"]}

        handler = Handler()
        request = MockAsyncRequest(_json_data={"enabled": False})

        result = asyncio.run(handler.process(request))
        assert result["enabled"] is False

    def test_validate_body_with_zero_value(self):
        """Test that field with 0 value is considered present."""

        class Handler:
            @validate_body(["count"])
            async def process(self, request):
                body = await request.json()
                return {"count": body["count"]}

        handler = Handler()
        request = MockAsyncRequest(_json_data={"count": 0})

        result = asyncio.run(handler.process(request))
        assert result["count"] == 0

    def test_api_endpoint_with_empty_path(self):
        """Test api_endpoint with empty path."""

        @api_endpoint(method="GET", path="")
        def handler():
            pass

        assert handler._api_metadata["path"] == ""

    def test_api_endpoint_with_special_characters_in_path(self):
        """Test api_endpoint with special characters in path."""

        @api_endpoint(method="GET", path="/api/v1/items/{id}/sub-items/{sub_id}")
        def handler():
            pass

        assert handler._api_metadata["path"] == "/api/v1/items/{id}/sub-items/{sub_id}"

    def test_require_quota_extracts_handler_from_args(self):
        """Test that require_quota can extract handler from positional args."""
        org = MockOrganization(
            id="org-123",
            debates_used_this_month=50,
            is_at_limit=False,
        )
        org.limits.debates_per_month = 100

        user_store = MockUserStore(organizations={"org-123": org})
        user_ctx = MockUserAuthContext(org_id="org-123")

        class MyHandler:
            def __init__(self):
                self.user_store = user_store

            @require_quota()
            def _create(self, request, user):
                return {"ok": True}

        handler_instance = MyHandler()
        # Pass handler-like object as first positional arg
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer token"}
        mock_request.user_store = user_store

        result = handler_instance._create(mock_request, user=user_ctx)
        assert result == {"ok": True}


# ===========================================================================
# Module Exports Tests
# ===========================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_exist(self):
        """Test that all exported names exist in module."""
        from aragora.server.handlers import api_decorators

        expected_exports = ["api_endpoint", "rate_limit", "validate_body", "require_quota"]

        for name in expected_exports:
            assert hasattr(api_decorators, name), f"Missing export: {name}"

    def test_all_list_matches_exports(self):
        """Test that __all__ list matches actual exports."""
        from aragora.server.handlers import api_decorators

        assert set(api_decorators.__all__) == {
            "api_endpoint",
            "rate_limit",
            "validate_body",
            "require_quota",
        }


# ===========================================================================
# extract_user_from_request Tests
# ===========================================================================


class TestExtractUserFromRequest:
    """Tests for the extract_user_from_request proxy function."""

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_delegates_to_jwt_auth(self, mock_jwt_extract):
        """Test that extract_user_from_request delegates to jwt_auth."""
        from aragora.server.handlers.api_decorators import extract_user_from_request

        mock_handler = MagicMock()
        mock_user_store = MagicMock()
        expected_user = MockUserAuthContext(user_id="test-user")
        mock_jwt_extract.return_value = expected_user

        result = extract_user_from_request(mock_handler, mock_user_store)

        mock_jwt_extract.assert_called_once_with(mock_handler, mock_user_store)
        assert result == expected_user

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_passes_none_user_store(self, mock_jwt_extract):
        """Test that None user_store is passed through."""
        from aragora.server.handlers.api_decorators import extract_user_from_request

        mock_handler = MagicMock()
        mock_jwt_extract.return_value = MockUserAuthContext()

        extract_user_from_request(mock_handler, None)

        mock_jwt_extract.assert_called_once_with(mock_handler, None)


# ===========================================================================
# Additional require_quota Tests
# ===========================================================================


class TestRequireQuotaHandlerExtraction:
    """Tests for handler extraction in require_quota."""

    def test_handler_from_kwargs(self):
        """Test handler extracted from kwargs."""
        org = MockOrganization(
            id="org-123",
            debates_used_this_month=10,
            is_at_limit=False,
        )
        org.limits.debates_per_month = 100

        user_store = MockUserStore(organizations={"org-123": org})
        user_ctx = MockUserAuthContext(org_id="org-123")

        class MyClass:
            @require_quota()
            def method(self, handler, user):
                return {"ok": True}

        instance = MyClass()
        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.user_store = user_store

        result = instance.method(handler=mock_handler, user=user_ctx)
        assert result == {"ok": True}

    def test_handler_from_positional_args_with_headers_attribute(self):
        """Test handler extracted from positional args by checking headers attr."""
        org = MockOrganization(
            id="org-123",
            debates_used_this_month=10,
            is_at_limit=False,
        )
        org.limits.debates_per_month = 100

        user_store = MockUserStore(organizations={"org-123": org})
        user_ctx = MockUserAuthContext(org_id="org-123")

        class MyClass:
            @require_quota()
            def method(self, some_arg, handler_arg, user):
                return {"ok": True}

        instance = MyClass()
        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.user_store = user_store

        # handler_arg has headers, so it should be found
        result = instance.method("string_arg", mock_handler, user=user_ctx)
        assert result == {"ok": True}

    def test_user_store_from_handler_class(self):
        """Test user_store extracted from handler.__class__.user_store."""
        org = MockOrganization(
            id="org-123",
            debates_used_this_month=10,
            is_at_limit=False,
        )
        org.limits.debates_per_month = 100

        the_user_store = MockUserStore(organizations={"org-123": org})
        user_ctx = MockUserAuthContext(org_id="org-123")

        class MyClass:
            @require_quota()
            def method(self, handler, user):
                return {"success": True}

        instance = MyClass()

        # Handler without instance user_store but with class-level one
        mock_handler = MagicMock()
        mock_handler.headers = {}
        # Remove instance attribute so it falls back to class attribute
        del mock_handler.user_store
        mock_handler.__class__.user_store = the_user_store

        result = instance.method(mock_handler, user=user_ctx)
        assert result == {"success": True}

    def test_increment_usage_exception_logged_but_not_raised(self):
        """Test that increment_usage exception is logged but doesn't fail request."""
        org = MockOrganization(
            id="org-123",
            debates_used_this_month=10,
            is_at_limit=False,
        )
        org.limits.debates_per_month = 100

        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org
        user_store.increment_usage.side_effect = RuntimeError("DB connection failed")
        user_ctx = MockUserAuthContext(org_id="org-123")

        class MyClass:
            @require_quota()
            def method(self, handler, user):
                return MagicMock(status_code=200)

        instance = MyClass()
        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.user_store = user_store

        # Should not raise, should return the result
        result = instance.method(mock_handler, user=user_ctx)
        assert result.status_code == 200

    def test_org_not_found_skips_quota_check(self):
        """Test that when org is not found, quota check is skipped."""
        user_store = MockUserStore(organizations={})  # Empty, org not found
        user_ctx = MockUserAuthContext(org_id="nonexistent-org")

        class MyClass:
            @require_quota()
            def method(self, handler, user):
                return {"created": True}

        instance = MyClass()
        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.user_store = user_store

        result = instance.method(mock_handler, user=user_ctx)
        assert result == {"created": True}

    def test_no_user_store_with_get_organization_by_id(self):
        """Test when user_store exists but doesn't have get_organization_by_id."""
        user_ctx = MockUserAuthContext(org_id="org-123")

        class MyClass:
            @require_quota()
            def method(self, handler, user):
                return {"created": True}

        instance = MyClass()
        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.user_store = MagicMock(spec=[])  # No get_organization_by_id

        result = instance.method(mock_handler, user=user_ctx)
        assert result == {"created": True}


class TestRequireQuotaResponseHandling:
    """Tests for require_quota response handling."""

    def test_result_without_status_code_assumes_200(self):
        """Test that results without status_code are treated as successful."""
        org = MockOrganization(
            id="org-123",
            debates_used_this_month=10,
            is_at_limit=False,
        )
        org.limits.debates_per_month = 100

        user_store = MockUserStore(organizations={"org-123": org})
        user_ctx = MockUserAuthContext(org_id="org-123")

        class MyClass:
            @require_quota()
            def method(self, handler, user):
                return {"no_status_code": True}  # Dict, no status_code attr

        instance = MyClass()
        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.user_store = user_store

        result = instance.method(mock_handler, user=user_ctx)
        assert result == {"no_status_code": True}
        # Usage should be incremented (defaults to 200 = success)
        assert ("org-123", 1) in user_store.usage_increments

    def test_result_none_assumes_200(self):
        """Test that None result is treated as successful."""
        org = MockOrganization(
            id="org-123",
            debates_used_this_month=10,
            is_at_limit=False,
        )
        org.limits.debates_per_month = 100

        user_store = MockUserStore(organizations={"org-123": org})
        user_ctx = MockUserAuthContext(org_id="org-123")

        class MyClass:
            @require_quota()
            def method(self, handler, user):
                return None

        instance = MyClass()
        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.user_store = user_store

        result = instance.method(mock_handler, user=user_ctx)
        assert result is None
        # Usage should be incremented
        assert ("org-123", 1) in user_store.usage_increments

    def test_quota_response_includes_all_fields(self):
        """Test that quota exceeded response includes all required fields."""
        org = MockOrganization(
            id="org-123",
            debates_used_this_month=100,
            is_at_limit=True,
        )
        org.limits.debates_per_month = 100
        org.tier = MagicMock(value="starter")

        user_store = MockUserStore(organizations={"org-123": org})
        user_ctx = MockUserAuthContext(org_id="org-123")

        class MyClass:
            @require_quota()
            def method(self, handler, user):
                return {"created": True}

        instance = MyClass()
        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.user_store = user_store

        result = instance.method(mock_handler, user=user_ctx)
        assert isinstance(result, HandlerResult)
        body = json.loads(result.body)

        assert body["error"] == "Monthly debate quota exceeded"
        assert body["code"] == "quota_exceeded"
        assert body["limit"] == 100
        assert body["used"] == 100
        assert body["remaining"] == 0
        assert body["tier"] == "starter"
        assert body["upgrade_url"] == "/pricing"
        assert "message" in body


class TestRequireQuotaAuthentication:
    """Tests for require_quota authentication handling."""

    @patch("aragora.server.handlers.api_decorators.extract_user_from_request")
    def test_authenticates_when_user_not_in_kwargs(self, mock_extract):
        """Test that authentication happens when user not provided."""
        mock_user = MockUserAuthContext(
            is_authenticated=True,
            user_id="authenticated-user",
            org_id=None,
        )
        mock_extract.return_value = mock_user

        class MyClass:
            @require_quota()
            def method(self, handler, user=None):
                return {"ok": True}

        instance = MyClass()
        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.user_store = None

        result = instance.method(handler=mock_handler)
        assert result == {"ok": True}
        mock_extract.assert_called_once()

    @patch("aragora.server.handlers.api_decorators.extract_user_from_request")
    def test_returns_401_with_error_reason(self, mock_extract):
        """Test that 401 includes error_reason from auth context."""
        mock_user = MockUserAuthContext(
            is_authenticated=False,
            error_reason="Token has expired",
        )
        mock_extract.return_value = mock_user

        class MyClass:
            @require_quota()
            def method(self, handler):
                return {"ok": True}

        instance = MyClass()
        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.user_store = None

        result = instance.method(handler=mock_handler)
        assert isinstance(result, HandlerResult)
        assert result.status_code == 401
        body = json.loads(result.body)
        assert "Token has expired" in body["error"]


# ===========================================================================
# Additional validate_body Tests
# ===========================================================================


class TestValidateBodySyncAdditional:
    """Additional tests for sync validate_body behavior."""

    def test_sync_json_not_callable(self):
        """Test sync handler when request.json is not callable."""

        class Handler:
            @validate_body(["field"])
            def process(self, request):
                return {"success": True}

        handler = Handler()

        class RequestWithNonCallableJson:
            json = "not a method"
            headers = {}

        request = RequestWithNonCallableJson()

        result = handler.process(request)
        assert isinstance(result, HandlerResult)
        assert result.status_code == 400

    def test_sync_missing_json_attribute(self):
        """Test sync handler when request doesn't have json attribute."""

        class Handler:
            @validate_body(["field"])
            def process(self, request):
                return {"success": True}

        handler = Handler()

        class RequestWithoutJson:
            headers = {}

        request = RequestWithoutJson()

        result = handler.process(request)
        assert isinstance(result, HandlerResult)
        assert result.status_code == 400


class TestValidateBodyAsyncAdditional:
    """Additional tests for async validate_body behavior."""

    def test_async_with_extra_positional_args(self):
        """Test async handler preserves extra positional args."""

        class Handler:
            @validate_body(["field"])
            async def process(self, request, arg1, arg2):
                body = await request.json()
                return {"field": body["field"], "arg1": arg1, "arg2": arg2}

        handler = Handler()
        request = MockAsyncRequest(_json_data={"field": "value"})

        result = asyncio.run(handler.process(request, "first", "second"))
        assert result["field"] == "value"
        assert result["arg1"] == "first"
        assert result["arg2"] == "second"

    def test_async_json_decode_error_message(self):
        """Test async JSON decode error includes appropriate message."""

        class Handler:
            @validate_body(["field"])
            async def process(self, request):
                return {"success": True}

        handler = Handler()
        request = MockAsyncRequest(_json_error=json.JSONDecodeError("msg", "doc", 0))

        result = asyncio.run(handler.process(request))
        body = json.loads(result.body)
        assert "Invalid JSON body" in body["error"]


# ===========================================================================
# Additional rate_limit Tests
# ===========================================================================


class TestRateLimitAsyncHandling:
    """Tests for rate_limit async wrapper behavior."""

    @patch("aragora.server.handlers.api_decorators._rate_limit")
    def test_async_wrapper_awaits_awaitable_result(self, mock_rate_limit):
        """Test that async wrapper properly awaits awaitable results."""

        # Create a decorated function that returns an awaitable
        call_count = {"value": 0}

        def make_decorator(func):
            def decorated(*args, **kwargs):
                call_count["value"] += 1
                coro = func(*args, **kwargs)
                return coro  # Return the coroutine directly

            return decorated

        mock_rate_limit.return_value = make_decorator

        @rate_limit(requests_per_minute=30)
        async def async_handler():
            return {"async": True}

        result = asyncio.run(async_handler())
        assert result == {"async": True}
        assert call_count["value"] == 1

    @patch("aragora.server.handlers.api_decorators._rate_limit")
    def test_async_wrapper_handles_non_awaitable_from_decorated(self, mock_rate_limit):
        """Test async wrapper handles non-awaitable results from decorated func."""

        def make_decorator(func):
            def decorated(*args, **kwargs):
                # Return non-awaitable directly
                return {"direct_result": True}

            return decorated

        mock_rate_limit.return_value = make_decorator

        @rate_limit(requests_per_minute=30)
        async def async_handler():
            return {"async": True}

        result = asyncio.run(async_handler())
        assert result == {"direct_result": True}


# ===========================================================================
# Decorator Stacking Additional Tests
# ===========================================================================


class TestDecoratorStackingAdditional:
    """Additional tests for decorator stacking scenarios."""

    @patch("aragora.server.handlers.api_decorators._rate_limit")
    def test_rate_limit_with_validate_body(self, mock_rate_limit):
        """Test rate_limit stacked with validate_body."""
        mock_decorator = MagicMock(side_effect=lambda f: f)
        mock_rate_limit.return_value = mock_decorator

        class Handler:
            @rate_limit(requests_per_minute=60)
            @validate_body(["name"])
            async def create(self, request):
                body = await request.json()
                return {"name": body["name"]}

        handler = Handler()

        # Valid request
        valid_request = MockAsyncRequest(_json_data={"name": "test"})
        result = asyncio.run(handler.create(valid_request))
        assert result["name"] == "test"

        # Invalid request
        invalid_request = MockAsyncRequest(_json_data={})
        result = asyncio.run(handler.create(invalid_request))
        assert isinstance(result, HandlerResult)
        assert result.status_code == 400

    def test_multiple_validate_body_decorators(self):
        """Test behavior with multiple validate_body decorators (not recommended)."""

        class Handler:
            @validate_body(["outer"])
            @validate_body(["inner"])
            async def process(self, request):
                body = await request.json()
                return {"both": True}

        handler = Handler()

        # Request missing outer field fails first
        request = MockAsyncRequest(_json_data={"inner": "value"})
        result = asyncio.run(handler.process(request))
        assert isinstance(result, HandlerResult)
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "outer" in body["error"]

    def test_api_endpoint_preserves_docstring(self):
        """Test that api_endpoint preserves function docstring."""

        @api_endpoint(method="GET", path="/test")
        def documented_handler():
            """This is the docstring."""
            pass

        # The function is returned directly, so docstring should be preserved
        assert documented_handler.__doc__ == """This is the docstring."""


# ===========================================================================
# Fixtures-based Tests (using pytest fixtures)
# ===========================================================================


@pytest.fixture
def mock_authenticated_user():
    """Fixture providing an authenticated user context."""
    return MockUserAuthContext(
        is_authenticated=True,
        user_id="fixture-user-123",
        email="fixture@example.com",
        org_id="fixture-org-123",
    )


@pytest.fixture
def mock_org_with_quota():
    """Fixture providing an organization with available quota."""
    org = MockOrganization(
        id="fixture-org-123",
        debates_used_this_month=50,
        is_at_limit=False,
    )
    org.limits.debates_per_month = 100
    return org


@pytest.fixture
def mock_user_store(mock_org_with_quota):
    """Fixture providing a user store with the mock organization."""
    return MockUserStore(organizations={"fixture-org-123": mock_org_with_quota})


class TestWithFixtures:
    """Tests using pytest fixtures for cleaner setup."""

    def test_quota_with_fixtures(self, mock_authenticated_user, mock_user_store):
        """Test require_quota using fixtures."""

        class Handler:
            @require_quota()
            def create(self, handler, user):
                return {"created": True}

        instance = Handler()
        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.user_store = mock_user_store

        result = instance.create(mock_handler, user=mock_authenticated_user)
        assert result == {"created": True}

    def test_quota_increments_with_fixtures(self, mock_authenticated_user, mock_user_store):
        """Test quota increment using fixtures."""

        class Handler:
            @require_quota(debate_count=3)
            def batch_create(self, handler, user):
                return MagicMock(status_code=200)

        instance = Handler()
        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.user_store = mock_user_store

        instance.batch_create(mock_handler, user=mock_authenticated_user)
        assert ("fixture-org-123", 3) in mock_user_store.usage_increments
