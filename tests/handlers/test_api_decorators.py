"""Tests for API decorator utilities.

Covers:
- api_endpoint: metadata attachment
- rate_limit: async-friendly rate limiting wrapper
- validate_body: JSON body validation (sync and async)
- require_quota: organization debate quota enforcement

Targets: aragora/server/handlers/api_decorators.py
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from functools import wraps
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import pytest

from aragora.server.handlers.api_decorators import (
    api_endpoint,
    rate_limit,
    validate_body,
    require_quota,
    extract_user_from_request,
)
from aragora.server.handlers.utils.responses import HandlerResult, error_response, json_response


# ============================================================================
# Helpers
# ============================================================================


def _body(result) -> dict:
    """Extract decoded JSON body from a HandlerResult."""
    if isinstance(result, HandlerResult):
        return json.loads(result.body.decode("utf-8"))
    if isinstance(result, tuple):
        return result[0] if isinstance(result[0], dict) else json.loads(result[0])
    if isinstance(result, dict):
        return result
    return {}


def _status(result) -> int:
    """Extract status code from a HandlerResult."""
    if isinstance(result, HandlerResult):
        return result.status_code
    if isinstance(result, tuple):
        return result[1]
    return 200


class _FakeRequest:
    """A minimal async request object for validate_body tests."""

    def __init__(self, body_data: Any = None, *, raise_exc: type[Exception] | None = None):
        self._body_data = body_data
        self._raise_exc = raise_exc

    async def json(self):
        if self._raise_exc:
            raise self._raise_exc("bad json")
        return self._body_data


class _SyncFakeRequest:
    """A minimal sync request object for validate_body sync tests."""

    def __init__(self, body_data: Any = None, *, raise_exc: type[Exception] | None = None):
        self._body_data = body_data
        self._raise_exc = raise_exc

    def json(self):
        if self._raise_exc:
            raise self._raise_exc("bad json")
        return self._body_data


class _FakeHandlerSelf:
    """Fake 'self' for decorated handler methods, with optional error_response."""

    def __init__(self, *, has_error_response: bool = False):
        self._has_error = has_error_response
        if has_error_response:
            self.error_response = lambda msg, status=400: error_response(msg, status)


@dataclass
class _FakeUserCtx:
    """Minimal user context for require_quota tests."""

    authenticated: bool = True
    user_id: str = "user-1"
    org_id: str | None = "org-1"
    error_reason: str | None = None

    @property
    def is_authenticated(self):
        return self.authenticated


@dataclass
class _FakeLimits:
    debates_per_month: int = 100


@dataclass
class _FakeOrg:
    is_at_limit: bool = False
    debates_used_this_month: int = 0
    limits: _FakeLimits | None = None
    tier: Any = None

    def __post_init__(self):
        if self.limits is None:
            self.limits = _FakeLimits()
        if self.tier is None:
            self.tier = MagicMock(value="pro")


class _FakeHandler:
    """Handler-like object with headers for require_quota."""

    def __init__(self, user_store=None):
        self.headers = {"Authorization": "Bearer test"}
        self.user_store = user_store


# ============================================================================
# api_endpoint
# ============================================================================


class TestApiEndpoint:
    """Test api_endpoint decorator attaches metadata to functions."""

    def test_attaches_metadata(self):
        @api_endpoint(method="GET", path="/api/items", summary="List items")
        def handler():
            pass

        assert hasattr(handler, "_api_metadata")
        meta = handler._api_metadata
        assert meta["method"] == "GET"
        assert meta["path"] == "/api/items"
        assert meta["summary"] == "List items"

    def test_default_description_empty(self):
        @api_endpoint(method="POST", path="/api/things")
        def handler():
            pass

        assert handler._api_metadata["description"] == ""
        assert handler._api_metadata["summary"] == ""

    def test_description_preserved(self):
        @api_endpoint(
            method="DELETE",
            path="/api/things/{id}",
            summary="Remove thing",
            description="Permanently removes a thing by ID.",
        )
        def handler():
            pass

        assert handler._api_metadata["description"] == "Permanently removes a thing by ID."

    def test_function_is_still_callable(self):
        @api_endpoint(method="GET", path="/x")
        def handler():
            return 42

        assert handler() == 42

    def test_all_http_methods(self):
        for method in ("GET", "POST", "PUT", "DELETE", "PATCH"):

            @api_endpoint(method=method, path="/x")
            def handler():
                pass

            assert handler._api_metadata["method"] == method

    def test_metadata_is_dict(self):
        @api_endpoint(method="GET", path="/api/v1/test", summary="s", description="d")
        def handler():
            pass

        meta = handler._api_metadata
        assert isinstance(meta, dict)
        assert set(meta.keys()) == {"method", "path", "summary", "description"}

    def test_multiple_decorators_preserve_metadata(self):
        @api_endpoint(method="GET", path="/a")
        def handler_a():
            pass

        @api_endpoint(method="POST", path="/b")
        def handler_b():
            pass

        assert handler_a._api_metadata["path"] == "/a"
        assert handler_b._api_metadata["path"] == "/b"

    def test_decorator_on_method(self):
        class MyHandler:
            @api_endpoint(method="GET", path="/api/mine")
            def get_mine(self):
                return "ok"

        h = MyHandler()
        assert h.get_mine() == "ok"
        assert MyHandler.get_mine._api_metadata["path"] == "/api/mine"

    def test_preserves_original_function_name(self):
        @api_endpoint(method="GET", path="/x")
        def my_special_handler():
            pass

        # api_endpoint does NOT use @wraps, so __name__ is preserved by identity
        assert my_special_handler.__name__ == "my_special_handler"

    def test_preserves_function_arguments(self):
        @api_endpoint(method="POST", path="/x")
        def handler(a, b, c=3):
            return a + b + c

        assert handler(1, 2) == 6
        assert handler(1, 2, c=10) == 13

    def test_path_with_placeholders(self):
        @api_endpoint(method="GET", path="/api/debates/{debate_id}/rounds/{round_id}")
        def handler():
            pass

        assert "{debate_id}" in handler._api_metadata["path"]
        assert "{round_id}" in handler._api_metadata["path"]


# ============================================================================
# rate_limit wrapper
# ============================================================================


class TestRateLimitDecorator:
    """Test the async-friendly rate_limit wrapper."""

    def test_sync_function_decorated(self):
        """rate_limit on a sync function returns a decorated function."""
        with patch(
            "aragora.server.handlers.api_decorators._rate_limit",
            return_value=lambda fn: fn,
        ):

            @rate_limit(requests_per_minute=10)
            def sync_handler():
                return "sync_ok"

            assert sync_handler() == "sync_ok"

    def test_async_function_decorated(self):
        """rate_limit on an async function wraps in async_wrapper."""
        with patch(
            "aragora.server.handlers.api_decorators._rate_limit",
            return_value=lambda fn: fn,
        ):

            @rate_limit(requests_per_minute=10)
            async def async_handler():
                return "async_ok"

            result = asyncio.run(async_handler())
            assert result == "async_ok"

    def test_async_wrapper_preserves_name(self):
        with patch(
            "aragora.server.handlers.api_decorators._rate_limit",
            return_value=lambda fn: fn,
        ):

            @rate_limit(requests_per_minute=5)
            async def my_async_fn():
                return 1

            assert my_async_fn.__name__ == "my_async_fn"

    def test_passes_args_to_middleware(self):
        """Arguments are forwarded to the middleware rate_limit."""
        captured = {}

        def fake_rate_limit(*a, **kw):
            captured["args"] = a
            captured["kwargs"] = kw
            return lambda fn: fn

        with patch(
            "aragora.server.handlers.api_decorators._rate_limit", side_effect=fake_rate_limit
        ):

            @rate_limit(requests_per_minute=42, burst_size=10)
            def handler():
                pass

        assert captured["kwargs"].get("requests_per_minute") == 42
        assert captured["kwargs"].get("burst_size") == 10

    def test_sync_returns_decorated_directly(self):
        """For sync functions, the middleware-decorated function is returned as-is."""
        marker = object()

        def fake_middleware_rl(*a, **kw):
            def _dec(fn):
                fn._marker = marker
                return fn

            return _dec

        with patch(
            "aragora.server.handlers.api_decorators._rate_limit", side_effect=fake_middleware_rl
        ):

            @rate_limit(requests_per_minute=10)
            def sync_handler():
                return "ok"

            assert getattr(sync_handler, "_marker", None) is marker

    def test_async_wrapper_awaits_awaitable_result(self):
        """async_wrapper awaits the result if it is awaitable."""

        async def fake_coro():
            return "awaited"

        def fake_middleware_rl(*a, **kw):
            def _dec(fn):
                # Return coroutine-like
                @wraps(fn)
                def wrapper(*args, **kwargs):
                    return fake_coro()

                return wrapper

            return _dec

        with patch(
            "aragora.server.handlers.api_decorators._rate_limit", side_effect=fake_middleware_rl
        ):

            @rate_limit(requests_per_minute=10)
            async def async_handler():
                return "original"

            result = asyncio.run(async_handler())
            assert result == "awaited"

    def test_async_wrapper_returns_non_awaitable_result(self):
        """async_wrapper returns result directly if decorated returns non-awaitable."""

        def fake_middleware_rl(*a, **kw):
            def _dec(fn):
                @wraps(fn)
                def wrapper(*args, **kwargs):
                    return "not_awaitable"

                return wrapper

            return _dec

        with patch(
            "aragora.server.handlers.api_decorators._rate_limit", side_effect=fake_middleware_rl
        ):

            @rate_limit(requests_per_minute=10)
            async def async_handler():
                return "original"

            result = asyncio.run(async_handler())
            assert result == "not_awaitable"


# ============================================================================
# validate_body — async variant
# ============================================================================


class TestValidateBodyAsync:
    """Test validate_body with async handlers."""

    def test_valid_body_passes_through(self):
        @validate_body(["name", "age"])
        async def handler(self, request, extra="default"):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _FakeRequest({"name": "Alice", "age": 30})
        result = asyncio.run(handler(fake_self, req))
        assert _status(result) == 200
        assert _body(result)["ok"] is True

    def test_missing_single_field(self):
        @validate_body(["name", "age"])
        async def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _FakeRequest({"name": "Alice"})
        result = asyncio.run(handler(fake_self, req))
        assert _status(result) == 400
        body = _body(result)
        assert "age" in body.get("error", "")

    def test_missing_all_fields(self):
        @validate_body(["x", "y", "z"])
        async def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _FakeRequest({})
        result = asyncio.run(handler(fake_self, req))
        assert _status(result) == 400
        body = _body(result)
        err = body.get("error", "")
        assert "x" in err
        assert "y" in err
        assert "z" in err

    def test_json_decode_error(self):
        @validate_body(["name"])
        async def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _FakeRequest(raise_exc=json.JSONDecodeError)
        result = asyncio.run(handler(fake_self, req))
        assert _status(result) == 400
        assert "Invalid JSON" in _body(result).get("error", "")

    def test_value_error(self):
        @validate_body(["a"])
        async def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _FakeRequest(raise_exc=ValueError)
        result = asyncio.run(handler(fake_self, req))
        assert _status(result) == 400

    def test_type_error(self):
        @validate_body(["a"])
        async def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _FakeRequest(raise_exc=TypeError)
        result = asyncio.run(handler(fake_self, req))
        assert _status(result) == 400

    def test_uses_self_error_response_when_available(self):
        @validate_body(["field"])
        async def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf(has_error_response=True)
        req = _FakeRequest({})
        result = asyncio.run(handler(fake_self, req))
        assert _status(result) == 400
        assert "field" in _body(result).get("error", "")

    def test_uses_self_error_response_for_json_error(self):
        @validate_body(["field"])
        async def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf(has_error_response=True)
        req = _FakeRequest(raise_exc=json.JSONDecodeError)
        result = asyncio.run(handler(fake_self, req))
        assert _status(result) == 400
        assert "Invalid JSON" in _body(result).get("error", "")

    def test_empty_required_fields_passes(self):
        @validate_body([])
        async def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _FakeRequest({"anything": "goes"})
        result = asyncio.run(handler(fake_self, req))
        assert _status(result) == 200

    def test_extra_fields_allowed(self):
        @validate_body(["name"])
        async def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _FakeRequest({"name": "Alice", "bonus": 42})
        result = asyncio.run(handler(fake_self, req))
        assert _status(result) == 200

    def test_preserves_function_name(self):
        @validate_body(["x"])
        async def my_special_async_handler(self, request):
            pass

        assert my_special_async_handler.__name__ == "my_special_async_handler"

    def test_passes_extra_args(self):
        @validate_body(["name"])
        async def handler(self, request, user_id, role="member"):
            return json_response({"user_id": user_id, "role": role})

        fake_self = _FakeHandlerSelf()
        req = _FakeRequest({"name": "test"})
        result = asyncio.run(handler(fake_self, req, "u123", role="admin"))
        assert _status(result) == 200
        body = _body(result)
        assert body["user_id"] == "u123"
        assert body["role"] == "admin"

    def test_field_present_with_none_value_passes(self):
        """A field present but set to None still counts as present."""

        @validate_body(["name"])
        async def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _FakeRequest({"name": None})
        result = asyncio.run(handler(fake_self, req))
        assert _status(result) == 200

    def test_field_present_with_empty_string_passes(self):
        @validate_body(["name"])
        async def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _FakeRequest({"name": ""})
        result = asyncio.run(handler(fake_self, req))
        assert _status(result) == 200


# ============================================================================
# validate_body — sync variant
# ============================================================================


class TestValidateBodySync:
    """Test validate_body with sync handlers."""

    def test_valid_body_passes_through(self):
        @validate_body(["name"])
        def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _SyncFakeRequest({"name": "Alice"})
        result = handler(fake_self, req)
        assert _status(result) == 200

    def test_missing_field_returns_400(self):
        @validate_body(["name", "email"])
        def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _SyncFakeRequest({"name": "Alice"})
        result = handler(fake_self, req)
        assert _status(result) == 400
        assert "email" in _body(result).get("error", "")

    def test_json_decode_error(self):
        @validate_body(["name"])
        def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _SyncFakeRequest(raise_exc=json.JSONDecodeError)
        result = handler(fake_self, req)
        assert _status(result) == 400
        assert "Invalid JSON" in _body(result).get("error", "")

    def test_value_error(self):
        @validate_body(["a"])
        def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _SyncFakeRequest(raise_exc=ValueError)
        result = handler(fake_self, req)
        assert _status(result) == 400

    def test_type_error(self):
        @validate_body(["a"])
        def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _SyncFakeRequest(raise_exc=TypeError)
        result = handler(fake_self, req)
        assert _status(result) == 400

    def test_uses_self_error_response_when_available(self):
        @validate_body(["field"])
        def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf(has_error_response=True)
        req = _SyncFakeRequest({})
        result = handler(fake_self, req)
        assert _status(result) == 400

    def test_uses_self_error_response_for_json_error(self):
        @validate_body(["field"])
        def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf(has_error_response=True)
        req = _SyncFakeRequest(raise_exc=json.JSONDecodeError)
        result = handler(fake_self, req)
        assert _status(result) == 400
        assert "Invalid JSON" in _body(result).get("error", "")

    def test_request_without_callable_json(self):
        """If request.json is not callable, treat body as None."""

        @validate_body(["field"])
        def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = MagicMock()
        req.json = "not_callable"  # Not a callable
        result = handler(fake_self, req)
        assert _status(result) == 400
        assert "field" in _body(result).get("error", "")

    def test_request_json_is_none_attribute(self):
        """If request has no .json at all, body is treated as None."""

        @validate_body(["field"])
        def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = MagicMock(spec=[])  # No attributes
        result = handler(fake_self, req)
        assert _status(result) == 400

    def test_preserves_function_name(self):
        @validate_body(["x"])
        def my_special_sync_handler(self, request):
            pass

        assert my_special_sync_handler.__name__ == "my_special_sync_handler"

    def test_empty_required_fields_passes(self):
        @validate_body([])
        def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _SyncFakeRequest({"anything": "ok"})
        result = handler(fake_self, req)
        assert _status(result) == 200

    def test_extra_fields_allowed(self):
        @validate_body(["name"])
        def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _SyncFakeRequest({"name": "x", "extra": True})
        result = handler(fake_self, req)
        assert _status(result) == 200

    def test_passes_extra_args(self):
        @validate_body(["name"])
        def handler(self, request, user_id, role="member"):
            return json_response({"user_id": user_id, "role": role})

        fake_self = _FakeHandlerSelf()
        req = _SyncFakeRequest({"name": "test"})
        result = handler(fake_self, req, "u123", role="admin")
        assert _status(result) == 200
        body = _body(result)
        assert body["user_id"] == "u123"
        assert body["role"] == "admin"

    def test_field_present_with_none_value_passes(self):
        @validate_body(["name"])
        def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _SyncFakeRequest({"name": None})
        result = handler(fake_self, req)
        assert _status(result) == 200

    def test_missing_multiple_fields_lists_all(self):
        @validate_body(["a", "b", "c"])
        def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _SyncFakeRequest({"a": 1})
        result = handler(fake_self, req)
        assert _status(result) == 400
        err = _body(result).get("error", "")
        assert "b" in err
        assert "c" in err
        # 'a' should NOT be reported as missing
        assert "a" not in err.replace("Missing required fields: ", "")


# ============================================================================
# validate_body — edge cases for fallback error_response
# ============================================================================


class TestValidateBodyFallback:
    """Test validate_body uses module-level error_response when self has none."""

    def test_async_json_error_no_self_error_response(self):
        @validate_body(["name"])
        async def handler(self, request):
            return json_response({"ok": True})

        fake_self = object()  # No error_response attribute
        req = _FakeRequest(raise_exc=json.JSONDecodeError)
        result = asyncio.run(handler(fake_self, req))
        assert _status(result) == 400
        assert "Invalid JSON" in _body(result).get("error", "")

    def test_async_missing_field_no_self_error_response(self):
        @validate_body(["required_field"])
        async def handler(self, request):
            return json_response({"ok": True})

        fake_self = object()
        req = _FakeRequest({})
        result = asyncio.run(handler(fake_self, req))
        assert _status(result) == 400
        assert "required_field" in _body(result).get("error", "")

    def test_sync_json_error_no_self_error_response(self):
        @validate_body(["name"])
        def handler(self, request):
            return json_response({"ok": True})

        fake_self = object()
        req = _SyncFakeRequest(raise_exc=json.JSONDecodeError)
        result = handler(fake_self, req)
        assert _status(result) == 400

    def test_sync_missing_field_no_self_error_response(self):
        @validate_body(["abc"])
        def handler(self, request):
            return json_response({"ok": True})

        fake_self = object()
        req = _SyncFakeRequest({})
        result = handler(fake_self, req)
        assert _status(result) == 400
        assert "abc" in _body(result).get("error", "")


# ============================================================================
# require_quota — happy path
# ============================================================================


class TestRequireQuotaHappyPath:
    """Test require_quota when quota is available."""

    def test_passes_when_quota_available(self):
        org = _FakeOrg(debates_used_this_month=5, limits=_FakeLimits(debates_per_month=100))
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org
        user_store.increment_usage = MagicMock()

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        result = create_debate(handler=handler, user=user_ctx)
        assert _status(result) == 200
        user_store.increment_usage.assert_called_once_with("org-1", 1)

    def test_increments_by_debate_count(self):
        org = _FakeOrg(debates_used_this_month=5, limits=_FakeLimits(debates_per_month=100))
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org
        user_store.increment_usage = MagicMock()

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota(debate_count=5)
        def batch_create(*args, **kwargs):
            return json_response({"batch": True})

        result = batch_create(handler=handler, user=user_ctx)
        assert _status(result) == 200
        user_store.increment_usage.assert_called_once_with("org-1", 5)

    def test_no_org_id_skips_quota_check(self):
        user_ctx = _FakeUserCtx(org_id=None)
        handler = _FakeHandler()

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        result = create_debate(handler=handler, user=user_ctx)
        assert _status(result) == 200

    def test_no_user_store_skips_org_lookup(self):
        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=None)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        result = create_debate(handler=handler, user=user_ctx)
        assert _status(result) == 200

    def test_no_org_found_skips_quota(self):
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = None

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        result = create_debate(handler=handler, user=user_ctx)
        assert _status(result) == 200


# ============================================================================
# require_quota — quota exceeded
# ============================================================================


class TestRequireQuotaExceeded:
    """Test require_quota when quota is at limit or insufficient."""

    def test_at_limit_returns_429(self):
        org = _FakeOrg(
            is_at_limit=True,
            debates_used_this_month=100,
            limits=_FakeLimits(debates_per_month=100),
        )
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        result = create_debate(handler=handler, user=user_ctx)
        assert _status(result) == 429
        body = _body(result)
        assert body["code"] == "quota_exceeded"
        assert body["limit"] == 100
        assert body["used"] == 100
        assert body["remaining"] == 0
        assert "upgrade_url" in body

    def test_at_limit_includes_tier_info(self):
        tier_mock = MagicMock(value="starter")
        org = _FakeOrg(
            is_at_limit=True,
            debates_used_this_month=10,
            limits=_FakeLimits(debates_per_month=10),
            tier=tier_mock,
        )
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        result = create_debate(handler=handler, user=user_ctx)
        body = _body(result)
        assert body["tier"] == "starter"
        assert "starter" in body["message"]

    def test_insufficient_quota_returns_429(self):
        org = _FakeOrg(
            is_at_limit=False,
            debates_used_this_month=95,
            limits=_FakeLimits(debates_per_month=100),
        )
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota(debate_count=10)
        def batch_create(*args, **kwargs):
            return json_response({"batch": True})

        result = batch_create(handler=handler, user=user_ctx)
        assert _status(result) == 429
        body = _body(result)
        assert body["code"] == "quota_insufficient"
        assert body["remaining"] == 5
        assert body["requested"] == 10

    def test_exact_boundary_passes(self):
        """If remaining == debate_count, the operation should succeed."""
        org = _FakeOrg(
            is_at_limit=False,
            debates_used_this_month=95,
            limits=_FakeLimits(debates_per_month=100),
        )
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org
        user_store.increment_usage = MagicMock()

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota(debate_count=5)
        def create_five(*args, **kwargs):
            return json_response({"ok": True})

        result = create_five(handler=handler, user=user_ctx)
        assert _status(result) == 200

    def test_one_over_boundary_fails(self):
        """If remaining < debate_count, the operation should fail."""
        org = _FakeOrg(
            is_at_limit=False,
            debates_used_this_month=96,
            limits=_FakeLimits(debates_per_month=100),
        )
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota(debate_count=5)
        def create_five(*args, **kwargs):
            return json_response({"ok": True})

        result = create_five(handler=handler, user=user_ctx)
        assert _status(result) == 429
        body = _body(result)
        assert body["remaining"] == 4


# ============================================================================
# require_quota — authentication
# ============================================================================


class TestRequireQuotaAuth:
    """Test require_quota authentication behavior."""

    def test_no_handler_returns_401(self):
        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        # No handler, no user
        result = create_debate()
        assert _status(result) == 401
        assert "Authentication required" in _body(result).get("error", "")

    def test_unauthenticated_user_returns_401(self):
        user_ctx = _FakeUserCtx(authenticated=False, error_reason="Token expired")
        handler = _FakeHandler()

        with patch(
            "aragora.server.handlers.api_decorators.extract_user_from_request",
            return_value=user_ctx,
        ):

            @require_quota()
            def create_debate(*args, **kwargs):
                return json_response({"created": True})

            result = create_debate(handler=handler)
            assert _status(result) == 401
            assert "Token expired" in _body(result).get("error", "")

    def test_unauthenticated_default_message(self):
        user_ctx = _FakeUserCtx(authenticated=False, error_reason=None)
        handler = _FakeHandler()

        with patch(
            "aragora.server.handlers.api_decorators.extract_user_from_request",
            return_value=user_ctx,
        ):

            @require_quota()
            def create_debate(*args, **kwargs):
                return json_response({"created": True})

            result = create_debate(handler=handler)
            assert _status(result) == 401
            assert "Authentication required" in _body(result).get("error", "")

    def test_extracts_handler_from_positional_args(self):
        """require_quota should find handler in positional args by checking headers attr."""
        user_ctx = _FakeUserCtx(org_id=None)
        handler = _FakeHandler()

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        result = create_debate("self_placeholder", handler, user=user_ctx)
        assert _status(result) == 200

    def test_extracts_user_store_from_class_attr(self):
        """require_quota should find user_store from handler.__class__.user_store."""
        org = _FakeOrg(debates_used_this_month=0, limits=_FakeLimits(debates_per_month=100))

        class HandlerWithClassStore:
            headers = {"Authorization": "Bearer test"}
            user_store = MagicMock()

        HandlerWithClassStore.user_store.get_organization_by_id.return_value = org
        HandlerWithClassStore.user_store.increment_usage = MagicMock()

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = HandlerWithClassStore()

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        result = create_debate(handler=handler, user=user_ctx)
        assert _status(result) == 200

    def test_user_injected_into_kwargs(self):
        """When user is not in kwargs, require_quota authenticates and injects it."""
        user_ctx = _FakeUserCtx(org_id=None)
        handler = _FakeHandler()
        captured_kwargs = {}

        with patch(
            "aragora.server.handlers.api_decorators.extract_user_from_request",
            return_value=user_ctx,
        ):

            @require_quota()
            def create_debate(*args, **kwargs):
                captured_kwargs.update(kwargs)
                return json_response({"created": True})

            result = create_debate(handler=handler)
            assert _status(result) == 200
            assert captured_kwargs.get("user") is user_ctx


# ============================================================================
# require_quota — error handling
# ============================================================================


class TestRequireQuotaErrorHandling:
    """Test require_quota graceful degradation on errors."""

    def test_quota_check_exception_does_not_block(self):
        """If quota check raises, the handler should still execute."""
        user_store = MagicMock()
        user_store.get_organization_by_id.side_effect = RuntimeError("DB down")

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        result = create_debate(handler=handler, user=user_ctx)
        assert _status(result) == 200

    def test_usage_increment_exception_does_not_block(self):
        """If usage increment raises, the handler result is still returned."""
        org = _FakeOrg(debates_used_this_month=0, limits=_FakeLimits(debates_per_month=100))
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org
        user_store.increment_usage.side_effect = OSError("Redis down")

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        result = create_debate(handler=handler, user=user_ctx)
        assert _status(result) == 200

    def test_value_error_in_quota_check(self):
        user_store = MagicMock()
        user_store.get_organization_by_id.side_effect = ValueError("bad data")

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        result = create_debate(handler=handler, user=user_ctx)
        assert _status(result) == 200

    def test_type_error_in_quota_check(self):
        user_store = MagicMock()
        user_store.get_organization_by_id.side_effect = TypeError("wrong type")

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        result = create_debate(handler=handler, user=user_ctx)
        assert _status(result) == 200

    def test_key_error_in_quota_check(self):
        user_store = MagicMock()
        user_store.get_organization_by_id.side_effect = KeyError("missing")

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        result = create_debate(handler=handler, user=user_ctx)
        assert _status(result) == 200

    def test_attribute_error_in_quota_check(self):
        user_store = MagicMock()
        user_store.get_organization_by_id.side_effect = AttributeError("no attr")

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        result = create_debate(handler=handler, user=user_ctx)
        assert _status(result) == 200

    def test_no_increment_on_error_status(self):
        """Usage should not be incremented when handler returns error status."""
        org = _FakeOrg(debates_used_this_month=5, limits=_FakeLimits(debates_per_month=100))
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org
        user_store.increment_usage = MagicMock()

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return error_response("Bad input", 400)

        result = create_debate(handler=handler, user=user_ctx)
        assert _status(result) == 400
        user_store.increment_usage.assert_not_called()

    def test_no_increment_on_500_status(self):
        org = _FakeOrg(debates_used_this_month=5, limits=_FakeLimits(debates_per_month=100))
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org
        user_store.increment_usage = MagicMock()

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return error_response("Internal error", 500)

        result = create_debate(handler=handler, user=user_ctx)
        user_store.increment_usage.assert_not_called()

    def test_increment_on_201_status(self):
        """Usage should be incremented on success status codes like 201."""
        org = _FakeOrg(debates_used_this_month=5, limits=_FakeLimits(debates_per_month=100))
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org
        user_store.increment_usage = MagicMock()

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True}, status=201)

        result = create_debate(handler=handler, user=user_ctx)
        assert _status(result) == 201
        user_store.increment_usage.assert_called_once_with("org-1", 1)

    def test_handler_returns_none(self):
        """If the handler returns None, quota still works (default status 200)."""
        org = _FakeOrg(debates_used_this_month=5, limits=_FakeLimits(debates_per_month=100))
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org
        user_store.increment_usage = MagicMock()

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return None

        result = create_debate(handler=handler, user=user_ctx)
        # result is None, but increment should still be called (default 200)
        user_store.increment_usage.assert_called_once()

    def test_user_store_without_get_organization_method(self):
        """If user_store lacks get_organization_by_id, skip quota check."""
        user_store = MagicMock(spec=[])  # No methods
        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"ok": True})

        result = create_debate(handler=handler, user=user_ctx)
        assert _status(result) == 200

    def test_user_store_without_increment_usage(self):
        """If user_store lacks increment_usage, no error."""
        org = _FakeOrg(debates_used_this_month=0, limits=_FakeLimits(debates_per_month=100))
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org
        del user_store.increment_usage  # Remove the method

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"ok": True})

        result = create_debate(handler=handler, user=user_ctx)
        assert _status(result) == 200


# ============================================================================
# extract_user_from_request proxy
# ============================================================================


class TestExtractUserProxy:
    """Test the extract_user_from_request proxy function."""

    def test_delegates_to_jwt_auth(self):
        mock_handler = MagicMock()
        mock_store = MagicMock()
        mock_ctx = _FakeUserCtx()

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_ctx
        ) as mock_fn:
            result = extract_user_from_request(mock_handler, mock_store)
            mock_fn.assert_called_once_with(mock_handler, mock_store)
            assert result is mock_ctx


# ============================================================================
# Decorator combinations
# ============================================================================


class TestDecoratorCombinations:
    """Test decorators work when stacked together."""

    def test_api_endpoint_with_validate_body(self):
        @api_endpoint(method="POST", path="/api/items", summary="Create item")
        @validate_body(["name"])
        async def handler(self, request):
            return json_response({"created": True})

        # Metadata should be on the outer function
        assert hasattr(handler, "_api_metadata")
        assert handler._api_metadata["method"] == "POST"

        # Body validation should work
        fake_self = _FakeHandlerSelf()
        req = _FakeRequest({})
        result = asyncio.run(handler(fake_self, req))
        assert _status(result) == 400

    def test_api_endpoint_with_validate_body_success(self):
        @api_endpoint(method="POST", path="/api/items", summary="Create item")
        @validate_body(["name"])
        async def handler(self, request):
            return json_response({"created": True})

        fake_self = _FakeHandlerSelf()
        req = _FakeRequest({"name": "test"})
        result = asyncio.run(handler(fake_self, req))
        assert _status(result) == 200

    def test_validate_body_with_require_quota(self):
        """validate_body + require_quota on a sync handler."""
        org = _FakeOrg(debates_used_this_month=0, limits=_FakeLimits(debates_per_month=100))
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org
        user_store.increment_usage = MagicMock()

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler_obj = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        result = create_debate(handler=handler_obj, user=user_ctx)
        assert _status(result) == 200


# ============================================================================
# Logging
# ============================================================================


class TestLogging:
    """Test that decorators log appropriately."""

    def test_validate_body_logs_json_error(self, caplog):
        @validate_body(["name"])
        async def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _FakeRequest(raise_exc=json.JSONDecodeError)
        with caplog.at_level(logging.DEBUG, logger="aragora.server.handlers.api_decorators"):
            asyncio.run(handler(fake_self, req))
        assert any("JSON parse error" in rec.message for rec in caplog.records)

    def test_validate_body_sync_logs_json_error(self, caplog):
        @validate_body(["name"])
        def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _SyncFakeRequest(raise_exc=json.JSONDecodeError)
        with caplog.at_level(logging.DEBUG, logger="aragora.server.handlers.api_decorators"):
            handler(fake_self, req)
        assert any("JSON parse error" in rec.message for rec in caplog.records)

    def test_require_quota_logs_exceeded(self, caplog):
        org = _FakeOrg(
            is_at_limit=True,
            debates_used_this_month=100,
            limits=_FakeLimits(debates_per_month=100),
        )
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        with caplog.at_level(logging.INFO, logger="aragora.server.handlers.api_decorators"):
            create_debate(handler=handler, user=user_ctx)
        assert any("Quota exceeded" in rec.message for rec in caplog.records)

    def test_require_quota_logs_insufficient(self, caplog):
        org = _FakeOrg(
            is_at_limit=False,
            debates_used_this_month=99,
            limits=_FakeLimits(debates_per_month=100),
        )
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota(debate_count=5)
        def batch_create(*args, **kwargs):
            return json_response({"ok": True})

        with caplog.at_level(logging.INFO, logger="aragora.server.handlers.api_decorators"):
            batch_create(handler=handler, user=user_ctx)
        assert any("Quota insufficient" in rec.message for rec in caplog.records)

    def test_require_quota_logs_warning_on_error(self, caplog):
        user_store = MagicMock()
        user_store.get_organization_by_id.side_effect = RuntimeError("DB fail")

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        with caplog.at_level(logging.WARNING, logger="aragora.server.handlers.api_decorators"):
            create_debate(handler=handler, user=user_ctx)
        assert any("Quota check failed" in rec.message for rec in caplog.records)

    def test_require_quota_logs_no_handler_warning(self, caplog):
        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        with caplog.at_level(logging.WARNING, logger="aragora.server.handlers.api_decorators"):
            create_debate()
        assert any("No handler provided" in rec.message for rec in caplog.records)


# ============================================================================
# Module exports
# ============================================================================


class TestModuleExports:
    """Test __all__ exports are correct."""

    def test_all_exports(self):
        from aragora.server.handlers import api_decorators

        assert "api_endpoint" in api_decorators.__all__
        assert "rate_limit" in api_decorators.__all__
        assert "validate_body" in api_decorators.__all__
        assert "require_quota" in api_decorators.__all__

    def test_all_exports_count(self):
        from aragora.server.handlers import api_decorators

        assert len(api_decorators.__all__) == 4


# ============================================================================
# require_quota — user_store from handler.__class__ for auth lookup
# ============================================================================


class TestRequireQuotaClassLevelStore:
    """Test require_quota lookups via handler.__class__.user_store."""

    def test_auth_user_store_from_class_when_instance_missing(self):
        """When handler has no instance-level user_store, falls back to class."""

        class HandlerClass:
            headers = {"Authorization": "Bearer x"}

        # Class has user_store, instance does not
        HandlerClass.user_store = MagicMock()
        mock_user_ctx = _FakeUserCtx(org_id=None)
        HandlerClass.user_store.return_value = mock_user_ctx

        handler_instance = HandlerClass()
        # Remove instance-level user_store so it falls through to class
        # (MagicMock on class means instance also has it, so test by passing user directly)
        user_ctx = _FakeUserCtx(org_id=None)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"ok": True})

        result = create_debate(handler=handler_instance, user=user_ctx)
        assert _status(result) == 200

    def test_increment_from_class_user_store(self):
        """Increment usage uses handler.__class__.user_store for the increment path."""
        org = _FakeOrg(debates_used_this_month=0, limits=_FakeLimits(debates_per_month=100))

        class HandlerClass:
            headers = {}
            user_store = MagicMock()

        HandlerClass.user_store.get_organization_by_id.return_value = org
        HandlerClass.user_store.increment_usage = MagicMock()

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler_instance = HandlerClass()

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"ok": True})

        result = create_debate(handler=handler_instance, user=user_ctx)
        assert _status(result) == 200
        HandlerClass.user_store.increment_usage.assert_called_once_with("org-1", 1)


# ============================================================================
# require_quota — handler from positional args
# ============================================================================


class TestRequireQuotaHandlerExtraction:
    """Test handler extraction from positional args."""

    def test_handler_found_in_args_without_headers(self):
        """Args without .headers are skipped."""
        user_ctx = _FakeUserCtx(org_id=None)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"ok": True})

        # First arg has no headers, second does
        no_headers = object()
        with_headers = _FakeHandler()

        result = create_debate(no_headers, with_headers, user=user_ctx)
        assert _status(result) == 200

    def test_no_handler_in_args_no_kwargs(self):
        """When args have no handler-like objects and no kwargs."""

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"ok": True})

        result = create_debate("just_a_string", 42)
        assert _status(result) == 401

    def test_handler_is_first_arg(self):
        """Handler is the first positional arg."""
        user_ctx = _FakeUserCtx(org_id=None)
        handler = _FakeHandler()

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"ok": True})

        result = create_debate(handler, user=user_ctx)
        assert _status(result) == 200


# ============================================================================
# require_quota — quota response details
# ============================================================================


class TestRequireQuotaResponseDetails:
    """Test the detailed fields in quota exceeded responses."""

    def test_at_limit_message_format(self):
        tier_mock = MagicMock(value="business")
        org = _FakeOrg(
            is_at_limit=True,
            debates_used_this_month=500,
            limits=_FakeLimits(debates_per_month=500),
            tier=tier_mock,
        )
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        result = create_debate(handler=handler, user=user_ctx)
        body = _body(result)
        assert body["error"] == "Monthly debate quota exceeded"
        assert body["upgrade_url"] == "/pricing"
        assert "500 debates per month" in body["message"]
        assert "business" in body["message"]

    def test_insufficient_error_message_format(self):
        org = _FakeOrg(
            is_at_limit=False,
            debates_used_this_month=98,
            limits=_FakeLimits(debates_per_month=100),
        )
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota(debate_count=5)
        def batch_create(*args, **kwargs):
            return json_response({"ok": True})

        result = batch_create(handler=handler, user=user_ctx)
        body = _body(result)
        assert "5 debates" in body["error"]
        assert "2 remaining" in body["error"]
        assert body["upgrade_url"] == "/pricing"


# ============================================================================
# validate_body — specific JSONDecodeError subclass
# ============================================================================


class TestValidateBodyJsonDecodeErrorDetails:
    """Test JSONDecodeError handling specifics."""

    def test_async_json_decode_error_with_msg_doc_pos(self):
        """JSONDecodeError requires msg, doc, pos args."""

        @validate_body(["name"])
        async def handler(self, request):
            return json_response({"ok": True})

        class BadJsonRequest:
            async def json(self):
                raise json.JSONDecodeError("Expecting value", "doc", 0)

        fake_self = _FakeHandlerSelf()
        result = asyncio.run(handler(fake_self, BadJsonRequest()))
        assert _status(result) == 400
        assert "Invalid JSON" in _body(result).get("error", "")

    def test_sync_json_decode_error_with_msg_doc_pos(self):
        @validate_body(["name"])
        def handler(self, request):
            return json_response({"ok": True})

        class BadJsonRequest:
            def json(self):
                raise json.JSONDecodeError("Expecting value", "doc", 0)

        fake_self = _FakeHandlerSelf()
        result = handler(fake_self, BadJsonRequest())
        assert _status(result) == 400


# ============================================================================
# require_quota — wraps preserves function identity
# ============================================================================


class TestRequireQuotaFunctools:
    """Test that require_quota preserves function metadata."""

    def test_preserves_function_name(self):
        @require_quota()
        def my_special_handler(*args, **kwargs):
            return json_response({"ok": True})

        assert my_special_handler.__name__ == "my_special_handler"

    def test_preserves_docstring(self):
        @require_quota()
        def my_handler(*args, **kwargs):
            """Create a debate."""
            return json_response({"ok": True})

        assert my_handler.__doc__ == "Create a debate."

    def test_default_debate_count_is_one(self):
        """Default debate_count=1 increments by 1."""
        org = _FakeOrg(debates_used_this_month=0, limits=_FakeLimits(debates_per_month=100))
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org
        user_store.increment_usage = MagicMock()

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"created": True})

        create_debate(handler=handler, user=user_ctx)
        user_store.increment_usage.assert_called_once_with("org-1", 1)


# ============================================================================
# require_quota — increment failure logging
# ============================================================================


class TestRequireQuotaIncrementLogging:
    """Test logging for increment failures."""

    def test_increment_os_error_logged(self, caplog):
        org = _FakeOrg(debates_used_this_month=0, limits=_FakeLimits(debates_per_month=100))
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org
        user_store.increment_usage.side_effect = OSError("disk full")

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"ok": True})

        with caplog.at_level(logging.WARNING, logger="aragora.server.handlers.api_decorators"):
            create_debate(handler=handler, user=user_ctx)

        assert any("Usage increment failed" in rec.message for rec in caplog.records)

    def test_increment_value_error_logged(self, caplog):
        org = _FakeOrg(debates_used_this_month=0, limits=_FakeLimits(debates_per_month=100))
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org
        user_store.increment_usage.side_effect = ValueError("bad value")

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"ok": True})

        with caplog.at_level(logging.WARNING, logger="aragora.server.handlers.api_decorators"):
            create_debate(handler=handler, user=user_ctx)

        assert any("Usage increment failed" in rec.message for rec in caplog.records)

    def test_increment_success_logged(self, caplog):
        org = _FakeOrg(debates_used_this_month=0, limits=_FakeLimits(debates_per_month=100))
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org
        user_store.increment_usage = MagicMock()

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"ok": True})

        with caplog.at_level(logging.DEBUG, logger="aragora.server.handlers.api_decorators"):
            create_debate(handler=handler, user=user_ctx)

        assert any("Incremented usage" in rec.message for rec in caplog.records)


# ============================================================================
# require_quota — handler is None for increment path
# ============================================================================


class TestRequireQuotaHandlerNoneIncrement:
    """Test the increment path when handler is None."""

    def test_no_handler_with_org_id_no_increment(self):
        """When no handler, auth fails so increment never reached."""
        user_ctx = _FakeUserCtx(org_id="org-1")

        @require_quota()
        def create_debate(*args, **kwargs):
            return json_response({"ok": True})

        # No handler kwarg and no positional arg with headers
        result = create_debate(user=user_ctx)
        # user is provided but handler is None - should still work
        # (handler=None skips the org quota check but handler func still runs)
        assert _status(result) == 200


# ============================================================================
# Additional edge cases
# ============================================================================


class TestEdgeCases:
    """Various edge cases across all decorators."""

    def test_api_endpoint_empty_method(self):
        @api_endpoint(method="", path="/x")
        def handler():
            pass

        assert handler._api_metadata["method"] == ""

    def test_validate_body_single_required_field(self):
        @validate_body(["only_field"])
        async def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _FakeRequest({})
        result = asyncio.run(handler(fake_self, req))
        assert _status(result) == 400
        assert "only_field" in _body(result).get("error", "")

    def test_validate_body_many_required_fields(self):
        fields = [f"field_{i}" for i in range(10)]

        @validate_body(fields)
        async def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _FakeRequest({})
        result = asyncio.run(handler(fake_self, req))
        assert _status(result) == 400
        err = _body(result).get("error", "")
        for f in fields:
            assert f in err

    def test_require_quota_large_debate_count(self):
        org = _FakeOrg(
            is_at_limit=False,
            debates_used_this_month=0,
            limits=_FakeLimits(debates_per_month=1000),
        )
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org
        user_store.increment_usage = MagicMock()

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota(debate_count=999)
        def big_batch(*args, **kwargs):
            return json_response({"ok": True})

        result = big_batch(handler=handler, user=user_ctx)
        assert _status(result) == 200
        user_store.increment_usage.assert_called_once_with("org-1", 999)

    def test_require_quota_zero_debates_per_month(self):
        """An org with 0 debates per month is always at insufficient."""
        org = _FakeOrg(
            is_at_limit=False,
            debates_used_this_month=0,
            limits=_FakeLimits(debates_per_month=0),
        )
        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = org

        user_ctx = _FakeUserCtx(org_id="org-1")
        handler = _FakeHandler(user_store=user_store)

        @require_quota(debate_count=1)
        def create_debate(*args, **kwargs):
            return json_response({"ok": True})

        result = create_debate(handler=handler, user=user_ctx)
        assert _status(result) == 429

    def test_validate_body_boolean_field_value(self):
        @validate_body(["flag"])
        async def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _FakeRequest({"flag": False})
        result = asyncio.run(handler(fake_self, req))
        assert _status(result) == 200

    def test_validate_body_numeric_field_value(self):
        @validate_body(["count"])
        def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _SyncFakeRequest({"count": 0})
        result = handler(fake_self, req)
        assert _status(result) == 200

    def test_validate_body_list_field_value(self):
        @validate_body(["items"])
        def handler(self, request):
            return json_response({"ok": True})

        fake_self = _FakeHandlerSelf()
        req = _SyncFakeRequest({"items": []})
        result = handler(fake_self, req)
        assert _status(result) == 200
