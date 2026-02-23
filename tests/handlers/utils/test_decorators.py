"""Comprehensive tests for handler utility decorators.

Tests cover:
- generate_trace_id
- map_exception_to_status / _EXCEPTION_STATUS_MAP
- validate_params
- handle_errors (sync + async)
- auto_error_response
- log_request
- has_permission / PERMISSION_MATRIX
- require_permission (sync + async, with/without handler)
- require_user_auth
- require_auth
- require_storage
- require_feature
- safe_fetch
- with_error_recovery
- deprecated_endpoint
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.utils.decorators import (
    PERMISSION_MATRIX,
    _EXCEPTION_STATUS_MAP,
    _test_user_context_override,
    auto_error_response,
    deprecated_endpoint,
    generate_trace_id,
    handle_errors,
    has_permission,
    log_request,
    map_exception_to_status,
    require_auth,
    require_feature,
    require_permission,
    require_storage,
    require_user_auth,
    safe_fetch,
    validate_params,
    with_error_recovery,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeUserCtx:
    """Minimal stand-in for UserAuthContext."""
    authenticated: bool = True
    user_id: str | None = "u-1"
    email: str | None = "u@example.com"
    org_id: str | None = "org-1"
    role: str = "admin"
    error_reason: str | None = None

    @property
    def is_authenticated(self) -> bool:
        return self.authenticated


class FakeHandler:
    """Simulates an HTTP handler with headers and optional user_store."""

    def __init__(self, headers: dict | None = None, user_store: Any = None):
        self.headers = headers or {}
        if user_store is not None:
            self.user_store = user_store


def _body_json(result: HandlerResult) -> dict:
    """Decode a HandlerResult body to dict."""
    return json.loads(result.body.decode("utf-8"))


# ============================================================================
# generate_trace_id
# ============================================================================

class TestGenerateTraceId:
    def test_returns_string(self):
        tid = generate_trace_id()
        assert isinstance(tid, str)

    def test_length_is_eight(self):
        tid = generate_trace_id()
        assert len(tid) == 8

    def test_unique_ids(self):
        ids = {generate_trace_id() for _ in range(50)}
        assert len(ids) == 50


# ============================================================================
# map_exception_to_status
# ============================================================================

class TestMapExceptionToStatus:
    def test_file_not_found(self):
        assert map_exception_to_status(FileNotFoundError("x")) == 404

    def test_key_error(self):
        assert map_exception_to_status(KeyError("k")) == 404

    def test_value_error(self):
        assert map_exception_to_status(ValueError("v")) == 400

    def test_type_error(self):
        assert map_exception_to_status(TypeError("t")) == 400

    def test_permission_error(self):
        assert map_exception_to_status(PermissionError("p")) == 403

    def test_timeout_error(self):
        assert map_exception_to_status(TimeoutError("t")) == 504

    def test_connection_error(self):
        assert map_exception_to_status(ConnectionError("c")) == 502

    def test_os_error(self):
        assert map_exception_to_status(OSError("o")) == 500

    def test_unknown_exception_uses_default(self):
        assert map_exception_to_status(RuntimeError("r")) == 500

    def test_custom_default(self):
        assert map_exception_to_status(RuntimeError("r"), default=418) == 418

    def test_exception_status_map_not_empty(self):
        assert len(_EXCEPTION_STATUS_MAP) > 40


# ============================================================================
# validate_params
# ============================================================================

class TestValidateParams:
    def test_int_param_extracted(self):
        @validate_params({"page": (int, 1, 1, 100)})
        def handler(query_params=None, **kw):
            return kw

        result = handler(query_params={"page": "5"})
        assert result["page"] == 5

    def test_int_param_clamped_to_min(self):
        @validate_params({"page": (int, 1, 1, 100)})
        def handler(query_params=None, **kw):
            return kw

        result = handler(query_params={"page": "-10"})
        assert result["page"] == 1

    def test_int_param_clamped_to_max(self):
        @validate_params({"page": (int, 1, 1, 100)})
        def handler(query_params=None, **kw):
            return kw

        result = handler(query_params={"page": "999"})
        assert result["page"] == 100

    def test_float_param_extracted(self):
        @validate_params({"threshold": (float, 0.5, 0.0, 1.0)})
        def handler(query_params=None, **kw):
            return kw

        result = handler(query_params={"threshold": "0.7"})
        assert result["threshold"] == pytest.approx(0.7)

    def test_float_clamped_min(self):
        @validate_params({"threshold": (float, 0.5, 0.0, 1.0)})
        def handler(query_params=None, **kw):
            return kw

        result = handler(query_params={"threshold": "-1.0"})
        assert result["threshold"] == pytest.approx(0.0)

    def test_float_clamped_max(self):
        @validate_params({"threshold": (float, 0.5, 0.0, 1.0)})
        def handler(query_params=None, **kw):
            return kw

        result = handler(query_params={"threshold": "5.0"})
        assert result["threshold"] == pytest.approx(1.0)

    def test_bool_param(self):
        @validate_params({"verbose": (bool, False, None, None)})
        def handler(query_params=None, **kw):
            return kw

        assert handler(query_params={"verbose": "true"})["verbose"] is True
        assert handler(query_params={})["verbose"] is False

    def test_str_param(self):
        @validate_params({"name": (str, "", None, None)})
        def handler(query_params=None, **kw):
            return kw

        result = handler(query_params={"name": "hello"})
        assert result["name"] == "hello"

    def test_str_param_truncated(self):
        @validate_params({"name": (str, "", None, 5)})
        def handler(query_params=None, **kw):
            return kw

        result = handler(query_params={"name": "abcdefgh"})
        assert result["name"] == "abcde"

    def test_missing_query_params_uses_defaults(self):
        @validate_params({"page": (int, 42, None, None)})
        def handler(query_params=None, **kw):
            return kw

        result = handler()  # no query_params at all
        assert result["page"] == 42

    def test_unknown_type_falls_back_to_get(self):
        @validate_params({"data": (list, [1, 2], None, None)})
        def handler(query_params=None, **kw):
            return kw

        result = handler(query_params={"data": [3, 4]})
        assert result["data"] == [3, 4]

    def test_positional_query_params(self):
        @validate_params({"n": (int, 0, None, None)})
        def handler(query_params=None, **kw):
            return kw

        result = handler({"n": "7"})  # positional arg
        assert result["n"] == 7

    def test_int_no_bounds(self):
        """Int param with None min/max should not clamp."""
        @validate_params({"x": (int, 0, None, None)})
        def handler(query_params=None, **kw):
            return kw

        assert handler(query_params={"x": "-999"})["x"] == -999

    def test_float_no_bounds(self):
        """Float param with None min/max should not clamp."""
        @validate_params({"x": (float, 0.0, None, None)})
        def handler(query_params=None, **kw):
            return kw

        assert handler(query_params={"x": "999.9"})["x"] == pytest.approx(999.9)

    def test_str_param_none_max(self):
        """String param with no max should not truncate."""
        @validate_params({"s": (str, "", None, None)})
        def handler(query_params=None, **kw):
            return kw

        long_str = "a" * 500
        assert handler(query_params={"s": long_str})["s"] == long_str

    def test_multiple_params(self):
        @validate_params({
            "page": (int, 1, 1, 50),
            "q": (str, "", None, 100),
            "verbose": (bool, False, None, None),
        })
        def handler(query_params=None, **kw):
            return kw

        result = handler(query_params={"page": "3", "q": "test", "verbose": "yes"})
        assert result["page"] == 3
        assert result["q"] == "test"
        assert result["verbose"] is True


# ============================================================================
# handle_errors (sync)
# ============================================================================

class TestHandleErrorsSync:
    def test_success_passes_through(self):
        @handle_errors("test op")
        def good():
            return {"ok": True}

        assert good() == {"ok": True}

    def test_value_error_returns_400(self):
        @handle_errors("test op")
        def bad():
            raise ValueError("nope")

        result = bad()
        assert isinstance(result, HandlerResult)
        assert result.status_code == 400

    def test_file_not_found_returns_404(self):
        @handle_errors("test op")
        def bad():
            raise FileNotFoundError("gone")

        result = bad()
        assert result.status_code == 404

    def test_permission_error_returns_403(self):
        @handle_errors("test op")
        def bad():
            raise PermissionError("denied")

        result = bad()
        assert result.status_code == 403

    def test_timeout_error_returns_504(self):
        @handle_errors("test op")
        def bad():
            raise TimeoutError("slow")

        result = bad()
        assert result.status_code == 504

    def test_unknown_error_uses_default_status(self):
        @handle_errors("test op", default_status=502)
        def bad():
            raise RuntimeError("boom")

        result = bad()
        assert result.status_code == 502

    def test_trace_id_in_header(self):
        @handle_errors("test op")
        def bad():
            raise RuntimeError("boom")

        result = bad()
        assert "X-Trace-Id" in result.headers
        assert len(result.headers["X-Trace-Id"]) == 8

    def test_preserves_function_name(self):
        @handle_errors("test op")
        def my_handler():
            pass

        assert my_handler.__name__ == "my_handler"


# ============================================================================
# handle_errors (async)
# ============================================================================

class TestHandleErrorsAsync:
    def test_async_success(self):
        @handle_errors("async op")
        async def good():
            return {"ok": True}

        result = asyncio.run(good())
        assert result == {"ok": True}

    def test_async_value_error(self):
        @handle_errors("async op")
        async def bad():
            raise ValueError("nope")

        result = asyncio.run(bad())
        assert isinstance(result, HandlerResult)
        assert result.status_code == 400

    def test_async_trace_id_in_header(self):
        @handle_errors("async op")
        async def bad():
            raise RuntimeError("boom")

        result = asyncio.run(bad())
        assert "X-Trace-Id" in result.headers

    def test_async_preserves_function_name(self):
        @handle_errors("async op")
        async def my_async_handler():
            pass

        assert my_async_handler.__name__ == "my_async_handler"


# ============================================================================
# auto_error_response
# ============================================================================

class TestAutoErrorResponse:
    def test_success(self):
        @auto_error_response("test op")
        def good():
            return HandlerResult(200, "application/json", b'{"ok":true}')

        result = good()
        assert result.status_code == 200

    def test_sqlite_operational_error(self):
        @auto_error_response("test op")
        def bad():
            raise sqlite3.OperationalError("db locked")

        result = bad()
        assert result.status_code == 503
        body = _body_json(result)
        assert "unavailable" in body["error"].lower()

    def test_permission_error(self):
        @auto_error_response("test op")
        def bad():
            raise PermissionError("denied")

        result = bad()
        assert result.status_code == 403

    def test_value_error(self):
        @auto_error_response("test op")
        def bad():
            raise ValueError("bad input")

        result = bad()
        assert result.status_code == 400

    def test_generic_error(self):
        @auto_error_response("test op")
        def bad():
            raise RuntimeError("oops")

        result = bad()
        assert result.status_code == 500

    def test_log_level_warning(self, caplog):
        @auto_error_response("test op", log_level="warning")
        def bad():
            raise RuntimeError("oops")

        with caplog.at_level(logging.WARNING):
            bad()
        assert any("Failed to test op" in r.message for r in caplog.records)

    def test_log_level_error(self, caplog):
        @auto_error_response("test op", log_level="error")
        def bad():
            raise RuntimeError("oops")

        with caplog.at_level(logging.ERROR):
            bad()
        assert any("Failed to test op" in r.message for r in caplog.records)

    def test_preserves_function_name(self):
        @auto_error_response("op")
        def my_func():
            pass

        assert my_func.__name__ == "my_func"


# ============================================================================
# log_request
# ============================================================================

class TestLogRequest:
    def test_success_logged(self, caplog):
        @log_request("create debate")
        def handler():
            return HandlerResult(200, "application/json", b'{"ok":true}')

        with caplog.at_level(logging.INFO):
            result = handler()
        assert result.status_code == 200
        assert any("create debate: started" in r.message for r in caplog.records)

    def test_error_status_logged_as_warning(self, caplog):
        @log_request("create debate")
        def handler():
            return {"status": 404, "body": "not found"}

        with caplog.at_level(logging.WARNING):
            handler()
        assert any("404" in r.message for r in caplog.records)

    def test_exception_re_raised(self):
        @log_request("fail op")
        def handler():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            handler()

    def test_none_result(self, caplog):
        @log_request("null op")
        def handler():
            return None

        with caplog.at_level(logging.INFO):
            result = handler()
        assert result is None

    def test_log_response_body(self, caplog):
        @log_request("body op", log_response=True)
        def handler():
            return HandlerResult(200, "application/json", b'{"small": true}')

        with caplog.at_level(logging.DEBUG):
            handler()
        # The response logging happens at DEBUG level
        # Just verify no crash occurs

    def test_preserves_function_name(self):
        @log_request("op")
        def my_handler():
            pass

        assert my_handler.__name__ == "my_handler"


# ============================================================================
# has_permission
# ============================================================================

@pytest.mark.no_auto_auth
class TestHasPermission:
    """Tests for has_permission use the real function (no conftest override)."""

    def test_member_can_read_debates(self):
        assert has_permission("member", "debates:read") is True

    def test_member_cannot_delete_debates(self):
        assert has_permission("member", "debates:delete") is False

    def test_admin_can_delete_debates(self):
        assert has_permission("admin", "debates:delete") is True

    def test_owner_can_do_anything(self):
        assert has_permission("owner", "org:billing") is True
        assert has_permission("owner", "admin:system") is True

    def test_empty_role_denied(self):
        assert has_permission("", "debates:read") is False

    def test_empty_permission_denied(self):
        assert has_permission("admin", "") is False

    def test_none_role_denied(self):
        assert has_permission(None, "debates:read") is False

    def test_none_permission_denied(self):
        assert has_permission("admin", None) is False

    def test_unknown_permission(self):
        assert has_permission("admin", "nonexistent:perm") is False

    def test_wildcard_admin_star(self):
        # "admin:*" only has "owner" role
        assert has_permission("owner", "admin:anything") is True
        assert has_permission("admin", "admin:anything") is False

    def test_permission_matrix_has_expected_keys(self):
        expected_prefixes = {"debates", "agents", "org", "plugins", "admin", "billing"}
        actual_prefixes = {k.split(":")[0] for k in PERMISSION_MATRIX}
        assert expected_prefixes.issubset(actual_prefixes)


# ============================================================================
# require_permission
# ============================================================================

@pytest.mark.no_auto_auth
class TestRequirePermission:
    """Tests for require_permission use real auth checks (no conftest override)."""

    @patch("aragora.server.handlers.utils.decorators._test_user_context_override", None)
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_authenticated_admin_allowed(self, mock_extract):
        mock_extract.return_value = FakeUserCtx(authenticated=True, role="admin")

        @require_permission("debates:create")
        def handler(handler=None, user=None):
            return {"user_id": user.user_id}

        h = FakeHandler(headers={"Authorization": "Bearer tok"})
        result = handler(handler=h)
        assert result == {"user_id": "u-1"}

    @patch("aragora.server.handlers.utils.decorators._test_user_context_override", None)
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_unauthenticated_returns_401(self, mock_extract):
        mock_extract.return_value = FakeUserCtx(
            authenticated=False, error_reason="Token expired"
        )

        @require_permission("debates:create")
        def handler(handler=None, user=None):
            return {"ok": True}

        h = FakeHandler(headers={})
        result = handler(handler=h)
        assert isinstance(result, HandlerResult)
        assert result.status_code == 401

    @patch("aragora.server.handlers.utils.decorators._test_user_context_override", None)
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_insufficient_role_returns_403(self, mock_extract):
        mock_extract.return_value = FakeUserCtx(authenticated=True, role="member")

        @require_permission("debates:delete")
        def handler(handler=None, user=None):
            return {"ok": True}

        h = FakeHandler(headers={"Authorization": "Bearer tok"})
        result = handler(handler=h)
        assert isinstance(result, HandlerResult)
        assert result.status_code == 403

    @patch("aragora.server.handlers.utils.decorators._test_user_context_override", None)
    def test_no_handler_returns_401(self):
        @require_permission("debates:read")
        def handler(user=None):
            return {"ok": True}

        result = handler()  # no handler kwarg, no args with .headers
        assert isinstance(result, HandlerResult)
        assert result.status_code == 401

    @patch("aragora.server.handlers.utils.decorators._test_user_context_override", None)
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_handler_found_via_positional_arg(self, mock_extract):
        mock_extract.return_value = FakeUserCtx(authenticated=True, role="owner")

        @require_permission("org:billing")
        def handler(req, user=None):
            return {"role": user.role}

        h = FakeHandler(headers={"Authorization": "Bearer tok"})
        result = handler(h)  # pass handler as positional arg
        assert result == {"role": "owner"}

    @patch("aragora.server.handlers.utils.decorators._test_user_context_override", None)
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_user_store_on_handler_instance(self, mock_extract):
        mock_extract.return_value = FakeUserCtx(authenticated=True, role="admin")
        store = MagicMock()

        @require_permission("debates:read")
        def handler(handler=None, user=None):
            return {"ok": True}

        h = FakeHandler(headers={"Authorization": "Bearer tok"}, user_store=store)
        result = handler(handler=h)
        assert result == {"ok": True}
        mock_extract.assert_called_once_with(h, store)

    @patch("aragora.server.handlers.utils.decorators._test_user_context_override", None)
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_user_store_on_handler_class(self, mock_extract):
        mock_extract.return_value = FakeUserCtx(authenticated=True, role="admin")
        store = MagicMock()

        class HandlerWithClassStore:
            user_store = store
            headers = {"Authorization": "Bearer tok"}

        @require_permission("debates:read")
        def handler(handler=None, user=None):
            return {"ok": True}

        h = HandlerWithClassStore()
        result = handler(handler=h)
        assert result == {"ok": True}
        mock_extract.assert_called_once_with(h, store)

    @patch("aragora.server.handlers.utils.decorators._test_user_context_override", None)
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_function_without_user_param(self, mock_extract):
        """Functions without 'user' param should still be called, just without user."""
        mock_extract.return_value = FakeUserCtx(authenticated=True, role="admin")

        @require_permission("debates:read")
        def handler(handler=None):
            return {"no_user": True}

        h = FakeHandler(headers={"Authorization": "Bearer tok"})
        result = handler(handler=h)
        assert result == {"no_user": True}

    @patch("aragora.server.handlers.utils.decorators._test_user_context_override", None)
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_async_require_permission(self, mock_extract):
        mock_extract.return_value = FakeUserCtx(authenticated=True, role="admin")

        @require_permission("debates:create")
        async def handler(handler=None, user=None):
            return {"user_id": user.user_id}

        h = FakeHandler(headers={"Authorization": "Bearer tok"})
        result = asyncio.run(handler(handler=h))
        assert result == {"user_id": "u-1"}

    @patch("aragora.server.handlers.utils.decorators._test_user_context_override", None)
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_async_insufficient_permission(self, mock_extract):
        mock_extract.return_value = FakeUserCtx(authenticated=True, role="member")

        @require_permission("debates:delete")
        async def handler(handler=None, user=None):
            return {"ok": True}

        h = FakeHandler(headers={"Authorization": "Bearer tok"})
        result = asyncio.run(handler(handler=h))
        assert isinstance(result, HandlerResult)
        assert result.status_code == 403

    @patch("aragora.server.handlers.utils.decorators._test_user_context_override", None)
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_unauthenticated_no_error_reason(self, mock_extract):
        """When error_reason is None, should use default message."""
        mock_extract.return_value = FakeUserCtx(
            authenticated=False, error_reason=None,
        )

        @require_permission("debates:read")
        def handler(handler=None, user=None):
            return {"ok": True}

        h = FakeHandler(headers={})
        result = handler(handler=h)
        assert result.status_code == 401
        body = _body_json(result)
        assert "Authentication required" in body["error"]


# ============================================================================
# require_permission with test override
# ============================================================================

@pytest.mark.no_auto_auth
class TestRequirePermissionTestOverride:
    def test_override_bypasses_auth_when_no_handler(self):
        import aragora.server.handlers.utils.decorators as dec

        override_ctx = FakeUserCtx(authenticated=True, role="admin")
        original = dec._test_user_context_override
        try:
            dec._test_user_context_override = override_ctx

            @require_permission("debates:read")
            def handler(user=None):
                return {"role": user.role}

            result = handler()  # no handler arg at all
            assert result == {"role": "admin"}
        finally:
            dec._test_user_context_override = original

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_override_used_on_failed_auth(self, mock_extract):
        import aragora.server.handlers.utils.decorators as dec

        mock_extract.return_value = FakeUserCtx(authenticated=False)
        override_ctx = FakeUserCtx(authenticated=True, role="owner")
        original = dec._test_user_context_override
        try:
            dec._test_user_context_override = override_ctx

            @require_permission("org:billing")
            def handler(handler=None, user=None):
                return {"role": user.role}

            h = FakeHandler(headers={})
            result = handler(handler=h)
            assert result == {"role": "owner"}
        finally:
            dec._test_user_context_override = original


# ============================================================================
# require_user_auth
# ============================================================================

@pytest.mark.no_auto_auth
class TestRequireUserAuth:
    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_authenticated_user(self, mock_extract):
        mock_extract.return_value = FakeUserCtx(authenticated=True, user_id="u-42")

        @require_user_auth
        def handler(handler=None, user=None):
            return {"uid": user.user_id}

        h = FakeHandler(headers={"Authorization": "Bearer tok"})
        result = handler(handler=h)
        assert result == {"uid": "u-42"}

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_unauthenticated_returns_401(self, mock_extract):
        mock_extract.return_value = FakeUserCtx(
            authenticated=False, error_reason="Expired"
        )

        @require_user_auth
        def handler(handler=None, user=None):
            return {"ok": True}

        h = FakeHandler(headers={})
        result = handler(handler=h)
        assert isinstance(result, HandlerResult)
        assert result.status_code == 401
        body = _body_json(result)
        assert "Expired" in body["error"]

    def test_no_handler_returns_401(self):
        @require_user_auth
        def handler(user=None):
            return {"ok": True}

        result = handler()
        assert isinstance(result, HandlerResult)
        assert result.status_code == 401

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_handler_via_positional_arg(self, mock_extract):
        mock_extract.return_value = FakeUserCtx(authenticated=True)

        @require_user_auth
        def handler(req, user=None):
            return {"ok": True}

        h = FakeHandler(headers={"Authorization": "Bearer tok"})
        result = handler(h)
        assert result == {"ok": True}

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_unauthenticated_no_error_reason(self, mock_extract):
        mock_extract.return_value = FakeUserCtx(
            authenticated=False, error_reason=None,
        )

        @require_user_auth
        def handler(handler=None, user=None):
            return {"ok": True}

        h = FakeHandler(headers={})
        result = handler(handler=h)
        assert result.status_code == 401
        body = _body_json(result)
        assert "Authentication required" in body["error"]


# ============================================================================
# require_auth (ARAGORA_API_TOKEN)
# ============================================================================

@pytest.mark.no_auto_auth
class TestRequireAuth:
    def test_no_handler_returns_401(self):
        @require_auth
        def handler():
            return {"ok": True}

        result = handler()
        assert isinstance(result, HandlerResult)
        assert result.status_code == 401

    @patch("aragora.server.auth.auth_config")
    def test_no_api_token_configured(self, mock_cfg):
        mock_cfg.api_token = None

        @require_auth
        def handler(handler=None):
            return {"ok": True}

        h = FakeHandler(headers={"Authorization": "Bearer tok"})
        result = handler(handler=h)
        assert result.status_code == 401
        body = _body_json(result)
        assert "ARAGORA_API_TOKEN" in body["error"]

    @patch("aragora.server.auth.auth_config")
    def test_missing_bearer_token(self, mock_cfg):
        mock_cfg.api_token = "secret-123"
        mock_cfg.validate_token.return_value = False

        @require_auth
        def handler(handler=None):
            return {"ok": True}

        h = FakeHandler(headers={})
        result = handler(handler=h)
        assert result.status_code == 401

    @patch("aragora.server.auth.auth_config")
    def test_invalid_token(self, mock_cfg):
        mock_cfg.api_token = "secret-123"
        mock_cfg.validate_token.return_value = False

        @require_auth
        def handler(handler=None):
            return {"ok": True}

        h = FakeHandler(headers={"Authorization": "Bearer bad-token"})
        result = handler(handler=h)
        assert result.status_code == 401

    @patch("aragora.server.auth.auth_config")
    def test_valid_token(self, mock_cfg):
        mock_cfg.api_token = "secret-123"
        mock_cfg.validate_token.return_value = True

        @require_auth
        def handler(handler=None):
            return {"ok": True}

        h = FakeHandler(headers={"Authorization": "Bearer valid-token"})
        result = handler(handler=h)
        assert result == {"ok": True}

    @patch("aragora.server.auth.auth_config")
    def test_handler_via_positional_arg(self, mock_cfg):
        mock_cfg.api_token = "secret-123"
        mock_cfg.validate_token.return_value = True

        @require_auth
        def handler(req):
            return {"ok": True}

        h = FakeHandler(headers={"Authorization": "Bearer good"})
        result = handler(h)
        assert result == {"ok": True}

    @patch("aragora.server.auth.auth_config")
    def test_non_bearer_prefix_ignored(self, mock_cfg):
        mock_cfg.api_token = "secret-123"
        mock_cfg.validate_token.return_value = False

        @require_auth
        def handler(handler=None):
            return {"ok": True}

        h = FakeHandler(headers={"Authorization": "Basic dXNlcjpwYXNz"})
        result = handler(handler=h)
        assert result.status_code == 401


# ============================================================================
# require_storage
# ============================================================================

class TestRequireStorage:
    def test_storage_available(self):
        class MyHandler:
            def get_storage(self):
                return MagicMock()

            @require_storage
            def do_work(self):
                return {"ok": True}

        h = MyHandler()
        assert h.do_work() == {"ok": True}

    def test_no_storage_returns_503(self):
        class MyHandler:
            def get_storage(self):
                return None

            @require_storage
            def do_work(self):
                return {"ok": True}

        h = MyHandler()
        result = h.do_work()
        assert isinstance(result, HandlerResult)
        assert result.status_code == 503

    def test_preserves_function_name(self):
        class MyHandler:
            def get_storage(self):
                return True

            @require_storage
            def my_method(self):
                pass

        assert MyHandler.my_method.__name__ == "my_method"


# ============================================================================
# require_feature
# ============================================================================

class TestRequireFeature:
    def test_feature_available(self):
        @require_feature(lambda: True, "Redis")
        def handler():
            return {"ok": True}

        assert handler() == {"ok": True}

    def test_feature_unavailable_default_503(self):
        @require_feature(lambda: False, "Redis")
        def handler():
            return {"ok": True}

        result = handler()
        assert isinstance(result, HandlerResult)
        assert result.status_code == 503
        body = _body_json(result)
        assert "Redis" in body["error"]

    def test_feature_unavailable_custom_status(self):
        @require_feature(lambda: False, "ML Model", status_code=501)
        def handler():
            return {"ok": True}

        result = handler()
        assert result.status_code == 501

    def test_preserves_function_name(self):
        @require_feature(lambda: True, "feature")
        def my_handler():
            pass

        assert my_handler.__name__ == "my_handler"


# ============================================================================
# safe_fetch
# ============================================================================

class TestSafeFetch:
    def test_success(self):
        data: dict[str, Any] = {}
        errors: dict[str, Any] = {}
        with safe_fetch(data, errors, "rankings", []):
            data["rankings"] = [1, 2, 3]

        assert data["rankings"] == [1, 2, 3]
        assert "rankings" not in errors

    def test_failure_uses_fallback(self):
        data: dict[str, Any] = {}
        errors: dict[str, Any] = {}
        with safe_fetch(data, errors, "rankings", {"agents": [], "count": 0}):
            raise RuntimeError("db down")

        assert data["rankings"] == {"agents": [], "count": 0}
        assert errors["rankings"] == "Fetch failed"

    def test_log_errors_true(self, caplog):
        data: dict[str, Any] = {}
        errors: dict[str, Any] = {}
        with caplog.at_level(logging.WARNING):
            with safe_fetch(data, errors, "rankings", [], log_errors=True):
                raise ValueError("bad query")

        assert any("safe_fetch" in r.message for r in caplog.records)

    def test_log_errors_false(self, caplog):
        data: dict[str, Any] = {}
        errors: dict[str, Any] = {}
        with caplog.at_level(logging.WARNING):
            with safe_fetch(data, errors, "rankings", [], log_errors=False):
                raise ValueError("bad query")

        # Should not log
        assert not any("safe_fetch" in r.message for r in caplog.records)


# ============================================================================
# with_error_recovery
# ============================================================================

class TestWithErrorRecovery:
    def test_success(self):
        @with_error_recovery(fallback_value=[])
        def handler():
            return [1, 2, 3]

        assert handler() == [1, 2, 3]

    def test_error_returns_fallback(self):
        @with_error_recovery(fallback_value={"error": True})
        def handler():
            raise RuntimeError("broken")

        assert handler() == {"error": True}

    def test_default_fallback_is_none(self):
        @with_error_recovery()
        def handler():
            raise RuntimeError("broken")

        assert handler() is None

    def test_log_errors_true(self, caplog):
        @with_error_recovery(log_errors=True)
        def handler():
            raise ValueError("bad")

        with caplog.at_level(logging.WARNING):
            handler()
        assert any("with_error_recovery" in r.message for r in caplog.records)

    def test_log_errors_false(self, caplog):
        @with_error_recovery(log_errors=False)
        def handler():
            raise ValueError("bad")

        with caplog.at_level(logging.WARNING):
            handler()
        assert not any("with_error_recovery" in r.message for r in caplog.records)

    def test_preserves_function_name(self):
        @with_error_recovery()
        def my_func():
            pass

        assert my_func.__name__ == "my_func"


# ============================================================================
# deprecated_endpoint
# ============================================================================

class TestDeprecatedEndpoint:
    def test_adds_deprecation_header(self):
        @deprecated_endpoint()
        def handler():
            return {"status": 200, "body": "ok"}

        result = handler()
        assert result["headers"]["Deprecation"] == "true"

    def test_replacement_link_header(self):
        @deprecated_endpoint(replacement="/api/v2/debates")
        def handler():
            return {"status": 200, "body": "ok"}

        result = handler()
        assert "/api/v2/debates" in result["headers"]["Link"]
        assert 'rel="successor-version"' in result["headers"]["Link"]

    def test_sunset_header(self):
        @deprecated_endpoint(sunset_date="2025-06-01")
        def handler():
            return {"status": 200, "body": "ok"}

        result = handler()
        assert "2025" in result["headers"]["Sunset"]
        assert "Jun" in result["headers"]["Sunset"]

    def test_invalid_sunset_date_ignored(self, caplog):
        @deprecated_endpoint(sunset_date="not-a-date")
        def handler():
            return {"status": 200, "body": "ok"}

        with caplog.at_level(logging.WARNING):
            result = handler()
        assert "Sunset" not in result["headers"]

    def test_custom_message_logged(self, caplog):
        @deprecated_endpoint(message="Old endpoint used")
        def handler():
            return {"status": 200, "body": "ok"}

        with caplog.at_level(logging.WARNING):
            handler()
        assert any("Old endpoint used" in r.message for r in caplog.records)

    def test_none_result_no_crash(self):
        @deprecated_endpoint()
        def handler():
            return None

        result = handler()
        assert result is None

    def test_non_dict_result_no_crash(self):
        @deprecated_endpoint()
        def handler():
            return "plain string"

        result = handler()
        assert result == "plain string"

    def test_existing_headers_preserved(self):
        @deprecated_endpoint(replacement="/api/v2/new")
        def handler():
            return {"status": 200, "headers": {"X-Custom": "val"}}

        result = handler()
        assert result["headers"]["X-Custom"] == "val"
        assert result["headers"]["Deprecation"] == "true"

    def test_preserves_function_name(self):
        @deprecated_endpoint()
        def my_handler():
            pass

        assert my_handler.__name__ == "my_handler"

    def test_all_three_headers(self):
        @deprecated_endpoint(
            replacement="/api/v2/debates",
            sunset_date="2025-12-31",
        )
        def handler():
            return {"status": 200, "body": "ok"}

        result = handler()
        headers = result["headers"]
        assert headers["Deprecation"] == "true"
        assert "Sunset" in headers
        assert "Link" in headers


# ============================================================================
# Edge case / integration tests
# ============================================================================

class TestEdgeCases:
    def test_handle_errors_with_key_error(self):
        @handle_errors("key lookup")
        def handler():
            raise KeyError("missing_field")

        result = handler()
        assert result.status_code == 404

    def test_handle_errors_connection_error(self):
        @handle_errors("connection")
        def handler():
            raise ConnectionError("refused")

        result = handler()
        assert result.status_code == 502

    def test_validate_params_preserves_function_name(self):
        @validate_params({"x": (int, 0, None, None)})
        def my_func(query_params=None, **kw):
            pass

        assert my_func.__name__ == "my_func"

    def test_stacking_decorators(self):
        """Verify that handle_errors + validate_params stack correctly."""
        @handle_errors("stacked")
        @validate_params({"page": (int, 1, 1, 10)})
        def handler(query_params=None, **kw):
            return {"page": kw["page"]}

        result = handler(query_params={"page": "5"})
        assert result == {"page": 5}

    def test_stacking_handle_errors_catches_inner(self):
        @handle_errors("stacked error")
        @validate_params({"page": (int, 1, 1, 10)})
        def handler(query_params=None, **kw):
            raise ValueError("inner failure")

        result = handler(query_params={"page": "5"})
        assert isinstance(result, HandlerResult)
        assert result.status_code == 400

    def test_require_feature_with_dynamic_check(self):
        """Feature check that changes at runtime."""
        flag = {"enabled": False}

        @require_feature(lambda: flag["enabled"], "Dynamic Feature")
        def handler():
            return {"ok": True}

        result = handler()
        assert isinstance(result, HandlerResult)
        assert result.status_code == 503

        flag["enabled"] = True
        result = handler()
        assert result == {"ok": True}

    def test_permission_matrix_roles_are_valid(self):
        """All roles in PERMISSION_MATRIX should be one of the known roles."""
        valid_roles = {"member", "admin", "owner"}
        for perm, roles in PERMISSION_MATRIX.items():
            for role in roles:
                assert role in valid_roles, f"Invalid role '{role}' in permission '{perm}'"
