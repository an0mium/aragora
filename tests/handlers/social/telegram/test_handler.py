"""Tests for main TelegramHandler class (aragora/server/handlers/social/telegram/handler.py).

Covers all public interface of TelegramHandler and get_telegram_handler factory:
- __init__: context initialization
- ROUTES: all three routes are listed
- can_handle: routing for known/unknown paths, all three routes
- handle: dispatch to status, set-webhook, webhook endpoints
  - Method enforcement (POST-only for webhook and set-webhook)
  - RBAC permission checking for set-webhook
  - Webhook secret verification
  - 404 for unknown paths
- handle_post: POST delegation and @handle_errors decorator
- get_telegram_handler: singleton factory, with/without server_context
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.social.telegram.handler import (
    TelegramHandler,
    get_telegram_handler,
    _telegram_handler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Mock HTTP handler
# ---------------------------------------------------------------------------


class MockHTTPHandler:
    """Minimal mock HTTP handler for TelegramHandler tests."""

    def __init__(
        self,
        method: str = "GET",
        body: dict | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.command = method
        self.client_address = ("127.0.0.1", 12345)
        self._body = body or {}
        raw = json.dumps(self._body).encode()
        self.rfile = MagicMock()
        self.rfile.read.return_value = raw
        self.headers = {
            "Content-Length": str(len(raw)),
            **(headers or {}),
        }

    def get_header(self, name: str, default: str = "") -> str:
        return self.headers.get(name, default)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the module-level singleton before each test."""
    import aragora.server.handlers.social.telegram.handler as handler_mod

    handler_mod._telegram_handler = None
    yield
    handler_mod._telegram_handler = None


@pytest.fixture
def handler():
    """Create a fresh TelegramHandler."""
    return TelegramHandler(ctx={"some_key": "some_val"})


@pytest.fixture
def handler_no_ctx():
    """Create a TelegramHandler without context."""
    return TelegramHandler()


@pytest.fixture
def mock_http_get():
    """MockHTTPHandler for GET requests."""
    return MockHTTPHandler(method="GET")


@pytest.fixture
def mock_http_post():
    """MockHTTPHandler for POST requests."""
    return MockHTTPHandler(method="POST")


@pytest.fixture
def _patch_check_permission_allow():
    """Patch _check_permission to always allow (return None)."""
    with patch.object(
        TelegramHandler,
        "_check_permission",
        return_value=None,
    ):
        yield


@pytest.fixture
def _patch_check_permission_deny():
    """Patch _check_permission to return a 403 error result."""
    denied = MagicMock(status_code=403, body=b'{"error":"Permission denied"}')
    with patch.object(
        TelegramHandler,
        "_check_permission",
        return_value=denied,
    ):
        yield denied


@pytest.fixture
def _patch_verify_secret_ok():
    """Patch _verify_secret to always return True."""
    with patch.object(
        TelegramHandler,
        "_verify_secret",
        return_value=True,
    ):
        yield


@pytest.fixture
def _patch_verify_secret_fail():
    """Patch _verify_secret to always return False."""
    with patch.object(
        TelegramHandler,
        "_verify_secret",
        return_value=False,
    ):
        yield


@pytest.fixture
def _patch_get_status():
    """Patch _get_status to return a known value."""
    status_result = MagicMock(
        status_code=200,
        body=b'{"enabled":true}',
    )
    with patch.object(
        TelegramHandler,
        "_get_status",
        return_value=status_result,
    ) as mock:
        yield mock


@pytest.fixture
def _patch_set_webhook():
    """Patch _set_webhook to return a known value."""
    webhook_result = MagicMock(
        status_code=200,
        body=b'{"status":"webhook configuration queued"}',
    )
    with patch.object(
        TelegramHandler,
        "_set_webhook",
        return_value=webhook_result,
    ) as mock:
        yield mock


@pytest.fixture
def _patch_handle_webhook():
    """Patch _handle_webhook to return a known value."""
    webhook_result = MagicMock(
        status_code=200,
        body=b'{"ok":true}',
    )
    with patch.object(
        TelegramHandler,
        "_handle_webhook",
        return_value=webhook_result,
    ) as mock:
        yield mock


# =====================================================================
# Tests: __init__
# =====================================================================


class TestInit:
    """TelegramHandler.__init__ tests."""

    def test_init_with_context(self, handler):
        """Context is stored when provided."""
        assert handler.ctx == {"some_key": "some_val"}

    def test_init_without_context(self, handler_no_ctx):
        """Defaults to empty dict when ctx is None."""
        assert handler_no_ctx.ctx == {}

    def test_init_none_context(self):
        """Explicitly passing None yields empty dict."""
        h = TelegramHandler(ctx=None)
        assert h.ctx == {}

    def test_init_empty_context(self):
        """Explicitly passing empty dict."""
        h = TelegramHandler(ctx={})
        assert h.ctx == {}


# =====================================================================
# Tests: ROUTES
# =====================================================================


class TestRoutes:
    """TelegramHandler.ROUTES constant tests."""

    def test_routes_contains_webhook(self, handler):
        assert "/api/v1/integrations/telegram/webhook" in handler.ROUTES

    def test_routes_contains_status(self, handler):
        assert "/api/v1/integrations/telegram/status" in handler.ROUTES

    def test_routes_contains_set_webhook(self, handler):
        assert "/api/v1/integrations/telegram/set-webhook" in handler.ROUTES

    def test_routes_count(self, handler):
        assert len(handler.ROUTES) == 3


# =====================================================================
# Tests: can_handle
# =====================================================================


class TestCanHandle:
    """TelegramHandler.can_handle tests."""

    def test_can_handle_webhook(self, handler):
        assert handler.can_handle("/api/v1/integrations/telegram/webhook") is True

    def test_can_handle_status(self, handler):
        assert handler.can_handle("/api/v1/integrations/telegram/status") is True

    def test_can_handle_set_webhook(self, handler):
        assert handler.can_handle("/api/v1/integrations/telegram/set-webhook") is True

    def test_cannot_handle_unknown_path(self, handler):
        assert handler.can_handle("/api/v1/integrations/telegram/unknown") is False

    def test_cannot_handle_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_cannot_handle_partial_path(self, handler):
        assert handler.can_handle("/api/v1/integrations/telegram") is False

    def test_can_handle_ignores_method(self, handler):
        """can_handle only checks path, not HTTP method."""
        assert handler.can_handle("/api/v1/integrations/telegram/webhook", "DELETE") is True


# =====================================================================
# Tests: handle - status route
# =====================================================================


class TestHandleStatus:
    """Tests for handle() dispatching to status endpoint."""

    def test_status_route_calls_get_status(self, handler, mock_http_get, _patch_get_status):
        result = handler.handle("/api/v1/integrations/telegram/status", {}, mock_http_get)
        _patch_get_status.assert_called_once_with(mock_http_get)

    def test_status_route_returns_result(self, handler, mock_http_get, _patch_get_status):
        result = handler.handle("/api/v1/integrations/telegram/status", {}, mock_http_get)
        assert _status(result) == 200

    def test_status_route_accepts_post(self, handler, mock_http_post, _patch_get_status):
        """Status route does not enforce method; it returns status for any method."""
        result = handler.handle("/api/v1/integrations/telegram/status", {}, mock_http_post)
        assert _status(result) == 200


# =====================================================================
# Tests: handle - set-webhook route
# =====================================================================


class TestHandleSetWebhook:
    """Tests for handle() dispatching to set-webhook endpoint."""

    def test_set_webhook_post_allowed(
        self, handler, mock_http_post, _patch_check_permission_allow, _patch_set_webhook
    ):
        result = handler.handle("/api/v1/integrations/telegram/set-webhook", {}, mock_http_post)
        assert _status(result) == 200
        _patch_set_webhook.assert_called_once_with(mock_http_post)

    def test_set_webhook_get_returns_405(self, handler, mock_http_get):
        result = handler.handle("/api/v1/integrations/telegram/set-webhook", {}, mock_http_get)
        assert _status(result) == 405
        assert "Method not allowed" in _body(result).get("error", "")

    def test_set_webhook_permission_denied(
        self, handler, mock_http_post, _patch_check_permission_deny
    ):
        result = handler.handle("/api/v1/integrations/telegram/set-webhook", {}, mock_http_post)
        assert _status(result) == 403

    def test_set_webhook_permission_check_called_with_admin(
        self, handler, mock_http_post, _patch_set_webhook
    ):
        """Verify _check_permission is called with PERM_TELEGRAM_ADMIN."""
        with patch.object(TelegramHandler, "_check_permission", return_value=None) as perm_mock:
            handler.handle("/api/v1/integrations/telegram/set-webhook", {}, mock_http_post)
            perm_mock.assert_called_once_with(mock_http_post, "telegram:admin")


# =====================================================================
# Tests: handle - webhook route
# =====================================================================


class TestHandleWebhook:
    """Tests for handle() dispatching to webhook endpoint."""

    def test_webhook_post_with_valid_secret(
        self, handler, mock_http_post, _patch_verify_secret_ok, _patch_handle_webhook
    ):
        result = handler.handle("/api/v1/integrations/telegram/webhook", {}, mock_http_post)
        assert _status(result) == 200
        _patch_handle_webhook.assert_called_once_with(mock_http_post)

    def test_webhook_get_returns_405(self, handler, mock_http_get):
        result = handler.handle("/api/v1/integrations/telegram/webhook", {}, mock_http_get)
        assert _status(result) == 405
        assert "Method not allowed" in _body(result).get("error", "")

    def test_webhook_secret_verification_fails_returns_401(
        self, handler, mock_http_post, _patch_verify_secret_fail
    ):
        result = handler.handle("/api/v1/integrations/telegram/webhook", {}, mock_http_post)
        assert _status(result) == 401
        assert "Unauthorized" in _body(result).get("error", "")

    def test_webhook_calls_verify_secret(self, handler, mock_http_post, _patch_handle_webhook):
        """_verify_secret is called before _handle_webhook."""
        with patch.object(TelegramHandler, "_verify_secret", return_value=True) as verify_mock:
            handler.handle("/api/v1/integrations/telegram/webhook", {}, mock_http_post)
            verify_mock.assert_called_once_with(mock_http_post)


# =====================================================================
# Tests: handle - unknown path
# =====================================================================


class TestHandleUnknown:
    """Tests for handle() with unrecognized paths."""

    def test_unknown_path_returns_404(self, handler, mock_http_get):
        result = handler.handle("/api/v1/integrations/telegram/foo", {}, mock_http_get)
        assert _status(result) == 404
        assert "Not found" in _body(result).get("error", "")

    def test_empty_path_returns_404(self, handler, mock_http_get):
        result = handler.handle("", {}, mock_http_get)
        assert _status(result) == 404

    def test_root_path_returns_404(self, handler, mock_http_get):
        result = handler.handle("/", {}, mock_http_get)
        assert _status(result) == 404


# =====================================================================
# Tests: handle_post
# =====================================================================


class TestHandlePost:
    """Tests for handle_post method."""

    def test_handle_post_delegates_to_handle(self, handler, mock_http_post, _patch_get_status):
        """handle_post should delegate to handle()."""
        result = handler.handle_post("/api/v1/integrations/telegram/status", {}, mock_http_post)
        assert _status(result) == 200

    def test_handle_post_passes_empty_query_params(self, handler, mock_http_post):
        """handle_post calls handle with empty dict for query_params."""
        with patch.object(handler, "handle", return_value=None) as handle_mock:
            handler.handle_post("/some/path", {"key": "val"}, mock_http_post)
            handle_mock.assert_called_once_with("/some/path", {}, mock_http_post)


# =====================================================================
# Tests: get_telegram_handler factory
# =====================================================================


class TestGetTelegramHandler:
    """Tests for get_telegram_handler singleton factory."""

    def test_returns_telegram_handler_instance(self):
        h = get_telegram_handler()
        assert isinstance(h, TelegramHandler)

    def test_returns_same_instance_on_repeated_calls(self):
        h1 = get_telegram_handler()
        h2 = get_telegram_handler()
        assert h1 is h2

    def test_server_context_is_passed(self):
        ctx = {"db": "fake_db"}
        h = get_telegram_handler(server_context=ctx)
        assert h.ctx == ctx

    def test_none_server_context_defaults_to_empty_dict(self):
        h = get_telegram_handler(server_context=None)
        assert h.ctx == {}

    def test_second_call_ignores_new_context(self):
        """Once singleton is created, subsequent calls don't replace it."""
        h1 = get_telegram_handler(server_context={"first": True})
        h2 = get_telegram_handler(server_context={"second": True})
        assert h1 is h2
        assert h1.ctx == {"first": True}


# =====================================================================
# Tests: handle - combined scenarios / edge cases
# =====================================================================


class TestHandleEdgeCases:
    """Edge cases and combined scenarios for handle()."""

    def test_query_params_ignored(self, handler, mock_http_get, _patch_get_status):
        """query_params argument is passed but not used in routing."""
        result = handler.handle(
            "/api/v1/integrations/telegram/status",
            {"key": "value", "filter": "active"},
            mock_http_get,
        )
        assert _status(result) == 200

    def test_webhook_post_method_check_before_secret_check(self, handler, mock_http_get):
        """GET on webhook returns 405 before even checking the secret."""
        with patch.object(TelegramHandler, "_verify_secret") as verify_mock:
            result = handler.handle("/api/v1/integrations/telegram/webhook", {}, mock_http_get)
            assert _status(result) == 405
            verify_mock.assert_not_called()

    def test_set_webhook_method_check_before_permission_check(self, handler, mock_http_get):
        """GET on set-webhook returns 405 before checking permissions."""
        with patch.object(TelegramHandler, "_check_permission") as perm_mock:
            result = handler.handle("/api/v1/integrations/telegram/set-webhook", {}, mock_http_get)
            assert _status(result) == 405
            perm_mock.assert_not_called()

    def test_set_webhook_permission_check_before_set_webhook_call(self, handler, mock_http_post):
        """When permission denied, _set_webhook is never called."""
        denied = MagicMock(status_code=403, body=b'{"error":"denied"}')
        with (
            patch.object(TelegramHandler, "_check_permission", return_value=denied),
            patch.object(TelegramHandler, "_set_webhook") as set_mock,
        ):
            result = handler.handle("/api/v1/integrations/telegram/set-webhook", {}, mock_http_post)
            assert _status(result) == 403
            set_mock.assert_not_called()

    def test_webhook_secret_fail_does_not_call_handle_webhook(
        self, handler, mock_http_post, _patch_verify_secret_fail
    ):
        """When secret verification fails, _handle_webhook is not called."""
        with patch.object(TelegramHandler, "_handle_webhook") as hook_mock:
            result = handler.handle("/api/v1/integrations/telegram/webhook", {}, mock_http_post)
            assert _status(result) == 401
            hook_mock.assert_not_called()

    def test_handle_returns_none_only_if_no_match(self, handler, mock_http_get):
        """Actually handle returns error_response(404) for no match, never None."""
        result = handler.handle("/no/match", {}, mock_http_get)
        assert result is not None
        assert _status(result) == 404


# =====================================================================
# Tests: MRO / Mixin composition
# =====================================================================


class TestMixinComposition:
    """Verify TelegramHandler composes all expected mixins."""

    def test_has_rbac_methods(self, handler):
        assert hasattr(handler, "_check_permission")
        assert hasattr(handler, "_get_auth_context")

    def test_has_message_methods(self, handler):
        assert hasattr(handler, "_send_message_async")
        assert hasattr(handler, "_answer_callback_async")

    def test_has_webhook_methods(self, handler):
        assert hasattr(handler, "_verify_secret")
        assert hasattr(handler, "_get_status")
        assert hasattr(handler, "_set_webhook")
        assert hasattr(handler, "_handle_webhook")

    def test_has_command_methods(self, handler):
        assert hasattr(handler, "_handle_command")
        assert hasattr(handler, "_command_start")
        assert hasattr(handler, "_command_help")

    def test_has_callback_methods(self, handler):
        assert hasattr(handler, "_handle_message")
        assert hasattr(handler, "_handle_callback_query")
        assert hasattr(handler, "_handle_inline_query")

    def test_is_base_handler(self, handler):
        from aragora.server.handlers.base import BaseHandler

        assert isinstance(handler, BaseHandler)
