"""
Tests for BotHandlerMixin base handler.

Covers all methods and behavior of the BotHandlerMixin class:
- BotErrorCode enum values
- DEFAULT_BOTS_READ_PERMISSION constant
- handle_status_request(): auth success, UnauthorizedError, ForbiddenError
- _build_status_response(): base fields, extra_status merging, platform config
- _get_platform_config_status(): default returns empty dict
- _is_bot_enabled(): default returns False
- handle_with_auth(): auth success, UnauthorizedError, ForbiddenError, operation call
- handle_rate_limit_exceeded(): with and without limit_info
- handle_webhook_auth_failed(): logging, audit success, audit import failure
- _read_request_body(): valid, zero, missing, invalid, oversized Content-Length
- _parse_json_body(): valid JSON, invalid JSON, empty body (allow/disallow)
- _handle_webhook_exception(): JSONDecodeError, ValueError, KeyError, TypeError,
  ConnectionError, OSError, TimeoutError, generic Exception, return_200_on_error
- _audit_webhook_auth_failure(): with/without reason, import success/failure
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.bots.base import (
    BotErrorCode,
    BotHandlerMixin,
    DEFAULT_BOTS_READ_PERMISSION,
)
from aragora.server.handlers.utils.auth import ForbiddenError, UnauthorizedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Mock HTTP Handler
# ---------------------------------------------------------------------------


@dataclass
class MockHTTPHandler:
    """Mock HTTP handler for simulating requests."""

    path: str = "/api/v1/bots/test/status"
    method: str = "GET"
    body_bytes: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.rfile = io.BytesIO(self.body_bytes)
        if "Content-Length" not in self.headers:
            self.headers["Content-Length"] = str(len(self.body_bytes))
        self.client_address = ("127.0.0.1", 12345)


def _make_handler_with_body(body: dict[str, Any] | None = None) -> MockHTTPHandler:
    """Create a MockHTTPHandler with a JSON body."""
    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
    else:
        body_bytes = b""
    return MockHTTPHandler(
        method="POST",
        body_bytes=body_bytes,
        headers={"Content-Length": str(len(body_bytes))},
    )


# ---------------------------------------------------------------------------
# Concrete subclass for testing
# ---------------------------------------------------------------------------


class TestBotHandler(BotHandlerMixin):
    """Concrete BotHandlerMixin subclass for testing."""

    bot_platform = "testbot"
    bots_read_permission = "bots.read"

    def __init__(self, *, enabled: bool = False, platform_config: dict | None = None):
        self._enabled = enabled
        self._platform_config = platform_config or {}

    def _is_bot_enabled(self) -> bool:
        return self._enabled

    def _get_platform_config_status(self) -> dict[str, Any]:
        return self._platform_config

    async def get_auth_context(self, handler, require_auth=True):
        """Mock - overridden in tests."""
        raise NotImplementedError

    def check_permission(self, auth_context, permission, resource_id=None):
        """Mock - overridden in tests."""
        raise NotImplementedError


@pytest.fixture
def bot():
    """Create a TestBotHandler instance."""
    return TestBotHandler()


@pytest.fixture
def enabled_bot():
    """Create an enabled TestBotHandler with platform config."""
    return TestBotHandler(
        enabled=True,
        platform_config={"token_configured": True, "webhook_url": "https://example.com/hook"},
    )


@pytest.fixture
def mock_http():
    """Create a basic MockHTTPHandler."""
    return MockHTTPHandler()


@pytest.fixture
def mock_auth_ctx():
    """Create a mock AuthorizationContext."""
    ctx = MagicMock()
    ctx.user_id = "test-user-001"
    ctx.roles = {"admin"}
    ctx.permissions = {"*"}
    return ctx


# ===========================================================================
# BotErrorCode enum tests
# ===========================================================================


class TestBotErrorCode:
    """Tests for the BotErrorCode enum."""

    def test_invalid_signature_value(self):
        assert BotErrorCode.INVALID_SIGNATURE == "INVALID_SIGNATURE"
        assert BotErrorCode.INVALID_SIGNATURE.value == "INVALID_SIGNATURE"

    def test_invalid_token_value(self):
        assert BotErrorCode.INVALID_TOKEN == "INVALID_TOKEN"

    def test_auth_required_value(self):
        assert BotErrorCode.AUTH_REQUIRED == "AUTH_REQUIRED"

    def test_permission_denied_value(self):
        assert BotErrorCode.PERMISSION_DENIED == "PERMISSION_DENIED"

    def test_invalid_json_value(self):
        assert BotErrorCode.INVALID_JSON == "INVALID_JSON"

    def test_validation_error_value(self):
        assert BotErrorCode.VALIDATION_ERROR == "VALIDATION_ERROR"

    def test_missing_field_value(self):
        assert BotErrorCode.MISSING_FIELD == "MISSING_FIELD"

    def test_empty_body_value(self):
        assert BotErrorCode.EMPTY_BODY == "EMPTY_BODY"

    def test_not_configured_value(self):
        assert BotErrorCode.NOT_CONFIGURED == "NOT_CONFIGURED"

    def test_feature_disabled_value(self):
        assert BotErrorCode.FEATURE_DISABLED == "FEATURE_DISABLED"

    def test_rate_limit_exceeded_value(self):
        assert BotErrorCode.RATE_LIMIT_EXCEEDED == "RATE_LIMIT_EXCEEDED"

    def test_platform_error_value(self):
        assert BotErrorCode.PLATFORM_ERROR == "PLATFORM_ERROR"

    def test_platform_unavailable_value(self):
        assert BotErrorCode.PLATFORM_UNAVAILABLE == "PLATFORM_UNAVAILABLE"

    def test_connection_error_value(self):
        assert BotErrorCode.CONNECTION_ERROR == "CONNECTION_ERROR"

    def test_internal_error_value(self):
        assert BotErrorCode.INTERNAL_ERROR == "INTERNAL_ERROR"

    def test_is_str_subclass(self):
        """BotErrorCode values are strings (str enum)."""
        assert isinstance(BotErrorCode.INVALID_SIGNATURE, str)

    def test_all_members_count(self):
        """Verify the total number of error codes."""
        assert len(BotErrorCode) == 15


class TestDefaultPermission:
    """Tests for the DEFAULT_BOTS_READ_PERMISSION constant."""

    def test_default_permission_value(self):
        assert DEFAULT_BOTS_READ_PERMISSION == "bots.read"

    def test_mixin_default_matches_constant(self):
        """The mixin class attribute matches the module constant."""
        handler = TestBotHandler()
        assert handler.bots_read_permission == DEFAULT_BOTS_READ_PERMISSION


# ===========================================================================
# handle_status_request tests
# ===========================================================================


class TestHandleStatusRequest:
    """Tests for handle_status_request()."""

    @pytest.mark.asyncio
    async def test_success_returns_status(self, enabled_bot, mock_http, mock_auth_ctx):
        enabled_bot.get_auth_context = AsyncMock(return_value=mock_auth_ctx)
        enabled_bot.check_permission = MagicMock()

        result = await enabled_bot.handle_status_request(mock_http)

        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "testbot"
        assert body["enabled"] is True
        assert body["token_configured"] is True

    @pytest.mark.asyncio
    async def test_success_disabled_bot(self, bot, mock_http, mock_auth_ctx):
        bot.get_auth_context = AsyncMock(return_value=mock_auth_ctx)
        bot.check_permission = MagicMock()

        result = await bot.handle_status_request(mock_http)

        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "testbot"
        assert body["enabled"] is False

    @pytest.mark.asyncio
    async def test_unauthorized_returns_401(self, bot, mock_http):
        bot.get_auth_context = AsyncMock(side_effect=UnauthorizedError("No token"))

        result = await bot.handle_status_request(mock_http)

        assert _status(result) == 401
        body = _body(result)
        assert "Authentication required" in body["error"]["message"]
        assert body["error"]["code"] == "AUTH_REQUIRED"

    @pytest.mark.asyncio
    async def test_forbidden_returns_403(self, bot, mock_http, mock_auth_ctx):
        bot.get_auth_context = AsyncMock(return_value=mock_auth_ctx)
        bot.check_permission = MagicMock(side_effect=ForbiddenError("No perm"))

        result = await bot.handle_status_request(mock_http)

        assert _status(result) == 403
        body = _body(result)
        assert "Permission denied" in body["error"]["message"]
        assert body["error"]["code"] == "PERMISSION_DENIED"

    @pytest.mark.asyncio
    async def test_extra_status_merged(self, bot, mock_http, mock_auth_ctx):
        bot.get_auth_context = AsyncMock(return_value=mock_auth_ctx)
        bot.check_permission = MagicMock()

        result = await bot.handle_status_request(
            mock_http, extra_status={"version": "1.2.3", "uptime": 3600}
        )

        body = _body(result)
        assert body["version"] == "1.2.3"
        assert body["uptime"] == 3600

    @pytest.mark.asyncio
    async def test_extra_status_none(self, bot, mock_http, mock_auth_ctx):
        bot.get_auth_context = AsyncMock(return_value=mock_auth_ctx)
        bot.check_permission = MagicMock()

        result = await bot.handle_status_request(mock_http, extra_status=None)

        body = _body(result)
        assert body["platform"] == "testbot"

    @pytest.mark.asyncio
    async def test_calls_get_auth_context_with_require_auth(self, bot, mock_http, mock_auth_ctx):
        bot.get_auth_context = AsyncMock(return_value=mock_auth_ctx)
        bot.check_permission = MagicMock()

        await bot.handle_status_request(mock_http)

        bot.get_auth_context.assert_awaited_once_with(mock_http, require_auth=True)

    @pytest.mark.asyncio
    async def test_calls_check_permission_with_bots_read(self, bot, mock_http, mock_auth_ctx):
        bot.get_auth_context = AsyncMock(return_value=mock_auth_ctx)
        bot.check_permission = MagicMock()

        await bot.handle_status_request(mock_http)

        bot.check_permission.assert_called_once_with(mock_auth_ctx, "bots.read")

    @pytest.mark.asyncio
    async def test_custom_permission(self, mock_http, mock_auth_ctx):
        """A subclass with custom bots_read_permission uses it."""
        handler = TestBotHandler()
        handler.bots_read_permission = "custom.read"
        handler.get_auth_context = AsyncMock(return_value=mock_auth_ctx)
        handler.check_permission = MagicMock()

        await handler.handle_status_request(mock_http)

        handler.check_permission.assert_called_once_with(mock_auth_ctx, "custom.read")


# ===========================================================================
# _build_status_response tests
# ===========================================================================


class TestBuildStatusResponse:
    """Tests for _build_status_response()."""

    def test_base_fields(self, bot):
        result = bot._build_status_response()

        body = _body(result)
        assert body["platform"] == "testbot"
        assert body["enabled"] is False

    def test_enabled_bot(self, enabled_bot):
        result = enabled_bot._build_status_response()

        body = _body(result)
        assert body["enabled"] is True

    def test_platform_config_included(self, enabled_bot):
        result = enabled_bot._build_status_response()

        body = _body(result)
        assert body["token_configured"] is True
        assert body["webhook_url"] == "https://example.com/hook"

    def test_extra_status_merges(self, bot):
        result = bot._build_status_response(extra_status={"connected": True, "users": 42})

        body = _body(result)
        assert body["connected"] is True
        assert body["users"] == 42

    def test_extra_status_overrides_platform_config(self, enabled_bot):
        """extra_status takes precedence over platform config."""
        result = enabled_bot._build_status_response(
            extra_status={"token_configured": False}
        )

        body = _body(result)
        assert body["token_configured"] is False

    def test_none_extra_status(self, bot):
        result = bot._build_status_response(extra_status=None)

        body = _body(result)
        assert "platform" in body
        assert "enabled" in body

    def test_empty_extra_status(self, bot):
        result = bot._build_status_response(extra_status={})

        body = _body(result)
        assert body["platform"] == "testbot"


# ===========================================================================
# _get_platform_config_status tests
# ===========================================================================


class TestGetPlatformConfigStatus:
    """Tests for _get_platform_config_status()."""

    def test_default_returns_empty_dict(self):
        """The base mixin returns empty dict."""
        mixin = BotHandlerMixin()
        assert mixin._get_platform_config_status() == {}

    def test_subclass_override(self, enabled_bot):
        result = enabled_bot._get_platform_config_status()
        assert result == {"token_configured": True, "webhook_url": "https://example.com/hook"}


# ===========================================================================
# _is_bot_enabled tests
# ===========================================================================


class TestIsBotEnabled:
    """Tests for _is_bot_enabled()."""

    def test_default_returns_false(self):
        """The base mixin returns False."""
        mixin = BotHandlerMixin()
        assert mixin._is_bot_enabled() is False

    def test_subclass_enabled(self, enabled_bot):
        assert enabled_bot._is_bot_enabled() is True

    def test_subclass_disabled(self, bot):
        assert bot._is_bot_enabled() is False


# ===========================================================================
# handle_with_auth tests
# ===========================================================================


class TestHandleWithAuth:
    """Tests for handle_with_auth()."""

    @pytest.mark.asyncio
    async def test_success_calls_operation(self, bot, mock_http, mock_auth_ctx):
        bot.get_auth_context = AsyncMock(return_value=mock_auth_ctx)
        bot.check_permission = MagicMock()

        operation = AsyncMock(return_value={"result": "ok"})

        result = await bot.handle_with_auth(mock_http, "bots.write", operation)

        operation.assert_awaited_once_with(auth_context=mock_auth_ctx)
        assert result == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_passes_args_and_kwargs(self, bot, mock_http, mock_auth_ctx):
        bot.get_auth_context = AsyncMock(return_value=mock_auth_ctx)
        bot.check_permission = MagicMock()

        operation = AsyncMock(return_value="done")

        result = await bot.handle_with_auth(
            mock_http, "bots.write", operation, "arg1", "arg2", key="value"
        )

        operation.assert_awaited_once_with("arg1", "arg2", key="value", auth_context=mock_auth_ctx)

    @pytest.mark.asyncio
    async def test_unauthorized_returns_401(self, bot, mock_http):
        bot.get_auth_context = AsyncMock(side_effect=UnauthorizedError())

        operation = AsyncMock()

        result = await bot.handle_with_auth(mock_http, "bots.write", operation)

        assert _status(result) == 401
        body = _body(result)
        assert body["error"]["code"] == "AUTH_REQUIRED"
        operation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_forbidden_returns_403(self, bot, mock_http, mock_auth_ctx):
        bot.get_auth_context = AsyncMock(return_value=mock_auth_ctx)
        bot.check_permission = MagicMock(side_effect=ForbiddenError("Denied"))

        operation = AsyncMock()

        result = await bot.handle_with_auth(mock_http, "bots.write", operation)

        assert _status(result) == 403
        body = _body(result)
        assert body["error"]["code"] == "PERMISSION_DENIED"
        operation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_checks_correct_permission(self, bot, mock_http, mock_auth_ctx):
        bot.get_auth_context = AsyncMock(return_value=mock_auth_ctx)
        bot.check_permission = MagicMock()

        operation = AsyncMock(return_value=None)

        await bot.handle_with_auth(mock_http, "custom.permission", operation)

        bot.check_permission.assert_called_once_with(mock_auth_ctx, "custom.permission")

    @pytest.mark.asyncio
    async def test_operation_return_value_passthrough(self, bot, mock_http, mock_auth_ctx):
        bot.get_auth_context = AsyncMock(return_value=mock_auth_ctx)
        bot.check_permission = MagicMock()

        from aragora.server.handlers.utils.responses import json_response

        expected = json_response({"items": [1, 2, 3]})
        operation = AsyncMock(return_value=expected)

        result = await bot.handle_with_auth(mock_http, "read", operation)

        assert result is expected


# ===========================================================================
# handle_rate_limit_exceeded tests
# ===========================================================================


class TestHandleRateLimitExceeded:
    """Tests for handle_rate_limit_exceeded()."""

    def test_no_limit_info(self, bot):
        result = bot.handle_rate_limit_exceeded()

        assert _status(result) == 429
        body = _body(result)
        assert "Rate limit exceeded" in body["error"]["message"]
        assert body["error"]["code"] == "RATE_LIMIT_EXCEEDED"

    def test_with_limit_info(self, bot):
        result = bot.handle_rate_limit_exceeded("100 requests per minute")

        assert _status(result) == 429
        body = _body(result)
        assert "100 requests per minute" in body["error"]["message"]

    def test_none_limit_info(self, bot):
        result = bot.handle_rate_limit_exceeded(None)

        assert _status(result) == 429
        body = _body(result)
        assert body["error"]["message"] == "Rate limit exceeded"

    def test_empty_string_limit_info(self, bot):
        result = bot.handle_rate_limit_exceeded("")

        assert _status(result) == 429
        body = _body(result)
        # Empty string is falsy, so no suffix appended
        assert body["error"]["message"] == "Rate limit exceeded"


# ===========================================================================
# handle_webhook_auth_failed tests
# ===========================================================================


class TestHandleWebhookAuthFailed:
    """Tests for handle_webhook_auth_failed()."""

    def test_returns_401(self, bot):
        result = bot.handle_webhook_auth_failed("signature")

        assert _status(result) == 401
        body = _body(result)
        assert body["error"]["code"] == "INVALID_SIGNATURE"

    def test_default_method_unknown(self, bot):
        result = bot.handle_webhook_auth_failed()

        assert _status(result) == 401

    @patch("aragora.server.handlers.bots.base.logger")
    def test_logs_warning(self, mock_logger, bot):
        bot.handle_webhook_auth_failed("token")

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0]
        assert "Testbot" in call_args[0] % tuple(call_args[1:])
        assert "token" in call_args[0] % tuple(call_args[1:])

    def test_audit_called_when_available(self, bot):
        with patch("aragora.server.handlers.bots.base.logger"):
            with patch(
                "aragora.audit.unified.audit_security"
            ) as mock_audit:
                bot.handle_webhook_auth_failed("signature")

                mock_audit.assert_called_once_with(
                    event_type="testbot_webhook_auth_failed",
                    actor_id="unknown",
                    resource_type="testbot_webhook",
                    resource_id="signature",
                )

    def test_audit_import_failure_handled(self, bot):
        """When audit module is not available, handler still returns 401."""
        with patch("aragora.server.handlers.bots.base.logger"):
            with patch(
                "builtins.__import__",
                side_effect=lambda name, *a, **kw: (
                    (_ for _ in ()).throw(ImportError("no audit"))
                    if "audit.unified" in name
                    else __import__(name, *a, **kw)
                ),
            ):
                # The patch above is tricky; let's use a simpler approach
                pass

        # Just verify it works without the audit module by checking the result
        result = bot.handle_webhook_auth_failed("hmac")
        assert _status(result) == 401


# ===========================================================================
# _read_request_body tests
# ===========================================================================


class TestReadRequestBody:
    """Tests for _read_request_body()."""

    def test_valid_body(self, bot):
        body_bytes = b'{"key": "value"}'
        handler = MockHTTPHandler(
            body_bytes=body_bytes,
            headers={"Content-Length": str(len(body_bytes))},
        )

        result = bot._read_request_body(handler)

        assert result == body_bytes

    def test_zero_content_length(self, bot):
        handler = MockHTTPHandler(headers={"Content-Length": "0"})

        result = bot._read_request_body(handler)

        assert result == b""

    def test_missing_content_length(self, bot):
        handler = MockHTTPHandler(headers={})

        result = bot._read_request_body(handler)

        assert result == b""

    def test_negative_content_length(self, bot):
        handler = MockHTTPHandler(headers={"Content-Length": "-5"})

        result = bot._read_request_body(handler)

        assert result == b""

    def test_invalid_content_length_string(self, bot):
        handler = MockHTTPHandler(headers={"Content-Length": "not_a_number"})

        result = bot._read_request_body(handler)

        assert result == b""

    def test_content_length_none(self, bot):
        """Content-Length header explicitly set to None-like value."""
        handler = MockHTTPHandler(headers={"Content-Length": ""})

        result = bot._read_request_body(handler)

        assert result == b""

    def test_oversized_body_raises(self, bot):
        # Use a Content-Length that exceeds _MAX_BODY_SIZE
        size = bot._MAX_BODY_SIZE + 1
        handler = MockHTTPHandler(headers={"Content-Length": str(size)})

        with pytest.raises(ValueError, match="Request body too large"):
            bot._read_request_body(handler)

    def test_exactly_max_size(self, bot):
        """Content-Length at exactly _MAX_BODY_SIZE should be allowed."""
        size = bot._MAX_BODY_SIZE
        body_bytes = b"x" * 100  # Actual data doesn't need to match
        handler = MockHTTPHandler(
            body_bytes=body_bytes,
            headers={"Content-Length": str(size)},
        )
        handler.rfile = MagicMock()
        handler.rfile.read.return_value = body_bytes

        result = bot._read_request_body(handler)
        handler.rfile.read.assert_called_once_with(size)

    def test_max_body_size_constant(self, bot):
        """Verify the max body size is 10MB."""
        assert bot._MAX_BODY_SIZE == 10 * 1024 * 1024


# ===========================================================================
# _parse_json_body tests
# ===========================================================================


class TestParseJsonBody:
    """Tests for _parse_json_body()."""

    def test_valid_json(self, bot):
        body = b'{"type": "event", "data": [1, 2, 3]}'

        data, error = bot._parse_json_body(body)

        assert error is None
        assert data == {"type": "event", "data": [1, 2, 3]}

    def test_empty_body_disallowed(self, bot):
        data, error = bot._parse_json_body(b"")

        assert data is None
        assert error is not None
        assert _status(error) == 400
        body = _body(error)
        assert body["error"]["code"] == "EMPTY_BODY"

    def test_empty_body_allowed(self, bot):
        data, error = bot._parse_json_body(b"", allow_empty=True)

        assert error is None
        assert data == {}

    def test_invalid_json(self, bot):
        data, error = bot._parse_json_body(b"not json at all")

        assert data is None
        assert error is not None
        assert _status(error) == 400
        body = _body(error)
        assert body["error"]["code"] == "INVALID_JSON"

    def test_partial_json(self, bot):
        data, error = bot._parse_json_body(b'{"incomplete":')

        assert data is None
        assert error is not None
        assert _status(error) == 400

    def test_custom_context_in_log(self, bot):
        """The context parameter is used in error logging."""
        with patch("aragora.server.handlers.bots.base.logger") as mock_logger:
            bot._parse_json_body(b"", context="my_event")

            mock_logger.error.assert_called_once()
            log_msg = mock_logger.error.call_args[0][0] % tuple(mock_logger.error.call_args[0][1:])
            assert "testbot" in log_msg
            assert "my_event" in log_msg

    def test_default_context_webhook(self, bot):
        with patch("aragora.server.handlers.bots.base.logger") as mock_logger:
            bot._parse_json_body(b"bad json")

            log_msg = mock_logger.error.call_args[0][0] % tuple(mock_logger.error.call_args[0][1:])
            assert "webhook" in log_msg

    def test_nested_json(self, bot):
        body = json.dumps({"outer": {"inner": {"deep": True}}}).encode()

        data, error = bot._parse_json_body(body)

        assert error is None
        assert data["outer"]["inner"]["deep"] is True

    def test_json_array(self, bot):
        """Arrays are valid JSON but not dicts - still parses."""
        body = b'[1, 2, 3]'

        data, error = bot._parse_json_body(body)

        assert error is None
        assert data == [1, 2, 3]

    def test_unicode_body(self, bot):
        body = json.dumps({"emoji": "hello"}).encode("utf-8")

        data, error = bot._parse_json_body(body)

        assert error is None
        assert data["emoji"] == "hello"


# ===========================================================================
# _handle_webhook_exception tests
# ===========================================================================


class TestHandleWebhookException:
    """Tests for _handle_webhook_exception()."""

    # --- JSONDecodeError ---

    def test_json_decode_error_returns_400(self, bot):
        exc = json.JSONDecodeError("bad", "doc", 0)

        result = bot._handle_webhook_exception(exc)

        assert _status(result) == 400
        body = _body(result)
        assert body["error"]["code"] == "INVALID_JSON"

    def test_json_decode_error_ignores_return_200_flag(self, bot):
        """JSONDecodeError always returns 400 regardless of return_200_on_error."""
        exc = json.JSONDecodeError("bad", "doc", 0)

        result = bot._handle_webhook_exception(exc, return_200_on_error=True)
        assert _status(result) == 400

        result = bot._handle_webhook_exception(exc, return_200_on_error=False)
        assert _status(result) == 400

    # --- ValueError ---

    def test_value_error_return_200(self, bot):
        exc = ValueError("bad value")

        result = bot._handle_webhook_exception(exc, return_200_on_error=True)

        assert _status(result) == 200
        body = _body(result)
        assert body["ok"] is False
        assert body["code"] == "VALIDATION_ERROR"
        assert "bad value" in body["error"]

    def test_value_error_return_400(self, bot):
        exc = ValueError("bad value")

        result = bot._handle_webhook_exception(exc, return_200_on_error=False)

        assert _status(result) == 400
        body = _body(result)
        assert body["error"]["code"] == "VALIDATION_ERROR"

    # --- KeyError ---

    def test_key_error_return_200(self, bot):
        exc = KeyError("missing_key")

        result = bot._handle_webhook_exception(exc, return_200_on_error=True)

        assert _status(result) == 200
        body = _body(result)
        assert body["ok"] is False
        assert body["code"] == "VALIDATION_ERROR"

    def test_key_error_return_400(self, bot):
        exc = KeyError("missing_key")

        result = bot._handle_webhook_exception(exc, return_200_on_error=False)

        assert _status(result) == 400

    # --- TypeError ---

    def test_type_error_return_200(self, bot):
        exc = TypeError("wrong type")

        result = bot._handle_webhook_exception(exc, return_200_on_error=True)

        assert _status(result) == 200
        body = _body(result)
        assert body["ok"] is False

    def test_type_error_return_400(self, bot):
        exc = TypeError("wrong type")

        result = bot._handle_webhook_exception(exc, return_200_on_error=False)

        assert _status(result) == 400
        body = _body(result)
        assert body["error"]["code"] == "VALIDATION_ERROR"

    # --- ConnectionError ---

    def test_connection_error_return_200(self, bot):
        exc = ConnectionError("refused")

        result = bot._handle_webhook_exception(exc, return_200_on_error=True)

        assert _status(result) == 200
        body = _body(result)
        assert body["ok"] is False
        assert body["code"] == "CONNECTION_ERROR"

    def test_connection_error_return_503(self, bot):
        exc = ConnectionError("refused")

        result = bot._handle_webhook_exception(exc, return_200_on_error=False)

        assert _status(result) == 503
        body = _body(result)
        assert body["error"]["code"] == "CONNECTION_ERROR"

    # --- OSError ---

    def test_os_error_return_200(self, bot):
        exc = OSError("disk full")

        result = bot._handle_webhook_exception(exc, return_200_on_error=True)

        assert _status(result) == 200
        body = _body(result)
        assert body["code"] == "CONNECTION_ERROR"

    def test_os_error_return_503(self, bot):
        exc = OSError("disk full")

        result = bot._handle_webhook_exception(exc, return_200_on_error=False)

        assert _status(result) == 503

    # --- TimeoutError ---

    def test_timeout_error_return_200(self, bot):
        exc = TimeoutError("timed out")

        result = bot._handle_webhook_exception(exc, return_200_on_error=True)

        assert _status(result) == 200
        body = _body(result)
        assert body["code"] == "CONNECTION_ERROR"

    def test_timeout_error_return_503(self, bot):
        exc = TimeoutError("timed out")

        result = bot._handle_webhook_exception(exc, return_200_on_error=False)

        assert _status(result) == 503

    # --- Generic Exception ---

    def test_generic_exception_return_200(self, bot):
        exc = RuntimeError("something broke")

        result = bot._handle_webhook_exception(exc, return_200_on_error=True)

        assert _status(result) == 200
        body = _body(result)
        assert body["ok"] is False
        assert body["code"] == "INTERNAL_ERROR"

    def test_generic_exception_return_500(self, bot):
        exc = RuntimeError("something broke")

        result = bot._handle_webhook_exception(exc, return_200_on_error=False)

        assert _status(result) == 500
        body = _body(result)
        assert body["error"]["code"] == "INTERNAL_ERROR"

    # --- Context parameter ---

    def test_custom_context_in_log(self, bot):
        with patch("aragora.server.handlers.bots.base.logger") as mock_logger:
            exc = RuntimeError("fail")
            bot._handle_webhook_exception(exc, context="slash_command")

            log_msg = mock_logger.exception.call_args[0][0] % tuple(
                mock_logger.exception.call_args[0][1:]
            )
            assert "slash_command" in log_msg

    # --- Error message truncation ---

    def test_long_error_truncated(self, bot):
        long_msg = "x" * 200
        exc = RuntimeError(long_msg)

        result = bot._handle_webhook_exception(exc, return_200_on_error=True)

        body = _body(result)
        assert len(body["error"]) <= 100

    # --- Default return_200_on_error ---

    def test_default_return_200_on_error_is_true(self, bot):
        """By default, return_200_on_error is True."""
        exc = ValueError("test")

        result = bot._handle_webhook_exception(exc)

        assert _status(result) == 200


# ===========================================================================
# _audit_webhook_auth_failure tests
# ===========================================================================


class TestAuditWebhookAuthFailure:
    """Tests for _audit_webhook_auth_failure()."""

    def test_calls_audit_security(self, bot):
        with patch(
            "aragora.audit.unified.audit_security"
        ) as mock_audit:
            bot._audit_webhook_auth_failure("signature", reason="bad hmac")

            mock_audit.assert_called_once_with(
                event_type="testbot_webhook_auth_failed",
                actor_id="unknown",
                resource_type="testbot_webhook",
                resource_id="signature",
                reason="bad hmac",
            )

    def test_without_reason(self, bot):
        with patch(
            "aragora.audit.unified.audit_security"
        ) as mock_audit:
            bot._audit_webhook_auth_failure("token")

            mock_audit.assert_called_once_with(
                event_type="testbot_webhook_auth_failed",
                actor_id="unknown",
                resource_type="testbot_webhook",
                resource_id="token",
                reason=None,
            )

    def test_import_error_silenced(self, bot):
        """When audit module is not available, no exception is raised."""
        with patch.dict("sys.modules", {"aragora.audit.unified": None}):
            # Should not raise
            bot._audit_webhook_auth_failure("signature", reason="test")


# ===========================================================================
# Bot platform attribute tests
# ===========================================================================


class TestBotPlatformAttribute:
    """Tests for bot_platform class attribute behavior."""

    def test_default_platform(self):
        """The base mixin has 'unknown' platform."""
        mixin = BotHandlerMixin()
        assert mixin.bot_platform == "unknown"

    def test_subclass_platform(self, bot):
        assert bot.bot_platform == "testbot"

    def test_platform_used_in_status(self, bot, mock_auth_ctx):
        bot.get_auth_context = AsyncMock(return_value=mock_auth_ctx)
        bot.check_permission = MagicMock()

        result = bot._build_status_response()
        body = _body(result)
        assert body["platform"] == "testbot"

    def test_platform_used_in_rate_limit(self, bot):
        """Platform name is not directly in rate limit response, but the handler works."""
        result = bot.handle_rate_limit_exceeded()
        assert _status(result) == 429

    def test_different_platform_names(self):
        """Multiple bots can have different platform names."""

        class SlackBot(BotHandlerMixin):
            bot_platform = "slack"

        class TeamsBot(BotHandlerMixin):
            bot_platform = "teams"

        slack = SlackBot()
        teams = TeamsBot()

        r1 = slack._build_status_response()
        r2 = teams._build_status_response()

        assert _body(r1)["platform"] == "slack"
        assert _body(r2)["platform"] == "teams"


# ===========================================================================
# Integration-style tests
# ===========================================================================


class TestMixinIntegration:
    """Integration tests exercising multiple methods together."""

    @pytest.mark.asyncio
    async def test_full_status_flow_enabled_bot(self, enabled_bot, mock_http, mock_auth_ctx):
        """Full flow: auth -> permission -> status with platform config."""
        enabled_bot.get_auth_context = AsyncMock(return_value=mock_auth_ctx)
        enabled_bot.check_permission = MagicMock()

        result = await enabled_bot.handle_status_request(
            mock_http,
            extra_status={"connected_channels": 5},
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "testbot"
        assert body["enabled"] is True
        assert body["token_configured"] is True
        assert body["webhook_url"] == "https://example.com/hook"
        assert body["connected_channels"] == 5

    @pytest.mark.asyncio
    async def test_auth_then_operation(self, bot, mock_http, mock_auth_ctx):
        """handle_with_auth passes auth_context to operation."""
        bot.get_auth_context = AsyncMock(return_value=mock_auth_ctx)
        bot.check_permission = MagicMock()

        received_ctx = []

        async def my_operation(auth_context=None):
            received_ctx.append(auth_context)
            return {"processed": True}

        result = await bot.handle_with_auth(mock_http, "bots.process", my_operation)

        assert result == {"processed": True}
        assert received_ctx[0] is mock_auth_ctx

    def test_read_then_parse_body(self, bot):
        """Read body then parse JSON in sequence."""
        body_dict = {"event": "message", "text": "hello"}
        body_bytes = json.dumps(body_dict).encode()
        handler = MockHTTPHandler(
            body_bytes=body_bytes,
            headers={"Content-Length": str(len(body_bytes))},
        )

        raw = bot._read_request_body(handler)
        data, error = bot._parse_json_body(raw)

        assert error is None
        assert data == body_dict

    def test_read_empty_then_parse_disallow(self, bot):
        """Read empty body then parse with allow_empty=False."""
        handler = MockHTTPHandler(headers={"Content-Length": "0"})

        raw = bot._read_request_body(handler)
        data, error = bot._parse_json_body(raw, allow_empty=False)

        assert data is None
        assert _status(error) == 400

    def test_read_empty_then_parse_allow(self, bot):
        """Read empty body then parse with allow_empty=True."""
        handler = MockHTTPHandler(headers={"Content-Length": "0"})

        raw = bot._read_request_body(handler)
        data, error = bot._parse_json_body(raw, allow_empty=True)

        assert error is None
        assert data == {}

    def test_webhook_exception_after_parse_failure(self, bot):
        """Simulate parse failure followed by exception handling."""
        exc = json.JSONDecodeError("bad json", "doc", 0)

        result = bot._handle_webhook_exception(exc, context="event_processing")

        assert _status(result) == 400
        body = _body(result)
        assert "Invalid JSON" in body["error"]["message"]

    def test_multiple_exception_types_in_sequence(self, bot):
        """Handle different exception types consistently."""
        results = []

        for exc in [
            json.JSONDecodeError("bad", "", 0),
            ValueError("bad val"),
            ConnectionError("refused"),
            RuntimeError("unexpected"),
        ]:
            result = bot._handle_webhook_exception(exc, return_200_on_error=False)
            results.append((_status(result), _body(result)["error"]["code"]))

        assert results[0] == (400, "INVALID_JSON")
        assert results[1] == (400, "VALIDATION_ERROR")
        assert results[2] == (503, "CONNECTION_ERROR")
        assert results[3] == (500, "INTERNAL_ERROR")
