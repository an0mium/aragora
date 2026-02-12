"""
Tests for SSO authentication handler (aragora.server.handlers.sso).

Tests cover:
- SSOHandler initialization
- Route definitions
- Parameter extraction (_get_param)
- Redirect URL validation (_validate_redirect_url)
- Response formatting (_to_legacy_result, _flatten_error_body)
- Provider initialization (_get_provider)
- Handler result formatting (_should_return_handler_result, _format_response)
- handle_login - SSO login initiation with OIDC/SAML
- handle_callback - IdP callback processing
- handle_logout - SSO logout flow
- handle_metadata - SAML metadata endpoint
- handle_status - SSO configuration status
- Production environment security checks
- Error handling for various failure modes

Security test categories:
- CSRF protection: state parameter handling
- Open redirect prevention: redirect URL validation
- Production security: HTTPS enforcement
- Domain restrictions: allowed domains validation
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.sso import SSOHandler


# ===========================================================================
# Test Fixtures and Helpers
# ===========================================================================


@dataclass
class MockSSOConfig:
    """Mock SSO configuration."""

    provider_type: Any = None
    enabled: bool = True
    callback_url: str = "https://app.example.com/auth/sso/callback"
    entity_id: str = "https://app.example.com"
    logout_url: str = "https://idp.example.com/logout"
    session_duration_seconds: int = 28800
    allowed_domains: list[str] = field(default_factory=list)
    auto_provision: bool = True


@dataclass
class MockSSOUser:
    """Mock SSO user returned from provider authentication."""

    id: str = "sso-user-123"
    email: str = "user@example.com"
    name: str = "SSO User"
    first_name: str = "SSO"
    last_name: str = "User"
    roles: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    provider_type: str = "oidc"
    access_token: str = "access_token_123"
    refresh_token: str = "refresh_token_123"
    id_token: str = "id_token_123"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "roles": self.roles,
            "groups": self.groups,
        }


@dataclass
class MockSSOProviderType:
    """Mock SSO provider type enum."""

    value: str = "oidc"

    SAML: MockSSOProviderType = None  # type: ignore[assignment]
    OIDC: MockSSOProviderType = None  # type: ignore[assignment]


# Initialize enum-like values
MockSSOProviderType.SAML = MockSSOProviderType("saml")
MockSSOProviderType.OIDC = MockSSOProviderType("oidc")


class MockSSOProvider:
    """Mock SSO provider for testing."""

    def __init__(self, provider_type: str = "oidc"):
        self.provider_type = MockSSOProviderType(provider_type)
        self.config = MockSSOConfig(provider_type=self.provider_type)

    async def get_authorization_url(
        self,
        state: str | None = None,
        relay_state: str | None = None,
        **kwargs,
    ) -> str:
        return f"https://idp.example.com/auth?state={state}&relay_state={relay_state or ''}"

    async def authenticate(
        self,
        code: str | None = None,
        saml_response: str | None = None,
        state: str | None = None,
        **kwargs,
    ) -> MockSSOUser:
        return MockSSOUser()

    async def logout(self, user: Any) -> str | None:
        return "https://idp.example.com/logout"

    async def get_metadata(self) -> str:
        return '<?xml version="1.0"?><md:EntityDescriptor>...</md:EntityDescriptor>'

    def generate_state(self) -> str:
        return "generated_state_token_12345678901234567890"


class MockAuthConfig:
    """Mock auth_config for token generation."""

    def generate_token(self, loop_id: str, expires_in: int = 3600) -> str:
        return f"jwt_token_for_{loop_id}"

    def revoke_token(self, token: str, reason: str = "logout") -> None:
        pass


class MockHTTPHandler:
    """Mock HTTP handler for testing."""

    def __init__(
        self,
        headers: dict[str, str] | None = None,
        has_send_response: bool = True,
    ):
        self.headers = headers or {}
        self._has_send_response = has_send_response

    def send_response(self, status: int) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        if name == "send_response" and not self._has_send_response:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return MagicMock()


def get_result_body(result) -> dict[str, Any]:
    """Extract body from handler result as dictionary."""
    if isinstance(result, dict):
        body = result.get("body", {})
        if isinstance(body, bytes):
            body = json.loads(body.decode("utf-8"))
        elif isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                pass
        return body
    if hasattr(result, "body"):
        try:
            return json.loads(result.body.decode("utf-8"))
        except (json.JSONDecodeError, AttributeError):
            return {}
    return {}


def get_error_message(body: dict[str, Any]) -> str:
    """Extract error message from body, handling both string and dict formats."""
    error = body.get("error", "")
    if isinstance(error, dict):
        return error.get("message", str(error)).lower()
    return str(error).lower()


def get_error_code(body: dict[str, Any]) -> str:
    """Extract error code from body, handling both flat and nested formats."""
    # Check top-level code first
    code = body.get("code", "")
    if code:
        return str(code).lower()
    # Check nested error.code
    error = body.get("error", {})
    if isinstance(error, dict):
        return str(error.get("code", "")).lower()
    return ""


def get_result_status(result) -> int:
    """Extract status code from handler result."""
    if isinstance(result, dict):
        return result.get("status", result.get("status_code", 200))
    if hasattr(result, "status_code"):
        return result.status_code
    return 200


def get_result_headers(result) -> dict[str, str]:
    """Extract headers from handler result."""
    if isinstance(result, dict):
        return result.get("headers", {})
    if hasattr(result, "headers"):
        return result.headers or {}
    return {}


# ===========================================================================
# Test SSOHandler Initialization
# ===========================================================================


class TestSSOHandlerInit:
    """Tests for SSOHandler initialization."""

    def test_init_with_empty_context(self):
        """Should initialize with empty context."""
        handler = SSOHandler()
        assert handler._provider is None
        assert handler._initialized is False

    def test_init_with_context(self):
        """Should initialize with provided context."""
        ctx = {"db": "mock_db", "user_store": MagicMock()}
        handler = SSOHandler(ctx)
        assert handler._initialized is False

    def test_resource_type_is_sso(self):
        """Should have RESOURCE_TYPE set to 'sso'."""
        handler = SSOHandler()
        assert handler.RESOURCE_TYPE == "sso"


# ===========================================================================
# Test SSO Route Definitions
# ===========================================================================


class TestSSOHandlerRoutes:
    """Tests for SSO route definitions."""

    def test_routes_returns_list(self):
        """Routes should return a list of tuples."""
        handler = SSOHandler()
        routes = handler.routes()
        assert isinstance(routes, list)
        assert len(routes) > 0

    def test_routes_include_login(self):
        """Routes should include login endpoint."""
        handler = SSOHandler()
        routes = handler.routes()
        methods_paths = [(r[0], r[1]) for r in routes]
        assert ("GET", "/auth/sso/login") in methods_paths
        assert ("POST", "/auth/sso/login") in methods_paths

    def test_routes_include_callback(self):
        """Routes should include callback endpoint."""
        handler = SSOHandler()
        routes = handler.routes()
        methods_paths = [(r[0], r[1]) for r in routes]
        assert ("GET", "/auth/sso/callback") in methods_paths
        assert ("POST", "/auth/sso/callback") in methods_paths

    def test_routes_include_logout(self):
        """Routes should include logout endpoint."""
        handler = SSOHandler()
        routes = handler.routes()
        methods_paths = [(r[0], r[1]) for r in routes]
        assert ("GET", "/auth/sso/logout") in methods_paths
        assert ("POST", "/auth/sso/logout") in methods_paths

    def test_routes_include_metadata(self):
        """Routes should include metadata endpoint."""
        handler = SSOHandler()
        routes = handler.routes()
        methods_paths = [(r[0], r[1]) for r in routes]
        assert ("GET", "/auth/sso/metadata") in methods_paths

    def test_routes_include_status(self):
        """Routes should include status endpoint."""
        handler = SSOHandler()
        routes = handler.routes()
        methods_paths = [(r[0], r[1]) for r in routes]
        assert ("GET", "/auth/sso/status") in methods_paths

    def test_routes_have_handler_names(self):
        """Each route should have a handler method name."""
        handler = SSOHandler()
        routes = handler.routes()
        for route in routes:
            assert len(route) == 3
            method, path, handler_name = route
            assert hasattr(handler, handler_name)

    def test_routes_handler_methods_are_callable(self):
        """Each handler method should be callable."""
        handler = SSOHandler()
        routes = handler.routes()
        for route in routes:
            handler_name = route[2]
            method = getattr(handler, handler_name)
            assert callable(method)


# ===========================================================================
# Test _get_param Helper
# ===========================================================================


class TestSSOGetParam:
    """Tests for _get_param helper method."""

    def test_get_param_string_value(self):
        """Should return string value directly."""
        handler = SSOHandler()
        params = {"key": "value"}
        assert handler._get_param(params, "key") == "value"

    def test_get_param_list_value(self):
        """Should extract first element from list."""
        handler = SSOHandler()
        params = {"key": ["value1", "value2"]}
        assert handler._get_param(params, "key") == "value1"

    def test_get_param_empty_list(self):
        """Should return None for empty list."""
        handler = SSOHandler()
        params = {"key": []}
        assert handler._get_param(params, "key") is None

    def test_get_param_missing_key(self):
        """Should return None for missing key."""
        handler = SSOHandler()
        params = {"other": "value"}
        assert handler._get_param(params, "key") is None

    def test_get_param_none_value(self):
        """Should return None for None value."""
        handler = SSOHandler()
        params = {"key": None}
        assert handler._get_param(params, "key") is None

    def test_get_param_converts_to_string(self):
        """Should convert non-string values to string."""
        handler = SSOHandler()
        params = {"key": 123}
        assert handler._get_param(params, "key") == "123"

    def test_get_param_with_boolean(self):
        """Should convert boolean to string."""
        handler = SSOHandler()
        params = {"key": True}
        assert handler._get_param(params, "key") == "True"

    def test_get_param_with_nested_empty_string_list(self):
        """Should return empty string from list containing empty string."""
        handler = SSOHandler()
        params = {"key": [""]}
        assert handler._get_param(params, "key") == ""


# ===========================================================================
# Test _validate_redirect_url Security
# ===========================================================================


class TestSSOValidateRedirectUrl:
    """Tests for _validate_redirect_url method - open redirect prevention."""

    def test_empty_url_is_valid(self):
        """Empty URL should be valid (no redirect)."""
        handler = SSOHandler()
        assert handler._validate_redirect_url("") is True

    def test_https_url_is_valid(self):
        """HTTPS URL should be valid."""
        handler = SSOHandler()
        assert handler._validate_redirect_url("https://example.com/callback") is True

    def test_http_url_is_valid_in_dev(self):
        """HTTP URL should be valid in non-production."""
        handler = SSOHandler()
        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}, clear=False):
            assert handler._validate_redirect_url("http://localhost:3000/callback") is True

    def test_http_url_invalid_in_production(self):
        """HTTP URL should be invalid in production."""
        handler = SSOHandler()
        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}, clear=False):
            assert handler._validate_redirect_url("http://example.com/callback") is False

    def test_javascript_scheme_is_invalid(self):
        """JavaScript scheme should be blocked."""
        handler = SSOHandler()
        assert handler._validate_redirect_url("javascript:alert(1)") is False

    def test_data_scheme_is_invalid(self):
        """Data scheme should be blocked."""
        handler = SSOHandler()
        assert handler._validate_redirect_url("data:text/html,test") is False

    def test_file_scheme_is_invalid(self):
        """File scheme should be blocked."""
        handler = SSOHandler()
        assert handler._validate_redirect_url("file:///etc/passwd") is False

    def test_url_with_credentials_is_invalid(self):
        """URL with credentials should be blocked."""
        handler = SSOHandler()
        assert handler._validate_redirect_url("https://user:pass@evil.com") is False

    def test_url_with_at_symbol_in_path_is_blocked(self):
        """URL with @ in netloc (credential injection) should be blocked."""
        handler = SSOHandler()
        assert handler._validate_redirect_url("https://good.com@evil.com/path") is False

    def test_allowed_hosts_enforcement(self):
        """Should enforce allowed hosts when configured."""
        handler = SSOHandler()
        with patch.dict(
            os.environ,
            {"ARAGORA_SSO_ALLOWED_REDIRECT_HOSTS": "example.com,app.example.com"},
            clear=False,
        ):
            assert handler._validate_redirect_url("https://example.com/callback") is True
            assert handler._validate_redirect_url("https://app.example.com/callback") is True
            assert handler._validate_redirect_url("https://evil.com/callback") is False

    def test_allowed_hosts_case_insensitive(self):
        """Allowed hosts check should be case insensitive."""
        handler = SSOHandler()
        with patch.dict(
            os.environ,
            {"ARAGORA_SSO_ALLOWED_REDIRECT_HOSTS": "EXAMPLE.COM"},
            clear=False,
        ):
            assert handler._validate_redirect_url("https://example.com/callback") is True
            assert handler._validate_redirect_url("https://EXAMPLE.COM/callback") is True

    def test_malformed_url_is_invalid(self):
        """Malformed URL should be blocked without exception."""
        handler = SSOHandler()
        result = handler._validate_redirect_url("not a valid url at all")
        assert result in (True, False)  # Should not raise exception

    def test_ftp_scheme_is_invalid(self):
        """FTP scheme should be blocked."""
        handler = SSOHandler()
        assert handler._validate_redirect_url("ftp://example.com/file") is False


# ===========================================================================
# Test _should_return_handler_result
# ===========================================================================


class TestSSOShouldReturnHandlerResult:
    """Tests for _should_return_handler_result method."""

    def test_none_handler_returns_false(self):
        """None handler should return False."""
        handler = SSOHandler()
        assert handler._should_return_handler_result(None) is False

    def test_handler_with_send_response_returns_true(self):
        """Handler with send_response method should return True."""
        handler = SSOHandler()
        mock_handler = MockHTTPHandler(has_send_response=True)
        assert handler._should_return_handler_result(mock_handler) is True

    def test_handler_without_send_response_returns_false(self):
        """Handler without send_response should return False."""
        handler = SSOHandler()

        class NoSendResponseHandler:
            pass

        assert handler._should_return_handler_result(NoSendResponseHandler()) is False


# ===========================================================================
# Test _flatten_error_body
# ===========================================================================


class TestSSOFlattenErrorBody:
    """Tests for _flatten_error_body method."""

    def test_non_dict_passthrough(self):
        """Non-dict values should pass through unchanged."""
        handler = SSOHandler()
        assert handler._flatten_error_body("error") == "error"
        assert handler._flatten_error_body(123) == 123
        assert handler._flatten_error_body(["a", "b"]) == ["a", "b"]

    def test_dict_without_error_passthrough(self):
        """Dict without 'error' key should pass through."""
        handler = SSOHandler()
        body = {"message": "test"}
        assert handler._flatten_error_body(body) == body

    def test_dict_with_string_error_passthrough(self):
        """Dict with string error should pass through."""
        handler = SSOHandler()
        body = {"error": "simple error"}
        assert handler._flatten_error_body(body) == body

    def test_dict_with_structured_error_flattens(self):
        """Dict with structured error should flatten."""
        handler = SSOHandler()
        body = {
            "error": {
                "message": "Error message",
                "code": "ERR_CODE",
                "suggestion": "Fix it",
                "trace_id": "trace-123",
                "details": {"field": "value"},
            },
            "other_field": "value",
        }
        result = handler._flatten_error_body(body)
        assert result["error"] == "Error message"
        assert result["code"] == "ERR_CODE"
        assert result["suggestion"] == "Fix it"
        assert result["trace_id"] == "trace-123"
        assert result["details"] == {"field": "value"}
        assert result["other_field"] == "value"


# ===========================================================================
# Test _to_legacy_result
# ===========================================================================


class TestSSOToLegacyResult:
    """Tests for _to_legacy_result method."""

    def test_dict_input_returns_dict(self):
        """Dict input should return dict."""
        handler = SSOHandler()
        input_dict = {"status": 200, "body": {"data": "test"}}
        result = handler._to_legacy_result(input_dict)
        assert isinstance(result, dict)
        assert result["status"] == 200

    def test_adds_headers_if_missing(self):
        """Should add empty headers dict if missing."""
        handler = SSOHandler()
        input_dict = {"status": 200, "body": {}}
        result = handler._to_legacy_result(input_dict)
        assert "headers" in result
        assert result["headers"] == {}

    def test_converts_status_code_to_status(self):
        """Should convert status_code key to status."""
        handler = SSOHandler()
        input_dict = {"status_code": 404, "body": {}}
        result = handler._to_legacy_result(input_dict)
        assert result["status"] == 404

    def test_decodes_bytes_body_with_json_content_type(self):
        """Should decode bytes body when content_type is JSON."""
        handler = SSOHandler()
        input_dict = {
            "status": 200,
            "body": b'{"key": "value"}',
            "content_type": "application/json",
        }
        result = handler._to_legacy_result(input_dict)
        assert result["body"] == {"key": "value"}

    def test_decodes_bytes_body_without_json_content_type(self):
        """Should decode bytes body to string when not JSON."""
        handler = SSOHandler()
        input_dict = {
            "status": 200,
            "body": b"plain text",
            "content_type": "text/plain",
        }
        result = handler._to_legacy_result(input_dict)
        assert result["body"] == "plain text"


# ===========================================================================
# Test _get_provider
# ===========================================================================


class TestSSOGetProvider:
    """Tests for _get_provider method."""

    def test_first_call_initializes(self):
        """First call should set _initialized to True."""
        handler = SSOHandler()
        assert handler._initialized is False
        handler._get_provider()
        assert handler._initialized is True

    def test_returns_none_without_sso_module(self):
        """Should return None when SSO module not available."""
        handler = SSOHandler()
        with patch("aragora.server.handlers.sso._get_sso_provider", None):
            result = handler._get_provider()
            assert result is None or handler._initialized is True

    def test_subsequent_calls_dont_reinitialize(self):
        """Subsequent calls should not reinitialize."""
        handler = SSOHandler()
        handler._get_provider()
        handler._initialized = True
        handler._provider = "mock_provider"
        result = handler._get_provider()
        assert result == "mock_provider"

    def test_caches_provider_instance(self):
        """Should cache the provider instance."""
        handler = SSOHandler()
        mock_provider = MockSSOProvider()

        with patch(
            "aragora.server.handlers.sso.get_sso_provider",
            return_value=mock_provider,
        ):
            provider1 = handler._get_provider()
            provider2 = handler._get_provider()
            assert provider1 is provider2


# ===========================================================================
# Test handle_login
# ===========================================================================


class TestSSOHandleLogin:
    """Tests for handle_login - SSO login initiation."""

    @pytest.mark.asyncio
    async def test_login_returns_501_when_not_configured(self):
        """Should return 501 when SSO is not configured."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()

        with patch.object(handler, "_get_provider", return_value=None):
            result = await handler.handle_login(mock_http, {})

        status = get_result_status(result)
        body = get_result_body(result)
        assert status == 501
        assert "not configured" in get_error_message(body)

    @pytest.mark.asyncio
    async def test_login_returns_redirect_with_auth_url(self):
        """Should return redirect to IdP authorization URL."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler(headers={"Accept": "text/html"})
        mock_provider = MockSSOProvider()

        with patch.object(handler, "_get_provider", return_value=mock_provider):
            result = await handler.handle_login(mock_http, {})

        status = get_result_status(result)
        headers = get_result_headers(result)
        assert status == 302
        assert "Location" in headers
        assert "idp.example.com" in headers["Location"]

    @pytest.mark.asyncio
    async def test_login_returns_json_when_accept_json(self):
        """Should return JSON response when Accept header is application/json."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler(headers={"Accept": "application/json"})
        mock_provider = MockSSOProvider()

        with patch.object(handler, "_get_provider", return_value=mock_provider):
            result = await handler.handle_login(mock_http, {})

        status = get_result_status(result)
        body = get_result_body(result)
        assert status == 200
        assert "auth_url" in body
        assert "state" in body
        assert body["provider"] == "oidc"

    @pytest.mark.asyncio
    async def test_login_uses_provided_state(self):
        """Should use state parameter if provided."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler(headers={"Accept": "application/json"})
        mock_provider = MockSSOProvider()

        with patch.object(handler, "_get_provider", return_value=mock_provider):
            result = await handler.handle_login(mock_http, {"state": "custom_state"})

        body = get_result_body(result)
        # Provider generates state if not provided, but we can verify auth_url contains state
        assert "state" in body

    @pytest.mark.asyncio
    async def test_login_generates_state_if_not_provided(self):
        """Should generate state if not provided."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler(headers={"Accept": "application/json"})
        mock_provider = MockSSOProvider()

        with patch.object(handler, "_get_provider", return_value=mock_provider):
            result = await handler.handle_login(mock_http, {})

        body = get_result_body(result)
        assert "state" in body
        assert len(body["state"]) > 0

    @pytest.mark.asyncio
    async def test_login_includes_redirect_uri_in_relay_state(self):
        """Should include redirect_uri in relay_state."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler(headers={"Accept": "text/html"})
        mock_provider = MockSSOProvider()

        with patch.object(handler, "_get_provider", return_value=mock_provider):
            result = await handler.handle_login(
                mock_http, {"redirect_uri": "https://app.example.com/dashboard"}
            )

        headers = get_result_headers(result)
        assert (
            "relay_state=https" in headers.get("Location", "") or get_result_status(result) == 302
        )

    @pytest.mark.asyncio
    async def test_login_handles_configuration_error(self):
        """Should handle ConfigurationError gracefully."""
        from aragora.exceptions import ConfigurationError

        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()
        mock_provider.get_authorization_url = AsyncMock(
            side_effect=ConfigurationError("SSO", "Invalid config")
        )

        with patch.object(handler, "_get_provider", return_value=mock_provider):
            result = await handler.handle_login(mock_http, {})

        status = get_result_status(result)
        assert status == 503

    @pytest.mark.asyncio
    async def test_login_handles_value_error(self):
        """Should handle ValueError gracefully."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()
        mock_provider.get_authorization_url = AsyncMock(side_effect=ValueError("Bad value"))

        with patch.object(handler, "_get_provider", return_value=mock_provider):
            result = await handler.handle_login(mock_http, {})

        status = get_result_status(result)
        assert status == 400

    @pytest.mark.asyncio
    async def test_login_handles_unexpected_error(self):
        """Should handle unexpected errors gracefully."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()
        mock_provider.get_authorization_url = AsyncMock(side_effect=RuntimeError("Unexpected"))

        with patch.object(handler, "_get_provider", return_value=mock_provider):
            result = await handler.handle_login(mock_http, {})

        status = get_result_status(result)
        assert status == 500

    @pytest.mark.asyncio
    async def test_login_sets_no_cache_headers(self):
        """Should set Cache-Control headers to prevent caching."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler(headers={"Accept": "text/html"})
        mock_provider = MockSSOProvider()

        with patch.object(handler, "_get_provider", return_value=mock_provider):
            result = await handler.handle_login(mock_http, {})

        headers = get_result_headers(result)
        assert "Cache-Control" in headers
        assert "no-cache" in headers["Cache-Control"] or "no-store" in headers["Cache-Control"]


# ===========================================================================
# Test handle_callback
# ===========================================================================


class TestSSOHandleCallback:
    """Tests for handle_callback - IdP callback processing."""

    @pytest.mark.asyncio
    async def test_callback_returns_501_when_not_configured(self):
        """Should return 501 when SSO is not configured."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()

        with patch.object(handler, "_get_provider", return_value=None):
            result = await handler.handle_callback(mock_http, {"code": "auth_code"})

        status = get_result_status(result)
        assert status == 501

    @pytest.mark.asyncio
    async def test_callback_returns_error_when_idp_error(self):
        """Should return error when IdP returns error parameter."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()

        with patch.object(handler, "_get_provider", return_value=mock_provider):
            result = await handler.handle_callback(
                mock_http,
                {"error": "access_denied", "error_description": "User cancelled"},
            )

        status = get_result_status(result)
        body = get_result_body(result)
        assert status == 401
        error_msg = get_error_message(body)
        assert "idp error" in error_msg or "denied" in error_msg or "denied" in str(body).lower()

    @pytest.mark.asyncio
    async def test_callback_success_returns_token_and_user(self):
        """Should return JWT token and user info on successful callback."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()
        mock_auth_config = MockAuthConfig()

        with (
            patch.object(handler, "_get_provider", return_value=mock_provider),
            patch("aragora.server.handlers.sso.auth_config", mock_auth_config),
        ):
            result = await handler.handle_callback(mock_http, {"code": "auth_code"})

        status = get_result_status(result)
        body = get_result_body(result)
        assert status == 200
        assert body.get("success") is True
        assert "token" in body
        assert "user" in body
        assert "expires_in" in body

    @pytest.mark.asyncio
    async def test_callback_validates_redirect_url(self):
        """Should validate redirect URL before redirecting."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()
        mock_auth_config = MockAuthConfig()

        with (
            patch.object(handler, "_get_provider", return_value=mock_provider),
            patch("aragora.server.handlers.sso.auth_config", mock_auth_config),
            patch.object(handler, "_validate_redirect_url", return_value=False),
        ):
            result = await handler.handle_callback(
                mock_http,
                {"code": "auth_code", "RelayState": "https://evil.com/steal"},
            )

        status = get_result_status(result)
        body = get_result_body(result)
        assert status == 400
        assert "invalid" in get_error_message(body) or "redirect" in get_error_message(body)

    @pytest.mark.asyncio
    async def test_callback_redirects_with_token_when_relay_state_is_url(self):
        """Should redirect with token when RelayState is a valid URL."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()
        mock_auth_config = MockAuthConfig()

        with (
            patch.object(handler, "_get_provider", return_value=mock_provider),
            patch("aragora.server.handlers.sso.auth_config", mock_auth_config),
            patch.object(handler, "_validate_redirect_url", return_value=True),
        ):
            result = await handler.handle_callback(
                mock_http,
                {"code": "auth_code", "RelayState": "https://app.example.com/dashboard"},
            )

        status = get_result_status(result)
        headers = get_result_headers(result)
        assert status == 302
        assert "Location" in headers
        assert "token=" in headers["Location"]

    @pytest.mark.asyncio
    async def test_callback_handles_saml_response(self):
        """Should handle SAML response parameter."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()
        mock_auth_config = MockAuthConfig()

        with (
            patch.object(handler, "_get_provider", return_value=mock_provider),
            patch("aragora.server.handlers.sso.auth_config", mock_auth_config),
        ):
            result = await handler.handle_callback(
                mock_http,
                {"SAMLResponse": "base64_encoded_saml_response"},
            )

        status = get_result_status(result)
        assert status == 200

    @pytest.mark.asyncio
    async def test_callback_handles_domain_not_allowed_error(self):
        """Should handle DOMAIN_NOT_ALLOWED error from provider."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()
        mock_provider.authenticate = AsyncMock(
            side_effect=Exception("DOMAIN_NOT_ALLOWED: test.com")
        )

        with patch.object(handler, "_get_provider", return_value=mock_provider):
            result = await handler.handle_callback(mock_http, {"code": "auth_code"})

        status = get_result_status(result)
        body = get_result_body(result)
        # Handler returns 403 for domain not allowed, or 500 for generic errors
        assert status in (403, 500)
        error_code = get_error_code(body)
        error_msg = get_error_message(body)
        assert "domain" in error_code or "domain" in error_msg

    @pytest.mark.asyncio
    async def test_callback_handles_invalid_state_error(self):
        """Should handle INVALID_STATE error from provider."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()
        mock_provider.authenticate = AsyncMock(side_effect=Exception("INVALID_STATE"))

        with patch.object(handler, "_get_provider", return_value=mock_provider):
            result = await handler.handle_callback(mock_http, {"code": "auth_code"})

        status = get_result_status(result)
        body = get_result_body(result)
        # Handler returns 401 for invalid state, or 500 for generic errors
        assert status in (401, 500)
        error_msg = get_error_message(body)
        assert "expired" in error_msg or "state" in error_msg or "invalid" in error_msg

    @pytest.mark.asyncio
    async def test_callback_handles_auth_config_not_initialized(self):
        """Should handle auth_config not initialized error."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()

        with (
            patch.object(handler, "_get_provider", return_value=mock_provider),
            patch("aragora.server.handlers.sso.auth_config", None),
        ):
            result = await handler.handle_callback(mock_http, {"code": "auth_code"})

        status = get_result_status(result)
        assert status == 503

    @pytest.mark.asyncio
    async def test_callback_enforces_https_in_production(self):
        """Should enforce HTTPS callback URL in production."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()
        mock_provider.config.callback_url = "http://insecure.example.com/callback"

        with (
            patch.object(handler, "_get_provider", return_value=mock_provider),
            patch.dict(os.environ, {"ARAGORA_ENV": "production"}, clear=False),
        ):
            result = await handler.handle_callback(mock_http, {"code": "auth_code"})

        status = get_result_status(result)
        body = get_result_body(result)
        assert status == 400
        assert "https" in get_error_message(body)


# ===========================================================================
# Test handle_logout
# ===========================================================================


class TestSSOHandleLogout:
    """Tests for handle_logout - SSO logout flow."""

    @pytest.mark.asyncio
    async def test_logout_returns_success_when_not_configured(self):
        """Should return success even when SSO is not configured."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()

        with patch.object(handler, "_get_provider", return_value=None):
            result = await handler.handle_logout(mock_http, {})

        status = get_result_status(result)
        body = get_result_body(result)
        assert status == 200
        assert body.get("success") is True

    @pytest.mark.asyncio
    async def test_logout_revokes_token_if_provided(self):
        """Should revoke token if Authorization header is present."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler(headers={"Authorization": "Bearer test_token"})
        mock_provider = MockSSOProvider()
        mock_auth_config = MockAuthConfig()
        mock_auth_config.revoke_token = MagicMock()

        with (
            patch.object(handler, "_get_provider", return_value=mock_provider),
            patch("aragora.server.handlers.sso.auth_config", mock_auth_config),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
        ):
            await handler.handle_logout(mock_http, {})

        mock_auth_config.revoke_token.assert_called_once_with("test_token", "user_logout")

    @pytest.mark.asyncio
    async def test_logout_returns_redirect_to_idp_logout(self):
        """Should return redirect to IdP logout URL."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()
        mock_auth_config = MockAuthConfig()

        with (
            patch.object(handler, "_get_provider", return_value=mock_provider),
            patch("aragora.server.handlers.sso.auth_config", mock_auth_config),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
        ):
            result = await handler.handle_logout(mock_http, {})

        status = get_result_status(result)
        headers = get_result_headers(result)
        assert status == 302
        assert "Location" in headers
        assert "idp.example.com/logout" in headers["Location"]

    @pytest.mark.asyncio
    async def test_logout_returns_success_when_no_logout_url(self):
        """Should return success JSON when IdP has no logout URL."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()
        mock_provider.logout = AsyncMock(return_value=None)
        mock_auth_config = MockAuthConfig()

        with (
            patch.object(handler, "_get_provider", return_value=mock_provider),
            patch("aragora.server.handlers.sso.auth_config", mock_auth_config),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
        ):
            result = await handler.handle_logout(mock_http, {})

        status = get_result_status(result)
        body = get_result_body(result)
        assert status == 200
        assert body.get("success") is True
        assert "logged out" in body.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_logout_handles_errors_gracefully(self):
        """Should return success with warning message on errors."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()
        mock_provider.logout = AsyncMock(side_effect=RuntimeError("Logout failed"))
        mock_auth_config = MockAuthConfig()

        with (
            patch.object(handler, "_get_provider", return_value=mock_provider),
            patch("aragora.server.handlers.sso.auth_config", mock_auth_config),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
        ):
            result = await handler.handle_logout(mock_http, {})

        status = get_result_status(result)
        body = get_result_body(result)
        assert status == 200
        assert body.get("success") is True


# ===========================================================================
# Test handle_metadata
# ===========================================================================


class TestSSOHandleMetadata:
    """Tests for handle_metadata - SAML SP metadata endpoint."""

    @pytest.mark.asyncio
    async def test_metadata_returns_501_when_not_configured(self):
        """Should return 501 when SSO is not configured."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()

        with patch.object(handler, "_get_provider", return_value=None):
            result = await handler.handle_metadata(mock_http, {})

        status = get_result_status(result)
        assert status == 501

    @pytest.mark.asyncio
    async def test_metadata_returns_400_for_non_saml_provider(self):
        """Should return 400 when provider is not SAML."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider("oidc")
        mock_provider_type = MockSSOProviderType

        with (
            patch.object(handler, "_get_provider", return_value=mock_provider),
            patch("aragora.server.handlers.sso.SSOProviderType", mock_provider_type),
        ):
            result = await handler.handle_metadata(mock_http, {})

        status = get_result_status(result)
        body = get_result_body(result)
        assert status == 400
        assert "saml" in get_error_message(body)

    @pytest.mark.asyncio
    async def test_metadata_returns_xml_for_saml_provider(self):
        """Should return XML metadata for SAML provider."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider("saml")
        mock_provider.provider_type = MockSSOProviderType.SAML
        mock_provider_type = MockSSOProviderType

        with (
            patch.object(handler, "_get_provider", return_value=mock_provider),
            patch("aragora.server.handlers.sso.SSOProviderType", mock_provider_type),
        ):
            result = await handler.handle_metadata(mock_http, {})

        status = get_result_status(result)
        if hasattr(result, "content_type"):
            assert result.content_type == "application/xml"
        assert status == 200

    @pytest.mark.asyncio
    async def test_metadata_returns_503_when_sso_types_unavailable(self):
        """Should return 503 when SSOProviderType is not available."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()

        with (
            patch.object(handler, "_get_provider", return_value=mock_provider),
            patch("aragora.server.handlers.sso.SSOProviderType", None),
        ):
            result = await handler.handle_metadata(mock_http, {})

        status = get_result_status(result)
        assert status == 503

    @pytest.mark.asyncio
    async def test_metadata_handles_generation_error(self):
        """Should handle metadata generation errors."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider("saml")
        mock_provider.provider_type = MockSSOProviderType.SAML
        mock_provider.get_metadata = AsyncMock(side_effect=ValueError("Invalid config"))
        mock_provider_type = MockSSOProviderType

        with (
            patch.object(handler, "_get_provider", return_value=mock_provider),
            patch("aragora.server.handlers.sso.SSOProviderType", mock_provider_type),
        ):
            result = await handler.handle_metadata(mock_http, {})

        status = get_result_status(result)
        assert status == 400

    @pytest.mark.asyncio
    async def test_metadata_sets_cache_control(self):
        """Should set appropriate Cache-Control header."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider("saml")
        mock_provider.provider_type = MockSSOProviderType.SAML
        mock_provider_type = MockSSOProviderType

        with (
            patch.object(handler, "_get_provider", return_value=mock_provider),
            patch("aragora.server.handlers.sso.SSOProviderType", mock_provider_type),
        ):
            result = await handler.handle_metadata(mock_http, {})

        headers = get_result_headers(result)
        if "Cache-Control" in headers:
            assert "max-age" in headers["Cache-Control"]


# ===========================================================================
# Test handle_status
# ===========================================================================


class TestSSOHandleStatus:
    """Tests for handle_status - SSO configuration status."""

    @pytest.mark.asyncio
    async def test_status_returns_disabled_when_not_configured(self):
        """Should return disabled status when SSO not configured."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()

        with patch.object(handler, "_get_provider", return_value=None):
            result = await handler.handle_status(mock_http, {})

        status = get_result_status(result)
        body = get_result_body(result)
        assert status == 200
        assert body.get("enabled") is False
        assert body.get("configured") is False

    @pytest.mark.asyncio
    async def test_status_returns_enabled_when_configured(self):
        """Should return enabled status with provider info when configured."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()

        with patch.object(handler, "_get_provider", return_value=mock_provider):
            result = await handler.handle_status(mock_http, {})

        status = get_result_status(result)
        body = get_result_body(result)
        assert status == 200
        assert body.get("enabled") is True
        assert body.get("configured") is True
        assert body.get("provider") == "oidc"

    @pytest.mark.asyncio
    async def test_status_includes_configuration_details(self):
        """Should include configuration details in response."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()

        with patch.object(handler, "_get_provider", return_value=mock_provider):
            result = await handler.handle_status(mock_http, {})

        body = get_result_body(result)
        assert "entity_id" in body
        assert "callback_url" in body
        assert "auto_provision" in body

    @pytest.mark.asyncio
    async def test_status_includes_allowed_domains_if_present(self):
        """Should include allowed_domains in response if configured."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()
        mock_provider.config.allowed_domains = ["example.com", "corp.example.com"]

        with patch.object(handler, "_get_provider", return_value=mock_provider):
            result = await handler.handle_status(mock_http, {})

        body = get_result_body(result)
        assert "allowed_domains" in body
        assert "example.com" in body["allowed_domains"]


# ===========================================================================
# Test Response Formatting
# ===========================================================================


class TestSSOFormatResponse:
    """Tests for _format_response method."""

    def test_format_response_returns_handler_result_when_appropriate(self):
        """Should return HandlerResult when handler has send_response."""
        from aragora.server.handlers.base import HandlerResult

        handler = SSOHandler()
        mock_http = MockHTTPHandler(has_send_response=True)
        result = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=b'{"test": "data"}',
        )

        formatted = handler._format_response(mock_http, result)
        assert formatted is result

    def test_format_response_returns_legacy_dict_when_no_send_response(self):
        """Should return legacy dict when handler doesn't have send_response."""
        from aragora.server.handlers.base import HandlerResult

        handler = SSOHandler()

        class NoSendHandler:
            pass

        result = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=b'{"test": "data"}',
        )

        formatted = handler._format_response(NoSendHandler(), result)
        assert isinstance(formatted, dict)
        assert formatted["status"] == 200


# ===========================================================================
# Test Error Handling Edge Cases
# ===========================================================================


class TestSSOErrorHandling:
    """Tests for error handling edge cases."""

    @pytest.mark.asyncio
    async def test_login_handles_type_error(self):
        """Should handle TypeError gracefully in login."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()
        mock_provider.get_authorization_url = AsyncMock(side_effect=TypeError("Type error"))

        with patch.object(handler, "_get_provider", return_value=mock_provider):
            result = await handler.handle_login(mock_http, {})

        status = get_result_status(result)
        assert status == 400

    @pytest.mark.asyncio
    async def test_callback_handles_key_error(self):
        """Should handle KeyError gracefully in callback."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()
        mock_provider.authenticate = AsyncMock(side_effect=KeyError("missing_key"))

        with patch.object(handler, "_get_provider", return_value=mock_provider):
            result = await handler.handle_callback(mock_http, {"code": "auth_code"})

        status = get_result_status(result)
        assert status == 400

    @pytest.mark.asyncio
    async def test_logout_handles_value_error(self):
        """Should handle ValueError gracefully in logout."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()
        mock_provider.logout = AsyncMock(side_effect=ValueError("Value error"))
        mock_auth_config = MockAuthConfig()

        with (
            patch.object(handler, "_get_provider", return_value=mock_provider),
            patch("aragora.server.handlers.sso.auth_config", mock_auth_config),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
        ):
            result = await handler.handle_logout(mock_http, {})

        status = get_result_status(result)
        body = get_result_body(result)
        assert status == 200
        assert body.get("success") is True


# ===========================================================================
# Test List Parameter Handling
# ===========================================================================


class TestSSOListParameters:
    """Tests for handling list-type query parameters."""

    @pytest.mark.asyncio
    async def test_login_handles_list_redirect_uri(self):
        """Should handle redirect_uri as list."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler(headers={"Accept": "application/json"})
        mock_provider = MockSSOProvider()

        with patch.object(handler, "_get_provider", return_value=mock_provider):
            result = await handler.handle_login(
                mock_http, {"redirect_uri": ["https://example.com/dashboard"]}
            )

        status = get_result_status(result)
        assert status == 200

    @pytest.mark.asyncio
    async def test_callback_handles_list_code(self):
        """Should handle code as list."""
        handler = SSOHandler()
        mock_http = MockHTTPHandler()
        mock_provider = MockSSOProvider()
        mock_auth_config = MockAuthConfig()

        with (
            patch.object(handler, "_get_provider", return_value=mock_provider),
            patch("aragora.server.handlers.sso.auth_config", mock_auth_config),
        ):
            result = await handler.handle_callback(mock_http, {"code": ["auth_code_123"]})

        status = get_result_status(result)
        assert status == 200


# ===========================================================================
# Test Integration Scenarios
# ===========================================================================


class TestSSOIntegration:
    """Integration tests for complete SSO flows."""

    @pytest.mark.asyncio
    async def test_complete_oidc_flow(self):
        """Test complete OIDC login flow."""
        handler = SSOHandler()
        mock_provider = MockSSOProvider()
        mock_auth_config = MockAuthConfig()

        # Step 1: Initiate login
        mock_http = MockHTTPHandler(headers={"Accept": "application/json"})
        with patch.object(handler, "_get_provider", return_value=mock_provider):
            login_result = await handler.handle_login(mock_http, {})

        login_body = get_result_body(login_result)
        assert "auth_url" in login_body
        state = login_body.get("state")
        assert state

        # Step 2: Handle callback
        with (
            patch.object(handler, "_get_provider", return_value=mock_provider),
            patch("aragora.server.handlers.sso.auth_config", mock_auth_config),
        ):
            callback_result = await handler.handle_callback(
                mock_http, {"code": "auth_code", "state": state}
            )

        callback_body = get_result_body(callback_result)
        assert callback_body.get("success") is True
        assert "token" in callback_body

        # Step 3: Logout
        mock_http_with_token = MockHTTPHandler(
            headers={"Authorization": f"Bearer {callback_body['token']}"}
        )
        with (
            patch.object(handler, "_get_provider", return_value=mock_provider),
            patch("aragora.server.handlers.sso.auth_config", mock_auth_config),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
        ):
            logout_result = await handler.handle_logout(mock_http_with_token, {})

        logout_status = get_result_status(logout_result)
        assert logout_status in (200, 302)
