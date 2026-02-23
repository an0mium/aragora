"""Tests for SSO handler (aragora/server/handlers/sso.py).

Covers all routes and behavior of the SSOHandler class:
- can_handle() routing and path normalization
- GET/POST /auth/sso/login       - Initiate SSO login
- GET/POST /auth/sso/callback    - Handle IdP callback
- GET/POST /auth/sso/logout      - Handle logout
- GET      /auth/sso/metadata    - SAML SP metadata
- GET      /auth/sso/status      - SSO configuration status
- SDK v2 alias routing
- Error paths, validation, edge cases
- Redirect URL validation (_validate_redirect_url)
- Legacy result conversion
- Provider resolution
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.base import HandlerResult
from aragora.server.handlers.sso import SSOHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult | dict) -> dict:
    """Extract the JSON body from a HandlerResult or dict."""
    if isinstance(result, HandlerResult):
        if isinstance(result.body, bytes):
            try:
                return json.loads(result.body.decode("utf-8"))
            except (ValueError, TypeError):
                return {}
        return result.body
    if isinstance(result, dict):
        b = result.get("body", result)
        if isinstance(b, (bytes, str)):
            try:
                return json.loads(b if isinstance(b, str) else b.decode("utf-8"))
            except (ValueError, TypeError):
                return {}
        if isinstance(b, dict):
            return b
        return result
    return {}


def _error_msg(result: HandlerResult | dict) -> str:
    """Extract error message string from a result, handling structured errors."""
    body = _body(result)
    error = body.get("error", "")
    if isinstance(error, dict):
        return error.get("message", str(error))
    return str(error)


def _status(result: HandlerResult | dict) -> int:
    """Extract HTTP status code from a HandlerResult or dict."""
    if isinstance(result, HandlerResult):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status", result.get("status_code", 200))
    return 200


class MockHTTPHandler:
    """Mock HTTP handler for testing (simulates BaseHTTPRequestHandler)."""

    def __init__(self, body: dict[str, Any] | None = None):
        self.rfile = MagicMock()
        self.command = "GET"
        self._body = body
        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {"Content-Length": str(len(body_bytes))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}

    def send_response(self, *args: Any, **kwargs: Any) -> None:
        """Mock send_response for HandlerResult detection."""
        pass


def _make_handler(
    body: dict[str, Any] | None = None,
    method: str = "GET",
    headers: dict[str, str] | None = None,
) -> MockHTTPHandler:
    """Create a MockHTTPHandler with optional body, method, and headers."""
    h = MockHTTPHandler(body=body)
    h.command = method
    if headers:
        h.headers.update(headers)
    return h


# ---------------------------------------------------------------------------
# Mock SSO objects
# ---------------------------------------------------------------------------


class MockProviderType:
    """Mock SSOProviderType enum value."""

    def __init__(self, value: str = "oidc"):
        self.value = value

    def __eq__(self, other: Any) -> bool:
        if hasattr(other, "value"):
            return self.value == other.value
        return self.value == other

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)


@dataclass
class MockSSOConfig:
    """Mock SSO config."""

    entity_id: str = "https://aragora.example.com"
    callback_url: str = "https://aragora.example.com/api/v1/auth/sso/callback"
    auto_provision: bool = True
    allowed_domains: list[str] = field(default_factory=lambda: ["example.com"])
    session_duration_seconds: int = 28800


class MockSSOProvider:
    """Mock SSO provider."""

    def __init__(
        self,
        provider_type_value: str = "oidc",
        config: MockSSOConfig | None = None,
    ):
        self.provider_type = MockProviderType(provider_type_value)
        self.config = config or MockSSOConfig()
        self.generate_state = MagicMock(return_value="mock-state-123")
        self.get_authorization_url = AsyncMock(
            return_value="https://idp.example.com/authorize?state=mock-state-123"
        )
        self.authenticate = AsyncMock()
        self.logout = AsyncMock(return_value=None)
        self.get_metadata = AsyncMock(return_value="<xml>metadata</xml>")


@dataclass
class MockSSOUser:
    """Mock authenticated SSO user."""

    id: str = "sso-user-123"
    email: str = "user@example.com"
    name: str = "Test User"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
        }


@dataclass
class MockArgoraUser:
    """Mock Aragora user from user store."""

    id: str = "aragora-user-456"
    email: str = "user@example.com"
    name: str = "Test User"


class MockUserStore:
    """Mock user store."""

    def __init__(self, existing_user: MockArgoraUser | None = None):
        self._existing = existing_user
        self._created: list[dict] = []

    def get_user_by_email(self, email: str) -> MockArgoraUser | None:
        if self._existing and self._existing.email == email:
            return self._existing
        return None

    def get_user_by_id(self, user_id: str) -> MockArgoraUser | None:
        if self._existing and self._existing.id == user_id:
            return self._existing
        return None

    def update_user(self, user_id: str, **kwargs: Any) -> None:
        pass

    def create_user(self, **kwargs: Any) -> MockArgoraUser:
        user = MockArgoraUser(
            id="new-user-789",
            email=kwargs.get("email", ""),
            name=kwargs.get("name", ""),
        )
        self._created.append(kwargs)
        return user


class MockAuthConfig:
    """Mock auth_config."""

    def generate_token(self, loop_id: str, expires_in: int) -> str:
        return f"mock-jwt-token-for-{loop_id}"

    def revoke_token(self, token: str, reason: str) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provider():
    """Create a mock SSO provider."""
    return MockSSOProvider()


@pytest.fixture
def handler(mock_provider):
    """Create an SSOHandler with mocked provider."""
    h = SSOHandler(server_context={})
    h._provider = mock_provider
    h._initialized = True
    return h


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset rate limiters between tests."""
    from aragora.server.handlers.utils.rate_limit import clear_all_limiters

    clear_all_limiters()
    yield
    clear_all_limiters()


@pytest.fixture(autouse=True)
def _clean_env():
    """Clean SSO-related environment variables between tests."""
    keys = [
        "ARAGORA_ENV",
        "ARAGORA_SSO_ALLOWED_REDIRECT_HOSTS",
    ]
    saved = {k: os.environ.get(k) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify that can_handle correctly accepts or rejects paths."""

    def test_auth_sso_login(self, handler):
        assert handler.can_handle("/auth/sso/login", "GET")

    def test_auth_sso_callback(self, handler):
        assert handler.can_handle("/auth/sso/callback", "GET")

    def test_auth_sso_logout(self, handler):
        assert handler.can_handle("/auth/sso/logout", "GET")

    def test_auth_sso_metadata(self, handler):
        assert handler.can_handle("/auth/sso/metadata", "GET")

    def test_auth_sso_status(self, handler):
        assert handler.can_handle("/auth/sso/status", "GET")

    def test_sdk_v2_login(self, handler):
        assert handler.can_handle("/api/v2/sso/login", "GET")

    def test_sdk_v2_callback(self, handler):
        assert handler.can_handle("/api/v2/sso/callback", "POST")

    def test_sdk_v2_logout(self, handler):
        assert handler.can_handle("/api/v2/sso/logout", "GET")

    def test_sdk_v2_status(self, handler):
        assert handler.can_handle("/api/v2/sso/status", "GET")

    def test_sdk_v2_metadata(self, handler):
        assert handler.can_handle("/api/v2/sso/metadata", "GET")

    def test_sdk_unversioned_login(self, handler):
        assert handler.can_handle("/api/sso/login", "GET")

    def test_sdk_unversioned_callback(self, handler):
        assert handler.can_handle("/api/sso/callback", "POST")

    def test_rejects_unrelated_path(self, handler):
        assert not handler.can_handle("/api/v1/debates", "GET")

    def test_rejects_root(self, handler):
        assert not handler.can_handle("/", "GET")

    def test_rejects_partial_sso_path(self, handler):
        assert not handler.can_handle("/auth/sso", "GET")

    def test_rejects_non_sso_auth(self, handler):
        assert not handler.can_handle("/auth/login", "GET")


# ============================================================================
# Path normalization
# ============================================================================


class TestNormalizeSSOPath:
    """Test _normalize_sso_path static method."""

    def test_passthrough_auth_path(self):
        assert SSOHandler._normalize_sso_path("/auth/sso/login") == "/auth/sso/login"

    def test_v2_to_auth(self):
        assert SSOHandler._normalize_sso_path("/api/v2/sso/login") == "/auth/sso/login"

    def test_v3_to_auth(self):
        assert SSOHandler._normalize_sso_path("/api/v3/sso/callback") == "/auth/sso/callback"

    def test_unversioned_api(self):
        assert SSOHandler._normalize_sso_path("/api/sso/logout") == "/auth/sso/logout"

    def test_non_sso_path_unchanged(self):
        result = SSOHandler._normalize_sso_path("/api/v2/debates")
        assert result == "/api/v2/debates"


# ============================================================================
# Initialization
# ============================================================================


class TestHandlerInit:
    """Test handler initialization."""

    def test_init_with_empty_context(self):
        h = SSOHandler(server_context={})
        assert h.ctx == {}

    def test_init_with_none_context(self):
        h = SSOHandler(server_context=None)
        assert h.ctx == {}

    def test_init_sets_uninitialized(self):
        h = SSOHandler(server_context={})
        assert h._initialized is False
        assert h._provider is None

    def test_routes_defined(self, handler):
        routes = handler.routes()
        assert len(routes) > 0
        # Base routes
        methods_paths = [(m, p) for m, p, _ in routes]
        assert ("GET", "/auth/sso/login") in methods_paths
        assert ("POST", "/auth/sso/login") in methods_paths
        assert ("GET", "/auth/sso/callback") in methods_paths
        assert ("POST", "/auth/sso/callback") in methods_paths
        assert ("GET", "/auth/sso/logout") in methods_paths
        assert ("POST", "/auth/sso/logout") in methods_paths
        assert ("GET", "/auth/sso/metadata") in methods_paths
        assert ("GET", "/auth/sso/status") in methods_paths

    def test_routes_include_sdk_v2_aliases(self, handler):
        routes = handler.routes()
        methods_paths = [(m, p) for m, p, _ in routes]
        assert ("GET", "/api/v2/sso/login") in methods_paths
        assert ("POST", "/api/v2/sso/login") in methods_paths

    def test_routes_include_unversioned_aliases(self, handler):
        routes = handler.routes()
        methods_paths = [(m, p) for m, p, _ in routes]
        assert ("GET", "/api/sso/login") in methods_paths

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "sso"

    def test_static_routes_list(self):
        assert "/auth/sso/login" in SSOHandler.ROUTES
        assert "/api/v2/sso/login" in SSOHandler.ROUTES
        assert "/api/sso/login" in SSOHandler.ROUTES


# ============================================================================
# _get_param helper
# ============================================================================


class TestGetParam:
    """Test _get_param parameter extraction."""

    def test_string_value(self, handler):
        assert handler._get_param({"key": "value"}, "key") == "value"

    def test_list_value(self, handler):
        assert handler._get_param({"key": ["val1", "val2"]}, "key") == "val1"

    def test_empty_list(self, handler):
        assert handler._get_param({"key": []}, "key") is None

    def test_missing_key(self, handler):
        assert handler._get_param({}, "key") is None

    def test_none_value(self, handler):
        assert handler._get_param({"key": None}, "key") is None

    def test_integer_value_converted(self, handler):
        assert handler._get_param({"key": 42}, "key") == "42"


# ============================================================================
# _validate_redirect_url
# ============================================================================


class TestValidateRedirectUrl:
    """Test redirect URL validation."""

    def test_empty_url_is_safe(self, handler):
        assert handler._validate_redirect_url("") is True

    def test_https_url_valid(self, handler):
        assert handler._validate_redirect_url("https://app.example.com/dashboard") is True

    def test_http_url_valid_non_production(self, handler):
        os.environ.pop("ARAGORA_ENV", None)
        assert handler._validate_redirect_url("http://localhost:3000/callback") is True

    def test_ftp_scheme_rejected(self, handler):
        assert handler._validate_redirect_url("ftp://evil.com/data") is False

    def test_javascript_scheme_rejected(self, handler):
        assert handler._validate_redirect_url("javascript:alert(1)") is False

    def test_credentials_in_url_rejected(self, handler):
        assert handler._validate_redirect_url("https://user:pass@evil.com") is False

    def test_allowed_hosts_enforced(self, handler):
        os.environ["ARAGORA_SSO_ALLOWED_REDIRECT_HOSTS"] = "app.example.com,admin.example.com"
        assert handler._validate_redirect_url("https://app.example.com/cb") is True
        assert handler._validate_redirect_url("https://evil.com/cb") is False

    def test_allowed_hosts_case_insensitive(self, handler):
        os.environ["ARAGORA_SSO_ALLOWED_REDIRECT_HOSTS"] = "App.Example.COM"
        assert handler._validate_redirect_url("https://app.example.com/cb") is True

    def test_allowed_hosts_with_port(self, handler):
        os.environ["ARAGORA_SSO_ALLOWED_REDIRECT_HOSTS"] = "localhost"
        assert handler._validate_redirect_url("https://localhost:8080/cb") is True

    def test_production_requires_https(self, handler):
        os.environ["ARAGORA_ENV"] = "production"
        assert handler._validate_redirect_url("http://app.example.com/cb") is False

    def test_production_https_ok(self, handler):
        os.environ["ARAGORA_ENV"] = "production"
        assert handler._validate_redirect_url("https://app.example.com/cb") is True

    def test_no_allowed_hosts_allows_all(self, handler):
        os.environ.pop("ARAGORA_SSO_ALLOWED_REDIRECT_HOSTS", None)
        assert handler._validate_redirect_url("https://any-host.com/path") is True


# ============================================================================
# handle_login
# ============================================================================


class TestHandleLogin:
    """Test SSO login initiation."""

    @pytest.mark.asyncio
    async def test_login_redirect(self, handler, mock_provider):
        h = _make_handler()
        result = await handler.handle_login(h, {})
        body = _body(result)
        status = _status(result)
        assert status == 302 or "auth_url" in body

    @pytest.mark.asyncio
    async def test_login_json_response(self, handler, mock_provider):
        h = _make_handler(headers={"Accept": "application/json"})
        result = await handler.handle_login(h, {})
        body = _body(result)
        assert "auth_url" in body
        assert body["auth_url"] == "https://idp.example.com/authorize?state=mock-state-123"
        assert body["state"] == "mock-state-123"
        assert body["provider"] == "oidc"

    @pytest.mark.asyncio
    async def test_login_with_redirect_uri(self, handler, mock_provider):
        h = _make_handler(headers={"Accept": "application/json"})
        result = await handler.handle_login(h, {"redirect_uri": "https://app.example.com"})
        body = _body(result)
        assert "auth_url" in body
        # Verify relay_state was passed
        mock_provider.get_authorization_url.assert_called_once()
        call_kwargs = mock_provider.get_authorization_url.call_args
        assert call_kwargs.kwargs.get("relay_state") == "https://app.example.com"

    @pytest.mark.asyncio
    async def test_login_with_redirect_uri_as_list(self, handler, mock_provider):
        h = _make_handler(headers={"Accept": "application/json"})
        result = await handler.handle_login(h, {"redirect_uri": ["https://app.example.com"]})
        body = _body(result)
        assert "auth_url" in body

    @pytest.mark.asyncio
    async def test_login_with_custom_state(self, handler, mock_provider):
        h = _make_handler(headers={"Accept": "application/json"})
        result = await handler.handle_login(h, {"state": "custom-state-xyz"})
        body = _body(result)
        assert body["state"] == "custom-state-xyz"
        mock_provider.generate_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_login_generates_state_when_empty(self, handler, mock_provider):
        h = _make_handler(headers={"Accept": "application/json"})
        result = await handler.handle_login(h, {})
        body = _body(result)
        mock_provider.generate_state.assert_called_once()
        assert body["state"] == "mock-state-123"

    @pytest.mark.asyncio
    async def test_login_no_provider_returns_501(self, handler):
        handler._provider = None
        h = _make_handler()
        result = await handler.handle_login(h, {})
        assert _status(result) == 501
        assert "not configured" in _error_msg(result).lower()

    @pytest.mark.asyncio
    async def test_login_configuration_error(self, handler, mock_provider):
        from aragora.exceptions import ConfigurationError

        mock_provider.get_authorization_url.side_effect = ConfigurationError(
            "SSO", "missing client_id"
        )
        h = _make_handler(headers={"Accept": "application/json"})
        result = await handler.handle_login(h, {})
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_login_value_error(self, handler, mock_provider):
        mock_provider.get_authorization_url.side_effect = ValueError("bad state")
        h = _make_handler(headers={"Accept": "application/json"})
        result = await handler.handle_login(h, {})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_login_key_error(self, handler, mock_provider):
        mock_provider.get_authorization_url.side_effect = KeyError("missing_key")
        h = _make_handler(headers={"Accept": "application/json"})
        result = await handler.handle_login(h, {})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_login_type_error(self, handler, mock_provider):
        mock_provider.get_authorization_url.side_effect = TypeError("wrong type")
        h = _make_handler(headers={"Accept": "application/json"})
        result = await handler.handle_login(h, {})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_login_connection_error(self, handler, mock_provider):
        mock_provider.get_authorization_url.side_effect = ConnectionError("timeout")
        h = _make_handler(headers={"Accept": "application/json"})
        result = await handler.handle_login(h, {})
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_login_timeout_error(self, handler, mock_provider):
        mock_provider.get_authorization_url.side_effect = TimeoutError("timed out")
        h = _make_handler(headers={"Accept": "application/json"})
        result = await handler.handle_login(h, {})
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_login_runtime_error(self, handler, mock_provider):
        mock_provider.get_authorization_url.side_effect = RuntimeError("oops")
        h = _make_handler(headers={"Accept": "application/json"})
        result = await handler.handle_login(h, {})
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_login_redirect_has_location_header(self, handler, mock_provider):
        h = _make_handler()  # No Accept header -> redirect
        result = await handler.handle_login(h, {})
        if isinstance(result, HandlerResult):
            assert result.status_code == 302
            assert "Location" in result.headers
            assert result.headers["Location"].startswith("https://idp.example.com")
        elif isinstance(result, dict):
            assert result.get("status") == 302
            headers = result.get("headers", {})
            assert "Location" in headers

    @pytest.mark.asyncio
    async def test_login_redirect_has_cache_control(self, handler, mock_provider):
        h = _make_handler()
        result = await handler.handle_login(h, {})
        if isinstance(result, HandlerResult):
            assert result.headers.get("Cache-Control") == "no-cache, no-store"
        elif isinstance(result, dict):
            headers = result.get("headers", {})
            assert headers.get("Cache-Control") == "no-cache, no-store"

    @pytest.mark.asyncio
    async def test_login_state_as_list(self, handler, mock_provider):
        h = _make_handler(headers={"Accept": "application/json"})
        result = await handler.handle_login(h, {"state": ["list-state"]})
        body = _body(result)
        assert body.get("state") == "list-state"


# ============================================================================
# handle_callback
# ============================================================================


class TestHandleCallback:
    """Test SSO callback handling."""

    @pytest.mark.asyncio
    async def test_callback_success_json(self, handler, mock_provider):
        sso_user = MockSSOUser()
        mock_provider.authenticate.return_value = sso_user
        user_store = MockUserStore()
        auth_cfg = MockAuthConfig()

        h = _make_handler()
        with (
            patch("aragora.server.handlers.sso.auth_config", auth_cfg),
            patch(
                "aragora.server.handlers.sso.SSOUser",
                MockSSOUser,
            ),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=user_store,
            ),
        ):
            result = await handler.handle_callback(
                h, {"code": "auth-code-123", "state": "mock-state"}
            )
        body = _body(result)
        assert _status(result) == 200
        assert body.get("success") is True
        assert "token" in body
        assert "user" in body

    @pytest.mark.asyncio
    async def test_callback_no_provider_returns_501(self, handler):
        handler._provider = None
        h = _make_handler()
        result = await handler.handle_callback(h, {})
        assert _status(result) == 501
        assert "not configured" in _error_msg(result).lower()

    @pytest.mark.asyncio
    async def test_callback_idp_error(self, handler, mock_provider):
        h = _make_handler()
        result = await handler.handle_callback(
            h, {"error": "access_denied", "error_description": "User denied consent"}
        )
        assert _status(result) == 401
        assert "User denied consent" in _error_msg(result)

    @pytest.mark.asyncio
    async def test_callback_idp_error_no_description(self, handler, mock_provider):
        h = _make_handler()
        result = await handler.handle_callback(h, {"error": "server_error"})
        assert _status(result) == 401
        assert "server_error" in _error_msg(result)

    @pytest.mark.asyncio
    async def test_callback_https_required_in_production(self, handler, mock_provider):
        os.environ["ARAGORA_ENV"] = "production"
        mock_provider.config.callback_url = "http://insecure.example.com/callback"
        h = _make_handler()
        result = await handler.handle_callback(h, {"code": "abc"})
        assert _status(result) == 400
        msg = _error_msg(result)
        assert "HTTPS" in msg or "https" in msg.lower()

    @pytest.mark.asyncio
    async def test_callback_https_ok_in_production(self, handler, mock_provider):
        os.environ["ARAGORA_ENV"] = "production"
        mock_provider.config.callback_url = "https://secure.example.com/callback"
        sso_user = MockSSOUser()
        mock_provider.authenticate.return_value = sso_user
        user_store = MockUserStore()
        auth_cfg = MockAuthConfig()

        h = _make_handler()
        with (
            patch("aragora.server.handlers.sso.auth_config", auth_cfg),
            patch(
                "aragora.server.handlers.sso.SSOUser",
                MockSSOUser,
            ),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=user_store,
            ),
        ):
            result = await handler.handle_callback(h, {"code": "auth-code", "state": "st"})
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_callback_creates_new_user(self, handler, mock_provider):
        sso_user = MockSSOUser(email="newuser@example.com", name="New User")
        mock_provider.authenticate.return_value = sso_user
        user_store = MockUserStore()  # No existing user
        auth_cfg = MockAuthConfig()

        h = _make_handler()
        with (
            patch("aragora.server.handlers.sso.auth_config", auth_cfg),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=user_store,
            ),
        ):
            result = await handler.handle_callback(h, {"code": "code", "state": "st"})
        assert _status(result) == 200
        assert len(user_store._created) == 1
        assert user_store._created[0]["email"] == "newuser@example.com"

    @pytest.mark.asyncio
    async def test_callback_updates_existing_user(self, handler, mock_provider):
        existing = MockArgoraUser(id="existing-123", email="user@example.com")
        sso_user = MockSSOUser(email="user@example.com", name="Updated Name")
        mock_provider.authenticate.return_value = sso_user
        user_store = MockUserStore(existing_user=existing)
        auth_cfg = MockAuthConfig()

        h = _make_handler()
        with (
            patch("aragora.server.handlers.sso.auth_config", auth_cfg),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=user_store,
            ),
        ):
            result = await handler.handle_callback(h, {"code": "code", "state": "st"})
        assert _status(result) == 200
        body = _body(result)
        # Token should use the aragora user id
        assert "existing-123" in body.get("token", "")

    @pytest.mark.asyncio
    async def test_callback_with_redirect(self, handler, mock_provider):
        sso_user = MockSSOUser()
        mock_provider.authenticate.return_value = sso_user
        user_store = MockUserStore()
        auth_cfg = MockAuthConfig()

        h = _make_handler()
        with (
            patch("aragora.server.handlers.sso.auth_config", auth_cfg),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=user_store,
            ),
        ):
            result = await handler.handle_callback(
                h,
                {
                    "code": "code",
                    "state": "https://app.example.com/dashboard",
                    "RelayState": "https://app.example.com/dashboard",
                },
            )
        status = _status(result)
        assert status == 302
        if isinstance(result, HandlerResult):
            loc = result.headers.get("Location", "")
        else:
            loc = result.get("headers", {}).get("Location", "")
        assert "token=" in loc
        assert "app.example.com" in loc

    @pytest.mark.asyncio
    async def test_callback_redirect_with_query_params(self, handler, mock_provider):
        sso_user = MockSSOUser()
        mock_provider.authenticate.return_value = sso_user
        user_store = MockUserStore()
        auth_cfg = MockAuthConfig()

        h = _make_handler()
        with (
            patch("aragora.server.handlers.sso.auth_config", auth_cfg),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=user_store,
            ),
        ):
            result = await handler.handle_callback(
                h,
                {
                    "code": "code",
                    "RelayState": "https://app.example.com/page?foo=bar",
                },
            )
        status = _status(result)
        assert status == 302
        if isinstance(result, HandlerResult):
            loc = result.headers.get("Location", "")
        else:
            loc = result.get("headers", {}).get("Location", "")
        # Should use & separator since URL already has ?
        assert "&token=" in loc

    @pytest.mark.asyncio
    async def test_callback_unsafe_redirect_blocked(self, handler, mock_provider):
        os.environ["ARAGORA_SSO_ALLOWED_REDIRECT_HOSTS"] = "safe.example.com"
        sso_user = MockSSOUser()
        mock_provider.authenticate.return_value = sso_user
        user_store = MockUserStore()
        auth_cfg = MockAuthConfig()

        h = _make_handler()
        with (
            patch("aragora.server.handlers.sso.auth_config", auth_cfg),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=user_store,
            ),
        ):
            result = await handler.handle_callback(
                h,
                {
                    "code": "code",
                    "RelayState": "https://evil.com/steal",
                },
            )
        assert _status(result) == 400
        msg = _error_msg(result)
        assert "redirect" in msg.lower() or "SSO_INVALID_REDIRECT" in str(_body(result))

    @pytest.mark.asyncio
    async def test_callback_no_auth_config(self, handler, mock_provider):
        sso_user = MockSSOUser()
        mock_provider.authenticate.return_value = sso_user
        user_store = MockUserStore()

        h = _make_handler()
        with (
            patch("aragora.server.handlers.sso.auth_config", None),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=user_store,
            ),
        ):
            result = await handler.handle_callback(h, {"code": "code", "state": "st"})
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_callback_domain_not_allowed(self, handler, mock_provider):
        from aragora.auth.sso import SSOAuthenticationError

        mock_provider.authenticate.side_effect = SSOAuthenticationError(
            "Domain not allowed",
            details={"code": "DOMAIN_NOT_ALLOWED"},
        )
        h = _make_handler()
        result = await handler.handle_callback(h, {"code": "code"})
        assert _status(result) == 403
        assert "domain" in _error_msg(result).lower()

    @pytest.mark.asyncio
    async def test_callback_invalid_state(self, handler, mock_provider):
        from aragora.auth.sso import SSOAuthenticationError

        mock_provider.authenticate.side_effect = SSOAuthenticationError(
            "Invalid state",
            details={"code": "INVALID_STATE"},
        )
        h = _make_handler()
        result = await handler.handle_callback(h, {"code": "code"})
        assert _status(result) == 401
        assert "expired" in _error_msg(result).lower()

    @pytest.mark.asyncio
    async def test_callback_generic_auth_error(self, handler, mock_provider):
        from aragora.auth.sso import SSOAuthenticationError

        mock_provider.authenticate.side_effect = SSOAuthenticationError("Bad token")
        h = _make_handler()
        result = await handler.handle_callback(h, {"code": "code"})
        assert _status(result) == 401
        assert "failed" in _error_msg(result).lower()

    @pytest.mark.asyncio
    async def test_callback_value_error(self, handler, mock_provider):
        mock_provider.authenticate.side_effect = ValueError("bad data")
        h = _make_handler()
        result = await handler.handle_callback(h, {"code": "code"})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_callback_connection_error(self, handler, mock_provider):
        mock_provider.authenticate.side_effect = ConnectionError("network")
        h = _make_handler()
        result = await handler.handle_callback(h, {"code": "code"})
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_callback_runtime_error(self, handler, mock_provider):
        mock_provider.authenticate.side_effect = RuntimeError("runtime issue")
        h = _make_handler()
        result = await handler.handle_callback(h, {"code": "code"})
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_callback_import_error(self, handler, mock_provider):
        mock_provider.authenticate.side_effect = ImportError("missing module")
        h = _make_handler()
        result = await handler.handle_callback(h, {"code": "code"})
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_callback_saml_response(self, handler, mock_provider):
        sso_user = MockSSOUser()
        mock_provider.authenticate.return_value = sso_user
        user_store = MockUserStore()
        auth_cfg = MockAuthConfig()

        h = _make_handler()
        with (
            patch("aragora.server.handlers.sso.auth_config", auth_cfg),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=user_store,
            ),
        ):
            result = await handler.handle_callback(
                h,
                {"SAMLResponse": "base64-saml-data", "RelayState": "orig-state"},
            )
        assert _status(result) == 200
        # Verify authenticate was called with saml_response
        call_kwargs = mock_provider.authenticate.call_args.kwargs
        assert call_kwargs.get("saml_response") == "base64-saml-data"

    @pytest.mark.asyncio
    async def test_callback_null_user_store(self, handler, mock_provider):
        sso_user = MockSSOUser()
        mock_provider.authenticate.return_value = sso_user
        auth_cfg = MockAuthConfig()

        h = _make_handler()
        with (
            patch("aragora.server.handlers.sso.auth_config", auth_cfg),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=None,
            ),
        ):
            result = await handler.handle_callback(h, {"code": "code", "state": "st"})
        assert _status(result) == 200
        body = _body(result)
        # Token uses SSO user id since no aragora_user
        assert "sso-user-123" in body.get("token", "")

    @pytest.mark.asyncio
    async def test_callback_non_production_allows_http(self, handler, mock_provider):
        os.environ.pop("ARAGORA_ENV", None)
        mock_provider.config.callback_url = "http://localhost:8080/callback"
        sso_user = MockSSOUser()
        mock_provider.authenticate.return_value = sso_user
        user_store = MockUserStore()
        auth_cfg = MockAuthConfig()

        h = _make_handler()
        with (
            patch("aragora.server.handlers.sso.auth_config", auth_cfg),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=user_store,
            ),
        ):
            result = await handler.handle_callback(h, {"code": "code", "state": "st"})
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_callback_expires_in_returned(self, handler, mock_provider):
        sso_user = MockSSOUser()
        mock_provider.authenticate.return_value = sso_user
        mock_provider.config.session_duration_seconds = 7200
        user_store = MockUserStore()
        auth_cfg = MockAuthConfig()

        h = _make_handler()
        with (
            patch("aragora.server.handlers.sso.auth_config", auth_cfg),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
            patch(
                "aragora.storage.user_store.singleton.get_user_store",
                return_value=user_store,
            ),
        ):
            result = await handler.handle_callback(h, {"code": "code", "state": "st"})
        body = _body(result)
        assert body.get("expires_in") == 7200


# ============================================================================
# handle_logout
# ============================================================================


class TestHandleLogout:
    """Test SSO logout handling."""

    @pytest.mark.asyncio
    async def test_logout_no_provider(self, handler):
        handler._provider = None
        h = _make_handler()
        result = await handler.handle_logout(h, {})
        body = _body(result)
        assert body.get("success") is True
        assert "logged out" in body.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_logout_success_no_redirect(self, handler, mock_provider):
        mock_provider.logout.return_value = None  # No IdP logout URL
        auth_cfg = MockAuthConfig()

        h = _make_handler(headers={"Authorization": "Bearer test-token-123"})
        with (
            patch("aragora.server.handlers.sso.auth_config", auth_cfg),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
        ):
            result = await handler.handle_logout(h, {})
        body = _body(result)
        assert body.get("success") is True
        assert "successfully" in body.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_logout_with_idp_redirect(self, handler, mock_provider):
        mock_provider.logout.return_value = "https://idp.example.com/logout"
        auth_cfg = MockAuthConfig()

        h = _make_handler(headers={"Authorization": "Bearer test-token-123"})
        with (
            patch("aragora.server.handlers.sso.auth_config", auth_cfg),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
        ):
            result = await handler.handle_logout(h, {})
        status = _status(result)
        assert status == 302
        if isinstance(result, HandlerResult):
            assert result.headers["Location"] == "https://idp.example.com/logout"
        else:
            assert result.get("headers", {}).get("Location") == "https://idp.example.com/logout"

    @pytest.mark.asyncio
    async def test_logout_no_bearer_token(self, handler, mock_provider):
        mock_provider.logout.return_value = None
        auth_cfg = MockAuthConfig()

        h = _make_handler()  # No Authorization header
        with (
            patch("aragora.server.handlers.sso.auth_config", auth_cfg),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
        ):
            result = await handler.handle_logout(h, {})
        body = _body(result)
        assert body.get("success") is True

    @pytest.mark.asyncio
    async def test_logout_no_auth_config(self, handler, mock_provider):
        h = _make_handler()
        with (
            patch("aragora.server.handlers.sso.auth_config", None),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
        ):
            result = await handler.handle_logout(h, {})
        body = _body(result)
        # Configuration error is caught and returns success with note
        assert body.get("success") is True
        assert "config" in body.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_logout_ssouser_not_imported(self, handler, mock_provider):
        auth_cfg = MockAuthConfig()
        h = _make_handler()
        with (
            patch("aragora.server.handlers.sso.auth_config", auth_cfg),
            patch("aragora.server.handlers.sso.SSOUser", None),
        ):
            result = await handler.handle_logout(h, {})
        body = _body(result)
        # ConfigurationError caught -> success with config errors
        assert body.get("success") is True

    @pytest.mark.asyncio
    async def test_logout_value_error(self, handler, mock_provider):
        auth_cfg = MockAuthConfig()
        mock_provider.logout.side_effect = ValueError("bad value")

        h = _make_handler()
        with (
            patch("aragora.server.handlers.sso.auth_config", auth_cfg),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
        ):
            result = await handler.handle_logout(h, {})
        body = _body(result)
        assert body.get("success") is True
        assert "errors" in body.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_logout_connection_error(self, handler, mock_provider):
        auth_cfg = MockAuthConfig()
        mock_provider.logout.side_effect = ConnectionError("network error")

        h = _make_handler()
        with (
            patch("aragora.server.handlers.sso.auth_config", auth_cfg),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
        ):
            result = await handler.handle_logout(h, {})
        body = _body(result)
        assert body.get("success") is True
        assert "errors" in body.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_logout_runtime_error(self, handler, mock_provider):
        auth_cfg = MockAuthConfig()
        mock_provider.logout.side_effect = RuntimeError("runtime issue")

        h = _make_handler()
        with (
            patch("aragora.server.handlers.sso.auth_config", auth_cfg),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
        ):
            result = await handler.handle_logout(h, {})
        body = _body(result)
        assert body.get("success") is True

    @pytest.mark.asyncio
    async def test_logout_redirect_has_cache_control(self, handler, mock_provider):
        mock_provider.logout.return_value = "https://idp.example.com/logout"
        auth_cfg = MockAuthConfig()

        h = _make_handler()
        with (
            patch("aragora.server.handlers.sso.auth_config", auth_cfg),
            patch("aragora.server.handlers.sso.SSOUser", MockSSOUser),
        ):
            result = await handler.handle_logout(h, {})
        if isinstance(result, HandlerResult):
            assert result.headers.get("Cache-Control") == "no-cache, no-store"
        elif isinstance(result, dict):
            assert result.get("headers", {}).get("Cache-Control") == "no-cache, no-store"


# ============================================================================
# handle_metadata
# ============================================================================


class TestHandleMetadata:
    """Test SAML metadata endpoint."""

    @pytest.mark.asyncio
    async def test_metadata_no_provider(self, handler):
        handler._provider = None
        h = _make_handler()
        result = await handler.handle_metadata(h, {})
        assert _status(result) == 501

    @pytest.mark.asyncio
    async def test_metadata_non_saml_provider(self, handler, mock_provider):
        mock_provider.provider_type = MockProviderType("oidc")
        h = _make_handler()
        with patch(
            "aragora.server.handlers.sso.SSOProviderType",
            type("MockEnum", (), {"SAML": MockProviderType("saml")}),
        ):
            result = await handler.handle_metadata(h, {})
        assert _status(result) == 400
        assert "saml" in _error_msg(result).lower()

    @pytest.mark.asyncio
    async def test_metadata_saml_success(self, handler, mock_provider):
        mock_provider.provider_type = MockProviderType("saml")
        mock_provider.get_metadata = AsyncMock(return_value="<EntityDescriptor/>")

        h = _make_handler()
        saml_type = MockProviderType("saml")
        with patch(
            "aragora.server.handlers.sso.SSOProviderType",
            type("MockEnum", (), {"SAML": saml_type}),
        ):
            result = await handler.handle_metadata(h, {})

        if isinstance(result, HandlerResult):
            assert result.status_code == 200
            assert result.content_type == "application/xml"
            assert b"EntityDescriptor" in result.body
        else:
            assert result.get("status") == 200

    @pytest.mark.asyncio
    async def test_metadata_provider_type_unavailable(self, handler, mock_provider):
        h = _make_handler()
        with patch("aragora.server.handlers.sso.SSOProviderType", None):
            result = await handler.handle_metadata(h, {})
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_metadata_value_error(self, handler, mock_provider):
        mock_provider.provider_type = MockProviderType("saml")
        mock_provider.get_metadata = AsyncMock(side_effect=ValueError("bad config"))
        saml_type = MockProviderType("saml")

        h = _make_handler()
        with patch(
            "aragora.server.handlers.sso.SSOProviderType",
            type("MockEnum", (), {"SAML": saml_type}),
        ):
            result = await handler.handle_metadata(h, {})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_metadata_runtime_error(self, handler, mock_provider):
        mock_provider.provider_type = MockProviderType("saml")
        mock_provider.get_metadata = AsyncMock(side_effect=RuntimeError("error"))
        saml_type = MockProviderType("saml")

        h = _make_handler()
        with patch(
            "aragora.server.handlers.sso.SSOProviderType",
            type("MockEnum", (), {"SAML": saml_type}),
        ):
            result = await handler.handle_metadata(h, {})
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_metadata_no_get_metadata_attr(self, handler, mock_provider):
        mock_provider.provider_type = MockProviderType("saml")
        # Remove get_metadata attribute
        if hasattr(mock_provider, "get_metadata"):
            delattr(mock_provider, "get_metadata")
        saml_type = MockProviderType("saml")

        h = _make_handler()
        with patch(
            "aragora.server.handlers.sso.SSOProviderType",
            type("MockEnum", (), {"SAML": saml_type}),
        ):
            result = await handler.handle_metadata(h, {})
        assert _status(result) == 400
        body = _body(result)
        assert "not available" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_metadata_cache_control_header(self, handler, mock_provider):
        mock_provider.provider_type = MockProviderType("saml")
        mock_provider.get_metadata = AsyncMock(return_value="<xml/>")
        saml_type = MockProviderType("saml")

        h = _make_handler()
        with patch(
            "aragora.server.handlers.sso.SSOProviderType",
            type("MockEnum", (), {"SAML": saml_type}),
        ):
            result = await handler.handle_metadata(h, {})
        if isinstance(result, HandlerResult):
            assert result.headers.get("Cache-Control") == "max-age=3600"


# ============================================================================
# handle_status
# ============================================================================


class TestHandleStatus:
    """Test SSO status endpoint."""

    @pytest.mark.asyncio
    async def test_status_no_provider(self, handler):
        handler._provider = None
        h = _make_handler()
        result = await handler.handle_status(h, {})
        body = _body(result)
        assert _status(result) == 200
        assert body["enabled"] is False
        assert body["configured"] is False
        assert body["provider"] is None

    @pytest.mark.asyncio
    async def test_status_with_provider(self, handler, mock_provider):
        h = _make_handler()
        result = await handler.handle_status(h, {})
        body = _body(result)
        assert _status(result) == 200
        assert body["enabled"] is True
        assert body["configured"] is True
        assert body["provider"] == "oidc"
        assert body["entity_id"] == "https://aragora.example.com"
        assert body["callback_url"] == "https://aragora.example.com/api/v1/auth/sso/callback"
        assert body["auto_provision"] is True

    @pytest.mark.asyncio
    async def test_status_includes_allowed_domains(self, handler, mock_provider):
        mock_provider.config.allowed_domains = ["example.com", "test.com"]
        h = _make_handler()
        result = await handler.handle_status(h, {})
        body = _body(result)
        assert body["allowed_domains"] == ["example.com", "test.com"]

    @pytest.mark.asyncio
    async def test_status_no_allowed_domains_attr(self, handler, mock_provider):
        # Config without allowed_domains attribute
        config = MockSSOConfig()
        delattr(config, "allowed_domains")
        mock_provider.config = config
        h = _make_handler()
        result = await handler.handle_status(h, {})
        body = _body(result)
        assert body["allowed_domains"] == []


# ============================================================================
# Provider resolution
# ============================================================================


class TestProviderResolution:
    """Test lazy-loading and resolution of SSO provider."""

    def test_get_provider_lazy_init(self):
        h = SSOHandler(server_context={})
        assert h._initialized is False
        with patch("aragora.server.handlers.sso.get_sso_provider", return_value=None):
            provider = h._get_provider()
        assert h._initialized is True
        assert provider is None

    def test_get_provider_caches(self):
        h = SSOHandler(server_context={})
        mock_prov = MockSSOProvider()
        with patch("aragora.server.handlers.sso.get_sso_provider", return_value=mock_prov):
            p1 = h._get_provider()
        # Second call should not re-init
        p2 = h._get_provider()
        assert p1 is p2

    def test_get_provider_handles_import_error(self):
        h = SSOHandler(server_context={})
        with patch(
            "aragora.server.handlers.sso.get_sso_provider", side_effect=ImportError("no mod")
        ):
            provider = h._get_provider()
        assert provider is None
        assert h._initialized is True

    def test_get_provider_warns_missing_callback_url(self):
        h = SSOHandler(server_context={})
        mock_prov = MockSSOProvider()
        mock_prov.config.callback_url = ""
        with (
            patch("aragora.server.handlers.sso.get_sso_provider", return_value=mock_prov),
            patch("aragora.server.handlers.sso.logger") as mock_logger,
        ):
            h._get_provider()
        mock_logger.warning.assert_called()
        assert "ARAGORA_SSO_CALLBACK_URL" in str(mock_logger.warning.call_args)

    def test_resolve_provider_direct_call(self, handler, mock_provider):
        provider = handler._resolve_provider()
        assert provider is mock_provider

    def test_resolve_provider_when_none(self, handler):
        handler._provider = None
        handler._initialized = True
        provider = handler._resolve_provider()
        assert provider is None


# ============================================================================
# _to_legacy_result / _format_response / _flatten_error_body
# ============================================================================


class TestLegacyResultConversion:
    """Test internal response formatting methods."""

    def test_flatten_error_body_no_error(self, handler):
        body = {"message": "ok"}
        assert handler._flatten_error_body(body) == {"message": "ok"}

    def test_flatten_error_body_string_error(self, handler):
        body = {"error": "something failed"}
        assert handler._flatten_error_body(body) == {"error": "something failed"}

    def test_flatten_error_body_dict_error(self, handler):
        body = {
            "error": {
                "message": "not found",
                "code": "NOT_FOUND",
                "suggestion": "try again",
            }
        }
        result = handler._flatten_error_body(body)
        assert result["error"] == "not found"
        assert result["code"] == "NOT_FOUND"
        assert result["suggestion"] == "try again"

    def test_flatten_error_body_non_dict(self, handler):
        assert handler._flatten_error_body("string") == "string"
        assert handler._flatten_error_body(42) == 42

    def test_should_return_handler_result_with_send_response(self, handler):
        h = _make_handler()  # Has send_response method
        assert handler._should_return_handler_result(h) is True

    def test_should_return_handler_result_none(self, handler):
        assert handler._should_return_handler_result(None) is False

    def test_should_return_handler_result_no_send_response(self, handler):
        obj = object()
        assert handler._should_return_handler_result(obj) is False

    def test_format_response_handler_result(self, handler):
        h = _make_handler()
        hr = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=b'{"ok":true}',
        )
        result = handler._format_response(h, hr)
        # With send_response, should return HandlerResult as-is
        assert isinstance(result, HandlerResult)
        assert result.status_code == 200

    def test_format_response_legacy(self, handler):
        hr = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=b'{"ok":true}',
        )
        result = handler._format_response(None, hr)
        assert isinstance(result, dict)
        assert result["status"] == 200
        assert result["body"]["ok"] is True

    def test_to_legacy_result_from_dict(self, handler):
        d = {
            "status_code": 404,
            "content_type": "application/json",
            "body": b'{"error":"not found"}',
        }
        result = handler._to_legacy_result(d)
        assert result["status"] == 404

    def test_to_legacy_result_from_handler_result(self, handler):
        hr = HandlerResult(
            status_code=201,
            content_type="application/json",
            body=b'{"created":true}',
        )
        result = handler._to_legacy_result(hr)
        assert result["status"] == 201
        assert result["body"]["created"] is True

    def test_to_legacy_result_non_json_body(self, handler):
        hr = HandlerResult(
            status_code=200,
            content_type="text/plain",
            body=b"hello",
        )
        result = handler._to_legacy_result(hr)
        assert result["body"] == "hello"

    def test_to_legacy_result_invalid_type(self, handler):
        with pytest.raises(TypeError, match="Expected HandlerResult"):
            handler._to_legacy_result("invalid")

    def test_to_legacy_result_dict_bytes_json_body(self, handler):
        d = {
            "body": b'{"key":"val"}',
            "content_type": "application/json",
        }
        result = handler._to_legacy_result(d)
        assert result["body"]["key"] == "val"


# ============================================================================
# Module-level function: get_sso_provider
# ============================================================================


class TestGetSSOProvider:
    """Test module-level get_sso_provider function."""

    def test_raises_when_not_available(self):
        from aragora.server.handlers.sso import get_sso_provider

        with patch("aragora.server.handlers.sso._get_sso_provider", None):
            with pytest.raises(ImportError, match="SSO auth module not available"):
                get_sso_provider()

    def test_returns_provider(self):
        from aragora.server.handlers.sso import get_sso_provider

        mock_prov = MockSSOProvider()
        with patch(
            "aragora.server.handlers.sso._get_sso_provider",
            return_value=mock_prov,
        ):
            result = get_sso_provider()
        assert result is mock_prov


# ============================================================================
# isinstance safety
# ============================================================================


class TestIsInstanceSafety:
    """Test _is_isinstance_safe and _safe_isinstance functions."""

    def test_is_isinstance_safe_normal(self):
        from aragora.server.handlers.sso import _is_isinstance_safe

        assert _is_isinstance_safe() is True

    def test_safe_isinstance_basic(self):
        from aragora.server.handlers.sso import _safe_isinstance

        assert _safe_isinstance("hello", str) is True
        assert _safe_isinstance(42, str) is False
        assert _safe_isinstance([1, 2], list) is True
        assert _safe_isinstance({}, dict) is True

    def test_safe_isinstance_tuple(self):
        from aragora.server.handlers.sso import _safe_isinstance

        assert _safe_isinstance("hello", (str, int)) is True
        assert _safe_isinstance(42, (str, int)) is True
        assert _safe_isinstance(3.14, (str, int)) is False
