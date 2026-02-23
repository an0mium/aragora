"""Tests for Microsoft Teams OAuth handler (aragora/server/handlers/social/teams_oauth.py).

Covers all routes and behavior of the TeamsOAuthHandler class:
- can_handle() route matching for all static and dynamic routes
- GET  /api/integrations/teams/install    - Initiate OAuth flow
- GET  /api/integrations/teams/callback   - Handle OAuth callback from Microsoft
- POST /api/integrations/teams/refresh    - Refresh expired tokens
- POST /api/integrations/teams/disconnect - Disconnect a tenant
- GET  /api/integrations/teams/tenants    - List all tenants
- GET  /api/integrations/teams/tenants/{id}        - Get tenant details
- DELETE /api/integrations/teams/tenants/{id}       - Delete tenant (via disconnect)
- GET  /api/integrations/teams/tenants/{id}/status  - Get tenant token status
- Method not allowed responses
- Permission denied paths
- Error handling and edge cases
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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


def _html(result) -> str:
    """Extract HTML body string from a HandlerResult."""
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8")
    return raw


# ---------------------------------------------------------------------------
# Lazy imports so conftest auto-auth patches run first
# ---------------------------------------------------------------------------


@pytest.fixture
def handler_module():
    """Import the handler module lazily (after conftest patches)."""
    import aragora.server.handlers.social.teams_oauth as mod

    return mod


@pytest.fixture
def handler_cls(handler_module):
    return handler_module.TeamsOAuthHandler


@pytest.fixture
def handler(handler_cls):
    """Create a TeamsOAuthHandler with empty context."""
    return handler_cls(ctx={})


# ---------------------------------------------------------------------------
# Environment patches
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch, handler_module):
    """Set default Teams OAuth credentials for tests."""
    monkeypatch.setattr(handler_module, "TEAMS_CLIENT_ID", "test-client-id")
    monkeypatch.setattr(handler_module, "TEAMS_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setattr(
        handler_module, "TEAMS_REDIRECT_URI", "https://example.com/api/integrations/teams/callback"
    )
    monkeypatch.setattr(
        handler_module, "TEAMS_SCOPES", "https://graph.microsoft.com/.default offline_access"
    )


# ---------------------------------------------------------------------------
# OAuth State Store mock
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_state_store():
    """Create a mock OAuth state store."""
    store = MagicMock()
    store.generate.return_value = "test-state-token-abc123def456"

    state_data = MagicMock()
    state_data.metadata = {"org_id": "org-001", "provider": "teams"}
    store.validate_and_consume.return_value = state_data

    return store


@pytest.fixture(autouse=True)
def _patch_state_store(monkeypatch, mock_state_store, handler_module):
    """Patch _get_state_store to return our mock."""
    monkeypatch.setattr(handler_module, "_get_state_store", lambda: mock_state_store)


# ---------------------------------------------------------------------------
# Mock tenant helpers
# ---------------------------------------------------------------------------


_UNSET = object()


def _make_tenant(
    tenant_id: str = "tenant-abc",
    tenant_name: str = "Contoso Ltd",
    is_active: bool = True,
    expires_at: float | None | object = _UNSET,
    refresh_token: str | None = "rt-refresh-xyz",
    bot_id: str = "bot-123",
    scopes: list[str] | None = None,
    installed_at: float | None = None,
    installed_by: str | None = "user-456",
    aragora_org_id: str | None = "org-001",
) -> MagicMock:
    """Build a mock TeamsTenant."""
    t = MagicMock()
    t.tenant_id = tenant_id
    t.tenant_name = tenant_name
    t.is_active = is_active
    t.expires_at = time.time() + 7200 if expires_at is _UNSET else expires_at
    t.refresh_token = refresh_token
    t.bot_id = bot_id
    t.scopes = scopes or ["https://graph.microsoft.com/.default"]
    t.installed_at = installed_at or time.time()
    t.installed_by = installed_by
    t.aragora_org_id = aragora_org_id
    return t


@pytest.fixture
def mock_tenant():
    return _make_tenant()


@pytest.fixture
def mock_tenant_store(mock_tenant):
    """Create a mock tenant store."""
    store = MagicMock()
    store.save.return_value = True
    store.get.return_value = mock_tenant
    store.list_all.return_value = [mock_tenant]
    store.update_tokens.return_value = True
    store.deactivate.return_value = True
    return store


# ---------------------------------------------------------------------------
# httpx mock helpers
# ---------------------------------------------------------------------------


def _make_httpx_mock(response_json: dict, status_code: int = 200):
    """Build a mock httpx AsyncClient + response."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = response_json
    mock_response.raise_for_status = MagicMock()
    mock_response.headers = {}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.get = AsyncMock(return_value=mock_response)
    return mock_client, mock_response


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify that can_handle correctly accepts or rejects paths."""

    def test_install_path(self, handler):
        assert handler.can_handle("/api/integrations/teams/install")

    def test_callback_path(self, handler):
        assert handler.can_handle("/api/integrations/teams/callback")

    def test_refresh_path(self, handler):
        assert handler.can_handle("/api/integrations/teams/refresh")

    def test_disconnect_path(self, handler):
        assert handler.can_handle("/api/integrations/teams/disconnect")

    def test_tenants_path(self, handler):
        assert handler.can_handle("/api/integrations/teams/tenants")

    def test_tenant_by_id_path(self, handler):
        assert handler.can_handle("/api/integrations/teams/tenants/some-id")

    def test_tenant_status_path(self, handler):
        assert handler.can_handle("/api/integrations/teams/tenants/some-id/status")

    def test_rejects_unrelated_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_rejects_partial_path(self, handler):
        assert not handler.can_handle("/api/integrations/teams")

    def test_rejects_extra_suffix_on_install(self, handler):
        assert not handler.can_handle("/api/integrations/teams/install/extra")

    def test_rejects_wrong_prefix(self, handler):
        assert not handler.can_handle("/api/v1/integrations/teams/install")

    def test_rejects_typo(self, handler):
        assert not handler.can_handle("/api/integrations/tems/install")

    def test_rejects_empty_path(self, handler):
        assert not handler.can_handle("")

    def test_rejects_root_path(self, handler):
        assert not handler.can_handle("/")

    def test_tenant_dynamic_id(self, handler):
        assert handler.can_handle("/api/integrations/teams/tenants/any-id-here")

    def test_tenant_status_dynamic_id(self, handler):
        assert handler.can_handle("/api/integrations/teams/tenants/abc-xyz/status")


# ============================================================================
# Handler initialization and factory
# ============================================================================


class TestInit:
    """Test handler initialization."""

    def test_default_ctx(self, handler_cls):
        h = handler_cls()
        assert h.ctx == {}

    def test_custom_ctx(self, handler_cls):
        ctx = {"key": "value"}
        h = handler_cls(ctx=ctx)
        assert h.ctx == ctx

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "connector"

    def test_routes_count(self, handler):
        assert len(handler.ROUTES) == 5

    def test_route_patterns_count(self, handler):
        assert len(handler.ROUTE_PATTERNS) == 2


class TestFactory:
    """Test the factory function."""

    def test_create_handler(self, handler_module):
        h = handler_module.create_teams_oauth_handler({"server": True})
        assert isinstance(h, handler_module.TeamsOAuthHandler)


# ============================================================================
# handle() wrapper (delegates to dispatch with GET)
# ============================================================================


class TestHandleWrapper:
    """Test the handle() BaseHandler-compatible entry point."""

    @pytest.mark.asyncio
    async def test_handle_delegates_to_dispatch_get(self, handler, mock_state_store):
        """handle(path, qp, handler) should delegate to dispatch(method='GET', ...)."""
        result = await handler.handle("/api/integrations/teams/install", {}, MagicMock())
        assert _status(result) == 302

    @pytest.mark.asyncio
    async def test_handle_callback(self, handler, mock_state_store, mock_tenant_store):
        """handle() can process the callback route."""
        mock_client, _ = _make_httpx_mock(
            {
                "access_token": "at-new",
                "refresh_token": "rt-new",
                "expires_in": 3600,
                "scope": "openid",
            }
        )
        # Mock org/me responses
        org_resp = MagicMock()
        org_resp.status_code = 200
        org_resp.json.return_value = {"value": [{"id": "tid-1", "displayName": "My Org"}]}
        me_resp = MagicMock()
        me_resp.status_code = 200
        me_resp.json.return_value = {"id": "me-id-1"}
        mock_client.get = AsyncMock(side_effect=[org_resp, me_resp])

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch(
                "aragora.storage.teams_tenant_store.get_teams_tenant_store",
                return_value=mock_tenant_store,
            ):
                result = await handler.handle(
                    "/api/integrations/teams/callback",
                    {"code": "auth-code", "state": "test-state-token-abc123def456"},
                    MagicMock(),
                )
        assert _status(result) == 200


# ============================================================================
# GET /api/integrations/teams/install
# ============================================================================


class TestInstall:
    """Tests for the /install endpoint."""

    @pytest.mark.asyncio
    async def test_returns_302_redirect(self, handler):
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/install",
        )
        assert _status(result) == 302

    @pytest.mark.asyncio
    async def test_redirect_location_contains_ms_oauth_url(self, handler):
        result = await handler.dispatch(method="GET", path="/api/integrations/teams/install")
        location = result.headers.get("Location", "")
        assert "login.microsoftonline.com" in location

    @pytest.mark.asyncio
    async def test_redirect_location_contains_client_id(self, handler):
        result = await handler.dispatch(method="GET", path="/api/integrations/teams/install")
        location = result.headers.get("Location", "")
        assert "client_id=test-client-id" in location

    @pytest.mark.asyncio
    async def test_redirect_location_contains_state(self, handler):
        result = await handler.dispatch(method="GET", path="/api/integrations/teams/install")
        location = result.headers.get("Location", "")
        assert "state=test-state-token-abc123def456" in location

    @pytest.mark.asyncio
    async def test_redirect_location_contains_redirect_uri(self, handler):
        result = await handler.dispatch(method="GET", path="/api/integrations/teams/install")
        location = result.headers.get("Location", "")
        assert "redirect_uri=" in location

    @pytest.mark.asyncio
    async def test_redirect_location_contains_scopes(self, handler):
        result = await handler.dispatch(method="GET", path="/api/integrations/teams/install")
        location = result.headers.get("Location", "")
        assert "scope=" in location

    @pytest.mark.asyncio
    async def test_redirect_no_cache(self, handler):
        result = await handler.dispatch(method="GET", path="/api/integrations/teams/install")
        assert result.headers.get("Cache-Control") == "no-store"

    @pytest.mark.asyncio
    async def test_state_generated_with_org_id(self, handler, mock_state_store):
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/install",
            query_params={"org_id": "org-999"},
        )
        assert _status(result) == 302
        mock_state_store.generate.assert_called_once()
        call_kwargs = mock_state_store.generate.call_args
        metadata = call_kwargs[1]["metadata"] if "metadata" in call_kwargs[1] else call_kwargs[0][0]
        # The metadata should include the org_id
        assert metadata["org_id"] == "org-999"

    @pytest.mark.asyncio
    async def test_install_not_configured(self, handler, handler_module, monkeypatch):
        monkeypatch.setattr(handler_module, "TEAMS_CLIENT_ID", None)
        result = await handler.dispatch(method="GET", path="/api/integrations/teams/install")
        assert _status(result) == 503
        assert "not configured" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_install_state_generation_failure(self, handler, mock_state_store):
        mock_state_store.generate.side_effect = RuntimeError("state store down")
        result = await handler.dispatch(method="GET", path="/api/integrations/teams/install")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_install_method_not_allowed_post(self, handler):
        result = await handler.dispatch(method="POST", path="/api/integrations/teams/install")
        assert _status(result) == 405

    @pytest.mark.asyncio
    async def test_install_localhost_fallback_no_redirect_uri(
        self, handler, handler_module, monkeypatch
    ):
        """When TEAMS_REDIRECT_URI is unset, localhost fallback should be used."""
        monkeypatch.setattr(handler_module, "TEAMS_REDIRECT_URI", None)
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/install",
            query_params={"host": "localhost:8080"},
        )
        assert _status(result) == 302
        location = result.headers.get("Location", "")
        # URL-encoded: localhost%3A8080
        assert "localhost" in location

    @pytest.mark.asyncio
    async def test_install_rejects_non_localhost_without_redirect_uri(
        self, handler, handler_module, monkeypatch
    ):
        """Non-localhost hosts are rejected when TEAMS_REDIRECT_URI is not set."""
        monkeypatch.setattr(handler_module, "TEAMS_REDIRECT_URI", None)
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/install",
            query_params={"host": "evil.com"},
        )
        assert _status(result) == 400
        assert "localhost" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_install_allows_127_0_0_1_fallback(self, handler, handler_module, monkeypatch):
        monkeypatch.setattr(handler_module, "TEAMS_REDIRECT_URI", None)
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/install",
            query_params={"host": "127.0.0.1:3000"},
        )
        assert _status(result) == 302

    @pytest.mark.asyncio
    async def test_install_allows_ipv6_localhost_fallback(
        self, handler, handler_module, monkeypatch
    ):
        monkeypatch.setattr(handler_module, "TEAMS_REDIRECT_URI", None)
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/install",
            query_params={"host": "[::1]:8080"},
        )
        assert _status(result) == 302


# ============================================================================
# GET /api/integrations/teams/callback
# ============================================================================


class TestCallback:
    """Tests for the /callback endpoint."""

    @pytest.fixture
    def token_response(self):
        return {
            "access_token": "at-new-token",
            "refresh_token": "rt-new-refresh",
            "expires_in": 3600,
            "scope": "https://graph.microsoft.com/.default",
        }

    @pytest.fixture
    def org_response(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"value": [{"id": "tid-123", "displayName": "Contoso"}]}
        return resp

    @pytest.fixture
    def me_response(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"id": "user-me-id"}
        return resp

    @pytest.fixture
    def httpx_setup(self, token_response, org_response, me_response, mock_tenant_store):
        """Set up httpx mocking for the full callback flow."""
        mock_client, token_resp = _make_httpx_mock(token_response)
        mock_client.get = AsyncMock(side_effect=[org_response, me_response])
        return mock_client, mock_tenant_store

    async def _call_callback(self, handler, query_params, httpx_setup):
        mock_client, tenant_store = httpx_setup
        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch(
                "aragora.storage.teams_tenant_store.get_teams_tenant_store",
                return_value=tenant_store,
            ):
                with patch(
                    "aragora.storage.teams_tenant_store.TeamsTenant",
                    side_effect=lambda **kw: MagicMock(**kw),
                ):
                    return await handler.dispatch(
                        method="GET",
                        path="/api/integrations/teams/callback",
                        query_params=query_params,
                    )

    @pytest.mark.asyncio
    async def test_successful_callback_returns_200(self, handler, httpx_setup):
        result = await self._call_callback(
            handler,
            {"code": "auth-code", "state": "test-state-token-abc123def456"},
            httpx_setup,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_successful_callback_returns_html(self, handler, httpx_setup):
        result = await self._call_callback(
            handler,
            {"code": "auth-code", "state": "test-state-token-abc123def456"},
            httpx_setup,
        )
        assert result.content_type == "text/html"
        html = _html(result)
        assert "Connected" in html

    @pytest.mark.asyncio
    async def test_callback_html_contains_tenant_name(self, handler, httpx_setup):
        result = await self._call_callback(
            handler,
            {"code": "auth-code", "state": "test-state-token-abc123def456"},
            httpx_setup,
        )
        html = _html(result)
        assert "Contoso" in html

    @pytest.mark.asyncio
    async def test_callback_error_from_microsoft(self, handler):
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/callback",
            query_params={"error": "access_denied", "error_description": "User denied"},
        )
        assert _status(result) == 400
        assert "access_denied" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_callback_missing_code(self, handler):
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/callback",
            query_params={"state": "some-state"},
        )
        assert _status(result) == 400
        assert "code" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_callback_missing_state(self, handler):
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/callback",
            query_params={"code": "auth-code"},
        )
        assert _status(result) == 400
        assert "state" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_callback_invalid_state(self, handler, mock_state_store):
        mock_state_store.validate_and_consume.return_value = None
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/callback",
            query_params={"code": "auth-code", "state": "bad-state"},
        )
        assert _status(result) == 400
        assert "state" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_callback_not_configured(self, handler, handler_module, monkeypatch):
        monkeypatch.setattr(handler_module, "TEAMS_CLIENT_ID", None)
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/callback",
            query_params={"code": "auth-code", "state": "test-state-token-abc123def456"},
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_callback_secret_not_configured(self, handler, handler_module, monkeypatch):
        monkeypatch.setattr(handler_module, "TEAMS_CLIENT_SECRET", None)
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/callback",
            query_params={"code": "auth-code", "state": "test-state-token-abc123def456"},
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_callback_token_exchange_http_error(self, handler, mock_tenant_store):
        import httpx

        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request", request=MagicMock(), response=mock_resp
        )

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/callback",
                query_params={"code": "auth-code", "state": "test-state-token-abc123def456"},
            )
        assert _status(result) == 500
        assert "exchange" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_callback_token_exchange_connection_error(self, handler):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=ConnectionError("network down"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/callback",
                query_params={"code": "auth-code", "state": "test-state-token-abc123def456"},
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_callback_no_access_token_in_response(self, handler, mock_tenant_store):
        mock_client, _ = _make_httpx_mock({"refresh_token": "rt-1", "expires_in": 3600})
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/callback",
                query_params={"code": "auth-code", "state": "test-state-token-abc123def456"},
            )
        assert _status(result) == 500
        assert "invalid" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_callback_org_lookup_failure_still_succeeds(
        self, handler, token_response, mock_tenant_store
    ):
        """If Graph API call for org info fails, callback should still work if tenant_id was found."""
        mock_client, _ = _make_httpx_mock(token_response)
        # First get (org) returns error, second get (me) also fails
        org_resp = MagicMock()
        org_resp.status_code = 403
        org_resp.json.return_value = {}
        me_resp = MagicMock()
        me_resp.status_code = 403
        me_resp.json.return_value = {}
        mock_client.get = AsyncMock(side_effect=[org_resp, me_resp])

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch(
                "aragora.storage.teams_tenant_store.get_teams_tenant_store",
                return_value=mock_tenant_store,
            ):
                with patch(
                    "aragora.storage.teams_tenant_store.TeamsTenant",
                    side_effect=lambda **kw: MagicMock(**kw),
                ):
                    result = await handler.dispatch(
                        method="GET",
                        path="/api/integrations/teams/callback",
                        query_params={
                            "code": "auth-code",
                            "state": "test-state-token-abc123def456",
                        },
                    )
        # No tenant_id could be determined -> 500
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_callback_graph_exception_fallback(
        self, handler, token_response, mock_tenant_store
    ):
        """ConnectionError during Graph calls should set fallback tenant_id."""
        mock_client, _ = _make_httpx_mock(token_response)
        mock_client.get = AsyncMock(side_effect=ConnectionError("graph down"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch(
                "aragora.storage.teams_tenant_store.get_teams_tenant_store",
                return_value=mock_tenant_store,
            ):
                with patch(
                    "aragora.storage.teams_tenant_store.TeamsTenant",
                    side_effect=lambda **kw: MagicMock(**kw),
                ):
                    result = await handler.dispatch(
                        method="GET",
                        path="/api/integrations/teams/callback",
                        query_params={
                            "code": "auth-code",
                            "state": "test-state-token-abc123def456",
                        },
                    )
        # tenant_id fallback is "unknown" -- so it succeeds
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_callback_store_save_failure(
        self, handler, token_response, org_response, me_response
    ):
        mock_client, _ = _make_httpx_mock(token_response)
        mock_client.get = AsyncMock(side_effect=[org_response, me_response])
        tenant_store = MagicMock()
        tenant_store.save.return_value = False

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch(
                "aragora.storage.teams_tenant_store.get_teams_tenant_store",
                return_value=tenant_store,
            ):
                with patch(
                    "aragora.storage.teams_tenant_store.TeamsTenant",
                    side_effect=lambda **kw: MagicMock(**kw),
                ):
                    result = await handler.dispatch(
                        method="GET",
                        path="/api/integrations/teams/callback",
                        query_params={
                            "code": "auth-code",
                            "state": "test-state-token-abc123def456",
                        },
                    )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_callback_tenant_store_import_error(
        self, handler, token_response, org_response, me_response
    ):
        mock_client, _ = _make_httpx_mock(token_response)
        mock_client.get = AsyncMock(side_effect=[org_response, me_response])

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch(
                "aragora.storage.teams_tenant_store.get_teams_tenant_store",
                side_effect=ImportError("no module"),
            ):
                with patch(
                    "aragora.storage.teams_tenant_store.TeamsTenant",
                    side_effect=ImportError("no module"),
                ):
                    result = await handler.dispatch(
                        method="GET",
                        path="/api/integrations/teams/callback",
                        query_params={
                            "code": "auth-code",
                            "state": "test-state-token-abc123def456",
                        },
                    )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_callback_method_not_allowed_post(self, handler):
        result = await handler.dispatch(
            method="POST",
            path="/api/integrations/teams/callback",
        )
        assert _status(result) == 405

    @pytest.mark.asyncio
    async def test_callback_no_auth_required(self, handler, httpx_setup):
        """Callback route should work without authentication."""
        result = await self._call_callback(
            handler,
            {"code": "auth-code", "state": "test-state-token-abc123def456"},
            httpx_setup,
        )
        # Should not return 401 -- callback is unauthenticated
        assert _status(result) != 401

    @pytest.mark.asyncio
    async def test_callback_state_metadata_none(self, handler, mock_state_store, httpx_setup):
        """State data with no metadata should still work (org_id = None)."""
        state_data = MagicMock()
        state_data.metadata = None
        mock_state_store.validate_and_consume.return_value = state_data
        result = await self._call_callback(
            handler,
            {"code": "auth-code", "state": "test-state-token-abc123def456"},
            httpx_setup,
        )
        assert _status(result) == 200


# ============================================================================
# POST /api/integrations/teams/refresh
# ============================================================================


class TestRefresh:
    """Tests for the /refresh endpoint."""

    @pytest.mark.asyncio
    async def test_successful_refresh(self, handler, mock_tenant_store):
        token_data = {
            "access_token": "at-refreshed",
            "refresh_token": "rt-refreshed",
            "expires_in": 7200,
        }
        mock_client, _ = _make_httpx_mock(token_data)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch(
                "aragora.storage.teams_tenant_store.get_teams_tenant_store",
                return_value=mock_tenant_store,
            ):
                result = await handler.dispatch(
                    method="POST",
                    path="/api/integrations/teams/refresh",
                    body={"tenant_id": "tenant-abc"},
                )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["tenant_id"] == "tenant-abc"
        assert body["expires_in"] == 7200

    @pytest.mark.asyncio
    async def test_refresh_missing_tenant_id(self, handler):
        result = await handler.dispatch(
            method="POST",
            path="/api/integrations/teams/refresh",
            body={},
        )
        assert _status(result) == 400
        assert "tenant_id" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_refresh_not_configured(self, handler, handler_module, monkeypatch):
        monkeypatch.setattr(handler_module, "TEAMS_CLIENT_ID", None)
        result = await handler.dispatch(
            method="POST",
            path="/api/integrations/teams/refresh",
            body={"tenant_id": "tid-1"},
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_refresh_tenant_not_found(self, handler):
        store = MagicMock()
        store.get.return_value = None
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="POST",
                path="/api/integrations/teams/refresh",
                body={"tenant_id": "nonexistent"},
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_refresh_no_refresh_token(self, handler):
        tenant = _make_tenant(refresh_token=None)
        store = MagicMock()
        store.get.return_value = tenant
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="POST",
                path="/api/integrations/teams/refresh",
                body={"tenant_id": "tid-1"},
            )
        assert _status(result) == 400
        assert "refresh" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_refresh_store_import_error(self, handler):
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            side_effect=ImportError("no module"),
        ):
            result = await handler.dispatch(
                method="POST",
                path="/api/integrations/teams/refresh",
                body={"tenant_id": "tid-1"},
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_refresh_http_error(self, handler, mock_tenant_store):
        import httpx

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=mock_resp
        )
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch(
                "aragora.storage.teams_tenant_store.get_teams_tenant_store",
                return_value=mock_tenant_store,
            ):
                result = await handler.dispatch(
                    method="POST",
                    path="/api/integrations/teams/refresh",
                    body={"tenant_id": "tenant-abc"},
                )
        assert _status(result) == 500
        assert "refresh" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_refresh_connection_error(self, handler, mock_tenant_store):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=TimeoutError("timed out"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch(
                "aragora.storage.teams_tenant_store.get_teams_tenant_store",
                return_value=mock_tenant_store,
            ):
                result = await handler.dispatch(
                    method="POST",
                    path="/api/integrations/teams/refresh",
                    body={"tenant_id": "tenant-abc"},
                )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_refresh_no_access_token_in_response(self, handler, mock_tenant_store):
        mock_client, _ = _make_httpx_mock({"expires_in": 3600})
        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch(
                "aragora.storage.teams_tenant_store.get_teams_tenant_store",
                return_value=mock_tenant_store,
            ):
                result = await handler.dispatch(
                    method="POST",
                    path="/api/integrations/teams/refresh",
                    body={"tenant_id": "tenant-abc"},
                )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_refresh_update_tokens_failure(self, handler):
        tenant = _make_tenant()
        store = MagicMock()
        store.get.return_value = tenant
        store.update_tokens.return_value = False
        token_data = {
            "access_token": "at-refreshed",
            "refresh_token": "rt-refreshed",
            "expires_in": 3600,
        }
        mock_client, _ = _make_httpx_mock(token_data)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch(
                "aragora.storage.teams_tenant_store.get_teams_tenant_store",
                return_value=store,
            ):
                result = await handler.dispatch(
                    method="POST",
                    path="/api/integrations/teams/refresh",
                    body={"tenant_id": "tenant-abc"},
                )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_refresh_keeps_old_refresh_token_when_not_returned(
        self, handler, mock_tenant_store
    ):
        """If Microsoft doesn't return a new refresh_token, the old one should be kept."""
        token_data = {
            "access_token": "at-refreshed",
            "expires_in": 3600,
            # No refresh_token in response
        }
        mock_client, _ = _make_httpx_mock(token_data)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch(
                "aragora.storage.teams_tenant_store.get_teams_tenant_store",
                return_value=mock_tenant_store,
            ):
                result = await handler.dispatch(
                    method="POST",
                    path="/api/integrations/teams/refresh",
                    body={"tenant_id": "tenant-abc"},
                )
        assert _status(result) == 200
        # Verify update_tokens was called with the old refresh token
        mock_tenant_store.update_tokens.assert_called_once()
        call_args = mock_tenant_store.update_tokens.call_args
        # refresh_token arg should be the old one ("rt-refresh-xyz" from _make_tenant)
        assert call_args[0][2] == "rt-refresh-xyz"

    @pytest.mark.asyncio
    async def test_refresh_method_not_allowed_get(self, handler):
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/refresh",
        )
        assert _status(result) == 405


# ============================================================================
# POST /api/integrations/teams/disconnect
# ============================================================================


class TestDisconnect:
    """Tests for the /disconnect endpoint."""

    @pytest.mark.asyncio
    async def test_successful_disconnect(self, handler, mock_tenant_store, mock_tenant):
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=mock_tenant_store,
        ):
            result = await handler.dispatch(
                method="POST",
                path="/api/integrations/teams/disconnect",
                body={"tenant_id": "tenant-abc"},
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["tenant_id"] == "tenant-abc"
        assert "disconnect" in body.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_disconnect_missing_tenant_id(self, handler):
        result = await handler.dispatch(
            method="POST",
            path="/api/integrations/teams/disconnect",
            body={},
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_disconnect_tenant_not_found(self, handler):
        store = MagicMock()
        store.get.return_value = None
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="POST",
                path="/api/integrations/teams/disconnect",
                body={"tenant_id": "nonexistent"},
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_disconnect_save_failure(self, handler):
        tenant = _make_tenant()
        store = MagicMock()
        store.get.return_value = tenant
        store.save.return_value = False
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="POST",
                path="/api/integrations/teams/disconnect",
                body={"tenant_id": "tid-1"},
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_disconnect_import_error(self, handler):
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            side_effect=ImportError("no module"),
        ):
            result = await handler.dispatch(
                method="POST",
                path="/api/integrations/teams/disconnect",
                body={"tenant_id": "tid-1"},
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_disconnect_runtime_error(self, handler):
        store = MagicMock()
        store.get.side_effect = RuntimeError("db corrupt")
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="POST",
                path="/api/integrations/teams/disconnect",
                body={"tenant_id": "tid-1"},
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_disconnect_deactivates_tenant(self, handler, mock_tenant_store, mock_tenant):
        """Disconnect should set is_active = False and save."""
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=mock_tenant_store,
        ):
            await handler.dispatch(
                method="POST",
                path="/api/integrations/teams/disconnect",
                body={"tenant_id": "tenant-abc"},
            )
        assert mock_tenant.is_active is False
        mock_tenant_store.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_method_not_allowed_get(self, handler):
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/disconnect",
        )
        assert _status(result) == 405


# ============================================================================
# GET /api/integrations/teams/tenants
# ============================================================================


class TestListTenants:
    """Tests for the /tenants endpoint."""

    @pytest.mark.asyncio
    async def test_list_tenants_success(self, handler, mock_tenant_store):
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=mock_tenant_store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants",
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert len(body["tenants"]) == 1
        assert body["tenants"][0]["tenant_id"] == "tenant-abc"

    @pytest.mark.asyncio
    async def test_list_tenants_empty(self, handler):
        store = MagicMock()
        store.list_all.return_value = []
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants",
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 0
        assert body["tenants"] == []

    @pytest.mark.asyncio
    async def test_list_tenants_token_valid(self, handler):
        """Token expiring in >1hr should be 'valid'."""
        tenant = _make_tenant(expires_at=time.time() + 7200)
        store = MagicMock()
        store.list_all.return_value = [tenant]
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(method="GET", path="/api/integrations/teams/tenants")
        body = _body(result)
        assert body["tenants"][0]["token_status"] == "valid"

    @pytest.mark.asyncio
    async def test_list_tenants_token_expired(self, handler):
        tenant = _make_tenant(expires_at=time.time() - 100)
        store = MagicMock()
        store.list_all.return_value = [tenant]
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(method="GET", path="/api/integrations/teams/tenants")
        body = _body(result)
        assert body["tenants"][0]["token_status"] == "expired"

    @pytest.mark.asyncio
    async def test_list_tenants_token_expiring_soon(self, handler):
        """Token expiring in <1hr should be 'expiring_soon'."""
        tenant = _make_tenant(expires_at=time.time() + 1800)
        store = MagicMock()
        store.list_all.return_value = [tenant]
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(method="GET", path="/api/integrations/teams/tenants")
        body = _body(result)
        assert body["tenants"][0]["token_status"] == "expiring_soon"

    @pytest.mark.asyncio
    async def test_list_tenants_token_no_expiry(self, handler):
        """Token with no expires_at should be 'valid'."""
        tenant = _make_tenant(expires_at=None)
        store = MagicMock()
        store.list_all.return_value = [tenant]
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(method="GET", path="/api/integrations/teams/tenants")
        body = _body(result)
        assert body["tenants"][0]["token_status"] == "valid"

    @pytest.mark.asyncio
    async def test_list_tenants_import_error(self, handler):
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            side_effect=ImportError("no module"),
        ):
            result = await handler.dispatch(method="GET", path="/api/integrations/teams/tenants")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_list_tenants_runtime_error(self, handler):
        store = MagicMock()
        store.list_all.side_effect = RuntimeError("db down")
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(method="GET", path="/api/integrations/teams/tenants")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_list_tenants_multiple(self, handler):
        t1 = _make_tenant(tenant_id="t1", tenant_name="Org A")
        t2 = _make_tenant(tenant_id="t2", tenant_name="Org B")
        store = MagicMock()
        store.list_all.return_value = [t1, t2]
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(method="GET", path="/api/integrations/teams/tenants")
        body = _body(result)
        assert body["total"] == 2
        ids = {t["tenant_id"] for t in body["tenants"]}
        assert ids == {"t1", "t2"}

    @pytest.mark.asyncio
    async def test_list_tenants_method_not_allowed_post(self, handler):
        result = await handler.dispatch(
            method="POST",
            path="/api/integrations/teams/tenants",
        )
        assert _status(result) == 405

    @pytest.mark.asyncio
    async def test_list_tenants_includes_all_fields(self, handler, mock_tenant_store):
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=mock_tenant_store,
        ):
            result = await handler.dispatch(method="GET", path="/api/integrations/teams/tenants")
        body = _body(result)
        t = body["tenants"][0]
        expected_keys = {
            "tenant_id",
            "tenant_name",
            "is_active",
            "token_status",
            "expires_at",
            "installed_at",
            "installed_by",
            "scopes",
            "aragora_org_id",
        }
        assert expected_keys.issubset(set(t.keys()))


# ============================================================================
# GET /api/integrations/teams/tenants/{id}
# ============================================================================


class TestGetTenant:
    """Tests for the GET /tenants/{id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_tenant_success(self, handler, mock_tenant_store):
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=mock_tenant_store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tenant-abc",
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["tenant_id"] == "tenant-abc"
        assert body["tenant_name"] == "Contoso Ltd"
        assert body["is_active"] is True

    @pytest.mark.asyncio
    async def test_get_tenant_not_found(self, handler):
        store = MagicMock()
        store.get.return_value = None
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/nonexistent",
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_tenant_token_valid(self, handler):
        tenant = _make_tenant(expires_at=time.time() + 100000)
        store = MagicMock()
        store.get.return_value = tenant
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid",
            )
        body = _body(result)
        assert body["token_status"] == "valid"

    @pytest.mark.asyncio
    async def test_get_tenant_token_expired(self, handler):
        tenant = _make_tenant(expires_at=time.time() - 100)
        store = MagicMock()
        store.get.return_value = tenant
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid",
            )
        body = _body(result)
        assert body["token_status"] == "expired"
        assert body["expires_in_seconds"] < 0

    @pytest.mark.asyncio
    async def test_get_tenant_token_expiring_soon(self, handler):
        tenant = _make_tenant(expires_at=time.time() + 1800)
        store = MagicMock()
        store.get.return_value = tenant
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid",
            )
        body = _body(result)
        assert body["token_status"] == "expiring_soon"

    @pytest.mark.asyncio
    async def test_get_tenant_token_expiring_today(self, handler):
        """Token expiring within 24hr but more than 1hr should be 'expiring_today'."""
        tenant = _make_tenant(expires_at=time.time() + 43200)  # 12 hours
        store = MagicMock()
        store.get.return_value = tenant
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid",
            )
        body = _body(result)
        assert body["token_status"] == "expiring_today"

    @pytest.mark.asyncio
    async def test_get_tenant_has_refresh_token(self, handler):
        tenant = _make_tenant(refresh_token="rt-yes")
        store = MagicMock()
        store.get.return_value = tenant
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid",
            )
        body = _body(result)
        assert body["has_refresh_token"] is True

    @pytest.mark.asyncio
    async def test_get_tenant_no_refresh_token(self, handler):
        tenant = _make_tenant(refresh_token=None)
        store = MagicMock()
        store.get.return_value = tenant
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid",
            )
        body = _body(result)
        assert body["has_refresh_token"] is False

    @pytest.mark.asyncio
    async def test_get_tenant_import_error(self, handler):
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            side_effect=ImportError("no module"),
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid-1",
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_tenant_runtime_error(self, handler):
        store = MagicMock()
        store.get.side_effect = RuntimeError("db error")
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid-1",
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_tenant_includes_all_fields(self, handler, mock_tenant_store):
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=mock_tenant_store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tenant-abc",
            )
        body = _body(result)
        expected_keys = {
            "tenant_id",
            "tenant_name",
            "is_active",
            "token_status",
            "expires_at",
            "expires_in_seconds",
            "has_refresh_token",
            "scopes",
            "installed_at",
            "installed_by",
            "bot_id",
            "aragora_org_id",
        }
        assert expected_keys.issubset(set(body.keys()))

    @pytest.mark.asyncio
    async def test_get_tenant_no_expires_at(self, handler):
        tenant = _make_tenant(expires_at=None)
        store = MagicMock()
        store.get.return_value = tenant
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid",
            )
        body = _body(result)
        assert body["token_status"] == "valid"
        assert body["expires_in_seconds"] is None


# ============================================================================
# DELETE /api/integrations/teams/tenants/{id}
# ============================================================================


class TestDeleteTenant:
    """Tests for DELETE /tenants/{id} which routes through _handle_disconnect."""

    @pytest.mark.asyncio
    async def test_delete_tenant(self, handler, mock_tenant_store, mock_tenant):
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=mock_tenant_store,
        ):
            result = await handler.dispatch(
                method="DELETE",
                path="/api/integrations/teams/tenants/tenant-abc",
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_delete_tenant_not_found(self, handler):
        store = MagicMock()
        store.get.return_value = None
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="DELETE",
                path="/api/integrations/teams/tenants/nonexistent",
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_delete_tenant_method_not_allowed_put(self, handler):
        result = await handler.dispatch(
            method="PUT",
            path="/api/integrations/teams/tenants/tid-1",
        )
        assert _status(result) == 405


# ============================================================================
# GET /api/integrations/teams/tenants/{id}/status
# ============================================================================


class TestTenantStatus:
    """Tests for the /tenants/{id}/status endpoint."""

    @pytest.mark.asyncio
    async def test_status_success(self, handler, mock_tenant_store, mock_tenant):
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=mock_tenant_store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tenant-abc/status",
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["tenant_id"] == "tenant-abc"
        assert "token_status" in body
        assert "can_refresh" in body

    @pytest.mark.asyncio
    async def test_status_not_found(self, handler):
        store = MagicMock()
        store.get.return_value = None
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/nonexistent/status",
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_status_token_valid(self, handler):
        tenant = _make_tenant(expires_at=time.time() + 100000)
        store = MagicMock()
        store.get.return_value = tenant
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid/status",
            )
        body = _body(result)
        assert body["token_status"] == "valid"
        assert body["expires_in_seconds"] > 0

    @pytest.mark.asyncio
    async def test_status_token_expired(self, handler):
        tenant = _make_tenant(expires_at=time.time() - 600)
        store = MagicMock()
        store.get.return_value = tenant
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid/status",
            )
        body = _body(result)
        assert body["token_status"] == "expired"

    @pytest.mark.asyncio
    async def test_status_token_expiring_soon(self, handler):
        tenant = _make_tenant(expires_at=time.time() + 1800)
        store = MagicMock()
        store.get.return_value = tenant
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid/status",
            )
        body = _body(result)
        assert body["token_status"] == "expiring_soon"

    @pytest.mark.asyncio
    async def test_status_token_expiring_today(self, handler):
        tenant = _make_tenant(expires_at=time.time() + 43200)
        store = MagicMock()
        store.get.return_value = tenant
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid/status",
            )
        body = _body(result)
        assert body["token_status"] == "expiring_today"

    @pytest.mark.asyncio
    async def test_status_can_refresh_true(self, handler):
        tenant = _make_tenant(refresh_token="rt-yes", is_active=True)
        store = MagicMock()
        store.get.return_value = tenant
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid/status",
            )
        body = _body(result)
        assert body["can_refresh"] is True

    @pytest.mark.asyncio
    async def test_status_can_refresh_false_no_token(self, handler):
        tenant = _make_tenant(refresh_token=None, is_active=True)
        store = MagicMock()
        store.get.return_value = tenant
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid/status",
            )
        body = _body(result)
        assert body["can_refresh"] is False

    @pytest.mark.asyncio
    async def test_status_can_refresh_false_inactive(self, handler):
        tenant = _make_tenant(refresh_token="rt-yes", is_active=False)
        store = MagicMock()
        store.get.return_value = tenant
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid/status",
            )
        body = _body(result)
        assert body["can_refresh"] is False

    @pytest.mark.asyncio
    async def test_status_refresh_recommended_in(self, handler):
        """When token is valid, refresh_recommended_in should be set."""
        tenant = _make_tenant(expires_at=time.time() + 7200)  # 2 hours
        store = MagicMock()
        store.get.return_value = tenant
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid/status",
            )
        body = _body(result)
        assert body["refresh_recommended_in"] is not None
        assert body["refresh_recommended_in"] >= 0

    @pytest.mark.asyncio
    async def test_status_refresh_recommended_none_expired(self, handler):
        """When token is expired, refresh_recommended_in should be None."""
        tenant = _make_tenant(expires_at=time.time() - 600)
        store = MagicMock()
        store.get.return_value = tenant
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid/status",
            )
        body = _body(result)
        assert body["refresh_recommended_in"] is None

    @pytest.mark.asyncio
    async def test_status_no_expires_at(self, handler):
        tenant = _make_tenant(expires_at=None)
        store = MagicMock()
        store.get.return_value = tenant
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid/status",
            )
        body = _body(result)
        assert body["token_status"] == "valid"
        assert body["expires_in_seconds"] is None

    @pytest.mark.asyncio
    async def test_status_import_error(self, handler):
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            side_effect=ImportError("no module"),
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid/status",
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_status_runtime_error(self, handler):
        store = MagicMock()
        store.get.side_effect = RuntimeError("db down")
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid/status",
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_status_method_not_allowed_post(self, handler):
        result = await handler.dispatch(
            method="POST",
            path="/api/integrations/teams/tenants/tid/status",
        )
        assert _status(result) == 405

    @pytest.mark.asyncio
    async def test_status_includes_all_fields(self, handler, mock_tenant_store):
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=mock_tenant_store,
        ):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tenant-abc/status",
            )
        body = _body(result)
        expected_keys = {
            "tenant_id",
            "tenant_name",
            "is_active",
            "token_status",
            "expires_at",
            "expires_in_seconds",
            "has_refresh_token",
            "refresh_recommended_in",
            "can_refresh",
            "scopes",
        }
        assert expected_keys.issubset(set(body.keys()))


# ============================================================================
# Not found / unmatched routes
# ============================================================================


class TestNotFound:
    """Tests for routes that don't match any endpoint."""

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self, handler):
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/unknown",
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_extra_path_segments_404(self, handler):
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/install/extra/stuff",
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_prefix_404(self, handler):
        result = await handler.dispatch(
            method="GET",
            path="/api/v1/integrations/teams/install",
        )
        assert _status(result) == 404


# ============================================================================
# Permission checks (no_auto_auth)
# ============================================================================


class TestPermissions:
    """Test RBAC permission enforcement."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_install_requires_auth(self, handler):
        """Install endpoint should require authentication."""
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/install",
            handler=MagicMock(),
        )
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_refresh_requires_auth(self, handler):
        result = await handler.dispatch(
            method="POST",
            path="/api/integrations/teams/refresh",
            body={"tenant_id": "tid-1"},
            handler=MagicMock(),
        )
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_disconnect_requires_auth(self, handler):
        result = await handler.dispatch(
            method="POST",
            path="/api/integrations/teams/disconnect",
            body={"tenant_id": "tid-1"},
            handler=MagicMock(),
        )
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_tenants_list_requires_auth(self, handler):
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/tenants",
            handler=MagicMock(),
        )
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_tenant_get_requires_auth(self, handler):
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/tenants/tid-1",
            handler=MagicMock(),
        )
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_tenant_status_requires_auth(self, handler):
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/tenants/tid-1/status",
            handler=MagicMock(),
        )
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_callback_does_not_require_auth(self, handler, mock_state_store):
        """Callback from Microsoft does NOT require auth - it validates state."""
        mock_state_store.validate_and_consume.return_value = None
        result = await handler.dispatch(
            method="GET",
            path="/api/integrations/teams/callback",
            query_params={"code": "c", "state": "bad"},
            handler=MagicMock(),
        )
        # Should get 400 for bad state, not 401 for missing auth
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_install_permission_denied(self, handler):
        """When _check_permission raises ForbiddenError, install returns 403."""
        from aragora.server.handlers.secure import ForbiddenError

        with patch.object(handler, "_check_permission", side_effect=ForbiddenError("denied")):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/install",
            )
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_refresh_permission_denied(self, handler):
        from aragora.server.handlers.secure import ForbiddenError

        with patch.object(handler, "_check_permission", side_effect=ForbiddenError("denied")):
            result = await handler.dispatch(
                method="POST",
                path="/api/integrations/teams/refresh",
                body={"tenant_id": "tid"},
            )
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_disconnect_permission_denied(self, handler):
        from aragora.server.handlers.secure import ForbiddenError

        with patch.object(handler, "_check_permission", side_effect=ForbiddenError("denied")):
            result = await handler.dispatch(
                method="POST",
                path="/api/integrations/teams/disconnect",
                body={"tenant_id": "tid"},
            )
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_tenants_list_permission_denied(self, handler):
        from aragora.server.handlers.secure import ForbiddenError

        with patch.object(handler, "_check_permission", side_effect=ForbiddenError("denied")):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants",
            )
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_tenant_get_permission_denied(self, handler):
        from aragora.server.handlers.secure import ForbiddenError

        with patch.object(handler, "_check_permission", side_effect=ForbiddenError("denied")):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid-1",
            )
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_tenant_delete_permission_denied(self, handler):
        from aragora.server.handlers.secure import ForbiddenError

        with patch.object(handler, "_check_permission", side_effect=ForbiddenError("denied")):
            result = await handler.dispatch(
                method="DELETE",
                path="/api/integrations/teams/tenants/tid-1",
            )
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_tenant_status_permission_denied(self, handler):
        from aragora.server.handlers.secure import ForbiddenError

        with patch.object(handler, "_check_permission", side_effect=ForbiddenError("denied")):
            result = await handler.dispatch(
                method="GET",
                path="/api/integrations/teams/tenants/tid-1/status",
            )
        assert _status(result) == 403


# ============================================================================
# _check_permission helper
# ============================================================================


class TestCheckPermission:
    """Tests for the _check_permission helper method."""

    def test_primary_permission_granted(self, handler):
        """If primary permission passes, returns True."""
        with patch.object(handler, "check_permission", return_value=True):
            assert handler._check_permission(MagicMock(), "teams:oauth:install") is True

    def test_primary_denied_fallback_granted(self, handler):
        """If primary fails but fallback passes, returns True."""
        from aragora.server.handlers.secure import ForbiddenError

        call_count = 0

        def side_effect(ctx, perm):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ForbiddenError("denied")
            return True

        with patch.object(handler, "check_permission", side_effect=side_effect):
            result = handler._check_permission(
                MagicMock(), "teams:oauth:install", fallback_permission="connectors.authorize"
            )
        assert result is True

    def test_both_denied_raises(self, handler):
        """If both primary and fallback fail, raises ForbiddenError."""
        from aragora.server.handlers.secure import ForbiddenError

        with patch.object(handler, "check_permission", side_effect=ForbiddenError("denied")):
            with pytest.raises(ForbiddenError):
                handler._check_permission(
                    MagicMock(),
                    "teams:oauth:install",
                    fallback_permission="connectors.authorize",
                )

    def test_primary_denied_no_fallback_raises(self, handler):
        """If primary fails and there is no fallback, raises."""
        from aragora.server.handlers.secure import ForbiddenError

        with patch.object(handler, "check_permission", side_effect=ForbiddenError("denied")):
            with pytest.raises(ForbiddenError):
                handler._check_permission(MagicMock(), "teams:oauth:install")


# ============================================================================
# Module-level constants
# ============================================================================


class TestModuleConstants:
    """Test module-level constants are correct."""

    def test_permission_constants(self, handler_module):
        assert handler_module.PERM_TEAMS_OAUTH_INSTALL == "teams:oauth:install"
        assert handler_module.PERM_TEAMS_OAUTH_CALLBACK == "teams:oauth:callback"
        assert handler_module.PERM_TEAMS_OAUTH_DISCONNECT == "teams:oauth:disconnect"
        assert handler_module.PERM_TEAMS_TENANT_MANAGE == "teams:tenant:manage"
        assert handler_module.PERM_TEAMS_ADMIN == "teams:admin"
        assert handler_module.CONNECTOR_AUTHORIZE == "connectors.authorize"

    def test_oauth_urls(self, handler_module):
        assert "login.microsoftonline.com" in handler_module.MS_OAUTH_AUTHORIZE_URL
        assert "login.microsoftonline.com" in handler_module.MS_OAUTH_TOKEN_URL

    def test_default_scopes(self, handler_module):
        assert "graph.microsoft.com" in handler_module.DEFAULT_SCOPES
        assert "offline_access" in handler_module.DEFAULT_SCOPES
