"""Tests for partner handler (aragora/server/handlers/partner.py).

Covers all routes and behavior of the PartnerHandler class:
- can_handle() routing for all static and dynamic routes
- POST /api/v1/partners/register           - Register as a partner
- GET  /api/v1/partners/me                 - Get current partner profile
- POST /api/v1/partners/keys               - Create API key
- GET  /api/v1/partners/keys               - List API keys
- POST /api/v1/partners/keys/{key_id}/rotate - Rotate API key
- DELETE /api/v1/partners/keys/{key_id}    - Revoke API key
- GET  /api/v1/partners/usage              - Get usage statistics
- POST /api/v1/partners/webhooks           - Configure webhook
- GET  /api/v1/partners/limits             - Get rate limits
- Input validation, error handling, edge cases
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.partner import PartnerHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract body dict from a HandlerResult."""
    if hasattr(result, "body"):
        return json.loads(result.body.decode("utf-8"))
    if isinstance(result, dict):
        return result.get("body", result)
    return {}


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status_code", result.get("status", 200))
    return 200


class MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to PartnerHandler."""

    def __init__(
        self,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.command = method
        self.rfile = MagicMock()
        self.client_address = ("127.0.0.1", 12345)

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers = {
                "Content-Length": str(len(raw)),
                "Content-Type": "application/json",
            }
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}

        # Overlay extra headers (e.g. X-Partner-ID)
        if headers:
            self.headers.update(headers)


def _make_auth_ctx(authenticated: bool = True):
    """Create a mock UserAuthContext that satisfies require_user_auth."""
    ctx = MagicMock()
    ctx.authenticated = authenticated
    ctx.is_authenticated = authenticated
    ctx.user_id = "test-user-001"
    ctx.email = "test@example.com"
    ctx.org_id = "test-org-001"
    ctx.role = "admin"
    ctx.error_reason = None if authenticated else "Invalid token"
    return ctx


# ---------------------------------------------------------------------------
# Mock partner domain objects
# ---------------------------------------------------------------------------


class MockPartnerTier(Enum):
    STARTER = "starter"
    GROWTH = "growth"
    ENTERPRISE = "enterprise"


class MockPartnerStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"


class MockPartner:
    """Mock Partner object."""

    def __init__(
        self,
        partner_id: str = "partner-001",
        name: str = "Test Partner",
        email: str = "partner@example.com",
        company: str | None = "TestCo",
        tier: MockPartnerTier = MockPartnerTier.STARTER,
        status: MockPartnerStatus = MockPartnerStatus.PENDING,
        referral_code: str = "REF12345",
        webhook_url: str | None = None,
    ):
        self.partner_id = partner_id
        self.name = name
        self.email = email
        self.company = company
        self.tier = tier
        self.status = status
        self.referral_code = referral_code
        self.webhook_url = webhook_url


class MockAPIKey:
    """Mock API key object."""

    def __init__(
        self,
        key_id: str = "key-001",
        key_prefix: str = "ara_test",
        name: str = "My API Key",
        scopes: list[str] | None = None,
        created_at: datetime | None = None,
        expires_at: datetime | None = None,
        is_active: bool = True,
    ):
        self.key_id = key_id
        self.key_prefix = key_prefix
        self.name = name
        self.scopes = scopes or ["debates:read"]
        self.created_at = created_at or datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.expires_at = expires_at
        self.is_active = is_active

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_id": self.key_id,
            "key_prefix": self.key_prefix,
            "name": self.name,
            "scopes": self.scopes,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
        }


class MockTierLimits:
    """Mock partner tier limits."""

    def __init__(self):
        self.requests_per_minute = 60
        self.requests_per_day = 10000
        self.debates_per_month = 100
        self.max_agents_per_debate = 10
        self.max_rounds = 5
        self.webhook_endpoints = 3
        self.revenue_share_percent = 20.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a PartnerHandler."""
    return PartnerHandler(server_context={})


@pytest.fixture
def auth_ctx():
    """Return an authenticated user context mock."""
    return _make_auth_ctx(authenticated=True)


@pytest.fixture
def unauth_ctx():
    """Return an unauthenticated user context mock."""
    return _make_auth_ctx(authenticated=False)


@pytest.fixture
def mock_partner_api():
    """Create a mock partner API with common stubs."""
    api = MagicMock()
    api._store = MagicMock()
    return api


# ---------------------------------------------------------------------------
# can_handle() routing
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Verify can_handle accepts or rejects paths correctly."""

    def test_register_route(self, handler):
        assert handler.can_handle("/api/v1/partners/register")

    def test_me_route(self, handler):
        assert handler.can_handle("/api/v1/partners/me")

    def test_keys_route(self, handler):
        assert handler.can_handle("/api/v1/partners/keys")

    def test_usage_route(self, handler):
        assert handler.can_handle("/api/v1/partners/usage")

    def test_webhooks_route(self, handler):
        assert handler.can_handle("/api/v1/partners/webhooks")

    def test_limits_route(self, handler):
        assert handler.can_handle("/api/v1/partners/limits")

    def test_key_rotate_route(self, handler):
        assert handler.can_handle("/api/v1/partners/keys/key-001/rotate")

    def test_key_delete_route(self, handler):
        assert handler.can_handle("/api/v1/partners/keys/key-001")

    def test_rejects_unrelated_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_rejects_partial_path(self, handler):
        assert not handler.can_handle("/api/v1/partners")

    def test_rejects_v2_path(self, handler):
        assert not handler.can_handle("/api/v2/partners/register")


# ---------------------------------------------------------------------------
# handle() route dispatch
# ---------------------------------------------------------------------------


class TestRouteDispatch:
    """Verify handle() routes to the right internal method."""

    def test_unmatched_route_returns_none(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(method="GET")
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/some/other/path", {}, mock_h)
        assert result is None

    def test_unmatched_method_returns_none(self, handler, auth_ctx):
        """PUT on /api/v1/partners/register is not routed."""
        mock_h = MockHTTPHandler(method="PUT")
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/register", {}, mock_h)
        assert result is None

    def test_get_on_register_returns_none(self, handler, auth_ctx):
        """GET on /api/v1/partners/register is not routed."""
        mock_h = MockHTTPHandler(method="GET")
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/register", {}, mock_h)
        assert result is None


# ---------------------------------------------------------------------------
# POST /api/v1/partners/register
# ---------------------------------------------------------------------------


class TestRegisterPartner:
    """Tests for partner registration endpoint."""

    def test_register_success(self, handler, auth_ctx):
        partner = MockPartner()
        mock_api = MagicMock()
        mock_api._store.get_partner_by_email.return_value = None
        mock_api.register_partner.return_value = partner

        mock_h = MockHTTPHandler(
            method="POST",
            body={"name": "Test Partner", "email": "partner@example.com", "company": "TestCo"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/register", {}, mock_h)

        assert _status(result) == 201
        body = _body(result)
        assert body["partner_id"] == "partner-001"
        assert body["name"] == "Test Partner"
        assert body["email"] == "partner@example.com"
        assert body["tier"] == "starter"
        assert body["status"] == "pending"
        assert body["referral_code"] == "REF12345"
        assert "message" in body

    def test_register_with_company(self, handler, auth_ctx):
        partner = MockPartner(company="Acme Inc")
        mock_api = MagicMock()
        mock_api._store.get_partner_by_email.return_value = None
        mock_api.register_partner.return_value = partner

        mock_h = MockHTTPHandler(
            method="POST",
            body={"name": "Acme Partner", "email": "partner@acme.com", "company": "Acme Inc"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/register", {}, mock_h)

        assert _status(result) == 201
        body = _body(result)
        assert body["company"] == "Acme Inc"

    def test_register_missing_name(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(
            method="POST",
            body={"email": "partner@example.com"},
        )

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/register", {}, mock_h)

        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    def test_register_missing_email(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(
            method="POST",
            body={"name": "Test Partner"},
        )

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/register", {}, mock_h)

        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    def test_register_empty_body(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(
            method="POST",
            body={},
        )

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/register", {}, mock_h)

        assert _status(result) == 400

    def test_register_duplicate_email(self, handler, auth_ctx):
        existing = MockPartner()
        mock_api = MagicMock()
        mock_api._store.get_partner_by_email.return_value = existing

        mock_h = MockHTTPHandler(
            method="POST",
            body={"name": "Test", "email": "partner@example.com"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/register", {}, mock_h)

        assert _status(result) == 409
        assert "already registered" in _body(result)["error"].lower()

    def test_register_wrong_content_type(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(method="POST", body={"name": "Test", "email": "t@t.com"})
        mock_h.headers["Content-Type"] = "text/plain"

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/register", {}, mock_h)

        assert _status(result) == 415
        assert "application/json" in _body(result)["error"]

    def test_register_body_too_large(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(method="POST", body={"name": "Test", "email": "t@t.com"})
        mock_h.headers["Content-Length"] = str(11 * 1024 * 1024)  # 11MB

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/register", {}, mock_h)

        assert _status(result) == 413
        assert "too large" in _body(result)["error"].lower()

    def test_register_invalid_json(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(method="POST")
        mock_h.headers["Content-Type"] = "application/json"
        mock_h.headers["Content-Length"] = "10"
        mock_h.rfile.read.return_value = b"not json!!"

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/register", {}, mock_h)

        assert _status(result) == 400
        assert "invalid" in _body(result)["error"].lower()

    def test_register_import_error(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(
            method="POST",
            body={"name": "Test", "email": "t@t.com"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch(
                "aragora.billing.partner.get_partner_api",
                side_effect=ImportError("no partner module"),
            ),
        ):
            result = handler.handle("/api/v1/partners/register", {}, mock_h)

        assert _status(result) == 500


# ---------------------------------------------------------------------------
# GET /api/v1/partners/me
# ---------------------------------------------------------------------------


class TestGetPartnerProfile:
    """Tests for getting partner profile."""

    def test_get_profile_success(self, handler, auth_ctx):
        stats = {"partner_id": "partner-001", "total_revenue": 1000, "api_calls": 500}
        mock_api = MagicMock()
        mock_api.get_partner_stats.return_value = stats

        mock_h = MockHTTPHandler(
            method="GET",
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/me", {}, mock_h)

        assert _status(result) == 200
        body = _body(result)
        assert body["partner_id"] == "partner-001"
        assert body["total_revenue"] == 1000

    def test_get_profile_missing_partner_id(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(method="GET")

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/me", {}, mock_h)

        assert _status(result) == 400
        assert "partner id" in _body(result)["error"].lower()

    def test_get_profile_not_found(self, handler, auth_ctx):
        mock_api = MagicMock()
        mock_api.get_partner_stats.side_effect = ValueError("Partner not found")

        mock_h = MockHTTPHandler(
            method="GET",
            headers={"X-Partner-ID": "nonexistent"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/me", {}, mock_h)

        assert _status(result) == 404

    def test_get_profile_unauthenticated(self, handler, unauth_ctx):
        mock_h = MockHTTPHandler(
            method="GET",
            headers={"X-Partner-ID": "partner-001"},
        )

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=unauth_ctx,
        ):
            result = handler.handle("/api/v1/partners/me", {}, mock_h)

        assert _status(result) == 401

    def test_get_profile_internal_error(self, handler, auth_ctx):
        mock_api = MagicMock()
        mock_api.get_partner_stats.side_effect = RuntimeError("db down")

        mock_h = MockHTTPHandler(
            method="GET",
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/me", {}, mock_h)

        assert _status(result) == 500


# ---------------------------------------------------------------------------
# POST /api/v1/partners/keys - Create API key
# ---------------------------------------------------------------------------


class TestCreateAPIKey:
    """Tests for API key creation endpoint."""

    def test_create_key_success(self, handler, auth_ctx):
        api_key = MockAPIKey()
        raw_key = "ara_test_1234567890abcdef"
        mock_api = MagicMock()
        mock_api.create_api_key.return_value = (api_key, raw_key)

        mock_h = MockHTTPHandler(
            method="POST",
            body={"name": "My Key", "scopes": ["debates:read"]},
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/keys", {}, mock_h)

        assert _status(result) == 201
        body = _body(result)
        assert body["key_id"] == "key-001"
        assert body["key"] == raw_key
        assert body["key_prefix"] == "ara_test"
        assert body["name"] == "My API Key"
        assert body["scopes"] == ["debates:read"]
        assert "warning" in body

    def test_create_key_default_name(self, handler, auth_ctx):
        api_key = MockAPIKey(name="API Key")
        mock_api = MagicMock()
        mock_api.create_api_key.return_value = (api_key, "ara_raw")

        mock_h = MockHTTPHandler(
            method="POST",
            body={},
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/keys", {}, mock_h)

        assert _status(result) == 201
        # Verify that "API Key" is the default name passed through
        mock_api.create_api_key.assert_called_once_with(
            partner_id="partner-001",
            name="API Key",
            scopes=None,
            expires_in_days=None,
        )

    def test_create_key_with_expiration(self, handler, auth_ctx):
        api_key = MockAPIKey(
            expires_at=datetime(2027, 1, 1, tzinfo=timezone.utc),
        )
        mock_api = MagicMock()
        mock_api.create_api_key.return_value = (api_key, "ara_raw")

        mock_h = MockHTTPHandler(
            method="POST",
            body={"name": "Expiring Key", "expires_in_days": 365},
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/keys", {}, mock_h)

        assert _status(result) == 201
        body = _body(result)
        assert body["expires_at"] is not None

    def test_create_key_no_expiration(self, handler, auth_ctx):
        api_key = MockAPIKey(expires_at=None)
        mock_api = MagicMock()
        mock_api.create_api_key.return_value = (api_key, "ara_raw")

        mock_h = MockHTTPHandler(
            method="POST",
            body={"name": "Permanent Key"},
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/keys", {}, mock_h)

        assert _status(result) == 201
        body = _body(result)
        assert body["expires_at"] is None

    def test_create_key_missing_partner_id(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(
            method="POST",
            body={"name": "Key"},
        )

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/keys", {}, mock_h)

        assert _status(result) == 400
        assert "partner id" in _body(result)["error"].lower()

    def test_create_key_wrong_content_type(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(method="POST", body={"name": "Key"})
        mock_h.headers["Content-Type"] = "text/xml"
        mock_h.headers["X-Partner-ID"] = "partner-001"

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/keys", {}, mock_h)

        assert _status(result) == 415

    def test_create_key_body_too_large(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(method="POST", body={"name": "Key"})
        mock_h.headers["Content-Length"] = str(11 * 1024 * 1024)
        mock_h.headers["X-Partner-ID"] = "partner-001"

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/keys", {}, mock_h)

        assert _status(result) == 413

    def test_create_key_invalid_json(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(method="POST")
        mock_h.headers["Content-Type"] = "application/json"
        mock_h.headers["Content-Length"] = "5"
        mock_h.headers["X-Partner-ID"] = "partner-001"
        mock_h.rfile.read.return_value = b"bad!!"

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/keys", {}, mock_h)

        assert _status(result) == 400

    def test_create_key_value_error(self, handler, auth_ctx):
        mock_api = MagicMock()
        mock_api.create_api_key.side_effect = ValueError("invalid scopes")

        mock_h = MockHTTPHandler(
            method="POST",
            body={"name": "Key", "scopes": ["invalid:scope"]},
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/keys", {}, mock_h)

        assert _status(result) == 400

    def test_create_key_internal_error(self, handler, auth_ctx):
        mock_api = MagicMock()
        mock_api.create_api_key.side_effect = RuntimeError("db error")

        mock_h = MockHTTPHandler(
            method="POST",
            body={"name": "Key"},
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/keys", {}, mock_h)

        assert _status(result) == 500


# ---------------------------------------------------------------------------
# GET /api/v1/partners/keys - List API keys
# ---------------------------------------------------------------------------


class TestListAPIKeys:
    """Tests for listing API keys."""

    def test_list_keys_success(self, handler, auth_ctx):
        keys = [
            MockAPIKey(key_id="key-001", is_active=True),
            MockAPIKey(key_id="key-002", is_active=True),
            MockAPIKey(key_id="key-003", is_active=False),
        ]
        mock_api = MagicMock()
        mock_api._store.list_partner_keys.return_value = keys

        mock_h = MockHTTPHandler(
            method="GET",
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/keys", {}, mock_h)

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 3
        assert body["active"] == 2
        assert len(body["keys"]) == 3

    def test_list_keys_empty(self, handler, auth_ctx):
        mock_api = MagicMock()
        mock_api._store.list_partner_keys.return_value = []

        mock_h = MockHTTPHandler(
            method="GET",
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/keys", {}, mock_h)

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 0
        assert body["active"] == 0
        assert body["keys"] == []

    def test_list_keys_missing_partner_id(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(method="GET")

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/keys", {}, mock_h)

        assert _status(result) == 400

    def test_list_keys_internal_error(self, handler, auth_ctx):
        mock_api = MagicMock()
        mock_api._store.list_partner_keys.side_effect = RuntimeError("store error")

        mock_h = MockHTTPHandler(
            method="GET",
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/keys", {}, mock_h)

        assert _status(result) == 500


# ---------------------------------------------------------------------------
# POST /api/v1/partners/keys/{key_id}/rotate
# ---------------------------------------------------------------------------


class TestRotateAPIKey:
    """Tests for API key rotation endpoint."""

    def test_rotate_key_success(self, handler, auth_ctx):
        new_key = MockAPIKey(key_id="key-002", name="Rotated Key")
        raw_key = "ara_new_rotated_key"
        mock_api = MagicMock()
        mock_api.rotate_api_key.return_value = (new_key, raw_key)

        mock_h = MockHTTPHandler(
            method="POST",
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/keys/key-001/rotate", {}, mock_h)

        assert _status(result) == 201
        body = _body(result)
        assert body["key_id"] == "key-002"
        assert body["key"] == raw_key
        assert body["rotated_from"] == "key-001"
        assert "warning" in body

    def test_rotate_key_missing_partner_id(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(method="POST")

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/keys/key-001/rotate", {}, mock_h)

        assert _status(result) == 400

    def test_rotate_key_not_found(self, handler, auth_ctx):
        mock_api = MagicMock()
        mock_api.rotate_api_key.side_effect = ValueError("Key not found")

        mock_h = MockHTTPHandler(
            method="POST",
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/keys/key-999/rotate", {}, mock_h)

        assert _status(result) == 400

    def test_rotate_key_internal_error(self, handler, auth_ctx):
        mock_api = MagicMock()
        mock_api.rotate_api_key.side_effect = RuntimeError("db error")

        mock_h = MockHTTPHandler(
            method="POST",
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/keys/key-001/rotate", {}, mock_h)

        assert _status(result) == 500


# ---------------------------------------------------------------------------
# DELETE /api/v1/partners/keys/{key_id}
# ---------------------------------------------------------------------------


class TestRevokeAPIKey:
    """Tests for API key revocation endpoint."""

    def test_revoke_key_success(self, handler, auth_ctx):
        mock_api = MagicMock()
        mock_api._store.revoke_api_key.return_value = True

        mock_h = MockHTTPHandler(
            method="DELETE",
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/keys/key-001", {}, mock_h)

        assert _status(result) == 200
        body = _body(result)
        assert body["key_id"] == "key-001"
        assert "revoked" in body["message"].lower()

    def test_revoke_key_not_found(self, handler, auth_ctx):
        mock_api = MagicMock()
        mock_api._store.revoke_api_key.return_value = False

        mock_h = MockHTTPHandler(
            method="DELETE",
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/keys/key-999", {}, mock_h)

        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()

    def test_revoke_key_missing_partner_id(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(method="DELETE")

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/keys/key-001", {}, mock_h)

        assert _status(result) == 400

    def test_revoke_key_internal_error(self, handler, auth_ctx):
        mock_api = MagicMock()
        mock_api._store.revoke_api_key.side_effect = RuntimeError("store error")

        mock_h = MockHTTPHandler(
            method="DELETE",
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/keys/key-001", {}, mock_h)

        assert _status(result) == 500


# ---------------------------------------------------------------------------
# GET /api/v1/partners/usage
# ---------------------------------------------------------------------------


class TestGetUsage:
    """Tests for usage statistics endpoint."""

    def test_get_usage_success(self, handler, auth_ctx):
        stats = {"api_calls": 500, "debates_run": 20, "revenue": 1200}
        mock_api = MagicMock()
        mock_api.get_partner_stats.return_value = stats

        mock_h = MockHTTPHandler(
            method="GET",
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/usage", {}, mock_h)

        assert _status(result) == 200
        body = _body(result)
        assert body["api_calls"] == 500
        assert body["debates_run"] == 20

    def test_get_usage_with_days_param(self, handler, auth_ctx):
        stats = {"api_calls": 100}
        mock_api = MagicMock()
        mock_api.get_partner_stats.return_value = stats

        mock_h = MockHTTPHandler(
            method="GET",
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/usage", {"days": "7"}, mock_h)

        assert _status(result) == 200
        # Verify days param was passed through
        mock_api.get_partner_stats.assert_called_once_with("partner-001", days=7)

    def test_get_usage_default_days(self, handler, auth_ctx):
        stats = {"api_calls": 100}
        mock_api = MagicMock()
        mock_api.get_partner_stats.return_value = stats

        mock_h = MockHTTPHandler(
            method="GET",
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/usage", {}, mock_h)

        assert _status(result) == 200
        # Default days = 30
        mock_api.get_partner_stats.assert_called_once_with("partner-001", days=30)

    def test_get_usage_missing_partner_id(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(method="GET")

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/usage", {}, mock_h)

        assert _status(result) == 400

    def test_get_usage_partner_not_found(self, handler, auth_ctx):
        mock_api = MagicMock()
        mock_api.get_partner_stats.side_effect = ValueError("Partner not found")

        mock_h = MockHTTPHandler(
            method="GET",
            headers={"X-Partner-ID": "nonexistent"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/usage", {}, mock_h)

        assert _status(result) == 404

    def test_get_usage_internal_error(self, handler, auth_ctx):
        mock_api = MagicMock()
        mock_api.get_partner_stats.side_effect = RuntimeError("db error")

        mock_h = MockHTTPHandler(
            method="GET",
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/usage", {}, mock_h)

        assert _status(result) == 500


# ---------------------------------------------------------------------------
# POST /api/v1/partners/webhooks
# ---------------------------------------------------------------------------


class TestConfigureWebhook:
    """Tests for webhook configuration endpoint."""

    def test_configure_webhook_success(self, handler, auth_ctx):
        partner = MockPartner(partner_id="partner-001")
        mock_api = MagicMock()
        mock_api._store.get_partner.return_value = partner
        mock_api.generate_webhook_secret.return_value = "whsec_test_secret"

        mock_h = MockHTTPHandler(
            method="POST",
            body={"url": "https://example.com/webhook"},
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
            patch(
                "aragora.server.handlers.partner.validate_webhook_url",
                return_value=(True, None),
            ),
        ):
            result = handler.handle("/api/v1/partners/webhooks", {}, mock_h)

        assert _status(result) == 200
        body = _body(result)
        assert body["webhook_url"] == "https://example.com/webhook"
        assert body["webhook_secret"] == "whsec_test_secret"
        assert "warning" in body

    def test_configure_webhook_missing_url(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(
            method="POST",
            body={},
            headers={"X-Partner-ID": "partner-001"},
        )

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/webhooks", {}, mock_h)

        assert _status(result) == 400
        assert "url" in _body(result)["error"].lower()

    def test_configure_webhook_http_url_rejected(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(
            method="POST",
            body={"url": "http://example.com/webhook"},
            headers={"X-Partner-ID": "partner-001"},
        )

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/webhooks", {}, mock_h)

        assert _status(result) == 400
        assert "https" in _body(result)["error"].lower()

    def test_configure_webhook_ssrf_blocked(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(
            method="POST",
            body={"url": "https://169.254.169.254/metadata"},
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch(
                "aragora.server.handlers.partner.validate_webhook_url",
                return_value=(False, "blocked IP"),
            ),
        ):
            result = handler.handle("/api/v1/partners/webhooks", {}, mock_h)

        assert _status(result) == 400
        assert "invalid webhook url" in _body(result)["error"].lower()

    def test_configure_webhook_partner_not_found(self, handler, auth_ctx):
        mock_api = MagicMock()
        mock_api._store.get_partner.return_value = None

        mock_h = MockHTTPHandler(
            method="POST",
            body={"url": "https://example.com/webhook"},
            headers={"X-Partner-ID": "nonexistent"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
            patch(
                "aragora.server.handlers.partner.validate_webhook_url",
                return_value=(True, None),
            ),
        ):
            result = handler.handle("/api/v1/partners/webhooks", {}, mock_h)

        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()

    def test_configure_webhook_missing_partner_id(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(
            method="POST",
            body={"url": "https://example.com/webhook"},
        )

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/webhooks", {}, mock_h)

        assert _status(result) == 400

    def test_configure_webhook_wrong_content_type(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(
            method="POST",
            body={"url": "https://example.com/webhook"},
        )
        mock_h.headers["Content-Type"] = "text/plain"
        mock_h.headers["X-Partner-ID"] = "partner-001"

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/webhooks", {}, mock_h)

        assert _status(result) == 415

    def test_configure_webhook_internal_error(self, handler, auth_ctx):
        mock_api = MagicMock()
        mock_api._store.get_partner.side_effect = RuntimeError("store error")

        mock_h = MockHTTPHandler(
            method="POST",
            body={"url": "https://example.com/webhook"},
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
            patch(
                "aragora.server.handlers.partner.validate_webhook_url",
                return_value=(True, None),
            ),
        ):
            result = handler.handle("/api/v1/partners/webhooks", {}, mock_h)

        assert _status(result) == 500


# ---------------------------------------------------------------------------
# GET /api/v1/partners/limits
# ---------------------------------------------------------------------------


class TestGetLimits:
    """Tests for rate limits endpoint."""

    def test_get_limits_success(self, handler, auth_ctx):
        partner = MockPartner(tier=MockPartnerTier.STARTER)
        limits = MockTierLimits()
        mock_api = MagicMock()
        mock_api._store.get_partner.return_value = partner
        mock_api.check_rate_limit.return_value = (True, {"requests_today": 50})

        tier_limits = {partner.tier: limits}

        mock_h = MockHTTPHandler(
            method="GET",
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
            patch("aragora.billing.partner.PARTNER_TIER_LIMITS", tier_limits),
        ):
            result = handler.handle("/api/v1/partners/limits", {}, mock_h)

        assert _status(result) == 200
        body = _body(result)
        assert body["tier"] == "starter"
        assert body["limits"]["requests_per_minute"] == 60
        assert body["limits"]["requests_per_day"] == 10000
        assert body["limits"]["debates_per_month"] == 100
        assert body["limits"]["max_agents_per_debate"] == 10
        assert body["limits"]["max_rounds"] == 5
        assert body["limits"]["webhook_endpoints"] == 3
        assert body["limits"]["revenue_share_percent"] == 20.0
        assert body["allowed"] is True
        assert body["current"]["requests_today"] == 50

    def test_get_limits_partner_not_found(self, handler, auth_ctx):
        mock_api = MagicMock()
        mock_api._store.get_partner.return_value = None

        mock_h = MockHTTPHandler(
            method="GET",
            headers={"X-Partner-ID": "nonexistent"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/limits", {}, mock_h)

        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()

    def test_get_limits_missing_partner_id(self, handler, auth_ctx):
        mock_h = MockHTTPHandler(method="GET")

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/limits", {}, mock_h)

        assert _status(result) == 400

    def test_get_limits_internal_error(self, handler, auth_ctx):
        mock_api = MagicMock()
        mock_api._store.get_partner.side_effect = RuntimeError("store error")

        mock_h = MockHTTPHandler(
            method="GET",
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/limits", {}, mock_h)

        assert _status(result) == 500


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------


class TestAuthentication:
    """Tests for authentication behavior across endpoints."""

    @pytest.mark.parametrize(
        "path,method",
        [
            ("/api/v1/partners/me", "GET"),
            ("/api/v1/partners/keys", "POST"),
            ("/api/v1/partners/keys", "GET"),
            ("/api/v1/partners/keys/key-001/rotate", "POST"),
            ("/api/v1/partners/keys/key-001", "DELETE"),
            ("/api/v1/partners/usage", "GET"),
            ("/api/v1/partners/webhooks", "POST"),
            ("/api/v1/partners/limits", "GET"),
        ],
    )
    def test_unauthenticated_returns_401(self, handler, unauth_ctx, path, method):
        """All @require_user_auth endpoints reject unauthenticated users."""
        body = {"name": "test", "email": "t@t.com", "url": "https://ex.com/wh"} if method == "POST" else None
        headers = {"X-Partner-ID": "partner-001"}
        mock_h = MockHTTPHandler(method=method, body=body, headers=headers)

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=unauth_ctx,
        ):
            result = handler.handle(path, {}, mock_h)

        assert _status(result) == 401


# ---------------------------------------------------------------------------
# Path parsing / key ID extraction
# ---------------------------------------------------------------------------


class TestPathParsing:
    """Tests for dynamic path segment extraction."""

    def test_rotate_extracts_key_id(self, handler, auth_ctx):
        """The rotate path /api/v1/partners/keys/{key_id}/rotate extracts key_id from parts[-2]."""
        new_key = MockAPIKey(key_id="key-new")
        mock_api = MagicMock()
        mock_api.rotate_api_key.return_value = (new_key, "ara_raw")

        mock_h = MockHTTPHandler(
            method="POST",
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/keys/my-special-key/rotate", {}, mock_h)

        assert _status(result) == 201
        # Verify the correct key_id was extracted and passed
        mock_api.rotate_api_key.assert_called_once_with(
            partner_id="partner-001",
            key_id="my-special-key",
        )

    def test_delete_extracts_key_id(self, handler, auth_ctx):
        """The delete path /api/v1/partners/keys/{key_id} extracts key_id from parts[-1]."""
        mock_api = MagicMock()
        mock_api._store.revoke_api_key.return_value = True

        mock_h = MockHTTPHandler(
            method="DELETE",
            headers={"X-Partner-ID": "partner-001"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch("aragora.billing.partner.get_partner_api", return_value=mock_api),
        ):
            result = handler.handle("/api/v1/partners/keys/uuid-key-456", {}, mock_h)

        assert _status(result) == 200
        mock_api._store.revoke_api_key.assert_called_once_with("uuid-key-456")

    def test_rotate_wrong_method_returns_none(self, handler, auth_ctx):
        """GET on the rotate path does not match."""
        mock_h = MockHTTPHandler(
            method="GET",
            headers={"X-Partner-ID": "partner-001"},
        )

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/keys/key-001/rotate", {}, mock_h)

        assert result is None

    def test_extra_segments_after_rotate_returns_none(self, handler, auth_ctx):
        """Path with extra segments after rotate is not matched (len(parts) != 7)."""
        mock_h = MockHTTPHandler(
            method="POST",
            headers={"X-Partner-ID": "partner-001"},
        )

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/keys/key-001/rotate/extra", {}, mock_h)

        assert result is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_register_no_content_type_header(self, handler, auth_ctx):
        """Missing Content-Type header triggers 415."""
        mock_h = MockHTTPHandler(method="POST", body={"name": "Test", "email": "t@t.com"})
        # Remove Content-Type header
        del mock_h.headers["Content-Type"]

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/register", {}, mock_h)

        # The handler checks (handler.headers.get("Content-Type") or "").lower()
        # which returns "" when missing, and "" does not startswith "application/json"
        assert _status(result) == 415

    def test_register_partner_api_import_error(self, handler, auth_ctx):
        """ImportError from get_partner_api returns 500."""
        mock_h = MockHTTPHandler(
            method="POST",
            body={"name": "Test", "email": "t@t.com"},
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch(
                "aragora.billing.partner.get_partner_api",
                side_effect=ImportError("no module"),
            ),
        ):
            result = handler.handle("/api/v1/partners/register", {}, mock_h)

        assert _status(result) == 500

    def test_webhook_url_with_ftp_scheme(self, handler, auth_ctx):
        """FTP URL is rejected since it doesn't start with https://."""
        mock_h = MockHTTPHandler(
            method="POST",
            body={"url": "ftp://example.com/webhook"},
            headers={"X-Partner-ID": "partner-001"},
        )

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/webhooks", {}, mock_h)

        assert _status(result) == 400

    def test_create_key_unauthenticated(self, handler, unauth_ctx):
        mock_h = MockHTTPHandler(
            method="POST",
            body={"name": "Key"},
            headers={"X-Partner-ID": "partner-001"},
        )

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=unauth_ctx,
        ):
            result = handler.handle("/api/v1/partners/keys", {}, mock_h)

        assert _status(result) == 401

    def test_handler_routes_class_attribute(self, handler):
        """ROUTES contains all expected static routes."""
        expected = [
            "/api/v1/partners/register",
            "/api/v1/partners/me",
            "/api/v1/partners/keys",
            "/api/v1/partners/usage",
            "/api/v1/partners/webhooks",
            "/api/v1/partners/limits",
        ]
        for route in expected:
            assert route in handler.ROUTES

    def test_register_empty_name_string(self, handler, auth_ctx):
        """Empty string name should be treated as missing."""
        mock_h = MockHTTPHandler(
            method="POST",
            body={"name": "", "email": "t@t.com"},
        )

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/register", {}, mock_h)

        assert _status(result) == 400

    def test_register_empty_email_string(self, handler, auth_ctx):
        """Empty string email should be treated as missing."""
        mock_h = MockHTTPHandler(
            method="POST",
            body={"name": "Test", "email": ""},
        )

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = handler.handle("/api/v1/partners/register", {}, mock_h)

        assert _status(result) == 400
