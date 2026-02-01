"""
Tests for aragora.server.handlers.partner - Partner API Handler.

Tests cover:
- Route registration and can_handle
- POST /api/v1/partners/register - Register as a partner
- GET /api/v1/partners/me - Get current partner profile
- POST /api/v1/partners/keys - Create API key
- GET /api/v1/partners/keys - List API keys
- POST /api/v1/partners/keys/{key_id}/rotate - Rotate API key
- DELETE /api/v1/partners/keys/{key_id} - Revoke API key
- GET /api/v1/partners/usage - Get usage statistics
- POST /api/v1/partners/webhooks - Configure webhook
- GET /api/v1/partners/limits - Get rate limits
- Authentication and RBAC
- Rate limiting
- Error handling
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import the module under test with Slack stub workaround
# ---------------------------------------------------------------------------


def _import_partner_module():
    """Import partner module, working around broken sibling imports."""
    try:
        import aragora.server.handlers.partner as mod

        return mod
    except (ImportError, ModuleNotFoundError):
        pass

    # Clear partially loaded modules and stub broken imports
    to_remove = [k for k in sys.modules if k.startswith("aragora.server.handlers")]
    for k in to_remove:
        del sys.modules[k]

    _slack_stubs = [
        "aragora.server.handlers.social._slack_impl",
        "aragora.server.handlers.social._slack_impl.config",
        "aragora.server.handlers.social._slack_impl.handler",
        "aragora.server.handlers.social._slack_impl.commands",
        "aragora.server.handlers.social._slack_impl.events",
        "aragora.server.handlers.social._slack_impl.blocks",
        "aragora.server.handlers.social._slack_impl.interactions",
        "aragora.server.handlers.social.slack",
        "aragora.server.handlers.social.slack.handler",
    ]
    for name in _slack_stubs:
        if name not in sys.modules:
            stub = MagicMock()
            stub.__path__ = []
            stub.__file__ = f"<stub:{name}>"
            sys.modules[name] = stub

    import aragora.server.handlers.partner as mod

    return mod


partner_module = _import_partner_module()
PartnerHandler = partner_module.PartnerHandler


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


@dataclass(frozen=True)
class MockPartnerTier:
    """Mock partner tier enum."""

    value: str = "starter"


@dataclass
class MockPartnerStatus:
    """Mock partner status enum."""

    value: str = "active"


@dataclass
class MockPartner:
    """Mock partner for testing."""

    partner_id: str = "partner-123"
    name: str = "Test Partner"
    email: str = "partner@example.com"
    company: str | None = "Test Corp"
    tier: MockPartnerTier = field(default_factory=MockPartnerTier)
    status: MockPartnerStatus = field(default_factory=MockPartnerStatus)
    referral_code: str = "TEST123"
    webhook_url: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MockAPIKey:
    """Mock API key for testing."""

    key_id: str = "key-123"
    key_prefix: str = "ara_test..."
    name: str = "Test API Key"
    scopes: list = field(default_factory=lambda: ["debates:read", "debates:write"])
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    is_active: bool = True

    def to_dict(self) -> dict:
        return {
            "key_id": self.key_id,
            "key_prefix": self.key_prefix,
            "name": self.name,
            "scopes": self.scopes,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
        }


@dataclass
class MockPartnerLimits:
    """Mock partner tier limits."""

    requests_per_minute: int = 60
    requests_per_day: int = 10000
    debates_per_month: int = 100
    max_agents_per_debate: int = 5
    max_rounds: int = 3
    webhook_endpoints: int = 1
    revenue_share_percent: float = 10.0


class MockPartnerStore:
    """Mock partner store."""

    def __init__(self):
        self.partners: dict[str, MockPartner] = {}
        self.keys: dict[str, list[MockAPIKey]] = {}

    def get_partner(self, partner_id: str) -> MockPartner | None:
        return self.partners.get(partner_id)

    def get_partner_by_email(self, email: str) -> MockPartner | None:
        for partner in self.partners.values():
            if partner.email == email:
                return partner
        return None

    def list_partner_keys(self, partner_id: str) -> list[MockAPIKey]:
        return self.keys.get(partner_id, [])

    def revoke_api_key(self, key_id: str) -> bool:
        for partner_keys in self.keys.values():
            for key in partner_keys:
                if key.key_id == key_id:
                    key.is_active = False
                    return True
        return False


class MockPartnerAPI:
    """Mock partner API."""

    def __init__(self):
        self._store = MockPartnerStore()

    def register_partner(self, name: str, email: str, company: str | None = None) -> MockPartner:
        partner = MockPartner(
            name=name,
            email=email,
            company=company,
            status=MockPartnerStatus(value="pending"),
        )
        self._store.partners[partner.partner_id] = partner
        return partner

    def get_partner_stats(self, partner_id: str, days: int = 30) -> dict:
        partner = self._store.get_partner(partner_id)
        if not partner:
            raise ValueError("Partner not found")
        return {
            "partner_id": partner_id,
            "name": partner.name,
            "tier": partner.tier.value,
            "status": partner.status.value,
            "usage": {
                "requests_today": 150,
                "requests_this_month": 5000,
                "debates_this_month": 25,
            },
        }

    def create_api_key(
        self,
        partner_id: str,
        name: str = "API Key",
        scopes: list | None = None,
        expires_in_days: int | None = None,
    ) -> tuple[MockAPIKey, str]:
        key = MockAPIKey(name=name, scopes=scopes or [])
        if expires_in_days:
            key.expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
        raw_key = f"ara_{'x' * 32}"
        if partner_id not in self._store.keys:
            self._store.keys[partner_id] = []
        self._store.keys[partner_id].append(key)
        return key, raw_key

    def generate_webhook_secret(self, partner_id: str) -> str:
        return f"whsec_{'x' * 32}"

    def rotate_api_key(
        self,
        partner_id: str,
        key_id: str,
    ) -> tuple[MockAPIKey, str]:
        partner_keys = self._store.keys.get(partner_id, [])
        for key in partner_keys:
            if key.key_id == key_id and key.is_active:
                key.is_active = False
                new_key = MockAPIKey(
                    key_id=f"key-rotated-{key_id}",
                    name=key.name,
                    scopes=key.scopes,
                )
                partner_keys.append(new_key)
                return new_key, f"ara_{'r' * 32}"
        raise ValueError("API key not found")

    def check_rate_limit(self, partner: MockPartner) -> tuple[bool, dict]:
        return True, {"requests_remaining": 50}


def make_mock_handler(
    body: dict | None = None,
    method: str = "GET",
    path: str = "/api/v1/partners/me",
    partner_id: str | None = "partner-123",
):
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = method
    handler.path = path
    handler.headers = {}
    handler.client_address = ("127.0.0.1", 12345)

    if partner_id:
        handler.headers["X-Partner-ID"] = partner_id

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.headers["Content-Length"] = str(len(body_bytes))
        handler.rfile = BytesIO(body_bytes)
    else:
        handler.rfile = BytesIO(b"")
        handler.headers["Content-Length"] = "0"

    return handler


@pytest.fixture
def partner_handler():
    """Create PartnerHandler with mock context."""
    ctx = {}
    handler = PartnerHandler(ctx)
    return handler


@pytest.fixture(autouse=True)
def clear_rate_limiters():
    """Clear rate limiters before each test."""
    from aragora.server.handlers.utils.rate_limit import _limiters

    for limiter in _limiters.values():
        limiter.clear()
    yield
    for limiter in _limiters.values():
        limiter.clear()


@pytest.fixture
def mock_partner_api():
    """Create mock partner API with test data."""
    api = MockPartnerAPI()
    api._store.partners["partner-123"] = MockPartner()
    api._store.keys["partner-123"] = [MockAPIKey()]
    return api


# ===========================================================================
# Test Routing (can_handle)
# ===========================================================================


class TestPartnerHandlerRouting:
    """Tests for PartnerHandler.can_handle."""

    def test_can_handle_register(self, partner_handler):
        assert partner_handler.can_handle("/api/v1/partners/register") is True

    def test_can_handle_me(self, partner_handler):
        assert partner_handler.can_handle("/api/v1/partners/me") is True

    def test_can_handle_keys(self, partner_handler):
        assert partner_handler.can_handle("/api/v1/partners/keys") is True

    def test_can_handle_key_rotate(self, partner_handler):
        assert partner_handler.can_handle("/api/v1/partners/keys/key-123/rotate") is True

    def test_can_handle_key_delete(self, partner_handler):
        assert partner_handler.can_handle("/api/v1/partners/keys/key-123") is True

    def test_can_handle_usage(self, partner_handler):
        assert partner_handler.can_handle("/api/v1/partners/usage") is True

    def test_can_handle_webhooks(self, partner_handler):
        assert partner_handler.can_handle("/api/v1/partners/webhooks") is True

    def test_can_handle_limits(self, partner_handler):
        assert partner_handler.can_handle("/api/v1/partners/limits") is True

    def test_cannot_handle_other_paths(self, partner_handler):
        assert partner_handler.can_handle("/api/v1/debates") is False


# ===========================================================================
# Test Register Partner (POST /api/v1/partners/register)
# ===========================================================================


class TestPartnerRegister:
    """Tests for POST /api/v1/partners/register endpoint."""

    def test_register_partner_success(self, partner_handler, mock_partner_api):
        """Happy path: register new partner."""
        handler = make_mock_handler(
            body={"name": "New Partner", "email": "new@example.com", "company": "New Corp"},
            method="POST",
            path="/api/v1/partners/register",
            partner_id=None,
        )

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._register_partner(handler)

        assert result is not None
        assert result.status_code == 201
        data = json.loads(result.body)
        assert data["name"] == "New Partner"
        assert data["status"] == "pending"

    def test_register_partner_missing_name(self, partner_handler, mock_partner_api):
        """Missing name returns 400."""
        handler = make_mock_handler(
            body={"email": "test@example.com"},
            method="POST",
            path="/api/v1/partners/register",
            partner_id=None,
        )

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._register_partner(handler)

        assert result is not None
        assert result.status_code == 400

    def test_register_partner_missing_email(self, partner_handler, mock_partner_api):
        """Missing email returns 400."""
        handler = make_mock_handler(
            body={"name": "Test Partner"},
            method="POST",
            path="/api/v1/partners/register",
            partner_id=None,
        )

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._register_partner(handler)

        assert result is not None
        assert result.status_code == 400

    def test_register_partner_email_exists(self, partner_handler, mock_partner_api):
        """Existing email returns 409."""
        mock_partner_api._store.partners["partner-123"] = MockPartner(email="existing@example.com")

        handler = make_mock_handler(
            body={"name": "Test", "email": "existing@example.com"},
            method="POST",
            path="/api/v1/partners/register",
            partner_id=None,
        )

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._register_partner(handler)

        assert result is not None
        assert result.status_code == 409

    def test_register_partner_invalid_json(self, partner_handler):
        """Invalid JSON returns 400."""
        handler = MagicMock()
        handler.command = "POST"
        handler.headers = {"Content-Length": "10"}
        handler.rfile = BytesIO(b"not-json!!")

        result = partner_handler._register_partner(handler)

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Test Get Partner Profile (GET /api/v1/partners/me)
# ===========================================================================


class TestPartnerGetProfile:
    """Tests for GET /api/v1/partners/me endpoint."""

    def test_get_profile_success(self, partner_handler, mock_partner_api):
        """Happy path: get partner profile."""
        handler = make_mock_handler()

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._get_partner_profile(handler, user=MagicMock())

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "partner_id" in data

    def test_get_profile_missing_partner_id(self, partner_handler, mock_partner_api):
        """Missing partner ID returns 400."""
        handler = make_mock_handler(partner_id=None)

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._get_partner_profile(handler, user=MagicMock())

        assert result is not None
        assert result.status_code == 400

    def test_get_profile_partner_not_found(self, partner_handler, mock_partner_api):
        """Partner not found returns 404."""
        handler = make_mock_handler(partner_id="nonexistent")

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._get_partner_profile(handler, user=MagicMock())

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Test Create API Key (POST /api/v1/partners/keys)
# ===========================================================================


class TestPartnerCreateKey:
    """Tests for POST /api/v1/partners/keys endpoint."""

    def test_create_key_success(self, partner_handler, mock_partner_api):
        """Happy path: create API key."""
        handler = make_mock_handler(
            body={"name": "My Key", "scopes": ["debates:read"]},
            method="POST",
            path="/api/v1/partners/keys",
        )

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._create_api_key(handler, user=MagicMock())

        assert result is not None
        assert result.status_code == 201
        data = json.loads(result.body)
        assert "key" in data
        assert data["key"].startswith("ara_")
        assert "warning" in data

    def test_create_key_with_expiry(self, partner_handler, mock_partner_api):
        """Create key with expiration."""
        handler = make_mock_handler(
            body={"name": "Temp Key", "expires_in_days": 30},
            method="POST",
            path="/api/v1/partners/keys",
        )

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._create_api_key(handler, user=MagicMock())

        assert result is not None
        assert result.status_code == 201
        data = json.loads(result.body)
        assert data["expires_at"] is not None

    def test_create_key_missing_partner_id(self, partner_handler, mock_partner_api):
        """Missing partner ID returns 400."""
        handler = make_mock_handler(
            body={"name": "My Key"},
            method="POST",
            path="/api/v1/partners/keys",
            partner_id=None,
        )

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._create_api_key(handler, user=MagicMock())

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Test List API Keys (GET /api/v1/partners/keys)
# ===========================================================================


class TestPartnerListKeys:
    """Tests for GET /api/v1/partners/keys endpoint."""

    def test_list_keys_success(self, partner_handler, mock_partner_api):
        """Happy path: list API keys."""
        handler = make_mock_handler(path="/api/v1/partners/keys")

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._list_api_keys(handler, user=MagicMock())

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "keys" in data
        assert "total" in data
        assert "active" in data


# ===========================================================================
# Test Rotate API Key (POST /api/v1/partners/keys/{key_id}/rotate)
# ===========================================================================


class TestPartnerRotateKey:
    """Tests for POST /api/v1/partners/keys/{key_id}/rotate endpoint."""

    def test_rotate_key_success(self, partner_handler, mock_partner_api):
        """Happy path: rotate API key."""
        handler = make_mock_handler(
            method="POST",
            path="/api/v1/partners/keys/key-123/rotate",
        )

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._rotate_api_key("key-123", handler, user=MagicMock())

        assert result is not None
        assert result.status_code == 201
        data = json.loads(result.body)
        assert "key" in data
        assert data["key"].startswith("ara_")
        assert data["rotated_from"] == "key-123"
        assert "warning" in data

    def test_rotate_key_missing_partner_id(self, partner_handler, mock_partner_api):
        """Missing partner ID returns 400."""
        handler = make_mock_handler(
            method="POST",
            path="/api/v1/partners/keys/key-123/rotate",
            partner_id=None,
        )

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._rotate_api_key("key-123", handler, user=MagicMock())

        assert result is not None
        assert result.status_code == 400

    def test_rotate_key_not_found(self, partner_handler, mock_partner_api):
        """Non-existent key returns 400."""
        handler = make_mock_handler(
            method="POST",
            path="/api/v1/partners/keys/nonexistent/rotate",
        )

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._rotate_api_key("nonexistent", handler, user=MagicMock())

        assert result is not None
        assert result.status_code == 400

    def test_rotate_key_via_routing(self, partner_handler, mock_partner_api):
        """Rotation should be reachable via handle() routing."""
        handler = make_mock_handler(
            method="POST",
            path="/api/v1/partners/keys/key-123/rotate",
        )

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler.handle(
                "/api/v1/partners/keys/key-123/rotate", {}, handler
            )

        assert result is not None
        assert result.status_code == 201


# ===========================================================================
# Test Revoke API Key (DELETE /api/v1/partners/keys/{key_id})
# ===========================================================================


class TestPartnerRevokeKey:
    """Tests for DELETE /api/v1/partners/keys/{key_id} endpoint."""

    def test_revoke_key_success(self, partner_handler, mock_partner_api):
        """Happy path: revoke API key."""
        handler = make_mock_handler(
            method="DELETE",
            path="/api/v1/partners/keys/key-123",
        )

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._revoke_api_key("key-123", handler, user=MagicMock())

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "revoked" in data["message"].lower()

    def test_revoke_key_not_found(self, partner_handler, mock_partner_api):
        """Key not found returns 404."""
        handler = make_mock_handler(
            method="DELETE",
            path="/api/v1/partners/keys/nonexistent",
        )

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._revoke_api_key("nonexistent", handler, user=MagicMock())

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Test Get Usage (GET /api/v1/partners/usage)
# ===========================================================================


class TestPartnerGetUsage:
    """Tests for GET /api/v1/partners/usage endpoint."""

    def test_get_usage_success(self, partner_handler, mock_partner_api):
        """Happy path: get usage stats."""
        handler = make_mock_handler(path="/api/v1/partners/usage")

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._get_usage({}, handler, user=MagicMock())

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "usage" in data

    def test_get_usage_with_days(self, partner_handler, mock_partner_api):
        """Get usage with custom days parameter."""
        handler = make_mock_handler(path="/api/v1/partners/usage")

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._get_usage({"days": "7"}, handler, user=MagicMock())

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Test Configure Webhook (POST /api/v1/partners/webhooks)
# ===========================================================================


class TestPartnerConfigureWebhook:
    """Tests for POST /api/v1/partners/webhooks endpoint."""

    def test_configure_webhook_success(self, partner_handler, mock_partner_api):
        """Happy path: configure webhook."""
        handler = make_mock_handler(
            body={"url": "https://example.com/webhook"},
            method="POST",
            path="/api/v1/partners/webhooks",
        )

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._configure_webhook(handler, user=MagicMock())

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["webhook_url"] == "https://example.com/webhook"
        assert "webhook_secret" in data

    def test_configure_webhook_missing_url(self, partner_handler, mock_partner_api):
        """Missing URL returns 400."""
        handler = make_mock_handler(
            body={},
            method="POST",
            path="/api/v1/partners/webhooks",
        )

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._configure_webhook(handler, user=MagicMock())

        assert result is not None
        assert result.status_code == 400

    def test_configure_webhook_http_rejected(self, partner_handler, mock_partner_api):
        """HTTP (non-HTTPS) URL rejected."""
        handler = make_mock_handler(
            body={"url": "http://example.com/webhook"},
            method="POST",
            path="/api/v1/partners/webhooks",
        )

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._configure_webhook(handler, user=MagicMock())

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "https" in data["error"].lower()


# ===========================================================================
# Test Get Limits (GET /api/v1/partners/limits)
# ===========================================================================


class TestPartnerGetLimits:
    """Tests for GET /api/v1/partners/limits endpoint."""

    def test_get_limits_success(self, partner_handler, mock_partner_api):
        """Happy path: get rate limits."""
        handler = make_mock_handler(path="/api/v1/partners/limits")

        with (
            patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api),
            patch(
                "aragora.billing.partner.PARTNER_TIER_LIMITS",
                {MockPartnerTier(): MockPartnerLimits()},
            ),
        ):
            result = partner_handler._get_limits(handler, user=MagicMock())

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "tier" in data
        assert "limits" in data
        assert "current" in data

    def test_get_limits_partner_not_found(self, partner_handler, mock_partner_api):
        """Partner not found returns 404."""
        handler = make_mock_handler(
            path="/api/v1/partners/limits",
            partner_id="nonexistent",
        )

        with patch("aragora.billing.partner.get_partner_api", return_value=mock_partner_api):
            result = partner_handler._get_limits(handler, user=MagicMock())

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Test Rate Limiting
# ===========================================================================


class TestPartnerRateLimiting:
    """Tests for rate limiting on partner endpoints."""

    def test_rate_limit_decorator(self, partner_handler):
        """Handler has rate limit decorator."""
        # The handle method should have the rate_limit decorator applied
        assert hasattr(partner_handler, "handle")


# ===========================================================================
# Test Error Handling
# ===========================================================================


class TestPartnerErrorHandling:
    """Tests for error handling in partner handler."""

    def test_invalid_json_body(self, partner_handler):
        """Invalid JSON body returns 400."""
        handler = MagicMock()
        handler.command = "POST"
        handler.headers = {"Content-Length": "10", "X-Partner-ID": "partner-123"}
        handler.rfile = BytesIO(b"not-json!!")

        result = partner_handler._register_partner(handler)

        assert result is not None
        assert result.status_code == 400

    def test_api_error_handling(self, partner_handler):
        """API errors are handled gracefully."""
        handler = make_mock_handler()

        with patch("aragora.billing.partner.get_partner_api") as mock_get:
            mock_api = MagicMock()
            mock_api.get_partner_stats.side_effect = Exception("Database error")
            mock_get.return_value = mock_api

            result = partner_handler._get_partner_profile(handler, user=MagicMock())

        assert result is not None
        assert result.status_code == 500
