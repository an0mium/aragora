"""
Comprehensive tests for the Partner handler at aragora/server/handlers/partner.py.

Tests cover:
- API key CRUD (create, list, revoke)
- Webhook configuration
- Rate limits endpoint
- Partner registration and profile
- Permission checks (require_permission("partner:read"))
- Input validation and error handling
- Edge cases and error paths

All imports are inside test functions/classes to avoid conftest issues.
Run with: pytest tests/server/handlers/test_partner.py -v --noconftest --timeout=30
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pre-stub the Slack handler modules to avoid circular ImportError when
# importing aragora.server.handlers (its __init__.py tries to import Slack).
# This must happen before any aragora.server.handlers imports.
# ---------------------------------------------------------------------------
import sys
import types

_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]

for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
):
    if _mod_name not in sys.modules:
        _m = types.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

# ---------------------------------------------------------------------------
# Now safe to import from the project
# ---------------------------------------------------------------------------

import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.partner import PartnerHandler
from aragora.rbac.models import AuthorizationContext
from aragora.rbac.decorators import PermissionDeniedError


# ---------------------------------------------------------------------------
# Fake data classes
# ---------------------------------------------------------------------------


class _FakeTier:
    def __init__(self, value):
        self.value = value


class _FakeStatus:
    def __init__(self, value):
        self.value = value


@dataclass
class _FakePartner:
    partner_id: str
    name: str
    email: str
    company: str | None
    tier: Any
    status: Any
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    webhook_url: str | None = None
    webhook_secret: str | None = None
    referral_code: str | None = "REFCODE1"

    def to_dict(self):
        return {
            "partner_id": self.partner_id,
            "name": self.name,
            "email": self.email,
            "company": self.company,
            "tier": self.tier.value if hasattr(self.tier, "value") else self.tier,
            "status": self.status.value if hasattr(self.status, "value") else self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "webhook_url": self.webhook_url,
            "referral_code": self.referral_code,
        }


@dataclass
class _FakeAPIKey:
    key_id: str
    partner_id: str
    key_prefix: str
    key_hash: str
    name: str
    scopes: list[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    last_used_at: datetime | None = None
    is_active: bool = True

    def to_dict(self):
        return {
            "key_id": self.key_id,
            "partner_id": self.partner_id,
            "key_prefix": self.key_prefix,
            "name": self.name,
            "scopes": self.scopes,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "is_active": self.is_active,
        }


@dataclass
class _FakeLimits:
    requests_per_minute: int = 60
    requests_per_day: int = 1000
    debates_per_month: int = 100
    max_agents_per_debate: int = 3
    max_rounds: int = 3
    webhook_endpoints: int = 1
    revenue_share_percent: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(method="GET", body=None, headers=None, partner_id=None):
    """Build a mock HTTP request handler."""
    h = MagicMock()
    h.command = method

    hdr = {}
    if partner_id:
        hdr["X-Partner-ID"] = partner_id
    if body is not None:
        raw = json.dumps(body).encode()
        hdr["Content-Length"] = str(len(raw))
        h.rfile = io.BytesIO(raw)
    else:
        hdr["Content-Length"] = "0"
        h.rfile = io.BytesIO(b"")
    if headers:
        hdr.update(headers)

    h.headers = hdr
    h.client_address = ("127.0.0.1", 12345)
    return h


def _parse(result):
    """Parse HandlerResult body."""
    if result is None:
        return None
    return json.loads(result.body.decode())


_PATCH_API = "aragora.billing.partner.get_partner_api"
_PATCH_AUTH = "aragora.billing.jwt_auth.extract_user_from_request"


def _mock_user():
    u = MagicMock()
    u.is_authenticated = True
    u.user_id = "user1"
    return u


def _admin_ctx():
    return AuthorizationContext(
        user_id="u1",
        org_id="org1",
        roles={"admin"},
        permissions={"partner:read"},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_globals():
    """Reset partner singleton and rate limiter state between tests."""
    import aragora.billing.partner as bp

    bp._partner_api = None
    from aragora.server.handlers.utils.rate_limit import _limiters, _limiters_lock

    with _limiters_lock:
        for limiter in _limiters.values():
            limiter.clear()
    yield
    bp._partner_api = None
    with _limiters_lock:
        for limiter in _limiters.values():
            limiter.clear()


@pytest.fixture(autouse=True)
def _bypass_rbac_and_auth(monkeypatch):
    """Auto-bypass RBAC and user auth for all tests (like the conftest does)."""
    from aragora.rbac import decorators

    original = decorators._get_context_from_args
    ctx = _admin_ctx()

    def _patched(args, kwargs, context_param):
        result = original(args, kwargs, context_param)
        return result if result is not None else ctx

    monkeypatch.setattr(decorators, "_get_context_from_args", _patched)

    # Also bypass require_user_auth by patching extract_user_from_request
    mock_user = _mock_user()
    monkeypatch.setattr(
        "aragora.billing.jwt_auth.extract_user_from_request",
        lambda handler, user_store=None: mock_user,
    )


@pytest.fixture
def ph():
    """PartnerHandler instance."""
    return PartnerHandler({})


@pytest.fixture
def partner():
    return _FakePartner(
        partner_id="partner_abc",
        name="Test Partner",
        email="test@example.com",
        company="TestCo",
        tier=_FakeTier("starter"),
        status=_FakeStatus("active"),
    )


@pytest.fixture
def api_key():
    return _FakeAPIKey(
        key_id="key_abc",
        partner_id="partner_abc",
        key_prefix="ara_abcdefgh",
        key_hash="somehash",
        name="Test Key",
        scopes=["debates:read", "debates:write"],
    )


# ===========================================================================
# can_handle
# ===========================================================================


class TestCanHandle:
    def test_register(self, ph):
        assert ph.can_handle("/api/v1/partners/register")

    def test_me(self, ph):
        assert ph.can_handle("/api/v1/partners/me")

    def test_keys(self, ph):
        assert ph.can_handle("/api/v1/partners/keys")

    def test_keys_specific(self, ph):
        assert ph.can_handle("/api/v1/partners/keys/key_abc")

    def test_usage(self, ph):
        assert ph.can_handle("/api/v1/partners/usage")

    def test_webhooks(self, ph):
        assert ph.can_handle("/api/v1/partners/webhooks")

    def test_limits(self, ph):
        assert ph.can_handle("/api/v1/partners/limits")

    def test_unknown(self, ph):
        assert not ph.can_handle("/api/v1/unknown")

    def test_partial(self, ph):
        assert not ph.can_handle("/api/v1/partners/other")


# ===========================================================================
# register partner
# ===========================================================================


class TestRegisterPartner:
    @patch(_PATCH_API)
    def test_success(self, mock_api, ph, partner):
        api = MagicMock()
        api._store.get_partner_by_email.return_value = None
        api.register_partner.return_value = partner
        mock_api.return_value = api

        h = _make_handler("POST", body={"name": "T", "email": "t@e.com"})
        r = ph._register_partner(h)
        assert r.status_code == 201
        d = _parse(r)
        assert d["partner_id"] == "partner_abc"
        assert "referral_code" in d
        assert "message" in d

    @patch(_PATCH_API)
    def test_with_company(self, mock_api, ph, partner):
        api = MagicMock()
        api._store.get_partner_by_email.return_value = None
        api.register_partner.return_value = partner
        mock_api.return_value = api

        h = _make_handler("POST", body={"name": "T", "email": "t@e.com", "company": "ACME"})
        assert ph._register_partner(h).status_code == 201

    @patch(_PATCH_API)
    def test_duplicate_email(self, mock_api, ph, partner):
        api = MagicMock()
        api._store.get_partner_by_email.return_value = partner
        mock_api.return_value = api

        h = _make_handler("POST", body={"name": "T", "email": "t@e.com"})
        r = ph._register_partner(h)
        assert r.status_code == 409

    def test_missing_name(self, ph):
        h = _make_handler("POST", body={"email": "t@e.com"})
        assert ph._register_partner(h).status_code == 400

    def test_missing_email(self, ph):
        h = _make_handler("POST", body={"name": "T"})
        assert ph._register_partner(h).status_code == 400

    def test_empty_body(self, ph):
        h = _make_handler("POST", body={})
        assert ph._register_partner(h).status_code == 400

    def test_invalid_json(self, ph):
        h = MagicMock()
        h.headers = {"Content-Length": "3"}
        h.rfile = io.BytesIO(b"abc")
        h.client_address = ("127.0.0.1", 1)
        assert ph._register_partner(h).status_code == 400

    @patch(_PATCH_API)
    def test_api_error(self, mock_api, ph):
        api = MagicMock()
        api._store.get_partner_by_email.return_value = None
        api.register_partner.side_effect = RuntimeError("DB down")
        mock_api.return_value = api

        h = _make_handler("POST", body={"name": "T", "email": "t@e.com"})
        assert ph._register_partner(h).status_code == 500


# ===========================================================================
# get partner profile
# ===========================================================================


class TestGetPartnerProfile:
    @patch(_PATCH_API)
    def test_success(self, mock_api, ph):
        api = MagicMock()
        api.get_partner_stats.return_value = {"partner": {}, "usage": {}}
        mock_api.return_value = api

        h = _make_handler("GET", partner_id="partner_abc")
        assert ph._get_partner_profile(h, user=_mock_user()).status_code == 200

    def test_no_partner_id(self, ph):
        assert ph._get_partner_profile(_make_handler("GET"), user=_mock_user()).status_code == 400

    @patch(_PATCH_API)
    def test_not_found(self, mock_api, ph):
        api = MagicMock()
        api.get_partner_stats.side_effect = ValueError("not found")
        mock_api.return_value = api

        h = _make_handler("GET", partner_id="bad")
        assert ph._get_partner_profile(h, user=_mock_user()).status_code == 404

    @patch(_PATCH_API)
    def test_generic_error(self, mock_api, ph):
        api = MagicMock()
        api.get_partner_stats.side_effect = RuntimeError("boom")
        mock_api.return_value = api

        h = _make_handler("GET", partner_id="p")
        assert ph._get_partner_profile(h, user=_mock_user()).status_code == 500


# ===========================================================================
# create API key
# ===========================================================================


class TestCreateAPIKey:
    @patch(_PATCH_API)
    def test_success(self, mock_api, ph, api_key):
        api = MagicMock()
        api.create_api_key.return_value = (api_key, "ara_raw_secret")
        mock_api.return_value = api

        h = _make_handler("POST", body={"name": "My Key"}, partner_id="partner_abc")
        r = ph._create_api_key(h, user=_mock_user())
        assert r.status_code == 201
        d = _parse(r)
        assert d["key"] == "ara_raw_secret"
        assert d["key_id"] == "key_abc"
        assert "warning" in d

    @patch(_PATCH_API)
    def test_with_scopes_and_expiry(self, mock_api, ph, api_key):
        api = MagicMock()
        api.create_api_key.return_value = (api_key, "ara_k")
        mock_api.return_value = api

        body = {"name": "S", "scopes": ["debates:read"], "expires_in_days": 90}
        h = _make_handler("POST", body=body, partner_id="partner_abc")
        ph._create_api_key(h, user=_mock_user())
        api.create_api_key.assert_called_once_with(
            partner_id="partner_abc",
            name="S",
            scopes=["debates:read"],
            expires_in_days=90,
        )

    @patch(_PATCH_API)
    def test_default_name(self, mock_api, ph, api_key):
        api = MagicMock()
        api.create_api_key.return_value = (api_key, "ara_k")
        mock_api.return_value = api

        h = _make_handler("POST", body={}, partner_id="partner_abc")
        ph._create_api_key(h, user=_mock_user())
        api.create_api_key.assert_called_once_with(
            partner_id="partner_abc",
            name="API Key",
            scopes=None,
            expires_in_days=None,
        )

    def test_no_partner_id(self, ph):
        h = _make_handler("POST", body={"name": "K"})
        assert ph._create_api_key(h, user=_mock_user()).status_code == 400

    @patch(_PATCH_API)
    def test_partner_not_active(self, mock_api, ph):
        api = MagicMock()
        api.create_api_key.side_effect = ValueError("not active")
        mock_api.return_value = api

        h = _make_handler("POST", body={"name": "K"}, partner_id="p")
        assert ph._create_api_key(h, user=_mock_user()).status_code == 400

    def test_invalid_json(self, ph):
        h = MagicMock()
        h.headers = {"Content-Length": "5", "X-Partner-ID": "p"}
        h.rfile = io.BytesIO(b"xxxxx")
        h.client_address = ("127.0.0.1", 1)
        assert ph._create_api_key(h, user=_mock_user()).status_code == 400

    @patch(_PATCH_API)
    def test_generic_error(self, mock_api, ph):
        api = MagicMock()
        api.create_api_key.side_effect = RuntimeError("boom")
        mock_api.return_value = api

        h = _make_handler("POST", body={"name": "K"}, partner_id="p")
        assert ph._create_api_key(h, user=_mock_user()).status_code == 500

    @patch(_PATCH_API)
    def test_with_expires_at(self, mock_api, ph):
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        key = _FakeAPIKey(
            key_id="key_e",
            partner_id="p",
            key_prefix="ara_e",
            key_hash="h",
            name="E",
            scopes=["debates:read"],
            expires_at=now + timedelta(days=30),
        )
        api = MagicMock()
        api.create_api_key.return_value = (key, "ara_r")
        mock_api.return_value = api

        h = _make_handler("POST", body={"name": "E"}, partner_id="p")
        d = _parse(ph._create_api_key(h, user=_mock_user()))
        assert d["expires_at"] is not None

    @patch(_PATCH_API)
    def test_without_expires_at(self, mock_api, ph):
        key = _FakeAPIKey(
            key_id="key_n",
            partner_id="p",
            key_prefix="ara_n",
            key_hash="h",
            name="N",
            scopes=[],
            expires_at=None,
        )
        api = MagicMock()
        api.create_api_key.return_value = (key, "ara_r")
        mock_api.return_value = api

        h = _make_handler("POST", body={"name": "N"}, partner_id="p")
        d = _parse(ph._create_api_key(h, user=_mock_user()))
        assert d["expires_at"] is None


# ===========================================================================
# list API keys
# ===========================================================================


class TestListAPIKeys:
    @patch(_PATCH_API)
    def test_success(self, mock_api, ph, api_key):
        api = MagicMock()
        api._store.list_partner_keys.return_value = [api_key]
        mock_api.return_value = api

        h = _make_handler("GET", partner_id="partner_abc")
        d = _parse(ph._list_api_keys(h, user=_mock_user()))
        assert d["total"] == 1
        assert d["active"] == 1

    @patch(_PATCH_API)
    def test_empty(self, mock_api, ph):
        api = MagicMock()
        api._store.list_partner_keys.return_value = []
        mock_api.return_value = api

        h = _make_handler("GET", partner_id="p")
        d = _parse(ph._list_api_keys(h, user=_mock_user()))
        assert d["total"] == 0
        assert d["active"] == 0

    def test_no_partner_id(self, ph):
        assert ph._list_api_keys(_make_handler("GET"), user=_mock_user()).status_code == 400

    @patch(_PATCH_API)
    def test_with_revoked(self, mock_api, ph, api_key):
        revoked = _FakeAPIKey(
            key_id="key_r",
            partner_id="p",
            key_prefix="ara_r",
            key_hash="h2",
            name="R",
            scopes=[],
            is_active=False,
        )
        api = MagicMock()
        api._store.list_partner_keys.return_value = [api_key, revoked]
        mock_api.return_value = api

        h = _make_handler("GET", partner_id="p")
        d = _parse(ph._list_api_keys(h, user=_mock_user()))
        assert d["total"] == 2
        assert d["active"] == 1

    @patch(_PATCH_API)
    def test_generic_error(self, mock_api, ph):
        api = MagicMock()
        api._store.list_partner_keys.side_effect = RuntimeError("DB")
        mock_api.return_value = api

        h = _make_handler("GET", partner_id="p")
        assert ph._list_api_keys(h, user=_mock_user()).status_code == 500


# ===========================================================================
# revoke API key
# ===========================================================================


class TestRevokeAPIKey:
    @patch(_PATCH_API)
    def test_success(self, mock_api, ph):
        api = MagicMock()
        api._store.revoke_api_key.return_value = True
        mock_api.return_value = api

        h = _make_handler("DELETE", partner_id="p")
        r = ph._revoke_api_key("key_abc", h, user=_mock_user())
        assert r.status_code == 200
        assert "revoked" in _parse(r)["message"].lower()

    @patch(_PATCH_API)
    def test_not_found(self, mock_api, ph):
        api = MagicMock()
        api._store.revoke_api_key.return_value = False
        mock_api.return_value = api

        h = _make_handler("DELETE", partner_id="p")
        assert ph._revoke_api_key("key_bad", h, user=_mock_user()).status_code == 404

    def test_no_partner_id(self, ph):
        h = _make_handler("DELETE")
        assert ph._revoke_api_key("k", h, user=_mock_user()).status_code == 400

    @patch(_PATCH_API)
    def test_store_error(self, mock_api, ph):
        api = MagicMock()
        api._store.revoke_api_key.side_effect = RuntimeError("DB")
        mock_api.return_value = api

        h = _make_handler("DELETE", partner_id="p")
        assert ph._revoke_api_key("k", h, user=_mock_user()).status_code == 500


# ===========================================================================
# get usage
# ===========================================================================


class TestGetUsage:
    @patch(_PATCH_API)
    def test_success(self, mock_api, ph):
        api = MagicMock()
        api.get_partner_stats.return_value = {"usage": {"total": 42}}
        mock_api.return_value = api

        h = _make_handler("GET", partner_id="p")
        assert ph._get_usage({}, h, user=_mock_user()).status_code == 200

    @patch(_PATCH_API)
    def test_with_days(self, mock_api, ph):
        api = MagicMock()
        api.get_partner_stats.return_value = {}
        mock_api.return_value = api

        h = _make_handler("GET", partner_id="p")
        ph._get_usage({"days": "7"}, h, user=_mock_user())
        api.get_partner_stats.assert_called_once_with("p", days=7)

    @patch(_PATCH_API)
    def test_invalid_days_defaults(self, mock_api, ph):
        api = MagicMock()
        api.get_partner_stats.return_value = {}
        mock_api.return_value = api

        h = _make_handler("GET", partner_id="p")
        ph._get_usage({"days": "xyz"}, h, user=_mock_user())
        api.get_partner_stats.assert_called_once_with("p", days=30)

    def test_no_partner_id(self, ph):
        assert ph._get_usage({}, _make_handler("GET"), user=_mock_user()).status_code == 400

    @patch(_PATCH_API)
    def test_not_found(self, mock_api, ph):
        api = MagicMock()
        api.get_partner_stats.side_effect = ValueError("nf")
        mock_api.return_value = api

        h = _make_handler("GET", partner_id="bad")
        assert ph._get_usage({}, h, user=_mock_user()).status_code == 404

    @patch(_PATCH_API)
    def test_generic_error(self, mock_api, ph):
        api = MagicMock()
        api.get_partner_stats.side_effect = RuntimeError("boom")
        mock_api.return_value = api

        h = _make_handler("GET", partner_id="p")
        assert ph._get_usage({}, h, user=_mock_user()).status_code == 500


# ===========================================================================
# configure webhook
# ===========================================================================


class TestConfigureWebhook:
    @patch(_PATCH_API)
    def test_success(self, mock_api, ph, partner):
        api = MagicMock()
        api._store.get_partner.return_value = partner
        api.generate_webhook_secret.return_value = "whsec_test"
        mock_api.return_value = api

        h = _make_handler("POST", body={"url": "https://example.com/wh"}, partner_id="p")
        r = ph._configure_webhook(h, user=_mock_user())
        assert r.status_code == 200
        d = _parse(r)
        assert d["webhook_url"] == "https://example.com/wh"
        assert d["webhook_secret"] == "whsec_test"
        assert "warning" in d

    def test_no_partner_id(self, ph):
        h = _make_handler("POST", body={"url": "https://x.com/wh"})
        assert ph._configure_webhook(h, user=_mock_user()).status_code == 400

    def test_missing_url(self, ph):
        h = _make_handler("POST", body={}, partner_id="p")
        assert ph._configure_webhook(h, user=_mock_user()).status_code == 400

    def test_non_https(self, ph):
        h = _make_handler("POST", body={"url": "http://x.com/wh"}, partner_id="p")
        r = ph._configure_webhook(h, user=_mock_user())
        assert r.status_code == 400
        assert "https" in _parse(r)["error"].lower()

    @patch(_PATCH_API)
    def test_partner_not_found(self, mock_api, ph):
        api = MagicMock()
        api._store.get_partner.return_value = None
        mock_api.return_value = api

        h = _make_handler("POST", body={"url": "https://x.com/wh"}, partner_id="p")
        assert ph._configure_webhook(h, user=_mock_user()).status_code == 404

    def test_invalid_json(self, ph):
        h = MagicMock()
        h.headers = {"Content-Length": "4", "X-Partner-ID": "p"}
        h.rfile = io.BytesIO(b"nope")
        h.client_address = ("127.0.0.1", 1)
        assert ph._configure_webhook(h, user=_mock_user()).status_code == 400

    @patch(_PATCH_API)
    def test_generic_error(self, mock_api, ph, partner):
        api = MagicMock()
        api._store.get_partner.return_value = partner
        api.generate_webhook_secret.side_effect = RuntimeError("boom")
        mock_api.return_value = api

        h = _make_handler("POST", body={"url": "https://x.com/wh"}, partner_id="p")
        assert ph._configure_webhook(h, user=_mock_user()).status_code == 500


# ===========================================================================
# get limits
# ===========================================================================


class TestGetLimits:
    @patch(_PATCH_API)
    def test_success(self, mock_api, ph, partner):
        api = MagicMock()
        api._store.get_partner.return_value = partner
        api.check_rate_limit.return_value = (True, {"remaining_minute": 50})
        mock_api.return_value = api

        with patch("aragora.billing.partner.PARTNER_TIER_LIMITS", {partner.tier: _FakeLimits()}):
            h = _make_handler("GET", partner_id="p")
            r = ph._get_limits(h, user=_mock_user())

        assert r.status_code == 200
        d = _parse(r)
        assert d["limits"]["requests_per_minute"] == 60
        assert d["allowed"] is True

    def test_no_partner_id(self, ph):
        assert ph._get_limits(_make_handler("GET"), user=_mock_user()).status_code == 400

    @patch(_PATCH_API)
    def test_partner_not_found(self, mock_api, ph):
        api = MagicMock()
        api._store.get_partner.return_value = None
        mock_api.return_value = api

        h = _make_handler("GET", partner_id="bad")
        assert ph._get_limits(h, user=_mock_user()).status_code == 404

    @patch(_PATCH_API)
    def test_generic_error(self, mock_api, ph, partner):
        api = MagicMock()
        api._store.get_partner.return_value = partner
        api.check_rate_limit.side_effect = RuntimeError("err")
        mock_api.return_value = api

        h = _make_handler("GET", partner_id="p")
        assert ph._get_limits(h, user=_mock_user()).status_code == 500


# ===========================================================================
# handle() routing (top-level dispatch with @require_permission + @rate_limit)
# ===========================================================================


class TestHandleRouting:
    """Test handle() dispatch. RBAC is auto-bypassed by _bypass_rbac fixture."""

    @patch(_PATCH_API)
    def test_register_post(self, mock_api, ph, partner):
        api = MagicMock()
        api._store.get_partner_by_email.return_value = None
        api.register_partner.return_value = partner
        mock_api.return_value = api

        h = _make_handler("POST", body={"name": "T", "email": "t@e.com"})
        r = ph.handle("/api/v1/partners/register", {}, h)
        assert r is not None and r.status_code == 201

    def test_unknown_returns_none(self, ph):
        r = ph.handle("/api/v1/partners/unknown", {}, _make_handler("GET"))
        assert r is None

    def test_wrong_method_returns_none(self, ph):
        r = ph.handle("/api/v1/partners/register", {}, _make_handler("GET"))
        assert r is None

    @patch(_PATCH_AUTH)
    @patch(_PATCH_API)
    def test_get_me(self, mock_api, mock_auth, ph):
        mock_auth.return_value = _mock_user()
        api = MagicMock()
        api.get_partner_stats.return_value = {"partner": {}}
        mock_api.return_value = api

        h = _make_handler("GET", partner_id="p")
        r = ph.handle("/api/v1/partners/me", {}, h)
        assert r is not None and r.status_code == 200

    @patch(_PATCH_AUTH)
    @patch(_PATCH_API)
    def test_delete_key(self, mock_api, mock_auth, ph):
        mock_auth.return_value = _mock_user()
        api = MagicMock()
        api._store.revoke_api_key.return_value = True
        mock_api.return_value = api

        h = _make_handler("DELETE", partner_id="p")
        r = ph.handle("/api/v1/partners/keys/key_123", {}, h)
        assert r is not None and r.status_code == 200

    @patch(_PATCH_AUTH)
    @patch(_PATCH_API)
    def test_post_keys(self, mock_api, mock_auth, ph, api_key):
        mock_auth.return_value = _mock_user()
        api = MagicMock()
        api.create_api_key.return_value = (api_key, "ara_r")
        mock_api.return_value = api

        h = _make_handler("POST", body={"name": "K"}, partner_id="p")
        r = ph.handle("/api/v1/partners/keys", {}, h)
        assert r is not None and r.status_code == 201

    @patch(_PATCH_AUTH)
    @patch(_PATCH_API)
    def test_get_keys(self, mock_api, mock_auth, ph, api_key):
        mock_auth.return_value = _mock_user()
        api = MagicMock()
        api._store.list_partner_keys.return_value = [api_key]
        mock_api.return_value = api

        h = _make_handler("GET", partner_id="p")
        r = ph.handle("/api/v1/partners/keys", {}, h)
        assert r is not None and r.status_code == 200

    @patch(_PATCH_AUTH)
    @patch(_PATCH_API)
    def test_get_usage_via_handle(self, mock_api, mock_auth, ph):
        mock_auth.return_value = _mock_user()
        api = MagicMock()
        api.get_partner_stats.return_value = {"usage": {}}
        mock_api.return_value = api

        h = _make_handler("GET", partner_id="p")
        r = ph.handle("/api/v1/partners/usage", {}, h)
        assert r is not None and r.status_code == 200

    @patch(_PATCH_AUTH)
    @patch(_PATCH_API)
    def test_post_webhooks_via_handle(self, mock_api, mock_auth, ph, partner):
        mock_auth.return_value = _mock_user()
        api = MagicMock()
        api._store.get_partner.return_value = partner
        api.generate_webhook_secret.return_value = "whsec_x"
        mock_api.return_value = api

        h = _make_handler("POST", body={"url": "https://x.com/wh"}, partner_id="p")
        r = ph.handle("/api/v1/partners/webhooks", {}, h)
        assert r is not None and r.status_code == 200

    @patch(_PATCH_AUTH)
    @patch(_PATCH_API)
    def test_get_limits_via_handle(self, mock_api, mock_auth, ph, partner):
        mock_auth.return_value = _mock_user()
        api = MagicMock()
        api._store.get_partner.return_value = partner
        api.check_rate_limit.return_value = (True, {})
        mock_api.return_value = api

        with patch("aragora.billing.partner.PARTNER_TIER_LIMITS", {partner.tier: _FakeLimits()}):
            h = _make_handler("GET", partner_id="p")
            r = ph.handle("/api/v1/partners/limits", {}, h)
        assert r is not None and r.status_code == 200


# ===========================================================================
# permission decorator
# ===========================================================================


class TestPermissionDecorator:
    def test_handle_is_decorated_with_require_permission(self, ph):
        """The handle() method has the @require_permission decorator applied."""
        # The decorator wraps the function, so __wrapped__ should exist
        assert hasattr(ph.handle, "__wrapped__")

    def test_proceeds_with_auto_bypass(self, ph):
        """With auto-bypass, handle() proceeds without explicit AuthorizationContext."""
        h = _make_handler("GET")
        # GET on /register -> no match -> None
        assert ph.handle("/api/v1/partners/register", {}, h) is None

    def test_handle_decorated_with_rate_limit(self, ph):
        """The handle() method is also decorated with @rate_limit."""
        # rate_limit + require_permission wrap the function in two layers
        inner = ph.handle
        found_wrapped = False
        depth = 0
        while hasattr(inner, "__wrapped__") and depth < 5:
            found_wrapped = True
            inner = inner.__wrapped__
            depth += 1
        assert found_wrapped


# ===========================================================================
# edge cases
# ===========================================================================


class TestEdgeCases:
    def test_zero_content_length(self, ph):
        h = MagicMock()
        h.headers = {"Content-Length": "0"}
        h.rfile = io.BytesIO(b"")
        h.client_address = ("127.0.0.1", 1)
        assert ph._register_partner(h).status_code == 400

    @patch(_PATCH_API)
    def test_register_response_fields(self, mock_api, ph, partner):
        api = MagicMock()
        api._store.get_partner_by_email.return_value = None
        api.register_partner.return_value = partner
        mock_api.return_value = api

        h = _make_handler("POST", body={"name": "T", "email": "t@e.com"})
        d = _parse(ph._register_partner(h))
        for key in ("partner_id", "name", "email", "tier", "status", "referral_code", "message"):
            assert key in d

    @patch(_PATCH_API)
    def test_create_key_response_fields(self, mock_api, ph, api_key):
        api = MagicMock()
        api.create_api_key.return_value = (api_key, "ara_raw")
        mock_api.return_value = api

        h = _make_handler("POST", body={"name": "K"}, partner_id="p")
        d = _parse(ph._create_api_key(h, user=_mock_user()))
        for key in (
            "key_id",
            "key",
            "key_prefix",
            "name",
            "scopes",
            "created_at",
            "expires_at",
            "warning",
        ):
            assert key in d

    @patch(_PATCH_API)
    def test_revoke_response_contains_key_id(self, mock_api, ph):
        api = MagicMock()
        api._store.revoke_api_key.return_value = True
        mock_api.return_value = api

        h = _make_handler("DELETE", partner_id="p")
        d = _parse(ph._revoke_api_key("key_xyz", h, user=_mock_user()))
        assert d["key_id"] == "key_xyz"

    @patch(_PATCH_API)
    def test_webhook_sets_partner_url(self, mock_api, ph, partner):
        """Webhook config should set webhook_url on the partner object."""
        api = MagicMock()
        api._store.get_partner.return_value = partner
        api.generate_webhook_secret.return_value = "whsec_s"
        mock_api.return_value = api

        h = _make_handler("POST", body={"url": "https://x.com/wh"}, partner_id="p")
        ph._configure_webhook(h, user=_mock_user())
        assert partner.webhook_url == "https://x.com/wh"

    @patch(_PATCH_API)
    def test_limits_all_fields(self, mock_api, ph, partner):
        """Limits response contains all tier limit fields."""
        api = MagicMock()
        api._store.get_partner.return_value = partner
        api.check_rate_limit.return_value = (False, {"remaining": 0})
        mock_api.return_value = api

        with patch("aragora.billing.partner.PARTNER_TIER_LIMITS", {partner.tier: _FakeLimits()}):
            h = _make_handler("GET", partner_id="p")
            d = _parse(ph._get_limits(h, user=_mock_user()))

        limits = d["limits"]
        for key in (
            "requests_per_minute",
            "requests_per_day",
            "debates_per_month",
            "max_agents_per_debate",
            "max_rounds",
            "webhook_endpoints",
            "revenue_share_percent",
        ):
            assert key in limits
        assert d["allowed"] is False
