"""Comprehensive tests for the PrivacyHandler (aragora/server/handlers/privacy.py).

Covers all privacy endpoints:
- GET  /api/v1/privacy/export          - Export user data (GDPR Article 15)
- GET  /api/v2/users/me/export         - V2 export alias
- GET  /api/v1/privacy/data-inventory  - Data categories inventory
- GET  /api/v2/users/me/data-inventory - V2 inventory alias
- DELETE /api/v1/privacy/account       - Account deletion (GDPR Article 17)
- DELETE /api/v2/users/me              - V2 account deletion alias
- GET  /api/v1/privacy/preferences     - Get privacy preferences
- POST /api/v1/privacy/preferences     - Update privacy preferences

Also covers:
- can_handle() routing
- _collect_user_data() helper
- _format_csv_export() helper
- _perform_account_deletion() helper
- _hash_for_audit() helper
- Method not allowed (405)
- Edge cases and error handling
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.billing.auth.context import UserAuthContext
from aragora.server.handlers.privacy import PrivacyHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to PrivacyHandler.handle."""

    def __init__(self, method: str = "GET", body: dict[str, Any] | None = None):
        self.command = method
        self.headers = {"Content-Length": "0"}
        self.rfile = MagicMock()

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers = {"Content-Length": str(len(raw))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}


# ---------------------------------------------------------------------------
# Mock data objects
# ---------------------------------------------------------------------------


class OrgTier:
    """Mock org tier with a value attribute."""

    def __init__(self, value: str = "professional"):
        self.value = value


@dataclass
class MockUser:
    """Mock user object matching the fields accessed in PrivacyHandler."""

    id: str = "user-001"
    email: str = "test@example.com"
    name: str = "Test User"
    role: str = "member"
    is_active: bool = True
    email_verified: bool = True
    created_at: datetime = field(default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime(2025, 6, 1, tzinfo=timezone.utc))
    last_login_at: datetime = field(
        default_factory=lambda: datetime(2026, 2, 1, tzinfo=timezone.utc)
    )
    mfa_enabled: bool = False
    api_key_prefix: str | None = None
    api_key_created_at: datetime | None = None
    api_key_expires_at: datetime | None = None
    api_key_hash: str | None = None
    org_id: str | None = None
    mfa_secret: str | None = None
    mfa_backup_codes: list[str] | None = None

    def verify_password(self, password: str) -> bool:
        """Mock password verification."""
        return password == "correct-password"


@dataclass
class MockOrg:
    """Mock organization."""

    id: str = "org-001"
    name: str = "Test Org"
    slug: str = "test-org"
    tier: Any = field(default_factory=lambda: OrgTier("professional"))
    owner_id: str = "user-001"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_user_store():
    """Create a mock user store with sensible defaults."""
    store = MagicMock()
    store.get_user_by_id.return_value = MockUser()
    store.get_organization_by_id.return_value = None
    store.get_user_oauth_providers.return_value = []
    store.get_user_preferences.return_value = {}
    store.get_audit_log.return_value = []
    store.get_usage_summary.return_value = None
    store.get_org_members.return_value = []
    store.log_audit_event.return_value = None
    store.set_user_preferences.return_value = None
    store.update_user.return_value = None
    store.unlink_oauth_provider.return_value = None
    store.remove_user_from_org.return_value = None
    return store


@pytest.fixture
def handler(mock_user_store):
    """Create a PrivacyHandler with a mock user store."""
    return PrivacyHandler(ctx={"user_store": mock_user_store})


@pytest.fixture
def handler_no_store():
    """Create a PrivacyHandler without a user store."""
    return PrivacyHandler(ctx={})


@pytest.fixture
def auth_context():
    """Create an authenticated user context."""
    return UserAuthContext(
        authenticated=True,
        user_id="user-001",
        email="test@example.com",
        org_id="org-001",
        role="admin",
        token_type="access",
        client_ip="127.0.0.1",
    )


@pytest.fixture(autouse=True)
def _patch_auth(monkeypatch, auth_context):
    """Patch extract_user_from_request to return authenticated context."""
    monkeypatch.setattr(
        "aragora.server.handlers.privacy.extract_user_from_request",
        lambda handler, user_store: auth_context,
    )


@pytest.fixture
def _patch_unauth(monkeypatch):
    """Patch extract_user_from_request to return unauthenticated context."""
    unauth = UserAuthContext(authenticated=False)
    monkeypatch.setattr(
        "aragora.server.handlers.privacy.extract_user_from_request",
        lambda handler, user_store: unauth,
    )


@pytest.fixture(autouse=True)
def _patch_rate_limit(monkeypatch):
    """Bypass rate limiting for tests."""
    monkeypatch.setattr(
        "aragora.server.handlers.privacy.rate_limit",
        lambda **kwargs: lambda fn: fn,
    )


# ============================================================================
# can_handle Tests
# ============================================================================


class TestCanHandle:
    """Tests for PrivacyHandler.can_handle()."""

    def test_handles_export_v1(self, handler):
        assert handler.can_handle("/api/v1/privacy/export") is True

    def test_handles_data_inventory_v1(self, handler):
        assert handler.can_handle("/api/v1/privacy/data-inventory") is True

    def test_handles_account_v1(self, handler):
        assert handler.can_handle("/api/v1/privacy/account") is True

    def test_handles_preferences_v1(self, handler):
        assert handler.can_handle("/api/v1/privacy/preferences") is True

    def test_handles_users_v1(self, handler):
        assert handler.can_handle("/api/v1/users") is True

    def test_handles_users_invite_v1(self, handler):
        assert handler.can_handle("/api/v1/users/invite") is True

    def test_handles_export_v2(self, handler):
        assert handler.can_handle("/api/v2/users/me/export") is True

    def test_handles_data_inventory_v2(self, handler):
        assert handler.can_handle("/api/v2/users/me/data-inventory") is True

    def test_handles_users_me_v2(self, handler):
        assert handler.can_handle("/api/v2/users/me") is True

    def test_rejects_unknown_path(self, handler):
        assert handler.can_handle("/api/v1/unknown") is False

    def test_rejects_partial_match(self, handler):
        assert handler.can_handle("/api/v1/privacy") is False


# ============================================================================
# Data Export Endpoint
# ============================================================================


class TestDataExport:
    """Tests for GET /api/v1/privacy/export and /api/v2/users/me/export."""

    def test_export_success_json(self, handler, mock_user_store):
        """Successful JSON export returns 200 with user data."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert "profile" in body
        assert body["profile"]["email"] == "test@example.com"

    def test_export_includes_metadata(self, handler, mock_user_store):
        """Export includes _export_metadata with legal info."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {}, mock_h)
        body = _body(result)
        meta = body["_export_metadata"]
        assert meta["data_controller"] == "Aragora"
        assert meta["contact"] == "privacy@aragora.ai"
        assert "GDPR" in meta["legal_basis"]
        assert meta["format"] == "json"

    def test_export_default_format_json(self, handler, mock_user_store):
        """Default format is json."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {}, mock_h)
        body = _body(result)
        assert body["_export_metadata"]["format"] == "json"

    def test_export_csv_format(self, handler, mock_user_store):
        """format=csv returns CSV response with correct content type."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {"format": "csv"}, mock_h)
        assert _status(result) == 200
        assert result.content_type == "text/csv; charset=utf-8"
        assert "Content-Disposition" in (result.headers or {})

    def test_export_csv_attachment_filename(self, handler, mock_user_store):
        """CSV export has correct attachment filename."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {"format": "csv"}, mock_h)
        disposition = result.headers["Content-Disposition"]
        assert "aragora_export_" in disposition
        assert disposition.startswith("attachment; filename=")

    def test_export_csv_contains_profile_section(self, handler, mock_user_store):
        """CSV export includes Profile section."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {"format": "csv"}, mock_h)
        csv_content = result.body.decode("utf-8")
        assert "Profile" in csv_content
        assert "test@example.com" in csv_content

    def test_export_v2_route(self, handler, mock_user_store):
        """V2 export route works the same as V1."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v2/users/me/export", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert "profile" in body

    @pytest.mark.usefixtures("_patch_unauth")
    def test_export_unauthenticated(self, handler):
        """Unauthenticated user gets 401."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {}, mock_h)
        assert _status(result) == 401

    def test_export_no_user_store(self, handler_no_store):
        """Missing user store returns 503."""
        mock_h = _MockHTTPHandler("GET")
        result = handler_no_store.handle("/api/v1/privacy/export", {}, mock_h)
        # Auth passes (mocked), but store is None => 503
        assert _status(result) == 503

    def test_export_user_not_found(self, handler, mock_user_store):
        """Non-existent user returns 404."""
        mock_user_store.get_user_by_id.return_value = None
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {}, mock_h)
        assert _status(result) == 404

    def test_export_with_api_key(self, handler, mock_user_store):
        """Export includes API key metadata when present."""
        user = MockUser(
            api_key_prefix="ara_test",
            api_key_created_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
            api_key_expires_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        mock_user_store.get_user_by_id.return_value = user
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {}, mock_h)
        body = _body(result)
        assert body["api_key"]["prefix"] == "ara_test"

    def test_export_with_org(self, handler, mock_user_store):
        """Export includes organization membership when present."""
        user = MockUser(org_id="org-001")
        mock_user_store.get_user_by_id.return_value = user
        mock_user_store.get_organization_by_id.return_value = MockOrg()
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {}, mock_h)
        body = _body(result)
        assert body["organization"]["name"] == "Test Org"
        assert body["organization"]["tier"] == "professional"

    def test_export_with_oauth_providers(self, handler, mock_user_store):
        """Export includes OAuth provider links when present."""
        mock_user_store.get_user_oauth_providers.return_value = [
            {"provider": "google", "linked_at": "2025-01-01"},
            {"provider": "github", "linked_at": "2025-02-01"},
        ]
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {}, mock_h)
        body = _body(result)
        assert len(body["oauth_providers"]) == 2
        assert body["oauth_providers"][0]["provider"] == "google"

    def test_export_with_preferences(self, handler, mock_user_store):
        """Export includes user preferences when present."""
        mock_user_store.get_user_preferences.return_value = {"theme": "dark"}
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {}, mock_h)
        body = _body(result)
        assert body["preferences"]["theme"] == "dark"

    def test_export_with_consent_records(self, handler, mock_user_store):
        """Export includes consent records from preferences."""
        mock_user_store.get_user_preferences.return_value = {
            "consent": {"marketing": True, "analytics": False},
        }
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {}, mock_h)
        body = _body(result)
        assert body["consent_records"]["marketing"] is True

    def test_export_with_audit_log(self, handler, mock_user_store):
        """Export includes audit log entries."""
        mock_user_store.get_audit_log.return_value = [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "action": "login",
                "resource_type": "session",
                "resource_id": "sess-001",
            }
        ]
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {}, mock_h)
        body = _body(result)
        assert len(body["audit_log"]) == 1
        assert body["audit_log"][0]["action"] == "login"

    def test_export_with_usage_summary(self, handler, mock_user_store):
        """Export includes usage summary when in an org."""
        user = MockUser(org_id="org-001")
        mock_user_store.get_user_by_id.return_value = user
        mock_user_store.get_usage_summary.return_value = {"debates": 10, "api_calls": 100}
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {}, mock_h)
        body = _body(result)
        assert body["usage_summary"]["debates"] == 10

    def test_export_profile_fields(self, handler, mock_user_store):
        """Export profile contains all expected fields."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {}, mock_h)
        body = _body(result)
        profile = body["profile"]
        assert profile["id"] == "user-001"
        assert profile["email"] == "test@example.com"
        assert profile["name"] == "Test User"
        assert profile["role"] == "member"
        assert profile["is_active"] is True
        assert profile["email_verified"] is True
        assert profile["mfa_enabled"] is False

    def test_export_profile_dates_are_iso(self, handler, mock_user_store):
        """Export profile dates are in ISO format."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {}, mock_h)
        body = _body(result)
        profile = body["profile"]
        # These should be parseable ISO dates
        datetime.fromisoformat(profile["created_at"])
        datetime.fromisoformat(profile["updated_at"])

    def test_export_metadata_timestamp(self, handler, mock_user_store):
        """Export metadata exported_at is a valid ISO timestamp."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {}, mock_h)
        body = _body(result)
        datetime.fromisoformat(body["_export_metadata"]["exported_at"])

    def test_export_no_api_key_when_absent(self, handler, mock_user_store):
        """Export omits api_key when user has no API key."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {}, mock_h)
        body = _body(result)
        assert "api_key" not in body

    def test_export_no_org_when_absent(self, handler, mock_user_store):
        """Export omits organization when user has no org."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {}, mock_h)
        body = _body(result)
        assert "organization" not in body

    def test_export_csv_with_org(self, handler, mock_user_store):
        """CSV export includes Organization section."""
        user = MockUser(org_id="org-001")
        mock_user_store.get_user_by_id.return_value = user
        mock_user_store.get_organization_by_id.return_value = MockOrg()
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {"format": "csv"}, mock_h)
        csv_content = result.body.decode("utf-8")
        assert "Organization" in csv_content
        assert "Test Org" in csv_content

    def test_export_csv_with_oauth(self, handler, mock_user_store):
        """CSV export includes OAuth Providers section."""
        mock_user_store.get_user_oauth_providers.return_value = [
            {"provider": "google", "linked_at": "2025-01-01"},
        ]
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/export", {"format": "csv"}, mock_h)
        csv_content = result.body.decode("utf-8")
        assert "OAuth Providers" in csv_content
        assert "google" in csv_content


# ============================================================================
# Data Inventory Endpoint
# ============================================================================


class TestDataInventory:
    """Tests for GET /api/v1/privacy/data-inventory."""

    def test_inventory_success(self, handler):
        """Data inventory returns 200 with categories."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/data-inventory", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert "categories" in body
        assert len(body["categories"]) == 5

    def test_inventory_category_names(self, handler):
        """Data inventory includes all 5 standard categories."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/data-inventory", {}, mock_h)
        body = _body(result)
        names = [c["name"] for c in body["categories"]]
        assert "Identifiers" in names
        assert "Internet Activity" in names
        assert "Geolocation" in names
        assert "Professional Information" in names
        assert "Inferences" in names

    def test_inventory_categories_have_required_fields(self, handler):
        """Each category has name, examples, purpose, and retention."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/data-inventory", {}, mock_h)
        body = _body(result)
        for cat in body["categories"]:
            assert "name" in cat
            assert "examples" in cat
            assert "purpose" in cat
            assert "retention" in cat

    def test_inventory_third_party_sharing(self, handler):
        """Data inventory includes third_party_sharing info."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/data-inventory", {}, mock_h)
        body = _body(result)
        assert "llm_providers" in body["third_party_sharing"]
        assert "analytics" in body["third_party_sharing"]

    def test_inventory_data_sold_false(self, handler):
        """Data inventory reports data_sold as False."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/data-inventory", {}, mock_h)
        body = _body(result)
        assert body["data_sold"] is False

    def test_inventory_opt_out_available(self, handler):
        """Data inventory reports opt_out_available as True."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/data-inventory", {}, mock_h)
        body = _body(result)
        assert body["opt_out_available"] is True

    def test_inventory_v2_route(self, handler):
        """V2 data inventory route works."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v2/users/me/data-inventory", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert "categories" in body

    @pytest.mark.usefixtures("_patch_unauth")
    def test_inventory_unauthenticated(self, handler):
        """Unauthenticated user gets 401."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/data-inventory", {}, mock_h)
        assert _status(result) == 401

    def test_inventory_llm_providers_list(self, handler):
        """LLM providers section lists known providers."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/data-inventory", {}, mock_h)
        body = _body(result)
        providers = body["third_party_sharing"]["llm_providers"]
        assert "Anthropic" in providers["recipients"]
        assert "OpenAI" in providers["recipients"]
        assert "Mistral" in providers["recipients"]


# ============================================================================
# Account Deletion Endpoint
# ============================================================================


class TestAccountDeletion:
    """Tests for DELETE /api/v1/privacy/account and /api/v2/users/me."""

    def test_delete_success(self, handler, mock_user_store):
        """Successful deletion returns 200 with deletion details."""
        mock_h = _MockHTTPHandler(
            "DELETE",
            body={
                "password": "correct-password",
                "confirm": True,
                "reason": "Closing account",
            },
        )
        result = handler.handle("/api/v1/privacy/account", {}, mock_h, method="DELETE")
        assert _status(result) == 200
        body = _body(result)
        assert body["message"] == "Account deleted successfully"
        assert "deletion_id" in body
        assert "data_deleted" in body
        assert "retention_note" in body

    def test_delete_v2_route(self, handler, mock_user_store):
        """V2 delete route works."""
        mock_h = _MockHTTPHandler(
            "DELETE",
            body={
                "password": "correct-password",
                "confirm": True,
            },
        )
        result = handler.handle("/api/v2/users/me", {}, mock_h, method="DELETE")
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_unauth")
    def test_delete_unauthenticated(self, handler):
        """Unauthenticated user gets 401."""
        mock_h = _MockHTTPHandler(
            "DELETE",
            body={
                "password": "pass",
                "confirm": True,
            },
        )
        result = handler.handle("/api/v1/privacy/account", {}, mock_h, method="DELETE")
        assert _status(result) == 401

    def test_delete_no_user_store(self, handler_no_store):
        """Missing user store returns 503."""
        mock_h = _MockHTTPHandler(
            "DELETE",
            body={
                "password": "pass",
                "confirm": True,
            },
        )
        result = handler_no_store.handle("/api/v1/privacy/account", {}, mock_h, method="DELETE")
        assert _status(result) == 503

    def test_delete_invalid_json_body(self, handler, mock_user_store):
        """Invalid JSON body returns 400."""
        mock_h = _MockHTTPHandler("DELETE")
        mock_h.rfile.read.return_value = b"not json"
        mock_h.headers = {"Content-Length": "8"}
        result = handler.handle("/api/v1/privacy/account", {}, mock_h, method="DELETE")
        assert _status(result) == 400

    def test_delete_missing_confirm(self, handler, mock_user_store):
        """Missing confirm flag returns 400."""
        mock_h = _MockHTTPHandler(
            "DELETE",
            body={
                "password": "correct-password",
            },
        )
        result = handler.handle("/api/v1/privacy/account", {}, mock_h, method="DELETE")
        assert _status(result) == 400
        body = _body(result)
        assert "confirm" in body.get("error", "").lower()

    def test_delete_confirm_false(self, handler, mock_user_store):
        """confirm=false returns 400."""
        mock_h = _MockHTTPHandler(
            "DELETE",
            body={
                "password": "correct-password",
                "confirm": False,
            },
        )
        result = handler.handle("/api/v1/privacy/account", {}, mock_h, method="DELETE")
        assert _status(result) == 400

    def test_delete_user_not_found(self, handler, mock_user_store):
        """Non-existent user returns 404."""
        mock_user_store.get_user_by_id.return_value = None
        mock_h = _MockHTTPHandler(
            "DELETE",
            body={
                "password": "pass",
                "confirm": True,
            },
        )
        result = handler.handle("/api/v1/privacy/account", {}, mock_h, method="DELETE")
        assert _status(result) == 404

    def test_delete_wrong_password(self, handler, mock_user_store):
        """Wrong password returns 401."""
        mock_h = _MockHTTPHandler(
            "DELETE",
            body={
                "password": "wrong-password",
                "confirm": True,
            },
        )
        result = handler.handle("/api/v1/privacy/account", {}, mock_h, method="DELETE")
        assert _status(result) == 401
        body = _body(result)
        assert "password" in body.get("error", "").lower()

    def test_delete_org_owner_with_members(self, handler, mock_user_store):
        """Org owner with other members cannot delete account."""
        user = MockUser(org_id="org-001")
        mock_user_store.get_user_by_id.return_value = user
        mock_user_store.get_organization_by_id.return_value = MockOrg(owner_id="user-001")
        mock_user_store.get_org_members.return_value = [
            {"id": "user-001"},
            {"id": "user-002"},
        ]
        mock_h = _MockHTTPHandler(
            "DELETE",
            body={
                "password": "correct-password",
                "confirm": True,
            },
        )
        result = handler.handle("/api/v1/privacy/account", {}, mock_h, method="DELETE")
        assert _status(result) == 400
        body = _body(result)
        assert "transfer ownership" in body.get("error", "").lower()

    def test_delete_org_owner_sole_member(self, handler, mock_user_store):
        """Org owner as sole member can delete account."""
        user = MockUser(org_id="org-001")
        mock_user_store.get_user_by_id.return_value = user
        mock_user_store.get_organization_by_id.return_value = MockOrg(owner_id="user-001")
        mock_user_store.get_org_members.return_value = [{"id": "user-001"}]
        mock_h = _MockHTTPHandler(
            "DELETE",
            body={
                "password": "correct-password",
                "confirm": True,
            },
        )
        result = handler.handle("/api/v1/privacy/account", {}, mock_h, method="DELETE")
        assert _status(result) == 200

    def test_delete_non_owner_in_org(self, handler, mock_user_store):
        """Non-owner member of org can delete their account."""
        user = MockUser(org_id="org-001")
        mock_user_store.get_user_by_id.return_value = user
        mock_user_store.get_organization_by_id.return_value = MockOrg(owner_id="user-999")
        mock_h = _MockHTTPHandler(
            "DELETE",
            body={
                "password": "correct-password",
                "confirm": True,
            },
        )
        result = handler.handle("/api/v1/privacy/account", {}, mock_h, method="DELETE")
        assert _status(result) == 200

    def test_delete_logs_audit_event(self, handler, mock_user_store):
        """Deletion logs audit events."""
        mock_h = _MockHTTPHandler(
            "DELETE",
            body={
                "password": "correct-password",
                "confirm": True,
            },
        )
        handler.handle("/api/v1/privacy/account", {}, mock_h, method="DELETE")
        assert mock_user_store.log_audit_event.call_count >= 1

    def test_delete_data_deleted_includes_preferences(self, handler, mock_user_store):
        """Deletion result includes preferences in data_deleted."""
        mock_h = _MockHTTPHandler(
            "DELETE",
            body={
                "password": "correct-password",
                "confirm": True,
            },
        )
        result = handler.handle("/api/v1/privacy/account", {}, mock_h, method="DELETE")
        body = _body(result)
        assert "preferences" in body["data_deleted"]

    def test_delete_data_deleted_includes_profile(self, handler, mock_user_store):
        """Deletion result includes profile in data_deleted."""
        mock_h = _MockHTTPHandler(
            "DELETE",
            body={
                "password": "correct-password",
                "confirm": True,
            },
        )
        result = handler.handle("/api/v1/privacy/account", {}, mock_h, method="DELETE")
        body = _body(result)
        assert "profile" in body["data_deleted"]


# ============================================================================
# Perform Account Deletion Helper
# ============================================================================


class TestPerformAccountDeletion:
    """Tests for _perform_account_deletion helper."""

    def test_unlinks_oauth_providers(self, handler, mock_user_store):
        """Deletion unlinks OAuth providers."""
        user = MockUser()
        mock_user_store.get_user_oauth_providers.return_value = [
            {"provider": "google"},
            {"provider": "github"},
        ]
        result = handler._perform_account_deletion(mock_user_store, user, "test")
        assert result["success"] is True
        assert "oauth_links:2" in result["data_deleted"]
        assert mock_user_store.unlink_oauth_provider.call_count == 2

    def test_clears_api_key(self, handler, mock_user_store):
        """Deletion clears API key data."""
        user = MockUser(api_key_hash="some-hash")
        mock_user_store.get_user_oauth_providers.return_value = []
        result = handler._perform_account_deletion(mock_user_store, user, "test")
        assert "api_key" in result["data_deleted"]

    def test_clears_mfa(self, handler, mock_user_store):
        """Deletion clears MFA data."""
        user = MockUser(mfa_enabled=True)
        mock_user_store.get_user_oauth_providers.return_value = []
        result = handler._perform_account_deletion(mock_user_store, user, "test")
        assert "mfa_data" in result["data_deleted"]

    def test_removes_from_org(self, handler, mock_user_store):
        """Deletion removes user from org."""
        user = MockUser(org_id="org-001")
        mock_user_store.get_user_oauth_providers.return_value = []
        result = handler._perform_account_deletion(mock_user_store, user, "test")
        assert "org_membership" in result["data_deleted"]
        mock_user_store.remove_user_from_org.assert_called_with("user-001")

    def test_anonymizes_profile(self, handler, mock_user_store):
        """Deletion anonymizes the user profile."""
        user = MockUser()
        mock_user_store.get_user_oauth_providers.return_value = []
        result = handler._perform_account_deletion(mock_user_store, user, "test")
        assert "profile" in result["data_deleted"]
        # Check that update_user was called with anonymized data
        calls = mock_user_store.update_user.call_args_list
        # Last update_user call should have the anonymized email
        last_call = calls[-1]
        assert last_call.kwargs.get("name") == "[Deleted User]"
        assert last_call.kwargs.get("is_active") is False
        assert "deleted_" in last_call.kwargs.get("email", "")

    def test_deletion_generates_uuid(self, handler, mock_user_store):
        """Deletion generates a unique deletion_id."""
        user = MockUser()
        mock_user_store.get_user_oauth_providers.return_value = []
        result = handler._perform_account_deletion(mock_user_store, user, "test")
        assert result["deletion_id"] is not None
        assert len(result["deletion_id"]) == 36  # UUID format

    def test_deletion_handles_exception(self, handler, mock_user_store):
        """Deletion handles exceptions gracefully."""
        user = MockUser()
        mock_user_store.get_user_oauth_providers.side_effect = RuntimeError("DB error")
        result = handler._perform_account_deletion(mock_user_store, user, "test")
        assert result["success"] is False
        assert "Account deletion failed" in result["error"]

    def test_deletion_skips_api_key_when_none(self, handler, mock_user_store):
        """No api_key in data_deleted when user has no API key hash."""
        user = MockUser(api_key_hash=None)
        mock_user_store.get_user_oauth_providers.return_value = []
        result = handler._perform_account_deletion(mock_user_store, user, "test")
        assert "api_key" not in result["data_deleted"]

    def test_deletion_skips_mfa_when_disabled(self, handler, mock_user_store):
        """No mfa_data in data_deleted when MFA is not enabled."""
        user = MockUser(mfa_enabled=False)
        mock_user_store.get_user_oauth_providers.return_value = []
        result = handler._perform_account_deletion(mock_user_store, user, "test")
        assert "mfa_data" not in result["data_deleted"]

    def test_deletion_skips_org_when_not_member(self, handler, mock_user_store):
        """No org_membership in data_deleted when user has no org."""
        user = MockUser(org_id=None)
        mock_user_store.get_user_oauth_providers.return_value = []
        result = handler._perform_account_deletion(mock_user_store, user, "test")
        assert "org_membership" not in result["data_deleted"]


# ============================================================================
# Privacy Preferences - GET
# ============================================================================


class TestGetPreferences:
    """Tests for GET /api/v1/privacy/preferences."""

    def test_get_defaults(self, handler, mock_user_store):
        """Default preferences when none set."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/preferences", {}, mock_h, method="GET")
        assert _status(result) == 200
        body = _body(result)
        assert body["do_not_sell"] is False
        assert body["marketing_opt_out"] is False
        assert body["analytics_opt_out"] is False
        assert body["third_party_sharing"] is True

    def test_get_existing_preferences(self, handler, mock_user_store):
        """Returns stored privacy preferences."""
        mock_user_store.get_user_preferences.return_value = {
            "privacy": {
                "do_not_sell": True,
                "marketing_opt_out": True,
                "analytics_opt_out": False,
                "third_party_sharing": False,
            }
        }
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/preferences", {}, mock_h, method="GET")
        body = _body(result)
        assert body["do_not_sell"] is True
        assert body["marketing_opt_out"] is True
        assert body["analytics_opt_out"] is False
        assert body["third_party_sharing"] is False

    @pytest.mark.usefixtures("_patch_unauth")
    def test_get_preferences_unauthenticated(self, handler):
        """Unauthenticated user gets 401."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/preferences", {}, mock_h, method="GET")
        assert _status(result) == 401

    def test_get_preferences_no_store(self, handler_no_store):
        """Missing user store returns 503."""
        mock_h = _MockHTTPHandler("GET")
        result = handler_no_store.handle("/api/v1/privacy/preferences", {}, mock_h, method="GET")
        assert _status(result) == 503

    def test_get_preferences_none_prefs(self, handler, mock_user_store):
        """Handles None return from get_user_preferences."""
        mock_user_store.get_user_preferences.return_value = None
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/preferences", {}, mock_h, method="GET")
        assert _status(result) == 200
        body = _body(result)
        assert body["do_not_sell"] is False

    def test_get_preferences_partial_settings(self, handler, mock_user_store):
        """Handles partial privacy preferences (some missing)."""
        mock_user_store.get_user_preferences.return_value = {
            "privacy": {"do_not_sell": True},
        }
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/preferences", {}, mock_h, method="GET")
        body = _body(result)
        assert body["do_not_sell"] is True
        assert body["marketing_opt_out"] is False  # Default


# ============================================================================
# Privacy Preferences - POST (Update)
# ============================================================================


class TestUpdatePreferences:
    """Tests for POST /api/v1/privacy/preferences."""

    def test_update_success(self, handler, mock_user_store):
        """Successfully update privacy preferences."""
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "do_not_sell": True,
                "marketing_opt_out": True,
            },
        )
        result = handler.handle("/api/v1/privacy/preferences", {}, mock_h, method="POST")
        assert _status(result) == 200
        body = _body(result)
        assert body["message"] == "Privacy preferences updated"
        assert body["preferences"]["do_not_sell"] is True
        assert body["preferences"]["marketing_opt_out"] is True

    def test_update_logs_audit_event(self, handler, mock_user_store):
        """Update logs an audit event."""
        mock_h = _MockHTTPHandler("POST", body={"do_not_sell": True})
        handler.handle("/api/v1/privacy/preferences", {}, mock_h, method="POST")
        mock_user_store.log_audit_event.assert_called_once()
        call_kwargs = mock_user_store.log_audit_event.call_args[1]
        assert call_kwargs["action"] == "privacy_preferences_updated"

    def test_update_saves_to_store(self, handler, mock_user_store):
        """Update saves preferences to user store."""
        mock_h = _MockHTTPHandler("POST", body={"analytics_opt_out": True})
        handler.handle("/api/v1/privacy/preferences", {}, mock_h, method="POST")
        mock_user_store.set_user_preferences.assert_called_once()

    def test_update_preserves_existing_preferences(self, handler, mock_user_store):
        """Update preserves non-privacy preferences."""
        mock_user_store.get_user_preferences.return_value = {
            "theme": "dark",
            "privacy": {"do_not_sell": False},
        }
        mock_h = _MockHTTPHandler("POST", body={"do_not_sell": True})
        handler.handle("/api/v1/privacy/preferences", {}, mock_h, method="POST")
        saved = mock_user_store.set_user_preferences.call_args[0][1]
        assert saved["theme"] == "dark"
        assert saved["privacy"]["do_not_sell"] is True

    @pytest.mark.usefixtures("_patch_unauth")
    def test_update_unauthenticated(self, handler):
        """Unauthenticated user gets 401."""
        mock_h = _MockHTTPHandler("POST", body={"do_not_sell": True})
        result = handler.handle("/api/v1/privacy/preferences", {}, mock_h, method="POST")
        assert _status(result) == 401

    def test_update_no_store(self, handler_no_store):
        """Missing user store returns 503."""
        mock_h = _MockHTTPHandler("POST", body={"do_not_sell": True})
        result = handler_no_store.handle("/api/v1/privacy/preferences", {}, mock_h, method="POST")
        assert _status(result) == 503

    def test_update_invalid_json(self, handler, mock_user_store):
        """Invalid JSON body returns 400."""
        mock_h = _MockHTTPHandler("POST")
        mock_h.rfile.read.return_value = b"not json"
        mock_h.headers = {"Content-Length": "8"}
        result = handler.handle("/api/v1/privacy/preferences", {}, mock_h, method="POST")
        assert _status(result) == 400

    def test_update_boolean_coercion(self, handler, mock_user_store):
        """Non-boolean values are coerced to bool."""
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "do_not_sell": 1,
                "marketing_opt_out": "",
            },
        )
        result = handler.handle("/api/v1/privacy/preferences", {}, mock_h, method="POST")
        body = _body(result)
        assert body["preferences"]["do_not_sell"] is True
        assert body["preferences"]["marketing_opt_out"] is False

    def test_update_ignores_unknown_fields(self, handler, mock_user_store):
        """Unknown fields in body are ignored."""
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "do_not_sell": True,
                "unknown_field": "value",
            },
        )
        result = handler.handle("/api/v1/privacy/preferences", {}, mock_h, method="POST")
        body = _body(result)
        assert "unknown_field" not in body["preferences"]

    def test_update_all_fields(self, handler, mock_user_store):
        """All four preference fields can be updated at once."""
        mock_h = _MockHTTPHandler(
            "POST",
            body={
                "do_not_sell": True,
                "marketing_opt_out": True,
                "analytics_opt_out": True,
                "third_party_sharing": False,
            },
        )
        result = handler.handle("/api/v1/privacy/preferences", {}, mock_h, method="POST")
        body = _body(result)
        prefs = body["preferences"]
        assert prefs["do_not_sell"] is True
        assert prefs["marketing_opt_out"] is True
        assert prefs["analytics_opt_out"] is True
        assert prefs["third_party_sharing"] is False

    def test_update_empty_body(self, handler, mock_user_store):
        """Empty body is valid (no changes made)."""
        mock_h = _MockHTTPHandler("POST", body={})
        result = handler.handle("/api/v1/privacy/preferences", {}, mock_h, method="POST")
        assert _status(result) == 200


# ============================================================================
# Method Not Allowed
# ============================================================================


class TestMethodNotAllowed:
    """Tests for method-not-allowed scenarios."""

    def test_post_to_export(self, handler):
        """POST to export endpoint returns 405."""
        mock_h = _MockHTTPHandler("POST", body={})
        result = handler.handle("/api/v1/privacy/export", {}, mock_h, method="POST")
        assert _status(result) == 405

    def test_post_to_data_inventory(self, handler):
        """POST to data-inventory endpoint returns 405."""
        mock_h = _MockHTTPHandler("POST", body={})
        result = handler.handle("/api/v1/privacy/data-inventory", {}, mock_h, method="POST")
        assert _status(result) == 405

    def test_get_to_account_delete(self, handler):
        """GET to account endpoint returns 405."""
        mock_h = _MockHTTPHandler("GET")
        result = handler.handle("/api/v1/privacy/account", {}, mock_h, method="GET")
        assert _status(result) == 405

    def test_put_to_preferences(self, handler):
        """PUT to preferences endpoint returns 405."""
        mock_h = _MockHTTPHandler("PUT", body={})
        result = handler.handle("/api/v1/privacy/preferences", {}, mock_h, method="PUT")
        assert _status(result) == 405


# ============================================================================
# Hash for Audit Helper
# ============================================================================


class TestHashForAudit:
    """Tests for _hash_for_audit helper."""

    def test_returns_16_char_hex(self, handler):
        """Hash is a 16-character hex string."""
        result = handler._hash_for_audit("test@example.com")
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_consistent_hashing(self, handler):
        """Same input produces same hash."""
        h1 = handler._hash_for_audit("test@example.com")
        h2 = handler._hash_for_audit("test@example.com")
        assert h1 == h2

    def test_different_inputs_different_hashes(self, handler):
        """Different inputs produce different hashes."""
        h1 = handler._hash_for_audit("user1@example.com")
        h2 = handler._hash_for_audit("user2@example.com")
        assert h1 != h2

    def test_matches_sha256_prefix(self, handler):
        """Hash matches first 16 chars of SHA-256."""
        value = "test@example.com"
        expected = hashlib.sha256(value.encode()).hexdigest()[:16]
        assert handler._hash_for_audit(value) == expected


# ============================================================================
# Collect User Data Helper
# ============================================================================


class TestCollectUserData:
    """Tests for _collect_user_data helper."""

    def test_basic_profile(self, handler, mock_user_store):
        """Collects basic profile data."""
        user = MockUser()
        data = handler._collect_user_data(mock_user_store, user)
        assert data["profile"]["id"] == "user-001"
        assert data["profile"]["email"] == "test@example.com"

    def test_no_api_key(self, handler, mock_user_store):
        """No api_key section when user has no API key."""
        user = MockUser(api_key_prefix=None)
        data = handler._collect_user_data(mock_user_store, user)
        assert "api_key" not in data

    def test_with_api_key(self, handler, mock_user_store):
        """Includes api_key section when present."""
        user = MockUser(
            api_key_prefix="ara_test",
            api_key_created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        data = handler._collect_user_data(mock_user_store, user)
        assert data["api_key"]["prefix"] == "ara_test"

    def test_no_org(self, handler, mock_user_store):
        """No organization section when user has no org."""
        user = MockUser(org_id=None)
        data = handler._collect_user_data(mock_user_store, user)
        assert "organization" not in data

    def test_null_dates_handled(self, handler, mock_user_store):
        """Null dates are handled (set to None in profile)."""
        user = MockUser(created_at=None, updated_at=None, last_login_at=None)
        data = handler._collect_user_data(mock_user_store, user)
        assert data["profile"]["created_at"] is None
        assert data["profile"]["updated_at"] is None
        assert data["profile"]["last_login_at"] is None

    def test_no_usage_without_org(self, handler, mock_user_store):
        """No usage_summary when user has no org."""
        user = MockUser(org_id=None)
        data = handler._collect_user_data(mock_user_store, user)
        assert "usage_summary" not in data


# ============================================================================
# Constructor and Resource Type
# ============================================================================


class TestHandlerInit:
    """Tests for handler initialization."""

    def test_default_context(self):
        """Handler can be created without context."""
        h = PrivacyHandler()
        assert h.ctx == {}

    def test_custom_context(self):
        """Handler accepts custom context."""
        ctx = {"user_store": MagicMock()}
        h = PrivacyHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_resource_type(self, handler):
        """Resource type is 'privacy'."""
        assert handler.RESOURCE_TYPE == "privacy"

    def test_routes_list(self, handler):
        """ROUTES contains all expected paths."""
        assert len(handler.ROUTES) == 9
