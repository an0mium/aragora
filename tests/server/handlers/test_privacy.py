"""
Tests for aragora.server.handlers.privacy - Privacy API handler.

Tests cover:
- Data export (GDPR Article 15)
- Data inventory (CCPA disclosure)
- Account deletion (GDPR Article 17)
- Privacy preferences (CCPA Do Not Sell)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Rate Limit Bypass for Testing
# ===========================================================================


from aragora.server.handlers.privacy import PrivacyHandler


def _always_allowed(key: str) -> bool:
    """Always allow requests for testing."""
    return True


@pytest.fixture(autouse=True)
def disable_rate_limits():
    """Disable rate limits for all tests in this module."""
    import sys

    rl_module = sys.modules["aragora.server.handlers.utils.rate_limit"]

    # Patch all existing limiters to always allow
    original_is_allowed = {}
    for name, limiter in rl_module._limiters.items():
        original_is_allowed[name] = limiter.is_allowed
        limiter.is_allowed = _always_allowed

    yield

    # Restore original is_allowed methods
    for name, original in original_is_allowed.items():
        if name in rl_module._limiters:
            rl_module._limiters[name].is_allowed = original


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockUser:
    """Mock user for testing."""

    id: str = "user-123"
    email: str = "test@example.com"
    name: str = "Test User"
    org_id: str | None = "org-123"
    role: str = "member"
    is_active: bool = True
    email_verified: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_login_at: datetime | None = None
    mfa_enabled: bool = False
    mfa_secret: str | None = None
    mfa_backup_codes: str | None = None
    api_key_hash: str | None = None
    api_key_prefix: str | None = "ara_1234"
    api_key_created_at: datetime | None = None
    api_key_expires_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "org_id": self.org_id,
            "role": self.role,
            "is_active": self.is_active,
            "mfa_enabled": self.mfa_enabled,
        }

    def verify_password(self, password: str) -> bool:
        return password == "correct_password"


@dataclass
class MockOrganization:
    """Mock organization for testing."""

    id: str = "org-123"
    name: str = "Test Org"
    slug: str = "test-org"
    owner_id: str = "user-123"
    tier: Any = field(default_factory=lambda: MagicMock(value="starter"))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "slug": self.slug, "owner_id": self.owner_id}


@dataclass
class MockAuthContext:
    """Mock authentication context."""

    is_authenticated: bool = True
    user_id: str = "user-123"
    email: str = "test@example.com"
    org_id: str | None = "org-123"
    role: str = "member"


class MockUserStore:
    """Mock user store for testing."""

    def __init__(self):
        self.users: dict[str, MockUser] = {}
        self.orgs: dict[str, MockOrganization] = {}
        self.oauth_providers: dict[str, list[dict]] = {}
        self.preferences: dict[str, dict] = {}
        self.audit_log: list[dict] = []
        self.updates: list[dict] = []

    def get_user_by_id(self, user_id: str) -> MockUser | None:
        return self.users.get(user_id)

    def get_organization_by_id(self, org_id: str) -> MockOrganization | None:
        return self.orgs.get(org_id)

    def get_user_oauth_providers(self, user_id: str) -> list[dict]:
        return self.oauth_providers.get(user_id, [])

    def get_user_preferences(self, user_id: str) -> dict | None:
        return self.preferences.get(user_id)

    def set_user_preferences(self, user_id: str, prefs: dict) -> bool:
        self.preferences[user_id] = prefs
        return True

    def get_audit_log(self, **kwargs) -> list[dict]:
        return self.audit_log

    def get_usage_summary(self, org_id: str) -> dict:
        return {"debates_used": 10, "debates_limit": 100}

    def log_audit_event(self, **kwargs) -> int:
        self.audit_log.append(kwargs)
        return len(self.audit_log)

    def update_user(self, user_id: str, **kwargs) -> bool:
        self.updates.append({"user_id": user_id, **kwargs})
        if user_id in self.users:
            for key, value in kwargs.items():
                if hasattr(self.users[user_id], key):
                    setattr(self.users[user_id], key, value)
        return True

    def unlink_oauth_provider(self, user_id: str, provider: str) -> bool:
        if user_id in self.oauth_providers:
            self.oauth_providers[user_id] = [
                p for p in self.oauth_providers[user_id] if p["provider"] != provider
            ]
        return True

    def remove_user_from_org(self, user_id: str) -> bool:
        if user_id in self.users:
            self.users[user_id].org_id = None
        return True

    def get_org_members(self, org_id: str) -> list[MockUser]:
        return [u for u in self.users.values() if u.org_id == org_id]


def get_status(result) -> int:
    """Extract status code from HandlerResult or tuple."""
    if hasattr(result, "status_code"):
        return result.status_code
    return result[1]


def get_body(result) -> dict:
    """Extract body from HandlerResult or tuple."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, bytes):
            return json.loads(body.decode("utf-8"))
        return json.loads(body)
    body = result[0]
    if isinstance(body, dict):
        return body
    if isinstance(body, bytes):
        return json.loads(body.decode("utf-8"))
    return json.loads(body)


def create_mock_handler(method: str = "GET", body: dict | None = None) -> MagicMock:
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = method
    handler.headers = {"Content-Type": "application/json"}

    if body:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.rfile = BytesIO(body_bytes)
        handler.headers["Content-Length"] = str(len(body_bytes))
    else:
        handler.rfile = BytesIO(b"")
        handler.headers["Content-Length"] = "0"

    return handler


@pytest.fixture
def user_store():
    """Create a mock user store with test data."""
    store = MockUserStore()
    store.users["user-123"] = MockUser()
    store.orgs["org-123"] = MockOrganization()
    store.oauth_providers["user-123"] = [
        {"provider": "google", "linked_at": "2026-01-01T00:00:00Z"}
    ]
    store.preferences["user-123"] = {
        "theme": "dark",
        "privacy": {"do_not_sell": False, "marketing_opt_out": True},
    }
    store.audit_log = [
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "login",
            "resource_type": "session",
            "resource_id": "sess-1",
        }
    ]
    return store


@pytest.fixture
def handler(user_store):
    """Create a privacy handler with context."""
    ctx = {"user_store": user_store}
    return PrivacyHandler(ctx)


@pytest.fixture
def auth_context():
    """Create a mock auth context."""
    return MockAuthContext()


# ===========================================================================
# can_handle Tests
# ===========================================================================


class TestCanHandle:
    """Tests for route matching."""

    def test_handles_export_route(self, handler):
        assert handler.can_handle("/api/privacy/export") is True

    def test_handles_data_inventory_route(self, handler):
        assert handler.can_handle("/api/privacy/data-inventory") is True

    def test_handles_account_route(self, handler):
        assert handler.can_handle("/api/privacy/account") is True

    def test_handles_preferences_route(self, handler):
        assert handler.can_handle("/api/privacy/preferences") is True

    def test_handles_v2_export_route(self, handler):
        assert handler.can_handle("/api/v2/users/me/export") is True

    def test_handles_v2_user_route(self, handler):
        assert handler.can_handle("/api/v2/users/me") is True

    def test_does_not_handle_other_routes(self, handler):
        assert handler.can_handle("/api/auth/login") is False
        assert handler.can_handle("/api/users/123") is False


# ===========================================================================
# Data Export Tests
# ===========================================================================


class TestDataExport:
    """Tests for data export functionality."""

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_export_requires_authentication(self, mock_extract, handler, user_store):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = handler.handle("/api/privacy/export", {}, mock_handler, "GET")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_export_returns_user_data(self, mock_extract, handler, user_store, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = handler.handle("/api/privacy/export", {}, mock_handler, "GET")

        assert get_status(result) == 200
        body = get_body(result)
        assert "profile" in body
        assert body["profile"]["email"] == "test@example.com"

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_export_includes_organization(self, mock_extract, handler, user_store, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = handler.handle("/api/privacy/export", {}, mock_handler, "GET")

        body = get_body(result)
        assert "organization" in body
        assert body["organization"]["name"] == "Test Org"

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_export_includes_oauth_providers(self, mock_extract, handler, user_store, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = handler.handle("/api/privacy/export", {}, mock_handler, "GET")

        body = get_body(result)
        assert "oauth_providers" in body
        assert len(body["oauth_providers"]) == 1
        assert body["oauth_providers"][0]["provider"] == "google"

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_export_includes_preferences(self, mock_extract, handler, user_store, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = handler.handle("/api/privacy/export", {}, mock_handler, "GET")

        body = get_body(result)
        assert "preferences" in body
        assert body["preferences"]["theme"] == "dark"

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_export_includes_audit_log(self, mock_extract, handler, user_store, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = handler.handle("/api/privacy/export", {}, mock_handler, "GET")

        body = get_body(result)
        assert "audit_log" in body
        assert len(body["audit_log"]) > 0

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_export_includes_metadata(self, mock_extract, handler, user_store, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = handler.handle("/api/privacy/export", {}, mock_handler, "GET")

        body = get_body(result)
        assert "_export_metadata" in body
        assert body["_export_metadata"]["data_controller"] == "Aragora"

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_export_csv_format(self, mock_extract, handler, user_store, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = handler.handle("/api/privacy/export", {"format": "csv"}, mock_handler, "GET")

        assert get_status(result) == 200
        # CSV returns bytes, not JSON
        assert isinstance(result[0], bytes)
        csv_content = result[0].decode("utf-8")
        assert "Profile" in csv_content
        assert "test@example.com" in csv_content

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_export_user_not_found(self, mock_extract, handler, user_store):
        mock_extract.return_value = MockAuthContext(user_id="nonexistent")
        mock_handler = create_mock_handler("GET")

        result = handler.handle("/api/privacy/export", {}, mock_handler, "GET")

        assert get_status(result) == 404


# ===========================================================================
# Data Inventory Tests
# ===========================================================================


class TestDataInventory:
    """Tests for data inventory functionality."""

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_inventory_requires_authentication(self, mock_extract, handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = handler.handle("/api/privacy/data-inventory", {}, mock_handler, "GET")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_inventory_returns_categories(self, mock_extract, handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = handler.handle("/api/privacy/data-inventory", {}, mock_handler, "GET")

        assert get_status(result) == 200
        body = get_body(result)
        assert "categories" in body
        assert len(body["categories"]) > 0

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_inventory_includes_third_party_sharing(self, mock_extract, handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = handler.handle("/api/privacy/data-inventory", {}, mock_handler, "GET")

        body = get_body(result)
        assert "third_party_sharing" in body
        assert "llm_providers" in body["third_party_sharing"]

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_inventory_shows_no_data_sold(self, mock_extract, handler, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = handler.handle("/api/privacy/data-inventory", {}, mock_handler, "GET")

        body = get_body(result)
        assert body["data_sold"] is False


# ===========================================================================
# Account Deletion Tests
# ===========================================================================


class TestAccountDeletion:
    """Tests for account deletion functionality."""

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_deletion_requires_authentication(self, mock_extract, handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("DELETE", {"password": "test", "confirm": True})

        result = handler.handle("/api/privacy/account", {}, mock_handler, "DELETE")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_deletion_requires_confirmation(self, mock_extract, handler, user_store, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("DELETE", {"password": "correct_password"})

        result = handler.handle("/api/privacy/account", {}, mock_handler, "DELETE")

        assert get_status(result) == 400
        body = get_body(result)
        assert "confirm" in body.get("error", "").lower()

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_deletion_requires_password(self, mock_extract, handler, user_store, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler(
            "DELETE", {"password": "wrong_password", "confirm": True}
        )

        result = handler.handle("/api/privacy/account", {}, mock_handler, "DELETE")

        assert get_status(result) == 401
        body = get_body(result)
        assert "password" in body.get("error", "").lower()

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_deletion_succeeds_with_correct_password(
        self, mock_extract, handler, user_store, auth_context
    ):
        # User is not org owner
        user_store.users["user-123"].role = "member"
        user_store.orgs["org-123"].owner_id = "other-user"
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler(
            "DELETE", {"password": "correct_password", "confirm": True, "reason": "Testing"}
        )

        result = handler.handle("/api/privacy/account", {}, mock_handler, "DELETE")

        assert get_status(result) == 200
        body = get_body(result)
        assert "deletion_id" in body
        assert "data_deleted" in body

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_deletion_blocks_org_owner_with_members(
        self, mock_extract, handler, user_store, auth_context
    ):
        # Add another member to the org
        user_store.users["user-456"] = MockUser(id="user-456", org_id="org-123")
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler(
            "DELETE", {"password": "correct_password", "confirm": True}
        )

        result = handler.handle("/api/privacy/account", {}, mock_handler, "DELETE")

        assert get_status(result) == 400
        body = get_body(result)
        assert "organization" in body.get("error", "").lower()

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_deletion_logs_audit_event(self, mock_extract, handler, user_store, auth_context):
        user_store.orgs["org-123"].owner_id = "other-user"
        mock_extract.return_value = auth_context
        initial_audit_count = len(user_store.audit_log)
        mock_handler = create_mock_handler(
            "DELETE", {"password": "correct_password", "confirm": True}
        )

        handler.handle("/api/privacy/account", {}, mock_handler, "DELETE")

        # Should have logged deletion request and completion
        assert len(user_store.audit_log) > initial_audit_count

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_deletion_anonymizes_user(self, mock_extract, handler, user_store, auth_context):
        user_store.orgs["org-123"].owner_id = "other-user"
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler(
            "DELETE", {"password": "correct_password", "confirm": True}
        )

        handler.handle("/api/privacy/account", {}, mock_handler, "DELETE")

        # Check that updates include anonymization
        email_updates = [u for u in user_store.updates if "email" in u]
        assert len(email_updates) > 0
        assert "deleted" in email_updates[-1]["email"]


# ===========================================================================
# Privacy Preferences Tests
# ===========================================================================


class TestPrivacyPreferences:
    """Tests for privacy preferences functionality."""

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_get_preferences_requires_auth(self, mock_extract, handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("GET")

        result = handler.handle("/api/privacy/preferences", {}, mock_handler, "GET")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_get_preferences_returns_defaults(
        self, mock_extract, handler, user_store, auth_context
    ):
        user_store.preferences["user-123"] = {}
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = handler.handle("/api/privacy/preferences", {}, mock_handler, "GET")

        assert get_status(result) == 200
        body = get_body(result)
        assert "do_not_sell" in body
        assert body["do_not_sell"] is False

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_get_preferences_returns_stored_values(
        self, mock_extract, handler, user_store, auth_context
    ):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = handler.handle("/api/privacy/preferences", {}, mock_handler, "GET")

        body = get_body(result)
        assert body["marketing_opt_out"] is True

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_update_preferences_requires_auth(self, mock_extract, handler):
        mock_extract.return_value = MockAuthContext(is_authenticated=False)
        mock_handler = create_mock_handler("POST", {"do_not_sell": True})

        result = handler.handle("/api/privacy/preferences", {}, mock_handler, "POST")

        assert get_status(result) == 401

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_update_preferences_sets_do_not_sell(
        self, mock_extract, handler, user_store, auth_context
    ):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("POST", {"do_not_sell": True})

        result = handler.handle("/api/privacy/preferences", {}, mock_handler, "POST")

        assert get_status(result) == 200
        body = get_body(result)
        assert body["preferences"]["do_not_sell"] is True

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_update_preferences_logs_audit(self, mock_extract, handler, user_store, auth_context):
        mock_extract.return_value = auth_context
        initial_count = len(user_store.audit_log)
        mock_handler = create_mock_handler("POST", {"analytics_opt_out": True})

        handler.handle("/api/privacy/preferences", {}, mock_handler, "POST")

        assert len(user_store.audit_log) > initial_count
        last_entry = user_store.audit_log[-1]
        assert last_entry["action"] == "privacy_preferences_updated"


# ===========================================================================
# V2 API Route Tests
# ===========================================================================


class TestV2Routes:
    """Tests for v2 API routes."""

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_v2_export_works(self, mock_extract, handler, user_store, auth_context):
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = handler.handle("/api/v2/users/me/export", {}, mock_handler, "GET")

        assert get_status(result) == 200

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_v2_delete_works(self, mock_extract, handler, user_store, auth_context):
        user_store.orgs["org-123"].owner_id = "other-user"
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler(
            "DELETE", {"password": "correct_password", "confirm": True}
        )

        result = handler.handle("/api/v2/users/me", {}, mock_handler, "DELETE")

        assert get_status(result) == 200


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_export_user_without_org(self, mock_extract, handler, user_store, auth_context):
        user_store.users["user-123"].org_id = None
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = handler.handle("/api/privacy/export", {}, mock_handler, "GET")

        assert get_status(result) == 200
        body = get_body(result)
        assert "organization" not in body

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_export_user_without_oauth(self, mock_extract, handler, user_store, auth_context):
        user_store.oauth_providers["user-123"] = []
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = handler.handle("/api/privacy/export", {}, mock_handler, "GET")

        assert get_status(result) == 200
        body = get_body(result)
        assert "oauth_providers" not in body

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_export_user_without_preferences(self, mock_extract, handler, user_store, auth_context):
        user_store.preferences["user-123"] = None
        mock_extract.return_value = auth_context
        mock_handler = create_mock_handler("GET")

        result = handler.handle("/api/privacy/export", {}, mock_handler, "GET")

        assert get_status(result) == 200
        body = get_body(result)
        assert "preferences" not in body

    def test_method_not_allowed(self, handler):
        mock_handler = create_mock_handler("PATCH")

        result = handler.handle("/api/privacy/export", {}, mock_handler, "PATCH")

        assert get_status(result) == 405

    @patch("aragora.server.handlers.privacy.extract_user_from_request")
    def test_no_user_store_returns_503(self, mock_extract, auth_context):
        mock_extract.return_value = auth_context
        handler_no_store = PrivacyHandler({})
        mock_handler = create_mock_handler("GET")

        result = handler_no_store.handle("/api/privacy/export", {}, mock_handler, "GET")

        assert get_status(result) == 503
