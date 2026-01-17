"""
Tests for PrivacyHandler endpoints.

Endpoints tested:
- GET /api/privacy/export - Export all user data (GDPR/CCPA)
- GET /api/privacy/data-inventory - Get data categories summary
- DELETE /api/privacy/account - Delete user account
- POST /api/privacy/preferences - Update privacy preferences
"""

import json
import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from aragora.server.handlers.privacy import PrivacyHandler
from aragora.server.handlers.base import clear_cache
from aragora.server.handlers.utils.rate_limit import _limiters, RateLimiter


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_user_store():
    """Create a mock user store."""
    store = Mock()
    store.get_user_by_id = Mock(return_value=None)
    store.get_user_preferences = Mock(return_value={})
    store.set_user_preferences = Mock()
    store.get_user_oauth_providers = Mock(return_value=[])
    store.get_organization_by_id = Mock(return_value=None)
    store.get_audit_log = Mock(return_value=[])
    store.get_usage_summary = Mock(return_value=None)
    store.log_audit_event = Mock()
    store.update_user = Mock()
    store.remove_user_from_org = Mock()
    store.unlink_oauth_provider = Mock()
    store.get_org_members = Mock(return_value=[])
    return store


@pytest.fixture
def privacy_handler(mock_user_store):
    """Create a PrivacyHandler with mock dependencies."""
    ctx = {
        "user_store": mock_user_store,
    }
    return PrivacyHandler(ctx)


@pytest.fixture
def privacy_handler_no_store():
    """Create a PrivacyHandler without user store."""
    ctx = {
        "user_store": None,
    }
    return PrivacyHandler(ctx)


@pytest.fixture
def mock_user():
    """Create a mock user object."""
    user = Mock()
    user.id = "user-123"
    user.email = "test@example.com"
    user.name = "Test User"
    user.role = "member"
    user.is_active = True
    user.email_verified = True
    user.created_at = datetime(2026, 1, 1, 0, 0, 0)
    user.updated_at = datetime(2026, 1, 10, 0, 0, 0)
    user.last_login_at = datetime(2026, 1, 15, 0, 0, 0)
    user.mfa_enabled = False
    user.api_key_prefix = None
    user.api_key_hash = None
    user.api_key_created_at = None
    user.api_key_expires_at = None
    user.org_id = None
    user.verify_password = Mock(return_value=True)
    return user


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler."""
    handler = Mock()
    handler.command = "GET"
    handler.headers = {"Authorization": "Bearer valid-token"}
    handler.rfile = Mock()
    return handler


@pytest.fixture
def mock_auth_context():
    """Create a mock authenticated context."""
    ctx = Mock()
    ctx.is_authenticated = True
    ctx.user_id = "user-123"
    return ctx


@pytest.fixture
def mock_unauth_context():
    """Create a mock unauthenticated context."""
    ctx = Mock()
    ctx.is_authenticated = False
    ctx.user_id = None
    return ctx


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches and rate limiters before and after each test."""
    clear_cache()
    # Clear all rate limiter buckets
    for limiter in _limiters.values():
        limiter.clear()
    _limiters.clear()
    yield
    clear_cache()
    for limiter in _limiters.values():
        limiter.clear()
    _limiters.clear()


# ============================================================================
# Route Matching Tests
# ============================================================================


class TestPrivacyRouting:
    """Tests for route matching."""

    def test_can_handle_export(self, privacy_handler):
        """Handler can handle /api/privacy/export."""
        assert privacy_handler.can_handle("/api/privacy/export") is True

    def test_can_handle_data_inventory(self, privacy_handler):
        """Handler can handle /api/privacy/data-inventory."""
        assert privacy_handler.can_handle("/api/privacy/data-inventory") is True

    def test_can_handle_account_delete(self, privacy_handler):
        """Handler can handle /api/privacy/account."""
        assert privacy_handler.can_handle("/api/privacy/account") is True

    def test_can_handle_preferences(self, privacy_handler):
        """Handler can handle /api/privacy/preferences."""
        assert privacy_handler.can_handle("/api/privacy/preferences") is True

    def test_can_handle_v2_export(self, privacy_handler):
        """Handler can handle /api/v2/users/me/export."""
        assert privacy_handler.can_handle("/api/v2/users/me/export") is True

    def test_can_handle_v2_data_inventory(self, privacy_handler):
        """Handler can handle /api/v2/users/me/data-inventory."""
        assert privacy_handler.can_handle("/api/v2/users/me/data-inventory") is True

    def test_can_handle_v2_users_me(self, privacy_handler):
        """Handler can handle /api/v2/users/me."""
        assert privacy_handler.can_handle("/api/v2/users/me") is True

    def test_cannot_handle_unrelated_routes(self, privacy_handler):
        """Handler doesn't handle unrelated routes."""
        assert privacy_handler.can_handle("/api/debates") is False
        assert privacy_handler.can_handle("/api/agents") is False
        assert privacy_handler.can_handle("/api/privacy/unknown") is False
        assert privacy_handler.can_handle("/api/users") is False


# ============================================================================
# GET /api/privacy/export Tests
# ============================================================================


class TestDataExport:
    """Tests for GET /api/privacy/export endpoint."""

    def test_export_unauthenticated(self, privacy_handler, mock_handler, mock_unauth_context):
        """Returns 401 when not authenticated."""
        with patch(
            "aragora.server.handlers.privacy.extract_user_from_request",
            return_value=mock_unauth_context,
        ):
            result = privacy_handler.handle("/api/privacy/export", {}, mock_handler)

        assert result is not None
        assert result.status_code == 401
        data = json.loads(result.body)
        assert "error" in data

    def test_export_no_user_store(self, privacy_handler_no_store, mock_handler, mock_auth_context):
        """Returns 503 when user store is unavailable."""
        with patch(
            "aragora.server.handlers.privacy.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            result = privacy_handler_no_store.handle("/api/privacy/export", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503
        data = json.loads(result.body)
        assert "error" in data

    def test_export_user_not_found(
        self, privacy_handler, mock_handler, mock_user_store, mock_auth_context
    ):
        """Returns 404 when user is not found."""
        mock_user_store.get_user_by_id.return_value = None

        with patch(
            "aragora.server.handlers.privacy.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            result = privacy_handler.handle("/api/privacy/export", {}, mock_handler)

        assert result is not None
        assert result.status_code == 404
        data = json.loads(result.body)
        assert "not found" in data["error"].lower()

    def test_export_success_json(
        self, privacy_handler, mock_handler, mock_user_store, mock_user, mock_auth_context
    ):
        """Returns user data in JSON format."""
        mock_user_store.get_user_by_id.return_value = mock_user

        with patch(
            "aragora.server.handlers.privacy.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            result = privacy_handler.handle("/api/privacy/export", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)

        # Check profile data
        assert "profile" in data
        assert data["profile"]["id"] == "user-123"
        assert data["profile"]["email"] == "test@example.com"
        assert data["profile"]["name"] == "Test User"

        # Check export metadata
        assert "_export_metadata" in data
        assert data["_export_metadata"]["format"] == "json"
        assert data["_export_metadata"]["data_controller"] == "Aragora"

    def test_export_csv_format(
        self, privacy_handler, mock_handler, mock_user_store, mock_user, mock_auth_context
    ):
        """Returns user data in CSV format."""
        mock_user_store.get_user_by_id.return_value = mock_user

        with patch(
            "aragora.server.handlers.privacy.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            result = privacy_handler.handle("/api/privacy/export", {"format": "csv"}, mock_handler)

        assert result is not None
        # CSV export returns a tuple (body, status, headers) instead of HandlerResult
        if isinstance(result, tuple):
            body, status, headers = result
            assert status == 200
            assert headers["Content-Type"] == "text/csv; charset=utf-8"
            assert "attachment" in headers["Content-Disposition"]
            csv_content = body.decode("utf-8")
        else:
            assert result.status_code == 200
            assert result.headers["Content-Type"] == "text/csv; charset=utf-8"
            assert "attachment" in result.headers["Content-Disposition"]
            csv_content = result.body.decode("utf-8")

        # Check CSV content contains profile data
        assert "Profile" in csv_content
        assert "email" in csv_content


# ============================================================================
# GET /api/privacy/data-inventory Tests
# ============================================================================


class TestDataInventory:
    """Tests for GET /api/privacy/data-inventory endpoint."""

    def test_inventory_unauthenticated(self, privacy_handler, mock_handler, mock_unauth_context):
        """Returns 401 when not authenticated."""
        with patch(
            "aragora.server.handlers.privacy.extract_user_from_request",
            return_value=mock_unauth_context,
        ):
            result = privacy_handler.handle("/api/privacy/data-inventory", {}, mock_handler)

        assert result is not None
        assert result.status_code == 401

    def test_inventory_success(self, privacy_handler, mock_handler, mock_auth_context):
        """Returns data inventory categories."""
        with patch(
            "aragora.server.handlers.privacy.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            result = privacy_handler.handle("/api/privacy/data-inventory", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)

        # Check inventory structure
        assert "categories" in data
        assert isinstance(data["categories"], list)
        assert len(data["categories"]) > 0

        # Check category structure
        category = data["categories"][0]
        assert "name" in category
        assert "examples" in category
        assert "purpose" in category
        assert "retention" in category

        # Check third party sharing info
        assert "third_party_sharing" in data
        assert "data_sold" in data
        assert data["data_sold"] is False
        assert "opt_out_available" in data


# ============================================================================
# DELETE /api/privacy/account Tests
# ============================================================================


class TestAccountDeletion:
    """Tests for DELETE /api/privacy/account endpoint."""

    @pytest.fixture(autouse=True)
    def bypass_rate_limit(self):
        """Bypass rate limiting for account deletion tests."""
        with patch.object(RateLimiter, "is_allowed", return_value=True):
            yield

    def test_delete_unauthenticated(self, privacy_handler, mock_handler, mock_unauth_context):
        """Returns 401 when not authenticated."""
        mock_handler.command = "DELETE"

        with patch(
            "aragora.server.handlers.privacy.extract_user_from_request",
            return_value=mock_unauth_context,
        ):
            result = privacy_handler.handle("/api/privacy/account", {}, mock_handler)

        assert result is not None
        assert result.status_code == 401

    def test_delete_no_confirmation(
        self, privacy_handler, mock_handler, mock_user_store, mock_user, mock_auth_context
    ):
        """Returns 400 when confirm is not true."""
        mock_handler.command = "DELETE"
        mock_user_store.get_user_by_id.return_value = mock_user

        with (
            patch(
                "aragora.server.handlers.privacy.extract_user_from_request",
                return_value=mock_auth_context,
            ),
            patch.object(privacy_handler, "read_json_body", return_value={"password": "test123"}),
        ):
            result = privacy_handler.handle("/api/privacy/account", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "confirm" in data["error"].lower()

    def test_delete_invalid_password(
        self, privacy_handler, mock_handler, mock_user_store, mock_user, mock_auth_context
    ):
        """Returns 401 when password is invalid."""
        mock_handler.command = "DELETE"
        mock_user_store.get_user_by_id.return_value = mock_user
        mock_user.verify_password.return_value = False

        with (
            patch(
                "aragora.server.handlers.privacy.extract_user_from_request",
                return_value=mock_auth_context,
            ),
            patch.object(
                privacy_handler,
                "read_json_body",
                return_value={"password": "wrong", "confirm": True},
            ),
        ):
            result = privacy_handler.handle("/api/privacy/account", {}, mock_handler)

        assert result is not None
        assert result.status_code == 401
        data = json.loads(result.body)
        assert "password" in data["error"].lower()

    def test_delete_success(
        self, privacy_handler, mock_handler, mock_user_store, mock_user, mock_auth_context
    ):
        """Successfully deletes user account."""
        mock_handler.command = "DELETE"
        mock_user_store.get_user_by_id.return_value = mock_user
        mock_user.verify_password.return_value = True

        with (
            patch(
                "aragora.server.handlers.privacy.extract_user_from_request",
                return_value=mock_auth_context,
            ),
            patch.object(
                privacy_handler,
                "read_json_body",
                return_value={"password": "correct", "confirm": True, "reason": "leaving"},
            ),
        ):
            result = privacy_handler.handle("/api/privacy/account", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)

        assert "message" in data
        assert "deleted" in data["message"].lower()
        assert "deletion_id" in data
        assert "data_deleted" in data


# ============================================================================
# POST /api/privacy/preferences Tests
# ============================================================================


class TestPrivacyPreferences:
    """Tests for privacy preferences endpoints."""

    def test_get_preferences_unauthenticated(
        self, privacy_handler, mock_handler, mock_unauth_context
    ):
        """Returns 401 when not authenticated."""
        with patch(
            "aragora.server.handlers.privacy.extract_user_from_request",
            return_value=mock_unauth_context,
        ):
            result = privacy_handler.handle(
                "/api/privacy/preferences", {}, mock_handler, method="GET"
            )

        assert result is not None
        assert result.status_code == 401

    def test_get_preferences_success(
        self, privacy_handler, mock_handler, mock_user_store, mock_auth_context
    ):
        """Returns privacy preferences."""
        mock_user_store.get_user_preferences.return_value = {
            "privacy": {
                "do_not_sell": True,
                "marketing_opt_out": True,
            }
        }

        with patch(
            "aragora.server.handlers.privacy.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            result = privacy_handler.handle(
                "/api/privacy/preferences", {}, mock_handler, method="GET"
            )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)

        assert data["do_not_sell"] is True
        assert data["marketing_opt_out"] is True
        assert "analytics_opt_out" in data
        assert "third_party_sharing" in data

    def test_update_preferences_success(
        self, privacy_handler, mock_handler, mock_user_store, mock_auth_context
    ):
        """Successfully updates privacy preferences."""
        mock_handler.command = "POST"
        mock_user_store.get_user_preferences.return_value = {}

        with (
            patch(
                "aragora.server.handlers.privacy.extract_user_from_request",
                return_value=mock_auth_context,
            ),
            patch.object(
                privacy_handler,
                "read_json_body",
                return_value={"do_not_sell": True, "analytics_opt_out": True},
            ),
        ):
            result = privacy_handler.handle(
                "/api/privacy/preferences", {}, mock_handler, method="POST"
            )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)

        assert "message" in data
        assert "preferences" in data
        assert data["preferences"]["do_not_sell"] is True
        assert data["preferences"]["analytics_opt_out"] is True

        # Verify preferences were saved
        mock_user_store.set_user_preferences.assert_called_once()

    def test_update_preferences_invalid_json(
        self, privacy_handler, mock_handler, mock_user_store, mock_auth_context
    ):
        """Returns 400 for invalid JSON body."""
        mock_handler.command = "POST"

        with (
            patch(
                "aragora.server.handlers.privacy.extract_user_from_request",
                return_value=mock_auth_context,
            ),
            patch.object(privacy_handler, "read_json_body", return_value=None),
        ):
            result = privacy_handler.handle(
                "/api/privacy/preferences", {}, mock_handler, method="POST"
            )

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "json" in data["error"].lower()


# ============================================================================
# Method Not Allowed Tests
# ============================================================================


class TestMethodNotAllowed:
    """Tests for method not allowed responses."""

    def test_post_to_export(self, privacy_handler, mock_handler):
        """Returns 405 for POST to export endpoint."""
        mock_handler.command = "POST"

        result = privacy_handler.handle("/api/privacy/export", {}, mock_handler)

        assert result is not None
        assert result.status_code == 405

    def test_post_to_data_inventory(self, privacy_handler, mock_handler):
        """Returns 405 for POST to data-inventory endpoint."""
        mock_handler.command = "POST"

        result = privacy_handler.handle("/api/privacy/data-inventory", {}, mock_handler)

        assert result is not None
        assert result.status_code == 405


# ============================================================================
# Handler Import Tests
# ============================================================================


class TestPrivacyHandlerImport:
    """Test PrivacyHandler import and export."""

    def test_handler_importable(self):
        """PrivacyHandler can be imported from handlers package."""
        from aragora.server.handlers import PrivacyHandler

        assert PrivacyHandler is not None

    def test_handler_in_all_exports(self):
        """PrivacyHandler is in __all__ exports."""
        from aragora.server.handlers import __all__

        assert "PrivacyHandler" in __all__
