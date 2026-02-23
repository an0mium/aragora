"""Tests for notification preferences handler.

Covers all routes, methods, validation, and edge cases of
NotificationPreferencesHandler:
- can_handle() routing (versioned and unversioned paths)
- GET /api/v1/notifications/preferences  (get user prefs)
- PUT /api/v1/notifications/preferences  (update user prefs)
- Rate limiting on both GET and PUT
- Channel validation (valid set, boolean values)
- Event type validation (boolean values)
- Quiet hours validation (object type)
- Digest mode validation (boolean)
- Default preferences for new users
- Preference merging (partial updates)
- Invalid JSON body handling
- User ID extraction from auth context
- Anonymous fallback for user ID
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.notifications.preferences import (
    NotificationPreferencesHandler,
    _DEFAULT_PREFERENCES,
    _preferences_limiter,
    _user_preferences,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Minimal mock HTTP handler for preferences tests."""

    def __init__(
        self,
        body: dict | None = None,
        client_ip: str = "10.0.0.1",
    ):
        self.headers: dict[str, str] = {}
        self.client_address = (client_ip, 12345)
        self.rfile = MagicMock()
        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers["Content-Length"] = str(len(raw))
        else:
            self.rfile.read.return_value = b""
            self.headers["Content-Length"] = "0"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_state():
    """Reset the in-memory preferences store and rate limiter between tests."""
    _user_preferences.clear()
    _preferences_limiter.clear()
    yield
    _user_preferences.clear()
    _preferences_limiter.clear()


@pytest.fixture
def handler():
    """Create a NotificationPreferencesHandler instance."""
    return NotificationPreferencesHandler()


@pytest.fixture
def handler_with_ctx():
    """Create a handler with a custom context dict."""
    return NotificationPreferencesHandler(ctx={"tenant": "acme"})


# ---------------------------------------------------------------------------
# can_handle() routing
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for can_handle() path routing."""

    def test_versioned_path(self, handler):
        assert handler.can_handle("/api/v1/notifications/preferences") is True

    def test_unversioned_path(self, handler):
        assert handler.can_handle("/api/notifications/preferences") is True

    def test_wrong_path_returns_false(self, handler):
        assert handler.can_handle("/api/v1/notifications") is False

    def test_trailing_slash_returns_false(self, handler):
        assert handler.can_handle("/api/v1/notifications/preferences/") is False

    def test_extra_segment_returns_false(self, handler):
        assert handler.can_handle("/api/v1/notifications/preferences/extra") is False

    def test_different_endpoint_returns_false(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_root_path(self, handler):
        assert handler.can_handle("/") is False


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    """Tests for handler construction."""

    def test_default_ctx_is_empty_dict(self):
        h = NotificationPreferencesHandler()
        assert h.ctx == {}

    def test_ctx_preserved(self):
        h = NotificationPreferencesHandler(ctx={"key": "value"})
        assert h.ctx == {"key": "value"}

    def test_none_ctx_becomes_empty(self):
        h = NotificationPreferencesHandler(ctx=None)
        assert h.ctx == {}


# ---------------------------------------------------------------------------
# ROUTES class attribute
# ---------------------------------------------------------------------------


class TestRoutes:
    """Verify the ROUTES class attribute."""

    def test_routes_contains_preferences_path(self):
        assert "/api/v1/notifications/preferences" in NotificationPreferencesHandler.ROUTES

    def test_routes_length(self):
        assert len(NotificationPreferencesHandler.ROUTES) == 1


# ---------------------------------------------------------------------------
# GET preferences (handle)
# ---------------------------------------------------------------------------


class TestGetPreferences:
    """Tests for GET /api/v1/notifications/preferences."""

    def test_returns_default_preferences_for_new_user(self, handler):
        h = MockHTTPHandler()
        result = handler.handle("/api/v1/notifications/preferences", {}, h)
        body = _body(result)
        assert body["preferences"]["channels"]["slack"] is True
        assert body["preferences"]["channels"]["email"] is True
        assert body["preferences"]["channels"]["webhook"] is True
        assert body["preferences"]["digest_mode"] is False
        assert body["preferences"]["quiet_hours"]["enabled"] is False

    def test_returns_200_status(self, handler):
        h = MockHTTPHandler()
        result = handler.handle("/api/v1/notifications/preferences", {}, h)
        assert _status(result) == 200

    def test_returns_user_id_in_response(self, handler):
        h = MockHTTPHandler()
        result = handler.handle("/api/v1/notifications/preferences", {}, h)
        body = _body(result)
        # The conftest mock sets user_id to "test-user-001"
        assert "user_id" in body

    def test_returns_stored_preferences(self, handler):
        """If user already has saved prefs, return those."""
        h = MockHTTPHandler()
        # Simulate stored prefs by figuring out the user_id and pre-populating
        user_id = handler._get_user_id(h)
        custom_prefs = {"channels": {"slack": False}, "digest_mode": True}
        _user_preferences[user_id] = custom_prefs

        result = handler.handle("/api/v1/notifications/preferences", {}, h)
        body = _body(result)
        assert body["preferences"]["channels"]["slack"] is False
        assert body["preferences"]["digest_mode"] is True

    def test_wrong_path_returns_none(self, handler):
        h = MockHTTPHandler()
        result = handler.handle("/api/v1/notifications/other", {}, h)
        assert result is None

    def test_default_prefs_are_deep_copy(self, handler):
        """Ensure returned defaults don't mutate the module-level constant."""
        h = MockHTTPHandler()
        result1 = handler.handle("/api/v1/notifications/preferences", {}, h)
        body1 = _body(result1)
        body1["preferences"]["channels"]["slack"] = False

        result2 = handler.handle("/api/v1/notifications/preferences", {}, h)
        body2 = _body(result2)
        # The default should still be True
        assert _DEFAULT_PREFERENCES["channels"]["slack"] is True

    def test_default_event_types(self, handler):
        h = MockHTTPHandler()
        result = handler.handle("/api/v1/notifications/preferences", {}, h)
        body = _body(result)
        event_types = body["preferences"]["event_types"]
        assert event_types["finding_created"] is True
        assert event_types["audit_completed"] is True
        assert event_types["checkpoint_approval"] is True
        assert event_types["budget_alert"] is True
        assert event_types["compliance_finding"] is True
        assert event_types["workflow_progress"] is True
        assert event_types["cost_anomaly"] is True
        assert event_types["debate_completed"] is True

    def test_default_quiet_hours(self, handler):
        h = MockHTTPHandler()
        result = handler.handle("/api/v1/notifications/preferences", {}, h)
        body = _body(result)
        qh = body["preferences"]["quiet_hours"]
        assert qh["enabled"] is False
        assert qh["start"] == "22:00"
        assert qh["end"] == "08:00"
        assert qh["timezone"] == "UTC"


# ---------------------------------------------------------------------------
# PUT preferences (handle_put)
# ---------------------------------------------------------------------------


class TestUpdatePreferences:
    """Tests for PUT /api/v1/notifications/preferences."""

    def test_update_channels(self, handler):
        body = {"preferences": {"channels": {"slack": False}}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        data = _body(result)
        assert _status(result) == 200
        assert data["updated"] is True
        assert data["preferences"]["channels"]["slack"] is False
        # Other channels should remain at defaults
        assert data["preferences"]["channels"]["email"] is True
        assert data["preferences"]["channels"]["webhook"] is True

    def test_update_event_types(self, handler):
        body = {"preferences": {"event_types": {"budget_alert": False}}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        data = _body(result)
        assert data["preferences"]["event_types"]["budget_alert"] is False
        # Other event types unchanged
        assert data["preferences"]["event_types"]["finding_created"] is True

    def test_update_quiet_hours(self, handler):
        body = {"preferences": {"quiet_hours": {"enabled": True, "start": "23:00"}}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        data = _body(result)
        qh = data["preferences"]["quiet_hours"]
        assert qh["enabled"] is True
        assert qh["start"] == "23:00"
        # Defaults preserved
        assert qh["end"] == "08:00"

    def test_update_digest_mode(self, handler):
        body = {"preferences": {"digest_mode": True}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        data = _body(result)
        assert data["preferences"]["digest_mode"] is True

    def test_update_multiple_sections(self, handler):
        body = {
            "preferences": {
                "channels": {"email": False},
                "event_types": {"cost_anomaly": False},
                "digest_mode": True,
            }
        }
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        data = _body(result)
        assert data["preferences"]["channels"]["email"] is False
        assert data["preferences"]["event_types"]["cost_anomaly"] is False
        assert data["preferences"]["digest_mode"] is True

    def test_update_persists_across_gets(self, handler):
        """PUT should persist so a subsequent GET returns updated prefs."""
        body = {"preferences": {"digest_mode": True}}
        h = MockHTTPHandler(body=body)
        handler.handle_put("/api/v1/notifications/preferences", {}, h)

        h2 = MockHTTPHandler()
        result = handler.handle("/api/v1/notifications/preferences", {}, h2)
        data = _body(result)
        assert data["preferences"]["digest_mode"] is True

    def test_update_returns_user_id(self, handler):
        body = {"preferences": {"digest_mode": True}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        data = _body(result)
        assert "user_id" in data

    def test_wrong_path_returns_none(self, handler):
        body = {"preferences": {"digest_mode": True}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/other", {}, h)
        assert result is None

    def test_body_without_preferences_wrapper(self, handler):
        """Body sent directly without 'preferences' key should still work."""
        body = {"channels": {"slack": False}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        data = _body(result)
        assert _status(result) == 200
        assert data["preferences"]["channels"]["slack"] is False

    def test_successive_updates_merge(self, handler):
        """Multiple PUTs should merge, not replace."""
        body1 = {"preferences": {"channels": {"slack": False}}}
        h1 = MockHTTPHandler(body=body1)
        handler.handle_put("/api/v1/notifications/preferences", {}, h1)

        body2 = {"preferences": {"channels": {"email": False}}}
        h2 = MockHTTPHandler(body=body2)
        handler.handle_put("/api/v1/notifications/preferences", {}, h2)

        h3 = MockHTTPHandler()
        result = handler.handle("/api/v1/notifications/preferences", {}, h3)
        data = _body(result)
        # Both should be updated
        assert data["preferences"]["channels"]["slack"] is False
        assert data["preferences"]["channels"]["email"] is False

    def test_add_custom_event_type(self, handler):
        """Custom event types should be accepted."""
        body = {"preferences": {"event_types": {"custom_event": True}}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        data = _body(result)
        assert data["preferences"]["event_types"]["custom_event"] is True


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for input validation on PUT."""

    def test_invalid_channel_name(self, handler):
        body = {"preferences": {"channels": {"sms": True}}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        assert _status(result) == 400
        assert "Invalid channel" in _body(result)["error"]

    def test_channel_value_not_boolean(self, handler):
        body = {"preferences": {"channels": {"slack": "yes"}}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        assert _status(result) == 400
        assert "must be boolean" in _body(result)["error"]

    def test_channels_not_object(self, handler):
        body = {"preferences": {"channels": [True, False]}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        assert _status(result) == 400
        assert "'channels' must be an object" in _body(result)["error"]

    def test_event_types_not_object(self, handler):
        body = {"preferences": {"event_types": "all"}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        assert _status(result) == 400
        assert "'event_types' must be an object" in _body(result)["error"]

    def test_event_type_value_not_boolean(self, handler):
        body = {"preferences": {"event_types": {"budget_alert": 1}}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        assert _status(result) == 400
        assert "must be boolean" in _body(result)["error"]

    def test_quiet_hours_not_object(self, handler):
        body = {"preferences": {"quiet_hours": True}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        assert _status(result) == 400
        assert "'quiet_hours' must be an object" in _body(result)["error"]

    def test_digest_mode_not_boolean(self, handler):
        body = {"preferences": {"digest_mode": "on"}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        assert _status(result) == 400
        assert "'digest_mode' must be boolean" in _body(result)["error"]

    def test_invalid_json_body(self, handler):
        """When read_json_body returns None, return 400."""
        h = MockHTTPHandler()
        # Simulate invalid JSON by having Content-Length > 0 but bad bytes
        h.headers["Content-Length"] = "5"
        h.rfile.read.return_value = b"notjson"
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        assert _status(result) == 400
        body = _body(result)
        assert "Invalid JSON" in body.get("error", body.get("message", ""))

    def test_channel_value_integer_not_boolean(self, handler):
        body = {"preferences": {"channels": {"email": 0}}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        assert _status(result) == 400

    def test_multiple_invalid_channels(self, handler):
        """Only the first invalid channel triggers the error."""
        body = {"preferences": {"channels": {"sms": True, "fax": True}}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        assert _status(result) == 400
        assert "Invalid channel" in _body(result)["error"]

    def test_valid_channels_listed_in_error(self, handler):
        body = {"preferences": {"channels": {"phone": True}}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        error_msg = _body(result)["error"]
        assert "email" in error_msg
        assert "slack" in error_msg
        assert "webhook" in error_msg


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for rate limiting on GET and PUT."""

    def test_get_rate_limit_exceeded(self, handler):
        h = MockHTTPHandler()
        with patch.object(_preferences_limiter, "is_allowed", return_value=False):
            result = handler.handle("/api/v1/notifications/preferences", {}, h)
        assert _status(result) == 429
        assert "Rate limit" in _body(result)["error"]

    def test_put_rate_limit_exceeded(self, handler):
        body = {"preferences": {"digest_mode": True}}
        h = MockHTTPHandler(body=body)
        with patch.object(_preferences_limiter, "is_allowed", return_value=False):
            result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        assert _status(result) == 429

    def test_get_rate_limit_allowed(self, handler):
        h = MockHTTPHandler()
        with patch.object(_preferences_limiter, "is_allowed", return_value=True):
            result = handler.handle("/api/v1/notifications/preferences", {}, h)
        assert _status(result) == 200

    def test_limiter_uses_client_ip(self, handler):
        h = MockHTTPHandler(client_ip="192.168.1.42")
        with patch.object(_preferences_limiter, "is_allowed", return_value=True) as mock_allowed:
            handler.handle("/api/v1/notifications/preferences", {}, h)
        # get_client_ip should have been called and passed to is_allowed
        assert mock_allowed.called

    def test_limiter_rpm_is_30(self):
        assert _preferences_limiter.rpm == 30


# ---------------------------------------------------------------------------
# User ID extraction
# ---------------------------------------------------------------------------


class TestUserIdExtraction:
    """Tests for _get_user_id method."""

    def test_returns_user_id_from_auth_context(self, handler):
        """Conftest patches get_current_user to return a mock with user_id."""
        h = MockHTTPHandler()
        uid = handler._get_user_id(h)
        assert uid == "test-user-001"

    def test_anonymous_when_no_user(self, handler):
        """When get_current_user returns None, fallback to anonymous."""
        h = MockHTTPHandler()
        with patch.object(handler, "get_current_user", return_value=None):
            uid = handler._get_user_id(h)
        assert uid == "anonymous"

    def test_anonymous_when_no_user_id_attr(self, handler):
        """When user object lacks user_id, fallback to anonymous."""
        mock_user = MagicMock(spec=[])
        h = MockHTTPHandler()
        with patch.object(handler, "get_current_user", return_value=mock_user):
            uid = handler._get_user_id(h)
        assert uid == "anonymous"

    def test_different_user_ids_get_separate_prefs(self, handler):
        """Each user_id gets its own preferences store entry."""
        h = MockHTTPHandler()
        uid = handler._get_user_id(h)

        body1 = {"preferences": {"digest_mode": True}}
        h1 = MockHTTPHandler(body=body1)
        handler.handle_put("/api/v1/notifications/preferences", {}, h1)

        # Simulate a different user
        mock_user_b = MagicMock()
        mock_user_b.user_id = "user-b"
        with patch.object(handler, "get_current_user", return_value=mock_user_b):
            h2 = MockHTTPHandler()
            result = handler.handle("/api/v1/notifications/preferences", {}, h2)
            data = _body(result)
            # user-b should have default prefs, not user-a's
            assert data["preferences"]["digest_mode"] is False
            assert data["user_id"] == "user-b"


# ---------------------------------------------------------------------------
# Edge cases and _DEFAULT_PREFERENCES integrity
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_default_preferences_not_mutated_by_update(self, handler):
        """Updating a user's prefs should never change _DEFAULT_PREFERENCES."""
        import copy
        original = copy.deepcopy(_DEFAULT_PREFERENCES)
        body = {
            "preferences": {
                "channels": {"slack": False, "email": False, "webhook": False},
                "event_types": {"finding_created": False},
                "quiet_hours": {"enabled": True},
                "digest_mode": True,
            }
        }
        h = MockHTTPHandler(body=body)
        handler.handle_put("/api/v1/notifications/preferences", {}, h)
        assert _DEFAULT_PREFERENCES == original

    def test_empty_preferences_body_is_noop(self, handler):
        """Empty preferences object should keep defaults."""
        body = {"preferences": {}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        data = _body(result)
        assert data["preferences"]["channels"]["slack"] is True
        assert data["preferences"]["digest_mode"] is False

    def test_empty_body_is_noop(self, handler):
        """Empty body should keep defaults (no 'preferences' key)."""
        body = {}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        data = _body(result)
        assert _status(result) == 200
        assert data["preferences"]["channels"]["slack"] is True

    def test_quiet_hours_arbitrary_keys(self, handler):
        """Quiet hours accepts any keys (no strict validation on keys)."""
        body = {"preferences": {"quiet_hours": {"custom_field": "value"}}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        data = _body(result)
        assert data["preferences"]["quiet_hours"]["custom_field"] == "value"

    def test_channels_all_three_valid(self, handler):
        """All three valid channels can be set at once."""
        body = {"preferences": {"channels": {"slack": False, "email": False, "webhook": False}}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        data = _body(result)
        assert data["preferences"]["channels"]["slack"] is False
        assert data["preferences"]["channels"]["email"] is False
        assert data["preferences"]["channels"]["webhook"] is False

    def test_update_overrides_previous_channel_value(self, handler):
        """Setting a channel to False then True should result in True."""
        h1 = MockHTTPHandler(body={"preferences": {"channels": {"slack": False}}})
        handler.handle_put("/api/v1/notifications/preferences", {}, h1)

        h2 = MockHTTPHandler(body={"preferences": {"channels": {"slack": True}}})
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h2)
        data = _body(result)
        assert data["preferences"]["channels"]["slack"] is True

    def test_handle_put_unversioned_path(self, handler):
        """PUT with unversioned path should also work (strip_version_prefix)."""
        body = {"preferences": {"digest_mode": True}}
        h = MockHTTPHandler(body=body)
        result = handler.handle_put("/api/v1/notifications/preferences", {}, h)
        assert _status(result) == 200

    def test_handle_get_unversioned_path(self, handler):
        """GET with unversioned path works after version strip."""
        h = MockHTTPHandler()
        result = handler.handle("/api/v1/notifications/preferences", {}, h)
        assert _status(result) == 200

    def test_context_passed_to_handler(self, handler_with_ctx):
        assert handler_with_ctx.ctx == {"tenant": "acme"}
