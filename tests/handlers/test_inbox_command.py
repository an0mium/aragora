"""Tests for inbox command handler (aragora/server/handlers/inbox_command.py).

Covers all routes and behavior of the InboxCommandHandler class:
- GET    /api/inbox/command          - Fetch prioritized inbox
- POST   /api/inbox/actions          - Execute quick action
- POST   /api/inbox/bulk-actions     - Execute bulk action
- GET    /api/inbox/sender-profile   - Get sender profile
- GET    /api/inbox/daily-digest     - Get daily digest
- POST   /api/inbox/reprioritize     - Trigger AI re-prioritization

Tests are organized into classes for each endpoint plus cross-cutting concerns:
- TestGetInbox: prioritized inbox fetching + filtering
- TestQuickAction: action execution + validation
- TestBulkAction: bulk actions + filter validation
- TestSenderProfile: sender profile lookup
- TestDailyDigest: daily digest stats
- TestReprioritize: AI re-prioritization
- TestValidation: input validation helpers
- TestIterableTTLCache: cache wrapper
- TestAuthentication: permission checks (no_auto_auth)
- TestRegisterRoutes: route registration
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp.test_utils import make_mocked_request

from aragora.server.handlers.inbox_command import (
    ALLOWED_ACTIONS,
    ALLOWED_BULK_FILTERS,
    ALLOWED_FORCE_TIERS,
    ALLOWED_PRIORITY_FILTERS,
    ALLOWED_SNOOZE_DURATIONS,
    MAX_EMAIL_ID_LENGTH,
    MAX_EMAIL_IDS_PER_REQUEST,
    MAX_PARAMS_KEYS,
    InboxCommandHandler,
    IterableTTLCache,
    _validate_email_address,
    _validate_email_id,
    _validate_params,
    _sanitize_string_param,
    register_routes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    method: str = "GET",
    path: str = "/api/inbox/command",
    query: str = "",
    body: dict[str, Any] | None = None,
    match_info: dict[str, str] | None = None,
) -> MagicMock:
    """Create a mock aiohttp request with given parameters."""
    full_path = f"{path}?{query}" if query else path
    request = make_mocked_request(method, full_path)

    if match_info:
        request.match_info.update(match_info)

    if body is not None:
        request.json = AsyncMock(return_value=body)
        request.text = AsyncMock(return_value=json.dumps(body))
        request.read = AsyncMock(return_value=json.dumps(body).encode())

    return request


def _parse_response(response) -> dict[str, Any]:
    """Parse response body as JSON."""
    return json.loads(response.body)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_inbox_auth(request, monkeypatch):
    """Patch get_auth_context in the inbox_command module.

    The handler does ``from ... import get_auth_context`` which creates a local
    binding.  The conftest patches the *source* module, but the local reference
    is stale.  We must also patch ``inbox_command.get_auth_context`` directly.
    """
    if "no_auto_auth" in [m.name for m in request.node.iter_markers()]:
        yield
        return

    from aragora.rbac.models import AuthorizationContext

    mock_ctx = AuthorizationContext(
        user_id="test-user-001",
        user_email="test@example.com",
        org_id="test-org-001",
        roles={"admin", "owner"},
        permissions={"*"},
    )

    async def _mock_get_auth_context(req, require_auth=False):
        return mock_ctx

    monkeypatch.setattr(
        "aragora.server.handlers.inbox_command.get_auth_context",
        _mock_get_auth_context,
    )
    yield


@pytest.fixture
def handler():
    """Create an InboxCommandHandler with mocked services."""
    with patch(
        "aragora.server.handlers.inbox_command.ServiceRegistry"
    ) as mock_registry_cls:
        mock_registry = MagicMock()
        mock_registry.has.return_value = False
        mock_registry_cls.get.return_value = mock_registry

        h = InboxCommandHandler(
            gmail_connector=None,
            prioritizer=None,
            sender_history=None,
        )
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
def _clear_email_cache():
    """Clear the module-level email cache between tests."""
    from aragora.server.handlers.inbox_command import _email_cache, _priority_results

    _email_cache.clear()
    _priority_results.clear()
    yield
    _email_cache.clear()
    _priority_results.clear()


# ============================================================================
# Validation Helpers
# ============================================================================


class TestValidateEmailId:
    """Tests for _validate_email_id function."""

    def test_valid_simple_id(self):
        assert _validate_email_id("abc123") == "abc123"

    def test_valid_id_with_hyphens(self):
        assert _validate_email_id("msg-123-abc") == "msg-123-abc"

    def test_valid_id_with_underscores(self):
        assert _validate_email_id("msg_123_abc") == "msg_123_abc"

    def test_valid_id_with_dots(self):
        assert _validate_email_id("msg.123.abc") == "msg.123.abc"

    def test_strips_whitespace(self):
        assert _validate_email_id("  abc123  ") == "abc123"

    def test_rejects_non_string(self):
        assert _validate_email_id(123) is None

    def test_rejects_none(self):
        assert _validate_email_id(None) is None

    def test_rejects_empty_string(self):
        assert _validate_email_id("") is None

    def test_rejects_whitespace_only(self):
        assert _validate_email_id("   ") is None

    def test_rejects_too_long(self):
        long_id = "a" * (MAX_EMAIL_ID_LENGTH + 1)
        assert _validate_email_id(long_id) is None

    def test_accepts_max_length(self):
        max_id = "a" * MAX_EMAIL_ID_LENGTH
        assert _validate_email_id(max_id) == max_id

    def test_rejects_special_characters(self):
        assert _validate_email_id("abc@123") is None

    def test_rejects_spaces_in_middle(self):
        assert _validate_email_id("abc 123") is None

    def test_rejects_shell_injection(self):
        assert _validate_email_id("abc; rm -rf /") is None

    def test_rejects_html_injection(self):
        assert _validate_email_id("<script>alert(1)</script>") is None

    def test_rejects_list(self):
        assert _validate_email_id(["abc"]) is None

    def test_rejects_dict(self):
        assert _validate_email_id({"id": "abc"}) is None


class TestValidateEmailAddress:
    """Tests for _validate_email_address function."""

    def test_valid_email(self):
        assert _validate_email_address("user@example.com") == "user@example.com"

    def test_valid_email_with_dots(self):
        assert _validate_email_address("first.last@example.com") == "first.last@example.com"

    def test_valid_email_with_plus(self):
        assert _validate_email_address("user+tag@example.com") == "user+tag@example.com"

    def test_strips_whitespace(self):
        assert _validate_email_address("  user@example.com  ") == "user@example.com"

    def test_rejects_non_string(self):
        assert _validate_email_address(123) is None

    def test_rejects_none(self):
        assert _validate_email_address(None) is None

    def test_rejects_empty(self):
        assert _validate_email_address("") is None

    def test_rejects_no_at_sign(self):
        assert _validate_email_address("userexample.com") is None

    def test_rejects_too_long(self):
        long_email = "a" * 310 + "@example.com"
        assert _validate_email_address(long_email) is None

    def test_rejects_whitespace_only(self):
        assert _validate_email_address("   ") is None


class TestSanitizeStringParam:
    """Tests for _sanitize_string_param function."""

    def test_normal_string(self):
        assert _sanitize_string_param("hello", 100) == "hello"

    def test_strips_whitespace(self):
        assert _sanitize_string_param("  hello  ", 100) == "hello"

    def test_truncates_to_max_length(self):
        assert _sanitize_string_param("abcdef", 3) == "abc"

    def test_non_string_returns_empty(self):
        assert _sanitize_string_param(123, 100) == ""

    def test_none_returns_empty(self):
        assert _sanitize_string_param(None, 100) == ""

    def test_list_returns_empty(self):
        assert _sanitize_string_param(["hello"], 100) == ""


class TestValidateParams:
    """Tests for _validate_params function."""

    def test_none_returns_empty_dict(self):
        assert _validate_params(None) == {}

    def test_valid_dict(self):
        assert _validate_params({"key": "value"}) == {"key": "value"}

    def test_non_dict_returns_none(self):
        assert _validate_params("invalid") is None

    def test_list_returns_none(self):
        assert _validate_params(["a", "b"]) is None

    def test_too_many_keys_returns_none(self):
        big_params = {f"key_{i}": i for i in range(MAX_PARAMS_KEYS + 1)}
        assert _validate_params(big_params) is None

    def test_max_keys_allowed(self):
        params = {f"key_{i}": i for i in range(MAX_PARAMS_KEYS)}
        assert _validate_params(params) == params

    def test_empty_dict(self):
        assert _validate_params({}) == {}


# ============================================================================
# GET /api/inbox/command - Fetch Prioritized Inbox
# ============================================================================


class TestGetInbox:
    """Tests for handle_get_inbox endpoint."""

    @pytest.mark.asyncio
    async def test_basic_get_inbox(self, handler):
        """Test fetching inbox with defaults returns demo data."""
        request = _make_request("GET", "/api/inbox/command")

        response = await handler.handle_get_inbox(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True
        assert "emails" in data
        assert "stats" in data
        assert "total" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_get_inbox_with_limit(self, handler):
        """Test fetching inbox with limit query param."""
        request = _make_request("GET", "/api/inbox/command", query="limit=2")

        response = await handler.handle_get_inbox(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True
        assert len(data["emails"]) <= 2

    @pytest.mark.asyncio
    async def test_get_inbox_with_offset(self, handler):
        """Test fetching inbox with offset."""
        request = _make_request("GET", "/api/inbox/command", query="offset=2")

        response = await handler.handle_get_inbox(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_get_inbox_with_priority_filter(self, handler):
        """Test filtering by priority level."""
        request = _make_request("GET", "/api/inbox/command", query="priority=critical")

        response = await handler.handle_get_inbox(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True
        # All returned emails should be critical
        for email in data["emails"]:
            assert email["priority"] == "critical"

    @pytest.mark.asyncio
    async def test_get_inbox_priority_filter_high(self, handler):
        """Test filtering by high priority."""
        request = _make_request("GET", "/api/inbox/command", query="priority=high")

        response = await handler.handle_get_inbox(request)

        assert response.status == 200
        data = _parse_response(response)
        for email in data["emails"]:
            assert email["priority"] == "high"

    @pytest.mark.asyncio
    async def test_get_inbox_invalid_priority_filter(self, handler):
        """Test rejection of invalid priority filter."""
        request = _make_request("GET", "/api/inbox/command", query="priority=urgent")

        response = await handler.handle_get_inbox(request)

        assert response.status == 400
        data = _parse_response(response)
        assert data["success"] is False
        assert "Invalid priority filter" in data["error"]

    @pytest.mark.asyncio
    async def test_get_inbox_priority_filter_case_insensitive(self, handler):
        """Test that priority filter is case-insensitive."""
        request = _make_request("GET", "/api/inbox/command", query="priority=CRITICAL")

        response = await handler.handle_get_inbox(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_get_inbox_unread_only(self, handler):
        """Test filtering to unread only."""
        request = _make_request("GET", "/api/inbox/command", query="unread_only=true")

        response = await handler.handle_get_inbox(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_get_inbox_unread_false(self, handler):
        """Test unread_only=false does not filter."""
        request = _make_request("GET", "/api/inbox/command", query="unread_only=false")

        response = await handler.handle_get_inbox(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_get_inbox_stats_structure(self, handler):
        """Test that stats have expected structure."""
        request = _make_request("GET", "/api/inbox/command")

        response = await handler.handle_get_inbox(request)

        data = _parse_response(response)
        stats = data["stats"]
        assert "total" in stats
        assert "unread" in stats
        assert "critical" in stats
        assert "high" in stats
        assert "medium" in stats
        assert "low" in stats
        assert "deferred" in stats
        assert "actionRequired" in stats

    @pytest.mark.asyncio
    async def test_get_inbox_all_priority_filters_valid(self, handler):
        """Test that all allowed priority filters are accepted."""
        for priority in ALLOWED_PRIORITY_FILTERS:
            request = _make_request("GET", "/api/inbox/command", query=f"priority={priority}")
            response = await handler.handle_get_inbox(request)
            assert response.status == 200, f"Priority filter '{priority}' should be accepted"

    @pytest.mark.asyncio
    async def test_get_inbox_internal_error(self, handler):
        """Test error handling when fetching fails."""
        with patch.object(handler, "_fetch_prioritized_emails", side_effect=RuntimeError("DB down")):
            request = _make_request("GET", "/api/inbox/command")
            response = await handler.handle_get_inbox(request)

        assert response.status == 500
        data = _parse_response(response)
        assert data["success"] is False
        assert data["error"] == "Internal server error"


# ============================================================================
# POST /api/inbox/actions - Quick Action
# ============================================================================


class TestQuickAction:
    """Tests for handle_quick_action endpoint."""

    @pytest.mark.asyncio
    async def test_archive_action(self, handler):
        """Test archiving an email."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "archive", "emailIds": ["msg-001"]},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True
        assert data["action"] == "archive"
        assert data["processed"] == 1

    @pytest.mark.asyncio
    async def test_snooze_action(self, handler):
        """Test snoozing an email."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "snooze", "emailIds": ["msg-001"], "params": {"duration": "1h"}},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True
        assert data["action"] == "snooze"

    @pytest.mark.asyncio
    async def test_reply_action(self, handler):
        """Test creating a reply draft."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "reply", "emailIds": ["msg-001"], "params": {"body": "Thanks!"}},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True
        assert data["action"] == "reply"

    @pytest.mark.asyncio
    async def test_forward_action(self, handler):
        """Test creating a forward draft."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={
                "action": "forward",
                "emailIds": ["msg-001"],
                "params": {"to": "colleague@example.com"},
            },
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True
        assert data["action"] == "forward"

    @pytest.mark.asyncio
    async def test_spam_action(self, handler):
        """Test marking as spam."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "spam", "emailIds": ["msg-001"]},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True
        assert data["action"] == "spam"

    @pytest.mark.asyncio
    async def test_mark_important_action(self, handler):
        """Test marking as important."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "mark_important", "emailIds": ["msg-001"]},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True
        assert data["action"] == "mark_important"

    @pytest.mark.asyncio
    async def test_delete_action(self, handler):
        """Test deleting an email."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "delete", "emailIds": ["msg-001"]},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True
        assert data["action"] == "delete"

    @pytest.mark.asyncio
    async def test_all_allowed_actions(self, handler):
        """Test that all allowed actions are accepted."""
        for action in ALLOWED_ACTIONS:
            request = _make_request(
                "POST",
                "/api/inbox/actions",
                body={"action": action, "emailIds": ["msg-001"]},
            )
            response = await handler.handle_quick_action(request)
            assert response.status == 200, f"Action '{action}' should be accepted"

    @pytest.mark.asyncio
    async def test_invalid_action_rejected(self, handler):
        """Test that invalid actions are rejected."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "destroy_everything", "emailIds": ["msg-001"]},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 400
        data = _parse_response(response)
        assert data["success"] is False
        assert "Invalid action" in data["error"]

    @pytest.mark.asyncio
    async def test_missing_action(self, handler):
        """Test that missing action returns error."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"emailIds": ["msg-001"]},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 400
        data = _parse_response(response)
        assert data["success"] is False
        assert "action is required" in data["error"]

    @pytest.mark.asyncio
    async def test_action_not_string(self, handler):
        """Test that non-string action returns error."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": 123, "emailIds": ["msg-001"]},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 400
        data = _parse_response(response)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_missing_email_ids(self, handler):
        """Test that missing emailIds returns error."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "archive"},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 400
        data = _parse_response(response)
        assert data["success"] is False
        assert "emailIds is required" in data["error"]

    @pytest.mark.asyncio
    async def test_empty_email_ids_list(self, handler):
        """Test that empty emailIds list returns error."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "archive", "emailIds": []},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 400
        data = _parse_response(response)
        assert data["success"] is False
        assert "non-empty list" in data["error"]

    @pytest.mark.asyncio
    async def test_email_ids_not_list(self, handler):
        """Test that non-list emailIds returns error."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "archive", "emailIds": "msg-001"},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 400
        data = _parse_response(response)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_email_ids_exceeds_max(self, handler):
        """Test that too many email IDs returns error."""
        ids = [f"msg-{i}" for i in range(MAX_EMAIL_IDS_PER_REQUEST + 1)]
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "archive", "emailIds": ids},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 400
        data = _parse_response(response)
        assert data["success"] is False
        assert "exceeds maximum" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_email_id_in_list(self, handler):
        """Test that invalid email ID in list returns error."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "archive", "emailIds": ["valid-id", "invalid@id!"]},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 400
        data = _parse_response(response)
        assert data["success"] is False
        assert "Invalid email ID" in data["error"]

    @pytest.mark.asyncio
    async def test_non_string_email_id_in_list(self, handler):
        """Test that non-string email ID in list returns error."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "archive", "emailIds": [123]},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 400
        data = _parse_response(response)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_invalid_params_object(self, handler):
        """Test that invalid params returns error."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "archive", "emailIds": ["msg-001"], "params": "invalid"},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 400
        data = _parse_response(response)
        assert data["success"] is False
        assert "Invalid params" in data["error"]

    @pytest.mark.asyncio
    async def test_params_too_many_keys(self, handler):
        """Test that params with too many keys returns error."""
        big_params = {f"key_{i}": "val" for i in range(MAX_PARAMS_KEYS + 1)}
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "archive", "emailIds": ["msg-001"], "params": big_params},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 400
        data = _parse_response(response)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_multiple_email_ids(self, handler):
        """Test action on multiple email IDs."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "archive", "emailIds": ["msg-001", "msg-002", "msg-003"]},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True
        assert data["processed"] == 3
        assert len(data["results"]) == 3

    @pytest.mark.asyncio
    async def test_action_case_insensitive(self, handler):
        """Test that actions are case-insensitive."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "ARCHIVE", "emailIds": ["msg-001"]},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["action"] == "archive"

    @pytest.mark.asyncio
    async def test_action_stripped(self, handler):
        """Test that action whitespace is stripped."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "  archive  ", "emailIds": ["msg-001"]},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["action"] == "archive"

    @pytest.mark.asyncio
    async def test_quick_action_internal_error(self, handler):
        """Test error handling when action execution fails."""
        with patch.object(handler, "_execute_action", side_effect=RuntimeError("Broken")):
            request = _make_request(
                "POST",
                "/api/inbox/actions",
                body={"action": "archive", "emailIds": ["msg-001"]},
            )
            response = await handler.handle_quick_action(request)

        assert response.status == 500
        data = _parse_response(response)
        assert data["success"] is False


# ============================================================================
# POST /api/inbox/bulk-actions - Bulk Action
# ============================================================================


class TestBulkAction:
    """Tests for handle_bulk_action endpoint."""

    @pytest.mark.asyncio
    async def test_bulk_action_basic(self, handler):
        """Test basic bulk action."""
        request = _make_request(
            "POST",
            "/api/inbox/bulk-actions",
            body={"action": "archive", "filter": "low"},
        )
        response = await handler.handle_bulk_action(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_bulk_action_missing_action(self, handler):
        """Test missing action in bulk request."""
        request = _make_request(
            "POST",
            "/api/inbox/bulk-actions",
            body={"filter": "low"},
        )
        response = await handler.handle_bulk_action(request)

        assert response.status == 400
        data = _parse_response(response)
        assert data["success"] is False
        assert "action and filter are required" in data["error"]

    @pytest.mark.asyncio
    async def test_bulk_action_missing_filter(self, handler):
        """Test missing filter in bulk request."""
        request = _make_request(
            "POST",
            "/api/inbox/bulk-actions",
            body={"action": "archive"},
        )
        response = await handler.handle_bulk_action(request)

        assert response.status == 400
        data = _parse_response(response)
        assert data["success"] is False
        assert "action and filter are required" in data["error"]

    @pytest.mark.asyncio
    async def test_bulk_action_invalid_action(self, handler):
        """Test invalid action in bulk request."""
        request = _make_request(
            "POST",
            "/api/inbox/bulk-actions",
            body={"action": "nuke", "filter": "low"},
        )
        response = await handler.handle_bulk_action(request)

        assert response.status == 400
        data = _parse_response(response)
        assert data["success"] is False
        assert "Invalid action" in data["error"]

    @pytest.mark.asyncio
    async def test_bulk_action_invalid_filter(self, handler):
        """Test invalid filter in bulk request."""
        request = _make_request(
            "POST",
            "/api/inbox/bulk-actions",
            body={"action": "archive", "filter": "invalid_filter"},
        )
        response = await handler.handle_bulk_action(request)

        assert response.status == 400
        data = _parse_response(response)
        assert data["success"] is False
        assert "Invalid filter" in data["error"]

    @pytest.mark.asyncio
    async def test_all_valid_filters(self, handler):
        """Test that all allowed bulk filters are accepted."""
        for f in ALLOWED_BULK_FILTERS:
            request = _make_request(
                "POST",
                "/api/inbox/bulk-actions",
                body={"action": "archive", "filter": f},
            )
            response = await handler.handle_bulk_action(request)
            assert response.status == 200, f"Filter '{f}' should be accepted"

    @pytest.mark.asyncio
    async def test_bulk_action_no_matching_emails(self, handler):
        """Test bulk action when no emails match filter."""
        request = _make_request(
            "POST",
            "/api/inbox/bulk-actions",
            body={"action": "archive", "filter": "spam"},
        )
        response = await handler.handle_bulk_action(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True
        assert data["processed"] == 0
        assert "No emails matched" in data["message"]

    @pytest.mark.asyncio
    async def test_bulk_action_with_matching_emails(self, handler):
        """Test bulk action when emails match filter in cache."""
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("email-1", {"id": "email-1", "priority": "low", "from": "a@b.com"})
        _email_cache.set("email-2", {"id": "email-2", "priority": "low", "from": "c@d.com"})
        _email_cache.set("email-3", {"id": "email-3", "priority": "high", "from": "e@f.com"})

        request = _make_request(
            "POST",
            "/api/inbox/bulk-actions",
            body={"action": "archive", "filter": "low"},
        )
        response = await handler.handle_bulk_action(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True
        assert data["processed"] == 2

    @pytest.mark.asyncio
    async def test_bulk_action_filter_all(self, handler):
        """Test bulk action with 'all' filter matches everything."""
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("email-1", {"id": "email-1", "priority": "low", "from": "a@b.com"})
        _email_cache.set("email-2", {"id": "email-2", "priority": "high", "from": "c@d.com"})

        request = _make_request(
            "POST",
            "/api/inbox/bulk-actions",
            body={"action": "archive", "filter": "all"},
        )
        response = await handler.handle_bulk_action(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True
        assert data["processed"] == 2

    @pytest.mark.asyncio
    async def test_bulk_action_filter_read(self, handler):
        """Test bulk action with 'read' filter."""
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("email-1", {"id": "email-1", "unread": False, "from": "a@b.com"})
        _email_cache.set("email-2", {"id": "email-2", "unread": True, "from": "c@d.com"})

        request = _make_request(
            "POST",
            "/api/inbox/bulk-actions",
            body={"action": "archive", "filter": "read"},
        )
        response = await handler.handle_bulk_action(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True
        assert data["processed"] == 1

    @pytest.mark.asyncio
    async def test_bulk_action_filter_deferred(self, handler):
        """Test bulk action with 'deferred' filter."""
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("email-1", {"id": "email-1", "priority": "defer", "from": "a@b.com"})
        _email_cache.set("email-2", {"id": "email-2", "priority": "high", "from": "c@d.com"})

        request = _make_request(
            "POST",
            "/api/inbox/bulk-actions",
            body={"action": "archive", "filter": "deferred"},
        )
        response = await handler.handle_bulk_action(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True
        assert data["processed"] == 1

    @pytest.mark.asyncio
    async def test_bulk_action_non_string_action(self, handler):
        """Test that non-string action returns error."""
        request = _make_request(
            "POST",
            "/api/inbox/bulk-actions",
            body={"action": 123, "filter": "low"},
        )
        response = await handler.handle_bulk_action(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_bulk_action_non_string_filter(self, handler):
        """Test that non-string filter returns error."""
        request = _make_request(
            "POST",
            "/api/inbox/bulk-actions",
            body={"action": "archive", "filter": 123},
        )
        response = await handler.handle_bulk_action(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_bulk_action_invalid_params(self, handler):
        """Test bulk action with invalid params."""
        request = _make_request(
            "POST",
            "/api/inbox/bulk-actions",
            body={"action": "archive", "filter": "low", "params": "not-a-dict"},
        )
        response = await handler.handle_bulk_action(request)

        assert response.status == 400
        data = _parse_response(response)
        assert "Invalid params" in data["error"]

    @pytest.mark.asyncio
    async def test_bulk_action_case_insensitive(self, handler):
        """Test that bulk action names are case-insensitive."""
        request = _make_request(
            "POST",
            "/api/inbox/bulk-actions",
            body={"action": "ARCHIVE", "filter": "LOW"},
        )
        response = await handler.handle_bulk_action(request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_bulk_action_internal_error(self, handler):
        """Test error handling when bulk action fails."""
        with patch.object(handler, "_get_emails_by_filter", side_effect=RuntimeError("Fail")):
            request = _make_request(
                "POST",
                "/api/inbox/bulk-actions",
                body={"action": "archive", "filter": "all"},
            )
            response = await handler.handle_bulk_action(request)

        assert response.status == 500
        data = _parse_response(response)
        assert data["success"] is False


# ============================================================================
# GET /api/inbox/sender-profile - Sender Profile
# ============================================================================


class TestSenderProfile:
    """Tests for handle_get_sender_profile endpoint."""

    @pytest.mark.asyncio
    async def test_get_sender_profile_basic(self, handler):
        """Test basic sender profile lookup."""
        request = _make_request(
            "GET",
            "/api/inbox/sender-profile",
            query="email=user@example.com",
        )
        response = await handler.handle_get_sender_profile(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True
        assert "profile" in data
        assert data["profile"]["email"] == "user@example.com"

    @pytest.mark.asyncio
    async def test_get_sender_profile_structure(self, handler):
        """Test sender profile response structure."""
        request = _make_request(
            "GET",
            "/api/inbox/sender-profile",
            query="email=user@example.com",
        )
        response = await handler.handle_get_sender_profile(request)

        data = _parse_response(response)
        profile = data["profile"]
        assert "email" in profile
        assert "name" in profile
        assert "isVip" in profile
        assert "isInternal" in profile
        assert "responseRate" in profile
        assert "avgResponseTime" in profile
        assert "totalEmails" in profile
        assert "lastContact" in profile

    @pytest.mark.asyncio
    async def test_get_sender_profile_missing_email(self, handler):
        """Test missing email parameter."""
        request = _make_request("GET", "/api/inbox/sender-profile")

        response = await handler.handle_get_sender_profile(request)

        assert response.status == 400
        data = _parse_response(response)
        assert data["success"] is False
        assert "email parameter is required" in data["error"]

    @pytest.mark.asyncio
    async def test_get_sender_profile_invalid_email(self, handler):
        """Test invalid email address."""
        request = _make_request(
            "GET",
            "/api/inbox/sender-profile",
            query="email=not-an-email",
        )
        response = await handler.handle_get_sender_profile(request)

        assert response.status == 400
        data = _parse_response(response)
        assert data["success"] is False
        assert "Invalid email address" in data["error"]

    @pytest.mark.asyncio
    async def test_get_sender_profile_empty_email(self, handler):
        """Test empty email parameter."""
        request = _make_request(
            "GET",
            "/api/inbox/sender-profile",
            query="email=",
        )
        response = await handler.handle_get_sender_profile(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_sender_profile_name_extraction(self, handler):
        """Test that name is extracted from email address."""
        request = _make_request(
            "GET",
            "/api/inbox/sender-profile",
            query="email=john.doe@example.com",
        )
        response = await handler.handle_get_sender_profile(request)

        data = _parse_response(response)
        assert data["profile"]["name"] == "john.doe"

    @pytest.mark.asyncio
    async def test_sender_profile_internal_error(self, handler):
        """Test error handling when profile lookup fails."""
        with patch.object(handler, "_get_sender_profile", side_effect=RuntimeError("Fail")):
            request = _make_request(
                "GET",
                "/api/inbox/sender-profile",
                query="email=user@example.com",
            )
            response = await handler.handle_get_sender_profile(request)

        assert response.status == 500
        data = _parse_response(response)
        assert data["success"] is False


# ============================================================================
# GET /api/inbox/daily-digest - Daily Digest
# ============================================================================


class TestDailyDigest:
    """Tests for handle_get_daily_digest endpoint."""

    @pytest.mark.asyncio
    async def test_get_daily_digest(self, handler):
        """Test fetching daily digest."""
        request = _make_request("GET", "/api/inbox/daily-digest")

        response = await handler.handle_get_daily_digest(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True
        assert "digest" in data

    @pytest.mark.asyncio
    async def test_daily_digest_structure(self, handler):
        """Test daily digest response structure."""
        request = _make_request("GET", "/api/inbox/daily-digest")

        response = await handler.handle_get_daily_digest(request)

        data = _parse_response(response)
        digest = data["digest"]
        assert "emailsReceived" in digest
        assert "emailsProcessed" in digest
        assert "criticalHandled" in digest
        assert "timeSaved" in digest
        assert "topSenders" in digest
        assert "categoryBreakdown" in digest

    @pytest.mark.asyncio
    async def test_daily_digest_with_cached_emails(self, handler):
        """Test digest reflects cached email data."""
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("email-1", {
            "id": "email-1",
            "priority": "critical",
            "from": "boss@company.com",
            "category": "Work",
        })
        _email_cache.set("email-2", {
            "id": "email-2",
            "priority": "low",
            "from": "news@example.com",
            "category": "Newsletter",
        })

        request = _make_request("GET", "/api/inbox/daily-digest")
        response = await handler.handle_get_daily_digest(request)

        data = _parse_response(response)
        digest = data["digest"]
        assert digest["emailsReceived"] == 2
        assert digest["criticalHandled"] == 1

    @pytest.mark.asyncio
    async def test_daily_digest_empty_cache(self, handler):
        """Test digest with empty cache."""
        request = _make_request("GET", "/api/inbox/daily-digest")

        response = await handler.handle_get_daily_digest(request)

        data = _parse_response(response)
        digest = data["digest"]
        assert digest["emailsReceived"] == 0

    @pytest.mark.asyncio
    async def test_daily_digest_internal_error(self, handler):
        """Test error handling when digest calculation fails."""
        with patch.object(handler, "_calculate_daily_digest", side_effect=RuntimeError("Fail")):
            request = _make_request("GET", "/api/inbox/daily-digest")
            response = await handler.handle_get_daily_digest(request)

        assert response.status == 500
        data = _parse_response(response)
        assert data["success"] is False


# ============================================================================
# POST /api/inbox/reprioritize - Reprioritize
# ============================================================================


class TestReprioritize:
    """Tests for handle_reprioritize endpoint."""

    @pytest.mark.asyncio
    async def test_reprioritize_all(self, handler):
        """Test reprioritizing all emails (no specific IDs)."""
        request = _make_request(
            "POST",
            "/api/inbox/reprioritize",
            body={},
        )
        response = await handler.handle_reprioritize(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True
        assert "reprioritized" in data
        assert "changes" in data

    @pytest.mark.asyncio
    async def test_reprioritize_specific_emails(self, handler):
        """Test reprioritizing specific email IDs."""
        request = _make_request(
            "POST",
            "/api/inbox/reprioritize",
            body={"emailIds": ["msg-001", "msg-002"]},
        )
        response = await handler.handle_reprioritize(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_reprioritize_with_force_tier(self, handler):
        """Test reprioritizing with forced tier."""
        request = _make_request(
            "POST",
            "/api/inbox/reprioritize",
            body={"force_tier": "tier_1_rules"},
        )
        response = await handler.handle_reprioritize(request)

        assert response.status == 200
        data = _parse_response(response)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_reprioritize_all_valid_tiers(self, handler):
        """Test all allowed force_tier values."""
        for tier in ALLOWED_FORCE_TIERS:
            request = _make_request(
                "POST",
                "/api/inbox/reprioritize",
                body={"force_tier": tier},
            )
            response = await handler.handle_reprioritize(request)
            assert response.status == 200, f"Tier '{tier}' should be accepted"

    @pytest.mark.asyncio
    async def test_reprioritize_invalid_force_tier(self, handler):
        """Test rejection of invalid force_tier."""
        request = _make_request(
            "POST",
            "/api/inbox/reprioritize",
            body={"force_tier": "tier_99_godmode"},
        )
        response = await handler.handle_reprioritize(request)

        assert response.status == 400
        data = _parse_response(response)
        assert data["success"] is False
        assert "Invalid force_tier" in data["error"]

    @pytest.mark.asyncio
    async def test_reprioritize_force_tier_not_string(self, handler):
        """Test that non-string force_tier is rejected."""
        request = _make_request(
            "POST",
            "/api/inbox/reprioritize",
            body={"force_tier": 123},
        )
        response = await handler.handle_reprioritize(request)

        assert response.status == 400
        data = _parse_response(response)
        assert "force_tier must be a string" in data["error"]

    @pytest.mark.asyncio
    async def test_reprioritize_email_ids_not_list(self, handler):
        """Test that non-list emailIds is rejected."""
        request = _make_request(
            "POST",
            "/api/inbox/reprioritize",
            body={"emailIds": "msg-001"},
        )
        response = await handler.handle_reprioritize(request)

        assert response.status == 400
        data = _parse_response(response)
        assert "emailIds must be a list" in data["error"]

    @pytest.mark.asyncio
    async def test_reprioritize_too_many_email_ids(self, handler):
        """Test that too many email IDs are rejected."""
        ids = [f"msg-{i}" for i in range(MAX_EMAIL_IDS_PER_REQUEST + 1)]
        request = _make_request(
            "POST",
            "/api/inbox/reprioritize",
            body={"emailIds": ids},
        )
        response = await handler.handle_reprioritize(request)

        assert response.status == 400
        data = _parse_response(response)
        assert "exceeds maximum" in data["error"]

    @pytest.mark.asyncio
    async def test_reprioritize_invalid_email_id(self, handler):
        """Test that invalid email ID is rejected."""
        request = _make_request(
            "POST",
            "/api/inbox/reprioritize",
            body={"emailIds": ["valid-id", "invalid@id!"]},
        )
        response = await handler.handle_reprioritize(request)

        assert response.status == 400
        data = _parse_response(response)
        assert "Invalid email ID" in data["error"]

    @pytest.mark.asyncio
    async def test_reprioritize_force_tier_case_insensitive(self, handler):
        """Test that force_tier is case-insensitive."""
        request = _make_request(
            "POST",
            "/api/inbox/reprioritize",
            body={"force_tier": "TIER_1_RULES"},
        )
        response = await handler.handle_reprioritize(request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_reprioritize_internal_error(self, handler):
        """Test error handling when reprioritization fails."""
        with patch.object(handler, "_reprioritize_emails", side_effect=RuntimeError("Fail")):
            request = _make_request(
                "POST",
                "/api/inbox/reprioritize",
                body={},
            )
            response = await handler.handle_reprioritize(request)

        assert response.status == 500
        data = _parse_response(response)
        assert data["success"] is False


# ============================================================================
# Action Sanitization (InboxActionsMixin._sanitize_action_params)
# ============================================================================


class TestSanitizeActionParams:
    """Tests for _sanitize_action_params method."""

    def test_snooze_valid_duration(self, handler):
        result = handler._sanitize_action_params("snooze", {"duration": "1h"})
        assert result["duration"] == "1h"

    def test_snooze_all_valid_durations(self, handler):
        for d in ALLOWED_SNOOZE_DURATIONS:
            result = handler._sanitize_action_params("snooze", {"duration": d})
            assert result["duration"] == d

    def test_snooze_invalid_duration_defaults(self, handler):
        result = handler._sanitize_action_params("snooze", {"duration": "99y"})
        assert result["duration"] == "1d"

    def test_snooze_missing_duration_defaults(self, handler):
        result = handler._sanitize_action_params("snooze", {})
        assert result["duration"] == "1d"

    def test_reply_sanitizes_body(self, handler):
        result = handler._sanitize_action_params("reply", {"body": "  Hello!  "})
        assert result["body"] == "Hello!"

    def test_reply_empty_body(self, handler):
        result = handler._sanitize_action_params("reply", {"body": ""})
        assert result["body"] == ""

    def test_forward_valid_email(self, handler):
        result = handler._sanitize_action_params("forward", {"to": "user@example.com"})
        assert result["to"] == "user@example.com"

    def test_forward_invalid_email(self, handler):
        result = handler._sanitize_action_params("forward", {"to": "not-an-email"})
        assert result["to"] == ""

    def test_mark_vip_with_sender(self, handler):
        result = handler._sanitize_action_params("mark_vip", {"sender": "vip@example.com"})
        assert result["sender"] == "vip@example.com"

    def test_mark_vip_invalid_sender(self, handler):
        result = handler._sanitize_action_params("mark_vip", {"sender": "not-an-email"})
        assert "sender" not in result

    def test_block_with_sender(self, handler):
        result = handler._sanitize_action_params("block", {"sender": "spam@evil.com"})
        assert result["sender"] == "spam@evil.com"

    def test_archive_returns_empty(self, handler):
        result = handler._sanitize_action_params("archive", {"extra": "ignored"})
        assert result == {}

    def test_spam_returns_empty(self, handler):
        result = handler._sanitize_action_params("spam", {"extra": "ignored"})
        assert result == {}

    def test_delete_returns_empty(self, handler):
        result = handler._sanitize_action_params("delete", {"extra": "ignored"})
        assert result == {}

    def test_mark_important_returns_empty(self, handler):
        result = handler._sanitize_action_params("mark_important", {"extra": "ignored"})
        assert result == {}


# ============================================================================
# IterableTTLCache
# ============================================================================


class TestIterableTTLCache:
    """Tests for IterableTTLCache wrapper."""

    @pytest.fixture
    def cache(self):
        """Create a test cache."""
        with patch("aragora.server.handlers.inbox_command.HybridTTLCache") as mock_htc:
            mock_inner = MagicMock()
            mock_inner.get.return_value = None
            mock_inner.invalidate.return_value = True
            mock_inner.stats = {"hits": 0, "misses": 0}
            mock_htc.return_value = mock_inner

            c = IterableTTLCache(name="test", maxsize=100, ttl_seconds=60)
            # Override the _cache with a simple dict-backed mock
            store: dict[str, Any] = {}

            def mock_get(key):
                return store.get(key)

            def mock_set(key, value):
                store[key] = value

            def mock_invalidate(key):
                return store.pop(key, None) is not None

            c._cache = MagicMock()
            c._cache.get = mock_get
            c._cache.set = mock_set
            c._cache.invalidate = mock_invalidate
            c._cache.stats = {"hits": 0, "misses": 0}
            return c

    def test_set_and_get(self, cache):
        cache.set("key1", {"data": "value"})
        assert cache.get("key1") == {"data": "value"}

    def test_dict_style_set_get(self, cache):
        cache["key1"] = {"data": "value"}
        assert cache["key1"] == {"data": "value"}

    def test_get_missing_returns_none(self, cache):
        assert cache.get("nonexistent") is None

    def test_dict_get_missing_raises_keyerror(self, cache):
        with pytest.raises(KeyError):
            _ = cache["nonexistent"]

    def test_contains(self, cache):
        cache.set("key1", {"data": "value"})
        assert "key1" in cache
        assert "key2" not in cache

    def test_items(self, cache):
        cache.set("k1", {"a": 1})
        cache.set("k2", {"b": 2})
        items = cache.items()
        assert len(items) == 2
        keys = {k for k, v in items}
        assert keys == {"k1", "k2"}

    def test_values(self, cache):
        cache.set("k1", {"a": 1})
        cache.set("k2", {"b": 2})
        vals = cache.values()
        assert len(vals) == 2

    def test_invalidate(self, cache):
        cache.set("k1", {"a": 1})
        assert "k1" in cache
        cache.invalidate("k1")
        assert "k1" not in cache

    def test_len(self, cache):
        assert len(cache) == 0
        cache.set("k1", {"a": 1})
        assert len(cache) == 1
        cache.set("k2", {"b": 2})
        assert len(cache) == 2

    def test_clear(self, cache):
        cache.set("k1", {"a": 1})
        cache.set("k2", {"b": 2})
        cache.clear()
        assert len(cache) == 0

    def test_stats(self, cache):
        stats = cache.stats
        assert isinstance(stats, dict)

    def test_items_removes_expired_keys(self, cache):
        """Test that items() cleans up keys whose values have expired."""
        cache.set("k1", {"a": 1})
        # Simulate expiry by making get return None for k1
        original_get = cache._cache.get
        def mock_get(key):
            if key == "k1":
                return None
            return original_get(key)
        cache._cache.get = mock_get

        items = cache.items()
        assert len(items) == 0
        assert len(cache) == 0  # Key was cleaned up


# ============================================================================
# Authentication Tests
# ============================================================================


class TestAuthentication:
    """Tests for authentication/permission checks."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_inbox_requires_authentication(self):
        """Test that inbox endpoints require authentication."""
        with patch(
            "aragora.server.handlers.inbox_command.ServiceRegistry"
        ) as mock_reg_cls:
            mock_reg = MagicMock()
            mock_reg.has.return_value = False
            mock_reg_cls.get.return_value = mock_reg

            h = InboxCommandHandler()
            h._initialized = True

            request = _make_request("GET", "/api/inbox/command")

            with pytest.raises(Exception):
                # Should raise HTTPUnauthorized or HTTPForbidden
                await h.handle_get_inbox(request)

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_quick_action_requires_authentication(self):
        """Test that quick action requires authentication."""
        with patch(
            "aragora.server.handlers.inbox_command.ServiceRegistry"
        ) as mock_reg_cls:
            mock_reg = MagicMock()
            mock_reg.has.return_value = False
            mock_reg_cls.get.return_value = mock_reg

            h = InboxCommandHandler()
            h._initialized = True

            request = _make_request(
                "POST",
                "/api/inbox/actions",
                body={"action": "archive", "emailIds": ["msg-001"]},
            )

            with pytest.raises(Exception):
                await h.handle_quick_action(request)

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_bulk_action_requires_authentication(self):
        """Test that bulk action requires authentication."""
        with patch(
            "aragora.server.handlers.inbox_command.ServiceRegistry"
        ) as mock_reg_cls:
            mock_reg = MagicMock()
            mock_reg.has.return_value = False
            mock_reg_cls.get.return_value = mock_reg

            h = InboxCommandHandler()
            h._initialized = True

            request = _make_request(
                "POST",
                "/api/inbox/bulk-actions",
                body={"action": "archive", "filter": "low"},
            )

            with pytest.raises(Exception):
                await h.handle_bulk_action(request)

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_sender_profile_requires_authentication(self):
        """Test that sender profile requires authentication."""
        with patch(
            "aragora.server.handlers.inbox_command.ServiceRegistry"
        ) as mock_reg_cls:
            mock_reg = MagicMock()
            mock_reg.has.return_value = False
            mock_reg_cls.get.return_value = mock_reg

            h = InboxCommandHandler()
            h._initialized = True

            request = _make_request(
                "GET",
                "/api/inbox/sender-profile",
                query="email=user@example.com",
            )

            with pytest.raises(Exception):
                await h.handle_get_sender_profile(request)

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_daily_digest_requires_authentication(self):
        """Test that daily digest requires authentication."""
        with patch(
            "aragora.server.handlers.inbox_command.ServiceRegistry"
        ) as mock_reg_cls:
            mock_reg = MagicMock()
            mock_reg.has.return_value = False
            mock_reg_cls.get.return_value = mock_reg

            h = InboxCommandHandler()
            h._initialized = True

            request = _make_request("GET", "/api/inbox/daily-digest")

            with pytest.raises(Exception):
                await h.handle_get_daily_digest(request)

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_reprioritize_requires_authentication(self):
        """Test that reprioritize requires authentication."""
        with patch(
            "aragora.server.handlers.inbox_command.ServiceRegistry"
        ) as mock_reg_cls:
            mock_reg = MagicMock()
            mock_reg.has.return_value = False
            mock_reg_cls.get.return_value = mock_reg

            h = InboxCommandHandler()
            h._initialized = True

            request = _make_request(
                "POST",
                "/api/inbox/reprioritize",
                body={},
            )

            with pytest.raises(Exception):
                await h.handle_reprioritize(request)


# ============================================================================
# Register Routes
# ============================================================================


class TestRegisterRoutes:
    """Tests for register_routes function."""

    def test_register_routes_creates_routes(self):
        """Test that register_routes adds all expected routes."""
        app = MagicMock()

        with patch(
            "aragora.server.handlers.inbox_command.ServiceRegistry"
        ) as mock_reg_cls:
            mock_reg = MagicMock()
            mock_reg.has.return_value = False
            mock_reg_cls.get.return_value = mock_reg

            register_routes(app)

        # Verify routes were registered
        assert app.router.add_get.called
        assert app.router.add_post.called

        # Check GET routes
        get_paths = [call[0][0] for call in app.router.add_get.call_args_list]
        assert "/api/inbox/command" in get_paths
        assert "/api/inbox/sender-profile" in get_paths
        assert "/api/inbox/daily-digest" in get_paths
        assert "/api/v1/inbox/command" in get_paths
        assert "/api/v1/inbox/sender-profile" in get_paths
        assert "/api/v1/inbox/daily-digest" in get_paths
        # Backward-compat aliases
        assert "/api/email/daily-digest" in get_paths
        assert "/api/email/sender-profile" in get_paths

        # Check POST routes
        post_paths = [call[0][0] for call in app.router.add_post.call_args_list]
        assert "/api/inbox/actions" in post_paths
        assert "/api/inbox/bulk-actions" in post_paths
        assert "/api/inbox/reprioritize" in post_paths
        assert "/api/v1/inbox/actions" in post_paths
        assert "/api/v1/inbox/bulk-actions" in post_paths
        assert "/api/v1/inbox/reprioritize" in post_paths


# ============================================================================
# Allowed Constants
# ============================================================================


class TestAllowedConstants:
    """Tests that allowlist constants have expected values."""

    def test_allowed_actions_contains_expected(self):
        expected = {"archive", "snooze", "reply", "forward", "spam", "mark_important", "mark_vip", "block", "delete"}
        assert ALLOWED_ACTIONS == expected

    def test_allowed_bulk_filters_contains_expected(self):
        expected = {"low", "deferred", "spam", "read", "all"}
        assert ALLOWED_BULK_FILTERS == expected

    def test_allowed_priority_filters_contains_expected(self):
        expected = {"critical", "high", "medium", "low", "defer"}
        assert ALLOWED_PRIORITY_FILTERS == expected

    def test_allowed_force_tiers_contains_expected(self):
        expected = {"tier_1_rules", "tier_2_lightweight", "tier_3_debate"}
        assert ALLOWED_FORCE_TIERS == expected

    def test_allowed_snooze_durations_contains_expected(self):
        expected = {"1h", "3h", "1d", "3d", "1w"}
        assert ALLOWED_SNOOZE_DURATIONS == expected

    def test_constants_are_frozensets(self):
        assert isinstance(ALLOWED_ACTIONS, frozenset)
        assert isinstance(ALLOWED_BULK_FILTERS, frozenset)
        assert isinstance(ALLOWED_PRIORITY_FILTERS, frozenset)
        assert isinstance(ALLOWED_FORCE_TIERS, frozenset)
        assert isinstance(ALLOWED_SNOOZE_DURATIONS, frozenset)


# ============================================================================
# Demo Mode (no gmail_connector)
# ============================================================================


class TestDemoMode:
    """Tests for demo mode when services are not available."""

    @pytest.mark.asyncio
    async def test_inbox_returns_demo_emails(self, handler):
        """Test that inbox returns demo data when no Gmail connector."""
        request = _make_request("GET", "/api/inbox/command")

        response = await handler.handle_get_inbox(request)

        data = _parse_response(response)
        assert data["success"] is True
        assert len(data["emails"]) > 0
        # Demo emails have known IDs (offset min_val=1, so first may be skipped)
        ids = [e["id"] for e in data["emails"]]
        # At least some demo emails should be present
        assert any(eid.startswith("demo_") for eid in ids)

    @pytest.mark.asyncio
    async def test_actions_work_in_demo_mode(self, handler):
        """Test that actions work in demo mode (no Gmail connector)."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "archive", "emailIds": ["demo_1"]},
        )
        response = await handler.handle_quick_action(request)

        data = _parse_response(response)
        assert data["success"] is True
        assert data["results"][0]["success"] is True
        assert data["results"][0]["result"]["demo"] is True

    @pytest.mark.asyncio
    async def test_snooze_in_demo_mode(self, handler):
        """Test snooze returns demo result."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "snooze", "emailIds": ["demo_1"], "params": {"duration": "3h"}},
        )
        response = await handler.handle_quick_action(request)

        data = _parse_response(response)
        assert data["success"] is True
        result = data["results"][0]["result"]
        assert result["snoozed"] is True
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_reply_in_demo_mode(self, handler):
        """Test reply creates demo draft."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "reply", "emailIds": ["demo_1"], "params": {"body": "Reply text"}},
        )
        response = await handler.handle_quick_action(request)

        data = _parse_response(response)
        result = data["results"][0]["result"]
        assert "draftId" in result
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_forward_in_demo_mode(self, handler):
        """Test forward creates demo draft."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "forward", "emailIds": ["demo_1"], "params": {"to": "user@example.com"}},
        )
        response = await handler.handle_quick_action(request)

        data = _parse_response(response)
        result = data["results"][0]["result"]
        assert "draftId" in result
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_reprioritize_without_prioritizer(self, handler):
        """Test reprioritize returns error when no prioritizer."""
        request = _make_request(
            "POST",
            "/api/inbox/reprioritize",
            body={},
        )
        response = await handler.handle_reprioritize(request)

        data = _parse_response(response)
        assert data["success"] is True
        assert data["reprioritized"] == 0

    @pytest.mark.asyncio
    async def test_sender_profile_without_services(self, handler):
        """Test sender profile basic info when no sender history."""
        request = _make_request(
            "GET",
            "/api/inbox/sender-profile",
            query="email=user@example.com",
        )
        response = await handler.handle_get_sender_profile(request)

        data = _parse_response(response)
        profile = data["profile"]
        assert profile["isVip"] is False
        assert profile["totalEmails"] == 0
        assert profile["responseRate"] == 0.0


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for various edge cases."""

    @pytest.mark.asyncio
    async def test_action_empty_string_action(self, handler):
        """Test empty string action is rejected."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "", "emailIds": ["msg-001"]},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_action_whitespace_only_action(self, handler):
        """Test whitespace-only action is rejected."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "   ", "emailIds": ["msg-001"]},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_bulk_action_empty_action(self, handler):
        """Test empty action in bulk request."""
        request = _make_request(
            "POST",
            "/api/inbox/bulk-actions",
            body={"action": "", "filter": "low"},
        )
        response = await handler.handle_bulk_action(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_bulk_action_empty_filter(self, handler):
        """Test empty filter in bulk request."""
        request = _make_request(
            "POST",
            "/api/inbox/bulk-actions",
            body={"action": "archive", "filter": ""},
        )
        response = await handler.handle_bulk_action(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_command_injection_in_action(self, handler):
        """Test command injection attempt in action field."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "archive; rm -rf /", "emailIds": ["msg-001"]},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 400
        data = _parse_response(response)
        assert "Invalid action" in data["error"]

    @pytest.mark.asyncio
    async def test_sql_injection_in_email_id(self, handler):
        """Test SQL injection attempt in email ID."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "archive", "emailIds": ["'; DROP TABLE emails;--"]},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_xss_in_email_id(self, handler):
        """Test XSS attempt in email ID."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "archive", "emailIds": ["<script>alert(1)</script>"]},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_null_email_id_in_list(self, handler):
        """Test null value in email ID list."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "archive", "emailIds": [None]},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_priority_filter_with_whitespace(self, handler):
        """Test priority filter with leading/trailing whitespace is trimmed."""
        request = _make_request(
            "GET",
            "/api/inbox/command",
            query="priority=%20critical%20",
        )
        response = await handler.handle_get_inbox(request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_none_params_defaults_to_empty(self, handler):
        """Test that None params default to empty dict."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "archive", "emailIds": ["msg-001"]},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_snooze_non_string_duration(self, handler):
        """Test snooze with non-string duration defaults to 1d."""
        request = _make_request(
            "POST",
            "/api/inbox/actions",
            body={"action": "snooze", "emailIds": ["msg-001"], "params": {"duration": 42}},
        )
        response = await handler.handle_quick_action(request)

        assert response.status == 200
        data = _parse_response(response)
        # Should use default duration, not crash
        assert data["success"] is True
