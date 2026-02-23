"""Tests for EmailHandler (aragora/server/handlers/email/handler.py).

Covers all routes, HTTP methods, success/error paths, validation, and edge cases:
- can_handle() routing for static and prefix routes
- POST /api/v1/email/prioritize
- POST /api/v1/email/rank-inbox
- POST /api/v1/email/feedback (+ validation)
- POST /api/v1/email/feedback/batch (+ validation)
- POST /api/v1/email/categorize
- POST /api/v1/email/categorize/batch (+ validation)
- POST /api/v1/email/categorize/apply-label (+ validation)
- GET  /api/v1/email/inbox (+ needs_auth, query param parsing)
- GET  /api/v1/email/config
- PUT  /api/v1/email/config
- POST /api/v1/email/vip
- DELETE /api/v1/email/vip
- POST /api/v1/email/gmail/oauth/url (+ validation)
- POST /api/v1/email/gmail/oauth/callback (+ validation)
- GET  /api/v1/email/gmail/status
- GET  /api/v1/email/context/:email_address
- POST /api/v1/email/context/boost
- _get_user_id() and _get_auth_context() helpers
- handle() returns None (async delegation)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.email.handler import EmailHandler
from aragora.server.handlers.utils.responses import HandlerResult


# ============================================================================
# Helpers
# ============================================================================


def _body(result: HandlerResult) -> dict:
    """Parse HandlerResult.body bytes into dict."""
    return json.loads(result.body)


def _status(result: HandlerResult) -> int:
    """Extract HTTP status code from a HandlerResult."""
    return result.status_code


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def ctx():
    """Minimal server context for EmailHandler."""
    return {}


@pytest.fixture
def ctx_with_auth():
    """Server context with a mock auth_context that has a user_id."""
    auth = MagicMock()
    auth.user_id = "user-42"
    return {"auth_context": auth}


@pytest.fixture
def handler(ctx):
    """Create an EmailHandler with a minimal context."""
    return EmailHandler(ctx)


@pytest.fixture
def handler_with_auth(ctx_with_auth):
    """Create an EmailHandler with an authenticated context."""
    return EmailHandler(ctx_with_auth)


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler."""
    mock = MagicMock()
    mock.command = "GET"
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = {}
    return mock


# ============================================================================
# can_handle() routing tests
# ============================================================================


class TestCanHandle:
    """Tests for EmailHandler.can_handle() routing logic."""

    @pytest.mark.parametrize("path", EmailHandler.ROUTES)
    def test_can_handle_all_static_routes(self, handler, path):
        """Every route listed in ROUTES should be handled."""
        assert handler.can_handle(path) is True

    def test_can_handle_prefix_route_with_email(self, handler):
        """Dynamic context route with email address is handled."""
        assert handler.can_handle("/api/v1/email/context/user@example.com") is True

    def test_can_handle_prefix_route_with_slug(self, handler):
        """Dynamic context route with arbitrary slug is handled."""
        assert handler.can_handle("/api/v1/email/context/some-slug") is True

    def test_cannot_handle_prefix_without_trailing_slash(self, handler):
        """The prefix without trailing slash is not handled (rstrip check)."""
        assert handler.can_handle("/api/v1/email/context") is False

    def test_prefix_with_trailing_slash_is_handled(self, handler):
        """The prefix with trailing slash passes the rstrip check."""
        # path "/api/v1/email/context/" != prefix.rstrip("/") == "/api/v1/email/context"
        assert handler.can_handle("/api/v1/email/context/") is True

    def test_cannot_handle_unknown_path(self, handler):
        """Unrelated paths are not handled."""
        assert handler.can_handle("/api/v1/debates/list") is False

    def test_cannot_handle_partial_match(self, handler):
        """A path that starts like a route but is different is not handled."""
        assert handler.can_handle("/api/v1/email/prioritize/extra") is False


# ============================================================================
# handle() returns None (async delegation)
# ============================================================================


class TestHandleReturnsNone:
    """The synchronous handle() method always returns None."""

    def test_handle_returns_none(self, handler, mock_http_handler):
        result = handler.handle("/api/v1/email/prioritize", {}, mock_http_handler)
        assert result is None


# ============================================================================
# _get_user_id() and _get_auth_context() helpers
# ============================================================================


class TestHelpers:
    """Tests for _get_user_id and _get_auth_context."""

    def test_get_user_id_default_when_no_auth(self, handler):
        """Returns 'default' when no auth context."""
        assert handler._get_user_id() == "default"

    def test_get_user_id_from_auth_context(self, handler_with_auth):
        """Returns user_id from auth context when present."""
        assert handler_with_auth._get_user_id() == "user-42"

    def test_get_auth_context_none(self, handler):
        """Returns None when no auth context."""
        assert handler._get_auth_context() is None

    def test_get_auth_context_present(self, handler_with_auth):
        """Returns the auth context object when present."""
        assert handler_with_auth._get_auth_context() is not None
        assert handler_with_auth._get_auth_context().user_id == "user-42"

    def test_get_user_id_default_when_no_user_id_attr(self):
        """Returns 'default' when auth_context exists but lacks user_id."""
        ctx = {"auth_context": object()}  # No user_id attribute
        h = EmailHandler(ctx)
        assert h._get_user_id() == "default"


# ============================================================================
# POST /api/v1/email/prioritize
# ============================================================================


class TestPostPrioritize:
    """Tests for handle_post_prioritize."""

    @pytest.mark.asyncio
    async def test_prioritize_success(self, handler):
        """Successful prioritization returns success_response."""
        mock_result = {"success": True, "score": 0.85, "tier": "high"}
        with patch(
            "aragora.server.handlers.email.handler.handle_prioritize_email",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_prioritize(
                {"email": {"id": "msg1", "subject": "Test"}, "force_tier": "tier_1_rules"}
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["data"]["score"] == 0.85

    @pytest.mark.asyncio
    async def test_prioritize_failure(self, handler):
        """Failed prioritization returns error_response with 400."""
        mock_result = {"success": False, "error": "Invalid email data"}
        with patch(
            "aragora.server.handlers.email.handler.handle_prioritize_email",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_prioritize({"email": {}})
        assert _status(result) == 400
        body = _body(result)
        assert body["error"] == "Invalid email data"

    @pytest.mark.asyncio
    async def test_prioritize_failure_unknown_error(self, handler):
        """Failed prioritization without error message uses 'Unknown error'."""
        mock_result = {"success": False}
        with patch(
            "aragora.server.handlers.email.handler.handle_prioritize_email",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_prioritize({"email": {}})
        assert _status(result) == 400
        body = _body(result)
        assert body["error"] == "Unknown error"

    @pytest.mark.asyncio
    async def test_prioritize_empty_email(self, handler):
        """Empty email data is forwarded to underlying handler."""
        mock_result = {"success": True, "score": 0.0}
        with patch(
            "aragora.server.handlers.email.handler.handle_prioritize_email",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            result = await handler.handle_post_prioritize({})
        # email defaults to {}
        mock_fn.assert_awaited_once()
        args = mock_fn.call_args
        assert args[0][0] == {}  # email_data
        assert _status(result) == 200


# ============================================================================
# POST /api/v1/email/rank-inbox
# ============================================================================


class TestPostRankInbox:
    """Tests for handle_post_rank_inbox."""

    @pytest.mark.asyncio
    async def test_rank_inbox_success(self, handler):
        mock_result = {"success": True, "ranked": [{"id": "1", "score": 0.9}]}
        with patch(
            "aragora.server.handlers.email.handler.handle_rank_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_rank_inbox(
                {"emails": [{"id": "1"}], "limit": 10}
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_rank_inbox_failure(self, handler):
        mock_result = {"success": False, "error": "Ranking failed"}
        with patch(
            "aragora.server.handlers.email.handler.handle_rank_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_rank_inbox({"emails": []})
        assert _status(result) == 400
        body = _body(result)
        assert body["error"] == "Ranking failed"

    @pytest.mark.asyncio
    async def test_rank_inbox_passes_limit(self, handler):
        """limit parameter is forwarded to handler."""
        mock_result = {"success": True, "ranked": []}
        with patch(
            "aragora.server.handlers.email.handler.handle_rank_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_post_rank_inbox({"emails": [], "limit": 5})
        assert mock_fn.call_args.kwargs["limit"] == 5


# ============================================================================
# POST /api/v1/email/feedback
# ============================================================================


class TestPostFeedback:
    """Tests for handle_post_feedback."""

    @pytest.mark.asyncio
    async def test_feedback_success(self, handler):
        mock_result = {"success": True, "recorded": True}
        with patch(
            "aragora.server.handlers.email.handler.handle_email_feedback",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_feedback(
                {"email_id": "msg1", "action": "archive", "email": {"id": "msg1"}}
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_feedback_missing_email_id(self, handler):
        """Missing email_id returns 400."""
        result = await handler.handle_post_feedback({"action": "archive"})
        assert _status(result) == 400
        body = _body(result)
        assert "email_id" in body["error"]

    @pytest.mark.asyncio
    async def test_feedback_missing_action(self, handler):
        """Missing action returns 400."""
        result = await handler.handle_post_feedback({"email_id": "msg1"})
        assert _status(result) == 400
        body = _body(result)
        assert "action" in body["error"]

    @pytest.mark.asyncio
    async def test_feedback_missing_both(self, handler):
        """Missing both email_id and action returns 400."""
        result = await handler.handle_post_feedback({})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_feedback_failure(self, handler):
        mock_result = {"success": False, "error": "Feedback store unavailable"}
        with patch(
            "aragora.server.handlers.email.handler.handle_email_feedback",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_feedback(
                {"email_id": "msg1", "action": "read"}
            )
        assert _status(result) == 400
        body = _body(result)
        assert body["error"] == "Feedback store unavailable"


# ============================================================================
# POST /api/v1/email/feedback/batch
# ============================================================================


class TestPostFeedbackBatch:
    """Tests for handle_post_feedback_batch."""

    @pytest.mark.asyncio
    async def test_feedback_batch_success(self, handler):
        mock_result = {"success": True, "processed": 3}
        with patch(
            "aragora.server.handlers.email.handler.handle_feedback_batch",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_feedback_batch(
                {"items": [{"email_id": "1", "action": "read"}]}
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["processed"] == 3

    @pytest.mark.asyncio
    async def test_feedback_batch_empty_items(self, handler):
        """Empty items array returns 400."""
        result = await handler.handle_post_feedback_batch({"items": []})
        assert _status(result) == 400
        body = _body(result)
        assert "items" in body["error"]

    @pytest.mark.asyncio
    async def test_feedback_batch_missing_items(self, handler):
        """Missing items key (defaults to []) returns 400."""
        result = await handler.handle_post_feedback_batch({})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_feedback_batch_failure(self, handler):
        mock_result = {"success": False, "error": "Batch error"}
        with patch(
            "aragora.server.handlers.email.handler.handle_feedback_batch",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_feedback_batch(
                {"items": [{"email_id": "1", "action": "read"}]}
            )
        assert _status(result) == 400
        assert _body(result)["error"] == "Batch error"


# ============================================================================
# POST /api/v1/email/categorize
# ============================================================================


class TestPostCategorize:
    """Tests for handle_post_categorize."""

    @pytest.mark.asyncio
    async def test_categorize_success(self, handler):
        mock_result = {"success": True, "category": "work"}
        with patch(
            "aragora.server.handlers.email.handler.handle_categorize_email",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_categorize(
                {"email": {"subject": "Meeting notes"}}
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["category"] == "work"

    @pytest.mark.asyncio
    async def test_categorize_failure(self, handler):
        mock_result = {"success": False, "error": "Categorizer unavailable"}
        with patch(
            "aragora.server.handlers.email.handler.handle_categorize_email",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_categorize({"email": {}})
        assert _status(result) == 400


# ============================================================================
# POST /api/v1/email/categorize/batch
# ============================================================================


class TestPostCategorizeBatch:
    """Tests for handle_post_categorize_batch."""

    @pytest.mark.asyncio
    async def test_categorize_batch_success(self, handler):
        mock_result = {"success": True, "results": [{"id": "1", "category": "work"}]}
        with patch(
            "aragora.server.handlers.email.handler.handle_categorize_batch",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_categorize_batch(
                {"emails": [{"id": "1"}], "concurrency": 5}
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_categorize_batch_empty_emails(self, handler):
        """Empty emails array returns 400."""
        result = await handler.handle_post_categorize_batch({"emails": []})
        assert _status(result) == 400
        body = _body(result)
        assert "emails" in body["error"]

    @pytest.mark.asyncio
    async def test_categorize_batch_missing_emails(self, handler):
        """Missing emails key (defaults to []) returns 400."""
        result = await handler.handle_post_categorize_batch({})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_categorize_batch_default_concurrency(self, handler):
        """Default concurrency is 10 if not provided."""
        mock_result = {"success": True, "results": []}
        with patch(
            "aragora.server.handlers.email.handler.handle_categorize_batch",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_post_categorize_batch({"emails": [{"id": "1"}]})
        # concurrency defaults to 10
        assert mock_fn.call_args[0][2] == 10

    @pytest.mark.asyncio
    async def test_categorize_batch_failure(self, handler):
        mock_result = {"success": False, "error": "Batch failed"}
        with patch(
            "aragora.server.handlers.email.handler.handle_categorize_batch",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_categorize_batch(
                {"emails": [{"id": "1"}]}
            )
        assert _status(result) == 400


# ============================================================================
# POST /api/v1/email/categorize/apply-label
# ============================================================================


class TestPostCategorizeApplyLabel:
    """Tests for handle_post_categorize_apply_label."""

    @pytest.mark.asyncio
    async def test_apply_label_success(self, handler):
        mock_result = {"success": True, "applied": True}
        with patch(
            "aragora.server.handlers.email.handler.handle_apply_category_label",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_categorize_apply_label(
                {"email_id": "msg1", "category": "work"}
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_apply_label_missing_email_id(self, handler):
        """Missing email_id returns 400."""
        result = await handler.handle_post_categorize_apply_label(
            {"category": "work"}
        )
        assert _status(result) == 400
        body = _body(result)
        assert "email_id" in body["error"]

    @pytest.mark.asyncio
    async def test_apply_label_missing_category(self, handler):
        """Missing category returns 400."""
        result = await handler.handle_post_categorize_apply_label(
            {"email_id": "msg1"}
        )
        assert _status(result) == 400
        body = _body(result)
        assert "category" in body["error"]

    @pytest.mark.asyncio
    async def test_apply_label_missing_both(self, handler):
        """Missing both email_id and category returns 400."""
        result = await handler.handle_post_categorize_apply_label({})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_apply_label_failure(self, handler):
        mock_result = {"success": False, "error": "Label not applied"}
        with patch(
            "aragora.server.handlers.email.handler.handle_apply_category_label",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_categorize_apply_label(
                {"email_id": "msg1", "category": "work"}
            )
        assert _status(result) == 400


# ============================================================================
# GET /api/v1/email/inbox
# ============================================================================


class TestGetInbox:
    """Tests for handle_get_inbox."""

    @pytest.mark.asyncio
    async def test_get_inbox_success(self, handler):
        mock_result = {"success": True, "emails": [{"id": "1"}]}
        with patch(
            "aragora.server.handlers.email.handler.handle_fetch_and_rank_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_get_inbox({})
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_get_inbox_needs_auth(self, handler):
        """When needs_auth is True, returns 401."""
        mock_result = {"needs_auth": True, "error": "Gmail not connected"}
        with patch(
            "aragora.server.handlers.email.handler.handle_fetch_and_rank_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_get_inbox({})
        assert _status(result) == 401
        body = _body(result)
        assert body["error"] == "Gmail not connected"

    @pytest.mark.asyncio
    async def test_get_inbox_error(self, handler):
        """Generic error returns 400."""
        mock_result = {"success": False, "error": "Fetch failed"}
        with patch(
            "aragora.server.handlers.email.handler.handle_fetch_and_rank_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_get_inbox({})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_inbox_unknown_error(self, handler):
        """Error without message defaults to 'Unknown error'."""
        mock_result = {"success": False}
        with patch(
            "aragora.server.handlers.email.handler.handle_fetch_and_rank_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_get_inbox({})
        assert _status(result) == 400
        assert _body(result)["error"] == "Unknown error"

    @pytest.mark.asyncio
    async def test_get_inbox_labels_parsing(self, handler):
        """Labels query param is split on comma."""
        mock_result = {"success": True, "emails": []}
        with patch(
            "aragora.server.handlers.email.handler.handle_fetch_and_rank_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_get_inbox({"labels": "INBOX,IMPORTANT"})
        assert mock_fn.call_args.kwargs["labels"] == ["INBOX", "IMPORTANT"]

    @pytest.mark.asyncio
    async def test_get_inbox_labels_none_when_empty(self, handler):
        """No labels param means labels=None."""
        mock_result = {"success": True, "emails": []}
        with patch(
            "aragora.server.handlers.email.handler.handle_fetch_and_rank_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_get_inbox({})
        assert mock_fn.call_args.kwargs["labels"] is None

    @pytest.mark.asyncio
    async def test_get_inbox_limit_parsing(self, handler):
        """limit query param is parsed as int."""
        mock_result = {"success": True, "emails": []}
        with patch(
            "aragora.server.handlers.email.handler.handle_fetch_and_rank_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_get_inbox({"limit": "25"})
        assert mock_fn.call_args.kwargs["limit"] == 25

    @pytest.mark.asyncio
    async def test_get_inbox_limit_invalid_defaults_to_50(self, handler):
        """Invalid limit defaults to 50."""
        mock_result = {"success": True, "emails": []}
        with patch(
            "aragora.server.handlers.email.handler.handle_fetch_and_rank_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_get_inbox({"limit": "not-a-number"})
        assert mock_fn.call_args.kwargs["limit"] == 50

    @pytest.mark.asyncio
    async def test_get_inbox_include_read_true(self, handler):
        """include_read=true is parsed correctly."""
        mock_result = {"success": True, "emails": []}
        with patch(
            "aragora.server.handlers.email.handler.handle_fetch_and_rank_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_get_inbox({"include_read": "true"})
        assert mock_fn.call_args.kwargs["include_read"] is True

    @pytest.mark.asyncio
    async def test_get_inbox_include_read_false(self, handler):
        """include_read with non-'true' value is False."""
        mock_result = {"success": True, "emails": []}
        with patch(
            "aragora.server.handlers.email.handler.handle_fetch_and_rank_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_get_inbox({"include_read": "false"})
        assert mock_fn.call_args.kwargs["include_read"] is False

    @pytest.mark.asyncio
    async def test_get_inbox_include_read_empty(self, handler):
        """Empty include_read defaults to False."""
        mock_result = {"success": True, "emails": []}
        with patch(
            "aragora.server.handlers.email.handler.handle_fetch_and_rank_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_get_inbox({})
        assert mock_fn.call_args.kwargs["include_read"] is False


# ============================================================================
# GET /api/v1/email/config
# ============================================================================


class TestGetConfig:
    """Tests for handle_get_config."""

    @pytest.mark.asyncio
    async def test_get_config_success(self, handler):
        mock_result = {"vip_emails": ["boss@co.com"], "weights": {}}
        with patch(
            "aragora.server.handlers.email.handler.handle_get_config",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_get_config({})
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["data"]["vip_emails"] == ["boss@co.com"]


# ============================================================================
# PUT /api/v1/email/config
# ============================================================================


class TestPutConfig:
    """Tests for handle_put_config."""

    @pytest.mark.asyncio
    async def test_put_config_success(self, handler):
        mock_result = {"success": True, "updated": True}
        with patch(
            "aragora.server.handlers.email.handler.handle_update_config",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_put_config({"vip_emails": ["new@co.com"]})
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_put_config_failure(self, handler):
        mock_result = {"success": False, "error": "Invalid config key"}
        with patch(
            "aragora.server.handlers.email.handler.handle_update_config",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_put_config({"bad_key": 1})
        assert _status(result) == 400
        assert _body(result)["error"] == "Invalid config key"

    @pytest.mark.asyncio
    async def test_put_config_passes_data(self, handler):
        """Entire data dict is forwarded to handle_update_config."""
        mock_result = {"success": True}
        with patch(
            "aragora.server.handlers.email.handler.handle_update_config",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            payload = {"weights": {"sender": 0.5}, "vip_emails": []}
            await handler.handle_put_config(payload)
        # Second positional arg is the data dict
        assert mock_fn.call_args[0][1] == payload


# ============================================================================
# POST /api/v1/email/vip
# ============================================================================


class TestPostVip:
    """Tests for handle_post_vip."""

    @pytest.mark.asyncio
    async def test_add_vip_email_success(self, handler):
        mock_result = {"success": True, "added": "boss@co.com"}
        with patch(
            "aragora.server.handlers.email.handler.handle_add_vip",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_vip({"email": "boss@co.com"})
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_add_vip_domain_success(self, handler):
        mock_result = {"success": True, "added": "company.com"}
        with patch(
            "aragora.server.handlers.email.handler.handle_add_vip",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_vip({"domain": "company.com"})
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_add_vip_failure(self, handler):
        mock_result = {"success": False, "error": "Already exists"}
        with patch(
            "aragora.server.handlers.email.handler.handle_add_vip",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_vip({"email": "boss@co.com"})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_add_vip_passes_both(self, handler):
        """Both email and domain are forwarded."""
        mock_result = {"success": True}
        with patch(
            "aragora.server.handlers.email.handler.handle_add_vip",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_post_vip({"email": "a@b.com", "domain": "b.com"})
        assert mock_fn.call_args[0][1] == "a@b.com"
        assert mock_fn.call_args[0][2] == "b.com"


# ============================================================================
# DELETE /api/v1/email/vip
# ============================================================================


class TestDeleteVip:
    """Tests for handle_delete_vip."""

    @pytest.mark.asyncio
    async def test_delete_vip_success(self, handler):
        mock_result = {"success": True, "removed": "boss@co.com"}
        with patch(
            "aragora.server.handlers.email.handler.handle_remove_vip",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_delete_vip({"email": "boss@co.com"})
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_delete_vip_failure(self, handler):
        mock_result = {"success": False, "error": "Not found"}
        with patch(
            "aragora.server.handlers.email.handler.handle_remove_vip",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_delete_vip({"email": "nobody@co.com"})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_delete_vip_domain(self, handler):
        """Domain removal is forwarded."""
        mock_result = {"success": True}
        with patch(
            "aragora.server.handlers.email.handler.handle_remove_vip",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_delete_vip({"domain": "old.com"})
        assert mock_fn.call_args[0][2] == "old.com"


# ============================================================================
# POST /api/v1/email/gmail/oauth/url
# ============================================================================


class TestPostGmailOauthUrl:
    """Tests for handle_post_gmail_oauth_url."""

    @pytest.mark.asyncio
    async def test_oauth_url_success(self, handler):
        mock_result = {"success": True, "url": "https://accounts.google.com/o/oauth2/..."}
        with patch(
            "aragora.server.handlers.email.handler.handle_gmail_oauth_url",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_gmail_oauth_url(
                {"redirect_uri": "https://app.example.com/callback", "state": "s1", "scopes": "full"}
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_oauth_url_missing_redirect_uri(self, handler):
        """Missing redirect_uri returns 400."""
        result = await handler.handle_post_gmail_oauth_url({})
        assert _status(result) == 400
        body = _body(result)
        assert "redirect_uri" in body["error"]

    @pytest.mark.asyncio
    async def test_oauth_url_failure(self, handler):
        mock_result = {"success": False, "error": "OAuth not configured"}
        with patch(
            "aragora.server.handlers.email.handler.handle_gmail_oauth_url",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_gmail_oauth_url(
                {"redirect_uri": "https://example.com/cb"}
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_oauth_url_defaults(self, handler):
        """Default state='' and scopes='readonly'."""
        mock_result = {"success": True, "url": "https://example.com"}
        with patch(
            "aragora.server.handlers.email.handler.handle_gmail_oauth_url",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_post_gmail_oauth_url(
                {"redirect_uri": "https://example.com/cb"}
            )
        assert mock_fn.call_args[0][1] == ""  # state default
        assert mock_fn.call_args[0][2] == "readonly"  # scopes default


# ============================================================================
# POST /api/v1/email/gmail/oauth/callback
# ============================================================================


class TestPostGmailOauthCallback:
    """Tests for handle_post_gmail_oauth_callback."""

    @pytest.mark.asyncio
    async def test_oauth_callback_success(self, handler):
        mock_result = {"success": True, "connected": True}
        with patch(
            "aragora.server.handlers.email.handler.handle_gmail_oauth_callback",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_gmail_oauth_callback(
                {"code": "auth_code_123", "redirect_uri": "https://example.com/cb"}
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_oauth_callback_missing_code(self, handler):
        """Missing code returns 400."""
        result = await handler.handle_post_gmail_oauth_callback(
            {"redirect_uri": "https://example.com/cb"}
        )
        assert _status(result) == 400
        body = _body(result)
        assert "code" in body["error"]

    @pytest.mark.asyncio
    async def test_oauth_callback_missing_redirect_uri(self, handler):
        """Missing redirect_uri returns 400."""
        result = await handler.handle_post_gmail_oauth_callback(
            {"code": "abc"}
        )
        assert _status(result) == 400
        body = _body(result)
        assert "redirect_uri" in body["error"]

    @pytest.mark.asyncio
    async def test_oauth_callback_missing_both(self, handler):
        """Missing both code and redirect_uri returns 400."""
        result = await handler.handle_post_gmail_oauth_callback({})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_oauth_callback_failure(self, handler):
        mock_result = {"success": False, "error": "Token exchange failed"}
        with patch(
            "aragora.server.handlers.email.handler.handle_gmail_oauth_callback",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_gmail_oauth_callback(
                {"code": "bad_code", "redirect_uri": "https://example.com/cb"}
            )
        assert _status(result) == 400


# ============================================================================
# GET /api/v1/email/gmail/status
# ============================================================================


class TestGetGmailStatus:
    """Tests for handle_get_gmail_status."""

    @pytest.mark.asyncio
    async def test_gmail_status_success(self, handler):
        mock_result = {"connected": True, "email": "user@gmail.com"}
        with patch(
            "aragora.server.handlers.email.handler.handle_gmail_status",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_get_gmail_status({})
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["data"]["connected"] is True

    @pytest.mark.asyncio
    async def test_gmail_status_not_connected(self, handler):
        """Status endpoint always returns 200, even if not connected."""
        mock_result = {"connected": False}
        with patch(
            "aragora.server.handlers.email.handler.handle_gmail_status",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_get_gmail_status({})
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["connected"] is False


# ============================================================================
# GET /api/v1/email/context/:email_address
# ============================================================================


class TestGetContext:
    """Tests for handle_get_context."""

    @pytest.mark.asyncio
    async def test_get_context_success(self, handler):
        mock_result = {"success": True, "context": {"slack": {}, "calendar": {}}}
        with patch(
            "aragora.server.handlers.email.handler.handle_get_context",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_get_context({}, "user@example.com")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_get_context_failure(self, handler):
        mock_result = {"success": False, "error": "Context unavailable"}
        with patch(
            "aragora.server.handlers.email.handler.handle_get_context",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_get_context({}, "user@example.com")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_context_passes_email_address(self, handler):
        """email_address is forwarded as first positional arg."""
        mock_result = {"success": True, "context": {}}
        with patch(
            "aragora.server.handlers.email.handler.handle_get_context",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_get_context({}, "alice@co.com")
        assert mock_fn.call_args[0][0] == "alice@co.com"


# ============================================================================
# POST /api/v1/email/context/boost
# ============================================================================


class TestPostContextBoost:
    """Tests for handle_post_context_boost."""

    @pytest.mark.asyncio
    async def test_context_boost_success(self, handler):
        mock_result = {"success": True, "boost": 0.15, "signals": ["vip_sender"]}
        with patch(
            "aragora.server.handlers.email.handler.handle_get_email_context_boost",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_context_boost(
                {"email": {"from_address": "ceo@co.com"}}
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["boost"] == 0.15

    @pytest.mark.asyncio
    async def test_context_boost_failure(self, handler):
        mock_result = {"success": False, "error": "Context service down"}
        with patch(
            "aragora.server.handlers.email.handler.handle_get_email_context_boost",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await handler.handle_post_context_boost({"email": {}})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_context_boost_empty_email(self, handler):
        """Empty email data is forwarded (email defaults to {})."""
        mock_result = {"success": True, "boost": 0.0}
        with patch(
            "aragora.server.handlers.email.handler.handle_get_email_context_boost",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_post_context_boost({})
        assert mock_fn.call_args[0][0] == {}


# ============================================================================
# User ID forwarding with auth context
# ============================================================================


class TestUserIdForwarding:
    """Verify user_id from auth context is passed to underlying handlers."""

    @pytest.mark.asyncio
    async def test_prioritize_forwards_user_id(self, handler_with_auth):
        mock_result = {"success": True}
        with patch(
            "aragora.server.handlers.email.handler.handle_prioritize_email",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler_with_auth.handle_post_prioritize({"email": {}})
        assert mock_fn.call_args[0][1] == "user-42"

    @pytest.mark.asyncio
    async def test_rank_inbox_forwards_user_id(self, handler_with_auth):
        mock_result = {"success": True}
        with patch(
            "aragora.server.handlers.email.handler.handle_rank_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler_with_auth.handle_post_rank_inbox({"emails": []})
        assert mock_fn.call_args[0][1] == "user-42"

    @pytest.mark.asyncio
    async def test_feedback_forwards_user_id(self, handler_with_auth):
        mock_result = {"success": True}
        with patch(
            "aragora.server.handlers.email.handler.handle_email_feedback",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler_with_auth.handle_post_feedback(
                {"email_id": "1", "action": "read"}
            )
        assert mock_fn.call_args[0][2] == "user-42"

    @pytest.mark.asyncio
    async def test_get_inbox_forwards_user_id(self, handler_with_auth):
        mock_result = {"success": True, "emails": []}
        with patch(
            "aragora.server.handlers.email.handler.handle_fetch_and_rank_inbox",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler_with_auth.handle_get_inbox({})
        assert mock_fn.call_args.kwargs["user_id"] == "user-42"

    @pytest.mark.asyncio
    async def test_gmail_status_forwards_user_id(self, handler_with_auth):
        mock_result = {"connected": True}
        with patch(
            "aragora.server.handlers.email.handler.handle_gmail_status",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler_with_auth.handle_get_gmail_status({})
        assert mock_fn.call_args[0][0] == "user-42"

    @pytest.mark.asyncio
    async def test_oauth_callback_forwards_user_id(self, handler_with_auth):
        mock_result = {"success": True}
        with patch(
            "aragora.server.handlers.email.handler.handle_gmail_oauth_callback",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler_with_auth.handle_post_gmail_oauth_callback(
                {"code": "c", "redirect_uri": "https://x.com"}
            )
        assert mock_fn.call_args[0][2] == "user-42"


# ============================================================================
# Auth context forwarding
# ============================================================================


class TestAuthContextForwarding:
    """Verify auth_context is passed to underlying handlers."""

    @pytest.mark.asyncio
    async def test_prioritize_passes_auth_context(self, handler_with_auth):
        mock_result = {"success": True}
        with patch(
            "aragora.server.handlers.email.handler.handle_prioritize_email",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler_with_auth.handle_post_prioritize({"email": {}})
        assert mock_fn.call_args.kwargs["auth_context"] is not None
        assert mock_fn.call_args.kwargs["auth_context"].user_id == "user-42"

    @pytest.mark.asyncio
    async def test_categorize_passes_auth_context(self, handler_with_auth):
        mock_result = {"success": True}
        with patch(
            "aragora.server.handlers.email.handler.handle_categorize_email",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler_with_auth.handle_post_categorize({"email": {}})
        assert mock_fn.call_args.kwargs["auth_context"] is not None

    @pytest.mark.asyncio
    async def test_add_vip_passes_auth_context(self, handler_with_auth):
        mock_result = {"success": True}
        with patch(
            "aragora.server.handlers.email.handler.handle_add_vip",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler_with_auth.handle_post_vip({"email": "a@b.com"})
        assert mock_fn.call_args.kwargs["auth_context"] is not None

    @pytest.mark.asyncio
    async def test_no_auth_context_passes_none(self, handler):
        """Handler without auth_context in ctx passes None."""
        mock_result = {"success": True}
        with patch(
            "aragora.server.handlers.email.handler.handle_prioritize_email",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            await handler.handle_post_prioritize({"email": {}})
        assert mock_fn.call_args.kwargs["auth_context"] is None


# ============================================================================
# Initialization and ROUTES
# ============================================================================


class TestInitialization:
    """Tests for EmailHandler initialization and class attributes."""

    def test_routes_list_has_expected_count(self):
        """ROUTES contains all 14 static routes."""
        assert len(EmailHandler.ROUTES) == 14

    def test_route_prefixes_has_context(self):
        """ROUTE_PREFIXES contains the context prefix."""
        assert "/api/v1/email/context/" in EmailHandler.ROUTE_PREFIXES

    def test_handler_initialization(self, ctx):
        """Handler initializes with context dict."""
        h = EmailHandler(ctx)
        assert h.ctx is ctx

    def test_handler_inherits_base_handler(self, handler):
        """EmailHandler is a BaseHandler subclass."""
        from aragora.server.handlers.base import BaseHandler

        assert isinstance(handler, BaseHandler)
