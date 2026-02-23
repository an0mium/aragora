"""Tests for handle_fetch_and_rank_inbox (aragora/server/handlers/email/inbox.py).

Covers:
- Successful inbox fetch with ranked results
- Not authenticated (missing access token) returns needs_auth
- Query building with include_read toggling
- Label forwarding (default and custom)
- Limit parameter enforcement
- Individual message fetch failures (partial results)
- All individual message fetches fail (empty inbox)
- Prioritizer rank_inbox integration
- Response structure validation (email fields, priority, total, fetched_at)
- Email with/without date, attachments, labels, read/starred/important flags
- Exception handling for each caught exception type
- Empty inbox (no message IDs returned)
- Limit truncation when message_ids exceed limit
- result.email_id matching logic (matched and unmatched)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# Mock data classes
# ============================================================================


class MockEmail:
    """Mock email message object matching connector output shape."""

    def __init__(
        self,
        id: str = "msg-1",
        thread_id: str = "thread-1",
        subject: str = "Test Subject",
        from_address: str = "sender@example.com",
        to_addresses: list[str] | None = None,
        date: datetime | None = None,
        snippet: str = "Preview text...",
        labels: list[str] | None = None,
        is_read: bool = False,
        is_starred: bool = False,
        is_important: bool = False,
        attachments: list[Any] | None = None,
    ):
        self.id = id
        self.thread_id = thread_id
        self.subject = subject
        self.from_address = from_address
        self.to_addresses = to_addresses or ["recipient@example.com"]
        self.date = date
        self.snippet = snippet
        self.labels = labels or ["INBOX"]
        self.is_read = is_read
        self.is_starred = is_starred
        self.is_important = is_important
        self.attachments = attachments if attachments is not None else []


class MockPriorityResult:
    """Mock priority result from the prioritizer."""

    def __init__(self, email_id: str = "msg-1", score: float = 0.85, tier: str = "high"):
        self.email_id = email_id
        self.score = score
        self.tier = tier

    def to_dict(self) -> dict[str, Any]:
        return {"email_id": self.email_id, "score": self.score, "tier": self.tier}


# ============================================================================
# Fixtures
# ============================================================================


def _make_connector(authenticated: bool = True, message_ids: list[str] | None = None):
    """Build a mock Gmail connector."""
    connector = AsyncMock()
    connector._access_token = "fake-token" if authenticated else None
    connector.list_messages = AsyncMock(return_value=(message_ids or [], None))
    connector.get_message = AsyncMock(side_effect=lambda mid: MockEmail(id=mid))
    return connector


def _make_prioritizer(results: list[MockPriorityResult] | None = None):
    """Build a mock email prioritizer."""
    prioritizer = AsyncMock()
    prioritizer.rank_inbox = AsyncMock(return_value=results or [])
    return prioritizer


@pytest.fixture
def mock_connector():
    return _make_connector(authenticated=True, message_ids=["msg-1", "msg-2"])


@pytest.fixture
def mock_prioritizer():
    return _make_prioritizer(
        [MockPriorityResult("msg-1", 0.9, "high"), MockPriorityResult("msg-2", 0.5, "low")]
    )


# ============================================================================
# Helpers
# ============================================================================

PATCH_CONNECTOR = "aragora.server.handlers.email.inbox.get_gmail_connector"
PATCH_PRIORITIZER = "aragora.server.handlers.email.inbox.get_prioritizer"


# ============================================================================
# Basic success path
# ============================================================================


class TestFetchAndRankInboxSuccess:
    """Successful inbox fetch and rank scenarios."""

    @pytest.mark.asyncio
    async def test_basic_success(self, mock_connector, mock_prioritizer):
        """Full happy path: fetch, rank, return structured response."""
        with (
            patch(PATCH_CONNECTOR, return_value=mock_connector),
            patch(PATCH_PRIORITIZER, return_value=mock_prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["success"] is True
        assert result["total"] == 2
        assert len(result["inbox"]) == 2
        assert "fetched_at" in result

    @pytest.mark.asyncio
    async def test_inbox_items_have_email_and_priority(self, mock_connector, mock_prioritizer):
        """Each inbox item contains email dict and priority dict."""
        with (
            patch(PATCH_CONNECTOR, return_value=mock_connector),
            patch(PATCH_PRIORITIZER, return_value=mock_prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        item = result["inbox"][0]
        assert "email" in item
        assert "priority" in item

    @pytest.mark.asyncio
    async def test_email_fields_in_response(self, mock_connector, mock_prioritizer):
        """The email dict contains all expected fields."""
        with (
            patch(PATCH_CONNECTOR, return_value=mock_connector),
            patch(PATCH_PRIORITIZER, return_value=mock_prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        email = result["inbox"][0]["email"]
        expected_keys = {
            "id",
            "thread_id",
            "subject",
            "from_address",
            "to_addresses",
            "date",
            "snippet",
            "labels",
            "is_read",
            "is_starred",
            "is_important",
            "has_attachments",
        }
        assert set(email.keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_priority_uses_to_dict(self, mock_connector, mock_prioritizer):
        """Priority section is the result of PriorityResult.to_dict()."""
        with (
            patch(PATCH_CONNECTOR, return_value=mock_connector),
            patch(PATCH_PRIORITIZER, return_value=mock_prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        priority = result["inbox"][0]["priority"]
        assert priority["score"] == 0.9
        assert priority["tier"] == "high"

    @pytest.mark.asyncio
    async def test_fetched_at_is_isoformat(self, mock_connector, mock_prioritizer):
        """fetched_at is a valid ISO datetime string."""
        with (
            patch(PATCH_CONNECTOR, return_value=mock_connector),
            patch(PATCH_PRIORITIZER, return_value=mock_prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        # Should parse without error
        datetime.fromisoformat(result["fetched_at"])

    @pytest.mark.asyncio
    async def test_total_matches_inbox_length(self, mock_connector, mock_prioritizer):
        """total equals the number of inbox items."""
        with (
            patch(PATCH_CONNECTOR, return_value=mock_connector),
            patch(PATCH_PRIORITIZER, return_value=mock_prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["total"] == len(result["inbox"])


# ============================================================================
# Authentication
# ============================================================================


class TestNotAuthenticated:
    """Connector with no access token returns needs_auth."""

    @pytest.mark.asyncio
    async def test_no_access_token_returns_needs_auth(self):
        """Missing access token returns needs_auth=True."""
        connector = _make_connector(authenticated=False)
        with patch(PATCH_CONNECTOR, return_value=connector):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["success"] is False
        assert result["needs_auth"] is True
        assert "OAuth" in result["error"]

    @pytest.mark.asyncio
    async def test_no_access_token_does_not_fetch(self):
        """When not authenticated, list_messages is never called."""
        connector = _make_connector(authenticated=False)
        with patch(PATCH_CONNECTOR, return_value=connector):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            await handle_fetch_and_rank_inbox()
        connector.list_messages.assert_not_awaited()


# ============================================================================
# Query building and parameters
# ============================================================================


class TestQueryBuilding:
    """Tests for query construction and parameter forwarding."""

    @pytest.mark.asyncio
    async def test_unread_query_when_include_read_false(self):
        """Default include_read=False adds is:unread to query."""
        connector = _make_connector(message_ids=[])
        prioritizer = _make_prioritizer()
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            await handle_fetch_and_rank_inbox(include_read=False)
        call_kwargs = connector.list_messages.call_args
        assert "is:unread" in call_kwargs.kwargs.get("query", call_kwargs[1].get("query", ""))

    @pytest.mark.asyncio
    async def test_no_unread_query_when_include_read_true(self):
        """include_read=True does not add is:unread."""
        connector = _make_connector(message_ids=[])
        prioritizer = _make_prioritizer()
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            await handle_fetch_and_rank_inbox(include_read=True)
        call_args = connector.list_messages.call_args
        query = call_args.kwargs.get("query", call_args[1].get("query", "N/A"))
        assert "is:unread" not in query

    @pytest.mark.asyncio
    async def test_empty_query_when_include_read_true(self):
        """include_read=True produces an empty query string."""
        connector = _make_connector(message_ids=[])
        prioritizer = _make_prioritizer()
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            await handle_fetch_and_rank_inbox(include_read=True)
        call_args = connector.list_messages.call_args
        query = call_args.kwargs.get("query", call_args[1].get("query", "N/A"))
        assert query == ""

    @pytest.mark.asyncio
    async def test_default_labels_inbox(self):
        """Default labels is ['INBOX'] when no labels are provided."""
        connector = _make_connector(message_ids=[])
        prioritizer = _make_prioritizer()
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            await handle_fetch_and_rank_inbox(labels=None)
        call_args = connector.list_messages.call_args
        label_ids = call_args.kwargs.get("label_ids", call_args[1].get("label_ids", None))
        assert label_ids == ["INBOX"]

    @pytest.mark.asyncio
    async def test_custom_labels_forwarded(self):
        """Custom labels are forwarded to list_messages."""
        connector = _make_connector(message_ids=[])
        prioritizer = _make_prioritizer()
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            await handle_fetch_and_rank_inbox(labels=["IMPORTANT", "STARRED"])
        call_args = connector.list_messages.call_args
        label_ids = call_args.kwargs.get("label_ids", call_args[1].get("label_ids", None))
        assert label_ids == ["IMPORTANT", "STARRED"]

    @pytest.mark.asyncio
    async def test_limit_forwarded_to_list_messages(self):
        """The limit parameter is forwarded as max_results."""
        connector = _make_connector(message_ids=[])
        prioritizer = _make_prioritizer()
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            await handle_fetch_and_rank_inbox(limit=25)
        call_args = connector.list_messages.call_args
        max_results = call_args.kwargs.get("max_results", call_args[1].get("max_results", None))
        assert max_results == 25

    @pytest.mark.asyncio
    async def test_limit_forwarded_to_prioritizer(self):
        """The limit parameter is forwarded to prioritizer.rank_inbox."""
        connector = _make_connector(message_ids=[])
        prioritizer = _make_prioritizer()
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            await handle_fetch_and_rank_inbox(limit=10)
        call_args = prioritizer.rank_inbox.call_args
        assert call_args.kwargs.get("limit", call_args[1].get("limit", None)) == 10

    @pytest.mark.asyncio
    async def test_user_id_forwarded_to_connector(self):
        """user_id is passed to get_gmail_connector."""
        connector = _make_connector(message_ids=[])
        prioritizer = _make_prioritizer()
        with (
            patch(PATCH_CONNECTOR, return_value=connector) as mock_get_conn,
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            await handle_fetch_and_rank_inbox(user_id="user-42")
        mock_get_conn.assert_called_once_with("user-42")

    @pytest.mark.asyncio
    async def test_user_id_forwarded_to_prioritizer(self):
        """user_id is passed to get_prioritizer."""
        connector = _make_connector(message_ids=[])
        prioritizer = _make_prioritizer()
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer) as mock_get_pri,
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            await handle_fetch_and_rank_inbox(user_id="user-42")
        mock_get_pri.assert_called_once_with("user-42")


# ============================================================================
# Message fetch partial failures
# ============================================================================


class TestPartialMessageFetchFailures:
    """Tests for individual message fetch errors (graceful degradation)."""

    @pytest.mark.asyncio
    async def test_single_message_fetch_failure_skipped(self):
        """One failing message does not break the entire inbox."""
        connector = _make_connector(message_ids=["ok-1", "fail-1", "ok-2"])

        async def get_msg(mid):
            if mid == "fail-1":
                raise ConnectionError("timeout")
            return MockEmail(id=mid)

        connector.get_message = AsyncMock(side_effect=get_msg)
        prioritizer = _make_prioritizer(
            [MockPriorityResult("ok-1"), MockPriorityResult("ok-2")]
        )
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["success"] is True
        assert result["total"] == 2

    @pytest.mark.asyncio
    async def test_timeout_error_skipped(self):
        """TimeoutError in get_message is caught and skipped."""
        connector = _make_connector(message_ids=["fail-1"])
        connector.get_message = AsyncMock(side_effect=TimeoutError("timed out"))
        prioritizer = _make_prioritizer()
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["success"] is True
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_os_error_skipped(self):
        """OSError in get_message is caught and skipped."""
        connector = _make_connector(message_ids=["fail-1"])
        connector.get_message = AsyncMock(side_effect=OSError("disk error"))
        prioritizer = _make_prioritizer()
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["success"] is True
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_value_error_skipped(self):
        """ValueError in get_message is caught and skipped."""
        connector = _make_connector(message_ids=["fail-1"])
        connector.get_message = AsyncMock(side_effect=ValueError("bad data"))
        prioritizer = _make_prioritizer()
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_key_error_skipped(self):
        """KeyError in get_message is caught and skipped."""
        connector = _make_connector(message_ids=["fail-1"])
        connector.get_message = AsyncMock(side_effect=KeyError("missing_field"))
        prioritizer = _make_prioritizer()
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_all_messages_fail(self):
        """All message fetches fail: inbox is empty but response is success."""
        connector = _make_connector(message_ids=["f1", "f2", "f3"])
        connector.get_message = AsyncMock(side_effect=ConnectionError("down"))
        prioritizer = _make_prioritizer()
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["success"] is True
        assert result["total"] == 0
        assert result["inbox"] == []


# ============================================================================
# Email data fields
# ============================================================================


class TestEmailDataFields:
    """Tests for correct email field serialization in the response."""

    @pytest.mark.asyncio
    async def test_email_date_isoformat(self):
        """Email date is serialized as ISO format string."""
        dt = datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        email = MockEmail(id="msg-1", date=dt)
        connector = _make_connector(message_ids=["msg-1"])
        connector.get_message = AsyncMock(return_value=email)
        prioritizer = _make_prioritizer([MockPriorityResult("msg-1")])
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["inbox"][0]["email"]["date"] == dt.isoformat()

    @pytest.mark.asyncio
    async def test_email_date_none(self):
        """Email with no date returns None for date field."""
        email = MockEmail(id="msg-1", date=None)
        connector = _make_connector(message_ids=["msg-1"])
        connector.get_message = AsyncMock(return_value=email)
        prioritizer = _make_prioritizer([MockPriorityResult("msg-1")])
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["inbox"][0]["email"]["date"] is None

    @pytest.mark.asyncio
    async def test_has_attachments_true(self):
        """Email with attachments reports has_attachments=True."""
        email = MockEmail(id="msg-1", attachments=[{"filename": "doc.pdf"}])
        connector = _make_connector(message_ids=["msg-1"])
        connector.get_message = AsyncMock(return_value=email)
        prioritizer = _make_prioritizer([MockPriorityResult("msg-1")])
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["inbox"][0]["email"]["has_attachments"] is True

    @pytest.mark.asyncio
    async def test_has_attachments_false(self):
        """Email without attachments reports has_attachments=False."""
        email = MockEmail(id="msg-1", attachments=[])
        connector = _make_connector(message_ids=["msg-1"])
        connector.get_message = AsyncMock(return_value=email)
        prioritizer = _make_prioritizer([MockPriorityResult("msg-1")])
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["inbox"][0]["email"]["has_attachments"] is False

    @pytest.mark.asyncio
    async def test_read_starred_important_flags(self):
        """Read, starred, and important flags are serialized correctly."""
        email = MockEmail(id="msg-1", is_read=True, is_starred=True, is_important=True)
        connector = _make_connector(message_ids=["msg-1"])
        connector.get_message = AsyncMock(return_value=email)
        prioritizer = _make_prioritizer([MockPriorityResult("msg-1")])
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        e = result["inbox"][0]["email"]
        assert e["is_read"] is True
        assert e["is_starred"] is True
        assert e["is_important"] is True

    @pytest.mark.asyncio
    async def test_labels_in_response(self):
        """Email labels are included in the response."""
        email = MockEmail(id="msg-1", labels=["INBOX", "IMPORTANT", "CATEGORY_PERSONAL"])
        connector = _make_connector(message_ids=["msg-1"])
        connector.get_message = AsyncMock(return_value=email)
        prioritizer = _make_prioritizer([MockPriorityResult("msg-1")])
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["inbox"][0]["email"]["labels"] == ["INBOX", "IMPORTANT", "CATEGORY_PERSONAL"]

    @pytest.mark.asyncio
    async def test_to_addresses_in_response(self):
        """to_addresses list is included in the response."""
        email = MockEmail(id="msg-1", to_addresses=["a@co.com", "b@co.com"])
        connector = _make_connector(message_ids=["msg-1"])
        connector.get_message = AsyncMock(return_value=email)
        prioritizer = _make_prioritizer([MockPriorityResult("msg-1")])
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["inbox"][0]["email"]["to_addresses"] == ["a@co.com", "b@co.com"]


# ============================================================================
# Matching logic: result.email_id -> email
# ============================================================================


class TestEmailIdMatching:
    """Tests for the email_id matching between ranked results and emails."""

    @pytest.mark.asyncio
    async def test_unmatched_result_excluded(self):
        """A priority result with no matching email is excluded from inbox."""
        email = MockEmail(id="msg-1")
        connector = _make_connector(message_ids=["msg-1"])
        connector.get_message = AsyncMock(return_value=email)
        # Prioritizer returns a result for an email that was not fetched
        prioritizer = _make_prioritizer(
            [MockPriorityResult("msg-1"), MockPriorityResult("msg-ghost")]
        )
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        # Only msg-1 should appear
        assert result["total"] == 1
        assert result["inbox"][0]["email"]["id"] == "msg-1"

    @pytest.mark.asyncio
    async def test_all_results_unmatched(self):
        """If no results match any email, inbox is empty."""
        connector = _make_connector(message_ids=[])
        prioritizer = _make_prioritizer([MockPriorityResult("ghost-1")])
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["success"] is True
        assert result["total"] == 0


# ============================================================================
# Empty inbox
# ============================================================================


class TestEmptyInbox:
    """Tests for empty inbox scenarios."""

    @pytest.mark.asyncio
    async def test_no_messages_returned(self):
        """Empty message list from connector produces empty inbox."""
        connector = _make_connector(message_ids=[])
        prioritizer = _make_prioritizer()
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["success"] is True
        assert result["inbox"] == []
        assert result["total"] == 0


# ============================================================================
# Limit truncation
# ============================================================================


class TestLimitTruncation:
    """Tests for message_ids[:limit] truncation."""

    @pytest.mark.asyncio
    async def test_message_ids_truncated_to_limit(self):
        """Only the first 'limit' message_ids are fetched."""
        connector = _make_connector(message_ids=["m1", "m2", "m3", "m4", "m5"])
        prioritizer = _make_prioritizer(
            [MockPriorityResult("m1"), MockPriorityResult("m2")]
        )
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            await handle_fetch_and_rank_inbox(limit=2)
        # get_message should be called only 2 times (limit=2)
        assert connector.get_message.await_count == 2

    @pytest.mark.asyncio
    async def test_default_limit_is_50(self):
        """Default limit parameter is 50."""
        ids = [f"m{i}" for i in range(60)]
        connector = _make_connector(message_ids=ids)
        prioritizer = _make_prioritizer()
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            await handle_fetch_and_rank_inbox()
        # Default limit=50, so only 50 messages fetched
        assert connector.get_message.await_count == 50


# ============================================================================
# Exception handling (outer try/except)
# ============================================================================


class TestOuterExceptionHandling:
    """Tests for the outer try/except that catches broad exceptions."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc_class",
        [
            ConnectionError,
            TimeoutError,
            OSError,
            ValueError,
            KeyError,
            AttributeError,
            RuntimeError,
        ],
    )
    async def test_caught_exception_returns_failure(self, exc_class):
        """Each exception type in the outer except returns success=False."""
        connector = _make_connector(message_ids=["m1"])
        connector.list_messages = AsyncMock(side_effect=exc_class("test error"))
        with patch(PATCH_CONNECTOR, return_value=connector):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["success"] is False
        assert result["error"] == "Failed to fetch inbox"

    @pytest.mark.asyncio
    async def test_prioritizer_exception_caught(self):
        """Exception in prioritizer.rank_inbox is caught by outer handler."""
        connector = _make_connector(message_ids=[])
        prioritizer = _make_prioritizer()
        prioritizer.rank_inbox = AsyncMock(side_effect=RuntimeError("rank failed"))
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["success"] is False
        assert result["error"] == "Failed to fetch inbox"

    @pytest.mark.asyncio
    async def test_connector_creation_exception_caught(self):
        """Exception in get_gmail_connector is caught."""
        with patch(PATCH_CONNECTOR, side_effect=RuntimeError("connector init fail")):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_attribute_error_on_connector(self):
        """AttributeError (e.g., missing _access_token) is caught."""
        connector = MagicMock()
        # Remove _access_token so accessing it raises AttributeError
        del connector._access_token
        type(connector)._access_token = property(
            lambda self: (_ for _ in ()).throw(AttributeError("no token attr"))
        )
        with patch(PATCH_CONNECTOR, return_value=connector):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            result = await handle_fetch_and_rank_inbox()
        assert result["success"] is False


# ============================================================================
# Default parameters
# ============================================================================


class TestDefaultParameters:
    """Tests for default parameter values."""

    @pytest.mark.asyncio
    async def test_default_user_id(self):
        """Default user_id is 'default'."""
        connector = _make_connector(message_ids=[])
        prioritizer = _make_prioritizer()
        with (
            patch(PATCH_CONNECTOR, return_value=connector) as mock_get,
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            await handle_fetch_and_rank_inbox()
        mock_get.assert_called_once_with("default")

    @pytest.mark.asyncio
    async def test_default_workspace_id(self):
        """Default workspace_id is 'default' (parameter exists but is not used internally)."""
        connector = _make_connector(message_ids=[])
        prioritizer = _make_prioritizer()
        with (
            patch(PATCH_CONNECTOR, return_value=connector),
            patch(PATCH_PRIORITIZER, return_value=prioritizer),
        ):
            from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

            # Should not raise
            await handle_fetch_and_rank_inbox(workspace_id="tenant-7")


# ============================================================================
# Module-level constants
# ============================================================================


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_perm_email_read_constant(self):
        from aragora.server.handlers.email.inbox import PERM_EMAIL_READ

        assert PERM_EMAIL_READ == "email:read"

    def test_perm_email_update_constant(self):
        from aragora.server.handlers.email.inbox import PERM_EMAIL_UPDATE

        assert PERM_EMAIL_UPDATE == "email:update"


# ============================================================================
# Decorator presence
# ============================================================================


class TestDecoratorPresence:
    """Tests confirming decorators are applied."""

    def test_function_has_rate_limit_marker(self):
        """The handler should be marked as rate-limited by the decorator."""
        from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

        assert getattr(handle_fetch_and_rank_inbox, "_rate_limited", False) is True

    def test_function_is_async(self):
        """The handler should be an async function."""
        import asyncio

        from aragora.server.handlers.email.inbox import handle_fetch_and_rank_inbox

        assert asyncio.iscoroutinefunction(handle_fetch_and_rank_inbox)
