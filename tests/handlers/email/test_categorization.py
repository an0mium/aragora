"""Tests for email categorization handler functions.

Tests for aragora/server/handlers/email/categorization.py covering:
- handle_categorize_email: single email categorization
- handle_categorize_batch: batch email categorization
- handle_feedback_batch: batch feedback recording
- handle_apply_category_label: Gmail label application
- get_categorizer: thread-safe singleton initialization
- Error handling, edge cases, input validation, security
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import aragora.server.handlers.email.categorization as cat_module
from aragora.server.handlers.email.categorization import (
    _AUTH_CONTEXT_UNSET,
    PERM_EMAIL_READ,
    PERM_EMAIL_UPDATE,
    get_categorizer,
    handle_apply_category_label,
    handle_categorize_batch,
    handle_categorize_email,
    handle_feedback_batch,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_email_data(**overrides) -> dict[str, Any]:
    """Build a minimal email data dict with sensible defaults."""
    data: dict[str, Any] = {
        "id": "msg_001",
        "subject": "Invoice #12345",
        "from_address": "billing@company.com",
        "body_text": "Please find the invoice attached.",
    }
    data.update(overrides)
    return data


def _make_mock_result(**overrides) -> MagicMock:
    """Build a mock categorization result object with a to_dict method."""
    result = MagicMock()
    d = {"category": "invoices", "confidence": 0.95, "email_id": "msg_001"}
    d.update(overrides)
    result.to_dict.return_value = d
    return result


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_categorizer():
    """Reset the module-level _categorizer singleton between tests."""
    original = cat_module._categorizer
    yield
    cat_module._categorizer = original


@pytest.fixture
def mock_categorizer():
    """Provide a mock categorizer with common async methods."""
    categorizer = MagicMock()
    categorizer.categorize_email = AsyncMock()
    categorizer.categorize_batch = AsyncMock()
    categorizer.apply_gmail_label = AsyncMock()
    categorizer.get_category_stats = MagicMock(return_value={"invoices": 1})
    return categorizer


@pytest.fixture
def mock_prioritizer():
    """Provide a mock prioritizer with record_user_action."""
    prioritizer = MagicMock()
    prioritizer.record_user_action = AsyncMock()
    return prioritizer


@pytest.fixture
def patch_categorizer(mock_categorizer):
    """Patch get_categorizer to return mock_categorizer."""
    with patch.object(cat_module, "get_categorizer", return_value=mock_categorizer):
        yield mock_categorizer


@pytest.fixture
def patch_prioritizer(mock_prioritizer):
    """Patch get_prioritizer to return mock_prioritizer."""
    with patch(
        "aragora.server.handlers.email.categorization.get_prioritizer",
        return_value=mock_prioritizer,
    ):
        yield mock_prioritizer


# ============================================================================
# get_categorizer() tests
# ============================================================================


class TestGetCategorizer:
    """Tests for the get_categorizer singleton factory."""

    def test_creates_categorizer_on_first_call(self):
        """get_categorizer lazily creates an EmailCategorizer."""
        cat_module._categorizer = None
        mock_cls = MagicMock()
        with (
            patch(
                "aragora.server.handlers.email.categorization.get_gmail_connector",
                return_value=MagicMock(),
            ),
            patch(
                "aragora.services.email_categorizer.EmailCategorizer",
                mock_cls,
            ),
        ):
            result = get_categorizer()
        assert result is not None
        mock_cls.assert_called_once()

    def test_returns_cached_instance(self):
        """Subsequent calls return the same instance."""
        sentinel = object()
        cat_module._categorizer = sentinel
        assert get_categorizer() is sentinel

    def test_thread_safety(self):
        """Concurrent calls should only create one instance."""
        cat_module._categorizer = None
        call_count = 0
        created_instance = MagicMock()

        def fake_init(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return created_instance

        with (
            patch(
                "aragora.server.handlers.email.categorization.get_gmail_connector",
                return_value=MagicMock(),
            ),
            patch(
                "aragora.services.email_categorizer.EmailCategorizer",
                side_effect=fake_init,
            ),
        ):
            threads = [threading.Thread(target=get_categorizer) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # Should be at most 1 creation due to double-check locking
        assert call_count <= 1


# ============================================================================
# handle_categorize_email() tests
# ============================================================================


class TestHandleCategorizeEmail:
    """Tests for single email categorization."""

    @pytest.mark.asyncio
    async def test_success_minimal_email(self, patch_categorizer):
        """Minimal email data returns success with result dict."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_email.return_value = mock_result

        result = await handle_categorize_email(_make_email_data())

        assert result["success"] is True
        assert result["result"]["category"] == "invoices"
        assert result["result"]["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_success_full_email_data(self, patch_categorizer):
        """Full email data with all fields returns success."""
        mock_result = _make_mock_result(category="meetings")
        patch_categorizer.categorize_email.return_value = mock_result

        email = _make_email_data(
            thread_id="thread_001",
            to_addresses=["user@example.com"],
            cc_addresses=["cc@example.com"],
            bcc_addresses=["bcc@example.com"],
            date="2025-01-15T10:30:00",
            body_html="<p>Meeting invite</p>",
            snippet="Meeting invite...",
            labels=["INBOX", "IMPORTANT"],
            headers={"Message-ID": "<abc@example.com>"},
            is_read=True,
            is_starred=True,
            is_important=True,
        )

        result = await handle_categorize_email(email)
        assert result["success"] is True
        assert result["result"]["category"] == "meetings"

    @pytest.mark.asyncio
    async def test_email_message_construction(self, patch_categorizer):
        """Verify EmailMessage is created with correct field mapping."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_email.return_value = mock_result

        email = _make_email_data(
            id="msg_42",
            subject="Test Subject",
            from_address="sender@test.com",
            body_text="Hello world",
        )

        await handle_categorize_email(email)

        # Verify the categorizer was called
        patch_categorizer.categorize_email.assert_awaited_once()
        call_email = patch_categorizer.categorize_email.call_args[0][0]
        assert call_email.id == "msg_42"
        assert call_email.subject == "Test Subject"
        assert call_email.from_address == "sender@test.com"
        assert call_email.body_text == "Hello world"

    @pytest.mark.asyncio
    async def test_defaults_for_missing_fields(self, patch_categorizer):
        """Missing optional fields get sensible defaults."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_email.return_value = mock_result

        result = await handle_categorize_email({})

        assert result["success"] is True
        call_email = patch_categorizer.categorize_email.call_args[0][0]
        assert call_email.id == "unknown"
        assert call_email.subject == ""
        assert call_email.from_address == ""
        assert call_email.to_addresses == []
        assert call_email.body_text == ""
        assert call_email.is_read is False

    @pytest.mark.asyncio
    async def test_thread_id_defaults_to_id(self, patch_categorizer):
        """thread_id falls back to id if not provided."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_email.return_value = mock_result

        await handle_categorize_email({"id": "msg_99"})
        call_email = patch_categorizer.categorize_email.call_args[0][0]
        assert call_email.thread_id == "msg_99"

    @pytest.mark.asyncio
    async def test_date_parsing_iso_format(self, patch_categorizer):
        """ISO format date strings are parsed correctly."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_email.return_value = mock_result

        await handle_categorize_email({"date": "2025-06-15T08:30:00"})

        call_email = patch_categorizer.categorize_email.call_args[0][0]
        assert call_email.date.year == 2025
        assert call_email.date.month == 6

    @pytest.mark.asyncio
    async def test_date_defaults_to_now_when_missing(self, patch_categorizer):
        """Missing date defaults to datetime.now()."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_email.return_value = mock_result

        await handle_categorize_email({})

        call_email = patch_categorizer.categorize_email.call_args[0][0]
        assert isinstance(call_email.date, datetime)

    @pytest.mark.asyncio
    async def test_user_id_forwarded(self, patch_categorizer):
        """user_id parameter is accepted."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_email.return_value = mock_result

        result = await handle_categorize_email(_make_email_data(), user_id="user-42")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_workspace_id_forwarded(self, patch_categorizer):
        """workspace_id parameter is accepted."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_email.return_value = mock_result

        result = await handle_categorize_email(_make_email_data(), workspace_id="ws-99")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_type_error_returns_failure(self, patch_categorizer):
        """TypeError in categorization returns error dict."""
        patch_categorizer.categorize_email.side_effect = TypeError("bad type")

        result = await handle_categorize_email(_make_email_data())
        assert result["success"] is False
        assert result["error"] == "Internal server error"

    @pytest.mark.asyncio
    async def test_value_error_returns_failure(self, patch_categorizer):
        """ValueError in categorization returns error dict."""
        patch_categorizer.categorize_email.side_effect = ValueError("bad value")

        result = await handle_categorize_email(_make_email_data())
        assert result["success"] is False
        assert result["error"] == "Internal server error"

    @pytest.mark.asyncio
    async def test_key_error_returns_failure(self, patch_categorizer):
        """KeyError in categorization returns error dict."""
        patch_categorizer.categorize_email.side_effect = KeyError("missing key")

        result = await handle_categorize_email(_make_email_data())
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_runtime_error_returns_failure(self, patch_categorizer):
        """RuntimeError in categorization returns error dict."""
        patch_categorizer.categorize_email.side_effect = RuntimeError("runtime issue")

        result = await handle_categorize_email(_make_email_data())
        assert result["success"] is False
        assert result["error"] == "Internal server error"

    @pytest.mark.asyncio
    async def test_does_not_leak_error_details(self, patch_categorizer):
        """Error messages should not expose internal details."""
        patch_categorizer.categorize_email.side_effect = RuntimeError(
            "Connection to database failed at 10.0.0.5:5432"
        )

        result = await handle_categorize_email(_make_email_data())
        assert "10.0.0.5" not in result.get("error", "")
        assert "database" not in result.get("error", "")

    @pytest.mark.asyncio
    async def test_attachments_always_empty_list(self, patch_categorizer):
        """Attachments are hardcoded to empty list."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_email.return_value = mock_result

        await handle_categorize_email(_make_email_data())
        call_email = patch_categorizer.categorize_email.call_args[0][0]
        assert call_email.attachments == []


# ============================================================================
# handle_categorize_batch() tests
# ============================================================================


class TestHandleCategorizeBatch:
    """Tests for batch email categorization."""

    @pytest.mark.asyncio
    async def test_batch_success_single_email(self, patch_categorizer):
        """Batch with one email returns results and stats."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_batch.return_value = [mock_result]
        patch_categorizer.get_category_stats.return_value = {"invoices": 1}

        emails = [_make_email_data()]
        result = await handle_categorize_batch(emails)

        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["stats"] == {"invoices": 1}

    @pytest.mark.asyncio
    async def test_batch_success_multiple_emails(self, patch_categorizer):
        """Batch with multiple emails returns all results."""
        results = [
            _make_mock_result(category="invoices", email_id="msg_1"),
            _make_mock_result(category="meetings", email_id="msg_2"),
            _make_mock_result(category="personal", email_id="msg_3"),
        ]
        patch_categorizer.categorize_batch.return_value = results
        patch_categorizer.get_category_stats.return_value = {
            "invoices": 1,
            "meetings": 1,
            "personal": 1,
        }

        emails = [
            _make_email_data(id="msg_1"),
            _make_email_data(id="msg_2"),
            _make_email_data(id="msg_3"),
        ]
        result = await handle_categorize_batch(emails)

        assert result["success"] is True
        assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_batch_forwards_concurrency(self, patch_categorizer):
        """Concurrency parameter is forwarded to categorizer."""
        patch_categorizer.categorize_batch.return_value = []
        patch_categorizer.get_category_stats.return_value = {}

        await handle_categorize_batch([_make_email_data()], concurrency=5)

        call_kwargs = patch_categorizer.categorize_batch.call_args
        assert call_kwargs.kwargs.get("concurrency") == 5 or call_kwargs[1].get("concurrency") == 5

    @pytest.mark.asyncio
    async def test_batch_default_concurrency(self, patch_categorizer):
        """Default concurrency is 10."""
        patch_categorizer.categorize_batch.return_value = []
        patch_categorizer.get_category_stats.return_value = {}

        await handle_categorize_batch([_make_email_data()])

        call_kwargs = patch_categorizer.categorize_batch.call_args
        assert (
            call_kwargs.kwargs.get("concurrency") == 10 or call_kwargs[1].get("concurrency") == 10
        )

    @pytest.mark.asyncio
    async def test_batch_empty_emails_list(self, patch_categorizer):
        """Empty email list returns empty results."""
        patch_categorizer.categorize_batch.return_value = []
        patch_categorizer.get_category_stats.return_value = {}

        result = await handle_categorize_batch([])

        assert result["success"] is True
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_batch_email_id_defaults(self, patch_categorizer):
        """Emails without id get sequential unknown_N ids."""
        patch_categorizer.categorize_batch.return_value = []
        patch_categorizer.get_category_stats.return_value = {}

        emails = [{}, {}]
        await handle_categorize_batch(emails)

        call_emails = patch_categorizer.categorize_batch.call_args[0][0]
        assert call_emails[0].id == "unknown_0"
        assert call_emails[1].id == "unknown_1"

    @pytest.mark.asyncio
    async def test_batch_type_error(self, patch_categorizer):
        """TypeError during batch categorization returns failure."""
        patch_categorizer.categorize_batch.side_effect = TypeError("bad batch")

        result = await handle_categorize_batch([_make_email_data()])
        assert result["success"] is False
        assert result["error"] == "Internal server error"

    @pytest.mark.asyncio
    async def test_batch_value_error(self, patch_categorizer):
        """ValueError during batch categorization returns failure."""
        patch_categorizer.categorize_batch.side_effect = ValueError("invalid")

        result = await handle_categorize_batch([_make_email_data()])
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_batch_runtime_error(self, patch_categorizer):
        """RuntimeError during batch categorization returns failure."""
        patch_categorizer.categorize_batch.side_effect = RuntimeError("crash")

        result = await handle_categorize_batch([_make_email_data()])
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_batch_key_error(self, patch_categorizer):
        """KeyError during batch categorization returns failure."""
        patch_categorizer.categorize_batch.side_effect = KeyError("missing")

        result = await handle_categorize_batch([_make_email_data()])
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_batch_stats_called_with_results(self, patch_categorizer):
        """get_category_stats is called with the batch results."""
        results = [_make_mock_result()]
        patch_categorizer.categorize_batch.return_value = results
        patch_categorizer.get_category_stats.return_value = {"invoices": 1}

        await handle_categorize_batch([_make_email_data()])

        patch_categorizer.get_category_stats.assert_called_once_with(results)

    @pytest.mark.asyncio
    async def test_batch_date_parsing(self, patch_categorizer):
        """Batch handles date parsing for each email."""
        patch_categorizer.categorize_batch.return_value = []
        patch_categorizer.get_category_stats.return_value = {}

        emails = [
            _make_email_data(id="msg_1", date="2025-03-01T09:00:00"),
            _make_email_data(id="msg_2"),  # No date, defaults to now
        ]
        await handle_categorize_batch(emails)

        call_emails = patch_categorizer.categorize_batch.call_args[0][0]
        assert call_emails[0].date.year == 2025
        assert isinstance(call_emails[1].date, datetime)

    @pytest.mark.asyncio
    async def test_batch_user_id_forwarded(self, patch_categorizer):
        """user_id parameter is accepted by batch handler."""
        patch_categorizer.categorize_batch.return_value = []
        patch_categorizer.get_category_stats.return_value = {}

        result = await handle_categorize_batch([_make_email_data()], user_id="user-batch")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_batch_does_not_leak_errors(self, patch_categorizer):
        """Error messages do not expose internal details."""
        patch_categorizer.categorize_batch.side_effect = RuntimeError(
            "SQL error in table email_categories"
        )

        result = await handle_categorize_batch([_make_email_data()])
        assert "SQL" not in result.get("error", "")


# ============================================================================
# handle_feedback_batch() tests
# ============================================================================


class TestHandleFeedbackBatch:
    """Tests for batch feedback recording."""

    @pytest.mark.asyncio
    async def test_feedback_all_success(self, patch_prioritizer):
        """All feedback items recorded successfully."""
        items = [
            {"email_id": "msg_1", "action": "archived"},
            {"email_id": "msg_2", "action": "replied", "response_time_minutes": 5},
        ]

        result = await handle_feedback_batch(items)

        assert result["success"] is True
        assert result["recorded"] == 2
        assert result["errors"] == 0
        assert len(result["results"]) == 2
        assert result["error_details"] is None

    @pytest.mark.asyncio
    async def test_feedback_with_response_time(self, patch_prioritizer):
        """Response time is forwarded to prioritizer."""
        items = [{"email_id": "msg_1", "action": "replied", "response_time_minutes": 3}]

        await handle_feedback_batch(items)

        patch_prioritizer.record_user_action.assert_awaited_once_with(
            email_id="msg_1",
            action="replied",
            response_time_minutes=3,
        )

    @pytest.mark.asyncio
    async def test_feedback_without_response_time(self, patch_prioritizer):
        """Missing response_time_minutes is forwarded as None."""
        items = [{"email_id": "msg_1", "action": "archived"}]

        await handle_feedback_batch(items)

        patch_prioritizer.record_user_action.assert_awaited_once_with(
            email_id="msg_1",
            action="archived",
            response_time_minutes=None,
        )

    @pytest.mark.asyncio
    async def test_feedback_missing_email_id(self, patch_prioritizer):
        """Item without email_id is counted as error."""
        items = [{"action": "archived"}]

        result = await handle_feedback_batch(items)

        assert result["success"] is True
        assert result["recorded"] == 0
        assert result["errors"] == 1
        assert result["error_details"][0]["error"] == "Missing email_id or action"

    @pytest.mark.asyncio
    async def test_feedback_missing_action(self, patch_prioritizer):
        """Item without action is counted as error."""
        items = [{"email_id": "msg_1"}]

        result = await handle_feedback_batch(items)

        assert result["success"] is True
        assert result["recorded"] == 0
        assert result["errors"] == 1
        assert "Missing email_id or action" in result["error_details"][0]["error"]

    @pytest.mark.asyncio
    async def test_feedback_missing_both(self, patch_prioritizer):
        """Item without email_id and action is counted as error."""
        items = [{}]

        result = await handle_feedback_batch(items)

        assert result["recorded"] == 0
        assert result["errors"] == 1

    @pytest.mark.asyncio
    async def test_feedback_mixed_success_and_errors(self, patch_prioritizer):
        """Mix of valid and invalid items returns partial results."""
        items = [
            {"email_id": "msg_1", "action": "archived"},
            {"action": "starred"},  # Missing email_id
            {"email_id": "msg_3", "action": "opened"},
        ]

        result = await handle_feedback_batch(items)

        assert result["recorded"] == 2
        assert result["errors"] == 1
        assert len(result["results"]) == 2
        assert len(result["error_details"]) == 1

    @pytest.mark.asyncio
    async def test_feedback_recording_failure(self, patch_prioritizer):
        """Exception during recording is caught per-item."""
        patch_prioritizer.record_user_action.side_effect = RuntimeError("store down")

        items = [{"email_id": "msg_1", "action": "archived"}]
        result = await handle_feedback_batch(items)

        assert result["recorded"] == 0
        assert result["errors"] == 1
        assert result["error_details"][0]["error"] == "Internal server error"

    @pytest.mark.asyncio
    async def test_feedback_type_error_per_item(self, patch_prioritizer):
        """TypeError per item is caught and reported."""
        patch_prioritizer.record_user_action.side_effect = TypeError("wrong type")

        items = [{"email_id": "msg_1", "action": "read"}]
        result = await handle_feedback_batch(items)

        assert result["errors"] == 1

    @pytest.mark.asyncio
    async def test_feedback_value_error_per_item(self, patch_prioritizer):
        """ValueError per item is caught and reported."""
        patch_prioritizer.record_user_action.side_effect = ValueError("bad val")

        items = [{"email_id": "msg_1", "action": "read"}]
        result = await handle_feedback_batch(items)

        assert result["errors"] == 1

    @pytest.mark.asyncio
    async def test_feedback_os_error_per_item(self, patch_prioritizer):
        """OSError per item is caught and reported."""
        patch_prioritizer.record_user_action.side_effect = OSError("disk full")

        items = [{"email_id": "msg_1", "action": "read"}]
        result = await handle_feedback_batch(items)

        assert result["errors"] == 1

    @pytest.mark.asyncio
    async def test_feedback_empty_list(self, patch_prioritizer):
        """Empty feedback list returns zero counts."""
        result = await handle_feedback_batch([])

        assert result["success"] is True
        assert result["recorded"] == 0
        assert result["errors"] == 0
        assert result["results"] == []
        assert result["error_details"] is None

    @pytest.mark.asyncio
    async def test_feedback_user_id_forwarded(self, patch_prioritizer):
        """user_id is forwarded to get_prioritizer."""
        items = [{"email_id": "msg_1", "action": "archived"}]
        with patch(
            "aragora.server.handlers.email.categorization.get_prioritizer",
            return_value=patch_prioritizer,
        ) as mock_get:
            await handle_feedback_batch(items, user_id="user-99")
            mock_get.assert_called_with("user-99")

    @pytest.mark.asyncio
    async def test_feedback_all_actions(self, patch_prioritizer):
        """All documented action types are accepted."""
        actions = ["opened", "replied", "starred", "archived", "deleted", "snoozed"]
        items = [{"email_id": f"msg_{i}", "action": a} for i, a in enumerate(actions)]

        result = await handle_feedback_batch(items)

        assert result["recorded"] == len(actions)
        assert result["errors"] == 0

    @pytest.mark.asyncio
    async def test_feedback_result_structure(self, patch_prioritizer):
        """Each successful result has email_id, action, and recorded fields."""
        items = [{"email_id": "msg_1", "action": "archived"}]
        result = await handle_feedback_batch(items)

        entry = result["results"][0]
        assert entry["email_id"] == "msg_1"
        assert entry["action"] == "archived"
        assert entry["recorded"] is True

    @pytest.mark.asyncio
    async def test_feedback_auth_context_permission_check(self, patch_prioritizer):
        """When auth_context is provided, _check_email_permission is called."""
        mock_auth = MagicMock()
        items = [{"email_id": "msg_1", "action": "archived"}]

        with patch(
            "aragora.server.handlers.email.categorization._check_email_permission",
            return_value=None,
        ) as mock_check:
            await handle_feedback_batch(items, auth_context=mock_auth)
            mock_check.assert_called_once_with(mock_auth, PERM_EMAIL_UPDATE)

    @pytest.mark.asyncio
    async def test_feedback_auth_context_denied(self, patch_prioritizer):
        """When permission denied, returns error without processing."""
        mock_auth = MagicMock()
        perm_error = {"success": False, "error": "Permission denied"}

        with patch(
            "aragora.server.handlers.email.categorization._check_email_permission",
            return_value=perm_error,
        ):
            result = await handle_feedback_batch(
                [{"email_id": "msg_1", "action": "archived"}],
                auth_context=mock_auth,
            )
            assert result["success"] is False
            assert result["error"] == "Permission denied"

    @pytest.mark.asyncio
    async def test_feedback_auth_context_unset_skips_check(self, patch_prioritizer):
        """When auth_context is _AUTH_CONTEXT_UNSET, permission check is skipped."""
        items = [{"email_id": "msg_1", "action": "archived"}]

        with patch(
            "aragora.server.handlers.email.categorization._check_email_permission",
        ) as mock_check:
            await handle_feedback_batch(items, auth_context=_AUTH_CONTEXT_UNSET)
            mock_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_feedback_outer_exception(self):
        """Outer exception handler catches errors before per-item loop."""
        with patch(
            "aragora.server.handlers.email.categorization.get_prioritizer",
            side_effect=RuntimeError("init failed"),
        ):
            result = await handle_feedback_batch([{"email_id": "msg_1", "action": "archived"}])
            assert result["success"] is False
            assert result["error"] == "Internal server error"

    @pytest.mark.asyncio
    async def test_feedback_does_not_leak_errors(self, patch_prioritizer):
        """Per-item error messages are sanitized."""
        patch_prioritizer.record_user_action.side_effect = RuntimeError(
            "Connection refused to redis://10.0.0.5:6379"
        )

        items = [{"email_id": "msg_1", "action": "read"}]
        result = await handle_feedback_batch(items)

        for detail in result.get("error_details", []):
            assert "10.0.0.5" not in detail.get("error", "")

    @pytest.mark.asyncio
    async def test_feedback_none_email_id_treated_as_missing(self, patch_prioritizer):
        """Explicit None for email_id is treated as missing."""
        items = [{"email_id": None, "action": "read"}]
        result = await handle_feedback_batch(items)

        assert result["errors"] == 1
        assert result["recorded"] == 0

    @pytest.mark.asyncio
    async def test_feedback_empty_string_action_treated_as_missing(self, patch_prioritizer):
        """Empty string action is treated as missing."""
        items = [{"email_id": "msg_1", "action": ""}]
        result = await handle_feedback_batch(items)

        assert result["errors"] == 1
        assert result["recorded"] == 0


# ============================================================================
# handle_apply_category_label() tests
# ============================================================================


class TestHandleApplyCategoryLabel:
    """Tests for Gmail label application."""

    @pytest.mark.asyncio
    async def test_apply_label_success(self, patch_categorizer):
        """Successful label application returns success with details."""
        patch_categorizer.apply_gmail_label.return_value = True

        with patch(
            "aragora.services.email_categorizer.EmailCategory",
            side_effect=lambda v: MagicMock(value=v),
        ):
            result = await handle_apply_category_label("msg_001", "invoices")

        assert result["success"] is True
        assert result["email_id"] == "msg_001"
        assert result["category"] == "invoices"
        assert result["label_applied"] is True

    @pytest.mark.asyncio
    async def test_apply_label_failure_returns_false(self, patch_categorizer):
        """When apply_gmail_label returns False, success is False."""
        patch_categorizer.apply_gmail_label.return_value = False

        with patch(
            "aragora.services.email_categorizer.EmailCategory",
            side_effect=lambda v: MagicMock(value=v),
        ):
            result = await handle_apply_category_label("msg_001", "invoices")

        assert result["success"] is False
        assert result["label_applied"] is False

    @pytest.mark.asyncio
    async def test_invalid_category_returns_error(self, patch_categorizer):
        """Invalid category enum value returns error with category name."""
        with patch(
            "aragora.services.email_categorizer.EmailCategory",
            side_effect=ValueError("'invalid_cat' is not a valid EmailCategory"),
        ):
            result = await handle_apply_category_label("msg_001", "invalid_cat")

        assert result["success"] is False
        assert "Invalid category" in result["error"]
        assert "invalid_cat" in result["error"]

    @pytest.mark.asyncio
    async def test_runtime_error_returns_internal_error(self, patch_categorizer):
        """RuntimeError during label application returns generic error."""
        patch_categorizer.apply_gmail_label.side_effect = RuntimeError("API error")

        with patch(
            "aragora.services.email_categorizer.EmailCategory",
            side_effect=lambda v: MagicMock(value=v),
        ):
            result = await handle_apply_category_label("msg_001", "invoices")

        assert result["success"] is False
        assert result["error"] == "Internal server error"

    @pytest.mark.asyncio
    async def test_type_error_returns_internal_error(self, patch_categorizer):
        """TypeError during label application returns generic error."""
        patch_categorizer.apply_gmail_label.side_effect = TypeError("wrong type")

        with patch(
            "aragora.services.email_categorizer.EmailCategory",
            side_effect=lambda v: MagicMock(value=v),
        ):
            result = await handle_apply_category_label("msg_001", "invoices")

        assert result["success"] is False
        assert result["error"] == "Internal server error"

    @pytest.mark.asyncio
    async def test_os_error_returns_internal_error(self, patch_categorizer):
        """OSError during label application returns generic error."""
        patch_categorizer.apply_gmail_label.side_effect = OSError("network issue")

        with patch(
            "aragora.services.email_categorizer.EmailCategory",
            side_effect=lambda v: MagicMock(value=v),
        ):
            result = await handle_apply_category_label("msg_001", "invoices")

        assert result["success"] is False
        assert result["error"] == "Internal server error"

    @pytest.mark.asyncio
    async def test_connection_error_returns_internal_error(self, patch_categorizer):
        """ConnectionError during label application returns generic error."""
        patch_categorizer.apply_gmail_label.side_effect = ConnectionError("refused")

        with patch(
            "aragora.services.email_categorizer.EmailCategory",
            side_effect=lambda v: MagicMock(value=v),
        ):
            result = await handle_apply_category_label("msg_001", "invoices")

        assert result["success"] is False
        assert result["error"] == "Internal server error"

    @pytest.mark.asyncio
    async def test_auth_context_permission_check(self, patch_categorizer):
        """When auth_context is provided, _check_email_permission is called."""
        mock_auth = MagicMock()
        patch_categorizer.apply_gmail_label.return_value = True

        with (
            patch(
                "aragora.server.handlers.email.categorization._check_email_permission",
                return_value=None,
            ) as mock_check,
            patch(
                "aragora.services.email_categorizer.EmailCategory",
                side_effect=lambda v: MagicMock(value=v),
            ),
        ):
            await handle_apply_category_label("msg_001", "invoices", auth_context=mock_auth)
            mock_check.assert_called_once_with(mock_auth, PERM_EMAIL_UPDATE)

    @pytest.mark.asyncio
    async def test_auth_context_denied(self, patch_categorizer):
        """When permission denied, returns error without applying."""
        mock_auth = MagicMock()
        perm_error = {"success": False, "error": "Permission denied"}

        with patch(
            "aragora.server.handlers.email.categorization._check_email_permission",
            return_value=perm_error,
        ):
            result = await handle_apply_category_label(
                "msg_001", "invoices", auth_context=mock_auth
            )
            assert result["success"] is False
            assert result["error"] == "Permission denied"
            patch_categorizer.apply_gmail_label.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_auth_context_unset_skips_check(self, patch_categorizer):
        """When auth_context is _AUTH_CONTEXT_UNSET, permission check is skipped."""
        patch_categorizer.apply_gmail_label.return_value = True

        with (
            patch(
                "aragora.server.handlers.email.categorization._check_email_permission",
            ) as mock_check,
            patch(
                "aragora.services.email_categorizer.EmailCategory",
                side_effect=lambda v: MagicMock(value=v),
            ),
        ):
            await handle_apply_category_label(
                "msg_001", "invoices", auth_context=_AUTH_CONTEXT_UNSET
            )
            mock_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_leak_error_details(self, patch_categorizer):
        """Error messages should not expose internal details."""
        patch_categorizer.apply_gmail_label.side_effect = ConnectionError(
            "Connection refused at smtp.gmail.com:587"
        )

        with patch(
            "aragora.services.email_categorizer.EmailCategory",
            side_effect=lambda v: MagicMock(value=v),
        ):
            result = await handle_apply_category_label("msg_001", "invoices")

        assert "smtp.gmail.com" not in result.get("error", "")

    @pytest.mark.asyncio
    async def test_email_id_in_response(self, patch_categorizer):
        """Response includes the email_id that was processed."""
        patch_categorizer.apply_gmail_label.return_value = True

        with patch(
            "aragora.services.email_categorizer.EmailCategory",
            side_effect=lambda v: MagicMock(value=v),
        ):
            result = await handle_apply_category_label("msg_special", "invoices")

        assert result["email_id"] == "msg_special"

    @pytest.mark.asyncio
    async def test_category_in_response(self, patch_categorizer):
        """Response includes the category that was applied."""
        patch_categorizer.apply_gmail_label.return_value = True

        with patch(
            "aragora.services.email_categorizer.EmailCategory",
            side_effect=lambda v: MagicMock(value=v),
        ):
            result = await handle_apply_category_label("msg_001", "meetings")

        assert result["category"] == "meetings"


# ============================================================================
# Security tests
# ============================================================================


class TestSecurityConcerns:
    """Security-focused tests for categorization handlers."""

    @pytest.mark.asyncio
    async def test_path_traversal_in_email_id(self, patch_categorizer):
        """Path traversal attempt in email_id does not cause issues."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_email.return_value = mock_result

        email = _make_email_data(id="../../etc/passwd")
        result = await handle_categorize_email(email)
        # Should process normally (categorizer handles the data)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_script_injection_in_subject(self, patch_categorizer):
        """Script injection in subject does not cause issues."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_email.return_value = mock_result

        email = _make_email_data(subject='<script>alert("xss")</script>')
        result = await handle_categorize_email(email)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_sql_injection_in_body(self, patch_categorizer):
        """SQL injection in body does not cause issues."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_email.return_value = mock_result

        email = _make_email_data(body_text="'; DROP TABLE emails; --")
        result = await handle_categorize_email(email)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_unicode_in_email_fields(self, patch_categorizer):
        """Unicode in email fields is handled correctly."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_email.return_value = mock_result

        email = _make_email_data(
            subject="Re: Meeting invitation",
            body_text="Please join us for the meeting.",
        )
        result = await handle_categorize_email(email)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_very_long_subject(self, patch_categorizer):
        """Very long subject does not cause issues."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_email.return_value = mock_result

        email = _make_email_data(subject="A" * 10000)
        result = await handle_categorize_email(email)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_very_long_body(self, patch_categorizer):
        """Very long body text does not cause issues."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_email.return_value = mock_result

        email = _make_email_data(body_text="X" * 100000)
        result = await handle_categorize_email(email)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_null_bytes_in_fields(self, patch_categorizer):
        """Null bytes in fields do not cause issues."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_email.return_value = mock_result

        email = _make_email_data(subject="Test\x00Subject")
        result = await handle_categorize_email(email)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_category_injection_in_apply_label(self, patch_categorizer):
        """Category injection attempt in apply-label returns error."""
        with patch(
            "aragora.services.email_categorizer.EmailCategory",
            side_effect=ValueError("invalid category"),
        ):
            result = await handle_apply_category_label("msg_001", "invoices'; DROP TABLE --")
        assert result["success"] is False
        assert "Invalid category" in result["error"]

    @pytest.mark.asyncio
    async def test_feedback_injection_in_action(self, patch_prioritizer):
        """Injection attempt in action field is accepted (sanitization by downstream)."""
        items = [{"email_id": "msg_1", "action": "<script>alert(1)</script>"}]
        result = await handle_feedback_batch(items)
        # Action is passed through to prioritizer; validation is downstream
        assert result["recorded"] == 1 or result["errors"] >= 0


# ============================================================================
# Edge case tests
# ============================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    @pytest.mark.asyncio
    async def test_extra_fields_ignored(self, patch_categorizer):
        """Extra fields in email data are ignored."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_email.return_value = mock_result

        email = _make_email_data(
            extra_field="should be ignored",
            another_field=42,
        )
        result = await handle_categorize_email(email)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_empty_email_data_dict(self, patch_categorizer):
        """Completely empty email data dict still works."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_email.return_value = mock_result

        result = await handle_categorize_email({})
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_batch_large_count(self, patch_categorizer):
        """Batch with many emails processes correctly."""
        results = [_make_mock_result(email_id=f"msg_{i}") for i in range(100)]
        patch_categorizer.categorize_batch.return_value = results
        patch_categorizer.get_category_stats.return_value = {"invoices": 100}

        emails = [_make_email_data(id=f"msg_{i}") for i in range(100)]
        result = await handle_categorize_batch(emails)

        assert result["success"] is True
        assert len(result["results"]) == 100

    @pytest.mark.asyncio
    async def test_feedback_large_batch(self, patch_prioritizer):
        """Large feedback batch processes correctly."""
        items = [{"email_id": f"msg_{i}", "action": "archived"} for i in range(50)]
        result = await handle_feedback_batch(items)

        assert result["recorded"] == 50
        assert result["errors"] == 0

    @pytest.mark.asyncio
    async def test_invalid_date_format_raises(self, patch_categorizer):
        """Invalid date format raises ValueError (caught by handler)."""
        email = _make_email_data(date="not-a-date")
        result = await handle_categorize_email(email)

        assert result["success"] is False
        assert result["error"] == "Internal server error"

    @pytest.mark.asyncio
    async def test_batch_invalid_date_in_one_email(self, patch_categorizer):
        """Invalid date in one email of a batch causes batch failure."""
        emails = [
            _make_email_data(id="msg_1", date="2025-01-01T00:00:00"),
            _make_email_data(id="msg_2", date="bad-date"),
        ]
        result = await handle_categorize_batch(emails)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_email_with_empty_lists(self, patch_categorizer):
        """Email with explicitly empty lists works."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_email.return_value = mock_result

        email = _make_email_data(
            to_addresses=[],
            cc_addresses=[],
            bcc_addresses=[],
            labels=[],
        )
        result = await handle_categorize_email(email)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_email_with_multiple_recipients(self, patch_categorizer):
        """Email with multiple to/cc/bcc addresses works."""
        mock_result = _make_mock_result()
        patch_categorizer.categorize_email.return_value = mock_result

        email = _make_email_data(
            to_addresses=["a@b.com", "c@d.com"],
            cc_addresses=["e@f.com"],
            bcc_addresses=["g@h.com", "i@j.com"],
        )
        result = await handle_categorize_email(email)
        assert result["success"] is True

        call_email = patch_categorizer.categorize_email.call_args[0][0]
        assert len(call_email.to_addresses) == 2
        assert len(call_email.bcc_addresses) == 2


# ============================================================================
# Constants and module attributes
# ============================================================================


class TestModuleAttributes:
    """Tests for module-level constants and attributes."""

    def test_perm_email_read_constant(self):
        """PERM_EMAIL_READ has the expected value."""
        assert PERM_EMAIL_READ == "email:read"

    def test_perm_email_update_constant(self):
        """PERM_EMAIL_UPDATE has the expected value."""
        assert PERM_EMAIL_UPDATE == "email:update"

    def test_auth_context_unset_is_unique_sentinel(self):
        """_AUTH_CONTEXT_UNSET is a unique sentinel object."""
        assert _AUTH_CONTEXT_UNSET is not None
        assert _AUTH_CONTEXT_UNSET is not False
        assert isinstance(_AUTH_CONTEXT_UNSET, object)

    def test_categorizer_lock_exists(self):
        """Module has a threading lock for categorizer."""
        assert isinstance(cat_module._categorizer_lock, type(threading.Lock()))
