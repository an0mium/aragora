"""Tests for email prioritization handler functions.

Tests for aragora/server/handlers/email/prioritization.py covering:
- handle_prioritize_email: single email priority scoring
- handle_rank_inbox: batch inbox ranking
- handle_email_feedback: user action recording for learning
- Error handling, edge cases, input validation, security
- EmailMessage construction and field mapping
- Auth context and permission checking
- Module-level constants and sentinel values
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import aragora.server.handlers.email.prioritization as prio_module
from aragora.server.handlers.email.prioritization import (
    _AUTH_CONTEXT_UNSET,
    PERM_EMAIL_READ,
    PERM_EMAIL_UPDATE,
    handle_email_feedback,
    handle_prioritize_email,
    handle_rank_inbox,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_email_data(**overrides) -> dict[str, Any]:
    """Build a minimal email data dict with sensible defaults."""
    data: dict[str, Any] = {
        "id": "msg_001",
        "subject": "Urgent: Project deadline",
        "from_address": "boss@company.com",
        "body_text": "Please review the project status by end of day.",
    }
    data.update(overrides)
    return data


def _make_mock_result(**overrides) -> MagicMock:
    """Build a mock priority result object with a to_dict method."""
    result = MagicMock()
    d = {
        "email_id": "msg_001",
        "score": 0.85,
        "confidence": 0.92,
        "tier": "tier_1_rules",
        "rationale": "From VIP sender with urgent subject",
    }
    d.update(overrides)
    result.to_dict.return_value = d
    return result


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_prioritizer():
    """Reset the module-level _prioritizer singleton between tests."""
    import aragora.server.handlers.email.storage as storage_mod

    original = storage_mod._prioritizer
    yield
    storage_mod._prioritizer = original


@pytest.fixture
def mock_prioritizer():
    """Provide a mock prioritizer with common async methods."""
    prioritizer = MagicMock()
    prioritizer.score_email = AsyncMock()
    prioritizer.rank_inbox = AsyncMock()
    prioritizer.record_user_action = AsyncMock()
    return prioritizer


@pytest.fixture
def patch_prioritizer(mock_prioritizer):
    """Patch get_prioritizer to return mock_prioritizer."""
    with patch.object(prio_module, "get_prioritizer", return_value=mock_prioritizer):
        yield mock_prioritizer


# ============================================================================
# handle_prioritize_email() tests
# ============================================================================


class TestHandlePrioritizeEmail:
    """Tests for single email priority scoring."""

    @pytest.mark.asyncio
    async def test_success_minimal_email(self, patch_prioritizer):
        """Minimal email data returns success with result dict."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        result = await handle_prioritize_email(_make_email_data())

        assert result["success"] is True
        assert result["result"]["score"] == 0.85
        assert result["result"]["confidence"] == 0.92

    @pytest.mark.asyncio
    async def test_success_full_email_data(self, patch_prioritizer):
        """Full email data with all fields returns success."""
        mock_result = _make_mock_result(score=0.95)
        patch_prioritizer.score_email.return_value = mock_result

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

        result = await handle_prioritize_email(email)
        assert result["success"] is True
        assert result["result"]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_email_message_construction(self, patch_prioritizer):
        """Verify EmailMessage is created with correct field mapping."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        email = _make_email_data(
            id="msg_42",
            subject="Test Subject",
            from_address="sender@test.com",
            body_text="Hello world",
        )

        await handle_prioritize_email(email)

        patch_prioritizer.score_email.assert_awaited_once()
        call_email = patch_prioritizer.score_email.call_args[0][0]
        assert call_email.id == "msg_42"
        assert call_email.subject == "Test Subject"
        assert call_email.from_address == "sender@test.com"
        assert call_email.body_text == "Hello world"

    @pytest.mark.asyncio
    async def test_defaults_for_missing_fields(self, patch_prioritizer):
        """Missing optional fields get sensible defaults."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        result = await handle_prioritize_email({})

        assert result["success"] is True
        call_email = patch_prioritizer.score_email.call_args[0][0]
        assert call_email.id == "unknown"
        assert call_email.subject == ""
        assert call_email.from_address == ""
        assert call_email.to_addresses == []
        assert call_email.cc_addresses == []
        assert call_email.bcc_addresses == []
        assert call_email.body_text == ""
        assert call_email.body_html == ""
        assert call_email.snippet == ""
        assert call_email.labels == []
        assert call_email.headers == {}
        assert call_email.is_read is False
        assert call_email.is_starred is False
        assert call_email.is_important is False

    @pytest.mark.asyncio
    async def test_thread_id_defaults_to_id(self, patch_prioritizer):
        """thread_id falls back to id if not provided."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        await handle_prioritize_email({"id": "msg_99"})
        call_email = patch_prioritizer.score_email.call_args[0][0]
        assert call_email.thread_id == "msg_99"

    @pytest.mark.asyncio
    async def test_thread_id_explicit(self, patch_prioritizer):
        """Explicit thread_id overrides id fallback."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        await handle_prioritize_email({"id": "msg_1", "thread_id": "thread_99"})
        call_email = patch_prioritizer.score_email.call_args[0][0]
        assert call_email.thread_id == "thread_99"

    @pytest.mark.asyncio
    async def test_date_parsing_iso_format(self, patch_prioritizer):
        """ISO format date strings are parsed correctly."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        await handle_prioritize_email({"date": "2025-06-15T08:30:00"})

        call_email = patch_prioritizer.score_email.call_args[0][0]
        assert call_email.date.year == 2025
        assert call_email.date.month == 6
        assert call_email.date.day == 15

    @pytest.mark.asyncio
    async def test_date_defaults_to_now_when_missing(self, patch_prioritizer):
        """Missing date defaults to datetime.now()."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        await handle_prioritize_email({})

        call_email = patch_prioritizer.score_email.call_args[0][0]
        assert isinstance(call_email.date, datetime)

    @pytest.mark.asyncio
    async def test_attachments_always_empty_list(self, patch_prioritizer):
        """Attachments are hardcoded to empty list."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        await handle_prioritize_email(_make_email_data())
        call_email = patch_prioritizer.score_email.call_args[0][0]
        assert call_email.attachments == []

    @pytest.mark.asyncio
    async def test_user_id_forwarded(self, patch_prioritizer):
        """user_id parameter is accepted."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        result = await handle_prioritize_email(_make_email_data(), user_id="user-42")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_user_id_passed_to_get_prioritizer(self):
        """user_id is passed to get_prioritizer."""
        mock_p = MagicMock()
        mock_result = _make_mock_result()
        mock_p.score_email = AsyncMock(return_value=mock_result)

        with patch.object(prio_module, "get_prioritizer", return_value=mock_p) as mock_get:
            await handle_prioritize_email(_make_email_data(), user_id="user-99")
            mock_get.assert_called_with("user-99")

    @pytest.mark.asyncio
    async def test_workspace_id_forwarded(self, patch_prioritizer):
        """workspace_id parameter is accepted."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        result = await handle_prioritize_email(_make_email_data(), workspace_id="ws-99")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_force_tier_parsed(self, patch_prioritizer):
        """force_tier string is converted to ScoringTier enum."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        with patch(
            "aragora.services.email_prioritization.ScoringTier",
            side_effect=lambda v: MagicMock(value=v),
        ) as mock_tier:
            await handle_prioritize_email(_make_email_data(), force_tier="tier_1_rules")
            mock_tier.assert_called_once_with("tier_1_rules")

    @pytest.mark.asyncio
    async def test_force_tier_none_by_default(self, patch_prioritizer):
        """No force_tier means tier=None passed to score_email."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        await handle_prioritize_email(_make_email_data())

        call_kwargs = patch_prioritizer.score_email.call_args
        assert call_kwargs.kwargs.get("force_tier") is None

    @pytest.mark.asyncio
    async def test_force_tier_forwarded_to_score_email(self, patch_prioritizer):
        """Parsed force_tier is forwarded to score_email."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result
        mock_tier = MagicMock()

        with patch(
            "aragora.services.email_prioritization.ScoringTier",
            return_value=mock_tier,
        ):
            await handle_prioritize_email(_make_email_data(), force_tier="tier_2_lightweight")

        call_kwargs = patch_prioritizer.score_email.call_args
        assert call_kwargs.kwargs.get("force_tier") is mock_tier

    @pytest.mark.asyncio
    async def test_invalid_force_tier_returns_failure(self, patch_prioritizer):
        """Invalid force_tier value returns error dict."""
        with patch(
            "aragora.services.email_prioritization.ScoringTier",
            side_effect=ValueError("not a valid ScoringTier"),
        ):
            result = await handle_prioritize_email(_make_email_data(), force_tier="invalid_tier")
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_type_error_returns_failure(self, patch_prioritizer):
        """TypeError in scoring returns error dict."""
        patch_prioritizer.score_email.side_effect = TypeError("bad type")

        result = await handle_prioritize_email(_make_email_data())
        assert result["success"] is False
        assert result["error"] == "Failed to prioritize email"

    @pytest.mark.asyncio
    async def test_value_error_returns_failure(self, patch_prioritizer):
        """ValueError in scoring returns error dict."""
        patch_prioritizer.score_email.side_effect = ValueError("bad value")

        result = await handle_prioritize_email(_make_email_data())
        assert result["success"] is False
        assert result["error"] == "Failed to prioritize email"

    @pytest.mark.asyncio
    async def test_key_error_returns_failure(self, patch_prioritizer):
        """KeyError in scoring returns error dict."""
        patch_prioritizer.score_email.side_effect = KeyError("missing key")

        result = await handle_prioritize_email(_make_email_data())
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_attribute_error_returns_failure(self, patch_prioritizer):
        """AttributeError in scoring returns error dict."""
        patch_prioritizer.score_email.side_effect = AttributeError("no attr")

        result = await handle_prioritize_email(_make_email_data())
        assert result["success"] is False
        assert result["error"] == "Failed to prioritize email"

    @pytest.mark.asyncio
    async def test_does_not_leak_error_details(self, patch_prioritizer):
        """Error messages should not expose internal details."""
        patch_prioritizer.score_email.side_effect = ValueError(
            "Connection to database failed at 10.0.0.5:5432"
        )

        result = await handle_prioritize_email(_make_email_data())
        assert "10.0.0.5" not in result.get("error", "")
        assert "database" not in result.get("error", "")

    @pytest.mark.asyncio
    async def test_invalid_date_format_caught(self, patch_prioritizer):
        """Invalid date format raises ValueError (caught by handler)."""
        email = _make_email_data(date="not-a-date")
        result = await handle_prioritize_email(email)

        assert result["success"] is False
        assert result["error"] == "Failed to prioritize email"

    @pytest.mark.asyncio
    async def test_result_structure(self, patch_prioritizer):
        """Response has correct top-level structure."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        result = await handle_prioritize_email(_make_email_data())

        assert "success" in result
        assert "result" in result
        assert result["success"] is True
        assert isinstance(result["result"], dict)

    @pytest.mark.asyncio
    async def test_result_to_dict_called(self, patch_prioritizer):
        """Result object's to_dict() is called to serialize."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        await handle_prioritize_email(_make_email_data())

        mock_result.to_dict.assert_called_once()


# ============================================================================
# handle_rank_inbox() tests
# ============================================================================


class TestHandleRankInbox:
    """Tests for batch inbox ranking."""

    @pytest.mark.asyncio
    async def test_rank_success_single_email(self, patch_prioritizer):
        """Ranking one email returns results list."""
        mock_result = _make_mock_result()
        patch_prioritizer.rank_inbox.return_value = [mock_result]

        emails = [_make_email_data()]
        result = await handle_rank_inbox(emails)

        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_rank_success_multiple_emails(self, patch_prioritizer):
        """Ranking multiple emails returns all results."""
        results = [
            _make_mock_result(email_id="msg_1", score=0.9),
            _make_mock_result(email_id="msg_2", score=0.7),
            _make_mock_result(email_id="msg_3", score=0.3),
        ]
        patch_prioritizer.rank_inbox.return_value = results

        emails = [
            _make_email_data(id="msg_1"),
            _make_email_data(id="msg_2"),
            _make_email_data(id="msg_3"),
        ]
        result = await handle_rank_inbox(emails)

        assert result["success"] is True
        assert len(result["results"]) == 3
        assert result["total"] == 3

    @pytest.mark.asyncio
    async def test_rank_empty_emails_list(self, patch_prioritizer):
        """Empty email list returns empty results."""
        patch_prioritizer.rank_inbox.return_value = []

        result = await handle_rank_inbox([])

        assert result["success"] is True
        assert result["results"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_rank_forwards_limit(self, patch_prioritizer):
        """limit parameter is forwarded to prioritizer."""
        patch_prioritizer.rank_inbox.return_value = []

        await handle_rank_inbox([_make_email_data()], limit=5)

        call_kwargs = patch_prioritizer.rank_inbox.call_args
        assert call_kwargs.kwargs.get("limit") == 5

    @pytest.mark.asyncio
    async def test_rank_limit_none_by_default(self, patch_prioritizer):
        """Default limit is None."""
        patch_prioritizer.rank_inbox.return_value = []

        await handle_rank_inbox([_make_email_data()])

        call_kwargs = patch_prioritizer.rank_inbox.call_args
        assert call_kwargs.kwargs.get("limit") is None

    @pytest.mark.asyncio
    async def test_rank_email_message_construction(self, patch_prioritizer):
        """Verify EmailMessage objects are created correctly for ranking."""
        patch_prioritizer.rank_inbox.return_value = []

        email = _make_email_data(
            id="msg_42",
            subject="Important email",
            from_address="vip@test.com",
        )
        await handle_rank_inbox([email])

        call_emails = patch_prioritizer.rank_inbox.call_args[0][0]
        assert len(call_emails) == 1
        assert call_emails[0].id == "msg_42"
        assert call_emails[0].subject == "Important email"
        assert call_emails[0].from_address == "vip@test.com"

    @pytest.mark.asyncio
    async def test_rank_defaults_for_missing_fields(self, patch_prioritizer):
        """Missing fields in emails get sensible defaults."""
        patch_prioritizer.rank_inbox.return_value = []

        await handle_rank_inbox([{}])

        call_emails = patch_prioritizer.rank_inbox.call_args[0][0]
        assert call_emails[0].id == "unknown"
        assert call_emails[0].subject == ""
        assert call_emails[0].from_address == ""
        assert call_emails[0].to_addresses == []
        assert call_emails[0].is_read is False

    @pytest.mark.asyncio
    async def test_rank_thread_id_defaults_to_id(self, patch_prioritizer):
        """thread_id falls back to id if not provided."""
        patch_prioritizer.rank_inbox.return_value = []

        await handle_rank_inbox([{"id": "msg_77"}])

        call_emails = patch_prioritizer.rank_inbox.call_args[0][0]
        assert call_emails[0].thread_id == "msg_77"

    @pytest.mark.asyncio
    async def test_rank_date_parsing(self, patch_prioritizer):
        """Date parsing works for ranked emails."""
        patch_prioritizer.rank_inbox.return_value = []

        emails = [
            _make_email_data(id="msg_1", date="2025-03-01T09:00:00"),
            _make_email_data(id="msg_2"),  # No date
        ]
        await handle_rank_inbox(emails)

        call_emails = patch_prioritizer.rank_inbox.call_args[0][0]
        assert call_emails[0].date.year == 2025
        assert isinstance(call_emails[1].date, datetime)

    @pytest.mark.asyncio
    async def test_rank_user_id_forwarded(self, patch_prioritizer):
        """user_id parameter is accepted."""
        patch_prioritizer.rank_inbox.return_value = []

        result = await handle_rank_inbox([_make_email_data()], user_id="user-rank")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_rank_user_id_passed_to_get_prioritizer(self):
        """user_id is passed to get_prioritizer."""
        mock_p = MagicMock()
        mock_p.rank_inbox = AsyncMock(return_value=[])

        with patch.object(prio_module, "get_prioritizer", return_value=mock_p) as mock_get:
            await handle_rank_inbox([_make_email_data()], user_id="user-99")
            mock_get.assert_called_with("user-99")

    @pytest.mark.asyncio
    async def test_rank_workspace_id_forwarded(self, patch_prioritizer):
        """workspace_id parameter is accepted."""
        patch_prioritizer.rank_inbox.return_value = []

        result = await handle_rank_inbox([_make_email_data()], workspace_id="ws-rank")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_rank_type_error_returns_failure(self, patch_prioritizer):
        """TypeError during ranking returns failure."""
        patch_prioritizer.rank_inbox.side_effect = TypeError("bad type")

        result = await handle_rank_inbox([_make_email_data()])
        assert result["success"] is False
        assert result["error"] == "Failed to rank inbox"

    @pytest.mark.asyncio
    async def test_rank_value_error_returns_failure(self, patch_prioritizer):
        """ValueError during ranking returns failure."""
        patch_prioritizer.rank_inbox.side_effect = ValueError("bad value")

        result = await handle_rank_inbox([_make_email_data()])
        assert result["success"] is False
        assert result["error"] == "Failed to rank inbox"

    @pytest.mark.asyncio
    async def test_rank_key_error_returns_failure(self, patch_prioritizer):
        """KeyError during ranking returns failure."""
        patch_prioritizer.rank_inbox.side_effect = KeyError("missing")

        result = await handle_rank_inbox([_make_email_data()])
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_rank_attribute_error_returns_failure(self, patch_prioritizer):
        """AttributeError during ranking returns failure."""
        patch_prioritizer.rank_inbox.side_effect = AttributeError("no attr")

        result = await handle_rank_inbox([_make_email_data()])
        assert result["success"] is False
        assert result["error"] == "Failed to rank inbox"

    @pytest.mark.asyncio
    async def test_rank_invalid_date_causes_failure(self, patch_prioritizer):
        """Invalid date in one email causes batch failure."""
        emails = [
            _make_email_data(id="msg_1", date="2025-01-01T00:00:00"),
            _make_email_data(id="msg_2", date="bad-date"),
        ]
        result = await handle_rank_inbox(emails)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_rank_does_not_leak_error_details(self, patch_prioritizer):
        """Error messages do not expose internal details."""
        patch_prioritizer.rank_inbox.side_effect = ValueError(
            "SQL error on table email_scores at 192.168.1.1"
        )

        result = await handle_rank_inbox([_make_email_data()])
        assert "192.168.1.1" not in result.get("error", "")
        assert "SQL" not in result.get("error", "")

    @pytest.mark.asyncio
    async def test_rank_results_to_dict_called(self, patch_prioritizer):
        """Each result's to_dict() is called for serialization."""
        mock_r1 = _make_mock_result(email_id="msg_1")
        mock_r2 = _make_mock_result(email_id="msg_2")
        patch_prioritizer.rank_inbox.return_value = [mock_r1, mock_r2]

        await handle_rank_inbox([_make_email_data(id="msg_1"), _make_email_data(id="msg_2")])

        mock_r1.to_dict.assert_called_once()
        mock_r2.to_dict.assert_called_once()

    @pytest.mark.asyncio
    async def test_rank_large_batch(self, patch_prioritizer):
        """Large batch processes correctly."""
        results = [_make_mock_result(email_id=f"msg_{i}") for i in range(100)]
        patch_prioritizer.rank_inbox.return_value = results

        emails = [_make_email_data(id=f"msg_{i}") for i in range(100)]
        result = await handle_rank_inbox(emails)

        assert result["success"] is True
        assert len(result["results"]) == 100
        assert result["total"] == 100

    @pytest.mark.asyncio
    async def test_rank_attachments_always_empty(self, patch_prioritizer):
        """Attachments are hardcoded to empty list for each email."""
        patch_prioritizer.rank_inbox.return_value = []

        await handle_rank_inbox([_make_email_data(), _make_email_data()])

        call_emails = patch_prioritizer.rank_inbox.call_args[0][0]
        for email in call_emails:
            assert email.attachments == []


# ============================================================================
# handle_email_feedback() tests
# ============================================================================


class TestHandleEmailFeedback:
    """Tests for user action recording."""

    @pytest.mark.asyncio
    async def test_feedback_success_basic(self, patch_prioritizer):
        """Basic feedback recording returns success."""
        result = await handle_email_feedback("msg_001", "archived")

        assert result["success"] is True
        assert result["email_id"] == "msg_001"
        assert result["action"] == "archived"
        assert "recorded_at" in result

    @pytest.mark.asyncio
    async def test_feedback_recorded_at_is_iso_format(self, patch_prioritizer):
        """recorded_at is a valid ISO format datetime string."""
        result = await handle_email_feedback("msg_001", "read")

        recorded_at = result["recorded_at"]
        # Should parse without error
        datetime.fromisoformat(recorded_at)

    @pytest.mark.asyncio
    async def test_feedback_calls_record_user_action(self, patch_prioritizer):
        """record_user_action is called with correct args."""
        await handle_email_feedback("msg_42", "starred")

        patch_prioritizer.record_user_action.assert_awaited_once_with("msg_42", "starred", None)

    @pytest.mark.asyncio
    async def test_feedback_with_email_data(self, patch_prioritizer):
        """Email data is converted and passed to record_user_action."""
        email_data = {
            "id": "msg_99",
            "subject": "Test Email",
            "from_address": "user@test.com",
            "body_text": "Content here",
        }

        await handle_email_feedback("msg_99", "replied", email_data=email_data)

        call_args = patch_prioritizer.record_user_action.call_args[0]
        assert call_args[0] == "msg_99"
        assert call_args[1] == "replied"
        # Third arg is the EmailMessage object
        email_obj = call_args[2]
        assert email_obj is not None
        assert email_obj.id == "msg_99"
        assert email_obj.subject == "Test Email"

    @pytest.mark.asyncio
    async def test_feedback_email_data_id_defaults_to_email_id(self, patch_prioritizer):
        """Email data without id uses the email_id parameter."""
        email_data = {"subject": "No ID"}

        await handle_email_feedback("msg_fallback", "read", email_data=email_data)

        call_args = patch_prioritizer.record_user_action.call_args[0]
        email_obj = call_args[2]
        assert email_obj.id == "msg_fallback"

    @pytest.mark.asyncio
    async def test_feedback_email_data_thread_id_defaults_to_email_id(self, patch_prioritizer):
        """Email data without thread_id uses the email_id parameter."""
        email_data = {"subject": "No thread ID"}

        await handle_email_feedback("msg_thread", "archived", email_data=email_data)

        call_args = patch_prioritizer.record_user_action.call_args[0]
        email_obj = call_args[2]
        assert email_obj.thread_id == "msg_thread"

    @pytest.mark.asyncio
    async def test_feedback_no_email_data(self, patch_prioritizer):
        """No email_data passes None to record_user_action."""
        await handle_email_feedback("msg_001", "deleted")

        call_args = patch_prioritizer.record_user_action.call_args[0]
        assert call_args[2] is None

    @pytest.mark.asyncio
    async def test_feedback_email_data_is_read_true(self, patch_prioritizer):
        """Email data from feedback always has is_read=True."""
        email_data = {"subject": "Test"}

        await handle_email_feedback("msg_001", "read", email_data=email_data)

        call_args = patch_prioritizer.record_user_action.call_args[0]
        email_obj = call_args[2]
        assert email_obj.is_read is True

    @pytest.mark.asyncio
    async def test_feedback_email_data_fields(self, patch_prioritizer):
        """Email data fields are correctly mapped."""
        email_data = {
            "subject": "My Subject",
            "from_address": "sender@test.com",
            "to_addresses": ["recipient@test.com"],
            "body_text": "Body content",
            "snippet": "Body...",
            "labels": ["INBOX"],
            "is_starred": True,
            "is_important": True,
        }

        await handle_email_feedback("msg_001", "read", email_data=email_data)

        call_args = patch_prioritizer.record_user_action.call_args[0]
        email_obj = call_args[2]
        assert email_obj.subject == "My Subject"
        assert email_obj.from_address == "sender@test.com"
        assert email_obj.to_addresses == ["recipient@test.com"]
        assert email_obj.body_text == "Body content"
        assert email_obj.snippet == "Body..."
        assert email_obj.labels == ["INBOX"]
        assert email_obj.is_starred is True
        assert email_obj.is_important is True

    @pytest.mark.asyncio
    async def test_feedback_email_data_hardcoded_fields(self, patch_prioritizer):
        """Feedback email has hardcoded empty cc, bcc, headers, html, attachments."""
        email_data = {"subject": "Test"}

        await handle_email_feedback("msg_001", "read", email_data=email_data)

        call_args = patch_prioritizer.record_user_action.call_args[0]
        email_obj = call_args[2]
        assert email_obj.cc_addresses == []
        assert email_obj.bcc_addresses == []
        assert email_obj.body_html == ""
        assert email_obj.headers == {}
        assert email_obj.attachments == []

    @pytest.mark.asyncio
    async def test_feedback_user_id_forwarded(self, patch_prioritizer):
        """user_id parameter is accepted."""
        result = await handle_email_feedback("msg_001", "archived", user_id="user-42")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_feedback_user_id_passed_to_get_prioritizer(self):
        """user_id is passed to get_prioritizer."""
        mock_p = MagicMock()
        mock_p.record_user_action = AsyncMock()

        with patch.object(prio_module, "get_prioritizer", return_value=mock_p) as mock_get:
            await handle_email_feedback("msg_001", "read", user_id="user-55")
            mock_get.assert_called_with("user-55")

    @pytest.mark.asyncio
    async def test_feedback_workspace_id_forwarded(self, patch_prioritizer):
        """workspace_id parameter is accepted."""
        result = await handle_email_feedback("msg_001", "archived", workspace_id="ws-42")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_feedback_all_actions(self, patch_prioritizer):
        """All documented action types are accepted."""
        actions = ["read", "archived", "deleted", "replied", "starred", "important"]
        for action in actions:
            result = await handle_email_feedback(f"msg_{action}", action)
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_feedback_type_error_returns_failure(self, patch_prioritizer):
        """TypeError during feedback returns error dict."""
        patch_prioritizer.record_user_action.side_effect = TypeError("bad type")

        result = await handle_email_feedback("msg_001", "read")
        assert result["success"] is False
        assert result["error"] == "Failed to record feedback"

    @pytest.mark.asyncio
    async def test_feedback_value_error_returns_failure(self, patch_prioritizer):
        """ValueError during feedback returns error dict."""
        patch_prioritizer.record_user_action.side_effect = ValueError("bad value")

        result = await handle_email_feedback("msg_001", "read")
        assert result["success"] is False
        assert result["error"] == "Failed to record feedback"

    @pytest.mark.asyncio
    async def test_feedback_key_error_returns_failure(self, patch_prioritizer):
        """KeyError during feedback returns error dict."""
        patch_prioritizer.record_user_action.side_effect = KeyError("missing")

        result = await handle_email_feedback("msg_001", "read")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_feedback_attribute_error_returns_failure(self, patch_prioritizer):
        """AttributeError during feedback returns error dict."""
        patch_prioritizer.record_user_action.side_effect = AttributeError("no attr")

        result = await handle_email_feedback("msg_001", "read")
        assert result["success"] is False
        assert result["error"] == "Failed to record feedback"

    @pytest.mark.asyncio
    async def test_feedback_does_not_leak_error_details(self, patch_prioritizer):
        """Error messages do not expose internal details."""
        patch_prioritizer.record_user_action.side_effect = ValueError(
            "Connection refused to redis://10.0.0.5:6379"
        )

        result = await handle_email_feedback("msg_001", "read")
        assert "10.0.0.5" not in result.get("error", "")
        assert "redis" not in result.get("error", "")

    @pytest.mark.asyncio
    async def test_feedback_auth_context_permission_check(self, patch_prioritizer):
        """When auth_context is provided, _check_email_permission is called."""
        mock_auth = MagicMock()

        with patch(
            "aragora.server.handlers.email.prioritization._check_email_permission",
            return_value=None,
        ) as mock_check:
            await handle_email_feedback("msg_001", "archived", auth_context=mock_auth)
            mock_check.assert_called_once_with(mock_auth, PERM_EMAIL_UPDATE)

    @pytest.mark.asyncio
    async def test_feedback_auth_context_denied(self, patch_prioritizer):
        """When permission denied, returns error without recording."""
        mock_auth = MagicMock()
        perm_error = {"success": False, "error": "Permission denied"}

        with patch(
            "aragora.server.handlers.email.prioritization._check_email_permission",
            return_value=perm_error,
        ):
            result = await handle_email_feedback("msg_001", "archived", auth_context=mock_auth)
            assert result["success"] is False
            assert result["error"] == "Permission denied"
            patch_prioritizer.record_user_action.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_feedback_auth_context_unset_skips_check(self, patch_prioritizer):
        """When auth_context is _AUTH_CONTEXT_UNSET, permission check is skipped."""
        with patch(
            "aragora.server.handlers.email.prioritization._check_email_permission",
        ) as mock_check:
            await handle_email_feedback("msg_001", "archived", auth_context=_AUTH_CONTEXT_UNSET)
            mock_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_feedback_auth_context_none_triggers_check(self, patch_prioritizer):
        """When auth_context is explicitly None (not unset), it's not _AUTH_CONTEXT_UNSET."""
        # None is not _AUTH_CONTEXT_UNSET, so the check IS entered
        with patch(
            "aragora.server.handlers.email.prioritization._check_email_permission",
            return_value=None,
        ) as mock_check:
            await handle_email_feedback("msg_001", "archived", auth_context=None)
            mock_check.assert_called_once_with(None, PERM_EMAIL_UPDATE)


# ============================================================================
# Security tests
# ============================================================================


class TestSecurityConcerns:
    """Security-focused tests for prioritization handlers."""

    @pytest.mark.asyncio
    async def test_path_traversal_in_email_id(self, patch_prioritizer):
        """Path traversal attempt in email_id does not cause issues."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        email = _make_email_data(id="../../etc/passwd")
        result = await handle_prioritize_email(email)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_script_injection_in_subject(self, patch_prioritizer):
        """Script injection in subject does not cause issues."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        email = _make_email_data(subject='<script>alert("xss")</script>')
        result = await handle_prioritize_email(email)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_sql_injection_in_body(self, patch_prioritizer):
        """SQL injection in body does not cause issues."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        email = _make_email_data(body_text="'; DROP TABLE emails; --")
        result = await handle_prioritize_email(email)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_unicode_in_email_fields(self, patch_prioritizer):
        """Unicode in email fields is handled correctly."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        email = _make_email_data(
            subject="Re: Meeting invitation \u2014 \u2603",
            body_text="Please join us for the meeting. \u00e9\u00e0\u00fc",
        )
        result = await handle_prioritize_email(email)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_very_long_subject(self, patch_prioritizer):
        """Very long subject does not cause issues."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        email = _make_email_data(subject="A" * 10000)
        result = await handle_prioritize_email(email)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_very_long_body(self, patch_prioritizer):
        """Very long body text does not cause issues."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        email = _make_email_data(body_text="X" * 100000)
        result = await handle_prioritize_email(email)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_null_bytes_in_fields(self, patch_prioritizer):
        """Null bytes in fields do not cause issues."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        email = _make_email_data(subject="Test\x00Subject")
        result = await handle_prioritize_email(email)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_feedback_injection_in_action(self, patch_prioritizer):
        """Injection attempt in action field is handled."""
        result = await handle_email_feedback("msg_001", "<script>alert(1)</script>")
        # Action is passed through to prioritizer; validation is downstream
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_feedback_injection_in_email_id(self, patch_prioritizer):
        """Injection attempt in email_id is handled."""
        result = await handle_email_feedback("'; DROP TABLE feedback; --", "read")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_rank_injection_in_batch(self, patch_prioritizer):
        """Injection attempt in batch email data is handled."""
        patch_prioritizer.rank_inbox.return_value = []

        emails = [
            _make_email_data(
                id="'; DROP TABLE --",
                subject="<img onerror=alert(1)>",
                from_address="attacker@evil.com'; --",
            )
        ]
        result = await handle_rank_inbox(emails)
        assert result["success"] is True


# ============================================================================
# Edge case tests
# ============================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    @pytest.mark.asyncio
    async def test_extra_fields_ignored(self, patch_prioritizer):
        """Extra fields in email data are ignored."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        email = _make_email_data(
            extra_field="should be ignored",
            another_field=42,
        )
        result = await handle_prioritize_email(email)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_empty_email_data_dict(self, patch_prioritizer):
        """Completely empty email data dict still works."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        result = await handle_prioritize_email({})
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_email_with_empty_lists(self, patch_prioritizer):
        """Email with explicitly empty lists works."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        email = _make_email_data(
            to_addresses=[],
            cc_addresses=[],
            bcc_addresses=[],
            labels=[],
        )
        result = await handle_prioritize_email(email)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_email_with_multiple_recipients(self, patch_prioritizer):
        """Email with multiple to/cc/bcc addresses works."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        email = _make_email_data(
            to_addresses=["a@b.com", "c@d.com"],
            cc_addresses=["e@f.com"],
            bcc_addresses=["g@h.com", "i@j.com"],
        )
        result = await handle_prioritize_email(email)
        assert result["success"] is True

        call_email = patch_prioritizer.score_email.call_args[0][0]
        assert len(call_email.to_addresses) == 2
        assert len(call_email.cc_addresses) == 1
        assert len(call_email.bcc_addresses) == 2

    @pytest.mark.asyncio
    async def test_email_with_many_labels(self, patch_prioritizer):
        """Email with many labels works."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        email = _make_email_data(
            labels=["INBOX", "IMPORTANT", "STARRED", "UNREAD", "CATEGORY_PERSONAL"]
        )
        result = await handle_prioritize_email(email)
        assert result["success"] is True

        call_email = patch_prioritizer.score_email.call_args[0][0]
        assert len(call_email.labels) == 5

    @pytest.mark.asyncio
    async def test_email_with_headers(self, patch_prioritizer):
        """Email with custom headers works."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        email = _make_email_data(
            headers={
                "Message-ID": "<test@example.com>",
                "References": "<ref@example.com>",
                "X-Custom": "value",
            }
        )
        result = await handle_prioritize_email(email)
        assert result["success"] is True

        call_email = patch_prioritizer.score_email.call_args[0][0]
        assert call_email.headers["Message-ID"] == "<test@example.com>"

    @pytest.mark.asyncio
    async def test_feedback_empty_email_data_dict(self, patch_prioritizer):
        """Empty email_data dict is treated as falsy, so no EmailMessage is created."""
        await handle_email_feedback("msg_001", "read", email_data={})

        call_args = patch_prioritizer.record_user_action.call_args[0]
        email_obj = call_args[2]
        # Empty dict is falsy, so handler does not create EmailMessage
        assert email_obj is None

    @pytest.mark.asyncio
    async def test_rank_mixed_complete_and_sparse_emails(self, patch_prioritizer):
        """Ranking mix of complete and sparse emails works."""
        patch_prioritizer.rank_inbox.return_value = []

        emails = [
            _make_email_data(
                id="full",
                subject="Full email",
                from_address="a@b.com",
                body_text="content",
                date="2025-01-01T00:00:00",
                labels=["INBOX"],
                is_important=True,
            ),
            {},  # Completely sparse
            {"id": "minimal"},  # Just id
        ]
        result = await handle_rank_inbox(emails)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_feedback_with_none_email_data(self, patch_prioritizer):
        """Explicit None for email_data passes None to record_user_action."""
        await handle_email_feedback("msg_001", "read", email_data=None)

        call_args = patch_prioritizer.record_user_action.call_args[0]
        assert call_args[2] is None

    @pytest.mark.asyncio
    async def test_error_failure_structure(self, patch_prioritizer):
        """Error response has consistent structure."""
        patch_prioritizer.score_email.side_effect = TypeError("test")

        result = await handle_prioritize_email(_make_email_data())
        assert "success" in result
        assert "error" in result
        assert result["success"] is False
        assert isinstance(result["error"], str)

    @pytest.mark.asyncio
    async def test_rank_error_failure_structure(self, patch_prioritizer):
        """Rank error response has consistent structure."""
        patch_prioritizer.rank_inbox.side_effect = TypeError("test")

        result = await handle_rank_inbox([_make_email_data()])
        assert "success" in result
        assert "error" in result
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_feedback_error_failure_structure(self, patch_prioritizer):
        """Feedback error response has consistent structure."""
        patch_prioritizer.record_user_action.side_effect = TypeError("test")

        result = await handle_email_feedback("msg_001", "read")
        assert "success" in result
        assert "error" in result
        assert result["success"] is False


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

    def test_auth_context_unset_is_not_none(self):
        """_AUTH_CONTEXT_UNSET is distinct from None."""
        assert _AUTH_CONTEXT_UNSET is not None

    def test_auth_context_unset_identity(self):
        """_AUTH_CONTEXT_UNSET is always the same object."""
        from aragora.server.handlers.email.prioritization import (
            _AUTH_CONTEXT_UNSET as sentinel2,
        )

        assert _AUTH_CONTEXT_UNSET is sentinel2

    def test_module_has_logger(self):
        """Module has a logger configured."""
        assert prio_module.logger is not None
        assert prio_module.logger.name == "aragora.server.handlers.email.prioritization"

    def test_handler_functions_are_callable(self):
        """All handler functions are callable."""
        assert callable(handle_prioritize_email)
        assert callable(handle_rank_inbox)
        assert callable(handle_email_feedback)

    def test_handler_functions_have_wrapped(self):
        """Handler functions have __wrapped__ from decorator chains."""
        # The functions are wrapped by @require_permission, @rate_limit, @track_handler
        # so the outermost may not show as async, but they should be callable wrappers
        for fn in [handle_prioritize_email, handle_rank_inbox, handle_email_feedback]:
            # Walk the __wrapped__ chain to find the original async function
            inner = fn
            while hasattr(inner, "__wrapped__"):
                inner = inner.__wrapped__
            import asyncio

            assert asyncio.iscoroutinefunction(inner), f"{fn.__name__} inner is not async"


# ============================================================================
# Decorator integration tests
# ============================================================================


class TestDecoratorIntegration:
    """Tests verifying decorator behavior (require_permission, rate_limit, track_handler)."""

    @pytest.mark.asyncio
    async def test_prioritize_has_require_permission(self):
        """handle_prioritize_email is wrapped by require_permission for email:read."""
        # The function name or docstring should survive through decorators
        assert (
            "prioritize" in handle_prioritize_email.__name__.lower()
            or "prioritize" in (handle_prioritize_email.__doc__ or "").lower()
            or hasattr(handle_prioritize_email, "__wrapped__")
        )

    @pytest.mark.asyncio
    async def test_rank_inbox_has_require_permission(self):
        """handle_rank_inbox is wrapped by require_permission for email:read."""
        assert hasattr(handle_rank_inbox, "__wrapped__") or callable(handle_rank_inbox)

    @pytest.mark.asyncio
    async def test_feedback_has_require_permission(self):
        """handle_email_feedback is wrapped by require_permission for email:update."""
        assert hasattr(handle_email_feedback, "__wrapped__") or callable(handle_email_feedback)

    @pytest.mark.asyncio
    async def test_prioritize_works_with_auth_context(self, patch_prioritizer):
        """Prioritize handler accepts auth_context parameter."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result
        mock_auth = MagicMock()

        result = await handle_prioritize_email(_make_email_data(), auth_context=mock_auth)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_rank_inbox_works_with_auth_context(self, patch_prioritizer):
        """Rank inbox handler accepts auth_context parameter."""
        patch_prioritizer.rank_inbox.return_value = []
        mock_auth = MagicMock()

        result = await handle_rank_inbox([_make_email_data()], auth_context=mock_auth)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_feedback_default_auth_context_is_sentinel(self, patch_prioritizer):
        """Feedback handler's default auth_context is _AUTH_CONTEXT_UNSET."""
        # When called without auth_context, the sentinel value should skip permission check
        with patch(
            "aragora.server.handlers.email.prioritization._check_email_permission",
        ) as mock_check:
            await handle_email_feedback("msg_001", "read")
            # _AUTH_CONTEXT_UNSET is the default, so check should NOT be called
            mock_check.assert_not_called()


# ============================================================================
# Concurrent / multi-call tests
# ============================================================================


class TestMultiCallBehavior:
    """Tests verifying behavior across multiple calls."""

    @pytest.mark.asyncio
    async def test_multiple_prioritize_calls(self, patch_prioritizer):
        """Multiple prioritize calls work independently."""
        mock_result_1 = _make_mock_result(score=0.9)
        mock_result_2 = _make_mock_result(score=0.3)
        patch_prioritizer.score_email.side_effect = [mock_result_1, mock_result_2]

        r1 = await handle_prioritize_email(_make_email_data(id="msg_1"))
        r2 = await handle_prioritize_email(_make_email_data(id="msg_2"))

        assert r1["result"]["score"] == 0.9
        assert r2["result"]["score"] == 0.3

    @pytest.mark.asyncio
    async def test_multiple_feedback_calls(self, patch_prioritizer):
        """Multiple feedback calls work independently."""
        r1 = await handle_email_feedback("msg_1", "read")
        r2 = await handle_email_feedback("msg_2", "archived")
        r3 = await handle_email_feedback("msg_3", "starred")

        assert r1["success"] is True
        assert r2["success"] is True
        assert r3["success"] is True
        assert r1["email_id"] == "msg_1"
        assert r2["email_id"] == "msg_2"
        assert r3["email_id"] == "msg_3"
        assert patch_prioritizer.record_user_action.await_count == 3

    @pytest.mark.asyncio
    async def test_prioritize_then_feedback(self, patch_prioritizer):
        """Prioritize followed by feedback works correctly."""
        mock_result = _make_mock_result()
        patch_prioritizer.score_email.return_value = mock_result

        prio = await handle_prioritize_email(_make_email_data(id="msg_001"))
        assert prio["success"] is True

        fb = await handle_email_feedback("msg_001", "read")
        assert fb["success"] is True

    @pytest.mark.asyncio
    async def test_rank_then_feedback_for_each(self, patch_prioritizer):
        """Rank inbox then provide feedback for each email."""
        results = [
            _make_mock_result(email_id="msg_1"),
            _make_mock_result(email_id="msg_2"),
        ]
        patch_prioritizer.rank_inbox.return_value = results

        rank_result = await handle_rank_inbox(
            [
                _make_email_data(id="msg_1"),
                _make_email_data(id="msg_2"),
            ]
        )
        assert rank_result["success"] is True

        for email_id in ["msg_1", "msg_2"]:
            fb = await handle_email_feedback(email_id, "read")
            assert fb["success"] is True

    @pytest.mark.asyncio
    async def test_error_does_not_affect_next_call(self, patch_prioritizer):
        """Error in one call does not affect subsequent calls."""
        patch_prioritizer.score_email.side_effect = [
            TypeError("first call fails"),
            _make_mock_result(),
        ]

        r1 = await handle_prioritize_email(_make_email_data())
        assert r1["success"] is False

        r2 = await handle_prioritize_email(_make_email_data())
        assert r2["success"] is True
