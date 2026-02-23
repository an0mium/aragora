"""Tests for email cross-channel context handlers.

Tests for aragora/server/handlers/email/context.py covering:
- handle_get_context: success path, service call, to_dict conversion, error handling
  for all caught exception types (ConnectionError, TimeoutError, OSError, ValueError),
  default parameter values, auth_context forwarding
- handle_get_email_context_boost: success path, EmailMessage construction from
  email_data dict, boost response field mapping, error handling for all caught
  exception types (TypeError, ValueError, KeyError, AttributeError, ConnectionError,
  TimeoutError), missing/default email_data fields, auth_context forwarding
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import aragora.server.handlers.email.storage as storage_module
from aragora.server.handlers.email.context import (
    PERM_EMAIL_READ,
    handle_get_context,
    handle_get_email_context_boost,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_auth_context(user_id: str = "user-1") -> MagicMock:
    """Build a mock auth context with a user_id attribute."""
    ctx = MagicMock()
    ctx.user_id = user_id
    return ctx


def _make_channel_context(user_email: str = "alice@example.com") -> MagicMock:
    """Build a mock ChannelContext returned by get_user_context."""
    ctx = MagicMock()
    ctx.user_email = user_email
    ctx.to_dict.return_value = {
        "user_email": user_email,
        "timestamp": "2026-02-23T10:00:00",
        "overall_activity_score": 0.5,
        "is_likely_busy": False,
        "suggested_response_window": None,
        "active_projects": [],
        "active_contacts": [],
        "slack": None,
        "drive": None,
        "calendar": None,
    }
    return ctx


def _make_email_context_boost(
    email_id: str = "email-1",
    total_boost: float = 0.35,
    slack_activity_boost: float = 0.15,
    drive_relevance_boost: float = 0.10,
    calendar_urgency_boost: float = 0.10,
) -> MagicMock:
    """Build a mock EmailContextBoost returned by get_email_context."""
    boost = MagicMock()
    boost.email_id = email_id
    boost.total_boost = total_boost
    boost.slack_activity_boost = slack_activity_boost
    boost.drive_relevance_boost = drive_relevance_boost
    boost.calendar_urgency_boost = calendar_urgency_boost
    boost.slack_reason = "Sender active on Slack"
    boost.drive_reason = "Related docs found"
    boost.calendar_reason = "Meeting in 30min"
    boost.related_slack_channels = ["#general", "#eng"]
    boost.related_drive_files = ["doc1.pdf"]
    boost.related_meetings = ["standup"]
    return boost


def _full_email_data() -> dict[str, Any]:
    """Return a fully populated email_data dict."""
    return {
        "id": "msg-123",
        "thread_id": "thr-456",
        "subject": "Q4 Review",
        "from_address": "boss@company.com",
        "to_addresses": ["me@company.com"],
        "body_text": "Please review the Q4 numbers.",
        "snippet": "Please review...",
    }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_context_service():
    """Reset the module-level _context_service singleton between tests."""
    orig = storage_module._context_service
    yield
    storage_module._context_service = orig


# ============================================================================
# handle_get_context - Success Paths
# ============================================================================


class TestGetContextSuccess:
    """Tests for the happy path of handle_get_context."""

    @pytest.mark.asyncio
    async def test_success_returns_context_dict(self):
        """Returns success=True with context from service.to_dict()."""
        mock_ctx = _make_channel_context("alice@example.com")
        mock_service = MagicMock()
        mock_service.get_user_context = AsyncMock(return_value=mock_ctx)

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_context(email_address="alice@example.com")

        assert result["success"] is True
        assert result["context"]["user_email"] == "alice@example.com"

    @pytest.mark.asyncio
    async def test_calls_service_with_email_address(self):
        """Passes the email_address to service.get_user_context."""
        mock_ctx = _make_channel_context()
        mock_service = MagicMock()
        mock_service.get_user_context = AsyncMock(return_value=mock_ctx)

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            await handle_get_context(email_address="specific@test.com")

        mock_service.get_user_context.assert_awaited_once_with("specific@test.com")

    @pytest.mark.asyncio
    async def test_context_to_dict_called(self):
        """Calls to_dict() on the returned ChannelContext."""
        mock_ctx = _make_channel_context()
        mock_service = MagicMock()
        mock_service.get_user_context = AsyncMock(return_value=mock_ctx)

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_context(email_address="alice@example.com")

        mock_ctx.to_dict.assert_called_once()
        assert result["context"] == mock_ctx.to_dict.return_value

    @pytest.mark.asyncio
    async def test_default_user_id(self):
        """user_id defaults to 'default'."""
        mock_ctx = _make_channel_context()
        mock_service = MagicMock()
        mock_service.get_user_context = AsyncMock(return_value=mock_ctx)

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            # Should not raise even without user_id
            result = await handle_get_context(email_address="x@y.com")

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_default_workspace_id(self):
        """workspace_id defaults to 'default'."""
        mock_ctx = _make_channel_context()
        mock_service = MagicMock()
        mock_service.get_user_context = AsyncMock(return_value=mock_ctx)

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_context(
                email_address="x@y.com",
                workspace_id="custom-ws",
            )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_custom_user_id_accepted(self):
        """Custom user_id is accepted without error."""
        mock_ctx = _make_channel_context()
        mock_service = MagicMock()
        mock_service.get_user_context = AsyncMock(return_value=mock_ctx)

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_context(
                email_address="x@y.com",
                user_id="custom-user",
            )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_context_dict_keys(self):
        """Result dict has exactly success and context keys."""
        mock_ctx = _make_channel_context()
        mock_service = MagicMock()
        mock_service.get_user_context = AsyncMock(return_value=mock_ctx)

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_context(email_address="x@y.com")

        assert set(result.keys()) == {"success", "context"}


# ============================================================================
# handle_get_context - Error Handling
# ============================================================================


class TestGetContextErrors:
    """Tests for error handling in handle_get_context."""

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """ConnectionError returns failure dict."""
        mock_service = MagicMock()
        mock_service.get_user_context = AsyncMock(
            side_effect=ConnectionError("cannot connect"),
        )

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_context(email_address="x@y.com")

        assert result["success"] is False
        assert result["error"] == "Failed to get context"

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        """TimeoutError returns failure dict."""
        mock_service = MagicMock()
        mock_service.get_user_context = AsyncMock(
            side_effect=TimeoutError("timed out"),
        )

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_context(email_address="x@y.com")

        assert result["success"] is False
        assert result["error"] == "Failed to get context"

    @pytest.mark.asyncio
    async def test_os_error(self):
        """OSError returns failure dict."""
        mock_service = MagicMock()
        mock_service.get_user_context = AsyncMock(
            side_effect=OSError("disk error"),
        )

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_context(email_address="x@y.com")

        assert result["success"] is False
        assert result["error"] == "Failed to get context"

    @pytest.mark.asyncio
    async def test_value_error(self):
        """ValueError returns failure dict."""
        mock_service = MagicMock()
        mock_service.get_user_context = AsyncMock(
            side_effect=ValueError("bad value"),
        )

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_context(email_address="x@y.com")

        assert result["success"] is False
        assert result["error"] == "Failed to get context"

    @pytest.mark.asyncio
    async def test_uncaught_exception_propagates(self):
        """Exceptions not in the caught set propagate normally."""
        mock_service = MagicMock()
        mock_service.get_user_context = AsyncMock(
            side_effect=RuntimeError("unexpected"),
        )

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            with pytest.raises(RuntimeError, match="unexpected"):
                await handle_get_context(email_address="x@y.com")

    @pytest.mark.asyncio
    async def test_error_dict_keys(self):
        """Error result dict has exactly success and error keys."""
        mock_service = MagicMock()
        mock_service.get_user_context = AsyncMock(
            side_effect=ConnectionError("fail"),
        )

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_context(email_address="x@y.com")

        assert set(result.keys()) == {"success", "error"}

    @pytest.mark.asyncio
    async def test_error_from_get_context_service(self):
        """Error raised by get_context_service itself is caught."""
        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            side_effect=ConnectionError("service init failed"),
        ):
            result = await handle_get_context(email_address="x@y.com")

        assert result["success"] is False
        assert result["error"] == "Failed to get context"

    @pytest.mark.asyncio
    async def test_error_from_to_dict(self):
        """Error raised by context.to_dict() is caught if in handled set."""
        mock_ctx = MagicMock()
        mock_ctx.to_dict.side_effect = ValueError("serialization error")
        mock_service = MagicMock()
        mock_service.get_user_context = AsyncMock(return_value=mock_ctx)

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_context(email_address="x@y.com")

        assert result["success"] is False
        assert result["error"] == "Failed to get context"


# ============================================================================
# handle_get_email_context_boost - Success Paths
# ============================================================================


class TestGetEmailContextBoostSuccess:
    """Tests for the happy path of handle_get_email_context_boost."""

    @pytest.mark.asyncio
    async def test_success_returns_boost_dict(self):
        """Returns success=True with boost field mapping."""
        mock_boost = _make_email_context_boost()
        mock_service = MagicMock()
        mock_service.get_email_context = AsyncMock(return_value=mock_boost)

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_email_context_boost(
                email_data=_full_email_data(),
            )

        assert result["success"] is True
        assert "boost" in result

    @pytest.mark.asyncio
    async def test_boost_email_id(self):
        """Boost response contains correct email_id."""
        mock_boost = _make_email_context_boost(email_id="msg-123")
        mock_service = MagicMock()
        mock_service.get_email_context = AsyncMock(return_value=mock_boost)

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_email_context_boost(
                email_data=_full_email_data(),
            )

        assert result["boost"]["email_id"] == "msg-123"

    @pytest.mark.asyncio
    async def test_boost_total_boost(self):
        """Boost response contains total_boost."""
        mock_boost = _make_email_context_boost(total_boost=0.35)
        mock_service = MagicMock()
        mock_service.get_email_context = AsyncMock(return_value=mock_boost)

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_email_context_boost(
                email_data=_full_email_data(),
            )

        assert result["boost"]["total_boost"] == 0.35

    @pytest.mark.asyncio
    async def test_boost_slack_fields(self):
        """Boost response contains Slack-related fields."""
        mock_boost = _make_email_context_boost(slack_activity_boost=0.25)
        mock_service = MagicMock()
        mock_service.get_email_context = AsyncMock(return_value=mock_boost)

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_email_context_boost(
                email_data=_full_email_data(),
            )

        boost = result["boost"]
        assert boost["slack_activity_boost"] == 0.25
        assert boost["slack_reason"] == "Sender active on Slack"
        assert boost["related_slack_channels"] == ["#general", "#eng"]

    @pytest.mark.asyncio
    async def test_boost_drive_fields(self):
        """Boost response contains Drive-related fields."""
        mock_boost = _make_email_context_boost(drive_relevance_boost=0.10)
        mock_service = MagicMock()
        mock_service.get_email_context = AsyncMock(return_value=mock_boost)

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_email_context_boost(
                email_data=_full_email_data(),
            )

        boost = result["boost"]
        assert boost["drive_relevance_boost"] == 0.10
        assert boost["drive_reason"] == "Related docs found"
        assert boost["related_drive_files"] == ["doc1.pdf"]

    @pytest.mark.asyncio
    async def test_boost_calendar_fields(self):
        """Boost response contains Calendar-related fields."""
        mock_boost = _make_email_context_boost(calendar_urgency_boost=0.10)
        mock_service = MagicMock()
        mock_service.get_email_context = AsyncMock(return_value=mock_boost)

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_email_context_boost(
                email_data=_full_email_data(),
            )

        boost = result["boost"]
        assert boost["calendar_urgency_boost"] == 0.10
        assert boost["calendar_reason"] == "Meeting in 30min"
        assert boost["related_meetings"] == ["standup"]

    @pytest.mark.asyncio
    async def test_boost_response_all_keys(self):
        """Boost dict contains exactly the expected set of keys."""
        mock_boost = _make_email_context_boost()
        mock_service = MagicMock()
        mock_service.get_email_context = AsyncMock(return_value=mock_boost)

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_email_context_boost(
                email_data=_full_email_data(),
            )

        expected_keys = {
            "email_id",
            "total_boost",
            "slack_activity_boost",
            "drive_relevance_boost",
            "calendar_urgency_boost",
            "slack_reason",
            "drive_reason",
            "calendar_reason",
            "related_slack_channels",
            "related_drive_files",
            "related_meetings",
        }
        assert set(result["boost"].keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_success_response_top_level_keys(self):
        """Success result dict has exactly success and boost keys."""
        mock_boost = _make_email_context_boost()
        mock_service = MagicMock()
        mock_service.get_email_context = AsyncMock(return_value=mock_boost)

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_email_context_boost(
                email_data=_full_email_data(),
            )

        assert set(result.keys()) == {"success", "boost"}


# ============================================================================
# handle_get_email_context_boost - EmailMessage Construction
# ============================================================================


class TestEmailMessageConstruction:
    """Tests for EmailMessage construction from email_data dict."""

    @pytest.mark.asyncio
    async def test_email_data_fields_passed_to_message(self):
        """Fields from email_data dict are mapped to EmailMessage."""
        mock_boost = _make_email_context_boost()
        mock_service = MagicMock()
        mock_service.get_email_context = AsyncMock(return_value=mock_boost)

        captured_email = []

        async def capture_email(email):
            captured_email.append(email)
            return mock_boost

        mock_service.get_email_context = capture_email

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            await handle_get_email_context_boost(
                email_data=_full_email_data(),
            )

        assert len(captured_email) == 1
        email = captured_email[0]
        assert email.id == "msg-123"
        assert email.thread_id == "thr-456"
        assert email.subject == "Q4 Review"
        assert email.from_address == "boss@company.com"
        assert email.to_addresses == ["me@company.com"]
        assert email.body_text == "Please review the Q4 numbers."
        assert email.snippet == "Please review..."

    @pytest.mark.asyncio
    async def test_missing_id_defaults_to_unknown(self):
        """Missing 'id' in email_data defaults to 'unknown'."""
        mock_boost = _make_email_context_boost()
        mock_service = MagicMock()
        captured_email = []

        async def capture_email(email):
            captured_email.append(email)
            return mock_boost

        mock_service.get_email_context = capture_email

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            await handle_get_email_context_boost(email_data={})

        assert captured_email[0].id == "unknown"

    @pytest.mark.asyncio
    async def test_missing_thread_id_defaults_to_unknown(self):
        """Missing 'thread_id' in email_data defaults to 'unknown'."""
        mock_boost = _make_email_context_boost()
        mock_service = MagicMock()
        captured_email = []

        async def capture_email(email):
            captured_email.append(email)
            return mock_boost

        mock_service.get_email_context = capture_email

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            await handle_get_email_context_boost(email_data={})

        assert captured_email[0].thread_id == "unknown"

    @pytest.mark.asyncio
    async def test_missing_subject_defaults_to_empty(self):
        """Missing 'subject' in email_data defaults to empty string."""
        mock_boost = _make_email_context_boost()
        mock_service = MagicMock()
        captured_email = []

        async def capture_email(email):
            captured_email.append(email)
            return mock_boost

        mock_service.get_email_context = capture_email

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            await handle_get_email_context_boost(email_data={})

        assert captured_email[0].subject == ""

    @pytest.mark.asyncio
    async def test_missing_from_address_defaults_to_empty(self):
        """Missing 'from_address' in email_data defaults to empty string."""
        mock_boost = _make_email_context_boost()
        mock_service = MagicMock()
        captured_email = []

        async def capture_email(email):
            captured_email.append(email)
            return mock_boost

        mock_service.get_email_context = capture_email

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            await handle_get_email_context_boost(email_data={})

        assert captured_email[0].from_address == ""

    @pytest.mark.asyncio
    async def test_missing_to_addresses_defaults_to_empty_list(self):
        """Missing 'to_addresses' in email_data defaults to empty list."""
        mock_boost = _make_email_context_boost()
        mock_service = MagicMock()
        captured_email = []

        async def capture_email(email):
            captured_email.append(email)
            return mock_boost

        mock_service.get_email_context = capture_email

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            await handle_get_email_context_boost(email_data={})

        assert captured_email[0].to_addresses == []

    @pytest.mark.asyncio
    async def test_missing_body_text_defaults_to_empty(self):
        """Missing 'body_text' in email_data defaults to empty string."""
        mock_boost = _make_email_context_boost()
        mock_service = MagicMock()
        captured_email = []

        async def capture_email(email):
            captured_email.append(email)
            return mock_boost

        mock_service.get_email_context = capture_email

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            await handle_get_email_context_boost(email_data={})

        assert captured_email[0].body_text == ""

    @pytest.mark.asyncio
    async def test_missing_snippet_defaults_to_empty(self):
        """Missing 'snippet' in email_data defaults to empty string."""
        mock_boost = _make_email_context_boost()
        mock_service = MagicMock()
        captured_email = []

        async def capture_email(email):
            captured_email.append(email)
            return mock_boost

        mock_service.get_email_context = capture_email

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            await handle_get_email_context_boost(email_data={})

        assert captured_email[0].snippet == ""

    @pytest.mark.asyncio
    async def test_hardcoded_defaults_in_email_message(self):
        """Verifies hardcoded defaults in EmailMessage construction."""
        mock_boost = _make_email_context_boost()
        mock_service = MagicMock()
        captured_email = []

        async def capture_email(email):
            captured_email.append(email)
            return mock_boost

        mock_service.get_email_context = capture_email

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            await handle_get_email_context_boost(email_data=_full_email_data())

        email = captured_email[0]
        assert email.cc_addresses == []
        assert email.bcc_addresses == []
        assert email.body_html == ""
        assert email.labels == []
        assert email.headers == {}
        assert email.attachments == []
        assert email.is_read is False
        assert email.is_starred is False
        assert email.is_important is False

    @pytest.mark.asyncio
    async def test_email_date_is_datetime(self):
        """EmailMessage date is set to a datetime."""
        from datetime import datetime

        mock_boost = _make_email_context_boost()
        mock_service = MagicMock()
        captured_email = []

        async def capture_email(email):
            captured_email.append(email)
            return mock_boost

        mock_service.get_email_context = capture_email

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            await handle_get_email_context_boost(email_data=_full_email_data())

        assert isinstance(captured_email[0].date, datetime)


# ============================================================================
# handle_get_email_context_boost - Error Handling
# ============================================================================


class TestGetEmailContextBoostErrors:
    """Tests for error handling in handle_get_email_context_boost."""

    @pytest.mark.asyncio
    async def test_type_error(self):
        """TypeError returns failure dict."""
        mock_service = MagicMock()
        mock_service.get_email_context = AsyncMock(
            side_effect=TypeError("wrong type"),
        )

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_email_context_boost(
                email_data=_full_email_data(),
            )

        assert result["success"] is False
        assert result["error"] == "Failed to get context boost"

    @pytest.mark.asyncio
    async def test_value_error(self):
        """ValueError returns failure dict."""
        mock_service = MagicMock()
        mock_service.get_email_context = AsyncMock(
            side_effect=ValueError("bad value"),
        )

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_email_context_boost(
                email_data=_full_email_data(),
            )

        assert result["success"] is False
        assert result["error"] == "Failed to get context boost"

    @pytest.mark.asyncio
    async def test_key_error(self):
        """KeyError returns failure dict."""
        mock_service = MagicMock()
        mock_service.get_email_context = AsyncMock(
            side_effect=KeyError("missing key"),
        )

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_email_context_boost(
                email_data=_full_email_data(),
            )

        assert result["success"] is False
        assert result["error"] == "Failed to get context boost"

    @pytest.mark.asyncio
    async def test_attribute_error(self):
        """AttributeError returns failure dict."""
        mock_service = MagicMock()
        mock_service.get_email_context = AsyncMock(
            side_effect=AttributeError("no such attr"),
        )

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_email_context_boost(
                email_data=_full_email_data(),
            )

        assert result["success"] is False
        assert result["error"] == "Failed to get context boost"

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """ConnectionError returns failure dict."""
        mock_service = MagicMock()
        mock_service.get_email_context = AsyncMock(
            side_effect=ConnectionError("no connection"),
        )

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_email_context_boost(
                email_data=_full_email_data(),
            )

        assert result["success"] is False
        assert result["error"] == "Failed to get context boost"

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        """TimeoutError returns failure dict."""
        mock_service = MagicMock()
        mock_service.get_email_context = AsyncMock(
            side_effect=TimeoutError("timed out"),
        )

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_email_context_boost(
                email_data=_full_email_data(),
            )

        assert result["success"] is False
        assert result["error"] == "Failed to get context boost"

    @pytest.mark.asyncio
    async def test_uncaught_exception_propagates(self):
        """Exceptions not in the caught set propagate normally."""
        mock_service = MagicMock()
        mock_service.get_email_context = AsyncMock(
            side_effect=RuntimeError("unexpected"),
        )

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            with pytest.raises(RuntimeError, match="unexpected"):
                await handle_get_email_context_boost(
                    email_data=_full_email_data(),
                )

    @pytest.mark.asyncio
    async def test_error_dict_keys(self):
        """Error result dict has exactly success and error keys."""
        mock_service = MagicMock()
        mock_service.get_email_context = AsyncMock(
            side_effect=TypeError("fail"),
        )

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_email_context_boost(
                email_data=_full_email_data(),
            )

        assert set(result.keys()) == {"success", "error"}

    @pytest.mark.asyncio
    async def test_error_from_get_context_service(self):
        """Error raised by get_context_service itself is caught."""
        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            side_effect=ConnectionError("service init failed"),
        ):
            result = await handle_get_email_context_boost(
                email_data=_full_email_data(),
            )

        assert result["success"] is False
        assert result["error"] == "Failed to get context boost"

    @pytest.mark.asyncio
    async def test_error_during_email_message_import(self):
        """Error during EmailMessage import is caught."""
        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            side_effect=AttributeError("import error"),
        ):
            result = await handle_get_email_context_boost(
                email_data=_full_email_data(),
            )

        assert result["success"] is False


# ============================================================================
# handle_get_email_context_boost - Default Parameters
# ============================================================================


class TestGetEmailContextBoostDefaults:
    """Tests for default parameter values on handle_get_email_context_boost."""

    @pytest.mark.asyncio
    async def test_default_user_id(self):
        """user_id defaults to 'default'."""
        mock_boost = _make_email_context_boost()
        mock_service = MagicMock()
        mock_service.get_email_context = AsyncMock(return_value=mock_boost)

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_email_context_boost(
                email_data=_full_email_data(),
            )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_default_workspace_id(self):
        """workspace_id defaults to 'default'."""
        mock_boost = _make_email_context_boost()
        mock_service = MagicMock()
        mock_service.get_email_context = AsyncMock(return_value=mock_boost)

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_email_context_boost(
                email_data=_full_email_data(),
                workspace_id="custom-ws",
            )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_custom_user_id_accepted(self):
        """Custom user_id is accepted without error."""
        mock_boost = _make_email_context_boost()
        mock_service = MagicMock()
        mock_service.get_email_context = AsyncMock(return_value=mock_boost)

        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_service,
        ):
            result = await handle_get_email_context_boost(
                email_data=_full_email_data(),
                user_id="custom-user",
            )

        assert result["success"] is True


# ============================================================================
# Permission Constants
# ============================================================================


class TestPermissionConstants:
    """Tests for module-level permission constants."""

    def test_perm_email_read_value(self):
        """PERM_EMAIL_READ is 'email:read'."""
        assert PERM_EMAIL_READ == "email:read"
