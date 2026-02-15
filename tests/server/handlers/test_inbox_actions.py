"""
Tests for InboxActionsMixin - email action execution methods.

Tests cover:
- _execute_action dispatch loop
- _perform_action routing to action handlers
- Individual action handlers (archive, snooze, reply, forward, spam, etc.)
- _sanitize_action_params
- _get_emails_by_filter
- Demo mode fallback when no Gmail connector
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.inbox_actions import InboxActionsMixin


# ===========================================================================
# Concrete Test Class
# ===========================================================================


class ConcreteInboxActions(InboxActionsMixin):
    """Concrete class combining InboxActionsMixin for testing."""

    def __init__(self, gmail_connector=None, prioritizer=None):
        self.gmail_connector = gmail_connector
        self.prioritizer = prioritizer


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _clear_email_cache():
    """Clear email cache between tests."""
    try:
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.clear()
    except (ImportError, AttributeError):
        pass
    yield
    try:
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.clear()
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def actions():
    """Create a ConcreteInboxActions with no connectors (demo mode)."""
    return ConcreteInboxActions()


@pytest.fixture
def actions_with_gmail():
    """Create a ConcreteInboxActions with a mock Gmail connector."""
    gmail = AsyncMock()
    return ConcreteInboxActions(gmail_connector=gmail)


# ===========================================================================
# _perform_action Routing
# ===========================================================================


class TestPerformAction:
    """Tests for _perform_action dispatching to correct handler."""

    @pytest.mark.asyncio
    async def test_archive_action(self, actions):
        result = await actions._perform_action("archive", "email-1", {})
        assert result["archived"] is True
        assert result.get("demo") is True  # No connector

    @pytest.mark.asyncio
    async def test_snooze_action(self, actions):
        result = await actions._perform_action("snooze", "email-1", {"duration": "1h"})
        assert result["snoozed"] is True
        assert "until" in result

    @pytest.mark.asyncio
    async def test_reply_action(self, actions):
        result = await actions._perform_action("reply", "email-1", {"body": "Thanks!"})
        assert "draftId" in result

    @pytest.mark.asyncio
    async def test_forward_action(self, actions):
        result = await actions._perform_action("forward", "email-1", {"to": "bob@test.com"})
        assert "draftId" in result

    @pytest.mark.asyncio
    async def test_spam_action(self, actions):
        result = await actions._perform_action("spam", "email-1", {})
        assert result["spam"] is True

    @pytest.mark.asyncio
    async def test_mark_important_action(self, actions):
        result = await actions._perform_action("mark_important", "email-1", {})
        assert result["important"] is True

    @pytest.mark.asyncio
    async def test_delete_action(self, actions):
        result = await actions._perform_action("delete", "email-1", {})
        assert result["deleted"] is True

    @pytest.mark.asyncio
    async def test_unknown_action_raises_value_error(self, actions):
        with pytest.raises(ValueError, match="Unknown action"):
            await actions._perform_action("explode", "email-1", {})


# ===========================================================================
# _execute_action Loop
# ===========================================================================


class TestExecuteAction:
    """Tests for _execute_action processing multiple emails."""

    @pytest.mark.asyncio
    async def test_processes_multiple_emails(self, actions):
        results = await actions._execute_action(
            "archive",
            ["email-1", "email-2"],
            {},
        )
        assert len(results) == 2
        assert all(r["success"] for r in results)

    @pytest.mark.asyncio
    async def test_handles_partial_failure(self, actions):
        # Patch one email to fail
        original = actions._perform_action

        call_count = 0

        async def side_effect(action, email_id, params):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("fail on first")
            return await original(action, email_id, params)

        actions._perform_action = side_effect

        results = await actions._execute_action(
            "archive",
            ["email-1", "email-2"],
            {},
        )
        assert len(results) == 2
        assert results[0]["success"] is False
        assert "error" in results[0]
        assert results[1]["success"] is True

    @pytest.mark.asyncio
    async def test_records_action_with_prioritizer(self):
        prioritizer = AsyncMock()
        prioritizer.record_user_action = AsyncMock()
        svc = ConcreteInboxActions(prioritizer=prioritizer)

        await svc._execute_action("archive", ["email-1"], {})
        prioritizer.record_user_action.assert_called_once_with(
            email_id="email-1",
            action="archive",
            email=None,
        )


# ===========================================================================
# Individual Action Handlers with Gmail Connector
# ===========================================================================


class TestActionsWithGmailConnector:
    """Tests for actions when Gmail connector is available."""

    @pytest.mark.asyncio
    async def test_archive_via_gmail(self, actions_with_gmail):
        actions_with_gmail.gmail_connector.archive_message = AsyncMock()
        result = await actions_with_gmail._archive_email("email-1", {})
        assert result["archived"] is True
        assert "demo" not in result

    @pytest.mark.asyncio
    async def test_archive_falls_back_on_gmail_error(self, actions_with_gmail):
        actions_with_gmail.gmail_connector.archive_message = AsyncMock(
            side_effect=ConnectionError("timeout")
        )
        result = await actions_with_gmail._archive_email("email-1", {})
        assert result["archived"] is True
        assert result.get("demo") is True

    @pytest.mark.asyncio
    async def test_spam_via_gmail(self, actions_with_gmail):
        actions_with_gmail.gmail_connector.mark_spam = AsyncMock()
        result = await actions_with_gmail._mark_spam("email-1", {})
        assert result["spam"] is True
        assert "demo" not in result

    @pytest.mark.asyncio
    async def test_mark_important_via_gmail(self, actions_with_gmail):
        actions_with_gmail.gmail_connector.modify_labels = AsyncMock()
        result = await actions_with_gmail._mark_important("email-1", {})
        assert result["important"] is True
        assert "demo" not in result

    @pytest.mark.asyncio
    async def test_delete_via_gmail(self, actions_with_gmail):
        actions_with_gmail.gmail_connector.trash_message = AsyncMock()
        result = await actions_with_gmail._delete_email("email-1", {})
        assert result["deleted"] is True
        assert "demo" not in result


# ===========================================================================
# Snooze Duration Parsing
# ===========================================================================


class TestSnoozeDuration:
    """Tests for snooze duration handling."""

    @pytest.mark.asyncio
    async def test_snooze_1h(self, actions):
        result = await actions._snooze_email("email-1", {"duration": "1h"})
        assert result["snoozed"] is True

    @pytest.mark.asyncio
    async def test_snooze_3d(self, actions):
        result = await actions._snooze_email("email-1", {"duration": "3d"})
        assert result["snoozed"] is True

    @pytest.mark.asyncio
    async def test_snooze_1w(self, actions):
        result = await actions._snooze_email("email-1", {"duration": "1w"})
        assert result["snoozed"] is True

    @pytest.mark.asyncio
    async def test_snooze_default_duration(self, actions):
        result = await actions._snooze_email("email-1", {})
        assert result["snoozed"] is True  # Should default to 1d


# ===========================================================================
# Reply and Forward Drafts
# ===========================================================================


class TestDrafts:
    """Tests for reply and forward draft creation."""

    @pytest.mark.asyncio
    async def test_reply_draft_via_gmail(self, actions_with_gmail):
        actions_with_gmail.gmail_connector.create_draft = AsyncMock(return_value="draft-123")
        result = await actions_with_gmail._create_reply_draft("email-1", {"body": "Thanks!"})
        assert result["draftId"] == "draft-123"

    @pytest.mark.asyncio
    async def test_reply_draft_demo_mode(self, actions):
        result = await actions._create_reply_draft("email-1", {"body": "Thanks!"})
        assert "draftId" in result
        assert result.get("demo") is True

    @pytest.mark.asyncio
    async def test_forward_draft_via_gmail(self, actions_with_gmail):
        actions_with_gmail.gmail_connector.create_forward_draft = AsyncMock(
            return_value="draft-fwd-123"
        )
        result = await actions_with_gmail._create_forward_draft("email-1", {"to": "bob@test.com"})
        assert result["draftId"] == "draft-fwd-123"

    @pytest.mark.asyncio
    async def test_forward_draft_demo_mode(self, actions):
        result = await actions._create_forward_draft("email-1", {"to": "bob@test.com"})
        assert "draftId" in result
        assert result.get("demo") is True


# ===========================================================================
# VIP and Block
# ===========================================================================


class TestVipAndBlock:
    """Tests for mark_vip and block_sender."""

    @pytest.mark.asyncio
    async def test_mark_vip_with_prioritizer(self):
        prioritizer = MagicMock()
        prioritizer.config.vip_addresses = set()
        svc = ConcreteInboxActions(prioritizer=prioritizer)

        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache["email-1"] = {"from": "vip@company.com"}

        result = await svc._mark_sender_vip("email-1", {})
        assert result["vip"] is True
        assert "vip@company.com" in prioritizer.config.vip_addresses

    @pytest.mark.asyncio
    async def test_block_sender_with_prioritizer(self):
        prioritizer = MagicMock()
        prioritizer.config.auto_archive_senders = set()
        svc = ConcreteInboxActions(prioritizer=prioritizer)

        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache["email-1"] = {"from": "spam@example.com"}

        result = await svc._block_sender("email-1", {})
        assert result["blocked"] is True
        assert "spam@example.com" in prioritizer.config.auto_archive_senders


# ===========================================================================
# _sanitize_action_params
# ===========================================================================


class TestSanitizeActionParams:
    """Tests for _sanitize_action_params."""

    def test_snooze_valid_duration(self, actions):
        params = actions._sanitize_action_params("snooze", {"duration": "1d"})
        assert params["duration"] == "1d"

    def test_snooze_invalid_duration_defaults(self, actions):
        params = actions._sanitize_action_params("snooze", {"duration": "99y"})
        assert params["duration"] == "1d"  # Safe default

    def test_archive_returns_empty(self, actions):
        params = actions._sanitize_action_params("archive", {"anything": "value"})
        assert params == {}


# ===========================================================================
# _get_emails_by_filter
# ===========================================================================


class TestGetEmailsByFilter:
    """Tests for _get_emails_by_filter."""

    @pytest.mark.asyncio
    async def test_invalid_filter_returns_empty(self, actions):
        result = await actions._get_emails_by_filter("nonexistent_filter")
        assert result == []

    @pytest.mark.asyncio
    async def test_all_filter_returns_all(self, actions):
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache["e1"] = {"priority": "high", "unread": True}
        _email_cache["e2"] = {"priority": "low", "unread": False}

        result = await actions._get_emails_by_filter("all")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_low_filter(self, actions):
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache["e1"] = {"priority": "low", "unread": True}
        _email_cache["e2"] = {"priority": "high", "unread": True}
        _email_cache["e3"] = {"priority": "defer", "unread": True}

        result = await actions._get_emails_by_filter("low")
        assert "e1" in result
        assert "e3" in result  # defer is in ["low", "defer"]
        assert "e2" not in result
