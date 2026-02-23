"""Tests for InboxActionsMixin (aragora/server/handlers/inbox_actions.py).

Covers all methods of the InboxActionsMixin class:
- _execute_action:          Dispatch action to handler + record in prioritizer
- _sanitize_action_params:  Validate/sanitize action-specific parameters
- _perform_action:          Dispatch to individual action handler by name
- _archive_email:           Archive via Gmail or demo mode
- _snooze_email:            Snooze with duration mapping
- _create_reply_draft:      Reply draft via Gmail or demo mode
- _create_forward_draft:    Forward draft via Gmail or demo mode
- _mark_spam:               Mark as spam via Gmail or demo mode
- _mark_important:          Mark important via Gmail or demo mode
- _mark_sender_vip:         Mark sender as VIP via prioritizer config
- _block_sender:            Block sender via prioritizer config
- _delete_email:            Delete via Gmail or demo mode
- _get_emails_by_filter:    Filter email IDs from cache by type

Test categories:
- Happy path for each action (Gmail mode and demo/fallback mode)
- Gmail connector error handling (graceful fallback)
- Parameter sanitization (valid, invalid, edge cases)
- _execute_action orchestration (success, failure, prioritizer recording)
- _perform_action dispatch (known actions, unknown action)
- _get_emails_by_filter (all filter types, empty cache, disallowed filters)
- VIP / block actions with email cache and params
- Security tests (injection, oversized input)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.inbox_actions import InboxActionsMixin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ConcreteInboxActions(InboxActionsMixin):
    """Concrete class for testing the mixin."""

    def __init__(
        self,
        gmail_connector: Any = None,
        prioritizer: Any = None,
    ):
        self.gmail_connector = gmail_connector
        self.prioritizer = prioritizer


def _make_gmail_connector(**methods: Any) -> MagicMock:
    """Build a mock GmailConnector with specified methods."""
    connector = MagicMock()
    for name, impl in methods.items():
        setattr(connector, name, impl)
    return connector


def _make_prioritizer() -> MagicMock:
    """Build a mock EmailPrioritizer with config and record_user_action."""
    p = MagicMock()
    p.record_user_action = AsyncMock()
    p.config = MagicMock()
    p.config.vip_addresses = set()
    p.config.auto_archive_senders = set()
    return p


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_email_cache():
    """Clear the module-level email cache between tests."""
    from aragora.server.handlers.inbox_command import _email_cache

    _email_cache.clear()
    yield
    _email_cache.clear()


@pytest.fixture
def gmail():
    """Gmail connector mock with all relevant methods."""
    return _make_gmail_connector(
        archive_message=AsyncMock(),
        snooze_message=AsyncMock(),
        create_draft=AsyncMock(return_value="draft-001"),
        create_forward_draft=AsyncMock(return_value="draft-fwd-001"),
        mark_spam=AsyncMock(),
        modify_labels=AsyncMock(),
        trash_message=AsyncMock(),
    )


@pytest.fixture
def prioritizer():
    return _make_prioritizer()


@pytest.fixture
def handler(gmail, prioritizer):
    """InboxActions handler with gmail and prioritizer."""
    return ConcreteInboxActions(gmail_connector=gmail, prioritizer=prioritizer)


@pytest.fixture
def demo_handler():
    """InboxActions handler with no gmail connector (demo mode)."""
    return ConcreteInboxActions(gmail_connector=None, prioritizer=None)


# ============================================================================
# _perform_action dispatch
# ============================================================================


class TestPerformAction:
    """Tests for _perform_action dispatch."""

    @pytest.mark.asyncio
    async def test_archive_dispatches(self, handler):
        result = await handler._perform_action("archive", "e1", {})
        assert result["archived"] is True

    @pytest.mark.asyncio
    async def test_snooze_dispatches(self, handler):
        result = await handler._perform_action("snooze", "e1", {"duration": "1h"})
        assert result["snoozed"] is True

    @pytest.mark.asyncio
    async def test_reply_dispatches(self, handler):
        result = await handler._perform_action("reply", "e1", {"body": "hi"})
        assert "draftId" in result

    @pytest.mark.asyncio
    async def test_forward_dispatches(self, handler):
        result = await handler._perform_action("forward", "e1", {"to": "a@b.com"})
        assert "draftId" in result

    @pytest.mark.asyncio
    async def test_spam_dispatches(self, handler):
        result = await handler._perform_action("spam", "e1", {})
        assert result["spam"] is True

    @pytest.mark.asyncio
    async def test_mark_important_dispatches(self, handler):
        result = await handler._perform_action("mark_important", "e1", {})
        assert result["important"] is True

    @pytest.mark.asyncio
    async def test_mark_vip_dispatches(self, handler):
        result = await handler._perform_action("mark_vip", "e1", {"sender": "x@y.com"})
        assert result.get("vip") is True or result.get("demo") is True

    @pytest.mark.asyncio
    async def test_block_dispatches(self, handler):
        result = await handler._perform_action("block", "e1", {"sender": "x@y.com"})
        assert result.get("blocked") is True or result.get("demo") is True

    @pytest.mark.asyncio
    async def test_delete_dispatches(self, handler):
        result = await handler._perform_action("delete", "e1", {})
        assert result["deleted"] is True

    @pytest.mark.asyncio
    async def test_unknown_action_raises(self, handler):
        with pytest.raises(ValueError, match="Unknown action"):
            await handler._perform_action("nonexistent", "e1", {})

    @pytest.mark.asyncio
    async def test_unknown_action_with_injection_attempt(self, handler):
        with pytest.raises(ValueError):
            await handler._perform_action("archive; rm -rf /", "e1", {})


# ============================================================================
# _archive_email
# ============================================================================


class TestArchiveEmail:
    """Tests for _archive_email."""

    @pytest.mark.asyncio
    async def test_archive_via_gmail(self, handler, gmail):
        result = await handler._archive_email("e1", {})
        assert result == {"archived": True}
        gmail.archive_message.assert_awaited_once_with("e1")

    @pytest.mark.asyncio
    async def test_archive_demo_mode(self, demo_handler):
        result = await demo_handler._archive_email("e1", {})
        assert result["archived"] is True
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_archive_gmail_oserror_falls_back(self, handler, gmail):
        gmail.archive_message = AsyncMock(side_effect=OSError("fail"))
        result = await handler._archive_email("e1", {})
        assert result["archived"] is True
        assert result.get("demo") is True

    @pytest.mark.asyncio
    async def test_archive_gmail_connection_error_falls_back(self, handler, gmail):
        gmail.archive_message = AsyncMock(side_effect=ConnectionError("timeout"))
        result = await handler._archive_email("e1", {})
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_archive_gmail_runtime_error_falls_back(self, handler, gmail):
        gmail.archive_message = AsyncMock(side_effect=RuntimeError("bad"))
        result = await handler._archive_email("e1", {})
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_archive_gmail_attribute_error_falls_back(self, handler, gmail):
        gmail.archive_message = AsyncMock(side_effect=AttributeError("missing"))
        result = await handler._archive_email("e1", {})
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_archive_no_archive_method_demo_mode(self):
        """Gmail connector exists but has no archive_message method."""
        connector = MagicMock(spec=[])
        h = ConcreteInboxActions(gmail_connector=connector)
        result = await h._archive_email("e1", {})
        assert result["demo"] is True


# ============================================================================
# _snooze_email
# ============================================================================


class TestSnoozeEmail:
    """Tests for _snooze_email."""

    @pytest.mark.asyncio
    async def test_snooze_via_gmail_1h(self, handler, gmail):
        result = await handler._snooze_email("e1", {"duration": "1h"})
        assert result["snoozed"] is True
        assert "until" in result
        gmail.snooze_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_snooze_via_gmail_3h(self, handler, gmail):
        result = await handler._snooze_email("e1", {"duration": "3h"})
        assert result["snoozed"] is True

    @pytest.mark.asyncio
    async def test_snooze_via_gmail_1d(self, handler, gmail):
        result = await handler._snooze_email("e1", {"duration": "1d"})
        assert result["snoozed"] is True

    @pytest.mark.asyncio
    async def test_snooze_via_gmail_3d(self, handler, gmail):
        result = await handler._snooze_email("e1", {"duration": "3d"})
        assert result["snoozed"] is True

    @pytest.mark.asyncio
    async def test_snooze_via_gmail_1w(self, handler, gmail):
        result = await handler._snooze_email("e1", {"duration": "1w"})
        assert result["snoozed"] is True

    @pytest.mark.asyncio
    async def test_snooze_default_duration(self, handler, gmail):
        """No duration param defaults to 1d."""
        result = await handler._snooze_email("e1", {})
        assert result["snoozed"] is True
        # Default is 1d — snooze_until should be ~24h from now
        until = datetime.fromisoformat(result["until"])
        now = datetime.now(timezone.utc)
        diff = until - now
        assert timedelta(hours=23) < diff < timedelta(hours=25)

    @pytest.mark.asyncio
    async def test_snooze_unknown_duration_defaults_1d(self, handler, gmail):
        result = await handler._snooze_email("e1", {"duration": "99y"})
        assert result["snoozed"] is True
        until = datetime.fromisoformat(result["until"])
        now = datetime.now(timezone.utc)
        diff = until - now
        assert timedelta(hours=23) < diff < timedelta(hours=25)

    @pytest.mark.asyncio
    async def test_snooze_demo_mode(self, demo_handler):
        result = await demo_handler._snooze_email("e1", {"duration": "1h"})
        assert result["snoozed"] is True
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_snooze_gmail_error_falls_back(self, handler, gmail):
        gmail.snooze_message = AsyncMock(side_effect=OSError("fail"))
        result = await handler._snooze_email("e1", {"duration": "1h"})
        assert result["snoozed"] is True
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_snooze_no_method_demo_mode(self):
        connector = MagicMock(spec=[])
        h = ConcreteInboxActions(gmail_connector=connector)
        result = await h._snooze_email("e1", {"duration": "3h"})
        assert result["demo"] is True


# ============================================================================
# _create_reply_draft
# ============================================================================


class TestCreateReplyDraft:
    """Tests for _create_reply_draft."""

    @pytest.mark.asyncio
    async def test_reply_via_gmail(self, handler, gmail):
        result = await handler._create_reply_draft("e1", {"body": "Thanks!"})
        assert result["draftId"] == "draft-001"
        gmail.create_draft.assert_awaited_once_with(in_reply_to="e1", body="Thanks!")

    @pytest.mark.asyncio
    async def test_reply_demo_mode(self, demo_handler):
        result = await demo_handler._create_reply_draft("e1", {"body": "hi"})
        assert result["draftId"] == "draft_e1"
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_reply_empty_body(self, handler, gmail):
        result = await handler._create_reply_draft("e1", {})
        assert result["draftId"] == "draft-001"
        gmail.create_draft.assert_awaited_once_with(in_reply_to="e1", body="")

    @pytest.mark.asyncio
    async def test_reply_gmail_error_falls_back(self, handler, gmail):
        gmail.create_draft = AsyncMock(side_effect=ConnectionError("net"))
        result = await handler._create_reply_draft("e1", {"body": "test"})
        assert result["draftId"] == "draft_e1"
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_reply_no_create_draft_method(self):
        connector = MagicMock(spec=[])
        h = ConcreteInboxActions(gmail_connector=connector)
        result = await h._create_reply_draft("e1", {"body": "hi"})
        assert result["demo"] is True


# ============================================================================
# _create_forward_draft
# ============================================================================


class TestCreateForwardDraft:
    """Tests for _create_forward_draft."""

    @pytest.mark.asyncio
    async def test_forward_via_gmail(self, handler, gmail):
        result = await handler._create_forward_draft("e1", {"to": "fwd@x.com"})
        assert result["draftId"] == "draft-fwd-001"
        gmail.create_forward_draft.assert_awaited_once_with(message_id="e1", to="fwd@x.com")

    @pytest.mark.asyncio
    async def test_forward_demo_mode(self, demo_handler):
        result = await demo_handler._create_forward_draft("e1", {"to": "fwd@x.com"})
        assert result["draftId"] == "draft_fwd_e1"
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_forward_gmail_error_falls_back(self, handler, gmail):
        gmail.create_forward_draft = AsyncMock(side_effect=RuntimeError("bad"))
        result = await handler._create_forward_draft("e1", {"to": "a@b.com"})
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_forward_no_method_demo_mode(self):
        connector = MagicMock(spec=[])
        h = ConcreteInboxActions(gmail_connector=connector)
        result = await h._create_forward_draft("e1", {"to": "x@y.com"})
        assert result["demo"] is True


# ============================================================================
# _mark_spam
# ============================================================================


class TestMarkSpam:
    """Tests for _mark_spam."""

    @pytest.mark.asyncio
    async def test_spam_via_gmail(self, handler, gmail):
        result = await handler._mark_spam("e1", {})
        assert result == {"spam": True}
        gmail.mark_spam.assert_awaited_once_with("e1")

    @pytest.mark.asyncio
    async def test_spam_demo_mode(self, demo_handler):
        result = await demo_handler._mark_spam("e1", {})
        assert result["spam"] is True
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_spam_gmail_error_falls_back(self, handler, gmail):
        gmail.mark_spam = AsyncMock(side_effect=OSError("net"))
        result = await handler._mark_spam("e1", {})
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_spam_no_method_demo_mode(self):
        connector = MagicMock(spec=[])
        h = ConcreteInboxActions(gmail_connector=connector)
        result = await h._mark_spam("e1", {})
        assert result["demo"] is True


# ============================================================================
# _mark_important
# ============================================================================


class TestMarkImportant:
    """Tests for _mark_important."""

    @pytest.mark.asyncio
    async def test_important_via_gmail(self, handler, gmail):
        result = await handler._mark_important("e1", {})
        assert result == {"important": True}
        gmail.modify_labels.assert_awaited_once_with("e1", add_labels=["IMPORTANT"])

    @pytest.mark.asyncio
    async def test_important_demo_mode(self, demo_handler):
        result = await demo_handler._mark_important("e1", {})
        assert result["important"] is True
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_important_gmail_error_falls_back(self, handler, gmail):
        gmail.modify_labels = AsyncMock(side_effect=AttributeError("fail"))
        result = await handler._mark_important("e1", {})
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_important_no_method_demo_mode(self):
        connector = MagicMock(spec=[])
        h = ConcreteInboxActions(gmail_connector=connector)
        result = await h._mark_important("e1", {})
        assert result["demo"] is True


# ============================================================================
# _mark_sender_vip
# ============================================================================


class TestMarkSenderVip:
    """Tests for _mark_sender_vip."""

    @pytest.mark.asyncio
    async def test_vip_from_cache(self, handler, prioritizer):
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"from": "boss@corp.com", "subject": "hi"})
        result = await handler._mark_sender_vip("e1", {})
        assert result["vip"] is True
        assert result["sender"] == "boss@corp.com"
        assert "boss@corp.com" in prioritizer.config.vip_addresses

    @pytest.mark.asyncio
    async def test_vip_from_params(self, handler, prioritizer):
        result = await handler._mark_sender_vip("e1", {"sender": "alice@corp.com"})
        assert result["vip"] is True
        assert result["sender"] == "alice@corp.com"
        assert "alice@corp.com" in prioritizer.config.vip_addresses

    @pytest.mark.asyncio
    async def test_vip_cache_takes_priority_over_params(self, handler, prioritizer):
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"from": "cached@corp.com"})
        result = await handler._mark_sender_vip("e1", {"sender": "param@corp.com"})
        assert result["sender"] == "cached@corp.com"

    @pytest.mark.asyncio
    async def test_vip_demo_when_no_sender(self, handler):
        """No cache entry and no sender param."""
        result = await handler._mark_sender_vip("e1", {})
        assert result["vip"] is True
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_vip_demo_when_no_prioritizer(self, demo_handler):
        result = await demo_handler._mark_sender_vip("e1", {"sender": "a@b.com"})
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_vip_sender_none_no_prioritizer(self):
        h = ConcreteInboxActions(gmail_connector=None, prioritizer=None)
        result = await h._mark_sender_vip("e1", {})
        assert result["demo"] is True


# ============================================================================
# _block_sender
# ============================================================================


class TestBlockSender:
    """Tests for _block_sender."""

    @pytest.mark.asyncio
    async def test_block_from_cache(self, handler, prioritizer):
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"from": "spam@evil.com"})
        result = await handler._block_sender("e1", {})
        assert result["blocked"] is True
        assert result["sender"] == "spam@evil.com"
        assert "spam@evil.com" in prioritizer.config.auto_archive_senders

    @pytest.mark.asyncio
    async def test_block_from_params(self, handler, prioritizer):
        result = await handler._block_sender("e1", {"sender": "junk@evil.com"})
        assert result["blocked"] is True
        assert result["sender"] == "junk@evil.com"

    @pytest.mark.asyncio
    async def test_block_cache_priority(self, handler, prioritizer):
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"from": "cached@evil.com"})
        result = await handler._block_sender("e1", {"sender": "param@evil.com"})
        assert result["sender"] == "cached@evil.com"

    @pytest.mark.asyncio
    async def test_block_demo_when_no_sender(self, handler):
        result = await handler._block_sender("e1", {})
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_block_demo_when_no_prioritizer(self):
        h = ConcreteInboxActions(gmail_connector=None, prioritizer=None)
        result = await h._block_sender("e1", {"sender": "a@b.com"})
        assert result["demo"] is True


# ============================================================================
# _delete_email
# ============================================================================


class TestDeleteEmail:
    """Tests for _delete_email."""

    @pytest.mark.asyncio
    async def test_delete_via_gmail(self, handler, gmail):
        result = await handler._delete_email("e1", {})
        assert result == {"deleted": True}
        gmail.trash_message.assert_awaited_once_with("e1")

    @pytest.mark.asyncio
    async def test_delete_demo_mode(self, demo_handler):
        result = await demo_handler._delete_email("e1", {})
        assert result["deleted"] is True
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_delete_gmail_error_falls_back(self, handler, gmail):
        gmail.trash_message = AsyncMock(side_effect=RuntimeError("err"))
        result = await handler._delete_email("e1", {})
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_delete_no_method_demo_mode(self):
        connector = MagicMock(spec=[])
        h = ConcreteInboxActions(gmail_connector=connector)
        result = await h._delete_email("e1", {})
        assert result["demo"] is True


# ============================================================================
# _execute_action
# ============================================================================


class TestExecuteAction:
    """Tests for _execute_action orchestration."""

    @pytest.mark.asyncio
    async def test_single_success(self, handler, prioritizer):
        results = await handler._execute_action("archive", ["e1"], {})
        assert len(results) == 1
        assert results[0]["emailId"] == "e1"
        assert results[0]["success"] is True
        assert results[0]["result"]["archived"] is True

    @pytest.mark.asyncio
    async def test_multiple_emails(self, handler):
        results = await handler._execute_action("archive", ["e1", "e2", "e3"], {})
        assert len(results) == 3
        assert all(r["success"] for r in results)

    @pytest.mark.asyncio
    async def test_records_user_action_when_prioritizer(self, handler, prioritizer):
        await handler._execute_action("archive", ["e1"], {})
        prioritizer.record_user_action.assert_awaited_once_with(
            email_id="e1", action="archive", email=None
        )

    @pytest.mark.asyncio
    async def test_no_record_when_no_prioritizer(self, demo_handler):
        results = await demo_handler._execute_action("archive", ["e1"], {})
        assert len(results) == 1
        assert results[0]["success"] is True

    @pytest.mark.asyncio
    async def test_action_failure_recorded(self, handler, gmail):
        """When _perform_action raises, error is captured per-email."""
        gmail.archive_message = AsyncMock(side_effect=ValueError("bad email"))
        # Because archive catches specific errors and falls back to demo,
        # but _execute_action catches errors from _perform_action. We need
        # an action that raises. Use unknown action via monkey-patch.
        # Actually, ValueError from _perform_action for unknown action.
        results = await handler._execute_action("nonexistent", ["e1"], {})
        assert len(results) == 1
        assert results[0]["success"] is False
        assert results[0]["error"] == "Action failed"

    @pytest.mark.asyncio
    async def test_partial_failure(self, handler, gmail):
        """One email succeeds, another fails."""
        call_count = 0
        original = handler._perform_action

        async def flaky_perform(action, email_id, params):
            nonlocal call_count
            call_count += 1
            if email_id == "e2":
                raise RuntimeError("flaky failure")
            return await original(action, email_id, params)

        handler._perform_action = flaky_perform
        results = await handler._execute_action("archive", ["e1", "e2", "e3"], {})
        assert len(results) == 3
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert results[2]["success"] is True

    @pytest.mark.asyncio
    async def test_empty_email_ids(self, handler):
        results = await handler._execute_action("archive", [], {})
        assert results == []

    @pytest.mark.asyncio
    async def test_key_error_caught(self, handler):
        """KeyError is in the caught exception tuple."""
        original = handler._perform_action

        async def raise_key_error(action, email_id, params):
            raise KeyError("missing")

        handler._perform_action = raise_key_error
        results = await handler._execute_action("archive", ["e1"], {})
        assert results[0]["success"] is False

    @pytest.mark.asyncio
    async def test_type_error_caught(self, handler):
        original = handler._perform_action

        async def raise_type_error(action, email_id, params):
            raise TypeError("bad type")

        handler._perform_action = raise_type_error
        results = await handler._execute_action("archive", ["e1"], {})
        assert results[0]["success"] is False

    @pytest.mark.asyncio
    async def test_attribute_error_caught(self, handler):
        original = handler._perform_action

        async def raise_attr_error(action, email_id, params):
            raise AttributeError("missing")

        handler._perform_action = raise_attr_error
        results = await handler._execute_action("archive", ["e1"], {})
        assert results[0]["success"] is False

    @pytest.mark.asyncio
    async def test_os_error_caught(self, handler):
        original = handler._perform_action

        async def raise_os_error(action, email_id, params):
            raise OSError("disk")

        handler._perform_action = raise_os_error
        results = await handler._execute_action("archive", ["e1"], {})
        assert results[0]["success"] is False

    @pytest.mark.asyncio
    async def test_connection_error_caught(self, handler):
        original = handler._perform_action

        async def raise_conn_error(action, email_id, params):
            raise ConnectionError("timeout")

        handler._perform_action = raise_conn_error
        results = await handler._execute_action("archive", ["e1"], {})
        assert results[0]["success"] is False


# ============================================================================
# _sanitize_action_params
# ============================================================================


class TestSanitizeActionParams:
    """Tests for _sanitize_action_params."""

    def test_snooze_valid_duration(self, handler):
        result = handler._sanitize_action_params("snooze", {"duration": "3h"})
        assert result == {"duration": "3h"}

    def test_snooze_valid_durations(self, handler):
        for d in ("1h", "3h", "1d", "3d", "1w"):
            result = handler._sanitize_action_params("snooze", {"duration": d})
            assert result["duration"] == d

    def test_snooze_invalid_duration_defaults(self, handler):
        result = handler._sanitize_action_params("snooze", {"duration": "99y"})
        assert result["duration"] == "1d"

    def test_snooze_missing_duration_defaults(self, handler):
        result = handler._sanitize_action_params("snooze", {})
        assert result["duration"] == "1d"

    def test_snooze_non_string_duration_defaults(self, handler):
        result = handler._sanitize_action_params("snooze", {"duration": 42})
        assert result["duration"] == "1d"

    def test_snooze_strips_whitespace(self, handler):
        result = handler._sanitize_action_params("snooze", {"duration": "  1h  "})
        assert result["duration"] == "1h"

    def test_reply_sanitizes_body(self, handler):
        result = handler._sanitize_action_params("reply", {"body": "  hello  "})
        assert result["body"] == "hello"

    def test_reply_empty_body(self, handler):
        result = handler._sanitize_action_params("reply", {"body": ""})
        assert result["body"] == ""

    def test_reply_non_string_body(self, handler):
        result = handler._sanitize_action_params("reply", {"body": 123})
        assert result["body"] == ""

    def test_reply_truncates_long_body(self, handler):
        long_body = "x" * 200_000
        result = handler._sanitize_action_params("reply", {"body": long_body})
        assert len(result["body"]) <= 100_000

    def test_forward_valid_email(self, handler):
        result = handler._sanitize_action_params("forward", {"to": "user@example.com"})
        assert result["to"] == "user@example.com"

    def test_forward_invalid_email(self, handler):
        result = handler._sanitize_action_params("forward", {"to": "not-an-email"})
        assert result["to"] == ""

    def test_forward_empty_to(self, handler):
        result = handler._sanitize_action_params("forward", {"to": ""})
        assert result["to"] == ""

    def test_forward_non_string_to(self, handler):
        result = handler._sanitize_action_params("forward", {"to": 42})
        assert result["to"] == ""

    def test_mark_vip_valid_sender(self, handler):
        result = handler._sanitize_action_params("mark_vip", {"sender": "x@y.com"})
        assert result["sender"] == "x@y.com"

    def test_mark_vip_invalid_sender(self, handler):
        result = handler._sanitize_action_params("mark_vip", {"sender": "bad"})
        assert "sender" not in result

    def test_mark_vip_empty_sender(self, handler):
        result = handler._sanitize_action_params("mark_vip", {"sender": ""})
        assert "sender" not in result

    def test_mark_vip_no_sender(self, handler):
        result = handler._sanitize_action_params("mark_vip", {})
        assert result == {}

    def test_block_valid_sender(self, handler):
        result = handler._sanitize_action_params("block", {"sender": "x@y.com"})
        assert result["sender"] == "x@y.com"

    def test_block_invalid_sender(self, handler):
        result = handler._sanitize_action_params("block", {"sender": "bad"})
        assert "sender" not in result

    def test_archive_returns_empty(self, handler):
        result = handler._sanitize_action_params("archive", {"extra": "value"})
        assert result == {}

    def test_spam_returns_empty(self, handler):
        result = handler._sanitize_action_params("spam", {"extra": "value"})
        assert result == {}

    def test_delete_returns_empty(self, handler):
        result = handler._sanitize_action_params("delete", {"extra": "value"})
        assert result == {}

    def test_mark_important_returns_empty(self, handler):
        result = handler._sanitize_action_params("mark_important", {"extra": "value"})
        assert result == {}

    def test_unknown_action_returns_empty(self, handler):
        result = handler._sanitize_action_params("unknown", {"key": "val"})
        assert result == {}


# ============================================================================
# _get_emails_by_filter
# ============================================================================


class TestGetEmailsByFilter:
    """Tests for _get_emails_by_filter."""

    @pytest.mark.asyncio
    async def test_filter_all(self, handler):
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"priority": "high"})
        _email_cache.set("e2", {"priority": "low"})
        ids = await handler._get_emails_by_filter("all")
        assert set(ids) == {"e1", "e2"}

    @pytest.mark.asyncio
    async def test_filter_low(self, handler):
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"priority": "low"})
        _email_cache.set("e2", {"priority": "defer"})
        _email_cache.set("e3", {"priority": "high"})
        ids = await handler._get_emails_by_filter("low")
        assert set(ids) == {"e1", "e2"}

    @pytest.mark.asyncio
    async def test_filter_deferred(self, handler):
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"priority": "defer"})
        _email_cache.set("e2", {"priority": "low"})
        ids = await handler._get_emails_by_filter("deferred")
        assert ids == ["e1"]

    @pytest.mark.asyncio
    async def test_filter_spam(self, handler):
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"priority": "spam"})
        _email_cache.set("e2", {"priority": "high"})
        ids = await handler._get_emails_by_filter("spam")
        assert ids == ["e1"]

    @pytest.mark.asyncio
    async def test_filter_read(self, handler):
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"unread": False})
        _email_cache.set("e2", {"unread": True})
        _email_cache.set("e3", {})  # unread defaults True via .get("unread", True)
        ids = await handler._get_emails_by_filter("read")
        assert ids == ["e1"]

    @pytest.mark.asyncio
    async def test_filter_disallowed(self, handler):
        ids = await handler._get_emails_by_filter("custom_filter")
        assert ids == []

    @pytest.mark.asyncio
    async def test_filter_empty_cache(self, handler):
        ids = await handler._get_emails_by_filter("all")
        assert ids == []

    @pytest.mark.asyncio
    async def test_filter_injection_attempt(self, handler):
        ids = await handler._get_emails_by_filter("all; DROP TABLE emails")
        assert ids == []

    @pytest.mark.asyncio
    async def test_filter_path_traversal_attempt(self, handler):
        ids = await handler._get_emails_by_filter("../../../etc/passwd")
        assert ids == []


# ============================================================================
# Security / Edge Cases
# ============================================================================


class TestSecurityEdgeCases:
    """Security and edge case tests."""

    @pytest.mark.asyncio
    async def test_execute_action_with_special_chars_email_id(self, handler):
        """Email IDs with special chars should not cause issues."""
        results = await handler._execute_action("archive", ["<script>alert(1)</script>"], {})
        assert len(results) == 1
        # The mixin itself does not validate email IDs -- that's inbox_command's job.
        # But it should not crash.
        assert results[0]["emailId"] == "<script>alert(1)</script>"

    @pytest.mark.asyncio
    async def test_sanitize_reply_with_script_injection(self, handler):
        result = handler._sanitize_action_params("reply", {"body": "<script>alert('xss')</script>"})
        # Body is sanitized by stripping/truncating, but not HTML-encoded
        # (encoding happens at display layer). It should not crash.
        assert isinstance(result["body"], str)

    @pytest.mark.asyncio
    async def test_sanitize_forward_with_email_injection(self, handler):
        result = handler._sanitize_action_params(
            "forward", {"to": "user@example.com\r\nBcc: evil@attacker.com"}
        )
        # Newlines in email address should fail validation
        assert result["to"] == ""

    def test_sanitize_snooze_with_long_duration(self, handler):
        result = handler._sanitize_action_params("snooze", {"duration": "x" * 10000})
        assert result["duration"] == "1d"

    @pytest.mark.asyncio
    async def test_archive_returns_dict_not_none(self, handler):
        result = await handler._archive_email("e1", {})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_snooze_returns_iso_timestamp(self, handler):
        result = await handler._snooze_email("e1", {"duration": "1h"})
        # Verify the until field is a valid ISO timestamp
        until = datetime.fromisoformat(result["until"])
        assert until > datetime.now(timezone.utc)

    @pytest.mark.asyncio
    async def test_execute_many_emails(self, handler):
        """Verify processing a large batch does not break."""
        ids = [f"email-{i}" for i in range(50)]
        results = await handler._execute_action("archive", ids, {})
        assert len(results) == 50
        assert all(r["success"] for r in results)

    @pytest.mark.asyncio
    async def test_vip_does_not_mutate_params(self, handler, prioritizer):
        params = {"sender": "a@b.com"}
        original = dict(params)
        await handler._mark_sender_vip("e1", params)
        assert params == original

    @pytest.mark.asyncio
    async def test_block_does_not_mutate_params(self, handler, prioritizer):
        params = {"sender": "a@b.com"}
        original = dict(params)
        await handler._block_sender("e1", params)
        assert params == original

    def test_sanitize_mark_vip_non_string_sender(self, handler):
        result = handler._sanitize_action_params("mark_vip", {"sender": 42})
        assert "sender" not in result

    def test_sanitize_block_non_string_sender(self, handler):
        result = handler._sanitize_action_params("block", {"sender": ["list"]})
        assert "sender" not in result
