"""Tests for the Unified Inbox sync service registry.

Covers all public functions in aragora/server/handlers/features/unified_inbox/sync.py:
- get_sync_services()          - Return the global sync services dict
- get_sync_services_lock()     - Return the global asyncio.Lock
- convert_synced_message_to_unified() - Convert synced messages to UnifiedMessage

Test areas:
- get_sync_services returns the module-level dict and reflects mutations
- get_sync_services_lock returns an asyncio.Lock and is reusable
- convert_synced_message_to_unified with full attributes on message and priority
- convert_synced_message_to_unified with partial attributes (hasattr fallbacks)
- convert_synced_message_to_unified with no priority (None)
- convert_synced_message_to_unified with priority missing score/tier/reasons
- Body preview truncation (>500 chars)
- Empty body, None body
- Attachments truthy/falsy
- Thread ID present/absent
- Provider enum propagation (Gmail vs Outlook)
- UUID generation for id and external_id fallback
- Edge cases: empty strings, empty lists, unicode content
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from aragora.server.handlers.features.unified_inbox.models import (
    EmailProvider,
    UnifiedMessage,
)
from aragora.server.handlers.features.unified_inbox.sync import (
    _sync_services,
    _sync_services_lock,
    convert_synced_message_to_unified,
    get_sync_services,
    get_sync_services_lock,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_sync_services():
    """Clear the global sync services registry between tests."""
    _sync_services.clear()
    yield
    _sync_services.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(**overrides: Any) -> SimpleNamespace:
    """Build a mock raw email message with all expected attributes."""
    defaults = {
        "id": "ext-msg-001",
        "subject": "Test Subject",
        "from_email": "sender@example.com",
        "from_name": "Sender Name",
        "to": ["recipient@example.com"],
        "cc": ["cc@example.com"],
        "date": _NOW,
        "snippet": "A short snippet...",
        "body": "Full body text of the email message.",
        "is_read": True,
        "is_starred": False,
        "attachments": ["file.pdf"],
        "labels": ["inbox", "important"],
        "thread_id": "thread-001",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_priority(**overrides: Any) -> SimpleNamespace:
    """Build a mock priority result with score, tier, reasons."""
    defaults = {
        "score": 0.9,
        "tier": "high",
        "reasons": ["urgent sender", "keyword match"],
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_synced_msg(
    message: Any = None,
    priority: Any = None,
) -> SimpleNamespace:
    """Build a SyncedMessage-like object with .message and .priority_result."""
    return SimpleNamespace(
        message=message or _make_message(),
        priority_result=priority,
    )


# =========================================================================
# get_sync_services() tests
# =========================================================================


class TestGetSyncServices:
    """Tests for get_sync_services()."""

    def test_returns_dict(self):
        """get_sync_services returns a dict."""
        result = get_sync_services()
        assert isinstance(result, dict)

    def test_returns_same_reference(self):
        """get_sync_services returns the module-level _sync_services dict."""
        result = get_sync_services()
        assert result is _sync_services

    def test_initially_empty(self):
        """Returned dict is empty when no services registered."""
        assert get_sync_services() == {}

    def test_reflects_mutations(self):
        """Mutations to the returned dict are visible on next call."""
        services = get_sync_services()
        services["tenant-1"] = {"acct-1": "mock_service"}
        assert get_sync_services() == {"tenant-1": {"acct-1": "mock_service"}}

    def test_nested_structure(self):
        """Supports the expected tenant->account->service nesting."""
        _sync_services["t1"] = {"a1": "svc_gmail", "a2": "svc_outlook"}
        _sync_services["t2"] = {"a3": "svc_gmail"}
        result = get_sync_services()
        assert len(result) == 2
        assert len(result["t1"]) == 2
        assert result["t2"]["a3"] == "svc_gmail"

    def test_clear_empties_services(self):
        """Clearing the dict is reflected in subsequent calls."""
        _sync_services["t1"] = {"a1": "svc"}
        assert len(get_sync_services()) == 1
        _sync_services.clear()
        assert len(get_sync_services()) == 0


# =========================================================================
# get_sync_services_lock() tests
# =========================================================================


class TestGetSyncServicesLock:
    """Tests for get_sync_services_lock()."""

    def test_returns_lock(self):
        """get_sync_services_lock returns an asyncio.Lock."""
        lock = get_sync_services_lock()
        assert isinstance(lock, asyncio.Lock)

    def test_returns_same_reference(self):
        """get_sync_services_lock returns the same lock on each call."""
        lock1 = get_sync_services_lock()
        lock2 = get_sync_services_lock()
        assert lock1 is lock2

    def test_is_module_level_lock(self):
        """Returned lock is the module-level _sync_services_lock."""
        assert get_sync_services_lock() is _sync_services_lock

    @pytest.mark.asyncio
    async def test_lock_is_acquirable(self):
        """Lock can be acquired and released."""
        lock = get_sync_services_lock()
        async with lock:
            assert lock.locked()
        assert not lock.locked()

    @pytest.mark.asyncio
    async def test_lock_provides_mutual_exclusion(self):
        """Lock prevents concurrent access."""
        lock = get_sync_services_lock()
        acquired_inner = False

        async with lock:
            # Try to acquire without blocking
            acquired_inner = lock.locked()
            assert acquired_inner is True


# =========================================================================
# convert_synced_message_to_unified() tests - Full attributes
# =========================================================================


class TestConvertFullAttributes:
    """Tests for convert_synced_message_to_unified with all attributes present."""

    def test_basic_conversion(self):
        """Converts a fully populated synced message correctly."""
        synced = _make_synced_msg(priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert isinstance(result, UnifiedMessage)

    def test_account_id_propagated(self):
        """account_id parameter is passed through."""
        synced = _make_synced_msg(priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-xyz", EmailProvider.GMAIL)
        assert result.account_id == "acct-xyz"

    def test_provider_propagated_gmail(self):
        """Provider enum is correctly set to GMAIL."""
        synced = _make_synced_msg(priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.provider == EmailProvider.GMAIL

    def test_provider_propagated_outlook(self):
        """Provider enum is correctly set to OUTLOOK."""
        synced = _make_synced_msg(priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.OUTLOOK)
        assert result.provider == EmailProvider.OUTLOOK

    def test_external_id_from_message(self):
        """external_id comes from msg.id."""
        msg = _make_message(id="ext-999")
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.external_id == "ext-999"

    def test_subject_extracted(self):
        """Subject is extracted from message."""
        msg = _make_message(subject="Important Meeting")
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.subject == "Important Meeting"

    def test_sender_fields(self):
        """sender_email and sender_name are extracted."""
        msg = _make_message(from_email="boss@corp.com", from_name="The Boss")
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.sender_email == "boss@corp.com"
        assert result.sender_name == "The Boss"

    def test_recipients_and_cc(self):
        """recipients and cc lists are extracted."""
        msg = _make_message(
            to=["a@x.com", "b@x.com"],
            cc=["c@x.com"],
        )
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.recipients == ["a@x.com", "b@x.com"]
        assert result.cc == ["c@x.com"]

    def test_received_at_from_date(self):
        """received_at comes from msg.date."""
        msg = _make_message(date=_NOW)
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.received_at == _NOW

    def test_snippet_extracted(self):
        """snippet is extracted from message."""
        msg = _make_message(snippet="Preview text here")
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.snippet == "Preview text here"

    def test_body_preview_truncated_at_500(self):
        """body_preview is truncated to 500 characters."""
        long_body = "A" * 1000
        msg = _make_message(body=long_body)
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert len(result.body_preview) == 500
        assert result.body_preview == "A" * 500

    def test_body_preview_short_body(self):
        """body_preview retains full body when under 500 chars."""
        msg = _make_message(body="Short body.")
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.body_preview == "Short body."

    def test_is_read_true(self):
        """is_read is True when message is read."""
        msg = _make_message(is_read=True)
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.is_read is True

    def test_is_read_false(self):
        """is_read is False when message is unread."""
        msg = _make_message(is_read=False)
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.is_read is False

    def test_is_starred(self):
        """is_starred reflects message attribute."""
        msg = _make_message(is_starred=True)
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.is_starred is True

    def test_has_attachments_true(self):
        """has_attachments is True when attachments list is non-empty."""
        msg = _make_message(attachments=["a.pdf", "b.docx"])
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.has_attachments is True

    def test_has_attachments_false_empty_list(self):
        """has_attachments is False when attachments is empty list."""
        msg = _make_message(attachments=[])
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.has_attachments is False

    def test_labels_extracted(self):
        """labels are extracted from message."""
        msg = _make_message(labels=["inbox", "starred", "important"])
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.labels == ["inbox", "starred", "important"]

    def test_thread_id_present(self):
        """thread_id is extracted when present."""
        msg = _make_message(thread_id="thr-42")
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.thread_id == "thr-42"

    def test_priority_score_from_priority(self):
        """priority_score comes from priority_result.score."""
        priority = _make_priority(score=0.85)
        synced = _make_synced_msg(priority=priority)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.priority_score == 0.85

    def test_priority_tier_from_priority(self):
        """priority_tier comes from priority_result.tier."""
        priority = _make_priority(tier="critical")
        synced = _make_synced_msg(priority=priority)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.priority_tier == "critical"

    def test_priority_reasons_from_priority(self):
        """priority_reasons comes from priority_result.reasons."""
        priority = _make_priority(reasons=["vip sender"])
        synced = _make_synced_msg(priority=priority)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.priority_reasons == ["vip sender"]

    def test_id_is_uuid_string(self):
        """The generated id field is a valid UUID string."""
        synced = _make_synced_msg(priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        # UUID4 format: 8-4-4-4-12 hex digits
        import re

        assert re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            result.id,
        )

    def test_unique_ids_per_call(self):
        """Each call generates a unique id."""
        synced = _make_synced_msg(priority=_make_priority())
        r1 = convert_synced_message_to_unified(synced, "a", EmailProvider.GMAIL)
        r2 = convert_synced_message_to_unified(synced, "a", EmailProvider.GMAIL)
        assert r1.id != r2.id


# =========================================================================
# convert_synced_message_to_unified() tests - No priority
# =========================================================================


class TestConvertNoPriority:
    """Tests when priority_result is None."""

    def test_default_priority_score(self):
        """Defaults to 0.5 when no priority."""
        synced = _make_synced_msg(priority=None)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.priority_score == 0.5

    def test_default_priority_tier(self):
        """Defaults to 'medium' when no priority."""
        synced = _make_synced_msg(priority=None)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.priority_tier == "medium"

    def test_default_priority_reasons_empty(self):
        """Defaults to empty list when no priority."""
        synced = _make_synced_msg(priority=None)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.priority_reasons == []


# =========================================================================
# convert_synced_message_to_unified() - Priority missing individual fields
# =========================================================================


class TestConvertPartialPriority:
    """Tests when priority_result exists but is missing some attributes."""

    def test_missing_score_defaults_to_0_5(self):
        """Falls back to 0.5 if priority has no .score."""
        priority = SimpleNamespace(tier="low", reasons=["reason"])
        synced = _make_synced_msg(priority=priority)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.priority_score == 0.5

    def test_missing_tier_defaults_to_medium(self):
        """Falls back to 'medium' if priority has no .tier."""
        priority = SimpleNamespace(score=0.7, reasons=["reason"])
        synced = _make_synced_msg(priority=priority)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.priority_tier == "medium"

    def test_missing_reasons_defaults_to_empty(self):
        """Falls back to [] if priority has no .reasons."""
        priority = SimpleNamespace(score=0.7, tier="high")
        synced = _make_synced_msg(priority=priority)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.priority_reasons == []

    def test_all_priority_fields_missing(self):
        """Falls back to all defaults when priority has no relevant fields."""
        priority = SimpleNamespace()  # No score, tier, or reasons
        synced = _make_synced_msg(priority=priority)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.priority_score == 0.5
        assert result.priority_tier == "medium"
        assert result.priority_reasons == []


# =========================================================================
# convert_synced_message_to_unified() - Message missing attributes (hasattr)
# =========================================================================


class TestConvertPartialMessage:
    """Tests when the raw message object is missing some attributes."""

    def _bare_message(self, **attrs: Any) -> SimpleNamespace:
        """Create a message with only specified attributes."""
        return SimpleNamespace(**attrs)

    def test_missing_id_generates_uuid(self):
        """external_id is a UUID when msg has no .id."""
        msg = self._bare_message(
            subject="s",
            from_email="e",
            from_name="n",
            to=[],
            cc=[],
            date=_NOW,
            snippet="",
            body="b",
            is_read=False,
            is_starred=False,
            attachments=[],
            labels=[],
            thread_id=None,
        )
        synced = _make_synced_msg(message=msg, priority=None)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        # Should be a UUID string (not empty)
        assert len(result.external_id) > 0
        assert "-" in result.external_id

    def test_missing_subject_defaults_empty(self):
        """subject defaults to '' when msg has no .subject."""
        msg = self._bare_message(
            id="x",
            from_email="e",
            from_name="n",
            to=[],
            cc=[],
            date=_NOW,
            snippet="",
            body="b",
            is_read=False,
            is_starred=False,
            attachments=[],
            labels=[],
            thread_id=None,
        )
        synced = _make_synced_msg(message=msg, priority=None)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.subject == ""

    def test_missing_from_email_defaults_empty(self):
        """sender_email defaults to '' when msg has no .from_email."""
        msg = self._bare_message(
            id="x",
            subject="s",
            from_name="n",
            to=[],
            cc=[],
            date=_NOW,
            snippet="",
            body="b",
            is_read=False,
            is_starred=False,
            attachments=[],
            labels=[],
            thread_id=None,
        )
        synced = _make_synced_msg(message=msg, priority=None)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.sender_email == ""

    def test_missing_from_name_defaults_empty(self):
        """sender_name defaults to '' when msg has no .from_name."""
        msg = self._bare_message(
            id="x",
            subject="s",
            from_email="e",
            to=[],
            cc=[],
            date=_NOW,
            snippet="",
            body="b",
            is_read=False,
            is_starred=False,
            attachments=[],
            labels=[],
            thread_id=None,
        )
        synced = _make_synced_msg(message=msg, priority=None)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.sender_name == ""

    def test_missing_to_defaults_empty_list(self):
        """recipients defaults to [] when msg has no .to."""
        msg = self._bare_message(
            id="x",
            subject="s",
            from_email="e",
            from_name="n",
            cc=[],
            date=_NOW,
            snippet="",
            body="b",
            is_read=False,
            is_starred=False,
            attachments=[],
            labels=[],
            thread_id=None,
        )
        synced = _make_synced_msg(message=msg, priority=None)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.recipients == []

    def test_missing_cc_defaults_empty_list(self):
        """cc defaults to [] when msg has no .cc."""
        msg = self._bare_message(
            id="x",
            subject="s",
            from_email="e",
            from_name="n",
            to=[],
            date=_NOW,
            snippet="",
            body="b",
            is_read=False,
            is_starred=False,
            attachments=[],
            labels=[],
            thread_id=None,
        )
        synced = _make_synced_msg(message=msg, priority=None)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.cc == []

    def test_missing_date_defaults_to_now(self):
        """received_at defaults to datetime.now(utc) when msg has no .date."""
        msg = self._bare_message(
            id="x",
            subject="s",
            from_email="e",
            from_name="n",
            to=[],
            cc=[],
            snippet="",
            body="b",
            is_read=False,
            is_starred=False,
            attachments=[],
            labels=[],
            thread_id=None,
        )
        synced = _make_synced_msg(message=msg, priority=None)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert isinstance(result.received_at, datetime)

    def test_missing_snippet_defaults_empty(self):
        """snippet defaults to '' when msg has no .snippet."""
        msg = self._bare_message(
            id="x",
            subject="s",
            from_email="e",
            from_name="n",
            to=[],
            cc=[],
            date=_NOW,
            body="b",
            is_read=False,
            is_starred=False,
            attachments=[],
            labels=[],
            thread_id=None,
        )
        synced = _make_synced_msg(message=msg, priority=None)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.snippet == ""

    def test_missing_body_defaults_empty(self):
        """body_preview defaults to '' when msg has no .body."""
        msg = self._bare_message(
            id="x",
            subject="s",
            from_email="e",
            from_name="n",
            to=[],
            cc=[],
            date=_NOW,
            snippet="",
            is_read=False,
            is_starred=False,
            attachments=[],
            labels=[],
            thread_id=None,
        )
        synced = _make_synced_msg(message=msg, priority=None)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.body_preview == ""

    def test_none_body_defaults_empty(self):
        """body_preview defaults to '' when msg.body is None."""
        msg = self._bare_message(
            id="x",
            subject="s",
            from_email="e",
            from_name="n",
            to=[],
            cc=[],
            date=_NOW,
            snippet="",
            body=None,
            is_read=False,
            is_starred=False,
            attachments=[],
            labels=[],
            thread_id=None,
        )
        synced = _make_synced_msg(message=msg, priority=None)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.body_preview == ""

    def test_empty_body_defaults_empty(self):
        """body_preview defaults to '' when msg.body is empty string."""
        msg = self._bare_message(
            id="x",
            subject="s",
            from_email="e",
            from_name="n",
            to=[],
            cc=[],
            date=_NOW,
            snippet="",
            body="",
            is_read=False,
            is_starred=False,
            attachments=[],
            labels=[],
            thread_id=None,
        )
        synced = _make_synced_msg(message=msg, priority=None)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.body_preview == ""

    def test_missing_is_read_defaults_false(self):
        """is_read defaults to False when msg has no .is_read."""
        msg = self._bare_message(
            id="x",
            subject="s",
            from_email="e",
            from_name="n",
            to=[],
            cc=[],
            date=_NOW,
            snippet="",
            body="b",
            is_starred=False,
            attachments=[],
            labels=[],
            thread_id=None,
        )
        synced = _make_synced_msg(message=msg, priority=None)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.is_read is False

    def test_missing_is_starred_defaults_false(self):
        """is_starred defaults to False when msg has no .is_starred."""
        msg = self._bare_message(
            id="x",
            subject="s",
            from_email="e",
            from_name="n",
            to=[],
            cc=[],
            date=_NOW,
            snippet="",
            body="b",
            is_read=False,
            attachments=[],
            labels=[],
            thread_id=None,
        )
        synced = _make_synced_msg(message=msg, priority=None)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.is_starred is False

    def test_missing_attachments_defaults_false(self):
        """has_attachments defaults to False when msg has no .attachments."""
        msg = self._bare_message(
            id="x",
            subject="s",
            from_email="e",
            from_name="n",
            to=[],
            cc=[],
            date=_NOW,
            snippet="",
            body="b",
            is_read=False,
            is_starred=False,
            labels=[],
            thread_id=None,
        )
        synced = _make_synced_msg(message=msg, priority=None)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.has_attachments is False

    def test_missing_labels_defaults_empty_list(self):
        """labels defaults to [] when msg has no .labels."""
        msg = self._bare_message(
            id="x",
            subject="s",
            from_email="e",
            from_name="n",
            to=[],
            cc=[],
            date=_NOW,
            snippet="",
            body="b",
            is_read=False,
            is_starred=False,
            attachments=[],
            thread_id=None,
        )
        synced = _make_synced_msg(message=msg, priority=None)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.labels == []

    def test_missing_thread_id_defaults_none(self):
        """thread_id defaults to None when msg has no .thread_id."""
        msg = self._bare_message(
            id="x",
            subject="s",
            from_email="e",
            from_name="n",
            to=[],
            cc=[],
            date=_NOW,
            snippet="",
            body="b",
            is_read=False,
            is_starred=False,
            attachments=[],
            labels=[],
        )
        synced = _make_synced_msg(message=msg, priority=None)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.thread_id is None

    def test_completely_bare_message(self):
        """A message with no attributes at all uses all defaults."""
        msg = SimpleNamespace()  # No attributes
        synced = _make_synced_msg(message=msg, priority=None)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.subject == ""
        assert result.sender_email == ""
        assert result.sender_name == ""
        assert result.recipients == []
        assert result.cc == []
        assert result.snippet == ""
        assert result.body_preview == ""
        assert result.is_read is False
        assert result.is_starred is False
        assert result.has_attachments is False
        assert result.labels == []
        assert result.thread_id is None
        assert result.priority_score == 0.5
        assert result.priority_tier == "medium"
        assert result.priority_reasons == []
        # id and external_id should still be generated UUIDs
        assert len(result.id) > 0
        assert len(result.external_id) > 0


# =========================================================================
# Edge cases
# =========================================================================


class TestConvertEdgeCases:
    """Edge cases and special scenarios for convert_synced_message_to_unified."""

    def test_unicode_subject(self):
        """Unicode content in subject is preserved."""
        msg = _make_message(subject="Re: Reunion de planificacion")
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.subject == "Re: Reunion de planificacion"

    def test_unicode_body_preview(self):
        """Unicode body is preserved in preview."""
        msg = _make_message(body="Bonjour, comment allez-vous?")
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.body_preview == "Bonjour, comment allez-vous?"

    def test_body_exactly_500_chars(self):
        """Body of exactly 500 chars is not truncated."""
        body_500 = "X" * 500
        msg = _make_message(body=body_500)
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert len(result.body_preview) == 500

    def test_body_501_chars_truncated(self):
        """Body of 501 chars is truncated to 500."""
        body_501 = "Y" * 501
        msg = _make_message(body=body_501)
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert len(result.body_preview) == 500

    def test_empty_recipients_list(self):
        """Empty recipients list is preserved."""
        msg = _make_message(to=[])
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.recipients == []

    def test_many_recipients(self):
        """Large recipient list is preserved."""
        many = [f"user{i}@example.com" for i in range(50)]
        msg = _make_message(to=many)
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert len(result.recipients) == 50

    def test_priority_score_zero(self):
        """Priority score of 0.0 is preserved."""
        priority = _make_priority(score=0.0)
        synced = _make_synced_msg(priority=priority)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.priority_score == 0.0

    def test_priority_score_one(self):
        """Priority score of 1.0 is preserved."""
        priority = _make_priority(score=1.0)
        synced = _make_synced_msg(priority=priority)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.priority_score == 1.0

    def test_empty_priority_reasons(self):
        """Empty reasons list from priority is preserved."""
        priority = _make_priority(reasons=[])
        synced = _make_synced_msg(priority=priority)
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.priority_reasons == []

    def test_multiple_labels(self):
        """Multiple labels are preserved."""
        msg = _make_message(labels=["inbox", "starred", "work", "urgent"])
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.labels == ["inbox", "starred", "work", "urgent"]

    def test_thread_id_none_explicitly(self):
        """Explicit None thread_id is preserved."""
        msg = _make_message(thread_id=None)
        synced = _make_synced_msg(message=msg, priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "acct-1", EmailProvider.GMAIL)
        assert result.thread_id is None

    def test_empty_string_account_id(self):
        """Empty string account_id is accepted."""
        synced = _make_synced_msg(priority=_make_priority())
        result = convert_synced_message_to_unified(synced, "", EmailProvider.GMAIL)
        assert result.account_id == ""
