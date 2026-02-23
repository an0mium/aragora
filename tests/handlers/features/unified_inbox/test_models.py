"""Tests for the Unified Inbox data models.

Covers all enums, dataclasses, serialization (to_dict), and record
conversion helpers defined in:
    aragora/server/handlers/features/unified_inbox/models.py

Test areas:
- EmailProvider enum values and membership
- AccountStatus enum values and membership
- TriageAction enum values and membership
- ConnectedAccount dataclass defaults, to_dict, edge cases
- UnifiedMessage dataclass defaults, to_dict, triage serialization
- TriageResult dataclass defaults, to_dict, schedule_for serialization
- InboxStats dataclass to_dict
- ensure_datetime helper (None, datetime, ISO string, invalid)
- account_to_record / record_to_account round-trip and edge cases
- message_to_record / record_to_message round-trip and edge cases
- triage_to_record / record_to_triage round-trip and edge cases
- Security / injection edge cases
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from aragora.server.handlers.features.unified_inbox.models import (
    AccountStatus,
    ConnectedAccount,
    EmailProvider,
    InboxStats,
    TriageAction,
    TriageResult,
    UnifiedMessage,
    account_to_record,
    ensure_datetime,
    message_to_record,
    record_to_account,
    record_to_message,
    record_to_triage,
    triage_to_record,
)


# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)
_LATER = datetime(2026, 2, 24, 8, 30, 0, tzinfo=timezone.utc)


def _make_account(**overrides: Any) -> ConnectedAccount:
    """Build a ConnectedAccount with sensible defaults."""
    defaults: dict[str, Any] = {
        "id": "acct-1",
        "provider": EmailProvider.GMAIL,
        "email_address": "user@gmail.com",
        "display_name": "Test User",
        "status": AccountStatus.CONNECTED,
        "connected_at": _NOW,
        "last_sync": _NOW,
        "total_messages": 42,
        "unread_count": 5,
        "sync_errors": 0,
        "metadata": {},
    }
    defaults.update(overrides)
    return ConnectedAccount(**defaults)


def _make_message(**overrides: Any) -> UnifiedMessage:
    """Build a UnifiedMessage with sensible defaults."""
    defaults: dict[str, Any] = {
        "id": "msg-1",
        "account_id": "acct-1",
        "provider": EmailProvider.GMAIL,
        "external_id": "ext-123",
        "subject": "Test Subject",
        "sender_email": "sender@example.com",
        "sender_name": "Sender Name",
        "recipients": ["user@gmail.com"],
        "cc": ["cc@example.com"],
        "received_at": _NOW,
        "snippet": "Preview text",
        "body_preview": "Full body preview",
        "is_read": False,
        "is_starred": True,
        "has_attachments": True,
        "labels": ["inbox", "important"],
        "thread_id": "thread-1",
        "priority_score": 0.75,
        "priority_tier": "high",
        "priority_reasons": ["VIP sender"],
        "triage_action": TriageAction.RESPOND_URGENT,
        "triage_rationale": "Important message from VIP",
    }
    defaults.update(overrides)
    return UnifiedMessage(**defaults)


def _make_triage(**overrides: Any) -> TriageResult:
    """Build a TriageResult with sensible defaults."""
    defaults: dict[str, Any] = {
        "message_id": "msg-1",
        "recommended_action": TriageAction.DEFER,
        "confidence": 0.85,
        "rationale": "Low priority content",
        "suggested_response": "Will follow up later",
        "delegate_to": "team-lead",
        "schedule_for": _LATER,
        "agents_involved": ["analyst", "expert"],
        "debate_summary": "Agents agreed to defer",
    }
    defaults.update(overrides)
    return TriageResult(**defaults)


def _make_stats(**overrides: Any) -> InboxStats:
    """Build an InboxStats with sensible defaults."""
    defaults: dict[str, Any] = {
        "total_accounts": 2,
        "total_messages": 100,
        "unread_count": 15,
        "messages_by_priority": {"high": 10, "medium": 60, "low": 30},
        "messages_by_provider": {"gmail": 70, "outlook": 30},
        "avg_response_time_hours": 2.5,
        "pending_triage": 8,
        "sync_health": {"gmail": "ok", "outlook": "ok"},
        "top_senders": [{"email": "boss@example.com", "count": 20}],
        "hourly_volume": [{"hour": 9, "count": 12}],
    }
    defaults.update(overrides)
    return InboxStats(**defaults)


# ===========================================================================
# EmailProvider Enum
# ===========================================================================


class TestEmailProvider:
    """Tests for the EmailProvider enum."""

    def test_gmail_value(self):
        assert EmailProvider.GMAIL.value == "gmail"

    def test_outlook_value(self):
        assert EmailProvider.OUTLOOK.value == "outlook"

    def test_member_count(self):
        assert len(EmailProvider) == 2

    def test_from_value_gmail(self):
        assert EmailProvider("gmail") is EmailProvider.GMAIL

    def test_from_value_outlook(self):
        assert EmailProvider("outlook") is EmailProvider.OUTLOOK

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            EmailProvider("yahoo")

    def test_name_attribute(self):
        assert EmailProvider.GMAIL.name == "GMAIL"
        assert EmailProvider.OUTLOOK.name == "OUTLOOK"

    def test_iteration(self):
        values = [p.value for p in EmailProvider]
        assert "gmail" in values
        assert "outlook" in values


# ===========================================================================
# AccountStatus Enum
# ===========================================================================


class TestAccountStatus:
    """Tests for the AccountStatus enum."""

    def test_pending_value(self):
        assert AccountStatus.PENDING.value == "pending"

    def test_connected_value(self):
        assert AccountStatus.CONNECTED.value == "connected"

    def test_syncing_value(self):
        assert AccountStatus.SYNCING.value == "syncing"

    def test_error_value(self):
        assert AccountStatus.ERROR.value == "error"

    def test_disconnected_value(self):
        assert AccountStatus.DISCONNECTED.value == "disconnected"

    def test_member_count(self):
        assert len(AccountStatus) == 5

    def test_from_value_connected(self):
        assert AccountStatus("connected") is AccountStatus.CONNECTED

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            AccountStatus("active")

    def test_all_values_are_strings(self):
        for status in AccountStatus:
            assert isinstance(status.value, str)


# ===========================================================================
# TriageAction Enum
# ===========================================================================


class TestTriageAction:
    """Tests for the TriageAction enum."""

    def test_respond_urgent_value(self):
        assert TriageAction.RESPOND_URGENT.value == "respond_urgent"

    def test_respond_normal_value(self):
        assert TriageAction.RESPOND_NORMAL.value == "respond_normal"

    def test_delegate_value(self):
        assert TriageAction.DELEGATE.value == "delegate"

    def test_schedule_value(self):
        assert TriageAction.SCHEDULE.value == "schedule"

    def test_archive_value(self):
        assert TriageAction.ARCHIVE.value == "archive"

    def test_delete_value(self):
        assert TriageAction.DELETE.value == "delete"

    def test_flag_value(self):
        assert TriageAction.FLAG.value == "flag"

    def test_defer_value(self):
        assert TriageAction.DEFER.value == "defer"

    def test_member_count(self):
        assert len(TriageAction) == 8

    def test_from_value_defer(self):
        assert TriageAction("defer") is TriageAction.DEFER

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            TriageAction("burn")


# ===========================================================================
# ConnectedAccount Dataclass
# ===========================================================================


class TestConnectedAccount:
    """Tests for the ConnectedAccount dataclass."""

    def test_basic_creation(self):
        acct = _make_account()
        assert acct.id == "acct-1"
        assert acct.provider is EmailProvider.GMAIL
        assert acct.email_address == "user@gmail.com"
        assert acct.display_name == "Test User"
        assert acct.status is AccountStatus.CONNECTED
        assert acct.connected_at == _NOW
        assert acct.last_sync == _NOW
        assert acct.total_messages == 42
        assert acct.unread_count == 5
        assert acct.sync_errors == 0
        assert acct.metadata == {}

    def test_defaults_for_optional_fields(self):
        acct = ConnectedAccount(
            id="acct-min",
            provider=EmailProvider.OUTLOOK,
            email_address="user@outlook.com",
            display_name="User",
            status=AccountStatus.PENDING,
            connected_at=_NOW,
        )
        assert acct.last_sync is None
        assert acct.total_messages == 0
        assert acct.unread_count == 0
        assert acct.sync_errors == 0
        assert acct.metadata == {}

    def test_to_dict_all_fields(self):
        acct = _make_account()
        d = acct.to_dict()
        assert d["id"] == "acct-1"
        assert d["provider"] == "gmail"
        assert d["email_address"] == "user@gmail.com"
        assert d["display_name"] == "Test User"
        assert d["status"] == "connected"
        assert d["connected_at"] == _NOW.isoformat()
        assert d["last_sync"] == _NOW.isoformat()
        assert d["total_messages"] == 42
        assert d["unread_count"] == 5
        assert d["sync_errors"] == 0

    def test_to_dict_last_sync_none(self):
        acct = _make_account(last_sync=None)
        d = acct.to_dict()
        assert d["last_sync"] is None

    def test_to_dict_provider_is_string(self):
        acct = _make_account(provider=EmailProvider.OUTLOOK)
        d = acct.to_dict()
        assert d["provider"] == "outlook"

    def test_to_dict_status_is_string(self):
        acct = _make_account(status=AccountStatus.ERROR)
        d = acct.to_dict()
        assert d["status"] == "error"

    def test_to_dict_does_not_include_metadata(self):
        acct = _make_account(metadata={"key": "val"})
        d = acct.to_dict()
        # The to_dict method does not serialize metadata
        assert "metadata" not in d

    def test_to_dict_keys(self):
        d = _make_account().to_dict()
        expected_keys = {
            "id",
            "provider",
            "email_address",
            "display_name",
            "status",
            "connected_at",
            "last_sync",
            "total_messages",
            "unread_count",
            "sync_errors",
        }
        assert set(d.keys()) == expected_keys


# ===========================================================================
# UnifiedMessage Dataclass
# ===========================================================================


class TestUnifiedMessage:
    """Tests for the UnifiedMessage dataclass."""

    def test_basic_creation(self):
        msg = _make_message()
        assert msg.id == "msg-1"
        assert msg.account_id == "acct-1"
        assert msg.provider is EmailProvider.GMAIL
        assert msg.external_id == "ext-123"
        assert msg.subject == "Test Subject"
        assert msg.sender_email == "sender@example.com"
        assert msg.sender_name == "Sender Name"
        assert msg.recipients == ["user@gmail.com"]
        assert msg.cc == ["cc@example.com"]
        assert msg.received_at == _NOW
        assert msg.snippet == "Preview text"
        assert msg.body_preview == "Full body preview"
        assert msg.is_read is False
        assert msg.is_starred is True
        assert msg.has_attachments is True
        assert msg.labels == ["inbox", "important"]
        assert msg.thread_id == "thread-1"
        assert msg.priority_score == 0.75
        assert msg.priority_tier == "high"
        assert msg.priority_reasons == ["VIP sender"]
        assert msg.triage_action is TriageAction.RESPOND_URGENT
        assert msg.triage_rationale == "Important message from VIP"

    def test_defaults_for_optional_fields(self):
        msg = UnifiedMessage(
            id="msg-min",
            account_id="acct-1",
            provider=EmailProvider.GMAIL,
            external_id="ext-1",
            subject="Sub",
            sender_email="s@e.com",
            sender_name="S",
            recipients=["r@e.com"],
            cc=[],
            received_at=_NOW,
            snippet="snip",
            body_preview="body",
            is_read=True,
            is_starred=False,
            has_attachments=False,
            labels=[],
        )
        assert msg.thread_id is None
        assert msg.priority_score == 0.5
        assert msg.priority_tier == "medium"
        assert msg.priority_reasons == []
        assert msg.triage_action is None
        assert msg.triage_rationale is None

    def test_to_dict_with_triage(self):
        msg = _make_message()
        d = msg.to_dict()
        assert d["id"] == "msg-1"
        assert d["provider"] == "gmail"
        assert d["sender"]["email"] == "sender@example.com"
        assert d["sender"]["name"] == "Sender Name"
        assert d["recipients"] == ["user@gmail.com"]
        assert d["cc"] == ["cc@example.com"]
        assert d["received_at"] == _NOW.isoformat()
        assert d["snippet"] == "Preview text"
        assert d["is_read"] is False
        assert d["is_starred"] is True
        assert d["has_attachments"] is True
        assert d["labels"] == ["inbox", "important"]
        assert d["thread_id"] == "thread-1"
        # Priority
        assert d["priority"]["score"] == 0.75
        assert d["priority"]["tier"] == "high"
        assert d["priority"]["reasons"] == ["VIP sender"]
        # Triage
        assert d["triage"] is not None
        assert d["triage"]["action"] == "respond_urgent"
        assert d["triage"]["rationale"] == "Important message from VIP"

    def test_to_dict_without_triage(self):
        msg = _make_message(triage_action=None, triage_rationale=None)
        d = msg.to_dict()
        assert d["triage"] is None

    def test_to_dict_triage_action_value_serialized(self):
        for action in TriageAction:
            msg = _make_message(triage_action=action, triage_rationale="test")
            d = msg.to_dict()
            assert d["triage"]["action"] == action.value

    def test_to_dict_keys(self):
        d = _make_message().to_dict()
        expected_keys = {
            "id",
            "account_id",
            "provider",
            "external_id",
            "subject",
            "sender",
            "recipients",
            "cc",
            "received_at",
            "snippet",
            "is_read",
            "is_starred",
            "has_attachments",
            "labels",
            "thread_id",
            "priority",
            "triage",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_sender_is_dict(self):
        d = _make_message().to_dict()
        assert isinstance(d["sender"], dict)
        assert "email" in d["sender"]
        assert "name" in d["sender"]

    def test_to_dict_priority_is_dict(self):
        d = _make_message().to_dict()
        assert isinstance(d["priority"], dict)
        assert "score" in d["priority"]
        assert "tier" in d["priority"]
        assert "reasons" in d["priority"]


# ===========================================================================
# TriageResult Dataclass
# ===========================================================================


class TestTriageResult:
    """Tests for the TriageResult dataclass."""

    def test_basic_creation(self):
        t = _make_triage()
        assert t.message_id == "msg-1"
        assert t.recommended_action is TriageAction.DEFER
        assert t.confidence == 0.85
        assert t.rationale == "Low priority content"
        assert t.suggested_response == "Will follow up later"
        assert t.delegate_to == "team-lead"
        assert t.schedule_for == _LATER
        assert t.agents_involved == ["analyst", "expert"]
        assert t.debate_summary == "Agents agreed to defer"

    def test_to_dict_all_fields(self):
        t = _make_triage()
        d = t.to_dict()
        assert d["message_id"] == "msg-1"
        assert d["recommended_action"] == "defer"
        assert d["confidence"] == 0.85
        assert d["rationale"] == "Low priority content"
        assert d["suggested_response"] == "Will follow up later"
        assert d["delegate_to"] == "team-lead"
        assert d["schedule_for"] == _LATER.isoformat()
        assert d["agents_involved"] == ["analyst", "expert"]
        assert d["debate_summary"] == "Agents agreed to defer"

    def test_to_dict_schedule_for_none(self):
        t = _make_triage(schedule_for=None)
        d = t.to_dict()
        assert d["schedule_for"] is None

    def test_to_dict_optional_none_fields(self):
        t = _make_triage(
            suggested_response=None,
            delegate_to=None,
            schedule_for=None,
            debate_summary=None,
        )
        d = t.to_dict()
        assert d["suggested_response"] is None
        assert d["delegate_to"] is None
        assert d["schedule_for"] is None
        assert d["debate_summary"] is None

    def test_to_dict_action_is_string(self):
        for action in TriageAction:
            t = _make_triage(recommended_action=action)
            d = t.to_dict()
            assert d["recommended_action"] == action.value

    def test_to_dict_keys(self):
        d = _make_triage().to_dict()
        expected_keys = {
            "message_id",
            "recommended_action",
            "confidence",
            "rationale",
            "suggested_response",
            "delegate_to",
            "schedule_for",
            "agents_involved",
            "debate_summary",
        }
        assert set(d.keys()) == expected_keys


# ===========================================================================
# InboxStats Dataclass
# ===========================================================================


class TestInboxStats:
    """Tests for the InboxStats dataclass."""

    def test_basic_creation(self):
        s = _make_stats()
        assert s.total_accounts == 2
        assert s.total_messages == 100
        assert s.unread_count == 15
        assert s.avg_response_time_hours == 2.5
        assert s.pending_triage == 8

    def test_to_dict_all_fields(self):
        s = _make_stats()
        d = s.to_dict()
        assert d["total_accounts"] == 2
        assert d["total_messages"] == 100
        assert d["unread_count"] == 15
        assert d["messages_by_priority"] == {"high": 10, "medium": 60, "low": 30}
        assert d["messages_by_provider"] == {"gmail": 70, "outlook": 30}
        assert d["avg_response_time_hours"] == 2.5
        assert d["pending_triage"] == 8
        assert d["sync_health"] == {"gmail": "ok", "outlook": "ok"}
        assert d["top_senders"] == [{"email": "boss@example.com", "count": 20}]
        assert d["hourly_volume"] == [{"hour": 9, "count": 12}]

    def test_to_dict_empty_collections(self):
        s = _make_stats(
            messages_by_priority={},
            messages_by_provider={},
            sync_health={},
            top_senders=[],
            hourly_volume=[],
        )
        d = s.to_dict()
        assert d["messages_by_priority"] == {}
        assert d["messages_by_provider"] == {}
        assert d["sync_health"] == {}
        assert d["top_senders"] == []
        assert d["hourly_volume"] == []

    def test_to_dict_keys(self):
        d = _make_stats().to_dict()
        expected_keys = {
            "total_accounts",
            "total_messages",
            "unread_count",
            "messages_by_priority",
            "messages_by_provider",
            "avg_response_time_hours",
            "pending_triage",
            "sync_health",
            "top_senders",
            "hourly_volume",
        }
        assert set(d.keys()) == expected_keys


# ===========================================================================
# ensure_datetime helper
# ===========================================================================


class TestEnsureDatetime:
    """Tests for the ensure_datetime conversion helper."""

    def test_none_returns_none(self):
        assert ensure_datetime(None) is None

    def test_datetime_returns_same(self):
        result = ensure_datetime(_NOW)
        assert result is _NOW

    def test_iso_string_returns_datetime(self):
        result = ensure_datetime("2026-02-23T12:00:00+00:00")
        assert isinstance(result, datetime)
        assert result.year == 2026
        assert result.month == 2
        assert result.day == 23

    def test_naive_iso_string(self):
        result = ensure_datetime("2026-01-15T10:30:00")
        assert isinstance(result, datetime)
        assert result.hour == 10
        assert result.minute == 30

    def test_date_only_iso_string(self):
        result = ensure_datetime("2026-03-01")
        assert isinstance(result, datetime)
        assert result.year == 2026
        assert result.month == 3

    def test_invalid_string_returns_none(self):
        assert ensure_datetime("not-a-date") is None

    def test_empty_string_returns_none(self):
        assert ensure_datetime("") is None

    def test_numeric_value_converted_via_str(self):
        # int/float are str()'d first; "12345" is not a valid ISO date
        assert ensure_datetime(12345) is None

    def test_boolean_returns_none(self):
        # str(True) = "True" which is not a valid ISO string
        assert ensure_datetime(True) is None


# ===========================================================================
# account_to_record / record_to_account
# ===========================================================================


class TestAccountRecordConversion:
    """Tests for account_to_record and record_to_account."""

    def test_account_to_record_fields(self):
        acct = _make_account()
        rec = account_to_record(acct)
        assert rec["id"] == "acct-1"
        assert rec["provider"] == "gmail"
        assert rec["email_address"] == "user@gmail.com"
        assert rec["display_name"] == "Test User"
        assert rec["status"] == "connected"
        assert rec["connected_at"] == _NOW
        assert rec["last_sync"] == _NOW
        assert rec["total_messages"] == 42
        assert rec["unread_count"] == 5
        assert rec["sync_errors"] == 0
        assert rec["metadata"] == {}

    def test_account_to_record_none_last_sync(self):
        acct = _make_account(last_sync=None)
        rec = account_to_record(acct)
        assert rec["last_sync"] is None

    def test_record_to_account_full(self):
        rec = {
            "id": "acct-2",
            "provider": "outlook",
            "email_address": "u@outlook.com",
            "display_name": "Outlook User",
            "status": "syncing",
            "connected_at": _NOW,
            "last_sync": _LATER,
            "total_messages": 10,
            "unread_count": 2,
            "sync_errors": 1,
            "metadata": {"foo": "bar"},
        }
        acct = record_to_account(rec)
        assert acct.id == "acct-2"
        assert acct.provider is EmailProvider.OUTLOOK
        assert acct.email_address == "u@outlook.com"
        assert acct.display_name == "Outlook User"
        assert acct.status is AccountStatus.SYNCING
        assert acct.connected_at == _NOW
        assert acct.last_sync == _LATER
        assert acct.total_messages == 10
        assert acct.unread_count == 2
        assert acct.sync_errors == 1
        assert acct.metadata == {"foo": "bar"}

    def test_record_to_account_minimal(self):
        rec = {"id": "acct-min", "provider": "gmail"}
        acct = record_to_account(rec)
        assert acct.id == "acct-min"
        assert acct.provider is EmailProvider.GMAIL
        assert acct.email_address == ""
        assert acct.display_name == ""
        assert acct.status is AccountStatus.PENDING
        assert isinstance(acct.connected_at, datetime)
        assert acct.last_sync is None
        assert acct.total_messages == 0
        assert acct.unread_count == 0
        assert acct.sync_errors == 0
        assert acct.metadata == {}

    def test_record_to_account_invalid_provider_raises(self):
        rec = {"id": "acct-bad", "provider": "yahoo"}
        with pytest.raises(ValueError):
            record_to_account(rec)

    def test_record_to_account_iso_string_dates(self):
        rec = {
            "id": "acct-iso",
            "provider": "gmail",
            "connected_at": "2026-02-23T12:00:00+00:00",
            "last_sync": "2026-02-24T08:30:00+00:00",
        }
        acct = record_to_account(rec)
        assert isinstance(acct.connected_at, datetime)
        assert isinstance(acct.last_sync, datetime)

    def test_record_to_account_null_metadata_becomes_dict(self):
        rec = {"id": "acct-x", "provider": "outlook", "metadata": None}
        acct = record_to_account(rec)
        assert acct.metadata == {}

    def test_roundtrip_account(self):
        original = _make_account()
        rec = account_to_record(original)
        restored = record_to_account(rec)
        assert restored.id == original.id
        assert restored.provider == original.provider
        assert restored.email_address == original.email_address
        assert restored.display_name == original.display_name
        assert restored.status == original.status
        assert restored.total_messages == original.total_messages
        assert restored.unread_count == original.unread_count
        assert restored.sync_errors == original.sync_errors
        assert restored.metadata == original.metadata


# ===========================================================================
# message_to_record / record_to_message
# ===========================================================================


class TestMessageRecordConversion:
    """Tests for message_to_record and record_to_message."""

    def test_message_to_record_fields(self):
        msg = _make_message()
        rec = message_to_record(msg)
        assert rec["id"] == "msg-1"
        assert rec["account_id"] == "acct-1"
        assert rec["provider"] == "gmail"
        assert rec["external_id"] == "ext-123"
        assert rec["subject"] == "Test Subject"
        assert rec["sender_email"] == "sender@example.com"
        assert rec["sender_name"] == "Sender Name"
        assert rec["recipients"] == ["user@gmail.com"]
        assert rec["cc"] == ["cc@example.com"]
        assert rec["received_at"] == _NOW
        assert rec["snippet"] == "Preview text"
        assert rec["body_preview"] == "Full body preview"
        assert rec["is_read"] is False
        assert rec["is_starred"] is True
        assert rec["has_attachments"] is True
        assert rec["labels"] == ["inbox", "important"]
        assert rec["thread_id"] == "thread-1"
        assert rec["priority_score"] == 0.75
        assert rec["priority_tier"] == "high"
        assert rec["priority_reasons"] == ["VIP sender"]
        assert rec["triage_action"] == "respond_urgent"
        assert rec["triage_rationale"] == "Important message from VIP"

    def test_message_to_record_no_triage(self):
        msg = _make_message(triage_action=None, triage_rationale=None)
        rec = message_to_record(msg)
        assert rec["triage_action"] is None
        assert rec["triage_rationale"] is None

    def test_record_to_message_full(self):
        rec = {
            "id": "msg-2",
            "account_id": "acct-2",
            "provider": "outlook",
            "external_id": "ext-456",
            "subject": "Outlook Message",
            "sender_email": "s@outlook.com",
            "sender_name": "Outlook Sender",
            "recipients": ["r@outlook.com"],
            "cc": ["cc1@e.com", "cc2@e.com"],
            "received_at": _NOW,
            "snippet": "snip",
            "body_preview": "body",
            "is_read": True,
            "is_starred": False,
            "has_attachments": True,
            "labels": ["inbox", "archive"],
            "thread_id": "thr-2",
            "priority_score": 0.3,
            "priority_tier": "low",
            "priority_reasons": ["newsletter"],
            "triage_action": "archive",
            "triage_rationale": "Auto-archived",
        }
        msg = record_to_message(rec)
        assert msg.id == "msg-2"
        assert msg.provider is EmailProvider.OUTLOOK
        assert msg.cc == ["cc1@e.com", "cc2@e.com"]
        assert msg.is_read is True
        assert msg.has_attachments is True
        assert msg.priority_score == 0.3
        assert msg.triage_action is TriageAction.ARCHIVE
        assert msg.triage_rationale == "Auto-archived"

    def test_record_to_message_minimal(self):
        rec = {
            "id": "msg-min",
            "account_id": "acct-1",
            "provider": "gmail",
        }
        msg = record_to_message(rec)
        assert msg.id == "msg-min"
        assert msg.external_id == ""
        assert msg.subject == ""
        assert msg.sender_email == ""
        assert msg.sender_name == ""
        assert msg.recipients == []
        assert msg.cc == []
        assert isinstance(msg.received_at, datetime)
        assert msg.snippet == ""
        assert msg.body_preview == ""
        assert msg.is_read is False
        assert msg.is_starred is False
        assert msg.has_attachments is False
        assert msg.labels == []
        assert msg.thread_id is None
        assert msg.priority_score == 0.0
        assert msg.priority_tier == "medium"
        assert msg.priority_reasons == []
        assert msg.triage_action is None
        assert msg.triage_rationale is None

    def test_record_to_message_null_lists_become_empty(self):
        rec = {
            "id": "msg-null",
            "account_id": "acct-1",
            "provider": "gmail",
            "recipients": None,
            "cc": None,
            "labels": None,
            "priority_reasons": None,
        }
        msg = record_to_message(rec)
        assert msg.recipients == []
        assert msg.cc == []
        assert msg.labels == []
        assert msg.priority_reasons == []

    def test_record_to_message_invalid_provider_raises(self):
        rec = {"id": "msg-bad", "account_id": "acct-1", "provider": "yahoo"}
        with pytest.raises(ValueError):
            record_to_message(rec)

    def test_record_to_message_triage_action_none_string(self):
        rec = {
            "id": "msg-no-triage",
            "account_id": "acct-1",
            "provider": "gmail",
            "triage_action": None,
        }
        msg = record_to_message(rec)
        assert msg.triage_action is None

    def test_record_to_message_each_triage_action(self):
        for action in TriageAction:
            rec = {
                "id": f"msg-{action.value}",
                "account_id": "acct-1",
                "provider": "gmail",
                "triage_action": action.value,
            }
            msg = record_to_message(rec)
            assert msg.triage_action is action

    def test_roundtrip_message(self):
        original = _make_message()
        rec = message_to_record(original)
        restored = record_to_message(rec)
        assert restored.id == original.id
        assert restored.account_id == original.account_id
        assert restored.provider == original.provider
        assert restored.external_id == original.external_id
        assert restored.subject == original.subject
        assert restored.sender_email == original.sender_email
        assert restored.sender_name == original.sender_name
        assert restored.recipients == original.recipients
        assert restored.cc == original.cc
        assert restored.snippet == original.snippet
        assert restored.body_preview == original.body_preview
        assert restored.is_read == original.is_read
        assert restored.is_starred == original.is_starred
        assert restored.has_attachments == original.has_attachments
        assert restored.labels == original.labels
        assert restored.thread_id == original.thread_id
        assert restored.priority_score == original.priority_score
        assert restored.priority_tier == original.priority_tier
        assert restored.priority_reasons == original.priority_reasons
        assert restored.triage_action == original.triage_action
        assert restored.triage_rationale == original.triage_rationale

    def test_roundtrip_message_no_triage(self):
        original = _make_message(triage_action=None, triage_rationale=None)
        rec = message_to_record(original)
        restored = record_to_message(rec)
        assert restored.triage_action is None
        assert restored.triage_rationale is None


# ===========================================================================
# triage_to_record / record_to_triage
# ===========================================================================


class TestTriageRecordConversion:
    """Tests for triage_to_record and record_to_triage."""

    def test_triage_to_record_fields(self):
        t = _make_triage()
        rec = triage_to_record(t)
        assert rec["message_id"] == "msg-1"
        assert rec["recommended_action"] == "defer"
        assert rec["confidence"] == 0.85
        assert rec["rationale"] == "Low priority content"
        assert rec["suggested_response"] == "Will follow up later"
        assert rec["delegate_to"] == "team-lead"
        assert rec["schedule_for"] == _LATER
        assert rec["agents_involved"] == ["analyst", "expert"]
        assert rec["debate_summary"] == "Agents agreed to defer"
        assert "created_at" in rec
        assert isinstance(rec["created_at"], datetime)

    def test_triage_to_record_none_optional_fields(self):
        t = _make_triage(
            suggested_response=None,
            delegate_to=None,
            schedule_for=None,
            debate_summary=None,
        )
        rec = triage_to_record(t)
        assert rec["suggested_response"] is None
        assert rec["delegate_to"] is None
        assert rec["schedule_for"] is None
        assert rec["debate_summary"] is None

    def test_triage_to_record_each_action(self):
        for action in TriageAction:
            t = _make_triage(recommended_action=action)
            rec = triage_to_record(t)
            assert rec["recommended_action"] == action.value

    def test_record_to_triage_full(self):
        rec = {
            "message_id": "msg-3",
            "recommended_action": "flag",
            "confidence": 0.92,
            "rationale": "Flagged for review",
            "suggested_response": "Review and respond",
            "delegate_to": "manager",
            "schedule_for": _LATER,
            "agents_involved": ["reviewer", "analyst"],
            "debate_summary": "Consensus reached",
        }
        t = record_to_triage(rec)
        assert t.message_id == "msg-3"
        assert t.recommended_action is TriageAction.FLAG
        assert t.confidence == 0.92
        assert t.rationale == "Flagged for review"
        assert t.suggested_response == "Review and respond"
        assert t.delegate_to == "manager"
        assert t.schedule_for == _LATER
        assert t.agents_involved == ["reviewer", "analyst"]
        assert t.debate_summary == "Consensus reached"

    def test_record_to_triage_minimal(self):
        rec = {
            "message_id": "msg-min",
            "recommended_action": "delete",
        }
        t = record_to_triage(rec)
        assert t.message_id == "msg-min"
        assert t.recommended_action is TriageAction.DELETE
        assert t.confidence == 0.0
        assert t.rationale == ""
        assert t.suggested_response is None
        assert t.delegate_to is None
        assert t.schedule_for is None
        assert t.agents_involved == []
        assert t.debate_summary is None

    def test_record_to_triage_null_agents_becomes_list(self):
        rec = {
            "message_id": "msg-x",
            "recommended_action": "defer",
            "agents_involved": None,
        }
        t = record_to_triage(rec)
        assert t.agents_involved == []

    def test_record_to_triage_invalid_action_raises(self):
        rec = {
            "message_id": "msg-bad",
            "recommended_action": "explode",
        }
        with pytest.raises(ValueError):
            record_to_triage(rec)

    def test_record_to_triage_schedule_for_iso_string(self):
        rec = {
            "message_id": "msg-sched",
            "recommended_action": "schedule",
            "schedule_for": "2026-02-24T08:30:00+00:00",
        }
        t = record_to_triage(rec)
        assert isinstance(t.schedule_for, datetime)

    def test_record_to_triage_schedule_for_none(self):
        rec = {
            "message_id": "msg-no-sched",
            "recommended_action": "defer",
            "schedule_for": None,
        }
        t = record_to_triage(rec)
        assert t.schedule_for is None

    def test_record_to_triage_schedule_for_invalid(self):
        rec = {
            "message_id": "msg-bad-sched",
            "recommended_action": "defer",
            "schedule_for": "not-a-date",
        }
        t = record_to_triage(rec)
        assert t.schedule_for is None

    def test_roundtrip_triage(self):
        original = _make_triage()
        rec = triage_to_record(original)
        # Remove created_at which is auto-generated, not in the dataclass
        restored = record_to_triage(rec)
        assert restored.message_id == original.message_id
        assert restored.recommended_action == original.recommended_action
        assert restored.confidence == original.confidence
        assert restored.rationale == original.rationale
        assert restored.suggested_response == original.suggested_response
        assert restored.delegate_to == original.delegate_to
        assert restored.schedule_for == original.schedule_for
        assert restored.agents_involved == original.agents_involved
        assert restored.debate_summary == original.debate_summary


# ===========================================================================
# Edge Cases and Security
# ===========================================================================


class TestEdgeCases:
    """Edge cases for serialization and conversion."""

    def test_empty_string_fields(self):
        acct = _make_account(
            email_address="",
            display_name="",
        )
        d = acct.to_dict()
        assert d["email_address"] == ""
        assert d["display_name"] == ""

    def test_unicode_in_subject(self):
        msg = _make_message(subject="Hello World")
        d = msg.to_dict()
        assert d["subject"] == "Hello World"

    def test_html_in_subject_not_escaped(self):
        msg = _make_message(subject="<script>alert('xss')</script>")
        d = msg.to_dict()
        # Model layer doesn't sanitize - it preserves data as-is
        assert "<script>" in d["subject"]

    def test_very_long_string_preserved(self):
        long_str = "a" * 10000
        msg = _make_message(subject=long_str)
        d = msg.to_dict()
        assert len(d["subject"]) == 10000

    def test_special_characters_in_email(self):
        acct = _make_account(email_address="user+tag@gmail.com")
        rec = account_to_record(acct)
        restored = record_to_account(rec)
        assert restored.email_address == "user+tag@gmail.com"

    def test_path_traversal_in_id(self):
        msg = _make_message(id="../../etc/passwd")
        rec = message_to_record(msg)
        restored = record_to_message(rec)
        assert restored.id == "../../etc/passwd"

    def test_sql_injection_in_subject(self):
        msg = _make_message(subject="'; DROP TABLE messages; --")
        d = msg.to_dict()
        assert d["subject"] == "'; DROP TABLE messages; --"

    def test_newlines_in_snippet(self):
        msg = _make_message(snippet="line1\nline2\rline3")
        d = msg.to_dict()
        assert "\n" in d["snippet"]

    def test_empty_lists_preserved(self):
        msg = _make_message(
            recipients=[],
            cc=[],
            labels=[],
            priority_reasons=[],
        )
        d = msg.to_dict()
        assert d["recipients"] == []
        assert d["cc"] == []
        assert d["labels"] == []
        assert d["priority"]["reasons"] == []

    def test_large_priority_score(self):
        msg = _make_message(priority_score=999.99)
        d = msg.to_dict()
        assert d["priority"]["score"] == 999.99

    def test_negative_priority_score(self):
        msg = _make_message(priority_score=-1.0)
        d = msg.to_dict()
        assert d["priority"]["score"] == -1.0

    def test_zero_confidence_triage(self):
        t = _make_triage(confidence=0.0)
        d = t.to_dict()
        assert d["confidence"] == 0.0

    def test_confidence_above_one(self):
        t = _make_triage(confidence=1.5)
        d = t.to_dict()
        assert d["confidence"] == 1.5

    def test_large_agents_involved_list(self):
        agents = [f"agent-{i}" for i in range(100)]
        t = _make_triage(agents_involved=agents)
        d = t.to_dict()
        assert len(d["agents_involved"]) == 100

    def test_stats_zero_values(self):
        s = _make_stats(
            total_accounts=0,
            total_messages=0,
            unread_count=0,
            avg_response_time_hours=0.0,
            pending_triage=0,
        )
        d = s.to_dict()
        assert d["total_accounts"] == 0
        assert d["total_messages"] == 0
        assert d["unread_count"] == 0
        assert d["avg_response_time_hours"] == 0.0
        assert d["pending_triage"] == 0

    def test_account_with_metadata(self):
        meta = {"oauth_token": "secret", "nested": {"key": "value"}}
        acct = _make_account(metadata=meta)
        rec = account_to_record(acct)
        assert rec["metadata"] == meta

    def test_record_to_message_bool_coercion(self):
        """Verify boolean fields correctly coerce falsy/truthy values."""
        rec = {
            "id": "msg-bool",
            "account_id": "acct-1",
            "provider": "gmail",
            "is_read": 0,
            "is_starred": 1,
            "has_attachments": "",
        }
        msg = record_to_message(rec)
        assert msg.is_read is False
        assert msg.is_starred is True
        assert msg.has_attachments is False

    def test_record_to_account_int_coercion(self):
        """Verify integer fields coerce string values."""
        rec = {
            "id": "acct-coerce",
            "provider": "gmail",
            "total_messages": "100",
            "unread_count": "5",
            "sync_errors": "2",
        }
        acct = record_to_account(rec)
        assert acct.total_messages == 100
        assert acct.unread_count == 5
        assert acct.sync_errors == 2

    def test_record_to_message_float_coercion(self):
        """Verify float fields coerce string values."""
        rec = {
            "id": "msg-float",
            "account_id": "acct-1",
            "provider": "gmail",
            "priority_score": "0.99",
        }
        msg = record_to_message(rec)
        assert msg.priority_score == pytest.approx(0.99)

    def test_record_to_triage_float_coercion(self):
        """Verify confidence coerces from string."""
        rec = {
            "message_id": "msg-conf",
            "recommended_action": "defer",
            "confidence": "0.77",
        }
        t = record_to_triage(rec)
        assert t.confidence == pytest.approx(0.77)
