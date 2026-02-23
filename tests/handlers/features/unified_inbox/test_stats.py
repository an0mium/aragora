"""Tests for the Unified Inbox statistics module.

Covers both public functions in:
    aragora/server/handlers/features/unified_inbox/stats.py

Test areas:
- compute_stats: basic aggregation, unread counting, priority distribution,
  provider distribution, top senders, sync health, pending triage, edge cases
- compute_trends: structure, period_days values, keys, static values
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from aragora.server.handlers.features.unified_inbox.models import (
    AccountStatus,
    EmailProvider,
    TriageAction,
    UnifiedMessage,
)
from aragora.server.handlers.features.unified_inbox.stats import (
    compute_stats,
    compute_trends,
)


# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)


def _make_account_record(**overrides: Any) -> dict[str, Any]:
    """Build a raw account record dict with sensible defaults."""
    defaults: dict[str, Any] = {
        "id": "acct-1",
        "provider": "gmail",
        "email_address": "user@gmail.com",
        "display_name": "Test User",
        "status": "connected",
        "connected_at": _NOW,
        "last_sync": _NOW,
        "total_messages": 42,
        "unread_count": 5,
        "sync_errors": 0,
        "metadata": {},
    }
    defaults.update(overrides)
    return defaults


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
        "cc": [],
        "received_at": _NOW,
        "snippet": "Preview text",
        "body_preview": "Full body preview",
        "is_read": False,
        "is_starred": False,
        "has_attachments": False,
        "labels": ["inbox"],
        "thread_id": None,
        "priority_score": 0.5,
        "priority_tier": "medium",
        "priority_reasons": [],
        "triage_action": None,
        "triage_rationale": None,
    }
    defaults.update(overrides)
    return UnifiedMessage(**defaults)


# ===========================================================================
# compute_stats - Basic Behavior
# ===========================================================================


class TestComputeStatsBasic:
    """Tests for basic compute_stats behavior."""

    def test_empty_inputs(self):
        """Empty account list and empty message list produce zeroed stats."""
        stats = compute_stats([], [])
        assert stats.total_accounts == 0
        assert stats.total_messages == 0
        assert stats.unread_count == 0
        assert stats.pending_triage == 0

    def test_total_accounts_count(self):
        """Total accounts matches the number of account records passed."""
        records = [
            _make_account_record(id="acct-1"),
            _make_account_record(id="acct-2"),
            _make_account_record(id="acct-3"),
        ]
        stats = compute_stats(records, [])
        assert stats.total_accounts == 3

    def test_total_messages_count(self):
        """Total messages matches the number of messages passed."""
        messages = [_make_message(id=f"msg-{i}") for i in range(7)]
        stats = compute_stats([], messages)
        assert stats.total_messages == 7

    def test_single_account_single_message(self):
        """Minimal valid input with one account and one message."""
        records = [_make_account_record()]
        messages = [_make_message()]
        stats = compute_stats(records, messages)
        assert stats.total_accounts == 1
        assert stats.total_messages == 1

    def test_avg_response_time_is_hardcoded(self):
        """Average response time is currently a hardcoded value."""
        stats = compute_stats([], [])
        assert stats.avg_response_time_hours == 4.5

    def test_hourly_volume_is_empty_list(self):
        """Hourly volume is currently an empty list placeholder."""
        stats = compute_stats([], [])
        assert stats.hourly_volume == []

    def test_stats_is_inbox_stats_instance(self):
        """Return type is an InboxStats dataclass."""
        from aragora.server.handlers.features.unified_inbox.models import InboxStats

        stats = compute_stats([], [])
        assert isinstance(stats, InboxStats)


# ===========================================================================
# compute_stats - Unread Count
# ===========================================================================


class TestComputeStatsUnread:
    """Tests for unread counting in compute_stats."""

    def test_all_unread(self):
        """All messages unread yields total count."""
        messages = [_make_message(id=f"msg-{i}", is_read=False) for i in range(5)]
        stats = compute_stats([], messages)
        assert stats.unread_count == 5

    def test_all_read(self):
        """All messages read yields zero unread."""
        messages = [_make_message(id=f"msg-{i}", is_read=True) for i in range(5)]
        stats = compute_stats([], messages)
        assert stats.unread_count == 0

    def test_mixed_read_unread(self):
        """Mixture of read and unread messages counted correctly."""
        messages = [
            _make_message(id="msg-1", is_read=True),
            _make_message(id="msg-2", is_read=False),
            _make_message(id="msg-3", is_read=True),
            _make_message(id="msg-4", is_read=False),
        ]
        stats = compute_stats([], messages)
        assert stats.unread_count == 2

    def test_no_messages_zero_unread(self):
        """No messages means zero unread."""
        stats = compute_stats([], [])
        assert stats.unread_count == 0


# ===========================================================================
# compute_stats - Priority Distribution
# ===========================================================================


class TestComputeStatsPriority:
    """Tests for messages_by_priority in compute_stats."""

    def test_all_priority_tiers_present(self):
        """Result always contains all four priority keys."""
        stats = compute_stats([], [])
        assert set(stats.messages_by_priority.keys()) == {"critical", "high", "medium", "low"}

    def test_all_tiers_zero_when_no_messages(self):
        """All priority counts are zero with no messages."""
        stats = compute_stats([], [])
        for tier in ("critical", "high", "medium", "low"):
            assert stats.messages_by_priority[tier] == 0

    def test_single_critical_message(self):
        """One critical message shows count 1 for critical, 0 for others."""
        messages = [_make_message(priority_tier="critical")]
        stats = compute_stats([], messages)
        assert stats.messages_by_priority["critical"] == 1
        assert stats.messages_by_priority["high"] == 0
        assert stats.messages_by_priority["medium"] == 0
        assert stats.messages_by_priority["low"] == 0

    def test_distribution_across_tiers(self):
        """Multiple messages distributed across tiers counted correctly."""
        messages = [
            _make_message(id="m1", priority_tier="critical"),
            _make_message(id="m2", priority_tier="critical"),
            _make_message(id="m3", priority_tier="high"),
            _make_message(id="m4", priority_tier="medium"),
            _make_message(id="m5", priority_tier="medium"),
            _make_message(id="m6", priority_tier="medium"),
            _make_message(id="m7", priority_tier="low"),
        ]
        stats = compute_stats([], messages)
        assert stats.messages_by_priority["critical"] == 2
        assert stats.messages_by_priority["high"] == 1
        assert stats.messages_by_priority["medium"] == 3
        assert stats.messages_by_priority["low"] == 1

    def test_unknown_priority_tier_not_counted(self):
        """A message with an unrecognized tier is not counted in any bucket."""
        messages = [_make_message(priority_tier="urgent")]
        stats = compute_stats([], messages)
        assert stats.messages_by_priority["critical"] == 0
        assert stats.messages_by_priority["high"] == 0
        assert stats.messages_by_priority["medium"] == 0
        assert stats.messages_by_priority["low"] == 0

    def test_all_same_tier(self):
        """All messages in the same tier produces correct count."""
        messages = [_make_message(id=f"m{i}", priority_tier="high") for i in range(10)]
        stats = compute_stats([], messages)
        assert stats.messages_by_priority["high"] == 10
        assert stats.messages_by_priority["critical"] == 0
        assert stats.messages_by_priority["medium"] == 0
        assert stats.messages_by_priority["low"] == 0


# ===========================================================================
# compute_stats - Provider Distribution
# ===========================================================================


class TestComputeStatsProvider:
    """Tests for messages_by_provider in compute_stats."""

    def test_both_providers_present(self):
        """Result always contains gmail and outlook keys."""
        stats = compute_stats([], [])
        assert set(stats.messages_by_provider.keys()) == {"gmail", "outlook"}

    def test_all_zero_when_no_messages(self):
        """Both provider counts are zero with no messages."""
        stats = compute_stats([], [])
        assert stats.messages_by_provider["gmail"] == 0
        assert stats.messages_by_provider["outlook"] == 0

    def test_gmail_only(self):
        """Messages only from Gmail counted correctly."""
        messages = [
            _make_message(id="m1", provider=EmailProvider.GMAIL),
            _make_message(id="m2", provider=EmailProvider.GMAIL),
        ]
        stats = compute_stats([], messages)
        assert stats.messages_by_provider["gmail"] == 2
        assert stats.messages_by_provider["outlook"] == 0

    def test_outlook_only(self):
        """Messages only from Outlook counted correctly."""
        messages = [
            _make_message(id="m1", provider=EmailProvider.OUTLOOK),
        ]
        stats = compute_stats([], messages)
        assert stats.messages_by_provider["gmail"] == 0
        assert stats.messages_by_provider["outlook"] == 1

    def test_mixed_providers(self):
        """Mixed provider messages counted correctly."""
        messages = [
            _make_message(id="m1", provider=EmailProvider.GMAIL),
            _make_message(id="m2", provider=EmailProvider.OUTLOOK),
            _make_message(id="m3", provider=EmailProvider.GMAIL),
            _make_message(id="m4", provider=EmailProvider.OUTLOOK),
            _make_message(id="m5", provider=EmailProvider.OUTLOOK),
        ]
        stats = compute_stats([], messages)
        assert stats.messages_by_provider["gmail"] == 2
        assert stats.messages_by_provider["outlook"] == 3


# ===========================================================================
# compute_stats - Top Senders
# ===========================================================================


class TestComputeStatsTopSenders:
    """Tests for top_senders in compute_stats."""

    def test_no_messages_empty_senders(self):
        """No messages yields empty top_senders list."""
        stats = compute_stats([], [])
        assert stats.top_senders == []

    def test_single_sender(self):
        """One message yields one top sender entry."""
        messages = [_make_message(sender_email="alice@example.com")]
        stats = compute_stats([], messages)
        assert len(stats.top_senders) == 1
        assert stats.top_senders[0]["email"] == "alice@example.com"
        assert stats.top_senders[0]["count"] == 1

    def test_multiple_senders_sorted_descending(self):
        """Multiple senders sorted by count descending."""
        messages = [
            _make_message(id="m1", sender_email="alice@example.com"),
            _make_message(id="m2", sender_email="bob@example.com"),
            _make_message(id="m3", sender_email="bob@example.com"),
            _make_message(id="m4", sender_email="carol@example.com"),
            _make_message(id="m5", sender_email="carol@example.com"),
            _make_message(id="m6", sender_email="carol@example.com"),
        ]
        stats = compute_stats([], messages)
        assert len(stats.top_senders) == 3
        assert stats.top_senders[0]["email"] == "carol@example.com"
        assert stats.top_senders[0]["count"] == 3
        assert stats.top_senders[1]["email"] == "bob@example.com"
        assert stats.top_senders[1]["count"] == 2
        assert stats.top_senders[2]["email"] == "alice@example.com"
        assert stats.top_senders[2]["count"] == 1

    def test_top_senders_limited_to_five(self):
        """Only the top 5 senders are returned even with more unique senders."""
        messages = []
        senders = [f"sender{i}@example.com" for i in range(8)]
        for i, sender in enumerate(senders):
            for _ in range(8 - i):
                messages.append(
                    _make_message(id=f"m-{sender}-{_}", sender_email=sender)
                )
        stats = compute_stats([], messages)
        assert len(stats.top_senders) == 5

    def test_top_senders_structure(self):
        """Each top sender entry has email and count keys."""
        messages = [_make_message(sender_email="test@example.com")]
        stats = compute_stats([], messages)
        entry = stats.top_senders[0]
        assert "email" in entry
        assert "count" in entry
        assert isinstance(entry["email"], str)
        assert isinstance(entry["count"], int)

    def test_same_sender_multiple_messages(self):
        """Same sender across many messages produces single entry with total count."""
        messages = [
            _make_message(id=f"m{i}", sender_email="frequent@example.com")
            for i in range(20)
        ]
        stats = compute_stats([], messages)
        assert len(stats.top_senders) == 1
        assert stats.top_senders[0]["email"] == "frequent@example.com"
        assert stats.top_senders[0]["count"] == 20


# ===========================================================================
# compute_stats - Sync Health
# ===========================================================================


class TestComputeStatsSyncHealth:
    """Tests for sync_health in compute_stats."""

    def test_no_accounts_zero_health(self):
        """No accounts yields all zero sync health."""
        stats = compute_stats([], [])
        assert stats.sync_health["accounts_healthy"] == 0
        assert stats.sync_health["accounts_error"] == 0
        assert stats.sync_health["total_sync_errors"] == 0

    def test_all_healthy_accounts(self):
        """All connected accounts counted as healthy."""
        records = [
            _make_account_record(id="acct-1", status="connected"),
            _make_account_record(id="acct-2", status="connected"),
        ]
        stats = compute_stats(records, [])
        assert stats.sync_health["accounts_healthy"] == 2
        assert stats.sync_health["accounts_error"] == 0

    def test_all_error_accounts(self):
        """All error accounts counted as errors."""
        records = [
            _make_account_record(id="acct-1", status="error", sync_errors=3),
            _make_account_record(id="acct-2", status="error", sync_errors=5),
        ]
        stats = compute_stats(records, [])
        assert stats.sync_health["accounts_healthy"] == 0
        assert stats.sync_health["accounts_error"] == 2
        assert stats.sync_health["total_sync_errors"] == 8

    def test_mixed_health_statuses(self):
        """Mix of connected, error, and other statuses."""
        records = [
            _make_account_record(id="acct-1", status="connected", sync_errors=0),
            _make_account_record(id="acct-2", status="error", sync_errors=2),
            _make_account_record(id="acct-3", status="syncing", sync_errors=1),
        ]
        stats = compute_stats(records, [])
        assert stats.sync_health["accounts_healthy"] == 1
        assert stats.sync_health["accounts_error"] == 1
        assert stats.sync_health["total_sync_errors"] == 3

    def test_sync_health_keys(self):
        """Sync health dict has the expected keys."""
        stats = compute_stats([], [])
        assert set(stats.sync_health.keys()) == {
            "accounts_healthy",
            "accounts_error",
            "total_sync_errors",
        }

    def test_pending_and_disconnected_not_counted_as_healthy_or_error(self):
        """Pending and disconnected accounts are neither healthy nor error."""
        records = [
            _make_account_record(id="acct-1", status="pending"),
            _make_account_record(id="acct-2", status="disconnected"),
        ]
        stats = compute_stats(records, [])
        assert stats.sync_health["accounts_healthy"] == 0
        assert stats.sync_health["accounts_error"] == 0


# ===========================================================================
# compute_stats - Pending Triage
# ===========================================================================


class TestComputeStatsPendingTriage:
    """Tests for pending_triage in compute_stats."""

    def test_no_messages_zero_pending(self):
        """No messages yields zero pending triage."""
        stats = compute_stats([], [])
        assert stats.pending_triage == 0

    def test_unread_without_triage_is_pending(self):
        """Unread message with no triage_action counts as pending."""
        messages = [_make_message(is_read=False, triage_action=None)]
        stats = compute_stats([], messages)
        assert stats.pending_triage == 1

    def test_read_without_triage_not_pending(self):
        """Read message with no triage_action does not count as pending."""
        messages = [_make_message(is_read=True, triage_action=None)]
        stats = compute_stats([], messages)
        assert stats.pending_triage == 0

    def test_unread_with_triage_not_pending(self):
        """Unread message that already has a triage_action is not pending."""
        messages = [
            _make_message(is_read=False, triage_action=TriageAction.ARCHIVE)
        ]
        stats = compute_stats([], messages)
        assert stats.pending_triage == 0

    def test_read_with_triage_not_pending(self):
        """Read message with triage_action is not pending."""
        messages = [
            _make_message(is_read=True, triage_action=TriageAction.RESPOND_URGENT)
        ]
        stats = compute_stats([], messages)
        assert stats.pending_triage == 0

    def test_multiple_pending_counted(self):
        """Multiple pending triage messages counted correctly."""
        messages = [
            _make_message(id="m1", is_read=False, triage_action=None),
            _make_message(id="m2", is_read=False, triage_action=None),
            _make_message(id="m3", is_read=True, triage_action=None),
            _make_message(id="m4", is_read=False, triage_action=TriageAction.DEFER),
        ]
        stats = compute_stats([], messages)
        assert stats.pending_triage == 2


# ===========================================================================
# compute_stats - Serialization via to_dict
# ===========================================================================


class TestComputeStatsToDict:
    """Tests that compute_stats result serializes correctly."""

    def test_to_dict_has_all_keys(self):
        """Serialized stats dict has all expected keys."""
        stats = compute_stats([], [])
        d = stats.to_dict()
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

    def test_to_dict_values_match(self):
        """Serialized values match computed attributes."""
        records = [_make_account_record(status="connected")]
        messages = [
            _make_message(id="m1", is_read=False, priority_tier="high"),
            _make_message(id="m2", is_read=True, priority_tier="low"),
        ]
        stats = compute_stats(records, messages)
        d = stats.to_dict()
        assert d["total_accounts"] == 1
        assert d["total_messages"] == 2
        assert d["unread_count"] == 1
        assert d["messages_by_priority"]["high"] == 1
        assert d["messages_by_priority"]["low"] == 1


# ===========================================================================
# compute_trends - Structure and Values
# ===========================================================================


class TestComputeTrends:
    """Tests for compute_trends function."""

    def test_period_days_matches_input(self):
        """period_days in result matches the days argument."""
        result = compute_trends(7)
        assert result["period_days"] == 7

    def test_period_days_different_values(self):
        """period_days reflects various input values."""
        for days in (1, 14, 30, 90, 365):
            result = compute_trends(days)
            assert result["period_days"] == days

    def test_top_level_keys(self):
        """Result has the expected top-level keys."""
        result = compute_trends(7)
        assert set(result.keys()) == {
            "period_days",
            "priority_trends",
            "volume_trend",
            "response_time_trend",
        }

    def test_priority_trends_keys(self):
        """Priority trends has all four tier keys."""
        result = compute_trends(7)
        assert set(result["priority_trends"].keys()) == {
            "critical",
            "high",
            "medium",
            "low",
        }

    def test_priority_trend_entry_structure(self):
        """Each priority trend entry has current, previous, change_pct."""
        result = compute_trends(7)
        for tier in ("critical", "high", "medium", "low"):
            entry = result["priority_trends"][tier]
            assert "current" in entry
            assert "previous" in entry
            assert "change_pct" in entry

    def test_priority_trend_critical_values(self):
        """Critical trend has the expected static values."""
        result = compute_trends(7)
        crit = result["priority_trends"]["critical"]
        assert crit["current"] == 5
        assert crit["previous"] == 8
        assert crit["change_pct"] == -37.5

    def test_priority_trend_high_values(self):
        """High trend has the expected static values."""
        result = compute_trends(7)
        high = result["priority_trends"]["high"]
        assert high["current"] == 15
        assert high["previous"] == 12
        assert high["change_pct"] == 25.0

    def test_priority_trend_medium_values(self):
        """Medium trend has the expected static values."""
        result = compute_trends(7)
        med = result["priority_trends"]["medium"]
        assert med["current"] == 45
        assert med["previous"] == 42
        assert med["change_pct"] == 7.1

    def test_priority_trend_low_values(self):
        """Low trend has the expected static values."""
        result = compute_trends(7)
        low = result["priority_trends"]["low"]
        assert low["current"] == 35
        assert low["previous"] == 38
        assert low["change_pct"] == -7.9

    def test_volume_trend_structure(self):
        """Volume trend has the expected keys."""
        result = compute_trends(7)
        vt = result["volume_trend"]
        assert set(vt.keys()) == {
            "current_daily_avg",
            "previous_daily_avg",
            "change_pct",
        }

    def test_volume_trend_values(self):
        """Volume trend has the expected static values."""
        result = compute_trends(7)
        vt = result["volume_trend"]
        assert vt["current_daily_avg"] == 25
        assert vt["previous_daily_avg"] == 22
        assert vt["change_pct"] == 13.6

    def test_response_time_trend_structure(self):
        """Response time trend has the expected keys."""
        result = compute_trends(7)
        rt = result["response_time_trend"]
        assert set(rt.keys()) == {
            "current_avg_hours",
            "previous_avg_hours",
            "change_pct",
        }

    def test_response_time_trend_values(self):
        """Response time trend has the expected static values."""
        result = compute_trends(7)
        rt = result["response_time_trend"]
        assert rt["current_avg_hours"] == 4.2
        assert rt["previous_avg_hours"] == 5.1
        assert rt["change_pct"] == -17.6

    def test_zero_days(self):
        """Zero days input still returns valid structure."""
        result = compute_trends(0)
        assert result["period_days"] == 0
        assert "priority_trends" in result

    def test_negative_days(self):
        """Negative days input still returns valid structure."""
        result = compute_trends(-1)
        assert result["period_days"] == -1
        assert "priority_trends" in result

    def test_large_days(self):
        """Very large days value handled gracefully."""
        result = compute_trends(100000)
        assert result["period_days"] == 100000


# ===========================================================================
# compute_stats - Edge Cases
# ===========================================================================


class TestComputeStatsEdgeCases:
    """Edge cases for compute_stats."""

    def test_large_message_list(self):
        """Stats computed correctly for a large set of messages."""
        messages = [
            _make_message(
                id=f"msg-{i}",
                is_read=(i % 3 == 0),
                priority_tier=["critical", "high", "medium", "low"][i % 4],
                provider=[EmailProvider.GMAIL, EmailProvider.OUTLOOK][i % 2],
                sender_email=f"sender{i % 10}@example.com",
            )
            for i in range(100)
        ]
        stats = compute_stats([], messages)
        assert stats.total_messages == 100
        # Every 3rd message is read, so unread = 100 - 34 = 66
        expected_read = sum(1 for i in range(100) if i % 3 == 0)
        assert stats.unread_count == 100 - expected_read
        # Priority: 100/4 = 25 each
        assert stats.messages_by_priority["critical"] == 25
        assert stats.messages_by_priority["high"] == 25
        assert stats.messages_by_priority["medium"] == 25
        assert stats.messages_by_priority["low"] == 25
        # Provider: 50/50
        assert stats.messages_by_provider["gmail"] == 50
        assert stats.messages_by_provider["outlook"] == 50
        # Top senders: 10 unique but only top 5
        assert len(stats.top_senders) == 5
        # Each of the 10 senders has 10 messages, top 5 all have 10
        assert stats.top_senders[0]["count"] == 10

    def test_account_with_high_sync_errors(self):
        """Large sync_errors value summed correctly."""
        records = [
            _make_account_record(id="acct-1", status="error", sync_errors=999),
        ]
        stats = compute_stats(records, [])
        assert stats.sync_health["total_sync_errors"] == 999

    def test_empty_sender_email(self):
        """Message with empty sender_email still counted."""
        messages = [_make_message(sender_email="")]
        stats = compute_stats([], messages)
        assert len(stats.top_senders) == 1
        assert stats.top_senders[0]["email"] == ""
        assert stats.top_senders[0]["count"] == 1

    def test_all_triaged_zero_pending(self):
        """Messages that all have triage actions result in zero pending."""
        messages = [
            _make_message(id="m1", is_read=False, triage_action=TriageAction.ARCHIVE),
            _make_message(id="m2", is_read=False, triage_action=TriageAction.DEFER),
            _make_message(id="m3", is_read=False, triage_action=TriageAction.FLAG),
        ]
        stats = compute_stats([], messages)
        assert stats.pending_triage == 0

    def test_account_record_with_missing_optional_fields(self):
        """Minimal account record (only id and provider) processes without error."""
        records = [{"id": "acct-min", "provider": "gmail"}]
        stats = compute_stats(records, [])
        assert stats.total_accounts == 1
        # Default status is "pending" which is neither connected nor error
        assert stats.sync_health["accounts_healthy"] == 0
        assert stats.sync_health["accounts_error"] == 0
