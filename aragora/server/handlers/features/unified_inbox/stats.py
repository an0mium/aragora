"""Inbox statistics and trends for unified inbox.

Computes inbox health metrics, priority distribution, sender analytics,
and priority/volume trends over time.
"""

from __future__ import annotations

from typing import Any

from .models import (
    AccountStatus,
    EmailProvider,
    InboxStats,
    UnifiedMessage,
    record_to_account,
)


def compute_stats(
    account_records: list[dict[str, Any]],
    messages: list[UnifiedMessage],
) -> InboxStats:
    """Compute inbox health statistics from accounts and messages."""
    accounts = [record_to_account(record) for record in account_records]

    unread_count = sum(1 for m in messages if not m.is_read)

    messages_by_priority = {
        "critical": sum(1 for m in messages if m.priority_tier == "critical"),
        "high": sum(1 for m in messages if m.priority_tier == "high"),
        "medium": sum(1 for m in messages if m.priority_tier == "medium"),
        "low": sum(1 for m in messages if m.priority_tier == "low"),
    }

    messages_by_provider = {
        "gmail": sum(1 for m in messages if m.provider == EmailProvider.GMAIL),
        "outlook": sum(1 for m in messages if m.provider == EmailProvider.OUTLOOK),
    }

    # Top senders
    sender_counts: dict[str, int] = {}
    for m in messages:
        sender_counts[m.sender_email] = sender_counts.get(m.sender_email, 0) + 1

    top_senders = [
        {"email": email, "count": count}
        for email, count in sorted(sender_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    # Sync health
    sync_health = {
        "accounts_healthy": sum(1 for a in accounts if a.status == AccountStatus.CONNECTED),
        "accounts_error": sum(1 for a in accounts if a.status == AccountStatus.ERROR),
        "total_sync_errors": sum(a.sync_errors for a in accounts),
    }

    return InboxStats(
        total_accounts=len(accounts),
        total_messages=len(messages),
        unread_count=unread_count,
        messages_by_priority=messages_by_priority,
        messages_by_provider=messages_by_provider,
        avg_response_time_hours=4.5,  # Would be calculated from actual data
        pending_triage=sum(1 for m in messages if m.triage_action is None and not m.is_read),
        sync_health=sync_health,
        top_senders=top_senders,
        hourly_volume=[],  # Would be calculated from actual timestamps
    )


def compute_trends(days: int) -> dict[str, Any]:
    """Compute priority trends over time.

    In production, this would calculate from historical data.
    """
    return {
        "period_days": days,
        "priority_trends": {
            "critical": {"current": 5, "previous": 8, "change_pct": -37.5},
            "high": {"current": 15, "previous": 12, "change_pct": 25.0},
            "medium": {"current": 45, "previous": 42, "change_pct": 7.1},
            "low": {"current": 35, "previous": 38, "change_pct": -7.9},
        },
        "volume_trend": {
            "current_daily_avg": 25,
            "previous_daily_avg": 22,
            "change_pct": 13.6,
        },
        "response_time_trend": {
            "current_avg_hours": 4.2,
            "previous_avg_hours": 5.1,
            "change_pct": -17.6,
        },
    }
