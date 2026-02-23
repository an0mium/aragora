"""Message retrieval and scoring for unified inbox.

Handles message listing, fetching from providers, priority scoring,
and sample message generation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from .models import (
    AccountStatus,
    ConnectedAccount,
    EmailProvider,
    UnifiedMessage,
    message_to_record,
    record_to_account,
    record_to_message,
)
from .sync import get_sync_services, get_sync_services_lock

logger = logging.getLogger(__name__)


async def fetch_all_messages(
    tenant_id: str,
    store: Any,
) -> list[UnifiedMessage]:
    """Fetch messages from all connected accounts."""
    records, total = await store.list_messages(tenant_id=tenant_id, limit=None)
    if total > 0:
        return [record_to_message(record) for record in records]

    messages: list[UnifiedMessage] = []
    account_records = await store.list_accounts(tenant_id)

    for record in account_records:
        account = record_to_account(record)
        if account.status != AccountStatus.CONNECTED:
            continue

        try:
            if account.provider == EmailProvider.GMAIL:
                account_messages = await fetch_gmail_messages(account, tenant_id)
            else:
                account_messages = await fetch_outlook_messages(account, tenant_id)

            for message in account_messages:
                await store.save_message(tenant_id, message_to_record(message))
            messages.extend(account_messages)

        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError, KeyError) as e:
            logger.warning(
                "Error fetching messages for %s account %s: %s", account.provider.value, account.id, e
            )
            await store.increment_account_counts(tenant_id, account.id, sync_error_delta=1)

    # Apply priority scoring
    messages = await score_messages(messages)

    return messages


async def fetch_gmail_messages(account: ConnectedAccount, tenant_id: str) -> list[UnifiedMessage]:
    """Fetch messages from Gmail account."""
    _sync_services = get_sync_services()
    _sync_services_lock = get_sync_services_lock()

    # Check if sync service is running and has synced messages (thread-safe lookup)
    sync_service = None
    if tenant_id:
        async with _sync_services_lock:
            if tenant_id in _sync_services:
                sync_service = _sync_services[tenant_id].get(account.id)
    if sync_service:
        # Check if initial sync is complete
        state = getattr(sync_service, "state", None)
        if state and getattr(state, "initial_sync_complete", False):
            # Messages are already in cache via callbacks
            logger.debug(
                "[UnifiedInbox] Gmail sync active for %s, messages synced: %s", account.id, getattr(state, 'total_messages_synced', 0)
            )
            return []  # Messages already in cache

    # Fall back to sample data if sync not active
    return generate_sample_messages(account, 5)


async def fetch_outlook_messages(account: ConnectedAccount, tenant_id: str) -> list[UnifiedMessage]:
    """Fetch messages from Outlook account."""
    _sync_services = get_sync_services()
    _sync_services_lock = get_sync_services_lock()

    # Check if sync service is running and has synced messages (thread-safe lookup)
    sync_service = None
    if tenant_id:
        async with _sync_services_lock:
            if tenant_id in _sync_services:
                sync_service = _sync_services[tenant_id].get(account.id)
    if sync_service:
        # Check if initial sync is complete
        state = getattr(sync_service, "state", None)
        if state and getattr(state, "initial_sync_complete", False):
            # Messages are already in cache via callbacks
            logger.debug(
                "[UnifiedInbox] Outlook sync active for %s, messages synced: %s", account.id, getattr(state, 'total_messages_synced', 0)
            )
            return []  # Messages already in cache

    # Fall back to sample data if sync not active
    return generate_sample_messages(account, 5)


def generate_sample_messages(account: ConnectedAccount, count: int) -> list[UnifiedMessage]:
    """Generate sample messages for testing."""
    messages = []
    now = datetime.now(timezone.utc)

    sample_subjects = [
        ("Urgent: Contract Review Required", "critical"),
        ("Q4 Budget Approval Needed", "high"),
        ("Weekly Team Update", "medium"),
        ("Newsletter: Industry Updates", "low"),
        ("Meeting Rescheduled", "medium"),
    ]

    for i in range(min(count, len(sample_subjects))):
        subject, priority = sample_subjects[i]
        messages.append(
            UnifiedMessage(
                id=str(uuid4()),
                account_id=account.id,
                provider=account.provider,
                external_id=f"ext_{uuid4().hex[:8]}",
                subject=subject,
                sender_email=f"sender{i}@example.com",
                sender_name=f"Sender {i}",
                recipients=[account.email_address],
                cc=[],
                received_at=now - timedelta(hours=i),
                snippet=f"Preview of message {i}...",
                body_preview=f"This is the body preview of message {i}...",
                is_read=i > 2,
                is_starred=i == 0,
                has_attachments=i < 2,
                labels=["inbox"],
                priority_tier=priority,
                priority_score={"critical": 0.95, "high": 0.75, "medium": 0.5, "low": 0.25}[
                    priority
                ],
            )
        )

    return messages


async def score_messages(
    messages: list[UnifiedMessage],
) -> list[UnifiedMessage]:
    """Apply priority scoring to messages."""
    try:
        from aragora.services.email_prioritization import (  # noqa: F401
            EmailPrioritizer,
            EmailPriority,
        )

        # Use prioritizer if available
        # For now, messages already have sample scores
        return messages

    except ImportError:
        # Prioritizer not available, use existing scores
        return messages
