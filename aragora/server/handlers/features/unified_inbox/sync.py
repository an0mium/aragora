"""Sync service registry for unified inbox.

Manages per-tenant, per-account sync services (Gmail/Outlook) and provides
the conversion function from synced messages to unified messages.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .models import EmailProvider, UnifiedMessage

logger = logging.getLogger(__name__)

# =============================================================================
# Sync Service Registry
# =============================================================================

# tenant_id -> account_id -> sync_service (GmailSyncService or OutlookSyncService)
_sync_services: dict[str, dict[str, Any]] = {}
_sync_services_lock = asyncio.Lock()  # Thread-safe access to _sync_services


def get_sync_services() -> dict[str, dict[str, Any]]:
    """Return the global sync services dict (for direct read access)."""
    return _sync_services


def get_sync_services_lock() -> asyncio.Lock:
    """Return the global sync services lock."""
    return _sync_services_lock


def convert_synced_message_to_unified(
    synced_msg: Any,
    account_id: str,
    provider: EmailProvider,
) -> UnifiedMessage:
    """Convert a SyncedMessage to UnifiedMessage format."""
    msg = synced_msg.message
    priority = synced_msg.priority_result

    # Extract priority info
    priority_score = 0.5
    priority_tier = "medium"
    priority_reasons: list[str] = []

    if priority:
        priority_score = priority.score if hasattr(priority, "score") else 0.5
        priority_tier = priority.tier if hasattr(priority, "tier") else "medium"
        priority_reasons = priority.reasons if hasattr(priority, "reasons") else []

    return UnifiedMessage(
        id=str(uuid4()),
        account_id=account_id,
        provider=provider,
        external_id=msg.id if hasattr(msg, "id") else str(uuid4()),
        subject=msg.subject if hasattr(msg, "subject") else "",
        sender_email=msg.from_email if hasattr(msg, "from_email") else "",
        sender_name=msg.from_name if hasattr(msg, "from_name") else "",
        recipients=msg.to if hasattr(msg, "to") else [],
        cc=msg.cc if hasattr(msg, "cc") else [],
        received_at=msg.date if hasattr(msg, "date") else datetime.now(timezone.utc),
        snippet=msg.snippet if hasattr(msg, "snippet") else "",
        body_preview=msg.body[:500] if hasattr(msg, "body") and msg.body else "",
        is_read=msg.is_read if hasattr(msg, "is_read") else False,
        is_starred=msg.is_starred if hasattr(msg, "is_starred") else False,
        has_attachments=bool(msg.attachments) if hasattr(msg, "attachments") else False,
        labels=msg.labels if hasattr(msg, "labels") else [],
        thread_id=msg.thread_id if hasattr(msg, "thread_id") else None,
        priority_score=priority_score,
        priority_tier=priority_tier,
        priority_reasons=priority_reasons,
    )
