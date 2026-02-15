"""
Inbox fetch and rank handlers.

Provides handlers for:
- Fetching and ranking inbox emails
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from aragora.rbac.decorators import require_permission
from aragora.server.handlers.utils.rate_limit import rate_limit

from .storage import (
    get_gmail_connector,
    get_prioritizer,
)

logger = logging.getLogger(__name__)

# RBAC permission constants
PERM_EMAIL_READ = "email:read"
PERM_EMAIL_UPDATE = "email:update"


@require_permission(PERM_EMAIL_READ, context_param="auth_context")
@rate_limit(requests_per_minute=10)  # SYNC operation
async def handle_fetch_and_rank_inbox(
    user_id: str = "default",
    workspace_id: str = "default",
    labels: list[str] | None = None,
    limit: int = 50,
    include_read: bool = False,
    auth_context: Any | None = None,
) -> dict[str, Any]:
    """
    Fetch inbox from Gmail and return ranked results.

    GET /api/email/inbox
    Query params:
        labels: Comma-separated labels (default: INBOX)
        limit: Max emails to fetch (default: 50)
        include_read: Include read emails (default: false)
        workspace_id: Tenant workspace ID for multi-tenant isolation

    This is the main endpoint for the inbox view - fetches emails
    and returns them pre-ranked by priority.
    """
    try:
        connector = get_gmail_connector(user_id)

        if not connector._access_token:
            return {
                "success": False,
                "error": "Not authenticated. Complete Gmail OAuth first.",
                "needs_auth": True,
            }

        # Build query
        query_parts = []
        if not include_read:
            query_parts.append("is:unread")

        query = " ".join(query_parts) if query_parts else ""

        # Fetch messages
        emails = []
        message_ids, _ = await connector.list_messages(
            query=query,
            label_ids=labels or ["INBOX"],
            max_results=limit,
        )

        for msg_id in message_ids[:limit]:
            try:
                msg = await connector.get_message(msg_id)
                emails.append(msg)
            except Exception as e:
                logger.warning(f"Failed to fetch message {msg_id}: {e}")

        # Rank emails
        prioritizer = get_prioritizer(user_id)
        ranked_results = await prioritizer.rank_inbox(emails, limit=limit)

        # Build response with email data + priority info
        inbox_items = []
        for result in ranked_results:
            # Find corresponding email
            email = next((e for e in emails if e.id == result.email_id), None)
            if email:
                inbox_items.append(
                    {
                        "email": {
                            "id": email.id,
                            "thread_id": email.thread_id,
                            "subject": email.subject,
                            "from_address": email.from_address,
                            "to_addresses": email.to_addresses,
                            "date": email.date.isoformat() if email.date else None,
                            "snippet": email.snippet,
                            "labels": email.labels,
                            "is_read": email.is_read,
                            "is_starred": email.is_starred,
                            "is_important": email.is_important,
                            "has_attachments": len(email.attachments) > 0,
                        },
                        "priority": result.to_dict(),
                    }
                )

        return {
            "success": True,
            "inbox": inbox_items,
            "total": len(inbox_items),
            "fetched_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.exception(f"Failed to fetch inbox: {e}")
        return {
            "success": False,
            "error": "Failed to fetch inbox",
        }
