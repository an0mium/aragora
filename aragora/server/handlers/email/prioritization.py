"""
Email prioritization handlers.

Provides handlers for:
- Single email scoring
- Inbox ranking
- User feedback recording
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from aragora.rbac.decorators import require_permission
from aragora.server.middleware.rate_limit import rate_limit
from aragora.observability.metrics import track_handler

from .storage import _check_email_permission, get_prioritizer

logger = logging.getLogger(__name__)

# RBAC permission constants
PERM_EMAIL_READ = "email:read"
PERM_EMAIL_UPDATE = "email:update"

_AUTH_CONTEXT_UNSET = object()


@require_permission(PERM_EMAIL_READ, context_param="auth_context")
@rate_limit(requests_per_minute=60)
@track_handler("email/prioritize")
async def handle_prioritize_email(
    email_data: dict[str, Any],
    user_id: str = "default",
    workspace_id: str = "default",
    force_tier: str | None = None,
    auth_context: Any | None = None,
) -> dict[str, Any]:
    """
    Score a single email for priority.

    POST /api/email/prioritize
    {
        "email": {
            "id": "msg_123",
            "subject": "Urgent: Project deadline",
            "from_address": "boss@company.com",
            "body_text": "...",
            "snippet": "...",
            "labels": ["INBOX", "IMPORTANT"],
            "is_important": true,
            "is_starred": false,
            "is_read": false
        },
        "force_tier": "tier_1_rules"  // Optional: force specific scoring tier
    }

    Returns:
        Priority result with score, confidence, and rationale
    """
    from aragora.connectors.enterprise.communication.models import EmailMessage
    from aragora.services.email_prioritization import ScoringTier

    try:
        # Convert dict to EmailMessage
        email = EmailMessage(
            id=email_data.get("id", "unknown"),
            thread_id=email_data.get("thread_id", email_data.get("id", "unknown")),
            subject=email_data.get("subject", ""),
            from_address=email_data.get("from_address", ""),
            to_addresses=email_data.get("to_addresses", []),
            cc_addresses=email_data.get("cc_addresses", []),
            bcc_addresses=email_data.get("bcc_addresses", []),
            date=(
                datetime.fromisoformat(email_data["date"])
                if email_data.get("date")
                else datetime.now()
            ),
            body_text=email_data.get("body_text", ""),
            body_html=email_data.get("body_html", ""),
            snippet=email_data.get("snippet", ""),
            labels=email_data.get("labels", []),
            headers=email_data.get("headers", {}),
            attachments=[],
            is_read=email_data.get("is_read", False),
            is_starred=email_data.get("is_starred", False),
            is_important=email_data.get("is_important", False),
        )

        # Get prioritizer
        prioritizer = get_prioritizer(user_id)

        # Parse force_tier if provided
        tier = None
        if force_tier:
            tier = ScoringTier(force_tier)

        # Score the email
        result = await prioritizer.score_email(email, force_tier=tier)

        return {
            "success": True,
            "result": result.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Failed to prioritize email: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@require_permission(PERM_EMAIL_READ, context_param="auth_context")
@rate_limit(requests_per_minute=60)
@track_handler("email/rank_inbox")
async def handle_rank_inbox(
    emails: list[dict[str, Any]],
    user_id: str = "default",
    workspace_id: str = "default",
    limit: int | None = None,
    auth_context: Any | None = None,
) -> dict[str, Any]:
    """
    Rank multiple emails by priority.

    POST /api/email/rank-inbox
    {
        "emails": [...],
        "limit": 50
    }

    Returns:
        Ranked list of email priority results
    """
    from aragora.connectors.enterprise.communication.models import EmailMessage

    try:
        # Convert dicts to EmailMessages
        email_messages = []
        for email_data in emails:
            email = EmailMessage(
                id=email_data.get("id", "unknown"),
                thread_id=email_data.get("thread_id", email_data.get("id", "unknown")),
                subject=email_data.get("subject", ""),
                from_address=email_data.get("from_address", ""),
                to_addresses=email_data.get("to_addresses", []),
                cc_addresses=email_data.get("cc_addresses", []),
                bcc_addresses=email_data.get("bcc_addresses", []),
                date=(
                    datetime.fromisoformat(email_data["date"])
                    if email_data.get("date")
                    else datetime.now()
                ),
                body_text=email_data.get("body_text", ""),
                body_html=email_data.get("body_html", ""),
                snippet=email_data.get("snippet", ""),
                labels=email_data.get("labels", []),
                headers=email_data.get("headers", {}),
                attachments=[],
                is_read=email_data.get("is_read", False),
                is_starred=email_data.get("is_starred", False),
                is_important=email_data.get("is_important", False),
            )
            email_messages.append(email)

        # Get prioritizer and rank
        prioritizer = get_prioritizer(user_id)
        results = await prioritizer.rank_inbox(email_messages, limit=limit)

        return {
            "success": True,
            "results": [r.to_dict() for r in results],
            "total": len(results),
        }

    except Exception as e:
        logger.exception(f"Failed to rank inbox: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@require_permission(PERM_EMAIL_UPDATE, context_param="auth_context")
@rate_limit(requests_per_minute=60)
@track_handler("email/feedback")
async def handle_email_feedback(
    email_id: str,
    action: str,
    user_id: str = "default",
    workspace_id: str = "default",
    email_data: dict[str, Any] | None = None,
    auth_context: Any | None = _AUTH_CONTEXT_UNSET,
) -> dict[str, Any]:
    """
    Record user action for learning.

    POST /api/email/feedback
    {
        "email_id": "msg_123",
        "action": "archived",  // read, archived, deleted, replied, starred, important
        "email": {...}  // Optional: full email data for context
    }
    """
    from aragora.connectors.enterprise.communication.models import EmailMessage

    if auth_context is not _AUTH_CONTEXT_UNSET:
        perm_error = _check_email_permission(auth_context, PERM_EMAIL_UPDATE)
        if perm_error:
            return perm_error

    try:
        # Convert email data if provided
        email = None
        if email_data:
            email = EmailMessage(
                id=email_data.get("id", email_id),
                thread_id=email_data.get("thread_id", email_id),
                subject=email_data.get("subject", ""),
                from_address=email_data.get("from_address", ""),
                to_addresses=email_data.get("to_addresses", []),
                cc_addresses=[],
                bcc_addresses=[],
                date=datetime.now(),
                body_text=email_data.get("body_text", ""),
                body_html="",
                snippet=email_data.get("snippet", ""),
                labels=email_data.get("labels", []),
                headers={},
                attachments=[],
                is_read=True,
                is_starred=email_data.get("is_starred", False),
                is_important=email_data.get("is_important", False),
            )

        # Record action
        prioritizer = get_prioritizer(user_id)
        await prioritizer.record_user_action(email_id, action, email)

        return {
            "success": True,
            "email_id": email_id,
            "action": action,
            "recorded_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.exception(f"Failed to record feedback: {e}")
        return {
            "success": False,
            "error": str(e),
        }
