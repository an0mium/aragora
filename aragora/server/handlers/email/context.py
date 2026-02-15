"""
Cross-channel context handlers for email.

Provides handlers for:
- Getting cross-channel context for an email address
- Getting context-based priority boosts
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from aragora.rbac.decorators import require_permission

from .storage import get_context_service

logger = logging.getLogger(__name__)

# RBAC permission constants
PERM_EMAIL_READ = "email:read"


@require_permission(PERM_EMAIL_READ, context_param="auth_context")
async def handle_get_context(
    email_address: str,
    user_id: str = "default",
    workspace_id: str = "default",
    auth_context: Any | None = None,
) -> dict[str, Any]:
    """
    Get cross-channel context for an email address.

    GET /api/email/context/:email_address

    Returns context from Slack, Drive, Calendar if available.
    """
    try:
        service = get_context_service()
        context = await service.get_user_context(email_address)

        return {
            "success": True,
            "context": context.to_dict(),
        }

    except (ConnectionError, TimeoutError, OSError, ValueError) as e:
        logger.exception(f"Failed to get context: {e}")
        return {
            "success": False,
            "error": "Failed to get context",
        }


@require_permission(PERM_EMAIL_READ, context_param="auth_context")
async def handle_get_email_context_boost(
    email_data: dict[str, Any],
    user_id: str = "default",
    workspace_id: str = "default",
    auth_context: Any | None = None,
) -> dict[str, Any]:
    """
    Get context-based priority boosts for an email.

    POST /api/email/context/boost
    {
        "email": {...}
    }

    Returns boost scores from cross-channel signals.
    """
    from aragora.connectors.enterprise.communication.models import EmailMessage

    try:
        # Convert to EmailMessage
        email = EmailMessage(
            id=email_data.get("id", "unknown"),
            thread_id=email_data.get("thread_id", "unknown"),
            subject=email_data.get("subject", ""),
            from_address=email_data.get("from_address", ""),
            to_addresses=email_data.get("to_addresses", []),
            cc_addresses=[],
            bcc_addresses=[],
            date=datetime.now(),
            body_text=email_data.get("body_text", ""),
            body_html="",
            snippet=email_data.get("snippet", ""),
            labels=[],
            headers={},
            attachments=[],
            is_read=False,
            is_starred=False,
            is_important=False,
        )

        service = get_context_service()
        boost = await service.get_email_context(email)

        return {
            "success": True,
            "boost": {
                "email_id": boost.email_id,
                "total_boost": boost.total_boost,
                "slack_activity_boost": boost.slack_activity_boost,
                "drive_relevance_boost": boost.drive_relevance_boost,
                "calendar_urgency_boost": boost.calendar_urgency_boost,
                "slack_reason": boost.slack_reason,
                "drive_reason": boost.drive_reason,
                "calendar_reason": boost.calendar_reason,
                "related_slack_channels": boost.related_slack_channels,
                "related_drive_files": boost.related_drive_files,
                "related_meetings": boost.related_meetings,
            },
        }

    except (TypeError, ValueError, KeyError, AttributeError, ConnectionError, TimeoutError) as e:
        logger.exception(f"Failed to get context boost: {e}")
        return {
            "success": False,
            "error": "Failed to get context boost",
        }
