"""
Email categorization handlers.

Provides handlers for:
- Single email categorization
- Batch categorization
- Batch feedback recording
- Gmail label application
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any

from aragora.rbac.decorators import require_permission
from aragora.server.handlers.utils.rate_limit import rate_limit

from .storage import (
    _check_email_permission,
    get_gmail_connector,
    get_prioritizer,
)

logger = logging.getLogger(__name__)

# RBAC permission constants
PERM_EMAIL_READ = "email:read"
PERM_EMAIL_UPDATE = "email:update"

_AUTH_CONTEXT_UNSET = object()

# Global categorizer instance
_categorizer: Any | None = None
_categorizer_lock = threading.Lock()


def get_categorizer():
    """Get or create email categorizer (thread-safe)."""
    global _categorizer
    if _categorizer is not None:
        return _categorizer

    with _categorizer_lock:
        if _categorizer is None:
            from aragora.services.email_categorizer import EmailCategorizer

            _categorizer = EmailCategorizer(gmail_connector=get_gmail_connector())
        return _categorizer


@require_permission(PERM_EMAIL_READ, context_param="auth_context")
@rate_limit(requests_per_minute=60)  # READ operation
async def handle_categorize_email(
    email_data: dict[str, Any],
    user_id: str = "default",
    workspace_id: str = "default",
    auth_context: Any | None = None,
) -> dict[str, Any]:
    """
    Categorize a single email into a smart folder.

    POST /api/v1/email/categorize
    {
        "email": {
            "id": "msg_123",
            "subject": "Invoice #12345",
            "from_address": "billing@company.com",
            "body_text": "..."
        }
    }

    Returns:
        Category result with confidence and suggested label
    """

    from aragora.connectors.enterprise.communication.models import EmailMessage

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

        categorizer = get_categorizer()
        result = await categorizer.categorize_email(email)

        return {
            "success": True,
            "result": result.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Failed to categorize email: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@require_permission(PERM_EMAIL_READ, context_param="auth_context")
@rate_limit(requests_per_minute=60)  # READ operation
async def handle_categorize_batch(
    emails: list[dict[str, Any]],
    user_id: str = "default",
    workspace_id: str = "default",
    concurrency: int = 10,
    auth_context: Any | None = None,
) -> dict[str, Any]:
    """
    Categorize multiple emails in batch.

    POST /api/v1/email/categorize/batch
    {
        "emails": [
            {"id": "msg_1", "subject": "...", ...},
            {"id": "msg_2", "subject": "...", ...}
        ],
        "concurrency": 10
    }

    Returns:
        List of categorization results
    """
    from aragora.connectors.enterprise.communication.models import EmailMessage

    try:
        # Convert dicts to EmailMessages
        email_objects: list[EmailMessage] = []
        for email_data in emails:
            email = EmailMessage(
                id=email_data.get("id", f"unknown_{len(email_objects)}"),
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
            email_objects.append(email)

        categorizer = get_categorizer()
        results = await categorizer.categorize_batch(email_objects, concurrency=concurrency)

        # Get stats
        stats = categorizer.get_category_stats(results)

        return {
            "success": True,
            "results": [r.to_dict() for r in results],
            "stats": stats,
        }

    except Exception as e:
        logger.exception(f"Failed to categorize batch: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@require_permission(PERM_EMAIL_UPDATE, context_param="auth_context")
@rate_limit(requests_per_minute=20)  # WRITE operation
async def handle_feedback_batch(
    feedback_items: list[dict[str, Any]],
    user_id: str = "default",
    workspace_id: str = "default",
    auth_context: Any | None = _AUTH_CONTEXT_UNSET,
) -> dict[str, Any]:
    """
    Record batch user actions for learning.

    POST /api/v1/email/feedback/batch
    {
        "items": [
            {"email_id": "msg_1", "action": "archived"},
            {"email_id": "msg_2", "action": "replied", "response_time_minutes": 5}
        ]
    }

    Actions: opened, replied, starred, archived, deleted, snoozed
    """
    if auth_context is not _AUTH_CONTEXT_UNSET:
        perm_error = _check_email_permission(auth_context, PERM_EMAIL_UPDATE)
        if perm_error:
            return perm_error

    try:
        prioritizer = get_prioritizer(user_id)
        results = []
        errors = []

        for item in feedback_items:
            email_id = item.get("email_id")
            action = item.get("action")
            response_time = item.get("response_time_minutes")

            if not email_id or not action:
                errors.append({"email_id": email_id, "error": "Missing email_id or action"})
                continue

            try:
                await prioritizer.record_user_action(
                    email_id=email_id,
                    action=action,
                    response_time_minutes=response_time,
                )
                results.append({"email_id": email_id, "action": action, "recorded": True})
            except Exception as e:
                errors.append({"email_id": email_id, "error": str(e)})

        return {
            "success": True,
            "recorded": len(results),
            "errors": len(errors),
            "results": results,
            "error_details": errors if errors else None,
        }

    except Exception as e:
        logger.exception(f"Failed to record batch feedback: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@require_permission(PERM_EMAIL_UPDATE, context_param="auth_context")
@rate_limit(requests_per_minute=20)  # WRITE operation
async def handle_apply_category_label(
    email_id: str,
    category: str,
    user_id: str = "default",
    workspace_id: str = "default",
    auth_context: Any | None = _AUTH_CONTEXT_UNSET,
) -> dict[str, Any]:
    """
    Apply Gmail label based on category.

    POST /api/v1/email/categorize/apply-label
    {
        "email_id": "msg_123",
        "category": "invoices"
    }
    """
    if auth_context is not _AUTH_CONTEXT_UNSET:
        perm_error = _check_email_permission(auth_context, PERM_EMAIL_UPDATE)
        if perm_error:
            return perm_error

    try:
        from aragora.services.email_categorizer import EmailCategory

        categorizer = get_categorizer()
        category_enum = EmailCategory(category)

        success = await categorizer.apply_gmail_label(email_id, category_enum)

        return {
            "success": success,
            "email_id": email_id,
            "category": category,
            "label_applied": success,
        }

    except ValueError:
        return {
            "success": False,
            "error": f"Invalid category: {category}",
        }
    except Exception as e:
        logger.exception(f"Failed to apply category label: {e}")
        return {
            "success": False,
            "error": str(e),
        }
