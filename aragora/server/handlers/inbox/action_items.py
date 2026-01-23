"""
HTTP API Handlers for Action Items.

Provides REST APIs for action item extraction and management:
- Extract action items from emails
- List pending action items
- Mark action items as completed
- Get action items with deadlines
- Batch extraction from multiple emails

Endpoints:
- POST /api/v1/inbox/actions/extract - Extract action items from email
- GET /api/v1/inbox/actions/pending - List pending action items
- POST /api/v1/inbox/actions/{id}/complete - Mark action item complete
- POST /api/v1/inbox/actions/{id}/status - Update action item status
- GET /api/v1/inbox/actions/due-soon - Get items due within timeframe
- POST /api/v1/inbox/actions/batch-extract - Batch extract from emails
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from aragora.server.handlers.base import (
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)

# Thread-safe service instances
_action_extractor: Optional[Any] = None
_action_extractor_lock = threading.Lock()
_meeting_detector: Optional[Any] = None
_meeting_detector_lock = threading.Lock()

# In-memory action item storage (replace with DB in production)
_action_items: Dict[str, Dict[str, Any]] = {}
_action_items_lock = threading.Lock()


def get_action_extractor():
    """Get or create action item extractor (thread-safe)."""
    global _action_extractor
    if _action_extractor is not None:
        return _action_extractor

    with _action_extractor_lock:
        if _action_extractor is None:
            from aragora.services.action_item_extractor import ActionItemExtractor

            _action_extractor = ActionItemExtractor()
        return _action_extractor


def get_meeting_detector():
    """Get or create meeting detector (thread-safe)."""
    global _meeting_detector
    if _meeting_detector is not None:
        return _meeting_detector

    with _meeting_detector_lock:
        if _meeting_detector is None:
            from aragora.services.meeting_detector import MeetingDetector

            _meeting_detector = MeetingDetector()
        return _meeting_detector


# =============================================================================
# Action Item Extraction Handlers
# =============================================================================


async def handle_extract_action_items(
    data: Dict[str, Any],
    user_id: str = "default",
) -> Dict[str, Any]:
    """
    Extract action items from an email.

    POST /api/v1/inbox/actions/extract
    Body: {
        email_id: str,
        subject: str,
        body: str,
        sender: str,
        to_addresses: list[str] (optional),
        extract_deadlines: bool (optional, default true),
        detect_assignees: bool (optional, default true)
    }
    """
    try:
        extractor = get_action_extractor()

        email_id = data.get("email_id", "")
        subject = data.get("subject", "")
        body = data.get("body", "")
        sender = data.get("sender", "")
        to_addresses = data.get("to_addresses", [])
        extract_deadlines = data.get("extract_deadlines", True)
        detect_assignees = data.get("detect_assignees", True)

        if not body and not subject:
            return error_response("Either subject or body is required", status=400)

        # Create email-like object
        class EmailLike:
            def __init__(self):
                self.id = email_id or f"inline_{hash((subject, body))}"
                self.subject = subject
                self.body_text = body
                self.from_address = sender
                self.to_addresses = to_addresses

        email = EmailLike()

        result = await extractor.extract_action_items(
            email,
            extract_deadlines=extract_deadlines,
            detect_assignees=detect_assignees,
        )

        # Store extracted items
        with _action_items_lock:
            for item in result.action_items:
                _action_items[item.id] = item.to_dict()

        return success_response(result.to_dict())

    except Exception as e:
        logger.exception("Failed to extract action items")
        return error_response(f"Extraction failed: {str(e)}", status=500)


async def handle_list_pending_actions(
    data: Dict[str, Any],
    user_id: str = "default",
) -> Dict[str, Any]:
    """
    List pending action items.

    GET /api/v1/inbox/actions/pending
    Query params:
        assignee: str (optional) - Filter by assignee email
        priority: str (optional) - Filter by priority (critical, high, medium, low)
        due_within_hours: int (optional) - Filter items due within N hours
        limit: int (optional, default 50) - Max items to return
        offset: int (optional, default 0) - Pagination offset
    """
    try:
        assignee_filter = data.get("assignee")
        priority_filter = data.get("priority")
        due_within = data.get("due_within_hours")
        limit = min(int(data.get("limit", 50)), 200)
        offset = int(data.get("offset", 0))

        with _action_items_lock:
            items = list(_action_items.values())

        # Filter by status (pending or in_progress)
        items = [item for item in items if item.get("status") in ("pending", "in_progress")]

        # Filter by assignee
        if assignee_filter:
            items = [
                item
                for item in items
                if item.get("assignee_email", "").lower() == assignee_filter.lower()
            ]

        # Filter by priority
        if priority_filter:
            priority_map = {"critical": 1, "high": 2, "medium": 3, "low": 4}
            target_priority = priority_map.get(priority_filter.lower())
            if target_priority:
                items = [item for item in items if item.get("priority") == target_priority]

        # Filter by due date
        if due_within:
            from datetime import timedelta

            now = datetime.now(timezone.utc)
            cutoff = now + timedelta(hours=int(due_within))

            def is_due_soon(item):
                deadline_str = item.get("deadline")
                if not deadline_str:
                    return False
                try:
                    deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
                    return deadline <= cutoff
                except (ValueError, TypeError):
                    return False

            items = [item for item in items if is_due_soon(item)]

        # Sort by deadline (items with deadlines first), then priority
        def sort_key(item):
            deadline = item.get("deadline")
            priority = item.get("priority", 3)
            if deadline:
                try:
                    dt = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
                    return (0, dt, priority)
                except (ValueError, TypeError):
                    pass
            return (1, datetime.max.replace(tzinfo=timezone.utc), priority)

        items.sort(key=sort_key)

        # Pagination
        total = len(items)
        items = items[offset : offset + limit]

        return success_response(
            {
                "action_items": items,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total,
            }
        )

    except Exception as e:
        logger.exception("Failed to list pending actions")
        return error_response(f"List failed: {str(e)}", status=500)


async def handle_complete_action(
    data: Dict[str, Any],
    action_id: str = "",
    user_id: str = "default",
) -> Dict[str, Any]:
    """
    Mark an action item as completed.

    POST /api/v1/inbox/actions/{id}/complete
    Body: {
        completed_by: str (optional) - Who completed it
        notes: str (optional) - Completion notes
    }
    """
    try:
        if not action_id:
            action_id = data.get("action_id", "")

        if not action_id:
            return error_response("action_id is required", status=400)

        with _action_items_lock:
            if action_id not in _action_items:
                return error_response("Action item not found", status=404)

            item = _action_items[action_id]
            item["status"] = "completed"
            item["completed_at"] = datetime.now(timezone.utc).isoformat()
            item["completed_by"] = data.get("completed_by", user_id)
            if data.get("notes"):
                item["completion_notes"] = data["notes"]

        return success_response(
            {
                "action_id": action_id,
                "status": "completed",
                "completed_at": item["completed_at"],
                "message": "Action item marked as completed",
            }
        )

    except Exception as e:
        logger.exception("Failed to complete action")
        return error_response(f"Complete failed: {str(e)}", status=500)


async def handle_update_action_status(
    data: Dict[str, Any],
    action_id: str = "",
    user_id: str = "default",
) -> Dict[str, Any]:
    """
    Update action item status.

    POST /api/v1/inbox/actions/{id}/status
    Body: {
        status: str - New status (pending, in_progress, completed, cancelled, deferred)
        notes: str (optional) - Status change notes
    }
    """
    try:
        if not action_id:
            action_id = data.get("action_id", "")

        if not action_id:
            return error_response("action_id is required", status=400)

        new_status = data.get("status")
        valid_statuses = {"pending", "in_progress", "completed", "cancelled", "deferred"}
        if not new_status or new_status not in valid_statuses:
            return error_response(
                f"Invalid status. Must be one of: {', '.join(valid_statuses)}", status=400
            )

        with _action_items_lock:
            if action_id not in _action_items:
                return error_response("Action item not found", status=404)

            item = _action_items[action_id]
            old_status = item.get("status")
            item["status"] = new_status
            item["status_updated_at"] = datetime.now(timezone.utc).isoformat()
            item["status_updated_by"] = user_id

            if new_status == "completed":
                item["completed_at"] = item["status_updated_at"]

            if data.get("notes"):
                if "status_history" not in item:
                    item["status_history"] = []
                item["status_history"].append(
                    {
                        "from": old_status,
                        "to": new_status,
                        "notes": data["notes"],
                        "at": item["status_updated_at"],
                        "by": user_id,
                    }
                )

        return success_response(
            {
                "action_id": action_id,
                "status": new_status,
                "previous_status": old_status,
                "updated_at": item["status_updated_at"],
                "message": f"Status updated to {new_status}",
            }
        )

    except Exception as e:
        logger.exception("Failed to update action status")
        return error_response(f"Update failed: {str(e)}", status=500)


async def handle_get_due_soon(
    data: Dict[str, Any],
    user_id: str = "default",
) -> Dict[str, Any]:
    """
    Get action items due soon.

    GET /api/v1/inbox/actions/due-soon
    Query params:
        hours: int (optional, default 24) - Items due within N hours
        include_overdue: bool (optional, default true) - Include past-due items
    """
    try:
        hours = int(data.get("hours", 24))
        include_overdue = data.get("include_overdue", True)
        if isinstance(include_overdue, str):
            include_overdue = include_overdue.lower() == "true"

        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours)

        with _action_items_lock:
            items = list(_action_items.values())

        # Filter to pending/in_progress items with deadlines
        due_soon = []
        overdue = []

        for item in items:
            if item.get("status") not in ("pending", "in_progress"):
                continue

            deadline_str = item.get("deadline")
            if not deadline_str:
                continue

            try:
                deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))

                if deadline < now:
                    if include_overdue:
                        item["is_overdue"] = True
                        item["overdue_hours"] = int((now - deadline).total_seconds() / 3600)
                        overdue.append(item)
                elif deadline <= cutoff:
                    item["is_overdue"] = False
                    item["hours_remaining"] = int((deadline - now).total_seconds() / 3600)
                    due_soon.append(item)

            except (ValueError, TypeError):
                continue

        # Sort by deadline
        overdue.sort(key=lambda x: x.get("deadline", ""))
        due_soon.sort(key=lambda x: x.get("deadline", ""))

        return success_response(
            {
                "due_soon": due_soon,
                "overdue": overdue,
                "due_soon_count": len(due_soon),
                "overdue_count": len(overdue),
                "total_urgent": len(due_soon) + len(overdue),
                "hours_window": hours,
            }
        )

    except Exception as e:
        logger.exception("Failed to get due soon items")
        return error_response(f"Query failed: {str(e)}", status=500)


async def handle_batch_extract(
    data: Dict[str, Any],
    user_id: str = "default",
) -> Dict[str, Any]:
    """
    Batch extract action items from multiple emails.

    POST /api/v1/inbox/actions/batch-extract
    Body: {
        emails: list[{
            email_id: str,
            subject: str,
            body: str,
            sender: str,
            to_addresses: list[str] (optional)
        }]
    }
    """
    try:
        extractor = get_action_extractor()

        emails_data = data.get("emails", [])
        if not emails_data:
            return error_response("emails list is required", status=400)

        if len(emails_data) > 50:
            return error_response("Maximum 50 emails per batch", status=400)

        results = []
        total_items = 0
        total_high_priority = 0

        for email_data in emails_data:
            # Create email-like object
            class EmailLike:
                def __init__(self, data):
                    self.id = data.get("email_id", f"batch_{hash(str(data))}")
                    self.subject = data.get("subject", "")
                    self.body_text = data.get("body", "")
                    self.from_address = data.get("sender", "")
                    self.to_addresses = data.get("to_addresses", [])

            email = EmailLike(email_data)

            try:
                result = await extractor.extract_action_items(email)

                # Store extracted items
                with _action_items_lock:
                    for item in result.action_items:
                        _action_items[item.id] = item.to_dict()

                results.append(
                    {
                        "email_id": email.id,
                        "success": True,
                        "action_items": [item.to_dict() for item in result.action_items],
                        "count": result.total_count,
                        "high_priority_count": result.high_priority_count,
                    }
                )

                total_items += result.total_count
                total_high_priority += result.high_priority_count

            except Exception as e:
                logger.warning(f"Failed to extract from email {email.id}: {e}")
                results.append(
                    {
                        "email_id": email.id,
                        "success": False,
                        "error": str(e),
                    }
                )

        return success_response(
            {
                "results": results,
                "total_emails": len(emails_data),
                "total_action_items": total_items,
                "total_high_priority": total_high_priority,
                "successful_extractions": sum(1 for r in results if r.get("success")),
            }
        )

    except Exception as e:
        logger.exception("Failed batch extraction")
        return error_response(f"Batch extraction failed: {str(e)}", status=500)


# =============================================================================
# Meeting Detection Handlers
# =============================================================================


async def handle_detect_meeting(
    data: Dict[str, Any],
    user_id: str = "default",
) -> Dict[str, Any]:
    """
    Detect meeting information from an email.

    POST /api/v1/inbox/meetings/detect
    Body: {
        email_id: str,
        subject: str,
        body: str,
        sender: str,
        check_calendar: bool (optional, default false)
    }
    """
    try:
        detector = get_meeting_detector()

        email_id = data.get("email_id", "")
        subject = data.get("subject", "")
        body = data.get("body", "")
        sender = data.get("sender", "")
        check_calendar = data.get("check_calendar", False)

        if not body and not subject:
            return error_response("Either subject or body is required", status=400)

        # Create email-like object
        class EmailLike:
            def __init__(self):
                self.id = email_id or f"meeting_{hash((subject, body))}"
                self.subject = subject
                self.body_text = body
                self.from_address = sender

        email = EmailLike()

        result = await detector.detect_meeting(
            email,
            check_calendar=check_calendar,
        )

        return success_response(result.to_dict())

    except Exception as e:
        logger.exception("Failed to detect meeting")
        return error_response(f"Detection failed: {str(e)}", status=500)


async def handle_auto_snooze_meeting(
    data: Dict[str, Any],
    user_id: str = "default",
) -> Dict[str, Any]:
    """
    Auto-snooze meeting emails until before meeting time.

    POST /api/v1/inbox/meetings/auto-snooze
    Body: {
        email_id: str,
        subject: str,
        body: str,
        sender: str,
        minutes_before: int (optional, default 30)
    }
    """
    try:
        detector = get_meeting_detector()

        email_id = data.get("email_id", "")
        subject = data.get("subject", "")
        body = data.get("body", "")
        sender = data.get("sender", "")
        minutes_before = int(data.get("minutes_before", 30))

        if not body and not subject:
            return error_response("Either subject or body is required", status=400)

        # Create email-like object
        class EmailLike:
            def __init__(self):
                self.id = email_id or f"snooze_{hash((subject, body))}"
                self.subject = subject
                self.body_text = body
                self.from_address = sender

        email = EmailLike()

        result = await detector.detect_meeting(email, check_calendar=False)

        if not result.is_meeting:
            return success_response(
                {
                    "email_id": email_id,
                    "is_meeting": False,
                    "snooze_scheduled": False,
                    "message": "Email does not appear to be meeting-related",
                }
            )

        if not result.start_time:
            return success_response(
                {
                    "email_id": email_id,
                    "is_meeting": True,
                    "snooze_scheduled": False,
                    "message": "Could not determine meeting time for snooze",
                }
            )

        # Calculate snooze time
        snooze_until = result.start_time - timedelta(minutes=minutes_before)
        now = datetime.now(timezone.utc)

        if snooze_until <= now:
            return success_response(
                {
                    "email_id": email_id,
                    "is_meeting": True,
                    "snooze_scheduled": False,
                    "message": "Meeting is too soon to snooze",
                    "meeting_start": result.start_time.isoformat(),
                }
            )

        return success_response(
            {
                "email_id": email_id,
                "is_meeting": True,
                "meeting_type": result.meeting_type.value,
                "meeting_title": result.title,
                "meeting_start": result.start_time.isoformat(),
                "snooze_scheduled": True,
                "snooze_until": snooze_until.isoformat(),
                "minutes_before": minutes_before,
                "meeting_links": [ml.to_dict() for ml in result.meeting_links],
            }
        )

    except Exception as e:
        logger.exception("Failed to auto-snooze meeting")
        return error_response(f"Auto-snooze failed: {str(e)}", status=500)


# =============================================================================
# Handler Registration
# =============================================================================


def get_action_items_handlers() -> Dict[str, Any]:
    """Get all action items handlers for registration."""
    return {
        "extract_action_items": handle_extract_action_items,
        "list_pending_actions": handle_list_pending_actions,
        "complete_action": handle_complete_action,
        "update_action_status": handle_update_action_status,
        "get_due_soon": handle_get_due_soon,
        "batch_extract": handle_batch_extract,
        "detect_meeting": handle_detect_meeting,
        "auto_snooze_meeting": handle_auto_snooze_meeting,
    }


__all__ = [
    "handle_extract_action_items",
    "handle_list_pending_actions",
    "handle_complete_action",
    "handle_update_action_status",
    "handle_get_due_soon",
    "handle_batch_extract",
    "handle_detect_meeting",
    "handle_auto_snooze_meeting",
    "get_action_items_handlers",
]
