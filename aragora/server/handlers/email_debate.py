"""
Email Vetted Decisionmaking HTTP Handler.

Provides REST API endpoints for multi-agent email vetted decisionmaking:
- POST /api/v1/email/prioritize - Prioritize a single email
- POST /api/v1/email/prioritize/batch - Prioritize multiple emails
- POST /api/v1/email/triage - Triage inbox with full categorization
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from aragora.rbac.decorators import require_permission

logger = logging.getLogger(__name__)


class EmailDebateHandler(BaseHandler):
    """
    Handler for email vetted decisionmaking API endpoints.

    Provides multi-agent email prioritization and triage.
    """

    ROUTES = [
        "/api/v1/email/prioritize",
        "/api/v1/email/prioritize/batch",
        "/api/v1/email/triage",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the request."""
        return path in self.ROUTES

    @require_permission("email:read")
    def handle(self, path: str, query_params: dict, handler=None) -> Optional[HandlerResult]:
        """Handle GET requests (not supported)."""
        return error_response("Use POST method for email vetted decisionmaking", 405)

    @require_permission("email:create")
    def handle_post(self, path: str, query_params: dict, handler=None) -> Optional[HandlerResult]:
        """Handle POST requests."""
        if path == "/api/v1/email/prioritize":
            return self._prioritize_single(handler)
        elif path == "/api/v1/email/prioritize/batch":
            return self._prioritize_batch(handler)
        elif path == "/api/v1/email/triage":
            return self._triage_inbox(handler)
        return None

    def _prioritize_single(self, handler) -> HandlerResult:
        """
        Prioritize a single email.

        Expected body:
        {
            "subject": "Meeting tomorrow",
            "body": "Hi, can we meet tomorrow at 3pm?",
            "sender": "john@example.com",
            "received_at": "2024-01-15T10:30:00Z",
            "message_id": "msg-123",
            "user_id": "user-456"
        }
        """
        import asyncio

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if not body.get("subject") and not body.get("body"):
            return error_response("Missing required field: subject or body", 400)

        try:
            from aragora.services.email_debate import EmailDebateService, EmailInput

            service = EmailDebateService(
                fast_mode=body.get("fast_mode", True),
                enable_pii_redaction=body.get("enable_pii_redaction", True),
            )

            # Parse received_at
            received_at = datetime.now(timezone.utc)
            if body.get("received_at"):
                try:
                    received_at = datetime.fromisoformat(body["received_at"].replace("Z", "+00:00"))
                except ValueError:
                    pass

            email = EmailInput(
                subject=body.get("subject", ""),
                body=body.get("body", ""),
                sender=body.get("sender", ""),
                received_at=received_at,
                message_id=body.get("message_id"),
                recipients=body.get("recipients", []),
                cc=body.get("cc", []),
                attachments=body.get("attachments", []),
            )

            user_id = body.get("user_id", "default")

            # Run async in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(service.prioritize_email(email, user_id))
            finally:
                loop.close()

            return json_response(result.to_dict())

        except Exception as e:
            logger.exception(f"Email prioritization failed: {e}")
            return error_response(f"Prioritization failed: {e}", 500)

    def _prioritize_batch(self, handler) -> HandlerResult:
        """
        Prioritize multiple emails.

        Expected body:
        {
            "emails": [
                {
                    "subject": "...",
                    "body": "...",
                    "sender": "...",
                    "received_at": "...",
                    "message_id": "..."
                }
            ],
            "user_id": "user-456",
            "max_concurrent": 5
        }
        """
        import asyncio

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if not body.get("emails"):
            return error_response("Missing required field: emails", 400)

        try:
            from aragora.services.email_debate import EmailDebateService, EmailInput

            service = EmailDebateService(
                fast_mode=body.get("fast_mode", True),
                enable_pii_redaction=body.get("enable_pii_redaction", True),
            )

            emails = []
            for e in body["emails"]:
                received_at = datetime.now(timezone.utc)
                if e.get("received_at"):
                    try:
                        received_at = datetime.fromisoformat(
                            e["received_at"].replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass

                emails.append(
                    EmailInput(
                        subject=e.get("subject", ""),
                        body=e.get("body", ""),
                        sender=e.get("sender", ""),
                        received_at=received_at,
                        message_id=e.get("message_id"),
                        recipients=e.get("recipients", []),
                        cc=e.get("cc", []),
                        attachments=e.get("attachments", []),
                    )
                )

            user_id = body.get("user_id", "default")
            max_concurrent = body.get("max_concurrent", 5)

            # Run async in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    service.prioritize_batch(emails, user_id, max_concurrent)
                )
            finally:
                loop.close()

            return json_response(
                {
                    "results": [r.to_dict() for r in result.results],
                    "total_emails": result.total_emails,
                    "processed_emails": result.processed_emails,
                    "duration_seconds": result.duration_seconds,
                    "urgent_count": result.urgent_count,
                    "action_required_count": result.action_required_count,
                    "errors": result.errors,
                }
            )

        except Exception as e:
            logger.exception(f"Batch prioritization failed: {e}")
            return error_response(f"Batch prioritization failed: {e}", 500)

    def _triage_inbox(self, handler) -> HandlerResult:
        """
        Full inbox triage with categorization and sorting.

        Expected body:
        {
            "emails": [...],
            "user_id": "user-456",
            "sort_by": "priority",
            "group_by": "category"
        }
        """
        import asyncio

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if not body.get("emails"):
            return error_response("Missing required field: emails", 400)

        try:
            from aragora.services.email_debate import EmailDebateService, EmailInput

            service = EmailDebateService(
                fast_mode=body.get("fast_mode", True),
                enable_pii_redaction=body.get("enable_pii_redaction", True),
            )

            emails = []
            for e in body["emails"]:
                received_at = datetime.now(timezone.utc)
                if e.get("received_at"):
                    try:
                        received_at = datetime.fromisoformat(
                            e["received_at"].replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass

                emails.append(
                    EmailInput(
                        subject=e.get("subject", ""),
                        body=e.get("body", ""),
                        sender=e.get("sender", ""),
                        received_at=received_at,
                        message_id=e.get("message_id"),
                        recipients=e.get("recipients", []),
                        cc=e.get("cc", []),
                        attachments=e.get("attachments", []),
                    )
                )

            user_id = body.get("user_id", "default")

            # Run async in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(service.prioritize_batch(emails, user_id))
            finally:
                loop.close()

            # Sort results (sort_by parameter reserved for future use)
            _sort_by = body.get("sort_by", "priority")
            priority_order = {"urgent": 0, "high": 1, "normal": 2, "low": 3, "spam": 4}

            sorted_results = sorted(
                result.results,
                key=lambda r: (
                    priority_order.get(r.priority.value, 5),
                    -r.confidence,
                ),
            )

            # Group by category if requested
            group_by = body.get("group_by")
            if group_by == "category":
                grouped: Dict[str, list] = {}
                for r in sorted_results:
                    key = r.category.value
                    if key not in grouped:
                        grouped[key] = []
                    grouped[key].append(r.to_dict())

                return json_response(
                    {
                        "grouped": grouped,
                        "total_emails": result.total_emails,
                        "urgent_count": result.urgent_count,
                        "action_required_count": result.action_required_count,
                        "duration_seconds": result.duration_seconds,
                    }
                )
            elif group_by == "priority":
                grouped = result.by_priority
                return json_response(
                    {
                        "grouped": {k: [r.to_dict() for r in v] for k, v in grouped.items()},
                        "total_emails": result.total_emails,
                        "urgent_count": result.urgent_count,
                        "action_required_count": result.action_required_count,
                        "duration_seconds": result.duration_seconds,
                    }
                )

            return json_response(
                {
                    "results": [r.to_dict() for r in sorted_results],
                    "total_emails": result.total_emails,
                    "urgent_count": result.urgent_count,
                    "action_required_count": result.action_required_count,
                    "duration_seconds": result.duration_seconds,
                }
            )

        except Exception as e:
            logger.exception(f"Inbox triage failed: {e}")
            return error_response(f"Inbox triage failed: {e}", 500)


__all__ = ["EmailDebateHandler"]
