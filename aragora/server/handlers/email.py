"""
HTTP API Handlers for Email Prioritization.

Provides REST and WebSocket APIs for intelligent email inbox management:
- Email scoring and prioritization
- Inbox ranking
- User feedback and learning
- Cross-channel context
- Gmail OAuth integration

Endpoints:
- POST /api/email/prioritize - Score a single email
- POST /api/email/rank-inbox - Rank multiple emails
- POST /api/email/feedback - Record user action for learning
- GET /api/email/context/:email_address - Get cross-channel context
- POST /api/email/gmail/oauth/url - Get Gmail OAuth URL
- POST /api/email/gmail/oauth/callback - Handle OAuth callback
- GET /api/email/inbox - Fetch and rank inbox
- GET /api/email/config - Get prioritization config
- PUT /api/email/config - Update prioritization config
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
    require_permission,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Persistent Storage
# =============================================================================

_email_store = None
_email_store_lock = threading.Lock()


def get_email_store():
    """Get or create the email store (lazy init, thread-safe)."""
    global _email_store
    if _email_store is not None:
        return _email_store

    with _email_store_lock:
        if _email_store is None:
            try:
                from aragora.storage.email_store import get_email_store as _get_store

                _email_store = _get_store()
                logger.info("[EmailHandler] Initialized persistent email store")
            except Exception as e:
                logger.warning(f"[EmailHandler] Failed to init email store: {e}")
        return _email_store


def _load_config_from_store(user_id: str, workspace_id: str = "default") -> Dict[str, Any]:
    """Load config from persistent store into memory cache."""
    store = get_email_store()
    if store:
        try:
            config = store.get_user_config(user_id, workspace_id)
            if config:
                return config
        except Exception as e:
            logger.warning(f"[EmailHandler] Failed to load config from store: {e}")
    return {}


def _save_config_to_store(
    user_id: str, config: Dict[str, Any], workspace_id: str = "default"
) -> None:
    """Save config to persistent store."""
    store = get_email_store()
    if store:
        try:
            store.save_user_config(user_id, workspace_id, config)
        except Exception as e:
            logger.warning(f"[EmailHandler] Failed to save config to store: {e}")


# Global instances (initialized lazily) with thread-safe access
_gmail_connector: Optional[Any] = None
_gmail_connector_lock = threading.Lock()
_prioritizer: Optional[Any] = None
_prioritizer_lock = threading.Lock()
_context_service: Optional[Any] = None
_context_service_lock = threading.Lock()
_user_configs: Dict[str, Dict[str, Any]] = {}
_user_configs_lock = threading.Lock()


def get_gmail_connector(user_id: str = "default"):
    """Get or create Gmail connector for a user (thread-safe)."""
    global _gmail_connector
    if _gmail_connector is not None:
        return _gmail_connector

    with _gmail_connector_lock:
        # Double-check after acquiring lock
        if _gmail_connector is None:
            from aragora.connectors.enterprise.communication.gmail import GmailConnector

            _gmail_connector = GmailConnector()
        return _gmail_connector


def get_prioritizer(user_id: str = "default"):
    """Get or create email prioritizer for a user (thread-safe)."""
    global _prioritizer
    if _prioritizer is not None:
        return _prioritizer

    with _prioritizer_lock:
        # Double-check after acquiring lock
        if _prioritizer is None:
            from aragora.services.email_prioritization import (
                EmailPrioritizer,
                EmailPrioritizationConfig,
            )

            # Load user config if available (thread-safe access)
            with _user_configs_lock:
                config_data = _user_configs.get(user_id, {}).copy()

            config = EmailPrioritizationConfig(
                vip_domains=set(config_data.get("vip_domains", [])),
                vip_addresses=set(config_data.get("vip_addresses", [])),
                internal_domains=set(config_data.get("internal_domains", [])),
                auto_archive_senders=set(config_data.get("auto_archive_senders", [])),
            )

            _prioritizer = EmailPrioritizer(
                gmail_connector=get_gmail_connector(user_id),
                config=config,
            )
        return _prioritizer


def get_context_service():
    """Get or create cross-channel context service (thread-safe)."""
    global _context_service
    if _context_service is not None:
        return _context_service

    with _context_service_lock:
        # Double-check after acquiring lock
        if _context_service is None:
            from aragora.services.cross_channel_context import CrossChannelContextService

            _context_service = CrossChannelContextService()
        return _context_service


# =============================================================================
# Email Prioritization Handlers
# =============================================================================


async def handle_prioritize_email(
    email_data: Dict[str, Any],
    user_id: str = "default",
    workspace_id: str = "default",
    force_tier: Optional[str] = None,
) -> Dict[str, Any]:
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
            date=datetime.fromisoformat(email_data["date"])
            if email_data.get("date")
            else datetime.now(),
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


async def handle_rank_inbox(
    emails: List[Dict[str, Any]],
    user_id: str = "default",
    workspace_id: str = "default",
    limit: Optional[int] = None,
) -> Dict[str, Any]:
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
                date=datetime.fromisoformat(email_data["date"])
                if email_data.get("date")
                else datetime.now(),
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


async def handle_email_feedback(
    email_id: str,
    action: str,
    user_id: str = "default",
    workspace_id: str = "default",
    email_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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


# =============================================================================
# Cross-Channel Context Handlers
# =============================================================================


async def handle_get_context(
    email_address: str,
    user_id: str = "default",
    workspace_id: str = "default",
) -> Dict[str, Any]:
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

    except Exception as e:
        logger.exception(f"Failed to get context: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_get_email_context_boost(
    email_data: Dict[str, Any],
    user_id: str = "default",
    workspace_id: str = "default",
) -> Dict[str, Any]:
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

    except Exception as e:
        logger.exception(f"Failed to get context boost: {e}")
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# Email Categorization Handlers
# =============================================================================


# Global categorizer instance
_categorizer: Optional[Any] = None
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


async def handle_categorize_email(
    email_data: Dict[str, Any],
    user_id: str = "default",
    workspace_id: str = "default",
) -> Dict[str, Any]:
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
            date=datetime.fromisoformat(email_data["date"])
            if email_data.get("date")
            else datetime.now(),
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


async def handle_categorize_batch(
    emails: List[Dict[str, Any]],
    user_id: str = "default",
    workspace_id: str = "default",
    concurrency: int = 10,
) -> Dict[str, Any]:
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
        email_objects: List[EmailMessage] = []
        for email_data in emails:
            email = EmailMessage(
                id=email_data.get("id", f"unknown_{len(email_objects)}"),
                thread_id=email_data.get("thread_id", email_data.get("id", "unknown")),
                subject=email_data.get("subject", ""),
                from_address=email_data.get("from_address", ""),
                to_addresses=email_data.get("to_addresses", []),
                cc_addresses=email_data.get("cc_addresses", []),
                bcc_addresses=email_data.get("bcc_addresses", []),
                date=datetime.fromisoformat(email_data["date"])
                if email_data.get("date")
                else datetime.now(),
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


async def handle_feedback_batch(
    feedback_items: List[Dict[str, Any]],
    user_id: str = "default",
    workspace_id: str = "default",
) -> Dict[str, Any]:
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


async def handle_apply_category_label(
    email_id: str,
    category: str,
    user_id: str = "default",
    workspace_id: str = "default",
) -> Dict[str, Any]:
    """
    Apply Gmail label based on category.

    POST /api/v1/email/categorize/apply-label
    {
        "email_id": "msg_123",
        "category": "invoices"
    }
    """
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


# =============================================================================
# Gmail OAuth Handlers
# =============================================================================


async def handle_gmail_oauth_url(
    redirect_uri: str,
    state: str = "",
    scopes: str = "readonly",  # "readonly" or "full"
) -> Dict[str, Any]:
    """
    Get Gmail OAuth authorization URL.

    POST /api/email/gmail/oauth/url
    {
        "redirect_uri": "https://app.example.com/oauth/callback",
        "state": "user_123",
        "scopes": "full"
    }
    """
    try:
        connector = get_gmail_connector()

        # Set scopes based on request
        if scopes == "full":
            from aragora.connectors.enterprise.communication.gmail import GMAIL_SCOPES_FULL

            connector._scopes = GMAIL_SCOPES_FULL
        else:
            from aragora.connectors.enterprise.communication.gmail import GMAIL_SCOPES_READONLY

            connector._scopes = GMAIL_SCOPES_READONLY

        url = connector.get_oauth_url(redirect_uri, state)

        return {
            "success": True,
            "oauth_url": url,
            "scopes": scopes,
        }

    except Exception as e:
        logger.exception(f"Failed to get OAuth URL: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_gmail_oauth_callback(
    code: str,
    redirect_uri: str,
    user_id: str = "default",
    workspace_id: str = "default",
) -> Dict[str, Any]:
    """
    Handle Gmail OAuth callback and store tokens.

    POST /api/email/gmail/oauth/callback
    {
        "code": "auth_code_from_google",
        "redirect_uri": "https://app.example.com/oauth/callback"
    }
    """
    try:
        connector = get_gmail_connector(user_id)
        await connector.authenticate(code=code, redirect_uri=redirect_uri)

        # Get user info to confirm
        user_info = await connector.get_user_info()

        return {
            "success": True,
            "authenticated": True,
            "email": user_info.get("emailAddress"),
            "messages_total": user_info.get("messagesTotal"),
        }

    except Exception as e:
        logger.exception(f"OAuth callback failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_gmail_status(
    user_id: str = "default",
    workspace_id: str = "default",
) -> Dict[str, Any]:
    """
    Check Gmail connection status.

    GET /api/email/gmail/status
    """
    try:
        connector = get_gmail_connector(user_id)
        is_authenticated = connector._access_token is not None

        result = {
            "success": True,
            "authenticated": is_authenticated,
        }

        if is_authenticated:
            try:
                user_info = await connector.get_user_info()
                result["email"] = user_info.get("emailAddress")
                result["messages_total"] = user_info.get("messagesTotal")
            except (KeyError, AttributeError) as e:
                logger.debug(f"Failed to extract user info fields: {e}")
                result["authenticated"] = False
                result["error"] = "Token expired or invalid"  # type: ignore[assignment]
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Network error checking Gmail status: {e}")
                result["authenticated"] = False
                result["error"] = "Token expired or invalid"  # type: ignore[assignment]
            except Exception as e:
                logger.warning(f"Unexpected error checking Gmail status: {e}")
                result["authenticated"] = False
                result["error"] = "Token expired or invalid"  # type: ignore[assignment]

        return result

    except Exception as e:
        logger.exception(f"Failed to check Gmail status: {e}")
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# Inbox Fetch and Rank Handler
# =============================================================================


async def handle_fetch_and_rank_inbox(
    user_id: str = "default",
    workspace_id: str = "default",
    labels: Optional[List[str]] = None,
    limit: int = 50,
    include_read: bool = False,
) -> Dict[str, Any]:
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
            "error": str(e),
        }


# =============================================================================
# Configuration Handlers
# =============================================================================


async def handle_get_config(
    user_id: str = "default",
    workspace_id: str = "default",
) -> Dict[str, Any]:
    """
    Get email prioritization configuration.

    GET /api/email/config

    Now loads from persistent store with in-memory cache fallback.
    """
    # Thread-safe read with snapshot
    with _user_configs_lock:
        config = _user_configs.get(user_id, {}).copy()

    # If not in memory, try loading from persistent store
    if not config:
        config = _load_config_from_store(user_id, workspace_id)
        if config:
            # Cache in memory
            with _user_configs_lock:
                _user_configs[user_id] = config.copy()

    return {
        "success": True,
        "config": {
            "vip_domains": list(config.get("vip_domains", [])),
            "vip_addresses": list(config.get("vip_addresses", [])),
            "internal_domains": list(config.get("internal_domains", [])),
            "auto_archive_senders": list(config.get("auto_archive_senders", [])),
            "tier_1_confidence_threshold": config.get("tier_1_confidence_threshold", 0.7),
            "tier_2_confidence_threshold": config.get("tier_2_confidence_threshold", 0.6),
            "enable_slack_signals": config.get("enable_slack_signals", True),
            "enable_calendar_signals": config.get("enable_calendar_signals", True),
            "enable_drive_signals": config.get("enable_drive_signals", True),
        },
    }


@require_permission("admin:system")
async def handle_update_config(
    user_id: str = "default",
    config_updates: Dict[str, Any] = None,
    workspace_id: str = "default",
) -> Dict[str, Any]:
    """
    Update email prioritization configuration.

    PUT /api/email/config
    {
        "vip_domains": ["importantclient.com"],
        "vip_addresses": ["ceo@company.com"],
        "internal_domains": ["mycompany.com"],
        "auto_archive_senders": ["newsletter@example.com"]
    }

    Now persists to SQLite for durability across restarts.
    """
    global _prioritizer

    try:
        if config_updates is None:
            config_updates = {}

        # Thread-safe config update
        with _user_configs_lock:
            # Get or create user config (load from store if not in memory)
            if user_id not in _user_configs:
                _user_configs[user_id] = _load_config_from_store(user_id, workspace_id)

            # Update config
            user_config = _user_configs[user_id]

            if "vip_domains" in config_updates:
                user_config["vip_domains"] = config_updates["vip_domains"]
            if "vip_addresses" in config_updates:
                user_config["vip_addresses"] = config_updates["vip_addresses"]
            if "internal_domains" in config_updates:
                user_config["internal_domains"] = config_updates["internal_domains"]
            if "auto_archive_senders" in config_updates:
                user_config["auto_archive_senders"] = config_updates["auto_archive_senders"]
            if "tier_1_confidence_threshold" in config_updates:
                user_config["tier_1_confidence_threshold"] = config_updates[
                    "tier_1_confidence_threshold"
                ]
            if "tier_2_confidence_threshold" in config_updates:
                user_config["tier_2_confidence_threshold"] = config_updates[
                    "tier_2_confidence_threshold"
                ]
            if "enable_slack_signals" in config_updates:
                user_config["enable_slack_signals"] = config_updates["enable_slack_signals"]
            if "enable_calendar_signals" in config_updates:
                user_config["enable_calendar_signals"] = config_updates["enable_calendar_signals"]
            if "enable_drive_signals" in config_updates:
                user_config["enable_drive_signals"] = config_updates["enable_drive_signals"]

            # Persist to store
            _save_config_to_store(user_id, user_config, workspace_id)

        # Reset prioritizer to pick up new config (thread-safe)
        with _prioritizer_lock:
            _prioritizer = None

        return {
            "success": True,
            "config": (await handle_get_config(user_id, workspace_id))["config"],
        }

    except Exception as e:
        logger.exception(f"Failed to update config: {e}")
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# VIP Management Handlers
# =============================================================================


async def handle_add_vip(
    user_id: str = "default",
    email: Optional[str] = None,
    domain: Optional[str] = None,
    workspace_id: str = "default",
) -> Dict[str, Any]:
    """
    Add a VIP email or domain.

    POST /api/email/vip
    {
        "email": "important@example.com"
    }
    or
    {
        "domain": "importantcompany.com"
    }

    Now persists to SQLite for durability.
    """
    global _prioritizer

    try:
        # Thread-safe config update
        with _user_configs_lock:
            if user_id not in _user_configs:
                _user_configs[user_id] = _load_config_from_store(user_id, workspace_id)

            config = _user_configs[user_id]

            if email:
                if "vip_addresses" not in config:
                    config["vip_addresses"] = []
                if email not in config["vip_addresses"]:
                    config["vip_addresses"].append(email)
                # Also add to dedicated VIP table for fast lookups
                store = get_email_store()
                if store:
                    try:
                        store.add_vip_sender(user_id, workspace_id, email)
                    except Exception as e:
                        logger.debug(f"Failed to add VIP sender to store: {e}")

            if domain:
                if "vip_domains" not in config:
                    config["vip_domains"] = []
                if domain not in config["vip_domains"]:
                    config["vip_domains"].append(domain)

            # Persist to store
            _save_config_to_store(user_id, config, workspace_id)

            result_addresses = list(config.get("vip_addresses", []))
            result_domains = list(config.get("vip_domains", []))

        # Reset prioritizer (thread-safe)
        with _prioritizer_lock:
            _prioritizer = None

        return {
            "success": True,
            "added": {"email": email, "domain": domain},
            "vip_addresses": result_addresses,
            "vip_domains": result_domains,
        }

    except Exception as e:
        logger.exception(f"Failed to add VIP: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_remove_vip(
    user_id: str = "default",
    email: Optional[str] = None,
    domain: Optional[str] = None,
    workspace_id: str = "default",
) -> Dict[str, Any]:
    """
    Remove a VIP email or domain.

    DELETE /api/email/vip
    {
        "email": "notimportant@example.com"
    }

    Now persists removal to SQLite.
    """
    global _prioritizer

    try:
        # Thread-safe config update
        with _user_configs_lock:
            if user_id not in _user_configs:
                # Load from store first
                _user_configs[user_id] = _load_config_from_store(user_id, workspace_id)

            config = _user_configs[user_id]
            removed = {"email": None, "domain": None}

            if email and "vip_addresses" in config:
                if email in config["vip_addresses"]:
                    config["vip_addresses"].remove(email)
                    removed["email"] = email  # type: ignore[assignment]
                    # Also remove from dedicated VIP table
                    store = get_email_store()
                    if store:
                        try:
                            store.remove_vip_sender(user_id, workspace_id, email)
                        except Exception as e:
                            logger.debug(f"Failed to remove VIP sender from store: {e}")

            if domain and "vip_domains" in config:
                if domain in config["vip_domains"]:
                    config["vip_domains"].remove(domain)
                    removed["domain"] = domain  # type: ignore[assignment]

            # Persist to store
            _save_config_to_store(user_id, config, workspace_id)

            result_addresses = list(config.get("vip_addresses", []))
            result_domains = list(config.get("vip_domains", []))

        # Reset prioritizer (thread-safe)
        with _prioritizer_lock:
            _prioritizer = None

        return {
            "success": True,
            "removed": removed,
            "vip_addresses": result_addresses,
            "vip_domains": result_domains,
        }

    except Exception as e:
        logger.exception(f"Failed to remove VIP: {e}")
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# Handler Class (for integration with server routing)
# =============================================================================


class EmailHandler(BaseHandler):
    """
    HTTP handler for email prioritization endpoints.

    Integrates with the Aragora server routing system.
    """

    ROUTES = [
        "/api/v1/email/prioritize",
        "/api/v1/email/rank-inbox",
        "/api/v1/email/feedback",
        "/api/v1/email/feedback/batch",
        "/api/v1/email/inbox",
        "/api/v1/email/config",
        "/api/v1/email/vip",
        "/api/v1/email/categorize",
        "/api/v1/email/categorize/batch",
        "/api/v1/email/categorize/apply-label",
        "/api/v1/email/gmail/oauth/url",
        "/api/v1/email/gmail/oauth/callback",
        "/api/v1/email/gmail/status",
        "/api/v1/email/context/boost",
    ]

    # Prefix for dynamic routes like /api/email/context/:email_address
    ROUTE_PREFIXES = ["/api/v1/email/context/"]

    def __init__(self, ctx: Dict[str, Any]):
        """Initialize with server context."""
        super().__init__(ctx)  # type: ignore[arg-type]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        if path in self.ROUTES:
            return True
        # Check prefix routes (e.g., /api/email/context/:email_address)
        for prefix in self.ROUTE_PREFIXES:
            if path.startswith(prefix) and path != prefix.rstrip("/"):
                return True
        return False

    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route email endpoint requests."""
        # This handler uses async methods, so we return None here
        # and let the server's async handling mechanism process it
        # The actual handling is done via HTTP method-specific handlers
        return None

    async def handle_post_prioritize(self, data: Dict[str, Any]) -> HandlerResult:
        """POST /api/email/prioritize"""
        email_data = data.get("email", {})
        force_tier = data.get("force_tier")
        user_id = self._get_user_id()

        result = await handle_prioritize_email(email_data, user_id, force_tier)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_rank_inbox(self, data: Dict[str, Any]) -> HandlerResult:
        """POST /api/email/rank-inbox"""
        emails = data.get("emails", [])
        limit = data.get("limit")
        user_id = self._get_user_id()

        result = await handle_rank_inbox(emails, user_id, limit)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_feedback(self, data: Dict[str, Any]) -> HandlerResult:
        """POST /api/email/feedback"""
        email_id = data.get("email_id")
        action = data.get("action")
        email_data = data.get("email")
        user_id = self._get_user_id()

        if not email_id or not action:
            return error_response("email_id and action required", 400)

        result = await handle_email_feedback(email_id, action, user_id, email_data)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_feedback_batch(self, data: Dict[str, Any]) -> HandlerResult:
        """POST /api/email/feedback/batch"""
        items = data.get("items", [])
        user_id = self._get_user_id()

        if not items:
            return error_response("items array required", 400)

        result = await handle_feedback_batch(items, user_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_categorize(self, data: Dict[str, Any]) -> HandlerResult:
        """POST /api/email/categorize"""
        email_data = data.get("email", {})
        user_id = self._get_user_id()

        result = await handle_categorize_email(email_data, user_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_categorize_batch(self, data: Dict[str, Any]) -> HandlerResult:
        """POST /api/email/categorize/batch"""
        emails = data.get("emails", [])
        concurrency = data.get("concurrency", 10)
        user_id = self._get_user_id()

        if not emails:
            return error_response("emails array required", 400)

        result = await handle_categorize_batch(emails, user_id, concurrency)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_categorize_apply_label(self, data: Dict[str, Any]) -> HandlerResult:
        """POST /api/email/categorize/apply-label"""
        email_id = data.get("email_id")
        category = data.get("category")
        user_id = self._get_user_id()

        if not email_id or not category:
            return error_response("email_id and category required", 400)

        result = await handle_apply_category_label(email_id, category, user_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_inbox(self, params: Dict[str, Any]) -> HandlerResult:
        """GET /api/email/inbox"""
        user_id = self._get_user_id()
        labels = params.get("labels", "").split(",") if params.get("labels") else None
        limit = int(params.get("limit", 50))
        include_read = params.get("include_read", "").lower() == "true"

        result = await handle_fetch_and_rank_inbox(
            user_id=user_id,
            labels=labels,
            limit=limit,
            include_read=include_read,
        )

        if result.get("success"):
            return success_response(result)
        elif result.get("needs_auth"):
            return error_response(result.get("error"), 401)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_config(self, params: Dict[str, Any]) -> HandlerResult:
        """GET /api/email/config"""
        user_id = self._get_user_id()
        result = await handle_get_config(user_id)
        return success_response(result)

    async def handle_put_config(self, data: Dict[str, Any]) -> HandlerResult:
        """PUT /api/email/config"""
        user_id = self._get_user_id()
        result = await handle_update_config(user_id, data)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_vip(self, data: Dict[str, Any]) -> HandlerResult:
        """POST /api/email/vip"""
        user_id = self._get_user_id()
        email = data.get("email")
        domain = data.get("domain")

        result = await handle_add_vip(user_id, email, domain)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_delete_vip(self, data: Dict[str, Any]) -> HandlerResult:
        """DELETE /api/email/vip"""
        user_id = self._get_user_id()
        email = data.get("email")
        domain = data.get("domain")

        result = await handle_remove_vip(user_id, email, domain)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_gmail_oauth_url(self, data: Dict[str, Any]) -> HandlerResult:
        """POST /api/email/gmail/oauth/url"""
        redirect_uri = data.get("redirect_uri")
        state = data.get("state", "")
        scopes = data.get("scopes", "readonly")

        if not redirect_uri:
            return error_response("redirect_uri required", 400)

        result = await handle_gmail_oauth_url(redirect_uri, state, scopes)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_gmail_oauth_callback(self, data: Dict[str, Any]) -> HandlerResult:
        """POST /api/email/gmail/oauth/callback"""
        code = data.get("code")
        redirect_uri = data.get("redirect_uri")
        user_id = self._get_user_id()

        if not code or not redirect_uri:
            return error_response("code and redirect_uri required", 400)

        result = await handle_gmail_oauth_callback(code, redirect_uri, user_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_gmail_status(self, params: Dict[str, Any]) -> HandlerResult:
        """GET /api/email/gmail/status"""
        user_id = self._get_user_id()
        result = await handle_gmail_status(user_id)
        return success_response(result)

    async def handle_get_context(self, params: Dict[str, Any], email_address: str) -> HandlerResult:
        """GET /api/email/context/:email_address"""
        user_id = self._get_user_id()
        result = await handle_get_context(email_address, user_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_context_boost(self, data: Dict[str, Any]) -> HandlerResult:
        """POST /api/email/context/boost"""
        email_data = data.get("email", {})
        user_id = self._get_user_id()

        result = await handle_get_email_context_boost(email_data, user_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    def _get_user_id(self) -> str:
        """Get user ID from auth context."""
        # In production, extract from JWT token
        auth_ctx = self.ctx.get("auth_context")
        if auth_ctx and hasattr(auth_ctx, "user_id"):
            return auth_ctx.user_id
        return "default"
