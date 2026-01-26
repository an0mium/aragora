"""
Inbox Command Center API Handler.

Provides unified API endpoints for the inbox command center including:
- Prioritized email fetching with cross-channel context
- Quick actions (archive, snooze, reply, forward)
- Bulk operations
- Daily digest statistics
- Sender profile lookups

Endpoints:
- GET /api/inbox/command - Fetch prioritized inbox
- POST /api/inbox/actions - Execute quick action
- POST /api/inbox/bulk-actions - Execute bulk action
- GET /api/inbox/sender-profile - Get sender profile
- GET /api/inbox/daily-digest - Get daily digest
- POST /api/inbox/reprioritize - Trigger AI re-prioritization
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, TypeVar

from aiohttp import web

from aragora.services import (
    ServiceRegistry,
    EmailPrioritizer,
    EmailPrioritizationConfig,
    SenderHistoryService,
)
from aragora.cache import HybridTTLCache, register_cache
from aragora.utils.redis_cache import RedisTTLCache

if TYPE_CHECKING:
    from aragora.connectors.enterprise.communication.gmail import GmailConnector

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=Dict[str, Any])


class IterableTTLCache(Generic[T]):
    """
    TTL cache wrapper that supports iteration for inbox operations.

    Wraps HybridTTLCache to provide dict-like iteration while maintaining
    Redis persistence for multi-instance deployments.
    """

    def __init__(self, name: str, maxsize: int, ttl_seconds: float) -> None:
        self._cache: RedisTTLCache[T] = HybridTTLCache(
            prefix=name,
            maxsize=maxsize,
            ttl_seconds=ttl_seconds,
        )
        self._keys: set[str] = set()  # Track keys for iteration
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        return self._cache.get(key)

    def set(self, key: str, value: T) -> None:
        """Store value in cache."""
        self._cache.set(key, value)
        with self._lock:
            self._keys.add(key)

    def __setitem__(self, key: str, value: T) -> None:
        """Dict-style assignment."""
        self.set(key, value)

    def __getitem__(self, key: str) -> T:
        """Dict-style access."""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None

    def items(self) -> list[tuple[str, T]]:
        """Return list of (key, value) pairs."""
        result: list[tuple[str, T]] = []
        with self._lock:
            for key in list(self._keys):
                value = self.get(key)
                if value is not None:
                    result.append((key, value))
                else:
                    self._keys.discard(key)
        return result

    def values(self) -> list[T]:
        """Return list of values."""
        return [v for _, v in self.items()]

    def invalidate(self, key: str) -> bool:
        """Remove key from cache."""
        with self._lock:
            self._keys.discard(key)
        return self._cache.invalidate(key)

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        return self._cache.stats


# Production-ready cache for prioritized emails (Redis when available, fallback to in-memory)
_email_cache: IterableTTLCache = IterableTTLCache(
    name="inbox_email_cache",
    maxsize=10000,
    ttl_seconds=3600,  # 1 hour TTL
)
_priority_results: IterableTTLCache = IterableTTLCache(
    name="inbox_priority_results",
    maxsize=1000,
    ttl_seconds=1800,  # 30 min TTL
)

# Register underlying caches for monitoring
register_cache("inbox_email", _email_cache._cache)
register_cache("inbox_priority", _priority_results._cache)


@dataclass
class InboxCommandHandler:
    """Handler for inbox command center API endpoints."""

    gmail_connector: Optional["GmailConnector"] = None
    prioritizer: Optional[EmailPrioritizer] = None
    sender_history: Optional[SenderHistoryService] = None
    _initialized: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize services from registry if not provided."""
        self._ensure_services()

    def _ensure_services(self) -> None:
        """Lazily initialize services from the registry."""
        if self._initialized:
            return

        registry = ServiceRegistry.get()

        # Try to get GmailConnector from registry
        if self.gmail_connector is None:
            try:
                from aragora.connectors.enterprise.communication.gmail import GmailConnector

                if registry.has(GmailConnector):
                    self.gmail_connector = registry.resolve(GmailConnector)
                    logger.debug("Resolved GmailConnector from registry")
            except Exception as e:
                logger.debug(f"GmailConnector not available: {e}")

        # Try to get or create EmailPrioritizer
        if self.prioritizer is None:
            if registry.has(EmailPrioritizer):
                self.prioritizer = registry.resolve(EmailPrioritizer)
                logger.debug("Resolved EmailPrioritizer from registry")
            elif self.gmail_connector is not None:
                # Create a prioritizer with the connector
                self.prioritizer = EmailPrioritizer(
                    gmail_connector=self.gmail_connector,
                    config=EmailPrioritizationConfig(),
                )
                registry.register(EmailPrioritizer, self.prioritizer)
                logger.info("Created and registered EmailPrioritizer")

        # Try to get SenderHistoryService
        if self.sender_history is None:
            if registry.has(SenderHistoryService):
                self.sender_history = registry.resolve(SenderHistoryService)
                logger.debug("Resolved SenderHistoryService from registry")

        self._initialized = True

    async def handle_get_inbox(self, request: web.Request) -> web.Response:
        """
        GET /api/inbox/command

        Fetch prioritized inbox with stats.

        Query params:
            - limit: Max emails to return (default 50)
            - offset: Pagination offset (default 0)
            - priority: Filter by priority level (critical, high, medium, low, defer)
            - unread_only: Only return unread emails (default false)
        """
        try:
            self._ensure_services()

            limit = int(request.query.get("limit", "50"))
            offset = int(request.query.get("offset", "0"))
            priority_filter = request.query.get("priority")
            unread_only = request.query.get("unread_only", "false").lower() == "true"

            # Get emails from service
            emails = await self._fetch_prioritized_emails(
                limit=limit,
                offset=offset,
                priority_filter=priority_filter,
                unread_only=unread_only,
            )

            # Calculate stats
            stats = await self._calculate_inbox_stats(emails)

            return web.json_response(
                {
                    "success": True,
                    "emails": emails,
                    "total": stats["total"],
                    "stats": stats,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        except Exception as e:
            logger.exception(f"Failed to fetch inbox: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    async def handle_quick_action(self, request: web.Request) -> web.Response:
        """
        POST /api/inbox/actions

        Execute quick action on email(s).

        Body:
            - action: Action to perform (archive, snooze, reply, forward, spam, etc.)
            - emailIds: List of email IDs to act on
            - params: Optional action-specific parameters
        """
        try:
            self._ensure_services()

            body = await request.json()
            action = body.get("action")
            email_ids = body.get("emailIds", [])
            params = body.get("params", {})

            if not action:
                return web.json_response(
                    {"success": False, "error": "action is required"},
                    status=400,
                )

            if not email_ids:
                return web.json_response(
                    {"success": False, "error": "emailIds is required"},
                    status=400,
                )

            # Execute action
            results = await self._execute_action(action, email_ids, params)

            return web.json_response(
                {
                    "success": True,
                    "action": action,
                    "processed": len(email_ids),
                    "results": results,
                }
            )
        except Exception as e:
            logger.exception(f"Failed to execute action: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    async def handle_bulk_action(self, request: web.Request) -> web.Response:
        """
        POST /api/inbox/bulk-actions

        Execute bulk action based on filter.

        Body:
            - action: Action to perform
            - filter: Filter to apply (low, deferred, spam, read, all)
            - params: Optional action-specific parameters
        """
        try:
            self._ensure_services()

            body = await request.json()
            action = body.get("action")
            filter_type = body.get("filter")
            params = body.get("params", {})

            if not action or not filter_type:
                return web.json_response(
                    {"success": False, "error": "action and filter are required"},
                    status=400,
                )

            # Get matching email IDs
            email_ids = await self._get_emails_by_filter(filter_type)

            if not email_ids:
                return web.json_response(
                    {
                        "success": True,
                        "action": action,
                        "processed": 0,
                        "message": "No emails matched the filter",
                    }
                )

            # Execute action
            results = await self._execute_action(action, email_ids, params)

            return web.json_response(
                {
                    "success": True,
                    "action": action,
                    "filter": filter_type,
                    "processed": len(email_ids),
                    "results": results,
                }
            )
        except Exception as e:
            logger.exception(f"Failed to execute bulk action: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    async def handle_get_sender_profile(self, request: web.Request) -> web.Response:
        """
        GET /api/inbox/sender-profile

        Get profile information for a sender.

        Query params:
            - email: Sender email address
        """
        try:
            self._ensure_services()

            email = request.query.get("email")
            if not email:
                return web.json_response(
                    {"success": False, "error": "email parameter is required"},
                    status=400,
                )

            profile = await self._get_sender_profile(email)
            return web.json_response({"success": True, "profile": profile})
        except Exception as e:
            logger.exception(f"Failed to get sender profile: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    async def handle_get_daily_digest(self, request: web.Request) -> web.Response:
        """
        GET /api/inbox/daily-digest

        Get daily digest statistics.
        """
        try:
            self._ensure_services()

            digest = await self._calculate_daily_digest()
            return web.json_response({"success": True, "digest": digest})
        except Exception as e:
            logger.exception(f"Failed to get daily digest: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    async def handle_reprioritize(self, request: web.Request) -> web.Response:
        """
        POST /api/inbox/reprioritize

        Trigger AI re-prioritization of inbox.

        Body:
            - emailIds: Optional list of specific email IDs to reprioritize
            - force_tier: Optional tier to force (tier_1_rules, tier_2_lightweight, tier_3_debate)
        """
        try:
            self._ensure_services()

            body = await request.json()
            email_ids = body.get("emailIds")
            force_tier = body.get("force_tier")

            # Trigger reprioritization
            result = await self._reprioritize_emails(email_ids, force_tier)

            return web.json_response(
                {
                    "success": True,
                    "reprioritized": result["count"],
                    "changes": result["changes"],
                    "tier_used": result.get("tier_used"),
                }
            )
        except Exception as e:
            logger.exception(f"Failed to reprioritize: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    # =========================================================================
    # Private helper methods - Service Integration
    # =========================================================================

    async def _fetch_prioritized_emails(
        self,
        limit: int,
        offset: int,
        priority_filter: Optional[str],
        unread_only: bool,
    ) -> List[Dict[str, Any]]:
        """Fetch and prioritize emails using the EmailPrioritizer service."""
        # If no Gmail connector, return demo data
        if self.gmail_connector is None:
            return self._get_demo_emails(limit, offset, priority_filter)

        try:
            # Fetch emails from Gmail
            emails: List[Any] = []
            fetch_limit = limit + offset + 50  # Fetch extra for filtering

            # Use list_messages instead of sync_items for inbox fetching
            messages = await self.gmail_connector.list_messages(max_results=fetch_limit)
            for msg in messages:
                if len(emails) >= fetch_limit:
                    break
                emails.append(msg)

            if not emails:
                return []

            # Prioritize emails
            if self.prioritizer:
                results = await self.prioritizer.rank_inbox(emails, limit=fetch_limit)

                # Convert to response format
                prioritized = []
                for result in results:
                    # Find the original email
                    email_data = next(
                        (e for e in emails if getattr(e, "id", None) == result.email_id),
                        None,
                    )

                    entry = {
                        "id": result.email_id,
                        "from": (
                            getattr(email_data, "from_address", "unknown")
                            if email_data
                            else "unknown"
                        ),
                        "subject": (
                            getattr(email_data, "subject", "No subject")
                            if email_data
                            else "No subject"
                        ),
                        "snippet": getattr(email_data, "snippet", "")[:200] if email_data else "",
                        "priority": result.priority.name.lower(),
                        "confidence": result.confidence,
                        "reasoning": result.rationale,
                        "tier_used": result.tier_used.value,
                        "scores": {
                            "sender": result.sender_score,
                            "urgency": result.content_urgency_score,
                            "context": result.context_relevance_score,
                            "time_sensitivity": result.time_sensitivity_score,
                        },
                        "suggested_labels": result.suggested_labels,
                        "auto_archive": result.auto_archive,
                        "timestamp": (
                            getattr(email_data, "date", datetime.utcnow()).isoformat()
                            if email_data
                            else datetime.utcnow().isoformat()
                        ),
                        "unread": getattr(email_data, "unread", True) if email_data else True,
                    }

                    # Apply filters
                    if priority_filter and entry["priority"] != priority_filter.lower():
                        continue
                    if unread_only and not entry["unread"]:
                        continue

                    prioritized.append(entry)

                    # Cache for bulk operations
                    _email_cache[result.email_id] = entry

                # Apply pagination
                return prioritized[offset : offset + limit]
            else:
                # No prioritizer - return basic list
                return [
                    {
                        "id": getattr(e, "id", f"email_{i}"),
                        "from": getattr(e, "from_address", "unknown"),
                        "subject": getattr(e, "subject", "No subject"),
                        "snippet": getattr(e, "snippet", "")[:200],
                        "priority": "medium",
                        "confidence": 0.5,
                        "timestamp": getattr(e, "date", datetime.utcnow()).isoformat(),
                        "unread": getattr(e, "unread", True),
                    }
                    for i, e in enumerate(emails[offset : offset + limit])
                ]

        except Exception as e:
            logger.warning(f"Failed to fetch from Gmail, using demo data: {e}")
            return self._get_demo_emails(limit, offset, priority_filter)

    def _get_demo_emails(
        self,
        limit: int,
        offset: int,
        priority_filter: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Return demo email data when services aren't available."""
        demo_emails = [
            {
                "id": "demo_1",
                "from": "ceo@company.com",
                "subject": "Q4 Strategy Review - Urgent Response Needed",
                "snippet": "Please review the attached strategy document and provide your feedback by EOD...",
                "priority": "critical",
                "confidence": 0.95,
                "reasoning": "VIP sender; deadline detected; reply expected",
                "tier_used": "tier_1_rules",
                "category": "Work",
                "timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                "unread": True,
            },
            {
                "id": "demo_2",
                "from": "client@bigcorp.com",
                "subject": "Contract renewal discussion",
                "snippet": "Following up on our conversation about the contract renewal. Can we schedule a call?",
                "priority": "high",
                "confidence": 0.85,
                "reasoning": "Client sender; reply expected",
                "tier_used": "tier_1_rules",
                "category": "Work",
                "timestamp": (datetime.utcnow() - timedelta(hours=5)).isoformat(),
                "unread": True,
            },
            {
                "id": "demo_3",
                "from": "notifications@github.com",
                "subject": "[aragora] PR #142: Fix memory leak in debate engine",
                "snippet": "A new pull request has been opened by contributor...",
                "priority": "medium",
                "confidence": 0.75,
                "reasoning": "Automated notification; work-related",
                "tier_used": "tier_1_rules",
                "category": "Updates",
                "timestamp": (datetime.utcnow() - timedelta(hours=8)).isoformat(),
                "unread": True,
            },
            {
                "id": "demo_4",
                "from": "newsletter@techblog.com",
                "subject": "This week in AI: Top stories and insights",
                "snippet": "Unsubscribe | View in browser. This week's top AI stories include...",
                "priority": "defer",
                "confidence": 0.92,
                "reasoning": "Newsletter detected; auto-archive candidate",
                "tier_used": "tier_1_rules",
                "category": "Newsletter",
                "timestamp": (datetime.utcnow() - timedelta(days=1)).isoformat(),
                "unread": False,
                "auto_archive": True,
            },
            {
                "id": "demo_5",
                "from": "team@company.com",
                "subject": "Weekly standup notes",
                "snippet": "Here are the notes from this week's standup meeting...",
                "priority": "low",
                "confidence": 0.8,
                "reasoning": "Internal sender; informational",
                "tier_used": "tier_1_rules",
                "category": "Work",
                "timestamp": (datetime.utcnow() - timedelta(days=1)).isoformat(),
                "unread": False,
            },
        ]

        # Populate cache for bulk actions to work in demo mode
        for email in demo_emails:
            _email_cache[str(email["id"])] = email

        # Apply filters
        if priority_filter:
            demo_emails = [e for e in demo_emails if e["priority"] == priority_filter.lower()]

        return demo_emails[offset : offset + limit]

    async def _calculate_inbox_stats(
        self,
        emails: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate inbox statistics from prioritized emails."""
        total = len(emails)

        # Count by priority
        priority_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "defer": 0,
        }

        for email in emails:
            priority = email.get("priority", "medium").lower()
            if priority in priority_counts:
                priority_counts[priority] += 1

        # Count unread
        unread_count = sum(1 for e in emails if e.get("unread", False))

        return {
            "total": total,
            "unread": unread_count,
            "critical": priority_counts["critical"],
            "high": priority_counts["high"],
            "medium": priority_counts["medium"],
            "low": priority_counts["low"],
            "deferred": priority_counts["defer"],
            "actionRequired": priority_counts["critical"] + priority_counts["high"],
        }

    async def _execute_action(
        self,
        action: str,
        email_ids: List[str],
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Execute action on emails using Gmail connector."""
        results = []
        for email_id in email_ids:
            try:
                result = await self._perform_action(action, email_id, params)
                results.append(
                    {
                        "emailId": email_id,
                        "success": True,
                        "result": result,
                    }
                )

                # Record action for learning
                if self.prioritizer:
                    email_data = _email_cache.get(email_id)
                    await self.prioritizer.record_user_action(
                        email_id=email_id,
                        action=action,
                        email=email_data,  # type: ignore[arg-type]
                    )

            except Exception as e:
                logger.warning(f"Action {action} failed for {email_id}: {e}")
                results.append(
                    {
                        "emailId": email_id,
                        "success": False,
                        "error": str(e),
                    }
                )
        return results

    async def _perform_action(
        self,
        action: str,
        email_id: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform a single action on an email."""
        action_handlers = {
            "archive": self._archive_email,
            "snooze": self._snooze_email,
            "reply": self._create_reply_draft,
            "forward": self._create_forward_draft,
            "spam": self._mark_spam,
            "mark_important": self._mark_important,
            "mark_vip": self._mark_sender_vip,
            "block": self._block_sender,
            "delete": self._delete_email,
        }

        handler = action_handlers.get(action)
        if not handler:
            raise ValueError(f"Unknown action: {action}")

        return await handler(email_id, params)

    async def _archive_email(self, email_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Archive an email via Gmail API."""
        if self.gmail_connector and hasattr(self.gmail_connector, "archive_message"):
            try:
                await self.gmail_connector.archive_message(email_id)  # type: ignore[attr-defined]
                logger.info(f"Archived email {email_id}")
                return {"archived": True}
            except Exception as e:
                logger.warning(f"Gmail archive failed: {e}")

        # Fallback to demo mode
        logger.info(f"[Demo] Archiving email {email_id}")
        return {"archived": True, "demo": True}

    async def _snooze_email(self, email_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Snooze an email."""
        duration = params.get("duration", "1d")
        # Parse duration to snooze until time
        duration_map = {
            "1h": timedelta(hours=1),
            "3h": timedelta(hours=3),
            "1d": timedelta(days=1),
            "3d": timedelta(days=3),
            "1w": timedelta(weeks=1),
        }
        delta = duration_map.get(duration, timedelta(days=1))
        snooze_until = datetime.utcnow() + delta

        if self.gmail_connector and hasattr(self.gmail_connector, "snooze_message"):
            try:
                await self.gmail_connector.snooze_message(email_id, snooze_until)
                logger.info(f"Snoozed email {email_id} until {snooze_until}")
                return {"snoozed": True, "until": snooze_until.isoformat()}
            except Exception as e:
                logger.warning(f"Gmail snooze failed: {e}")

        logger.info(f"[Demo] Snoozing email {email_id} for {duration}")
        return {"snoozed": True, "until": snooze_until.isoformat(), "demo": True}

    async def _create_reply_draft(self, email_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a reply draft."""
        body = params.get("body", "")

        if self.gmail_connector and hasattr(self.gmail_connector, "create_draft"):
            try:
                draft_id = await self.gmail_connector.create_draft(
                    in_reply_to=email_id,
                    body=body,
                )
                logger.info(f"Created reply draft for {email_id}")
                return {"draftId": draft_id}
            except Exception as e:
                logger.warning(f"Gmail draft creation failed: {e}")

        logger.info(f"[Demo] Creating reply draft for {email_id}")
        return {"draftId": f"draft_{email_id}", "demo": True}

    async def _create_forward_draft(self, email_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a forward draft."""
        to = params.get("to", "")

        if self.gmail_connector and hasattr(self.gmail_connector, "create_forward_draft"):
            try:
                draft_id = await self.gmail_connector.create_forward_draft(
                    message_id=email_id,
                    to=to,
                )
                logger.info(f"Created forward draft for {email_id}")
                return {"draftId": draft_id}
            except Exception as e:
                logger.warning(f"Gmail forward draft failed: {e}")

        logger.info(f"[Demo] Creating forward draft for {email_id}")
        return {"draftId": f"draft_fwd_{email_id}", "demo": True}

    async def _mark_spam(self, email_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mark email as spam."""
        if self.gmail_connector and hasattr(self.gmail_connector, "mark_spam"):
            try:
                await self.gmail_connector.mark_spam(email_id)
                logger.info(f"Marked {email_id} as spam")
                return {"spam": True}
            except Exception as e:
                logger.warning(f"Gmail mark spam failed: {e}")

        logger.info(f"[Demo] Marking {email_id} as spam")
        return {"spam": True, "demo": True}

    async def _mark_important(self, email_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mark email as important."""
        if self.gmail_connector and hasattr(self.gmail_connector, "modify_labels"):
            try:
                await self.gmail_connector.modify_labels(
                    email_id,
                    add_labels=["IMPORTANT"],
                )
                logger.info(f"Marked {email_id} as important")
                return {"important": True}
            except Exception as e:
                logger.warning(f"Gmail modify labels failed: {e}")

        logger.info(f"[Demo] Marking {email_id} as important")
        return {"important": True, "demo": True}

    async def _mark_sender_vip(self, email_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mark sender as VIP."""
        email_data = _email_cache.get(email_id)
        sender = email_data.get("from") if email_data else params.get("sender")

        if sender and self.prioritizer:
            # Add to VIP list in config
            self.prioritizer.config.vip_addresses.add(sender)
            logger.info(f"Marked sender {sender} as VIP")
            return {"vip": True, "sender": sender}

        logger.info(f"[Demo] Marking sender of {email_id} as VIP")
        return {"vip": True, "demo": True}

    async def _block_sender(self, email_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Block sender."""
        email_data = _email_cache.get(email_id)
        sender = email_data.get("from") if email_data else params.get("sender")

        if sender and self.prioritizer:
            # Add to auto-archive list
            self.prioritizer.config.auto_archive_senders.add(sender)
            logger.info(f"Blocked sender {sender}")
            return {"blocked": True, "sender": sender}

        logger.info(f"[Demo] Blocking sender of {email_id}")
        return {"blocked": True, "demo": True}

    async def _delete_email(self, email_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete email."""
        if self.gmail_connector and hasattr(self.gmail_connector, "trash_message"):
            try:
                await self.gmail_connector.trash_message(email_id)
                logger.info(f"Deleted email {email_id}")
                return {"deleted": True}
            except Exception as e:
                logger.warning(f"Gmail delete failed: {e}")

        logger.info(f"[Demo] Deleting email {email_id}")
        return {"deleted": True, "demo": True}

    async def _get_emails_by_filter(self, filter_type: str) -> List[str]:
        """Get email IDs matching filter from cache."""
        filter_map = {
            "low": ["low", "defer"],
            "deferred": ["defer"],
            "spam": ["spam"],
            "read": lambda e: not e.get("unread", True),
            "all": None,
        }

        matching_ids = []
        filter_value = filter_map.get(filter_type)

        for email_id, email_data in _email_cache.items():
            if filter_value is None:
                matching_ids.append(email_id)
            elif callable(filter_value):
                if filter_value(email_data):
                    matching_ids.append(email_id)
            elif isinstance(filter_value, list):
                if email_data.get("priority") in filter_value:
                    matching_ids.append(email_id)

        return matching_ids

    async def _get_sender_profile(self, email: str) -> Dict[str, Any]:
        """Get sender profile information from SenderHistoryService."""
        if self.sender_history:
            try:
                stats = await self.sender_history.get_sender_stats(
                    user_id="default",
                    sender_email=email,
                )
                if stats:
                    avg_hours = (
                        stats.avg_response_time_minutes / 60.0
                        if stats.avg_response_time_minutes
                        else None
                    )
                    return {
                        "email": email,
                        "name": email.split("@")[0],
                        "isVip": stats.is_vip,
                        "isInternal": False,
                        "responseRate": stats.reply_rate,
                        "avgResponseTime": f"{avg_hours:.1f}h" if avg_hours else "N/A",
                        "totalEmails": stats.total_emails,
                        "lastContact": (
                            stats.last_email_date.strftime("%Y-%m-%d")
                            if stats.last_email_date
                            else "Never"
                        ),
                    }
            except Exception as e:
                logger.warning(f"Failed to get sender stats: {e}")

        # Check prioritizer config for VIP status
        is_vip = False
        if self.prioritizer:
            is_vip = email.lower() in {a.lower() for a in self.prioritizer.config.vip_addresses}
            domain = email.split("@")[-1] if "@" in email else ""
            is_vip = is_vip or domain.lower() in {
                d.lower() for d in self.prioritizer.config.vip_domains
            }

        # Return basic profile
        return {
            "email": email,
            "name": email.split("@")[0],
            "isVip": is_vip,
            "isInternal": False,
            "responseRate": 0.0,
            "avgResponseTime": "N/A",
            "totalEmails": 0,
            "lastContact": "Unknown",
        }

    async def _calculate_daily_digest(self) -> Dict[str, Any]:
        """Calculate daily digest statistics."""
        # Try to get real stats from sender history (if method exists)
        if self.sender_history and hasattr(self.sender_history, "get_daily_summary"):
            try:
                today_stats = await self.sender_history.get_daily_summary(user_id="default")  # type: ignore[attr-defined]
                if today_stats:
                    return today_stats  # type: ignore[return-value]
            except Exception as e:
                logger.debug(f"Daily summary not available: {e}")

        # Use cached data stats
        emails_in_cache = list(_email_cache.values())
        critical_count = sum(1 for e in emails_in_cache if e.get("priority") == "critical")

        # Compute top senders from cache
        sender_counts: Dict[str, int] = {}
        for email in emails_in_cache:
            sender = email.get("from", "unknown")
            sender_counts[sender] = sender_counts.get(sender, 0) + 1

        sender_list: List[Dict[str, Any]] = [
            {"name": k, "count": v} for k, v in sender_counts.items()
        ]
        top_senders = sorted(
            sender_list,
            key=lambda x: x["count"],
            reverse=True,
        )[:5]

        # Compute category breakdown
        category_counts: Dict[str, int] = {}
        for email in emails_in_cache:
            category = email.get("category", "General")
            category_counts[category] = category_counts.get(category, 0) + 1

        total = len(emails_in_cache) or 1  # Avoid division by zero
        category_breakdown = [
            {
                "category": cat,
                "count": count,
                "percentage": round(count / total * 100),
            }
            for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])
        ]

        return {
            "emailsReceived": len(emails_in_cache),
            "emailsProcessed": len(emails_in_cache),
            "criticalHandled": critical_count,
            "timeSaved": f"{len(emails_in_cache) * 2} min",  # Estimate 2 min saved per email
            "topSenders": top_senders
            or [
                {"name": "team@company.com", "count": 0},
            ],
            "categoryBreakdown": category_breakdown
            or [
                {"category": "General", "count": 0, "percentage": 100},
            ],
        }

    async def _reprioritize_emails(
        self,
        email_ids: Optional[List[str]],
        force_tier: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reprioritize emails using AI."""
        from aragora.services.email_prioritization import ScoringTier

        if not self.prioritizer:
            return {
                "count": 0,
                "changes": [],
                "error": "Prioritizer not available",
            }

        # Get emails to reprioritize
        if email_ids:
            emails_to_process = [
                (eid, _email_cache.get(eid)) for eid in email_ids if eid in _email_cache
            ]
        else:
            emails_to_process = list(_email_cache.items())

        if not emails_to_process:
            return {"count": 0, "changes": []}

        # Map force_tier string to ScoringTier enum
        tier_map = {
            "tier_1_rules": ScoringTier.TIER_1_RULES,
            "tier_2_lightweight": ScoringTier.TIER_2_LIGHTWEIGHT,
            "tier_3_debate": ScoringTier.TIER_3_DEBATE,
        }
        scoring_tier = tier_map.get(force_tier) if force_tier else None

        changes: List[Dict[str, Any]] = []
        processed_count = 0

        # Re-score each email
        for email_id, cached_email in emails_to_process:
            if not cached_email:
                continue

            old_priority = cached_email.get("priority", "medium")
            old_confidence = cached_email.get("confidence", 0.5)

            try:
                # Get fresh emails from Gmail if connector available
                if self.gmail_connector:
                    try:
                        email_msg = await self.gmail_connector.get_message(email_id)
                        if email_msg:
                            result = await self.prioritizer.score_email(
                                email_msg, force_tier=scoring_tier
                            )

                            new_priority = result.priority.name.lower()
                            new_confidence = result.confidence

                            # Update cache (get, modify, set pattern for TTL cache)
                            updated_email = dict(cached_email)
                            updated_email.update(
                                {
                                    "priority": new_priority,
                                    "confidence": new_confidence,
                                    "reasoning": result.rationale,
                                    "tier_used": result.tier_used.value,
                                    "scores": {
                                        "sender": result.sender_score,
                                        "urgency": result.content_urgency_score,
                                        "context": result.context_relevance_score,
                                        "time_sensitivity": result.time_sensitivity_score,
                                    },
                                    "suggested_labels": result.suggested_labels,
                                    "auto_archive": result.auto_archive,
                                }
                            )
                            _email_cache.set(email_id, updated_email)

                            # Track if priority changed
                            if old_priority != new_priority:
                                changes.append(
                                    {
                                        "email_id": email_id,
                                        "old_priority": old_priority,
                                        "new_priority": new_priority,
                                        "old_confidence": old_confidence,
                                        "new_confidence": new_confidence,
                                        "tier_used": result.tier_used.value,
                                    }
                                )

                            processed_count += 1
                    except Exception as e:
                        logger.debug(f"Could not fetch email {email_id}: {e}")
                        processed_count += 1
                else:
                    # No Gmail connector - just mark as processed
                    processed_count += 1

            except Exception as e:
                logger.warning(f"Failed to reprioritize email {email_id}: {e}")

        return {
            "count": processed_count,
            "changes": changes,
            "tier_used": force_tier or "auto",
        }


def register_routes(app: web.Application) -> None:
    """Register inbox command center routes."""
    handler = InboxCommandHandler()

    # Main inbox endpoints
    app.router.add_get("/api/inbox/command", handler.handle_get_inbox)
    app.router.add_post("/api/inbox/actions", handler.handle_quick_action)
    app.router.add_post("/api/inbox/bulk-actions", handler.handle_bulk_action)
    app.router.add_get("/api/inbox/sender-profile", handler.handle_get_sender_profile)
    app.router.add_get("/api/inbox/daily-digest", handler.handle_get_daily_digest)
    app.router.add_post("/api/inbox/reprioritize", handler.handle_reprioritize)

    # API v1 endpoints
    app.router.add_get("/api/v1/inbox/command", handler.handle_get_inbox)
    app.router.add_post("/api/v1/inbox/actions", handler.handle_quick_action)
    app.router.add_post("/api/v1/inbox/bulk-actions", handler.handle_bulk_action)
    app.router.add_get("/api/v1/inbox/sender-profile", handler.handle_get_sender_profile)
    app.router.add_get("/api/v1/inbox/daily-digest", handler.handle_get_daily_digest)
    app.router.add_post("/api/v1/inbox/reprioritize", handler.handle_reprioritize)

    # Aliases for backward compatibility
    app.router.add_get("/api/email/daily-digest", handler.handle_get_daily_digest)
    app.router.add_get("/api/email/sender-profile", handler.handle_get_sender_profile)

    logger.info("Registered inbox command center routes")
