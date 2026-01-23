"""
Support Platform API Handlers.

Unified API for customer support and helpdesk platforms:
- Zendesk (tickets, users, organizations)
- Freshdesk (tickets, contacts, companies)
- Intercom (conversations, contacts, companies)
- Help Scout (conversations, customers, mailboxes)

Usage:
    GET    /api/v1/support/platforms              - List connected platforms
    POST   /api/v1/support/connect                - Connect a platform
    DELETE /api/v1/support/{platform}             - Disconnect platform

    GET    /api/v1/support/tickets                - List tickets (cross-platform)
    GET    /api/v1/support/{platform}/tickets     - Platform tickets
    POST   /api/v1/support/{platform}/tickets     - Create ticket
    PUT    /api/v1/support/{platform}/tickets/{id} - Update ticket

    GET    /api/v1/support/metrics                - Support metrics overview
    POST   /api/v1/support/triage                 - AI-powered ticket triage
    POST   /api/v1/support/auto-respond           - Generate response suggestions
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from aragora.server.handlers.secure import SecureHandler
from aragora.server.handlers.utils.decorators import has_permission

logger = logging.getLogger(__name__)


# Platform credentials storage
_platform_credentials: dict[str, dict[str, Any]] = {}
_platform_connectors: dict[str, Any] = {}


SUPPORTED_PLATFORMS = {
    "zendesk": {
        "name": "Zendesk",
        "description": "Enterprise helpdesk and customer service",
        "features": ["tickets", "users", "organizations", "macros", "triggers"],
    },
    "freshdesk": {
        "name": "Freshdesk",
        "description": "Cloud-based customer support software",
        "features": ["tickets", "contacts", "companies", "automations", "canned_responses"],
    },
    "intercom": {
        "name": "Intercom",
        "description": "Conversational customer engagement platform",
        "features": ["conversations", "contacts", "companies", "articles", "bots"],
    },
    "helpscout": {
        "name": "Help Scout",
        "description": "Customer service platform for growing teams",
        "features": ["conversations", "customers", "mailboxes", "saved_replies"],
    },
}


@dataclass
class UnifiedTicket:
    """Unified ticket/conversation representation across platforms."""

    id: str
    platform: str
    subject: str | None
    description: str | None
    status: str
    priority: str | None
    requester_email: str | None
    requester_name: str | None
    assignee_id: str | None
    assignee_name: str | None
    tags: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    first_response_at: datetime | None = None
    resolved_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "platform": self.platform,
            "subject": self.subject,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "requester_email": self.requester_email,
            "requester_name": self.requester_name,
            "assignee_id": self.assignee_id,
            "assignee_name": self.assignee_name,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "first_response_at": self.first_response_at.isoformat()
            if self.first_response_at
            else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


class SupportHandler(SecureHandler):
    """Handler for support platform API endpoints."""

    RESOURCE_TYPE = "support"

    ROUTES = [
        "/api/v1/support/platforms",
        "/api/v1/support/connect",
        "/api/v1/support/{platform}",
        "/api/v1/support/tickets",
        "/api/v1/support/{platform}/tickets",
        "/api/v1/support/{platform}/tickets/{ticket_id}",
        "/api/v1/support/{platform}/tickets/{ticket_id}/reply",
        "/api/v1/support/metrics",
        "/api/v1/support/triage",
        "/api/v1/support/auto-respond",
        "/api/v1/support/search",
    ]

    def _check_permission(self, request: Any, permission: str) -> dict[str, Any] | None:
        """Check if user has the required permission."""
        user = self.get_current_user(request)
        if user:
            user_role = user.role if hasattr(user, "role") else None
            if not has_permission(user_role, permission):
                return self._error_response(403, f"Permission denied: {permission} required")
        return None

    async def handle_request(self, request: Any) -> Any:
        """Route request to appropriate handler."""
        method = request.method
        path = str(request.path)

        # Parse path components
        platform = None
        ticket_id = None

        parts = path.replace("/api/v1/support/", "").split("/")
        if parts and parts[0] in SUPPORTED_PLATFORMS:
            platform = parts[0]
            if len(parts) > 2 and parts[1] == "tickets":
                ticket_id = parts[2]

        # Route to handlers
        if path.endswith("/platforms") and method == "GET":
            return await self._list_platforms(request)

        elif path.endswith("/connect") and method == "POST":
            if err := self._check_permission(request, "support:configure"):
                return err
            return await self._connect_platform(request)

        elif platform and path.endswith(f"/{platform}") and method == "DELETE":
            if err := self._check_permission(request, "support:configure"):
                return err
            return await self._disconnect_platform(request, platform)

        # Tickets
        elif path.endswith("/tickets") and not platform and method == "GET":
            if err := self._check_permission(request, "support:read"):
                return err
            return await self._list_all_tickets(request)

        elif platform and "tickets" in path:
            if path.endswith("/reply") and method == "POST":
                if err := self._check_permission(request, "support:write"):
                    return err
                return await self._reply_to_ticket(request, platform, ticket_id or "")
            elif method == "GET" and not ticket_id:
                if err := self._check_permission(request, "support:read"):
                    return err
                return await self._list_platform_tickets(request, platform)
            elif method == "POST" and not ticket_id:
                if err := self._check_permission(request, "support:write"):
                    return err
                return await self._create_ticket(request, platform)
            elif method == "PUT" and ticket_id:
                if err := self._check_permission(request, "support:write"):
                    return err
                return await self._update_ticket(request, platform, ticket_id)
            elif method == "GET" and ticket_id:
                if err := self._check_permission(request, "support:read"):
                    return err
                return await self._get_ticket(request, platform, ticket_id)

        # Metrics
        elif path.endswith("/metrics") and method == "GET":
            if err := self._check_permission(request, "support:read"):
                return err
            return await self._get_metrics(request)

        # Triage
        elif path.endswith("/triage") and method == "POST":
            if err := self._check_permission(request, "support:write"):
                return err
            return await self._triage_tickets(request)

        # Auto-respond
        elif path.endswith("/auto-respond") and method == "POST":
            if err := self._check_permission(request, "support:write"):
                return err
            return await self._generate_response(request)

        # Search
        elif path.endswith("/search") and method == "POST":
            if err := self._check_permission(request, "support:read"):
                return err
            return await self._search_tickets(request)

        return self._error_response(404, "Endpoint not found")

    async def _list_platforms(self, request: Any) -> dict[str, Any]:
        """List all supported support platforms and connection status."""
        platforms = []
        for platform_id, meta in SUPPORTED_PLATFORMS.items():
            connected = platform_id in _platform_credentials
            platforms.append(
                {
                    "id": platform_id,
                    "name": meta["name"],
                    "description": meta["description"],
                    "features": meta["features"],
                    "connected": connected,
                    "connected_at": _platform_credentials.get(platform_id, {}).get("connected_at"),
                }
            )

        return self._json_response(
            200,
            {
                "platforms": platforms,
                "connected_count": sum(1 for p in platforms if p["connected"]),
            },
        )

    async def _connect_platform(self, request: Any) -> dict[str, Any]:
        """Connect a support platform with credentials."""
        try:
            body = await self._get_json_body(request)
        except Exception as e:
            return self._error_response(400, f"Invalid JSON body: {e}")

        platform = body.get("platform")
        if not platform:
            return self._error_response(400, "Platform is required")

        if platform not in SUPPORTED_PLATFORMS:
            return self._error_response(400, f"Unsupported platform: {platform}")

        credentials = body.get("credentials", {})
        if not credentials:
            return self._error_response(400, "Credentials are required")

        required_fields = self._get_required_credentials(platform)
        missing = [f for f in required_fields if f not in credentials]
        if missing:
            return self._error_response(400, f"Missing required credentials: {', '.join(missing)}")

        _platform_credentials[platform] = {
            "credentials": credentials,
            "connected_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            connector = await self._get_connector(platform)
            if connector:
                _platform_connectors[platform] = connector
        except Exception as e:
            logger.warning(f"Could not initialize {platform} connector: {e}")

        logger.info(f"Connected support platform: {platform}")

        return self._json_response(
            200,
            {
                "message": f"Successfully connected to {SUPPORTED_PLATFORMS[platform]['name']}",
                "platform": platform,
                "connected_at": _platform_credentials[platform]["connected_at"],
            },
        )

    async def _disconnect_platform(self, request: Any, platform: str) -> dict[str, Any]:
        """Disconnect a support platform."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        if platform in _platform_connectors:
            connector = _platform_connectors[platform]
            if hasattr(connector, "close"):
                await connector.close()
            del _platform_connectors[platform]

        del _platform_credentials[platform]

        logger.info(f"Disconnected support platform: {platform}")

        return self._json_response(
            200,
            {
                "message": f"Disconnected from {SUPPORTED_PLATFORMS[platform]['name']}",
                "platform": platform,
            },
        )

    # Ticket operations

    async def _list_all_tickets(self, request: Any) -> dict[str, Any]:
        """List tickets from all connected platforms."""
        status = request.query.get("status")
        priority = request.query.get("priority")
        limit = int(request.query.get("limit", 100))

        all_tickets: list[dict[str, Any]] = []

        tasks = []
        for platform in _platform_credentials.keys():
            tasks.append(self._fetch_platform_tickets(platform, status, priority, limit))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for platform, result in zip(_platform_credentials.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching tickets from {platform}: {result}")
                continue
            all_tickets.extend(result)

        # Sort by created_at descending
        all_tickets.sort(key=lambda t: t.get("created_at") or "", reverse=True)

        return self._json_response(
            200,
            {
                "tickets": all_tickets[:limit],
                "total": len(all_tickets),
                "platforms_queried": list(_platform_credentials.keys()),
            },
        )

    async def _fetch_platform_tickets(
        self,
        platform: str,
        status: str | None = None,
        priority: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch tickets from a specific platform."""
        connector = await self._get_connector(platform)
        if not connector:
            return []

        try:
            if platform == "zendesk":
                tickets = await connector.get_tickets(status=status, limit=limit)
                return [self._normalize_zendesk_ticket(t) for t in tickets]

            elif platform == "freshdesk":
                tickets = await connector.get_tickets(status=status, limit=limit)
                return [self._normalize_freshdesk_ticket(t) for t in tickets]

            elif platform == "intercom":
                conversations = await connector.get_conversations(state=status, limit=limit)
                return [self._normalize_intercom_conversation(c) for c in conversations]

            elif platform == "helpscout":
                conversations = await connector.get_conversations(status=status, limit=limit)
                return [self._normalize_helpscout_conversation(c) for c in conversations]

        except Exception as e:
            logger.error(f"Error fetching {platform} tickets: {e}")

        return []

    async def _list_platform_tickets(self, request: Any, platform: str) -> dict[str, Any]:
        """List tickets from a specific platform."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        status = request.query.get("status")
        priority = request.query.get("priority")
        limit = int(request.query.get("limit", 100))

        tickets = await self._fetch_platform_tickets(platform, status, priority, limit)

        return self._json_response(
            200,
            {
                "tickets": tickets,
                "total": len(tickets),
                "platform": platform,
            },
        )

    async def _get_ticket(self, request: Any, platform: str, ticket_id: str) -> dict[str, Any]:
        """Get a specific ticket with its conversation history."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        try:
            if platform == "zendesk":
                ticket = await connector.get_ticket(int(ticket_id))
                comments = await connector.get_ticket_comments(int(ticket_id))
                return self._json_response(
                    200,
                    {
                        **self._normalize_zendesk_ticket(ticket),
                        "comments": [self._normalize_zendesk_comment(c) for c in comments],
                    },
                )

            elif platform == "freshdesk":
                ticket = await connector.get_ticket(int(ticket_id))
                conversations = await connector.get_ticket_conversations(int(ticket_id))
                return self._json_response(
                    200,
                    {
                        **self._normalize_freshdesk_ticket(ticket),
                        "conversations": conversations,
                    },
                )

            elif platform == "intercom":
                conversation = await connector.get_conversation(ticket_id)
                return self._json_response(200, self._normalize_intercom_conversation(conversation))

            elif platform == "helpscout":
                conversation = await connector.get_conversation(int(ticket_id))
                threads = await connector.get_conversation_threads(int(ticket_id))
                return self._json_response(
                    200,
                    {
                        **self._normalize_helpscout_conversation(conversation),
                        "threads": threads,
                    },
                )

        except Exception as e:
            return self._error_response(404, f"Ticket not found: {e}")

        return self._error_response(400, "Unsupported platform")

    async def _create_ticket(self, request: Any, platform: str) -> dict[str, Any]:
        """Create a new ticket."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        try:
            body = await self._get_json_body(request)
        except Exception as e:
            return self._error_response(400, f"Invalid JSON body: {e}")

        subject = body.get("subject")
        description = body.get("description")
        requester_email = body.get("requester_email")

        if not description:
            return self._error_response(400, "Description is required")

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        try:
            if platform == "zendesk":
                ticket = await connector.create_ticket(
                    subject=subject or "Support Request",
                    description=description,
                    requester_email=requester_email,
                    priority=body.get("priority"),
                    tags=body.get("tags", []),
                )
                return self._json_response(201, self._normalize_zendesk_ticket(ticket))

            elif platform == "freshdesk":
                ticket = await connector.create_ticket(
                    subject=subject or "Support Request",
                    description=description,
                    email=requester_email,
                    priority=self._map_priority_to_freshdesk(body.get("priority")),
                    tags=body.get("tags", []),
                )
                return self._json_response(201, self._normalize_freshdesk_ticket(ticket))

            elif platform == "intercom":
                conversation = await connector.create_conversation(
                    user_id=body.get("user_id"),
                    body=description,
                )
                return self._json_response(201, self._normalize_intercom_conversation(conversation))

            elif platform == "helpscout":
                conversation = await connector.create_conversation(
                    mailbox_id=body.get("mailbox_id"),
                    customer_email=requester_email,
                    subject=subject or "Support Request",
                    text=description,
                )
                return self._json_response(
                    201, self._normalize_helpscout_conversation(conversation)
                )

        except Exception as e:
            return self._error_response(500, f"Failed to create ticket: {e}")

        return self._error_response(400, "Unsupported platform")

    async def _update_ticket(
        self,
        request: Any,
        platform: str,
        ticket_id: str,
    ) -> dict[str, Any]:
        """Update a ticket."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        try:
            body = await self._get_json_body(request)
        except Exception as e:
            return self._error_response(400, f"Invalid JSON body: {e}")

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        try:
            if platform == "zendesk":
                updates = {}
                if "status" in body:
                    updates["status"] = body["status"]
                if "priority" in body:
                    updates["priority"] = body["priority"]
                if "assignee_id" in body:
                    updates["assignee_id"] = body["assignee_id"]
                if "tags" in body:
                    updates["tags"] = body["tags"]

                ticket = await connector.update_ticket(int(ticket_id), **updates)
                return self._json_response(200, self._normalize_zendesk_ticket(ticket))

            elif platform == "freshdesk":
                updates = {}
                if "status" in body:
                    updates["status"] = self._map_status_to_freshdesk(body["status"])
                if "priority" in body:
                    updates["priority"] = self._map_priority_to_freshdesk(body["priority"])
                if "assignee_id" in body:
                    updates["responder_id"] = body["assignee_id"]

                ticket = await connector.update_ticket(int(ticket_id), **updates)
                return self._json_response(200, self._normalize_freshdesk_ticket(ticket))

            elif platform == "intercom":
                if "status" in body:
                    if body["status"] in ["closed", "resolved"]:
                        await connector.close_conversation(ticket_id)
                    elif body["status"] == "open":
                        await connector.open_conversation(ticket_id)
                conversation = await connector.get_conversation(ticket_id)
                return self._json_response(200, self._normalize_intercom_conversation(conversation))

            elif platform == "helpscout":
                updates = {}
                if "status" in body:
                    updates["status"] = body["status"]
                if "assignee_id" in body:
                    updates["assignee"] = body["assignee_id"]

                await connector.update_conversation(int(ticket_id), **updates)
                conversation = await connector.get_conversation(int(ticket_id))
                return self._json_response(
                    200, self._normalize_helpscout_conversation(conversation)
                )

        except Exception as e:
            return self._error_response(500, f"Failed to update ticket: {e}")

        return self._error_response(400, "Unsupported platform")

    async def _reply_to_ticket(
        self,
        request: Any,
        platform: str,
        ticket_id: str,
    ) -> dict[str, Any]:
        """Reply to a ticket."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        try:
            body = await self._get_json_body(request)
        except Exception as e:
            return self._error_response(400, f"Invalid JSON body: {e}")

        message = body.get("message")
        if not message:
            return self._error_response(400, "Message is required")

        public = body.get("public", True)

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        try:
            if platform == "zendesk":
                await connector.add_ticket_comment(
                    int(ticket_id),
                    body=message,
                    public=public,
                )
                return self._json_response(200, {"message": "Reply added successfully"})

            elif platform == "freshdesk":
                await connector.reply_to_ticket(
                    int(ticket_id),
                    body=message,
                )
                return self._json_response(200, {"message": "Reply added successfully"})

            elif platform == "intercom":
                await connector.reply_to_conversation(
                    ticket_id,
                    body=message,
                    message_type="comment" if not public else "note",
                )
                return self._json_response(200, {"message": "Reply added successfully"})

            elif platform == "helpscout":
                await connector.add_reply(
                    int(ticket_id),
                    text=message,
                )
                return self._json_response(200, {"message": "Reply added successfully"})

        except Exception as e:
            return self._error_response(500, f"Failed to add reply: {e}")

        return self._error_response(400, "Unsupported platform")

    # Metrics

    async def _get_metrics(self, request: Any) -> dict[str, Any]:
        """Get support metrics overview."""
        days = int(request.query.get("days", 7))

        metrics: dict[str, Any] = {
            "period_days": days,
            "platforms": {},
            "totals": {
                "total_tickets": 0,
                "open_tickets": 0,
                "resolved_tickets": 0,
                "avg_response_time_hours": 0,
                "avg_resolution_time_hours": 0,
            },
        }

        for platform in _platform_credentials.keys():
            try:
                platform_metrics = await self._fetch_platform_metrics(platform, days)
                metrics["platforms"][platform] = platform_metrics

                metrics["totals"]["total_tickets"] += platform_metrics.get("total_tickets", 0)
                metrics["totals"]["open_tickets"] += platform_metrics.get("open_tickets", 0)
                metrics["totals"]["resolved_tickets"] += platform_metrics.get("resolved_tickets", 0)

            except Exception as e:
                logger.error(f"Error fetching {platform} metrics: {e}")
                metrics["platforms"][platform] = {"error": str(e)}

        return self._json_response(200, metrics)

    async def _fetch_platform_metrics(self, platform: str, days: int) -> dict[str, Any]:
        """Fetch metrics from a specific platform."""
        connector = await self._get_connector(platform)
        if not connector:
            return {"error": "Connector not available"}

        # Fetch recent tickets to calculate metrics
        tickets = await self._fetch_platform_tickets(platform, limit=1000)

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent_tickets = [
            t
            for t in tickets
            if t.get("created_at")
            and datetime.fromisoformat(t["created_at"].replace("Z", "+00:00")) >= cutoff
        ]

        open_count = sum(1 for t in recent_tickets if t.get("status") in ["open", "new", "pending"])
        resolved_count = sum(
            1 for t in recent_tickets if t.get("status") in ["solved", "closed", "resolved"]
        )

        return {
            "total_tickets": len(recent_tickets),
            "open_tickets": open_count,
            "resolved_tickets": resolved_count,
            "pending_tickets": len(recent_tickets) - open_count - resolved_count,
            "tickets_by_priority": self._count_by_priority(recent_tickets),
        }

    # Triage

    async def _triage_tickets(self, request: Any) -> dict[str, Any]:
        """AI-powered ticket triage using multi-agent analysis."""
        try:
            body = await self._get_json_body(request)
        except Exception as e:
            return self._error_response(400, f"Invalid JSON body: {e}")

        ticket_ids = body.get("ticket_ids", [])
        platform = body.get("platform")

        if not ticket_ids and not platform:
            # Triage all open tickets from all platforms
            all_tickets = await self._list_all_tickets(request)
            tickets_data = all_tickets.get("body", {}).get("tickets", [])
            tickets_to_triage = [t for t in tickets_data if t.get("status") in ["open", "new"]]
        else:
            tickets_to_triage = []
            for tid in ticket_ids:
                connector = await self._get_connector(platform)
                if connector:
                    try:
                        if platform == "zendesk":
                            ticket = await connector.get_ticket(int(tid))
                            tickets_to_triage.append(self._normalize_zendesk_ticket(ticket))
                    except Exception:
                        pass

        # Perform triage analysis
        triage_results = []
        for ticket in tickets_to_triage[:50]:  # Limit to 50 tickets
            triage_results.append(
                {
                    "ticket_id": ticket.get("id"),
                    "platform": ticket.get("platform"),
                    "subject": ticket.get("subject"),
                    "suggested_priority": self._suggest_priority(ticket),
                    "suggested_category": self._suggest_category(ticket),
                    "sentiment": self._analyze_sentiment(ticket),
                    "urgency_score": self._calculate_urgency(ticket),
                    "suggested_response_template": self._suggest_response_template(ticket),
                }
            )

        # Sort by urgency
        triage_results.sort(key=lambda t: t["urgency_score"], reverse=True)

        return self._json_response(
            200,
            {
                "triage_id": str(uuid4()),
                "tickets_analyzed": len(triage_results),
                "results": triage_results,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    # Auto-respond

    async def _generate_response(self, request: Any) -> dict[str, Any]:
        """Generate AI response suggestions for a ticket."""
        try:
            body = await self._get_json_body(request)
        except Exception as e:
            return self._error_response(400, f"Invalid JSON body: {e}")

        ticket_id = body.get("ticket_id")
        platform = body.get("platform")
        body.get("context", "")

        if not ticket_id or not platform:
            return self._error_response(400, "ticket_id and platform are required")

        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        # Fetch ticket details
        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        try:
            ticket_data = {}
            if platform == "zendesk":
                ticket = await connector.get_ticket(int(ticket_id))
                ticket_data = self._normalize_zendesk_ticket(ticket)
                comments = await connector.get_ticket_comments(int(ticket_id))
                ticket_data["history"] = [self._normalize_zendesk_comment(c) for c in comments]
            # Similar for other platforms...

        except Exception as e:
            return self._error_response(404, f"Ticket not found: {e}")

        # Generate response suggestions
        # In production, this would use the multi-agent debate system
        suggestions = [
            {
                "type": "acknowledgment",
                "tone": "professional",
                "message": f"Thank you for contacting us regarding '{ticket_data.get('subject', 'your inquiry')}'. We understand your concern and are working to resolve this as quickly as possible.",
            },
            {
                "type": "solution",
                "tone": "helpful",
                "message": "Based on your description, here are some steps that might help resolve your issue...",
            },
            {
                "type": "follow_up",
                "tone": "friendly",
                "message": "I wanted to follow up on your recent inquiry. Have you had a chance to try the suggested solution? Please let us know if you need any further assistance.",
            },
        ]

        return self._json_response(
            200,
            {
                "ticket_id": ticket_id,
                "platform": platform,
                "suggestions": suggestions,
                "ticket_context": {
                    "subject": ticket_data.get("subject"),
                    "status": ticket_data.get("status"),
                    "priority": ticket_data.get("priority"),
                },
            },
        )

    # Search

    async def _search_tickets(self, request: Any) -> dict[str, Any]:
        """Search tickets across platforms."""
        try:
            body = await self._get_json_body(request)
        except Exception as e:
            return self._error_response(400, f"Invalid JSON body: {e}")

        query = body.get("query", "")
        platforms = body.get("platforms", list(_platform_credentials.keys()))
        limit = body.get("limit", 50)

        results: list[dict[str, Any]] = []

        for platform in platforms:
            if platform not in _platform_credentials:
                continue

            connector = await self._get_connector(platform)
            if not connector:
                continue

            try:
                if platform == "zendesk":
                    tickets = await connector.search_tickets(query, limit=limit)
                    results.extend([self._normalize_zendesk_ticket(t) for t in tickets])

                elif platform == "freshdesk":
                    tickets = await connector.search_tickets(query, limit=limit)
                    results.extend([self._normalize_freshdesk_ticket(t) for t in tickets])

            except Exception as e:
                logger.error(f"Error searching {platform}: {e}")

        return self._json_response(
            200,
            {
                "query": query,
                "results": results[:limit],
                "total": len(results),
            },
        )

    # Helper methods

    def _get_required_credentials(self, platform: str) -> list[str]:
        """Get required credential fields for a platform."""
        requirements = {
            "zendesk": ["subdomain", "email", "api_token"],
            "freshdesk": ["domain", "api_key"],
            "intercom": ["access_token"],
            "helpscout": ["app_id", "app_secret"],
        }
        return requirements.get(platform, [])

    async def _get_connector(self, platform: str) -> Any | None:
        """Get or create a connector for a platform."""
        if platform in _platform_connectors:
            return _platform_connectors[platform]

        if platform not in _platform_credentials:
            return None

        creds = _platform_credentials[platform]["credentials"]

        try:
            if platform == "zendesk":
                from aragora.connectors.support.zendesk import (
                    ZendeskConnector,
                    ZendeskCredentials,
                )

                connector = ZendeskConnector(ZendeskCredentials(**creds))

            elif platform == "freshdesk":
                from aragora.connectors.support.freshdesk import (
                    FreshdeskConnector,
                    FreshdeskCredentials,
                )

                connector = FreshdeskConnector(FreshdeskCredentials(**creds))

            elif platform == "intercom":
                from aragora.connectors.support.intercom import (
                    IntercomConnector,
                    IntercomCredentials,
                )

                connector = IntercomConnector(IntercomCredentials(**creds))

            elif platform == "helpscout":
                from aragora.connectors.support.helpscout import (
                    HelpScoutConnector,
                    HelpScoutCredentials,
                )

                connector = HelpScoutConnector(HelpScoutCredentials(**creds))

            else:
                return None

            _platform_connectors[platform] = connector
            return connector

        except Exception as e:
            logger.error(f"Failed to create {platform} connector: {e}")
            return None

    def _normalize_zendesk_ticket(self, ticket: Any) -> dict[str, Any]:
        """Normalize Zendesk ticket."""
        return {
            "id": str(ticket.id),
            "platform": "zendesk",
            "subject": ticket.subject,
            "description": ticket.description,
            "status": ticket.status.value if hasattr(ticket.status, "value") else ticket.status,
            "priority": ticket.priority.value
            if hasattr(ticket.priority, "value")
            else ticket.priority,
            "requester_email": ticket.requester_email,
            "requester_name": ticket.requester_name,
            "assignee_id": str(ticket.assignee_id) if ticket.assignee_id else None,
            "assignee_name": ticket.assignee_name if hasattr(ticket, "assignee_name") else None,
            "tags": ticket.tags,
            "created_at": ticket.created_at.isoformat() if ticket.created_at else None,
            "updated_at": ticket.updated_at.isoformat() if ticket.updated_at else None,
        }

    def _normalize_zendesk_comment(self, comment: Any) -> dict[str, Any]:
        """Normalize Zendesk comment."""
        return {
            "id": str(comment.id),
            "body": comment.body,
            "public": comment.public,
            "author_id": str(comment.author_id) if comment.author_id else None,
            "created_at": comment.created_at.isoformat() if comment.created_at else None,
        }

    def _normalize_freshdesk_ticket(self, ticket: Any) -> dict[str, Any]:
        """Normalize Freshdesk ticket."""
        return {
            "id": str(ticket.id),
            "platform": "freshdesk",
            "subject": ticket.subject,
            "description": ticket.description,
            "status": self._map_freshdesk_status(ticket.status),
            "priority": self._map_freshdesk_priority(ticket.priority),
            "requester_email": ticket.email,
            "requester_name": ticket.name,
            "assignee_id": str(ticket.responder_id) if ticket.responder_id else None,
            "tags": ticket.tags,
            "created_at": ticket.created_at.isoformat() if ticket.created_at else None,
            "updated_at": ticket.updated_at.isoformat() if ticket.updated_at else None,
        }

    def _normalize_intercom_conversation(self, conv: Any) -> dict[str, Any]:
        """Normalize Intercom conversation."""
        return {
            "id": conv.id,
            "platform": "intercom",
            "subject": conv.title if hasattr(conv, "title") else None,
            "description": conv.source.body
            if hasattr(conv, "source") and hasattr(conv.source, "body")
            else None,
            "status": conv.state,
            "priority": conv.priority.value
            if hasattr(conv, "priority") and hasattr(conv.priority, "value")
            else None,
            "requester_email": conv.contacts[0].email
            if hasattr(conv, "contacts") and conv.contacts
            else None,
            "created_at": conv.created_at.isoformat()
            if hasattr(conv, "created_at") and conv.created_at
            else None,
            "updated_at": conv.updated_at.isoformat()
            if hasattr(conv, "updated_at") and conv.updated_at
            else None,
        }

    def _normalize_helpscout_conversation(self, conv: Any) -> dict[str, Any]:
        """Normalize Help Scout conversation."""
        return {
            "id": str(conv.id),
            "platform": "helpscout",
            "subject": conv.subject,
            "description": conv.preview if hasattr(conv, "preview") else None,
            "status": conv.status.value if hasattr(conv.status, "value") else conv.status,
            "priority": None,  # Help Scout doesn't have priority on conversations
            "requester_email": conv.customer.email
            if hasattr(conv, "customer") and conv.customer
            else None,
            "requester_name": f"{conv.customer.first_name} {conv.customer.last_name}".strip()
            if hasattr(conv, "customer") and conv.customer
            else None,
            "assignee_id": str(conv.assignee.id)
            if hasattr(conv, "assignee") and conv.assignee
            else None,
            "tags": conv.tags if hasattr(conv, "tags") else [],
            "created_at": conv.created_at.isoformat()
            if hasattr(conv, "created_at") and conv.created_at
            else None,
        }

    def _map_freshdesk_status(self, status: int) -> str:
        """Map Freshdesk status code to string."""
        status_map = {2: "open", 3: "pending", 4: "resolved", 5: "closed"}
        return status_map.get(status, "unknown")

    def _map_freshdesk_priority(self, priority: int) -> str:
        """Map Freshdesk priority code to string."""
        priority_map = {1: "low", 2: "medium", 3: "high", 4: "urgent"}
        return priority_map.get(priority, "medium")

    def _map_status_to_freshdesk(self, status: str) -> int:
        """Map status string to Freshdesk code."""
        status_map = {"open": 2, "pending": 3, "resolved": 4, "closed": 5}
        return status_map.get(status.lower(), 2)

    def _map_priority_to_freshdesk(self, priority: str | None) -> int:
        """Map priority string to Freshdesk code."""
        if not priority:
            return 2
        priority_map = {"low": 1, "medium": 2, "high": 3, "urgent": 4}
        return priority_map.get(priority.lower(), 2)

    def _count_by_priority(self, tickets: list[dict[str, Any]]) -> dict[str, int]:
        """Count tickets by priority."""
        counts: dict[str, int] = {}
        for ticket in tickets:
            priority = ticket.get("priority") or "unset"
            counts[priority] = counts.get(priority, 0) + 1
        return counts

    def _suggest_priority(self, ticket: dict[str, Any]) -> str:
        """Suggest priority based on ticket content."""
        subject = (ticket.get("subject") or "").lower()
        description = (ticket.get("description") or "").lower()

        urgent_keywords = ["urgent", "emergency", "down", "broken", "critical", "asap"]
        high_keywords = ["important", "error", "bug", "issue", "problem"]

        text = f"{subject} {description}"

        if any(kw in text for kw in urgent_keywords):
            return "urgent"
        elif any(kw in text for kw in high_keywords):
            return "high"
        else:
            return "medium"

    def _suggest_category(self, ticket: dict[str, Any]) -> str:
        """Suggest category based on ticket content."""
        subject = (ticket.get("subject") or "").lower()
        description = (ticket.get("description") or "").lower()
        text = f"{subject} {description}"

        categories = {
            "billing": ["billing", "invoice", "payment", "charge", "refund", "subscription"],
            "technical": ["error", "bug", "crash", "not working", "broken", "issue"],
            "account": ["password", "login", "account", "access", "permission"],
            "feature_request": ["feature", "request", "suggestion", "would be nice", "please add"],
            "general": [],
        }

        for category, keywords in categories.items():
            if any(kw in text for kw in keywords):
                return category

        return "general"

    def _analyze_sentiment(self, ticket: dict[str, Any]) -> str:
        """Analyze sentiment of ticket."""
        description = (ticket.get("description") or "").lower()

        negative_words = ["frustrated", "angry", "disappointed", "terrible", "awful", "worst"]
        positive_words = ["thanks", "appreciate", "great", "excellent", "helpful"]

        neg_count = sum(1 for w in negative_words if w in description)
        pos_count = sum(1 for w in positive_words if w in description)

        if neg_count > pos_count:
            return "negative"
        elif pos_count > neg_count:
            return "positive"
        return "neutral"

    def _calculate_urgency(self, ticket: dict[str, Any]) -> float:
        """Calculate urgency score (0-1)."""
        score = 0.5

        priority = ticket.get("priority", "").lower()
        if priority == "urgent":
            score += 0.3
        elif priority == "high":
            score += 0.2
        elif priority == "low":
            score -= 0.2

        sentiment = self._analyze_sentiment(ticket)
        if sentiment == "negative":
            score += 0.1

        return min(max(score, 0), 1)

    def _suggest_response_template(self, ticket: dict[str, Any]) -> str:
        """Suggest a response template."""
        category = self._suggest_category(ticket)

        templates = {
            "billing": "billing_inquiry",
            "technical": "technical_support",
            "account": "account_assistance",
            "feature_request": "feature_request_acknowledgment",
            "general": "general_response",
        }

        return templates.get(category, "general_response")

    async def _get_json_body(self, request: Any) -> dict[str, Any]:
        """Parse JSON body from request."""
        body = await request.json()
        return body if isinstance(body, dict) else {}

    def _json_response(self, status: int, data: Any) -> dict[str, Any]:
        """Create a JSON response."""
        return {
            "status_code": status,
            "headers": {"Content-Type": "application/json"},
            "body": data,
        }

    def _error_response(self, status: int, message: str) -> dict[str, Any]:
        """Create an error response."""
        return self._json_response(status, {"error": message})


__all__ = ["SupportHandler"]
