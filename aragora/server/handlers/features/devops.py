"""
DevOps Incident Management API Handler.

Provides REST APIs for incident management via PagerDuty:
- Create and manage incidents
- Add investigation notes
- Acknowledge and resolve incidents
- Query on-call schedules
- Service health status
- Webhook handling for incident updates

Endpoints:
- POST /api/v1/incidents                    - Create incident
- GET  /api/v1/incidents                    - List incidents
- GET  /api/v1/incidents/{id}               - Get incident details
- POST /api/v1/incidents/{id}/acknowledge   - Acknowledge incident
- POST /api/v1/incidents/{id}/resolve       - Resolve incident
- POST /api/v1/incidents/{id}/reassign      - Reassign incident
- POST /api/v1/incidents/{id}/notes         - Add note
- GET  /api/v1/incidents/{id}/notes         - List notes
- GET  /api/v1/oncall                       - Get current on-call
- GET  /api/v1/oncall/services/{id}         - Get on-call for service
- GET  /api/v1/services                     - List services
- GET  /api/v1/services/{id}                - Get service details
- POST /api/v1/webhooks/pagerduty           - PagerDuty webhook handler
- GET  /api/v1/devops/status                - Connection status
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
    success_response,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Connector Instance Management
# =============================================================================

_connector_instances: Dict[str, Any] = {}  # tenant_id -> PagerDutyConnector
_active_contexts: Dict[str, Any] = {}  # tenant_id -> context manager


async def get_pagerduty_connector(tenant_id: str):
    """Get or create PagerDuty connector for tenant."""
    if tenant_id not in _connector_instances:
        try:
            import os

            from aragora.connectors.devops.pagerduty import (
                PagerDutyConnector,
                PagerDutyCredentials,
            )

            api_key = os.getenv("PAGERDUTY_API_KEY")
            email = os.getenv("PAGERDUTY_EMAIL")
            webhook_secret = os.getenv("PAGERDUTY_WEBHOOK_SECRET")

            if not api_key or not email:
                return None

            credentials = PagerDutyCredentials(
                api_key=api_key,
                email=email,
                webhook_secret=webhook_secret,
            )

            connector = PagerDutyConnector(credentials)
            # Enter context to initialize client
            await connector.__aenter__()
            _connector_instances[tenant_id] = connector
            _active_contexts[tenant_id] = connector

        except ImportError:
            return None
        except Exception as e:
            logger.error(f"Failed to initialize PagerDuty connector: {e}")
            return None

    return _connector_instances.get(tenant_id)


# =============================================================================
# Handler Class
# =============================================================================


class DevOpsHandler(BaseHandler):
    """Handler for DevOps incident management API endpoints."""

    ROUTES = [
        "/api/v1/incidents",
        "/api/v1/incidents/{incident_id}",
        "/api/v1/incidents/{incident_id}/acknowledge",
        "/api/v1/incidents/{incident_id}/resolve",
        "/api/v1/incidents/{incident_id}/reassign",
        "/api/v1/incidents/{incident_id}/notes",
        "/api/v1/incidents/{incident_id}/merge",
        "/api/v1/oncall",
        "/api/v1/oncall/services/{service_id}",
        "/api/v1/services",
        "/api/v1/services/{service_id}",
        "/api/v1/webhooks/pagerduty",
        "/api/v1/devops/status",
    ]

    def __init__(self, server_context: Optional[Dict[str, Any]] = None):
        """Initialize handler with optional server context."""
        super().__init__(server_context or {})  # type: ignore[arg-type]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return (
            path.startswith("/api/v1/incidents")
            or path.startswith("/api/v1/oncall")
            or path.startswith("/api/v1/services")
            or path.startswith("/api/v1/webhooks/pagerduty")
            or path.startswith("/api/v1/devops")
        )

    async def handle(  # type: ignore[override]
        self, request: Any, path: str, method: str
    ) -> HandlerResult:
        """Route requests to appropriate handler methods."""
        try:
            tenant_id = self._get_tenant_id(request)

            # Status check
            if path == "/api/v1/devops/status" and method == "GET":
                return await self._handle_status(request, tenant_id)

            # Webhook
            if path == "/api/v1/webhooks/pagerduty" and method == "POST":
                return await self._handle_pagerduty_webhook(request, tenant_id)

            # List/create incidents
            if path == "/api/v1/incidents":
                if method == "GET":
                    return await self._handle_list_incidents(request, tenant_id)
                elif method == "POST":
                    return await self._handle_create_incident(request, tenant_id)

            # On-call
            if path == "/api/v1/oncall" and method == "GET":
                return await self._handle_get_oncall(request, tenant_id)

            # Services list
            if path == "/api/v1/services" and method == "GET":
                return await self._handle_list_services(request, tenant_id)

            # Incident-specific paths
            if path.startswith("/api/v1/incidents/"):
                parts = path.split("/")
                if len(parts) >= 4:
                    incident_id = parts[3]

                    # GET /incidents/{id}
                    if len(parts) == 4 and method == "GET":
                        return await self._handle_get_incident(request, tenant_id, incident_id)

                    # Actions on incident
                    if len(parts) == 5:
                        action = parts[4]
                        if action == "acknowledge" and method == "POST":
                            return await self._handle_acknowledge_incident(
                                request, tenant_id, incident_id
                            )
                        elif action == "resolve" and method == "POST":
                            return await self._handle_resolve_incident(
                                request, tenant_id, incident_id
                            )
                        elif action == "reassign" and method == "POST":
                            return await self._handle_reassign_incident(
                                request, tenant_id, incident_id
                            )
                        elif action == "merge" and method == "POST":
                            return await self._handle_merge_incidents(
                                request, tenant_id, incident_id
                            )
                        elif action == "notes":
                            if method == "GET":
                                return await self._handle_list_notes(
                                    request, tenant_id, incident_id
                                )
                            elif method == "POST":
                                return await self._handle_add_note(request, tenant_id, incident_id)

            # On-call for specific service
            if path.startswith("/api/v1/oncall/services/"):
                parts = path.split("/")
                if len(parts) == 5 and method == "GET":
                    service_id = parts[4]
                    return await self._handle_get_oncall_for_service(request, tenant_id, service_id)

            # Service details
            if path.startswith("/api/v1/services/"):
                parts = path.split("/")
                if len(parts) == 4 and method == "GET":
                    service_id = parts[3]
                    return await self._handle_get_service(request, tenant_id, service_id)

            return error_response("Not found", 404)

        except Exception as e:
            logger.exception(f"Error in devops handler: {e}")
            return error_response(f"Internal error: {str(e)}", 500)

    def _get_tenant_id(self, request: Any) -> str:
        """Extract tenant ID from request context."""
        return getattr(request, "tenant_id", "default")

    # =========================================================================
    # Status
    # =========================================================================

    async def _handle_status(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get PagerDuty connection status."""
        import os

        api_key = os.getenv("PAGERDUTY_API_KEY")
        email = os.getenv("PAGERDUTY_EMAIL")

        return success_response(
            {
                "configured": bool(api_key and email),
                "api_key_set": bool(api_key),
                "email_set": bool(email),
                "webhook_secret_set": bool(os.getenv("PAGERDUTY_WEBHOOK_SECRET")),
            }
        )

    # =========================================================================
    # Incidents
    # =========================================================================

    async def _handle_list_incidents(self, request: Any, tenant_id: str) -> HandlerResult:
        """List incidents with filtering.

        Query params:
        - status: Filter by status (triggered, acknowledged, resolved)
        - service_ids: Comma-separated service IDs
        - urgency: Filter by urgency (high, low)
        - limit: Max results (default 25)
        - offset: Pagination offset
        """
        connector = await get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        params = self._get_query_params(request)
        statuses = params.get("status", "").split(",") if params.get("status") else None
        service_ids = (
            params.get("service_ids", "").split(",") if params.get("service_ids") else None
        )
        urgencies = params.get("urgency", "").split(",") if params.get("urgency") else None
        limit = int(params.get("limit", 25))
        offset = int(params.get("offset", 0))

        try:
            incidents, has_more = await connector.list_incidents(
                statuses=statuses,
                service_ids=service_ids,
                urgencies=urgencies,
                limit=limit,
                offset=offset,
            )

            return success_response(
                {
                    "incidents": [
                        {
                            "id": inc.id,
                            "title": inc.title,
                            "status": inc.status.value,
                            "urgency": inc.urgency.value,
                            "service_id": inc.service_id,
                            "service_name": inc.service_name,
                            "incident_number": inc.incident_number,
                            "created_at": (inc.created_at.isoformat() if inc.created_at else None),
                            "html_url": inc.html_url,
                        }
                        for inc in incidents
                    ],
                    "count": len(incidents),
                    "has_more": has_more,
                }
            )

        except Exception as e:
            logger.error(f"Failed to list incidents: {e}")
            return error_response(f"Failed to list incidents: {e}", 500)

    async def _handle_create_incident(self, request: Any, tenant_id: str) -> HandlerResult:
        """Create a new incident.

        Request body:
        {
            "title": "Critical: Database connection failure",
            "service_id": "PSERVICE123",
            "urgency": "high",  // "high" or "low"
            "body": "Detailed description of the issue",
            "escalation_policy_id": "PESCPOL123",  // Optional
            "priority_id": "PPRI123"  // Optional
        }
        """
        connector = await get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        body = await self._get_json_body(request)

        # Validate required fields
        if not body.get("title"):
            return error_response("title is required", 400)
        if not body.get("service_id"):
            return error_response("service_id is required", 400)

        try:
            from aragora.connectors.devops.pagerduty import (
                IncidentCreateRequest,
                IncidentUrgency,
            )

            urgency = IncidentUrgency(body.get("urgency", "high"))

            create_request = IncidentCreateRequest(
                title=body["title"],
                service_id=body["service_id"],
                urgency=urgency,
                description=body.get("body"),
                escalation_policy_id=body.get("escalation_policy_id"),
                priority_id=body.get("priority_id"),
            )

            incident = await connector.create_incident(create_request)

            logger.info(f"[DevOps] Created incident {incident.id} for tenant {tenant_id}")

            return json_response(
                {
                    "incident": {
                        "id": incident.id,
                        "title": incident.title,
                        "status": incident.status.value,
                        "urgency": incident.urgency.value,
                        "service_id": incident.service_id,
                        "incident_number": incident.incident_number,
                        "html_url": incident.html_url,
                    },
                    "message": "Incident created successfully",
                },
                status=201,
            )

        except Exception as e:
            logger.error(f"Failed to create incident: {e}")
            return error_response(f"Failed to create incident: {e}", 500)

    async def _handle_get_incident(
        self, request: Any, tenant_id: str, incident_id: str
    ) -> HandlerResult:
        """Get incident details."""
        connector = await get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        try:
            incident = await connector.get_incident(incident_id)

            return success_response(
                {
                    "incident": {
                        "id": incident.id,
                        "title": incident.title,
                        "status": incident.status.value,
                        "urgency": incident.urgency.value,
                        "service_id": incident.service_id,
                        "service_name": incident.service_name,
                        "incident_number": incident.incident_number,
                        "created_at": (
                            incident.created_at.isoformat() if incident.created_at else None
                        ),
                        "html_url": incident.html_url,
                        "description": incident.description,
                        "assignees": incident.assignees,
                        "priority": incident.priority,
                    }
                }
            )

        except Exception as e:
            logger.error(f"Failed to get incident {incident_id}: {e}")
            return error_response(f"Failed to get incident: {e}", 500)

    async def _handle_acknowledge_incident(
        self, request: Any, tenant_id: str, incident_id: str
    ) -> HandlerResult:
        """Acknowledge an incident."""
        connector = await get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        try:
            incident = await connector.acknowledge_incident(incident_id)

            logger.info(f"[DevOps] Acknowledged incident {incident_id} for tenant {tenant_id}")

            return success_response(
                {
                    "incident": {
                        "id": incident.id,
                        "status": incident.status.value,
                    },
                    "message": "Incident acknowledged",
                }
            )

        except Exception as e:
            logger.error(f"Failed to acknowledge incident {incident_id}: {e}")
            return error_response(f"Failed to acknowledge incident: {e}", 500)

    async def _handle_resolve_incident(
        self, request: Any, tenant_id: str, incident_id: str
    ) -> HandlerResult:
        """Resolve an incident.

        Request body:
        {
            "resolution": "Fixed by restarting the service"
        }
        """
        connector = await get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        body = await self._get_json_body(request)
        resolution = body.get("resolution")

        try:
            incident = await connector.resolve_incident(incident_id, resolution)

            logger.info(f"[DevOps] Resolved incident {incident_id} for tenant {tenant_id}")

            return success_response(
                {
                    "incident": {
                        "id": incident.id,
                        "status": incident.status.value,
                    },
                    "message": "Incident resolved",
                }
            )

        except Exception as e:
            logger.error(f"Failed to resolve incident {incident_id}: {e}")
            return error_response(f"Failed to resolve incident: {e}", 500)

    async def _handle_reassign_incident(
        self, request: Any, tenant_id: str, incident_id: str
    ) -> HandlerResult:
        """Reassign an incident.

        Request body:
        {
            "user_ids": ["PUSER123"],  // or
            "escalation_policy_id": "PESCPOL123"
        }
        """
        connector = await get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        body = await self._get_json_body(request)
        user_ids = body.get("user_ids")
        escalation_policy_id = body.get("escalation_policy_id")

        if not user_ids and not escalation_policy_id:
            return error_response("Either user_ids or escalation_policy_id is required", 400)

        try:
            incident = await connector.reassign_incident(
                incident_id,
                user_ids=user_ids,
                escalation_policy_id=escalation_policy_id,
            )

            logger.info(f"[DevOps] Reassigned incident {incident_id} for tenant {tenant_id}")

            return success_response(
                {
                    "incident": {
                        "id": incident.id,
                        "assignees": incident.assignees,
                    },
                    "message": "Incident reassigned",
                }
            )

        except Exception as e:
            logger.error(f"Failed to reassign incident {incident_id}: {e}")
            return error_response(f"Failed to reassign incident: {e}", 500)

    async def _handle_merge_incidents(
        self, request: Any, tenant_id: str, incident_id: str
    ) -> HandlerResult:
        """Merge incidents into this one.

        Request body:
        {
            "source_incident_ids": ["PINC456", "PINC789"]
        }
        """
        connector = await get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        body = await self._get_json_body(request)
        source_ids = body.get("source_incident_ids", [])

        if not source_ids:
            return error_response("source_incident_ids is required", 400)

        try:
            incident = await connector.merge_incidents(incident_id, source_ids)

            logger.info(f"[DevOps] Merged {len(source_ids)} incidents into {incident_id}")

            return success_response(
                {
                    "incident": {
                        "id": incident.id,
                        "title": incident.title,
                    },
                    "message": f"Merged {len(source_ids)} incidents",
                }
            )

        except Exception as e:
            logger.error(f"Failed to merge incidents into {incident_id}: {e}")
            return error_response(f"Failed to merge incidents: {e}", 500)

    # =========================================================================
    # Notes
    # =========================================================================

    async def _handle_list_notes(
        self, request: Any, tenant_id: str, incident_id: str
    ) -> HandlerResult:
        """List notes for an incident."""
        connector = await get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        try:
            notes = await connector.list_notes(incident_id)

            return success_response(
                {
                    "notes": [
                        {
                            "id": note.id,
                            "content": note.content,
                            "created_at": (
                                note.created_at.isoformat() if note.created_at else None
                            ),
                            "user": (
                                {"id": note.user.id, "name": note.user.name} if note.user else None
                            ),
                        }
                        for note in notes
                    ],
                    "count": len(notes),
                }
            )

        except Exception as e:
            logger.error(f"Failed to list notes for {incident_id}: {e}")
            return error_response(f"Failed to list notes: {e}", 500)

    async def _handle_add_note(
        self, request: Any, tenant_id: str, incident_id: str
    ) -> HandlerResult:
        """Add a note to an incident.

        Request body:
        {
            "content": "Investigation update: found root cause..."
        }
        """
        connector = await get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        body = await self._get_json_body(request)
        content = body.get("content")

        if not content:
            return error_response("content is required", 400)

        try:
            note = await connector.add_note(incident_id, content)

            logger.info(f"[DevOps] Added note to incident {incident_id}")

            return json_response(
                {
                    "note": {
                        "id": note.id,
                        "content": note.content,
                        "created_at": (note.created_at.isoformat() if note.created_at else None),
                    },
                    "message": "Note added",
                },
                status=201,
            )

        except Exception as e:
            logger.error(f"Failed to add note to {incident_id}: {e}")
            return error_response(f"Failed to add note: {e}", 500)

    # =========================================================================
    # On-Call
    # =========================================================================

    async def _handle_get_oncall(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get current on-call schedules."""
        connector = await get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        params = self._get_query_params(request)
        schedule_ids = (
            params.get("schedule_ids", "").split(",") if params.get("schedule_ids") else None
        )

        try:
            schedules = await connector.get_on_call(schedule_ids=schedule_ids)

            return success_response(
                {
                    "oncall": [
                        {
                            "schedule_id": sched.schedule_id,
                            "schedule_name": sched.schedule_name,
                            "user": {
                                "id": sched.user.id,
                                "name": sched.user.name,
                                "email": sched.user.email,
                            },
                            "start": sched.start.isoformat(),
                            "end": sched.end.isoformat(),
                            "escalation_level": sched.escalation_level,
                        }
                        for sched in schedules
                    ],
                    "count": len(schedules),
                }
            )

        except Exception as e:
            logger.error(f"Failed to get on-call: {e}")
            return error_response(f"Failed to get on-call: {e}", 500)

    async def _handle_get_oncall_for_service(
        self, request: Any, tenant_id: str, service_id: str
    ) -> HandlerResult:
        """Get current on-call for a specific service."""
        connector = await get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        try:
            schedules = await connector.get_current_on_call_for_service(service_id)

            return success_response(
                {
                    "service_id": service_id,
                    "oncall": [
                        {
                            "schedule_id": sched.schedule_id,
                            "schedule_name": sched.schedule_name,
                            "user": {
                                "id": sched.user.id,
                                "name": sched.user.name,
                                "email": sched.user.email,
                            },
                            "escalation_level": sched.escalation_level,
                        }
                        for sched in schedules
                    ],
                }
            )

        except Exception as e:
            logger.error(f"Failed to get on-call for service {service_id}: {e}")
            return error_response(f"Failed to get on-call: {e}", 500)

    # =========================================================================
    # Services
    # =========================================================================

    async def _handle_list_services(self, request: Any, tenant_id: str) -> HandlerResult:
        """List PagerDuty services."""
        connector = await get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        params = self._get_query_params(request)
        limit = int(params.get("limit", 25))
        offset = int(params.get("offset", 0))

        try:
            services, has_more = await connector.list_services(limit=limit, offset=offset)

            return success_response(
                {
                    "services": [
                        {
                            "id": svc.id,
                            "name": svc.name,
                            "description": svc.description,
                            "status": svc.status.value,
                            "html_url": svc.html_url,
                        }
                        for svc in services
                    ],
                    "count": len(services),
                    "has_more": has_more,
                }
            )

        except Exception as e:
            logger.error(f"Failed to list services: {e}")
            return error_response(f"Failed to list services: {e}", 500)

    async def _handle_get_service(
        self, request: Any, tenant_id: str, service_id: str
    ) -> HandlerResult:
        """Get service details."""
        connector = await get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        try:
            service = await connector.get_service(service_id)

            return success_response(
                {
                    "service": {
                        "id": service.id,
                        "name": service.name,
                        "description": service.description,
                        "status": service.status.value,
                        "escalation_policy_id": service.escalation_policy_id,
                        "html_url": service.html_url,
                        "created_at": (
                            service.created_at.isoformat() if service.created_at else None
                        ),
                    }
                }
            )

        except Exception as e:
            logger.error(f"Failed to get service {service_id}: {e}")
            return error_response(f"Failed to get service: {e}", 500)

    # =========================================================================
    # Webhooks
    # =========================================================================

    async def _handle_pagerduty_webhook(self, request: Any, tenant_id: str) -> HandlerResult:
        """Handle PagerDuty webhook notifications."""
        try:
            # Get raw body for signature verification
            raw_body = await self._get_raw_body(request)
            signature = self._get_header(request, "X-PagerDuty-Signature")

            connector = await get_pagerduty_connector(tenant_id)
            if connector:
                # Verify signature if configured
                if not connector.verify_webhook_signature(raw_body, signature or ""):
                    logger.warning("[DevOps] Invalid PagerDuty webhook signature")
                    # Don't reject - could be misconfigured

            body = await self._get_json_body(request)
            payload = None

            if connector:
                payload = connector.parse_webhook(body)

            event_type = payload.event_type if payload else body.get("event", {}).get("event_type")

            logger.info(f"[DevOps] PagerDuty webhook: event_type={event_type}")

            # Emit event for downstream processing
            await self._emit_connector_event(
                event_type=event_type or "unknown",
                tenant_id=tenant_id,
                data={
                    "payload": (
                        payload.to_dict() if payload and hasattr(payload, "to_dict") else body
                    ),
                },
            )

            return success_response(
                {
                    "received": True,
                    "event_type": event_type,
                }
            )

        except Exception as e:
            logger.error(f"Error processing PagerDuty webhook: {e}")
            return success_response({"received": True, "error": str(e)})

    # =========================================================================
    # Utilities
    # =========================================================================

    def _get_query_params(self, request: Any) -> Dict[str, str]:
        """Extract query parameters from request."""
        if hasattr(request, "query"):
            return dict(request.query)
        if hasattr(request, "query_string"):
            from urllib.parse import parse_qs

            return {k: v[0] for k, v in parse_qs(request.query_string).items()}
        return {}

    async def _get_json_body(self, request: Any) -> Dict[str, Any]:
        """Parse JSON body from request."""
        if hasattr(request, "json"):
            if callable(request.json):
                return await request.json()
            return request.json
        return {}

    async def _get_raw_body(self, request: Any) -> bytes:
        """Get raw request body."""
        if hasattr(request, "body"):
            if callable(request.body):
                return await request.body()
            return request.body
        if hasattr(request, "read"):
            return await request.read()
        return b""

    def _get_header(self, request: Any, name: str) -> Optional[str]:
        """Get request header."""
        if hasattr(request, "headers"):
            return request.headers.get(name)
        return None

    async def _emit_connector_event(
        self,
        event_type: str,
        tenant_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Emit a connector event for downstream processing.

        Events can trigger workflows, update dashboards, or send notifications.
        """
        try:
            from aragora.events.types import StreamEventType

            event_data = {
                "connector": "pagerduty",
                "event_type": event_type,
                "tenant_id": tenant_id,
                **data,
            }

            # Log structured event for processing pipelines
            logger.info(
                f"[DevOps] Connector event: {event_type}",
                extra={"event_data": event_data},
            )

            # If we have a server context with an emitter, emit the event
            if self.ctx and "emitter" in self.ctx:
                emitter = self.ctx["emitter"]  # type: ignore[typeddict-item]
                emitter.emit(
                    StreamEventType.CONNECTOR_PAGERDUTY_INCIDENT.value,
                    event_data,
                )
        except Exception as e:
            logger.debug(f"[DevOps] Event emission skipped: {e}")


# =============================================================================
# Factory
# =============================================================================


def create_devops_handler(
    server_context: Optional[Dict[str, Any]] = None,
) -> DevOpsHandler:
    """Create a devops handler instance."""
    return DevOpsHandler(server_context)


__all__ = ["DevOpsHandler", "create_devops_handler"]
