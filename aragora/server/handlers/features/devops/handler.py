"""DevOps Incident Management API Handler.

Stability: STABLE

Provides REST APIs for incident management via PagerDuty:
- Create and manage incidents
- Add investigation notes
- Acknowledge and resolve incidents
- Query on-call schedules
- Service health status
- Webhook handling for incident updates

Features:
- Circuit breaker pattern for resilient PagerDuty API access
- Rate limiting (30-60 requests/minute depending on endpoint)
- RBAC permission checks (devops:read, devops:write, devops:webhook)
- Comprehensive input validation with safe ID patterns
- Webhook signature verification

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
from typing import Any

from ...base import (
    HandlerResult,
    error_response,
    json_response,
    success_response,
)
from ...secure import ForbiddenError, SecureHandler, UnauthorizedError
from ...utils import parse_json_body
from ...utils.rate_limit import rate_limit
from aragora.server.validation.query_params import safe_query_int, safe_query_string

from .circuit_breaker import (
    get_devops_circuit_breaker,
    get_devops_circuit_breaker_status,
)
import aragora.server.handlers.features.devops.connector as _connector_mod
from .validation import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NOTE_CONTENT_LENGTH,
    MAX_RESOLUTION_LENGTH,
    MAX_SOURCE_INCIDENT_IDS,
    MAX_TITLE_LENGTH,
    MAX_USER_IDS,
    VALID_INCIDENT_STATUSES,
    VALID_URGENCIES,
    validate_id_list,
    validate_pagerduty_id,
    validate_string_field,
    validate_urgency,
)

# Permission constants for DevOps operations
DEVOPS_READ_PERMISSION = "devops:read"
DEVOPS_WRITE_PERMISSION = "devops:write"
DEVOPS_WEBHOOK_PERMISSION = "devops:webhook"

logger = logging.getLogger(__name__)


class DevOpsHandler(SecureHandler):
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

    def __init__(self, server_context: dict[str, Any] | None = None):
        """Initialize handler with optional server context."""
        # ServerContext is a TypedDict with total=False, so empty dict is valid
        ctx: dict[str, Any] = server_context if server_context is not None else dict()
        super().__init__(ctx)

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return (
            path.startswith("/api/v1/incidents")
            or path.startswith("/api/v1/oncall")
            or path.startswith("/api/v1/services")
            or path.startswith("/api/v1/webhooks/pagerduty")
            or path.startswith("/api/v1/devops")
        )

    async def handle(
        self,
        path_or_request: Any,
        query_params_or_path: dict[str, Any] | str = "",
        handler_or_method: Any = "GET",
    ) -> HandlerResult:
        """Route requests to appropriate handler methods.

        This method has a flexible signature to support both:
        - Legacy pattern: handle(path, query_params, handler)
        - Async pattern: handle(request, path, method)

        The implementation detects which pattern is being used.
        """
        # Detect calling pattern based on argument types
        if isinstance(query_params_or_path, str):
            # Async pattern: handle(request, path, method)
            request = path_or_request
            path = query_params_or_path
            method = str(handler_or_method) if handler_or_method else "GET"
        else:
            # Legacy pattern: handle(path, query_params, handler) - not used for DevOps
            # For this handler, we expect the async pattern
            raise TypeError("DevOpsHandler.handle expects (request, path, method) signature")

        # RBAC: Require authentication and appropriate permission
        try:
            auth_context = await self.get_auth_context(request, require_auth=True)
            # Determine required permission based on method and path
            if path == "/api/v1/webhooks/pagerduty":
                self.check_permission(auth_context, DEVOPS_WEBHOOK_PERMISSION)
            elif method in ("POST", "PUT", "DELETE"):
                self.check_permission(auth_context, DEVOPS_WRITE_PERMISSION)
            else:
                self.check_permission(auth_context, DEVOPS_READ_PERMISSION)
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            logger.warning("Handler error: %s", e)
            return error_response("Permission denied", 403)

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
            return error_response("Internal server error", 500)

    def _get_tenant_id(self, request: Any) -> str:
        """Extract tenant ID from request context."""
        return getattr(request, "tenant_id", "default")

    # =========================================================================
    # Status
    # =========================================================================

    @rate_limit(requests_per_minute=60)
    async def _handle_status(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get PagerDuty connection status and circuit breaker status."""
        import os

        api_key = os.getenv("PAGERDUTY_API_KEY")
        email = os.getenv("PAGERDUTY_EMAIL")

        circuit_status = get_devops_circuit_breaker_status()

        return success_response(
            {
                "configured": bool(api_key and email),
                "api_key_set": bool(api_key),
                "email_set": bool(email),
                "webhook_secret_set": bool(os.getenv("PAGERDUTY_WEBHOOK_SECRET")),
                "circuit_breaker": circuit_status,
            }
        )

    # =========================================================================
    # Incidents
    # =========================================================================

    @rate_limit(requests_per_minute=60)
    async def _handle_list_incidents(self, request: Any, tenant_id: str) -> HandlerResult:
        """List incidents with filtering.

        Query params:
        - status: Filter by status (triggered, acknowledged, resolved)
        - service_ids: Comma-separated service IDs
        - urgency: Filter by urgency (high, low)
        - limit: Max results (default 25)
        - offset: Pagination offset
        """
        # Check circuit breaker
        circuit_breaker = get_devops_circuit_breaker()
        if not circuit_breaker.is_allowed():
            return error_response("PagerDuty service temporarily unavailable", 503)

        connector = await _connector_mod.get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        params = self._get_query_params(request)

        # Validate and parse status filter
        statuses = None
        status_param = safe_query_string(params, "status", "", max_length=100)
        if status_param:
            statuses = [s.strip() for s in status_param.split(",") if s.strip()]
            # Validate statuses
            for status in statuses:
                if status not in VALID_INCIDENT_STATUSES:
                    return error_response(
                        f"Invalid status '{status}'. Valid: {', '.join(VALID_INCIDENT_STATUSES)}",
                        400,
                    )

        # Validate and parse service_ids filter
        service_ids = None
        service_ids_param = safe_query_string(params, "service_ids", "", max_length=500)
        if service_ids_param:
            service_ids = [s.strip() for s in service_ids_param.split(",") if s.strip()]
            # Limit to 20 service IDs; allow short IDs for read filters
            service_ids = service_ids[:20]

        # Validate and parse urgency filter
        urgencies = None
        urgency_param = safe_query_string(params, "urgency", "", max_length=50)
        if urgency_param:
            urgencies = [u.strip() for u in urgency_param.split(",") if u.strip()]
            for urgency in urgencies:
                if urgency not in VALID_URGENCIES:
                    return error_response(
                        f"Invalid urgency '{urgency}'. Valid: {', '.join(VALID_URGENCIES)}",
                        400,
                    )

        limit = safe_query_int(params, "limit", default=25, min_val=1, max_val=100)
        offset = safe_query_int(params, "offset", default=0, min_val=0, max_val=10000)

        try:
            incidents, has_more = await connector.list_incidents(
                statuses=statuses,
                service_ids=service_ids,
                urgencies=urgencies,
                limit=limit,
                offset=offset,
            )

            circuit_breaker.record_success()

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
            circuit_breaker.record_failure()
            logger.error(f"Failed to list incidents: {e}")
            return error_response("Failed to list incidents", 500)

    @rate_limit(requests_per_minute=30)
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
        # Check circuit breaker
        circuit_breaker = get_devops_circuit_breaker()
        if not circuit_breaker.is_allowed():
            return error_response("PagerDuty service temporarily unavailable", 503)

        connector = await _connector_mod.get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        body = await self._get_json_body(request)

        # Validate title
        title, err = validate_string_field(
            body.get("title"), "title", required=True, max_length=MAX_TITLE_LENGTH
        )
        if err:
            return error_response(err, 400)

        # Validate service_id
        is_valid, err = validate_pagerduty_id(body.get("service_id", ""), "service_id")
        if not is_valid:
            return error_response(err, 400)
        service_id = body["service_id"]

        # Validate urgency
        urgency_str = validate_urgency(body.get("urgency"))

        # Validate description (optional)
        description, err = validate_string_field(
            body.get("body"), "body", required=False, max_length=MAX_DESCRIPTION_LENGTH
        )
        if err:
            return error_response(err, 400)

        # Validate escalation_policy_id (optional)
        escalation_policy_id = body.get("escalation_policy_id")
        if escalation_policy_id:
            is_valid, err = validate_pagerduty_id(escalation_policy_id, "escalation_policy_id")
            if not is_valid:
                return error_response(err, 400)

        # Validate priority_id (optional)
        priority_id = body.get("priority_id")
        if priority_id:
            is_valid, err = validate_pagerduty_id(priority_id, "priority_id")
            if not is_valid:
                return error_response(err, 400)

        try:
            from aragora.connectors.devops.pagerduty import (
                IncidentCreateRequest,
                IncidentUrgency,
            )

            urgency = IncidentUrgency(urgency_str)

            create_request = IncidentCreateRequest(
                title=title,
                service_id=service_id,
                urgency=urgency,
                description=description,
                escalation_policy_id=escalation_policy_id,
                priority_id=priority_id,
            )

            incident = await connector.create_incident(create_request)

            circuit_breaker.record_success()
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
            circuit_breaker.record_failure()
            logger.error(f"Failed to create incident: {e}")
            return error_response("Failed to create incident", 500)

    @rate_limit(requests_per_minute=60)
    async def _handle_get_incident(
        self, request: Any, tenant_id: str, incident_id: str
    ) -> HandlerResult:
        """Get incident details."""
        # Validate incident_id
        is_valid, err = validate_pagerduty_id(incident_id, "incident_id")
        if not is_valid:
            return error_response(err, 400)

        # Check circuit breaker
        circuit_breaker = get_devops_circuit_breaker()
        if not circuit_breaker.is_allowed():
            return error_response("PagerDuty service temporarily unavailable", 503)

        connector = await _connector_mod.get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        try:
            incident = await connector.get_incident(incident_id)

            circuit_breaker.record_success()

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
            circuit_breaker.record_failure()
            logger.error(f"Failed to get incident {incident_id}: {e}")
            return error_response("Failed to get incident", 500)

    @rate_limit(requests_per_minute=30)
    async def _handle_acknowledge_incident(
        self, request: Any, tenant_id: str, incident_id: str
    ) -> HandlerResult:
        """Acknowledge an incident."""
        # Validate incident_id
        is_valid, err = validate_pagerduty_id(incident_id, "incident_id")
        if not is_valid:
            return error_response(err, 400)

        # Check circuit breaker
        circuit_breaker = get_devops_circuit_breaker()
        if not circuit_breaker.is_allowed():
            return error_response("PagerDuty service temporarily unavailable", 503)

        connector = await _connector_mod.get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        try:
            incident = await connector.acknowledge_incident(incident_id)

            circuit_breaker.record_success()
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
            circuit_breaker.record_failure()
            logger.error(f"Failed to acknowledge incident {incident_id}: {e}")
            return error_response("Failed to acknowledge incident", 500)

    @rate_limit(requests_per_minute=30)
    async def _handle_resolve_incident(
        self, request: Any, tenant_id: str, incident_id: str
    ) -> HandlerResult:
        """Resolve an incident.

        Request body:
        {
            "resolution": "Fixed by restarting the service"
        }
        """
        # Validate incident_id
        is_valid, err = validate_pagerduty_id(incident_id, "incident_id")
        if not is_valid:
            return error_response(err, 400)

        # Check circuit breaker
        circuit_breaker = get_devops_circuit_breaker()
        if not circuit_breaker.is_allowed():
            return error_response("PagerDuty service temporarily unavailable", 503)

        connector = await _connector_mod.get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        body = await self._get_json_body(request)

        # Validate resolution (optional)
        resolution, err = validate_string_field(
            body.get("resolution"), "resolution", required=False, max_length=MAX_RESOLUTION_LENGTH
        )
        if err:
            return error_response(err, 400)

        try:
            incident = await connector.resolve_incident(incident_id, resolution)

            circuit_breaker.record_success()
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
            circuit_breaker.record_failure()
            logger.error(f"Failed to resolve incident {incident_id}: {e}")
            return error_response("Failed to resolve incident", 500)

    @rate_limit(requests_per_minute=30)
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
        # Validate incident_id
        is_valid, err = validate_pagerduty_id(incident_id, "incident_id")
        if not is_valid:
            return error_response(err, 400)

        # Check circuit breaker
        circuit_breaker = get_devops_circuit_breaker()
        if not circuit_breaker.is_allowed():
            return error_response("PagerDuty service temporarily unavailable", 503)

        connector = await _connector_mod.get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        body = await self._get_json_body(request)
        user_ids = body.get("user_ids")
        escalation_policy_id = body.get("escalation_policy_id")

        if not user_ids and not escalation_policy_id:
            return error_response("Either user_ids or escalation_policy_id is required", 400)

        # Validate user_ids (if provided)
        if user_ids:
            validated_user_ids, err = validate_id_list(user_ids, "user_ids", max_items=MAX_USER_IDS)
            if err:
                return error_response(err, 400)
            user_ids = validated_user_ids

        # Validate escalation_policy_id (if provided)
        if escalation_policy_id:
            is_valid, err = validate_pagerduty_id(escalation_policy_id, "escalation_policy_id")
            if not is_valid:
                return error_response(err, 400)

        try:
            incident = await connector.reassign_incident(
                incident_id,
                user_ids=user_ids,
                escalation_policy_id=escalation_policy_id,
            )

            circuit_breaker.record_success()
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
            circuit_breaker.record_failure()
            logger.error(f"Failed to reassign incident {incident_id}: {e}")
            return error_response("Failed to reassign incident", 500)

    @rate_limit(requests_per_minute=20)
    async def _handle_merge_incidents(
        self, request: Any, tenant_id: str, incident_id: str
    ) -> HandlerResult:
        """Merge incidents into this one.

        Request body:
        {
            "source_incident_ids": ["PINC456", "PINC789"]
        }
        """
        # Validate incident_id
        is_valid, err = validate_pagerduty_id(incident_id, "incident_id")
        if not is_valid:
            return error_response(err, 400)

        # Check circuit breaker
        circuit_breaker = get_devops_circuit_breaker()
        if not circuit_breaker.is_allowed():
            return error_response("PagerDuty service temporarily unavailable", 503)

        connector = await _connector_mod.get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        body = await self._get_json_body(request)
        source_ids = body.get("source_incident_ids", [])

        if not source_ids:
            return error_response("source_incident_ids is required", 400)

        # Validate source_incident_ids
        validated_source_ids, err = validate_id_list(
            source_ids, "source_incident_ids", max_items=MAX_SOURCE_INCIDENT_IDS
        )
        if err:
            return error_response(err, 400)

        try:
            incident = await connector.merge_incidents(incident_id, validated_source_ids)

            circuit_breaker.record_success()
            logger.info(f"[DevOps] Merged {len(validated_source_ids)} incidents into {incident_id}")

            return success_response(
                {
                    "incident": {
                        "id": incident.id,
                        "title": incident.title,
                    },
                    "message": f"Merged {len(validated_source_ids)} incidents",
                }
            )

        except Exception as e:
            circuit_breaker.record_failure()
            logger.error(f"Failed to merge incidents into {incident_id}: {e}")
            return error_response("Failed to merge incidents", 500)

    # =========================================================================
    # Notes
    # =========================================================================

    @rate_limit(requests_per_minute=60)
    async def _handle_list_notes(
        self, request: Any, tenant_id: str, incident_id: str
    ) -> HandlerResult:
        """List notes for an incident."""
        # Validate incident_id
        is_valid, err = validate_pagerduty_id(incident_id, "incident_id")
        if not is_valid:
            return error_response(err, 400)

        # Check circuit breaker
        circuit_breaker = get_devops_circuit_breaker()
        if not circuit_breaker.is_allowed():
            return error_response("PagerDuty service temporarily unavailable", 503)

        connector = await _connector_mod.get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        try:
            notes = await connector.list_notes(incident_id)

            circuit_breaker.record_success()

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
            circuit_breaker.record_failure()
            logger.error(f"Failed to list notes for {incident_id}: {e}")
            return error_response("Failed to list notes", 500)

    @rate_limit(requests_per_minute=30)
    async def _handle_add_note(
        self, request: Any, tenant_id: str, incident_id: str
    ) -> HandlerResult:
        """Add a note to an incident.

        Request body:
        {
            "content": "Investigation update: found root cause..."
        }
        """
        # Validate incident_id
        is_valid, err = validate_pagerduty_id(incident_id, "incident_id")
        if not is_valid:
            return error_response(err, 400)

        # Check circuit breaker
        circuit_breaker = get_devops_circuit_breaker()
        if not circuit_breaker.is_allowed():
            return error_response("PagerDuty service temporarily unavailable", 503)

        connector = await _connector_mod.get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        body = await self._get_json_body(request)

        # Validate content
        content, err = validate_string_field(
            body.get("content"), "content", required=True, max_length=MAX_NOTE_CONTENT_LENGTH
        )
        if err:
            return error_response(err, 400)

        try:
            note = await connector.add_note(incident_id, content)

            circuit_breaker.record_success()
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
            circuit_breaker.record_failure()
            logger.error(f"Failed to add note to {incident_id}: {e}")
            return error_response("Failed to add note", 500)

    # =========================================================================
    # On-Call
    # =========================================================================

    @rate_limit(requests_per_minute=60)
    async def _handle_get_oncall(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get current on-call schedules."""
        # Check circuit breaker
        circuit_breaker = get_devops_circuit_breaker()
        if not circuit_breaker.is_allowed():
            return error_response("PagerDuty service temporarily unavailable", 503)

        connector = await _connector_mod.get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        params = self._get_query_params(request)

        # Validate and parse schedule_ids filter
        schedule_ids = None
        schedule_ids_param = safe_query_string(params, "schedule_ids", "", max_length=500)
        if schedule_ids_param:
            schedule_ids = [s.strip() for s in schedule_ids_param.split(",") if s.strip()]
            for sid in schedule_ids[:20]:  # Limit to 20 schedule IDs
                is_valid, err = validate_pagerduty_id(sid, "schedule_id")
                if not is_valid:
                    return error_response(err, 400)
            schedule_ids = schedule_ids[:20]

        try:
            schedules = await connector.get_on_call(schedule_ids=schedule_ids)

            circuit_breaker.record_success()

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
            circuit_breaker.record_failure()
            logger.error(f"Failed to get on-call: {e}")
            return error_response("Failed to get on-call", 500)

    async def _handle_get_oncall_for_service(
        self, request: Any, tenant_id: str, service_id: str
    ) -> HandlerResult:
        """Get current on-call for a specific service."""
        connector = await _connector_mod.get_pagerduty_connector(tenant_id)
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
            return error_response("Failed to retrieve on-call info", 500)

    # =========================================================================
    # Services
    # =========================================================================

    async def _handle_list_services(self, request: Any, tenant_id: str) -> HandlerResult:
        """List PagerDuty services."""
        connector = await _connector_mod.get_pagerduty_connector(tenant_id)
        if not connector:
            return error_response("PagerDuty not configured", 503)

        params = self._get_query_params(request)
        limit = safe_query_int(params, "limit", default=25, min_val=1, max_val=1000)
        offset = safe_query_int(params, "offset", default=0, min_val=0, max_val=100000)

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
            return error_response("Failed to list services", 500)

    async def _handle_get_service(
        self, request: Any, tenant_id: str, service_id: str
    ) -> HandlerResult:
        """Get service details."""
        connector = await _connector_mod.get_pagerduty_connector(tenant_id)
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
            return error_response("Failed to retrieve service", 500)

    # =========================================================================
    # Webhooks
    # =========================================================================

    async def _handle_pagerduty_webhook(self, request: Any, tenant_id: str) -> HandlerResult:
        """Handle PagerDuty webhook notifications."""
        try:
            # Get raw body for signature verification
            raw_body = await self._get_raw_body(request)
            signature = self._get_header(request, "X-PagerDuty-Signature")

            connector = await _connector_mod.get_pagerduty_connector(tenant_id)
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
            return success_response({"received": True, "error": "Webhook processing failed"})

    # =========================================================================
    # Utilities
    # =========================================================================

    def _get_query_params(self, request: Any) -> dict[str, str]:
        """Extract query parameters from request."""
        if hasattr(request, "query"):
            return dict(request.query)
        if hasattr(request, "query_string"):
            from urllib.parse import parse_qs

            return {k: v[0] for k, v in parse_qs(request.query_string).items()}
        return {}

    async def _get_json_body(self, request: Any) -> dict[str, Any]:
        """Parse JSON body from request."""
        if hasattr(request, "json"):
            if callable(request.json):
                try:
                    return await request.json()
                except (ValueError, TypeError):
                    body, _err = await parse_json_body(request, context="devops._get_json_body")
                    return body if body is not None else {}
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

    def _get_header(self, request: Any, name: str) -> str | None:
        """Get request header."""
        if hasattr(request, "headers"):
            return request.headers.get(name)
        return None

    async def _emit_connector_event(
        self,
        event_type: str,
        tenant_id: str,
        data: dict[str, Any],
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
            if self.ctx and isinstance(self.ctx, dict) and "emitter" in self.ctx:
                emitter = self.ctx["emitter"]
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
    server_context: dict[str, Any] | None = None,
) -> DevOpsHandler:
    """Create a devops handler instance."""
    return DevOpsHandler(server_context)
