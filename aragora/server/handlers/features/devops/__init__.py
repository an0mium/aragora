"""
DevOps Incident Management Handler (PagerDuty).

Provides endpoints for:
- incidents (list/create/get/ack/resolve/reassign/merge)
- incident notes (list/add)
- on-call schedules
- services
- PagerDuty webhooks
- status checks
"""

from __future__ import annotations

import inspect
import logging
import os
from datetime import datetime
from typing import Any

from aragora.rbac.checker import get_permission_checker
from aragora.server.handlers.base import BaseHandler
from aragora.server.handlers.utils.auth import (
    ForbiddenError,
    UnauthorizedError,
    get_auth_context,
)
from aragora.server.handlers.utils.responses import HandlerResult, error_response, json_response
from aragora.server.validation.query_params import safe_query_int

from .circuit_breaker import (
    DevOpsCircuitBreaker,
    get_devops_circuit_breaker,
    get_devops_circuit_breaker_status,
)
from .connector import (
    _active_contexts,
    _connector_instances,
    clear_connector_instances,
    get_pagerduty_connector,
)
from .validation import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NOTE_CONTENT_LENGTH,
    MAX_RESOLUTION_LENGTH,
    MAX_SOURCE_INCIDENT_IDS,
    MAX_TITLE_LENGTH,
    MAX_USER_IDS,
    VALID_INCIDENT_STATUSES,
    VALID_URGENCIES,
    validate_id_list as _validate_id_list,
    validate_pagerduty_id as _validate_pagerduty_id,
    validate_string_field as _validate_string_field,
    validate_urgency as _validate_urgency,
)

logger = logging.getLogger(__name__)

DEVOPS_READ_PERMISSION = "devops:read"
DEVOPS_WRITE_PERMISSION = "devops:write"
DEVOPS_WEBHOOK_PERMISSION = "devops:webhook"


def create_devops_handler(server_context: dict[str, Any] | None = None) -> DevOpsHandler:
    """Factory to create a DevOpsHandler instance."""
    return DevOpsHandler(server_context=server_context or {})


def _clear_devops_components() -> None:
    """Clear cached connectors and reset circuit breaker."""
    clear_connector_instances()
    try:
        get_devops_circuit_breaker().reset()
    except (RuntimeError, ValueError, AttributeError) as e:
        logger.debug("Failed to reset devops circuit breaker: %s", e)


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _value_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "value"):
        return value.value
    return value


class DevOpsHandler(BaseHandler):
    """Handler for DevOps incident management endpoints."""

    ROUTES = [
        "/api/v1/incidents",
        "/api/v1/incidents/*",
        "/api/v1/oncall",
        "/api/v1/oncall/*",
        "/api/v1/services",
        "/api/v1/services/*",
        "/api/v1/webhooks/pagerduty",
        "/api/v1/devops/status",
    ]

    async def get_auth_context(self, request: Any, require_auth: bool = False):
        return await get_auth_context(request, require_auth=require_auth)

    def check_permission(self, auth_ctx: Any, permission: str, resource_id: str | None = None):
        checker = get_permission_checker()
        decision = checker.check_permission(auth_ctx, permission, resource_id)
        if not decision.allowed:
            logger.warning("Permission denied: %s", permission)
            raise ForbiddenError("Permission denied", permission=permission)

    def can_handle(self, path: str, method: str = "GET") -> bool:
        if path.startswith("/api/v1/incidents"):
            return True
        if path.startswith("/api/v1/oncall"):
            return True
        if path.startswith("/api/v1/services"):
            return True
        if path.startswith("/api/v1/webhooks/pagerduty"):
            return True
        if path.startswith("/api/v1/devops"):
            return True
        return False

    async def handle(self, request: Any, path: str, method: str) -> HandlerResult | None:  # type: ignore[override]
        if not self.can_handle(path):
            return None

        try:
            auth_ctx = await self.get_auth_context(request, require_auth=True)
            permission = self._get_permission_for_request(path, method)
            if permission is None:
                return error_response("Not found", 404)
            self.check_permission(auth_ctx, permission)

            tenant_id = self._get_tenant_id(request)
            parts = path.strip("/").split("/")

            if path == "/api/v1/devops/status" and method == "GET":
                return self._handle_status()

            if parts[:3] == ["api", "v1", "incidents"]:
                if len(parts) == 3:
                    if method == "GET":
                        return await self._handle_list_incidents(request, tenant_id)
                    if method == "POST":
                        return await self._handle_create_incident(request, tenant_id)
                if len(parts) >= 4:
                    incident_id = parts[3]
                    if len(parts) == 4 and method == "GET":
                        return await self._handle_get_incident(request, tenant_id, incident_id)
                    if len(parts) == 5 and method == "POST":
                        action = parts[4]
                        if action == "acknowledge":
                            return await self._handle_acknowledge_incident(
                                request, tenant_id, incident_id
                            )
                        if action == "resolve":
                            return await self._handle_resolve_incident(
                                request, tenant_id, incident_id
                            )
                        if action == "reassign":
                            return await self._handle_reassign_incident(
                                request, tenant_id, incident_id
                            )
                        if action == "merge":
                            return await self._handle_merge_incidents(
                                request, tenant_id, incident_id
                            )
                        if action == "notes":
                            return await self._handle_add_note(request, tenant_id, incident_id)
                    if len(parts) == 5 and method == "GET" and parts[4] == "notes":
                        return await self._handle_list_notes(request, tenant_id, incident_id)

            if parts[:3] == ["api", "v1", "oncall"]:
                if len(parts) == 3 and method == "GET":
                    return await self._handle_get_oncall(request, tenant_id)
                if len(parts) == 5 and parts[3] == "services" and method == "GET":
                    return await self._handle_get_oncall_for_service(request, tenant_id, parts[4])

            if parts[:3] == ["api", "v1", "services"]:
                if len(parts) == 3 and method == "GET":
                    return await self._handle_list_services(request, tenant_id)
                if len(parts) == 4 and method == "GET":
                    return await self._handle_get_service(request, tenant_id, parts[3])

            if path == "/api/v1/webhooks/pagerduty" and method == "POST":
                return await self._handle_pagerduty_webhook(request, tenant_id)

            return error_response("Not found", 404)

        except UnauthorizedError as e:
            logger.warning("Handler error: %s", e)
            return error_response("Authentication required", 401)
        except ForbiddenError:
            return error_response("Permission denied", 403)
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
            logger.exception("Unhandled DevOps handler error: %s", e)
            return error_response("Internal server error", 500)

    def _get_permission_for_request(self, path: str, method: str) -> str | None:
        if path == "/api/v1/webhooks/pagerduty" and method == "POST":
            return DEVOPS_WEBHOOK_PERMISSION
        if method == "GET":
            return DEVOPS_READ_PERMISSION
        if method == "POST":
            return DEVOPS_WRITE_PERMISSION
        return None

    def _get_tenant_id(self, request: Any) -> str:
        return getattr(request, "tenant_id", None) or "default"

    def _get_query_params(self, request: Any) -> dict[str, Any]:
        query = getattr(request, "query", None)
        return query if isinstance(query, dict) else {}

    async def _get_json_body(self, request: Any) -> dict[str, Any]:
        if hasattr(request, "json"):
            json_attr = request.json
            if callable(json_attr):
                result = json_attr()
                if inspect.isawaitable(result):
                    return await result
                return result or {}
            return json_attr or {}
        return {}

    async def _get_raw_body(self, request: Any) -> bytes:
        if hasattr(request, "body"):
            body_attr = request.body
            if callable(body_attr):
                result = body_attr()
                if inspect.isawaitable(result):
                    return await result
                return result or b""
            if isinstance(body_attr, (bytes, bytearray)):
                return bytes(body_attr)
        if hasattr(request, "read"):
            read_attr = request.read
            if callable(read_attr):
                result = read_attr()
                if inspect.isawaitable(result):
                    return await result
                return result or b""
        return b""

    def _get_header(self, request: Any, name: str) -> str | None:
        headers = getattr(request, "headers", None)
        if isinstance(headers, dict):
            return headers.get(name)
        return None

    async def _emit_connector_event(self, event_type: str, tenant_id: str, data: dict[str, Any]):
        ctx = getattr(self, "ctx", None)
        if not ctx or not isinstance(ctx, dict):
            return
        emitter = ctx.get("event_emitter")
        if not emitter:
            return
        try:
            result = emitter.emit(event_type, tenant_id=tenant_id, data=data)
            if inspect.isawaitable(result):
                await result
        except (TypeError, AttributeError, RuntimeError):
            logger.debug("Failed to emit connector event", exc_info=True)

    def _handle_status(self) -> HandlerResult:
        api_key = os.getenv("PAGERDUTY_API_KEY")
        email = os.getenv("PAGERDUTY_EMAIL")
        webhook_secret = os.getenv("PAGERDUTY_WEBHOOK_SECRET")
        configured = bool(api_key and email)

        return json_response(
            {
                "data": {
                    "configured": configured,
                    "api_key_set": bool(api_key),
                    "email_set": bool(email),
                    "webhook_secret_set": bool(webhook_secret),
                    "circuit_breaker": get_devops_circuit_breaker_status(),
                }
            }
        )

    async def _handle_list_incidents(self, request: Any, tenant_id: str) -> HandlerResult:
        cb = get_devops_circuit_breaker()
        if not cb.is_allowed():
            return error_response("DevOps service temporarily unavailable", 503)

        connector = await get_pagerduty_connector(tenant_id)
        if connector is None:
            return error_response("PagerDuty connector unavailable", 503)

        query = self._get_query_params(request)

        statuses = _split_csv(query.get("status"))
        for status in statuses:
            if status not in VALID_INCIDENT_STATUSES:
                return error_response(f"Invalid status: {status}", 400)

        service_ids = _split_csv(query.get("service_ids")) or None

        urgencies = [u.lower() for u in _split_csv(query.get("urgency"))]
        for urgency in urgencies:
            if urgency not in VALID_URGENCIES:
                return error_response(f"Invalid urgency: {urgency}", 400)

        limit = safe_query_int(query, "limit", default=25, min_val=1, max_val=100)
        offset = safe_query_int(query, "offset", default=0, min_val=0, max_val=10000)

        try:
            result = await connector.list_incidents(
                statuses=statuses or None,
                service_ids=service_ids,
                urgencies=urgencies or None,
                limit=limit,
                offset=offset,
            )
            if isinstance(result, tuple) and len(result) == 2:
                incidents, has_more = result
            else:
                incidents, has_more = result, False
            cb.record_success()
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            cb.record_failure()
            logger.error("Failed to list incidents: %s", e)
            return error_response("Internal server error", 500)

        return json_response(
            {
                "data": {
                    "incidents": [self._incident_to_dict(i) for i in incidents],
                    "count": len(incidents),
                    "has_more": bool(has_more),
                }
            }
        )

    async def _handle_create_incident(self, request: Any, tenant_id: str) -> HandlerResult:
        connector = await get_pagerduty_connector(tenant_id)
        if connector is None:
            return error_response("PagerDuty connector unavailable", 503)

        body = await self._get_json_body(request)

        title, err = _validate_string_field(
            body.get("title"), "title", required=True, max_length=MAX_TITLE_LENGTH
        )
        if err:
            return error_response(err, 400)

        service_id = body.get("service_id")
        ok, err = _validate_pagerduty_id(service_id, "service_id")
        if not ok:
            return error_response(err or "Invalid service_id", 400)

        description, err = _validate_string_field(
            body.get("body") or body.get("description"),
            "description",
            required=False,
            max_length=MAX_DESCRIPTION_LENGTH,
        )
        if err:
            return error_response(err, 400)

        urgency = _validate_urgency(body.get("urgency"))

        priority_id = body.get("priority_id")
        escalation_policy_id = body.get("escalation_policy_id")
        incident_key = body.get("incident_key")
        assignments = body.get("user_ids") or body.get("assignments")

        try:
            from aragora.connectors.devops.pagerduty import (
                IncidentCreateRequest,
                IncidentUrgency,
            )

            urgency_obj = IncidentUrgency(urgency)
            request_obj = IncidentCreateRequest(
                title=title,
                service_id=service_id,
                urgency=urgency_obj,
                description=description,
                priority_id=priority_id,
                escalation_policy_id=escalation_policy_id,
                incident_key=incident_key,
                assignments=assignments,
            )
            incident = await connector.create_incident(request_obj)
        except (
            ImportError,
            ConnectionError,
            TimeoutError,
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
        ) as e:
            logger.error("Failed to create incident: %s", e)
            return error_response("Internal server error", 500)

        return json_response({"incident": self._incident_to_dict(incident)}, status=201)

    async def _handle_get_incident(
        self, request: Any, tenant_id: str, incident_id: str
    ) -> HandlerResult:
        ok, err = _validate_pagerduty_id(incident_id, "incident_id")
        if not ok:
            return error_response(err or "Invalid incident_id", 400)

        connector = await get_pagerduty_connector(tenant_id)
        if connector is None:
            return error_response("PagerDuty connector unavailable", 503)

        try:
            incident = await connector.get_incident(incident_id)
            return json_response({"data": {"incident": self._incident_to_dict(incident)}})
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            logger.error("Failed to get incident: %s", e)
            return error_response("Internal server error", 500)

    async def _handle_acknowledge_incident(
        self, request: Any, tenant_id: str, incident_id: str
    ) -> HandlerResult:
        ok, err = _validate_pagerduty_id(incident_id, "incident_id")
        if not ok:
            return error_response(err or "Invalid incident_id", 400)

        connector = await get_pagerduty_connector(tenant_id)
        if connector is None:
            return error_response("PagerDuty connector unavailable", 503)

        try:
            incident = await connector.acknowledge_incident(incident_id)
            return json_response(
                {
                    "data": {
                        "incident": self._incident_to_dict(incident),
                        "message": "Incident acknowledged",
                    }
                }
            )
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            logger.error("Failed to acknowledge incident: %s", e)
            return error_response("Internal server error", 500)

    async def _handle_resolve_incident(
        self, request: Any, tenant_id: str, incident_id: str
    ) -> HandlerResult:
        ok, err = _validate_pagerduty_id(incident_id, "incident_id")
        if not ok:
            return error_response(err or "Invalid incident_id", 400)

        connector = await get_pagerduty_connector(tenant_id)
        if connector is None:
            return error_response("PagerDuty connector unavailable", 503)

        body = await self._get_json_body(request)
        resolution, err = _validate_string_field(
            body.get("resolution"),
            "resolution",
            required=False,
            max_length=MAX_RESOLUTION_LENGTH,
        )
        if err:
            return error_response(err, 400)

        try:
            incident = await connector.resolve_incident(incident_id, resolution)
            return json_response(
                {
                    "data": {
                        "incident": self._incident_to_dict(incident),
                        "message": "Incident resolved",
                    }
                }
            )
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            logger.error("Failed to resolve incident: %s", e)
            return error_response("Internal server error", 500)

    async def _handle_reassign_incident(
        self, request: Any, tenant_id: str, incident_id: str
    ) -> HandlerResult:
        ok, err = _validate_pagerduty_id(incident_id, "incident_id")
        if not ok:
            return error_response(err or "Invalid incident_id", 400)

        connector = await get_pagerduty_connector(tenant_id)
        if connector is None:
            return error_response("PagerDuty connector unavailable", 503)

        body = await self._get_json_body(request)
        user_ids, err = _validate_id_list(body.get("user_ids"), "user_ids", max_items=MAX_USER_IDS)
        if err:
            return error_response(err, 400)

        escalation_policy_id = body.get("escalation_policy_id")
        if escalation_policy_id:
            ok, err = _validate_pagerduty_id(escalation_policy_id, "escalation_policy_id")
            if not ok:
                return error_response(err or "Invalid escalation_policy_id", 400)

        if not user_ids and not escalation_policy_id:
            return error_response("user_ids or escalation_policy_id is required", 400)

        try:
            kwargs: dict[str, Any] = {"user_ids": user_ids or []}
            sig = None
            try:
                sig = inspect.signature(connector.reassign_incident)
            except (ValueError, TypeError):
                sig = None
            if escalation_policy_id and sig and "escalation_policy_id" in sig.parameters:
                kwargs["escalation_policy_id"] = escalation_policy_id
            incident = await connector.reassign_incident(incident_id, **kwargs)
            return json_response(
                {
                    "data": {
                        "incident": self._incident_to_dict(incident),
                        "message": "Incident reassigned",
                    }
                }
            )
        except (ConnectionError, TimeoutError, OSError, ValueError, TypeError, RuntimeError) as e:
            logger.error("Failed to reassign incident: %s", e)
            return error_response("Internal server error", 500)

    async def _handle_merge_incidents(
        self, request: Any, tenant_id: str, incident_id: str
    ) -> HandlerResult:
        ok, err = _validate_pagerduty_id(incident_id, "incident_id")
        if not ok:
            return error_response(err or "Invalid incident_id", 400)

        connector = await get_pagerduty_connector(tenant_id)
        if connector is None:
            return error_response("PagerDuty connector unavailable", 503)

        body = await self._get_json_body(request)
        source_ids, err = _validate_id_list(
            body.get("source_incident_ids"),
            "source_incident_ids",
            max_items=MAX_SOURCE_INCIDENT_IDS,
        )
        if err:
            return error_response(err, 400)
        if not source_ids:
            return error_response("source_incident_ids is required", 400)

        try:
            incident = await connector.merge_incidents(incident_id, source_ids)
            return json_response(
                {
                    "data": {
                        "incident": self._incident_to_dict(incident),
                        "message": f"Merged {len(source_ids)} incidents",
                    }
                }
            )
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            logger.error("Failed to merge incidents: %s", e)
            return error_response("Internal server error", 500)

    async def _handle_list_notes(
        self, request: Any, tenant_id: str, incident_id: str
    ) -> HandlerResult:
        ok, err = _validate_pagerduty_id(incident_id, "incident_id")
        if not ok:
            return error_response(err or "Invalid incident_id", 400)

        connector = await get_pagerduty_connector(tenant_id)
        if connector is None:
            return error_response("PagerDuty connector unavailable", 503)

        try:
            notes = await connector.list_notes(incident_id)
            payload = [self._note_to_dict(n) for n in notes]
            return json_response({"data": {"notes": payload, "count": len(payload)}})
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            logger.error("Failed to list notes: %s", e)
            return error_response("Internal server error", 500)

    async def _handle_add_note(
        self, request: Any, tenant_id: str, incident_id: str
    ) -> HandlerResult:
        ok, err = _validate_pagerduty_id(incident_id, "incident_id")
        if not ok:
            return error_response(err or "Invalid incident_id", 400)

        connector = await get_pagerduty_connector(tenant_id)
        if connector is None:
            return error_response("PagerDuty connector unavailable", 503)

        body = await self._get_json_body(request)
        content, err = _validate_string_field(
            body.get("content"), "content", required=True, max_length=MAX_NOTE_CONTENT_LENGTH
        )
        if err:
            return error_response(err, 400)

        try:
            note = await connector.add_note(incident_id, content)
            return json_response({"note": self._note_to_dict(note)}, status=201)
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            logger.error("Failed to add note: %s", e)
            return error_response("Internal server error", 500)

    async def _handle_get_oncall(self, request: Any, tenant_id: str) -> HandlerResult:
        connector = await get_pagerduty_connector(tenant_id)
        if connector is None:
            return error_response("PagerDuty connector unavailable", 503)

        query = self._get_query_params(request)
        schedule_ids = _split_csv(query.get("schedule_ids")) or None

        try:
            schedules = await connector.get_on_call(schedule_ids=schedule_ids)
            payload = [self._oncall_to_dict(s) for s in schedules]
            return json_response({"data": {"oncall": payload, "count": len(payload)}})
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            logger.error("Failed to get on-call: %s", e)
            return error_response("Internal server error", 500)

    async def _handle_get_oncall_for_service(
        self, request: Any, tenant_id: str, service_id: str
    ) -> HandlerResult:
        ok, err = _validate_pagerduty_id(service_id, "service_id")
        if not ok:
            return error_response(err or "Invalid service_id", 400)

        connector = await get_pagerduty_connector(tenant_id)
        if connector is None:
            return error_response("PagerDuty connector unavailable", 503)

        try:
            schedules = await connector.get_current_on_call_for_service(service_id)
            payload = [self._oncall_to_dict(s) for s in schedules]
            return json_response(
                {"data": {"service_id": service_id, "oncall": payload, "count": len(payload)}}
            )
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            logger.error("Failed to get on-call for service: %s", e)
            return error_response("Internal server error", 500)

    async def _handle_list_services(self, request: Any, tenant_id: str) -> HandlerResult:
        connector = await get_pagerduty_connector(tenant_id)
        if connector is None:
            return error_response("PagerDuty connector unavailable", 503)

        query = self._get_query_params(request)
        limit = safe_query_int(query, "limit", default=25, min_val=1, max_val=100)
        offset = safe_query_int(query, "offset", default=0, min_val=0, max_val=10000)

        try:
            result = await connector.list_services(limit=limit, offset=offset)
            if isinstance(result, tuple) and len(result) == 2:
                services, has_more = result
            else:
                services, has_more = result, False
            payload = [self._service_to_dict(s) for s in services]
            return json_response(
                {"data": {"services": payload, "count": len(payload), "has_more": has_more}}
            )
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            logger.error("Failed to list services: %s", e)
            return error_response("Internal server error", 500)

    async def _handle_get_service(
        self, request: Any, tenant_id: str, service_id: str
    ) -> HandlerResult:
        ok, err = _validate_pagerduty_id(service_id, "service_id")
        if not ok:
            return error_response(err or "Invalid service_id", 400)

        connector = await get_pagerduty_connector(tenant_id)
        if connector is None:
            return error_response("PagerDuty connector unavailable", 503)

        try:
            service = await connector.get_service(service_id)
            return json_response({"data": {"service": self._service_to_dict(service)}})
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            logger.error("Failed to get service: %s", e)
            return error_response("Internal server error", 500)

    async def _handle_pagerduty_webhook(self, request: Any, tenant_id: str) -> HandlerResult:
        raw_body = await self._get_raw_body(request)
        payload = await self._get_json_body(request)
        event_type = None

        connector = await get_pagerduty_connector(tenant_id)
        if connector is not None:
            try:
                signature = self._get_header(request, "X-PagerDuty-Signature")
                if signature is not None:
                    connector.verify_webhook_signature(signature, raw_body)
            except (ValueError, AttributeError, RuntimeError):
                logger.debug("Webhook signature verification failed", exc_info=True)

            try:
                parsed = connector.parse_webhook(payload)
                if parsed is not None:
                    event_type = getattr(parsed, "event_type", None)
                if hasattr(parsed, "to_dict"):
                    payload = parsed.to_dict()
            except (ValueError, KeyError, AttributeError):
                logger.debug("Webhook parse failed", exc_info=True)

        if event_type is None:
            event_type = (
                payload.get("event", {}).get("event_type") if isinstance(payload, dict) else None
            )

        await self._emit_connector_event(
            event_type or "pagerduty.webhook",
            tenant_id=tenant_id,
            data=payload if isinstance(payload, dict) else {},
        )

        return json_response(
            {
                "data": {
                    "received": True,
                    "event_type": event_type,
                }
            }
        )

    def _incident_to_dict(self, incident: Any) -> dict[str, Any]:
        return {
            "id": getattr(incident, "id", None),
            "title": getattr(incident, "title", None),
            "status": _value_or_none(getattr(incident, "status", None)),
            "urgency": _value_or_none(getattr(incident, "urgency", None)),
            "service_id": getattr(incident, "service_id", None),
            "service_name": getattr(incident, "service_name", None),
            "incident_number": getattr(incident, "incident_number", None),
            "created_at": self._isoformat(getattr(incident, "created_at", None)),
            "html_url": getattr(incident, "html_url", None),
            "description": getattr(incident, "description", None),
            "assignees": getattr(incident, "assignees", None),
            "priority": _value_or_none(getattr(incident, "priority", None)),
        }

    def _note_to_dict(self, note: Any) -> dict[str, Any]:
        user = getattr(note, "user", None)
        user_dict = None
        if user is not None:
            user_dict = {
                "id": getattr(user, "id", None),
                "name": getattr(user, "name", None),
                "email": getattr(user, "email", None),
            }
        return {
            "id": getattr(note, "id", None),
            "content": getattr(note, "content", None),
            "created_at": self._isoformat(getattr(note, "created_at", None)),
            "user": user_dict,
        }

    def _oncall_to_dict(self, sched: Any) -> dict[str, Any]:
        user = getattr(sched, "user", None)
        user_dict = None
        if user is not None:
            user_dict = {
                "id": getattr(user, "id", None),
                "name": getattr(user, "name", None),
                "email": getattr(user, "email", None),
            }
        return {
            "schedule_id": getattr(sched, "schedule_id", None),
            "schedule_name": getattr(sched, "schedule_name", None),
            "user": user_dict,
            "start": self._isoformat(getattr(sched, "start", None)),
            "end": self._isoformat(getattr(sched, "end", None)),
            "escalation_level": getattr(sched, "escalation_level", None),
        }

    def _service_to_dict(self, service: Any) -> dict[str, Any]:
        return {
            "id": getattr(service, "id", None),
            "name": getattr(service, "name", None),
            "description": getattr(service, "description", None),
            "status": _value_or_none(getattr(service, "status", None)),
            "html_url": getattr(service, "html_url", None),
            "escalation_policy_id": getattr(service, "escalation_policy_id", None),
            "created_at": self._isoformat(getattr(service, "created_at", None)),
        }

    def _isoformat(self, value: Any) -> str | None:
        if isinstance(value, datetime):
            return value.isoformat()
        return None


__all__ = [
    "DevOpsHandler",
    "DevOpsCircuitBreaker",
    "create_devops_handler",
    "get_pagerduty_connector",
    "get_devops_circuit_breaker",
    "get_devops_circuit_breaker_status",
    "_clear_devops_components",
    "_connector_instances",
    "_active_contexts",
    "_validate_pagerduty_id",
    "_validate_urgency",
    "_validate_string_field",
    "_validate_id_list",
]
