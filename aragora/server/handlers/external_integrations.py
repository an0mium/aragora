"""
External Integrations API Handler.

Provides REST API endpoints for managing external automation integrations:
- Zapier: REST Hook triggers and actions
- Make (Integromat): Webhook modules and actions
- n8n: Custom nodes and webhooks

Endpoints:
- POST   /api/integrations/zapier/apps              - Create Zapier app
- GET    /api/integrations/zapier/apps              - List Zapier apps
- DELETE /api/integrations/zapier/apps/:id          - Delete Zapier app
- POST   /api/integrations/zapier/triggers          - Subscribe to trigger
- DELETE /api/integrations/zapier/triggers/:id      - Unsubscribe trigger
- GET    /api/integrations/zapier/triggers          - List trigger types

- POST   /api/integrations/make/connections         - Create Make connection
- GET    /api/integrations/make/connections         - List Make connections
- DELETE /api/integrations/make/connections/:id     - Delete Make connection
- POST   /api/integrations/make/webhooks            - Register webhook
- DELETE /api/integrations/make/webhooks/:id        - Unregister webhook
- GET    /api/integrations/make/modules             - List available modules

- POST   /api/integrations/n8n/credentials          - Create n8n credential
- GET    /api/integrations/n8n/credentials          - List n8n credentials
- DELETE /api/integrations/n8n/credentials/:id      - Delete n8n credential
- POST   /api/integrations/n8n/webhooks             - Register webhook
- DELETE /api/integrations/n8n/webhooks/:id         - Unregister webhook
- GET    /api/integrations/n8n/nodes                - Get node definitions
"""

import logging
import time
from typing import Any, Dict, List, Optional

from aragora.integrations.zapier import ZapierIntegration, get_zapier_integration
from aragora.integrations.make import MakeIntegration, get_make_integration
from aragora.integrations.n8n import N8nIntegration, get_n8n_integration
from aragora.server.handlers.base import (
    SAFE_ID_PATTERN,
    BaseHandler,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.responses import HandlerResult

logger = logging.getLogger(__name__)

# RBAC imports (optional - graceful degradation if not available)
try:
    from aragora.rbac import (
        AuthorizationContext,
        check_permission,
        PermissionDeniedError,
    )
    from aragora.billing.auth import extract_user_from_request
    RBAC_AVAILABLE = True
except ImportError:
    RBAC_AVAILABLE = False

# Metrics imports (optional)
try:
    from aragora.observability.metrics import record_rbac_check
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

    def record_rbac_check(*args, **kwargs):
        pass


# =============================================================================
# External Integrations Handler
# =============================================================================


class ExternalIntegrationsHandler(BaseHandler):
    """Handler for external integrations API endpoints."""

    # Routes this handler responds to
    routes = [
        # Zapier
        "POST /api/integrations/zapier/apps",
        "GET /api/integrations/zapier/apps",
        "DELETE /api/integrations/zapier/apps/:id",
        "POST /api/integrations/zapier/triggers",
        "DELETE /api/integrations/zapier/triggers/:id",
        "GET /api/integrations/zapier/triggers",
        # Make
        "POST /api/integrations/make/connections",
        "GET /api/integrations/make/connections",
        "DELETE /api/integrations/make/connections/:id",
        "POST /api/integrations/make/webhooks",
        "DELETE /api/integrations/make/webhooks/:id",
        "GET /api/integrations/make/modules",
        # n8n
        "POST /api/integrations/n8n/credentials",
        "GET /api/integrations/n8n/credentials",
        "DELETE /api/integrations/n8n/credentials/:id",
        "POST /api/integrations/n8n/webhooks",
        "DELETE /api/integrations/n8n/webhooks/:id",
        "GET /api/integrations/n8n/nodes",
        # Test endpoints
        "POST /api/integrations/:platform/test",
    ]

    ROUTES = [
        "/api/integrations/zapier/apps",
        "/api/integrations/zapier/triggers",
        "/api/integrations/make/connections",
        "/api/integrations/make/webhooks",
        "/api/integrations/make/modules",
        "/api/integrations/n8n/credentials",
        "/api/integrations/n8n/webhooks",
        "/api/integrations/n8n/nodes",
    ]

    @staticmethod
    def can_handle(path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path.startswith("/api/integrations/")

    def __init__(self, server_context: dict):
        """Initialize with server context."""
        super().__init__(server_context)
        self._zapier: Optional[ZapierIntegration] = None
        self._make: Optional[MakeIntegration] = None
        self._n8n: Optional[N8nIntegration] = None

    # =========================================================================
    # RBAC Helper Methods
    # =========================================================================

    def _get_auth_context(self, handler: Any) -> Optional[AuthorizationContext]:
        """Extract authorization context from the request."""
        if not RBAC_AVAILABLE:
            return None

        try:
            # Try to get user info from request
            user_info = extract_user_from_request(handler)
            if not user_info:
                return None

            return AuthorizationContext(
                user_id=user_info.get("user_id", "anonymous"),
                roles=user_info.get("roles", []),
                org_id=user_info.get("org_id") or user_info.get("tenant_id"),
            )
        except Exception as e:
            logger.debug(f"Could not extract auth context: {e}")
            return None

    def _check_permission(
        self, handler: Any, permission_key: str, resource_id: Optional[str] = None
    ) -> Optional[HandlerResult]:
        """
        Check if current user has permission. Returns error response if denied.

        Args:
            handler: The HTTP handler
            permission_key: Permission like "integrations.read" or "integrations.create"
            resource_id: Optional resource ID for resource-specific permissions

        Returns:
            None if allowed, error HandlerResult if denied
        """
        if not RBAC_AVAILABLE:
            logger.debug(f"RBAC not available, allowing {permission_key}")
            return None

        context = self._get_auth_context(handler)
        if context is None:
            # No auth context means RBAC not configured for this request
            return None

        try:
            decision = check_permission(context, permission_key, resource_id)
            if not decision.allowed:
                logger.warning(
                    f"Permission denied: {permission_key} for user {context.user_id}: {decision.reason}"
                )
                record_rbac_check(permission_key, allowed=False, handler="ExternalIntegrationsHandler")
                return error_response(f"Permission denied: {decision.reason}", 403)
            record_rbac_check(permission_key, allowed=True)
        except PermissionDeniedError as e:
            logger.warning(f"Permission denied: {permission_key} for user {context.user_id}: {e}")
            record_rbac_check(permission_key, allowed=False, handler="ExternalIntegrationsHandler")
            return error_response(f"Permission denied: {str(e)}", 403)

        return None

    def _get_zapier(self) -> ZapierIntegration:
        """Get or create Zapier integration instance."""
        if self._zapier is None:
            self._zapier = get_zapier_integration()
        return self._zapier

    def _get_make(self) -> MakeIntegration:
        """Get or create Make integration instance."""
        if self._make is None:
            self._make = get_make_integration()
        return self._make

    def _get_n8n(self) -> N8nIntegration:
        """Get or create n8n integration instance."""
        if self._n8n is None:
            self._n8n = get_n8n_integration()
        return self._n8n

    # =========================================================================
    # GET Handlers
    # =========================================================================

    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle GET requests for external integrations endpoints."""

        # Zapier endpoints
        if path == "/api/integrations/zapier/apps":
            return self._handle_list_zapier_apps(query_params, handler)
        if path == "/api/integrations/zapier/triggers":
            return self._handle_list_zapier_trigger_types()

        # Make endpoints
        if path == "/api/integrations/make/connections":
            return self._handle_list_make_connections(query_params, handler)
        if path == "/api/integrations/make/modules":
            return self._handle_list_make_modules()

        # n8n endpoints
        if path == "/api/integrations/n8n/credentials":
            return self._handle_list_n8n_credentials(query_params, handler)
        if path == "/api/integrations/n8n/nodes":
            return self._handle_get_n8n_nodes()

        return None

    # =========================================================================
    # POST Handlers
    # =========================================================================

    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests for external integrations endpoints."""

        # Zapier endpoints
        if path == "/api/integrations/zapier/apps":
            body, err = self.read_json_body_validated(handler)
            if err:
                return err
            return self._handle_create_zapier_app(body, handler)

        if path == "/api/integrations/zapier/triggers":
            body, err = self.read_json_body_validated(handler)
            if err:
                return err
            return self._handle_subscribe_zapier_trigger(body, handler)

        # Make endpoints
        if path == "/api/integrations/make/connections":
            body, err = self.read_json_body_validated(handler)
            if err:
                return err
            return self._handle_create_make_connection(body, handler)

        if path == "/api/integrations/make/webhooks":
            body, err = self.read_json_body_validated(handler)
            if err:
                return err
            return self._handle_register_make_webhook(body, handler)

        # n8n endpoints
        if path == "/api/integrations/n8n/credentials":
            body, err = self.read_json_body_validated(handler)
            if err:
                return err
            return self._handle_create_n8n_credential(body, handler)

        if path == "/api/integrations/n8n/webhooks":
            body, err = self.read_json_body_validated(handler)
            if err:
                return err
            return self._handle_register_n8n_webhook(body, handler)

        # Test endpoints
        if path.endswith("/test"):
            parts = path.split("/")
            if len(parts) >= 4:
                platform = parts[3]
                return self._handle_test_integration(platform, handler)

        return None

    # =========================================================================
    # DELETE Handlers
    # =========================================================================

    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle DELETE requests for external integrations endpoints."""

        # Zapier app deletion
        if path.startswith("/api/integrations/zapier/apps/"):
            app_id, err = self.extract_path_param(path, 4, "app_id", SAFE_ID_PATTERN)
            if err:
                return err
            return self._handle_delete_zapier_app(app_id, handler)

        # Zapier trigger unsubscribe
        if path.startswith("/api/integrations/zapier/triggers/"):
            parts = path.split("/")
            if len(parts) >= 5:
                trigger_id = parts[4]
                app_id = query_params.get("app_id", [""])[0]
                return self._handle_unsubscribe_zapier_trigger(app_id, trigger_id, handler)

        # Make connection deletion
        if path.startswith("/api/integrations/make/connections/"):
            conn_id, err = self.extract_path_param(path, 4, "conn_id", SAFE_ID_PATTERN)
            if err:
                return err
            return self._handle_delete_make_connection(conn_id, handler)

        # Make webhook unregister
        if path.startswith("/api/integrations/make/webhooks/"):
            parts = path.split("/")
            if len(parts) >= 5:
                webhook_id = parts[4]
                conn_id = query_params.get("connection_id", [""])[0]
                return self._handle_unregister_make_webhook(conn_id, webhook_id, handler)

        # n8n credential deletion
        if path.startswith("/api/integrations/n8n/credentials/"):
            cred_id, err = self.extract_path_param(path, 4, "cred_id", SAFE_ID_PATTERN)
            if err:
                return err
            return self._handle_delete_n8n_credential(cred_id, handler)

        # n8n webhook unregister
        if path.startswith("/api/integrations/n8n/webhooks/"):
            parts = path.split("/")
            if len(parts) >= 5:
                webhook_id = parts[4]
                cred_id = query_params.get("credential_id", [""])[0]
                return self._handle_unregister_n8n_webhook(cred_id, webhook_id, handler)

        return None

    # =========================================================================
    # Zapier Handlers
    # =========================================================================

    def _handle_list_zapier_apps(
        self, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle GET /api/integrations/zapier/apps - list Zapier apps."""
        # Check RBAC permission
        perm_error = self._check_permission(handler, "integrations.read")
        if perm_error:
            return perm_error

        user = self.get_current_user(handler)
        workspace_id = query_params.get("workspace_id", [None])[0]

        zapier = self._get_zapier()
        apps = zapier.list_apps(workspace_id)

        return json_response({
            "apps": [
                {
                    "id": app.id,
                    "workspace_id": app.workspace_id,
                    "created_at": app.created_at,
                    "active": app.active,
                    "trigger_count": app.trigger_count,
                    "action_count": app.action_count,
                }
                for app in apps
            ],
            "count": len(apps),
        })

    def _handle_list_zapier_trigger_types(self) -> HandlerResult:
        """Handle GET /api/integrations/zapier/triggers - list trigger types."""
        zapier = self._get_zapier()

        return json_response({
            "triggers": zapier.TRIGGER_TYPES,
            "actions": zapier.ACTION_TYPES,
        })

    def _handle_create_zapier_app(self, body: dict, handler: Any) -> HandlerResult:
        """Handle POST /api/integrations/zapier/apps - create Zapier app."""
        # Check RBAC permission - creating integrations exposes API keys
        perm_error = self._check_permission(handler, "integrations.create")
        if perm_error:
            return perm_error

        workspace_id = body.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400)

        zapier = self._get_zapier()
        app = zapier.create_app(workspace_id)

        return json_response(
            {
                "app": {
                    "id": app.id,
                    "workspace_id": app.workspace_id,
                    "api_key": app.api_key,
                    "api_secret": app.api_secret,
                    "created_at": app.created_at,
                },
                "message": "Zapier app created. Save the api_key and api_secret - they won't be shown again.",
            },
            status=201,
        )

    def _handle_delete_zapier_app(self, app_id: str, handler: Any) -> HandlerResult:
        """Handle DELETE /api/integrations/zapier/apps/:id - delete Zapier app."""
        # Check RBAC permission
        perm_error = self._check_permission(handler, "integrations.delete", app_id)
        if perm_error:
            return perm_error

        zapier = self._get_zapier()

        if zapier.delete_app(app_id):
            return json_response({"deleted": True, "app_id": app_id})
        else:
            return error_response(f"Zapier app not found: {app_id}", 404)

    def _handle_subscribe_zapier_trigger(
        self, body: dict, handler: Any
    ) -> HandlerResult:
        """Handle POST /api/integrations/zapier/triggers - subscribe to trigger."""
        app_id = body.get("app_id")
        trigger_type = body.get("trigger_type")
        webhook_url = body.get("webhook_url")

        if not app_id:
            return error_response("app_id is required", 400)
        if not trigger_type:
            return error_response("trigger_type is required", 400)
        if not webhook_url:
            return error_response("webhook_url is required", 400)

        zapier = self._get_zapier()
        trigger = zapier.subscribe_trigger(
            app_id=app_id,
            trigger_type=trigger_type,
            webhook_url=webhook_url,
            workspace_id=body.get("workspace_id"),
            debate_tags=body.get("debate_tags"),
            min_confidence=body.get("min_confidence"),
        )

        if trigger:
            return json_response(
                {
                    "trigger": {
                        "id": trigger.id,
                        "trigger_type": trigger.trigger_type,
                        "webhook_url": trigger.webhook_url,
                        "created_at": trigger.created_at,
                    }
                },
                status=201,
            )
        else:
            return error_response("Failed to subscribe trigger", 400)

    def _handle_unsubscribe_zapier_trigger(
        self, app_id: str, trigger_id: str, handler: Any
    ) -> HandlerResult:
        """Handle DELETE /api/integrations/zapier/triggers/:id - unsubscribe."""
        if not app_id:
            return error_response("app_id query parameter is required", 400)

        zapier = self._get_zapier()

        if zapier.unsubscribe_trigger(app_id, trigger_id):
            return json_response({"deleted": True, "trigger_id": trigger_id})
        else:
            return error_response(f"Trigger not found: {trigger_id}", 404)

    # =========================================================================
    # Make Handlers
    # =========================================================================

    def _handle_list_make_connections(
        self, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle GET /api/integrations/make/connections - list connections."""
        # Check RBAC permission
        perm_error = self._check_permission(handler, "integrations.read")
        if perm_error:
            return perm_error

        workspace_id = query_params.get("workspace_id", [None])[0]

        make = self._get_make()
        connections = make.list_connections(workspace_id)

        return json_response({
            "connections": [
                {
                    "id": conn.id,
                    "workspace_id": conn.workspace_id,
                    "created_at": conn.created_at,
                    "active": conn.active,
                    "total_operations": conn.total_operations,
                    "webhooks_count": len(conn.webhooks),
                }
                for conn in connections
            ],
            "count": len(connections),
        })

    def _handle_list_make_modules(self) -> HandlerResult:
        """Handle GET /api/integrations/make/modules - list available modules."""
        make = self._get_make()

        return json_response({
            "modules": make.MODULE_TYPES,
        })

    def _handle_create_make_connection(
        self, body: dict, handler: Any
    ) -> HandlerResult:
        """Handle POST /api/integrations/make/connections - create connection."""
        # Check RBAC permission - creating integrations exposes API keys
        perm_error = self._check_permission(handler, "integrations.create")
        if perm_error:
            return perm_error

        workspace_id = body.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400)

        make = self._get_make()
        connection = make.create_connection(workspace_id)

        return json_response(
            {
                "connection": {
                    "id": connection.id,
                    "workspace_id": connection.workspace_id,
                    "api_key": connection.api_key,
                    "created_at": connection.created_at,
                },
                "message": "Make connection created. Save the api_key - it won't be shown again.",
            },
            status=201,
        )

    def _handle_delete_make_connection(
        self, conn_id: str, handler: Any
    ) -> HandlerResult:
        """Handle DELETE /api/integrations/make/connections/:id - delete."""
        # Check RBAC permission
        perm_error = self._check_permission(handler, "integrations.delete", conn_id)
        if perm_error:
            return perm_error

        make = self._get_make()

        if make.delete_connection(conn_id):
            return json_response({"deleted": True, "connection_id": conn_id})
        else:
            return error_response(f"Make connection not found: {conn_id}", 404)

    def _handle_register_make_webhook(
        self, body: dict, handler: Any
    ) -> HandlerResult:
        """Handle POST /api/integrations/make/webhooks - register webhook."""
        conn_id = body.get("connection_id")
        module_type = body.get("module_type")
        webhook_url = body.get("webhook_url")

        if not conn_id:
            return error_response("connection_id is required", 400)
        if not module_type:
            return error_response("module_type is required", 400)
        if not webhook_url:
            return error_response("webhook_url is required", 400)

        make = self._get_make()
        webhook = make.register_webhook(
            conn_id=conn_id,
            module_type=module_type,
            webhook_url=webhook_url,
            workspace_id=body.get("workspace_id"),
            event_filter=body.get("event_filter"),
        )

        if webhook:
            return json_response(
                {
                    "webhook": {
                        "id": webhook.id,
                        "module_type": webhook.module_type,
                        "webhook_url": webhook.webhook_url,
                        "created_at": webhook.created_at,
                    }
                },
                status=201,
            )
        else:
            return error_response("Failed to register webhook", 400)

    def _handle_unregister_make_webhook(
        self, conn_id: str, webhook_id: str, handler: Any
    ) -> HandlerResult:
        """Handle DELETE /api/integrations/make/webhooks/:id - unregister."""
        if not conn_id:
            return error_response("connection_id query parameter is required", 400)

        make = self._get_make()

        if make.unregister_webhook(conn_id, webhook_id):
            return json_response({"deleted": True, "webhook_id": webhook_id})
        else:
            return error_response(f"Webhook not found: {webhook_id}", 404)

    # =========================================================================
    # n8n Handlers
    # =========================================================================

    def _handle_list_n8n_credentials(
        self, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle GET /api/integrations/n8n/credentials - list credentials."""
        # Check RBAC permission
        perm_error = self._check_permission(handler, "integrations.read")
        if perm_error:
            return perm_error

        workspace_id = query_params.get("workspace_id", [None])[0]

        n8n = self._get_n8n()
        credentials = n8n.list_credentials(workspace_id)

        return json_response({
            "credentials": [
                {
                    "id": cred.id,
                    "workspace_id": cred.workspace_id,
                    "api_url": cred.api_url,
                    "created_at": cred.created_at,
                    "active": cred.active,
                    "operation_count": cred.operation_count,
                    "webhooks_count": len(cred.webhooks),
                }
                for cred in credentials
            ],
            "count": len(credentials),
        })

    def _handle_get_n8n_nodes(self) -> HandlerResult:
        """Handle GET /api/integrations/n8n/nodes - get node definitions."""
        n8n = self._get_n8n()

        return json_response({
            "node": n8n.get_node_definition(),
            "trigger": n8n.get_trigger_node_definition(),
            "credential": n8n.get_credential_definition(),
            "events": n8n.EVENT_TYPES,
        })

    def _handle_create_n8n_credential(
        self, body: dict, handler: Any
    ) -> HandlerResult:
        """Handle POST /api/integrations/n8n/credentials - create credential."""
        # Check RBAC permission - creating credentials exposes API keys
        perm_error = self._check_permission(handler, "integrations.create")
        if perm_error:
            return perm_error

        workspace_id = body.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400)

        n8n = self._get_n8n()
        credential = n8n.create_credential(
            workspace_id=workspace_id,
            api_url=body.get("api_url"),
        )

        return json_response(
            {
                "credential": {
                    "id": credential.id,
                    "workspace_id": credential.workspace_id,
                    "api_key": credential.api_key,
                    "api_url": credential.api_url,
                    "created_at": credential.created_at,
                },
                "message": "n8n credential created. Save the api_key - it won't be shown again.",
            },
            status=201,
        )

    def _handle_delete_n8n_credential(
        self, cred_id: str, handler: Any
    ) -> HandlerResult:
        """Handle DELETE /api/integrations/n8n/credentials/:id - delete."""
        # Check RBAC permission
        perm_error = self._check_permission(handler, "integrations.delete", cred_id)
        if perm_error:
            return perm_error

        n8n = self._get_n8n()

        if n8n.delete_credential(cred_id):
            return json_response({"deleted": True, "credential_id": cred_id})
        else:
            return error_response(f"n8n credential not found: {cred_id}", 404)

    def _handle_register_n8n_webhook(
        self, body: dict, handler: Any
    ) -> HandlerResult:
        """Handle POST /api/integrations/n8n/webhooks - register webhook."""
        cred_id = body.get("credential_id")
        events = body.get("events", [])

        if not cred_id:
            return error_response("credential_id is required", 400)
        if not events:
            return error_response("events is required (list of event types)", 400)

        n8n = self._get_n8n()
        webhook = n8n.register_webhook(
            cred_id=cred_id,
            events=events,
            workflow_id=body.get("workflow_id"),
            node_id=body.get("node_id"),
            workspace_id=body.get("workspace_id"),
        )

        if webhook:
            return json_response(
                {
                    "webhook": {
                        "id": webhook.id,
                        "webhook_path": webhook.webhook_path,
                        "events": webhook.events,
                        "created_at": webhook.created_at,
                    }
                },
                status=201,
            )
        else:
            return error_response("Failed to register webhook", 400)

    def _handle_unregister_n8n_webhook(
        self, cred_id: str, webhook_id: str, handler: Any
    ) -> HandlerResult:
        """Handle DELETE /api/integrations/n8n/webhooks/:id - unregister."""
        if not cred_id:
            return error_response("credential_id query parameter is required", 400)

        n8n = self._get_n8n()

        if n8n.unregister_webhook(cred_id, webhook_id):
            return json_response({"deleted": True, "webhook_id": webhook_id})
        else:
            return error_response(f"Webhook not found: {webhook_id}", 404)

    # =========================================================================
    # Test Handler
    # =========================================================================

    def _handle_test_integration(
        self, platform: str, handler: Any
    ) -> HandlerResult:
        """Handle POST /api/integrations/:platform/test - test integration."""
        if platform == "zapier":
            zapier = self._get_zapier()
            return json_response({
                "platform": "zapier",
                "status": "ok",
                "apps_count": len(zapier._apps),
                "trigger_types": list(zapier.TRIGGER_TYPES.keys()),
                "action_types": list(zapier.ACTION_TYPES.keys()),
            })

        elif platform == "make":
            make = self._get_make()
            return json_response({
                "platform": "make",
                "status": "ok",
                "connections_count": len(make._connections),
                "module_types": list(make.MODULE_TYPES.keys()),
            })

        elif platform == "n8n":
            n8n = self._get_n8n()
            return json_response({
                "platform": "n8n",
                "status": "ok",
                "credentials_count": len(n8n._credentials),
                "event_types": list(n8n.EVENT_TYPES.keys()),
            })

        else:
            return error_response(f"Unknown platform: {platform}", 400)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ExternalIntegrationsHandler",
]
