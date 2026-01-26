"""
Webhook API Handler.

Provides REST API endpoints for webhook management:
- POST   /api/webhooks              - Register a new webhook
- GET    /api/webhooks              - List registered webhooks
- GET    /api/webhooks/:id          - Get specific webhook
- DELETE /api/webhooks/:id          - Delete a webhook
- POST   /api/webhooks/:id/test     - Send a test event to webhook
- GET    /api/webhooks/events       - List available event types

Webhooks receive HTTP POST requests when subscribed events occur.
All webhook payloads include HMAC-SHA256 signatures for verification.

Webhook configurations are persisted to SQLite (default) or Redis+SQLite for
multi-instance deployments. This ensures webhooks survive server restarts.
"""

import hashlib
import hmac
import logging
import time
from typing import Any, Optional

from aragora.server.handlers.base import (
    SAFE_ID_PATTERN,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.responses import HandlerResult
from aragora.server.handlers.secure import SecureHandler
from aragora.server.handlers.utils.url_security import validate_webhook_url

# RBAC imports - graceful fallback if not available
try:
    from aragora.rbac import AuthorizationContext, check_permission

    RBAC_AVAILABLE = True
except ImportError:
    RBAC_AVAILABLE = False
    AuthorizationContext = None  # type: ignore[misc]
    check_permission = None

# Import durable storage from storage module
from aragora.storage.webhook_config_store import (
    WEBHOOK_EVENTS,
    WebhookConfig,
    WebhookConfigStoreBackend,
    get_webhook_config_store,
)

# Unified audit logging
try:
    from aragora.audit.unified import audit_data

    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False
    audit_data = None

logger = logging.getLogger(__name__)

# Rate limits for webhook operations
WEBHOOK_REGISTER_RPM = 10  # Max 10 webhook registrations per minute
WEBHOOK_TEST_RPM = 5  # Max 5 test deliveries per minute
WEBHOOK_LIST_RPM = 60  # Max 60 list operations per minute


# Backward compatibility alias - the old WebhookStore interface is now provided
# by WebhookConfigStoreBackend from aragora.storage.webhook_config_store
WebhookStore = WebhookConfigStoreBackend


def get_webhook_store() -> WebhookConfigStoreBackend:
    """Get or create the webhook store.

    Returns a durable storage backend (SQLite by default, Redis+SQLite for
    multi-instance deployments). Webhooks survive server restarts.

    Configure via environment:
    - ARAGORA_WEBHOOK_CONFIG_STORE_BACKEND: "sqlite" (default), "redis", or "memory"
    - ARAGORA_DATA_DIR: Directory for SQLite database
    - ARAGORA_REDIS_URL: Redis connection URL (for redis backend)
    """
    return get_webhook_config_store()


# =============================================================================
# Webhook Signature Utilities
# =============================================================================


def generate_signature(payload: str, secret: str) -> str:
    """
    Generate HMAC-SHA256 signature for webhook payload.

    Args:
        payload: JSON string payload
        secret: Webhook secret key

    Returns:
        Hex-encoded signature with sha256= prefix
    """
    signature = hmac.new(
        secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"sha256={signature}"


def verify_signature(payload: str, signature: str, secret: str) -> bool:
    """
    Verify webhook signature.

    Args:
        payload: JSON string payload
        signature: Signature header value (sha256=...)
        secret: Webhook secret key

    Returns:
        True if signature is valid
    """
    expected = generate_signature(payload, secret)
    return hmac.compare_digest(signature, expected)


# =============================================================================
# Webhook Handler
# =============================================================================


class WebhookHandler(SecureHandler):
    """Handler for webhook management API endpoints.

    Extends SecureHandler for JWT-based authentication, RBAC permission
    enforcement, and security audit logging.
    """

    # Resource type for audit logging
    RESOURCE_TYPE = "webhook"

    # Routes this handler responds to
    routes = [
        "POST /api/webhooks",
        "GET /api/webhooks",
        "GET /api/webhooks/events",
        "GET /api/webhooks/slo/status",
        "POST /api/webhooks/slo/test",
        "GET /api/webhooks/:id",
        "DELETE /api/webhooks/:id",
        "PATCH /api/webhooks/:id",
        "POST /api/webhooks/:id/test",
        # Dead-letter queue endpoints
        "GET /api/webhooks/dead-letter",
        "GET /api/webhooks/dead-letter/:id",
        "POST /api/webhooks/dead-letter/:id/retry",
        "DELETE /api/webhooks/dead-letter/:id",
        "GET /api/webhooks/queue/stats",
    ]

    ROUTES = [
        "/api/v1/webhooks",
        "/api/v1/webhooks/events",
        "/api/v1/webhooks/slo/status",
        "/api/v1/webhooks/dead-letter",
        "/api/v1/webhooks/queue/stats",
    ]

    @staticmethod
    def can_handle(path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path.startswith("/api/v1/webhooks")

    def __init__(self, server_context: dict):
        """Initialize with server context."""
        super().__init__(server_context)  # type: ignore[arg-type]
        self._webhook_store: Optional[WebhookStore] = None

    def _get_webhook_store(self) -> WebhookStore:
        """Get or create webhook store instance."""
        if self._webhook_store is None:
            if "webhook_store" in self.ctx:
                self._webhook_store = self.ctx["webhook_store"]
            else:
                self._webhook_store = get_webhook_store()
                self.ctx["webhook_store"] = self._webhook_store
        return self._webhook_store

    def _get_auth_context(self, handler) -> Optional[AuthorizationContext]:
        """Build RBAC authorization context from request."""
        if not RBAC_AVAILABLE or AuthorizationContext is None:
            return None

        user = self.get_current_user(handler)
        if not user:
            return None

        # User context has user_id and potentially role info
        return AuthorizationContext(
            user_id=user.user_id,
            roles=set([user.role]) if hasattr(user, "role") and user.role else set(),
            org_id=getattr(user, "org_id", None),
        )

    def _check_rbac_permission(self, handler, permission_key: str) -> Optional[HandlerResult]:
        """
        Check RBAC permission.

        Returns None if allowed, or an error response if denied.
        """
        if not RBAC_AVAILABLE:
            return None

        rbac_ctx = self._get_auth_context(handler)
        if not rbac_ctx:
            # No auth context - rely on existing auth checks
            return None

        decision = check_permission(rbac_ctx, permission_key)
        if not decision.allowed:
            logger.warning(
                f"RBAC denied: user={rbac_ctx.user_id} permission={permission_key} "
                f"reason={decision.reason}"
            )
            return error_response(
                f"Permission denied: {decision.reason}",
                403,
            )

        return None

    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle GET requests for webhook endpoints."""
        # GET /api/webhooks/events - list available event types
        if path == "/api/v1/webhooks/events":
            return self._handle_list_events()

        # GET /api/webhooks/slo/status - get SLO webhook status
        if path == "/api/v1/webhooks/slo/status":
            return self._handle_slo_status()

        # GET /api/webhooks/dead-letter - list dead-letter queue
        if path == "/api/v1/webhooks/dead-letter":
            return self._handle_list_dead_letters(query_params, handler)

        # GET /api/webhooks/dead-letter/:id - get specific dead-letter delivery
        if path.startswith("/api/v1/webhooks/dead-letter/") and not path.endswith("/retry"):
            delivery_id, err = self.extract_path_param(path, 4, "delivery_id", SAFE_ID_PATTERN)
            if err:
                return err
            return self._handle_get_dead_letter(delivery_id, handler)

        # GET /api/webhooks/queue/stats - get queue statistics
        if path == "/api/v1/webhooks/queue/stats":
            return self._handle_queue_stats(handler)

        # GET /api/webhooks/:id
        if path.startswith("/api/v1/webhooks/") and path.count("/") == 4:
            webhook_id, err = self.extract_path_param(path, 3, "webhook_id", SAFE_ID_PATTERN)
            if err:
                return err
            return self._handle_get_webhook(webhook_id, handler)

        # GET /api/webhooks - list all webhooks
        if path == "/api/v1/webhooks":
            return self._handle_list_webhooks(query_params, handler)

        return None

    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests for webhook endpoints."""
        # POST /api/webhooks/slo/test - send test SLO violation notification
        if path == "/api/v1/webhooks/slo/test":
            return self._handle_slo_test()

        # POST /api/webhooks/dead-letter/:id/retry - retry dead-letter delivery
        if path.endswith("/retry") and "/dead-letter/" in path:
            delivery_id, err = self.extract_path_param(path, 4, "delivery_id", SAFE_ID_PATTERN)
            if err:
                return err
            return self._handle_retry_dead_letter(delivery_id, handler)

        # POST /api/v1/webhooks/:id/test
        if path.endswith("/test") and path.count("/") == 5:
            webhook_id, err = self.extract_path_param(path, 3, "webhook_id", SAFE_ID_PATTERN)
            if err:
                return err
            return self._handle_test_webhook(webhook_id, handler)

        # POST /api/webhooks - register new webhook
        if path == "/api/v1/webhooks":
            body, err = self.read_json_body_validated(handler)
            if err:
                return err
            return self._handle_register_webhook(body, handler)

        return None

    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle DELETE requests for webhook endpoints."""
        # DELETE /api/webhooks/dead-letter/:id - remove from dead-letter queue
        if "/dead-letter/" in path:
            delivery_id, err = self.extract_path_param(path, 4, "delivery_id", SAFE_ID_PATTERN)
            if err:
                return err
            return self._handle_delete_dead_letter(delivery_id, handler)

        # DELETE /api/webhooks/:id
        if path.startswith("/api/v1/webhooks/") and path.count("/") == 4:
            webhook_id, err = self.extract_path_param(path, 3, "webhook_id", SAFE_ID_PATTERN)
            if err:
                return err
            return self._handle_delete_webhook(webhook_id, handler)

        return None

    def handle_patch(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle PATCH requests for webhook endpoints."""
        # PATCH /api/webhooks/:id
        if path.startswith("/api/v1/webhooks/") and path.count("/") == 4:
            webhook_id, err = self.extract_path_param(path, 3, "webhook_id", SAFE_ID_PATTERN)
            if err:
                return err
            body, err = self.read_json_body_validated(handler)
            if err:
                return err
            return self._handle_update_webhook(webhook_id, body, handler)

        return None

    # =========================================================================
    # Handler Methods
    # =========================================================================

    def _handle_list_events(self) -> HandlerResult:
        """Handle GET /api/webhooks/events - list available event types."""
        events = sorted(WEBHOOK_EVENTS)
        return json_response(
            {
                "events": events,
                "count": len(events),
                "description": {
                    "debate_start": "Fired when a debate begins",
                    "debate_end": "Fired when a debate completes",
                    "consensus": "Fired when consensus is reached",
                    "round_start": "Fired at the start of each debate round",
                    "agent_message": "Fired when an agent sends a message",
                    "vote": "Fired when a vote is cast",
                    "insight_extracted": "Fired when a new insight is extracted",
                    "memory_stored": "Fired when memory is stored",
                    "memory_retrieved": "Fired when memory is retrieved",
                    "claim_verification_result": "Fired when a claim is verified",
                    "formal_verification_result": "Fired when formal verification completes",
                    "gauntlet_complete": "Fired when gauntlet stress-test completes",
                    "gauntlet_verdict": "Fired when gauntlet verdict is determined",
                    "receipt_ready": "Fired when a receipt is ready",
                    "receipt_exported": "Fired when a receipt is exported",
                    "graph_branch_created": "Fired when a graph debate branches",
                    "graph_branch_merged": "Fired when graph branches merge",
                    "genesis_evolution": "Fired when agent population evolves",
                    "breakpoint": "Fired when a human intervention breakpoint triggers",
                    "breakpoint_resolved": "Fired when a breakpoint is resolved",
                    "agent_elo_updated": "Fired when agent ELO rating is updated",
                    "knowledge_indexed": "Fired when knowledge is indexed",
                    "knowledge_queried": "Fired when knowledge is queried",
                    "mound_updated": "Fired when knowledge mound is updated",
                    "calibration_update": "Fired when calibration data is updated",
                    "evidence_found": "Fired when evidence is found",
                    "agent_calibration_changed": "Fired when agent calibration changes",
                    "agent_fallback_triggered": "Fired when agent fallback is triggered",
                    "explanation_ready": "Fired when an explanation is ready",
                },
            }
        )

    def _handle_list_webhooks(self, query_params: dict, handler: Any) -> HandlerResult:
        """Handle GET /api/webhooks - list all webhooks."""
        # Get optional user context for filtering
        user = self.get_current_user(handler)
        user_id = user.user_id if user else None

        active_only = query_params.get("active_only", ["false"])[0].lower() == "true"

        store = self._get_webhook_store()
        webhooks = store.list(user_id=user_id, active_only=active_only)

        return json_response(
            {
                "webhooks": [w.to_dict(include_secret=False) for w in webhooks],
                "count": len(webhooks),
            }
        )

    def _handle_get_webhook(self, webhook_id: str, handler: Any) -> HandlerResult:
        """Handle GET /api/webhooks/:id - get specific webhook."""
        store = self._get_webhook_store()
        webhook = store.get(webhook_id)

        if not webhook:
            return error_response(f"Webhook not found: {webhook_id}", 404)

        # Check ownership
        user = self.get_current_user(handler)
        if user and webhook.user_id and webhook.user_id != user.user_id:
            return error_response("Access denied", 403)

        return json_response({"webhook": webhook.to_dict(include_secret=False)})

    def _handle_register_webhook(self, body: dict, handler: Any) -> HandlerResult:
        """Handle POST /api/webhooks - register new webhook."""
        # RBAC permission check
        rbac_error = self._check_rbac_permission(handler, "webhooks.create")
        if rbac_error:
            return rbac_error

        url = body.get("url", "").strip()
        if not url:
            return error_response("URL is required", 400)

        # Validate URL format and check for SSRF
        is_valid, error_msg = validate_webhook_url(url, allow_localhost=False)
        if not is_valid:
            return error_response(f"Invalid webhook URL: {error_msg}", 400)

        events = body.get("events", [])
        if not events:
            return error_response("At least one event type is required", 400)

        # Validate events
        invalid_events = [e for e in events if e != "*" and e not in WEBHOOK_EVENTS]
        if invalid_events:
            return error_response(
                f"Invalid event types: {', '.join(invalid_events)}. "
                f"Use GET /api/webhooks/events for available types.",
                400,
            )

        # Get user context
        user = self.get_current_user(handler)
        user_id = user.user_id if user else None

        store = self._get_webhook_store()
        webhook = store.register(
            url=url,
            events=events,
            name=body.get("name"),
            description=body.get("description"),
            user_id=user_id,
        )

        # Audit log: webhook created
        if AUDIT_AVAILABLE and audit_data:
            audit_data(
                user_id=user_id or "anonymous",
                resource_type="webhook",
                resource_id=webhook.id,
                action="create",
                events=events,
            )

        # Return with secret (only on creation)
        return json_response(
            {
                "webhook": webhook.to_dict(include_secret=True),
                "message": "Webhook registered successfully. Save the secret - it won't be shown again.",
            },
            status=201,
        )

    def _handle_delete_webhook(self, webhook_id: str, handler: Any) -> HandlerResult:
        """Handle DELETE /api/webhooks/:id - delete webhook."""
        # RBAC permission check
        rbac_error = self._check_rbac_permission(handler, "webhooks.delete")
        if rbac_error:
            return rbac_error

        store = self._get_webhook_store()
        webhook = store.get(webhook_id)

        if not webhook:
            return error_response(f"Webhook not found: {webhook_id}", 404)

        # Check ownership
        user = self.get_current_user(handler)
        if user and webhook.user_id and webhook.user_id != user.user_id:
            return error_response("Access denied", 403)

        store.delete(webhook_id)

        # Audit log: webhook deleted
        if AUDIT_AVAILABLE and audit_data:
            audit_data(
                user_id=user.user_id if user else "anonymous",
                resource_type="webhook",
                resource_id=webhook_id,
                action="delete",
            )

        return json_response(
            {
                "deleted": True,
                "webhook_id": webhook_id,
            }
        )

    def _handle_update_webhook(self, webhook_id: str, body: dict, handler: Any) -> HandlerResult:
        """Handle PATCH /api/webhooks/:id - update webhook."""
        # RBAC permission check
        rbac_error = self._check_rbac_permission(handler, "webhooks.update")
        if rbac_error:
            return rbac_error

        store = self._get_webhook_store()
        webhook = store.get(webhook_id)

        if not webhook:
            return error_response(f"Webhook not found: {webhook_id}", 404)

        # Check ownership
        user = self.get_current_user(handler)
        if user and webhook.user_id and webhook.user_id != user.user_id:
            return error_response("Access denied", 403)

        # Validate URL if provided (SSRF check)
        new_url = body.get("url")
        if new_url:
            is_valid, error_msg = validate_webhook_url(new_url, allow_localhost=False)
            if not is_valid:
                return error_response(f"Invalid webhook URL: {error_msg}", 400)

        # Validate events if provided
        events = body.get("events")
        if events:
            invalid_events = [e for e in events if e != "*" and e not in WEBHOOK_EVENTS]
            if invalid_events:
                return error_response(f"Invalid event types: {', '.join(invalid_events)}", 400)

        updated = store.update(
            webhook_id=webhook_id,
            url=body.get("url"),
            events=events,
            active=body.get("active"),
            name=body.get("name"),
            description=body.get("description"),
        )

        # Audit log: webhook updated
        if AUDIT_AVAILABLE and audit_data:
            audit_data(
                user_id=user.user_id if user else "anonymous",
                resource_type="webhook",
                resource_id=webhook_id,
                action="update",
            )

        return json_response({"webhook": updated.to_dict(include_secret=False)})

    def _handle_test_webhook(self, webhook_id: str, handler: Any) -> HandlerResult:
        """Handle POST /api/webhooks/:id/test - send test event."""
        # RBAC permission check
        rbac_error = self._check_rbac_permission(handler, "webhooks.test")
        if rbac_error:
            return rbac_error

        store = self._get_webhook_store()
        webhook = store.get(webhook_id)

        if not webhook:
            return error_response(f"Webhook not found: {webhook_id}", 404)

        # Check ownership
        user = self.get_current_user(handler)
        if user and webhook.user_id and webhook.user_id != user.user_id:
            return error_response("Access denied", 403)

        # Import here to avoid circular dependency
        from aragora.events.dispatcher import dispatch_webhook

        # Create test payload
        test_event = {
            "event": "test",
            "webhook_id": webhook_id,
            "timestamp": time.time(),
            "data": {
                "message": "This is a test webhook delivery",
                "webhook_name": webhook.name or webhook.id,
            },
        }

        # Dispatch synchronously for testing
        success, status_code, error = dispatch_webhook(webhook, test_event)

        if success:
            return json_response(
                {
                    "success": True,
                    "status_code": status_code,
                    "message": "Test webhook delivered successfully",
                }
            )
        else:
            return json_response(
                {
                    "success": False,
                    "status_code": status_code,
                    "error": error,
                    "message": "Test webhook delivery failed",
                },
                status=502,
            )

    def _handle_slo_status(self) -> HandlerResult:
        """Handle GET /api/webhooks/slo/status - get SLO webhook status."""
        try:
            from aragora.observability.metrics.slo import (
                get_slo_webhook_status,
                get_violation_state,
            )

            webhook_status = get_slo_webhook_status()
            violation_state = get_violation_state()

            # enabled means initialized (callback is set)
            is_enabled = webhook_status.get("enabled", False)

            return json_response(
                {
                    "slo_webhooks": {
                        "enabled": is_enabled,
                        "initialized": is_enabled,  # Same as enabled
                        "config": webhook_status.get("config"),
                        "notifications_sent": webhook_status.get("notifications_sent", 0),
                        "recoveries_sent": webhook_status.get("recoveries_sent", 0),
                    },
                    "violation_state": violation_state,
                    "active_violations": sum(
                        1 for v in violation_state.values() if v.get("in_violation", False)
                    ),
                }
            )
        except ImportError:
            return json_response(
                {
                    "slo_webhooks": {
                        "enabled": False,
                        "initialized": False,
                        "error": "SLO module not available",
                    },
                    "violation_state": {},
                    "active_violations": 0,
                }
            )
        except Exception as e:
            logger.error(f"Error getting SLO webhook status: {e}")
            return error_response(f"Failed to get SLO status: {e}", 500)

    def _handle_slo_test(self) -> HandlerResult:
        """Handle POST /api/webhooks/slo/test - send test SLO violation notification."""
        try:
            from aragora.observability.metrics.slo import (
                get_slo_webhook_status,
                notify_slo_violation,
            )

            status = get_slo_webhook_status()
            if not status.get("enabled", False):
                return error_response(
                    "SLO webhooks are not enabled. Initialize with init_slo_webhooks() first.",
                    400,
                )

            # Send a test violation notification
            success = notify_slo_violation(
                operation="test_operation",
                percentile="p99",
                latency_ms=1500.0,
                threshold_ms=500.0,
                severity="minor",
                context={"test": True, "message": "This is a test SLO violation notification"},
                cooldown_seconds=0.0,  # Bypass cooldown for test
            )

            if success:
                return json_response(
                    {
                        "success": True,
                        "message": "Test SLO violation notification sent successfully",
                        "details": {
                            "operation": "test_operation",
                            "percentile": "p99",
                            "latency_ms": 1500.0,
                            "threshold_ms": 500.0,
                            "severity": "minor",
                        },
                    }
                )
            else:
                return json_response(
                    {
                        "success": False,
                        "message": "Test SLO violation notification was not sent (may be on cooldown or filtered)",
                    },
                    status=200,
                )

        except ImportError:
            return error_response("SLO module not available", 500)
        except Exception as e:
            logger.error(f"Error sending test SLO notification: {e}")
            return error_response(f"Failed to send test notification: {e}", 500)

    # =========================================================================
    # Dead-Letter Queue Handlers
    # =========================================================================

    def _handle_list_dead_letters(self, query_params: dict, handler: Any) -> HandlerResult:
        """Handle GET /api/webhooks/dead-letter - list dead-letter deliveries."""
        # RBAC permission check
        rbac_error = self._check_rbac_permission(handler, "webhooks.admin")
        if rbac_error:
            return rbac_error

        try:
            import asyncio
            from aragora.webhooks.retry_queue import get_retry_queue

            queue = get_retry_queue()
            limit = int(query_params.get("limit", ["100"])[0])
            limit = min(limit, 1000)  # Cap at 1000

            # Run async method
            loop = asyncio.new_event_loop()
            try:
                dead_letters = loop.run_until_complete(queue.get_dead_letters(limit))
            finally:
                loop.close()

            return json_response(
                {
                    "dead_letters": [d.to_dict() for d in dead_letters],
                    "count": len(dead_letters),
                    "limit": limit,
                }
            )

        except ImportError:
            return error_response("Webhook retry queue not available", 500)
        except Exception as e:
            logger.error(f"Error listing dead letters: {e}")
            return error_response(f"Failed to list dead letters: {e}", 500)

    def _handle_get_dead_letter(self, delivery_id: str, handler: Any) -> HandlerResult:
        """Handle GET /api/webhooks/dead-letter/:id - get specific dead-letter delivery."""
        # RBAC permission check
        rbac_error = self._check_rbac_permission(handler, "webhooks.admin")
        if rbac_error:
            return rbac_error

        try:
            import asyncio
            from aragora.webhooks.retry_queue import get_retry_queue, DeliveryStatus

            queue = get_retry_queue()

            # Run async method
            loop = asyncio.new_event_loop()
            try:
                delivery = loop.run_until_complete(queue.get_delivery(delivery_id))
            finally:
                loop.close()

            if not delivery:
                return error_response(f"Delivery not found: {delivery_id}", 404)

            if delivery.status != DeliveryStatus.DEAD_LETTER:
                return error_response(f"Delivery {delivery_id} is not in dead-letter queue", 400)

            return json_response({"delivery": delivery.to_dict()})

        except ImportError:
            return error_response("Webhook retry queue not available", 500)
        except Exception as e:
            logger.error(f"Error getting dead letter: {e}")
            return error_response(f"Failed to get dead letter: {e}", 500)

    def _handle_retry_dead_letter(self, delivery_id: str, handler: Any) -> HandlerResult:
        """Handle POST /api/webhooks/dead-letter/:id/retry - retry dead-letter delivery."""
        # RBAC permission check
        rbac_error = self._check_rbac_permission(handler, "webhooks.admin")
        if rbac_error:
            return rbac_error

        try:
            import asyncio
            from aragora.webhooks.retry_queue import get_retry_queue

            queue = get_retry_queue()

            # Run async method
            loop = asyncio.new_event_loop()
            try:
                success = loop.run_until_complete(queue.retry_dead_letter(delivery_id))
            finally:
                loop.close()

            if not success:
                return error_response(
                    f"Delivery {delivery_id} not found or not in dead-letter queue", 404
                )

            # Audit log
            user = self.get_current_user(handler)
            if AUDIT_AVAILABLE and audit_data:
                audit_data(
                    user_id=user.user_id if user else "anonymous",
                    resource_type="webhook_delivery",
                    resource_id=delivery_id,
                    action="retry",
                )

            return json_response(
                {
                    "success": True,
                    "delivery_id": delivery_id,
                    "message": "Delivery queued for retry",
                }
            )

        except ImportError:
            return error_response("Webhook retry queue not available", 500)
        except Exception as e:
            logger.error(f"Error retrying dead letter: {e}")
            return error_response(f"Failed to retry dead letter: {e}", 500)

    def _handle_delete_dead_letter(self, delivery_id: str, handler: Any) -> HandlerResult:
        """Handle DELETE /api/webhooks/dead-letter/:id - remove from dead-letter queue."""
        # RBAC permission check
        rbac_error = self._check_rbac_permission(handler, "webhooks.admin")
        if rbac_error:
            return rbac_error

        try:
            import asyncio
            from aragora.webhooks.retry_queue import get_retry_queue

            queue = get_retry_queue()

            # Run async method
            loop = asyncio.new_event_loop()
            try:
                success = loop.run_until_complete(queue.cancel_delivery(delivery_id))
            finally:
                loop.close()

            if not success:
                return error_response(f"Delivery not found: {delivery_id}", 404)

            # Audit log
            user = self.get_current_user(handler)
            if AUDIT_AVAILABLE and audit_data:
                audit_data(
                    user_id=user.user_id if user else "anonymous",
                    resource_type="webhook_delivery",
                    resource_id=delivery_id,
                    action="delete",
                )

            return json_response(
                {
                    "deleted": True,
                    "delivery_id": delivery_id,
                }
            )

        except ImportError:
            return error_response("Webhook retry queue not available", 500)
        except Exception as e:
            logger.error(f"Error deleting dead letter: {e}")
            return error_response(f"Failed to delete dead letter: {e}", 500)

    def _handle_queue_stats(self, handler: Any) -> HandlerResult:
        """Handle GET /api/webhooks/queue/stats - get queue statistics."""
        # RBAC permission check (read-only stats can use lower permission)
        rbac_error = self._check_rbac_permission(handler, "webhooks.read")
        if rbac_error:
            return rbac_error

        try:
            import asyncio
            from aragora.webhooks.retry_queue import get_retry_queue

            queue = get_retry_queue()

            # Run async method
            loop = asyncio.new_event_loop()
            try:
                stats = loop.run_until_complete(queue.get_stats())
            finally:
                loop.close()

            return json_response({"stats": stats})

        except ImportError:
            return error_response("Webhook retry queue not available", 500)
        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            return error_response(f"Failed to get queue stats: {e}", 500)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "WebhookHandler",
    "WebhookConfig",
    "WebhookStore",
    "get_webhook_store",
    "generate_signature",
    "verify_signature",
    "WEBHOOK_EVENTS",
]
