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
"""

import hashlib
import hmac
import logging
import secrets
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

from aragora.server.handlers.base import (
    SAFE_ID_PATTERN,
    BaseHandler,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.rate_limit import rate_limit
from aragora.server.handlers.utils.responses import HandlerResult
from aragora.server.handlers.utils.url_security import validate_webhook_url

logger = logging.getLogger(__name__)

# Rate limits for webhook operations
WEBHOOK_REGISTER_RPM = 10  # Max 10 webhook registrations per minute
WEBHOOK_TEST_RPM = 5       # Max 5 test deliveries per minute
WEBHOOK_LIST_RPM = 60      # Max 60 list operations per minute


# =============================================================================
# Webhook Data Models
# =============================================================================

# Events that can trigger webhooks (subset of StreamEventType)
WEBHOOK_EVENTS: Set[str] = {
    # Debate lifecycle
    "debate_start",
    "debate_end",
    "consensus",
    "round_start",
    # Agent events
    "agent_message",
    "vote",
    # Memory/learning
    "insight_extracted",
    "memory_stored",
    "memory_retrieved",
    # Verification
    "claim_verification_result",
    "formal_verification_result",
    # Gauntlet
    "gauntlet_complete",
    "gauntlet_verdict",
    "receipt_ready",  # Receipt is ready for export
    "receipt_exported",  # Receipt has been exported
    # Graph debates
    "graph_branch_created",
    "graph_branch_merged",
    # Genesis evolution
    "genesis_evolution",
    # Breakpoints
    "breakpoint",
    "breakpoint_resolved",
    # Cross-pollination events
    "agent_elo_updated",
    "knowledge_indexed",
    "knowledge_queried",
    "mound_updated",
    "calibration_update",
    "evidence_found",
    "agent_calibration_changed",
    "agent_fallback_triggered",
    # Explainability
    "explanation_ready",  # Decision explanation is available
}


@dataclass
class WebhookConfig:
    """Configuration for a registered webhook."""

    id: str
    url: str
    events: List[str]
    secret: str  # Used for HMAC signature
    active: bool = True
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Optional metadata
    name: Optional[str] = None
    description: Optional[str] = None

    # Delivery tracking
    last_delivery_at: Optional[float] = None
    last_delivery_status: Optional[int] = None
    delivery_count: int = 0
    failure_count: int = 0

    # Owner (for multi-tenant)
    user_id: Optional[str] = None
    workspace_id: Optional[str] = None

    def to_dict(self, include_secret: bool = False) -> dict:
        """Convert to dict, optionally excluding secret."""
        result = asdict(self)
        if not include_secret:
            result.pop("secret", None)
        return result

    def matches_event(self, event_type: str) -> bool:
        """Check if this webhook should receive the given event."""
        if not self.active:
            return False
        # "*" means all events
        if "*" in self.events:
            return event_type in WEBHOOK_EVENTS
        return event_type in self.events


class WebhookStore:
    """
    In-memory webhook storage with optional persistence.

    For production, this should be backed by a database.
    """

    def __init__(self):
        self._webhooks: Dict[str, WebhookConfig] = {}

    def register(
        self,
        url: str,
        events: List[str],
        name: Optional[str] = None,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> WebhookConfig:
        """Register a new webhook."""
        webhook_id = str(uuid.uuid4())
        secret = secrets.token_urlsafe(32)

        webhook = WebhookConfig(
            id=webhook_id,
            url=url,
            events=events,
            secret=secret,
            name=name,
            description=description,
            user_id=user_id,
            workspace_id=workspace_id,
        )

        self._webhooks[webhook_id] = webhook
        logger.info(f"Registered webhook {webhook_id} for events: {events}")
        return webhook

    def get(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get webhook by ID."""
        return self._webhooks.get(webhook_id)

    def list(
        self,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        active_only: bool = False,
    ) -> List[WebhookConfig]:
        """List webhooks with optional filtering."""
        webhooks = list(self._webhooks.values())

        if user_id:
            webhooks = [w for w in webhooks if w.user_id == user_id]
        if workspace_id:
            webhooks = [w for w in webhooks if w.workspace_id == workspace_id]
        if active_only:
            webhooks = [w for w in webhooks if w.active]

        return sorted(webhooks, key=lambda w: w.created_at, reverse=True)

    def delete(self, webhook_id: str) -> bool:
        """Delete webhook by ID."""
        if webhook_id in self._webhooks:
            del self._webhooks[webhook_id]
            logger.info(f"Deleted webhook {webhook_id}")
            return True
        return False

    def update(
        self,
        webhook_id: str,
        url: Optional[str] = None,
        events: Optional[List[str]] = None,
        active: Optional[bool] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[WebhookConfig]:
        """Update webhook configuration."""
        webhook = self._webhooks.get(webhook_id)
        if not webhook:
            return None

        if url is not None:
            webhook.url = url
        if events is not None:
            webhook.events = events
        if active is not None:
            webhook.active = active
        if name is not None:
            webhook.name = name
        if description is not None:
            webhook.description = description

        webhook.updated_at = time.time()
        return webhook

    def record_delivery(
        self,
        webhook_id: str,
        status_code: int,
        success: bool = True,
    ) -> None:
        """Record webhook delivery attempt."""
        webhook = self._webhooks.get(webhook_id)
        if webhook:
            webhook.last_delivery_at = time.time()
            webhook.last_delivery_status = status_code
            webhook.delivery_count += 1
            if not success:
                webhook.failure_count += 1

    def get_for_event(self, event_type: str) -> List[WebhookConfig]:
        """Get all active webhooks that should receive the given event."""
        return [w for w in self._webhooks.values() if w.matches_event(event_type)]


# Global webhook store (can be replaced with DB-backed implementation)
_webhook_store: Optional[WebhookStore] = None


def get_webhook_store() -> WebhookStore:
    """Get or create the webhook store."""
    global _webhook_store
    if _webhook_store is None:
        _webhook_store = WebhookStore()
    return _webhook_store


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


class WebhookHandler(BaseHandler):
    """Handler for webhook management API endpoints."""

    # Routes this handler responds to
    routes = [
        "POST /api/webhooks",
        "GET /api/webhooks",
        "GET /api/webhooks/events",
        "GET /api/webhooks/:id",
        "DELETE /api/webhooks/:id",
        "PATCH /api/webhooks/:id",
        "POST /api/webhooks/:id/test",
    ]

    ROUTES = [
        "/api/webhooks",
        "/api/webhooks/events",
    ]

    @staticmethod
    def can_handle(path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path.startswith("/api/webhooks")

    def __init__(self, server_context: dict):
        """Initialize with server context."""
        super().__init__(server_context)
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

    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle GET requests for webhook endpoints."""
        # GET /api/webhooks/events - list available event types
        if path == "/api/webhooks/events":
            return self._handle_list_events()

        # GET /api/webhooks/:id
        if path.startswith("/api/webhooks/") and path.count("/") == 3:
            webhook_id, err = self.extract_path_param(path, 2, "webhook_id", SAFE_ID_PATTERN)
            if err:
                return err
            return self._handle_get_webhook(webhook_id, handler)

        # GET /api/webhooks - list all webhooks
        if path == "/api/webhooks":
            return self._handle_list_webhooks(query_params, handler)

        return None

    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests for webhook endpoints."""
        # POST /api/webhooks/:id/test
        if path.endswith("/test") and path.count("/") == 4:
            webhook_id, err = self.extract_path_param(path, 2, "webhook_id", SAFE_ID_PATTERN)
            if err:
                return err
            return self._handle_test_webhook(webhook_id, handler)

        # POST /api/webhooks - register new webhook
        if path == "/api/webhooks":
            body, err = self.read_json_body_validated(handler)
            if err:
                return err
            return self._handle_register_webhook(body, handler)

        return None

    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle DELETE requests for webhook endpoints."""
        # DELETE /api/webhooks/:id
        if path.startswith("/api/webhooks/") and path.count("/") == 3:
            webhook_id, err = self.extract_path_param(path, 2, "webhook_id", SAFE_ID_PATTERN)
            if err:
                return err
            return self._handle_delete_webhook(webhook_id, handler)

        return None

    def handle_patch(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle PATCH requests for webhook endpoints."""
        # PATCH /api/webhooks/:id
        if path.startswith("/api/webhooks/") and path.count("/") == 3:
            webhook_id, err = self.extract_path_param(path, 2, "webhook_id", SAFE_ID_PATTERN)
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
                    "claim_verification_result": "Fired when a claim is verified",
                    "formal_verification_result": "Fired when formal verification completes",
                    "gauntlet_complete": "Fired when gauntlet stress-test completes",
                    "gauntlet_verdict": "Fired when gauntlet verdict is determined",
                    "graph_branch_created": "Fired when a graph debate branches",
                    "graph_branch_merged": "Fired when graph branches merge",
                    "genesis_evolution": "Fired when agent population evolves",
                    "breakpoint": "Fired when a human intervention breakpoint triggers",
                    "breakpoint_resolved": "Fired when a breakpoint is resolved",
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
        store = self._get_webhook_store()
        webhook = store.get(webhook_id)

        if not webhook:
            return error_response(f"Webhook not found: {webhook_id}", 404)

        # Check ownership
        user = self.get_current_user(handler)
        if user and webhook.user_id and webhook.user_id != user.user_id:
            return error_response("Access denied", 403)

        store.delete(webhook_id)
        return json_response(
            {
                "deleted": True,
                "webhook_id": webhook_id,
            }
        )

    def _handle_update_webhook(self, webhook_id: str, body: dict, handler: Any) -> HandlerResult:
        """Handle PATCH /api/webhooks/:id - update webhook."""
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

        return json_response({"webhook": updated.to_dict(include_secret=False)})

    def _handle_test_webhook(self, webhook_id: str, handler: Any) -> HandlerResult:
        """Handle POST /api/webhooks/:id/test - send test event."""
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
