"""
Tests for Automation Platform Webhooks Handler.

Tests cover:
- AutomationHandler routing (can_handle)
- GET /api/v1/webhooks - List webhook subscriptions
- GET /api/v1/webhooks/{id} - Get subscription details
- GET /api/v1/webhooks/events - List available event types
- GET /api/v1/webhooks/platforms - List supported platforms
- POST /api/v1/webhooks - Create subscription
- POST /api/v1/webhooks/subscribe - Create subscription (alias)
- POST /api/v1/webhooks/{id}/test - Test subscription
- POST /api/v1/webhooks/dispatch - Dispatch event (internal)
- DELETE /api/v1/webhooks/{id} - Remove subscription
- GET /api/v1/n8n/node - n8n node definition
- GET /api/v1/n8n/credentials - n8n credentials definition
- GET /api/v1/n8n/trigger - n8n trigger definition
- RBAC permission checks
- Error handling
"""

import sys
import types as _types_mod

# Pre-stub Slack modules to avoid circular ImportError
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import json
import pytest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aragora.server.handlers.integrations.automation import (
    AutomationHandler,
    WEBHOOKS_READ,
    WEBHOOKS_CREATE,
    WEBHOOKS_DELETE,
    WEBHOOKS_DISPATCH,
    INTEGRATIONS_ADMIN,
)


# =============================================================================
# Mock Data Classes
# =============================================================================


class MockEventType(Enum):
    DEBATE_STARTED = "debate.started"
    DEBATE_COMPLETED = "debate.completed"
    CONSENSUS_REACHED = "consensus.reached"
    VOTE_CAST = "vote.cast"
    TEST_EVENT = "test.event"


@dataclass
class MockSubscription:
    """Mock webhook subscription."""

    id: str = "sub_123"
    webhook_url: str = "https://hooks.example.com/abc123"
    events: list = field(default_factory=lambda: [MockEventType.DEBATE_COMPLETED])
    workspace_id: str = None
    user_id: str = None
    name: str = "Test Subscription"
    secret: str = "webhook_secret_123"
    verified: bool = True
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "webhook_url": self.webhook_url,
            "events": [e.value if hasattr(e, "value") else e for e in self.events],
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "name": self.name,
            "verified": self.verified,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class MockDispatchResult:
    """Mock dispatch result."""

    subscription_id: str = "sub_123"
    success: bool = True
    status_code: int = 200
    error: str = None
    duration_ms: float = 50.0


# =============================================================================
# Helpers
# =============================================================================


def _parse_result(result):
    """Parse HandlerResult into (body_dict, status_code)."""
    if result is None:
        return {}, 404
    body = json.loads(result.body) if hasattr(result, "body") and result.body else {}
    status = result.status_code if hasattr(result, "status_code") else 200
    return body, status


def _make_mock_handler(
    *,
    authenticated: bool = True,
    user_id: str = "user_123",
    org_id: str = "org_123",
    role: str = "member",
    client_ip: str = "127.0.0.1",
    json_body: dict = None,
    raw_body: bytes = None,
):
    """Build a mock HTTP handler."""
    handler = MagicMock()

    # Mock request
    handler.request = MagicMock()
    handler.request.headers = {"Authorization": "Bearer test_token"}

    if json_body is not None:
        handler.request.body = json.dumps(json_body).encode("utf-8")
    elif raw_body is not None:
        handler.request.body = raw_body
    else:
        handler.request.body = b"{}"

    return handler


def _make_auth_context(
    *,
    is_authenticated: bool = True,
    user_id: str = "user_123",
    org_id: str = "org_123",
    role: str = "admin",
    client_ip: str = "127.0.0.1",
):
    """Create a mock auth context."""
    ctx = MagicMock()
    ctx.is_authenticated = is_authenticated
    ctx.user_id = user_id
    ctx.org_id = org_id
    ctx.role = role
    ctx.client_ip = client_ip
    return ctx


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def handler():
    """Create an AutomationHandler instance."""
    return AutomationHandler()


@pytest.fixture
def mock_zapier():
    """Create a mock Zapier connector."""
    connector = MagicMock()
    connector.list_subscriptions = MagicMock(return_value=[MockSubscription()])
    connector.get_subscription = MagicMock(return_value=MockSubscription())
    connector.subscribe = AsyncMock(return_value=MockSubscription())
    connector.unsubscribe = AsyncMock(return_value=True)
    connector.dispatch_event = AsyncMock(return_value=[MockDispatchResult()])
    connector.test_subscription = AsyncMock(return_value=True)
    return connector


@pytest.fixture
def mock_n8n():
    """Create a mock n8n connector."""
    connector = MagicMock()
    connector.list_subscriptions = MagicMock(return_value=[])
    connector.get_subscription = MagicMock(return_value=None)
    connector.subscribe = AsyncMock(return_value=MockSubscription())
    connector.unsubscribe = AsyncMock(return_value=False)
    connector.dispatch_event = AsyncMock(return_value=[])
    connector.get_node_definition = MagicMock(
        return_value={
            "name": "aragora",
            "displayName": "Aragora",
            "version": 1,
        }
    )
    connector.get_credentials_definition = MagicMock(
        return_value={
            "name": "aragonApiKey",
            "displayName": "Aragora API Key",
        }
    )
    connector.get_trigger_definition = MagicMock(
        return_value={
            "name": "aragonTrigger",
            "displayName": "Aragora Trigger",
        }
    )
    return connector


@pytest.fixture
def mock_auth_allowed():
    """Mock auth context that allows permissions."""
    auth_ctx = _make_auth_context()

    def mock_extract(*args, **kwargs):
        return auth_ctx

    return mock_extract


@pytest.fixture
def mock_auth_denied():
    """Mock auth context that denies permissions."""
    auth_ctx = _make_auth_context(is_authenticated=False, user_id=None)

    def mock_extract(*args, **kwargs):
        return auth_ctx

    return mock_extract


# =============================================================================
# Tests: Permission Constants
# =============================================================================


class TestPermissionConstants:
    """Test permission constant definitions."""

    def test_webhooks_read(self):
        assert WEBHOOKS_READ == "webhooks.read"

    def test_webhooks_create(self):
        assert WEBHOOKS_CREATE == "webhooks.create"

    def test_webhooks_delete(self):
        assert WEBHOOKS_DELETE == "webhooks.delete"

    def test_webhooks_dispatch(self):
        assert WEBHOOKS_DISPATCH == "webhooks.all"

    def test_integrations_admin(self):
        assert INTEGRATIONS_ADMIN == "connectors.authorize"


# =============================================================================
# Tests: can_handle
# =============================================================================


class TestCanHandle:
    """Tests for route matching."""

    def test_can_handle_webhooks_base(self, handler):
        assert handler.can_handle("/api/v1/webhooks") is True

    def test_can_handle_webhooks_subscribe(self, handler):
        assert handler.can_handle("/api/v1/webhooks/subscribe") is True

    def test_can_handle_webhooks_events(self, handler):
        assert handler.can_handle("/api/v1/webhooks/events") is True

    def test_can_handle_webhooks_dispatch(self, handler):
        assert handler.can_handle("/api/v1/webhooks/dispatch") is True

    def test_can_handle_webhooks_platforms(self, handler):
        assert handler.can_handle("/api/v1/webhooks/platforms") is True

    def test_can_handle_n8n_node(self, handler):
        assert handler.can_handle("/api/v1/n8n/node") is True

    def test_can_handle_n8n_credentials(self, handler):
        assert handler.can_handle("/api/v1/n8n/credentials") is True

    def test_can_handle_n8n_trigger(self, handler):
        assert handler.can_handle("/api/v1/n8n/trigger") is True

    def test_cannot_handle_unrelated(self, handler):
        assert handler.can_handle("/api/v1/debates") is False


# =============================================================================
# Tests: GET /api/v1/webhooks - List Subscriptions
# =============================================================================


class TestListWebhooks:
    """Tests for GET webhooks list endpoint."""

    def test_unauthenticated_returns_401(self, handler, mock_auth_denied):
        """Returns 401 when not authenticated."""
        mock_handler = _make_mock_handler(authenticated=False)

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            mock_auth_denied,
        ):
            result = handler.handle_get("/api/v1/webhooks", {}, mock_handler)

        body, status = _parse_result(result)
        assert status == 401

    def test_list_webhooks_success(self, handler, mock_zapier, mock_auth_allowed):
        """Successfully lists webhooks."""
        handler._connectors = {"zapier": mock_zapier, "n8n": MagicMock(), "generic": mock_zapier}
        mock_handler = _make_mock_handler()

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = handler.handle_get("/api/v1/webhooks", {}, mock_handler)

        body, status = _parse_result(result)
        assert status == 200
        assert "subscriptions" in body["data"]
        assert "count" in body["data"]

    def test_list_webhooks_with_platform_filter(self, handler, mock_zapier, mock_auth_allowed):
        """Lists webhooks filtered by platform."""
        mock_n8n = MagicMock()
        mock_n8n.list_subscriptions = MagicMock(return_value=[])
        handler._connectors = {"zapier": mock_zapier, "n8n": mock_n8n, "generic": mock_zapier}
        mock_handler = _make_mock_handler()

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = handler.handle_get(
                "/api/v1/webhooks",
                {"platform": "zapier"},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 200
        # Only zapier was called
        mock_zapier.list_subscriptions.assert_called()
        mock_n8n.list_subscriptions.assert_not_called()


# =============================================================================
# Tests: GET /api/v1/webhooks/{id} - Get Subscription
# =============================================================================


class TestGetWebhook:
    """Tests for GET single webhook endpoint."""

    def test_get_webhook_success(self, handler, mock_zapier, mock_auth_allowed):
        """Successfully gets a webhook."""
        handler._connectors = {"zapier": mock_zapier, "n8n": MagicMock(), "generic": mock_zapier}
        mock_handler = _make_mock_handler()

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = handler.handle_get(
                "/api/v1/webhooks/sub_123",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["id"] == "sub_123"

    def test_get_webhook_not_found(self, handler, mock_auth_allowed):
        """Returns 404 when webhook not found."""
        mock_connector = MagicMock()
        mock_connector.get_subscription = MagicMock(return_value=None)
        handler._connectors = {
            "zapier": mock_connector,
            "n8n": mock_connector,
            "generic": mock_connector,
        }
        mock_handler = _make_mock_handler()

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = handler.handle_get(
                "/api/v1/webhooks/nonexistent",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 404


# =============================================================================
# Tests: GET /api/v1/webhooks/events - List Events
# =============================================================================


class TestListEvents:
    """Tests for GET events list endpoint."""

    def test_list_events_success(self, handler, mock_auth_allowed):
        """Successfully lists available events."""
        mock_handler = _make_mock_handler()

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = handler.handle_get("/api/v1/webhooks/events", {}, mock_handler)

        body, status = _parse_result(result)
        assert status == 200
        assert "events" in body["data"]
        assert "categories" in body["data"]
        assert "count" in body["data"]


# =============================================================================
# Tests: GET /api/v1/webhooks/platforms - List Platforms
# =============================================================================


class TestListPlatforms:
    """Tests for GET platforms list endpoint."""

    def test_list_platforms_success(self, handler, mock_auth_allowed):
        """Successfully lists supported platforms."""
        mock_handler = _make_mock_handler()

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = handler.handle_get("/api/v1/webhooks/platforms", {}, mock_handler)

        body, status = _parse_result(result)
        assert status == 200
        assert "platforms" in body["data"]
        platforms = body["data"]["platforms"]
        platform_ids = [p["id"] for p in platforms]
        assert "zapier" in platform_ids
        assert "n8n" in platform_ids
        assert "generic" in platform_ids


# =============================================================================
# Tests: POST /api/v1/webhooks - Subscribe
# =============================================================================


class TestSubscribe:
    """Tests for POST subscribe endpoint."""

    @pytest.mark.asyncio
    async def test_subscribe_success(self, handler, mock_zapier, mock_auth_allowed):
        """Successfully creates a subscription."""
        handler._connectors = {"zapier": mock_zapier, "n8n": MagicMock(), "generic": mock_zapier}
        mock_handler = _make_mock_handler(
            json_body={
                "webhook_url": "https://hooks.example.com/new",
                "events": ["debate.completed"],
                "platform": "zapier",
                "name": "New Subscription",
            }
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = await handler.handle_post("/api/v1/webhooks", {}, mock_handler)

        body, status = _parse_result(result)
        assert status == 201
        assert "subscription" in body
        assert "secret" in body
        mock_zapier.subscribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_subscribe_missing_url(self, handler, mock_auth_allowed):
        """Returns 400 when webhook_url is missing."""
        mock_handler = _make_mock_handler(
            json_body={
                "events": ["debate.completed"],
            }
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = await handler.handle_post("/api/v1/webhooks", {}, mock_handler)

        body, status = _parse_result(result)
        assert status == 400
        assert "webhook_url" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_subscribe_missing_events(self, handler, mock_auth_allowed):
        """Returns 400 when events is missing."""
        mock_handler = _make_mock_handler(
            json_body={
                "webhook_url": "https://hooks.example.com/new",
            }
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = await handler.handle_post("/api/v1/webhooks", {}, mock_handler)

        body, status = _parse_result(result)
        assert status == 400
        assert "events" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_subscribe_invalid_event(self, handler, mock_auth_allowed):
        """Returns 400 when event type is invalid."""
        mock_handler = _make_mock_handler(
            json_body={
                "webhook_url": "https://hooks.example.com/new",
                "events": ["invalid.event.type"],
            }
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = await handler.handle_post("/api/v1/webhooks", {}, mock_handler)

        body, status = _parse_result(result)
        assert status == 400

    @pytest.mark.asyncio
    async def test_subscribe_unknown_platform(self, handler, mock_auth_allowed):
        """Returns 400 when platform is unknown."""
        mock_handler = _make_mock_handler(
            json_body={
                "webhook_url": "https://hooks.example.com/new",
                "events": ["debate.completed"],
                "platform": "unknown_platform",
            }
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = await handler.handle_post("/api/v1/webhooks", {}, mock_handler)

        body, status = _parse_result(result)
        assert status == 400


# =============================================================================
# Tests: POST /api/v1/webhooks/{id}/test - Test Subscription
# =============================================================================


class TestTestWebhook:
    """Tests for POST test webhook endpoint."""

    @pytest.mark.asyncio
    async def test_webhook_success(self, handler, mock_zapier, mock_auth_allowed):
        """Successfully tests a webhook."""
        handler._connectors = {"zapier": mock_zapier, "n8n": MagicMock(), "generic": mock_zapier}
        handler._zapier = mock_zapier
        mock_handler = _make_mock_handler()

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = await handler.handle_post(
                "/api/v1/webhooks/sub_123/test",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["success"] is True

    @pytest.mark.asyncio
    async def test_webhook_not_found(self, handler, mock_auth_allowed):
        """Returns 404 when webhook not found."""
        mock_connector = MagicMock()
        mock_connector.get_subscription = MagicMock(return_value=None)
        handler._connectors = {
            "zapier": mock_connector,
            "n8n": mock_connector,
            "generic": mock_connector,
        }
        mock_handler = _make_mock_handler()

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = await handler.handle_post(
                "/api/v1/webhooks/nonexistent/test",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 404


# =============================================================================
# Tests: POST /api/v1/webhooks/dispatch - Dispatch Event
# =============================================================================


class TestDispatchEvent:
    """Tests for POST dispatch event endpoint."""

    @pytest.mark.asyncio
    async def test_dispatch_success(self, handler, mock_zapier, mock_auth_allowed):
        """Successfully dispatches an event."""
        handler._connectors = {"zapier": mock_zapier, "n8n": MagicMock(), "generic": mock_zapier}
        mock_handler = _make_mock_handler(
            json_body={
                "event_type": "debate.completed",
                "payload": {"debate_id": "debate_123"},
            }
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = await handler.handle_post(
                "/api/v1/webhooks/dispatch",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 200
        assert "dispatched" in body["data"]
        assert "success" in body["data"]
        assert "results" in body["data"]

    @pytest.mark.asyncio
    async def test_dispatch_missing_event_type(self, handler, mock_auth_allowed):
        """Returns 400 when event_type is missing."""
        mock_handler = _make_mock_handler(
            json_body={
                "payload": {"debate_id": "debate_123"},
            }
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = await handler.handle_post(
                "/api/v1/webhooks/dispatch",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 400

    @pytest.mark.asyncio
    async def test_dispatch_invalid_event_type(self, handler, mock_auth_allowed):
        """Returns 400 when event_type is invalid."""
        mock_handler = _make_mock_handler(
            json_body={
                "event_type": "invalid.event",
                "payload": {},
            }
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = await handler.handle_post(
                "/api/v1/webhooks/dispatch",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 400


# =============================================================================
# Tests: DELETE /api/v1/webhooks/{id} - Unsubscribe
# =============================================================================


class TestUnsubscribe:
    """Tests for DELETE webhook endpoint."""

    def test_unsubscribe_success(self, handler, mock_zapier, mock_auth_allowed):
        """Successfully deletes a subscription."""
        handler._connectors = {"zapier": mock_zapier, "n8n": MagicMock(), "generic": mock_zapier}
        mock_handler = _make_mock_handler()

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = handler.handle_delete(
                "/api/v1/webhooks/sub_123",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 200

    def test_unsubscribe_not_found(self, handler, mock_auth_allowed):
        """Returns 404 when webhook not found."""
        mock_connector = MagicMock()
        mock_connector.unsubscribe = AsyncMock(return_value=False)
        handler._connectors = {
            "zapier": mock_connector,
            "n8n": mock_connector,
            "generic": mock_connector,
        }
        mock_handler = _make_mock_handler()

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = handler.handle_delete(
                "/api/v1/webhooks/nonexistent",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 404


# =============================================================================
# Tests: n8n Endpoints
# =============================================================================


class TestN8nEndpoints:
    """Tests for n8n-specific endpoints."""

    def test_get_n8n_node(self, handler, mock_n8n, mock_auth_allowed):
        """Successfully gets n8n node definition."""
        handler._n8n = mock_n8n
        mock_handler = _make_mock_handler()

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = handler.handle_get("/api/v1/n8n/node", {}, mock_handler)

        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["name"] == "aragora"

    def test_get_n8n_credentials(self, handler, mock_n8n, mock_auth_allowed):
        """Successfully gets n8n credentials definition."""
        handler._n8n = mock_n8n
        mock_handler = _make_mock_handler()

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = handler.handle_get("/api/v1/n8n/credentials", {}, mock_handler)

        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["name"] == "aragonApiKey"

    def test_get_n8n_trigger(self, handler, mock_n8n, mock_auth_allowed):
        """Successfully gets n8n trigger definition."""
        handler._n8n = mock_n8n
        mock_handler = _make_mock_handler()

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = handler.handle_get("/api/v1/n8n/trigger", {}, mock_handler)

        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["name"] == "aragonTrigger"


# =============================================================================
# Tests: RBAC Permission Checks
# =============================================================================


class TestRBACPermissions:
    """Tests for RBAC permission enforcement."""

    def test_webhooks_read_permission_denied(self, handler):
        """Returns 403 when webhooks.read permission denied."""
        mock_handler = _make_mock_handler()
        auth_ctx = _make_auth_context()

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=False, reason="Permission denied"),
            ),
        ):
            result = handler.handle_get("/api/v1/webhooks", {}, mock_handler)

        body, status = _parse_result(result)
        assert status == 403

    @pytest.mark.asyncio
    async def test_webhooks_create_permission_denied(self, handler):
        """Returns 403 when webhooks.create permission denied."""
        mock_handler = _make_mock_handler(
            json_body={
                "webhook_url": "https://example.com",
                "events": ["debate.completed"],
            }
        )
        auth_ctx = _make_auth_context()

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=False, reason="Permission denied"),
            ),
        ):
            result = await handler.handle_post("/api/v1/webhooks", {}, mock_handler)

        body, status = _parse_result(result)
        assert status == 403

    def test_webhooks_delete_permission_denied(self, handler):
        """Returns 403 when webhooks.delete permission denied."""
        mock_handler = _make_mock_handler()
        auth_ctx = _make_auth_context()

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=False, reason="Permission denied"),
            ),
        ):
            result = handler.handle_delete("/api/v1/webhooks/sub_123", {}, mock_handler)

        body, status = _parse_result(result)
        assert status == 403

    def test_n8n_requires_integrations_admin(self, handler):
        """n8n endpoints require integrations:admin permission."""
        mock_handler = _make_mock_handler()
        auth_ctx = _make_auth_context()

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=False, reason="Not admin"),
            ),
        ):
            result = handler.handle_get("/api/v1/n8n/node", {}, mock_handler)

        body, status = _parse_result(result)
        assert status == 403


# =============================================================================
# Tests: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_subscribe_internal_error(self, handler, mock_auth_allowed):
        """Returns 500 on internal error during subscribe."""
        mock_connector = MagicMock()
        mock_connector.subscribe = AsyncMock(side_effect=RuntimeError("Internal error"))
        handler._connectors = {
            "zapier": mock_connector,
            "n8n": mock_connector,
            "generic": mock_connector,
        }
        mock_handler = _make_mock_handler(
            json_body={
                "webhook_url": "https://example.com",
                "events": ["debate.completed"],
            }
        )

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            result = await handler.handle_post("/api/v1/webhooks", {}, mock_handler)

        body, status = _parse_result(result)
        assert status == 500

    def test_invalid_json_body(self, handler, mock_auth_allowed):
        """Handles invalid JSON body gracefully."""
        mock_handler = _make_mock_handler(raw_body=b"not valid json")

        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                mock_auth_allowed,
            ),
            patch(
                "aragora.rbac.check_permission",
                return_value=MagicMock(allowed=True, reason=None),
            ),
        ):
            # This should handle the error gracefully
            result = handler.handle_get("/api/v1/webhooks", {}, mock_handler)

        # Should not crash
        body, status = _parse_result(result)
        assert status in [200, 400]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
