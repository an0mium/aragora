"""Tests for Automation Platform Webhook Handler.

Tests the AutomationHandler which provides REST API endpoints for
managing webhook subscriptions with automation platforms (Zapier, n8n, generic).

Endpoints tested:
- GET /api/v1/webhooks - List subscriptions
- GET /api/v1/webhooks/{id} - Get subscription details
- GET /api/v1/webhooks/events - List available event types
- GET /api/v1/webhooks/platforms - List supported platforms
- GET /api/v1/n8n/node - Get n8n node definition
- GET /api/v1/n8n/credentials - Get n8n credentials definition
- GET /api/v1/n8n/trigger - Get n8n trigger definition
- POST /api/v1/webhooks/subscribe - Create subscription
- POST /api/v1/webhooks - Create subscription (alias)
- POST /api/v1/webhooks/{id}/test - Test a subscription
- POST /api/v1/webhooks/dispatch - Dispatch event
- DELETE /api/v1/webhooks/{id} - Remove subscription
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.automation.base import (
    AutomationEventType,
    WebhookDeliveryResult,
    WebhookSubscription,
)
from aragora.server.handlers.base import HandlerResult
from aragora.server.handlers.integrations.automation import AutomationHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


def _status(result: HandlerResult) -> int:
    """Extract HTTP status code from HandlerResult."""
    return result.status_code


def _make_mock_handler(body: dict | None = None) -> MagicMock:
    """Create a mock HTTP handler with an optional JSON body."""
    handler = MagicMock()
    if body is not None:
        raw = json.dumps(body).encode("utf-8")
        handler.request = MagicMock()
        handler.request.body = raw
        handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(raw)),
        }
        handler.rfile = MagicMock()
        handler.rfile.read.return_value = raw
    else:
        handler.request = MagicMock()
        handler.request.body = b"{}"
        handler.headers = {"Content-Length": "2"}
        handler.rfile = MagicMock()
        handler.rfile.read.return_value = b"{}"
    return handler


def _make_subscription(
    sub_id: str = "sub-001",
    webhook_url: str = "https://hooks.example.com/test",
    events: set | None = None,
    platform: str = "zapier",
    workspace_id: str | None = "ws-001",
    verified: bool = False,
    secret: str = "test-secret",
) -> WebhookSubscription:
    """Create a WebhookSubscription for testing."""
    return WebhookSubscription(
        id=sub_id,
        webhook_url=webhook_url,
        events=events or {AutomationEventType.DEBATE_COMPLETED},
        platform=platform,
        workspace_id=workspace_id,
        verified=verified,
        secret=secret,
    )


def _make_delivery_result(
    subscription_id: str = "sub-001",
    success: bool = True,
    status_code: int = 200,
    error: str | None = None,
    duration_ms: float = 42.0,
) -> WebhookDeliveryResult:
    """Create a WebhookDeliveryResult for testing."""
    return WebhookDeliveryResult(
        subscription_id=subscription_id,
        event_type=AutomationEventType.TEST_EVENT,
        success=success,
        status_code=status_code,
        error=error,
        duration_ms=duration_ms,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def server_context():
    """Create mock server context."""
    return {
        "storage": MagicMock(),
        "user_store": MagicMock(),
    }


@pytest.fixture
def handler(server_context):
    """Create AutomationHandler with RBAC bypass."""
    h = AutomationHandler(server_context)
    # Bypass RBAC checks for unit tests (conftest patches SecureHandler but
    # this handler uses its own _check_rbac_permission).
    h._check_rbac_permission = MagicMock(return_value=None)
    return h


# ============================================================================
# can_handle
# ============================================================================


class TestCanHandle:
    """Tests for AutomationHandler.can_handle."""

    def test_handles_webhooks_root(self, handler):
        assert handler.can_handle("/api/v1/webhooks") is True

    def test_handles_webhooks_subscribe(self, handler):
        assert handler.can_handle("/api/v1/webhooks/subscribe") is True

    def test_handles_webhooks_events(self, handler):
        assert handler.can_handle("/api/v1/webhooks/events") is True

    def test_handles_webhooks_dispatch(self, handler):
        assert handler.can_handle("/api/v1/webhooks/dispatch") is True

    def test_handles_webhooks_platforms(self, handler):
        assert handler.can_handle("/api/v1/webhooks/platforms") is True

    def test_handles_webhook_by_id(self, handler):
        assert handler.can_handle("/api/v1/webhooks/abc-123") is True

    def test_handles_webhook_test(self, handler):
        assert handler.can_handle("/api/v1/webhooks/abc-123/test") is True

    def test_handles_n8n_node(self, handler):
        assert handler.can_handle("/api/v1/n8n/node") is True

    def test_handles_n8n_credentials(self, handler):
        assert handler.can_handle("/api/v1/n8n/credentials") is True

    def test_handles_n8n_trigger(self, handler):
        assert handler.can_handle("/api/v1/n8n/trigger") is True

    def test_rejects_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False
        assert handler.can_handle("/api/v1/agents") is False
        assert handler.can_handle("/api/v2/webhooks") is False

    def test_rejects_partial_match(self, handler):
        assert handler.can_handle("/api/v1/webhook") is False


# ============================================================================
# GET /api/v1/webhooks - List subscriptions
# ============================================================================


class TestListWebhooks:
    """Tests for GET /api/v1/webhooks."""

    def test_list_webhooks_empty(self, handler):
        mock_http = _make_mock_handler()
        result = handler.handle_get("/api/v1/webhooks", {}, mock_http)

        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        data = body["data"]
        assert data["subscriptions"] == []
        assert data["count"] == 0

    def test_list_webhooks_returns_all(self, handler):
        sub = _make_subscription()
        handler._zapier._subscriptions[sub.id] = sub

        mock_http = _make_mock_handler()
        result = handler.handle_get("/api/v1/webhooks", {}, mock_http)

        assert _status(result) == 200
        data = _body(result)["data"]
        # zapier and generic point to same connector, so sub appears twice
        assert data["count"] >= 1
        ids = [s["id"] for s in data["subscriptions"]]
        assert sub.id in ids

    def test_list_webhooks_filter_by_platform(self, handler):
        sub = _make_subscription()
        handler._zapier._subscriptions[sub.id] = sub

        mock_http = _make_mock_handler()
        result = handler.handle_get("/api/v1/webhooks", {"platform": "n8n"}, mock_http)

        assert _status(result) == 200
        data = _body(result)["data"]
        # n8n has no subs, but zapier and generic are skipped by platform filter
        assert data["count"] == 0

    def test_list_webhooks_filter_by_workspace(self, handler):
        sub = _make_subscription(workspace_id="ws-test")
        handler._zapier._subscriptions[sub.id] = sub

        mock_http = _make_mock_handler()
        result = handler.handle_get("/api/v1/webhooks", {"workspace_id": "ws-other"}, mock_http)

        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["count"] == 0

    def test_list_webhooks_filter_by_event(self, handler):
        sub = _make_subscription(events={AutomationEventType.DEBATE_COMPLETED})
        handler._zapier._subscriptions[sub.id] = sub

        mock_http = _make_mock_handler()
        result = handler.handle_get(
            "/api/v1/webhooks",
            {"event": "debate.completed"},
            mock_http,
        )

        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["count"] >= 1


# ============================================================================
# GET /api/v1/webhooks/{id} - Get subscription details
# ============================================================================


class TestGetWebhook:
    """Tests for GET /api/v1/webhooks/{id}."""

    def test_get_existing_webhook(self, handler):
        sub = _make_subscription(sub_id="sub-found")
        handler._zapier._subscriptions[sub.id] = sub

        mock_http = _make_mock_handler()
        result = handler.handle_get("/api/v1/webhooks/sub-found", {}, mock_http)

        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["id"] == "sub-found"
        assert data["webhook_url"] == "https://hooks.example.com/test"

    def test_get_nonexistent_webhook(self, handler):
        mock_http = _make_mock_handler()
        result = handler.handle_get("/api/v1/webhooks/nonexistent", {}, mock_http)

        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body["error"].lower()

    def test_get_webhook_does_not_match_reserved(self, handler):
        """Ensure IDs matching reserved keywords (events, subscribe, etc.)
        are not routed to _get_webhook."""
        mock_http = _make_mock_handler()
        # /api/v1/webhooks/events should be handled by _list_events, not _get_webhook
        result = handler.handle_get("/api/v1/webhooks/events", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        # _list_events returns events list in data
        assert "events" in body["data"]


# ============================================================================
# GET /api/v1/webhooks/events - List event types
# ============================================================================


class TestListEvents:
    """Tests for GET /api/v1/webhooks/events."""

    def test_list_events_success(self, handler):
        mock_http = _make_mock_handler()
        result = handler.handle_get("/api/v1/webhooks/events", {}, mock_http)

        assert _status(result) == 200
        data = _body(result)["data"]
        assert "events" in data
        assert "categories" in data
        assert data["count"] > 0

    def test_list_events_has_all_types(self, handler):
        mock_http = _make_mock_handler()
        result = handler.handle_get("/api/v1/webhooks/events", {}, mock_http)

        data = _body(result)["data"]
        event_values = {e["type"] for e in data["events"]}
        # Verify a selection of known event types are present
        assert "debate.started" in event_values
        assert "debate.completed" in event_values
        assert "consensus.reached" in event_values
        assert "test.event" in event_values

    def test_list_events_categories_grouped(self, handler):
        mock_http = _make_mock_handler()
        result = handler.handle_get("/api/v1/webhooks/events", {}, mock_http)

        categories = _body(result)["data"]["categories"]
        assert "debate" in categories
        assert "consensus" in categories
        # Each category should have a list of events
        for cat, events in categories.items():
            assert isinstance(events, list)
            assert len(events) > 0


# ============================================================================
# GET /api/v1/webhooks/platforms - List platforms
# ============================================================================


class TestListPlatforms:
    """Tests for GET /api/v1/webhooks/platforms."""

    def test_list_platforms_success(self, handler):
        mock_http = _make_mock_handler()
        result = handler.handle_get("/api/v1/webhooks/platforms", {}, mock_http)

        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["count"] == 3
        platform_ids = [p["id"] for p in data["platforms"]]
        assert "zapier" in platform_ids
        assert "n8n" in platform_ids
        assert "generic" in platform_ids

    def test_platform_has_required_fields(self, handler):
        mock_http = _make_mock_handler()
        result = handler.handle_get("/api/v1/webhooks/platforms", {}, mock_http)

        for platform in _body(result)["data"]["platforms"]:
            assert "id" in platform
            assert "name" in platform
            assert "description" in platform
            assert "documentation" in platform
            assert "features" in platform
            assert isinstance(platform["features"], list)


# ============================================================================
# GET /api/v1/n8n/* - n8n definitions
# ============================================================================


class TestN8NDefinitions:
    """Tests for n8n node/credentials/trigger definition endpoints."""

    def test_get_n8n_node(self, handler):
        mock_http = _make_mock_handler()
        result = handler.handle_get("/api/v1/n8n/node", {}, mock_http)

        assert _status(result) == 200
        data = _body(result)["data"]
        assert isinstance(data, dict)
        # Node definition should have standard n8n fields
        assert "displayName" in data or "name" in data or "description" in data

    def test_get_n8n_credentials(self, handler):
        mock_http = _make_mock_handler()
        result = handler.handle_get("/api/v1/n8n/credentials", {}, mock_http)

        assert _status(result) == 200
        data = _body(result)["data"]
        assert isinstance(data, dict)

    def test_get_n8n_trigger(self, handler):
        mock_http = _make_mock_handler()
        result = handler.handle_get("/api/v1/n8n/trigger", {}, mock_http)

        assert _status(result) == 200
        data = _body(result)["data"]
        assert isinstance(data, dict)


# ============================================================================
# GET - unhandled path returns None
# ============================================================================


class TestGetUnhandled:
    """Tests for GET requests on paths not handled by this handler."""

    def test_unhandled_path_returns_none(self, handler):
        mock_http = _make_mock_handler()
        result = handler.handle_get("/api/v1/debates", {}, mock_http)
        assert result is None


# ============================================================================
# POST /api/v1/webhooks/subscribe - Create subscription
# ============================================================================


class TestSubscribe:
    """Tests for POST /api/v1/webhooks/subscribe (and /api/v1/webhooks)."""

    @pytest.mark.asyncio
    async def test_subscribe_success(self, handler):
        body = {
            "webhook_url": "https://hooks.example.com/callback",
            "events": ["debate.completed"],
            "platform": "zapier",
            "workspace_id": "ws-001",
            "name": "My Webhook",
        }
        mock_http = _make_mock_handler(body)

        with patch(
            "aragora.server.handlers.integrations.automation.validate_webhook_url",
            return_value=(True, None),
        ):
            result = await handler.handle_post("/api/v1/webhooks/subscribe", {}, mock_http)

        assert _status(result) == 201
        resp = _body(result)
        assert "subscription" in resp
        assert "secret" in resp
        assert resp["message"] == "Webhook subscription created successfully"
        sub = resp["subscription"]
        assert sub["webhook_url"] == "https://hooks.example.com/callback"

    @pytest.mark.asyncio
    async def test_subscribe_via_webhooks_root(self, handler):
        """POST /api/v1/webhooks should also trigger subscription."""
        body = {
            "url": "https://hooks.example.com/alt",
            "events": ["debate.started"],
        }
        mock_http = _make_mock_handler(body)

        with patch(
            "aragora.server.handlers.integrations.automation.validate_webhook_url",
            return_value=(True, None),
        ):
            result = await handler.handle_post("/api/v1/webhooks", {}, mock_http)

        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_subscribe_missing_url(self, handler):
        body = {"events": ["debate.completed"]}
        mock_http = _make_mock_handler(body)

        result = await handler.handle_post("/api/v1/webhooks/subscribe", {}, mock_http)

        assert _status(result) == 400
        assert "webhook_url" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_subscribe_missing_events(self, handler):
        body = {"webhook_url": "https://hooks.example.com/callback"}
        mock_http = _make_mock_handler(body)

        with patch(
            "aragora.server.handlers.integrations.automation.validate_webhook_url",
            return_value=(True, None),
        ):
            result = await handler.handle_post("/api/v1/webhooks/subscribe", {}, mock_http)

        assert _status(result) == 400
        assert "events" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_subscribe_invalid_event_type(self, handler):
        body = {
            "webhook_url": "https://hooks.example.com/callback",
            "events": ["bogus.event"],
        }
        mock_http = _make_mock_handler(body)

        with patch(
            "aragora.server.handlers.integrations.automation.validate_webhook_url",
            return_value=(True, None),
        ):
            result = await handler.handle_post("/api/v1/webhooks/subscribe", {}, mock_http)

        assert _status(result) == 400
        assert "event" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_subscribe_unknown_platform(self, handler):
        body = {
            "webhook_url": "https://hooks.example.com/callback",
            "events": ["debate.completed"],
            "platform": "unknown_platform",
        }
        mock_http = _make_mock_handler(body)

        with patch(
            "aragora.server.handlers.integrations.automation.validate_webhook_url",
            return_value=(True, None),
        ):
            result = await handler.handle_post("/api/v1/webhooks/subscribe", {}, mock_http)

        assert _status(result) == 400
        assert "unknown" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_subscribe_ssrf_blocked(self, handler):
        body = {
            "webhook_url": "http://169.254.169.254/metadata",
            "events": ["debate.completed"],
        }
        mock_http = _make_mock_handler(body)

        with patch(
            "aragora.server.handlers.integrations.automation.validate_webhook_url",
            return_value=(False, "blocked IP range"),
        ):
            result = await handler.handle_post("/api/v1/webhooks/subscribe", {}, mock_http)

        assert _status(result) == 400
        assert "invalid webhook url" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_subscribe_invalid_body(self, handler):
        """Test handling of unparseable request body."""
        mock_http = MagicMock()
        mock_http.request = MagicMock()
        mock_http.request.body = b"not-json{"
        mock_http.headers = {"Content-Length": "10"}

        result = await handler.handle_post("/api/v1/webhooks/subscribe", {}, mock_http)

        assert _status(result) == 400
        assert "invalid" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_subscribe_connector_failure(self, handler):
        """Test that connector errors are handled gracefully."""
        body = {
            "webhook_url": "https://hooks.example.com/callback",
            "events": ["debate.completed"],
            "platform": "zapier",
        }
        mock_http = _make_mock_handler(body)

        with patch(
            "aragora.server.handlers.integrations.automation.validate_webhook_url",
            return_value=(True, None),
        ):
            handler._zapier.subscribe = AsyncMock(side_effect=RuntimeError("connection refused"))
            result = await handler.handle_post("/api/v1/webhooks/subscribe", {}, mock_http)

        assert _status(result) == 500
        assert "failed" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_subscribe_with_url_field_alias(self, handler):
        """Test that 'url' field works as alias for 'webhook_url'."""
        body = {
            "url": "https://hooks.example.com/alias",
            "events": ["test.event"],
        }
        mock_http = _make_mock_handler(body)

        with patch(
            "aragora.server.handlers.integrations.automation.validate_webhook_url",
            return_value=(True, None),
        ):
            result = await handler.handle_post("/api/v1/webhooks/subscribe", {}, mock_http)

        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_subscribe_defaults_to_generic_platform(self, handler):
        """Omitting platform defaults to 'generic'."""
        body = {
            "webhook_url": "https://hooks.example.com/generic",
            "events": ["debate.started"],
        }
        mock_http = _make_mock_handler(body)

        with patch(
            "aragora.server.handlers.integrations.automation.validate_webhook_url",
            return_value=(True, None),
        ):
            result = await handler.handle_post("/api/v1/webhooks/subscribe", {}, mock_http)

        assert _status(result) == 201


# ============================================================================
# POST /api/v1/webhooks/{id}/test - Test subscription
# ============================================================================


class TestTestWebhook:
    """Tests for POST /api/v1/webhooks/{id}/test."""

    @pytest.mark.asyncio
    async def test_test_webhook_zapier_success(self, handler):
        sub = _make_subscription(sub_id="sub-test", platform="zapier")
        handler._zapier._subscriptions[sub.id] = sub
        handler._zapier.test_subscription = AsyncMock(return_value=True)

        mock_http = _make_mock_handler()
        result = await handler.handle_post("/api/v1/webhooks/sub-test/test", {}, mock_http)

        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["success"] is True
        assert "successful" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_test_webhook_zapier_failure(self, handler):
        sub = _make_subscription(sub_id="sub-fail", platform="zapier")
        handler._zapier._subscriptions[sub.id] = sub
        handler._zapier.test_subscription = AsyncMock(return_value=False)

        mock_http = _make_mock_handler()
        result = await handler.handle_post("/api/v1/webhooks/sub-fail/test", {}, mock_http)

        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_test_webhook_n8n(self, handler):
        sub = _make_subscription(sub_id="sub-n8n")
        handler._n8n._subscriptions[sub.id] = sub
        handler._n8n.dispatch_event = AsyncMock(
            return_value=[_make_delivery_result(subscription_id="sub-n8n", success=True)]
        )

        mock_http = _make_mock_handler()
        result = await handler.handle_post("/api/v1/webhooks/sub-n8n/test", {}, mock_http)

        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_test_webhook_not_found(self, handler):
        mock_http = _make_mock_handler()
        result = await handler.handle_post("/api/v1/webhooks/nonexistent/test", {}, mock_http)

        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()


# ============================================================================
# POST /api/v1/webhooks/dispatch - Dispatch event
# ============================================================================


class TestDispatchEvent:
    """Tests for POST /api/v1/webhooks/dispatch."""

    @pytest.mark.asyncio
    async def test_dispatch_success(self, handler):
        delivery = _make_delivery_result(success=True)
        handler._zapier.dispatch_event = AsyncMock(return_value=[delivery])
        handler._n8n.dispatch_event = AsyncMock(return_value=[])

        body = {
            "event_type": "debate.completed",
            "payload": {"debate_id": "d-001"},
        }
        mock_http = _make_mock_handler(body)
        result = await handler.handle_post("/api/v1/webhooks/dispatch", {}, mock_http)

        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["success"] >= 1
        assert data["failed"] == 0

    @pytest.mark.asyncio
    async def test_dispatch_with_event_alias(self, handler):
        """'event' field works as alias for 'event_type'."""
        handler._zapier.dispatch_event = AsyncMock(return_value=[])
        handler._n8n.dispatch_event = AsyncMock(return_value=[])

        body = {"event": "test.event", "payload": {}}
        mock_http = _make_mock_handler(body)
        result = await handler.handle_post("/api/v1/webhooks/dispatch", {}, mock_http)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_dispatch_missing_event_type(self, handler):
        body = {"payload": {}}
        mock_http = _make_mock_handler(body)
        result = await handler.handle_post("/api/v1/webhooks/dispatch", {}, mock_http)

        assert _status(result) == 400
        assert "event_type" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_dispatch_invalid_event_type(self, handler):
        body = {"event_type": "invalid.event.type"}
        mock_http = _make_mock_handler(body)
        result = await handler.handle_post("/api/v1/webhooks/dispatch", {}, mock_http)

        assert _status(result) == 400
        assert "invalid" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_dispatch_invalid_body(self, handler):
        mock_http = MagicMock()
        mock_http.request = MagicMock()
        mock_http.request.body = b"<not-json>"
        mock_http.headers = {"Content-Length": "10"}

        result = await handler.handle_post("/api/v1/webhooks/dispatch", {}, mock_http)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_dispatch_mixed_results(self, handler):
        """Test dispatch with both successful and failed deliveries."""
        success = _make_delivery_result(subscription_id="s1", success=True)
        failure = _make_delivery_result(
            subscription_id="s2", success=False, status_code=500, error="timeout"
        )
        handler._zapier.dispatch_event = AsyncMock(return_value=[success, failure])
        handler._n8n.dispatch_event = AsyncMock(return_value=[])

        body = {"event_type": "debate.completed", "payload": {}}
        mock_http = _make_mock_handler(body)
        result = await handler.handle_post("/api/v1/webhooks/dispatch", {}, mock_http)

        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["dispatched"] >= 2
        assert data["success"] >= 1
        assert data["failed"] >= 1
        # Results should contain subscription_id, success, status_code, error, duration_ms
        for r in data["results"]:
            assert "subscription_id" in r
            assert "success" in r
            assert "status_code" in r
            assert "duration_ms" in r

    @pytest.mark.asyncio
    async def test_dispatch_with_workspace_filter(self, handler):
        handler._zapier.dispatch_event = AsyncMock(return_value=[])
        handler._n8n.dispatch_event = AsyncMock(return_value=[])

        body = {
            "event_type": "debate.completed",
            "payload": {},
            "workspace_id": "ws-filter",
        }
        mock_http = _make_mock_handler(body)
        result = await handler.handle_post("/api/v1/webhooks/dispatch", {}, mock_http)

        assert _status(result) == 200
        # Verify workspace_id was forwarded
        for call in handler._zapier.dispatch_event.call_args_list:
            assert call.kwargs.get("workspace_id") == "ws-filter"


# ============================================================================
# DELETE /api/v1/webhooks/{id} - Remove subscription
# ============================================================================


class TestUnsubscribe:
    """Tests for DELETE /api/v1/webhooks/{id}."""

    def test_unsubscribe_success(self, handler):
        sub = _make_subscription(sub_id="sub-del")
        handler._zapier._subscriptions[sub.id] = sub

        mock_http = _make_mock_handler()

        # Patch asyncio.new_event_loop used by _unsubscribe
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = True

        with patch("asyncio.new_event_loop", return_value=mock_loop):
            result = handler.handle_delete("/api/v1/webhooks/sub-del", {}, mock_http)

        assert _status(result) == 200
        data = _body(result)["data"]
        assert "removed" in data["message"].lower()

    def test_unsubscribe_not_found(self, handler):
        mock_http = _make_mock_handler()

        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = False

        with patch("asyncio.new_event_loop", return_value=mock_loop):
            result = handler.handle_delete("/api/v1/webhooks/nonexistent", {}, mock_http)

        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()

    def test_unsubscribe_unhandled_path(self, handler):
        mock_http = _make_mock_handler()
        result = handler.handle_delete("/api/v1/debates/abc", {}, mock_http)
        assert result is None


# ============================================================================
# POST - unhandled path returns None
# ============================================================================


class TestPostUnhandled:
    """Tests for POST requests on paths not handled by this handler."""

    @pytest.mark.asyncio
    async def test_unhandled_post_returns_none(self, handler):
        mock_http = _make_mock_handler()
        result = await handler.handle_post("/api/v1/debates", {}, mock_http)
        assert result is None


# ============================================================================
# RBAC permission checks
# ============================================================================


class TestRBACPermissions:
    """Test that correct RBAC permissions are checked for each route."""

    @pytest.fixture
    def handler_with_rbac(self, server_context):
        """Create handler with RBAC tracking (not bypassed)."""
        h = AutomationHandler(server_context)
        h._check_rbac_permission = MagicMock(return_value=None)
        return h

    def test_list_webhooks_checks_read_permission(self, handler_with_rbac):
        h = handler_with_rbac
        mock_http = _make_mock_handler()
        h.handle_get("/api/v1/webhooks", {}, mock_http)
        h._check_rbac_permission.assert_called_with(mock_http, "webhooks.read")

    def test_get_webhook_checks_read_permission(self, handler_with_rbac):
        h = handler_with_rbac
        sub = _make_subscription(sub_id="sub-rbac")
        h._zapier._subscriptions[sub.id] = sub

        mock_http = _make_mock_handler()
        h.handle_get("/api/v1/webhooks/sub-rbac", {}, mock_http)
        h._check_rbac_permission.assert_called_with(mock_http, "webhooks.read", "sub-rbac")

    def test_list_events_checks_read_permission(self, handler_with_rbac):
        h = handler_with_rbac
        mock_http = _make_mock_handler()
        h.handle_get("/api/v1/webhooks/events", {}, mock_http)
        h._check_rbac_permission.assert_called_with(mock_http, "webhooks.read")

    def test_list_platforms_checks_read_permission(self, handler_with_rbac):
        h = handler_with_rbac
        mock_http = _make_mock_handler()
        h.handle_get("/api/v1/webhooks/platforms", {}, mock_http)
        h._check_rbac_permission.assert_called_with(mock_http, "webhooks.read")

    def test_n8n_node_checks_admin_permission(self, handler_with_rbac):
        h = handler_with_rbac
        mock_http = _make_mock_handler()
        h.handle_get("/api/v1/n8n/node", {}, mock_http)
        h._check_rbac_permission.assert_called_with(mock_http, "connectors.authorize")

    def test_n8n_credentials_checks_admin_permission(self, handler_with_rbac):
        h = handler_with_rbac
        mock_http = _make_mock_handler()
        h.handle_get("/api/v1/n8n/credentials", {}, mock_http)
        h._check_rbac_permission.assert_called_with(mock_http, "connectors.authorize")

    def test_n8n_trigger_checks_admin_permission(self, handler_with_rbac):
        h = handler_with_rbac
        mock_http = _make_mock_handler()
        h.handle_get("/api/v1/n8n/trigger", {}, mock_http)
        h._check_rbac_permission.assert_called_with(mock_http, "connectors.authorize")

    @pytest.mark.asyncio
    async def test_subscribe_checks_create_permission(self, handler_with_rbac):
        h = handler_with_rbac
        body = {
            "webhook_url": "https://hooks.example.com/cb",
            "events": ["debate.completed"],
        }
        mock_http = _make_mock_handler(body)

        with patch(
            "aragora.server.handlers.integrations.automation.validate_webhook_url",
            return_value=(True, None),
        ):
            await h.handle_post("/api/v1/webhooks/subscribe", {}, mock_http)

        h._check_rbac_permission.assert_called_with(mock_http, "webhooks.create")

    @pytest.mark.asyncio
    async def test_dispatch_checks_all_permission(self, handler_with_rbac):
        h = handler_with_rbac
        h._zapier.dispatch_event = AsyncMock(return_value=[])
        h._n8n.dispatch_event = AsyncMock(return_value=[])

        body = {"event_type": "test.event", "payload": {}}
        mock_http = _make_mock_handler(body)
        await h.handle_post("/api/v1/webhooks/dispatch", {}, mock_http)

        h._check_rbac_permission.assert_called_with(mock_http, "webhooks.all")

    def test_delete_checks_delete_permission(self, handler_with_rbac):
        h = handler_with_rbac
        mock_http = _make_mock_handler()

        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = False

        with patch("asyncio.new_event_loop", return_value=mock_loop):
            h.handle_delete("/api/v1/webhooks/sub-xyz", {}, mock_http)

        h._check_rbac_permission.assert_called_with(mock_http, "webhooks.delete", "sub-xyz")

    def test_rbac_denied_returns_error(self, server_context):
        """When RBAC denies access, the error result is returned."""
        from aragora.server.handlers.base import error_response

        h = AutomationHandler(server_context)
        denied = error_response("Permission denied", status=403)
        h._check_rbac_permission = MagicMock(return_value=denied)

        mock_http = _make_mock_handler()
        result = h.handle_get("/api/v1/webhooks", {}, mock_http)

        assert _status(result) == 403


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_handler_initializes_connectors(self, handler):
        """Handler should have zapier, n8n, and generic connectors."""
        assert "zapier" in handler._connectors
        assert "n8n" in handler._connectors
        assert "generic" in handler._connectors

    def test_resource_type_is_webhook(self, handler):
        assert handler.RESOURCE_TYPE == "webhook"

    def test_routes_list(self, handler):
        """Verify ROUTES contains all expected static paths."""
        expected = {
            "/api/v1/webhooks",
            "/api/v1/webhooks/subscribe",
            "/api/v1/webhooks/events",
            "/api/v1/webhooks/dispatch",
            "/api/v1/webhooks/platforms",
            "/api/v1/n8n/node",
            "/api/v1/n8n/credentials",
            "/api/v1/n8n/trigger",
        }
        assert set(handler.ROUTES) == expected

    @pytest.mark.asyncio
    async def test_subscribe_empty_events_list(self, handler):
        """Passing empty events list should be rejected."""
        body = {
            "webhook_url": "https://hooks.example.com/cb",
            "events": [],
        }
        mock_http = _make_mock_handler(body)

        with patch(
            "aragora.server.handlers.integrations.automation.validate_webhook_url",
            return_value=(True, None),
        ):
            result = await handler.handle_post("/api/v1/webhooks/subscribe", {}, mock_http)

        assert _status(result) == 400
        assert "events" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_test_webhook_id_extraction(self, handler):
        """Verify webhook ID is correctly extracted from /webhooks/{id}/test path."""
        sub = _make_subscription(sub_id="exact-id-check")
        handler._zapier._subscriptions[sub.id] = sub
        handler._zapier.test_subscription = AsyncMock(return_value=True)

        mock_http = _make_mock_handler()
        result = await handler.handle_post("/api/v1/webhooks/exact-id-check/test", {}, mock_http)

        assert _status(result) == 200
        handler._zapier.test_subscription.assert_called_once_with(sub)

    @pytest.mark.asyncio
    async def test_dispatch_handles_sync_connector(self, handler):
        """Test that dispatch handles connectors that return lists directly (non-awaitable)."""
        delivery = _make_delivery_result(success=True)
        # Return a plain list (not a coroutine) to test the sync branch
        handler._zapier.dispatch_event = MagicMock(return_value=[delivery])
        handler._n8n.dispatch_event = MagicMock(return_value=[])

        body = {"event_type": "test.event", "payload": {}}
        mock_http = _make_mock_handler(body)
        result = await handler.handle_post("/api/v1/webhooks/dispatch", {}, mock_http)

        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["dispatched"] >= 1

    @pytest.mark.asyncio
    async def test_dispatch_handles_non_list_result(self, handler):
        """Test that dispatch handles a connector returning a single result (not a list)."""
        delivery = _make_delivery_result(success=True)
        # Return a single result object (not wrapped in list)
        handler._zapier.dispatch_event = MagicMock(return_value=delivery)
        handler._n8n.dispatch_event = MagicMock(return_value=[])

        body = {"event_type": "test.event", "payload": {}}
        mock_http = _make_mock_handler(body)
        result = await handler.handle_post("/api/v1/webhooks/dispatch", {}, mock_http)

        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["dispatched"] >= 1

    def test_get_request_body_no_request_attr(self, handler):
        """_get_request_body returns empty dict when handler lacks request."""
        mock_http = MagicMock(spec=[])  # No attributes
        result = handler._get_request_body(mock_http)
        assert result == {}

    def test_get_request_body_string_body(self, handler):
        """_get_request_body handles string body."""
        mock_http = MagicMock()
        mock_http.request.body = '{"key": "value"}'
        result = handler._get_request_body(mock_http)
        assert result == {"key": "value"}

    def test_get_request_body_dict_body(self, handler):
        """_get_request_body handles dict body (already parsed)."""
        mock_http = MagicMock()
        mock_http.request.body = {"already": "parsed"}
        result = handler._get_request_body(mock_http)
        assert result == {"already": "parsed"}
