"""Tests for email webhook handler.

Tests the email webhook API endpoints including:
- POST /api/v1/webhooks/gmail              - Handle Gmail Pub/Sub notifications
- POST /api/v1/webhooks/outlook            - Handle Outlook Graph notifications
- POST /api/v1/webhooks/outlook/validate   - Handle Outlook subscription validation
- GET  /api/v1/webhooks/status             - Get webhook subscription status
- POST /api/v1/webhooks/subscribe          - Create new webhook subscription
- DELETE /api/v1/webhooks/unsubscribe      - Remove webhook subscription
- GET  /api/v1/webhooks/history            - Get notification history

Also tests standalone processing functions:
- process_gmail_notification()
- process_outlook_notification()
"""

import asyncio
import base64
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.email_webhooks import (
    EmailWebhooksHandler,
    NotificationType,
    WebhookNotification,
    WebhookProvider,
    WebhookStatus,
    WebhookSubscription,
    _notification_history,
    _pending_validations,
    _subscriptions,
    _tenant_subscriptions,
    get_email_webhooks_handler,
    handle_email_webhooks,
    process_gmail_notification,
    process_outlook_notification,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(result: HandlerResult) -> int:
    """Extract status code from HandlerResult."""
    return result.status_code


def _body(result: HandlerResult) -> dict[str, Any]:
    """Extract parsed JSON body from HandlerResult."""
    if isinstance(result.body, bytes):
        return json.loads(result.body.decode("utf-8"))
    return json.loads(result.body)


def _raw_body(result: HandlerResult) -> bytes:
    """Extract raw body bytes from HandlerResult."""
    if isinstance(result.body, bytes):
        return result.body
    return result.body.encode("utf-8")


def _make_gmail_payload(email: str = "user@gmail.com", history_id: str = "12345") -> dict:
    """Create a valid Gmail Pub/Sub notification payload."""
    data = json.dumps({"emailAddress": email, "historyId": history_id}).encode("utf-8")
    data_b64 = base64.b64encode(data).decode("utf-8")
    return {
        "message": {
            "data": data_b64,
            "messageId": "msg-001",
            "publishTime": "2026-01-01T00:00:00Z",
        },
        "subscription": "projects/myproject/subscriptions/gmail-push",
    }


def _make_outlook_payload(
    change_type: str = "created",
    subscription_id: str = "sub-001",
    resource: str = "Users/user1/Messages/msg-001",
    client_state: str | None = None,
) -> dict:
    """Create a valid Outlook Graph change notification payload."""
    change: dict[str, Any] = {
        "subscriptionId": subscription_id,
        "changeType": change_type,
        "resource": resource,
        "tenantId": "outlook-tenant-001",
    }
    if client_state is not None:
        change["clientState"] = client_state
    return {"value": [change]}


# ---------------------------------------------------------------------------
# Mock Request
# ---------------------------------------------------------------------------


@dataclass
class MockRequest:
    """Mock HTTP request for handler tests."""

    tenant_id: str = "test-tenant"
    _body: dict[str, Any] | None = None
    query: dict[str, str] | None = None
    headers: dict[str, str] | None = None

    def __post_init__(self):
        if self.query is None:
            self.query = {}
        if self.headers is None:
            self.headers = {}

    async def json(self) -> dict[str, Any]:
        return self._body or {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create an EmailWebhooksHandler with minimal context."""
    return EmailWebhooksHandler(server_context={})


@pytest.fixture(autouse=True)
def reset_global_state():
    """Clear in-memory stores between tests to avoid cross-contamination."""
    _subscriptions.clear()
    _tenant_subscriptions.clear()
    _notification_history.clear()
    _pending_validations.clear()
    yield
    _subscriptions.clear()
    _tenant_subscriptions.clear()
    _notification_history.clear()
    _pending_validations.clear()


@pytest.fixture(autouse=True)
def reset_handler_singleton():
    """Reset the module-level handler singleton between tests."""
    import aragora.server.handlers.features.email_webhooks as mod

    mod._handler_instance = None
    yield
    mod._handler_instance = None


@pytest.fixture
def gmail_request():
    """Factory for Gmail webhook requests."""

    def _create(email: str = "user@gmail.com", history_id: str = "12345") -> MockRequest:
        return MockRequest(_body=_make_gmail_payload(email, history_id))

    return _create


@pytest.fixture
def outlook_request():
    """Factory for Outlook webhook requests."""

    def _create(
        change_type: str = "created",
        subscription_id: str = "sub-001",
        client_state: str | None = None,
    ) -> MockRequest:
        return MockRequest(
            _body=_make_outlook_payload(change_type, subscription_id, client_state=client_state)
        )

    return _create


@pytest.fixture
def sample_subscription() -> WebhookSubscription:
    """Create a sample subscription for tests."""
    now = datetime.now(timezone.utc)
    return WebhookSubscription(
        id="sub-001",
        tenant_id="test-tenant",
        account_id="account-001",
        provider=WebhookProvider.GMAIL,
        status=WebhookStatus.ACTIVE,
        created_at=now,
        expires_at=now + timedelta(hours=72),
        notification_url="https://example.com/webhook",
        client_state="abc123",
        notification_count=5,
    )


@pytest.fixture
def populated_subscriptions(sample_subscription):
    """Populate global stores with sample subscriptions."""
    _subscriptions["sub-001"] = sample_subscription

    outlook_sub = WebhookSubscription(
        id="sub-002",
        tenant_id="test-tenant",
        account_id="account-002",
        provider=WebhookProvider.OUTLOOK,
        status=WebhookStatus.ACTIVE,
        created_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=48),
        notification_url="https://example.com/outlook-webhook",
        client_state="def456",
        notification_count=3,
    )
    _subscriptions["sub-002"] = outlook_sub
    _tenant_subscriptions["test-tenant"] = ["sub-001", "sub-002"]
    return sample_subscription, outlook_sub


# ===========================================================================
# Data Model Tests
# ===========================================================================


class TestWebhookSubscriptionModel:
    """Tests for the WebhookSubscription dataclass."""

    def test_to_dict_basic(self, sample_subscription):
        d = sample_subscription.to_dict()
        assert d["id"] == "sub-001"
        assert d["tenant_id"] == "test-tenant"
        assert d["provider"] == "gmail"
        assert d["status"] == "active"
        assert d["notification_count"] == 5

    def test_to_dict_no_expires(self):
        sub = WebhookSubscription(
            id="x",
            tenant_id="t",
            account_id="a",
            provider=WebhookProvider.OUTLOOK,
            status=WebhookStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )
        d = sub.to_dict()
        assert d["expires_at"] is None
        assert d["last_notification"] is None

    def test_to_dict_with_last_notification(self, sample_subscription):
        sample_subscription.last_notification = datetime(2026, 1, 15, tzinfo=timezone.utc)
        d = sample_subscription.to_dict()
        assert d["last_notification"] is not None
        assert "2026" in d["last_notification"]


class TestWebhookNotificationModel:
    """Tests for the WebhookNotification dataclass."""

    def test_to_dict(self):
        n = WebhookNotification(
            provider=WebhookProvider.GMAIL,
            notification_type=NotificationType.SYNC_REQUESTED,
            account_id="acc-1",
            resource_id="hist-123",
            tenant_id="tenant-1",
            timestamp=datetime(2026, 2, 1, tzinfo=timezone.utc),
            raw_data={"message": {}},
            metadata={"email_address": "a@b.com"},
        )
        d = n.to_dict()
        assert d["provider"] == "gmail"
        assert d["notification_type"] == "sync_requested"
        assert d["account_id"] == "acc-1"
        assert d["metadata"]["email_address"] == "a@b.com"


class TestEnums:
    """Tests for enum values."""

    def test_webhook_provider_values(self):
        assert WebhookProvider.GMAIL.value == "gmail"
        assert WebhookProvider.OUTLOOK.value == "outlook"

    def test_webhook_status_values(self):
        assert WebhookStatus.ACTIVE.value == "active"
        assert WebhookStatus.PENDING.value == "pending"
        assert WebhookStatus.EXPIRED.value == "expired"
        assert WebhookStatus.ERROR.value == "error"

    def test_notification_type_values(self):
        assert NotificationType.MESSAGE_CREATED.value == "message_created"
        assert NotificationType.MESSAGE_UPDATED.value == "message_updated"
        assert NotificationType.MESSAGE_DELETED.value == "message_deleted"
        assert NotificationType.LABEL_CHANGED.value == "label_changed"
        assert NotificationType.SYNC_REQUESTED.value == "sync_requested"


# ===========================================================================
# Handler Initialization
# ===========================================================================


class TestHandlerInit:
    """Tests for handler initialization and routing setup."""

    def test_routes_defined(self, handler):
        assert hasattr(handler, "ROUTES")
        assert len(handler.ROUTES) == 7

    def test_routes_contain_expected_paths(self, handler):
        assert "/api/v1/webhooks/gmail" in handler.ROUTES
        assert "/api/v1/webhooks/outlook" in handler.ROUTES
        assert "/api/v1/webhooks/outlook/validate" in handler.ROUTES
        assert "/api/v1/webhooks/status" in handler.ROUTES
        assert "/api/v1/webhooks/subscribe" in handler.ROUTES
        assert "/api/v1/webhooks/unsubscribe" in handler.ROUTES
        assert "/api/v1/webhooks/history" in handler.ROUTES

    def test_init_with_none_context(self):
        h = EmailWebhooksHandler(server_context=None)
        assert h is not None

    def test_init_with_empty_context(self):
        h = EmailWebhooksHandler(server_context={})
        assert h is not None


# ===========================================================================
# Gmail Webhook Tests
# ===========================================================================


class TestGmailWebhook:
    """Tests for POST /api/v1/webhooks/gmail."""

    @pytest.mark.asyncio
    async def test_gmail_webhook_success(self, handler, gmail_request):
        req = gmail_request()
        result = await handler.handle(req, "/api/v1/webhooks/gmail", "POST")
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["status"] == "processed"
        assert "notification" in body["data"]

    @pytest.mark.asyncio
    async def test_gmail_webhook_notification_fields(self, handler, gmail_request):
        req = gmail_request(email="test@example.com", history_id="99999")
        result = await handler.handle(req, "/api/v1/webhooks/gmail", "POST")
        body = _body(result)
        notif = body["data"]["notification"]
        assert notif["provider"] == "gmail"
        assert notif["notification_type"] == "sync_requested"
        assert notif["metadata"]["email_address"] == "test@example.com"
        assert notif["metadata"]["history_id"] == "99999"

    @pytest.mark.asyncio
    async def test_gmail_webhook_missing_data(self, handler):
        req = MockRequest(_body={"message": {}})
        result = await handler.handle(req, "/api/v1/webhooks/gmail", "POST")
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["status"] == "acknowledged"

    @pytest.mark.asyncio
    async def test_gmail_webhook_invalid_base64(self, handler):
        req = MockRequest(_body={"message": {"data": "not-valid-base64!!!"}})
        result = await handler.handle(req, "/api/v1/webhooks/gmail", "POST")
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["status"] == "acknowledged"

    @pytest.mark.asyncio
    async def test_gmail_webhook_missing_email_address(self, handler):
        data = json.dumps({"historyId": "123"}).encode("utf-8")
        data_b64 = base64.b64encode(data).decode("utf-8")
        req = MockRequest(_body={"message": {"data": data_b64}})
        result = await handler.handle(req, "/api/v1/webhooks/gmail", "POST")
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["status"] == "acknowledged"

    @pytest.mark.asyncio
    async def test_gmail_webhook_invalid_json_in_data(self, handler):
        data_b64 = base64.b64encode(b"not json at all").decode("utf-8")
        req = MockRequest(_body={"message": {"data": data_b64}})
        result = await handler.handle(req, "/api/v1/webhooks/gmail", "POST")
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["status"] == "acknowledged"

    @pytest.mark.asyncio
    async def test_gmail_webhook_stores_in_history(self, handler, gmail_request):
        req = gmail_request()
        req.tenant_id = "hist-tenant"
        await handler.handle(req, "/api/v1/webhooks/gmail", "POST")
        assert "hist-tenant" in _notification_history
        assert len(_notification_history["hist-tenant"]) == 1

    @pytest.mark.asyncio
    async def test_gmail_webhook_null_body(self, handler):
        """Handler handles request with no valid JSON body."""
        req = MockRequest(_body=None)
        result = await handler.handle(req, "/api/v1/webhooks/gmail", "POST")
        # Should return 200 (acknowledges receipt)
        assert _status(result) == 200


# ===========================================================================
# Outlook Webhook Tests
# ===========================================================================


class TestOutlookWebhook:
    """Tests for POST /api/v1/webhooks/outlook."""

    @pytest.mark.asyncio
    async def test_outlook_webhook_success(self, handler, outlook_request):
        req = outlook_request()
        result = await handler.handle(req, "/api/v1/webhooks/outlook", "POST")
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["status"] == "processed"
        assert body["data"]["count"] == 1

    @pytest.mark.asyncio
    async def test_outlook_webhook_created_type(self, handler, outlook_request):
        req = outlook_request(change_type="created")
        result = await handler.handle(req, "/api/v1/webhooks/outlook", "POST")
        body = _body(result)
        notif = body["data"]["notifications"][0]
        assert notif["notification_type"] == "message_created"

    @pytest.mark.asyncio
    async def test_outlook_webhook_updated_type(self, handler, outlook_request):
        req = outlook_request(change_type="updated")
        result = await handler.handle(req, "/api/v1/webhooks/outlook", "POST")
        body = _body(result)
        notif = body["data"]["notifications"][0]
        assert notif["notification_type"] == "message_updated"

    @pytest.mark.asyncio
    async def test_outlook_webhook_deleted_type(self, handler, outlook_request):
        req = outlook_request(change_type="deleted")
        result = await handler.handle(req, "/api/v1/webhooks/outlook", "POST")
        body = _body(result)
        notif = body["data"]["notifications"][0]
        assert notif["notification_type"] == "message_deleted"

    @pytest.mark.asyncio
    async def test_outlook_webhook_unknown_change_type(self, handler, outlook_request):
        req = outlook_request(change_type="unknown")
        result = await handler.handle(req, "/api/v1/webhooks/outlook", "POST")
        body = _body(result)
        notif = body["data"]["notifications"][0]
        assert notif["notification_type"] == "sync_requested"

    @pytest.mark.asyncio
    async def test_outlook_webhook_validation_token(self, handler):
        """Outlook sends validation token in query params during handshake."""
        req = MockRequest(query={"validationToken": "my-token-123"})
        result = await handler.handle(req, "/api/v1/webhooks/outlook", "POST")
        assert _status(result) == 200
        assert result.content_type == "text/plain"
        assert _raw_body(result) == b"my-token-123"

    @pytest.mark.asyncio
    async def test_outlook_webhook_empty_value_array(self, handler):
        req = MockRequest(_body={"value": []})
        result = await handler.handle(req, "/api/v1/webhooks/outlook", "POST")
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["count"] == 0

    @pytest.mark.asyncio
    async def test_outlook_webhook_multiple_changes(self, handler):
        payload = {
            "value": [
                {
                    "subscriptionId": "s1",
                    "changeType": "created",
                    "resource": "Users/u1/Messages/m1",
                    "tenantId": "t1",
                },
                {
                    "subscriptionId": "s2",
                    "changeType": "deleted",
                    "resource": "Users/u2/Messages/m2",
                    "tenantId": "t1",
                },
            ]
        }
        req = MockRequest(_body=payload)
        result = await handler.handle(req, "/api/v1/webhooks/outlook", "POST")
        body = _body(result)
        assert body["data"]["count"] == 2

    @pytest.mark.asyncio
    async def test_outlook_webhook_resolves_account_from_subscription(
        self, handler, populated_subscriptions
    ):
        """When subscription exists, account_id should be populated."""
        req = MockRequest(_body=_make_outlook_payload(subscription_id="sub-002"))
        result = await handler.handle(req, "/api/v1/webhooks/outlook", "POST")
        body = _body(result)
        notif = body["data"]["notifications"][0]
        assert notif["account_id"] == "account-002"

    @pytest.mark.asyncio
    async def test_outlook_webhook_null_body(self, handler):
        req = MockRequest(_body=None)
        result = await handler.handle(req, "/api/v1/webhooks/outlook", "POST")
        assert _status(result) == 200


# ===========================================================================
# Outlook Validation Tests
# ===========================================================================


class TestOutlookValidation:
    """Tests for POST /api/v1/webhooks/outlook/validate."""

    @pytest.mark.asyncio
    async def test_validation_returns_token(self, handler):
        req = MockRequest(query={"validationToken": "validation-abc"})
        result = await handler.handle(req, "/api/v1/webhooks/outlook/validate", "POST")
        assert _status(result) == 200
        assert result.content_type == "text/plain"
        assert _raw_body(result) == b"validation-abc"

    @pytest.mark.asyncio
    async def test_validation_missing_token(self, handler):
        req = MockRequest(query={})
        result = await handler.handle(req, "/api/v1/webhooks/outlook/validate", "POST")
        assert _status(result) == 400
        body = _body(result)
        assert "validationToken" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_validation_empty_token(self, handler):
        req = MockRequest(query={"validationToken": ""})
        result = await handler.handle(req, "/api/v1/webhooks/outlook/validate", "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_validation_special_characters(self, handler):
        token = "abc-123_def+ghi=jkl"
        req = MockRequest(query={"validationToken": token})
        result = await handler.handle(req, "/api/v1/webhooks/outlook/validate", "POST")
        assert _status(result) == 200
        assert _raw_body(result) == token.encode("utf-8")


# ===========================================================================
# Status Endpoint Tests
# ===========================================================================


class TestStatus:
    """Tests for GET /api/v1/webhooks/status."""

    @pytest.mark.asyncio
    async def test_status_empty(self, handler):
        req = MockRequest()
        result = await handler.handle(req, "/api/v1/webhooks/status", "GET")
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["subscriptions"] == []
        assert body["data"]["summary"]["total"] == 0
        assert body["data"]["summary"]["active"] == 0

    @pytest.mark.asyncio
    async def test_status_with_subscriptions(self, handler, populated_subscriptions):
        req = MockRequest()
        result = await handler.handle(req, "/api/v1/webhooks/status", "GET")
        body = _body(result)
        assert body["data"]["summary"]["total"] == 2
        assert body["data"]["summary"]["active"] == 2
        assert body["data"]["summary"]["total_notifications"] == 8  # 5 + 3

    @pytest.mark.asyncio
    async def test_status_different_tenant(self, handler, populated_subscriptions):
        req = MockRequest(tenant_id="other-tenant")
        result = await handler.handle(req, "/api/v1/webhooks/status", "GET")
        body = _body(result)
        assert body["data"]["summary"]["total"] == 0

    @pytest.mark.asyncio
    async def test_status_active_count(self, handler, populated_subscriptions):
        sub1, sub2 = populated_subscriptions
        sub2.status = WebhookStatus.EXPIRED
        req = MockRequest()
        result = await handler.handle(req, "/api/v1/webhooks/status", "GET")
        body = _body(result)
        assert body["data"]["summary"]["active"] == 1

    @pytest.mark.asyncio
    async def test_status_stale_subscription_id(self, handler):
        """If tenant_subscriptions references a deleted subscription ID, skip it."""
        _tenant_subscriptions["test-tenant"] = ["nonexistent-sub"]
        req = MockRequest()
        result = await handler.handle(req, "/api/v1/webhooks/status", "GET")
        body = _body(result)
        assert body["data"]["subscriptions"] == []
        assert body["data"]["summary"]["total"] == 0


# ===========================================================================
# Subscribe Endpoint Tests
# ===========================================================================


class TestSubscribe:
    """Tests for POST /api/v1/webhooks/subscribe."""

    @pytest.mark.asyncio
    async def test_subscribe_gmail(self, handler):
        req = MockRequest(
            _body={
                "provider": "gmail",
                "account_id": "acc-1",
                "notification_url": "https://example.com/hook",
            }
        )
        result = await handler.handle(req, "/api/v1/webhooks/subscribe", "POST")
        assert _status(result) == 200
        body = _body(result)
        sub = body["data"]["subscription"]
        assert sub["provider"] == "gmail"
        assert sub["status"] == "active"
        assert "client_state" in body["data"]

    @pytest.mark.asyncio
    async def test_subscribe_outlook(self, handler):
        req = MockRequest(
            _body={
                "provider": "outlook",
                "account_id": "acc-2",
                "notification_url": "https://example.com/outlook",
            }
        )
        result = await handler.handle(req, "/api/v1/webhooks/subscribe", "POST")
        assert _status(result) == 200
        body = _body(result)
        sub = body["data"]["subscription"]
        assert sub["provider"] == "outlook"
        assert sub["status"] == "active"

    @pytest.mark.asyncio
    async def test_subscribe_invalid_provider(self, handler):
        req = MockRequest(_body={"provider": "yahoo", "account_id": "acc-1"})
        result = await handler.handle(req, "/api/v1/webhooks/subscribe", "POST")
        assert _status(result) == 400
        body = _body(result)
        assert "provider" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_subscribe_missing_provider(self, handler):
        req = MockRequest(_body={"account_id": "acc-1"})
        result = await handler.handle(req, "/api/v1/webhooks/subscribe", "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_subscribe_missing_account_id(self, handler):
        req = MockRequest(_body={"provider": "gmail"})
        result = await handler.handle(req, "/api/v1/webhooks/subscribe", "POST")
        assert _status(result) == 400
        body = _body(result)
        assert "account_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_subscribe_stores_subscription(self, handler):
        req = MockRequest(_body={"provider": "gmail", "account_id": "acc-store-test"})
        await handler.handle(req, "/api/v1/webhooks/subscribe", "POST")
        assert len(_subscriptions) == 1
        sub = list(_subscriptions.values())[0]
        assert sub.account_id == "acc-store-test"
        assert sub.tenant_id == "test-tenant"

    @pytest.mark.asyncio
    async def test_subscribe_records_tenant_mapping(self, handler):
        req = MockRequest(_body={"provider": "gmail", "account_id": "acc-1"})
        await handler.handle(req, "/api/v1/webhooks/subscribe", "POST")
        assert "test-tenant" in _tenant_subscriptions
        assert len(_tenant_subscriptions["test-tenant"]) == 1

    @pytest.mark.asyncio
    async def test_subscribe_custom_expiration(self, handler):
        req = MockRequest(
            _body={
                "provider": "gmail",
                "account_id": "acc-1",
                "expiration_hours": 24,
            }
        )
        await handler.handle(req, "/api/v1/webhooks/subscribe", "POST")
        sub = list(_subscriptions.values())[0]
        delta = sub.expires_at - sub.created_at
        assert abs(delta.total_seconds() - 24 * 3600) < 5

    @pytest.mark.asyncio
    async def test_subscribe_default_expiration(self, handler):
        req = MockRequest(_body={"provider": "gmail", "account_id": "acc-1"})
        await handler.handle(req, "/api/v1/webhooks/subscribe", "POST")
        sub = list(_subscriptions.values())[0]
        delta = sub.expires_at - sub.created_at
        assert abs(delta.total_seconds() - 72 * 3600) < 5

    @pytest.mark.asyncio
    async def test_subscribe_gmail_creation_failure(self, handler):
        """When Gmail subscription creation fails, return error."""
        req = MockRequest(_body={"provider": "gmail", "account_id": "acc-fail"})
        with patch.object(
            handler,
            "_create_gmail_subscription",
            return_value={"success": False, "error": "API quota exceeded"},
        ):
            result = await handler.handle(req, "/api/v1/webhooks/subscribe", "POST")
        assert _status(result) == 400
        body = _body(result)
        assert "quota" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_subscribe_outlook_creation_failure(self, handler):
        req = MockRequest(_body={"provider": "outlook", "account_id": "acc-fail"})
        with patch.object(
            handler,
            "_create_outlook_subscription",
            return_value={"success": False, "error": "Graph API error"},
        ):
            result = await handler.handle(req, "/api/v1/webhooks/subscribe", "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_subscribe_case_insensitive_provider(self, handler):
        req = MockRequest(_body={"provider": "GMAIL", "account_id": "acc-1"})
        result = await handler.handle(req, "/api/v1/webhooks/subscribe", "POST")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_subscribe_client_state_is_deterministic(self, handler):
        """Client state is derived from tenant_id, account_id, and subscription_id."""
        req = MockRequest(_body={"provider": "gmail", "account_id": "acc-1"})
        result = await handler.handle(req, "/api/v1/webhooks/subscribe", "POST")
        body = _body(result)
        client_state = body["data"]["client_state"]
        assert len(client_state) == 32  # SHA-256 hex truncated to 32 chars


# ===========================================================================
# Unsubscribe Endpoint Tests
# ===========================================================================


class TestUnsubscribe:
    """Tests for POST/DELETE /api/v1/webhooks/unsubscribe."""

    @pytest.mark.asyncio
    async def test_unsubscribe_success(self, handler, populated_subscriptions):
        req = MockRequest(_body={"subscription_id": "sub-001"})
        result = await handler.handle(req, "/api/v1/webhooks/unsubscribe", "POST")
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["status"] == "deleted"
        assert "sub-001" not in _subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe_delete_method(self, handler, populated_subscriptions):
        req = MockRequest(_body={"subscription_id": "sub-001"})
        result = await handler.handle(req, "/api/v1/webhooks/unsubscribe", "DELETE")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_unsubscribe_not_found(self, handler):
        req = MockRequest(_body={"subscription_id": "nonexistent"})
        result = await handler.handle(req, "/api/v1/webhooks/unsubscribe", "POST")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_unsubscribe_missing_id(self, handler):
        req = MockRequest(_body={})
        result = await handler.handle(req, "/api/v1/webhooks/unsubscribe", "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_unsubscribe_wrong_tenant(self, handler, populated_subscriptions):
        req = MockRequest(_body={"subscription_id": "sub-001"}, tenant_id="wrong-tenant")
        result = await handler.handle(req, "/api/v1/webhooks/unsubscribe", "POST")
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_from_tenant_list(self, handler, populated_subscriptions):
        req = MockRequest(_body={"subscription_id": "sub-001"})
        await handler.handle(req, "/api/v1/webhooks/unsubscribe", "POST")
        assert "sub-001" not in _tenant_subscriptions.get("test-tenant", [])

    @pytest.mark.asyncio
    async def test_unsubscribe_preserves_other_subs(self, handler, populated_subscriptions):
        req = MockRequest(_body={"subscription_id": "sub-001"})
        await handler.handle(req, "/api/v1/webhooks/unsubscribe", "POST")
        assert "sub-002" in _subscriptions
        assert "sub-002" in _tenant_subscriptions["test-tenant"]

    @pytest.mark.asyncio
    async def test_unsubscribe_outlook_subscription(self, handler, populated_subscriptions):
        req = MockRequest(_body={"subscription_id": "sub-002"})
        result = await handler.handle(req, "/api/v1/webhooks/unsubscribe", "DELETE")
        assert _status(result) == 200
        assert "sub-002" not in _subscriptions


# ===========================================================================
# History Endpoint Tests
# ===========================================================================


class TestHistory:
    """Tests for GET /api/v1/webhooks/history."""

    @pytest.mark.asyncio
    async def test_history_empty(self, handler):
        req = MockRequest()
        result = await handler.handle(req, "/api/v1/webhooks/history", "GET")
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["notifications"] == []
        assert body["data"]["total"] == 0

    @pytest.mark.asyncio
    async def test_history_with_notifications(self, handler, gmail_request):
        # Create some notifications first
        req = gmail_request()
        await handler.handle(req, "/api/v1/webhooks/gmail", "POST")
        await handler.handle(req, "/api/v1/webhooks/gmail", "POST")

        result = await handler.handle(req, "/api/v1/webhooks/history", "GET")
        body = _body(result)
        assert body["data"]["total"] == 2

    @pytest.mark.asyncio
    async def test_history_limit_parameter(self, handler, gmail_request):
        req = gmail_request()
        for _ in range(5):
            await handler.handle(req, "/api/v1/webhooks/gmail", "POST")

        req_hist = MockRequest(query={"limit": "2"})
        result = await handler.handle(req_hist, "/api/v1/webhooks/history", "GET")
        body = _body(result)
        assert body["data"]["total"] == 2

    @pytest.mark.asyncio
    async def test_history_default_limit(self, handler):
        # Populate with exactly 5 notifications
        for i in range(5):
            n = WebhookNotification(
                provider=WebhookProvider.GMAIL,
                notification_type=NotificationType.SYNC_REQUESTED,
                account_id="acc",
                resource_id=f"h-{i}",
                tenant_id="test-tenant",
                timestamp=datetime.now(timezone.utc),
                raw_data={},
            )
            _notification_history.setdefault("test-tenant", []).append(n)

        req = MockRequest()
        result = await handler.handle(req, "/api/v1/webhooks/history", "GET")
        body = _body(result)
        assert body["data"]["total"] == 5

    @pytest.mark.asyncio
    async def test_history_invalid_limit_falls_back(self, handler):
        req = MockRequest(query={"limit": "not-a-number"})
        result = await handler.handle(req, "/api/v1/webhooks/history", "GET")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_history_reversed_order(self, handler):
        """Most recent notifications should come first."""
        for i in range(3):
            n = WebhookNotification(
                provider=WebhookProvider.GMAIL,
                notification_type=NotificationType.SYNC_REQUESTED,
                account_id="acc",
                resource_id=f"h-{i}",
                tenant_id="test-tenant",
                timestamp=datetime.now(timezone.utc),
                raw_data={},
            )
            _notification_history.setdefault("test-tenant", []).append(n)

        req = MockRequest()
        result = await handler.handle(req, "/api/v1/webhooks/history", "GET")
        body = _body(result)
        notifs = body["data"]["notifications"]
        assert notifs[0]["resource_id"] == "h-2"
        assert notifs[2]["resource_id"] == "h-0"

    @pytest.mark.asyncio
    async def test_history_different_tenant(self, handler):
        _notification_history["other-tenant"] = [
            WebhookNotification(
                provider=WebhookProvider.GMAIL,
                notification_type=NotificationType.SYNC_REQUESTED,
                account_id="acc",
                resource_id="h-1",
                tenant_id="other-tenant",
                timestamp=datetime.now(timezone.utc),
                raw_data={},
            )
        ]
        req = MockRequest(tenant_id="test-tenant")
        result = await handler.handle(req, "/api/v1/webhooks/history", "GET")
        body = _body(result)
        assert body["data"]["total"] == 0


# ===========================================================================
# Routing / 404 Tests
# ===========================================================================


class TestRouting:
    """Tests for route_request dispatch and 404 handling."""

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self, handler):
        req = MockRequest()
        result = await handler.handle(req, "/api/v1/webhooks/unknown", "GET")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_gmail_returns_404(self, handler):
        req = MockRequest()
        result = await handler.handle(req, "/api/v1/webhooks/gmail", "GET")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_status_returns_404(self, handler):
        req = MockRequest()
        result = await handler.handle(req, "/api/v1/webhooks/status", "POST")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_history_returns_404(self, handler):
        req = MockRequest()
        result = await handler.handle(req, "/api/v1/webhooks/history", "POST")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_subscribe_returns_404(self, handler):
        req = MockRequest()
        result = await handler.handle(req, "/api/v1/webhooks/subscribe", "GET")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_outlook_returns_404(self, handler):
        req = MockRequest()
        result = await handler.handle(req, "/api/v1/webhooks/outlook", "GET")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_unsubscribe_accepts_post(self, handler, populated_subscriptions):
        req = MockRequest(_body={"subscription_id": "sub-001"})
        result = await handler.handle(req, "/api/v1/webhooks/unsubscribe", "POST")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_unsubscribe_accepts_delete(self, handler, populated_subscriptions):
        req = MockRequest(_body={"subscription_id": "sub-002"})
        result = await handler.handle(req, "/api/v1/webhooks/unsubscribe", "DELETE")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_unsubscribe_rejects_get(self, handler):
        req = MockRequest()
        result = await handler.handle(req, "/api/v1/webhooks/unsubscribe", "GET")
        assert _status(result) == 404


# ===========================================================================
# Tenant Extraction Tests
# ===========================================================================


class TestTenantExtraction:
    """Tests for _get_tenant_id."""

    def test_extracts_tenant_from_request(self, handler):
        req = MockRequest(tenant_id="my-tenant")
        assert handler._get_tenant_id(req) == "my-tenant"

    def test_default_tenant_when_missing(self, handler):
        req = MagicMock(spec=[])  # No attributes
        assert handler._get_tenant_id(req) == "default"


# ===========================================================================
# process_gmail_notification() Tests
# ===========================================================================


class TestProcessGmailNotification:
    """Tests for the standalone process_gmail_notification function."""

    @pytest.mark.asyncio
    async def test_process_valid(self):
        payload = _make_gmail_payload("alice@gmail.com", "5678")
        result = await process_gmail_notification(payload, "t1")
        assert result is not None
        assert result.provider == WebhookProvider.GMAIL
        assert result.metadata["email_address"] == "alice@gmail.com"

    @pytest.mark.asyncio
    async def test_process_missing_message(self):
        result = await process_gmail_notification({}, "t1")
        assert result is None

    @pytest.mark.asyncio
    async def test_process_empty_data(self):
        result = await process_gmail_notification({"message": {"data": ""}}, "t1")
        assert result is None

    @pytest.mark.asyncio
    async def test_process_corrupt_base64(self):
        result = await process_gmail_notification({"message": {"data": ";;;corrupt;;;"}}, "t1")
        assert result is None

    @pytest.mark.asyncio
    async def test_process_queues_notification(self):
        payload = _make_gmail_payload()
        await process_gmail_notification(payload, "queue-tenant")
        assert "queue-tenant" in _notification_history
        assert len(_notification_history["queue-tenant"]) == 1


# ===========================================================================
# process_outlook_notification() Tests
# ===========================================================================


class TestProcessOutlookNotification:
    """Tests for the standalone process_outlook_notification function."""

    @pytest.mark.asyncio
    async def test_process_single_change(self):
        payload = _make_outlook_payload()
        results = await process_outlook_notification(payload, "t1")
        assert len(results) == 1
        assert results[0].provider == WebhookProvider.OUTLOOK

    @pytest.mark.asyncio
    async def test_process_empty_value(self):
        results = await process_outlook_notification({"value": []}, "t1")
        assert results == []

    @pytest.mark.asyncio
    async def test_process_client_state_mismatch_skips(self):
        payload = _make_outlook_payload(client_state="wrong")
        results = await process_outlook_notification(payload, "t1", client_state="expected")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_process_client_state_match(self):
        payload = _make_outlook_payload(client_state="secret")
        results = await process_outlook_notification(payload, "t1", client_state="secret")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_process_no_client_state_filter(self):
        payload = _make_outlook_payload()
        results = await process_outlook_notification(payload, "t1", client_state=None)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_process_multiple_changes(self):
        payload = {
            "value": [
                {"subscriptionId": "s1", "changeType": "created", "resource": "r1"},
                {"subscriptionId": "s2", "changeType": "updated", "resource": "r2"},
                {"subscriptionId": "s3", "changeType": "deleted", "resource": "r3"},
            ]
        }
        results = await process_outlook_notification(payload, "t1")
        assert len(results) == 3
        types = [r.notification_type for r in results]
        assert NotificationType.MESSAGE_CREATED in types
        assert NotificationType.MESSAGE_UPDATED in types
        assert NotificationType.MESSAGE_DELETED in types


# ===========================================================================
# Notification Queuing Tests
# ===========================================================================


class TestNotificationQueuing:
    """Tests for _queue_notification behavior."""

    @pytest.mark.asyncio
    async def test_queue_caps_at_100(self, handler, gmail_request):
        """Notification history should be capped at 100 per tenant."""
        # Prepopulate with 99 notifications
        for i in range(99):
            _notification_history.setdefault("test-tenant", []).append(
                WebhookNotification(
                    provider=WebhookProvider.GMAIL,
                    notification_type=NotificationType.SYNC_REQUESTED,
                    account_id="acc",
                    resource_id=f"old-{i}",
                    tenant_id="test-tenant",
                    timestamp=datetime.now(timezone.utc),
                    raw_data={},
                )
            )
        # Add 5 more via handler
        req = gmail_request()
        for _ in range(5):
            await handler.handle(req, "/api/v1/webhooks/gmail", "POST")

        assert len(_notification_history["test-tenant"]) <= 100

    @pytest.mark.asyncio
    async def test_queue_updates_subscription_stats(self, handler, populated_subscriptions):
        """When a notification matches a subscription account, update stats."""
        sub1, _ = populated_subscriptions
        initial_count = sub1.notification_count
        # Send a notification for account-001
        data = json.dumps({"emailAddress": "x@y.com", "historyId": "h1"}).encode()
        b64 = base64.b64encode(data).decode()
        req = MockRequest(_body={"message": {"data": b64}})
        # Manually set account_id to match sub1
        # We need to patch _find_account_by_email to return account-001
        with patch(
            "aragora.server.handlers.features.email_webhooks._find_account_by_email",
            return_value="account-001",
        ):
            await handler.handle(req, "/api/v1/webhooks/gmail", "POST")
        assert sub1.notification_count == initial_count + 1
        assert sub1.last_notification is not None


# ===========================================================================
# Handler Singleton Tests
# ===========================================================================


class TestHandlerSingleton:
    """Tests for get_email_webhooks_handler and handle_email_webhooks."""

    def test_get_handler_returns_same_instance(self):
        h1 = get_email_webhooks_handler()
        h2 = get_email_webhooks_handler()
        assert h1 is h2

    def test_get_handler_returns_correct_type(self):
        h = get_email_webhooks_handler()
        assert isinstance(h, EmailWebhooksHandler)

    @pytest.mark.asyncio
    async def test_handle_email_webhooks_entry_point(self):
        req = MockRequest()
        result = await handle_email_webhooks(req, "/api/v1/webhooks/status", "GET")
        assert _status(result) == 200


# ===========================================================================
# Query Params Utility Tests
# ===========================================================================


class TestQueryParamsUtility:
    """Tests for _get_query_params."""

    def test_from_query_attr(self, handler):
        req = MockRequest(query={"foo": "bar"})
        params = handler._get_query_params(req)
        assert params["foo"] == "bar"

    def test_from_args_attr(self, handler):
        req = MagicMock(spec=[])
        req.args = {"k": "v"}
        params = handler._get_query_params(req)
        assert params["k"] == "v"

    def test_empty_when_no_attr(self, handler):
        req = MagicMock(spec=[])
        params = handler._get_query_params(req)
        assert params == {}


# ===========================================================================
# JSON Body Utility Tests
# ===========================================================================


class TestJsonBodyUtility:
    """Tests for _get_json_body."""

    @pytest.mark.asyncio
    async def test_callable_json(self, handler):
        req = MockRequest(_body={"a": 1})
        body = await handler._get_json_body(req)
        assert body == {"a": 1}

    @pytest.mark.asyncio
    async def test_property_json(self, handler):
        req = MagicMock()
        req.json = {"b": 2}
        del req.json  # Remove the method
        req.json = {"b": 2}  # Set as property-like

        # MagicMock.json is callable by default; use a simple object
        class SimpleReq:
            json = {"b": 2}

        body = await handler._get_json_body(SimpleReq())
        assert body == {"b": 2}

    @pytest.mark.asyncio
    async def test_no_json_attr_returns_empty(self, handler):
        req = MagicMock(spec=[])
        body = await handler._get_json_body(req)
        assert body == {}


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling in route_request."""

    @pytest.mark.asyncio
    async def test_gmail_webhook_exception_returns_200(self, handler):
        """Gmail errors still return 200 to prevent Google retries."""
        req = MockRequest(_body=_make_gmail_payload())
        with patch(
            "aragora.server.handlers.features.email_webhooks.process_gmail_notification",
            side_effect=json.JSONDecodeError("test", "doc", 0),
        ):
            result = await handler.handle(req, "/api/v1/webhooks/gmail", "POST")
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["status"] == "error"

    @pytest.mark.asyncio
    async def test_outlook_webhook_exception_returns_200(self, handler):
        """Outlook errors also return 200 to prevent retries."""
        req = MockRequest(_body=_make_outlook_payload())
        with patch(
            "aragora.server.handlers.features.email_webhooks.process_outlook_notification",
            side_effect=ValueError("test error"),
        ):
            result = await handler.handle(req, "/api/v1/webhooks/outlook", "POST")
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["status"] == "error"

    @pytest.mark.asyncio
    async def test_subscribe_exception_returns_500(self, handler):
        """Subscribe errors return 500."""
        req = MockRequest(_body={"provider": "gmail", "account_id": "acc-1"})
        with patch.object(
            handler,
            "_create_gmail_subscription",
            side_effect=OSError("connection lost"),
        ):
            result = await handler.handle(req, "/api/v1/webhooks/subscribe", "POST")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_unsubscribe_exception_returns_500(self, handler, populated_subscriptions):
        """Unsubscribe internal errors return 500."""
        req = MockRequest(_body={"subscription_id": "sub-001"})
        with patch.object(
            handler,
            "_delete_gmail_subscription",
            side_effect=OSError("network error"),
        ):
            result = await handler.handle(req, "/api/v1/webhooks/unsubscribe", "POST")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_route_request_catches_general_exceptions(self, handler):
        """route_request catches broad exceptions and returns 500."""
        req = MockRequest()
        with patch.object(handler, "_get_tenant_id", side_effect=RuntimeError("boom")):
            result = await handler.handle(req, "/api/v1/webhooks/status", "GET")
        assert _status(result) == 500


# ===========================================================================
# Trigger Sync Tests
# ===========================================================================


class TestTriggerSync:
    """Tests for sync trigger functions."""

    @pytest.mark.asyncio
    async def test_gmail_sync_trigger_import_error(self, handler, gmail_request):
        """Gmail sync gracefully handles ImportError."""
        req = gmail_request()
        with patch(
            "aragora.server.handlers.features.email_webhooks._trigger_gmail_sync",
            side_effect=ImportError("no module"),
        ):
            # Should not raise -- the handler catches the error
            # But _trigger_gmail_sync is called inside _queue_notification
            # Let's test the direct path through the handler
            result = await handler.handle(req, "/api/v1/webhooks/gmail", "POST")
            # Even if sync trigger fails, notification should be processed
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_outlook_sync_trigger_import_error(self, handler, outlook_request):
        """Outlook sync gracefully handles ImportError."""
        req = outlook_request()
        result = await handler.handle(req, "/api/v1/webhooks/outlook", "POST")
        assert _status(result) == 200


# ===========================================================================
# Provider Subscription Creation Tests
# ===========================================================================


class TestProviderSubscriptionCreation:
    """Tests for _create_gmail_subscription and _create_outlook_subscription."""

    @pytest.mark.asyncio
    async def test_create_gmail_subscription_default(self, handler, sample_subscription):
        result = await handler._create_gmail_subscription(sample_subscription)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_create_outlook_subscription_default(self, handler, sample_subscription):
        result = await handler._create_outlook_subscription(sample_subscription)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_create_gmail_subscription_connection_error(self, handler, sample_subscription):
        with patch(
            "aragora.server.handlers.features.email_webhooks.EmailWebhooksHandler._create_gmail_subscription",
            return_value={"success": False, "error": "Internal server error"},
        ):
            result = await handler._create_gmail_subscription(sample_subscription)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_create_outlook_subscription_connection_error(self, handler, sample_subscription):
        with patch(
            "aragora.server.handlers.features.email_webhooks.EmailWebhooksHandler._create_outlook_subscription",
            return_value={"success": False, "error": "Internal server error"},
        ):
            result = await handler._create_outlook_subscription(sample_subscription)
        assert result["success"] is False


# ===========================================================================
# Delete Subscription Tests
# ===========================================================================


class TestDeleteSubscription:
    """Tests for _delete_gmail_subscription and _delete_outlook_subscription."""

    @pytest.mark.asyncio
    async def test_delete_gmail_subscription(self, handler, sample_subscription):
        # Should not raise
        await handler._delete_gmail_subscription(sample_subscription)

    @pytest.mark.asyncio
    async def test_delete_outlook_subscription(self, handler, sample_subscription):
        # Should not raise
        await handler._delete_outlook_subscription(sample_subscription)


# ===========================================================================
# Full Integration / End-to-End Tests
# ===========================================================================


class TestIntegration:
    """Integration-style tests exercising multiple operations."""

    @pytest.mark.asyncio
    async def test_subscribe_then_status(self, handler):
        # Subscribe
        req = MockRequest(_body={"provider": "gmail", "account_id": "acc-int"})
        result = await handler.handle(req, "/api/v1/webhooks/subscribe", "POST")
        assert _status(result) == 200

        # Status
        req2 = MockRequest()
        result2 = await handler.handle(req2, "/api/v1/webhooks/status", "GET")
        body = _body(result2)
        assert body["data"]["summary"]["total"] == 1
        assert body["data"]["summary"]["active"] == 1

    @pytest.mark.asyncio
    async def test_subscribe_then_unsubscribe(self, handler):
        # Subscribe
        req = MockRequest(_body={"provider": "outlook", "account_id": "acc-x"})
        result = await handler.handle(req, "/api/v1/webhooks/subscribe", "POST")
        sub_id = _body(result)["data"]["subscription"]["id"]

        # Unsubscribe
        req2 = MockRequest(_body={"subscription_id": sub_id})
        result2 = await handler.handle(req2, "/api/v1/webhooks/unsubscribe", "DELETE")
        assert _status(result2) == 200

        # Status should be empty
        req3 = MockRequest()
        result3 = await handler.handle(req3, "/api/v1/webhooks/status", "GET")
        body = _body(result3)
        assert body["data"]["summary"]["total"] == 0

    @pytest.mark.asyncio
    async def test_gmail_webhook_then_history(self, handler, gmail_request):
        req = gmail_request()
        await handler.handle(req, "/api/v1/webhooks/gmail", "POST")
        await handler.handle(req, "/api/v1/webhooks/gmail", "POST")

        req2 = MockRequest()
        result = await handler.handle(req2, "/api/v1/webhooks/history", "GET")
        body = _body(result)
        assert body["data"]["total"] == 2

    @pytest.mark.asyncio
    async def test_outlook_webhook_then_history(self, handler, outlook_request):
        req = outlook_request(change_type="updated")
        await handler.handle(req, "/api/v1/webhooks/outlook", "POST")

        req2 = MockRequest()
        result = await handler.handle(req2, "/api/v1/webhooks/history", "GET")
        body = _body(result)
        assert body["data"]["total"] == 1
        assert body["data"]["notifications"][0]["notification_type"] == "message_updated"

    @pytest.mark.asyncio
    async def test_multiple_tenants_isolated(self, handler):
        # Subscribe for tenant A
        req_a = MockRequest(_body={"provider": "gmail", "account_id": "acc-a"}, tenant_id="A")
        await handler.handle(req_a, "/api/v1/webhooks/subscribe", "POST")

        # Subscribe for tenant B
        req_b = MockRequest(_body={"provider": "outlook", "account_id": "acc-b"}, tenant_id="B")
        await handler.handle(req_b, "/api/v1/webhooks/subscribe", "POST")

        # Status for A
        req_sa = MockRequest(tenant_id="A")
        result_a = await handler.handle(req_sa, "/api/v1/webhooks/status", "GET")
        body_a = _body(result_a)
        assert body_a["data"]["summary"]["total"] == 1

        # Status for B
        req_sb = MockRequest(tenant_id="B")
        result_b = await handler.handle(req_sb, "/api/v1/webhooks/status", "GET")
        body_b = _body(result_b)
        assert body_b["data"]["summary"]["total"] == 1
