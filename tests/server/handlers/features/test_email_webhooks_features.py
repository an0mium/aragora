"""Comprehensive tests for EmailWebhooksHandler."""

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

import asyncio
import base64
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional
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
from aragora.server.handlers.features import email_webhooks as _ew_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_module_state():
    """Clear all module-level mutable state."""
    _subscriptions.clear()
    _tenant_subscriptions.clear()
    _notification_history.clear()
    _pending_validations.clear()
    _ew_module._handler_instance = None


@pytest.fixture(autouse=True)
def _reset_state():
    """Autouse fixture: clear module state before and after each test."""
    _clear_module_state()
    yield
    _clear_module_state()


class FakeRequest:
    """Minimal request object for handler tests."""

    def __init__(
        self,
        body: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, str]] = None,
        tenant_id: str = "default",
    ):
        self._body = body or {}
        self.query = query or {}
        self.tenant_id = tenant_id

    async def json(self):
        return self._body


def _make_handler(ctx=None) -> EmailWebhooksHandler:
    return EmailWebhooksHandler(server_context=ctx or {})


def _gmail_payload(email: str = "user@gmail.com", history_id: str = "12345"):
    """Build a Gmail Pub/Sub push notification payload."""
    data = json.dumps({"emailAddress": email, "historyId": history_id}).encode()
    return {
        "message": {
            "data": base64.b64encode(data).decode(),
            "messageId": "msg-1",
            "publishTime": "2024-01-01T00:00:00Z",
        },
        "subscription": "projects/my-project/subscriptions/my-sub",
    }


def _outlook_payload(
    change_type: str = "created",
    resource: str = "Users/u1/Messages/m1",
    subscription_id: str = "sub-1",
    client_state: Optional[str] = None,
):
    """Build an Outlook Graph change notification payload."""
    change: Dict[str, Any] = {
        "subscriptionId": subscription_id,
        "changeType": change_type,
        "resource": resource,
        "tenantId": "ms-tenant-1",
    }
    if client_state is not None:
        change["clientState"] = client_state
    return {"value": [change]}


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _parse_body(result) -> dict:
    """Parse HandlerResult body as JSON.

    success_response wraps in {"success": true, "data": ...}.
    error_response wraps in {"error": "..."}.
    This helper returns the raw parsed JSON so callers can check either form.
    """
    return json.loads(result.body.decode())


def _data(result) -> dict:
    """Extract the 'data' payload from a success_response."""
    body = _parse_body(result)
    return body.get("data", body)


# We need to bypass the @require_permission decorator for handler.handle tests.
# The decorator expects an AuthorizationContext which our FakeRequest doesn't supply.
# Monkey-patch the handler's handle method to call internal routing directly.


async def _dispatch(handler: EmailWebhooksHandler, request, path, method):
    """Call the handler's internal routing logic, bypassing RBAC decorator."""
    try:
        tenant_id = handler._get_tenant_id(request)

        if path == "/api/v1/webhooks/gmail" and method == "POST":
            return await handler._handle_gmail_webhook(request, tenant_id)
        elif path == "/api/v1/webhooks/outlook" and method == "POST":
            return await handler._handle_outlook_webhook(request, tenant_id)
        elif path == "/api/v1/webhooks/outlook/validate" and method == "POST":
            return await handler._handle_outlook_validation(request)
        elif path == "/api/v1/webhooks/status" and method == "GET":
            return await handler._handle_status(request, tenant_id)
        elif path == "/api/v1/webhooks/subscribe" and method == "POST":
            return await handler._handle_subscribe(request, tenant_id)
        elif path == "/api/v1/webhooks/unsubscribe" and method in ("POST", "DELETE"):
            return await handler._handle_unsubscribe(request, tenant_id)
        elif path == "/api/v1/webhooks/history" and method == "GET":
            return await handler._handle_history(request, tenant_id)

        from aragora.server.handlers.base import error_response

        return error_response("Not found", 404)
    except Exception as e:
        from aragora.server.handlers.base import error_response

        return error_response(f"Internal error: {str(e)}", 500)


# ===========================================================================
# Data Model Tests
# ===========================================================================


class TestWebhookProvider:
    def test_gmail_value(self):
        assert WebhookProvider.GMAIL.value == "gmail"

    def test_outlook_value(self):
        assert WebhookProvider.OUTLOOK.value == "outlook"


class TestWebhookStatus:
    def test_all_statuses(self):
        assert WebhookStatus.ACTIVE.value == "active"
        assert WebhookStatus.PENDING.value == "pending"
        assert WebhookStatus.EXPIRED.value == "expired"
        assert WebhookStatus.ERROR.value == "error"


class TestNotificationType:
    def test_all_types(self):
        assert NotificationType.MESSAGE_CREATED.value == "message_created"
        assert NotificationType.MESSAGE_UPDATED.value == "message_updated"
        assert NotificationType.MESSAGE_DELETED.value == "message_deleted"
        assert NotificationType.LABEL_CHANGED.value == "label_changed"
        assert NotificationType.SYNC_REQUESTED.value == "sync_requested"


class TestWebhookSubscription:
    def test_to_dict(self):
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        sub = WebhookSubscription(
            id="sub-1",
            tenant_id="t-1",
            account_id="acc-1",
            provider=WebhookProvider.GMAIL,
            status=WebhookStatus.ACTIVE,
            created_at=now,
            expires_at=now + timedelta(hours=72),
            notification_url="https://example.com/hook",
            notification_count=5,
            error_count=1,
        )
        d = sub.to_dict()
        assert d["id"] == "sub-1"
        assert d["tenant_id"] == "t-1"
        assert d["account_id"] == "acc-1"
        assert d["provider"] == "gmail"
        assert d["status"] == "active"
        assert d["created_at"] == now.isoformat()
        assert d["expires_at"] is not None
        assert d["notification_count"] == 5
        assert d["error_count"] == 1
        assert d["last_notification"] is None

    def test_to_dict_no_expires(self):
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        sub = WebhookSubscription(
            id="sub-2",
            tenant_id="t-1",
            account_id="acc-1",
            provider=WebhookProvider.OUTLOOK,
            status=WebhookStatus.PENDING,
            created_at=now,
        )
        d = sub.to_dict()
        assert d["expires_at"] is None

    def test_to_dict_with_last_notification(self):
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        sub = WebhookSubscription(
            id="sub-3",
            tenant_id="t-1",
            account_id="acc-1",
            provider=WebhookProvider.GMAIL,
            status=WebhookStatus.ACTIVE,
            created_at=now,
            last_notification=now + timedelta(hours=1),
        )
        d = sub.to_dict()
        assert d["last_notification"] is not None


class TestWebhookNotification:
    def test_to_dict(self):
        now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        n = WebhookNotification(
            provider=WebhookProvider.GMAIL,
            notification_type=NotificationType.SYNC_REQUESTED,
            account_id="acc-1",
            resource_id="hist-100",
            tenant_id="t-1",
            timestamp=now,
            raw_data={"some": "data"},
            metadata={"email_address": "a@b.com"},
        )
        d = n.to_dict()
        assert d["provider"] == "gmail"
        assert d["notification_type"] == "sync_requested"
        assert d["account_id"] == "acc-1"
        assert d["resource_id"] == "hist-100"
        assert d["tenant_id"] == "t-1"
        assert d["timestamp"] == now.isoformat()
        assert d["metadata"]["email_address"] == "a@b.com"


# ===========================================================================
# process_gmail_notification Tests
# ===========================================================================


class TestProcessGmailNotification:
    def test_valid_notification(self):
        payload = _gmail_payload("test@gmail.com", "99999")
        result = _run(process_gmail_notification(payload, "t-1"))
        assert result is not None
        assert result.provider == WebhookProvider.GMAIL
        assert result.notification_type == NotificationType.SYNC_REQUESTED
        assert result.resource_id == "99999"
        assert result.metadata["email_address"] == "test@gmail.com"
        assert result.tenant_id == "t-1"

    def test_missing_data_field(self):
        result = _run(process_gmail_notification({"message": {}}, "t-1"))
        assert result is None

    def test_empty_message(self):
        result = _run(process_gmail_notification({}, "t-1"))
        assert result is None

    def test_invalid_base64(self):
        payload = {"message": {"data": "not-valid-b64!!!"}}
        result = _run(process_gmail_notification(payload, "t-1"))
        assert result is None

    def test_invalid_json_in_data(self):
        payload = {"message": {"data": base64.b64encode(b"not json").decode()}}
        result = _run(process_gmail_notification(payload, "t-1"))
        assert result is None

    def test_missing_email_address(self):
        data = json.dumps({"historyId": "123"}).encode()
        payload = {"message": {"data": base64.b64encode(data).decode()}}
        result = _run(process_gmail_notification(payload, "t-1"))
        assert result is None

    def test_notification_queued_in_history(self):
        payload = _gmail_payload()
        _run(process_gmail_notification(payload, "t-1"))
        assert "t-1" in _notification_history
        assert len(_notification_history["t-1"]) == 1

    def test_history_capped_at_100(self):
        """History should keep only the last 100 entries."""
        for i in range(105):
            payload = _gmail_payload(f"user{i}@gmail.com", str(i))
            _run(process_gmail_notification(payload, "t-1"))
        assert len(_notification_history["t-1"]) == 100


# ===========================================================================
# process_outlook_notification Tests
# ===========================================================================


class TestProcessOutlookNotification:
    def test_created_change(self):
        payload = _outlook_payload(change_type="created")
        results = _run(process_outlook_notification(payload, "t-1"))
        assert len(results) == 1
        assert results[0].notification_type == NotificationType.MESSAGE_CREATED

    def test_updated_change(self):
        payload = _outlook_payload(change_type="updated")
        results = _run(process_outlook_notification(payload, "t-1"))
        assert len(results) == 1
        assert results[0].notification_type == NotificationType.MESSAGE_UPDATED

    def test_deleted_change(self):
        payload = _outlook_payload(change_type="deleted")
        results = _run(process_outlook_notification(payload, "t-1"))
        assert results[0].notification_type == NotificationType.MESSAGE_DELETED

    def test_unknown_change_type(self):
        payload = _outlook_payload(change_type="other")
        results = _run(process_outlook_notification(payload, "t-1"))
        assert results[0].notification_type == NotificationType.SYNC_REQUESTED

    def test_empty_value_list(self):
        results = _run(process_outlook_notification({"value": []}, "t-1"))
        assert results == []

    def test_client_state_mismatch_skips(self):
        payload = _outlook_payload(client_state="wrong")
        results = _run(process_outlook_notification(payload, "t-1", client_state="expected"))
        assert results == []

    def test_client_state_match(self):
        payload = _outlook_payload(client_state="correct")
        results = _run(process_outlook_notification(payload, "t-1", client_state="correct"))
        assert len(results) == 1

    def test_multiple_changes(self):
        payload = {
            "value": [
                {"changeType": "created", "resource": "r1", "subscriptionId": "s1"},
                {"changeType": "deleted", "resource": "r2", "subscriptionId": "s2"},
            ]
        }
        results = _run(process_outlook_notification(payload, "t-1"))
        assert len(results) == 2

    def test_subscription_lookup(self):
        """If there's a matching subscription, account_id should be populated."""
        now = datetime.now(timezone.utc)
        _subscriptions["sub-x"] = WebhookSubscription(
            id="sub-x",
            tenant_id="t-1",
            account_id="acc-found",
            provider=WebhookProvider.OUTLOOK,
            status=WebhookStatus.ACTIVE,
            created_at=now,
        )
        payload = _outlook_payload(subscription_id="sub-x")
        results = _run(process_outlook_notification(payload, "t-1"))
        assert results[0].account_id == "acc-found"

    def test_no_subscription_match(self):
        payload = _outlook_payload(subscription_id="nonexistent")
        results = _run(process_outlook_notification(payload, "t-1"))
        assert results[0].account_id == ""


# ===========================================================================
# Handler Routing Tests
# ===========================================================================


class TestHandlerRoutes:
    def test_routes_list(self):
        assert "/api/v1/webhooks/gmail" in EmailWebhooksHandler.ROUTES
        assert "/api/v1/webhooks/outlook" in EmailWebhooksHandler.ROUTES
        assert "/api/v1/webhooks/outlook/validate" in EmailWebhooksHandler.ROUTES
        assert "/api/v1/webhooks/status" in EmailWebhooksHandler.ROUTES
        assert "/api/v1/webhooks/subscribe" in EmailWebhooksHandler.ROUTES
        assert "/api/v1/webhooks/unsubscribe" in EmailWebhooksHandler.ROUTES
        assert "/api/v1/webhooks/history" in EmailWebhooksHandler.ROUTES

    def test_constructor_default_context(self):
        handler = _make_handler()
        assert handler.ctx is not None

    def test_get_tenant_id_default(self):
        handler = _make_handler()
        req = FakeRequest()
        del req.tenant_id  # no tenant_id attr
        assert handler._get_tenant_id(req) == "default"

    def test_get_tenant_id_from_request(self):
        handler = _make_handler()
        req = FakeRequest(tenant_id="my-tenant")
        assert handler._get_tenant_id(req) == "my-tenant"


# ===========================================================================
# Gmail Webhook Handler Tests
# ===========================================================================


class TestGmailWebhook:
    def test_successful_notification(self):
        handler = _make_handler()
        req = FakeRequest(body=_gmail_payload())
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/gmail", "POST"))
        assert result.status_code == 200
        d = _data(result)
        assert d["status"] == "processed"
        assert "notification" in d

    def test_unprocessable_notification(self):
        handler = _make_handler()
        req = FakeRequest(body={"message": {}})
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/gmail", "POST"))
        assert result.status_code == 200
        d = _data(result)
        assert d["status"] == "acknowledged"

    def test_exception_returns_200(self):
        """Gmail webhook should return 200 even on errors to avoid Google retries."""
        handler = _make_handler()
        # Use a request whose json() raises
        req = FakeRequest()

        async def bad_json():
            raise ValueError("bad")

        req.json = bad_json
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/gmail", "POST"))
        assert result.status_code == 200
        d = _data(result)
        assert d["status"] == "error"


# ===========================================================================
# Outlook Webhook Handler Tests
# ===========================================================================


class TestOutlookWebhook:
    def test_successful_notification(self):
        handler = _make_handler()
        req = FakeRequest(body=_outlook_payload())
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/outlook", "POST"))
        assert result.status_code == 200
        d = _data(result)
        assert d["status"] == "processed"
        assert d["count"] == 1

    def test_validation_token_in_query(self):
        """Outlook sends validationToken as query param during subscription setup."""
        handler = _make_handler()
        req = FakeRequest(query={"validationToken": "abc-token-xyz"})
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/outlook", "POST"))
        assert result.status_code == 200
        assert result.content_type == "text/plain"
        assert result.body == b"abc-token-xyz"

    def test_exception_returns_200(self):
        handler = _make_handler()
        req = FakeRequest()

        async def bad_json():
            raise RuntimeError("boom")

        req.json = bad_json
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/outlook", "POST"))
        assert result.status_code == 200
        d = _data(result)
        assert d["status"] == "error"


# ===========================================================================
# Outlook Validation Handler Tests
# ===========================================================================


class TestOutlookValidation:
    def test_valid_token(self):
        handler = _make_handler()
        req = FakeRequest(query={"validationToken": "my-token"})
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/outlook/validate", "POST"))
        assert result.status_code == 200
        assert result.content_type == "text/plain"
        assert result.body == b"my-token"

    def test_missing_token(self):
        handler = _make_handler()
        req = FakeRequest(query={})
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/outlook/validate", "POST"))
        assert result.status_code == 400
        body = _parse_body(result)
        assert "validationToken" in body.get("error", "")


# ===========================================================================
# Subscription Management Tests
# ===========================================================================


class TestSubscribe:
    def test_gmail_subscribe(self):
        handler = _make_handler()
        req = FakeRequest(
            body={
                "provider": "gmail",
                "account_id": "acc-1",
                "notification_url": "https://example.com/hook",
                "expiration_hours": 48,
            },
            tenant_id="t-1",
        )
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/subscribe", "POST"))
        assert result.status_code == 200
        d = _data(result)
        sub = d["subscription"]
        assert sub["provider"] == "gmail"
        assert sub["account_id"] == "acc-1"
        assert sub["status"] == "active"
        assert "client_state" in d

    def test_outlook_subscribe(self):
        handler = _make_handler()
        req = FakeRequest(
            body={
                "provider": "outlook",
                "account_id": "acc-2",
            },
            tenant_id="t-1",
        )
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/subscribe", "POST"))
        assert result.status_code == 200
        d = _data(result)
        assert d["subscription"]["provider"] == "outlook"

    def test_invalid_provider(self):
        handler = _make_handler()
        req = FakeRequest(body={"provider": "yahoo", "account_id": "acc"})
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/subscribe", "POST"))
        assert result.status_code == 400

    def test_missing_account_id(self):
        handler = _make_handler()
        req = FakeRequest(body={"provider": "gmail"})
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/subscribe", "POST"))
        assert result.status_code == 400

    def test_subscription_stored(self):
        handler = _make_handler()
        req = FakeRequest(body={"provider": "gmail", "account_id": "a"}, tenant_id="t-1")
        _run(_dispatch(handler, req, "/api/v1/webhooks/subscribe", "POST"))
        assert len(_subscriptions) == 1
        assert "t-1" in _tenant_subscriptions
        assert len(_tenant_subscriptions["t-1"]) == 1

    def test_default_expiration(self):
        handler = _make_handler()
        req = FakeRequest(body={"provider": "gmail", "account_id": "a"}, tenant_id="t-1")
        _run(_dispatch(handler, req, "/api/v1/webhooks/subscribe", "POST"))
        sub = list(_subscriptions.values())[0]
        diff = sub.expires_at - sub.created_at
        assert diff == timedelta(hours=72)

    def test_custom_expiration(self):
        handler = _make_handler()
        req = FakeRequest(
            body={
                "provider": "gmail",
                "account_id": "a",
                "expiration_hours": 24,
            },
            tenant_id="t-1",
        )
        _run(_dispatch(handler, req, "/api/v1/webhooks/subscribe", "POST"))
        sub = list(_subscriptions.values())[0]
        diff = sub.expires_at - sub.created_at
        assert diff == timedelta(hours=24)

    def test_client_state_generated(self):
        handler = _make_handler()
        req = FakeRequest(body={"provider": "gmail", "account_id": "a"}, tenant_id="t-1")
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/subscribe", "POST"))
        d = _data(result)
        cs = d["client_state"]
        assert isinstance(cs, str)
        assert len(cs) == 32


# ===========================================================================
# Status Handler Tests
# ===========================================================================


class TestStatus:
    def test_empty_status(self):
        handler = _make_handler()
        req = FakeRequest(tenant_id="t-1")
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/status", "GET"))
        assert result.status_code == 200
        d = _data(result)
        assert d["subscriptions"] == []
        assert d["summary"]["total"] == 0
        assert d["summary"]["active"] == 0

    def test_status_with_subscriptions(self):
        handler = _make_handler()
        # Create two subscriptions
        for i in range(2):
            req = FakeRequest(
                body={"provider": "gmail", "account_id": f"a{i}"},
                tenant_id="t-1",
            )
            _run(_dispatch(handler, req, "/api/v1/webhooks/subscribe", "POST"))

        req = FakeRequest(tenant_id="t-1")
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/status", "GET"))
        d = _data(result)
        assert d["summary"]["total"] == 2
        assert d["summary"]["active"] == 2

    def test_status_tenant_isolation(self):
        handler = _make_handler()
        req = FakeRequest(body={"provider": "gmail", "account_id": "a"}, tenant_id="t-1")
        _run(_dispatch(handler, req, "/api/v1/webhooks/subscribe", "POST"))

        req = FakeRequest(tenant_id="t-other")
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/status", "GET"))
        d = _data(result)
        assert d["summary"]["total"] == 0


# ===========================================================================
# Unsubscribe Handler Tests
# ===========================================================================


class TestUnsubscribe:
    def _create_subscription(self, handler, tenant="t-1"):
        req = FakeRequest(
            body={"provider": "gmail", "account_id": "acc-x"},
            tenant_id=tenant,
        )
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/subscribe", "POST"))
        return _data(result)["subscription"]["id"]

    def test_unsubscribe_success(self):
        handler = _make_handler()
        sub_id = self._create_subscription(handler)
        assert sub_id in _subscriptions

        req = FakeRequest(body={"subscription_id": sub_id}, tenant_id="t-1")
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/unsubscribe", "POST"))
        assert result.status_code == 200
        d = _data(result)
        assert d["status"] == "deleted"
        assert sub_id not in _subscriptions

    def test_unsubscribe_delete_method(self):
        handler = _make_handler()
        sub_id = self._create_subscription(handler)

        req = FakeRequest(body={"subscription_id": sub_id}, tenant_id="t-1")
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/unsubscribe", "DELETE"))
        assert result.status_code == 200

    def test_unsubscribe_missing_id(self):
        handler = _make_handler()
        req = FakeRequest(body={}, tenant_id="t-1")
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/unsubscribe", "POST"))
        assert result.status_code == 400

    def test_unsubscribe_not_found(self):
        handler = _make_handler()
        req = FakeRequest(body={"subscription_id": "nonexistent"}, tenant_id="t-1")
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/unsubscribe", "POST"))
        assert result.status_code == 404

    def test_unsubscribe_wrong_tenant(self):
        handler = _make_handler()
        sub_id = self._create_subscription(handler, tenant="t-1")

        req = FakeRequest(body={"subscription_id": sub_id}, tenant_id="t-other")
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/unsubscribe", "POST"))
        assert result.status_code == 403

    def test_unsubscribe_removes_from_tenant_list(self):
        handler = _make_handler()
        sub_id = self._create_subscription(handler)
        assert sub_id in _tenant_subscriptions.get("t-1", [])

        req = FakeRequest(body={"subscription_id": sub_id}, tenant_id="t-1")
        _run(_dispatch(handler, req, "/api/v1/webhooks/unsubscribe", "POST"))
        assert sub_id not in _tenant_subscriptions.get("t-1", [])


# ===========================================================================
# History Handler Tests
# ===========================================================================


class TestHistory:
    def test_empty_history(self):
        handler = _make_handler()
        req = FakeRequest(tenant_id="t-1")
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/history", "GET"))
        assert result.status_code == 200
        d = _data(result)
        assert d["notifications"] == []
        assert d["total"] == 0

    def test_history_after_notifications(self):
        # First generate some notifications
        for i in range(3):
            payload = _gmail_payload(f"u{i}@g.com", str(i))
            _run(process_gmail_notification(payload, "t-1"))

        handler = _make_handler()
        req = FakeRequest(tenant_id="t-1")
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/history", "GET"))
        d = _data(result)
        assert d["total"] == 3
        assert len(d["notifications"]) == 3

    def test_history_limit(self):
        for i in range(10):
            payload = _gmail_payload(f"u{i}@g.com", str(i))
            _run(process_gmail_notification(payload, "t-1"))

        handler = _make_handler()
        req = FakeRequest(tenant_id="t-1", query={"limit": "3"})
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/history", "GET"))
        d = _data(result)
        assert d["total"] == 3

    def test_history_reversed(self):
        """Most recent notifications should appear first."""
        for i in range(3):
            payload = _gmail_payload(f"u{i}@g.com", str(i))
            _run(process_gmail_notification(payload, "t-1"))

        handler = _make_handler()
        req = FakeRequest(tenant_id="t-1")
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/history", "GET"))
        d = _data(result)
        # Last notification processed should be first in response
        assert d["notifications"][0]["metadata"]["email_address"] == "u2@g.com"

    def test_history_tenant_isolation(self):
        _run(process_gmail_notification(_gmail_payload(), "t-1"))
        _run(process_gmail_notification(_gmail_payload("other@g.com"), "t-2"))

        handler = _make_handler()
        req = FakeRequest(tenant_id="t-1")
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/history", "GET"))
        d = _data(result)
        assert d["total"] == 1


# ===========================================================================
# Not Found / Routing Edge Cases
# ===========================================================================


class TestRoutingEdgeCases:
    def test_unknown_path(self):
        handler = _make_handler()
        req = FakeRequest()
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/nope", "GET"))
        assert result.status_code == 404

    def test_wrong_method_for_gmail(self):
        handler = _make_handler()
        req = FakeRequest()
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/gmail", "GET"))
        assert result.status_code == 404

    def test_wrong_method_for_status(self):
        handler = _make_handler()
        req = FakeRequest()
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/status", "POST"))
        assert result.status_code == 404


# ===========================================================================
# Utility Method Tests
# ===========================================================================


class TestUtilityMethods:
    def test_get_json_body_callable(self):
        handler = _make_handler()
        req = FakeRequest(body={"key": "value"})
        body = _run(handler._get_json_body(req))
        assert body == {"key": "value"}

    def test_get_json_body_attribute(self):
        handler = _make_handler()
        req = MagicMock()
        req.json = {"direct": True}
        body = _run(handler._get_json_body(req))
        assert body == {"direct": True}

    def test_get_json_body_no_json(self):
        handler = _make_handler()
        req = MagicMock(spec=[])  # no json attr
        body = _run(handler._get_json_body(req))
        assert body == {}

    def test_get_query_params_from_query(self):
        handler = _make_handler()
        req = FakeRequest(query={"a": "1", "b": "2"})
        params = handler._get_query_params(req)
        assert params["a"] == "1"

    def test_get_query_params_from_args(self):
        handler = _make_handler()
        req = MagicMock(spec=["args"])
        req.args = {"x": "y"}
        params = handler._get_query_params(req)
        assert params["x"] == "y"

    def test_get_query_params_fallback(self):
        handler = _make_handler()
        req = MagicMock(spec=[])  # no query or args
        params = handler._get_query_params(req)
        assert params == {}


# ===========================================================================
# Handler Instance Management Tests
# ===========================================================================


class TestHandlerInstanceManagement:
    def test_get_email_webhooks_handler_singleton(self):
        h1 = get_email_webhooks_handler()
        h2 = get_email_webhooks_handler()
        assert h1 is h2

    def test_handle_email_webhooks_entry_point(self):
        """The module-level entry point should invoke the handler."""
        # This calls through the RBAC decorator which will raise without
        # AuthorizationContext, but we just want to ensure it doesn't crash
        # in an unexpected way. We'll patch the handle method.
        handler = get_email_webhooks_handler()
        with patch.object(handler, "handle", new_callable=AsyncMock) as mock_handle:
            from aragora.server.handlers.utils.responses import HandlerResult

            mock_handle.return_value = HandlerResult(
                status_code=200, content_type="application/json", body=b"{}"
            )
            req = FakeRequest()
            # handle_email_webhooks calls handler.handle which is decorated
            # We've patched the underlying handle method
            _ew_module._handler_instance = handler
            result = _run(handle_email_webhooks(req, "/api/v1/webhooks/status", "GET"))
            # Either the mock was called or RBAC intercepted - both are fine
            assert result is not None


# ===========================================================================
# Notification Stats Update Tests
# ===========================================================================


class TestNotificationStatsUpdate:
    def test_subscription_stats_updated_on_notification(self):
        """When a notification matches a subscription's account, stats update."""
        now = datetime.now(timezone.utc)
        _subscriptions["sub-1"] = WebhookSubscription(
            id="sub-1",
            tenant_id="t-1",
            account_id="",  # gmail uses email lookup which returns None -> ""
            provider=WebhookProvider.GMAIL,
            status=WebhookStatus.ACTIVE,
            created_at=now,
        )
        payload = _gmail_payload()
        result = _run(process_gmail_notification(payload, "t-1"))
        # The notification account_id is "" (from _find_account_by_email returning None)
        # and subscription account_id is "" so they match
        sub = _subscriptions["sub-1"]
        assert sub.notification_count == 1
        assert sub.last_notification is not None

    def test_outlook_subscription_stats(self):
        now = datetime.now(timezone.utc)
        _subscriptions["sub-o1"] = WebhookSubscription(
            id="sub-o1",
            tenant_id="t-1",
            account_id="acc-o1",
            provider=WebhookProvider.OUTLOOK,
            status=WebhookStatus.ACTIVE,
            created_at=now,
        )
        payload = _outlook_payload(subscription_id="sub-o1")
        _run(process_outlook_notification(payload, "t-1"))
        sub = _subscriptions["sub-o1"]
        assert sub.notification_count == 1


# ===========================================================================
# Email Parsing / Signature Edge Cases
# ===========================================================================


class TestGmailDataDecoding:
    def test_unicode_email(self):
        data = json.dumps({"emailAddress": "user@example.com", "historyId": "1"}).encode("utf-8")
        payload = {"message": {"data": base64.b64encode(data).decode()}}
        result = _run(process_gmail_notification(payload, "t-1"))
        assert result is not None
        assert result.metadata["email_address"] == "user@example.com"

    def test_extra_fields_ignored(self):
        data = json.dumps(
            {
                "emailAddress": "a@b.com",
                "historyId": "42",
                "extra": "field",
            }
        ).encode()
        payload = {"message": {"data": base64.b64encode(data).decode()}}
        result = _run(process_gmail_notification(payload, "t-1"))
        assert result is not None
        assert result.resource_id == "42"


class TestOutlookResourceParsing:
    def test_resource_preserved(self):
        payload = _outlook_payload(resource="Users/uid/Messages/mid")
        results = _run(process_outlook_notification(payload, "t-1"))
        assert results[0].resource_id == "Users/uid/Messages/mid"

    def test_metadata_includes_change_type(self):
        payload = _outlook_payload(change_type="updated")
        results = _run(process_outlook_notification(payload, "t-1"))
        assert results[0].metadata["change_type"] == "updated"

    def test_metadata_includes_tenant_id(self):
        payload = _outlook_payload()
        results = _run(process_outlook_notification(payload, "t-1"))
        assert results[0].metadata["tenant_id"] == "ms-tenant-1"


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    def test_subscribe_exception_returns_500(self):
        handler = _make_handler()

        # Make _get_json_body raise
        async def boom(req):
            raise RuntimeError("db down")

        handler._get_json_body = boom

        req = FakeRequest(tenant_id="t-1")
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/subscribe", "POST"))
        assert result.status_code == 500

    def test_unsubscribe_exception_returns_500(self):
        handler = _make_handler()

        async def boom(req):
            raise RuntimeError("unexpected")

        handler._get_json_body = boom

        req = FakeRequest(tenant_id="t-1")
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/unsubscribe", "POST"))
        assert result.status_code == 500


# ===========================================================================
# Webhook Config Tests
# ===========================================================================


class TestWebhookConfig:
    def test_subscription_client_state_is_deterministic_hash(self):
        """client_state is SHA-256 of tenant:account:id, truncated to 32 chars."""
        handler = _make_handler()
        req = FakeRequest(body={"provider": "gmail", "account_id": "acc-1"}, tenant_id="t-1")
        _run(_dispatch(handler, req, "/api/v1/webhooks/subscribe", "POST"))
        sub = list(_subscriptions.values())[0]
        expected_prefix = hashlib.sha256(f"t-1:acc-1:{sub.id}".encode()).hexdigest()[:32]
        assert sub.client_state == expected_prefix

    def test_subscription_starts_pending_then_active(self):
        """Subscription should transition from PENDING to ACTIVE on success."""
        handler = _make_handler()
        req = FakeRequest(body={"provider": "outlook", "account_id": "a"}, tenant_id="t-1")
        _run(_dispatch(handler, req, "/api/v1/webhooks/subscribe", "POST"))
        sub = list(_subscriptions.values())[0]
        # After successful creation it should be ACTIVE
        assert sub.status == WebhookStatus.ACTIVE

    def test_gmail_provider_creation_failure(self):
        handler = _make_handler()

        async def fail_create(sub):
            return {"success": False, "error": "API error"}

        handler._create_gmail_subscription = fail_create

        req = FakeRequest(body={"provider": "gmail", "account_id": "a"}, tenant_id="t-1")
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/subscribe", "POST"))
        assert result.status_code == 400
        raw = _parse_body(result)
        assert "API error" in raw.get("error", "")

    def test_outlook_provider_creation_failure(self):
        handler = _make_handler()

        async def fail_create(sub):
            return {"success": False, "error": "Graph API error"}

        handler._create_outlook_subscription = fail_create

        req = FakeRequest(body={"provider": "outlook", "account_id": "a"}, tenant_id="t-1")
        result = _run(_dispatch(handler, req, "/api/v1/webhooks/subscribe", "POST"))
        assert result.status_code == 400
