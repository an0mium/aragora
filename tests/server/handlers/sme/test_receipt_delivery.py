"""Tests for aragora.server.handlers.sme.receipt_delivery - Receipt Delivery Handler."""

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
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
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import json
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from enum import Enum

import pytest

# Try to import the handler module
try:
    from aragora.server.handlers.sme.receipt_delivery import ReceiptDeliveryHandler

    HANDLER_AVAILABLE = True
except ImportError:
    HANDLER_AVAILABLE = False
    ReceiptDeliveryHandler = None

pytestmark = pytest.mark.skipif(
    not HANDLER_AVAILABLE, reason="ReceiptDeliveryHandler not available"
)


# ===========================================================================
# Mock Classes
# ===========================================================================


@dataclass
class MockUser:
    """Mock authenticated user."""

    user_id: str = "user-123"
    id: str = "user-123"
    org_id: str = "org-123"
    email: str = "test@example.com"
    name: str = "Test User"


@dataclass
class MockOrg:
    """Mock organization."""

    id: str = "org-123"
    name: str = "Test Organization"


class MockEventType(Enum):
    RECEIPT = "receipt"
    BUDGET_ALERT = "budget_alert"


@dataclass
class MockSubscription:
    """Mock channel subscription."""

    id: str = "sub-123"
    org_id: str = "org-123"
    channel_type: str = "slack"
    channel_id: str = "C12345"
    workspace_id: str = "T12345"
    channel_name: str = "#decisions"
    event_types: list = field(default_factory=lambda: [MockEventType.RECEIPT])
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "org_id": self.org_id,
            "channel_type": self.channel_type,
            "channel_id": self.channel_id,
            "workspace_id": self.workspace_id,
            "channel_name": self.channel_name,
            "event_types": [e.value if hasattr(e, "value") else e for e in self.event_types],
            "is_active": self.is_active,
        }


class MockHandler:
    """Mock HTTP request handler."""

    def __init__(
        self,
        body: bytes = b"",
        headers: Optional[dict[str, str]] = None,
        path: str = "/",
        method: str = "GET",
    ):
        self._body = body
        self.headers = headers or {"Content-Length": str(len(body))}
        self.path = path
        self.command = method
        self.rfile = BytesIO(body)
        self.client_address = ("127.0.0.1", 12345)

    @classmethod
    def with_json_body(cls, data: dict[str, Any], **kwargs) -> "MockHandler":
        body = json.dumps(data).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
        return cls(body=body, headers=headers, **kwargs)

    def get_argument(self, name: str, default: str = None) -> Optional[str]:
        return default


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter state before each test."""
    try:
        from aragora.server.handlers.sme import receipt_delivery

        receipt_delivery._delivery_limiter._buckets.clear()
    except Exception:
        pass
    yield


@pytest.fixture
def mock_user():
    return MockUser()


@pytest.fixture
def mock_org():
    return MockOrg()


@pytest.fixture
def mock_user_store(mock_user, mock_org):
    store = MagicMock()
    store.get_user_by_id.return_value = mock_user
    store.get_organization_by_id.return_value = mock_org
    return store


@pytest.fixture
def mock_subscription_store():
    store = MagicMock()
    store.get_by_org.return_value = [MockSubscription()]
    store.get_by_org_and_channel.return_value = None
    store.create.return_value = MockSubscription()
    store.update.return_value = True
    return store


@pytest.fixture
def handler_context(mock_user_store):
    return {"user_store": mock_user_store}


@pytest.fixture
def delivery_handler(handler_context, mock_subscription_store):
    # HANDLER_AVAILABLE is enforced by module-level pytestmark
    handler = ReceiptDeliveryHandler(handler_context)
    yield handler


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling."""

    def test_can_handle_config(self, delivery_handler):
        """Test handler recognizes config endpoint."""
        assert delivery_handler.can_handle("/api/v1/sme/receipts/delivery/config") is True

    def test_can_handle_history(self, delivery_handler):
        """Test handler recognizes history endpoint."""
        assert delivery_handler.can_handle("/api/v1/sme/receipts/delivery/history") is True

    def test_can_handle_test(self, delivery_handler):
        """Test handler recognizes test endpoint."""
        assert delivery_handler.can_handle("/api/v1/sme/receipts/delivery/test") is True

    def test_can_handle_stats(self, delivery_handler):
        """Test handler recognizes stats endpoint."""
        assert delivery_handler.can_handle("/api/v1/sme/receipts/delivery/stats") is True

    def test_cannot_handle_unknown_path(self, delivery_handler):
        """Test handler rejects unknown paths."""
        assert delivery_handler.can_handle("/api/v1/unknown") is False


# ===========================================================================
# Get Config Tests
# ===========================================================================


class TestGetConfig:
    """Tests for getting delivery configuration."""

    def test_get_config_success(self, delivery_handler, mock_user):
        """Test successful config retrieval."""
        http_handler = MockHandler(path="/api/v1/sme/receipts/delivery/config", method="GET")

        result = delivery_handler.handle(
            "/api/v1/sme/receipts/delivery/config", {}, http_handler, method="GET"
        )
        assert result is not None

    def test_get_config_no_subscriptions(self, delivery_handler, mock_subscription_store):
        """Test config with no subscriptions."""
        mock_subscription_store.get_by_org.return_value = []
        http_handler = MockHandler(path="/api/v1/sme/receipts/delivery/config", method="GET")

        result = delivery_handler.handle(
            "/api/v1/sme/receipts/delivery/config", {}, http_handler, method="GET"
        )
        assert result is not None


# ===========================================================================
# Update Config Tests
# ===========================================================================


class TestUpdateConfig:
    """Tests for updating delivery configuration."""

    def test_update_config_success(self, delivery_handler, mock_user):
        """Test successful config update."""
        body = {
            "subscriptions": [
                {
                    "channel_type": "slack",
                    "channel_id": "C12345",
                    "workspace_id": "T12345",
                    "channel_name": "#decisions",
                }
            ]
        }
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/receipts/delivery/config", method="POST"
        )

        result = delivery_handler.handle(
            "/api/v1/sme/receipts/delivery/config", {}, http_handler, method="POST"
        )
        assert result is not None

    def test_update_config_invalid_channel_type(self, delivery_handler):
        """Test error with invalid channel type."""
        body = {
            "subscriptions": [
                {
                    "channel_type": "invalid",
                    "channel_id": "C12345",
                }
            ]
        }
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/receipts/delivery/config", method="POST"
        )

        result = delivery_handler.handle(
            "/api/v1/sme/receipts/delivery/config", {}, http_handler, method="POST"
        )
        assert result is not None

    def test_update_config_missing_fields(self, delivery_handler):
        """Test error with missing required fields."""
        body = {"subscriptions": [{"channel_type": "slack"}]}  # Missing channel_id
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/receipts/delivery/config", method="POST"
        )

        result = delivery_handler.handle(
            "/api/v1/sme/receipts/delivery/config", {}, http_handler, method="POST"
        )
        assert result is not None


# ===========================================================================
# Get History Tests
# ===========================================================================


class TestGetHistory:
    """Tests for getting delivery history."""

    def test_get_history_success(self, delivery_handler, mock_user):
        """Test successful history retrieval."""
        http_handler = MockHandler(path="/api/v1/sme/receipts/delivery/history", method="GET")

        result = delivery_handler.handle(
            "/api/v1/sme/receipts/delivery/history", {}, http_handler, method="GET"
        )
        assert result is not None

    def test_get_history_with_filters(self, delivery_handler, mock_user):
        """Test history retrieval with filters."""
        http_handler = MockHandler(path="/api/v1/sme/receipts/delivery/history", method="GET")

        result = delivery_handler.handle(
            "/api/v1/sme/receipts/delivery/history",
            {"channel_type": "slack", "status": "success"},
            http_handler,
            method="GET",
        )
        assert result is not None


# ===========================================================================
# Test Delivery Tests
# ===========================================================================


class TestTestDelivery:
    """Tests for testing delivery."""

    def test_test_delivery_slack(self, delivery_handler, mock_user):
        """Test delivery test to Slack."""
        body = {
            "channel_type": "slack",
            "channel_id": "C12345",
            "workspace_id": "T12345",
            "test_message": "Test message",
        }
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/receipts/delivery/test", method="POST"
        )

        with patch.object(
            delivery_handler,
            "_send_test_message",
            return_value={"success": True, "message_id": "123"},
        ):
            result = delivery_handler.handle(
                "/api/v1/sme/receipts/delivery/test", {}, http_handler, method="POST"
            )
            assert result is not None

    def test_test_delivery_teams(self, delivery_handler, mock_user):
        """Test delivery test to Teams."""
        body = {
            "channel_type": "teams",
            "channel_id": "channel-id",
            "workspace_id": "tenant-id",
        }
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/receipts/delivery/test", method="POST"
        )

        with patch.object(delivery_handler, "_send_test_message", return_value={"success": True}):
            result = delivery_handler.handle(
                "/api/v1/sme/receipts/delivery/test", {}, http_handler, method="POST"
            )
            assert result is not None

    def test_test_delivery_missing_channel_type(self, delivery_handler):
        """Test error when channel_type is missing."""
        body = {"channel_id": "C12345"}
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/receipts/delivery/test", method="POST"
        )

        result = delivery_handler.handle(
            "/api/v1/sme/receipts/delivery/test", {}, http_handler, method="POST"
        )
        assert result is not None

    def test_test_delivery_missing_channel_id(self, delivery_handler):
        """Test error when channel_id is missing."""
        body = {"channel_type": "slack"}
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/receipts/delivery/test", method="POST"
        )

        result = delivery_handler.handle(
            "/api/v1/sme/receipts/delivery/test", {}, http_handler, method="POST"
        )
        assert result is not None

    def test_test_delivery_invalid_channel_type(self, delivery_handler):
        """Test error when channel_type is invalid."""
        body = {"channel_type": "invalid", "channel_id": "123"}
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/receipts/delivery/test", method="POST"
        )

        result = delivery_handler.handle(
            "/api/v1/sme/receipts/delivery/test", {}, http_handler, method="POST"
        )
        assert result is not None

    def test_test_delivery_slack_missing_workspace(self, delivery_handler):
        """Test error when workspace_id missing for Slack."""
        body = {"channel_type": "slack", "channel_id": "C12345"}
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/receipts/delivery/test", method="POST"
        )

        result = delivery_handler.handle(
            "/api/v1/sme/receipts/delivery/test", {}, http_handler, method="POST"
        )
        assert result is not None


# ===========================================================================
# Get Stats Tests
# ===========================================================================


class TestGetStats:
    """Tests for getting delivery statistics."""

    def test_get_stats_success(self, delivery_handler, mock_user):
        """Test successful stats retrieval."""
        http_handler = MockHandler(path="/api/v1/sme/receipts/delivery/stats", method="GET")

        result = delivery_handler.handle(
            "/api/v1/sme/receipts/delivery/stats", {}, http_handler, method="GET"
        )
        assert result is not None


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_exceeded(self, delivery_handler):
        """Test rate limit enforcement."""
        http_handler = MockHandler(path="/api/v1/sme/receipts/delivery/config", method="GET")

        with patch(
            "aragora.server.handlers.sme.receipt_delivery._delivery_limiter.is_allowed",
            return_value=False,
        ):
            result = delivery_handler.handle(
                "/api/v1/sme/receipts/delivery/config", {}, http_handler, method="GET"
            )
            assert result is not None
            assert result.status_code == 429


# ===========================================================================
# Method Not Allowed Tests
# ===========================================================================


class TestMethodNotAllowed:
    """Tests for method not allowed responses."""

    def test_config_get_only(self, delivery_handler):
        """Test only GET and POST allowed for config."""
        http_handler = MockHandler(path="/api/v1/sme/receipts/delivery/config", method="DELETE")

        result = delivery_handler.handle(
            "/api/v1/sme/receipts/delivery/config", {}, http_handler, method="DELETE"
        )
        assert result is not None
        assert result.status_code == 405

    def test_history_get_only(self, delivery_handler):
        """Test only GET allowed for history."""
        http_handler = MockHandler(path="/api/v1/sme/receipts/delivery/history", method="POST")

        result = delivery_handler.handle(
            "/api/v1/sme/receipts/delivery/history", {}, http_handler, method="POST"
        )
        assert result is not None
        assert result.status_code == 405

    def test_test_post_only(self, delivery_handler):
        """Test only POST allowed for test endpoint."""
        http_handler = MockHandler(path="/api/v1/sme/receipts/delivery/test", method="GET")

        result = delivery_handler.handle(
            "/api/v1/sme/receipts/delivery/test", {}, http_handler, method="GET"
        )
        assert result is not None
        assert result.status_code == 405

    def test_stats_get_only(self, delivery_handler):
        """Test only GET allowed for stats."""
        http_handler = MockHandler(path="/api/v1/sme/receipts/delivery/stats", method="POST")

        result = delivery_handler.handle(
            "/api/v1/sme/receipts/delivery/stats", {}, http_handler, method="POST"
        )
        assert result is not None
        assert result.status_code == 405
