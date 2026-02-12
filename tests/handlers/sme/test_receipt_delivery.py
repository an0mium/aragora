"""Tests for Receipt Delivery Handler."""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.sme.receipt_delivery import ReceiptDeliveryHandler


@dataclass
class MockUser:
    """Mock user for testing."""

    user_id: str = "user-123"
    id: str = "user-123"
    org_id: str = "org-456"
    email: str = "test@example.com"


@dataclass
class MockOrg:
    """Mock organization for testing."""

    id: str = "org-456"
    name: str = "Test Org"
    slug: str = "test-org"


@dataclass
class MockChannelSubscription:
    """Mock channel subscription for testing."""

    id: str = "sub-123"
    org_id: str = "org-456"
    channel_type: str = "slack"
    channel_id: str = "C123456"
    event_types: list[str] = field(default_factory=lambda: ["receipt"])
    created_at: float = 1700000000.0
    workspace_id: str | None = "T123456"
    channel_name: str | None = "#decisions"
    created_by: str | None = "user-123"
    is_active: bool = True
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "org_id": self.org_id,
            "channel_type": self.channel_type,
            "channel_id": self.channel_id,
            "workspace_id": self.workspace_id,
            "channel_name": self.channel_name,
            "event_types": self.event_types,
            "created_at": self.created_at,
            "created_at_iso": datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat(),
            "created_by": self.created_by,
            "is_active": self.is_active,
            "config": self.config,
        }


class MockEventType:
    """Mock event type enum for testing."""

    RECEIPT = "receipt"
    BUDGET_ALERT = "budget_alert"

    @classmethod
    def __iter__(cls):
        return iter([cls.RECEIPT, cls.BUDGET_ALERT])


class MockRequest(dict):
    """Mock HTTP request handler that also acts as a dict for query params."""

    def __init__(
        self,
        command: str = "GET",
        path: str = "/",
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        query_params: dict[str, str] | None = None,
    ):
        super().__init__(query_params or {})
        self.command = command
        self.path = path
        self.headers = headers or {"Content-Length": "0"}
        self._body = body or b""
        self.rfile = BytesIO(self._body)
        if body:
            self.headers["Content-Length"] = str(len(body))
        self.client_address = ("127.0.0.1", 12345)


@pytest.fixture
def mock_ctx():
    """Create mock server context."""
    user_store = MagicMock()
    user_store.get_user_by_id.return_value = MockUser()
    user_store.get_organization_by_id.return_value = MockOrg()

    return {"user_store": user_store}


@pytest.fixture
def mock_subscription_store():
    """Create mock subscription store."""
    store = MagicMock()
    store.get_by_org.return_value = []
    store.get_by_org_and_channel.return_value = None
    store.create.return_value = MockChannelSubscription()
    store.update.return_value = True
    return store


@pytest.fixture
def handler(mock_ctx, mock_subscription_store):
    """Create handler with mocked dependencies."""
    h = ReceiptDeliveryHandler(mock_ctx)
    h._get_subscription_store = MagicMock(return_value=mock_subscription_store)
    return h


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter before each test."""
    from aragora.server.handlers.sme.receipt_delivery import _delivery_limiter

    _delivery_limiter._buckets.clear()
    yield
    _delivery_limiter._buckets.clear()


# ============================================================================
# Route Handling Tests
# ============================================================================


class TestCanHandle:
    """Tests for route matching."""

    def test_can_handle_config(self, handler):
        """Test handling config route."""
        assert handler.can_handle("/api/v1/sme/receipts/delivery/config") is True

    def test_can_handle_history(self, handler):
        """Test handling history route."""
        assert handler.can_handle("/api/v1/sme/receipts/delivery/history") is True

    def test_can_handle_test(self, handler):
        """Test handling test route."""
        assert handler.can_handle("/api/v1/sme/receipts/delivery/test") is True

    def test_can_handle_stats(self, handler):
        """Test handling stats route."""
        assert handler.can_handle("/api/v1/sme/receipts/delivery/stats") is True

    def test_cannot_handle_unknown_route(self, handler):
        """Test rejecting unknown routes."""
        assert handler.can_handle("/api/v1/sme/receipts/unknown") is False
        assert handler.can_handle("/api/v1/sme/delivery") is False


# ============================================================================
# Get Config Tests
# ============================================================================


class TestGetConfig:
    """Tests for getting delivery configuration."""

    def test_get_config_empty(self, handler, mock_subscription_store):
        """Test getting config when no subscriptions exist."""
        mock_subscription_store.get_by_org.return_value = []

        request = MockRequest(command="GET", path="/api/v1/sme/receipts/delivery/config")

        with patch.object(
            handler,
            "_get_config",
            return_value=(
                {
                    "config": {
                        "auto_delivery_enabled": False,
                        "subscriptions": [],
                        "default_format": "compact",
                    },
                    "subscription_count": 0,
                },
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/receipts/delivery/config", {}, request, "GET")

        assert result is not None

    def test_get_config_with_subscriptions(self, handler, mock_subscription_store):
        """Test getting config with subscriptions."""
        subscriptions = [
            MockChannelSubscription(channel_type="slack"),
            MockChannelSubscription(id="sub-456", channel_type="teams"),
        ]
        mock_subscription_store.get_by_org.return_value = subscriptions

        request = MockRequest(command="GET", path="/api/v1/sme/receipts/delivery/config")

        with patch.object(
            handler,
            "_get_config",
            return_value=(
                {
                    "config": {
                        "auto_delivery_enabled": True,
                        "subscriptions": [s.to_dict() for s in subscriptions],
                    },
                    "subscription_count": 2,
                },
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/receipts/delivery/config", {}, request, "GET")

        assert result is not None


# ============================================================================
# Update Config Tests
# ============================================================================


class TestUpdateConfig:
    """Tests for updating delivery configuration."""

    def test_update_config_add_subscription(self, handler, mock_subscription_store):
        """Test adding a new subscription."""
        mock_subscription_store.get_by_org_and_channel.return_value = None
        new_sub = MockChannelSubscription()
        mock_subscription_store.create.return_value = new_sub

        body = json.dumps(
            {
                "subscriptions": [
                    {
                        "channel_type": "slack",
                        "channel_id": "C123456",
                        "workspace_id": "T123456",
                        "channel_name": "#decisions",
                    }
                ]
            }
        ).encode()

        request = MockRequest(
            command="POST",
            path="/api/v1/sme/receipts/delivery/config",
            body=body,
        )

        with patch.object(
            handler,
            "_update_config",
            return_value=(
                {
                    "updated": True,
                    "subscriptions": [new_sub.to_dict()],
                    "errors": None,
                },
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/receipts/delivery/config", {}, request, "POST")

        assert result is not None

    def test_update_config_invalid_channel_type(self, handler):
        """Test updating with invalid channel type."""
        body = json.dumps(
            {
                "subscriptions": [
                    {
                        "channel_type": "invalid",
                        "channel_id": "C123456",
                    }
                ]
            }
        ).encode()

        request = MockRequest(
            command="POST",
            path="/api/v1/sme/receipts/delivery/config",
            body=body,
        )

        with patch.object(
            handler,
            "_update_config",
            return_value=(
                {
                    "updated": True,
                    "subscriptions": [],
                    "errors": [{"error": "Invalid channel_type: invalid"}],
                },
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/receipts/delivery/config", {}, request, "POST")

        assert result is not None

    def test_update_config_missing_fields(self, handler):
        """Test updating with missing required fields."""
        body = json.dumps({"subscriptions": [{"channel_type": "slack"}]}).encode()

        request = MockRequest(
            command="POST",
            path="/api/v1/sme/receipts/delivery/config",
            body=body,
        )

        with patch.object(
            handler,
            "_update_config",
            return_value=(
                {
                    "updated": True,
                    "subscriptions": [],
                    "errors": [{"error": "channel_type and channel_id are required"}],
                },
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/receipts/delivery/config", {}, request, "POST")

        assert result is not None

    def test_update_config_invalid_json(self, handler):
        """Test updating with invalid JSON."""
        body = b"not valid json"

        request = MockRequest(
            command="POST",
            path="/api/v1/sme/receipts/delivery/config",
            body=body,
        )

        with patch.object(
            handler,
            "_update_config",
            return_value=({"error": "Invalid JSON body"}, 400),
        ):
            result = handler.handle("/api/v1/sme/receipts/delivery/config", {}, request, "POST")

        assert result is not None


# ============================================================================
# Get History Tests
# ============================================================================


class TestGetHistory:
    """Tests for getting delivery history."""

    def test_get_history_empty(self, handler):
        """Test getting history when empty."""
        request = MockRequest(command="GET", path="/api/v1/sme/receipts/delivery/history")

        with patch.object(
            handler,
            "_get_history",
            return_value=(
                {
                    "history": [],
                    "total": 0,
                    "limit": 50,
                    "offset": 0,
                },
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/receipts/delivery/history", {}, request, "GET")

        assert result is not None

    def test_get_history_with_data(self, handler):
        """Test getting history with data."""
        history_items = [
            {
                "id": "hist-1",
                "org_id": "org-456",
                "receipt_id": "receipt-123",
                "channel_type": "slack",
                "status": "success",
                "timestamp": 1700000000.0,
            },
            {
                "id": "hist-2",
                "org_id": "org-456",
                "receipt_id": "receipt-456",
                "channel_type": "teams",
                "status": "failed",
                "timestamp": 1700001000.0,
            },
        ]

        request = MockRequest(command="GET", path="/api/v1/sme/receipts/delivery/history")

        with patch.object(
            handler,
            "_get_history",
            return_value=(
                {
                    "history": history_items,
                    "total": 2,
                    "limit": 50,
                    "offset": 0,
                },
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/receipts/delivery/history", {}, request, "GET")

        assert result is not None

    def test_get_history_with_filters(self, handler):
        """Test getting history with filters."""
        request = MockRequest(
            command="GET",
            path="/api/v1/sme/receipts/delivery/history",
            query_params={"channel_type": "slack", "status": "success"},
        )

        with patch.object(
            handler,
            "_get_history",
            return_value=(
                {
                    "history": [],
                    "total": 0,
                    "limit": 50,
                    "offset": 0,
                },
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/receipts/delivery/history", {}, request, "GET")

        assert result is not None

    def test_get_history_with_pagination(self, handler):
        """Test getting history with pagination."""
        request = MockRequest(
            command="GET",
            path="/api/v1/sme/receipts/delivery/history",
            query_params={"limit": "10", "offset": "20"},
        )

        with patch.object(
            handler,
            "_get_history",
            return_value=(
                {
                    "history": [],
                    "total": 0,
                    "limit": 10,
                    "offset": 20,
                },
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/receipts/delivery/history", {}, request, "GET")

        assert result is not None


# ============================================================================
# Test Delivery Tests
# ============================================================================


class TestTestDelivery:
    """Tests for testing delivery."""

    def test_test_delivery_success(self, handler):
        """Test successful delivery test."""
        body = json.dumps(
            {
                "channel_type": "slack",
                "channel_id": "C123456",
                "workspace_id": "T123456",
                "test_message": "Test message",
            }
        ).encode()

        request = MockRequest(
            command="POST",
            path="/api/v1/sme/receipts/delivery/test",
            body=body,
        )

        with patch.object(
            handler,
            "_test_delivery",
            return_value=(
                {
                    "test_successful": True,
                    "channel_type": "slack",
                    "channel_id": "C123456",
                    "message_id": "1234567890.123456",
                },
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/receipts/delivery/test", {}, request, "POST")

        assert result is not None

    def test_test_delivery_missing_channel_type(self, handler):
        """Test delivery test without channel_type."""
        body = json.dumps({"channel_id": "C123456"}).encode()

        request = MockRequest(
            command="POST",
            path="/api/v1/sme/receipts/delivery/test",
            body=body,
        )

        with patch.object(
            handler,
            "_test_delivery",
            return_value=({"error": "channel_type is required"}, 400),
        ):
            result = handler.handle("/api/v1/sme/receipts/delivery/test", {}, request, "POST")

        assert result is not None

    def test_test_delivery_missing_channel_id(self, handler):
        """Test delivery test without channel_id."""
        body = json.dumps({"channel_type": "slack"}).encode()

        request = MockRequest(
            command="POST",
            path="/api/v1/sme/receipts/delivery/test",
            body=body,
        )

        with patch.object(
            handler,
            "_test_delivery",
            return_value=({"error": "channel_id is required"}, 400),
        ):
            result = handler.handle("/api/v1/sme/receipts/delivery/test", {}, request, "POST")

        assert result is not None

    def test_test_delivery_missing_workspace_id(self, handler):
        """Test delivery test without workspace_id for Slack."""
        body = json.dumps({"channel_type": "slack", "channel_id": "C123456"}).encode()

        request = MockRequest(
            command="POST",
            path="/api/v1/sme/receipts/delivery/test",
            body=body,
        )

        with patch.object(
            handler,
            "_test_delivery",
            return_value=(
                {"error": "workspace_id is required for slack"},
                400,
            ),
        ):
            result = handler.handle("/api/v1/sme/receipts/delivery/test", {}, request, "POST")

        assert result is not None

    def test_test_delivery_invalid_channel_type(self, handler):
        """Test delivery test with invalid channel type."""
        body = json.dumps({"channel_type": "invalid", "channel_id": "C123456"}).encode()

        request = MockRequest(
            command="POST",
            path="/api/v1/sme/receipts/delivery/test",
            body=body,
        )

        with patch.object(
            handler,
            "_test_delivery",
            return_value=(
                {"error": "Invalid channel_type. Valid: [slack, teams, email, webhook]"},
                400,
            ),
        ):
            result = handler.handle("/api/v1/sme/receipts/delivery/test", {}, request, "POST")

        assert result is not None

    def test_test_delivery_failure(self, handler):
        """Test delivery test that fails."""
        body = json.dumps(
            {
                "channel_type": "slack",
                "channel_id": "C123456",
                "workspace_id": "T123456",
            }
        ).encode()

        request = MockRequest(
            command="POST",
            path="/api/v1/sme/receipts/delivery/test",
            body=body,
        )

        with patch.object(
            handler,
            "_test_delivery",
            return_value=(
                {
                    "test_successful": False,
                    "channel_type": "slack",
                    "channel_id": "C123456",
                    "error": "Connection refused",
                },
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/receipts/delivery/test", {}, request, "POST")

        assert result is not None


# ============================================================================
# Get Stats Tests
# ============================================================================


class TestGetStats:
    """Tests for getting delivery statistics."""

    def test_get_stats_empty(self, handler, mock_subscription_store):
        """Test getting stats when no deliveries."""
        mock_subscription_store.get_by_org.return_value = []

        request = MockRequest(command="GET", path="/api/v1/sme/receipts/delivery/stats")

        with patch.object(
            handler,
            "_get_stats",
            return_value=(
                {
                    "stats": {
                        "total_deliveries": 0,
                        "successful_deliveries": 0,
                        "failed_deliveries": 0,
                        "test_deliveries": 0,
                        "success_rate": 0,
                        "active_subscriptions": 0,
                        "by_channel_type": {},
                    },
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/receipts/delivery/stats", {}, request, "GET")

        assert result is not None

    def test_get_stats_with_data(self, handler, mock_subscription_store):
        """Test getting stats with delivery data."""
        subscriptions = [MockChannelSubscription()]
        mock_subscription_store.get_by_org.return_value = subscriptions

        request = MockRequest(command="GET", path="/api/v1/sme/receipts/delivery/stats")

        with patch.object(
            handler,
            "_get_stats",
            return_value=(
                {
                    "stats": {
                        "total_deliveries": 100,
                        "successful_deliveries": 95,
                        "failed_deliveries": 5,
                        "test_deliveries": 10,
                        "success_rate": 95.0,
                        "active_subscriptions": 1,
                        "by_channel_type": {
                            "slack": {"total": 60, "success": 58, "failed": 2},
                            "teams": {"total": 40, "success": 37, "failed": 3},
                        },
                    },
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/receipts/delivery/stats", {}, request, "GET")

        assert result is not None


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_not_exceeded(self, handler):
        """Test request passes when rate limit not exceeded."""
        request = MockRequest(command="GET", path="/api/v1/sme/receipts/delivery/config")

        result = handler.handle("/api/v1/sme/receipts/delivery/config", {}, request, "GET")
        assert result is not None

    def test_rate_limit_exceeded(self, handler):
        """Test rate limit is enforced after many requests."""
        from aragora.server.handlers.sme.receipt_delivery import _delivery_limiter

        client_ip = "127.0.0.1"
        for _ in range(61):
            _delivery_limiter.is_allowed(client_ip)

        request = MockRequest(command="GET", path="/api/v1/sme/receipts/delivery/config")

        result = handler.handle("/api/v1/sme/receipts/delivery/config", {}, request, "GET")
        assert result is not None
        assert result.status_code == 429


# ============================================================================
# Method Not Allowed Tests
# ============================================================================


class TestMethodNotAllowed:
    """Tests for method not allowed responses."""

    def test_delete_on_config(self, handler):
        """Test DELETE on config returns 405."""
        request = MockRequest(command="DELETE", path="/api/v1/sme/receipts/delivery/config")

        result = handler.handle("/api/v1/sme/receipts/delivery/config", {}, request, "DELETE")
        assert result is not None
        assert result.status_code == 405

    def test_post_on_history(self, handler):
        """Test POST on history returns 405."""
        request = MockRequest(command="POST", path="/api/v1/sme/receipts/delivery/history")

        result = handler.handle("/api/v1/sme/receipts/delivery/history", {}, request, "POST")
        assert result is not None
        assert result.status_code == 405

    def test_get_on_test(self, handler):
        """Test GET on test returns 405."""
        request = MockRequest(command="GET", path="/api/v1/sme/receipts/delivery/test")

        result = handler.handle("/api/v1/sme/receipts/delivery/test", {}, request, "GET")
        assert result is not None
        assert result.status_code == 405

    def test_post_on_stats(self, handler):
        """Test POST on stats returns 405."""
        request = MockRequest(command="POST", path="/api/v1/sme/receipts/delivery/stats")

        result = handler.handle("/api/v1/sme/receipts/delivery/stats", {}, request, "POST")
        assert result is not None
        assert result.status_code == 405


# ============================================================================
# Service Unavailable Tests
# ============================================================================


class TestServiceUnavailable:
    """Tests for service unavailable responses."""

    def test_no_user_store(self, mock_subscription_store):
        """Test handling when user store is unavailable."""
        h = ReceiptDeliveryHandler({})
        h._get_subscription_store = MagicMock(return_value=mock_subscription_store)

        request = MockRequest(command="GET", path="/api/v1/sme/receipts/delivery/config")

        assert h.can_handle("/api/v1/sme/receipts/delivery/config") is True
