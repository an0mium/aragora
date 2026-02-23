"""Tests for e-commerce platform handler.

Tests the e-commerce API endpoints including:
- GET  /api/v1/ecommerce/platforms          - List connected platforms
- POST /api/v1/ecommerce/connect            - Connect a platform
- DELETE /api/v1/ecommerce/{platform}       - Disconnect platform
- GET  /api/v1/ecommerce/orders             - List orders (cross-platform)
- GET  /api/v1/ecommerce/{platform}/orders  - Platform orders
- GET  /api/v1/ecommerce/{platform}/orders/{id} - Get order details
- GET  /api/v1/ecommerce/products           - List products
- GET  /api/v1/ecommerce/{platform}/products - Platform products
- GET  /api/v1/ecommerce/{platform}/products/{id} - Get product details
- GET  /api/v1/ecommerce/inventory          - Get inventory levels
- POST /api/v1/ecommerce/sync-inventory     - Sync inventory across platforms
- GET  /api/v1/ecommerce/fulfillment        - Get fulfillment status
- POST /api/v1/ecommerce/ship               - Create shipment
- GET  /api/v1/ecommerce/metrics            - Get e-commerce metrics
- GET  /api/v1/ecommerce/circuit-breaker    - Get circuit breaker status
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class MockRequest:
    """Mock HTTP request for testing the ecommerce handler."""

    path: str = "/api/v1/ecommerce/platforms"
    method: str = "GET"
    query: dict[str, Any] = field(default_factory=dict)
    _body: dict[str, Any] | None = None

    async def json(self) -> dict[str, Any]:
        return self._body or {}


def _status(result: dict[str, Any]) -> int:
    """Extract status code from handler response dict."""
    return result.get("status_code", 0)


def _body(result: dict[str, Any]) -> dict[str, Any]:
    """Extract body from handler response dict."""
    return result.get("body", {})


def _error(result: dict[str, Any]) -> str:
    """Extract error message from handler response dict."""
    return _body(result).get("error", "")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create an EcommerceHandler instance with empty context."""
    from aragora.server.handlers.features.ecommerce.handler import EcommerceHandler

    return EcommerceHandler(ctx={})


@pytest.fixture(autouse=True)
def reset_ecommerce_state():
    """Reset global platform state before/after each test."""
    from aragora.server.handlers.features.ecommerce.models import (
        _platform_credentials,
        _platform_connectors,
    )

    _platform_credentials.clear()
    _platform_connectors.clear()
    yield
    _platform_credentials.clear()
    _platform_connectors.clear()


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset rate limiter state between tests."""
    try:
        from aragora.server.middleware.rate_limit.registry import (
            reset_rate_limiters as _reset,
        )

        _reset()
    except ImportError:
        pass
    yield
    try:
        from aragora.server.middleware.rate_limit.registry import (
            reset_rate_limiters as _reset,
        )

        _reset()
    except ImportError:
        pass


@pytest.fixture
def connected_shopify():
    """Pre-connect the shopify platform so endpoints that need it pass."""
    from aragora.server.handlers.features.ecommerce.models import _platform_credentials

    _platform_credentials["shopify"] = {
        "credentials": {
            "shop_url": "https://myshop.myshopify.com",
            "access_token": "shpat_abc123",
        },
        "connected_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def connected_shipstation():
    """Pre-connect the shipstation platform."""
    from aragora.server.handlers.features.ecommerce.models import _platform_credentials

    _platform_credentials["shipstation"] = {
        "credentials": {
            "api_key": "key123",
            "api_secret": "secret456",
        },
        "connected_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def connected_walmart():
    """Pre-connect the walmart platform."""
    from aragora.server.handlers.features.ecommerce.models import _platform_credentials

    _platform_credentials["walmart"] = {
        "credentials": {
            "client_id": "wm-client-id",
            "client_secret": "wm-client-secret",
        },
        "connected_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def mock_connector():
    """Create a generic mock connector with async methods."""
    connector = AsyncMock()
    connector.get_orders = AsyncMock(return_value=[])
    connector.get_order = AsyncMock(return_value=MagicMock())
    connector.get_products = AsyncMock(return_value=[])
    connector.get_product = AsyncMock(return_value=MagicMock())
    connector.get_items = AsyncMock(return_value=[])
    connector.get_item = AsyncMock(return_value=MagicMock())
    connector.get_inventory_levels = AsyncMock(return_value=[])
    connector.get_inventory = AsyncMock(return_value=[])
    connector.update_inventory = AsyncMock()
    connector.get_shipments = AsyncMock(return_value=[])
    connector.create_label = AsyncMock(return_value=MagicMock())
    connector.create_fulfillment = AsyncMock(return_value=MagicMock())
    connector.close = AsyncMock()
    return connector


# ---------------------------------------------------------------------------
# can_handle() Routing
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Test can_handle routing for all e-commerce paths."""

    def test_platforms_path(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/platforms")

    def test_connect_path(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/connect", "POST")

    def test_disconnect_path(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/shopify", "DELETE")

    def test_orders_path(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/orders")

    def test_platform_orders_path(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/shopify/orders")

    def test_platform_order_detail_path(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/shopify/orders/12345")

    def test_products_path(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/products")

    def test_platform_products_path(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/shopify/products")

    def test_platform_product_detail_path(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/shopify/products/p-123")

    def test_inventory_path(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/inventory")

    def test_sync_inventory_path(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/sync-inventory", "POST")

    def test_fulfillment_path(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/fulfillment")

    def test_ship_path(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/ship", "POST")

    def test_metrics_path(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/metrics")

    def test_circuit_breaker_path(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/circuit-breaker")

    def test_rejects_non_ecommerce_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_rejects_other_handler_path(self, handler):
        assert not handler.can_handle("/api/v1/users/me")

    def test_rejects_root(self, handler):
        assert not handler.can_handle("/")

    def test_routes_list_is_populated(self, handler):
        assert len(handler.ROUTES) >= 15


# ---------------------------------------------------------------------------
# GET /api/v1/ecommerce/platforms
# ---------------------------------------------------------------------------


class TestListPlatforms:
    """Test listing supported platforms."""

    @pytest.mark.asyncio
    async def test_list_platforms_success(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/platforms", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert "platforms" in body
        assert len(body["platforms"]) == 3  # shopify, shipstation, walmart

    @pytest.mark.asyncio
    async def test_list_platforms_none_connected(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/platforms", method="GET")
        result = await handler.handle_request(req)
        body = _body(result)
        assert body["connected_count"] == 0
        for p in body["platforms"]:
            assert p["connected"] is False

    @pytest.mark.asyncio
    async def test_list_platforms_one_connected(self, handler, connected_shopify):
        req = MockRequest(path="/api/v1/ecommerce/platforms", method="GET")
        result = await handler.handle_request(req)
        body = _body(result)
        assert body["connected_count"] == 1

    @pytest.mark.asyncio
    async def test_platform_metadata_present(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/platforms", method="GET")
        result = await handler.handle_request(req)
        platforms = _body(result)["platforms"]
        for p in platforms:
            assert "id" in p
            assert "name" in p
            assert "description" in p
            assert "features" in p
            assert isinstance(p["features"], list)


# ---------------------------------------------------------------------------
# POST /api/v1/ecommerce/connect
# ---------------------------------------------------------------------------


class TestConnectPlatform:
    """Test connecting an e-commerce platform."""

    @pytest.mark.asyncio
    async def test_connect_shopify_success(self, handler):
        req = MockRequest(
            path="/api/v1/ecommerce/connect",
            method="POST",
            _body={
                "platform": "shopify",
                "credentials": {
                    "shop_url": "https://myshop.myshopify.com",
                    "access_token": "shpat_abc",
                },
            },
        )
        with patch.object(handler, "_get_json_body", return_value=req._body):
            with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
                result = await handler.handle_request(req)
        assert _status(result) == 200
        assert "connected_at" in _body(result)

    @pytest.mark.asyncio
    async def test_connect_shipstation_success(self, handler):
        req = MockRequest(
            path="/api/v1/ecommerce/connect",
            method="POST",
            _body={
                "platform": "shipstation",
                "credentials": {
                    "api_key": "key",
                    "api_secret": "secret",
                },
            },
        )
        with patch.object(handler, "_get_json_body", return_value=req._body):
            with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
                result = await handler.handle_request(req)
        assert _status(result) == 200
        assert _body(result)["platform"] == "shipstation"

    @pytest.mark.asyncio
    async def test_connect_walmart_success(self, handler):
        req = MockRequest(
            path="/api/v1/ecommerce/connect",
            method="POST",
            _body={
                "platform": "walmart",
                "credentials": {
                    "client_id": "cid",
                    "client_secret": "csec",
                },
            },
        )
        with patch.object(handler, "_get_json_body", return_value=req._body):
            with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
                result = await handler.handle_request(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_connect_missing_platform(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/connect", method="POST")
        with patch.object(handler, "_get_json_body", return_value={}):
            result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "required" in _error(result).lower()

    @pytest.mark.asyncio
    async def test_connect_unsupported_platform(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/connect", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={"platform": "etsy", "credentials": {"key": "val"}},
        ):
            result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "unsupported" in _error(result).lower()

    @pytest.mark.asyncio
    async def test_connect_missing_credentials(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/connect", method="POST")
        with patch.object(handler, "_get_json_body", return_value={"platform": "shopify"}):
            result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "credentials" in _error(result).lower()

    @pytest.mark.asyncio
    async def test_connect_missing_required_credential_fields(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/connect", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={
                "platform": "shopify",
                "credentials": {"shop_url": "https://test.myshopify.com"},
            },
        ):
            result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "missing" in _error(result).lower()

    @pytest.mark.asyncio
    async def test_connect_invalid_json_body(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/connect", method="POST")
        with patch.object(handler, "_get_json_body", side_effect=ValueError("bad json")):
            result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_connect_credential_not_string(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/connect", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={
                "platform": "shipstation",
                "credentials": {"api_key": 12345, "api_secret": "secret"},
            },
        ):
            result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "string" in _error(result).lower()

    @pytest.mark.asyncio
    async def test_connect_credential_empty_value(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/connect", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={
                "platform": "shipstation",
                "credentials": {"api_key": "  ", "api_secret": "secret"},
            },
        ):
            result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "empty" in _error(result).lower()

    @pytest.mark.asyncio
    async def test_connect_credential_too_long(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/connect", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={
                "platform": "shipstation",
                "credentials": {"api_key": "x" * 2000, "api_secret": "secret"},
            },
        ):
            result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "length" in _error(result).lower()

    @pytest.mark.asyncio
    async def test_connect_credentials_not_dict(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/connect", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={"platform": "shopify", "credentials": "not-a-dict"},
        ):
            result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "object" in _error(result).lower()

    @pytest.mark.asyncio
    async def test_connect_shopify_invalid_url(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/connect", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={
                "platform": "shopify",
                "credentials": {
                    "shop_url": "not-a-url",
                    "access_token": "shpat_abc",
                },
            },
        ):
            result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_connect_invalid_platform_id_format(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/connect", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={"platform": "shop!fy", "credentials": {"k": "v"}},
        ):
            result = await handler.handle_request(req)
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# DELETE /api/v1/ecommerce/{platform}
# ---------------------------------------------------------------------------


class TestDisconnectPlatform:
    """Test disconnecting an e-commerce platform."""

    @pytest.mark.asyncio
    async def test_disconnect_success(self, handler, connected_shopify):
        req = MockRequest(path="/api/v1/ecommerce/shopify", method="DELETE")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        assert "disconnected" in _body(result).get("message", "").lower()

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/shopify", method="DELETE")
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_disconnect_closes_connector(self, handler, connected_shopify, mock_connector):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        _platform_connectors["shopify"] = mock_connector
        req = MockRequest(path="/api/v1/ecommerce/shopify", method="DELETE")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        mock_connector.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up_credentials(self, handler, connected_shopify):
        from aragora.server.handlers.features.ecommerce.models import _platform_credentials

        req = MockRequest(path="/api/v1/ecommerce/shopify", method="DELETE")
        await handler.handle_request(req)
        assert "shopify" not in _platform_credentials


# ---------------------------------------------------------------------------
# GET /api/v1/ecommerce/orders  (cross-platform)
# ---------------------------------------------------------------------------


class TestListAllOrders:
    """Test listing orders across all platforms."""

    @pytest.mark.asyncio
    async def test_list_orders_no_platforms(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/orders", method="GET", query={})
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["orders"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_orders_with_connected_platform(
        self, handler, connected_shopify, mock_connector
    ):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        mock_connector.get_orders = AsyncMock(return_value=[])
        _platform_connectors["shopify"] = mock_connector

        req = MockRequest(path="/api/v1/ecommerce/orders", method="GET", query={})
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_list_orders_pagination_params(self, handler):
        req = MockRequest(
            path="/api/v1/ecommerce/orders",
            method="GET",
            query={"limit": "10", "offset": "5"},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        body = _body(result)
        assert body["limit"] == 10
        assert body["offset"] == 5

    @pytest.mark.asyncio
    async def test_list_orders_invalid_pagination(self, handler):
        req = MockRequest(
            path="/api/v1/ecommerce/orders",
            method="GET",
            query={"limit": "-1"},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_list_orders_circuit_breaker_open(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/orders", method="GET", query={})
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = False
            result = await handler.handle_request(req)
        assert _status(result) == 503
        assert "unavailable" in _error(result).lower()


# ---------------------------------------------------------------------------
# GET /api/v1/ecommerce/{platform}/orders
# ---------------------------------------------------------------------------


class TestListPlatformOrders:
    """Test listing orders for a specific platform."""

    @pytest.mark.asyncio
    async def test_platform_orders_success(self, handler, connected_shopify, mock_connector):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        mock_connector.get_orders = AsyncMock(return_value=[])
        _platform_connectors["shopify"] = mock_connector

        req = MockRequest(
            path="/api/v1/ecommerce/shopify/orders",
            method="GET",
            query={},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 200
        assert _body(result)["platform"] == "shopify"

    @pytest.mark.asyncio
    async def test_platform_orders_not_connected(self, handler):
        req = MockRequest(
            path="/api/v1/ecommerce/shopify/orders",
            method="GET",
            query={},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_platform_orders_with_status_filter(
        self, handler, connected_shopify, mock_connector
    ):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        mock_connector.get_orders = AsyncMock(return_value=[])
        _platform_connectors["shopify"] = mock_connector

        req = MockRequest(
            path="/api/v1/ecommerce/shopify/orders",
            method="GET",
            query={"status": "fulfilled"},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# GET /api/v1/ecommerce/{platform}/orders/{order_id}
# ---------------------------------------------------------------------------


class TestGetOrder:
    """Test getting a specific order."""

    @pytest.mark.asyncio
    async def test_get_order_shopify(self, handler, connected_shopify, mock_connector):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        mock_order = MagicMock()
        mock_order.id = 12345
        mock_order.order_number = "1001"
        mock_order.fulfillment_status = "fulfilled"
        mock_order.financial_status = "paid"
        mock_order.email = "buyer@test.com"
        mock_order.customer = MagicMock(first_name="John", last_name="Doe")
        mock_order.total_price = "99.99"
        mock_order.subtotal_price = "89.99"
        mock_order.total_tax = "10.00"
        mock_order.currency = "USD"
        mock_order.line_items = []
        mock_order.created_at = datetime.now(timezone.utc)
        mock_order.updated_at = datetime.now(timezone.utc)
        mock_order.total_shipping_price_set = MagicMock()
        mock_order.total_shipping_price_set.shop_money.amount = "5.00"

        mock_connector.get_order = AsyncMock(return_value=mock_order)
        _platform_connectors["shopify"] = mock_connector

        req = MockRequest(
            path="/api/v1/ecommerce/shopify/orders/12345",
            method="GET",
            query={},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 200
        assert _body(result)["platform"] == "shopify"

    @pytest.mark.asyncio
    async def test_get_order_not_connected(self, handler):
        req = MockRequest(
            path="/api/v1/ecommerce/shopify/orders/12345",
            method="GET",
            query={},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_order_connector_unavailable(self, handler, connected_shopify):
        req = MockRequest(
            path="/api/v1/ecommerce/shopify/orders/12345",
            method="GET",
            query={},
        )
        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_order_connection_error(self, handler, connected_shopify, mock_connector):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        mock_connector.get_order = AsyncMock(side_effect=ConnectionError("timeout"))
        _platform_connectors["shopify"] = mock_connector

        req = MockRequest(
            path="/api/v1/ecommerce/shopify/orders/12345",
            method="GET",
            query={},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_order_value_error(self, handler, connected_shopify, mock_connector):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        mock_connector.get_order = AsyncMock(side_effect=ValueError("bad id"))
        _platform_connectors["shopify"] = mock_connector

        req = MockRequest(
            path="/api/v1/ecommerce/shopify/orders/12345",
            method="GET",
            query={},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_order_not_found(self, handler, connected_shopify, mock_connector):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        mock_connector.get_order = AsyncMock(side_effect=KeyError("not found"))
        _platform_connectors["shopify"] = mock_connector

        req = MockRequest(
            path="/api/v1/ecommerce/shopify/orders/12345",
            method="GET",
            query={},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# GET /api/v1/ecommerce/products  (cross-platform)
# ---------------------------------------------------------------------------


class TestListAllProducts:
    """Test listing products across all platforms."""

    @pytest.mark.asyncio
    async def test_list_products_empty(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/products", method="GET", query={})
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 200
        assert _body(result)["products"] == []
        assert _body(result)["total"] == 0

    @pytest.mark.asyncio
    async def test_list_products_invalid_pagination(self, handler):
        req = MockRequest(
            path="/api/v1/ecommerce/products",
            method="GET",
            query={"limit": "abc"},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# GET /api/v1/ecommerce/{platform}/products
# ---------------------------------------------------------------------------


class TestListPlatformProducts:
    """Test listing products for a specific platform."""

    @pytest.mark.asyncio
    async def test_platform_products_success(self, handler, connected_shopify, mock_connector):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        mock_connector.get_products = AsyncMock(return_value=[])
        _platform_connectors["shopify"] = mock_connector

        req = MockRequest(
            path="/api/v1/ecommerce/shopify/products",
            method="GET",
            query={},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 200
        assert _body(result)["platform"] == "shopify"

    @pytest.mark.asyncio
    async def test_platform_products_not_connected(self, handler):
        req = MockRequest(
            path="/api/v1/ecommerce/shopify/products",
            method="GET",
            query={},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# GET /api/v1/ecommerce/{platform}/products/{product_id}
# ---------------------------------------------------------------------------


class TestGetProduct:
    """Test getting a specific product."""

    @pytest.mark.asyncio
    async def test_get_product_shopify(self, handler, connected_shopify, mock_connector):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        mock_product = MagicMock()
        mock_product.id = 999
        mock_product.title = "Test Widget"
        mock_product.status = "active"
        mock_product.vendor = "TestVendor"
        mock_product.product_type = "Widget"
        mock_product.tags = "sale, new"
        mock_product.created_at = datetime.now(timezone.utc)
        variant = MagicMock()
        variant.sku = "SKU-001"
        variant.barcode = "1234567890"
        variant.price = "19.99"
        variant.compare_at_price = "29.99"
        variant.inventory_quantity = 50
        mock_product.variants = [variant]
        mock_product.images = []

        mock_connector.get_product = AsyncMock(return_value=mock_product)
        _platform_connectors["shopify"] = mock_connector

        req = MockRequest(
            path="/api/v1/ecommerce/shopify/products/999",
            method="GET",
            query={},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 200
        assert _body(result)["title"] == "Test Widget"

    @pytest.mark.asyncio
    async def test_get_product_not_connected(self, handler):
        req = MockRequest(
            path="/api/v1/ecommerce/shopify/products/999",
            method="GET",
            query={},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_product_connector_unavailable(self, handler, connected_shopify):
        req = MockRequest(
            path="/api/v1/ecommerce/shopify/products/999",
            method="GET",
            query={},
        )
        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_product_connection_error(self, handler, connected_shopify, mock_connector):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        mock_connector.get_product = AsyncMock(side_effect=ConnectionError("down"))
        _platform_connectors["shopify"] = mock_connector

        req = MockRequest(
            path="/api/v1/ecommerce/shopify/products/999",
            method="GET",
            query={},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_product_value_error(self, handler, connected_shopify, mock_connector):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        mock_connector.get_product = AsyncMock(side_effect=ValueError("bad"))
        _platform_connectors["shopify"] = mock_connector

        req = MockRequest(
            path="/api/v1/ecommerce/shopify/products/999",
            method="GET",
            query={},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_product_not_found_error(self, handler, connected_shopify, mock_connector):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        mock_connector.get_product = AsyncMock(side_effect=KeyError("missing"))
        _platform_connectors["shopify"] = mock_connector

        req = MockRequest(
            path="/api/v1/ecommerce/shopify/products/999",
            method="GET",
            query={},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# GET /api/v1/ecommerce/inventory
# ---------------------------------------------------------------------------


class TestGetInventory:
    """Test getting inventory levels."""

    @pytest.mark.asyncio
    async def test_inventory_empty(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/inventory", method="GET", query={})
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 200
        assert _body(result)["inventory"] == {}

    @pytest.mark.asyncio
    async def test_inventory_with_sku_filter(self, handler, connected_walmart, mock_connector):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        mock_connector.get_inventory = AsyncMock(return_value=[])
        _platform_connectors["walmart"] = mock_connector

        req = MockRequest(
            path="/api/v1/ecommerce/inventory",
            method="GET",
            query={"sku": "SKU-001"},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_inventory_invalid_sku(self, handler):
        req = MockRequest(
            path="/api/v1/ecommerce/inventory",
            method="GET",
            query={"sku": "!invalid!"},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_inventory_platform_error_graceful(
        self, handler, connected_shopify, mock_connector
    ):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        mock_connector.get_inventory_levels = AsyncMock(side_effect=ConnectionError("down"))
        _platform_connectors["shopify"] = mock_connector

        req = MockRequest(path="/api/v1/ecommerce/inventory", method="GET", query={})
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 200
        # Error should be captured in the inventory data, not fail the whole request
        assert "shopify" in _body(result)["inventory"]
        assert "error" in _body(result)["inventory"]["shopify"][0]


# ---------------------------------------------------------------------------
# POST /api/v1/ecommerce/sync-inventory
# ---------------------------------------------------------------------------


class TestSyncInventory:
    """Test syncing inventory across platforms."""

    @pytest.mark.asyncio
    async def test_sync_inventory_success(self, handler, connected_walmart, mock_connector):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        _platform_connectors["walmart"] = mock_connector

        req = MockRequest(path="/api/v1/ecommerce/sync-inventory", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={
                "sku": "SKU-001",
                "quantity": 42,
                "target_platforms": ["walmart"],
            },
        ):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 200
        assert _body(result)["sku"] == "SKU-001"
        assert _body(result)["quantity"] == 42

    @pytest.mark.asyncio
    async def test_sync_inventory_missing_sku(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/sync-inventory", method="POST")
        with patch.object(handler, "_get_json_body", return_value={"quantity": 10}):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_sync_inventory_no_quantity_no_source(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/sync-inventory", method="POST")
        with patch.object(handler, "_get_json_body", return_value={"sku": "SKU-001"}):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_sync_inventory_invalid_json(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/sync-inventory", method="POST")
        with patch.object(handler, "_get_json_body", side_effect=ValueError("bad")):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_sync_inventory_negative_quantity(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/sync-inventory", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={"sku": "SKU-001", "quantity": -5},
        ):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_sync_inventory_invalid_sku_format(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/sync-inventory", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={"sku": "!bad!", "quantity": 10},
        ):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_sync_inventory_target_not_list(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/sync-inventory", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={
                "sku": "SKU-001",
                "quantity": 10,
                "target_platforms": "walmart",
            },
        ):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_sync_inventory_too_many_targets(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/sync-inventory", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={
                "sku": "SKU-001",
                "quantity": 10,
                "target_platforms": [f"platform{i}" for i in range(51)],
            },
        ):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_sync_inventory_unsupported_source_platform(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/sync-inventory", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={
                "sku": "SKU-001",
                "source_platform": "etsy",
            },
        ):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_sync_inventory_target_not_connected(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/sync-inventory", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={
                "sku": "SKU-001",
                "quantity": 5,
                "target_platforms": ["walmart"],
            },
        ):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 200
        results = _body(result)["results"]
        assert results["walmart"]["error"] == "Not connected"


# ---------------------------------------------------------------------------
# GET /api/v1/ecommerce/fulfillment
# ---------------------------------------------------------------------------


class TestGetFulfillmentStatus:
    """Test getting fulfillment status."""

    @pytest.mark.asyncio
    async def test_fulfillment_empty(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/fulfillment", method="GET", query={})
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 200
        assert _body(result)["fulfillments"] == []
        assert _body(result)["total"] == 0

    @pytest.mark.asyncio
    async def test_fulfillment_with_order_id(self, handler, connected_shipstation, mock_connector):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        _platform_connectors["shipstation"] = mock_connector

        req = MockRequest(
            path="/api/v1/ecommerce/fulfillment",
            method="GET",
            query={"order_id": "12345"},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_fulfillment_with_platform_filter(
        self, handler, connected_shipstation, mock_connector
    ):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        _platform_connectors["shipstation"] = mock_connector

        req = MockRequest(
            path="/api/v1/ecommerce/fulfillment",
            method="GET",
            query={"platform": "shipstation"},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_fulfillment_invalid_order_id(self, handler):
        req = MockRequest(
            path="/api/v1/ecommerce/fulfillment",
            method="GET",
            query={"order_id": "!bad!"},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_fulfillment_invalid_platform_filter(self, handler):
        req = MockRequest(
            path="/api/v1/ecommerce/fulfillment",
            method="GET",
            query={"platform": "!!!"},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# POST /api/v1/ecommerce/ship
# ---------------------------------------------------------------------------


class TestCreateShipment:
    """Test creating a shipment."""

    @pytest.mark.asyncio
    async def test_create_shipment_success(self, handler, connected_shipstation, mock_connector):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        label = MagicMock()
        label.shipment_id = 789
        label.tracking_number = "1Z999AA10123456784"
        label.label_data = "https://label.example.com/pdf"
        mock_connector.create_label = AsyncMock(return_value=label)
        _platform_connectors["shipstation"] = mock_connector

        req = MockRequest(path="/api/v1/ecommerce/ship", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={
                "platform": "shipstation",
                "order_id": "12345",
                "carrier": "ups",
                "service": "ground",
            },
        ):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 201
        body = _body(result)
        assert body["tracking_number"] == "1Z999AA10123456784"
        assert body["platform"] == "shipstation"

    @pytest.mark.asyncio
    async def test_create_shipment_missing_order_id(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/ship", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={"platform": "shipstation"},
        ):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "order_id" in _error(result).lower()

    @pytest.mark.asyncio
    async def test_create_shipment_invalid_carrier(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/ship", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={
                "platform": "shipstation",
                "order_id": "12345",
                "carrier": "inv@lid!",
            },
        ):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "carrier" in _error(result).lower()

    @pytest.mark.asyncio
    async def test_create_shipment_invalid_service(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/ship", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={
                "platform": "shipstation",
                "order_id": "12345",
                "service": "bad service!",
            },
        ):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "service" in _error(result).lower()

    @pytest.mark.asyncio
    async def test_create_shipment_tracking_too_long(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/ship", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={
                "platform": "shipstation",
                "order_id": "12345",
                "tracking_number": "T" * 200,
            },
        ):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "tracking" in _error(result).lower()

    @pytest.mark.asyncio
    async def test_create_shipment_not_connected(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/ship", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={
                "platform": "shipstation",
                "order_id": "12345",
            },
        ):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_create_shipment_connection_error(
        self, handler, connected_shipstation, mock_connector
    ):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        mock_connector.create_label = AsyncMock(side_effect=ConnectionError("down"))
        _platform_connectors["shipstation"] = mock_connector

        req = MockRequest(path="/api/v1/ecommerce/ship", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={
                "platform": "shipstation",
                "order_id": "12345",
            },
        ):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_create_shipment_runtime_error(
        self, handler, connected_shipstation, mock_connector
    ):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        mock_connector.create_label = AsyncMock(side_effect=RuntimeError("oops"))
        _platform_connectors["shipstation"] = mock_connector

        req = MockRequest(path="/api/v1/ecommerce/ship", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={
                "platform": "shipstation",
                "order_id": "12345",
            },
        ):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_create_shipment_invalid_json(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/ship", method="POST")
        with patch.object(handler, "_get_json_body", side_effect=ValueError("bad json")):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_shipment_invalid_declared_value(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/ship", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={
                "platform": "shipstation",
                "order_id": "12345",
                "declared_value": -100,
            },
        ):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_shipment_invalid_currency(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/ship", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={
                "platform": "shipstation",
                "order_id": "12345",
                "currency": "XYZ",
            },
        ):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_shipment_shopify_fulfillment(
        self, handler, connected_shopify, mock_connector
    ):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        fulfillment = MagicMock()
        fulfillment.id = 555
        fulfillment.status = "success"
        mock_connector.create_fulfillment = AsyncMock(return_value=fulfillment)
        _platform_connectors["shopify"] = mock_connector

        req = MockRequest(path="/api/v1/ecommerce/ship", method="POST")
        with patch.object(
            handler,
            "_get_json_body",
            return_value={
                "platform": "shopify",
                "order_id": "12345",
                "carrier": "ups",
            },
        ):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 201
        assert _body(result)["platform"] == "shopify"
        assert _body(result)["fulfillment_id"] == "555"


# ---------------------------------------------------------------------------
# GET /api/v1/ecommerce/metrics
# ---------------------------------------------------------------------------


class TestGetMetrics:
    """Test getting e-commerce metrics."""

    @pytest.mark.asyncio
    async def test_metrics_no_platforms(self, handler):
        req = MockRequest(path="/api/v1/ecommerce/metrics", method="GET", query={})
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["totals"]["total_orders"] == 0
        assert body["totals"]["total_revenue"] == 0

    @pytest.mark.asyncio
    async def test_metrics_with_days_param(self, handler):
        req = MockRequest(
            path="/api/v1/ecommerce/metrics",
            method="GET",
            query={"days": "7"},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 200
        assert _body(result)["period_days"] == 7

    @pytest.mark.asyncio
    async def test_metrics_platform_error_captured(
        self, handler, connected_shopify, mock_connector
    ):
        """When _fetch_platform_metrics raises, the error is captured gracefully."""
        req = MockRequest(path="/api/v1/ecommerce/metrics", method="GET", query={})
        with patch.object(
            handler,
            "_fetch_platform_metrics",
            new_callable=AsyncMock,
            side_effect=ConnectionError("down"),
        ):
            with patch.object(handler, "_circuit_breaker") as cb:
                cb.can_proceed.return_value = True
                result = await handler.handle_request(req)
        assert _status(result) == 200
        assert "error" in _body(result)["platforms"].get("shopify", {})

    @pytest.mark.asyncio
    async def test_metrics_orders_error_returns_zero_metrics(
        self, handler, connected_shopify, mock_connector
    ):
        """When platform order fetch fails silently, metrics show zeros."""
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        mock_connector.get_orders = AsyncMock(side_effect=ConnectionError("down"))
        _platform_connectors["shopify"] = mock_connector

        req = MockRequest(path="/api/v1/ecommerce/metrics", method="GET", query={})
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 200
        shopify_metrics = _body(result)["platforms"].get("shopify", {})
        assert shopify_metrics.get("total_orders") == 0


# ---------------------------------------------------------------------------
# GET /api/v1/ecommerce/circuit-breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    """Test circuit breaker status endpoint."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_status(self, handler):
        req = MockRequest(
            path="/api/v1/ecommerce/circuit-breaker",
            method="GET",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        # Should contain status information
        body = _body(result)
        assert isinstance(body, dict)

    def test_get_circuit_breaker_status_method(self, handler):
        status = handler.get_circuit_breaker_status()
        assert isinstance(status, dict)


# ---------------------------------------------------------------------------
# 404 / Unknown Endpoint
# ---------------------------------------------------------------------------


class TestUnknownEndpoint:
    """Test that unrecognized paths return 404."""

    @pytest.mark.asyncio
    async def test_unknown_endpoint(self, handler):
        req = MockRequest(
            path="/api/v1/ecommerce/unknown-endpoint",
            method="GET",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_on_connect(self, handler):
        req = MockRequest(
            path="/api/v1/ecommerce/connect",
            method="GET",
        )
        result = await handler.handle_request(req)
        # /connect only accepts POST, GET should fall through to 404
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_on_ship(self, handler):
        req = MockRequest(
            path="/api/v1/ecommerce/ship",
            method="GET",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# Path Parameter Extraction
# ---------------------------------------------------------------------------


class TestPathParameterExtraction:
    """Test that path parameters are correctly extracted from URLs."""

    @pytest.mark.asyncio
    async def test_platform_parsed_from_path(self, handler, connected_shopify, mock_connector):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        mock_connector.get_orders = AsyncMock(return_value=[])
        _platform_connectors["shopify"] = mock_connector

        req = MockRequest(
            path="/api/v1/ecommerce/shopify/orders",
            method="GET",
            query={},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 200
        assert _body(result)["platform"] == "shopify"

    @pytest.mark.asyncio
    async def test_order_id_parsed_from_path(self, handler, connected_shopify, mock_connector):
        from aragora.server.handlers.features.ecommerce.models import _platform_connectors

        mock_order = MagicMock()
        mock_order.id = 42
        mock_order.order_number = "1001"
        mock_order.fulfillment_status = None
        mock_order.financial_status = "paid"
        mock_order.email = "test@test.com"
        mock_order.customer = None
        mock_order.total_price = "10.00"
        mock_order.subtotal_price = "10.00"
        mock_order.total_tax = "0.00"
        mock_order.currency = "USD"
        mock_order.line_items = []
        mock_order.created_at = None
        mock_order.updated_at = None
        mock_connector.get_order = AsyncMock(return_value=mock_order)
        _platform_connectors["shopify"] = mock_connector

        req = MockRequest(
            path="/api/v1/ecommerce/shopify/orders/42",
            method="GET",
            query={},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        assert _status(result) == 200
        # Verify the connector was called with the correct parsed order ID
        mock_connector.get_order.assert_awaited_once_with(42)

    @pytest.mark.asyncio
    async def test_non_platform_segment_not_parsed(self, handler):
        """When the first segment is not a supported platform, platform should be None."""
        req = MockRequest(
            path="/api/v1/ecommerce/orders",
            method="GET",
            query={},
        )
        with patch.object(handler, "_circuit_breaker") as cb:
            cb.can_proceed.return_value = True
            result = await handler.handle_request(req)
        # This should hit the cross-platform orders endpoint
        assert _status(result) == 200
        assert "platforms_queried" in _body(result)


# ---------------------------------------------------------------------------
# _with_circuit_breaker Helper
# ---------------------------------------------------------------------------


class TestCircuitBreakerHelper:
    """Test the _with_circuit_breaker wrapper logic."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_rejects(self, handler):
        handler._circuit_breaker = MagicMock()
        handler._circuit_breaker.can_proceed.return_value = False

        async def dummy():
            return {"status_code": 200, "body": {}}

        result = await handler._with_circuit_breaker(dummy)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_success(self, handler):
        handler._circuit_breaker = MagicMock()
        handler._circuit_breaker.can_proceed.return_value = True

        async def dummy():
            return {"status_code": 200, "body": {}}

        result = await handler._with_circuit_breaker(dummy)
        handler._circuit_breaker.record_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_failure_on_connection_error(self, handler):
        handler._circuit_breaker = MagicMock()
        handler._circuit_breaker.can_proceed.return_value = True

        async def fail():
            raise ConnectionError("down")

        result = await handler._with_circuit_breaker(fail)
        handler._circuit_breaker.record_failure.assert_called_once()
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_failure_on_timeout(self, handler):
        handler._circuit_breaker = MagicMock()
        handler._circuit_breaker.can_proceed.return_value = True

        async def fail():
            raise TimeoutError("timed out")

        result = await handler._with_circuit_breaker(fail)
        handler._circuit_breaker.record_failure.assert_called_once()

    @pytest.mark.asyncio
    async def test_circuit_breaker_does_not_count_value_errors(self, handler):
        handler._circuit_breaker = MagicMock()
        handler._circuit_breaker.can_proceed.return_value = True

        async def fail():
            raise ValueError("bad input")

        with pytest.raises(ValueError):
            await handler._with_circuit_breaker(fail)
        handler._circuit_breaker.record_failure.assert_not_called()


# ---------------------------------------------------------------------------
# Sanitization Helpers
# ---------------------------------------------------------------------------


class TestSanitizationHelpers:
    """Test financial and currency sanitization methods."""

    def test_sanitize_financial_amount_normal(self, handler):
        assert handler._sanitize_financial_amount("99.99") == 99.99

    def test_sanitize_financial_amount_negative_clamped(self, handler):
        assert handler._sanitize_financial_amount("-10.00") == 0.0

    def test_sanitize_financial_amount_large_capped(self, handler):
        assert handler._sanitize_financial_amount("999999999.99") == 99_999_999.99

    def test_sanitize_financial_amount_invalid(self, handler):
        assert handler._sanitize_financial_amount("not-a-number") == 0.0

    def test_sanitize_financial_amount_none(self, handler):
        assert handler._sanitize_financial_amount(None) == 0.0

    def test_sanitize_financial_amount_zero(self, handler):
        assert handler._sanitize_financial_amount(0) == 0.0

    def test_sanitize_currency_code_valid(self, handler):
        assert handler._sanitize_currency_code("USD") == "USD"

    def test_sanitize_currency_code_lowercase(self, handler):
        assert handler._sanitize_currency_code("eur") == "EUR"

    def test_sanitize_currency_code_unknown(self, handler):
        assert handler._sanitize_currency_code("XYZ") == "USD"

    def test_sanitize_currency_code_none(self, handler):
        assert handler._sanitize_currency_code(None) == "USD"

    def test_sanitize_currency_code_empty(self, handler):
        assert handler._sanitize_currency_code("") == "USD"

    def test_sanitize_currency_code_not_string(self, handler):
        assert handler._sanitize_currency_code(123) == "USD"

    def test_sanitize_currency_code_with_spaces(self, handler):
        assert handler._sanitize_currency_code(" GBP ") == "GBP"


# ---------------------------------------------------------------------------
# Normalization Methods
# ---------------------------------------------------------------------------


class TestNormalizeMethods:
    """Test order and product normalization across platforms."""

    def _make_shopify_order(self, **overrides):
        order = MagicMock()
        order.id = overrides.get("id", 1)
        order.order_number = overrides.get("order_number", "1001")
        order.fulfillment_status = overrides.get("fulfillment_status")
        order.financial_status = overrides.get("financial_status", "paid")
        order.email = overrides.get("email", "a@b.com")
        order.customer = overrides.get("customer")
        order.total_price = overrides.get("total_price", "50.00")
        order.subtotal_price = overrides.get("subtotal_price", "45.00")
        order.total_tax = overrides.get("total_tax", "5.00")
        order.currency = overrides.get("currency", "USD")
        order.line_items = overrides.get("line_items", [])
        order.created_at = overrides.get("created_at", datetime.now(timezone.utc))
        order.updated_at = overrides.get("updated_at", datetime.now(timezone.utc))
        if "total_shipping_price_set" not in overrides:
            tss = MagicMock()
            tss.shop_money.amount = "3.00"
            order.total_shipping_price_set = tss
        return order

    def test_normalize_shopify_order(self, handler):
        order = self._make_shopify_order()
        result = handler._normalize_shopify_order(order)
        assert result["platform"] == "shopify"
        assert result["id"] == "1"
        assert result["status"] == "unfulfilled"  # None -> "unfulfilled"

    def test_normalize_shopify_order_with_customer(self, handler):
        customer = MagicMock(first_name="Jane", last_name="Doe")
        order = self._make_shopify_order(customer=customer)
        result = handler._normalize_shopify_order(order)
        assert result["customer_name"] == "Jane Doe"

    def test_normalize_shopify_order_no_customer(self, handler):
        order = self._make_shopify_order(customer=None)
        result = handler._normalize_shopify_order(order)
        assert result["customer_name"] is None

    def test_normalize_shipstation_order(self, handler):
        order = MagicMock()
        order.order_id = 100
        order.order_number = "SS-100"
        order.order_status = "shipped"
        order.payment_status = "paid"
        order.customer_email = "c@d.com"
        order.ship_to = MagicMock(name="John Smith")
        order.order_total = "75.00"
        order.shipping_amount = "8.00"
        order.tax_amount = "6.00"
        order.order_date = datetime.now(timezone.utc)
        result = handler._normalize_shipstation_order(order)
        assert result["platform"] == "shipstation"
        assert result["id"] == "100"
        assert result["currency"] == "USD"

    def test_normalize_walmart_order(self, handler):
        order = MagicMock()
        order.purchase_order_id = "WM-001"
        order.customer_order_id = "CO-001"
        order.order_status = MagicMock(value="Created")
        order.customer_email = "w@e.com"
        order.shipping_info = None
        order.order_total = "120.00"
        order.order_date = datetime.now(timezone.utc)
        result = handler._normalize_walmart_order(order)
        assert result["platform"] == "walmart"
        assert result["status"] == "Created"
        assert result["customer_name"] is None

    def test_normalize_shopify_product(self, handler):
        product = MagicMock()
        product.id = 42
        product.title = "Widget"
        product.status = "active"
        product.vendor = "WidgetCo"
        product.product_type = "Hardware"
        product.tags = "sale, featured"
        product.created_at = datetime.now(timezone.utc)
        variant = MagicMock()
        variant.sku = "W-001"
        variant.barcode = "123456"
        variant.price = "29.99"
        variant.compare_at_price = "39.99"
        variant.inventory_quantity = 100
        product.variants = [variant]
        product.images = []
        result = handler._normalize_shopify_product(product)
        assert result["platform"] == "shopify"
        assert result["title"] == "Widget"
        assert result["tags"] == ["sale", "featured"]

    def test_normalize_shopify_product_no_variant(self, handler):
        product = MagicMock()
        product.id = 43
        product.title = "Gadget"
        product.status = "draft"
        product.vendor = None
        product.product_type = None
        product.tags = ""
        product.created_at = None
        product.variants = []
        product.images = []
        result = handler._normalize_shopify_product(product)
        assert result["sku"] is None
        assert result["price"] == 0
        assert result["tags"] == []

    def test_normalize_walmart_item(self, handler):
        item = MagicMock()
        item.sku = "WM-SKU-001"
        item.product_name = "Mega Widget"
        item.gtin = "0012345678"
        item.price = MagicMock(amount="15.50")
        item.quantity = 200
        item.lifecycle_status = MagicMock(value="ACTIVE")
        item.brand = "BrandX"
        item.product_type = "Electronics"
        result = handler._normalize_walmart_item(item)
        assert result["platform"] == "walmart"
        assert result["id"] == "WM-SKU-001"
        assert result["tags"] == []
        assert result["images"] == []


# ---------------------------------------------------------------------------
# Required Credentials
# ---------------------------------------------------------------------------


class TestRequiredCredentials:
    """Test _get_required_credentials helper."""

    def test_shopify_credentials(self, handler):
        creds = handler._get_required_credentials("shopify")
        assert "shop_url" in creds
        assert "access_token" in creds

    def test_shipstation_credentials(self, handler):
        creds = handler._get_required_credentials("shipstation")
        assert "api_key" in creds
        assert "api_secret" in creds

    def test_walmart_credentials(self, handler):
        creds = handler._get_required_credentials("walmart")
        assert "client_id" in creds
        assert "client_secret" in creds

    def test_unknown_platform_empty(self, handler):
        creds = handler._get_required_credentials("unknown")
        assert creds == []


# ---------------------------------------------------------------------------
# Handler Initialization
# ---------------------------------------------------------------------------


class TestHandlerInit:
    """Test handler construction."""

    def test_init_with_none_context(self):
        from aragora.server.handlers.features.ecommerce.handler import EcommerceHandler

        h = EcommerceHandler(ctx=None)
        assert h.ctx == {}

    def test_init_with_server_context(self):
        from aragora.server.handlers.features.ecommerce.handler import EcommerceHandler

        h = EcommerceHandler(server_context={"key": "val"})
        assert h.ctx == {"key": "val"}

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "ecommerce"
