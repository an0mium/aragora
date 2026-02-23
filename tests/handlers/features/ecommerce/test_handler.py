"""Tests for the e-commerce platform API handler.

Covers all routes and behavior of the EcommerceHandler class:
- can_handle() routing for all defined paths
- GET /api/v1/ecommerce/circuit-breaker - circuit breaker status
- GET /api/v1/ecommerce/platforms - list platforms
- POST /api/v1/ecommerce/connect - connect platform
- DELETE /api/v1/ecommerce/{platform} - disconnect platform
- GET /api/v1/ecommerce/orders - list all orders
- GET /api/v1/ecommerce/{platform}/orders - list platform orders
- GET /api/v1/ecommerce/{platform}/orders/{id} - get order details
- GET /api/v1/ecommerce/products - list all products
- GET /api/v1/ecommerce/{platform}/products - list platform products
- GET /api/v1/ecommerce/{platform}/products/{id} - get product details
- GET /api/v1/ecommerce/inventory - get inventory levels
- POST /api/v1/ecommerce/sync-inventory - sync inventory
- GET /api/v1/ecommerce/fulfillment - fulfillment status
- POST /api/v1/ecommerce/ship - create shipment
- GET /api/v1/ecommerce/metrics - e-commerce metrics
- Circuit breaker protection
- Input validation (platform IDs, resource IDs, pagination, SKU, financial amounts)
- Error handling (connection errors, unsupported platforms, not connected)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.ecommerce.handler import EcommerceHandler
from aragora.server.handlers.features.ecommerce.models import (
    SUPPORTED_PLATFORMS,
    _platform_credentials,
    _platform_connectors,
)
from aragora.server.handlers.features.ecommerce.circuit_breaker import (
    reset_ecommerce_circuit_breaker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract body dict from a handler result."""
    if isinstance(result, dict):
        return result.get("body", result)
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a handler result."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


@dataclass
class MockRequest:
    """Mock async HTTP request for EcommerceHandler."""

    method: str = "GET"
    path: str = "/"
    query: dict[str, str] = field(default_factory=dict)
    _body: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)

    async def json(self) -> dict[str, Any]:
        return self._body or {}

    async def body(self) -> bytes:
        return json.dumps(self._body or {}).encode()

    async def read(self) -> bytes:
        return json.dumps(self._body or {}).encode()


@dataclass
class MockShopifyCustomer:
    first_name: str = "John"
    last_name: str = "Doe"


@dataclass
class MockLineItem:
    title: str = "Widget"
    quantity: int = 2
    price: str = "19.99"
    sku: str = "WDG-001"


@dataclass
class MockShippingMoney:
    amount: str = "5.99"


@dataclass
class MockShippingPriceSet:
    shop_money: MockShippingMoney = field(default_factory=MockShippingMoney)


@dataclass
class MockShopifyOrder:
    id: int = 1001
    order_number: int = 1042
    fulfillment_status: str | None = "unfulfilled"
    financial_status: str = "paid"
    email: str = "john@example.com"
    customer: MockShopifyCustomer = field(default_factory=MockShopifyCustomer)
    total_price: str = "45.97"
    subtotal_price: str = "39.98"
    total_shipping_price_set: MockShippingPriceSet = field(default_factory=MockShippingPriceSet)
    total_tax: str = "3.00"
    currency: str = "USD"
    line_items: list[MockLineItem] = field(default_factory=lambda: [MockLineItem()])
    created_at: datetime = field(
        default_factory=lambda: datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime(2026, 2, 2, 12, 0, 0, tzinfo=timezone.utc)
    )


@dataclass
class MockShopifyFulfillment:
    id: int = 5001
    status: str = "success"
    tracking_number: str = "1Z999AA10123456784"
    tracking_url: str = "https://track.example.com/1Z999AA10123456784"
    tracking_company: str = "UPS"


@dataclass
class MockShopifyOrderWithFulfillments(MockShopifyOrder):
    fulfillments: list[MockShopifyFulfillment] = field(
        default_factory=lambda: [MockShopifyFulfillment()]
    )


@dataclass
class MockShopifyVariant:
    sku: str = "WDG-001"
    barcode: str = "012345678901"
    price: str = "19.99"
    compare_at_price: str = "24.99"
    inventory_quantity: int = 50


@dataclass
class MockShopifyProduct:
    id: int = 2001
    title: str = "Widget Pro"
    variants: list[MockShopifyVariant] = field(default_factory=lambda: [MockShopifyVariant()])
    status: str = "active"
    vendor: str = "Acme Corp"
    product_type: str = "Gadgets"
    tags: str = "sale, featured"
    images: list = field(default_factory=list)
    created_at: datetime = field(
        default_factory=lambda: datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    )


@dataclass
class MockShipStationShipTo:
    name: str = "Jane Smith"


@dataclass
class MockShipStationOrder:
    order_id: int = 3001
    order_number: str = "SS-3001"
    order_status: str = "awaiting_shipment"
    payment_status: str = "paid"
    customer_email: str = "jane@example.com"
    ship_to: MockShipStationShipTo = field(default_factory=MockShipStationShipTo)
    order_total: str = "89.99"
    shipping_amount: str = "9.99"
    tax_amount: str = "7.20"
    order_date: datetime = field(
        default_factory=lambda: datetime(2026, 2, 3, 10, 0, 0, tzinfo=timezone.utc)
    )


@dataclass
class MockShipStationShipment:
    order_id: int = 3001
    shipment_id: int = 4001
    shipment_status: str = "shipped"
    tracking_number: str = "9400111899223456789012"
    carrier_code: str = "usps"


@dataclass
class MockShipStationLabel:
    shipment_id: int = 4001
    tracking_number: str = "9400111899223456789012"
    label_data: str = "https://labels.example.com/4001.pdf"


class MockWalmartOrderStatus:
    value = "Created"


@dataclass
class MockWalmartPostalAddress:
    name: str = "Bob Wilson"


@dataclass
class MockWalmartShippingInfo:
    postal_address: MockWalmartPostalAddress = field(default_factory=MockWalmartPostalAddress)


@dataclass
class MockWalmartOrder:
    purchase_order_id: str = "WM-5001"
    customer_order_id: str = "WMCO-5001"
    order_status: MockWalmartOrderStatus = field(default_factory=MockWalmartOrderStatus)
    customer_email: str = "bob@example.com"
    shipping_info: MockWalmartShippingInfo = field(default_factory=MockWalmartShippingInfo)
    order_total: str = "149.99"
    order_date: datetime = field(
        default_factory=lambda: datetime(2026, 2, 4, 14, 0, 0, tzinfo=timezone.utc)
    )


class MockWalmartItemPrice:
    amount = "29.99"


class MockWalmartLifecycleStatus:
    value = "ACTIVE"


@dataclass
class MockWalmartItem:
    sku: str = "WM-SKU-001"
    product_name: str = "Walmart Widget"
    price: MockWalmartItemPrice = field(default_factory=MockWalmartItemPrice)
    gtin: str = "098765432109"
    quantity: int = 100
    lifecycle_status: MockWalmartLifecycleStatus = field(default_factory=MockWalmartLifecycleStatus)
    brand: str = "Walmart Brand"
    product_type: str = "Electronics"


@dataclass
class MockInventoryLevel:
    inventory_item_id: int = 6001
    location_id: int = 7001
    available: int = 25


@dataclass
class MockWalmartInventory:
    sku: str = "WM-SKU-001"
    quantity: int = 100


@dataclass
class MockShopifyFulfillmentCreated:
    id: int = 9001
    status: str = "success"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_platform_state():
    """Reset platform credentials and connectors before each test."""
    _platform_credentials.clear()
    _platform_connectors.clear()
    reset_ecommerce_circuit_breaker()
    yield
    _platform_credentials.clear()
    _platform_connectors.clear()


@pytest.fixture
def handler():
    """Create an EcommerceHandler instance."""
    return EcommerceHandler({})


@pytest.fixture
def mock_shopify_connector():
    """Create a mock Shopify connector."""
    c = AsyncMock()
    c.get_orders = AsyncMock(return_value=[MockShopifyOrder()])
    c.get_order = AsyncMock(return_value=MockShopifyOrder())
    c.get_products = AsyncMock(return_value=[MockShopifyProduct()])
    c.get_product = AsyncMock(return_value=MockShopifyProduct())
    c.get_inventory_levels = AsyncMock(return_value=[MockInventoryLevel()])
    c.create_fulfillment = AsyncMock(return_value=MockShopifyFulfillmentCreated())
    c.close = AsyncMock()
    return c


@pytest.fixture
def mock_shipstation_connector():
    """Create a mock ShipStation connector."""
    c = AsyncMock()
    c.get_orders = AsyncMock(return_value=[MockShipStationOrder()])
    c.get_order = AsyncMock(return_value=MockShipStationOrder())
    c.get_shipments = AsyncMock(return_value=[MockShipStationShipment()])
    c.create_label = AsyncMock(return_value=MockShipStationLabel())
    c.close = AsyncMock()
    return c


@pytest.fixture
def mock_walmart_connector():
    """Create a mock Walmart connector."""
    c = AsyncMock()
    c.get_orders = AsyncMock(return_value=[MockWalmartOrder()])
    c.get_order = AsyncMock(return_value=MockWalmartOrder())
    c.get_items = AsyncMock(return_value=[MockWalmartItem()])
    c.get_item = AsyncMock(return_value=MockWalmartItem())
    c.get_inventory = AsyncMock(return_value=[MockWalmartInventory()])
    c.update_inventory = AsyncMock()
    c.close = AsyncMock()
    return c


def _connect_shopify(connector=None):
    """Set up Shopify as connected."""
    _platform_credentials["shopify"] = {
        "credentials": {"shop_url": "https://test.myshopify.com", "access_token": "shpat_xxx"},
        "connected_at": "2026-02-01T00:00:00+00:00",
    }
    if connector:
        _platform_connectors["shopify"] = connector


def _connect_shipstation(connector=None):
    """Set up ShipStation as connected."""
    _platform_credentials["shipstation"] = {
        "credentials": {"api_key": "key123", "api_secret": "secret456"},
        "connected_at": "2026-02-01T00:00:00+00:00",
    }
    if connector:
        _platform_connectors["shipstation"] = connector


def _connect_walmart(connector=None):
    """Set up Walmart as connected."""
    _platform_credentials["walmart"] = {
        "credentials": {"client_id": "wm_id", "client_secret": "wm_secret"},
        "connected_at": "2026-02-01T00:00:00+00:00",
    }
    if connector:
        _platform_connectors["walmart"] = connector


# ---------------------------------------------------------------------------
# can_handle Tests
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for can_handle routing."""

    def test_handles_platforms_path(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/platforms") is True

    def test_handles_connect_path(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/connect", "POST") is True

    def test_handles_orders_path(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/orders") is True

    def test_handles_platform_orders(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/shopify/orders") is True

    def test_handles_platform_order_by_id(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/shopify/orders/123") is True

    def test_handles_products_path(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/products") is True

    def test_handles_inventory_path(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/inventory") is True

    def test_handles_sync_inventory(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/sync-inventory", "POST") is True

    def test_handles_fulfillment(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/fulfillment") is True

    def test_handles_ship(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/ship", "POST") is True

    def test_handles_metrics(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/metrics") is True

    def test_handles_circuit_breaker(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/circuit-breaker") is True

    def test_handles_platform_disconnect(self, handler):
        assert handler.can_handle("/api/v1/ecommerce/shopify", "DELETE") is True

    def test_does_not_handle_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_does_not_handle_health(self, handler):
        assert handler.can_handle("/api/v1/health") is False


# ---------------------------------------------------------------------------
# ROUTES Tests
# ---------------------------------------------------------------------------


class TestRoutesDefinition:
    """Tests for the ROUTES class attribute."""

    def test_routes_count(self, handler):
        assert len(handler.ROUTES) == 15

    def test_routes_all_start_with_ecommerce(self, handler):
        for route in handler.ROUTES:
            assert "/ecommerce/" in route


# ---------------------------------------------------------------------------
# Circuit Breaker Status Tests
# ---------------------------------------------------------------------------


class TestCircuitBreakerStatus:
    """Tests for GET /api/v1/ecommerce/circuit-breaker."""

    @pytest.mark.asyncio
    async def test_get_circuit_breaker_status(self, handler):
        request = MockRequest(method="GET", path="/api/v1/ecommerce/circuit-breaker")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert "state" in body

    @pytest.mark.asyncio
    async def test_circuit_breaker_defaults_to_closed(self, handler):
        request = MockRequest(method="GET", path="/api/v1/ecommerce/circuit-breaker")
        result = await handler.handle_request(request)
        body = _body(result)
        assert body["state"] == "closed"


# ---------------------------------------------------------------------------
# List Platforms Tests
# ---------------------------------------------------------------------------


class TestListPlatforms:
    """Tests for GET /api/v1/ecommerce/platforms."""

    @pytest.mark.asyncio
    async def test_list_platforms_returns_all(self, handler):
        request = MockRequest(method="GET", path="/api/v1/ecommerce/platforms")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert len(body["platforms"]) == len(SUPPORTED_PLATFORMS)
        assert body["connected_count"] == 0

    @pytest.mark.asyncio
    async def test_list_platforms_shows_connected(self, handler):
        _connect_shopify()
        request = MockRequest(method="GET", path="/api/v1/ecommerce/platforms")
        result = await handler.handle_request(request)
        body = _body(result)
        assert body["connected_count"] == 1
        shopify = next(p for p in body["platforms"] if p["id"] == "shopify")
        assert shopify["connected"] is True

    @pytest.mark.asyncio
    async def test_list_platforms_includes_metadata(self, handler):
        request = MockRequest(method="GET", path="/api/v1/ecommerce/platforms")
        result = await handler.handle_request(request)
        body = _body(result)
        platform = body["platforms"][0]
        assert "name" in platform
        assert "description" in platform
        assert "features" in platform
        assert "connected" in platform

    @pytest.mark.asyncio
    async def test_list_platforms_not_connected_has_null_timestamp(self, handler):
        request = MockRequest(method="GET", path="/api/v1/ecommerce/platforms")
        result = await handler.handle_request(request)
        body = _body(result)
        for platform in body["platforms"]:
            assert platform["connected_at"] is None


# ---------------------------------------------------------------------------
# Connect Platform Tests
# ---------------------------------------------------------------------------


class TestConnectPlatform:
    """Tests for POST /api/v1/ecommerce/connect."""

    @pytest.mark.asyncio
    async def test_connect_shopify_success(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/connect",
            _body={
                "platform": "shopify",
                "credentials": {
                    "shop_url": "https://test.myshopify.com",
                    "access_token": "shpat_xxx",
                },
            },
        )
        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "shopify"
        assert "connected_at" in body

    @pytest.mark.asyncio
    async def test_connect_shipstation_success(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/connect",
            _body={
                "platform": "shipstation",
                "credentials": {
                    "api_key": "key123",
                    "api_secret": "secret456",
                },
            },
        )
        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "shipstation"

    @pytest.mark.asyncio
    async def test_connect_walmart_success(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/connect",
            _body={
                "platform": "walmart",
                "credentials": {
                    "client_id": "wm_id",
                    "client_secret": "wm_secret",
                },
            },
        )
        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "walmart"

    @pytest.mark.asyncio
    async def test_connect_missing_platform(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/connect",
            _body={"credentials": {"key": "value"}},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_connect_unsupported_platform(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/connect",
            _body={"platform": "etsy", "credentials": {"key": "value"}},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400
        assert "Unsupported" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_connect_missing_credentials(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/connect",
            _body={"platform": "shopify"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400
        assert "Credentials" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_connect_missing_required_credential_fields(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/connect",
            _body={
                "platform": "shopify",
                "credentials": {"shop_url": "https://test.myshopify.com"},
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400
        assert "Missing" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_connect_credential_not_string(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/connect",
            _body={
                "platform": "shipstation",
                "credentials": {"api_key": 12345, "api_secret": "secret"},
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400
        assert "must be a string" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_connect_credential_empty_string(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/connect",
            _body={
                "platform": "shipstation",
                "credentials": {"api_key": "  ", "api_secret": "secret"},
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400
        assert "cannot be empty" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_connect_credential_too_long(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/connect",
            _body={
                "platform": "shipstation",
                "credentials": {
                    "api_key": "x" * 2000,
                    "api_secret": "secret",
                },
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400
        assert "exceeds maximum length" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_connect_credentials_not_object(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/connect",
            _body={"platform": "shopify", "credentials": "not-a-dict"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400
        assert "must be an object" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_connect_shopify_invalid_url(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/connect",
            _body={
                "platform": "shopify",
                "credentials": {
                    "shop_url": "not-a-url",
                    "access_token": "shpat_xxx",
                },
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_connect_invalid_platform_format(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/connect",
            _body={
                "platform": "plat form!",
                "credentials": {"key": "val"},
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_connect_invalid_json_body(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/connect",
            _body=None,
        )
        result = await handler.handle_request(request)
        # Empty body => no platform field
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# Disconnect Platform Tests
# ---------------------------------------------------------------------------


class TestDisconnectPlatform:
    """Tests for DELETE /api/v1/ecommerce/{platform}."""

    @pytest.mark.asyncio
    async def test_disconnect_shopify(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        request = MockRequest(method="DELETE", path="/api/v1/ecommerce/shopify")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "shopify"
        assert "shopify" not in _platform_credentials
        mock_shopify_connector.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self, handler):
        request = MockRequest(method="DELETE", path="/api/v1/ecommerce/shopify")
        result = await handler.handle_request(request)
        assert _status(result) == 404
        assert "not connected" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_disconnect_without_connector_in_cache(self, handler):
        _connect_shopify()  # no connector in _platform_connectors
        request = MockRequest(method="DELETE", path="/api/v1/ecommerce/shopify")
        result = await handler.handle_request(request)
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# List All Orders Tests
# ---------------------------------------------------------------------------


class TestListAllOrders:
    """Tests for GET /api/v1/ecommerce/orders."""

    @pytest.mark.asyncio
    async def test_list_all_orders_no_platforms(self, handler):
        request = MockRequest(method="GET", path="/api/v1/ecommerce/orders")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["orders"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_all_orders_from_shopify(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        request = MockRequest(method="GET", path="/api/v1/ecommerce/orders")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert len(body["orders"]) == 1
        assert body["orders"][0]["platform"] == "shopify"
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_list_all_orders_multiple_platforms(
        self, handler, mock_shopify_connector, mock_walmart_connector
    ):
        _connect_shopify(mock_shopify_connector)
        _connect_walmart(mock_walmart_connector)
        request = MockRequest(method="GET", path="/api/v1/ecommerce/orders")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        platforms = {o["platform"] for o in body["orders"]}
        assert "shopify" in platforms
        assert "walmart" in platforms

    @pytest.mark.asyncio
    async def test_list_all_orders_pagination_params(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        request = MockRequest(
            method="GET",
            path="/api/v1/ecommerce/orders",
            query={"limit": "10", "offset": "0"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["limit"] == 10
        assert body["offset"] == 0

    @pytest.mark.asyncio
    async def test_list_all_orders_invalid_pagination(self, handler):
        _connect_shopify()
        request = MockRequest(
            method="GET",
            path="/api/v1/ecommerce/orders",
            query={"limit": "abc"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_list_all_orders_platform_error_handled(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        mock_shopify_connector.get_orders.side_effect = ConnectionError("timeout")
        request = MockRequest(method="GET", path="/api/v1/ecommerce/orders")
        result = await handler.handle_request(request)
        # Should still succeed but with empty results for that platform
        assert _status(result) == 200
        body = _body(result)
        assert body["orders"] == []


# ---------------------------------------------------------------------------
# List Platform Orders Tests
# ---------------------------------------------------------------------------


class TestListPlatformOrders:
    """Tests for GET /api/v1/ecommerce/{platform}/orders."""

    @pytest.mark.asyncio
    async def test_list_shopify_orders(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        request = MockRequest(method="GET", path="/api/v1/ecommerce/shopify/orders")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "shopify"
        assert len(body["orders"]) == 1
        order = body["orders"][0]
        assert order["id"] == "1001"
        assert order["platform"] == "shopify"
        assert order["customer_name"] == "John Doe"

    @pytest.mark.asyncio
    async def test_list_platform_orders_not_connected(self, handler):
        request = MockRequest(method="GET", path="/api/v1/ecommerce/shopify/orders")
        result = await handler.handle_request(request)
        assert _status(result) == 404
        assert "not connected" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_list_platform_orders_with_status_filter(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        request = MockRequest(
            method="GET",
            path="/api/v1/ecommerce/shopify/orders",
            query={"status": "fulfilled"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 200
        mock_shopify_connector.get_orders.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_shipstation_orders(self, handler, mock_shipstation_connector):
        _connect_shipstation(mock_shipstation_connector)
        request = MockRequest(method="GET", path="/api/v1/ecommerce/shipstation/orders")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "shipstation"
        order = body["orders"][0]
        assert order["platform"] == "shipstation"
        assert order["id"] == "3001"

    @pytest.mark.asyncio
    async def test_list_walmart_orders(self, handler, mock_walmart_connector):
        _connect_walmart(mock_walmart_connector)
        request = MockRequest(method="GET", path="/api/v1/ecommerce/walmart/orders")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "walmart"
        order = body["orders"][0]
        assert order["platform"] == "walmart"
        assert order["id"] == "WM-5001"


# ---------------------------------------------------------------------------
# Get Order Tests
# ---------------------------------------------------------------------------


class TestGetOrder:
    """Tests for GET /api/v1/ecommerce/{platform}/orders/{id}."""

    @pytest.mark.asyncio
    async def test_get_shopify_order(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        request = MockRequest(method="GET", path="/api/v1/ecommerce/shopify/orders/1001")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "1001"
        assert body["platform"] == "shopify"
        assert body["total_price"] == 45.97
        assert body["currency"] == "USD"

    @pytest.mark.asyncio
    async def test_get_shipstation_order(self, handler, mock_shipstation_connector):
        _connect_shipstation(mock_shipstation_connector)
        request = MockRequest(method="GET", path="/api/v1/ecommerce/shipstation/orders/3001")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "3001"
        assert body["platform"] == "shipstation"

    @pytest.mark.asyncio
    async def test_get_walmart_order(self, handler, mock_walmart_connector):
        _connect_walmart(mock_walmart_connector)
        request = MockRequest(method="GET", path="/api/v1/ecommerce/walmart/orders/WM5001")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "WM-5001"

    @pytest.mark.asyncio
    async def test_get_order_not_connected(self, handler):
        request = MockRequest(method="GET", path="/api/v1/ecommerce/shopify/orders/1001")
        result = await handler.handle_request(request)
        assert _status(result) == 404
        assert "not connected" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_get_order_connector_unavailable(self, handler):
        _connect_shopify()
        # No connector in cache, and _get_connector returns None by patching
        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            request = MockRequest(method="GET", path="/api/v1/ecommerce/shopify/orders/1001")
            result = await handler.handle_request(request)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_order_connection_error(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        mock_shopify_connector.get_order.side_effect = ConnectionError("down")
        request = MockRequest(method="GET", path="/api/v1/ecommerce/shopify/orders/1001")
        result = await handler.handle_request(request)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_order_not_found(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        mock_shopify_connector.get_order.side_effect = KeyError("not found")
        request = MockRequest(method="GET", path="/api/v1/ecommerce/shopify/orders/9999")
        result = await handler.handle_request(request)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_order_value_error(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        mock_shopify_connector.get_order.side_effect = ValueError("bad")
        request = MockRequest(method="GET", path="/api/v1/ecommerce/shopify/orders/1001")
        result = await handler.handle_request(request)
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# List All Products Tests
# ---------------------------------------------------------------------------


class TestListAllProducts:
    """Tests for GET /api/v1/ecommerce/products."""

    @pytest.mark.asyncio
    async def test_list_all_products_no_platforms(self, handler):
        request = MockRequest(method="GET", path="/api/v1/ecommerce/products")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["products"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_products_from_shopify(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        request = MockRequest(method="GET", path="/api/v1/ecommerce/products")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert len(body["products"]) == 1
        product = body["products"][0]
        assert product["platform"] == "shopify"
        assert product["title"] == "Widget Pro"

    @pytest.mark.asyncio
    async def test_list_products_pagination(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        request = MockRequest(
            method="GET",
            path="/api/v1/ecommerce/products",
            query={"limit": "5", "offset": "0"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["limit"] == 5

    @pytest.mark.asyncio
    async def test_list_products_invalid_pagination(self, handler):
        request = MockRequest(
            method="GET",
            path="/api/v1/ecommerce/products",
            query={"limit": "-1"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_list_products_from_walmart(self, handler, mock_walmart_connector):
        _connect_walmart(mock_walmart_connector)
        request = MockRequest(method="GET", path="/api/v1/ecommerce/products")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert len(body["products"]) == 1
        product = body["products"][0]
        assert product["platform"] == "walmart"

    @pytest.mark.asyncio
    async def test_list_products_platform_error_handled(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        mock_shopify_connector.get_products.side_effect = ConnectionError("down")
        request = MockRequest(method="GET", path="/api/v1/ecommerce/products")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["products"] == []


# ---------------------------------------------------------------------------
# List Platform Products Tests
# ---------------------------------------------------------------------------


class TestListPlatformProducts:
    """Tests for GET /api/v1/ecommerce/{platform}/products."""

    @pytest.mark.asyncio
    async def test_list_shopify_products(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        request = MockRequest(method="GET", path="/api/v1/ecommerce/shopify/products")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "shopify"
        assert len(body["products"]) == 1

    @pytest.mark.asyncio
    async def test_list_platform_products_not_connected(self, handler):
        request = MockRequest(method="GET", path="/api/v1/ecommerce/shopify/products")
        result = await handler.handle_request(request)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_list_walmart_products(self, handler, mock_walmart_connector):
        _connect_walmart(mock_walmart_connector)
        request = MockRequest(method="GET", path="/api/v1/ecommerce/walmart/products")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "walmart"


# ---------------------------------------------------------------------------
# Get Product Tests
# ---------------------------------------------------------------------------


class TestGetProduct:
    """Tests for GET /api/v1/ecommerce/{platform}/products/{id}."""

    @pytest.mark.asyncio
    async def test_get_shopify_product(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        request = MockRequest(method="GET", path="/api/v1/ecommerce/shopify/products/2001")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "2001"
        assert body["platform"] == "shopify"
        assert body["sku"] == "WDG-001"
        assert body["price"] == 19.99

    @pytest.mark.asyncio
    async def test_get_walmart_product(self, handler, mock_walmart_connector):
        _connect_walmart(mock_walmart_connector)
        request = MockRequest(method="GET", path="/api/v1/ecommerce/walmart/products/WM-SKU-001")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "walmart"
        assert body["sku"] == "WM-SKU-001"

    @pytest.mark.asyncio
    async def test_get_product_not_connected(self, handler):
        request = MockRequest(method="GET", path="/api/v1/ecommerce/shopify/products/2001")
        result = await handler.handle_request(request)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_product_connector_unavailable(self, handler):
        _connect_shopify()
        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            request = MockRequest(method="GET", path="/api/v1/ecommerce/shopify/products/2001")
            result = await handler.handle_request(request)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_product_connection_error(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        mock_shopify_connector.get_product.side_effect = ConnectionError("down")
        request = MockRequest(method="GET", path="/api/v1/ecommerce/shopify/products/2001")
        result = await handler.handle_request(request)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_product_not_found(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        mock_shopify_connector.get_product.side_effect = KeyError("nope")
        request = MockRequest(method="GET", path="/api/v1/ecommerce/shopify/products/9999")
        result = await handler.handle_request(request)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_product_invalid_id(self, handler):
        _connect_shopify()
        request = MockRequest(
            method="GET", path="/api/v1/ecommerce/shopify/products/../../etc/passwd"
        )
        result = await handler.handle_request(request)
        # The path traversal chars should fail resource ID validation
        assert _status(result) in (400, 404)


# ---------------------------------------------------------------------------
# Inventory Tests
# ---------------------------------------------------------------------------


class TestGetInventory:
    """Tests for GET /api/v1/ecommerce/inventory."""

    @pytest.mark.asyncio
    async def test_get_inventory_no_platforms(self, handler):
        request = MockRequest(method="GET", path="/api/v1/ecommerce/inventory")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["inventory"] == {}

    @pytest.mark.asyncio
    async def test_get_inventory_shopify(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        request = MockRequest(method="GET", path="/api/v1/ecommerce/inventory")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert "shopify" in body["inventory"]
        level = body["inventory"]["shopify"][0]
        assert level["available"] == 25

    @pytest.mark.asyncio
    async def test_get_inventory_walmart(self, handler, mock_walmart_connector):
        _connect_walmart(mock_walmart_connector)
        request = MockRequest(
            method="GET",
            path="/api/v1/ecommerce/inventory",
            query={"sku": "WM-SKU-001"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert "walmart" in body["inventory"]

    @pytest.mark.asyncio
    async def test_get_inventory_invalid_sku(self, handler):
        request = MockRequest(
            method="GET",
            path="/api/v1/ecommerce/inventory",
            query={"sku": "!!!invalid!!!"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_inventory_platform_error(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        mock_shopify_connector.get_inventory_levels.side_effect = ConnectionError("down")
        request = MockRequest(method="GET", path="/api/v1/ecommerce/inventory")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["inventory"]["shopify"][0]["error"] == "Failed to fetch inventory"


# ---------------------------------------------------------------------------
# Sync Inventory Tests
# ---------------------------------------------------------------------------


class TestSyncInventory:
    """Tests for POST /api/v1/ecommerce/sync-inventory."""

    @pytest.mark.asyncio
    async def test_sync_inventory_with_quantity(self, handler, mock_walmart_connector):
        _connect_walmart(mock_walmart_connector)
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/sync-inventory",
            _body={
                "sku": "WM-SKU-001",
                "quantity": 50,
                "target_platforms": ["walmart"],
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["sku"] == "WM-SKU-001"
        assert body["quantity"] == 50

    @pytest.mark.asyncio
    async def test_sync_inventory_missing_sku(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/sync-inventory",
            _body={"quantity": 10},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400
        assert "SKU" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_sync_inventory_invalid_quantity(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/sync-inventory",
            _body={"sku": "SKU001", "quantity": -5},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_sync_inventory_no_quantity_no_source(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/sync-inventory",
            _body={"sku": "SKU001"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400
        assert (
            "quantity" in _body(result)["error"].lower()
            or "source" in _body(result)["error"].lower()
        )

    @pytest.mark.asyncio
    async def test_sync_inventory_from_source_platform(self, handler, mock_walmart_connector):
        _connect_walmart(mock_walmart_connector)
        # Source platform returns quantity
        mock_walmart_connector.get_inventory.return_value = [MockWalmartInventory(quantity=75)]
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/sync-inventory",
            _body={
                "sku": "WM-SKU-001",
                "source_platform": "walmart",
                "target_platforms": ["walmart"],
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["quantity"] == 75

    @pytest.mark.asyncio
    async def test_sync_inventory_target_not_connected(self, handler, mock_walmart_connector):
        _connect_walmart(mock_walmart_connector)
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/sync-inventory",
            _body={
                "sku": "SKU001",
                "quantity": 10,
                "target_platforms": ["shopify"],
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["results"]["shopify"]["error"] == "Not connected"

    @pytest.mark.asyncio
    async def test_sync_inventory_target_platforms_not_list(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/sync-inventory",
            _body={
                "sku": "SKU001",
                "quantity": 10,
                "target_platforms": "walmart",
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400
        assert "list" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_sync_inventory_too_many_targets(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/sync-inventory",
            _body={
                "sku": "SKU001",
                "quantity": 10,
                "target_platforms": [f"platform{i}" for i in range(51)],
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400
        assert "max 50" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_sync_inventory_invalid_json_body(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/sync-inventory",
            _body=None,
        )
        result = await handler.handle_request(request)
        # Empty body => missing sku
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_sync_inventory_skips_source_platform_in_targets(
        self, handler, mock_walmart_connector
    ):
        _connect_walmart(mock_walmart_connector)
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/sync-inventory",
            _body={
                "sku": "WM-SKU-001",
                "quantity": 10,
                "source_platform": "walmart",
                "target_platforms": ["walmart"],
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        # walmart is source, so it should be skipped in results
        assert "walmart" not in body["results"]

    @pytest.mark.asyncio
    async def test_sync_inventory_connector_error(self, handler, mock_walmart_connector):
        _connect_walmart(mock_walmart_connector)
        mock_walmart_connector.update_inventory.side_effect = ConnectionError("fail")
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/sync-inventory",
            _body={
                "sku": "WM-SKU-001",
                "quantity": 10,
                "target_platforms": ["walmart"],
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert "error" in body["results"]["walmart"]

    @pytest.mark.asyncio
    async def test_sync_inventory_invalid_sku_format(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/sync-inventory",
            _body={"sku": "!!!", "quantity": 10},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# Fulfillment Tests
# ---------------------------------------------------------------------------


class TestGetFulfillmentStatus:
    """Tests for GET /api/v1/ecommerce/fulfillment."""

    @pytest.mark.asyncio
    async def test_fulfillment_no_platforms(self, handler):
        request = MockRequest(method="GET", path="/api/v1/ecommerce/fulfillment")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["fulfillments"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_fulfillment_shopify(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        mock_shopify_connector.get_order.return_value = MockShopifyOrderWithFulfillments()
        request = MockRequest(
            method="GET",
            path="/api/v1/ecommerce/fulfillment",
            query={"order_id": "1001", "platform": "shopify"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert len(body["fulfillments"]) == 1
        f = body["fulfillments"][0]
        assert f["platform"] == "shopify"
        assert f["tracking_number"] == "1Z999AA10123456784"

    @pytest.mark.asyncio
    async def test_fulfillment_shipstation(self, handler, mock_shipstation_connector):
        _connect_shipstation(mock_shipstation_connector)
        request = MockRequest(
            method="GET",
            path="/api/v1/ecommerce/fulfillment",
            query={"platform": "shipstation"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert len(body["fulfillments"]) == 1
        f = body["fulfillments"][0]
        assert f["platform"] == "shipstation"
        assert f["carrier"] == "usps"

    @pytest.mark.asyncio
    async def test_fulfillment_invalid_order_id(self, handler):
        request = MockRequest(
            method="GET",
            path="/api/v1/ecommerce/fulfillment",
            query={"order_id": "!!!invalid!!!"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_fulfillment_invalid_platform_filter(self, handler):
        request = MockRequest(
            method="GET",
            path="/api/v1/ecommerce/fulfillment",
            query={"platform": "inv@lid"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_fulfillment_connector_error(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        mock_shopify_connector.get_order.side_effect = ConnectionError("fail")
        request = MockRequest(
            method="GET",
            path="/api/v1/ecommerce/fulfillment",
            query={"order_id": "1001", "platform": "shopify"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["fulfillments"] == []


# ---------------------------------------------------------------------------
# Create Shipment Tests
# ---------------------------------------------------------------------------


class TestCreateShipment:
    """Tests for POST /api/v1/ecommerce/ship."""

    @pytest.mark.asyncio
    async def test_create_shipstation_shipment(self, handler, mock_shipstation_connector):
        _connect_shipstation(mock_shipstation_connector)
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/ship",
            _body={
                "platform": "shipstation",
                "order_id": "3001",
                "carrier": "usps",
                "service": "usps_priority",
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 201
        body = _body(result)
        assert body["platform"] == "shipstation"
        assert body["tracking_number"] == "9400111899223456789012"

    @pytest.mark.asyncio
    async def test_create_shopify_shipment(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/ship",
            _body={
                "platform": "shopify",
                "order_id": "1001",
                "carrier": "ups",
                "tracking_number": "1Z999AA10123456784",
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 201
        body = _body(result)
        assert body["platform"] == "shopify"
        assert body["fulfillment_id"] == "9001"

    @pytest.mark.asyncio
    async def test_create_shipment_missing_order_id(self, handler):
        _connect_shipstation()
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/ship",
            _body={"platform": "shipstation", "carrier": "usps"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400
        assert "order_id" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_shipment_platform_not_connected(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/ship",
            _body={"platform": "shipstation", "order_id": "3001"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 404
        assert "not connected" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_shipment_connector_unavailable(self, handler):
        _connect_shipstation()
        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            request = MockRequest(
                method="POST",
                path="/api/v1/ecommerce/ship",
                _body={"platform": "shipstation", "order_id": "3001"},
            )
            result = await handler.handle_request(request)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_create_shipment_connection_error(self, handler, mock_shipstation_connector):
        _connect_shipstation(mock_shipstation_connector)
        mock_shipstation_connector.create_label.side_effect = ConnectionError("fail")
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/ship",
            _body={"platform": "shipstation", "order_id": "3001"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_create_shipment_invalid_carrier(self, handler):
        _connect_shipstation()
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/ship",
            _body={
                "platform": "shipstation",
                "order_id": "3001",
                "carrier": "inv@lid carrier!",
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_shipment_invalid_service(self, handler):
        _connect_shipstation()
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/ship",
            _body={
                "platform": "shipstation",
                "order_id": "3001",
                "service": "inv@lid!",
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_shipment_tracking_number_too_long(self, handler):
        _connect_shipstation()
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/ship",
            _body={
                "platform": "shipstation",
                "order_id": "3001",
                "tracking_number": "T" * 200,
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_shipment_invalid_declared_value(self, handler):
        _connect_shipstation()
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/ship",
            _body={
                "platform": "shipstation",
                "order_id": "3001",
                "declared_value": -10,
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_shipment_invalid_currency(self, handler):
        _connect_shipstation()
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/ship",
            _body={
                "platform": "shipstation",
                "order_id": "3001",
                "currency": "INVALID",
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_shipment_default_platform_is_shipstation(self, handler):
        _connect_shipstation()
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/ship",
            _body={"order_id": "3001"},
        )
        # No explicit platform, defaults to shipstation
        # shipstation connected but no connector available
        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            result = await handler.handle_request(request)
        assert _status(result) == 500  # connector unavailable

    @pytest.mark.asyncio
    async def test_create_shipment_invalid_json_body(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/ship",
            _body=None,
        )
        result = await handler.handle_request(request)
        # Empty body => missing order_id
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_shipment_application_error(self, handler, mock_shipstation_connector):
        _connect_shipstation(mock_shipstation_connector)
        mock_shipstation_connector.create_label.side_effect = ValueError("bad input")
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/ship",
            _body={"platform": "shipstation", "order_id": "3001"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------


class TestMetrics:
    """Tests for GET /api/v1/ecommerce/metrics."""

    @pytest.mark.asyncio
    async def test_metrics_no_platforms(self, handler):
        request = MockRequest(method="GET", path="/api/v1/ecommerce/metrics")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["totals"]["total_orders"] == 0
        assert body["totals"]["total_revenue"] == 0
        assert body["totals"]["average_order_value"] == 0

    @pytest.mark.asyncio
    async def test_metrics_with_shopify(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        request = MockRequest(method="GET", path="/api/v1/ecommerce/metrics")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["totals"]["total_orders"] == 1
        assert body["totals"]["total_revenue"] == 45.97
        assert body["period_days"] == 30

    @pytest.mark.asyncio
    async def test_metrics_with_custom_days(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        request = MockRequest(
            method="GET",
            path="/api/v1/ecommerce/metrics",
            query={"days": "7"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["period_days"] == 7

    @pytest.mark.asyncio
    async def test_metrics_platform_connection_error_returns_zero_metrics(
        self, handler, mock_shopify_connector
    ):
        """When orders fetch fails with ConnectionError, metrics returns zeros (error is swallowed)."""
        _connect_shopify(mock_shopify_connector)
        mock_shopify_connector.get_orders.side_effect = ConnectionError("fail")
        request = MockRequest(method="GET", path="/api/v1/ecommerce/metrics")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        # _fetch_platform_orders catches ConnectionError and returns [],
        # so _fetch_platform_metrics gets zero orders (no outer exception propagates).
        assert body["platforms"]["shopify"]["total_orders"] == 0

    @pytest.mark.asyncio
    async def test_metrics_platform_attribute_error_shows_error(self, handler):
        """When _fetch_platform_metrics itself raises, the error is captured."""
        _connect_shopify()
        with patch.object(
            handler,
            "_fetch_platform_metrics",
            side_effect=AttributeError("missing attr"),
        ):
            request = MockRequest(method="GET", path="/api/v1/ecommerce/metrics")
            result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert "error" in body["platforms"]["shopify"]

    @pytest.mark.asyncio
    async def test_metrics_average_order_value(self, handler, mock_shopify_connector):
        # Two orders with different prices
        orders = [
            MockShopifyOrder(id=1001, total_price="100.00"),
            MockShopifyOrder(id=1002, total_price="200.00"),
        ]
        mock_shopify_connector.get_orders.return_value = orders
        _connect_shopify(mock_shopify_connector)
        request = MockRequest(method="GET", path="/api/v1/ecommerce/metrics")
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["totals"]["total_orders"] == 2
        assert body["totals"]["total_revenue"] == 300.0
        assert body["totals"]["average_order_value"] == 150.0


# ---------------------------------------------------------------------------
# Circuit Breaker Protection Tests
# ---------------------------------------------------------------------------


class TestCircuitBreakerProtection:
    """Tests for circuit breaker behavior on protected endpoints."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_rejects_order_requests(self, handler):
        _connect_shopify()
        # Open the circuit breaker
        cb = handler._circuit_breaker
        for _ in range(10):
            cb.record_failure()
        request = MockRequest(method="GET", path="/api/v1/ecommerce/orders")
        result = await handler.handle_request(request)
        assert _status(result) == 503
        assert "temporarily unavailable" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_rejects_product_requests(self, handler):
        _connect_shopify()
        cb = handler._circuit_breaker
        for _ in range(10):
            cb.record_failure()
        request = MockRequest(method="GET", path="/api/v1/ecommerce/products")
        result = await handler.handle_request(request)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_rejects_inventory_requests(self, handler):
        cb = handler._circuit_breaker
        for _ in range(10):
            cb.record_failure()
        request = MockRequest(method="GET", path="/api/v1/ecommerce/inventory")
        result = await handler.handle_request(request)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_rejects_metrics(self, handler):
        cb = handler._circuit_breaker
        for _ in range(10):
            cb.record_failure()
        request = MockRequest(method="GET", path="/api/v1/ecommerce/metrics")
        result = await handler.handle_request(request)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_success(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        cb = handler._circuit_breaker
        initial_status = cb.get_status()
        request = MockRequest(method="GET", path="/api/v1/ecommerce/orders")
        await handler.handle_request(request)
        # Success should be recorded (no failures increase)
        status = cb.get_status()
        assert status["failure_count"] == initial_status["failure_count"]

    @pytest.mark.asyncio
    async def test_platforms_endpoint_ignores_circuit_breaker(self, handler):
        """Platforms list does not go through circuit breaker."""
        cb = handler._circuit_breaker
        for _ in range(10):
            cb.record_failure()
        request = MockRequest(method="GET", path="/api/v1/ecommerce/platforms")
        result = await handler.handle_request(request)
        # Platforms endpoint should work even with open circuit breaker
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# Not Found Tests
# ---------------------------------------------------------------------------


class TestNotFound:
    """Tests for unmatched routes."""

    @pytest.mark.asyncio
    async def test_unknown_ecommerce_endpoint(self, handler):
        request = MockRequest(method="GET", path="/api/v1/ecommerce/unknown-endpoint")
        result = await handler.handle_request(request)
        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_wrong_method_for_connect(self, handler):
        request = MockRequest(method="GET", path="/api/v1/ecommerce/connect")
        result = await handler.handle_request(request)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_for_ship(self, handler):
        request = MockRequest(method="GET", path="/api/v1/ecommerce/ship")
        result = await handler.handle_request(request)
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# Normalization Helper Tests
# ---------------------------------------------------------------------------


class TestNormalization:
    """Tests for data normalization helpers."""

    def test_sanitize_financial_amount_valid(self, handler):
        assert handler._sanitize_financial_amount("19.99") == 19.99

    def test_sanitize_financial_amount_negative(self, handler):
        assert handler._sanitize_financial_amount("-5.00") == 0.0

    def test_sanitize_financial_amount_huge(self, handler):
        assert handler._sanitize_financial_amount("999999999.99") == 99_999_999.99

    def test_sanitize_financial_amount_invalid(self, handler):
        assert handler._sanitize_financial_amount("not-a-number") == 0.0

    def test_sanitize_financial_amount_none(self, handler):
        assert handler._sanitize_financial_amount(None) == 0.0

    def test_sanitize_currency_code_valid(self, handler):
        assert handler._sanitize_currency_code("usd") == "USD"
        assert handler._sanitize_currency_code("EUR") == "EUR"

    def test_sanitize_currency_code_invalid(self, handler):
        assert handler._sanitize_currency_code("XYZ") == "USD"

    def test_sanitize_currency_code_none(self, handler):
        assert handler._sanitize_currency_code(None) == "USD"

    def test_sanitize_currency_code_not_string(self, handler):
        assert handler._sanitize_currency_code(123) == "USD"

    def test_normalize_shopify_order(self, handler):
        order = MockShopifyOrder()
        result = handler._normalize_shopify_order(order)
        assert result["id"] == "1001"
        assert result["platform"] == "shopify"
        assert result["customer_name"] == "John Doe"
        assert result["currency"] == "USD"
        assert result["total_price"] == 45.97
        assert len(result["line_items"]) == 1

    def test_normalize_shopify_order_no_customer(self, handler):
        order = MockShopifyOrder(customer=None)
        result = handler._normalize_shopify_order(order)
        assert result["customer_name"] is None

    def test_normalize_shipstation_order(self, handler):
        order = MockShipStationOrder()
        result = handler._normalize_shipstation_order(order)
        assert result["id"] == "3001"
        assert result["platform"] == "shipstation"
        assert result["customer_name"] == "Jane Smith"

    def test_normalize_walmart_order(self, handler):
        order = MockWalmartOrder()
        result = handler._normalize_walmart_order(order)
        assert result["id"] == "WM-5001"
        assert result["platform"] == "walmart"
        assert result["customer_name"] == "Bob Wilson"

    def test_normalize_shopify_product(self, handler):
        product = MockShopifyProduct()
        result = handler._normalize_shopify_product(product)
        assert result["id"] == "2001"
        assert result["platform"] == "shopify"
        assert result["sku"] == "WDG-001"
        assert result["price"] == 19.99
        assert result["tags"] == ["sale", "featured"]

    def test_normalize_shopify_product_no_variants(self, handler):
        product = MockShopifyProduct(variants=[])
        result = handler._normalize_shopify_product(product)
        assert result["sku"] is None
        assert result["price"] == 0
        assert result["inventory_quantity"] == 0

    def test_normalize_walmart_item(self, handler):
        item = MockWalmartItem()
        result = handler._normalize_walmart_item(item)
        assert result["id"] == "WM-SKU-001"
        assert result["platform"] == "walmart"
        assert result["vendor"] == "Walmart Brand"


# ---------------------------------------------------------------------------
# Required Credentials Tests
# ---------------------------------------------------------------------------


class TestRequiredCredentials:
    """Tests for _get_required_credentials helper."""

    def test_shopify_requires_shop_url_and_token(self, handler):
        creds = handler._get_required_credentials("shopify")
        assert "shop_url" in creds
        assert "access_token" in creds

    def test_shipstation_requires_api_key_and_secret(self, handler):
        creds = handler._get_required_credentials("shipstation")
        assert "api_key" in creds
        assert "api_secret" in creds

    def test_walmart_requires_client_id_and_secret(self, handler):
        creds = handler._get_required_credentials("walmart")
        assert "client_id" in creds
        assert "client_secret" in creds

    def test_unknown_platform_returns_empty(self, handler):
        assert handler._get_required_credentials("unknown") == []


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_pagination_offset_exceeds_results(self, handler, mock_shopify_connector):
        _connect_shopify(mock_shopify_connector)
        request = MockRequest(
            method="GET",
            path="/api/v1/ecommerce/orders",
            query={"offset": "1000"},
        )
        result = await handler.handle_request(request)
        assert _status(result) == 200
        body = _body(result)
        assert body["orders"] == []

    @pytest.mark.asyncio
    async def test_invalid_resource_id_in_path(self, handler):
        """Resource IDs with special characters should be rejected."""
        _connect_shopify()
        request = MockRequest(
            method="GET", path="/api/v1/ecommerce/shopify/orders/<script>alert(1)</script>"
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_carrier_too_long(self, handler):
        _connect_shipstation()
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/ship",
            _body={
                "platform": "shipstation",
                "order_id": "3001",
                "carrier": "x" * 100,
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_service_too_long(self, handler):
        _connect_shipstation()
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/ship",
            _body={
                "platform": "shipstation",
                "order_id": "3001",
                "service": "x" * 100,
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400

    def test_get_circuit_breaker_status_method(self, handler):
        status = handler.get_circuit_breaker_status()
        assert "state" in status
        assert "failure_count" in status

    @pytest.mark.asyncio
    async def test_declared_value_zero_rejected(self, handler):
        _connect_shipstation()
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/ship",
            _body={
                "platform": "shipstation",
                "order_id": "3001",
                "declared_value": 0,
            },
        )
        result = await handler.handle_request(request)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_connect_stores_credentials(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/connect",
            _body={
                "platform": "walmart",
                "credentials": {
                    "client_id": "wm_id",
                    "client_secret": "wm_secret",
                },
            },
        )
        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            await handler.handle_request(request)
        assert "walmart" in _platform_credentials
        assert _platform_credentials["walmart"]["credentials"]["client_id"] == "wm_id"

    @pytest.mark.asyncio
    async def test_unsupported_platform_for_shipment(self, handler):
        _connect_walmart()
        request = MockRequest(
            method="POST",
            path="/api/v1/ecommerce/ship",
            _body={
                "platform": "walmart",
                "order_id": "WM5001",
            },
        )
        with patch.object(
            handler, "_get_connector", new_callable=AsyncMock, return_value=MagicMock()
        ):
            result = await handler.handle_request(request)
        assert _status(result) == 400
        assert "Unsupported" in _body(result)["error"]
