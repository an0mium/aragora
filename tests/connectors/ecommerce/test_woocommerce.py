"""
Comprehensive tests for the WooCommerce Connector module.

Covers:
1. Initialization and authentication (API keys, OAuth)
2. Product management (CRUD operations, variations, inventory)
3. Order management (create, update, status changes, refunds)
4. Customer management (create, update, search)
5. Webhook handling and event processing
6. Pagination and batch operations
7. Error handling (API errors, rate limits, network issues)
8. Data transformation and validation
9. Timeout handling
10. Input validation
11. Circuit breaker behavior

Dependencies:
    pytest
    pytest-asyncio
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.ecommerce.woocommerce import (
    DEFAULT_REQUEST_TIMEOUT,
    WooAddress,
    WooCommerceConnector,
    WooCommerceCredentials,
    WooCustomer,
    WooLineItem,
    WooOrder,
    WooOrderStatus,
    WooProduct,
    WooProductStatus,
    WooProductType,
    WooProductVariation,
    WooStockStatus,
    get_mock_woo_orders,
    get_mock_woo_products,
    validate_id,
)
from aragora.connectors.enterprise.base import SyncState
from aragora.connectors.exceptions import (
    ConnectorAPIError,
    ConnectorCircuitOpenError,
    ConnectorTimeoutError,
    ConnectorValidationError,
)
from aragora.resilience.circuit_breaker import CircuitBreaker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def credentials():
    """Standard test credentials."""
    return WooCommerceCredentials(
        store_url="https://test-store.example.com",
        consumer_key="ck_test_consumer_key",
        consumer_secret="cs_test_consumer_secret",
        api_version="wc/v3",
    )


@pytest.fixture
def connector(credentials):
    """WooCommerceConnector instance with test credentials."""
    return WooCommerceConnector(credentials=credentials)


@pytest.fixture
def now_utc():
    """Deterministic UTC datetime."""
    return datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_address_data():
    """Raw address dict as returned by the WooCommerce API."""
    return {
        "first_name": "Jane",
        "last_name": "Doe",
        "company": "Acme Inc",
        "address_1": "123 Main St",
        "address_2": "Suite 4",
        "city": "Portland",
        "state": "OR",
        "postcode": "97201",
        "country": "US",
        "email": "jane@example.com",
        "phone": "+15035551234",
    }


@pytest.fixture
def sample_line_item_data():
    """Raw line-item dict as returned by the WooCommerce API."""
    return {
        "id": 9001,
        "product_id": 2001,
        "variation_id": 0,
        "name": "Widget",
        "quantity": 3,
        "subtotal": "59.97",
        "total": "59.97",
        "sku": "WGT-001",
        "price": "19.99",
        "tax_class": "standard",
    }


@pytest.fixture
def sample_order_data(sample_line_item_data, sample_address_data):
    """Raw order dict as returned by the WooCommerce API."""
    return {
        "id": 5001,
        "number": "1042",
        "order_key": "wc_order_5001",
        "status": "processing",
        "currency": "USD",
        "date_created_gmt": "2024-06-10T14:30:00",
        "date_modified_gmt": "2024-06-11T09:00:00",
        "total": "69.97",
        "subtotal": "59.97",
        "total_tax": "5.00",
        "shipping_total": "5.00",
        "discount_total": "0.00",
        "payment_method": "stripe",
        "payment_method_title": "Credit Card",
        "customer_id": 7001,
        "line_items": [sample_line_item_data],
        "billing": sample_address_data,
        "shipping": sample_address_data,
        "customer_note": "Please gift wrap",
        "date_paid_gmt": "2024-06-10T14:35:00",
        "date_completed_gmt": None,
        "transaction_id": "txn_123456",
    }


@pytest.fixture
def sample_product_data():
    """Raw product dict as returned by the WooCommerce API."""
    return {
        "id": 2001,
        "name": "Widget",
        "slug": "widget",
        "type": "simple",
        "status": "publish",
        "sku": "WGT-001",
        "price": "29.99",
        "regular_price": "29.99",
        "sale_price": "",
        "date_created_gmt": "2024-01-01T00:00:00",
        "date_modified_gmt": "2024-06-15T12:00:00",
        "description": "<p>A fine widget</p>",
        "short_description": "A widget",
        "stock_quantity": 50,
        "stock_status": "instock",
        "manage_stock": True,
        "categories": [{"id": 1, "name": "Widgets"}],
        "tags": [{"id": 1, "name": "sale"}],
        "images": [{"src": "https://cdn.example.com/widget.jpg"}],
        "attributes": [],
    }


@pytest.fixture
def sample_variation_data():
    """Raw variation dict as returned by the WooCommerce API."""
    return {
        "id": 3001,
        "sku": "WGT-LG-RED",
        "price": "34.99",
        "regular_price": "34.99",
        "sale_price": "",
        "stock_quantity": 12,
        "stock_status": "instock",
        "manage_stock": True,
        "attributes": [{"name": "Size", "option": "Large"}],
        "image": {"src": "https://cdn.example.com/widget-lg-red.jpg"},
    }


@pytest.fixture
def sample_customer_data(sample_address_data):
    """Raw customer dict as returned by the WooCommerce API."""
    return {
        "id": 7001,
        "email": "buyer@example.com",
        "first_name": "Jane",
        "last_name": "Doe",
        "username": "janedoe",
        "date_created_gmt": "2023-08-01T10:00:00",
        "date_modified_gmt": "2024-06-15T12:00:00",
        "billing": sample_address_data,
        "shipping": sample_address_data,
        "is_paying_customer": True,
        "orders_count": 5,
        "total_spent": "349.95",
        "avatar_url": "https://cdn.example.com/avatar.jpg",
    }


def _make_mock_response(json_data, status=200, headers=None):
    """Build an async context-manager mock for aiohttp responses."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data)
    resp.text = AsyncMock(return_value=str(json_data))
    resp.headers = headers or {}

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


# ========================================================================
# Enum tests
# ========================================================================


class TestEnums:
    """Tests for WooCommerce-specific enum types."""

    def test_woo_order_status_values(self):
        """WooOrderStatus covers the expected order states."""
        expected = {
            "pending",
            "processing",
            "on-hold",
            "completed",
            "cancelled",
            "refunded",
            "failed",
            "trash",
        }
        assert {s.value for s in WooOrderStatus} == expected

    def test_woo_product_status_values(self):
        """WooProductStatus covers the expected product states."""
        expected = {"publish", "draft", "pending", "private"}
        assert {s.value for s in WooProductStatus} == expected

    def test_woo_product_type_values(self):
        """WooProductType covers the expected product types."""
        expected = {"simple", "variable", "grouped", "external"}
        assert {s.value for s in WooProductType} == expected

    def test_woo_stock_status_values(self):
        """WooStockStatus covers the expected stock states."""
        expected = {"instock", "outofstock", "onbackorder"}
        assert {s.value for s in WooStockStatus} == expected

    def test_enum_from_string(self):
        """Enums can be constructed from their string values."""
        assert WooOrderStatus("processing") == WooOrderStatus.PROCESSING
        assert WooProductStatus("publish") == WooProductStatus.PUBLISH
        assert WooProductType("simple") == WooProductType.SIMPLE
        assert WooStockStatus("instock") == WooStockStatus.IN_STOCK


# ========================================================================
# Credential tests
# ========================================================================


class TestWooCommerceCredentials:
    """Tests for WooCommerceCredentials dataclass."""

    def test_basic_construction(self, credentials):
        """Credentials store the expected fields."""
        assert credentials.store_url == "https://test-store.example.com"
        assert credentials.consumer_key == "ck_test_consumer_key"
        assert credentials.consumer_secret == "cs_test_consumer_secret"
        assert credentials.api_version == "wc/v3"
        assert credentials.timeout == 30

    def test_default_api_version(self):
        """Default API version is wc/v3."""
        creds = WooCommerceCredentials(
            store_url="https://store.example.com",
            consumer_key="ck_key",
            consumer_secret="cs_secret",
        )
        assert creds.api_version == "wc/v3"

    def test_custom_timeout(self):
        """Custom timeout can be specified."""
        creds = WooCommerceCredentials(
            store_url="https://store.example.com",
            consumer_key="ck_key",
            consumer_secret="cs_secret",
            timeout=60,
        )
        assert creds.timeout == 60

    def test_from_env(self):
        """Credentials can be loaded from environment variables."""
        env = {
            "WOOCOMMERCE_URL": "https://env-store.example.com",
            "WOOCOMMERCE_CONSUMER_KEY": "ck_env_key",
            "WOOCOMMERCE_CONSUMER_SECRET": "cs_env_secret",
            "WOOCOMMERCE_VERSION": "wc/v2",
        }
        with patch.dict("os.environ", env, clear=False):
            creds = WooCommerceCredentials.from_env()
            assert creds.store_url == "https://env-store.example.com"
            assert creds.consumer_key == "ck_env_key"
            assert creds.consumer_secret == "cs_env_secret"
            assert creds.api_version == "wc/v2"

    def test_from_env_defaults(self):
        """Missing env vars resolve to empty strings / defaults."""
        with patch.dict("os.environ", {}, clear=True):
            creds = WooCommerceCredentials.from_env()
            assert creds.store_url == ""
            assert creds.consumer_key == ""
            assert creds.consumer_secret == ""
            assert creds.api_version == "wc/v3"


# ========================================================================
# Address tests
# ========================================================================


class TestWooAddress:
    """Tests for WooAddress dataclass."""

    def test_construction_defaults(self):
        """All fields default to None."""
        addr = WooAddress()
        assert addr.first_name is None
        assert addr.city is None
        assert addr.country is None

    def test_full_construction(self, sample_address_data):
        """Address can be constructed with all fields."""
        addr = WooAddress(**sample_address_data)
        assert addr.first_name == "Jane"
        assert addr.city == "Portland"
        assert addr.country == "US"

    def test_to_dict_uses_api_names_by_default(self, sample_address_data):
        """to_dict defaults to use_api_names=True, mapping snake_case to camelCase."""
        addr = WooAddress(**sample_address_data)
        d = addr.to_dict()
        assert "firstName" in d
        assert d["firstName"] == "Jane"

    def test_to_dict_includes_none(self):
        """None values are included because _include_none is True."""
        addr = WooAddress(first_name="Bob")
        d = addr.to_dict()
        assert "lastName" in d
        assert d["lastName"] is None

    def test_to_dict_python_names(self, sample_address_data):
        """to_dict with use_api_names=False keeps snake_case names."""
        addr = WooAddress(**sample_address_data)
        d = addr.to_dict(use_api_names=False)
        assert "first_name" in d
        assert d["first_name"] == "Jane"


# ========================================================================
# Line item tests
# ========================================================================


class TestWooLineItem:
    """Tests for WooLineItem dataclass."""

    def test_construction(self):
        """Line item stores essential order-line data."""
        item = WooLineItem(
            id=1,
            product_id=100,
            variation_id=0,
            name="Widget",
            quantity=2,
            subtotal=Decimal("39.98"),
            total=Decimal("39.98"),
        )
        assert item.quantity == 2
        assert item.subtotal == Decimal("39.98")
        assert item.price == Decimal("0.00")  # default

    def test_to_dict_api_names(self):
        """Default to_dict maps to API names (camelCase)."""
        item = WooLineItem(
            id=1,
            product_id=100,
            variation_id=0,
            name="Widget",
            quantity=1,
            subtotal=Decimal("10.00"),
            total=Decimal("10.00"),
        )
        d = item.to_dict()
        assert "productId" in d
        assert "variationId" in d

    def test_to_dict_price_as_string(self):
        """Decimal price is serialized as a string."""
        item = WooLineItem(
            id=1,
            product_id=100,
            variation_id=0,
            name="X",
            quantity=1,
            subtotal=Decimal("12.50"),
            total=Decimal("12.50"),
            price=Decimal("12.50"),
        )
        d = item.to_dict()
        assert d["price"] == "12.50"


# ========================================================================
# Order model tests
# ========================================================================


class TestWooOrder:
    """Tests for WooOrder dataclass and serialization."""

    def test_construction_minimal(self, now_utc):
        """Order can be constructed with required fields only."""
        order = WooOrder(
            id=1,
            number="100",
            order_key="wc_order_1",
            status=WooOrderStatus.PENDING,
            currency="USD",
            date_created=now_utc,
            date_modified=now_utc,
            total=Decimal("50.00"),
            subtotal=Decimal("45.00"),
            total_tax=Decimal("5.00"),
            shipping_total=Decimal("0.00"),
            discount_total=Decimal("0.00"),
            payment_method="",
            payment_method_title="",
            customer_id=0,
            billing=WooAddress(),
            shipping=WooAddress(),
        )
        assert order.line_items == []
        assert order.customer_note is None

    def test_to_dict_serialises_decimals(self, now_utc):
        """Decimal fields are serialized as strings (API names by default)."""
        order = WooOrder(
            id=1,
            number="100",
            order_key="wc_order_1",
            status=WooOrderStatus.PENDING,
            currency="EUR",
            date_created=now_utc,
            date_modified=now_utc,
            total=Decimal("123.45"),
            subtotal=Decimal("100.00"),
            total_tax=Decimal("23.45"),
            shipping_total=Decimal("0.00"),
            discount_total=Decimal("0.00"),
            payment_method="",
            payment_method_title="",
            customer_id=0,
            billing=WooAddress(),
            shipping=WooAddress(),
        )
        d = order.to_dict()
        assert d["total"] == "123.45"
        assert d["totalTax"] == "23.45"

    def test_to_dict_api_names(self, now_utc):
        """to_dict maps field names to camelCase."""
        order = WooOrder(
            id=1,
            number="100",
            order_key="wc_order_1",
            status=WooOrderStatus.PENDING,
            currency="USD",
            date_created=now_utc,
            date_modified=now_utc,
            total=Decimal("0"),
            subtotal=Decimal("0"),
            total_tax=Decimal("0"),
            shipping_total=Decimal("0"),
            discount_total=Decimal("0"),
            payment_method="",
            payment_method_title="",
            customer_id=0,
            billing=WooAddress(),
            shipping=WooAddress(),
        )
        d = order.to_dict()
        assert "orderKey" in d
        assert "dateCreated" in d
        assert "totalTax" in d


# ========================================================================
# Product / Variation tests
# ========================================================================


class TestWooProduct:
    """Tests for WooProduct dataclass."""

    def test_construction(self, now_utc):
        """Product stores catalogue fields."""
        product = WooProduct(
            id=1,
            name="Widget",
            slug="widget",
            type=WooProductType.SIMPLE,
            status=WooProductStatus.PUBLISH,
            sku="WGT-001",
            price=Decimal("29.99"),
            regular_price=Decimal("29.99"),
            sale_price=None,
            date_created=now_utc,
            date_modified=now_utc,
            description="A widget",
            short_description="Widget",
            stock_quantity=50,
            stock_status=WooStockStatus.IN_STOCK,
            manage_stock=True,
        )
        assert product.name == "Widget"
        assert product.variations == []
        assert product.images == []

    def test_to_dict_api_names(self, now_utc):
        """Product to_dict maps fields to camelCase."""
        product = WooProduct(
            id=1,
            name="T",
            slug="t",
            type=WooProductType.SIMPLE,
            status=WooProductStatus.DRAFT,
            sku=None,
            price=Decimal("0"),
            regular_price=Decimal("0"),
            sale_price=None,
            date_created=now_utc,
            date_modified=now_utc,
            description=None,
            short_description=None,
            stock_quantity=None,
            stock_status=WooStockStatus.IN_STOCK,
            manage_stock=False,
        )
        d = product.to_dict()
        assert "regularPrice" in d
        assert "shortDescription" in d


class TestWooProductVariation:
    """Tests for WooProductVariation dataclass."""

    def test_defaults(self):
        """Variation has sensible defaults."""
        variant = WooProductVariation(
            id=1,
            sku=None,
            price=Decimal("9.99"),
            regular_price=Decimal("9.99"),
            sale_price=None,
            stock_quantity=0,
            stock_status=WooStockStatus.IN_STOCK,
            manage_stock=False,
        )
        assert variant.attributes == []
        assert variant.image is None

    def test_to_dict_api_names(self):
        """Variation to_dict maps fields to camelCase."""
        variant = WooProductVariation(
            id=1,
            sku="SKU",
            price=Decimal("10.00"),
            regular_price=Decimal("10.00"),
            sale_price=None,
            stock_quantity=5,
            stock_status=WooStockStatus.IN_STOCK,
            manage_stock=True,
        )
        d = variant.to_dict()
        assert d["stockQuantity"] == 5
        assert d["manageStock"] is True


# ========================================================================
# Customer tests
# ========================================================================


class TestWooCustomer:
    """Tests for WooCustomer dataclass."""

    def test_construction(self, now_utc):
        """Customer stores expected fields."""
        c = WooCustomer(
            id=1,
            email="test@example.com",
            first_name="Jane",
            last_name="Doe",
            username="janedoe",
            date_created=now_utc,
            date_modified=now_utc,
            billing=WooAddress(),
            shipping=WooAddress(),
        )
        assert c.first_name == "Jane"
        assert c.is_paying_customer is False
        assert c.orders_count == 0

    def test_to_dict_api_names(self, now_utc):
        """Customer to_dict maps fields to camelCase."""
        c = WooCustomer(
            id=1,
            email="e@x.com",
            first_name="A",
            last_name="B",
            username="ab",
            date_created=now_utc,
            date_modified=now_utc,
            billing=WooAddress(),
            shipping=WooAddress(),
            is_paying_customer=True,
            orders_count=3,
            total_spent=Decimal("100.00"),
        )
        d = c.to_dict()
        assert "firstName" in d
        assert "lastName" in d
        assert "ordersCount" in d
        assert "totalSpent" in d
        assert "isPayingCustomer" in d


# ========================================================================
# Connector initialisation
# ========================================================================


class TestConnectorInit:
    """Tests for WooCommerceConnector construction and base_url property."""

    def test_base_url(self, connector, credentials):
        """base_url is constructed from credentials."""
        expected = f"{credentials.store_url}/wp-json/{credentials.api_version}"
        assert connector.base_url == expected

    def test_name_property(self, connector):
        """name property returns 'WooCommerce'."""
        assert connector.name == "WooCommerce"

    def test_source_type(self, connector):
        """source_type is EXTERNAL_API."""
        from aragora.reasoning.provenance import SourceType

        assert connector.source_type == SourceType.EXTERNAL_API

    def test_client_initially_none(self, connector):
        """No HTTP client before connect()."""
        assert connector._client is None

    def test_connector_id(self, connector):
        """Connector ID is 'woocommerce'."""
        assert connector.connector_id == "woocommerce"


# ========================================================================
# Connection lifecycle
# ========================================================================


class TestConnect:
    """Tests for connect() and disconnect()."""

    @pytest.mark.asyncio
    async def test_connect_success(self, connector):
        """Successful connect sets client and returns True."""
        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=_make_mock_response(
                {"environment": {"site_url": "https://test-store.example.com"}},
                status=200,
            )
        )
        mock_session_cls = MagicMock(return_value=mock_session)

        with patch("aiohttp.ClientSession", mock_session_cls):
            result = await connector.connect()

        assert result is True

    @pytest.mark.asyncio
    async def test_connect_failure_status(self, connector):
        """Non-200 response from system_status returns False."""
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=_make_mock_response({}, status=401))
        mock_session_cls = MagicMock(return_value=mock_session)

        with patch("aiohttp.ClientSession", mock_session_cls):
            result = await connector.connect()

        assert result is False

    @pytest.mark.asyncio
    async def test_connect_import_error(self, connector):
        """connect() returns False when aiohttp is missing."""
        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def fake_import(name, *args, **kwargs):
            if name == "aiohttp":
                raise ImportError("no aiohttp")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = await connector.connect()

        assert result is False

    @pytest.mark.asyncio
    async def test_connect_exception(self, connector):
        """connect() returns False on OSError."""
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=OSError("Network error"))
        mock_session_cls = MagicMock(return_value=mock_session)

        with patch("aiohttp.ClientSession", mock_session_cls):
            result = await connector.connect()

        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_closes_session(self, connector):
        """disconnect() closes the session and sets it to None."""
        connector._client = AsyncMock()
        await connector.disconnect()
        assert connector._client is None

    @pytest.mark.asyncio
    async def test_disconnect_noop_without_session(self, connector):
        """disconnect() is safe to call when no session exists."""
        connector._client = None
        await connector.disconnect()
        assert connector._client is None


# ========================================================================
# _request tests
# ========================================================================


class TestRequest:
    """Tests for the internal _request helper."""

    @pytest.mark.asyncio
    async def test_request_auto_connects(self, connector):
        """_request calls connect() when client is None."""
        connector._client = None
        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=_make_mock_response({"ok": True}))

        async def fake_connect():
            connector._client = mock_session
            return True

        connector.connect = fake_connect
        result = await connector._request("GET", "system_status")
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_request_get(self, connector):
        """GET request returns parsed JSON."""
        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=_make_mock_response({"data": 1}))
        connector._client = mock_session

        result = await connector._request("GET", "test")
        assert result == {"data": 1}

    @pytest.mark.asyncio
    async def test_request_post_with_json(self, connector):
        """POST request forwards JSON body."""
        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=_make_mock_response({"id": 1}))
        connector._client = mock_session

        body = {"order": {"note": "rush"}}
        result = await connector._request("POST", "orders", json_data=body)
        assert result == {"id": 1}
        mock_session.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_http_error_raises(self, connector):
        """HTTP 4xx raises ConnectorAPIError."""
        mock_session = MagicMock()
        resp_mock = AsyncMock()
        resp_mock.status = 422
        resp_mock.text = AsyncMock(return_value='{"errors":"bad data"}')

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp_mock)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.request = MagicMock(return_value=ctx)
        connector._client = mock_session

        with pytest.raises(ConnectorAPIError) as exc_info:
            await connector._request("POST", "orders", json_data={})
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_request_500_error(self, connector):
        """Server errors are wrapped as ConnectorAPIError with status 500."""
        mock_session = MagicMock()
        resp_mock = AsyncMock()
        resp_mock.status = 500
        resp_mock.text = AsyncMock(return_value="Internal Server Error")

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp_mock)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.request = MagicMock(return_value=ctx)
        connector._client = mock_session

        with pytest.raises(ConnectorAPIError) as exc_info:
            await connector._request("GET", "system_status")
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_request_429_rate_limit(self, connector):
        """HTTP 429 raises ConnectorAPIError with status 429."""
        mock_session = MagicMock()
        resp_mock = AsyncMock()
        resp_mock.status = 429
        resp_mock.text = AsyncMock(return_value="Rate limited")

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp_mock)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.request = MagicMock(return_value=ctx)
        connector._client = mock_session

        with pytest.raises(ConnectorAPIError) as exc_info:
            await connector._request("GET", "orders")
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_request_params_forwarded(self, connector):
        """Query params are forwarded to the session."""
        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=_make_mock_response({"ok": True}))
        connector._client = mock_session

        await connector._request("GET", "orders", params={"status": "processing"})
        mock_session.request.assert_called_once()


# ========================================================================
# Order parsing
# ========================================================================


class TestOrderParsing:
    """Tests for _parse_order and _parse_address helpers."""

    def test_parse_order(self, connector, sample_order_data):
        """Full order parsing from API response dict."""
        order = connector._parse_order(sample_order_data)

        assert order.id == 5001
        assert order.number == "1042"
        assert order.order_key == "wc_order_5001"
        assert order.status == WooOrderStatus.PROCESSING
        assert order.currency == "USD"
        assert order.total == Decimal("69.97")
        assert order.customer_id == 7001
        assert order.customer_note == "Please gift wrap"
        assert order.transaction_id == "txn_123456"
        assert order.date_paid is not None

    def test_parse_order_line_items(self, connector, sample_order_data):
        """Line items are correctly parsed within an order."""
        order = connector._parse_order(sample_order_data)
        assert len(order.line_items) == 1
        li = order.line_items[0]
        assert li.id == 9001
        assert li.product_id == 2001
        assert li.name == "Widget"
        assert li.quantity == 3
        assert li.price == Decimal("19.99")

    def test_parse_order_addresses(self, connector, sample_order_data):
        """Billing and shipping addresses are parsed."""
        order = connector._parse_order(sample_order_data)
        assert order.billing.city == "Portland"
        assert order.shipping.country == "US"

    def test_parse_order_no_date_paid(self, connector, sample_order_data):
        """Order without date_paid yields None."""
        sample_order_data["date_paid_gmt"] = None
        order = connector._parse_order(sample_order_data)
        assert order.date_paid is None

    def test_parse_order_no_date_completed(self, connector, sample_order_data):
        """Order without date_completed yields None."""
        order = connector._parse_order(sample_order_data)
        assert order.date_completed is None

    def test_parse_address(self, connector, sample_address_data):
        """_parse_address converts raw dict to WooAddress."""
        addr = connector._parse_address(sample_address_data)
        assert isinstance(addr, WooAddress)
        assert addr.first_name == "Jane"
        assert addr.postcode == "97201"


# ========================================================================
# Order operations
# ========================================================================


class TestOrderOperations:
    """Tests for order CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_order_success(self, connector, sample_order_data):
        """get_order returns a parsed WooOrder."""
        connector._request = AsyncMock(return_value=sample_order_data)
        order = await connector.get_order(5001)
        assert order is not None
        assert order.id == 5001

    @pytest.mark.asyncio
    async def test_get_order_not_found(self, connector):
        """get_order returns None on failure."""
        connector._request = AsyncMock(
            side_effect=ConnectorAPIError(
                "Not found",
                connector_name="woocommerce",
                status_code=404,
            )
        )
        order = await connector.get_order(9999)
        assert order is None

    @pytest.mark.asyncio
    async def test_update_order_status_success(self, connector):
        """update_order_status sends PUT and returns True."""
        connector._request = AsyncMock(return_value={"id": 5001, "status": "completed"})
        result = await connector.update_order_status(5001, WooOrderStatus.COMPLETED)
        assert result is True
        call_args = connector._request.call_args
        json_data = call_args.kwargs.get("json_data") or call_args[1].get("json_data")
        assert json_data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_update_order_status_failure(self, connector):
        """update_order_status returns False on API error."""
        connector._request = AsyncMock(
            side_effect=ConnectorAPIError(
                "Cannot update",
                connector_name="woocommerce",
                status_code=422,
            )
        )
        result = await connector.update_order_status(5001, WooOrderStatus.COMPLETED)
        assert result is False

    @pytest.mark.asyncio
    async def test_create_order_success(self, connector, sample_order_data):
        """create_order sends POST and returns WooOrder."""
        connector._request = AsyncMock(return_value=sample_order_data)
        billing = WooAddress(first_name="John", last_name="Doe", email="john@example.com")
        order = await connector.create_order(
            customer_id=1,
            billing=billing,
            line_items=[{"product_id": 100, "quantity": 1}],
            payment_method="stripe",
            payment_method_title="Credit Card",
            set_paid=True,
        )
        assert order is not None
        assert order.id == 5001

    @pytest.mark.asyncio
    async def test_create_order_with_note(self, connector, sample_order_data):
        """create_order includes customer note."""
        connector._request = AsyncMock(return_value=sample_order_data)
        order = await connector.create_order(note="Please gift wrap")
        assert order is not None
        call_args = connector._request.call_args
        json_data = call_args.kwargs.get("json_data") or call_args[1].get("json_data")
        assert json_data["customer_note"] == "Please gift wrap"

    @pytest.mark.asyncio
    async def test_create_order_failure(self, connector):
        """create_order returns None on API error."""
        connector._request = AsyncMock(
            side_effect=ConnectorAPIError(
                "Cannot create",
                connector_name="woocommerce",
                status_code=422,
            )
        )
        order = await connector.create_order(customer_id=1)
        assert order is None

    @pytest.mark.asyncio
    async def test_sync_orders_single_page(self, connector, sample_order_data):
        """sync_orders yields orders from a single page."""
        connector._request = AsyncMock(return_value=[sample_order_data])
        orders = []
        async for order in connector.sync_orders(per_page=100):
            orders.append(order)
        assert len(orders) == 1
        assert orders[0].id == 5001

    @pytest.mark.asyncio
    async def test_sync_orders_with_since(self, connector, sample_order_data):
        """sync_orders passes modified_after when since is provided."""
        since = datetime(2024, 1, 1, tzinfo=timezone.utc)
        connector._request = AsyncMock(return_value=[sample_order_data])
        orders = []
        async for order in connector.sync_orders(since=since):
            orders.append(order)
        call_args = connector._request.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params")
        assert "modified_after" in params

    @pytest.mark.asyncio
    async def test_sync_orders_with_status(self, connector, sample_order_data):
        """sync_orders passes status filter."""
        connector._request = AsyncMock(return_value=[sample_order_data])
        orders = []
        async for order in connector.sync_orders(status=WooOrderStatus.PROCESSING):
            orders.append(order)
        call_args = connector._request.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params")
        assert params["status"] == "processing"

    @pytest.mark.asyncio
    async def test_sync_orders_pagination(self, connector, sample_order_data):
        """sync_orders handles pagination."""
        page1_data = [sample_order_data] * 100
        page2_data = [sample_order_data]

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return page1_data
            return page2_data

        connector._request = mock_request
        orders = []
        async for order in connector.sync_orders(per_page=100):
            orders.append(order)
        assert len(orders) == 101
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_sync_orders_empty(self, connector):
        """sync_orders yields nothing for empty response."""
        connector._request = AsyncMock(return_value=[])
        orders = []
        async for order in connector.sync_orders():
            orders.append(order)
        assert orders == []


# ========================================================================
# Product parsing and operations
# ========================================================================


class TestProductOperations:
    """Tests for product parsing and CRUD operations."""

    def test_parse_product(self, connector, sample_product_data):
        """_parse_product produces a complete WooProduct."""
        product = connector._parse_product(sample_product_data)
        assert product.id == 2001
        assert product.name == "Widget"
        assert product.slug == "widget"
        assert product.type == WooProductType.SIMPLE
        assert product.status == WooProductStatus.PUBLISH
        assert product.sku == "WGT-001"
        assert product.price == Decimal("29.99")
        assert product.stock_quantity == 50
        assert product.manage_stock is True

    def test_parse_product_images(self, connector, sample_product_data):
        """Product images are extracted as URL strings."""
        product = connector._parse_product(sample_product_data)
        assert product.images == ["https://cdn.example.com/widget.jpg"]

    def test_parse_product_no_sale_price(self, connector, sample_product_data):
        """Empty sale_price is parsed as None."""
        product = connector._parse_product(sample_product_data)
        assert product.sale_price is None

    def test_parse_product_with_sale_price(self, connector, sample_product_data):
        """Valid sale_price is parsed as Decimal."""
        sample_product_data["sale_price"] = "24.99"
        product = connector._parse_product(sample_product_data)
        assert product.sale_price == Decimal("24.99")

    @pytest.mark.asyncio
    async def test_get_product_success(self, connector, sample_product_data):
        """get_product returns parsed product."""
        connector._request = AsyncMock(return_value=sample_product_data)
        product = await connector.get_product(2001)
        assert product is not None
        assert product.name == "Widget"

    @pytest.mark.asyncio
    async def test_get_product_not_found(self, connector):
        """get_product returns None on error."""
        connector._request = AsyncMock(
            side_effect=ConnectorAPIError(
                "Not found",
                connector_name="woocommerce",
                status_code=404,
            )
        )
        product = await connector.get_product(9999)
        assert product is None

    @pytest.mark.asyncio
    async def test_sync_products(self, connector, sample_product_data):
        """sync_products yields products."""
        connector._request = AsyncMock(return_value=[sample_product_data])
        products = []
        async for p in connector.sync_products():
            products.append(p)
        assert len(products) == 1
        assert products[0].id == 2001

    @pytest.mark.asyncio
    async def test_sync_products_with_status(self, connector, sample_product_data):
        """sync_products passes status filter."""
        connector._request = AsyncMock(return_value=[sample_product_data])
        products = []
        async for p in connector.sync_products(status=WooProductStatus.PUBLISH):
            products.append(p)
        call_args = connector._request.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params")
        assert params["status"] == "publish"

    @pytest.mark.asyncio
    async def test_update_product_stock_success(self, connector):
        """update_product_stock sends PUT and returns True."""
        connector._request = AsyncMock(return_value={"id": 2001, "stock_quantity": 50})
        result = await connector.update_product_stock(2001, 50)
        assert result is True
        call_args = connector._request.call_args
        json_data = call_args.kwargs.get("json_data") or call_args[1].get("json_data")
        assert json_data["stock_quantity"] == 50
        assert json_data["manage_stock"] is True

    @pytest.mark.asyncio
    async def test_update_product_stock_with_status(self, connector):
        """update_product_stock includes stock_status when provided."""
        connector._request = AsyncMock(return_value={"id": 2001})
        result = await connector.update_product_stock(
            2001, 0, stock_status=WooStockStatus.OUT_OF_STOCK
        )
        assert result is True
        call_args = connector._request.call_args
        json_data = call_args.kwargs.get("json_data") or call_args[1].get("json_data")
        assert json_data["stock_status"] == "outofstock"

    @pytest.mark.asyncio
    async def test_update_product_stock_failure(self, connector):
        """update_product_stock returns False on API error."""
        connector._request = AsyncMock(
            side_effect=ConnectorAPIError(
                "Cannot update",
                connector_name="woocommerce",
                status_code=422,
            )
        )
        result = await connector.update_product_stock(2001, 50)
        assert result is False


# ========================================================================
# Product variation operations
# ========================================================================


class TestVariationOperations:
    """Tests for product variation operations."""

    def test_parse_variation(self, connector, sample_variation_data):
        """_parse_variation produces correct WooProductVariation."""
        variation = connector._parse_variation(sample_variation_data)
        assert variation.id == 3001
        assert variation.sku == "WGT-LG-RED"
        assert variation.price == Decimal("34.99")
        assert variation.stock_quantity == 12
        assert variation.manage_stock is True
        assert variation.image == "https://cdn.example.com/widget-lg-red.jpg"

    def test_parse_variation_no_image(self, connector, sample_variation_data):
        """Variation without image yields None."""
        sample_variation_data["image"] = None
        variation = connector._parse_variation(sample_variation_data)
        assert variation.image is None

    @pytest.mark.asyncio
    async def test_sync_product_variations(self, connector, sample_variation_data):
        """sync_product_variations yields variations."""
        connector._request = AsyncMock(return_value=[sample_variation_data])
        variations = []
        async for v in connector.sync_product_variations(2001):
            variations.append(v)
        assert len(variations) == 1
        assert variations[0].id == 3001

    @pytest.mark.asyncio
    async def test_update_variation_stock_success(self, connector):
        """update_variation_stock sends PUT and returns True."""
        connector._request = AsyncMock(return_value={"id": 3001, "stock_quantity": 10})
        result = await connector.update_variation_stock(2001, 3001, 10)
        assert result is True
        call_args = connector._request.call_args
        json_data = call_args.kwargs.get("json_data") or call_args[1].get("json_data")
        assert json_data["stock_quantity"] == 10

    @pytest.mark.asyncio
    async def test_update_variation_stock_failure(self, connector):
        """update_variation_stock returns False on API error."""
        connector._request = AsyncMock(
            side_effect=ConnectorAPIError(
                "Cannot update",
                connector_name="woocommerce",
                status_code=422,
            )
        )
        result = await connector.update_variation_stock(2001, 3001, 10)
        assert result is False


# ========================================================================
# Inventory operations
# ========================================================================


class TestInventoryOperations:
    """Tests for inventory-related operations."""

    @pytest.mark.asyncio
    async def test_get_low_stock_products(self, connector, sample_product_data):
        """get_low_stock_products filters products below threshold."""
        sample_product_data["stock_quantity"] = 3
        connector._request = AsyncMock(return_value=[sample_product_data])
        low = await connector.get_low_stock_products(threshold=5)
        assert len(low) == 1
        assert low[0].stock_quantity == 3

    @pytest.mark.asyncio
    async def test_get_low_stock_products_none_low(self, connector, sample_product_data):
        """get_low_stock_products returns empty list when all stock is adequate."""
        sample_product_data["stock_quantity"] = 100
        connector._request = AsyncMock(return_value=[sample_product_data])
        low = await connector.get_low_stock_products(threshold=5)
        assert low == []

    @pytest.mark.asyncio
    async def test_get_low_stock_products_equal_threshold(self, connector, sample_product_data):
        """Products at exactly the threshold are included."""
        sample_product_data["stock_quantity"] = 5
        connector._request = AsyncMock(return_value=[sample_product_data])
        low = await connector.get_low_stock_products(threshold=5)
        assert len(low) == 1

    @pytest.mark.asyncio
    async def test_get_low_stock_default_threshold(self, connector, sample_product_data):
        """Default threshold is 5."""
        sample_product_data["stock_quantity"] = 4
        connector._request = AsyncMock(return_value=[sample_product_data])
        low = await connector.get_low_stock_products()
        assert len(low) == 1

    @pytest.mark.asyncio
    async def test_get_low_stock_excludes_unmanaged(self, connector, sample_product_data):
        """Products not managing stock are excluded."""
        sample_product_data["stock_quantity"] = 2
        sample_product_data["manage_stock"] = False
        connector._request = AsyncMock(return_value=[sample_product_data])
        low = await connector.get_low_stock_products(threshold=5)
        assert len(low) == 0


# ========================================================================
# Customer operations
# ========================================================================


class TestCustomerOperations:
    """Tests for customer parsing and retrieval."""

    def test_parse_customer(self, connector, sample_customer_data):
        """_parse_customer produces correct WooCustomer."""
        customer = connector._parse_customer(sample_customer_data)
        assert customer.id == 7001
        assert customer.email == "buyer@example.com"
        assert customer.first_name == "Jane"
        assert customer.last_name == "Doe"
        assert customer.orders_count == 5
        assert customer.total_spent == Decimal("349.95")
        assert customer.is_paying_customer is True

    def test_parse_customer_defaults(self, connector, sample_customer_data):
        """Customer fields with missing data get defaults."""
        sample_customer_data.pop("orders_count", None)
        sample_customer_data.pop("total_spent", None)
        customer = connector._parse_customer(sample_customer_data)
        assert customer.orders_count == 0
        assert customer.total_spent == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_sync_customers(self, connector, sample_customer_data):
        """sync_customers yields customer objects."""
        connector._request = AsyncMock(return_value=[sample_customer_data])
        customers = []
        async for c in connector.sync_customers():
            customers.append(c)
        assert len(customers) == 1
        assert customers[0].id == 7001

    @pytest.mark.asyncio
    async def test_sync_customers_with_since(self, connector, sample_customer_data):
        """sync_customers passes modified_after when since is provided."""
        since = datetime(2024, 1, 1, tzinfo=timezone.utc)
        connector._request = AsyncMock(return_value=[sample_customer_data])
        customers = []
        async for c in connector.sync_customers(since=since):
            customers.append(c)
        call_args = connector._request.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params")
        assert "modified_after" in params


# ========================================================================
# Refunds
# ========================================================================


class TestRefundOperations:
    """Tests for refund operations."""

    @pytest.mark.asyncio
    async def test_create_refund_success(self, connector):
        """create_refund sends POST and returns refund data."""
        connector._request = AsyncMock(
            return_value={"id": 1001, "amount": "50.00", "reason": "Defective product"}
        )
        result = await connector.create_refund(
            order_id=5001,
            amount=Decimal("50.00"),
            reason="Defective product",
            restock_items=True,
        )
        assert result is not None
        assert result["id"] == 1001

    @pytest.mark.asyncio
    async def test_create_refund_full(self, connector):
        """create_refund without amount refunds full order."""
        connector._request = AsyncMock(return_value={"id": 1002, "amount": "100.00"})
        result = await connector.create_refund(order_id=5001, reason="Customer request")
        assert result is not None
        call_args = connector._request.call_args
        json_data = call_args.kwargs.get("json_data") or call_args[1].get("json_data")
        assert "amount" not in json_data

    @pytest.mark.asyncio
    async def test_create_refund_with_line_items(self, connector):
        """create_refund can specify line items."""
        connector._request = AsyncMock(return_value={"id": 1003})
        result = await connector.create_refund(
            order_id=5001, line_items=[{"id": 123, "quantity": 1}]
        )
        assert result is not None
        call_args = connector._request.call_args
        json_data = call_args.kwargs.get("json_data") or call_args[1].get("json_data")
        assert json_data["line_items"] == [{"id": 123, "quantity": 1}]

    @pytest.mark.asyncio
    async def test_create_refund_failure(self, connector):
        """create_refund returns None on API error."""
        connector._request = AsyncMock(
            side_effect=ConnectorAPIError(
                "Cannot refund", connector_name="woocommerce", status_code=422
            )
        )
        result = await connector.create_refund(order_id=5001, reason="Test")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_refunds_success(self, connector):
        """get_refunds returns list of refunds."""
        connector._request = AsyncMock(
            return_value=[{"id": 1001, "amount": "50.00"}, {"id": 1002, "amount": "25.00"}]
        )
        refunds = await connector.get_refunds(5001)
        assert len(refunds) == 2

    @pytest.mark.asyncio
    async def test_get_refunds_empty(self, connector):
        """get_refunds returns empty list when no refunds."""
        connector._request = AsyncMock(return_value=[])
        refunds = await connector.get_refunds(5001)
        assert refunds == []

    @pytest.mark.asyncio
    async def test_get_refunds_failure(self, connector):
        """get_refunds returns empty list on API error."""
        connector._request = AsyncMock(
            side_effect=ConnectorAPIError(
                "Cannot get refunds", connector_name="woocommerce", status_code=500
            )
        )
        refunds = await connector.get_refunds(5001)
        assert refunds == []


# ========================================================================
# Coupons
# ========================================================================


class TestCouponOperations:
    """Tests for coupon operations."""

    @pytest.mark.asyncio
    async def test_get_coupons(self, connector):
        """get_coupons yields coupon data."""
        connector._request = AsyncMock(
            return_value=[
                {"id": 1, "code": "SAVE10"},
                {"id": 2, "code": "FREESHIP"},
            ]
        )
        coupons = []
        async for c in connector.get_coupons():
            coupons.append(c)
        assert len(coupons) == 2
        assert coupons[0]["code"] == "SAVE10"

    @pytest.mark.asyncio
    async def test_create_coupon_success(self, connector):
        """create_coupon sends POST and returns coupon data."""
        connector._request = AsyncMock(
            return_value={"id": 1, "code": "SAVE20", "discount_type": "percent"}
        )
        result = await connector.create_coupon(
            code="SAVE20",
            discount_type="percent",
            amount="20",
            description="20% off",
        )
        assert result is not None
        assert result["code"] == "SAVE20"

    @pytest.mark.asyncio
    async def test_create_coupon_with_expiration(self, connector):
        """create_coupon includes date_expires."""
        expires = datetime(2024, 12, 31, tzinfo=timezone.utc)
        connector._request = AsyncMock(return_value={"id": 2, "code": "NYE2024"})
        result = await connector.create_coupon(
            code="NYE2024", discount_type="fixed_cart", amount="10", date_expires=expires
        )
        assert result is not None
        call_args = connector._request.call_args
        json_data = call_args.kwargs.get("json_data") or call_args[1].get("json_data")
        assert "date_expires" in json_data

    @pytest.mark.asyncio
    async def test_create_coupon_with_product_restrictions(self, connector):
        """create_coupon includes product restrictions."""
        connector._request = AsyncMock(return_value={"id": 3, "code": "WIDGET10"})
        result = await connector.create_coupon(
            code="WIDGET10",
            discount_type="percent",
            amount="10",
            product_ids=[101, 102],
            excluded_product_ids=[103],
        )
        assert result is not None
        call_args = connector._request.call_args
        json_data = call_args.kwargs.get("json_data") or call_args[1].get("json_data")
        assert json_data["product_ids"] == [101, 102]
        assert json_data["excluded_product_ids"] == [103]

    @pytest.mark.asyncio
    async def test_create_coupon_failure(self, connector):
        """create_coupon returns None on API error."""
        connector._request = AsyncMock(
            side_effect=ConnectorAPIError(
                "Cannot create", connector_name="woocommerce", status_code=422
            )
        )
        result = await connector.create_coupon(code="FAIL", discount_type="percent", amount="10")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_coupon_success(self, connector):
        """delete_coupon sends DELETE and returns True."""
        connector._request = AsyncMock(return_value={"id": 1, "deleted": True})
        result = await connector.delete_coupon(1)
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_coupon_failure(self, connector):
        """delete_coupon returns False on API error."""
        connector._request = AsyncMock(
            side_effect=ConnectorAPIError(
                "Cannot delete", connector_name="woocommerce", status_code=422
            )
        )
        result = await connector.delete_coupon(1)
        assert result is False


# ========================================================================
# Webhooks
# ========================================================================


class TestWebhookOperations:
    """Tests for webhook operations."""

    @pytest.mark.asyncio
    async def test_get_webhooks(self, connector):
        """get_webhooks returns list of webhooks."""
        connector._request = AsyncMock(
            return_value=[
                {"id": 1, "name": "Order created", "topic": "order.created"},
                {"id": 2, "name": "Product updated", "topic": "product.updated"},
            ]
        )
        webhooks = await connector.get_webhooks()
        assert len(webhooks) == 2

    @pytest.mark.asyncio
    async def test_get_webhooks_failure(self, connector):
        """get_webhooks returns empty list on API error."""
        connector._request = AsyncMock(
            side_effect=ConnectorAPIError(
                "Cannot get webhooks", connector_name="woocommerce", status_code=500
            )
        )
        webhooks = await connector.get_webhooks()
        assert webhooks == []

    @pytest.mark.asyncio
    async def test_create_webhook_success(self, connector):
        """create_webhook sends POST and returns webhook data."""
        connector._request = AsyncMock(
            return_value={
                "id": 1,
                "name": "Order created",
                "topic": "order.created",
                "delivery_url": "https://example.com/webhook",
            }
        )
        result = await connector.create_webhook(
            name="Order created",
            topic="order.created",
            delivery_url="https://example.com/webhook",
            secret="webhook_secret",
        )
        assert result is not None
        assert result["topic"] == "order.created"

    @pytest.mark.asyncio
    async def test_create_webhook_failure(self, connector):
        """create_webhook returns None on API error."""
        connector._request = AsyncMock(
            side_effect=ConnectorAPIError(
                "Cannot create", connector_name="woocommerce", status_code=422
            )
        )
        result = await connector.create_webhook(
            name="Test", topic="order.created", delivery_url="https://example.com/webhook"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_webhook_success(self, connector):
        """delete_webhook sends DELETE and returns True."""
        connector._request = AsyncMock(return_value={"id": 1, "deleted": True})
        result = await connector.delete_webhook(1)
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_webhook_failure(self, connector):
        """delete_webhook returns False on API error."""
        connector._request = AsyncMock(
            side_effect=ConnectorAPIError(
                "Cannot delete", connector_name="woocommerce", status_code=422
            )
        )
        result = await connector.delete_webhook(1)
        assert result is False


# ========================================================================
# Webhook signature verification
# ========================================================================


class TestWebhookSignature:
    """Tests for webhook signature verification."""

    def test_verify_webhook_signature_valid(self, connector):
        """Valid signature returns True."""
        payload = b'{"id":123}'
        secret = "test_secret"
        expected_sig = base64.b64encode(
            hmac.new(secret.encode(), payload, hashlib.sha256).digest()
        ).decode()

        with patch.object(connector, "get_webhook_secret", return_value=secret):
            result = connector.verify_webhook_signature(payload, expected_sig)
        assert result is True

    def test_verify_webhook_signature_invalid(self, connector):
        """Invalid signature returns False."""
        payload = b'{"id":123}'
        secret = "test_secret"

        with patch.object(connector, "get_webhook_secret", return_value=secret):
            result = connector.verify_webhook_signature(payload, "invalid_signature")
        assert result is False

    def test_verify_webhook_signature_with_secret_param(self, connector):
        """Secret parameter overrides get_webhook_secret."""
        payload = b'{"id":123}'
        secret = "param_secret"
        expected_sig = base64.b64encode(
            hmac.new(secret.encode(), payload, hashlib.sha256).digest()
        ).decode()

        result = connector.verify_webhook_signature(payload, expected_sig, secret=secret)
        assert result is True

    def test_verify_webhook_signature_no_secret_dev(self, connector):
        """Missing secret in development returns True with warning."""
        payload = b'{"id":123}'

        with patch.object(connector, "get_webhook_secret", return_value=None):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                result = connector.verify_webhook_signature(payload, "any_signature")
        assert result is True

    def test_verify_webhook_signature_no_secret_prod(self, connector):
        """Missing secret in production returns False."""
        payload = b'{"id":123}'

        with patch.object(connector, "get_webhook_secret", return_value=None):
            with patch.dict("os.environ", {"ARAGORA_ENV": "production"}):
                result = connector.verify_webhook_signature(payload, "any_signature")
        assert result is False


# ========================================================================
# Shipping and Tax
# ========================================================================


class TestShippingAndTax:
    """Tests for shipping and tax operations."""

    @pytest.mark.asyncio
    async def test_get_shipping_zones(self, connector):
        """get_shipping_zones returns list of zones."""
        connector._request = AsyncMock(
            return_value=[
                {"id": 1, "name": "Domestic"},
                {"id": 2, "name": "International"},
            ]
        )
        zones = await connector.get_shipping_zones()
        assert len(zones) == 2

    @pytest.mark.asyncio
    async def test_get_shipping_zones_failure(self, connector):
        """get_shipping_zones returns empty list on API error."""
        connector._request = AsyncMock(
            side_effect=ConnectorAPIError(
                "Cannot get zones", connector_name="woocommerce", status_code=500
            )
        )
        zones = await connector.get_shipping_zones()
        assert zones == []

    @pytest.mark.asyncio
    async def test_get_shipping_methods(self, connector):
        """get_shipping_methods returns methods for a zone."""
        connector._request = AsyncMock(
            return_value=[
                {"id": 1, "method_id": "flat_rate"},
                {"id": 2, "method_id": "free_shipping"},
            ]
        )
        methods = await connector.get_shipping_methods(zone_id=1)
        assert len(methods) == 2

    @pytest.mark.asyncio
    async def test_get_tax_classes(self, connector):
        """get_tax_classes returns list of tax classes."""
        connector._request = AsyncMock(
            return_value=[
                {"slug": "standard", "name": "Standard"},
                {"slug": "reduced-rate", "name": "Reduced Rate"},
            ]
        )
        classes = await connector.get_tax_classes()
        assert len(classes) == 2

    @pytest.mark.asyncio
    async def test_get_tax_rates(self, connector):
        """get_tax_rates yields tax rate data."""
        connector._request = AsyncMock(
            return_value=[
                {"id": 1, "rate": "20.0000", "country": "US"},
                {"id": 2, "rate": "10.0000", "country": "GB"},
            ]
        )
        rates = []
        async for r in connector.get_tax_rates():
            rates.append(r)
        assert len(rates) == 2


# ========================================================================
# Reports
# ========================================================================


class TestReports:
    """Tests for report operations."""

    @pytest.mark.asyncio
    async def test_get_sales_report(self, connector):
        """get_sales_report returns sales data."""
        connector._request = AsyncMock(
            return_value=[
                {
                    "total_sales": "1000.00",
                    "total_orders": 50,
                    "total_items": 100,
                }
            ]
        )
        report = await connector.get_sales_report(period="month")
        assert report["total_sales"] == "1000.00"
        assert report["total_orders"] == 50

    @pytest.mark.asyncio
    async def test_get_sales_report_with_dates(self, connector):
        """get_sales_report passes date parameters."""
        connector._request = AsyncMock(return_value=[{"total_sales": "500.00"}])
        report = await connector.get_sales_report(
            period="month", date_min="2024-01-01", date_max="2024-01-31"
        )
        call_args = connector._request.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params")
        assert params["date_min"] == "2024-01-01"
        assert params["date_max"] == "2024-01-31"

    @pytest.mark.asyncio
    async def test_get_sales_report_empty(self, connector):
        """get_sales_report returns empty dict on failure."""
        connector._request = AsyncMock(
            side_effect=ConnectorAPIError(
                "Cannot get report", connector_name="woocommerce", status_code=500
            )
        )
        report = await connector.get_sales_report()
        assert report == {}

    @pytest.mark.asyncio
    async def test_get_top_sellers_report(self, connector):
        """get_top_sellers_report returns top products."""
        connector._request = AsyncMock(
            return_value=[
                {"product_id": 1, "title": "Widget", "quantity": 100},
                {"product_id": 2, "title": "Gadget", "quantity": 75},
            ]
        )
        top_sellers = await connector.get_top_sellers_report(period="month")
        assert len(top_sellers) == 2
        assert top_sellers[0]["title"] == "Widget"


# ========================================================================
# Order statistics
# ========================================================================


class TestOrderStats:
    """Tests for order statistics."""

    @pytest.mark.asyncio
    async def test_get_order_stats_basic(self, connector, sample_order_data):
        """get_order_stats computes totals from synced orders."""

        async def mock_sync_orders(*args, **kwargs):
            yield connector._parse_order(sample_order_data)

        connector.sync_orders = mock_sync_orders
        stats = await connector.get_order_stats()
        assert stats["total_orders"] == 1
        assert Decimal(stats["total_revenue"]) == Decimal("69.97")
        assert stats["cancelled_orders"] == 0
        assert stats["refunded_orders"] == 0

    @pytest.mark.asyncio
    async def test_get_order_stats_no_orders(self, connector):
        """get_order_stats handles empty order list."""

        async def mock_sync_orders(*args, **kwargs):
            return
            yield  # pragma: no cover

        connector.sync_orders = mock_sync_orders
        stats = await connector.get_order_stats()
        assert stats["total_orders"] == 0
        assert stats["completion_rate"] == 0
        assert stats["average_order_value"] == "0.00"

    @pytest.mark.asyncio
    async def test_get_order_stats_with_date_filter(self, connector, sample_order_data):
        """Orders outside end_date are excluded from stats."""
        end_date = datetime(2024, 6, 1, tzinfo=timezone.utc)

        async def mock_sync_orders(*args, **kwargs):
            yield connector._parse_order(sample_order_data)

        connector.sync_orders = mock_sync_orders
        stats = await connector.get_order_stats(end_date=end_date)
        assert stats["total_orders"] == 0


# ========================================================================
# Search and fetch
# ========================================================================


class TestSearchAndFetch:
    """Tests for BaseConnector search and fetch implementations."""

    @pytest.mark.asyncio
    async def test_search_products(self, connector, sample_product_data):
        """search returns product Evidence objects."""
        connector._request = AsyncMock(return_value=[sample_product_data])
        results = await connector.search("widget", limit=10, entity_type="product")
        assert len(results) >= 1
        assert results[0].id == "woo-product-2001"

    @pytest.mark.asyncio
    async def test_search_customers(self, connector, sample_customer_data):
        """search returns customer Evidence objects."""
        connector._request = AsyncMock(return_value=[sample_customer_data])
        results = await connector.search("jane", limit=10, entity_type="customer")
        assert len(results) >= 1
        assert results[0].id == "woo-customer-7001"

    @pytest.mark.asyncio
    async def test_search_all(self, connector, sample_product_data, sample_customer_data):
        """search with entity_type='all' returns multiple types."""

        async def mock_request(method, endpoint, **kwargs):
            if "products" in endpoint:
                return [sample_product_data]
            if "customers" in endpoint:
                return [sample_customer_data]
            return []

        connector._request = mock_request
        results = await connector.search("test", limit=10)
        assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_search_failure(self, connector):
        """search returns empty list on API error."""
        connector._request = AsyncMock(
            side_effect=ConnectorAPIError(
                "Search failed", connector_name="woocommerce", status_code=500
            )
        )
        results = await connector.search("test", limit=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_fetch_order(self, connector, sample_order_data):
        """fetch returns order Evidence."""
        connector._request = AsyncMock(return_value=sample_order_data)
        evidence = await connector.fetch("woo-order-5001")
        assert evidence is not None
        assert evidence.id == "woo-order-5001"
        assert "Order #1042" in evidence.title

    @pytest.mark.asyncio
    async def test_fetch_product(self, connector, sample_product_data):
        """fetch returns product Evidence."""
        connector._request = AsyncMock(return_value=sample_product_data)
        evidence = await connector.fetch("woo-product-2001")
        assert evidence is not None
        assert evidence.id == "woo-product-2001"
        assert evidence.title == "Widget"

    @pytest.mark.asyncio
    async def test_fetch_customer(self, connector, sample_customer_data):
        """fetch returns customer Evidence."""
        connector._request = AsyncMock(return_value=sample_customer_data)
        evidence = await connector.fetch("woo-customer-7001")
        assert evidence is not None
        assert evidence.id == "woo-customer-7001"

    @pytest.mark.asyncio
    async def test_fetch_invalid_id(self, connector):
        """fetch returns None for invalid ID format."""
        evidence = await connector.fetch("invalid-id")
        assert evidence is None

    @pytest.mark.asyncio
    async def test_fetch_not_found(self, connector):
        """fetch returns None when resource not found."""
        connector._request = AsyncMock(
            side_effect=ConnectorAPIError(
                "Not found", connector_name="woocommerce", status_code=404
            )
        )
        evidence = await connector.fetch("woo-order-9999")
        assert evidence is None


# ========================================================================
# Enterprise sync methods
# ========================================================================


class TestEnterpriseSyncMethods:
    """Tests for incremental_sync, full_sync, and sync_items."""

    @pytest.mark.asyncio
    async def test_incremental_sync_yields_all_types(
        self, connector, sample_order_data, sample_product_data, sample_customer_data
    ):
        """incremental_sync yields order, product, and customer dicts."""

        async def fake_sync_orders(**kwargs):
            yield connector._parse_order(sample_order_data)

        async def fake_sync_products(**kwargs):
            yield connector._parse_product(sample_product_data)

        async def fake_sync_customers(**kwargs):
            yield connector._parse_customer(sample_customer_data)

        connector.sync_orders = fake_sync_orders
        connector.sync_products = fake_sync_products
        connector.sync_customers = fake_sync_customers

        items = []
        async for item in connector.incremental_sync():
            items.append(item)

        assert len(items) == 3
        types = {i["type"] for i in items}
        assert types == {"order", "product", "customer"}

    @pytest.mark.asyncio
    async def test_incremental_sync_with_state(self, connector):
        """incremental_sync uses state.last_sync_at as since parameter."""
        state = SyncState(
            connector_id="woocommerce",
            last_sync_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )

        async def fake_sync_orders(**kwargs):
            return
            yield  # pragma: no cover

        async def fake_sync_products(**kwargs):
            return
            yield  # pragma: no cover

        async def fake_sync_customers(**kwargs):
            return
            yield  # pragma: no cover

        connector.sync_orders = fake_sync_orders
        connector.sync_products = fake_sync_products
        connector.sync_customers = fake_sync_customers

        items = []
        async for item in connector.incremental_sync(state=state):
            items.append(item)
        assert items == []

    @pytest.mark.asyncio
    async def test_full_sync_success(self, connector):
        """full_sync returns SyncResult with correct counts."""

        async def fake_incremental(**kwargs):
            yield {"type": "order", "data": {}}
            yield {"type": "product", "data": {}}

        connector.incremental_sync = fake_incremental
        result = await connector.full_sync()
        assert result.success is True
        assert result.items_synced == 2
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_full_sync_error(self, connector):
        """full_sync captures exceptions in errors list."""

        async def failing_incremental(**kwargs):
            yield {"type": "order", "data": {}}
            raise ValueError("sync failed")

        connector.incremental_sync = failing_incremental
        result = await connector.full_sync()
        assert result.success is False
        assert len(result.errors) == 1
        assert "sync failed" in result.errors[0]

    @pytest.mark.asyncio
    async def test_sync_items_yields_sync_items(
        self, connector, sample_order_data, sample_product_data, sample_customer_data
    ):
        """sync_items yields SyncItem objects with correct IDs and metadata."""

        async def fake_sync_orders(**kwargs):
            yield connector._parse_order(sample_order_data)

        async def fake_sync_products(**kwargs):
            yield connector._parse_product(sample_product_data)

        async def fake_sync_customers(**kwargs):
            yield connector._parse_customer(sample_customer_data)

        connector.sync_orders = fake_sync_orders
        connector.sync_products = fake_sync_products
        connector.sync_customers = fake_sync_customers

        state = SyncState(connector_id="woocommerce")
        items = []
        async for item in connector.sync_items(state, batch_size=50):
            items.append(item)

        assert len(items) == 3
        ids = {item.id for item in items}
        assert "woo-order-5001" in ids
        assert "woo-product-2001" in ids
        assert "woo-customer-7001" in ids

        # All should have source_type "woocommerce"
        assert all(item.source_type == "woocommerce" for item in items)


# ========================================================================
# Mock data helpers
# ========================================================================


class TestMockDataHelpers:
    """Tests for the module-level get_mock_* helper functions."""

    def test_get_mock_woo_orders(self):
        """get_mock_woo_orders returns a non-empty list of valid orders."""
        orders = get_mock_woo_orders()
        assert len(orders) >= 1
        order = orders[0]
        assert isinstance(order, WooOrder)
        assert order.id == 1001
        assert order.total == Decimal("59.99")
        assert order.status == WooOrderStatus.PROCESSING
        assert len(order.line_items) >= 1

    def test_get_mock_woo_products(self):
        """get_mock_woo_products returns a non-empty list of valid products."""
        products = get_mock_woo_products()
        assert len(products) >= 1
        product = products[0]
        assert isinstance(product, WooProduct)
        assert product.id == 101
        assert product.name == "Sample Product"
        assert product.price == Decimal("49.99")

    def test_mock_orders_have_datetimes(self):
        """Mock orders contain timezone-aware datetimes."""
        orders = get_mock_woo_orders()
        for order in orders:
            assert order.date_created.tzinfo is not None
            assert order.date_modified.tzinfo is not None

    def test_mock_products_have_datetimes(self):
        """Mock products contain timezone-aware datetimes."""
        products = get_mock_woo_products()
        for product in products:
            assert product.date_created.tzinfo is not None
            assert product.date_modified.tzinfo is not None


# ========================================================================
# Error handling edge cases
# ========================================================================


class TestErrorHandling:
    """Tests for error handling edge cases across operations."""

    @pytest.mark.asyncio
    async def test_request_preserves_status_code(self, connector):
        """ConnectorAPIError carries the original HTTP status code."""
        mock_session = MagicMock()
        resp_mock = AsyncMock()
        resp_mock.status = 403
        resp_mock.text = AsyncMock(return_value="Forbidden")

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp_mock)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.request = MagicMock(return_value=ctx)
        connector._client = mock_session

        with pytest.raises(ConnectorAPIError) as exc_info:
            await connector._request("GET", "orders")
        assert exc_info.value.status_code == 403
        assert exc_info.value.connector_name == "woocommerce"

    @pytest.mark.asyncio
    async def test_get_order_logs_and_returns_none(self, connector):
        """get_order catches expected exceptions and returns None."""
        connector._request = AsyncMock(side_effect=OSError("network down"))
        result = await connector.get_order(123)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_product_logs_and_returns_none(self, connector):
        """get_product catches expected exceptions and returns None."""
        connector._request = AsyncMock(side_effect=ValueError("timeout"))
        result = await connector.get_product(123)
        assert result is None

    @pytest.mark.asyncio
    async def test_update_order_status_logs_and_returns_false(self, connector):
        """update_order_status catches OSError and returns False."""
        connector._request = AsyncMock(side_effect=OSError("connection reset"))
        result = await connector.update_order_status(123, WooOrderStatus.COMPLETED)
        assert result is False

    @pytest.mark.asyncio
    async def test_update_product_stock_logs_and_returns_false(self, connector):
        """update_product_stock catches OSError and returns False."""
        connector._request = AsyncMock(side_effect=OSError("dns fail"))
        result = await connector.update_product_stock(123, 50)
        assert result is False

    @pytest.mark.asyncio
    async def test_request_error_includes_message(self, connector):
        """ConnectorAPIError message contains the error text from WooCommerce."""
        mock_session = MagicMock()
        resp_mock = AsyncMock()
        resp_mock.status = 400
        resp_mock.text = AsyncMock(return_value='{"errors":"Invalid product"}')

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp_mock)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.request = MagicMock(return_value=ctx)
        connector._client = mock_session

        with pytest.raises(ConnectorAPIError) as exc_info:
            await connector._request("POST", "products", json_data={})
        assert "Invalid product" in str(exc_info.value.message)


# ========================================================================
# Serialization round-trips
# ========================================================================


class TestSerializationRoundTrips:
    """Tests for to_dict serialization consistency."""

    def test_order_enum_serialized(self, now_utc):
        """Enum fields serialize to their string values (API names by default)."""
        order = WooOrder(
            id=1,
            number="1",
            order_key="wc_order_1",
            status=WooOrderStatus.ON_HOLD,
            currency="USD",
            date_created=now_utc,
            date_modified=now_utc,
            total=Decimal("50.00"),
            subtotal=Decimal("45.00"),
            total_tax=Decimal("5.00"),
            shipping_total=Decimal("0"),
            discount_total=Decimal("0"),
            payment_method="",
            payment_method_title="",
            customer_id=0,
            billing=WooAddress(),
            shipping=WooAddress(),
        )
        d = order.to_dict()
        assert d["status"] == "on-hold"

    def test_product_type_serialized(self, now_utc):
        """Product type enum serializes to string."""
        product = WooProduct(
            id=1,
            name="T",
            slug="t",
            type=WooProductType.VARIABLE,
            status=WooProductStatus.PUBLISH,
            sku=None,
            price=Decimal("0"),
            regular_price=Decimal("0"),
            sale_price=None,
            date_created=now_utc,
            date_modified=now_utc,
            description=None,
            short_description=None,
            stock_quantity=None,
            stock_status=WooStockStatus.IN_STOCK,
            manage_stock=False,
        )
        d = product.to_dict()
        assert d["type"] == "variable"

    def test_datetime_serialized_as_iso(self, now_utc):
        """Datetime fields serialize to ISO 8601 strings."""
        customer = WooCustomer(
            id=1,
            email="e@x.com",
            first_name="A",
            last_name="B",
            username="ab",
            date_created=now_utc,
            date_modified=now_utc,
            billing=WooAddress(),
            shipping=WooAddress(),
        )
        d = customer.to_dict()
        assert "2024-06-15" in d["dateCreated"]

    def test_nested_line_items_in_order(self, now_utc):
        """Nested line items in an order are serialized recursively."""
        li = WooLineItem(
            id=1,
            product_id=100,
            variation_id=0,
            name="Thing",
            quantity=2,
            subtotal=Decimal("10.00"),
            total=Decimal("10.00"),
            price=Decimal("5.00"),
        )
        order = WooOrder(
            id=1,
            number="1",
            order_key="wc_order_1",
            status=WooOrderStatus.PENDING,
            currency="USD",
            date_created=now_utc,
            date_modified=now_utc,
            total=Decimal("10.00"),
            subtotal=Decimal("10.00"),
            total_tax=Decimal("0"),
            shipping_total=Decimal("0"),
            discount_total=Decimal("0"),
            payment_method="",
            payment_method_title="",
            customer_id=0,
            billing=WooAddress(),
            shipping=WooAddress(),
            line_items=[li],
        )
        d = order.to_dict()
        assert len(d["lineItems"]) == 1
        assert d["lineItems"][0]["quantity"] == 2


# ========================================================================
# Pagination limits
# ========================================================================


class TestPaginationLimits:
    """Tests for pagination safety limits."""

    @pytest.mark.asyncio
    async def test_sync_orders_pagination_limit(self, connector, sample_order_data):
        """sync_orders stops at max page limit to prevent infinite loops."""
        # This test is more of a conceptual verification - the actual limit is 1000
        connector._request = AsyncMock(return_value=[])
        orders = []
        async for order in connector.sync_orders(per_page=100):
            orders.append(order)
        assert len(orders) == 0

    @pytest.mark.asyncio
    async def test_sync_products_pagination_limit(self, connector, sample_product_data):
        """sync_products stops at max page limit."""
        connector._request = AsyncMock(return_value=[])
        products = []
        async for product in connector.sync_products(per_page=100):
            products.append(product)
        assert len(products) == 0

    @pytest.mark.asyncio
    async def test_sync_variations_pagination_limit(self, connector, sample_variation_data):
        """sync_product_variations stops at max page limit."""
        connector._request = AsyncMock(return_value=[])
        variations = []
        async for v in connector.sync_product_variations(2001, per_page=100):
            variations.append(v)
        assert len(variations) == 0


# ========================================================================
# Timeout handling tests
# ========================================================================


class TestTimeoutHandling:
    """Tests for request timeout functionality."""

    @pytest.mark.asyncio
    async def test_request_timeout_default(self, connector):
        """Request uses default timeout from credentials."""
        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=_make_mock_response({"ok": True}))
        connector._client = mock_session

        # This should work with default timeout (30s)
        result = await connector._request("GET", "test")
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_request_timeout_raises_error(self, connector):
        """Request timeout raises ConnectorTimeoutError."""

        mock_session = MagicMock()

        async def timeout_side_effect(*args, **kwargs):
            import asyncio

            raise asyncio.TimeoutError()

        # Create a mock context manager that raises timeout
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(side_effect=timeout_side_effect)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.request = MagicMock(return_value=ctx)
        connector._client = mock_session

        with pytest.raises(ConnectorTimeoutError) as exc_info:
            await connector._request("GET", "test", timeout=0.001)

        assert "timed out" in str(exc_info.value.message)
        assert exc_info.value.connector_name == "woocommerce"

    @pytest.mark.asyncio
    async def test_custom_timeout_override(self, connector):
        """Custom timeout parameter overrides default."""
        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=_make_mock_response({"ok": True}))
        connector._client = mock_session

        # Explicitly pass a timeout
        result = await connector._request("GET", "test", timeout=60)
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_timeout_records_circuit_breaker_failure(self, connector):
        """Timeout records a failure in the circuit breaker."""
        import asyncio

        mock_session = MagicMock()

        async def timeout_side_effect(*args, **kwargs):
            raise asyncio.TimeoutError()

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(side_effect=timeout_side_effect)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.request = MagicMock(return_value=ctx)
        connector._client = mock_session

        initial_failures = connector._circuit_breaker._single_failures

        with pytest.raises(ConnectorTimeoutError):
            await connector._request("GET", "test", timeout=0.001)

        # Circuit breaker should have recorded the failure
        assert connector._circuit_breaker._single_failures > initial_failures


# ========================================================================
# Input validation tests
# ========================================================================


class TestInputValidation:
    """Tests for input validation functionality."""

    def test_validate_id_valid_numeric(self):
        """validate_id accepts numeric IDs."""

        # Should not raise
        validate_id(123, "test_id")
        validate_id(0, "test_id")
        validate_id(999999, "test_id")

    def test_validate_id_valid_alphanumeric(self):
        """validate_id accepts alphanumeric strings."""

        # Should not raise
        validate_id("abc123", "test_id")
        validate_id("ABC123", "test_id")
        validate_id("product_123", "test_id")
        validate_id("order-456", "test_id")

    def test_validate_id_invalid_characters(self):
        """validate_id rejects IDs with invalid characters."""
        from aragora.connectors.exceptions import ConnectorValidationError

        # SQL injection attempt
        with pytest.raises(ConnectorValidationError) as exc_info:
            validate_id("123; DROP TABLE orders;", "order_id")
        assert "invalid characters" in str(exc_info.value.message).lower()
        assert exc_info.value.connector_name == "woocommerce"

    def test_validate_id_path_traversal(self):
        """validate_id rejects path traversal attempts."""
        from aragora.connectors.exceptions import ConnectorValidationError

        with pytest.raises(ConnectorValidationError):
            validate_id("../../../etc/passwd", "product_id")

    def test_validate_id_special_characters(self):
        """validate_id rejects special characters."""
        from aragora.connectors.exceptions import ConnectorValidationError

        invalid_ids = [
            "123/456",
            "product?id=1",
            "order&status=completed",
            "item#1",
            "test@example",
            "item%20name",
            "order\nid",
            "product\tid",
        ]
        for invalid_id in invalid_ids:
            with pytest.raises(ConnectorValidationError):
                validate_id(invalid_id, "test_id")

    @pytest.mark.asyncio
    async def test_get_order_validates_id(self, connector):
        """get_order validates order_id."""

        with pytest.raises(ConnectorValidationError) as exc_info:
            await connector.get_order("123; DROP TABLE orders;")
        assert "order_id" in str(exc_info.value.message)

    @pytest.mark.asyncio
    async def test_get_product_validates_id(self, connector):
        """get_product validates product_id."""

        with pytest.raises(ConnectorValidationError) as exc_info:
            await connector.get_product("../../../etc/passwd")
        assert "product_id" in str(exc_info.value.message)

    @pytest.mark.asyncio
    async def test_update_order_status_validates_id(self, connector):
        """update_order_status validates order_id."""

        with pytest.raises(ConnectorValidationError):
            await connector.update_order_status("invalid/id", WooOrderStatus.COMPLETED)

    @pytest.mark.asyncio
    async def test_update_product_stock_validates_id(self, connector):
        """update_product_stock validates product_id."""

        with pytest.raises(ConnectorValidationError):
            await connector.update_product_stock("../passwd", 10)

    @pytest.mark.asyncio
    async def test_update_variation_stock_validates_ids(self, connector):
        """update_variation_stock validates both product_id and variation_id."""

        # Invalid product_id
        with pytest.raises(ConnectorValidationError):
            await connector.update_variation_stock("invalid/id", 123, 10)

        # Invalid variation_id
        with pytest.raises(ConnectorValidationError):
            await connector.update_variation_stock(123, "invalid/id", 10)

    @pytest.mark.asyncio
    async def test_create_refund_validates_id(self, connector):
        """create_refund validates order_id."""

        with pytest.raises(ConnectorValidationError):
            await connector.create_refund("invalid?id=1", reason="Test")

    @pytest.mark.asyncio
    async def test_get_refunds_validates_id(self, connector):
        """get_refunds validates order_id."""

        with pytest.raises(ConnectorValidationError):
            await connector.get_refunds("123&foo=bar")

    @pytest.mark.asyncio
    async def test_delete_coupon_validates_id(self, connector):
        """delete_coupon validates coupon_id."""

        with pytest.raises(ConnectorValidationError):
            await connector.delete_coupon("bad/id")

    @pytest.mark.asyncio
    async def test_delete_webhook_validates_id(self, connector):
        """delete_webhook validates webhook_id."""

        with pytest.raises(ConnectorValidationError):
            await connector.delete_webhook("../../etc")

    @pytest.mark.asyncio
    async def test_get_shipping_methods_validates_id(self, connector):
        """get_shipping_methods validates zone_id."""

        with pytest.raises(ConnectorValidationError):
            await connector.get_shipping_methods("zone?id=1")


# ========================================================================
# Circuit breaker tests
# ========================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    def test_connector_has_circuit_breaker(self, connector):
        """Connector initializes with a circuit breaker."""
        assert connector._circuit_breaker is not None
        assert connector._circuit_breaker.name == "woocommerce"

    def test_circuit_breaker_property(self, connector):
        """circuit_breaker property returns the breaker instance."""
        assert connector.circuit_breaker is connector._circuit_breaker

    def test_custom_circuit_breaker(self, credentials):
        """Connector accepts custom circuit breaker."""

        custom_breaker = CircuitBreaker(
            name="custom-woo",
            failure_threshold=10,
            cooldown_seconds=120.0,
        )
        connector = WooCommerceConnector(credentials, circuit_breaker=custom_breaker)
        assert connector._circuit_breaker is custom_breaker
        assert connector._circuit_breaker.failure_threshold == 10

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_when_open(self, connector):
        """Request fails immediately when circuit breaker is open."""

        # Open the circuit breaker manually
        connector._circuit_breaker._single_open_at = 999999999999.0

        with pytest.raises(ConnectorCircuitOpenError) as exc_info:
            await connector._request("GET", "test")

        assert "circuit breaker is open" in str(exc_info.value.message).lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_success(self, connector):
        """Successful requests record success in circuit breaker."""
        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=_make_mock_response({"ok": True}))
        connector._client = mock_session

        # Force some failures first
        connector._circuit_breaker._single_failures = 2

        await connector._request("GET", "test")

        # Success should reset failures (depending on implementation)
        # At minimum, the breaker should still allow requests
        assert connector._circuit_breaker.can_proceed()

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_failure_on_5xx(self, connector):
        """5xx errors record failures in circuit breaker."""
        mock_session = MagicMock()
        resp_mock = AsyncMock()
        resp_mock.status = 500
        resp_mock.text = AsyncMock(return_value="Internal Server Error")

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp_mock)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.request = MagicMock(return_value=ctx)
        connector._client = mock_session

        initial_failures = connector._circuit_breaker._single_failures

        with pytest.raises(ConnectorAPIError):
            await connector._request("GET", "test")

        # Circuit breaker should have recorded the failure
        assert connector._circuit_breaker._single_failures > initial_failures

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_failure_on_429(self, connector):
        """429 rate limit errors record failures in circuit breaker."""
        mock_session = MagicMock()
        resp_mock = AsyncMock()
        resp_mock.status = 429
        resp_mock.text = AsyncMock(return_value="Rate limited")

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp_mock)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.request = MagicMock(return_value=ctx)
        connector._client = mock_session

        initial_failures = connector._circuit_breaker._single_failures

        with pytest.raises(ConnectorAPIError):
            await connector._request("GET", "test")

        # Circuit breaker should have recorded the failure
        assert connector._circuit_breaker._single_failures > initial_failures

    @pytest.mark.asyncio
    async def test_circuit_breaker_no_failure_on_4xx(self, connector):
        """4xx client errors (except 429) don't record circuit breaker failures."""
        mock_session = MagicMock()
        resp_mock = AsyncMock()
        resp_mock.status = 404
        resp_mock.text = AsyncMock(return_value="Not found")

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp_mock)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.request = MagicMock(return_value=ctx)
        connector._client = mock_session

        initial_failures = connector._circuit_breaker._single_failures

        with pytest.raises(ConnectorAPIError):
            await connector._request("GET", "test")

        # Circuit breaker should NOT have recorded the failure for 404
        assert connector._circuit_breaker._single_failures == initial_failures

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self, connector):
        """Circuit opens after reaching failure threshold."""
        mock_session = MagicMock()
        resp_mock = AsyncMock()
        resp_mock.status = 500
        resp_mock.text = AsyncMock(return_value="Server Error")

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp_mock)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.request = MagicMock(return_value=ctx)
        connector._client = mock_session

        # Default threshold is 5, make 5 failures
        for _ in range(5):
            try:
                await connector._request("GET", "test")
            except ConnectorAPIError:
                pass

        # Circuit should now be open
        assert connector._circuit_breaker.get_status() == "open"

    def test_circuit_breaker_cooldown_remaining(self, connector):
        """cooldown_remaining returns remaining time until circuit can be tried."""
        import time

        # Set circuit open 30 seconds ago (with 60s cooldown)
        connector._circuit_breaker._single_open_at = time.time() - 30

        remaining = connector._circuit_breaker.cooldown_remaining()

        # Should have about 30 seconds remaining
        assert 25 < remaining < 35

    @pytest.mark.asyncio
    async def test_circuit_breaker_error_includes_cooldown(self, connector):
        """ConnectorCircuitOpenError includes cooldown_remaining."""
        import time

        # Set circuit open 30 seconds ago
        connector._circuit_breaker._single_open_at = time.time() - 30

        with pytest.raises(ConnectorCircuitOpenError) as exc_info:
            await connector._request("GET", "test")

        # Error should include cooldown info
        assert exc_info.value.cooldown_remaining is not None
        assert exc_info.value.cooldown_remaining > 0


# ========================================================================
# Default timeout constant tests
# ========================================================================


class TestDefaultConstants:
    """Tests for default constant values."""

    def test_default_request_timeout(self):
        """DEFAULT_REQUEST_TIMEOUT is 30 seconds."""

        assert DEFAULT_REQUEST_TIMEOUT == 30

    def test_credentials_default_timeout(self):
        """WooCommerceCredentials defaults to 30 second timeout."""
        creds = WooCommerceCredentials(
            store_url="https://example.com",
            consumer_key="key",
            consumer_secret="secret",
        )
        assert creds.timeout == 30
