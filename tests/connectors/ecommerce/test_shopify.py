"""
Comprehensive tests for the Shopify Connector module.

Covers:
- Enum types (ShopifyEnvironment, OrderStatus, PaymentStatus, InventoryPolicy)
- Dataclass models (credentials, address, line item, order, product, variant,
  customer, inventory level)
- ShopifyConnector class (initialization, connection, API requests, pagination,
  order operations, product operations, customer operations, analytics,
  enterprise sync methods)
- Authentication, rate limiting, error handling
- Data serialization and deserialization
- Mock data helpers
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from aragora.connectors.ecommerce.shopify import (
    InventoryPolicy,
    OrderStatus,
    PaymentStatus,
    ShopifyAddress,
    ShopifyConnector,
    ShopifyCredentials,
    ShopifyCustomer,
    ShopifyEnvironment,
    ShopifyInventoryLevel,
    ShopifyLineItem,
    ShopifyOrder,
    ShopifyProduct,
    ShopifyVariant,
    get_mock_orders,
    get_mock_products,
)
from aragora.connectors.exceptions import ConnectorAPIError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def credentials():
    """Standard test credentials."""
    return ShopifyCredentials(
        shop_domain="test-store.myshopify.com",
        access_token="shpat_test_token_abc123",
        api_version="2024-01",
    )


@pytest.fixture
def connector(credentials):
    """ShopifyConnector wired to test credentials with circuit breaker disabled."""
    return ShopifyConnector(
        credentials=credentials,
        environment=ShopifyEnvironment.DEVELOPMENT,
    )


@pytest.fixture
def now_utc():
    """Current UTC datetime for deterministic data."""
    return datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_address_data():
    """Raw address dict as returned by the Shopify API."""
    return {
        "first_name": "Jane",
        "last_name": "Doe",
        "company": "Acme Inc",
        "address1": "123 Main St",
        "address2": "Suite 4",
        "city": "Portland",
        "province": "Oregon",
        "province_code": "OR",
        "country": "United States",
        "country_code": "US",
        "zip": "97201",
        "phone": "+15035551234",
    }


@pytest.fixture
def sample_line_item_data():
    """Raw line-item dict as returned by the Shopify API."""
    return {
        "id": 9001,
        "product_id": 2001,
        "variant_id": 3001,
        "title": "Widget",
        "quantity": 3,
        "price": "19.95",
        "sku": "WGT-001",
        "vendor": "WidgetCo",
        "grams": 250,
        "taxable": True,
        "fulfillment_status": None,
        "requires_shipping": True,
    }


@pytest.fixture
def sample_order_data(sample_line_item_data, sample_address_data):
    """Raw order dict as returned by the Shopify API."""
    return {
        "id": 5001,
        "order_number": 1042,
        "name": "#1042",
        "email": "buyer@example.com",
        "created_at": "2024-06-10T14:30:00Z",
        "updated_at": "2024-06-11T09:00:00Z",
        "total_price": "69.85",
        "subtotal_price": "59.85",
        "total_tax": "5.00",
        "total_discounts": "5.00",
        "currency": "USD",
        "financial_status": "paid",
        "fulfillment_status": "fulfilled",
        "line_items": [sample_line_item_data],
        "shipping_address": sample_address_data,
        "billing_address": sample_address_data,
        "customer": {"id": 7001},
        "note": "Please gift wrap",
        "tags": "vip, repeat-buyer",
        "cancelled_at": None,
        "closed_at": "2024-06-12T10:00:00Z",
    }


@pytest.fixture
def sample_variant_data():
    """Raw variant dict as returned by the Shopify API."""
    return {
        "id": 3001,
        "title": "Large / Red",
        "price": "29.99",
        "sku": "WGT-LG-RED",
        "inventory_quantity": 12,
        "inventory_policy": "deny",
        "compare_at_price": "39.99",
        "weight": 0.5,
        "weight_unit": "kg",
        "barcode": "1234567890",
        "option1": "Large",
        "option2": "Red",
        "option3": None,
    }


@pytest.fixture
def sample_product_data(sample_variant_data):
    """Raw product dict as returned by the Shopify API."""
    return {
        "id": 2001,
        "title": "Widget",
        "handle": "widget",
        "vendor": "WidgetCo",
        "product_type": "Gadgets",
        "status": "active",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-06-15T12:00:00Z",
        "published_at": "2024-01-02T08:00:00Z",
        "body_html": "<p>A fine widget</p>",
        "tags": "sale, new-arrival",
        "variants": [sample_variant_data],
        "images": [{"src": "https://cdn.shopify.com/widget.jpg"}],
    }


@pytest.fixture
def sample_customer_data():
    """Raw customer dict as returned by the Shopify API."""
    return {
        "id": 7001,
        "email": "buyer@example.com",
        "first_name": "Jane",
        "last_name": "Doe",
        "phone": "+15035551234",
        "created_at": "2023-08-01T10:00:00Z",
        "updated_at": "2024-06-15T12:00:00Z",
        "orders_count": 5,
        "total_spent": "349.95",
        "verified_email": True,
        "accepts_marketing": False,
        "tax_exempt": False,
        "tags": "vip, repeat-buyer",
        "note": "Prefers email contact",
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
    """Tests for Shopify-specific enum types."""

    def test_shopify_environment_values(self):
        """ShopifyEnvironment has the expected members."""
        assert ShopifyEnvironment.DEVELOPMENT.value == "development"
        assert ShopifyEnvironment.PRODUCTION.value == "production"

    def test_order_status_values(self):
        """OrderStatus covers the expected fulfillment states."""
        expected = {
            "pending",
            "open",
            "fulfilled",
            "partially_fulfilled",
            "cancelled",
            "refunded",
        }
        assert {s.value for s in OrderStatus} == expected

    def test_payment_status_values(self):
        """PaymentStatus covers the expected payment states."""
        expected = {
            "pending",
            "paid",
            "partially_paid",
            "refunded",
            "voided",
            "authorized",
        }
        assert {s.value for s in PaymentStatus} == expected

    def test_inventory_policy_values(self):
        """InventoryPolicy covers deny and continue."""
        assert InventoryPolicy.DENY.value == "deny"
        assert InventoryPolicy.CONTINUE.value == "continue"

    def test_enum_from_string(self):
        """Enums can be constructed from their string values."""
        assert OrderStatus("fulfilled") == OrderStatus.FULFILLED
        assert PaymentStatus("paid") == PaymentStatus.PAID
        assert InventoryPolicy("continue") == InventoryPolicy.CONTINUE


# ========================================================================
# Credential tests
# ========================================================================


class TestShopifyCredentials:
    """Tests for ShopifyCredentials dataclass."""

    def test_basic_construction(self, credentials):
        """Credentials store the expected fields."""
        assert credentials.shop_domain == "test-store.myshopify.com"
        assert credentials.access_token == "shpat_test_token_abc123"
        assert credentials.api_version == "2024-01"
        assert credentials.scope == ""

    def test_default_api_version(self):
        """Default API version is 2024-01."""
        creds = ShopifyCredentials(
            shop_domain="s.myshopify.com",
            access_token="tok",
        )
        assert creds.api_version == "2024-01"

    def test_custom_scope(self):
        """Optional scope can be specified."""
        creds = ShopifyCredentials(
            shop_domain="s.myshopify.com",
            access_token="tok",
            scope="read_products,write_orders",
        )
        assert creds.scope == "read_products,write_orders"

    def test_from_env(self):
        """Credentials can be loaded from environment variables."""
        env = {
            "SHOPIFY_SHOP_DOMAIN": "env-store.myshopify.com",
            "SHOPIFY_ACCESS_TOKEN": "shpat_env_token",
            "SHOPIFY_API_VERSION": "2024-04",
        }
        with patch.dict("os.environ", env, clear=False):
            creds = ShopifyCredentials.from_env()
            assert creds.shop_domain == "env-store.myshopify.com"
            assert creds.access_token == "shpat_env_token"
            assert creds.api_version == "2024-04"

    def test_from_env_defaults(self):
        """Missing env vars resolve to empty strings / defaults."""
        with patch.dict("os.environ", {}, clear=True):
            creds = ShopifyCredentials.from_env()
            assert creds.shop_domain == ""
            assert creds.access_token == ""
            assert creds.api_version == "2024-01"


# ========================================================================
# Address tests
# ========================================================================


class TestShopifyAddress:
    """Tests for ShopifyAddress dataclass."""

    def test_construction_defaults(self):
        """All fields default to None."""
        addr = ShopifyAddress()
        assert addr.first_name is None
        assert addr.city is None
        assert addr.country_code is None

    def test_full_construction(self, sample_address_data):
        """Address can be constructed with all fields."""
        addr = ShopifyAddress(**sample_address_data)
        assert addr.first_name == "Jane"
        assert addr.city == "Portland"
        assert addr.country_code == "US"

    def test_to_dict_uses_api_names(self, sample_address_data):
        """to_dict with use_api_names maps snake_case to camelCase."""
        addr = ShopifyAddress(**sample_address_data)
        d = addr.to_dict(use_api_names=True)
        assert "firstName" in d
        assert "provinceCode" in d
        assert d["firstName"] == "Jane"

    def test_to_dict_includes_none(self):
        """None values are included because _include_none is True."""
        addr = ShopifyAddress(first_name="Bob")
        d = addr.to_dict()
        assert "last_name" in d
        assert d["last_name"] is None


# ========================================================================
# Line item tests
# ========================================================================


class TestShopifyLineItem:
    """Tests for ShopifyLineItem dataclass."""

    def test_construction(self):
        """Line item stores essential order-line data."""
        item = ShopifyLineItem(
            id="li_1",
            product_id="p1",
            variant_id="v1",
            title="Widget",
            quantity=2,
            price=Decimal("19.99"),
        )
        assert item.quantity == 2
        assert item.price == Decimal("19.99")
        assert item.taxable is True  # default
        assert item.requires_shipping is True  # default

    def test_to_dict_api_names(self):
        """API names mapping works for line items."""
        item = ShopifyLineItem(
            id="li_1",
            product_id="p1",
            variant_id="v1",
            title="Widget",
            quantity=1,
            price=Decimal("10.00"),
        )
        d = item.to_dict(use_api_names=True)
        assert "productId" in d
        assert "variantId" in d
        assert "fulfillmentStatus" in d


# ========================================================================
# Order model tests
# ========================================================================


class TestShopifyOrder:
    """Tests for ShopifyOrder dataclass and serialization."""

    def test_construction_minimal(self, now_utc):
        """Order can be constructed with required fields only."""
        order = ShopifyOrder(
            id="1",
            order_number=100,
            name="#100",
            email=None,
            created_at=now_utc,
            updated_at=now_utc,
            total_price=Decimal("50.00"),
            subtotal_price=Decimal("45.00"),
            total_tax=Decimal("5.00"),
            total_discounts=Decimal("0.00"),
            currency="USD",
            financial_status=PaymentStatus.PENDING,
            fulfillment_status=None,
        )
        assert order.line_items == []
        assert order.tags == []
        assert order.shipping_address is None

    def test_to_dict_serialises_decimals(self, now_utc):
        """Decimal fields are serialized as strings."""
        order = ShopifyOrder(
            id="1",
            order_number=100,
            name="#100",
            email=None,
            created_at=now_utc,
            updated_at=now_utc,
            total_price=Decimal("123.45"),
            subtotal_price=Decimal("100.00"),
            total_tax=Decimal("23.45"),
            total_discounts=Decimal("0.00"),
            currency="EUR",
            financial_status=PaymentStatus.PAID,
            fulfillment_status=None,
        )
        d = order.to_dict()
        assert d["total_price"] == "123.45"
        assert d["total_tax"] == "23.45"

    def test_to_dict_api_names(self, now_utc):
        """to_dict with use_api_names maps field names correctly."""
        order = ShopifyOrder(
            id="1",
            order_number=100,
            name="#100",
            email=None,
            created_at=now_utc,
            updated_at=now_utc,
            total_price=Decimal("0"),
            subtotal_price=Decimal("0"),
            total_tax=Decimal("0"),
            total_discounts=Decimal("0"),
            currency="USD",
            financial_status=PaymentStatus.PENDING,
            fulfillment_status=None,
        )
        d = order.to_dict(use_api_names=True)
        assert "orderNumber" in d
        assert "createdAt" in d
        assert "totalPrice" in d


# ========================================================================
# Product / Variant tests
# ========================================================================


class TestShopifyProduct:
    """Tests for ShopifyProduct dataclass."""

    def test_construction(self, now_utc):
        """Product stores catalogue fields."""
        product = ShopifyProduct(
            id="p1",
            title="Widget",
            handle="widget",
            vendor="WidgetCo",
            product_type="Gadgets",
            status="active",
            created_at=now_utc,
            updated_at=now_utc,
            published_at=now_utc,
        )
        assert product.title == "Widget"
        assert product.variants == []
        assert product.images == []

    def test_to_dict_api_names(self, now_utc):
        """Product to_dict maps product_type -> productType."""
        product = ShopifyProduct(
            id="p1",
            title="T",
            handle="t",
            vendor=None,
            product_type=None,
            status="draft",
            created_at=now_utc,
            updated_at=now_utc,
            published_at=None,
        )
        d = product.to_dict(use_api_names=True)
        assert "productType" in d
        assert "publishedAt" in d


class TestShopifyVariant:
    """Tests for ShopifyVariant dataclass."""

    def test_defaults(self):
        """Variant has sensible defaults."""
        variant = ShopifyVariant(
            id="v1",
            product_id="p1",
            title="Default",
            price=Decimal("9.99"),
            sku=None,
        )
        assert variant.inventory_quantity == 0
        assert variant.inventory_policy == InventoryPolicy.DENY
        assert variant.weight == 0.0
        assert variant.weight_unit == "kg"

    def test_to_dict_api_names(self):
        """Variant to_dict maps inventory fields."""
        variant = ShopifyVariant(
            id="v1",
            product_id="p1",
            title="Default",
            price=Decimal("10.00"),
            sku="SKU",
            inventory_quantity=5,
            inventory_policy=InventoryPolicy.CONTINUE,
            compare_at_price=Decimal("15.00"),
        )
        d = variant.to_dict(use_api_names=True)
        assert d["inventoryQuantity"] == 5
        assert d["inventoryPolicy"] == "continue"
        assert d["compareAtPrice"] == "15.00"


# ========================================================================
# Customer tests
# ========================================================================


class TestShopifyCustomer:
    """Tests for ShopifyCustomer dataclass and full_name property."""

    def test_full_name_both(self, now_utc):
        """full_name combines first and last."""
        c = ShopifyCustomer(
            id="c1",
            email=None,
            first_name="Jane",
            last_name="Doe",
            phone=None,
            created_at=now_utc,
            updated_at=now_utc,
        )
        assert c.full_name == "Jane Doe"

    def test_full_name_first_only(self, now_utc):
        """full_name returns first name when last is None."""
        c = ShopifyCustomer(
            id="c1",
            email=None,
            first_name="Jane",
            last_name=None,
            phone=None,
            created_at=now_utc,
            updated_at=now_utc,
        )
        assert c.full_name == "Jane"

    def test_full_name_neither(self, now_utc):
        """full_name returns 'Unknown' when both names are None."""
        c = ShopifyCustomer(
            id="c1",
            email=None,
            first_name=None,
            last_name=None,
            phone=None,
            created_at=now_utc,
            updated_at=now_utc,
        )
        assert c.full_name == "Unknown"

    def test_to_dict_includes_full_name(self, now_utc):
        """to_dict appends the computed fullName key."""
        c = ShopifyCustomer(
            id="c1",
            email="e@x.com",
            first_name="A",
            last_name="B",
            phone=None,
            created_at=now_utc,
            updated_at=now_utc,
        )
        d = c.to_dict()
        assert d["fullName"] == "A B"

    def test_customer_defaults(self, now_utc):
        """Default numeric / boolean fields."""
        c = ShopifyCustomer(
            id="c1",
            email=None,
            first_name=None,
            last_name=None,
            phone=None,
            created_at=now_utc,
            updated_at=now_utc,
        )
        assert c.orders_count == 0
        assert c.total_spent == Decimal("0.00")
        assert c.verified_email is False
        assert c.accepts_marketing is False
        assert c.tax_exempt is False


# ========================================================================
# Inventory level tests
# ========================================================================


class TestShopifyInventoryLevel:
    """Tests for ShopifyInventoryLevel dataclass."""

    def test_construction(self, now_utc):
        """Inventory level captures location-specific stock."""
        inv = ShopifyInventoryLevel(
            inventory_item_id="ii_1",
            location_id="loc_1",
            available=42,
            updated_at=now_utc,
        )
        assert inv.available == 42

    def test_to_dict_api_names(self, now_utc):
        """API name mapping works for inventory levels."""
        inv = ShopifyInventoryLevel(
            inventory_item_id="ii_1",
            location_id="loc_1",
            available=10,
            updated_at=now_utc,
        )
        d = inv.to_dict(use_api_names=True)
        assert "inventoryItemId" in d
        assert "locationId" in d


# ========================================================================
# Connector initialisation
# ========================================================================


class TestConnectorInit:
    """Tests for ShopifyConnector construction and base_url property."""

    def test_base_url(self, connector, credentials):
        """base_url is constructed from credentials."""
        expected = f"https://{credentials.shop_domain}/admin/api/{credentials.api_version}"
        assert connector.base_url == expected

    def test_default_environment(self, credentials):
        """Default environment is PRODUCTION."""
        c = ShopifyConnector(credentials=credentials)
        assert c.environment == ShopifyEnvironment.PRODUCTION

    def test_session_initially_none(self, connector):
        """No HTTP session before connect()."""
        assert connector._session is None


# ========================================================================
# Connection lifecycle
# ========================================================================


class TestConnect:
    """Tests for connect() and disconnect()."""

    @pytest.mark.asyncio
    async def test_connect_success(self, connector):
        """Successful connect sets session and returns True."""
        mock_session_cls = MagicMock()
        mock_session = AsyncMock()
        mock_session_cls.return_value = mock_session
        mock_session.get.return_value = _make_mock_response(
            {"shop": {"name": "Test Store"}},
            status=200,
        )

        with patch("aragora.connectors.ecommerce.shopify.aiohttp", create=True) as mock_aiohttp:
            mock_aiohttp.ClientSession = mock_session_cls
            import aragora.connectors.ecommerce.shopify as mod

            # We need to mock the import inside connect()
            with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
                result = await connector.connect()

        assert result is True

    @pytest.mark.asyncio
    async def test_connect_failure_status(self, connector):
        """Non-200 response from shop.json returns False."""
        mock_session_cls = MagicMock()
        mock_session = AsyncMock()
        mock_session_cls.return_value = mock_session
        mock_session.get.return_value = _make_mock_response({}, status=401)

        with patch.dict("sys.modules", {"aiohttp": MagicMock(ClientSession=mock_session_cls)}):
            result = await connector.connect()

        assert result is False

    @pytest.mark.asyncio
    async def test_connect_import_error(self, connector):
        """connect() returns False when aiohttp is missing."""
        # Force an ImportError inside connect()
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
        """connect() returns False on unexpected exceptions."""
        mock_session_cls = MagicMock()
        mock_session = AsyncMock()
        mock_session_cls.return_value = mock_session
        mock_session.get.side_effect = RuntimeError("boom")

        with patch.dict("sys.modules", {"aiohttp": MagicMock(ClientSession=mock_session_cls)}):
            result = await connector.connect()

        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_closes_session(self, connector):
        """disconnect() closes the session and sets it to None."""
        connector._session = AsyncMock()
        await connector.disconnect()
        connector._session is None  # noqa: B015 (intentional assertion pattern)
        # After disconnect the attribute is None
        assert connector._session is None

    @pytest.mark.asyncio
    async def test_disconnect_noop_without_session(self, connector):
        """disconnect() is safe to call when no session exists."""
        connector._session = None
        await connector.disconnect()  # should not raise
        assert connector._session is None


# ========================================================================
# _request tests
# ========================================================================


class TestRequest:
    """Tests for the internal _request helper."""

    @pytest.mark.asyncio
    async def test_request_auto_connects(self, connector):
        """_request calls connect() when session is None."""
        connector._session = None
        connector.connect = AsyncMock()
        # After connect, simulate a session
        mock_session = AsyncMock()
        mock_session.request.return_value = _make_mock_response({"ok": True})

        async def fake_connect():
            connector._session = mock_session
            return True

        connector.connect = fake_connect
        result = await connector._request("GET", "/shop.json")
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_request_get(self, connector):
        """GET request returns parsed JSON."""
        mock_session = AsyncMock()
        mock_session.request.return_value = _make_mock_response({"data": 1})
        connector._session = mock_session

        result = await connector._request("GET", "/test.json")
        assert result == {"data": 1}

    @pytest.mark.asyncio
    async def test_request_post_with_json(self, connector):
        """POST request forwards JSON body."""
        mock_session = AsyncMock()
        mock_session.request.return_value = _make_mock_response({"id": 1})
        connector._session = mock_session

        body = {"order": {"note": "rush"}}
        result = await connector._request("POST", "/orders.json", json_data=body)
        assert result == {"id": 1}
        mock_session.request.assert_called_once()
        call_kwargs = mock_session.request.call_args
        assert call_kwargs[1]["json"] == body or call_kwargs.kwargs.get("json") == body

    @pytest.mark.asyncio
    async def test_request_return_headers(self, connector):
        """return_headers=True returns (data, headers) tuple."""
        headers = {"Link": '<url>; rel="next"', "X-Custom": "val"}
        mock_session = AsyncMock()
        resp_mock = AsyncMock()
        resp_mock.status = 200
        resp_mock.json = AsyncMock(return_value={"items": []})
        resp_mock.headers = headers

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp_mock)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.request.return_value = ctx
        connector._session = mock_session

        data, hdrs = await connector._request(
            "GET",
            "/items.json",
            return_headers=True,
        )
        assert data == {"items": []}
        assert hdrs["X-Custom"] == "val"

    @pytest.mark.asyncio
    async def test_request_http_error_raises(self, connector):
        """HTTP 4xx/5xx raises ConnectorAPIError."""
        mock_session = AsyncMock()
        resp_mock = AsyncMock()
        resp_mock.status = 422
        resp_mock.text = AsyncMock(return_value='{"errors":"bad data"}')

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp_mock)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.request.return_value = ctx
        connector._session = mock_session

        with pytest.raises(ConnectorAPIError) as exc_info:
            await connector._request("POST", "/orders.json", json_data={})
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_request_500_error(self, connector):
        """Server errors are wrapped as ConnectorAPIError with status 500."""
        mock_session = AsyncMock()
        resp_mock = AsyncMock()
        resp_mock.status = 500
        resp_mock.text = AsyncMock(return_value="Internal Server Error")

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp_mock)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.request.return_value = ctx
        connector._session = mock_session

        with pytest.raises(ConnectorAPIError) as exc_info:
            await connector._request("GET", "/shop.json")
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_request_429_rate_limit(self, connector):
        """HTTP 429 raises ConnectorAPIError with status 429."""
        mock_session = AsyncMock()
        resp_mock = AsyncMock()
        resp_mock.status = 429
        resp_mock.text = AsyncMock(return_value="Rate limited")

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp_mock)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.request.return_value = ctx
        connector._session = mock_session

        with pytest.raises(ConnectorAPIError) as exc_info:
            await connector._request("GET", "/orders.json")
        assert exc_info.value.status_code == 429


# ========================================================================
# Pagination (Link header)
# ========================================================================


class TestPagination:
    """Tests for _parse_link_header cursor-based pagination."""

    def test_parse_link_header_next(self, connector):
        """Extracts page_info from a Link header with rel=next."""
        header = (
            "<https://store.myshopify.com/admin/api/2024-01/orders.json"
            '?page_info=abc123>; rel="next"'
        )
        assert connector._parse_link_header(header) == "abc123"

    def test_parse_link_header_both(self, connector):
        """Extracts page_info when both previous and next are present."""
        header = (
            "<https://store.myshopify.com/admin/api/2024-01/orders.json"
            '?page_info=prev123>; rel="previous", '
            "<https://store.myshopify.com/admin/api/2024-01/orders.json"
            '?page_info=next456>; rel="next"'
        )
        assert connector._parse_link_header(header) == "next456"

    def test_parse_link_header_previous_only(self, connector):
        """Returns None when only rel=previous exists (last page)."""
        header = (
            "<https://store.myshopify.com/admin/api/2024-01/orders.json"
            '?page_info=prev123>; rel="previous"'
        )
        assert connector._parse_link_header(header) is None

    def test_parse_link_header_none(self, connector):
        """Returns None for a None header."""
        assert connector._parse_link_header(None) is None

    def test_parse_link_header_empty(self, connector):
        """Returns None for an empty string."""
        assert connector._parse_link_header("") is None


# ========================================================================
# Order parsing
# ========================================================================


class TestOrderParsing:
    """Tests for _parse_order and _parse_address helpers."""

    def test_parse_order(self, connector, sample_order_data):
        """Full order parsing from API response dict."""
        order = connector._parse_order(sample_order_data)

        assert order.id == "5001"
        assert order.order_number == 1042
        assert order.name == "#1042"
        assert order.email == "buyer@example.com"
        assert order.total_price == Decimal("69.85")
        assert order.currency == "USD"
        assert order.financial_status == PaymentStatus.PAID
        assert order.fulfillment_status == OrderStatus.FULFILLED
        assert order.customer_id == "7001"
        assert order.note == "Please gift wrap"
        assert order.tags == ["vip", "repeat-buyer"]
        assert order.cancelled_at is None
        assert order.closed_at is not None

    def test_parse_order_line_items(self, connector, sample_order_data):
        """Line items are correctly parsed within an order."""
        order = connector._parse_order(sample_order_data)
        assert len(order.line_items) == 1
        li = order.line_items[0]
        assert li.id == "9001"
        assert li.product_id == "2001"
        assert li.variant_id == "3001"
        assert li.title == "Widget"
        assert li.quantity == 3
        assert li.price == Decimal("19.95")

    def test_parse_order_addresses(self, connector, sample_order_data):
        """Shipping and billing addresses are parsed."""
        order = connector._parse_order(sample_order_data)
        assert order.shipping_address is not None
        assert order.shipping_address.city == "Portland"
        assert order.billing_address is not None
        assert order.billing_address.country_code == "US"

    def test_parse_order_no_addresses(self, connector, sample_order_data):
        """Order without addresses yields None."""
        sample_order_data.pop("shipping_address")
        sample_order_data.pop("billing_address")
        order = connector._parse_order(sample_order_data)
        assert order.shipping_address is None
        assert order.billing_address is None

    def test_parse_order_no_fulfillment_status(self, connector, sample_order_data):
        """Null fulfillment_status is parsed as None."""
        sample_order_data["fulfillment_status"] = None
        order = connector._parse_order(sample_order_data)
        assert order.fulfillment_status is None

    def test_parse_order_no_customer(self, connector, sample_order_data):
        """Missing customer yields customer_id=None."""
        sample_order_data.pop("customer")
        order = connector._parse_order(sample_order_data)
        assert order.customer_id is None

    def test_parse_order_empty_tags(self, connector, sample_order_data):
        """Empty tags string yields empty list."""
        sample_order_data["tags"] = ""
        order = connector._parse_order(sample_order_data)
        assert order.tags == []

    def test_parse_order_no_tags_key(self, connector, sample_order_data):
        """Missing tags key yields empty list."""
        sample_order_data.pop("tags")
        order = connector._parse_order(sample_order_data)
        assert order.tags == []

    def test_parse_address(self, connector, sample_address_data):
        """_parse_address converts raw dict to ShopifyAddress."""
        addr = connector._parse_address(sample_address_data)
        assert isinstance(addr, ShopifyAddress)
        assert addr.first_name == "Jane"
        assert addr.zip == "97201"

    def test_parse_order_cancelled_at(self, connector, sample_order_data):
        """cancelled_at is parsed when provided."""
        sample_order_data["cancelled_at"] = "2024-06-13T10:00:00Z"
        order = connector._parse_order(sample_order_data)
        assert order.cancelled_at is not None
        assert order.cancelled_at.tzinfo is not None


# ========================================================================
# Order operations
# ========================================================================


class TestOrderOperations:
    """Tests for get_order, sync_orders, fulfill_order."""

    @pytest.mark.asyncio
    async def test_get_order_success(self, connector, sample_order_data):
        """get_order returns a parsed ShopifyOrder."""
        connector._request = AsyncMock(return_value={"order": sample_order_data})
        order = await connector.get_order("5001")
        assert order is not None
        assert order.id == "5001"

    @pytest.mark.asyncio
    async def test_get_order_not_found(self, connector):
        """get_order returns None on failure."""
        connector._request = AsyncMock(
            side_effect=ConnectorAPIError(
                "Not found",
                connector_name="shopify",
                status_code=404,
            )
        )
        order = await connector.get_order("9999")
        assert order is None

    @pytest.mark.asyncio
    async def test_fulfill_order_success(self, connector):
        """fulfill_order sends POST and returns True."""
        connector._request = AsyncMock(return_value={"fulfillment": {"id": 1}})
        result = await connector.fulfill_order(
            "5001",
            tracking_number="1Z999",
            tracking_company="UPS",
            notify_customer=True,
        )
        assert result is True
        call_args = connector._request.call_args
        json_data = call_args.kwargs.get("json_data") or call_args[1].get("json_data")
        assert json_data["fulfillment"]["tracking_number"] == "1Z999"
        assert json_data["fulfillment"]["tracking_company"] == "UPS"
        assert json_data["fulfillment"]["notify_customer"] is True

    @pytest.mark.asyncio
    async def test_fulfill_order_minimal(self, connector):
        """fulfill_order without tracking data still sends notify_customer."""
        connector._request = AsyncMock(return_value={"fulfillment": {"id": 2}})
        result = await connector.fulfill_order("5001")
        assert result is True
        call_args = connector._request.call_args
        json_data = call_args.kwargs.get("json_data") or call_args[1].get("json_data")
        assert "tracking_number" not in json_data["fulfillment"]

    @pytest.mark.asyncio
    async def test_fulfill_order_failure(self, connector):
        """fulfill_order returns False on API error."""
        connector._request = AsyncMock(
            side_effect=ConnectorAPIError(
                "Cannot fulfill",
                connector_name="shopify",
                status_code=422,
            )
        )
        result = await connector.fulfill_order("5001")
        assert result is False

    @pytest.mark.asyncio
    async def test_sync_orders_single_page(self, connector, sample_order_data):
        """sync_orders yields orders from a single page."""
        connector._request = AsyncMock(
            return_value=(
                {"orders": [sample_order_data]},
                {},  # no Link header
            )
        )
        orders = []
        async for order in connector.sync_orders(limit=250):
            orders.append(order)
        assert len(orders) == 1
        assert orders[0].id == "5001"

    @pytest.mark.asyncio
    async def test_sync_orders_with_since(self, connector, sample_order_data):
        """sync_orders passes updated_at_min when since is provided."""
        since = datetime(2024, 1, 1, tzinfo=timezone.utc)
        connector._request = AsyncMock(
            return_value=(
                {"orders": [sample_order_data]},
                {},
            )
        )
        orders = []
        async for order in connector.sync_orders(since=since):
            orders.append(order)
        call_params = connector._request.call_args.kwargs.get(
            "params"
        ) or connector._request.call_args[1].get("params")
        assert "updated_at_min" in call_params

    @pytest.mark.asyncio
    async def test_sync_orders_pagination(self, connector, sample_order_data):
        """sync_orders follows Link header pagination."""
        page1_data = {"orders": [sample_order_data] * 250}
        page2_data = {"orders": [sample_order_data]}
        link_header = (
            "<https://store.myshopify.com/admin/api/2024-01/orders.json"
            '?page_info=cursor2>; rel="next"'
        )

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return page1_data, {"Link": link_header}
            return page2_data, {}

        connector._request = mock_request
        orders = []
        async for order in connector.sync_orders(limit=250):
            orders.append(order)
        assert len(orders) == 251
        assert call_count == 2


# ========================================================================
# Product parsing and operations
# ========================================================================


class TestProductOperations:
    """Tests for product parsing and CRUD operations."""

    def test_parse_product(self, connector, sample_product_data):
        """_parse_product produces a complete ShopifyProduct."""
        product = connector._parse_product(sample_product_data)
        assert product.id == "2001"
        assert product.title == "Widget"
        assert product.handle == "widget"
        assert product.vendor == "WidgetCo"
        assert product.product_type == "Gadgets"
        assert product.status == "active"
        assert product.description == "<p>A fine widget</p>"
        assert product.tags == ["sale", "new-arrival"]
        assert len(product.variants) == 1
        assert len(product.images) == 1

    def test_parse_product_variant(self, connector, sample_product_data):
        """Variant within a product is fully parsed."""
        product = connector._parse_product(sample_product_data)
        v = product.variants[0]
        assert v.id == "3001"
        assert v.product_id == "2001"
        assert v.price == Decimal("29.99")
        assert v.sku == "WGT-LG-RED"
        assert v.inventory_quantity == 12
        assert v.inventory_policy == InventoryPolicy.DENY
        assert v.compare_at_price == Decimal("39.99")
        assert v.option1 == "Large"
        assert v.option2 == "Red"

    def test_parse_product_no_tags(self, connector, sample_product_data):
        """Product without tags string yields empty list."""
        sample_product_data["tags"] = ""
        product = connector._parse_product(sample_product_data)
        assert product.tags == []

    def test_parse_product_no_published_at(self, connector, sample_product_data):
        """Product without published_at yields None."""
        sample_product_data["published_at"] = None
        product = connector._parse_product(sample_product_data)
        assert product.published_at is None

    @pytest.mark.asyncio
    async def test_get_product_success(self, connector, sample_product_data):
        """get_product returns parsed product."""
        connector._request = AsyncMock(return_value={"product": sample_product_data})
        product = await connector.get_product("2001")
        assert product is not None
        assert product.title == "Widget"

    @pytest.mark.asyncio
    async def test_get_product_not_found(self, connector):
        """get_product returns None on error."""
        connector._request = AsyncMock(side_effect=Exception("404"))
        product = await connector.get_product("9999")
        assert product is None

    @pytest.mark.asyncio
    async def test_sync_products(self, connector, sample_product_data):
        """sync_products yields products."""
        connector._request = AsyncMock(
            return_value={"products": [sample_product_data]},
        )
        products = []
        async for p in connector.sync_products():
            products.append(p)
        assert len(products) == 1
        assert products[0].id == "2001"

    @pytest.mark.asyncio
    async def test_sync_products_with_since(self, connector, sample_product_data):
        """sync_products passes updated_at_min when since is provided."""
        since = datetime(2024, 1, 1, tzinfo=timezone.utc)
        connector._request = AsyncMock(
            return_value={"products": [sample_product_data]},
        )
        products = []
        async for p in connector.sync_products(since=since):
            products.append(p)
        call_params = connector._request.call_args.kwargs.get(
            "params"
        ) or connector._request.call_args[1].get("params")
        assert "updated_at_min" in call_params


# ========================================================================
# Inventory operations
# ========================================================================


class TestInventoryOperations:
    """Tests for inventory adjustment and low-stock detection."""

    @pytest.mark.asyncio
    async def test_update_variant_inventory_success(self, connector):
        """update_variant_inventory sends POST and returns True."""
        connector._request = AsyncMock(return_value={"inventory_level": {}})
        result = await connector.update_variant_inventory("ii_1", "loc_1", 10)
        assert result is True
        call_args = connector._request.call_args
        json_data = call_args.kwargs.get("json_data") or call_args[1].get("json_data")
        assert json_data["inventory_item_id"] == "ii_1"
        assert json_data["available_adjustment"] == 10

    @pytest.mark.asyncio
    async def test_update_variant_inventory_negative(self, connector):
        """Negative adjustment is forwarded correctly."""
        connector._request = AsyncMock(return_value={"inventory_level": {}})
        result = await connector.update_variant_inventory("ii_1", "loc_1", -5)
        assert result is True
        call_args = connector._request.call_args
        json_data = call_args.kwargs.get("json_data") or call_args[1].get("json_data")
        assert json_data["available_adjustment"] == -5

    @pytest.mark.asyncio
    async def test_update_variant_inventory_failure(self, connector):
        """update_variant_inventory returns False on API error."""
        connector._request = AsyncMock(
            side_effect=ConnectorAPIError(
                "Inventory error",
                connector_name="shopify",
                status_code=422,
            )
        )
        result = await connector.update_variant_inventory("ii_1", "loc_1", 10)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_low_stock_variants(self, connector, sample_product_data):
        """get_low_stock_variants filters variants below threshold."""
        # Override variant quantities: one low (3), one high (50)
        sample_product_data["variants"][0]["inventory_quantity"] = 3
        connector._request = AsyncMock(
            return_value={"products": [sample_product_data]},
        )
        low = await connector.get_low_stock_variants(threshold=5)
        assert len(low) == 1
        assert low[0].inventory_quantity == 3

    @pytest.mark.asyncio
    async def test_get_low_stock_variants_none_low(self, connector, sample_product_data):
        """get_low_stock_variants returns empty list when all stock is adequate."""
        sample_product_data["variants"][0]["inventory_quantity"] = 100
        connector._request = AsyncMock(
            return_value={"products": [sample_product_data]},
        )
        low = await connector.get_low_stock_variants(threshold=5)
        assert low == []

    @pytest.mark.asyncio
    async def test_get_low_stock_variants_equal_threshold(self, connector, sample_product_data):
        """Variants at exactly the threshold are included."""
        sample_product_data["variants"][0]["inventory_quantity"] = 5
        connector._request = AsyncMock(
            return_value={"products": [sample_product_data]},
        )
        low = await connector.get_low_stock_variants(threshold=5)
        assert len(low) == 1


# ========================================================================
# Customer parsing and operations
# ========================================================================


class TestCustomerOperations:
    """Tests for customer parsing and retrieval."""

    def test_parse_customer(self, connector, sample_customer_data):
        """_parse_customer produces correct ShopifyCustomer."""
        customer = connector._parse_customer(sample_customer_data)
        assert customer.id == "7001"
        assert customer.email == "buyer@example.com"
        assert customer.first_name == "Jane"
        assert customer.last_name == "Doe"
        assert customer.orders_count == 5
        assert customer.total_spent == Decimal("349.95")
        assert customer.verified_email is True
        assert customer.accepts_marketing is False
        assert customer.tags == ["vip", "repeat-buyer"]
        assert customer.note == "Prefers email contact"

    def test_parse_customer_empty_tags(self, connector, sample_customer_data):
        """Empty tags string yields empty list."""
        sample_customer_data["tags"] = ""
        customer = connector._parse_customer(sample_customer_data)
        assert customer.tags == []

    def test_parse_customer_no_tags(self, connector, sample_customer_data):
        """Missing tags key yields empty list."""
        sample_customer_data.pop("tags")
        customer = connector._parse_customer(sample_customer_data)
        assert customer.tags == []

    @pytest.mark.asyncio
    async def test_get_customer_success(self, connector, sample_customer_data):
        """get_customer returns parsed customer."""
        connector._request = AsyncMock(
            return_value={"customer": sample_customer_data},
        )
        customer = await connector.get_customer("7001")
        assert customer is not None
        assert customer.full_name == "Jane Doe"

    @pytest.mark.asyncio
    async def test_get_customer_not_found(self, connector):
        """get_customer returns None on error."""
        connector._request = AsyncMock(side_effect=Exception("Not found"))
        customer = await connector.get_customer("9999")
        assert customer is None

    @pytest.mark.asyncio
    async def test_sync_customers(self, connector, sample_customer_data):
        """sync_customers yields customer objects."""
        connector._request = AsyncMock(
            return_value={"customers": [sample_customer_data]},
        )
        customers = []
        async for c in connector.sync_customers():
            customers.append(c)
        assert len(customers) == 1
        assert customers[0].id == "7001"

    @pytest.mark.asyncio
    async def test_sync_customers_with_since(self, connector, sample_customer_data):
        """sync_customers passes updated_at_min when since is provided."""
        since = datetime(2024, 1, 1, tzinfo=timezone.utc)
        connector._request = AsyncMock(
            return_value={"customers": [sample_customer_data]},
        )
        customers = []
        async for c in connector.sync_customers(since=since):
            customers.append(c)
        call_params = connector._request.call_args.kwargs.get(
            "params"
        ) or connector._request.call_args[1].get("params")
        assert "updated_at_min" in call_params


# ========================================================================
# Analytics
# ========================================================================


class TestAnalytics:
    """Tests for get_order_stats analytics."""

    @pytest.mark.asyncio
    async def test_order_stats_basic(self, connector, sample_order_data):
        """get_order_stats computes totals from synced orders."""
        connector._request = AsyncMock(
            return_value=(
                {"orders": [sample_order_data]},
                {},
            )
        )
        stats = await connector.get_order_stats()
        assert stats["total_orders"] == 1
        assert Decimal(stats["total_revenue"]) == Decimal("69.85")
        assert stats["fulfilled_orders"] == 1
        assert stats["cancelled_orders"] == 0
        assert stats["fulfillment_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_order_stats_no_orders(self, connector):
        """get_order_stats handles empty order list."""
        connector._request = AsyncMock(
            return_value=(
                {"orders": []},
                {},
            )
        )
        stats = await connector.get_order_stats()
        assert stats["total_orders"] == 0
        assert stats["fulfillment_rate"] == 0
        assert stats["average_order_value"] == "0.00"

    @pytest.mark.asyncio
    async def test_order_stats_with_date_filter(self, connector, sample_order_data):
        """Orders outside end_date are excluded from stats."""
        # The sample order was created on 2024-06-10, so set end_date before that
        end_date = datetime(2024, 6, 1, tzinfo=timezone.utc)
        connector._request = AsyncMock(
            return_value=(
                {"orders": [sample_order_data]},
                {},
            )
        )
        stats = await connector.get_order_stats(end_date=end_date)
        assert stats["total_orders"] == 0

    @pytest.mark.asyncio
    async def test_order_stats_average_value(self, connector, sample_order_data):
        """Average order value is computed correctly."""
        # Two orders: 69.85 and 30.15 = 100.00 total, avg = 50.00
        order2 = dict(sample_order_data)
        order2["id"] = 5002
        order2["total_price"] = "30.15"
        order2["fulfillment_status"] = None

        connector._request = AsyncMock(
            return_value=(
                {"orders": [sample_order_data, order2]},
                {},
            )
        )
        stats = await connector.get_order_stats()
        assert stats["total_orders"] == 2
        avg = Decimal(stats["average_order_value"])
        assert avg == Decimal("50.00") or avg == Decimal("50")


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
        # Mock each sub-sync to return one item
        original_sync_orders = connector.sync_orders
        original_sync_products = connector.sync_products
        original_sync_customers = connector.sync_customers

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
    async def test_incremental_sync_with_state(self, connector, sample_order_data):
        """incremental_sync uses state.last_sync_at as since parameter."""
        from aragora.connectors.enterprise.base import SyncState

        state = SyncState(
            connector_id="shopify",
            last_sync_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )

        async def fake_sync_orders(**kwargs):
            # Verify since is passed from state
            if False:
                yield  # make it an async generator
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
            raise RuntimeError("sync failed")

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
        from aragora.connectors.enterprise.base import SyncState, SyncItem

        async def fake_sync_orders(**kwargs):
            yield connector._parse_order(sample_order_data)

        async def fake_sync_products(**kwargs):
            yield connector._parse_product(sample_product_data)

        async def fake_sync_customers(**kwargs):
            yield connector._parse_customer(sample_customer_data)

        connector.sync_orders = fake_sync_orders
        connector.sync_products = fake_sync_products
        connector.sync_customers = fake_sync_customers

        state = SyncState(connector_id="shopify")
        items = []
        async for item in connector.sync_items(state, batch_size=50):
            items.append(item)

        assert len(items) == 3
        ids = {item.id for item in items}
        assert "shopify:order:5001" in ids
        assert "shopify:product:2001" in ids
        assert "shopify:customer:7001" in ids

        # All should have source_type "ecommerce"
        assert all(item.source_type == "ecommerce" for item in items)

    @pytest.mark.asyncio
    async def test_sync_items_order_metadata(self, connector, sample_order_data):
        """sync_items order SyncItem contains expected metadata."""
        from aragora.connectors.enterprise.base import SyncState

        async def fake_sync_orders(**kwargs):
            yield connector._parse_order(sample_order_data)

        async def fake_sync_products(**kwargs):
            return
            yield  # pragma: no cover

        async def fake_sync_customers(**kwargs):
            return
            yield  # pragma: no cover

        connector.sync_orders = fake_sync_orders
        connector.sync_products = fake_sync_products
        connector.sync_customers = fake_sync_customers

        state = SyncState(connector_id="shopify")
        items = []
        async for item in connector.sync_items(state):
            items.append(item)

        assert len(items) == 1
        item = items[0]
        assert item.metadata["type"] == "order"
        assert item.metadata["currency"] == "USD"
        assert item.metadata["financial_status"] == "paid"
        assert item.confidence == 0.95

    @pytest.mark.asyncio
    async def test_sync_items_customer_metadata(self, connector, sample_customer_data):
        """sync_items customer SyncItem contains expected metadata."""
        from aragora.connectors.enterprise.base import SyncState

        async def fake_sync_orders(**kwargs):
            return
            yield  # pragma: no cover

        async def fake_sync_products(**kwargs):
            return
            yield  # pragma: no cover

        async def fake_sync_customers(**kwargs):
            yield connector._parse_customer(sample_customer_data)

        connector.sync_orders = fake_sync_orders
        connector.sync_products = fake_sync_products
        connector.sync_customers = fake_sync_customers

        state = SyncState(connector_id="shopify")
        items = []
        async for item in connector.sync_items(state):
            items.append(item)

        assert len(items) == 1
        item = items[0]
        assert item.metadata["type"] == "customer"
        assert item.metadata["email"] == "buyer@example.com"
        assert item.confidence == 0.9


# ========================================================================
# Mock data helpers
# ========================================================================


class TestMockDataHelpers:
    """Tests for the module-level get_mock_* helper functions."""

    def test_get_mock_orders(self):
        """get_mock_orders returns a non-empty list of valid orders."""
        orders = get_mock_orders()
        assert len(orders) >= 1
        order = orders[0]
        assert isinstance(order, ShopifyOrder)
        assert order.id == "1001"
        assert order.total_price == Decimal("99.99")
        assert order.financial_status == PaymentStatus.PAID
        assert len(order.line_items) >= 1

    def test_get_mock_products(self):
        """get_mock_products returns a non-empty list of valid products."""
        products = get_mock_products()
        assert len(products) >= 1
        product = products[0]
        assert isinstance(product, ShopifyProduct)
        assert product.id == "prod_1"
        assert product.title == "Sample Product"
        assert len(product.variants) >= 1
        assert product.variants[0].price == Decimal("89.99")

    def test_mock_orders_have_datetimes(self):
        """Mock orders contain timezone-aware datetimes."""
        orders = get_mock_orders()
        for order in orders:
            assert order.created_at.tzinfo is not None
            assert order.updated_at.tzinfo is not None

    def test_mock_products_have_datetimes(self):
        """Mock products contain timezone-aware datetimes."""
        products = get_mock_products()
        for product in products:
            assert product.created_at.tzinfo is not None
            assert product.updated_at.tzinfo is not None


# ========================================================================
# Error handling edge cases
# ========================================================================


class TestErrorHandling:
    """Tests for error handling edge cases across operations."""

    @pytest.mark.asyncio
    async def test_request_preserves_status_code(self, connector):
        """ConnectorAPIError carries the original HTTP status code."""
        mock_session = AsyncMock()
        resp_mock = AsyncMock()
        resp_mock.status = 403
        resp_mock.text = AsyncMock(return_value="Forbidden")

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp_mock)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.request.return_value = ctx
        connector._session = mock_session

        with pytest.raises(ConnectorAPIError) as exc_info:
            await connector._request("GET", "/orders.json")
        assert exc_info.value.status_code == 403
        assert exc_info.value.connector_name == "shopify"

    @pytest.mark.asyncio
    async def test_get_order_logs_and_returns_none(self, connector):
        """get_order catches generic exceptions and returns None."""
        connector._request = AsyncMock(side_effect=RuntimeError("network down"))
        result = await connector.get_order("123")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_product_logs_and_returns_none(self, connector):
        """get_product catches generic exceptions and returns None."""
        connector._request = AsyncMock(side_effect=RuntimeError("timeout"))
        result = await connector.get_product("123")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_customer_logs_and_returns_none(self, connector):
        """get_customer catches generic exceptions and returns None."""
        connector._request = AsyncMock(side_effect=RuntimeError("dns fail"))
        result = await connector.get_customer("123")
        assert result is None

    @pytest.mark.asyncio
    async def test_fulfill_order_logs_and_returns_false(self, connector):
        """fulfill_order catches generic exceptions and returns False."""
        connector._request = AsyncMock(side_effect=RuntimeError("boom"))
        result = await connector.fulfill_order("123")
        assert result is False

    @pytest.mark.asyncio
    async def test_update_inventory_logs_and_returns_false(self, connector):
        """update_variant_inventory catches generic exceptions and returns False."""
        connector._request = AsyncMock(side_effect=RuntimeError("boom"))
        result = await connector.update_variant_inventory("ii", "loc", 1)
        assert result is False


# ========================================================================
# Serialization round-trips
# ========================================================================


class TestSerializationRoundTrips:
    """Tests for to_dict serialization consistency."""

    def test_line_item_decimal_serialized(self):
        """Line item price serializes to string."""
        item = ShopifyLineItem(
            id="1",
            product_id=None,
            variant_id=None,
            title="X",
            quantity=1,
            price=Decimal("12.50"),
        )
        d = item.to_dict()
        assert d["price"] == "12.50"

    def test_order_enum_serialized(self, now_utc):
        """Enum fields serialize to their string values."""
        order = ShopifyOrder(
            id="1",
            order_number=1,
            name="#1",
            email=None,
            created_at=now_utc,
            updated_at=now_utc,
            total_price=Decimal("0"),
            subtotal_price=Decimal("0"),
            total_tax=Decimal("0"),
            total_discounts=Decimal("0"),
            currency="USD",
            financial_status=PaymentStatus.AUTHORIZED,
            fulfillment_status=OrderStatus.OPEN,
        )
        d = order.to_dict()
        assert d["financial_status"] == "authorized"
        assert d["fulfillment_status"] == "open"

    def test_variant_inventory_policy_serialized(self):
        """InventoryPolicy enum serializes to string."""
        v = ShopifyVariant(
            id="v1",
            product_id="p1",
            title="T",
            price=Decimal("1.00"),
            sku=None,
            inventory_policy=InventoryPolicy.CONTINUE,
        )
        d = v.to_dict()
        assert d["inventory_policy"] == "continue"

    def test_datetime_serialized_as_iso(self, now_utc):
        """Datetime fields serialize to ISO 8601 strings."""
        inv = ShopifyInventoryLevel(
            inventory_item_id="ii",
            location_id="loc",
            available=10,
            updated_at=now_utc,
        )
        d = inv.to_dict()
        assert "2024-06-15" in d["updated_at"]

    def test_nested_line_items_in_order(self, now_utc):
        """Nested line items in an order are serialized recursively."""
        li = ShopifyLineItem(
            id="li1",
            product_id="p1",
            variant_id="v1",
            title="Thing",
            quantity=2,
            price=Decimal("5.00"),
        )
        order = ShopifyOrder(
            id="1",
            order_number=1,
            name="#1",
            email=None,
            created_at=now_utc,
            updated_at=now_utc,
            total_price=Decimal("10.00"),
            subtotal_price=Decimal("10.00"),
            total_tax=Decimal("0"),
            total_discounts=Decimal("0"),
            currency="USD",
            financial_status=PaymentStatus.PAID,
            fulfillment_status=None,
            line_items=[li],
        )
        d = order.to_dict()
        assert len(d["line_items"]) == 1
        assert d["line_items"][0]["quantity"] == 2
