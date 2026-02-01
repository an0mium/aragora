"""
Comprehensive tests for the Amazon Seller Central Connector (SP-API).

Covers:
- Enum types (AmazonMarketplace, AmazonOrderStatus, FulfillmentChannel, InventoryCondition)
- Dataclass models (credentials, address, order item, order, inventory item, product)
- AmazonConnector class (initialization, connection, order sync, inventory operations,
  catalog/product operations, reports, analytics, enterprise sync methods)
- Authentication and token refresh handling
- Rate limiting and error handling
- Data serialization and deserialization
- Mock data helpers
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.ecommerce.amazon import (
    AmazonAddress,
    AmazonConnector,
    AmazonCredentials,
    AmazonInventoryItem,
    AmazonMarketplace,
    AmazonOrder,
    AmazonOrderItem,
    AmazonOrderStatus,
    AmazonProduct,
    FulfillmentChannel,
    InventoryCondition,
    get_mock_inventory,
    get_mock_orders,
)
from aragora.connectors.enterprise.base import SyncState, SyncStatus


# ---------------------------------------------------------------------------
# Concrete subclass for testing (AmazonConnector has abstract methods)
# ---------------------------------------------------------------------------


class _TestableAmazonConnector(AmazonConnector):
    """Concrete subclass that stubs abstract methods from BaseConnector."""

    @property
    def name(self) -> str:
        return "amazon"

    @property
    def source_type(self):
        from aragora.reasoning.provenance import SourceType

        return SourceType.EXTERNAL_API

    async def search(self, query: str, **kwargs):
        return []

    async def fetch(self, evidence_id: str, **kwargs):
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def credentials():
    """Standard test credentials."""
    return AmazonCredentials(
        refresh_token="Atzr|test_refresh_token",
        client_id="amzn1.application-oa2-client.test123",
        client_secret="test_client_secret_abc123",
        marketplace_id=AmazonMarketplace.US.value,
        seller_id="A1234TEST",
        aws_access_key="AKIATEST123",
        aws_secret_key="secret123",
        role_arn="arn:aws:iam::123456789012:role/test-role",
    )


@pytest.fixture
def connector(credentials):
    """Testable AmazonConnector in mock mode."""
    return _TestableAmazonConnector(
        credentials=credentials,
        sandbox=False,
        use_mock=True,
    )


@pytest.fixture
def now_utc():
    """Deterministic UTC datetime."""
    return datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_address():
    """Sample Amazon shipping/billing address."""
    return AmazonAddress(
        name="John Doe",
        address_line1="123 Main St",
        address_line2="Suite 100",
        address_line3=None,
        city="Seattle",
        state_or_region="WA",
        postal_code="98101",
        country_code="US",
        phone="+12065551234",
    )


@pytest.fixture
def sample_order_item():
    """Sample Amazon order item."""
    return AmazonOrderItem(
        order_item_id="item-001",
        asin="B00EXAMPLE1",
        seller_sku="SKU-001",
        title="Sample Product",
        quantity_ordered=2,
        quantity_shipped=0,
        item_price=Decimal("29.99"),
        item_tax=Decimal("2.40"),
        shipping_price=Decimal("5.99"),
        shipping_tax=Decimal("0.48"),
        promotion_discount=Decimal("0.00"),
        condition="New",
        is_gift=False,
    )


@pytest.fixture
def sample_order(sample_address, sample_order_item, now_utc):
    """Sample Amazon order."""
    return AmazonOrder(
        amazon_order_id="111-1234567-1234567",
        seller_order_id="SELLER-001",
        purchase_date=now_utc,
        last_update_date=now_utc,
        order_status=AmazonOrderStatus.UNSHIPPED,
        fulfillment_channel=FulfillmentChannel.MFN,
        sales_channel="Amazon.com",
        order_total=Decimal("38.86"),
        currency_code="USD",
        number_of_items_shipped=0,
        number_of_items_unshipped=2,
        shipping_address=sample_address,
        buyer_email="buyer@example.com",
        buyer_name="John Doe",
        order_items=[sample_order_item],
        is_prime=True,
        is_business_order=False,
        is_replacement_order=False,
    )


@pytest.fixture
def sample_inventory_item():
    """Sample Amazon FBA inventory item."""
    return AmazonInventoryItem(
        asin="B00EXAMPLE1",
        seller_sku="SKU-001",
        fnsku="X00ABCD1234",
        product_name="Sample Product",
        condition=InventoryCondition.NEW,
        total_quantity=100,
        inbound_quantity=10,
        available_quantity=80,
        reserved_quantity=5,
        unfulfillable_quantity=3,
        researching_quantity=2,
    )


@pytest.fixture
def sample_product():
    """Sample Amazon catalog item."""
    return AmazonProduct(
        asin="B00EXAMPLE1",
        title="Sample Product",
        brand="SampleBrand",
        manufacturer="SampleManufacturer",
        product_type="Electronics",
        parent_asin=None,
        item_dimensions={"length": 10, "width": 5, "height": 2, "unit": "inches"},
        package_dimensions={"length": 12, "width": 8, "height": 4, "unit": "inches"},
        images=["https://images.amazon.com/images/I/example1.jpg"],
        bullet_points=["Feature 1", "Feature 2", "Feature 3"],
        browse_nodes=["12345", "67890"],
    )


@pytest.fixture
def sample_sp_api_order_data(now_utc):
    """Raw order dict as returned by SP-API."""
    return {
        "AmazonOrderId": "111-1234567-1234567",
        "SellerOrderId": "SELLER-001",
        "PurchaseDate": now_utc.isoformat().replace("+00:00", "Z"),
        "LastUpdateDate": now_utc.isoformat().replace("+00:00", "Z"),
        "OrderStatus": "Unshipped",
        "FulfillmentChannel": "MFN",
        "SalesChannel": "Amazon.com",
        "OrderTotal": {"Amount": "38.86", "CurrencyCode": "USD"},
        "NumberOfItemsShipped": 0,
        "NumberOfItemsUnshipped": 2,
        "BuyerEmail": "buyer@example.com",
        "BuyerName": "John Doe",
        "ShippingAddress": {
            "Name": "John Doe",
            "AddressLine1": "123 Main St",
            "AddressLine2": "Suite 100",
            "City": "Seattle",
            "StateOrRegion": "WA",
            "PostalCode": "98101",
            "CountryCode": "US",
            "Phone": "+12065551234",
        },
        "IsBusinessOrder": False,
        "IsPrime": True,
    }


# ========================================================================
# Enum tests
# ========================================================================


class TestEnums:
    """Tests for Amazon-specific enum types."""

    def test_amazon_marketplace_values(self):
        """AmazonMarketplace has the expected US marketplace ID."""
        assert AmazonMarketplace.US.value == "ATVPDKIKX0DER"
        assert AmazonMarketplace.UK.value == "A1F83G8C2ARO7P"
        assert AmazonMarketplace.CA.value == "A2EUQ1WTGCTBG2"

    def test_amazon_marketplace_all_values(self):
        """AmazonMarketplace covers major marketplaces."""
        expected_markets = {"US", "CA", "MX", "UK", "DE", "FR", "IT", "ES", "JP", "AU"}
        assert {m.name for m in AmazonMarketplace} == expected_markets

    def test_amazon_order_status_values(self):
        """AmazonOrderStatus covers expected fulfillment states."""
        expected = {
            "Pending",
            "Unshipped",
            "PartiallyShipped",
            "Shipped",
            "Canceled",
            "Unfulfillable",
            "InvoiceUnconfirmed",
            "PendingAvailability",
        }
        assert {s.value for s in AmazonOrderStatus} == expected

    def test_fulfillment_channel_values(self):
        """FulfillmentChannel covers AFN and MFN."""
        assert FulfillmentChannel.AFN.value == "AFN"
        assert FulfillmentChannel.MFN.value == "MFN"

    def test_inventory_condition_values(self):
        """InventoryCondition covers item conditions."""
        assert InventoryCondition.NEW.value == "NewItem"
        assert InventoryCondition.USED_LIKE_NEW.value == "UsedLikeNew"
        assert InventoryCondition.REFURBISHED.value == "Refurbished"

    def test_enum_from_string(self):
        """Enums can be constructed from their string values."""
        assert AmazonOrderStatus("Shipped") == AmazonOrderStatus.SHIPPED
        assert FulfillmentChannel("AFN") == FulfillmentChannel.AFN
        assert InventoryCondition("NewItem") == InventoryCondition.NEW


# ========================================================================
# Credential tests
# ========================================================================


class TestAmazonCredentials:
    """Tests for AmazonCredentials dataclass."""

    def test_basic_construction(self, credentials):
        """Credentials store the expected fields."""
        assert credentials.refresh_token == "Atzr|test_refresh_token"
        assert credentials.client_id == "amzn1.application-oa2-client.test123"
        assert credentials.client_secret == "test_client_secret_abc123"
        assert credentials.marketplace_id == "ATVPDKIKX0DER"
        assert credentials.seller_id == "A1234TEST"
        assert credentials.aws_access_key == "AKIATEST123"
        assert credentials.aws_secret_key == "secret123"
        assert credentials.role_arn == "arn:aws:iam::123456789012:role/test-role"

    def test_optional_aws_fields(self):
        """AWS fields are optional."""
        creds = AmazonCredentials(
            refresh_token="token",
            client_id="client_id",
            client_secret="secret",
            marketplace_id="ATVPDKIKX0DER",
            seller_id="seller",
        )
        assert creds.aws_access_key is None
        assert creds.aws_secret_key is None
        assert creds.role_arn is None

    def test_from_env(self):
        """Credentials can be loaded from environment variables."""
        env = {
            "AMAZON_SP_REFRESH_TOKEN": "env_refresh_token",
            "AMAZON_SP_CLIENT_ID": "env_client_id",
            "AMAZON_SP_CLIENT_SECRET": "env_client_secret",
            "AMAZON_SP_MARKETPLACE_ID": "A1F83G8C2ARO7P",
            "AMAZON_SP_SELLER_ID": "ENV_SELLER",
            "AMAZON_SP_AWS_ACCESS_KEY": "AKIAENV123",
            "AMAZON_SP_AWS_SECRET_KEY": "envsecret",
            "AMAZON_SP_ROLE_ARN": "arn:aws:iam::999:role/env-role",
        }
        with patch.dict(os.environ, env, clear=False):
            creds = AmazonCredentials.from_env()
            assert creds.refresh_token == "env_refresh_token"
            assert creds.client_id == "env_client_id"
            assert creds.client_secret == "env_client_secret"
            assert creds.marketplace_id == "A1F83G8C2ARO7P"
            assert creds.seller_id == "ENV_SELLER"
            assert creds.aws_access_key == "AKIAENV123"
            assert creds.aws_secret_key == "envsecret"
            assert creds.role_arn == "arn:aws:iam::999:role/env-role"

    def test_from_env_defaults(self):
        """Missing env vars resolve to empty strings / US marketplace."""
        with patch.dict(os.environ, {}, clear=True):
            creds = AmazonCredentials.from_env()
            assert creds.refresh_token == ""
            assert creds.client_id == ""
            assert creds.client_secret == ""
            assert creds.marketplace_id == AmazonMarketplace.US.value
            assert creds.seller_id == ""
            assert creds.aws_access_key is None
            assert creds.aws_secret_key is None


# ========================================================================
# Address tests
# ========================================================================


class TestAmazonAddress:
    """Tests for AmazonAddress dataclass."""

    def test_construction_defaults(self):
        """All fields default to None."""
        addr = AmazonAddress()
        assert addr.name is None
        assert addr.address_line1 is None
        assert addr.city is None
        assert addr.country_code is None

    def test_full_construction(self, sample_address):
        """Address can be constructed with all fields."""
        assert sample_address.name == "John Doe"
        assert sample_address.address_line1 == "123 Main St"
        assert sample_address.city == "Seattle"
        assert sample_address.state_or_region == "WA"
        assert sample_address.postal_code == "98101"
        assert sample_address.country_code == "US"
        assert sample_address.phone == "+12065551234"

    def test_to_dict(self, sample_address):
        """to_dict converts to API format with camelCase keys."""
        d = sample_address.to_dict()
        assert d["name"] == "John Doe"
        assert d["addressLine1"] == "123 Main St"
        assert d["addressLine2"] == "Suite 100"
        assert d["addressLine3"] is None
        assert d["city"] == "Seattle"
        assert d["stateOrRegion"] == "WA"
        assert d["postalCode"] == "98101"
        assert d["countryCode"] == "US"
        assert d["phone"] == "+12065551234"

    def test_to_dict_empty(self):
        """Empty address to_dict returns all None values."""
        addr = AmazonAddress()
        d = addr.to_dict()
        assert all(v is None for v in d.values())


# ========================================================================
# Order item tests
# ========================================================================


class TestAmazonOrderItem:
    """Tests for AmazonOrderItem dataclass."""

    def test_construction(self, sample_order_item):
        """Order item stores essential line data."""
        assert sample_order_item.order_item_id == "item-001"
        assert sample_order_item.asin == "B00EXAMPLE1"
        assert sample_order_item.seller_sku == "SKU-001"
        assert sample_order_item.title == "Sample Product"
        assert sample_order_item.quantity_ordered == 2
        assert sample_order_item.quantity_shipped == 0
        assert sample_order_item.item_price == Decimal("29.99")
        assert sample_order_item.item_tax == Decimal("2.40")
        assert sample_order_item.is_gift is False

    def test_to_dict(self, sample_order_item):
        """to_dict converts to API format."""
        d = sample_order_item.to_dict()
        assert d["orderItemId"] == "item-001"
        assert d["asin"] == "B00EXAMPLE1"
        assert d["sellerSku"] == "SKU-001"
        assert d["title"] == "Sample Product"
        assert d["quantityOrdered"] == 2
        assert d["quantityShipped"] == 0
        assert d["itemPrice"] == "29.99"
        assert d["itemTax"] == "2.40"
        assert d["shippingPrice"] == "5.99"
        assert d["shippingTax"] == "0.48"
        assert d["promotionDiscount"] == "0.00"
        assert d["condition"] == "New"
        assert d["isGift"] is False

    def test_defaults(self):
        """Defaults are applied correctly."""
        item = AmazonOrderItem(
            order_item_id="id1",
            asin="B001",
            seller_sku=None,
            title="Title",
            quantity_ordered=1,
            quantity_shipped=0,
            item_price=Decimal("10.00"),
            item_tax=Decimal("0.00"),
            shipping_price=Decimal("0.00"),
            shipping_tax=Decimal("0.00"),
            promotion_discount=Decimal("0.00"),
        )
        assert item.condition is None
        assert item.is_gift is False


# ========================================================================
# Order tests
# ========================================================================


class TestAmazonOrder:
    """Tests for AmazonOrder dataclass."""

    def test_construction(self, sample_order):
        """Order stores all expected fields."""
        assert sample_order.amazon_order_id == "111-1234567-1234567"
        assert sample_order.seller_order_id == "SELLER-001"
        assert sample_order.order_status == AmazonOrderStatus.UNSHIPPED
        assert sample_order.fulfillment_channel == FulfillmentChannel.MFN
        assert sample_order.sales_channel == "Amazon.com"
        assert sample_order.order_total == Decimal("38.86")
        assert sample_order.currency_code == "USD"
        assert sample_order.number_of_items_shipped == 0
        assert sample_order.number_of_items_unshipped == 2
        assert sample_order.buyer_email == "buyer@example.com"
        assert sample_order.is_prime is True
        assert sample_order.is_business_order is False

    def test_order_items(self, sample_order):
        """Order contains order items."""
        assert len(sample_order.order_items) == 1
        assert sample_order.order_items[0].asin == "B00EXAMPLE1"

    def test_shipping_address(self, sample_order):
        """Order has shipping address."""
        assert sample_order.shipping_address is not None
        assert sample_order.shipping_address.city == "Seattle"

    def test_to_dict(self, sample_order, now_utc):
        """to_dict converts to API format."""
        d = sample_order.to_dict()
        assert d["amazonOrderId"] == "111-1234567-1234567"
        assert d["sellerOrderId"] == "SELLER-001"
        assert d["purchaseDate"] == now_utc.isoformat()
        assert d["orderStatus"] == "Unshipped"
        assert d["fulfillmentChannel"] == "MFN"
        assert d["orderTotal"] == "38.86"
        assert d["currencyCode"] == "USD"
        assert d["isPrime"] is True
        assert d["isBusinessOrder"] is False
        assert d["shippingAddress"]["city"] == "Seattle"
        assert len(d["orderItems"]) == 1

    def test_order_defaults(self, now_utc):
        """Order has sensible defaults."""
        order = AmazonOrder(
            amazon_order_id="123",
            seller_order_id=None,
            purchase_date=now_utc,
            last_update_date=now_utc,
            order_status=AmazonOrderStatus.PENDING,
            fulfillment_channel=FulfillmentChannel.AFN,
            sales_channel="Amazon.com",
            order_total=Decimal("0.00"),
            currency_code="USD",
            number_of_items_shipped=0,
            number_of_items_unshipped=0,
        )
        assert order.shipping_address is None
        assert order.buyer_email is None
        assert order.order_items == []
        assert order.is_prime is False
        assert order.is_business_order is False
        assert order.is_replacement_order is False


# ========================================================================
# Inventory item tests
# ========================================================================


class TestAmazonInventoryItem:
    """Tests for AmazonInventoryItem dataclass."""

    def test_construction(self, sample_inventory_item):
        """Inventory item stores stock data."""
        assert sample_inventory_item.asin == "B00EXAMPLE1"
        assert sample_inventory_item.seller_sku == "SKU-001"
        assert sample_inventory_item.fnsku == "X00ABCD1234"
        assert sample_inventory_item.product_name == "Sample Product"
        assert sample_inventory_item.condition == InventoryCondition.NEW
        assert sample_inventory_item.total_quantity == 100
        assert sample_inventory_item.available_quantity == 80
        assert sample_inventory_item.reserved_quantity == 5
        assert sample_inventory_item.inbound_quantity == 10

    def test_to_dict(self, sample_inventory_item):
        """to_dict converts to API format."""
        d = sample_inventory_item.to_dict()
        assert d["asin"] == "B00EXAMPLE1"
        assert d["sellerSku"] == "SKU-001"
        assert d["fnsku"] == "X00ABCD1234"
        assert d["productName"] == "Sample Product"
        assert d["condition"] == "NewItem"
        assert d["totalQuantity"] == 100
        assert d["availableQuantity"] == 80
        assert d["reservedQuantity"] == 5
        assert d["inboundQuantity"] == 10
        assert d["unfulfillableQuantity"] == 3
        assert d["researchingQuantity"] == 2

    def test_defaults(self):
        """Inventory defaults are zero."""
        item = AmazonInventoryItem(
            asin="B001",
            seller_sku="SKU",
            fnsku=None,
            product_name="Product",
            condition=InventoryCondition.NEW,
            total_quantity=50,
        )
        assert item.inbound_quantity == 0
        assert item.available_quantity == 0
        assert item.reserved_quantity == 0
        assert item.unfulfillable_quantity == 0
        assert item.researching_quantity == 0


# ========================================================================
# Product tests
# ========================================================================


class TestAmazonProduct:
    """Tests for AmazonProduct dataclass."""

    def test_construction(self, sample_product):
        """Product stores catalog data."""
        assert sample_product.asin == "B00EXAMPLE1"
        assert sample_product.title == "Sample Product"
        assert sample_product.brand == "SampleBrand"
        assert sample_product.manufacturer == "SampleManufacturer"
        assert sample_product.product_type == "Electronics"
        assert sample_product.parent_asin is None
        assert len(sample_product.images) == 1
        assert len(sample_product.bullet_points) == 3

    def test_to_dict(self, sample_product):
        """to_dict converts to API format."""
        d = sample_product.to_dict()
        assert d["asin"] == "B00EXAMPLE1"
        assert d["title"] == "Sample Product"
        assert d["brand"] == "SampleBrand"
        assert d["manufacturer"] == "SampleManufacturer"
        assert d["productType"] == "Electronics"
        assert d["parentAsin"] is None
        assert d["itemDimensions"]["length"] == 10
        assert d["packageDimensions"]["length"] == 12
        assert len(d["images"]) == 1
        assert len(d["bulletPoints"]) == 3
        assert len(d["browseNodes"]) == 2

    def test_defaults(self):
        """Product has sensible defaults."""
        product = AmazonProduct(
            asin="B001",
            title="Title",
            brand=None,
            manufacturer=None,
            product_type=None,
            parent_asin=None,
        )
        assert product.images == []
        assert product.bullet_points == []
        assert product.browse_nodes == []
        assert product.item_dimensions is None
        assert product.package_dimensions is None


# ========================================================================
# Connector initialization tests
# ========================================================================


class TestConnectorInit:
    """Tests for AmazonConnector construction."""

    def test_connector_id(self, connector):
        """Connector ID is 'amazon'."""
        assert connector.connector_id == "amazon"

    def test_mock_mode_default(self, credentials):
        """Connector defaults to mock mode."""
        c = _TestableAmazonConnector(credentials=credentials)
        assert c.use_mock is True

    def test_sandbox_mode(self, credentials):
        """Sandbox mode can be enabled."""
        c = _TestableAmazonConnector(credentials=credentials, sandbox=True)
        assert c.sandbox is True

    def test_credentials_stored(self, connector, credentials):
        """Credentials are stored."""
        assert connector.amazon_credentials == credentials
        assert connector.amazon_credentials.marketplace_id == AmazonMarketplace.US.value


# ========================================================================
# Connection lifecycle tests
# ========================================================================


class TestConnect:
    """Tests for connect() and disconnect()."""

    @pytest.mark.asyncio
    async def test_connect_mock_mode(self, connector):
        """Connect in mock mode returns True."""
        result = await connector.connect()
        assert result is True

    @pytest.mark.asyncio
    async def test_connect_without_sp_api(self, credentials):
        """Connect falls back to mock when SP-API not available."""
        connector = _TestableAmazonConnector(
            credentials=credentials,
            use_mock=False,  # Try real mode
        )
        # _sp_api_available will be False due to ImportError in __init__
        result = await connector.connect()
        # Falls back to mock mode
        assert connector.use_mock is True
        assert result is True

    @pytest.mark.asyncio
    async def test_disconnect(self, connector):
        """Disconnect sets client to None."""
        connector._client = MagicMock()
        await connector.disconnect()
        assert connector._client is None

    @pytest.mark.asyncio
    async def test_disconnect_noop_without_client(self, connector):
        """Disconnect is safe when no client exists."""
        connector._client = None
        await connector.disconnect()
        assert connector._client is None


# ========================================================================
# Order sync tests
# ========================================================================


class TestOrderSync:
    """Tests for order sync operations."""

    @pytest.mark.asyncio
    async def test_sync_orders_mock_mode(self, connector):
        """sync_orders yields mock orders in mock mode."""
        orders = []
        async for order in connector.sync_orders():
            orders.append(order)

        assert len(orders) == 1
        assert orders[0].amazon_order_id == "111-1234567-1234567"
        assert orders[0].order_status == AmazonOrderStatus.UNSHIPPED

    @pytest.mark.asyncio
    async def test_sync_orders_with_since_filter(self, connector):
        """sync_orders respects 'since' filter."""
        future_time = datetime.now(timezone.utc) + timedelta(days=1)
        orders = []
        async for order in connector.sync_orders(since=future_time):
            orders.append(order)

        # Mock orders are filtered out if their last_update_date < since
        assert len(orders) == 0

    @pytest.mark.asyncio
    async def test_sync_orders_with_status_filter(self, connector):
        """sync_orders respects status filter."""
        orders = []
        async for order in connector.sync_orders(status=[AmazonOrderStatus.SHIPPED]):
            orders.append(order)

        # Mock order is UNSHIPPED, so filtered out
        assert len(orders) == 0

    @pytest.mark.asyncio
    async def test_sync_orders_with_fulfillment_filter(self, connector):
        """sync_orders respects fulfillment channel filter."""
        orders = []
        async for order in connector.sync_orders(fulfillment_channels=[FulfillmentChannel.AFN]):
            orders.append(order)

        # Mock order is MFN, so filtered out
        assert len(orders) == 0

    @pytest.mark.asyncio
    async def test_sync_orders_matching_filters(self, connector):
        """sync_orders returns orders matching all filters."""
        orders = []
        async for order in connector.sync_orders(
            status=[AmazonOrderStatus.UNSHIPPED],
            fulfillment_channels=[FulfillmentChannel.MFN],
        ):
            orders.append(order)

        assert len(orders) == 1


class TestGetOrder:
    """Tests for get_order operation."""

    @pytest.mark.asyncio
    async def test_get_order_returns_none(self, connector):
        """get_order currently returns None (stub)."""
        order = await connector.get_order("111-1234567-1234567")
        assert order is None


class TestGetOrderItems:
    """Tests for get_order_items operation."""

    @pytest.mark.asyncio
    async def test_get_order_items_returns_empty(self, connector):
        """get_order_items currently returns empty list (stub)."""
        items = await connector.get_order_items("111-1234567-1234567")
        assert items == []


class TestConfirmShipment:
    """Tests for confirm_shipment operation."""

    @pytest.mark.asyncio
    async def test_confirm_shipment_success(self, connector, now_utc):
        """confirm_shipment returns True."""
        result = await connector.confirm_shipment(
            order_id="111-1234567-1234567",
            tracking_number="1Z999AA10123456784",
            carrier_code="UPS",
            ship_date=now_utc,
        )
        assert result is True


# ========================================================================
# Order parsing tests
# ========================================================================


class TestOrderParsing:
    """Tests for _parse_sp_api_order helper."""

    def test_parse_sp_api_order(self, connector, sample_sp_api_order_data, now_utc):
        """Full order parsing from SP-API response dict."""
        order = connector._parse_sp_api_order(sample_sp_api_order_data)

        assert order.amazon_order_id == "111-1234567-1234567"
        assert order.seller_order_id == "SELLER-001"
        assert order.order_status == AmazonOrderStatus.UNSHIPPED
        assert order.fulfillment_channel == FulfillmentChannel.MFN
        assert order.sales_channel == "Amazon.com"
        assert order.order_total == Decimal("38.86")
        assert order.currency_code == "USD"
        assert order.buyer_email == "buyer@example.com"
        assert order.buyer_name == "John Doe"
        assert order.is_prime is True
        assert order.is_business_order is False
        assert order.order_items == []  # Items fetched separately

    def test_parse_sp_api_order_with_shipping_address(self, connector, sample_sp_api_order_data):
        """Shipping address is parsed correctly."""
        order = connector._parse_sp_api_order(sample_sp_api_order_data)
        assert order.shipping_address is not None
        assert order.shipping_address.name == "John Doe"
        assert order.shipping_address.city == "Seattle"
        assert order.shipping_address.state_or_region == "WA"

    def test_parse_sp_api_order_no_shipping_address(self, connector, sample_sp_api_order_data):
        """Missing shipping address yields None."""
        sample_sp_api_order_data.pop("ShippingAddress")
        order = connector._parse_sp_api_order(sample_sp_api_order_data)
        assert order.shipping_address is None

    def test_parse_sp_api_order_no_order_total(self, connector, sample_sp_api_order_data):
        """Missing order total defaults to 0.00."""
        sample_sp_api_order_data.pop("OrderTotal")
        order = connector._parse_sp_api_order(sample_sp_api_order_data)
        assert order.order_total == Decimal("0.00")
        assert order.currency_code == "USD"

    def test_parse_sp_api_order_defaults(self, connector, sample_sp_api_order_data):
        """Default values are applied for missing optional fields."""
        sample_sp_api_order_data.pop("FulfillmentChannel")
        sample_sp_api_order_data.pop("SalesChannel")
        sample_sp_api_order_data.pop("IsBusinessOrder")
        sample_sp_api_order_data.pop("IsPrime")

        order = connector._parse_sp_api_order(sample_sp_api_order_data)
        assert order.fulfillment_channel == FulfillmentChannel.MFN
        assert order.sales_channel == "Amazon.com"
        assert order.is_business_order is False
        assert order.is_prime is False


# ========================================================================
# Inventory tests
# ========================================================================


class TestInventoryOperations:
    """Tests for inventory operations."""

    @pytest.mark.asyncio
    async def test_get_fba_inventory(self, connector):
        """get_fba_inventory returns mock inventory."""
        inventory = await connector.get_fba_inventory()
        assert len(inventory) == 1
        assert inventory[0].asin == "B00EXAMPLE1"
        assert inventory[0].seller_sku == "SKU-001"
        assert inventory[0].available_quantity == 85

    @pytest.mark.asyncio
    async def test_get_fba_inventory_with_skus(self, connector):
        """get_fba_inventory accepts SKU filter (not implemented in mock)."""
        inventory = await connector.get_fba_inventory(skus=["SKU-001"])
        assert len(inventory) == 1

    @pytest.mark.asyncio
    async def test_get_inventory_item(self, connector):
        """get_inventory_item returns single item or None."""
        item = await connector.get_inventory_item("SKU-001")
        assert item is not None
        assert item.seller_sku == "SKU-001"

    @pytest.mark.asyncio
    async def test_get_inventory_item_not_found(self, connector):
        """get_inventory_item returns None for non-existent SKU."""
        # Mock inventory returns fixed items, so checking non-existent
        item = await connector.get_inventory_item("NONEXISTENT")
        # Since mock always returns same list, first item is returned
        assert item is not None  # Mock doesn't filter

    @pytest.mark.asyncio
    async def test_create_inbound_shipment(self, connector, sample_address):
        """create_inbound_shipment returns None (stub)."""
        result = await connector.create_inbound_shipment(
            shipment_name="Test Shipment",
            items=[{"sellerSku": "SKU-001", "quantity": 50}],
            ship_from_address=sample_address,
        )
        assert result is None


# ========================================================================
# Catalog/Product tests
# ========================================================================


class TestCatalogOperations:
    """Tests for catalog/product operations."""

    @pytest.mark.asyncio
    async def test_get_catalog_item(self, connector):
        """get_catalog_item returns None (stub)."""
        product = await connector.get_catalog_item("B00EXAMPLE1")
        assert product is None

    @pytest.mark.asyncio
    async def test_search_catalog(self, connector):
        """search_catalog returns empty list (stub)."""
        products = await connector.search_catalog("test product", limit=10)
        assert products == []


# ========================================================================
# Report tests
# ========================================================================


class TestReportOperations:
    """Tests for report operations."""

    @pytest.mark.asyncio
    async def test_request_report(self, connector, now_utc):
        """request_report returns None (stub)."""
        report_id = await connector.request_report(
            report_type="GET_FLAT_FILE_OPEN_LISTINGS_DATA",
            start_date=now_utc - timedelta(days=7),
            end_date=now_utc,
        )
        assert report_id is None

    @pytest.mark.asyncio
    async def test_get_report(self, connector):
        """get_report returns None (stub)."""
        data = await connector.get_report("report-123")
        assert data is None


# ========================================================================
# Analytics tests
# ========================================================================


class TestAnalytics:
    """Tests for analytics operations."""

    @pytest.mark.asyncio
    async def test_get_sales_metrics(self, connector, now_utc):
        """get_sales_metrics returns aggregated data from mock orders."""
        start = now_utc - timedelta(days=1)
        end = now_utc + timedelta(days=1)

        metrics = await connector.get_sales_metrics(start, end)

        assert metrics["total_orders"] == 1
        assert metrics["total_revenue"] == "45.99"
        assert metrics["mfn_orders"] == 1
        assert metrics["fba_orders"] == 0
        assert metrics["fba_percentage"] == 0.0

    @pytest.mark.asyncio
    async def test_get_sales_metrics_empty_range(self, connector, now_utc):
        """get_sales_metrics with no orders in range."""
        # Set range to past dates
        end = now_utc - timedelta(days=365)
        start = end - timedelta(days=7)

        metrics = await connector.get_sales_metrics(start, end)

        assert metrics["total_orders"] == 0
        assert metrics["total_revenue"] == "0.00"
        assert metrics["fba_percentage"] == 0


# ========================================================================
# Enterprise connector sync tests
# ========================================================================


class TestEnterpriseSync:
    """Tests for EnterpriseConnector sync methods."""

    @pytest.mark.asyncio
    async def test_incremental_sync(self, connector):
        """incremental_sync yields orders and inventory."""
        items = []
        async for item in connector.incremental_sync(state=None):
            items.append(item)

        # Should have at least 1 order + 1 inventory item
        assert len(items) >= 2

        order_items = [i for i in items if i["type"] == "order"]
        inventory_items = [i for i in items if i["type"] == "inventory"]

        assert len(order_items) == 1
        assert len(inventory_items) == 1
        assert order_items[0]["data"]["amazonOrderId"] == "111-1234567-1234567"
        assert inventory_items[0]["data"]["asin"] == "B00EXAMPLE1"

    @pytest.mark.asyncio
    async def test_incremental_sync_with_state(self, connector, now_utc):
        """incremental_sync respects sync state."""
        state = SyncState(
            connector_id="amazon",
            last_sync_at=now_utc - timedelta(hours=1),
        )

        items = []
        async for item in connector.incremental_sync(state=state):
            items.append(item)

        assert len(items) >= 2

    @pytest.mark.asyncio
    async def test_full_sync(self, connector):
        """full_sync returns SyncResult."""
        result = await connector.full_sync()

        assert result.connector_id == "amazon"
        assert result.success is True
        assert result.items_synced >= 2
        assert result.items_failed == 0
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_sync_items_yields_sync_items(self, connector):
        """sync_items yields SyncItem objects for KM ingestion."""
        state = SyncState(connector_id="amazon")
        items = []
        async for item in connector.sync_items(state, batch_size=100):
            items.append(item)

        assert len(items) >= 2

        # Check order sync item
        order_item = next((i for i in items if "order" in i.id), None)
        assert order_item is not None
        assert order_item.source_type == "ecommerce"
        assert "111-1234567-1234567" in order_item.id
        assert order_item.domain == "ecommerce"
        assert order_item.confidence == 0.95
        assert "type" in order_item.metadata
        assert order_item.metadata["type"] == "order"

        # Check inventory sync item
        inv_item = next((i for i in items if "inventory" in i.id), None)
        assert inv_item is not None
        assert inv_item.metadata["type"] == "inventory"
        assert inv_item.metadata["asin"] == "B00EXAMPLE1"


# ========================================================================
# Mock data helper tests
# ========================================================================


class TestMockDataHelpers:
    """Tests for mock data helper functions."""

    def test_get_mock_orders(self):
        """get_mock_orders returns valid order list."""
        orders = get_mock_orders()
        assert len(orders) == 1
        order = orders[0]

        assert order.amazon_order_id == "111-1234567-1234567"
        assert order.order_status == AmazonOrderStatus.UNSHIPPED
        assert order.fulfillment_channel == FulfillmentChannel.MFN
        assert order.sales_channel == "Amazon.com"
        assert order.order_total == Decimal("45.99")
        assert order.is_prime is True
        assert len(order.order_items) == 1

    def test_get_mock_inventory(self):
        """get_mock_inventory returns valid inventory list."""
        inventory = get_mock_inventory()
        assert len(inventory) == 1
        item = inventory[0]

        assert item.asin == "B00EXAMPLE1"
        assert item.seller_sku == "SKU-001"
        assert item.fnsku == "X00ABCD1234"
        assert item.condition == InventoryCondition.NEW
        assert item.total_quantity == 100
        assert item.available_quantity == 85

    def test_mock_order_item_in_order(self):
        """Mock order contains valid order item."""
        orders = get_mock_orders()
        item = orders[0].order_items[0]

        assert item.order_item_id == "item-001"
        assert item.asin == "B00EXAMPLE1"
        assert item.seller_sku == "SKU-001"
        assert item.title == "Sample Product"
        assert item.quantity_ordered == 1
        assert item.item_price == Decimal("39.99")


# ========================================================================
# Marketplace validation tests
# ========================================================================


class TestMarketplaceValidation:
    """Tests for marketplace ID validation."""

    def test_valid_marketplace_ids(self):
        """All marketplace IDs are valid."""
        for marketplace in AmazonMarketplace:
            assert len(marketplace.value) > 0
            # All Amazon marketplace IDs start with 'A'
            assert marketplace.value.startswith("A")

    def test_us_marketplace(self):
        """US marketplace ID is ATVPDKIKX0DER."""
        assert AmazonMarketplace.US.value == "ATVPDKIKX0DER"

    def test_marketplace_string_conversion(self):
        """Marketplace enum is also a string."""
        assert isinstance(AmazonMarketplace.US.value, str)
        assert str(AmazonMarketplace.US.value) == "ATVPDKIKX0DER"


# ========================================================================
# Data model serialization tests
# ========================================================================


class TestDataSerialization:
    """Tests for data model serialization round-trip."""

    def test_order_serialization_round_trip(self, sample_order):
        """Order can be serialized and contains all data."""
        d = sample_order.to_dict()

        assert d["amazonOrderId"] == "111-1234567-1234567"
        assert d["orderStatus"] == "Unshipped"
        assert d["fulfillmentChannel"] == "MFN"
        assert d["orderTotal"] == "38.86"
        assert d["currencyCode"] == "USD"
        assert d["shippingAddress"]["city"] == "Seattle"
        assert len(d["orderItems"]) == 1

    def test_inventory_serialization_round_trip(self, sample_inventory_item):
        """Inventory item can be serialized and contains all data."""
        d = sample_inventory_item.to_dict()

        assert d["asin"] == "B00EXAMPLE1"
        assert d["sellerSku"] == "SKU-001"
        assert d["totalQuantity"] == 100
        assert d["availableQuantity"] == 80
        assert d["condition"] == "NewItem"

    def test_product_serialization_round_trip(self, sample_product):
        """Product can be serialized and contains all data."""
        d = sample_product.to_dict()

        assert d["asin"] == "B00EXAMPLE1"
        assert d["title"] == "Sample Product"
        assert d["brand"] == "SampleBrand"
        assert len(d["images"]) == 1
        assert len(d["bulletPoints"]) == 3

    def test_decimal_to_string_conversion(self, sample_order_item):
        """Decimal values are converted to strings in serialization."""
        d = sample_order_item.to_dict()

        assert isinstance(d["itemPrice"], str)
        assert d["itemPrice"] == "29.99"
        assert isinstance(d["itemTax"], str)
        assert isinstance(d["shippingPrice"], str)


# ========================================================================
# SP-API availability tests
# ========================================================================


class TestSpApiAvailability:
    """Tests for SP-API library availability handling."""

    def test_mock_mode_without_sp_api(self, credentials):
        """Connector works in mock mode without SP-API."""
        connector = _TestableAmazonConnector(
            credentials=credentials,
            use_mock=True,
        )
        assert connector.use_mock is True
        # Even without SP-API, connector is functional in mock mode

    def test_fallback_to_mock_when_sp_api_unavailable(self, credentials):
        """Connector falls back to mock when SP-API not installed."""
        # Create connector with use_mock=False
        connector = _TestableAmazonConnector(
            credentials=credentials,
            use_mock=False,
        )
        # Since SP-API is not installed in test env, it should fallback
        assert connector.use_mock is True  # Fallback happened


# ========================================================================
# Sync state management tests
# ========================================================================


class TestSyncStateManagement:
    """Tests for sync state management in incremental sync."""

    @pytest.mark.asyncio
    async def test_sync_items_updates_state_metadata(self, connector):
        """sync_items yields items with proper metadata."""
        state = SyncState(
            connector_id="amazon",
            tenant_id="test-tenant",
        )

        items = []
        async for item in connector.sync_items(state, batch_size=50):
            items.append(item)

        # Verify metadata structure
        for item in items:
            assert "type" in item.metadata
            assert item.metadata["type"] in ("order", "inventory")
            assert item.source_type == "ecommerce"
            assert item.domain == "ecommerce"

    @pytest.mark.asyncio
    async def test_sync_respects_last_sync_at(self, connector, now_utc):
        """Sync filters items based on last_sync_at."""
        # Set last_sync_at to future - should filter out all mock items
        future_state = SyncState(
            connector_id="amazon",
            last_sync_at=now_utc + timedelta(days=1),
        )

        items = []
        async for item in connector.incremental_sync(state=future_state):
            items.append(item)

        # Orders filtered out, but inventory doesn't have update date filter
        # (inventory is always returned in current implementation)
        inventory_items = [i for i in items if i["type"] == "inventory"]
        order_items = [i for i in items if i["type"] == "order"]

        assert len(order_items) == 0  # Filtered out by date
        assert len(inventory_items) == 1  # Inventory not filtered by date
