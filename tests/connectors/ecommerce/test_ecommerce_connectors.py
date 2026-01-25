"""
Tests for E-commerce Platform Connectors.

Tests for Shopify, Amazon, WooCommerce, and ShipStation connectors.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch


class TestShopifyConnector:
    """Tests for Shopify connector."""

    def test_shopify_credentials(self):
        """Test ShopifyCredentials dataclass."""
        from aragora.connectors.ecommerce.shopify import ShopifyCredentials

        creds = ShopifyCredentials(
            shop_domain="test-store.myshopify.com",
            access_token="shpat_test_token",
            api_version="2024-01",
        )

        assert creds.shop_domain == "test-store.myshopify.com"
        assert creds.access_token == "shpat_test_token"
        assert creds.api_version == "2024-01"

    def test_order_enums(self):
        """Test order status enums."""
        from aragora.connectors.ecommerce.shopify import OrderStatus, PaymentStatus

        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FULFILLED.value == "fulfilled"
        assert PaymentStatus.PAID.value == "paid"
        assert PaymentStatus.REFUNDED.value == "refunded"

    def test_mock_orders(self):
        """Test mock orders generation."""
        from aragora.connectors.ecommerce.shopify import get_mock_orders

        orders = get_mock_orders()

        assert len(orders) >= 1
        order = orders[0]
        assert order.id == "1001"
        assert order.order_number == 1001
        assert order.total_price == Decimal("99.99")
        assert len(order.line_items) >= 1

    def test_mock_products(self):
        """Test mock products generation."""
        from aragora.connectors.ecommerce.shopify import get_mock_products

        products = get_mock_products()

        assert len(products) >= 1
        product = products[0]
        assert product.id == "prod_1"
        assert product.title == "Sample Product"
        assert len(product.variants) >= 1
        assert product.variants[0].price == Decimal("89.99")

    def test_shopify_line_item(self):
        """Test ShopifyLineItem dataclass."""
        from aragora.connectors.ecommerce.shopify import ShopifyLineItem

        item = ShopifyLineItem(
            id="item_123",
            product_id="prod_456",
            variant_id="var_789",
            title="Test Product",
            quantity=2,
            price=Decimal("49.99"),
            sku="TEST-SKU",
        )

        assert item.id == "item_123"
        assert item.quantity == 2
        assert item.price == Decimal("49.99")

    def test_shopify_address(self):
        """Test ShopifyAddress dataclass."""
        from aragora.connectors.ecommerce.shopify import ShopifyAddress

        addr = ShopifyAddress(
            first_name="John",
            last_name="Doe",
            address1="123 Main St",
            city="New York",
            province="NY",
            country="US",
            zip="10001",
        )

        assert addr.first_name == "John"
        assert addr.city == "New York"
        assert addr.country == "US"


class TestAmazonConnector:
    """Tests for Amazon Seller connector."""

    def test_amazon_credentials(self):
        """Test AmazonCredentials dataclass."""
        from aragora.connectors.ecommerce.amazon import AmazonCredentials

        creds = AmazonCredentials(
            client_id="amzn1.application.test",
            client_secret="client_secret",
            refresh_token="refresh_token",
            seller_id="AXXXXXXXXXX",
            marketplace_id="ATVPDKIKX0DER",
        )

        assert creds.client_id == "amzn1.application.test"
        assert creds.seller_id == "AXXXXXXXXXX"

    def test_amazon_marketplace_enum(self):
        """Test AmazonMarketplace enum."""
        from aragora.connectors.ecommerce.amazon import AmazonMarketplace

        assert AmazonMarketplace.US.value == "ATVPDKIKX0DER"

    def test_amazon_order_status_enum(self):
        """Test AmazonOrderStatus enum."""
        from aragora.connectors.ecommerce.amazon import AmazonOrderStatus

        assert AmazonOrderStatus.PENDING.value == "Pending"
        assert AmazonOrderStatus.SHIPPED.value == "Shipped"

    def test_mock_amazon_orders(self):
        """Test mock Amazon orders."""
        from aragora.connectors.ecommerce.amazon import get_mock_orders

        orders = get_mock_orders()

        assert len(orders) >= 1
        order = orders[0]
        assert order.amazon_order_id is not None
        assert order.purchase_date is not None

    def test_mock_amazon_inventory(self):
        """Test mock Amazon inventory."""
        from aragora.connectors.ecommerce.amazon import get_mock_inventory

        inventory = get_mock_inventory()

        assert len(inventory) >= 1
        item = inventory[0]
        assert item.seller_sku is not None


class TestWooCommerceConnector:
    """Tests for WooCommerce connector."""

    def test_woocommerce_credentials(self):
        """Test WooCommerceCredentials dataclass."""
        from aragora.connectors.ecommerce.woocommerce import WooCommerceCredentials

        creds = WooCommerceCredentials(
            store_url="https://store.example.com",
            consumer_key="ck_test_key",
            consumer_secret="cs_test_secret",
        )

        assert creds.store_url == "https://store.example.com"
        assert creds.consumer_key == "ck_test_key"

    def test_woo_order_status_enum(self):
        """Test WooOrderStatus enum."""
        from aragora.connectors.ecommerce.woocommerce import WooOrderStatus

        assert WooOrderStatus.PENDING.value == "pending"
        assert WooOrderStatus.COMPLETED.value == "completed"
        assert WooOrderStatus.PROCESSING.value == "processing"

    def test_woo_product_status_enum(self):
        """Test WooProductStatus enum."""
        from aragora.connectors.ecommerce.woocommerce import WooProductStatus

        assert WooProductStatus.PUBLISH.value == "publish"
        assert WooProductStatus.DRAFT.value == "draft"

    def test_mock_woo_orders(self):
        """Test mock WooCommerce orders."""
        from aragora.connectors.ecommerce.woocommerce import get_mock_woo_orders

        orders = get_mock_woo_orders()

        assert len(orders) >= 1
        order = orders[0]
        assert order.id is not None

    def test_mock_woo_products(self):
        """Test mock WooCommerce products."""
        from aragora.connectors.ecommerce.woocommerce import get_mock_woo_products

        products = get_mock_woo_products()

        assert len(products) >= 1
        product = products[0]
        assert product.id is not None
        assert product.name is not None


class TestShipStationConnector:
    """Tests for ShipStation connector."""

    def test_shipstation_credentials(self):
        """Test ShipStationCredentials dataclass."""
        from aragora.connectors.ecommerce.shipstation import ShipStationCredentials

        creds = ShipStationCredentials(
            api_key="api_key",
            api_secret="api_secret",
        )

        assert creds.api_key == "api_key"
        assert creds.api_secret == "api_secret"

    def test_shipment_status_enum(self):
        """Test ShipmentStatus enum."""
        from aragora.connectors.ecommerce.shipstation import ShipmentStatus

        assert ShipmentStatus.LABEL_CREATED.value == "label_created"
        assert ShipmentStatus.DELIVERED.value == "delivered"

    def test_mock_shipstation_order(self):
        """Test mock ShipStation order."""
        from aragora.connectors.ecommerce.shipstation import get_mock_order

        order = get_mock_order()

        assert order.order_id is not None
        assert order.order_number is not None

    def test_mock_shipment(self):
        """Test mock ShipStation shipment."""
        from aragora.connectors.ecommerce.shipstation import get_mock_shipment

        shipment = get_mock_shipment()

        assert shipment.shipment_id is not None


class TestEcommercePackageImports:
    """Test that e-commerce package imports work correctly."""

    def test_shopify_imports(self):
        """Test Shopify can be imported from package."""
        from aragora.connectors.ecommerce import (
            ShopifyConnector,
            ShopifyCredentials,
            ShopifyOrder,
            ShopifyProduct,
            OrderStatus,
            PaymentStatus,
        )

        assert ShopifyConnector is not None
        assert ShopifyCredentials is not None

    def test_amazon_imports(self):
        """Test Amazon can be imported from package."""
        from aragora.connectors.ecommerce import (
            AmazonConnector,
            AmazonCredentials,
            AmazonMarketplace,
            AmazonOrder,
            AmazonOrderStatus,
        )

        assert AmazonConnector is not None
        assert AmazonCredentials is not None

    def test_woocommerce_imports(self):
        """Test WooCommerce can be imported from package."""
        from aragora.connectors.ecommerce import (
            WooCommerceConnector,
            WooCommerceCredentials,
            WooOrder,
            WooProduct,
            WooOrderStatus,
        )

        assert WooCommerceConnector is not None
        assert WooCommerceCredentials is not None

    def test_shipstation_imports(self):
        """Test ShipStation can be imported from package."""
        from aragora.connectors.ecommerce import (
            ShipStationConnector,
            ShipStationCredentials,
            Shipment,
            ShipmentStatus,
        )

        assert ShipStationConnector is not None
        assert ShipStationCredentials is not None


class TestInventoryPolicy:
    """Test inventory policy enum."""

    def test_inventory_policy_values(self):
        """Test InventoryPolicy enum values."""
        from aragora.connectors.ecommerce.shopify import InventoryPolicy

        assert InventoryPolicy.DENY.value == "deny"
        assert InventoryPolicy.CONTINUE.value == "continue"


class TestShopifyConnectorParsing:
    """Tests for Shopify connector parsing methods using a concrete test subclass."""

    @pytest.fixture
    def connector(self):
        """Create a test-friendly Shopify connector instance."""
        from aragora.connectors.ecommerce.shopify import ShopifyConnector, ShopifyCredentials
        from aragora.connectors.base import Evidence
        from aragora.reasoning.provenance import SourceType
        from typing import AsyncIterator

        # Create a concrete subclass for testing
        class TestableShopifyConnector(ShopifyConnector):
            """Concrete implementation for testing."""

            @property
            def name(self) -> str:
                return "test_shopify"

            @property
            def source_type(self) -> SourceType:
                return SourceType.EXTERNAL_API

            async def search(self, query: str, **kwargs) -> list[Evidence]:
                return []

            async def fetch(self, evidence_id: str) -> Evidence | None:
                return None

            async def sync_items(self, **kwargs) -> AsyncIterator:
                return
                yield  # Make it a generator

        credentials = ShopifyCredentials(
            shop_domain="test-store.myshopify.com",
            access_token="shpat_test_token",
            api_version="2024-01",
        )
        return TestableShopifyConnector(credentials)

    def test_base_url(self, connector):
        """Test base URL is correctly constructed."""
        assert connector.base_url == "https://test-store.myshopify.com/admin/api/2024-01"

    def test_parse_link_header_with_next(self, connector):
        """Test Link header parsing for pagination."""
        link_header = (
            '<https://test.myshopify.com/admin/api/2024-01/orders.json?page_info=abc123>; rel="next", '
            '<https://test.myshopify.com/admin/api/2024-01/orders.json?page_info=xyz789>; rel="previous"'
        )
        result = connector._parse_link_header(link_header)
        assert result == "abc123"

    def test_parse_link_header_no_next(self, connector):
        """Test Link header parsing when no next page."""
        link_header = '<https://test.myshopify.com/admin/api/2024-01/orders.json?page_info=xyz789>; rel="previous"'
        result = connector._parse_link_header(link_header)
        assert result is None

    def test_parse_link_header_none(self, connector):
        """Test Link header parsing with None input."""
        result = connector._parse_link_header(None)
        assert result is None

    def test_parse_order(self, connector):
        """Test order parsing from API response."""
        order_data = {
            "id": 123456,
            "order_number": 1001,
            "name": "#1001",
            "email": "customer@example.com",
            "created_at": "2024-01-15T10:30:00Z",
            "updated_at": "2024-01-15T10:35:00Z",
            "total_price": "99.99",
            "subtotal_price": "89.99",
            "total_tax": "10.00",
            "total_discounts": "0.00",
            "currency": "USD",
            "financial_status": "paid",
            "fulfillment_status": None,
            "line_items": [
                {
                    "id": 111,
                    "product_id": 222,
                    "variant_id": 333,
                    "title": "Test Product",
                    "quantity": 2,
                    "price": "44.99",
                    "sku": "TEST-SKU",
                }
            ],
            "tags": "tag1, tag2",
        }

        order = connector._parse_order(order_data)

        assert order.id == "123456"
        assert order.order_number == 1001
        assert order.total_price == Decimal("99.99")
        assert len(order.line_items) == 1
        assert order.line_items[0].quantity == 2
        assert order.tags == ["tag1", "tag2"]

    def test_parse_address(self, connector):
        """Test address parsing."""
        address_data = {
            "first_name": "John",
            "last_name": "Doe",
            "address1": "123 Main St",
            "city": "New York",
            "province": "NY",
            "country": "United States",
            "zip": "10001",
        }

        address = connector._parse_address(address_data)

        assert address.first_name == "John"
        assert address.city == "New York"
        assert address.zip == "10001"

    def test_parse_product(self, connector):
        """Test product parsing from API response."""
        product_data = {
            "id": 555,
            "title": "Test Product",
            "handle": "test-product",
            "vendor": "Test Vendor",
            "product_type": "Test Type",
            "status": "active",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-15T00:00:00Z",
            "published_at": "2024-01-02T00:00:00Z",
            "body_html": "<p>Product description</p>",
            "tags": "tag1, tag2",
            "variants": [
                {
                    "id": 666,
                    "title": "Default",
                    "price": "49.99",
                    "sku": "TEST-SKU",
                    "inventory_quantity": 10,
                    "inventory_policy": "deny",
                }
            ],
            "images": [{"src": "https://example.com/image.jpg"}],
        }

        product = connector._parse_product(product_data)

        assert product.id == "555"
        assert product.title == "Test Product"
        assert len(product.variants) == 1
        assert product.variants[0].price == Decimal("49.99")
        assert product.images == ["https://example.com/image.jpg"]

    def test_parse_customer(self, connector):
        """Test customer parsing from API response."""
        customer_data = {
            "id": 777,
            "email": "customer@example.com",
            "first_name": "Jane",
            "last_name": "Smith",
            "phone": "+1234567890",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-15T00:00:00Z",
            "orders_count": 5,
            "total_spent": "499.95",
            "verified_email": True,
            "accepts_marketing": False,
            "tax_exempt": False,
            "tags": "vip, wholesale",
        }

        customer = connector._parse_customer(customer_data)

        assert customer.id == "777"
        assert customer.email == "customer@example.com"
        assert customer.full_name == "Jane Smith"
        assert customer.orders_count == 5
        assert customer.total_spent == Decimal("499.95")
        assert customer.tags == ["vip", "wholesale"]


class TestShopifyConnectorDataclassToDict:
    """Test dataclass to_dict methods for API serialization."""

    def test_order_to_dict(self):
        """Test ShopifyOrder.to_dict()."""
        from aragora.connectors.ecommerce.shopify import (
            ShopifyOrder,
            ShopifyLineItem,
            OrderStatus,
            PaymentStatus,
        )

        now = datetime.now(timezone.utc)
        order = ShopifyOrder(
            id="1001",
            order_number=1001,
            name="#1001",
            email="test@example.com",
            created_at=now,
            updated_at=now,
            total_price=Decimal("99.99"),
            subtotal_price=Decimal("89.99"),
            total_tax=Decimal("10.00"),
            total_discounts=Decimal("0.00"),
            currency="USD",
            financial_status=PaymentStatus.PAID,
            fulfillment_status=OrderStatus.PENDING,
            line_items=[
                ShopifyLineItem(
                    id="li_1",
                    product_id="prod_1",
                    variant_id="var_1",
                    title="Test",
                    quantity=1,
                    price=Decimal("89.99"),
                )
            ],
        )

        result = order.to_dict()

        assert result["id"] == "1001"
        assert result["totalPrice"] == "99.99"
        assert result["financialStatus"] == "paid"
        assert result["fulfillmentStatus"] == "pending"
        assert len(result["lineItems"]) == 1

    def test_product_to_dict(self):
        """Test ShopifyProduct.to_dict()."""
        from aragora.connectors.ecommerce.shopify import ShopifyProduct, ShopifyVariant

        now = datetime.now(timezone.utc)
        product = ShopifyProduct(
            id="prod_1",
            title="Test Product",
            handle="test-product",
            vendor="Test Vendor",
            product_type="Widget",
            status="active",
            created_at=now,
            updated_at=now,
            published_at=now,
            variants=[
                ShopifyVariant(
                    id="var_1",
                    product_id="prod_1",
                    title="Default",
                    price=Decimal("49.99"),
                    sku="SKU-001",
                )
            ],
        )

        result = product.to_dict()

        assert result["id"] == "prod_1"
        assert result["title"] == "Test Product"
        assert len(result["variants"]) == 1
        assert result["variants"][0]["price"] == "49.99"

    def test_variant_to_dict(self):
        """Test ShopifyVariant.to_dict()."""
        from aragora.connectors.ecommerce.shopify import ShopifyVariant, InventoryPolicy

        variant = ShopifyVariant(
            id="var_1",
            product_id="prod_1",
            title="Large",
            price=Decimal("59.99"),
            sku="SKU-L",
            inventory_quantity=25,
            inventory_policy=InventoryPolicy.DENY,
            compare_at_price=Decimal("69.99"),
            weight=1.5,
            weight_unit="kg",
            option1="Large",
        )

        result = variant.to_dict()

        assert result["id"] == "var_1"
        assert result["price"] == "59.99"
        assert result["compareAtPrice"] == "69.99"
        assert result["inventoryPolicy"] == "deny"
        assert result["option1"] == "Large"

    def test_customer_to_dict(self):
        """Test ShopifyCustomer.to_dict()."""
        from aragora.connectors.ecommerce.shopify import ShopifyCustomer

        now = datetime.now(timezone.utc)
        customer = ShopifyCustomer(
            id="cust_1",
            email="customer@example.com",
            first_name="John",
            last_name="Doe",
            phone="+1234567890",
            created_at=now,
            updated_at=now,
            orders_count=10,
            total_spent=Decimal("999.99"),
            tags=["vip"],
        )

        result = customer.to_dict()

        assert result["id"] == "cust_1"
        assert result["fullName"] == "John Doe"
        assert result["totalSpent"] == "999.99"
        assert result["ordersCount"] == 10

    def test_address_to_dict(self):
        """Test ShopifyAddress.to_dict()."""
        from aragora.connectors.ecommerce.shopify import ShopifyAddress

        address = ShopifyAddress(
            first_name="Jane",
            last_name="Smith",
            company="ACME Inc",
            address1="456 Oak Ave",
            city="Los Angeles",
            province="CA",
            country="US",
            zip="90001",
        )

        result = address.to_dict()

        assert result["firstName"] == "Jane"
        assert result["company"] == "ACME Inc"
        assert result["city"] == "Los Angeles"

    def test_inventory_level_to_dict(self):
        """Test ShopifyInventoryLevel.to_dict()."""
        from aragora.connectors.ecommerce.shopify import ShopifyInventoryLevel

        now = datetime.now(timezone.utc)
        level = ShopifyInventoryLevel(
            inventory_item_id="inv_1",
            location_id="loc_1",
            available=50,
            updated_at=now,
        )

        result = level.to_dict()

        assert result["inventoryItemId"] == "inv_1"
        assert result["locationId"] == "loc_1"
        assert result["available"] == 50


class TestAmazonConnectorAsync:
    """Async tests for Amazon connector with mocked API responses."""

    @pytest.fixture
    def credentials(self):
        """Create test credentials."""
        from aragora.connectors.ecommerce.amazon import AmazonCredentials, AmazonMarketplace

        return AmazonCredentials(
            refresh_token="test_refresh_token",
            client_id="amzn1.application.test",
            client_secret="test_secret",
            marketplace_id=AmazonMarketplace.US.value,
            seller_id="AXXXXXXXXXX",
        )

    def test_amazon_order_item_to_dict(self):
        """Test AmazonOrderItem.to_dict()."""
        from aragora.connectors.ecommerce.amazon import AmazonOrderItem

        item = AmazonOrderItem(
            order_item_id="item_1",
            asin="B000TEST123",
            seller_sku="TEST-SKU",
            title="Test Product",
            quantity_ordered=2,
            quantity_shipped=1,
            item_price=Decimal("29.99"),
            item_tax=Decimal("2.99"),
            shipping_price=Decimal("5.99"),
            shipping_tax=Decimal("0.59"),
            promotion_discount=Decimal("0.00"),
        )

        result = item.to_dict()

        assert result["asin"] == "B000TEST123"
        assert result["quantityOrdered"] == 2
        assert result["itemPrice"] == "29.99"

    def test_amazon_order_to_dict(self):
        """Test AmazonOrder.to_dict()."""
        from aragora.connectors.ecommerce.amazon import (
            AmazonOrder,
            AmazonOrderStatus,
            FulfillmentChannel,
        )

        now = datetime.now(timezone.utc)
        order = AmazonOrder(
            amazon_order_id="111-1111111-1111111",
            seller_order_id="SELLER-001",
            purchase_date=now,
            last_update_date=now,
            order_status=AmazonOrderStatus.SHIPPED,
            fulfillment_channel=FulfillmentChannel.AFN,
            sales_channel="Amazon.com",
            order_total=Decimal("99.99"),
            currency_code="USD",
            number_of_items_shipped=2,
            number_of_items_unshipped=0,
            is_prime=True,
        )

        result = order.to_dict()

        assert result["amazonOrderId"] == "111-1111111-1111111"
        assert result["orderStatus"] == "Shipped"
        assert result["isPrime"] is True

    def test_amazon_address_to_dict(self):
        """Test AmazonAddress.to_dict()."""
        from aragora.connectors.ecommerce.amazon import AmazonAddress

        address = AmazonAddress(
            name="John Doe",
            address_line1="123 Main St",
            city="Seattle",
            state_or_region="WA",
            postal_code="98101",
            country_code="US",
        )

        result = address.to_dict()

        assert result["name"] == "John Doe"
        assert result["city"] == "Seattle"
        assert result["countryCode"] == "US"

    def test_amazon_inventory_item_to_dict(self):
        """Test AmazonInventoryItem.to_dict()."""
        from aragora.connectors.ecommerce.amazon import AmazonInventoryItem, InventoryCondition

        item = AmazonInventoryItem(
            asin="B000TEST123",
            seller_sku="TEST-SKU",
            fnsku="X000TEST123",
            product_name="Test Product",
            condition=InventoryCondition.NEW,
            total_quantity=100,
            available_quantity=80,
            reserved_quantity=20,
        )

        result = item.to_dict()

        assert result["sellerSku"] == "TEST-SKU"
        assert result["totalQuantity"] == 100
        assert result["condition"] == "NewItem"
        assert result["availableQuantity"] == 80


class TestShipStationConnectorDataclasses:
    """Tests for ShipStation connector dataclasses."""

    def test_shipstation_order_dataclass(self):
        """Test ShipStationOrder dataclass creation."""
        from aragora.connectors.ecommerce.shipstation import (
            ShipStationOrder,
            OrderItem,
            ShipStationAddress,
            OrderStatus,
        )

        now = datetime.now(timezone.utc)
        order = ShipStationOrder(
            order_id=12345,
            order_number="ORD-001",
            order_key="key_123",
            order_date=now,
            order_status=OrderStatus.AWAITING_SHIPMENT,
            customer_email="customer@example.com",
            items=[
                OrderItem(
                    order_item_id=111,
                    sku="TEST-SKU",
                    name="Test Product",
                    quantity=2,
                    unit_price=Decimal("40.50"),
                )
            ],
            ship_to=ShipStationAddress(
                name="John Doe",
                street1="123 Main St",
                city="New York",
                state="NY",
                postal_code="10001",
                country="US",
            ),
        )

        assert order.order_id == 12345
        assert order.order_number == "ORD-001"
        assert len(order.items) == 1
        assert order.items[0].quantity == 2

    def test_shipstation_address_to_api(self):
        """Test ShipStationAddress.to_api() method."""
        from aragora.connectors.ecommerce.shipstation import ShipStationAddress

        address = ShipStationAddress(
            name="John Doe",
            street1="123 Main St",
            city="New York",
            state="NY",
            postal_code="10001",
            country="US",
        )

        result = address.to_api()

        assert result["name"] == "John Doe"
        assert result["city"] == "New York"
        assert result["postalCode"] == "10001"

    def test_order_item_to_api(self):
        """Test OrderItem.to_api() method."""
        from aragora.connectors.ecommerce.shipstation import OrderItem

        item = OrderItem(
            order_item_id=111,
            sku="TEST-SKU",
            name="Test Product",
            quantity=2,
            unit_price=Decimal("40.50"),
        )

        result = item.to_api()

        assert result["sku"] == "TEST-SKU"
        assert result["name"] == "Test Product"
        assert result["quantity"] == 2


class TestWooCommerceConnectorDataclasses:
    """Tests for WooCommerce connector dataclasses."""

    def test_woo_order_to_dict(self):
        """Test WooOrder.to_dict()."""
        from aragora.connectors.ecommerce.woocommerce import (
            WooOrder,
            WooOrderStatus,
            WooLineItem,
            WooAddress,
        )

        now = datetime.now(timezone.utc)
        order = WooOrder(
            id=1001,
            number="1001",
            order_key="wc_order_abc123",
            status=WooOrderStatus.PROCESSING,
            currency="USD",
            date_created=now,
            date_modified=now,
            total=Decimal("149.99"),
            subtotal=Decimal("139.99"),
            total_tax=Decimal("10.00"),
            shipping_total=Decimal("0.00"),
            discount_total=Decimal("0.00"),
            payment_method="stripe",
            payment_method_title="Credit Card",
            customer_id=42,
            billing=WooAddress(
                first_name="John",
                last_name="Doe",
                address_1="123 Main St",
                city="New York",
                state="NY",
                postcode="10001",
                country="US",
                email="customer@example.com",
            ),
            shipping=WooAddress(
                first_name="John",
                last_name="Doe",
                address_1="123 Main St",
                city="New York",
                state="NY",
                postcode="10001",
                country="US",
            ),
            line_items=[
                WooLineItem(
                    id=1,
                    product_id=101,
                    variation_id=0,
                    name="Test Product",
                    quantity=2,
                    subtotal=Decimal("139.98"),
                    total=Decimal("139.98"),
                    price=Decimal("69.99"),
                    sku="TEST-SKU",
                )
            ],
        )

        result = order.to_dict()

        assert result["id"] == 1001
        assert result["status"] == "processing"
        assert result["total"] == "149.99"
        assert len(result["lineItems"]) == 1

    def test_woo_product_to_dict(self):
        """Test WooProduct.to_dict()."""
        from aragora.connectors.ecommerce.woocommerce import (
            WooProduct,
            WooProductStatus,
            WooProductType,
            WooStockStatus,
        )

        now = datetime.now(timezone.utc)
        product = WooProduct(
            id=101,
            name="Test Product",
            slug="test-product",
            type=WooProductType.SIMPLE,
            status=WooProductStatus.PUBLISH,
            sku="TEST-SKU",
            price=Decimal("49.99"),
            regular_price=Decimal("59.99"),
            sale_price=Decimal("49.99"),
            date_created=now,
            date_modified=now,
            description="<p>Test description</p>",
            short_description="Short desc",
            stock_quantity=25,
            stock_status=WooStockStatus.IN_STOCK,
            manage_stock=True,
            categories=[{"id": 1, "name": "Widgets"}],
            tags=[{"id": 1, "name": "sale"}],
        )

        result = product.to_dict()

        assert result["id"] == 101
        assert result["name"] == "Test Product"
        assert result["status"] == "publish"
        assert result["price"] == "49.99"
        assert result["stockStatus"] == "instock"

    def test_woo_address_to_dict(self):
        """Test WooAddress.to_dict()."""
        from aragora.connectors.ecommerce.woocommerce import WooAddress

        address = WooAddress(
            first_name="Jane",
            last_name="Smith",
            address_1="456 Oak Ave",
            city="Los Angeles",
            state="CA",
            postcode="90001",
            country="US",
            email="jane@example.com",
        )

        result = address.to_dict()

        assert result["firstName"] == "Jane"
        assert result["city"] == "Los Angeles"
        assert result["email"] == "jane@example.com"

    def test_woo_line_item_to_dict(self):
        """Test WooLineItem.to_dict()."""
        from aragora.connectors.ecommerce.woocommerce import WooLineItem

        item = WooLineItem(
            id=1,
            product_id=101,
            variation_id=0,
            name="Test Product",
            quantity=3,
            subtotal=Decimal("89.97"),
            total=Decimal("89.97"),
            price=Decimal("29.99"),
            sku="TEST-123",
        )

        result = item.to_dict()

        assert result["id"] == 1
        assert result["name"] == "Test Product"
        assert result["quantity"] == 3
        assert result["price"] == "29.99"
