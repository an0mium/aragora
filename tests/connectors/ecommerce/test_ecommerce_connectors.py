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
