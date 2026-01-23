"""Tests for WooCommerce connector."""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from aragora.connectors.ecommerce.woocommerce import (
    WooCommerceConnector,
    WooCommerceCredentials,
    WooOrder,
    WooOrderStatus,
    WooProduct,
    WooProductStatus,
    WooProductType,
    WooProductVariation,
    WooCustomer,
    WooAddress,
    WooLineItem,
    WooStockStatus,
    get_mock_woo_orders,
    get_mock_woo_products,
)


@pytest.fixture
def credentials():
    """Test credentials."""
    return WooCommerceCredentials(
        store_url="https://test.example.com",
        consumer_key="ck_test123",
        consumer_secret="cs_test456",
        api_version="wc/v3",
    )


@pytest.fixture
def connector(credentials):
    """Test connector."""
    return WooCommerceConnector(credentials)


@pytest.fixture
def mock_order_data():
    """Mock order API response."""
    return {
        "id": 1001,
        "number": "1001",
        "order_key": "wc_order_test1001",
        "status": "processing",
        "currency": "USD",
        "date_created_gmt": "2024-01-15T10:30:00",
        "date_modified_gmt": "2024-01-15T11:00:00",
        "total": "99.99",
        "subtotal": "89.99",
        "total_tax": "10.00",
        "shipping_total": "5.00",
        "discount_total": "0.00",
        "payment_method": "stripe",
        "payment_method_title": "Credit Card",
        "customer_id": 42,
        "billing": {
            "first_name": "John",
            "last_name": "Doe",
            "email": "john@example.com",
            "phone": "555-1234",
        },
        "shipping": {
            "first_name": "John",
            "last_name": "Doe",
            "city": "New York",
        },
        "line_items": [
            {
                "id": 1,
                "product_id": 101,
                "variation_id": 0,
                "name": "Test Product",
                "quantity": 2,
                "subtotal": "89.99",
                "total": "89.99",
                "sku": "TST-001",
                "price": "44.99",
            }
        ],
        "customer_note": "Test note",
        "date_paid_gmt": "2024-01-15T10:31:00",
        "date_completed_gmt": None,
        "transaction_id": "txn_123",
    }


@pytest.fixture
def mock_product_data():
    """Mock product API response."""
    return {
        "id": 101,
        "name": "Test Product",
        "slug": "test-product",
        "type": "simple",
        "status": "publish",
        "sku": "TST-001",
        "price": "49.99",
        "regular_price": "59.99",
        "sale_price": "49.99",
        "date_created_gmt": "2024-01-01T00:00:00",
        "date_modified_gmt": "2024-01-10T00:00:00",
        "description": "A test product",
        "short_description": "Test",
        "stock_quantity": 25,
        "stock_status": "instock",
        "manage_stock": True,
        "categories": [{"id": 1, "name": "Category"}],
        "tags": [],
        "images": [{"src": "https://example.com/img.jpg"}],
        "attributes": [],
    }


@pytest.fixture
def mock_customer_data():
    """Mock customer API response."""
    return {
        "id": 42,
        "email": "john@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "username": "johndoe",
        "date_created_gmt": "2024-01-01T00:00:00",
        "date_modified_gmt": "2024-01-15T00:00:00",
        "billing": {"first_name": "John", "last_name": "Doe"},
        "shipping": {"first_name": "John", "last_name": "Doe"},
        "is_paying_customer": True,
        "orders_count": 5,
        "total_spent": "499.95",
        "avatar_url": "https://example.com/avatar.jpg",
    }


class TestWooCommerceCredentials:
    """Tests for WooCommerceCredentials."""

    def test_from_env(self):
        """Test creating credentials from environment."""
        with patch.dict(
            "os.environ",
            {
                "WOOCOMMERCE_URL": "https://store.example.com",
                "WOOCOMMERCE_CONSUMER_KEY": "ck_key",
                "WOOCOMMERCE_CONSUMER_SECRET": "cs_secret",
                "WOOCOMMERCE_VERSION": "wc/v3",
            },
        ):
            creds = WooCommerceCredentials.from_env()
            assert creds.store_url == "https://store.example.com"
            assert creds.consumer_key == "ck_key"
            assert creds.consumer_secret == "cs_secret"
            assert creds.api_version == "wc/v3"

    def test_defaults(self):
        """Test default values."""
        creds = WooCommerceCredentials(
            store_url="https://store.example.com",
            consumer_key="ck_key",
            consumer_secret="cs_secret",
        )
        assert creds.api_version == "wc/v3"
        assert creds.timeout == 30


class TestWooCommerceConnector:
    """Tests for WooCommerceConnector."""

    def test_base_url(self, connector, credentials):
        """Test base URL construction."""
        expected = f"{credentials.store_url}/wp-json/{credentials.api_version}"
        assert connector.base_url == expected

    @pytest.mark.asyncio
    async def test_connect_success(self, connector):
        """Test successful connection."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"environment": {"site_url": "test"}})

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await connector.connect()
            assert result is True

    @pytest.mark.asyncio
    async def test_connect_failure(self, connector):
        """Test connection failure."""
        mock_response = AsyncMock()
        mock_response.status = 401

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await connector.connect()
            assert result is False

    @pytest.mark.asyncio
    async def test_disconnect(self, connector):
        """Test disconnect."""
        mock_client = AsyncMock()
        connector._client = mock_client
        await connector.disconnect()
        mock_client.close.assert_called_once()
        assert connector._client is None


class TestOrderParsing:
    """Tests for order parsing."""

    def test_parse_order(self, connector, mock_order_data):
        """Test parsing order from API response."""
        order = connector._parse_order(mock_order_data)

        assert order.id == 1001
        assert order.number == "1001"
        assert order.status == WooOrderStatus.PROCESSING
        assert order.currency == "USD"
        assert order.total == Decimal("99.99")
        assert order.customer_id == 42
        assert order.billing.first_name == "John"
        assert order.billing.email == "john@example.com"
        assert len(order.line_items) == 1
        assert order.line_items[0].product_id == 101
        assert order.line_items[0].quantity == 2

    def test_parse_address(self, connector):
        """Test parsing address."""
        address_data = {
            "first_name": "Jane",
            "last_name": "Smith",
            "city": "Boston",
            "country": "US",
        }
        address = connector._parse_address(address_data)

        assert address.first_name == "Jane"
        assert address.last_name == "Smith"
        assert address.city == "Boston"
        assert address.country == "US"


class TestProductParsing:
    """Tests for product parsing."""

    def test_parse_product(self, connector, mock_product_data):
        """Test parsing product from API response."""
        product = connector._parse_product(mock_product_data)

        assert product.id == 101
        assert product.name == "Test Product"
        assert product.type == WooProductType.SIMPLE
        assert product.status == WooProductStatus.PUBLISH
        assert product.price == Decimal("49.99")
        assert product.stock_quantity == 25
        assert product.manage_stock is True
        assert len(product.images) == 1

    def test_parse_variation(self, connector):
        """Test parsing product variation."""
        variation_data = {
            "id": 201,
            "sku": "TST-001-RED",
            "price": "54.99",
            "regular_price": "64.99",
            "sale_price": "54.99",
            "stock_quantity": 10,
            "stock_status": "instock",
            "manage_stock": True,
            "attributes": [{"name": "Color", "option": "Red"}],
            "image": {"src": "https://example.com/red.jpg"},
        }
        variation = connector._parse_variation(variation_data)

        assert variation.id == 201
        assert variation.sku == "TST-001-RED"
        assert variation.price == Decimal("54.99")
        assert variation.stock_quantity == 10
        assert variation.image == "https://example.com/red.jpg"


class TestCustomerParsing:
    """Tests for customer parsing."""

    def test_parse_customer(self, connector, mock_customer_data):
        """Test parsing customer from API response."""
        customer = connector._parse_customer(mock_customer_data)

        assert customer.id == 42
        assert customer.email == "john@example.com"
        assert customer.first_name == "John"
        assert customer.last_name == "Doe"
        assert customer.orders_count == 5
        assert customer.total_spent == Decimal("499.95")
        assert customer.is_paying_customer is True


class TestDataclassToDict:
    """Tests for dataclass to_dict methods."""

    def test_address_to_dict(self):
        """Test WooAddress to_dict."""
        address = WooAddress(
            first_name="John",
            last_name="Doe",
            city="NYC",
            country="US",
        )
        data = address.to_dict()

        assert data["firstName"] == "John"
        assert data["lastName"] == "Doe"
        assert data["city"] == "NYC"
        assert data["country"] == "US"

    def test_line_item_to_dict(self):
        """Test WooLineItem to_dict."""
        item = WooLineItem(
            id=1,
            product_id=101,
            variation_id=0,
            name="Test",
            quantity=2,
            subtotal=Decimal("99.99"),
            total=Decimal("99.99"),
            sku="TST",
        )
        data = item.to_dict()

        assert data["id"] == 1
        assert data["productId"] == 101
        assert data["quantity"] == 2
        assert data["total"] == "99.99"

    def test_order_to_dict(self):
        """Test WooOrder to_dict."""
        now = datetime.now(timezone.utc)
        order = WooOrder(
            id=1001,
            number="1001",
            order_key="wc_order_test",
            status=WooOrderStatus.PROCESSING,
            currency="USD",
            date_created=now,
            date_modified=now,
            total=Decimal("99.99"),
            subtotal=Decimal("89.99"),
            total_tax=Decimal("10.00"),
            shipping_total=Decimal("5.00"),
            discount_total=Decimal("0.00"),
            payment_method="stripe",
            payment_method_title="Credit Card",
            customer_id=42,
            billing=WooAddress(first_name="John"),
            shipping=WooAddress(first_name="John"),
        )
        data = order.to_dict()

        assert data["id"] == 1001
        assert data["status"] == "processing"
        assert data["total"] == "99.99"
        assert data["currency"] == "USD"

    def test_product_to_dict(self):
        """Test WooProduct to_dict."""
        now = datetime.now(timezone.utc)
        product = WooProduct(
            id=101,
            name="Test",
            slug="test",
            type=WooProductType.SIMPLE,
            status=WooProductStatus.PUBLISH,
            sku="TST",
            price=Decimal("49.99"),
            regular_price=Decimal("59.99"),
            sale_price=Decimal("49.99"),
            date_created=now,
            date_modified=now,
            description="Test product",
            short_description="Test",
            stock_quantity=25,
            stock_status=WooStockStatus.IN_STOCK,
            manage_stock=True,
        )
        data = product.to_dict()

        assert data["id"] == 101
        assert data["name"] == "Test"
        assert data["type"] == "simple"
        assert data["stockQuantity"] == 25


class TestWebhookSignature:
    """Tests for webhook signature verification."""

    def test_verify_webhook_signature_valid(self, connector):
        """Test valid webhook signature verification."""
        import base64
        import hashlib
        import hmac

        payload = b'{"test": "data"}'
        secret = "test_secret"
        signature = base64.b64encode(
            hmac.new(secret.encode(), payload, hashlib.sha256).digest()
        ).decode()

        assert connector.verify_webhook_signature(payload, signature, secret) is True

    def test_verify_webhook_signature_invalid(self, connector):
        """Test invalid webhook signature verification."""
        payload = b'{"test": "data"}'
        secret = "test_secret"
        invalid_signature = "invalid_signature"

        assert connector.verify_webhook_signature(payload, invalid_signature, secret) is False


class TestMockData:
    """Tests for mock data helpers."""

    def test_get_mock_orders(self):
        """Test mock orders generation."""
        orders = get_mock_woo_orders()
        assert len(orders) == 1
        assert orders[0].id == 1001
        assert orders[0].status == WooOrderStatus.PROCESSING
        assert len(orders[0].line_items) == 1

    def test_get_mock_products(self):
        """Test mock products generation."""
        products = get_mock_woo_products()
        assert len(products) == 1
        assert products[0].id == 101
        assert products[0].sku == "WOO-001"
        assert products[0].stock_quantity == 50


class TestEnums:
    """Tests for enum values."""

    def test_order_status_values(self):
        """Test WooOrderStatus values."""
        assert WooOrderStatus.PENDING.value == "pending"
        assert WooOrderStatus.PROCESSING.value == "processing"
        assert WooOrderStatus.COMPLETED.value == "completed"
        assert WooOrderStatus.CANCELLED.value == "cancelled"
        assert WooOrderStatus.REFUNDED.value == "refunded"

    def test_product_status_values(self):
        """Test WooProductStatus values."""
        assert WooProductStatus.PUBLISH.value == "publish"
        assert WooProductStatus.DRAFT.value == "draft"
        assert WooProductStatus.PENDING.value == "pending"
        assert WooProductStatus.PRIVATE.value == "private"

    def test_product_type_values(self):
        """Test WooProductType values."""
        assert WooProductType.SIMPLE.value == "simple"
        assert WooProductType.VARIABLE.value == "variable"
        assert WooProductType.GROUPED.value == "grouped"
        assert WooProductType.EXTERNAL.value == "external"

    def test_stock_status_values(self):
        """Test WooStockStatus values."""
        assert WooStockStatus.IN_STOCK.value == "instock"
        assert WooStockStatus.OUT_OF_STOCK.value == "outofstock"
        assert WooStockStatus.ON_BACKORDER.value == "onbackorder"
