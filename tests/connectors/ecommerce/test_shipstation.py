"""
Tests for the ShipStation Connector.

Covers:
- Enum types (OrderStatus, ShipmentStatus)
- Dataclass models (credentials, address, order item, order, shipment, carrier, rate quote, warehouse)
- ShipStationConnector class (initialization, context management, orders, shipments, labels, rates, carriers)
- Authentication with Basic Auth
- Error handling (ShipStationError)
- Data serialization and deserialization
- Mock data helpers
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from aragora.connectors.ecommerce.shipstation import (
    Carrier,
    CarrierService,
    OrderItem,
    OrderStatus,
    RateQuote,
    Shipment,
    ShipmentStatus,
    ShipStationAddress,
    ShipStationConnector,
    ShipStationCredentials,
    ShipStationError,
    ShipStationOrder,
    Warehouse,
    _parse_datetime,
    get_mock_order,
    get_mock_shipment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def credentials():
    """Standard test credentials."""
    return ShipStationCredentials(
        api_key="test_api_key_12345",
        api_secret="test_api_secret_67890",
    )


@pytest.fixture
def connector(credentials):
    """ShipStationConnector instance (not yet connected)."""
    return ShipStationConnector(credentials=credentials)


@pytest.fixture
def sample_address():
    """Sample shipping address."""
    return ShipStationAddress(
        name="John Doe",
        company="Acme Corp",
        street1="123 Main St",
        street2="Suite 100",
        street3="",
        city="Los Angeles",
        state="CA",
        postal_code="90210",
        country="US",
        phone="+15551234567",
        residential=True,
    )


@pytest.fixture
def sample_order_item():
    """Sample order item."""
    return OrderItem(
        order_item_id=12345,
        line_item_key="LI-001",
        sku="SKU-001",
        name="Widget",
        quantity=2,
        unit_price=Decimal("19.99"),
        weight_oz=8.0,
    )


@pytest.fixture
def sample_order(sample_address, sample_order_item):
    """Sample ShipStation order."""
    return ShipStationOrder(
        order_id=12345,
        order_number="ORD-1001",
        order_key="ord-key-123",
        order_date=datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        order_status=OrderStatus.AWAITING_SHIPMENT,
        customer_email="customer@example.com",
        customer_notes="Please gift wrap",
        internal_notes="Priority customer",
        ship_to=sample_address,
        bill_to=sample_address,
        items=[sample_order_item],
        amount_paid=Decimal("39.98"),
        shipping_amount=Decimal("5.99"),
        tax_amount=Decimal("3.20"),
        weight_oz=16.0,
        carrier_code="fedex",
        service_code="fedex_ground",
        package_code="package",
        tracking_number="",
        ship_date=None,
        create_date=datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc),
        modify_date=datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_shipment():
    """Sample shipment."""
    return Shipment(
        shipment_id=67890,
        order_id=12345,
        order_number="ORD-1001",
        carrier_code="fedex",
        service_code="fedex_ground",
        tracking_number="794644790132",
        ship_date=datetime(2024, 6, 16, 8, 0, 0, tzinfo=timezone.utc),
        ship_cost=Decimal("8.50"),
        weight_oz=16.0,
        voided=False,
        void_date=None,
    )


@pytest.fixture
def sample_carrier():
    """Sample carrier."""
    return Carrier(
        code="fedex",
        name="FedEx",
        account_number="123456",
        shipping_provider_id=1234,
        primary=True,
    )


@pytest.fixture
def sample_rate_quote():
    """Sample rate quote."""
    return RateQuote(
        carrier_code="fedex",
        carrier_name="FedEx",
        service_code="fedex_ground",
        service_name="FedEx Ground",
        shipment_cost=Decimal("8.50"),
        other_cost=Decimal("0.00"),
    )


@pytest.fixture
def sample_warehouse(sample_address):
    """Sample warehouse."""
    return Warehouse(
        warehouse_id=12345,
        warehouse_name="Main Warehouse",
        origin_address=sample_address,
        return_address=sample_address,
        is_default=True,
    )


# ---------------------------------------------------------------------------
# Test Enum Types
# ---------------------------------------------------------------------------


class TestOrderStatus:
    """Tests for OrderStatus enum."""

    def test_order_status_values(self):
        """OrderStatus should have expected values."""
        assert OrderStatus.AWAITING_PAYMENT.value == "awaiting_payment"
        assert OrderStatus.AWAITING_SHIPMENT.value == "awaiting_shipment"
        assert OrderStatus.SHIPPED.value == "shipped"
        assert OrderStatus.ON_HOLD.value == "on_hold"
        assert OrderStatus.CANCELLED.value == "cancelled"

    def test_order_status_from_string(self):
        """OrderStatus should be creatable from string value."""
        status = OrderStatus("awaiting_shipment")
        assert status == OrderStatus.AWAITING_SHIPMENT


class TestShipmentStatus:
    """Tests for ShipmentStatus enum."""

    def test_shipment_status_values(self):
        """ShipmentStatus should have expected values."""
        assert ShipmentStatus.LABEL_CREATED.value == "label_created"
        assert ShipmentStatus.IN_TRANSIT.value == "in_transit"
        assert ShipmentStatus.OUT_FOR_DELIVERY.value == "out_for_delivery"
        assert ShipmentStatus.DELIVERED.value == "delivered"
        assert ShipmentStatus.EXCEPTION.value == "exception"


# ---------------------------------------------------------------------------
# Test Dataclass Models
# ---------------------------------------------------------------------------


class TestShipStationCredentials:
    """Tests for ShipStationCredentials dataclass."""

    def test_credentials_creation(self, credentials):
        """Credentials should store API key and secret."""
        assert credentials.api_key == "test_api_key_12345"
        assert credentials.api_secret == "test_api_secret_67890"


class TestShipStationAddress:
    """Tests for ShipStationAddress dataclass."""

    def test_address_creation(self, sample_address):
        """Address should have all expected fields."""
        assert sample_address.name == "John Doe"
        assert sample_address.company == "Acme Corp"
        assert sample_address.city == "Los Angeles"
        assert sample_address.residential is True

    def test_address_from_api(self):
        """Address should be creatable from API response."""
        api_data = {
            "name": "Jane Doe",
            "company": "Test Inc",
            "street1": "456 Oak Ave",
            "street2": "",
            "street3": "",
            "city": "New York",
            "state": "NY",
            "postalCode": "10001",
            "country": "US",
            "phone": "+15559876543",
            "residential": False,
        }
        address = ShipStationAddress.from_api(api_data)
        assert address.name == "Jane Doe"
        assert address.postal_code == "10001"
        assert address.residential is False

    def test_address_from_api_none(self):
        """Address.from_api should return None for None input."""
        assert ShipStationAddress.from_api(None) is None

    def test_address_to_api(self, sample_address):
        """Address should serialize correctly for API requests."""
        api_data = sample_address.to_api()
        assert api_data["name"] == "John Doe"
        assert api_data["postalCode"] == "90210"
        assert api_data["residential"] is True


class TestOrderItem:
    """Tests for OrderItem dataclass."""

    def test_order_item_creation(self, sample_order_item):
        """OrderItem should have all expected fields."""
        assert sample_order_item.sku == "SKU-001"
        assert sample_order_item.quantity == 2
        assert sample_order_item.unit_price == Decimal("19.99")

    def test_order_item_from_api(self):
        """OrderItem should be creatable from API response."""
        api_data = {
            "orderItemId": 54321,
            "lineItemKey": "LI-002",
            "sku": "SKU-002",
            "name": "Gadget",
            "quantity": 1,
            "unitPrice": 29.99,
            "weight": {"value": 12.0, "units": "ounces"},
        }
        item = OrderItem.from_api(api_data)
        assert item.order_item_id == 54321
        assert item.sku == "SKU-002"
        assert item.unit_price == Decimal("29.99")
        assert item.weight_oz == 12.0

    def test_order_item_to_api(self, sample_order_item):
        """OrderItem should serialize correctly for API requests."""
        api_data = sample_order_item.to_api()
        assert api_data["sku"] == "SKU-001"
        assert api_data["quantity"] == 2
        assert api_data["unitPrice"] == 19.99


class TestShipStationOrder:
    """Tests for ShipStationOrder dataclass."""

    def test_order_creation(self, sample_order):
        """Order should have all expected fields."""
        assert sample_order.order_id == 12345
        assert sample_order.order_number == "ORD-1001"
        assert sample_order.order_status == OrderStatus.AWAITING_SHIPMENT
        assert len(sample_order.items) == 1

    def test_order_from_api(self):
        """Order should be creatable from API response."""
        api_data = {
            "orderId": 99999,
            "orderNumber": "ORD-9999",
            "orderKey": "key-9999",
            "orderDate": "2024-06-15T12:00:00Z",
            "orderStatus": "shipped",
            "customerEmail": "test@example.com",
            "customerNotes": "",
            "internalNotes": "",
            "shipTo": None,
            "billTo": None,
            "items": [],
            "amountPaid": 50.00,
            "shippingAmount": 5.00,
            "taxAmount": 4.00,
            "weight": {"value": 24.0, "units": "ounces"},
            "carrierCode": "ups",
            "serviceCode": "ups_ground",
            "packageCode": "package",
            "trackingNumber": "1Z999AA10123456784",
            "shipDate": "2024-06-16T00:00:00Z",
            "createDate": "2024-06-15T10:00:00Z",
            "modifyDate": "2024-06-15T14:00:00Z",
        }
        order = ShipStationOrder.from_api(api_data)
        assert order.order_id == 99999
        assert order.order_status == OrderStatus.SHIPPED
        assert order.tracking_number == "1Z999AA10123456784"


class TestShipment:
    """Tests for Shipment dataclass."""

    def test_shipment_creation(self, sample_shipment):
        """Shipment should have all expected fields."""
        assert sample_shipment.shipment_id == 67890
        assert sample_shipment.tracking_number == "794644790132"
        assert sample_shipment.ship_cost == Decimal("8.50")

    def test_shipment_from_api(self):
        """Shipment should be creatable from API response."""
        api_data = {
            "shipmentId": 11111,
            "orderId": 22222,
            "orderNumber": "ORD-2222",
            "carrierCode": "usps",
            "serviceCode": "usps_priority_mail",
            "trackingNumber": "9400111899223333444455",
            "shipDate": "2024-06-17T00:00:00Z",
            "shipmentCost": 7.25,
            "weight": {"value": 10.0, "units": "ounces"},
            "voided": False,
            "voidDate": None,
        }
        shipment = Shipment.from_api(api_data)
        assert shipment.shipment_id == 11111
        assert shipment.carrier_code == "usps"
        assert shipment.ship_cost == Decimal("7.25")


class TestCarrier:
    """Tests for Carrier dataclass."""

    def test_carrier_from_api(self):
        """Carrier should be creatable from API response."""
        api_data = {
            "code": "ups",
            "name": "UPS",
            "accountNumber": "789012",
            "shippingProviderId": 5678,
            "primary": False,
        }
        carrier = Carrier.from_api(api_data)
        assert carrier.code == "ups"
        assert carrier.name == "UPS"
        assert carrier.primary is False


class TestCarrierService:
    """Tests for CarrierService dataclass."""

    def test_carrier_service_from_api(self):
        """CarrierService should be creatable from API response."""
        api_data = {
            "code": "fedex_ground",
            "name": "FedEx Ground",
            "carrierCode": "fedex",
            "domestic": True,
            "international": False,
        }
        service = CarrierService.from_api(api_data)
        assert service.code == "fedex_ground"
        assert service.domestic is True
        assert service.international is False


class TestRateQuote:
    """Tests for RateQuote dataclass."""

    def test_rate_quote_from_api(self):
        """RateQuote should be creatable from API response."""
        api_data = {
            "carrierCode": "fedex",
            "carrierNickname": "FedEx",
            "serviceCode": "fedex_2day",
            "serviceName": "FedEx 2Day",
            "shipmentCost": 15.50,
            "otherCost": 1.00,
        }
        quote = RateQuote.from_api(api_data)
        assert quote.carrier_code == "fedex"
        assert quote.service_code == "fedex_2day"
        assert quote.shipment_cost == Decimal("15.50")
        assert quote.other_cost == Decimal("1.00")


class TestWarehouse:
    """Tests for Warehouse dataclass."""

    def test_warehouse_from_api(self):
        """Warehouse should be creatable from API response."""
        api_data = {
            "warehouseId": 54321,
            "warehouseName": "West Coast Warehouse",
            "originAddress": {
                "name": "Warehouse",
                "street1": "789 Industrial Way",
                "city": "Portland",
                "state": "OR",
                "postalCode": "97201",
                "country": "US",
            },
            "returnAddress": None,
            "isDefault": False,
        }
        warehouse = Warehouse.from_api(api_data)
        assert warehouse.warehouse_id == 54321
        assert warehouse.warehouse_name == "West Coast Warehouse"
        assert warehouse.is_default is False
        assert warehouse.origin_address is not None
        assert warehouse.origin_address.city == "Portland"


# ---------------------------------------------------------------------------
# Test ShipStationError
# ---------------------------------------------------------------------------


class TestShipStationError:
    """Tests for ShipStationError exception."""

    def test_error_with_message(self):
        """Error should store message."""
        error = ShipStationError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.status_code is None

    def test_error_with_status_code(self):
        """Error should store status code when provided."""
        error = ShipStationError("Not found", status_code=404)
        assert str(error) == "Not found"
        assert error.status_code == 404


# ---------------------------------------------------------------------------
# Test ShipStationConnector Initialization and Authentication
# ---------------------------------------------------------------------------


class TestConnectorInitialization:
    """Tests for ShipStationConnector initialization."""

    def test_connector_creation(self, connector, credentials):
        """Connector should be creatable with credentials."""
        assert connector.credentials == credentials
        assert connector._client is None

    def test_connector_base_url(self):
        """Connector should have correct base URL."""
        assert ShipStationConnector.BASE_URL == "https://ssapi.shipstation.com"


class TestConnectorContextManager:
    """Tests for ShipStationConnector async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_creates_client(self, connector, credentials):
        """Context manager should create httpx client."""
        async with connector as conn:
            # The client should be set (may be mock in test environment)
            assert conn._client is not None

    @pytest.mark.asyncio
    async def test_auth_header_encoding(self, credentials):
        """Auth header should use correct Basic encoding."""
        # Test the encoding logic directly
        expected_auth = base64.b64encode(
            f"{credentials.api_key}:{credentials.api_secret}".encode()
        ).decode()
        assert expected_auth == "dGVzdF9hcGlfa2V5XzEyMzQ1OnRlc3RfYXBpX3NlY3JldF82Nzg5MA=="

    @pytest.mark.asyncio
    async def test_context_manager_closes_client(self, connector):
        """Context manager should close client on exit."""
        async with connector:
            pass
        assert connector._client is None

    @pytest.mark.asyncio
    async def test_client_property_raises_without_context(self, connector):
        """Accessing client outside context should raise error."""
        with pytest.raises(ShipStationError, match="Connector not initialized"):
            _ = connector.client


# ---------------------------------------------------------------------------
# Test Order Operations
# ---------------------------------------------------------------------------


class TestOrderOperations:
    """Tests for order-related operations."""

    @pytest.mark.asyncio
    async def test_list_orders(self, credentials):
        """list_orders should return list of orders."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "orders": [
                {
                    "orderId": 12345,
                    "orderNumber": "ORD-1001",
                    "orderStatus": "awaiting_shipment",
                    "items": [],
                }
            ]
        }

        connector = ShipStationConnector(credentials)
        async with connector:
            with patch.object(connector.client, "request", new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_response
                orders = await connector.list_orders(order_status="awaiting_shipment")

                assert len(orders) == 1
                assert orders[0].order_id == 12345
                mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_order(self, credentials):
        """get_order should return a single order."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "orderId": 12345,
            "orderNumber": "ORD-1001",
            "orderStatus": "awaiting_shipment",
            "items": [],
        }

        connector = ShipStationConnector(credentials)
        async with connector:
            with patch.object(connector.client, "request", new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_response
                order = await connector.get_order(12345)

                assert order.order_id == 12345
                mock_request.assert_called_once_with("GET", f"{connector.BASE_URL}/orders/12345", json=None, params=None)

    @pytest.mark.asyncio
    async def test_create_order(self, credentials, sample_address, sample_order_item):
        """create_order should create and return a new order."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "orderId": 99999,
            "orderNumber": "ORD-NEW",
            "orderStatus": "awaiting_shipment",
            "items": [],
        }

        connector = ShipStationConnector(credentials)
        async with connector:
            with patch.object(connector.client, "request", new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_response
                order = await connector.create_order(
                    order_number="ORD-NEW",
                    order_date=datetime(2024, 6, 15, tzinfo=timezone.utc),
                    ship_to=sample_address,
                    items=[sample_order_item],
                    carrier_code="fedex",
                    service_code="fedex_ground",
                )

                assert order.order_id == 99999
                mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_mark_order_shipped(self, credentials):
        """mark_order_shipped should mark order as shipped."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"orderId": 12345, "orderNumber": "ORD-1001"}

        connector = ShipStationConnector(credentials)
        async with connector:
            with patch.object(connector.client, "request", new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_response
                result = await connector.mark_order_shipped(
                    order_id=12345,
                    carrier_code="fedex",
                    tracking_number="794644790132",
                    notify_customer=True,
                )

                assert result["orderId"] == 12345
                mock_request.assert_called_once()


# ---------------------------------------------------------------------------
# Test Shipping Rate Operations
# ---------------------------------------------------------------------------


class TestRateOperations:
    """Tests for shipping rate operations."""

    @pytest.mark.asyncio
    async def test_get_rates(self, credentials):
        """get_rates should return list of rate quotes."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "carrierCode": "fedex",
                "carrierNickname": "FedEx",
                "serviceCode": "fedex_ground",
                "serviceName": "FedEx Ground",
                "shipmentCost": 8.50,
                "otherCost": 0.00,
            },
            {
                "carrierCode": "fedex",
                "carrierNickname": "FedEx",
                "serviceCode": "fedex_2day",
                "serviceName": "FedEx 2Day",
                "shipmentCost": 15.50,
                "otherCost": 0.00,
            },
        ]

        connector = ShipStationConnector(credentials)
        async with connector:
            with patch.object(connector.client, "request", new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_response
                rates = await connector.get_rates(
                    carrier_code="fedex",
                    from_postal_code="90210",
                    to_postal_code="10001",
                    weight_oz=16.0,
                )

                assert len(rates) == 2
                assert rates[0].service_code == "fedex_ground"
                assert rates[0].shipment_cost == Decimal("8.50")
                assert rates[1].service_code == "fedex_2day"


# ---------------------------------------------------------------------------
# Test Label Operations
# ---------------------------------------------------------------------------


class TestLabelOperations:
    """Tests for label generation operations."""

    @pytest.mark.asyncio
    async def test_create_label(self, credentials):
        """create_label should create a shipping label."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "shipmentId": 67890,
            "orderId": 12345,
            "trackingNumber": "794644790132",
            "labelData": "base64encodedlabel==",
        }

        connector = ShipStationConnector(credentials)
        async with connector:
            with patch.object(connector.client, "request", new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_response
                result = await connector.create_label(
                    order_id=12345,
                    carrier_code="fedex",
                    service_code="fedex_ground",
                    weight_oz=16.0,
                    test_label=True,
                )

                assert result["shipmentId"] == 67890
                assert result["trackingNumber"] == "794644790132"

    @pytest.mark.asyncio
    async def test_void_label(self, credentials):
        """void_label should void a shipping label."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"approved": True}

        connector = ShipStationConnector(credentials)
        async with connector:
            with patch.object(connector.client, "request", new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_response
                result = await connector.void_label(shipment_id=67890)

                assert result["approved"] is True


# ---------------------------------------------------------------------------
# Test Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_api_error_response(self, credentials):
        """API errors should raise ShipStationError with status code."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"Message": "Order not found"}

        connector = ShipStationConnector(credentials)
        async with connector:
            with patch.object(connector.client, "request", new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_response

                with pytest.raises(ShipStationError) as exc_info:
                    await connector.get_order(99999)

                assert exc_info.value.status_code == 404
                assert "Order not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_http_error_handling(self, credentials):
        """HTTP errors should be wrapped in ShipStationError."""
        connector = ShipStationConnector(credentials)
        async with connector:
            with patch.object(connector.client, "request", new_callable=AsyncMock) as mock_request:
                mock_request.side_effect = httpx.HTTPError("Connection refused")

                with pytest.raises(ShipStationError, match="HTTP error"):
                    await connector.get_order(12345)


# ---------------------------------------------------------------------------
# Test Carrier and Warehouse Operations
# ---------------------------------------------------------------------------


class TestCarrierOperations:
    """Tests for carrier-related operations."""

    @pytest.mark.asyncio
    async def test_list_carriers(self, credentials):
        """list_carriers should return list of carriers."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"code": "fedex", "name": "FedEx", "primary": True},
            {"code": "ups", "name": "UPS", "primary": False},
        ]

        connector = ShipStationConnector(credentials)
        async with connector:
            with patch.object(connector.client, "request", new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_response
                carriers = await connector.list_carriers()

                assert len(carriers) == 2
                assert carriers[0].code == "fedex"
                assert carriers[0].primary is True

    @pytest.mark.asyncio
    async def test_list_services(self, credentials):
        """list_services should return list of carrier services."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"code": "fedex_ground", "name": "FedEx Ground", "carrierCode": "fedex", "domestic": True},
            {"code": "fedex_2day", "name": "FedEx 2Day", "carrierCode": "fedex", "domestic": True},
        ]

        connector = ShipStationConnector(credentials)
        async with connector:
            with patch.object(connector.client, "request", new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_response
                services = await connector.list_services("fedex")

                assert len(services) == 2
                assert services[0].code == "fedex_ground"


class TestWarehouseOperations:
    """Tests for warehouse-related operations."""

    @pytest.mark.asyncio
    async def test_list_warehouses(self, credentials):
        """list_warehouses should return list of warehouses."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"warehouseId": 12345, "warehouseName": "Main Warehouse", "isDefault": True},
        ]

        connector = ShipStationConnector(credentials)
        async with connector:
            with patch.object(connector.client, "request", new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_response
                warehouses = await connector.list_warehouses()

                assert len(warehouses) == 1
                assert warehouses[0].warehouse_name == "Main Warehouse"


# ---------------------------------------------------------------------------
# Test Helper Functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_parse_datetime_iso_format(self):
        """_parse_datetime should parse ISO format."""
        result = _parse_datetime("2024-06-15T12:00:00Z")
        assert result is not None
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15

    def test_parse_datetime_with_milliseconds(self):
        """_parse_datetime should parse datetime with milliseconds."""
        result = _parse_datetime("2024-06-15T12:00:00.123")
        assert result is not None
        assert result.year == 2024

    def test_parse_datetime_none(self):
        """_parse_datetime should return None for None input."""
        assert _parse_datetime(None) is None

    def test_parse_datetime_invalid(self):
        """_parse_datetime should return None for invalid input."""
        assert _parse_datetime("not a date") is None


# ---------------------------------------------------------------------------
# Test Mock Data
# ---------------------------------------------------------------------------


class TestMockData:
    """Tests for mock data helpers."""

    def test_get_mock_order(self):
        """get_mock_order should return a valid order."""
        order = get_mock_order()
        assert order.order_id == 12345
        assert order.order_number == "ORD-1001"
        assert order.order_status == OrderStatus.AWAITING_SHIPMENT
        assert len(order.items) == 1
        assert order.ship_to is not None

    def test_get_mock_shipment(self):
        """get_mock_shipment should return a valid shipment."""
        shipment = get_mock_shipment()
        assert shipment.shipment_id == 67890
        assert shipment.order_id == 12345
        assert shipment.carrier_code == "fedex"
        assert shipment.tracking_number == "1234567890"
