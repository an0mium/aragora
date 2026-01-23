"""
ShipStation Connector.

Integration with ShipStation shipping and fulfillment:
- Orders and shipments
- Carriers and services
- Labels and tracking
- Warehouses and inventory
- Webhooks for real-time updates

Requires ShipStation API key and secret.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OrderStatus(str, Enum):
    """ShipStation order status."""

    AWAITING_PAYMENT = "awaiting_payment"
    AWAITING_SHIPMENT = "awaiting_shipment"
    SHIPPED = "shipped"
    ON_HOLD = "on_hold"
    CANCELLED = "cancelled"


class ShipmentStatus(str, Enum):
    """Shipment status."""

    LABEL_CREATED = "label_created"
    IN_TRANSIT = "in_transit"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    EXCEPTION = "exception"


@dataclass
class ShipStationCredentials:
    """ShipStation API credentials."""

    api_key: str
    api_secret: str


@dataclass
class ShipStationAddress:
    """Shipping address."""

    name: str = ""
    company: str = ""
    street1: str = ""
    street2: str = ""
    street3: str = ""
    city: str = ""
    state: str = ""
    postal_code: str = ""
    country: str = ""
    phone: str = ""
    residential: bool = True

    @classmethod
    def from_api(cls, data: dict[str, Any] | None) -> ShipStationAddress | None:
        if not data:
            return None
        return cls(
            name=data.get("name", ""),
            company=data.get("company", ""),
            street1=data.get("street1", ""),
            street2=data.get("street2", ""),
            street3=data.get("street3", ""),
            city=data.get("city", ""),
            state=data.get("state", ""),
            postal_code=data.get("postalCode", ""),
            country=data.get("country", ""),
            phone=data.get("phone", ""),
            residential=data.get("residential", True),
        )

    def to_api(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "company": self.company,
            "street1": self.street1,
            "street2": self.street2,
            "street3": self.street3,
            "city": self.city,
            "state": self.state,
            "postalCode": self.postal_code,
            "country": self.country,
            "phone": self.phone,
            "residential": self.residential,
        }


@dataclass
class OrderItem:
    """An order item."""

    order_item_id: int | None = None
    line_item_key: str = ""
    sku: str = ""
    name: str = ""
    quantity: int = 1
    unit_price: Decimal = Decimal("0")
    weight_oz: float = 0.0

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> OrderItem:
        return cls(
            order_item_id=data.get("orderItemId"),
            line_item_key=data.get("lineItemKey", ""),
            sku=data.get("sku", ""),
            name=data.get("name", ""),
            quantity=data.get("quantity", 1),
            unit_price=Decimal(str(data.get("unitPrice", 0))),
            weight_oz=data.get("weight", {}).get("value", 0) if data.get("weight") else 0,
        )

    def to_api(self) -> dict[str, Any]:
        return {
            "lineItemKey": self.line_item_key,
            "sku": self.sku,
            "name": self.name,
            "quantity": self.quantity,
            "unitPrice": float(self.unit_price),
        }


@dataclass
class ShipStationOrder:
    """A ShipStation order."""

    order_id: int | None = None
    order_number: str = ""
    order_key: str = ""
    order_date: datetime | None = None
    order_status: OrderStatus = OrderStatus.AWAITING_SHIPMENT
    customer_email: str = ""
    customer_notes: str = ""
    internal_notes: str = ""
    ship_to: ShipStationAddress | None = None
    bill_to: ShipStationAddress | None = None
    items: list[OrderItem] = field(default_factory=list)
    amount_paid: Decimal = Decimal("0")
    shipping_amount: Decimal = Decimal("0")
    tax_amount: Decimal = Decimal("0")
    weight_oz: float = 0.0
    carrier_code: str = ""
    service_code: str = ""
    package_code: str = ""
    tracking_number: str = ""
    ship_date: datetime | None = None
    create_date: datetime | None = None
    modify_date: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> ShipStationOrder:
        return cls(
            order_id=data.get("orderId"),
            order_number=data.get("orderNumber", ""),
            order_key=data.get("orderKey", ""),
            order_date=_parse_datetime(data.get("orderDate")),
            order_status=OrderStatus(data.get("orderStatus", "awaiting_shipment")),
            customer_email=data.get("customerEmail", ""),
            customer_notes=data.get("customerNotes", ""),
            internal_notes=data.get("internalNotes", ""),
            ship_to=ShipStationAddress.from_api(data.get("shipTo")),
            bill_to=ShipStationAddress.from_api(data.get("billTo")),
            items=[OrderItem.from_api(i) for i in data.get("items", [])],
            amount_paid=Decimal(str(data.get("amountPaid", 0))),
            shipping_amount=Decimal(str(data.get("shippingAmount", 0))),
            tax_amount=Decimal(str(data.get("taxAmount", 0))),
            weight_oz=data.get("weight", {}).get("value", 0) if data.get("weight") else 0,
            carrier_code=data.get("carrierCode", ""),
            service_code=data.get("serviceCode", ""),
            package_code=data.get("packageCode", ""),
            tracking_number=data.get("trackingNumber", ""),
            ship_date=_parse_datetime(data.get("shipDate")),
            create_date=_parse_datetime(data.get("createDate")),
            modify_date=_parse_datetime(data.get("modifyDate")),
        )


@dataclass
class Shipment:
    """A ShipStation shipment."""

    shipment_id: int
    order_id: int
    order_number: str = ""
    carrier_code: str = ""
    service_code: str = ""
    tracking_number: str = ""
    ship_date: datetime | None = None
    ship_cost: Decimal = Decimal("0")
    weight_oz: float = 0.0
    voided: bool = False
    void_date: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Shipment:
        return cls(
            shipment_id=data["shipmentId"],
            order_id=data.get("orderId", 0),
            order_number=data.get("orderNumber", ""),
            carrier_code=data.get("carrierCode", ""),
            service_code=data.get("serviceCode", ""),
            tracking_number=data.get("trackingNumber", ""),
            ship_date=_parse_datetime(data.get("shipDate")),
            ship_cost=Decimal(str(data.get("shipmentCost", 0))),
            weight_oz=data.get("weight", {}).get("value", 0) if data.get("weight") else 0,
            voided=data.get("voided", False),
            void_date=_parse_datetime(data.get("voidDate")),
        )


@dataclass
class Carrier:
    """A shipping carrier."""

    code: str
    name: str
    account_number: str = ""
    shipping_provider_id: int | None = None
    primary: bool = False

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Carrier:
        return cls(
            code=data["code"],
            name=data["name"],
            account_number=data.get("accountNumber", ""),
            shipping_provider_id=data.get("shippingProviderId"),
            primary=data.get("primary", False),
        )


@dataclass
class CarrierService:
    """A carrier service (shipping method)."""

    code: str
    name: str
    carrier_code: str
    domestic: bool = True
    international: bool = False

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> CarrierService:
        return cls(
            code=data["code"],
            name=data["name"],
            carrier_code=data.get("carrierCode", ""),
            domestic=data.get("domestic", True),
            international=data.get("international", False),
        )


@dataclass
class RateQuote:
    """A shipping rate quote."""

    carrier_code: str
    carrier_name: str
    service_code: str
    service_name: str
    shipment_cost: Decimal
    other_cost: Decimal = Decimal("0")

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> RateQuote:
        return cls(
            carrier_code=data.get("carrierCode", ""),
            carrier_name=data.get("carrierNickname", ""),
            service_code=data.get("serviceCode", ""),
            service_name=data.get("serviceName", ""),
            shipment_cost=Decimal(str(data.get("shipmentCost", 0))),
            other_cost=Decimal(str(data.get("otherCost", 0))),
        )


@dataclass
class Warehouse:
    """A ShipStation warehouse."""

    warehouse_id: int
    warehouse_name: str
    origin_address: ShipStationAddress | None = None
    return_address: ShipStationAddress | None = None
    is_default: bool = False

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Warehouse:
        return cls(
            warehouse_id=data["warehouseId"],
            warehouse_name=data.get("warehouseName", ""),
            origin_address=ShipStationAddress.from_api(data.get("originAddress")),
            return_address=ShipStationAddress.from_api(data.get("returnAddress")),
            is_default=data.get("isDefault", False),
        )


class ShipStationError(Exception):
    """ShipStation API error."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ShipStationConnector:
    """
    ShipStation API connector.

    Features:
    - Order management
    - Shipment creation and tracking
    - Rate shopping
    - Label generation
    - Carrier and service management
    - Warehouse management

    Usage:
        ```python
        credentials = ShipStationCredentials(
            api_key="...",
            api_secret="..."
        )

        async with ShipStationConnector(credentials) as ss:
            # List orders awaiting shipment
            orders = await ss.list_orders(order_status="awaiting_shipment")

            # Get shipping rates
            rates = await ss.get_rates(
                carrier_code="fedex",
                from_postal_code="90210",
                to_postal_code="10001",
                weight_oz=16
            )

            # Create a shipment
            shipment = await ss.create_label(
                order_id=12345,
                carrier_code="fedex",
                service_code="fedex_ground"
            )
        ```
    """

    BASE_URL = "https://ssapi.shipstation.com"

    def __init__(self, credentials: ShipStationCredentials):
        self.credentials = credentials
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> ShipStationConnector:
        auth = base64.b64encode(
            f"{self.credentials.api_key}:{self.credentials.api_secret}".encode()
        ).decode()
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if not self._client:
            raise ShipStationError("Connector not initialized. Use async context manager.")
        return self._client

    async def _request(
        self,
        method: str,
        endpoint: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.BASE_URL}{endpoint}"
        try:
            response = await self.client.request(method, url, json=json, params=params)

            if response.status_code == 204:
                return {}

            data = response.json()

            if response.status_code >= 400:
                message = data.get("Message", data.get("message", "Unknown error"))
                raise ShipStationError(message=message, status_code=response.status_code)

            return data
        except httpx.HTTPError as e:
            raise ShipStationError(f"HTTP error: {e}") from e

    # -------------------------------------------------------------------------
    # Orders
    # -------------------------------------------------------------------------

    async def list_orders(
        self,
        order_status: str | None = None,
        order_number: str | None = None,
        customer_name: str | None = None,
        create_date_start: datetime | None = None,
        create_date_end: datetime | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> list[ShipStationOrder]:
        """List orders with filters."""
        params: dict[str, Any] = {"page": page, "pageSize": min(page_size, 500)}
        if order_status:
            params["orderStatus"] = order_status
        if order_number:
            params["orderNumber"] = order_number
        if customer_name:
            params["customerName"] = customer_name
        if create_date_start:
            params["createDateStart"] = create_date_start.strftime("%Y-%m-%d")
        if create_date_end:
            params["createDateEnd"] = create_date_end.strftime("%Y-%m-%d")

        data = await self._request("GET", "/orders", params=params)
        return [ShipStationOrder.from_api(o) for o in data.get("orders", [])]

    async def get_order(self, order_id: int) -> ShipStationOrder:
        """Get an order by ID."""
        data = await self._request("GET", f"/orders/{order_id}")
        return ShipStationOrder.from_api(data)

    async def create_order(
        self,
        order_number: str,
        order_date: datetime,
        ship_to: ShipStationAddress,
        items: list[OrderItem],
        carrier_code: str | None = None,
        service_code: str | None = None,
    ) -> ShipStationOrder:
        """Create a new order."""
        order_data: dict[str, Any] = {
            "orderNumber": order_number,
            "orderDate": order_date.strftime("%Y-%m-%dT%H:%M:%S"),
            "orderStatus": "awaiting_shipment",
            "shipTo": ship_to.to_api(),
            "items": [item.to_api() for item in items],
        }
        if carrier_code:
            order_data["carrierCode"] = carrier_code
        if service_code:
            order_data["serviceCode"] = service_code

        data = await self._request("POST", "/orders/createorder", json=order_data)
        return ShipStationOrder.from_api(data)

    async def update_order(self, order_id: int, **updates) -> ShipStationOrder:
        """Update an order."""
        updates["orderId"] = order_id
        data = await self._request("POST", "/orders/createorder", json=updates)
        return ShipStationOrder.from_api(data)

    async def delete_order(self, order_id: int) -> None:
        """Delete an order."""
        await self._request("DELETE", f"/orders/{order_id}")

    async def mark_order_shipped(
        self,
        order_id: int,
        carrier_code: str,
        tracking_number: str,
        ship_date: datetime | None = None,
        notify_customer: bool = True,
    ) -> dict:
        """Mark an order as shipped."""
        data = {
            "orderId": order_id,
            "carrierCode": carrier_code,
            "trackingNumber": tracking_number,
            "notifyCustomer": notify_customer,
        }
        if ship_date:
            data["shipDate"] = ship_date.strftime("%Y-%m-%d")

        return await self._request("POST", "/orders/markasshipped", json=data)

    # -------------------------------------------------------------------------
    # Shipments
    # -------------------------------------------------------------------------

    async def list_shipments(
        self,
        order_id: int | None = None,
        tracking_number: str | None = None,
        ship_date_start: datetime | None = None,
        ship_date_end: datetime | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> list[Shipment]:
        """List shipments."""
        params: dict[str, Any] = {"page": page, "pageSize": min(page_size, 500)}
        if order_id:
            params["orderId"] = order_id
        if tracking_number:
            params["trackingNumber"] = tracking_number
        if ship_date_start:
            params["shipDateStart"] = ship_date_start.strftime("%Y-%m-%d")
        if ship_date_end:
            params["shipDateEnd"] = ship_date_end.strftime("%Y-%m-%d")

        data = await self._request("GET", "/shipments", params=params)
        return [Shipment.from_api(s) for s in data.get("shipments", [])]

    async def create_label(
        self,
        order_id: int,
        carrier_code: str,
        service_code: str,
        package_code: str = "package",
        weight_oz: float | None = None,
        test_label: bool = False,
    ) -> dict:
        """Create a shipping label for an order."""
        label_data = {
            "orderId": order_id,
            "carrierCode": carrier_code,
            "serviceCode": service_code,
            "packageCode": package_code,
            "testLabel": test_label,
        }
        if weight_oz:
            label_data["weight"] = {"value": weight_oz, "units": "ounces"}

        return await self._request("POST", "/orders/createlabelfororder", json=label_data)

    async def void_label(self, shipment_id: int) -> dict:
        """Void a shipping label."""
        return await self._request("POST", "/shipments/voidlabel", json={"shipmentId": shipment_id})

    # -------------------------------------------------------------------------
    # Carriers and Services
    # -------------------------------------------------------------------------

    async def list_carriers(self) -> list[Carrier]:
        """List available carriers."""
        data = await self._request("GET", "/carriers")
        return [Carrier.from_api(c) for c in data]

    async def list_services(self, carrier_code: str) -> list[CarrierService]:
        """List services for a carrier."""
        data = await self._request(
            "GET", "/carriers/listservices", params={"carrierCode": carrier_code}
        )
        return [CarrierService.from_api(s) for s in data]

    async def list_packages(self, carrier_code: str) -> list[dict]:
        """List package types for a carrier."""
        return await self._request(
            "GET", "/carriers/listpackages", params={"carrierCode": carrier_code}
        )

    # -------------------------------------------------------------------------
    # Rates
    # -------------------------------------------------------------------------

    async def get_rates(
        self,
        carrier_code: str,
        from_postal_code: str,
        to_postal_code: str,
        weight_oz: float,
        to_country: str = "US",
        from_country: str = "US",
        service_code: str | None = None,
    ) -> list[RateQuote]:
        """Get shipping rate quotes."""
        rate_data = {
            "carrierCode": carrier_code,
            "fromPostalCode": from_postal_code,
            "toPostalCode": to_postal_code,
            "toCountry": to_country,
            "fromCountry": from_country,
            "weight": {"value": weight_oz, "units": "ounces"},
        }
        if service_code:
            rate_data["serviceCode"] = service_code

        data = await self._request("POST", "/shipments/getrates", json=rate_data)
        return [RateQuote.from_api(r) for r in data]

    # -------------------------------------------------------------------------
    # Warehouses
    # -------------------------------------------------------------------------

    async def list_warehouses(self) -> list[Warehouse]:
        """List warehouses."""
        data = await self._request("GET", "/warehouses")
        return [Warehouse.from_api(w) for w in data]

    async def get_warehouse(self, warehouse_id: int) -> Warehouse:
        """Get a warehouse by ID."""
        data = await self._request("GET", f"/warehouses/{warehouse_id}")
        return Warehouse.from_api(data)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        try:
            return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f")
        except (ValueError, AttributeError):
            return None


# -----------------------------------------------------------------------------
# Mock Data
# -----------------------------------------------------------------------------


def get_mock_order() -> ShipStationOrder:
    return ShipStationOrder(
        order_id=12345,
        order_number="ORD-1001",
        order_status=OrderStatus.AWAITING_SHIPMENT,
        customer_email="customer@example.com",
        ship_to=ShipStationAddress(
            name="John Doe",
            street1="123 Main St",
            city="Los Angeles",
            state="CA",
            postal_code="90210",
            country="US",
        ),
        items=[
            OrderItem(
                sku="SKU-001",
                name="Widget",
                quantity=2,
                unit_price=Decimal("19.99"),
            )
        ],
        amount_paid=Decimal("39.98"),
    )


def get_mock_shipment() -> Shipment:
    return Shipment(
        shipment_id=67890,
        order_id=12345,
        order_number="ORD-1001",
        carrier_code="fedex",
        service_code="fedex_ground",
        tracking_number="1234567890",
        ship_cost=Decimal("8.50"),
    )
