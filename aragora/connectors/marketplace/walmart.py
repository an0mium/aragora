"""
Walmart Seller Center Connector.

Integration with Walmart Marketplace API:
- Orders and fulfillment
- Inventory management
- Product catalog (items)
- Pricing and promotions
- Returns and refunds
- Reports and analytics

Requires Walmart Client ID and Client Secret.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OrderStatus(str, Enum):
    """Walmart order status."""

    CREATED = "Created"
    ACKNOWLEDGED = "Acknowledged"
    SHIPPED = "Shipped"
    DELIVERED = "Delivered"
    CANCELLED = "Cancelled"
    REFUND = "Refund"


class ItemPublishStatus(str, Enum):
    """Item publish status."""

    PUBLISHED = "PUBLISHED"
    UNPUBLISHED = "UNPUBLISHED"
    STAGE = "STAGE"
    IN_PROGRESS = "IN_PROGRESS"
    SYSTEM_PROBLEM = "SYSTEM_PROBLEM"


class LifecycleStatus(str, Enum):
    """Item lifecycle status."""

    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    RETIRED = "RETIRED"


class FulfillmentType(str, Enum):
    """Fulfillment type."""

    SELLER = "SELLER"  # Seller fulfilled
    WFS = "WFS"  # Walmart Fulfillment Services


class ReturnStatus(str, Enum):
    """Return status."""

    INITIATED = "INITIATED"
    IN_TRANSIT = "IN_TRANSIT"
    RECEIVED = "RECEIVED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


@dataclass
class WalmartCredentials:
    """Walmart API credentials."""

    client_id: str
    client_secret: str
    environment: str = "production"  # or "sandbox"

    @property
    def base_url(self) -> str:
        if self.environment == "sandbox":
            return "https://sandbox.walmartapis.com"
        return "https://marketplace.walmartapis.com"


@dataclass
class WalmartAddress:
    """Shipping/billing address."""

    name: str
    address1: str
    city: str
    state: str
    postal_code: str
    country: str = "USA"
    address2: str | None = None
    phone: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> WalmartAddress:
        """Create from API response."""
        return cls(
            name=data.get("name", ""),
            address1=data.get("address1", ""),
            address2=data.get("address2"),
            city=data.get("city", ""),
            state=data.get("state", ""),
            postal_code=data.get("postalCode", ""),
            country=data.get("country", "USA"),
            phone=data.get("phone"),
        )


@dataclass
class OrderLine:
    """Order line item."""

    line_number: str
    item_id: str
    sku: str
    product_name: str
    quantity: int
    unit_price: Decimal
    total_price: Decimal
    status: str
    fulfillment_type: FulfillmentType = FulfillmentType.SELLER
    tracking_number: str | None = None
    carrier: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> OrderLine:
        """Create from API response."""
        charges = data.get("charges", {}).get("charge", [])
        unit_price = Decimal("0")
        for charge in charges:
            if charge.get("chargeType") == "PRODUCT":
                unit_price = Decimal(str(charge.get("chargeAmount", {}).get("amount", 0)))
                break

        item = data.get("item", {})
        return cls(
            line_number=data.get("lineNumber", ""),
            item_id=item.get("productId", ""),
            sku=item.get("sku", ""),
            product_name=item.get("productName", ""),
            quantity=int(data.get("orderLineQuantity", {}).get("amount", 1)),
            unit_price=unit_price,
            total_price=unit_price * int(data.get("orderLineQuantity", {}).get("amount", 1)),
            status=data.get("orderLineStatuses", {})
            .get("orderLineStatus", [{}])[0]
            .get("status", ""),
            fulfillment_type=FulfillmentType(
                data.get("fulfillment", {}).get("fulfillmentOption", "SELLER")
            ),
            tracking_number=data.get("orderLineStatuses", {})
            .get("orderLineStatus", [{}])[0]
            .get("trackingInfo", {})
            .get("trackingNumber"),
            carrier=data.get("orderLineStatuses", {})
            .get("orderLineStatus", [{}])[0]
            .get("trackingInfo", {})
            .get("carrierName", {})
            .get("carrier"),
        )


@dataclass
class WalmartOrder:
    """Walmart marketplace order."""

    purchase_order_id: str
    customer_order_id: str
    order_date: datetime
    status: OrderStatus
    shipping_address: WalmartAddress
    order_lines: list[OrderLine] = field(default_factory=list)
    total_amount: Decimal = Decimal("0")
    ship_by_date: datetime | None = None
    deliver_by_date: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> WalmartOrder:
        """Create from API response."""
        shipping_info = data.get("shippingInfo", {}).get("postalAddress", {})
        order_lines = [
            OrderLine.from_api(line) for line in data.get("orderLines", {}).get("orderLine", [])
        ]
        total = sum((line.total_price for line in order_lines), Decimal(0))

        return cls(
            purchase_order_id=data.get("purchaseOrderId", ""),
            customer_order_id=data.get("customerOrderId", ""),
            order_date=_parse_datetime(data.get("orderDate")) or datetime.now(),
            status=OrderStatus(data.get("orderStatus", "Created")),
            shipping_address=WalmartAddress.from_api(shipping_info),
            order_lines=order_lines,
            total_amount=total,
            ship_by_date=_parse_datetime(data.get("shipByDate")),
            deliver_by_date=_parse_datetime(data.get("deliverByDate")),
        )


@dataclass
class WalmartItem:
    """Walmart catalog item."""

    item_id: str
    sku: str
    product_name: str
    brand: str
    price: Decimal
    publish_status: ItemPublishStatus
    lifecycle_status: LifecycleStatus
    upc: str | None = None
    gtin: str | None = None
    image_url: str | None = None
    product_type: str | None = None
    shelf: list[str] = field(default_factory=list)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> WalmartItem:
        """Create from API response."""
        return cls(
            item_id=data.get("itemId", data.get("wpid", "")),
            sku=data.get("sku", ""),
            product_name=data.get("productName", ""),
            brand=data.get("brand", ""),
            price=Decimal(str(data.get("price", {}).get("amount", 0))),
            publish_status=ItemPublishStatus(data.get("publishedStatus", "UNPUBLISHED")),
            lifecycle_status=LifecycleStatus(data.get("lifecycleStatus", "ACTIVE")),
            upc=data.get("upc"),
            gtin=data.get("gtin"),
            image_url=data.get("productImageUrl"),
            product_type=data.get("productType"),
            shelf=data.get("shelf", []),
        )


@dataclass
class InventoryItem:
    """Inventory information."""

    sku: str
    quantity: int
    fulfillment_lag_time: int = 1
    ship_node: str | None = None
    last_updated: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> InventoryItem:
        """Create from API response."""
        return cls(
            sku=data.get("sku", ""),
            quantity=int(data.get("quantity", {}).get("amount", 0)),
            fulfillment_lag_time=int(data.get("fulfillmentLagTime", 1)),
            ship_node=data.get("shipNode"),
            last_updated=_parse_datetime(data.get("lastUpdatedDate")),
        )


@dataclass
class WalmartReturn:
    """Return request."""

    return_order_id: str
    customer_order_id: str
    return_date: datetime
    status: ReturnStatus
    return_lines: list[dict[str, Any]] = field(default_factory=list)
    refund_amount: Decimal = Decimal("0")

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> WalmartReturn:
        """Create from API response."""
        return cls(
            return_order_id=data.get("returnOrderId", ""),
            customer_order_id=data.get("customerOrderId", ""),
            return_date=_parse_datetime(data.get("returnDate")) or datetime.now(),
            status=ReturnStatus(data.get("returnStatus", "INITIATED")),
            return_lines=data.get("returnLines", []),
            refund_amount=Decimal(str(data.get("refundAmount", {}).get("amount", 0))),
        )


@dataclass
class FeedStatus:
    """Feed submission status."""

    feed_id: str
    feed_type: str
    status: str
    items_received: int = 0
    items_succeeded: int = 0
    items_failed: int = 0
    submitted_at: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> FeedStatus:
        """Create from API response."""
        return cls(
            feed_id=data.get("feedId", ""),
            feed_type=data.get("feedType", ""),
            status=data.get("feedStatus", ""),
            items_received=int(data.get("itemsReceived", 0)),
            items_succeeded=int(data.get("itemsSucceeded", 0)),
            items_failed=int(data.get("itemsFailed", 0)),
            submitted_at=_parse_datetime(data.get("feedDate")),
        )


class WalmartError(Exception):
    """Walmart API error."""

    def __init__(self, message: str, code: str | None = None, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class WalmartConnector:
    """
    Walmart Seller Center API connector.

    Provides integration with Walmart Marketplace for:
    - Order management and fulfillment
    - Inventory tracking
    - Product catalog management
    - Returns processing
    - Feed management
    """

    def __init__(self, credentials: WalmartCredentials):
        self.credentials = credentials
        self._client: httpx.AsyncClient | None = None
        self._access_token: str | None = None
        self._token_expires_at: datetime | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.credentials.base_url,
                timeout=30.0,
            )
        return self._client

    async def _ensure_token(self) -> str:
        """Ensure we have a valid access token."""
        if (
            self._access_token
            and self._token_expires_at
            and datetime.now() < self._token_expires_at
        ):
            return self._access_token

        client = await self._get_client()
        response = await client.post(
            "/v3/token",
            data={
                "grant_type": "client_credentials",
            },
            auth=(self.credentials.client_id, self.credentials.client_secret),
            headers={
                "Accept": "application/json",
                "WM_SVC.NAME": "Walmart Marketplace",
                "WM_QOS.CORRELATION_ID": str(uuid.uuid4()),
            },
        )
        response.raise_for_status()
        data = response.json()
        self._access_token = data["access_token"]
        expires_in = int(data.get("expires_in", 900))
        self._token_expires_at = datetime.now().replace(second=0)
        from datetime import timedelta

        self._token_expires_at += timedelta(seconds=expires_in - 60)
        return self._access_token

    def _get_headers(self, token: str) -> dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "WM_SVC.NAME": "Walmart Marketplace",
            "WM_QOS.CORRELATION_ID": str(uuid.uuid4()),
        }

    async def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json_data: dict | None = None,
    ) -> dict[str, Any]:
        """Make authenticated API request."""
        token = await self._ensure_token()
        client = await self._get_client()
        headers = self._get_headers(token)

        response = await client.request(
            method,
            path,
            params=params,
            json=json_data,
            headers=headers,
        )

        if response.status_code >= 400:
            try:
                error_data = response.json()
                errors = error_data.get("errors", [{}])
                error = errors[0] if errors else {}
                raise WalmartError(
                    message=error.get("description", response.text),
                    code=error.get("code"),
                    details=error_data,
                )
            except (ValueError, KeyError):
                raise WalmartError(f"HTTP {response.status_code}: {response.text}")

        if response.status_code == 204:
            return {}
        return response.json()

    # =========================================================================
    # Orders
    # =========================================================================

    async def get_orders(
        self,
        status: OrderStatus | None = None,
        created_start_date: datetime | None = None,
        created_end_date: datetime | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[WalmartOrder], str | None]:
        """
        Get orders with optional filtering.

        Returns tuple of (orders, next_cursor).
        """
        params: dict[str, Any] = {"limit": min(limit, 200)}

        if status:
            params["status"] = status.value
        if created_start_date:
            params["createdStartDate"] = created_start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        if created_end_date:
            params["createdEndDate"] = created_end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        if cursor:
            params["nextCursor"] = cursor

        data = await self._request("GET", "/v3/orders", params=params)

        orders = [
            WalmartOrder.from_api(order)
            for order in data.get("list", {}).get("elements", {}).get("order", [])
        ]
        next_cursor = data.get("list", {}).get("meta", {}).get("nextCursor")

        return orders, next_cursor

    async def get_order(self, purchase_order_id: str) -> WalmartOrder:
        """Get a single order by ID."""
        data = await self._request("GET", f"/v3/orders/{purchase_order_id}")
        return WalmartOrder.from_api(data.get("order", data))

    async def acknowledge_order(self, purchase_order_id: str) -> bool:
        """Acknowledge receipt of an order."""
        await self._request("POST", f"/v3/orders/{purchase_order_id}/acknowledge")
        return True

    async def ship_order_lines(
        self,
        purchase_order_id: str,
        line_shipments: list[dict[str, Any]],
    ) -> bool:
        """
        Ship order lines with tracking information.

        line_shipments format:
        [
            {
                "line_number": "1",
                "carrier": "UPS",
                "tracking_number": "1Z...",
                "ship_date": datetime,
                "method_code": "Ground",
            }
        ]
        """
        order_shipment = {
            "orderShipment": {
                "orderLines": {
                    "orderLine": [
                        {
                            "lineNumber": ship["line_number"],
                            "orderLineStatuses": {
                                "orderLineStatus": [
                                    {
                                        "status": "Shipped",
                                        "statusQuantity": {
                                            "unitOfMeasurement": "EACH",
                                            "amount": "1",
                                        },
                                        "trackingInfo": {
                                            "shipDateTime": ship.get(
                                                "ship_date", datetime.now()
                                            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                            "carrierName": {"carrier": ship["carrier"]},
                                            "methodCode": ship.get("method_code", "Standard"),
                                            "trackingNumber": ship["tracking_number"],
                                        },
                                    }
                                ]
                            },
                        }
                        for ship in line_shipments
                    ]
                }
            }
        }

        await self._request(
            "POST",
            f"/v3/orders/{purchase_order_id}/shipping",
            json_data=order_shipment,
        )
        return True

    async def cancel_order_lines(
        self,
        purchase_order_id: str,
        line_numbers: list[str],
    ) -> bool:
        """Cancel specific order lines."""
        order_cancellation = {
            "orderCancellation": {
                "orderLines": {
                    "orderLine": [
                        {
                            "lineNumber": line_num,
                            "orderLineStatuses": {
                                "orderLineStatus": [
                                    {
                                        "status": "Cancelled",
                                        "cancellationReason": "CANCEL_BY_SELLER",
                                        "statusQuantity": {
                                            "unitOfMeasurement": "EACH",
                                            "amount": "1",
                                        },
                                    }
                                ]
                            },
                        }
                        for line_num in line_numbers
                    ]
                }
            }
        }

        await self._request(
            "POST",
            f"/v3/orders/{purchase_order_id}/cancel",
            json_data=order_cancellation,
        )
        return True

    # =========================================================================
    # Inventory
    # =========================================================================

    async def get_inventory(
        self,
        sku: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[InventoryItem]:
        """Get inventory levels."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if sku:
            params["sku"] = sku

        data = await self._request("GET", "/v3/inventory", params=params)

        return [
            InventoryItem.from_api(item) for item in data.get("elements", {}).get("inventories", [])
        ]

    async def update_inventory(
        self,
        sku: str,
        quantity: int,
        fulfillment_lag_time: int = 1,
    ) -> InventoryItem:
        """Update inventory for a SKU."""
        inventory_data = {
            "sku": sku,
            "quantity": {
                "unit": "EACH",
                "amount": quantity,
            },
            "fulfillmentLagTime": fulfillment_lag_time,
        }

        data = await self._request(
            "PUT",
            "/v3/inventory",
            params={"sku": sku},
            json_data=inventory_data,
        )
        return InventoryItem.from_api(data)

    async def bulk_update_inventory(
        self,
        updates: list[dict[str, Any]],
    ) -> FeedStatus:
        """
        Bulk update inventory via feed.

        updates format:
        [
            {"sku": "ABC123", "quantity": 100, "fulfillment_lag_time": 1},
            ...
        ]
        """
        feed_data = {
            "InventoryHeader": {
                "version": "1.4",
            },
            "Inventory": [
                {
                    "sku": u["sku"],
                    "quantity": {
                        "unit": "EACH",
                        "amount": u["quantity"],
                    },
                    "fulfillmentLagTime": u.get("fulfillment_lag_time", 1),
                }
                for u in updates
            ],
        }

        data = await self._request(
            "POST",
            "/v3/feeds",
            params={"feedType": "inventory"},
            json_data=feed_data,
        )
        return FeedStatus.from_api(data)

    # =========================================================================
    # Items / Catalog
    # =========================================================================

    async def get_items(
        self,
        sku: str | None = None,
        publish_status: ItemPublishStatus | None = None,
        lifecycle_status: LifecycleStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[WalmartItem]:
        """Get catalog items."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if sku:
            params["sku"] = sku
        if publish_status:
            params["publishedStatus"] = publish_status.value
        if lifecycle_status:
            params["lifecycleStatus"] = lifecycle_status.value

        data = await self._request("GET", "/v3/items", params=params)

        return [WalmartItem.from_api(item) for item in data.get("ItemResponse", [])]

    async def get_item(self, item_id: str) -> WalmartItem:
        """Get a single item by ID."""
        data = await self._request("GET", f"/v3/items/{item_id}")
        return WalmartItem.from_api(data)

    async def retire_item(self, sku: str) -> bool:
        """Retire/archive an item."""
        await self._request("DELETE", f"/v3/items/{sku}")
        return True

    async def update_price(
        self,
        sku: str,
        price: Decimal,
        compare_at_price: Decimal | None = None,
    ) -> bool:
        """Update item price."""
        pricing: dict[str, Any] = {
            "sku": sku,
            "pricing": [
                {
                    "currentPrice": {
                        "currency": "USD",
                        "amount": float(price),
                    },
                }
            ],
        }

        if compare_at_price:
            pricing["pricing"][0]["comparisonPrice"] = {
                "currency": "USD",
                "amount": float(compare_at_price),
            }

        await self._request("PUT", "/v3/prices", json_data=pricing)
        return True

    async def bulk_update_prices(
        self,
        price_updates: list[dict[str, Any]],
    ) -> FeedStatus:
        """
        Bulk update prices via feed.

        price_updates format:
        [
            {"sku": "ABC123", "price": Decimal("19.99"), "compare_at_price": Decimal("24.99")},
            ...
        ]
        """
        feed_data = {
            "PriceHeader": {
                "version": "1.7",
            },
            "Price": [
                {
                    "sku": u["sku"],
                    "pricing": [
                        {
                            "currentPrice": {
                                "currency": "USD",
                                "amount": float(u["price"]),
                            },
                            **(
                                {
                                    "comparisonPrice": {
                                        "currency": "USD",
                                        "amount": float(u["compare_at_price"]),
                                    }
                                }
                                if u.get("compare_at_price")
                                else {}
                            ),
                        }
                    ],
                }
                for u in price_updates
            ],
        }

        data = await self._request(
            "POST",
            "/v3/feeds",
            params={"feedType": "price"},
            json_data=feed_data,
        )
        return FeedStatus.from_api(data)

    # =========================================================================
    # Returns
    # =========================================================================

    async def get_returns(
        self,
        return_status: ReturnStatus | None = None,
        return_start_date: datetime | None = None,
        return_end_date: datetime | None = None,
        limit: int = 100,
    ) -> list[WalmartReturn]:
        """Get returns."""
        params: dict[str, Any] = {"limit": limit}
        if return_status:
            params["returnStatus"] = return_status.value
        if return_start_date:
            params["returnCreationStartDate"] = return_start_date.strftime("%Y-%m-%d")
        if return_end_date:
            params["returnCreationEndDate"] = return_end_date.strftime("%Y-%m-%d")

        data = await self._request("GET", "/v3/returns", params=params)

        return [WalmartReturn.from_api(ret) for ret in data.get("returnOrders", [])]

    async def issue_refund(
        self,
        return_order_id: str,
        refund_lines: list[dict[str, Any]],
    ) -> bool:
        """
        Issue refund for a return.

        refund_lines format:
        [
            {"return_line_number": "1", "quantity": 1, "refund_amount": Decimal("19.99")},
            ...
        ]
        """
        refund_data = {
            "customerOrderId": return_order_id,
            "refundLines": [
                {
                    "returnLineNumber": line["return_line_number"],
                    "refundComments": "Refund processed",
                    "refundCharges": {
                        "refundCharge": [
                            {
                                "refundChargeType": "PRODUCT",
                                "refundChargeAmount": {
                                    "currency": "USD",
                                    "amount": float(line["refund_amount"]),
                                },
                            }
                        ]
                    },
                }
                for line in refund_lines
            ],
        }

        await self._request(
            "POST",
            f"/v3/returns/{return_order_id}/refund",
            json_data=refund_data,
        )
        return True

    # =========================================================================
    # Feeds
    # =========================================================================

    async def get_feed_status(self, feed_id: str) -> FeedStatus:
        """Get status of a feed submission."""
        data = await self._request("GET", f"/v3/feeds/{feed_id}")
        return FeedStatus.from_api(data)

    async def get_all_feed_statuses(
        self,
        feed_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[FeedStatus]:
        """Get all feed statuses."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if feed_type:
            params["feedType"] = feed_type

        data = await self._request("GET", "/v3/feeds", params=params)

        return [FeedStatus.from_api(feed) for feed in data.get("results", {}).get("feed", [])]

    # =========================================================================
    # Reports
    # =========================================================================

    async def get_available_reports(self) -> list[dict[str, Any]]:
        """Get available report types."""
        data = await self._request("GET", "/v3/reports/reportTypes")
        return data.get("reportTypes", [])

    async def request_report(
        self,
        report_type: str,
        report_version: str = "v1",
    ) -> str:
        """
        Request a new report generation.

        Returns the report request ID.
        """
        data = await self._request(
            "POST",
            "/v3/reports/reportRequests",
            json_data={
                "reportType": report_type,
                "reportVersion": report_version,
            },
        )
        return data.get("requestId", "")

    async def get_report_status(
        self,
        request_id: str,
    ) -> dict[str, Any]:
        """Get report generation status."""
        data = await self._request(
            "GET",
            f"/v3/reports/reportRequests/{request_id}",
        )
        return data

    async def download_report(
        self,
        request_id: str,
    ) -> bytes:
        """Download a generated report."""
        token = await self._ensure_token()
        client = await self._get_client()
        headers = self._get_headers(token)

        response = await client.get(
            "/v3/reports/downloadReport",
            params={"requestId": request_id},
            headers=headers,
        )
        response.raise_for_status()
        return response.content

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> WalmartConnector:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse ISO datetime string."""
    if not value:
        return None
    try:
        # Handle various formats
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None
    except Exception:
        return None


def get_mock_order() -> WalmartOrder:
    """Get a mock order for testing."""
    return WalmartOrder(
        purchase_order_id="2024010112345",
        customer_order_id="ABC12345",
        order_date=datetime.now(),
        status=OrderStatus.ACKNOWLEDGED,
        shipping_address=WalmartAddress(
            name="John Doe",
            address1="123 Main St",
            city="Bentonville",
            state="AR",
            postal_code="72712",
        ),
        order_lines=[
            OrderLine(
                line_number="1",
                item_id="12345678",
                sku="TEST-SKU-001",
                product_name="Test Product",
                quantity=2,
                unit_price=Decimal("29.99"),
                total_price=Decimal("59.98"),
                status="Acknowledged",
            )
        ],
        total_amount=Decimal("59.98"),
    )


def get_mock_item() -> WalmartItem:
    """Get a mock item for testing."""
    return WalmartItem(
        item_id="12345678",
        sku="TEST-SKU-001",
        product_name="Test Product",
        brand="Test Brand",
        price=Decimal("29.99"),
        publish_status=ItemPublishStatus.PUBLISHED,
        lifecycle_status=LifecycleStatus.ACTIVE,
        upc="012345678901",
    )
