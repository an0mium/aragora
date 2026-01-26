"""
WooCommerce Connector.

Provides integration with WooCommerce (WordPress) stores:
- OAuth 1.0a / Basic auth
- Orders sync and management
- Products and variations
- Customers
- Inventory tracking
- Webhooks

Dependencies:
    pip install woocommerce

Environment Variables:
    WOOCOMMERCE_URL - Store URL (https://store.example.com)
    WOOCOMMERCE_CONSUMER_KEY - API consumer key
    WOOCOMMERCE_CONSUMER_SECRET - API consumer secret
    WOOCOMMERCE_VERSION - API version (wc/v3)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

from aragora.connectors.base import Evidence
from aragora.connectors.enterprise.base import EnterpriseConnector, SyncItem, SyncResult, SyncState
from aragora.connectors.exceptions import ConnectorAPIError
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)


class WooOrderStatus(str, Enum):
    """WooCommerce order status."""

    PENDING = "pending"
    PROCESSING = "processing"
    ON_HOLD = "on-hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    FAILED = "failed"
    TRASH = "trash"


class WooProductStatus(str, Enum):
    """WooCommerce product status."""

    PUBLISH = "publish"
    DRAFT = "draft"
    PENDING = "pending"
    PRIVATE = "private"


class WooProductType(str, Enum):
    """WooCommerce product type."""

    SIMPLE = "simple"
    VARIABLE = "variable"
    GROUPED = "grouped"
    EXTERNAL = "external"


class WooStockStatus(str, Enum):
    """WooCommerce stock status."""

    IN_STOCK = "instock"
    OUT_OF_STOCK = "outofstock"
    ON_BACKORDER = "onbackorder"


@dataclass
class WooCommerceCredentials:
    """WooCommerce API credentials."""

    store_url: str
    consumer_key: str
    consumer_secret: str
    api_version: str = "wc/v3"
    timeout: int = 30

    @classmethod
    def from_env(cls) -> "WooCommerceCredentials":
        """Create credentials from environment variables."""
        return cls(
            store_url=os.environ.get("WOOCOMMERCE_URL", ""),
            consumer_key=os.environ.get("WOOCOMMERCE_CONSUMER_KEY", ""),
            consumer_secret=os.environ.get("WOOCOMMERCE_CONSUMER_SECRET", ""),
            api_version=os.environ.get("WOOCOMMERCE_VERSION", "wc/v3"),
        )


@dataclass
class WooAddress:
    """WooCommerce address."""

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    address_1: Optional[str] = None
    address_2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postcode: Optional[str] = None
    country: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "firstName": self.first_name,
            "lastName": self.last_name,
            "company": self.company,
            "address1": self.address_1,
            "address2": self.address_2,
            "city": self.city,
            "state": self.state,
            "postcode": self.postcode,
            "country": self.country,
            "email": self.email,
            "phone": self.phone,
        }


@dataclass
class WooLineItem:
    """WooCommerce order line item."""

    id: int
    product_id: int
    variation_id: int
    name: str
    quantity: int
    subtotal: Decimal
    total: Decimal
    sku: Optional[str] = None
    price: Decimal = Decimal("0.00")
    tax_class: Optional[str] = None
    taxes: List[Dict[str, Any]] = field(default_factory=list)
    meta_data: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "productId": self.product_id,
            "variationId": self.variation_id,
            "name": self.name,
            "quantity": self.quantity,
            "subtotal": str(self.subtotal),
            "total": str(self.total),
            "sku": self.sku,
            "price": str(self.price),
            "taxClass": self.tax_class,
        }


@dataclass
class WooOrder:
    """WooCommerce order."""

    id: int
    number: str
    order_key: str
    status: WooOrderStatus
    currency: str
    date_created: datetime
    date_modified: datetime
    total: Decimal
    subtotal: Decimal
    total_tax: Decimal
    shipping_total: Decimal
    discount_total: Decimal
    payment_method: str
    payment_method_title: str
    customer_id: int
    billing: WooAddress
    shipping: WooAddress
    line_items: List[WooLineItem] = field(default_factory=list)
    customer_note: Optional[str] = None
    date_paid: Optional[datetime] = None
    date_completed: Optional[datetime] = None
    cart_hash: Optional[str] = None
    transaction_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "number": self.number,
            "orderKey": self.order_key,
            "status": self.status.value,
            "currency": self.currency,
            "dateCreated": self.date_created.isoformat(),
            "dateModified": self.date_modified.isoformat(),
            "total": str(self.total),
            "subtotal": str(self.subtotal),
            "totalTax": str(self.total_tax),
            "shippingTotal": str(self.shipping_total),
            "discountTotal": str(self.discount_total),
            "paymentMethod": self.payment_method,
            "paymentMethodTitle": self.payment_method_title,
            "customerId": self.customer_id,
            "billing": self.billing.to_dict(),
            "shipping": self.shipping.to_dict(),
            "lineItems": [item.to_dict() for item in self.line_items],
            "customerNote": self.customer_note,
            "datePaid": self.date_paid.isoformat() if self.date_paid else None,
            "dateCompleted": self.date_completed.isoformat() if self.date_completed else None,
            "transactionId": self.transaction_id,
        }


@dataclass
class WooProductVariation:
    """WooCommerce product variation."""

    id: int
    sku: Optional[str]
    price: Decimal
    regular_price: Decimal
    sale_price: Optional[Decimal]
    stock_quantity: Optional[int]
    stock_status: WooStockStatus
    manage_stock: bool
    attributes: List[Dict[str, str]] = field(default_factory=list)
    image: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sku": self.sku,
            "price": str(self.price),
            "regularPrice": str(self.regular_price),
            "salePrice": str(self.sale_price) if self.sale_price else None,
            "stockQuantity": self.stock_quantity,
            "stockStatus": self.stock_status.value,
            "manageStock": self.manage_stock,
            "attributes": self.attributes,
            "image": self.image,
        }


@dataclass
class WooProduct:
    """WooCommerce product."""

    id: int
    name: str
    slug: str
    type: WooProductType
    status: WooProductStatus
    sku: Optional[str]
    price: Decimal
    regular_price: Decimal
    sale_price: Optional[Decimal]
    date_created: datetime
    date_modified: datetime
    description: Optional[str]
    short_description: Optional[str]
    stock_quantity: Optional[int]
    stock_status: WooStockStatus
    manage_stock: bool
    categories: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[Dict[str, Any]] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    variations: List[WooProductVariation] = field(default_factory=list)
    attributes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "type": self.type.value,
            "status": self.status.value,
            "sku": self.sku,
            "price": str(self.price),
            "regularPrice": str(self.regular_price),
            "salePrice": str(self.sale_price) if self.sale_price else None,
            "dateCreated": self.date_created.isoformat(),
            "dateModified": self.date_modified.isoformat(),
            "description": self.description,
            "shortDescription": self.short_description,
            "stockQuantity": self.stock_quantity,
            "stockStatus": self.stock_status.value,
            "manageStock": self.manage_stock,
            "categories": self.categories,
            "tags": self.tags,
            "images": self.images,
            "variations": [v.to_dict() for v in self.variations],
            "attributes": self.attributes,
        }


@dataclass
class WooCustomer:
    """WooCommerce customer."""

    id: int
    email: str
    first_name: str
    last_name: str
    username: str
    date_created: datetime
    date_modified: datetime
    billing: WooAddress
    shipping: WooAddress
    is_paying_customer: bool = False
    orders_count: int = 0
    total_spent: Decimal = Decimal("0.00")
    avatar_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "firstName": self.first_name,
            "lastName": self.last_name,
            "username": self.username,
            "dateCreated": self.date_created.isoformat(),
            "dateModified": self.date_modified.isoformat(),
            "billing": self.billing.to_dict(),
            "shipping": self.shipping.to_dict(),
            "isPayingCustomer": self.is_paying_customer,
            "ordersCount": self.orders_count,
            "totalSpent": str(self.total_spent),
            "avatarUrl": self.avatar_url,
        }


class WooCommerceConnector(EnterpriseConnector):
    """
    Connector for WooCommerce (WordPress) stores.

    Supports:
    - Order sync and management
    - Product and variation management
    - Customer data sync
    - Inventory tracking
    - Webhook handling

    Example:
        ```python
        credentials = WooCommerceCredentials.from_env()
        connector = WooCommerceConnector(credentials)

        # Sync orders
        async for order in connector.sync_orders(since=last_sync):
            process_order(order)

        # Update stock
        await connector.update_product_stock(product_id=123, quantity=50)
        ```
    """

    def __init__(self, credentials: WooCommerceCredentials):
        """Initialize WooCommerce connector.

        Args:
            credentials: WooCommerce API credentials
        """
        super().__init__(connector_id="woocommerce", tenant_id="default")
        self._woo_credentials = credentials
        self._client: Any = None

    @property
    def name(self) -> str:
        """Human-readable name for this connector."""
        return "WooCommerce"

    @property
    def source_type(self) -> SourceType:
        """The source type for this connector."""
        return SourceType.EXTERNAL_API

    @property
    def base_url(self) -> str:
        """Get API base URL."""
        return f"{self._woo_credentials.store_url}/wp-json/{self._woo_credentials.api_version}"

    async def connect(self) -> bool:
        """Establish connection to WooCommerce API."""
        try:
            import aiohttp
            from aiohttp import BasicAuth

            auth = BasicAuth(
                self._woo_credentials.consumer_key,
                self._woo_credentials.consumer_secret,
            )
            self._client = aiohttp.ClientSession(auth=auth)

            # Test connection
            async with self._client.get(f"{self.base_url}/system_status") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(
                        f"Connected to WooCommerce store: {data.get('environment', {}).get('site_url', 'unknown')}"
                    )
                    return True
                else:
                    logger.error(f"Failed to connect: HTTP {resp.status}")
                    return False

        except ImportError:
            logger.error("aiohttp package not installed. Run: pip install aiohttp")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to WooCommerce: {e}")
            return False

    async def disconnect(self) -> None:
        """Close WooCommerce connection."""
        if self._client:
            await self._client.close()
            self._client = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make an API request.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            json_data: JSON body data

        Returns:
            Response data
        """
        if not self._client:
            await self.connect()

        url = f"{self.base_url}/{endpoint}"
        async with self._client.request(method, url, params=params, json=json_data) as resp:
            if resp.status >= 400:
                error_text = await resp.text()
                raise ConnectorAPIError(
                    f"WooCommerce API error: {error_text}",
                    connector_name="woocommerce",
                    status_code=resp.status,
                )
            return await resp.json()

    # =========================================================================
    # Orders
    # =========================================================================

    async def sync_orders(
        self,
        since: Optional[datetime] = None,
        status: Optional[WooOrderStatus] = None,
        per_page: int = 100,
    ) -> AsyncIterator[WooOrder]:
        """Sync orders from WooCommerce.

        Args:
            since: Only fetch orders modified after this time
            status: Filter by order status
            per_page: Page size

        Yields:
            WooOrder objects
        """
        params: Dict[str, Any] = {"per_page": per_page, "page": 1}
        if since:
            params["modified_after"] = since.isoformat()
        if status:
            params["status"] = status.value

        while True:
            data = await self._request("GET", "orders", params=params)

            if not data:
                break

            for order_data in data:
                yield self._parse_order(order_data)

            if len(data) < per_page:
                break

            params["page"] += 1

    def _parse_order(self, data: Dict[str, Any]) -> WooOrder:
        """Parse order from API response."""
        line_items = [
            WooLineItem(
                id=item["id"],
                product_id=item["product_id"],
                variation_id=item.get("variation_id", 0),
                name=item["name"],
                quantity=item["quantity"],
                subtotal=Decimal(str(item["subtotal"])),
                total=Decimal(str(item["total"])),
                sku=item.get("sku"),
                price=Decimal(str(item.get("price", "0"))),
                tax_class=item.get("tax_class"),
            )
            for item in data.get("line_items", [])
        ]

        return WooOrder(
            id=data["id"],
            number=data["number"],
            order_key=data["order_key"],
            status=WooOrderStatus(data["status"]),
            currency=data["currency"],
            date_created=datetime.fromisoformat(data["date_created_gmt"] + "+00:00"),
            date_modified=datetime.fromisoformat(data["date_modified_gmt"] + "+00:00"),
            total=Decimal(str(data["total"])),
            subtotal=Decimal(str(data.get("subtotal", "0"))),
            total_tax=Decimal(str(data["total_tax"])),
            shipping_total=Decimal(str(data["shipping_total"])),
            discount_total=Decimal(str(data["discount_total"])),
            payment_method=data["payment_method"],
            payment_method_title=data["payment_method_title"],
            customer_id=data["customer_id"],
            billing=self._parse_address(data.get("billing", {})),
            shipping=self._parse_address(data.get("shipping", {})),
            line_items=line_items,
            customer_note=data.get("customer_note"),
            date_paid=(
                datetime.fromisoformat(data["date_paid_gmt"] + "+00:00")
                if data.get("date_paid_gmt")
                else None
            ),
            date_completed=(
                datetime.fromisoformat(data["date_completed_gmt"] + "+00:00")
                if data.get("date_completed_gmt")
                else None
            ),
            transaction_id=data.get("transaction_id"),
        )

    def _parse_address(self, data: Dict[str, Any]) -> WooAddress:
        """Parse address from API response."""
        return WooAddress(
            first_name=data.get("first_name"),
            last_name=data.get("last_name"),
            company=data.get("company"),
            address_1=data.get("address_1"),
            address_2=data.get("address_2"),
            city=data.get("city"),
            state=data.get("state"),
            postcode=data.get("postcode"),
            country=data.get("country"),
            email=data.get("email"),
            phone=data.get("phone"),
        )

    async def get_order(self, order_id: int) -> Optional[WooOrder]:
        """Get a single order by ID."""
        try:
            data = await self._request("GET", f"orders/{order_id}")
            return self._parse_order(data)
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    async def update_order_status(
        self,
        order_id: int,
        status: WooOrderStatus,
    ) -> bool:
        """Update order status.

        Args:
            order_id: Order ID
            status: New status

        Returns:
            True if successful
        """
        try:
            await self._request(
                "PUT",
                f"orders/{order_id}",
                json_data={"status": status.value},
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update order {order_id}: {e}")
            return False

    async def create_order(
        self,
        customer_id: Optional[int] = None,
        billing: Optional[WooAddress] = None,
        shipping: Optional[WooAddress] = None,
        line_items: Optional[List[Dict[str, Any]]] = None,
        payment_method: str = "",
        payment_method_title: str = "",
        set_paid: bool = False,
        note: Optional[str] = None,
    ) -> Optional[WooOrder]:
        """Create a new order.

        Args:
            customer_id: Customer ID (0 for guest)
            billing: Billing address
            shipping: Shipping address
            line_items: List of line items [{"product_id": 123, "quantity": 1}]
            payment_method: Payment method ID
            payment_method_title: Payment method title
            set_paid: Mark order as paid
            note: Customer note

        Returns:
            Created WooOrder or None on failure
        """
        try:
            order_data: Dict[str, Any] = {
                "payment_method": payment_method,
                "payment_method_title": payment_method_title,
                "set_paid": set_paid,
            }

            if customer_id is not None:
                order_data["customer_id"] = customer_id
            if billing:
                order_data["billing"] = {
                    "first_name": billing.first_name or "",
                    "last_name": billing.last_name or "",
                    "company": billing.company or "",
                    "address_1": billing.address_1 or "",
                    "address_2": billing.address_2 or "",
                    "city": billing.city or "",
                    "state": billing.state or "",
                    "postcode": billing.postcode or "",
                    "country": billing.country or "",
                    "email": billing.email or "",
                    "phone": billing.phone or "",
                }
            if shipping:
                order_data["shipping"] = {
                    "first_name": shipping.first_name or "",
                    "last_name": shipping.last_name or "",
                    "company": shipping.company or "",
                    "address_1": shipping.address_1 or "",
                    "address_2": shipping.address_2 or "",
                    "city": shipping.city or "",
                    "state": shipping.state or "",
                    "postcode": shipping.postcode or "",
                    "country": shipping.country or "",
                }
            if line_items:
                order_data["line_items"] = line_items
            if note:
                order_data["customer_note"] = note

            data = await self._request("POST", "orders", json_data=order_data)
            return self._parse_order(data)
        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            return None

    async def get_order_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get order statistics.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            Dictionary with order statistics
        """
        total_orders = 0
        total_revenue = Decimal("0.00")
        completed_orders = 0
        cancelled_orders = 0
        refunded_orders = 0
        status_counts: Dict[str, int] = {}

        async for order in self.sync_orders(since=start_date):
            if end_date and order.date_created > end_date:
                continue

            total_orders += 1
            total_revenue += order.total

            status = order.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

            if order.status == WooOrderStatus.COMPLETED:
                completed_orders += 1
            elif order.status == WooOrderStatus.CANCELLED:
                cancelled_orders += 1
            elif order.status == WooOrderStatus.REFUNDED:
                refunded_orders += 1

        return {
            "total_orders": total_orders,
            "total_revenue": str(total_revenue),
            "completed_orders": completed_orders,
            "cancelled_orders": cancelled_orders,
            "refunded_orders": refunded_orders,
            "completion_rate": completed_orders / total_orders if total_orders > 0 else 0,
            "average_order_value": (
                str(total_revenue / total_orders) if total_orders > 0 else "0.00"
            ),
            "status_breakdown": status_counts,
        }

    # =========================================================================
    # Products
    # =========================================================================

    async def sync_products(
        self,
        since: Optional[datetime] = None,
        status: Optional[WooProductStatus] = None,
        per_page: int = 100,
    ) -> AsyncIterator[WooProduct]:
        """Sync products from WooCommerce.

        Args:
            since: Only fetch products modified after this time
            status: Filter by product status
            per_page: Page size

        Yields:
            WooProduct objects
        """
        params: Dict[str, Any] = {"per_page": per_page, "page": 1}
        if since:
            params["modified_after"] = since.isoformat()
        if status:
            params["status"] = status.value

        while True:
            data = await self._request("GET", "products", params=params)

            if not data:
                break

            for product_data in data:
                yield self._parse_product(product_data)

            if len(data) < per_page:
                break

            params["page"] += 1

    def _parse_product(self, data: Dict[str, Any]) -> WooProduct:
        """Parse product from API response."""
        return WooProduct(
            id=data["id"],
            name=data["name"],
            slug=data["slug"],
            type=WooProductType(data.get("type", "simple")),
            status=WooProductStatus(data.get("status", "publish")),
            sku=data.get("sku"),
            price=Decimal(str(data.get("price", "0") or "0")),
            regular_price=Decimal(str(data.get("regular_price", "0") or "0")),
            sale_price=Decimal(str(data["sale_price"])) if data.get("sale_price") else None,
            date_created=datetime.fromisoformat(data["date_created_gmt"] + "+00:00"),
            date_modified=datetime.fromisoformat(data["date_modified_gmt"] + "+00:00"),
            description=data.get("description"),
            short_description=data.get("short_description"),
            stock_quantity=data.get("stock_quantity"),
            stock_status=WooStockStatus(data.get("stock_status", "instock")),
            manage_stock=data.get("manage_stock", False),
            categories=data.get("categories", []),
            tags=data.get("tags", []),
            images=[img["src"] for img in data.get("images", [])],
            attributes=data.get("attributes", []),
        )

    async def get_product(self, product_id: int) -> Optional[WooProduct]:
        """Get a single product by ID."""
        try:
            data = await self._request("GET", f"products/{product_id}")
            return self._parse_product(data)
        except Exception as e:
            logger.error(f"Failed to get product {product_id}: {e}")
            return None

    async def update_product_stock(
        self,
        product_id: int,
        quantity: int,
        stock_status: Optional[WooStockStatus] = None,
    ) -> bool:
        """Update product stock.

        Args:
            product_id: Product ID
            quantity: New stock quantity
            stock_status: Optional stock status

        Returns:
            True if successful
        """
        try:
            update_data: Dict[str, Any] = {
                "stock_quantity": quantity,
                "manage_stock": True,
            }
            if stock_status:
                update_data["stock_status"] = stock_status.value

            await self._request(
                "PUT",
                f"products/{product_id}",
                json_data=update_data,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update product stock {product_id}: {e}")
            return False

    async def get_low_stock_products(
        self,
        threshold: int = 5,
    ) -> List[WooProduct]:
        """Get products with low stock.

        Args:
            threshold: Stock level threshold

        Returns:
            List of low stock products
        """
        low_stock = []
        async for product in self.sync_products():
            if product.manage_stock and product.stock_quantity is not None:
                if product.stock_quantity <= threshold:
                    low_stock.append(product)
        return low_stock

    async def sync_product_variations(
        self,
        product_id: int,
        per_page: int = 100,
    ) -> AsyncIterator[WooProductVariation]:
        """Sync variations for a variable product.

        Args:
            product_id: Parent product ID
            per_page: Page size

        Yields:
            WooProductVariation objects
        """
        params: Dict[str, Any] = {"per_page": per_page, "page": 1}

        while True:
            data = await self._request("GET", f"products/{product_id}/variations", params=params)

            if not data:
                break

            for variation_data in data:
                yield self._parse_variation(variation_data)

            if len(data) < per_page:
                break

            params["page"] += 1

    def _parse_variation(self, data: Dict[str, Any]) -> WooProductVariation:
        """Parse variation from API response."""
        return WooProductVariation(
            id=data["id"],
            sku=data.get("sku"),
            price=Decimal(str(data.get("price", "0") or "0")),
            regular_price=Decimal(str(data.get("regular_price", "0") or "0")),
            sale_price=Decimal(str(data["sale_price"])) if data.get("sale_price") else None,
            stock_quantity=data.get("stock_quantity"),
            stock_status=WooStockStatus(data.get("stock_status", "instock")),
            manage_stock=data.get("manage_stock", False),
            attributes=data.get("attributes", []),
            image=data.get("image", {}).get("src") if data.get("image") else None,
        )

    async def update_variation_stock(
        self,
        product_id: int,
        variation_id: int,
        quantity: int,
        stock_status: Optional[WooStockStatus] = None,
    ) -> bool:
        """Update variation stock.

        Args:
            product_id: Parent product ID
            variation_id: Variation ID
            quantity: New stock quantity
            stock_status: Optional stock status

        Returns:
            True if successful
        """
        try:
            update_data: Dict[str, Any] = {
                "stock_quantity": quantity,
                "manage_stock": True,
            }
            if stock_status:
                update_data["stock_status"] = stock_status.value

            await self._request(
                "PUT",
                f"products/{product_id}/variations/{variation_id}",
                json_data=update_data,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update variation stock {variation_id}: {e}")
            return False

    # =========================================================================
    # Customers
    # =========================================================================

    async def sync_customers(
        self,
        since: Optional[datetime] = None,
        per_page: int = 100,
    ) -> AsyncIterator[WooCustomer]:
        """Sync customers from WooCommerce.

        Args:
            since: Only fetch customers modified after this time
            per_page: Page size

        Yields:
            WooCustomer objects
        """
        params: Dict[str, Any] = {"per_page": per_page, "page": 1}
        if since:
            params["modified_after"] = since.isoformat()

        while True:
            data = await self._request("GET", "customers", params=params)

            if not data:
                break

            for customer_data in data:
                yield self._parse_customer(customer_data)

            if len(data) < per_page:
                break

            params["page"] += 1

    def _parse_customer(self, data: Dict[str, Any]) -> WooCustomer:
        """Parse customer from API response."""
        return WooCustomer(
            id=data["id"],
            email=data["email"],
            first_name=data["first_name"],
            last_name=data["last_name"],
            username=data["username"],
            date_created=datetime.fromisoformat(data["date_created_gmt"] + "+00:00"),
            date_modified=datetime.fromisoformat(data["date_modified_gmt"] + "+00:00"),
            billing=self._parse_address(data.get("billing", {})),
            shipping=self._parse_address(data.get("shipping", {})),
            is_paying_customer=data.get("is_paying_customer", False),
            orders_count=data.get("orders_count", 0),
            total_spent=Decimal(str(data.get("total_spent", "0"))),
            avatar_url=data.get("avatar_url"),
        )

    # =========================================================================
    # Refunds
    # =========================================================================

    async def create_refund(
        self,
        order_id: int,
        amount: Optional[Decimal] = None,
        reason: str = "",
        line_items: Optional[List[Dict[str, Any]]] = None,
        restock_items: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Create a refund for an order.

        Args:
            order_id: Order ID to refund
            amount: Refund amount (if None, refunds full order)
            reason: Reason for refund
            line_items: Specific line items to refund [{"id": 123, "quantity": 1}]
            restock_items: Whether to restock items

        Returns:
            Refund data or None on failure
        """
        try:
            refund_data: Dict[str, Any] = {
                "reason": reason,
                "restock_items": restock_items,
            }

            if amount is not None:
                refund_data["amount"] = str(amount)
            if line_items:
                refund_data["line_items"] = line_items

            data = await self._request(
                "POST",
                f"orders/{order_id}/refunds",
                json_data=refund_data,
            )
            return data
        except Exception as e:
            logger.error(f"Failed to create refund for order {order_id}: {e}")
            return None

    async def get_refunds(self, order_id: int) -> List[Dict[str, Any]]:
        """Get refunds for an order.

        Args:
            order_id: Order ID

        Returns:
            List of refund data
        """
        try:
            data = await self._request("GET", f"orders/{order_id}/refunds")
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Failed to get refunds for order {order_id}: {e}")
            return []

    # =========================================================================
    # Coupons
    # =========================================================================

    async def get_coupons(
        self,
        per_page: int = 100,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Sync coupons from WooCommerce.

        Args:
            per_page: Page size

        Yields:
            Coupon data dictionaries
        """
        params: Dict[str, Any] = {"per_page": per_page, "page": 1}

        while True:
            data = await self._request("GET", "coupons", params=params)

            if not data:
                break

            for coupon in data:
                yield coupon

            if len(data) < per_page:
                break

            params["page"] += 1

    async def create_coupon(
        self,
        code: str,
        discount_type: str = "percent",
        amount: str = "0",
        description: str = "",
        date_expires: Optional[datetime] = None,
        individual_use: bool = False,
        usage_limit: Optional[int] = None,
        product_ids: Optional[List[int]] = None,
        excluded_product_ids: Optional[List[int]] = None,
        minimum_amount: Optional[str] = None,
        maximum_amount: Optional[str] = None,
        free_shipping: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Create a new coupon.

        Args:
            code: Coupon code
            discount_type: Type (percent, fixed_cart, fixed_product)
            amount: Discount amount
            description: Coupon description
            date_expires: Expiration date
            individual_use: Cannot be combined with other coupons
            usage_limit: Total usage limit
            product_ids: Products coupon applies to
            excluded_product_ids: Products coupon doesn't apply to
            minimum_amount: Minimum order amount
            maximum_amount: Maximum order amount
            free_shipping: Whether coupon grants free shipping

        Returns:
            Created coupon data or None on failure
        """
        try:
            coupon_data: Dict[str, Any] = {
                "code": code,
                "discount_type": discount_type,
                "amount": amount,
                "description": description,
                "individual_use": individual_use,
                "free_shipping": free_shipping,
            }

            if date_expires:
                coupon_data["date_expires"] = date_expires.isoformat()
            if usage_limit is not None:
                coupon_data["usage_limit"] = usage_limit
            if product_ids:
                coupon_data["product_ids"] = product_ids
            if excluded_product_ids:
                coupon_data["excluded_product_ids"] = excluded_product_ids
            if minimum_amount:
                coupon_data["minimum_amount"] = minimum_amount
            if maximum_amount:
                coupon_data["maximum_amount"] = maximum_amount

            data = await self._request("POST", "coupons", json_data=coupon_data)
            return data
        except Exception as e:
            logger.error(f"Failed to create coupon {code}: {e}")
            return None

    async def delete_coupon(self, coupon_id: int, force: bool = True) -> bool:
        """Delete a coupon.

        Args:
            coupon_id: Coupon ID
            force: Permanently delete (vs trash)

        Returns:
            True if successful
        """
        try:
            await self._request(
                "DELETE",
                f"coupons/{coupon_id}",
                params={"force": force},
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete coupon {coupon_id}: {e}")
            return False

    # =========================================================================
    # Webhooks
    # =========================================================================

    async def get_webhooks(self) -> List[Dict[str, Any]]:
        """Get all registered webhooks.

        Returns:
            List of webhook data
        """
        try:
            data = await self._request("GET", "webhooks")
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Failed to get webhooks: {e}")
            return []

    async def create_webhook(
        self,
        name: str,
        topic: str,
        delivery_url: str,
        secret: str = "",
        status: str = "active",
    ) -> Optional[Dict[str, Any]]:
        """Register a new webhook.

        Args:
            name: Webhook name
            topic: Event topic (e.g., order.created, product.updated)
            delivery_url: URL to receive webhook payloads
            secret: Secret for payload signing
            status: Webhook status (active, paused, disabled)

        Returns:
            Created webhook data or None on failure
        """
        try:
            webhook_data = {
                "name": name,
                "topic": topic,
                "delivery_url": delivery_url,
                "secret": secret,
                "status": status,
            }
            data = await self._request("POST", "webhooks", json_data=webhook_data)
            return data
        except Exception as e:
            logger.error(f"Failed to create webhook {name}: {e}")
            return None

    async def delete_webhook(self, webhook_id: int, force: bool = True) -> bool:
        """Delete a webhook.

        Args:
            webhook_id: Webhook ID
            force: Permanently delete

        Returns:
            True if successful
        """
        try:
            await self._request(
                "DELETE",
                f"webhooks/{webhook_id}",
                params={"force": force},
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete webhook {webhook_id}: {e}")
            return False

    def verify_webhook_signature(  # type: ignore[override]
        self,
        payload: bytes,
        signature: str,
        secret: str,
    ) -> bool:
        """Verify webhook payload signature.

        Args:
            payload: Raw webhook payload bytes
            signature: X-WC-Webhook-Signature header value
            secret: Webhook secret

        Returns:
            True if signature is valid
        """
        import base64
        import hashlib
        import hmac

        expected = base64.b64encode(
            hmac.new(secret.encode(), payload, hashlib.sha256).digest()
        ).decode()
        return hmac.compare_digest(expected, signature)

    # =========================================================================
    # Shipping
    # =========================================================================

    async def get_shipping_zones(self) -> List[Dict[str, Any]]:
        """Get all shipping zones.

        Returns:
            List of shipping zone data
        """
        try:
            data = await self._request("GET", "shipping/zones")
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Failed to get shipping zones: {e}")
            return []

    async def get_shipping_methods(self, zone_id: int) -> List[Dict[str, Any]]:
        """Get shipping methods for a zone.

        Args:
            zone_id: Shipping zone ID

        Returns:
            List of shipping method data
        """
        try:
            data = await self._request("GET", f"shipping/zones/{zone_id}/methods")
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Failed to get shipping methods for zone {zone_id}: {e}")
            return []

    # =========================================================================
    # Tax
    # =========================================================================

    async def get_tax_classes(self) -> List[Dict[str, Any]]:
        """Get all tax classes.

        Returns:
            List of tax class data
        """
        try:
            data = await self._request("GET", "taxes/classes")
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Failed to get tax classes: {e}")
            return []

    async def get_tax_rates(self, per_page: int = 100) -> AsyncIterator[Dict[str, Any]]:
        """Get tax rates.

        Args:
            per_page: Page size

        Yields:
            Tax rate data dictionaries
        """
        params: Dict[str, Any] = {"per_page": per_page, "page": 1}

        while True:
            data = await self._request("GET", "taxes", params=params)

            if not data:
                break

            for rate in data:
                yield rate

            if len(data) < per_page:
                break

            params["page"] += 1

    # =========================================================================
    # Reports
    # =========================================================================

    async def get_sales_report(
        self,
        period: str = "month",
        date_min: Optional[str] = None,
        date_max: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get sales report.

        Args:
            period: Report period (week, month, last_month, year)
            date_min: Start date (YYYY-MM-DD)
            date_max: End date (YYYY-MM-DD)

        Returns:
            Sales report data
        """
        params: Dict[str, Any] = {"period": period}
        if date_min:
            params["date_min"] = date_min
        if date_max:
            params["date_max"] = date_max

        try:
            data = await self._request("GET", "reports/sales", params=params)
            return data[0] if isinstance(data, list) and data else {}
        except Exception as e:
            logger.error(f"Failed to get sales report: {e}")
            return {}

    async def get_top_sellers_report(
        self,
        period: str = "month",
    ) -> List[Dict[str, Any]]:
        """Get top selling products report.

        Args:
            period: Report period (week, month, last_month, year)

        Returns:
            List of top selling product data
        """
        try:
            data = await self._request(
                "GET",
                "reports/top_sellers",
                params={"period": period},
            )
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Failed to get top sellers report: {e}")
            return []

    # =========================================================================
    # BaseConnector abstract methods
    # =========================================================================

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> List[Evidence]:
        """Search WooCommerce for relevant data.

        Searches across orders, products, and customers.

        Args:
            query: Search query string
            limit: Maximum results to return
            **kwargs: Additional search options (entity_type: str)

        Returns:
            List of Evidence objects
        """
        results: List[Evidence] = []
        entity_type = kwargs.get("entity_type", "all")

        try:
            # Search products
            if entity_type in ("all", "product"):
                params = {"search": query, "per_page": min(limit, 100)}
                products = await self._request("GET", "products", params=params)
                for p in products[:limit]:
                    results.append(
                        Evidence(
                            id=f"woo-product-{p['id']}",
                            source_type=self.source_type,
                            source_id=str(p["id"]),
                            content=f"{p['name']}: {p.get('short_description', '')}",
                            title=p["name"],
                            url=p.get("permalink"),
                            metadata={"type": "product", "sku": p.get("sku")},
                        )
                    )

            # Search customers
            if entity_type in ("all", "customer"):
                params = {"search": query, "per_page": min(limit, 100)}
                customers = await self._request("GET", "customers", params=params)
                for c in customers[:limit]:
                    results.append(
                        Evidence(
                            id=f"woo-customer-{c['id']}",
                            source_type=self.source_type,
                            source_id=str(c["id"]),
                            content=f"{c.get('first_name', '')} {c.get('last_name', '')} - {c.get('email', '')}",
                            title=f"Customer: {c.get('email', 'unknown')}",
                            metadata={"type": "customer"},
                        )
                    )

        except Exception as e:
            logger.error(f"Search failed: {e}")

        return results[:limit]

    async def fetch(self, evidence_id: str) -> Optional[Evidence]:
        """Fetch a specific piece of evidence by ID.

        Args:
            evidence_id: Evidence ID (format: woo-{type}-{id})

        Returns:
            Evidence object or None if not found
        """
        try:
            parts = evidence_id.split("-")
            if len(parts) < 3 or parts[0] != "woo":
                return None

            entity_type = parts[1]
            entity_id = parts[2]

            if entity_type == "order":
                order = await self.get_order(int(entity_id))
                if order:
                    return Evidence(
                        id=evidence_id,
                        source_type=self.source_type,
                        source_id=str(order.id),
                        content=f"Order #{order.number} - {order.status.value} - ${order.total}",
                        title=f"Order #{order.number}",
                        metadata={"type": "order", "data": order.to_dict()},
                    )

            elif entity_type == "product":
                product = await self.get_product(int(entity_id))
                if product:
                    return Evidence(
                        id=evidence_id,
                        source_type=self.source_type,
                        source_id=str(product.id),
                        content=f"{product.name}: {product.description or ''}",
                        title=product.name,
                        metadata={"type": "product", "data": product.to_dict()},
                    )

            elif entity_type == "customer":
                data = await self._request("GET", f"customers/{entity_id}")
                customer = self._parse_customer(data)
                return Evidence(
                    id=evidence_id,
                    source_type=self.source_type,
                    source_id=str(customer.id),
                    content=f"{customer.first_name} {customer.last_name} - {customer.email}",
                    title=f"Customer: {customer.email}",
                    metadata={"type": "customer", "data": customer.to_dict()},
                )

        except Exception as e:
            logger.error(f"Failed to fetch {evidence_id}: {e}")

        return None

    async def sync_items(
        self,
        state: SyncState,
        batch_size: int = 100,
    ) -> AsyncIterator[SyncItem]:
        """Sync items from WooCommerce for Knowledge Mound.

        Args:
            state: Sync state with cursor/timestamp
            batch_size: Number of items per batch

        Yields:
            SyncItem objects for orders, products, customers
        """
        since = state.last_sync_at

        # Sync orders
        async for order in self.sync_orders(since=since, per_page=batch_size):
            yield SyncItem(
                id=f"woo-order-{order.id}",
                content=f"Order #{order.number} - {order.status.value} - ${order.total}",
                metadata=order.to_dict(),
                updated_at=order.date_modified,  # type: ignore[call-arg]
                source_type="woocommerce",
                source_id=str(order.id),
            )

        # Sync products
        async for product in self.sync_products(since=since, per_page=batch_size):
            yield SyncItem(
                id=f"woo-product-{product.id}",
                content=f"{product.name}: {product.short_description or ''}",
                metadata=product.to_dict(),
                updated_at=product.date_modified,  # type: ignore[call-arg]
                source_type="woocommerce",
                source_id=str(product.id),
            )

        # Sync customers
        async for customer in self.sync_customers(since=since, per_page=batch_size):
            yield SyncItem(
                id=f"woo-customer-{customer.id}",
                content=f"{customer.first_name} {customer.last_name} - {customer.email}",
                metadata=customer.to_dict(),
                updated_at=customer.date_modified,  # type: ignore[call-arg]
                source_type="woocommerce",
                source_id=str(customer.id),
            )

    # =========================================================================
    # EnterpriseConnector implementation
    # =========================================================================

    async def incremental_sync(
        self,
        state: Optional[SyncState] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Perform incremental sync of WooCommerce data.

        Args:
            state: Previous sync state

        Yields:
            Data items as dictionaries
        """
        since = state.last_sync_at if state else None

        # Sync orders
        async for order in self.sync_orders(since=since):
            yield {"type": "order", "data": order.to_dict()}

        # Sync products
        async for product in self.sync_products(since=since):
            yield {"type": "product", "data": product.to_dict()}

        # Sync customers
        async for customer in self.sync_customers(since=since):
            yield {"type": "customer", "data": customer.to_dict()}

    async def full_sync(self) -> SyncResult:
        """Perform full sync of WooCommerce data."""
        start_time = datetime.now(timezone.utc)
        items_synced = 0
        errors: List[str] = []

        try:
            async for _ in self.incremental_sync():
                items_synced += 1
        except Exception as e:
            errors.append(str(e))

        duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return SyncResult(
            connector_id=self.name,
            success=len(errors) == 0,
            items_synced=items_synced,
            items_updated=0,
            items_skipped=0,
            items_failed=len(errors),
            duration_ms=duration,
            errors=errors,
        )


# =========================================================================
# Mock data for testing
# =========================================================================


def get_mock_woo_orders() -> List[WooOrder]:
    """Get mock WooCommerce orders for testing."""
    now = datetime.now(timezone.utc)
    return [
        WooOrder(
            id=1001,
            number="1001",
            order_key="wc_order_mock1001",
            status=WooOrderStatus.PROCESSING,
            currency="USD",
            date_created=now,
            date_modified=now,
            total=Decimal("59.99"),
            subtotal=Decimal("49.99"),
            total_tax=Decimal("5.00"),
            shipping_total=Decimal("5.00"),
            discount_total=Decimal("0.00"),
            payment_method="stripe",
            payment_method_title="Credit Card",
            customer_id=1,
            billing=WooAddress(
                first_name="John",
                last_name="Doe",
                email="john@example.com",
            ),
            shipping=WooAddress(
                first_name="John",
                last_name="Doe",
            ),
            line_items=[
                WooLineItem(
                    id=1,
                    product_id=101,
                    variation_id=0,
                    name="Sample Product",
                    quantity=1,
                    subtotal=Decimal("49.99"),
                    total=Decimal("49.99"),
                    sku="WOO-001",
                )
            ],
        ),
    ]


def get_mock_woo_products() -> List[WooProduct]:
    """Get mock WooCommerce products for testing."""
    now = datetime.now(timezone.utc)
    return [
        WooProduct(
            id=101,
            name="Sample Product",
            slug="sample-product",
            type=WooProductType.SIMPLE,
            status=WooProductStatus.PUBLISH,
            sku="WOO-001",
            price=Decimal("49.99"),
            regular_price=Decimal("49.99"),
            sale_price=None,
            date_created=now,
            date_modified=now,
            description="A sample WooCommerce product",
            short_description="Sample product",
            stock_quantity=50,
            stock_status=WooStockStatus.IN_STOCK,
            manage_stock=True,
        ),
    ]


__all__ = [
    "WooCommerceConnector",
    "WooCommerceCredentials",
    "WooOrder",
    "WooOrderStatus",
    "WooProduct",
    "WooProductStatus",
    "WooProductType",
    "WooProductVariation",
    "WooCustomer",
    "WooAddress",
    "WooLineItem",
    "WooStockStatus",
    "get_mock_woo_orders",
    "get_mock_woo_products",
]
