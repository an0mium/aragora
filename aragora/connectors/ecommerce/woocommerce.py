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

from aragora.connectors.enterprise.base import EnterpriseConnector, SyncResult, SyncState

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
                raise Exception(f"WooCommerce API error {resp.status}: {error_text}")
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
            date_paid=datetime.fromisoformat(data["date_paid_gmt"] + "+00:00")
            if data.get("date_paid_gmt")
            else None,
            date_completed=datetime.fromisoformat(data["date_completed_gmt"] + "+00:00")
            if data.get("date_completed_gmt")
            else None,
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
