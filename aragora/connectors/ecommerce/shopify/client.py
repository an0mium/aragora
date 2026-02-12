"""
Shopify Connector.

Provides integration with Shopify stores for e-commerce operations:
- OAuth 2.0 authentication (embedded app flow)
- Orders sync (create, update, fulfill)
- Products and inventory management
- Customers and customer segments
- Analytics and reports

Dependencies:
    pip install shopify

Environment Variables:
    SHOPIFY_API_KEY - Shopify app API key
    SHOPIFY_API_SECRET - Shopify app secret
    SHOPIFY_API_VERSION - API version (e.g., '2024-01')
    SHOPIFY_SHOP_DOMAIN - Store domain (store.myshopify.com)
    SHOPIFY_ACCESS_TOKEN - Store access token (for private apps)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional
from collections.abc import AsyncIterator

from aragora.connectors.enterprise.base import EnterpriseConnector, SyncItem, SyncResult, SyncState
from aragora.connectors.exceptions import ConnectorAPIError

from .models import (
    InventoryPolicy,
    OrderStatus,
    PaymentStatus,
    ShopifyAddress,
    ShopifyCredentials,
    ShopifyCustomer,
    ShopifyEnvironment,
    ShopifyLineItem,
    ShopifyOrder,
    ShopifyProduct,
    ShopifyVariant,
)

logger = logging.getLogger(__name__)

_MAX_PAGES = 1000  # Safety cap for pagination loops


class ShopifyConnector(EnterpriseConnector):
    """
    Connector for Shopify e-commerce platform.

    Supports:
    - Order management (sync, create, update, fulfill)
    - Product and variant management
    - Customer data sync
    - Inventory level tracking
    - Webhook event handling

    Example:
        ```python
        credentials = ShopifyCredentials.from_env()
        connector = ShopifyConnector(credentials)

        # Sync orders
        async for order in connector.sync_orders(since=last_sync):
            process_order(order)

        # Get low stock products
        low_stock = await connector.get_low_stock_variants(threshold=5)
        ```
    """

    def __init__(
        self,
        credentials: ShopifyCredentials,
        environment: ShopifyEnvironment = ShopifyEnvironment.PRODUCTION,
    ):
        """Initialize Shopify connector.

        Args:
            credentials: Shopify OAuth credentials
            environment: Development or production environment
        """
        super().__init__(connector_id="shopify", tenant_id="default")
        self._shop_credentials = credentials
        self.environment = environment
        self._session: Any = None

    @property
    def base_url(self) -> str:
        """Get Shopify API base URL."""
        return f"https://{self._shop_credentials.shop_domain}/admin/api/{self._shop_credentials.api_version}"

    async def connect(self) -> bool:
        """Establish connection to Shopify API."""
        try:
            import aiohttp

            headers = {
                "X-Shopify-Access-Token": self._shop_credentials.access_token,
                "Content-Type": "application/json",
            }
            self._session = aiohttp.ClientSession(headers=headers)

            # Test connection by fetching shop info
            async with self._session.get(f"{self.base_url}/shop.json") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"Connected to Shopify store: {data['shop']['name']}")
                    return True
                else:
                    logger.error(f"Failed to connect to Shopify: {resp.status}")
                    return False

        except ImportError:
            logger.error("aiohttp package not installed. Run: pip install aiohttp")
            return False
        except OSError as e:
            logger.error(f"Failed to connect to Shopify: {e}")
            return False
        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Failed to connect to Shopify: {e}")
            return False

    async def disconnect(self) -> None:
        """Close Shopify connection."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        return_headers: bool = False,
    ) -> Any:
        """Make an API request to Shopify.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (e.g., /orders.json)
            params: Query parameters
            json_data: JSON body data
            return_headers: If True, returns (data, headers) tuple

        Returns:
            Response JSON data, or (data, headers) if return_headers=True
        """
        if not self._session:
            await self.connect()

        url = f"{self.base_url}{endpoint}"
        async with self._session.request(method, url, params=params, json=json_data) as resp:
            if resp.status >= 400:
                error_text = await resp.text()
                raise ConnectorAPIError(
                    f"Shopify API error: {error_text}",
                    connector_name="shopify",
                    status_code=resp.status,
                )
            data = await resp.json()
            if return_headers:
                return data, dict(resp.headers)
            return data

    def _parse_link_header(self, link_header: str | None) -> str | None:
        """Parse the Link header to extract next page_info.

        Shopify uses cursor-based pagination with Link headers:
        <url>; rel="next", <url>; rel="previous"

        Args:
            link_header: The Link header value

        Returns:
            The page_info parameter for the next page, or None if no next page
        """
        if not link_header:
            return None

        import re

        # Find the "next" link
        for part in link_header.split(","):
            if 'rel="next"' in part:
                # Extract URL from <url>
                match = re.search(r"<([^>]+)>", part)
                if match:
                    url = match.group(1)
                    # Extract page_info from URL
                    page_info_match = re.search(r"page_info=([^&]+)", url)
                    if page_info_match:
                        return page_info_match.group(1)
        return None

    # =========================================================================
    # Orders
    # =========================================================================

    async def sync_orders(
        self,
        since: datetime | None = None,
        status: str = "any",
        limit: int = 250,
    ) -> AsyncIterator[ShopifyOrder]:
        """Sync orders from Shopify.

        Args:
            since: Only fetch orders updated after this time
            status: Order status filter (any, open, closed, cancelled)
            limit: Page size

        Yields:
            ShopifyOrder objects
        """
        params: dict[str, Any] = {
            "status": status,
            "limit": limit,
        }
        if since:
            params["updated_at_min"] = since.isoformat()

        page_info = None
        for _page in range(_MAX_PAGES):
            if page_info:
                # When using page_info, only include it and limit (other params are encoded in cursor)
                params = {"page_info": page_info, "limit": limit}

            data, headers = await self._request(
                "GET", "/orders.json", params=params, return_headers=True
            )

            orders = data.get("orders", [])
            for order_data in orders:
                yield self._parse_order(order_data)

            # Check if we got fewer items than limit (last page)
            if len(orders) < limit:
                break

            # Parse Link header for next page cursor
            page_info = self._parse_link_header(headers.get("Link") or headers.get("link"))
            if not page_info:
                break
        else:
            logger.warning("Pagination safety cap reached for Shopify orders")

    def _parse_order(self, data: dict[str, Any]) -> ShopifyOrder:
        """Parse order data from API response."""
        line_items = [
            ShopifyLineItem(
                id=str(item["id"]),
                product_id=str(item.get("product_id")) if item.get("product_id") else None,
                variant_id=str(item.get("variant_id")) if item.get("variant_id") else None,
                title=item["title"],
                quantity=item["quantity"],
                price=Decimal(str(item["price"])),
                sku=item.get("sku"),
                vendor=item.get("vendor"),
                grams=item.get("grams", 0),
                taxable=item.get("taxable", True),
                fulfillment_status=item.get("fulfillment_status"),
                requires_shipping=item.get("requires_shipping", True),
            )
            for item in data.get("line_items", [])
        ]

        shipping = data.get("shipping_address")
        billing = data.get("billing_address")

        return ShopifyOrder(
            id=str(data["id"]),
            order_number=data["order_number"],
            name=data["name"],
            email=data.get("email"),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")),
            total_price=Decimal(str(data["total_price"])),
            subtotal_price=Decimal(str(data["subtotal_price"])),
            total_tax=Decimal(str(data["total_tax"])),
            total_discounts=Decimal(str(data["total_discounts"])),
            currency=data["currency"],
            financial_status=PaymentStatus(data.get("financial_status", "pending")),
            fulfillment_status=(
                OrderStatus(data["fulfillment_status"]) if data.get("fulfillment_status") else None
            ),
            line_items=line_items,
            shipping_address=self._parse_address(shipping) if shipping else None,
            billing_address=self._parse_address(billing) if billing else None,
            customer_id=str(data["customer"]["id"]) if data.get("customer") else None,
            note=data.get("note"),
            tags=data.get("tags", "").split(", ") if data.get("tags") else [],
            cancelled_at=(
                datetime.fromisoformat(data["cancelled_at"].replace("Z", "+00:00"))
                if data.get("cancelled_at")
                else None
            ),
            closed_at=(
                datetime.fromisoformat(data["closed_at"].replace("Z", "+00:00"))
                if data.get("closed_at")
                else None
            ),
        )

    def _parse_address(self, data: dict[str, Any]) -> ShopifyAddress:
        """Parse address data."""
        return ShopifyAddress(
            first_name=data.get("first_name"),
            last_name=data.get("last_name"),
            company=data.get("company"),
            address1=data.get("address1"),
            address2=data.get("address2"),
            city=data.get("city"),
            province=data.get("province"),
            province_code=data.get("province_code"),
            country=data.get("country"),
            country_code=data.get("country_code"),
            zip=data.get("zip"),
            phone=data.get("phone"),
        )

    async def get_order(self, order_id: str) -> ShopifyOrder | None:
        """Get a single order by ID.

        Args:
            order_id: Shopify order ID

        Returns:
            ShopifyOrder or None if not found
        """
        try:
            data = await self._request("GET", f"/orders/{order_id}.json")
            return self._parse_order(data["order"])
        except (ConnectorAPIError, OSError, RuntimeError, ValueError, KeyError) as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    async def fulfill_order(
        self,
        order_id: str,
        tracking_number: str | None = None,
        tracking_company: str | None = None,
        notify_customer: bool = True,
    ) -> bool:
        """Mark an order as fulfilled.

        Args:
            order_id: Order to fulfill
            tracking_number: Optional tracking number
            tracking_company: Optional carrier name
            notify_customer: Whether to send notification email

        Returns:
            True if successful
        """
        try:
            fulfillment_data: dict[str, Any] = {
                "fulfillment": {
                    "notify_customer": notify_customer,
                }
            }
            if tracking_number:
                fulfillment_data["fulfillment"]["tracking_number"] = tracking_number
            if tracking_company:
                fulfillment_data["fulfillment"]["tracking_company"] = tracking_company

            await self._request(
                "POST",
                f"/orders/{order_id}/fulfillments.json",
                json_data=fulfillment_data,
            )
            return True
        except (ConnectorAPIError, OSError, RuntimeError, ValueError) as e:
            logger.error(f"Failed to fulfill order {order_id}: {e}")
            return False

    # =========================================================================
    # Products
    # =========================================================================

    async def sync_products(
        self,
        since: datetime | None = None,
        status: str = "active",
        limit: int = 250,
    ) -> AsyncIterator[ShopifyProduct]:
        """Sync products from Shopify.

        Args:
            since: Only fetch products updated after this time
            status: Product status filter (active, archived, draft)
            limit: Page size

        Yields:
            ShopifyProduct objects
        """
        params: dict[str, Any] = {
            "status": status,
            "limit": limit,
        }
        if since:
            params["updated_at_min"] = since.isoformat()

        data = await self._request("GET", "/products.json", params=params)

        for product_data in data.get("products", []):
            yield self._parse_product(product_data)

    def _parse_product(self, data: dict[str, Any]) -> ShopifyProduct:
        """Parse product data from API response."""
        variants = [
            ShopifyVariant(
                id=str(v["id"]),
                product_id=str(data["id"]),
                title=v["title"],
                price=Decimal(str(v["price"])),
                sku=v.get("sku"),
                inventory_quantity=v.get("inventory_quantity", 0),
                inventory_policy=InventoryPolicy(v.get("inventory_policy", "deny")),
                compare_at_price=(
                    Decimal(str(v["compare_at_price"])) if v.get("compare_at_price") else None
                ),
                weight=float(v.get("weight", 0)),
                weight_unit=v.get("weight_unit", "kg"),
                barcode=v.get("barcode"),
                option1=v.get("option1"),
                option2=v.get("option2"),
                option3=v.get("option3"),
            )
            for v in data.get("variants", [])
        ]

        images = [img["src"] for img in data.get("images", [])]

        return ShopifyProduct(
            id=str(data["id"]),
            title=data["title"],
            handle=data["handle"],
            vendor=data.get("vendor"),
            product_type=data.get("product_type"),
            status=data.get("status", "active"),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")),
            published_at=(
                datetime.fromisoformat(data["published_at"].replace("Z", "+00:00"))
                if data.get("published_at")
                else None
            ),
            description=data.get("body_html"),
            tags=data.get("tags", "").split(", ") if data.get("tags") else [],
            variants=variants,
            images=images,
        )

    async def get_product(self, product_id: str) -> ShopifyProduct | None:
        """Get a single product by ID."""
        try:
            data = await self._request("GET", f"/products/{product_id}.json")
            return self._parse_product(data["product"])
        except (ConnectorAPIError, OSError, RuntimeError, ValueError, KeyError) as e:
            logger.error(f"Failed to get product {product_id}: {e}")
            return None

    async def update_variant_inventory(
        self,
        inventory_item_id: str,
        location_id: str,
        adjustment: int,
    ) -> bool:
        """Adjust inventory level for a variant.

        Args:
            inventory_item_id: Inventory item ID
            location_id: Location ID
            adjustment: Quantity adjustment (+/-)

        Returns:
            True if successful
        """
        try:
            await self._request(
                "POST",
                "/inventory_levels/adjust.json",
                json_data={
                    "inventory_item_id": inventory_item_id,
                    "location_id": location_id,
                    "available_adjustment": adjustment,
                },
            )
            return True
        except (ConnectorAPIError, OSError, RuntimeError, ValueError) as e:
            logger.error(f"Failed to adjust inventory: {e}")
            return False

    async def get_low_stock_variants(
        self,
        threshold: int = 5,
    ) -> list[ShopifyVariant]:
        """Get variants with low stock.

        Args:
            threshold: Stock level threshold

        Returns:
            List of variants with inventory below threshold
        """
        low_stock = []
        async for product in self.sync_products():
            for variant in product.variants:
                if variant.inventory_quantity <= threshold:
                    low_stock.append(variant)
        return low_stock

    # =========================================================================
    # Customers
    # =========================================================================

    async def sync_customers(
        self,
        since: datetime | None = None,
        limit: int = 250,
    ) -> AsyncIterator[ShopifyCustomer]:
        """Sync customers from Shopify.

        Args:
            since: Only fetch customers updated after this time
            limit: Page size

        Yields:
            ShopifyCustomer objects
        """
        params: dict[str, Any] = {"limit": limit}
        if since:
            params["updated_at_min"] = since.isoformat()

        data = await self._request("GET", "/customers.json", params=params)

        for customer_data in data.get("customers", []):
            yield self._parse_customer(customer_data)

    def _parse_customer(self, data: dict[str, Any]) -> ShopifyCustomer:
        """Parse customer data from API response."""
        return ShopifyCustomer(
            id=str(data["id"]),
            email=data.get("email"),
            first_name=data.get("first_name"),
            last_name=data.get("last_name"),
            phone=data.get("phone"),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")),
            orders_count=data.get("orders_count", 0),
            total_spent=Decimal(str(data.get("total_spent", "0.00"))),
            verified_email=data.get("verified_email", False),
            accepts_marketing=data.get("accepts_marketing", False),
            tax_exempt=data.get("tax_exempt", False),
            tags=data.get("tags", "").split(", ") if data.get("tags") else [],
            note=data.get("note"),
        )

    async def get_customer(self, customer_id: str) -> ShopifyCustomer | None:
        """Get a single customer by ID."""
        try:
            data = await self._request("GET", f"/customers/{customer_id}.json")
            return self._parse_customer(data["customer"])
        except (ConnectorAPIError, OSError, RuntimeError, ValueError, KeyError) as e:
            logger.error(f"Failed to get customer {customer_id}: {e}")
            return None

    # =========================================================================
    # Analytics
    # =========================================================================

    async def get_order_stats(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Get order statistics.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            Dictionary with order statistics
        """
        total_orders = 0
        total_revenue = Decimal("0.00")
        fulfilled_orders = 0
        cancelled_orders = 0

        async for order in self.sync_orders(since=start_date):
            if end_date and order.created_at > end_date:
                continue

            total_orders += 1
            total_revenue += order.total_price

            if order.fulfillment_status == OrderStatus.FULFILLED:
                fulfilled_orders += 1
            elif order.fulfillment_status == OrderStatus.CANCELLED:
                cancelled_orders += 1

        return {
            "total_orders": total_orders,
            "total_revenue": str(total_revenue),
            "fulfilled_orders": fulfilled_orders,
            "cancelled_orders": cancelled_orders,
            "fulfillment_rate": fulfilled_orders / total_orders if total_orders > 0 else 0,
            "average_order_value": (
                str(total_revenue / total_orders) if total_orders > 0 else "0.00"
            ),
        }

    # =========================================================================
    # EnterpriseConnector implementation
    # =========================================================================

    async def incremental_sync(
        self,
        state: SyncState | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Perform incremental sync of all Shopify data.

        Args:
            state: Previous sync state for resumption

        Yields:
            Data items (orders, products, customers) as dictionaries
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
        """Perform full sync of all Shopify data."""
        start_time = datetime.now(timezone.utc)
        items_synced = 0
        errors: list[str] = []

        try:
            async for _ in self.incremental_sync():
                items_synced += 1
        except (ConnectorAPIError, OSError, RuntimeError, ValueError, KeyError) as e:
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

    async def sync_items(
        self,
        state: SyncState,
        batch_size: int = 100,
    ) -> AsyncIterator[SyncItem]:
        """
        Yield Shopify data as SyncItems for incremental sync.

        Args:
            state: Current sync state with cursor
            batch_size: Number of items per batch

        Yields:
            SyncItem objects for Knowledge Mound ingestion
        """
        since = state.last_sync_at if state else None

        # Sync orders
        async for order in self.sync_orders(since=since, limit=batch_size):
            content_parts = [f"# Order {order.name}"]
            content_parts.append(f"\nTotal: {order.currency} {order.total_price}")
            content_parts.append(f"\nStatus: {order.financial_status.value}")
            if order.fulfillment_status:
                content_parts.append(f"\nFulfillment: {order.fulfillment_status.value}")
            content_parts.append(f"\nItems: {len(order.line_items)}")

            yield SyncItem(
                id=f"shopify:order:{order.id}",
                source_type="ecommerce",
                source_id=order.id,
                content="\n".join(content_parts),
                title=f"Order {order.name}",
                url="",
                author=order.email or "",
                created_at=order.created_at,
                updated_at=order.updated_at,
                domain="ecommerce",
                confidence=0.95,
                metadata={
                    "type": "order",
                    "order_number": order.order_number,
                    "currency": order.currency,
                    "total_price": str(order.total_price),
                    "financial_status": order.financial_status.value,
                    "fulfillment_status": (
                        order.fulfillment_status.value if order.fulfillment_status else None
                    ),
                    "line_item_count": len(order.line_items),
                    "customer_id": order.customer_id,
                },
            )

        # Sync products
        async for product in self.sync_products(since=since, limit=batch_size):
            content_parts = [f"# {product.title}"]
            if product.description:
                content_parts.append(f"\n{product.description}")
            content_parts.append(f"\nVendor: {product.vendor or 'N/A'}")
            content_parts.append(f"\nVariants: {len(product.variants)}")
            if product.tags:
                content_parts.append(f"\nTags: {', '.join(product.tags)}")

            yield SyncItem(
                id=f"shopify:product:{product.id}",
                source_type="ecommerce",
                source_id=product.id,
                content="\n".join(content_parts),
                title=product.title,
                url="",
                author=product.vendor or "",
                created_at=product.created_at,
                updated_at=product.updated_at,
                domain="ecommerce",
                confidence=0.95,
                metadata={
                    "type": "product",
                    "handle": product.handle,
                    "vendor": product.vendor,
                    "product_type": product.product_type,
                    "status": product.status,
                    "variant_count": len(product.variants),
                    "tags": product.tags,
                },
            )

        # Sync customers
        async for customer in self.sync_customers(since=since, limit=batch_size):
            content_parts = [f"# {customer.full_name}"]
            if customer.email:
                content_parts.append(f"\nEmail: {customer.email}")
            content_parts.append(f"\nOrders: {customer.orders_count}")
            content_parts.append(f"\nTotal Spent: {customer.total_spent}")

            yield SyncItem(
                id=f"shopify:customer:{customer.id}",
                source_type="ecommerce",
                source_id=customer.id,
                content="\n".join(content_parts),
                title=customer.full_name,
                url="",
                author="",
                created_at=customer.created_at,
                updated_at=customer.updated_at,
                domain="ecommerce",
                confidence=0.9,
                metadata={
                    "type": "customer",
                    "email": customer.email,
                    "orders_count": customer.orders_count,
                    "total_spent": str(customer.total_spent),
                    "verified_email": customer.verified_email,
                    "accepts_marketing": customer.accepts_marketing,
                    "tags": customer.tags,
                },
            )


__all__ = ["ShopifyConnector"]
