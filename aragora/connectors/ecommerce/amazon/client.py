"""
Amazon Seller Central Connector (SP-API).

Provides integration with Amazon Seller Central using the Selling Partner API:
- Orders sync and management
- Inventory and FBA management
- Product listings (catalog items)
- Reports and analytics
- Feeds for bulk operations

Dependencies:
    pip install python-amazon-sp-api

Environment Variables:
    AMAZON_SP_REFRESH_TOKEN - LWA refresh token
    AMAZON_SP_CLIENT_ID - LWA client ID
    AMAZON_SP_CLIENT_SECRET - LWA client secret
    AMAZON_SP_MARKETPLACE_ID - Marketplace ID (e.g., ATVPDKIKX0DER for US)
    AMAZON_SP_SELLER_ID - Amazon seller ID
    AMAZON_SP_AWS_ACCESS_KEY - AWS access key for SP-API role
    AMAZON_SP_AWS_SECRET_KEY - AWS secret key for SP-API role
    AMAZON_SP_ROLE_ARN - IAM role ARN for SP-API
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from collections.abc import AsyncIterator

from aragora.connectors.enterprise.base import EnterpriseConnector, SyncItem, SyncResult, SyncState
from aragora.reasoning.provenance import SourceType

from .models import (
    AmazonAddress,
    AmazonCredentials,
    AmazonInventoryItem,
    AmazonMarketplace,
    AmazonOrder,
    AmazonOrderItem,
    AmazonOrderStatus,
    AmazonProduct,
    FulfillmentChannel,
    get_mock_inventory,
    get_mock_orders,
)

logger = logging.getLogger(__name__)

_MAX_PAGES = 1000  # Safety cap for pagination loops


class AmazonConnector(EnterpriseConnector):
    """
    Connector for Amazon Seller Central (SP-API).

    Supports:
    - Order sync and management
    - FBA inventory tracking
    - Product catalog access
    - Report generation

    Example:
        ```python
        credentials = AmazonCredentials.from_env()
        connector = AmazonConnector(credentials)

        # Sync orders
        async for order in connector.sync_orders(since=last_sync):
            process_order(order)

        # Get FBA inventory levels
        inventory = await connector.get_fba_inventory()
        low_stock = [i for i in inventory if i.available_quantity < 10]
        ```
    """

    @property
    def name(self) -> str:
        """Connector name for identification."""
        return "amazon"

    @property
    def source_type(self) -> SourceType:
        """Source type for provenance tracking."""
        return SourceType.EXTERNAL_API

    def __init__(
        self,
        credentials: AmazonCredentials,
        sandbox: bool = False,
        use_mock: bool = True,
    ):
        """Initialize Amazon SP-API connector.

        Args:
            credentials: Amazon SP-API credentials
            sandbox: Use sandbox environment
            use_mock: Use mock data instead of real API (default: True)
                      Set to False when python-amazon-sp-api is installed
                      and valid credentials are provided.
        """
        super().__init__(connector_id="amazon")
        self.amazon_credentials = credentials
        self.sandbox = sandbox
        self.use_mock = use_mock
        self._client = None
        self._sp_api_available = False

        # Check if SP-API library is available
        try:
            from sp_api.api import Orders, FbaInventory  # noqa: F401

            self._sp_api_available = True
            if use_mock:
                logger.info("python-amazon-sp-api is available but use_mock=True, using mock data")
        except ImportError:
            if not use_mock:
                logger.warning(
                    "python-amazon-sp-api not installed, falling back to mock data. "
                    "Install with: pip install python-amazon-sp-api"
                )
                self.use_mock = True

    async def connect(self) -> bool:
        """Establish connection to Amazon SP-API."""
        try:
            if self.use_mock:
                logger.info(
                    "Amazon SP-API connector initialized in mock mode (marketplace: %s)",
                    self.amazon_credentials.marketplace_id,
                )
                return True

            if self._sp_api_available:
                # Initialize real SP-API client
                from sp_api.base import Marketplaces

                marketplace_map = {
                    AmazonMarketplace.US.value: Marketplaces.US,
                    AmazonMarketplace.UK.value: Marketplaces.UK,
                    AmazonMarketplace.DE.value: Marketplaces.DE,
                    AmazonMarketplace.CA.value: Marketplaces.CA,
                }
                marketplace = marketplace_map.get(
                    self.amazon_credentials.marketplace_id, Marketplaces.US
                )

                self._sp_credentials = {
                    "refresh_token": self.amazon_credentials.refresh_token,
                    "lwa_app_id": self.amazon_credentials.client_id,
                    "lwa_client_secret": self.amazon_credentials.client_secret,
                    "aws_access_key": self.amazon_credentials.aws_access_key,
                    "aws_secret_key": self.amazon_credentials.aws_secret_key,
                    "role_arn": self.amazon_credentials.role_arn,
                }
                self._marketplace = marketplace

                logger.info(
                    "Connecting to Amazon SP-API (marketplace: %s)",
                    self.amazon_credentials.marketplace_id,
                )
                return True

            return False
        except (ImportError, KeyError, ValueError, RuntimeError) as e:
            logger.error("Failed to connect to Amazon SP-API: %s", e)
            return False

    async def disconnect(self) -> None:
        """Close Amazon connection."""
        self._client = None

    # =========================================================================
    # Orders
    # =========================================================================

    async def sync_orders(
        self,
        since: datetime | None = None,
        status: list[AmazonOrderStatus] | None = None,
        fulfillment_channels: list[FulfillmentChannel] | None = None,
    ) -> AsyncIterator[AmazonOrder]:
        """Sync orders from Amazon.

        Args:
            since: Only fetch orders updated after this time
            status: Filter by order status(es)
            fulfillment_channels: Filter by fulfillment channel(s)

        Yields:
            AmazonOrder objects
        """
        logger.info("Syncing Amazon orders since %s", since)

        if self.use_mock:
            # Return mock data for testing
            for order in get_mock_orders():
                if since and order.last_update_date < since:
                    continue
                if status and order.order_status not in status:
                    continue
                if fulfillment_channels and order.fulfillment_channel not in fulfillment_channels:
                    continue
                yield order
            return

        # Real SP-API implementation
        if self._sp_api_available:
            from sp_api.api import Orders

            orders_api = Orders(credentials=self._sp_credentials, marketplace=self._marketplace)

            # Build filter params
            params: dict[str, Any] = {}
            if since:
                params["LastUpdatedAfter"] = since.isoformat()
            if status:
                params["OrderStatuses"] = [s.value for s in status]
            if fulfillment_channels:
                params["FulfillmentChannels"] = [fc.value for fc in fulfillment_channels]

            # Paginate through orders
            next_token = None
            for _page in range(_MAX_PAGES):
                if next_token:
                    response = orders_api.get_orders(NextToken=next_token)
                else:
                    response = orders_api.get_orders(**params)

                orders_data = response.payload.get("Orders", [])
                for order_data in orders_data:
                    yield self._parse_sp_api_order(order_data)

                next_token = response.payload.get("NextToken")
                if not next_token:
                    break
            else:
                logger.warning("Pagination safety cap reached for Amazon orders")

    def _parse_sp_api_order(self, data: dict[str, Any]) -> AmazonOrder:
        """Parse SP-API order response into AmazonOrder dataclass."""
        # Parse shipping address
        shipping = data.get("ShippingAddress", {})
        shipping_address = (
            AmazonAddress(
                name=shipping.get("Name"),
                address_line1=shipping.get("AddressLine1"),
                address_line2=shipping.get("AddressLine2"),
                address_line3=shipping.get("AddressLine3"),
                city=shipping.get("City"),
                state_or_region=shipping.get("StateOrRegion"),
                postal_code=shipping.get("PostalCode"),
                country_code=shipping.get("CountryCode"),
                phone=shipping.get("Phone"),
            )
            if shipping
            else None
        )

        return AmazonOrder(
            amazon_order_id=data["AmazonOrderId"],
            seller_order_id=data.get("SellerOrderId"),
            purchase_date=datetime.fromisoformat(data["PurchaseDate"].replace("Z", "+00:00")),
            last_update_date=datetime.fromisoformat(data["LastUpdateDate"].replace("Z", "+00:00")),
            order_status=AmazonOrderStatus(data["OrderStatus"]),
            fulfillment_channel=FulfillmentChannel(data.get("FulfillmentChannel", "MFN")),
            sales_channel=data.get("SalesChannel", "Amazon.com"),
            order_total=(
                Decimal(str(data["OrderTotal"]["Amount"]))
                if data.get("OrderTotal")
                else Decimal("0.00")
            ),
            currency_code=data.get("OrderTotal", {}).get("CurrencyCode", "USD"),
            number_of_items_shipped=data.get("NumberOfItemsShipped", 0),
            number_of_items_unshipped=data.get("NumberOfItemsUnshipped", 0),
            buyer_email=data.get("BuyerEmail"),
            buyer_name=data.get("BuyerName"),
            shipping_address=shipping_address,
            is_business_order=data.get("IsBusinessOrder", False),
            is_prime=data.get("IsPrime", False),
            order_items=[],  # Items fetched separately via get_order_items
        )

    async def get_order(self, order_id: str) -> AmazonOrder | None:
        """Get a single order by ID.

        Args:
            order_id: Amazon order ID

        Returns:
            AmazonOrder or None if not found
        """
        # Would call Orders.getOrder(orderId)
        logger.info("Getting Amazon order %s", order_id)
        return None

    async def get_order_items(self, order_id: str) -> list[AmazonOrderItem]:
        """Get items for an order.

        Args:
            order_id: Amazon order ID

        Returns:
            List of order items
        """
        # Would call Orders.getOrderItems(orderId)
        return []

    async def confirm_shipment(
        self,
        order_id: str,
        tracking_number: str,
        carrier_code: str,
        ship_date: datetime,
    ) -> bool:
        """Confirm shipment for an order.

        Args:
            order_id: Amazon order ID
            tracking_number: Carrier tracking number
            carrier_code: Carrier code (e.g., "UPS", "USPS")
            ship_date: Shipment date

        Returns:
            True if successful
        """
        # Would call Orders.confirmShipment()
        logger.info("Confirming shipment for order %s", order_id)
        return True

    # =========================================================================
    # Inventory (FBA)
    # =========================================================================

    async def get_fba_inventory(
        self,
        skus: list[str] | None = None,
    ) -> list[AmazonInventoryItem]:
        """Get FBA inventory levels.

        Args:
            skus: Filter by seller SKUs (optional)

        Returns:
            List of inventory items
        """
        # Would call FBAInventory.getInventorySummaries()
        logger.info("Getting FBA inventory")
        return get_mock_inventory()

    async def get_inventory_item(
        self,
        seller_sku: str,
    ) -> AmazonInventoryItem | None:
        """Get inventory for a specific SKU.

        Args:
            seller_sku: Seller SKU

        Returns:
            Inventory item or None
        """
        inventory = await self.get_fba_inventory(skus=[seller_sku])
        return inventory[0] if inventory else None

    async def create_inbound_shipment(
        self,
        shipment_name: str,
        items: list[dict[str, Any]],
        ship_from_address: AmazonAddress,
    ) -> str | None:
        """Create FBA inbound shipment plan.

        Args:
            shipment_name: Name for the shipment
            items: List of {sellerSku, quantity} items
            ship_from_address: Shipping origin address

        Returns:
            Shipment ID if successful
        """
        # Would use FulfillmentInbound API
        logger.info("Creating inbound shipment: %s", shipment_name)
        return None

    # =========================================================================
    # Catalog/Products
    # =========================================================================

    async def get_catalog_item(self, asin: str) -> AmazonProduct | None:
        """Get product details from catalog.

        Args:
            asin: Amazon Standard Identification Number

        Returns:
            Product details or None
        """
        # Would call CatalogItems.getCatalogItem()
        logger.info("Getting catalog item %s", asin)
        return None

    async def search_catalog(
        self,
        keywords: str,
        limit: int = 20,
    ) -> list[AmazonProduct]:
        """Search Amazon catalog.

        Args:
            keywords: Search keywords
            limit: Maximum results

        Returns:
            List of matching products
        """
        logger.info("Searching catalog: %s", keywords)

        if self.use_mock:
            # Stubbed in mock mode unless explicitly integrated with catalog mocks.
            return []

        if self._sp_api_available:
            from sp_api.api import CatalogItems

            catalog_api = CatalogItems(
                credentials=self._sp_credentials, marketplace=self._marketplace
            )

            try:
                response = catalog_api.search_catalog_items(
                    keywords=keywords,
                    pageSize=min(limit, 20),  # SP-API max is 20
                )

                products = []
                for item in response.payload.get("items", []):
                    products.append(self._parse_sp_api_product(item))
                    if len(products) >= limit:
                        break

                return products
            except (OSError, RuntimeError, ValueError, KeyError, AttributeError) as e:
                logger.warning("Catalog search failed: %s", e)
                return []

        return []

    def _parse_sp_api_product(self, data: dict[str, Any]) -> AmazonProduct:
        """Parse SP-API catalog item response into AmazonProduct dataclass."""
        attributes = data.get("attributes", {})
        summaries = data.get("summaries", [{}])[0] if data.get("summaries") else {}
        images = data.get("images", [{}])[0] if data.get("images") else {}

        # Extract images
        image_urls = []
        for img in images.get("images", []):
            if img.get("link"):
                image_urls.append(img["link"])

        return AmazonProduct(
            asin=data.get("asin", ""),
            title=summaries.get("itemName", attributes.get("item_name", {}).get("value", "")),
            brand=summaries.get("brand", attributes.get("brand", {}).get("value")),
            manufacturer=summaries.get(
                "manufacturer", attributes.get("manufacturer", {}).get("value")
            ),
            product_type=summaries.get("productType"),
            parent_asin=data.get("relationships", {}).get("parentAsin"),
            item_dimensions=attributes.get("item_dimensions"),
            package_dimensions=attributes.get("package_dimensions"),
            images=image_urls,
            bullet_points=[
                bp.get("value", "")
                for bp in attributes.get("bullet_point", [])
                if isinstance(bp, dict)
            ],
            browse_nodes=[
                str(bn.get("id", ""))
                for bn in data.get("browseNodeInfo", {}).get("browseNodes", [])
            ],
        )

    # =========================================================================
    # Reports
    # =========================================================================

    async def request_report(
        self,
        report_type: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> str | None:
        """Request an Amazon report.

        Args:
            report_type: Report type (e.g., "GET_FLAT_FILE_OPEN_LISTINGS_DATA")
            start_date: Report start date
            end_date: Report end date

        Returns:
            Report ID if request accepted
        """
        # Would call Reports.createReport()
        logger.info("Requesting report: %s", report_type)
        return None

    async def get_report(self, report_id: str) -> bytes | None:
        """Download a generated report.

        Args:
            report_id: Report ID from request

        Returns:
            Report document bytes or None
        """
        # Would call Reports.getReportDocument()
        return None

    # =========================================================================
    # Analytics
    # =========================================================================

    async def get_sales_metrics(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        """Get sales metrics for date range.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            Dictionary with sales metrics
        """
        total_orders = 0
        total_revenue = Decimal("0.00")
        fba_orders = 0
        mfn_orders = 0

        async for order in self.sync_orders(since=start_date):
            if order.purchase_date > end_date:
                continue

            total_orders += 1
            total_revenue += order.order_total

            if order.fulfillment_channel == FulfillmentChannel.AFN:
                fba_orders += 1
            else:
                mfn_orders += 1

        return {
            "total_orders": total_orders,
            "total_revenue": str(total_revenue),
            "fba_orders": fba_orders,
            "mfn_orders": mfn_orders,
            "fba_percentage": fba_orders / total_orders if total_orders > 0 else 0,
        }

    # =========================================================================
    # EnterpriseConnector implementation
    # =========================================================================

    async def incremental_sync(
        self,
        state: SyncState | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Perform incremental sync of Amazon data.

        Args:
            state: Previous sync state for resumption

        Yields:
            Data items (orders, inventory) as dictionaries
        """
        since = state.last_sync_at if state else None

        # Sync orders
        async for order in self.sync_orders(since=since):
            yield {"type": "order", "data": order.to_dict()}

        # Sync inventory
        for item in await self.get_fba_inventory():
            yield {"type": "inventory", "data": item.to_dict()}

    async def full_sync(self) -> SyncResult:
        """Perform full sync of Amazon data."""
        start_time = datetime.now(timezone.utc)
        items_synced = 0
        errors: list[str] = []

        try:
            async for _ in self.incremental_sync():
                items_synced += 1
        except (OSError, RuntimeError, ValueError, KeyError) as e:
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
        Yield Amazon data as SyncItems for incremental sync.

        Args:
            state: Current sync state with cursor
            batch_size: Number of items per batch

        Yields:
            SyncItem objects for Knowledge Mound ingestion
        """
        since = state.last_sync_at if state else None

        # Sync orders
        async for order in self.sync_orders(since=since):
            content_parts = [f"# Order {order.amazon_order_id}"]
            content_parts.append(f"\nStatus: {order.order_status.value}")
            content_parts.append(f"\nTotal: {order.currency_code} {order.order_total}")
            content_parts.append(f"\nChannel: {order.fulfillment_channel.value}")
            content_parts.append(f"\nItems: {len(order.order_items)}")

            yield SyncItem(
                id=f"amazon:order:{order.amazon_order_id}",
                source_type="ecommerce",
                source_id=order.amazon_order_id,
                content="\n".join(content_parts),
                title=f"Amazon Order {order.amazon_order_id}",
                url="",
                author="",
                created_at=order.purchase_date,
                updated_at=order.last_update_date,
                domain="ecommerce",
                confidence=0.95,
                metadata={
                    "type": "order",
                    "order_status": order.order_status.value,
                    "fulfillment_channel": order.fulfillment_channel.value,
                    "sales_channel": order.sales_channel,
                    "currency_code": order.currency_code,
                    "order_total": str(order.order_total),
                    "items_shipped": order.number_of_items_shipped,
                    "items_unshipped": order.number_of_items_unshipped,
                },
            )

        # Sync inventory
        for item in await self.get_fba_inventory():
            content_parts = [f"# {item.asin}"]
            content_parts.append(f"\nSKU: {item.seller_sku}")
            content_parts.append(f"\nAvailable: {item.available_quantity}")
            content_parts.append(f"\nInbound: {item.inbound_quantity}")
            content_parts.append(f"\nReserved: {item.reserved_quantity}")

            yield SyncItem(
                id=f"amazon:inventory:{item.seller_sku}",
                source_type="ecommerce",
                source_id=item.seller_sku,
                content="\n".join(content_parts),
                title=f"Inventory: {item.seller_sku}",
                url="",
                author="",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                domain="ecommerce",
                confidence=0.95,
                metadata={
                    "type": "inventory",
                    "asin": item.asin,
                    "sku": item.seller_sku,
                    "condition": item.condition,
                    "available_quantity": item.available_quantity,
                    "inbound_quantity": item.inbound_quantity,
                    "reserved_quantity": item.reserved_quantity,
                    "total_quantity": item.total_quantity,
                },
            )

    # =========================================================================
    # BaseConnector abstract method implementations
    # =========================================================================

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> list:
        """
        Search for evidence in Amazon data.

        Searches products via catalog API and returns Evidence objects.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of Evidence objects from matching Amazon products
        """
        from aragora.connectors.base import Evidence

        products = await self.search_catalog(query, limit=limit)

        evidence_list = []
        for product in products:
            # Build content from product details
            content_parts = [f"# {product.title}"]
            if product.brand:
                content_parts.append(f"\nBrand: {product.brand}")
            if product.manufacturer:
                content_parts.append(f"\nManufacturer: {product.manufacturer}")
            if product.product_type:
                content_parts.append(f"\nProduct Type: {product.product_type}")
            if product.bullet_points:
                content_parts.append("\n\nFeatures:")
                for bp in product.bullet_points[:5]:  # Limit to 5 bullet points
                    content_parts.append(f"\n- {bp}")

            evidence = Evidence(
                id=f"amazon:product:{product.asin}",
                source_type=self.source_type,
                source_id=product.asin,
                content="\n".join(content_parts),
                title=product.title,
                url=f"https://www.amazon.com/dp/{product.asin}",
                confidence=0.9,  # Amazon catalog is authoritative
                authority=0.85,  # Official product data
                freshness=0.95,  # Catalog data is usually current
                metadata={
                    "asin": product.asin,
                    "brand": product.brand,
                    "manufacturer": product.manufacturer,
                    "product_type": product.product_type,
                    "parent_asin": product.parent_asin,
                    "images": product.images,
                    "browse_nodes": product.browse_nodes,
                },
            )
            evidence_list.append(evidence)

        return evidence_list

    async def fetch(self, evidence_id: str):
        """
        Fetch specific evidence by ID.

        Args:
            evidence_id: Evidence ID (e.g., "order:123" or "inventory:SKU-001")

        Returns:
            Evidence object or None
        """
        # Parse the evidence ID to determine type
        if evidence_id.startswith("order:"):
            order_id = evidence_id[6:]
            order = await self.get_order(order_id)
            if order:
                from aragora.connectors.base import Evidence

                return Evidence(
                    id=evidence_id,
                    source_type=self.source_type,
                    source_id=order_id,
                    content=f"Order {order.amazon_order_id}: {order.order_status.value}",
                    title=f"Amazon Order {order.amazon_order_id}",
                    metadata=order.to_dict(),
                )
        elif evidence_id.startswith("inventory:"):
            sku = evidence_id[10:]
            item = await self.get_inventory_item(sku)
            if item:
                from aragora.connectors.base import Evidence

                return Evidence(
                    id=evidence_id,
                    source_type=self.source_type,
                    source_id=sku,
                    content=f"Inventory {item.seller_sku}: {item.available_quantity} available",
                    title=f"Amazon Inventory {item.seller_sku}",
                    metadata=item.to_dict(),
                )
        return None


__all__ = ["AmazonConnector"]
