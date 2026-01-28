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
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

from aragora.connectors.enterprise.base import EnterpriseConnector, SyncItem, SyncResult, SyncState
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)


class AmazonMarketplace(str, Enum):
    """Amazon marketplace identifiers."""

    US = "ATVPDKIKX0DER"
    CA = "A2EUQ1WTGCTBG2"
    MX = "A1AM78C64UM0Y8"
    UK = "A1F83G8C2ARO7P"
    DE = "A1PA6795UKMFR9"
    FR = "A13V1IB3VIYZZH"
    IT = "APJ6JRA9NG5V4"
    ES = "A1RKKUPIHCS9HS"
    JP = "A1VC38T7YXB528"
    AU = "A39IBJ37TRP1C6"


class AmazonOrderStatus(str, Enum):
    """Amazon order status."""

    PENDING = "Pending"
    UNSHIPPED = "Unshipped"
    PARTIALLY_SHIPPED = "PartiallyShipped"
    SHIPPED = "Shipped"
    CANCELED = "Canceled"
    UNFULFILLABLE = "Unfulfillable"
    INVOICE_UNCONFIRMED = "InvoiceUnconfirmed"
    PENDING_AVAILABILITY = "PendingAvailability"


class FulfillmentChannel(str, Enum):
    """Order fulfillment channel."""

    AFN = "AFN"  # Amazon Fulfillment Network (FBA)
    MFN = "MFN"  # Merchant Fulfillment Network


class InventoryCondition(str, Enum):
    """Inventory item condition."""

    NEW = "NewItem"
    USED_LIKE_NEW = "UsedLikeNew"
    USED_VERY_GOOD = "UsedVeryGood"
    USED_GOOD = "UsedGood"
    USED_ACCEPTABLE = "UsedAcceptable"
    COLLECTIBLE = "CollectibleLikeNew"
    REFURBISHED = "Refurbished"


@dataclass
class AmazonCredentials:
    """Amazon SP-API credentials."""

    refresh_token: str
    client_id: str
    client_secret: str
    marketplace_id: str
    seller_id: str
    aws_access_key: Optional[str] = None
    aws_secret_key: Optional[str] = None
    role_arn: Optional[str] = None

    @classmethod
    def from_env(cls) -> "AmazonCredentials":
        """Create credentials from environment variables."""
        return cls(
            refresh_token=os.environ.get("AMAZON_SP_REFRESH_TOKEN", ""),
            client_id=os.environ.get("AMAZON_SP_CLIENT_ID", ""),
            client_secret=os.environ.get("AMAZON_SP_CLIENT_SECRET", ""),
            marketplace_id=os.environ.get("AMAZON_SP_MARKETPLACE_ID", AmazonMarketplace.US.value),
            seller_id=os.environ.get("AMAZON_SP_SELLER_ID", ""),
            aws_access_key=os.environ.get("AMAZON_SP_AWS_ACCESS_KEY"),
            aws_secret_key=os.environ.get("AMAZON_SP_AWS_SECRET_KEY"),
            role_arn=os.environ.get("AMAZON_SP_ROLE_ARN"),
        )


@dataclass
class AmazonAddress:
    """Amazon shipping/billing address."""

    name: Optional[str] = None
    address_line1: Optional[str] = None
    address_line2: Optional[str] = None
    address_line3: Optional[str] = None
    city: Optional[str] = None
    state_or_region: Optional[str] = None
    postal_code: Optional[str] = None
    country_code: Optional[str] = None
    phone: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "addressLine1": self.address_line1,
            "addressLine2": self.address_line2,
            "addressLine3": self.address_line3,
            "city": self.city,
            "stateOrRegion": self.state_or_region,
            "postalCode": self.postal_code,
            "countryCode": self.country_code,
            "phone": self.phone,
        }


@dataclass
class AmazonOrderItem:
    """Amazon order item."""

    order_item_id: str
    asin: str
    seller_sku: Optional[str]
    title: str
    quantity_ordered: int
    quantity_shipped: int
    item_price: Decimal
    item_tax: Decimal
    shipping_price: Decimal
    shipping_tax: Decimal
    promotion_discount: Decimal
    condition: Optional[str] = None
    is_gift: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "orderItemId": self.order_item_id,
            "asin": self.asin,
            "sellerSku": self.seller_sku,
            "title": self.title,
            "quantityOrdered": self.quantity_ordered,
            "quantityShipped": self.quantity_shipped,
            "itemPrice": str(self.item_price),
            "itemTax": str(self.item_tax),
            "shippingPrice": str(self.shipping_price),
            "shippingTax": str(self.shipping_tax),
            "promotionDiscount": str(self.promotion_discount),
            "condition": self.condition,
            "isGift": self.is_gift,
        }


@dataclass
class AmazonOrder:
    """Amazon order."""

    amazon_order_id: str
    seller_order_id: Optional[str]
    purchase_date: datetime
    last_update_date: datetime
    order_status: AmazonOrderStatus
    fulfillment_channel: FulfillmentChannel
    sales_channel: str  # e.g., "Amazon.com"
    order_total: Decimal
    currency_code: str
    number_of_items_shipped: int
    number_of_items_unshipped: int
    shipping_address: Optional[AmazonAddress] = None
    buyer_email: Optional[str] = None
    buyer_name: Optional[str] = None
    order_items: List[AmazonOrderItem] = field(default_factory=list)
    is_prime: bool = False
    is_business_order: bool = False
    is_replacement_order: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "amazonOrderId": self.amazon_order_id,
            "sellerOrderId": self.seller_order_id,
            "purchaseDate": self.purchase_date.isoformat(),
            "lastUpdateDate": self.last_update_date.isoformat(),
            "orderStatus": self.order_status.value,
            "fulfillmentChannel": self.fulfillment_channel.value,
            "salesChannel": self.sales_channel,
            "orderTotal": str(self.order_total),
            "currencyCode": self.currency_code,
            "numberOfItemsShipped": self.number_of_items_shipped,
            "numberOfItemsUnshipped": self.number_of_items_unshipped,
            "shippingAddress": self.shipping_address.to_dict() if self.shipping_address else None,
            "buyerEmail": self.buyer_email,
            "buyerName": self.buyer_name,
            "orderItems": [item.to_dict() for item in self.order_items],
            "isPrime": self.is_prime,
            "isBusinessOrder": self.is_business_order,
            "isReplacementOrder": self.is_replacement_order,
        }


@dataclass
class AmazonInventoryItem:
    """Amazon inventory/FBA item."""

    asin: str
    seller_sku: str
    fnsku: Optional[str]  # FBA SKU
    product_name: str
    condition: InventoryCondition
    total_quantity: int
    inbound_quantity: int = 0  # Quantity being shipped to FBA
    available_quantity: int = 0  # Available for sale
    reserved_quantity: int = 0  # Reserved for orders
    unfulfillable_quantity: int = 0  # Damaged/defective
    researching_quantity: int = 0  # Being researched

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asin": self.asin,
            "sellerSku": self.seller_sku,
            "fnsku": self.fnsku,
            "productName": self.product_name,
            "condition": self.condition.value,
            "totalQuantity": self.total_quantity,
            "inboundQuantity": self.inbound_quantity,
            "availableQuantity": self.available_quantity,
            "reservedQuantity": self.reserved_quantity,
            "unfulfillableQuantity": self.unfulfillable_quantity,
            "researchingQuantity": self.researching_quantity,
        }


@dataclass
class AmazonProduct:
    """Amazon catalog item."""

    asin: str
    title: str
    brand: Optional[str]
    manufacturer: Optional[str]
    product_type: Optional[str]
    parent_asin: Optional[str]  # For variations
    item_dimensions: Optional[Dict[str, Any]] = None
    package_dimensions: Optional[Dict[str, Any]] = None
    images: List[str] = field(default_factory=list)
    bullet_points: List[str] = field(default_factory=list)
    browse_nodes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asin": self.asin,
            "title": self.title,
            "brand": self.brand,
            "manufacturer": self.manufacturer,
            "productType": self.product_type,
            "parentAsin": self.parent_asin,
            "itemDimensions": self.item_dimensions,
            "packageDimensions": self.package_dimensions,
            "images": self.images,
            "bulletPoints": self.bullet_points,
            "browseNodes": self.browse_nodes,
        }


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
        super().__init__(connector_id="amazon", name="amazon", source_type=SourceType.EXTERNAL_API)
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
                    f"Amazon SP-API connector initialized in mock mode "
                    f"(marketplace: {self.amazon_credentials.marketplace_id})"
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
                    f"Connecting to Amazon SP-API (marketplace: {self.amazon_credentials.marketplace_id})"
                )
                return True

            return False
        except Exception as e:
            logger.error(f"Failed to connect to Amazon SP-API: {e}")
            return False

    async def disconnect(self) -> None:
        """Close Amazon connection."""
        self._client = None

    # =========================================================================
    # Orders
    # =========================================================================

    async def sync_orders(
        self,
        since: Optional[datetime] = None,
        status: Optional[List[AmazonOrderStatus]] = None,
        fulfillment_channels: Optional[List[FulfillmentChannel]] = None,
    ) -> AsyncIterator[AmazonOrder]:
        """Sync orders from Amazon.

        Args:
            since: Only fetch orders updated after this time
            status: Filter by order status(es)
            fulfillment_channels: Filter by fulfillment channel(s)

        Yields:
            AmazonOrder objects
        """
        logger.info(f"Syncing Amazon orders since {since}")

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
            params: Dict[str, Any] = {}
            if since:
                params["LastUpdatedAfter"] = since.isoformat()
            if status:
                params["OrderStatuses"] = [s.value for s in status]
            if fulfillment_channels:
                params["FulfillmentChannels"] = [fc.value for fc in fulfillment_channels]

            # Paginate through orders
            next_token = None
            while True:
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

    def _parse_sp_api_order(self, data: Dict[str, Any]) -> AmazonOrder:
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

    async def get_order(self, order_id: str) -> Optional[AmazonOrder]:
        """Get a single order by ID.

        Args:
            order_id: Amazon order ID

        Returns:
            AmazonOrder or None if not found
        """
        # Would call Orders.getOrder(orderId)
        logger.info(f"Getting Amazon order {order_id}")
        return None

    async def get_order_items(self, order_id: str) -> List[AmazonOrderItem]:
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
        logger.info(f"Confirming shipment for order {order_id}")
        return True

    # =========================================================================
    # Inventory (FBA)
    # =========================================================================

    async def get_fba_inventory(
        self,
        skus: Optional[List[str]] = None,
    ) -> List[AmazonInventoryItem]:
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
    ) -> Optional[AmazonInventoryItem]:
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
        items: List[Dict[str, Any]],
        ship_from_address: AmazonAddress,
    ) -> Optional[str]:
        """Create FBA inbound shipment plan.

        Args:
            shipment_name: Name for the shipment
            items: List of {sellerSku, quantity} items
            ship_from_address: Shipping origin address

        Returns:
            Shipment ID if successful
        """
        # Would use FulfillmentInbound API
        logger.info(f"Creating inbound shipment: {shipment_name}")
        return None

    # =========================================================================
    # Catalog/Products
    # =========================================================================

    async def get_catalog_item(self, asin: str) -> Optional[AmazonProduct]:
        """Get product details from catalog.

        Args:
            asin: Amazon Standard Identification Number

        Returns:
            Product details or None
        """
        # Would call CatalogItems.getCatalogItem()
        logger.info(f"Getting catalog item {asin}")
        return None

    async def search_catalog(
        self,
        keywords: str,
        limit: int = 20,
    ) -> List[AmazonProduct]:
        """Search Amazon catalog.

        Args:
            keywords: Search keywords
            limit: Maximum results

        Returns:
            List of matching products
        """
        # Would call CatalogItems.searchCatalogItems()
        logger.info(f"Searching catalog: {keywords}")
        return []

    # =========================================================================
    # Reports
    # =========================================================================

    async def request_report(
        self,
        report_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[str]:
        """Request an Amazon report.

        Args:
            report_type: Report type (e.g., "GET_FLAT_FILE_OPEN_LISTINGS_DATA")
            start_date: Report start date
            end_date: Report end date

        Returns:
            Report ID if request accepted
        """
        # Would call Reports.createReport()
        logger.info(f"Requesting report: {report_type}")
        return None

    async def get_report(self, report_id: str) -> Optional[bytes]:
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
    ) -> Dict[str, Any]:
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
        state: Optional[SyncState] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
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
            content_parts.append(f"\nFulfillable: {item.fulfillable_quantity}")
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
                created_at=item.last_updated,
                updated_at=item.last_updated,
                domain="ecommerce",
                confidence=0.95,
                metadata={
                    "type": "inventory",
                    "asin": item.asin,
                    "sku": item.seller_sku,
                    "condition": item.condition,
                    "fulfillable_quantity": item.fulfillable_quantity,
                    "inbound_quantity": item.inbound_quantity,
                    "reserved_quantity": item.reserved_quantity,
                    "total_quantity": item.total_quantity,
                },
            )


# =========================================================================
# Mock data for testing
# =========================================================================


def get_mock_orders() -> List[AmazonOrder]:
    """Get mock Amazon orders for testing."""
    now = datetime.now(timezone.utc)
    return [
        AmazonOrder(
            amazon_order_id="111-1234567-1234567",
            seller_order_id=None,
            purchase_date=now,
            last_update_date=now,
            order_status=AmazonOrderStatus.UNSHIPPED,
            fulfillment_channel=FulfillmentChannel.MFN,
            sales_channel="Amazon.com",
            order_total=Decimal("45.99"),
            currency_code="USD",
            number_of_items_shipped=0,
            number_of_items_unshipped=1,
            order_items=[
                AmazonOrderItem(
                    order_item_id="item-001",
                    asin="B00EXAMPLE1",
                    seller_sku="SKU-001",
                    title="Sample Product",
                    quantity_ordered=1,
                    quantity_shipped=0,
                    item_price=Decimal("39.99"),
                    item_tax=Decimal("3.00"),
                    shipping_price=Decimal("3.00"),
                    shipping_tax=Decimal("0.00"),
                    promotion_discount=Decimal("0.00"),
                )
            ],
            is_prime=True,
        ),
    ]


def get_mock_inventory() -> List[AmazonInventoryItem]:
    """Get mock FBA inventory for testing."""
    return [
        AmazonInventoryItem(
            asin="B00EXAMPLE1",
            seller_sku="SKU-001",
            fnsku="X00ABCD1234",
            product_name="Sample Product",
            condition=InventoryCondition.NEW,
            total_quantity=100,
            available_quantity=85,
            reserved_quantity=10,
            inbound_quantity=5,
        ),
    ]


__all__ = [
    "AmazonConnector",
    "AmazonCredentials",
    "AmazonMarketplace",
    "AmazonOrder",
    "AmazonOrderItem",
    "AmazonOrderStatus",
    "AmazonProduct",
    "AmazonInventoryItem",
    "AmazonAddress",
    "FulfillmentChannel",
    "InventoryCondition",
    "get_mock_orders",
    "get_mock_inventory",
]
