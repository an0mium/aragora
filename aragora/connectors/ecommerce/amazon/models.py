"""
Amazon Seller Central data models.

Contains enums and dataclasses for Amazon SP-API data structures:
- Marketplaces and order statuses
- Credentials configuration
- Order, inventory, and product dataclasses
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Optional


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
    aws_access_key: str | None = None
    aws_secret_key: str | None = None
    role_arn: str | None = None

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

    name: str | None = None
    address_line1: str | None = None
    address_line2: str | None = None
    address_line3: str | None = None
    city: str | None = None
    state_or_region: str | None = None
    postal_code: str | None = None
    country_code: str | None = None
    phone: str | None = None

    def to_dict(self) -> dict[str, Any]:
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
    seller_sku: str | None
    title: str
    quantity_ordered: int
    quantity_shipped: int
    item_price: Decimal
    item_tax: Decimal
    shipping_price: Decimal
    shipping_tax: Decimal
    promotion_discount: Decimal
    condition: str | None = None
    is_gift: bool = False

    def to_dict(self) -> dict[str, Any]:
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
    seller_order_id: str | None
    purchase_date: datetime
    last_update_date: datetime
    order_status: AmazonOrderStatus
    fulfillment_channel: FulfillmentChannel
    sales_channel: str  # e.g., "Amazon.com"
    order_total: Decimal
    currency_code: str
    number_of_items_shipped: int
    number_of_items_unshipped: int
    shipping_address: AmazonAddress | None = None
    buyer_email: str | None = None
    buyer_name: str | None = None
    order_items: list[AmazonOrderItem] = field(default_factory=list)
    is_prime: bool = False
    is_business_order: bool = False
    is_replacement_order: bool = False

    def to_dict(self) -> dict[str, Any]:
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
    fnsku: str | None  # FBA SKU
    product_name: str
    condition: InventoryCondition
    total_quantity: int
    inbound_quantity: int = 0  # Quantity being shipped to FBA
    available_quantity: int = 0  # Available for sale
    reserved_quantity: int = 0  # Reserved for orders
    unfulfillable_quantity: int = 0  # Damaged/defective
    researching_quantity: int = 0  # Being researched

    def to_dict(self) -> dict[str, Any]:
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
    brand: str | None
    manufacturer: str | None
    product_type: str | None
    parent_asin: str | None  # For variations
    item_dimensions: Optional[dict[str, Any]] = None
    package_dimensions: Optional[dict[str, Any]] = None
    images: list[str] = field(default_factory=list)
    bullet_points: list[str] = field(default_factory=list)
    browse_nodes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
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


# =========================================================================
# Mock data for testing
# =========================================================================


def get_mock_orders() -> list[AmazonOrder]:
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


def get_mock_inventory() -> list[AmazonInventoryItem]:
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


def get_mock_products(keywords: str, limit: int = 10) -> list[AmazonProduct]:
    """Get mock Amazon products for testing catalog search.

    Args:
        keywords: Search keywords (used to customize results)
        limit: Maximum number of products to return

    Returns:
        List of mock AmazonProduct objects
    """
    # Base mock products
    mock_products = [
        AmazonProduct(
            asin="B00EXAMPLE1",
            title="Wireless Bluetooth Headphones with Noise Cancellation",
            brand="TechSound",
            manufacturer="TechSound Electronics",
            product_type="HEADPHONES",
            parent_asin=None,
            images=["https://images.amazon.com/images/I/example1.jpg"],
            bullet_points=[
                "Active Noise Cancellation for immersive sound",
                "40-hour battery life on single charge",
                "Comfortable over-ear design",
                "Bluetooth 5.0 connectivity",
                "Built-in microphone for calls",
            ],
            browse_nodes=["12097479011", "172541"],
        ),
        AmazonProduct(
            asin="B00EXAMPLE2",
            title="Premium Ergonomic Office Chair with Lumbar Support",
            brand="ComfortSeat",
            manufacturer="ComfortSeat Furniture Co.",
            product_type="OFFICE_CHAIR",
            parent_asin=None,
            images=["https://images.amazon.com/images/I/example2.jpg"],
            bullet_points=[
                "Adjustable lumbar support for all-day comfort",
                "Breathable mesh back",
                "Height and armrest adjustable",
                "360-degree swivel",
                "Supports up to 300 lbs",
            ],
            browse_nodes=["1069132", "1063306"],
        ),
        AmazonProduct(
            asin="B00EXAMPLE3",
            title="Stainless Steel Water Bottle - 32oz Insulated",
            brand="HydroMax",
            manufacturer="HydroMax Beverages",
            product_type="WATER_BOTTLE",
            parent_asin="B00PARENT01",
            images=["https://images.amazon.com/images/I/example3.jpg"],
            bullet_points=[
                "Double-wall vacuum insulation",
                "Keeps drinks cold 24hrs or hot 12hrs",
                "BPA-free and leak-proof",
                "Wide mouth for easy cleaning",
                "Fits standard cup holders",
            ],
            browse_nodes=["16225373011", "2619525011"],
        ),
        AmazonProduct(
            asin="B00EXAMPLE4",
            title="Smart LED Desk Lamp with USB Charging Port",
            brand="BrightDesk",
            manufacturer="BrightDesk Lighting",
            product_type="DESK_LAMP",
            parent_asin=None,
            images=["https://images.amazon.com/images/I/example4.jpg"],
            bullet_points=[
                "5 color temperatures and 10 brightness levels",
                "Built-in USB charging port",
                "Touch control panel",
                "Flexible gooseneck design",
                "Eye-caring LED technology",
            ],
            browse_nodes=["1063282", "495224"],
        ),
        AmazonProduct(
            asin="B00EXAMPLE5",
            title="Mechanical Gaming Keyboard RGB Backlit",
            brand="GameMaster",
            manufacturer="GameMaster Peripherals",
            product_type="KEYBOARD",
            parent_asin=None,
            images=["https://images.amazon.com/images/I/example5.jpg"],
            bullet_points=[
                "Cherry MX Blue switches for tactile feedback",
                "Full RGB backlighting with customization software",
                "Aluminum frame construction",
                "Anti-ghosting with N-key rollover",
                "Dedicated media controls",
            ],
            browse_nodes=["12879431", "402052011"],
        ),
    ]

    # Filter by keywords if provided (simple keyword matching)
    if keywords:
        keywords_lower = keywords.lower()
        filtered = [
            p
            for p in mock_products
            if keywords_lower in p.title.lower()
            or (p.brand and keywords_lower in p.brand.lower())
            or (p.product_type and keywords_lower in p.product_type.lower())
        ]
        # If no matches, return all mock products (simulate broad search)
        if not filtered:
            filtered = mock_products
    else:
        filtered = mock_products

    return filtered[:limit]


__all__ = [
    "AmazonAddress",
    "AmazonCredentials",
    "AmazonInventoryItem",
    "AmazonMarketplace",
    "AmazonOrder",
    "AmazonOrderItem",
    "AmazonOrderStatus",
    "AmazonProduct",
    "FulfillmentChannel",
    "InventoryCondition",
    "get_mock_inventory",
    "get_mock_orders",
    "get_mock_products",
]
