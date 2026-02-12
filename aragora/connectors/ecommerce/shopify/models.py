"""
Shopify data models.

Contains enums and dataclasses for Shopify API data structures:
- Order status and payment status enums
- Credentials configuration
- Order, product, customer, and inventory dataclasses
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from aragora.connectors.model_base import ConnectorDataclass


class ShopifyEnvironment(str, Enum):
    """Shopify environment."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"


class OrderStatus(str, Enum):
    """Order fulfillment status."""

    PENDING = "pending"
    OPEN = "open"
    FULFILLED = "fulfilled"
    PARTIALLY_FULFILLED = "partially_fulfilled"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class PaymentStatus(str, Enum):
    """Order payment status."""

    PENDING = "pending"
    PAID = "paid"
    PARTIALLY_PAID = "partially_paid"
    REFUNDED = "refunded"
    VOIDED = "voided"
    AUTHORIZED = "authorized"


class InventoryPolicy(str, Enum):
    """Inventory tracking policy."""

    DENY = "deny"  # Don't allow overselling
    CONTINUE = "continue"  # Allow overselling


@dataclass
class ShopifyCredentials:
    """OAuth credentials for Shopify."""

    shop_domain: str
    access_token: str
    api_version: str = "2024-01"
    scope: str = ""

    @classmethod
    def from_env(cls) -> ShopifyCredentials:
        """Create credentials from environment variables."""
        return cls(
            shop_domain=os.environ.get("SHOPIFY_SHOP_DOMAIN", ""),
            access_token=os.environ.get("SHOPIFY_ACCESS_TOKEN", ""),
            api_version=os.environ.get("SHOPIFY_API_VERSION", "2024-01"),
        )


@dataclass
class ShopifyAddress(ConnectorDataclass):
    """Shipping or billing address."""

    _field_mapping = {
        "first_name": "firstName",
        "last_name": "lastName",
        "province_code": "provinceCode",
        "country_code": "countryCode",
    }
    _include_none = True

    first_name: str | None = None
    last_name: str | None = None
    company: str | None = None
    address1: str | None = None
    address2: str | None = None
    city: str | None = None
    province: str | None = None
    province_code: str | None = None
    country: str | None = None
    country_code: str | None = None
    zip: str | None = None
    phone: str | None = None

    def to_dict(self, exclude=None, use_api_names=True) -> dict[str, Any]:
        return super().to_dict(exclude=exclude, use_api_names=use_api_names)


@dataclass
class ShopifyLineItem(ConnectorDataclass):
    """Order line item."""

    _field_mapping = {
        "product_id": "productId",
        "variant_id": "variantId",
        "fulfillment_status": "fulfillmentStatus",
        "requires_shipping": "requiresShipping",
    }
    _include_none = True

    id: str
    product_id: str | None
    variant_id: str | None
    title: str
    quantity: int
    price: Decimal
    sku: str | None = None
    vendor: str | None = None
    grams: int = 0
    taxable: bool = True
    fulfillment_status: str | None = None
    requires_shipping: bool = True

    def to_dict(self, exclude=None, use_api_names=True) -> dict[str, Any]:
        return super().to_dict(exclude=exclude, use_api_names=use_api_names)


@dataclass
class ShopifyOrder(ConnectorDataclass):
    """Shopify order."""

    _field_mapping = {
        "order_number": "orderNumber",
        "created_at": "createdAt",
        "updated_at": "updatedAt",
        "total_price": "totalPrice",
        "subtotal_price": "subtotalPrice",
        "total_tax": "totalTax",
        "total_discounts": "totalDiscounts",
        "financial_status": "financialStatus",
        "fulfillment_status": "fulfillmentStatus",
        "line_items": "lineItems",
        "shipping_address": "shippingAddress",
        "billing_address": "billingAddress",
        "customer_id": "customerId",
        "cancelled_at": "cancelledAt",
        "closed_at": "closedAt",
    }
    _include_none = True

    id: str
    order_number: int
    name: str  # e.g., "#1001"
    email: str | None
    created_at: datetime
    updated_at: datetime
    total_price: Decimal
    subtotal_price: Decimal
    total_tax: Decimal
    total_discounts: Decimal
    currency: str
    financial_status: PaymentStatus
    fulfillment_status: OrderStatus | None
    line_items: list[ShopifyLineItem] = field(default_factory=list)
    shipping_address: ShopifyAddress | None = None
    billing_address: ShopifyAddress | None = None
    customer_id: str | None = None
    note: str | None = None
    tags: list[str] = field(default_factory=list)
    cancelled_at: datetime | None = None
    closed_at: datetime | None = None

    def to_dict(self, exclude=None, use_api_names=True) -> dict[str, Any]:
        return super().to_dict(exclude=exclude, use_api_names=use_api_names)


@dataclass
class ShopifyProduct(ConnectorDataclass):
    """Shopify product."""

    _field_mapping = {
        "product_type": "productType",
        "created_at": "createdAt",
        "updated_at": "updatedAt",
        "published_at": "publishedAt",
    }
    _include_none = True

    id: str
    title: str
    handle: str
    vendor: str | None
    product_type: str | None
    status: str  # active, archived, draft
    created_at: datetime
    updated_at: datetime
    published_at: datetime | None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    variants: list[ShopifyVariant] = field(default_factory=list)
    images: list[str] = field(default_factory=list)  # Image URLs

    def to_dict(self, exclude=None, use_api_names=True) -> dict[str, Any]:
        return super().to_dict(exclude=exclude, use_api_names=use_api_names)


@dataclass
class ShopifyVariant(ConnectorDataclass):
    """Product variant."""

    _field_mapping = {
        "product_id": "productId",
        "inventory_quantity": "inventoryQuantity",
        "inventory_policy": "inventoryPolicy",
        "compare_at_price": "compareAtPrice",
        "weight_unit": "weightUnit",
    }
    _include_none = True

    id: str
    product_id: str
    title: str
    price: Decimal
    sku: str | None
    inventory_quantity: int = 0
    inventory_policy: InventoryPolicy = InventoryPolicy.DENY
    compare_at_price: Decimal | None = None
    weight: float = 0.0
    weight_unit: str = "kg"
    barcode: str | None = None
    option1: str | None = None
    option2: str | None = None
    option3: str | None = None

    def to_dict(self, exclude=None, use_api_names=True) -> dict[str, Any]:
        return super().to_dict(exclude=exclude, use_api_names=use_api_names)


@dataclass
class ShopifyCustomer(ConnectorDataclass):
    """Shopify customer."""

    _field_mapping = {
        "first_name": "firstName",
        "last_name": "lastName",
        "created_at": "createdAt",
        "updated_at": "updatedAt",
        "orders_count": "ordersCount",
        "total_spent": "totalSpent",
        "verified_email": "verifiedEmail",
        "accepts_marketing": "acceptsMarketing",
        "tax_exempt": "taxExempt",
    }
    _include_none = True

    id: str
    email: str | None
    first_name: str | None
    last_name: str | None
    phone: str | None
    created_at: datetime
    updated_at: datetime
    orders_count: int = 0
    total_spent: Decimal = Decimal("0.00")
    verified_email: bool = False
    accepts_marketing: bool = False
    tax_exempt: bool = False
    tags: list[str] = field(default_factory=list)
    note: str | None = None

    @property
    def full_name(self) -> str:
        """Get customer full name."""
        parts = [p for p in [self.first_name, self.last_name] if p]
        return " ".join(parts) or "Unknown"

    def to_dict(self, exclude=None, use_api_names=True) -> dict[str, Any]:
        result = super().to_dict(exclude=exclude, use_api_names=use_api_names)
        result["fullName"] = self.full_name
        return result


@dataclass
class ShopifyInventoryLevel(ConnectorDataclass):
    """Inventory level at a location."""

    _field_mapping = {
        "inventory_item_id": "inventoryItemId",
        "location_id": "locationId",
        "updated_at": "updatedAt",
    }
    _include_none = True

    inventory_item_id: str
    location_id: str
    available: int
    updated_at: datetime

    def to_dict(self, exclude=None, use_api_names=True) -> dict[str, Any]:
        return super().to_dict(exclude=exclude, use_api_names=use_api_names)


# =========================================================================
# Mock data for testing
# =========================================================================


def get_mock_orders() -> list[ShopifyOrder]:
    """Get mock Shopify orders for testing."""
    now = datetime.now(timezone.utc)
    return [
        ShopifyOrder(
            id="1001",
            order_number=1001,
            name="#1001",
            email="customer@example.com",
            created_at=now,
            updated_at=now,
            total_price=Decimal("99.99"),
            subtotal_price=Decimal("89.99"),
            total_tax=Decimal("10.00"),
            total_discounts=Decimal("0.00"),
            currency="USD",
            financial_status=PaymentStatus.PAID,
            fulfillment_status=OrderStatus.PENDING,
            line_items=[
                ShopifyLineItem(
                    id="li_1",
                    product_id="prod_1",
                    variant_id="var_1",
                    title="Sample Product",
                    quantity=1,
                    price=Decimal("89.99"),
                    sku="SKU-001",
                )
            ],
            customer_id="cust_1",
        ),
    ]


def get_mock_products() -> list[ShopifyProduct]:
    """Get mock Shopify products for testing."""
    now = datetime.now(timezone.utc)
    return [
        ShopifyProduct(
            id="prod_1",
            title="Sample Product",
            handle="sample-product",
            vendor="Demo Vendor",
            product_type="Physical",
            status="active",
            created_at=now,
            updated_at=now,
            published_at=now,
            description="A sample product for testing",
            variants=[
                ShopifyVariant(
                    id="var_1",
                    product_id="prod_1",
                    title="Default",
                    price=Decimal("89.99"),
                    sku="SKU-001",
                    inventory_quantity=25,
                )
            ],
        ),
    ]


__all__ = [
    "InventoryPolicy",
    "OrderStatus",
    "PaymentStatus",
    "ShopifyAddress",
    "ShopifyCredentials",
    "ShopifyCustomer",
    "ShopifyEnvironment",
    "ShopifyInventoryLevel",
    "ShopifyLineItem",
    "ShopifyOrder",
    "ShopifyProduct",
    "ShopifyVariant",
    "get_mock_orders",
    "get_mock_products",
]
