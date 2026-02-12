"""
WooCommerce data models.

Contains enums, dataclasses, validation helpers, and mock data factories
for WooCommerce orders, products, customers, and related entities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from aragora.connectors.exceptions import ConnectorValidationError
from aragora.connectors.model_base import ConnectorDataclass

# Regex pattern for validating IDs (alphanumeric, underscore, hyphen only)
ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

# Default request timeout in seconds
DEFAULT_REQUEST_TIMEOUT = 30


def validate_id(id_value: int | str, field_name: str = "ID") -> None:
    """Validate that an ID contains only alphanumeric characters.

    Args:
        id_value: The ID to validate
        field_name: Name of the field for error messages

    Raises:
        ConnectorValidationError: If the ID contains invalid characters
    """
    str_id = str(id_value)
    if not ID_PATTERN.match(str_id):
        raise ConnectorValidationError(
            f"Invalid {field_name}: '{str_id}' contains invalid characters. "
            "Only alphanumeric characters, underscores, and hyphens are allowed.",
            connector_name="woocommerce",
            field=field_name,
        )


# =========================================================================
# Enums
# =========================================================================


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


# =========================================================================
# Dataclasses
# =========================================================================


@dataclass
class WooCommerceCredentials:
    """WooCommerce API credentials."""

    store_url: str
    consumer_key: str
    consumer_secret: str
    api_version: str = "wc/v3"
    timeout: int = 30

    @classmethod
    def from_env(cls) -> WooCommerceCredentials:
        """Create credentials from environment variables."""
        import os

        return cls(
            store_url=os.environ.get("WOOCOMMERCE_URL", ""),
            consumer_key=os.environ.get("WOOCOMMERCE_CONSUMER_KEY", ""),
            consumer_secret=os.environ.get("WOOCOMMERCE_CONSUMER_SECRET", ""),
            api_version=os.environ.get("WOOCOMMERCE_VERSION", "wc/v3"),
        )


@dataclass
class WooAddress(ConnectorDataclass):
    """WooCommerce address."""

    _field_mapping = {
        "first_name": "firstName",
        "last_name": "lastName",
        "address_1": "address1",
        "address_2": "address2",
    }
    _include_none = True

    first_name: str | None = None
    last_name: str | None = None
    company: str | None = None
    address_1: str | None = None
    address_2: str | None = None
    city: str | None = None
    state: str | None = None
    postcode: str | None = None
    country: str | None = None
    email: str | None = None
    phone: str | None = None

    def to_dict(self, exclude=None, use_api_names=True) -> dict[str, Any]:
        return super().to_dict(exclude=exclude, use_api_names=use_api_names)


@dataclass
class WooLineItem(ConnectorDataclass):
    """WooCommerce order line item."""

    _field_mapping = {
        "product_id": "productId",
        "variation_id": "variationId",
        "tax_class": "taxClass",
        "meta_data": "metaData",
    }
    _include_none = True
    _exclude_fields = {"taxes", "meta_data"}  # Match original output

    id: int
    product_id: int
    variation_id: int
    name: str
    quantity: int
    subtotal: Decimal
    total: Decimal
    sku: str | None = None
    price: Decimal = Decimal("0.00")
    tax_class: str | None = None
    taxes: list[dict[str, Any]] = field(default_factory=list)
    meta_data: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, exclude=None, use_api_names=True) -> dict[str, Any]:
        return super().to_dict(exclude=exclude, use_api_names=use_api_names)


@dataclass
class WooOrder(ConnectorDataclass):
    """WooCommerce order."""

    _field_mapping = {
        "order_key": "orderKey",
        "date_created": "dateCreated",
        "date_modified": "dateModified",
        "total_tax": "totalTax",
        "shipping_total": "shippingTotal",
        "discount_total": "discountTotal",
        "payment_method": "paymentMethod",
        "payment_method_title": "paymentMethodTitle",
        "customer_id": "customerId",
        "line_items": "lineItems",
        "customer_note": "customerNote",
        "date_paid": "datePaid",
        "date_completed": "dateCompleted",
        "transaction_id": "transactionId",
    }
    _include_none = True
    _exclude_fields = {"cart_hash"}

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
    line_items: list[WooLineItem] = field(default_factory=list)
    customer_note: str | None = None
    date_paid: datetime | None = None
    date_completed: datetime | None = None
    cart_hash: str | None = None
    transaction_id: str | None = None

    def to_dict(self, exclude=None, use_api_names=True) -> dict[str, Any]:
        return super().to_dict(exclude=exclude, use_api_names=use_api_names)


@dataclass
class WooProductVariation(ConnectorDataclass):
    """WooCommerce product variation."""

    _field_mapping = {
        "regular_price": "regularPrice",
        "sale_price": "salePrice",
        "stock_quantity": "stockQuantity",
        "stock_status": "stockStatus",
        "manage_stock": "manageStock",
    }
    _include_none = True

    id: int
    sku: str | None
    price: Decimal
    regular_price: Decimal
    sale_price: Decimal | None
    stock_quantity: int | None
    stock_status: WooStockStatus
    manage_stock: bool
    attributes: list[dict[str, str]] = field(default_factory=list)
    image: str | None = None

    def to_dict(self, exclude=None, use_api_names=True) -> dict[str, Any]:
        return super().to_dict(exclude=exclude, use_api_names=use_api_names)


@dataclass
class WooProduct(ConnectorDataclass):
    """WooCommerce product."""

    _field_mapping = {
        "regular_price": "regularPrice",
        "sale_price": "salePrice",
        "date_created": "dateCreated",
        "date_modified": "dateModified",
        "short_description": "shortDescription",
        "stock_quantity": "stockQuantity",
        "stock_status": "stockStatus",
        "manage_stock": "manageStock",
    }
    _include_none = True

    id: int
    name: str
    slug: str
    type: WooProductType
    status: WooProductStatus
    sku: str | None
    price: Decimal
    regular_price: Decimal
    sale_price: Decimal | None
    date_created: datetime
    date_modified: datetime
    description: str | None
    short_description: str | None
    stock_quantity: int | None
    stock_status: WooStockStatus
    manage_stock: bool
    categories: list[dict[str, Any]] = field(default_factory=list)
    tags: list[dict[str, Any]] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    variations: list[WooProductVariation] = field(default_factory=list)
    attributes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, exclude=None, use_api_names=True) -> dict[str, Any]:
        return super().to_dict(exclude=exclude, use_api_names=use_api_names)


@dataclass
class WooCustomer(ConnectorDataclass):
    """WooCommerce customer."""

    _field_mapping = {
        "first_name": "firstName",
        "last_name": "lastName",
        "date_created": "dateCreated",
        "date_modified": "dateModified",
        "is_paying_customer": "isPayingCustomer",
        "orders_count": "ordersCount",
        "total_spent": "totalSpent",
        "avatar_url": "avatarUrl",
    }
    _include_none = True

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
    avatar_url: str | None = None

    def to_dict(self, exclude=None, use_api_names=True) -> dict[str, Any]:
        return super().to_dict(exclude=exclude, use_api_names=use_api_names)


# =========================================================================
# Mock data for testing
# =========================================================================


def get_mock_woo_orders() -> list[WooOrder]:
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


def get_mock_woo_products() -> list[WooProduct]:
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
    "DEFAULT_REQUEST_TIMEOUT",
    "ID_PATTERN",
    "WooAddress",
    "WooCommerceCredentials",
    "WooCustomer",
    "WooLineItem",
    "WooOrder",
    "WooOrderStatus",
    "WooProduct",
    "WooProductStatus",
    "WooProductType",
    "WooProductVariation",
    "WooStockStatus",
    "get_mock_woo_orders",
    "get_mock_woo_products",
    "validate_id",
]
