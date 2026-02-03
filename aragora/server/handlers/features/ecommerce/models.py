"""Data models for e-commerce platform integration.

Provides unified representations of orders, products, and platform metadata
that normalize data across Shopify, ShipStation, and Walmart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# Platform credentials storage
_platform_credentials: dict[str, dict[str, Any]] = {}
_platform_connectors: dict[str, Any] = {}


SUPPORTED_PLATFORMS = {
    "shopify": {
        "name": "Shopify",
        "description": "E-commerce platform for online stores",
        "features": ["orders", "products", "customers", "inventory", "fulfillment"],
    },
    "shipstation": {
        "name": "ShipStation",
        "description": "Shipping and fulfillment platform",
        "features": ["shipments", "orders", "carriers", "labels", "tracking"],
    },
    "walmart": {
        "name": "Walmart Marketplace",
        "description": "Walmart seller marketplace",
        "features": ["orders", "items", "inventory", "pricing", "reports"],
    },
}


@dataclass
class UnifiedOrder:
    """Unified order representation across platforms."""

    id: str
    platform: str
    order_number: str
    status: str
    financial_status: str | None
    fulfillment_status: str | None
    customer_email: str | None
    customer_name: str | None
    total_price: float
    subtotal: float
    shipping_price: float
    tax: float
    currency: str
    line_items: list[dict[str, Any]] = field(default_factory=list)
    shipping_address: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "platform": self.platform,
            "order_number": self.order_number,
            "status": self.status,
            "financial_status": self.financial_status,
            "fulfillment_status": self.fulfillment_status,
            "customer_email": self.customer_email,
            "customer_name": self.customer_name,
            "total_price": self.total_price,
            "subtotal": self.subtotal,
            "shipping_price": self.shipping_price,
            "tax": self.tax,
            "currency": self.currency,
            "line_items": self.line_items,
            "shipping_address": self.shipping_address,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class UnifiedProduct:
    """Unified product representation."""

    id: str
    platform: str
    title: str
    sku: str | None
    barcode: str | None
    price: float
    compare_at_price: float | None
    inventory_quantity: int
    status: str
    vendor: str | None
    product_type: str | None
    tags: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    created_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "platform": self.platform,
            "title": self.title,
            "sku": self.sku,
            "barcode": self.barcode,
            "price": self.price,
            "compare_at_price": self.compare_at_price,
            "inventory_quantity": self.inventory_quantity,
            "status": self.status,
            "vendor": self.vendor,
            "product_type": self.product_type,
            "tags": self.tags,
            "images": self.images,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
