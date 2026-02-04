"""
Shopify Connector Package.

Re-exports all public symbols for backward compatibility.
"""

from aragora.connectors.ecommerce.shopify.client import ShopifyConnector
from aragora.connectors.ecommerce.shopify.models import (
    InventoryPolicy,
    OrderStatus,
    PaymentStatus,
    ShopifyAddress,
    ShopifyCredentials,
    ShopifyCustomer,
    ShopifyEnvironment,
    ShopifyInventoryLevel,
    ShopifyLineItem,
    ShopifyOrder,
    ShopifyProduct,
    ShopifyVariant,
    get_mock_orders,
    get_mock_products,
)

__all__ = [
    "ShopifyConnector",
    "ShopifyCredentials",
    "ShopifyEnvironment",
    "ShopifyOrder",
    "ShopifyProduct",
    "ShopifyVariant",
    "ShopifyCustomer",
    "ShopifyAddress",
    "ShopifyLineItem",
    "ShopifyInventoryLevel",
    "OrderStatus",
    "PaymentStatus",
    "InventoryPolicy",
    "get_mock_orders",
    "get_mock_products",
]
