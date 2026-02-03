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

# Re-export everything for backward compatibility.
# `from aragora.connectors.ecommerce.woocommerce import WooCommerceConnector` still works.

from aragora.connectors.ecommerce.woocommerce.client import WooCommerceConnector
from aragora.connectors.ecommerce.woocommerce.models import (
    DEFAULT_REQUEST_TIMEOUT,
    WooAddress,
    WooCommerceCredentials,
    WooCustomer,
    WooLineItem,
    WooOrder,
    WooOrderStatus,
    WooProduct,
    WooProductStatus,
    WooProductType,
    WooProductVariation,
    WooStockStatus,
    get_mock_woo_orders,
    get_mock_woo_products,
    validate_id,
)

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
    "validate_id",
    "DEFAULT_REQUEST_TIMEOUT",
]
