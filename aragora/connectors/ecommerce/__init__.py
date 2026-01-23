"""
E-commerce Connectors.

Integrations for e-commerce platforms:
- Shopify (orders, products, customers)
- Amazon Seller (SP-API) (orders, inventory, catalog)
- eBay (orders, listings) - planned
- WooCommerce (WordPress) (orders, products, customers)
- Magento (Adobe Commerce) - planned
- TikTok Shop - planned
"""

from aragora.connectors.ecommerce.shopify import (
    ShopifyConnector,
    ShopifyCredentials,
    ShopifyEnvironment,
    ShopifyOrder,
    ShopifyProduct,
    ShopifyVariant,
    ShopifyCustomer,
    ShopifyAddress,
    ShopifyLineItem,
    ShopifyInventoryLevel,
    OrderStatus,
    PaymentStatus,
    InventoryPolicy,
    get_mock_orders,
    get_mock_products,
)
from aragora.connectors.ecommerce.amazon import (
    AmazonConnector,
    AmazonCredentials,
    AmazonMarketplace,
    AmazonOrder,
    AmazonOrderItem,
    AmazonOrderStatus,
    AmazonProduct,
    AmazonInventoryItem,
    AmazonAddress,
    FulfillmentChannel,
    InventoryCondition,
    get_mock_orders as get_mock_amazon_orders,
    get_mock_inventory as get_mock_amazon_inventory,
)
from aragora.connectors.ecommerce.woocommerce import (
    WooCommerceConnector,
    WooCommerceCredentials,
    WooOrder,
    WooOrderStatus,
    WooProduct,
    WooProductStatus,
    WooProductType,
    WooProductVariation,
    WooCustomer,
    WooAddress,
    WooLineItem,
    WooStockStatus,
    get_mock_woo_orders,
    get_mock_woo_products,
)

__all__ = [
    # Shopify
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
    # Amazon
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
    "get_mock_amazon_orders",
    "get_mock_amazon_inventory",
    # WooCommerce
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
