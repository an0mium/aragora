"""
Amazon Seller Central Connector Package.

Re-exports all public symbols for backward compatibility.
"""

from aragora.connectors.ecommerce.amazon.client import AmazonConnector
from aragora.connectors.ecommerce.amazon.models import (
    AmazonAddress,
    AmazonCredentials,
    AmazonInventoryItem,
    AmazonMarketplace,
    AmazonOrder,
    AmazonOrderItem,
    AmazonOrderStatus,
    AmazonProduct,
    FulfillmentChannel,
    InventoryCondition,
    get_mock_inventory,
    get_mock_orders,
    get_mock_products,
)

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
    "get_mock_products",
]
