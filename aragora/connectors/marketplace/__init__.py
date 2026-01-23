"""
Marketplace Connectors.

Integrations for online marketplaces:
- Walmart Seller Center (orders, inventory, catalog)
- eBay - planned
- Etsy - planned
"""

from aragora.connectors.marketplace.walmart import (
    WalmartConnector,
    WalmartCredentials,
    WalmartOrder,
    WalmartItem,
    WalmartAddress,
    WalmartReturn,
    WalmartError,
    OrderLine,
    InventoryItem,
    FeedStatus,
    OrderStatus,
    ItemPublishStatus,
    LifecycleStatus,
    FulfillmentType,
    ReturnStatus,
    get_mock_order,
    get_mock_item,
)

__all__ = [
    "WalmartConnector",
    "WalmartCredentials",
    "WalmartOrder",
    "WalmartItem",
    "WalmartAddress",
    "WalmartReturn",
    "WalmartError",
    "OrderLine",
    "InventoryItem",
    "FeedStatus",
    "OrderStatus",
    "ItemPublishStatus",
    "LifecycleStatus",
    "FulfillmentType",
    "ReturnStatus",
    "get_mock_order",
    "get_mock_item",
]
