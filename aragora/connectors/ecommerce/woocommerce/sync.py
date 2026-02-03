"""
WooCommerce synchronization logic.

Provides the sync_items method for Knowledge Mound integration,
yielding SyncItem objects for orders, products, and customers.
"""

from __future__ import annotations

from typing import AsyncIterator

from aragora.connectors.enterprise.base import SyncItem, SyncState


async def sync_items_from_connector(
    connector,
    state: SyncState,
    batch_size: int = 100,
) -> AsyncIterator[SyncItem]:
    """Sync items from WooCommerce for Knowledge Mound.

    This is a standalone helper that operates on a WooCommerceConnector instance.
    The connector's own sync_items method delegates to this function.

    Args:
        connector: A WooCommerceConnector instance
        state: Sync state with cursor/timestamp
        batch_size: Number of items per batch

    Yields:
        SyncItem objects for orders, products, customers
    """
    since = state.last_sync_at

    # Sync orders
    async for order in connector.sync_orders(since=since, per_page=batch_size):
        yield SyncItem(
            id=f"woo-order-{order.id}",
            content=f"Order #{order.number} - {order.status.value} - ${order.total}",
            source_type="woocommerce",
            source_id=str(order.id),
            updated_at=order.date_modified,
            metadata=order.to_dict(),
        )

    # Sync products
    async for product in connector.sync_products(since=since, per_page=batch_size):
        yield SyncItem(
            id=f"woo-product-{product.id}",
            content=f"{product.name}: {product.short_description or ''}",
            source_type="woocommerce",
            source_id=str(product.id),
            updated_at=product.date_modified,
            metadata=product.to_dict(),
        )

    # Sync customers
    async for customer in connector.sync_customers(since=since, per_page=batch_size):
        yield SyncItem(
            id=f"woo-customer-{customer.id}",
            content=f"{customer.first_name} {customer.last_name} - {customer.email}",
            source_type="woocommerce",
            source_id=str(customer.id),
            updated_at=customer.date_modified,
            metadata=customer.to_dict(),
        )


__all__ = [
    "sync_items_from_connector",
]
