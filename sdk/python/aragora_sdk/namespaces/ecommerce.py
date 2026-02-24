"""
E-commerce Namespace API

Provides methods for unified e-commerce platform integration:
- Platform connections (Shopify, ShipStation, Walmart)
- Cross-platform order management
- Product and inventory management
- Fulfillment and shipping operations
- E-commerce metrics
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class EcommerceAPI:
    """
    Synchronous E-commerce API.

    Unified interface for e-commerce platforms including Shopify,
    ShipStation, and Walmart Marketplace.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> platforms = client.ecommerce.list_platforms()
        >>> orders = client.ecommerce.list_orders(status="pending")
        >>> client.ecommerce.sync_inventory()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # =========================================================================
    # Platform Management
    # =========================================================================

    def list_platforms(self) -> dict[str, Any]:
        """
        List connected e-commerce platforms.

        Returns:
            Dict with connected platforms and their configurations.
        """
        return self._client.request("GET", "/api/v1/ecommerce/platforms")

    def connect(self, **kwargs: Any) -> dict[str, Any]:
        """
        Connect an e-commerce platform.

        Args:
            **kwargs: Connection parameters including:
                - platform: Platform type (shopify, shipstation, walmart)
                - api_key: Platform API key
                - store_url: Store URL (for Shopify)

        Returns:
            Dict with connection status and platform details.
        """
        return self._client.request("POST", "/api/v1/ecommerce/connect", json=kwargs)

    def disconnect(self, platform: str) -> dict[str, Any]:
        """
        Disconnect an e-commerce platform.

        Args:
            platform: Platform identifier to disconnect.

        Returns:
            Dict with disconnection confirmation.
        """
        return self._client.request("DELETE", f"/api/v1/ecommerce/{platform}")

    # =========================================================================
    # Orders
    # =========================================================================

    def list_orders(self, status: str | None = None, limit: int = 20) -> dict[str, Any]:
        """
        List orders across all connected platforms.

        Args:
            status: Filter by order status (pending, processing, shipped, delivered).
            limit: Maximum number of orders to return.

        Returns:
            Dict with orders list and pagination info.
        """
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return self._client.request("GET", "/api/v1/ecommerce/orders", params=params)

    def list_platform_orders(
        self, platform: str, status: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """
        List orders for a specific platform.

        Args:
            platform: Platform identifier.
            status: Filter by order status.
            limit: Maximum number of orders to return.

        Returns:
            Dict with platform-specific orders.
        """
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return self._client.request("GET", f"/api/v1/ecommerce/{platform}/orders", params=params)

    def get_order(self, platform: str, order_id: str) -> dict[str, Any]:
        """
        Get order details from a specific platform.

        Args:
            platform: Platform identifier.
            order_id: Order identifier.

        Returns:
            Dict with order details.
        """
        return self._client.request("GET", f"/api/v1/ecommerce/{platform}/orders/{order_id}")

    # =========================================================================
    # Products
    # =========================================================================

    def list_products(self, category: str | None = None, limit: int = 20) -> dict[str, Any]:
        """
        List products across all connected platforms.

        Args:
            category: Filter by product category.
            limit: Maximum number of products to return.

        Returns:
            Dict with products list.
        """
        params: dict[str, Any] = {"limit": limit}
        if category:
            params["category"] = category
        return self._client.request("GET", "/api/v1/ecommerce/products", params=params)

    def list_platform_products(self, platform: str, limit: int = 20) -> dict[str, Any]:
        """
        List products for a specific platform.

        Args:
            platform: Platform identifier.
            limit: Maximum number of products to return.

        Returns:
            Dict with platform-specific products.
        """
        params: dict[str, Any] = {"limit": limit}
        return self._client.request("GET", f"/api/v1/ecommerce/{platform}/products", params=params)

    def get_product(self, platform: str, product_id: str) -> dict[str, Any]:
        """
        Get product details from a specific platform.

        Args:
            platform: Platform identifier.
            product_id: Product identifier.

        Returns:
            Dict with product details.
        """
        return self._client.request("GET", f"/api/v1/ecommerce/{platform}/products/{product_id}")

    # =========================================================================
    # Inventory
    # =========================================================================

    def get_inventory(self, product_id: str | None = None) -> dict[str, Any]:
        """
        Get inventory levels across platforms.

        Args:
            product_id: Optional product ID to filter inventory for.

        Returns:
            Dict with inventory levels.
        """
        params: dict[str, Any] = {}
        if product_id:
            params["product_id"] = product_id
        return self._client.request("GET", "/api/v1/ecommerce/inventory", params=params)

    def sync_inventory(self, **kwargs: Any) -> dict[str, Any]:
        """
        Sync inventory levels across connected platforms.

        Args:
            **kwargs: Sync parameters including:
                - platforms: List of platforms to sync (default: all)
                - product_ids: Optional list of specific products to sync

        Returns:
            Dict with sync results and any discrepancies found.
        """
        return self._client.request("POST", "/api/v1/ecommerce/sync-inventory", json=kwargs)

    # =========================================================================
    # Fulfillment & Shipping
    # =========================================================================

    def get_fulfillment(self, **kwargs: Any) -> dict[str, Any]:
        """
        Get fulfillment status across platforms.

        Returns:
            Dict with fulfillment status and pending items.
        """
        return self._client.request("GET", "/api/v1/ecommerce/fulfillment", params=kwargs or None)

    def ship(self, **kwargs: Any) -> dict[str, Any]:
        """
        Create a shipment for an order.

        Args:
            **kwargs: Shipping parameters including:
                - order_id: Order to ship
                - platform: Platform for the order
                - carrier: Shipping carrier
                - tracking_number: Tracking number
                - service: Shipping service level

        Returns:
            Dict with shipment confirmation and tracking info.
        """
        return self._client.request("POST", "/api/v1/ecommerce/ship", json=kwargs)

    # =========================================================================
    # Metrics & Health
    # =========================================================================

    def get_metrics(self) -> dict[str, Any]:
        """
        Get e-commerce metrics overview.

        Returns:
            Dict with metrics including order volume, revenue,
            fulfillment rates, and platform health.
        """
        return self._client.request("GET", "/api/v1/ecommerce/metrics")

    def get_circuit_breaker(self) -> dict[str, Any]:
        """
        Get e-commerce circuit breaker status.

        Returns:
            Dict with circuit breaker state for each platform.
        """
        return self._client.request("GET", "/api/v1/ecommerce/circuit-breaker")


class AsyncEcommerceAPI:
    """
    Asynchronous E-commerce API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     platforms = await client.ecommerce.list_platforms()
        ...     orders = await client.ecommerce.list_orders(status="pending")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # =========================================================================
    # Platform Management
    # =========================================================================

    async def list_platforms(self) -> dict[str, Any]:
        """List connected e-commerce platforms."""
        return await self._client.request("GET", "/api/v1/ecommerce/platforms")

    async def connect(self, **kwargs: Any) -> dict[str, Any]:
        """Connect an e-commerce platform."""
        return await self._client.request("POST", "/api/v1/ecommerce/connect", json=kwargs)

    async def disconnect(self, platform: str) -> dict[str, Any]:
        """Disconnect an e-commerce platform."""
        return await self._client.request("DELETE", f"/api/v1/ecommerce/{platform}")

    # =========================================================================
    # Orders
    # =========================================================================

    async def list_orders(self, status: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List orders across all connected platforms."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return await self._client.request("GET", "/api/v1/ecommerce/orders", params=params)

    async def list_platform_orders(
        self, platform: str, status: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """List orders for a specific platform."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return await self._client.request(
            "GET", f"/api/v1/ecommerce/{platform}/orders", params=params
        )

    async def get_order(self, platform: str, order_id: str) -> dict[str, Any]:
        """Get order details from a specific platform."""
        return await self._client.request("GET", f"/api/v1/ecommerce/{platform}/orders/{order_id}")

    # =========================================================================
    # Products
    # =========================================================================

    async def list_products(self, category: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List products across all connected platforms."""
        params: dict[str, Any] = {"limit": limit}
        if category:
            params["category"] = category
        return await self._client.request("GET", "/api/v1/ecommerce/products", params=params)

    async def list_platform_products(self, platform: str, limit: int = 20) -> dict[str, Any]:
        """List products for a specific platform."""
        params: dict[str, Any] = {"limit": limit}
        return await self._client.request(
            "GET", f"/api/v1/ecommerce/{platform}/products", params=params
        )

    async def get_product(self, platform: str, product_id: str) -> dict[str, Any]:
        """Get product details from a specific platform."""
        return await self._client.request(
            "GET", f"/api/v1/ecommerce/{platform}/products/{product_id}"
        )

    # =========================================================================
    # Inventory
    # =========================================================================

    async def get_inventory(self, product_id: str | None = None) -> dict[str, Any]:
        """Get inventory levels across platforms."""
        params: dict[str, Any] = {}
        if product_id:
            params["product_id"] = product_id
        return await self._client.request("GET", "/api/v1/ecommerce/inventory", params=params)

    async def sync_inventory(self, **kwargs: Any) -> dict[str, Any]:
        """Sync inventory levels across connected platforms."""
        return await self._client.request("POST", "/api/v1/ecommerce/sync-inventory", json=kwargs)

    # =========================================================================
    # Fulfillment & Shipping
    # =========================================================================

    async def get_fulfillment(self, **kwargs: Any) -> dict[str, Any]:
        """Get fulfillment status across platforms."""
        return await self._client.request(
            "GET", "/api/v1/ecommerce/fulfillment", params=kwargs or None
        )

    async def ship(self, **kwargs: Any) -> dict[str, Any]:
        """Create a shipment for an order."""
        return await self._client.request("POST", "/api/v1/ecommerce/ship", json=kwargs)

    # =========================================================================
    # Metrics & Health
    # =========================================================================

    async def get_metrics(self) -> dict[str, Any]:
        """Get e-commerce metrics overview."""
        return await self._client.request("GET", "/api/v1/ecommerce/metrics")

    async def get_circuit_breaker(self) -> dict[str, Any]:
        """Get e-commerce circuit breaker status."""
        return await self._client.request("GET", "/api/v1/ecommerce/circuit-breaker")
