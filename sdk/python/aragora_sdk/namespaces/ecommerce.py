"""
E-commerce Namespace API

Provides methods for e-commerce integration:
- Product management
- Order tracking
- Inventory management
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class EcommerceAPI:
    """Synchronous E-commerce API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_products(self, category: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List products."""
        params: dict[str, Any] = {"limit": limit}
        if category:
            params["category"] = category
        return self._client.request("GET", "/api/v1/ecommerce/products", params=params)

    def get_product(self, product_id: str) -> dict[str, Any]:
        """Get product by ID."""
        return self._client.request("GET", f"/api/v1/ecommerce/products/{product_id}")

    def analyze_product(self, product_id: str) -> dict[str, Any]:
        """Analyze a product with AI."""
        return self._client.request("POST", f"/api/v1/ecommerce/products/{product_id}/analyze")

    def list_orders(self, status: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List orders."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return self._client.request("GET", "/api/v1/ecommerce/orders", params=params)

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Get order by ID."""
        return self._client.request("GET", f"/api/v1/ecommerce/orders/{order_id}")

    def get_inventory(self, product_id: str | None = None) -> dict[str, Any]:
        """Get inventory status."""
        params: dict[str, Any] = {}
        if product_id:
            params["product_id"] = product_id
        return self._client.request("GET", "/api/v1/ecommerce/inventory", params=params)

    def sync(self, provider: str) -> dict[str, Any]:
        """Sync with e-commerce provider."""
        return self._client.request("POST", f"/api/v1/ecommerce/sync/{provider}")

    def get_analytics(self, period: str = "30d") -> dict[str, Any]:
        """Get e-commerce analytics."""
        return self._client.request("GET", "/api/v1/ecommerce/analytics", params={"period": period})


class AsyncEcommerceAPI:
    """Asynchronous E-commerce API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_products(self, category: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List products."""
        params: dict[str, Any] = {"limit": limit}
        if category:
            params["category"] = category
        return await self._client.request("GET", "/api/v1/ecommerce/products", params=params)

    async def get_product(self, product_id: str) -> dict[str, Any]:
        """Get product by ID."""
        return await self._client.request("GET", f"/api/v1/ecommerce/products/{product_id}")

    async def analyze_product(self, product_id: str) -> dict[str, Any]:
        """Analyze a product with AI."""
        return await self._client.request(
            "POST", f"/api/v1/ecommerce/products/{product_id}/analyze"
        )

    async def list_orders(self, status: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List orders."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return await self._client.request("GET", "/api/v1/ecommerce/orders", params=params)

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Get order by ID."""
        return await self._client.request("GET", f"/api/v1/ecommerce/orders/{order_id}")

    async def get_inventory(self, product_id: str | None = None) -> dict[str, Any]:
        """Get inventory status."""
        params: dict[str, Any] = {}
        if product_id:
            params["product_id"] = product_id
        return await self._client.request("GET", "/api/v1/ecommerce/inventory", params=params)

    async def sync(self, provider: str) -> dict[str, Any]:
        """Sync with e-commerce provider."""
        return await self._client.request("POST", f"/api/v1/ecommerce/sync/{provider}")

    async def get_analytics(self, period: str = "30d") -> dict[str, Any]:
        """Get e-commerce analytics."""
        return await self._client.request(
            "GET", "/api/v1/ecommerce/analytics", params={"period": period}
        )
