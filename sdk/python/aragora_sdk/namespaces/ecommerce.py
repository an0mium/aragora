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

    def list_orders(self, status: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List orders."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return self._client.request("GET", "/api/v1/ecommerce/orders", params=params)

    def get_inventory(self, product_id: str | None = None) -> dict[str, Any]:
        """Get inventory status."""
        params: dict[str, Any] = {}
        if product_id:
            params["product_id"] = product_id
        return self._client.request("GET", "/api/v1/ecommerce/inventory", params=params)

    def get_circuit_breaker(self) -> dict[str, Any]:
        """
        Get e-commerce circuit breaker status.

        GET /api/v1/ecommerce/circuit-breaker

        Returns:
            Dict with circuit breaker status
        """
        return self._client.request("GET", "/api/v1/ecommerce/circuit-breaker")


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

    async def list_orders(self, status: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List orders."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return await self._client.request("GET", "/api/v1/ecommerce/orders", params=params)

    async def get_inventory(self, product_id: str | None = None) -> dict[str, Any]:
        """Get inventory status."""
        params: dict[str, Any] = {}
        if product_id:
            params["product_id"] = product_id
        return await self._client.request("GET", "/api/v1/ecommerce/inventory", params=params)

    async def get_circuit_breaker(self) -> dict[str, Any]:
        """Get e-commerce circuit breaker status. GET /api/v1/ecommerce/circuit-breaker"""
        return await self._client.request("GET", "/api/v1/ecommerce/circuit-breaker")
