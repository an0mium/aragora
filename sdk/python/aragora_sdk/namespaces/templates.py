"""
Templates Namespace API

Provides methods for template management:
- List available templates
- Get template categories
- Get recommendations
- Register custom templates
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class TemplatesAPI:
    """Synchronous Templates API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(self, **params: Any) -> dict[str, Any]:
        """List available templates.

        Args:
            **params: Filter parameters (category, limit, offset, etc.).

        Returns:
            Dict with templates array and pagination.
        """
        return self._client.request("GET", "/api/v1/templates", params=params or None)

    def get_categories(self) -> dict[str, Any]:
        """Get template categories.

        Returns:
            Dict with available template categories.
        """
        return self._client.request("GET", "/api/v1/templates/categories")

    def recommend(self, **params: Any) -> dict[str, Any]:
        """Get template recommendations.

        Args:
            **params: Recommendation parameters (context, task_type, etc.).

        Returns:
            Dict with recommended templates.
        """
        return self._client.request("GET", "/api/v1/templates/recommend", params=params or None)

    def register(self, **kwargs: Any) -> dict[str, Any]:
        """Register a custom template.

        Args:
            **kwargs: Template definition (name, description, config, etc.).

        Returns:
            Dict with registered template details.
        """
        return self._client.request("POST", "/api/v1/templates/registry", json=kwargs)


class AsyncTemplatesAPI:
    """Asynchronous Templates API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(self, **params: Any) -> dict[str, Any]:
        """List available templates."""
        return await self._client.request("GET", "/api/v1/templates", params=params or None)

    async def get_categories(self) -> dict[str, Any]:
        """Get template categories."""
        return await self._client.request("GET", "/api/v1/templates/categories")

    async def recommend(self, **params: Any) -> dict[str, Any]:
        """Get template recommendations."""
        return await self._client.request("GET", "/api/v1/templates/recommend", params=params or None)

    async def register(self, **kwargs: Any) -> dict[str, Any]:
        """Register a custom template."""
        return await self._client.request("POST", "/api/v1/templates/registry", json=kwargs)
