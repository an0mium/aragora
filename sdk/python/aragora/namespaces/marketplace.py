"""
Marketplace namespace for template discovery and deployment.

Provides API access to the template marketplace for discovering,
deploying, and managing workflow templates.
"""

from __future__ import annotations

from typing import Any, Literal


class MarketplaceAPI:
    """Synchronous marketplace API."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def list_templates(
        self,
        category: str | None = None,
        sort_by: Literal["downloads", "rating", "recent"] = "downloads",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List available marketplace templates.

        Args:
            category: Filter by category
            sort_by: Sort order
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of templates with pagination
        """
        params: dict[str, Any] = {
            "sort_by": sort_by,
            "limit": limit,
            "offset": offset,
        }
        if category:
            params["category"] = category

        return self._client._request("GET", "/api/v1/marketplace/templates", params=params)

    def search_templates(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Search marketplace templates.

        Args:
            query: Search query
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Matching templates
        """
        params: dict[str, Any] = {"query": query, "limit": limit, "offset": offset}
        return self._client._request("GET", "/api/v1/marketplace/templates/search", params=params)

    def get_template(self, template_id: str) -> dict[str, Any]:
        """
        Get a template by ID.

        Args:
            template_id: Template identifier

        Returns:
            Template details
        """
        return self._client._request("GET", f"/api/v1/marketplace/templates/{template_id}")

    def get_template_reviews(
        self,
        template_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Get reviews for a template.

        Args:
            template_id: Template identifier
            limit: Maximum reviews
            offset: Pagination offset

        Returns:
            Template reviews
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client._request(
            "GET", f"/api/v1/marketplace/templates/{template_id}/reviews", params=params
        )

    def deploy_template(
        self,
        template_id: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Deploy a template to your workspace.

        Args:
            template_id: Template identifier
            name: Custom name for deployed workflow
            config: Template configuration overrides

        Returns:
            Deployment result with workflow_id
        """
        data: dict[str, Any] = {}
        if name:
            data["name"] = name
        if config:
            data["config"] = config

        return self._client._request(
            "POST", f"/api/v1/marketplace/templates/{template_id}/deploy", json=data
        )

    def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """
        Get deployment status.

        Args:
            deployment_id: Deployment identifier

        Returns:
            Deployment status
        """
        return self._client._request("GET", f"/api/v1/marketplace/deployments/{deployment_id}")

    def list_categories(self) -> dict[str, Any]:
        """
        List available template categories.

        Returns:
            List of categories
        """
        return self._client._request("GET", "/api/v1/marketplace/categories")

    def get_featured(self) -> dict[str, Any]:
        """
        Get featured templates.

        Returns:
            Featured templates list
        """
        return self._client._request("GET", "/api/v1/marketplace/featured")

    def submit_review(
        self,
        template_id: str,
        rating: int,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """
        Submit a review for a template.

        Args:
            template_id: Template identifier
            rating: Rating (1-5)
            comment: Review comment

        Returns:
            Review submission confirmation
        """
        data: dict[str, Any] = {"rating": rating}
        if comment:
            data["comment"] = comment

        return self._client._request(
            "POST", f"/api/v1/marketplace/templates/{template_id}/reviews", json=data
        )

    def list_my_deployments(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List templates deployed to your workspace.

        Args:
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of deployments
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client._request("GET", "/api/v1/marketplace/my-deployments", params=params)


class AsyncMarketplaceAPI:
    """Asynchronous marketplace API."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def list_templates(
        self,
        category: str | None = None,
        sort_by: Literal["downloads", "rating", "recent"] = "downloads",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List available marketplace templates."""
        params: dict[str, Any] = {
            "sort_by": sort_by,
            "limit": limit,
            "offset": offset,
        }
        if category:
            params["category"] = category

        return await self._client._request("GET", "/api/v1/marketplace/templates", params=params)

    async def search_templates(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Search marketplace templates."""
        params: dict[str, Any] = {"query": query, "limit": limit, "offset": offset}
        return await self._client._request(
            "GET", "/api/v1/marketplace/templates/search", params=params
        )

    async def get_template(self, template_id: str) -> dict[str, Any]:
        """Get a template by ID."""
        return await self._client._request("GET", f"/api/v1/marketplace/templates/{template_id}")

    async def get_template_reviews(
        self,
        template_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get reviews for a template."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client._request(
            "GET",
            f"/api/v1/marketplace/templates/{template_id}/reviews",
            params=params,
        )

    async def deploy_template(
        self,
        template_id: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Deploy a template to your workspace."""
        data: dict[str, Any] = {}
        if name:
            data["name"] = name
        if config:
            data["config"] = config

        return await self._client._request(
            "POST", f"/api/v1/marketplace/templates/{template_id}/deploy", json=data
        )

    async def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get deployment status."""
        return await self._client._request(
            "GET", f"/api/v1/marketplace/deployments/{deployment_id}"
        )

    async def list_categories(self) -> dict[str, Any]:
        """List available template categories."""
        return await self._client._request("GET", "/api/v1/marketplace/categories")

    async def get_featured(self) -> dict[str, Any]:
        """Get featured templates."""
        return await self._client._request("GET", "/api/v1/marketplace/featured")

    async def submit_review(
        self,
        template_id: str,
        rating: int,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """Submit a review for a template."""
        data: dict[str, Any] = {"rating": rating}
        if comment:
            data["comment"] = comment

        return await self._client._request(
            "POST", f"/api/v1/marketplace/templates/{template_id}/reviews", json=data
        )

    async def list_my_deployments(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List templates deployed to your workspace."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client._request(
            "GET", "/api/v1/marketplace/my-deployments", params=params
        )
