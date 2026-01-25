"""Marketplace API for the Aragora SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from aragora_client.client import AragoraClient


class MarketplaceAuthor(BaseModel):
    """Template author information."""

    id: str
    name: str
    verified: bool = False
    avatar_url: str | None = None


class MarketplaceTemplate(BaseModel):
    """Marketplace template model."""

    id: str
    name: str
    description: str | None = None
    category: str | None = None
    industry: str | None = None
    author: MarketplaceAuthor | None = None
    version: str | None = None
    downloads: int = 0
    rating: float | None = None
    ratings_count: int = 0
    price: float | None = None
    is_free: bool = True
    tags: list[str] | None = None
    preview_url: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class MarketplaceReview(BaseModel):
    """Template review."""

    id: str
    template_id: str
    user_id: str
    rating: int
    comment: str | None = None
    created_at: str | None = None


class MarketplacePurchase(BaseModel):
    """Template purchase record."""

    id: str
    template_id: str
    user_id: str
    price: float
    purchased_at: str
    license_type: str | None = None


class MarketplaceAPI:
    """API for marketplace operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # ==========================================================================
    # Browse and Search
    # ==========================================================================

    async def browse(
        self,
        *,
        category: str | None = None,
        industry: str | None = None,
        query: str | None = None,
        sort_by: str | None = None,
        free_only: bool = False,
        limit: int = 20,
        offset: int = 0,
    ) -> list[MarketplaceTemplate]:
        """Browse marketplace templates.

        Args:
            category: Filter by category
            industry: Filter by industry
            query: Search query
            sort_by: Sort field (downloads, rating, newest)
            free_only: Only show free templates
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List of marketplace templates
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if category:
            params["category"] = category
        if industry:
            params["industry"] = industry
        if query:
            params["query"] = query
        if sort_by:
            params["sort_by"] = sort_by
        if free_only:
            params["free_only"] = True

        data = await self._client._get("/api/v1/marketplace/templates", params=params)
        return [
            MarketplaceTemplate.model_validate(t) for t in data.get("templates", [])
        ]

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> list[MarketplaceTemplate]:
        """Search marketplace templates.

        Args:
            query: Search query
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List of matching templates
        """
        params: dict[str, Any] = {"query": query, "limit": limit, "offset": offset}
        data = await self._client._get("/api/v1/marketplace/search", params=params)
        return [
            MarketplaceTemplate.model_validate(t) for t in data.get("templates", [])
        ]

    async def get_template(self, template_id: str) -> MarketplaceTemplate:
        """Get a marketplace template by ID.

        Args:
            template_id: Template ID

        Returns:
            Marketplace template details
        """
        data = await self._client._get(f"/api/v1/marketplace/templates/{template_id}")
        return MarketplaceTemplate.model_validate(data)

    # ==========================================================================
    # Featured and Trending
    # ==========================================================================

    async def get_featured(self) -> list[MarketplaceTemplate]:
        """Get featured marketplace templates.

        Returns:
            List of featured templates
        """
        data = await self._client._get("/api/v1/marketplace/featured")
        return [
            MarketplaceTemplate.model_validate(t) for t in data.get("templates", [])
        ]

    async def get_trending(self) -> list[MarketplaceTemplate]:
        """Get trending marketplace templates.

        Returns:
            List of trending templates
        """
        data = await self._client._get("/api/v1/marketplace/trending")
        return [
            MarketplaceTemplate.model_validate(t) for t in data.get("templates", [])
        ]

    async def get_new_releases(
        self,
        *,
        limit: int = 10,
    ) -> list[MarketplaceTemplate]:
        """Get newly released templates.

        Args:
            limit: Maximum number of results

        Returns:
            List of new templates
        """
        params: dict[str, Any] = {"limit": limit}
        data = await self._client._get("/api/v1/marketplace/new", params=params)
        return [
            MarketplaceTemplate.model_validate(t) for t in data.get("templates", [])
        ]

    # ==========================================================================
    # Categories
    # ==========================================================================

    async def get_categories(self) -> list[str]:
        """Get available marketplace categories.

        Returns:
            List of category names
        """
        data = await self._client._get("/api/v1/marketplace/categories")
        return data.get("categories", [])

    async def get_industries(self) -> list[str]:
        """Get available marketplace industries.

        Returns:
            List of industry names
        """
        data = await self._client._get("/api/v1/marketplace/industries")
        return data.get("industries", [])

    # ==========================================================================
    # Reviews
    # ==========================================================================

    async def get_reviews(
        self,
        template_id: str,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> list[MarketplaceReview]:
        """Get reviews for a template.

        Args:
            template_id: Template ID
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List of reviews
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        data = await self._client._get(
            f"/api/v1/marketplace/templates/{template_id}/reviews", params=params
        )
        return [MarketplaceReview.model_validate(r) for r in data.get("reviews", [])]

    async def add_review(
        self,
        template_id: str,
        *,
        rating: int,
        comment: str | None = None,
    ) -> MarketplaceReview:
        """Add a review for a template.

        Args:
            template_id: Template ID
            rating: Rating (1-5)
            comment: Optional review comment

        Returns:
            Created review
        """
        body: dict[str, Any] = {"rating": rating}
        if comment:
            body["comment"] = comment

        data = await self._client._post(
            f"/api/v1/marketplace/templates/{template_id}/reviews", body
        )
        return MarketplaceReview.model_validate(data)

    # ==========================================================================
    # Purchases and Downloads
    # ==========================================================================

    async def purchase(
        self,
        template_id: str,
        *,
        license_type: str = "standard",
    ) -> MarketplacePurchase:
        """Purchase a marketplace template.

        Args:
            template_id: Template ID
            license_type: License type (standard, extended)

        Returns:
            Purchase record
        """
        body: dict[str, Any] = {"license_type": license_type}
        data = await self._client._post(
            f"/api/v1/marketplace/templates/{template_id}/purchase", body
        )
        return MarketplacePurchase.model_validate(data)

    async def download(
        self,
        template_id: str,
    ) -> dict[str, Any]:
        """Download a marketplace template.

        Args:
            template_id: Template ID

        Returns:
            Template content and metadata
        """
        return await self._client._get(
            f"/api/v1/marketplace/templates/{template_id}/download"
        )

    async def get_my_purchases(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MarketplacePurchase]:
        """Get user's purchased templates.

        Args:
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List of purchases
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        data = await self._client._get("/api/v1/marketplace/purchases", params=params)
        return [
            MarketplacePurchase.model_validate(p) for p in data.get("purchases", [])
        ]

    # ==========================================================================
    # Publishing (for template creators)
    # ==========================================================================

    async def publish(
        self,
        *,
        name: str,
        description: str,
        category: str,
        content: dict[str, Any],
        industry: str | None = None,
        tags: list[str] | None = None,
        price: float | None = None,
        preview_url: str | None = None,
    ) -> MarketplaceTemplate:
        """Publish a new template to the marketplace.

        Args:
            name: Template name
            description: Template description
            category: Template category
            content: Template content
            industry: Target industry
            tags: Template tags
            price: Price (None for free)
            preview_url: Preview image URL

        Returns:
            Published template
        """
        body: dict[str, Any] = {
            "name": name,
            "description": description,
            "category": category,
            "content": content,
        }
        if industry:
            body["industry"] = industry
        if tags:
            body["tags"] = tags
        if price is not None:
            body["price"] = price
        if preview_url:
            body["preview_url"] = preview_url

        data = await self._client._post("/api/v1/marketplace/templates", body)
        return MarketplaceTemplate.model_validate(data)

    async def update_template(
        self,
        template_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        content: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        price: float | None = None,
    ) -> MarketplaceTemplate:
        """Update an owned template.

        Args:
            template_id: Template ID
            name: Updated name
            description: Updated description
            content: Updated content
            tags: Updated tags
            price: Updated price

        Returns:
            Updated template
        """
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if content is not None:
            body["content"] = content
        if tags is not None:
            body["tags"] = tags
        if price is not None:
            body["price"] = price

        data = await self._client._put(
            f"/api/v1/marketplace/templates/{template_id}", body
        )
        return MarketplaceTemplate.model_validate(data)

    async def unpublish(self, template_id: str) -> None:
        """Unpublish an owned template.

        Args:
            template_id: Template ID
        """
        await self._client._delete(f"/api/v1/marketplace/templates/{template_id}")
