"""
MarketplaceService -- unified service layer for the marketplace pilot.

Combines the existing MarketplaceCatalog (templates, agent packs, skills,
connectors) with user-facing operations such as listing, searching,
installing, and rating items.

This service is the single entry point used by HTTP handlers and will be
the foundation for the full marketplace GA release.

Usage:
    from aragora.marketplace.service import MarketplaceService

    svc = MarketplaceService()

    # Browse
    listings = svc.list_listings(item_type="template")

    # Detail
    item = svc.get_listing("tpl-code-review")

    # Install
    result = svc.install_listing("tpl-code-review", user_id="user-1")

    # Rate
    svc.rate_listing("tpl-code-review", user_id="user-1", score=5, review="Great!")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .catalog import InstallResult, MarketplaceCatalog, MarketplaceItem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rating model (per-item, not template-specific)
# ---------------------------------------------------------------------------


@dataclass
class ListingRating:
    """A user rating for a marketplace listing."""

    user_id: str
    item_id: str
    score: int  # 1-5
    review: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not 1 <= self.score <= 5:
            raise ValueError("Score must be between 1 and 5")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "user_id": self.user_id,
            "item_id": self.item_id,
            "score": self.score,
            "review": self.review,
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class MarketplaceService:
    """
    Unified marketplace service for the pilot launch.

    Wraps ``MarketplaceCatalog`` and adds rating, per-user install tracking,
    and rich query support.

    Thread-safe: all mutations operate on simple dicts guarded by the GIL.
    For production, the backing store will be replaced with Postgres.
    """

    def __init__(self, catalog: MarketplaceCatalog | None = None) -> None:
        self._catalog = catalog or MarketplaceCatalog(seed=True)
        # item_id -> list of ratings
        self._ratings: dict[str, list[ListingRating]] = {}
        # user_id -> set of installed item_ids
        self._user_installs: dict[str, set[str]] = {}

    # ----- Listing queries --------------------------------------------------

    def list_listings(
        self,
        *,
        item_type: str | None = None,
        tag: str | None = None,
        search: str | None = None,
        category: str | None = None,
        featured_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Browse marketplace listings with optional filters.

        Args:
            item_type: Filter by type (template, agent_pack, skill, connector).
            tag: Filter to items containing this tag.
            search: Free-text search over name and description.
            category: Alias for ``tag`` (kept for backwards compat with frontend).
            featured_only: If True, return only featured items.
            limit: Maximum items to return.
            offset: Pagination offset.

        Returns:
            Dict with ``items``, ``total``, ``limit``, ``offset`` keys.
        """
        if featured_only:
            items = self._catalog.get_featured()
        else:
            effective_tag = tag or category
            items = self._catalog.list_items(
                item_type=item_type,
                tag=effective_tag,
                search=search,
            )

        total = len(items)
        page = items[offset : offset + limit]

        return {
            "items": [self._enrich(i) for i in page],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def get_listing(self, item_id: str) -> dict[str, Any] | None:
        """Get full details for a single listing.

        Args:
            item_id: The marketplace item identifier.

        Returns:
            Enriched item dict or None if not found.
        """
        item = self._catalog.get_item(item_id)
        if item is None:
            return None
        return self._enrich(item)

    # ----- Install ----------------------------------------------------------

    def install_listing(self, item_id: str, user_id: str) -> InstallResult:
        """Install a marketplace listing for a user.

        Args:
            item_id: The item to install.
            user_id: The installing user.

        Returns:
            InstallResult with success status.
        """
        result = self._catalog.install_item(item_id)
        if result.success:
            self._user_installs.setdefault(user_id, set()).add(item_id)
            logger.info("User %s installed marketplace item %s", user_id, item_id)
        return result

    def get_user_installs(self, user_id: str) -> list[str]:
        """Return list of item IDs installed by a user."""
        return sorted(self._user_installs.get(user_id, set()))

    # ----- Rating -----------------------------------------------------------

    def rate_listing(
        self,
        item_id: str,
        *,
        user_id: str,
        score: int,
        review: str | None = None,
    ) -> dict[str, Any]:
        """Rate or update rating for a listing.

        If the user already rated this item, the existing rating is replaced.

        Args:
            item_id: Item to rate.
            user_id: Rating user.
            score: 1-5 integer.
            review: Optional review text.

        Returns:
            Dict with ``success``, ``average_rating``, ``total_ratings``.

        Raises:
            ValueError: If score is outside 1-5.
            KeyError: If item_id not found in catalog.
        """
        if self._catalog.get_item(item_id) is None:
            raise KeyError(f"Item not found: {item_id}")

        rating = ListingRating(
            user_id=user_id,
            item_id=item_id,
            score=score,
            review=review,
        )

        # Upsert: replace existing rating by same user
        ratings = self._ratings.setdefault(item_id, [])
        self._ratings[item_id] = [r for r in ratings if r.user_id != user_id]
        self._ratings[item_id].append(rating)

        avg = self.get_average_rating(item_id)
        total = len(self._ratings[item_id])

        logger.info(
            "User %s rated item %s: %d/5 (avg now %.1f from %d ratings)",
            user_id,
            item_id,
            score,
            avg,
            total,
        )

        return {
            "success": True,
            "average_rating": avg,
            "total_ratings": total,
        }

    def get_ratings(self, item_id: str) -> list[dict[str, Any]]:
        """Get all ratings for an item."""
        return [r.to_dict() for r in self._ratings.get(item_id, [])]

    def get_average_rating(self, item_id: str) -> float:
        """Compute average rating for an item, or 0.0 if no ratings."""
        ratings = self._ratings.get(item_id, [])
        if not ratings:
            return 0.0
        return round(sum(r.score for r in ratings) / len(ratings), 1)

    # ----- Stats ------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics for the marketplace."""
        type_counts = self._catalog.get_types()
        return {
            "total_items": self._catalog.item_count,
            "types": type_counts,
            "total_ratings": sum(len(rs) for rs in self._ratings.values()),
            "total_installs": sum(len(ids) for ids in self._user_installs.values()),
        }

    # ----- Internal ---------------------------------------------------------

    def _enrich(self, item: MarketplaceItem) -> dict[str, Any]:
        """Enrich a catalog item with computed fields (ratings etc.)."""
        d = item.to_dict()
        d["average_rating"] = self.get_average_rating(item.id)
        d["total_ratings"] = len(self._ratings.get(item.id, []))
        return d


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_service: MarketplaceService | None = None


def get_marketplace_service() -> MarketplaceService:
    """Get or create the global MarketplaceService instance."""
    global _service
    if _service is None:
        _service = MarketplaceService()
    return _service


def reset_marketplace_service() -> None:
    """Reset the global instance (for testing)."""
    global _service
    _service = None
