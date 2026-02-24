"""
Marketplace Catalog for listing and discovery.

Provides a unified catalog across item types (workflow templates, agent packs,
skills, and connectors) with search, installation, and curation support.

This is the high-level discovery layer that sits above the existing
TemplateRegistry (marketplace) and SkillMarketplace (skills) stores.

Usage:
    from aragora.marketplace.catalog import MarketplaceCatalog

    catalog = MarketplaceCatalog()

    # Browse all items
    items = catalog.list_items()

    # Search by type
    templates = catalog.list_items(item_type="template")

    # Full-text search
    results = catalog.list_items(search="code review")

    # Get featured / popular
    featured = catalog.get_featured()
    popular = catalog.get_popular(limit=5)

    # Install
    result = catalog.install_item("tpl-code-review")

    # Submit a new community item
    item_id = catalog.submit_item(my_item)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

VALID_ITEM_TYPES = frozenset({"template", "agent_pack", "skill", "connector"})


@dataclass
class MarketplaceItem:
    """A single item in the marketplace catalog."""

    id: str
    name: str
    type: str  # "template" | "agent_pack" | "skill" | "connector"
    description: str
    author: str
    version: str
    downloads: int = 0
    rating: float = 0.0
    tags: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    install_command: str = ""
    featured: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.type not in VALID_ITEM_TYPES:
            raise ValueError(
                f"Invalid item type '{self.type}'. "
                f"Must be one of: {', '.join(sorted(VALID_ITEM_TYPES))}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "author": self.author,
            "version": self.version,
            "downloads": self.downloads,
            "rating": round(self.rating, 1),
            "tags": self.tags,
            "requirements": self.requirements,
            "install_command": self.install_command,
            "featured": self.featured,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class InstallResult:
    """Result of installing a marketplace item."""

    success: bool
    item_id: str
    installed_path: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "item_id": self.item_id,
            "installed_path": self.installed_path,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Seed catalog data
# ---------------------------------------------------------------------------


def _build_seed_catalog() -> list[MarketplaceItem]:
    """Build the default seed catalog with built-in items."""
    items: list[MarketplaceItem] = []

    # --- Workflow templates (5) ---
    items.append(
        MarketplaceItem(
            id="tpl-code-review",
            name="Code Review Pipeline",
            type="template",
            description=(
                "Multi-agent code review workflow with security, performance, "
                "and maintainability perspectives that synthesizes findings into "
                "a unified report."
            ),
            author="aragora",
            version="1.0.0",
            downloads=1240,
            rating=4.7,
            tags=["code", "review", "security", "quality"],
            requirements=[],
            install_command="aragora install template tpl-code-review",
            featured=True,
        )
    )
    items.append(
        MarketplaceItem(
            id="tpl-doc-analysis",
            name="Document Analysis",
            type="template",
            description=(
                "Workflow for analyzing long-form documents: extracts key points, "
                "identifies inconsistencies, and generates an executive summary."
            ),
            author="aragora",
            version="1.0.0",
            downloads=890,
            rating=4.5,
            tags=["document", "analysis", "summary", "extraction"],
            requirements=[],
            install_command="aragora install template tpl-doc-analysis",
            featured=True,
        )
    )
    items.append(
        MarketplaceItem(
            id="tpl-risk-assessment",
            name="Risk Assessment",
            type="template",
            description=(
                "Structured risk evaluation workflow that identifies threats, "
                "scores likelihood and impact, and produces a mitigation plan."
            ),
            author="aragora",
            version="1.0.0",
            downloads=720,
            rating=4.6,
            tags=["risk", "assessment", "compliance", "enterprise"],
            requirements=[],
            install_command="aragora install template tpl-risk-assessment",
            featured=True,
        )
    )
    items.append(
        MarketplaceItem(
            id="tpl-brainstorm",
            name="Brainstorm Session",
            type="template",
            description=(
                "Collaborative ideation template with divergent and convergent "
                "phases, idea clustering, and priority voting."
            ),
            author="aragora",
            version="1.0.0",
            downloads=650,
            rating=4.3,
            tags=["brainstorm", "ideation", "creative", "collaboration"],
            requirements=[],
            install_command="aragora install template tpl-brainstorm",
            featured=False,
        )
    )
    items.append(
        MarketplaceItem(
            id="tpl-compliance-check",
            name="Compliance Check",
            type="template",
            description=(
                "Automated compliance verification workflow that evaluates "
                "policies against regulatory frameworks (SOC 2, GDPR, HIPAA) "
                "and generates audit-ready reports."
            ),
            author="aragora",
            version="1.0.0",
            downloads=580,
            rating=4.4,
            tags=["compliance", "audit", "regulation", "enterprise"],
            requirements=[],
            install_command="aragora install template tpl-compliance-check",
            featured=False,
        )
    )

    # --- Agent packs (5) ---
    items.append(
        MarketplaceItem(
            id="pack-speed",
            name="Speed-Focused Pack",
            type="agent_pack",
            description=(
                "Agent configuration optimized for fast turnaround: "
                "uses lightweight models with aggressive concurrency."
            ),
            author="aragora",
            version="1.0.0",
            downloads=980,
            rating=4.2,
            tags=["speed", "performance", "lightweight"],
            requirements=[],
            install_command="aragora install agent-pack pack-speed",
            featured=True,
        )
    )
    items.append(
        MarketplaceItem(
            id="pack-accuracy",
            name="Accuracy-Focused Pack",
            type="agent_pack",
            description=(
                "Agent configuration prioritizing correctness: frontier models "
                "with extended reasoning and cross-verification."
            ),
            author="aragora",
            version="1.0.0",
            downloads=870,
            rating=4.8,
            tags=["accuracy", "quality", "verification"],
            requirements=[],
            install_command="aragora install agent-pack pack-accuracy",
            featured=True,
        )
    )
    items.append(
        MarketplaceItem(
            id="pack-creative",
            name="Creative Pack",
            type="agent_pack",
            description=(
                "Agent mix tuned for creative tasks: brainstorming, "
                "writing, design thinking, and lateral problem solving."
            ),
            author="aragora",
            version="1.0.0",
            downloads=640,
            rating=4.4,
            tags=["creative", "brainstorm", "writing"],
            requirements=[],
            install_command="aragora install agent-pack pack-creative",
            featured=False,
        )
    )
    items.append(
        MarketplaceItem(
            id="pack-analytical",
            name="Analytical Pack",
            type="agent_pack",
            description=(
                "Agent configuration for data-heavy work: financial analysis, "
                "statistical reasoning, and evidence-based decisions."
            ),
            author="aragora",
            version="1.0.0",
            downloads=550,
            rating=4.5,
            tags=["analytical", "data", "finance", "statistics"],
            requirements=[],
            install_command="aragora install agent-pack pack-analytical",
            featured=False,
        )
    )
    items.append(
        MarketplaceItem(
            id="pack-balanced",
            name="Balanced Pack",
            type="agent_pack",
            description=(
                "General-purpose agent mix balancing speed, accuracy, and cost. "
                "Good default for most debate topics."
            ),
            author="aragora",
            version="1.0.0",
            downloads=1100,
            rating=4.3,
            tags=["balanced", "general", "default"],
            requirements=[],
            install_command="aragora install agent-pack pack-balanced",
            featured=False,
        )
    )

    # --- Skills (5) ---
    items.append(
        MarketplaceItem(
            id="skill-summarize",
            name="Summarize",
            type="skill",
            description=(
                "Produces concise summaries of text, debates, or documents "
                "with configurable length and detail level."
            ),
            author="aragora",
            version="1.0.0",
            downloads=1500,
            rating=4.6,
            tags=["summarize", "text", "nlp"],
            requirements=[],
            install_command="aragora install skill skill-summarize",
            featured=True,
        )
    )
    items.append(
        MarketplaceItem(
            id="skill-translate",
            name="Translate",
            type="skill",
            description=(
                "Multi-language translation skill supporting 50+ languages "
                "with context-aware terminology handling."
            ),
            author="aragora",
            version="1.0.0",
            downloads=1020,
            rating=4.4,
            tags=["translate", "language", "i18n"],
            requirements=[],
            install_command="aragora install skill skill-translate",
            featured=False,
        )
    )
    items.append(
        MarketplaceItem(
            id="skill-extract",
            name="Extract",
            type="skill",
            description=(
                "Structured data extraction from unstructured text: "
                "entities, dates, amounts, relationships, and key facts."
            ),
            author="aragora",
            version="1.0.0",
            downloads=930,
            rating=4.5,
            tags=["extract", "ner", "structured-data", "nlp"],
            requirements=[],
            install_command="aragora install skill skill-extract",
            featured=True,
        )
    )
    items.append(
        MarketplaceItem(
            id="skill-classify",
            name="Classify",
            type="skill",
            description=(
                "Text classification skill with zero-shot and few-shot modes. "
                "Supports custom label sets and confidence scoring."
            ),
            author="aragora",
            version="1.0.0",
            downloads=780,
            rating=4.3,
            tags=["classify", "categorize", "nlp"],
            requirements=[],
            install_command="aragora install skill skill-classify",
            featured=False,
        )
    )
    items.append(
        MarketplaceItem(
            id="skill-compare",
            name="Compare",
            type="skill",
            description=(
                "Side-by-side comparison of documents, proposals, or options "
                "with structured diff output and recommendation."
            ),
            author="aragora",
            version="1.0.0",
            downloads=670,
            rating=4.2,
            tags=["compare", "diff", "analysis"],
            requirements=[],
            install_command="aragora install skill skill-compare",
            featured=False,
        )
    )

    return items


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class MarketplaceCatalog:
    """
    Unified marketplace catalog for discovery and installation.

    Provides browsing, searching, installation, and curation of
    marketplace items across all types (templates, agent packs,
    skills, connectors).

    Items are stored in-memory and seeded with built-in entries
    on construction. Community submissions go through a review
    queue before appearing in search results.
    """

    def __init__(self, seed: bool = True) -> None:
        """Initialize the catalog.

        Args:
            seed: If True, populate with built-in items.
        """
        self._items: dict[str, MarketplaceItem] = {}
        self._pending: dict[str, MarketplaceItem] = {}

        if seed:
            for item in _build_seed_catalog():
                self._items[item.id] = item

    # ----- browsing / search -----------------------------------------------

    def list_items(
        self,
        item_type: str | None = None,
        tag: str | None = None,
        search: str | None = None,
    ) -> list[MarketplaceItem]:
        """Browse the catalog with optional filters.

        Args:
            item_type: Filter by item type (template, agent_pack, skill, connector).
            tag: Filter to items containing this tag.
            search: Free-text search over name and description (case-insensitive).

        Returns:
            List of matching MarketplaceItems sorted by downloads descending.
        """
        results = list(self._items.values())

        if item_type is not None:
            results = [i for i in results if i.type == item_type]

        if tag is not None:
            tag_lower = tag.lower()
            results = [i for i in results if tag_lower in [t.lower() for t in i.tags]]

        if search is not None:
            query = search.lower()
            results = [
                i
                for i in results
                if query in i.name.lower() or query in i.description.lower()
            ]

        results.sort(key=lambda i: i.downloads, reverse=True)
        return results

    def get_item(self, item_id: str) -> MarketplaceItem | None:
        """Get a single item by ID.

        Args:
            item_id: The item identifier.

        Returns:
            MarketplaceItem if found, None otherwise.
        """
        return self._items.get(item_id)

    # ----- installation ----------------------------------------------------

    def install_item(self, item_id: str) -> InstallResult:
        """Install an item from the catalog.

        Increments the download counter and returns an InstallResult.

        Args:
            item_id: The item to install.

        Returns:
            InstallResult indicating success or failure.
        """
        item = self._items.get(item_id)
        if item is None:
            return InstallResult(
                success=False,
                item_id=item_id,
                errors=[f"Item not found: {item_id}"],
            )

        # Increment download count
        item.downloads += 1

        # Determine installed path based on type
        type_dirs = {
            "template": "workflows/templates",
            "agent_pack": "agents/packs",
            "skill": "skills/installed",
            "connector": "connectors/installed",
        }
        installed_path = f"{type_dirs.get(item.type, 'installed')}/{item.id}"

        logger.info("Installed marketplace item: %s (%s)", item.id, item.type)
        return InstallResult(
            success=True,
            item_id=item_id,
            installed_path=installed_path,
        )

    # ----- curation --------------------------------------------------------

    def get_featured(self) -> list[MarketplaceItem]:
        """Return curated featured items.

        Returns:
            List of items marked as featured, sorted by downloads descending.
        """
        featured = [i for i in self._items.values() if i.featured]
        featured.sort(key=lambda i: i.downloads, reverse=True)
        return featured

    def get_popular(self, limit: int = 10) -> list[MarketplaceItem]:
        """Return the most downloaded items.

        Args:
            limit: Maximum number of items to return.

        Returns:
            List of items sorted by download count descending.
        """
        items = sorted(self._items.values(), key=lambda i: i.downloads, reverse=True)
        return items[:limit]

    # ----- submission ------------------------------------------------------

    def submit_item(self, item: MarketplaceItem) -> str:
        """Submit a new item for review.

        The item is placed in a pending queue and will not appear in
        search results until approved.

        Args:
            item: The MarketplaceItem to submit.

        Returns:
            The assigned item ID (may differ from the input if one was
            not provided or already taken).
        """
        # Generate a unique ID if the submitted one collides
        final_id = item.id
        if final_id in self._items or final_id in self._pending:
            final_id = f"{item.id}-{uuid.uuid4().hex[:8]}"
            item = MarketplaceItem(
                id=final_id,
                name=item.name,
                type=item.type,
                description=item.description,
                author=item.author,
                version=item.version,
                downloads=0,
                rating=0.0,
                tags=list(item.tags),
                requirements=list(item.requirements),
                install_command=item.install_command,
                featured=False,
                created_at=datetime.now(timezone.utc),
            )

        self._pending[final_id] = item
        logger.info("Item submitted for review: %s by %s", final_id, item.author)
        return final_id

    # ----- admin helpers (for review workflow) ------------------------------

    def approve_item(self, item_id: str) -> bool:
        """Approve a pending item, moving it into the published catalog.

        Args:
            item_id: The pending item to approve.

        Returns:
            True if the item was found and approved.
        """
        item = self._pending.pop(item_id, None)
        if item is None:
            return False
        self._items[item.id] = item
        logger.info("Approved marketplace item: %s", item_id)
        return True

    def reject_item(self, item_id: str) -> bool:
        """Reject a pending item, removing it from the review queue.

        Args:
            item_id: The pending item to reject.

        Returns:
            True if the item was found and rejected.
        """
        return self._pending.pop(item_id, None) is not None

    def get_pending(self) -> list[MarketplaceItem]:
        """Return all items awaiting review.

        Returns:
            List of pending MarketplaceItems.
        """
        return list(self._pending.values())

    # ----- introspection ---------------------------------------------------

    @property
    def item_count(self) -> int:
        """Total number of published items in the catalog."""
        return len(self._items)

    def get_types(self) -> dict[str, int]:
        """Return a count of items by type.

        Returns:
            Dict mapping item type to count.
        """
        counts: dict[str, int] = {}
        for item in self._items.values():
            counts[item.type] = counts.get(item.type, 0) + 1
        return counts
