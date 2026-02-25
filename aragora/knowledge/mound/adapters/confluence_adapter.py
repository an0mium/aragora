"""
ConfluenceAdapter - Bridges Confluence page data to the Knowledge Mound.

Adapts Confluence pages from the enterprise collaboration connector to KM
nodes, enabling cross-debate use of wiki-sourced knowledge (titles, spaces,
content excerpts, labels).

The adapter provides:
- Confluence page-to-KM node conversion with metadata
- Space- and label-based filtering
- Topic-based search across indexed pages
- Graceful degradation when connector data is unavailable
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import KnowledgeItem

EventCallback = Callable[[str, dict[str, Any]], None]

logger = logging.getLogger(__name__)

from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter
from aragora.knowledge.mound.adapters._semantic_mixin import SemanticSearchMixin
from aragora.knowledge.mound.adapters._reverse_flow_base import ReverseFlowMixin
from aragora.knowledge.mound.adapters._fusion_mixin import FusionMixin
from aragora.knowledge.mound.adapters._types import SyncResult


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ConfluenceSearchResult:
    """Wrapper for Confluence search results with similarity metadata."""

    page_id: str
    title: str
    space_key: str
    excerpt: str
    labels: list[str]
    last_modified: str
    similarity: float = 0.0


@dataclass
class ConfluencePageRecord:
    """Lightweight representation of a Confluence page for adapter storage.

    Decouples the adapter from the full ConfluencePage connector type.
    """

    page_id: str
    title: str
    space_key: str
    body: str = ""
    version: int = 1
    url: str = ""
    created_by: str = ""
    updated_by: str = ""
    labels: list[str] = field(default_factory=list)
    parent_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_confluence_page(cls, page: Any) -> ConfluencePageRecord:
        """Create a ConfluencePageRecord from a connector ConfluencePage object."""
        return cls(
            page_id=getattr(page, "id", ""),
            title=getattr(page, "title", ""),
            space_key=getattr(page, "space_key", ""),
            body=getattr(page, "body", ""),
            version=getattr(page, "version", 1),
            url=getattr(page, "url", ""),
            created_by=getattr(page, "created_by", ""),
            updated_by=getattr(page, "updated_by", ""),
            labels=getattr(page, "labels", []),
            parent_id=getattr(page, "parent_id", None),
            created_at=getattr(page, "created_at", None) or datetime.now(timezone.utc),
            updated_at=getattr(page, "updated_at", None) or datetime.now(timezone.utc),
        )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class ConfluenceAdapter(FusionMixin, ReverseFlowMixin, SemanticSearchMixin, KnowledgeMoundAdapter):
    """
    Adapter that bridges Confluence page data to the Knowledge Mound.

    Provides page persistence with metadata extraction, space-based
    filtering, and cross-debate access to Confluence-sourced knowledge.

    Usage:
        adapter = ConfluenceAdapter()
        adapter.store_page(confluence_page)
        await adapter.sync_to_km(mound)
        results = await adapter.search_by_topic("deployment runbook")
    """

    adapter_name = "confluence"
    source_type = "external"

    def __init__(
        self,
        enable_dual_write: bool = False,
        event_callback: EventCallback | None = None,
        enable_resilience: bool = True,
    ):
        super().__init__(
            enable_dual_write=enable_dual_write,
            event_callback=event_callback,
            enable_resilience=enable_resilience,
        )
        self._pending_pages: list[ConfluencePageRecord] = []
        self._synced_pages: dict[str, ConfluencePageRecord] = {}

    def store_page(self, page: Any) -> None:
        """Store a Confluence page for KM sync.

        Args:
            page: A ConfluencePage or ConfluencePageRecord object.
        """
        if isinstance(page, ConfluencePageRecord):
            record = page
        else:
            record = ConfluencePageRecord.from_confluence_page(page)

        record.metadata["km_sync_pending"] = True
        record.metadata["km_sync_requested_at"] = datetime.now(timezone.utc).isoformat()
        self._pending_pages.append(record)

        self._emit_event(
            "km_adapter_forward_sync",
            {
                "adapter": self.adapter_name,
                "page_id": record.page_id,
                "title": record.title[:100],
                "space_key": record.space_key,
            },
        )

    def get(self, record_id: str) -> ConfluencePageRecord | None:
        """Get a page record by ID."""
        clean_id = record_id[5:] if record_id.startswith("conf_") else record_id
        return self._synced_pages.get(clean_id)

    async def get_async(self, record_id: str) -> ConfluencePageRecord | None:
        """Async version of get."""
        return self.get(record_id)

    async def search_by_topic(
        self,
        query: str,
        limit: int = 10,
        space_filter: str = "",
    ) -> list[ConfluenceSearchResult]:
        """Search stored pages by topic similarity.

        Args:
            query: Search query text.
            limit: Max results to return.
            space_filter: Optional space key filter.

        Returns:
            List of ConfluenceSearchResult sorted by relevance.
        """
        results: list[ConfluenceSearchResult] = []
        query_lower = query.lower()

        all_records = list(self._synced_pages.values()) + self._pending_pages
        for record in all_records:
            if space_filter and record.space_key.lower() != space_filter.lower():
                continue

            title_lower = record.title.lower()
            body_lower = record.body.lower()

            if query_lower in title_lower:
                similarity = 0.9
            elif any(word in title_lower for word in query_lower.split()):
                similarity = 0.7
            elif query_lower in body_lower:
                similarity = 0.5
            else:
                continue

            results.append(
                ConfluenceSearchResult(
                    page_id=record.page_id,
                    title=record.title,
                    space_key=record.space_key,
                    excerpt=record.body[:200],
                    labels=record.labels,
                    last_modified=record.updated_at.isoformat() if record.updated_at else "",
                    similarity=similarity,
                )
            )

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    def to_knowledge_item(self, record: ConfluencePageRecord) -> KnowledgeItem:
        """Convert a ConfluencePageRecord to a KnowledgeItem for KM storage."""
        from aragora.knowledge.mound.types import KnowledgeItem, KnowledgeSource
        from aragora.knowledge.unified.types import ConfidenceLevel

        content = f"Confluence: {record.title} (space: {record.space_key})"
        if record.body:
            content += f"\n\n{record.body[:500]}"
        if record.labels:
            content += f"\nLabels: {', '.join(record.labels)}"

        return KnowledgeItem(
            id=f"conf_{record.page_id}",
            content=content,
            source=KnowledgeSource.EXTERNAL,
            source_id=record.page_id,
            confidence=ConfidenceLevel.MEDIUM,
            created_at=record.created_at,
            updated_at=record.updated_at,
            metadata={
                "subcategory": "confluence",
                "title": record.title,
                "space_key": record.space_key,
                "version": record.version,
                "url": record.url,
                "created_by": record.created_by,
                "updated_by": record.updated_by,
                "labels": record.labels,
                "parent_id": record.parent_id,
            },
        )

    async def sync_to_km(
        self,
        mound: Any,
        min_confidence: float = 0.0,
        batch_size: int = 50,
    ) -> SyncResult:
        """Sync pending pages to Knowledge Mound.

        Args:
            mound: The KnowledgeMound instance.
            min_confidence: Minimum confidence to sync.
            batch_size: Max records per batch.

        Returns:
            SyncResult with sync statistics.
        """
        start = datetime.now(timezone.utc)
        synced = 0
        skipped = 0
        failed = 0
        errors: list[str] = []

        pending = self._pending_pages[:batch_size]

        for record in pending:
            try:
                km_item = self.to_knowledge_item(record)

                if hasattr(mound, "store_item"):
                    await mound.store_item(km_item)
                elif hasattr(mound, "store"):
                    await mound.store(km_item)
                elif hasattr(mound, "_semantic_store"):
                    await mound._semantic_store.store(km_item)

                record.metadata["km_sync_pending"] = False
                record.metadata["km_synced_at"] = datetime.now(timezone.utc).isoformat()
                record.metadata["km_item_id"] = km_item.id

                self._synced_pages[record.page_id] = record
                synced += 1

                self._emit_event(
                    "km_adapter_forward_sync_complete",
                    {
                        "adapter": self.adapter_name,
                        "page_id": record.page_id,
                        "title": record.title[:100],
                        "km_item_id": km_item.id,
                    },
                )

            except (RuntimeError, ValueError, OSError, AttributeError) as e:
                failed += 1
                error_msg = f"Failed to sync page {record.title}: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)
                record.metadata["km_sync_error"] = f"Sync failed: {type(e).__name__}"

        synced_ids = {r.page_id for r in pending if r.metadata.get("km_sync_pending") is False}
        self._pending_pages = [r for r in self._pending_pages if r.page_id not in synced_ids]

        duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        return SyncResult(
            records_synced=synced,
            records_skipped=skipped,
            records_failed=failed,
            errors=errors,
            duration_ms=duration_ms,
        )

    async def sync_from_km(
        self,
        mound: Any,
        query: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Query KM for Confluence-sourced knowledge relevant to a topic.

        Args:
            mound: The KnowledgeMound instance.
            query: Topic to search for.
            limit: Max results.

        Returns:
            List of knowledge items matching the query.
        """
        results: list[dict[str, Any]] = []
        try:
            if hasattr(mound, "search"):
                items = await mound.search(
                    query=query,
                    source_filter="external",
                    limit=limit,
                )
                for item in items if items else []:
                    meta = getattr(item, "metadata", {})
                    if meta.get("subcategory") == "confluence":
                        results.append(
                            {
                                "id": getattr(item, "id", ""),
                                "content": getattr(item, "content", ""),
                                "metadata": meta,
                            }
                        )
        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.warning("sync_from_km failed: %s", e)
        return results

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about stored page records."""
        all_records = list(self._synced_pages.values())
        spaces: dict[str, int] = {}
        for r in all_records:
            spaces[r.space_key] = spaces.get(r.space_key, 0) + 1
        return {
            "total_synced": len(self._synced_pages),
            "pending_sync": len(self._pending_pages),
            "space_distribution": spaces,
            "total_labels": len({label for r in all_records for label in r.labels}),
        }

    # --- SemanticSearchMixin required methods ---

    def _get_record_by_id(self, record_id: str) -> ConfluencePageRecord | None:
        return self.get(record_id)

    def _record_to_dict(self, record: Any, similarity: float = 0.0) -> dict[str, Any]:
        return {
            "id": record.page_id,
            "title": record.title,
            "space_key": record.space_key,
            "labels": record.labels,
            "version": record.version,
            "similarity": similarity,
        }

    # --- ReverseFlowMixin required methods ---

    def _get_record_for_validation(self, source_id: str) -> ConfluencePageRecord | None:
        return self.get(source_id)

    def _apply_km_validation(
        self,
        record: Any,
        km_confidence: float,
        cross_refs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        record.metadata["km_validated"] = True
        record.metadata["km_validation_confidence"] = km_confidence
        record.metadata["km_validation_timestamp"] = datetime.now(timezone.utc).isoformat()
        if cross_refs:
            record.metadata["km_cross_references"] = cross_refs
        return True

    def _extract_source_id(self, item: dict[str, Any]) -> str | None:
        source_id = item.get("source_id", "")
        if source_id.startswith("conf_"):
            return source_id[5:]
        return source_id or None

    # --- FusionMixin required methods ---

    def _get_fusion_sources(self) -> list[str]:
        return ["debate", "jira"]

    def _extract_fusible_data(self, km_item: dict[str, Any]) -> dict[str, Any] | None:
        meta = km_item.get("metadata", {})
        if meta.get("subcategory") == "confluence":
            return {
                "title": meta.get("title", ""),
                "space_key": meta.get("space_key", ""),
            }
        return None

    def _apply_fusion_result(
        self,
        record: Any,
        fusion_result: Any,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        record.metadata["fusion_applied"] = True
        record.metadata["fusion_timestamp"] = datetime.now(timezone.utc).isoformat()
        return True
