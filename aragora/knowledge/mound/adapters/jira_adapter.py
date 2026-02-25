"""
JiraAdapter - Bridges Jira ticket data to the Knowledge Mound.

Adapts Jira issues from the enterprise collaboration connector to KM nodes,
enabling cross-debate use of ticket-sourced knowledge (summaries, statuses,
priorities, labels).

The adapter provides:
- Jira ticket-to-KM node conversion with full metadata
- Status- and priority-based filtering
- Topic-based search across indexed tickets
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
class JiraSearchResult:
    """Wrapper for Jira search results with similarity metadata."""

    ticket_id: str
    key: str
    summary: str
    status: str
    priority: str
    assignee: str
    labels: list[str]
    similarity: float = 0.0


@dataclass
class JiraTicketRecord:
    """Lightweight representation of a Jira ticket for adapter storage.

    Decouples the adapter from the full JiraIssue connector type.
    """

    ticket_id: str
    key: str
    project_key: str
    summary: str
    description: str = ""
    issue_type: str = ""
    status: str = ""
    priority: str = ""
    assignee: str = ""
    reporter: str = ""
    labels: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    url: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_jira_issue(cls, issue: Any) -> JiraTicketRecord:
        """Create a JiraTicketRecord from a connector JiraIssue object."""
        return cls(
            ticket_id=getattr(issue, "id", ""),
            key=getattr(issue, "key", ""),
            project_key=getattr(issue, "project_key", ""),
            summary=getattr(issue, "summary", ""),
            description=getattr(issue, "description", ""),
            issue_type=getattr(issue, "issue_type", ""),
            status=getattr(issue, "status", ""),
            priority=getattr(issue, "priority", ""),
            assignee=getattr(issue, "assignee", ""),
            reporter=getattr(issue, "reporter", ""),
            labels=getattr(issue, "labels", []),
            components=getattr(issue, "components", []),
            url=getattr(issue, "url", ""),
            created_at=getattr(issue, "created_at", None) or datetime.now(timezone.utc),
            updated_at=getattr(issue, "updated_at", None) or datetime.now(timezone.utc),
        )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class JiraAdapter(FusionMixin, ReverseFlowMixin, SemanticSearchMixin, KnowledgeMoundAdapter):
    """
    Adapter that bridges Jira ticket data to the Knowledge Mound.

    Provides ticket persistence with metadata extraction, status-based
    search, and cross-debate access to Jira-sourced knowledge.

    Usage:
        adapter = JiraAdapter()
        adapter.store_ticket(jira_issue)
        await adapter.sync_to_km(mound)
        results = await adapter.search_by_topic("authentication bug")
    """

    adapter_name = "jira"
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
        self._pending_tickets: list[JiraTicketRecord] = []
        self._synced_tickets: dict[str, JiraTicketRecord] = {}

    def store_ticket(self, ticket: Any) -> None:
        """Store a Jira ticket for KM sync.

        Args:
            ticket: A JiraIssue or JiraTicketRecord object.
        """
        if isinstance(ticket, JiraTicketRecord):
            record = ticket
        else:
            record = JiraTicketRecord.from_jira_issue(ticket)

        record.metadata["km_sync_pending"] = True
        record.metadata["km_sync_requested_at"] = datetime.now(timezone.utc).isoformat()
        self._pending_tickets.append(record)

        self._emit_event(
            "km_adapter_forward_sync",
            {
                "adapter": self.adapter_name,
                "ticket_id": record.ticket_id,
                "key": record.key,
                "summary": record.summary[:100],
                "status": record.status,
            },
        )

    def get(self, record_id: str) -> JiraTicketRecord | None:
        """Get a ticket record by ID."""
        clean_id = record_id[5:] if record_id.startswith("jira_") else record_id
        return self._synced_tickets.get(clean_id)

    async def get_async(self, record_id: str) -> JiraTicketRecord | None:
        """Async version of get."""
        return self.get(record_id)

    async def search_by_topic(
        self,
        query: str,
        limit: int = 10,
        status_filter: str = "",
        priority_filter: str = "",
    ) -> list[JiraSearchResult]:
        """Search stored tickets by topic similarity.

        Args:
            query: Search query text.
            limit: Max results to return.
            status_filter: Optional status filter (e.g. "In Progress").
            priority_filter: Optional priority filter (e.g. "High").

        Returns:
            List of JiraSearchResult sorted by relevance.
        """
        results: list[JiraSearchResult] = []
        query_lower = query.lower()

        all_records = list(self._synced_tickets.values()) + self._pending_tickets
        for record in all_records:
            if status_filter and record.status.lower() != status_filter.lower():
                continue
            if priority_filter and record.priority.lower() != priority_filter.lower():
                continue

            summary_lower = record.summary.lower()
            desc_lower = record.description.lower()

            if query_lower in summary_lower:
                similarity = 0.9
            elif any(word in summary_lower for word in query_lower.split()):
                similarity = 0.7
            elif query_lower in desc_lower:
                similarity = 0.5
            else:
                continue

            results.append(
                JiraSearchResult(
                    ticket_id=record.ticket_id,
                    key=record.key,
                    summary=record.summary,
                    status=record.status,
                    priority=record.priority,
                    assignee=record.assignee,
                    labels=record.labels,
                    similarity=similarity,
                )
            )

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    def to_knowledge_item(self, record: JiraTicketRecord) -> KnowledgeItem:
        """Convert a JiraTicketRecord to a KnowledgeItem for KM storage."""
        from aragora.knowledge.mound.types import KnowledgeItem, KnowledgeSource
        from aragora.knowledge.unified.types import ConfidenceLevel

        content = f"[{record.key}] {record.summary}"
        if record.description:
            content += f"\n\n{record.description[:500]}"
        content += f"\nStatus: {record.status} | Priority: {record.priority}"
        if record.labels:
            content += f"\nLabels: {', '.join(record.labels)}"

        # Map priority to confidence
        priority_confidence = {
            "highest": 0.9,
            "high": 0.7,
            "medium": 0.5,
            "low": 0.3,
            "lowest": 0.1,
        }
        confidence = priority_confidence.get(record.priority.lower(), 0.5)

        return KnowledgeItem(
            id=f"jira_{record.ticket_id}",
            content=content,
            source=KnowledgeSource.EXTERNAL,
            source_id=record.ticket_id,
            confidence=ConfidenceLevel.from_float(confidence),
            created_at=record.created_at,
            updated_at=record.updated_at,
            metadata={
                "subcategory": "jira",
                "key": record.key,
                "project_key": record.project_key,
                "summary": record.summary,
                "status": record.status,
                "priority": record.priority,
                "assignee": record.assignee,
                "issue_type": record.issue_type,
                "labels": record.labels,
                "components": record.components,
                "url": record.url,
            },
        )

    async def sync_to_km(
        self,
        mound: Any,
        min_confidence: float = 0.0,
        batch_size: int = 50,
    ) -> SyncResult:
        """Sync pending tickets to Knowledge Mound.

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

        pending = self._pending_tickets[:batch_size]

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

                self._synced_tickets[record.ticket_id] = record
                synced += 1

                self._emit_event(
                    "km_adapter_forward_sync_complete",
                    {
                        "adapter": self.adapter_name,
                        "ticket_id": record.ticket_id,
                        "key": record.key,
                        "km_item_id": km_item.id,
                    },
                )

            except (RuntimeError, ValueError, OSError, AttributeError) as e:
                failed += 1
                error_msg = f"Failed to sync ticket {record.key}: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)
                record.metadata["km_sync_error"] = f"Sync failed: {type(e).__name__}"

        synced_ids = {r.ticket_id for r in pending if r.metadata.get("km_sync_pending") is False}
        self._pending_tickets = [r for r in self._pending_tickets if r.ticket_id not in synced_ids]

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
        """Query KM for Jira-sourced knowledge relevant to a topic.

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
                    if meta.get("subcategory") == "jira":
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
        """Get statistics about stored ticket records."""
        all_records = list(self._synced_tickets.values())
        statuses = {}
        for r in all_records:
            statuses[r.status] = statuses.get(r.status, 0) + 1
        return {
            "total_synced": len(self._synced_tickets),
            "pending_sync": len(self._pending_tickets),
            "status_distribution": statuses,
            "projects": list({r.project_key for r in all_records if r.project_key}),
        }

    # --- SemanticSearchMixin required methods ---

    def _get_record_by_id(self, record_id: str) -> JiraTicketRecord | None:
        return self.get(record_id)

    def _record_to_dict(self, record: Any, similarity: float = 0.0) -> dict[str, Any]:
        return {
            "id": record.ticket_id,
            "key": record.key,
            "summary": record.summary,
            "status": record.status,
            "priority": record.priority,
            "labels": record.labels,
            "similarity": similarity,
        }

    # --- ReverseFlowMixin required methods ---

    def _get_record_for_validation(self, source_id: str) -> JiraTicketRecord | None:
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
        if source_id.startswith("jira_"):
            return source_id[5:]
        return source_id or None

    # --- FusionMixin required methods ---

    def _get_fusion_sources(self) -> list[str]:
        return ["debate", "workflow"]

    def _extract_fusible_data(self, km_item: dict[str, Any]) -> dict[str, Any] | None:
        meta = km_item.get("metadata", {})
        if meta.get("subcategory") == "jira":
            return {
                "status": meta.get("status", ""),
                "priority": meta.get("priority", ""),
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
