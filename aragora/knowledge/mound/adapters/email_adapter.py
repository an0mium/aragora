"""
EmailAdapter - Bridges email data (Gmail/Outlook) to the Knowledge Mound.

Adapts email messages from communication connectors to KM nodes, enabling
cross-debate use of email-sourced knowledge. Includes first-pass PII
sanitization to strip email addresses, phone numbers, and SSN patterns
from email bodies before indexing.

The adapter provides:
- Email-to-KM node conversion with metadata (sender, subject, date, thread_id)
- PII sanitization via regex-based stripping
- Topic-based search across indexed emails
- Graceful degradation when connector data is unavailable
"""

from __future__ import annotations

import logging
import re
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
# PII sanitization helpers
# ---------------------------------------------------------------------------

# Email address pattern
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# Phone number patterns (US/international)
_PHONE_RE = re.compile(
    r"(?:\+?1[-.\s]?)?"  # optional country code
    r"(?:\(?\d{3}\)?[-.\s]?)"  # area code
    r"\d{3}[-.\s]?\d{4}"  # number
)

# SSN pattern (XXX-XX-XXXX)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def sanitize_pii(text: str) -> str:
    """Strip PII from text using regex-based first-pass filters.

    Removes:
    - Email addresses -> [EMAIL]
    - Phone numbers -> [PHONE]
    - SSN patterns -> [SSN]

    This is a lightweight filter, not a full DLP solution.

    Args:
        text: Raw text potentially containing PII.

    Returns:
        Sanitized text with PII tokens replaced.
    """
    text = _SSN_RE.sub("[SSN]", text)
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    return text


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class EmailSearchResult:
    """Wrapper for email search results with similarity metadata."""

    email_id: str
    subject: str
    sender: str
    date: str
    thread_id: str
    snippet: str
    similarity: float = 0.0


@dataclass
class EmailRecord:
    """Lightweight representation of an email for adapter storage.

    Decouples the adapter from the full EmailMessage connector type.
    """

    email_id: str
    thread_id: str
    subject: str
    sender: str
    body: str
    date: datetime
    labels: list[str] = field(default_factory=list)
    importance_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_email_message(cls, msg: Any) -> EmailRecord:
        """Create an EmailRecord from a connector EmailMessage object."""
        return cls(
            email_id=getattr(msg, "id", ""),
            thread_id=getattr(msg, "thread_id", ""),
            subject=getattr(msg, "subject", ""),
            sender=getattr(msg, "from_address", ""),
            body=getattr(msg, "body_text", "") or getattr(msg, "snippet", ""),
            date=getattr(msg, "date", datetime.now(timezone.utc)),
            labels=getattr(msg, "labels", []),
            importance_score=getattr(msg, "importance_score", 0.0),
        )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class EmailAdapter(FusionMixin, ReverseFlowMixin, SemanticSearchMixin, KnowledgeMoundAdapter):
    """
    Adapter that bridges email data to the Knowledge Mound.

    Provides email persistence with PII sanitization, thread-based grouping,
    and topic search for cross-debate use of email-sourced knowledge.

    Usage:
        adapter = EmailAdapter()
        adapter.store_email(email_message)
        await adapter.sync_to_km(mound)
        results = await adapter.search_by_topic("quarterly report")
    """

    adapter_name = "email"
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
        self._pending_emails: list[EmailRecord] = []
        self._synced_emails: dict[str, EmailRecord] = {}

    def store_email(self, email: Any) -> None:
        """Store an email for KM sync.

        Args:
            email: An EmailMessage or EmailRecord object.
        """
        if isinstance(email, EmailRecord):
            record = email
        else:
            record = EmailRecord.from_email_message(email)

        record.metadata["km_sync_pending"] = True
        record.metadata["km_sync_requested_at"] = datetime.now(timezone.utc).isoformat()
        self._pending_emails.append(record)

        self._emit_event(
            "km_adapter_forward_sync",
            {
                "adapter": self.adapter_name,
                "email_id": record.email_id,
                "subject": record.subject[:100],
                "thread_id": record.thread_id,
            },
        )

    def get(self, record_id: str) -> EmailRecord | None:
        """Get an email record by ID."""
        clean_id = record_id[4:] if record_id.startswith("eml_") else record_id
        return self._synced_emails.get(clean_id)

    async def get_async(self, record_id: str) -> EmailRecord | None:
        """Async version of get."""
        return self.get(record_id)

    async def search_by_topic(
        self,
        query: str,
        limit: int = 10,
    ) -> list[EmailSearchResult]:
        """Search stored emails by topic similarity.

        Args:
            query: Search query text.
            limit: Max results to return.

        Returns:
            List of EmailSearchResult sorted by relevance.
        """
        results: list[EmailSearchResult] = []
        query_lower = query.lower()

        all_records = list(self._synced_emails.values()) + self._pending_emails
        for record in all_records:
            subject_lower = record.subject.lower()
            body_lower = record.body.lower()

            if query_lower in subject_lower:
                similarity = 0.9
            elif any(word in subject_lower for word in query_lower.split()):
                similarity = 0.7
            elif query_lower in body_lower:
                similarity = 0.5
            else:
                continue

            results.append(
                EmailSearchResult(
                    email_id=record.email_id,
                    subject=record.subject,
                    sender=sanitize_pii(record.sender),
                    date=record.date.isoformat() if record.date else "",
                    thread_id=record.thread_id,
                    snippet=sanitize_pii(record.body[:200]),
                    similarity=similarity,
                )
            )

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:limit]

    def to_knowledge_item(self, record: EmailRecord) -> KnowledgeItem:
        """Convert an EmailRecord to a KnowledgeItem for KM storage."""
        from aragora.knowledge.mound.types import KnowledgeItem, KnowledgeSource
        from aragora.knowledge.unified.types import ConfidenceLevel

        sanitized_body = sanitize_pii(record.body)
        sanitized_sender = sanitize_pii(record.sender)

        content = f"Email: {record.subject}\n\n{sanitized_body[:500]}"

        return KnowledgeItem(
            id=f"eml_{record.email_id}",
            content=content,
            source=KnowledgeSource.EXTERNAL,
            source_id=record.email_id,
            confidence=ConfidenceLevel.from_float(record.importance_score),
            created_at=record.created_at,
            updated_at=record.created_at,
            metadata={
                "subcategory": "email",
                "subject": record.subject,
                "sender": sanitized_sender,
                "date": record.date.isoformat() if record.date else "",
                "thread_id": record.thread_id,
                "labels": record.labels,
                "importance_score": record.importance_score,
            },
        )

    async def sync_to_km(
        self,
        mound: Any,
        min_confidence: float = 0.0,
        batch_size: int = 50,
    ) -> SyncResult:
        """Sync pending email records to Knowledge Mound.

        Args:
            mound: The KnowledgeMound instance.
            min_confidence: Minimum importance score to sync.
            batch_size: Max records per batch.

        Returns:
            SyncResult with sync statistics.
        """
        start = datetime.now(timezone.utc)
        synced = 0
        skipped = 0
        failed = 0
        errors: list[str] = []

        pending = self._pending_emails[:batch_size]

        for record in pending:
            if record.importance_score < min_confidence:
                skipped += 1
                continue

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

                self._synced_emails[record.email_id] = record
                synced += 1

                self._emit_event(
                    "km_adapter_forward_sync_complete",
                    {
                        "adapter": self.adapter_name,
                        "email_id": record.email_id,
                        "km_item_id": km_item.id,
                    },
                )

            except (RuntimeError, ValueError, OSError, AttributeError) as e:
                failed += 1
                error_msg = f"Failed to sync email {record.email_id}: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)
                record.metadata["km_sync_error"] = f"Sync failed: {type(e).__name__}"

        synced_ids = {r.email_id for r in pending if r.metadata.get("km_sync_pending") is False}
        self._pending_emails = [r for r in self._pending_emails if r.email_id not in synced_ids]

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
        """Query KM for email-sourced knowledge relevant to a topic.

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
                    if meta.get("subcategory") == "email":
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
        """Get statistics about stored email records."""
        all_records = list(self._synced_emails.values())
        threads = {r.thread_id for r in all_records if r.thread_id}
        return {
            "total_synced": len(self._synced_emails),
            "pending_sync": len(self._pending_emails),
            "unique_threads": len(threads),
            "avg_importance": (
                sum(r.importance_score for r in all_records) / len(all_records)
                if all_records
                else 0.0
            ),
        }

    # --- SemanticSearchMixin required methods ---

    def _get_record_by_id(self, record_id: str) -> EmailRecord | None:
        return self.get(record_id)

    def _record_to_dict(self, record: Any, similarity: float = 0.0) -> dict[str, Any]:
        return {
            "id": record.email_id,
            "subject": record.subject,
            "sender": sanitize_pii(record.sender),
            "thread_id": record.thread_id,
            "date": record.date.isoformat() if record.date else "",
            "similarity": similarity,
        }

    # --- ReverseFlowMixin required methods ---

    def _get_record_for_validation(self, source_id: str) -> EmailRecord | None:
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
        if source_id.startswith("eml_"):
            return source_id[4:]
        return source_id or None

    # --- FusionMixin required methods ---

    def _get_fusion_sources(self) -> list[str]:
        return ["debate", "compliance"]

    def _extract_fusible_data(self, km_item: dict[str, Any]) -> dict[str, Any] | None:
        meta = km_item.get("metadata", {})
        if meta.get("subcategory") == "email":
            return {
                "importance_score": meta.get("importance_score", 0.0),
                "subject": meta.get("subject", ""),
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
