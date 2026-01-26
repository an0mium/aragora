"""
ProvenanceAdapter - Bridges Evidence Provenance to the Knowledge Mound.

This adapter enables automatic persistence of evidence provenance:

- Data flow IN: Verified evidence records are stored as knowledge items
- Data flow IN: Citations link claims to supporting evidence
- Data flow IN: Provenance chains track evidence transformations
- Reverse flow: KM can retrieve past evidence for similar queries

The adapter provides:
- Automatic extraction of verified evidence to knowledge items
- Citation persistence with claim-evidence relationships
- Bidirectional linking between provenance and knowledge items
- Chain integrity verification for compliance

"Every piece of evidence has a story - where it came from, how it was transformed."
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.reasoning.provenance import (
        ProvenanceManager,
        ProvenanceRecord,
    )

from aragora.knowledge.unified.types import (
    ConfidenceLevel,
    KnowledgeItem,
    KnowledgeSource,
    RelationshipType,
)

# Map provenance sources to knowledge sources
PROVENANCE_SOURCE = KnowledgeSource.DEBATE  # Evidence from debates

logger = logging.getLogger(__name__)

# Type alias for event callback
EventCallback = Callable[[str, Dict[str, Any]], None]


class ProvenanceAdapterError(Exception):
    """Base exception for provenance adapter errors."""

    pass


class ChainNotFoundError(ProvenanceAdapterError):
    """Raised when a provenance chain is not found."""

    pass


@dataclass
class ProvenanceIngestionResult:
    """Result of ingesting provenance data into Knowledge Mound."""

    chain_id: str
    debate_id: str
    records_ingested: int
    citations_ingested: int
    relationships_created: int
    knowledge_item_ids: List[str]
    errors: List[str]

    @property
    def success(self) -> bool:
        """Check if ingestion was successful."""
        return len(self.errors) == 0 and self.records_ingested > 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "chain_id": self.chain_id,
            "debate_id": self.debate_id,
            "records_ingested": self.records_ingested,
            "citations_ingested": self.citations_ingested,
            "relationships_created": self.relationships_created,
            "knowledge_item_ids": self.knowledge_item_ids,
            "errors": self.errors,
            "success": self.success,
        }


class ProvenanceAdapter:
    """
    Adapter that bridges Evidence Provenance to the Knowledge Mound.

    Provides methods to:
    - Ingest verified evidence as knowledge items
    - Store citations with claim-evidence relationships
    - Create relationships between provenance components
    - Retrieve past evidence for context

    Usage:
        from aragora.reasoning.provenance import ProvenanceManager
        from aragora.knowledge.mound.adapters import ProvenanceAdapter
        from aragora.knowledge.mound.core import KnowledgeMound

        mound = KnowledgeMound()
        adapter = ProvenanceAdapter(mound)

        # Ingest provenance data
        result = await adapter.ingest_provenance(manager, workspace_id="ws-123")

        # Query related evidence
        related = await adapter.find_related_evidence("contract terms", limit=5)
    """

    ID_PREFIX = "prov_"
    RECORD_PREFIX = "rec_"
    CITATION_PREFIX = "cite_"
    MIN_CONFIDENCE_FOR_EVIDENCE = 0.5

    def __init__(
        self,
        mound: Optional[Any] = None,
        provenance_store: Optional[Any] = None,
        enable_dual_write: bool = True,
        event_callback: Optional[EventCallback] = None,
        auto_ingest: bool = True,
    ):
        """
        Initialize the adapter.

        Args:
            mound: Optional KnowledgeMound instance
            provenance_store: Optional ProvenanceStore for persistence
            enable_dual_write: If True, writes go to both provenance store and KM
            event_callback: Optional callback for emitting events
            auto_ingest: If True, automatically ingest on manager save
        """
        self._mound = mound
        self._provenance_store = provenance_store
        self._enable_dual_write = enable_dual_write
        self._event_callback = event_callback
        self._auto_ingest = auto_ingest
        self._ingested_chains: Dict[str, ProvenanceIngestionResult] = {}

    def set_event_callback(self, callback: EventCallback) -> None:
        """Set the event callback for WebSocket notifications."""
        self._event_callback = callback

    def set_mound(self, mound: Any) -> None:
        """Set the Knowledge Mound instance."""
        self._mound = mound

    def set_provenance_store(self, store: Any) -> None:
        """Set the ProvenanceStore instance."""
        self._provenance_store = store

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event if callback is configured."""
        if self._event_callback:
            try:
                self._event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Failed to emit event {event_type}: {e}")

    async def ingest_provenance(
        self,
        manager: "ProvenanceManager",
        workspace_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> ProvenanceIngestionResult:
        """
        Ingest provenance data into the Knowledge Mound.

        Extracts verified evidence records, citations, and chain data,
        storing them as knowledge items with provenance tracking.

        Args:
            manager: The ProvenanceManager to ingest
            workspace_id: Optional workspace for scoping
            tags: Optional tags to apply to all items

        Returns:
            ProvenanceIngestionResult with counts and any errors
        """
        errors: List[str] = []
        knowledge_item_ids: List[str] = []
        records_ingested = 0
        citations_ingested = 0
        relationships_created = 0

        if not self._mound:
            errors.append("Knowledge Mound not configured")
            return ProvenanceIngestionResult(
                chain_id=manager.chain.chain_id,
                debate_id=manager.debate_id,
                records_ingested=0,
                citations_ingested=0,
                relationships_created=0,
                knowledge_item_ids=[],
                errors=errors,
            )

        base_tags = tags or []
        base_tags.extend(
            [
                f"chain:{manager.chain.chain_id}",
                f"debate:{manager.debate_id}",
            ]
        )

        # 1. Ingest verified evidence records
        for record in manager.chain.records:
            # Only ingest verified or high-confidence records
            if not record.verified and record.confidence < self.MIN_CONFIDENCE_FOR_EVIDENCE:
                continue

            try:
                item = self._record_to_knowledge_item(
                    record,
                    manager,
                    workspace_id,
                    base_tags,
                )
                item_id = await self._store_item(item)
                if item_id:
                    knowledge_item_ids.append(item_id)
                    records_ingested += 1
            except Exception as e:
                errors.append(f"Failed to ingest record {record.id}: {str(e)[:100]}")
                logger.warning(f"Record ingestion failed: {e}")

        # 2. Ingest citations
        for citation in manager.graph.citations.values():
            try:
                item = self._citation_to_knowledge_item(
                    citation,
                    manager,
                    workspace_id,
                    base_tags,
                )
                item_id = await self._store_item(item)
                if item_id:
                    knowledge_item_ids.append(item_id)
                    citations_ingested += 1

                    # Create relationship between claim and evidence
                    claim_item_id = f"{self.RECORD_PREFIX}{citation.claim_id}"
                    evidence_item_id = f"{self.RECORD_PREFIX}{citation.evidence_id}"

                    rel_type = (
                        RelationshipType.SUPPORTS
                        if citation.support_type == "supports"
                        else RelationshipType.CONTRADICTS
                        if citation.support_type == "contradicts"
                        else RelationshipType.RELATED
                    )

                    try:
                        await self._create_relationship(
                            source_id=evidence_item_id,
                            target_id=claim_item_id,
                            relationship_type=rel_type,
                        )
                        relationships_created += 1
                    except Exception:
                        pass  # Non-critical

            except Exception as e:
                errors.append(f"Failed to ingest citation: {str(e)[:100]}")
                logger.warning(f"Citation ingestion failed: {e}")

        # 3. Create chain summary item
        try:
            summary_item = self._chain_to_summary_item(manager, workspace_id, base_tags)
            summary_id = await self._store_item(summary_item)
            if summary_id:
                knowledge_item_ids.append(summary_id)

                # Create relationships from summary to records
                for item_id in knowledge_item_ids[:-1]:
                    if item_id.startswith(self.RECORD_PREFIX):
                        try:
                            await self._create_relationship(
                                source_id=summary_id,
                                target_id=item_id,
                                relationship_type=RelationshipType.CONTAINS,
                            )
                            relationships_created += 1
                        except Exception:
                            pass
        except Exception as e:
            errors.append(f"Failed to create chain summary: {str(e)[:100]}")

        result = ProvenanceIngestionResult(
            chain_id=manager.chain.chain_id,
            debate_id=manager.debate_id,
            records_ingested=records_ingested,
            citations_ingested=citations_ingested,
            relationships_created=relationships_created,
            knowledge_item_ids=knowledge_item_ids,
            errors=errors,
        )

        # Cache result
        self._ingested_chains[manager.chain.chain_id] = result

        # Emit event
        self._emit_event(
            "provenance_ingested",
            {
                "chain_id": manager.chain.chain_id,
                "debate_id": manager.debate_id,
                "records_ingested": records_ingested,
                "citations_ingested": citations_ingested,
            },
        )

        logger.info(
            "provenance_ingested",
            extra={
                "chain_id": manager.chain.chain_id,
                "debate_id": manager.debate_id,
                "records": records_ingested,
                "citations": citations_ingested,
                "relationships": relationships_created,
            },
        )

        return result

    def _record_to_knowledge_item(
        self,
        record: "ProvenanceRecord",
        manager: "ProvenanceManager",
        workspace_id: Optional[str],
        tags: List[str],
    ) -> KnowledgeItem:
        """Convert a provenance record to a knowledge item."""
        item_id = f"{self.RECORD_PREFIX}{record.id}"

        # Map confidence
        if record.confidence >= 0.9:
            confidence = ConfidenceLevel.VERIFIED
        elif record.confidence >= 0.7:
            confidence = ConfidenceLevel.HIGH
        elif record.confidence >= 0.5:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW

        now = datetime.now(timezone.utc)

        return KnowledgeItem(
            id=item_id,
            content=record.content,
            source=PROVENANCE_SOURCE,
            source_id=manager.debate_id,
            confidence=confidence,
            created_at=now,
            updated_at=now,
            metadata={
                "chain_id": manager.chain.chain_id,
                "debate_id": manager.debate_id,
                "record_id": record.id,
                "content_hash": record.content_hash,
                "source_type": record.source_type.value,
                "source_id": record.source_id,
                "transformation": record.transformation.value,
                "transformation_note": record.transformation_note,
                "verified": record.verified,
                "verifier_id": record.verifier_id,
                "parent_ids": record.parent_ids,
                "original_timestamp": record.timestamp.isoformat(),
                "workspace_id": workspace_id or "",
                "tags": tags
                + [
                    "provenance_record",
                    f"source_type:{record.source_type.value}",
                    f"transformation:{record.transformation.value}",
                ],
                "item_type": "provenance_record",
            },
        )

    def _citation_to_knowledge_item(
        self,
        citation: Any,  # Citation type
        manager: "ProvenanceManager",
        workspace_id: Optional[str],
        tags: List[str],
    ) -> KnowledgeItem:
        """Convert a citation to a knowledge item."""
        citation_id = f"{citation.claim_id}:{citation.evidence_id}"
        citation_hash = hashlib.sha256(citation_id.encode()).hexdigest()[:12]
        item_id = f"{self.CITATION_PREFIX}{citation_hash}"

        # Build content
        content = f"Citation: {citation.support_type}"
        if citation.citation_text:
            content += f"\n\n{citation.citation_text}"

        now = datetime.now(timezone.utc)

        return KnowledgeItem(
            id=item_id,
            content=content,
            source=PROVENANCE_SOURCE,
            source_id=manager.debate_id,
            confidence=ConfidenceLevel.MEDIUM,
            created_at=now,
            updated_at=now,
            metadata={
                "chain_id": manager.chain.chain_id,
                "debate_id": manager.debate_id,
                "claim_id": citation.claim_id,
                "evidence_id": citation.evidence_id,
                "relevance": citation.relevance,
                "support_type": citation.support_type,
                "citation_text": citation.citation_text,
                "workspace_id": workspace_id or "",
                "tags": tags
                + [
                    "provenance_citation",
                    f"support_type:{citation.support_type}",
                ],
                "item_type": "provenance_citation",
            },
        )

    def _chain_to_summary_item(
        self,
        manager: "ProvenanceManager",
        workspace_id: Optional[str],
        tags: List[str],
    ) -> KnowledgeItem:
        """Create a summary knowledge item for the chain."""
        item_id = f"{self.ID_PREFIX}{manager.chain.chain_id}"

        # Verify chain integrity
        chain_valid, errors = manager.verify_chain_integrity()

        # Build summary
        source_types = set(r.source_type.value for r in manager.chain.records)
        verified_count = sum(1 for r in manager.chain.records if r.verified)

        summary = (
            f"Provenance Chain: {manager.chain.chain_id}\n\n"
            f"Debate: {manager.debate_id}\n"
            f"Records: {len(manager.chain.records)}\n"
            f"Verified: {verified_count}\n"
            f"Citations: {len(manager.graph.citations)}\n"
            f"Source Types: {', '.join(source_types)}\n"
            f"Chain Integrity: {'valid' if chain_valid else 'compromised'}"
        )

        if errors:
            summary += f"\nIntegrity Errors: {len(errors)}"

        now = datetime.now(timezone.utc)

        return KnowledgeItem(
            id=item_id,
            content=summary,
            source=PROVENANCE_SOURCE,
            source_id=manager.debate_id,
            confidence=ConfidenceLevel.HIGH if chain_valid else ConfidenceLevel.LOW,
            created_at=now,
            updated_at=now,
            metadata={
                "chain_id": manager.chain.chain_id,
                "debate_id": manager.debate_id,
                "genesis_hash": manager.chain.genesis_hash,
                "record_count": len(manager.chain.records),
                "verified_count": verified_count,
                "citation_count": len(manager.graph.citations),
                "source_types": list(source_types),
                "chain_valid": chain_valid,
                "integrity_errors": errors,
                "chain_created_at": manager.chain.created_at.isoformat(),
                "workspace_id": workspace_id or "",
                "tags": tags + ["provenance_chain", "summary"],
                "item_type": "provenance_summary",
            },
        )

    async def _store_item(self, item: KnowledgeItem) -> Optional[str]:
        """Store a knowledge item in the mound."""
        if not self._mound:
            return None

        try:
            if hasattr(self._mound, "store"):
                result = await self._mound.store(item)
                return result.id if hasattr(result, "id") else item.id
            elif hasattr(self._mound, "ingest"):
                await self._mound.ingest(item)
                return item.id
            return item.id
        except Exception as e:
            logger.warning(f"Failed to store item {item.id}: {e}")
            return None

    async def _create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
    ) -> bool:
        """Create a relationship between knowledge items."""
        if not self._mound:
            return False

        try:
            if hasattr(self._mound, "link"):
                await self._mound.link(
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=relationship_type,
                )
                return True
        except Exception as e:
            logger.debug(f"Failed to create relationship: {e}")
        return False

    async def find_related_evidence(
        self,
        query: str,
        workspace_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[KnowledgeItem]:
        """
        Find evidence related to a query.

        Args:
            query: Search query
            workspace_id: Optional workspace filter
            limit: Maximum results

        Returns:
            List of related evidence knowledge items
        """
        if not self._mound:
            return []

        try:
            if hasattr(self._mound, "query"):
                results = await self._mound.query(
                    query=query,
                    tags=["provenance_record"],
                    workspace_id=workspace_id,
                    limit=limit,
                )
                return results.items if hasattr(results, "items") else []
        except Exception as e:
            logger.warning(f"Failed to find related evidence: {e}")

        return []

    async def find_citations_for_claim(
        self,
        claim_id: str,
        workspace_id: Optional[str] = None,
    ) -> List[KnowledgeItem]:
        """
        Find citations for a specific claim.

        Args:
            claim_id: The claim ID to find citations for
            workspace_id: Optional workspace filter

        Returns:
            List of citation knowledge items
        """
        if not self._mound:
            return []

        try:
            if hasattr(self._mound, "query"):
                results = await self._mound.query(
                    query=f"claim:{claim_id}",
                    tags=["provenance_citation"],
                    workspace_id=workspace_id,
                    limit=100,
                )
                return results.items if hasattr(results, "items") else []
        except Exception as e:
            logger.warning(f"Failed to find citations for claim: {e}")

        return []

    def get_ingestion_result(self, chain_id: str) -> Optional[ProvenanceIngestionResult]:
        """Get the ingestion result for a chain."""
        return self._ingested_chains.get(chain_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        total_records = sum(r.records_ingested for r in self._ingested_chains.values())
        total_citations = sum(r.citations_ingested for r in self._ingested_chains.values())
        total_errors = sum(len(r.errors) for r in self._ingested_chains.values())

        return {
            "chains_processed": len(self._ingested_chains),
            "total_records_ingested": total_records,
            "total_citations_ingested": total_citations,
            "total_errors": total_errors,
            "mound_connected": self._mound is not None,
            "provenance_store_connected": self._provenance_store is not None,
            "auto_ingest_enabled": self._auto_ingest,
        }


__all__ = [
    "ProvenanceAdapter",
    "ProvenanceAdapterError",
    "ChainNotFoundError",
    "ProvenanceIngestionResult",
]
