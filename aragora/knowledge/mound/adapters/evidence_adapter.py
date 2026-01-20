"""
EvidenceAdapter - Bridges EvidenceStore to the Knowledge Mound.

This adapter enables bidirectional integration between the Evidence system
and the Knowledge Mound:

- Data flow IN: Evidence snippets with quality scores are stored in KM
- Data flow OUT: Similar evidence is retrieved for deduplication/enrichment
- Reverse flow: KM validation feeds back to evidence reliability scores

The adapter provides:
- Unified search interface (search_by_topic, search_similar)
- Bidirectional sync with dual-write support
- Reliability-to-confidence mapping
- Provenance tracking for evidence chains
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.evidence.store import EvidenceStore
    from aragora.knowledge.mound.types import IngestionRequest

logger = logging.getLogger(__name__)


@dataclass
class EvidenceSearchResult:
    """Wrapper for evidence search results with adapter metadata."""

    evidence: Dict[str, Any]
    relevance_score: float = 0.0
    matched_topics: List[str] = None

    def __post_init__(self) -> None:
        if self.matched_topics is None:
            self.matched_topics = []


class EvidenceAdapter:
    """
    Adapter that bridges EvidenceStore to the Knowledge Mound.

    Provides methods for the Knowledge Mound's federated query system:
    - search_by_topic: Topic-based evidence search
    - search_similar: Find similar evidence for deduplication
    - to_knowledge_item: Convert evidence to unified format
    - store: Store content with KM sync

    Usage:
        from aragora.evidence.store import EvidenceStore
        from aragora.knowledge.mound.adapters import EvidenceAdapter

        store = EvidenceStore()
        adapter = EvidenceAdapter(store)

        # Search for evidence
        results = adapter.search_by_topic("contract law", limit=10)

        # Convert to knowledge items
        items = [adapter.to_knowledge_item(r) for r in results]
    """

    ID_PREFIX = "ev_"
    MIN_RELIABILITY = 0.6
    MIN_QUALITY = 0.7

    def __init__(
        self,
        store: "EvidenceStore",
        enable_dual_write: bool = False,
    ):
        """
        Initialize the adapter.

        Args:
            store: The EvidenceStore instance to wrap
            enable_dual_write: If True, writes go to both systems during migration
        """
        self._store = store
        self._enable_dual_write = enable_dual_write

    @property
    def store(self) -> "EvidenceStore":
        """Access the underlying EvidenceStore."""
        return self._store

    def search_by_topic(
        self,
        query: str,
        limit: int = 10,
        min_reliability: float = 0.0,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search evidence by topic query.

        This method wraps EvidenceStore.search_evidence() to provide the
        interface expected by KnowledgeMound._query_evidence().

        Args:
            query: Search query (keywords are OR'd)
            limit: Maximum results to return
            min_reliability: Minimum reliability score threshold
            source: Optional source filter (e.g., "github", "web")

        Returns:
            List of evidence dicts matching the query
        """
        results = self._store.search_evidence(
            query=query,
            limit=limit,
            min_reliability=min_reliability,
        )

        # Filter by source if specified
        if source:
            results = [r for r in results if r.get("source") == source]

        return results

    def search_similar(
        self,
        content: str,
        limit: int = 5,
        min_similarity: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Find similar evidence for deduplication.

        Args:
            content: Content to find similar evidence for
            limit: Maximum results
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar evidence items
        """
        # Use content hash-based lookup for exact matches
        import hashlib
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:32]

        existing = self._store.get_evidence_by_hash(content_hash)
        if existing:
            return [existing]

        # Fall back to text search for partial matches
        # Extract key terms for search
        words = content.split()[:10]  # First 10 words
        query = " ".join(words)

        return self.search_by_topic(query, limit=limit)

    def get(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific evidence item by ID.

        Args:
            evidence_id: The evidence ID (may be prefixed with "ev_" from mound)

        Returns:
            Evidence dict or None
        """
        # Strip mound prefix if present
        if evidence_id.startswith(self.ID_PREFIX):
            evidence_id = evidence_id[len(self.ID_PREFIX):]

        return self._store.get_evidence(evidence_id)

    def to_knowledge_item(self, evidence: Dict[str, Any]) -> "KnowledgeItem":
        """
        Convert evidence dict to a KnowledgeItem.

        Args:
            evidence: The evidence dictionary

        Returns:
            KnowledgeItem for unified knowledge mound API
        """
        from aragora.knowledge.unified.types import (
            ConfidenceLevel,
            KnowledgeItem,
            KnowledgeSource,
        )

        # Map reliability score to confidence level
        reliability = evidence.get("reliability_score", 0.5)
        if reliability >= 0.9:
            confidence = ConfidenceLevel.VERIFIED
        elif reliability >= 0.7:
            confidence = ConfidenceLevel.HIGH
        elif reliability >= 0.5:
            confidence = ConfidenceLevel.MEDIUM
        elif reliability >= 0.3:
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.UNVERIFIED

        # Parse quality scores if available
        quality_scores = {}
        if evidence.get("quality_scores_json"):
            import json
            try:
                quality_scores = json.loads(evidence["quality_scores_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        # Parse enriched metadata if available
        enriched_metadata = {}
        if evidence.get("enriched_metadata_json"):
            import json
            try:
                enriched_metadata = json.loads(evidence["enriched_metadata_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        # Build metadata
        metadata: Dict[str, Any] = {
            "source": evidence.get("source", "unknown"),
            "title": evidence.get("title", ""),
            "url": evidence.get("url", ""),
            "reliability_score": reliability,
            "quality_scores": quality_scores,
            "enriched": enriched_metadata,
        }

        # Parse timestamps
        created_at = evidence.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_at = datetime.utcnow()
        elif created_at is None:
            created_at = datetime.utcnow()

        updated_at = evidence.get("updated_at")
        if isinstance(updated_at, str):
            try:
                updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            except ValueError:
                updated_at = created_at
        elif updated_at is None:
            updated_at = created_at

        return KnowledgeItem(
            id=f"{self.ID_PREFIX}{evidence['id']}",
            content=evidence.get("snippet", ""),
            source=KnowledgeSource.EVIDENCE,
            source_id=evidence["id"],
            confidence=confidence,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
            importance=reliability,
        )

    def from_ingestion_request(
        self,
        request: "IngestionRequest",
        evidence_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert an IngestionRequest to EvidenceStore save_evidence() parameters.

        Args:
            request: The ingestion request from Knowledge Mound
            evidence_id: Optional ID to use (generates one if not provided)

        Returns:
            Dict of parameters for EvidenceStore.save_evidence()
        """
        import uuid

        return {
            "evidence_id": evidence_id or f"mound_{uuid.uuid4().hex[:12]}",
            "source": request.metadata.get("source", "knowledge_mound"),
            "title": request.metadata.get("title", "Knowledge Mound Entry"),
            "snippet": request.content,
            "url": request.metadata.get("url", ""),
            "reliability_score": request.confidence,
            "metadata": {
                "source_type": request.source_type.value if hasattr(request.source_type, 'value') else str(request.source_type),
                "debate_id": request.debate_id,
                "document_id": request.document_id,
                "agent_id": request.agent_id,
                "user_id": request.user_id,
                "workspace_id": request.workspace_id,
                "topics": request.topics,
                "mound_metadata": request.metadata,
            },
        }

    def store(
        self,
        evidence_id: str,
        source: str,
        title: str,
        snippet: str,
        url: str = "",
        reliability_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        debate_id: Optional[str] = None,
    ) -> str:
        """
        Store evidence with optional KM sync.

        Args:
            evidence_id: Unique evidence ID
            source: Source name (e.g., "github", "web")
            title: Evidence title
            snippet: Evidence content
            url: Source URL
            reliability_score: Reliability score (0-1)
            metadata: Additional metadata
            debate_id: Optional debate association

        Returns:
            The evidence ID (may be deduplicated)
        """
        return self._store.save_evidence(
            evidence_id=evidence_id,
            source=source,
            title=title,
            snippet=snippet,
            url=url,
            reliability_score=reliability_score,
            metadata=metadata,
            debate_id=debate_id,
        )

    def mark_used_in_consensus(
        self,
        evidence_id: str,
        debate_id: str,
    ) -> None:
        """
        Mark evidence as used in consensus (boost reliability).

        Args:
            evidence_id: The evidence ID
            debate_id: The debate ID where consensus was reached
        """
        self._store.mark_used_in_consensus(debate_id, evidence_id)

    async def update_reliability_from_km(
        self,
        evidence_id: str,
        km_validation: Dict[str, Any],
    ) -> None:
        """
        Update evidence reliability based on KM validation feedback.

        This is the reverse flow: KM validation improves evidence scores.

        Args:
            evidence_id: The evidence ID
            km_validation: Validation data from Knowledge Mound
        """
        # Strip prefix if present
        if evidence_id.startswith(self.ID_PREFIX):
            evidence_id = evidence_id[len(self.ID_PREFIX):]

        # Get current evidence
        evidence = self._store.get_evidence(evidence_id)
        if not evidence:
            logger.warning(f"Evidence not found for KM validation: {evidence_id}")
            return

        # Calculate new reliability based on KM feedback
        current_reliability = evidence.get("reliability_score", 0.5)
        km_confidence = km_validation.get("confidence", 0.5)
        validation_count = km_validation.get("validation_count", 1)

        # Weighted average: more validations = more weight on KM confidence
        weight = min(0.5, validation_count * 0.1)  # Max 50% weight
        new_reliability = (
            current_reliability * (1 - weight) +
            km_confidence * weight
        )

        # Update the evidence
        self._store.update_evidence(
            evidence_id,
            reliability_score=new_reliability,
        )

        logger.info(
            f"Updated evidence reliability from KM: {evidence_id} "
            f"{current_reliability:.2f} -> {new_reliability:.2f}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the evidence store."""
        return self._store.get_stats()

    def get_debate_evidence(
        self,
        debate_id: str,
        min_relevance: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Get all evidence associated with a debate.

        Args:
            debate_id: The debate ID
            min_relevance: Minimum relevance score

        Returns:
            List of evidence items for the debate
        """
        return self._store.get_debate_evidence(debate_id, min_relevance)


__all__ = [
    "EvidenceAdapter",
    "EvidenceSearchResult",
]
