"""
Base Connector - Abstract interface for evidence sources.

All connectors inherit from BaseConnector and implement:
- search(): Find relevant evidence for a query
- fetch(): Retrieve specific evidence by ID
- record(): Store evidence in provenance chain
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import hashlib

from aragora.reasoning.provenance import (
    ProvenanceManager,
    ProvenanceRecord,
    SourceType,
    TransformationType,
)


@dataclass
class Evidence:
    """
    A piece of evidence from an external source.

    Contains the content and metadata needed for
    provenance tracking and reliability scoring.
    """

    id: str
    source_type: SourceType
    source_id: str  # URL, file path, issue number, etc.
    content: str
    title: str = ""

    # Metadata
    created_at: Optional[str] = None
    author: Optional[str] = None
    url: Optional[str] = None

    # Reliability indicators
    confidence: float = 0.5  # Base confidence in source
    freshness: float = 1.0   # How recent (1.0 = current, decays over time)
    authority: float = 0.5   # Source authority (0-1)

    # Additional context
    metadata: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """Compute content hash."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]

    @property
    def reliability_score(self) -> float:
        """Combined reliability score."""
        # Weighted combination of factors
        return (
            0.4 * self.confidence +
            0.3 * self.freshness +
            0.3 * self.authority
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "content": self.content,
            "title": self.title,
            "created_at": self.created_at,
            "author": self.author,
            "url": self.url,
            "confidence": self.confidence,
            "freshness": self.freshness,
            "authority": self.authority,
            "reliability_score": self.reliability_score,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Evidence":
        """Reconstruct Evidence from dictionary (for cache deserialization)."""
        from aragora.reasoning.provenance import SourceType

        # Handle source_type as either string or SourceType enum
        source_type = data.get("source_type", "web_search")
        if isinstance(source_type, str):
            try:
                source_type = SourceType(source_type)
            except ValueError:
                source_type = SourceType.WEB_SEARCH

        return cls(
            id=data["id"],
            source_type=source_type,
            source_id=data["source_id"],
            content=data["content"],
            title=data.get("title", ""),
            created_at=data.get("created_at"),
            author=data.get("author"),
            url=data.get("url"),
            confidence=data.get("confidence", 0.5),
            freshness=data.get("freshness", 1.0),
            authority=data.get("authority", 0.5),
            metadata=data.get("metadata", {}),
        )


class BaseConnector(ABC):
    """
    Abstract base class for evidence connectors.

    Provides common functionality for searching, fetching,
    and recording evidence with provenance tracking.
    """

    def __init__(
        self,
        provenance: Optional[ProvenanceManager] = None,
        default_confidence: float = 0.5,
        max_cache_entries: int = 500,
    ):
        self.provenance = provenance
        self.default_confidence = default_confidence
        self._cache: OrderedDict[str, Evidence] = OrderedDict()
        self._max_cache_entries = max_cache_entries

    def _cache_put(self, evidence_id: str, evidence: Evidence) -> None:
        """Add to cache with LRU eviction."""
        if evidence_id in self._cache:
            self._cache.move_to_end(evidence_id)
        self._cache[evidence_id] = evidence
        while len(self._cache) > self._max_cache_entries:
            self._cache.popitem(last=False)

    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """The source type for this connector."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this connector."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> list[Evidence]:
        """
        Search for evidence matching a query.

        Args:
            query: Search query string
            limit: Maximum results to return
            **kwargs: Connector-specific options

        Returns:
            List of Evidence objects
        """
        pass

    @abstractmethod
    async def fetch(self, evidence_id: str) -> Optional[Evidence]:
        """
        Fetch a specific piece of evidence by ID.

        Args:
            evidence_id: Unique identifier for the evidence

        Returns:
            Evidence object or None if not found
        """
        pass

    def record_evidence(
        self,
        evidence: Evidence,
        claim_id: Optional[str] = None,
        support_type: str = "supports",
    ) -> Optional[ProvenanceRecord]:
        """
        Record evidence in the provenance chain.

        Args:
            evidence: The evidence to record
            claim_id: Optional claim to link as citation
            support_type: How evidence relates to claim

        Returns:
            ProvenanceRecord or None if no provenance manager
        """
        if not self.provenance:
            return None

        # Record in chain
        record = self.provenance.record_evidence(
            content=evidence.content,
            source_type=evidence.source_type,
            source_id=evidence.source_id,
            confidence=evidence.reliability_score,
            metadata={
                "title": evidence.title,
                "author": evidence.author,
                "url": evidence.url,
                "content_hash": evidence.content_hash,
            },
        )

        # Create citation if claim_id provided
        if claim_id:
            self.provenance.cite_evidence(
                claim_id=claim_id,
                evidence_id=record.id,
                relevance=evidence.reliability_score,
                support_type=support_type,
                citation_text=evidence.content[:200],
            )

        return record

    async def search_and_record(
        self,
        query: str,
        claim_id: Optional[str] = None,
        limit: int = 5,
        **kwargs,
    ) -> list[tuple[Evidence, Optional[ProvenanceRecord]]]:
        """
        Search for evidence and record all results.

        Returns:
            List of (Evidence, ProvenanceRecord) tuples
        """
        results = await self.search(query, limit=limit, **kwargs)

        recorded = []
        for evidence in results:
            record = self.record_evidence(evidence, claim_id)
            recorded.append((evidence, record))

        return recorded

    def calculate_freshness(self, created_at: str) -> float:
        """
        Calculate freshness score based on age.

        Recent content (< 7 days) = 1.0
        Decays exponentially to 0.1 for old content (> 1 year)
        """
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            age_days = (datetime.now(created.tzinfo) - created).days

            if age_days < 7:
                return 1.0
            elif age_days < 30:
                return 0.9
            elif age_days < 90:
                return 0.7
            elif age_days < 365:
                return 0.5
            else:
                return 0.3
        except (ValueError, TypeError, AttributeError):
            return 0.5  # Unknown age

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source={self.source_type.value})"


# Alias for backward compatibility
Connector = BaseConnector
