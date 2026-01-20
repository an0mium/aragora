"""
Fact Registry - Enhanced fact management with staleness tracking.

Provides centralized management of facts with:
- Vertical-specific staleness tracking and confidence decay
- Multi-backend vector store integration
- Automatic reverification triggers
- Cross-vertical fact relationships

Usage:
    from aragora.knowledge.mound.fact_registry import FactRegistry

    registry = FactRegistry(fact_store, vector_store)
    await registry.initialize()

    # Register a new fact
    fact = await registry.register(
        statement="API keys should never be committed to version control",
        vertical="software",
        category="best_practice",
        confidence=0.9,
    )

    # Query facts
    results = await registry.query("security best practices", verticals=["software"])
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from aragora.knowledge.types import ValidationStatus
from aragora.knowledge.mound_core import ProvenanceType
from aragora.knowledge.mound.verticals import VerticalRegistry

logger = logging.getLogger(__name__)


@dataclass
class RegisteredFact:
    """
    A fact in the registry with full lifecycle tracking.

    Extends the base Fact type with:
    - Staleness tracking and confidence decay
    - Vertical classification
    - Provenance chain
    - Verification history
    """

    id: str
    statement: str

    # Classification
    vertical: str = "general"
    category: str = "general"

    # Confidence with decay
    base_confidence: float = 0.5
    verification_date: datetime = field(default_factory=datetime.now)
    decay_rate: float = 0.02  # Per day, varies by vertical

    # Validation
    validation_status: ValidationStatus = ValidationStatus.UNVERIFIED
    verification_count: int = 0
    contradiction_count: int = 0

    # Provenance
    source_type: ProvenanceType = ProvenanceType.DOCUMENT
    source_ids: list[str] = field(default_factory=list)
    contributing_agents: list[str] = field(default_factory=list)

    # Context
    workspace_id: str = ""
    topics: list[str] = field(default_factory=list)

    # Relationships
    supports: list[str] = field(default_factory=list)  # Fact IDs this supports
    contradicts: list[str] = field(default_factory=list)  # Fact IDs this contradicts
    derived_from: list[str] = field(default_factory=list)  # Source fact IDs

    # Embedding (cached)
    embedding: Optional[list[float]] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    superseded_at: Optional[datetime] = None
    superseded_by: Optional[str] = None

    @property
    def staleness_days(self) -> float:
        """Days since last verification."""
        delta = datetime.now() - self.verification_date
        return delta.total_seconds() / 86400

    @property
    def current_confidence(self) -> float:
        """
        Confidence adjusted for staleness.

        Applies exponential decay based on time since last verification.
        Verification count provides a boost, contradiction count a penalty.
        """
        # Calculate decay
        decay = min(0.9, self.staleness_days * self.decay_rate)
        adjusted = self.base_confidence * (1.0 - decay)

        # Verification count boosts confidence
        verification_bonus = min(0.2, self.verification_count * 0.02)

        # Contradiction count reduces confidence
        contradiction_penalty = min(0.3, self.contradiction_count * 0.05)

        return max(0.0, min(1.0, adjusted + verification_bonus - contradiction_penalty))

    @property
    def needs_reverification(self) -> bool:
        """Check if fact is stale enough to need reverification."""
        return self.current_confidence < 0.5 * self.base_confidence

    @property
    def is_superseded(self) -> bool:
        """Check if this fact has been superseded by another."""
        return self.superseded_at is not None

    def refresh(self, new_confidence: Optional[float] = None) -> None:
        """Mark fact as freshly verified."""
        self.verification_date = datetime.now()
        self.verification_count += 1
        if new_confidence is not None:
            self.base_confidence = new_confidence
        self.updated_at = datetime.now()

    def add_contradiction(self, fact_id: str) -> None:
        """Record a contradicting fact."""
        if fact_id not in self.contradicts:
            self.contradicts.append(fact_id)
            self.contradiction_count += 1
            self.updated_at = datetime.now()

    def supersede(self, new_fact_id: str) -> None:
        """Mark this fact as superseded by another."""
        self.superseded_at = datetime.now()
        self.superseded_by = new_fact_id
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "statement": self.statement,
            "vertical": self.vertical,
            "category": self.category,
            "base_confidence": self.base_confidence,
            "current_confidence": self.current_confidence,
            "verification_date": self.verification_date.isoformat(),
            "decay_rate": self.decay_rate,
            "validation_status": self.validation_status.value,
            "verification_count": self.verification_count,
            "contradiction_count": self.contradiction_count,
            "source_type": self.source_type.value,
            "source_ids": self.source_ids,
            "contributing_agents": self.contributing_agents,
            "workspace_id": self.workspace_id,
            "topics": self.topics,
            "supports": self.supports,
            "contradicts": self.contradicts,
            "derived_from": self.derived_from,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "superseded_at": self.superseded_at.isoformat() if self.superseded_at else None,
            "superseded_by": self.superseded_by,
            "needs_reverification": self.needs_reverification,
            "is_superseded": self.is_superseded,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegisteredFact:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            statement=data["statement"],
            vertical=data.get("vertical", "general"),
            category=data.get("category", "general"),
            base_confidence=data.get("base_confidence", 0.5),
            verification_date=(
                datetime.fromisoformat(data["verification_date"])
                if data.get("verification_date")
                else datetime.now()
            ),
            decay_rate=data.get("decay_rate", 0.02),
            validation_status=(
                ValidationStatus(data["validation_status"])
                if data.get("validation_status")
                else ValidationStatus.UNVERIFIED
            ),
            verification_count=data.get("verification_count", 0),
            contradiction_count=data.get("contradiction_count", 0),
            source_type=(
                ProvenanceType(data["source_type"])
                if data.get("source_type")
                else ProvenanceType.DOCUMENT
            ),
            source_ids=data.get("source_ids", []),
            contributing_agents=data.get("contributing_agents", []),
            workspace_id=data.get("workspace_id", ""),
            topics=data.get("topics", []),
            supports=data.get("supports", []),
            contradicts=data.get("contradicts", []),
            derived_from=data.get("derived_from", []),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.now()
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if data.get("updated_at")
                else datetime.now()
            ),
            superseded_at=(
                datetime.fromisoformat(data["superseded_at"]) if data.get("superseded_at") else None
            ),
            superseded_by=data.get("superseded_by"),
        )


class FactRegistry:
    """
    Central registry for facts with multi-vertical support.

    Integrates with:
    - Vector stores for semantic search
    - Vertical modules for decay rates
    - Knowledge Mound for storage

    Key features:
    - Vertical-specific confidence decay rates
    - Near-duplicate detection via embeddings
    - Automatic reverification tracking
    - Cross-vertical fact relationships
    """

    def __init__(
        self,
        vector_store: Optional[Any] = None,  # BaseVectorStore
        embedding_service: Optional[Any] = None,  # For generating embeddings
    ):
        """
        Initialize fact registry.

        Args:
            vector_store: Vector store for semantic search
            embedding_service: Service for generating embeddings
        """
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._facts: dict[str, RegisteredFact] = {}  # In-memory cache
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the registry and connect to stores."""
        if self._vector_store:
            await self._vector_store.connect()
        self._initialized = True

    async def register(
        self,
        statement: str,
        vertical: str = "general",
        category: str = "general",
        confidence: float = 0.5,
        source_type: ProvenanceType = ProvenanceType.DOCUMENT,
        source_ids: Optional[list[str]] = None,
        workspace_id: str = "",
        topics: Optional[list[str]] = None,
        check_duplicates: bool = True,
        **kwargs,
    ) -> RegisteredFact:
        """
        Register a new fact with vertical classification.

        Args:
            statement: The fact statement
            vertical: Vertical classification
            category: Category within vertical
            confidence: Initial confidence score
            source_type: Type of source
            source_ids: IDs of sources
            workspace_id: Workspace scope
            topics: Related topics
            check_duplicates: Whether to check for near-duplicates

        Returns:
            The registered fact
        """
        # Get vertical-specific decay rate
        decay_rate = self._get_decay_rate(vertical, category)

        # Generate embedding for duplicate detection and search
        embedding = None
        if self._embedding_service and check_duplicates:
            embedding = await self._get_embedding(statement)

            # Check for near-duplicates
            if self._vector_store and embedding:
                similar = await self._vector_store.search(
                    embedding=embedding,
                    limit=3,
                    namespace=vertical,
                    min_score=0.95,
                )

                for match in similar:
                    # Near-duplicate found - update existing instead
                    existing_id = match.id
                    existing = self._facts.get(existing_id)
                    if existing:
                        existing.refresh()
                        existing.verification_count += 1
                        logger.debug(
                            f"Updated existing fact {existing_id} instead of creating duplicate"
                        )
                        return existing

        # Create new fact
        fact = RegisteredFact(
            id=f"fact_{uuid.uuid4().hex[:12]}",
            statement=statement,
            vertical=vertical,
            category=category,
            base_confidence=confidence,
            decay_rate=decay_rate,
            source_type=source_type,
            source_ids=source_ids or [],
            workspace_id=workspace_id,
            topics=topics or [],
            embedding=embedding,
        )

        # Store in memory cache
        self._facts[fact.id] = fact

        # Store in vector store for semantic search
        if self._vector_store and embedding:
            await self._vector_store.upsert(
                id=fact.id,
                embedding=embedding,
                content=fact.statement,
                metadata={
                    "vertical": fact.vertical,
                    "category": fact.category,
                    "workspace_id": fact.workspace_id,
                    "confidence": fact.base_confidence,
                    "created_at": fact.created_at.isoformat(),
                },
                namespace=fact.vertical,
            )

        logger.debug(f"Registered fact {fact.id} in vertical {vertical}")
        return fact

    async def query(
        self,
        query: str,
        verticals: Optional[list[str]] = None,
        workspace_id: Optional[str] = None,
        min_confidence: float = 0.3,
        include_stale: bool = False,
        include_superseded: bool = False,
        limit: int = 10,
    ) -> list[RegisteredFact]:
        """
        Query facts with vertical filtering.

        Args:
            query: Search query
            verticals: Filter by verticals (None = all)
            workspace_id: Filter by workspace
            min_confidence: Minimum current confidence
            include_stale: Include facts needing reverification
            include_superseded: Include superseded facts
            limit: Maximum results

        Returns:
            List of matching facts sorted by confidence
        """
        if not self._vector_store:
            # Fall back to in-memory search
            return self._search_memory(
                query,
                verticals,
                workspace_id,
                min_confidence,
                include_stale,
                include_superseded,
                limit,
            )

        # Generate embedding for semantic search
        embedding = await self._get_embedding(query)
        if not embedding:
            return self._search_memory(
                query,
                verticals,
                workspace_id,
                min_confidence,
                include_stale,
                include_superseded,
                limit,
            )

        # Query each requested vertical namespace
        namespaces = verticals or ["general"]
        all_results = []

        for namespace in namespaces:
            filters = {}
            if workspace_id:
                filters["workspace_id"] = workspace_id

            results = await self._vector_store.search(
                embedding=embedding,
                limit=limit,
                namespace=namespace,
                filters=filters if filters else None,
            )
            all_results.extend(results)

        # Load full facts and filter
        facts = []
        for result in all_results:
            fact = self._facts.get(result.id)
            if not fact:
                # Try to reconstruct from search result
                fact = self._result_to_fact(result)
                if fact:
                    self._facts[fact.id] = fact

            if fact:
                # Apply filters
                if fact.current_confidence < min_confidence:
                    continue
                if not include_stale and fact.needs_reverification:
                    continue
                if not include_superseded and fact.is_superseded:
                    continue

                facts.append(fact)

        # Sort by current confidence
        facts.sort(key=lambda f: f.current_confidence, reverse=True)
        return facts[:limit]

    async def get_fact(self, fact_id: str) -> Optional[RegisteredFact]:
        """Get a fact by ID."""
        return self._facts.get(fact_id)

    async def refresh_fact(
        self,
        fact_id: str,
        new_confidence: Optional[float] = None,
    ) -> bool:
        """
        Refresh a fact (mark as verified).

        Args:
            fact_id: Fact ID to refresh
            new_confidence: Optional new confidence score

        Returns:
            True if fact was found and refreshed
        """
        fact = self._facts.get(fact_id)
        if fact:
            fact.refresh(new_confidence)
            return True
        return False

    async def add_contradiction(
        self,
        fact_id: str,
        contradicting_fact_id: str,
    ) -> bool:
        """
        Record a contradiction between facts.

        Args:
            fact_id: Fact being contradicted
            contradicting_fact_id: Fact that contradicts

        Returns:
            True if recorded successfully
        """
        fact = self._facts.get(fact_id)
        if fact:
            fact.add_contradiction(contradicting_fact_id)

            # Also record the reverse relationship
            contradicting = self._facts.get(contradicting_fact_id)
            if contradicting:
                contradicting.add_contradiction(fact_id)

            return True
        return False

    async def supersede_fact(
        self,
        old_fact_id: str,
        new_fact_id: str,
    ) -> bool:
        """
        Mark a fact as superseded by another.

        Args:
            old_fact_id: Fact being superseded
            new_fact_id: New fact that supersedes

        Returns:
            True if superseded successfully
        """
        old_fact = self._facts.get(old_fact_id)
        if old_fact:
            old_fact.supersede(new_fact_id)

            # Record derivation on new fact
            new_fact = self._facts.get(new_fact_id)
            if new_fact:
                if old_fact_id not in new_fact.derived_from:
                    new_fact.derived_from.append(old_fact_id)

            return True
        return False

    async def get_stale_facts(
        self,
        workspace_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[RegisteredFact]:
        """Get facts that need reverification."""
        stale = []
        for fact in self._facts.values():
            if workspace_id and fact.workspace_id != workspace_id:
                continue
            if fact.needs_reverification and not fact.is_superseded:
                stale.append(fact)

        # Sort by staleness (most stale first)
        stale.sort(key=lambda f: f.staleness_days, reverse=True)
        return stale[:limit]

    async def get_stats(
        self,
        workspace_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get registry statistics."""
        facts = list(self._facts.values())
        if workspace_id:
            facts = [f for f in facts if f.workspace_id == workspace_id]

        # Count by vertical
        by_vertical: dict[str, int] = {}
        for fact in facts:
            by_vertical[fact.vertical] = by_vertical.get(fact.vertical, 0) + 1

        # Count by status
        stale_count = sum(1 for f in facts if f.needs_reverification)
        superseded_count = sum(1 for f in facts if f.is_superseded)

        # Average confidence
        confidences = [f.current_confidence for f in facts if not f.is_superseded]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return {
            "total_facts": len(facts),
            "by_vertical": by_vertical,
            "stale_count": stale_count,
            "superseded_count": superseded_count,
            "average_confidence": avg_confidence,
        }

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _get_decay_rate(self, vertical: str, category: str) -> float:
        """Get decay rate based on vertical and category."""
        # Try to get from vertical module
        vertical_module = VerticalRegistry.get(vertical)
        if vertical_module:
            return vertical_module.get_decay_rate(category)

        # Default rates by vertical
        default_rates = {
            "software": {
                "vulnerability": 0.05,
                "secret": 0.1,
                "best_practice": 0.01,
                "default": 0.02,
            },
            "legal": {
                "regulation": 0.005,
                "case_law": 0.01,
                "default": 0.015,
            },
            "healthcare": {
                "clinical_guideline": 0.02,
                "drug_interaction": 0.03,
                "default": 0.025,
            },
            "accounting": {
                "regulation": 0.005,
                "tax_rule": 0.02,
                "default": 0.01,
            },
            "general": {
                "default": 0.02,
            },
        }

        vertical_rates = default_rates.get(vertical, default_rates["general"])
        return vertical_rates.get(category, vertical_rates["default"])

    async def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Generate embedding for text."""
        if self._embedding_service:
            try:
                return await self._embedding_service.embed(text)
            except (RuntimeError, ConnectionError, TimeoutError) as e:
                logger.warning(f"Failed to generate embedding: {e}")
            except Exception as e:
                logger.exception(f"Unexpected embedding generation error: {e}")
        return None

    def _search_memory(
        self,
        query: str,
        verticals: Optional[list[str]],
        workspace_id: Optional[str],
        min_confidence: float,
        include_stale: bool,
        include_superseded: bool,
        limit: int,
    ) -> list[RegisteredFact]:
        """Simple in-memory search by keyword matching."""
        query_words = set(query.lower().split())

        results = []
        for fact in self._facts.values():
            # Apply filters
            if verticals and fact.vertical not in verticals:
                continue
            if workspace_id and fact.workspace_id != workspace_id:
                continue
            if fact.current_confidence < min_confidence:
                continue
            if not include_stale and fact.needs_reverification:
                continue
            if not include_superseded and fact.is_superseded:
                continue

            # Score by keyword overlap
            fact_words = set(fact.statement.lower().split())
            overlap = len(query_words & fact_words)

            if overlap > 0:
                results.append((fact, overlap))

        # Sort by overlap and confidence
        results.sort(key=lambda x: (x[1], x[0].current_confidence), reverse=True)
        return [r[0] for r in results[:limit]]

    def _result_to_fact(self, result: Any) -> Optional[RegisteredFact]:
        """Reconstruct fact from search result."""
        try:
            return RegisteredFact(
                id=result.id,
                statement=result.content,
                vertical=result.metadata.get("vertical", "general"),
                category=result.metadata.get("category", "general"),
                base_confidence=result.metadata.get("confidence", 0.5),
                workspace_id=result.metadata.get("workspace_id", ""),
                embedding=result.embedding,
            )
        except (KeyError, AttributeError, ValueError) as e:
            logger.debug(f"Could not convert search result to fact: {e}")
            return None
