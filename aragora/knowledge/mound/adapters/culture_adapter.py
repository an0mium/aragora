"""
CultureAdapter - Bridges Culture Accumulator to the Knowledge Mound.

This adapter enables bidirectional integration between the culture system
and the Knowledge Mound:

- Data flow IN: Culture patterns stored in KM for persistence
- Data flow OUT: Historical patterns retrieved to inform debates
- Reverse flow: KM culture history informs new workspace initialization

The adapter provides:
- Culture pattern storage after debate observations
- Pattern retrieval for debate protocol configuration
- Cross-workspace culture pattern promotion
- Culture profile generation from stored patterns

ID Prefixes:
- cp_: Culture pattern records
- cpr_: Culture profile records
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import (
        CulturePattern,
        CulturePatternType,
        CultureProfile,
    )

logger = logging.getLogger(__name__)


@dataclass
class CultureSearchResult:
    """Wrapper for culture search results with adapter metadata."""

    pattern: Dict[str, Any]
    relevance_score: float = 0.0
    workspace_id: str = ""
    observation_count: int = 0


@dataclass
class StoredCulturePattern:
    """Represents a culture pattern stored in the KM."""

    id: str
    workspace_id: str
    pattern_type: str
    pattern_key: str
    pattern_value: Dict[str, Any]
    observation_count: int
    confidence: float
    first_observed: str
    last_observed: str
    contributing_debates: List[str]
    metadata: Dict[str, Any]


class CultureAdapter:
    """
    Adapter that bridges CultureAccumulator to the Knowledge Mound.

    Provides methods for persisting and retrieving culture patterns:
    - store_pattern: Store a culture pattern after observation
    - load_patterns: Load patterns for a workspace
    - get_dominant_pattern: Get the dominant pattern for a type
    - promote_to_organization: Promote workspace pattern to org level

    Usage:
        from aragora.knowledge.mound.adapters import CultureAdapter

        adapter = CultureAdapter()

        # After debate observation, store pattern
        adapter.store_pattern(pattern)

        # For new debate, load workspace patterns
        patterns = adapter.load_patterns("workspace_001")
    """

    PATTERN_PREFIX = "cp_"
    PROFILE_PREFIX = "cpr_"

    # Confidence thresholds
    MIN_OBSERVATIONS_FOR_STORAGE = 3  # Minimum observations to persist
    MIN_CONFIDENCE_FOR_PROMOTION = 0.8  # Confidence needed for org promotion

    def __init__(self, mound: Optional[Any] = None):
        """Initialize the culture adapter.

        Args:
            mound: Optional KnowledgeMound instance. If not provided,
                   will attempt to get singleton on first use.
        """
        self._mound = mound
        self._pattern_cache: Dict[str, Dict[str, StoredCulturePattern]] = {}

    def _get_mound(self) -> Optional[Any]:
        """Get the Knowledge Mound instance."""
        if self._mound is not None:
            return self._mound

        try:
            from aragora.knowledge.mound import get_knowledge_mound
            self._mound = get_knowledge_mound()
            return self._mound
        except (ImportError, Exception) as e:
            logger.debug(f"Could not get KnowledgeMound: {e}")
            return None

    def store_pattern(
        self,
        pattern: "CulturePattern",
        workspace_id: Optional[str] = None,
    ) -> Optional[str]:
        """Store a culture pattern in the Knowledge Mound.

        Args:
            pattern: CulturePattern to store
            workspace_id: Override workspace (defaults to pattern.workspace_id)

        Returns:
            Node ID of stored pattern, or None if storage failed
        """
        if pattern.observation_count < self.MIN_OBSERVATIONS_FOR_STORAGE:
            logger.debug(
                f"Pattern has too few observations ({pattern.observation_count}), skipping"
            )
            return None

        mound = self._get_mound()
        if not mound:
            logger.debug("KnowledgeMound not available for pattern storage")
            return None

        ws_id = workspace_id or pattern.workspace_id
        pattern_id = f"{self.PATTERN_PREFIX}{pattern.id}"

        try:
            import asyncio
            from aragora.knowledge.mound.types import IngestionRequest, KnowledgeSource

            # Build content from pattern
            content = self._pattern_to_content(pattern)

            request = IngestionRequest(
                content=content,
                workspace_id=ws_id,
                source_type=KnowledgeSource.INSIGHT,  # Culture is derived insight
                node_type="culture_pattern",
                confidence=pattern.confidence,
                tier="slow",  # Culture patterns are relatively stable
                metadata={
                    "pattern_id": pattern.id,
                    "pattern_type": pattern.pattern_type.value if hasattr(pattern.pattern_type, 'value') else str(pattern.pattern_type),
                    "pattern_key": pattern.pattern_key,
                    "pattern_value": pattern.pattern_value,
                    "observation_count": pattern.observation_count,
                    "first_observed": pattern.first_observed_at.isoformat() if hasattr(pattern.first_observed_at, 'isoformat') else str(pattern.first_observed_at),
                    "last_observed": pattern.last_observed_at.isoformat() if hasattr(pattern.last_observed_at, 'isoformat') else str(pattern.last_observed_at),
                    "contributing_debates": pattern.contributing_debates[:20],  # Limit list size
                    "adapter": "culture",
                },
            )

            async def do_store():
                result = await mound.store(request)
                return result.node_id

            # Execute async
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't block, return None
                    asyncio.create_task(do_store())
                    return pattern_id  # Return expected ID
                else:
                    return loop.run_until_complete(do_store())
            except RuntimeError:
                return asyncio.run(do_store())

        except ImportError as e:
            logger.debug(f"Import error storing pattern: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to store culture pattern: {e}")
            return None

    def _pattern_to_content(self, pattern: "CulturePattern") -> str:
        """Convert a CulturePattern to searchable content string."""
        pattern_type = pattern.pattern_type.value if hasattr(pattern.pattern_type, 'value') else str(pattern.pattern_type)

        parts = [
            f"Culture Pattern: {pattern_type}",
            f"Key: {pattern.pattern_key}",
            f"Value: {pattern.pattern_value}",
            f"Observations: {pattern.observation_count}",
            f"Confidence: {pattern.confidence:.2f}",
        ]

        if pattern.metadata:
            if "domain" in pattern.metadata:
                parts.append(f"Domain: {pattern.metadata['domain']}")
            if "description" in pattern.metadata:
                parts.append(f"Description: {pattern.metadata['description']}")

        return " | ".join(parts)

    def load_patterns(
        self,
        workspace_id: str,
        pattern_types: Optional[List["CulturePatternType"]] = None,
        min_confidence: float = 0.5,
        limit: int = 50,
    ) -> List[StoredCulturePattern]:
        """Load culture patterns from the Knowledge Mound.

        Args:
            workspace_id: Workspace to load patterns for
            pattern_types: Optional filter by pattern type
            min_confidence: Minimum confidence threshold
            limit: Maximum patterns to return

        Returns:
            List of StoredCulturePattern
        """
        mound = self._get_mound()
        if not mound:
            return []

        try:
            import asyncio

            async def do_query():
                # Build query based on pattern types
                if pattern_types:
                    type_strings = [
                        pt.value if hasattr(pt, 'value') else str(pt)
                        for pt in pattern_types
                    ]
                    query = f"culture pattern {' '.join(type_strings)}"
                else:
                    query = "culture pattern"

                results = await mound.query(
                    query=query,
                    sources=("local",),
                    filters={"workspace_id": workspace_id, "node_type": "culture_pattern"},
                    limit=limit,
                )

                return [
                    self._item_to_stored_pattern(item)
                    for item in results.items
                    if item.confidence >= min_confidence
                ]

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Return cached if available
                    cache_key = f"{workspace_id}:{pattern_types}:{min_confidence}"
                    if cache_key in self._pattern_cache:
                        return list(self._pattern_cache[cache_key].values())
                    return []
                else:
                    return loop.run_until_complete(do_query())
            except RuntimeError:
                return asyncio.run(do_query())

        except Exception as e:
            logger.debug(f"Failed to load culture patterns: {e}")
            return []

    def _item_to_stored_pattern(self, item: Any) -> StoredCulturePattern:
        """Convert a KnowledgeItem to StoredCulturePattern."""
        metadata = item.metadata or {}

        return StoredCulturePattern(
            id=item.id,
            workspace_id=item.workspace_id if hasattr(item, 'workspace_id') else "",
            pattern_type=metadata.get("pattern_type", "unknown"),
            pattern_key=metadata.get("pattern_key", ""),
            pattern_value=metadata.get("pattern_value", {}),
            observation_count=metadata.get("observation_count", 0),
            confidence=item.confidence if hasattr(item, 'confidence') else 0.5,
            first_observed=metadata.get("first_observed", ""),
            last_observed=metadata.get("last_observed", ""),
            contributing_debates=metadata.get("contributing_debates", []),
            metadata=metadata,
        )

    def get_dominant_pattern(
        self,
        workspace_id: str,
        pattern_type: "CulturePatternType",
    ) -> Optional[StoredCulturePattern]:
        """Get the dominant pattern for a specific type.

        Args:
            workspace_id: Workspace to query
            pattern_type: Type of pattern to retrieve

        Returns:
            The highest-confidence pattern of the given type, or None
        """
        patterns = self.load_patterns(
            workspace_id=workspace_id,
            pattern_types=[pattern_type],
            min_confidence=0.0,
            limit=10,
        )

        if not patterns:
            return None

        # Return highest confidence pattern
        return max(patterns, key=lambda p: (p.confidence, p.observation_count))

    def get_protocol_hints(
        self,
        workspace_id: str,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get protocol configuration hints from culture patterns.

        Args:
            workspace_id: Workspace to query
            domain: Optional domain to filter patterns

        Returns:
            Dict of protocol hints derived from patterns
        """
        hints: Dict[str, Any] = {}

        patterns = self.load_patterns(
            workspace_id=workspace_id,
            min_confidence=0.6,
            limit=20,
        )

        for pattern in patterns:
            # Map pattern types to protocol hints
            if pattern.pattern_type == "decision_style":
                hints["recommended_consensus"] = pattern.pattern_value.get(
                    "preferred_consensus", pattern.pattern_key
                )

            elif pattern.pattern_type == "risk_tolerance":
                risk_level = pattern.pattern_value.get("level", pattern.pattern_key)
                if risk_level == "conservative":
                    hints["extra_critique_rounds"] = hints.get("extra_critique_rounds", 0) + 1
                elif risk_level == "aggressive":
                    hints["early_consensus_threshold"] = 0.7

            elif pattern.pattern_type == "debate_dynamics":
                avg_rounds = pattern.pattern_value.get("avg_rounds_to_consensus", 0)
                if avg_rounds > 5:
                    hints["expected_long_debate"] = True
                elif avg_rounds < 2:
                    hints["quick_consensus_likely"] = True

            # Filter by domain if specified
            if domain and pattern.metadata.get("domain") == domain:
                if pattern.pattern_type == "domain_expertise":
                    if "domain_hints" not in hints:
                        hints["domain_hints"] = []
                    hints["domain_hints"].append({
                        "key": pattern.pattern_key,
                        "value": pattern.pattern_value,
                        "confidence": pattern.confidence,
                    })

        return hints

    def promote_to_organization(
        self,
        pattern: StoredCulturePattern,
        org_id: str,
        promoted_by: str = "system",
    ) -> Optional[str]:
        """Promote a workspace pattern to organization level.

        Args:
            pattern: Pattern to promote
            org_id: Organization ID
            promoted_by: User/system that initiated promotion

        Returns:
            Node ID of promoted pattern, or None if failed
        """
        if pattern.confidence < self.MIN_CONFIDENCE_FOR_PROMOTION:
            logger.debug(
                f"Pattern confidence {pattern.confidence:.2f} below promotion threshold"
            )
            return None

        mound = self._get_mound()
        if not mound:
            return None

        try:
            import asyncio
            from aragora.knowledge.mound.types import IngestionRequest, KnowledgeSource

            # Build promoted content
            content = f"Organization Culture Pattern: {pattern.pattern_type} | "
            content += f"Key: {pattern.pattern_key} | Value: {pattern.pattern_value}"

            request = IngestionRequest(
                content=content,
                workspace_id=f"org_{org_id}",  # Org-level workspace
                source_type=KnowledgeSource.INSIGHT,
                node_type="org_culture_pattern",
                confidence=pattern.confidence,
                tier="glacial",  # Org patterns are very stable
                metadata={
                    "source_workspace": pattern.workspace_id,
                    "source_pattern_id": pattern.id,
                    "pattern_type": pattern.pattern_type,
                    "pattern_key": pattern.pattern_key,
                    "pattern_value": pattern.pattern_value,
                    "promoted_by": promoted_by,
                    "promoted_at": datetime.now().isoformat(),
                    "observation_count": pattern.observation_count,
                    "adapter": "culture",
                },
            )

            async def do_store():
                result = await mound.store(request)
                return result.node_id

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(do_store())
                    return f"org_{pattern.id}"
                else:
                    return loop.run_until_complete(do_store())
            except RuntimeError:
                return asyncio.run(do_store())

        except Exception as e:
            logger.warning(f"Failed to promote pattern to organization: {e}")
            return None

    def sync_to_mound(
        self,
        patterns: List["CulturePattern"],
        workspace_id: str,
    ) -> int:
        """Sync multiple patterns to the Knowledge Mound.

        Args:
            patterns: List of patterns to sync
            workspace_id: Target workspace

        Returns:
            Number of patterns successfully stored
        """
        stored_count = 0

        for pattern in patterns:
            if self.store_pattern(pattern, workspace_id):
                stored_count += 1

        logger.info(
            f"Synced {stored_count}/{len(patterns)} culture patterns to KM "
            f"for workspace {workspace_id}"
        )
        return stored_count

    def load_from_mound(
        self,
        workspace_id: str,
    ) -> "CultureProfile":
        """Load a full culture profile from the Knowledge Mound.

        Args:
            workspace_id: Workspace to load profile for

        Returns:
            CultureProfile reconstructed from stored patterns
        """
        from aragora.knowledge.mound.types import CulturePatternType, CultureProfile

        patterns = self.load_patterns(workspace_id, limit=100)

        # Group by pattern type
        patterns_by_type: Dict[CulturePatternType, List] = {}
        for pattern in patterns:
            try:
                pt = CulturePatternType(pattern.pattern_type)
            except ValueError:
                continue

            if pt not in patterns_by_type:
                patterns_by_type[pt] = []
            patterns_by_type[pt].append(pattern)

        # Calculate dominant traits
        dominant_traits = {}
        for pt, type_patterns in patterns_by_type.items():
            if type_patterns:
                best = max(type_patterns, key=lambda p: p.confidence)
                dominant_traits[pt.value] = {
                    "key": best.pattern_key,
                    "value": best.pattern_value,
                    "confidence": best.confidence,
                }

        # Build profile (note: patterns field expects CulturePattern objects,
        # but we're returning StoredCulturePattern - caller should convert)
        return CultureProfile(
            workspace_id=workspace_id,
            patterns=patterns_by_type,
            generated_at=datetime.now(),
            total_observations=sum(p.observation_count for p in patterns),
            dominant_traits=dominant_traits,
        )
