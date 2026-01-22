"""
ContinuumAdapter - Bridges ContinuumMemory to the Knowledge Mound.

This adapter enables bidirectional integration between ContinuumMemory's
multi-tier system and the Knowledge Mound:

- Data flow IN: ContinuumMemory entries with importance scores are stored in KM
- Data flow OUT: Similar memories are retrieved for context/grounding
- Reverse flow: KM validation feeds back to memory tier promotions/demotions

The adapter provides:
- Unified search interface (search_by_keyword)
- Bidirectional sync (store to both systems)
- Tier-to-importance mapping
- Cross-reference tracking
- **KM validation → tier adjustment (reverse flow)**
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.memory.continuum import ContinuumMemory, ContinuumMemoryEntry
    from aragora.knowledge.mound.types import KnowledgeItem, IngestionRequest

# Type alias for event callback
EventCallback = Callable[[str, Dict[str, Any]], None]

logger = logging.getLogger(__name__)


@dataclass
class KMValidationResult:
    """Result of Knowledge Mound validation for a memory item.

    This represents feedback from KM analysis that can improve
    ContinuumMemory tier placement and importance scores.
    """

    memory_id: str
    km_confidence: float  # 0.0-1.0 KM's confidence in the memory
    cross_debate_utility: float = 0.0  # How useful across debates (0.0-1.0)
    validation_count: int = 1  # Number of validations/uses
    was_contradicted: bool = False  # If KM found contradicting evidence
    was_supported: bool = False  # If KM found supporting evidence
    recommendation: str = "keep"  # "promote", "demote", "keep", "review"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationSyncResult:
    """Result of batch syncing KM validations to ContinuumMemory."""

    total_processed: int = 0
    promoted: int = 0
    demoted: int = 0
    updated: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)
    duration_ms: int = 0


@dataclass
class ContinuumSearchResult:
    """Wrapper for continuum memory search results with adapter metadata."""

    entry: "ContinuumMemoryEntry"
    relevance_score: float = 0.0
    matched_keywords: List[str] = None

    def __post_init__(self) -> None:
        if self.matched_keywords is None:
            self.matched_keywords = []


class ContinuumAdapter:
    """
    Adapter that bridges ContinuumMemory to the Knowledge Mound.

    Provides methods that the Knowledge Mound expects for federated queries:
    - search_by_keyword: Text-based search across tiers
    - to_knowledge_item: Convert entries to unified format
    - sync_from_mound: Store mound items in continuum memory

    Usage:
        from aragora.memory.continuum import ContinuumMemory
        from aragora.knowledge.mound.adapters import ContinuumAdapter

        continuum = ContinuumMemory()
        adapter = ContinuumAdapter(continuum)

        # Search for memories
        results = adapter.search_by_keyword("type errors", limit=10)

        # Convert to knowledge items
        items = [adapter.to_knowledge_item(r) for r in results]
    """

    def __init__(
        self,
        continuum: "ContinuumMemory",
        enable_dual_write: bool = False,
        event_callback: Optional[EventCallback] = None,
    ):
        """
        Initialize the adapter.

        Args:
            continuum: The ContinuumMemory instance to wrap
            enable_dual_write: If True, writes go to both systems during migration
            event_callback: Optional callback for emitting events (event_type, data)
        """
        self._continuum = continuum
        self._enable_dual_write = enable_dual_write
        self._event_callback = event_callback

    def set_event_callback(self, callback: EventCallback) -> None:
        """Set the event callback for WebSocket notifications."""
        self._event_callback = callback

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event if callback is configured."""
        if self._event_callback:
            try:
                self._event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Failed to emit event {event_type}: {e}")

    def _record_metric(self, operation: str, success: bool, latency: float) -> None:
        """Record Prometheus metric for adapter operation and check SLOs.

        Args:
            operation: Operation name (search, store, sync, semantic_search)
            success: Whether operation succeeded
            latency: Operation latency in seconds
        """
        latency_ms = latency * 1000  # Convert to milliseconds

        try:
            from aragora.observability.metrics.km import (
                record_km_operation,
                record_km_adapter_sync,
            )

            record_km_operation(operation, success, latency)
            if operation in ("store", "sync"):
                record_km_adapter_sync("continuum", "forward", success)
        except ImportError:
            pass  # Metrics not available
        except Exception as e:
            logger.debug(f"Failed to record metric: {e}")

        # Check SLOs and alert on violations
        try:
            from aragora.observability.metrics.slo import check_and_record_slo_with_recovery

            # Map operation to SLO name
            slo_mapping = {
                "search": "adapter_reverse",
                "store": "adapter_forward_sync",
                "sync": "adapter_sync",
                "semantic_search": "adapter_semantic_search",
            }
            slo_name = slo_mapping.get(operation, "adapter_sync")

            passed, message = check_and_record_slo_with_recovery(
                operation=slo_name,
                latency_ms=latency_ms,
                context={
                    "adapter": "continuum",
                    "operation": operation,
                    "success": success,
                },
            )
            if not passed:
                logger.debug(f"Continuum adapter SLO violation: {message}")
        except ImportError:
            pass  # SLO metrics not available
        except Exception as e:
            logger.debug(f"Failed to check SLO: {e}")

    @property
    def continuum(self) -> "ContinuumMemory":
        """Access the underlying ContinuumMemory."""
        return self._continuum

    def search_by_keyword(
        self,
        query: str,
        limit: int = 10,
        tiers: Optional[List[str]] = None,
        min_importance: float = 0.0,
    ) -> List["ContinuumMemoryEntry"]:
        """
        Search continuum memory by keyword query.

        This method wraps ContinuumMemory.retrieve() to provide the interface
        expected by KnowledgeMound._query_continuum().

        Args:
            query: Search query (keywords are OR'd)
            limit: Maximum results to return
            tiers: Optional list of tier names to filter (e.g., ["fast", "medium"])
            min_importance: Minimum importance threshold

        Returns:
            List of ContinuumMemoryEntry objects matching the query
        """
        from aragora.memory.tier_manager import MemoryTier

        # Convert tier names to MemoryTier enums
        tier_enums = None
        if tiers:
            tier_enums = []
            for tier_name in tiers:
                try:
                    tier_enums.append(MemoryTier(tier_name))
                except ValueError:
                    logger.warning(f"Unknown tier: {tier_name}, skipping")

        # Use ContinuumMemory's retrieve method
        entries = self._continuum.retrieve(
            query=query,
            tiers=tier_enums,
            limit=limit,
            min_importance=min_importance,
        )

        return list(entries)

    def get(self, entry_id: str) -> Optional["ContinuumMemoryEntry"]:
        """
        Get a specific entry by ID.

        Args:
            entry_id: The entry ID (may be prefixed with "cm_" from mound)

        Returns:
            ContinuumMemoryEntry or None
        """
        # Strip mound prefix if present
        if entry_id.startswith("cm_"):
            entry_id = entry_id[3:]

        return self._continuum.get(entry_id)

    async def get_async(self, entry_id: str) -> Optional["ContinuumMemoryEntry"]:
        """Async version of get for compatibility."""
        # Strip mound prefix if present
        if entry_id.startswith("cm_"):
            entry_id = entry_id[3:]

        return await self._continuum.get_async(entry_id)

    def to_knowledge_item(self, entry: "ContinuumMemoryEntry") -> "KnowledgeItem":
        """
        Convert a ContinuumMemoryEntry to a KnowledgeItem.

        Args:
            entry: The continuum memory entry

        Returns:
            KnowledgeItem for unified knowledge mound API
        """
        from aragora.knowledge.mound.types import (
            ConfidenceLevel,
            KnowledgeItem,
            KnowledgeSource,
        )

        # Map tier to confidence level
        tier_to_confidence = {
            "fast": ConfidenceLevel.LOW,  # Fast tier is volatile
            "medium": ConfidenceLevel.MEDIUM,
            "slow": ConfidenceLevel.HIGH,
            "glacial": ConfidenceLevel.VERIFIED,  # Glacial is most stable
        }
        confidence = tier_to_confidence.get(entry.tier.value, ConfidenceLevel.MEDIUM)

        # Build metadata
        metadata: Dict[str, Any] = {
            "tier": entry.tier.value,
            "surprise_score": entry.surprise_score,
            "consolidation_score": entry.consolidation_score,
            "update_count": entry.update_count,
            "success_rate": entry.success_rate,
        }
        if entry.red_line:
            metadata["red_line"] = True
            metadata["red_line_reason"] = entry.red_line_reason
        if entry.tags:
            metadata["tags"] = entry.tags
        if entry.cross_references:
            metadata["cross_references"] = entry.cross_references

        return KnowledgeItem(
            id=entry.knowledge_mound_id,  # Uses "cm_" prefix
            content=entry.content,
            source=KnowledgeSource.CONTINUUM,
            source_id=entry.id,
            confidence=confidence,
            created_at=datetime.fromisoformat(entry.created_at),
            updated_at=datetime.fromisoformat(entry.updated_at),
            metadata=metadata,
            importance=entry.importance,
        )

    def from_ingestion_request(
        self,
        request: "IngestionRequest",
        entry_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert an IngestionRequest to ContinuumMemory add() parameters.

        Args:
            request: The ingestion request from Knowledge Mound
            entry_id: Optional ID to use (generates one if not provided)

        Returns:
            Dict of parameters for ContinuumMemory.add()
        """
        import uuid
        from aragora.memory.tier_manager import MemoryTier

        # Map KnowledgeMound tier to ContinuumMemory tier
        tier_mapping = {
            1: MemoryTier.FAST,
            2: MemoryTier.MEDIUM,
            3: MemoryTier.SLOW,
            4: MemoryTier.GLACIAL,
        }
        tier = tier_mapping.get(request.tier, MemoryTier.SLOW)  # type: ignore[call-overload]

        return {
            "id": entry_id or f"mound_{uuid.uuid4().hex[:12]}",
            "content": request.content,
            "tier": tier,
            "importance": request.confidence,
            "metadata": {
                "source_type": request.source_type.value,
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
        content: str,
        importance: float = 0.5,
        tier: str = "slow",
        entry_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store content in continuum memory.

        Args:
            content: The content to store
            importance: Importance score (0-1)
            tier: Tier name ("fast", "medium", "slow", "glacial")
            entry_id: Optional ID (generated if not provided)
            metadata: Optional metadata dict

        Returns:
            The entry ID
        """
        import uuid
        from aragora.memory.tier_manager import MemoryTier

        if entry_id is None:
            entry_id = f"mound_{uuid.uuid4().hex[:12]}"

        tier_enum = MemoryTier(tier)

        self._continuum.add(
            id=entry_id,
            content=content,
            tier=tier_enum,
            importance=importance,
            metadata=metadata or {},
        )

        return entry_id

    def link_to_mound(
        self,
        entry_id: str,
        mound_node_id: str,
    ) -> None:
        """
        Link a continuum entry to a knowledge mound node.

        Creates a cross-reference from the continuum entry to the mound node,
        enabling bidirectional navigation.

        Args:
            entry_id: The continuum entry ID
            mound_node_id: The knowledge mound node ID
        """
        entry = self._continuum.get(entry_id)
        if entry:
            entry.add_cross_reference(mound_node_id)
            # Save the updated entry
            self._continuum.update(
                entry_id,
                metadata=entry.metadata,
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the continuum memory."""
        return self._continuum.get_stats()

    def get_tier_metrics(self) -> Dict[str, Any]:
        """Get per-tier metrics."""
        return self._continuum.get_tier_metrics()

    def search_similar(
        self,
        content: str,
        limit: int = 5,
        min_similarity: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Find similar memory entries for deduplication.

        Args:
            content: Content to find similar entries for
            limit: Maximum results
            min_similarity: Minimum similarity threshold (currently uses keyword match)

        Returns:
            List of similar memory entries as dicts
        """
        import time

        start = time.time()
        success = False

        try:
            # Extract key terms for search (first 10 words)
            words = content.split()[:10]
            query = " ".join(words)

            entries = self.search_by_keyword(query, limit=limit)

            # Convert to dict format for consistency with other adapters
            results = [
                {
                    "id": e.id,
                    "content": e.content,
                    "tier": e.tier.value,
                    "importance": e.importance,
                    "surprise_score": e.surprise_score,
                    "consolidation_score": e.consolidation_score,
                    "update_count": e.update_count,
                    "success_rate": e.success_rate,
                    "created_at": e.created_at,
                    "updated_at": e.updated_at,
                    "metadata": e.metadata,
                }
                for e in entries
            ]

            # Emit dashboard event for reverse flow query
            self._emit_event(
                "km_adapter_reverse_query",
                {
                    "source": "continuum",
                    "query_preview": query[:50] + "..." if len(query) > 50 else query,
                    "results_count": len(results),
                    "limit": limit,
                },
            )

            success = True
            return results
        finally:
            self._record_metric("search", success, time.time() - start)

    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.6,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic vector search over memory entries.

        Uses the Knowledge Mound's SemanticStore for embedding-based similarity
        search, falling back to keyword search if embeddings aren't available.

        Args:
            query: The search query
            limit: Maximum results to return
            min_similarity: Minimum similarity threshold (0.0-1.0)
            tenant_id: Optional tenant filter

        Returns:
            List of matching entries with similarity scores
        """
        import time

        start = time.time()
        success = False

        try:
            # Try semantic search first
            try:
                from aragora.knowledge.mound.semantic_store import (
                    SemanticStore,
                )

                # Get or create semantic store
                store = SemanticStore()  # type: ignore[call-arg]

                # Search using embeddings
                results = await store.search_similar(  # type: ignore[call-arg]
                    query=query,
                    tenant_id=tenant_id or "default",
                    limit=limit,
                    min_similarity=min_similarity,
                    source_type="continuum",
                )

                # Enrich results with full memory entries
                enriched = []
                for r in results:
                    # Try to get the full entry from memory
                    entry_id = r.source_id
                    if entry_id.startswith("cm_"):
                        entry_id = entry_id[3:]

                    entry = self._continuum.get(entry_id)
                    if entry:
                        enriched.append(
                            {
                                "id": entry.id,
                                "content": entry.content,
                                "tier": entry.tier.value,
                                "importance": entry.importance,
                                "similarity": r.similarity,
                                "domain": r.domain,
                                "created_at": entry.created_at,
                                "updated_at": entry.updated_at,
                                "metadata": entry.metadata,
                            }
                        )
                    else:
                        # Entry may have been evicted from memory
                        enriched.append(
                            {
                                "id": r.source_id,
                                "similarity": r.similarity,
                                "domain": r.domain,
                                "importance": r.importance,
                                "metadata": r.metadata,
                            }
                        )

                success = True
                logger.debug(f"Semantic search returned {len(enriched)} results for '{query[:50]}'")

                # Emit event
                self._emit_event(
                    "km_adapter_semantic_search",
                    {
                        "source": "continuum",
                        "query_preview": query[:50],
                        "results_count": len(enriched),
                        "search_type": "vector",
                    },
                )

                return enriched

            except ImportError:
                logger.debug("SemanticStore not available, falling back to keyword search")
            except Exception as e:
                logger.debug(f"Semantic search failed, falling back: {e}")

            # Fallback to keyword search
            results = self.search_similar(query, limit=limit, min_similarity=min_similarity)  # type: ignore[assignment]
            success = True
            return results  # type: ignore[return-value]

        finally:
            self._record_metric("semantic_search", success, time.time() - start)

    def store_memory(self, entry: "ContinuumMemoryEntry") -> None:
        """
        Store a memory entry in the Knowledge Mound (forward flow).

        This is called by ContinuumMemory when a high-importance memory
        is added and should be synced to KM for cross-session persistence.

        Args:
            entry: The ContinuumMemoryEntry to store in KM
        """
        # This method is a hook for KM sync. The actual KM storage happens
        # when sync_memory_to_mound is called with a mound instance.
        # For now, we just log the intent - actual sync requires mound reference.
        logger.debug(
            f"Memory marked for KM sync: {entry.id} "
            f"(tier={entry.tier.value}, importance={entry.importance:.2f})"
        )
        # Mark the entry as pending KM sync in metadata
        if not entry.metadata.get("km_sync_pending"):
            entry.metadata["km_sync_pending"] = True
            entry.metadata["km_sync_requested_at"] = datetime.now().isoformat()
            # Update the entry in the store
            self._continuum.update(
                entry.id,
                metadata=entry.metadata,
            )

        # Emit dashboard event for forward sync
        self._emit_event(
            "km_adapter_forward_sync",
            {
                "source": "continuum",
                "memory_id": entry.id,
                "tier": entry.tier.value,
                "importance": entry.importance,
                "content_preview": entry.content[:100] + "..."
                if len(entry.content) > 100
                else entry.content,
            },
        )

    # =========================================================================
    # Reverse Flow Methods (KM → ContinuumMemory)
    # =========================================================================

    async def update_continuum_from_km(
        self,
        memory_id: str,
        km_validation: KMValidationResult,
    ) -> bool:
        """
        Update continuum memory entry based on KM validation feedback.

        This is the reverse flow: KM validation improves continuum memory placement.

        If KM determines an item has high cross-debate utility:
        - Promote to higher tier (FAST→MEDIUM→SLOW→GLACIAL)
        - Increase importance score
        - Mark as KM-validated

        If KM determines an item is low-value or contradicted:
        - Demote to lower tier
        - Decrease importance
        - Mark for review

        Args:
            memory_id: The continuum memory entry ID
            km_validation: Validation data from Knowledge Mound

        Returns:
            True if the entry was updated, False if not found or skipped
        """
        from aragora.memory.tier_manager import MemoryTier

        # Strip mound prefix if present
        if memory_id.startswith("cm_"):
            memory_id = memory_id[3:]

        # Get current entry
        entry = self._continuum.get(memory_id)
        if not entry:
            logger.warning(f"Continuum entry not found for KM validation: {memory_id}")
            return False

        # Calculate new importance based on KM feedback
        current_importance = entry.importance
        km_confidence = km_validation.km_confidence
        validation_count = km_validation.validation_count
        cross_debate_utility = km_validation.cross_debate_utility

        # Weighted average: more validations = more weight on KM confidence
        # Also factor in cross-debate utility
        weight = min(0.5, validation_count * 0.1)  # Max 50% weight
        utility_boost = cross_debate_utility * 0.1  # Up to 10% boost

        new_importance = current_importance * (1 - weight) + km_confidence * weight + utility_boost
        new_importance = min(1.0, max(0.0, new_importance))  # Clamp to [0, 1]

        # Determine if tier change is needed based on recommendation
        recommendation = km_validation.recommendation
        tier_changed = False

        if recommendation == "promote" and entry.tier != MemoryTier.GLACIAL:
            # Promote to a more permanent tier
            tier_order = [MemoryTier.FAST, MemoryTier.MEDIUM, MemoryTier.SLOW, MemoryTier.GLACIAL]
            current_idx = tier_order.index(entry.tier)
            if current_idx < len(tier_order) - 1:
                new_tier = tier_order[current_idx + 1]
                tier_changed = self._continuum.promote_entry(memory_id, new_tier)
                if tier_changed:
                    logger.info(
                        f"Promoted continuum entry from KM validation: {memory_id} "
                        f"{entry.tier.value} -> {new_tier.value}"
                    )

        elif recommendation == "demote" and entry.tier != MemoryTier.FAST:
            # Demote to a less permanent tier
            tier_order = [MemoryTier.FAST, MemoryTier.MEDIUM, MemoryTier.SLOW, MemoryTier.GLACIAL]
            current_idx = tier_order.index(entry.tier)
            if current_idx > 0:
                new_tier = tier_order[current_idx - 1]
                tier_changed = self._continuum.demote_entry(memory_id, new_tier)
                if tier_changed:
                    logger.info(
                        f"Demoted continuum entry from KM validation: {memory_id} "
                        f"{entry.tier.value} -> {new_tier.value}"
                    )

        # Check if importance changed significantly
        importance_changed = abs(new_importance - current_importance) > 0.01

        # Always update metadata to mark as KM-validated
        # Even if importance/tier don't change, we want to track validation
        metadata = entry.metadata.copy()
        metadata["km_validated"] = True
        metadata["km_validation_count"] = validation_count
        metadata["km_confidence"] = km_confidence
        metadata["km_cross_debate_utility"] = cross_debate_utility
        if km_validation.was_contradicted:
            metadata["km_contradicted"] = True
        if km_validation.was_supported:
            metadata["km_supported"] = True

        # Update the entry via direct method
        self._continuum.update(
            memory_id,
            importance=new_importance if importance_changed else None,
            metadata=metadata,
        )

        if importance_changed:
            logger.info(
                f"Updated continuum entry from KM: {memory_id} "
                f"importance {current_importance:.2f} -> {new_importance:.2f}"
            )
        else:
            logger.debug(
                f"Updated continuum metadata from KM: {memory_id} "
                f"(importance unchanged at {current_importance:.2f})"
            )

        # Return True if any meaningful update was made
        return True

    async def sync_validations_to_continuum(
        self,
        workspace_id: str,
        validations: List[KMValidationResult],
        min_confidence: float = 0.7,
    ) -> ValidationSyncResult:
        """
        Batch sync KM validations back to ContinuumMemory.

        Processes a list of KM validations and updates the corresponding
        continuum memory entries. High-confidence validations can trigger
        tier promotions; low-confidence or contradicted items can trigger demotions.

        Args:
            workspace_id: Workspace ID for filtering
            validations: List of KM validation results
            min_confidence: Minimum confidence to apply changes (default 0.7)

        Returns:
            ValidationSyncResult with counts of promotions, demotions, and updates
        """
        import time

        start_time = time.time()
        result = ValidationSyncResult()
        result.total_processed = len(validations)

        for validation in validations:
            try:
                # Skip low-confidence validations
                if validation.km_confidence < min_confidence:
                    result.skipped += 1
                    continue

                # Apply validation
                updated = await self.update_continuum_from_km(
                    validation.memory_id,
                    validation,
                )

                if updated:
                    if validation.recommendation == "promote":
                        result.promoted += 1
                    elif validation.recommendation == "demote":
                        result.demoted += 1
                    else:
                        result.updated += 1
                else:
                    result.skipped += 1

            except Exception as e:
                error_msg = f"Error validating {validation.memory_id}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        result.duration_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"KM validation sync complete: "
            f"promoted={result.promoted}, demoted={result.demoted}, "
            f"updated={result.updated}, skipped={result.skipped}, "
            f"errors={len(result.errors)}, duration={result.duration_ms}ms"
        )

        return result

    async def get_km_validated_entries(
        self,
        limit: int = 50,
        min_km_confidence: float = 0.7,
    ) -> List["ContinuumMemoryEntry"]:
        """
        Get continuum entries that have been validated by KM.

        Useful for:
        - Finding high-quality memories for context injection
        - Auditing which memories have KM validation
        - Building training data from validated examples

        Args:
            limit: Maximum entries to return
            min_km_confidence: Minimum KM confidence score

        Returns:
            List of validated continuum memory entries
        """
        # Retrieve all entries and filter by KM validation
        all_entries = self._continuum.retrieve(limit=limit * 2)

        validated = []
        for entry in all_entries:
            km_validated = entry.metadata.get("km_validated", False)
            km_confidence = entry.metadata.get("km_confidence", 0.0)

            if km_validated and km_confidence >= min_km_confidence:
                validated.append(entry)

            if len(validated) >= limit:
                break

        return validated

    async def sync_memory_to_mound(
        self,
        mound: Any,
        workspace_id: str,
        min_importance: float = 0.7,
        limit: int = 100,
        tiers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Sync high-importance continuum memories to the Knowledge Mound.

        This is the forward flow: significant memories are stored in KM for
        long-term persistence, semantic search, and cross-debate retrieval.

        Args:
            mound: KnowledgeMound instance to sync to
            workspace_id: Workspace ID for the KM entries
            min_importance: Minimum importance threshold (default 0.7)
            limit: Maximum entries to sync
            tiers: Optional list of tier names to filter

        Returns:
            Dict with sync statistics (synced, skipped, errors)
        """
        from aragora.knowledge.mound.types import (
            IngestionRequest,
            SourceType,
        )

        result: dict[str, Any] = {
            "synced": 0,
            "skipped": 0,
            "already_synced": 0,
            "errors": [],
        }

        # Get high-importance memories
        entries = self.search_by_keyword(
            query="",  # Empty query returns all
            limit=limit,
            tiers=tiers,
            min_importance=min_importance,
        )

        for entry in entries:
            try:
                # Check if already synced to KM
                if entry.metadata.get("km_synced"):
                    result["already_synced"] += 1
                    continue

                # Skip if below importance threshold
                if entry.importance < min_importance:
                    result["skipped"] += 1
                    continue

                # Create ingestion request
                request = IngestionRequest(
                    content=entry.content,
                    source_type=SourceType.CONTINUUM,
                    workspace_id=workspace_id,
                    confidence=entry.importance,
                    tier=self._tier_to_km_tier(entry.tier.value),  # type: ignore[arg-type]
                    metadata={
                        "continuum_id": entry.id,
                        "continuum_tier": entry.tier.value,
                        "surprise_score": entry.surprise_score,
                        "consolidation_score": entry.consolidation_score,
                        "tags": entry.tags,
                    },
                )

                # Ingest into KM
                km_id = await mound.ingest(request)

                # Mark entry as synced in continuum
                entry_metadata = entry.metadata.copy()
                entry_metadata["km_synced"] = True
                entry_metadata["km_node_id"] = km_id
                self._continuum.update(entry.id, metadata=entry_metadata)

                # Create bidirectional link
                self.link_to_mound(entry.id, km_id)

                result["synced"] += 1
                logger.debug(f"Synced continuum entry to KM: {entry.id} -> {km_id}")

            except Exception as e:
                error_msg = f"Error syncing {entry.id}: {e}"
                logger.warning(error_msg)
                result["errors"].append(error_msg)

        logger.info(
            f"Memory to KM sync complete: synced={result['synced']}, "
            f"skipped={result['skipped']}, already_synced={result['already_synced']}, "
            f"errors={len(result['errors'])}"
        )

        return result

    def _tier_to_km_tier(self, tier_name: str) -> int:
        """Convert continuum tier name to KM tier number."""
        mapping = {
            "fast": 1,
            "medium": 2,
            "slow": 3,
            "glacial": 4,
        }
        return mapping.get(tier_name, 3)

    def get_reverse_sync_stats(self) -> Dict[str, Any]:
        """
        Get statistics about reverse sync (KM → ContinuumMemory).

        Returns counts of KM-validated entries by tier and validation status.
        """
        stats: dict[str, Any] = {
            "total_km_validated": 0,
            "km_validated_by_tier": {},
            "km_supported": 0,
            "km_contradicted": 0,
            "avg_km_confidence": 0.0,
            "avg_cross_debate_utility": 0.0,
        }

        # Sample entries to compute stats
        all_entries = self._continuum.retrieve(limit=1000)

        confidence_sum = 0.0
        utility_sum = 0.0
        validated_count = 0

        for entry in all_entries:
            if entry.metadata.get("km_validated"):
                validated_count += 1
                stats["total_km_validated"] += 1

                tier = entry.tier.value
                stats["km_validated_by_tier"][tier] = stats["km_validated_by_tier"].get(tier, 0) + 1

                if entry.metadata.get("km_supported"):
                    stats["km_supported"] += 1
                if entry.metadata.get("km_contradicted"):
                    stats["km_contradicted"] += 1

                confidence_sum += entry.metadata.get("km_confidence", 0.0)
                utility_sum += entry.metadata.get("km_cross_debate_utility", 0.0)

        if validated_count > 0:
            stats["avg_km_confidence"] = round(confidence_sum / validated_count, 3)
            stats["avg_cross_debate_utility"] = round(utility_sum / validated_count, 3)

        return stats
