"""Semantic search mixin for Knowledge Mound adapters.

Provides unified semantic vector search functionality that can be mixed into
any adapter. Consolidates ~350 lines of duplicated semantic search code
across 7 adapters (consensus, continuum, evidence, belief, insights, pulse, cost).

Usage:
    from aragora.knowledge.mound.adapters._semantic_mixin import SemanticSearchMixin

    class MyAdapter(SemanticSearchMixin, KnowledgeMoundAdapter):
        adapter_name = "my_adapter"
        source_type = "my_source"  # For SemanticStore

        def _get_record_by_id(self, record_id: str) -> Optional[Any]:
            # Return the full record from your source system
            return self._source.get(record_id)

        def _record_to_dict(self, record: Any) -> Dict[str, Any]:
            # Convert your record to a dictionary
            return record.to_dict()
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # SemanticStore import for type hints

logger = logging.getLogger(__name__)


class SemanticSearchMixin:
    """Mixin providing semantic vector search for adapters.

    Provides:
    - semantic_search(): Vector-based similarity search with fallback
    - _enrich_semantic_results(): Convert search results to full records
    - Automatic metrics recording and event emission

    Required from inheriting class:
    - adapter_name: str identifying the adapter for metrics
    - source_type: str identifying the source for SemanticStore
    - _get_record_by_id(): Method to fetch full records
    - _record_to_dict(): Method to convert records to dicts
    - _record_metric(): Method from KnowledgeMoundAdapter
    - _emit_event(): Method from KnowledgeMoundAdapter
    - search_similar(): Fallback keyword search method
    """

    # Override in subclass to identify the source type for SemanticStore
    source_type: str = "unknown"

    # Expected from KnowledgeMoundAdapter or subclass
    adapter_name: str

    def _record_metric(self, operation: str, success: bool, latency: float) -> None:
        """Expected from KnowledgeMoundAdapter."""
        pass  # Will be provided by base class

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Expected from KnowledgeMoundAdapter."""
        pass  # Will be provided by base class

    @abstractmethod
    def _get_record_by_id(self, record_id: str) -> Optional[Any]:
        """Get a full record by its ID from the source system.

        Args:
            record_id: The record identifier.

        Returns:
            The full record, or None if not found.
        """
        raise NotImplementedError

    @abstractmethod
    def _record_to_dict(self, record: Any, similarity: float = 0.0) -> Dict[str, Any]:
        """Convert a record to a dictionary for API responses.

        Args:
            record: The record to convert.
            similarity: Optional similarity score to include.

        Returns:
            Dictionary representation of the record.
        """
        raise NotImplementedError

    def _extract_record_id(self, source_id: str) -> str:
        """Extract the actual record ID from a prefixed source ID.

        Override if your adapter uses prefixed IDs (e.g., "cs_123" -> "123").

        Args:
            source_id: The source ID from SemanticStore.

        Returns:
            The actual record ID for lookup.
        """
        # Default: use as-is
        return source_id

    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.6,
        tenant_id: Optional[str] = None,
        fallback_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform semantic vector search over records.

        Uses the Knowledge Mound's SemanticStore for embedding-based similarity
        search, falling back to keyword search if embeddings aren't available.

        Args:
            query: The search query.
            limit: Maximum results to return.
            min_similarity: Minimum similarity threshold (0.0-1.0).
            tenant_id: Optional tenant filter.
            fallback_fn: Optional fallback function if semantic search fails.

        Returns:
            List of matching records with similarity scores.
        """
        start = time.time()
        success = False

        try:
            # Try semantic search first
            try:
                from aragora.knowledge.mound.semantic_store import SemanticStore

                # Get or create semantic store
                store = SemanticStore()  # type: ignore[call-arg]

                # Search using embeddings
                results = await store.search_similar(  # type: ignore[call-arg]
                    query=query,
                    tenant_id=tenant_id or "default",
                    limit=limit,
                    min_similarity=min_similarity,
                    source_type=self.source_type,
                )

                # Enrich results with full records
                enriched = self._enrich_semantic_results(results)

                success = True
                logger.debug(
                    f"[{self.adapter_name}] Semantic search returned "
                    f"{len(enriched)} results for '{query[:50]}'"
                )

                # Emit event
                self._emit_event(
                    "km_adapter_semantic_search",
                    {
                        "source": self.source_type,
                        "query_preview": query[:50],
                        "results_count": len(enriched),
                        "search_type": "vector",
                    },
                )

                return enriched

            except ImportError:
                logger.debug(
                    f"[{self.adapter_name}] SemanticStore not available, "
                    "falling back to keyword search"
                )
            except Exception as e:
                logger.debug(f"[{self.adapter_name}] Semantic search failed, falling back: {e}")

            # Fallback to keyword search
            if fallback_fn:
                results = fallback_fn(query, limit=limit, min_confidence=min_similarity)
            elif hasattr(self, "search_similar"):
                results = self.search_similar(query, limit=limit, min_confidence=min_similarity)  # type: ignore[attr-defined]
            else:
                results = []

            success = True
            return results

        finally:
            self._record_metric("semantic_search", success, time.time() - start)

    def _enrich_semantic_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Enrich semantic search results with full record data.

        Args:
            results: Raw results from SemanticStore.

        Returns:
            List of enriched result dictionaries.
        """
        enriched = []
        for r in results:
            # Extract record ID (may be prefixed)
            record_id = self._extract_record_id(r.source_id)

            # Try to get the full record
            record = self._get_record_by_id(record_id)
            if record:
                # Convert to dict with similarity
                result_dict = self._record_to_dict(record, similarity=r.similarity)
                enriched.append(result_dict)
            else:
                # Record not found in source, include basic info
                enriched.append(
                    {
                        "id": r.source_id,
                        "similarity": r.similarity,
                        "domain": getattr(r, "domain", None),
                        "importance": getattr(r, "importance", 0.0),
                        "metadata": getattr(r, "metadata", {}),
                    }
                )

        return enriched


__all__ = ["SemanticSearchMixin"]
