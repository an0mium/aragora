"""Semantic search mixin for Knowledge Mound adapters.

Provides unified semantic vector search functionality that can be mixed into
any adapter. Consolidates ~350 lines of duplicated semantic search code
across 7 adapters (consensus, continuum, evidence, belief, insights, pulse, cost).

NOTE: This is a mixin class designed to be composed with KnowledgeMoundAdapter.
Attribute accesses like self._emit_event, self._record_metric, etc. are provided
by the composed class. The ``# type: ignore[attr-defined]`` comments suppress
mypy warnings that are expected for this mixin pattern.

Usage:
    from aragora.knowledge.mound.adapters._semantic_mixin import SemanticSearchMixin

    class MyAdapter(SemanticSearchMixin, KnowledgeMoundAdapter):
        adapter_name = "my_adapter"
        source_type = "my_source"  # For SemanticStore

        # Optional: Override for custom record lookup (defaults use common patterns)
        def _get_record_by_id(self, record_id: str) -> Any | None:
            return self._source.get(record_id)

        # Optional: Override for custom serialization (defaults handle dicts, dataclasses)
        def _record_to_dict(self, record: Any) -> dict[str, Any]:
            return record.to_dict()

Default implementations are provided for _get_record_by_id() and _record_to_dict()
that handle common adapter patterns:
- _get_record_by_id(): Tries get() method, _source.get(), and dict storage lookups
- _record_to_dict(): Handles dicts, dataclasses, to_dict() methods, and common attributes
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from aragora.knowledge.mound.semantic_store import SemanticSearchResult

logger = logging.getLogger(__name__)


class _AdapterProtocol(Protocol):
    """Protocol for adapter methods expected by SemanticSearchMixin."""

    adapter_name: str
    source_type: str

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None: ...
    def _record_metric(self, operation: str, success: bool, duration: float) -> None: ...


class SemanticSearchMixin:
    """Mixin providing semantic vector search for adapters.

    Provides:
    - semantic_search(): Vector-based similarity search with fallback
    - _enrich_semantic_results(): Convert search results to full records
    - _get_record_by_id(): Default record lookup (can be overridden)
    - _record_to_dict(): Default record serialization (can be overridden)
    - Automatic metrics recording and event emission

    Required from inheriting class:
    - adapter_name: str identifying the adapter for metrics
    - source_type: str identifying the source for SemanticStore

    Optional (have defaults but can be overridden):
    - _get_record_by_id(): Custom record lookup logic
    - _record_to_dict(): Custom serialization logic
    - _record_metric(): Metrics recording (defaults to no-op)
    - _emit_event(): Event emission (defaults to no-op)
    - search_similar(): Fallback keyword search method
    """

    # Override in subclass to identify the source type for SemanticStore
    source_type: str = "unknown"

    # Expected from KnowledgeMoundAdapter or subclass
    adapter_name: str

    # Note: _emit_event and _record_metric are expected from KnowledgeMoundAdapter
    # via inheritance. Do NOT add stub implementations here as they would
    # shadow the real implementations due to MRO when this mixin is listed
    # before KnowledgeMoundAdapter in the inheritance chain.

    def _get_record_by_id(self, record_id: str) -> Any | None:
        """Get a full record by its ID from the source system.

        This default implementation attempts to find a record using common
        adapter patterns. Override this method for custom record lookup logic.

        Lookup strategy:
        1. Try adapter's get() method if available
        2. Try adapter's _source.get() if _source is set
        3. Try looking up in common storage attributes (_records, _items, _data)

        Args:
            record_id: The record identifier.

        Returns:
            The full record, or None if not found.
        """
        # Strategy 1: Try adapter's get() method
        get_method = getattr(self, "get", None)
        if callable(get_method):
            try:
                result = get_method(record_id)
                if result is not None:
                    return result
            except Exception as e:
                logger.debug(f"get() method failed for {record_id}: {e}")

        # Strategy 2: Try _source.get() if _source exists
        source = getattr(self, "_source", None)
        if source is not None:
            get_from_source = getattr(source, "get", None)
            if callable(get_from_source):
                try:
                    result = get_from_source(record_id)
                    if result is not None:
                        return result
                except Exception as e:
                    logger.debug(f"_source.get() failed for {record_id}: {e}")

        # Strategy 3: Look up in common storage attributes
        for attr_name in ("_records", "_items", "_data", "_beliefs", "_cruxes"):
            storage = getattr(self, attr_name, None)
            if isinstance(storage, dict):
                if record_id in storage:
                    return storage[record_id]
                # Try with adapter-specific prefix stripping
                stripped_id = self._extract_record_id(record_id)
                if stripped_id != record_id and stripped_id in storage:
                    return storage[stripped_id]

        logger.debug(
            f"[{self.adapter_name}] Record not found: {record_id}. "
            f"Override _get_record_by_id() for custom lookup logic."
        )
        return None

    def _record_to_dict(self, record: Any, similarity: float = 0.0) -> dict[str, Any]:
        """Convert a record to a dictionary for API responses.

        This default implementation handles common record types:
        - dict: Returns as-is with similarity added
        - dataclass: Uses dataclasses.asdict() if available
        - object with to_dict(): Calls the method
        - object: Extracts common attributes

        Override this method for custom serialization logic.

        Args:
            record: The record to convert.
            similarity: Optional similarity score to include.

        Returns:
            Dictionary representation of the record.
        """
        result: dict[str, Any]

        # Already a dict
        if isinstance(record, dict):
            result = dict(record)
            result["similarity"] = similarity
            return result

        # Has to_dict() method
        if hasattr(record, "to_dict") and callable(record.to_dict):
            try:
                result = record.to_dict()
                result["similarity"] = similarity
                return result
            except Exception as e:
                logger.debug(f"to_dict() failed: {e}")

        # Is a dataclass
        try:
            import dataclasses

            if dataclasses.is_dataclass(record) and not isinstance(record, type):
                result = dataclasses.asdict(record)
                result["similarity"] = similarity
                return result
        except Exception as e:
            logger.debug(f"dataclasses.asdict() failed: {e}")

        # Extract common attributes manually
        result = {"similarity": similarity}

        # Common ID fields
        for id_field in ("id", "record_id", "node_id", "claim_id"):
            if hasattr(record, id_field):
                result["id"] = getattr(record, id_field)
                break

        # Common content fields
        for content_field in ("content", "text", "statement", "topic", "description"):
            if hasattr(record, content_field):
                result["content"] = getattr(record, content_field)
                break

        # Common confidence/score fields
        for score_field in ("confidence", "score", "reliability", "importance"):
            if hasattr(record, score_field):
                value = getattr(record, score_field)
                # Handle objects with .value (like enums or Probability)
                if hasattr(value, "value"):
                    value = value.value
                elif hasattr(value, "p_true"):
                    value = value.p_true
                result[score_field] = value

        # Common timestamp fields
        for ts_field in ("created_at", "timestamp", "updated_at", "detected_at"):
            if hasattr(record, ts_field):
                ts_value = getattr(record, ts_field)
                if hasattr(ts_value, "isoformat"):
                    result[ts_field] = ts_value.isoformat()
                else:
                    result[ts_field] = str(ts_value)

        # Common metadata field
        if hasattr(record, "metadata"):
            result["metadata"] = getattr(record, "metadata")

        # Domain/category fields
        for domain_field in ("domain", "category", "type"):
            if hasattr(record, domain_field):
                value = getattr(record, domain_field)
                if hasattr(value, "value"):
                    result[domain_field] = value.value
                else:
                    result[domain_field] = value

        return result

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
        tenant_id: str | None = None,
        fallback_fn: Optional[Callable[..., list[dict[str, Any]]]] = None,
    ) -> list[dict[str, Any]]:
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
                from aragora.config import DB_KNOWLEDGE_PATH
                from aragora.knowledge.mound.semantic_store import SemanticStore

                # Get or create semantic store with default path
                semantic_db_path: str | Path = DB_KNOWLEDGE_PATH / "mound_semantic.db"
                store = SemanticStore(db_path=semantic_db_path)

                # Search using embeddings
                results: list[SemanticSearchResult] = await store.search_similar(
                    query=query,
                    tenant_id=tenant_id or "default",
                    limit=limit,
                    min_similarity=min_similarity,
                    source_types=[self.source_type],
                )

                # Enrich results with full records
                enriched = self._enrich_semantic_results(results)

                success = True
                logger.debug(
                    f"[{self.adapter_name}] Semantic search returned "
                    f"{len(enriched)} results for '{query[:50]}'"
                )

                # Emit event - _emit_event provided by KnowledgeMoundAdapter via inheritance
                adapter = cast(_AdapterProtocol, self)
                adapter._emit_event(
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
            fallback_results: list[dict[str, Any]]
            if fallback_fn:
                fallback_results = fallback_fn(query, limit=limit, min_confidence=min_similarity)
            else:
                # Try to use adapter's search_similar method if available
                search_method: Callable[..., list[dict[str, Any]]] | None = getattr(
                    self, "search_similar", None
                )
                if search_method is not None:
                    fallback_results = search_method(
                        query, limit=limit, min_confidence=min_similarity
                    )
                else:
                    fallback_results = []

            success = True
            return fallback_results

        finally:
            # _record_metric provided by KnowledgeMoundAdapter via inheritance
            adapter = cast(_AdapterProtocol, self)
            adapter._record_metric("semantic_search", success, time.time() - start)

    def _enrich_semantic_results(self, results: list[Any]) -> list[dict[str, Any]]:
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
