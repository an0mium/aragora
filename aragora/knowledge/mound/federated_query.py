"""
Federated Query Aggregator - Query across multiple KM adapters simultaneously.

Enables unified queries across all KM adapters (Evidence, Belief, Insights, ELO,
Pulse, etc.) with result aggregation and ranking.

Usage:
    from aragora.knowledge.mound.federated_query import FederatedQueryAggregator

    aggregator = FederatedQueryAggregator()
    aggregator.register_adapter("evidence", evidence_adapter)
    aggregator.register_adapter("belief", belief_adapter)

    results = await aggregator.query(
        query="climate change",
        sources=["evidence", "belief", "insights"],
        limit=20,
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import KnowledgeItem

logger = logging.getLogger(__name__)


class QuerySource(str, Enum):
    """Available query sources (adapters)."""

    EVIDENCE = "evidence"
    BELIEF = "belief"
    INSIGHTS = "insights"
    ELO = "elo"
    PULSE = "pulse"
    COST = "cost"
    CONTINUUM = "continuum"
    CONSENSUS = "consensus"
    CRITIQUE = "critique"
    ALL = "all"


@dataclass
class FederatedResult:
    """A single result from a federated query."""

    source: QuerySource
    item: Any  # The actual item from the adapter
    relevance_score: float = 0.0
    adapter_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        """Extract content from the item."""
        if hasattr(self.item, "content"):
            return self.item.content
        if isinstance(self.item, dict):
            return self.item.get("content", str(self.item))
        return str(self.item)

    @property
    def id(self) -> str:
        """Extract ID from the item."""
        if hasattr(self.item, "id"):
            return self.item.id
        if isinstance(self.item, dict):
            return self.item.get("id", "")
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source.value,
            "content": self.content,
            "id": self.id,
            "relevance_score": self.relevance_score,
            "adapter_metadata": self.adapter_metadata,
        }


@dataclass
class FederatedQueryResult:
    """Result of a federated query across adapters."""

    query: str
    results: List[FederatedResult] = field(default_factory=list)
    total_count: int = 0
    sources_queried: List[QuerySource] = field(default_factory=list)
    sources_succeeded: List[QuerySource] = field(default_factory=list)
    sources_failed: List[QuerySource] = field(default_factory=list)
    execution_time_ms: float = 0
    errors: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_count": self.total_count,
            "sources_queried": [s.value for s in self.sources_queried],
            "sources_succeeded": [s.value for s in self.sources_succeeded],
            "sources_failed": [s.value for s in self.sources_failed],
            "execution_time_ms": self.execution_time_ms,
            "errors": self.errors,
        }


@dataclass
class AdapterRegistration:
    """Registration info for an adapter."""

    name: str
    adapter: Any
    search_method: str
    enabled: bool = True
    weight: float = 1.0  # Weight for ranking results
    metadata: Dict[str, Any] = field(default_factory=dict)


# Type for custom relevance scoring functions
RelevanceScorer = Callable[[Any, str], float]


class FederatedQueryAggregator:
    """
    Aggregates queries across multiple KM adapters.

    Features:
    - Parallel querying across adapters
    - Result aggregation and deduplication
    - Relevance-based ranking
    - Partial failure handling
    - Query caching (optional)
    """

    def __init__(
        self,
        parallel: bool = True,
        timeout_seconds: float = 10.0,
        default_limit: int = 20,
        deduplicate: bool = True,
        relevance_scorer: Optional[RelevanceScorer] = None,
    ):
        """
        Initialize the aggregator.

        Args:
            parallel: Run queries in parallel (default True)
            timeout_seconds: Timeout per adapter query
            default_limit: Default result limit per source
            deduplicate: Remove duplicate results (by content hash)
            relevance_scorer: Custom function to score result relevance
        """
        self._parallel = parallel
        self._timeout_seconds = timeout_seconds
        self._default_limit = default_limit
        self._deduplicate = deduplicate
        self._relevance_scorer = relevance_scorer or self._default_scorer

        # Registered adapters
        self._adapters: Dict[QuerySource, AdapterRegistration] = {}

        # Query statistics
        self._total_queries = 0
        self._successful_queries = 0

    def register_adapter(
        self,
        source: QuerySource | str,
        adapter: Any,
        search_method: str = "search_by_topic",
        weight: float = 1.0,
        enabled: bool = True,
    ) -> None:
        """
        Register an adapter for federated queries.

        Args:
            source: Query source identifier
            adapter: The adapter instance
            search_method: Method name to call for searches
            weight: Weight for result ranking
            enabled: Whether this adapter is enabled
        """
        if isinstance(source, str):
            source = QuerySource(source)

        if not hasattr(adapter, search_method):
            logger.warning(
                f"Adapter for {source.value} missing search method: {search_method}"
            )
            # Try common alternatives
            for alt in ["search", "search_similar", "search_by_keyword", "query"]:
                if hasattr(adapter, alt):
                    search_method = alt
                    break
            else:
                logger.error(f"No suitable search method found for {source.value}")
                return

        self._adapters[source] = AdapterRegistration(
            name=source.value,
            adapter=adapter,
            search_method=search_method,
            weight=weight,
            enabled=enabled,
        )

        logger.debug(f"Registered adapter: {source.value} (method={search_method})")

    def unregister_adapter(self, source: QuerySource | str) -> bool:
        """Unregister an adapter."""
        if isinstance(source, str):
            source = QuerySource(source)

        if source in self._adapters:
            del self._adapters[source]
            return True
        return False

    def enable_adapter(self, source: QuerySource | str) -> bool:
        """Enable an adapter."""
        if isinstance(source, str):
            source = QuerySource(source)

        if source in self._adapters:
            self._adapters[source].enabled = True
            return True
        return False

    def disable_adapter(self, source: QuerySource | str) -> bool:
        """Disable an adapter."""
        if isinstance(source, str):
            source = QuerySource(source)

        if source in self._adapters:
            self._adapters[source].enabled = False
            return True
        return False

    async def query(
        self,
        query: str,
        sources: Optional[List[QuerySource | str]] = None,
        limit: Optional[int] = None,
        min_relevance: float = 0.0,
        **kwargs,
    ) -> FederatedQueryResult:
        """
        Execute a federated query across adapters.

        Args:
            query: The search query
            sources: List of sources to query (None = all enabled)
            limit: Max results per source
            min_relevance: Minimum relevance score to include
            **kwargs: Additional kwargs passed to adapter search methods

        Returns:
            FederatedQueryResult with aggregated results
        """
        start_time = time.time()
        self._total_queries += 1

        limit = limit or self._default_limit

        # Determine sources to query
        if sources is None or QuerySource.ALL in sources:
            query_sources = [
                s for s, r in self._adapters.items() if r.enabled
            ]
        else:
            query_sources = [
                QuerySource(s) if isinstance(s, str) else s
                for s in sources
                if (QuerySource(s) if isinstance(s, str) else s) in self._adapters
            ]

        result = FederatedQueryResult(
            query=query,
            sources_queried=query_sources,
        )

        if not query_sources:
            result.execution_time_ms = (time.time() - start_time) * 1000
            return result

        # Query adapters
        if self._parallel:
            adapter_results = await self._query_parallel(query, query_sources, limit, **kwargs)
        else:
            adapter_results = await self._query_sequential(query, query_sources, limit, **kwargs)

        # Process results
        all_results = []
        for source, (items, error) in adapter_results.items():
            if error:
                result.sources_failed.append(source)
                result.errors[source.value] = error
            else:
                result.sources_succeeded.append(source)
                weight = self._adapters[source].weight

                for item in items:
                    relevance = self._relevance_scorer(item, query) * weight
                    if relevance >= min_relevance:
                        all_results.append(FederatedResult(
                            source=source,
                            item=item,
                            relevance_score=relevance,
                            adapter_metadata={"weight": weight},
                        ))

        # Deduplicate if enabled
        if self._deduplicate:
            all_results = self._deduplicate_results(all_results)

        # Sort by relevance
        all_results.sort(key=lambda r: r.relevance_score, reverse=True)

        result.results = all_results
        result.total_count = len(all_results)
        result.execution_time_ms = (time.time() - start_time) * 1000

        if result.sources_succeeded:
            self._successful_queries += 1

        return result

    async def _query_parallel(
        self,
        query: str,
        sources: List[QuerySource],
        limit: int,
        **kwargs,
    ) -> Dict[QuerySource, Tuple[List[Any], Optional[str]]]:
        """Query adapters in parallel."""
        tasks = {}
        for source in sources:
            tasks[source] = asyncio.create_task(
                self._query_single(source, query, limit, **kwargs)
            )

        results = {}
        for source, task in tasks.items():
            try:
                items = await asyncio.wait_for(task, timeout=self._timeout_seconds)
                results[source] = (items, None)
            except asyncio.TimeoutError:
                results[source] = ([], f"Timeout after {self._timeout_seconds}s")
            except Exception as e:
                results[source] = ([], str(e))

        return results

    async def _query_sequential(
        self,
        query: str,
        sources: List[QuerySource],
        limit: int,
        **kwargs,
    ) -> Dict[QuerySource, Tuple[List[Any], Optional[str]]]:
        """Query adapters sequentially."""
        results = {}
        for source in sources:
            try:
                items = await self._query_single(source, query, limit, **kwargs)
                results[source] = (items, None)
            except Exception as e:
                results[source] = ([], str(e))

        return results

    async def _query_single(
        self,
        source: QuerySource,
        query: str,
        limit: int,
        **kwargs,
    ) -> List[Any]:
        """Query a single adapter."""
        registration = self._adapters[source]
        adapter = registration.adapter
        method = getattr(adapter, registration.search_method)

        # Call the search method
        if asyncio.iscoroutinefunction(method):
            result = await method(query=query, limit=limit, **kwargs)
        else:
            result = method(query=query, limit=limit, **kwargs)

        # Handle different return types
        if isinstance(result, list):
            return result
        if hasattr(result, "items"):
            return result.items
        if hasattr(result, "results"):
            return result.results

        return [result] if result else []

    def _default_scorer(self, item: Any, query: str) -> float:
        """
        Default relevance scorer based on content similarity.

        Uses simple keyword matching. Replace with embedding-based
        scoring for better results.
        """
        content = ""
        if hasattr(item, "content"):
            content = item.content.lower()
        elif isinstance(item, dict):
            content = str(item.get("content", item)).lower()
        else:
            content = str(item).lower()

        query_terms = query.lower().split()
        if not query_terms:
            return 0.5

        matches = sum(1 for term in query_terms if term in content)
        return min(1.0, matches / len(query_terms))

    def _deduplicate_results(
        self,
        results: List[FederatedResult],
    ) -> List[FederatedResult]:
        """
        Remove duplicate results based on content hash.

        Keeps the result with the highest relevance score.
        """
        seen: Dict[str, FederatedResult] = {}

        for result in results:
            content_hash = self._hash_content(result.content)

            if content_hash not in seen:
                seen[content_hash] = result
            elif result.relevance_score > seen[content_hash].relevance_score:
                seen[content_hash] = result

        return list(seen.values())

    def _hash_content(self, content: str) -> str:
        """Hash content for deduplication."""
        import hashlib
        # Normalize and hash
        normalized = " ".join(content.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        return {
            "total_queries": self._total_queries,
            "successful_queries": self._successful_queries,
            "success_rate": (
                self._successful_queries / max(self._total_queries, 1) * 100
            ),
            "registered_adapters": len(self._adapters),
            "enabled_adapters": sum(1 for r in self._adapters.values() if r.enabled),
            "adapters": {
                s.value: {
                    "enabled": r.enabled,
                    "weight": r.weight,
                    "search_method": r.search_method,
                }
                for s, r in self._adapters.items()
            },
        }

    def get_registered_sources(self) -> List[str]:
        """Get list of registered source names."""
        return [s.value for s in self._adapters.keys()]


__all__ = [
    "FederatedQueryAggregator",
    "FederatedQueryResult",
    "FederatedResult",
    "QuerySource",
]
