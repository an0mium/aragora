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


class EmbeddingRelevanceScorer:
    """
    Relevance scorer using semantic embeddings for better accuracy.

    Uses cosine similarity between query and content embeddings to compute
    relevance scores. Falls back to keyword matching if embeddings unavailable.

    Usage:
        scorer = EmbeddingRelevanceScorer()
        await scorer.initialize()

        aggregator = FederatedQueryAggregator(
            relevance_scorer=scorer.score,
        )
    """

    def __init__(self, cache_size: int = 1000):
        """
        Initialize the embedding scorer.

        Args:
            cache_size: Max number of embeddings to cache
        """
        self._provider = None
        self._initialized = False
        self._cache: Dict[str, List[float]] = {}
        self._cache_size = cache_size
        self._query_embedding: Optional[List[float]] = None
        self._current_query: Optional[str] = None

    async def initialize(self) -> bool:
        """
        Initialize the embedding provider.

        Returns:
            True if initialization succeeded
        """
        if self._initialized:
            return True

        try:
            from aragora.core.embeddings import get_default_provider

            self._provider = get_default_provider()
            self._initialized = True
            logger.debug("EmbeddingRelevanceScorer initialized with provider")
            return True
        except ImportError:
            logger.debug("Embedding provider not available, using keyword fallback")
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize embedding provider: {e}")
            return False

    def _get_content(self, item: Any) -> str:
        """Extract content from an item."""
        if hasattr(item, "content"):
            return str(item.content)
        if isinstance(item, dict):
            return str(item.get("content", item))
        return str(item)

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:16]

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text, using cache if available."""
        if not self._provider:
            return None

        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]

        try:
            embedding = await self._provider.embed_async(text)

            # LRU-style cache management
            if len(self._cache) >= self._cache_size:
                # Remove oldest entry
                oldest = next(iter(self._cache))
                del self._cache[oldest]

            self._cache[key] = embedding
            return embedding
        except Exception as e:
            logger.debug(f"Failed to get embedding: {e}")
            return None

    async def prepare_query(self, query: str) -> None:
        """
        Pre-compute query embedding for efficient batch scoring.

        Call this before scoring multiple items against the same query.

        Args:
            query: The search query
        """
        if query != self._current_query:
            self._current_query = query
            self._query_embedding = await self._get_embedding(query)

    def score(self, item: Any, query: str) -> float:
        """
        Score item relevance to query (sync interface for compatibility).

        Note: For best results, call prepare_query() first to pre-compute
        the query embedding.

        Args:
            item: The item to score
            query: The search query

        Returns:
            Relevance score between 0 and 1
        """
        # If we have a cached query embedding, use it
        if self._query_embedding and query == self._current_query:
            content = self._get_content(item)
            content_key = self._cache_key(content)

            if content_key in self._cache:
                try:
                    from aragora.core.embeddings.service import cosine_similarity
                    sim = cosine_similarity(self._query_embedding, self._cache[content_key])
                    # Normalize from [-1, 1] to [0, 1]
                    return (sim + 1) / 2
                except Exception:
                    pass

        # Fallback to keyword matching
        return self._keyword_score(item, query)

    def _keyword_score(self, item: Any, query: str) -> float:
        """Fallback keyword-based scoring."""
        content = self._get_content(item).lower()
        query_terms = query.lower().split()
        if not query_terms:
            return 0.5

        matches = sum(1 for term in query_terms if term in content)
        return min(1.0, matches / len(query_terms))

    async def score_async(self, item: Any, query: str) -> float:
        """
        Score item relevance using embeddings (async interface).

        Args:
            item: The item to score
            query: The search query

        Returns:
            Relevance score between 0 and 1
        """
        if not self._initialized:
            await self.initialize()

        if not self._provider:
            return self._keyword_score(item, query)

        # Get query embedding (use cached if available)
        if query != self._current_query:
            await self.prepare_query(query)

        if not self._query_embedding:
            return self._keyword_score(item, query)

        # Get content embedding
        content = self._get_content(item)
        content_embedding = await self._get_embedding(content)

        if not content_embedding:
            return self._keyword_score(item, query)

        try:
            from aragora.core.embeddings.service import cosine_similarity
            sim = cosine_similarity(self._query_embedding, content_embedding)
            # Normalize from [-1, 1] to [0, 1]
            return (sim + 1) / 2
        except Exception as e:
            logger.debug(f"Cosine similarity failed: {e}")
            return self._keyword_score(item, query)


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

        # Record Prometheus metrics
        self._record_prometheus_metrics(result)

        return result

    def _record_prometheus_metrics(self, result: FederatedQueryResult) -> None:
        """Record federated query metrics to Prometheus."""
        try:
            from aragora.observability.metrics import (
                record_km_federated_query,
                set_km_active_adapters,
            )

            success = len(result.sources_succeeded) > 0
            record_km_federated_query(len(result.sources_queried), success)
            set_km_active_adapters(len([r for r in self._adapters.values() if r.enabled]))
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to record Prometheus metrics: {e}")

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

        Uses simple keyword matching as fallback when embeddings unavailable.
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
    "EmbeddingRelevanceScorer",
]
