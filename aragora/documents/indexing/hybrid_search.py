"""
Hybrid search combining BM25 keyword search with vector similarity.

Implements Reciprocal Rank Fusion (RRF) to combine results from
different retrieval methods for improved recall and precision.

Requirements:
    pip install weaviate-client

Usage:
    from aragora.documents.indexing.hybrid_search import HybridSearcher

    searcher = HybridSearcher(store, embedder)
    results = await searcher.search("query", limit=10)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from aragora.documents.indexing.weaviate_store import (
    WeaviateStore,
    SearchResult,
)

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...


@dataclass
class HybridResult:
    """A single result from hybrid search with combined scoring."""

    chunk_id: str
    document_id: str
    content: str
    combined_score: float
    vector_score: float
    keyword_score: float
    chunk_type: str = "text"
    heading_context: str = ""
    start_page: int = 0
    end_page: int = 0
    rank_vector: int = 0
    rank_keyword: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "combined_score": self.combined_score,
            "vector_score": self.vector_score,
            "keyword_score": self.keyword_score,
            "chunk_type": self.chunk_type,
            "heading_context": self.heading_context,
            "start_page": self.start_page,
            "end_page": self.end_page,
        }


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""

    # Weight for vector search (0-1), keyword weight = 1 - vector_weight
    vector_weight: float = 0.7
    keyword_weight: float = 0.3

    # RRF constant (k in 1/(k + rank))
    rrf_k: int = 60

    # Limits for individual searches before fusion
    vector_limit: int = 50
    keyword_limit: int = 50

    # Minimum score thresholds
    min_vector_score: float = 0.0
    min_keyword_score: float = 0.0
    min_combined_score: float = 0.0

    # Query expansion settings
    enable_query_expansion: bool = False
    max_expanded_terms: int = 5


class HybridSearcher:
    """
    Hybrid search engine combining BM25 and vector similarity.

    Uses Reciprocal Rank Fusion (RRF) to combine results from
    keyword and vector searches, providing better retrieval
    quality than either method alone.
    """

    def __init__(
        self,
        store: WeaviateStore,
        embedder: EmbeddingProvider,
        config: Optional[HybridSearchConfig] = None,
    ):
        """
        Initialize hybrid searcher.

        Args:
            store: Weaviate store for retrieval
            embedder: Embedding provider for query vectorization
            config: Search configuration
        """
        self.store = store
        self.embedder = embedder
        self.config = config or HybridSearchConfig()

    async def search(
        self,
        query: str,
        limit: int = 10,
        document_ids: Optional[list[str]] = None,
        vector_weight: Optional[float] = None,
    ) -> list[HybridResult]:
        """
        Perform hybrid search combining keyword and vector retrieval.

        Args:
            query: Search query text
            limit: Maximum results to return
            document_ids: Optional filter to specific documents
            vector_weight: Override default vector weight (0-1)

        Returns:
            List of hybrid search results
        """
        # Use provided weight or default
        v_weight = vector_weight if vector_weight is not None else self.config.vector_weight
        k_weight = 1.0 - v_weight

        # Generate query embedding
        query_embedding = await self.embedder.embed(query)

        # Run vector and keyword searches in parallel
        vector_task = self.store.search_vector(
            embedding=query_embedding,
            limit=self.config.vector_limit,
            document_ids=document_ids,
            min_score=self.config.min_vector_score,
        )
        keyword_task = self.store.search_keyword(
            query=query,
            limit=self.config.keyword_limit,
            document_ids=document_ids,
        )

        vector_results, keyword_results = await asyncio.gather(vector_task, keyword_task)

        # Fuse results using RRF
        fused = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
            vector_weight=v_weight,
            keyword_weight=k_weight,
        )

        # Filter by minimum combined score and limit
        filtered = [r for r in fused if r.combined_score >= self.config.min_combined_score]

        return filtered[:limit]

    async def search_vector_only(
        self,
        query: str,
        limit: int = 10,
        document_ids: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """
        Perform vector-only search.

        Args:
            query: Search query text
            limit: Maximum results to return
            document_ids: Optional filter to specific documents

        Returns:
            List of search results
        """
        query_embedding = await self.embedder.embed(query)
        return await self.store.search_vector(
            embedding=query_embedding,
            limit=limit,
            document_ids=document_ids,
        )

    async def search_keyword_only(
        self,
        query: str,
        limit: int = 10,
        document_ids: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """
        Perform keyword-only (BM25) search.

        Args:
            query: Search query text
            limit: Maximum results to return
            document_ids: Optional filter to specific documents

        Returns:
            List of search results
        """
        return await self.store.search_keyword(
            query=query,
            limit=limit,
            document_ids=document_ids,
        )

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[SearchResult],
        keyword_results: list[SearchResult],
        vector_weight: float,
        keyword_weight: float,
    ) -> list[HybridResult]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF score = weight * 1/(k + rank) for each retrieval method.
        This approach is robust to score calibration differences.

        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            vector_weight: Weight for vector scores
            keyword_weight: Weight for keyword scores

        Returns:
            Fused and ranked results
        """
        k = self.config.rrf_k

        # Build lookup maps
        chunk_data: dict[str, dict] = {}

        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result.chunk_id
            rrf_score = vector_weight * (1.0 / (k + rank))

            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = {
                    "result": result,
                    "vector_score": result.score,
                    "keyword_score": 0.0,
                    "rrf_vector": rrf_score,
                    "rrf_keyword": 0.0,
                    "rank_vector": rank,
                    "rank_keyword": 0,
                }
            else:
                chunk_data[chunk_id]["vector_score"] = result.score
                chunk_data[chunk_id]["rrf_vector"] = rrf_score
                chunk_data[chunk_id]["rank_vector"] = rank

        # Process keyword results
        for rank, result in enumerate(keyword_results, start=1):
            chunk_id = result.chunk_id
            rrf_score = keyword_weight * (1.0 / (k + rank))

            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = {
                    "result": result,
                    "vector_score": 0.0,
                    "keyword_score": result.score,
                    "rrf_vector": 0.0,
                    "rrf_keyword": rrf_score,
                    "rank_vector": 0,
                    "rank_keyword": rank,
                }
            else:
                chunk_data[chunk_id]["keyword_score"] = result.score
                chunk_data[chunk_id]["rrf_keyword"] = rrf_score
                chunk_data[chunk_id]["rank_keyword"] = rank

        # Build hybrid results with combined scores
        hybrid_results = []
        for chunk_id, data in chunk_data.items():
            combined_score = data["rrf_vector"] + data["rrf_keyword"]
            result = data["result"]

            hybrid_results.append(
                HybridResult(
                    chunk_id=chunk_id,
                    document_id=result.document_id,
                    content=result.content,
                    combined_score=combined_score,
                    vector_score=data["vector_score"],
                    keyword_score=data["keyword_score"],
                    chunk_type=result.chunk_type,
                    heading_context=result.heading_context,
                    start_page=result.start_page,
                    end_page=result.end_page,
                    rank_vector=data["rank_vector"],
                    rank_keyword=data["rank_keyword"],
                )
            )

        # Sort by combined score descending
        hybrid_results.sort(key=lambda x: x.combined_score, reverse=True)
        return hybrid_results

    async def rerank(
        self,
        query: str,
        results: list[HybridResult],
        reranker: Callable[[str, list[str]], list[float]],
    ) -> list[HybridResult]:
        """
        Rerank results using a cross-encoder or other reranker.

        Args:
            query: Original search query
            results: Initial search results
            reranker: Async function that takes (query, documents) and returns scores

        Returns:
            Reranked results
        """
        if not results:
            return results

        # Get content for reranking
        documents = [r.content for r in results]

        # Get rerank scores
        scores = await reranker(query, documents)

        # Update combined scores with rerank scores
        for result, score in zip(results, scores):
            result.combined_score = score
            result.metadata["rerank_score"] = score

        # Sort by new scores
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results


class SimpleEmbedder:
    """
    Simple embedding provider using aragora's existing embedding system.

    Falls back to a basic approach if embeddings aren't available.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self._embedder: Optional[Any] = None

    async def _get_embedder(self):
        """Lazily load the embedding provider."""
        if self._embedder is None:
            try:
                from aragora.memory.embeddings import get_embedder

                self._embedder = get_embedder(provider="openai", model=self.model)
            except ImportError:
                logger.warning("Embeddings module not available, using fallback")
                self._embedder = None
        return self._embedder

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        embedder = await self._get_embedder()
        if embedder:
            return await embedder.embed(text)
        else:
            # Fallback: simple bag-of-words style embedding
            return self._fallback_embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        embedder = await self._get_embedder()
        if embedder and hasattr(embedder, "embed_batch"):
            return await embedder.embed_batch(texts)
        else:
            # Fallback: embed one by one
            return [await self.embed(text) for text in texts]

    def _fallback_embed(self, text: str, dim: int = 1536) -> list[float]:
        """Simple fallback embedding using word hashing."""
        import hashlib

        # Hash-based pseudo-embedding (not for production use)
        words = text.lower().split()
        embedding = [0.0] * dim

        for word in words:
            h = hashlib.md5(word.encode(), usedforsecurity=False).hexdigest()
            for i in range(0, len(h), 2):
                idx = int(h[i : i + 2], 16) % dim
                embedding[idx] += 1.0

        # Normalize
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding


async def create_hybrid_searcher(
    store: Optional[WeaviateStore] = None,
    config: Optional[HybridSearchConfig] = None,
) -> HybridSearcher:
    """
    Create a hybrid searcher with default embedder.

    Args:
        store: Weaviate store (creates new one if not provided)
        config: Search configuration

    Returns:
        Configured HybridSearcher
    """
    from aragora.documents.indexing.weaviate_store import get_weaviate_store

    if store is None:
        store = get_weaviate_store()
        if not store.is_connected:
            await store.connect()

    embedder = SimpleEmbedder()
    return HybridSearcher(store, embedder, config)


__all__ = [
    "HybridSearcher",
    "HybridSearchConfig",
    "HybridResult",
    "SimpleEmbedder",
    "EmbeddingProvider",
    "create_hybrid_searcher",
]
