"""
Hybrid Memory Search - Combine vector + keyword search for ContinuumMemory.

Uses Reciprocal Rank Fusion (RRF) to merge results from vector similarity
and keyword (FTS5) searches for improved retrieval quality.

Adapted from aragora/documents/indexing/hybrid_search.py pattern.

Example:
    from aragora.memory.hybrid_search import HybridMemorySearch

    search = HybridMemorySearch(continuum_memory)
    results = await search.search("circuit breaker pattern", limit=10)
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:
    from aragora.memory.continuum import ContinuumMemory

logger = logging.getLogger(__name__)

__all__ = [
    "HybridMemorySearch",
    "HybridMemoryConfig",
    "MemorySearchResult",
    "KeywordIndex",
    "get_hybrid_memory_search",
]


@dataclass
class MemorySearchResult:
    """A search result from hybrid memory search."""

    memory_id: str
    content: str
    tier: str
    importance: float
    combined_score: float
    vector_score: float
    keyword_score: float
    vector_rank: int = 0
    keyword_rank: int = 0
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "tier": self.tier,
            "importance": self.importance,
            "combined_score": self.combined_score,
            "vector_score": self.vector_score,
            "keyword_score": self.keyword_score,
            "vector_rank": self.vector_rank,
            "keyword_rank": self.keyword_rank,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class HybridMemoryConfig:
    """Configuration for hybrid memory search."""

    # RRF constant (k in 1/(k + rank))
    rrf_k: int = 60

    # Weight for vector vs keyword (must sum to 1.0)
    vector_weight: float = 0.6
    keyword_weight: float = 0.4

    # Limits for individual searches before fusion
    vector_limit: int = 50
    keyword_limit: int = 50

    # Minimum score thresholds (0-1)
    min_combined_score: float = 0.0

    # Tier filtering
    tiers: Optional[list[str]] = None  # None means all tiers

    # Importance threshold
    min_importance: float = 0.0


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        ...


class KeywordIndex:
    """
    SQLite FTS5 keyword index for memory content.

    Provides full-text search capabilities alongside the ContinuumMemory store.
    Creates a virtual FTS5 table for efficient keyword matching.
    """

    def __init__(self, db_path: str | Path):
        """
        Initialize keyword index.

        Args:
            db_path: Path to SQLite database (should match ContinuumMemory db)
        """
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_fts_table()

    def _get_connection(self) -> sqlite3.Connection:
        """Get SQLite connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_fts_table(self) -> None:
        """Create FTS5 virtual table if it doesn't exist."""
        try:
            conn = self._get_connection()
            # Use a standalone FTS5 table (not content-synced) for simplicity
            # This avoids trigger conflicts and allows manual sync via rebuild_index
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    memory_id,
                    content
                )
            """)
            conn.commit()
        except sqlite3.Error as e:
            logger.warning(f"Failed to create FTS table: {e}")

    def rebuild_index(self) -> int:
        """
        Rebuild the FTS index from the main table.

        Returns:
            Number of entries indexed
        """
        conn = self._get_connection()
        try:
            # Clear existing FTS data
            conn.execute("DELETE FROM memory_fts")

            # Repopulate from main table
            conn.execute("""
                INSERT INTO memory_fts(memory_id, content)
                SELECT id, content FROM continuum_memory
            """)
            conn.commit()

            # Get count
            cursor = conn.execute("SELECT COUNT(*) FROM memory_fts")
            count = cursor.fetchone()[0]
            logger.info(f"Rebuilt FTS index with {count} entries")
            return count
        except sqlite3.Error as e:
            logger.error(f"Failed to rebuild FTS index: {e}")
            return 0

    def search(
        self,
        query: str,
        limit: int = 50,
        tiers: Optional[list[str]] = None,
        min_importance: float = 0.0,
    ) -> list[tuple[str, str, float, str, float]]:
        """
        Search for memories matching keywords.

        Uses FTS5 BM25 ranking for relevance scoring.

        Args:
            query: Search query (supports FTS5 query syntax)
            limit: Maximum results
            tiers: Optional tier filter
            min_importance: Minimum importance threshold

        Returns:
            List of (id, content, bm25_score, tier, importance) tuples
        """
        conn = self._get_connection()

        # Escape special FTS5 characters and build query
        safe_query = self._escape_fts_query(query)

        try:
            # Join with main table to get tier and importance
            sql = """
                SELECT
                    fts.memory_id,
                    cm.content,
                    bm25(memory_fts) as score,
                    cm.tier,
                    cm.importance
                FROM memory_fts fts
                JOIN continuum_memory cm ON fts.memory_id = cm.id
                WHERE memory_fts MATCH ?
                    AND cm.importance >= ?
            """
            params: list[Any] = [safe_query, min_importance]

            if tiers:
                placeholders = ",".join("?" * len(tiers))
                sql += f" AND cm.tier IN ({placeholders})"
                params.extend(tiers)

            sql += " ORDER BY score LIMIT ?"
            params.append(limit)

            cursor = conn.execute(sql, params)
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.warning(f"FTS search failed: {e}")
            return []

    def _escape_fts_query(self, query: str) -> str:
        """Escape special FTS5 characters in query."""
        # Split into words and search for any match (OR semantics)
        # Strip potentially problematic characters
        cleaned = query.replace('"', "").replace("'", "").replace("*", "")
        words = cleaned.split()

        if not words:
            return '""'

        # Use OR to find any matching term
        # Quote each word to handle special characters
        terms = [f'"{word}"' for word in words if word]
        return " OR ".join(terms)

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


class HybridMemorySearch:
    """
    Hybrid search combining vector similarity and keyword search for memories.

    Uses Reciprocal Rank Fusion (RRF) to combine results, which is robust
    to score calibration differences between retrieval methods.
    """

    def __init__(
        self,
        continuum_memory: "ContinuumMemory",
        embedder: Optional[EmbeddingProvider] = None,
        config: Optional[HybridMemoryConfig] = None,
    ):
        """
        Initialize hybrid memory search.

        Args:
            continuum_memory: The ContinuumMemory instance to search
            embedder: Optional embedding provider for vector search
            config: Search configuration
        """
        self.memory = continuum_memory
        self.embedder = embedder
        self.config = config or HybridMemoryConfig()

        # Initialize keyword index using same database
        self._keyword_index = KeywordIndex(continuum_memory.db_path)

    @property
    def keyword_index(self) -> KeywordIndex:
        """Get the keyword index."""
        return self._keyword_index

    async def search(
        self,
        query: str,
        limit: int = 10,
        tiers: Optional[list[str]] = None,
        vector_weight: Optional[float] = None,
        min_importance: float = 0.0,
    ) -> list[MemorySearchResult]:
        """
        Perform hybrid search combining vector and keyword retrieval.

        Args:
            query: Search query text
            limit: Maximum results to return
            tiers: Optional tier filter (e.g., ["slow", "glacial"])
            vector_weight: Override default vector weight (0-1)
            min_importance: Minimum importance threshold

        Returns:
            List of search results sorted by combined score
        """
        # Use provided weight or default
        v_weight = vector_weight if vector_weight is not None else self.config.vector_weight
        k_weight = 1.0 - v_weight

        effective_tiers = tiers or self.config.tiers

        # Run searches in parallel
        vector_task = self._vector_search(query, effective_tiers, min_importance)
        keyword_task = self._keyword_search(query, effective_tiers, min_importance)

        vector_results, keyword_results = await asyncio.gather(vector_task, keyword_task)

        # Fuse results using RRF
        fused = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
            vector_weight=v_weight,
            keyword_weight=k_weight,
        )

        # Filter by minimum combined score
        filtered = [r for r in fused if r.combined_score >= self.config.min_combined_score]

        return filtered[:limit]

    async def _vector_search(
        self,
        query: str,
        tiers: Optional[list[str]],
        min_importance: float,
    ) -> list[tuple[str, str, float, str, float]]:
        """
        Perform vector similarity search.

        Uses Knowledge Mound adapter if available, otherwise returns empty.

        Returns:
            List of (id, content, score, tier, importance) tuples
        """
        # Try using KM adapter for vector search
        if self.memory._km_adapter:
            try:
                similar = self.memory.query_km_for_similar(
                    content=query,
                    limit=self.config.vector_limit,
                    min_similarity=0.5,
                )
                # Convert KM results to our format
                results = []
                for item in similar:
                    # KM results may have different structure, adapt as needed
                    memory_id = item.get("id", item.get("memory_id", ""))
                    content = item.get("content", "")
                    score = item.get("similarity", item.get("score", 0.0))
                    tier = item.get("tier", "slow")
                    importance = item.get("importance", 0.5)

                    if importance >= min_importance:
                        if tiers is None or tier in tiers:
                            results.append((memory_id, content, score, tier, importance))
                return results
            except Exception as e:
                logger.debug(f"Vector search via KM failed: {e}")

        # Fallback: use embedder if available
        if self.embedder:
            try:
                # Would need vector store integration here
                # For now, return empty and rely on keyword search
                pass
            except Exception as e:
                logger.debug(f"Vector search via embedder failed: {e}")

        return []

    async def _keyword_search(
        self,
        query: str,
        tiers: Optional[list[str]],
        min_importance: float,
    ) -> list[tuple[str, str, float, str, float]]:
        """
        Perform keyword (FTS5) search.

        Returns:
            List of (id, content, score, tier, importance) tuples
        """
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            self._keyword_index.search,
            query,
            self.config.keyword_limit,
            tiers,
            min_importance,
        )
        return results

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[tuple[str, str, float, str, float]],
        keyword_results: list[tuple[str, str, float, str, float]],
        vector_weight: float,
        keyword_weight: float,
    ) -> list[MemorySearchResult]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF score = weight * 1/(k + rank) for each retrieval method.

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
        memory_data: dict[str, dict] = {}

        # Process vector results
        for rank, (memory_id, content, score, tier, importance) in enumerate(
            vector_results, start=1
        ):
            rrf_score = vector_weight * (1.0 / (k + rank))

            if memory_id not in memory_data:
                memory_data[memory_id] = {
                    "content": content,
                    "tier": tier,
                    "importance": importance,
                    "vector_score": score,
                    "keyword_score": 0.0,
                    "rrf_vector": rrf_score,
                    "rrf_keyword": 0.0,
                    "rank_vector": rank,
                    "rank_keyword": 0,
                }
            else:
                memory_data[memory_id]["vector_score"] = score
                memory_data[memory_id]["rrf_vector"] = rrf_score
                memory_data[memory_id]["rank_vector"] = rank

        # Process keyword results
        for rank, (memory_id, content, score, tier, importance) in enumerate(
            keyword_results, start=1
        ):
            rrf_score = keyword_weight * (1.0 / (k + rank))

            if memory_id not in memory_data:
                memory_data[memory_id] = {
                    "content": content,
                    "tier": tier,
                    "importance": importance,
                    "vector_score": 0.0,
                    "keyword_score": abs(score),  # BM25 scores are negative
                    "rrf_vector": 0.0,
                    "rrf_keyword": rrf_score,
                    "rank_vector": 0,
                    "rank_keyword": rank,
                }
            else:
                memory_data[memory_id]["keyword_score"] = abs(score)
                memory_data[memory_id]["rrf_keyword"] = rrf_score
                memory_data[memory_id]["rank_keyword"] = rank

        # Build search results with combined scores
        results = []
        for memory_id, data in memory_data.items():
            combined_score = data["rrf_vector"] + data["rrf_keyword"]

            results.append(
                MemorySearchResult(
                    memory_id=memory_id,
                    content=data["content"],
                    tier=data["tier"],
                    importance=data["importance"],
                    combined_score=combined_score,
                    vector_score=data["vector_score"],
                    keyword_score=data["keyword_score"],
                    vector_rank=data["rank_vector"],
                    keyword_rank=data["rank_keyword"],
                )
            )

        # Sort by combined score descending
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results

    async def search_keyword_only(
        self,
        query: str,
        limit: int = 10,
        tiers: Optional[list[str]] = None,
        min_importance: float = 0.0,
    ) -> list[MemorySearchResult]:
        """
        Perform keyword-only search.

        Args:
            query: Search query text
            limit: Maximum results
            tiers: Optional tier filter
            min_importance: Minimum importance threshold

        Returns:
            List of search results
        """
        results = await self._keyword_search(query, tiers, min_importance)

        return [
            MemorySearchResult(
                memory_id=memory_id,
                content=content,
                tier=tier,
                importance=importance,
                combined_score=abs(score),
                vector_score=0.0,
                keyword_score=abs(score),
                keyword_rank=rank,
            )
            for rank, (memory_id, content, score, tier, importance) in enumerate(
                results[:limit], start=1
            )
        ]

    async def search_vector_only(
        self,
        query: str,
        limit: int = 10,
        tiers: Optional[list[str]] = None,
        min_importance: float = 0.0,
    ) -> list[MemorySearchResult]:
        """
        Perform vector-only search.

        Args:
            query: Search query text
            limit: Maximum results
            tiers: Optional tier filter
            min_importance: Minimum importance threshold

        Returns:
            List of search results
        """
        results = await self._vector_search(query, tiers, min_importance)

        return [
            MemorySearchResult(
                memory_id=memory_id,
                content=content,
                tier=tier,
                importance=importance,
                combined_score=score,
                vector_score=score,
                keyword_score=0.0,
                vector_rank=rank,
            )
            for rank, (memory_id, content, score, tier, importance) in enumerate(
                results[:limit], start=1
            )
        ]

    def rebuild_keyword_index(self) -> int:
        """
        Rebuild the keyword search index.

        Useful after bulk data loading or if index gets out of sync.

        Returns:
            Number of entries indexed
        """
        return self._keyword_index.rebuild_index()

    def close(self) -> None:
        """Close resources."""
        self._keyword_index.close()


# Singleton instance
_hybrid_search: Optional[HybridMemorySearch] = None


def get_hybrid_memory_search(
    continuum_memory: Optional["ContinuumMemory"] = None,
    config: Optional[HybridMemoryConfig] = None,
) -> HybridMemorySearch:
    """
    Get the global HybridMemorySearch singleton.

    Args:
        continuum_memory: ContinuumMemory instance (required on first call)
        config: Optional search configuration

    Returns:
        HybridMemorySearch instance
    """
    global _hybrid_search

    if _hybrid_search is None:
        if continuum_memory is None:
            # Try to get default memory instance
            from aragora.memory.continuum import get_continuum_memory

            continuum_memory = get_continuum_memory()

        _hybrid_search = HybridMemorySearch(
            continuum_memory=continuum_memory,
            config=config,
        )

    return _hybrid_search
