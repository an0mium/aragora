"""
RLM hierarchy cache.

Extracted from bridge.py for maintainability.
Provides RLMHierarchyCache for caching compression hierarchies.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .types import CompressionResult

from .types import AbstractionLevel, AbstractionNode, CompressionResult, RLMContext

logger = logging.getLogger(__name__)


# Import metrics for cache tracking (lazy to avoid circular imports)
def _record_rlm_cache_hit() -> None:
    try:
        from aragora.observability.metrics import record_rlm_cache_hit
        record_rlm_cache_hit()
    except ImportError:
        pass


def _record_rlm_cache_miss() -> None:
    try:
        from aragora.observability.metrics import record_rlm_cache_miss
        record_rlm_cache_miss()
    except ImportError:
        pass


class RLMHierarchyCache:
    """
    Cache for RLM compression hierarchies using Knowledge Mound.

    Stores compression results so they can be reused for similar tasks,
    avoiding expensive recompression of similar content.
    """

    def __init__(self, knowledge_mound: Optional[Any] = None):
        """
        Initialize the hierarchy cache.

        Args:
            knowledge_mound: KnowledgeMound instance for persistence
        """
        self.knowledge_mound = knowledge_mound
        self._local_cache: dict[str, "CompressionResult"] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _compute_task_hash(self, content: str, source_type: str = "text") -> str:
        """Compute a hash for the content to use as cache key."""
        import hashlib

        # Use first 2000 chars + length + source_type for hash
        # This balances uniqueness with collision tolerance
        key_content = f"{content[:2000]}|{len(content)}|{source_type}"
        return hashlib.sha256(key_content.encode()).hexdigest()[:32]

    async def get_cached(
        self,
        content: str,
        source_type: str = "text",
    ) -> Optional["CompressionResult"]:
        """
        Retrieve cached compression result if available.

        Args:
            content: Content that would be compressed
            source_type: Type of content (text, debate, code)

        Returns:
            Cached CompressionResult or None if not found
        """
        task_hash = self._compute_task_hash(content, source_type)

        # Check local cache first
        if task_hash in self._local_cache:
            self._cache_hits += 1
            _record_rlm_cache_hit()
            logger.debug(f"[RLMHierarchyCache] Local cache hit for {task_hash[:8]}")
            return self._local_cache[task_hash]

        # Check knowledge mound if available
        if self.knowledge_mound:
            try:
                # Query for cached hierarchy
                results = await self.knowledge_mound.query_semantic(
                    query=f"rlm_hierarchy:{task_hash}",
                    limit=1,
                    min_confidence=0.9,
                )

                if results and results.items:
                    item = results.items[0]
                    metadata = getattr(item, "metadata", {})

                    # Verify hash match
                    if metadata.get("task_hash") == task_hash:
                        # Deserialize compression result
                        cached_data = metadata.get("hierarchy_data")
                        if cached_data:
                            result = self._deserialize_compression(cached_data)
                            if result:
                                self._local_cache[task_hash] = result
                                self._cache_hits += 1
                                _record_rlm_cache_hit()
                                logger.debug(
                                    f"[RLMHierarchyCache] Knowledge Mound hit for {task_hash[:8]}"
                                )
                                return result

            except Exception as e:
                logger.debug(f"[RLMHierarchyCache] Mound query failed: {e}")

        self._cache_misses += 1
        _record_rlm_cache_miss()
        return None

    async def store(
        self,
        content: str,
        source_type: str,
        compression: "CompressionResult",
    ) -> None:
        """
        Store compression result in cache.

        Args:
            content: Original content that was compressed
            source_type: Type of content
            compression: The compression result to cache
        """
        task_hash = self._compute_task_hash(content, source_type)

        # Always store in local cache
        self._local_cache[task_hash] = compression

        # Store in knowledge mound if available
        if self.knowledge_mound:
            try:
                from aragora.knowledge.mound.types import IngestionRequest, KnowledgeSource

                # Serialize compression for storage
                serialized = self._serialize_compression(compression)

                await self.knowledge_mound.store(
                    IngestionRequest(
                        content=f"RLM compression hierarchy for {source_type} content",
                        source_type=KnowledgeSource.INTERNAL,
                        workspace_id=self.knowledge_mound.workspace_id,
                        metadata={
                            "task_hash": task_hash,
                            "source_type": source_type,
                            "compression_ratio": compression.compression_ratio.get(
                                AbstractionLevel.SUMMARY, 1.0
                            ),
                            "original_tokens": compression.original_tokens,
                            "estimated_fidelity": compression.estimated_fidelity,
                            "hierarchy_data": serialized,
                            "rlm_cache": True,  # Tag for identification
                        },
                    )
                )

                logger.debug(
                    f"[RLMHierarchyCache] Stored hierarchy {task_hash[:8]} "
                    f"(ratio={compression.compression_ratio}, fidelity={compression.estimated_fidelity:.2f})"
                )

            except Exception as e:
                logger.debug(f"[RLMHierarchyCache] Mound storage failed: {e}")

    def _serialize_compression(self, compression: "CompressionResult") -> dict:
        """Serialize compression result for storage."""
        # Serialize key topics and level summaries
        level_data = {}
        for level, nodes in compression.context.levels.items():
            level_data[level.value] = [
                {
                    "id": n.id,
                    "content": n.content[:1000],  # Truncate for storage efficiency
                    "token_count": n.token_count,
                    "key_topics": n.key_topics[:5] if n.key_topics else [],
                }
                for n in nodes[:10]  # Limit nodes per level
            ]

        return {
            "original_tokens": compression.original_tokens,
            "compressed_tokens": {k.value: v for k, v in compression.compressed_tokens.items()},
            "compression_ratio": {k.value: v for k, v in compression.compression_ratio.items()},
            "estimated_fidelity": compression.estimated_fidelity,
            "key_topics": compression.key_topics_extracted[:10],
            "levels": level_data,
        }

    def _deserialize_compression(self, data: dict) -> Optional["CompressionResult"]:
        """Deserialize compression result from storage."""
        try:
            # Rebuild context
            context = RLMContext(
                original_content="[cached - original not stored]",
                original_tokens=data.get("original_tokens", 0),
                source_type="cached",
            )

            # Rebuild levels
            for level_str, nodes_data in data.get("levels", {}).items():
                try:
                    level = AbstractionLevel(level_str)
                except ValueError:
                    continue

                nodes = []
                for nd in nodes_data:
                    node = AbstractionNode(
                        id=nd["id"],
                        level=level,
                        content=nd["content"],
                        token_count=nd["token_count"],
                        key_topics=nd.get("key_topics", []),
                    )
                    nodes.append(node)
                    context.nodes_by_id[node.id] = node

                context.levels[level] = nodes

            # Rebuild compression ratios
            compressed_tokens = {}
            compression_ratio = {}
            for level_str, val in data.get("compressed_tokens", {}).items():
                try:
                    level = AbstractionLevel(level_str)
                    compressed_tokens[level] = val
                except ValueError:
                    pass

            for level_str, val in data.get("compression_ratio", {}).items():
                try:
                    level = AbstractionLevel(level_str)
                    compression_ratio[level] = val
                except ValueError:
                    pass

            return CompressionResult(
                context=context,
                original_tokens=data.get("original_tokens", 0),
                compressed_tokens=compressed_tokens,
                compression_ratio=compression_ratio,
                time_seconds=0.0,  # Not stored
                sub_calls_made=0,  # Not stored
                cache_hits=1,  # This is a cache hit
                estimated_fidelity=data.get("estimated_fidelity", 0.8),
                key_topics_extracted=data.get("key_topics", []),
            )

        except Exception as e:
            logger.warning(f"[RLMHierarchyCache] Deserialization failed: {e}")
            return None

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total > 0 else 0.0,
            "local_cache_size": len(self._local_cache),
        }

    def clear_local(self) -> None:
        """Clear the local in-memory cache."""
        self._local_cache.clear()
        logger.debug("[RLMHierarchyCache] Local cache cleared")


__all__ = ["RLMHierarchyCache"]
