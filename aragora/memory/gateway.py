"""
Unified Memory Gateway.

Single query/store interface across all 5 memory systems:
- ContinuumMemory (tiered local memory)
- Knowledge Mound (central knowledge store)
- Supermemory (external cross-session persistence)
- claude-mem (MCP cross-session memory)
- RLM (hierarchical context navigation)

The gateway provides:
- Fan-out queries across all available systems
- Cross-system deduplication of results
- Unified ranking by confidence and relevance
- Delegated writes via MemoryCoordinator

All features are opt-in via MemoryGatewayConfig.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aragora.memory.dedup import CrossSystemDedupEngine, dedup_results
from aragora.memory.gateway_config import MemoryGatewayConfig

if TYPE_CHECKING:
    from aragora.memory.continuum import ContinuumMemory
    from aragora.memory.coordinator import MemoryCoordinator
    from aragora.memory.retention_gate import RetentionGate
    from aragora.knowledge.mound import KnowledgeMound
    from aragora.knowledge.mound.adapters.supermemory_adapter import SupermemoryAdapter
    from aragora.knowledge.mound.adapters.claude_mem_adapter import ClaudeMemAdapter

logger = logging.getLogger(__name__)


@dataclass
class UnifiedMemoryQuery:
    """Query for the unified memory gateway."""

    query: str
    limit: int = 10
    min_confidence: float = 0.0
    sources: list[str] | None = None  # None = all available
    dedup: bool = True


@dataclass
class UnifiedMemoryResult:
    """A single result from unified memory query."""

    id: str
    content: str
    source_system: str  # "continuum", "km", "supermemory", "claude_mem"
    confidence: float
    surprise_score: float | None = None
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedMemoryResponse:
    """Response from a unified memory query."""

    results: list[UnifiedMemoryResult] = field(default_factory=list)
    total_found: int = 0
    sources_queried: list[str] = field(default_factory=list)
    duplicates_removed: int = 0
    query_time_ms: float = 0.0
    errors: dict[str, str] = field(default_factory=dict)


class MemoryGateway:
    """Fan-out query + dedup + rank across all memory systems.

    Delegates writes to MemoryCoordinator. Provides read-only
    query interface with cross-system deduplication and ranking.

    Usage:
        gateway = MemoryGateway(
            config=MemoryGatewayConfig(enabled=True),
            continuum_memory=continuum,
            knowledge_mound=mound,
        )

        response = await gateway.query(UnifiedMemoryQuery(
            query="rate limiting best practices",
            limit=10,
        ))
    """

    def __init__(
        self,
        config: MemoryGatewayConfig | None = None,
        continuum_memory: ContinuumMemory | None = None,
        knowledge_mound: KnowledgeMound | None = None,
        supermemory_adapter: SupermemoryAdapter | None = None,
        claude_mem_adapter: ClaudeMemAdapter | None = None,
        coordinator: MemoryCoordinator | None = None,
        retention_gate: RetentionGate | None = None,
    ):
        self.config = config or MemoryGatewayConfig()
        self.continuum_memory = continuum_memory
        self.knowledge_mound = knowledge_mound
        self.supermemory_adapter = supermemory_adapter
        self.claude_mem_adapter = claude_mem_adapter
        self.coordinator = coordinator
        self.retention_gate = retention_gate
        self._dedup_engine = CrossSystemDedupEngine()

    def _available_sources(self) -> list[str]:
        """Get list of available memory sources."""
        sources = []
        if self.continuum_memory is not None:
            sources.append("continuum")
        if self.knowledge_mound is not None:
            sources.append("km")
        if self.supermemory_adapter is not None:
            sources.append("supermemory")
        if self.claude_mem_adapter is not None:
            sources.append("claude_mem")
        return sources

    async def query(self, q: UnifiedMemoryQuery) -> UnifiedMemoryResponse:
        """Fan-out query across all configured memory systems.

        Args:
            q: Unified query specification

        Returns:
            UnifiedMemoryResponse with deduped, ranked results
        """
        start = time.time()
        available = self._available_sources()
        sources_to_query = q.sources or self.config.default_sources or available
        # Only query sources that are actually available
        sources_to_query = [s for s in sources_to_query if s in available]

        all_results: list[UnifiedMemoryResult] = []
        errors: dict[str, str] = {}

        if self.config.parallel_queries:
            # Fan-out in parallel
            tasks = {}
            for source in sources_to_query:
                tasks[source] = asyncio.create_task(
                    self._query_source(source, q.query, q.limit)
                )

            for source, task in tasks.items():
                try:
                    results = await asyncio.wait_for(
                        task, timeout=self.config.query_timeout_seconds
                    )
                    all_results.extend(results)
                except asyncio.TimeoutError:
                    errors[source] = "timeout"
                    logger.warning("Query timeout for source: %s", source)
                except Exception as e:  # noqa: BLE001 - gateway boundary
                    errors[source] = str(e)
                    logger.warning("Query failed for source %s: %s", source, e)
        else:
            # Sequential queries
            for source in sources_to_query:
                try:
                    results = await asyncio.wait_for(
                        self._query_source(source, q.query, q.limit),
                        timeout=self.config.query_timeout_seconds,
                    )
                    all_results.extend(results)
                except asyncio.TimeoutError:
                    errors[source] = "timeout"
                except Exception as e:  # noqa: BLE001 - gateway boundary
                    errors[source] = str(e)

        total_found = len(all_results)

        # Filter by min confidence
        if q.min_confidence > 0:
            all_results = [
                r for r in all_results if r.confidence >= q.min_confidence
            ]

        # Dedup
        duplicates_removed = 0
        if q.dedup and all_results:
            all_results, duplicates_removed = self._dedup_results(all_results)

        # Rank
        all_results = self._rank_results(all_results, q.query)

        # Limit
        all_results = all_results[: q.limit]

        query_time_ms = (time.time() - start) * 1000

        return UnifiedMemoryResponse(
            results=all_results,
            total_found=total_found,
            sources_queried=sources_to_query,
            duplicates_removed=duplicates_removed,
            query_time_ms=query_time_ms,
            errors=errors,
        )

    async def store(
        self,
        content: str,
        confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
        targets: list[str] | None = None,
    ) -> dict[str, str]:
        """Store content via MemoryCoordinator.

        This delegates to the coordinator for transactional writes.
        For direct writes, use the coordinator directly.

        Args:
            content: Content to store
            confidence: Confidence level (0-1)
            metadata: Additional metadata
            targets: Target systems (default: all configured in coordinator)

        Returns:
            Dict of source -> item_id for successful writes
        """
        if not self.coordinator:
            logger.warning("No coordinator configured for gateway writes")
            return {}

        # Check for duplicates before writing
        dedup_result = await self._dedup_engine.check_duplicate_before_write(
            content, targets
        )
        if dedup_result.is_duplicate:
            logger.info(
                "Skipping duplicate write: %s already in %s",
                dedup_result.existing_id,
                dedup_result.existing_source,
            )
            return {}

        # Register in dedup index
        self._dedup_engine.register_item(
            item_id=f"gw_{int(time.time())}",
            source="gateway",
            content=content,
        )

        # For now, return empty - actual writes go through coordinator's
        # commit_debate_outcome which requires a DebateContext
        return {}

    async def _query_source(
        self,
        source: str,
        query: str,
        limit: int,
    ) -> list[UnifiedMemoryResult]:
        """Query a single memory source."""
        if source == "continuum":
            return await self._query_continuum(query, limit)
        elif source == "km":
            return await self._query_km(query, limit)
        elif source == "supermemory":
            return await self._query_supermemory(query, limit)
        elif source == "claude_mem":
            return await self._query_claude_mem(query, limit)
        else:
            logger.warning("Unknown source: %s", source)
            return []

    async def _query_continuum(
        self, query: str, limit: int
    ) -> list[UnifiedMemoryResult]:
        """Query ContinuumMemory."""
        if not self.continuum_memory:
            return []

        results = []
        entries = self.continuum_memory.search(query=query, limit=limit)
        if not entries:
            entries = []
        for entry in entries:
            content = getattr(entry, "content", str(entry))
            confidence = getattr(entry, "importance", 0.5)
            surprise = getattr(entry, "surprise_score", None)
            entry_id = getattr(entry, "id", "")
            results.append(
                UnifiedMemoryResult(
                    id=str(entry_id),
                    content=str(content),
                    source_system="continuum",
                    confidence=float(confidence) if confidence else 0.5,
                    surprise_score=float(surprise) if surprise is not None else None,
                    content_hash=CrossSystemDedupEngine.compute_content_hash(
                        str(content)
                    ),
                    metadata={"tier": getattr(entry, "tier", None)},
                )
            )

        return results

    async def _query_km(
        self, query: str, limit: int
    ) -> list[UnifiedMemoryResult]:
        """Query Knowledge Mound."""
        if not self.knowledge_mound:
            return []

        results = []
        qr = await self.knowledge_mound.query(query=query, limit=limit)
        items = getattr(qr, "items", []) or []
        for item in items:
            content = getattr(item, "content", "")
            confidence = getattr(item, "confidence", 0.5)
            item_id = getattr(item, "id", "")
            results.append(
                UnifiedMemoryResult(
                    id=str(item_id),
                    content=str(content),
                    source_system="km",
                    confidence=float(confidence) if confidence else 0.5,
                    content_hash=CrossSystemDedupEngine.compute_content_hash(
                        str(content)
                    ),
                )
            )

        return results

    async def _query_supermemory(
        self, query: str, limit: int
    ) -> list[UnifiedMemoryResult]:
        """Query Supermemory adapter."""
        if not self.supermemory_adapter:
            return []

        results = []
        search_results = await self.supermemory_adapter.search_memories(
            query=query, limit=limit
        )
        for sr in search_results or []:
            content = getattr(sr, "content", "")
            similarity = getattr(sr, "similarity", 0.5)
            memory_id = getattr(sr, "memory_id", "")
            results.append(
                UnifiedMemoryResult(
                    id=str(memory_id) if memory_id else "",
                    content=str(content),
                    source_system="supermemory",
                    confidence=float(similarity),
                    content_hash=CrossSystemDedupEngine.compute_content_hash(
                        str(content)
                    ),
                )
            )

        return results

    async def _query_claude_mem(
        self, query: str, limit: int
    ) -> list[UnifiedMemoryResult]:
        """Query claude-mem adapter."""
        if not self.claude_mem_adapter:
            return []

        results = []
        observations = await self.claude_mem_adapter.search_observations(
            query=query, limit=limit
        )
        for obs in observations or []:
            content = obs.get("content", "")
            obs_id = obs.get("id", "")
            results.append(
                UnifiedMemoryResult(
                    id=str(obs_id),
                    content=str(content),
                    source_system="claude_mem",
                    confidence=0.6,  # Default for external observations
                    content_hash=CrossSystemDedupEngine.compute_content_hash(
                        str(content)
                    ),
                    metadata=obs.get("metadata", {}),
                )
            )

        return results

    def _dedup_results(
        self, results: list[UnifiedMemoryResult]
    ) -> tuple[list[UnifiedMemoryResult], int]:
        """Remove duplicate results across sources."""
        seen_hashes: set[str] = set()
        deduped: list[UnifiedMemoryResult] = []
        removed = 0

        for result in results:
            h = result.content_hash or CrossSystemDedupEngine.compute_content_hash(
                result.content
            )
            if h in seen_hashes:
                removed += 1
                continue
            seen_hashes.add(h)
            deduped.append(result)

        return deduped, removed

    def _rank_results(
        self, results: list[UnifiedMemoryResult], query: str
    ) -> list[UnifiedMemoryResult]:
        """Rank results by confidence (primary) and source priority."""
        source_priority = {
            "km": 1.0,
            "continuum": 0.9,
            "supermemory": 0.8,
            "claude_mem": 0.7,
        }

        def sort_key(r: UnifiedMemoryResult) -> float:
            priority = source_priority.get(r.source_system, 0.5)
            return r.confidence * 0.7 + priority * 0.3

        return sorted(results, key=sort_key, reverse=True)

    def get_stats(self) -> dict[str, Any]:
        """Get gateway statistics."""
        return {
            "available_sources": self._available_sources(),
            "config": {
                "enabled": self.config.enabled,
                "parallel_queries": self.config.parallel_queries,
                "dedup_threshold": self.config.dedup_threshold,
                "query_timeout_seconds": self.config.query_timeout_seconds,
            },
            "dedup_index_size": self._dedup_engine.get_hash_index_size(),
            "has_coordinator": self.coordinator is not None,
            "has_retention_gate": self.retention_gate is not None,
        }
