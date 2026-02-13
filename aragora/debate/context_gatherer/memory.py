"""
ContinuumMemory context retrieval mixin for ContextGatherer.

Contains methods for retrieving relevant memories from ContinuumMemory
for debate context, including tier-aware retrieval and glacial insights.
"""

import logging
from typing import Any

from .constants import MAX_CONTINUUM_CACHE_SIZE

logger = logging.getLogger(__name__)


class MemoryMixin:
    """Mixin providing ContinuumMemory context retrieval methods."""

    # Type hints for attributes defined in main class
    _continuum_context_cache: dict[str, str]

    def _get_task_hash(self, task: str) -> str:
        """Generate a cache key from task to prevent cache leaks between debates."""
        raise NotImplementedError("Must be implemented by main class")

    def _enforce_cache_limit(self, cache: dict, max_size: int) -> None:
        """Enforce maximum cache size using FIFO eviction."""
        raise NotImplementedError("Must be implemented by main class")

    def get_continuum_context(
        self,
        continuum_memory: Any,
        domain: str,
        task: str,
        include_glacial_insights: bool = True,
        tenant_id: str | None = None,
        auth_context: Any | None = None,
    ) -> tuple[str, list[str], dict[str, Any]]:
        """Retrieve relevant memories from ContinuumMemory for debate context.

        Uses the debate task and domain to query for related past learnings.
        Enhanced with tier-aware retrieval, confidence markers, and glacial insights.

        Args:
            continuum_memory: ContinuumMemory instance to query
            domain: The debate domain (e.g., "programming", "ethics")
            task: The debate task description
            include_glacial_insights: Whether to include long-term glacial tier insights

        Returns:
            Tuple of:
            - Formatted context string
            - List of retrieved memory IDs (for outcome tracking)
            - Dict mapping memory ID to tier (for analytics)
        """
        # Check task-keyed cache first
        task_hash = self._get_task_hash(task)
        if hasattr(self, "_continuum_context_cache") and task_hash in self._continuum_context_cache:
            return self._continuum_context_cache[task_hash], [], {}

        if not continuum_memory:
            return "", [], {}

        try:
            try:
                from aragora.memory.access import (
                    emit_denial_telemetry,
                    filter_entries,
                    has_memory_read_access,
                )
            except Exception:
                emit_denial_telemetry = None  # type: ignore[assignment]
                filter_entries = None  # type: ignore[assignment]
                has_memory_read_access = None  # type: ignore[assignment]

            if (
                auth_context is not None
                and has_memory_read_access is not None
                and not has_memory_read_access(auth_context)
            ):
                if emit_denial_telemetry is not None:
                    emit_denial_telemetry(
                        "debate.context_gatherer.continuum",
                        auth_context,
                        "missing_memory_read_permission",
                    )
                return "", [], {}

            query = f"{domain}: {task[:200]}"
            all_memories = []
            retrieved_ids = []
            retrieved_tiers = {}

            # 1. Retrieve recent memories from fast/medium/slow tiers
            memories = continuum_memory.retrieve(
                query=query,
                limit=5,
                min_importance=0.3,
                include_glacial=False,  # Get recent memories first
                tenant_id=tenant_id,
            )
            all_memories.extend(memories)

            # 2. Also retrieve glacial tier insights for cross-session learning
            if include_glacial_insights:
                glacial_insights = []
                # Backwards-compatible API path used by existing tests/mocks.
                if hasattr(continuum_memory, "get_glacial_insights"):
                    try:
                        glacial_insights = continuum_memory.get_glacial_insights(
                            query=task[:100],
                            limit=3,
                            tenant_id=tenant_id,
                        )
                    except TypeError:
                        # Some implementations may not accept keyword args.
                        glacial_insights = continuum_memory.get_glacial_insights(task[:100], 3)
                else:
                    from aragora.memory.tier_manager import MemoryTier

                    glacial_insights = continuum_memory.retrieve(
                        query=task[:100],
                        tiers=[MemoryTier.GLACIAL],
                        limit=3,
                        min_importance=0.4,  # Higher threshold for long-term patterns
                        include_glacial=True,
                        tenant_id=tenant_id,
                    )
                if glacial_insights:
                    logger.info(
                        "  [continuum] Retrieved %s glacial insights for cross-session learning",
                        len(glacial_insights),
                    )
                    all_memories.extend(glacial_insights)

            if filter_entries is not None and auth_context is not None:
                all_memories = filter_entries(
                    all_memories,
                    auth_context,
                    source="debate.context_gatherer.continuum",
                )

            if not all_memories:
                return "", [], {}

            # Track retrieved memory IDs and tiers for outcome updates and analytics
            retrieved_ids = [
                getattr(mem, "id", None) for mem in all_memories if getattr(mem, "id", None)
            ]
            retrieved_tiers = {
                getattr(mem, "id", None): getattr(mem, "tier", None)
                for mem in all_memories
                if getattr(mem, "id", None) and getattr(mem, "tier", None)
            }

            # Format memories with confidence markers based on consolidation
            context_parts = ["[Previous learnings relevant to this debate:]"]

            # Format recent memories (fast/medium/slow)
            recent_mems = [
                m
                for m in all_memories
                if getattr(m, "tier", None) and getattr(m, "tier").value != "glacial"
            ]
            for mem in recent_mems[:3]:
                content = mem.content[:200] if hasattr(mem, "content") else str(mem)[:200]
                tier = mem.tier.value if hasattr(mem, "tier") else "unknown"
                consolidation = getattr(mem, "consolidation_score", 0.5)
                confidence = (
                    "high" if consolidation > 0.7 else "medium" if consolidation > 0.4 else "low"
                )
                context_parts.append(f"- [{tier}|{confidence}] {content}")

            # Format glacial insights separately (long-term patterns)
            glacial_mems = [
                m
                for m in all_memories
                if getattr(m, "tier", None) and getattr(m, "tier").value == "glacial"
            ]
            if glacial_mems:
                context_parts.append("\n[Long-term patterns from previous sessions:]")
                for mem in glacial_mems[:2]:
                    content = mem.content[:250] if hasattr(mem, "content") else str(mem)[:250]
                    consolidation = getattr(mem, "consolidation_score", 0.8)
                    context_parts.append(f"- [glacial|foundational] {content}")

            context = "\n".join(context_parts)
            self._enforce_cache_limit(self._continuum_context_cache, MAX_CONTINUUM_CACHE_SIZE)
            self._continuum_context_cache[task_hash] = context
            logger.info(
                "  [continuum] Retrieved %s recent + %s glacial memories for domain '%s'",
                len(recent_mems),
                len(glacial_mems),
                domain,
            )
            return context, retrieved_ids, retrieved_tiers
        except (AttributeError, TypeError, ValueError) as e:
            # Expected errors from memory system
            logger.warning("  [continuum] Memory retrieval error: %s", e)
            return "", [], {}
        except (KeyError, IndexError, RuntimeError, OSError) as e:
            # Unexpected error - log with more detail but don't crash debate
            logger.warning(
                "  [continuum] Unexpected memory error (type=%s): %s", type(e).__name__, e
            )
            return "", [], {}
