"""
Continuum Memory Glacial Operations.

Extracted from continuum.py for maintainability.
Provides glacial tier access for cross-session learning patterns.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.memory.continuum import ContinuumMemoryEntry
    from aragora.memory.tier_manager import MemoryTier

logger = logging.getLogger(__name__)


class ContinuumGlacialMixin:
    """
    Mixin providing glacial tier operations for ContinuumMemory.

    Enables cross-session learning by retrieving long-term patterns
    from the glacial tier (30-day half-life foundational knowledge).
    """

    # These must be provided by the main class
    hyperparams: Dict[str, Any]

    def connection(self) -> Any:
        """Get database connection context manager."""
        raise NotImplementedError

    def retrieve(
        self,
        query: Optional[str] = None,
        tiers: Optional[List["MemoryTier"]] = None,
        limit: int = 10,
        min_importance: float = 0.0,
        include_glacial: bool = True,
        tier: Optional[Any] = None,
    ) -> List["ContinuumMemoryEntry"]:
        """Retrieve memories - must be implemented by main class."""
        raise NotImplementedError

    def get_glacial_insights(
        self,
        topic: Optional[str] = None,
        limit: int = 10,
        min_importance: float = 0.3,
    ) -> List["ContinuumMemoryEntry"]:
        """
        Retrieve long-term patterns from the glacial tier for cross-session learning.

        The glacial tier stores foundational knowledge that persists across cycles
        (30-day half-life). This method provides targeted access to these insights
        for context gathering in debates and nomic cycles.

        Args:
            topic: Optional topic/query to filter relevant insights
            limit: Maximum entries to return (default 10)
            min_importance: Minimum importance threshold (default 0.3)

        Returns:
            List of glacial tier entries sorted by importance and relevance

        Example:
            # In debate orchestrator context phase:
            insights = await cms.get_glacial_insights_async(topic="error handling")
            for insight in insights:
                context.add_background(insight.content)
        """
        from aragora.memory.tier_manager import MemoryTier

        return self.retrieve(
            query=topic,
            tiers=[MemoryTier.GLACIAL],
            limit=limit,
            min_importance=min_importance,
            include_glacial=True,
        )

    async def get_glacial_insights_async(
        self,
        topic: Optional[str] = None,
        limit: int = 10,
        min_importance: float = 0.3,
    ) -> List["ContinuumMemoryEntry"]:
        """Async wrapper for get_glacial_insights() for use in async contexts."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_glacial_insights(
                topic=topic,
                limit=limit,
                min_importance=min_importance,
            ),
        )

    def get_cross_session_patterns(
        self,
        domain: Optional[str] = None,
        include_slow: bool = True,
        limit: int = 20,
    ) -> List["ContinuumMemoryEntry"]:
        """
        Get patterns that persist across sessions (slow + glacial tiers).

        Useful for:
        - Nomic loop context gathering (cross-cycle learning)
        - Agent team selection (historical performance patterns)
        - Debate strategy (recurring themes and successful approaches)

        Args:
            domain: Optional domain filter (e.g., 'code', 'research', 'creative')
            include_slow: Include slow tier in addition to glacial
            limit: Maximum entries to return

        Returns:
            Combined list from slow and glacial tiers, sorted by importance
        """
        from aragora.memory.tier_manager import MemoryTier

        tiers = [MemoryTier.GLACIAL]
        if include_slow:
            tiers.append(MemoryTier.SLOW)

        entries = self.retrieve(
            query=domain,
            tiers=tiers,
            limit=limit,
            min_importance=0.2,
            include_glacial=True,
        )

        # Sort by importance (highest first)
        return sorted(entries, key=lambda e: e.importance, reverse=True)

    async def get_cross_session_patterns_async(
        self,
        domain: Optional[str] = None,
        include_slow: bool = True,
        limit: int = 20,
    ) -> List["ContinuumMemoryEntry"]:
        """Async wrapper for get_cross_session_patterns()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_cross_session_patterns(
                domain=domain,
                include_slow=include_slow,
                limit=limit,
            ),
        )

    def get_glacial_tier_stats(self) -> Dict[str, Any]:
        """
        Get statistics specifically for the glacial tier.

        Useful for monitoring long-term memory health and deciding
        when to promote entries from slow to glacial tier.
        """
        from aragora.utils.json_helpers import safe_json_loads

        with self.connection() as conn:
            cursor = conn.cursor()

            # Get counts and averages for glacial tier
            cursor.execute(
                """
                SELECT
                    COUNT(*) as count,
                    AVG(importance) as avg_importance,
                    AVG(surprise_score) as avg_surprise,
                    AVG(consolidation_score) as avg_consolidation,
                    AVG(update_count) as avg_updates,
                    SUM(CASE WHEN red_line = 1 THEN 1 ELSE 0 END) as red_line_count,
                    MIN(created_at) as oldest_entry,
                    MAX(updated_at) as newest_update
                FROM continuum_memory
                WHERE tier = 'glacial'
                """
            )
            row = cursor.fetchone()

            # Get top tags in glacial tier
            cursor.execute(
                """
                SELECT metadata FROM continuum_memory
                WHERE tier = 'glacial'
                LIMIT 100
                """
            )
            tag_counts: Dict[str, int] = {}
            for (metadata_json,) in cursor.fetchall():
                metadata: dict[str, Any] = safe_json_loads(metadata_json, {})
                for tag in metadata.get("tags", []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "tier": "glacial",
            "count": row[0] or 0,
            "avg_importance": round(row[1] or 0, 3),
            "avg_surprise": round(row[2] or 0, 3),
            "avg_consolidation": round(row[3] or 0, 3),
            "avg_updates": round(row[4] or 0, 1),
            "red_line_count": row[5] or 0,
            "oldest_entry": row[6],
            "newest_update": row[7],
            "top_tags": [{"tag": t, "count": c} for t, c in top_tags],
            "max_entries": self.hyperparams["max_entries_per_tier"]["glacial"],
            "utilization": round(
                (row[0] or 0) / self.hyperparams["max_entries_per_tier"]["glacial"], 3
            ),
        }


__all__ = ["ContinuumGlacialMixin"]
