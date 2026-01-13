"""
MCP Memory Tools.

Continuum memory operations: query, store, pressure monitoring.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


async def query_memory_tool(
    query: str,
    tier: str = "all",
    limit: int = 10,
    min_importance: float = 0.0,
) -> Dict[str, Any]:
    """
    Query memories from the continuum memory system.

    Args:
        query: Search query for memory content
        tier: Memory tier (fast, medium, slow, glacial, all)
        limit: Max memories to return (1-100)
        min_importance: Minimum importance score (0-1)

    Returns:
        Dict with matching memories and count
    """
    limit = min(max(limit, 1), 100)
    memories: List[Dict[str, Any]] = []

    try:
        from aragora.memory.continuum import ContinuumMemory, MemoryTier

        continuum = ContinuumMemory()

        # Parse tier
        tiers = None
        if tier != "all":
            try:
                tiers = [MemoryTier[tier.upper()]]
            except KeyError:
                pass

        results = continuum.retrieve(
            query=query,
            tiers=tiers or list(MemoryTier),
            limit=limit,
            min_importance=min_importance,
        )

        for m in results:
            memories.append(
                {
                    "id": m.id,
                    "tier": m.tier.name.lower(),
                    "content": m.content[:500] + "..." if len(m.content) > 500 else m.content,
                    "importance": round(m.importance, 3),
                    "created_at": m.created_at if m.created_at else None,
                }
            )

    except ImportError:
        logger.warning("Continuum memory not available")
    except Exception as e:
        logger.warning(f"Memory query failed: {e}")

    return {
        "memories": memories,
        "count": len(memories),
        "query": query,
        "tier": tier,
    }


async def store_memory_tool(
    content: str,
    tier: str = "medium",
    importance: float = 0.5,
    tags: str = "",
) -> Dict[str, Any]:
    """
    Store a memory in the continuum memory system.

    Args:
        content: Memory content to store
        tier: Memory tier (fast, medium, slow, glacial)
        importance: Importance score (0-1)
        tags: Comma-separated tags

    Returns:
        Dict with stored memory ID and status
    """
    if not content:
        return {"error": "content is required"}

    try:
        from aragora.memory.continuum import ContinuumMemory, MemoryTier

        continuum = ContinuumMemory()

        # Parse tier
        try:
            memory_tier = MemoryTier[tier.upper()]
        except KeyError:
            memory_tier = MemoryTier.MEDIUM

        # Generate ID and store memory using the add method
        memory_id = f"mcp_{uuid.uuid4().hex[:12]}"
        continuum.add(
            id=memory_id,
            content=content,
            tier=memory_tier,
            importance=min(max(importance, 0.0), 1.0),
        )

        return {
            "success": True,
            "memory_id": memory_id,
            "tier": memory_tier.name.lower(),
            "importance": importance,
        }

    except ImportError:
        return {"error": "Continuum memory not available"}
    except Exception as e:
        return {"error": f"Failed to store memory: {e}"}


async def get_memory_pressure_tool() -> Dict[str, Any]:
    """
    Get current memory pressure and utilization.

    Returns:
        Dict with pressure score, status, and tier utilization
    """
    try:
        from aragora.memory.continuum import ContinuumMemory

        continuum = ContinuumMemory()
        pressure = continuum.get_memory_pressure()
        stats = continuum.get_stats()

        # Determine status
        if pressure < 0.5:
            status = "normal"
        elif pressure < 0.8:
            status = "elevated"
        elif pressure < 0.9:
            status = "high"
        else:
            status = "critical"

        return {
            "pressure": round(pressure, 3),
            "status": status,
            "total_memories": stats.get("total_memories", 0),
            "tier_stats": stats.get("by_tier", {}),
            "cleanup_recommended": pressure > 0.9,
        }

    except ImportError:
        return {"error": "Continuum memory not available"}
    except Exception as e:
        return {"error": f"Failed to get memory pressure: {e}"}


__all__ = [
    "query_memory_tool",
    "store_memory_tool",
    "get_memory_pressure_tool",
]
