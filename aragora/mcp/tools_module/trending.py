"""
MCP Trending Tools.

Pulse integration for trending topic discovery.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


async def list_trending_topics_tool(
    platform: str = "all",
    category: str = "",
    min_score: float = 0.5,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Get trending topics from Pulse that could make good debates.

    Args:
        platform: Source platform ('hackernews', 'reddit', 'arxiv', 'all')
        category: Topic category filter
        min_score: Minimum topic score (0-1)
        limit: Max topics to return

    Returns:
        Dict with scored topics and count
    """
    topics: List[Dict[str, Any]] = []

    try:
        from aragora.pulse import PulseManager, SchedulerConfig, TopicSelector

        # Create pulse manager and fetch topics
        pulse_manager = PulseManager()
        platforms_list = [platform] if platform != "all" else None
        raw_topics = await pulse_manager.get_trending_topics(
            platforms=platforms_list,
            limit_per_platform=limit * 2,
        )

        # Score topics
        config = SchedulerConfig()
        selector = TopicSelector(config)

        for topic in raw_topics:
            if platform != "all" and topic.platform != platform:
                continue
            if category and getattr(topic, "category", "").lower() != category.lower():
                continue

            topic_score = selector.score_topic(topic)

            if topic_score.score >= min_score:
                topics.append(
                    {
                        "topic": topic.topic,
                        "platform": topic.platform,
                        "category": getattr(topic, "category", ""),
                        "score": round(topic_score.score, 3),
                        "volume": getattr(topic, "volume", 0),
                        "debate_potential": "high" if topic_score.score > 0.7 else "medium",
                    }
                )

        topics.sort(key=lambda x: x["score"], reverse=True)
        topics = topics[:limit]

    except ImportError:
        logger.warning("Pulse module not available")
    except Exception as e:
        logger.warning(f"Could not fetch trending topics: {e}")

    return {
        "topics": topics,
        "count": len(topics),
        "platform": platform,
        "category": category or "(all)",
        "min_score": min_score,
    }


__all__ = ["list_trending_topics_tool"]
