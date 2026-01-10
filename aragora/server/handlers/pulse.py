"""
Pulse and trending topics endpoint handlers.

Endpoints:
- GET /api/pulse/trending - Get trending topics from multiple sources
- GET /api/pulse/suggest - Suggest a trending topic for debate
- GET /api/pulse/analytics - Get analytics on trending topic debate outcomes
"""

from __future__ import annotations

import logging
from typing import Optional

from aragora.config import DB_TIMEOUT_SECONDS
from aragora.server.http_utils import run_async

logger = logging.getLogger(__name__)
from .base import (
    BaseHandler, HandlerResult, json_response, error_response,
    get_int_param, get_string_param, validate_path_segment, SAFE_ID_PATTERN,
    feature_unavailable_response, auto_error_response,
)


# Shared PulseManager singleton for analytics tracking
# This allows FeedbackPhase to record outcomes that persist across requests
_shared_pulse_manager = None


def get_pulse_manager():
    """Get or create the shared PulseManager singleton.

    Returns:
        PulseManager instance for recording and retrieving analytics
    """
    global _shared_pulse_manager
    if _shared_pulse_manager is None:
        try:
            from aragora.pulse.ingestor import (
                PulseManager, TwitterIngestor, HackerNewsIngestor, RedditIngestor
            )
            _shared_pulse_manager = PulseManager()
            _shared_pulse_manager.add_ingestor("hackernews", HackerNewsIngestor())
            _shared_pulse_manager.add_ingestor("reddit", RedditIngestor())
            _shared_pulse_manager.add_ingestor("twitter", TwitterIngestor())
        except ImportError:
            return None
    return _shared_pulse_manager


class PulseHandler(BaseHandler):
    """Handler for pulse/trending topic endpoints."""

    ROUTES = [
        "/api/pulse/trending",
        "/api/pulse/suggest",
        "/api/pulse/analytics",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route pulse requests to appropriate methods."""
        if path == "/api/pulse/trending":
            limit = get_int_param(query_params, 'limit', 10)
            return self._get_trending_topics(min(limit, 50))

        if path == "/api/pulse/suggest":
            category = get_string_param(query_params, 'category')
            if category:
                is_valid, err = validate_path_segment(category, "category", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
            return self._suggest_debate_topic(category)

        if path == "/api/pulse/analytics":
            return self._get_analytics()

        return None

    def _run_async_safely(self, coro_factory, timeout: float = None) -> list:
        """Run an async coroutine safely, handling event loop edge cases.

        Uses run_async() from http_utils which properly handles:
        1. No running event loop - uses asyncio.run() directly
        2. Running event loop - uses ThreadPoolExecutor to avoid nested loop

        Args:
            coro_factory: Callable that returns a coroutine (called inside executor)
            timeout: Optional timeout in seconds (defaults to DB_TIMEOUT_SECONDS)

        Returns:
            Result from coroutine, or empty list on failure
        """
        try:
            return run_async(coro_factory())
        except Exception as e:
            logger.warning("Async fetch failed: %s", e)
            return []

    def _get_trending_topics(self, limit: int) -> HandlerResult:
        """Get trending topics from multiple pulse ingestors.

        Uses real-time data sources:
        - Hacker News: Front page stories via Algolia API (free, no auth required)
        - Reddit: Hot posts from tech/science subreddits (public JSON API)
        - Twitter: Requires API key, falls back to mock data if not configured

        Response maps internal fields to frontend expectations:
        - platform → source
        - volume → score (normalized 0-1)
        """
        try:
            from aragora.pulse.ingestor import (
                PulseManager, TwitterIngestor, HackerNewsIngestor, RedditIngestor
            )
        except ImportError:
            return feature_unavailable_response("pulse")

        try:
            # Create manager with multiple real ingestors
            manager = PulseManager()
            manager.add_ingestor("hackernews", HackerNewsIngestor())
            manager.add_ingestor("reddit", RedditIngestor())
            manager.add_ingestor("twitter", TwitterIngestor())

            # Fetch trending topics asynchronously from all sources
            async def fetch():
                return await manager.get_trending_topics(limit_per_platform=limit)

            topics = self._run_async_safely(fetch)

            # Normalize scores: find max volume and scale to 0-1
            max_volume = max((t.volume for t in topics), default=1) or 1

            return json_response({
                "topics": [
                    {
                        "topic": t.topic,
                        "source": t.platform,  # Map platform → source for frontend
                        "score": round(t.volume / max_volume, 3),  # Normalized 0-1 score
                        "volume": t.volume,  # Keep raw volume for reference
                        "category": t.category,
                    }
                    for t in topics
                ],
                "count": len(topics),
                "sources": list(manager.ingestors.keys()),
            })

        except Exception as e:
            return error_response(f"Failed to fetch trending topics: {e}", 500)

    def _suggest_debate_topic(self, category: str | None = None) -> HandlerResult:
        """Suggest a trending topic for debate.

        Args:
            category: Optional category filter (tech, ai, science, etc.)

        Returns topic suitable for debate with prompt formatting.
        """
        try:
            from aragora.pulse.ingestor import (
                PulseManager, TwitterIngestor, HackerNewsIngestor, RedditIngestor
            )
        except ImportError:
            return feature_unavailable_response("pulse")

        try:
            # Create manager with ingestors
            manager = PulseManager()
            manager.add_ingestor("hackernews", HackerNewsIngestor())
            manager.add_ingestor("reddit", RedditIngestor())
            manager.add_ingestor("twitter", TwitterIngestor())

            # Fetch trending topics
            async def fetch():
                filters = {"categories": [category]} if category else None
                return await manager.get_trending_topics(limit_per_platform=10, filters=filters)

            topics = self._run_async_safely(fetch)

            # Select best topic for debate
            selected = manager.select_topic_for_debate(topics)

            if not selected:
                return json_response({
                    "topic": None,
                    "message": "No suitable topics found",
                }, status=404)

            return json_response({
                "topic": selected.topic,
                "debate_prompt": selected.to_debate_prompt(),
                "source": selected.platform,
                "category": selected.category,
                "volume": selected.volume,
            })

        except Exception as e:
            return error_response(f"Failed to suggest debate topic: {e}", 500)

    @auto_error_response("get pulse analytics")
    def _get_analytics(self) -> HandlerResult:
        """Get analytics on trending topic debate outcomes.

        Returns analytics data including:
        - total_debates: Total number of debates with trending topics
        - consensus_rate: Percentage that reached consensus
        - avg_confidence: Average confidence score
        - by_platform: Breakdown by source platform
        - by_category: Breakdown by topic category
        - recent_outcomes: Last 10 debate outcomes
        """
        manager = get_pulse_manager()
        if not manager:
            return feature_unavailable_response("pulse")

        analytics = manager.get_analytics()
        return json_response(analytics)
