"""
Pulse and trending topics endpoint handlers.

Endpoints:
- GET /api/pulse/trending - Get trending topics from multiple sources
- GET /api/pulse/suggest - Suggest a trending topic for debate
"""

from typing import Optional
from .base import BaseHandler, HandlerResult, json_response, error_response, get_int_param


class PulseHandler(BaseHandler):
    """Handler for pulse/trending topic endpoints."""

    ROUTES = [
        "/api/pulse/trending",
        "/api/pulse/suggest",
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
            category = query_params.get('category')
            return self._suggest_debate_topic(category)

        return None

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
            return error_response("Pulse module not available", 503)

        try:
            import asyncio

            # Create manager with multiple real ingestors
            manager = PulseManager()
            manager.add_ingestor("hackernews", HackerNewsIngestor())
            manager.add_ingestor("reddit", RedditIngestor())
            manager.add_ingestor("twitter", TwitterIngestor())

            # Fetch trending topics asynchronously from all sources
            async def fetch():
                return await manager.get_trending_topics(limit_per_platform=limit)

            # Run in event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        topics = pool.submit(asyncio.run, fetch()).result()
                else:
                    topics = asyncio.run(fetch())
            except RuntimeError:
                topics = asyncio.run(fetch())

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

    def _suggest_debate_topic(self, category: str = None) -> HandlerResult:
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
            return error_response("Pulse module not available", 503)

        try:
            import asyncio

            # Create manager with ingestors
            manager = PulseManager()
            manager.add_ingestor("hackernews", HackerNewsIngestor())
            manager.add_ingestor("reddit", RedditIngestor())
            manager.add_ingestor("twitter", TwitterIngestor())

            # Fetch trending topics
            async def fetch():
                filters = {"categories": [category]} if category else None
                return await manager.get_trending_topics(limit_per_platform=10, filters=filters)

            # Run in event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        topics = pool.submit(asyncio.run, fetch()).result()
                else:
                    topics = asyncio.run(fetch())
            except RuntimeError:
                topics = asyncio.run(fetch())

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
