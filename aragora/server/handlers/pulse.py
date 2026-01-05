"""
Pulse and trending topics endpoint handlers.

Endpoints:
- GET /api/pulse/trending - Get trending topics from multiple sources
"""

from typing import Optional
from .base import BaseHandler, HandlerResult, json_response, error_response, get_int_param


class PulseHandler(BaseHandler):
    """Handler for pulse/trending topic endpoints."""

    ROUTES = [
        "/api/pulse/trending",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route pulse requests to appropriate methods."""
        if path == "/api/pulse/trending":
            limit = get_int_param(query_params, 'limit', 10)
            return self._get_trending_topics(min(limit, 50))

        return None

    def _get_trending_topics(self, limit: int) -> HandlerResult:
        """Get trending topics from multiple pulse ingestors.

        Uses real-time data sources:
        - Hacker News: Front page stories via Algolia API (free, no auth required)
        - Reddit: Hot posts from tech/science subreddits (public JSON API)
        - Twitter: Requires API key, falls back to mock data if not configured
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

            return json_response({
                "topics": [
                    {
                        "topic": t.topic,
                        "platform": t.platform,
                        "volume": t.volume,
                        "category": t.category,
                    }
                    for t in topics
                ],
                "count": len(topics),
                "sources": list(manager.ingestors.keys()),
            })

        except Exception as e:
            return error_response(f"Failed to fetch trending topics: {e}", 500)
