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
    feature_unavailable_response, auto_error_response, ttl_cache, safe_error_message,
)


# Shared PulseManager singleton for analytics tracking
# This allows FeedbackPhase to record outcomes that persist across requests
import threading
_pulse_lock = threading.Lock()
_shared_pulse_manager = None


def get_pulse_manager():
    """Get or create the shared PulseManager singleton.

    Thread-safe initialization using double-checked locking.

    Returns:
        PulseManager instance for recording and retrieving analytics
    """
    global _shared_pulse_manager
    if _shared_pulse_manager is None:
        with _pulse_lock:
            # Double-check after acquiring lock
            if _shared_pulse_manager is None:
                try:
                    from aragora.pulse.ingestor import (
                        PulseManager, TwitterIngestor, HackerNewsIngestor, RedditIngestor
                    )
                    manager = PulseManager()
                    manager.add_ingestor("hackernews", HackerNewsIngestor())
                    manager.add_ingestor("reddit", RedditIngestor())
                    manager.add_ingestor("twitter", TwitterIngestor())
                    _shared_pulse_manager = manager
                except ImportError:
                    return None
    return _shared_pulse_manager


class PulseHandler(BaseHandler):
    """Handler for pulse/trending topic endpoints."""

    ROUTES = [
        "/api/pulse/trending",
        "/api/pulse/suggest",
        "/api/pulse/analytics",
        "/api/pulse/debate-topic",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route pulse requests to appropriate methods."""
        logger.debug(f"Pulse request: {path} params={query_params}")
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

    @ttl_cache(ttl_seconds=300, key_prefix="pulse_trending")
    def _get_trending_topics(self, limit: int) -> HandlerResult:
        """Get trending topics from multiple pulse ingestors.

        Uses real-time data sources:
        - Hacker News: Front page stories via Algolia API (free, no auth required)
        - Reddit: Hot posts from tech/science subreddits (public JSON API)
        - Twitter: Requires API key, falls back to mock data if not configured

        Response maps internal fields to frontend expectations:
        - platform → source
        - volume → score (normalized 0-1)

        Results are cached for 5 minutes to reduce API load.
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

            logger.info(f"Retrieved {len(topics)} trending topics from {len(manager.ingestors)} sources")
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
            return error_response(safe_error_message(e, "fetch trending topics"), 500)

    @ttl_cache(ttl_seconds=300, key_prefix="pulse_suggest")
    def _suggest_debate_topic(self, category: str | None = None) -> HandlerResult:
        """Suggest a trending topic for debate.

        Args:
            category: Optional category filter (tech, ai, science, etc.)

        Returns topic suitable for debate with prompt formatting.
        Results are cached for 5 minutes per category.
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
                logger.info("No suitable debate topic found in trending data")
                return json_response({
                    "topic": None,
                    "message": "No suitable topics found",
                }, status=404)

            logger.info(f"Suggested debate topic: '{selected.topic}' from {selected.platform}")
            return json_response({
                "topic": selected.topic,
                "debate_prompt": selected.to_debate_prompt(),
                "source": selected.platform,
                "category": selected.category,
                "volume": selected.volume,
            })

        except Exception as e:
            return error_response(safe_error_message(e, "suggest debate topic"), 500)

    @ttl_cache(ttl_seconds=60, key_prefix="pulse_analytics")
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

        Cached for 60 seconds for near-real-time updates.
        """
        manager = get_pulse_manager()
        if not manager:
            return feature_unavailable_response("pulse")

        analytics = manager.get_analytics()
        return json_response(analytics)

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Handle POST requests for pulse endpoints."""
        if path == "/api/pulse/debate-topic":
            return self._start_debate_on_topic(handler)
        return None

    def _start_debate_on_topic(self, handler) -> HandlerResult:
        """Start a debate on a trending topic.

        POST /api/pulse/debate-topic
        Body: {
            "topic": "The topic to debate",
            "agents": ["anthropic-api", "openai-api"],  // Optional
            "rounds": 3,  // Optional, default 3
            "consensus": "majority"  // Optional
        }

        Returns: {
            "debate_id": "...",
            "status": "started",
            "topic": "...",
            "agents": [...]
        }
        """
        import json as json_module

        try:
            # Read request body
            content_length = int(handler.headers.get('Content-Length', 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json_module.loads(body.decode('utf-8'))
            else:
                return error_response("Request body is required", 400)
        except (json_module.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        topic = data.get("topic")
        if not topic:
            return error_response("topic is required", 400)

        # Validate topic
        is_valid, err = validate_path_segment(topic[:50], "topic", SAFE_ID_PATTERN)
        # Note: topic can contain spaces, so we only validate first 50 chars for safety

        try:
            from aragora import Arena, Environment, DebateProtocol
            from aragora.agents import get_agents_by_names
        except ImportError:
            return feature_unavailable_response("debate")

        # Get parameters
        agent_names = data.get("agents", ["anthropic-api", "openai-api"])
        rounds = data.get("rounds", 3)
        consensus = data.get("consensus", "majority")

        # Validate parameters
        rounds = min(max(int(rounds), 1), 10)  # Clamp 1-10

        try:
            # Create environment
            env = Environment(
                task=f"Debate the following trending topic: {topic}",
                context=f"This topic is currently trending and warrants thoughtful analysis from multiple perspectives.",
            )

            # Get agents
            agents = get_agents_by_names(agent_names[:5])  # Max 5 agents

            if not agents:
                return error_response("No valid agents available", 400)

            # Create protocol
            protocol = DebateProtocol(
                rounds=rounds,
                consensus=consensus,
            )

            # Create arena
            arena = Arena.from_env(env, agents, protocol)

            # Run debate asynchronously
            async def run_debate():
                return await arena.run()

            result = self._run_async_safely(run_debate)

            if result is None:
                return error_response("Debate failed to complete", 500)

            # Record outcome with pulse manager
            manager = get_pulse_manager()
            if manager:
                try:
                    from aragora.pulse.ingestor import TrendingTopic
                    # Create a minimal topic for tracking
                    tracking_topic = TrendingTopic(
                        topic=topic,
                        platform="manual",
                        volume=1,
                        category="user_submitted",
                    )
                    manager.record_debate_outcome(tracking_topic, result)
                except Exception as e:
                    logger.warning(f"Failed to record debate outcome: {e}")

            return json_response({
                "debate_id": result.id,
                "status": "completed",
                "topic": topic,
                "agents": [a.name for a in agents],
                "consensus_reached": result.consensus_reached,
                "confidence": result.confidence,
                "final_answer": result.final_answer[:500] if result.final_answer else None,
                "rounds_used": result.rounds_used,
            })

        except Exception as e:
            logger.error(f"Failed to run debate on topic: {e}")
            return error_response(safe_error_message(e, "start debate"), 500)
