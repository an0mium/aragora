"""
Pulse and trending topics endpoint handlers.

Endpoints:
- GET /api/pulse/trending - Get trending topics from multiple sources
- GET /api/pulse/suggest - Suggest a trending topic for debate
- GET /api/pulse/analytics - Get analytics on trending topic debate outcomes
- POST /api/pulse/debate-topic - Start a debate on a trending topic

Scheduler endpoints:
- GET /api/pulse/scheduler/status - Current scheduler state and metrics
- POST /api/pulse/scheduler/start - Start the scheduler
- POST /api/pulse/scheduler/stop - Stop the scheduler
- POST /api/pulse/scheduler/pause - Pause the scheduler
- POST /api/pulse/scheduler/resume - Resume the scheduler
- PATCH /api/pulse/scheduler/config - Update scheduler configuration
- GET /api/pulse/scheduler/history - Get scheduled debate history
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from typing import Optional

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

from aragora.config import DB_TIMEOUT_SECONDS
from aragora.server.http_utils import run_async

logger = logging.getLogger(__name__)
from .base import (
    BaseHandler, HandlerResult, json_response, error_response,
    get_int_param, get_string_param, validate_path_segment, SAFE_ID_PATTERN,
    feature_unavailable_response, auto_error_response, ttl_cache, safe_error_message,
    require_auth,
)
from .utils.rate_limit import rate_limit


# Shared PulseManager singleton for analytics tracking
# This allows FeedbackPhase to record outcomes that persist across requests
import threading
_pulse_lock = threading.Lock()
_shared_pulse_manager = None
_shared_scheduler = None
_shared_debate_store = None

MAX_TOPIC_LENGTH = 200


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


def get_scheduled_debate_store():
    """Get or create the shared ScheduledDebateStore singleton."""
    global _shared_debate_store
    if _shared_debate_store is None:
        with _pulse_lock:
            if _shared_debate_store is None:
                try:
                    from aragora.pulse.store import ScheduledDebateStore
                    from aragora.config.legacy import DATA_DIR
                    db_path = DATA_DIR / "scheduled_debates.db"
                    _shared_debate_store = ScheduledDebateStore(db_path)
                except (ImportError, OSError, sqlite3.Error) as e:
                    logger.warning(f"Failed to initialize ScheduledDebateStore: {e}")
                    return None
    return _shared_debate_store


def get_pulse_scheduler():
    """Get or create the shared PulseDebateScheduler singleton.

    Note: The scheduler is created but not started automatically.
    Call scheduler.start() to begin scheduling debates.
    """
    global _shared_scheduler
    if _shared_scheduler is None:
        with _pulse_lock:
            if _shared_scheduler is None:
                try:
                    from aragora.pulse.scheduler import PulseDebateScheduler, SchedulerConfig
                    manager = get_pulse_manager()
                    store = get_scheduled_debate_store()
                    if manager and store:
                        _shared_scheduler = PulseDebateScheduler(manager, store)
                        logger.info("PulseDebateScheduler singleton created")
                except (ImportError, OSError, sqlite3.Error, RuntimeError) as e:
                    logger.warning(f"Failed to initialize PulseDebateScheduler: {e}")
                    return None
    return _shared_scheduler


class PulseHandler(BaseHandler):
    """Handler for pulse/trending topic endpoints."""

    ROUTES = [
        "/api/pulse/trending",
        "/api/pulse/suggest",
        "/api/pulse/analytics",
        "/api/pulse/debate-topic",
        "/api/pulse/scheduler/status",
        "/api/pulse/scheduler/start",
        "/api/pulse/scheduler/stop",
        "/api/pulse/scheduler/pause",
        "/api/pulse/scheduler/resume",
        "/api/pulse/scheduler/config",
        "/api/pulse/scheduler/history",
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

        if path == "/api/pulse/scheduler/status":
            return self._get_scheduler_status()

        if path == "/api/pulse/scheduler/history":
            limit = get_int_param(query_params, 'limit', 50)
            offset = get_int_param(query_params, 'offset', 0)
            platform = get_string_param(query_params, 'platform')
            return self._get_scheduler_history(min(limit, 100), offset, platform)

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
        except (asyncio.TimeoutError, RuntimeError, OSError) as e:
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

        except (asyncio.TimeoutError, RuntimeError, ValueError, KeyError) as e:
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

        except (asyncio.TimeoutError, RuntimeError, ValueError, KeyError) as e:
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
        if path == "/api/pulse/scheduler/start":
            return self._start_scheduler(handler)
        if path == "/api/pulse/scheduler/stop":
            return self._stop_scheduler(handler)
        if path == "/api/pulse/scheduler/pause":
            return self._pause_scheduler(handler)
        if path == "/api/pulse/scheduler/resume":
            return self._resume_scheduler(handler)
        return None

    def handle_patch(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Handle PATCH requests for pulse endpoints."""
        if path == "/api/pulse/scheduler/config":
            return self._update_scheduler_config(handler)
        return None

    @require_auth
    @rate_limit(rpm=5, limiter_name="pulse_debate_topic")
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

        topic = data.get("topic", "")
        if not isinstance(topic, str):
            return error_response("topic must be a string", 400)
        topic = topic.strip()
        if not topic:
            return error_response("topic is required", 400)
        if len(topic) > MAX_TOPIC_LENGTH:
            return error_response(f"topic exceeds {MAX_TOPIC_LENGTH} characters", 400)
        if any(ch in topic for ch in ("\x00", "\n", "\r")):
            return error_response("topic contains invalid characters", 400)

        try:
            from aragora import Arena, Environment, DebateProtocol
            from aragora.agents import get_agents_by_names
        except ImportError:
            return feature_unavailable_response("debate")

        # Get parameters
        agent_names = data.get("agents", ["anthropic-api", "openai-api"])
        if isinstance(agent_names, str):
            agent_names = [a.strip() for a in agent_names.split(",") if a.strip()]
        if not isinstance(agent_names, list):
            return error_response("agents must be a list or comma-separated string", 400)
        rounds = data.get("rounds", 3)
        consensus = data.get("consensus", "majority")

        # Validate parameters
        try:
            rounds = min(max(int(rounds), 1), 10)  # Clamp 1-10
        except (TypeError, ValueError):
            rounds = 3
        consensus = str(consensus).strip()
        if consensus not in {"majority", "unanimous", "judge", "none"}:
            return error_response("consensus must be one of: majority, unanimous, judge, none", 400)

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
                except (ValueError, TypeError, AttributeError) as e:
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

        except (RuntimeError, asyncio.TimeoutError, ValueError, KeyError) as e:
            logger.error(f"Failed to run debate on topic: {e}")
            return error_response(safe_error_message(e, "start debate"), 500)

    # ==================== Scheduler Endpoints ====================

    @auto_error_response("get scheduler status")
    def _get_scheduler_status(self) -> HandlerResult:
        """Get current scheduler status.

        GET /api/pulse/scheduler/status

        Returns scheduler state, configuration, and metrics.
        """
        scheduler = get_pulse_scheduler()
        if not scheduler:
            return feature_unavailable_response("pulse scheduler")

        status = scheduler.get_status()

        # Add store analytics
        store = get_scheduled_debate_store()
        if store:
            status["store_analytics"] = store.get_analytics()

        return json_response(status)

    @require_auth
    @rate_limit(rpm=5, limiter_name="scheduler_control")
    @auto_error_response("start scheduler")
    def _start_scheduler(self, handler) -> HandlerResult:
        """Start the pulse debate scheduler.

        POST /api/pulse/scheduler/start

        The scheduler will poll for trending topics and create debates
        automatically based on its configuration.
        """
        scheduler = get_pulse_scheduler()
        if not scheduler:
            return feature_unavailable_response("pulse scheduler")

        # Set up the debate creator callback if not already set
        if not scheduler._debate_creator:
            async def create_debate(topic_text: str, rounds: int, threshold: float):
                try:
                    from aragora import Arena, Environment, DebateProtocol
                    from aragora.agents import get_agents_by_names

                    env = Environment(task=topic_text)
                    agents = get_agents_by_names(["anthropic-api", "openai-api"])
                    protocol = DebateProtocol(rounds=rounds, consensus="majority")

                    if not agents:
                        logger.warning("No agents available for scheduled debate")
                        return None

                    arena = Arena.from_env(env, agents, protocol)
                    result = await arena.run()

                    return {
                        "debate_id": result.id,
                        "consensus_reached": result.consensus_reached,
                        "confidence": result.confidence,
                        "rounds_used": result.rounds_used,
                    }
                except (ImportError, RuntimeError, asyncio.TimeoutError) as e:
                    logger.error(f"Scheduled debate creation failed: {e}")
                    return None

            scheduler.set_debate_creator(create_debate)

        # Start the scheduler
        async def start():
            await scheduler.start()

        try:
            self._run_async_safely(start)
            return json_response({
                "success": True,
                "message": "Scheduler started",
                "state": scheduler.state.value,
            })
        except RuntimeError as e:
            return error_response(str(e), 400)

    @require_auth
    @rate_limit(rpm=5, limiter_name="scheduler_control")
    @auto_error_response("stop scheduler")
    def _stop_scheduler(self, handler) -> HandlerResult:
        """Stop the pulse debate scheduler.

        POST /api/pulse/scheduler/stop
        Body: { "graceful": true }  // Optional, default true
        """
        scheduler = get_pulse_scheduler()
        if not scheduler:
            return feature_unavailable_response("pulse scheduler")

        # Read body for graceful flag
        import json as json_module
        graceful = True
        try:
            content_length = int(handler.headers.get('Content-Length', 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json_module.loads(body.decode('utf-8'))
                graceful = data.get("graceful", True)
        except (ValueError, json_module.JSONDecodeError, UnicodeDecodeError) as e:
            # Failed to parse body, use default graceful=True
            logger.debug(f"Failed to parse stop request body, using graceful=True: {e}")

        async def stop():
            await scheduler.stop(graceful=graceful)

        self._run_async_safely(stop)

        return json_response({
            "success": True,
            "message": f"Scheduler stopped (graceful={graceful})",
            "state": scheduler.state.value,
        })

    @require_auth
    @rate_limit(rpm=5, limiter_name="scheduler_control")
    @auto_error_response("pause scheduler")
    def _pause_scheduler(self, handler) -> HandlerResult:
        """Pause the pulse debate scheduler.

        POST /api/pulse/scheduler/pause
        """
        scheduler = get_pulse_scheduler()
        if not scheduler:
            return feature_unavailable_response("pulse scheduler")

        async def pause():
            await scheduler.pause()

        self._run_async_safely(pause)

        return json_response({
            "success": True,
            "message": "Scheduler paused",
            "state": scheduler.state.value,
        })

    @require_auth
    @rate_limit(rpm=5, limiter_name="scheduler_control")
    @auto_error_response("resume scheduler")
    def _resume_scheduler(self, handler) -> HandlerResult:
        """Resume the pulse debate scheduler.

        POST /api/pulse/scheduler/resume
        """
        scheduler = get_pulse_scheduler()
        if not scheduler:
            return feature_unavailable_response("pulse scheduler")

        async def resume():
            await scheduler.resume()

        self._run_async_safely(resume)

        return json_response({
            "success": True,
            "message": "Scheduler resumed",
            "state": scheduler.state.value,
        })

    @require_auth
    @rate_limit(rpm=10, limiter_name="scheduler_config")
    @auto_error_response("update scheduler config")
    def _update_scheduler_config(self, handler) -> HandlerResult:
        """Update scheduler configuration.

        PATCH /api/pulse/scheduler/config
        Body: {
            "poll_interval_seconds": 300,
            "max_debates_per_hour": 6,
            "min_volume_threshold": 100,
            "allowed_categories": ["tech", "ai", "science"],
            ...
        }
        """
        scheduler = get_pulse_scheduler()
        if not scheduler:
            return feature_unavailable_response("pulse scheduler")

        import json as json_module
        try:
            content_length = int(handler.headers.get('Content-Length', 0))
            if content_length == 0:
                return error_response("Request body is required", 400)
            body = handler.rfile.read(content_length)
            updates = json_module.loads(body.decode('utf-8'))
        except (json_module.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        if not isinstance(updates, dict):
            return error_response("Body must be a JSON object", 400)

        # Validate config keys
        valid_keys = {
            "poll_interval_seconds", "platforms", "max_debates_per_hour",
            "min_interval_between_debates", "min_volume_threshold",
            "min_controversy_score", "allowed_categories", "blocked_categories",
            "dedup_window_hours", "debate_rounds", "consensus_threshold"
        }
        invalid_keys = set(updates.keys()) - valid_keys
        if invalid_keys:
            return error_response(f"Invalid config keys: {invalid_keys}", 400)

        scheduler.update_config(updates)

        return json_response({
            "success": True,
            "message": f"Updated config keys: {list(updates.keys())}",
            "config": scheduler.config.to_dict(),
        })

    @auto_error_response("get scheduler history")
    def _get_scheduler_history(
        self,
        limit: int,
        offset: int,
        platform: str | None,
    ) -> HandlerResult:
        """Get scheduled debate history.

        GET /api/pulse/scheduler/history?limit=50&offset=0&platform=hackernews
        """
        store = get_scheduled_debate_store()
        if not store:
            return feature_unavailable_response("pulse scheduler")

        records = store.get_history(limit=limit, offset=offset, platform=platform)

        return json_response({
            "debates": [
                {
                    "id": r.id,
                    "topic": r.topic_text,
                    "platform": r.platform,
                    "category": r.category,
                    "volume": r.volume,
                    "debate_id": r.debate_id,
                    "created_at": r.created_at,
                    "hours_ago": r.hours_ago,
                    "consensus_reached": r.consensus_reached,
                    "confidence": r.confidence,
                    "rounds_used": r.rounds_used,
                    "scheduler_run_id": r.scheduler_run_id,
                }
                for r in records
            ],
            "count": len(records),
            "total": store.count_total(),
            "limit": limit,
            "offset": offset,
        })
