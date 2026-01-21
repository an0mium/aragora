"""
Decision Routing Middleware.

Provides unified routing of incoming requests through the DecisionRouter,
ensuring consistent handling across all channels (Slack, Teams, Discord,
Telegram, WhatsApp, Email, Web, API).

Features:
- Unified deduplication across channels
- Response caching at router level
- Origin tracking for bidirectional routing
- Consistent error handling

Usage:
    from aragora.server.middleware.decision_routing import (
        route_decision,
        get_decision_router,
        DecisionRoutingMiddleware,
    )

    # As a decorator
    @route_decision(channel="slack")
    async def handle_slack_message(request):
        ...

    # As middleware
    middleware = DecisionRoutingMiddleware(router)
    result = await middleware.process(request)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

# Type variables
F = TypeVar("F", bound=Callable[..., Any])

# Deduplication window (5 seconds)
DEDUPE_WINDOW_SECONDS = 5.0

# Response cache TTL (1 hour)
CACHE_TTL_SECONDS = 3600.0


@dataclass
class RoutingContext:
    """Context for a routed request."""

    channel: str  # slack, teams, discord, telegram, whatsapp, email, web, api
    channel_id: str  # Channel/chat ID
    user_id: str  # User who sent the request
    request_id: str  # Unique request ID
    message_id: Optional[str] = None  # Original message ID
    thread_id: Optional[str] = None  # Thread ID if threaded
    workspace_id: Optional[str] = None  # Workspace/org ID
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "channel_id": self.channel_id,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "message_id": self.message_id,
            "thread_id": self.thread_id,
            "workspace_id": self.workspace_id,
            "metadata": self.metadata,
        }


class RequestDeduplicator:
    """
    Deduplicates identical requests across all channels.

    Uses content hashing to detect duplicate requests within a time window.
    This prevents double-processing when the same message is received
    multiple times (e.g., webhook retries, multi-device delivery).
    """

    def __init__(self, window_seconds: float = DEDUPE_WINDOW_SECONDS):
        self._window_seconds = window_seconds
        self._seen: Dict[str, float] = {}  # hash -> timestamp
        self._in_flight: Dict[str, asyncio.Future] = {}  # hash -> future
        self._lock = asyncio.Lock()

    def _compute_hash(self, content: str, user_id: str, channel: str) -> str:
        """Compute a deduplication hash."""
        data = f"{channel}:{user_id}:{content}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    async def check_and_mark(
        self,
        content: str,
        user_id: str,
        channel: str,
    ) -> tuple[bool, Optional[asyncio.Future]]:
        """
        Check if this request is a duplicate.

        Returns:
            Tuple of (is_duplicate, optional_future_to_await)
            If is_duplicate is True and future is not None, await the future
            to get the result of the in-flight request.
        """
        request_hash = self._compute_hash(content, user_id, channel)
        now = time.time()

        async with self._lock:
            # Clean up old entries
            expired = [
                h for h, ts in self._seen.items()
                if now - ts > self._window_seconds
            ]
            for h in expired:
                del self._seen[h]
                self._in_flight.pop(h, None)

            # Check if already seen
            if request_hash in self._seen:
                logger.debug(f"Duplicate request detected: {request_hash[:8]}...")
                # Return the in-flight future if available
                return True, self._in_flight.get(request_hash)

            # Mark as seen and create in-flight future
            self._seen[request_hash] = now
            self._in_flight[request_hash] = asyncio.get_event_loop().create_future()
            return False, None

    async def complete(
        self,
        content: str,
        user_id: str,
        channel: str,
        result: Any,
    ) -> None:
        """Mark a request as complete and resolve any waiting duplicates."""
        request_hash = self._compute_hash(content, user_id, channel)

        async with self._lock:
            future = self._in_flight.get(request_hash)
            if future and not future.done():
                future.set_result(result)

    async def fail(
        self,
        content: str,
        user_id: str,
        channel: str,
        error: Exception,
    ) -> None:
        """Mark a request as failed."""
        request_hash = self._compute_hash(content, user_id, channel)

        async with self._lock:
            future = self._in_flight.get(request_hash)
            if future and not future.done():
                future.set_exception(error)
            # Remove from seen so retries work
            self._seen.pop(request_hash, None)
            self._in_flight.pop(request_hash, None)


class ResponseCache:
    """
    Caches responses for identical queries.

    Uses content + context hashing to cache responses, reducing
    redundant processing for repeated questions.
    """

    def __init__(self, ttl_seconds: float = CACHE_TTL_SECONDS, max_size: int = 1000):
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._cache: Dict[str, tuple[Any, float]] = {}  # hash -> (result, timestamp)
        self._lock = asyncio.Lock()

    def _compute_hash(self, content: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Compute a cache key hash."""
        ctx_str = json.dumps(context, sort_keys=True) if context else ""
        data = f"{content}:{ctx_str}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    async def get(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Get a cached response if available."""
        cache_key = self._compute_hash(content, context)
        now = time.time()

        async with self._lock:
            if cache_key in self._cache:
                result, timestamp = self._cache[cache_key]
                if now - timestamp < self._ttl_seconds:
                    logger.debug(f"Cache hit for: {cache_key[:8]}...")
                    return result
                else:
                    del self._cache[cache_key]

        return None

    async def set(
        self,
        content: str,
        result: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Cache a response."""
        cache_key = self._compute_hash(content, context)
        now = time.time()

        async with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[cache_key] = (result, now)

    async def clear(self) -> int:
        """Clear the cache. Returns number of entries cleared."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count


class DecisionRoutingMiddleware:
    """
    Middleware that routes requests through DecisionRouter.

    Provides:
    - Unified entry point for all channels
    - Request deduplication
    - Response caching
    - Origin tracking for bidirectional routing
    - Metrics and tracing
    """

    def __init__(
        self,
        enable_deduplication: bool = True,
        enable_caching: bool = True,
        dedupe_window: float = DEDUPE_WINDOW_SECONDS,
        cache_ttl: float = CACHE_TTL_SECONDS,
    ):
        self._enable_deduplication = enable_deduplication
        self._enable_caching = enable_caching
        self._deduplicator = RequestDeduplicator(dedupe_window) if enable_deduplication else None
        self._cache = ResponseCache(cache_ttl) if enable_caching else None
        self._router = None  # Lazy loaded

    def _get_router(self):
        """Get or create the DecisionRouter."""
        if self._router is None:
            try:
                from aragora.core.decision import DecisionRouter
                self._router = DecisionRouter()
            except ImportError:
                logger.warning("DecisionRouter not available")
        return self._router

    async def process(
        self,
        content: str,
        context: RoutingContext,
        decision_type: str = "debate",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process a request through the routing middleware.

        Args:
            content: The request content/question
            context: Routing context with channel info
            decision_type: Type of decision (debate, workflow, gauntlet, quick)
            **kwargs: Additional arguments for the router

        Returns:
            Response dictionary with result and metadata
        """
        start_time = time.time()

        # Check cache first
        if self._cache:
            cache_context = {
                "channel": context.channel,
                "workspace_id": context.workspace_id,
            }
            cached = await self._cache.get(content, cache_context)
            if cached:
                logger.info(f"Serving cached response for {context.request_id}")
                return {
                    "success": True,
                    "cached": True,
                    "result": cached,
                    "request_id": context.request_id,
                }

        # Check for duplicate in-flight request
        if self._deduplicator:
            is_duplicate, future = await self._deduplicator.check_and_mark(
                content, context.user_id, context.channel
            )
            if is_duplicate:
                if future:
                    logger.info(f"Waiting for in-flight duplicate: {context.request_id}")
                    try:
                        result = await asyncio.wait_for(future, timeout=300.0)
                        return {
                            "success": True,
                            "deduplicated": True,
                            "result": result,
                            "request_id": context.request_id,
                        }
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout waiting for duplicate: {context.request_id}")
                else:
                    return {
                        "success": False,
                        "error": "Duplicate request detected",
                        "request_id": context.request_id,
                    }

        try:
            # Register origin for bidirectional routing
            await self._register_origin(content, context)

            # Route through DecisionRouter if available
            router = self._get_router()
            if router:
                result = await self._route_via_decision_router(
                    content, context, decision_type, **kwargs
                )
            else:
                # Fallback: direct debate
                result = await self._fallback_route(content, context, **kwargs)

            duration = time.time() - start_time

            # Cache successful results
            if self._cache and result.get("success"):
                cache_context = {
                    "channel": context.channel,
                    "workspace_id": context.workspace_id,
                }
                await self._cache.set(content, result.get("answer"), cache_context)

            # Complete deduplication
            if self._deduplicator:
                await self._deduplicator.complete(
                    content, context.user_id, context.channel, result.get("answer")
                )

            logger.info(
                f"Routed request {context.request_id} via {context.channel} "
                f"in {duration:.2f}s"
            )

            return {
                "success": True,
                "result": result,
                "request_id": context.request_id,
                "duration_seconds": duration,
            }

        except Exception as e:
            logger.error(f"Routing error for {context.request_id}: {e}")

            if self._deduplicator:
                await self._deduplicator.fail(
                    content, context.user_id, context.channel, e
                )

            return {
                "success": False,
                "error": str(e),
                "request_id": context.request_id,
            }

    async def _register_origin(self, content: str, context: RoutingContext) -> None:
        """Register the origin for bidirectional routing."""
        try:
            from aragora.server.debate_origin import register_debate_origin

            register_debate_origin(
                debate_id=context.request_id,
                platform=context.channel,
                channel_id=context.channel_id,
                user_id=context.user_id,
                thread_id=context.thread_id,
                message_id=context.message_id,
                metadata={
                    "workspace_id": context.workspace_id,
                    "content_preview": content[:100] if content else "",
                    **context.metadata,
                },
            )
        except ImportError:
            logger.debug("debate_origin not available for origin registration")
        except Exception as e:
            logger.warning(f"Failed to register origin: {e}")

    async def _route_via_decision_router(
        self,
        content: str,
        context: RoutingContext,
        decision_type: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Route via the full DecisionRouter."""
        from aragora.core.decision import (
            DecisionRequest,
            DecisionType,
            InputSource,
            RequestContext,
        )

        # Map channel to input source
        source_map = {
            "slack": InputSource.SLACK,
            "teams": InputSource.TEAMS,
            "discord": InputSource.DISCORD,
            "telegram": InputSource.TELEGRAM,
            "whatsapp": InputSource.WHATSAPP,
            "email": InputSource.EMAIL,
            "gmail": InputSource.GMAIL,
            "web": InputSource.HTTP_API,
            "api": InputSource.HTTP_API,
            "websocket": InputSource.WEBSOCKET,
            "cli": InputSource.CLI,
        }
        source = source_map.get(context.channel.lower(), InputSource.HTTP_API)

        # Map decision type
        type_map = {
            "debate": DecisionType.DEBATE,
            "workflow": DecisionType.WORKFLOW,
            "gauntlet": DecisionType.GAUNTLET,
            "quick": DecisionType.QUICK,
        }
        dtype = type_map.get(decision_type.lower(), DecisionType.DEBATE)

        # Build context - channel_id and thread_id go in metadata
        request_metadata = {
            **context.metadata,
            "channel_id": context.channel_id,
            "thread_id": context.thread_id,
        }
        request_context = RequestContext(
            user_id=context.user_id,
            workspace_id=context.workspace_id,
            metadata=request_metadata,
        )

        # Build request
        request = DecisionRequest(
            request_id=context.request_id,
            content=content,
            decision_type=dtype,
            source=source,
            context=request_context,
        )

        # Route
        router = self._get_router()
        result = await router.route(request)

        return {
            "success": result.success,
            "answer": result.answer,
            "confidence": result.confidence,
            "consensus_reached": result.consensus_reached,
            "reasoning": result.reasoning,
            "duration_seconds": result.duration_seconds,
            "error": result.error,
        }

    async def _fallback_route(
        self,
        content: str,
        context: RoutingContext,
        **kwargs,
    ) -> Dict[str, Any]:
        """Fallback routing when DecisionRouter is not available."""
        try:
            from aragora.debate import Arena, Environment, DebateProtocol

            env = Environment(task=content)
            protocol = DebateProtocol(rounds=2, consensus="majority")

            # Run with default agents
            arena = Arena(env, protocol=protocol)
            result = await arena.run()

            return {
                "success": True,
                "answer": result.get("consensus", {}).get("answer", ""),
                "confidence": result.get("consensus", {}).get("confidence", 0.0),
                "consensus_reached": result.get("consensus_reached", False),
            }

        except Exception as e:
            logger.error(f"Fallback routing failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }


# Global middleware instance
_middleware: Optional[DecisionRoutingMiddleware] = None
_middleware_lock = asyncio.Lock()


async def get_decision_middleware() -> DecisionRoutingMiddleware:
    """Get or create the global DecisionRoutingMiddleware."""
    global _middleware
    if _middleware is None:
        async with _middleware_lock:
            if _middleware is None:
                _middleware = DecisionRoutingMiddleware()
    return _middleware


def route_decision(
    channel: str,
    decision_type: str = "debate",
    enable_caching: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to route handler results through DecisionRouter.

    Args:
        channel: Channel name (slack, teams, discord, etc.)
        decision_type: Type of decision (debate, workflow, gauntlet, quick)
        enable_caching: Whether to cache responses

    Usage:
        @route_decision(channel="slack")
        async def handle_slack_message(content, user_id, channel_id):
            return {"content": content, "user_id": user_id}
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract content and context from handler result
            handler_result = await func(*args, **kwargs)

            if not isinstance(handler_result, dict):
                return handler_result

            # Get middleware
            middleware = await get_decision_middleware()

            # Build routing context
            context = RoutingContext(
                channel=channel,
                channel_id=handler_result.get("channel_id", ""),
                user_id=handler_result.get("user_id", ""),
                request_id=handler_result.get("request_id", f"{channel}-{time.time()}"),
                message_id=handler_result.get("message_id"),
                thread_id=handler_result.get("thread_id"),
                workspace_id=handler_result.get("workspace_id"),
                metadata=handler_result.get("metadata", {}),
            )

            content = handler_result.get("content", "")

            # Route through middleware
            result = await middleware.process(
                content=content,
                context=context,
                decision_type=decision_type,
            )

            return result

        return wrapper  # type: ignore
    return decorator


def reset_decision_middleware() -> None:
    """Reset the global middleware (for testing)."""
    global _middleware
    _middleware = None


__all__ = [
    "RoutingContext",
    "RequestDeduplicator",
    "ResponseCache",
    "DecisionRoutingMiddleware",
    "get_decision_middleware",
    "route_decision",
    "reset_decision_middleware",
]
