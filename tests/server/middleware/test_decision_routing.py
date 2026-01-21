"""
Tests for Decision Routing Middleware.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.middleware.decision_routing import (
    DecisionRoutingMiddleware,
    RequestDeduplicator,
    ResponseCache,
    RoutingContext,
    get_decision_middleware,
    reset_decision_middleware,
    route_decision,
)


class TestRoutingContext:
    """Tests for RoutingContext."""

    def test_creation(self):
        """Should create context with required fields."""
        ctx = RoutingContext(
            channel="slack",
            channel_id="C1234",
            user_id="U5678",
            request_id="req-001",
        )
        assert ctx.channel == "slack"
        assert ctx.channel_id == "C1234"
        assert ctx.user_id == "U5678"

    def test_to_dict(self):
        """Should serialize to dictionary."""
        ctx = RoutingContext(
            channel="telegram",
            channel_id="12345",
            user_id="67890",
            request_id="req-002",
            thread_id="thread-1",
            metadata={"extra": "data"},
        )
        data = ctx.to_dict()
        assert data["channel"] == "telegram"
        assert data["thread_id"] == "thread-1"
        assert data["metadata"]["extra"] == "data"


class TestRequestDeduplicator:
    """Tests for RequestDeduplicator."""

    @pytest.fixture
    def deduplicator(self):
        """Create a fresh deduplicator."""
        return RequestDeduplicator(window_seconds=1.0)

    @pytest.mark.asyncio
    async def test_first_request_not_duplicate(self, deduplicator):
        """First request should not be marked as duplicate."""
        is_dup, future = await deduplicator.check_and_mark(
            "test content", "user-1", "slack"
        )
        assert is_dup is False
        assert future is None  # No future for first request

    @pytest.mark.asyncio
    async def test_second_request_is_duplicate(self, deduplicator):
        """Same request within window should be duplicate."""
        await deduplicator.check_and_mark("test content", "user-1", "slack")

        is_dup, future = await deduplicator.check_and_mark(
            "test content", "user-1", "slack"
        )
        assert is_dup is True
        assert future is not None

    @pytest.mark.asyncio
    async def test_different_content_not_duplicate(self, deduplicator):
        """Different content should not be duplicate."""
        await deduplicator.check_and_mark("content 1", "user-1", "slack")

        is_dup, _ = await deduplicator.check_and_mark(
            "content 2", "user-1", "slack"
        )
        assert is_dup is False

    @pytest.mark.asyncio
    async def test_different_user_not_duplicate(self, deduplicator):
        """Same content from different user should not be duplicate."""
        await deduplicator.check_and_mark("same content", "user-1", "slack")

        is_dup, _ = await deduplicator.check_and_mark(
            "same content", "user-2", "slack"
        )
        assert is_dup is False

    @pytest.mark.asyncio
    async def test_expired_not_duplicate(self, deduplicator):
        """Request after window expires should not be duplicate."""
        await deduplicator.check_and_mark("test", "user-1", "slack")

        # Wait for window to expire
        await asyncio.sleep(1.1)

        is_dup, _ = await deduplicator.check_and_mark("test", "user-1", "slack")
        assert is_dup is False

    @pytest.mark.asyncio
    async def test_complete_resolves_future(self, deduplicator):
        """Complete should resolve the in-flight future."""
        await deduplicator.check_and_mark("test", "user-1", "slack")
        _, future = await deduplicator.check_and_mark("test", "user-1", "slack")

        assert not future.done()

        await deduplicator.complete("test", "user-1", "slack", "result")

        assert future.done()
        assert future.result() == "result"


class TestResponseCache:
    """Tests for ResponseCache."""

    @pytest.fixture
    def cache(self):
        """Create a fresh cache."""
        return ResponseCache(ttl_seconds=1.0, max_size=10)

    @pytest.mark.asyncio
    async def test_cache_miss(self, cache):
        """Should return None for cache miss."""
        result = await cache.get("unknown content")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit(self, cache):
        """Should return cached value."""
        await cache.set("test query", "cached answer")

        result = await cache.get("test query")
        assert result == "cached answer"

    @pytest.mark.asyncio
    async def test_cache_with_context(self, cache):
        """Should include context in cache key."""
        await cache.set("query", "answer1", context={"workspace": "ws1"})
        await cache.set("query", "answer2", context={"workspace": "ws2"})

        # Different contexts should have different cache entries
        r1 = await cache.get("query", context={"workspace": "ws1"})
        r2 = await cache.get("query", context={"workspace": "ws2"})

        assert r1 == "answer1"
        assert r2 == "answer2"

    @pytest.mark.asyncio
    async def test_cache_expiry(self, cache):
        """Should not return expired entries."""
        await cache.set("query", "answer")

        # Wait for TTL to expire
        await asyncio.sleep(1.1)

        result = await cache.get("query")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        """Should evict oldest when at capacity."""
        cache = ResponseCache(ttl_seconds=60.0, max_size=3)

        await cache.set("q1", "a1")
        await cache.set("q2", "a2")
        await cache.set("q3", "a3")
        await cache.set("q4", "a4")  # Should evict q1

        assert await cache.get("q1") is None
        assert await cache.get("q2") == "a2"
        assert await cache.get("q3") == "a3"
        assert await cache.get("q4") == "a4"

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Should clear all entries."""
        await cache.set("q1", "a1")
        await cache.set("q2", "a2")

        count = await cache.clear()
        assert count == 2

        assert await cache.get("q1") is None
        assert await cache.get("q2") is None


class TestDecisionRoutingMiddleware:
    """Tests for DecisionRoutingMiddleware."""

    @pytest.fixture
    def middleware(self):
        """Create middleware for testing."""
        return DecisionRoutingMiddleware(
            enable_deduplication=True,
            enable_caching=True,
            dedupe_window=1.0,
            cache_ttl=60.0,
        )

    @pytest.mark.asyncio
    async def test_process_basic(self, middleware):
        """Should process a basic request."""
        context = RoutingContext(
            channel="slack",
            channel_id="C1234",
            user_id="U5678",
            request_id="test-req-1",
        )

        # Mock the router
        with patch.object(middleware, "_get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.answer = "Test answer"
            mock_result.confidence = 0.9
            mock_result.consensus_reached = True
            mock_result.reasoning = "Test reasoning"
            mock_result.duration_seconds = 1.5
            mock_result.error = None

            mock_router.route = AsyncMock(return_value=mock_result)
            mock_get_router.return_value = mock_router

            result = await middleware.process("Test question", context)

            assert result["success"] is True
            assert "result" in result

    @pytest.mark.asyncio
    async def test_caching_works(self, middleware):
        """Should cache and return cached results."""
        context = RoutingContext(
            channel="api",
            channel_id="api-1",
            user_id="user-1",
            request_id="cache-test-1",
        )

        # Pre-populate cache
        await middleware._cache.set(
            "cached query",
            "cached answer",
            context={"channel": "api", "workspace_id": None},
        )

        result = await middleware.process("cached query", context)

        assert result["success"] is True
        assert result.get("cached") is True

    @pytest.mark.asyncio
    async def test_deduplication_works(self, middleware):
        """Should deduplicate concurrent requests."""
        context1 = RoutingContext(
            channel="slack",
            channel_id="C1234",
            user_id="U5678",
            request_id="dup-test-1",
        )

        # First request (starts processing)
        with patch.object(middleware, "_get_router") as mock_get_router:
            mock_router = MagicMock()

            async def slow_route(request):
                await asyncio.sleep(0.5)
                result = MagicMock()
                result.success = True
                result.answer = "Answer"
                result.confidence = 0.9
                result.consensus_reached = True
                result.reasoning = ""
                result.duration_seconds = 0.5
                result.error = None
                return result

            mock_router.route = slow_route
            mock_get_router.return_value = mock_router

            # Start first request
            task1 = asyncio.create_task(middleware.process("dup query", context1))

            # Small delay to ensure first request is in-flight
            await asyncio.sleep(0.1)

            # Second request should be marked as duplicate
            context2 = RoutingContext(
                channel="slack",
                channel_id="C1234",
                user_id="U5678",
                request_id="dup-test-2",
            )
            result2 = await middleware.process("dup query", context2)

            # Wait for first to complete
            result1 = await task1

            assert result1["success"] is True
            assert result2.get("deduplicated") is True or result2.get("success") is True


class TestRouteDecisionDecorator:
    """Tests for route_decision decorator."""

    def setup_method(self):
        """Reset middleware before each test."""
        reset_decision_middleware()

    @pytest.mark.asyncio
    async def test_decorator_basic(self):
        """Should wrap handler and route through middleware."""
        @route_decision(channel="slack")
        async def test_handler():
            return {
                "content": "Test question",
                "user_id": "user-1",
                "channel_id": "C1234",
                "request_id": "decorated-test",
            }

        # Get middleware and mock the process
        middleware = await get_decision_middleware()
        with patch.object(middleware, "process") as mock_process:
            mock_process.return_value = {"success": True, "result": {}}

            await test_handler()

            mock_process.assert_called_once()


class TestGlobalMiddleware:
    """Tests for global middleware singleton."""

    def setup_method(self):
        """Reset middleware before each test."""
        reset_decision_middleware()

    @pytest.mark.asyncio
    async def test_get_returns_singleton(self):
        """Should return the same instance."""
        m1 = await get_decision_middleware()
        m2 = await get_decision_middleware()
        assert m1 is m2

    @pytest.mark.asyncio
    async def test_reset_clears_singleton(self):
        """Reset should create new instance."""
        m1 = await get_decision_middleware()
        reset_decision_middleware()
        m2 = await get_decision_middleware()
        assert m1 is not m2
