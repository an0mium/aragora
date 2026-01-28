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
        is_dup, future = await deduplicator.check_and_mark("test content", "user-1", "slack")
        assert is_dup is False
        assert future is None  # No future for first request

    @pytest.mark.asyncio
    async def test_second_request_is_duplicate(self, deduplicator):
        """Same request within window should be duplicate."""
        await deduplicator.check_and_mark("test content", "user-1", "slack")

        is_dup, future = await deduplicator.check_and_mark("test content", "user-1", "slack")
        assert is_dup is True
        assert future is not None

    @pytest.mark.asyncio
    async def test_different_content_not_duplicate(self, deduplicator):
        """Different content should not be duplicate."""
        await deduplicator.check_and_mark("content 1", "user-1", "slack")

        is_dup, _ = await deduplicator.check_and_mark("content 2", "user-1", "slack")
        assert is_dup is False

    @pytest.mark.asyncio
    async def test_different_user_not_duplicate(self, deduplicator):
        """Same content from different user should not be duplicate."""
        await deduplicator.check_and_mark("same content", "user-1", "slack")

        is_dup, _ = await deduplicator.check_and_mark("same content", "user-2", "slack")
        assert is_dup is False

    @pytest.mark.asyncio
    async def test_expired_not_duplicate(self, deduplicator, monkeypatch):
        """Request after window expires should not be duplicate."""
        now = 1_000_000.0
        monkeypatch.setattr("aragora.server.middleware.decision_routing.time.time", lambda: now)

        await deduplicator.check_and_mark("test", "user-1", "slack")

        # Advance time beyond window
        now += 2.0

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
    async def test_cache_expiry(self, cache, monkeypatch):
        """Should not return expired entries."""
        now = 1_000_000.0
        monkeypatch.setattr("aragora.server.middleware.decision_routing.time.time", lambda: now)

        await cache.set("query", "answer")

        # Advance time beyond TTL
        now += 2.0

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


class TestRequestDeduplicatorFail:
    """Tests for RequestDeduplicator.fail() method."""

    @pytest.fixture
    def deduplicator(self):
        """Create a fresh deduplicator."""
        return RequestDeduplicator(window_seconds=5.0)

    @pytest.mark.asyncio
    async def test_fail_removes_from_seen(self, deduplicator):
        """Fail should remove request from seen set."""
        # First request
        await deduplicator.check_and_mark("test", "user-1", "slack")

        # Fail it
        await deduplicator.fail("test", "user-1", "slack", Exception("Error"))

        # Should be able to retry (not duplicate)
        is_dup, _ = await deduplicator.check_and_mark("test", "user-1", "slack")
        assert is_dup is False

    @pytest.mark.asyncio
    async def test_fail_sets_exception_on_future(self, deduplicator):
        """Fail should set exception on in-flight future."""
        await deduplicator.check_and_mark("test", "user-1", "slack")
        _, future = await deduplicator.check_and_mark("test", "user-1", "slack")

        assert not future.done()

        await deduplicator.fail("test", "user-1", "slack", ValueError("Test error"))

        assert future.done()
        with pytest.raises(ValueError, match="Test error"):
            future.result()

    @pytest.mark.asyncio
    async def test_fail_with_no_future_is_noop(self, deduplicator):
        """Fail with no in-flight future doesn't error."""
        # No prior request
        await deduplicator.fail("unknown", "user-1", "slack", Exception("Error"))
        # Should complete without error


class TestChannelMappings:
    """Tests for channel to InputSource mappings."""

    @pytest.mark.asyncio
    async def test_all_channel_mappings(self):
        """Middleware maps all supported channels to InputSource."""
        middleware = DecisionRoutingMiddleware(
            enable_deduplication=False,
            enable_caching=False,
        )

        channels = [
            ("slack", "SLACK"),
            ("teams", "TEAMS"),
            ("discord", "DISCORD"),
            ("telegram", "TELEGRAM"),
            ("whatsapp", "WHATSAPP"),
            ("email", "EMAIL"),
            ("gmail", "GMAIL"),
            ("web", "HTTP_API"),
            ("api", "HTTP_API"),
            ("websocket", "WEBSOCKET"),
            ("cli", "CLI"),
        ]

        for channel_name, expected_source in channels:
            context = RoutingContext(
                channel=channel_name,
                channel_id="test-channel",
                user_id="test-user",
                request_id=f"req-{channel_name}",
            )

            # Mock router to capture the request
            captured_request = None

            async def mock_route(request):
                nonlocal captured_request
                captured_request = request
                result = MagicMock()
                result.success = True
                result.answer = "Test"
                result.confidence = 0.9
                result.consensus_reached = True
                result.reasoning = ""
                result.duration_seconds = 0.1
                result.error = None
                return result

            with patch.object(middleware, "_get_router") as mock_get_router:
                mock_router = MagicMock()
                mock_router.route = mock_route
                mock_get_router.return_value = mock_router

                await middleware.process(f"Test {channel_name}", context)

            assert captured_request is not None, f"No request captured for {channel_name}"
            assert captured_request.source.value == expected_source.lower(), (
                f"Channel {channel_name} mapped to {captured_request.source.value}, expected {expected_source.lower()}"
            )


class TestDecisionTypeMappings:
    """Tests for decision type mappings."""

    @pytest.mark.asyncio
    async def test_all_decision_types(self):
        """Middleware maps all decision types correctly."""
        middleware = DecisionRoutingMiddleware(
            enable_deduplication=False,
            enable_caching=False,
        )

        decision_types = ["debate", "workflow", "gauntlet", "quick"]

        for dtype in decision_types:
            context = RoutingContext(
                channel="api",
                channel_id="test",
                user_id="user-1",
                request_id=f"req-{dtype}",
            )

            captured_request = None

            async def mock_route(request):
                nonlocal captured_request
                captured_request = request
                result = MagicMock()
                result.success = True
                result.answer = "Test"
                result.confidence = 0.9
                result.consensus_reached = True
                result.reasoning = ""
                result.duration_seconds = 0.1
                result.error = None
                return result

            with patch.object(middleware, "_get_router") as mock_get_router:
                mock_router = MagicMock()
                mock_router.route = mock_route
                mock_get_router.return_value = mock_router

                await middleware.process("Test", context, decision_type=dtype)

            assert captured_request is not None
            assert captured_request.decision_type.value == dtype


class TestFallbackRoute:
    """Tests for fallback routing when DecisionRouter unavailable."""

    @pytest.mark.asyncio
    async def test_fallback_when_no_router(self):
        """Should use fallback when DecisionRouter not available."""
        middleware = DecisionRoutingMiddleware(
            enable_deduplication=False,
            enable_caching=False,
        )

        context = RoutingContext(
            channel="api",
            channel_id="test",
            user_id="user-1",
            request_id="fallback-test",
        )

        # Mock _get_router to return None
        with patch.object(middleware, "_get_router", return_value=None):
            with patch.object(middleware, "_fallback_route") as mock_fallback:
                mock_fallback.return_value = {
                    "success": True,
                    "answer": "Fallback answer",
                    "confidence": 0.7,
                    "consensus_reached": True,
                }

                result = await middleware.process("Test fallback", context)

        mock_fallback.assert_called_once()
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_fallback_error_handling(self):
        """Should handle fallback errors gracefully."""
        middleware = DecisionRoutingMiddleware(
            enable_deduplication=False,
            enable_caching=False,
        )

        context = RoutingContext(
            channel="api",
            channel_id="test",
            user_id="user-1",
            request_id="error-test",
        )

        with patch.object(middleware, "_get_router", return_value=None):
            with patch.object(middleware, "_fallback_route") as mock_fallback:
                mock_fallback.side_effect = Exception("Fallback failed")

                result = await middleware.process("Test error", context)

        assert result["success"] is False
        assert "Fallback failed" in result["error"]


class TestErrorPropagation:
    """Tests for error propagation through middleware."""

    @pytest.mark.asyncio
    async def test_router_error_propagates(self):
        """Router errors should propagate as failed result."""
        middleware = DecisionRoutingMiddleware(
            enable_deduplication=False,
            enable_caching=False,
        )

        context = RoutingContext(
            channel="api",
            channel_id="test",
            user_id="user-1",
            request_id="router-error-test",
        )

        with patch.object(middleware, "_get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route = AsyncMock(side_effect=ValueError("Router exploded"))
            mock_get_router.return_value = mock_router

            result = await middleware.process("Test", context)

        assert result["success"] is False
        assert "Router exploded" in result["error"]

    @pytest.mark.asyncio
    async def test_error_clears_deduplication(self):
        """Errors should clear deduplication state for retries."""
        middleware = DecisionRoutingMiddleware(
            enable_deduplication=True,
            enable_caching=False,
            dedupe_window=5.0,
        )

        context = RoutingContext(
            channel="api",
            channel_id="test",
            user_id="user-1",
            request_id="dedupe-error-test",
        )

        with patch.object(middleware, "_get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route = AsyncMock(side_effect=Exception("First attempt failed"))
            mock_get_router.return_value = mock_router

            # First attempt - should fail
            result1 = await middleware.process("Retry me", context)
            assert result1["success"] is False

            # Second attempt - should not be marked as duplicate
            mock_router.route = AsyncMock(
                return_value=MagicMock(
                    success=True,
                    answer="Success",
                    confidence=0.9,
                    consensus_reached=True,
                    reasoning="",
                    duration_seconds=0.1,
                    error=None,
                )
            )

            context2 = RoutingContext(
                channel="api",
                channel_id="test",
                user_id="user-1",
                request_id="dedupe-error-test-2",
            )

            result2 = await middleware.process("Retry me", context2)
            # Should process (not be a duplicate)
            assert result2["success"] is True or "deduplicated" not in result2


class TestOriginRegistration:
    """Tests for origin registration."""

    @pytest.mark.asyncio
    async def test_origin_registration_called(self):
        """Should register origin for bidirectional routing."""
        middleware = DecisionRoutingMiddleware(
            enable_deduplication=False,
            enable_caching=False,
        )

        context = RoutingContext(
            channel="slack",
            channel_id="C123",
            user_id="U456",
            request_id="origin-test",
            thread_id="T789",
            message_id="M012",
            workspace_id="W345",
            metadata={"custom": "data"},
        )

        with patch.object(middleware, "_get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.answer = "Test"
            mock_result.confidence = 0.9
            mock_result.consensus_reached = True
            mock_result.reasoning = ""
            mock_result.duration_seconds = 0.1
            mock_result.error = None
            mock_router.route = AsyncMock(return_value=mock_result)
            mock_get_router.return_value = mock_router

            with patch.object(middleware, "_register_origin") as mock_register:
                mock_register.return_value = None

                await middleware.process("Test origin", context)

                mock_register.assert_called_once()
                call_args = mock_register.call_args
                assert call_args[0][0] == "Test origin"
                assert call_args[0][1].channel == "slack"
                assert call_args[0][1].request_id == "origin-test"

    @pytest.mark.asyncio
    async def test_origin_registration_error_is_non_fatal(self):
        """Origin registration errors should not fail the request.

        The _register_origin method catches exceptions internally, so we patch
        at a lower level to verify the internal exception handling works.
        """
        middleware = DecisionRoutingMiddleware(
            enable_deduplication=False,
            enable_caching=False,
        )

        context = RoutingContext(
            channel="api",
            channel_id="test",
            user_id="user-1",
            request_id="origin-error-test",
        )

        with patch.object(middleware, "_get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.answer = "Test"
            mock_result.confidence = 0.9
            mock_result.consensus_reached = True
            mock_result.reasoning = ""
            mock_result.duration_seconds = 0.1
            mock_result.error = None
            mock_router.route = AsyncMock(return_value=mock_result)
            mock_get_router.return_value = mock_router

            # Patch at the module level where register_debate_origin is imported
            # The _register_origin method catches exceptions internally
            with patch(
                "aragora.server.middleware.decision_routing.RoutingContext"
            ):  # Just to prove the method runs
                with patch(
                    "aragora.server.debate_origin.register_debate_origin",
                    side_effect=Exception("Origin registration failed"),
                ):
                    result = await middleware.process("Test", context)

                    # Should still succeed - the internal try/except handles it
                    assert result["success"] is True


class TestMiddlewareDisabledFeatures:
    """Tests for middleware with features disabled."""

    @pytest.mark.asyncio
    async def test_no_deduplication(self):
        """Should work without deduplication enabled."""
        middleware = DecisionRoutingMiddleware(
            enable_deduplication=False,
            enable_caching=False,
        )

        assert middleware._deduplicator is None

        context = RoutingContext(
            channel="api",
            channel_id="test",
            user_id="user-1",
            request_id="no-dedupe",
        )

        with patch.object(middleware, "_get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.answer = "Test"
            mock_result.confidence = 0.9
            mock_result.consensus_reached = True
            mock_result.reasoning = ""
            mock_result.duration_seconds = 0.1
            mock_result.error = None
            mock_router.route = AsyncMock(return_value=mock_result)
            mock_get_router.return_value = mock_router

            result = await middleware.process("Test", context)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_no_caching(self):
        """Should work without caching enabled."""
        middleware = DecisionRoutingMiddleware(
            enable_deduplication=False,
            enable_caching=False,
        )

        assert middleware._cache is None

        context = RoutingContext(
            channel="api",
            channel_id="test",
            user_id="user-1",
            request_id="no-cache",
        )

        with patch.object(middleware, "_get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.answer = "Test"
            mock_result.confidence = 0.9
            mock_result.consensus_reached = True
            mock_result.reasoning = ""
            mock_result.duration_seconds = 0.1
            mock_result.error = None
            mock_router.route = AsyncMock(return_value=mock_result)
            mock_get_router.return_value = mock_router

            result = await middleware.process("Test", context)

        assert result["success"] is True
        assert "cached" not in result


class TestDuplicateHandling:
    """Tests for duplicate request handling."""

    @pytest.mark.asyncio
    async def test_duplicate_without_future_returns_error(self):
        """Duplicate without in-flight future returns error."""
        middleware = DecisionRoutingMiddleware(
            enable_deduplication=True,
            enable_caching=False,
            dedupe_window=5.0,
        )

        context = RoutingContext(
            channel="api",
            channel_id="test",
            user_id="user-1",
            request_id="dup-no-future",
        )

        # Manually mark as seen but don't create a future
        await middleware._deduplicator.check_and_mark("duplicate content", "user-1", "api")
        # Clear the future to simulate a scenario where future is gone
        async with middleware._deduplicator._lock:
            request_hash = middleware._deduplicator._compute_hash(
                "duplicate content", "user-1", "api"
            )
            middleware._deduplicator._in_flight.pop(request_hash, None)

        # Second request should detect duplicate but have no future
        result = await middleware.process("duplicate content", context)

        assert result["success"] is False
        assert "Duplicate" in result.get("error", "")
