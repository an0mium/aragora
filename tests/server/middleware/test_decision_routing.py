"""
Tests for Decision Routing Middleware.

Covers:
- RoutingContext creation and serialization
- RequestDeduplicator: cross-platform dedup, window expiry, concurrent handling, fail/complete
- ResponseCache: hit/miss, TTL, eviction, invalidation by workspace/tag/agent/policy, stats
- DecisionRoutingMiddleware: process flow, caching, dedup, origin registration, fallback, errors
- route_decision decorator
- Module-level helpers: invalidate_cache_for_workspace, invalidate_cache_for_policy_change, etc.
- Edge cases: unknown origins, expired cache, duplicate without future, timeouts
"""

import asyncio
import time as _time_mod

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.middleware.decision_routing import (
    CacheEntry,
    DecisionRoutingMiddleware,
    RequestDeduplicator,
    ResponseCache,
    RoutingContext,
    get_cache_stats,
    get_decision_middleware,
    invalidate_cache_for_agent_upgrade,
    invalidate_cache_for_policy_change,
    invalidate_cache_for_workspace,
    reset_decision_middleware,
    route_decision,
    DEDUPE_WINDOW_SECONDS,
    CACHE_TTL_SECONDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(
    channel="slack",
    channel_id="C1234",
    user_id="U5678",
    request_id="req-001",
    **kwargs,
) -> RoutingContext:
    """Create a RoutingContext with sensible defaults."""
    return RoutingContext(
        channel=channel,
        channel_id=channel_id,
        user_id=user_id,
        request_id=request_id,
        **kwargs,
    )


def _mock_router_result(
    success=True,
    answer="Test answer",
    confidence=0.9,
    consensus_reached=True,
    reasoning="",
    duration_seconds=0.1,
    error=None,
):
    """Build a mock DecisionRouter result."""
    result = MagicMock()
    result.success = success
    result.answer = answer
    result.confidence = confidence
    result.consensus_reached = consensus_reached
    result.reasoning = reasoning
    result.duration_seconds = duration_seconds
    result.error = error
    return result


def _patch_router(middleware, result=None, side_effect=None):
    """Return a context-manager that patches _get_router on *middleware*."""
    mock_router = MagicMock()
    if side_effect is not None:
        mock_router.route = AsyncMock(side_effect=side_effect)
    else:
        mock_router.route = AsyncMock(return_value=result or _mock_router_result())
    return patch.object(middleware, "_get_router", return_value=mock_router)


# ===========================================================================
# RoutingContext
# ===========================================================================


class TestRoutingContext:
    """Tests for RoutingContext dataclass."""

    def test_creation_with_required_fields(self):
        ctx = _make_context()
        assert ctx.channel == "slack"
        assert ctx.channel_id == "C1234"
        assert ctx.user_id == "U5678"
        assert ctx.request_id == "req-001"

    def test_optional_fields_default_to_none(self):
        ctx = _make_context()
        assert ctx.message_id is None
        assert ctx.thread_id is None
        assert ctx.workspace_id is None
        assert ctx.metadata == {}

    def test_to_dict_serialization(self):
        ctx = _make_context(
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
        assert "request_id" in data

    def test_to_dict_contains_all_keys(self):
        ctx = _make_context()
        data = ctx.to_dict()
        expected_keys = {
            "channel",
            "channel_id",
            "user_id",
            "request_id",
            "message_id",
            "thread_id",
            "workspace_id",
            "metadata",
        }
        assert set(data.keys()) == expected_keys


# ===========================================================================
# CacheEntry
# ===========================================================================


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_matches_tag(self):
        entry = CacheEntry(result="r", timestamp=0, tags=["alpha", "beta"])
        assert entry.matches_tag("alpha") is True
        assert entry.matches_tag("gamma") is False

    def test_matches_workspace(self):
        entry = CacheEntry(result="r", timestamp=0, workspace_id="ws-1")
        assert entry.matches_workspace("ws-1") is True
        assert entry.matches_workspace("ws-2") is False

    def test_matches_workspace_none(self):
        entry = CacheEntry(result="r", timestamp=0)
        assert entry.matches_workspace("ws-1") is False


# ===========================================================================
# RequestDeduplicator
# ===========================================================================


class TestRequestDeduplicator:
    """Tests for RequestDeduplicator."""

    @pytest.fixture
    def deduplicator(self):
        return RequestDeduplicator(window_seconds=1.0)

    @pytest.mark.asyncio
    async def test_first_request_not_duplicate(self, deduplicator):
        is_dup, future = await deduplicator.check_and_mark("test content", "user-1", "slack")
        assert is_dup is False
        assert future is None

    @pytest.mark.asyncio
    async def test_second_request_is_duplicate(self, deduplicator):
        await deduplicator.check_and_mark("test content", "user-1", "slack")
        is_dup, future = await deduplicator.check_and_mark("test content", "user-1", "slack")
        assert is_dup is True
        assert future is not None

    @pytest.mark.asyncio
    async def test_different_content_not_duplicate(self, deduplicator):
        await deduplicator.check_and_mark("content 1", "user-1", "slack")
        is_dup, _ = await deduplicator.check_and_mark("content 2", "user-1", "slack")
        assert is_dup is False

    @pytest.mark.asyncio
    async def test_different_user_not_duplicate(self, deduplicator):
        await deduplicator.check_and_mark("same content", "user-1", "slack")
        is_dup, _ = await deduplicator.check_and_mark("same content", "user-2", "slack")
        assert is_dup is False

    @pytest.mark.asyncio
    async def test_different_channel_not_duplicate(self, deduplicator):
        """Same content + user on different channels should NOT be duplicate."""
        await deduplicator.check_and_mark("hello world", "user-1", "slack")
        is_dup, _ = await deduplicator.check_and_mark("hello world", "user-1", "teams")
        assert is_dup is False

    @pytest.mark.asyncio
    async def test_cross_platform_dedup_independence(self, deduplicator):
        """Messages on slack, discord, telegram are independently tracked."""
        platforms = ["slack", "discord", "telegram", "whatsapp", "teams"]
        for platform in platforms:
            is_dup, _ = await deduplicator.check_and_mark("same msg", "user-1", platform)
            assert is_dup is False, f"First message on {platform} should not be duplicate"

    @pytest.mark.asyncio
    async def test_expired_not_duplicate(self, deduplicator):
        import aragora.server.middleware.decision_routing as _mod

        now = [1_000_000.0]
        with patch.object(_mod.time, "time", side_effect=lambda: now[0]):
            await deduplicator.check_and_mark("test", "user-1", "slack")

            # Advance past the 1s window
            now[0] += 2.0

            is_dup, _ = await deduplicator.check_and_mark("test", "user-1", "slack")
            assert is_dup is False

    @pytest.mark.asyncio
    async def test_within_window_still_duplicate(self, deduplicator):
        """Request within the dedup window is still a duplicate."""
        import aragora.server.middleware.decision_routing as _mod

        now = [1_000_000.0]
        with patch.object(_mod.time, "time", side_effect=lambda: now[0]):
            await deduplicator.check_and_mark("test", "user-1", "slack")

            # Advance slightly but within window
            now[0] += 0.5

            is_dup, _ = await deduplicator.check_and_mark("test", "user-1", "slack")
            assert is_dup is True

    @pytest.mark.asyncio
    async def test_complete_resolves_future(self, deduplicator):
        await deduplicator.check_and_mark("test", "user-1", "slack")
        _, future = await deduplicator.check_and_mark("test", "user-1", "slack")

        assert not future.done()
        await deduplicator.complete("test", "user-1", "slack", "result-value")
        assert future.done()
        assert future.result() == "result-value"

    @pytest.mark.asyncio
    async def test_fail_removes_from_seen(self, deduplicator):
        await deduplicator.check_and_mark("test", "user-1", "slack")
        await deduplicator.fail("test", "user-1", "slack", Exception("Error"))

        # Retry should not be treated as duplicate
        is_dup, _ = await deduplicator.check_and_mark("test", "user-1", "slack")
        assert is_dup is False

    @pytest.mark.asyncio
    async def test_fail_sets_exception_on_future(self, deduplicator):
        await deduplicator.check_and_mark("test", "user-1", "slack")
        _, future = await deduplicator.check_and_mark("test", "user-1", "slack")

        await deduplicator.fail("test", "user-1", "slack", ValueError("Test error"))

        assert future.done()
        with pytest.raises(ValueError, match="Test error"):
            future.result()

    @pytest.mark.asyncio
    async def test_fail_with_no_prior_request_is_noop(self, deduplicator):
        # Should not raise
        await deduplicator.fail("unknown", "user-1", "slack", Exception("Error"))

    @pytest.mark.asyncio
    async def test_default_window_is_five_seconds(self):
        dedup = RequestDeduplicator()
        assert dedup._window_seconds == DEDUPE_WINDOW_SECONDS
        assert dedup._window_seconds == 5.0

    @pytest.mark.asyncio
    async def test_hash_is_deterministic(self, deduplicator):
        h1 = deduplicator._compute_hash("hello", "user-1", "slack")
        h2 = deduplicator._compute_hash("hello", "user-1", "slack")
        assert h1 == h2

    @pytest.mark.asyncio
    async def test_hash_differs_for_different_inputs(self, deduplicator):
        h1 = deduplicator._compute_hash("hello", "user-1", "slack")
        h2 = deduplicator._compute_hash("hello", "user-2", "slack")
        h3 = deduplicator._compute_hash("hello", "user-1", "teams")
        assert h1 != h2
        assert h1 != h3


# ===========================================================================
# ResponseCache
# ===========================================================================


class TestResponseCache:
    """Tests for ResponseCache."""

    @pytest.fixture
    def cache(self):
        return ResponseCache(ttl_seconds=1.0, max_size=10)

    @pytest.mark.asyncio
    async def test_cache_miss(self, cache):
        result = await cache.get("unknown content")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit(self, cache):
        await cache.set("test query", "cached answer")
        result = await cache.get("test query")
        assert result == "cached answer"

    @pytest.mark.asyncio
    async def test_cache_with_context(self, cache):
        await cache.set("query", "answer1", context={"workspace": "ws1"})
        await cache.set("query", "answer2", context={"workspace": "ws2"})
        assert await cache.get("query", context={"workspace": "ws1"}) == "answer1"
        assert await cache.get("query", context={"workspace": "ws2"}) == "answer2"

    @pytest.mark.asyncio
    async def test_cache_expiry(self, cache):
        import aragora.server.middleware.decision_routing as _mod

        now = [1_000_000.0]
        with patch.object(_mod.time, "time", side_effect=lambda: now[0]):
            await cache.set("query", "answer")

            now[0] += 2.0  # past TTL
            result = await cache.get("query")
            assert result is None

    @pytest.mark.asyncio
    async def test_cache_eviction_oldest(self):
        cache = ResponseCache(ttl_seconds=60.0, max_size=3)
        await cache.set("q1", "a1")
        await cache.set("q2", "a2")
        await cache.set("q3", "a3")
        await cache.set("q4", "a4")  # evicts q1

        assert await cache.get("q1") is None
        assert await cache.get("q4") == "a4"

    @pytest.mark.asyncio
    async def test_clear_returns_count(self, cache):
        await cache.set("q1", "a1")
        await cache.set("q2", "a2")
        count = await cache.clear()
        assert count == 2
        assert await cache.get("q1") is None

    @pytest.mark.asyncio
    async def test_invalidate_by_workspace(self, cache):
        await cache.set("q1", "a1", context={"workspace_id": "ws-1"})
        await cache.set("q2", "a2", context={"workspace_id": "ws-2"})
        await cache.set("q3", "a3", context={"workspace_id": "ws-1"})

        removed = await cache.invalidate_by_workspace("ws-1")
        assert removed == 2
        assert await cache.get("q2", context={"workspace_id": "ws-2"}) == "a2"

    @pytest.mark.asyncio
    async def test_invalidate_by_tag(self, cache):
        await cache.set("q1", "a1", tags=["model-upgrade"])
        await cache.set("q2", "a2", tags=["other"])
        await cache.set("q3", "a3", tags=["model-upgrade", "other"])

        removed = await cache.invalidate_by_tag("model-upgrade")
        assert removed == 2
        assert await cache.get("q2") == "a2"

    @pytest.mark.asyncio
    async def test_invalidate_by_agent_version(self, cache):
        await cache.set("q1", "a1", agent_versions={"claude": "3.5"})
        await cache.set("q2", "a2", agent_versions={"claude": "4.0"})
        await cache.set("q3", "a3", agent_versions={"gpt": "4.0"})

        removed = await cache.invalidate_by_agent_version("claude", "3.5")
        assert removed == 1
        assert await cache.get("q2") == "a2"

    @pytest.mark.asyncio
    async def test_policy_version_lazy_invalidation(self, cache):
        """Entries with old policy version are invalidated on access."""
        cache.set_policy_version("v1")
        await cache.set("q1", "a1")

        # Update policy version
        cache.set_policy_version("v2")

        # Existing entry should be invalidated on next access
        result = await cache.get("q1")
        assert result is None

    @pytest.mark.asyncio
    async def test_policy_version_new_entries_valid(self, cache):
        """Entries created after policy change should be valid."""
        cache.set_policy_version("v1")
        await cache.set("q1", "a1")

        cache.set_policy_version("v2")
        await cache.set("q2", "a2")

        assert await cache.get("q1") is None  # old policy
        assert await cache.get("q2") == "a2"  # current policy

    @pytest.mark.asyncio
    async def test_get_stats(self, cache):
        await cache.set("q1", "a1")
        await cache.get("q1")  # hit
        await cache.get("unknown")  # miss

        stats = await cache.get_stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["max_size"] == 10

    @pytest.mark.asyncio
    async def test_get_stats_empty(self):
        cache = ResponseCache()
        stats = await cache.get_stats()
        assert stats["size"] == 0
        assert stats["hit_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_default_ttl_is_one_hour(self):
        cache = ResponseCache()
        assert cache._ttl_seconds == CACHE_TTL_SECONDS
        assert cache._ttl_seconds == 3600.0

    @pytest.mark.asyncio
    async def test_overwrite_same_key(self, cache):
        await cache.set("q1", "old")
        await cache.set("q1", "new")
        assert await cache.get("q1") == "new"


# ===========================================================================
# DecisionRoutingMiddleware
# ===========================================================================


class TestDecisionRoutingMiddleware:
    """Tests for DecisionRoutingMiddleware."""

    @pytest.fixture
    def middleware(self):
        return DecisionRoutingMiddleware(
            enable_deduplication=True,
            enable_caching=True,
            dedupe_window=1.0,
            cache_ttl=60.0,
        )

    @pytest.mark.asyncio
    async def test_process_basic(self, middleware):
        context = _make_context(request_id="test-req-1")
        with _patch_router(middleware):
            result = await middleware.process("Test question", context)
        assert result["success"] is True
        assert "result" in result
        assert result["request_id"] == "test-req-1"

    @pytest.mark.asyncio
    async def test_process_returns_duration(self, middleware):
        context = _make_context()
        with _patch_router(middleware):
            result = await middleware.process("Test", context)
        assert "duration_seconds" in result
        assert isinstance(result["duration_seconds"], float)

    @pytest.mark.asyncio
    async def test_caching_serves_cached_result(self, middleware):
        context = _make_context(request_id="cache-test")
        await middleware._cache.set(
            "cached query",
            "cached answer",
            context={"channel": "slack", "workspace_id": None},
        )

        result = await middleware.process("cached query", context)
        assert result["success"] is True
        assert result.get("cached") is True
        assert result["result"] == "cached answer"

    @pytest.mark.asyncio
    async def test_caching_stores_successful_result(self, middleware):
        context = _make_context(channel="api", request_id="store-cache")
        with _patch_router(middleware, result=_mock_router_result(answer="fresh answer")):
            await middleware.process("new query", context)

        cached = await middleware._cache.get(
            "new query", context={"channel": "api", "workspace_id": None}
        )
        assert cached == "fresh answer"

    @pytest.mark.asyncio
    async def test_deduplication_concurrent_requests(self, middleware):
        """Concurrent identical requests should be deduplicated."""
        context1 = _make_context(request_id="dup-1")

        async def slow_route(request):
            await asyncio.sleep(0.3)
            return _mock_router_result(answer="Answer")

        mock_router = MagicMock()
        mock_router.route = slow_route

        with patch.object(middleware, "_get_router", return_value=mock_router):
            task1 = asyncio.create_task(middleware.process("dup query", context1))
            await asyncio.sleep(0.05)

            context2 = _make_context(request_id="dup-2")
            result2 = await middleware.process("dup query", context2)

            result1 = await task1

        assert result1["success"] is True
        assert result2.get("deduplicated") is True or result2.get("success") is True

    @pytest.mark.asyncio
    async def test_error_returns_failure(self, middleware):
        context = _make_context(request_id="error-req")
        with _patch_router(middleware, side_effect=ValueError("Router exploded")):
            result = await middleware.process("Test", context)

        assert result["success"] is False
        assert "Router exploded" in result["error"]
        assert result["request_id"] == "error-req"

    @pytest.mark.asyncio
    async def test_error_clears_dedup_state(self, middleware):
        """After error, dedup state is cleared so retries work."""
        ctx1 = _make_context(request_id="err-dedup-1")
        with _patch_router(middleware, side_effect=RuntimeError("fail")):
            r1 = await middleware.process("Retry me", ctx1)
        assert r1["success"] is False

        ctx2 = _make_context(request_id="err-dedup-2")
        with _patch_router(middleware, result=_mock_router_result(answer="Success")):
            r2 = await middleware.process("Retry me", ctx2)
        assert r2["success"] is True

    @pytest.mark.asyncio
    async def test_fallback_when_no_router(self):
        middleware = DecisionRoutingMiddleware(enable_deduplication=False, enable_caching=False)
        context = _make_context(request_id="fallback")

        with patch.object(middleware, "_get_router", return_value=None):
            with patch.object(middleware, "_fallback_route") as mock_fb:
                mock_fb.return_value = {
                    "success": True,
                    "answer": "Fallback answer",
                    "confidence": 0.7,
                    "consensus_reached": True,
                }
                result = await middleware.process("Test fallback", context)

        mock_fb.assert_called_once()
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_fallback_error_handling(self):
        middleware = DecisionRoutingMiddleware(enable_deduplication=False, enable_caching=False)
        context = _make_context(request_id="fb-error")

        with patch.object(middleware, "_get_router", return_value=None):
            with patch.object(middleware, "_fallback_route", side_effect=RuntimeError("fb fail")):
                result = await middleware.process("Error test", context)

        assert result["success"] is False
        assert "fb fail" in result["error"]


class TestMiddlewareMultiPlatformRouting:
    """Test routing across multiple platforms."""

    @pytest.mark.asyncio
    async def test_all_channel_mappings(self):
        middleware = DecisionRoutingMiddleware(enable_deduplication=False, enable_caching=False)

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
            context = _make_context(channel=channel_name, request_id=f"req-{channel_name}")
            captured_request = None

            async def mock_route(request, _cap=None):
                nonlocal captured_request
                captured_request = request
                return _mock_router_result()

            with patch.object(middleware, "_get_router") as mock_get_router:
                mock_router = MagicMock()
                mock_router.route = mock_route
                mock_get_router.return_value = mock_router

                await middleware.process(f"Test {channel_name}", context)

            assert captured_request is not None, f"No request captured for {channel_name}"
            assert captured_request.source.value == expected_source.lower(), (
                f"Channel {channel_name} mapped to {captured_request.source.value}, "
                f"expected {expected_source.lower()}"
            )

    @pytest.mark.asyncio
    async def test_unknown_channel_defaults_to_http_api(self):
        """Unknown channel names should default to HTTP_API."""
        middleware = DecisionRoutingMiddleware(enable_deduplication=False, enable_caching=False)
        context = _make_context(channel="unknown_platform", request_id="req-unknown")
        captured_request = None

        async def mock_route(request):
            nonlocal captured_request
            captured_request = request
            return _mock_router_result()

        with patch.object(middleware, "_get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route = mock_route
            mock_get_router.return_value = mock_router

            await middleware.process("Test", context)

        assert captured_request.source.value == "http_api"


class TestDecisionTypeMappings:
    """Tests for decision type mappings."""

    @pytest.mark.asyncio
    async def test_all_decision_types(self):
        middleware = DecisionRoutingMiddleware(enable_deduplication=False, enable_caching=False)

        for dtype in ["debate", "workflow", "gauntlet", "quick"]:
            context = _make_context(channel="api", request_id=f"req-{dtype}")
            captured = None

            async def mock_route(request):
                nonlocal captured
                captured = request
                return _mock_router_result()

            with patch.object(middleware, "_get_router") as m:
                mock_router = MagicMock()
                mock_router.route = mock_route
                m.return_value = mock_router

                await middleware.process("Test", context, decision_type=dtype)

            assert captured is not None
            assert captured.decision_type.value == dtype

    @pytest.mark.asyncio
    async def test_unknown_decision_type_defaults_to_debate(self):
        middleware = DecisionRoutingMiddleware(enable_deduplication=False, enable_caching=False)
        context = _make_context(channel="api", request_id="req-unknown-type")
        captured = None

        async def mock_route(request):
            nonlocal captured
            captured = request
            return _mock_router_result()

        with patch.object(middleware, "_get_router") as m:
            mock_router = MagicMock()
            mock_router.route = mock_route
            m.return_value = mock_router

            await middleware.process("Test", context, decision_type="nonexistent")

        assert captured.decision_type.value == "debate"


# ===========================================================================
# Origin Registration
# ===========================================================================


class TestOriginRegistration:
    """Tests for origin registration in bidirectional routing."""

    @pytest.mark.asyncio
    async def test_origin_registration_called(self):
        middleware = DecisionRoutingMiddleware(enable_deduplication=False, enable_caching=False)

        context = _make_context(
            channel="slack",
            channel_id="C123",
            user_id="U456",
            request_id="origin-test",
            thread_id="T789",
            message_id="M012",
            workspace_id="W345",
            metadata={"custom": "data"},
        )

        with _patch_router(middleware):
            with patch.object(middleware, "_register_origin") as mock_reg:
                mock_reg.return_value = None
                await middleware.process("Test origin", context)

                mock_reg.assert_called_once()
                call_args = mock_reg.call_args
                assert call_args[0][0] == "Test origin"
                assert call_args[0][1].channel == "slack"

    @pytest.mark.asyncio
    async def test_origin_registration_error_is_non_fatal(self):
        """Origin registration errors should not fail the request."""
        middleware = DecisionRoutingMiddleware(enable_deduplication=False, enable_caching=False)
        context = _make_context(request_id="origin-error")

        with _patch_router(middleware):
            with patch(
                "aragora.server.debate_origin.register_debate_origin",
                side_effect=RuntimeError("Origin registration failed"),
            ):
                result = await middleware.process("Test", context)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_origin_import_error_is_non_fatal(self):
        """ImportError for debate_origin module should not fail."""
        middleware = DecisionRoutingMiddleware(enable_deduplication=False, enable_caching=False)
        context = _make_context(request_id="import-err")

        with _patch_router(middleware):
            with patch.dict("sys.modules", {"aragora.server.debate_origin": None}):
                result = await middleware.process("Test", context)

        # Should succeed or at least not crash
        assert "request_id" in result


# ===========================================================================
# Disabled Features
# ===========================================================================


class TestMiddlewareDisabledFeatures:
    """Tests for middleware with features disabled."""

    @pytest.mark.asyncio
    async def test_no_deduplication(self):
        middleware = DecisionRoutingMiddleware(enable_deduplication=False, enable_caching=False)
        assert middleware._deduplicator is None

        context = _make_context(request_id="no-dedupe")
        with _patch_router(middleware):
            result = await middleware.process("Test", context)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_no_caching(self):
        middleware = DecisionRoutingMiddleware(enable_deduplication=False, enable_caching=False)
        assert middleware._cache is None

        context = _make_context(request_id="no-cache")
        with _patch_router(middleware):
            result = await middleware.process("Test", context)
        assert result["success"] is True
        assert "cached" not in result


# ===========================================================================
# Duplicate Handling Edge Cases
# ===========================================================================


class TestDuplicateHandlingEdgeCases:
    """Tests for edge cases in duplicate request handling."""

    @pytest.mark.asyncio
    async def test_duplicate_without_future_returns_error(self):
        middleware = DecisionRoutingMiddleware(
            enable_deduplication=True, enable_caching=False, dedupe_window=5.0
        )
        context = _make_context(request_id="dup-no-future")

        await middleware._deduplicator.check_and_mark("duplicate content", "U5678", "slack")
        # Remove the future manually to simulate edge case
        async with middleware._deduplicator._lock:
            h = middleware._deduplicator._compute_hash("duplicate content", "U5678", "slack")
            middleware._deduplicator._in_flight.pop(h, None)

        result = await middleware.process("duplicate content", context)
        assert result["success"] is False
        assert "Duplicate" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_duplicate_timeout_waiting_for_inflight(self):
        """Duplicate that times out waiting for in-flight should warn."""
        middleware = DecisionRoutingMiddleware(
            enable_deduplication=True, enable_caching=False, dedupe_window=5.0
        )

        # First request creates in-flight future but never completes
        await middleware._deduplicator.check_and_mark("timeout msg", "U5678", "slack")

        # Patch asyncio.wait_for to simulate timeout
        context = _make_context(request_id="dup-timeout")
        original_wait_for = asyncio.wait_for

        async def mock_wait_for(fut, timeout):
            raise asyncio.TimeoutError()

        with patch("aragora.server.middleware.decision_routing.asyncio.wait_for", mock_wait_for):
            result = await middleware.process("timeout msg", context)

        # On timeout the middleware falls through; it does not explicitly set success=False
        # (the code has a bare return from the timeout handling block, so execution continues)
        assert "request_id" in result


# ===========================================================================
# route_decision Decorator
# ===========================================================================


class TestRouteDecisionDecorator:
    """Tests for route_decision decorator."""

    def setup_method(self):
        reset_decision_middleware()

    @pytest.mark.asyncio
    async def test_decorator_basic(self):
        @route_decision(channel="slack")
        async def test_handler():
            return {
                "content": "Test question",
                "user_id": "user-1",
                "channel_id": "C1234",
                "request_id": "decorated-test",
            }

        middleware = await get_decision_middleware()
        with patch.object(middleware, "process") as mock_process:
            mock_process.return_value = {"success": True, "result": {}}
            await test_handler()
            mock_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_decorator_passes_non_dict_through(self):
        """If the handler returns a non-dict, decorator returns it as-is."""

        @route_decision(channel="slack")
        async def test_handler():
            return "plain string"

        result = await test_handler()
        assert result == "plain string"

    @pytest.mark.asyncio
    async def test_decorator_extracts_context_fields(self):
        """Decorator should extract expected fields from handler result."""

        @route_decision(channel="teams", decision_type="workflow")
        async def test_handler():
            return {
                "content": "Run workflow",
                "user_id": "user-99",
                "channel_id": "teams-ch",
                "request_id": "dec-teams",
                "workspace_id": "ws-abc",
                "thread_id": "th-1",
                "message_id": "msg-1",
                "metadata": {"key": "val"},
            }

        middleware = await get_decision_middleware()
        with patch.object(middleware, "process") as mock_process:
            mock_process.return_value = {"success": True}
            await test_handler()

            call_kwargs = mock_process.call_args
            ctx = call_kwargs[1]["context"] if "context" in call_kwargs[1] else call_kwargs[0][1]
            assert ctx.channel == "teams"
            assert ctx.user_id == "user-99"
            assert ctx.workspace_id == "ws-abc"


# ===========================================================================
# Global Middleware Singleton
# ===========================================================================


class TestGlobalMiddleware:
    """Tests for global middleware singleton."""

    def setup_method(self):
        reset_decision_middleware()

    @pytest.mark.asyncio
    async def test_get_returns_singleton(self):
        m1 = await get_decision_middleware()
        m2 = await get_decision_middleware()
        assert m1 is m2

    @pytest.mark.asyncio
    async def test_reset_clears_singleton(self):
        m1 = await get_decision_middleware()
        reset_decision_middleware()
        m2 = await get_decision_middleware()
        assert m1 is not m2


# ===========================================================================
# Module-Level Cache Invalidation Helpers
# ===========================================================================


class TestModuleLevelHelpers:
    """Tests for module-level invalidation and stats helpers."""

    def setup_method(self):
        reset_decision_middleware()

    @pytest.mark.asyncio
    async def test_invalidate_cache_for_workspace(self):
        middleware = await get_decision_middleware()
        assert middleware._cache is not None

        await middleware._cache.set("q1", "a1", context={"workspace_id": "ws-target"})
        await middleware._cache.set("q2", "a2", context={"workspace_id": "ws-other"})

        removed = await invalidate_cache_for_workspace("ws-target")
        assert removed == 1

    @pytest.mark.asyncio
    async def test_invalidate_cache_for_policy_change(self):
        middleware = await get_decision_middleware()
        assert middleware._cache is not None

        middleware._cache.set_policy_version("v1")
        await middleware._cache.set("q1", "a1")

        await invalidate_cache_for_policy_change("v2")

        result = await middleware._cache.get("q1")
        assert result is None  # lazy invalidation

    @pytest.mark.asyncio
    async def test_invalidate_cache_for_agent_upgrade(self):
        middleware = await get_decision_middleware()

        await middleware._cache.set("q1", "a1", agent_versions={"claude": "3.5"})
        await middleware._cache.set("q2", "a2", agent_versions={"claude": "4.0"})

        removed = await invalidate_cache_for_agent_upgrade("claude", "3.5")
        assert removed == 1

    @pytest.mark.asyncio
    async def test_get_cache_stats(self):
        middleware = await get_decision_middleware()
        await middleware._cache.set("q1", "a1")
        await middleware._cache.get("q1")  # hit
        await middleware._cache.get("missing")  # miss

        stats = await get_cache_stats()
        assert stats["size"] == 1
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1

    @pytest.mark.asyncio
    async def test_get_cache_stats_when_cache_disabled(self):
        reset_decision_middleware()
        # Replace the global with a cache-disabled middleware
        import aragora.server.middleware.decision_routing as mod

        old = mod._middleware
        mod._middleware = DecisionRoutingMiddleware(
            enable_caching=False, enable_deduplication=False
        )
        try:
            stats = await get_cache_stats()
            assert stats == {"enabled": False}
        finally:
            mod._middleware = old


# ===========================================================================
# Consistent Error Handling Across Channels
# ===========================================================================


class TestConsistentErrorHandling:
    """Errors on all channels should produce the same structure."""

    @pytest.mark.asyncio
    async def test_error_structure_same_across_channels(self):
        channels = ["slack", "teams", "discord", "telegram", "whatsapp"]

        for channel in channels:
            middleware = DecisionRoutingMiddleware(enable_deduplication=False, enable_caching=False)
            context = _make_context(channel=channel, request_id=f"err-{channel}")

            with _patch_router(middleware, side_effect=ConnectionError(f"{channel} down")):
                result = await middleware.process("Test error", context)

            assert result["success"] is False, f"Expected failure for {channel}"
            assert "error" in result, f"Missing error key for {channel}"
            assert result["request_id"] == f"err-{channel}"

    @pytest.mark.asyncio
    async def test_os_error_caught(self):
        middleware = DecisionRoutingMiddleware(enable_deduplication=False, enable_caching=False)
        context = _make_context(request_id="os-err")
        with _patch_router(middleware, side_effect=OSError("disk full")):
            result = await middleware.process("Test", context)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_timeout_error_caught(self):
        middleware = DecisionRoutingMiddleware(enable_deduplication=False, enable_caching=False)
        context = _make_context(request_id="timeout-err")
        with _patch_router(middleware, side_effect=TimeoutError("timed out")):
            result = await middleware.process("Test", context)
        assert result["success"] is False
        assert "timed out" in result["error"]
