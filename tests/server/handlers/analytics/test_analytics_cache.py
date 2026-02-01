"""
Tests for Analytics Dashboard Cache Module.

Tests cover:
- AnalyticsDashboardCache singleton behavior
- Workspace-scoped caching
- TTL expiration
- Cache invalidation (single key, workspace, all)
- Decorator behavior (cached_analytics, cached_analytics_org)
- Cache hit/miss tracking
- Invalidation hooks
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.analytics.cache import (
    CACHE_CONFIGS,
    AnalyticsDashboardCache,
    CacheConfig,
    cached_analytics,
    cached_analytics_org,
    get_analytics_dashboard_cache,
    invalidate_analytics_cache,
    on_agent_performance_update,
    on_cost_event,
    on_debate_completed,
)


@pytest.fixture
def cache_instance():
    """Create a fresh cache instance for testing."""
    cache = AnalyticsDashboardCache()
    cache._ensure_initialized()
    return cache


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before each test."""
    AnalyticsDashboardCache._instance = None
    yield
    AnalyticsDashboardCache._instance = None


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_cache_config_defaults(self):
        """CacheConfig has sensible defaults."""
        config = CacheConfig(ttl_seconds=60, key_prefix="test")
        assert config.ttl_seconds == 60
        assert config.key_prefix == "test"
        assert config.maxsize == 200

    def test_cache_config_custom_maxsize(self):
        """CacheConfig accepts custom maxsize."""
        config = CacheConfig(ttl_seconds=300, key_prefix="custom", maxsize=500)
        assert config.maxsize == 500

    def test_predefined_cache_configs_exist(self):
        """Predefined cache configs exist for expected types."""
        expected_types = [
            "summary",
            "trends",
            "agents",
            "remediation",
            "cost",
            "tokens",
            "deliberations",
        ]
        for cache_type in expected_types:
            assert cache_type in CACHE_CONFIGS
            assert CACHE_CONFIGS[cache_type].ttl_seconds > 0
            assert CACHE_CONFIGS[cache_type].key_prefix.startswith("analytics_dashboard_")


class TestAnalyticsDashboardCache:
    """Tests for AnalyticsDashboardCache class."""

    def test_singleton_pattern(self):
        """Cache follows singleton pattern."""
        cache1 = AnalyticsDashboardCache.get_instance()
        cache2 = AnalyticsDashboardCache.get_instance()
        assert cache1 is cache2

    def test_get_analytics_dashboard_cache_returns_singleton(self):
        """get_analytics_dashboard_cache returns singleton."""
        cache1 = get_analytics_dashboard_cache()
        cache2 = get_analytics_dashboard_cache()
        assert cache1 is cache2

    def test_lazy_initialization(self):
        """Caches are lazily initialized."""
        cache = AnalyticsDashboardCache()
        assert not cache._initialized
        cache._ensure_initialized()
        assert cache._initialized
        assert len(cache._caches) > 0

    def test_get_set_basic(self, cache_instance):
        """Basic get/set operations work."""
        cache_instance.set("summary", "workspace-1", {"data": "test"}, "30d")
        result = cache_instance.get("summary", "workspace-1", "30d")
        assert result == {"data": "test"}

    def test_get_returns_none_for_missing(self, cache_instance):
        """Get returns None for missing keys."""
        result = cache_instance.get("summary", "nonexistent", "30d")
        assert result is None

    def test_cache_scoped_by_workspace(self, cache_instance):
        """Cache entries are scoped by workspace."""
        cache_instance.set("summary", "ws-1", {"ws": 1}, "30d")
        cache_instance.set("summary", "ws-2", {"ws": 2}, "30d")

        assert cache_instance.get("summary", "ws-1", "30d") == {"ws": 1}
        assert cache_instance.get("summary", "ws-2", "30d") == {"ws": 2}

    def test_cache_scoped_by_time_range(self, cache_instance):
        """Cache entries are scoped by time range."""
        cache_instance.set("summary", "ws-1", {"range": "24h"}, "24h")
        cache_instance.set("summary", "ws-1", {"range": "30d"}, "30d")

        assert cache_instance.get("summary", "ws-1", "24h") == {"range": "24h"}
        assert cache_instance.get("summary", "ws-1", "30d") == {"range": "30d"}

    def test_cache_scoped_by_extra_args(self, cache_instance):
        """Cache entries are scoped by extra args."""
        cache_instance.set("trends", "ws-1", {"gran": "hour"}, "30d", "hour")
        cache_instance.set("trends", "ws-1", {"gran": "day"}, "30d", "day")

        assert cache_instance.get("trends", "ws-1", "30d", "hour") == {"gran": "hour"}
        assert cache_instance.get("trends", "ws-1", "30d", "day") == {"gran": "day"}

    def test_invalidate_single_key(self, cache_instance):
        """Invalidate removes specific key."""
        cache_instance.set("summary", "ws-1", {"data": "test"}, "30d")
        assert cache_instance.get("summary", "ws-1", "30d") is not None

        result = cache_instance.invalidate("summary", "ws-1", "30d")
        assert result is True
        assert cache_instance.get("summary", "ws-1", "30d") is None

    def test_invalidate_returns_false_for_missing(self, cache_instance):
        """Invalidate returns False for missing keys."""
        result = cache_instance.invalidate("summary", "nonexistent", "30d")
        assert result is False

    def test_invalidate_workspace(self, cache_instance):
        """Invalidate workspace clears all entries for workspace."""
        # Set multiple entries for same workspace
        cache_instance.set("summary", "ws-1", {"type": "summary"}, "30d")
        cache_instance.set("trends", "ws-1", {"type": "trends"}, "30d")
        cache_instance.set("agents", "ws-1", {"type": "agents"}, "30d")
        # Set entry for different workspace
        cache_instance.set("summary", "ws-2", {"type": "summary"}, "30d")

        # Invalidate ws-1
        cleared = cache_instance.invalidate_workspace("ws-1")
        assert cleared >= 3

        # ws-1 entries should be gone
        assert cache_instance.get("summary", "ws-1", "30d") is None
        assert cache_instance.get("trends", "ws-1", "30d") is None
        assert cache_instance.get("agents", "ws-1", "30d") is None

        # ws-2 entry should remain
        assert cache_instance.get("summary", "ws-2", "30d") == {"type": "summary"}

    def test_invalidate_all(self, cache_instance):
        """Invalidate all clears all cache entries."""
        cache_instance.set("summary", "ws-1", {"data": 1}, "30d")
        cache_instance.set("summary", "ws-2", {"data": 2}, "30d")
        cache_instance.set("trends", "ws-1", {"data": 3}, "30d")

        cleared = cache_instance.invalidate_all()
        assert cleared >= 3

        assert cache_instance.get("summary", "ws-1", "30d") is None
        assert cache_instance.get("summary", "ws-2", "30d") is None
        assert cache_instance.get("trends", "ws-1", "30d") is None

    def test_get_stats(self, cache_instance):
        """Get stats returns statistics for all caches."""
        cache_instance.set("summary", "ws-1", {"data": 1}, "30d")
        cache_instance.get("summary", "ws-1", "30d")  # Hit
        cache_instance.get("summary", "ws-1", "24h")  # Miss

        stats = cache_instance.get_stats()
        assert "summary" in stats
        assert stats["summary"]["hits"] >= 1
        assert stats["summary"]["misses"] >= 1

    def test_get_summary_stats(self, cache_instance):
        """Get summary stats aggregates across caches."""
        cache_instance.set("summary", "ws-1", {"data": 1}, "30d")
        cache_instance.set("trends", "ws-1", {"data": 2}, "30d")

        summary = cache_instance.get_summary_stats()
        assert "cache_count" in summary
        assert "total_size" in summary
        assert "total_hits" in summary
        assert "total_misses" in summary
        assert "overall_hit_rate" in summary
        assert summary["total_size"] >= 2


class TestInvalidateAnalyticsCache:
    """Tests for invalidate_analytics_cache function."""

    def test_invalidate_specific_workspace(self):
        """Invalidate specific workspace works."""
        cache = get_analytics_dashboard_cache()
        cache.set("summary", "ws-test", {"data": 1}, "30d")

        cleared = invalidate_analytics_cache("ws-test")
        assert cleared >= 1
        assert cache.get("summary", "ws-test", "30d") is None

    def test_invalidate_all_when_no_workspace(self):
        """Invalidate all when no workspace specified."""
        cache = get_analytics_dashboard_cache()
        cache.set("summary", "ws-1", {"data": 1}, "30d")
        cache.set("summary", "ws-2", {"data": 2}, "30d")

        cleared = invalidate_analytics_cache(None)
        assert cleared >= 2


class TestCachedAnalyticsDecorator:
    """Tests for @cached_analytics decorator."""

    def test_caches_successful_response(self):
        """Decorator caches successful responses."""
        call_count = 0

        @dataclass
        class MockResult:
            status_code: int = 200
            data: str = "test"

        class MockHandler:
            @cached_analytics("summary", workspace_key="workspace_id")
            def get_summary(self, query_params, handler=None, user=None):
                nonlocal call_count
                call_count += 1
                return MockResult()

        handler = MockHandler()
        query = {"workspace_id": "ws-1", "time_range": "30d"}

        # First call - cache miss
        result1 = handler.get_summary(query)
        assert call_count == 1
        assert result1.status_code == 200

        # Second call - cache hit
        result2 = handler.get_summary(query)
        assert call_count == 1  # Still 1, no new call
        assert result2.status_code == 200

    def test_skips_caching_without_workspace(self):
        """Decorator skips caching without workspace_id."""
        call_count = 0

        @dataclass
        class MockResult:
            status_code: int = 200

        class MockHandler:
            @cached_analytics("summary", workspace_key="workspace_id")
            def get_summary(self, query_params, handler=None, user=None):
                nonlocal call_count
                call_count += 1
                return MockResult()

        handler = MockHandler()
        query = {"time_range": "30d"}  # No workspace_id

        # Should always call the function
        handler.get_summary(query)
        handler.get_summary(query)
        assert call_count == 2

    def test_does_not_cache_error_responses(self):
        """Decorator does not cache error responses."""
        call_count = 0

        @dataclass
        class MockResult:
            status_code: int = 500

        class MockHandler:
            @cached_analytics("summary", workspace_key="workspace_id")
            def get_summary(self, query_params, handler=None, user=None):
                nonlocal call_count
                call_count += 1
                return MockResult()

        handler = MockHandler()
        query = {"workspace_id": "ws-1", "time_range": "30d"}

        handler.get_summary(query)
        handler.get_summary(query)
        assert call_count == 2  # Called twice, not cached

    def test_respects_extra_keys(self):
        """Decorator respects extra cache keys."""
        call_count = 0

        @dataclass
        class MockResult:
            status_code: int = 200
            data: str = ""

        class MockHandler:
            @cached_analytics(
                "trends",
                workspace_key="workspace_id",
                extra_keys=["granularity"],
            )
            def get_trends(self, query_params, handler=None, user=None):
                nonlocal call_count
                call_count += 1
                return MockResult(data=query_params.get("granularity", "day"))

        handler = MockHandler()

        # Different granularity = different cache entry
        query1 = {"workspace_id": "ws-1", "time_range": "30d", "granularity": "hour"}
        query2 = {"workspace_id": "ws-1", "time_range": "30d", "granularity": "day"}

        handler.get_trends(query1)
        handler.get_trends(query2)
        assert call_count == 2  # Both called (different cache keys)

        handler.get_trends(query1)  # Cache hit
        assert call_count == 2


class TestCachedAnalyticsOrgDecorator:
    """Tests for @cached_analytics_org decorator."""

    def test_caches_by_org_id(self):
        """Decorator caches by org_id."""
        call_count = 0

        @dataclass
        class MockResult:
            status_code: int = 200

        class MockHandler:
            @cached_analytics_org("tokens", org_key="org_id", days_key="days")
            def get_tokens(self, query_params, handler=None, user=None):
                nonlocal call_count
                call_count += 1
                return MockResult()

        handler = MockHandler()
        query = {"org_id": "org-1", "days": "30"}

        handler.get_tokens(query)
        handler.get_tokens(query)
        assert call_count == 1  # Only called once

    def test_skips_caching_without_org(self):
        """Decorator skips caching without org_id."""
        call_count = 0

        @dataclass
        class MockResult:
            status_code: int = 200

        class MockHandler:
            @cached_analytics_org("tokens", org_key="org_id", days_key="days")
            def get_tokens(self, query_params, handler=None, user=None):
                nonlocal call_count
                call_count += 1
                return MockResult()

        handler = MockHandler()
        query = {"days": "30"}  # No org_id

        handler.get_tokens(query)
        handler.get_tokens(query)
        assert call_count == 2


class TestInvalidationHooks:
    """Tests for cache invalidation hooks."""

    def test_on_debate_completed_invalidates_relevant_caches(self):
        """on_debate_completed invalidates summary, trends, deliberations."""
        cache = get_analytics_dashboard_cache()
        cache.set("summary", "ws-1", {"data": 1}, "30d")
        cache.set("trends", "ws-1", {"data": 2}, "30d")
        cache.set("deliberations", "ws-1", {"data": 3}, "30d")
        cache.set("agents", "ws-1", {"data": 4}, "30d")

        on_debate_completed("ws-1")

        # These should be cleared
        assert cache.get("summary", "ws-1", "30d") is None
        assert cache.get("trends", "ws-1", "30d") is None
        assert cache.get("deliberations", "ws-1", "30d") is None
        # This should remain
        assert cache.get("agents", "ws-1", "30d") is not None

    def test_on_agent_performance_update_invalidates_agents(self):
        """on_agent_performance_update invalidates agents cache."""
        cache = get_analytics_dashboard_cache()
        cache.set("agents", "ws-1", {"data": 1}, "30d")
        cache.set("summary", "ws-1", {"data": 2}, "30d")

        on_agent_performance_update("ws-1")

        # Agents should be cleared
        assert cache.get("agents", "ws-1", "30d") is None
        # Summary should remain
        assert cache.get("summary", "ws-1", "30d") is not None

    def test_on_cost_event_invalidates_cost_and_tokens(self):
        """on_cost_event invalidates cost and tokens caches."""
        cache = get_analytics_dashboard_cache()
        cache.set("cost", "org-1", {"data": 1}, "30d")
        cache.set("tokens", "org-1", {"data": 2}, "30")
        cache.set("summary", "org-1", {"data": 3}, "30d")

        on_cost_event("org-1")

        # Cost and tokens should be cleared
        assert cache.get("cost", "org-1", "30d") is None
        assert cache.get("tokens", "org-1", "30") is None
        # Summary should remain
        assert cache.get("summary", "org-1", "30d") is not None


class TestCacheTTL:
    """Tests for cache TTL behavior."""

    def test_summary_cache_has_short_ttl(self):
        """Summary cache has short TTL (60s)."""
        config = CACHE_CONFIGS["summary"]
        assert config.ttl_seconds == 60

    def test_other_caches_have_longer_ttl(self):
        """Other caches have longer TTL (300s)."""
        for cache_type in ["trends", "agents", "cost", "tokens"]:
            config = CACHE_CONFIGS[cache_type]
            assert config.ttl_seconds >= 300

    def test_remediation_cache_has_memory_ttl(self):
        """Remediation cache uses memory TTL."""
        config = CACHE_CONFIGS["remediation"]
        # Should be CACHE_TTL_ANALYTICS_MEMORY (1800s / 30 min by default)
        assert config.ttl_seconds >= 300
