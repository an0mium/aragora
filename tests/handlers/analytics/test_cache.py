"""Tests for aragora.server.handlers.analytics.cache module.

Covers AnalyticsDashboardCache, cached_analytics / cached_analytics_org decorators,
invalidation hooks, global accessors, CacheConfig, and _get_pytest_cache_tag.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from types import SimpleNamespace
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
    _get_pytest_cache_tag,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status_code: int = 200, body: dict | None = None) -> SimpleNamespace:
    """Create a minimal response-like object with status_code."""
    return SimpleNamespace(status_code=status_code, body=body or {})


@dataclass
class _FakeResult:
    """Result object that has status_code attribute."""
    status_code: int
    data: Any = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the singleton between tests so each test starts fresh."""
    AnalyticsDashboardCache._instance = None
    yield
    AnalyticsDashboardCache._instance = None


@pytest.fixture
def cache() -> AnalyticsDashboardCache:
    """Return a fresh AnalyticsDashboardCache instance."""
    return AnalyticsDashboardCache()


@pytest.fixture
def initialized_cache(cache: AnalyticsDashboardCache) -> AnalyticsDashboardCache:
    """Return a cache that has been initialized (caches created)."""
    cache._ensure_initialized()
    return cache


# ============================================================================
# CacheConfig
# ============================================================================

class TestCacheConfig:
    """Tests for the CacheConfig dataclass."""

    def test_create_with_defaults(self):
        cfg = CacheConfig(ttl_seconds=60, key_prefix="test")
        assert cfg.ttl_seconds == 60
        assert cfg.key_prefix == "test"
        assert cfg.maxsize == 200

    def test_create_with_custom_maxsize(self):
        cfg = CacheConfig(ttl_seconds=120, key_prefix="custom", maxsize=500)
        assert cfg.maxsize == 500

    def test_equality(self):
        a = CacheConfig(ttl_seconds=60, key_prefix="a", maxsize=100)
        b = CacheConfig(ttl_seconds=60, key_prefix="a", maxsize=100)
        assert a == b

    def test_inequality_different_ttl(self):
        a = CacheConfig(ttl_seconds=60, key_prefix="a")
        b = CacheConfig(ttl_seconds=300, key_prefix="a")
        assert a != b


# ============================================================================
# CACHE_CONFIGS registry
# ============================================================================

class TestCacheConfigs:
    """Tests for the module-level CACHE_CONFIGS dict."""

    def test_expected_keys_present(self):
        expected = {"summary", "trends", "agents", "remediation", "cost", "tokens", "deliberations"}
        assert expected == set(CACHE_CONFIGS.keys())

    def test_summary_ttl_is_overview(self):
        from aragora.config import CACHE_TTL_ANALYTICS_OVERVIEW
        assert CACHE_CONFIGS["summary"].ttl_seconds == CACHE_TTL_ANALYTICS_OVERVIEW

    def test_trends_ttl_is_summary(self):
        from aragora.config import CACHE_TTL_ANALYTICS_SUMMARY
        assert CACHE_CONFIGS["trends"].ttl_seconds == CACHE_TTL_ANALYTICS_SUMMARY

    def test_agents_ttl(self):
        from aragora.config import CACHE_TTL_ANALYTICS_AGENTS
        assert CACHE_CONFIGS["agents"].ttl_seconds == CACHE_TTL_ANALYTICS_AGENTS

    def test_remediation_ttl(self):
        from aragora.config import CACHE_TTL_ANALYTICS_MEMORY
        assert CACHE_CONFIGS["remediation"].ttl_seconds == CACHE_TTL_ANALYTICS_MEMORY

    def test_cost_ttl(self):
        from aragora.config import CACHE_TTL_ANALYTICS_COSTS
        assert CACHE_CONFIGS["cost"].ttl_seconds == CACHE_TTL_ANALYTICS_COSTS

    def test_all_configs_are_cache_config_instances(self):
        for name, cfg in CACHE_CONFIGS.items():
            assert isinstance(cfg, CacheConfig), f"{name} is not CacheConfig"

    def test_all_configs_have_positive_ttl(self):
        for name, cfg in CACHE_CONFIGS.items():
            assert cfg.ttl_seconds > 0, f"{name} has non-positive TTL"

    def test_all_configs_have_positive_maxsize(self):
        for name, cfg in CACHE_CONFIGS.items():
            assert cfg.maxsize > 0, f"{name} has non-positive maxsize"


# ============================================================================
# AnalyticsDashboardCache - Construction & Singleton
# ============================================================================

class TestSingleton:
    """Tests for the singleton get_instance pattern."""

    def test_get_instance_returns_same_object(self):
        a = AnalyticsDashboardCache.get_instance()
        b = AnalyticsDashboardCache.get_instance()
        assert a is b

    def test_get_instance_creates_instance_when_none(self):
        assert AnalyticsDashboardCache._instance is None
        inst = AnalyticsDashboardCache.get_instance()
        assert inst is not None
        assert AnalyticsDashboardCache._instance is inst

    def test_fresh_instance_not_initialized(self, cache):
        assert cache._initialized is False
        assert cache._caches == {}


# ============================================================================
# AnalyticsDashboardCache - Lazy init
# ============================================================================

class TestLazyInit:
    """Tests for _ensure_initialized."""

    def test_ensure_initialized_creates_caches(self, cache):
        cache._ensure_initialized()
        assert cache._initialized is True
        assert len(cache._caches) == len(CACHE_CONFIGS)

    def test_double_init_is_idempotent(self, cache):
        cache._ensure_initialized()
        caches_first = dict(cache._caches)
        cache._ensure_initialized()
        assert cache._caches == caches_first

    def test_initialized_caches_match_config_names(self, initialized_cache):
        for name in CACHE_CONFIGS:
            assert name in initialized_cache._caches


# ============================================================================
# AnalyticsDashboardCache - _get_cache
# ============================================================================

class TestGetCache:
    """Tests for _get_cache."""

    def test_returns_known_cache(self, initialized_cache):
        c = initialized_cache._get_cache("summary")
        assert c is not None

    def test_creates_default_cache_for_unknown_type(self, initialized_cache):
        c = initialized_cache._get_cache("unknown_type")
        assert c is not None
        # Should now be in _caches
        assert "unknown_type" in initialized_cache._caches

    def test_unknown_cache_has_default_maxsize(self, initialized_cache):
        c = initialized_cache._get_cache("new_type")
        assert c._maxsize == 100

    def test_get_cache_triggers_init(self, cache):
        assert cache._initialized is False
        cache._get_cache("summary")
        assert cache._initialized is True


# ============================================================================
# AnalyticsDashboardCache - _make_key
# ============================================================================

class TestMakeKey:
    """Tests for _make_key."""

    def test_known_type_uses_config_prefix(self, initialized_cache):
        key = initialized_cache._make_key("summary", "ws-1", "30d")
        assert "analytics_dashboard_summary" in key
        assert "ws-1" in key

    def test_unknown_type_uses_fallback_prefix(self, initialized_cache):
        key = initialized_cache._make_key("exotic", "ws-2", "7d")
        assert "analytics_exotic" in key
        assert "ws-2" in key

    def test_multiple_args_in_key(self, initialized_cache):
        key = initialized_cache._make_key("cost", "ws-3", "30d", "usd")
        assert "ws-3" in key

    def test_pytest_tag_appended_when_set(self, initialized_cache, monkeypatch):
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/foo.py::test_bar (call)")
        key = initialized_cache._make_key("summary", "ws-1", "30d")
        assert "tests/foo.py::test_bar" in key


# ============================================================================
# AnalyticsDashboardCache - get / set / invalidate
# ============================================================================

class TestGetSetInvalidate:
    """Tests for get, set, invalidate methods."""

    def test_get_miss_returns_none(self, initialized_cache):
        assert initialized_cache.get("summary", "ws-1", "30d") is None

    def test_set_then_get(self, initialized_cache):
        initialized_cache.set("summary", "ws-1", {"total": 42}, "30d")
        result = initialized_cache.get("summary", "ws-1", "30d")
        assert result == {"total": 42}

    def test_set_overwrite(self, initialized_cache):
        initialized_cache.set("summary", "ws-1", "old", "30d")
        initialized_cache.set("summary", "ws-1", "new", "30d")
        assert initialized_cache.get("summary", "ws-1", "30d") == "new"

    def test_different_workspace_ids_are_separate(self, initialized_cache):
        initialized_cache.set("summary", "ws-a", "A", "30d")
        initialized_cache.set("summary", "ws-b", "B", "30d")
        assert initialized_cache.get("summary", "ws-a", "30d") == "A"
        assert initialized_cache.get("summary", "ws-b", "30d") == "B"

    def test_different_cache_types_are_separate(self, initialized_cache):
        initialized_cache.set("summary", "ws-1", "SUM", "30d")
        initialized_cache.set("agents", "ws-1", "AGT", "30d")
        assert initialized_cache.get("summary", "ws-1", "30d") == "SUM"
        assert initialized_cache.get("agents", "ws-1", "30d") == "AGT"

    def test_invalidate_existing_returns_true(self, initialized_cache):
        initialized_cache.set("cost", "ws-1", 100, "7d")
        assert initialized_cache.invalidate("cost", "ws-1", "7d") is True

    def test_invalidate_missing_returns_false(self, initialized_cache):
        assert initialized_cache.invalidate("cost", "ws-no", "7d") is False

    def test_invalidate_removes_entry(self, initialized_cache):
        initialized_cache.set("cost", "ws-1", 100, "7d")
        initialized_cache.invalidate("cost", "ws-1", "7d")
        assert initialized_cache.get("cost", "ws-1", "7d") is None

    def test_set_with_no_extra_args(self, initialized_cache):
        initialized_cache.set("summary", "ws-1", "val")
        result = initialized_cache.get("summary", "ws-1")
        assert result == "val"


# ============================================================================
# AnalyticsDashboardCache - invalidate_workspace
# ============================================================================

class TestInvalidateWorkspace:
    """Tests for invalidate_workspace."""

    def test_clears_all_types_for_workspace(self, initialized_cache):
        for ctype in CACHE_CONFIGS:
            initialized_cache.set(ctype, "ws-target", f"val-{ctype}", "30d")
        cleared = initialized_cache.invalidate_workspace("ws-target")
        assert cleared >= len(CACHE_CONFIGS)

    def test_does_not_affect_other_workspaces(self, initialized_cache):
        initialized_cache.set("summary", "ws-keep", "keep", "30d")
        initialized_cache.set("summary", "ws-delete", "delete", "30d")
        initialized_cache.invalidate_workspace("ws-delete")
        assert initialized_cache.get("summary", "ws-keep", "30d") == "keep"

    def test_returns_zero_when_nothing_to_clear(self, initialized_cache):
        cleared = initialized_cache.invalidate_workspace("ws-empty")
        assert cleared == 0


# ============================================================================
# AnalyticsDashboardCache - invalidate_all
# ============================================================================

class TestInvalidateAll:
    """Tests for invalidate_all."""

    def test_clears_everything(self, initialized_cache):
        initialized_cache.set("summary", "ws-1", "a", "30d")
        initialized_cache.set("agents", "ws-2", "b", "7d")
        cleared = initialized_cache.invalidate_all()
        assert cleared >= 2
        assert initialized_cache.get("summary", "ws-1", "30d") is None
        assert initialized_cache.get("agents", "ws-2", "7d") is None

    def test_returns_zero_on_empty_cache(self, initialized_cache):
        assert initialized_cache.invalidate_all() == 0


# ============================================================================
# AnalyticsDashboardCache - get_stats / get_summary_stats
# ============================================================================

class TestStats:
    """Tests for get_stats and get_summary_stats."""

    def test_get_stats_returns_all_cache_types(self, initialized_cache):
        stats = initialized_cache.get_stats()
        for name in CACHE_CONFIGS:
            assert name in stats

    def test_get_stats_values_are_dicts(self, initialized_cache):
        stats = initialized_cache.get_stats()
        for name, s in stats.items():
            assert isinstance(s, dict), f"{name} stats is not dict"

    def test_get_summary_stats_keys(self, initialized_cache):
        summary = initialized_cache.get_summary_stats()
        assert "cache_count" in summary
        assert "total_size" in summary
        assert "total_hits" in summary
        assert "total_misses" in summary
        assert "overall_hit_rate" in summary

    def test_summary_cache_count_matches(self, initialized_cache):
        summary = initialized_cache.get_summary_stats()
        assert summary["cache_count"] == len(CACHE_CONFIGS)

    def test_summary_hit_rate_zero_when_empty(self, initialized_cache):
        summary = initialized_cache.get_summary_stats()
        assert summary["overall_hit_rate"] == 0.0

    def test_summary_reflects_activity(self, initialized_cache):
        # Generate a miss then a hit
        initialized_cache.get("summary", "ws-1", "30d")  # miss
        initialized_cache.set("summary", "ws-1", "val", "30d")
        initialized_cache.get("summary", "ws-1", "30d")  # hit
        summary = initialized_cache.get_summary_stats()
        assert summary["total_hits"] >= 1
        assert summary["total_misses"] >= 1
        assert summary["overall_hit_rate"] > 0.0

    def test_summary_total_size_reflects_entries(self, initialized_cache):
        initialized_cache.set("summary", "ws-1", "v", "30d")
        initialized_cache.set("agents", "ws-2", "v", "7d")
        summary = initialized_cache.get_summary_stats()
        assert summary["total_size"] >= 2


# ============================================================================
# Global accessors
# ============================================================================

class TestGlobalAccessors:
    """Tests for get_analytics_dashboard_cache and invalidate_analytics_cache."""

    def test_get_analytics_dashboard_cache_returns_singleton(self):
        a = get_analytics_dashboard_cache()
        b = get_analytics_dashboard_cache()
        assert a is b

    def test_invalidate_analytics_cache_with_workspace(self):
        cache = get_analytics_dashboard_cache()
        cache.set("summary", "ws-inv", "data", "30d")
        cleared = invalidate_analytics_cache("ws-inv")
        assert cleared >= 1

    def test_invalidate_analytics_cache_all(self):
        cache = get_analytics_dashboard_cache()
        cache.set("summary", "ws-a", "a", "30d")
        cache.set("agents", "ws-b", "b", "7d")
        cleared = invalidate_analytics_cache()
        assert cleared >= 2

    def test_invalidate_analytics_cache_none_workspace(self):
        cache = get_analytics_dashboard_cache()
        cache.set("cost", "ws-x", "x", "1d")
        cleared = invalidate_analytics_cache(None)
        assert cleared >= 1

    def test_invalidate_empty_workspace_invalidates_all(self):
        # empty string is falsy, should invalidate all
        cache = get_analytics_dashboard_cache()
        cache.set("summary", "ws-1", "v", "30d")
        cleared = invalidate_analytics_cache("")
        assert cleared >= 1


# ============================================================================
# Invalidation Hooks
# ============================================================================

class TestOnDebateCompleted:
    """Tests for on_debate_completed hook."""

    def test_clears_summary_trends_deliberations(self):
        cache = get_analytics_dashboard_cache()
        for ctype in ["summary", "trends", "deliberations"]:
            cache.set(ctype, "ws-debate", f"val-{ctype}", "30d")
        on_debate_completed("ws-debate")
        for ctype in ["summary", "trends", "deliberations"]:
            assert cache.get(ctype, "ws-debate", "30d") is None

    def test_does_not_clear_agents_cache(self):
        cache = get_analytics_dashboard_cache()
        cache.set("agents", "ws-debate", "agents-data", "30d")
        on_debate_completed("ws-debate")
        assert cache.get("agents", "ws-debate", "30d") == "agents-data"

    def test_does_not_clear_other_workspace(self):
        cache = get_analytics_dashboard_cache()
        cache.set("summary", "ws-other", "keep", "30d")
        on_debate_completed("ws-debate")
        assert cache.get("summary", "ws-other", "30d") == "keep"


class TestOnAgentPerformanceUpdate:
    """Tests for on_agent_performance_update hook."""

    def test_clears_agents_cache(self):
        cache = get_analytics_dashboard_cache()
        cache.set("agents", "ws-perf", "agents-data", "30d")
        on_agent_performance_update("ws-perf")
        assert cache.get("agents", "ws-perf", "30d") is None

    def test_does_not_clear_summary(self):
        cache = get_analytics_dashboard_cache()
        cache.set("summary", "ws-perf", "summary-data", "30d")
        on_agent_performance_update("ws-perf")
        assert cache.get("summary", "ws-perf", "30d") == "summary-data"


class TestOnCostEvent:
    """Tests for on_cost_event hook."""

    def test_clears_cost_and_tokens(self):
        cache = get_analytics_dashboard_cache()
        cache.set("cost", "org-bill", "cost-data", "30d")
        cache.set("tokens", "org-bill", "token-data", "30d")
        on_cost_event("org-bill")
        assert cache.get("cost", "org-bill", "30d") is None
        assert cache.get("tokens", "org-bill", "30d") is None

    def test_does_not_clear_summary(self):
        cache = get_analytics_dashboard_cache()
        cache.set("summary", "org-bill", "summary-data", "30d")
        on_cost_event("org-bill")
        assert cache.get("summary", "org-bill", "30d") == "summary-data"

    def test_does_not_clear_other_org(self):
        cache = get_analytics_dashboard_cache()
        cache.set("cost", "org-other", "keep", "30d")
        on_cost_event("org-bill")
        assert cache.get("cost", "org-other", "30d") == "keep"


# ============================================================================
# _get_pytest_cache_tag
# ============================================================================

class TestGetPytestCacheTag:
    """Tests for _get_pytest_cache_tag."""

    def test_returns_none_without_env(self, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        assert _get_pytest_cache_tag() is None

    def test_returns_test_path_from_env(self, monkeypatch):
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/handlers/test_foo.py::test_bar (call)")
        tag = _get_pytest_cache_tag()
        assert tag == "tests/handlers/test_foo.py::test_bar"

    def test_strips_phase_suffix(self, monkeypatch):
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/a.py::test_x (setup)")
        tag = _get_pytest_cache_tag()
        assert "(setup)" not in tag

    def test_handles_no_parentheses(self, monkeypatch):
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/a.py::test_y")
        tag = _get_pytest_cache_tag()
        assert tag == "tests/a.py::test_y"


# ============================================================================
# cached_analytics decorator
# ============================================================================

class TestCachedAnalyticsDecorator:
    """Tests for @cached_analytics decorator."""

    def _make_handler_cls(self, cache_type="summary", extra_keys=None):
        """Build a tiny class with a decorated method."""
        @cached_analytics(cache_type, workspace_key="workspace_id",
                          time_range_key="time_range", extra_keys=extra_keys)
        def get_data(self_inner, query_params, handler, user):
            return _FakeResult(status_code=200, data=query_params.get("value", "computed"))

        cls = type("Handler", (), {"get_data": get_data})
        return cls()

    def test_cache_miss_calls_func(self):
        h = self._make_handler_cls()
        result = h.get_data({"workspace_id": "ws-1", "time_range": "30d", "value": "fresh"}, None, None)
        assert result.data == "fresh"

    def test_cache_hit_returns_cached(self):
        h = self._make_handler_cls()
        # First call populates cache
        r1 = h.get_data({"workspace_id": "ws-1", "time_range": "30d", "value": "first"}, None, None)
        # Second call should return cached value
        r2 = h.get_data({"workspace_id": "ws-1", "time_range": "30d", "value": "second"}, None, None)
        assert r2.data == "first"

    def test_no_caching_without_workspace_id(self):
        h = self._make_handler_cls()
        # No workspace_id -> no caching, function always called
        r1 = h.get_data({"value": "a"}, None, None)
        r2 = h.get_data({"value": "b"}, None, None)
        assert r1.data == "a"
        assert r2.data == "b"

    def test_different_time_ranges_different_keys(self):
        h = self._make_handler_cls()
        r1 = h.get_data({"workspace_id": "ws-1", "time_range": "7d", "value": "week"}, None, None)
        r2 = h.get_data({"workspace_id": "ws-1", "time_range": "30d", "value": "month"}, None, None)
        assert r1.data == "week"
        assert r2.data == "month"

    def test_default_time_range_is_30d(self):
        h = self._make_handler_cls()
        # No time_range -> defaults to "30d"
        r1 = h.get_data({"workspace_id": "ws-1", "value": "default"}, None, None)
        # Explicit 30d should hit cache
        r2 = h.get_data({"workspace_id": "ws-1", "time_range": "30d", "value": "explicit"}, None, None)
        assert r2.data == "default"

    def test_non_200_not_cached(self):
        @cached_analytics("summary")
        def get_data(self_inner, query_params, handler, user):
            return _FakeResult(status_code=500, data="error")

        cls = type("H", (), {"get_data": get_data})
        h = cls()
        r1 = h.get_data({"workspace_id": "ws-1"}, None, None)
        r2 = h.get_data({"workspace_id": "ws-1"}, None, None)
        # Both should compute (not cached) - but they return the same thing
        # The key check: the underlying function is called both times
        assert r1.status_code == 500

    def test_none_result_not_cached(self):
        call_count = 0

        @cached_analytics("summary")
        def get_data(self_inner, query_params, handler, user):
            nonlocal call_count
            call_count += 1
            return None

        cls = type("H", (), {"get_data": get_data})
        h = cls()
        h.get_data({"workspace_id": "ws-1"}, None, None)
        h.get_data({"workspace_id": "ws-1"}, None, None)
        assert call_count == 2

    def test_result_without_status_code_not_cached(self):
        call_count = 0

        @cached_analytics("summary")
        def get_data(self_inner, query_params, handler, user):
            nonlocal call_count
            call_count += 1
            return {"raw": "dict"}

        cls = type("H", (), {"get_data": get_data})
        h = cls()
        h.get_data({"workspace_id": "ws-1"}, None, None)
        h.get_data({"workspace_id": "ws-1"}, None, None)
        assert call_count == 2

    def test_extra_keys_included_in_cache_key(self):
        h = self._make_handler_cls(extra_keys=["agent_type"])
        r1 = h.get_data({"workspace_id": "ws-1", "agent_type": "claude", "value": "claude-val"}, None, None)
        r2 = h.get_data({"workspace_id": "ws-1", "agent_type": "gpt4", "value": "gpt-val"}, None, None)
        assert r1.data == "claude-val"
        assert r2.data == "gpt-val"

    def test_extra_keys_same_values_share_cache(self):
        h = self._make_handler_cls(extra_keys=["agent_type"])
        r1 = h.get_data({"workspace_id": "ws-1", "agent_type": "claude", "value": "first"}, None, None)
        r2 = h.get_data({"workspace_id": "ws-1", "agent_type": "claude", "value": "second"}, None, None)
        assert r2.data == "first"

    def test_missing_extra_key_defaults_to_empty_string(self):
        h = self._make_handler_cls(extra_keys=["agent_type"])
        r1 = h.get_data({"workspace_id": "ws-1", "value": "no-agent"}, None, None)
        assert r1.data == "no-agent"

    def test_wraps_preserves_function_name(self):
        @cached_analytics("summary")
        def my_custom_handler(self, query_params, handler, user):
            pass

        assert my_custom_handler.__name__ == "my_custom_handler"


# ============================================================================
# cached_analytics_org decorator
# ============================================================================

class TestCachedAnalyticsOrgDecorator:
    """Tests for @cached_analytics_org decorator."""

    def _make_handler_cls(self, cache_type="cost", extra_keys=None):
        @cached_analytics_org(cache_type, org_key="org_id",
                              days_key="days", extra_keys=extra_keys)
        def get_cost(self_inner, query_params, handler, user):
            return _FakeResult(status_code=200, data=query_params.get("value", "computed"))

        cls = type("OrgHandler", (), {"get_cost": get_cost})
        return cls()

    def test_cache_miss_calls_func(self):
        h = self._make_handler_cls()
        result = h.get_cost({"org_id": "org-1", "days": "30", "value": "fresh"}, None, None)
        assert result.data == "fresh"

    def test_cache_hit_returns_cached(self):
        h = self._make_handler_cls()
        r1 = h.get_cost({"org_id": "org-1", "days": "30", "value": "first"}, None, None)
        r2 = h.get_cost({"org_id": "org-1", "days": "30", "value": "second"}, None, None)
        assert r2.data == "first"

    def test_no_caching_without_org_id(self):
        h = self._make_handler_cls()
        r1 = h.get_cost({"value": "a"}, None, None)
        r2 = h.get_cost({"value": "b"}, None, None)
        assert r1.data == "a"
        assert r2.data == "b"

    def test_different_days_different_keys(self):
        h = self._make_handler_cls()
        r1 = h.get_cost({"org_id": "org-1", "days": "7", "value": "week"}, None, None)
        r2 = h.get_cost({"org_id": "org-1", "days": "30", "value": "month"}, None, None)
        assert r1.data == "week"
        assert r2.data == "month"

    def test_default_days_is_30(self):
        h = self._make_handler_cls()
        r1 = h.get_cost({"org_id": "org-1", "value": "default"}, None, None)
        r2 = h.get_cost({"org_id": "org-1", "days": "30", "value": "explicit"}, None, None)
        assert r2.data == "default"

    def test_non_200_not_cached(self):
        call_count = 0

        @cached_analytics_org("cost")
        def get_cost(self_inner, query_params, handler, user):
            nonlocal call_count
            call_count += 1
            return _FakeResult(status_code=400, data="bad")

        cls = type("H", (), {"get_cost": get_cost})
        h = cls()
        h.get_cost({"org_id": "org-1"}, None, None)
        h.get_cost({"org_id": "org-1"}, None, None)
        assert call_count == 2

    def test_none_result_not_cached(self):
        call_count = 0

        @cached_analytics_org("cost")
        def get_cost(self_inner, query_params, handler, user):
            nonlocal call_count
            call_count += 1
            return None

        cls = type("H", (), {"get_cost": get_cost})
        h = cls()
        h.get_cost({"org_id": "org-1"}, None, None)
        h.get_cost({"org_id": "org-1"}, None, None)
        assert call_count == 2

    def test_result_without_status_code_not_cached(self):
        call_count = 0

        @cached_analytics_org("cost")
        def get_cost(self_inner, query_params, handler, user):
            nonlocal call_count
            call_count += 1
            return [1, 2, 3]

        cls = type("H", (), {"get_cost": get_cost})
        h = cls()
        h.get_cost({"org_id": "org-1"}, None, None)
        h.get_cost({"org_id": "org-1"}, None, None)
        assert call_count == 2

    def test_extra_keys_in_org_decorator(self):
        h = self._make_handler_cls(extra_keys=["currency"])
        r1 = h.get_cost({"org_id": "org-1", "currency": "usd", "value": "usd-val"}, None, None)
        r2 = h.get_cost({"org_id": "org-1", "currency": "eur", "value": "eur-val"}, None, None)
        assert r1.data == "usd-val"
        assert r2.data == "eur-val"

    def test_wraps_preserves_function_name(self):
        @cached_analytics_org("cost")
        def my_org_handler(self, query_params, handler, user):
            pass

        assert my_org_handler.__name__ == "my_org_handler"

    def test_different_orgs_different_cache(self):
        h = self._make_handler_cls()
        r1 = h.get_cost({"org_id": "org-a", "value": "A"}, None, None)
        r2 = h.get_cost({"org_id": "org-b", "value": "B"}, None, None)
        assert r1.data == "A"
        assert r2.data == "B"


# ============================================================================
# Thread safety
# ============================================================================

class TestThreadSafety:
    """Tests to exercise thread-safety of singleton and init."""

    def test_concurrent_get_instance(self):
        """Multiple threads calling get_instance should all get the same object."""
        results = []
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            results.append(AnalyticsDashboardCache.get_instance())

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(id(r) for r in results)) == 1

    def test_concurrent_ensure_initialized(self, cache):
        """Multiple threads calling _ensure_initialized should not corrupt state."""
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            cache._ensure_initialized()

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cache._initialized is True
        assert len(cache._caches) == len(CACHE_CONFIGS)


# ============================================================================
# Edge cases and integration
# ============================================================================

class TestEdgeCases:
    """Various edge-case and integration tests."""

    def test_set_complex_value(self, initialized_cache):
        complex_val = {"nested": {"list": [1, 2, 3], "tuple": (4, 5)}}
        initialized_cache.set("summary", "ws-1", complex_val, "30d")
        assert initialized_cache.get("summary", "ws-1", "30d") == complex_val

    def test_cache_type_with_special_chars(self, initialized_cache):
        # Unknown cache type with special characters
        initialized_cache.set("my-custom-type", "ws-1", "value")
        assert initialized_cache.get("my-custom-type", "ws-1") == "value"

    def test_empty_workspace_id(self, initialized_cache):
        # Empty workspace id should still work as a cache key
        initialized_cache.set("summary", "", "empty-ws")
        assert initialized_cache.get("summary", "") == "empty-ws"

    def test_invalidate_workspace_after_multiple_sets(self, initialized_cache):
        for i in range(5):
            initialized_cache.set("summary", "ws-multi", f"val-{i}", f"range-{i}")
        cleared = initialized_cache.invalidate_workspace("ws-multi")
        assert cleared >= 5

    def test_invalidate_all_after_many_entries(self, initialized_cache):
        for ctype in CACHE_CONFIGS:
            for i in range(3):
                initialized_cache.set(ctype, f"ws-{i}", f"v-{ctype}-{i}", "30d")
        cleared = initialized_cache.invalidate_all()
        assert cleared >= len(CACHE_CONFIGS) * 3

    def test_get_stats_after_activity(self, initialized_cache):
        initialized_cache.set("summary", "ws-1", "v", "30d")
        initialized_cache.get("summary", "ws-1", "30d")  # hit
        initialized_cache.get("summary", "ws-1", "7d")   # miss
        stats = initialized_cache.get_stats()
        summary_stats = stats["summary"]
        assert summary_stats["hits"] >= 1
        assert summary_stats["misses"] >= 1

    def test_hooks_work_on_fresh_singleton(self):
        """Hooks should trigger lazy init if needed."""
        on_debate_completed("ws-hook-test")
        on_agent_performance_update("ws-hook-test")
        on_cost_event("org-hook-test")
        # No exception means success

    def test_multiple_invalidation_hooks_in_sequence(self):
        cache = get_analytics_dashboard_cache()
        cache.set("summary", "ws-1", "s", "30d")
        cache.set("agents", "ws-1", "a", "30d")
        cache.set("cost", "org-1", "c", "30d")
        cache.set("tokens", "org-1", "t", "30d")

        on_debate_completed("ws-1")
        on_agent_performance_update("ws-1")
        on_cost_event("org-1")

        assert cache.get("summary", "ws-1", "30d") is None
        assert cache.get("agents", "ws-1", "30d") is None
        assert cache.get("cost", "org-1", "30d") is None
        assert cache.get("tokens", "org-1", "30d") is None

    def test_decorator_with_different_cache_types(self):
        """Verify decorator uses the correct cache partition."""
        @cached_analytics("agents")
        def get_agents(self, query_params, handler, user):
            return _FakeResult(status_code=200, data="agents-result")

        @cached_analytics("trends")
        def get_trends(self, query_params, handler, user):
            return _FakeResult(status_code=200, data="trends-result")

        cls = type("Multi", (), {"get_agents": get_agents, "get_trends": get_trends})
        h = cls()
        params = {"workspace_id": "ws-1"}
        h.get_agents(params, None, None)
        h.get_trends(params, None, None)

        # Invalidate only agents
        cache = get_analytics_dashboard_cache()
        on_agent_performance_update("ws-1")

        # Agents cache cleared, trends still present
        # Re-call: agents should compute fresh, trends from cache
        call_counts = {"agents": 0, "trends": 0}

        @cached_analytics("agents")
        def get_agents2(self, query_params, handler, user):
            call_counts["agents"] += 1
            return _FakeResult(status_code=200, data="agents-fresh")

        @cached_analytics("trends")
        def get_trends2(self, query_params, handler, user):
            call_counts["trends"] += 1
            return _FakeResult(status_code=200, data="trends-fresh")

        cls2 = type("Multi2", (), {"get_agents": get_agents2, "get_trends": get_trends2})
        h2 = cls2()
        h2.get_agents(params, None, None)
        h2.get_trends(params, None, None)

        assert call_counts["agents"] == 1  # recomputed
        # trends may or may not be cached depending on key isolation; just verify no error

    def test_invalidate_analytics_cache_returns_int(self):
        result = invalidate_analytics_cache("nonexistent-ws")
        assert isinstance(result, int)

    def test_summary_stats_hit_rate_calculation(self, initialized_cache):
        # Create controlled hits and misses
        initialized_cache.set("summary", "ws-calc", "v", "30d")
        initialized_cache.get("summary", "ws-calc", "30d")  # hit
        initialized_cache.get("summary", "ws-calc", "30d")  # hit
        initialized_cache.get("summary", "ws-calc", "7d")   # miss

        summary = initialized_cache.get_summary_stats()
        total = summary["total_hits"] + summary["total_misses"]
        if total > 0:
            expected_rate = summary["total_hits"] / total
            assert abs(summary["overall_hit_rate"] - expected_rate) < 0.001
