"""Tests for aragora.debate.context.cache — ContextCache."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from aragora.debate.context.cache import ContextCache


# ---------------------------------------------------------------------------
# ContextCache — init
# ---------------------------------------------------------------------------


class TestContextCacheInit:
    def test_defaults(self):
        cache = ContextCache()
        assert cache._max_evidence_size > 0
        assert cache._max_context_size > 0
        assert cache._max_continuum_size > 0
        assert cache._max_trending_size > 0

    def test_custom_sizes(self):
        cache = ContextCache(
            max_evidence_size=10,
            max_context_size=20,
            max_continuum_size=30,
            max_trending_size=5,
        )
        assert cache._max_evidence_size == 10
        assert cache._max_context_size == 20
        assert cache._max_continuum_size == 30
        assert cache._max_trending_size == 5

    def test_empty_caches(self):
        cache = ContextCache()
        assert cache._research_evidence_pack == {}
        assert cache._research_context_cache == {}
        assert cache._continuum_context_cache == {}
        assert cache._trending_topics_cache == []


# ---------------------------------------------------------------------------
# get_task_hash
# ---------------------------------------------------------------------------


class TestGetTaskHash:
    def test_returns_string(self):
        h = ContextCache.get_task_hash("test task")
        assert isinstance(h, str)
        assert len(h) == 16

    def test_deterministic(self):
        h1 = ContextCache.get_task_hash("test task")
        h2 = ContextCache.get_task_hash("test task")
        assert h1 == h2

    def test_different_tasks(self):
        h1 = ContextCache.get_task_hash("task A")
        h2 = ContextCache.get_task_hash("task B")
        assert h1 != h2

    def test_hex_format(self):
        h = ContextCache.get_task_hash("any task")
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Evidence pack
# ---------------------------------------------------------------------------


class TestEvidencePack:
    def test_get_miss(self):
        cache = ContextCache()
        assert cache.get_evidence_pack("task") is None

    def test_set_and_get(self):
        cache = ContextCache()
        pack = {"snippets": []}
        cache.set_evidence_pack("task", pack)
        assert cache.get_evidence_pack("task") is pack

    def test_task_isolation(self):
        cache = ContextCache()
        cache.set_evidence_pack("task1", "pack1")
        cache.set_evidence_pack("task2", "pack2")
        assert cache.get_evidence_pack("task1") == "pack1"
        assert cache.get_evidence_pack("task2") == "pack2"

    def test_get_latest(self):
        cache = ContextCache()
        cache.set_evidence_pack("task1", "pack1")
        cache.set_evidence_pack("task2", "pack2")
        assert cache.get_latest_evidence_pack() == "pack2"

    def test_get_latest_empty(self):
        cache = ContextCache()
        assert cache.get_latest_evidence_pack() is None

    def test_eviction(self):
        cache = ContextCache(max_evidence_size=2)
        cache.set_evidence_pack("t1", "p1")
        cache.set_evidence_pack("t2", "p2")
        cache.set_evidence_pack("t3", "p3")  # Should evict t1
        assert cache.get_evidence_pack("t1") is None
        assert cache.get_evidence_pack("t3") == "p3"


# ---------------------------------------------------------------------------
# Context cache
# ---------------------------------------------------------------------------


class TestContextOperations:
    def test_get_miss(self):
        cache = ContextCache()
        assert cache.get_context("task") is None

    def test_set_and_get(self):
        cache = ContextCache()
        cache.set_context("task", "context text")
        assert cache.get_context("task") == "context text"

    def test_eviction(self):
        cache = ContextCache(max_context_size=2)
        cache.set_context("t1", "c1")
        cache.set_context("t2", "c2")
        cache.set_context("t3", "c3")
        assert cache.get_context("t1") is None


# ---------------------------------------------------------------------------
# Continuum context cache
# ---------------------------------------------------------------------------


class TestContinuumContext:
    def test_get_miss(self):
        cache = ContextCache()
        assert cache.get_continuum_context("task") is None

    def test_set_and_get(self):
        cache = ContextCache()
        cache.set_continuum_context("task", "continuum data")
        assert cache.get_continuum_context("task") == "continuum data"

    def test_eviction(self):
        cache = ContextCache(max_continuum_size=2)
        cache.set_continuum_context("t1", "c1")
        cache.set_continuum_context("t2", "c2")
        cache.set_continuum_context("t3", "c3")
        assert cache.get_continuum_context("t1") is None


# ---------------------------------------------------------------------------
# Trending topics
# ---------------------------------------------------------------------------


class TestTrendingTopics:
    def test_get_empty(self):
        cache = ContextCache()
        assert cache.get_trending_topics() == []

    def test_set_and_get(self):
        cache = ContextCache()
        topics = ["topic1", "topic2"]
        cache.set_trending_topics(topics)
        assert cache.get_trending_topics() == ["topic1", "topic2"]

    def test_truncated_to_max(self):
        cache = ContextCache(max_trending_size=2)
        cache.set_trending_topics(["a", "b", "c", "d"])
        assert len(cache.get_trending_topics()) == 2

    def test_set_makes_copy(self):
        cache = ContextCache()
        topics = ["a", "b"]
        cache.set_trending_topics(topics)
        topics.append("c")
        assert len(cache.get_trending_topics()) == 2


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_all(self):
        cache = ContextCache()
        cache.set_evidence_pack("t", "p")
        cache.set_context("t", "c")
        cache.set_continuum_context("t", "cc")
        cache.set_trending_topics(["topic"])
        cache.clear()
        assert cache.get_evidence_pack("t") is None
        assert cache.get_context("t") is None
        assert cache.get_continuum_context("t") is None
        assert cache.get_trending_topics() == []

    def test_clear_specific_task(self):
        cache = ContextCache()
        cache.set_evidence_pack("t1", "p1")
        cache.set_evidence_pack("t2", "p2")
        cache.set_context("t1", "c1")
        cache.set_context("t2", "c2")
        cache.clear(task="t1")
        assert cache.get_evidence_pack("t1") is None
        assert cache.get_evidence_pack("t2") == "p2"
        assert cache.get_context("t1") is None
        assert cache.get_context("t2") == "c2"

    def test_clear_specific_preserves_trending(self):
        cache = ContextCache()
        cache.set_trending_topics(["topic"])
        cache.clear(task="t1")
        # Trending not cleared for task-specific clear
        assert cache.get_trending_topics() == ["topic"]


# ---------------------------------------------------------------------------
# merge_evidence_pack
# ---------------------------------------------------------------------------


class TestMergeEvidencePack:
    def test_no_existing(self):
        cache = ContextCache()
        pack = MagicMock()
        pack.snippets = [MagicMock(id="s1")]
        pack.total_searched = 10
        result = cache.merge_evidence_pack("task", pack)
        assert result is pack

    def test_merge_with_existing(self):
        cache = ContextCache()
        existing = MagicMock()
        existing.snippets = [MagicMock(id="s1")]
        existing.total_searched = 5
        cache.set_evidence_pack("task", existing)

        new_pack = MagicMock()
        new_pack.snippets = [MagicMock(id="s2")]
        new_pack.total_searched = 3

        result = cache.merge_evidence_pack("task", new_pack)
        assert len(result.snippets) == 2
        assert result.total_searched == 8

    def test_deduplication(self):
        cache = ContextCache()
        existing = MagicMock()
        existing.snippets = [MagicMock(id="s1")]
        existing.total_searched = 5
        cache.set_evidence_pack("task", existing)

        new_pack = MagicMock()
        new_pack.snippets = [MagicMock(id="s1"), MagicMock(id="s2")]  # s1 is duplicate
        new_pack.total_searched = 3

        result = cache.merge_evidence_pack("task", new_pack)
        assert len(result.snippets) == 2  # s1 + s2, not s1 + s1 + s2


# ---------------------------------------------------------------------------
# _enforce_cache_limit
# ---------------------------------------------------------------------------


class TestEnforceCacheLimit:
    def test_under_limit(self):
        cache = ContextCache()
        d = {"a": 1, "b": 2}
        cache._enforce_cache_limit(d, 5)
        assert len(d) == 2

    def test_at_limit(self):
        cache = ContextCache()
        d = {"a": 1, "b": 2}
        cache._enforce_cache_limit(d, 2)
        assert len(d) == 1  # Evicted one

    def test_fifo_order(self):
        cache = ContextCache()
        d = {"first": 1, "second": 2, "third": 3}
        cache._enforce_cache_limit(d, 2)
        assert "first" not in d
        assert "second" not in d
        assert "third" in d
