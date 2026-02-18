"""
Tests for aragora/debate/convergence/cache.py

Covers:
- CachedSimilarity dataclass
- PairwiseSimilarityCache: put/get, symmetric keys, TTL expiry, LRU eviction, stats, clear
- Global cache manager: get/create, cleanup stale, at-capacity eviction, timestamps
- cleanup_similarity_cache, cleanup_stale_similarity_caches, evict_expired_cache_entries
- cleanup_stale_caches (two-phase public cleanup)
- get_cache_manager_stats
- _PeriodicCacheCleanup: start/stop, is_running, get_stats (no timing loops tested)
- _ensure_periodic_cleanup_started, stop_periodic_cleanup, get_periodic_cleanup_stats
- Global state isolation via fixtures
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _reset_global_state():
    """Reset all global state in the cache module to a clean baseline."""
    import aragora.debate.convergence.cache as cache_mod

    with cache_mod._similarity_cache_lock:
        cache_mod._similarity_cache_manager.clear()
        cache_mod._similarity_cache_timestamps.clear()

    # Stop and null-out the periodic cleanup singleton
    if cache_mod._periodic_cleanup is not None:
        cache_mod._periodic_cleanup.stop(timeout=2.0)
        cache_mod._periodic_cleanup = None


@pytest.fixture(autouse=True)
def isolated_global_state():
    """Ensure each test starts and ends with clean global cache state."""
    _reset_global_state()
    yield
    _reset_global_state()


@pytest.fixture
def cache():
    """A fresh PairwiseSimilarityCache with small defaults for fast tests."""
    from aragora.debate.convergence.cache import PairwiseSimilarityCache

    return PairwiseSimilarityCache(session_id="test-session", max_size=10, ttl_seconds=60.0)


@pytest.fixture
def tiny_cache():
    """Cache with max_size=3 to exercise LRU eviction easily."""
    from aragora.debate.convergence.cache import PairwiseSimilarityCache

    return PairwiseSimilarityCache(session_id="tiny-session", max_size=3, ttl_seconds=60.0)


# ===========================================================================
# CachedSimilarity dataclass
# ===========================================================================


class TestCachedSimilarity:
    def test_fields_assigned(self):
        from aragora.debate.convergence.cache import CachedSimilarity

        cs = CachedSimilarity(similarity=0.87, computed_at=12345.0)
        assert cs.similarity == 0.87
        assert cs.computed_at == 12345.0

    def test_zero_similarity(self):
        from aragora.debate.convergence.cache import CachedSimilarity

        cs = CachedSimilarity(similarity=0.0, computed_at=0.0)
        assert cs.similarity == 0.0
        assert cs.computed_at == 0.0

    def test_full_similarity(self):
        from aragora.debate.convergence.cache import CachedSimilarity

        cs = CachedSimilarity(similarity=1.0, computed_at=9999.9)
        assert cs.similarity == 1.0

    def test_equality(self):
        from aragora.debate.convergence.cache import CachedSimilarity

        a = CachedSimilarity(similarity=0.5, computed_at=100.0)
        b = CachedSimilarity(similarity=0.5, computed_at=100.0)
        assert a == b

    def test_inequality_on_similarity(self):
        from aragora.debate.convergence.cache import CachedSimilarity

        a = CachedSimilarity(similarity=0.5, computed_at=100.0)
        b = CachedSimilarity(similarity=0.6, computed_at=100.0)
        assert a != b


# ===========================================================================
# PairwiseSimilarityCache — basic attributes
# ===========================================================================


class TestPairwiseSimilarityCacheInit:
    def test_session_id_stored(self, cache):
        assert cache.session_id == "test-session"

    def test_max_size_stored(self, cache):
        assert cache.max_size == 10

    def test_ttl_stored(self, cache):
        assert cache.ttl_seconds == 60.0

    def test_initial_cache_empty(self, cache):
        stats = cache.get_stats()
        assert stats["size"] == 0

    def test_initial_hits_misses_zero(self, cache):
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_initial_hit_rate_zero(self, cache):
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.0

    def test_default_max_size(self):
        from aragora.debate.convergence.cache import PairwiseSimilarityCache

        c = PairwiseSimilarityCache(session_id="x")
        assert c.max_size == 1024

    def test_default_ttl(self):
        from aragora.debate.convergence.cache import PairwiseSimilarityCache

        c = PairwiseSimilarityCache(session_id="x")
        assert c.ttl_seconds == 600.0


# ===========================================================================
# _hash_text and _make_key internals
# ===========================================================================


class TestHashAndKey:
    def test_hash_text_returns_16_chars(self, cache):
        h = cache._hash_text("hello world")
        assert len(h) == 16

    def test_hash_text_is_hex(self, cache):
        h = cache._hash_text("hello world")
        int(h, 16)  # should not raise

    def test_hash_text_deterministic(self, cache):
        assert cache._hash_text("foo") == cache._hash_text("foo")

    def test_hash_text_differs_for_different_inputs(self, cache):
        assert cache._hash_text("hello") != cache._hash_text("world")

    def test_make_key_symmetric(self, cache):
        key_ab = cache._make_key("alpha", "beta")
        key_ba = cache._make_key("beta", "alpha")
        assert key_ab == key_ba

    def test_make_key_contains_colon(self, cache):
        key = cache._make_key("a", "b")
        assert ":" in key

    def test_make_key_same_text(self, cache):
        # Degenerate case: both texts identical
        key = cache._make_key("same", "same")
        assert key is not None and len(key) > 0

    def test_make_key_different_texts_differ(self, cache):
        assert cache._make_key("x", "y") != cache._make_key("x", "z")


# ===========================================================================
# PairwiseSimilarityCache — put / get
# ===========================================================================


class TestPutGet:
    def test_miss_on_empty_cache(self, cache):
        assert cache.get("hello", "world") is None

    def test_put_then_get(self, cache):
        cache.put("alpha", "beta", 0.75)
        result = cache.get("alpha", "beta")
        assert result == pytest.approx(0.75)

    def test_get_symmetric_ab_ba(self, cache):
        cache.put("alpha", "beta", 0.75)
        assert cache.get("beta", "alpha") == pytest.approx(0.75)

    def test_put_ba_get_ab(self, cache):
        cache.put("beta", "alpha", 0.42)
        assert cache.get("alpha", "beta") == pytest.approx(0.42)

    def test_put_overwrite_same_pair(self, cache):
        cache.put("a", "b", 0.3)
        cache.put("a", "b", 0.9)
        assert cache.get("a", "b") == pytest.approx(0.9)

    def test_put_zero_similarity(self, cache):
        cache.put("a", "b", 0.0)
        assert cache.get("a", "b") == pytest.approx(0.0)

    def test_put_one_similarity(self, cache):
        cache.put("a", "b", 1.0)
        assert cache.get("a", "b") == pytest.approx(1.0)

    def test_different_pairs_independent(self, cache):
        cache.put("a", "b", 0.1)
        cache.put("c", "d", 0.9)
        assert cache.get("a", "b") == pytest.approx(0.1)
        assert cache.get("c", "d") == pytest.approx(0.9)

    def test_miss_increments_misses(self, cache):
        cache.get("x", "y")
        assert cache.get_stats()["misses"] == 1

    def test_hit_increments_hits(self, cache):
        cache.put("a", "b", 0.5)
        cache.get("a", "b")
        assert cache.get_stats()["hits"] == 1

    def test_hit_rate_calculation(self, cache):
        cache.put("a", "b", 0.5)
        cache.get("a", "b")  # hit
        cache.get("c", "d")  # miss
        stats = cache.get_stats()
        assert stats["hit_rate"] == pytest.approx(0.5)

    def test_size_increases_on_put(self, cache):
        cache.put("a", "b", 0.5)
        assert cache.get_stats()["size"] == 1


# ===========================================================================
# TTL expiry
# ===========================================================================


class TestTTLExpiry:
    def test_get_returns_none_after_ttl(self, monkeypatch):
        from aragora.debate.convergence.cache import PairwiseSimilarityCache

        c = PairwiseSimilarityCache(session_id="s", max_size=10, ttl_seconds=30.0)

        fake_now = [1000.0]

        def fake_time():
            return fake_now[0]

        monkeypatch.setattr("aragora.debate.convergence.cache.time.time", fake_time)

        c.put("a", "b", 0.8)  # stored at t=1000

        fake_now[0] = 1031.0  # 31 seconds later — past 30s TTL
        assert c.get("a", "b") is None

    def test_get_returns_value_before_ttl(self, monkeypatch):
        from aragora.debate.convergence.cache import PairwiseSimilarityCache

        c = PairwiseSimilarityCache(session_id="s", max_size=10, ttl_seconds=30.0)

        fake_now = [1000.0]

        def fake_time():
            return fake_now[0]

        monkeypatch.setattr("aragora.debate.convergence.cache.time.time", fake_time)

        c.put("a", "b", 0.8)

        fake_now[0] = 1029.0  # 29 seconds — still valid
        assert c.get("a", "b") == pytest.approx(0.8)

    def test_expired_entry_removed_from_cache(self, monkeypatch):
        from aragora.debate.convergence.cache import PairwiseSimilarityCache

        c = PairwiseSimilarityCache(session_id="s", max_size=10, ttl_seconds=10.0)

        fake_now = [500.0]

        def fake_time():
            return fake_now[0]

        monkeypatch.setattr("aragora.debate.convergence.cache.time.time", fake_time)

        c.put("a", "b", 0.5)
        assert c.get_stats()["size"] == 1

        fake_now[0] = 515.0  # expired
        c.get("a", "b")  # triggers lazy removal
        assert c.get_stats()["size"] == 0

    def test_expired_increments_misses(self, monkeypatch):
        from aragora.debate.convergence.cache import PairwiseSimilarityCache

        c = PairwiseSimilarityCache(session_id="s", max_size=10, ttl_seconds=5.0)

        fake_now = [200.0]

        def fake_time():
            return fake_now[0]

        monkeypatch.setattr("aragora.debate.convergence.cache.time.time", fake_time)

        c.put("a", "b", 0.3)
        fake_now[0] = 210.0
        c.get("a", "b")
        assert c.get_stats()["misses"] == 1

    def test_expired_entries_counted_in_stats(self, monkeypatch):
        from aragora.debate.convergence.cache import PairwiseSimilarityCache

        c = PairwiseSimilarityCache(session_id="s", max_size=10, ttl_seconds=5.0)

        fake_now = [100.0]

        def fake_time():
            return fake_now[0]

        monkeypatch.setattr("aragora.debate.convergence.cache.time.time", fake_time)

        c.put("a", "b", 0.5)
        c.put("c", "d", 0.7)
        # Advance past TTL without calling get (lazy eviction not triggered)
        fake_now[0] = 110.0
        stats = c.get_stats()
        assert stats["expired_entries"] == 2


# ===========================================================================
# LRU eviction
# ===========================================================================


class TestLRUEviction:
    def test_put_evicts_oldest_when_full(self, tiny_cache):
        # Fill to max_size=3
        tiny_cache.put("a", "b", 0.1)
        tiny_cache.put("c", "d", 0.2)
        tiny_cache.put("e", "f", 0.3)
        assert tiny_cache.get_stats()["size"] == 3

        # Adding a 4th entry should evict the LRU entry
        tiny_cache.put("g", "h", 0.4)
        assert tiny_cache.get_stats()["size"] == 3

    def test_lru_entry_evicted_first(self, tiny_cache):
        tiny_cache.put("a", "b", 0.1)
        tiny_cache.put("c", "d", 0.2)
        tiny_cache.put("e", "f", 0.3)

        # Access "a","b" to make it most recently used
        tiny_cache.get("a", "b")

        # Now "c","d" is LRU — adding new entry should evict it
        tiny_cache.put("g", "h", 0.4)

        assert tiny_cache.get("c", "d") is None  # evicted
        assert tiny_cache.get("a", "b") == pytest.approx(0.1)  # still there
        assert tiny_cache.get("g", "h") == pytest.approx(0.4)  # new entry

    def test_put_at_exactly_max_size_ok(self, tiny_cache):
        for i in range(3):
            tiny_cache.put(f"text_{i}", f"other_{i}", float(i) / 10)
        assert tiny_cache.get_stats()["size"] == 3

    def test_cache_never_exceeds_max_size(self):
        from aragora.debate.convergence.cache import PairwiseSimilarityCache

        c = PairwiseSimilarityCache(session_id="x", max_size=5)
        for i in range(20):
            c.put(f"text_{i}", f"pair_{i}", float(i) / 20)
        assert c.get_stats()["size"] <= 5

    def test_max_size_reported_in_stats(self, tiny_cache):
        assert tiny_cache.get_stats()["max_size"] == 3


# ===========================================================================
# clear()
# ===========================================================================


class TestClear:
    def test_clear_empties_cache(self, cache):
        cache.put("a", "b", 0.5)
        cache.put("c", "d", 0.7)
        cache.clear()
        assert cache.get_stats()["size"] == 0

    def test_clear_resets_hits(self, cache):
        cache.put("a", "b", 0.5)
        cache.get("a", "b")
        cache.clear()
        assert cache.get_stats()["hits"] == 0

    def test_clear_resets_misses(self, cache):
        cache.get("x", "y")
        cache.clear()
        assert cache.get_stats()["misses"] == 0

    def test_clear_resets_hit_rate(self, cache):
        cache.put("a", "b", 0.5)
        cache.get("a", "b")
        cache.clear()
        assert cache.get_stats()["hit_rate"] == 0.0

    def test_get_after_clear_returns_none(self, cache):
        cache.put("a", "b", 0.5)
        cache.clear()
        assert cache.get("a", "b") is None

    def test_clear_on_empty_cache_ok(self, cache):
        cache.clear()  # should not raise
        assert cache.get_stats()["size"] == 0


# ===========================================================================
# evict_expired()
# ===========================================================================


class TestEvictExpired:
    def test_no_eviction_when_all_fresh(self, monkeypatch):
        from aragora.debate.convergence.cache import PairwiseSimilarityCache

        fake_now = [1000.0]
        monkeypatch.setattr("aragora.debate.convergence.cache.time.time", lambda: fake_now[0])

        c = PairwiseSimilarityCache(session_id="s", max_size=10, ttl_seconds=60.0)
        c.put("a", "b", 0.5)
        c.put("c", "d", 0.7)

        fake_now[0] = 1030.0  # 30s later — still fresh
        evicted = c.evict_expired()
        assert evicted == 0
        assert c.get_stats()["size"] == 2

    def test_evicts_all_expired_entries(self, monkeypatch):
        from aragora.debate.convergence.cache import PairwiseSimilarityCache

        fake_now = [1000.0]
        monkeypatch.setattr("aragora.debate.convergence.cache.time.time", lambda: fake_now[0])

        c = PairwiseSimilarityCache(session_id="s", max_size=10, ttl_seconds=10.0)
        c.put("a", "b", 0.5)
        c.put("c", "d", 0.7)

        fake_now[0] = 1015.0  # both expired
        evicted = c.evict_expired()
        assert evicted == 2
        assert c.get_stats()["size"] == 0

    def test_evicts_only_expired_not_fresh(self, monkeypatch):
        from aragora.debate.convergence.cache import PairwiseSimilarityCache

        fake_now = [1000.0]
        monkeypatch.setattr("aragora.debate.convergence.cache.time.time", lambda: fake_now[0])

        c = PairwiseSimilarityCache(session_id="s", max_size=10, ttl_seconds=20.0)
        c.put("a", "b", 0.5)  # at t=1000

        fake_now[0] = 1015.0
        c.put("c", "d", 0.7)  # at t=1015

        fake_now[0] = 1025.0  # "a","b" expired (25s), "c","d" fresh (10s)
        evicted = c.evict_expired()
        assert evicted == 1
        assert c.get("c", "d") == pytest.approx(0.7)

    def test_evict_expired_on_empty_returns_zero(self, cache):
        assert cache.evict_expired() == 0


# ===========================================================================
# get_stats()
# ===========================================================================


class TestGetStats:
    def test_stats_keys_present(self, cache):
        stats = cache.get_stats()
        expected_keys = {
            "session_id",
            "size",
            "max_size",
            "hits",
            "misses",
            "hit_rate",
            "expired_entries",
            "ttl_seconds",
        }
        assert expected_keys.issubset(stats.keys())

    def test_stats_session_id(self, cache):
        assert cache.get_stats()["session_id"] == "test-session"

    def test_stats_ttl_seconds(self, cache):
        assert cache.get_stats()["ttl_seconds"] == 60.0

    def test_stats_hit_rate_with_all_hits(self, cache):
        cache.put("a", "b", 0.5)
        cache.get("a", "b")
        cache.get("a", "b")
        stats = cache.get_stats()
        assert stats["hit_rate"] == 1.0

    def test_stats_hit_rate_with_all_misses(self, cache):
        cache.get("x", "y")
        cache.get("p", "q")
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.0


# ===========================================================================
# Thread safety basics
# ===========================================================================


class TestThreadSafety:
    def test_concurrent_puts_do_not_exceed_max_size(self):
        from aragora.debate.convergence.cache import PairwiseSimilarityCache

        c = PairwiseSimilarityCache(session_id="threaded", max_size=50)
        errors = []

        def writer(start):
            try:
                for i in range(start, start + 30):
                    c.put(f"text_{i}", f"other_{i}", float(i) / 100)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i * 30,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert c.get_stats()["size"] <= 50

    def test_concurrent_get_put_no_exception(self):
        from aragora.debate.convergence.cache import PairwiseSimilarityCache

        c = PairwiseSimilarityCache(session_id="threaded2", max_size=20)
        for i in range(10):
            c.put(f"a_{i}", f"b_{i}", 0.5)

        errors = []

        def reader():
            try:
                for i in range(10):
                    c.get(f"a_{i}", f"b_{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ===========================================================================
# Global cache manager — get_pairwise_similarity_cache
# ===========================================================================


class TestGetPairwiseSimilarityCache:
    def test_returns_pairwise_similarity_cache(self):
        from aragora.debate.convergence.cache import (
            PairwiseSimilarityCache,
            get_pairwise_similarity_cache,
        )

        c = get_pairwise_similarity_cache("session-1")
        assert isinstance(c, PairwiseSimilarityCache)

    def test_same_session_id_returns_same_object(self):
        from aragora.debate.convergence.cache import get_pairwise_similarity_cache

        c1 = get_pairwise_similarity_cache("session-x")
        c2 = get_pairwise_similarity_cache("session-x")
        assert c1 is c2

    def test_different_session_ids_return_different_objects(self):
        from aragora.debate.convergence.cache import get_pairwise_similarity_cache

        c1 = get_pairwise_similarity_cache("session-a")
        c2 = get_pairwise_similarity_cache("session-b")
        assert c1 is not c2

    def test_session_id_stored_on_cache(self):
        from aragora.debate.convergence.cache import get_pairwise_similarity_cache

        c = get_pairwise_similarity_cache("my-session")
        assert c.session_id == "my-session"

    def test_max_size_passed_through(self):
        from aragora.debate.convergence.cache import get_pairwise_similarity_cache

        c = get_pairwise_similarity_cache("s", max_size=256)
        assert c.max_size == 256

    def test_ttl_passed_through(self):
        from aragora.debate.convergence.cache import get_pairwise_similarity_cache

        c = get_pairwise_similarity_cache("s", ttl_seconds=120.0)
        assert c.ttl_seconds == 120.0

    def test_timestamp_updated_on_second_access(self, monkeypatch):
        import aragora.debate.convergence.cache as cache_mod

        fake_now = [1000.0]
        monkeypatch.setattr(cache_mod, "time", __import__("time"))

        c1 = cache_mod.get_pairwise_similarity_cache("ts-session")
        # Record initial timestamp
        ts1 = cache_mod._similarity_cache_timestamps.get("ts-session")

        # Advance time and re-access — timestamp should update
        time.sleep(0.01)
        cache_mod.get_pairwise_similarity_cache("ts-session")
        ts2 = cache_mod._similarity_cache_timestamps.get("ts-session")

        assert ts2 >= ts1

    def test_at_capacity_oldest_evicted(self, monkeypatch):
        import aragora.debate.convergence.cache as cache_mod
        import aragora.debate.convergence as conv_pkg

        # Patch effective max to a small number on BOTH the module and the
        # package re-export so _effective_max_similarity_caches() sees it.
        monkeypatch.setattr(cache_mod, "MAX_SIMILARITY_CACHES", 2)
        monkeypatch.setattr(cache_mod, "DEFAULT_MAX_SIMILARITY_CACHES", 1)  # must differ so pkg wins
        monkeypatch.setattr(conv_pkg, "MAX_SIMILARITY_CACHES", 2)

        cache_mod.get_pairwise_similarity_cache("old-1")
        cache_mod.get_pairwise_similarity_cache("old-2")

        with cache_mod._similarity_cache_lock:
            assert len(cache_mod._similarity_cache_manager) == 2

        # Adding a third should evict one old entry (net result <= 2)
        cache_mod.get_pairwise_similarity_cache("new-3")

        with cache_mod._similarity_cache_lock:
            assert len(cache_mod._similarity_cache_manager) <= 2


# ===========================================================================
# cleanup_similarity_cache
# ===========================================================================


class TestCleanupSimilarityCache:
    def test_removes_session_from_manager(self):
        import aragora.debate.convergence.cache as cache_mod

        cache_mod.get_pairwise_similarity_cache("to-remove")
        assert "to-remove" in cache_mod._similarity_cache_manager

        cache_mod.cleanup_similarity_cache("to-remove")
        assert "to-remove" not in cache_mod._similarity_cache_manager

    def test_removes_timestamp(self):
        import aragora.debate.convergence.cache as cache_mod

        cache_mod.get_pairwise_similarity_cache("ts-remove")
        cache_mod.cleanup_similarity_cache("ts-remove")
        assert "ts-remove" not in cache_mod._similarity_cache_timestamps

    def test_unknown_session_is_noop(self):
        import aragora.debate.convergence.cache as cache_mod

        # Should not raise for unknown session
        cache_mod.cleanup_similarity_cache("nonexistent-session")

    def test_cache_data_cleared_on_removal(self):
        import aragora.debate.convergence.cache as cache_mod

        c = cache_mod.get_pairwise_similarity_cache("clear-on-remove")
        c.put("a", "b", 0.5)
        cache_mod.cleanup_similarity_cache("clear-on-remove")
        # After cleanup the cache is cleared; get returns None
        assert c.get("a", "b") is None


# ===========================================================================
# cleanup_stale_similarity_caches
# ===========================================================================


class TestCleanupStaleSimilarityCaches:
    def test_no_cleanup_when_all_fresh(self, monkeypatch):
        import aragora.debate.convergence.cache as cache_mod

        fake_now = [1000.0]
        monkeypatch.setattr(cache_mod.time, "time", lambda: fake_now[0])

        cache_mod.get_pairwise_similarity_cache("fresh-1")
        cache_mod.get_pairwise_similarity_cache("fresh-2")

        fake_now[0] = 1500.0  # only 500s later
        cleaned = cache_mod.cleanup_stale_similarity_caches(max_age_seconds=3600)
        assert cleaned == 0
        with cache_mod._similarity_cache_lock:
            assert len(cache_mod._similarity_cache_manager) == 2

    def test_cleanup_removes_stale_sessions(self, monkeypatch):
        import aragora.debate.convergence.cache as cache_mod

        fake_now = [1000.0]
        monkeypatch.setattr(cache_mod.time, "time", lambda: fake_now[0])

        cache_mod.get_pairwise_similarity_cache("stale-1")
        cache_mod.get_pairwise_similarity_cache("stale-2")

        fake_now[0] = 5000.0  # 4000s later — well past 3600s
        cleaned = cache_mod.cleanup_stale_similarity_caches(max_age_seconds=3600)
        assert cleaned == 2

        with cache_mod._similarity_cache_lock:
            assert "stale-1" not in cache_mod._similarity_cache_manager
            assert "stale-2" not in cache_mod._similarity_cache_manager

    def test_cleanup_returns_count(self, monkeypatch):
        import aragora.debate.convergence.cache as cache_mod

        fake_now = [2000.0]
        monkeypatch.setattr(cache_mod.time, "time", lambda: fake_now[0])

        for i in range(5):
            cache_mod.get_pairwise_similarity_cache(f"stale-{i}")

        fake_now[0] = 10000.0
        cleaned = cache_mod.cleanup_stale_similarity_caches(max_age_seconds=1)
        assert cleaned == 5

    def test_cleanup_only_removes_stale(self, monkeypatch):
        import aragora.debate.convergence.cache as cache_mod

        fake_now = [1000.0]
        monkeypatch.setattr(cache_mod.time, "time", lambda: fake_now[0])

        cache_mod.get_pairwise_similarity_cache("old")

        fake_now[0] = 2000.0
        cache_mod.get_pairwise_similarity_cache("young")

        fake_now[0] = 3000.0  # old is 2000s, young is 1000s
        cleaned = cache_mod.cleanup_stale_similarity_caches(max_age_seconds=1500)
        assert cleaned == 1
        with cache_mod._similarity_cache_lock:
            assert "young" in cache_mod._similarity_cache_manager

    def test_default_max_age_used_when_not_specified(self, monkeypatch):
        import aragora.debate.convergence.cache as cache_mod

        fake_now = [1000.0]
        monkeypatch.setattr(cache_mod.time, "time", lambda: fake_now[0])

        cache_mod.get_pairwise_similarity_cache("default-age-test")
        # Advance past CACHE_MANAGER_TTL_SECONDS (3600)
        fake_now[0] = 1000.0 + cache_mod.CACHE_MANAGER_TTL_SECONDS + 1
        cleaned = cache_mod.cleanup_stale_similarity_caches()
        assert cleaned == 1


# ===========================================================================
# evict_expired_cache_entries
# ===========================================================================


class TestEvictExpiredCacheEntries:
    def test_returns_zero_when_no_caches(self):
        from aragora.debate.convergence.cache import evict_expired_cache_entries

        result = evict_expired_cache_entries()
        assert result == 0

    def test_returns_zero_when_entries_fresh(self, monkeypatch):
        import aragora.debate.convergence.cache as cache_mod

        fake_now = [1000.0]
        monkeypatch.setattr(cache_mod.time, "time", lambda: fake_now[0])

        c = cache_mod.get_pairwise_similarity_cache("fresh-entries")
        c.put("a", "b", 0.5)

        fake_now[0] = 1010.0  # still fresh (TTL=600)
        result = cache_mod.evict_expired_cache_entries()
        assert result == 0

    def test_evicts_expired_across_all_caches(self, monkeypatch):
        import aragora.debate.convergence.cache as cache_mod

        fake_now = [1000.0]
        monkeypatch.setattr(cache_mod.time, "time", lambda: fake_now[0])

        c1 = cache_mod.get_pairwise_similarity_cache("session-ev-1", ttl_seconds=5.0)
        c2 = cache_mod.get_pairwise_similarity_cache("session-ev-2", ttl_seconds=5.0)
        c1.put("a", "b", 0.3)
        c1.put("c", "d", 0.7)
        c2.put("e", "f", 0.9)

        fake_now[0] = 1010.0  # all expired (>5s TTL)
        result = cache_mod.evict_expired_cache_entries()
        assert result == 3


# ===========================================================================
# cleanup_stale_caches (public two-phase)
# ===========================================================================


class TestCleanupStaleCaches:
    def test_returns_dict_with_expected_keys(self):
        from aragora.debate.convergence.cache import cleanup_stale_caches

        result = cleanup_stale_caches()
        expected = {"cleaned_count", "entries_evicted", "remaining_count", "cleanup_time", "periodic_cleanup_running"}
        assert expected.issubset(result.keys())

    def test_cleaned_count_in_result(self, monkeypatch):
        import aragora.debate.convergence.cache as cache_mod

        fake_now = [1000.0]
        monkeypatch.setattr(cache_mod.time, "time", lambda: fake_now[0])

        cache_mod.get_pairwise_similarity_cache("stale-pub-1")
        fake_now[0] = 10000.0
        result = cache_mod.cleanup_stale_caches(max_age_seconds=1)
        assert result["cleaned_count"] >= 1

    def test_remaining_count_reflects_state(self):
        import aragora.debate.convergence.cache as cache_mod

        cache_mod.get_pairwise_similarity_cache("active-1")
        cache_mod.get_pairwise_similarity_cache("active-2")
        result = cache_mod.cleanup_stale_caches(max_age_seconds=99999)
        assert result["remaining_count"] == 2

    def test_cleanup_time_is_recent(self):
        from aragora.debate.convergence.cache import cleanup_stale_caches

        before = time.time()
        result = cleanup_stale_caches()
        after = time.time()
        assert before <= result["cleanup_time"] <= after

    def test_periodic_cleanup_running_key_exists(self):
        from aragora.debate.convergence.cache import cleanup_stale_caches

        result = cleanup_stale_caches()
        assert "periodic_cleanup_running" in result

    def test_none_max_age_uses_default(self):
        from aragora.debate.convergence.cache import cleanup_stale_caches

        # Should not raise and should use CACHE_MANAGER_TTL_SECONDS default
        result = cleanup_stale_caches(max_age_seconds=None)
        assert "cleaned_count" in result


# ===========================================================================
# get_cache_manager_stats
# ===========================================================================


class TestGetCacheManagerStats:
    def test_returns_dict_with_expected_keys(self):
        from aragora.debate.convergence.cache import get_cache_manager_stats

        stats = get_cache_manager_stats()
        expected = {
            "active_caches",
            "max_caches",
            "cache_ttl_seconds",
            "cleanup_interval_seconds",
            "periodic_cleanup",
            "caches",
        }
        assert expected.issubset(stats.keys())

    def test_active_caches_zero_when_empty(self):
        from aragora.debate.convergence.cache import get_cache_manager_stats

        stats = get_cache_manager_stats()
        assert stats["active_caches"] == 0

    def test_active_caches_reflects_sessions(self):
        import aragora.debate.convergence.cache as cache_mod

        cache_mod.get_pairwise_similarity_cache("mgr-1")
        cache_mod.get_pairwise_similarity_cache("mgr-2")
        stats = cache_mod.get_cache_manager_stats()
        assert stats["active_caches"] == 2

    def test_caches_dict_contains_session_details(self):
        import aragora.debate.convergence.cache as cache_mod

        cache_mod.get_pairwise_similarity_cache("detail-session")
        stats = cache_mod.get_cache_manager_stats()
        assert "detail-session" in stats["caches"]
        session_info = stats["caches"]["detail-session"]
        assert "age_seconds" in session_info
        assert "stats" in session_info

    def test_max_caches_is_positive(self):
        from aragora.debate.convergence.cache import get_cache_manager_stats

        stats = get_cache_manager_stats()
        assert stats["max_caches"] > 0

    def test_cache_ttl_seconds(self):
        import aragora.debate.convergence.cache as cache_mod

        stats = cache_mod.get_cache_manager_stats()
        assert stats["cache_ttl_seconds"] == cache_mod.CACHE_MANAGER_TTL_SECONDS

    def test_cleanup_interval_seconds(self):
        import aragora.debate.convergence.cache as cache_mod

        stats = cache_mod.get_cache_manager_stats()
        assert stats["cleanup_interval_seconds"] == cache_mod.PERIODIC_CLEANUP_INTERVAL_SECONDS

    def test_periodic_cleanup_key_is_dict(self):
        from aragora.debate.convergence.cache import get_cache_manager_stats

        stats = get_cache_manager_stats()
        assert isinstance(stats["periodic_cleanup"], dict)


# ===========================================================================
# _PeriodicCacheCleanup
# ===========================================================================


class TestPeriodicCacheCleanup:
    @pytest.fixture
    def cleanup(self):
        from aragora.debate.convergence.cache import _PeriodicCacheCleanup

        # Use a very long interval so the background loop won't fire during tests
        c = _PeriodicCacheCleanup(interval_seconds=9999.0)
        yield c
        if c.is_running():
            c.stop(timeout=2.0)

    def test_not_running_before_start(self, cleanup):
        assert not cleanup.is_running()

    def test_running_after_start(self, cleanup):
        cleanup.start()
        assert cleanup.is_running()

    def test_not_running_after_stop(self, cleanup):
        cleanup.start()
        cleanup.stop(timeout=2.0)
        assert not cleanup.is_running()

    def test_start_idempotent(self, cleanup):
        cleanup.start()
        cleanup.start()  # second call should be a no-op
        assert cleanup.is_running()

    def test_stop_before_start_noop(self, cleanup):
        cleanup.stop()  # should not raise
        assert not cleanup.is_running()

    def test_get_stats_keys(self, cleanup):
        stats = cleanup.get_stats()
        expected = {
            "running",
            "interval_seconds",
            "total_caches_cleaned",
            "total_entries_evicted",
            "last_cleanup_time",
        }
        assert expected.issubset(stats.keys())

    def test_get_stats_interval_seconds(self, cleanup):
        stats = cleanup.get_stats()
        assert stats["interval_seconds"] == 9999.0

    def test_get_stats_initial_totals_zero(self, cleanup):
        stats = cleanup.get_stats()
        assert stats["total_caches_cleaned"] == 0
        assert stats["total_entries_evicted"] == 0

    def test_get_stats_last_cleanup_time_none_initially(self, cleanup):
        stats = cleanup.get_stats()
        assert stats["last_cleanup_time"] is None

    def test_get_stats_running_reflects_state(self, cleanup):
        assert not cleanup.get_stats()["running"]
        cleanup.start()
        assert cleanup.get_stats()["running"]

    def test_thread_is_daemon(self, cleanup):
        cleanup.start()
        thread = cleanup._thread
        assert thread is not None
        assert thread.daemon is True


# ===========================================================================
# _ensure_periodic_cleanup_started / stop_periodic_cleanup / get_periodic_cleanup_stats
# ===========================================================================


class TestPeriodicCleanupGlobalFunctions:
    def test_ensure_starts_cleanup(self):
        import aragora.debate.convergence.cache as cache_mod

        cache_mod._ensure_periodic_cleanup_started()
        assert cache_mod._periodic_cleanup is not None
        assert cache_mod._periodic_cleanup.is_running()

    def test_ensure_idempotent(self):
        import aragora.debate.convergence.cache as cache_mod

        cache_mod._ensure_periodic_cleanup_started()
        first = cache_mod._periodic_cleanup
        cache_mod._ensure_periodic_cleanup_started()
        assert cache_mod._periodic_cleanup is first  # same instance

    def test_stop_periodic_cleanup(self):
        import aragora.debate.convergence.cache as cache_mod

        cache_mod._ensure_periodic_cleanup_started()
        assert cache_mod._periodic_cleanup.is_running()
        cache_mod.stop_periodic_cleanup()
        assert not cache_mod._periodic_cleanup.is_running()

    def test_stop_when_not_started_noop(self):
        import aragora.debate.convergence.cache as cache_mod

        # _periodic_cleanup is None (reset by fixture)
        cache_mod.stop_periodic_cleanup()  # should not raise

    def test_get_periodic_cleanup_stats_when_none(self):
        import aragora.debate.convergence.cache as cache_mod

        # Ensure it's None (fixture resets)
        assert cache_mod._periodic_cleanup is None
        stats = cache_mod.get_periodic_cleanup_stats()
        assert stats["running"] is False
        assert stats["total_caches_cleaned"] == 0
        assert stats["total_entries_evicted"] == 0
        assert stats["last_cleanup_time"] is None
        assert stats["interval_seconds"] == cache_mod.PERIODIC_CLEANUP_INTERVAL_SECONDS

    def test_get_periodic_cleanup_stats_when_running(self):
        import aragora.debate.convergence.cache as cache_mod

        cache_mod._ensure_periodic_cleanup_started()
        stats = cache_mod.get_periodic_cleanup_stats()
        assert stats["running"] is True

    def test_restart_after_stop(self):
        import aragora.debate.convergence.cache as cache_mod

        cache_mod._ensure_periodic_cleanup_started()
        cache_mod.stop_periodic_cleanup()
        assert not cache_mod._periodic_cleanup.is_running()
        # Start again
        cache_mod._ensure_periodic_cleanup_started()
        assert cache_mod._periodic_cleanup.is_running()


# ===========================================================================
# Edge cases and integration
# ===========================================================================


class TestEdgeCases:
    def test_empty_string_texts(self, cache):
        cache.put("", "", 1.0)
        assert cache.get("", "") == pytest.approx(1.0)

    def test_very_long_texts(self, cache):
        long_text = "x" * 10000
        other_long = "y" * 10000
        cache.put(long_text, other_long, 0.33)
        assert cache.get(long_text, other_long) == pytest.approx(0.33)

    def test_unicode_texts(self, cache):
        cache.put("こんにちは", "привет", 0.12)
        assert cache.get("こんにちは", "привет") == pytest.approx(0.12)

    def test_negative_similarity_stored(self, cache):
        # Unusual but shouldn't crash
        cache.put("a", "b", -0.5)
        assert cache.get("a", "b") == pytest.approx(-0.5)

    def test_same_text_for_both_keys(self, cache):
        cache.put("identical", "identical", 1.0)
        assert cache.get("identical", "identical") == pytest.approx(1.0)

    def test_multiple_sessions_isolated(self):
        import aragora.debate.convergence.cache as cache_mod

        c1 = cache_mod.get_pairwise_similarity_cache("iso-1")
        c2 = cache_mod.get_pairwise_similarity_cache("iso-2")
        c1.put("a", "b", 0.9)
        assert c2.get("a", "b") is None

    def test_cache_accessible_after_get_pairwise(self):
        import aragora.debate.convergence.cache as cache_mod

        c = cache_mod.get_pairwise_similarity_cache("accessible")
        c.put("x", "y", 0.55)
        # Re-fetching same session gives same cache with data intact
        c2 = cache_mod.get_pairwise_similarity_cache("accessible")
        assert c2.get("x", "y") == pytest.approx(0.55)

    def test_cleanup_similarity_cache_then_re_create(self):
        import aragora.debate.convergence.cache as cache_mod

        c1 = cache_mod.get_pairwise_similarity_cache("recreate-me")
        c1.put("a", "b", 0.7)
        cache_mod.cleanup_similarity_cache("recreate-me")

        # New cache for same session starts fresh
        c2 = cache_mod.get_pairwise_similarity_cache("recreate-me")
        assert c2.get("a", "b") is None

    def test_evict_expired_cache_entries_handles_error(self, monkeypatch):
        """evict_expired_cache_entries should swallow exceptions from individual caches."""
        import aragora.debate.convergence.cache as cache_mod

        # Add a bad cache entry directly
        bad_cache = MagicMock()
        bad_cache.evict_expired.side_effect = RuntimeError("simulated error")
        bad_cache.session_id = "bad-session"

        with cache_mod._similarity_cache_lock:
            cache_mod._similarity_cache_manager["bad-session"] = bad_cache
            cache_mod._similarity_cache_timestamps["bad-session"] = time.time()

        # Should not raise despite the error
        result = cache_mod.evict_expired_cache_entries()
        assert isinstance(result, int)
