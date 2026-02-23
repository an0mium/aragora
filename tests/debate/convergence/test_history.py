"""
Tests for aragora/debate/convergence/history.py

Covers:
- ConvergenceHistoryStore: init defaults, custom max_records
- store: returns topic_hash, record retrievable, deterministic hash
- store: per_round_similarity defaults to empty list
- store: LRU eviction at max_records (oldest removed)
- store: with ContinuumMemory (mock add called with SLOW tier)
- store: ContinuumMemory errors caught gracefully
- find_similar: no records returns empty, exact match, partial overlap,
  no overlap, limit, sorted by Jaccard relevance
- get_record: existing hash, unknown hash
- get_stats: empty store, with records and correct averages
- clear: removes all records and topic_index
- Singleton: get/set/init convergence_history_store
"""

from __future__ import annotations

import hashlib
import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton before and after each test."""
    import aragora.debate.convergence.history as mod

    original = mod._convergence_history_store
    mod._convergence_history_store = None
    yield
    mod._convergence_history_store = original


@pytest.fixture
def store():
    """A fresh ConvergenceHistoryStore with default settings."""
    from aragora.debate.convergence.history import ConvergenceHistoryStore

    return ConvergenceHistoryStore()


@pytest.fixture
def small_store():
    """A ConvergenceHistoryStore with max_records=3 for eviction tests."""
    from aragora.debate.convergence.history import ConvergenceHistoryStore

    return ConvergenceHistoryStore(max_records=3)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


class TestInit:
    """Test ConvergenceHistoryStore initialization."""

    def test_defaults(self):
        from aragora.debate.convergence.history import ConvergenceHistoryStore

        s = ConvergenceHistoryStore()
        assert s._continuum_memory is None
        assert s._max_records == 1000
        assert s._records == {}
        assert s._topic_index == {}

    def test_custom_max_records(self):
        from aragora.debate.convergence.history import ConvergenceHistoryStore

        s = ConvergenceHistoryStore(max_records=50)
        assert s._max_records == 50

    def test_with_continuum_memory(self):
        from aragora.debate.convergence.history import ConvergenceHistoryStore

        mock_mem = MagicMock()
        s = ConvergenceHistoryStore(continuum_memory=mock_mem)
        assert s._continuum_memory is mock_mem


# ---------------------------------------------------------------------------
# Store tests
# ---------------------------------------------------------------------------


class TestStore:
    """Test the store() method."""

    def test_returns_topic_hash(self, store):
        topic = "Design a rate limiter"
        topic_hash = store.store(topic, convergence_round=3, total_rounds=5, final_similarity=0.85)
        expected = hashlib.md5(topic.encode(), usedforsecurity=False).hexdigest()
        assert topic_hash == expected

    def test_record_retrievable(self, store):
        topic = "Design a rate limiter"
        topic_hash = store.store(topic, convergence_round=3, total_rounds=5, final_similarity=0.85)
        record = store.get_record(topic_hash)
        assert record is not None
        assert record["topic_hash"] == topic_hash
        assert record["convergence_round"] == 3
        assert record["total_rounds"] == 5
        assert record["final_similarity"] == 0.85
        assert record["debate_id"] == ""
        assert "timestamp" in record

    def test_deterministic_hash(self, store):
        """Same topic always produces the same hash."""
        topic = "Evaluate trade-offs of microservices"
        h1 = store.store(topic, convergence_round=2, total_rounds=4, final_similarity=0.9)
        h2 = store.store(topic, convergence_round=3, total_rounds=5, final_similarity=0.95)
        assert h1 == h2

    def test_per_round_similarity_defaults_to_empty_list(self, store):
        topic_hash = store.store(
            "topic A", convergence_round=1, total_rounds=3, final_similarity=0.7
        )
        record = store.get_record(topic_hash)
        assert record["per_round_similarity"] == []

    def test_per_round_similarity_stored(self, store):
        sims = [0.3, 0.6, 0.85]
        topic_hash = store.store(
            "topic B",
            convergence_round=3,
            total_rounds=3,
            final_similarity=0.85,
            per_round_similarity=sims,
        )
        record = store.get_record(topic_hash)
        assert record["per_round_similarity"] == sims

    def test_debate_id_stored(self, store):
        topic_hash = store.store(
            "topic C",
            convergence_round=2,
            total_rounds=4,
            final_similarity=0.8,
            debate_id="debate-123",
        )
        record = store.get_record(topic_hash)
        assert record["debate_id"] == "debate-123"

    def test_lru_eviction_at_max_records(self, small_store):
        """When at max_records, the oldest record (by timestamp) is evicted."""
        s = small_store  # max_records=3

        # Store 3 records with controlled timestamps
        h1 = s.store("topic one", convergence_round=1, total_rounds=3, final_similarity=0.5)
        h2 = s.store("topic two", convergence_round=2, total_rounds=3, final_similarity=0.6)
        h3 = s.store("topic three", convergence_round=3, total_rounds=3, final_similarity=0.7)

        # All 3 should be present
        assert s.get_record(h1) is not None
        assert s.get_record(h2) is not None
        assert s.get_record(h3) is not None

        # Store a 4th -- oldest (h1) should be evicted
        h4 = s.store("topic four", convergence_round=4, total_rounds=5, final_similarity=0.8)
        assert s.get_record(h1) is None, "Oldest record should have been evicted"
        assert s.get_record(h2) is not None
        assert s.get_record(h3) is not None
        assert s.get_record(h4) is not None

    def test_lru_eviction_removes_from_topic_index(self, small_store):
        """Evicted records are also removed from the topic_index."""
        s = small_store

        h1 = s.store("alpha topic", convergence_round=1, total_rounds=2, final_similarity=0.5)
        s.store("beta topic", convergence_round=1, total_rounds=2, final_similarity=0.6)
        s.store("gamma topic", convergence_round=1, total_rounds=2, final_similarity=0.7)

        assert h1 in s._topic_index

        # Trigger eviction
        s.store("delta topic", convergence_round=1, total_rounds=2, final_similarity=0.8)
        assert h1 not in s._topic_index

    def test_store_with_continuum_memory(self):
        """When ContinuumMemory is provided, store persists to it."""
        from aragora.debate.convergence.history import ConvergenceHistoryStore

        mock_mem = MagicMock()
        mock_tier = MagicMock()
        mock_tier.SLOW = "SLOW"

        s = ConvergenceHistoryStore(continuum_memory=mock_mem)

        with patch(
            "aragora.debate.convergence.history.MemoryTier",
            mock_tier,
            create=True,
        ):
            # Patch the import inside store()
            with patch.dict(
                "sys.modules",
                {"aragora.memory.continuum": MagicMock(MemoryTier=mock_tier)},
            ):
                topic_hash = s.store(
                    "persistent topic",
                    convergence_round=2,
                    total_rounds=4,
                    final_similarity=0.88,
                    debate_id="d-42",
                )

        mock_mem.add.assert_called_once()
        call_kwargs = mock_mem.add.call_args
        assert call_kwargs.kwargs["id"] == f"convergence_history_{topic_hash}"
        assert call_kwargs.kwargs["tier"] == "SLOW"
        assert call_kwargs.kwargs["importance"] == 0.4
        assert call_kwargs.kwargs["metadata"]["topic_hash"] == topic_hash
        assert call_kwargs.kwargs["metadata"]["debate_id"] == "d-42"

    def test_store_continuum_memory_error_caught(self):
        """ContinuumMemory errors are caught gracefully without propagating."""
        from aragora.debate.convergence.history import ConvergenceHistoryStore

        mock_mem = MagicMock()

        s = ConvergenceHistoryStore(continuum_memory=mock_mem)

        # Make the import succeed but add() raise
        with patch.dict(
            "sys.modules",
            {"aragora.memory.continuum": MagicMock()},
        ):
            mock_mem.add.side_effect = RuntimeError("connection lost")
            # Should not raise
            topic_hash = s.store(
                "failing topic",
                convergence_round=1,
                total_rounds=2,
                final_similarity=0.5,
            )

        # Record should still be stored in-memory
        assert s.get_record(topic_hash) is not None

    def test_store_continuum_memory_import_error_caught(self):
        """ImportError from ContinuumMemory is caught gracefully."""
        from aragora.debate.convergence.history import ConvergenceHistoryStore

        mock_mem = MagicMock()
        s = ConvergenceHistoryStore(continuum_memory=mock_mem)

        # Make the import inside store() raise ImportError
        with patch.dict("sys.modules", {"aragora.memory.continuum": None}):
            # Should not raise
            topic_hash = s.store(
                "import error topic",
                convergence_round=1,
                total_rounds=2,
                final_similarity=0.5,
            )

        assert s.get_record(topic_hash) is not None


# ---------------------------------------------------------------------------
# find_similar tests
# ---------------------------------------------------------------------------


class TestFindSimilar:
    """Test the find_similar() method."""

    def test_no_records_returns_empty(self, store):
        assert store.find_similar("anything") == []

    def test_exact_topic_match(self, store):
        topic = "Design a rate limiter for APIs"
        store.store(topic, convergence_round=3, total_rounds=5, final_similarity=0.9)
        results = store.find_similar(topic)
        assert len(results) == 1
        assert results[0]["convergence_round"] == 3

    def test_partial_keyword_overlap(self, store):
        store.store(
            "design a caching strategy",
            convergence_round=2,
            total_rounds=4,
            final_similarity=0.8,
        )
        # "design" and "a" overlap
        results = store.find_similar("design a messaging queue")
        assert len(results) == 1

    def test_no_keyword_overlap_returns_nothing(self, store):
        store.store(
            "kubernetes deployment strategies",
            convergence_round=2,
            total_rounds=4,
            final_similarity=0.8,
        )
        results = store.find_similar("quantum entanglement paradox")
        assert len(results) == 0

    def test_respects_limit(self, store):
        for i in range(10):
            store.store(
                f"common topic variation {i}",
                convergence_round=i,
                total_rounds=10,
                final_similarity=0.5 + i * 0.05,
            )
        results = store.find_similar("common topic variation", limit=3)
        assert len(results) == 3

    def test_sorted_by_jaccard_relevance(self, store):
        """Results should be sorted by Jaccard similarity descending."""
        # Record with fewer overlapping words
        store.store(
            "alpha beta gamma delta epsilon",
            convergence_round=1,
            total_rounds=3,
            final_similarity=0.5,
        )
        # Record with more overlapping words
        store.store(
            "alpha beta gamma",
            convergence_round=2,
            total_rounds=3,
            final_similarity=0.7,
        )

        # Search with "alpha beta gamma" -- the second record has perfect Jaccard (1.0)
        # while the first has 3/5 = 0.6
        results = store.find_similar("alpha beta gamma")
        assert len(results) == 2
        assert results[0]["convergence_round"] == 2  # Perfect match first
        assert results[1]["convergence_round"] == 1  # Partial match second

    def test_empty_topic_returns_empty(self, store):
        """An empty topic string with no words should return empty."""
        store.store("some topic", convergence_round=1, total_rounds=2, final_similarity=0.5)
        # Empty string after split produces no words
        results = store.find_similar("")
        assert results == []


# ---------------------------------------------------------------------------
# get_record tests
# ---------------------------------------------------------------------------


class TestGetRecord:
    """Test the get_record() method."""

    def test_existing_hash_returns_record(self, store):
        topic_hash = store.store(
            "test topic", convergence_round=2, total_rounds=4, final_similarity=0.8
        )
        record = store.get_record(topic_hash)
        assert record is not None
        assert record["topic_hash"] == topic_hash

    def test_unknown_hash_returns_none(self, store):
        assert store.get_record("nonexistent_hash_value") is None


# ---------------------------------------------------------------------------
# get_stats tests
# ---------------------------------------------------------------------------


class TestGetStats:
    """Test the get_stats() method."""

    def test_empty_store(self, store):
        stats = store.get_stats()
        assert stats["record_count"] == 0
        assert stats["max_records"] == 1000
        assert "avg_convergence_round" not in stats
        assert "avg_total_rounds" not in stats
        assert "avg_final_similarity" not in stats

    def test_with_records_averages_correct(self, store):
        store.store("topic A", convergence_round=2, total_rounds=4, final_similarity=0.8)
        store.store("topic B", convergence_round=4, total_rounds=6, final_similarity=0.9)

        stats = store.get_stats()
        assert stats["record_count"] == 2
        assert stats["max_records"] == 1000
        assert stats["avg_convergence_round"] == pytest.approx(3.0)  # (2+4)/2
        assert stats["avg_total_rounds"] == pytest.approx(5.0)  # (4+6)/2
        assert stats["avg_final_similarity"] == pytest.approx(0.85)  # (0.8+0.9)/2

    def test_single_record(self, store):
        store.store("single", convergence_round=5, total_rounds=10, final_similarity=0.95)
        stats = store.get_stats()
        assert stats["record_count"] == 1
        assert stats["avg_convergence_round"] == pytest.approx(5.0)
        assert stats["avg_total_rounds"] == pytest.approx(10.0)
        assert stats["avg_final_similarity"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# clear tests
# ---------------------------------------------------------------------------


class TestClear:
    """Test the clear() method."""

    def test_removes_all_records_and_topic_index(self, store):
        store.store("topic A", convergence_round=1, total_rounds=2, final_similarity=0.5)
        store.store("topic B", convergence_round=2, total_rounds=3, final_similarity=0.6)

        assert len(store._records) == 2
        assert len(store._topic_index) == 2

        store.clear()

        assert len(store._records) == 0
        assert len(store._topic_index) == 0

    def test_clear_then_get_stats_shows_empty(self, store):
        store.store("topic", convergence_round=1, total_rounds=2, final_similarity=0.5)
        store.clear()
        stats = store.get_stats()
        assert stats["record_count"] == 0

    def test_clear_then_find_similar_returns_empty(self, store):
        store.store("searchable topic", convergence_round=1, total_rounds=2, final_similarity=0.5)
        store.clear()
        assert store.find_similar("searchable topic") == []


# ---------------------------------------------------------------------------
# Singleton / module-level function tests
# ---------------------------------------------------------------------------


class TestSingleton:
    """Test module-level singleton management functions."""

    def test_get_returns_none_initially(self):
        from aragora.debate.convergence.history import get_convergence_history_store

        assert get_convergence_history_store() is None

    def test_init_creates_and_returns(self):
        from aragora.debate.convergence.history import (
            get_convergence_history_store,
            init_convergence_history_store,
        )

        s = init_convergence_history_store()
        assert s is not None
        assert get_convergence_history_store() is s

    def test_init_returns_existing_on_second_call(self):
        from aragora.debate.convergence.history import init_convergence_history_store

        s1 = init_convergence_history_store(max_records=100)
        s2 = init_convergence_history_store(max_records=200)
        assert s1 is s2
        assert s1._max_records == 100  # First call's config wins

    def test_set_convergence_history_store_sets(self):
        from aragora.debate.convergence.history import (
            ConvergenceHistoryStore,
            get_convergence_history_store,
            set_convergence_history_store,
        )

        s = ConvergenceHistoryStore(max_records=42)
        set_convergence_history_store(s)
        assert get_convergence_history_store() is s

    def test_set_convergence_history_store_clears(self):
        from aragora.debate.convergence.history import (
            get_convergence_history_store,
            init_convergence_history_store,
            set_convergence_history_store,
        )

        init_convergence_history_store()
        assert get_convergence_history_store() is not None

        set_convergence_history_store(None)
        assert get_convergence_history_store() is None

    def test_init_with_continuum_memory(self):
        from aragora.debate.convergence.history import init_convergence_history_store

        mock_mem = MagicMock()
        s = init_convergence_history_store(continuum_memory=mock_mem)
        assert s._continuum_memory is mock_mem


# ---------------------------------------------------------------------------
# Edge case / thread-safety tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and thread safety."""

    def test_topic_snippet_truncated_to_200_chars(self, store):
        """Topic index stores at most 200 characters (lowercased)."""
        long_topic = "x" * 500
        topic_hash = store.store(
            long_topic, convergence_round=1, total_rounds=2, final_similarity=0.5
        )
        assert len(store._topic_index[topic_hash]) == 200

    def test_topic_index_is_lowercased(self, store):
        topic = "UPPERCASE TOPIC"
        topic_hash = store.store(topic, convergence_round=1, total_rounds=2, final_similarity=0.5)
        assert store._topic_index[topic_hash] == "uppercase topic"

    def test_overwrite_same_topic(self, store):
        """Storing the same topic twice overwrites the record."""
        topic = "same topic"
        store.store(topic, convergence_round=1, total_rounds=3, final_similarity=0.5)
        topic_hash = store.store(topic, convergence_round=2, total_rounds=5, final_similarity=0.9)

        record = store.get_record(topic_hash)
        assert record["convergence_round"] == 2
        assert record["total_rounds"] == 5
        assert record["final_similarity"] == 0.9

    def test_record_has_timestamp(self, store):
        before = time.time()
        topic_hash = store.store(
            "timed topic", convergence_round=1, total_rounds=2, final_similarity=0.5
        )
        after = time.time()

        record = store.get_record(topic_hash)
        assert before <= record["timestamp"] <= after
