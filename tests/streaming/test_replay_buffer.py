"""Tests for aragora.streaming.replay_buffer."""

from __future__ import annotations

import time

import pytest

from aragora.streaming.replay_buffer import (
    BufferedEvent,
    EventReplayBuffer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_buffer(
    max_events: int = 100,
    window_seconds: float = 300.0,
    max_bytes: int = 50 * 1024 * 1024,
) -> EventReplayBuffer:
    return EventReplayBuffer(
        max_events=max_events,
        window_seconds=window_seconds,
        max_bytes=max_bytes,
    )


def _small_data(n: int = 1) -> str:
    """Return a small JSON string for testing."""
    return f'{{"type":"test","seq":{n}}}'


# ---------------------------------------------------------------------------
# Basic append and replay
# ---------------------------------------------------------------------------


class TestReplayBufferBasic:
    """Test basic append, replay, and metadata operations."""

    def test_append_and_replay(self):
        buf = _make_buffer()
        buf.append(seq=1, data=_small_data(1), debate_id="d1")
        buf.append(seq=2, data=_small_data(2), debate_id="d1")
        buf.append(seq=3, data=_small_data(3), debate_id="d1")

        missed = buf.replay_from_seq("d1", seq_number=1)
        assert len(missed) == 2
        assert '"seq":2' in missed[0]
        assert '"seq":3' in missed[1]

    def test_replay_from_seq_zero_returns_all(self):
        buf = _make_buffer()
        for i in range(1, 6):
            buf.append(seq=i, data=_small_data(i), debate_id="d1")
        missed = buf.replay_from_seq("d1", seq_number=0)
        assert len(missed) == 5

    def test_replay_from_latest_returns_empty(self):
        buf = _make_buffer()
        for i in range(1, 4):
            buf.append(seq=i, data=_small_data(i), debate_id="d1")
        missed = buf.replay_from_seq("d1", seq_number=3)
        assert len(missed) == 0

    def test_replay_nonexistent_debate_returns_empty(self):
        buf = _make_buffer()
        missed = buf.replay_from_seq("nope", seq_number=0)
        assert missed == []

    def test_get_latest_seq(self):
        buf = _make_buffer()
        buf.append(seq=10, data="a", debate_id="d1")
        buf.append(seq=20, data="b", debate_id="d1")
        assert buf.get_latest_seq("d1") == 20

    def test_get_oldest_seq(self):
        buf = _make_buffer()
        buf.append(seq=5, data="a", debate_id="d1")
        buf.append(seq=10, data="b", debate_id="d1")
        assert buf.get_oldest_seq("d1") == 5

    def test_latest_seq_empty_returns_zero(self):
        buf = _make_buffer()
        assert buf.get_latest_seq("d1") == 0

    def test_oldest_seq_empty_returns_zero(self):
        buf = _make_buffer()
        assert buf.get_oldest_seq("d1") == 0

    def test_count_total(self):
        buf = _make_buffer()
        buf.append(seq=1, data="a", debate_id="d1")
        buf.append(seq=2, data="b", debate_id="d2")
        assert buf.count() == 2

    def test_count_per_debate(self):
        buf = _make_buffer()
        buf.append(seq=1, data="a", debate_id="d1")
        buf.append(seq=2, data="b", debate_id="d1")
        buf.append(seq=3, data="c", debate_id="d2")
        assert buf.count("d1") == 2
        assert buf.count("d2") == 1


# ---------------------------------------------------------------------------
# Debate filtering
# ---------------------------------------------------------------------------


class TestReplayBufferFiltering:
    """Test that events are properly filtered by debate."""

    def test_replay_only_returns_matching_debate(self):
        buf = _make_buffer()
        buf.append(seq=1, data="d1-event", debate_id="d1")
        buf.append(seq=2, data="d2-event", debate_id="d2")
        buf.append(seq=3, data="d1-event2", debate_id="d1")

        d1_events = buf.replay_from_seq("d1", seq_number=0)
        assert len(d1_events) == 2
        assert all("d1" in e for e in d1_events)

        d2_events = buf.replay_from_seq("d2", seq_number=0)
        assert len(d2_events) == 1

    def test_remove_debate(self):
        buf = _make_buffer()
        buf.append(seq=1, data="a", debate_id="d1")
        buf.append(seq=2, data="b", debate_id="d1")
        buf.append(seq=3, data="c", debate_id="d2")

        removed = buf.remove_debate("d1")
        assert removed == 2
        assert buf.count("d1") == 0
        assert buf.count("d2") == 1

    def test_remove_nonexistent_debate(self):
        buf = _make_buffer()
        removed = buf.remove_debate("nope")
        assert removed == 0


# ---------------------------------------------------------------------------
# Max events (ring buffer eviction)
# ---------------------------------------------------------------------------


class TestReplayBufferMaxEvents:
    """Test that max_events limit is enforced."""

    def test_evicts_oldest_when_full(self):
        buf = _make_buffer(max_events=5)
        for i in range(1, 8):
            buf.append(seq=i, data=_small_data(i), debate_id="d1")
        assert buf.count() == 5
        assert buf.get_oldest_seq("d1") == 3
        assert buf.get_latest_seq("d1") == 7

    def test_eviction_updates_byte_count(self):
        buf = _make_buffer(max_events=3)
        for i in range(1, 6):
            buf.append(seq=i, data=_small_data(i), debate_id="d1")
        assert buf.count() == 3
        # Byte count should be roughly 3 events' worth
        assert buf.current_bytes() > 0


# ---------------------------------------------------------------------------
# Time-based pruning
# ---------------------------------------------------------------------------


class TestReplayBufferPruning:
    """Test time-based expiration pruning."""

    def test_prune_expired_removes_old_events(self):
        buf = _make_buffer(window_seconds=0.05)
        buf.append(seq=1, data="old", debate_id="d1")
        time.sleep(0.06)
        buf.append(seq=2, data="new", debate_id="d1")

        pruned = buf.prune_expired()
        assert pruned == 1
        assert buf.count() == 1
        assert buf.get_oldest_seq("d1") == 2

    def test_prune_no_expired_returns_zero(self):
        buf = _make_buffer(window_seconds=300.0)
        buf.append(seq=1, data="fresh", debate_id="d1")
        pruned = buf.prune_expired()
        assert pruned == 0

    def test_prune_empty_buffer(self):
        buf = _make_buffer()
        pruned = buf.prune_expired()
        assert pruned == 0


# ---------------------------------------------------------------------------
# Memory bounds
# ---------------------------------------------------------------------------


class TestReplayBufferMemoryBounds:
    """Test memory-bounded behavior."""

    def test_enforces_max_bytes(self):
        # Use a very small max_bytes to force eviction
        buf = _make_buffer(max_events=10000, max_bytes=200)
        for i in range(100):
            buf.append(seq=i, data="x" * 50, debate_id="d1")
        assert buf.current_bytes() <= 200 + 100  # Allow some overhead

    def test_metrics_track_evictions(self):
        buf = _make_buffer(max_events=3)
        for i in range(10):
            buf.append(seq=i, data=_small_data(i), debate_id="d1")
        metrics = buf.get_metrics()
        assert metrics["total_appended"] == 10
        assert metrics["total_evicted"] >= 7


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestReplayBufferEdgeCases:
    """Test edge cases and validation."""

    def test_empty_data_rejected(self):
        buf = _make_buffer()
        result = buf.append(seq=1, data="", debate_id="d1")
        assert result is False
        assert buf.count() == 0

    def test_empty_debate_id_rejected(self):
        buf = _make_buffer()
        result = buf.append(seq=1, data="valid", debate_id="")
        assert result is False
        assert buf.count() == 0

    def test_clear_removes_all(self):
        buf = _make_buffer()
        for i in range(10):
            buf.append(seq=i, data=_small_data(i), debate_id="d1")
        buf.clear()
        assert buf.count() == 0
        assert buf.current_bytes() == 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestReplayBufferMetrics:
    """Test buffer metrics reporting."""

    def test_get_metrics_returns_complete_data(self):
        buf = _make_buffer(max_events=50, window_seconds=100.0, max_bytes=1024)
        buf.append(seq=1, data="test", debate_id="d1")
        metrics = buf.get_metrics()
        assert "buffered_events" in metrics
        assert "current_bytes" in metrics
        assert "max_events" in metrics
        assert "max_bytes" in metrics
        assert "window_seconds" in metrics
        assert "total_appended" in metrics
        assert "total_evicted" in metrics
        assert "total_pruned" in metrics
        assert metrics["buffered_events"] == 1
        assert metrics["total_appended"] == 1
