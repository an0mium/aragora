"""Tests for aragora.debate.context_gatherer.memory — MemoryMixin."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Concrete implementation for testing the mixin
# ---------------------------------------------------------------------------


class ConcreteMemoryMixin:
    """Minimal concrete class that uses MemoryMixin."""

    def __init__(self):
        self._continuum_context_cache: dict[str, str] = {}

    def _get_task_hash(self, task: str) -> str:
        import hashlib

        return hashlib.md5(task.encode()).hexdigest()[:16]

    def _enforce_cache_limit(self, cache: dict, max_size: int) -> None:
        while len(cache) >= max_size:
            oldest = next(iter(cache))
            del cache[oldest]


# Apply the mixin's method
from aragora.debate.context_gatherer.memory import MemoryMixin

# Attach the mixin method to our concrete class
ConcreteMemoryMixin.get_continuum_context = MemoryMixin.get_continuum_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(
    id: str = "mem-1",
    content: str = "test memory",
    tier_value: str = "medium",
    consolidation_score: float = 0.6,
) -> MagicMock:
    mem = MagicMock()
    mem.id = id
    mem.content = content
    mem.tier = MagicMock()
    mem.tier.value = tier_value
    mem.consolidation_score = consolidation_score
    return mem


def _make_cm(memories=None, glacial_insights=None):
    """Create a mock ContinuumMemory."""
    cm = MagicMock()
    cm.retrieve.return_value = memories or []
    if glacial_insights is not None:
        cm.get_glacial_insights.return_value = glacial_insights
    return cm


# ---------------------------------------------------------------------------
# get_continuum_context — early returns
# ---------------------------------------------------------------------------


class TestEarlyReturns:
    def test_cache_hit(self):
        obj = ConcreteMemoryMixin()
        task = "test task"
        task_hash = obj._get_task_hash(task)
        obj._continuum_context_cache[task_hash] = "cached context"
        result = obj.get_continuum_context(MagicMock(), "domain", task)
        assert result == ("cached context", [], {})

    def test_no_continuum_memory(self):
        obj = ConcreteMemoryMixin()
        result = obj.get_continuum_context(None, "domain", "task")
        assert result == ("", [], {})

    def test_falsy_continuum_memory(self):
        obj = ConcreteMemoryMixin()
        result = obj.get_continuum_context(False, "domain", "task")
        assert result == ("", [], {})


# ---------------------------------------------------------------------------
# get_continuum_context — memory retrieval
# ---------------------------------------------------------------------------


class TestMemoryRetrieval:
    def test_calls_retrieve_with_query(self):
        obj = ConcreteMemoryMixin()
        cm = _make_cm()
        obj.get_continuum_context(cm, "programming", "fix the bug")
        cm.retrieve.assert_called_once()
        call_kwargs = cm.retrieve.call_args[1]
        assert "programming" in call_kwargs["query"]
        assert "fix the bug" in call_kwargs["query"]
        assert call_kwargs["limit"] == 5
        assert call_kwargs["min_importance"] == 0.3
        assert call_kwargs["include_glacial"] is False

    def test_truncates_task_in_query(self):
        obj = ConcreteMemoryMixin()
        cm = _make_cm()
        long_task = "x" * 500
        obj.get_continuum_context(cm, "d", long_task)
        query = cm.retrieve.call_args[1]["query"]
        # Task should be truncated to 200 chars
        assert len(query) <= 210  # domain + ": " + 200 chars

    def test_empty_memories_returns_empty(self):
        obj = ConcreteMemoryMixin()
        cm = _make_cm(memories=[], glacial_insights=[])
        result = obj.get_continuum_context(cm, "d", "task")
        assert result == ("", [], {})


# ---------------------------------------------------------------------------
# get_continuum_context — glacial insights
# ---------------------------------------------------------------------------


class TestGlacialInsights:
    def test_glacial_insights_retrieved(self):
        obj = ConcreteMemoryMixin()
        glacial = _make_memory("g1", "long-term pattern", "glacial", 0.9)
        cm = _make_cm(memories=[], glacial_insights=[glacial])
        ctx, ids, tiers = obj.get_continuum_context(cm, "d", "task")
        cm.get_glacial_insights.assert_called_once()
        assert "g1" in ids
        assert "glacial" in ctx

    def test_glacial_insights_disabled(self):
        obj = ConcreteMemoryMixin()
        cm = _make_cm(memories=[_make_memory()])
        ctx, ids, tiers = obj.get_continuum_context(
            cm, "d", "task", include_glacial_insights=False
        )
        assert not hasattr(cm.get_glacial_insights, "called") or not cm.get_glacial_insights.called

    def test_glacial_type_error_fallback(self):
        obj = ConcreteMemoryMixin()
        cm = MagicMock()
        cm.retrieve.return_value = [_make_memory()]
        cm.get_glacial_insights.side_effect = TypeError("no kwargs")
        # Should fall back to positional args
        ctx, ids, tiers = obj.get_continuum_context(cm, "d", "task")
        assert cm.get_glacial_insights.call_count == 2  # First call fails, second with positional

    def test_no_get_glacial_insights_method(self):
        obj = ConcreteMemoryMixin()
        cm = MagicMock(spec=["retrieve"])  # No get_glacial_insights
        cm.retrieve.return_value = [_make_memory()]
        with patch("aragora.memory.tier_manager.MemoryTier") as MockTier:
            ctx, ids, tiers = obj.get_continuum_context(cm, "d", "task")
        # Should have called retrieve again with tiers=[MemoryTier.GLACIAL]
        assert cm.retrieve.call_count == 2


# ---------------------------------------------------------------------------
# get_continuum_context — formatting
# ---------------------------------------------------------------------------


class TestContextFormatting:
    def test_recent_memory_formatted_with_tier(self):
        obj = ConcreteMemoryMixin()
        mem = _make_memory("m1", "important finding", "medium", 0.6)
        cm = _make_cm(memories=[mem], glacial_insights=[])
        ctx, ids, tiers = obj.get_continuum_context(cm, "d", "task")
        assert "[medium|medium]" in ctx
        assert "important finding" in ctx

    def test_high_consolidation_labeled_high(self):
        obj = ConcreteMemoryMixin()
        mem = _make_memory("m1", "solid insight", "fast", 0.8)
        cm = _make_cm(memories=[mem], glacial_insights=[])
        ctx, ids, tiers = obj.get_continuum_context(cm, "d", "task")
        assert "[fast|high]" in ctx

    def test_low_consolidation_labeled_low(self):
        obj = ConcreteMemoryMixin()
        mem = _make_memory("m1", "weak signal", "slow", 0.2)
        cm = _make_cm(memories=[mem], glacial_insights=[])
        ctx, ids, tiers = obj.get_continuum_context(cm, "d", "task")
        assert "[slow|low]" in ctx

    def test_glacial_memory_formatted_separately(self):
        obj = ConcreteMemoryMixin()
        glacial = _make_memory("g1", "cross-session pattern", "glacial", 0.9)
        cm = _make_cm(memories=[], glacial_insights=[glacial])
        ctx, ids, tiers = obj.get_continuum_context(cm, "d", "task")
        assert "[glacial|foundational]" in ctx
        assert "Long-term patterns" in ctx

    def test_header_always_present(self):
        obj = ConcreteMemoryMixin()
        mem = _make_memory()
        cm = _make_cm(memories=[mem], glacial_insights=[])
        ctx, ids, tiers = obj.get_continuum_context(cm, "d", "task")
        assert "Previous learnings" in ctx

    def test_content_truncated_to_200(self):
        obj = ConcreteMemoryMixin()
        long_content = "x" * 500
        mem = _make_memory("m1", long_content, "medium")
        cm = _make_cm(memories=[mem], glacial_insights=[])
        ctx, ids, tiers = obj.get_continuum_context(cm, "d", "task")
        # Each memory line content truncated to 200
        lines = [l for l in ctx.split("\n") if l.startswith("- [")]
        for line in lines:
            # The bracketed prefix is ~20 chars, content should be ≤200
            assert len(line) < 250

    def test_max_three_recent_memories(self):
        obj = ConcreteMemoryMixin()
        mems = [_make_memory(f"m{i}", f"content_{i}", "medium") for i in range(10)]
        cm = _make_cm(memories=mems, glacial_insights=[])
        ctx, ids, tiers = obj.get_continuum_context(cm, "d", "task")
        recent_lines = [l for l in ctx.split("\n") if "[medium|" in l]
        assert len(recent_lines) <= 3

    def test_max_two_glacial_memories(self):
        obj = ConcreteMemoryMixin()
        glacials = [_make_memory(f"g{i}", f"glacial_{i}", "glacial") for i in range(5)]
        cm = _make_cm(memories=[], glacial_insights=glacials)
        ctx, ids, tiers = obj.get_continuum_context(cm, "d", "task")
        glacial_lines = [l for l in ctx.split("\n") if "[glacial|" in l]
        assert len(glacial_lines) <= 2


# ---------------------------------------------------------------------------
# get_continuum_context — caching
# ---------------------------------------------------------------------------


class TestCaching:
    def test_result_cached(self):
        obj = ConcreteMemoryMixin()
        mem = _make_memory()
        cm = _make_cm(memories=[mem], glacial_insights=[])
        ctx1, _, _ = obj.get_continuum_context(cm, "d", "task")
        assert len(obj._continuum_context_cache) == 1
        # Second call should hit cache
        ctx2, _, _ = obj.get_continuum_context(cm, "d", "task")
        assert ctx2 == ctx1
        # retrieve should only be called once (from first call)
        assert cm.retrieve.call_count == 1

    def test_different_tasks_cached_separately(self):
        obj = ConcreteMemoryMixin()
        cm = _make_cm(memories=[_make_memory()], glacial_insights=[])
        obj.get_continuum_context(cm, "d", "task_a")
        obj.get_continuum_context(cm, "d", "task_b")
        assert len(obj._continuum_context_cache) == 2


# ---------------------------------------------------------------------------
# get_continuum_context — ID and tier tracking
# ---------------------------------------------------------------------------


class TestIdTracking:
    def test_returns_memory_ids(self):
        obj = ConcreteMemoryMixin()
        mems = [_make_memory("m1"), _make_memory("m2")]
        cm = _make_cm(memories=mems, glacial_insights=[])
        _, ids, _ = obj.get_continuum_context(cm, "d", "task")
        assert "m1" in ids
        assert "m2" in ids

    def test_returns_tier_mapping(self):
        obj = ConcreteMemoryMixin()
        mem = _make_memory("m1", tier_value="fast")
        cm = _make_cm(memories=[mem], glacial_insights=[])
        _, _, tiers = obj.get_continuum_context(cm, "d", "task")
        assert "m1" in tiers

    def test_memories_without_id_excluded(self):
        obj = ConcreteMemoryMixin()
        mem = MagicMock()
        mem.id = None
        mem.content = "test"
        mem.tier = MagicMock()
        mem.tier.value = "medium"
        mem.consolidation_score = 0.5
        cm = _make_cm(memories=[mem], glacial_insights=[])
        _, ids, tiers = obj.get_continuum_context(cm, "d", "task")
        assert len(ids) == 0


# ---------------------------------------------------------------------------
# get_continuum_context — RBAC access control
# ---------------------------------------------------------------------------


class TestAccessControl:
    def test_denied_access_returns_empty(self):
        obj = ConcreteMemoryMixin()
        cm = _make_cm(memories=[_make_memory()])
        auth_ctx = MagicMock()
        with patch("aragora.memory.access.has_memory_read_access", return_value=False) as mock_access, \
             patch("aragora.memory.access.emit_denial_telemetry") as mock_deny, \
             patch("aragora.memory.access.filter_entries"):
            result = obj.get_continuum_context(cm, "d", "task", auth_context=auth_ctx)
        assert result == ("", [], {})
        mock_deny.assert_called_once()

    def test_no_auth_context_skips_check(self):
        obj = ConcreteMemoryMixin()
        mem = _make_memory()
        cm = _make_cm(memories=[mem], glacial_insights=[])
        ctx, ids, tiers = obj.get_continuum_context(cm, "d", "task", auth_context=None)
        assert ctx != ""  # Should still return context


# ---------------------------------------------------------------------------
# get_continuum_context — error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_attribute_error_returns_empty(self):
        obj = ConcreteMemoryMixin()
        cm = MagicMock()
        cm.retrieve.side_effect = AttributeError("no method")
        result = obj.get_continuum_context(cm, "d", "task")
        assert result == ("", [], {})

    def test_type_error_returns_empty(self):
        obj = ConcreteMemoryMixin()
        cm = MagicMock()
        cm.retrieve.side_effect = TypeError("bad args")
        result = obj.get_continuum_context(cm, "d", "task")
        assert result == ("", [], {})

    def test_runtime_error_returns_empty(self):
        obj = ConcreteMemoryMixin()
        cm = MagicMock()
        cm.retrieve.side_effect = RuntimeError("internal")
        result = obj.get_continuum_context(cm, "d", "task")
        assert result == ("", [], {})

    def test_key_error_returns_empty(self):
        obj = ConcreteMemoryMixin()
        cm = MagicMock()
        cm.retrieve.side_effect = KeyError("missing")
        result = obj.get_continuum_context(cm, "d", "task")
        assert result == ("", [], {})

    def test_os_error_returns_empty(self):
        obj = ConcreteMemoryMixin()
        cm = MagicMock()
        cm.retrieve.side_effect = OSError("disk")
        result = obj.get_continuum_context(cm, "d", "task")
        assert result == ("", [], {})
