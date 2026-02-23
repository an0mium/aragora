"""Tests for aragora.debate.strategy module.

Covers StrategyRecommendation dataclass, DebateStrategy initialization,
estimate_rounds (sync), estimate_rounds_async, and get_relevant_context,
including all tier-matching branches, error handling, and edge cases.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers: lightweight stand-ins so we never import real heavy modules
# ---------------------------------------------------------------------------


def _make_memory_entry(entry_id: str = "mem-1", success_rate: float = 0.95):
    """Create a lightweight mock ContinuumMemoryEntry with required attrs."""
    entry = SimpleNamespace(id=entry_id, success_rate=success_rate)
    return entry


def _make_memory_tier_module():
    """Return a fake MemoryTier enum namespace with GLACIAL/SLOW/MEDIUM."""
    tier = SimpleNamespace(
        GLACIAL="glacial",
        SLOW="slow",
        MEDIUM="medium",
        FAST="fast",
    )
    return SimpleNamespace(MemoryTier=tier)


# We patch the import inside strategy.py so it never touches the real module
_TIER_PATCH = "aragora.memory.tier_manager"


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from aragora.debate.strategy import DebateStrategy, StrategyRecommendation


# ===================================================================
# StrategyRecommendation dataclass
# ===================================================================


class TestStrategyRecommendation:
    """Tests for the StrategyRecommendation dataclass."""

    def test_fields_present(self):
        rec = StrategyRecommendation(
            estimated_rounds=3,
            confidence=0.85,
            reasoning="test reasoning",
            relevant_memories=["m1", "m2"],
        )
        assert rec.estimated_rounds == 3
        assert rec.confidence == 0.85
        assert rec.reasoning == "test reasoning"
        assert rec.relevant_memories == ["m1", "m2"]

    def test_equality(self):
        a = StrategyRecommendation(2, 0.9, "r", ["x"])
        b = StrategyRecommendation(2, 0.9, "r", ["x"])
        assert a == b

    def test_empty_memories(self):
        rec = StrategyRecommendation(5, 0.0, "no memory", [])
        assert rec.relevant_memories == []


# ===================================================================
# DebateStrategy.__init__
# ===================================================================


class TestDebateStrategyInit:
    """Tests for DebateStrategy constructor defaults and custom values."""

    def test_default_values(self):
        strategy = DebateStrategy()
        assert strategy.continuum_memory is None
        assert strategy.quick_validation_rounds == 2
        assert strategy.standard_rounds == 3
        assert strategy.exploration_rounds == 5
        assert strategy.high_confidence_threshold == 0.9
        assert strategy.medium_confidence_threshold == 0.7

    def test_custom_values(self):
        mem = MagicMock()
        strategy = DebateStrategy(
            continuum_memory=mem,
            quick_validation_rounds=1,
            standard_rounds=4,
            exploration_rounds=7,
            high_confidence_threshold=0.85,
            medium_confidence_threshold=0.6,
        )
        assert strategy.continuum_memory is mem
        assert strategy.quick_validation_rounds == 1
        assert strategy.standard_rounds == 4
        assert strategy.exploration_rounds == 7
        assert strategy.high_confidence_threshold == 0.85
        assert strategy.medium_confidence_threshold == 0.6

    def test_class_constants(self):
        assert DebateStrategy.QUICK_VALIDATION_ROUNDS == 2
        assert DebateStrategy.STANDARD_DEBATE_ROUNDS == 3
        assert DebateStrategy.EXPLORATION_ROUNDS == 5
        assert DebateStrategy.HIGH_CONFIDENCE_THRESHOLD == 0.9
        assert DebateStrategy.MEDIUM_CONFIDENCE_THRESHOLD == 0.7


# ===================================================================
# estimate_rounds  (sync)
# ===================================================================


class TestEstimateRoundsNoMemory:
    """estimate_rounds when no ContinuumMemory is attached."""

    def test_no_memory_returns_exploration(self):
        strategy = DebateStrategy(continuum_memory=None)
        rec = strategy.estimate_rounds("Design a widget")
        assert rec.estimated_rounds == strategy.exploration_rounds
        assert rec.confidence == 0.0
        assert "No memory" in rec.reasoning
        assert rec.relevant_memories == []

    def test_no_memory_custom_default_rounds(self):
        strategy = DebateStrategy(continuum_memory=None)
        rec = strategy.estimate_rounds("task", default_rounds=10)
        assert rec.estimated_rounds == 10

    def test_no_memory_default_rounds_none(self):
        strategy = DebateStrategy(continuum_memory=None, exploration_rounds=6)
        rec = strategy.estimate_rounds("task", default_rounds=None)
        assert rec.estimated_rounds == 6


class TestEstimateRoundsGlacialHighConfidence:
    """estimate_rounds when glacial tier returns high-confidence entries."""

    def _make_strategy_with_glacial(self, entries, **kwargs):
        """Build strategy whose memory returns *entries* for glacial tier."""
        fake_tier = _make_memory_tier_module()
        mem = MagicMock()

        def _retrieve(query, tiers, limit, min_importance):
            tier_val = tiers[0]
            if tier_val == fake_tier.MemoryTier.GLACIAL:
                return entries
            return []

        mem.retrieve = MagicMock(side_effect=_retrieve)
        strategy = DebateStrategy(continuum_memory=mem, **kwargs)
        return strategy, fake_tier

    def test_high_confidence_returns_quick_rounds(self):
        entries = [_make_memory_entry("g1", 0.95)]
        strategy, fake_tier = self._make_strategy_with_glacial(entries)
        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            rec = strategy.estimate_rounds("known topic")
        assert rec.estimated_rounds == 2
        assert rec.confidence == 0.95
        assert "Quick validation" in rec.reasoning
        assert rec.relevant_memories == ["g1"]

    def test_respect_minimum_true_enforces_at_least_2(self):
        entries = [_make_memory_entry("g1", 0.99)]
        strategy, fake_tier = self._make_strategy_with_glacial(entries, quick_validation_rounds=1)
        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            rec = strategy.estimate_rounds("task", respect_minimum=True)
        assert rec.estimated_rounds == 2  # max(2, 1) = 2

    def test_respect_minimum_false_allows_below_2(self):
        entries = [_make_memory_entry("g1", 0.99)]
        strategy, fake_tier = self._make_strategy_with_glacial(entries, quick_validation_rounds=1)
        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            rec = strategy.estimate_rounds("task", respect_minimum=False)
        assert rec.estimated_rounds == 1  # no minimum enforced

    def test_multiple_glacial_entries_uses_max(self):
        entries = [
            _make_memory_entry("g1", 0.7),
            _make_memory_entry("g2", 0.92),
            _make_memory_entry("g3", 0.88),
        ]
        strategy, fake_tier = self._make_strategy_with_glacial(entries)
        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            rec = strategy.estimate_rounds("task")
        assert rec.estimated_rounds == 2
        assert rec.confidence == 0.92
        assert set(rec.relevant_memories) == {"g1", "g2", "g3"}


class TestEstimateRoundsSlowTier:
    """estimate_rounds when glacial is low but slow tier has medium confidence."""

    def test_slow_medium_confidence(self):
        fake_tier = _make_memory_tier_module()
        mem = MagicMock()

        def _retrieve(query, tiers, limit, min_importance):
            tier_val = tiers[0]
            if tier_val == fake_tier.MemoryTier.GLACIAL:
                # Return entries but below high threshold
                return [_make_memory_entry("g1", 0.6)]
            elif tier_val == fake_tier.MemoryTier.SLOW:
                return [_make_memory_entry("s1", 0.8)]
            return []

        mem.retrieve = MagicMock(side_effect=_retrieve)
        strategy = DebateStrategy(continuum_memory=mem)

        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            rec = strategy.estimate_rounds("partially known task")

        assert rec.estimated_rounds == 3  # standard rounds
        assert rec.confidence == 0.8
        assert "slow-tier" in rec.reasoning
        assert rec.relevant_memories == ["s1"]

    def test_slow_below_medium_threshold_falls_through(self):
        """Slow tier entries below medium threshold should fall through to medium tier check."""
        fake_tier = _make_memory_tier_module()
        mem = MagicMock()

        def _retrieve(query, tiers, limit, min_importance):
            tier_val = tiers[0]
            if tier_val == fake_tier.MemoryTier.GLACIAL:
                return []
            elif tier_val == fake_tier.MemoryTier.SLOW:
                return [_make_memory_entry("s1", 0.5)]  # Below 0.7 threshold
            elif tier_val == fake_tier.MemoryTier.MEDIUM:
                return [_make_memory_entry("m1", 0.9)]
            return []

        mem.retrieve = MagicMock(side_effect=_retrieve)
        strategy = DebateStrategy(continuum_memory=mem)

        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            rec = strategy.estimate_rounds("task")

        assert rec.estimated_rounds == 3
        # Medium tier confidence is discounted by 0.8
        assert rec.confidence == pytest.approx(0.72)  # 0.9 * 0.8
        assert "medium-tier" in rec.reasoning.lower()


class TestEstimateRoundsMediumTier:
    """estimate_rounds when only medium tier has matches."""

    def test_medium_tier_discount(self):
        fake_tier = _make_memory_tier_module()
        mem = MagicMock()

        def _retrieve(query, tiers, limit, min_importance):
            tier_val = tiers[0]
            if tier_val == fake_tier.MemoryTier.MEDIUM:
                return [_make_memory_entry("m1", 1.0)]
            return []

        mem.retrieve = MagicMock(side_effect=_retrieve)
        strategy = DebateStrategy(continuum_memory=mem)

        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            rec = strategy.estimate_rounds("recent task")

        assert rec.estimated_rounds == 3  # standard
        assert rec.confidence == pytest.approx(0.8)  # 1.0 * 0.8
        assert "medium-tier" in rec.reasoning.lower()
        assert rec.relevant_memories == ["m1"]


class TestEstimateRoundsNoMatches:
    """estimate_rounds when all tiers return empty."""

    def test_no_matches_returns_exploration(self):
        fake_tier = _make_memory_tier_module()
        mem = MagicMock()
        mem.retrieve = MagicMock(return_value=[])
        strategy = DebateStrategy(continuum_memory=mem)

        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            rec = strategy.estimate_rounds("totally novel task")

        assert rec.estimated_rounds == 5  # exploration
        assert rec.confidence == 0.0
        assert "No relevant prior knowledge" in rec.reasoning
        assert rec.relevant_memories == []


class TestEstimateRoundsError:
    """estimate_rounds error handling."""

    @pytest.mark.parametrize(
        "exc_class",
        [RuntimeError, ValueError, TypeError, AttributeError, KeyError, OSError],
    )
    def test_caught_exceptions_return_default(self, exc_class):
        fake_tier = _make_memory_tier_module()
        mem = MagicMock()
        mem.retrieve = MagicMock(side_effect=exc_class("boom"))
        strategy = DebateStrategy(continuum_memory=mem)

        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            rec = strategy.estimate_rounds("task")

        assert rec.estimated_rounds == 5  # exploration (default)
        assert rec.confidence == 0.0
        assert "boom" in rec.reasoning
        assert rec.relevant_memories == []

    def test_error_uses_custom_default_rounds(self):
        fake_tier = _make_memory_tier_module()
        mem = MagicMock()
        mem.retrieve = MagicMock(side_effect=RuntimeError("db down"))
        strategy = DebateStrategy(continuum_memory=mem)

        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            rec = strategy.estimate_rounds("task", default_rounds=7)

        assert rec.estimated_rounds == 7

    def test_glacial_ok_slow_error(self):
        """Error raised during slow tier query after glacial returned empty."""
        fake_tier = _make_memory_tier_module()
        mem = MagicMock()
        call_count = 0

        def _retrieve(query, tiers, limit, min_importance):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return []  # glacial: empty
            raise ValueError("slow tier exploded")

        mem.retrieve = MagicMock(side_effect=_retrieve)
        strategy = DebateStrategy(continuum_memory=mem)

        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            rec = strategy.estimate_rounds("task")

        assert rec.estimated_rounds == 5  # default
        assert "slow tier exploded" in rec.reasoning


# ===================================================================
# estimate_rounds_async
# ===================================================================


class TestEstimateRoundsAsync:
    """Async variant of estimate_rounds."""

    @pytest.mark.asyncio
    async def test_no_memory_async(self):
        strategy = DebateStrategy(continuum_memory=None)
        rec = await strategy.estimate_rounds_async("async task")
        assert rec.estimated_rounds == 5
        assert rec.confidence == 0.0

    @pytest.mark.asyncio
    async def test_glacial_high_confidence_async(self):
        fake_tier = _make_memory_tier_module()
        mem = MagicMock()

        async def _retrieve_async(query, tiers, limit, min_importance):
            tier_val = tiers[0]
            if tier_val == fake_tier.MemoryTier.GLACIAL:
                return [_make_memory_entry("ag1", 0.95)]
            return []

        mem.retrieve_async = MagicMock(side_effect=_retrieve_async)
        strategy = DebateStrategy(continuum_memory=mem)

        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            rec = await strategy.estimate_rounds_async("known async")

        assert rec.estimated_rounds == 2
        assert rec.confidence == 0.95
        assert rec.relevant_memories == ["ag1"]

    @pytest.mark.asyncio
    async def test_slow_tier_async(self):
        fake_tier = _make_memory_tier_module()
        mem = MagicMock()

        async def _retrieve_async(query, tiers, limit, min_importance):
            tier_val = tiers[0]
            if tier_val == fake_tier.MemoryTier.GLACIAL:
                return []
            elif tier_val == fake_tier.MemoryTier.SLOW:
                return [_make_memory_entry("as1", 0.75)]
            return []

        mem.retrieve_async = MagicMock(side_effect=_retrieve_async)
        strategy = DebateStrategy(continuum_memory=mem)

        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            rec = await strategy.estimate_rounds_async("partial async")

        assert rec.estimated_rounds == 3
        assert rec.confidence == 0.75

    @pytest.mark.asyncio
    async def test_medium_tier_async(self):
        fake_tier = _make_memory_tier_module()
        mem = MagicMock()

        async def _retrieve_async(query, tiers, limit, min_importance):
            tier_val = tiers[0]
            if tier_val == fake_tier.MemoryTier.MEDIUM:
                return [_make_memory_entry("am1", 0.9)]
            return []

        mem.retrieve_async = MagicMock(side_effect=_retrieve_async)
        strategy = DebateStrategy(continuum_memory=mem)

        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            rec = await strategy.estimate_rounds_async("recent async")

        assert rec.estimated_rounds == 3
        assert rec.confidence == pytest.approx(0.72)

    @pytest.mark.asyncio
    async def test_no_matches_async(self):
        fake_tier = _make_memory_tier_module()
        mem = MagicMock()

        async def _retrieve_async(query, tiers, limit, min_importance):
            return []

        mem.retrieve_async = MagicMock(side_effect=_retrieve_async)
        strategy = DebateStrategy(continuum_memory=mem)

        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            rec = await strategy.estimate_rounds_async("novel async")

        assert rec.estimated_rounds == 5
        assert rec.confidence == 0.0
        assert "No relevant prior knowledge" in rec.reasoning

    @pytest.mark.asyncio
    async def test_error_async(self):
        fake_tier = _make_memory_tier_module()
        mem = MagicMock()

        async def _retrieve_async(query, tiers, limit, min_importance):
            raise RuntimeError("async boom")

        mem.retrieve_async = MagicMock(side_effect=_retrieve_async)
        strategy = DebateStrategy(continuum_memory=mem)

        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            rec = await strategy.estimate_rounds_async("broken async")

        assert rec.estimated_rounds == 5
        assert "async boom" in rec.reasoning

    @pytest.mark.asyncio
    async def test_async_custom_default(self):
        strategy = DebateStrategy(continuum_memory=None)
        rec = await strategy.estimate_rounds_async("task", default_rounds=8)
        assert rec.estimated_rounds == 8

    @pytest.mark.asyncio
    async def test_async_respect_minimum_false(self):
        fake_tier = _make_memory_tier_module()
        mem = MagicMock()

        async def _retrieve_async(query, tiers, limit, min_importance):
            tier_val = tiers[0]
            if tier_val == fake_tier.MemoryTier.GLACIAL:
                return [_make_memory_entry("ag1", 0.99)]
            return []

        mem.retrieve_async = MagicMock(side_effect=_retrieve_async)
        strategy = DebateStrategy(continuum_memory=mem, quick_validation_rounds=1)

        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            rec = await strategy.estimate_rounds_async("task", respect_minimum=False)

        assert rec.estimated_rounds == 1


# ===================================================================
# get_relevant_context
# ===================================================================


class TestGetRelevantContext:
    """Tests for get_relevant_context method."""

    def test_no_memory_returns_empty(self):
        strategy = DebateStrategy(continuum_memory=None)
        result = strategy.get_relevant_context("task")
        assert result == []

    def test_returns_entries(self):
        fake_tier = _make_memory_tier_module()
        entries = [
            _make_memory_entry("c1", 0.9),
            _make_memory_entry("c2", 0.85),
        ]
        mem = MagicMock()
        mem.retrieve = MagicMock(return_value=entries)
        strategy = DebateStrategy(continuum_memory=mem)

        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            result = strategy.get_relevant_context("task", limit=5)

        assert len(result) == 2
        assert result[0].id == "c1"
        assert result[1].id == "c2"
        # Verify called with both GLACIAL and SLOW tiers
        call_kwargs = mem.retrieve.call_args
        assert fake_tier.MemoryTier.GLACIAL in call_kwargs.kwargs["tiers"]
        assert fake_tier.MemoryTier.SLOW in call_kwargs.kwargs["tiers"]
        assert call_kwargs.kwargs["limit"] == 5

    def test_default_limit(self):
        fake_tier = _make_memory_tier_module()
        mem = MagicMock()
        mem.retrieve = MagicMock(return_value=[])
        strategy = DebateStrategy(continuum_memory=mem)

        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            strategy.get_relevant_context("task")

        assert mem.retrieve.call_args.kwargs["limit"] == 3

    def test_error_returns_empty(self):
        fake_tier = _make_memory_tier_module()
        mem = MagicMock()
        mem.retrieve = MagicMock(side_effect=RuntimeError("context boom"))
        strategy = DebateStrategy(continuum_memory=mem)

        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            result = strategy.get_relevant_context("task")

        assert result == []

    @pytest.mark.parametrize(
        "exc_class",
        [ValueError, TypeError, AttributeError, KeyError, OSError],
    )
    def test_various_exceptions_handled(self, exc_class):
        fake_tier = _make_memory_tier_module()
        mem = MagicMock()
        mem.retrieve = MagicMock(side_effect=exc_class("error"))
        strategy = DebateStrategy(continuum_memory=mem)

        with patch.dict(sys.modules, {_TIER_PATCH: fake_tier}):
            result = strategy.get_relevant_context("task")

        assert result == []
