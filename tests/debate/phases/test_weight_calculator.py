"""
Tests for weight calculation module.

Tests cover:
- WeightFactors dataclass and total property
- WeightCalculatorConfig dataclass
- WeightCache: get/put, TTL expiry, ELO invalidation, agent invalidation,
  LRU eviction, stats, thread safety
- get_weight_cache / cleanup_weight_cache global manager
- WeightCalculator with all factors enabled/disabled
- Individual weight factor methods (reputation, reliability, consistency,
  calibration, elo_skill)
- Session-based caching: hit, miss, invalidation
- Self-vote mitigation modes: exclude, downweight, log_only
- Verbosity normalization
- Bounds clamping (min_weight, max_weight)
- Error handling: graceful degradation when dependencies fail
- ELO skill weight formula with specific values
"""

import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.phases.weight_calculator import (
    WeightCache,
    WeightCalculator,
    WeightCalculatorConfig,
    WeightFactors,
    cleanup_weight_cache,
    get_weight_cache,
    _weight_cache_manager,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str
    role: str = "debater"


@dataclass
class MockRating:
    """Mock ELO rating for testing."""

    elo: float = 1500.0
    calibration_score: float = 0.5
    domain_elos: dict | None = None


@dataclass
class MockConsistency:
    """Mock consistency result for testing."""

    consistency_score: float = 0.8


@dataclass
class MockVote:
    """Mock vote for testing."""

    agent: str = "agent1"
    choice: str = "agent2"
    reasoning: str = "I prefer this proposal"
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_global_cache():
    """Clean the global weight cache manager between tests."""
    _weight_cache_manager.clear()
    yield
    _weight_cache_manager.clear()


# ===========================================================================
# WeightFactors
# ===========================================================================

class TestWeightFactors:
    """Tests for WeightFactors dataclass."""

    def test_default_factors_all_one(self):
        """All default factors are 1.0."""
        factors = WeightFactors()
        assert factors.reputation == 1.0
        assert factors.reliability == 1.0
        assert factors.consistency == 1.0
        assert factors.calibration == 1.0
        assert factors.elo_skill == 1.0
        assert factors.self_vote == 1.0
        assert factors.verbosity == 1.0

    def test_total_is_product_of_all_seven_factors(self):
        """Total multiplies all seven factor fields together."""
        factors = WeightFactors(
            reputation=1.5,
            reliability=0.8,
            consistency=0.9,
            calibration=1.1,
            elo_skill=1.3,
            self_vote=0.5,
            verbosity=0.7,
        )
        expected = 1.5 * 0.8 * 0.9 * 1.1 * 1.3 * 0.5 * 0.7
        assert factors.total == pytest.approx(expected)

    def test_total_default_is_one(self):
        """Default total equals 1.0 (all defaults multiplied)."""
        assert WeightFactors().total == 1.0

    def test_single_zero_factor_makes_total_zero(self):
        """A single zero factor zeroes out the total."""
        factors = WeightFactors(self_vote=0.0)
        assert factors.total == 0.0


# ===========================================================================
# WeightCache
# ===========================================================================

class TestWeightCache:
    """Tests for WeightCache class."""

    def test_get_returns_none_on_miss(self):
        """get() returns None for an uncached agent."""
        cache = WeightCache("sess-1")
        assert cache.get("alice", "general") is None

    def test_put_and_get_round_trip(self):
        """put() then get() returns the stored tuple."""
        cache = WeightCache("sess-1")
        factors = WeightFactors(reputation=1.2)
        cache.put("alice", "general", 1.2, factors)

        result = cache.get("alice", "general")
        assert result is not None
        weight, returned_factors = result
        assert weight == 1.2
        assert returned_factors.reputation == 1.2

    def test_ttl_expiry(self):
        """Entries expire after ttl_seconds."""
        cache = WeightCache("sess-1", ttl_seconds=0.05)
        cache.put("alice", "general", 1.0, WeightFactors())

        assert cache.get("alice", "general") is not None
        time.sleep(0.08)
        assert cache.get("alice", "general") is None

    def test_elo_invalidation_causes_miss(self):
        """invalidate_elo() makes existing entries stale."""
        cache = WeightCache("sess-1")
        cache.put("alice", "general", 1.0, WeightFactors())
        assert cache.get("alice", "general") is not None

        cache.invalidate_elo()
        assert cache.get("alice", "general") is None

    def test_agent_invalidation_removes_only_target(self):
        """invalidate_agent() removes only that agent's entries."""
        cache = WeightCache("sess-1")
        cache.put("alice", "general", 1.0, WeightFactors())
        cache.put("bob", "general", 1.5, WeightFactors())

        cache.invalidate_agent("alice")

        assert cache.get("alice", "general") is None
        assert cache.get("bob", "general") is not None

    def test_lru_eviction_at_max_size(self):
        """Oldest entry is evicted when cache reaches max_size."""
        cache = WeightCache("sess-1", max_size=2)
        cache.put("a1", "d", 1.0, WeightFactors())
        time.sleep(0.01)
        cache.put("a2", "d", 2.0, WeightFactors())
        time.sleep(0.01)
        # This should evict a1 (oldest)
        cache.put("a3", "d", 3.0, WeightFactors())

        assert cache.get("a1", "d") is None
        assert cache.get("a2", "d") is not None
        assert cache.get("a3", "d") is not None

    def test_clear_resets_everything(self):
        """clear() empties cache and resets stats."""
        cache = WeightCache("sess-1")
        cache.put("alice", "general", 1.0, WeightFactors())
        cache.get("alice", "general")  # hit
        cache.get("bob", "general")    # miss

        cache.clear()

        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_stats_tracking(self):
        """get_stats() tracks hits, misses, and hit_rate."""
        cache = WeightCache("sess-1")
        cache.put("alice", "general", 1.0, WeightFactors())

        cache.get("alice", "general")  # hit
        cache.get("alice", "general")  # hit
        cache.get("bob", "general")    # miss

        stats = cache.get_stats()
        assert stats["session_id"] == "sess-1"
        assert stats["size"] == 1
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3)

    def test_thread_safety_concurrent_puts(self):
        """Concurrent puts do not corrupt cache state."""
        cache = WeightCache("sess-1", max_size=256)
        errors = []

        def writer(agent_id: int):
            try:
                for i in range(50):
                    name = f"agent-{agent_id}-{i}"
                    cache.put(name, "general", float(i), WeightFactors())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        stats = cache.get_stats()
        assert stats["size"] <= 256


# ===========================================================================
# Global cache manager
# ===========================================================================

class TestGlobalCacheManager:
    """Tests for get_weight_cache / cleanup_weight_cache."""

    def test_get_creates_new_cache(self):
        """get_weight_cache creates a new cache for unknown session."""
        cache = get_weight_cache("new-session")
        assert cache.session_id == "new-session"

    def test_get_returns_same_instance(self):
        """Repeated calls return the same cache object."""
        c1 = get_weight_cache("s1")
        c2 = get_weight_cache("s1")
        assert c1 is c2

    def test_cleanup_removes_cache(self):
        """cleanup_weight_cache removes the session's cache."""
        get_weight_cache("s1")
        assert "s1" in _weight_cache_manager

        cleanup_weight_cache("s1")
        assert "s1" not in _weight_cache_manager

    def test_cleanup_nonexistent_session_is_safe(self):
        """cleanup_weight_cache on unknown session does not raise."""
        cleanup_weight_cache("does-not-exist")


# ===========================================================================
# WeightCalculator
# ===========================================================================

class TestWeightCalculator:
    """Tests for WeightCalculator class."""

    # --- Default / basic ---

    def test_default_weight_no_systems(self):
        """Weight is 1.0 when no external systems are configured."""
        calc = WeightCalculator()
        assert calc.get_weight("agent1") == 1.0

    def test_disabled_factors_all_yield_one(self):
        """When all factors disabled, weight is always 1.0."""
        config = WeightCalculatorConfig(
            enable_reputation=False,
            enable_reliability=False,
            enable_consistency=False,
            enable_calibration=False,
            enable_elo_skill=False,
        )
        memory = MagicMock()
        memory.get_vote_weight.return_value = 99.0

        calc = WeightCalculator(memory=memory, config=config)
        weight, factors = calc.get_weight_with_factors("agent1")

        assert factors.reputation == 1.0
        assert factors.reliability == 1.0
        assert factors.consistency == 1.0
        assert factors.calibration == 1.0
        assert factors.elo_skill == 1.0
        assert weight == 1.0

    # --- Individual factor methods ---

    def test_reputation_weight_from_memory(self):
        """_get_reputation_weight delegates to memory.get_vote_weight."""
        memory = MagicMock()
        memory.get_vote_weight.return_value = 1.3
        calc = WeightCalculator(memory=memory)

        w = calc.get_weight("agent1")
        memory.get_vote_weight.assert_called_with("agent1")
        assert w == pytest.approx(1.3, rel=0.01)

    def test_reliability_weight_from_agent_weights(self):
        """_get_reliability_weight looks up agent_weights dict."""
        calc = WeightCalculator(agent_weights={"a1": 0.85})
        _, factors = calc.get_weight_with_factors("a1")
        assert factors.reliability == 0.85

    def test_reliability_weight_missing_agent_defaults_to_one(self):
        """Agent not in agent_weights dict gets reliability 1.0."""
        calc = WeightCalculator(agent_weights={"other": 0.5})
        _, factors = calc.get_weight_with_factors("unknown")
        assert factors.reliability == 1.0

    def test_consistency_weight_maps_score(self):
        """Consistency score 0-1 maps to 0.5-1.0."""
        fd = MagicMock()
        fd.get_agent_consistency.return_value = MockConsistency(consistency_score=0.6)

        calc = WeightCalculator(flip_detector=fd)
        _, factors = calc.get_weight_with_factors("a1")

        # 0.5 + (0.6 * 0.5) = 0.8
        assert factors.consistency == pytest.approx(0.8)

    def test_consistency_weight_perfect_score(self):
        """Consistency score 1.0 maps to weight 1.0."""
        fd = MagicMock()
        fd.get_agent_consistency.return_value = MockConsistency(consistency_score=1.0)
        calc = WeightCalculator(flip_detector=fd)
        _, factors = calc.get_weight_with_factors("a1")
        assert factors.consistency == pytest.approx(1.0)

    def test_consistency_weight_zero_score(self):
        """Consistency score 0.0 maps to weight 0.5."""
        fd = MagicMock()
        fd.get_agent_consistency.return_value = MockConsistency(consistency_score=0.0)
        calc = WeightCalculator(flip_detector=fd)
        _, factors = calc.get_weight_with_factors("a1")
        assert factors.consistency == pytest.approx(0.5)

    def test_calibration_weight_from_ratings_cache(self):
        """Calibration uses cached ELO ratings when available."""
        elo = MagicMock()
        elo.get_ratings_batch.return_value = {
            "a1": MockRating(calibration_score=0.8),
        }
        calc = WeightCalculator(elo_system=elo)
        calc.compute_weights([MockAgent("a1")])
        _, factors = calc.get_weight_with_factors("a1")
        # 0.5 + 0.8 = 1.3
        assert factors.calibration == pytest.approx(1.3)

    def test_calibration_weight_from_callback_fallback(self):
        """Calibration falls back to callback when not in ratings cache."""
        cb = MagicMock(return_value=1.15)
        calc = WeightCalculator(get_calibration_weight=cb)
        _, factors = calc.get_weight_with_factors("a1")
        cb.assert_called_with("a1")
        assert factors.calibration == pytest.approx(1.15)

    # --- ELO skill weight formula ---

    def test_elo_skill_weight_at_baseline_is_one(self):
        """ELO at baseline produces weight 1.0."""
        elo = MagicMock()
        elo.get_rating.return_value = MockRating(elo=1500.0)
        calc = WeightCalculator(elo_system=elo)

        _, factors = calc.get_weight_with_factors("a1")
        assert factors.elo_skill == pytest.approx(1.0)

    def test_elo_skill_weight_above_baseline(self):
        """ELO 2000 with defaults: 1.0 + ((2000-1500)/500)*0.3 = 1.3."""
        elo = MagicMock()
        elo.get_rating.return_value = MockRating(elo=2000.0)
        calc = WeightCalculator(elo_system=elo)

        _, factors = calc.get_weight_with_factors("a1")
        assert factors.elo_skill == pytest.approx(1.3)

    def test_elo_skill_weight_below_baseline(self):
        """ELO 1000 with defaults: 1.0 + ((1000-1500)/500)*0.3 = 0.7."""
        elo = MagicMock()
        elo.get_rating.return_value = MockRating(elo=1000.0)
        calc = WeightCalculator(elo_system=elo)

        _, factors = calc.get_weight_with_factors("a1")
        assert factors.elo_skill == pytest.approx(0.7)

    def test_elo_skill_weight_clamped_to_min(self):
        """Very low ELO is clamped to elo_min_weight (0.5)."""
        elo = MagicMock()
        elo.get_rating.return_value = MockRating(elo=0.0)
        calc = WeightCalculator(elo_system=elo)

        _, factors = calc.get_weight_with_factors("a1")
        # 1.0 + ((0-1500)/500)*0.3 = 1.0 + (-3.0*0.3) = 0.1, clamped to 0.5
        assert factors.elo_skill == pytest.approx(0.5)

    def test_elo_skill_weight_clamped_to_max(self):
        """Very high ELO is clamped to elo_max_weight (2.0)."""
        elo = MagicMock()
        elo.get_rating.return_value = MockRating(elo=10000.0)
        calc = WeightCalculator(elo_system=elo)

        _, factors = calc.get_weight_with_factors("a1")
        assert factors.elo_skill == pytest.approx(2.0)

    # --- Bounds clamping ---

    def test_weight_clamped_to_max(self):
        """Final weight is clamped to config.max_weight."""
        config = WeightCalculatorConfig(max_weight=2.0)
        memory = MagicMock()
        memory.get_vote_weight.return_value = 10.0
        calc = WeightCalculator(memory=memory, config=config)

        assert calc.get_weight("a1") == 2.0

    def test_weight_clamped_to_min(self):
        """Final weight is clamped to config.min_weight."""
        config = WeightCalculatorConfig(min_weight=0.5)
        memory = MagicMock()
        memory.get_vote_weight.return_value = 0.01
        calc = WeightCalculator(memory=memory, config=config)

        assert calc.get_weight("a1") == 0.5

    # --- Batch compute ---

    def test_compute_weights_batch(self):
        """compute_weights returns a dict for all agents."""
        calc = WeightCalculator(
            agent_weights={"a1": 1.5, "a2": 0.9},
        )
        agents = [MockAgent("a1"), MockAgent("a2")]
        weights = calc.compute_weights(agents)

        assert weights["a1"] == pytest.approx(1.5)
        assert weights["a2"] == pytest.approx(0.9)

    # --- Self-vote mitigation ---

    @patch("aragora.debate.phases.weight_calculator.detect_self_vote", return_value=True)
    def test_self_vote_exclude_mode(self, _mock_detect):
        """In 'exclude' mode, self-votes get weight 0.0 (clamped to min)."""
        config = WeightCalculatorConfig(
            enable_self_vote_mitigation=True,
            self_vote_mode="exclude",
            enable_reputation=False,
            enable_reliability=False,
            enable_consistency=False,
            enable_calibration=False,
            enable_elo_skill=False,
            min_weight=0.0,
        )
        calc = WeightCalculator(config=config)
        vote = MockVote(agent="agent1", choice="agent1")

        w = calc.get_weight_for_vote("agent1", vote)
        assert w == pytest.approx(0.0)

    @patch("aragora.debate.phases.weight_calculator.detect_self_vote", return_value=True)
    def test_self_vote_downweight_mode(self, _mock_detect):
        """In 'downweight' mode, self-votes are multiplied by downweight factor."""
        config = WeightCalculatorConfig(
            enable_self_vote_mitigation=True,
            self_vote_mode="downweight",
            self_vote_downweight=0.5,
            enable_reputation=False,
            enable_reliability=False,
            enable_consistency=False,
            enable_calibration=False,
            enable_elo_skill=False,
        )
        calc = WeightCalculator(config=config)
        vote = MockVote(agent="agent1", choice="agent1")

        w = calc.get_weight_for_vote("agent1", vote)
        assert w == pytest.approx(0.5)

    @patch("aragora.debate.phases.weight_calculator.detect_self_vote", return_value=True)
    def test_self_vote_log_only_mode(self, _mock_detect):
        """In 'log_only' mode, self-votes keep weight 1.0."""
        config = WeightCalculatorConfig(
            enable_self_vote_mitigation=True,
            self_vote_mode="log_only",
            enable_reputation=False,
            enable_reliability=False,
            enable_consistency=False,
            enable_calibration=False,
            enable_elo_skill=False,
        )
        calc = WeightCalculator(config=config)
        vote = MockVote(agent="agent1", choice="agent1")

        w = calc.get_weight_for_vote("agent1", vote)
        assert w == pytest.approx(1.0)

    @patch("aragora.debate.phases.weight_calculator.detect_self_vote", return_value=False)
    def test_self_vote_not_detected_keeps_full_weight(self, _mock_detect):
        """Non-self-votes keep the self_vote factor at 1.0."""
        config = WeightCalculatorConfig(
            enable_self_vote_mitigation=True,
            self_vote_mode="exclude",
            enable_reputation=False,
            enable_reliability=False,
            enable_consistency=False,
            enable_calibration=False,
            enable_elo_skill=False,
        )
        calc = WeightCalculator(config=config)
        vote = MockVote(agent="agent1", choice="agent2")

        w = calc.get_weight_for_vote("agent1", vote)
        assert w == pytest.approx(1.0)

    # --- Verbosity normalization ---

    @patch("aragora.debate.phases.weight_calculator.calculate_verbosity_factor", return_value=0.75)
    def test_verbosity_normalization_applies_factor(self, _mock_verbosity):
        """Verbosity penalty multiplies into the final weight."""
        config = WeightCalculatorConfig(
            enable_verbosity_normalization=True,
            enable_reputation=False,
            enable_reliability=False,
            enable_consistency=False,
            enable_calibration=False,
            enable_elo_skill=False,
        )
        calc = WeightCalculator(config=config)
        calc._current_proposals = {"agent2": "x" * 5000}
        vote = MockVote(agent="agent1", choice="agent2")

        w = calc.get_weight_for_vote("agent1", vote)
        assert w == pytest.approx(0.75)

    def test_verbosity_disabled_by_default(self):
        """Verbosity normalization is disabled by default."""
        config = WeightCalculatorConfig()
        assert config.enable_verbosity_normalization is False

    # --- Session caching ---

    def test_cache_hit_avoids_recomputation(self):
        """Second get_weight call uses cached value, not memory."""
        memory = MagicMock()
        memory.get_vote_weight.return_value = 1.3

        calc = WeightCalculator(
            memory=memory,
            session_id="sess-1",
            enable_cache=True,
        )

        w1 = calc.get_weight("a1")
        w2 = calc.get_weight("a1")

        assert w1 == w2 == pytest.approx(1.3)
        # Memory should only be called once; second call hits cache
        assert memory.get_vote_weight.call_count == 1

    def test_cache_disabled_when_no_session_id(self):
        """Without session_id, caching is disabled."""
        memory = MagicMock()
        memory.get_vote_weight.return_value = 1.3
        calc = WeightCalculator(memory=memory, enable_cache=True, session_id=None)

        calc.get_weight("a1")
        calc.get_weight("a1")

        assert memory.get_vote_weight.call_count == 2

    def test_invalidate_elo_cache_forces_recompute(self):
        """invalidate_elo_cache makes next get_weight recompute."""
        memory = MagicMock()
        memory.get_vote_weight.return_value = 1.3

        calc = WeightCalculator(memory=memory, session_id="sess-1")
        calc.get_weight("a1")
        calc.invalidate_elo_cache()
        calc.get_weight("a1")

        assert memory.get_vote_weight.call_count == 2

    def test_invalidate_agent_cache_forces_recompute(self):
        """invalidate_agent_cache clears only that agent."""
        memory = MagicMock()
        memory.get_vote_weight.return_value = 1.0
        calc = WeightCalculator(memory=memory, session_id="sess-1")

        calc.get_weight("a1")
        calc.get_weight("a2")
        calc.invalidate_agent_cache("a1")

        # a1 recomputed, a2 still cached
        calc.get_weight("a1")
        calc.get_weight("a2")

        a1_calls = [c for c in memory.get_vote_weight.call_args_list if c.args == ("a1",)]
        a2_calls = [c for c in memory.get_vote_weight.call_args_list if c.args == ("a2",)]
        assert len(a1_calls) == 2
        assert len(a2_calls) == 1

    def test_cleanup_removes_session_cache(self):
        """cleanup() removes the session from global cache manager."""
        calc = WeightCalculator(session_id="sess-cleanup")
        calc.get_weight("a1")

        assert "sess-cleanup" in _weight_cache_manager
        calc.cleanup()
        assert "sess-cleanup" not in _weight_cache_manager

    def test_get_cache_stats_returns_dict(self):
        """get_cache_stats returns stats when caching is enabled."""
        calc = WeightCalculator(session_id="sess-stats")
        calc.get_weight("a1")

        stats = calc.get_cache_stats()
        assert stats is not None
        assert "hits" in stats
        assert "misses" in stats

    def test_get_cache_stats_none_when_disabled(self):
        """get_cache_stats returns None when caching is disabled."""
        calc = WeightCalculator()
        assert calc.get_cache_stats() is None

    # --- Error handling / graceful degradation ---

    def test_memory_error_returns_default(self):
        """Memory errors degrade gracefully to 1.0."""
        memory = MagicMock()
        memory.get_vote_weight.side_effect = RuntimeError("Memory error")
        calc = WeightCalculator(memory=memory)

        assert calc.get_weight("a1") == 1.0

    def test_flip_detector_error_returns_default(self):
        """FlipDetector errors degrade gracefully to 1.0."""
        fd = MagicMock()
        fd.get_agent_consistency.side_effect = RuntimeError("FD error")
        calc = WeightCalculator(flip_detector=fd)

        assert calc.get_weight("a1") == 1.0

    def test_calibration_callback_error_returns_default(self):
        """Calibration callback errors degrade gracefully to 1.0."""
        cb = MagicMock(side_effect=RuntimeError("Cal error"))
        calc = WeightCalculator(get_calibration_weight=cb)

        assert calc.get_weight("a1") == 1.0

    def test_elo_system_error_returns_default(self):
        """ELO system errors degrade gracefully to 1.0."""
        elo = MagicMock()
        elo.get_rating.side_effect = RuntimeError("ELO error")
        elo.get_ratings_batch.side_effect = RuntimeError("batch error")
        calc = WeightCalculator(elo_system=elo)

        assert calc.get_weight("a1") == 1.0

    def test_batch_prefetch_failure_degrades_gracefully(self):
        """If batch prefetch fails, individual weights still work."""
        elo = MagicMock()
        elo.get_ratings_batch.side_effect = RuntimeError("batch fail")
        elo.get_rating.return_value = MockRating(elo=2000.0)

        calc = WeightCalculator(elo_system=elo)
        agents = [MockAgent("a1")]
        weights = calc.compute_weights(agents)

        # Despite batch failure, get_rating fallback works
        assert weights["a1"] == pytest.approx(1.3)

    # --- compute_weights_with_context ---

    @patch("aragora.debate.phases.weight_calculator.detect_self_vote", return_value=False)
    def test_compute_weights_with_context(self, _mock_detect):
        """compute_weights_with_context runs bias mitigation for each vote."""
        config = WeightCalculatorConfig(
            enable_self_vote_mitigation=True,
            enable_reputation=False,
            enable_reliability=False,
            enable_consistency=False,
            enable_calibration=False,
            enable_elo_skill=False,
        )
        calc = WeightCalculator(config=config)
        agents = [MockAgent("a1"), MockAgent("a2")]
        votes = [
            MockVote(agent="a1", choice="a2"),
            MockVote(agent="a2", choice="a1"),
        ]
        proposals = {"a1": "proposal 1", "a2": "proposal 2"}

        weights = calc.compute_weights_with_context(agents, votes, proposals)

        assert "a1" in weights
        assert "a2" in weights
        # Context should be cleared afterwards
        assert calc._current_votes == []
        assert calc._current_proposals == {}
