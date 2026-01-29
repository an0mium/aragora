"""
Tests for PerformanceAdapter - Unified adapter for agent performance metrics.

Tests cover:
- Dataclasses: KMEloPattern, EloAdjustmentRecommendation, EloSyncResult,
  RatingSearchResult, AgentExpertise, ExpertiseSearchResult
- Initialization and configuration
- Rating storage and retrieval
- Match storage and retrieval
- Calibration storage and retrieval
- Relationship storage (with threshold)
- Expertise storage and retrieval
- Domain expert queries with caching
- Cache invalidation and statistics
- Domain detection
- Event system
- KM pattern analysis (reverse flow)
- ELO adjustment computation
- Keyword search and get_all_records
- Backwards compatibility aliases
"""

import pytest
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.adapters.performance_adapter import (
    AgentExpertise,
    EloAdapter,
    EloAdjustmentRecommendation,
    EloSyncResult,
    ExpertiseSearchResult,
    KMEloPattern,
    PerformanceAdapter,
    RankingAdapter,
    RatingSearchResult,
)


# =============================================================================
# Mock Helpers
# =============================================================================


def _make_mock_rating(
    agent_name: str = "claude",
    elo: float = 1500.0,
    wins: int = 10,
    losses: int = 5,
    draws: int = 2,
    domain_elos: dict | None = None,
    debates_count: int = 17,
    games_played: int = 17,
    critiques_accepted: int = 8,
    critiques_total: int = 12,
    critique_acceptance_rate: float = 0.667,
    calibration_correct: int = 7,
    calibration_total: int = 10,
    calibration_accuracy: float = 0.7,
    updated_at: str = "2025-01-15T10:00:00+00:00",
) -> MagicMock:
    """Create a mock AgentRating."""
    rating = MagicMock()
    rating.agent_name = agent_name
    rating.elo = elo
    rating.domain_elos = domain_elos or {"security": 1600, "coding": 1450}
    rating.wins = wins
    rating.losses = losses
    rating.draws = draws
    rating.debates_count = debates_count
    rating.win_rate = wins / max(1, wins + losses + draws)
    rating.games_played = games_played
    rating.critiques_accepted = critiques_accepted
    rating.critiques_total = critiques_total
    rating.critique_acceptance_rate = critique_acceptance_rate
    rating.calibration_correct = calibration_correct
    rating.calibration_total = calibration_total
    rating.calibration_accuracy = calibration_accuracy
    rating.updated_at = updated_at
    return rating


def _make_mock_match(
    debate_id: str = "d-123",
    winner: str = "claude",
    participants: list | None = None,
    domain: str = "security",
    scores: dict | None = None,
    created_at: str = "2025-01-15T10:00:00+00:00",
) -> MagicMock:
    """Create a mock MatchResult."""
    match = MagicMock()
    match.debate_id = debate_id
    match.winner = winner
    match.participants = participants or ["claude", "gpt4"]
    match.domain = domain
    match.scores = scores or {"claude": 0.8, "gpt4": 0.6}
    match.created_at = created_at
    return match


def _make_mock_relationship(
    agent_a: str = "claude",
    agent_b: str = "gpt4",
    debates_together: int = 10,
    a_wins_vs_b: int = 6,
    b_wins_vs_a: int = 3,
    draws: int = 1,
) -> MagicMock:
    """Create a mock RelationshipMetrics."""
    metrics = MagicMock()
    metrics.agent_a = agent_a
    metrics.agent_b = agent_b
    metrics.debates_together = debates_together
    metrics.a_wins_vs_b = a_wins_vs_b
    metrics.b_wins_vs_a = b_wins_vs_a
    metrics.draws = draws
    metrics.avg_elo_diff = 50.0
    metrics.synergy_score = 0.7
    return metrics


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestKMEloPattern:
    """Tests for KMEloPattern dataclass."""

    def test_create_with_required_fields(self):
        """Should create KMEloPattern with required fields and defaults."""
        pattern = KMEloPattern(
            agent_name="claude",
            pattern_type="success_contributor",
            confidence=0.85,
        )

        assert pattern.agent_name == "claude"
        assert pattern.pattern_type == "success_contributor"
        assert pattern.confidence == 0.85
        assert pattern.observation_count == 1
        assert pattern.domain is None
        assert pattern.debate_ids == []
        assert pattern.metadata == {}

    def test_create_with_all_fields(self):
        """Should create KMEloPattern with all fields populated."""
        pattern = KMEloPattern(
            agent_name="gpt4",
            pattern_type="domain_expert",
            confidence=0.92,
            observation_count=15,
            domain="security",
            debate_ids=["d-1", "d-2"],
            metadata={"domain_item_count": 15},
        )

        assert pattern.domain == "security"
        assert pattern.observation_count == 15
        assert len(pattern.debate_ids) == 2
        assert pattern.metadata["domain_item_count"] == 15


class TestEloAdjustmentRecommendation:
    """Tests for EloAdjustmentRecommendation dataclass."""

    def test_create_with_defaults(self):
        """Should create recommendation with correct defaults."""
        rec = EloAdjustmentRecommendation(
            agent_name="claude",
            adjustment=25.0,
            reason="KM success pattern",
        )

        assert rec.agent_name == "claude"
        assert rec.adjustment == 25.0
        assert rec.confidence == 0.7
        assert rec.domain is None
        assert rec.applied is False
        assert rec.patterns == []

    def test_create_with_all_fields(self):
        """Should accept all fields."""
        pattern = KMEloPattern("claude", "success_contributor", 0.9)
        rec = EloAdjustmentRecommendation(
            agent_name="claude",
            adjustment=-10.0,
            reason="contradiction source",
            patterns=[pattern],
            confidence=0.85,
            domain="security",
            applied=True,
        )

        assert rec.adjustment == -10.0
        assert rec.applied is True
        assert len(rec.patterns) == 1


class TestEloSyncResult:
    """Tests for EloSyncResult dataclass."""

    def test_default_values(self):
        """Should have all zero/empty defaults."""
        result = EloSyncResult()

        assert result.total_patterns == 0
        assert result.adjustments_recommended == 0
        assert result.adjustments_applied == 0
        assert result.adjustments_skipped == 0
        assert result.total_elo_change == 0.0
        assert result.agents_affected == []
        assert result.errors == []
        assert result.duration_ms == 0


class TestRatingSearchResult:
    """Tests for RatingSearchResult dataclass."""

    def test_create_with_defaults(self):
        """Should create with default relevance score."""
        result = RatingSearchResult(rating={"agent_name": "claude", "elo": 1500})

        assert result.rating["agent_name"] == "claude"
        assert result.relevance_score == 0.0


class TestExpertiseSearchResult:
    """Tests for ExpertiseSearchResult dataclass."""

    def test_post_init_defaults_matched_domains(self):
        """Should default matched_domains to empty list via __post_init__."""
        result = ExpertiseSearchResult(expertise={"agent": "test"})

        assert result.matched_domains == []
        assert result.relevance_score == 0.0

    def test_preserves_explicit_matched_domains(self):
        """Should preserve explicitly set matched_domains."""
        result = ExpertiseSearchResult(
            expertise={"agent": "test"},
            relevance_score=0.9,
            matched_domains=["coding", "architecture"],
        )

        assert result.matched_domains == ["coding", "architecture"]


# =============================================================================
# Initialization Tests
# =============================================================================


class TestPerformanceAdapterInit:
    """Tests for PerformanceAdapter initialization."""

    def test_default_init(self):
        """Should initialize with default values."""
        adapter = PerformanceAdapter(enable_resilience=False)

        assert adapter.elo_system is None
        assert adapter._enable_dual_write is False
        assert adapter._event_callback is None
        assert adapter._cache_ttl_seconds == 60.0
        assert len(adapter._ratings) == 0
        assert len(adapter._matches) == 0
        assert len(adapter._calibrations) == 0
        assert len(adapter._relationships) == 0
        assert len(adapter._expertise) == 0

    def test_init_with_elo_system(self):
        """Should accept EloSystem."""
        mock_elo = MagicMock()
        adapter = PerformanceAdapter(elo_system=mock_elo, enable_resilience=False)

        assert adapter.elo_system is mock_elo

    def test_init_with_custom_cache_ttl(self):
        """Should accept custom cache TTL."""
        adapter = PerformanceAdapter(cache_ttl_seconds=300.0, enable_resilience=False)

        assert adapter._cache_ttl_seconds == 300.0

    def test_init_with_event_callback(self):
        """Should accept event callback."""
        cb = MagicMock()
        adapter = PerformanceAdapter(event_callback=cb, enable_resilience=False)

        assert adapter._event_callback is cb

    def test_adapter_name_and_prefixes(self):
        """Should have correct class-level constants."""
        assert PerformanceAdapter.adapter_name == "performance"
        assert PerformanceAdapter.source_type == "elo"
        assert PerformanceAdapter.ELO_PREFIX == "el_"
        assert PerformanceAdapter.EXPERTISE_PREFIX == "ex_"
        assert PerformanceAdapter.DOMAIN_PREFIX == "dm_"
        assert PerformanceAdapter.ID_PREFIX == "el_"


# =============================================================================
# Event System Tests
# =============================================================================


class TestEventSystem:
    """Tests for event emission."""

    def test_set_event_callback(self):
        """Should update the event callback."""
        adapter = PerformanceAdapter(enable_resilience=False)
        cb = MagicMock()
        adapter.set_event_callback(cb)

        assert adapter._event_callback is cb

    def test_emit_event_calls_callback(self):
        """Should call callback with event type and data."""
        cb = MagicMock()
        adapter = PerformanceAdapter(event_callback=cb, enable_resilience=False)

        adapter._emit_event("test_event", {"key": "value"})

        cb.assert_called_once_with("test_event", {"key": "value"})

    def test_emit_event_no_callback(self):
        """Should silently skip if no callback configured."""
        adapter = PerformanceAdapter(enable_resilience=False)
        # Should not raise
        adapter._emit_event("test_event", {"key": "value"})

    def test_emit_event_swallows_callback_error(self):
        """Should log warning but not raise if callback errors."""
        cb = MagicMock(side_effect=RuntimeError("callback failed"))
        adapter = PerformanceAdapter(event_callback=cb, enable_resilience=False)

        # Should not raise
        adapter._emit_event("test_event", {"key": "value"})


# =============================================================================
# Rating Storage Tests
# =============================================================================


class TestStoreRating:
    """Tests for store_rating method."""

    def test_store_rating_returns_id(self):
        """Should return a rating ID with el_ prefix."""
        adapter = PerformanceAdapter(enable_resilience=False)
        rating = _make_mock_rating()

        rating_id = adapter.store_rating(rating, debate_id="d-1")

        assert rating_id.startswith("el_claude_")
        assert rating_id in adapter._ratings

    def test_store_rating_data_fields(self):
        """Should store all rating fields correctly."""
        adapter = PerformanceAdapter(enable_resilience=False)
        rating = _make_mock_rating(elo=1600.0, wins=12, losses=3)

        rating_id = adapter.store_rating(rating, debate_id="d-1", reason="win")

        stored = adapter._ratings[rating_id]
        assert stored["agent_name"] == "claude"
        assert stored["elo"] == 1600.0
        assert stored["wins"] == 12
        assert stored["losses"] == 3
        assert stored["debate_id"] == "d-1"
        assert stored["reason"] == "win"

    def test_store_rating_updates_agent_index(self):
        """Should index rating by agent name."""
        adapter = PerformanceAdapter(enable_resilience=False)
        rating = _make_mock_rating()

        rating_id = adapter.store_rating(rating)

        assert "claude" in adapter._agent_ratings
        assert rating_id in adapter._agent_ratings["claude"]

    def test_store_rating_updates_domain_index(self):
        """Should index rating by domain."""
        adapter = PerformanceAdapter(enable_resilience=False)
        rating = _make_mock_rating(domain_elos={"security": 1600})

        rating_id = adapter.store_rating(rating)

        assert "security" in adapter._domain_ratings
        assert rating_id in adapter._domain_ratings["security"]


# =============================================================================
# Match Storage Tests
# =============================================================================


class TestStoreMatch:
    """Tests for store_match method."""

    def test_store_match_returns_id(self):
        """Should return a match ID with el_match_ prefix."""
        adapter = PerformanceAdapter(enable_resilience=False)
        match = _make_mock_match(debate_id="d-42")

        match_id = adapter.store_match(match)

        assert match_id == "el_match_d-42"
        assert match_id in adapter._matches

    def test_store_match_indexes_participants(self):
        """Should index match by all participants."""
        adapter = PerformanceAdapter(enable_resilience=False)
        match = _make_mock_match(participants=["claude", "gpt4", "gemini"])

        match_id = adapter.store_match(match)

        for agent in ["claude", "gpt4", "gemini"]:
            assert agent in adapter._agent_matches
            assert match_id in adapter._agent_matches[agent]


# =============================================================================
# Calibration Storage Tests
# =============================================================================


class TestStoreCalibration:
    """Tests for store_calibration method."""

    def test_store_calibration_returns_id(self):
        """Should return a calibration ID."""
        adapter = PerformanceAdapter(enable_resilience=False)

        cal_id = adapter.store_calibration(
            agent_name="claude",
            debate_id="d-5",
            predicted_winner="claude",
            predicted_confidence=0.8,
            actual_winner="claude",
            was_correct=True,
            brier_score=0.04,
        )

        assert cal_id == "el_cal_claude_d-5"
        assert cal_id in adapter._calibrations

    def test_store_calibration_data(self):
        """Should store all calibration fields."""
        adapter = PerformanceAdapter(enable_resilience=False)

        cal_id = adapter.store_calibration(
            agent_name="gpt4",
            debate_id="d-7",
            predicted_winner="gpt4",
            predicted_confidence=0.6,
            actual_winner="claude",
            was_correct=False,
            brier_score=0.36,
        )

        stored = adapter._calibrations[cal_id]
        assert stored["was_correct"] is False
        assert stored["brier_score"] == 0.36
        assert stored["predicted_winner"] == "gpt4"
        assert stored["actual_winner"] == "claude"


# =============================================================================
# Relationship Storage Tests
# =============================================================================


class TestStoreRelationship:
    """Tests for store_relationship method."""

    def test_store_relationship_above_threshold(self):
        """Should store relationship when debates_together >= threshold."""
        adapter = PerformanceAdapter(enable_resilience=False)
        metrics = _make_mock_relationship(debates_together=10)

        rel_id = adapter.store_relationship(metrics)

        assert rel_id is not None
        assert rel_id == "el_rel_claude_gpt4"
        assert rel_id in adapter._relationships

    def test_store_relationship_below_threshold(self):
        """Should return None when debates_together < threshold."""
        adapter = PerformanceAdapter(enable_resilience=False)
        metrics = _make_mock_relationship(debates_together=3)

        rel_id = adapter.store_relationship(metrics)

        assert rel_id is None
        assert len(adapter._relationships) == 0


# =============================================================================
# Expertise Storage Tests
# =============================================================================


class TestStoreAgentExpertise:
    """Tests for store_agent_expertise method."""

    def test_store_expertise_significant_delta(self):
        """Should store expertise when delta >= MIN_ELO_CHANGE."""
        adapter = PerformanceAdapter(enable_resilience=False)

        eid = adapter.store_agent_expertise("claude", "security", 1600, delta=50)

        assert eid is not None
        assert eid.startswith("ex_")
        assert "claude:security" in adapter._expertise

    def test_store_expertise_small_delta_returns_none(self):
        """Should skip storage when delta < MIN_ELO_CHANGE."""
        adapter = PerformanceAdapter(enable_resilience=False)

        eid = adapter.store_agent_expertise("claude", "security", 1510, delta=10)

        assert eid is None
        assert len(adapter._expertise) == 0

    def test_store_expertise_updates_indices(self):
        """Should update domain and agent indices."""
        adapter = PerformanceAdapter(enable_resilience=False)

        adapter.store_agent_expertise("claude", "security", 1600, delta=50)

        assert "security" in adapter._domain_agents
        assert "claude" in adapter._domain_agents["security"]
        assert "claude" in adapter._agent_domains
        assert "security" in adapter._agent_domains["claude"]

    def test_store_expertise_increments_debate_count(self):
        """Should increment debate_count on repeated storage."""
        adapter = PerformanceAdapter(enable_resilience=False)

        adapter.store_agent_expertise("claude", "security", 1600, delta=50)
        adapter.store_agent_expertise("claude", "security", 1650, delta=50)

        data = adapter._expertise["claude:security"]
        assert data["debate_count"] == 2

    def test_store_expertise_confidence_calculation(self):
        """Should calculate confidence as debate_count / MIN_DEBATES_FOR_CONFIDENCE."""
        adapter = PerformanceAdapter(enable_resilience=False)

        adapter.store_agent_expertise("claude", "security", 1600, delta=50)

        data = adapter._expertise["claude:security"]
        assert data["confidence"] == 1 / 5  # 1 debate / MIN_DEBATES_FOR_CONFIDENCE(5)

    def test_store_expertise_appends_history(self):
        """Should append history records."""
        adapter = PerformanceAdapter(enable_resilience=False)

        adapter.store_agent_expertise("claude", "security", 1600, delta=50, debate_id="d-1")
        adapter.store_agent_expertise("claude", "coding", 1500, delta=30, debate_id="d-2")

        assert len(adapter._agent_history["claude"]) == 2

    def test_store_expertise_invalidates_cache(self):
        """Should invalidate domain cache on store."""
        adapter = PerformanceAdapter(enable_resilience=False)

        # Populate cache entry
        adapter._domain_experts_cache["security:10:0.0"] = (time.time(), [])

        adapter.store_agent_expertise("claude", "security", 1600, delta=50)

        assert "security:10:0.0" not in adapter._domain_experts_cache


# =============================================================================
# Rating Retrieval Tests
# =============================================================================


class TestRatingRetrieval:
    """Tests for rating retrieval methods."""

    def test_get_rating_by_id(self):
        """Should retrieve rating by full ID."""
        adapter = PerformanceAdapter(enable_resilience=False)
        rating = _make_mock_rating()
        rating_id = adapter.store_rating(rating)

        result = adapter.get_rating(rating_id)

        assert result is not None
        assert result["agent_name"] == "claude"

    def test_get_rating_auto_prefix(self):
        """Should add el_ prefix if missing."""
        adapter = PerformanceAdapter(enable_resilience=False)
        adapter._ratings["el_test_id"] = {"agent_name": "claude"}

        result = adapter.get_rating("test_id")

        assert result is not None
        assert result["agent_name"] == "claude"

    def test_get_rating_not_found(self):
        """Should return None for missing rating."""
        adapter = PerformanceAdapter(enable_resilience=False)

        assert adapter.get_rating("nonexistent") is None

    def test_get_agent_skill_history(self):
        """Should return rating snapshots newest first."""
        adapter = PerformanceAdapter(enable_resilience=False)
        r1 = _make_mock_rating(elo=1500)
        r2 = _make_mock_rating(elo=1550)

        adapter.store_rating(r1)
        adapter.store_rating(r2)

        history = adapter.get_agent_skill_history("claude", limit=10)

        assert len(history) == 2
        # Newest first (sorted by created_at desc)
        assert history[0]["elo"] >= history[-1]["elo"] or len(history) <= 1

    def test_get_agent_skill_history_respects_limit(self):
        """Should limit returned results."""
        adapter = PerformanceAdapter(enable_resilience=False)
        for i in range(5):
            adapter.store_rating(_make_mock_rating(elo=1500 + i * 10))

        history = adapter.get_agent_skill_history("claude", limit=2)

        assert len(history) == 2


# =============================================================================
# Match and Calibration Retrieval Tests
# =============================================================================


class TestMatchAndCalibrationRetrieval:
    """Tests for match/calibration retrieval."""

    def test_get_match_by_id(self):
        """Should retrieve match by ID."""
        adapter = PerformanceAdapter(enable_resilience=False)
        match = _make_mock_match(debate_id="d-99")
        adapter.store_match(match)

        result = adapter.get_match("el_match_d-99")

        assert result is not None
        assert result["winner"] == "claude"

    def test_get_match_auto_prefix(self):
        """Should add prefix if missing."""
        adapter = PerformanceAdapter(enable_resilience=False)
        match = _make_mock_match(debate_id="d-99")
        adapter.store_match(match)

        result = adapter.get_match("d-99")

        assert result is not None

    def test_get_agent_matches(self):
        """Should return matches for an agent."""
        adapter = PerformanceAdapter(enable_resilience=False)
        adapter.store_match(_make_mock_match(debate_id="d-1"))
        adapter.store_match(_make_mock_match(debate_id="d-2"))

        matches = adapter.get_agent_matches("claude")

        assert len(matches) == 2

    def test_get_agent_calibration_history(self):
        """Should return calibrations for agent."""
        adapter = PerformanceAdapter(enable_resilience=False)
        adapter.store_calibration("claude", "d-1", "claude", 0.8, "claude", True, 0.04)
        adapter.store_calibration("gpt4", "d-2", "gpt4", 0.7, "claude", False, 0.3)

        result = adapter.get_agent_calibration_history("claude")

        assert len(result) == 1
        assert result[0]["agent_name"] == "claude"

    def test_get_relationship_both_orderings(self):
        """Should find relationship regardless of agent order."""
        adapter = PerformanceAdapter(enable_resilience=False)
        metrics = _make_mock_relationship(agent_a="claude", agent_b="gpt4")
        adapter.store_relationship(metrics)

        # Both orderings should work
        assert adapter.get_relationship("claude", "gpt4") is not None
        assert adapter.get_relationship("gpt4", "claude") is not None


# =============================================================================
# Expertise Retrieval and Cache Tests
# =============================================================================


class TestExpertiseRetrieval:
    """Tests for expertise retrieval methods."""

    def test_get_agent_expertise_by_domain(self):
        """Should retrieve expertise for specific domain."""
        adapter = PerformanceAdapter(enable_resilience=False)
        adapter.store_agent_expertise("claude", "security", 1600, delta=50)

        result = adapter.get_agent_expertise("claude", domain="security")

        assert result is not None
        assert result["elo"] == 1600

    def test_get_agent_expertise_all_domains(self):
        """Should return all domains when no domain specified."""
        adapter = PerformanceAdapter(enable_resilience=False)
        adapter.store_agent_expertise("claude", "security", 1600, delta=50)
        adapter.store_agent_expertise("claude", "coding", 1500, delta=30)

        result = adapter.get_agent_expertise("claude")

        assert isinstance(result, dict)
        assert "security" in result
        assert "coding" in result

    def test_get_agent_expertise_not_found(self):
        """Should return None for unknown agent."""
        adapter = PerformanceAdapter(enable_resilience=False)

        assert adapter.get_agent_expertise("unknown") is None

    def test_get_domain_experts_sorted_by_elo(self):
        """Should return experts sorted by ELO descending."""
        adapter = PerformanceAdapter(enable_resilience=False)
        adapter.store_agent_expertise("claude", "security", 1600, delta=50)
        adapter.store_agent_expertise("gpt4", "security", 1700, delta=60)
        adapter.store_agent_expertise("gemini", "security", 1550, delta=40)

        experts = adapter.get_domain_experts("security")

        assert len(experts) == 3
        assert experts[0].agent_name == "gpt4"
        assert experts[0].elo == 1700
        assert experts[2].agent_name == "gemini"

    def test_get_domain_experts_respects_min_confidence(self):
        """Should filter by minimum confidence."""
        adapter = PerformanceAdapter(enable_resilience=False)
        adapter.store_agent_expertise("claude", "security", 1600, delta=50)
        # confidence = 1/5 = 0.2 for first entry

        experts = adapter.get_domain_experts("security", min_confidence=0.5)

        assert len(experts) == 0

    def test_get_domain_experts_cache_hit(self):
        """Should return cached results on second call."""
        adapter = PerformanceAdapter(enable_resilience=False)
        adapter.store_agent_expertise("claude", "security", 1600, delta=50)

        # First call - cache miss
        adapter.get_domain_experts("security")
        assert adapter._cache_misses == 1
        assert adapter._cache_hits == 0

        # Second call - cache hit
        adapter.get_domain_experts("security")
        assert adapter._cache_hits == 1

    def test_get_domain_experts_cache_expired(self):
        """Should re-query when cache is expired."""
        adapter = PerformanceAdapter(cache_ttl_seconds=0.01, enable_resilience=False)
        adapter.store_agent_expertise("claude", "security", 1600, delta=50)

        adapter.get_domain_experts("security")
        time.sleep(0.02)  # Wait for expiry
        adapter.get_domain_experts("security")

        # Both should be misses (second expired)
        assert adapter._cache_misses == 2

    def test_get_domain_experts_bypass_cache(self):
        """Should bypass cache when use_cache=False."""
        adapter = PerformanceAdapter(enable_resilience=False)
        adapter.store_agent_expertise("claude", "security", 1600, delta=50)

        adapter.get_domain_experts("security", use_cache=False)
        adapter.get_domain_experts("security", use_cache=False)

        assert adapter._cache_hits == 0
        assert adapter._cache_misses == 2

    def test_get_agent_history(self):
        """Should return history records newest first."""
        adapter = PerformanceAdapter(enable_resilience=False)
        adapter.store_agent_expertise("claude", "security", 1600, delta=50, debate_id="d-1")
        adapter.store_agent_expertise("claude", "coding", 1500, delta=30, debate_id="d-2")

        history = adapter.get_agent_history("claude")

        assert len(history) == 2

    def test_get_agent_history_domain_filter(self):
        """Should filter history by domain."""
        adapter = PerformanceAdapter(enable_resilience=False)
        adapter.store_agent_expertise("claude", "security", 1600, delta=50)
        adapter.store_agent_expertise("claude", "coding", 1500, delta=30)

        history = adapter.get_agent_history("claude", domain="security")

        assert len(history) == 1
        assert history[0]["domain"] == "security"


# =============================================================================
# Cache Management Tests
# =============================================================================


class TestCacheManagement:
    """Tests for cache invalidation and statistics."""

    def test_invalidate_all_cache(self):
        """Should clear entire cache when no domain specified."""
        adapter = PerformanceAdapter(enable_resilience=False)
        adapter._domain_experts_cache["security:10:0.0"] = (time.time(), [])
        adapter._domain_experts_cache["coding:10:0.0"] = (time.time(), [])

        count = adapter.invalidate_domain_cache()

        assert count == 2
        assert len(adapter._domain_experts_cache) == 0

    def test_invalidate_specific_domain(self):
        """Should only clear entries for specified domain."""
        adapter = PerformanceAdapter(enable_resilience=False)
        adapter._domain_experts_cache["security:10:0.0"] = (time.time(), [])
        adapter._domain_experts_cache["coding:10:0.0"] = (time.time(), [])

        count = adapter.invalidate_domain_cache("security")

        assert count == 1
        assert "coding:10:0.0" in adapter._domain_experts_cache

    def test_get_cache_stats(self):
        """Should return correct cache statistics."""
        adapter = PerformanceAdapter(enable_resilience=False)
        adapter._cache_hits = 10
        adapter._cache_misses = 5
        adapter._domain_experts_cache["test"] = (time.time(), [])

        stats = adapter.get_cache_stats()

        assert stats["cache_hits"] == 10
        assert stats["cache_misses"] == 5
        assert stats["cache_size"] == 1
        assert stats["hit_rate"] == pytest.approx(10 / 15)
        assert stats["ttl_seconds"] == 60.0

    def test_get_cache_stats_zero_total(self):
        """Should return 0.0 hit_rate when no queries made."""
        adapter = PerformanceAdapter(enable_resilience=False)

        stats = adapter.get_cache_stats()

        assert stats["hit_rate"] == 0.0


# =============================================================================
# Domain Detection Tests
# =============================================================================


class TestDomainDetection:
    """Tests for detect_domain method."""

    def test_detect_security_domain(self):
        """Should detect security keywords."""
        adapter = PerformanceAdapter(enable_resilience=False)

        assert adapter.detect_domain("How to prevent SQL injection attacks?") == "security"

    def test_detect_coding_domain(self):
        """Should detect coding keywords."""
        adapter = PerformanceAdapter(enable_resilience=False)

        assert adapter.detect_domain("Implement a sorting algorithm") == "coding"

    def test_detect_general_domain(self):
        """Should default to general for unrecognized topics."""
        adapter = PerformanceAdapter(enable_resilience=False)

        assert adapter.detect_domain("What is the meaning of life?") == "general"

    def test_get_all_domains(self):
        """Should list all domains with stored expertise."""
        adapter = PerformanceAdapter(enable_resilience=False)
        adapter.store_agent_expertise("claude", "security", 1600, delta=50)
        adapter.store_agent_expertise("gpt4", "coding", 1500, delta=30)

        domains = adapter.get_all_domains()

        assert set(domains) == {"security", "coding"}


# =============================================================================
# KM Pattern Analysis Tests (Reverse Flow)
# =============================================================================


class TestKMPatternAnalysis:
    """Tests for analyze_km_patterns_for_agent."""

    @pytest.mark.asyncio
    async def test_empty_items_returns_no_patterns(self):
        """Should return empty list for no items."""
        adapter = PerformanceAdapter(enable_resilience=False)

        patterns = await adapter.analyze_km_patterns_for_agent("claude", [])

        assert patterns == []

    @pytest.mark.asyncio
    async def test_detect_success_contributor(self):
        """Should detect success_contributor pattern."""
        adapter = PerformanceAdapter(enable_resilience=False)

        items = [{"metadata": {"outcome_success": True, "debate_id": f"d-{i}"}} for i in range(5)]

        patterns = await adapter.analyze_km_patterns_for_agent("claude", items)

        types = [p.pattern_type for p in patterns]
        assert "success_contributor" in types

    @pytest.mark.asyncio
    async def test_detect_contradiction_source(self):
        """Should detect contradiction_source pattern."""
        adapter = PerformanceAdapter(enable_resilience=False)

        # Need at least 3 contradictions and >= 30% rate
        items = [{"metadata": {"was_contradicted": True, "debate_id": f"d-{i}"}} for i in range(5)]

        patterns = await adapter.analyze_km_patterns_for_agent("claude", items)

        types = [p.pattern_type for p in patterns]
        assert "contradiction_source" in types

    @pytest.mark.asyncio
    async def test_detect_domain_expert(self):
        """Should detect domain_expert pattern."""
        adapter = PerformanceAdapter(enable_resilience=False)

        items = [{"metadata": {"domain": "security", "debate_id": f"d-{i}"}} for i in range(6)]

        patterns = await adapter.analyze_km_patterns_for_agent("claude", items)

        types = [p.pattern_type for p in patterns]
        assert "domain_expert" in types

    @pytest.mark.asyncio
    async def test_detect_crux_resolver(self):
        """Should detect crux_resolver pattern."""
        adapter = PerformanceAdapter(enable_resilience=False)

        items = [{"metadata": {"crux_resolved": True, "debate_id": f"d-{i}"}} for i in range(4)]

        patterns = await adapter.analyze_km_patterns_for_agent("claude", items)

        types = [p.pattern_type for p in patterns]
        assert "crux_resolver" in types

    @pytest.mark.asyncio
    async def test_min_confidence_filtering(self):
        """Should filter patterns below min_confidence threshold."""
        adapter = PerformanceAdapter(enable_resilience=False)

        # Only 3 successes out of 5 => rate=0.6, confidence=min(0.95, 0.7)=0.7
        items = [
            {"metadata": {"outcome_success": True, "debate_id": f"d-{i}"}} for i in range(3)
        ] + [{"metadata": {}} for _ in range(2)]

        patterns = await adapter.analyze_km_patterns_for_agent("claude", items, min_confidence=0.95)

        assert len(patterns) == 0

    @pytest.mark.asyncio
    async def test_patterns_stored_on_adapter(self):
        """Should store detected patterns on the adapter."""
        adapter = PerformanceAdapter(enable_resilience=False)

        items = [{"metadata": {"outcome_success": True, "debate_id": f"d-{i}"}} for i in range(5)]

        await adapter.analyze_km_patterns_for_agent("claude", items)

        stored = adapter.get_agent_km_patterns("claude")
        assert len(stored) > 0


# =============================================================================
# ELO Adjustment Computation Tests
# =============================================================================


class TestComputeEloAdjustment:
    """Tests for compute_elo_adjustment method."""

    def test_no_patterns_returns_none(self):
        """Should return None for empty patterns list."""
        adapter = PerformanceAdapter(enable_resilience=False)

        result = adapter.compute_elo_adjustment([])

        assert result is None

    def test_success_contributor_positive_adjustment(self):
        """Should produce positive adjustment for success patterns."""
        adapter = PerformanceAdapter(enable_resilience=False)

        pattern = KMEloPattern(
            agent_name="claude",
            pattern_type="success_contributor",
            confidence=0.9,
            observation_count=10,
        )

        rec = adapter.compute_elo_adjustment([pattern])

        assert rec is not None
        assert rec.adjustment > 0
        assert rec.agent_name == "claude"

    def test_contradiction_source_negative_adjustment(self):
        """Should produce negative adjustment for contradiction patterns."""
        adapter = PerformanceAdapter(enable_resilience=False)

        pattern = KMEloPattern(
            agent_name="claude",
            pattern_type="contradiction_source",
            confidence=0.8,
            observation_count=8,
        )

        rec = adapter.compute_elo_adjustment([pattern])

        assert rec is not None
        assert rec.adjustment < 0

    def test_max_adjustment_clamping(self):
        """Should clamp adjustment to max_adjustment."""
        adapter = PerformanceAdapter(enable_resilience=False)

        # Many strong positive patterns
        patterns = [
            KMEloPattern("claude", "success_contributor", 0.95, observation_count=20),
            KMEloPattern("claude", "crux_resolver", 0.95, observation_count=20),
            KMEloPattern("claude", "domain_expert", 0.95, observation_count=20, domain="security"),
        ]

        rec = adapter.compute_elo_adjustment(patterns, max_adjustment=30.0)

        assert rec is not None
        assert abs(rec.adjustment) <= 30.0

    def test_tiny_adjustment_returns_none(self):
        """Should return None when adjustment < 2.0."""
        adapter = PerformanceAdapter(enable_resilience=False)

        pattern = KMEloPattern(
            agent_name="claude",
            pattern_type="success_contributor",
            confidence=0.1,  # Very low confidence
            observation_count=1,
        )

        rec = adapter.compute_elo_adjustment([pattern])

        assert rec is None

    def test_recommendation_added_to_pending(self):
        """Should add recommendation to pending list."""
        adapter = PerformanceAdapter(enable_resilience=False)

        pattern = KMEloPattern("claude", "success_contributor", 0.9, observation_count=10)
        adapter.compute_elo_adjustment([pattern])

        pending = adapter.get_pending_adjustments()
        assert len(pending) == 1


# =============================================================================
# Search and Record Access Tests
# =============================================================================


class TestSearchAndRecordAccess:
    """Tests for search_by_keyword, get, get_all_records."""

    def test_get_record_by_id_rating(self):
        """Should find rating by ID."""
        adapter = PerformanceAdapter(enable_resilience=False)
        rating = _make_mock_rating()
        rid = adapter.store_rating(rating)

        assert adapter.get(rid) is not None

    def test_search_by_keyword_agent_name(self):
        """Should find records matching agent name."""
        adapter = PerformanceAdapter(enable_resilience=False)
        adapter.store_rating(_make_mock_rating(agent_name="claude"))
        adapter.store_agent_expertise("claude", "security", 1600, delta=50)

        results = adapter.search_by_keyword("claude")

        assert len(results) >= 2

    def test_search_by_keyword_with_record_type(self):
        """Should filter by record_type."""
        adapter = PerformanceAdapter(enable_resilience=False)
        adapter.store_rating(_make_mock_rating())
        adapter.store_agent_expertise("claude", "security", 1600, delta=50)

        results = adapter.search_by_keyword("claude", record_type="expertise")

        assert all(r.get("_match_type") == "expertise" for r in results)

    def test_get_all_records_unfiltered(self):
        """Should return all record types when unfiltered."""
        adapter = PerformanceAdapter(enable_resilience=False)
        adapter.store_rating(_make_mock_rating())
        adapter.store_match(_make_mock_match())
        adapter.store_agent_expertise("claude", "security", 1600, delta=50)

        results = adapter.get_all_records()

        assert len(results) >= 3

    def test_get_all_records_filtered_by_type(self):
        """Should filter by record_type."""
        adapter = PerformanceAdapter(enable_resilience=False)
        adapter.store_rating(_make_mock_rating())
        adapter.store_match(_make_mock_match())

        results = adapter.get_all_records(record_type="match")

        assert len(results) == 1


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for get_stats and reverse flow stats."""

    def test_get_stats_empty(self):
        """Should return zero stats on fresh adapter."""
        adapter = PerformanceAdapter(enable_resilience=False)

        stats = adapter.get_stats()

        assert stats["total_ratings"] == 0
        assert stats["total_matches"] == 0
        assert stats["total_expertise_records"] == 0
        assert stats["km_adjustments_applied"] == 0

    def test_get_stats_after_operations(self):
        """Should reflect stored data counts."""
        adapter = PerformanceAdapter(enable_resilience=False)
        adapter.store_rating(_make_mock_rating())
        adapter.store_match(_make_mock_match())
        adapter.store_agent_expertise("claude", "security", 1600, delta=50)

        stats = adapter.get_stats()

        assert stats["total_ratings"] == 1
        assert stats["total_matches"] == 1
        assert stats["total_expertise_records"] == 1

    def test_get_reverse_flow_stats(self):
        """Should return reverse flow statistics."""
        adapter = PerformanceAdapter(enable_resilience=False)

        stats = adapter.get_reverse_flow_stats()

        assert stats["agents_with_patterns"] == 0
        assert stats["total_patterns"] == 0
        assert stats["pending_adjustments"] == 0
        assert stats["applied_adjustments"] == 0


# =============================================================================
# Backwards Compatibility Tests
# =============================================================================


class TestBackwardsCompatibility:
    """Tests for backward compatibility aliases."""

    def test_elo_adapter_alias(self):
        """EloAdapter should be an alias for PerformanceAdapter."""
        assert EloAdapter is PerformanceAdapter

    def test_ranking_adapter_alias(self):
        """RankingAdapter should be an alias for PerformanceAdapter."""
        assert RankingAdapter is PerformanceAdapter

    def test_elo_adapter_creates_performance_adapter(self):
        """Should be able to instantiate via EloAdapter name."""
        adapter = EloAdapter(enable_resilience=False)

        assert isinstance(adapter, PerformanceAdapter)
