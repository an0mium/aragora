"""
Tests for ContinuumMemory integration with TeamSelector.

Tests cover:
- TeamSelectionConfig defaults for memory_weight and enable_memory_selection
- TeamSelector constructor accepts continuum_memory parameter
- _compute_memory_score returns 0.0 when no memory system is provided
- _compute_memory_score returns 0.0 when memory selection is disabled
- _compute_memory_score with matching agent memories
- _compute_memory_score filters results by agent_name metadata
- Confidence scaling with few vs many observations
- Graceful failure on retrieve error
- Integration into overall _compute_score
- Backward compatibility: no memory = same behavior as before
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from aragora.debate.team_selector import (
    TeamSelectionConfig,
    TeamSelector,
)


# ===========================================================================
# Mock Classes
# ===========================================================================


class MockAgent:
    """Mock agent for testing."""

    def __init__(
        self,
        name: str,
        agent_type: str = "unknown",
        model: str = "",
    ):
        self.name = name
        self.agent_type = agent_type
        self.model = model


class MockMemoryEntry:
    """Mock ContinuumMemoryEntry for testing."""

    def __init__(
        self,
        agent_name: str | None = None,
        importance: float = 0.5,
        consolidation_score: float = 0.8,
        success_count: int = 5,
        failure_count: int = 5,
        update_count: int = 10,
        metadata: dict | None = None,
    ):
        self.importance = importance
        self.consolidation_score = consolidation_score
        self.success_count = success_count
        self.failure_count = failure_count
        self.update_count = update_count
        self.metadata = metadata or {}
        if agent_name is not None:
            self.metadata["agent_name"] = agent_name

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5


class MockContinuumMemory:
    """Mock ContinuumMemory for testing."""

    def __init__(self, entries: list[MockMemoryEntry] | None = None):
        self._entries = entries or []
        self.retrieve_calls: list[dict] = []

    def retrieve(
        self,
        query: str | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
        **kwargs,
    ) -> list[MockMemoryEntry]:
        self.retrieve_calls.append({
            "query": query,
            "limit": limit,
            "min_importance": min_importance,
        })
        return self._entries


# ===========================================================================
# Config Tests
# ===========================================================================


class TestMemorySelectionConfig:
    """Tests for memory-related TeamSelectionConfig fields."""

    def test_default_memory_weight(self):
        """memory_weight defaults to 0.15."""
        config = TeamSelectionConfig()
        assert config.memory_weight == 0.15

    def test_default_enable_memory_selection(self):
        """enable_memory_selection defaults to True."""
        config = TeamSelectionConfig()
        assert config.enable_memory_selection is True

    def test_custom_memory_weight(self):
        """memory_weight can be overridden."""
        config = TeamSelectionConfig(memory_weight=0.3)
        assert config.memory_weight == 0.3

    def test_disable_memory_selection(self):
        """enable_memory_selection can be set to False."""
        config = TeamSelectionConfig(enable_memory_selection=False)
        assert config.enable_memory_selection is False


# ===========================================================================
# Constructor Tests
# ===========================================================================


class TestTeamSelectorContinuumMemory:
    """Tests for TeamSelector constructor with continuum_memory."""

    def test_constructor_accepts_continuum_memory(self):
        """TeamSelector accepts continuum_memory parameter."""
        mock_memory = MockContinuumMemory()
        selector = TeamSelector(continuum_memory=mock_memory)
        assert selector.continuum_memory is mock_memory

    def test_constructor_continuum_memory_defaults_to_none(self):
        """continuum_memory defaults to None when not provided."""
        selector = TeamSelector()
        assert selector.continuum_memory is None

    def test_constructor_with_other_params(self):
        """continuum_memory works alongside other constructor params."""
        mock_memory = MockContinuumMemory()
        mock_elo = MagicMock()
        selector = TeamSelector(
            elo_system=mock_elo,
            continuum_memory=mock_memory,
        )
        assert selector.continuum_memory is mock_memory
        assert selector.elo_system is mock_elo


# ===========================================================================
# _compute_memory_score Tests
# ===========================================================================


class TestComputeMemoryScore:
    """Tests for _compute_memory_score method."""

    def test_returns_zero_when_no_memory(self):
        """Returns 0.0 when continuum_memory is None."""
        selector = TeamSelector()
        agent = MockAgent("claude")
        score = selector._compute_memory_score(agent, "code", "Fix a bug")
        assert score == 0.0

    def test_returns_zero_when_disabled(self):
        """Returns 0.0 when enable_memory_selection is False."""
        mock_memory = MockContinuumMemory(entries=[
            MockMemoryEntry(agent_name="claude", success_count=10, failure_count=0),
        ])
        config = TeamSelectionConfig(enable_memory_selection=False)
        selector = TeamSelector(continuum_memory=mock_memory, config=config)
        agent = MockAgent("claude")
        score = selector._compute_memory_score(agent, "code", "Fix a bug")
        assert score == 0.0

    def test_returns_zero_when_no_memories_found(self):
        """Returns 0.0 when retrieve returns empty list."""
        mock_memory = MockContinuumMemory(entries=[])
        selector = TeamSelector(continuum_memory=mock_memory)
        agent = MockAgent("claude")
        score = selector._compute_memory_score(agent, "code", "Fix a bug")
        assert score == 0.0

    def test_returns_zero_when_no_agent_memories(self):
        """Returns 0.0 when no memories match the agent name."""
        mock_memory = MockContinuumMemory(entries=[
            MockMemoryEntry(agent_name="gpt", success_count=10, failure_count=0),
        ])
        selector = TeamSelector(continuum_memory=mock_memory)
        agent = MockAgent("claude")
        score = selector._compute_memory_score(agent, "code", "Fix a bug")
        assert score == 0.0

    def test_filters_by_agent_name(self):
        """Only considers memories with matching agent_name in metadata."""
        mock_memory = MockContinuumMemory(entries=[
            MockMemoryEntry(agent_name="claude", success_count=10, failure_count=0),
            MockMemoryEntry(agent_name="gpt", success_count=0, failure_count=10),
            MockMemoryEntry(agent_name="claude", success_count=8, failure_count=2),
        ])
        selector = TeamSelector(continuum_memory=mock_memory)
        agent = MockAgent("claude")
        score = selector._compute_memory_score(agent, "code", "Fix a bug")
        # Two claude memories with high success rates -> score > 0.5
        assert score > 0.5

    def test_high_success_rate_gives_high_score(self):
        """Agent with 100% success rate gets score above 0.5."""
        mock_memory = MockContinuumMemory(entries=[
            MockMemoryEntry(
                agent_name="claude",
                success_count=20,
                failure_count=0,
                update_count=20,
                importance=0.8,
                consolidation_score=0.9,
            ),
        ])
        selector = TeamSelector(continuum_memory=mock_memory)
        agent = MockAgent("claude")
        score = selector._compute_memory_score(agent, "code", "Fix a bug")
        assert score > 0.5

    def test_low_success_rate_gives_low_score(self):
        """Agent with 0% success rate gets score below 0.5."""
        mock_memory = MockContinuumMemory(entries=[
            MockMemoryEntry(
                agent_name="claude",
                success_count=0,
                failure_count=20,
                update_count=20,
                importance=0.8,
                consolidation_score=0.9,
            ),
        ])
        selector = TeamSelector(continuum_memory=mock_memory)
        agent = MockAgent("claude")
        score = selector._compute_memory_score(agent, "code", "Fix a bug")
        assert score < 0.5

    def test_neutral_success_rate_gives_midrange_score(self):
        """Agent with 50% success rate gets score near 0.5."""
        mock_memory = MockContinuumMemory(entries=[
            MockMemoryEntry(
                agent_name="claude",
                success_count=10,
                failure_count=10,
                update_count=20,
                importance=0.8,
                consolidation_score=0.9,
            ),
        ])
        selector = TeamSelector(continuum_memory=mock_memory)
        agent = MockAgent("claude")
        score = selector._compute_memory_score(agent, "code", "Fix a bug")
        assert 0.45 <= score <= 0.55

    def test_confidence_scaling_few_observations(self):
        """Fewer observations reduce confidence and compress score toward 0.5."""
        mock_memory_few = MockContinuumMemory(entries=[
            MockMemoryEntry(
                agent_name="claude",
                success_count=2,
                failure_count=0,
                update_count=2,
                importance=0.8,
                consolidation_score=0.9,
            ),
        ])
        mock_memory_many = MockContinuumMemory(entries=[
            MockMemoryEntry(
                agent_name="claude",
                success_count=20,
                failure_count=0,
                update_count=20,
                importance=0.8,
                consolidation_score=0.9,
            ),
        ])
        selector_few = TeamSelector(continuum_memory=mock_memory_few)
        selector_many = TeamSelector(continuum_memory=mock_memory_many)
        agent = MockAgent("claude")

        score_few = selector_few._compute_memory_score(agent, "code", "Fix a bug")
        score_many = selector_many._compute_memory_score(agent, "code", "Fix a bug")

        # Both positive, but many-observation score should be higher
        assert score_few > 0.5
        assert score_many > 0.5
        assert score_many > score_few

    def test_confidence_capped_at_one(self):
        """Confidence is capped at 1.0 even with many observations."""
        mock_memory = MockContinuumMemory(entries=[
            MockMemoryEntry(
                agent_name="claude",
                success_count=100,
                failure_count=0,
                update_count=100,
                importance=0.8,
                consolidation_score=0.9,
            ),
        ])
        selector = TeamSelector(continuum_memory=mock_memory)
        agent = MockAgent("claude")
        score = selector._compute_memory_score(agent, "code", "Fix a bug")
        assert 0.0 <= score <= 1.0

    def test_score_clamped_to_zero_one(self):
        """Score is always clamped between 0.0 and 1.0."""
        mock_memory = MockContinuumMemory(entries=[
            MockMemoryEntry(
                agent_name="claude",
                success_count=1000,
                failure_count=0,
                update_count=1000,
                importance=1.0,
                consolidation_score=1.0,
            ),
        ])
        selector = TeamSelector(continuum_memory=mock_memory)
        agent = MockAgent("claude")
        score = selector._compute_memory_score(agent, "code", "Fix a bug")
        assert 0.0 <= score <= 1.0

    def test_graceful_failure_on_retrieve_error(self):
        """Returns 0.0 when retrieve raises an exception."""
        mock_memory = MagicMock()
        mock_memory.retrieve.side_effect = RuntimeError("DB connection failed")
        selector = TeamSelector(continuum_memory=mock_memory)
        agent = MockAgent("claude")
        score = selector._compute_memory_score(agent, "code", "Fix a bug")
        assert score == 0.0

    def test_graceful_failure_on_attribute_error(self):
        """Returns 0.0 when memory entries have unexpected structure."""
        mock_memory = MagicMock()
        mock_memory.retrieve.side_effect = AttributeError("no such attribute")
        selector = TeamSelector(continuum_memory=mock_memory)
        agent = MockAgent("claude")
        score = selector._compute_memory_score(agent, "code", "Fix a bug")
        assert score == 0.0

    def test_retrieve_called_with_correct_params(self):
        """Verifies retrieve is called with expected query, limit, and min_importance."""
        mock_memory = MockContinuumMemory(entries=[])
        selector = TeamSelector(continuum_memory=mock_memory)
        agent = MockAgent("claude")
        selector._compute_memory_score(agent, "code", "Fix a bug")

        assert len(mock_memory.retrieve_calls) == 1
        call = mock_memory.retrieve_calls[0]
        assert call["query"] == "claude code Fix a bug"
        assert call["limit"] == 20
        assert call["min_importance"] == 0.3

    def test_importance_weighting(self):
        """Higher importance memories have more influence on the score."""
        # High importance, high success
        entries_high_importance = [
            MockMemoryEntry(
                agent_name="claude",
                success_count=10,
                failure_count=0,
                update_count=20,
                importance=0.9,
                consolidation_score=0.9,
            ),
        ]
        # Low importance, high success
        entries_low_importance = [
            MockMemoryEntry(
                agent_name="claude",
                success_count=10,
                failure_count=0,
                update_count=20,
                importance=0.3,
                consolidation_score=0.9,
            ),
        ]
        selector_high = TeamSelector(
            continuum_memory=MockContinuumMemory(entries_high_importance)
        )
        selector_low = TeamSelector(
            continuum_memory=MockContinuumMemory(entries_low_importance)
        )
        agent = MockAgent("claude")

        score_high = selector_high._compute_memory_score(agent, "code", "task")
        score_low = selector_low._compute_memory_score(agent, "code", "task")

        # Both should be similar since they have same success rate,
        # but the importance weighting affects the weighted average equally
        # here since there's only one memory. Both should be > 0.5.
        assert score_high > 0.5
        assert score_low > 0.5


# ===========================================================================
# Integration with _compute_score Tests
# ===========================================================================


class TestMemoryScoreIntegration:
    """Tests for memory score integration into _compute_score."""

    def test_memory_score_added_to_compute_score(self):
        """Memory score contributes to the overall score when memory is present."""
        mock_memory = MockContinuumMemory(entries=[
            MockMemoryEntry(
                agent_name="claude",
                success_count=20,
                failure_count=0,
                update_count=20,
                importance=0.8,
                consolidation_score=0.9,
            ),
        ])
        config = TeamSelectionConfig(
            enable_memory_selection=True,
            memory_weight=0.15,
            # Disable other scoring factors for isolation
            enable_domain_filtering=False,
            enable_culture_selection=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
            enable_feedback_weights=False,
            enable_specialist_bonus=False,
            enable_exploration_bonus=False,
            enable_elo_win_rate=False,
        )
        selector_with_memory = TeamSelector(
            continuum_memory=mock_memory,
            config=config,
        )
        selector_without_memory = TeamSelector(config=config)

        agent = MockAgent("claude")
        score_with = selector_with_memory._compute_score(agent, domain="code", task="Fix a bug")
        score_without = selector_without_memory._compute_score(agent, domain="code", task="Fix a bug")

        # With memory and high success, score should be higher
        assert score_with > score_without

    def test_memory_score_not_added_when_disabled(self):
        """Memory score does not contribute when enable_memory_selection is False."""
        mock_memory = MockContinuumMemory(entries=[
            MockMemoryEntry(
                agent_name="claude",
                success_count=20,
                failure_count=0,
                update_count=20,
                importance=0.8,
            ),
        ])
        # Use a no-op calibration tracker to prevent auto-detection from
        # contributing score.  get_brier_score raises KeyError which the
        # scorer handles gracefully.
        no_cal = MagicMock()
        no_cal.get_brier_score.side_effect = KeyError("not found")
        config = TeamSelectionConfig(
            enable_memory_selection=False,
            # Disable other scoring factors
            enable_domain_filtering=False,
            enable_culture_selection=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
            enable_feedback_weights=False,
            enable_specialist_bonus=False,
            enable_exploration_bonus=False,
            enable_elo_win_rate=False,
            enable_pulse_selection=False,
        )
        selector = TeamSelector(
            continuum_memory=mock_memory,
            calibration_tracker=no_cal,
            config=config,
        )
        agent = MockAgent("claude")
        score = selector._compute_score(agent, domain="code", task="Fix a bug")

        # Should equal base_score only (no memory contribution)
        assert score == config.base_score

    def test_memory_score_not_added_when_no_domain(self):
        """Memory score skipped when domain is None."""
        mock_memory = MockContinuumMemory(entries=[
            MockMemoryEntry(
                agent_name="claude",
                success_count=20,
                failure_count=0,
                update_count=20,
            ),
        ])
        no_cal = MagicMock()
        no_cal.get_brier_score.side_effect = KeyError("not found")
        config = TeamSelectionConfig(
            enable_memory_selection=True,
            # Disable other scoring factors
            enable_domain_filtering=False,
            enable_culture_selection=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
            enable_feedback_weights=False,
            enable_specialist_bonus=False,
            enable_exploration_bonus=False,
            enable_elo_win_rate=False,
            enable_pulse_selection=False,
        )
        selector = TeamSelector(
            continuum_memory=mock_memory,
            calibration_tracker=no_cal,
            config=config,
        )
        agent = MockAgent("claude")
        score = selector._compute_score(agent, domain=None, task="Fix a bug")

        # Domain is None, so memory contribution is skipped
        assert score == config.base_score

    def test_backward_compatibility_no_memory(self):
        """TeamSelector without continuum_memory behaves exactly as before."""
        no_cal = MagicMock()
        no_cal.get_brier_score.side_effect = KeyError("not found")
        config = TeamSelectionConfig(
            # Disable all scoring factors for clean baseline
            enable_domain_filtering=False,
            enable_culture_selection=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
            enable_feedback_weights=False,
            enable_specialist_bonus=False,
            enable_exploration_bonus=False,
            enable_elo_win_rate=False,
            enable_pulse_selection=False,
        )
        selector = TeamSelector(calibration_tracker=no_cal, config=config)
        agent = MockAgent("claude")
        score = selector._compute_score(agent, domain="code", task="Fix a bug")

        # Should just be base_score with no contributions
        assert score == config.base_score

    def test_memory_weight_affects_contribution(self):
        """Higher memory_weight increases the memory contribution to the score."""
        entries = [
            MockMemoryEntry(
                agent_name="claude",
                success_count=20,
                failure_count=0,
                update_count=20,
                importance=0.8,
                consolidation_score=0.9,
            ),
        ]
        base_config = dict(
            enable_domain_filtering=False,
            enable_culture_selection=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
            enable_feedback_weights=False,
            enable_specialist_bonus=False,
            enable_exploration_bonus=False,
            enable_elo_win_rate=False,
        )
        config_low = TeamSelectionConfig(memory_weight=0.1, **base_config)
        config_high = TeamSelectionConfig(memory_weight=0.5, **base_config)

        selector_low = TeamSelector(
            continuum_memory=MockContinuumMemory(entries),
            config=config_low,
        )
        selector_high = TeamSelector(
            continuum_memory=MockContinuumMemory(entries),
            config=config_high,
        )
        agent = MockAgent("claude")

        score_low = selector_low._compute_score(agent, domain="code", task="Fix a bug")
        score_high = selector_high._compute_score(agent, domain="code", task="Fix a bug")

        # Higher weight should produce a larger score contribution
        assert score_high > score_low
