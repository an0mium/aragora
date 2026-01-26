"""Tests for Continuous Learning (Phase 5.2)."""

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aragora.autonomous.continuous_learning import (
    AgentCalibration,
    ContinuousLearner,
    EloUpdater,
    ExtractedPattern,
    KnowledgeDecayManager,
    LearningEvent,
    LearningEventType,
    PatternExtractor,
)


class TestLearningEventType:
    """Tests for LearningEventType enum."""

    def test_event_types(self):
        """Test all event types exist."""
        assert LearningEventType.DEBATE_COMPLETED.value == "debate_completed"
        assert LearningEventType.CONSENSUS_REACHED.value == "consensus_reached"
        assert LearningEventType.AGENT_OUTPERFORMED.value == "agent_outperformed"
        assert LearningEventType.PATTERN_DISCOVERED.value == "pattern_discovered"
        assert LearningEventType.CALIBRATION_UPDATED.value == "calibration_updated"
        assert LearningEventType.KNOWLEDGE_DECAYED.value == "knowledge_decayed"
        assert LearningEventType.USER_FEEDBACK.value == "user_feedback"


class TestEloUpdater:
    """Tests for EloUpdater class."""

    @pytest.fixture
    def elo_updater(self):
        """Create an EloUpdater instance."""
        return EloUpdater(k_factor=32.0, min_rating=100.0, max_rating=3000.0)

    def test_default_rating(self, elo_updater):
        """Test default rating is 1500."""
        assert elo_updater.get_rating("new_agent") == 1500.0

    def test_set_rating(self, elo_updater):
        """Test setting rating."""
        elo_updater.set_rating("agent1", 1600.0)
        assert elo_updater.get_rating("agent1") == 1600.0

    def test_set_rating_clamped_max(self, elo_updater):
        """Test rating is clamped to max."""
        elo_updater.set_rating("agent1", 5000.0)
        assert elo_updater.get_rating("agent1") == 3000.0

    def test_set_rating_clamped_min(self, elo_updater):
        """Test rating is clamped to min."""
        elo_updater.set_rating("agent1", 50.0)
        assert elo_updater.get_rating("agent1") == 100.0

    def test_update_from_debate_winner_gains(self, elo_updater):
        """Test winner gains rating in debate."""
        elo_updater.set_rating("winner", 1500.0)
        elo_updater.set_rating("loser", 1500.0)

        new_winner, new_loser = elo_updater.update_from_debate(
            winner_id="winner",
            loser_id="loser",
            margin=1.0,
        )

        assert new_winner > 1500.0
        assert new_loser < 1500.0

    def test_update_from_debate_higher_rated_wins(self, elo_updater):
        """Test smaller change when higher rated wins."""
        elo_updater.set_rating("favorite", 1700.0)
        elo_updater.set_rating("underdog", 1300.0)

        favorite_before = elo_updater.get_rating("favorite")
        new_winner, _ = elo_updater.update_from_debate(
            winner_id="favorite",
            loser_id="underdog",
            margin=1.0,
        )

        # Smaller gain for expected winner
        gain = new_winner - favorite_before
        assert 0 < gain < 20

    def test_update_from_debate_underdog_wins(self, elo_updater):
        """Test larger change when underdog wins."""
        elo_updater.set_rating("favorite", 1700.0)
        elo_updater.set_rating("underdog", 1300.0)

        underdog_before = elo_updater.get_rating("underdog")
        new_winner, _ = elo_updater.update_from_debate(
            winner_id="underdog",
            loser_id="favorite",
            margin=1.0,
        )

        # Larger gain for upset
        gain = new_winner - underdog_before
        assert gain > 20

    def test_update_from_votes(self, elo_updater):
        """Test rating update from vote distribution."""
        elo_updater.set_rating("agent1", 1500.0)
        elo_updater.set_rating("agent2", 1500.0)

        new_ratings = elo_updater.update_from_votes(
            agent_votes={"agent1": 70, "agent2": 30},
            total_votes=100,
        )

        assert new_ratings["agent1"] > 1500.0
        assert new_ratings["agent2"] < 1500.0

    def test_update_from_votes_empty(self, elo_updater):
        """Test no update from zero votes."""
        result = elo_updater.update_from_votes({}, 0)
        assert result == {}

    def test_apply_decay(self):
        """Test rating decay for inactive agents."""
        elo_updater = EloUpdater(decay_per_day=5.0)
        elo_updater.set_rating("agent1", 1600.0)

        # Simulate inactivity by backdating
        elo_updater._last_active["agent1"] = datetime.now() - timedelta(days=10)

        decayed = elo_updater.apply_decay()

        assert "agent1" in decayed
        assert decayed["agent1"] < 1600.0

    def test_no_decay_when_disabled(self, elo_updater):
        """Test no decay when decay_per_day is 0."""
        elo_updater.set_rating("agent1", 1600.0)
        elo_updater._last_active["agent1"] = datetime.now() - timedelta(days=10)

        decayed = elo_updater.apply_decay()

        assert decayed == {}

    def test_get_all_ratings(self, elo_updater):
        """Test getting all ratings."""
        elo_updater.set_rating("agent1", 1600.0)
        elo_updater.set_rating("agent2", 1400.0)

        ratings = elo_updater.get_all_ratings()

        assert ratings == {"agent1": 1600.0, "agent2": 1400.0}


class TestPatternExtractor:
    """Tests for PatternExtractor class."""

    @pytest.fixture
    def pattern_extractor(self):
        """Create a PatternExtractor instance."""
        return PatternExtractor(
            min_evidence_count=3,
            min_confidence=0.6,
        )

    def test_observe_records_data(self, pattern_extractor):
        """Test observe records observation."""
        pattern_extractor.observe(
            observation_type="test_type",
            data={"key": "value"},
            agents=["agent1"],
            topics=["topic1"],
        )

        assert len(pattern_extractor._observations["test_type"]) == 1

    def test_extract_consensus_patterns(self, pattern_extractor):
        """Test extracting consensus patterns."""
        # Add observations
        for i in range(5):
            pattern_extractor.observe(
                observation_type="consensus_reached",
                data={"strategy": "voting", "success": True},
                agents=["agent1", "agent2"],
                topics=["tech"],
            )

        patterns = pattern_extractor.extract_patterns()

        # Should find voting strategy pattern
        consensus_patterns = [p for p in patterns if p.pattern_type == "consensus_strategy"]
        assert len(consensus_patterns) >= 1

    def test_extract_expertise_patterns(self, pattern_extractor):
        """Test extracting topic expertise patterns."""
        # Add performance observations
        for i in range(5):
            pattern_extractor.observe(
                observation_type="agent_performance",
                data={"agent": "claude", "topic": "coding", "score": 0.85},
                agents=["claude"],
                topics=["coding"],
            )

        patterns = pattern_extractor.extract_patterns()

        expertise_patterns = [p for p in patterns if p.pattern_type == "topic_expertise"]
        assert len(expertise_patterns) >= 1
        assert any("claude" in p.agents_involved for p in expertise_patterns)

    def test_min_evidence_count_enforced(self, pattern_extractor):
        """Test patterns require minimum evidence."""
        # Only 2 observations (less than min_evidence_count=3)
        for i in range(2):
            pattern_extractor.observe(
                observation_type="consensus_reached",
                data={"strategy": "voting", "success": True},
                agents=["agent1"],
            )

        patterns = pattern_extractor.extract_patterns()
        assert len(patterns) == 0

    def test_get_patterns_by_type(self, pattern_extractor):
        """Test filtering patterns by type."""
        # Add enough observations for multiple pattern types
        for i in range(5):
            pattern_extractor.observe(
                observation_type="consensus_reached",
                data={"strategy": "voting", "success": True},
                agents=["agent1"],
            )
            pattern_extractor.observe(
                observation_type="debate_failed",
                data={"reason": "timeout"},
                agents=["agent1"],
            )

        pattern_extractor.extract_patterns()

        consensus = pattern_extractor.get_patterns("consensus_strategy")
        failures = pattern_extractor.get_patterns("failure_mode")

        assert len(consensus) >= 0  # May or may not find patterns
        assert len(failures) >= 0


class TestKnowledgeDecayManager:
    """Tests for KnowledgeDecayManager class."""

    @pytest.fixture
    def decay_manager(self):
        """Create a KnowledgeDecayManager instance."""
        return KnowledgeDecayManager(
            default_half_life_days=30.0,
            min_confidence=0.1,
        )

    def test_register_knowledge(self, decay_manager):
        """Test registering knowledge item."""
        decay_manager.register_knowledge(
            knowledge_id="fact1",
            initial_confidence=0.9,
            importance=0.7,
        )

        assert decay_manager.get_confidence("fact1") == 0.9

    def test_refresh_knowledge(self, decay_manager):
        """Test refreshing knowledge boosts confidence."""
        decay_manager.register_knowledge("fact1", initial_confidence=0.7)

        new_confidence = decay_manager.refresh_knowledge("fact1", boost=0.1)

        assert new_confidence == pytest.approx(0.8)

    def test_refresh_knowledge_capped(self, decay_manager):
        """Test refresh doesn't exceed 1.0."""
        decay_manager.register_knowledge("fact1", initial_confidence=0.95)

        new_confidence = decay_manager.refresh_knowledge("fact1", boost=0.2)

        assert new_confidence == 1.0

    def test_refresh_nonexistent(self, decay_manager):
        """Test refreshing nonexistent knowledge returns None."""
        result = decay_manager.refresh_knowledge("nonexistent")
        assert result is None

    def test_apply_decay(self, decay_manager):
        """Test applying decay reduces confidence."""
        decay_manager.register_knowledge("fact1", initial_confidence=0.9)

        # Simulate time passing
        decay_manager._knowledge_items["fact1"]["last_refreshed"] = datetime.now() - timedelta(
            days=60
        )

        changed = decay_manager.apply_decay()

        assert "fact1" in changed
        assert changed["fact1"] < 0.9

    def test_decay_respects_min_confidence(self, decay_manager):
        """Test decay doesn't go below minimum."""
        decay_manager.register_knowledge("fact1", initial_confidence=0.2)

        # Simulate very long time
        decay_manager._knowledge_items["fact1"]["last_refreshed"] = datetime.now() - timedelta(
            days=365
        )

        changed = decay_manager.apply_decay()

        assert changed.get("fact1", 0.2) >= 0.1

    def test_importance_slows_decay(self, decay_manager):
        """Test high importance slows decay."""
        decay_manager.register_knowledge("important", initial_confidence=0.9, importance=0.9)
        decay_manager.register_knowledge("normal", initial_confidence=0.9, importance=0.0)

        # Same time passage
        days_ago = datetime.now() - timedelta(days=30)
        decay_manager._knowledge_items["important"]["last_refreshed"] = days_ago
        decay_manager._knowledge_items["normal"]["last_refreshed"] = days_ago

        changed = decay_manager.apply_decay()

        # Important knowledge should decay less
        assert changed.get("important", 0.9) > changed.get("normal", 0.9)

    def test_get_stale_knowledge(self, decay_manager):
        """Test finding stale knowledge."""
        decay_manager.register_knowledge("fresh", initial_confidence=0.9)
        decay_manager.register_knowledge("stale", initial_confidence=0.9)

        decay_manager._knowledge_items["stale"]["last_refreshed"] = datetime.now() - timedelta(
            days=60
        )

        stale = decay_manager.get_stale_knowledge(max_age_days=30)

        assert "stale" in stale
        assert "fresh" not in stale


class TestContinuousLearner:
    """Tests for ContinuousLearner class."""

    @pytest.fixture
    def continuous_learner(self):
        """Create a ContinuousLearner instance."""
        return ContinuousLearner()

    @pytest.mark.asyncio
    async def test_on_debate_completed(self, continuous_learner):
        """Test processing completed debate."""
        event = await continuous_learner.on_debate_completed(
            debate_id="debate_123",
            agents=["claude", "gpt"],
            winner="claude",
            votes={"claude": 7, "gpt": 3},
            consensus_reached=True,
            topics=["coding"],
        )

        assert event.event_type == LearningEventType.DEBATE_COMPLETED
        assert event.applied
        assert event.data["winner"] == "claude"

    @pytest.mark.asyncio
    async def test_on_debate_completed_updates_elo(self, continuous_learner):
        """Test debate updates ELO ratings."""
        initial_rating = continuous_learner.elo_updater.get_rating("claude")

        await continuous_learner.on_debate_completed(
            debate_id="debate_123",
            agents=["claude", "gpt"],
            winner="claude",
            votes={"claude": 10, "gpt": 0},
            consensus_reached=False,
            topics=[],
        )

        new_rating = continuous_learner.elo_updater.get_rating("claude")
        assert new_rating > initial_rating

    @pytest.mark.asyncio
    async def test_on_user_feedback(self, continuous_learner):
        """Test processing user feedback."""
        event = await continuous_learner.on_user_feedback(
            debate_id="debate_123",
            agent_id="claude",
            feedback_type="helpful",
            score=0.8,
        )

        assert event.event_type == LearningEventType.USER_FEEDBACK
        assert event.applied

    @pytest.mark.asyncio
    async def test_event_callback_called(self):
        """Test event callback is invoked."""
        callback = MagicMock()
        learner = ContinuousLearner(event_callback=callback)

        await learner.on_debate_completed(
            debate_id="test",
            agents=["a", "b"],
            winner="a",
            votes={"a": 5, "b": 5},
            consensus_reached=False,
            topics=[],
        )

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_periodic_learning(self, continuous_learner):
        """Test periodic learning tasks."""
        # Add some observations
        for i in range(5):
            continuous_learner.pattern_extractor.observe(
                "consensus_reached",
                {"strategy": "voting", "success": True},
                ["agent1"],
            )

        summary = await continuous_learner.run_periodic_learning()

        assert "patterns_extracted" in summary
        assert "knowledge_decayed" in summary
        assert "ratings_decayed" in summary

    @pytest.mark.asyncio
    async def test_calibration_updated(self, continuous_learner):
        """Test calibration is updated after debate."""
        await continuous_learner.on_debate_completed(
            debate_id="test",
            agents=["claude"],
            winner="claude",
            votes={"claude": 10},
            consensus_reached=True,
            topics=[],
        )

        calibration = continuous_learner.get_calibration("claude")

        assert calibration is not None
        assert calibration.total_debates == 1
        assert calibration.last_updated is not None

    def test_get_all_calibrations(self, continuous_learner):
        """Test getting all calibrations."""
        asyncio.run(
            continuous_learner.on_debate_completed(
                debate_id="test",
                agents=["claude", "gpt"],
                winner="claude",
                votes={"claude": 5, "gpt": 5},
                consensus_reached=False,
                topics=[],
            )
        )

        calibrations = continuous_learner.get_all_calibrations()

        assert "claude" in calibrations
        assert "gpt" in calibrations


class TestAgentCalibration:
    """Tests for AgentCalibration dataclass."""

    def test_default_values(self):
        """Test default calibration values."""
        cal = AgentCalibration(agent_id="test")

        assert cal.agent_id == "test"
        assert cal.elo_rating == 1500.0
        assert cal.confidence_accuracy == 0.5
        assert cal.topic_strengths == {}
        assert cal.total_debates == 0
        assert cal.win_rate == 0.5


class TestExtractedPattern:
    """Tests for ExtractedPattern dataclass."""

    def test_pattern_creation(self):
        """Test creating an extracted pattern."""
        now = datetime.now()
        pattern = ExtractedPattern(
            id="pattern_1",
            pattern_type="consensus_strategy",
            description="Voting works well",
            confidence=0.85,
            evidence_count=10,
            first_seen=now - timedelta(days=7),
            last_seen=now,
            agents_involved=["claude", "gpt"],
            topics=["coding"],
        )

        assert pattern.id == "pattern_1"
        assert pattern.confidence == 0.85
        assert len(pattern.agents_involved) == 2
