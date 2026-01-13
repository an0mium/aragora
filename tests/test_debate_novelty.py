"""
Tests for Novelty Tracker module.

Tests cover:
- NoveltyScore dataclass operations
- NoveltyResult dataclass operations
- NoveltyTracker initialization
- First round novelty (maximally novel)
- Subsequent round novelty computation
- Low novelty detection
- History management
- Agent trajectories and debate summaries
"""

import pytest
from unittest.mock import Mock, patch

from aragora.debate.novelty import (
    NoveltyScore,
    NoveltyResult,
    NoveltyTracker,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_backend():
    """Create a mock similarity backend for predictable tests."""
    backend = Mock()
    backend.compute_similarity = Mock(return_value=0.5)
    return backend


@pytest.fixture
def tracker(mock_backend):
    """Create a novelty tracker with mock backend."""
    return NoveltyTracker(backend=mock_backend, low_novelty_threshold=0.15)


@pytest.fixture
def sample_proposals():
    """Sample proposals for testing."""
    return {
        "claude": "We should implement feature X using pattern A",
        "gpt4": "I propose using pattern B for feature X",
        "gemini": "Feature X could benefit from pattern C",
    }


# ============================================================================
# NoveltyScore Tests
# ============================================================================


class TestNoveltyScore:
    """Tests for NoveltyScore dataclass."""

    def test_create_score(self):
        """Test creating a novelty score."""
        score = NoveltyScore(
            agent="claude",
            round_num=2,
            novelty=0.75,
            max_similarity=0.25,
            most_similar_to="gpt4",
            prior_proposals_count=3,
        )

        assert score.agent == "claude"
        assert score.novelty == 0.75
        assert score.max_similarity == 0.25
        assert score.most_similar_to == "gpt4"

    def test_is_low_novelty_below_threshold(self):
        """Test is_low_novelty with score below threshold."""
        score = NoveltyScore(
            agent="test",
            round_num=1,
            novelty=0.10,  # Below default 0.15 threshold
            max_similarity=0.90,
        )

        assert score.is_low_novelty() is True
        assert score.is_low_novelty(threshold=0.15) is True

    def test_is_low_novelty_above_threshold(self):
        """Test is_low_novelty with score above threshold."""
        score = NoveltyScore(
            agent="test",
            round_num=1,
            novelty=0.50,
            max_similarity=0.50,
        )

        assert score.is_low_novelty() is False
        assert score.is_low_novelty(threshold=0.15) is False

    def test_is_low_novelty_custom_threshold(self):
        """Test is_low_novelty with custom threshold."""
        score = NoveltyScore(
            agent="test",
            round_num=1,
            novelty=0.30,
            max_similarity=0.70,
        )

        assert score.is_low_novelty(threshold=0.25) is False
        assert score.is_low_novelty(threshold=0.50) is True


# ============================================================================
# NoveltyResult Tests
# ============================================================================


class TestNoveltyResult:
    """Tests for NoveltyResult dataclass."""

    def test_create_result(self):
        """Test creating a novelty result."""
        result = NoveltyResult(
            round_num=3,
            per_agent_novelty={"claude": 0.8, "gpt4": 0.6},
            avg_novelty=0.7,
            min_novelty=0.6,
            max_novelty=0.8,
        )

        assert result.round_num == 3
        assert result.avg_novelty == 0.7
        assert len(result.per_agent_novelty) == 2

    def test_has_low_novelty_with_low_agents(self):
        """Test has_low_novelty when agents are flagged."""
        result = NoveltyResult(
            round_num=2,
            low_novelty_agents=["claude", "gpt4"],
        )

        assert result.has_low_novelty() is True

    def test_has_low_novelty_without_low_agents(self):
        """Test has_low_novelty when no agents flagged."""
        result = NoveltyResult(
            round_num=2,
            low_novelty_agents=[],
        )

        assert result.has_low_novelty() is False


# ============================================================================
# NoveltyTracker Initialization Tests
# ============================================================================


class TestNoveltyTrackerInit:
    """Tests for NoveltyTracker initialization."""

    def test_initialization_with_backend(self, mock_backend):
        """Test initialization with explicit backend."""
        tracker = NoveltyTracker(backend=mock_backend)

        assert tracker.backend == mock_backend
        assert tracker.low_novelty_threshold == 0.15
        assert len(tracker.history) == 0
        assert len(tracker.scores) == 0

    def test_initialization_custom_threshold(self, mock_backend):
        """Test initialization with custom threshold."""
        tracker = NoveltyTracker(
            backend=mock_backend,
            low_novelty_threshold=0.25,
        )

        assert tracker.low_novelty_threshold == 0.25

    def test_initialization_auto_backend(self):
        """Test initialization with auto backend selection."""
        tracker = NoveltyTracker()

        # Should have some backend (auto-selected)
        assert tracker.backend is not None


# ============================================================================
# First Round Novelty Tests
# ============================================================================


class TestFirstRoundNovelty:
    """Tests for first round novelty computation."""

    def test_first_round_maximally_novel(self, tracker, sample_proposals):
        """Test that first round proposals are maximally novel."""
        result = tracker.compute_novelty(sample_proposals, round_num=1)

        # All should be maximally novel (no history)
        for agent, novelty in result.per_agent_novelty.items():
            assert novelty == 1.0

        assert result.avg_novelty == 1.0
        assert result.min_novelty == 1.0
        assert result.max_novelty == 1.0
        assert result.has_low_novelty() is False

    def test_first_round_details(self, tracker, sample_proposals):
        """Test first round novelty details."""
        result = tracker.compute_novelty(sample_proposals, round_num=1)

        for agent in sample_proposals:
            score = result.details[agent]
            assert score.novelty == 1.0
            assert score.max_similarity == 0.0
            assert score.most_similar_to is None
            assert score.prior_proposals_count == 0


# ============================================================================
# Subsequent Round Novelty Tests
# ============================================================================


class TestSubsequentRoundNovelty:
    """Tests for novelty computation in subsequent rounds."""

    def test_second_round_novelty(self, tracker, sample_proposals, mock_backend):
        """Test novelty computation in second round."""
        # Round 1
        tracker.compute_novelty(sample_proposals, round_num=1)
        tracker.add_to_history(sample_proposals)

        # Mock returns 0.5 similarity, so novelty = 0.5
        mock_backend.compute_similarity.return_value = 0.5

        # Round 2 - different proposals
        round2_proposals = {
            "claude": "A completely different approach",
            "gpt4": "Another strategy entirely",
        }
        result = tracker.compute_novelty(round2_proposals, round_num=2)

        # Novelty = 1 - 0.5 = 0.5
        for novelty in result.per_agent_novelty.values():
            assert novelty == 0.5

    def test_high_similarity_low_novelty(self, tracker, sample_proposals, mock_backend):
        """Test that high similarity results in low novelty."""
        # Round 1
        tracker.compute_novelty(sample_proposals, round_num=1)
        tracker.add_to_history(sample_proposals)

        # Mock high similarity (0.9) -> low novelty (0.1)
        mock_backend.compute_similarity.return_value = 0.9

        round2_proposals = {"claude": "Similar proposal"}
        result = tracker.compute_novelty(round2_proposals, round_num=2)

        assert result.per_agent_novelty["claude"] == pytest.approx(0.1)  # 1 - 0.9
        assert result.min_novelty == pytest.approx(0.1)
        assert "claude" in result.low_novelty_agents  # Below 0.15 threshold

    def test_most_similar_to_tracking(self, tracker, mock_backend):
        """Test that most similar agent is tracked."""
        # Round 1 with two agents
        round1 = {"claude": "Proposal A", "gpt4": "Proposal B"}
        tracker.compute_novelty(round1, round_num=1)
        tracker.add_to_history(round1)

        # Configure mock to return different similarities
        def similarity_mock(a, b):
            if "Proposal A" in b:
                return 0.8  # More similar to claude
            return 0.3  # Less similar to gpt4

        mock_backend.compute_similarity.side_effect = similarity_mock

        round2 = {"gemini": "New proposal"}
        result = tracker.compute_novelty(round2, round_num=2)

        # Should track that gemini was most similar to claude
        assert result.details["gemini"].most_similar_to == "claude"


# ============================================================================
# History Management Tests
# ============================================================================


class TestHistoryManagement:
    """Tests for proposal history management."""

    def test_add_to_history(self, tracker, sample_proposals):
        """Test adding proposals to history."""
        tracker.add_to_history(sample_proposals)

        assert len(tracker.history) == 1
        assert tracker.history[0] == sample_proposals

    def test_history_accumulates(self, tracker):
        """Test that history accumulates across rounds."""
        tracker.add_to_history({"a1": "P1"})
        tracker.add_to_history({"a2": "P2"})
        tracker.add_to_history({"a3": "P3"})

        assert len(tracker.history) == 3

    def test_history_is_copied(self, tracker):
        """Test that history stores copies, not references."""
        proposals = {"agent": "original"}
        tracker.add_to_history(proposals)

        # Modify original
        proposals["agent"] = "modified"

        # History should be unchanged
        assert tracker.history[0]["agent"] == "original"

    def test_reset_clears_history(self, tracker, sample_proposals):
        """Test that reset clears history."""
        tracker.add_to_history(sample_proposals)
        tracker.compute_novelty(sample_proposals, round_num=1)

        tracker.reset()

        assert len(tracker.history) == 0
        assert len(tracker.scores) == 0


# ============================================================================
# Agent Trajectory Tests
# ============================================================================


class TestAgentTrajectory:
    """Tests for agent novelty trajectories."""

    def test_get_agent_trajectory(self, tracker, mock_backend):
        """Test getting novelty trajectory for an agent."""
        # Round 1 - novel
        tracker.compute_novelty({"claude": "P1"}, round_num=1)
        tracker.add_to_history({"claude": "P1"})

        # Round 2 - less novel
        mock_backend.compute_similarity.return_value = 0.4
        tracker.compute_novelty({"claude": "P2"}, round_num=2)
        tracker.add_to_history({"claude": "P2"})

        # Round 3 - even less novel
        mock_backend.compute_similarity.return_value = 0.7
        tracker.compute_novelty({"claude": "P3"}, round_num=3)

        trajectory = tracker.get_agent_novelty_trajectory("claude")

        assert len(trajectory) == 3
        assert trajectory[0] == 1.0  # First round maximally novel
        assert trajectory[1] == pytest.approx(0.6)  # 1 - 0.4
        assert trajectory[2] == pytest.approx(0.3)  # 1 - 0.7

    def test_get_trajectory_missing_agent(self, tracker, mock_backend):
        """Test trajectory for agent not in all rounds."""
        tracker.compute_novelty({"claude": "P1"}, round_num=1)
        tracker.compute_novelty({"gpt4": "P2"}, round_num=2)

        trajectory = tracker.get_agent_novelty_trajectory("claude")

        assert trajectory == [1.0, 0.0]  # 0.0 for rounds where agent didn't participate


# ============================================================================
# Debate Summary Tests
# ============================================================================


class TestDebateSummary:
    """Tests for debate novelty summaries."""

    def test_get_summary_empty(self, tracker):
        """Test summary with no scores."""
        summary = tracker.get_debate_novelty_summary()

        assert summary["overall_avg"] == 1.0
        assert summary["overall_min"] == 1.0
        assert summary["total_rounds"] == 0

    def test_get_summary_with_data(self, tracker, mock_backend):
        """Test summary with computed scores."""
        # Round 1
        tracker.compute_novelty({"a1": "P1", "a2": "P2"}, round_num=1)
        tracker.add_to_history({"a1": "P1", "a2": "P2"})

        # Round 2 with some low novelty
        mock_backend.compute_similarity.return_value = 0.9  # 0.1 novelty
        tracker.compute_novelty({"a1": "P3"}, round_num=2)

        summary = tracker.get_debate_novelty_summary()

        assert summary["total_rounds"] == 2
        assert summary["rounds_with_low_novelty"] == 1
        assert 2 in summary["low_novelty_agents_by_round"]

    def test_summary_overall_min(self, tracker, mock_backend):
        """Test overall minimum in summary."""
        tracker.compute_novelty({"a1": "P1"}, round_num=1)
        tracker.add_to_history({"a1": "P1"})

        mock_backend.compute_similarity.return_value = 0.8  # 0.2 novelty
        tracker.compute_novelty({"a1": "P2"}, round_num=2)

        summary = tracker.get_debate_novelty_summary()

        # Min should be 0.2 from round 2
        assert summary["overall_min"] == pytest.approx(0.2)


# ============================================================================
# Low Novelty Detection Tests
# ============================================================================


class TestLowNoveltyDetection:
    """Tests for low novelty detection and flagging."""

    def test_detects_low_novelty_agents(self, tracker, mock_backend):
        """Test that low novelty agents are correctly flagged."""
        tracker.compute_novelty({"a1": "P1"}, round_num=1)
        tracker.add_to_history({"a1": "P1"})

        # Very high similarity = very low novelty
        mock_backend.compute_similarity.return_value = 0.95

        result = tracker.compute_novelty({"a1": "P2", "a2": "P3"}, round_num=2)

        # Both should be flagged (0.05 novelty < 0.15 threshold)
        assert "a1" in result.low_novelty_agents
        assert "a2" in result.low_novelty_agents

    def test_custom_threshold_affects_detection(self, mock_backend):
        """Test that custom threshold affects low novelty detection."""
        # High threshold tracker
        tracker = NoveltyTracker(backend=mock_backend, low_novelty_threshold=0.5)

        tracker.compute_novelty({"a1": "P1"}, round_num=1)
        tracker.add_to_history({"a1": "P1"})

        mock_backend.compute_similarity.return_value = 0.6  # 0.4 novelty

        result = tracker.compute_novelty({"a1": "P2"}, round_num=2)

        # 0.4 < 0.5 threshold, should be flagged
        assert "a1" in result.low_novelty_agents
