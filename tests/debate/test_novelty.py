"""Comprehensive tests for novelty tracking in multi-agent debates."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.novelty import (
    CodebaseNoveltyChecker,
    CodebaseNoveltyResult,
    NoveltyResult,
    NoveltyScore,
    NoveltyTracker,
)


# ---------------------------------------------------------------------------
# Test helpers / mocks
# ---------------------------------------------------------------------------


class MockSimilarityBackend:
    """Mock similarity backend with controllable similarity scores."""

    def __init__(self):
        self.similarity_map: dict[tuple[str, str], float] = {}
        self.default_similarity: float = 0.0

    def set_similarity(self, text1: str, text2: str, score: float) -> None:
        """Set similarity score for a specific pair."""
        # Store both orderings
        self.similarity_map[(text1, text2)] = score
        self.similarity_map[(text2, text1)] = score

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity using the configured map."""
        return self.similarity_map.get((text1, text2), self.default_similarity)


# ---------------------------------------------------------------------------
# NoveltyScore tests
# ---------------------------------------------------------------------------


def test_novelty_score_defaults():
    """Test NoveltyScore default values."""
    score = NoveltyScore(
        agent="agent-1",
        round_num=1,
        novelty=0.8,
        max_similarity=0.2,
    )
    assert score.agent == "agent-1"
    assert score.round_num == 1
    assert score.novelty == 0.8
    assert score.max_similarity == 0.2
    assert score.most_similar_to is None
    assert score.prior_proposals_count == 0


def test_novelty_score_is_low_novelty_below_threshold():
    """Test is_low_novelty returns True when below threshold."""
    score = NoveltyScore(
        agent="agent-1",
        round_num=2,
        novelty=0.10,  # Below default 0.15
        max_similarity=0.90,
    )
    assert score.is_low_novelty() is True


def test_novelty_score_is_low_novelty_at_threshold():
    """Test is_low_novelty returns False when exactly at threshold."""
    score = NoveltyScore(
        agent="agent-1",
        round_num=2,
        novelty=0.15,  # Exactly at threshold
        max_similarity=0.85,
    )
    # At threshold is not considered low (uses < not <=)
    assert score.is_low_novelty(threshold=0.15) is False
    # Just below threshold is low
    score_below = NoveltyScore(
        agent="agent-1",
        round_num=2,
        novelty=0.14,
        max_similarity=0.86,
    )
    assert score_below.is_low_novelty(threshold=0.15) is True


def test_novelty_score_is_low_novelty_above_threshold():
    """Test is_low_novelty returns False when above threshold."""
    score = NoveltyScore(
        agent="agent-1",
        round_num=2,
        novelty=0.20,  # Above default 0.15
        max_similarity=0.80,
    )
    assert score.is_low_novelty() is False


def test_novelty_score_is_low_novelty_custom_threshold():
    """Test is_low_novelty with custom threshold."""
    score = NoveltyScore(
        agent="agent-1",
        round_num=2,
        novelty=0.25,
        max_similarity=0.75,
    )
    assert score.is_low_novelty(threshold=0.30) is True
    assert score.is_low_novelty(threshold=0.20) is False


# ---------------------------------------------------------------------------
# NoveltyResult tests
# ---------------------------------------------------------------------------


def test_novelty_result_defaults():
    """Test NoveltyResult default values."""
    result = NoveltyResult(round_num=1)
    assert result.round_num == 1
    assert result.per_agent_novelty == {}
    assert result.avg_novelty == 0.0
    assert result.min_novelty == 1.0
    assert result.max_novelty == 0.0
    assert result.low_novelty_agents == []
    assert result.details == {}


def test_novelty_result_has_low_novelty_true():
    """Test has_low_novelty returns True when agents present."""
    result = NoveltyResult(
        round_num=2,
        low_novelty_agents=["agent-1", "agent-2"],
    )
    assert result.has_low_novelty() is True


def test_novelty_result_has_low_novelty_false():
    """Test has_low_novelty returns False when no agents."""
    result = NoveltyResult(
        round_num=2,
        low_novelty_agents=[],
    )
    assert result.has_low_novelty() is False


# ---------------------------------------------------------------------------
# NoveltyTracker tests
# ---------------------------------------------------------------------------


def test_novelty_tracker_first_round_all_novel():
    """Test first round proposals are all maximally novel."""
    mock_backend = MockSimilarityBackend()
    tracker = NoveltyTracker(backend=mock_backend)

    proposals = {
        "agent-1": "Proposal A",
        "agent-2": "Proposal B",
        "agent-3": "Proposal C",
    }

    result = tracker.compute_novelty(proposals, round_num=1)

    # All should be maximally novel
    assert len(result.per_agent_novelty) == 3
    assert all(novelty == 1.0 for novelty in result.per_agent_novelty.values())
    assert result.avg_novelty == 1.0
    assert result.min_novelty == 1.0
    assert result.max_novelty == 1.0
    assert result.low_novelty_agents == []
    assert result.has_low_novelty() is False

    # Check details
    for agent in proposals:
        score = result.details[agent]
        assert score.novelty == 1.0
        assert score.max_similarity == 0.0
        assert score.most_similar_to is None
        assert score.prior_proposals_count == 0


def test_novelty_tracker_second_round_high_similarity_low_novelty():
    """Test second round with high similarity results in low novelty."""
    mock_backend = MockSimilarityBackend()
    tracker = NoveltyTracker(backend=mock_backend, low_novelty_threshold=0.15)

    # Round 1
    round1_proposals = {
        "agent-1": "Implement feature X",
        "agent-2": "Add capability Y",
    }
    tracker.compute_novelty(round1_proposals, round_num=1)
    tracker.add_to_history(round1_proposals)

    # Round 2 - agent-1 proposes very similar idea
    round2_proposals = {
        "agent-1": "Also implement feature X",  # Very similar to own round 1
        "agent-2": "Completely different idea Z",  # Novel
    }

    # Set high similarity for agent-1's proposals
    mock_backend.set_similarity("Also implement feature X", "Implement feature X", 0.95)
    mock_backend.set_similarity("Also implement feature X", "Add capability Y", 0.10)
    mock_backend.set_similarity("Completely different idea Z", "Implement feature X", 0.05)
    mock_backend.set_similarity("Completely different idea Z", "Add capability Y", 0.05)

    result = tracker.compute_novelty(round2_proposals, round_num=2)

    # agent-1 should have low novelty (1 - 0.95 = 0.05)
    assert result.per_agent_novelty["agent-1"] == pytest.approx(0.05, abs=0.01)
    assert result.per_agent_novelty["agent-2"] == pytest.approx(0.95, abs=0.01)

    # agent-1 should be flagged as low novelty
    assert "agent-1" in result.low_novelty_agents
    assert "agent-2" not in result.low_novelty_agents
    assert result.has_low_novelty() is True

    # Check details
    score1 = result.details["agent-1"]
    assert score1.novelty == pytest.approx(0.05, abs=0.01)
    assert score1.max_similarity == pytest.approx(0.95, abs=0.01)
    assert score1.most_similar_to == "agent-1"
    assert score1.prior_proposals_count == 2


def test_novelty_tracker_second_round_low_similarity_high_novelty():
    """Test second round with low similarity results in high novelty."""
    mock_backend = MockSimilarityBackend()
    mock_backend.default_similarity = 0.05  # Low default
    tracker = NoveltyTracker(backend=mock_backend)

    # Round 1
    round1_proposals = {
        "agent-1": "Old idea A",
    }
    tracker.compute_novelty(round1_proposals, round_num=1)
    tracker.add_to_history(round1_proposals)

    # Round 2
    round2_proposals = {
        "agent-1": "Completely new idea B",
    }

    result = tracker.compute_novelty(round2_proposals, round_num=2)

    # Should have high novelty (1 - 0.05 = 0.95)
    assert result.per_agent_novelty["agent-1"] == pytest.approx(0.95, abs=0.01)
    assert result.low_novelty_agents == []
    assert result.has_low_novelty() is False


def test_novelty_tracker_multiple_rounds_history_accumulation():
    """Test novelty computation across multiple rounds with history."""
    mock_backend = MockSimilarityBackend()
    tracker = NoveltyTracker(backend=mock_backend)

    # Round 1
    round1 = {"agent-1": "Idea 1"}
    tracker.compute_novelty(round1, round_num=1)
    tracker.add_to_history(round1)

    # Round 2
    round2 = {"agent-1": "Idea 2"}
    mock_backend.set_similarity("Idea 2", "Idea 1", 0.20)
    result2 = tracker.compute_novelty(round2, round_num=2)
    assert result2.details["agent-1"].prior_proposals_count == 1
    tracker.add_to_history(round2)

    # Round 3 - should compare against both round 1 and 2
    round3 = {"agent-1": "Idea 3"}
    mock_backend.set_similarity("Idea 3", "Idea 1", 0.10)
    mock_backend.set_similarity("Idea 3", "Idea 2", 0.30)  # Higher similarity
    result3 = tracker.compute_novelty(round3, round_num=3)

    # Should use max similarity (0.30), so novelty = 0.70
    assert result3.per_agent_novelty["agent-1"] == pytest.approx(0.70, abs=0.01)
    assert result3.details["agent-1"].prior_proposals_count == 2
    assert result3.details["agent-1"].max_similarity == pytest.approx(0.30, abs=0.01)


def test_novelty_tracker_add_to_history():
    """Test add_to_history stores proposals correctly."""
    tracker = NoveltyTracker(backend=MockSimilarityBackend())

    proposals = {"agent-1": "Test proposal"}
    tracker.add_to_history(proposals)

    assert len(tracker.history) == 1
    assert tracker.history[0] == {"agent-1": "Test proposal"}

    # Add another round
    proposals2 = {"agent-2": "Another proposal"}
    tracker.add_to_history(proposals2)

    assert len(tracker.history) == 2
    assert tracker.history[1] == {"agent-2": "Another proposal"}


def test_novelty_tracker_add_to_history_copies_dict():
    """Test add_to_history creates a copy to prevent mutation."""
    tracker = NoveltyTracker(backend=MockSimilarityBackend())

    proposals = {"agent-1": "Test"}
    tracker.add_to_history(proposals)

    # Mutate original
    proposals["agent-2"] = "Added later"

    # History should not be affected
    assert len(tracker.history[0]) == 1
    assert "agent-2" not in tracker.history[0]


def test_novelty_tracker_low_novelty_agents_detected():
    """Test low novelty agents are correctly identified."""
    mock_backend = MockSimilarityBackend()
    tracker = NoveltyTracker(backend=mock_backend, low_novelty_threshold=0.20)

    # Round 1
    round1 = {"agent-1": "A", "agent-2": "B"}
    tracker.compute_novelty(round1, round_num=1)
    tracker.add_to_history(round1)

    # Round 2
    round2 = {"agent-1": "A2", "agent-2": "B2", "agent-3": "C"}

    # agent-1: high similarity (0.85) -> novelty 0.15 (LOW)
    # agent-2: medium similarity (0.50) -> novelty 0.50 (OK)
    # agent-3: low similarity (0.10) -> novelty 0.90 (OK)
    mock_backend.set_similarity("A2", "A", 0.85)
    mock_backend.set_similarity("A2", "B", 0.10)
    mock_backend.set_similarity("B2", "A", 0.20)
    mock_backend.set_similarity("B2", "B", 0.50)
    mock_backend.set_similarity("C", "A", 0.10)
    mock_backend.set_similarity("C", "B", 0.10)

    result = tracker.compute_novelty(round2, round_num=2)

    # Only agent-1 should be flagged
    assert result.low_novelty_agents == ["agent-1"]
    assert result.has_low_novelty() is True


def test_novelty_tracker_get_agent_novelty_trajectory():
    """Test get_agent_novelty_trajectory returns scores across rounds."""
    mock_backend = MockSimilarityBackend()
    mock_backend.default_similarity = 0.10
    tracker = NoveltyTracker(backend=mock_backend)

    # Round 1
    tracker.compute_novelty({"agent-1": "A"}, round_num=1)
    tracker.add_to_history({"agent-1": "A"})

    # Round 2
    tracker.compute_novelty({"agent-1": "B"}, round_num=2)
    tracker.add_to_history({"agent-1": "B"})

    # Round 3
    tracker.compute_novelty({"agent-1": "C"}, round_num=3)

    trajectory = tracker.get_agent_novelty_trajectory("agent-1")

    # Should have 3 scores
    assert len(trajectory) == 3
    assert trajectory[0] == 1.0  # First round always novel
    assert trajectory[1] == pytest.approx(0.90, abs=0.01)  # 1 - 0.10
    assert trajectory[2] == pytest.approx(0.90, abs=0.01)  # 1 - 0.10


def test_novelty_tracker_get_agent_novelty_trajectory_missing_agent():
    """Test get_agent_novelty_trajectory handles missing agent gracefully."""
    tracker = NoveltyTracker(backend=MockSimilarityBackend())

    tracker.compute_novelty({"agent-1": "A"}, round_num=1)
    trajectory = tracker.get_agent_novelty_trajectory("agent-2")

    # Should return 0.0 for rounds where agent didn't participate
    assert trajectory == [0.0]


def test_novelty_tracker_get_debate_novelty_summary_empty():
    """Test get_debate_novelty_summary with no scores."""
    tracker = NoveltyTracker(backend=MockSimilarityBackend())

    summary = tracker.get_debate_novelty_summary()

    assert summary["overall_avg"] == 1.0
    assert summary["overall_min"] == 1.0
    assert summary["rounds_with_low_novelty"] == 0
    assert summary["total_rounds"] == 0


def test_novelty_tracker_get_debate_novelty_summary_with_data():
    """Test get_debate_novelty_summary computes correct statistics."""
    mock_backend = MockSimilarityBackend()
    tracker = NoveltyTracker(backend=mock_backend, low_novelty_threshold=0.20)

    # Round 1
    tracker.compute_novelty({"agent-1": "A", "agent-2": "B"}, round_num=1)
    tracker.add_to_history({"agent-1": "A", "agent-2": "B"})

    # Round 2 - one low novelty
    mock_backend.set_similarity("A2", "A", 0.85)
    mock_backend.set_similarity("A2", "B", 0.10)
    mock_backend.set_similarity("B2", "A", 0.10)
    mock_backend.set_similarity("B2", "B", 0.10)
    tracker.compute_novelty({"agent-1": "A2", "agent-2": "B2"}, round_num=2)
    tracker.add_to_history({"agent-1": "A2", "agent-2": "B2"})

    # Round 3 - no low novelty
    mock_backend.default_similarity = 0.05
    tracker.compute_novelty({"agent-1": "C", "agent-2": "D"}, round_num=3)

    summary = tracker.get_debate_novelty_summary()

    # All novelties: [1.0, 1.0, 0.15, 0.90, 0.95, 0.95]
    # Avg = (1.0 + 1.0 + 0.15 + 0.90 + 0.95 + 0.95) / 6 ≈ 0.825
    assert summary["overall_avg"] == pytest.approx(0.825, abs=0.01)
    assert summary["overall_min"] == pytest.approx(0.15, abs=0.01)
    assert summary["rounds_with_low_novelty"] == 1  # Only round 2
    assert summary["total_rounds"] == 3
    assert 2 in summary["low_novelty_agents_by_round"]
    assert summary["low_novelty_agents_by_round"][2] == ["agent-1"]


def test_novelty_tracker_reset():
    """Test reset clears all state."""
    mock_backend = MockSimilarityBackend()
    tracker = NoveltyTracker(backend=mock_backend)

    # Add some data
    tracker.compute_novelty({"agent-1": "A"}, round_num=1)
    tracker.add_to_history({"agent-1": "A"})

    # Reset
    tracker.reset()

    assert len(tracker.history) == 0
    assert len(tracker.scores) == 0


def test_novelty_tracker_empty_proposals():
    """Test compute_novelty handles empty proposals."""
    tracker = NoveltyTracker(backend=MockSimilarityBackend())

    result = tracker.compute_novelty({}, round_num=1)

    assert result.per_agent_novelty == {}
    assert result.avg_novelty == 1.0  # Default when no values
    assert result.min_novelty == 1.0
    assert result.max_novelty == 1.0
    assert result.low_novelty_agents == []


def test_novelty_tracker_initialization_with_custom_threshold():
    """Test NoveltyTracker initialization with custom threshold."""
    mock_backend = MockSimilarityBackend()
    tracker = NoveltyTracker(backend=mock_backend, low_novelty_threshold=0.25)

    assert tracker.low_novelty_threshold == 0.25


def test_novelty_tracker_initialization_with_auto_backend():
    """Test NoveltyTracker uses auto backend when not provided."""
    with patch("aragora.debate.novelty.get_similarity_backend") as mock_get_backend:
        mock_backend = MockSimilarityBackend()
        mock_get_backend.return_value = mock_backend

        tracker = NoveltyTracker()

        mock_get_backend.assert_called_once_with("auto")
        assert tracker.backend is mock_backend


# ---------------------------------------------------------------------------
# CodebaseNoveltyResult tests
# ---------------------------------------------------------------------------


def test_codebase_novelty_result_defaults():
    """Test CodebaseNoveltyResult default values."""
    result = CodebaseNoveltyResult(
        proposal="Test proposal",
        agent="agent-1",
        is_novel=True,
        max_similarity=0.4,
    )
    assert result.proposal == "Test proposal"
    assert result.agent == "agent-1"
    assert result.is_novel is True
    assert result.max_similarity == 0.4
    assert result.most_similar_feature is None
    assert result.feature_module is None
    assert result.warning is None


def test_codebase_novelty_result_to_dict():
    """Test CodebaseNoveltyResult.to_dict conversion."""
    result = CodebaseNoveltyResult(
        proposal="Add feature X",
        agent="agent-1",
        is_novel=False,
        max_similarity=0.85,
        most_similar_feature="Feature X",
        feature_module="module.py",
        warning="Duplicate feature",
    )

    result_dict = result.to_dict()

    assert result_dict == {
        "agent": "agent-1",
        "is_novel": False,
        "max_similarity": 0.85,
        "most_similar_feature": "Feature X",
        "feature_module": "module.py",
        "warning": "Duplicate feature",
    }
    # Proposal should not be in dict
    assert "proposal" not in result_dict


# ---------------------------------------------------------------------------
# CodebaseNoveltyChecker tests
# ---------------------------------------------------------------------------


def test_codebase_novelty_checker_extract_features_from_table():
    """Test feature extraction from table format."""
    context = """
## Existing Features

| Feature | Module | Status |
|---------|--------|--------|
| WebSocket Streaming | server/stream | Stable |
| Memory System | memory/ | Stable |
| Agent Ranking | ranking/elo.py | Stable |
"""

    mock_backend = MockSimilarityBackend()
    checker = CodebaseNoveltyChecker(context, backend=mock_backend)

    # Should extract 3 features
    assert len(checker.features) >= 3
    feature_names = [f["name"] for f in checker.features]
    assert "WebSocket Streaming" in feature_names
    assert "Memory System" in feature_names
    assert "Agent Ranking" in feature_names


def test_codebase_novelty_checker_extract_features_from_bullets():
    """Test feature extraction from bullet point format."""
    context = """
## Features

- Authentication: OIDC and SAML SSO support
- Rate Limiting: Token bucket implementation
- Backup System: Incremental backups with retention
"""

    mock_backend = MockSimilarityBackend()
    checker = CodebaseNoveltyChecker(context, backend=mock_backend)

    feature_names = [f["name"] for f in checker.features]
    assert "Authentication" in feature_names
    assert "Rate Limiting" in feature_names
    assert "Backup System" in feature_names


def test_codebase_novelty_checker_extract_key_terms():
    """Test extraction of capitalized key terms."""
    context = """
The system includes ELO Rankings and RLM integration.
We also have Knowledge Mound for semantic search.
"""

    mock_backend = MockSimilarityBackend()
    checker = CodebaseNoveltyChecker(context, backend=mock_backend)

    # Should extract capitalized terms
    feature_names = [f["name"] for f in checker.features]
    # May contain: ELO, RLM, Knowledge Mound
    assert any("ELO" in name or "Knowledge" in name for name in feature_names)


def test_codebase_novelty_checker_check_proposal_novel():
    """Test check_proposal identifies novel proposal."""
    context = """
| Feature | Module |
|---------|--------|
| Memory System | memory/ |
"""

    mock_backend = MockSimilarityBackend()
    mock_backend.default_similarity = 0.30  # Below threshold
    checker = CodebaseNoveltyChecker(context, backend=mock_backend, novelty_threshold=0.65)

    result = checker.check_proposal("Implement real-time analytics", "agent-1")

    assert result.is_novel is True
    assert result.max_similarity == pytest.approx(0.30, abs=0.01)
    assert result.warning is None


def test_codebase_novelty_checker_check_proposal_non_novel():
    """Test check_proposal identifies non-novel proposal."""
    context = """
| Feature | Module |
|---------|--------|
| WebSocket Streaming | server/stream |
"""

    mock_backend = MockSimilarityBackend()
    checker = CodebaseNoveltyChecker(context, backend=mock_backend, novelty_threshold=0.65)

    proposal = "Add streaming support"
    # High similarity to "WebSocket Streaming: server/stream"
    mock_backend.set_similarity(proposal, "WebSocket Streaming: server/stream", 0.80)

    result = checker.check_proposal(proposal, "agent-1")

    assert result.is_novel is False
    assert result.max_similarity == pytest.approx(0.80, abs=0.01)
    assert result.most_similar_feature == "WebSocket Streaming"
    assert result.feature_module == "server/stream"
    assert result.warning is not None
    assert "WebSocket Streaming" in result.warning


def test_codebase_novelty_checker_synonym_matching():
    """Test synonym matching boosts similarity."""
    context = """
| Feature | Module |
|---------|--------|
| Streaming API | server/stream |
"""

    mock_backend = MockSimilarityBackend()
    mock_backend.default_similarity = 0.40  # Base similarity
    checker = CodebaseNoveltyChecker(context, backend=mock_backend, novelty_threshold=0.55)

    # Proposal mentions "websocket" which is synonym of "streaming"
    result = checker.check_proposal("Add websocket support", "agent-1")

    # Should boost similarity to at least 0.6
    assert result.max_similarity >= 0.60
    assert result.is_novel is False  # Above threshold 0.55


def test_codebase_novelty_checker_direct_name_match_boost():
    """Test direct feature name match boosts similarity."""
    context = """
| Feature | Module |
|---------|--------|
| Dashboard | ui/dashboard |
"""

    mock_backend = MockSimilarityBackend()
    mock_backend.default_similarity = 0.20  # Low base similarity
    checker = CodebaseNoveltyChecker(context, backend=mock_backend)

    # Proposal contains "dashboard" directly
    result = checker.check_proposal("Improve the dashboard UI", "agent-1")

    # Should boost similarity to at least 0.7
    assert result.max_similarity >= 0.70
    assert result.is_novel is False  # Above threshold


def test_codebase_novelty_checker_check_proposals_multiple():
    """Test check_proposals handles multiple agents."""
    context = """
| Feature | Module |
|---------|--------|
| Memory System | memory/ |
"""

    mock_backend = MockSimilarityBackend()
    checker = CodebaseNoveltyChecker(context, backend=mock_backend)

    proposals = {
        "agent-1": "Add caching layer",  # Novel
        "agent-2": "Enhance memory system",  # Non-novel
    }

    mock_backend.set_similarity("Add caching layer", "Memory System: memory/", 0.40)
    mock_backend.set_similarity("Enhance memory system", "Memory System: memory/", 0.75)

    results = checker.check_proposals(proposals)

    assert len(results) == 2
    assert results["agent-1"].is_novel is True
    assert results["agent-2"].is_novel is False


def test_codebase_novelty_checker_get_non_novel_proposals():
    """Test get_non_novel_proposals filters correctly."""
    context = """
| Feature | Module |
|---------|--------|
| Feature A | module_a |
"""

    mock_backend = MockSimilarityBackend()
    checker = CodebaseNoveltyChecker(context, backend=mock_backend)

    proposals = {
        "agent-1": "Novel idea",
        "agent-2": "Duplicate of Feature A",
        "agent-3": "Another novel idea",
    }

    mock_backend.set_similarity("Novel idea", "Feature A: module_a", 0.30)
    mock_backend.set_similarity("Duplicate of Feature A", "Feature A: module_a", 0.85)
    mock_backend.set_similarity("Another novel idea", "Feature A: module_a", 0.25)

    non_novel = checker.get_non_novel_proposals(proposals)

    assert len(non_novel) == 1
    assert non_novel[0].agent == "agent-2"
    assert non_novel[0].is_novel is False


def test_codebase_novelty_checker_empty_features():
    """Test checker handles empty features gracefully."""
    context = ""  # No features

    mock_backend = MockSimilarityBackend()
    checker = CodebaseNoveltyChecker(context, backend=mock_backend)

    result = checker.check_proposal("Any proposal", "agent-1")

    # With no features, should be novel by default
    assert result.is_novel is True
    assert result.max_similarity == 0.0
    assert result.warning is not None
    assert "No codebase features" in result.warning


def test_codebase_novelty_checker_empty_context():
    """Test checker with empty context string."""
    mock_backend = MockSimilarityBackend()
    checker = CodebaseNoveltyChecker("", backend=mock_backend)

    # Should have empty or minimal features
    assert len(checker.features) == 0 or all(f["name"] == "" for f in checker.features)


def test_codebase_novelty_checker_initialization_with_custom_threshold():
    """Test CodebaseNoveltyChecker initialization with custom threshold."""
    mock_backend = MockSimilarityBackend()
    checker = CodebaseNoveltyChecker("", backend=mock_backend, novelty_threshold=0.75)

    assert checker.novelty_threshold == 0.75


def test_codebase_novelty_checker_initialization_with_auto_backend():
    """Test CodebaseNoveltyChecker uses auto backend when not provided."""
    with patch("aragora.debate.novelty.get_similarity_backend") as mock_get_backend:
        mock_backend = MockSimilarityBackend()
        mock_get_backend.return_value = mock_backend

        checker = CodebaseNoveltyChecker("Test context")

        mock_get_backend.assert_called_once_with("auto")
        assert checker.backend is mock_backend


def test_codebase_novelty_checker_feature_synonyms_coverage():
    """Test that FEATURE_SYNONYMS dictionary is comprehensive."""
    # Just verify the structure exists and has expected keys
    from aragora.debate.novelty import CodebaseNoveltyChecker

    assert "streaming" in CodebaseNoveltyChecker.FEATURE_SYNONYMS
    assert "spectator" in CodebaseNoveltyChecker.FEATURE_SYNONYMS
    assert "dashboard" in CodebaseNoveltyChecker.FEATURE_SYNONYMS
    assert "memory" in CodebaseNoveltyChecker.FEATURE_SYNONYMS
    assert "learning" in CodebaseNoveltyChecker.FEATURE_SYNONYMS
    assert "consensus" in CodebaseNoveltyChecker.FEATURE_SYNONYMS
    assert "novelty" in CodebaseNoveltyChecker.FEATURE_SYNONYMS

    # Verify synonyms are lists
    assert isinstance(CodebaseNoveltyChecker.FEATURE_SYNONYMS["streaming"], list)
    assert len(CodebaseNoveltyChecker.FEATURE_SYNONYMS["streaming"]) > 0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_novelty_tracker_integration_realistic_debate():
    """Integration test simulating realistic debate flow."""
    mock_backend = MockSimilarityBackend()
    tracker = NoveltyTracker(backend=mock_backend, low_novelty_threshold=0.15)

    # Round 1 - initial proposals
    round1 = {
        "agent-1": "Implement rate limiting using token bucket algorithm",
        "agent-2": "Add caching layer with Redis",
        "agent-3": "Improve error handling with circuit breakers",
    }
    result1 = tracker.compute_novelty(round1, round_num=1)
    assert all(n == 1.0 for n in result1.per_agent_novelty.values())
    tracker.add_to_history(round1)

    # Round 2 - refinements (should be reasonably novel)
    round2 = {
        "agent-1": "Use sliding window rate limiting instead",
        "agent-2": "Implement write-through caching strategy",
        "agent-3": "Add retry logic with exponential backoff",
    }
    # Set moderate similarities (refined ideas, not duplicates)
    for prop2 in round2.values():
        for prop1 in round1.values():
            mock_backend.set_similarity(prop2, prop1, 0.45)

    result2 = tracker.compute_novelty(round2, round_num=2)
    assert all(0.5 < n < 0.6 for n in result2.per_agent_novelty.values())
    assert result2.has_low_novelty() is False
    tracker.add_to_history(round2)

    # Round 3 - one agent repeats idea (low novelty)
    round3 = {
        "agent-1": "Still think token bucket is best",  # Back to round 1
        "agent-2": "Consider memcached as alternative",
        "agent-3": "What about rate limiting with leaky bucket?",
    }
    mock_backend.set_similarity(
        "Still think token bucket is best",
        "Implement rate limiting using token bucket algorithm",
        0.90,
    )
    for prop1 in round1.values():
        mock_backend.set_similarity("Consider memcached as alternative", prop1, 0.30)
        mock_backend.set_similarity(
            "What about rate limiting with leaky bucket?",
            prop1,
            0.35,
        )
    for prop2 in round2.values():
        mock_backend.set_similarity("Still think token bucket is best", prop2, 0.35)
        mock_backend.set_similarity("Consider memcached as alternative", prop2, 0.30)
        mock_backend.set_similarity(
            "What about rate limiting with leaky bucket?",
            prop2,
            0.35,
        )

    result3 = tracker.compute_novelty(round3, round_num=3)
    assert "agent-1" in result3.low_novelty_agents
    assert result3.has_low_novelty() is True

    # Verify summary
    summary = tracker.get_debate_novelty_summary()
    assert summary["total_rounds"] == 3
    assert summary["rounds_with_low_novelty"] == 1


# ---------------------------------------------------------------------------
# Additional edge cases and coverage tests
# ---------------------------------------------------------------------------


def test_novelty_tracker_single_agent_debate():
    """Test novelty tracking with a single agent across rounds."""
    mock_backend = MockSimilarityBackend()
    tracker = NoveltyTracker(backend=mock_backend)

    # Round 1
    tracker.compute_novelty({"agent-1": "Initial idea"}, round_num=1)
    tracker.add_to_history({"agent-1": "Initial idea"})

    # Round 2 - same agent evolves idea
    mock_backend.set_similarity("Evolved idea", "Initial idea", 0.50)
    result2 = tracker.compute_novelty({"agent-1": "Evolved idea"}, round_num=2)

    assert result2.per_agent_novelty["agent-1"] == pytest.approx(0.50, abs=0.01)
    assert len(result2.per_agent_novelty) == 1


def test_novelty_tracker_agent_joins_mid_debate():
    """Test novelty when new agent joins in later rounds."""
    mock_backend = MockSimilarityBackend()
    tracker = NoveltyTracker(backend=mock_backend)

    # Round 1 - only agent-1
    tracker.compute_novelty({"agent-1": "Idea A"}, round_num=1)
    tracker.add_to_history({"agent-1": "Idea A"})

    # Round 2 - agent-2 joins
    mock_backend.set_similarity("Idea B", "Idea A", 0.30)
    result2 = tracker.compute_novelty({"agent-1": "Idea A2", "agent-2": "Idea B"}, round_num=2)

    # Both agents should have valid novelty scores
    assert "agent-1" in result2.per_agent_novelty
    assert "agent-2" in result2.per_agent_novelty


def test_novelty_tracker_agent_leaves_mid_debate():
    """Test novelty when agent stops participating."""
    mock_backend = MockSimilarityBackend()
    tracker = NoveltyTracker(backend=mock_backend)

    # Round 1 - both agents
    tracker.compute_novelty({"agent-1": "A", "agent-2": "B"}, round_num=1)
    tracker.add_to_history({"agent-1": "A", "agent-2": "B"})

    # Round 2 - only agent-1
    mock_backend.set_similarity("C", "A", 0.20)
    mock_backend.set_similarity("C", "B", 0.20)
    result2 = tracker.compute_novelty({"agent-1": "C"}, round_num=2)

    assert len(result2.per_agent_novelty) == 1
    assert "agent-1" in result2.per_agent_novelty
    assert "agent-2" not in result2.per_agent_novelty


def test_novelty_score_with_all_fields():
    """Test NoveltyScore with all fields populated."""
    score = NoveltyScore(
        agent="agent-test",
        round_num=5,
        novelty=0.42,
        max_similarity=0.58,
        most_similar_to="agent-previous",
        prior_proposals_count=12,
    )
    assert score.agent == "agent-test"
    assert score.round_num == 5
    assert score.novelty == 0.42
    assert score.max_similarity == 0.58
    assert score.most_similar_to == "agent-previous"
    assert score.prior_proposals_count == 12


def test_novelty_tracker_aggregate_metrics_single_agent():
    """Test aggregate metrics work correctly with single agent."""
    tracker = NoveltyTracker(backend=MockSimilarityBackend())

    result = tracker.compute_novelty({"agent-1": "Solo proposal"}, round_num=1)

    assert result.avg_novelty == 1.0
    assert result.min_novelty == 1.0
    assert result.max_novelty == 1.0


def test_novelty_tracker_aggregate_metrics_multiple_agents():
    """Test aggregate metrics with multiple agents and varied novelty."""
    mock_backend = MockSimilarityBackend()
    tracker = NoveltyTracker(backend=mock_backend, low_novelty_threshold=0.20)

    # Round 1
    tracker.compute_novelty({"agent-1": "A", "agent-2": "B", "agent-3": "C"}, round_num=1)
    tracker.add_to_history({"agent-1": "A", "agent-2": "B", "agent-3": "C"})

    # Round 2 with varied similarities
    round2 = {"agent-1": "A2", "agent-2": "B2", "agent-3": "C2"}
    mock_backend.set_similarity("A2", "A", 0.80)  # novelty = 0.20
    mock_backend.set_similarity("A2", "B", 0.10)
    mock_backend.set_similarity("A2", "C", 0.10)
    mock_backend.set_similarity("B2", "A", 0.10)
    mock_backend.set_similarity("B2", "B", 0.50)  # novelty = 0.50
    mock_backend.set_similarity("B2", "C", 0.10)
    mock_backend.set_similarity("C2", "A", 0.10)
    mock_backend.set_similarity("C2", "B", 0.10)
    mock_backend.set_similarity("C2", "C", 0.30)  # novelty = 0.70

    result = tracker.compute_novelty(round2, round_num=2)

    # avg = (0.20 + 0.50 + 0.70) / 3 ≈ 0.467
    assert result.avg_novelty == pytest.approx(0.467, abs=0.01)
    assert result.min_novelty == pytest.approx(0.20, abs=0.01)
    assert result.max_novelty == pytest.approx(0.70, abs=0.01)


def test_codebase_novelty_checker_table_header_filtering():
    """Test that table headers are not extracted as features."""
    context = """
| Feature | Module | Status |
|---------|--------|--------|
| Real Feature | real_module | OK |
"""

    checker = CodebaseNoveltyChecker(context, backend=MockSimilarityBackend())

    feature_names = [f["name"] for f in checker.features]
    # "Feature" and "Module" headers should be filtered out
    assert "Feature" not in feature_names
    assert "Module" not in feature_names
    assert "Real Feature" in feature_names


def test_codebase_novelty_checker_section_tracking():
    """Test that features track their section."""
    context = """
## Core Features

| Feature | Module |
|---------|--------|
| Feature A | module_a |

## Beta Features

| Feature | Module |
|---------|--------|
| Feature B | module_b |
"""

    checker = CodebaseNoveltyChecker(context, backend=MockSimilarityBackend())

    features_dict = {f["name"]: f for f in checker.features}
    assert features_dict["Feature A"]["section"] == "core features"
    assert features_dict["Feature B"]["section"] == "beta features"


def test_codebase_novelty_checker_multiple_synonym_matches():
    """Test multiple synonym matches in proposal."""
    context = """
| Feature | Module |
|---------|--------|
| Streaming System | stream/ |
"""

    mock_backend = MockSimilarityBackend()
    mock_backend.default_similarity = 0.20
    checker = CodebaseNoveltyChecker(context, backend=mock_backend)

    # Proposal has multiple synonyms: "real-time" and "live"
    result = checker.check_proposal("Add real-time live updates", "agent-1")

    # Should boost similarity
    assert result.max_similarity >= 0.60


def test_codebase_novelty_result_all_fields_populated():
    """Test CodebaseNoveltyResult with all fields."""
    result = CodebaseNoveltyResult(
        proposal="Full proposal text",
        agent="test-agent",
        is_novel=False,
        max_similarity=0.92,
        most_similar_feature="Exact Match Feature",
        feature_module="exact/module.py",
        warning="This is a duplicate",
    )

    assert result.proposal == "Full proposal text"
    assert result.agent == "test-agent"
    assert result.is_novel is False
    assert result.max_similarity == 0.92
    assert result.most_similar_feature == "Exact Match Feature"
    assert result.feature_module == "exact/module.py"
    assert result.warning == "This is a duplicate"


def test_novelty_tracker_compute_stores_result():
    """Test that compute_novelty stores results in scores list."""
    tracker = NoveltyTracker(backend=MockSimilarityBackend())

    assert len(tracker.scores) == 0

    tracker.compute_novelty({"agent-1": "A"}, round_num=1)
    assert len(tracker.scores) == 1

    tracker.compute_novelty({"agent-1": "B"}, round_num=2)
    assert len(tracker.scores) == 2

    assert tracker.scores[0].round_num == 1
    assert tracker.scores[1].round_num == 2


def test_novelty_tracker_most_similar_to_different_agent():
    """Test that most_similar_to can be a different agent."""
    mock_backend = MockSimilarityBackend()
    tracker = NoveltyTracker(backend=mock_backend)

    # Round 1 - two agents
    tracker.compute_novelty({"agent-1": "Idea A", "agent-2": "Idea B"}, round_num=1)
    tracker.add_to_history({"agent-1": "Idea A", "agent-2": "Idea B"})

    # Round 2 - agent-1 proposes something similar to agent-2's idea
    mock_backend.set_similarity("Similar to B", "Idea A", 0.20)
    mock_backend.set_similarity("Similar to B", "Idea B", 0.85)

    result = tracker.compute_novelty({"agent-1": "Similar to B"}, round_num=2)

    score = result.details["agent-1"]
    assert score.most_similar_to == "agent-2"
    assert score.max_similarity == pytest.approx(0.85, abs=0.01)


def test_codebase_novelty_checker_case_insensitive_matching():
    """Test that feature matching is case-insensitive."""
    context = """
| Feature | Module |
|---------|--------|
| Dashboard | ui/dashboard |
"""

    mock_backend = MockSimilarityBackend()
    mock_backend.default_similarity = 0.10
    checker = CodebaseNoveltyChecker(context, backend=mock_backend)

    # Lowercase "dashboard" should still match
    result = checker.check_proposal("improve the DASHBOARD", "agent-1")

    # Should get name match boost
    assert result.max_similarity >= 0.70


def test_novelty_tracker_zero_similarity():
    """Test novelty with zero similarity (maximally different)."""
    mock_backend = MockSimilarityBackend()
    mock_backend.default_similarity = 0.0
    tracker = NoveltyTracker(backend=mock_backend)

    tracker.compute_novelty({"agent-1": "A"}, round_num=1)
    tracker.add_to_history({"agent-1": "A"})

    result = tracker.compute_novelty({"agent-1": "B"}, round_num=2)

    # novelty = 1 - 0 = 1.0
    assert result.per_agent_novelty["agent-1"] == 1.0


def test_novelty_tracker_perfect_similarity():
    """Test novelty with perfect similarity (identical proposals)."""
    mock_backend = MockSimilarityBackend()
    tracker = NoveltyTracker(backend=mock_backend)

    tracker.compute_novelty({"agent-1": "Exact same"}, round_num=1)
    tracker.add_to_history({"agent-1": "Exact same"})

    mock_backend.set_similarity("Exact same", "Exact same", 1.0)
    result = tracker.compute_novelty({"agent-1": "Exact same"}, round_num=2)

    # novelty = 1 - 1.0 = 0.0
    assert result.per_agent_novelty["agent-1"] == 0.0
    assert "agent-1" in result.low_novelty_agents


def test_codebase_novelty_checker_key_terms_limit():
    """Test that key term extraction limits to 50 terms."""
    # Create context with many capitalized terms
    terms = [f"Feature{i} System" for i in range(100)]
    context = " ".join(terms)

    checker = CodebaseNoveltyChecker(context, backend=MockSimilarityBackend())

    # Should extract at most 50 key terms
    extracted_terms = [f["name"] for f in checker.features if f["section"] == "extracted"]
    assert len(extracted_terms) <= 50


def test_novelty_result_low_novelty_agents_order():
    """Test that low_novelty_agents maintains list order."""
    result = NoveltyResult(
        round_num=1,
        low_novelty_agents=["agent-3", "agent-1", "agent-2"],
    )

    # Order should be preserved
    assert result.low_novelty_agents == ["agent-3", "agent-1", "agent-2"]


def test_codebase_novelty_checker_empty_proposal():
    """Test checker handles empty proposal string."""
    context = """
| Feature | Module |
|---------|--------|
| Feature A | module_a |
"""

    mock_backend = MockSimilarityBackend()
    checker = CodebaseNoveltyChecker(context, backend=mock_backend)

    result = checker.check_proposal("", "agent-1")

    # Empty proposal should still work
    assert isinstance(result, CodebaseNoveltyResult)
    assert result.agent == "agent-1"


def test_novelty_tracker_get_agent_trajectory_partial_participation():
    """Test trajectory for agent that doesn't participate in all rounds."""
    mock_backend = MockSimilarityBackend()
    tracker = NoveltyTracker(backend=mock_backend)

    # Round 1 - agent-1 only
    tracker.compute_novelty({"agent-1": "A"}, round_num=1)
    tracker.add_to_history({"agent-1": "A"})

    # Round 2 - agent-2 only
    tracker.compute_novelty({"agent-2": "B"}, round_num=2)
    tracker.add_to_history({"agent-2": "B"})

    # Round 3 - agent-1 returns
    mock_backend.default_similarity = 0.10
    tracker.compute_novelty({"agent-1": "C"}, round_num=3)

    trajectory = tracker.get_agent_novelty_trajectory("agent-1")

    # Should be [1.0, 0.0, 0.90] - missing round shows as 0.0
    assert len(trajectory) == 3
    assert trajectory[0] == 1.0
    assert trajectory[1] == 0.0  # Didn't participate
    assert trajectory[2] == pytest.approx(0.90, abs=0.01)


def test_codebase_novelty_checker_bullet_without_colon():
    """Test bullet point extraction handles items without colons."""
    context = """
- Feature A: Description
- Just a feature name
- Feature B: Another description
"""

    checker = CodebaseNoveltyChecker(context, backend=MockSimilarityBackend())

    # Should only extract items with colons
    feature_names = [f["name"] for f in checker.features]
    assert "Feature A" in feature_names
    assert "Feature B" in feature_names
    # "Just a feature name" should not be extracted (no colon)


def test_novelty_tracker_summary_low_novelty_agents_by_round():
    """Test summary includes low_novelty_agents_by_round mapping."""
    mock_backend = MockSimilarityBackend()
    tracker = NoveltyTracker(backend=mock_backend, low_novelty_threshold=0.15)

    # Round 1
    tracker.compute_novelty({"agent-1": "A"}, round_num=1)
    tracker.add_to_history({"agent-1": "A"})

    # Round 2 - low novelty
    mock_backend.set_similarity("A2", "A", 0.90)
    tracker.compute_novelty({"agent-1": "A2"}, round_num=2)
    tracker.add_to_history({"agent-1": "A2"})

    # Round 3 - OK novelty
    mock_backend.set_similarity("B", "A", 0.10)
    mock_backend.set_similarity("B", "A2", 0.10)
    tracker.compute_novelty({"agent-1": "B"}, round_num=3)

    summary = tracker.get_debate_novelty_summary()

    # Should have mapping only for round 2
    assert 2 in summary["low_novelty_agents_by_round"]
    assert summary["low_novelty_agents_by_round"][2] == ["agent-1"]
    assert 1 not in summary["low_novelty_agents_by_round"]
    assert 3 not in summary["low_novelty_agents_by_round"]
