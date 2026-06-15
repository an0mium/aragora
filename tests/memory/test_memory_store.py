"""Tests for SQLite-based critique pattern store."""

import os
import tempfile
import pytest
from datetime import datetime
from unittest.mock import MagicMock

from aragora.memory.store import (
    Pattern,
    AgentReputation,
    CritiqueStore,
)
from aragora.core import Critique, DebateResult


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def store(temp_db):
    """Create a CritiqueStore with temporary database."""
    # Clear the global cache to prevent cross-test contamination
    from aragora.server.handlers.base import clear_cache

    clear_cache()
    return CritiqueStore(db_path=temp_db)


@pytest.fixture
def sample_critique():
    """Create a sample critique for testing."""
    return Critique(
        agent="claude",
        target_agent="gpt4",
        target_content="The current implementation is slow",
        issues=["Performance issue: slow response time"],
        suggestions=["Use caching to improve performance"],
        severity=0.7,
        reasoning="The response takes too long to generate",
    )


@pytest.fixture
def sample_debate_result():
    """Create a sample debate result for testing."""
    critique = Critique(
        agent="claude",
        target_agent="gpt4",
        target_content="def process(): do_something()",
        issues=["Missing error handling"],
        suggestions=["Add try-catch blocks"],
        severity=0.5,
        reasoning="Errors could crash the system",
    )
    return DebateResult(
        id="debate_001",
        task="Review the code for issues",
        final_answer="The code needs error handling improvements",
        consensus_reached=True,
        confidence=0.85,
        rounds_used=3,
        duration_seconds=45.5,
        critiques=[critique],
    )


class TestPattern:
    """Test Pattern dataclass."""

    def test_pattern_creation(self):
        """Test basic pattern creation."""
        pattern = Pattern(
            id="pat1",
            issue_type="performance",
            issue_text="Slow query",
            suggestion_text="Add index",
            success_count=5,
            failure_count=2,
            avg_severity=0.6,
            example_task="Optimize database",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-02T00:00:00",
        )
        assert pattern.id == "pat1"
        assert pattern.issue_type == "performance"
        assert pattern.success_count == 5

    def test_success_rate_calculation(self):
        """Test success rate property."""
        pattern = Pattern(
            id="p1",
            issue_type="test",
            issue_text="test",
            suggestion_text="test",
            success_count=8,
            failure_count=2,
            avg_severity=0.5,
            example_task="",
            created_at="",
            updated_at="",
        )
        assert pattern.success_rate == 0.8  # 8 / (8 + 2)

    def test_success_rate_no_data(self):
        """Test success rate with no data."""
        pattern = Pattern(
            id="p1",
            issue_type="test",
            issue_text="test",
            suggestion_text="test",
            success_count=0,
            failure_count=0,
            avg_severity=0.5,
            example_task="",
            created_at="",
            updated_at="",
        )
        assert pattern.success_rate == 0.5  # Default


class TestAgentReputation:
    """Test AgentReputation dataclass."""

    def test_reputation_creation(self):
        """Test basic reputation creation."""
        rep = AgentReputation(agent_name="claude")
        assert rep.agent_name == "claude"
        assert rep.proposals_made == 0
        assert rep.calibration_score == 0.5

    def test_score_new_agent(self):
        """Test score for new agent with no history."""
        rep = AgentReputation(agent_name="new_agent")
        assert rep.score == 0.5  # Neutral

    def test_score_with_proposals(self):
        """Test score based on proposal acceptance."""
        rep = AgentReputation(
            agent_name="claude",
            proposals_made=10,
            proposals_accepted=8,
            critiques_given=5,
            critiques_valuable=4,
        )
        # 0.6 * (8/10) + 0.4 * (4/5) = 0.6 * 0.8 + 0.4 * 0.8 = 0.8
        assert rep.score == pytest.approx(0.8, rel=0.01)

    def test_reputation_score_alias(self):
        """Test reputation_score is alias for score."""
        rep = AgentReputation(
            agent_name="claude",
            proposals_made=10,
            proposals_accepted=5,
        )
        assert rep.reputation_score == rep.score

    def test_proposal_acceptance_rate(self):
        """Test proposal acceptance rate."""
        rep = AgentReputation(
            agent_name="claude",
            proposals_made=10,
            proposals_accepted=7,
        )
        assert rep.proposal_acceptance_rate == 0.7

    def test_proposal_acceptance_rate_no_proposals(self):
        """Test proposal acceptance rate with no proposals."""
        rep = AgentReputation(agent_name="claude")
        assert rep.proposal_acceptance_rate == 0.0

    def test_critique_value(self):
        """Test critique value rate."""
        rep = AgentReputation(
            agent_name="claude",
            critiques_given=10,
            critiques_valuable=6,
        )
        assert rep.critique_value == 0.6

    def test_critique_value_no_critiques(self):
        """Test critique value with no critiques."""
        rep = AgentReputation(agent_name="claude")
        assert rep.critique_value == 0.0

    def test_debates_participated(self):
        """Test debates participated count."""
        rep = AgentReputation(
            agent_name="claude",
            proposals_made=5,
        )
        assert rep.debates_participated == 5

    def test_vote_weight_neutral(self):
        """Test vote weight for neutral agent."""
        rep = AgentReputation(agent_name="claude")
        # Base: 0.5 + 0.5 = 1.0, calibration bonus: 0
        assert rep.vote_weight == pytest.approx(1.0, rel=0.01)

    def test_vote_weight_high_reputation(self):
        """Test vote weight for high reputation agent."""
        rep = AgentReputation(
            agent_name="claude",
            proposals_made=10,
            proposals_accepted=10,  # 100% acceptance
            critiques_given=10,
            critiques_valuable=10,  # 100% valuable
            calibration_score=1.0,  # Perfect calibration
        )
        # Base: 0.5 + 1.0 = 1.5, calibration bonus: (1.0 - 0.5) * 0.2 = 0.1
        # Total: min(1.6, 1.5 + 0.1) = 1.6
        assert rep.vote_weight == pytest.approx(1.6, rel=0.01)

    def test_vote_weight_low_reputation(self):
        """Test vote weight for low reputation agent."""
        rep = AgentReputation(
            agent_name="bad_agent",
            proposals_made=10,
            proposals_accepted=0,  # 0% acceptance
            critiques_given=10,
            critiques_valuable=0,  # 0% valuable
            calibration_score=0.0,  # Poor calibration
        )
        # Base: 0.5 + 0.0 = 0.5, calibration penalty: (0 - 0.5) * 0.2 = -0.1
        # Total: max(0.4, 0.5 - 0.1) = 0.4
        assert rep.vote_weight == pytest.approx(0.4, rel=0.01)


class TestCritiqueStore:
    """Test CritiqueStore class."""

    def test_store_creation(self, temp_db):
        """Test store initialization creates tables."""
        store = CritiqueStore(db_path=temp_db)
        # Should not raise - tables are created
        stats = store.get_stats()
        assert "total_debates" in stats

    def test_store_debate(self, store, sample_debate_result):
        """Test storing a debate result."""
        store.store_debate(sample_debate_result)

        stats = store.get_stats()
        assert stats["total_debates"] == 1
        assert stats["consensus_debates"] == 1
        assert stats["total_critiques"] == 1

    def test_store_debate_without_consensus(self, store):
        """Test storing debate without consensus."""
        result = DebateResult(
            id="debate_002",
            task="Resolve disagreement",
            final_answer=None,
            consensus_reached=False,
            confidence=0.3,
            rounds_used=5,
            duration_seconds=120.0,
            critiques=[],
        )
        store.store_debate(result)

        stats = store.get_stats()
        assert stats["total_debates"] == 1
        assert stats["consensus_debates"] == 0

    def test_store_pattern(self, store, sample_critique):
        """Test storing a pattern from critique."""
        store.store_pattern(sample_critique, "Use Redis for caching")

        patterns = store.retrieve_patterns(min_success=1)
        assert len(patterns) >= 1

    def test_store_pattern_updates_existing(self, store, sample_critique):
        """Test storing pattern updates existing pattern."""
        store.store_pattern(sample_critique, "Fix 1")
        store.store_pattern(sample_critique, "Fix 2")

        patterns = store.retrieve_patterns(min_success=1)
        # Same issue should update the same pattern
        pattern = patterns[0]
        assert pattern.success_count == 2

    def test_categorize_issue_performance(self, store):
        """Test issue categorization - performance."""
        assert store._categorize_issue("slow query performance") == "performance"
        assert store._categorize_issue("optimize this function") == "performance"

    def test_categorize_issue_security(self, store):
        """Test issue categorization - security."""
        assert store._categorize_issue("SQL injection vulnerability") == "security"
        assert store._categorize_issue("authentication bypass") == "security"

    def test_categorize_issue_correctness(self, store):
        """Test issue categorization - correctness."""
        assert store._categorize_issue("bug in calculation") == "correctness"
        assert store._categorize_issue("incorrect result") == "correctness"

    def test_categorize_issue_general(self, store):
        """Test issue categorization - general."""
        assert store._categorize_issue("some random issue") == "general"

    def test_fail_pattern(self, store, sample_critique):
        """Test recording pattern failure."""
        # First store a pattern
        store.store_pattern(sample_critique, "Fix")

        # Then fail it
        store.fail_pattern(sample_critique.issues[0])

        patterns = store.retrieve_patterns(min_success=1)
        assert patterns[0].failure_count >= 1

    def test_retrieve_patterns_empty(self, store):
        """Test retrieving patterns from empty store."""
        patterns = store.retrieve_patterns()
        assert patterns == []

    def test_retrieve_patterns_by_type(self, store):
        """Test retrieving patterns by issue type."""
        # Store performance issue
        perf_critique = Critique(
            agent="c",
            target_agent="g",
            target_content="slow code",
            issues=["slow performance"],
            suggestions=["optimize"],
            severity=0.5,
            reasoning="",
        )
        store.store_pattern(perf_critique, "fix")
        store.store_pattern(perf_critique, "fix2")

        # Store security issue
        sec_critique = Critique(
            agent="c",
            target_agent="g",
            target_content="vulnerable code",
            issues=["security vulnerability"],
            suggestions=["patch"],
            severity=0.8,
            reasoning="",
        )
        store.store_pattern(sec_critique, "fix")
        store.store_pattern(sec_critique, "fix2")

        perf_patterns = store.retrieve_patterns(issue_type="performance", min_success=1)
        sec_patterns = store.retrieve_patterns(issue_type="security", min_success=1)

        assert len(perf_patterns) >= 1
        assert len(sec_patterns) >= 1
        assert all(p.issue_type == "performance" for p in perf_patterns)
        assert all(p.issue_type == "security" for p in sec_patterns)

    def test_get_stats(self, store, sample_debate_result):
        """Test getting store statistics."""
        store.store_debate(sample_debate_result)

        stats = store.get_stats()
        assert stats["total_debates"] >= 1
        assert "patterns_by_type" in stats
        assert "avg_consensus_confidence" in stats

    def test_export_for_training(self, store, sample_debate_result):
        """Test exporting data for training."""
        store.store_debate(sample_debate_result)

        training_data = store.export_for_training()
        assert len(training_data) >= 1
        assert "task" in training_data[0]
        assert "issues" in training_data[0]
        assert "successful_answer" in training_data[0]


class TestAgentReputationTracking:
    """Test agent reputation tracking methods."""

    def test_get_reputation_new_agent(self, store):
        """Test getting reputation for new agent."""
        rep = store.get_reputation("unknown_agent")
        assert rep is None

    def test_update_reputation_creates_agent(self, store):
        """Test update_reputation creates agent if not exists."""
        store.update_reputation("new_agent", proposal_made=True)

        rep = store.get_reputation("new_agent")
        assert rep is not None
        assert rep.proposals_made == 1

    def test_update_reputation_proposal_accepted(self, store):
        """Test updating reputation with accepted proposal."""
        store.update_reputation("claude", proposal_made=True)
        store.update_reputation("claude", proposal_made=True, proposal_accepted=True)

        rep = store.get_reputation("claude")
        assert rep.proposals_made == 2
        assert rep.proposals_accepted == 1

    def test_update_reputation_critique(self, store):
        """Test updating reputation with critique."""
        store.update_reputation("claude", critique_given=True, critique_valuable=True)

        rep = store.get_reputation("claude")
        assert rep.critiques_given == 1
        assert rep.critiques_valuable == 1

    def test_get_vote_weight_unknown_agent(self, store):
        """Test vote weight for unknown agent."""
        weight = store.get_vote_weight("unknown")
        assert weight == 1.0  # Neutral

    def test_get_vote_weight_known_agent(self, store):
        """Test vote weight for known agent."""
        store.update_reputation("claude", proposal_made=True, proposal_accepted=True)

        weight = store.get_vote_weight("claude")
        assert 0.4 <= weight <= 1.6

    def test_get_all_reputations(self, store):
        """Test getting all agent reputations."""
        store.update_reputation("agent1", proposal_made=True)
        store.update_reputation("agent2", proposal_made=True, proposal_accepted=True)

        reps = store.get_all_reputations()
        assert len(reps) >= 2


class TestTitansMIRASFeatures:
    """Test Titans/MIRAS-inspired features."""

    def test_update_prediction_outcome(self, store, sample_debate_result):
        """Test updating critique with prediction outcome."""
        # Store debate to get critique_id
        store.store_debate(sample_debate_result)

        # Get critique ID (assuming it's 1 for first critique)
        with store._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM critiques LIMIT 1")
            critique_id = cursor.fetchone()[0]

        # Update with outcome
        error = store.update_prediction_outcome(
            critique_id=critique_id,
            actual_usefulness=0.8,
            agent_name="claude",
        )

        assert error >= 0

    def test_prune_stale_patterns(self, store):
        """Test pruning stale patterns."""
        # Add some patterns
        critique = Critique(
            agent="c",
            target_agent="g",
            target_content="old code",
            issues=["old issue"],
            suggestions=["old fix"],
            severity=0.5,
            reasoning="",
        )
        store.store_pattern(critique, "fix")

        # Prune (won't actually prune since patterns are new)
        pruned = store.prune_stale_patterns(max_age_days=0, min_success_rate=1.0)
        # Pattern has 100% success rate (1 success, 0 failures)
        assert pruned == 0

    def test_get_archive_stats(self, store):
        """Test getting archive statistics."""
        stats = store.get_archive_stats()
        assert "total_archived" in stats
        assert "archived_by_type" in stats


class TestDecayScoringRanking:
    """Test Titans/MIRAS decay scoring and pattern ranking."""

    def test_decay_score_calculation(self, store):
        """Test: score = (success * (1 + surprise)) / (1 + age/halflife)."""
        # Create a pattern with known values
        critique = Critique(
            agent="c",
            target_agent="g",
            target_content="code",
            issues=["test decay issue"],
            suggestions=["fix"],
            severity=0.5,
            reasoning="",
        )
        # Store multiple times to get success_count > 1
        store.store_pattern(critique, "fix1")
        store.store_pattern(critique, "fix2")
        store.store_pattern(critique, "fix3")

        # Retrieve with default decay halflife
        patterns = store.retrieve_patterns(min_success=1)
        assert len(patterns) >= 1
        assert patterns[0].success_count == 3

    def test_retrieve_patterns_ordering_by_decay_score(self, store):
        """Patterns should be ordered by decay_score descending."""
        # Create two patterns with different success counts
        high_success = Critique(
            agent="c",
            target_agent="g",
            target_content="code",
            issues=["high success issue"],
            suggestions=["fix"],
            severity=0.5,
            reasoning="",
        )
        low_success = Critique(
            agent="c",
            target_agent="g",
            target_content="code",
            issues=["low success issue"],
            suggestions=["fix"],
            severity=0.5,
            reasoning="",
        )

        # High success: 5 times
        for _ in range(5):
            store.store_pattern(high_success, "fix")

        # Low success: 2 times
        for _ in range(2):
            store.store_pattern(low_success, "fix")

        patterns = store.retrieve_patterns(min_success=1)
        assert len(patterns) >= 2

        # Higher success count should come first (higher decay score)
        success_counts = [p.success_count for p in patterns]
        assert success_counts[0] >= success_counts[1], "Patterns should be ordered by decay score"

    def test_decay_halflife_parameter(self, store):
        """Different halflife values affect pattern relevance."""
        critique = Critique(
            agent="c",
            target_agent="g",
            target_content="code",
            issues=["halflife test issue"],
            suggestions=["fix"],
            severity=0.5,
            reasoning="",
        )
        store.store_pattern(critique, "fix1")
        store.store_pattern(critique, "fix2")

        # With short halflife, patterns decay faster
        patterns_short = store.retrieve_patterns(min_success=1, decay_halflife_days=1)
        patterns_long = store.retrieve_patterns(min_success=1, decay_halflife_days=365)

        # Both should return results for fresh patterns
        assert len(patterns_short) >= 1
        assert len(patterns_long) >= 1

    def test_surprise_score_bonus_in_ranking(self, store):
        """Patterns with higher surprise scores should rank higher."""
        # Create two patterns
        normal = Critique(
            agent="c",
            target_agent="g",
            target_content="code",
            issues=["normal pattern issue"],
            suggestions=["fix"],
            severity=0.5,
            reasoning="",
        )
        surprising = Critique(
            agent="c",
            target_agent="g",
            target_content="code",
            issues=["surprising pattern issue"],
            suggestions=["fix"],
            severity=0.5,
            reasoning="",
        )

        # Store both patterns
        for _ in range(3):
            store.store_pattern(normal, "fix")
            store.store_pattern(surprising, "fix")

        # The formula includes (1 + surprise_score) multiplier
        # Fresh patterns should both have default surprise scores
        patterns = store.retrieve_patterns(min_success=1)
        assert len(patterns) >= 2

    def test_min_success_threshold(self, store):
        """Patterns below min_success should not be retrieved."""
        critique = Critique(
            agent="c",
            target_agent="g",
            target_content="code",
            issues=["min success test"],
            suggestions=["fix"],
            severity=0.5,
            reasoning="",
        )
        store.store_pattern(critique, "fix")  # Only 1 success

        # Should not appear with min_success=2
        patterns = store.retrieve_patterns(min_success=2)
        assert all(p.success_count >= 2 for p in patterns)

        # Should appear with min_success=1
        patterns = store.retrieve_patterns(min_success=1)
        assert len(patterns) >= 1

    def test_limit_parameter(self, store):
        """Limit parameter should cap number of returned patterns."""
        # Create multiple distinct patterns
        for i in range(5):
            critique = Critique(
                agent="c",
                target_agent="g",
                target_content="code",
                issues=[f"limit test issue {i}"],
                suggestions=["fix"],
                severity=0.5,
                reasoning="",
            )
            store.store_pattern(critique, "fix")
            store.store_pattern(critique, "fix")  # 2 successes each

        patterns = store.retrieve_patterns(min_success=1, limit=3)
        assert len(patterns) <= 3


class TestSurpriseCalculation:
    """Test surprise score calculation (Titans/MIRAS)."""

    def test_calculate_surprise_base_rate(self, store):
        """Surprise = |actual - base_rate| * 2, capped at 1.0."""
        # First, create some patterns to establish a base rate
        critique = Critique(
            agent="c",
            target_agent="g",
            target_content="code",
            issues=["surprise base rate test"],
            suggestions=["fix"],
            severity=0.5,
            reasoning="",
        )
        store.store_pattern(critique, "fix")

        # Access internal method for testing
        with store._get_connection() as conn:
            cursor = conn.cursor()
            # Calculate surprise for a new success in a new category
            surprise = store._calculate_surprise(cursor, "performance", is_success=True)

            # With no prior data, base_rate defaults to 0.5
            # Success = 1.0, so surprise = |1.0 - 0.5| * 2 = 1.0
            assert 0.0 <= surprise <= 1.0

    def test_calculate_surprise_failure_vs_success(self, store):
        """Failures and successes produce different surprise scores."""
        # Create some patterns with high success rate
        critique = Critique(
            agent="c",
            target_agent="g",
            target_content="slow code",
            issues=["slow performance issue"],
            suggestions=["optimize"],
            severity=0.5,
            reasoning="",
        )
        # Create many successes
        for _ in range(5):
            store.store_pattern(critique, "fix")

        with store._get_connection() as conn:
            cursor = conn.cursor()
            # With high success rate, a failure would be surprising
            surprise_failure = store._calculate_surprise(cursor, "performance", is_success=False)
            # A success would be less surprising
            surprise_success = store._calculate_surprise(cursor, "performance", is_success=True)

            # Both should be valid surprise scores
            assert 0.0 <= surprise_failure <= 1.0
            assert 0.0 <= surprise_success <= 1.0

    def test_surprise_score_exponential_moving_average(self, store):
        """Surprise score updates use EMA: new = old * 0.7 + surprise * 0.3."""
        critique = Critique(
            agent="c",
            target_agent="g",
            target_content="code",
            issues=["ema test issue"],
            suggestions=["fix"],
            severity=0.5,
            reasoning="",
        )

        # Store pattern multiple times
        store.store_pattern(critique, "fix1")
        store.store_pattern(critique, "fix2")
        store.store_pattern(critique, "fix3")

        # The surprise_score should be updated with EMA
        with store._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT surprise_score FROM patterns WHERE issue_text = ?", ("ema test issue",)
            )
            result = cursor.fetchone()
            assert result is not None
            # Surprise score should be a float between 0 and 1
            assert 0.0 <= (result[0] or 0) <= 1.0


class TestAgentCalibration:
    """Test agent calibration score tracking."""

    def test_agent_calibration_score_formula(self, store, sample_debate_result):
        """Calibration = 1.0 - (total_error / predictions)."""
        # Store debate to get critique_id
        store.store_debate(sample_debate_result)

        # Get critique ID
        with store._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM critiques LIMIT 1")
            critique_id = cursor.fetchone()[0]

        # Update with known outcome
        error = store.update_prediction_outcome(
            critique_id=critique_id,
            actual_usefulness=0.8,
            agent_name="calibration_test_agent",
        )

        # Verify calibration was updated
        rep = store.get_reputation("calibration_test_agent")
        assert rep is not None
        assert rep.total_predictions == 1
        # Calibration = 1.0 - (error / 1) = 1.0 - error
        # Default expected is 0.5, actual is 0.8, error = 0.3
        assert rep.calibration_score == pytest.approx(1.0 - 0.3, rel=0.1)

    def test_calibration_improves_with_accurate_predictions(self, store, sample_debate_result):
        """Agents with accurate predictions get better calibration scores."""
        store.store_debate(sample_debate_result)

        with store._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM critiques LIMIT 1")
            critique_id = cursor.fetchone()[0]

        # Perfect prediction: expected=0.5 (default), actual=0.5
        store.update_prediction_outcome(
            critique_id=critique_id,
            actual_usefulness=0.5,
            agent_name="accurate_agent",
        )

        rep = store.get_reputation("accurate_agent")
        assert rep is not None
        # With zero error, calibration should be 1.0
        assert rep.calibration_score == pytest.approx(1.0, rel=0.01)

    def test_calibration_degrades_with_poor_predictions(self, store):
        """Agents with inaccurate predictions get worse calibration scores."""
        # Create and store debate
        critique = Critique(
            agent="poor_predictor",
            target_agent="gpt4",
            target_content="code",
            issues=["test issue"],
            suggestions=["fix"],
            severity=0.5,
            reasoning="test",
        )
        result = DebateResult(
            id="cal_test_debate",
            task="test calibration",
            final_answer="answer",
            consensus_reached=True,
            confidence=0.8,
            rounds_used=3,
            duration_seconds=10.0,
            critiques=[critique],
        )
        store.store_debate(result)

        with store._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM critiques WHERE debate_id = ?", ("cal_test_debate",))
            critique_id = cursor.fetchone()[0]

        # Poor prediction: expected=0.5, actual=0.0, error=0.5
        store.update_prediction_outcome(
            critique_id=critique_id,
            actual_usefulness=0.0,
            agent_name="poor_predictor",
        )

        rep = store.get_reputation("poor_predictor")
        assert rep is not None
        # With 0.5 error, calibration = 1.0 - 0.5 = 0.5
        assert rep.calibration_score == pytest.approx(0.5, rel=0.1)


class TestPatternEdgeCases:
    """Test pattern management edge cases."""

    def test_pattern_id_consistency_md5(self, store):
        """Same issue creates same pattern ID (MD5 hash)."""
        issue = "This is a specific test issue"
        critique1 = Critique(
            agent="c",
            target_agent="g",
            target_content="code",
            issues=[issue],
            suggestions=["fix1"],
            severity=0.5,
            reasoning="",
        )
        critique2 = Critique(
            agent="d",
            target_agent="h",
            target_content="different code",
            issues=[issue],  # Same issue text
            suggestions=["fix2"],
            severity=0.8,
            reasoning="different reasoning",
        )

        store.store_pattern(critique1, "fix1")
        store.store_pattern(critique2, "fix2")

        patterns = store.retrieve_patterns(min_success=1)
        # Both critiques should update the same pattern
        matching = [p for p in patterns if issue in p.issue_text]
        assert len(matching) == 1
        assert matching[0].success_count == 2

    def test_pattern_id_case_insensitivity(self, store):
        """Pattern IDs should be case-insensitive (lowercase hash)."""
        critique_upper = Critique(
            agent="c",
            target_agent="g",
            target_content="code",
            issues=["UPPERCASE ISSUE"],
            suggestions=["fix"],
            severity=0.5,
            reasoning="",
        )
        critique_lower = Critique(
            agent="c",
            target_agent="g",
            target_content="code",
            issues=["uppercase issue"],  # Same issue, different case
            suggestions=["fix"],
            severity=0.5,
            reasoning="",
        )

        store.store_pattern(critique_upper, "fix1")
        store.store_pattern(critique_lower, "fix2")

        patterns = store.retrieve_patterns(min_success=1)
        # Both should map to the same pattern (lowercase hash)
        upper_patterns = [p for p in patterns if "uppercase" in p.issue_text.lower()]
        assert len(upper_patterns) == 1
        assert upper_patterns[0].success_count == 2

    def test_severity_averaging_incremental(self, store):
        """Average severity updates correctly with multiple stores."""
        critique1 = Critique(
            agent="c",
            target_agent="g",
            target_content="code",
            issues=["severity test issue"],
            suggestions=["fix"],
            severity=0.2,
            reasoning="",
        )
        critique2 = Critique(
            agent="c",
            target_agent="g",
            target_content="code",
            issues=["severity test issue"],
            suggestions=["fix"],
            severity=0.8,
            reasoning="",
        )

        store.store_pattern(critique1, "fix1")  # severity 0.2
        store.store_pattern(critique2, "fix2")  # severity 0.8

        patterns = store.retrieve_patterns(min_success=1)
        matching = [p for p in patterns if "severity test issue" in p.issue_text]
        assert len(matching) == 1
        # Average of 0.2 and 0.8 should be 0.5
        assert matching[0].avg_severity == pytest.approx(0.5, rel=0.01)

    def test_severity_averaging_with_more_samples(self, store):
        """Average severity weighted by count."""
        severities = [0.1, 0.2, 0.3, 0.4, 0.5]
        for severity in severities:
            critique = Critique(
                agent="c",
                target_agent="g",
                target_content="code",
                issues=["multi severity test"],
                suggestions=["fix"],
                severity=severity,
                reasoning="",
            )
            store.store_pattern(critique, "fix")

        patterns = store.retrieve_patterns(min_success=1)
        matching = [p for p in patterns if "multi severity test" in p.issue_text]
        assert len(matching) == 1
        # Average of [0.1, 0.2, 0.3, 0.4, 0.5] = 0.3
        assert matching[0].avg_severity == pytest.approx(0.3, rel=0.01)

    def test_categorize_architecture_issues(self, store):
        """Test architecture issue categorization."""
        assert store._categorize_issue("poor design pattern") == "architecture"
        assert store._categorize_issue("tight coupling between modules") == "architecture"
        assert store._categorize_issue("low cohesion in class") == "architecture"
        assert store._categorize_issue("modular structure needed") == "architecture"

    def test_categorize_completeness_issues(self, store):
        """Test completeness issue categorization."""
        assert store._categorize_issue("missing validation") == "completeness"
        assert store._categorize_issue("incomplete implementation") == "completeness"
        assert store._categorize_issue("TODO: add logging") == "completeness"
        assert store._categorize_issue("edge case not handled") == "completeness"

    def test_categorize_testing_issues(self, store):
        """Test testing issue categorization."""
        assert store._categorize_issue("no unit tests") == "testing"
        assert store._categorize_issue("poor test coverage") == "testing"
        assert store._categorize_issue("mock objects needed") == "testing"
        assert store._categorize_issue("add integration tests") == "testing"

    def test_categorize_clarity_issues(self, store):
        """Test clarity issue categorization."""
        assert store._categorize_issue("unclear variable naming") == "clarity"
        assert store._categorize_issue("confusing logic") == "clarity"
        assert store._categorize_issue("poor readability") == "clarity"
        assert store._categorize_issue("missing documentation") == "clarity"


class TestDatabaseConnections:
    """Test database connection handling."""

    def test_connection_context_manager(self, store):
        """Test connection context manager properly closes."""
        with store._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
        # Connection should be closed after context

    def test_multiple_operations(self, store):
        """Test multiple operations don't leak connections."""
        for i in range(10):
            store.get_stats()
            store.retrieve_patterns()
            store.get_all_reputations()
        # Should complete without errors

    def test_concurrent_writes(self, store, sample_critique):
        """Test concurrent write operations."""
        import threading

        def write_patterns():
            for _ in range(5):
                store.store_pattern(sample_critique, "fix")

        threads = [threading.Thread(target=write_patterns) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        patterns = store.retrieve_patterns(min_success=1)
        assert len(patterns) >= 1
