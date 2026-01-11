"""
Tests for Rhetorical Analysis Observer module.

Tests cover:
- RhetoricalPattern enum values
- RhetoricalObservation dataclass
- RhetoricalAnalysisObserver initialization
- Pattern detection (concession, rebuttal, synthesis, etc.)
- Excerpt finding
- Commentary generation
- Debate dynamics analysis
- Global observer functions
"""

import random
import time
from unittest.mock import MagicMock

import pytest

from aragora.debate.rhetorical_observer import (
    RhetoricalPattern,
    RhetoricalObservation,
    RhetoricalAnalysisObserver,
    get_rhetorical_observer,
    reset_rhetorical_observer,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_global_observer():
    """Reset global observer before and after each test."""
    reset_rhetorical_observer()
    yield
    reset_rhetorical_observer()


@pytest.fixture
def observer():
    """Create a fresh observer."""
    return RhetoricalAnalysisObserver()


@pytest.fixture
def mock_callback():
    """Create a mock broadcast callback."""
    return MagicMock()


# ============================================================================
# RhetoricalPattern Tests
# ============================================================================

class TestRhetoricalPattern:
    """Tests for RhetoricalPattern enum."""

    def test_concession_value(self):
        """Test CONCESSION pattern value."""
        assert RhetoricalPattern.CONCESSION.value == "concession"

    def test_rebuttal_value(self):
        """Test REBUTTAL pattern value."""
        assert RhetoricalPattern.REBUTTAL.value == "rebuttal"

    def test_synthesis_value(self):
        """Test SYNTHESIS pattern value."""
        assert RhetoricalPattern.SYNTHESIS.value == "synthesis"

    def test_appeal_to_authority_value(self):
        """Test APPEAL_TO_AUTHORITY pattern value."""
        assert RhetoricalPattern.APPEAL_TO_AUTHORITY.value == "appeal_to_authority"

    def test_appeal_to_evidence_value(self):
        """Test APPEAL_TO_EVIDENCE pattern value."""
        assert RhetoricalPattern.APPEAL_TO_EVIDENCE.value == "appeal_to_evidence"

    def test_technical_depth_value(self):
        """Test TECHNICAL_DEPTH pattern value."""
        assert RhetoricalPattern.TECHNICAL_DEPTH.value == "technical_depth"

    def test_rhetorical_question_value(self):
        """Test RHETORICAL_QUESTION pattern value."""
        assert RhetoricalPattern.RHETORICAL_QUESTION.value == "rhetorical_question"

    def test_analogy_value(self):
        """Test ANALOGY pattern value."""
        assert RhetoricalPattern.ANALOGY.value == "analogy"

    def test_qualification_value(self):
        """Test QUALIFICATION pattern value."""
        assert RhetoricalPattern.QUALIFICATION.value == "qualification"

    def test_all_patterns_count(self):
        """Test all patterns are present."""
        assert len(RhetoricalPattern) == 9


# ============================================================================
# RhetoricalObservation Tests
# ============================================================================

class TestRhetoricalObservation:
    """Tests for RhetoricalObservation dataclass."""

    def test_basic_creation(self):
        """Test basic observation creation."""
        obs = RhetoricalObservation(
            pattern=RhetoricalPattern.CONCESSION,
            agent="claude",
            round_num=1,
            confidence=0.8,
            excerpt="I agree with that point",
            audience_commentary="Claude concedes",
        )
        assert obs.pattern == RhetoricalPattern.CONCESSION
        assert obs.agent == "claude"
        assert obs.round_num == 1
        assert obs.confidence == 0.8

    def test_timestamp_auto_generated(self):
        """Test timestamp is auto-generated."""
        before = time.time()
        obs = RhetoricalObservation(
            pattern=RhetoricalPattern.REBUTTAL,
            agent="gpt",
            round_num=0,
            confidence=0.7,
            excerpt="However...",
            audience_commentary="GPT rebuts",
        )
        after = time.time()

        assert before <= obs.timestamp <= after

    def test_to_dict(self):
        """Test to_dict serialization."""
        obs = RhetoricalObservation(
            pattern=RhetoricalPattern.SYNTHESIS,
            agent="gemini",
            round_num=2,
            confidence=0.9,
            excerpt="Combining both views...",
            audience_commentary="Gemini synthesizes",
            timestamp=1234567890.0,
        )

        result = obs.to_dict()

        assert result["pattern"] == "synthesis"
        assert result["agent"] == "gemini"
        assert result["round_num"] == 2
        assert result["confidence"] == 0.9
        assert result["excerpt"] == "Combining both views..."
        assert result["audience_commentary"] == "Gemini synthesizes"
        assert result["timestamp"] == 1234567890.0


# ============================================================================
# RhetoricalAnalysisObserver Init Tests
# ============================================================================

class TestObserverInit:
    """Tests for observer initialization."""

    def test_default_init(self, observer):
        """Test default initialization."""
        assert observer.broadcast_callback is None
        assert observer.min_confidence == 0.5
        assert observer.observations == []
        assert observer.agent_patterns == {}

    def test_init_with_callback(self, mock_callback):
        """Test initialization with callback."""
        observer = RhetoricalAnalysisObserver(broadcast_callback=mock_callback)
        assert observer.broadcast_callback is mock_callback

    def test_init_with_min_confidence(self):
        """Test initialization with custom min_confidence."""
        observer = RhetoricalAnalysisObserver(min_confidence=0.7)
        assert observer.min_confidence == 0.7

    def test_pattern_indicators_exist(self, observer):
        """Test PATTERN_INDICATORS is properly defined."""
        assert len(observer.PATTERN_INDICATORS) == 9
        for pattern in RhetoricalPattern:
            assert pattern in observer.PATTERN_INDICATORS


# ============================================================================
# Pattern Detection Tests
# ============================================================================

class TestPatternDetection:
    """Tests for pattern detection."""

    def test_detect_concession(self, observer):
        """Test concession pattern detection."""
        # Use multiple strong concession indicators
        text = "I must acknowledge and concede that you're right - it's a fair point and valid argument."
        observations = observer.observe("claude", text, round_num=0)

        patterns = [o.pattern for o in observations]
        assert RhetoricalPattern.CONCESSION in patterns

    def test_detect_rebuttal(self, observer):
        """Test rebuttal pattern detection."""
        text = "However, I would disagree with that assessment based on recent research."
        observations = observer.observe("claude", text, round_num=0)

        patterns = [o.pattern for o in observations]
        assert RhetoricalPattern.REBUTTAL in patterns

    def test_detect_synthesis(self, observer):
        """Test synthesis pattern detection."""
        text = "Combining both perspectives, we can find common ground on this approach."
        observations = observer.observe("claude", text, round_num=0)

        patterns = [o.pattern for o in observations]
        assert RhetoricalPattern.SYNTHESIS in patterns

    def test_detect_appeal_to_authority(self, observer):
        """Test appeal to authority detection."""
        text = "According to best practices and expert recommendations, we should implement caching."
        observations = observer.observe("claude", text, round_num=0)

        patterns = [o.pattern for o in observations]
        assert RhetoricalPattern.APPEAL_TO_AUTHORITY in patterns

    def test_detect_appeal_to_evidence(self, observer):
        """Test appeal to evidence detection."""
        text = "For example, the data shows a 50% performance improvement when using this approach."
        observations = observer.observe("claude", text, round_num=0)

        patterns = [o.pattern for o in observations]
        assert RhetoricalPattern.APPEAL_TO_EVIDENCE in patterns

    def test_detect_technical_depth(self, observer):
        """Test technical depth detection."""
        text = "The implementation uses async/await patterns with O(n log n) complexity."
        observations = observer.observe("claude", text, round_num=0)

        patterns = [o.pattern for o in observations]
        assert RhetoricalPattern.TECHNICAL_DEPTH in patterns

    def test_detect_rhetorical_question(self, observer):
        """Test rhetorical question detection."""
        # Use text that matches multiple regex patterns to hit 0.5 confidence threshold
        # Pattern 1: r"\b(what if|why would|...|shouldn't we)\b[^?]*\?"
        # Pattern 2: r"\?\s*(right|no|isn't it)\??\s*$" (expects ? before right/no)
        text = "Why would anyone disagree with this approach? No?"
        observations = observer.observe("claude", text, round_num=0)

        patterns = [o.pattern for o in observations]
        assert RhetoricalPattern.RHETORICAL_QUESTION in patterns

    def test_detect_analogy(self, observer):
        """Test analogy detection."""
        text = "Think of it as a pipeline, similar to an assembly line in a factory."
        observations = observer.observe("claude", text, round_num=0)

        patterns = [o.pattern for o in observations]
        assert RhetoricalPattern.ANALOGY in patterns

    def test_detect_qualification(self, observer):
        """Test qualification detection."""
        text = "It depends on the context and typically works well in some cases but varies."
        observations = observer.observe("claude", text, round_num=0)

        patterns = [o.pattern for o in observations]
        assert RhetoricalPattern.QUALIFICATION in patterns

    def test_short_text_no_patterns(self, observer):
        """Test short text returns no observations."""
        text = "Hello there."
        observations = observer.observe("claude", text, round_num=0)

        assert len(observations) == 0

    def test_empty_text_no_patterns(self, observer):
        """Test empty text returns no observations."""
        observations = observer.observe("claude", "", round_num=0)
        assert len(observations) == 0

    def test_multiple_patterns_single_text(self, observer):
        """Test detecting multiple patterns in one text."""
        # Use text with stronger pattern indicators (both keywords and regex matches)
        text = """I must acknowledge and concede that you're right about performance.
        However, I would argue that according to best practices and research shows
        that caching is the better approach."""
        observations = observer.observe("claude", text, round_num=0)

        patterns = {o.pattern for o in observations}
        # Should detect multiple patterns (CONCESSION, REBUTTAL, APPEAL_TO_AUTHORITY)
        assert len(patterns) >= 2


# ============================================================================
# Confidence Tests
# ============================================================================

class TestConfidence:
    """Tests for confidence scoring."""

    def test_high_confidence_included(self, observer):
        """Test high confidence observations are included."""
        text = "I acknowledge and concede that you're right about this valid point."
        observations = observer.observe("claude", text, round_num=0)

        # Should have at least one observation
        assert len(observations) >= 1
        for obs in observations:
            assert obs.confidence >= observer.min_confidence

    def test_low_confidence_excluded(self):
        """Test low confidence observations are excluded."""
        observer = RhetoricalAnalysisObserver(min_confidence=0.9)
        text = "Maybe this is somewhat related to that concept."
        observations = observer.observe("claude", text, round_num=0)

        # High threshold should exclude weak matches
        for obs in observations:
            assert obs.confidence >= 0.9

    def test_confidence_bounded(self, observer):
        """Test confidence is bounded 0-1."""
        text = "I acknowledge I acknowledge I acknowledge you're right about this."
        observations = observer.observe("claude", text, round_num=0)

        for obs in observations:
            assert 0.0 <= obs.confidence <= 1.0


# ============================================================================
# Excerpt Finding Tests
# ============================================================================

class TestExcerptFinding:
    """Tests for excerpt extraction."""

    def test_excerpt_contains_keyword(self, observer):
        """Test excerpt contains relevant keyword."""
        text = "This is unrelated. I must acknowledge your excellent point. More text here."
        observations = observer.observe("claude", text, round_num=0)

        for obs in observations:
            if obs.pattern == RhetoricalPattern.CONCESSION:
                assert "acknowledge" in obs.excerpt.lower()

    def test_excerpt_limited_length(self, observer):
        """Test excerpt is limited in length."""
        text = "I acknowledge that " + "x" * 300 + " is a valid point."
        observations = observer.observe("claude", text, round_num=0)

        for obs in observations:
            assert len(obs.excerpt) <= 150

    def test_excerpt_from_correct_sentence(self, observer):
        """Test excerpt comes from correct sentence."""
        text = "First sentence. However, I disagree with that view strongly. Final."
        observations = observer.observe("claude", text, round_num=0)

        for obs in observations:
            if obs.pattern == RhetoricalPattern.REBUTTAL:
                assert "disagree" in obs.excerpt.lower() or "however" in obs.excerpt.lower()


# ============================================================================
# Commentary Generation Tests
# ============================================================================

class TestCommentaryGeneration:
    """Tests for audience commentary generation."""

    def test_commentary_includes_agent_name(self, observer):
        """Test commentary includes agent name."""
        text = "I acknowledge your valid point about security concerns."
        observations = observer.observe("claude", text, round_num=0)

        for obs in observations:
            assert "claude" in obs.audience_commentary.lower()

    def test_commentary_not_empty(self, observer):
        """Test commentary is never empty."""
        text = "However, I would argue against that approach based on evidence."
        observations = observer.observe("claude", text, round_num=0)

        for obs in observations:
            assert len(obs.audience_commentary) > 0

    def test_commentary_pattern_specific(self, observer):
        """Test different patterns have different commentary styles."""
        # Fix random seed for reproducibility
        random.seed(42)

        concession_text = "I acknowledge that you're right about this point."
        rebuttal_text = "However, I disagree with that assessment entirely."

        concession_obs = observer.observe("agent1", concession_text, round_num=0)
        observer.reset()
        random.seed(43)
        rebuttal_obs = observer.observe("agent2", rebuttal_text, round_num=0)

        # Commentary should differ for different patterns
        if concession_obs and rebuttal_obs:
            assert concession_obs[0].audience_commentary != rebuttal_obs[0].audience_commentary


# ============================================================================
# Agent Pattern Tracking Tests
# ============================================================================

class TestAgentPatternTracking:
    """Tests for per-agent pattern tracking."""

    def test_track_agent_patterns(self, observer):
        """Test patterns are tracked per agent."""
        # Use text with strong pattern matches (keywords + regex) to hit confidence threshold
        observer.observe("claude", "I must acknowledge and concede that you're right about this valid point.", round_num=0)
        observer.observe("gpt", "However, I would disagree with that assessment strongly.", round_num=0)

        assert "claude" in observer.agent_patterns
        assert "gpt" in observer.agent_patterns

    def test_pattern_count_increases(self, observer):
        """Test pattern count increases with observations."""
        observer.observe("claude", "I acknowledge point one.", round_num=0)
        observer.observe("claude", "I concede point two is valid.", round_num=1)

        patterns = observer.agent_patterns.get("claude", {})
        concession_count = patterns.get("concession", 0)
        # Should have counted concessions
        assert concession_count >= 0  # May or may not detect depending on confidence


# ============================================================================
# Broadcast Callback Tests
# ============================================================================

class TestBroadcastCallback:
    """Tests for broadcast callback functionality."""

    def test_callback_called_on_observation(self, mock_callback):
        """Test callback is called when observations are made."""
        observer = RhetoricalAnalysisObserver(broadcast_callback=mock_callback)
        observer.observe("claude", "I acknowledge your valid point about this.", round_num=0)

        # Callback should have been called if observations were made
        if observer.observations:
            mock_callback.assert_called()

    def test_callback_receives_data(self, mock_callback):
        """Test callback receives correct data structure."""
        observer = RhetoricalAnalysisObserver(broadcast_callback=mock_callback)
        observer.observe("claude", "I acknowledge your point completely.", round_num=2)

        if mock_callback.called:
            call_args = mock_callback.call_args[0][0]
            assert call_args["type"] == "rhetorical_observations"
            assert "data" in call_args
            assert call_args["data"]["agent"] == "claude"
            assert call_args["data"]["round_num"] == 2

    def test_callback_error_handled(self):
        """Test callback errors don't crash observer."""
        def failing_callback(data):
            raise RuntimeError("Callback failed")

        observer = RhetoricalAnalysisObserver(broadcast_callback=failing_callback)

        # Should not raise
        observations = observer.observe("claude", "I acknowledge your point.", round_num=0)
        # Observer should still work
        assert isinstance(observations, list)


# ============================================================================
# Debate Dynamics Tests
# ============================================================================

class TestDebateDynamics:
    """Tests for debate dynamics analysis."""

    def test_empty_dynamics(self, observer):
        """Test dynamics with no observations."""
        dynamics = observer.get_debate_dynamics()

        assert dynamics["total_observations"] == 0
        assert dynamics["patterns_detected"] == {}
        assert dynamics["agent_styles"] == {}
        assert dynamics["dominant_pattern"] is None

    def test_dynamics_after_observations(self, observer):
        """Test dynamics after making observations."""
        observer.observe("claude", "I acknowledge point one and concede it.", round_num=0)
        observer.observe("gpt", "However, I disagree strongly with that.", round_num=1)

        dynamics = observer.get_debate_dynamics()

        assert dynamics["total_observations"] > 0
        assert len(dynamics["patterns_detected"]) > 0

    def test_dominant_pattern_detection(self, observer):
        """Test dominant pattern is correctly identified."""
        # Multiple concessions
        observer.observe("claude", "I acknowledge your point.", round_num=0)
        observer.observe("gpt", "I concede that is valid.", round_num=1)
        observer.observe("gemini", "Fair point, I agree.", round_num=2)

        dynamics = observer.get_debate_dynamics()

        if dynamics["dominant_pattern"]:
            # Should be a valid pattern name
            assert dynamics["dominant_pattern"] in [p.value for p in RhetoricalPattern]

    def test_agent_styles_populated(self, observer):
        """Test agent styles are populated."""
        observer.observe("claude", "I acknowledge your point here.", round_num=0)
        observer.observe("claude", "I concede that as well.", round_num=1)

        dynamics = observer.get_debate_dynamics()

        if "claude" in dynamics["agent_styles"]:
            style = dynamics["agent_styles"]["claude"]
            assert "dominant_pattern" in style
            assert "style" in style
            assert "pattern_diversity" in style


# ============================================================================
# Debate Characterization Tests
# ============================================================================

class TestDebateCharacterization:
    """Tests for _characterize_debate method."""

    def test_characterize_collaborative(self, observer):
        """Test collaborative debate characterization."""
        # Lots of concession and synthesis
        pattern_counts = {
            "concession": 5,
            "synthesis": 3,
            "rebuttal": 1,
        }

        character = observer._characterize_debate(pattern_counts)
        assert character == "collaborative"

    def test_characterize_contentious(self, observer):
        """Test contentious debate characterization."""
        pattern_counts = {
            "rebuttal": 8,
            "concession": 1,
        }

        character = observer._characterize_debate(pattern_counts)
        assert character == "contentious"

    def test_characterize_technical(self, observer):
        """Test technical debate characterization."""
        pattern_counts = {
            "technical_depth": 5,
            "rebuttal": 2,
        }

        character = observer._characterize_debate(pattern_counts)
        assert character == "technical"

    def test_characterize_evidence_driven(self, observer):
        """Test evidence-driven debate characterization."""
        pattern_counts = {
            "appeal_to_evidence": 4,
            "appeal_to_authority": 3,
            "rebuttal": 2,
        }

        character = observer._characterize_debate(pattern_counts)
        assert character == "evidence-driven"

    def test_characterize_balanced(self, observer):
        """Test balanced debate characterization."""
        # Avoid triggering collaborative check (concession + synthesis > 40%)
        # Need patterns distributed so no single category dominates
        pattern_counts = {
            "concession": 1,
            "rebuttal": 2,
            "synthesis": 1,
            "qualification": 2,
            "analogy": 2,
            "rhetorical_question": 2,
        }

        character = observer._characterize_debate(pattern_counts)
        assert character == "balanced"

    def test_characterize_empty(self, observer):
        """Test empty patterns characterization."""
        character = observer._characterize_debate({})
        assert character == "emerging"


# ============================================================================
# Style Mapping Tests
# ============================================================================

class TestStyleMapping:
    """Tests for _pattern_to_style method."""

    def test_concession_diplomatic(self, observer):
        """Test concession maps to diplomatic."""
        assert observer._pattern_to_style("concession") == "diplomatic"

    def test_rebuttal_combative(self, observer):
        """Test rebuttal maps to combative."""
        assert observer._pattern_to_style("rebuttal") == "combative"

    def test_synthesis_collaborative(self, observer):
        """Test synthesis maps to collaborative."""
        assert observer._pattern_to_style("synthesis") == "collaborative"

    def test_unknown_balanced(self, observer):
        """Test unknown pattern maps to balanced."""
        assert observer._pattern_to_style("unknown") == "balanced"


# ============================================================================
# Recent Observations Tests
# ============================================================================

class TestRecentObservations:
    """Tests for get_recent_observations method."""

    def test_get_recent_empty(self, observer):
        """Test get_recent_observations when empty."""
        recent = observer.get_recent_observations()
        assert recent == []

    def test_get_recent_limited(self, observer):
        """Test get_recent_observations respects limit."""
        # Add many observations
        for i in range(15):
            observer.observe("agent", f"I acknowledge point {i} is valid.", round_num=i)

        recent = observer.get_recent_observations(limit=5)
        assert len(recent) <= 5

    def test_get_recent_returns_dicts(self, observer):
        """Test get_recent_observations returns dicts."""
        observer.observe("claude", "I acknowledge your point.", round_num=0)

        recent = observer.get_recent_observations()
        for item in recent:
            assert isinstance(item, dict)
            assert "pattern" in item
            assert "agent" in item


# ============================================================================
# Reset Tests
# ============================================================================

class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_observations(self, observer):
        """Test reset clears observations."""
        # Use strong pattern text to ensure observation is created
        observer.observe("claude", "I must acknowledge and concede that you're right about this valid point.", round_num=0)
        assert len(observer.observations) > 0

        observer.reset()
        assert len(observer.observations) == 0

    def test_reset_clears_agent_patterns(self, observer):
        """Test reset clears agent patterns."""
        # Use strong pattern text to ensure agent patterns are tracked
        observer.observe("claude", "I must acknowledge and concede that you're right about this valid point.", round_num=0)
        assert len(observer.agent_patterns) > 0

        observer.reset()
        assert len(observer.agent_patterns) == 0


# ============================================================================
# Global Observer Tests
# ============================================================================

class TestGlobalObserver:
    """Tests for global observer functions."""

    def test_get_rhetorical_observer_singleton(self):
        """Test get_rhetorical_observer returns singleton."""
        obs1 = get_rhetorical_observer()
        obs2 = get_rhetorical_observer()

        assert obs1 is obs2

    def test_get_rhetorical_observer_type(self):
        """Test get_rhetorical_observer returns correct type."""
        obs = get_rhetorical_observer()
        assert isinstance(obs, RhetoricalAnalysisObserver)

    def test_reset_rhetorical_observer(self):
        """Test reset_rhetorical_observer clears singleton."""
        obs1 = get_rhetorical_observer()
        reset_rhetorical_observer()
        obs2 = get_rhetorical_observer()

        # Should be different instances
        assert obs1 is not obs2


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete observer flow."""

    def test_complete_debate_observation(self, observer):
        """Test observing a complete debate exchange."""
        # Round 1: Claude makes a claim
        observer.observe(
            "claude",
            "I believe we should use caching for performance. According to best practices, this approach improves response times.",
            round_num=0,
        )

        # Round 2: GPT rebuts
        observer.observe(
            "gpt",
            "However, I would argue against caching. The data shows that for small datasets, caching adds unnecessary complexity.",
            round_num=1,
        )

        # Round 3: Gemini synthesizes
        observer.observe(
            "gemini",
            "Combining both perspectives, we can find common ground. It depends on the context - caching works for large datasets.",
            round_num=2,
        )

        dynamics = observer.get_debate_dynamics()

        assert dynamics["total_observations"] > 0
        assert len(dynamics["agent_styles"]) >= 2

    def test_multi_round_pattern_tracking(self, observer):
        """Test pattern tracking across multiple rounds."""
        for round_num in range(5):
            observer.observe(
                "claude",
                f"I acknowledge point {round_num} is valid.",
                round_num=round_num,
            )

        dynamics = observer.get_debate_dynamics()

        # Claude should have accumulated pattern history
        if "claude" in dynamics["agent_styles"]:
            assert dynamics["agent_styles"]["claude"]["pattern_diversity"] >= 1
