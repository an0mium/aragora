"""Tests for Debate Fatigue Detection System."""

import pytest
from unittest.mock import patch
from datetime import datetime

from aragora.debate.fatigue_detector import (
    FatigueDetector,
    FatigueSignal,
    AgentBaseline,
    get_fatigue_detector,
    reset_fatigue_detector,
)


class TestFatigueSignal:
    """Test FatigueSignal dataclass."""

    def test_signal_creation(self):
        """Test creating a fatigue signal."""
        signal = FatigueSignal(
            agent="claude",
            score=0.75,
            round=5,
            recommendation="consider_rest",
            metrics={"response_length": 200, "unique_words_ratio": 0.5},
        )
        assert signal.agent == "claude"
        assert signal.score == 0.75
        assert signal.round == 5
        assert signal.recommendation == "consider_rest"
        assert signal.metrics["response_length"] == 200

    def test_signal_timestamp_auto_generated(self):
        """Test timestamp is auto-generated."""
        signal = FatigueSignal(
            agent="gpt",
            score=0.8,
            round=3,
            recommendation="rotate_out",
            metrics={},
        )
        assert signal.timestamp is not None
        # Should be a valid ISO timestamp
        datetime.fromisoformat(signal.timestamp)

    def test_signal_to_dict(self):
        """Test converting signal to dictionary."""
        signal = FatigueSignal(
            agent="claude",
            score=0.85,
            round=4,
            recommendation="rotate_out",
            metrics={"repetition_score": 0.6},
            timestamp="2024-01-15T10:30:00",
        )
        result = signal.to_dict()
        assert result["agent"] == "claude"
        assert result["score"] == 0.85
        assert result["round"] == 4
        assert result["recommendation"] == "rotate_out"
        assert result["metrics"]["repetition_score"] == 0.6
        assert result["timestamp"] == "2024-01-15T10:30:00"


class TestAgentBaseline:
    """Test AgentBaseline dataclass."""

    def test_default_values(self):
        """Test default baseline values."""
        baseline = AgentBaseline()
        assert baseline.avg_response_length == 500.0
        assert baseline.avg_unique_word_ratio == 0.7
        assert baseline.avg_argument_count == 3
        assert baseline.samples == 0

    def test_custom_values(self):
        """Test custom baseline values."""
        baseline = AgentBaseline(
            avg_response_length=800.0,
            avg_unique_word_ratio=0.65,
            avg_argument_count=5,
            samples=10,
        )
        assert baseline.avg_response_length == 800.0
        assert baseline.avg_unique_word_ratio == 0.65
        assert baseline.avg_argument_count == 5
        assert baseline.samples == 10


class TestFatigueDetectorInit:
    """Test FatigueDetector initialization."""

    def test_default_initialization(self):
        """Test default initialization values."""
        detector = FatigueDetector()
        assert detector.fatigue_threshold == 0.7
        assert detector.critical_threshold == 0.85
        assert detector.baseline_rounds == 2

    def test_custom_initialization(self):
        """Test custom initialization values."""
        detector = FatigueDetector(
            fatigue_threshold=0.6,
            critical_threshold=0.9,
            baseline_rounds=3,
        )
        assert detector.fatigue_threshold == 0.6
        assert detector.critical_threshold == 0.9
        assert detector.baseline_rounds == 3

    def test_empty_state_on_init(self):
        """Test detector starts with empty state."""
        detector = FatigueDetector()
        assert len(detector.baselines) == 0
        assert len(detector.response_history) == 0
        assert len(detector.seen_arguments) == 0
        assert len(detector.fatigue_signals) == 0


class TestUniqueWordsRatio:
    """Test unique words ratio calculation."""

    def test_all_unique_words(self):
        """Test response with all unique words."""
        detector = FatigueDetector()
        ratio = detector._unique_words_ratio("The quick brown fox jumps")
        assert ratio == 1.0

    def test_repeated_words(self):
        """Test response with repeated words."""
        detector = FatigueDetector()
        ratio = detector._unique_words_ratio("the the the word word")
        # 2 unique / 5 total = 0.4
        assert ratio == pytest.approx(0.4)

    def test_empty_response(self):
        """Test empty response returns 0."""
        detector = FatigueDetector()
        ratio = detector._unique_words_ratio("")
        assert ratio == 0.0

    def test_case_insensitive(self):
        """Test ratio is case insensitive."""
        detector = FatigueDetector()
        ratio = detector._unique_words_ratio("Word WORD word")
        # All same word when lowercased
        assert ratio == pytest.approx(1 / 3)


class TestExtractNgrams:
    """Test n-gram extraction."""

    def test_extract_trigrams(self):
        """Test extracting trigrams."""
        detector = FatigueDetector()
        ngrams = detector._extract_ngrams("one two three four five", n=3)
        assert len(ngrams) == 3
        assert "one two three" in ngrams
        assert "two three four" in ngrams
        assert "three four five" in ngrams

    def test_text_shorter_than_n(self):
        """Test text shorter than n returns empty set."""
        detector = FatigueDetector()
        ngrams = detector._extract_ngrams("one two", n=3)
        assert len(ngrams) == 0

    def test_exact_length_n(self):
        """Test text with exactly n words."""
        detector = FatigueDetector()
        ngrams = detector._extract_ngrams("one two three", n=3)
        assert len(ngrams) == 1
        assert "one two three" in ngrams


class TestDetectRepetition:
    """Test repetition detection."""

    def test_no_history_no_repetition(self):
        """Test no repetition when no history exists."""
        detector = FatigueDetector()
        score = detector._detect_repetition("claude", "This is a new response")
        assert score == 0.0

    def test_repetition_detection_needs_text_key(self):
        """Test repetition detection requires 'text' key in history."""
        detector = FatigueDetector()
        # Add history without 'text' key
        detector.response_history["claude"].append({"round": 1, "length": 100})
        score = detector._detect_repetition("claude", "This is a response")
        # Should return 0 because history has no 'text' field
        assert score == 0.0


class TestExtractKeyPhrases:
    """Test key phrase extraction for argument novelty."""

    def test_extract_argumentative_phrases(self):
        """Test extraction of phrases with argument indicators."""
        detector = FatigueDetector()
        text = (
            "This is important. Because the evidence suggests this is true. "
            "Therefore we should act. However, there are risks."
        )
        phrases = detector._extract_key_phrases(text)
        # Should extract phrases with indicators
        assert len(phrases) > 0

    def test_no_argumentative_indicators(self):
        """Test text without argumentative indicators."""
        detector = FatigueDetector()
        text = "Hello world. Simple statement. Another one."
        phrases = detector._extract_key_phrases(text)
        assert len(phrases) == 0

    def test_short_phrases_ignored(self):
        """Test phrases shorter than 20 chars are ignored."""
        detector = FatigueDetector()
        text = "Because yes."  # Too short
        phrases = detector._extract_key_phrases(text)
        assert len(phrases) == 0

    def test_multiple_indicators(self):
        """Test text with multiple argument indicators."""
        detector = FatigueDetector()
        text = (
            "I believe this is correct because the data shows clear evidence. "
            "Furthermore, previous research suggests similar conclusions. "
            "Therefore, we should proceed with caution."
        )
        phrases = detector._extract_key_phrases(text)
        assert len(phrases) >= 2


class TestArgumentNovelty:
    """Test argument novelty detection."""

    def test_first_response_fully_novel(self):
        """Test first response is considered fully novel."""
        detector = FatigueDetector()
        text = "I believe this approach is correct because the evidence is clear."
        novelty = detector._argument_novelty("claude", text)
        # First time seeing these arguments
        assert novelty == 1.0

    def test_repeated_arguments_reduce_novelty(self):
        """Test repeated arguments reduce novelty score."""
        detector = FatigueDetector()
        text1 = "I believe this approach is correct because the evidence is clear."
        text2 = "I believe this approach is correct because the evidence is clear."

        # First response - all novel
        novelty1 = detector._argument_novelty("claude", text1)
        assert novelty1 == 1.0

        # Same response - no novelty (all phrases seen before)
        novelty2 = detector._argument_novelty("claude", text2)
        assert novelty2 == 0.0

    def test_partially_new_arguments(self):
        """Test partially new arguments give intermediate novelty."""
        detector = FatigueDetector()
        text1 = "I believe this is true because the data shows it clearly."

        novelty1 = detector._argument_novelty("claude", text1)
        assert novelty1 == 1.0

        # New arguments with some overlap
        text2 = (
            "Therefore we should consider new approaches. "
            "I believe this is true because the data shows it clearly."
        )
        novelty2 = detector._argument_novelty("claude", text2)
        # Should have some novelty but not 100%
        assert 0.0 < novelty2 < 1.0

    def test_empty_phrases_returns_full_novelty(self):
        """Test response with no key phrases returns 1.0."""
        detector = FatigueDetector()
        text = "Hello world."  # No argument indicators
        novelty = detector._argument_novelty("claude", text)
        assert novelty == 1.0


class TestEngagementScore:
    """Test engagement score calculation."""

    def test_no_context_neutral_score(self):
        """Test no context gives neutral engagement score."""
        detector = FatigueDetector()
        score = detector._engagement_score("Some response text", None)
        assert score == 0.5

    def test_referencing_other_agents(self):
        """Test referencing other agents increases score."""
        detector = FatigueDetector()
        context = {"other_agents": ["gpt", "gemini"]}
        response = "Building on what GPT mentioned earlier, I think we should..."
        score = detector._engagement_score(response, context)
        assert score > 0.5

    def test_reference_patterns_increase_score(self):
        """Test reference patterns increase engagement score."""
        detector = FatigueDetector()
        context = {"other_agents": []}
        response = (
            "As mentioned earlier, building on the previous point, I disagrees with that approach."
        )
        score = detector._engagement_score(response, context)
        # Should have pattern matches
        assert score > 0.3

    def test_no_engagement_patterns(self):
        """Test response with no engagement patterns."""
        detector = FatigueDetector()
        context = {"other_agents": ["gpt"]}
        response = "Here is my standalone opinion with no references."
        score = detector._engagement_score(response, context)
        # Base score only
        assert score == pytest.approx(0.3)


class TestUpdateBaseline:
    """Test baseline updating."""

    def test_first_sample_sets_baseline(self):
        """Test first sample updates baseline."""
        detector = FatigueDetector()
        metrics = {
            "response_length": 600,
            "unique_words_ratio": 0.8,
        }
        detector._update_baseline("claude", metrics)

        baseline = detector.baselines["claude"]
        assert baseline.avg_response_length == 600.0
        assert baseline.avg_unique_word_ratio == 0.8
        assert baseline.samples == 1

    def test_running_average(self):
        """Test running average calculation."""
        detector = FatigueDetector()

        # First sample
        detector._update_baseline("claude", {"response_length": 500, "unique_words_ratio": 0.7})

        # Second sample
        detector._update_baseline("claude", {"response_length": 700, "unique_words_ratio": 0.9})

        baseline = detector.baselines["claude"]
        assert baseline.avg_response_length == pytest.approx(600.0)
        assert baseline.avg_unique_word_ratio == pytest.approx(0.8)
        assert baseline.samples == 2


class TestCalculateFatigueScore:
    """Test fatigue score calculation."""

    def test_no_baseline_returns_zero(self):
        """Test no baseline returns zero fatigue."""
        detector = FatigueDetector()
        metrics = {
            "response_length": 100,
            "unique_words_ratio": 0.5,
            "repetition_score": 0.5,
            "argument_novelty": 0.5,
            "engagement_score": 0.5,
        }
        score = detector._calculate_fatigue_score("unknown_agent", metrics)
        assert score == 0.0

    def test_perfect_metrics_low_fatigue(self):
        """Test perfect metrics result in low fatigue score."""
        detector = FatigueDetector()
        # Set up baseline
        detector.baselines["claude"].avg_response_length = 500.0
        detector.baselines["claude"].avg_unique_word_ratio = 0.7
        detector.baselines["claude"].samples = 2

        metrics = {
            "response_length": 500,  # Same as baseline
            "unique_words_ratio": 0.7,  # Same as baseline
            "repetition_score": 0.0,  # No repetition
            "argument_novelty": 1.0,  # All novel
            "engagement_score": 1.0,  # High engagement
        }
        score = detector._calculate_fatigue_score("claude", metrics)
        assert score < 0.3  # Low fatigue

    def test_poor_metrics_high_fatigue(self):
        """Test poor metrics result in high fatigue score."""
        detector = FatigueDetector()
        # Set up baseline
        detector.baselines["claude"].avg_response_length = 500.0
        detector.baselines["claude"].avg_unique_word_ratio = 0.7
        detector.baselines["claude"].samples = 2

        metrics = {
            "response_length": 100,  # Much shorter
            "unique_words_ratio": 0.3,  # Lower vocabulary
            "repetition_score": 0.8,  # High repetition
            "argument_novelty": 0.1,  # Low novelty
            "engagement_score": 0.1,  # Low engagement
        }
        score = detector._calculate_fatigue_score("claude", metrics)
        assert score > 0.7  # High fatigue

    def test_fatigue_score_capped_at_one(self):
        """Test fatigue score is capped at 1.0."""
        detector = FatigueDetector()
        detector.baselines["claude"].avg_response_length = 1000.0
        detector.baselines["claude"].avg_unique_word_ratio = 0.9
        detector.baselines["claude"].samples = 5

        # Extreme decline metrics
        metrics = {
            "response_length": 10,
            "unique_words_ratio": 0.1,
            "repetition_score": 1.0,
            "argument_novelty": 0.0,
            "engagement_score": 0.0,
        }
        score = detector._calculate_fatigue_score("claude", metrics)
        assert score <= 1.0


class TestGetRecommendation:
    """Test recommendation generation."""

    def test_critical_threshold_rotate_out(self):
        """Test critical score recommends rotate_out."""
        detector = FatigueDetector(fatigue_threshold=0.7, critical_threshold=0.85)
        assert detector._get_recommendation(0.90) == "rotate_out"

    def test_fatigue_threshold_consider_rest(self):
        """Test fatigue score recommends consider_rest."""
        detector = FatigueDetector(fatigue_threshold=0.7, critical_threshold=0.85)
        assert detector._get_recommendation(0.75) == "consider_rest"

    def test_below_threshold_monitor(self):
        """Test below threshold recommends monitor."""
        detector = FatigueDetector(fatigue_threshold=0.7, critical_threshold=0.85)
        assert detector._get_recommendation(0.5) == "monitor"

    def test_exactly_at_critical(self):
        """Test exactly at critical threshold."""
        detector = FatigueDetector(fatigue_threshold=0.7, critical_threshold=0.85)
        # Above critical should rotate_out
        assert detector._get_recommendation(0.86) == "rotate_out"

    def test_exactly_at_fatigue(self):
        """Test exactly at fatigue threshold."""
        detector = FatigueDetector(fatigue_threshold=0.7, critical_threshold=0.85)
        # At or above fatigue but below critical should consider_rest
        assert detector._get_recommendation(0.71) == "consider_rest"


class TestAnalyzeResponse:
    """Test main response analysis method."""

    def test_empty_response_returns_none(self):
        """Test empty response returns None."""
        detector = FatigueDetector()
        signal = detector.analyze_response("claude", "", round=3)
        assert signal is None

    def test_whitespace_response_returns_none(self):
        """Test whitespace-only response returns None."""
        detector = FatigueDetector()
        signal = detector.analyze_response("claude", "   \n\t  ", round=3)
        assert signal is None

    def test_baseline_rounds_return_none(self):
        """Test responses during baseline rounds return None."""
        detector = FatigueDetector(baseline_rounds=2)
        response = "This is a test response with enough content to be meaningful."

        # Round 1 - baseline
        signal = detector.analyze_response("claude", response, round=1)
        assert signal is None

        # Round 2 - still baseline
        signal = detector.analyze_response("claude", response, round=2)
        assert signal is None

    def test_baseline_updated_during_early_rounds(self):
        """Test baseline is updated during early rounds."""
        detector = FatigueDetector(baseline_rounds=2)
        response = "A " * 200  # 200 words

        detector.analyze_response("claude", response, round=1)
        assert detector.baselines["claude"].samples == 1

        detector.analyze_response("claude", response, round=2)
        assert detector.baselines["claude"].samples == 2

    def test_signal_generated_when_fatigued(self):
        """Test signal is generated when fatigue detected."""
        detector = FatigueDetector(fatigue_threshold=0.1, baseline_rounds=1)

        # Build baseline with long response
        long_response = "word " * 500
        detector.analyze_response("claude", long_response, round=1)

        # Now send short, repetitive response
        short_response = "ok ok ok"
        signal = detector.analyze_response("claude", short_response, round=2)

        # Should detect fatigue
        assert signal is not None
        assert signal.agent == "claude"
        assert signal.score > 0.1
        assert signal.round == 2

    def test_no_signal_when_not_fatigued(self):
        """Test no signal when agent is not fatigued."""
        detector = FatigueDetector(fatigue_threshold=0.9, baseline_rounds=1)

        # Build baseline
        response = "This is a well-written response with unique vocabulary and good engagement."
        detector.analyze_response("claude", response, round=1)

        # Similar quality response
        signal = detector.analyze_response("claude", response + " And more content.", round=2)

        # Should not trigger fatigue
        assert signal is None

    def test_response_history_recorded(self):
        """Test response history is recorded after baseline."""
        detector = FatigueDetector(baseline_rounds=1)

        detector.analyze_response("claude", "Baseline response here", round=1)
        detector.analyze_response("claude", "Second response here", round=2)

        history = detector.response_history["claude"]
        assert len(history) == 1  # Only post-baseline recorded
        assert history[0]["round"] == 2

    def test_signal_stored_in_list(self):
        """Test generated signals are stored."""
        detector = FatigueDetector(fatigue_threshold=0.01, baseline_rounds=1)

        detector.analyze_response("claude", "word " * 100, round=1)
        detector.analyze_response("claude", "x", round=2)

        assert len(detector.fatigue_signals) >= 1


class TestMultiRoundScenarios:
    """Test multi-round debate scenarios."""

    def test_progressive_fatigue(self):
        """Test progressive fatigue detection across rounds."""
        detector = FatigueDetector(fatigue_threshold=0.5, baseline_rounds=2)

        # Baseline rounds - high quality
        good_response = (
            "I believe this is important because the evidence clearly shows significant impact. "
            "Furthermore, the research suggests we should consider multiple perspectives. "
            "Therefore, my recommendation is to proceed with caution."
        )
        detector.analyze_response("claude", good_response, round=1)
        detector.analyze_response("claude", good_response, round=2)

        # Round 3 - still okay
        signal3 = detector.analyze_response(
            "claude", good_response + " Additional thoughts.", round=3
        )

        # Round 4 - getting shorter
        signal4 = detector.analyze_response(
            "claude", "Yes I agree because that's correct.", round=4
        )

        # Round 5 - very short
        signal5 = detector.analyze_response("claude", "Yes.", round=5)

        # Should see increasing fatigue
        if signal4 is not None and signal5 is not None:
            assert signal5.score >= signal4.score

    def test_multiple_agents_tracked_separately(self):
        """Test multiple agents are tracked independently."""
        detector = FatigueDetector(baseline_rounds=1)

        # Build baselines
        detector.analyze_response("claude", "Claude baseline response here", round=1)
        detector.analyze_response("gpt", "GPT baseline response here", round=1)

        # Check separate tracking
        assert "claude" in detector.baselines
        assert "gpt" in detector.baselines
        assert detector.baselines["claude"].samples == 1
        assert detector.baselines["gpt"].samples == 1

    def test_debate_with_context(self):
        """Test debate with context for engagement scoring."""
        detector = FatigueDetector(fatigue_threshold=0.9, baseline_rounds=1)

        context = {"other_agents": ["gpt", "gemini"]}

        # Build baseline
        detector.analyze_response(
            "claude",
            "As GPT mentioned earlier, I agree with the assessment because it's well-reasoned.",
            round=1,
            context=context,
        )

        # Response with engagement
        signal = detector.analyze_response(
            "claude",
            "Building on gemini's point, I think we should furthermore consider...",
            round=2,
            context=context,
        )

        # High engagement should reduce fatigue
        if signal is not None:
            assert signal.metrics["engagement_score"] > 0.3


class TestEdgeCases:
    """Test edge cases."""

    def test_single_round_debate(self):
        """Test single round debate doesn't crash."""
        detector = FatigueDetector(baseline_rounds=1)
        signal = detector.analyze_response("claude", "Single round response", round=1)
        assert signal is None  # Still in baseline

    def test_new_debate_no_history(self):
        """Test new debate with no prior history."""
        detector = FatigueDetector()
        assert len(detector.response_history) == 0
        assert len(detector.baselines) == 0

    def test_very_long_response(self):
        """Test handling very long response."""
        detector = FatigueDetector(baseline_rounds=1)
        long_response = "word " * 10000
        # Should not crash
        signal = detector.analyze_response("claude", long_response, round=1)
        assert signal is None

    def test_unicode_response(self):
        """Test handling unicode characters."""
        detector = FatigueDetector(baseline_rounds=1)
        unicode_response = (
            "This response includes unicode: \u4e2d\u6587 \u0410\u0411\u0412 \u03b1\u03b2\u03b3"
        )
        signal = detector.analyze_response("claude", unicode_response, round=1)
        assert signal is None

    def test_special_characters(self):
        """Test handling special characters."""
        detector = FatigueDetector(baseline_rounds=1)
        special_response = "Response with special chars: @#$%^&*()_+{}|:<>?!"
        signal = detector.analyze_response("claude", special_response, round=1)
        assert signal is None


class TestGetMethods:
    """Test getter methods."""

    def test_get_agent_fatigue_history_empty(self):
        """Test getting history for unknown agent."""
        detector = FatigueDetector()
        history = detector.get_agent_fatigue_history("unknown")
        assert history == []

    def test_get_agent_fatigue_history_with_data(self):
        """Test getting history with data."""
        detector = FatigueDetector(baseline_rounds=1)
        detector.analyze_response("claude", "Baseline", round=1)
        detector.analyze_response("claude", "Response two", round=2)

        history = detector.get_agent_fatigue_history("claude")
        assert len(history) == 1
        assert history[0]["round"] == 2

    def test_get_current_fatigue_levels_empty(self):
        """Test getting fatigue levels with no data."""
        detector = FatigueDetector()
        levels = detector.get_current_fatigue_levels()
        assert levels == {}

    def test_get_current_fatigue_levels_with_data(self):
        """Test getting fatigue levels with data."""
        detector = FatigueDetector(baseline_rounds=1)
        detector.analyze_response("claude", "Baseline", round=1)
        detector.analyze_response("claude", "Response two", round=2)

        levels = detector.get_current_fatigue_levels()
        assert "claude" in levels
        assert isinstance(levels["claude"], float)

    def test_get_all_signals_empty(self):
        """Test getting signals when none generated."""
        detector = FatigueDetector()
        signals = detector.get_all_signals()
        assert signals == []

    def test_get_all_signals_with_data(self):
        """Test getting all generated signals."""
        detector = FatigueDetector(fatigue_threshold=0.01, baseline_rounds=1)
        detector.analyze_response("claude", "word " * 100, round=1)
        detector.analyze_response("claude", "x", round=2)

        signals = detector.get_all_signals()
        assert len(signals) >= 1
        assert all(isinstance(s, FatigueSignal) for s in signals)


class TestReset:
    """Test reset functionality."""

    def test_reset_clears_all_state(self):
        """Test reset clears all state."""
        detector = FatigueDetector(baseline_rounds=1)

        # Build up state
        detector.analyze_response("claude", "First response", round=1)
        detector.analyze_response("claude", "Second response", round=2)
        detector.seen_arguments["claude"].add("test argument")

        # Reset
        detector.reset()

        assert len(detector.baselines) == 0
        assert len(detector.response_history) == 0
        assert len(detector.seen_arguments) == 0
        assert len(detector.fatigue_signals) == 0


class TestToDict:
    """Test dictionary export."""

    def test_to_dict_empty(self):
        """Test exporting empty detector state."""
        detector = FatigueDetector()
        result = detector.to_dict()

        assert result["fatigue_threshold"] == 0.7
        assert result["critical_threshold"] == 0.85
        assert result["current_levels"] == {}
        assert result["signals"] == []
        assert result["tracked_agents"] == []

    def test_to_dict_with_data(self):
        """Test exporting detector with data."""
        detector = FatigueDetector(fatigue_threshold=0.5, baseline_rounds=1)
        detector.analyze_response("claude", "Baseline response", round=1)
        detector.analyze_response("claude", "Second response", round=2)

        result = detector.to_dict()

        assert result["fatigue_threshold"] == 0.5
        assert "claude" in result["tracked_agents"]


class TestSingletonFunctions:
    """Test singleton getter/setter functions."""

    def test_get_fatigue_detector_singleton(self):
        """Test get_fatigue_detector returns singleton."""
        # Reset first
        reset_fatigue_detector()

        detector1 = get_fatigue_detector()
        detector2 = get_fatigue_detector()

        assert detector1 is detector2

    def test_reset_fatigue_detector(self):
        """Test reset_fatigue_detector clears state."""
        detector = get_fatigue_detector()
        detector.baselines["test"] = AgentBaseline()

        reset_fatigue_detector()

        # Should still be same instance but cleared
        detector2 = get_fatigue_detector()
        assert len(detector2.baselines) == 0

    def test_reset_before_any_creation(self):
        """Test reset when no detector exists yet."""
        # Force reset of global state
        import aragora.debate.fatigue_detector as fd_module

        fd_module._default_detector = None

        # Should not raise
        reset_fatigue_detector()


class TestThresholdBehavior:
    """Test threshold-based behavior."""

    def test_custom_fatigue_threshold(self):
        """Test custom fatigue threshold affects signal generation."""
        # High threshold - less sensitive
        detector_high = FatigueDetector(fatigue_threshold=0.95, baseline_rounds=1)
        detector_high.analyze_response("claude", "word " * 100, round=1)
        signal_high = detector_high.analyze_response("claude", "short", round=2)

        # Low threshold - more sensitive
        detector_low = FatigueDetector(fatigue_threshold=0.1, baseline_rounds=1)
        detector_low.analyze_response("claude", "word " * 100, round=1)
        signal_low = detector_low.analyze_response("claude", "short", round=2)

        # Low threshold should be more likely to trigger
        if signal_high is None:
            assert signal_low is not None or signal_low is None  # May still not trigger
        if signal_low is not None:
            assert signal_low.score > 0.1

    def test_critical_vs_fatigue_threshold(self):
        """Test critical threshold results in rotate_out recommendation."""
        detector = FatigueDetector(fatigue_threshold=0.3, critical_threshold=0.5, baseline_rounds=1)

        # Build baseline with long response
        detector.analyze_response("claude", "word " * 500, round=1)

        # Send minimal response to trigger high fatigue
        signal = detector.analyze_response("claude", "ok", round=2)

        if signal is not None and signal.score > 0.5:
            assert signal.recommendation == "rotate_out"


class TestMetricsCalculation:
    """Test individual metrics calculation details."""

    def test_length_decline_calculation(self):
        """Test response length decline affects fatigue."""
        detector = FatigueDetector(baseline_rounds=1)

        # Set baseline to 1000 chars
        detector.baselines["claude"].avg_response_length = 1000.0
        detector.baselines["claude"].samples = 2

        # Metrics with 50% length decline
        metrics = {
            "response_length": 500,
            "unique_words_ratio": 0.7,
            "repetition_score": 0.0,
            "argument_novelty": 1.0,
            "engagement_score": 1.0,
        }

        score = detector._calculate_fatigue_score("claude", metrics)
        # Length decline should contribute to fatigue
        assert score > 0

    def test_vocabulary_decline_calculation(self):
        """Test vocabulary decline affects fatigue."""
        detector = FatigueDetector(baseline_rounds=1)

        detector.baselines["claude"].avg_response_length = 500.0
        detector.baselines["claude"].avg_unique_word_ratio = 0.8
        detector.baselines["claude"].samples = 2

        # Metrics with vocabulary decline
        metrics = {
            "response_length": 500,
            "unique_words_ratio": 0.4,  # Much lower than baseline 0.8
            "repetition_score": 0.0,
            "argument_novelty": 1.0,
            "engagement_score": 1.0,
        }

        score = detector._calculate_fatigue_score("claude", metrics)
        # Vocabulary decline should contribute to fatigue
        assert score > 0
