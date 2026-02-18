"""Tests for aragora.debate.rhetorical_observer module."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.rhetorical_observer import (
    ClaimRelationship,
    FallacyDetection,
    FallacyType,
    PremiseChain,
    RhetoricalAnalysisObserver,
    RhetoricalObservation,
    RhetoricalPattern,
    StructuralAnalysisResult,
    StructuralAnalyzer,
    get_rhetorical_observer,
    reset_rhetorical_observer,
)


# ---------------------------------------------------------------------------
# Dataclass to_dict tests
# ---------------------------------------------------------------------------


class TestRhetoricalObservationToDict:
    def test_to_dict_returns_all_fields(self):
        obs = RhetoricalObservation(
            pattern=RhetoricalPattern.CONCESSION,
            agent="claude",
            round_num=2,
            confidence=0.8,
            excerpt="I acknowledge that point",
            audience_commentary="claude concedes",
            timestamp=1000.0,
        )
        d = obs.to_dict()
        assert d["pattern"] == "concession"
        assert d["agent"] == "claude"
        assert d["round_num"] == 2
        assert d["confidence"] == 0.8
        assert d["excerpt"] == "I acknowledge that point"
        assert d["audience_commentary"] == "claude concedes"
        assert d["timestamp"] == 1000.0


class TestFallacyDetectionToDict:
    def test_to_dict_returns_all_fields(self):
        fd = FallacyDetection(
            fallacy_type=FallacyType.STRAW_MAN,
            confidence=0.6,
            excerpt="so you're saying",
            explanation="Misrepresents",
            agent="gpt4",
            round_num=1,
        )
        d = fd.to_dict()
        assert d["fallacy_type"] == "straw_man"
        assert d["confidence"] == 0.6
        assert d["agent"] == "gpt4"
        assert d["round_num"] == 1


class TestPremiseChainToDict:
    def test_to_dict_returns_all_fields(self):
        pc = PremiseChain(
            premises=["A is true", "B follows from A"],
            conclusion="Therefore C",
            agent="claude",
            confidence=0.6,
            has_gap=True,
        )
        d = pc.to_dict()
        assert d["premises"] == ["A is true", "B follows from A"]
        assert d["conclusion"] == "Therefore C"
        assert d["has_gap"] is True


class TestClaimRelationshipToDict:
    def test_to_dict_returns_all_fields(self):
        cr = ClaimRelationship(
            source_claim="X supports Y",
            target_claim="Y is true",
            relation="supports",
            confidence=0.7,
            agent="claude",
        )
        d = cr.to_dict()
        assert d["source_claim"] == "X supports Y"
        assert d["target_claim"] == "Y is true"
        assert d["relation"] == "supports"


class TestStructuralAnalysisResultToDict:
    def test_to_dict_converts_nested_objects(self):
        result = StructuralAnalysisResult(
            fallacies=[
                FallacyDetection(
                    fallacy_type=FallacyType.AD_HOMINEM,
                    confidence=0.5,
                    excerpt="you always",
                    explanation="Attacks person",
                )
            ],
            premise_chains=[
                PremiseChain(
                    premises=["A"], conclusion="B", agent="x", confidence=0.6
                )
            ],
            unsupported_claims=["claim1"],
            contradictions=[("sent_a", "sent_b")],
            claim_relationships=[
                ClaimRelationship(
                    source_claim="c1",
                    target_claim="c2",
                    relation="supports",
                    confidence=0.5,
                )
            ],
            confidence=0.6,
        )
        d = result.to_dict()
        assert len(d["fallacies"]) == 1
        assert d["fallacies"][0]["fallacy_type"] == "ad_hominem"
        assert len(d["premise_chains"]) == 1
        assert d["unsupported_claims"] == ["claim1"]
        assert d["contradictions"] == [("sent_a", "sent_b")]
        assert len(d["claim_relationships"]) == 1


# ---------------------------------------------------------------------------
# StructuralAnalyzer tests
# ---------------------------------------------------------------------------


class TestStructuralAnalyzerShortContent:
    def test_short_content_returns_empty_result(self):
        analyzer = StructuralAnalyzer()
        result = analyzer.analyze("too short")
        assert result.fallacies == []
        assert result.premise_chains == []
        assert result.unsupported_claims == []
        assert result.contradictions == []
        assert result.claim_relationships == []
        assert result.confidence == 0.0

    def test_empty_content_returns_empty_result(self):
        analyzer = StructuralAnalyzer()
        result = analyzer.analyze("")
        assert result.confidence == 0.0


class TestStructuralAnalyzerFallacyDetection:
    def test_ad_hominem_detected_via_keywords(self):
        analyzer = StructuralAnalyzer()
        content = (
            "You clearly don't understand this topic at all. "
            "Your ignorance is showing and you never get things right."
        )
        result = analyzer.analyze(content, agent="agent1", round_num=1)
        fallacy_types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.AD_HOMINEM in fallacy_types
        ad_hom = [f for f in result.fallacies if f.fallacy_type == FallacyType.AD_HOMINEM][0]
        assert ad_hom.confidence >= 0.3
        assert ad_hom.agent == "agent1"
        assert ad_hom.round_num == 1

    def test_straw_man_detected_via_keywords(self):
        analyzer = StructuralAnalyzer()
        content = (
            "So you're saying that we should abandon all testing completely? "
            "What you really mean is that quality does not matter at all."
        )
        result = analyzer.analyze(content, agent="agent2")
        fallacy_types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.STRAW_MAN in fallacy_types

    def test_false_dilemma_detected(self):
        analyzer = StructuralAnalyzer()
        content = (
            "Either we rewrite the entire system from scratch or we accept it will always be broken. "
            "There are only two options available to us here."
        )
        result = analyzer.analyze(content)
        fallacy_types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.FALSE_DILEMMA in fallacy_types

    def test_fallacy_below_threshold_not_reported(self):
        analyzer = StructuralAnalyzer()
        # Content with no strong fallacy indicators
        content = (
            "The database should be optimized for read-heavy workloads. "
            "We could use caching to improve performance significantly."
        )
        result = analyzer.analyze(content)
        # Should have no ad_hominem, straw_man, etc.
        for f in result.fallacies:
            assert f.confidence >= 0.3


class TestStructuralAnalyzerPremiseChains:
    def test_therefore_conclusion_with_because_premise(self):
        analyzer = StructuralAnalyzer()
        content = (
            "The system handles millions of requests daily. "
            "Because the load is increasing, we need more capacity. "
            "Therefore we must scale horizontally to handle the growth."
        )
        result = analyzer.analyze(content, agent="claude")
        assert len(result.premise_chains) >= 1
        # At least one chain should have a conclusion with "therefore"
        conclusions = [c.conclusion.lower() for c in result.premise_chains]
        assert any("therefore" in c for c in conclusions)

    def test_premise_chain_with_gap(self):
        analyzer = StructuralAnalyzer()
        content = (
            "The market is growing rapidly. "
            "Competitors are investing heavily. "
            "Thus we need to act now to capture market share."
        )
        result = analyzer.analyze(content, agent="agent1")
        assert len(result.premise_chains) >= 1
        # Find the chain with "thus" conclusion
        thus_chains = [c for c in result.premise_chains if "thus" in c.conclusion.lower()]
        if thus_chains:
            chain = thus_chains[0]
            # Premises lack "because"/"since" so has_gap should be True
            assert chain.has_gap is True
            assert chain.confidence == 0.4

    def test_because_chain_premise_follows_claim(self):
        analyzer = StructuralAnalyzer()
        content = (
            "We need to refactor this module. "
            "Because the current design has too many circular dependencies. "
            "The team agrees on the approach."
        )
        result = analyzer.analyze(content, agent="dev")
        assert len(result.premise_chains) >= 1


class TestStructuralAnalyzerUnsupportedClaims:
    def test_assertions_without_evidence_detected(self):
        analyzer = StructuralAnalyzer()
        content = (
            "This approach is definitely the best one available. "
            "It will certainly outperform all alternatives easily. "
            "There is no doubt this is the right path forward."
        )
        result = analyzer.analyze(content)
        assert len(result.unsupported_claims) >= 1

    def test_assertion_with_evidence_not_flagged(self):
        analyzer = StructuralAnalyzer()
        content = (
            "This approach is definitely the best because the benchmarks show 3x improvement. "
            "According to the data we collected over six months of testing this system."
        )
        result = analyzer.analyze(content)
        # The assertion "definitely" has evidence in the same sentence
        # so the unsupported claims list may be empty or smaller
        for claim in result.unsupported_claims:
            # Any claims flagged should not contain evidence markers
            assert "because" not in claim.lower() or "according" not in claim.lower()


class TestStructuralAnalyzerContradictions:
    def test_should_vs_should_not(self):
        analyzer = StructuralAnalyzer()
        content = (
            "We should adopt microservices for this project. "
            "However, we should not adopt microservices until the team is ready."
        )
        result = analyzer.analyze(content)
        assert len(result.contradictions) >= 1
        pair = result.contradictions[0]
        assert isinstance(pair, tuple)
        assert len(pair) == 2

    def test_can_vs_cannot(self):
        analyzer = StructuralAnalyzer()
        content = (
            "The system can handle ten thousand concurrent connections. "
            "Unfortunately the system cannot handle more than a hundred users."
        )
        result = analyzer.analyze(content)
        assert len(result.contradictions) >= 1


class TestStructuralAnalyzerClaimRelationships:
    def test_support_relationship_with_history(self):
        analyzer = StructuralAnalyzer()
        # First analyze to seed claim history
        analyzer.analyze(
            "Horizontal scaling improves throughput for web applications significantly.",
            agent="agent1",
        )
        # Second analyze with support marker and overlapping words
        result = analyzer.analyze(
            "This evidence supports the claim that horizontal scaling improves throughput significantly.",
            agent="agent2",
        )
        support_rels = [r for r in result.claim_relationships if r.relation == "supports"]
        assert len(support_rels) >= 1
        assert support_rels[0].agent == "agent2"

    def test_contradiction_relationship_with_history(self):
        analyzer = StructuralAnalyzer()
        analyzer.analyze(
            "Microservices are the best architecture for large distributed systems.",
            agent="agent1",
        )
        result = analyzer.analyze(
            "This contradicts the idea that microservices are the best architecture for systems.",
            agent="agent2",
        )
        contra_rels = [r for r in result.claim_relationships if r.relation == "contradicts"]
        assert len(contra_rels) >= 1

    def test_refinement_relationship_with_history(self):
        analyzer = StructuralAnalyzer()
        analyzer.analyze(
            "Caching is helpful for improving database performance in applications.",
            agent="agent1",
        )
        result = analyzer.analyze(
            "This refines the point about caching being helpful for improving database performance.",
            agent="agent2",
        )
        refine_rels = [r for r in result.claim_relationships if r.relation == "refines"]
        assert len(refine_rels) >= 1


class TestStructuralAnalyzerWordOverlap:
    def test_identical_texts_return_one(self):
        analyzer = StructuralAnalyzer()
        assert analyzer._word_overlap("the cat sat on the mat", "the cat sat on the mat") == 1.0

    def test_no_overlap_returns_zero(self):
        analyzer = StructuralAnalyzer()
        # Words must be >= 3 chars to count
        assert analyzer._word_overlap("aaa bbb ccc", "ddd eee fff") == 0.0

    def test_partial_overlap(self):
        analyzer = StructuralAnalyzer()
        overlap = analyzer._word_overlap(
            "the database handles queries efficiently",
            "the database processes data efficiently",
        )
        assert 0.0 < overlap < 1.0

    def test_empty_text_returns_zero(self):
        analyzer = StructuralAnalyzer()
        assert analyzer._word_overlap("", "some text here") == 0.0
        assert analyzer._word_overlap("hello world", "") == 0.0

    def test_short_words_excluded(self):
        analyzer = StructuralAnalyzer()
        # "a", "an", "is" are < 3 chars, excluded
        assert analyzer._word_overlap("a an is", "a an is") == 0.0


class TestStructuralAnalyzerAggregation:
    def test_get_all_fallacies_aggregates_across_analyses(self):
        analyzer = StructuralAnalyzer()
        analyzer.analyze(
            "You clearly don't understand this topic at all. Your ignorance is evident.",
            agent="a1",
        )
        analyzer.analyze(
            "So you're saying that we should do nothing? What you really mean is giving up.",
            agent="a2",
        )
        all_fallacies = analyzer.get_all_fallacies()
        assert len(all_fallacies) >= 2

    def test_get_fallacy_summary_counts_by_type(self):
        analyzer = StructuralAnalyzer()
        analyzer.analyze(
            "You clearly don't understand this topic. Your ignorance is appalling.",
            agent="a1",
        )
        summary = analyzer.get_fallacy_summary()
        assert isinstance(summary, dict)
        # ad_hominem should be present
        assert "ad_hominem" in summary
        assert summary["ad_hominem"] >= 1

    def test_reset_clears_all_state(self):
        analyzer = StructuralAnalyzer()
        analyzer.analyze(
            "You clearly don't understand this. Your bias is obvious here.",
            agent="a1",
        )
        assert len(analyzer._all_results) >= 1
        analyzer.reset()
        assert len(analyzer._all_results) == 0
        assert len(analyzer._claim_history) == 0
        assert len(analyzer.get_all_fallacies()) == 0


# ---------------------------------------------------------------------------
# RhetoricalAnalysisObserver tests
# ---------------------------------------------------------------------------


class TestObserverShortContent:
    def test_short_content_returns_empty_list(self):
        observer = RhetoricalAnalysisObserver()
        result = observer.observe("claude", "short")
        assert result == []

    def test_empty_content_returns_empty_list(self):
        observer = RhetoricalAnalysisObserver()
        result = observer.observe("claude", "")
        assert result == []


class TestObserverPatternDetection:
    """Test detection of each rhetorical pattern."""

    def test_concession_detected(self):
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        result = observer.observe(
            "claude",
            "I acknowledge that you make a fair point about the architecture design.",
        )
        patterns = [o.pattern for o in result]
        assert RhetoricalPattern.CONCESSION in patterns

    def test_rebuttal_detected(self):
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        result = observer.observe(
            "claude",
            "However, I would argue that this approach is fundamentally flawed in practice.",
        )
        patterns = [o.pattern for o in result]
        assert RhetoricalPattern.REBUTTAL in patterns

    def test_synthesis_detected(self):
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        result = observer.observe(
            "claude",
            "Combining both perspectives, we can find common ground between these views and integrate the approaches.",
        )
        patterns = [o.pattern for o in result]
        assert RhetoricalPattern.SYNTHESIS in patterns

    def test_appeal_to_authority_detected(self):
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        result = observer.observe(
            "claude",
            "According to the latest research, experts say that best practices recommend this approach.",
        )
        patterns = [o.pattern for o in result]
        assert RhetoricalPattern.APPEAL_TO_AUTHORITY in patterns

    def test_appeal_to_evidence_detected(self):
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        result = observer.observe(
            "claude",
            "For example, the evidence suggests that this pattern works. Data shows improvements specifically.",
        )
        patterns = [o.pattern for o in result]
        assert RhetoricalPattern.APPEAL_TO_EVIDENCE in patterns

    def test_technical_depth_detected(self):
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        result = observer.observe(
            "claude",
            "The implementation uses async threading with O(n log n) complexity for the algorithm design.",
        )
        patterns = [o.pattern for o in result]
        assert RhetoricalPattern.TECHNICAL_DEPTH in patterns

    def test_rhetorical_question_detected(self):
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        result = observer.observe(
            "claude",
            "What if we considered a completely different approach? Shouldn't we think about this more carefully?",
        )
        patterns = [o.pattern for o in result]
        assert RhetoricalPattern.RHETORICAL_QUESTION in patterns

    def test_analogy_detected(self):
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        result = observer.observe(
            "claude",
            "Think of it as a pipeline, similar to how water flows through pipes in the plumbing system.",
        )
        patterns = [o.pattern for o in result]
        assert RhetoricalPattern.ANALOGY in patterns

    def test_qualification_detected(self):
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        result = observer.observe(
            "claude",
            "It depends on the context. Typically this approach works, but in some cases it may fail.",
        )
        patterns = [o.pattern for o in result]
        assert RhetoricalPattern.QUALIFICATION in patterns


class TestObserverMinConfidenceFiltering:
    def test_high_min_confidence_filters_weak_patterns(self):
        observer = RhetoricalAnalysisObserver(min_confidence=0.95)
        result = observer.observe(
            "claude",
            "However, I think there might be another approach to this problem.",
        )
        # With very high threshold, weak signals should be filtered out
        assert len(result) == 0

    def test_low_min_confidence_allows_more_patterns(self):
        observer_low = RhetoricalAnalysisObserver(min_confidence=0.15)
        observer_high = RhetoricalAnalysisObserver(min_confidence=0.8)
        content = "However, this approach is not quite right. I acknowledge the valid point."
        low_result = observer_low.observe("claude", content)
        high_result = observer_high.observe("gpt", content)
        assert len(low_result) >= len(high_result)


class TestObserverBroadcastCallback:
    def test_broadcast_called_with_observations(self):
        callback = MagicMock()
        observer = RhetoricalAnalysisObserver(
            broadcast_callback=callback, min_confidence=0.3
        )
        observer.observe(
            "claude",
            "I acknowledge that you make a fair point about the architecture design.",
        )
        assert callback.called
        call_args = callback.call_args[0][0]
        assert call_args["type"] == "rhetorical_observations"
        assert call_args["data"]["agent"] == "claude"
        assert len(call_args["data"]["observations"]) >= 1

    def test_broadcast_not_called_when_no_observations(self):
        callback = MagicMock()
        observer = RhetoricalAnalysisObserver(
            broadcast_callback=callback, min_confidence=0.99
        )
        observer.observe("claude", "A simple statement with no rhetorical patterns at all.")
        callback.assert_not_called()

    def test_broadcast_callback_error_swallowed(self):
        callback = MagicMock(side_effect=RuntimeError("broadcast failed"))
        observer = RhetoricalAnalysisObserver(
            broadcast_callback=callback, min_confidence=0.3
        )
        # Should not raise
        result = observer.observe(
            "claude",
            "I acknowledge that you raise a fair point about design here.",
        )
        # Observations are still returned even when broadcast fails
        assert len(result) >= 1


class TestObserverAgentPatternTracking:
    def test_agent_patterns_tracked(self):
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        observer.observe(
            "claude",
            "I acknowledge that you make a fair point about the design approach.",
        )
        observer.observe(
            "claude",
            "However, I would argue the alternative is significantly better.",
        )
        assert "claude" in observer.agent_patterns
        patterns = observer.agent_patterns["claude"]
        assert len(patterns) >= 1

    def test_multiple_agents_tracked_separately(self):
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        observer.observe(
            "claude",
            "I acknowledge that this is a valid and fair point about architecture.",
        )
        observer.observe(
            "gpt4",
            "However, I would argue that the opposite conclusion is warranted.",
        )
        assert "claude" in observer.agent_patterns
        assert "gpt4" in observer.agent_patterns


class TestObserverDebateDynamics:
    def test_empty_observations_returns_minimal_dynamics(self):
        observer = RhetoricalAnalysisObserver()
        dynamics = observer.get_debate_dynamics()
        assert dynamics["total_observations"] == 0
        assert dynamics["patterns_detected"] == {}
        assert dynamics["agent_styles"] == {}
        assert dynamics["dominant_pattern"] is None

    def test_collaborative_debate_character(self):
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        # Feed concession + synthesis heavy content
        for _ in range(3):
            observer.observe(
                "claude",
                "I acknowledge your fair point. Combining both perspectives, we find common ground and integrate.",
            )
        dynamics = observer.get_debate_dynamics()
        assert dynamics["debate_character"] == "collaborative"

    def test_contentious_debate_character(self):
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        for _ in range(5):
            observer.observe(
                "claude",
                "However, I disagree with that assessment. On the contrary, it is wrong. Actually, in fact you are mistaken.",
            )
        dynamics = observer.get_debate_dynamics()
        assert dynamics["debate_character"] == "contentious"

    def test_technical_debate_character(self):
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        for _ in range(5):
            observer.observe(
                "claude",
                "The implementation uses async threading with O(n log n) complexity for the algorithm architecture.",
            )
        dynamics = observer.get_debate_dynamics()
        assert dynamics["debate_character"] == "technical"

    def test_evidence_driven_debate_character(self):
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        for _ in range(5):
            observer.observe(
                "claude",
                "According to the latest research, studies show improvement. For example, data shows significant gains specifically.",
            )
        dynamics = observer.get_debate_dynamics()
        assert dynamics["debate_character"] == "evidence-driven"

    def test_balanced_debate_character(self):
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        # Mix of patterns - no single one dominates > 40%
        observer.observe(
            "claude",
            "I acknowledge your fair point about the design of this system.",
        )
        observer.observe(
            "gpt4",
            "However, I would argue that we need a different approach entirely.",
        )
        observer.observe(
            "gemini",
            "It depends on the context. Typically this works, but in some cases it fails.",
        )
        observer.observe(
            "mistral",
            "Think of it as a pipeline, similar to how water flows through plumbing systems.",
        )
        dynamics = observer.get_debate_dynamics()
        assert dynamics["debate_character"] == "balanced"

    def test_emerging_debate_character_empty_counts(self):
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        result = observer._characterize_debate({})
        assert result == "emerging"


class TestObserverPatternToStyle:
    @pytest.mark.parametrize(
        "pattern,expected_style",
        [
            ("concession", "diplomatic"),
            ("rebuttal", "combative"),
            ("synthesis", "collaborative"),
            ("appeal_to_authority", "scholarly"),
            ("appeal_to_evidence", "empirical"),
            ("technical_depth", "technical"),
            ("rhetorical_question", "socratic"),
            ("analogy", "illustrative"),
            ("qualification", "nuanced"),
            ("unknown_pattern", "balanced"),
        ],
    )
    def test_pattern_to_style_mapping(self, pattern, expected_style):
        observer = RhetoricalAnalysisObserver()
        assert observer._pattern_to_style(pattern) == expected_style


class TestObserverStructuralIntegration:
    def test_structural_analyzer_called_during_observe(self):
        analyzer = StructuralAnalyzer()
        observer = RhetoricalAnalysisObserver(
            structural_analyzer=analyzer, min_confidence=0.3
        )
        observer.observe(
            "claude",
            "I acknowledge your fair point. You clearly don't understand this topic and your ignorance shows.",
        )
        # Structural results should be populated
        results = observer.get_structural_results()
        assert len(results) >= 1

    def test_structural_confidence_enriches_observations(self):
        analyzer = StructuralAnalyzer()
        observer = RhetoricalAnalysisObserver(
            structural_analyzer=analyzer, min_confidence=0.3
        )
        # Content that triggers both rhetorical patterns and structural analysis
        content = (
            "I acknowledge your fair point. "
            "You clearly don't understand this and your ignorance is obvious. "
            "Your bias prevents you from seeing the truth."
        )
        observations = observer.observe("claude", content)
        # When structural analyzer returns confidence > 0, observations
        # should use max(keyword_conf, structural_conf)
        if observations:
            assert all(o.confidence >= 0.3 for o in observations)

    def test_structural_analyzer_error_swallowed(self):
        analyzer = MagicMock()
        analyzer.analyze.side_effect = RuntimeError("analysis failed")
        observer = RhetoricalAnalysisObserver(
            structural_analyzer=analyzer, min_confidence=0.3
        )
        # Should not raise
        result = observer.observe(
            "claude",
            "I acknowledge that this is a fair point about the system.",
        )
        # Observations still returned (just without structural enrichment)
        assert isinstance(result, list)


class TestObserverRecentObservations:
    def test_get_recent_observations_respects_limit(self):
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        for i in range(5):
            observer.observe(
                f"agent{i}",
                "I acknowledge that you make a very fair point about this design.",
            )
        recent = observer.get_recent_observations(limit=2)
        assert len(recent) <= 2
        # Should be the last 2 observations (as dicts)
        assert all(isinstance(r, dict) for r in recent)

    def test_get_recent_observations_default_limit(self):
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        for i in range(15):
            observer.observe(
                f"agent{i}",
                "I acknowledge that you make a really fair point about this design.",
            )
        recent = observer.get_recent_observations()
        assert len(recent) <= 10


class TestObserverReset:
    def test_reset_clears_all_state(self):
        analyzer = StructuralAnalyzer()
        observer = RhetoricalAnalysisObserver(
            structural_analyzer=analyzer, min_confidence=0.3
        )
        observer.observe(
            "claude",
            "I acknowledge your fair point. You clearly don't understand though.",
        )
        assert len(observer.observations) >= 1
        assert len(observer.agent_patterns) >= 1

        observer.reset()
        assert observer.observations == []
        assert observer.agent_patterns == {}
        assert observer._structural_results == []
        # Structural analyzer should also be reset
        assert len(analyzer._all_results) == 0
        assert len(analyzer._claim_history) == 0


# ---------------------------------------------------------------------------
# Global function tests
# ---------------------------------------------------------------------------


class TestGlobalFunctions:
    def setup_method(self):
        reset_rhetorical_observer()

    def teardown_method(self):
        reset_rhetorical_observer()

    def test_get_rhetorical_observer_returns_singleton(self):
        obs1 = get_rhetorical_observer()
        obs2 = get_rhetorical_observer()
        assert obs1 is obs2

    def test_reset_rhetorical_observer_clears_singleton(self):
        obs1 = get_rhetorical_observer()
        reset_rhetorical_observer()
        obs2 = get_rhetorical_observer()
        assert obs1 is not obs2

    def test_get_rhetorical_observer_returns_correct_type(self):
        obs = get_rhetorical_observer()
        assert isinstance(obs, RhetoricalAnalysisObserver)
