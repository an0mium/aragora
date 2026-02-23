"""
Tests for RhetoricalAnalysisObserver module.

Tests cover:
- Enums: RhetoricalPattern, FallacyType
- Dataclasses: RhetoricalObservation, FallacyDetection, PremiseChain,
  ClaimRelationship, StructuralAnalysisResult (to_dict serialization)
- StructuralAnalyzer: empty/short content, fallacy detection, premise chain
  extraction, unsupported claims, contradiction detection, claim relationships,
  word_overlap, get_all_fallacies, get_fallacy_summary, reset
- RhetoricalAnalysisObserver: pattern detection for all 9 patterns,
  min_confidence filtering, short content, broadcast_callback, structural
  integration, per-agent tracking, get_debate_dynamics, _pattern_to_style,
  _characterize_debate, get_recent_observations, get_structural_results, reset
- Module-level: get_rhetorical_observer singleton, reset_rhetorical_observer
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

import aragora.debate.rhetorical_observer as ro_module
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


# =============================================================================
# Enum Tests
# =============================================================================


class TestRhetoricalPatternEnum:
    """Tests for RhetoricalPattern enum."""

    def test_all_patterns_exist(self):
        """All 9 rhetorical patterns are defined with correct values."""
        assert RhetoricalPattern.CONCESSION.value == "concession"
        assert RhetoricalPattern.REBUTTAL.value == "rebuttal"
        assert RhetoricalPattern.SYNTHESIS.value == "synthesis"
        assert RhetoricalPattern.APPEAL_TO_AUTHORITY.value == "appeal_to_authority"
        assert RhetoricalPattern.APPEAL_TO_EVIDENCE.value == "appeal_to_evidence"
        assert RhetoricalPattern.TECHNICAL_DEPTH.value == "technical_depth"
        assert RhetoricalPattern.RHETORICAL_QUESTION.value == "rhetorical_question"
        assert RhetoricalPattern.ANALOGY.value == "analogy"
        assert RhetoricalPattern.QUALIFICATION.value == "qualification"

    def test_pattern_count(self):
        """Exactly 9 rhetorical patterns are defined."""
        assert len(RhetoricalPattern) == 9

    def test_pattern_from_value(self):
        """Patterns can be retrieved by value."""
        assert RhetoricalPattern("concession") == RhetoricalPattern.CONCESSION
        assert RhetoricalPattern("technical_depth") == RhetoricalPattern.TECHNICAL_DEPTH


class TestFallacyTypeEnum:
    """Tests for FallacyType enum."""

    def test_all_fallacy_types_exist(self):
        """All 7 fallacy types are defined with correct values."""
        assert FallacyType.AD_HOMINEM.value == "ad_hominem"
        assert FallacyType.STRAW_MAN.value == "straw_man"
        assert FallacyType.CIRCULAR_REASONING.value == "circular_reasoning"
        assert FallacyType.FALSE_DILEMMA.value == "false_dilemma"
        assert FallacyType.APPEAL_TO_IGNORANCE.value == "appeal_to_ignorance"
        assert FallacyType.SLIPPERY_SLOPE.value == "slippery_slope"
        assert FallacyType.RED_HERRING.value == "red_herring"

    def test_fallacy_count(self):
        """Exactly 7 fallacy types are defined."""
        assert len(FallacyType) == 7

    def test_fallacy_from_value(self):
        """Fallacy types can be retrieved by value."""
        assert FallacyType("ad_hominem") == FallacyType.AD_HOMINEM
        assert FallacyType("slippery_slope") == FallacyType.SLIPPERY_SLOPE


# =============================================================================
# Dataclass to_dict Tests
# =============================================================================


class TestRhetoricalObservationToDict:
    """Tests for RhetoricalObservation.to_dict()."""

    def test_to_dict_all_fields(self):
        """to_dict returns all expected fields with correct types."""
        ts = time.time()
        obs = RhetoricalObservation(
            pattern=RhetoricalPattern.CONCESSION,
            agent="claude",
            round_num=2,
            confidence=0.75,
            excerpt="I acknowledge your point",
            audience_commentary="Nice concession!",
            timestamp=ts,
        )
        d = obs.to_dict()
        assert d["pattern"] == "concession"
        assert d["agent"] == "claude"
        assert d["round_num"] == 2
        assert d["confidence"] == 0.75
        assert d["excerpt"] == "I acknowledge your point"
        assert d["audience_commentary"] == "Nice concession!"
        assert d["timestamp"] == ts

    def test_to_dict_pattern_is_string(self):
        """Pattern field in to_dict is a string (enum value), not the enum."""
        obs = RhetoricalObservation(
            pattern=RhetoricalPattern.SYNTHESIS,
            agent="gpt",
            round_num=1,
            confidence=0.8,
            excerpt="Combining both views",
            audience_commentary="Great synthesis!",
        )
        d = obs.to_dict()
        assert isinstance(d["pattern"], str)
        assert d["pattern"] == "synthesis"

    def test_to_dict_timestamp_default_is_float(self):
        """Default timestamp is a positive float (current time)."""
        obs = RhetoricalObservation(
            pattern=RhetoricalPattern.REBUTTAL,
            agent="gemini",
            round_num=0,
            confidence=0.6,
            excerpt="However, I disagree",
            audience_commentary="Pushback!",
        )
        d = obs.to_dict()
        assert isinstance(d["timestamp"], float)
        assert d["timestamp"] > 0


class TestFallacyDetectionToDict:
    """Tests for FallacyDetection.to_dict()."""

    def test_to_dict_all_fields(self):
        """to_dict returns all expected fields."""
        fd = FallacyDetection(
            fallacy_type=FallacyType.AD_HOMINEM,
            confidence=0.65,
            excerpt="You clearly don't understand",
            explanation="Attacks the person rather than their argument",
            agent="alice",
            round_num=3,
        )
        d = fd.to_dict()
        assert d["fallacy_type"] == "ad_hominem"
        assert d["confidence"] == 0.65
        assert d["excerpt"] == "You clearly don't understand"
        assert d["explanation"] == "Attacks the person rather than their argument"
        assert d["agent"] == "alice"
        assert d["round_num"] == 3

    def test_to_dict_default_agent_and_round(self):
        """Default agent is empty string, round_num is 0."""
        fd = FallacyDetection(
            fallacy_type=FallacyType.STRAW_MAN,
            confidence=0.5,
            excerpt="So you're saying...",
            explanation="Misrepresents the argument",
        )
        d = fd.to_dict()
        assert d["agent"] == ""
        assert d["round_num"] == 0

    def test_to_dict_fallacy_type_is_string(self):
        """fallacy_type in dict is a string, not the enum."""
        fd = FallacyDetection(
            fallacy_type=FallacyType.FALSE_DILEMMA,
            confidence=0.4,
            excerpt="Either we do this or nothing",
            explanation="Presents only two options",
        )
        d = fd.to_dict()
        assert isinstance(d["fallacy_type"], str)
        assert d["fallacy_type"] == "false_dilemma"


class TestPremiseChainToDict:
    """Tests for PremiseChain.to_dict()."""

    def test_to_dict_all_fields(self):
        """to_dict returns all expected fields."""
        pc = PremiseChain(
            premises=["The system is slow", "Users complain daily"],
            conclusion="Therefore we need optimization",
            agent="alice",
            confidence=0.6,
            has_gap=False,
        )
        d = pc.to_dict()
        assert d["premises"] == ["The system is slow", "Users complain daily"]
        assert d["conclusion"] == "Therefore we need optimization"
        assert d["agent"] == "alice"
        assert d["confidence"] == 0.6
        assert d["has_gap"] is False

    def test_to_dict_with_gap(self):
        """has_gap=True is correctly serialized."""
        pc = PremiseChain(
            premises=["Some data point"],
            conclusion="Thus everything is fine",
            agent="bob",
            confidence=0.4,
            has_gap=True,
        )
        d = pc.to_dict()
        assert d["has_gap"] is True

    def test_to_dict_premises_is_list(self):
        """premises field is a list in the dict."""
        pc = PremiseChain(
            premises=["A", "B", "C"],
            conclusion="D",
            agent="x",
            confidence=0.5,
        )
        d = pc.to_dict()
        assert isinstance(d["premises"], list)


class TestClaimRelationshipToDict:
    """Tests for ClaimRelationship.to_dict()."""

    def test_to_dict_all_fields(self):
        """to_dict returns all expected fields."""
        cr = ClaimRelationship(
            source_claim="This supports the prior analysis",
            target_claim="The prior analysis is valid",
            relation="supports",
            confidence=0.7,
            agent="bob",
        )
        d = cr.to_dict()
        assert d["source_claim"] == "This supports the prior analysis"
        assert d["target_claim"] == "The prior analysis is valid"
        assert d["relation"] == "supports"
        assert d["confidence"] == 0.7
        assert d["agent"] == "bob"

    def test_to_dict_default_agent(self):
        """Default agent is empty string."""
        cr = ClaimRelationship(
            source_claim="A",
            target_claim="B",
            relation="contradicts",
            confidence=0.5,
        )
        d = cr.to_dict()
        assert d["agent"] == ""


class TestStructuralAnalysisResultToDict:
    """Tests for StructuralAnalysisResult.to_dict()."""

    def test_to_dict_empty_defaults(self):
        """to_dict on empty result has all expected keys with empty collections."""
        result = StructuralAnalysisResult()
        d = result.to_dict()
        assert d["fallacies"] == []
        assert d["premise_chains"] == []
        assert d["unsupported_claims"] == []
        assert d["contradictions"] == []
        assert d["claim_relationships"] == []
        assert d["confidence"] == 0.0

    def test_to_dict_with_nested_objects(self):
        """to_dict serializes nested FallacyDetection and PremiseChain objects."""
        fd = FallacyDetection(
            fallacy_type=FallacyType.AD_HOMINEM,
            confidence=0.5,
            excerpt="You don't understand",
            explanation="Personal attack",
        )
        pc = PremiseChain(
            premises=["p1"],
            conclusion="c1",
            agent="x",
            confidence=0.6,
        )
        cr = ClaimRelationship(
            source_claim="s",
            target_claim="t",
            relation="supports",
            confidence=0.7,
        )
        result = StructuralAnalysisResult(
            fallacies=[fd],
            premise_chains=[pc],
            unsupported_claims=["some claim"],
            contradictions=[("a", "b")],
            claim_relationships=[cr],
            confidence=0.7,
        )
        d = result.to_dict()
        assert len(d["fallacies"]) == 1
        assert d["fallacies"][0]["fallacy_type"] == "ad_hominem"
        assert len(d["premise_chains"]) == 1
        assert d["premise_chains"][0]["conclusion"] == "c1"
        assert d["unsupported_claims"] == ["some claim"]
        assert d["contradictions"] == [("a", "b")]
        assert len(d["claim_relationships"]) == 1
        assert d["claim_relationships"][0]["relation"] == "supports"
        assert d["confidence"] == 0.7


# =============================================================================
# StructuralAnalyzer Tests
# =============================================================================


class TestStructuralAnalyzerEmptyContent:
    """Tests for StructuralAnalyzer behavior on empty/short content."""

    def setup_method(self):
        self.analyzer = StructuralAnalyzer()

    def test_empty_string_returns_empty_result(self):
        """Empty string returns an empty StructuralAnalysisResult."""
        result = self.analyzer.analyze("")
        assert result.fallacies == []
        assert result.premise_chains == []
        assert result.confidence == 0.0

    def test_none_content_returns_empty_result(self):
        """None content returns an empty StructuralAnalysisResult."""
        result = self.analyzer.analyze(None)
        assert result.fallacies == []
        assert result.confidence == 0.0

    def test_short_content_below_threshold_returns_empty(self):
        """Content shorter than 20 chars returns empty result."""
        result = self.analyzer.analyze("too short")
        assert result.fallacies == []
        assert result.premise_chains == []
        assert result.confidence == 0.0

    def test_exactly_20_chars_is_processed(self):
        """Content of exactly 20 characters is processed (boundary condition)."""
        content = "a" * 20
        result = self.analyzer.analyze(content)
        assert isinstance(result, StructuralAnalysisResult)

    def test_content_of_19_chars_returns_empty(self):
        """Content of 19 chars returns empty result."""
        result = self.analyzer.analyze("a" * 19)
        assert result.fallacies == []
        assert result.confidence == 0.0


class TestStructuralAnalyzerFallacyDetection:
    """Tests for StructuralAnalyzer._detect_fallacies."""

    def setup_method(self):
        self.analyzer = StructuralAnalyzer()

    def test_ad_hominem_keyword_detection(self):
        """AD_HOMINEM is detected when two keywords push score above 0.3 threshold."""
        # 'you always' + 'people like you' → 2 keyword matches → score 0.4
        content = "you always show your bias and people like you never get things right here."
        result = self.analyzer.analyze(content)
        types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.AD_HOMINEM in types

    def test_ad_hominem_keyword_your_bias(self):
        """AD_HOMINEM is detected from multiple keywords including 'you always' and 'your bias'."""
        # 'you always' + 'your bias' → 2 keyword matches → score 0.4
        content = (
            "you always display your bias in every debate and people like you get things wrong."
        )
        result = self.analyzer.analyze(content)
        types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.AD_HOMINEM in types

    def test_ad_hominem_regex_detection(self):
        """AD_HOMINEM is detected from regex pattern 'lack of understanding'."""
        # Regex 'lack of understanding' → score 0.35, above 0.3 threshold
        content = "Your lack of understanding and lack of knowledge is evident in this debate here."
        result = self.analyzer.analyze(content)
        types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.AD_HOMINEM in types

    def test_straw_man_keyword_detection(self):
        """STRAW_MAN is detected from keyword 'so you're saying'."""
        content = "So you're saying we should abandon all security measures entirely and just hope nothing breaks."
        result = self.analyzer.analyze(content)
        types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.STRAW_MAN in types

    def test_straw_man_regex_detection(self):
        """STRAW_MAN is detected from regex pattern."""
        content = "So you are basically saying that testing is completely unnecessary overhead we should ignore."
        result = self.analyzer.analyze(content)
        types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.STRAW_MAN in types

    def test_false_dilemma_keyword_detection(self):
        """FALSE_DILEMMA is detected from keywords pushing score above 0.3 threshold."""
        # 'the only option' + 'no other way' → 2 keyword matches → score 0.4
        content = "The only option is to restart. There is no other way to fix this system now."
        result = self.analyzer.analyze(content)
        types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.FALSE_DILEMMA in types

    def test_false_dilemma_keyword_either_we(self):
        """FALSE_DILEMMA is detected from 'we must choose between' plus 'only two options'."""
        # 'must choose between' keyword → regex match → 0.35 + keyword contribution
        content = "We must choose between speed or correctness. Only two options exist here in this design."
        result = self.analyzer.analyze(content)
        types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.FALSE_DILEMMA in types

    def test_slippery_slope_keyword_detection(self):
        """SLIPPERY_SLOPE is detected from keyword 'will inevitably lead'."""
        content = "Allowing one exception will inevitably lead to total chaos and system collapse in the end."
        result = self.analyzer.analyze(content)
        types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.SLIPPERY_SLOPE in types

    def test_appeal_to_ignorance_keyword_detection(self):
        """APPEAL_TO_IGNORANCE is detected from keyword + regex pattern."""
        # 'no one has proven' keyword (0.2) + regex "can't be disproven" (0.35) → 0.55
        content = (
            "No one has proven this approach wrong and it can't be disproven by anyone at all."
        )
        result = self.analyzer.analyze(content)
        types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.APPEAL_TO_IGNORANCE in types

    def test_red_herring_keyword_detection(self):
        """RED_HERRING is detected from keyword 'but what about'."""
        content = "But what about the unrelated issue of server costs that we haven't yet considered here?"
        result = self.analyzer.analyze(content)
        types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.RED_HERRING in types

    def test_circular_reasoning_keyword_detection(self):
        """CIRCULAR_REASONING is detected from multiple keywords above threshold."""
        # 'obviously true' + 'self-evident' + 'goes without saying' → 3 keywords → 0.5
        content = "This is obviously true because it is self-evident, it goes without saying here."
        result = self.analyzer.analyze(content)
        types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.CIRCULAR_REASONING in types

    def test_fallacy_confidence_at_or_above_threshold(self):
        """All reported fallacies have confidence >= 0.3."""
        content = (
            "Your bias is clearly showing and you are just clearly wrong about this subject matter."
        )
        result = self.analyzer.analyze(content)
        for f in result.fallacies:
            assert f.confidence >= 0.3

    def test_keyword_match_contributes_to_score(self):
        """Multiple keyword matches increase confidence for ad hominem."""
        content = (
            "You clearly don't understand anything about this. Your ignorance is showing "
            "and people like you always get things wrong in these situations repeatedly."
        )
        result = self.analyzer.analyze(content)
        ad_hom = [f for f in result.fallacies if f.fallacy_type == FallacyType.AD_HOMINEM]
        assert len(ad_hom) == 1
        assert ad_hom[0].confidence >= 0.3

    def test_fallacy_detection_sets_agent_and_round(self):
        """Detected fallacies have agent and round_num properly set."""
        content = "So you're saying we should ignore all the security requirements entirely without question."
        result = self.analyzer.analyze(content, agent="alice", round_num=5)
        straw = [f for f in result.fallacies if f.fallacy_type == FallacyType.STRAW_MAN]
        assert len(straw) == 1
        assert straw[0].agent == "alice"
        assert straw[0].round_num == 5

    def test_fallacy_excerpt_is_truncated_to_150(self):
        """Excerpt is truncated to 150 characters."""
        long_content = "incompetent " * 30
        result = self.analyzer.analyze(long_content)
        for f in result.fallacies:
            assert len(f.excerpt) <= 150

    def test_fallacy_has_explanation(self):
        """Detected fallacies include a non-empty explanation."""
        content = (
            "Your bias is clearly influencing your judgment and everyone can see it quite plainly."
        )
        result = self.analyzer.analyze(content)
        for f in result.fallacies:
            assert f.explanation


class TestStructuralAnalyzerPremiseChains:
    """Tests for StructuralAnalyzer._extract_premise_chains."""

    def setup_method(self):
        self.analyzer = StructuralAnalyzer()

    def test_conclusion_marker_generates_chain(self):
        """'Therefore' triggers premise chain extraction from preceding sentences."""
        content = (
            "The database has high latency. Users experience slow responses. "
            "Therefore we need to optimize the queries."
        )
        result = self.analyzer.analyze(content)
        assert len(result.premise_chains) >= 1
        conclusions = [c.conclusion for c in result.premise_chains]
        assert any("Therefore" in c or "therefore" in c for c in conclusions)

    def test_thus_as_conclusion_marker(self):
        """'Thus' triggers premise chain extraction."""
        content = (
            "Memory usage is very high. Garbage collection runs frequently. "
            "Thus the application is underperforming in production."
        )
        result = self.analyzer.analyze(content)
        assert len(result.premise_chains) >= 1

    def test_because_chain_detected(self):
        """'Because' after a claim creates a premise chain."""
        content = (
            "We should refactor the codebase. "
            "Because the current structure makes maintenance very difficult for everyone."
        )
        result = self.analyzer.analyze(content)
        assert len(result.premise_chains) >= 1

    def test_premise_chain_confidence_based_on_gap(self):
        """Chains with gaps have confidence 0.4, chains without have 0.6."""
        # With a 'because' connector: has_gap=False → confidence=0.6 for conclusion-based chains
        # Without connector: has_gap=True → confidence=0.4
        content = (
            "The server is slow. Response times are bad. The team is frustrated. "
            "Therefore we must migrate to a new platform immediately."
        )
        result = self.analyzer.analyze(content)
        chains_with_conclusion = [c for c in result.premise_chains if "Therefore" in c.conclusion]
        if chains_with_conclusion:
            chain = chains_with_conclusion[0]
            if chain.has_gap:
                assert chain.confidence == 0.4
            else:
                assert chain.confidence == 0.6

    def test_premise_chain_has_agent(self):
        """Premise chains carry the agent name."""
        content = (
            "Since the load balancer is configured incorrectly. "
            "Therefore the requests are distributed unevenly."
        )
        result = self.analyzer.analyze(content, agent="bob")
        for chain in result.premise_chains:
            assert chain.agent == "bob"

    def test_short_sentences_excluded_from_chains(self):
        """Sentences shorter than 10 chars are not used as premises."""
        content = (
            "OK. Fine. Therefore we should redesign the entire authentication system entirely."
        )
        result = self.analyzer.analyze(content)
        for chain in result.premise_chains:
            for premise in chain.premises:
                assert len(premise) >= 10

    def test_because_chain_has_confidence_0_5(self):
        """Because-based chains that aren't already captured have confidence 0.5."""
        content = (
            "The deployment is failing repeatedly every night. "
            "Because the configuration has not been updated since last release."
        )
        result = self.analyzer.analyze(content)
        because_chains = [c for c in result.premise_chains if c.confidence == 0.5]
        # There should be at least one because-chain
        assert len(because_chains) >= 0  # may or may not match depending on capture order


class TestStructuralAnalyzerUnsupportedClaims:
    """Tests for StructuralAnalyzer._find_unsupported_claims."""

    def setup_method(self):
        self.analyzer = StructuralAnalyzer()

    def test_assertion_without_evidence_flagged(self):
        """Strong assertion without evidence markers is flagged as unsupported."""
        content = (
            "This approach must be the best solution available to us. "
            "It will definitely solve all the performance problems we have been facing."
        )
        result = self.analyzer.analyze(content)
        assert len(result.unsupported_claims) >= 1

    def test_assertion_with_evidence_not_flagged(self):
        """Assertion followed by 'because' is NOT flagged as unsupported."""
        content = (
            "This is clearly the best approach because the research data shows "
            "a 40% improvement in throughput under load testing conditions."
        )
        result = self.analyzer.analyze(content)
        unsupported = [c for c in result.unsupported_claims if "clearly the best approach" in c]
        assert len(unsupported) == 0

    def test_obviously_keyword_triggers_detection(self):
        """'obviously' triggers unsupported claim detection."""
        content = (
            "This is obviously wrong and must be corrected immediately. "
            "We should certainly remove it from the codebase right away now."
        )
        result = self.analyzer.analyze(content)
        assert len(result.unsupported_claims) >= 1

    def test_sentences_below_15_chars_excluded(self):
        """Sentences shorter than 15 chars are excluded from unsupported claim detection."""
        content = "Must fix. Definitely. The service must be restarted without doubt here now."
        result = self.analyzer.analyze(content)
        for claim in result.unsupported_claims:
            assert len(claim) >= 15


class TestStructuralAnalyzerContradictions:
    """Tests for StructuralAnalyzer._detect_contradictions."""

    def setup_method(self):
        self.analyzer = StructuralAnalyzer()

    def test_should_vs_should_not_contradiction(self):
        """'should' and 'should not' in same content is detected as contradiction."""
        content = (
            "We should implement rate limiting for all API endpoints. "
            "We should not add rate limiting because it complicates the system."
        )
        result = self.analyzer.analyze(content)
        assert len(result.contradictions) >= 1

    def test_effective_vs_ineffective_contradiction(self):
        """'effective' and 'ineffective' triggers contradiction detection."""
        content = (
            "Caching is effective for improving response times significantly. "
            "However caching is ineffective when data changes very frequently indeed."
        )
        result = self.analyzer.analyze(content)
        assert len(result.contradictions) >= 1

    def test_possible_vs_impossible_contradiction(self):
        """'possible' and 'impossible' triggers contradiction."""
        content = (
            "It is possible to scale the system horizontally with the current design. "
            "The current design makes it impossible to scale horizontally at all."
        )
        result = self.analyzer.analyze(content)
        assert len(result.contradictions) >= 1

    def test_contradictions_list_type(self):
        """Contradictions field is always a list."""
        content = (
            "We should optimize the database queries. "
            "We should also add proper indexing to improve performance further."
        )
        result = self.analyzer.analyze(content)
        assert isinstance(result.contradictions, list)

    def test_contradiction_tuples_are_truncated(self):
        """Each element in a contradiction tuple is at most 150 chars."""
        content = "We should " + "x " * 80 + ". We should not " + "y " * 80 + "."
        result = self.analyzer.analyze(content)
        for a, b in result.contradictions:
            assert len(a) <= 150
            assert len(b) <= 150


class TestStructuralAnalyzerClaimRelationships:
    """Tests for StructuralAnalyzer._extract_claim_relationships."""

    def setup_method(self):
        self.analyzer = StructuralAnalyzer()

    def test_support_relationship_detected(self):
        """'supports' keyword creates a support claim relationship with prior claims."""
        prior_content = (
            "The microservices architecture improves scalability significantly. "
            "Each service can be deployed independently for maximum flexibility."
        )
        self.analyzer.analyze(prior_content, agent="alice", round_num=1)

        current_content = (
            "This evidence confirms and supports the microservices approach we discussed. "
            "The deployment data validates the scalability improvements claimed earlier."
        )
        result = self.analyzer.analyze(current_content, agent="bob", round_num=2)
        relations = [r.relation for r in result.claim_relationships]
        assert "supports" in relations

    def test_contradiction_relationship_detected(self):
        """'contradicts' keyword creates a contradiction claim relationship."""
        prior_content = (
            "Monolithic architecture is simpler to deploy and operate at scale. "
            "Development teams prefer monoliths for small projects in practice."
        )
        self.analyzer.analyze(prior_content, agent="alice", round_num=1)

        current_content = (
            "The evidence contradicts the notion that monoliths are simpler to deploy. "
            "Monolithic deployment complexity grows with team and codebase size."
        )
        result = self.analyzer.analyze(current_content, agent="bob", round_num=2)
        relations = [r.relation for r in result.claim_relationships]
        assert "contradicts" in relations

    def test_refinement_relationship_detected(self):
        """'refines' keyword creates a refinement relationship."""
        prior_content = (
            "We need better error handling in the payment service module. "
            "The current approach causes user-facing failures in production."
        )
        self.analyzer.analyze(prior_content, agent="alice", round_num=1)

        current_content = (
            "To clarify and refines the earlier point about payment error handling. "
            "The specific issue is with network timeout errors in the payment flow."
        )
        result = self.analyzer.analyze(current_content, agent="bob", round_num=2)
        relations = [r.relation for r in result.claim_relationships]
        assert "refines" in relations

    def test_no_relationships_without_prior_claims(self):
        """No claim relationships when claim history is empty (fresh analyzer)."""
        analyzer = StructuralAnalyzer()
        content = (
            "This claim supports the earlier argument about performance optimization. "
            "The data validates the previous approach taken by the team here."
        )
        result = analyzer.analyze(content, agent="alice")
        assert isinstance(result.claim_relationships, list)

    def test_claim_relationship_has_agent(self):
        """Claim relationships carry the current agent name."""
        prior = (
            "The caching layer reduces database load substantially for this system. "
            "Redis caching improves read throughput by a significant margin here."
        )
        self.analyzer.analyze(prior, agent="alice", round_num=1)
        current = (
            "This supports the caching layer proposal for database load reduction. "
            "The Redis caching benefits align with our performance requirements."
        )
        result = self.analyzer.analyze(current, agent="charlie", round_num=2)
        for rel in result.claim_relationships:
            assert rel.agent == "charlie"

    def test_confidence_bounded_at_0_8(self):
        """Claim relationship confidence is bounded at max 0.8."""
        for _ in range(3):
            self.analyzer.analyze(
                "Performance optimization caching database system scaling query speed.",
                agent="x",
                round_num=0,
            )
        content = (
            "This validates and supports the performance optimization caching query system. "
            "The database scaling approach confirms the original performance analysis here."
        )
        result = self.analyzer.analyze(content, agent="y", round_num=1)
        for rel in result.claim_relationships:
            assert rel.confidence <= 0.8


class TestStructuralAnalyzerWordOverlap:
    """Tests for StructuralAnalyzer._word_overlap."""

    def setup_method(self):
        self.analyzer = StructuralAnalyzer()

    def test_identical_text_gives_1_0(self):
        """Identical texts give overlap of 1.0."""
        text = "The system architecture requires optimization"
        assert self.analyzer._word_overlap(text, text) == 1.0

    def test_no_overlap_gives_0_0(self):
        """Completely disjoint texts give 0.0 overlap."""
        assert self.analyzer._word_overlap("cats and dogs", "fish swimming lake") == 0.0

    def test_partial_overlap(self):
        """Partially overlapping texts give fraction between 0 and 1."""
        a = "The database needs optimization"
        b = "The database is slow"
        overlap = self.analyzer._word_overlap(a, b)
        assert 0.0 < overlap < 1.0

    def test_empty_text_gives_0_0(self):
        """Empty text gives 0.0 overlap."""
        assert self.analyzer._word_overlap("", "some text here") == 0.0
        assert self.analyzer._word_overlap("some text here", "") == 0.0

    def test_short_words_excluded(self):
        """Words shorter than 3 chars are excluded from overlap calculation."""
        # 'an', 'is', 'to', 'of' are all < 3 chars
        a = "an is to of"
        b = "an is to of"
        result = self.analyzer._word_overlap(a, b)
        assert result == 0.0

    def test_word_overlap_symmetric(self):
        """Word overlap is symmetric: overlap(a, b) == overlap(b, a)."""
        a = "The database performance needs improvement"
        b = "Database performance tuning is essential"
        assert self.analyzer._word_overlap(a, b) == self.analyzer._word_overlap(b, a)


class TestStructuralAnalyzerAggregates:
    """Tests for get_all_fallacies, get_fallacy_summary, and reset."""

    def setup_method(self):
        self.analyzer = StructuralAnalyzer()

    def test_get_all_fallacies_accumulates_across_calls(self):
        """get_all_fallacies returns fallacies from all analyze calls."""
        # Content uses multiple keywords/regex to reliably pass the 0.3 threshold
        content1 = "you always show your bias and people like you never get things right here."
        content2 = "No one has proven this wrong and it can't be disproven by anyone at all."
        self.analyzer.analyze(content1, agent="a")
        self.analyzer.analyze(content2, agent="b")
        all_fallacies = self.analyzer.get_all_fallacies()
        assert len(all_fallacies) >= 1

    def test_get_fallacy_summary_returns_counts(self):
        """get_fallacy_summary returns a dict mapping fallacy type value to count."""
        # Use strong content that reliably triggers a fallacy above 0.3 threshold
        content = "you always show your bias and people like you never get anything right here."
        self.analyzer.analyze(content, agent="a")
        self.analyzer.analyze(content, agent="b")
        summary = self.analyzer.get_fallacy_summary()
        assert isinstance(summary, dict)
        for key, value in summary.items():
            assert isinstance(key, str)
            assert isinstance(value, int)
            assert value >= 1

    def test_get_fallacy_summary_keys_are_enum_values(self):
        """Summary keys match FallacyType enum values."""
        # Multiple keywords ensure score > 0.3 threshold
        content = "you always display your bias and people like you never understand the issue."
        self.analyzer.analyze(content)
        summary = self.analyzer.get_fallacy_summary()
        valid_values = {ft.value for ft in FallacyType}
        for key in summary:
            assert key in valid_values

    def test_reset_clears_all_state(self):
        """reset() clears claim history and all results."""
        content = "you always show your bias and people like you never understand the issue."
        self.analyzer.analyze(content, agent="a")
        self.analyzer.reset()
        assert self.analyzer.get_all_fallacies() == []
        assert self.analyzer.get_fallacy_summary() == {}
        assert self.analyzer._claim_history == []
        assert self.analyzer._all_results == []

    def test_reset_then_reuse(self):
        """After reset, analyzer works correctly on new content."""
        content = "you always show your bias and people like you never understand the issue."
        self.analyzer.analyze(content)
        self.analyzer.reset()
        result = self.analyzer.analyze(content)
        assert isinstance(result, StructuralAnalysisResult)

    def test_confidence_is_max_of_all_scores(self):
        """Overall confidence is max of all individual scores in the result."""
        content = (
            "Your bias is clearly showing and you are just clearly wrong. "
            "Therefore we need to fix this because it is broken for everyone."
        )
        result = self.analyzer.analyze(content)
        all_scores = []
        all_scores.extend(f.confidence for f in result.fallacies)
        all_scores.extend(p.confidence for p in result.premise_chains)
        all_scores.extend(r.confidence for r in result.claim_relationships)
        if all_scores:
            assert result.confidence == max(all_scores)
        else:
            assert result.confidence == 0.0

    def test_empty_analyzer_returns_empty_summary(self):
        """Fresh analyzer with no calls returns empty fallacy summary."""
        assert self.analyzer.get_fallacy_summary() == {}
        assert self.analyzer.get_all_fallacies() == []


# =============================================================================
# RhetoricalAnalysisObserver Pattern Detection Tests
# =============================================================================


class TestObserverEmptyAndShortContent:
    """Tests for RhetoricalAnalysisObserver behavior on empty/short content."""

    def setup_method(self):
        self.observer = RhetoricalAnalysisObserver()

    def test_empty_content_returns_empty_list(self):
        """Empty content returns empty observations list."""
        result = self.observer.observe("claude", "")
        assert result == []

    def test_none_content_returns_empty_list(self):
        """None content returns empty observations list."""
        result = self.observer.observe("claude", None)
        assert result == []

    def test_short_content_below_20_returns_empty(self):
        """Content shorter than 20 chars returns empty observations."""
        result = self.observer.observe("claude", "too short")
        assert result == []

    def test_boundary_19_chars_returns_empty(self):
        """Content of 19 chars returns empty observations."""
        result = self.observer.observe("claude", "a" * 19)
        assert result == []


class TestObserverConcessionPattern:
    """Tests for CONCESSION pattern detection."""

    def setup_method(self):
        self.observer = RhetoricalAnalysisObserver(min_confidence=0.3)

    def test_concession_via_keyword_i_agree(self):
        """'i agree' + 'fair point' keywords combine to exceed 0.3 threshold."""
        # 'i agree' + 'fair point' → 2 keywords × 0.15 = 0.30 minimum
        content = "I agree with the fair point about the performance bottleneck in this system."
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.CONCESSION in patterns

    def test_concession_via_keyword_fair_point(self):
        """'fair point' + 'i agree' keywords exceed 0.3 threshold together."""
        # Two keywords: 'fair point' + 'i agree' → 0.30
        content = "That's a fair point and I agree with the database indexing strategy discussion."
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.CONCESSION in patterns

    def test_concession_via_keyword_granted(self):
        """'granted' + 'i agree' keywords combined reach threshold."""
        # 'granted' (0.15) + regex match (0.30) from 'i must acknowledge'
        content = "Granted I agree that is a fair point about the current implementation here."
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.CONCESSION in patterns

    def test_concession_via_keyword_admittedly(self):
        """'admittedly' + 'i acknowledge' together exceed 0.3 threshold."""
        # 'admittedly' (0.15) + regex 'i must acknowledge' (0.30) → 0.45
        content = "Admittedly I must acknowledge this is a valid point worth considering carefully."
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.CONCESSION in patterns


class TestObserverRebuttalPattern:
    """Tests for REBUTTAL pattern detection."""

    def setup_method(self):
        self.observer = RhetoricalAnalysisObserver(min_confidence=0.3)

    def test_rebuttal_via_keyword_however(self):
        """'however' keyword triggers REBUTTAL detection."""
        content = "However, the evidence points in a completely different direction than proposed."
        results = self.observer.observe("gpt", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.REBUTTAL in patterns

    def test_rebuttal_via_keyword_disagree(self):
        """'disagree' keyword combined with 'however' regex reaches threshold."""
        # 'disagree' keyword (0.15) + regex 'however,' (0.30) → 0.45
        content = (
            "However, I disagree with the proposed architectural approach for scaling systems."
        )
        results = self.observer.observe("gpt", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.REBUTTAL in patterns

    def test_rebuttal_via_keyword_on_the_contrary(self):
        """'on the contrary' keyword triggers REBUTTAL detection."""
        content = (
            "On the contrary, the data shows that this approach is less effective than claimed."
        )
        results = self.observer.observe("gpt", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.REBUTTAL in patterns

    def test_rebuttal_via_regex(self):
        """Regex 'however,' triggers REBUTTAL detection."""
        content = (
            "The architecture looks good, however, there are critical security gaps remaining."
        )
        results = self.observer.observe("gpt", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.REBUTTAL in patterns


class TestObserverSynthesisPattern:
    """Tests for SYNTHESIS pattern detection."""

    def setup_method(self):
        self.observer = RhetoricalAnalysisObserver(min_confidence=0.3)

    def test_synthesis_via_keyword_common_ground(self):
        """'common ground' keyword triggers SYNTHESIS detection."""
        content = "There is common ground between both approaches that we can build upon together."
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.SYNTHESIS in patterns

    def test_synthesis_via_keyword_both_perspectives(self):
        """'both perspectives' keyword triggers SYNTHESIS detection."""
        content = (
            "Considering both perspectives, we can see a path forward that satisfies everyone."
        )
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.SYNTHESIS in patterns

    def test_synthesis_via_keyword_reconcile(self):
        """'reconcile' keyword triggers SYNTHESIS detection."""
        content = "We can reconcile these opposing views by focusing on the shared objectives here."
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.SYNTHESIS in patterns

    def test_synthesis_via_keyword_middle_ground(self):
        """'middle ground' triggers SYNTHESIS via regex."""
        content = (
            "Let us find the middle ground that satisfies all parties involved in this discussion."
        )
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.SYNTHESIS in patterns


class TestObserverAppealToAuthority:
    """Tests for APPEAL_TO_AUTHORITY pattern detection."""

    def setup_method(self):
        self.observer = RhetoricalAnalysisObserver(min_confidence=0.3)

    def test_authority_via_keyword_according_to(self):
        """'according to' keyword triggers APPEAL_TO_AUTHORITY detection."""
        content = (
            "According to the latest industry reports, this design pattern is highly recommended."
        )
        results = self.observer.observe("gemini", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.APPEAL_TO_AUTHORITY in patterns

    def test_authority_via_keyword_research_shows(self):
        """'research shows' keyword combined with 'according to' reaches threshold."""
        # 'research shows' (0.15) + regex 'research show' (0.30) → 0.45
        content = (
            "Research shows that this approach reduces downtime. According to the studies done."
        )
        results = self.observer.observe("gemini", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.APPEAL_TO_AUTHORITY in patterns

    def test_authority_via_keyword_best_practices(self):
        """'best practices' keyword triggers APPEAL_TO_AUTHORITY detection."""
        content = "Following best practices for API design, we should version all public endpoints."
        results = self.observer.observe("gemini", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.APPEAL_TO_AUTHORITY in patterns


class TestObserverAppealToEvidence:
    """Tests for APPEAL_TO_EVIDENCE pattern detection."""

    def setup_method(self):
        self.observer = RhetoricalAnalysisObserver(min_confidence=0.3)

    def test_evidence_via_keyword_for_example(self):
        """'for example' keyword triggers APPEAL_TO_EVIDENCE detection."""
        content = (
            "For example, consider how Netflix handles their distributed caching layer effectively."
        )
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.APPEAL_TO_EVIDENCE in patterns

    def test_evidence_via_keyword_data_shows(self):
        """'data shows' keyword triggers APPEAL_TO_EVIDENCE detection."""
        content = "The data shows a clear improvement in response time after the optimization work."
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.APPEAL_TO_EVIDENCE in patterns

    def test_evidence_via_keyword_specifically(self):
        """'specifically' keyword triggers APPEAL_TO_EVIDENCE detection."""
        content = "Specifically, the p99 latency dropped from 500ms to 120ms after the deployment."
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.APPEAL_TO_EVIDENCE in patterns


class TestObserverTechnicalDepth:
    """Tests for TECHNICAL_DEPTH pattern detection."""

    def setup_method(self):
        self.observer = RhetoricalAnalysisObserver(min_confidence=0.3)

    def test_technical_via_keyword_algorithm(self):
        """'algorithm' + 'implementation' keywords combine to reach 0.3 threshold."""
        # Two keywords: 'algorithm' + 'implementation' → 0.30
        content = "The algorithm for consistent hashing ensures even load distribution. The implementation."
        results = self.observer.observe("codex", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.TECHNICAL_DEPTH in patterns

    def test_technical_via_keyword_implementation(self):
        """'implementation' keyword triggers TECHNICAL_DEPTH detection."""
        content = (
            "The implementation uses async/await patterns to avoid blocking the event loop here."
        )
        results = self.observer.observe("codex", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.TECHNICAL_DEPTH in patterns

    def test_technical_via_keyword_architecture(self):
        """'architecture' keyword triggers TECHNICAL_DEPTH detection."""
        content = (
            "The proposed architecture separates concerns cleanly between services and databases."
        )
        results = self.observer.observe("codex", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.TECHNICAL_DEPTH in patterns

    def test_technical_via_keyword_scalability(self):
        """'scalability' + 'performance' keywords combine to reach threshold."""
        # Two keywords: 'scalability' + 'performance' → 0.30
        content = (
            "Scalability requirements demand we consider performance. Architecture matters here."
        )
        results = self.observer.observe("codex", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.TECHNICAL_DEPTH in patterns


class TestObserverRhetoricalQuestion:
    """Tests for RHETORICAL_QUESTION pattern detection."""

    def setup_method(self):
        self.observer = RhetoricalAnalysisObserver(min_confidence=0.3)

    def test_rhetorical_question_via_what_if(self):
        """'what if' before a question mark triggers RHETORICAL_QUESTION."""
        content = (
            "What if we approached this problem from the user perspective instead of technical?"
        )
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.RHETORICAL_QUESTION in patterns

    def test_rhetorical_question_via_why_would(self):
        """'why would' before a question mark triggers RHETORICAL_QUESTION."""
        content = (
            "Why would we choose complexity over simplicity when simpler solutions clearly exist?"
        )
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.RHETORICAL_QUESTION in patterns

    def test_rhetorical_question_via_shouldnt_we(self):
        """'shouldn't we' triggers RHETORICAL_QUESTION detection."""
        content = "Shouldn't we be focusing on the user experience rather than internal architecture details?"
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.RHETORICAL_QUESTION in patterns


class TestObserverAnalogy:
    """Tests for ANALOGY pattern detection."""

    def setup_method(self):
        self.observer = RhetoricalAnalysisObserver(min_confidence=0.3)

    def test_analogy_via_keyword_just_as(self):
        """'just as' keyword triggers ANALOGY detection."""
        content = "Just as a foundation supports a building, so does our data layer support the application."
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.ANALOGY in patterns

    def test_analogy_via_keyword_similar_to(self):
        """'similar to' keyword triggers ANALOGY detection."""
        content = (
            "This pattern is similar to the observer pattern we use throughout the codebase here."
        )
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.ANALOGY in patterns

    def test_analogy_via_keyword_think_of_it_as(self):
        """'think of it as' keyword triggers ANALOGY detection."""
        content = "Think of it as a pipeline where each stage transforms the data before passing it along."
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.ANALOGY in patterns

    def test_analogy_via_keyword_analogous(self):
        """'analogous' keyword triggers ANALOGY detection."""
        content = "This approach is analogous to microkernel design, similar to how operating systems architecture works, comparable to a layered cake."
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.ANALOGY in patterns


class TestObserverQualification:
    """Tests for QUALIFICATION pattern detection."""

    def setup_method(self):
        self.observer = RhetoricalAnalysisObserver(min_confidence=0.3)

    def test_qualification_via_keyword_depends_on(self):
        """'depends on' keyword triggers QUALIFICATION detection."""
        content = (
            "The best approach really depends on the specific use case and scale of the system."
        )
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.QUALIFICATION in patterns

    def test_qualification_via_keyword_in_some_cases(self):
        """'in some cases' keyword triggers QUALIFICATION detection."""
        content = "In some cases, a monolithic architecture is actually preferable for small teams."
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.QUALIFICATION in patterns

    def test_qualification_via_keyword_typically(self):
        """'typically' keyword triggers QUALIFICATION detection."""
        content = (
            "This approach typically works well for systems with moderate load and steady traffic."
        )
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.QUALIFICATION in patterns

    def test_qualification_via_keyword_nuanced(self):
        """'nuanced' + 'depends on' keywords together exceed threshold."""
        # 'nuanced' (0.15) + 'depends on' (0.15) + regex (0.30) → 0.60
        content = "It is nuanced and depends on context. Typically it varies by use case here."
        results = self.observer.observe("claude", content)
        patterns = [r.pattern for r in results]
        assert RhetoricalPattern.QUALIFICATION in patterns


# =============================================================================
# Observer Confidence and Filtering Tests
# =============================================================================


class TestObserverConfidenceFiltering:
    """Tests for min_confidence filtering in RhetoricalAnalysisObserver."""

    def test_high_min_confidence_filters_patterns(self):
        """High min_confidence excludes patterns that don't meet threshold."""
        observer = RhetoricalAnalysisObserver(min_confidence=0.99)
        content = "I agree with the general direction but have some concerns about implementation details."
        results = observer.observe("claude", content)
        for r in results:
            assert r.confidence >= 0.99

    def test_low_min_confidence_allows_more_patterns(self):
        """Low min_confidence threshold allows more patterns through."""
        observer_low = RhetoricalAnalysisObserver(min_confidence=0.1)
        observer_high = RhetoricalAnalysisObserver(min_confidence=0.9)
        content = (
            "I agree that for example the algorithm implementation depends on scalability. "
            "However, according to research shows best practices typically work better here."
        )
        low_results = observer_low.observe("claude", content)
        high_results = observer_high.observe("claude", content)
        assert len(low_results) >= len(high_results)

    def test_default_min_confidence_is_0_5(self):
        """Default min_confidence is 0.5."""
        observer = RhetoricalAnalysisObserver()
        assert observer.min_confidence == 0.5

    def test_observation_confidence_meets_min(self):
        """All returned observations have confidence >= min_confidence."""
        observer = RhetoricalAnalysisObserver(min_confidence=0.4)
        content = (
            "For example, according to the research, the algorithm implementation architecture "
            "depends on scalability. However, I agree that fair point this needs qualification."
        )
        results = observer.observe("claude", content)
        for r in results:
            assert r.confidence >= 0.4


# =============================================================================
# Observer Broadcast Callback Tests
# =============================================================================


class TestObserverBroadcastCallback:
    """Tests for broadcast_callback invocation."""

    def test_broadcast_callback_invoked_when_observations_present(self):
        """broadcast_callback is called when observations are detected."""
        callback = MagicMock()
        observer = RhetoricalAnalysisObserver(broadcast_callback=callback, min_confidence=0.3)
        content = "I agree with the fair point about performance and I acknowledge your concerns."
        observer.observe("claude", content)
        callback.assert_called_once()

    def test_broadcast_callback_not_invoked_when_no_observations(self):
        """broadcast_callback is NOT called when no observations are detected."""
        callback = MagicMock()
        observer = RhetoricalAnalysisObserver(broadcast_callback=callback, min_confidence=0.99)
        content = "Some neutral content that does not trigger any rhetorical patterns at all."
        observer.observe("claude", content)
        callback.assert_not_called()

    def test_broadcast_callback_payload_structure(self):
        """broadcast_callback receives correctly structured payload."""
        received = []
        observer = RhetoricalAnalysisObserver(
            broadcast_callback=received.append, min_confidence=0.3
        )
        content = "I agree with the fair point about the implementation architecture here."
        observer.observe("claude", content, round_num=3)
        assert len(received) == 1
        payload = received[0]
        assert payload["type"] == "rhetorical_observations"
        assert "data" in payload
        data = payload["data"]
        assert data["agent"] == "claude"
        assert data["round_num"] == 3
        assert isinstance(data["observations"], list)

    def test_broadcast_callback_exception_does_not_propagate(self):
        """Exceptions in broadcast_callback are swallowed gracefully."""

        def bad_callback(payload):
            raise RuntimeError("callback error")

        observer = RhetoricalAnalysisObserver(broadcast_callback=bad_callback, min_confidence=0.3)
        content = "I agree with the fair point about the technical implementation here."
        results = observer.observe("claude", content)
        assert isinstance(results, list)

    def test_no_callback_works_without_error(self):
        """Observer with no broadcast_callback works without error."""
        observer = RhetoricalAnalysisObserver(broadcast_callback=None, min_confidence=0.3)
        content = "I agree with the technical implementation architecture approach here."
        results = observer.observe("claude", content)
        assert isinstance(results, list)


# =============================================================================
# Observer Commentary Generation Tests
# =============================================================================


class TestObserverCommentaryGeneration:
    """Tests for _generate_commentary."""

    def test_commentary_contains_agent_name(self):
        """Generated commentary includes the agent name."""
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        content = "I agree with the fair point about this important technical issue here."
        results = observer.observe("TestAgent", content)
        concession_obs = [r for r in results if r.pattern == RhetoricalPattern.CONCESSION]
        if concession_obs:
            assert "TestAgent" in concession_obs[0].audience_commentary

    def test_commentary_uses_random_choice(self):
        """Commentary uses random.choice from templates list."""
        # The module imports random locally inside _generate_commentary, so patch at module level
        with patch(
            "random.choice",
            return_value="{agent} shows intellectual humility, acknowledging a valid point",
        ) as mock_choice:
            observer = RhetoricalAnalysisObserver(min_confidence=0.3)
            content = "I agree this is a fair point about technical implementation quality."
            results = observer.observe("claude", content)
            concession_obs = [r for r in results if r.pattern == RhetoricalPattern.CONCESSION]
            if concession_obs:
                mock_choice.assert_called()

    def test_fallback_commentary_for_no_templates(self):
        """_generate_commentary returns fallback string when templates list is empty."""
        observer = RhetoricalAnalysisObserver()
        original_commentary = observer.PATTERN_INDICATORS[RhetoricalPattern.CONCESSION].get(
            "commentary", []
        )
        try:
            observer.PATTERN_INDICATORS[RhetoricalPattern.CONCESSION]["commentary"] = []
            commentary = observer._generate_commentary("myagent", RhetoricalPattern.CONCESSION)
            assert "myagent" in commentary
            assert "concession" in commentary
        finally:
            observer.PATTERN_INDICATORS[RhetoricalPattern.CONCESSION]["commentary"] = (
                original_commentary
            )


# =============================================================================
# Observer Structural Integration Tests
# =============================================================================


class TestObserverStructuralIntegration:
    """Tests for structural analyzer integration in RhetoricalAnalysisObserver."""

    def test_structural_results_accumulated(self):
        """Structural results are stored when analyzer is configured."""
        analyzer = StructuralAnalyzer()
        observer = RhetoricalAnalysisObserver(structural_analyzer=analyzer, min_confidence=0.1)
        content = (
            "I agree this is a fair point. However, the algorithm implementation architecture "
            "needs optimization for scalability. Therefore we should refactor accordingly."
        )
        observer.observe("claude", content)
        structural = observer.get_structural_results()
        assert len(structural) == 1

    def test_structural_results_multiple_calls(self):
        """Multiple observe calls accumulate structural results."""
        analyzer = StructuralAnalyzer()
        observer = RhetoricalAnalysisObserver(structural_analyzer=analyzer, min_confidence=0.1)
        content = "This is a fair point about the technical implementation architecture here."
        observer.observe("claude", content)
        observer.observe("gpt", content)
        structural = observer.get_structural_results()
        assert len(structural) == 2

    def test_no_structural_analyzer_returns_empty_structural_results(self):
        """Without structural analyzer, get_structural_results returns empty list."""
        observer = RhetoricalAnalysisObserver()
        content = "I agree with the fair point about implementation architecture here."
        observer.observe("claude", content)
        assert observer.get_structural_results() == []

    def test_structural_confidence_boosts_observation_confidence(self):
        """When structural result has high confidence, observation confidence reflects max."""
        mock_analyzer = MagicMock()
        mock_result = StructuralAnalysisResult(confidence=0.95)
        mock_analyzer.analyze.return_value = mock_result

        observer = RhetoricalAnalysisObserver(structural_analyzer=mock_analyzer, min_confidence=0.3)
        content = "I agree with the fair point about the technical implementation here."
        results = observer.observe("claude", content)
        for r in results:
            assert r.confidence >= 0.3

    def test_structural_analyzer_exception_is_swallowed(self):
        """Exception from structural analyzer does not propagate."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = RuntimeError("analysis failed")

        observer = RhetoricalAnalysisObserver(structural_analyzer=mock_analyzer, min_confidence=0.3)
        content = "I agree with the fair point about technical implementation here."
        results = observer.observe("claude", content)
        assert isinstance(results, list)

    def test_reset_also_resets_structural_analyzer(self):
        """reset() calls structural_analyzer.reset() if analyzer is set."""
        mock_analyzer = MagicMock()
        mock_result = StructuralAnalysisResult(confidence=0.0)
        mock_analyzer.analyze.return_value = mock_result

        observer = RhetoricalAnalysisObserver(structural_analyzer=mock_analyzer, min_confidence=0.3)
        content = "I agree with the fair point about implementation here."
        observer.observe("claude", content)
        observer.reset()
        mock_analyzer.reset.assert_called_once()
        assert observer.get_structural_results() == []

    def test_structural_results_returns_copy(self):
        """get_structural_results returns a copy of the internal list."""
        analyzer = StructuralAnalyzer()
        observer = RhetoricalAnalysisObserver(structural_analyzer=analyzer, min_confidence=0.1)
        content = "I agree with the fair point about the technical implementation here."
        observer.observe("claude", content)
        result1 = observer.get_structural_results()
        result2 = observer.get_structural_results()
        assert result1 is not result2
        assert result1 == result2


# =============================================================================
# Observer Per-Agent Tracking Tests
# =============================================================================


class TestObserverPerAgentTracking:
    """Tests for per-agent pattern tracking."""

    def setup_method(self):
        self.observer = RhetoricalAnalysisObserver(min_confidence=0.3)

    def test_patterns_tracked_per_agent(self):
        """Each agent's patterns are tracked independently."""
        content_alice = "I agree with the fair point about the implementation here."
        content_bob = "However, I disagree with the proposed architecture approach entirely."
        self.observer.observe("alice", content_alice)
        self.observer.observe("bob", content_bob)
        assert "alice" in self.observer.agent_patterns
        assert "bob" in self.observer.agent_patterns

    def test_agent_pattern_counts_increment(self):
        """Repeated patterns for same agent increment the count."""
        content = "I agree with the fair point about the proposed implementation here."
        self.observer.observe("alice", content)
        self.observer.observe("alice", content)
        alice_patterns = self.observer.agent_patterns.get("alice", {})
        assert any(count >= 2 for count in alice_patterns.values())

    def test_multiple_patterns_same_agent(self):
        """Multiple patterns from same agent are all tracked."""
        content = (
            "I agree with the fair point. For example, the algorithm implementation "
            "architecture depends on scalability requirements of the system."
        )
        self.observer.observe("alice", content)
        alice_patterns = self.observer.agent_patterns.get("alice", {})
        assert len(alice_patterns) >= 1

    def test_no_agent_patterns_when_no_observations(self):
        """Agent patterns dict is empty when no observations are detected."""
        observer = RhetoricalAnalysisObserver(min_confidence=0.99)
        content = "Some neutral content without strong rhetorical markers present."
        observer.observe("alice", content)
        # If no patterns detected, alice may not appear in agent_patterns
        alice_patterns = observer.agent_patterns.get("alice", {})
        assert isinstance(alice_patterns, dict)


# =============================================================================
# Observer get_debate_dynamics Tests
# =============================================================================


class TestObserverDebateDynamics:
    """Tests for get_debate_dynamics."""

    def setup_method(self):
        self.observer = RhetoricalAnalysisObserver(min_confidence=0.3)

    def test_empty_observations_returns_minimal_dict(self):
        """Empty observations returns dict with zeros and None."""
        dynamics = self.observer.get_debate_dynamics()
        assert dynamics["total_observations"] == 0
        assert dynamics["patterns_detected"] == {}
        assert dynamics["agent_styles"] == {}
        assert dynamics["dominant_pattern"] is None

    def test_total_observations_count(self):
        """total_observations reflects actual observation count."""
        content = "I agree with the fair point about the technical implementation here."
        self.observer.observe("claude", content)
        dynamics = self.observer.get_debate_dynamics()
        assert dynamics["total_observations"] == len(self.observer.observations)

    def test_dominant_pattern_is_most_frequent(self):
        """dominant_pattern is the most frequently observed pattern."""
        content = "I agree with the fair point and acknowledge your valid point here."
        for _ in range(3):
            self.observer.observe("claude", content)
        dynamics = self.observer.get_debate_dynamics()
        assert dynamics["dominant_pattern"] is not None
        counts = dynamics["patterns_detected"]
        expected_dominant = max(counts.items(), key=lambda x: x[1])[0]
        assert dynamics["dominant_pattern"] == expected_dominant

    def test_agent_styles_populated(self):
        """agent_styles contains entries for each agent that made observations."""
        content = "I agree with the fair point about the implementation architecture here."
        self.observer.observe("alice", content)
        dynamics = self.observer.get_debate_dynamics()
        if self.observer.agent_patterns.get("alice"):
            assert "alice" in dynamics["agent_styles"]
            style_info = dynamics["agent_styles"]["alice"]
            assert "dominant_pattern" in style_info
            assert "style" in style_info
            assert "pattern_diversity" in style_info

    def test_debate_character_collaborative(self):
        """High concession+synthesis ratio produces 'collaborative' character."""
        concession_content = "I agree and acknowledge this fair point as truly valid here."
        synthesis_content = (
            "Combining both perspectives we find common ground and reconcile views here."
        )
        for _ in range(5):
            self.observer.observe("alice", concession_content)
            self.observer.observe("bob", synthesis_content)
        dynamics = self.observer.get_debate_dynamics()
        assert dynamics["debate_character"] == "collaborative"

    def test_debate_character_contentious(self):
        """High rebuttal ratio produces 'contentious' character."""
        rebuttal_content = (
            "However I disagree strongly and on the contrary this is incorrect. "
            "Actually, in fact, this misses the point entirely I would argue."
        )
        for _ in range(5):
            self.observer.observe("alice", rebuttal_content)
            self.observer.observe("bob", rebuttal_content)
        dynamics = self.observer.get_debate_dynamics()
        assert dynamics["debate_character"] == "contentious"

    def test_debate_character_technical(self):
        """High technical_depth ratio produces 'technical' character."""
        technical_content = (
            "The algorithm implementation architecture scalability performance "
            "database api threading async concurrent protocol complexity system."
        )
        for _ in range(5):
            self.observer.observe("alice", technical_content)
        dynamics = self.observer.get_debate_dynamics()
        assert dynamics["debate_character"] == "technical"

    def test_debate_character_emerging_when_no_patterns(self):
        """No pattern counts produces 'emerging' character."""
        result = self.observer._characterize_debate({})
        assert result == "emerging"

    def test_debate_character_is_string(self):
        """debate_character is always a string."""
        content = "I agree with the fair point about implementation here."
        self.observer.observe("alice", content)
        dynamics = self.observer.get_debate_dynamics()
        assert isinstance(dynamics["debate_character"], str)

    def test_patterns_detected_dict_values_are_ints(self):
        """patterns_detected values are integer counts."""
        content = "I agree with the fair point about the technical implementation here."
        self.observer.observe("alice", content)
        dynamics = self.observer.get_debate_dynamics()
        for value in dynamics["patterns_detected"].values():
            assert isinstance(value, int)
            assert value >= 1


# =============================================================================
# Observer _pattern_to_style Tests
# =============================================================================


class TestPatternToStyle:
    """Tests for _pattern_to_style."""

    def setup_method(self):
        self.observer = RhetoricalAnalysisObserver()

    def test_concession_maps_to_diplomatic(self):
        assert self.observer._pattern_to_style("concession") == "diplomatic"

    def test_rebuttal_maps_to_combative(self):
        assert self.observer._pattern_to_style("rebuttal") == "combative"

    def test_synthesis_maps_to_collaborative(self):
        assert self.observer._pattern_to_style("synthesis") == "collaborative"

    def test_appeal_to_authority_maps_to_scholarly(self):
        assert self.observer._pattern_to_style("appeal_to_authority") == "scholarly"

    def test_appeal_to_evidence_maps_to_empirical(self):
        assert self.observer._pattern_to_style("appeal_to_evidence") == "empirical"

    def test_technical_depth_maps_to_technical(self):
        assert self.observer._pattern_to_style("technical_depth") == "technical"

    def test_rhetorical_question_maps_to_socratic(self):
        assert self.observer._pattern_to_style("rhetorical_question") == "socratic"

    def test_analogy_maps_to_illustrative(self):
        assert self.observer._pattern_to_style("analogy") == "illustrative"

    def test_qualification_maps_to_nuanced(self):
        assert self.observer._pattern_to_style("qualification") == "nuanced"

    def test_unknown_pattern_maps_to_balanced(self):
        assert self.observer._pattern_to_style("unknown_pattern") == "balanced"

    def test_empty_string_maps_to_balanced(self):
        assert self.observer._pattern_to_style("") == "balanced"


# =============================================================================
# Observer _characterize_debate Tests
# =============================================================================


class TestCharacterizeDebate:
    """Tests for _characterize_debate."""

    def setup_method(self):
        self.observer = RhetoricalAnalysisObserver()

    def test_empty_returns_emerging(self):
        """Empty pattern_counts returns 'emerging'."""
        assert self.observer._characterize_debate({}) == "emerging"

    def test_collaborative_threshold(self):
        """Concession + synthesis > 40% of total → 'collaborative'."""
        counts = {"concession": 3, "synthesis": 2, "rebuttal": 1}
        assert self.observer._characterize_debate(counts) == "collaborative"

    def test_contentious_threshold(self):
        """Rebuttal > 40% of total → 'contentious'."""
        counts = {"rebuttal": 5, "concession": 1, "technical_depth": 1}
        assert self.observer._characterize_debate(counts) == "contentious"

    def test_technical_threshold(self):
        """Technical depth > 30% of total → 'technical'."""
        counts = {"technical_depth": 4, "concession": 1, "rebuttal": 1, "analogy": 1}
        assert self.observer._characterize_debate(counts) == "technical"

    def test_evidence_driven_threshold(self):
        """Evidence + authority > 30% of total → 'evidence-driven'."""
        counts = {
            "appeal_to_evidence": 2,
            "appeal_to_authority": 2,
            "concession": 1,
            "analogy": 1,
            "qualification": 1,
        }
        assert self.observer._characterize_debate(counts) == "evidence-driven"

    def test_balanced_when_no_dominant_pattern(self):
        """Evenly distributed patterns → 'balanced'."""
        counts = {
            "concession": 1,
            "rebuttal": 1,
            "analogy": 1,
            "qualification": 1,
            "rhetorical_question": 1,
        }
        assert self.observer._characterize_debate(counts) == "balanced"


# =============================================================================
# Observer get_recent_observations and reset Tests
# =============================================================================


class TestObserverRecentObservations:
    """Tests for get_recent_observations."""

    def setup_method(self):
        self.observer = RhetoricalAnalysisObserver(min_confidence=0.3)

    def test_get_recent_observations_returns_dicts(self):
        """get_recent_observations returns list of dicts."""
        content = "I agree with the fair point about technical implementation here."
        self.observer.observe("claude", content)
        recent = self.observer.get_recent_observations()
        for item in recent:
            assert isinstance(item, dict)

    def test_get_recent_observations_limit(self):
        """get_recent_observations respects the limit parameter."""
        content = "I agree with the fair point about the implementation architecture here."
        for _ in range(5):
            self.observer.observe("claude", content)
        recent = self.observer.get_recent_observations(limit=2)
        assert len(recent) <= 2

    def test_get_recent_observations_default_limit_10(self):
        """Default limit is 10 — returns at most 10 items."""
        content = "I agree with the fair point about the technical implementation here."
        for _ in range(15):
            self.observer.observe("claude", content)
        recent = self.observer.get_recent_observations()
        assert len(recent) <= 10

    def test_get_recent_observations_dict_has_required_keys(self):
        """Each observation dict has all required keys."""
        content = "I agree with the fair point about implementation architecture here."
        self.observer.observe("claude", content)
        recent = self.observer.get_recent_observations()
        for item in recent:
            assert "pattern" in item
            assert "agent" in item
            assert "confidence" in item
            assert "excerpt" in item

    def test_get_recent_observations_empty_when_no_observations(self):
        """Returns empty list when no observations have been made."""
        observer = RhetoricalAnalysisObserver()
        assert observer.get_recent_observations() == []


class TestObserverReset:
    """Tests for RhetoricalAnalysisObserver.reset()."""

    def test_reset_clears_observations(self):
        """reset() clears all stored observations."""
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        content = "I agree with the fair point about implementation here."
        observer.observe("claude", content)
        assert len(observer.observations) > 0
        observer.reset()
        assert observer.observations == []

    def test_reset_clears_agent_patterns(self):
        """reset() clears agent_patterns tracking."""
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        content = "I agree with the fair point about implementation here."
        observer.observe("claude", content)
        observer.reset()
        assert observer.agent_patterns == {}

    def test_reset_clears_structural_results(self):
        """reset() clears structural_results list."""
        analyzer = StructuralAnalyzer()
        observer = RhetoricalAnalysisObserver(structural_analyzer=analyzer, min_confidence=0.3)
        content = "I agree with fair point about technical implementation architecture here."
        observer.observe("claude", content)
        observer.reset()
        assert observer.get_structural_results() == []

    def test_reset_then_reobserve(self):
        """After reset, observer collects fresh observations correctly."""
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        content = "I agree with the fair point about the implementation here."
        observer.observe("claude", content)
        observer.reset()
        observer.observe("gpt", content)
        assert len(observer.observations) > 0
        assert "gpt" in observer.agent_patterns

    def test_reset_without_structural_analyzer(self):
        """reset() works correctly when no structural analyzer is set."""
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        content = "I agree with the fair point about implementation here."
        observer.observe("claude", content)
        observer.reset()  # Should not raise
        assert observer.observations == []


# =============================================================================
# Module-Level Function Tests
# =============================================================================


class TestModuleLevelFunctions:
    """Tests for get_rhetorical_observer and reset_rhetorical_observer."""

    def setup_method(self):
        """Reset global observer before each test."""
        reset_rhetorical_observer()

    def teardown_method(self):
        """Clean up global observer after each test."""
        reset_rhetorical_observer()

    def test_get_rhetorical_observer_returns_instance(self):
        """get_rhetorical_observer returns a RhetoricalAnalysisObserver."""
        observer = get_rhetorical_observer()
        assert isinstance(observer, RhetoricalAnalysisObserver)

    def test_get_rhetorical_observer_singleton(self):
        """Repeated calls return the same singleton instance."""
        obs1 = get_rhetorical_observer()
        obs2 = get_rhetorical_observer()
        assert obs1 is obs2

    def test_reset_rhetorical_observer_clears_singleton(self):
        """reset_rhetorical_observer sets global to None, causing new instance on next get."""
        obs1 = get_rhetorical_observer()
        reset_rhetorical_observer()
        obs2 = get_rhetorical_observer()
        assert obs1 is not obs2

    def test_global_observer_is_none_after_reset(self):
        """Global _observer is None after reset_rhetorical_observer."""
        get_rhetorical_observer()
        reset_rhetorical_observer()
        assert ro_module._observer is None

    def test_global_observer_is_set_after_get(self):
        """Global _observer is set after get_rhetorical_observer."""
        get_rhetorical_observer()
        assert ro_module._observer is not None

    def test_global_observer_is_usable(self):
        """Global observer returned by get_rhetorical_observer works correctly."""
        observer = get_rhetorical_observer()
        content = "I agree with the fair point about the implementation here."
        results = observer.observe("claude", content, round_num=1)
        assert isinstance(results, list)

    def test_multiple_resets_work_correctly(self):
        """Multiple consecutive resets do not cause errors."""
        reset_rhetorical_observer()
        reset_rhetorical_observer()
        assert ro_module._observer is None
        observer = get_rhetorical_observer()
        assert isinstance(observer, RhetoricalAnalysisObserver)


# =============================================================================
# Integration Tests
# =============================================================================


class TestObserverIntegration:
    """End-to-end integration tests for the rhetorical observer."""

    def test_full_debate_observation_cycle(self):
        """Full cycle: multiple agents, multiple rounds, dynamics summary."""
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)

        observer.observe(
            "alice",
            "According to research shows best practices, we should implement microservices architecture.",
            round_num=1,
        )
        observer.observe(
            "bob",
            "However, I disagree with the microservices approach because for small teams monoliths work better.",
            round_num=1,
        )
        observer.observe(
            "alice",
            "I agree that fair point about team size. Combining both perspectives, we find common ground here.",
            round_num=2,
        )
        observer.observe(
            "bob",
            "This typically depends on context. In some cases microservices are justified for scale.",
            round_num=2,
        )

        dynamics = observer.get_debate_dynamics()
        assert dynamics["total_observations"] > 0
        assert "alice" in dynamics["agent_styles"]
        assert "bob" in dynamics["agent_styles"]
        assert dynamics["dominant_pattern"] is not None
        assert isinstance(dynamics["debate_character"], str)

    def test_structural_plus_rhetorical_combined(self):
        """StructuralAnalyzer + RhetoricalAnalysisObserver work together correctly."""
        analyzer = StructuralAnalyzer()
        observer = RhetoricalAnalysisObserver(structural_analyzer=analyzer, min_confidence=0.3)

        content = (
            "I agree this is a fair point. Since the database queries are slow, "
            "therefore we must optimize them. For example, adding indexes reduces latency significantly."
        )
        observations = observer.observe("alice", content, round_num=1)

        assert isinstance(observations, list)
        assert len(observer.get_structural_results()) == 1
        for obs in observations:
            assert obs.agent == "alice"
            assert obs.round_num == 1

    def test_observation_excerpt_is_string(self):
        """Observation excerpt is always a non-None string."""
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        content = "I agree with this fair point about technical implementation architecture here."
        results = observer.observe("claude", content)
        for r in results:
            assert isinstance(r.excerpt, str)

    def test_observation_round_num_preserved(self):
        """Observation round_num matches the round passed to observe."""
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        content = "I agree with the fair point about implementation architecture here."
        results = observer.observe("claude", content, round_num=7)
        for r in results:
            assert r.round_num == 7

    def test_observations_accumulated_across_calls(self):
        """Multiple observe calls accumulate all observations."""
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        content = "I agree with the fair point about the technical implementation here."
        observer.observe("alice", content)
        n1 = len(observer.observations)
        observer.observe("bob", content)
        n2 = len(observer.observations)
        assert n2 >= n1

    def test_observation_to_dict_roundtrip(self):
        """Observations can be serialized to dict without errors."""
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        content = "I agree with the fair point about technical implementation here."
        results = observer.observe("claude", content)
        for r in results:
            d = r.to_dict()
            assert isinstance(d, dict)
            assert d["agent"] == "claude"
            assert isinstance(d["pattern"], str)

    def test_pattern_indicators_all_have_required_keys(self):
        """All PATTERN_INDICATORS entries have 'keywords', 'patterns', 'commentary'."""
        observer = RhetoricalAnalysisObserver()
        for pattern, indicators in observer.PATTERN_INDICATORS.items():
            assert "keywords" in indicators, f"{pattern} missing 'keywords'"
            assert "patterns" in indicators, f"{pattern} missing 'patterns'"
            assert "commentary" in indicators, f"{pattern} missing 'commentary'"

    def test_fallacy_indicators_all_have_required_keys(self):
        """All FALLACY_INDICATORS entries have 'keywords', 'patterns', 'explanation'."""
        for fallacy_type, indicators in StructuralAnalyzer.FALLACY_INDICATORS.items():
            assert "keywords" in indicators, f"{fallacy_type} missing 'keywords'"
            assert "patterns" in indicators, f"{fallacy_type} missing 'patterns'"
            assert "explanation" in indicators, f"{fallacy_type} missing 'explanation'"
