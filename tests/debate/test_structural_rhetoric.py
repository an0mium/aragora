"""
Tests for Structural Rhetorical Analysis.

Tests cover:
- FallacyType enum values
- FallacyDetection, PremiseChain, ClaimRelationship dataclasses
- StructuralAnalysisResult dataclass
- StructuralAnalyzer fallacy detection (all 7 types)
- Premise chain identification
- Unsupported claim detection
- Contradiction detection
- Claim relationship tracking
- Integration with RhetoricalAnalysisObserver
- Integration with ArgumentCartographer
- Combined confidence calculation
"""

import pytest

from aragora.debate.rhetorical_observer import (
    ClaimRelationship,
    FallacyDetection,
    FallacyType,
    PremiseChain,
    RhetoricalAnalysisObserver,
    StructuralAnalysisResult,
    StructuralAnalyzer,
)
from aragora.visualization.mapper import (
    ArgumentCartographer,
    EdgeRelation,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def analyzer():
    """Create a fresh structural analyzer."""
    return StructuralAnalyzer()


@pytest.fixture
def observer_with_structural():
    """Create an observer with structural analysis enabled."""
    return RhetoricalAnalysisObserver(
        structural_analyzer=StructuralAnalyzer(),
        min_confidence=0.3,
    )


@pytest.fixture
def cartographer():
    """Create a fresh argument cartographer."""
    cart = ArgumentCartographer()
    cart.set_debate_context("test-debate", "Test topic")
    return cart


# ============================================================================
# FallacyType Enum Tests
# ============================================================================


class TestFallacyType:
    """Tests for FallacyType enum."""

    def test_all_fallacy_types_exist(self):
        """Should have all 7 expected fallacy types."""
        expected = [
            "ad_hominem",
            "straw_man",
            "circular_reasoning",
            "false_dilemma",
            "appeal_to_ignorance",
            "slippery_slope",
            "red_herring",
        ]
        actual = [f.value for f in FallacyType]
        assert sorted(actual) == sorted(expected)

    def test_fallacy_type_count(self):
        """Should have exactly 7 fallacy types."""
        assert len(FallacyType) == 7


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestFallacyDetection:
    """Tests for FallacyDetection dataclass."""

    def test_create_detection(self):
        """Should create a fallacy detection with all fields."""
        detection = FallacyDetection(
            fallacy_type=FallacyType.AD_HOMINEM,
            confidence=0.8,
            excerpt="you're just biased",
            explanation="Attacks the person",
            agent="claude",
            round_num=2,
        )
        assert detection.fallacy_type == FallacyType.AD_HOMINEM
        assert detection.confidence == 0.8
        assert detection.agent == "claude"
        assert detection.round_num == 2

    def test_detection_to_dict(self):
        """Should serialize to dictionary."""
        detection = FallacyDetection(
            fallacy_type=FallacyType.STRAW_MAN,
            confidence=0.7,
            excerpt="so you're saying",
            explanation="Misrepresents argument",
            agent="gemini",
            round_num=1,
        )
        data = detection.to_dict()
        assert data["fallacy_type"] == "straw_man"
        assert data["confidence"] == 0.7
        assert data["agent"] == "gemini"

    def test_detection_defaults(self):
        """Should have sensible defaults for agent and round_num."""
        detection = FallacyDetection(
            fallacy_type=FallacyType.RED_HERRING,
            confidence=0.5,
            excerpt="but what about",
            explanation="Diverts attention",
        )
        assert detection.agent == ""
        assert detection.round_num == 0


class TestPremiseChain:
    """Tests for PremiseChain dataclass."""

    def test_create_chain(self):
        """Should create a premise chain."""
        chain = PremiseChain(
            premises=["Premise A", "Premise B"],
            conclusion="Therefore, conclusion C",
            agent="claude",
            confidence=0.6,
        )
        assert len(chain.premises) == 2
        assert chain.conclusion == "Therefore, conclusion C"
        assert chain.has_gap is False

    def test_chain_with_gap(self):
        """Should track unsupported leaps."""
        chain = PremiseChain(
            premises=["Some assertion"],
            conclusion="Therefore, big conclusion",
            agent="claude",
            confidence=0.4,
            has_gap=True,
        )
        assert chain.has_gap is True

    def test_chain_to_dict(self):
        """Should serialize to dictionary."""
        chain = PremiseChain(
            premises=["P1", "P2"],
            conclusion="C",
            agent="agent",
            confidence=0.5,
            has_gap=True,
        )
        data = chain.to_dict()
        assert data["premises"] == ["P1", "P2"]
        assert data["has_gap"] is True


class TestClaimRelationship:
    """Tests for ClaimRelationship dataclass."""

    def test_create_relationship(self):
        """Should create a claim relationship."""
        rel = ClaimRelationship(
            source_claim="This supports the approach",
            target_claim="We should use caching",
            relation="supports",
            confidence=0.7,
            agent="claude",
        )
        assert rel.relation == "supports"
        assert rel.confidence == 0.7

    def test_relationship_to_dict(self):
        """Should serialize to dictionary."""
        rel = ClaimRelationship(
            source_claim="A",
            target_claim="B",
            relation="contradicts",
            confidence=0.6,
        )
        data = rel.to_dict()
        assert data["relation"] == "contradicts"
        assert data["confidence"] == 0.6


class TestStructuralAnalysisResult:
    """Tests for StructuralAnalysisResult dataclass."""

    def test_empty_result(self):
        """Should create empty result with defaults."""
        result = StructuralAnalysisResult()
        assert result.fallacies == []
        assert result.premise_chains == []
        assert result.unsupported_claims == []
        assert result.contradictions == []
        assert result.claim_relationships == []
        assert result.confidence == 0.0

    def test_result_to_dict(self):
        """Should serialize to dictionary with nested structures."""
        result = StructuralAnalysisResult(
            fallacies=[
                FallacyDetection(
                    fallacy_type=FallacyType.AD_HOMINEM,
                    confidence=0.8,
                    excerpt="test",
                    explanation="test",
                )
            ],
            confidence=0.8,
        )
        data = result.to_dict()
        assert len(data["fallacies"]) == 1
        assert data["fallacies"][0]["fallacy_type"] == "ad_hominem"
        assert data["confidence"] == 0.8


# ============================================================================
# StructuralAnalyzer Fallacy Detection Tests
# ============================================================================


class TestFallacyDetectionByType:
    """Tests for detecting each specific fallacy type."""

    def test_detect_ad_hominem(self, analyzer):
        """Should detect ad hominem attacks."""
        content = (
            "Your argument about caching is invalid. "
            "You're clearly just biased and your limited understanding "
            "of distributed systems makes your opinion worthless."
        )
        result = analyzer.analyze(content, agent="critic", round_num=1)
        fallacy_types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.AD_HOMINEM in fallacy_types

    def test_detect_straw_man(self, analyzer):
        """Should detect straw man arguments."""
        content = (
            "So you're saying that we should just abandon all security "
            "measures and let anyone access the database directly? "
            "That's clearly a terrible idea and shows poor judgment."
        )
        result = analyzer.analyze(content, agent="critic", round_num=1)
        fallacy_types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.STRAW_MAN in fallacy_types

    def test_detect_circular_reasoning(self, analyzer):
        """Should detect circular reasoning."""
        content = (
            "This approach is correct because it is the right way to do things. "
            "We know it's true because it is true and self-evident. "
            "It goes without saying that this is obviously true."
        )
        result = analyzer.analyze(content, agent="agent", round_num=1)
        fallacy_types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.CIRCULAR_REASONING in fallacy_types

    def test_detect_false_dilemma(self, analyzer):
        """Should detect false dilemma fallacies."""
        content = (
            "We must choose between using microservices or having a completely "
            "unmaintainable monolith. There are only two options here. "
            "Either we rewrite everything or the project fails."
        )
        result = analyzer.analyze(content, agent="agent", round_num=1)
        fallacy_types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.FALSE_DILEMMA in fallacy_types

    def test_detect_appeal_to_ignorance(self, analyzer):
        """Should detect appeal to ignorance fallacies."""
        content = (
            "No one has proven otherwise that this approach will fail. "
            "Since there is no evidence against this design pattern, "
            "it must be the correct choice for our architecture."
        )
        result = analyzer.analyze(content, agent="agent", round_num=1)
        fallacy_types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.APPEAL_TO_IGNORANCE in fallacy_types

    def test_detect_slippery_slope(self, analyzer):
        """Should detect slippery slope fallacies."""
        content = (
            "If we allow this one exception to the coding standard, "
            "next thing you know everyone will ignore all the rules. "
            "This opens the door to complete chaos in the codebase "
            "and will inevitably lead to total project failure."
        )
        result = analyzer.analyze(content, agent="agent", round_num=1)
        fallacy_types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.SLIPPERY_SLOPE in fallacy_types

    def test_detect_red_herring(self, analyzer):
        """Should detect red herring diversions."""
        content = (
            "Instead of discussing the database schema, but what about "
            "the fact that our competitor just released a new product? "
            "Let's not forget about the marketing budget either. "
            "Speaking of which, that's a different and unrelated topic entirely."
        )
        result = analyzer.analyze(content, agent="agent", round_num=1)
        fallacy_types = [f.fallacy_type for f in result.fallacies]
        assert FallacyType.RED_HERRING in fallacy_types


# ============================================================================
# Premise Chain Tests
# ============================================================================


class TestPremiseChainDetection:
    """Tests for premise chain extraction."""

    def test_extract_conclusion_chain(self, analyzer):
        """Should extract premise chains with conclusion markers."""
        content = (
            "The system handles 1000 requests per second. "
            "Our user base is growing at 50% per month. "
            "Therefore, we need to implement horizontal scaling soon."
        )
        result = analyzer.analyze(content, agent="claude", round_num=1)
        assert len(result.premise_chains) >= 1
        chain = result.premise_chains[0]
        assert len(chain.premises) >= 1
        assert "therefore" in chain.conclusion.lower() or "scaling" in chain.conclusion.lower()

    def test_extract_because_chain(self, analyzer):
        """Should extract premise chains with because markers."""
        content = (
            "We should use Redis for caching. "
            "Because it provides sub-millisecond latency and supports clustering. "
            "This makes it ideal for our use case."
        )
        result = analyzer.analyze(content, agent="agent", round_num=1)
        assert len(result.premise_chains) >= 1

    def test_gap_detection_in_chain(self, analyzer):
        """Should detect gaps in premise chains (unsupported leaps)."""
        content = (
            "The weather is nice today. "
            "Stocks went up yesterday. "
            "Thus, we should refactor the authentication module."
        )
        result = analyzer.analyze(content, agent="agent", round_num=1)
        # Should detect a chain with a gap (premises have no causal connectors)
        chains_with_gaps = [c for c in result.premise_chains if c.has_gap]
        assert len(chains_with_gaps) >= 1

    def test_no_chain_for_short_content(self, analyzer):
        """Should not extract chains from very short content."""
        content = "This is a short sentence about nothing."
        result = analyzer.analyze(content, agent="agent", round_num=1)
        assert len(result.premise_chains) == 0


# ============================================================================
# Unsupported Claims Tests
# ============================================================================


class TestUnsupportedClaims:
    """Tests for unsupported claim detection."""

    def test_detect_unsupported_strong_assertion(self, analyzer):
        """Should detect strong assertions without evidence."""
        content = (
            "This approach is definitely the best. "
            "It will certainly solve all our problems. "
            "There is no doubt about this whatsoever."
        )
        result = analyzer.analyze(content, agent="agent", round_num=1)
        assert len(result.unsupported_claims) >= 1

    def test_supported_claim_not_flagged(self, analyzer):
        """Should not flag claims that have supporting evidence."""
        content = (
            "This approach is definitely the best because our benchmarks "
            "show a 40% improvement in throughput. "
            "The data clearly demonstrates the advantage."
        )
        result = analyzer.analyze(content, agent="agent", round_num=1)
        # The "definitely" claim has evidence in same sentence
        # so it should not be flagged as unsupported
        unsupported_with_because = [
            c for c in result.unsupported_claims if "because" in c.lower()
        ]
        assert len(unsupported_with_because) == 0


# ============================================================================
# Contradiction Detection Tests
# ============================================================================


class TestContradictionDetection:
    """Tests for internal contradiction detection."""

    def test_detect_should_contradiction(self, analyzer):
        """Should detect should vs should not contradictions."""
        content = (
            "We should implement caching for performance. "
            "But at the same time we should not add any caching layer."
        )
        result = analyzer.analyze(content, agent="agent", round_num=1)
        assert len(result.contradictions) >= 1

    def test_detect_effective_ineffective(self, analyzer):
        """Should detect effective vs ineffective contradictions."""
        content = (
            "The rate limiter is effective at preventing abuse. "
            "However, the rate limiter is ineffective and needs replacement."
        )
        result = analyzer.analyze(content, agent="agent", round_num=1)
        assert len(result.contradictions) >= 1

    def test_no_contradiction_in_consistent_text(self, analyzer):
        """Should not flag contradictions in consistent content."""
        content = (
            "We should implement caching for performance. "
            "We should also add monitoring. "
            "Both measures will improve reliability."
        )
        result = analyzer.analyze(content, agent="agent", round_num=1)
        assert len(result.contradictions) == 0


# ============================================================================
# Claim Relationship Tests
# ============================================================================


class TestClaimRelationships:
    """Tests for cross-message claim relationship tracking."""

    def test_detect_support_relationship(self, analyzer):
        """Should detect support relationships across messages."""
        # First message establishes a claim
        analyzer.analyze(
            "We should use PostgreSQL for the database layer.",
            agent="claude",
            round_num=1,
        )

        # Second message supports it
        result = analyzer.analyze(
            "This approach confirms and validates the database layer "
            "decision to use PostgreSQL for reliability.",
            agent="gemini",
            round_num=2,
        )
        support_rels = [r for r in result.claim_relationships if r.relation == "supports"]
        assert len(support_rels) >= 1

    def test_detect_contradiction_relationship(self, analyzer):
        """Should detect contradiction relationships across messages."""
        # First message
        analyzer.analyze(
            "The microservice architecture is the right approach for scalability.",
            agent="claude",
            round_num=1,
        )

        # Second message contradicts
        result = analyzer.analyze(
            "This analysis contradicts the microservice architecture approach "
            "which is at odds with our scalability requirements.",
            agent="gemini",
            round_num=2,
        )
        contra_rels = [r for r in result.claim_relationships if r.relation == "contradicts"]
        assert len(contra_rels) >= 1

    def test_detect_refinement_relationship(self, analyzer):
        """Should detect refinement relationships across messages."""
        # First message
        analyzer.analyze(
            "We need better caching for the API responses.",
            agent="claude",
            round_num=1,
        )

        # Second message refines
        result = analyzer.analyze(
            "This extends and elaborates on the caching for API responses "
            "by adding a TTL-based invalidation strategy.",
            agent="gemini",
            round_num=2,
        )
        refine_rels = [r for r in result.claim_relationships if r.relation == "refines"]
        assert len(refine_rels) >= 1


# ============================================================================
# StructuralAnalyzer State Tests
# ============================================================================


class TestStructuralAnalyzerState:
    """Tests for analyzer state management."""

    def test_get_all_fallacies(self, analyzer):
        """Should accumulate fallacies across multiple analyses."""
        analyzer.analyze(
            "You're clearly just biased and your limited understanding is showing.",
            agent="agent1",
            round_num=1,
        )
        analyzer.analyze(
            "So you're saying we should just give up entirely? That's absurd.",
            agent="agent2",
            round_num=2,
        )
        all_fallacies = analyzer.get_all_fallacies()
        assert len(all_fallacies) >= 2

    def test_get_fallacy_summary(self, analyzer):
        """Should provide fallacy type counts."""
        analyzer.analyze(
            "You're clearly just biased and your limited understanding is evident.",
            agent="agent1",
            round_num=1,
        )
        summary = analyzer.get_fallacy_summary()
        assert isinstance(summary, dict)
        assert "ad_hominem" in summary
        assert summary["ad_hominem"] >= 1

    def test_reset_clears_state(self, analyzer):
        """Should clear all state on reset."""
        analyzer.analyze(
            "You're clearly just biased and your limited understanding is clear.",
            agent="agent1",
            round_num=1,
        )
        assert len(analyzer.get_all_fallacies()) >= 1

        analyzer.reset()
        assert len(analyzer.get_all_fallacies()) == 0
        assert len(analyzer._claim_history) == 0

    def test_short_content_returns_empty(self, analyzer):
        """Should return empty result for very short content."""
        result = analyzer.analyze("Too short", agent="agent")
        assert result.fallacies == []
        assert result.premise_chains == []
        assert result.confidence == 0.0

    def test_empty_content_returns_empty(self, analyzer):
        """Should return empty result for empty content."""
        result = analyzer.analyze("", agent="agent")
        assert result.fallacies == []
        assert result.confidence == 0.0


# ============================================================================
# Observer Integration Tests
# ============================================================================


class TestObserverWithStructuralAnalysis:
    """Tests for RhetoricalAnalysisObserver with structural analyzer."""

    def test_observer_with_structural_analyzer(self, observer_with_structural):
        """Should run structural analysis alongside keyword detection."""
        content = (
            "However, I disagree with the approach. "
            "You're clearly just biased and your limited understanding "
            "of the problem space makes your argument invalid."
        )
        observations = observer_with_structural.observe(
            agent="critic", content=content, round_num=1
        )
        # Should have keyword-based observations
        assert len(observations) >= 1

        # Should have structural results
        structural = observer_with_structural.get_structural_results()
        assert len(structural) >= 1
        assert any(
            f.fallacy_type == FallacyType.AD_HOMINEM
            for r in structural
            for f in r.fallacies
        )

    def test_combined_confidence_max(self, observer_with_structural):
        """Combined confidence should use max(keyword, structural)."""
        # Content with both keyword patterns and structural signals
        content = (
            "I must acknowledge that this is a fair point. "
            "However, you're clearly just biased in your assessment "
            "and your limited understanding undermines your argument."
        )
        observations = observer_with_structural.observe(
            agent="agent", content=content, round_num=1
        )
        # The structural analyzer may boost confidence
        for obs in observations:
            assert obs.confidence >= 0.3  # At least min_confidence

    def test_observer_without_structural(self):
        """Observer without structural analyzer should work normally."""
        observer = RhetoricalAnalysisObserver(min_confidence=0.3)
        content = (
            "I acknowledge this is a fair point and a valid argument. "
            "However, I would disagree with the conclusion."
        )
        observations = observer.observe(agent="agent", content=content, round_num=1)
        assert len(observations) >= 1
        assert observer.get_structural_results() == []

    def test_observer_reset_clears_structural(self, observer_with_structural):
        """Reset should clear structural results too."""
        content = (
            "You're clearly just biased and your limited understanding "
            "of the problem is evident in your arguments."
        )
        observer_with_structural.observe(agent="agent", content=content, round_num=1)
        assert len(observer_with_structural.get_structural_results()) >= 1

        observer_with_structural.reset()
        assert len(observer_with_structural.get_structural_results()) == 0


# ============================================================================
# ArgumentCartographer Integration Tests
# ============================================================================


class TestCartographerStructuralAnnotation:
    """Tests for ArgumentCartographer structural annotation integration."""

    def test_add_structural_annotation(self, cartographer):
        """Should add structural annotation to node metadata."""
        node_id = cartographer.update_from_message(
            agent="claude",
            content="I propose we implement caching",
            role="proposer",
            round_num=1,
        )

        annotation = {
            "fallacies": [
                {"fallacy_type": "ad_hominem", "confidence": 0.8}
            ],
        }
        result = cartographer.add_structural_annotation(node_id, annotation)

        assert result is True
        node = cartographer.nodes[node_id]
        assert "structural" in node.metadata
        assert len(node.metadata["structural"]) == 1

    def test_annotation_nonexistent_node(self, cartographer):
        """Should return False for nonexistent node."""
        result = cartographer.add_structural_annotation(
            "nonexistent", {"fallacies": []}
        )
        assert result is False

    def test_annotation_creates_edges_for_relationships(self, cartographer):
        """Should create edges when annotation includes claim relationships."""
        # Add two nodes
        node1_id = cartographer.update_from_message(
            agent="claude",
            content="We should use PostgreSQL for the database layer in production",
            role="proposer",
            round_num=1,
        )
        node2_id = cartographer.update_from_message(
            agent="gemini",
            content="This contradicts the database layer decision about PostgreSQL",
            role="critic",
            round_num=1,
        )

        # Add annotation with a relationship pointing to node1's content
        annotation = {
            "claim_relationships": [
                {
                    "source_claim": "This contradicts the database layer decision",
                    "target_claim": "We should use PostgreSQL for the database layer in production",
                    "relation": "contradicts",
                    "confidence": 0.7,
                }
            ],
        }
        initial_edge_count = len(cartographer.edges)
        cartographer.add_structural_annotation(node2_id, annotation)

        # Should have added at least one edge from structural analysis
        structural_edges = [
            e
            for e in cartographer.edges[initial_edge_count:]
            if e.metadata.get("source") == "structural_analysis"
        ]
        assert len(structural_edges) >= 1
        assert structural_edges[0].relation == EdgeRelation.REFUTES

    def test_multiple_annotations_on_same_node(self, cartographer):
        """Should accumulate multiple annotations on the same node."""
        node_id = cartographer.update_from_message(
            agent="agent",
            content="Test content for multiple annotations",
            role="agent",
            round_num=1,
        )

        cartographer.add_structural_annotation(node_id, {"type": "fallacy"})
        cartographer.add_structural_annotation(node_id, {"type": "chain"})

        node = cartographer.nodes[node_id]
        assert len(node.metadata["structural"]) == 2

    def test_end_to_end_structural_to_cartographer(self, cartographer):
        """End-to-end: analyzer produces result, feeds to cartographer."""
        analyzer = StructuralAnalyzer()

        node_id = cartographer.update_from_message(
            agent="claude",
            content=(
                "You're clearly just biased and your limited understanding "
                "of the problem makes your opinion invalid. "
                "We should definitely use the monolith approach."
            ),
            role="agent",
            round_num=1,
        )

        result = analyzer.analyze(
            "You're clearly just biased and your limited understanding "
            "of the problem makes your opinion invalid.",
            agent="claude",
            round_num=1,
        )

        annotation = result.to_dict()
        success = cartographer.add_structural_annotation(node_id, annotation)
        assert success is True
        assert "structural" in cartographer.nodes[node_id].metadata
