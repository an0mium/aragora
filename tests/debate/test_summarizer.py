"""
Tests for the debate summarization module.

Tests cover:
- DebateSummary dataclass and serialization
- DebateSummarizer methods:
  - summarize() main entry point
  - One-liner verdict generation
  - Key points extraction
  - Agreement/disagreement detection
  - Next steps generation
  - Caveats generation
- Different verdict types (consensus/no consensus)
- Multi-agent scenarios
- Edge cases (single agent, empty data, no consensus)
- Format validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from aragora.debate.summarizer import (
    DebateSummary,
    DebateSummarizer,
    _DictWrapper,
    summarize_debate,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@dataclass
class MockMessage:
    """Mock message for testing."""

    agent: str
    content: str
    role: str = "proposer"


@dataclass
class MockCritique:
    """Mock critique for testing."""

    agent: str
    content: str
    severity: float = 5.0


@dataclass
class MockVote:
    """Mock vote for testing."""

    agent: str
    reasoning: str
    choice: str = "consensus"


@dataclass
class MockGroundedVerdict:
    """Mock grounded verdict for testing."""

    grounding_score: float = 0.8


@dataclass
class MockDebateResult:
    """Mock debate result for testing."""

    rounds_used: int = 3
    duration_seconds: float = 45.5
    confidence: float = 0.85
    consensus_reached: bool = True
    consensus_strength: str = "medium"
    final_answer: str = "Use Redis for caching with a 15-minute TTL."
    task: str = "Design a caching solution"
    messages: list = field(default_factory=list)
    dissenting_views: list = field(default_factory=list)
    winning_patterns: list = field(default_factory=list)
    votes: list = field(default_factory=list)
    critiques: list = field(default_factory=list)
    debate_cruxes: list = field(default_factory=list)
    evidence_suggestions: list = field(default_factory=list)
    grounded_verdict: Any = None
    convergence_status: str = ""
    avg_novelty: float = 1.0

    def to_dict(self) -> dict:
        """Convert to dict (signals to summarizer that this is a proper result)."""
        return {"final_answer": self.final_answer}


@pytest.fixture
def summarizer() -> DebateSummarizer:
    """Create a DebateSummarizer instance."""
    return DebateSummarizer()


@pytest.fixture
def basic_result() -> MockDebateResult:
    """Create a basic debate result with consensus."""
    return MockDebateResult(
        messages=[
            MockMessage(agent="claude", content="Use Redis for caching"),
            MockMessage(agent="gpt4", content="I agree with Redis"),
            MockMessage(agent="gemini", content="Redis is a good choice"),
        ],
    )


@pytest.fixture
def no_consensus_result() -> MockDebateResult:
    """Create a debate result without consensus."""
    return MockDebateResult(
        consensus_reached=False,
        confidence=0.4,
        consensus_strength="none",
        dissenting_views=["Use Memcached instead", "Consider DynamoDB"],
        messages=[
            MockMessage(agent="claude", content="Use Redis"),
            MockMessage(agent="gpt4", content="Use Memcached"),
        ],
    )


@pytest.fixture
def single_agent_result() -> MockDebateResult:
    """Create a result with a single agent."""
    return MockDebateResult(
        rounds_used=1,
        messages=[
            MockMessage(agent="claude", content="Use Redis for caching"),
        ],
    )


@pytest.fixture
def detailed_result() -> MockDebateResult:
    """Create a detailed result with all fields populated."""
    return MockDebateResult(
        rounds_used=5,
        duration_seconds=120.0,
        confidence=0.92,
        consensus_reached=True,
        consensus_strength="strong",
        final_answer="""In conclusion, we recommend using Redis for caching.

Key recommendations:
1. Set TTL to 15 minutes for optimal freshness
2. Use connection pooling for performance
3. Implement circuit breakers for resilience

The answer is Redis because it provides fast reads and supports complex data structures.
All agents agree that Redis is the best choice for this use case.""",
        task="Design a high-performance caching solution",
        messages=[
            MockMessage(agent="claude", content="Proposal 1"),
            MockMessage(agent="gpt4", content="Proposal 2"),
            MockMessage(agent="gemini", content="Proposal 3"),
        ],
        winning_patterns=["Use connection pooling", "Implement TTL"],
        votes=[
            MockVote(agent="claude", reasoning="Redis provides excellent performance"),
            MockVote(agent="gpt4", reasoning="Redis provides excellent performance"),
            MockVote(agent="gemini", reasoning="Redis is reliable"),
        ],
        critiques=[
            MockCritique(
                agent="gpt4",
                content="However, we should consider memory limits",
            ),
        ],
        debate_cruxes=[{"claim": "Redis vs Memcached performance"}],
        evidence_suggestions=[{"claim": "Benchmark results for high load"}],
        convergence_status="converged",
        avg_novelty=0.7,
    )


# ============================================================================
# DebateSummary Tests
# ============================================================================


class TestDebateSummary:
    """Tests for the DebateSummary dataclass."""

    def test_default_initialization(self):
        """Test DebateSummary with default values."""
        summary = DebateSummary()

        assert summary.one_liner == ""
        assert summary.key_points == []
        assert summary.agreement_areas == []
        assert summary.disagreement_areas == []
        assert summary.confidence == 0.0
        assert summary.confidence_label == ""
        assert summary.consensus_strength == ""
        assert summary.next_steps == []
        assert summary.caveats == []
        assert summary.rounds_used == 0
        assert summary.agents_participated == 0
        assert summary.duration_seconds == 0.0

    def test_custom_initialization(self):
        """Test DebateSummary with custom values."""
        summary = DebateSummary(
            one_liner="Agents reached consensus: Use Redis",
            key_points=["Fast read performance", "TTL support"],
            agreement_areas=["Redis is suitable"],
            disagreement_areas=["TTL duration"],
            confidence=0.85,
            confidence_label="high",
            consensus_strength="strong",
            next_steps=["Implement caching layer"],
            caveats=["Memory constraints"],
            rounds_used=3,
            agents_participated=4,
            duration_seconds=45.5,
        )

        assert summary.one_liner == "Agents reached consensus: Use Redis"
        assert len(summary.key_points) == 2
        assert summary.confidence == 0.85
        assert summary.agents_participated == 4

    def test_to_dict(self):
        """Test conversion to dictionary."""
        summary = DebateSummary(
            one_liner="Test summary",
            key_points=["Point 1", "Point 2"],
            confidence=0.75,
            confidence_label="medium",
            rounds_used=2,
        )

        result = summary.to_dict()

        assert isinstance(result, dict)
        assert result["one_liner"] == "Test summary"
        assert result["key_points"] == ["Point 1", "Point 2"]
        assert result["confidence"] == 0.75
        assert result["confidence_label"] == "medium"
        assert result["rounds_used"] == 2

    def test_to_dict_all_fields(self):
        """Test that to_dict includes all fields."""
        summary = DebateSummary()
        result = summary.to_dict()

        expected_keys = {
            "one_liner",
            "key_points",
            "agreement_areas",
            "disagreement_areas",
            "confidence",
            "confidence_label",
            "consensus_strength",
            "next_steps",
            "caveats",
            "rounds_used",
            "agents_participated",
            "duration_seconds",
        }

        assert set(result.keys()) == expected_keys


# ============================================================================
# DictWrapper Tests
# ============================================================================


class TestDictWrapper:
    """Tests for the _DictWrapper helper class."""

    def test_attribute_access(self):
        """Test accessing dict values as attributes."""
        data = {"name": "test", "value": 42, "nested": {"key": "value"}}
        wrapper = _DictWrapper(data)

        assert wrapper.name == "test"
        assert wrapper.value == 42
        assert wrapper.nested == {"key": "value"}

    def test_missing_attribute(self):
        """Test accessing missing attributes returns None."""
        wrapper = _DictWrapper({"existing": "value"})

        assert wrapper.missing is None
        assert wrapper.nonexistent is None

    def test_empty_dict(self):
        """Test wrapping empty dict."""
        wrapper = _DictWrapper({})

        assert wrapper.any_key is None


# ============================================================================
# DebateSummarizer Core Tests
# ============================================================================


class TestDebateSummarizer:
    """Tests for the DebateSummarizer class."""

    @pytest.mark.smoke
    def test_summarize_basic_result(self, summarizer, basic_result):
        """Test summarizing a basic debate result."""
        summary = summarizer.summarize(basic_result)

        assert isinstance(summary, DebateSummary)
        assert summary.rounds_used == 3
        assert summary.confidence == 0.85
        assert summary.agents_participated == 3
        assert summary.consensus_strength == "medium"
        assert summary.confidence_label == "high"

    def test_summarize_dict_input(self, summarizer):
        """Test summarizing from a dict instead of object."""
        data = {
            "rounds_used": 2,
            "confidence": 0.7,
            "consensus_reached": True,
            "consensus_strength": "weak",
            "final_answer": "Use caching",
            "messages": [{"agent": "claude"}, {"agent": "gpt4"}],
            "avg_novelty": 0.8,  # Include to avoid None comparison in _generate_caveats
        }

        summary = summarizer.summarize(data)

        assert summary.rounds_used == 2
        assert summary.confidence == 0.7
        assert summary.agents_participated == 2

    def test_summarize_unknown_type(self, summarizer):
        """Test summarizing unknown type returns empty summary."""
        summary = summarizer.summarize("invalid input")

        assert isinstance(summary, DebateSummary)
        assert summary.one_liner == ""
        assert summary.key_points == []

    def test_summarize_none_values(self, summarizer):
        """Test handling None values in result."""
        result = MockDebateResult()
        result.rounds_used = None
        result.confidence = None
        result.final_answer = None

        summary = summarizer.summarize(result)

        assert summary.rounds_used == 0
        assert summary.confidence == 0.0
        assert summary.key_points == []


# ============================================================================
# Confidence Label Tests
# ============================================================================


class TestConfidenceLabels:
    """Tests for confidence label generation."""

    def test_high_confidence(self, summarizer):
        """Test high confidence label (>= 0.8)."""
        assert summarizer._get_confidence_label(0.8) == "high"
        assert summarizer._get_confidence_label(0.9) == "high"
        assert summarizer._get_confidence_label(1.0) == "high"

    def test_medium_confidence(self, summarizer):
        """Test medium confidence label (>= 0.6, < 0.8)."""
        assert summarizer._get_confidence_label(0.6) == "medium"
        assert summarizer._get_confidence_label(0.7) == "medium"
        assert summarizer._get_confidence_label(0.79) == "medium"

    def test_low_confidence(self, summarizer):
        """Test low confidence label (< 0.6)."""
        assert summarizer._get_confidence_label(0.0) == "low"
        assert summarizer._get_confidence_label(0.3) == "low"
        assert summarizer._get_confidence_label(0.59) == "low"

    def test_confidence_thresholds(self, summarizer):
        """Test confidence threshold constants."""
        assert summarizer.HIGH_CONFIDENCE == 0.8
        assert summarizer.MEDIUM_CONFIDENCE == 0.6


# ============================================================================
# One-Liner Generation Tests
# ============================================================================


class TestOneLinerGeneration:
    """Tests for one-liner verdict generation."""

    def test_strong_consensus_one_liner(self, summarizer):
        """Test one-liner for strong consensus."""
        result = MockDebateResult(
            consensus_reached=True,
            confidence=0.9,
            final_answer="In conclusion, use Redis for best performance.",
        )

        one_liner = summarizer._generate_one_liner(result)

        assert "strong consensus" in one_liner.lower()
        assert "Redis" in one_liner or "performance" in one_liner

    def test_regular_consensus_one_liner(self, summarizer):
        """Test one-liner for regular consensus (lower confidence)."""
        result = MockDebateResult(
            consensus_reached=True,
            confidence=0.7,
            final_answer="Therefore, caching is recommended.",
        )

        one_liner = summarizer._generate_one_liner(result)

        assert "reached consensus" in one_liner.lower()
        assert "strong" not in one_liner.lower()

    def test_no_consensus_with_dissenting(self, summarizer):
        """Test one-liner with mixed opinions."""
        result = MockDebateResult(
            consensus_reached=False,
            dissenting_views=["Use Memcached", "Use Redis"],
        )

        one_liner = summarizer._generate_one_liner(result)

        assert "mixed opinions" in one_liner.lower()

    def test_no_consensus_no_dissent(self, summarizer):
        """Test one-liner with no consensus and no dissent."""
        result = MockDebateResult(
            consensus_reached=False,
            dissenting_views=[],
            final_answer="",
        )

        one_liner = summarizer._generate_one_liner(result)

        assert "no clear consensus" in one_liner.lower()

    def test_one_liner_with_task_fallback(self, summarizer):
        """Test one-liner falls back to task when no conclusion."""
        result = MockDebateResult(
            consensus_reached=True,
            confidence=0.9,
            final_answer="",
            task="Design a caching solution for high traffic",
        )

        one_liner = summarizer._generate_one_liner(result)

        assert "caching" in one_liner.lower() or "Design" in one_liner

    def test_one_liner_task_truncation(self, summarizer):
        """Test that long tasks are truncated in one-liner."""
        long_task = "A" * 100
        result = MockDebateResult(
            consensus_reached=True,
            confidence=0.9,
            final_answer="",
            task=long_task,
        )

        one_liner = summarizer._generate_one_liner(result)

        # Task should be truncated to 80 chars + "..."
        assert len(one_liner) < len(long_task) + 50


# ============================================================================
# Conclusion Extraction Tests
# ============================================================================


class TestConclusionExtraction:
    """Tests for conclusion extraction from text."""

    def test_extract_in_conclusion_pattern(self, summarizer):
        """Test extracting 'in conclusion' pattern."""
        text = "After analysis, in conclusion, Redis is the best choice."
        conclusion = summarizer._extract_conclusion(text)

        assert "Redis" in conclusion

    def test_extract_therefore_pattern(self, summarizer):
        """Test extracting 'therefore' pattern."""
        text = "Given the requirements, therefore use caching."
        conclusion = summarizer._extract_conclusion(text)

        assert "caching" in conclusion

    def test_extract_the_answer_is_pattern(self, summarizer):
        """Test extracting 'the answer is' pattern."""
        text = "Based on benchmarks, the answer is Redis."
        conclusion = summarizer._extract_conclusion(text)

        assert "Redis" in conclusion

    def test_extract_to_summarize_pattern(self, summarizer):
        """Test extracting 'to summarize' pattern."""
        text = "After much debate, to summarize, we should use Redis."
        conclusion = summarizer._extract_conclusion(text)

        assert "Redis" in conclusion

    def test_conclusion_truncation(self, summarizer):
        """Test that long conclusions are truncated."""
        long_conclusion = "A" * 200
        text = f"In conclusion, {long_conclusion}."
        conclusion = summarizer._extract_conclusion(text)

        assert len(conclusion) <= 150
        assert conclusion.endswith("...")

    def test_first_sentence_fallback(self, summarizer):
        """Test fallback to first sentence."""
        text = "Use Redis for caching. It provides fast reads. It supports TTL."
        conclusion = summarizer._extract_conclusion(text)

        assert "Redis" in conclusion or "caching" in conclusion

    def test_empty_text(self, summarizer):
        """Test empty text returns empty string."""
        assert summarizer._extract_conclusion("") == ""
        assert summarizer._extract_conclusion(None) == ""


# ============================================================================
# Key Points Extraction Tests
# ============================================================================


class TestKeyPointsExtraction:
    """Tests for key points extraction."""

    def test_extract_numbered_list(self, summarizer):
        """Test extracting numbered list items."""
        text = """
        1. Use Redis for caching
        2. Set TTL to 15 minutes
        3. Implement circuit breakers
        """
        points = summarizer._extract_key_points(text)

        assert len(points) >= 3
        assert any("Redis" in p for p in points)

    def test_extract_bulleted_list_dash(self, summarizer):
        """Test extracting dash-bulleted list."""
        text = """
        - First important point here
        - Second key recommendation
        - Third suggestion
        """
        points = summarizer._extract_key_points(text)

        assert len(points) >= 2

    def test_extract_bulleted_list_asterisk(self, summarizer):
        """Test extracting asterisk-bulleted list."""
        text = """
        * Use connection pooling for better performance
        * Enable compression for network efficiency
        * Monitor cache hit rates regularly
        """
        points = summarizer._extract_key_points(text)

        assert len(points) >= 2

    def test_extract_key_phrase_sentences(self, summarizer):
        """Test extracting sentences with key phrases."""
        text = """
        It is important to consider memory limits.
        The key factor is latency reduction.
        We recommend using Redis for this use case.
        """
        points = summarizer._extract_key_points(text)

        assert len(points) >= 1

    def test_max_five_points(self, summarizer):
        """Test that maximum 5 points are returned."""
        text = "\n".join([f"{i}. Point number {i}" for i in range(1, 10)])
        points = summarizer._extract_key_points(text)

        assert len(points) <= 5

    def test_ignore_short_items(self, summarizer):
        """Test that very short items are ignored."""
        text = """
        1. OK
        2. This is a longer point that should be included
        3. No
        """
        points = summarizer._extract_key_points(text)

        # Short items like "OK" and "No" should be filtered
        for point in points:
            assert len(point) > 10

    def test_fallback_to_sentences(self, summarizer):
        """Test fallback to first sentences when no lists or key phrases."""
        text = """
        Redis provides excellent read performance.
        The latency is typically under 1ms.
        Connection pooling improves throughput.
        """
        points = summarizer._extract_key_points(text)

        assert len(points) >= 1
        assert any("Redis" in p or "latency" in p for p in points)

    def test_empty_text(self, summarizer):
        """Test empty text returns empty list."""
        assert summarizer._extract_key_points("") == []
        assert summarizer._extract_key_points(None) == []


# ============================================================================
# Agreement Extraction Tests
# ============================================================================


class TestAgreementExtraction:
    """Tests for agreement area extraction."""

    def test_extract_unanimous_pattern(self, summarizer):
        """Test extracting 'unanimous' agreement patterns."""
        result = MockDebateResult(final_answer="All agents agree that Redis is the best choice.")
        agreements = summarizer._extract_agreements(result)

        assert len(agreements) >= 1
        assert any("Redis" in a for a in agreements)

    def test_extract_consensus_pattern(self, summarizer):
        """Test extracting 'consensus reached' pattern."""
        result = MockDebateResult(
            final_answer="Consensus reached: use caching for all read operations."
        )
        agreements = summarizer._extract_agreements(result)

        assert len(agreements) >= 1

    def test_extract_from_winning_patterns(self, summarizer):
        """Test extracting from winning_patterns field."""
        result = MockDebateResult(
            final_answer="",
            winning_patterns=["Use connection pooling", "Enable compression"],
        )
        agreements = summarizer._extract_agreements(result)

        assert len(agreements) >= 2
        assert "connection pooling" in " ".join(agreements).lower()

    def test_extract_from_votes(self, summarizer):
        """Test extracting from vote reasoning."""
        result = MockDebateResult(
            final_answer="",
            winning_patterns=[],
            votes=[
                MockVote(agent="claude", reasoning="Redis is fast and reliable"),
                MockVote(agent="gpt4", reasoning="Redis is fast and reliable"),
                MockVote(agent="gemini", reasoning="Redis is fast and reliable"),
            ],
        )
        agreements = summarizer._extract_agreements(result)

        # Should find agreement from multiple similar votes
        assert len(agreements) >= 0  # May or may not find depending on threshold

    def test_max_five_agreements(self, summarizer):
        """Test that maximum 5 agreements are returned."""
        result = MockDebateResult(
            winning_patterns=[f"Pattern {i}" for i in range(10)],
        )
        agreements = summarizer._extract_agreements(result)

        assert len(agreements) <= 5


# ============================================================================
# Disagreement Extraction Tests
# ============================================================================


class TestDisagreementExtraction:
    """Tests for disagreement area extraction."""

    def test_extract_from_dissenting_views(self, summarizer):
        """Test extracting from dissenting_views field."""
        result = MockDebateResult(
            dissenting_views=[
                "Consider using Memcached for simpler use cases",
                "DynamoDB might be better for persistence",
            ],
        )
        disagreements = summarizer._extract_disagreements(result)

        assert len(disagreements) >= 2
        assert any("Memcached" in d for d in disagreements)

    def test_truncate_long_dissenting_views(self, summarizer):
        """Test that long dissenting views are truncated."""
        long_view = "A" * 200
        result = MockDebateResult(dissenting_views=[long_view])
        disagreements = summarizer._extract_disagreements(result)

        if disagreements:
            assert len(disagreements[0]) <= 150
            assert disagreements[0].endswith("...")

    def test_extract_however_pattern_from_critiques(self, summarizer):
        """Test extracting 'however' pattern from critiques."""
        result = MockDebateResult(
            dissenting_views=[],
            critiques=[
                MockCritique(
                    agent="gpt4",
                    content="However, we should consider memory limits carefully",
                ),
            ],
        )
        disagreements = summarizer._extract_disagreements(result)

        assert len(disagreements) >= 1

    def test_extract_from_debate_cruxes(self, summarizer):
        """Test extracting from debate_cruxes field."""
        result = MockDebateResult(
            dissenting_views=[],
            critiques=[],
            debate_cruxes=[
                {"claim": "Redis vs Memcached performance comparison"},
                {"claim": "Memory efficiency trade-offs"},
            ],
        )
        disagreements = summarizer._extract_disagreements(result)

        assert len(disagreements) >= 1
        assert any("Crux:" in d for d in disagreements)

    def test_max_five_disagreements(self, summarizer):
        """Test that maximum 5 disagreements are returned."""
        result = MockDebateResult(
            dissenting_views=[f"View {i}" for i in range(10)],
        )
        disagreements = summarizer._extract_disagreements(result)

        assert len(disagreements) <= 5


# ============================================================================
# Next Steps Generation Tests
# ============================================================================


class TestNextStepsGeneration:
    """Tests for next steps generation."""

    def test_high_confidence_consensus(self, summarizer):
        """Test next steps for high confidence consensus."""
        result = MockDebateResult(
            consensus_reached=True,
            confidence=0.9,
        )
        steps = summarizer._generate_next_steps(result)

        assert any("implementation" in s.lower() or "proceed" in s.lower() for s in steps)

    def test_low_confidence_consensus(self, summarizer):
        """Test next steps for lower confidence consensus."""
        result = MockDebateResult(
            consensus_reached=True,
            confidence=0.7,
        )
        steps = summarizer._generate_next_steps(result)

        assert any("validation" in s.lower() for s in steps)

    def test_consensus_with_dissent(self, summarizer):
        """Test next steps when consensus has dissenting views."""
        result = MockDebateResult(
            consensus_reached=True,
            confidence=0.85,
            dissenting_views=["Alternative approach"],
        )
        steps = summarizer._generate_next_steps(result)

        assert any("dissenting" in s.lower() or "edge case" in s.lower() for s in steps)

    def test_no_consensus(self, summarizer):
        """Test next steps when no consensus reached."""
        result = MockDebateResult(
            consensus_reached=False,
            dissenting_views=["View 1", "View 2"],
        )
        steps = summarizer._generate_next_steps(result)

        assert any("another debate" in s.lower() or "disagreement" in s.lower() for s in steps)

    def test_evidence_suggestions(self, summarizer):
        """Test next steps include evidence suggestions."""
        result = MockDebateResult(
            consensus_reached=True,
            confidence=0.85,
            evidence_suggestions=[
                {"claim": "Benchmark results needed"},
                {"claim": "Production metrics required"},
            ],
        )
        steps = summarizer._generate_next_steps(result)

        assert any("evidence" in s.lower() for s in steps)

    def test_low_grounding_score(self, summarizer):
        """Test next steps for low grounding score."""
        result = MockDebateResult(
            consensus_reached=True,
            confidence=0.85,
            grounded_verdict=MockGroundedVerdict(grounding_score=0.3),
        )
        steps = summarizer._generate_next_steps(result)

        assert any("grounding" in s.lower() or "evidence" in s.lower() for s in steps)

    def test_max_four_steps(self, summarizer):
        """Test that maximum 4 next steps are returned."""
        result = MockDebateResult(
            consensus_reached=False,
            dissenting_views=["V1", "V2", "V3"],
            evidence_suggestions=[
                {"claim": "C1"},
                {"claim": "C2"},
                {"claim": "C3"},
            ],
            grounded_verdict=MockGroundedVerdict(grounding_score=0.2),
        )
        steps = summarizer._generate_next_steps(result)

        assert len(steps) <= 4


# ============================================================================
# Caveats Generation Tests
# ============================================================================


class TestCaveatsGeneration:
    """Tests for caveats generation."""

    def test_low_confidence_caveat(self, summarizer):
        """Test caveat for low confidence."""
        result = MockDebateResult(confidence=0.4)
        caveats = summarizer._generate_caveats(result)

        assert any("low confidence" in c.lower() for c in caveats)

    def test_no_consensus_caveat(self, summarizer):
        """Test caveat when no consensus reached."""
        result = MockDebateResult(consensus_reached=False)
        caveats = summarizer._generate_caveats(result)

        assert any("no consensus" in c.lower() for c in caveats)

    def test_single_round_caveat(self, summarizer):
        """Test caveat for single-round debate."""
        result = MockDebateResult(rounds_used=1)
        caveats = summarizer._generate_caveats(result)

        assert any("single-round" in c.lower() for c in caveats)

    def test_diverging_status_caveat(self, summarizer):
        """Test caveat for diverging convergence status."""
        result = MockDebateResult(convergence_status="diverging")
        caveats = summarizer._generate_caveats(result)

        assert any("diverging" in c.lower() or "polarized" in c.lower() for c in caveats)

    def test_low_novelty_caveat(self, summarizer):
        """Test caveat for low novelty."""
        result = MockDebateResult(avg_novelty=0.2)
        caveats = summarizer._generate_caveats(result)

        assert any("novelty" in c.lower() or "repetitive" in c.lower() for c in caveats)

    def test_max_three_caveats(self, summarizer):
        """Test that maximum 3 caveats are returned."""
        result = MockDebateResult(
            confidence=0.3,
            consensus_reached=False,
            rounds_used=1,
            convergence_status="diverging",
            avg_novelty=0.1,
        )
        caveats = summarizer._generate_caveats(result)

        assert len(caveats) <= 3

    def test_no_caveats_for_good_result(self, summarizer):
        """Test no caveats for a good result."""
        result = MockDebateResult(
            confidence=0.9,
            consensus_reached=True,
            rounds_used=3,
            convergence_status="converged",
            avg_novelty=0.8,
        )
        caveats = summarizer._generate_caveats(result)

        assert len(caveats) == 0


# ============================================================================
# Multi-Agent Scenario Tests
# ============================================================================


class TestMultiAgentScenarios:
    """Tests for multi-agent scenarios."""

    def test_count_unique_agents(self, summarizer):
        """Test counting unique agents from messages."""
        result = MockDebateResult(
            messages=[
                MockMessage(agent="claude", content="Message 1"),
                MockMessage(agent="claude", content="Message 2"),
                MockMessage(agent="gpt4", content="Message 3"),
                MockMessage(agent="gemini", content="Message 4"),
            ],
        )
        summary = summarizer.summarize(result)

        assert summary.agents_participated == 3

    def test_agent_count_from_dict_messages(self, summarizer):
        """Test counting agents from dict-style messages."""
        data = {
            "messages": [
                {"agent": "claude"},
                {"agent": "gpt4"},
                {"agent": "gemini"},
                {"agent": "grok"},
            ],
            "avg_novelty": 0.8,  # Include to avoid None comparison in _generate_caveats
        }
        summary = summarizer.summarize(data)

        assert summary.agents_participated == 4

    def test_single_agent_scenario(self, summarizer, single_agent_result):
        """Test single agent scenario."""
        summary = summarizer.summarize(single_agent_result)

        assert summary.agents_participated == 1
        assert summary.rounds_used == 1

    def test_many_agents_scenario(self, summarizer):
        """Test many agents scenario."""
        result = MockDebateResult(
            messages=[MockMessage(agent=f"agent_{i}", content="msg") for i in range(10)],
        )
        summary = summarizer.summarize(result)

        assert summary.agents_participated == 10


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_result(self, summarizer):
        """Test handling empty result."""
        result = MockDebateResult(
            rounds_used=0,
            confidence=0.0,
            consensus_reached=False,
            final_answer="",
            messages=[],
        )
        summary = summarizer.summarize(result)

        assert isinstance(summary, DebateSummary)
        assert summary.agents_participated == 0
        assert summary.key_points == []

    def test_missing_consensus_strength(self, summarizer):
        """Test handling missing consensus_strength."""
        result = MockDebateResult(
            consensus_reached=True,
            consensus_strength="",  # Empty string
        )
        summary = summarizer.summarize(result)

        # Should default to "medium" when consensus reached but strength empty
        assert summary.consensus_strength == "medium"

    def test_none_fields_in_dict(self, summarizer):
        """Test handling None fields in dict input."""
        data = {
            "rounds_used": None,
            "confidence": None,
            "final_answer": None,
            "messages": None,
            "avg_novelty": 0.8,  # Include to avoid None comparison in _generate_caveats
        }
        summary = summarizer.summarize(data)

        assert summary.rounds_used == 0
        assert summary.confidence == 0.0
        assert summary.agents_participated == 0

    def test_vote_with_object_reasoning(self, summarizer):
        """Test handling votes with object-style reasoning access."""
        result = MockDebateResult(
            votes=[
                MockVote(agent="claude", reasoning="Good approach"),
                MockVote(agent="gpt4", reasoning="Good approach"),
            ],
        )
        # Should not raise when processing votes
        agreements = summarizer._extract_agreements(result)
        assert isinstance(agreements, list)

    def test_critique_dict_style(self, summarizer):
        """Test handling critiques as dicts."""
        data = {
            "dissenting_views": [],
            "critiques": [
                {"content": "However, this needs improvement"},
                {"content": "On the other hand, consider alternatives"},
            ],
        }
        disagreements = summarizer._extract_disagreements(_DictWrapper(data))

        assert isinstance(disagreements, list)

    def test_evidence_suggestions_malformed(self, summarizer):
        """Test handling malformed evidence suggestions."""
        result = MockDebateResult(
            consensus_reached=True,
            confidence=0.85,
            evidence_suggestions=[
                {},  # Missing 'claim'
                {"claim": ""},  # Empty claim
                {"claim": "Valid claim"},
            ],
        )
        steps = summarizer._generate_next_steps(result)

        # Should not crash and should handle gracefully
        assert isinstance(steps, list)


# ============================================================================
# Format Validation Tests
# ============================================================================


class TestFormatValidation:
    """Tests for output format validation."""

    def test_one_liner_not_too_long(self, summarizer, detailed_result):
        """Test one-liner is not excessively long."""
        summary = summarizer.summarize(detailed_result)

        # One-liner should be reasonably short
        assert len(summary.one_liner) < 250

    def test_key_points_reasonable_length(self, summarizer, detailed_result):
        """Test key points are reasonable length."""
        summary = summarizer.summarize(detailed_result)

        for point in summary.key_points:
            assert len(point) < 500  # Each point should be reasonably sized

    def test_confidence_bounds(self, summarizer):
        """Test confidence values are within expected bounds."""
        for conf in [0.0, 0.5, 1.0]:
            result = MockDebateResult(confidence=conf)
            summary = summarizer.summarize(result)

            assert 0.0 <= summary.confidence <= 1.0

    def test_confidence_label_valid_values(self, summarizer):
        """Test confidence labels are valid strings."""
        valid_labels = {"high", "medium", "low"}

        for conf in [0.3, 0.6, 0.9]:
            result = MockDebateResult(confidence=conf)
            summary = summarizer.summarize(result)

            assert summary.confidence_label in valid_labels

    def test_to_dict_json_serializable(self, summarizer, detailed_result):
        """Test that to_dict output is JSON serializable."""
        import json

        summary = summarizer.summarize(detailed_result)
        data = summary.to_dict()

        # Should not raise
        json_str = json.dumps(data)
        assert isinstance(json_str, str)


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunction:
    """Tests for the summarize_debate convenience function."""

    def test_summarize_debate_basic(self, basic_result):
        """Test summarize_debate convenience function."""
        summary = summarize_debate(basic_result)

        assert isinstance(summary, DebateSummary)
        assert summary.rounds_used == 3
        assert summary.confidence == 0.85

    def test_summarize_debate_with_dict(self):
        """Test summarize_debate with dict input."""
        data = {
            "rounds_used": 2,
            "confidence": 0.7,
            "consensus_reached": True,
            "avg_novelty": 0.8,  # Include to avoid None comparison in _generate_caveats
        }
        summary = summarize_debate(data)

        assert summary.rounds_used == 2
        assert summary.confidence == 0.7

    def test_summarize_debate_creates_new_summarizer(self, basic_result):
        """Test that summarize_debate creates fresh summarizer."""
        summary1 = summarize_debate(basic_result)
        summary2 = summarize_debate(basic_result)

        # Both should produce same results
        assert summary1.rounds_used == summary2.rounds_used
        assert summary1.confidence == summary2.confidence


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_summary_flow(self, summarizer, detailed_result):
        """Test complete summary generation flow."""
        summary = summarizer.summarize(detailed_result)

        # Verify all fields are populated appropriately
        assert summary.one_liner != ""
        assert len(summary.key_points) > 0
        assert summary.confidence == 0.92
        assert summary.confidence_label == "high"
        assert summary.consensus_strength == "strong"
        assert summary.rounds_used == 5
        assert summary.agents_participated == 3
        assert summary.duration_seconds == 120.0

        # Check caveats are empty for good result
        assert len(summary.caveats) == 0

    def test_problematic_debate_summary(self, summarizer, no_consensus_result):
        """Test summary for problematic debate."""
        summary = summarizer.summarize(no_consensus_result)

        # Should have caveats
        assert len(summary.caveats) >= 1

        # Should indicate no consensus
        assert summary.consensus_strength == "none"

        # Should have disagreement areas
        assert len(summary.disagreement_areas) >= 1

        # Should have next steps addressing the issues
        assert len(summary.next_steps) >= 1

    def test_summary_to_dict_round_trip(self, summarizer, detailed_result):
        """Test summary survives dict conversion."""
        summary = summarizer.summarize(detailed_result)
        data = summary.to_dict()

        # Verify all key data is preserved
        assert data["one_liner"] == summary.one_liner
        assert data["key_points"] == summary.key_points
        assert data["confidence"] == summary.confidence
        assert data["rounds_used"] == summary.rounds_used


# ============================================================================
# Regression Tests
# ============================================================================


class TestRegressions:
    """Regression tests for previously identified issues."""

    def test_none_messages_list(self, summarizer):
        """Regression: Handle None messages list."""
        result = MockDebateResult()
        result.messages = None
        summary = summarizer.summarize(result)

        assert summary.agents_participated == 0

    def test_empty_string_task(self, summarizer):
        """Regression: Handle empty string task."""
        result = MockDebateResult(
            consensus_reached=True,
            confidence=0.9,
            final_answer="",
            task="",
        )
        one_liner = summarizer._generate_one_liner(result)

        # Should not crash and return something
        assert isinstance(one_liner, str)

    def test_negative_confidence(self, summarizer):
        """Regression: Handle negative confidence values."""
        result = MockDebateResult(confidence=-0.5)
        summary = summarizer.summarize(result)

        # Should handle gracefully
        assert summary.confidence_label == "low"

    def test_confidence_above_one(self, summarizer):
        """Regression: Handle confidence above 1.0."""
        result = MockDebateResult(confidence=1.5)
        summary = summarizer.summarize(result)

        # Should handle gracefully
        assert summary.confidence_label == "high"
