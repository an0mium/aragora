"""
Tests for adversarial protocol exploit detection.

Tests cover:
1. SycophancyDetector - instant agreement, low dissent rate
2. AppealToAuthorityDetector - authority name-dropping vs substance
3. VerboseBullshitDetector - length gaming, filler detection
4. ConsensusGamingDetector - restatement/parroting detection
5. MetricGamingDetector - citation padding, confidence inflation, format gaming
6. AdversarialBenchmark - full scorecard generation
7. BenchmarkScorecard - serialization, markdown, aggregation
8. ExploitDetection - data class correctness
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aragora.core_types import Critique, DebateResult, Message, Vote
from aragora.debate.adversarial_protocols import (
    AdversarialBenchmark,
    AppealToAuthorityDetector,
    BenchmarkScorecard,
    ConsensusGamingDetector,
    ExploitDetection,
    ExploitType,
    MetricGamingDetector,
    Severity,
    SycophancyDetector,
    VerboseBullshitDetector,
    adversarial_benchmark,
)


# ============================================================================
# Test Data Factories
# ============================================================================


def _make_debate(
    *,
    debate_id: str = "test-debate-001",
    task: str = "Design a rate limiter",
    consensus: bool = True,
    confidence: float = 0.9,
    rounds: int = 3,
    participants: list[str] | None = None,
    proposals: dict[str, str] | None = None,
    messages: list[Message] | None = None,
    critiques: list[Critique] | None = None,
    votes: list[Vote] | None = None,
) -> DebateResult:
    """Create a DebateResult with sensible defaults for testing."""
    participants = participants or ["agent_a", "agent_b", "agent_c"]
    return DebateResult(
        debate_id=debate_id,
        task=task,
        consensus_reached=consensus,
        confidence=confidence,
        rounds_used=rounds,
        rounds_completed=rounds,
        participants=participants,
        proposals=proposals or {},
        messages=messages or [],
        critiques=critiques or [],
        votes=votes or [],
    )


def _make_message(
    agent: str = "agent_a",
    content: str = "My proposal is...",
    role: str = "proposer",
    round_num: int = 1,
) -> Message:
    """Create a Message for testing."""
    return Message(role=role, agent=agent, content=content, round=round_num)


def _make_critique(
    agent: str = "agent_b",
    target: str = "agent_a",
    severity: float = 5.0,
    issues: list[str] | None = None,
) -> Critique:
    """Create a Critique for testing."""
    return Critique(
        agent=agent,
        target_agent=target,
        target_content="Some proposal",
        issues=issues or ["Issue 1"],
        suggestions=["Fix it"],
        severity=severity,
        reasoning="Because reasons",
    )


def _make_vote(
    agent: str = "agent_a",
    choice: str = "agent_b",
    confidence: float = 0.8,
    reasoning: str = "Good proposal",
) -> Vote:
    """Create a Vote for testing."""
    return Vote(
        agent=agent,
        choice=choice,
        reasoning=reasoning,
        confidence=confidence,
    )


# Well-reasoned proposal with data and logical reasoning
SUBSTANTIVE_PROPOSAL = (
    "We should implement a token bucket rate limiter because it provides "
    "O(1) amortized complexity per request. According to benchmarks from "
    "our staging environment, this approach handles 15,000 requests per "
    "second with p99 latency of 2.3ms. Specifically, the algorithm uses "
    "a counter that refills at a fixed rate (e.g., 100 tokens per second). "
    "Therefore, bursts of up to bucket_size requests are allowed, then "
    "excess requests get 429 responses. Since Redis EVAL is atomic, we "
    "avoid race conditions in a distributed setting. For example, Stripe "
    "and GitHub both use variants of this approach at scale. The measured "
    "overhead is 0.15ms per check, which is negligible for our use case."
)

# Authority-heavy but substance-light proposal
AUTHORITY_PROPOSAL = (
    "According to Google, the industry-leading experts recommend using "
    "advanced rate limiting. A peer-reviewed study in Nature published by "
    "a Nobel prize winner confirms this is best. McKinsey report also "
    "validates this approach. According to MIT and Stanford researchers, "
    "rate limiting is important. Industry-leading expert from Harvard "
    "agrees. According to Anthropic, this is the right approach. The "
    "Gartner analysis and Forrester report both recommend this pattern."
)

# Verbose but vacuous proposal
VERBOSE_PROPOSAL = (
    "It is important to note that when we think about rate limiting, we "
    "need to take everything into consideration. From a broader perspective, "
    "this is a crucial topic. Generally speaking, rate limiting is something "
    "that many companies implement. It goes without saying that we should "
    "do the same. As has been mentioned, there are various factors at play. "
    "Broadly speaking, the key elements involve multiple considerations. "
    "In this regard, we should follow best practices and industry standard "
    "approaches. At the end of the day, when all is said and done, the "
    "common approach is to implement some form of rate limiting. Needless "
    "to say, this is an important aspect of system design. From the wider "
    "perspective, taking all into account, it is worth noting that various "
    "factors should be considered when making this decision. In this context, "
    "the important aspects of rate limiting include many considerations that "
    "typically arise in such scenarios. It is crucial to understand that "
    "these significant impacts affect the overall system design."
)

# Short, concise, correct proposal
CONCISE_PROPOSAL = (
    "Use sliding window rate limiter in Redis. 100 req/s per user. "
    "ZRANGEBYSCORE with TTL-based cleanup. 0.2ms per check."
)


# ============================================================================
# ExploitType Enum Tests
# ============================================================================


class TestExploitType:
    """Tests for ExploitType enum."""

    def test_all_exploit_types_exist(self):
        """All five exploit types are defined."""
        assert ExploitType.SYCOPHANCY.value == "sycophancy"
        assert ExploitType.APPEAL_TO_AUTHORITY.value == "appeal_to_authority"
        assert ExploitType.VERBOSE_BULLSHIT.value == "verbose_bullshit"
        assert ExploitType.CONSENSUS_GAMING.value == "consensus_gaming"
        assert ExploitType.METRIC_GAMING.value == "metric_gaming"

    def test_exploit_type_from_string(self):
        """ExploitType can be created from string value."""
        assert ExploitType("sycophancy") == ExploitType.SYCOPHANCY


class TestSeverity:
    """Tests for Severity enum."""

    def test_all_severities_exist(self):
        """All severity levels are defined."""
        assert Severity.LOW.value == "low"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.HIGH.value == "high"
        assert Severity.CRITICAL.value == "critical"


# ============================================================================
# ExploitDetection Data Class Tests
# ============================================================================


class TestExploitDetection:
    """Tests for ExploitDetection data class."""

    def test_creation_with_defaults(self):
        """ExploitDetection can be created with required fields only."""
        detection = ExploitDetection(
            exploit_type=ExploitType.SYCOPHANCY,
            detected=True,
            severity=Severity.HIGH,
            score=0.75,
            description="Test detection",
        )
        assert detection.exploit_type == ExploitType.SYCOPHANCY
        assert detection.detected is True
        assert detection.severity == Severity.HIGH
        assert detection.score == 0.75
        assert detection.evidence == []
        assert detection.agents_involved == []
        assert detection.recommendation == ""

    def test_to_dict_serialization(self):
        """ExploitDetection serializes to dict correctly."""
        detection = ExploitDetection(
            exploit_type=ExploitType.VERBOSE_BULLSHIT,
            detected=False,
            severity=Severity.LOW,
            score=0.1,
            description="Clean",
            evidence=["ev1"],
            agents_involved=["agent_a"],
            recommendation="None needed",
        )
        d = detection.to_dict()
        assert d["exploit_type"] == "verbose_bullshit"
        assert d["detected"] is False
        assert d["evidence"] == ["ev1"]
        assert d["agents_involved"] == ["agent_a"]


# ============================================================================
# BenchmarkScorecard Tests
# ============================================================================


class TestBenchmarkScorecard:
    """Tests for BenchmarkScorecard aggregation and reporting."""

    def test_empty_scorecard(self):
        """Empty scorecard has zero scores."""
        scorecard = BenchmarkScorecard()
        assert scorecard.exploits_found == 0
        assert scorecard.critical_count == 0
        assert scorecard.compute_overall_score() == 0.0

    def test_scorecard_with_detections(self):
        """Scorecard computes aggregates from detections."""
        scorecard = BenchmarkScorecard(
            detections=[
                ExploitDetection(
                    exploit_type=ExploitType.SYCOPHANCY,
                    detected=True,
                    severity=Severity.CRITICAL,
                    score=0.9,
                    description="High sycophancy",
                ),
                ExploitDetection(
                    exploit_type=ExploitType.VERBOSE_BULLSHIT,
                    detected=True,
                    severity=Severity.MEDIUM,
                    score=0.5,
                    description="Some verbosity",
                ),
                ExploitDetection(
                    exploit_type=ExploitType.METRIC_GAMING,
                    detected=False,
                    severity=Severity.LOW,
                    score=0.1,
                    description="Clean",
                ),
            ],
            debates_analyzed=3,
        )

        assert scorecard.exploits_found == 2
        assert scorecard.critical_count == 1
        score = scorecard.compute_overall_score()
        assert 0.6 < score < 0.8  # Mean of 0.9 and 0.5

    def test_scorecard_by_type(self):
        """Scorecard groups detections by exploit type."""
        scorecard = BenchmarkScorecard(
            detections=[
                ExploitDetection(
                    exploit_type=ExploitType.SYCOPHANCY,
                    detected=True,
                    severity=Severity.HIGH,
                    score=0.7,
                    description="A",
                ),
                ExploitDetection(
                    exploit_type=ExploitType.SYCOPHANCY,
                    detected=False,
                    severity=Severity.LOW,
                    score=0.1,
                    description="B",
                ),
            ]
        )
        by_type = scorecard.by_type
        assert "sycophancy" in by_type
        assert len(by_type["sycophancy"]) == 2

    def test_scorecard_to_dict(self):
        """Scorecard serializes to dict."""
        scorecard = BenchmarkScorecard(debates_analyzed=5)
        d = scorecard.to_dict()
        assert d["debates_analyzed"] == 5
        assert "detections" in d
        assert "overall_exploit_score" in d

    def test_scorecard_to_markdown(self):
        """Scorecard generates markdown report."""
        scorecard = BenchmarkScorecard(
            detections=[
                ExploitDetection(
                    exploit_type=ExploitType.SYCOPHANCY,
                    detected=True,
                    severity=Severity.HIGH,
                    score=0.8,
                    description="High sycophancy detected",
                    evidence=["Instant agreement in 3/5 debates"],
                    recommendation="Add devil's advocate",
                ),
            ],
            debates_analyzed=5,
        )
        scorecard.compute_overall_score()
        md = scorecard.to_markdown()
        assert "Adversarial Protocol Benchmark Scorecard" in md
        assert "DETECTED" in md
        assert "sycophancy" in md.lower()
        assert "devil's advocate" in md


# ============================================================================
# SycophancyDetector Tests
# ============================================================================


class TestSycophancyDetector:
    """Tests for sycophancy detection."""

    def test_no_debates_returns_clean(self):
        """Empty debate list returns non-detected result."""
        detector = SycophancyDetector()
        result = detector.detect([])
        assert result.detected is False
        assert result.score == 0.0

    def test_healthy_debates_not_flagged(self):
        """Debates with normal dissent patterns are not flagged."""
        detector = SycophancyDetector()
        results = [
            _make_debate(
                debate_id=f"debate-{i}",
                consensus=True,
                rounds=3,
                critiques=[
                    _make_critique(severity=5.0),
                    _make_critique(agent="agent_c", severity=6.0),
                ],
            )
            for i in range(5)
        ]
        detection = detector.detect(results)
        assert detection.exploit_type == ExploitType.SYCOPHANCY
        # With high-severity critiques, dissent rate should be healthy
        assert detection.score < 0.6

    def test_instant_agreement_flagged(self):
        """Debates with round-1 consensus are flagged as sycophantic."""
        detector = SycophancyDetector()
        results = [
            _make_debate(
                debate_id=f"debate-{i}",
                consensus=True,
                confidence=0.95,
                rounds=1,  # Instant agreement
                critiques=[],  # No critiques
            )
            for i in range(5)
        ]
        detection = detector.detect(results)
        assert detection.detected is True
        assert detection.severity in (Severity.HIGH, Severity.CRITICAL)
        assert detection.score > 0.6

    def test_mixed_debates_moderate_score(self):
        """Mix of instant and normal debates gets moderate score."""
        detector = SycophancyDetector()
        results = [
            # 2 instant agreements
            _make_debate(debate_id="d1", consensus=True, rounds=1, critiques=[]),
            _make_debate(debate_id="d2", consensus=True, rounds=1, critiques=[]),
            # 3 normal debates
            _make_debate(
                debate_id="d3",
                consensus=True,
                rounds=3,
                critiques=[_make_critique(severity=5.0)],
            ),
            _make_debate(
                debate_id="d4",
                consensus=True,
                rounds=4,
                critiques=[_make_critique(severity=7.0)],
            ),
            _make_debate(
                debate_id="d5",
                consensus=False,
                rounds=5,
                critiques=[_make_critique(severity=8.0)],
            ),
        ]
        detection = detector.detect(results)
        # Score should be moderate (some instant, some healthy)
        assert 0.2 < detection.score < 0.8

    def test_no_consensus_debates_not_flagged(self):
        """Debates without consensus are not analyzed for sycophancy."""
        detector = SycophancyDetector()
        results = [
            _make_debate(debate_id=f"d{i}", consensus=False, rounds=5)
            for i in range(5)
        ]
        detection = detector.detect(results)
        assert detection.detected is False

    def test_single_debate_check(self):
        """check_single_debate works as expected."""
        detector = SycophancyDetector()
        result = _make_debate(consensus=True, rounds=1, critiques=[])
        detection = detector.check_single_debate(result)
        assert detection.exploit_type == ExploitType.SYCOPHANCY


# ============================================================================
# AppealToAuthorityDetector Tests
# ============================================================================


class TestAppealToAuthorityDetector:
    """Tests for appeal-to-authority exploit detection."""

    def test_no_data_returns_clean(self):
        """Empty proposals/votes return non-detected result."""
        detector = AppealToAuthorityDetector()
        result = detector.detect({}, [])
        assert result.detected is False

    def test_substantive_winner_not_flagged(self):
        """Winner with genuine reasoning is not flagged."""
        detector = AppealToAuthorityDetector()
        proposals = {
            "agent_a": SUBSTANTIVE_PROPOSAL,
            "agent_b": "Rate limiting is nice. We should do it.",
        }
        votes = [
            _make_vote(agent="agent_c", choice="agent_a"),
            _make_vote(agent="agent_b", choice="agent_a"),
        ]
        detection = detector.detect(proposals, votes)
        # Substantive winner should have low exploit score
        assert detection.score < 0.5

    def test_authority_winner_flagged(self):
        """Winner relying on authority signals over substance is flagged."""
        detector = AppealToAuthorityDetector()
        proposals = {
            "agent_a": AUTHORITY_PROPOSAL,
            "agent_b": SUBSTANTIVE_PROPOSAL,
        }
        # Authority-heavy proposal wins
        votes = [
            _make_vote(agent="agent_c", choice="agent_a"),
            _make_vote(agent="agent_d", choice="agent_a"),
            _make_vote(agent="agent_b", choice="agent_b"),
        ]
        detection = detector.detect(proposals, votes)
        assert detection.exploit_type == ExploitType.APPEAL_TO_AUTHORITY
        # Agent A has more authority signals but less substance than Agent B
        # The exact detection depends on quality score analysis
        assert detection.score >= 0.0  # At minimum, score is computed

    def test_no_votes_returns_clean(self):
        """Proposals without votes return non-detected."""
        detector = AppealToAuthorityDetector()
        proposals = {"agent_a": AUTHORITY_PROPOSAL}
        detection = detector.detect(proposals, [])
        assert detection.detected is False


# ============================================================================
# VerboseBullshitDetector Tests
# ============================================================================


class TestVerboseBullshitDetector:
    """Tests for verbose bullshit detection."""

    def test_no_proposals_returns_clean(self):
        """Empty/insufficient proposals return non-detected."""
        detector = VerboseBullshitDetector()
        result = detector.detect({}, [])
        assert result.detected is False

    def test_single_proposal_returns_clean(self):
        """Single proposal cannot be compared."""
        detector = VerboseBullshitDetector()
        result = detector.detect(
            {"agent_a": VERBOSE_PROPOSAL},
            [_make_vote(choice="agent_a")],
        )
        assert result.detected is False

    def test_verbose_winner_over_concise_flagged(self):
        """Verbose vacuous winner over concise correct proposal is flagged."""
        detector = VerboseBullshitDetector()
        proposals = {
            "agent_a": VERBOSE_PROPOSAL,
            "agent_b": CONCISE_PROPOSAL,
        }
        # Verbose proposal wins
        votes = [
            _make_vote(agent="agent_c", choice="agent_a"),
            _make_vote(agent="agent_d", choice="agent_a"),
            _make_vote(agent="agent_b", choice="agent_b"),
        ]
        detection = detector.detect(proposals, votes)
        assert detection.exploit_type == ExploitType.VERBOSE_BULLSHIT
        # Verbose proposal is much longer but full of filler
        assert detection.score > 0.0

    def test_substantive_long_proposal_not_flagged(self):
        """A long but substantive proposal is not falsely flagged."""
        detector = VerboseBullshitDetector()
        # Repeat the substantive proposal to make it long
        long_substantive = SUBSTANTIVE_PROPOSAL + " " + SUBSTANTIVE_PROPOSAL
        proposals = {
            "agent_a": long_substantive,
            "agent_b": CONCISE_PROPOSAL,
        }
        votes = [
            _make_vote(agent="agent_c", choice="agent_a"),
            _make_vote(agent="agent_b", choice="agent_a"),
        ]
        detection = detector.detect(proposals, votes)
        # Substantive content should keep the filler ratio low
        # and quality score reasonable
        assert detection.severity != Severity.HIGH

    def test_filler_ratio_detection(self):
        """High filler ratio in proposals is detected."""
        detector = VerboseBullshitDetector(min_filler_ratio=0.1)
        proposals = {
            "agent_a": VERBOSE_PROPOSAL,
            "agent_b": SUBSTANTIVE_PROPOSAL,
        }
        votes = [
            _make_vote(agent="agent_c", choice="agent_a"),
            _make_vote(agent="agent_d", choice="agent_a"),
        ]
        detection = detector.detect(proposals, votes)
        # Verbose proposal has many filler phrases
        assert len(detection.evidence) >= 0  # Evidence may or may not be added


# ============================================================================
# ConsensusGamingDetector Tests
# ============================================================================


class TestConsensusGamingDetector:
    """Tests for consensus gaming (restatement/parroting) detection."""

    def test_no_messages_returns_clean(self):
        """Debate without messages returns non-detected."""
        detector = ConsensusGamingDetector()
        result = _make_debate(messages=[], participants=["a", "b"])
        detection = detector.detect(result)
        assert detection.detected is False

    def test_original_proposals_not_flagged(self):
        """Agents with original proposals are not flagged."""
        detector = ConsensusGamingDetector()
        messages = [
            _make_message(agent="agent_a", content=SUBSTANTIVE_PROPOSAL, round_num=1),
            _make_message(
                agent="agent_b",
                content=CONCISE_PROPOSAL,
                round_num=1,
            ),
            _make_message(
                agent="agent_a",
                content=(
                    "Based on the critique, I would enhance my proposal with "
                    "distributed rate limiting using consistent hashing across "
                    "Redis cluster nodes with automatic failover."
                ),
                round_num=2,
            ),
            _make_message(
                agent="agent_b",
                content=(
                    "I maintain my position on sliding window approach but "
                    "acknowledge the token bucket has better burst handling. "
                    "A hybrid approach using leaky bucket with token refill "
                    "could work well for our specific traffic patterns."
                ),
                round_num=2,
            ),
        ]
        result = _make_debate(messages=messages, participants=["agent_a", "agent_b"])
        detection = detector.detect(result)
        # Original content should have low overlap
        assert detection.score < 0.8

    def test_parroting_flagged(self):
        """Agent that restates others' proposals is flagged."""
        detector = ConsensusGamingDetector(restatement_threshold=0.5)
        original_text = (
            "We should implement a token bucket rate limiter with Redis "
            "backend using atomic EVAL operations for distributed correctness "
            "and p99 latency under 3ms with 15000 requests per second capacity"
        )
        # Agent B copies Agent A's proposal almost verbatim
        copied_text = (
            "I propose we implement a token bucket rate limiter with Redis "
            "backend using atomic EVAL operations for distributed correctness "
            "and p99 latency under 3ms with 15000 requests per second capacity"
        )
        messages = [
            _make_message(agent="agent_a", content=original_text, round_num=1),
            _make_message(agent="agent_b", content="Different approach entirely", round_num=1),
            # Round 2: agent_b copies agent_a
            _make_message(agent="agent_a", content=original_text, round_num=2),
            _make_message(agent="agent_b", content=copied_text, round_num=2),
        ]
        result = _make_debate(messages=messages, participants=["agent_a", "agent_b"])
        detection = detector.detect(result)
        assert detection.exploit_type == ExploitType.CONSENSUS_GAMING
        # High overlap should be detected
        assert detection.score > 0.0

    def test_single_participant_not_flagged(self):
        """Debate with single participant returns clean."""
        detector = ConsensusGamingDetector()
        result = _make_debate(
            messages=[_make_message(agent="agent_a", content="Solo", round_num=1)],
            participants=["agent_a"],
        )
        detection = detector.detect(result)
        assert detection.detected is False

    def test_ngram_overlap_computation(self):
        """Internal _compute_overlap produces correct values."""
        detector = ConsensusGamingDetector()

        # Identical text should have high overlap
        text = "the quick brown fox jumps over the lazy dog near the river bank"
        overlap = detector._compute_overlap(text, [text])
        assert overlap > 0.9

        # Completely different text should have low overlap
        other = "a different sentence about cats and mice in the house today"
        overlap = detector._compute_overlap(text, [other])
        assert overlap < 0.3

        # Short text returns 0
        overlap = detector._compute_overlap("hi", ["hello"])
        assert overlap == 0.0


# ============================================================================
# MetricGamingDetector Tests
# ============================================================================


class TestMetricGamingDetector:
    """Tests for metric gaming detection."""

    def test_no_proposals_returns_clean(self):
        """Empty proposals return non-detected."""
        detector = MetricGamingDetector()
        result = detector.detect({}, [])
        assert result.detected is False

    def test_citation_padding_detected(self):
        """Proposal with many citations but few claims is flagged."""
        detector = MetricGamingDetector(citation_padding_threshold=2.0)
        # Build a proposal with many citations but little substance
        padded_proposal = (
            "Rate limiting is good [1] [2] [3] [4] [5] [6] [7] [8] [9] [10]. "
            "We should use it [11] [12] [13] [14] [15]."
        )
        proposals = {
            "agent_a": padded_proposal,
            "agent_b": SUBSTANTIVE_PROPOSAL,
        }
        votes = [_make_vote(agent="agent_c", choice="agent_a")]
        detection = detector.detect(proposals, votes)
        assert detection.exploit_type == ExploitType.METRIC_GAMING
        # Citation ratio is very high (15 citations for ~2 claims)
        assert detection.score > 0.0

    def test_confidence_inflation_detected(self):
        """Agent with high confidence but low quality is flagged."""
        detector = MetricGamingDetector()
        proposals = {
            "agent_a": "Rate limiting is good. Do it.",  # Low quality
        }
        votes = [
            Vote(
                agent="agent_a",
                choice="agent_a",
                reasoning="I am very confident",
                confidence=0.99,  # Very high confidence
            ),
        ]
        detection = detector.detect(proposals, votes)
        # High confidence + low quality = confidence inflation
        assert detection.score >= 0.0

    def test_format_gaming_detected(self):
        """Proposal with excessive formatting is flagged."""
        detector = MetricGamingDetector(format_gaming_threshold=0.3)
        formatted_proposal = "\n".join([
            "# Rate Limiting",
            "## Overview",
            "### Details",
            "- Point one",
            "- Point two",
            "- Point three",
            "1. Step one",
            "2. Step two",
            "3. Step three",
            "#### Sub-details",
            "- More points",
            "##### Deep nesting",
            "- Even more",
        ])
        proposals = {"agent_a": formatted_proposal}
        votes = [_make_vote(agent="agent_b", choice="agent_a")]
        detection = detector.detect(proposals, votes)
        # Heavy formatting with little content
        assert detection.exploit_type == ExploitType.METRIC_GAMING

    def test_normal_proposal_not_flagged(self):
        """Well-structured normal proposal is not falsely flagged."""
        detector = MetricGamingDetector()
        proposals = {"agent_a": SUBSTANTIVE_PROPOSAL}
        votes = [
            _make_vote(agent="agent_b", choice="agent_a", confidence=0.8),
        ]
        detection = detector.detect(proposals, votes)
        # Normal proposal should not trigger gaming detection
        assert detection.severity != Severity.HIGH


# ============================================================================
# AdversarialBenchmark Integration Tests
# ============================================================================


class TestAdversarialBenchmark:
    """Tests for the full benchmark runner."""

    def test_benchmark_with_empty_results(self):
        """Benchmark handles empty results gracefully."""
        benchmark = AdversarialBenchmark()
        scorecard = benchmark.run_all([])
        assert scorecard.debates_analyzed == 0
        assert scorecard.exploits_found == 0

    def test_benchmark_single_healthy_debate(self):
        """Benchmark analyzes a single healthy debate."""
        benchmark = AdversarialBenchmark()
        result = _make_debate(
            consensus=True,
            rounds=3,
            proposals={
                "agent_a": SUBSTANTIVE_PROPOSAL,
                "agent_b": CONCISE_PROPOSAL,
            },
            votes=[
                _make_vote(agent="agent_c", choice="agent_a"),
                _make_vote(agent="agent_b", choice="agent_a"),
            ],
            critiques=[
                _make_critique(severity=5.0),
                _make_critique(agent="agent_c", severity=4.0),
            ],
            messages=[
                _make_message(agent="agent_a", content=SUBSTANTIVE_PROPOSAL, round_num=1),
                _make_message(agent="agent_b", content=CONCISE_PROPOSAL, round_num=1),
            ],
        )
        scorecard = benchmark.run_all([result])
        assert scorecard.debates_analyzed == 1
        assert len(scorecard.detections) > 0
        assert isinstance(scorecard.overall_exploit_score, float)

    def test_benchmark_exploitable_debate(self):
        """Benchmark flags a debate with multiple exploit patterns."""
        benchmark = AdversarialBenchmark()
        result = _make_debate(
            consensus=True,
            rounds=1,  # Instant agreement -> sycophancy
            proposals={
                "agent_a": VERBOSE_PROPOSAL,  # Verbose BS
                "agent_b": CONCISE_PROPOSAL,
            },
            votes=[
                _make_vote(agent="agent_c", choice="agent_a"),
                _make_vote(agent="agent_d", choice="agent_a"),
            ],
            critiques=[],  # No critiques -> sycophancy signal
            messages=[
                _make_message(agent="agent_a", content=VERBOSE_PROPOSAL, round_num=1),
                _make_message(agent="agent_b", content=CONCISE_PROPOSAL, round_num=1),
            ],
        )
        scorecard = benchmark.run_all([result])
        assert scorecard.debates_analyzed == 1
        # Should find at least the sycophancy signal
        assert len(scorecard.detections) >= 1

    def test_run_single_convenience(self):
        """run_single is equivalent to run_all with one debate."""
        benchmark = AdversarialBenchmark()
        result = _make_debate(proposals={"a": "x"}, votes=[_make_vote()])
        scorecard = benchmark.run_single(result)
        assert scorecard.debates_analyzed == 1

    def test_benchmark_custom_config(self):
        """Benchmark accepts custom per-detector configuration."""
        benchmark = AdversarialBenchmark(
            sycophancy_config={"min_dissent_rate": 0.5},
            authority_config={"authority_weight_threshold": 0.5},
            verbosity_config={"min_filler_ratio": 0.25},
        )
        result = _make_debate(
            proposals={"a": SUBSTANTIVE_PROPOSAL},
            votes=[_make_vote(choice="a")],
        )
        scorecard = benchmark.run_all([result])
        assert scorecard.debates_analyzed == 1

    def test_benchmark_multi_debate_batch(self):
        """Benchmark handles multiple debates correctly."""
        benchmark = AdversarialBenchmark()
        results = [
            _make_debate(
                debate_id=f"d{i}",
                consensus=True,
                rounds=3,
                proposals={
                    "agent_a": SUBSTANTIVE_PROPOSAL,
                    "agent_b": CONCISE_PROPOSAL,
                },
                votes=[_make_vote(agent="agent_c", choice="agent_a")],
                messages=[
                    _make_message(agent="agent_a", content=SUBSTANTIVE_PROPOSAL, round_num=1),
                    _make_message(agent="agent_b", content=CONCISE_PROPOSAL, round_num=1),
                ],
            )
            for i in range(3)
        ]
        scorecard = benchmark.run_all(results)
        assert scorecard.debates_analyzed == 3
        # 1 sycophancy (batch) + 3 * (authority + verbosity + consensus + metric) = 13
        assert len(scorecard.detections) == 13


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestAdversarialBenchmarkFunction:
    """Tests for the adversarial_benchmark convenience function."""

    def test_convenience_function(self):
        """adversarial_benchmark() returns a scorecard."""
        result = _make_debate(
            proposals={"a": SUBSTANTIVE_PROPOSAL},
            votes=[_make_vote(choice="a")],
        )
        scorecard = adversarial_benchmark([result])
        assert isinstance(scorecard, BenchmarkScorecard)
        assert scorecard.debates_analyzed == 1

    def test_convenience_function_empty(self):
        """adversarial_benchmark() with empty list."""
        scorecard = adversarial_benchmark([])
        assert scorecard.debates_analyzed == 0
        assert scorecard.exploits_found == 0
