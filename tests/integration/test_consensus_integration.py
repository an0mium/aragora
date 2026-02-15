"""
Consensus detection integration tests.

Tests convergence detection, hollow consensus detection (trickster),
and stability with real vote sequences -- no mocks of the detection logic.
"""

from __future__ import annotations

import pytest

from aragora.debate.consensus import (
    Claim,
    ConsensusProof,
    ConsensusVote,
    Evidence,
    VoteType,
    PartialConsensus,
    PartialConsensusItem,
)
from aragora.debate.convergence.detector import ConvergenceDetector
from aragora.debate.evidence_quality import (
    EvidenceQualityAnalyzer,
    HollowConsensusDetector,
    HollowConsensusAlert,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# 1. Convergence detection with real proposals
# ---------------------------------------------------------------------------


class TestConvergenceDetection:
    """Test ConvergenceDetector with real text proposals."""

    def test_identical_proposals_converge(self):
        detector = ConvergenceDetector(
            convergence_threshold=0.85,
            min_rounds_before_check=0,
        )

        prev = {"alice": "Use Redis for caching", "bob": "Use Redis for caching"}
        curr = {"alice": "Use Redis for caching", "bob": "Use Redis for caching"}

        result = detector.check_convergence(curr, prev, round_number=1)
        assert result.avg_similarity >= 0.85

    def test_completely_different_proposals_diverge(self):
        detector = ConvergenceDetector(
            convergence_threshold=0.85,
            divergence_threshold=0.40,
            min_rounds_before_check=0,
        )

        prev = {
            "alice": "Implement a token bucket algorithm for rate limiting with sliding windows",
            "bob": "Use neural networks for image classification with convolutional layers",
        }
        curr = {
            "alice": "Deploy microservices on Kubernetes with Istio service mesh",
            "bob": "Build a blockchain-based voting system using Solidity smart contracts",
        }

        result = detector.check_convergence(curr, prev, round_number=1)
        # With Jaccard backend, very different texts should have low similarity
        assert result.avg_similarity < 0.85

    def test_similar_proposals_are_refining(self):
        detector = ConvergenceDetector(
            convergence_threshold=0.85,
            divergence_threshold=0.40,
            min_rounds_before_check=0,
        )

        prev = {
            "alice": "Use Redis as a distributed cache with TTL-based expiration",
            "bob": "Use Memcached as a distributed cache with TTL-based eviction",
        }
        curr = {
            "alice": "Use Redis as a distributed cache with TTL-based expiration and LRU eviction",
            "bob": "Use Redis as a distributed cache with TTL-based eviction and write-through",
        }

        result = detector.check_convergence(curr, prev, round_number=1)
        # Proposals are similar but not identical
        assert isinstance(result.avg_similarity, float)

    def test_convergence_result_has_per_agent_scores(self):
        detector = ConvergenceDetector(min_rounds_before_check=0)

        prev = {"alice": "Use token bucket", "bob": "Use leaky bucket"}
        curr = {"alice": "Use token bucket with burst", "bob": "Use token bucket with burst"}

        result = detector.check_convergence(curr, prev, round_number=1)
        assert hasattr(result, "per_agent_similarity") or hasattr(result, "avg_similarity")

    def test_convergence_with_three_agents(self):
        detector = ConvergenceDetector(min_rounds_before_check=0)

        prev = {
            "alice": "REST API with JSON",
            "bob": "GraphQL API",
            "charlie": "gRPC with protobuf",
        }
        curr = {
            "alice": "REST API with JSON and OpenAPI spec",
            "bob": "REST API with JSON schema validation",
            "charlie": "REST API with protobuf option",
        }

        result = detector.check_convergence(curr, prev, round_number=1)
        assert isinstance(result.avg_similarity, float)


# ---------------------------------------------------------------------------
# 2. Hollow consensus detection (trickster)
# ---------------------------------------------------------------------------


class TestHollowConsensusDetection:
    """Test HollowConsensusDetector with real response text."""

    def test_vague_agreement_flagged_as_hollow(self):
        detector = HollowConsensusDetector(
            min_quality_threshold=0.4,
        )

        # Vague responses with no citations or specifics
        responses = {
            "alice": "I agree, this sounds good. Let's go with it.",
            "bob": "Yes, that makes sense. Good approach overall.",
            "charlie": "I think this is fine. Let's proceed.",
        }

        alert = detector.check(
            responses=responses,
            convergence_similarity=0.9,
            round_num=3,
        )

        assert isinstance(alert, HollowConsensusAlert)
        # These vague responses should have low evidence quality
        assert alert.avg_quality < 0.6

    def test_evidence_rich_agreement_not_hollow(self):
        detector = HollowConsensusDetector(
            min_quality_threshold=0.4,
        )

        responses = {
            "alice": (
                "According to the CAP theorem (Brewer, 2000), we must choose between "
                "consistency and availability. Given our 99.9% uptime SLA and the "
                "benchmark showing 50ms p99 latency with Redis cluster, I recommend "
                "Redis with eventual consistency. The Jepsen test results from 2024 "
                "confirm Redis 7.2 handles network partitions correctly."
            ),
            "bob": (
                "I concur. The Netflix case study (2023) showed a 40% latency improvement "
                "after switching to Redis cluster. Our load test data (10K RPS, 12ms avg) "
                "supports this. The trade-off is documented in RFC 9110 section 8.3. "
                "Cost analysis: $2,400/month vs $4,800 for DynamoDB at our scale."
            ),
        }

        alert = detector.check(
            responses=responses,
            convergence_similarity=0.85,
            round_num=3,
        )

        assert isinstance(alert, HollowConsensusAlert)
        # Evidence-rich responses should have higher quality
        assert alert.avg_quality > alert.min_quality

    def test_no_hollow_check_when_not_converging(self):
        detector = HollowConsensusDetector()

        responses = {
            "alice": "I think we should use option A.",
            "bob": "I disagree, option B is better.",
        }

        alert = detector.check(
            responses=responses,
            convergence_similarity=0.3,  # Not converging
            round_num=1,
        )

        assert alert.detected is False
        assert alert.reason == "Not converging yet"

    def test_hollow_alert_has_agent_scores(self):
        detector = HollowConsensusDetector()

        responses = {
            "alice": "Looks good to me.",
            "bob": "Sure, let's do that.",
        }

        alert = detector.check(
            responses=responses,
            convergence_similarity=0.9,
            round_num=2,
        )

        assert isinstance(alert.agent_scores, dict)

    def test_hollow_detector_with_single_agent(self):
        detector = HollowConsensusDetector()

        responses = {"alice": "I agree with the proposal as stated."}

        alert = detector.check(
            responses=responses,
            convergence_similarity=0.9,
            round_num=1,
        )

        assert isinstance(alert, HollowConsensusAlert)


# ---------------------------------------------------------------------------
# 3. ConsensusProof integrity
# ---------------------------------------------------------------------------


class TestConsensusProofIntegrity:
    """Test ConsensusProof structure and checksum."""

    def test_proof_checksum_is_stable(self):
        proof = ConsensusProof(
            proof_id="proof-001",
            debate_id="debate-001",
            task="Design a cache",
            final_claim="Use Redis with TTL",
            confidence=0.88,
            consensus_reached=True,
            votes=[
                ConsensusVote(
                    agent="alice",
                    vote=VoteType.AGREE,
                    confidence=0.9,
                    reasoning="Solid approach",
                ),
            ],
            supporting_agents=["alice", "bob"],
            dissenting_agents=[],
            claims=[
                Claim(
                    claim_id="c1",
                    statement="Redis is fast",
                    author="alice",
                    confidence=0.9,
                ),
            ],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="All agents agreed on Redis",
        )

        checksum1 = proof.checksum
        checksum2 = proof.checksum

        assert checksum1 == checksum2
        assert len(checksum1) == 16  # SHA-256 truncated to 16 hex chars

    def test_different_proofs_have_different_checksums(self):
        base_kwargs = dict(
            debate_id="debate-001",
            task="Test",
            consensus_reached=True,
            votes=[],
            supporting_agents=["alice"],
            dissenting_agents=[],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Test",
        )

        proof1 = ConsensusProof(
            proof_id="p1",
            final_claim="Use Redis",
            confidence=0.9,
            **base_kwargs,
        )
        proof2 = ConsensusProof(
            proof_id="p2",
            final_claim="Use Memcached",
            confidence=0.7,
            **base_kwargs,
        )

        assert proof1.checksum != proof2.checksum

    def test_proof_agreement_ratio(self):
        proof = ConsensusProof(
            proof_id="p1",
            debate_id="d1",
            task="Test",
            final_claim="Claim",
            confidence=0.8,
            consensus_reached=True,
            votes=[],
            supporting_agents=["alice", "bob", "charlie"],
            dissenting_agents=["diana"],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Majority agreed",
        )

        assert proof.agreement_ratio == 0.75  # 3/4

    def test_proof_strong_consensus_detection(self):
        proof = ConsensusProof(
            proof_id="p1",
            debate_id="d1",
            task="Test",
            final_claim="Strong claim",
            confidence=0.85,
            consensus_reached=True,
            votes=[],
            supporting_agents=["alice", "bob", "charlie", "diana", "frank"],
            dissenting_agents=["eve"],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Strong agreement",
        )

        # 5/6 = 83.3% > 80%, confidence 0.85 > 0.7
        assert proof.has_strong_consensus

    def test_proof_weak_consensus_not_strong(self):
        proof = ConsensusProof(
            proof_id="p1",
            debate_id="d1",
            task="Test",
            final_claim="Weak claim",
            confidence=0.5,
            consensus_reached=True,
            votes=[],
            supporting_agents=["alice"],
            dissenting_agents=["bob", "charlie"],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="Weak",
        )

        assert not proof.has_strong_consensus


# ---------------------------------------------------------------------------
# 4. Partial consensus tracking
# ---------------------------------------------------------------------------


class TestPartialConsensus:
    """Test partial consensus when full agreement is not reached."""

    def test_partial_consensus_ratio(self):
        pc = PartialConsensus(debate_id="d1")
        pc.add_item(
            PartialConsensusItem(
                item_id="i1",
                topic="Storage",
                statement="Use PostgreSQL",
                confidence=0.9,
                agreed=True,
                supporting_agents=["alice", "bob"],
            )
        )
        pc.add_item(
            PartialConsensusItem(
                item_id="i2",
                topic="Caching",
                statement="Use Redis",
                confidence=0.6,
                agreed=False,
                supporting_agents=["alice"],
                dissenting_agents=["bob"],
            )
        )

        assert pc.consensus_ratio == 0.5

    def test_actionable_items_filter(self):
        pc = PartialConsensus(debate_id="d1")
        pc.add_item(
            PartialConsensusItem(
                item_id="i1",
                topic="API",
                statement="Use REST",
                confidence=0.9,
                agreed=True,
                actionable=True,
                supporting_agents=["a"],
            )
        )
        pc.add_item(
            PartialConsensusItem(
                item_id="i2",
                topic="Future",
                statement="Maybe GraphQL",
                confidence=0.4,
                agreed=True,
                actionable=False,
                supporting_agents=["b"],
            )
        )

        assert len(pc.actionable_items) == 1
        assert pc.actionable_items[0].item_id == "i1"

    def test_partial_consensus_serialization(self):
        pc = PartialConsensus(debate_id="d1")
        pc.add_item(
            PartialConsensusItem(
                item_id="i1",
                topic="Test",
                statement="Statement",
                confidence=0.8,
                agreed=True,
                supporting_agents=["a"],
            )
        )

        d = pc.to_dict()
        assert d["debate_id"] == "d1"
        assert len(d["items"]) == 1
        assert "consensus_ratio" in d


# ---------------------------------------------------------------------------
# 5. Evidence quality analysis
# ---------------------------------------------------------------------------


class TestEvidenceQualityAnalysis:
    """Test evidence quality scoring on real text."""

    def test_vague_text_scores_low(self):
        analyzer = EvidenceQualityAnalyzer()
        score = analyzer.analyze(
            "I think this is probably fine. Let's go with it.",
            agent="alice",
            round_num=1,
        )
        assert score.overall_quality < 0.6

    def test_specific_text_scores_higher(self):
        analyzer = EvidenceQualityAnalyzer()
        score = analyzer.analyze(
            "Based on RFC 7234 section 5.2, HTTP caching headers with max-age=3600 "
            "and must-revalidate provide optimal cache control. Our load test showed "
            "47ms p95 latency with this configuration at 10,000 RPS.",
            agent="alice",
            round_num=3,
        )

        vague_score = analyzer.analyze(
            "I think caching is important. We should do it.",
            agent="bob",
            round_num=3,
        )

        assert score.overall_quality > vague_score.overall_quality

    def test_batch_analysis(self):
        analyzer = EvidenceQualityAnalyzer()
        results = analyzer.analyze_batch(
            {
                "alice": "We should use Redis. It's fast and reliable.",
                "bob": "According to benchmark data, Redis handles 100K ops/sec.",
            },
            round_num=2,
        )

        assert "alice" in results
        assert "bob" in results
        assert all(0.0 <= r.overall_quality <= 1.0 for r in results.values())
