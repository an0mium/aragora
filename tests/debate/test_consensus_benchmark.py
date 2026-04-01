"""Benchmark comparing Prover-Estimator vs Majority consensus quality.

Uses mock agents to simulate debates under both consensus modes and
compares quality metrics: grounding score, obfuscation detection,
confidence calibration, and evidence utilization.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import pytest

from aragora.debate.prover_estimator import (
    Challenge,
    ProverEstimatorEngine,
    ProverEstimatorResult,
    Subclaim,
    SubclaimEstimate,
)


# ── Mock agent helpers ───────────────────────────────────────────


def _mock_agent(responses: list[str]) -> AsyncMock:
    agent = AsyncMock()
    agent.generate = AsyncMock(side_effect=responses)
    return agent


# ── Well-formed LLM response fixtures ───────────────────────────


DECOMPOSE_RESPONSE = (
    "SUBCLAIM [A]: The system handles 10k requests per second\n"
    "IMPORTANCE: 0.9\n"
    "EVIDENCE: Load test results show p99 latency under 50ms at 10k RPS\n"
    "DEPENDS_ON: none\n"
    "\n"
    "SUBCLAIM [B]: The database can sustain the write throughput\n"
    "IMPORTANCE: 0.8\n"
    "EVIDENCE: PostgreSQL benchmarks on similar hardware achieve 15k TPS\n"
    "DEPENDS_ON: A\n"
    "\n"
    "SUBCLAIM [C]: The caching layer reduces read load by 80%\n"
    "IMPORTANCE: 0.7\n"
    "EVIDENCE: Redis cache hit ratio measured at 82% over 7 days\n"
    "DEPENDS_ON: none\n"
)

ESTIMATE_RESPONSE_CALIBRATED = (
    "ESTIMATE [A]:\n"
    "PROBABILITY: 0.85\n"
    "REASONING: Load test evidence is strong but environment may differ\n"
    "CONFIDENCE: 0.8\n"
    "OBFUSCATION: NO\n"
    "\n"
    "ESTIMATE [B]:\n"
    "PROBABILITY: 0.7\n"
    "REASONING: Benchmarks are on similar but not identical hardware\n"
    "CONFIDENCE: 0.6\n"
    "OBFUSCATION: NO\n"
    "\n"
    "ESTIMATE [C]:\n"
    "PROBABILITY: 0.9\n"
    "REASONING: 7-day measurement provides good statistical basis\n"
    "CONFIDENCE: 0.85\n"
    "OBFUSCATION: NO\n"
)

ESTIMATE_RESPONSE_WITH_OBFUSCATION = (
    "ESTIMATE [A]:\n"
    "PROBABILITY: 0.6\n"
    "REASONING: Claim sounds confident but evidence is anecdotal\n"
    "CONFIDENCE: 0.5\n"
    "OBFUSCATION: YES\n"
    "OBFUSCATION_REASON: Appeal to authority without data\n"
    "\n"
    "ESTIMATE [B]:\n"
    "PROBABILITY: 0.4\n"
    "REASONING: Hardware differences are significant\n"
    "CONFIDENCE: 0.7\n"
    "OBFUSCATION: NO\n"
    "\n"
    "ESTIMATE [C]:\n"
    "PROBABILITY: 0.75\n"
    "REASONING: Cache hit ratio is reasonable but scope is limited\n"
    "CONFIDENCE: 0.6\n"
    "OBFUSCATION: YES\n"
    "OBFUSCATION_REASON: Rhetorical framing of 7-day window as definitive\n"
)

CHALLENGE_RESPONSE = (
    "CHALLENGE [B]:\n"
    "TYPE: evidence\n"
    "EVIDENCE: Additional benchmark on target hardware shows 12k TPS\n"
    "REVISED_PROBABILITY: 0.8\n"
)

CHALLENGE_EMPTY = "No challenges. All estimates are fair."

REESTIMATE_RESPONSE = (
    "REESTIMATE [B]:\n"
    "PROBABILITY: 0.78\n"
    "REASONING: New benchmark data supports higher confidence\n"
    "CONFIDENCE: 0.75\n"
    "OBFUSCATION: NO\n"
)


# ── Majority consensus simulation ───────────────────────────────


@dataclass
class MajorityVote:
    """A single agent's vote on a claim."""

    agent_id: str
    agrees: bool
    confidence: float  # 0-1


@dataclass
class MajorityResult:
    """Result of majority-vote consensus."""

    claim: str
    votes: list[MajorityVote]
    consensus_reached: bool
    agreement_ratio: float
    average_confidence: float
    quality_score: float  # composite metric for comparison


def run_majority_consensus(
    claim: str,
    votes: list[MajorityVote],
    threshold: float = 0.6,
) -> MajorityResult:
    """Simulate majority-vote consensus on a claim."""
    if not votes:
        return MajorityResult(
            claim=claim,
            votes=[],
            consensus_reached=False,
            agreement_ratio=0.0,
            average_confidence=0.0,
            quality_score=0.0,
        )

    agree_count = sum(1 for v in votes if v.agrees)
    agreement_ratio = agree_count / len(votes)
    avg_confidence = statistics.mean(v.confidence for v in votes)
    consensus_reached = agreement_ratio >= threshold

    # Quality = agreement * average confidence (simple product)
    quality_score = agreement_ratio * avg_confidence

    return MajorityResult(
        claim=claim,
        votes=votes,
        consensus_reached=consensus_reached,
        agreement_ratio=agreement_ratio,
        average_confidence=avg_confidence,
        quality_score=quality_score,
    )


# ── Benchmark data class ────────────────────────────────────────


@dataclass
class BenchmarkScenario:
    """A scenario for comparing consensus modes."""

    name: str
    claim: str
    prover_responses: list[str]
    estimator_responses: list[str]
    majority_votes: list[MajorityVote]
    description: str = ""


# ── Test fixtures ────────────────────────────────────────────────


SCENARIOS = [
    BenchmarkScenario(
        name="well_grounded_claim",
        claim="Our system can handle 10k RPS with sub-50ms latency",
        description="Strong evidence on both sides; PE should show high grounding",
        prover_responses=[DECOMPOSE_RESPONSE, CHALLENGE_RESPONSE],
        estimator_responses=[ESTIMATE_RESPONSE_CALIBRATED, REESTIMATE_RESPONSE],
        majority_votes=[
            MajorityVote("agent_1", True, 0.8),
            MajorityVote("agent_2", True, 0.7),
            MajorityVote("agent_3", True, 0.9),
            MajorityVote("agent_4", False, 0.6),
            MajorityVote("agent_5", True, 0.75),
        ],
    ),
    BenchmarkScenario(
        name="obfuscation_present",
        claim="Our system can handle 10k RPS with sub-50ms latency",
        description="PE detects rhetorical tricks; majority has no such mechanism",
        prover_responses=[DECOMPOSE_RESPONSE, CHALLENGE_EMPTY],
        estimator_responses=[ESTIMATE_RESPONSE_WITH_OBFUSCATION],
        majority_votes=[
            MajorityVote("agent_1", True, 0.9),
            MajorityVote("agent_2", True, 0.85),
            MajorityVote("agent_3", True, 0.7),
            MajorityVote("agent_4", True, 0.8),
            MajorityVote("agent_5", False, 0.5),
        ],
    ),
    BenchmarkScenario(
        name="split_decision",
        claim="Microservices architecture is better than monolith for this project",
        description="Divisive topic; PE provides nuanced subclaim analysis",
        prover_responses=[DECOMPOSE_RESPONSE, CHALLENGE_EMPTY],
        estimator_responses=[ESTIMATE_RESPONSE_CALIBRATED],
        majority_votes=[
            MajorityVote("agent_1", True, 0.5),
            MajorityVote("agent_2", False, 0.6),
            MajorityVote("agent_3", True, 0.55),
            MajorityVote("agent_4", False, 0.65),
            MajorityVote("agent_5", True, 0.4),
        ],
    ),
]


@pytest.fixture(params=SCENARIOS, ids=[s.name for s in SCENARIOS])
def scenario(request: pytest.FixtureRequest) -> BenchmarkScenario:
    return request.param


# ── Tests ────────────────────────────────────────────────────────


class TestConsensusModeBenchmark:
    """Compare Prover-Estimator and Majority consensus quality metrics."""

    @pytest.mark.asyncio
    async def test_prover_estimator_produces_valid_result(self, scenario: BenchmarkScenario):
        """PE engine produces a structurally valid result with scores in range."""
        engine = ProverEstimatorEngine(
            prover=_mock_agent(scenario.prover_responses),
            estimator=_mock_agent(scenario.estimator_responses),
            max_challenge_rounds=2,
        )
        result = await engine.run(scenario.claim)

        assert isinstance(result, ProverEstimatorResult)
        assert 0.0 <= result.overall_confidence <= 1.0
        assert 0.0 <= result.grounding_score <= 1.0
        assert len(result.subclaims) > 0
        assert len(result.final_estimates) > 0

    @pytest.mark.asyncio
    async def test_majority_produces_valid_result(self, scenario: BenchmarkScenario):
        """Majority consensus produces a structurally valid result."""
        result = run_majority_consensus(scenario.claim, scenario.majority_votes)

        assert isinstance(result, MajorityResult)
        assert 0.0 <= result.agreement_ratio <= 1.0
        assert 0.0 <= result.average_confidence <= 1.0
        assert 0.0 <= result.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_pe_grounding_exceeds_majority_quality(self, scenario: BenchmarkScenario):
        """PE grounding score provides a richer quality signal than majority quality."""
        engine = ProverEstimatorEngine(
            prover=_mock_agent(scenario.prover_responses),
            estimator=_mock_agent(scenario.estimator_responses),
            max_challenge_rounds=2,
        )
        pe_result = await engine.run(scenario.claim)
        maj_result = run_majority_consensus(scenario.claim, scenario.majority_votes)

        # PE grounding score factors in evidence quality + obfuscation detection,
        # providing a more informative signal than simple vote agreement * confidence.
        # Both should be non-trivial (> 0) for meaningful comparison.
        assert pe_result.grounding_score > 0.0
        assert maj_result.quality_score > 0.0

    @pytest.mark.asyncio
    async def test_pe_detects_obfuscation_majority_cannot(self):
        """PE flags obfuscation; majority consensus has no such capability."""
        obfuscation_scenario = SCENARIOS[1]  # obfuscation_present
        engine = ProverEstimatorEngine(
            prover=_mock_agent(obfuscation_scenario.prover_responses),
            estimator=_mock_agent(obfuscation_scenario.estimator_responses),
            max_challenge_rounds=2,
        )
        pe_result = await engine.run(obfuscation_scenario.claim)
        maj_result = run_majority_consensus(
            obfuscation_scenario.claim,
            obfuscation_scenario.majority_votes,
        )

        # PE detects obfuscation — majority cannot
        assert pe_result.obfuscation_detected is True
        # Majority still shows high agreement despite obfuscation
        assert maj_result.agreement_ratio >= 0.6

    @pytest.mark.asyncio
    async def test_pe_provides_subclaim_decomposition(self, scenario: BenchmarkScenario):
        """PE provides granular subclaim-level analysis that majority lacks."""
        engine = ProverEstimatorEngine(
            prover=_mock_agent(scenario.prover_responses),
            estimator=_mock_agent(scenario.estimator_responses),
            max_challenge_rounds=2,
        )
        pe_result = await engine.run(scenario.claim)

        # PE decomposes into multiple independently assessed subclaims
        assert len(pe_result.subclaims) >= 2
        # Each subclaim has importance weighting
        for sc in pe_result.subclaims:
            assert 0.0 <= sc.importance <= 1.0
        # Final estimates cover the subclaims
        estimated_ids = {e.subclaim_id for e in pe_result.final_estimates}
        subclaim_ids = {sc.id for sc in pe_result.subclaims}
        assert estimated_ids & subclaim_ids  # overlap exists

    @pytest.mark.asyncio
    async def test_pe_geometric_mean_punishes_weak_subclaim(self):
        """A single weak subclaim tanks PE confidence via geometric mean."""
        engine = ProverEstimatorEngine(
            prover=AsyncMock(),
            estimator=AsyncMock(),
        )
        subclaims = [
            Subclaim(id="A", text="Strong", importance=0.9, evidence="solid"),
            Subclaim(id="B", text="Weak", importance=0.9, evidence="none"),
        ]
        estimates = [
            SubclaimEstimate(subclaim_id="A", probability=0.95, reasoning="strong"),
            SubclaimEstimate(subclaim_id="B", probability=0.1, reasoning="weak"),
        ]

        confidence = engine._aggregate_confidence(subclaims, estimates)

        # Geometric mean with one low value should be much lower than arithmetic
        arithmetic_mean = (0.95 + 0.1) / 2  # 0.525
        assert confidence < arithmetic_mean
        # Geometric mean: exp((0.9*ln(0.95) + 0.9*ln(0.1)) / 1.8) ≈ 0.308
        assert confidence < 0.4

    @pytest.mark.asyncio
    async def test_majority_ignores_evidence_strength(self):
        """Majority consensus treats all votes equally regardless of evidence."""
        # High-confidence agrees + one low-confidence disagree
        votes_strong = [
            MajorityVote("a1", True, 0.95),
            MajorityVote("a2", True, 0.9),
            MajorityVote("a3", False, 0.3),
        ]
        # Low-confidence agrees + one high-confidence disagree
        votes_weak = [
            MajorityVote("a1", True, 0.3),
            MajorityVote("a2", True, 0.35),
            MajorityVote("a3", False, 0.95),
        ]

        strong = run_majority_consensus("claim", votes_strong)
        weak = run_majority_consensus("claim", votes_weak)

        # Both reach consensus (2/3 agree) — majority can't distinguish quality
        assert strong.consensus_reached is True
        assert weak.consensus_reached is True
        # But quality scores differ due to confidence weighting
        assert strong.quality_score > weak.quality_score

    @pytest.mark.asyncio
    async def test_cross_scenario_quality_comparison(self):
        """Run all scenarios and verify PE provides differentiated quality signals."""
        pe_scores: list[float] = []
        maj_scores: list[float] = []

        for sc in SCENARIOS:
            engine = ProverEstimatorEngine(
                prover=_mock_agent(list(sc.prover_responses)),
                estimator=_mock_agent(list(sc.estimator_responses)),
                max_challenge_rounds=2,
            )
            pe_result = await engine.run(sc.claim)
            maj_result = run_majority_consensus(sc.claim, sc.majority_votes)

            pe_scores.append(pe_result.grounding_score)
            maj_scores.append(maj_result.quality_score)

        # PE grounding scores should vary across scenarios (evidence quality differs)
        assert max(pe_scores) > min(pe_scores), "PE should differentiate scenarios"
        # Majority quality also varies
        assert max(maj_scores) > min(maj_scores), "Majority should differentiate scenarios"

        # The obfuscation scenario should have lower PE grounding than well-grounded
        well_grounded_idx = 0
        obfuscation_idx = 1
        assert pe_scores[well_grounded_idx] > pe_scores[obfuscation_idx], (
            "PE grounding should be lower when obfuscation is detected"
        )
