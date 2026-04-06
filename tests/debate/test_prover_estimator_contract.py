"""Contract tests for Prover-Estimator ↔ Consensus integration.

Validates that ProverEstimatorResult objects produced by the engine
are correctly transformed into ConsensusProof artifacts by
build_proof_from_prover_estimator.
"""

from __future__ import annotations

import math

import pytest

from aragora.debate.consensus import build_proof_from_prover_estimator
from aragora.debate.prover_estimator import (
    Challenge,
    ProverEstimatorEngine,
    ProverEstimatorResult,
    Subclaim,
    SubclaimEstimate,
)


# ── Helpers ──────────────────────────────────────────────────────


def _make_result(
    *,
    subclaims: list[Subclaim] | None = None,
    initial_estimates: list[SubclaimEstimate] | None = None,
    final_estimates: list[SubclaimEstimate] | None = None,
    challenges: list[Challenge] | None = None,
    overall_confidence: float = 0.75,
    grounding_score: float = 0.8,
    obfuscation_detected: bool = False,
    metadata: dict | None = None,
) -> ProverEstimatorResult:
    """Build a minimal ProverEstimatorResult for contract testing."""
    sc = (
        subclaims
        if subclaims is not None
        else [
            Subclaim(id="A", text="Earth orbits Sun", importance=0.9, evidence="Kepler's laws"),
            Subclaim(
                id="B", text="Orbit is elliptical", importance=0.7, evidence="Observation data"
            ),
        ]
    )
    ie = (
        initial_estimates
        if initial_estimates is not None
        else [
            SubclaimEstimate(
                subclaim_id="A",
                probability=0.95,
                reasoning="Well established",
                confidence_in_estimate=0.9,
            ),
            SubclaimEstimate(
                subclaim_id="B",
                probability=0.85,
                reasoning="Strong evidence",
                confidence_in_estimate=0.8,
            ),
        ]
    )
    fe = final_estimates if final_estimates is not None else ie
    ch = challenges if challenges is not None else []
    return ProverEstimatorResult(
        original_claim="The Earth orbits the Sun in an elliptical path",
        subclaims=sc,
        initial_estimates=ie,
        challenges=ch,
        final_estimates=fe,
        overall_confidence=overall_confidence,
        grounding_score=grounding_score,
        obfuscation_detected=obfuscation_detected,
        metadata=metadata or {},
    )


# ── Proof structure contract ─────────────────────────────────────


class TestProofStructure:
    """build_proof_from_prover_estimator must return a well-formed ConsensusProof."""

    def test_proof_has_required_fields(self):
        result = _make_result()
        proof = build_proof_from_prover_estimator(result)

        assert proof.final_claim == result.original_claim
        assert proof.confidence == result.overall_confidence
        assert proof.consensus_reached is True  # confidence >= 0.5, no obfuscation
        assert proof.debate_id == "pe-debate"  # default when no debate_id attr
        assert proof.proof_id  # non-empty

    def test_metadata_carries_pe_fields(self):
        result = _make_result(grounding_score=0.65, metadata={"custom": "value"})
        proof = build_proof_from_prover_estimator(result)

        assert proof.metadata["consensus_mode"] == "prover_estimator"
        assert proof.metadata["grounding_score"] == 0.65
        assert proof.metadata["obfuscation_detected"] is False
        assert proof.metadata["subclaim_count"] == 2
        assert proof.metadata["challenge_count"] == 0
        assert proof.metadata["custom"] == "value"

    def test_empty_subclaims_produce_valid_proof(self):
        result = _make_result(
            subclaims=[],
            initial_estimates=[],
            final_estimates=[],
            overall_confidence=0.0,
        )
        proof = build_proof_from_prover_estimator(result)

        assert proof.confidence == 0.0
        assert proof.consensus_reached is False
        assert proof.metadata["subclaim_count"] == 0


# ── Vote mapping contract ────────────────────────────────────────


class TestVoteMapping:
    """Prover and estimator votes must reflect protocol semantics."""

    def test_high_confidence_yields_two_agree_votes(self):
        result = _make_result(overall_confidence=0.8)
        proof = build_proof_from_prover_estimator(result)

        assert len(proof.votes) == 2
        prover_vote = next(v for v in proof.votes if v.agent == "prover")
        estimator_vote = next(v for v in proof.votes if v.agent == "estimator")

        assert prover_vote.vote.value == "agree"
        assert estimator_vote.vote.value == "agree"

    def test_low_confidence_yields_conditional_estimator(self):
        result = _make_result(overall_confidence=0.3)
        proof = build_proof_from_prover_estimator(result)

        estimator_vote = next(v for v in proof.votes if v.agent == "estimator")
        assert estimator_vote.vote.value == "conditional"

    def test_obfuscation_triggers_dissent(self):
        result = _make_result(obfuscation_detected=True, overall_confidence=0.6)
        proof = build_proof_from_prover_estimator(result)

        assert proof.consensus_reached is False  # obfuscation blocks consensus
        assert len(proof.dissents) >= 1
        assert any("obfuscation" in d.reasons[0].lower() for d in proof.dissents)


# ── Evidence chain contract ──────────────────────────────────────


class TestEvidenceChain:
    """Subclaim evidence and estimator probabilities must map to proof evidence."""

    def test_subclaims_produce_evidence_entries(self):
        result = _make_result()
        proof = build_proof_from_prover_estimator(result)

        # Each subclaim with evidence generates a prover evidence entry
        prover_evidence = [e for e in proof.evidence_chain if e.source == "prover"]
        assert len(prover_evidence) == 2  # both subclaims have evidence

    def test_estimates_produce_data_evidence(self):
        result = _make_result()
        proof = build_proof_from_prover_estimator(result)

        estimator_evidence = [e for e in proof.evidence_chain if e.source == "estimator"]
        assert len(estimator_evidence) == 2  # one per subclaim estimate

    def test_subclaim_without_evidence_skipped(self):
        sc = [Subclaim(id="X", text="No evidence claim", importance=0.5, evidence="")]
        est = [SubclaimEstimate(subclaim_id="X", probability=0.6, reasoning="Guess")]
        result = _make_result(subclaims=sc, initial_estimates=est, final_estimates=est)
        proof = build_proof_from_prover_estimator(result)

        prover_evidence = [e for e in proof.evidence_chain if e.source == "prover"]
        assert len(prover_evidence) == 0  # empty evidence → no entry


# ── Challenge → tension mapping ──────────────────────────────────


class TestChallengeTensionMapping:
    """Challenges must map to unresolved tensions in the proof."""

    def test_challenges_become_tensions(self):
        challenges = [
            Challenge(
                subclaim_id="A",
                challenge_type="evidence",
                evidence="New data",
                revised_probability=0.99,
            ),
            Challenge(
                subclaim_id="B",
                challenge_type="methodology",
                evidence="Flawed method",
                revised_probability=0.5,
            ),
        ]
        result = _make_result(challenges=challenges)
        proof = build_proof_from_prover_estimator(result)

        assert len(proof.unresolved_tensions) == 2
        assert proof.metadata["challenge_count"] == 2

    def test_no_challenges_no_tensions(self):
        result = _make_result(challenges=[])
        proof = build_proof_from_prover_estimator(result)

        assert len(proof.unresolved_tensions) == 0


# ── Aggregation contract ─────────────────────────────────────────


class TestAggregationContract:
    """Engine aggregation must use importance-weighted geometric mean."""

    def test_geometric_mean_basic(self):
        engine = ProverEstimatorEngine.__new__(ProverEstimatorEngine)
        subclaims = [
            Subclaim(id="A", text="Claim A", importance=1.0),
            Subclaim(id="B", text="Claim B", importance=1.0),
        ]
        estimates = [
            SubclaimEstimate(subclaim_id="A", probability=0.8),
            SubclaimEstimate(subclaim_id="B", probability=0.8),
        ]
        result = engine._aggregate_confidence(subclaims, estimates)
        expected = math.exp((1.0 * math.log(0.8) + 1.0 * math.log(0.8)) / 2.0)
        assert abs(result - expected) < 1e-9
        assert abs(result - 0.8) < 1e-9  # equal probs → same as individual

    def test_low_critical_subclaim_tanks_confidence(self):
        """A single low-probability high-importance subclaim must drag down overall confidence."""
        engine = ProverEstimatorEngine.__new__(ProverEstimatorEngine)
        subclaims = [
            Subclaim(id="A", text="Strong", importance=0.9),
            Subclaim(id="B", text="Weak critical", importance=0.9),
        ]
        estimates = [
            SubclaimEstimate(subclaim_id="A", probability=0.95),
            SubclaimEstimate(subclaim_id="B", probability=0.1),
        ]
        result = engine._aggregate_confidence(subclaims, estimates)
        # Geometric mean of 0.95 and 0.1 (equal weight) ≈ 0.308
        assert result < 0.5  # must be significantly below the arithmetic mean

    def test_empty_estimates_returns_zero(self):
        engine = ProverEstimatorEngine.__new__(ProverEstimatorEngine)
        result = engine._aggregate_confidence([], [])
        assert result == 0.0

    def test_grounding_score_factors(self):
        engine = ProverEstimatorEngine.__new__(ProverEstimatorEngine)
        subclaims = [
            Subclaim(id="A", text="Claim", importance=0.8, evidence="Has evidence"),
        ]
        estimates = [SubclaimEstimate(subclaim_id="A", probability=0.9)]
        challenges = [
            Challenge(subclaim_id="A", challenge_type="evidence", evidence="data"),
        ]
        score = engine._compute_grounding_score(subclaims, estimates, challenges)
        # evidence_ratio=1.0, obfuscation_ratio=1.0, challenge_quality=1.0
        assert abs(score - 1.0) < 1e-9


# ── Round-trip contract ──────────────────────────────────────────


class TestRoundTrip:
    """Full result → proof → verify round-trip preserves semantics."""

    def test_confidence_preserved(self):
        for conf in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = _make_result(overall_confidence=conf)
            proof = build_proof_from_prover_estimator(result)
            assert proof.confidence == conf

    def test_claim_text_preserved(self):
        result = _make_result()
        result.original_claim = "Custom claim text with special chars: <>&"
        proof = build_proof_from_prover_estimator(result)
        assert proof.final_claim == "Custom claim text with special chars: <>&"
