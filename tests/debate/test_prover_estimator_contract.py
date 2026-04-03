"""Contract tests for prover-estimator consensus mode.

Verifies that ``build_proof_from_prover_estimator`` produces a
ConsensusProof that satisfies the documented structural contract
regardless of the upstream ProverEstimatorResult implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from aragora.debate.consensus import (
    ConsensusProof,
    VoteType,
    build_proof_from_prover_estimator,
)


# ---------------------------------------------------------------------------
# Stub types that duck-type the real ProverEstimatorResult hierarchy
# ---------------------------------------------------------------------------


@dataclass
class _Subclaim:
    id: str
    text: str
    importance: float = 0.7
    evidence: str = ""


@dataclass
class _Estimate:
    subclaim_id: str
    probability: float


@dataclass
class _Challenge:
    subclaim_id: str
    challenge_type: str = "evidence"
    evidence: str = ""


@dataclass
class _PEResult:
    debate_id: str = "pe-contract-test"
    original_claim: str = "Test claim for contract"
    subclaims: list[_Subclaim] = field(default_factory=list)
    initial_estimates: list[_Estimate] = field(default_factory=list)
    final_estimates: list[_Estimate] = field(default_factory=list)
    challenges: list[_Challenge] = field(default_factory=list)
    overall_confidence: float = 0.75
    grounding_score: float = 0.8
    obfuscation_detected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def basic_pe_result() -> _PEResult:
    """A minimal valid prover-estimator result."""
    sc = _Subclaim(id="sc-1", text="Sub-claim A", importance=0.7, evidence="Some evidence")
    return _PEResult(
        subclaims=[sc],
        final_estimates=[_Estimate(subclaim_id="sc-1", probability=0.85)],
    )


@pytest.fixture()
def obfuscation_pe_result() -> _PEResult:
    """A result where obfuscation was detected."""
    sc = _Subclaim(id="sc-2", text="Suspicious claim", importance=0.6, evidence="Vague support")
    return _PEResult(
        subclaims=[sc],
        final_estimates=[_Estimate(subclaim_id="sc-2", probability=0.3)],
        challenges=[
            _Challenge(subclaim_id="sc-2", challenge_type="obfuscation", evidence="counter")
        ],
        overall_confidence=0.4,
        grounding_score=0.2,
        obfuscation_detected=True,
    )


@pytest.fixture()
def multi_subclaim_pe_result() -> _PEResult:
    """A result with multiple subclaims and challenges."""
    return _PEResult(
        subclaims=[
            _Subclaim(id="sc-a", text="Claim A", importance=0.9, evidence="Strong evidence A"),
            _Subclaim(id="sc-b", text="Claim B", importance=0.5, evidence="Moderate evidence B"),
            _Subclaim(id="sc-c", text="Claim C", importance=0.3),
        ],
        final_estimates=[
            _Estimate(subclaim_id="sc-a", probability=0.9),
            _Estimate(subclaim_id="sc-b", probability=0.4),
        ],
        challenges=[
            _Challenge(subclaim_id="sc-b", challenge_type="evidence", evidence="counter-evidence"),
        ],
        overall_confidence=0.65,
        grounding_score=0.7,
    )


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestProverEstimatorContract:
    """Contract: build_proof_from_prover_estimator must produce a
    ConsensusProof with the required structural invariants."""

    def test_returns_consensus_proof(self, basic_pe_result: _PEResult) -> None:
        proof = build_proof_from_prover_estimator(basic_pe_result)
        assert isinstance(proof, ConsensusProof)

    def test_metadata_consensus_mode(self, basic_pe_result: _PEResult) -> None:
        proof = build_proof_from_prover_estimator(basic_pe_result)
        assert proof.metadata["consensus_mode"] == "prover_estimator"

    def test_metadata_grounding_score(self, basic_pe_result: _PEResult) -> None:
        proof = build_proof_from_prover_estimator(basic_pe_result)
        assert proof.metadata["grounding_score"] == basic_pe_result.grounding_score

    def test_metadata_obfuscation_flag(self, basic_pe_result: _PEResult) -> None:
        proof = build_proof_from_prover_estimator(basic_pe_result)
        assert proof.metadata["obfuscation_detected"] is False

    def test_metadata_subclaim_count(self, basic_pe_result: _PEResult) -> None:
        proof = build_proof_from_prover_estimator(basic_pe_result)
        assert proof.metadata["subclaim_count"] == len(basic_pe_result.subclaims)

    def test_metadata_challenge_count(self, basic_pe_result: _PEResult) -> None:
        proof = build_proof_from_prover_estimator(basic_pe_result)
        assert proof.metadata["challenge_count"] == len(basic_pe_result.challenges)

    def test_debate_id_preserved(self, basic_pe_result: _PEResult) -> None:
        proof = build_proof_from_prover_estimator(basic_pe_result)
        assert proof.debate_id == basic_pe_result.debate_id

    def test_final_claim_is_original_claim(self, basic_pe_result: _PEResult) -> None:
        proof = build_proof_from_prover_estimator(basic_pe_result)
        assert proof.final_claim == basic_pe_result.original_claim

    def test_confidence_matches_overall(self, basic_pe_result: _PEResult) -> None:
        proof = build_proof_from_prover_estimator(basic_pe_result)
        assert proof.confidence == basic_pe_result.overall_confidence

    def test_prover_vote_always_agree(self, basic_pe_result: _PEResult) -> None:
        proof = build_proof_from_prover_estimator(basic_pe_result)
        prover_votes = [v for v in proof.votes if v.agent == "prover"]
        assert len(prover_votes) == 1
        assert prover_votes[0].vote == VoteType.AGREE

    def test_estimator_vote_agree_when_confident(self, basic_pe_result: _PEResult) -> None:
        proof = build_proof_from_prover_estimator(basic_pe_result)
        estimator_votes = [v for v in proof.votes if v.agent == "estimator"]
        assert len(estimator_votes) == 1
        assert estimator_votes[0].vote == VoteType.AGREE

    def test_estimator_vote_conditional_when_low_confidence(
        self, obfuscation_pe_result: _PEResult
    ) -> None:
        proof = build_proof_from_prover_estimator(obfuscation_pe_result)
        estimator_votes = [v for v in proof.votes if v.agent == "estimator"]
        assert len(estimator_votes) == 1
        assert estimator_votes[0].vote == VoteType.CONDITIONAL

    def test_consensus_reached_when_confident_no_obfuscation(
        self, basic_pe_result: _PEResult
    ) -> None:
        proof = build_proof_from_prover_estimator(basic_pe_result)
        assert proof.consensus_reached is True

    def test_consensus_not_reached_when_obfuscation(self, obfuscation_pe_result: _PEResult) -> None:
        proof = build_proof_from_prover_estimator(obfuscation_pe_result)
        assert proof.consensus_reached is False

    def test_obfuscation_creates_dissent(self, obfuscation_pe_result: _PEResult) -> None:
        proof = build_proof_from_prover_estimator(obfuscation_pe_result)
        assert len(proof.dissents) >= 1
        assert proof.dissents[0].agent == "estimator"
        assert proof.dissents[0].dissent_type == "procedural"

    def test_no_dissent_without_obfuscation(self, basic_pe_result: _PEResult) -> None:
        proof = build_proof_from_prover_estimator(basic_pe_result)
        assert len(proof.dissents) == 0

    def test_claims_match_subclaims(self, multi_subclaim_pe_result: _PEResult) -> None:
        proof = build_proof_from_prover_estimator(multi_subclaim_pe_result)
        assert len(proof.claims) == len(multi_subclaim_pe_result.subclaims)

    def test_challenges_become_tensions(self, multi_subclaim_pe_result: _PEResult) -> None:
        proof = build_proof_from_prover_estimator(multi_subclaim_pe_result)
        assert len(proof.unresolved_tensions) == len(multi_subclaim_pe_result.challenges)

    def test_checksum_is_stable(self, basic_pe_result: _PEResult) -> None:
        proof = build_proof_from_prover_estimator(basic_pe_result)
        c1 = proof.checksum
        c2 = proof.checksum
        assert c1 == c2
        assert len(c1) == 16  # SHA-256 truncated to 16 hex chars

    def test_evidence_chain_populated(self, basic_pe_result: _PEResult) -> None:
        proof = build_proof_from_prover_estimator(basic_pe_result)
        assert len(proof.evidence_chain) > 0

    def test_reasoning_summary_mentions_protocol(self, basic_pe_result: _PEResult) -> None:
        proof = build_proof_from_prover_estimator(basic_pe_result)
        assert "Prover-Estimator" in proof.reasoning_summary

    def test_empty_result_produces_valid_proof(self) -> None:
        """Even a minimal/empty PE result should produce a valid proof."""
        proof = build_proof_from_prover_estimator(_PEResult())
        assert isinstance(proof, ConsensusProof)
        assert proof.metadata["consensus_mode"] == "prover_estimator"
        assert proof.consensus_reached is True  # 0.75 >= 0.5, no obfuscation

    def test_custom_metadata_forwarded(self) -> None:
        result = _PEResult(metadata={"custom_key": "custom_value"})
        proof = build_proof_from_prover_estimator(result)
        assert proof.metadata["custom_key"] == "custom_value"

    def test_rounds_equal_challenge_count(self, multi_subclaim_pe_result: _PEResult) -> None:
        proof = build_proof_from_prover_estimator(multi_subclaim_pe_result)
        assert proof.rounds_to_consensus == len(multi_subclaim_pe_result.challenges)
