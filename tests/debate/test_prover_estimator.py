"""Tests for the Prover-Estimator truth-seeking debate protocol."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock

import pytest

from aragora.debate.prover_estimator import (
    Challenge,
    ProverEstimatorEngine,
    Subclaim,
    SubclaimEstimate,
)


# ── Fixtures ──────────────────────────────────────────────────────


def make_engine(
    prover_responses: list[str] | None = None,
    estimator_responses: list[str] | None = None,
    max_challenge_rounds: int = 2,
    context: str = "",
) -> ProverEstimatorEngine:
    """Create engine with mock agents."""
    prover = AsyncMock()
    estimator = AsyncMock()

    if prover_responses:
        prover.generate = AsyncMock(side_effect=prover_responses)
    if estimator_responses:
        estimator.generate = AsyncMock(side_effect=estimator_responses)

    return ProverEstimatorEngine(
        prover=prover,
        estimator=estimator,
        max_challenge_rounds=max_challenge_rounds,
        context=context,
    )


# ── Subclaim Parsing ──────────────────────────────────────────────


class TestParseSubclaims:
    def test_parse_single_subclaim(self):
        engine = make_engine()
        text = (
            "SUBCLAIM [A]: The Earth orbits the Sun\n"
            "IMPORTANCE: 0.9\n"
            "EVIDENCE: Kepler's laws and centuries of astronomical observation\n"
            "DEPENDS_ON: none\n"
        )
        result = engine._parse_subclaims(text)
        assert len(result) == 1
        assert result[0].id == "A"
        assert result[0].text == "The Earth orbits the Sun"
        assert result[0].importance == 0.9
        assert "Kepler" in result[0].evidence
        assert result[0].depends_on == []

    def test_parse_multiple_subclaims(self):
        engine = make_engine()
        text = (
            "SUBCLAIM [A]: First claim\n"
            "IMPORTANCE: 0.8\n"
            "EVIDENCE: Evidence A\n"
            "DEPENDS_ON: none\n"
            "\n"
            "SUBCLAIM [B]: Second claim\n"
            "IMPORTANCE: 0.6\n"
            "EVIDENCE: Evidence B\n"
            "DEPENDS_ON: A\n"
        )
        result = engine._parse_subclaims(text)
        assert len(result) == 2
        assert result[0].id == "A"
        assert result[1].id == "B"
        assert result[1].depends_on == ["A"]

    def test_parse_importance_clamped(self):
        engine = make_engine()
        text = "SUBCLAIM [X]: Overclaimed\nIMPORTANCE: 1.5\nEVIDENCE: none\nDEPENDS_ON: none\n"
        result = engine._parse_subclaims(text)
        assert len(result) == 1
        assert result[0].importance == 1.0


# ── Estimate Parsing ──────────────────────────────────────────────


class TestParseEstimates:
    def test_parse_estimate_no_obfuscation(self):
        engine = make_engine()
        text = (
            "ESTIMATE [A]:\n"
            "PROBABILITY: 0.85\n"
            "REASONING: Strong evidence supports this\n"
            "CONFIDENCE: 0.7\n"
            "OBFUSCATION: NO\n"
        )
        result = engine._parse_estimates(text)
        assert len(result) == 1
        assert result[0].subclaim_id == "A"
        assert result[0].probability == 0.85
        assert result[0].confidence_in_estimate == 0.7
        assert result[0].obfuscation_flag is False

    def test_parse_estimate_with_obfuscation(self):
        engine = make_engine()
        text = (
            "ESTIMATE [B]:\n"
            "PROBABILITY: 0.3\n"
            "REASONING: Weak evidence, mostly rhetorical\n"
            "CONFIDENCE: 0.6\n"
            "OBFUSCATION: YES\n"
            "OBFUSCATION_REASON: Uses appeal to authority without data\n"
        )
        result = engine._parse_estimates(text)
        assert len(result) == 1
        assert result[0].obfuscation_flag is True
        assert "authority" in result[0].obfuscation_reason


# ── Challenge Parsing ─────────────────────────────────────────────


class TestParseChallenges:
    def test_parse_evidence_challenge(self):
        engine = make_engine()
        text = (
            "CHALLENGE [A]:\n"
            "TYPE: evidence\n"
            "EVIDENCE: NASA data from 2024 confirms orbital parameters\n"
            "REVISED_PROBABILITY: 0.95\n"
        )
        result = engine._parse_challenges(text)
        assert len(result) == 1
        assert result[0].subclaim_id == "A"
        assert result[0].challenge_type == "evidence"
        assert result[0].revised_probability == 0.95

    def test_parse_methodology_challenge(self):
        engine = make_engine()
        text = (
            "CHALLENGE [B]:\n"
            "TYPE: methodology\n"
            "EVIDENCE: The sample size was too small for statistical significance\n"
            "REVISED_PROBABILITY: 0.4\n"
        )
        result = engine._parse_challenges(text)
        assert len(result) == 1
        assert result[0].challenge_type == "methodology"

    def test_no_challenges_returns_empty(self):
        engine = make_engine()
        text = "All estimates seem fair. No challenges needed."
        result = engine._parse_challenges(text)
        assert len(result) == 0


# ── Re-estimate Parsing ──────────────────────────────────────────


class TestParseReestimates:
    def test_parse_reestimate(self):
        engine = make_engine()
        text = (
            "REESTIMATE [A]:\n"
            "PROBABILITY: 0.92\n"
            "REASONING: The NASA data is compelling, revising upward\n"
            "CONFIDENCE: 0.85\n"
            "OBFUSCATION: NO\n"
        )
        result = engine._parse_reestimates(text)
        assert len(result) == 1
        assert result[0].subclaim_id == "A"
        assert result[0].probability == 0.92
        assert result[0].confidence_in_estimate == 0.85

    def test_parse_reestimate_detects_obfuscation(self):
        engine = make_engine()
        text = (
            "REESTIMATE [C]:\n"
            "PROBABILITY: 0.35\n"
            "REASONING: Challenge used emotional language, not data\n"
            "CONFIDENCE: 0.7\n"
            "OBFUSCATION: YES\n"
            "OBFUSCATION_REASON: Challenge relied on fear-mongering rather than evidence\n"
        )
        result = engine._parse_reestimates(text)
        assert len(result) == 1
        assert result[0].obfuscation_flag is True
        assert "fear" in result[0].obfuscation_reason.lower()


# ── Aggregation ───────────────────────────────────────────────────


class TestAggregation:
    def test_geometric_mean_all_high(self):
        engine = make_engine()
        subclaims = [
            Subclaim(id="A", text="a", importance=1.0),
            Subclaim(id="B", text="b", importance=1.0),
        ]
        estimates = [
            SubclaimEstimate(subclaim_id="A", probability=0.9),
            SubclaimEstimate(subclaim_id="B", probability=0.9),
        ]
        result = engine._aggregate_confidence(subclaims, estimates)
        assert abs(result - 0.9) < 0.01

    def test_geometric_mean_one_low_tanks_result(self):
        """A single low-probability critical subclaim should tank confidence."""
        engine = make_engine()
        subclaims = [
            Subclaim(id="A", text="a", importance=1.0),
            Subclaim(id="B", text="b", importance=1.0),
        ]
        estimates = [
            SubclaimEstimate(subclaim_id="A", probability=0.9),
            SubclaimEstimate(subclaim_id="B", probability=0.1),
        ]
        result = engine._aggregate_confidence(subclaims, estimates)
        # Geometric mean of 0.9 and 0.1 = sqrt(0.09) ≈ 0.3
        assert result < 0.35
        assert result > 0.25

    def test_importance_weighting(self):
        """Low-importance subclaims should have less influence."""
        engine = make_engine()
        subclaims = [
            Subclaim(id="A", text="a", importance=1.0),  # Critical
            Subclaim(id="B", text="b", importance=0.1),  # Minor
        ]
        estimates = [
            SubclaimEstimate(subclaim_id="A", probability=0.9),
            SubclaimEstimate(subclaim_id="B", probability=0.1),  # Low but unimportant
        ]
        result = engine._aggregate_confidence(subclaims, estimates)
        # B's low probability should barely affect the result due to low importance
        assert result > 0.7

    def test_empty_estimates(self):
        engine = make_engine()
        result = engine._aggregate_confidence([], [])
        assert result == 0.0

    def test_grounding_score(self):
        engine = make_engine()
        subclaims = [
            Subclaim(id="A", text="a", importance=0.9, evidence="solid data"),
            Subclaim(id="B", text="b", importance=0.5, evidence=""),
        ]
        estimates = [
            SubclaimEstimate(subclaim_id="A", probability=0.8, obfuscation_flag=False),
            SubclaimEstimate(subclaim_id="B", probability=0.6, obfuscation_flag=False),
        ]
        challenges = [
            Challenge(subclaim_id="A", challenge_type="evidence", evidence="data"),
        ]
        score = engine._compute_grounding_score(subclaims, estimates, challenges)
        # 50% evidence (1/2), 100% no obfuscation, 100% evidence challenges
        assert 0.5 < score < 1.0


# ── Full Pipeline ─────────────────────────────────────────────────


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_full_run_no_challenges(self):
        """Test full pipeline when estimator agrees with prover."""
        decompose_response = (
            "SUBCLAIM [A]: Water boils at 100C at sea level\n"
            "IMPORTANCE: 0.9\n"
            "EVIDENCE: Basic thermodynamics\n"
            "DEPENDS_ON: none\n"
        )
        estimate_response = (
            "ESTIMATE [A]:\n"
            "PROBABILITY: 0.95\n"
            "REASONING: Well-established physics\n"
            "CONFIDENCE: 0.9\n"
            "OBFUSCATION: NO\n"
        )
        # Prover sees fair estimate, no challenges
        no_challenge = "All estimates are fair. No challenges needed."

        engine = make_engine(
            prover_responses=[decompose_response, no_challenge],
            estimator_responses=[estimate_response],
        )

        result = await engine.run("Water boils at 100C at sea level")
        assert len(result.subclaims) == 1
        assert result.overall_confidence > 0.9
        assert not result.obfuscation_detected
        assert result.grounding_score > 0.5

    @pytest.mark.asyncio
    async def test_full_run_with_challenge(self):
        """Test full pipeline with one round of challenge."""
        decompose_response = (
            "SUBCLAIM [A]: AI will surpass human intelligence by 2030\n"
            "IMPORTANCE: 0.9\n"
            "EVIDENCE: Current scaling trends\n"
            "DEPENDS_ON: none\n"
            "\n"
            "SUBCLAIM [B]: Compute costs will continue declining\n"
            "IMPORTANCE: 0.7\n"
            "EVIDENCE: Moore's law successors\n"
            "DEPENDS_ON: none\n"
        )
        estimate_response = (
            "ESTIMATE [A]:\n"
            "PROBABILITY: 0.3\n"
            "REASONING: Highly uncertain timeline\n"
            "CONFIDENCE: 0.4\n"
            "OBFUSCATION: NO\n"
            "\n"
            "ESTIMATE [B]:\n"
            "PROBABILITY: 0.7\n"
            "REASONING: Historical trend supports this\n"
            "CONFIDENCE: 0.6\n"
            "OBFUSCATION: NO\n"
        )
        challenge_response = (
            "CHALLENGE [A]:\n"
            "TYPE: evidence\n"
            "EVIDENCE: Recent benchmarks show accelerating capability gains\n"
            "REVISED_PROBABILITY: 0.5\n"
        )
        reestimate_response = (
            "REESTIMATE [A]:\n"
            "PROBABILITY: 0.4\n"
            "REASONING: Benchmark data is noted but timeline still uncertain\n"
            "CONFIDENCE: 0.5\n"
            "OBFUSCATION: NO\n"
        )
        # After reestimate, prover has no more challenges
        no_more_challenges = "No further challenges."

        engine = make_engine(
            prover_responses=[decompose_response, challenge_response, no_more_challenges],
            estimator_responses=[estimate_response, reestimate_response],
            max_challenge_rounds=2,
        )

        result = await engine.run("AI will surpass human intelligence by 2030")
        assert len(result.subclaims) == 2
        assert len(result.challenges) == 1
        assert result.challenges[0].challenge_type == "evidence"
        assert result.overall_confidence > 0.0
        assert not result.obfuscation_detected

    @pytest.mark.asyncio
    async def test_empty_decomposition(self):
        """Test graceful handling when prover fails to decompose."""
        engine = make_engine(
            prover_responses=["I can't decompose this claim."],
            estimator_responses=[],
        )

        result = await engine.run("???")
        assert len(result.subclaims) == 0
        assert result.overall_confidence == 0.0
        assert "error" in result.metadata
