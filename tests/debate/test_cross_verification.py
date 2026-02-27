"""Tests for the Cross-Verification hallucination detection protocol."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aragora.debate.cross_verification import (
    CrossVerificationEngine,
    CrossVerificationResult,
    VerificationPass,
)


# ── Helpers ───────────────────────────────────────────────────────


def make_engine(
    responses: list[str] | None = None,
    grounding_threshold: float = 0.15,
) -> CrossVerificationEngine:
    """Create engine with mock verifier agent."""
    verifier = AsyncMock()
    if responses:
        verifier.generate = AsyncMock(side_effect=responses)
    return CrossVerificationEngine(
        verifier=verifier,
        grounding_threshold=grounding_threshold,
    )


# ── Parse Tests ───────────────────────────────────────────────────


class TestPassParsing:
    def test_parse_full_response(self):
        engine = make_engine()
        text = (
            "CONFIDENCE: 0.85\n"
            "VERDICT: supported\n"
            "REASONING: The evidence clearly shows X\n"
            "EVIDENCE_CITED: Study from 2024\n- Dataset analysis"
        )
        result = engine._parse_pass_response(text, "full_context")
        assert result.pass_type == "full_context"
        assert result.confidence == 0.85
        assert result.verdict == "supported"
        assert "evidence clearly shows" in result.reasoning
        assert len(result.evidence_cited) == 2

    def test_parse_no_evidence(self):
        engine = make_engine()
        text = (
            "CONFIDENCE: 0.4\n"
            "VERDICT: uncertain\n"
            "REASONING: Cannot determine without evidence\n"
            "EVIDENCE_CITED: none"
        )
        result = engine._parse_pass_response(text, "minimal_context")
        assert result.confidence == 0.4
        assert result.verdict == "uncertain"
        assert result.evidence_cited == []

    def test_parse_clamps_confidence(self):
        engine = make_engine()
        text = (
            "CONFIDENCE: 1.5\nVERDICT: supported\nREASONING: over-confident\nEVIDENCE_CITED: none"
        )
        result = engine._parse_pass_response(text, "full_context")
        assert result.confidence == 1.0

    def test_parse_missing_fields_defaults(self):
        engine = make_engine()
        text = "Some unstructured response without expected fields"
        result = engine._parse_pass_response(text, "adversarial_context")
        assert result.confidence == 0.5
        assert result.verdict == "uncertain"


# ── Hallucination Risk Tests ──────────────────────────────────────


class TestHallucinationRisk:
    def test_grounded_claim_low_risk(self):
        """Well-grounded: high with evidence, low without."""
        engine = make_engine()
        full = VerificationPass(
            pass_type="full_context",
            confidence=0.9,
            verdict="supported",
            evidence_cited=["study A", "dataset B"],
        )
        minimal = VerificationPass(
            pass_type="minimal_context",
            confidence=0.4,
            verdict="uncertain",
        )
        adversarial = VerificationPass(
            pass_type="adversarial_context",
            confidence=0.85,
            verdict="supported",
        )
        risk = engine._compute_hallucination_risk(full, minimal, adversarial)
        assert risk < 0.3

    def test_hallucinated_claim_high_risk(self):
        """Hallucinated: equally confident with or without evidence."""
        engine = make_engine()
        full = VerificationPass(
            pass_type="full_context",
            confidence=0.9,
            verdict="supported",
            evidence_cited=[],  # No evidence cited despite high confidence
        )
        minimal = VerificationPass(
            pass_type="minimal_context",
            confidence=0.88,  # Nearly same confidence without evidence
            verdict="supported",
        )
        adversarial = VerificationPass(
            pass_type="adversarial_context",
            confidence=0.5,  # Easily swayed by irrelevant context
            verdict="uncertain",
        )
        risk = engine._compute_hallucination_risk(full, minimal, adversarial)
        assert risk > 0.5

    def test_uncertain_claim_moderate_risk(self):
        """Uncertain: appropriately low confidence everywhere."""
        engine = make_engine()
        full = VerificationPass(
            pass_type="full_context",
            confidence=0.5,
            verdict="uncertain",
            evidence_cited=["weak source"],
        )
        minimal = VerificationPass(
            pass_type="minimal_context",
            confidence=0.45,
            verdict="uncertain",
        )
        adversarial = VerificationPass(
            pass_type="adversarial_context",
            confidence=0.48,
            verdict="uncertain",
        )
        risk = engine._compute_hallucination_risk(full, minimal, adversarial)
        # Low confidence + low delta = moderate risk but not extreme
        assert 0.2 < risk < 0.7


# ── Grounding Delta Tests ────────────────────────────────────────


class TestGroundingDelta:
    @pytest.mark.asyncio
    async def test_grounded_claim_detected(self):
        """Agent is much more confident with evidence → grounded."""
        engine = make_engine(
            responses=[
                # Full context: high confidence
                "CONFIDENCE: 0.92\nVERDICT: supported\n"
                "REASONING: Data clearly supports claim\n"
                "EVIDENCE_CITED: Study A\n- Dataset B",
                # Minimal context: low confidence
                "CONFIDENCE: 0.35\nVERDICT: uncertain\n"
                "REASONING: Cannot determine from claim alone\n"
                "EVIDENCE_CITED: none",
                # Adversarial context: stays high (not swayed)
                "CONFIDENCE: 0.88\nVERDICT: supported\n"
                "REASONING: Irrelevant context doesn't change assessment\n"
                "EVIDENCE_CITED: Original claim evidence",
            ],
        )
        result = await engine.verify(
            claim="X causes Y",
            context="Multiple RCTs show X→Y with p<0.01",
        )
        assert result.is_grounded is True
        assert result.grounding_delta > 0.15
        assert result.hallucination_risk < 0.4

    @pytest.mark.asyncio
    async def test_hallucinated_claim_detected(self):
        """Agent is equally confident with and without evidence → hallucinated."""
        engine = make_engine(
            responses=[
                # Full context: high confidence
                "CONFIDENCE: 0.90\nVERDICT: supported\n"
                "REASONING: This is well known\n"
                "EVIDENCE_CITED: none",
                # Minimal context: still high (suspicious!)
                "CONFIDENCE: 0.87\nVERDICT: supported\n"
                "REASONING: This is commonly accepted\n"
                "EVIDENCE_CITED: none",
                # Adversarial: drops (easily swayed)
                "CONFIDENCE: 0.45\nVERDICT: uncertain\n"
                "REASONING: Context suggests otherwise\n"
                "EVIDENCE_CITED: none",
            ],
        )
        result = await engine.verify(
            claim="Obscure fact that seems true",
            context="Some vague supporting context",
        )
        assert result.is_grounded is False
        assert result.grounding_delta < 0.15
        assert result.hallucination_risk > 0.4

    @pytest.mark.asyncio
    async def test_three_passes_present(self):
        """All three verification passes should be in the result."""
        engine = make_engine(
            responses=[
                "CONFIDENCE: 0.7\nVERDICT: supported\nREASONING: ok\nEVIDENCE_CITED: none",
                "CONFIDENCE: 0.3\nVERDICT: uncertain\nREASONING: no evidence\nEVIDENCE_CITED: none",
                "CONFIDENCE: 0.6\nVERDICT: uncertain\nREASONING: irrelevant\nEVIDENCE_CITED: none",
            ],
        )
        result = await engine.verify(claim="test", context="test context")
        assert len(result.passes) == 3
        pass_types = {p.pass_type for p in result.passes}
        assert pass_types == {"full_context", "minimal_context", "adversarial_context"}


# ── Explanation Tests ─────────────────────────────────────────────


class TestExplanation:
    def test_grounded_explanation(self):
        engine = make_engine()
        explanation = engine._generate_explanation(
            grounding_delta=0.4,
            adversarial_resistance=0.1,
            hallucination_risk=0.15,
            is_grounded=True,
        )
        assert "GROUNDED" in explanation
        assert "LOW" in explanation

    def test_ungrounded_high_risk_explanation(self):
        engine = make_engine()
        explanation = engine._generate_explanation(
            grounding_delta=0.02,
            adversarial_resistance=0.4,
            hallucination_risk=0.8,
            is_grounded=False,
        )
        assert "UNGROUNDED" in explanation
        assert "HIGH" in explanation

    def test_moderate_risk_explanation(self):
        engine = make_engine()
        explanation = engine._generate_explanation(
            grounding_delta=0.1,
            adversarial_resistance=0.2,
            hallucination_risk=0.5,
            is_grounded=False,
        )
        assert "MODERATE" in explanation


# ── Batch Verification Tests ─────────────────────────────────────


class TestBatchVerification:
    @pytest.mark.asyncio
    async def test_batch_verify(self):
        """Verify multiple claims in batch."""
        # 2 claims × 3 passes = 6 responses
        engine = make_engine(
            responses=[
                "CONFIDENCE: 0.9\nVERDICT: supported\nREASONING: yes\nEVIDENCE_CITED: data",
                "CONFIDENCE: 0.3\nVERDICT: uncertain\nREASONING: no\nEVIDENCE_CITED: none",
                "CONFIDENCE: 0.85\nVERDICT: supported\nREASONING: stable\nEVIDENCE_CITED: data",
                "CONFIDENCE: 0.6\nVERDICT: uncertain\nREASONING: maybe\nEVIDENCE_CITED: weak",
                "CONFIDENCE: 0.55\nVERDICT: uncertain\nREASONING: hmm\nEVIDENCE_CITED: none",
                "CONFIDENCE: 0.5\nVERDICT: uncertain\nREASONING: ok\nEVIDENCE_CITED: none",
            ],
        )
        results = await engine.verify_batch(
            [
                {"claim": "Claim A", "context": "Evidence A"},
                {"claim": "Claim B", "context": "Evidence B"},
            ]
        )
        assert len(results) == 2
        assert all(isinstance(r, CrossVerificationResult) for r in results)
        # First claim has high delta → grounded
        assert results[0].is_grounded is True


# ── Error Handling Tests ──────────────────────────────────────────


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_agent_failure_graceful(self):
        """If agent fails on a pass, result should still be usable."""
        verifier = AsyncMock()
        verifier.generate = AsyncMock(
            side_effect=[
                "CONFIDENCE: 0.8\nVERDICT: supported\nREASONING: ok\nEVIDENCE_CITED: data",
                RuntimeError("API timeout"),
                "CONFIDENCE: 0.7\nVERDICT: supported\nREASONING: ok\nEVIDENCE_CITED: data",
            ]
        )
        engine = CrossVerificationEngine(verifier=verifier)
        result = await engine.verify(claim="test", context="test context")
        # Should complete with 3 passes (failed one defaults to 0.5/uncertain)
        assert len(result.passes) == 3
        assert result.passes[1].confidence == 0.5
        assert result.passes[1].verdict == "uncertain"
