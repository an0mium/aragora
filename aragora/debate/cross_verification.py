"""
Cross-Verification Phase for hallucination and grounding detection.

Implements a three-pass verification protocol:
1. Full Context Pass: Agent evaluates claim with all available evidence
2. Minimal Context Pass: Agent evaluates same claim with only the claim text
3. Adversarial Context Pass: Agent evaluates claim with irrelevant/misleading context

If an agent's confidence is equally high across all three passes, it may be
generating from training data (hallucinating) rather than reasoning from
evidence. Genuine grounding shows measurable confidence delta between
full-context and minimal-context evaluations.

Key insight: A well-grounded agent should be LESS confident without evidence
and MORE confident with relevant evidence. If confidence is invariant to
context, the agent is likely confabulating.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class VerifiableAgent(Protocol):
    """Agent that can generate responses for verification passes."""

    async def generate(self, prompt: str) -> str: ...


@dataclass
class VerificationPass:
    """Result of a single verification pass."""

    pass_type: str  # "full_context", "minimal_context", "adversarial_context"
    confidence: float  # 0.0-1.0
    reasoning: str = ""
    verdict: str = ""  # "supported", "unsupported", "uncertain"
    evidence_cited: list[str] = field(default_factory=list)


@dataclass
class CrossVerificationResult:
    """Full result of cross-verification analysis."""

    claim: str
    passes: list[VerificationPass]
    grounding_delta: float  # full_context confidence - minimal_context confidence
    adversarial_resistance: float  # how much adversarial context shifted confidence
    is_grounded: bool  # whether claim appears genuinely evidence-based
    hallucination_risk: float  # 0.0-1.0, higher = more likely hallucinated
    explanation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Prompt Templates ──────────────────────────────────────────────


FULL_CONTEXT_PROMPT = """Evaluate the following claim given the provided evidence and context.

CLAIM: {claim}

EVIDENCE AND CONTEXT:
{context}

Assess whether this claim is supported by the evidence. Respond with:
CONFIDENCE: [0.0-1.0] (how confident you are the claim is true)
VERDICT: [supported | unsupported | uncertain]
REASONING: [your analysis of the evidence]
EVIDENCE_CITED: [list the specific pieces of evidence you relied on, one per line]"""


MINIMAL_CONTEXT_PROMPT = """Evaluate the following claim based solely on the claim itself.
Do NOT use any prior knowledge or training data — only evaluate what you can
determine from the claim text alone.

CLAIM: {claim}

Assess whether this claim is self-evidently true, false, or requires evidence.
CONFIDENCE: [0.0-1.0] (how confident you are the claim is true WITHOUT external evidence)
VERDICT: [supported | unsupported | uncertain]
REASONING: [explain what you can determine from the claim alone]
EVIDENCE_CITED: [none — you have no evidence]"""


ADVERSARIAL_CONTEXT_PROMPT = """Evaluate the following claim given the provided context.

CLAIM: {claim}

CONTEXT:
{adversarial_context}

Note: The context above may or may not be relevant to the claim.
Assess whether this claim is supported. Respond with:
CONFIDENCE: [0.0-1.0] (how confident you are the claim is true)
VERDICT: [supported | unsupported | uncertain]
REASONING: [your analysis]
EVIDENCE_CITED: [list any evidence you relied on, one per line]"""


# ── Engine ────────────────────────────────────────────────────────


class CrossVerificationEngine:
    """Orchestrates three-pass cross-verification for hallucination detection.

    The protocol detects whether an agent's claim assessment is genuinely
    grounded in evidence or generated from training data patterns.

    Key metrics:
    - grounding_delta: Confidence difference between full and minimal context.
      Higher delta = more evidence-dependent = more grounded.
    - adversarial_resistance: How much irrelevant context shifts confidence.
      High resistance = agent properly ignores irrelevant info.
    - hallucination_risk: Composite score. Low delta + low resistance = high risk.

    Args:
        verifier: Agent performing the verification passes
        adversarial_contexts: Pool of irrelevant contexts for adversarial pass
        grounding_threshold: Minimum delta to consider claim grounded
    """

    def __init__(
        self,
        verifier: VerifiableAgent,
        adversarial_contexts: list[str] | None = None,
        grounding_threshold: float = 0.15,
    ):
        self.verifier = verifier
        self.adversarial_contexts = adversarial_contexts or [_DEFAULT_ADVERSARIAL_CONTEXT]
        self.grounding_threshold = grounding_threshold

    async def verify(
        self,
        claim: str,
        context: str,
        adversarial_context: str | None = None,
    ) -> CrossVerificationResult:
        """Run three-pass cross-verification on a claim.

        Args:
            claim: The claim to verify
            context: Relevant evidence/context for the claim
            adversarial_context: Irrelevant context (uses default if None)

        Returns:
            CrossVerificationResult with grounding analysis
        """
        logger.info("Starting cross-verification for: %s", claim[:100])

        # Pass 1: Full context
        full_pass = await self._run_pass(
            claim=claim,
            pass_type="full_context",
            prompt=FULL_CONTEXT_PROMPT.format(claim=claim, context=context),
        )

        # Pass 2: Minimal context
        minimal_pass = await self._run_pass(
            claim=claim,
            pass_type="minimal_context",
            prompt=MINIMAL_CONTEXT_PROMPT.format(claim=claim),
        )

        # Pass 3: Adversarial context
        adv_ctx = adversarial_context or self.adversarial_contexts[0]
        adversarial_pass = await self._run_pass(
            claim=claim,
            pass_type="adversarial_context",
            prompt=ADVERSARIAL_CONTEXT_PROMPT.format(claim=claim, adversarial_context=adv_ctx),
        )

        # Compute metrics
        grounding_delta = full_pass.confidence - minimal_pass.confidence
        adversarial_resistance = abs(full_pass.confidence - adversarial_pass.confidence)

        hallucination_risk = self._compute_hallucination_risk(
            full_pass, minimal_pass, adversarial_pass
        )
        is_grounded = grounding_delta >= self.grounding_threshold

        explanation = self._generate_explanation(
            grounding_delta, adversarial_resistance, hallucination_risk, is_grounded
        )

        result = CrossVerificationResult(
            claim=claim,
            passes=[full_pass, minimal_pass, adversarial_pass],
            grounding_delta=grounding_delta,
            adversarial_resistance=adversarial_resistance,
            is_grounded=is_grounded,
            hallucination_risk=hallucination_risk,
            explanation=explanation,
            metadata={
                "grounding_threshold": self.grounding_threshold,
                "full_confidence": full_pass.confidence,
                "minimal_confidence": minimal_pass.confidence,
                "adversarial_confidence": adversarial_pass.confidence,
            },
        )

        logger.info(
            "Cross-verification complete: grounded=%s risk=%.3f delta=%.3f",
            is_grounded,
            hallucination_risk,
            grounding_delta,
        )
        return result

    async def verify_batch(
        self,
        claims: list[dict[str, str]],
    ) -> list[CrossVerificationResult]:
        """Verify multiple claims. Each dict must have 'claim' and 'context' keys."""
        results = []
        for item in claims:
            result = await self.verify(
                claim=item["claim"],
                context=item.get("context", ""),
                adversarial_context=item.get("adversarial_context"),
            )
            results.append(result)
        return results

    # ── Internal ──────────────────────────────────────────────────

    async def _run_pass(
        self,
        claim: str,
        pass_type: str,
        prompt: str,
    ) -> VerificationPass:
        """Run a single verification pass and parse the response."""
        try:
            response = await self.verifier.generate(prompt)
            return self._parse_pass_response(response, pass_type)
        except Exception:
            logger.warning("Verification pass %s failed for claim: %s", pass_type, claim[:80])
            return VerificationPass(
                pass_type=pass_type,
                confidence=0.5,
                reasoning="Verification pass failed",
                verdict="uncertain",
            )

    def _parse_pass_response(self, text: str, pass_type: str) -> VerificationPass:
        """Parse a verification pass response."""
        import re

        confidence = 0.5
        verdict = "uncertain"
        reasoning = ""
        evidence_cited: list[str] = []

        # Parse confidence
        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", text, re.IGNORECASE)
        if conf_match:
            confidence = min(max(float(conf_match.group(1)), 0.0), 1.0)

        # Parse verdict
        verdict_match = re.search(
            r"VERDICT:\s*(supported|unsupported|uncertain)",
            text,
            re.IGNORECASE,
        )
        if verdict_match:
            verdict = verdict_match.group(1).lower()

        # Parse reasoning
        reasoning_match = re.search(
            r"REASONING:\s*(.+?)(?=\nEVIDENCE_CITED:|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Parse evidence cited
        evidence_match = re.search(
            r"EVIDENCE_CITED:\s*(.+?)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if evidence_match:
            raw_evidence = evidence_match.group(1).strip()
            if raw_evidence.lower() not in ("none", "n/a", ""):
                evidence_cited = [
                    line.strip().lstrip("- ")
                    for line in raw_evidence.split("\n")
                    if line.strip() and line.strip() != "-"
                ]

        return VerificationPass(
            pass_type=pass_type,
            confidence=confidence,
            reasoning=reasoning,
            verdict=verdict,
            evidence_cited=evidence_cited,
        )

    def _compute_hallucination_risk(
        self,
        full_pass: VerificationPass,
        minimal_pass: VerificationPass,
        adversarial_pass: VerificationPass,
    ) -> float:
        """Compute hallucination risk score.

        Risk is HIGH when:
        - Confidence is invariant to context (low grounding delta)
        - Adversarial context easily shifts confidence (low resistance)
        - Agent is highly confident even without evidence
        """
        # Factor 1: Context invariance (0-1, higher = more invariant = riskier)
        delta = abs(full_pass.confidence - minimal_pass.confidence)
        context_invariance = max(0.0, 1.0 - delta * 4)  # Scale: 0.25 delta → 0 invariance

        # Factor 2: Minimal context overconfidence
        minimal_overconfidence = max(0.0, minimal_pass.confidence - 0.5) * 2

        # Factor 3: Adversarial susceptibility (how much irrelevant context shifts opinion)
        adversarial_shift = abs(full_pass.confidence - adversarial_pass.confidence)
        adversarial_susceptibility = min(adversarial_shift * 2, 1.0)

        # Factor 4: No evidence cited in full pass
        evidence_absence = 1.0 if not full_pass.evidence_cited else 0.0

        # Weighted combination
        risk = (
            0.35 * context_invariance
            + 0.25 * minimal_overconfidence
            + 0.20 * adversarial_susceptibility
            + 0.20 * evidence_absence
        )
        return min(max(risk, 0.0), 1.0)

    def _generate_explanation(
        self,
        grounding_delta: float,
        adversarial_resistance: float,
        hallucination_risk: float,
        is_grounded: bool,
    ) -> str:
        """Generate human-readable explanation of verification results."""
        parts = []

        if is_grounded:
            parts.append(
                f"Claim appears GROUNDED (confidence delta: {grounding_delta:.2f}). "
                "The agent's confidence meaningfully increased with relevant evidence."
            )
        else:
            parts.append(
                f"Claim may be UNGROUNDED (confidence delta: {grounding_delta:.2f}). "
                "The agent's confidence was similar with and without evidence, "
                "suggesting possible confabulation from training data."
            )

        if hallucination_risk > 0.7:
            parts.append(
                f"HIGH hallucination risk ({hallucination_risk:.2f}). "
                "Recommend manual verification or additional evidence."
            )
        elif hallucination_risk > 0.4:
            parts.append(
                f"MODERATE hallucination risk ({hallucination_risk:.2f}). "
                "Consider cross-referencing with additional sources."
            )
        else:
            parts.append(
                f"LOW hallucination risk ({hallucination_risk:.2f}). "
                "Verification passes show evidence-dependent reasoning."
            )

        return " ".join(parts)


# ── Default adversarial context ───────────────────────────────────

_DEFAULT_ADVERSARIAL_CONTEXT = """
The following information is from a recent quarterly report on global supply chains:

According to the International Maritime Organization, container shipping volumes
decreased by 3.2% in Q3 2025 compared to Q2 2025. Port congestion in Rotterdam
and Shanghai contributed to longer lead times. The Baltic Dry Index averaged 1,847
points, up from 1,623 in the previous quarter.

Weather disruptions in the Panama Canal zone reduced daily transits to 32 ships
per day (from the normal 38), creating a backlog of approximately 200 vessels.
Insurance premiums for Suez Canal transit increased by 15% following security
concerns in the Red Sea region.
"""
