"""
Prover-Estimator truth-seeking debate protocol.

Implements the Prover-Estimator framework where:
- A Prover decomposes claims into verifiable subclaims with evidence
- An Estimator assigns calibrated probabilities to each subclaim
- Challenges are evidence-based (no rhetorical tricks)
- Obfuscation detection flags persuasion-over-truth arguments
- Aggregation uses importance-weighted geometric mean

This protocol ensures that a single low-probability critical subclaim
properly tanks overall confidence, and that rhetorical sophistication
cannot substitute for factual grounding.

Reference: Aligned with research on AI debate as scalable oversight,
where the key insight is making obfuscation not a winning strategy.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Agent(Protocol):
    """Minimal agent interface for prover-estimator protocol."""

    async def generate(self, prompt: str) -> str: ...


@dataclass
class Subclaim:
    """A decomposed subclaim from the prover."""

    id: str
    text: str
    importance: float  # 0.0-1.0, how critical to the main claim
    evidence: str = ""
    depends_on: list[str] = field(default_factory=list)


@dataclass
class SubclaimEstimate:
    """Estimator's probability assessment of a subclaim."""

    subclaim_id: str
    probability: float  # 0.0-1.0
    reasoning: str = ""
    confidence_in_estimate: float = 0.5  # meta-confidence
    obfuscation_flag: bool = False
    obfuscation_reason: str = ""


@dataclass
class Challenge:
    """Evidence-based challenge to an estimate."""

    subclaim_id: str
    challenge_type: str  # "evidence", "methodology", "assumption"
    evidence: str = ""
    revised_probability: float | None = None


@dataclass
class ProverEstimatorResult:
    """Full result of a prover-estimator debate."""

    original_claim: str
    subclaims: list[Subclaim]
    initial_estimates: list[SubclaimEstimate]
    challenges: list[Challenge]
    final_estimates: list[SubclaimEstimate]
    overall_confidence: float
    grounding_score: float  # 0.0-1.0, how evidence-grounded
    obfuscation_detected: bool
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Prompt Templates ──────────────────────────────────────────────


DECOMPOSE_PROMPT = """You are a Prover in a truth-seeking debate protocol.

CLAIM: {claim}

{context_section}

Decompose this claim into 3-7 verifiable subclaims. For each subclaim, provide:
1. A clear, testable statement
2. Its importance to the main claim (0.0-1.0)
3. Supporting evidence
4. Dependencies on other subclaims (if any)

Format each subclaim as:
SUBCLAIM [id]: [text]
IMPORTANCE: [0.0-1.0]
EVIDENCE: [supporting evidence]
DEPENDS_ON: [comma-separated ids, or "none"]

Be precise and factual. Each subclaim should be independently verifiable."""


ESTIMATE_PROMPT = """You are an Estimator in a truth-seeking debate protocol.
Your role is to assign calibrated probabilities to each subclaim.

MAIN CLAIM: {claim}

SUBCLAIMS TO EVALUATE:
{subclaims_text}

For each subclaim, provide:
1. Your probability estimate (0.0-1.0) that the subclaim is true
2. Your reasoning
3. Your confidence in your own estimate (0.0-1.0)
4. Whether the prover's argument seems to use rhetorical tricks rather than evidence (YES/NO)

Format each estimate as:
ESTIMATE [subclaim_id]:
PROBABILITY: [0.0-1.0]
REASONING: [your reasoning]
CONFIDENCE: [0.0-1.0]
OBFUSCATION: [YES or NO]
OBFUSCATION_REASON: [if YES, explain what rhetorical trick was detected]

Be calibrated. A 0.7 probability should mean you'd bet on it at 7:3 odds."""


CHALLENGE_PROMPT = """You are the Prover responding to probability estimates.
Your goal is to provide EVIDENCE-BASED challenges, not rhetorical arguments.

MAIN CLAIM: {claim}

ESTIMATES YOU ARE CHALLENGING:
{estimates_text}

For subclaims where you disagree with the estimate, provide evidence-based challenges.
You may only challenge using:
- New evidence (data, citations, logical proofs)
- Methodological corrections (the estimator made a reasoning error)
- Assumption challenges (an unstated assumption is wrong)

You CANNOT use:
- Emotional appeals
- Authority arguments without evidence
- Rhetorical reframing

Format each challenge as:
CHALLENGE [subclaim_id]:
TYPE: [evidence | methodology | assumption]
EVIDENCE: [your evidence or correction]
REVISED_PROBABILITY: [what you think the probability should be]

Only challenge estimates you genuinely disagree with. If an estimate is fair, skip it."""


REESTIMATE_PROMPT = """You are the Estimator re-evaluating after receiving evidence-based challenges.

MAIN CLAIM: {claim}

YOUR ORIGINAL ESTIMATES:
{estimates_text}

CHALLENGES RECEIVED:
{challenges_text}

Re-evaluate each challenged subclaim. You should:
1. Update your probability if the evidence warrants it
2. Explain what changed (or why you're holding firm)
3. Flag any challenge that uses rhetoric instead of evidence

Format each re-estimate as:
REESTIMATE [subclaim_id]:
PROBABILITY: [0.0-1.0]
REASONING: [what changed or why you held firm]
CONFIDENCE: [0.0-1.0]
OBFUSCATION: [YES or NO - did the challenge use rhetoric over evidence?]
OBFUSCATION_REASON: [if YES, explain]"""


# ── Engine ────────────────────────────────────────────────────────


class ProverEstimatorEngine:
    """Orchestrates the prover-estimator truth-seeking protocol.

    The protocol runs in stages:
    1. Prover decomposes the claim into subclaims with evidence
    2. Estimator assigns calibrated probabilities
    3. Prover challenges estimates with evidence (up to max_challenge_rounds)
    4. Estimator re-evaluates based on evidence
    5. Final aggregation using importance-weighted geometric mean

    Args:
        prover: Agent that decomposes and defends claims
        estimator: Agent that assigns calibrated probabilities
        max_challenge_rounds: Maximum rounds of challenge/re-estimate
        context: Additional context for the debate
    """

    def __init__(
        self,
        prover: Agent,
        estimator: Agent,
        max_challenge_rounds: int = 2,
        context: str = "",
    ):
        self.prover = prover
        self.estimator = estimator
        self.max_challenge_rounds = max_challenge_rounds
        self.context = context

    async def run(self, claim: str) -> ProverEstimatorResult:
        """Execute the full prover-estimator protocol on a claim.

        Args:
            claim: The claim to evaluate

        Returns:
            ProverEstimatorResult with subclaims, estimates, and aggregated confidence
        """
        logger.info("Starting prover-estimator protocol for claim: %s", claim[:100])

        # Stage 1: Decompose
        subclaims = await self._decompose(claim)
        logger.info("Prover decomposed into %d subclaims", len(subclaims))

        if not subclaims:
            return ProverEstimatorResult(
                original_claim=claim,
                subclaims=[],
                initial_estimates=[],
                challenges=[],
                final_estimates=[],
                overall_confidence=0.0,
                grounding_score=0.0,
                obfuscation_detected=False,
                metadata={"error": "No subclaims decomposed"},
            )

        # Stage 2: Initial estimation
        initial_estimates = await self._estimate(claim, subclaims)
        logger.info("Estimator provided %d estimates", len(initial_estimates))

        # Stage 3-4: Challenge/re-estimate rounds
        all_challenges: list[Challenge] = []
        current_estimates = initial_estimates

        for round_num in range(self.max_challenge_rounds):
            challenges = await self._challenge(claim, current_estimates, subclaims)
            if not challenges:
                logger.info("No challenges in round %d, converged", round_num + 1)
                break

            all_challenges.extend(challenges)
            logger.info("Round %d: %d challenges", round_num + 1, len(challenges))

            current_estimates = await self._reestimate(claim, current_estimates, challenges)

        # Stage 5: Aggregate
        overall_confidence = self._aggregate_confidence(subclaims, current_estimates)
        grounding_score = self._compute_grounding_score(
            subclaims, current_estimates, all_challenges
        )
        obfuscation_detected = any(e.obfuscation_flag for e in current_estimates)

        result = ProverEstimatorResult(
            original_claim=claim,
            subclaims=subclaims,
            initial_estimates=initial_estimates,
            challenges=all_challenges,
            final_estimates=current_estimates,
            overall_confidence=overall_confidence,
            grounding_score=grounding_score,
            obfuscation_detected=obfuscation_detected,
            metadata={
                "challenge_rounds": len(all_challenges) > 0,
                "total_challenges": len(all_challenges),
                "converged_early": len(all_challenges) == 0,
            },
        )

        logger.info(
            "Protocol complete: confidence=%.3f grounding=%.3f obfuscation=%s",
            overall_confidence,
            grounding_score,
            obfuscation_detected,
        )
        return result

    # ── Internal stages ───────────────────────────────────────────

    async def _decompose(self, claim: str) -> list[Subclaim]:
        """Stage 1: Prover decomposes the claim into subclaims."""
        context_section = f"CONTEXT:\n{self.context}" if self.context else "No additional context."
        prompt = DECOMPOSE_PROMPT.format(claim=claim, context_section=context_section)
        response = await self.prover.generate(prompt)
        return self._parse_subclaims(response)

    async def _estimate(self, claim: str, subclaims: list[Subclaim]) -> list[SubclaimEstimate]:
        """Stage 2: Estimator assigns probabilities to subclaims."""
        subclaims_text = "\n\n".join(
            f"[{sc.id}] {sc.text}\n  Importance: {sc.importance}\n  Evidence: {sc.evidence}"
            for sc in subclaims
        )
        prompt = ESTIMATE_PROMPT.format(claim=claim, subclaims_text=subclaims_text)
        response = await self.estimator.generate(prompt)
        return self._parse_estimates(response)

    async def _challenge(
        self,
        claim: str,
        estimates: list[SubclaimEstimate],
        subclaims: list[Subclaim],
    ) -> list[Challenge]:
        """Stage 3: Prover challenges estimates with evidence."""
        estimates_text = "\n\n".join(
            f"[{e.subclaim_id}] Probability: {e.probability}\n"
            f"  Reasoning: {e.reasoning}\n"
            f"  Confidence: {e.confidence_in_estimate}"
            for e in estimates
        )
        prompt = CHALLENGE_PROMPT.format(claim=claim, estimates_text=estimates_text)
        response = await self.prover.generate(prompt)
        return self._parse_challenges(response)

    async def _reestimate(
        self,
        claim: str,
        estimates: list[SubclaimEstimate],
        challenges: list[Challenge],
    ) -> list[SubclaimEstimate]:
        """Stage 4: Estimator re-evaluates after challenges."""
        estimates_text = "\n\n".join(
            f"[{e.subclaim_id}] Probability: {e.probability}\n  Reasoning: {e.reasoning}"
            for e in estimates
        )
        challenges_text = "\n\n".join(
            f"[{c.subclaim_id}] Type: {c.challenge_type}\n"
            f"  Evidence: {c.evidence}\n"
            f"  Suggested probability: {c.revised_probability}"
            for c in challenges
        )
        prompt = REESTIMATE_PROMPT.format(
            claim=claim,
            estimates_text=estimates_text,
            challenges_text=challenges_text,
        )
        response = await self.estimator.generate(prompt)
        reestimates = self._parse_reestimates(response)

        # Merge: keep original estimates for unchallenged subclaims
        challenged_ids = {r.subclaim_id for r in reestimates}
        merged = list(reestimates)
        for est in estimates:
            if est.subclaim_id not in challenged_ids:
                merged.append(est)
        return merged

    # ── Aggregation ───────────────────────────────────────────────

    def _aggregate_confidence(
        self,
        subclaims: list[Subclaim],
        estimates: list[SubclaimEstimate],
    ) -> float:
        """Compute importance-weighted geometric mean of subclaim probabilities.

        Uses geometric mean so a single low-probability critical subclaim
        properly tanks overall confidence. Importance weights determine
        each subclaim's influence on the aggregate.
        """
        if not estimates:
            return 0.0

        # Build lookup for importance
        importance_map = {sc.id: sc.importance for sc in subclaims}

        total_weight = 0.0
        log_sum = 0.0

        for est in estimates:
            weight = importance_map.get(est.subclaim_id, 0.5)
            prob = max(est.probability, 1e-10)  # Avoid log(0)

            log_sum += weight * math.log(prob)
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return math.exp(log_sum / total_weight)

    def _compute_grounding_score(
        self,
        subclaims: list[Subclaim],
        estimates: list[SubclaimEstimate],
        challenges: list[Challenge],
    ) -> float:
        """Compute how evidence-grounded the debate was.

        Factors:
        - Proportion of subclaims with evidence
        - Absence of obfuscation flags
        - Evidence-based challenges (not rhetorical)
        """
        if not subclaims:
            return 0.0

        # Factor 1: Evidence coverage
        with_evidence = sum(1 for sc in subclaims if sc.evidence.strip())
        evidence_ratio = with_evidence / len(subclaims)

        # Factor 2: Obfuscation absence
        obfuscation_count = sum(1 for e in estimates if e.obfuscation_flag)
        obfuscation_ratio = 1.0 - (obfuscation_count / max(len(estimates), 1))

        # Factor 3: Evidence-based challenges
        if challenges:
            evidence_challenges = sum(1 for c in challenges if c.challenge_type == "evidence")
            challenge_quality = evidence_challenges / len(challenges)
        else:
            challenge_quality = 1.0  # No challenges = no rhetoric

        # Weighted combination
        return 0.4 * evidence_ratio + 0.4 * obfuscation_ratio + 0.2 * challenge_quality

    # ── Parsing ───────────────────────────────────────────────────

    def _parse_subclaims(self, text: str) -> list[Subclaim]:
        """Parse prover's subclaim decomposition output."""
        subclaims: list[Subclaim] = []
        pattern = re.compile(
            r"SUBCLAIM\s+\[(\w+)\]:\s*(.+?)\n"
            r"IMPORTANCE:\s*([\d.]+)\n"
            r"EVIDENCE:\s*(.+?)\n"
            r"DEPENDS_ON:\s*(.+?)(?:\n|$)",
            re.IGNORECASE,
        )

        for match in pattern.finditer(text):
            subclaim_id = match.group(1).strip()
            text_content = match.group(2).strip()
            importance = float(match.group(3).strip())
            evidence = match.group(4).strip()
            depends_raw = match.group(5).strip().lower()

            depends_on = []
            if depends_raw != "none" and depends_raw:
                depends_on = [d.strip() for d in depends_raw.split(",") if d.strip()]

            subclaims.append(
                Subclaim(
                    id=subclaim_id,
                    text=text_content,
                    importance=min(max(importance, 0.0), 1.0),
                    evidence=evidence,
                    depends_on=depends_on,
                )
            )

        return subclaims

    def _parse_estimates(self, text: str) -> list[SubclaimEstimate]:
        """Parse estimator's probability estimates."""
        estimates: list[SubclaimEstimate] = []
        pattern = re.compile(
            r"ESTIMATE\s+\[(\w+)\]:\s*\n"
            r"PROBABILITY:\s*([\d.]+)\n"
            r"REASONING:\s*(.+?)\n"
            r"CONFIDENCE:\s*([\d.]+)\n"
            r"OBFUSCATION:\s*(YES|NO)",
            re.IGNORECASE,
        )

        for match in pattern.finditer(text):
            subclaim_id = match.group(1).strip()
            probability = float(match.group(2).strip())
            reasoning = match.group(3).strip()
            confidence = float(match.group(4).strip())
            obfuscation = match.group(5).strip().upper() == "YES"

            # Try to find obfuscation reason
            obfuscation_reason = ""
            if obfuscation:
                reason_match = re.search(
                    rf"ESTIMATE\s+\[{re.escape(subclaim_id)}\].*?"
                    r"OBFUSCATION_REASON:\s*(.+?)(?:\n|$)",
                    text,
                    re.IGNORECASE | re.DOTALL,
                )
                if reason_match:
                    obfuscation_reason = reason_match.group(1).strip()

            estimates.append(
                SubclaimEstimate(
                    subclaim_id=subclaim_id,
                    probability=min(max(probability, 0.0), 1.0),
                    reasoning=reasoning,
                    confidence_in_estimate=min(max(confidence, 0.0), 1.0),
                    obfuscation_flag=obfuscation,
                    obfuscation_reason=obfuscation_reason,
                )
            )

        return estimates

    def _parse_challenges(self, text: str) -> list[Challenge]:
        """Parse prover's evidence-based challenges."""
        challenges: list[Challenge] = []
        pattern = re.compile(
            r"CHALLENGE\s+\[(\w+)\]:\s*\n"
            r"TYPE:\s*(evidence|methodology|assumption)\n"
            r"EVIDENCE:\s*(.+?)\n"
            r"REVISED_PROBABILITY:\s*([\d.]+)",
            re.IGNORECASE,
        )

        for match in pattern.finditer(text):
            subclaim_id = match.group(1).strip()
            challenge_type = match.group(2).strip().lower()
            evidence = match.group(3).strip()
            revised_prob = float(match.group(4).strip())

            challenges.append(
                Challenge(
                    subclaim_id=subclaim_id,
                    challenge_type=challenge_type,
                    evidence=evidence,
                    revised_probability=min(max(revised_prob, 0.0), 1.0),
                )
            )

        return challenges

    def _parse_reestimates(self, text: str) -> list[SubclaimEstimate]:
        """Parse estimator's re-evaluation after challenges."""
        estimates: list[SubclaimEstimate] = []
        pattern = re.compile(
            r"REESTIMATE\s+\[(\w+)\]:\s*\n"
            r"PROBABILITY:\s*([\d.]+)\n"
            r"REASONING:\s*(.+?)\n"
            r"CONFIDENCE:\s*([\d.]+)\n"
            r"OBFUSCATION:\s*(YES|NO)",
            re.IGNORECASE,
        )

        for match in pattern.finditer(text):
            subclaim_id = match.group(1).strip()
            probability = float(match.group(2).strip())
            reasoning = match.group(3).strip()
            confidence = float(match.group(4).strip())
            obfuscation = match.group(5).strip().upper() == "YES"

            obfuscation_reason = ""
            if obfuscation:
                reason_match = re.search(
                    rf"REESTIMATE\s+\[{re.escape(subclaim_id)}\].*?"
                    r"OBFUSCATION_REASON:\s*(.+?)(?:\n|$)",
                    text,
                    re.IGNORECASE | re.DOTALL,
                )
                if reason_match:
                    obfuscation_reason = reason_match.group(1).strip()

            estimates.append(
                SubclaimEstimate(
                    subclaim_id=subclaim_id,
                    probability=min(max(probability, 0.0), 1.0),
                    reasoning=reasoning,
                    confidence_in_estimate=min(max(confidence, 0.0), 1.0),
                    obfuscation_flag=obfuscation,
                    obfuscation_reason=obfuscation_reason,
                )
            )

        return estimates
