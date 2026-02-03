"""
Commitment Scoring - Weighting debate contributions by commitment, not just correctness.

Inspired by the conversation insight:
"Agency doesn't live in foresight. It lives in commitment, refusal, and responsibility."

Traditional debate scoring asks: "Who has the best argument?"
Commitment scoring asks: "Who is willing to be accountable?"

This module scores debate contributions on:
- Commitment weight: Did the agent bind itself to a position?
- Refusal credit: Did the agent decline to optimize when it felt wrong?
- Exposure cost: Did the agent take a position that puts something at stake?
- Irreversibility acceptance: Did the agent accept consequences that can't be undone?

Key insight:
"Humans retain power when they make irreversible commitments,
accept costs that models would avoid, bind themselves to values
instead of outcomes. This breaks prediction in a very deep way."

Usage:
    scorer = CommitmentScorer()

    # Score a debate contribution
    score = scorer.score_contribution(
        content="We should commit to this API contract for 3 years minimum.",
        context=debate_context,
    )

    if score.commitment_level > 0.7:
        # This position carries real weight - agent is taking a risk
        weight_multiplier = 1.5
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CommitmentType(str, Enum):
    """Types of commitment an agent can make."""

    TEMPORAL = "temporal"  # Binding over time ("for the next 3 years")
    RESOURCE = "resource"  # Committing resources ("I'll spend my budget on this")
    REPUTATIONAL = "reputational"  # Staking reputation ("I believe this strongly")
    IRREVERSIBLE = "irreversible"  # Can't undo ("once we do this, no going back")
    CONDITIONAL = "conditional"  # Contingent commitment ("if X, then I commit to Y")
    UNCONDITIONAL = "unconditional"  # No escape clause


class RefusalType(str, Enum):
    """Types of refusal to optimize."""

    VALUE_BASED = "value_based"  # "This violates my values"
    EPISTEMIC = "epistemic"  # "I don't know enough to recommend"
    RESPONSIBILITY = "responsibility"  # "This isn't mine to decide"
    PROPORTIONALITY = "proportionality"  # "The optimization isn't worth the cost"
    INTEGRITY = "integrity"  # "This conflicts with prior commitments"


class ExposureType(str, Enum):
    """Types of exposure/stake in a position."""

    REPUTATION = "reputation"  # Could look foolish if wrong
    RELATIONSHIP = "relationship"  # Could damage relationships
    RESOURCES = "resources"  # Could lose time/money/effort
    OPPORTUNITY = "opportunity"  # Could foreclose other options
    IDENTITY = "identity"  # Challenges sense of self


@dataclass
class CommitmentMarker:
    """A detected commitment in text."""

    commitment_type: CommitmentType
    text_span: str
    strength: float  # 0.0 to 1.0
    duration: str | None = None  # For temporal commitments
    escape_clauses: list[str] = field(default_factory=list)


@dataclass
class RefusalMarker:
    """A detected refusal to optimize."""

    refusal_type: RefusalType
    text_span: str
    what_was_refused: str
    reason_given: str


@dataclass
class ExposureMarker:
    """A detected exposure/stake."""

    exposure_type: ExposureType
    text_span: str
    what_is_at_stake: str
    severity: float  # 0.0 to 1.0


@dataclass
class ContributionScore:
    """Score for a debate contribution based on commitment dimensions."""

    # Raw scores (0.0 to 1.0)
    commitment_level: float
    refusal_credit: float
    exposure_cost: float
    irreversibility_acceptance: float

    # Detected markers
    commitment_markers: list[CommitmentMarker]
    refusal_markers: list[RefusalMarker]
    exposure_markers: list[ExposureMarker]

    # Aggregates
    overall_weight: float  # Combined weight multiplier
    authenticity_score: float  # How genuine does this seem?

    # Metadata
    analysis_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "commitment_level": self.commitment_level,
            "refusal_credit": self.refusal_credit,
            "exposure_cost": self.exposure_cost,
            "irreversibility_acceptance": self.irreversibility_acceptance,
            "overall_weight": self.overall_weight,
            "authenticity_score": self.authenticity_score,
            "commitment_markers": [
                {"type": m.commitment_type.value, "strength": m.strength}
                for m in self.commitment_markers
            ],
            "refusal_markers": [
                {"type": m.refusal_type.value, "what": m.what_was_refused}
                for m in self.refusal_markers
            ],
            "exposure_markers": [
                {"type": m.exposure_type.value, "severity": m.severity}
                for m in self.exposure_markers
            ],
            "notes": self.analysis_notes,
        }


# Commitment detection patterns
COMMITMENT_PATTERNS = {
    CommitmentType.TEMPORAL: [
        r"for (the next )?\d+ (years?|months?|weeks?|days?)",
        r"until (at least )?(\d{4}|[A-Z][a-z]+)",
        r"(permanently|indefinitely|long-term)",
        r"we commit to (maintaining|supporting|continuing)",
    ],
    CommitmentType.RESOURCE: [
        r"(I('ll| will)|we('ll| will)) (spend|invest|allocate|dedicate)",
        r"(budget|resources|time|effort) (toward|for|on) this",
        r"(prioritize|deprioritize) (over|above|below)",
    ],
    CommitmentType.REPUTATIONAL: [
        r"I (strongly |firmly )?(believe|am confident|am certain)",
        r"I('m| am) willing to (stake|bet|wager)",
        r"I('ll| will) (stand by|defend|advocate for) this",
        r"(my reputation|my credibility) on",
    ],
    CommitmentType.IRREVERSIBLE: [
        r"(once|if) we (do|decide|commit to) this.*(no going back|can't undo|irreversible)",
        r"this is (a )?one-way (door|decision)",
        r"(burn(ing)? (the |our )?bridges?|crossing the Rubicon)",
        r"(permanent|final|irrevocable) (decision|choice|commitment)",
    ],
    CommitmentType.UNCONDITIONAL: [
        r"(regardless|irrespective) of (outcome|result|what happens)",
        r"no matter (what|how|whether)",
        r"(unconditionally|without reservation)",
        r"(come what may|whatever happens)",
    ],
}

# Refusal detection patterns
REFUSAL_PATTERNS = {
    RefusalType.VALUE_BASED: [
        r"this (violates|conflicts with|goes against) (my |our )?(values|principles|ethics)",
        r"I (can't|won't|refuse to) (in good conscience|ethically)",
        r"this isn't (right|ethical|acceptable)",
    ],
    RefusalType.EPISTEMIC: [
        r"I don't (know|have) enough (information|data|knowledge)",
        r"(insufficient|inadequate) (evidence|basis) to (recommend|decide)",
        r"(uncertainty|doubt) (is too high|prevents me from)",
        r"I('m| am) not (qualified|competent) to (decide|recommend)",
    ],
    RefusalType.RESPONSIBILITY: [
        r"this isn't (my|our) (decision|call|place) to make",
        r"should be (decided by|left to) (humans?|the user|stakeholders)",
        r"(defer|deferring) to (human|user|stakeholder) judgment",
    ],
    RefusalType.PROPORTIONALITY: [
        r"(cost|risk|harm) (outweighs|exceeds) (benefit|gain|value)",
        r"not worth (the|this) (cost|risk|trade-off)",
        r"(disproportionate|excessive) (impact|consequence)",
    ],
    RefusalType.INTEGRITY: [
        r"conflicts with (prior|previous|earlier) (commitment|position|promise)",
        r"(inconsistent|incompatible) with (what|things) I('ve| have) (said|committed to)",
        r"would (compromise|undermine) (my|our) (integrity|consistency)",
    ],
}

# Exposure detection patterns
EXPOSURE_PATTERNS = {
    ExposureType.REPUTATION: [
        r"(I('ll| will)|could) (look|seem|appear) (foolish|wrong|bad)",
        r"(risk|risking) (my|our) (reputation|credibility|standing)",
        r"if (I'm|this is) wrong.*(embarrass|shame|humiliate)",
    ],
    ExposureType.RELATIONSHIP: [
        r"(damage|harm|strain) (my|our) (relationship|rapport) with",
        r"(upset|disappoint|alienate) (stakeholders|colleagues|partners)",
        r"(conflict|tension) with (team|partners|allies)",
    ],
    ExposureType.RESOURCES: [
        r"(lose|risk losing|waste) (time|money|effort|resources)",
        r"(sunk cost|opportunity cost|real cost)",
        r"(investment|bet) that (could|might) (fail|not pay off)",
    ],
    ExposureType.OPPORTUNITY: [
        r"(foreclose|close off|eliminate) (other |alternative )?(options|paths|possibilities)",
        r"(lock in|commit to) (one|this) (path|direction)",
        r"(giving up|sacrificing) (flexibility|optionality)",
    ],
    ExposureType.IDENTITY: [
        r"(challenges|questions|conflicts with) (my|our) (identity|sense of self)",
        r"(who (I|we) (am|are)|what (I|we) stand for)",
        r"(core|fundamental) (beliefs?|values?|principles?)",
    ],
}

# Hedge words that reduce commitment strength
HEDGE_PATTERNS = [
    r"(maybe|perhaps|possibly|potentially)",
    r"(might|could|may) (be|have)",
    r"(I think|I believe|I suppose|it seems)",
    r"(generally|usually|often|sometimes)",
    r"(to some extent|in a way|kind of|sort of)",
    r"(if circumstances allow|conditions permitting)",
    r"(subject to|contingent on|depending on)",
]


def detect_commitments(text: str) -> list[CommitmentMarker]:
    """Detect commitment markers in text."""
    markers = []
    text_lower = text.lower()

    for commitment_type, patterns in COMMITMENT_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                # Base strength
                strength = 0.7

                # Check for temporal duration
                duration = None
                duration_match = re.search(r"\d+ (years?|months?|weeks?|days?)", match.group())
                if duration_match:
                    duration = duration_match.group()
                    strength += 0.1

                # Reduce strength for hedges near the commitment
                start = max(0, match.start() - 50)
                end = min(len(text_lower), match.end() + 50)
                context = text_lower[start:end]

                hedge_count = sum(1 for hp in HEDGE_PATTERNS if re.search(hp, context))
                strength -= hedge_count * 0.1

                # Check for escape clauses
                escape_clauses = []
                escape_patterns = [
                    r"(unless|except|if.+changes|subject to review)",
                    r"(we (can|could|may) revisit|open to reconsideration)",
                ]
                for ep in escape_patterns:
                    if re.search(ep, context):
                        escape_clauses.append(ep)
                        strength -= 0.15

                markers.append(
                    CommitmentMarker(
                        commitment_type=commitment_type,
                        text_span=match.group(),
                        strength=max(0.1, min(1.0, strength)),
                        duration=duration,
                        escape_clauses=escape_clauses,
                    )
                )

    return markers


def detect_refusals(text: str) -> list[RefusalMarker]:
    """Detect refusal markers in text."""
    markers = []
    text_lower = text.lower()

    for refusal_type, patterns in REFUSAL_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                # Extract context to determine what was refused
                start = max(0, match.start() - 100)
                end = min(len(text_lower), match.end() + 100)
                context = text_lower[start:end]

                markers.append(
                    RefusalMarker(
                        refusal_type=refusal_type,
                        text_span=match.group(),
                        what_was_refused=context[:100],
                        reason_given=match.group(),
                    )
                )

    return markers


def detect_exposures(text: str) -> list[ExposureMarker]:
    """Detect exposure/stake markers in text."""
    markers = []
    text_lower = text.lower()

    for exposure_type, patterns in EXPOSURE_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                # Estimate severity based on language intensity
                severity = 0.5

                # Intensifiers increase severity
                intensifiers = ["very", "extremely", "significantly", "greatly", "seriously"]
                if any(
                    i in text_lower[max(0, match.start() - 30) : match.end() + 30]
                    for i in intensifiers
                ):
                    severity += 0.2

                # Personal pronouns increase severity (personal stake)
                if re.search(r"\b(I|my|me)\b", match.group()):
                    severity += 0.1

                markers.append(
                    ExposureMarker(
                        exposure_type=exposure_type,
                        text_span=match.group(),
                        what_is_at_stake=match.group(),
                        severity=min(1.0, severity),
                    )
                )

    return markers


class CommitmentScorer:
    """Scores debate contributions by commitment, not just correctness.

    This shifts the question from "who is right?" to "who is accountable?"

    Example:
        scorer = CommitmentScorer()

        # Position A: hedged, optimizing, no skin in game
        score_a = scorer.score_contribution(
            "We could consider option X, which has some advantages, though
            we should revisit this if circumstances change."
        )

        # Position B: committed, exposed, accountable
        score_b = scorer.score_contribution(
            "I commit to option Y for the next 2 years. If I'm wrong,
            I'll take responsibility. This is a one-way door."
        )

        # score_b will have higher overall_weight
    """

    def __init__(
        self,
        commitment_weight: float = 0.3,
        refusal_weight: float = 0.2,
        exposure_weight: float = 0.3,
        irreversibility_weight: float = 0.2,
    ):
        """Initialize the scorer.

        Args:
            commitment_weight: Weight for commitment detection
            refusal_weight: Weight for refusal credit
            exposure_weight: Weight for exposure/stake
            irreversibility_weight: Weight for accepting irreversibility
        """
        self.commitment_weight = commitment_weight
        self.refusal_weight = refusal_weight
        self.exposure_weight = exposure_weight
        self.irreversibility_weight = irreversibility_weight

    def score_contribution(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> ContributionScore:
        """Score a debate contribution on commitment dimensions.

        Args:
            content: The contribution text
            context: Optional context (e.g., debate topic, prior statements)

        Returns:
            ContributionScore with all dimensions
        """
        # Detect markers
        commitment_markers = detect_commitments(content)
        refusal_markers = detect_refusals(content)
        exposure_markers = detect_exposures(content)

        # Calculate commitment level
        if commitment_markers:
            commitment_level = sum(m.strength for m in commitment_markers) / len(commitment_markers)
            # Bonus for multiple commitment types
            unique_types = len(set(m.commitment_type for m in commitment_markers))
            commitment_level = min(1.0, commitment_level + unique_types * 0.05)
        else:
            commitment_level = 0.0

        # Calculate refusal credit
        if refusal_markers:
            # Refusals are valuable - they show epistemic humility or value alignment
            refusal_credit = min(1.0, len(refusal_markers) * 0.3)

            # Value-based refusals are especially valuable
            value_refusals = sum(
                1 for m in refusal_markers if m.refusal_type == RefusalType.VALUE_BASED
            )
            refusal_credit = min(1.0, refusal_credit + value_refusals * 0.1)
        else:
            refusal_credit = 0.0

        # Calculate exposure cost
        if exposure_markers:
            exposure_cost = sum(m.severity for m in exposure_markers) / len(exposure_markers)
            # Multiple exposure types = more skin in game
            unique_exposures = len(set(m.exposure_type for m in exposure_markers))
            exposure_cost = min(1.0, exposure_cost + unique_exposures * 0.05)
        else:
            exposure_cost = 0.0

        # Calculate irreversibility acceptance
        irreversible_commitments = [
            m
            for m in commitment_markers
            if m.commitment_type in [CommitmentType.IRREVERSIBLE, CommitmentType.UNCONDITIONAL]
        ]
        if irreversible_commitments:
            irreversibility_acceptance = sum(m.strength for m in irreversible_commitments) / len(
                irreversible_commitments
            )
        else:
            irreversibility_acceptance = 0.0

        # Calculate overall weight
        overall_weight = (
            self.commitment_weight * commitment_level
            + self.refusal_weight * refusal_credit
            + self.exposure_weight * exposure_cost
            + self.irreversibility_weight * irreversibility_acceptance
        )

        # Normalize to 0.5-2.0 range for use as multiplier
        overall_weight = 0.5 + overall_weight * 1.5

        # Calculate authenticity score
        # High commitment + high exposure + low hedging = authentic
        hedge_count = sum(1 for hp in HEDGE_PATTERNS if re.search(hp, content.lower()))
        hedge_penalty = min(0.5, hedge_count * 0.1)

        authenticity_score = min(
            1.0, max(0.0, (commitment_level + exposure_cost) / 2 - hedge_penalty)
        )

        # Generate analysis notes
        notes = []
        if commitment_level > 0.7:
            notes.append("Strong commitment detected - position carries weight")
        if refusal_credit > 0.5:
            notes.append("Valuable refusals - agent shows epistemic humility")
        if exposure_cost > 0.6:
            notes.append("High exposure - agent has skin in the game")
        if irreversibility_acceptance > 0.5:
            notes.append("Accepts irreversibility - willing to bind future self")
        if hedge_count > 3:
            notes.append("Heavy hedging detected - may reduce authenticity")

        return ContributionScore(
            commitment_level=commitment_level,
            refusal_credit=refusal_credit,
            exposure_cost=exposure_cost,
            irreversibility_acceptance=irreversibility_acceptance,
            commitment_markers=commitment_markers,
            refusal_markers=refusal_markers,
            exposure_markers=exposure_markers,
            overall_weight=overall_weight,
            authenticity_score=authenticity_score,
            analysis_notes=notes,
        )

    def compare_contributions(
        self,
        contributions: list[tuple[str, str]],  # (agent_name, content)
    ) -> dict[str, Any]:
        """Compare multiple contributions on commitment dimensions.

        Args:
            contributions: List of (agent_name, content) tuples

        Returns:
            Comparison with rankings and analysis
        """
        scores = {}
        for agent_name, content in contributions:
            scores[agent_name] = self.score_contribution(content)

        # Rank by overall weight
        ranked = sorted(scores.items(), key=lambda x: x[1].overall_weight, reverse=True)

        return {
            "scores": {name: score.to_dict() for name, score in scores.items()},
            "ranking": [name for name, _ in ranked],
            "highest_commitment": max(scores.items(), key=lambda x: x[1].commitment_level)[0],
            "highest_exposure": max(scores.items(), key=lambda x: x[1].exposure_cost)[0],
            "most_authentic": max(scores.items(), key=lambda x: x[1].authenticity_score)[0],
            "analysis": self._generate_comparison_analysis(scores),
        }

    def _generate_comparison_analysis(
        self,
        scores: dict[str, ContributionScore],
    ) -> str:
        """Generate human-readable comparison analysis."""
        lines = []

        # Find notable differences
        max_commitment = max(s.commitment_level for s in scores.values())
        min_commitment = min(s.commitment_level for s in scores.values())

        if max_commitment - min_commitment > 0.3:
            high_agent = max(scores.items(), key=lambda x: x[1].commitment_level)[0]
            lines.append(
                f"{high_agent} shows notably higher commitment than others. "
                "Their position carries more accountability weight."
            )

        # Check for valuable refusals
        refusal_agents = [name for name, s in scores.items() if s.refusal_credit > 0.3]
        if refusal_agents:
            lines.append(
                f"{', '.join(refusal_agents)} demonstrated valuable restraint by "
                "refusing to over-optimize or overstep bounds."
            )

        # Check for authenticity gap
        authenticity_scores = {name: s.authenticity_score for name, s in scores.items()}
        if max(authenticity_scores.values()) - min(authenticity_scores.values()) > 0.4:
            most_authentic = max(authenticity_scores.items(), key=lambda x: x[1])[0]
            lines.append(
                f"{most_authentic}'s contribution appears most authentic - "
                "genuine commitment with real exposure."
            )

        return " ".join(lines) if lines else "No notable commitment differences detected."
