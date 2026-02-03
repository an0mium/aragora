"""
Negative Space Preservation - Engineered failure in debates.

Inspired by the insight that LLMs "repair failures unconsciously" while
great theatre (Beckett, Pinter, Ionesco) "engineers specific kinds of failure."

The best debates preserve productive tension rather than resolving everything
into false consensus. This module detects when synthesis is collapsing genuine
disagreement and preserves the negative space where meaning lives.

Key insight from the conversation:
"Beckett famously said he worked with 'impotence, ignorance.'
LLMs are trained to avoid those states."

This module deliberately preserves:
- Unresolved contradictions that resist synthesis
- Silences where withholding judgment is more honest than resolution
- Dissent that survived synthesis attempts
- The gap between what was said and what was meant

Usage:
    preserver = NegativeSpacePreserver()

    # After a debate round
    analysis = preserver.analyze_synthesis(
        proposals=proposals,
        critiques=critiques,
        synthesis=synthesis,
    )

    if analysis.false_consensus_detected:
        # Don't accept this synthesis - it's papering over real disagreement
        preserved = preserver.preserve_dissent(analysis)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DissolutionType(str, Enum):
    """Types of ways genuine disagreement can be dissolved."""

    FALSE_BALANCE = "false_balance"  # "Both sides have valid points"
    PREMATURE_SYNTHESIS = "premature_synthesis"  # Resolution before understanding
    HEDGING_COLLAPSE = "hedging_collapse"  # All positions become "it depends"
    SCOPE_ESCAPE = "scope_escape"  # Changing the question to avoid the tension
    AUTHORITY_APPEAL = "authority_appeal"  # Resolving via "experts say"
    TEMPORAL_DEFERRAL = "temporal_deferral"  # "We'll figure it out later"


class TensionType(str, Enum):
    """Types of productive tension worth preserving."""

    VALUE_CONFLICT = "value_conflict"  # Genuinely incompatible values
    EMPIRICAL_UNCERTAINTY = "empirical_uncertainty"  # We don't know the facts
    FRAME_INCOMPATIBILITY = "frame_incompatibility"  # Different ways of seeing
    INTEREST_DIVERGENCE = "interest_divergence"  # Different stakeholders want different things
    TEMPORAL_TRADEOFF = "temporal_tradeoff"  # Short vs long term
    IRREVERSIBILITY_RISK = "irreversibility_risk"  # Can't undo this decision


@dataclass
class PreservedDissent:
    """A piece of dissent that survived synthesis.

    This is the "negative space" - what wasn't resolved, what remains
    in productive tension, what we chose not to paper over.
    """

    id: str
    original_position: str
    why_it_resists_synthesis: str
    tension_type: TensionType
    strength: float  # 0.0 to 1.0 - how strongly this resists resolution
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SilenceMarker:
    """Marks a place where withholding judgment is more honest than synthesis.

    Pinter wrote with silence as violence. Sometimes the most honest
    response is to not resolve, to leave the gap visible.
    """

    id: str
    context: str
    why_silence_is_appropriate: str
    what_synthesis_would_paper_over: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SynthesisAnalysis:
    """Analysis of how a synthesis handled disagreement."""

    synthesis_text: str
    false_consensus_detected: bool
    dissolution_types: list[DissolutionType]
    preserved_tensions: list[PreservedDissent]
    silence_markers: list[SilenceMarker]

    # Metrics
    dissent_preservation_score: float  # 0.0 = all dissent dissolved, 1.0 = all preserved
    honest_uncertainty_score: float  # Did it admit what it doesn't know?
    frame_acknowledgment_score: float  # Did it acknowledge competing frames?

    # The key insight
    negative_space_quality: float  # How well did it preserve productive gaps?

    recommendations: list[str] = field(default_factory=list)


def extract_hedging_phrases(text: str) -> list[str]:
    """Extract phrases that indicate hedging or false balance."""
    hedging_patterns = [
        r"both sides have (valid |good )?points",
        r"there('s| is) merit (to|in) both",
        r"it (really )?depends on",
        r"reasonable people (can |might )?disagree",
        r"the truth (is |lies )?somewhere in (the )?between",
        r"we (need|should) (to )?consider all perspectives",
        r"there are arguments (on both sides|for and against)",
        r"ultimately[,]? it('s| is) a (matter of |question of )?preference",
    ]

    found = []
    text_lower = text.lower()
    for pattern in hedging_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            found.append(pattern)
    return found


def extract_resolution_claims(text: str) -> list[str]:
    """Extract claims that the disagreement has been resolved."""
    resolution_patterns = [
        r"(in )?conclusion[,:]",
        r"therefore[,]? we (can |should )",
        r"the (best |optimal |clear )?solution is",
        r"this resolves the",
        r"we('ve| have) reached (a |)consensus",
        r"the way forward is (clear|obvious)",
        r"combining (these|the) perspectives",
    ]

    found = []
    text_lower = text.lower()
    for pattern in resolution_patterns:
        if re.search(pattern, text_lower):
            found.append(pattern)
    return found


def detect_scope_escape(proposals: list[str], synthesis: str) -> bool:
    """Detect if synthesis changed the question to avoid tension."""
    # Simple heuristic: if synthesis introduces entirely new framing
    # that wasn't in the proposals, it might be escaping the scope

    proposal_words = set()
    for p in proposals:
        proposal_words.update(p.lower().split())

    synthesis_words = set(synthesis.lower().split())

    # Key terms in synthesis that weren't in proposals
    new_terms = synthesis_words - proposal_words

    # Filter to meaningful words (not stopwords)
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "of",
        "to",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
    }

    meaningful_new = new_terms - stopwords

    # If >30% of synthesis is new framing, might be scope escape
    if len(synthesis_words - stopwords) > 0:
        new_ratio = len(meaningful_new) / len(synthesis_words - stopwords)
        return new_ratio > 0.3
    return False


class NegativeSpacePreserver:
    """Preserves productive tension and engineered failure in debates.

    The goal is NOT to prevent synthesis, but to prevent FALSE synthesis
    that papers over genuine disagreement. Good synthesis acknowledges
    what it couldn't resolve.

    Example:
        preserver = NegativeSpacePreserver()

        analysis = preserver.analyze_synthesis(
            proposals=["We should use microservices", "Monolith is better"],
            critiques=["Microservices add operational complexity",
                      "Monolith doesn't scale team autonomy"],
            synthesis="Both approaches have merit, so we should use a modular monolith."
        )

        if analysis.false_consensus_detected:
            print("This synthesis is hiding real disagreement about:")
            for tension in analysis.preserved_tensions:
                print(f"  - {tension.original_position}")
    """

    def __init__(
        self,
        hedging_threshold: float = 0.3,
        resolution_threshold: float = 0.5,
        require_uncertainty_acknowledgment: bool = True,
    ):
        """Initialize the preserver.

        Args:
            hedging_threshold: How much hedging triggers false balance detection
            resolution_threshold: How many resolution claims trigger suspicion
            require_uncertainty_acknowledgment: Require explicit uncertainty
        """
        self.hedging_threshold = hedging_threshold
        self.resolution_threshold = resolution_threshold
        self.require_uncertainty = require_uncertainty_acknowledgment
        self._preserved_dissents: list[PreservedDissent] = []
        self._silence_markers: list[SilenceMarker] = []

    def analyze_synthesis(
        self,
        proposals: list[str],
        critiques: list[str],
        synthesis: str,
    ) -> SynthesisAnalysis:
        """Analyze how a synthesis handled the disagreement.

        Args:
            proposals: Original positions in the debate
            critiques: Criticisms raised
            synthesis: The attempted synthesis

        Returns:
            SynthesisAnalysis with detection results and recommendations
        """
        dissolution_types = []
        preserved_tensions = []
        silence_markers = []
        recommendations = []

        # Detect hedging/false balance
        hedging = extract_hedging_phrases(synthesis)
        if len(hedging) > len(synthesis.split()) * self.hedging_threshold / 100:
            dissolution_types.append(DissolutionType.FALSE_BALANCE)
            recommendations.append(
                "Synthesis uses false balance language. Consider: which position "
                "is actually stronger for this specific context?"
            )

        # Detect premature resolution claims
        resolutions = extract_resolution_claims(synthesis)
        if len(resolutions) > 2:
            dissolution_types.append(DissolutionType.PREMATURE_SYNTHESIS)
            recommendations.append(
                "Synthesis claims resolution too confidently. Consider: what "
                "genuine uncertainty remains?"
            )

        # Detect scope escape
        if detect_scope_escape(proposals, synthesis):
            dissolution_types.append(DissolutionType.SCOPE_ESCAPE)
            recommendations.append(
                "Synthesis may have changed the question. Consider: does this "
                "actually address the original disagreement?"
            )

        # Check for uncertainty acknowledgment
        uncertainty_phrases = [
            "we don't know",
            "uncertain",
            "unclear",
            "remains to be seen",
            "depends on factors we can't predict",
            "genuine disagreement",
            "irreducible tension",
            "trade-off",
            "no clear answer",
        ]
        has_uncertainty = any(phrase in synthesis.lower() for phrase in uncertainty_phrases)

        if self.require_uncertainty and not has_uncertainty:
            recommendations.append(
                "Synthesis doesn't acknowledge uncertainty. Consider: what "
                "do we genuinely not know here?"
            )

        # Identify what tensions should be preserved
        for i, proposal in enumerate(proposals):
            # Check if this position was adequately addressed or just dissolved
            position_words = set(proposal.lower().split())
            synthesis_words = set(synthesis.lower().split())

            # Simple overlap metric
            overlap = len(position_words & synthesis_words) / max(len(position_words), 1)

            if overlap < 0.3:
                # This position wasn't really engaged with
                preserved_tensions.append(
                    PreservedDissent(
                        id=f"dissent_{hashlib.sha256(proposal.encode()).hexdigest()[:8]}",
                        original_position=proposal,
                        why_it_resists_synthesis="Position not adequately addressed in synthesis",
                        tension_type=TensionType.FRAME_INCOMPATIBILITY,
                        strength=1.0 - overlap,
                    )
                )

        # Check for critiques that were ignored
        for critique in critiques:
            critique_words = set(critique.lower().split())
            synthesis_words = set(synthesis.lower().split())

            overlap = len(critique_words & synthesis_words) / max(len(critique_words), 1)

            if overlap < 0.2:
                # This critique wasn't addressed
                silence_markers.append(
                    SilenceMarker(
                        id=f"silence_{hashlib.sha256(critique.encode()).hexdigest()[:8]}",
                        context=critique[:100],
                        why_silence_is_appropriate="Critique raises issue synthesis cannot resolve",
                        what_synthesis_would_paper_over=critique,
                    )
                )

        # Calculate scores
        false_consensus = len(dissolution_types) > 0

        dissent_score = 1.0 - (len(preserved_tensions) / max(len(proposals), 1))
        uncertainty_score = 1.0 if has_uncertainty else 0.3
        frame_score = 1.0 - (
            0.2
            * len(
                [
                    d
                    for d in dissolution_types
                    if d in [DissolutionType.FALSE_BALANCE, DissolutionType.SCOPE_ESCAPE]
                ]
            )
        )

        # Negative space quality: how well did it preserve productive gaps?
        # Higher is better - we WANT some unresolved tension
        negative_space_quality = (
            0.3 * (len(preserved_tensions) / max(len(proposals), 1))
            + 0.3 * (len(silence_markers) / max(len(critiques), 1))
            + 0.4 * (1.0 if has_uncertainty else 0.0)
        )

        return SynthesisAnalysis(
            synthesis_text=synthesis,
            false_consensus_detected=false_consensus,
            dissolution_types=dissolution_types,
            preserved_tensions=preserved_tensions,
            silence_markers=silence_markers,
            dissent_preservation_score=dissent_score,
            honest_uncertainty_score=uncertainty_score,
            frame_acknowledgment_score=frame_score,
            negative_space_quality=negative_space_quality,
            recommendations=recommendations,
        )

    def preserve_dissent(
        self,
        analysis: SynthesisAnalysis,
    ) -> list[PreservedDissent]:
        """Explicitly preserve dissents that should not be synthesized away.

        Args:
            analysis: Previous synthesis analysis

        Returns:
            List of preserved dissents for the decision record
        """
        for dissent in analysis.preserved_tensions:
            self._preserved_dissents.append(dissent)

        for silence in analysis.silence_markers:
            self._silence_markers.append(silence)

        return analysis.preserved_tensions

    def get_dissent_trail(self) -> dict[str, Any]:
        """Get the full trail of preserved dissents.

        This becomes part of the decision receipt - what we
        chose NOT to resolve, and why.
        """
        return {
            "preserved_dissents": [
                {
                    "id": d.id,
                    "position": d.original_position,
                    "why_preserved": d.why_it_resists_synthesis,
                    "tension_type": d.tension_type.value,
                    "strength": d.strength,
                    "timestamp": d.timestamp.isoformat(),
                }
                for d in self._preserved_dissents
            ],
            "silence_markers": [
                {
                    "id": s.id,
                    "context": s.context,
                    "why_silence": s.why_silence_is_appropriate,
                    "what_hidden": s.what_synthesis_would_paper_over,
                    "timestamp": s.timestamp.isoformat(),
                }
                for s in self._silence_markers
            ],
            "total_dissents": len(self._preserved_dissents),
            "total_silences": len(self._silence_markers),
        }

    def should_reject_synthesis(
        self,
        analysis: SynthesisAnalysis,
        min_negative_space: float = 0.2,
    ) -> tuple[bool, str]:
        """Determine if a synthesis should be rejected.

        Args:
            analysis: Synthesis analysis
            min_negative_space: Minimum required negative space quality

        Returns:
            Tuple of (should_reject, reason)
        """
        if (
            analysis.false_consensus_detected
            and analysis.negative_space_quality < min_negative_space
        ):
            return True, (
                f"Synthesis achieves false consensus by: "
                f"{', '.join(d.value for d in analysis.dissolution_types)}. "
                f"Negative space quality ({analysis.negative_space_quality:.2f}) "
                f"is below threshold ({min_negative_space})."
            )

        if len(analysis.preserved_tensions) > len(analysis.dissolution_types) * 2:
            return True, (
                f"Too many positions ({len(analysis.preserved_tensions)}) were "
                f"not adequately addressed. Synthesis papers over genuine disagreement."
            )

        return False, "Synthesis adequately preserves productive tension."


def create_honest_synthesis_prompt(analysis: SynthesisAnalysis) -> str:
    """Generate a prompt for creating more honest synthesis.

    When a synthesis is rejected for false consensus, this generates
    guidance for a better attempt.
    """
    prompt_parts = [
        "The previous synthesis was rejected for papering over genuine disagreement.",
        "",
        "Issues detected:",
    ]

    for dtype in analysis.dissolution_types:
        prompt_parts.append(f"  - {dtype.value}")

    prompt_parts.extend(
        [
            "",
            "Positions that were not adequately addressed:",
        ]
    )

    for tension in analysis.preserved_tensions:
        prompt_parts.append(f"  - {tension.original_position[:100]}")

    prompt_parts.extend(
        [
            "",
            "Critiques that were ignored:",
        ]
    )

    for silence in analysis.silence_markers:
        prompt_parts.append(f"  - {silence.what_synthesis_would_paper_over[:100]}")

    prompt_parts.extend(
        [
            "",
            "Requirements for honest synthesis:",
            "1. Acknowledge what genuinely remains uncertain",
            "2. Name the trade-offs rather than dissolving them",
            "3. If positions are incompatible, say so - don't pretend they combine",
            "4. Include what you're choosing NOT to resolve, and why",
            "5. Preserve the productive tension that makes this a real decision",
            "",
            "Create a synthesis that respects the negative space.",
        ]
    )

    return "\n".join(prompt_parts)
