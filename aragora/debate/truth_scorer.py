"""Persuasion vs Truth Scorer for debate arguments.

Scores the ratio of persuasion techniques to genuine evidence in any argument.
This goes beyond fallacy detection — it quantifies HOW MUCH of an argument's
persuasive force comes from evidence vs rhetoric.

Key insight: A well-structured argument can contain both evidence AND rhetoric.
This scorer doesn't flag rhetoric as "bad" — it measures the balance, so that
debate outcomes can weight evidence-backed claims higher.

Outputs:
- evidence_score: how evidence-backed the argument is (0-1)
- rhetoric_score: how rhetoric-heavy the argument is (0-1)
- truth_ratio: evidence / (evidence + rhetoric), the core metric
- evidence_items: specific evidence cited
- rhetoric_items: specific rhetorical techniques used
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class EvidenceType(Enum):
    """Types of evidence that support truth-seeking."""

    DATA = "data"  # numbers, statistics, measurements
    CITATION = "citation"  # references to studies, papers, reports
    LOGICAL_PROOF = "logical_proof"  # deductive/inductive chains
    EXAMPLE = "example"  # concrete instances supporting a general claim
    EXPERIMENT = "experiment"  # empirical test results
    EXPERT_WITH_DATA = "expert_with_data"  # expert opinion backed by data


class RhetoricType(Enum):
    """Types of rhetoric that persuade without evidence."""

    EMOTIONAL_APPEAL = "emotional_appeal"
    AUTHORITY_WITHOUT_DATA = "authority_without_data"
    SOCIAL_PROOF = "social_proof"  # "everyone agrees"
    ANCHORING = "anchoring"  # setting a reference point to bias judgment
    FRAMING = "framing"  # presenting same facts in misleading frame
    HEDGING = "hedging"  # excessive qualification to avoid commitment
    CONFIDENCE_DISPLAY = "confidence_display"  # asserting certainty without basis
    REPETITION = "repetition"  # repeating claim as if repetition = proof


@dataclass
class EvidenceItem:
    """A detected piece of evidence in an argument."""

    evidence_type: EvidenceType
    text: str
    strength: float = 0.5  # 0-1, how strong this piece of evidence is
    verifiable: bool = True


@dataclass
class RhetoricItem:
    """A detected rhetorical technique in an argument."""

    rhetoric_type: RhetoricType
    text: str
    intensity: float = 0.5  # 0-1, how strongly this technique is used


@dataclass
class TruthScore:
    """Complete truth-vs-persuasion assessment of an argument."""

    evidence_score: float  # 0-1, overall evidence backing
    rhetoric_score: float  # 0-1, overall rhetoric usage
    truth_ratio: float  # evidence / (evidence + rhetoric)
    evidence_items: list[EvidenceItem] = field(default_factory=list)
    rhetoric_items: list[RhetoricItem] = field(default_factory=list)
    overall_assessment: str = ""

    def to_dict(self) -> dict:
        return {
            "evidence_score": self.evidence_score,
            "rhetoric_score": self.rhetoric_score,
            "truth_ratio": self.truth_ratio,
            "evidence_count": len(self.evidence_items),
            "rhetoric_count": len(self.rhetoric_items),
            "overall_assessment": self.overall_assessment,
        }


# ── Detection Patterns ────────────────────────────────────────────

# Evidence indicators: patterns that suggest genuine evidence
EVIDENCE_PATTERNS: dict[EvidenceType, list[str]] = {
    EvidenceType.DATA: [
        r"\b\d+(\.\d+)?%",  # percentages
        r"\b\d+(\.\d+)?\s*(ms|seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",  # durations
        r"\b\d+(\.\d+)?\s*(MB|GB|TB|KB|bytes?|requests?|users?|items?)\b",  # quantities
        r"\bp\s*[<>=]\s*0\.\d+\b",  # p-values
        r"\b(increased|decreased|improved|reduced)\s+by\s+\d+",  # measured changes
        r"\b(average|median|mean|std|variance|correlation)\b",  # statistical terms
    ],
    EvidenceType.CITATION: [
        r"\b(according to|as shown in|as reported by|per)\b",
        r"\b(study|paper|research|report|survey|analysis)\s+(by|from|in)\b",
        r"\(\d{4}\)",  # year citations like (2024)
        r"\b(doi|isbn|arxiv|pmid)\b",
        r"\bhttps?://\S+",  # URLs
    ],
    EvidenceType.LOGICAL_PROOF: [
        r"\b(therefore|thus|hence|consequently|it follows that)\b",
        r"\b(if\s+.+\s+then\s+.+)\b",
        r"\b(because|since|given that|assuming that)\b.*\b(therefore|we can conclude)\b",
        r"\b(necessary|sufficient)\s+condition\b",
    ],
    EvidenceType.EXAMPLE: [
        r"\b(for example|for instance|such as|e\.g\.|consider the case)\b",
        r"\b(specifically|in particular|one example is)\b",
        r"\b(when we|in practice|in the real world)\b",
    ],
    EvidenceType.EXPERIMENT: [
        r"\b(test(ed|ing)?|experiment(ed)?|measured|benchmark(ed)?|profil(ed|ing))\b",
        r"\b(control group|treatment|sample size|n\s*=\s*\d+)\b",
        r"\b(reproduced|replicated|validated|verified)\b",
    ],
    EvidenceType.EXPERT_WITH_DATA: [
        r"\b(expert|researcher|scientist|professor)\b.*\b(found|showed|demonstrated)\b",
        r"\b(peer.reviewed|published)\b.*\b(evidence|findings|results)\b",
    ],
}

# Rhetoric indicators: patterns that suggest persuasion over evidence
RHETORIC_PATTERNS: dict[RhetoricType, list[str]] = {
    RhetoricType.EMOTIONAL_APPEAL: [
        r"\b(devastating|catastrophic|incredible|amazing|terrible|wonderful|shocking)\b",
        r"\b(imagine|picture this|think about how|feel)\b",
        r"\b(danger(ous)?|threat|risk|crisis|emergency|urgent)\b",
        r"\b(dream|hope|fear|worry|excited)\b",
    ],
    RhetoricType.AUTHORITY_WITHOUT_DATA: [
        r"\b(experts say|authorities agree|it is well known|common knowledge)\b",
        r"\b(obviously|clearly|undeniably|unquestionably|indisputably)\b",
        r"\b(no one would deny|everyone knows|it goes without saying)\b",
    ],
    RhetoricType.SOCIAL_PROOF: [
        r"\b(everyone|most people|the majority|consensus|widely accepted)\b",
        r"\b(growing number|increasingly|more and more|trend)\b.*\b(agree|accept|believe)\b",
        r"\b(popular|mainstream|standard|conventional)\s+(view|opinion|wisdom)\b",
    ],
    RhetoricType.ANCHORING: [
        r"\b(compared to|relative to|in contrast|unlike)\b.*\b(much|far|significantly)\b",
        r"\b(at least|at most|no more than|no less than)\b",
        r"\b(only|just|merely)\s+\d+",  # "only 5%" (minimizing)
        r"\b(as much as|up to|over)\s+\d+",  # "up to 90%" (maximizing)
    ],
    RhetoricType.FRAMING: [
        r"\b(the real question is|what we should really ask|the key issue)\b",
        r"\b(look at it this way|from another perspective|if you think about it)\b",
        r"\b(reframe|redefine|rethink|reconsider)\b",
    ],
    RhetoricType.HEDGING: [
        r"\b(might|maybe|perhaps|possibly|potentially|could|somewhat|fairly|rather)\b",
        r"\b(it seems|it appears|one might argue|it could be said)\b",
        r"\b(to some extent|in a way|sort of|kind of)\b",
    ],
    RhetoricType.CONFIDENCE_DISPLAY: [
        r"\b(absolutely|definitely|certainly|without a doubt|guaranteed)\b",
        r"\b(proven|established|settled|conclusive)\b",
        r"\b(the fact is|the truth is|the reality is|make no mistake)\b",
    ],
    RhetoricType.REPETITION: [],  # Detected by counting repeated phrases
}


class TruthScorer:
    """Scores the persuasion-to-truth ratio of debate arguments.

    Usage:
        scorer = TruthScorer()
        score = scorer.score("The data shows a 40% improvement...")
        print(f"Truth ratio: {score.truth_ratio:.2f}")

    The truth_ratio (0-1) is the key metric:
    - 1.0 = purely evidence-based argument
    - 0.5 = equal mix of evidence and rhetoric
    - 0.0 = purely rhetorical argument

    This can be used to weight debate votes: evidence-backed positions
    carry more weight in consensus determination.
    """

    def __init__(
        self,
        evidence_weight: float = 1.0,
        rhetoric_penalty: float = 1.0,
    ):
        self.evidence_weight = evidence_weight
        self.rhetoric_penalty = rhetoric_penalty

    def score(self, text: str) -> TruthScore:
        """Score an argument for truth vs persuasion content.

        Args:
            text: The argument text to analyze

        Returns:
            TruthScore with evidence/rhetoric breakdown and truth ratio
        """
        evidence_items = self._detect_evidence(text)
        rhetoric_items = self._detect_rhetoric(text)

        # Check for repetition (special case: detected by phrase frequency)
        repetition_items = self._detect_repetition(text)
        rhetoric_items.extend(repetition_items)

        # Compute scores
        evidence_score = self._compute_evidence_score(evidence_items)
        rhetoric_score = self._compute_rhetoric_score(rhetoric_items)

        # Truth ratio: evidence / (evidence + rhetoric)
        total = evidence_score * self.evidence_weight + rhetoric_score * self.rhetoric_penalty
        truth_ratio = (evidence_score * self.evidence_weight / total) if total > 0 else 0.5

        assessment = self._generate_assessment(truth_ratio, evidence_items, rhetoric_items)

        return TruthScore(
            evidence_score=evidence_score,
            rhetoric_score=rhetoric_score,
            truth_ratio=truth_ratio,
            evidence_items=evidence_items,
            rhetoric_items=rhetoric_items,
            overall_assessment=assessment,
        )

    def score_debate_round(self, contributions: list[dict[str, str]]) -> dict[str, TruthScore]:
        """Score all contributions in a debate round.

        Args:
            contributions: List of {"agent": name, "text": content} dicts

        Returns:
            Dict mapping agent name to their TruthScore
        """
        return {entry["agent"]: self.score(entry["text"]) for entry in contributions}

    # ── Detection ─────────────────────────────────────────────────

    def _detect_evidence(self, text: str) -> list[EvidenceItem]:
        """Detect evidence indicators in text."""
        items: list[EvidenceItem] = []
        text_lower = text.lower()

        for evidence_type, patterns in EVIDENCE_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    # Get surrounding context (up to 100 chars)
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end].strip()

                    items.append(
                        EvidenceItem(
                            evidence_type=evidence_type,
                            text=context,
                            strength=self._evidence_strength(evidence_type),
                            verifiable=evidence_type
                            in (EvidenceType.DATA, EvidenceType.CITATION, EvidenceType.EXPERIMENT),
                        )
                    )

        return items

    def _detect_rhetoric(self, text: str) -> list[RhetoricItem]:
        """Detect rhetorical techniques in text."""
        items: list[RhetoricItem] = []
        text_lower = text.lower()

        for rhetoric_type, patterns in RHETORIC_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end].strip()

                    items.append(
                        RhetoricItem(
                            rhetoric_type=rhetoric_type,
                            text=context,
                            intensity=self._rhetoric_intensity(rhetoric_type),
                        )
                    )

        return items

    def _detect_repetition(self, text: str) -> list[RhetoricItem]:
        """Detect argument-by-repetition (same phrase repeated 3+ times)."""
        items: list[RhetoricItem] = []
        # Extract significant phrases (3+ words)
        words = text.lower().split()
        if len(words) < 9:
            return items

        phrase_counts: dict[str, int] = {}
        for i in range(len(words) - 2):
            phrase = " ".join(words[i : i + 3])
            # Skip common filler phrases
            if phrase in ("in the the", "of the the", "and the the"):
                continue
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        for phrase, count in phrase_counts.items():
            if count >= 3:
                items.append(
                    RhetoricItem(
                        rhetoric_type=RhetoricType.REPETITION,
                        text=f'"{phrase}" repeated {count} times',
                        intensity=min(count / 5, 1.0),
                    )
                )

        return items

    # ── Scoring ───────────────────────────────────────────────────

    def _evidence_strength(self, evidence_type: EvidenceType) -> float:
        """Return strength weight for evidence type."""
        weights = {
            EvidenceType.DATA: 0.9,
            EvidenceType.CITATION: 0.8,
            EvidenceType.EXPERIMENT: 0.95,
            EvidenceType.LOGICAL_PROOF: 0.7,
            EvidenceType.EXAMPLE: 0.5,
            EvidenceType.EXPERT_WITH_DATA: 0.85,
        }
        return weights.get(evidence_type, 0.5)

    def _rhetoric_intensity(self, rhetoric_type: RhetoricType) -> float:
        """Return intensity weight for rhetoric type."""
        weights = {
            RhetoricType.EMOTIONAL_APPEAL: 0.8,
            RhetoricType.AUTHORITY_WITHOUT_DATA: 0.7,
            RhetoricType.SOCIAL_PROOF: 0.6,
            RhetoricType.ANCHORING: 0.5,
            RhetoricType.FRAMING: 0.6,
            RhetoricType.HEDGING: 0.3,
            RhetoricType.CONFIDENCE_DISPLAY: 0.7,
            RhetoricType.REPETITION: 0.5,
        }
        return weights.get(rhetoric_type, 0.5)

    def _compute_evidence_score(self, items: list[EvidenceItem]) -> float:
        """Compute overall evidence score from detected items."""
        if not items:
            return 0.0
        # Weighted average of strengths, capped at 1.0
        total_strength = sum(item.strength for item in items)
        return min(total_strength / max(len(items), 1) * min(len(items) / 2, 1.5), 1.0)

    def _compute_rhetoric_score(self, items: list[RhetoricItem]) -> float:
        """Compute overall rhetoric score from detected items."""
        if not items:
            return 0.0
        total_intensity = sum(item.intensity for item in items)
        return min(total_intensity / max(len(items), 1) * min(len(items) / 2, 1.5), 1.0)

    def _generate_assessment(
        self,
        truth_ratio: float,
        evidence_items: list[EvidenceItem],
        rhetoric_items: list[RhetoricItem],
    ) -> str:
        """Generate human-readable assessment."""
        if truth_ratio >= 0.8:
            quality = "Strongly evidence-based"
        elif truth_ratio >= 0.6:
            quality = "Mostly evidence-based with some rhetoric"
        elif truth_ratio >= 0.4:
            quality = "Mixed evidence and rhetoric"
        elif truth_ratio >= 0.2:
            quality = "Mostly rhetorical with some evidence"
        else:
            quality = "Primarily rhetorical"

        evidence_types = {e.evidence_type.value for e in evidence_items}
        rhetoric_types = {r.rhetoric_type.value for r in rhetoric_items}

        parts = [f"{quality} (truth ratio: {truth_ratio:.2f})."]
        if evidence_types:
            parts.append(f"Evidence types: {', '.join(sorted(evidence_types))}.")
        if rhetoric_types:
            parts.append(f"Rhetoric types: {', '.join(sorted(rhetoric_types))}.")

        return " ".join(parts)
