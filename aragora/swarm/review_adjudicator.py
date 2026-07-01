"""Review Adjudicator (M0, epic #8747 / issue #8748).

Escapes the PR-review "nitpick treadmill" — two adversarial reviewers who never
both PASS a substantial diff, each surfacing fresh advisory ``[P2]/[P3]`` nits —
by *adjudicating the findings themselves* instead of hand-refereeing round after
round. It **composes** existing Aragora primitives (it does NOT reimplement
them):

* :class:`aragora_debate.evidence.EvidenceQualityAnalyzer` — scores a finding's
  specificity / concreteness / evidence diversity.
* :func:`aragora.cli.commands.review_queue_comment_verdicts.has_blocking_finding_or_label`
  — the SAME ``[P0]/[P1]`` detector the gate already trusts (the hard bar).

The adjudicator fires ONLY on a stall (quorum unsatisfied, dissent present, at
least one supportive signal) and returns one of:

* ``SETTLE``   — all dissent is thin (below the groundedness bar): the treadmill
  escape. The thin findings are preserved for follow-up, never discarded.
* ``BLOCK``    — any ``[P0]/[P1]`` is present, OR a grounded advisory finding
  stands with no grounded counter-support. Never suppress a real finding.
* ``ESCALATE`` — grounded dissent AND grounded support both exist: a genuine
  two-sided material disagreement that a human should settle (a crux). M0 emits
  a summary escalation; crux-finder bridging (#8747) is a fast-follow.
* ``NOT_APPLICABLE`` — no dissent, or no supportive signal (genuine rejection is
  not a stall). The adjudicator abstains and the existing gate decides.

The groundedness score is a deterministic heuristic composed from the analyzer's
sub-scores (see :func:`score_groundedness`). It is intentionally a fast, cheap
first pass; a fast-follow can plug :class:`aragora.evaluation.llm_judge.LLMJudge`
as the scorer for "real intelligence" on the ambiguous middle. The bias is
conservative: SETTLE requires *every* dissenting finding to be clearly thin, so
the adjudicator errs toward BLOCK/ESCALATE rather than ever wrongly suppressing.

Gated behind ``ARAGORA_ENABLE_REVIEW_ADJUDICATOR`` (default OFF → byte-identical
to today).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, Sequence

from aragora.cli.commands.review_queue_comment_verdicts import (
    has_blocking_finding_or_label,
    highest_blocking_severity,
)

# Groundedness at or above this bar means a finding is substantive enough that it
# must NOT be silently suppressed. Tuned so a concrete finding (file:line + repro
# + specific values) scores well above it and a vague "would be nice" nit well
# below. Conservative by design; override via ``groundedness_bar=``.
DEFAULT_GROUNDEDNESS_BAR = 0.5

_ENABLE_FLAG = "ARAGORA_ENABLE_REVIEW_ADJUDICATOR"
_TRUTHY = {"1", "true", "yes", "on"}


def review_adjudicator_enabled() -> bool:
    """True when the adjudicator is explicitly enabled (default OFF)."""
    return os.getenv(_ENABLE_FLAG, "").strip().lower() in _TRUTHY


class AdjudicationVerdict(str, Enum):
    SETTLE = "adjudicated_settle"
    BLOCK = "adjudicated_block"
    ESCALATE = "adjudicated_escalate"
    NOT_APPLICABLE = "not_applicable"


class _ReviewFinding(Protocol):
    """Duck-typed EvidenceItem: family + body + verdict + supportive."""

    family: str
    body: str
    verdict: str

    @property
    def supportive(self) -> bool: ...


@dataclass
class FindingAssessment:
    family: str
    verdict: str
    is_blocking: bool
    highest_severity: str | None
    groundedness: float
    grounded: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "verdict": self.verdict,
            "is_blocking": self.is_blocking,
            "highest_severity": self.highest_severity,
            "groundedness": self.groundedness,
            "grounded": self.grounded,
        }


@dataclass
class AdjudicationResult:
    verdict: AdjudicationVerdict
    reason: str
    assessments: list[FindingAssessment] = field(default_factory=list)
    settled_findings: list[str] = field(default_factory=list)
    escalated_findings: list[str] = field(default_factory=list)
    blocking_findings: list[str] = field(default_factory=list)
    groundedness_bar: float = DEFAULT_GROUNDEDNESS_BAR

    def to_receipt_dict(self) -> dict[str, Any]:
        """Serializable adjudication summary for the DecisionReceipt/audit log."""
        return {
            "kind": "review_adjudication.v1",
            "verdict": self.verdict.value,
            "reason": self.reason,
            "groundedness_bar": self.groundedness_bar,
            "assessments": [a.to_dict() for a in self.assessments],
            "settled_findings": self.settled_findings,
            "escalated_findings": self.escalated_findings,
            "blocking_findings": self.blocking_findings,
        }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_groundedness(body: str, *, analyzer: Any | None = None) -> float:
    """Deterministic 0-1 groundedness heuristic for a single review finding.

    Composed from :class:`EvidenceQualityAnalyzer` sub-scores that empirically
    separate a concrete finding from a vague one on short review text:

    * ``specificity_score`` — file:line refs, concrete values, named symbols.
    * specific/(specific+vague) phrase ratio — concreteness vs hedging.
    * ``evidence_diversity`` — variety of evidence markers (repro, data, refs).

    The analyzer's own ``overall_quality`` is diluted on short findings (citation
    density / temporal relevance dominate), so we recombine the discriminating
    dimensions directly. Weights favor specificity and concreteness.
    """
    if analyzer is None:
        # Imported lazily so importing this module never hard-requires the
        # aragora-debate package (keeps the gate import-light).
        from aragora_debate.evidence import EvidenceQualityAnalyzer

        analyzer = EvidenceQualityAnalyzer()

    score = analyzer.analyze(body or "", agent="reviewer")
    total_phrases = score.specific_phrase_count + score.vague_phrase_count
    phrase_ratio = score.specific_phrase_count / total_phrases if total_phrases else 0.5
    composite = 0.5 * score.specificity_score + 0.3 * phrase_ratio + 0.2 * score.evidence_diversity
    return round(_clamp(composite), 4)


def adjudicate(
    items: Sequence[_ReviewFinding],
    *,
    groundedness_bar: float = DEFAULT_GROUNDEDNESS_BAR,
    scorer: Callable[[str], float] = score_groundedness,
) -> AdjudicationResult:
    """Adjudicate a stalled review over ``items`` (EvidenceItem-shaped).

    Pure and deterministic given a deterministic ``scorer``. Does not mutate the
    items and never contacts the network.
    """
    items = list(items)

    # Hard bar first: any [P0]/[P1] anywhere -> BLOCK, no scoring, no exceptions.
    blocking = [it for it in items if has_blocking_finding_or_label(it.body)]

    assessments: list[FindingAssessment] = []
    for it in items:
        is_blocking = has_blocking_finding_or_label(it.body)
        grounded_score = scorer(it.body)
        assessments.append(
            FindingAssessment(
                family=it.family,
                verdict=it.verdict,
                is_blocking=is_blocking,
                highest_severity=highest_blocking_severity(it.body),
                groundedness=grounded_score,
                grounded=grounded_score >= groundedness_bar,
            )
        )

    if blocking:
        return AdjudicationResult(
            verdict=AdjudicationVerdict.BLOCK,
            reason="blocking [P0]/[P1] finding present; hard bar — not adjudicable",
            assessments=assessments,
            blocking_findings=[it.body for it in blocking],
            groundedness_bar=groundedness_bar,
        )

    dissenting = [it for it in items if it.verdict == "changes_requested"]
    supportive = [it for it in items if getattr(it, "supportive", it.verdict == "pass")]

    if not dissenting:
        return AdjudicationResult(
            verdict=AdjudicationVerdict.NOT_APPLICABLE,
            reason="no dissent to adjudicate",
            assessments=assessments,
            groundedness_bar=groundedness_bar,
        )
    if not supportive:
        return AdjudicationResult(
            verdict=AdjudicationVerdict.NOT_APPLICABLE,
            reason="no supportive signal; genuine rejection, not a stall",
            assessments=assessments,
            groundedness_bar=groundedness_bar,
        )

    grounded_dissent = [it for it in dissenting if scorer(it.body) >= groundedness_bar]
    if not grounded_dissent:
        return AdjudicationResult(
            verdict=AdjudicationVerdict.SETTLE,
            reason=(
                "all dissent is advisory and below the groundedness bar; settling "
                "and filing the findings for follow-up"
            ),
            assessments=assessments,
            settled_findings=[it.body for it in dissenting],
            groundedness_bar=groundedness_bar,
        )

    grounded_support = [it for it in supportive if scorer(it.body) >= groundedness_bar]
    if grounded_support:
        return AdjudicationResult(
            verdict=AdjudicationVerdict.ESCALATE,
            reason=(
                "grounded dissent AND grounded support: a material two-sided "
                "disagreement (crux) — escalating for human settlement"
            ),
            assessments=assessments,
            escalated_findings=[it.body for it in grounded_dissent],
            groundedness_bar=groundedness_bar,
        )

    return AdjudicationResult(
        verdict=AdjudicationVerdict.BLOCK,
        reason=(
            "grounded advisory dissent stands with no grounded counter-support; "
            "the finding must be addressed"
        ),
        assessments=assessments,
        blocking_findings=[it.body for it in grounded_dissent],
        groundedness_bar=groundedness_bar,
    )
