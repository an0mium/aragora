"""BriefBuilder — deterministic ReviewBrief assembly from panel votes.

Pure-function synthesis layer between agent outputs and the ``ReviewBrief``
schema in ``aragora.review.protocol``. The caller (a successor PR's panel
orchestrator) hands in a tuple of ``PanelVote`` values, and BriefBuilder
produces a ``ReviewBrief`` deterministic in its inputs (same votes +
binding → same packet_sha).

Scope (#6306): closes the explicit-dissent + confidence-emission gap by
routing each ``SynthesisPolicy`` to a deterministic recommendation,
populating ``DissentingView`` for any vote whose position differs from
the brief's recommendation, and computing both ``overall_confidence``
and ``disagreement_score`` from the panel itself.

Out of scope (deliberate, per #6306 sequencing):
  - agent invocation, debate engine wiring (orchestrator successor PR)
  - cost/budget enforcement (#6305 layer)
  - receipt/evidence extension (#6307 layer)
  - converting metadata-heuristic ``PRReviewProtocolPacket`` into a
    ``ReviewBrief`` — those shapes are not lossless and should not
    masquerade as real heterogeneous briefs.

Determinism contract: same ``votes`` (in order) + same binding fields +
same ``synthesis_policy`` + same ``generated_at`` ⇒ same ``packet_sha``.
The caller is responsible for fixing ``generated_at``; the builder does
not call ``datetime.now()``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Mapping

from aragora.review.protocol import (
    DissentingView,
    DissentPosition,
    Recommendation,
    ReviewBrief,
    ReviewRole,
    RoleFinding,
    SynthesisPolicy,
)

# Position → Recommendation mapping (canonical, stable across the codebase).
#
# Identical mapping the queue layer assumes when surfacing
# packet recommendations, so a brief and a packet for the same PR never
# disagree on what an APPROVE position implies.
_POSITION_TO_RECOMMENDATION: dict[DissentPosition, Recommendation] = {
    DissentPosition.APPROVE: Recommendation.APPROVE_CANDIDATE,
    DissentPosition.REQUEST_CHANGES: Recommendation.REPAIR_FIRST,
    DissentPosition.DEFER: Recommendation.NEEDS_HUMAN_ATTENTION,
}


@dataclass(frozen=True, slots=True)
class PanelVote:
    """One panel member's complete contribution to a brief.

    Bundles the per-role finding (which becomes a row in
    ``ReviewBrief.role_findings``) with the agent's vote (used for
    synthesis + dissent emission).

    ``reason`` is required even on majority votes so the builder can
    construct a ``DissentingView`` later if the synthesis policy ends up
    treating this vote as dissenting (e.g., a high-confidence APPROVE
    that loses to a higher-confidence REQUEST_CHANGES under WEIGHTED).
    """

    finding: RoleFinding
    position: DissentPosition
    reason: str


def build_brief(
    *,
    votes: Iterable[PanelVote],
    pr_number: int,
    repo: str,
    head_sha: str,
    base_sha: str,
    top_line: str,
    validation_summary: str,
    generated_at: str,
    synthesis_policy: SynthesisPolicy,
    output_roles: tuple[ReviewRole, ...] | None = None,
    total_cost_usd: float = 0.0,
    total_wall_clock_ms: int = 0,
    findings_severity_counts: Mapping[str, int] | None = None,
) -> ReviewBrief:
    """Build a deterministic ReviewBrief from panel votes.

    Pure function. Same inputs → same ``packet_sha``. The caller is
    responsible for ensuring the votes come from a real heterogeneous
    panel; this layer does not enforce heterogeneity (lives in the
    orchestrator successor PR).

    ``output_roles`` enforces ``PRReviewProtocol.output_roles`` coverage
    when provided: each declared role MUST appear in exactly one vote.
    Panel members carrying roles NOT in ``output_roles`` (e.g., a
    ``SYNTHESIZER`` panelist used by ``SYNTHESIZER_AGENT`` policy) are
    permitted as extras and still appear in ``role_findings``. Pass
    ``None`` (default) to skip the coverage check.

    Raises:
      ValueError: if ``votes`` is empty.
      ValueError: if ``synthesis_policy`` is ``SYNTHESIZER_AGENT`` and
        the panel does not contain exactly one ``ReviewRole.SYNTHESIZER``.
      ValueError: if ``output_roles`` is provided and any declared role
        is missing or appears in more than one vote.
    """
    votes_tuple = tuple(votes)
    if not votes_tuple:
        raise ValueError("votes must be non-empty")
    if output_roles is not None:
        _validate_output_role_coverage(votes_tuple, output_roles)

    recommendation = _resolve_recommendation(votes_tuple, synthesis_policy)
    dissent = _build_dissent(votes_tuple, recommendation)
    overall_confidence = _aggregate_confidence(votes_tuple)
    disagreement_score = _disagreement_score(votes_tuple)
    role_findings = tuple(v.finding for v in votes_tuple)
    agent_roster = tuple(v.finding.agent for v in votes_tuple)
    severity_counts = dict(findings_severity_counts) if findings_severity_counts else {}

    def _make(packet_sha: str) -> ReviewBrief:
        return ReviewBrief(
            pr_number=pr_number,
            repo=repo,
            head_sha=head_sha,
            base_sha=base_sha,
            packet_sha=packet_sha,
            recommendation=recommendation,
            top_line=top_line,
            role_findings=role_findings,
            dissent=dissent,
            validation_summary=validation_summary,
            overall_confidence=overall_confidence,
            disagreement_score=disagreement_score,
            total_cost_usd=total_cost_usd,
            total_wall_clock_ms=total_wall_clock_ms,
            agent_roster=agent_roster,
            generated_at=generated_at,
            findings_severity_counts=severity_counts,
        )

    return _make(compute_packet_sha(_make("")))


def compute_packet_sha(brief: ReviewBrief) -> str:
    """Deterministic SHA-256 over the brief content, excluding ``packet_sha``.

    Implements the preimage rule documented in
    ``aragora.review.protocol.ReviewBrief``: serialize ``to_dict()``
    minus the ``"packet_sha"`` key as canonical JSON
    (``sort_keys=True``, no whitespace, ensure_ascii=False), UTF-8
    encode, hash with SHA-256, return hex.
    """
    payload = brief.to_dict()
    payload.pop("packet_sha", None)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_recommendation(
    votes: tuple[PanelVote, ...],
    policy: SynthesisPolicy,
) -> Recommendation:
    if policy is SynthesisPolicy.MAJORITY:
        return _majority(votes)
    if policy is SynthesisPolicy.WEIGHTED:
        return _weighted(votes)
    if policy is SynthesisPolicy.SYNTHESIZER_AGENT:
        return _synthesizer(votes)
    if policy is SynthesisPolicy.UNANIMOUS_OR_ESCALATE:
        return _unanimous(votes)
    raise ValueError(f"unsupported synthesis policy: {policy!r}")


def _majority(votes: tuple[PanelVote, ...]) -> Recommendation:
    counts: dict[DissentPosition, int] = {}
    for v in votes:
        counts[v.position] = counts.get(v.position, 0) + 1
    return _winner_or_escalate(counts)


def _clamp_confidence(value: float) -> float:
    """Clamp a per-finding confidence into ``[0.0, 1.0]``.

    Used by both ``_weighted`` (for recommendation policy) and
    ``_aggregate_confidence`` (for the emitted brief), so a malformed
    upstream confidence value cannot make the recommendation and the
    brief disagree on what counts as "in range."
    """
    return min(1.0, max(0.0, value))


def _weighted(votes: tuple[PanelVote, ...]) -> Recommendation:
    weights: dict[DissentPosition, float] = {}
    for v in votes:
        weights[v.position] = weights.get(v.position, 0.0) + _clamp_confidence(v.finding.confidence)
    return _winner_or_escalate(weights)


def _synthesizer(votes: tuple[PanelVote, ...]) -> Recommendation:
    synthesizers = [v for v in votes if v.finding.role is ReviewRole.SYNTHESIZER]
    if len(synthesizers) != 1:
        raise ValueError(
            "SynthesisPolicy.SYNTHESIZER_AGENT requires exactly one panel "
            f"member with role=ReviewRole.SYNTHESIZER; got {len(synthesizers)}"
        )
    return _POSITION_TO_RECOMMENDATION[synthesizers[0].position]


def _unanimous(votes: tuple[PanelVote, ...]) -> Recommendation:
    positions = {v.position for v in votes}
    if len(positions) == 1:
        return _POSITION_TO_RECOMMENDATION[positions.pop()]
    return Recommendation.NEEDS_HUMAN_ATTENTION


def _winner_or_escalate(scores: Mapping[DissentPosition, float]) -> Recommendation:
    """Pick the highest-scored position, or escalate on a tie."""
    if not scores:
        return Recommendation.NEEDS_HUMAN_ATTENTION
    top_score = max(scores.values())
    winners = [p for p, s in scores.items() if s == top_score]
    if len(winners) > 1:
        return Recommendation.NEEDS_HUMAN_ATTENTION
    return _POSITION_TO_RECOMMENDATION[winners[0]]


def _build_dissent(
    votes: tuple[PanelVote, ...],
    recommendation: Recommendation,
) -> tuple[DissentingView, ...]:
    """Any vote whose position doesn't map to the brief recommendation."""
    return tuple(
        DissentingView(
            agent=v.finding.agent,
            position=v.position,
            reason=v.reason,
            role=v.finding.role,
        )
        for v in votes
        if _POSITION_TO_RECOMMENDATION[v.position] is not recommendation
    )


def _aggregate_confidence(votes: tuple[PanelVote, ...]) -> float:
    """Mean of per-finding confidence, clamped per-input to [0.0, 1.0].

    ``ReviewBrief.overall_confidence`` is documented as 0.0..1.0; raw
    means could escape that range if upstream emits malformed values.
    Shares ``_clamp_confidence`` with ``_weighted`` so the recommendation
    policy and the emitted brief use the same definition of "in range."
    """
    return sum(_clamp_confidence(v.finding.confidence) for v in votes) / len(votes)


def _validate_output_role_coverage(
    votes: tuple[PanelVote, ...],
    output_roles: tuple[ReviewRole, ...],
) -> None:
    """Enforce ``PRReviewProtocol.output_roles``: one finding per declared role.

    Per the protocol docstring, a brief MUST cover every declared role
    exactly once. Missing → non-conformant; duplicated → ambiguous which
    finding to render in the role section.
    """
    counts: dict[ReviewRole, int] = {}
    for v in votes:
        counts[v.finding.role] = counts.get(v.finding.role, 0) + 1
    missing = [r for r in output_roles if counts.get(r, 0) == 0]
    duplicated = [r for r in output_roles if counts.get(r, 0) > 1]
    if missing or duplicated:
        parts = []
        if missing:
            parts.append(f"missing roles: {[r.value for r in missing]}")
        if duplicated:
            parts.append(f"duplicated roles: {[r.value for r in duplicated]}")
        raise ValueError("PRReviewProtocol.output_roles coverage violated — " + "; ".join(parts))


def _disagreement_score(votes: tuple[PanelVote, ...]) -> float:
    """1 - (largest position-share fraction).

    0.0 if every panel member votes the same; approaches 1.0 as the
    panel splits more evenly across positions.
    """
    counts: dict[DissentPosition, int] = {}
    for v in votes:
        counts[v.position] = counts.get(v.position, 0) + 1
    return round(1.0 - (max(counts.values()) / len(votes)), 4)
