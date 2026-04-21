"""Protocol B executor for PDB Mode 3.

Runs the bounded three-phase Protocol B pipeline:

1. Load + validate panel config.
2. Resolve configured slots through
   :class:`aragora.review.provider_slots.ProviderSlotResolver`.
3. Evaluate + reserve budget with :mod:`aragora.pdb.budget`.
4. Findings round — one prompt per active slot.
5. Critique round — one prompt per active slot, with peer findings
   as context.
6. Synthesis pass — one prompt through the configured synthesizer slot.
7. Return both packet-facing and brief-facing output via
   :func:`aragora.review.builder.build_brief`.

**PR 2 boundary**: this executor is transport-neutral. It does NOT
enqueue workers, touch storage, write artifacts, or call HTTP routes.
A :class:`ProviderInvoker` Protocol lets callers inject real or mocked
per-phase provider execution; PR 2's tests exercise the full pipeline
with a deterministic mock invoker.

Dissent preservation is the core contract. At each of the three layers
the spec requires — reviewer outputs, :class:`PRReviewProtocolPacket`,
and :class:`aragora.review.protocol.ReviewBrief`\\ ``.dissent`` — the
per-slot identity, lens, position, and reason survive.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

from aragora.pdb import budget as budget_mod
from aragora.pdb.budget import (
    PDBBudgetDecision,
    PDBBudgetLedger,
    PDBBudgetStatus,
    SlotCostEstimator,
    evaluate_budget,
)
from aragora.pdb.panel_config import (
    PDBPanelConfig,
    PDBPanelSlot,
    load_panel_config,
    panel_slots as panel_slots_for,
)
from aragora.pdb.prompts import critique_prompt, findings_prompt, synthesis_prompt
from aragora.review.builder import PanelVote, build_brief
from aragora.review.policy import ReviewPolicy
from aragora.review.protocol import (
    DissentPosition,
    Recommendation,
    ReviewBrief,
    ReviewRole,
    RoleFinding,
    SynthesisPolicy,
)
from aragora.review.provider_slots import (
    ProviderSlotAvailabilitySummary,
    ProviderSlotResolution,
    ProviderSlotResolver,
)
from aragora.swarm.pr_review_protocol import (
    PRReviewBinding,
    PRReviewFinding,
    PRReviewProtocolPacket,
    PROTOCOL_VERSION,
)

__all__ = [
    "PDBExecutionInput",
    "PDBExecutionResult",
    "PDBExecutionStatus",
    "ProviderInvoker",
    "SlotCritiqueResponse",
    "SlotFindingsResponse",
    "SynthesisResponse",
    "run_protocol_b",
    "STATUS_PANEL_EXECUTED",
    "STATUS_PANEL_DEGRADED",
    "STATUS_BUDGET_EXCEEDED",
    "STATUS_FAILED_CLOSED",
]


STATUS_PANEL_EXECUTED = "panel_executed"
STATUS_PANEL_DEGRADED = "panel_degraded"
STATUS_BUDGET_EXCEEDED = "budget_exceeded"
STATUS_FAILED_CLOSED = "failed_closed"

_RECOMMENDATION_CLASS_BY_DECISION: Mapping[Recommendation, str] = {
    Recommendation.APPROVE_CANDIDATE: "approve_candidate",
    Recommendation.REPAIR_FIRST: "repair_first",
    Recommendation.NEEDS_HUMAN_ATTENTION: "needs_human_attention",
}

# Map review_role string (stable across YAML + protocol.py) to the
# enum used by the brief builder. Slots whose review_role does not
# land in this table fall back to SKEPTIC so heterodox lenses still
# route through the builder without masquerading as core reviewers.
_REVIEW_ROLE_BY_NAME: Mapping[str, ReviewRole] = {
    "logic_reviewer": ReviewRole.LOGIC,
    "security_reviewer": ReviewRole.SECURITY,
    "maintainability_reviewer": ReviewRole.MAINTAINABILITY,
    "skeptic": ReviewRole.SKEPTIC,
    "synthesizer": ReviewRole.SYNTHESIZER,
}


# ---------------------------------------------------------------------------
# Status + input + output dataclasses
# ---------------------------------------------------------------------------


class PDBExecutionStatus(str, Enum):
    """Outcome of :func:`run_protocol_b`."""

    SUCCESS = "success"
    DEGRADED = "degraded"
    BUDGET_EXCEEDED = "budget_exceeded"
    FAILED_CLOSED = "failed_closed"


@dataclass(frozen=True, slots=True)
class PDBExecutionInput:
    """Transport-neutral payload accepted by :func:`run_protocol_b`.

    No file handles, no live git processes, no HTTP request objects.
    ``diff_excerpt`` is prepared text; PR3 callers that load diffs
    lazily MUST materialize them before invoking the executor.
    """

    binding: PRReviewBinding
    packet: PRReviewProtocolPacket
    packet_sha: str
    pr_title: str
    pr_body: str
    labels: tuple[str, ...]
    changed_files: tuple[str, ...]
    diff_excerpt: str
    validation_summary: Mapping[str, Any]
    panel_id: str
    policy: ReviewPolicy


@dataclass(frozen=True, slots=True)
class SlotFindingsResponse:
    """Structured output from one slot's findings phase."""

    slot_id: str
    provider: str
    model: str
    position: DissentPosition
    confidence: float
    summary: str
    top_findings: tuple[PRReviewFinding, ...]
    contested_finding_ids: tuple[str, ...]
    reason: str
    latency_ms: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class SlotCritiqueResponse:
    """Structured output from one slot's critique phase.

    ``position`` and ``confidence`` are the *post-peer-read* values
    that get piped into the synthesizer via :class:`PanelVote`. The
    findings-phase values live in :class:`SlotFindingsResponse`.
    """

    slot_id: str
    provider: str
    position: DissentPosition
    confidence: float
    reason: str
    agrees_with: tuple[str, ...] = ()
    disagrees_with: tuple[str, ...] = ()
    contested_finding_ids: tuple[str, ...] = ()
    latency_ms: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class SynthesisResponse:
    """Structured output from the synthesis pass."""

    slot_id: str
    provider: str
    model: str
    top_line: str
    validation_summary: str
    position: DissentPosition
    confidence: float
    preserved_dissent: tuple[Mapping[str, str], ...] = ()
    latency_ms: int = 0
    cost_usd: float = 0.0


class ProviderInvoker(Protocol):
    """Phase-aware provider driver.

    PR 2 only ships a :class:`Protocol` + a mocked test invoker. PR 3
    wires a real provider-backed implementation behind the same
    interface. The executor only depends on this Protocol, so wiring
    stays testable.
    """

    def findings(
        self,
        *,
        slot: PDBPanelSlot,
        provider: str,
        prompt: str,
        binding: PRReviewBinding,
    ) -> SlotFindingsResponse: ...

    def critique(
        self,
        *,
        slot: PDBPanelSlot,
        provider: str,
        prompt: str,
        peer_findings: Mapping[str, SlotFindingsResponse],
        binding: PRReviewBinding,
    ) -> SlotCritiqueResponse: ...

    def synthesize(
        self,
        *,
        synthesizer_slot: PDBPanelSlot,
        provider: str,
        prompt: str,
        votes: Sequence[PanelVote],
        binding: PRReviewBinding,
    ) -> SynthesisResponse: ...


@dataclass(frozen=True, slots=True)
class PDBExecutionResult:
    """Everything PR 3 needs to wire transport + artifact writing."""

    status: PDBExecutionStatus
    packet: PRReviewProtocolPacket
    brief: ReviewBrief | None
    budget_decision: PDBBudgetDecision
    active_roster: tuple[str, ...]
    missing_slots: tuple[str, ...]
    degrade_reasons: tuple[str, ...]
    failure_reason: str | None
    findings_by_slot: Mapping[str, SlotFindingsResponse]
    critiques_by_slot: Mapping[str, SlotCritiqueResponse]
    synthesis: SynthesisResponse | None
    resolutions: tuple[ProviderSlotResolution, ...]
    availability_summary: ProviderSlotAvailabilitySummary
    actual_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "active_roster": list(self.active_roster),
            "missing_slots": list(self.missing_slots),
            "degrade_reasons": list(self.degrade_reasons),
            "failure_reason": self.failure_reason,
            "budget_decision": self.budget_decision.to_dict(),
            "actual_cost_usd": round(self.actual_cost_usd, 4),
            "packet": self.packet.to_dict(),
            "brief": self.brief.to_dict() if self.brief is not None else None,
        }


# ---------------------------------------------------------------------------
# Executor entry point
# ---------------------------------------------------------------------------


def run_protocol_b(
    input: PDBExecutionInput,
    *,
    invoker: ProviderInvoker,
    config: PDBPanelConfig | None = None,
    ledger: PDBBudgetLedger | None = None,
    resolver: ProviderSlotResolver | None = None,
    cost_estimator: SlotCostEstimator | None = None,
    clock: "_Clock | None" = None,
) -> PDBExecutionResult:
    """Run Protocol B end-to-end and return the structured outcome.

    ``invoker`` is the only required non-input argument; everything
    else has a default. Tests pass an in-memory ledger, a seeded
    resolver, and a mock invoker for determinism.
    """
    cfg = config if config is not None else load_panel_config()
    if input.panel_id not in cfg.panels:
        raise ValueError(f"unknown panel_id {input.panel_id!r} (not in config.panels)")

    now = (clock or _default_clock)()

    slots = panel_slots_for(cfg, input.panel_id)
    panel = cfg.panels[input.panel_id]
    synthesizer_slot = cfg.slots[panel.synthesizer_slot]

    resolver = resolver or ProviderSlotResolver()
    resolutions = tuple(resolver.resolve_slot(slot.to_provider_slot_definition()) for slot in slots)
    availability = resolver.summarize(list(resolutions))
    provider_by_slot: dict[str, str] = {
        res.slot_id: res.selected_provider
        for res in resolutions
        if res.selected_provider is not None
    }

    # --- fail-closed checks on resolution ---------------------------------
    required_missing = tuple(
        slot.slot_id for slot in slots if slot.required and slot.slot_id not in provider_by_slot
    )
    synth_missing = synthesizer_slot.slot_id not in provider_by_slot

    if required_missing or synth_missing:
        reason_parts = []
        if required_missing:
            reason_parts.append("required slots unresolved: " + ", ".join(required_missing))
        if synth_missing:
            reason_parts.append(f"synthesizer slot {synthesizer_slot.slot_id!r} unresolved")
        failure_reason = "; ".join(reason_parts)
        return _fail_closed_result(
            input=input,
            panel_slots=slots,
            synthesizer_slot=synthesizer_slot,
            resolutions=resolutions,
            availability=availability,
            missing=required_missing + ((synthesizer_slot.slot_id,) if synth_missing else ()),
            failure_reason=failure_reason,
            cfg_budget=cfg.budget,
            ledger=ledger,
            cost_estimator=cost_estimator,
            now=now,
        )

    # --- budget evaluation ------------------------------------------------
    working_ledger = ledger or PDBBudgetLedger(daily_cap_usd=cfg.budget.per_day_usd)
    findings_available = tuple(slot for slot in slots if slot.slot_id in provider_by_slot)
    decision = evaluate_budget(
        findings_slots=findings_available,
        synthesizer_slot=synthesizer_slot,
        budget=cfg.budget,
        ledger=working_ledger,
        estimator=cost_estimator,
        review_budget=input.policy.budget,
    )

    if decision.status is PDBBudgetStatus.BUDGET_EXCEEDED:
        return _budget_exceeded_result(
            input=input,
            panel_slots=slots,
            synthesizer_slot=synthesizer_slot,
            resolutions=resolutions,
            availability=availability,
            decision=decision,
            now=now,
        )

    reservation = budget_mod.reserve(decision, working_ledger)

    # --- ensure heterogeneity floor --------------------------------------
    funded_slot_ids = set(decision.funded_slots)
    funded_slots = tuple(slot for slot in slots if slot.slot_id in funded_slot_ids)
    distinct_families = {slot.family for slot in funded_slots}
    if len(distinct_families) < 2:
        reservation.release_unused()
        return _fail_closed_result(
            input=input,
            panel_slots=slots,
            synthesizer_slot=synthesizer_slot,
            resolutions=resolutions,
            availability=availability,
            missing=tuple(decision.dropped_slots),
            failure_reason=(
                "funded panel collapsed below two model families "
                f"(families={sorted(distinct_families)})"
            ),
            cfg_budget=cfg.budget,
            ledger=working_ledger,
            cost_estimator=cost_estimator,
            now=now,
        )

    degrade_reasons: list[str] = []
    missing_slots: list[str] = [sid for sid in decision.dropped_slots]
    if missing_slots:
        degrade_reasons.append(decision.reason)

    # --- findings round ---------------------------------------------------
    findings_by_slot: dict[str, SlotFindingsResponse] = {}
    for slot in funded_slots:
        provider = provider_by_slot[slot.slot_id]
        prompt = findings_prompt(
            slot=slot,
            binding=input.binding,
            pr_title=input.pr_title,
            pr_body=input.pr_body,
            labels=input.labels,
            changed_files=input.changed_files,
            diff_excerpt=input.diff_excerpt,
            validation_summary=input.validation_summary,
        )
        try:
            findings_response = invoker.findings(
                slot=slot,
                provider=provider,
                prompt=prompt,
                binding=input.binding,
            )
        except Exception as exc:  # noqa: BLE001 — handled per-slot
            if slot.required or slot.slot_id == synthesizer_slot.slot_id:
                reservation.release_unused()
                return _fail_closed_result(
                    input=input,
                    panel_slots=slots,
                    synthesizer_slot=synthesizer_slot,
                    resolutions=resolutions,
                    availability=availability,
                    missing=(slot.slot_id,),
                    failure_reason=(f"required slot {slot.slot_id!r} findings failed: {exc}"),
                    cfg_budget=cfg.budget,
                    ledger=working_ledger,
                    cost_estimator=cost_estimator,
                    now=now,
                )
            degrade_reasons.append(f"optional slot {slot.slot_id!r} findings failed: {exc}")
            missing_slots.append(slot.slot_id)
            continue
        reservation.charge(findings_response.cost_usd)
        findings_by_slot[slot.slot_id] = findings_response

    active_findings_slots = tuple(slot for slot in funded_slots if slot.slot_id in findings_by_slot)
    if not active_findings_slots:
        reservation.release_unused()
        return _fail_closed_result(
            input=input,
            panel_slots=slots,
            synthesizer_slot=synthesizer_slot,
            resolutions=resolutions,
            availability=availability,
            missing=tuple(missing_slots),
            failure_reason="no findings completed; panel collapsed to empty",
            cfg_budget=cfg.budget,
            ledger=working_ledger,
            cost_estimator=cost_estimator,
            now=now,
        )

    # re-check heterogeneity floor after optional-slot findings failures
    post_findings_families = {slot.family for slot in active_findings_slots}
    if len(post_findings_families) < 2:
        reservation.release_unused()
        return _fail_closed_result(
            input=input,
            panel_slots=slots,
            synthesizer_slot=synthesizer_slot,
            resolutions=resolutions,
            availability=availability,
            missing=tuple(missing_slots),
            failure_reason=("active roster collapsed below two model families after findings"),
            cfg_budget=cfg.budget,
            ledger=working_ledger,
            cost_estimator=cost_estimator,
            now=now,
        )

    # --- critique round ---------------------------------------------------
    critiques_by_slot: dict[str, SlotCritiqueResponse] = {}
    peer_findings_payload: dict[str, str] = {
        sid: _summarize_findings_for_peers(resp) for sid, resp in findings_by_slot.items()
    }
    for slot in active_findings_slots:
        provider = provider_by_slot[slot.slot_id]
        prompt = critique_prompt(
            slot=slot,
            binding=input.binding,
            pr_title=input.pr_title,
            pr_body=input.pr_body,
            labels=input.labels,
            changed_files=input.changed_files,
            peer_findings=peer_findings_payload,
        )
        try:
            critique_response = invoker.critique(
                slot=slot,
                provider=provider,
                prompt=prompt,
                peer_findings=findings_by_slot,
                binding=input.binding,
            )
        except Exception as exc:  # noqa: BLE001 — handled per-slot
            if slot.required or slot.slot_id == synthesizer_slot.slot_id:
                reservation.release_unused()
                return _fail_closed_result(
                    input=input,
                    panel_slots=slots,
                    synthesizer_slot=synthesizer_slot,
                    resolutions=resolutions,
                    availability=availability,
                    missing=(slot.slot_id,),
                    failure_reason=(f"required slot {slot.slot_id!r} critique failed: {exc}"),
                    cfg_budget=cfg.budget,
                    ledger=working_ledger,
                    cost_estimator=cost_estimator,
                    now=now,
                )
            # Optional-slot critique timeout after findings succeeded is
            # an explicit degrade per spec §Degrade vs fail-closed.
            degrade_reasons.append(f"optional slot {slot.slot_id!r} critique failed: {exc}")
            missing_slots.append(slot.slot_id)
            continue
        reservation.charge(critique_response.cost_usd)
        critiques_by_slot[slot.slot_id] = critique_response

    # Panel votes: one per slot that completed both findings + critique.
    panel_votes = tuple(
        _panel_vote_for(
            slot=slot,
            findings=findings_by_slot[slot.slot_id],
            critique=critiques_by_slot[slot.slot_id],
        )
        for slot in active_findings_slots
        if slot.slot_id in critiques_by_slot
    )
    if not panel_votes:
        reservation.release_unused()
        return _fail_closed_result(
            input=input,
            panel_slots=slots,
            synthesizer_slot=synthesizer_slot,
            resolutions=resolutions,
            availability=availability,
            missing=tuple(missing_slots),
            failure_reason="no critique votes completed; cannot synthesize",
            cfg_budget=cfg.budget,
            ledger=working_ledger,
            cost_estimator=cost_estimator,
            now=now,
        )

    # Ensure synthesizer slot finished both phases; otherwise fail closed.
    if synthesizer_slot.slot_id not in critiques_by_slot:
        reservation.release_unused()
        return _fail_closed_result(
            input=input,
            panel_slots=slots,
            synthesizer_slot=synthesizer_slot,
            resolutions=resolutions,
            availability=availability,
            missing=(synthesizer_slot.slot_id,),
            failure_reason=(
                f"synthesizer slot {synthesizer_slot.slot_id!r} did not complete the critique phase"
            ),
            cfg_budget=cfg.budget,
            ledger=working_ledger,
            cost_estimator=cost_estimator,
            now=now,
        )

    # --- synthesis pass ---------------------------------------------------
    synth_provider = provider_by_slot[synthesizer_slot.slot_id]
    synth_prompt_text = synthesis_prompt(
        synthesizer_slot=synthesizer_slot,
        binding=input.binding,
        pr_title=input.pr_title,
        pr_body=input.pr_body,
        labels=input.labels,
        changed_files=input.changed_files,
        votes=panel_votes,
    )
    try:
        synthesis = invoker.synthesize(
            synthesizer_slot=synthesizer_slot,
            provider=synth_provider,
            prompt=synth_prompt_text,
            votes=panel_votes,
            binding=input.binding,
        )
    except Exception as exc:  # noqa: BLE001 — synthesizer is required
        reservation.release_unused()
        return _fail_closed_result(
            input=input,
            panel_slots=slots,
            synthesizer_slot=synthesizer_slot,
            resolutions=resolutions,
            availability=availability,
            missing=(synthesizer_slot.slot_id,),
            failure_reason=f"synthesis pass failed: {exc}",
            cfg_budget=cfg.budget,
            ledger=working_ledger,
            cost_estimator=cost_estimator,
            now=now,
        )
    reservation.charge(synthesis.cost_usd)

    # Build the brief via the landed builder. We append the synthesizer's
    # own SYNTHESIZER-role vote as an extra, which the builder permits.
    synthesizer_vote = _synthesizer_panel_vote(synthesizer_slot, synthesis)
    votes_for_brief = panel_votes + (synthesizer_vote,)
    brief = build_brief(
        votes=votes_for_brief,
        pr_number=input.binding.pr_number,
        repo=input.binding.repo,
        head_sha=input.binding.head_sha,
        base_sha=input.binding.base_sha,
        top_line=synthesis.top_line,
        validation_summary=synthesis.validation_summary,
        generated_at=now,
        synthesis_policy=SynthesisPolicy.SYNTHESIZER_AGENT,
        output_roles=None,
        total_cost_usd=round(reservation.actual_spend_usd, 4),
        total_wall_clock_ms=sum(resp.latency_ms for resp in findings_by_slot.values())
        + sum(resp.latency_ms for resp in critiques_by_slot.values())
        + synthesis.latency_ms,
    )

    # packet-facing projection
    status_value = (
        STATUS_PANEL_DEGRADED if missing_slots or degrade_reasons else STATUS_PANEL_EXECUTED
    )
    packet = _build_executed_packet(
        input=input,
        resolutions=resolutions,
        availability=availability,
        brief=brief,
        active_slot_ids=tuple(slot.slot_id for slot in active_findings_slots),
        missing_slots=tuple(missing_slots),
        findings_by_slot=findings_by_slot,
        critiques_by_slot=critiques_by_slot,
        synthesis=synthesis,
        status=status_value,
        decision=decision,
        actual_cost_usd=reservation.actual_spend_usd,
        slot_lookup={slot.slot_id: slot for slot in slots},
    )

    released = reservation.release_unused()
    _ = released  # the ledger already absorbed the release; kept for clarity

    exec_status = (
        PDBExecutionStatus.DEGRADED
        if status_value == STATUS_PANEL_DEGRADED
        else PDBExecutionStatus.SUCCESS
    )
    return PDBExecutionResult(
        status=exec_status,
        packet=packet,
        brief=brief,
        budget_decision=decision,
        active_roster=tuple(slot.slot_id for slot in active_findings_slots),
        missing_slots=tuple(missing_slots),
        degrade_reasons=tuple(degrade_reasons),
        failure_reason=None,
        findings_by_slot=dict(findings_by_slot),
        critiques_by_slot=dict(critiques_by_slot),
        synthesis=synthesis,
        resolutions=resolutions,
        availability_summary=availability,
        actual_cost_usd=round(reservation.actual_spend_usd, 4),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _panel_vote_for(
    *,
    slot: PDBPanelSlot,
    findings: SlotFindingsResponse,
    critique: SlotCritiqueResponse,
) -> PanelVote:
    role = _REVIEW_ROLE_BY_NAME.get(slot.review_role, ReviewRole.SKEPTIC)
    finding = RoleFinding(
        role=role,
        agent=f"{slot.slot_id}:{findings.provider}",
        model=findings.model,
        confidence=critique.confidence,
        finding_text=findings.summary or findings.reason,
        latency_ms=findings.latency_ms + critique.latency_ms,
        cost_usd=round(findings.cost_usd + critique.cost_usd, 6),
    )
    return PanelVote(
        finding=finding,
        position=critique.position,
        reason=critique.reason or findings.reason,
    )


def _synthesizer_panel_vote(
    slot: PDBPanelSlot,
    synthesis: SynthesisResponse,
) -> PanelVote:
    finding = RoleFinding(
        role=ReviewRole.SYNTHESIZER,
        agent=f"{slot.slot_id}:synthesizer:{synthesis.provider}",
        model=synthesis.model,
        confidence=synthesis.confidence,
        finding_text=synthesis.top_line,
        latency_ms=synthesis.latency_ms,
        cost_usd=synthesis.cost_usd,
    )
    return PanelVote(
        finding=finding,
        position=synthesis.position,
        reason=synthesis.top_line,
    )


def _summarize_findings_for_peers(resp: SlotFindingsResponse) -> str:
    rows = [
        f"recommendation: {resp.position.value}",
        f"confidence: {resp.confidence:.2f}",
        f"summary: {resp.summary}",
    ]
    for idx, finding in enumerate(resp.top_findings[:5], start=1):
        rows.append(
            f"  finding[{idx}] {finding.category}/{finding.severity} "
            f"{finding.finding_id}: {finding.summary}"
        )
    if resp.contested_finding_ids:
        rows.append("contested: " + ", ".join(resp.contested_finding_ids))
    return "\n".join(rows)


def _dissenting_views_for_packet(
    brief_recommendation: Recommendation,
    panel_votes: Sequence[PanelVote],
    critiques: Mapping[str, SlotCritiqueResponse],
    findings: Mapping[str, SlotFindingsResponse],
    slots: Mapping[str, PDBPanelSlot],
) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for vote in panel_votes:
        slot_id = vote.finding.agent.split(":", 1)[0]
        recommended_for_vote = _position_to_recommendation(vote.position)
        if recommended_for_vote is brief_recommendation:
            continue
        slot = slots.get(slot_id)
        contested = critiques[slot_id].contested_finding_ids if slot_id in critiques else ()
        views.append(
            {
                "slot_id": slot_id,
                "lens": slot.lens if slot else "unknown",
                "recommendation": vote.position.value,
                "reason": vote.reason,
                "contested_finding_ids": list(contested),
            }
        )
    return views


def _position_to_recommendation(position: DissentPosition) -> Recommendation:
    if position is DissentPosition.APPROVE:
        return Recommendation.APPROVE_CANDIDATE
    if position is DissentPosition.REQUEST_CHANGES:
        return Recommendation.REPAIR_FIRST
    return Recommendation.NEEDS_HUMAN_ATTENTION


def _aggregate_top_findings(
    findings_by_slot: Mapping[str, SlotFindingsResponse],
    limit: int = 5,
) -> list[PRReviewFinding]:
    """Aggregate per-slot findings into the packet's top_findings array.

    Preserves per-slot origin by prefixing the ``finding_id`` with the
    slot id so duplicates across slots are never silently merged.
    """
    agg: list[PRReviewFinding] = []
    for slot_id, resp in findings_by_slot.items():
        for finding in resp.top_findings:
            tagged_id = (
                finding.finding_id
                if finding.finding_id.startswith(f"{slot_id}:")
                else f"{slot_id}:{finding.finding_id}"
            )
            agg.append(
                PRReviewFinding(
                    finding_id=tagged_id,
                    category=finding.category,
                    severity=finding.severity,
                    summary=finding.summary,
                    evidence=list(finding.evidence),
                    source=f"slot:{slot_id}",
                )
            )
    # Severity-weighted stable sort so HIGH severity leads the packet.
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    agg.sort(key=lambda f: (severity_rank.get(f.severity, 3), f.finding_id))
    return agg[:limit]


def _build_executed_packet(
    *,
    input: PDBExecutionInput,
    resolutions: tuple[ProviderSlotResolution, ...],
    availability: ProviderSlotAvailabilitySummary,
    brief: ReviewBrief,
    active_slot_ids: tuple[str, ...],
    missing_slots: tuple[str, ...],
    findings_by_slot: Mapping[str, SlotFindingsResponse],
    critiques_by_slot: Mapping[str, SlotCritiqueResponse],
    synthesis: SynthesisResponse,
    status: str,
    decision: PDBBudgetDecision,
    actual_cost_usd: float,
    slot_lookup: Mapping[str, PDBPanelSlot],
) -> PRReviewProtocolPacket:
    # Re-derive panel votes keyed by the actual slot id (agent string
    # uses "slot_id:provider" prefix).
    panel_votes = [
        PanelVote(
            finding=RoleFinding(
                role=_REVIEW_ROLE_BY_NAME.get(slot_lookup[sid].review_role, ReviewRole.SKEPTIC),
                agent=f"{sid}:{findings_by_slot[sid].provider}",
                model=findings_by_slot[sid].model,
                confidence=critiques_by_slot[sid].confidence,
                finding_text=findings_by_slot[sid].summary,
                latency_ms=findings_by_slot[sid].latency_ms + critiques_by_slot[sid].latency_ms,
                cost_usd=round(
                    findings_by_slot[sid].cost_usd + critiques_by_slot[sid].cost_usd,
                    6,
                ),
            ),
            position=critiques_by_slot[sid].position,
            reason=critiques_by_slot[sid].reason,
        )
        for sid in active_slot_ids
        if sid in critiques_by_slot
    ]

    dissenting_views = _dissenting_views_for_packet(
        brief_recommendation=brief.recommendation,
        panel_votes=tuple(panel_votes),
        critiques=critiques_by_slot,
        findings=findings_by_slot,
        slots=slot_lookup,
    )
    dissent_count = len(dissenting_views)
    if dissent_count == 0:
        dissent_summary = "Panel reached consensus after one critique round; no dissent recorded."
    else:
        dissent_summary = (
            f"{dissent_count} slot(s) dissent from the top-line recommendation; "
            "machine-readable detail in dissenting_views."
        )

    top_findings = _aggregate_top_findings(findings_by_slot, limit=5)

    cost_estimate = {
        "currency": "USD",
        "estimated_total": round(decision.total_estimated_usd, 4),
        "actual_total": round(actual_cost_usd, 4),
        "per_brief_cap": round(decision.per_brief_cap_usd, 4),
        "per_day_remaining_before": round(decision.per_day_remaining_before_usd, 4),
        "funded_slots": list(decision.funded_slots),
        "dropped_slots": list(decision.dropped_slots),
        "basis": "protocol_b.v1 heterogeneous panel execution",
    }

    validation_summary = dict(input.validation_summary)
    validation_summary.setdefault("synthesis_top_line", synthesis.top_line)
    validation_summary.setdefault("synthesis_confidence", synthesis.confidence)
    validation_summary["active_roster"] = list(active_slot_ids)
    validation_summary["missing_slots"] = list(missing_slots)

    return PRReviewProtocolPacket(
        protocol_version=PROTOCOL_VERSION,
        status=status,
        binding=input.binding,
        review_roles=list(input.packet.review_roles),
        provider_slots=list(resolutions),
        availability_summary=availability,
        recommendation_class=_RECOMMENDATION_CLASS_BY_DECISION[brief.recommendation],
        recommendation_reason=synthesis.top_line,
        confidence=round(brief.overall_confidence, 2),
        confidence_basis=status,
        dissent_summary=dissent_summary,
        dissenting_views=dissenting_views,
        validation_summary=validation_summary,
        top_findings=top_findings,
        cost_estimate=cost_estimate,
    )


def _fail_closed_result(
    *,
    input: PDBExecutionInput,
    panel_slots: tuple[PDBPanelSlot, ...],
    synthesizer_slot: PDBPanelSlot,
    resolutions: tuple[ProviderSlotResolution, ...],
    availability: ProviderSlotAvailabilitySummary,
    missing: tuple[str, ...],
    failure_reason: str,
    cfg_budget: Any,
    ledger: PDBBudgetLedger | None,
    cost_estimator: SlotCostEstimator | None,
    now: str,
) -> PDBExecutionResult:
    decision = _placeholder_decision(
        panel_slots=panel_slots,
        synthesizer_slot=synthesizer_slot,
        cfg_budget=cfg_budget,
        ledger=ledger,
        cost_estimator=cost_estimator,
        status=PDBBudgetStatus.ALLOWED,  # budget wasn't the bind; resolution/heterogeneity was
        reason=failure_reason,
    )
    packet = PRReviewProtocolPacket(
        protocol_version=PROTOCOL_VERSION,
        status=STATUS_FAILED_CLOSED,
        binding=input.binding,
        review_roles=list(input.packet.review_roles),
        provider_slots=list(resolutions),
        availability_summary=availability,
        recommendation_class="needs_human_attention",
        recommendation_reason=failure_reason,
        confidence=0.0,
        confidence_basis=STATUS_FAILED_CLOSED,
        dissent_summary=(
            "Protocol B failed closed before producing a heterogeneous brief; "
            "no dissent is available to surface."
        ),
        dissenting_views=[],
        validation_summary=dict(input.validation_summary),
        top_findings=[],
        cost_estimate={
            "currency": "USD",
            "estimated_total": 0.0,
            "actual_total": 0.0,
            "basis": "fail_closed",
        },
    )
    return PDBExecutionResult(
        status=PDBExecutionStatus.FAILED_CLOSED,
        packet=packet,
        brief=None,
        budget_decision=decision,
        active_roster=(),
        missing_slots=missing,
        degrade_reasons=(),
        failure_reason=failure_reason,
        findings_by_slot={},
        critiques_by_slot={},
        synthesis=None,
        resolutions=resolutions,
        availability_summary=availability,
        actual_cost_usd=0.0,
    )


def _budget_exceeded_result(
    *,
    input: PDBExecutionInput,
    panel_slots: tuple[PDBPanelSlot, ...],
    synthesizer_slot: PDBPanelSlot,
    resolutions: tuple[ProviderSlotResolution, ...],
    availability: ProviderSlotAvailabilitySummary,
    decision: PDBBudgetDecision,
    now: str,
) -> PDBExecutionResult:
    packet = PRReviewProtocolPacket(
        protocol_version=PROTOCOL_VERSION,
        status=STATUS_BUDGET_EXCEEDED,
        binding=input.binding,
        review_roles=list(input.packet.review_roles),
        provider_slots=list(resolutions),
        availability_summary=availability,
        recommendation_class="needs_human_attention",
        recommendation_reason=decision.reason,
        confidence=0.0,
        confidence_basis=STATUS_BUDGET_EXCEEDED,
        dissent_summary=(
            "Protocol B was denied by budget before execution; no dissent is available."
        ),
        dissenting_views=[],
        validation_summary=dict(input.validation_summary),
        top_findings=[],
        cost_estimate={
            "currency": "USD",
            "estimated_total": round(decision.total_estimated_usd, 4),
            "actual_total": 0.0,
            "per_brief_cap": round(decision.per_brief_cap_usd, 4),
            "per_day_remaining_before": round(decision.per_day_remaining_before_usd, 4),
            "binding_cap": decision.binding_cap,
            "basis": "budget_exceeded",
        },
    )
    return PDBExecutionResult(
        status=PDBExecutionStatus.BUDGET_EXCEEDED,
        packet=packet,
        brief=None,
        budget_decision=decision,
        active_roster=(),
        missing_slots=decision.dropped_slots,
        degrade_reasons=(),
        failure_reason=decision.reason,
        findings_by_slot={},
        critiques_by_slot={},
        synthesis=None,
        resolutions=resolutions,
        availability_summary=availability,
        actual_cost_usd=0.0,
    )


def _placeholder_decision(
    *,
    panel_slots: tuple[PDBPanelSlot, ...],
    synthesizer_slot: PDBPanelSlot,
    cfg_budget: Any,
    ledger: PDBBudgetLedger | None,
    cost_estimator: SlotCostEstimator | None,
    status: PDBBudgetStatus,
    reason: str,
) -> PDBBudgetDecision:
    working_ledger = ledger or PDBBudgetLedger(daily_cap_usd=cfg_budget.per_day_usd)
    # Estimate but do not reserve; used only for reporting inside
    # fail-closed results.
    slot_estimates = {
        slot.slot_id: (cost_estimator or SlotCostEstimator()).estimate(slot) for slot in panel_slots
    }
    return PDBBudgetDecision(
        status=status,
        total_estimated_usd=0.0,
        per_brief_cap_usd=cfg_budget.per_brief_usd,
        per_day_cap_usd=cfg_budget.per_day_usd,
        per_day_spent_before_usd=working_ledger.spent_today_usd,
        per_day_remaining_before_usd=working_ledger.headroom_usd(),
        funded_slots=(),
        dropped_slots=tuple(slot.slot_id for slot in panel_slots),
        slot_estimates_usd=slot_estimates,
        synthesis_estimate_usd=budget_mod.DEFAULT_SYNTHESIS_COST_USD,
        reason=reason,
        binding_cap=None,
    )


# Clock helpers ---------------------------------------------------------


class _Clock(Protocol):
    def __call__(self) -> str: ...


def _default_clock() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
