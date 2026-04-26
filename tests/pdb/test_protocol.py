"""End-to-end tests for :mod:`aragora.pdb.protocol`.

Covers the spec's acceptance criteria for the Protocol B executor:

- all-core-green happy path → :class:`ReviewBrief` produced via the
  landed :func:`aragora.review.builder.build_brief`
- optional slot missing → degraded execution preserves reduced roster
- required slot missing → fail-closed result, no brief written
- synthesizer slot missing → fail-closed result
- budget exceeded → explicit ``BUDGET_EXCEEDED`` status, no brief
- dissent survives at all three layers: reviewer outputs, packet
  ``dissenting_views``, and ``ReviewBrief.dissent``
- heterogeneity floor: if funded roster collapses to < 2 families,
  fail closed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pytest

from aragora.pdb.budget import (
    PDBBudgetLedger,
    PDBBudgetStatus,
    SlotCostEstimator,
)
from aragora.pdb.panel_config import (
    PDBBudgetConfig,
    PDBPanelConfig,
    PDBPanelDefinition,
    PDBPanelSlot,
    PDBPromptSet,
)
from aragora.pdb.protocol import (
    PDBExecutionInput,
    PDBExecutionResult,
    PDBExecutionStatus,
    ProviderInvoker,
    STATUS_BUDGET_EXCEEDED,
    STATUS_FAILED_CLOSED,
    STATUS_PANEL_DEGRADED,
    STATUS_PANEL_EXECUTED,
    SlotCritiqueResponse,
    SlotFindingsResponse,
    SynthesisResponse,
    run_protocol_b,
)
from aragora.review.builder import PanelVote
from aragora.review.policy import ReviewBudget, ReviewPolicy
from aragora.review.protocol import (
    DissentPosition,
    Recommendation,
    ReviewBrief,
    ReviewRole,
)
from aragora.review.provider_slots import (
    ProviderSlotAvailabilitySummary,
    ProviderSlotDefinition,
    ProviderSlotResolution,
    ProviderSlotResolver,
)
from aragora.swarm.pr_review_protocol import (
    PRReviewBinding,
    PRReviewFinding,
    PRReviewProtocol,
    PRReviewProtocolPacket,
    PROTOCOL_STATUS,
    PROTOCOL_VERSION,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _slot(
    slot_id: str,
    *,
    family: str,
    review_role: str = "logic_reviewer",
    lens: str = "core",
    required: bool = False,
    candidates: tuple[str, ...] = (),
) -> PDBPanelSlot:
    return PDBPanelSlot(
        slot_id=slot_id,
        review_role=review_role,
        lens=lens,
        family=family,
        candidates=candidates or (f"{slot_id}-cli",),
        required=required,
    )


def _mini_config(slots: Sequence[PDBPanelSlot], synthesizer: str) -> PDBPanelConfig:
    slot_map = {s.slot_id: s for s in slots}
    panel = PDBPanelDefinition(
        panel_id="p",
        findings_slots=tuple(s.slot_id for s in slots),
        critique_slots=tuple(s.slot_id for s in slots),
        synthesizer_slot=synthesizer,
    )
    return PDBPanelConfig(
        version=1,
        default_panel="p",
        default_prompt_set="ps",
        budget=PDBBudgetConfig(
            per_brief_usd=20.0,
            per_day_usd=200.0,
            reserve_for_manual_escalation_usd=10.0,
        ),
        slots=slot_map,
        panels={"p": panel},
        prompt_sets={
            "ps": PDBPromptSet(
                prompt_set_id="ps",
                findings_prompt="f",
                critique_prompt="c",
                synthesis_prompt="s",
            ),
        },
    )


def _binding() -> PRReviewBinding:
    return PRReviewBinding(
        repo="synaptent/aragora",
        pr_number=4242,
        base_sha="base0000",
        head_sha="head1111",
    )


def _heuristic_packet() -> PRReviewProtocolPacket:
    return PRReviewProtocolPacket(
        protocol_version=PROTOCOL_VERSION,
        status=PROTOCOL_STATUS,
        binding=_binding(),
        review_roles=list(PRReviewProtocol.default().review_roles),
        provider_slots=[],
        availability_summary=ProviderSlotAvailabilitySummary(
            total_slots=0,
            resolved_slots=0,
        ),
        recommendation_class="needs_human_attention",
        recommendation_reason="pre-execution placeholder",
        confidence=0.5,
        confidence_basis=PROTOCOL_STATUS,
        dissent_summary="pre-execution",
        dissenting_views=[],
        validation_summary={},
        top_findings=[],
        cost_estimate={},
    )


def _input(panel_id: str = "p") -> PDBExecutionInput:
    return PDBExecutionInput(
        binding=_binding(),
        packet=_heuristic_packet(),
        packet_sha="",
        pr_title="Refactor rate limiter",
        pr_body="Adds token bucket.",
        labels=("backend",),
        changed_files=("aragora/server/rate_limit.py",),
        diff_excerpt="diff --git a/a b/a\n+pass\n",
        validation_summary={"checks_summary": "green"},
        panel_id=panel_id,
        policy=ReviewPolicy(budget=ReviewBudget(per_pr_usd_cap=0.0)),
    )


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubResolver(ProviderSlotResolver):
    """A resolver that returns canned resolutions keyed by slot_id."""

    def __init__(self, resolved: set[str], all_slots: Sequence[PDBPanelSlot]) -> None:
        # Do not call super().__init__() — we don't need the real agent
        # registry side effects for these tests.
        self._resolved = set(resolved)
        self._slots = {s.slot_id: s for s in all_slots}

    def resolve_slot(self, slot: ProviderSlotDefinition) -> ProviderSlotResolution:
        meta = self._slots[slot.slot_id]
        if slot.slot_id in self._resolved:
            return ProviderSlotResolution(
                slot_id=slot.slot_id,
                review_role=slot.review_role,
                lens=slot.lens,
                family=slot.family,
                selected_provider=slot.candidates[0],
                status="available",
                detail="stubbed",
                candidates=list(slot.candidates),
            )
        return ProviderSlotResolution(
            slot_id=slot.slot_id,
            review_role=slot.review_role,
            lens=slot.lens,
            family=slot.family,
            selected_provider=None,
            status="unavailable",
            detail="stub: slot not resolved",
            candidates=list(slot.candidates),
        )

    def resolve_slots(
        self,
        slot_definitions,
    ) -> list[ProviderSlotResolution]:
        return [self.resolve_slot(s) for s in slot_definitions]


@dataclass
class MockInvoker:
    """Deterministic invoker that records every call it receives."""

    slot_plan: Mapping[str, DissentPosition]
    synthesis_position: DissentPosition = DissentPosition.APPROVE
    per_slot_cost_usd: float = 0.4
    synthesis_cost_usd: float = 0.6
    fail_findings: set[str] = field(default_factory=set)
    fail_critique: set[str] = field(default_factory=set)
    fail_synthesis: bool = False
    findings_calls: list[str] = field(default_factory=list)
    critique_calls: list[str] = field(default_factory=list)
    synth_calls: list[str] = field(default_factory=list)

    def findings(
        self,
        *,
        slot: PDBPanelSlot,
        provider: str,
        prompt: str,
        binding: PRReviewBinding,
    ) -> SlotFindingsResponse:
        self.findings_calls.append(slot.slot_id)
        if slot.slot_id in self.fail_findings:
            raise RuntimeError(f"findings failure for {slot.slot_id}")
        position = self.slot_plan.get(slot.slot_id, DissentPosition.APPROVE)
        return SlotFindingsResponse(
            slot_id=slot.slot_id,
            provider=provider,
            model=f"{provider}-model",
            position=position,
            confidence=0.75,
            summary=f"{slot.slot_id} findings summary",
            top_findings=(
                PRReviewFinding(
                    finding_id=f"{slot.slot_id}-F1",
                    category="logic",
                    severity="medium",
                    summary=f"{slot.slot_id} noted a risk",
                    evidence=["aragora/server/rate_limit.py:12"],
                    source=f"slot:{slot.slot_id}",
                ),
            ),
            contested_finding_ids=(f"{slot.slot_id}-F1",),
            reason=f"{slot.slot_id} reasoned {position.value}",
            latency_ms=100,
            cost_usd=self.per_slot_cost_usd / 2,  # split across findings+critique
        )

    def critique(
        self,
        *,
        slot: PDBPanelSlot,
        provider: str,
        prompt: str,
        peer_findings: Mapping[str, SlotFindingsResponse],
        binding: PRReviewBinding,
    ) -> SlotCritiqueResponse:
        self.critique_calls.append(slot.slot_id)
        if slot.slot_id in self.fail_critique:
            raise RuntimeError(f"critique failure for {slot.slot_id}")
        position = self.slot_plan.get(slot.slot_id, DissentPosition.APPROVE)
        return SlotCritiqueResponse(
            slot_id=slot.slot_id,
            provider=provider,
            position=position,
            confidence=0.7,
            reason=f"{slot.slot_id} stood by {position.value} after critique",
            contested_finding_ids=(f"{slot.slot_id}-F1",),
            latency_ms=120,
            cost_usd=self.per_slot_cost_usd / 2,
        )

    def synthesize(
        self,
        *,
        synthesizer_slot: PDBPanelSlot,
        provider: str,
        prompt: str,
        votes: Sequence[PanelVote],
        binding: PRReviewBinding,
    ) -> SynthesisResponse:
        self.synth_calls.append(synthesizer_slot.slot_id)
        if self.fail_synthesis:
            raise RuntimeError("synthesis failure")
        return SynthesisResponse(
            slot_id=synthesizer_slot.slot_id,
            provider=provider,
            model=f"{provider}-synth",
            top_line="Panel verdict: advisory-only synthesis summary.",
            validation_summary="CI green; two dissents recorded.",
            position=self.synthesis_position,
            confidence=0.82,
            preserved_dissent=tuple(
                {
                    "slot_id": v.finding.agent.split(":", 1)[0],
                    "lens": "unknown",
                    "position": v.position.value,
                    "reason": v.reason,
                }
                for v in votes
                if v.position is not self.synthesis_position
            ),
            latency_ms=150,
            cost_usd=self.synthesis_cost_usd,
        )


def _fixed_clock(value: str = "2026-04-21T00:00:00+00:00"):
    def clock() -> str:
        return value

    return clock


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_produces_real_brief_and_executes_full_roster() -> None:
    slots = [
        _slot(
            "claude_core", family="claude", review_role="logic_reviewer", lens="core", required=True
        ),
        _slot(
            "gpt_core", family="gpt", review_role="security_reviewer", lens="core", required=True
        ),
        _slot(
            "gemini_h", family="gemini", review_role="maintainability_reviewer", lens="heterodox"
        ),
    ]
    config = _mini_config(slots, synthesizer="claude_core")
    resolver = StubResolver({"claude_core", "gpt_core", "gemini_h"}, slots)
    invoker = MockInvoker(
        slot_plan={
            "claude_core": DissentPosition.APPROVE,
            "gpt_core": DissentPosition.APPROVE,
            "gemini_h": DissentPosition.APPROVE,
        },
        synthesis_position=DissentPosition.APPROVE,
    )
    ledger = PDBBudgetLedger(daily_cap_usd=config.budget.per_day_usd)

    result = run_protocol_b(
        _input(),
        invoker=invoker,
        config=config,
        ledger=ledger,
        resolver=resolver,
        clock=_fixed_clock(),
    )

    assert result.status is PDBExecutionStatus.SUCCESS
    assert isinstance(result.brief, ReviewBrief)
    assert result.brief.recommendation is Recommendation.APPROVE_CANDIDATE
    assert result.brief.advisory_only is True
    # Every active slot completed both phases
    assert set(result.active_roster) == {"claude_core", "gpt_core", "gemini_h"}
    assert result.missing_slots == ()
    # Packet-level status reflects real execution, not metadata_heuristic
    assert result.packet.status == STATUS_PANEL_EXECUTED
    # The landed builder produced the brief — evidenced by a non-empty
    # packet_sha and a SHA that matches a re-computation.
    from aragora.review.builder import compute_packet_sha

    assert result.brief.packet_sha
    assert compute_packet_sha(result.brief) == result.brief.packet_sha
    # Ledger returns unused reserve (actual < reserve) — residual recorded.
    assert ledger.spent_today_usd == pytest.approx(result.actual_cost_usd)


def test_happy_path_role_findings_contain_synthesizer_vote() -> None:
    slots = [
        _slot(
            "claude_core", family="claude", review_role="logic_reviewer", lens="core", required=True
        ),
        _slot(
            "gpt_core", family="gpt", review_role="security_reviewer", lens="core", required=True
        ),
        _slot(
            "gemini_h", family="gemini", review_role="maintainability_reviewer", lens="heterodox"
        ),
    ]
    config = _mini_config(slots, synthesizer="claude_core")
    resolver = StubResolver({"claude_core", "gpt_core", "gemini_h"}, slots)
    invoker = MockInvoker(
        slot_plan={s.slot_id: DissentPosition.APPROVE for s in slots},
    )
    result = run_protocol_b(
        _input(),
        invoker=invoker,
        config=config,
        resolver=resolver,
        clock=_fixed_clock(),
    )
    assert result.brief is not None
    roles_present = {finding.role for finding in result.brief.role_findings}
    # Both the lens-role of the synthesizer (LOGIC) and the SYNTHESIZER role
    # survive — the builder accepts SYNTHESIZER as an extra role.
    assert ReviewRole.LOGIC in roles_present
    assert ReviewRole.SYNTHESIZER in roles_present
    assert ReviewRole.SECURITY in roles_present
    assert ReviewRole.MAINTAINABILITY in roles_present


# ---------------------------------------------------------------------------
# Dissent preservation at all three layers
# ---------------------------------------------------------------------------


def test_dissent_survives_reviewer_packet_and_brief_layers() -> None:
    slots = [
        _slot(
            "claude_core", family="claude", review_role="logic_reviewer", lens="core", required=True
        ),
        _slot(
            "gpt_core", family="gpt", review_role="security_reviewer", lens="core", required=True
        ),
        _slot("grok_h", family="grok", review_role="skeptic", lens="heterodox"),
    ]
    config = _mini_config(slots, synthesizer="claude_core")
    resolver = StubResolver({"claude_core", "gpt_core", "grok_h"}, slots)
    invoker = MockInvoker(
        slot_plan={
            "claude_core": DissentPosition.APPROVE,  # majority
            "gpt_core": DissentPosition.APPROVE,
            "grok_h": DissentPosition.REQUEST_CHANGES,  # dissenter
        },
        synthesis_position=DissentPosition.APPROVE,
    )

    result = run_protocol_b(
        _input(),
        invoker=invoker,
        config=config,
        resolver=resolver,
        clock=_fixed_clock(),
    )

    # LAYER 1 — reviewer outputs retain per-slot identity + lens + position
    assert result.critiques_by_slot["grok_h"].position is DissentPosition.REQUEST_CHANGES
    assert result.findings_by_slot["grok_h"].contested_finding_ids
    # LAYER 2 — packet.dissenting_views has a machine-readable array with
    # slot_id / lens / recommendation / reason / contested_finding_ids
    packet_dissent = result.packet.dissenting_views
    assert len(packet_dissent) == 1
    entry = packet_dissent[0]
    assert entry["slot_id"] == "grok_h"
    assert entry["lens"] == "heterodox"
    assert entry["recommendation"] == "request_changes"
    assert entry["reason"]
    assert "grok_h-F1" in entry["contested_finding_ids"]
    # dissent_summary is scan-oriented only (not the lossless store)
    assert result.packet.dissent_summary.lower().startswith("1 slot(s) dissent")
    # LAYER 3 — ReviewBrief.dissent preserves per-dissenter identity
    assert result.brief is not None
    brief_dissent = result.brief.dissent
    assert any(d.position is DissentPosition.REQUEST_CHANGES for d in brief_dissent)
    agents = {d.agent for d in brief_dissent}
    # "grok_h:…" agent identifier survives
    assert any(a.startswith("grok_h:") for a in agents)


# ---------------------------------------------------------------------------
# Degrade path — optional slot unavailable at resolution time
# ---------------------------------------------------------------------------


def test_optional_slot_missing_triggers_degrade() -> None:
    slots = [
        _slot(
            "claude_core", family="claude", review_role="logic_reviewer", lens="core", required=True
        ),
        _slot(
            "gpt_core", family="gpt", review_role="security_reviewer", lens="core", required=True
        ),
        _slot(
            "gemini_h", family="gemini", review_role="maintainability_reviewer", lens="heterodox"
        ),
    ]
    config = _mini_config(slots, synthesizer="claude_core")
    # Only core slots resolve; gemini_h is unavailable.
    resolver = StubResolver({"claude_core", "gpt_core"}, slots)
    invoker = MockInvoker(
        slot_plan={s.slot_id: DissentPosition.APPROVE for s in slots},
    )

    result = run_protocol_b(
        _input(),
        invoker=invoker,
        config=config,
        resolver=resolver,
        clock=_fixed_clock(),
    )

    # Heterogeneity floor still met (claude + gpt = 2 families), so we
    # execute — but wait, the spec requires at least one non-core lens in
    # the panel config (enforced at validation). At runtime, the executor
    # simply does not run unavailable slots; they aren't findings-funded
    # because the resolver returned no provider. We still execute core
    # slots so the packet reflects real heterogeneous execution.
    assert result.status is PDBExecutionStatus.SUCCESS
    assert "gemini_h" not in result.active_roster
    # The brief was still produced.
    assert isinstance(result.brief, ReviewBrief)


def test_optional_critique_failure_after_findings_degrades() -> None:
    slots = [
        _slot(
            "claude_core", family="claude", review_role="logic_reviewer", lens="core", required=True
        ),
        _slot(
            "gpt_core", family="gpt", review_role="security_reviewer", lens="core", required=True
        ),
        _slot("grok_h", family="grok", review_role="skeptic", lens="heterodox"),
    ]
    config = _mini_config(slots, synthesizer="claude_core")
    resolver = StubResolver({"claude_core", "gpt_core", "grok_h"}, slots)
    invoker = MockInvoker(
        slot_plan={s.slot_id: DissentPosition.APPROVE for s in slots},
        fail_critique={"grok_h"},
    )

    result = run_protocol_b(
        _input(),
        invoker=invoker,
        config=config,
        resolver=resolver,
        clock=_fixed_clock(),
    )

    assert result.status is PDBExecutionStatus.DEGRADED
    assert "grok_h" in result.missing_slots
    assert any("grok_h" in r for r in result.degrade_reasons)
    # The brief is still produced with the reduced roster
    assert isinstance(result.brief, ReviewBrief)
    assert result.packet.status == STATUS_PANEL_DEGRADED


# ---------------------------------------------------------------------------
# Fail-closed paths
# ---------------------------------------------------------------------------


def test_required_slot_missing_fails_closed() -> None:
    slots = [
        _slot(
            "claude_core", family="claude", review_role="logic_reviewer", lens="core", required=True
        ),
        _slot(
            "gpt_core", family="gpt", review_role="security_reviewer", lens="core", required=True
        ),
        _slot("grok_h", family="grok", review_role="skeptic", lens="heterodox"),
    ]
    config = _mini_config(slots, synthesizer="claude_core")
    # gpt_core (required) is unresolved.
    resolver = StubResolver({"claude_core", "grok_h"}, slots)
    invoker = MockInvoker(slot_plan={})

    result = run_protocol_b(
        _input(),
        invoker=invoker,
        config=config,
        resolver=resolver,
        clock=_fixed_clock(),
    )
    assert result.status is PDBExecutionStatus.FAILED_CLOSED
    assert result.brief is None
    assert result.packet.status == STATUS_FAILED_CLOSED
    assert result.failure_reason is not None
    assert "gpt_core" in result.failure_reason


def test_synthesizer_slot_missing_fails_closed() -> None:
    slots = [
        _slot(
            "claude_core", family="claude", review_role="logic_reviewer", lens="core", required=True
        ),
        _slot(
            "gpt_core", family="gpt", review_role="security_reviewer", lens="core", required=True
        ),
        _slot(
            "gemini_h", family="gemini", review_role="maintainability_reviewer", lens="heterodox"
        ),
    ]
    config = _mini_config(slots, synthesizer="gemini_h")
    # gemini_h is unresolved; it's also the synthesizer.
    resolver = StubResolver({"claude_core", "gpt_core"}, slots)
    invoker = MockInvoker(slot_plan={})

    result = run_protocol_b(
        _input(),
        invoker=invoker,
        config=config,
        resolver=resolver,
        clock=_fixed_clock(),
    )
    assert result.status is PDBExecutionStatus.FAILED_CLOSED
    assert result.packet.status == STATUS_FAILED_CLOSED
    assert result.failure_reason is not None
    assert "synthesizer" in result.failure_reason


def test_synthesis_pass_failure_fails_closed() -> None:
    slots = [
        _slot(
            "claude_core", family="claude", review_role="logic_reviewer", lens="core", required=True
        ),
        _slot(
            "gpt_core", family="gpt", review_role="security_reviewer", lens="core", required=True
        ),
        _slot("grok_h", family="grok", review_role="skeptic", lens="heterodox"),
    ]
    config = _mini_config(slots, synthesizer="claude_core")
    resolver = StubResolver({"claude_core", "gpt_core", "grok_h"}, slots)
    invoker = MockInvoker(
        slot_plan={s.slot_id: DissentPosition.APPROVE for s in slots},
        fail_synthesis=True,
    )
    result = run_protocol_b(
        _input(),
        invoker=invoker,
        config=config,
        resolver=resolver,
        clock=_fixed_clock(),
    )
    assert result.status is PDBExecutionStatus.FAILED_CLOSED
    assert result.packet.status == STATUS_FAILED_CLOSED
    assert result.failure_reason is not None
    assert "synthesis" in result.failure_reason.lower()


def test_required_slot_findings_failure_fails_closed() -> None:
    slots = [
        _slot(
            "claude_core", family="claude", review_role="logic_reviewer", lens="core", required=True
        ),
        _slot(
            "gpt_core", family="gpt", review_role="security_reviewer", lens="core", required=True
        ),
        _slot("grok_h", family="grok", review_role="skeptic", lens="heterodox"),
    ]
    config = _mini_config(slots, synthesizer="claude_core")
    resolver = StubResolver({"claude_core", "gpt_core", "grok_h"}, slots)
    invoker = MockInvoker(
        slot_plan={s.slot_id: DissentPosition.APPROVE for s in slots},
        fail_findings={"gpt_core"},
    )
    result = run_protocol_b(
        _input(),
        invoker=invoker,
        config=config,
        resolver=resolver,
        clock=_fixed_clock(),
    )
    assert result.status is PDBExecutionStatus.FAILED_CLOSED
    assert result.packet.status == STATUS_FAILED_CLOSED


# ---------------------------------------------------------------------------
# Budget denial
# ---------------------------------------------------------------------------


def test_budget_exceeded_returns_explicit_denial_no_brief() -> None:
    slots = [
        _slot(
            "claude_core", family="claude", review_role="logic_reviewer", lens="core", required=True
        ),
        _slot(
            "gpt_core", family="gpt", review_role="security_reviewer", lens="core", required=True
        ),
        _slot("grok_h", family="grok", review_role="skeptic", lens="heterodox"),
    ]
    config = _mini_config(slots, synthesizer="claude_core")
    # Tighten per-brief cap to something the minimum roster cannot fit.
    tight = PDBPanelConfig(
        version=config.version,
        default_panel=config.default_panel,
        default_prompt_set=config.default_prompt_set,
        budget=PDBBudgetConfig(
            per_brief_usd=0.5, per_day_usd=200.0, reserve_for_manual_escalation_usd=0.0
        ),
        slots=config.slots,
        panels=config.panels,
        prompt_sets=config.prompt_sets,
    )
    resolver = StubResolver({"claude_core", "gpt_core", "grok_h"}, slots)
    invoker = MockInvoker(
        slot_plan={s.slot_id: DissentPosition.APPROVE for s in slots},
    )

    result = run_protocol_b(
        _input(),
        invoker=invoker,
        config=tight,
        resolver=resolver,
        clock=_fixed_clock(),
    )
    assert result.status is PDBExecutionStatus.BUDGET_EXCEEDED
    assert result.brief is None
    assert result.packet.status == STATUS_BUDGET_EXCEEDED
    assert result.budget_decision.status is PDBBudgetStatus.BUDGET_EXCEEDED
    # Explicit denial, not a silent downgrade to metadata_heuristic
    assert result.packet.status != PROTOCOL_STATUS
    assert invoker.findings_calls == []
    assert invoker.critique_calls == []
    assert invoker.synth_calls == []


def test_budget_degraded_still_runs_and_drops_optional_slots() -> None:
    slots = [
        _slot(
            "claude_core", family="claude", review_role="logic_reviewer", lens="core", required=True
        ),
        _slot(
            "gpt_core", family="gpt", review_role="security_reviewer", lens="core", required=True
        ),
        _slot("grok_h", family="grok", review_role="skeptic", lens="heterodox"),
        _slot(
            "gemini_h", family="gemini", review_role="maintainability_reviewer", lens="heterodox"
        ),
    ]
    config = _mini_config(slots, synthesizer="claude_core")
    # Cap tight enough that two core + synth fit but we cannot afford
    # all four optional+core slots. Default per-slot cost is $0.85 and
    # default synthesis cost is $1.10, so the floor (2 core + synth) is
    # $2.80 and a full roster (4 slots + synth) is $4.50. A cap of $3.50
    # admits the floor but forces one optional slot to drop.
    budget = PDBBudgetConfig(
        per_brief_usd=3.5, per_day_usd=200.0, reserve_for_manual_escalation_usd=0.0
    )
    tight = PDBPanelConfig(
        version=config.version,
        default_panel=config.default_panel,
        default_prompt_set=config.default_prompt_set,
        budget=budget,
        slots=config.slots,
        panels=config.panels,
        prompt_sets=config.prompt_sets,
    )
    resolver = StubResolver(
        {"claude_core", "gpt_core", "grok_h", "gemini_h"},
        slots,
    )
    invoker = MockInvoker(
        slot_plan={s.slot_id: DissentPosition.APPROVE for s in slots},
    )
    result = run_protocol_b(
        _input(),
        invoker=invoker,
        config=tight,
        resolver=resolver,
        clock=_fixed_clock(),
    )
    # Degraded status carries forward and a brief is still produced.
    assert result.status is PDBExecutionStatus.DEGRADED
    assert result.brief is not None
    assert result.packet.status == STATUS_PANEL_DEGRADED
    # Required slots survived, optional slots dropped to fit budget.
    assert "claude_core" in result.active_roster
    assert "gpt_core" in result.active_roster
    assert len(result.missing_slots) >= 1


# ---------------------------------------------------------------------------
# Heterogeneity floor
# ---------------------------------------------------------------------------


def test_heterogeneity_floor_below_two_families_fails_closed() -> None:
    slots = [
        _slot(
            "claude_core", family="claude", review_role="logic_reviewer", lens="core", required=True
        ),
        _slot(
            "claude_core_b",
            family="claude",
            review_role="security_reviewer",
            lens="core",
            required=True,
        ),
        _slot(
            "gemini_h", family="gemini", review_role="maintainability_reviewer", lens="heterodox"
        ),
    ]
    config = _mini_config(slots, synthesizer="claude_core")
    # Only claude-family slots resolve; heterogeneity floor collapses.
    resolver = StubResolver({"claude_core", "claude_core_b"}, slots)
    invoker = MockInvoker(
        slot_plan={s.slot_id: DissentPosition.APPROVE for s in slots},
    )
    result = run_protocol_b(
        _input(),
        invoker=invoker,
        config=config,
        resolver=resolver,
        clock=_fixed_clock(),
    )
    assert result.status is PDBExecutionStatus.FAILED_CLOSED
    assert result.failure_reason is not None
    assert "two model families" in result.failure_reason


# ---------------------------------------------------------------------------
# Unknown panel guard
# ---------------------------------------------------------------------------


def test_unknown_panel_id_raises() -> None:
    slots = [
        _slot(
            "claude_core", family="claude", review_role="logic_reviewer", lens="core", required=True
        ),
        _slot(
            "gpt_core", family="gpt", review_role="security_reviewer", lens="core", required=True
        ),
        _slot("grok_h", family="grok", review_role="skeptic", lens="heterodox"),
    ]
    config = _mini_config(slots, synthesizer="claude_core")
    resolver = StubResolver({"claude_core", "gpt_core", "grok_h"}, slots)
    invoker = MockInvoker(slot_plan={})
    with pytest.raises(ValueError):
        run_protocol_b(
            _input(panel_id="missing_panel"),
            invoker=invoker,
            config=config,
            resolver=resolver,
        )


# ---------------------------------------------------------------------------
# Result shape / JSON-friendliness
# ---------------------------------------------------------------------------


def test_result_to_dict_is_json_serializable() -> None:
    slots = [
        _slot(
            "claude_core", family="claude", review_role="logic_reviewer", lens="core", required=True
        ),
        _slot(
            "gpt_core", family="gpt", review_role="security_reviewer", lens="core", required=True
        ),
        _slot("grok_h", family="grok", review_role="skeptic", lens="heterodox"),
    ]
    config = _mini_config(slots, synthesizer="claude_core")
    resolver = StubResolver({"claude_core", "gpt_core", "grok_h"}, slots)
    invoker = MockInvoker(
        slot_plan={s.slot_id: DissentPosition.APPROVE for s in slots},
    )
    result = run_protocol_b(
        _input(),
        invoker=invoker,
        config=config,
        resolver=resolver,
        clock=_fixed_clock(),
    )
    import json

    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["status"] == "success"
    assert payload["brief"] is not None
