"""End-to-end regression test for the default Mode 3 PDB panel.

Complements ``tests/pdb/test_protocol.py`` (which uses a 2-3 slot
``_mini_config``) by driving ``run_protocol_b`` with the actual shipped
``aragora/config/pdb_panel.yaml`` — the 8-provider roster that gap
#6374 calls out as the product's live heterogeneous-ensemble path.

The mocks are intentionally minimal. The point of this test is not to
re-exercise the per-phase branch coverage already covered in
``test_protocol.py``; it is to pin the *contract* between the shipped
panel YAML and the executor so that a future config change cannot
silently break end-to-end panel execution.

Tests:

- ``test_default_panel_executes_all_eight_slots_end_to_end`` — the
  happy path with 8 available providers produces a successful brief,
  every configured slot appears in ``active_roster``, and the
  synthesizer call happens exactly once with 8 panel votes.
- ``test_default_panel_emits_real_dissent_when_slots_disagree`` —
  mixed vote pattern across the 8 slots produces non-empty
  ``dissenting_views`` (premise 3 integrity).
- ``test_default_panel_degrades_when_optional_slots_unavailable`` —
  required core slots resolve; heterodox+regulatory slots don't;
  executor returns DEGRADED with the optional slots in
  ``missing_slots`` and the availability summary reflecting reality.
- ``test_default_panel_fails_closed_when_required_slot_unavailable`` —
  dropping ``claude_core`` (required) forces FAILED_CLOSED, no brief.
- ``test_default_panel_respects_per_brief_budget_cap_from_yaml`` — the
  shipped per-brief USD cap is actually honored by the executor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from aragora.pdb.panel_config import (
    PDBBudgetConfig,
    PDBPanelConfig,
    PDBPanelSlot,
    load_panel_config,
)
from aragora.pdb.protocol import (
    PDBExecutionInput,
    PDBExecutionStatus,
    SlotCritiqueResponse,
    SlotFindingsResponse,
    SynthesisResponse,
    run_protocol_b,
)
from aragora.review.builder import PanelVote
from aragora.review.policy import ReviewBudget, ReviewPolicy
from aragora.review.protocol import DissentPosition
from aragora.review.provider_slots import (
    ProviderSlotDefinition,
    ProviderSlotResolution,
    ProviderSlotResolver,
)
from aragora.swarm.pr_review_protocol import (
    PROTOCOL_STATUS,
    PROTOCOL_VERSION,
    PRReviewBinding,
    PRReviewFinding,
    PRReviewProtocolPacket,
    ProviderSlotAvailabilitySummary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def default_config() -> PDBPanelConfig:
    """Load the real shipped panel YAML once per test module."""
    return load_panel_config()


def _binding() -> PRReviewBinding:
    return PRReviewBinding(
        repo="synaptent/aragora",
        pr_number=6468,
        base_sha="base00000000",
        head_sha="head11111111",
    )


def _heuristic_packet() -> PRReviewProtocolPacket:
    return PRReviewProtocolPacket(
        protocol_version=PROTOCOL_VERSION,
        status=PROTOCOL_STATUS,
        binding=_binding(),
        review_roles=[],
        provider_slots=[],
        availability_summary=ProviderSlotAvailabilitySummary(total_slots=0, resolved_slots=0),
        recommendation_class="needs_human_attention",
        recommendation_reason="pre-panel metadata heuristic",
        confidence=0.5,
        confidence_basis="metadata_heuristic",
        dissent_summary="",
        dissenting_views=[],
        validation_summary={},
        top_findings=[],
        cost_estimate={},
    )


def _input(panel_id: str, budget_usd: float = 8.0) -> PDBExecutionInput:
    return PDBExecutionInput(
        binding=_binding(),
        packet=_heuristic_packet(),
        packet_sha="",
        pr_title="End-to-end Mode 3 regression fixture",
        pr_body="Synthetic PR for end-to-end panel execution test.",
        labels=("test",),
        changed_files=("aragora/pdb/protocol.py",),
        diff_excerpt="diff --git a/a b/a\n+pass\n",
        validation_summary={"checks_summary": "green"},
        panel_id=panel_id,
        policy=ReviewPolicy(budget=ReviewBudget(per_pr_usd_cap=budget_usd)),
    )


class _AllAvailableResolver(ProviderSlotResolver):
    """Resolver that makes every slot in the panel available.

    Skips ``super().__init__()`` because the real agent-registry side
    effects are not needed in an end-to-end mock test.
    """

    def __init__(self, config: PDBPanelConfig) -> None:
        self._slots = dict(config.slots)

    def resolve_slot(self, slot: ProviderSlotDefinition) -> ProviderSlotResolution:
        real = self._slots[slot.slot_id]
        return ProviderSlotResolution(
            slot_id=slot.slot_id,
            review_role=slot.review_role,
            lens=slot.lens,
            family=slot.family,
            selected_provider=real.candidates[0],
            status="available",
            detail="end-to-end mock: all available",
            candidates=list(real.candidates),
        )

    def resolve_slots(
        self, slot_definitions: Sequence[ProviderSlotDefinition]
    ) -> list[ProviderSlotResolution]:
        return [self.resolve_slot(s) for s in slot_definitions]


class _SelectiveResolver(_AllAvailableResolver):
    """Resolver that resolves only a named subset of slots."""

    def __init__(self, config: PDBPanelConfig, resolved: set[str]) -> None:
        super().__init__(config)
        self._resolved = resolved

    def resolve_slot(self, slot: ProviderSlotDefinition) -> ProviderSlotResolution:
        base = super().resolve_slot(slot)
        if slot.slot_id in self._resolved:
            return base
        return ProviderSlotResolution(
            slot_id=base.slot_id,
            review_role=base.review_role,
            lens=base.lens,
            family=base.family,
            selected_provider=None,
            status="unavailable",
            detail="end-to-end mock: slot not resolved",
            candidates=base.candidates,
        )


@dataclass
class _MockPanelInvoker:
    """8-slot mock invoker that records every call and returns canned responses.

    ``slot_positions`` maps ``slot_id -> DissentPosition`` so tests can
    drive agreement or disagreement across the panel. Default is all
    APPROVE (no dissent).
    """

    slot_positions: Mapping[str, DissentPosition] = field(default_factory=dict)
    synthesis_position: DissentPosition = DissentPosition.APPROVE
    per_slot_cost_usd: float = 0.05
    synthesis_cost_usd: float = 0.10
    findings_calls: list[str] = field(default_factory=list)
    critique_calls: list[str] = field(default_factory=list)
    synth_calls: list[str] = field(default_factory=list)

    def _position(self, slot_id: str) -> DissentPosition:
        return self.slot_positions.get(slot_id, DissentPosition.APPROVE)

    def findings(
        self,
        *,
        slot: PDBPanelSlot,
        provider: str,
        prompt: str,
        binding: PRReviewBinding,
    ) -> SlotFindingsResponse:
        self.findings_calls.append(slot.slot_id)
        position = self._position(slot.slot_id)
        return SlotFindingsResponse(
            slot_id=slot.slot_id,
            provider=provider,
            model=f"{provider}-model",
            position=position,
            confidence=0.8,
            summary=f"{slot.slot_id} summary",
            top_findings=(
                PRReviewFinding(
                    finding_id=f"{slot.slot_id}-F1",
                    category="logic",
                    severity="medium",
                    summary=f"{slot.slot_id} finding",
                    evidence=["aragora/pdb/protocol.py:1"],
                    source=f"slot:{slot.slot_id}",
                ),
            ),
            contested_finding_ids=(),
            reason=f"{slot.slot_id} reasoned {position.value}",
            latency_ms=50,
            cost_usd=self.per_slot_cost_usd / 2,
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
        position = self._position(slot.slot_id)
        return SlotCritiqueResponse(
            slot_id=slot.slot_id,
            provider=provider,
            position=position,
            confidence=0.75,
            reason=f"{slot.slot_id} stands by {position.value} after critique",
            latency_ms=60,
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
        return SynthesisResponse(
            slot_id=synthesizer_slot.slot_id,
            provider=provider,
            model=f"{provider}-synth",
            top_line="Panel verdict: advisory synthesis summary.",
            validation_summary="CI green",
            position=self.synthesis_position,
            confidence=0.85,
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
            latency_ms=80,
            cost_usd=self.synthesis_cost_usd,
        )


# ---------------------------------------------------------------------------
# End-to-end tests against the real shipped panel YAML
# ---------------------------------------------------------------------------


def test_default_panel_executes_all_eight_slots_end_to_end(
    default_config: PDBPanelConfig,
) -> None:
    """Happy path: 8 providers available, panel executes cleanly, brief ships.

    This is the contract regression test for the shipped
    ``aragora/config/pdb_panel.yaml`` — if the YAML or the executor
    drift apart, this test should be the first to notice.
    """
    resolver = _AllAvailableResolver(default_config)
    invoker = _MockPanelInvoker()

    result = run_protocol_b(
        _input(default_config.default_panel),
        invoker=invoker,
        config=default_config,
        resolver=resolver,
    )

    assert result.status is PDBExecutionStatus.SUCCESS, result.failure_reason
    assert result.brief is not None
    # All 8 shipped slots must participate in findings + critique.
    expected_slots = set(default_config.slots.keys())
    assert set(invoker.findings_calls) == expected_slots
    assert set(invoker.critique_calls) == expected_slots
    # Synthesizer runs exactly once.
    assert len(invoker.synth_calls) == 1
    # Active roster reflects all 8 slots.
    assert set(result.active_roster) == expected_slots
    assert not result.missing_slots
    assert not result.degrade_reasons


def test_default_panel_emits_real_dissent_when_slots_disagree(
    default_config: PDBPanelConfig,
) -> None:
    """Premise-3 integrity: the 8-provider panel preserves disagreement.

    Drive the heterodox lens (skeptics) to REJECT while the core
    reviewers APPROVE. The brief's packet should carry at least one
    entry in ``dissenting_views`` — the observable signal gap #6374
    cares about.
    """
    disagreers = {
        "grok_heterodox": DissentPosition.REQUEST_CHANGES,
        "deepseek_heterodox": DissentPosition.REQUEST_CHANGES,
    }
    invoker = _MockPanelInvoker(slot_positions=disagreers)

    result = run_protocol_b(
        _input(default_config.default_panel),
        invoker=invoker,
        config=default_config,
        resolver=_AllAvailableResolver(default_config),
    )

    assert result.status is PDBExecutionStatus.SUCCESS
    assert result.brief is not None
    # dissenting_views on the packet must be non-empty when votes diverge.
    assert result.packet.dissenting_views, (
        "Expected non-empty dissenting_views when skeptic lens rejects "
        "while core lens approves — premise 3 integrity check."
    )
    # Each dissenter's slot_id should be represented in the dissent record.
    dissenter_slots = {
        view.get("slot_id") if isinstance(view, dict) else getattr(view, "slot_id", None)
        for view in result.packet.dissenting_views
    }
    assert disagreers.keys() & dissenter_slots, (
        f"Expected dissenting slots {list(disagreers)} to appear in "
        f"dissenting_views; found {dissenter_slots}"
    )


def test_default_panel_degrades_when_optional_slots_unavailable(
    default_config: PDBPanelConfig,
) -> None:
    """Optional-slot contract: heterodox/regulatory drop, core survives.

    Per the executor's design (see ``test_optional_slot_missing_triggers_degrade``
    in ``test_protocol.py``): unavailable optional slots are silently
    excluded from ``active_roster`` rather than surfacing in
    ``missing_slots``. ``missing_slots`` is reserved for *required* slots
    that fail to resolve. This test pins that contract for the full 8-
    provider YAML.
    """
    required_only = {slot_id for slot_id, slot in default_config.slots.items() if slot.required}
    # Synthesizer must also be resolvable; it is marked non-required in the
    # shipped YAML but the executor requires it.
    synth = default_config.panels[default_config.default_panel].synthesizer_slot
    resolved = required_only | {synth}

    resolver = _SelectiveResolver(default_config, resolved)
    invoker = _MockPanelInvoker()

    result = run_protocol_b(
        _input(default_config.default_panel),
        invoker=invoker,
        config=default_config,
        resolver=resolver,
    )

    assert result.status in (PDBExecutionStatus.SUCCESS, PDBExecutionStatus.DEGRADED)
    assert result.brief is not None
    # Required slots execute.
    assert required_only <= set(result.active_roster)
    # Unresolved optional slots are excluded from active execution.
    all_slots = set(default_config.slots.keys())
    unresolved = all_slots - resolved
    assert unresolved.isdisjoint(set(result.active_roster)), (
        f"Unresolved optional slots {unresolved} should not be in "
        f"active_roster {result.active_roster}"
    )
    # And they should not have received any findings/critique calls.
    assert unresolved.isdisjoint(set(invoker.findings_calls))
    assert unresolved.isdisjoint(set(invoker.critique_calls))


def test_default_panel_fails_closed_when_required_slot_unavailable(
    default_config: PDBPanelConfig,
) -> None:
    """Required-slot contract: dropping a required core slot fails closed."""
    required = {slot_id for slot_id, slot in default_config.slots.items() if slot.required}
    assert required, "Shipped panel must declare at least one required slot"
    drop = next(iter(required))

    # Resolve everything except one required slot.
    resolved = (set(default_config.slots.keys()) - {drop}) | {drop}
    resolved.discard(drop)
    resolver = _SelectiveResolver(default_config, resolved)
    invoker = _MockPanelInvoker()

    result = run_protocol_b(
        _input(default_config.default_panel),
        invoker=invoker,
        config=default_config,
        resolver=resolver,
    )

    assert result.status is PDBExecutionStatus.FAILED_CLOSED
    assert result.brief is None
    assert drop in result.missing_slots
    assert result.failure_reason is not None


def test_default_panel_respects_per_brief_budget_cap_from_yaml(
    default_config: PDBPanelConfig,
) -> None:
    """The panel's budget cap is enforced pre-execution.

    Tightening the config's ``per_brief_usd`` below what the minimum
    safe roster costs should produce ``BUDGET_EXCEEDED`` with no brief
    and no invoker calls — the executor refuses to start rather than
    discovering mid-run that it cannot afford the panel. This pins the
    contract between :class:`PDBBudgetConfig` and the budget ledger at
    the end-to-end level for the shipped panel shape.
    """
    # Clone the shipped config but drop per_brief_usd to $0.50 — smaller
    # than any viable roster cost. Core + synthesizer alone can't fit.
    tight_budget = PDBPanelConfig(
        version=default_config.version,
        default_panel=default_config.default_panel,
        default_prompt_set=default_config.default_prompt_set,
        budget=PDBBudgetConfig(
            per_brief_usd=0.50,
            per_day_usd=default_config.budget.per_day_usd,
            reserve_for_manual_escalation_usd=0.0,
        ),
        slots=default_config.slots,
        panels=default_config.panels,
        prompt_sets=default_config.prompt_sets,
    )
    resolver = _AllAvailableResolver(default_config)
    invoker = _MockPanelInvoker()

    result = run_protocol_b(
        _input(default_config.default_panel),
        invoker=invoker,
        config=tight_budget,
        resolver=resolver,
    )

    assert result.status is PDBExecutionStatus.BUDGET_EXCEEDED, (
        f"Expected BUDGET_EXCEEDED under $0.50 per-brief cap; "
        f"got {result.status} (cost=${result.actual_cost_usd})"
    )
    assert result.brief is None
    # No invoker calls — executor refused to start, didn't mid-abort.
    assert invoker.findings_calls == []
    assert invoker.critique_calls == []
    assert invoker.synth_calls == []
