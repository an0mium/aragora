"""Unit tests for the H1 multi-gate readiness aggregator."""

from __future__ import annotations

import pytest

from aragora.swarm.h1_readiness import (
    H1_GATES,
    GateInput,
    aggregate_readiness,
    render_markdown,
)


def _ready(gate_id: str) -> GateInput:
    return GateInput(gate_id=gate_id, status="ready", contract_doc=f"docs/status/{gate_id}.md")


def _advisory(gate_id: str) -> GateInput:
    return GateInput(gate_id=gate_id, status="advisory_in_progress")


def _blocked(gate_id: str) -> GateInput:
    return GateInput(gate_id=gate_id, status="blocked")


def _in_progress(gate_id: str) -> GateInput:
    return GateInput(gate_id=gate_id, status="in_progress")


class TestAggregateReadiness:
    def test_all_four_ready_overall_ready(self) -> None:
        inputs = [_ready(g) for g in H1_GATES]
        out = aggregate_readiness(inputs)
        assert out.overall_status == "ready"
        assert out.ready_count == 4
        assert "may be marked complete" in out.next_action

    def test_one_blocked_overrides_everything(self) -> None:
        inputs = [_ready("H1-01"), _ready("H1-02"), _blocked("H1-03"), _ready("H1-04")]
        out = aggregate_readiness(inputs)
        assert out.overall_status == "blocked"
        assert out.blocked_count == 1
        assert "H1-03" in out.next_action

    def test_two_blocked_lists_both(self) -> None:
        inputs = [_blocked("H1-01"), _ready("H1-02"), _blocked("H1-03"), _ready("H1-04")]
        out = aggregate_readiness(inputs)
        assert out.overall_status == "blocked"
        assert "H1-01" in out.next_action and "H1-03" in out.next_action

    def test_in_progress_takes_priority_over_advisory(self) -> None:
        inputs = [_ready("H1-01"), _advisory("H1-02"), _in_progress("H1-03"), _advisory("H1-04")]
        out = aggregate_readiness(inputs)
        assert out.overall_status == "in_progress"
        assert "H1-03" in out.next_action

    def test_advisory_when_no_blocker_or_in_progress(self) -> None:
        inputs = [_ready("H1-01"), _advisory("H1-02"), _ready("H1-03"), _ready("H1-04")]
        out = aggregate_readiness(inputs)
        assert out.overall_status == "advisory_in_progress"
        assert "H1-02" in out.next_action
        assert "advisory → canonical" in out.next_action

    def test_missing_gates_become_unknown(self) -> None:
        out = aggregate_readiness([_ready("H1-01")])
        assert out.unknown_count == 3
        assert out.overall_status == "unknown"
        assert "Run per-gate readiness checks" in out.next_action

    def test_empty_inputs_all_unknown(self) -> None:
        out = aggregate_readiness([])
        assert out.unknown_count == 4
        assert out.ready_count == 0
        assert out.overall_status == "unknown"

    def test_input_order_does_not_matter(self) -> None:
        a = aggregate_readiness([_ready(g) for g in H1_GATES])
        b = aggregate_readiness([_ready(g) for g in reversed(H1_GATES)])
        assert a == b

    def test_per_gate_always_in_canonical_order(self) -> None:
        out = aggregate_readiness([_ready(g) for g in reversed(H1_GATES)])
        assert tuple(gi.gate_id for gi in out.per_gate) == H1_GATES

    def test_evidence_preserved(self) -> None:
        gi = GateInput(
            gate_id="H1-01",
            status="ready",
            contract_doc="x.md",
            evidence={"dispatched": 16, "merged_only": [5126]},
        )
        out = aggregate_readiness([gi])
        retained = out.per_gate[0]
        assert retained.evidence["dispatched"] == 16
        assert retained.evidence["merged_only"] == [5126]


class TestRenderMarkdown:
    def test_renders_all_ready(self) -> None:
        out = aggregate_readiness([_ready(g) for g in H1_GATES])
        md = render_markdown(out)
        assert "# H1 multi-gate readiness" in md
        assert "`ready`" in md
        assert "Ready: **4**/4" in md
        for g in H1_GATES:
            assert g in md

    def test_renders_blocked(self) -> None:
        out = aggregate_readiness(
            [_ready("H1-01"), _ready("H1-02"), _blocked("H1-03"), _ready("H1-04")]
        )
        md = render_markdown(out)
        assert "`BLOCKED`" in md
        assert "Resolve blocker(s) on H1-03" in md

    def test_render_is_deterministic(self) -> None:
        inputs = [_advisory("H1-01"), _ready("H1-02"), _ready("H1-03"), _ready("H1-04")]
        a = render_markdown(aggregate_readiness(inputs))
        b = render_markdown(aggregate_readiness(list(reversed(inputs))))
        assert a == b

    def test_renders_contract_em_dash_for_missing(self) -> None:
        out = aggregate_readiness([])
        md = render_markdown(out)
        # Each row should have "—" in the contract column (4 unknown gates)
        assert md.count("| —") >= 4


class TestImmutability:
    def test_gate_input_frozen(self) -> None:
        gi = _ready("H1-01")
        with pytest.raises(Exception):
            gi.status = "blocked"  # type: ignore[misc]

    def test_multi_gate_readiness_frozen(self) -> None:
        out = aggregate_readiness([_ready(g) for g in H1_GATES])
        with pytest.raises(Exception):
            out.overall_status = "blocked"  # type: ignore[misc]
