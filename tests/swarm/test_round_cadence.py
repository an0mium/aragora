"""Unit tests for round_cadence aggregator."""

from __future__ import annotations

import pytest

from aragora.swarm.round_cadence import (
    PhaseReceipt,
    aggregate_cadence,
    render_markdown,
)


def _r(
    round_id: str,
    phase: str,
    status: str = "complete",
    pr_number: int | None = None,
    halt_tripped: bool = False,
) -> PhaseReceipt:
    return PhaseReceipt(
        round_id=round_id,
        phase=phase,
        status=status,
        pr_number=pr_number,
        halt_tripped=halt_tripped,
    )


class TestAggregateCadence:
    def test_empty(self) -> None:
        out = aggregate_cadence([])
        assert out.total_rounds == 0
        assert out.total_phases == 0
        assert out.total_prs_opened == 0
        assert out.per_round == ()

    def test_single_round_single_phase(self) -> None:
        out = aggregate_cadence([_r("2026-04-30", "A", pr_number=6800)])
        assert out.total_rounds == 1
        assert out.total_phases == 1
        assert out.total_complete_phases == 1
        assert out.total_prs_opened == 1
        assert out.per_round[0].round_id == "2026-04-30"
        assert out.per_round[0].pr_numbers == (6800,)

    def test_round_phase_completion_rate(self) -> None:
        receipts = [
            _r("r1", "A", "complete"),
            _r("r1", "B", "complete"),
            _r("r1", "C", "blocked"),
        ]
        out = aggregate_cadence(receipts)
        assert out.total_complete_phases == 2
        assert out.total_blocked_phases == 1

    def test_pr_dedup_per_round(self) -> None:
        receipts = [
            _r("r1", "A", pr_number=100),
            _r("r1", "B", pr_number=100),  # same PR, doesn't double-count
            _r("r1", "C", pr_number=101),
        ]
        out = aggregate_cadence(receipts)
        assert out.per_round[0].pr_numbers == (100, 101)
        assert out.total_prs_opened == 2

    def test_pr_none_skipped(self) -> None:
        receipts = [_r("r1", "A", pr_number=None), _r("r1", "B", pr_number=None)]
        out = aggregate_cadence(receipts)
        assert out.per_round[0].pr_numbers == ()
        assert out.total_prs_opened == 0

    def test_halt_tripped_propagates(self) -> None:
        receipts = [_r("r1", "A"), _r("r1", "B", halt_tripped=True)]
        out = aggregate_cadence(receipts)
        assert out.per_round[0].halt_tripped is True
        assert out.rounds_with_halt == 1

    def test_rounds_sorted_alphabetically(self) -> None:
        # Pass rounds in non-sorted order
        receipts = [
            _r("2026-04-30c", "A"),
            _r("2026-04-30", "A"),
            _r("2026-04-30b", "A"),
        ]
        out = aggregate_cadence(receipts)
        assert tuple(rs.round_id for rs in out.per_round) == (
            "2026-04-30",
            "2026-04-30b",
            "2026-04-30c",
        )

    def test_phases_sorted_within_round(self) -> None:
        receipts = [
            _r("r1", "C"),
            _r("r1", "A"),
            _r("r1", "B"),
        ]
        out = aggregate_cadence(receipts)
        assert tuple(p.phase for p in out.per_round[0].phases_by_letter) == ("A", "B", "C")

    def test_case_insensitive_phase_normalize(self) -> None:
        receipts = [_r("r1", "a"), _r("r1", "B")]
        out = aggregate_cadence(receipts)
        # Both phases should be present, sorted as "A" "B"
        assert len(out.per_round[0].phases_by_letter) == 2

    def test_multi_round_totals(self) -> None:
        receipts = [
            _r("r1", "A", pr_number=10),
            _r("r1", "B", "blocked"),
            _r("r2", "A", pr_number=20),
            _r("r2", "B", pr_number=21),
        ]
        out = aggregate_cadence(receipts)
        assert out.total_rounds == 2
        assert out.total_phases == 4
        assert out.total_complete_phases == 3
        assert out.total_blocked_phases == 1
        assert out.total_prs_opened == 3


class TestRenderMarkdown:
    def test_empty(self) -> None:
        out = aggregate_cadence([])
        md = render_markdown(out)
        assert "No rounds found" in md

    def test_renders_header_and_rows(self) -> None:
        receipts = [
            _r("r1", "A", pr_number=100),
            _r("r1", "B", "blocked"),
            _r("r2", "A", pr_number=200, halt_tripped=True),
        ]
        out = aggregate_cadence(receipts)
        md = render_markdown(out)
        assert "Total rounds:** 2" in md
        assert "`r1`" in md and "`r2`" in md
        assert "**HALT**" in md
        assert "#100" in md and "#200" in md

    def test_completion_rate(self) -> None:
        receipts = [_r("r1", "A"), _r("r1", "B"), _r("r1", "C", "blocked")]
        out = aggregate_cadence(receipts)
        md = render_markdown(out)
        # 2/3 = 66.7%
        assert "66.7%" in md

    def test_render_is_deterministic(self) -> None:
        receipts = [_r("r1", "A", pr_number=1), _r("r2", "A", pr_number=2)]
        a = render_markdown(aggregate_cadence(receipts))
        b = render_markdown(aggregate_cadence(list(reversed(receipts))))
        assert a == b


class TestImmutability:
    def test_phase_receipt_frozen(self) -> None:
        r = _r("r1", "A")
        with pytest.raises(Exception):
            r.status = "blocked"  # type: ignore[misc]

    def test_round_summary_frozen(self) -> None:
        out = aggregate_cadence([_r("r1", "A")])
        with pytest.raises(Exception):
            out.per_round[0].round_id = "x"  # type: ignore[misc]
