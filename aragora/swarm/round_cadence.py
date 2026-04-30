"""Cross-round cadence metrics for the autonomous evolve-round loop.

Each evolve-round drops phase receipts into a directory like
``.aragora/evolve-round/<round-id>/dogfood/phase-<letter>-receipt.json``.
Over time, these receipts form an audit trail of how many phases each
round completed, how many PRs each round opened, and what the
phase-by-phase outcomes looked like.

This module is a pure offline aggregator: given a list of receipt
records (already loaded from disk by the caller), it returns one
``RoundCadenceSummary`` with deterministic totals plus a per-round
breakdown. No filesystem or GitHub calls are made here — the caller
loads the JSON files via ``scripts/render_round_cadence.py``.

The intent is so an operator can answer:

- How many rounds have we run, total?
- How many PRs did each round open?
- What's the per-round phase completion rate?
- Which rounds tripped a halt?
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Public phase status values we count.
PHASE_STATUSES = ("complete", "in_progress", "blocked", "skipped", "unknown")


@dataclass(frozen=True, slots=True)
class PhaseReceipt:
    """One loaded phase-receipt.json from a round."""

    round_id: str
    phase: str
    status: str
    pr_number: int | None = None
    halt_tripped: bool = False
    name: str = ""


@dataclass(frozen=True, slots=True)
class RoundSummary:
    """Per-round aggregation across all of a round's phase receipts."""

    round_id: str
    total_phases: int
    complete_phases: int
    blocked_phases: int
    pr_numbers: tuple[int, ...]
    halt_tripped: bool
    phases_by_letter: tuple[PhaseReceipt, ...]


@dataclass(frozen=True, slots=True)
class RoundCadenceSummary:
    """Aggregated cadence across all rounds."""

    total_rounds: int
    total_phases: int
    total_complete_phases: int
    total_blocked_phases: int
    total_prs_opened: int
    rounds_with_halt: int
    per_round: tuple[RoundSummary, ...] = field(default_factory=tuple)


def _normalize_phase(letter: str) -> str:
    """Phase letters are case-insensitive single chars."""
    s = (letter or "").strip().upper()
    return s[:1] if s else "?"


def aggregate_cadence(receipts: list[PhaseReceipt]) -> RoundCadenceSummary:
    """Compute one cadence summary from a list of phase receipts.

    Output is deterministic: rounds are sorted by ``round_id`` ascending,
    phases within each round are sorted by their (uppercased) letter.
    """
    by_round: dict[str, list[PhaseReceipt]] = {}
    for r in receipts:
        by_round.setdefault(r.round_id, []).append(r)

    per_round_list: list[RoundSummary] = []
    total_phases = 0
    total_complete = 0
    total_blocked = 0
    total_prs = 0
    rounds_with_halt = 0

    for round_id in sorted(by_round.keys()):
        group = by_round[round_id]
        group_sorted = tuple(sorted(group, key=lambda r: _normalize_phase(r.phase)))

        complete = sum(1 for r in group_sorted if r.status == "complete")
        blocked = sum(1 for r in group_sorted if r.status == "blocked")
        pr_nums = tuple(sorted({r.pr_number for r in group_sorted if isinstance(r.pr_number, int)}))
        any_halt = any(r.halt_tripped for r in group_sorted)

        per_round_list.append(
            RoundSummary(
                round_id=round_id,
                total_phases=len(group_sorted),
                complete_phases=complete,
                blocked_phases=blocked,
                pr_numbers=pr_nums,
                halt_tripped=any_halt,
                phases_by_letter=group_sorted,
            )
        )

        total_phases += len(group_sorted)
        total_complete += complete
        total_blocked += blocked
        total_prs += len(pr_nums)
        if any_halt:
            rounds_with_halt += 1

    return RoundCadenceSummary(
        total_rounds=len(per_round_list),
        total_phases=total_phases,
        total_complete_phases=total_complete,
        total_blocked_phases=total_blocked,
        total_prs_opened=total_prs,
        rounds_with_halt=rounds_with_halt,
        per_round=tuple(per_round_list),
    )


def render_markdown(summary: RoundCadenceSummary) -> str:
    """Deterministic Markdown rendering of the cadence summary.

    No clocks, no platform-specific paths, no random ordering —
    same input always produces the same output.
    """
    if summary.total_rounds == 0:
        return "# Round cadence\n\nNo rounds found.\n"

    completion_rate = (
        100.0 * summary.total_complete_phases / summary.total_phases
        if summary.total_phases > 0
        else 0.0
    )

    lines: list[str] = [
        "# Round cadence",
        "",
        f"**Total rounds:** {summary.total_rounds}",
        f"**Total phases run:** {summary.total_phases}",
        f"**Phases complete:** {summary.total_complete_phases} ({completion_rate:.1f}%)",
        f"**Phases blocked:** {summary.total_blocked_phases}",
        f"**Total PRs opened:** {summary.total_prs_opened}",
        f"**Rounds with halt-trip:** {summary.rounds_with_halt}",
        "",
        "## Per-round",
        "",
        "| Round | Phases | Complete | Blocked | PRs | Halt |",
        "| --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for rs in summary.per_round:
        halt_marker = "**HALT**" if rs.halt_tripped else "—"
        prs_str = ", ".join(f"#{n}" for n in rs.pr_numbers) or "—"
        lines.append(
            f"| `{rs.round_id}` | {rs.total_phases} | "
            f"{rs.complete_phases} | {rs.blocked_phases} | {prs_str} | {halt_marker} |"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "PHASE_STATUSES",
    "PhaseReceipt",
    "RoundCadenceSummary",
    "RoundSummary",
    "aggregate_cadence",
    "render_markdown",
]
