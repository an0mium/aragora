"""Turn observed Codex/git state into a recommended advisory steering directive.

This is the pure decision core behind ``scripts/auto_steer_codex.py``: given a set
of observed signals (backlog size, stale merged-PR ledger entries, Claude-owned
PRs to keep Codex off), it composes a single :class:`SteeringDirective`
recommendation. It does **no I/O** -- the caller gathers signals (gh/git/digest)
and supplies the timestamp -- so it is fully deterministic and unit-testable.

By construction the recommendation can only ever ADD caution (the directive's own
``__post_init__`` rejects anything outside the steerable vocabulary), so a bad
recommendation can at worst waste a Codex cycle -- it can never loosen the gate.
The CLI defaults to ``--dry-run`` and writes nothing without an explicit
``--apply``; this module never writes either.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field

from .codex_steer import SteeringDirective


@dataclass(frozen=True)
class SteerSignals:
    """Observed inputs to a steering recommendation (gathered by the caller)."""

    issued_at: str
    open_codex_prs: int = 0
    backlog_threshold: int = 140
    stale_ledger_prs: Sequence[int] = field(default_factory=tuple)
    claude_owned_prs: Sequence[int] = field(default_factory=tuple)
    issued_by: str = "claude-auto-steer"


@dataclass(frozen=True)
class SteerRecommendation:
    """A recommended directive (or ``None``) plus human-readable rationale."""

    directive: SteeringDirective | None
    rationale: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "directive": self.directive.to_dict() if self.directive is not None else None,
            "rationale": list(self.rationale),
        }


def build_recommendation(signals: SteerSignals) -> SteerRecommendation:
    """Compose a monotonic-restrictive steering recommendation from signals.

    Detects, in priority order: backlog pressure (-> forbid ``create_pr`` so the
    fleet drains rather than creates), stale already-merged ledger entries
    (-> an advisory prune note), and Claude-owned PRs (-> pin them off-limits to
    Codex). Returns ``directive=None`` when nothing is steerable.
    """
    rationale: list[str] = []
    add_forbidden: list[str] = []
    notes: list[str] = []

    off_limits = sorted({pr for pr in signals.claude_owned_prs if pr > 0})
    stale = sorted({pr for pr in signals.stale_ledger_prs if pr > 0})

    if signals.open_codex_prs >= signals.backlog_threshold:
        add_forbidden.append("create_pr")
        rationale.append(
            f"backlog {signals.open_codex_prs} >= threshold {signals.backlog_threshold}: "
            "recommend drain-over-create"
        )
        notes.append(
            f"Backlog unhealthy ({signals.open_codex_prs} open codex PRs): DRAIN/REPAIR ONLY "
            "this cycle -- do not create net-new PRs/branches."
        )

    if stale:
        rationale.append(f"{len(stale)} stale already-merged ledger entries")
        notes.append(
            f"Conductor: prune stale ledger entries for already-merged PRs {stale}; skip "
            "merge_ready_prompt for any PR whose head != live origin/main head."
        )

    if off_limits:
        rationale.append(f"{len(off_limits)} Claude-owned PR(s) pinned off-limits: {off_limits}")

    if not add_forbidden and not off_limits and not notes:
        return SteerRecommendation(None, ("no steerable conditions detected; no directive",))

    directive = SteeringDirective(
        issued_by=signals.issued_by,
        issued_at=signals.issued_at,
        add_forbidden_actions=add_forbidden,
        off_limits_prs=off_limits,
        note=" ".join(notes) if notes else None,
    )
    return SteerRecommendation(directive, tuple(rationale))
