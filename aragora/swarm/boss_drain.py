"""Boss-loop drain driver: classify open PRs and run a bounded drain pass.

When the boss loop is over the open-PR cap, this turns the live PR queue into a
batch of :class:`~aragora.swarm.drain_policy.DrainCandidate`, runs the bounded
:func:`~aragora.swarm.drain_pass.run_drain_pass`, and executes the decisions.

Safe by construction:
- **MERGE never re-implements the gate.** A PR is only MERGE-eligible if an
  injected ``merge_authorized_fn`` (wired to ``settle_one_pr`` in the boss loop)
  says so — same authority, same tier/quorum/checks rules. Unknown-authority PRs
  default to NOT mergeable, so nothing slips through.
- **CLOSE only on truly-superseded** — empty (0 changed files) or an explicit
  superseded list. A merely-red PR is REPAIR, never closed.
- **off-limits is enforced** — branch-prefix matches (e.g. Factory's
  ``structex/``, ``claude/fusion-``) and pinned PR numbers are marked
  ``off_limits`` → LEAVE, never touched.
- **REPAIR is bounded** by the drain-pass caps (default 2/pass) → no worker storm.

All I/O is injected (``list_open_prs_fn``, ``view_pr_fn``, ``merge_authorized_fn``,
``execute_fn``) so the driver is unit-testable; the boss-loop hook supplies the
real gh-backed implementations.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from aragora.swarm.drain_pass import DrainPassPolicy
from aragora.swarm.drain_pass import DrainPassResult
from aragora.swarm.drain_pass import run_drain_pass
from aragora.swarm.drain_policy import DrainAction
from aragora.swarm.drain_policy import DrainCandidate

# Default branch prefixes that belong to OTHER fleets / lanes — never drained.
DEFAULT_OFF_LIMITS_PREFIXES: tuple[str, ...] = ("structex/", "claude/fusion-")

ListOpenPRsFn = Callable[
    [], Sequence[dict[str, Any]]
]  # -> PR view dicts (number, headRefName, ...)
ViewPRFn = Callable[
    [int], dict[str, Any] | None
]  # -> detailed PR view (changedFiles, mergeable...)
MergeAuthorizedFn = Callable[[int], tuple[bool, int]]  # (authorized, tier) — wired to settle_one_pr
ExecuteFn = Callable[[int, DrainAction], bool]


@dataclass(frozen=True)
class DrainContext:
    """Inputs that bound/guard a drain pass (off-limits + supersession)."""

    off_limits_prefixes: tuple[str, ...] = DEFAULT_OFF_LIMITS_PREFIXES
    off_limits_prs: frozenset[int] = field(default_factory=frozenset)
    superseded_prs: frozenset[int] = field(default_factory=frozenset)
    owned_prs: frozenset[int] = field(default_factory=frozenset)


def classify_candidate(
    view: dict[str, Any],
    ctx: DrainContext,
    *,
    merge_authorized: bool,
    tier: int,
) -> DrainCandidate:
    """Build a DrainCandidate from a gh PR view + an authority verdict. Pure.

    ``merge_authorized``/``tier`` come from the gate (settle_one_pr) for PRs we
    bothered to check; for the rest pass ``merge_authorized=False`` (and any
    tier) so they route to REPAIR/LEAVE, never an unchecked MERGE.
    """
    number = int(view.get("number", 0))
    branch = str(view.get("headRefName", ""))
    changed = view.get("changedFiles", view.get("changed_files", 1))
    has_changes = bool(changed and int(changed) > 0)
    off_limits = number in ctx.off_limits_prs or any(
        branch.startswith(p) for p in ctx.off_limits_prefixes
    )
    return DrainCandidate(
        pr=number,
        has_changes=has_changes,
        superseded=number in ctx.superseded_prs,
        off_limits=off_limits,
        owned_by_other_agent=number in ctx.owned_prs,
        required_checks_green=merge_authorized,  # authority bundles checks+quorum+mergeable
        quorum_satisfied=merge_authorized,
        mergeable=merge_authorized,
        tier=tier,
    )


def build_candidates(
    ctx: DrainContext,
    *,
    list_open_prs_fn: ListOpenPRsFn,
    view_pr_fn: ViewPRFn,
    merge_authorized_fn: MergeAuthorizedFn,
    max_classify: int = 60,
) -> list[DrainCandidate]:
    """Classify up to ``max_classify`` open PRs into DrainCandidates (bounded I/O).

    Off-limits / empty PRs are classified WITHOUT the (expensive) authority check
    — they route to LEAVE/CLOSE on cheap signal alone. Only PRs that are
    non-off-limits and have changes pay for a ``merge_authorized_fn`` call.
    """
    candidates: list[DrainCandidate] = []
    for view in list(list_open_prs_fn())[:max_classify]:
        number = int(view.get("number", 0))
        if number <= 0:
            continue
        branch = str(view.get("headRefName", ""))
        off_limits = number in ctx.off_limits_prs or any(
            branch.startswith(p) for p in ctx.off_limits_prefixes
        )
        # Cheap routes first: off-limits / owned / explicitly-superseded need no authority probe.
        if off_limits or number in ctx.owned_prs or number in ctx.superseded_prs:
            candidates.append(classify_candidate(view, ctx, merge_authorized=False, tier=4))
            continue
        detail = view_pr_fn(number) or view
        changed = detail.get("changedFiles", detail.get("changed_files", 1))
        if not (changed and int(changed) > 0):  # empty -> CLOSE_SUPERSEDED, no authority probe
            candidates.append(classify_candidate(detail, ctx, merge_authorized=False, tier=4))
            continue
        authorized, tier = merge_authorized_fn(number)
        candidates.append(classify_candidate(detail, ctx, merge_authorized=authorized, tier=tier))
    return candidates


def run_boss_drain(
    ctx: DrainContext,
    pass_policy: DrainPassPolicy,
    *,
    list_open_prs_fn: ListOpenPRsFn,
    view_pr_fn: ViewPRFn,
    merge_authorized_fn: MergeAuthorizedFn,
    execute_fn: ExecuteFn,
    max_classify: int = 60,
) -> DrainPassResult:
    """End-to-end bounded drain pass over the live open-PR queue."""
    candidates = build_candidates(
        ctx,
        list_open_prs_fn=list_open_prs_fn,
        view_pr_fn=view_pr_fn,
        merge_authorized_fn=merge_authorized_fn,
        max_classify=max_classify,
    )
    return run_drain_pass(pass_policy, candidates, execute_fn)
