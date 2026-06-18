"""Pure routing/gating decision for the operator-driven PR settlement CLI.

``scripts/settle_pr.py`` orchestrates the end-to-end settlement runbook by
*composing* the existing tools (``collect_quorum_evidence.py`` for evidence,
``auto_merge_quorum_green.py`` for Tier 0-2 merges, ``settle_tier4_pr.py`` for
Tier 3-4 human settlement). This module is its single decision brain: given an
already-fetched quorum/tier state it decides whether settlement can proceed and
by which route, accumulating every blocker. It performs **no I/O**, so the
routing/gating logic stays deterministic and unit-testable.

The actual merge gate is never reimplemented here -- the composed tools remain
authoritative and enforce it. This module only decides *which* of them to invoke
and refuses to proceed when the model quorum is not genuinely satisfied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# Tier 0-2 can settle unattended on a green model quorum (auto-merge path);
# Tier 3-4 require a head-bound human risk settlement by a trusted operator.
AUTO_MERGE_MAX_TIER = 2

ROUTE_AUTO_MERGE = "auto_merge_green"
ROUTE_OPERATOR_TIER4 = "operator_human_settlement"
ROUTE_BLOCKED = "blocked"


@dataclass(frozen=True)
class SettlementPlan:
    """The decided route + readiness for one PR. Pure output of :func:`plan_settlement`."""

    tier: int | None
    route: str
    quorum_satisfied: bool
    requires_operator_login: bool
    ready_to_mutate: bool
    blockers: tuple[str, ...]


def plan_settlement(
    *,
    tier: int | None,
    quorum_satisfied: bool,
    supportive_families: Sequence[str],
    unresolved_dissent: bool = False,
    operator_login_provided: bool = False,
) -> SettlementPlan:
    """Decide the settlement route and whether it is safe to proceed.

    ``ready_to_mutate`` is True only when the model quorum is satisfied, the tier
    is classified, and any route-specific precondition is met (Tier 3-4 needs an
    operator login). Every reason a mutation is refused is collected in
    ``blockers`` so the CLI can surface all of them at once.
    """
    blockers: list[str] = []
    supportive = sorted({str(f) for f in supportive_families})

    if not quorum_satisfied:
        blockers.append(f"model quorum not satisfied (supportive families: {supportive or 'none'})")

    if tier is None:
        route = ROUTE_BLOCKED
        blockers.append("tier unknown (merge-packet did not classify the PR)")
    elif tier <= AUTO_MERGE_MAX_TIER:
        route = ROUTE_AUTO_MERGE
        if unresolved_dissent:
            blockers.append("unresolved model dissent (Tier 0-2 cannot auto-merge over a dissent)")
    else:
        route = ROUTE_OPERATOR_TIER4

    requires_operator_login = route == ROUTE_OPERATOR_TIER4
    if requires_operator_login and not operator_login_provided:
        blockers.append(
            f"Tier {tier} requires a human risk settlement: pass --operator-login <gh-login>"
        )

    ready_to_mutate = not blockers and route != ROUTE_BLOCKED
    return SettlementPlan(
        tier=tier,
        route=route,
        quorum_satisfied=quorum_satisfied,
        requires_operator_login=requires_operator_login,
        ready_to_mutate=ready_to_mutate,
        blockers=tuple(blockers),
    )


def tier4_settle_commands(
    *, repo: str, pr: int, head: str, operator_login: str, no_app_token: bool = False
) -> list[str]:
    """The exact, head-bound ``settle_tier4_pr.py`` commands a trusted operator runs
    to record the human risk settlement and merge. The CLI surfaces these rather
    than executing them: the Tier-4 human settlement must be a deliberate operator
    act, never an automated one."""
    base = (
        f"python3 scripts/settle_tier4_pr.py --pr {pr} --head {head} "
        f"--trusted-operator-login {operator_login} --repo {repo}"
    )
    prefix = "ARAGORA_DISABLE_GITHUB_APP_TOKEN=1 " if no_app_token else ""
    return [f"{base} --check", f"{base} --settle-only", f"{prefix}{base} --merge-apply"]


def summarize_collect(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a ``collect_quorum_evidence --json`` payload into the fields the
    planner and diagnostics need. Tolerates the error envelope (``{"error": ...}``)
    by reporting an unsatisfied quorum with no tier."""
    if payload.get("error"):
        return {
            "error": str(payload["error"]),
            "tier": None,
            "head_sha": payload.get("head_sha"),
            "quorum_satisfied": False,
            "supportive_families": [],
            "dissenting_families": [],
            "items": [],
            "failures": list(payload.get("failures") or []),
        }
    items = []
    for it in payload.get("items") or []:
        items.append(
            {
                "family": it.get("family"),
                "verdict": it.get("verdict"),
                "would_count": bool(it.get("would_count")),
                "counted_reviewer_ids": list(it.get("counted_reviewer_ids") or []),
                "problems": list(it.get("problems") or []),
            }
        )
    return {
        "error": None,
        "tier": payload.get("tier"),
        "head_sha": payload.get("head_sha"),
        "quorum_satisfied": bool(payload.get("has_supportive_quorum")),
        "supportive_families": list(payload.get("supportive_families") or []),
        "dissenting_families": list(payload.get("dissenting_families") or []),
        "items": items,
        "failures": list(payload.get("failures") or []),
    }
