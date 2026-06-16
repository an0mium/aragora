"""Claim-first lane dispatch for the merge-advance swarm.

The cross-agent lane-lease registry already exists end to end:
``scripts/claim_active_agent_lane.py`` writes atomic claims into
``.aragora/agent-bridge/lanes.json`` and ``scripts/identify_lane_owner.py``
reads them back with owner-lease liveness. The chronic failure mode is *not* a
missing primitive -- it is, in that script's own words, that "many sessions
(Claude Code, Codex CLI, Factory Droid, standalone scripts) never write a lane
claim, so the registry tends to stay empty even when 5-10 concurrent agents are
running." Each session then free-picks "the highest-value open PR" and several
converge on the same one, doing duplicate evidence/settlement work that is
thrown away when one of them lands first.

This module is the missing *front half* of the loop: given the merge-blocked PR
candidates and the set of PRs that already have a LIVE owner (liveness resolved
by the canonical ``identify_lane_owner`` so this stays a pure decision), it
hands each free worker exactly one unclaimed PR and a short, constant,
claim-first prompt. It is the complement of :mod:`aragora.swarm.conductor`,
which decides the *follow-up* once a worker finishes.

Pure-core by design: no GitHub calls, no ``lanes.json`` writes, no process
spawning happen here. The CLI shell resolves candidates (via
``merge_quorum_io``) and live claims (via ``identify_lane_owner``) and feeds
them in as JSON; the live spawning is delegated to
``aragora.swarm.worker_launcher``. That keeps every dispatch decision unit
testable without a network or a worktree.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_MAX_WORKERS = 3
# Conservative ref allowlist. The branch is interpolated into a prompt that
# tells the worker to shell out (``--branch <branch>``); GitHub permits ``$``,
# `` ` ``, ``\``, and ``{`` in refs, so any branch outside this set is replaced
# with a safe placeholder rather than embedded verbatim (defends both the
# shell-injection surface and ``str.format`` KeyError on a literal ``{``).
_SAFE_BRANCH = re.compile(r"^[A-Za-z0-9._/-]+$")
# A worker prompt deliberately small and constant: the guardrails live in the
# claim/merge-gate tooling and the lane registry, not in pasted text that grows
# every turn. ``{...}`` placeholders are filled by :func:`build_worker_prompt`.
WORKER_PROMPT_TEMPLATE = """\
You are an Aragora lane worker. Session id: {session_id}. \
Assigned lane: PR #{pr} (branch {branch}) in {repo} ONLY.

1. CLAIM-OR-YIELD. Run:
   python3 scripts/identify_lane_owner.py --pr {pr} --json
   If it reports a LIVE owner whose owner_session != {session_id}, print \
"yielding: owned by <owner>" and STOP -- do not work this PR.
   Otherwise claim it:
   python3 scripts/claim_active_agent_lane.py --lane-id {lane_id} \
--owner-session {session_id} \
--pr-number {pr} --branch {branch} --source {target_agent} --status active \
--next-action "advance #{pr}"

2. GROUND from live state for #{pr} ONLY (gh pr view/checks; \
review-queue merge-packet --pr {pr} --json). Trust live state, never memory.

3. ADVANCE ONE BOUNDED STEP toward merge: rerun one stale failed required \
check, OR collect one exact-head two-family evidence set (only if \
review-queue evidence-lint reports would_count=true), OR make one narrow \
repair in an isolated worktree. Never merge, admin-merge, settle Tier-4, \
touch branch protection, or touch another PR.

4. REPORT + RELEASE. Print the new head SHA, the action taken, its result, \
and the single exact remaining blocker. Refresh the claim heartbeat; if #{pr} \
merged or you are blocked with no safe next action, release the lane.

Stop after one bounded unit. Do NOT scout the queue or pick a different PR -- \
assignment comes from the conductor."""


@dataclass
class LaneAssignment:
    """One free worker paired with exactly one unclaimed merge-blocked PR."""

    pr: int
    branch: str
    owner_session: str
    head: str = ""


@dataclass
class DispatchPlan:
    assignments: list[LaneAssignment] = field(default_factory=list)
    # PRs skipped because a live owner already holds the lane: {pr: owner}.
    owned: dict[int, str] = field(default_factory=dict)
    # Unclaimed candidates left over once max_workers was reached.
    deferred: list[int] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignments": [asdict(a) for a in self.assignments],
            "owned": {str(pr): owner for pr, owner in self.owned.items()},
            "deferred": list(self.deferred),
            "reason": self.reason,
        }


def default_session_id(pr: int) -> str:
    return f"codex-lane-pr{pr}-{int(time.time())}-{uuid.uuid4().hex[:8]}"


def select_assignments(
    *,
    candidates: Sequence[dict[str, Any]],
    live_claims_by_pr: dict[int, str],
    max_workers: int = DEFAULT_MAX_WORKERS,
    session_id_for: Callable[[int], str] = default_session_id,
) -> DispatchPlan:
    """Assign unclaimed candidates to free workers, never reassigning a live lane.

    ``candidates`` is the merge-blocked PR list in priority order; each entry is
    ``{"number": int, "branch": str, "head": str?}``. ``live_claims_by_pr`` maps
    a PR number to the owner_session of its current LIVE owner (liveness already
    resolved by ``identify_lane_owner`` -- a stale/terminal owner must NOT appear
    here, so its lane is reassignable). Assignment is order-preserving and capped
    at ``max_workers`` so the dispatcher applies backpressure instead of
    spawning unbounded workers.
    """
    plan = DispatchPlan()
    cap = max(0, int(max_workers))
    for entry in candidates:
        pr = entry.get("number")
        if not isinstance(pr, int):
            continue
        owner = live_claims_by_pr.get(pr)
        if owner:
            # A live owner already holds this lane: never double-assign it.
            plan.owned[pr] = owner
            continue
        if len(plan.assignments) >= cap:
            plan.deferred.append(pr)
            continue
        plan.assignments.append(
            LaneAssignment(
                pr=pr,
                branch=str(entry.get("branch") or "").strip(),
                owner_session=session_id_for(pr),
                head=str(entry.get("head") or "").strip(),
            )
        )
    plan.reason = (
        f"assigned {len(plan.assignments)} of {len(candidates)} candidate(s); "
        f"{len(plan.owned)} already live-owned, {len(plan.deferred)} deferred "
        f"(max_workers={cap})"
    )
    return plan


def build_worker_prompt(
    *, pr: int, branch: str, session_id: str, repo: str, target_agent: str = "codex"
) -> str:
    """The short, constant claim-first prompt for one assigned lane.

    ``target_agent`` is recorded as the claim ``--source`` so lane attribution
    matches the agent actually dispatched (not a hardcoded ``codex``). A branch
    that fails ``_SAFE_BRANCH`` is replaced with a placeholder so a hostile or
    brace-bearing ref can neither inject into the shell-out nor break
    ``str.format``.
    """
    # session_id is interpolated into lane_id and the --lane-id/--owner-session
    # shell-out. The default generator is safe, but session_id_for is a public
    # injection seam -- fail fast on a non-conforming value rather than emit a
    # prompt that could break or inject.
    if not _SAFE_BRANCH.match(session_id or ""):
        raise ValueError(f"unsafe session_id {session_id!r}: must match {_SAFE_BRANCH.pattern}")
    # target_agent is interpolated unquoted into the claim shell-out (--source);
    # it is operator-supplied (--target-agent), so validate it too.
    if not _SAFE_BRANCH.match(target_agent or ""):
        raise ValueError(f"unsafe target_agent {target_agent!r}: must match {_SAFE_BRANCH.pattern}")
    # repo is interpolated into the gh shell-out the worker is told to run; gate it
    # with the same allowlist ("owner/name" matches: alnum, '.', '/', '-', '_').
    if not _SAFE_BRANCH.match(repo or ""):
        raise ValueError(f"unsafe repo {repo!r}: must match {_SAFE_BRANCH.pattern}")
    safe_branch = branch if (branch and _SAFE_BRANCH.match(branch)) else f"(branch for #{pr})"
    return WORKER_PROMPT_TEMPLATE.format(
        session_id=session_id,
        lane_id=f"lane-{pr}-{session_id}",
        pr=pr,
        branch=safe_branch,
        repo=repo,
        target_agent=target_agent,
    )


def live_claims_from_arg(value: Any) -> dict[int, str]:
    """Accept either ``{"<pr>": "owner"}`` or ``[{"pr": n, "owner_session": s}]``."""
    claims: dict[int, str] = {}
    if isinstance(value, dict):
        for pr, owner in value.items():
            try:
                claims[int(pr)] = str(owner or "")
            except (TypeError, ValueError):
                continue
    elif isinstance(value, list):
        for row in value:
            if not isinstance(row, dict):
                continue
            pr = row.get("pr", row.get("pr_number", row.get("number")))
            if pr is None:
                continue
            owner = row.get("owner_session", row.get("owner"))
            try:
                claims[int(pr)] = str(owner or "")
            except (TypeError, ValueError):
                continue
    return {pr: owner for pr, owner in claims.items() if owner}
