"""Tests for the boss-loop drain driver (injected I/O; safe-by-construction)."""

from __future__ import annotations

from aragora.swarm.boss_drain import (
    DrainContext,
    build_candidates,
    classify_candidate,
    make_repair_order,
    make_repair_prompt,
    run_boss_drain,
    touches_merge_authority,
)
from aragora.swarm.drain_pass import DrainPassPolicy
from aragora.swarm.drain_policy import (
    DrainAction,
    DrainCandidate,
    DrainPolicy,
    decide_drain_action,
)


def test_off_limits_pr_never_repairs_even_with_changes() -> None:
    # A drain-repair worker spawns on DrainAction.REPAIR. An off-limits PR (Factory
    # structex/* or claude/fusion-*) with changes must map to LEAVE, never REPAIR —
    # otherwise dispatch_repair would push to another fleet's branch (cross-fleet
    # collision). off_limits LEAVE must win regardless of has_changes / authority.
    for authorized in (True, False):
        d = decide_drain_action(
            DrainPolicy(),
            DrainCandidate(
                pr=1,
                has_changes=True,
                off_limits=True,
                required_checks_green=authorized,
                quorum_satisfied=authorized,
                mergeable=authorized,
                tier=0,
            ),
        )
        assert d.action is DrainAction.LEAVE


def test_owned_pr_never_repairs() -> None:
    d = decide_drain_action(
        DrainPolicy(),
        DrainCandidate(pr=2, has_changes=True, owned_by_other_agent=True, tier=0),
    )
    assert d.action is DrainAction.LEAVE


def test_repair_prompt_is_scope_locked() -> None:
    pr = make_repair_prompt(8460, "codex/red")
    # must pin the branch, forbid scope-broadening, forbid touching gate logic, bound effort
    assert "codex/red" in pr and "#8460" in pr
    assert "ONLY" in pr and "branch" in pr
    assert "review_queue" in pr and "quorum" in pr  # gate logic explicitly off-limits
    assert "STOP" in pr  # bounded — must not thrash


def test_make_repair_order_defaults_to_codex() -> None:
    o = make_repair_order(8460, "codex/red")
    assert o.pr == 8460 and o.branch == "codex/red" and o.agent == "codex"
    assert o.to_dict()["prompt"] == o.prompt


def test_merge_authority_files_force_off_limits() -> None:
    # A green, mergeable, low-tier PR that touches the EVIDENCE PARSER (gate logic)
    # must be LEFT — the drain may never auto-merge merge-authority surfaces (#8467 class).
    view = {
        "number": 8467,
        "headRefName": "codex/evidence-fix",
        "changedFiles": 2,
        "files": [
            {"path": "aragora/cli/commands/review_queue_comment_verdicts.py"},
            {"path": "tests/cli/commands/test_review_queue.py"},
        ],
    }
    c = classify_candidate(view, DrainContext(), merge_authorized=True, tier=0)
    assert c.off_limits is True  # gate logic — never auto-merged


def test_touches_merge_authority_patterns() -> None:
    assert touches_merge_authority(["aragora/swarm/quorum_evidence.py"]) is True
    assert touches_merge_authority(["scripts/settle_one_pr.py"]) is True
    assert touches_merge_authority([".github/workflows/aragora-merge-quorum.yml"]) is True
    assert touches_merge_authority(["aragora/debate/orchestrator.py"]) is False


def test_classify_off_limits_branch_prefix() -> None:
    c = classify_candidate(
        {"number": 1, "headRefName": "structex/p2-docs", "changedFiles": 3},
        DrainContext(),
        merge_authorized=True,
        tier=0,
    )
    assert c.off_limits is True  # Factory branch — must never be touched


def test_classify_off_limits_pinned_number() -> None:
    c = classify_candidate(
        {"number": 99, "headRefName": "codex/foo", "changedFiles": 3},
        DrainContext(off_limits_prs=frozenset({99})),
        merge_authorized=True,
        tier=0,
    )
    assert c.off_limits is True


def test_classify_empty_has_no_changes() -> None:
    c = classify_candidate(
        {"number": 2, "headRefName": "codex/x", "changedFiles": 0},
        DrainContext(),
        merge_authorized=False,
        tier=4,
    )
    assert c.has_changes is False  # -> CLOSE_SUPERSEDED downstream


def test_classify_authorized_sets_mergeable_bundle() -> None:
    c = classify_candidate(
        {"number": 3, "headRefName": "codex/x", "changedFiles": 5},
        DrainContext(),
        merge_authorized=True,
        tier=1,
    )
    assert c.required_checks_green and c.quorum_satisfied and c.mergeable and c.tier == 1


def _views() -> list[dict]:
    return [
        {"number": 10, "headRefName": "codex/green", "changedFiles": 4},  # -> authority probe
        {"number": 11, "headRefName": "codex/empty", "changedFiles": 0},  # -> CLOSE (no probe)
        {
            "number": 12,
            "headRefName": "structex/x",
            "changedFiles": 9,
        },  # -> off-limits LEAVE (no probe)
        {"number": 13, "headRefName": "codex/red", "changedFiles": 7},  # -> REPAIR
    ]


def test_build_candidates_only_probes_promising_prs() -> None:
    probed: list[int] = []
    viewed: list[int] = []

    def auth(n: int) -> tuple[bool, int]:
        probed.append(n)
        return (n == 10, 0)  # only #10 authorized

    def view(n: int) -> dict:
        viewed.append(n)
        return next(v for v in _views() if v["number"] == n)

    cands = build_candidates(
        DrainContext(),
        list_open_prs_fn=_views,
        view_pr_fn=view,
        merge_authorized_fn=auth,
        max_classify=60,
    )
    # off-limits (#12) and empty (#11) must NOT be authority-probed
    assert probed == [10, 13]
    # cheap pre-filter: an off-limits-by-prefix PR (#12) skips the detail fetch
    # entirely (the +300-gh-call regression fix). #11 still needs the fetch to
    # discover it is empty.
    assert 12 not in viewed
    assert set(viewed) == {10, 11, 13}
    by_pr = {c.pr: c for c in cands}
    assert by_pr[10].mergeable is True
    assert by_pr[11].has_changes is False
    assert by_pr[12].off_limits is True
    assert by_pr[13].mergeable is False


def test_build_candidates_respects_max_classify() -> None:
    many = [{"number": 100 + i, "headRefName": "codex/x", "changedFiles": 2} for i in range(50)]
    seen: list[int] = []
    build_candidates(
        DrainContext(),
        list_open_prs_fn=lambda: many,
        view_pr_fn=lambda n: {"number": n, "headRefName": "codex/x", "changedFiles": 2},
        merge_authorized_fn=lambda n: (seen.append(n) or (False, 4)),
        max_classify=5,
    )
    assert len(seen) == 5  # only first 5 classified/probed


def test_run_boss_drain_routes_and_executes_safely() -> None:
    executed: dict[int, DrainAction] = {}

    def auth(n: int) -> tuple[bool, int]:
        return (n == 10, 0)

    res = run_boss_drain(
        DrainContext(),
        DrainPassPolicy(max_repairs_per_pass=5),
        list_open_prs_fn=_views,
        view_pr_fn=lambda n: next(v for v in _views() if v["number"] == n),
        merge_authorized_fn=auth,
        execute_fn=lambda pr, a: executed.__setitem__(pr, a) or True,
        max_classify=60,
    )
    assert executed.get(10) is DrainAction.MERGE
    assert executed.get(11) is DrainAction.CLOSE_SUPERSEDED
    assert 12 not in executed  # off-limits LEFT, never executed
    assert executed.get(13) is DrainAction.REPAIR
    assert {c.pr for c in res.left} == {12}


def test_unauthorized_pr_never_merges() -> None:
    # Authority says NO for everything -> no MERGE, only REPAIR/CLOSE/LEAVE.
    res = run_boss_drain(
        DrainContext(),
        DrainPassPolicy(),
        list_open_prs_fn=lambda: [{"number": 5, "headRefName": "codex/x", "changedFiles": 3}],
        view_pr_fn=lambda n: {"number": 5, "headRefName": "codex/x", "changedFiles": 3},
        merge_authorized_fn=lambda n: (False, 0),
        execute_fn=lambda pr, a: True,
        max_classify=60,
    )
    actions = {p.action for p in res.planned}
    assert DrainAction.MERGE not in actions
