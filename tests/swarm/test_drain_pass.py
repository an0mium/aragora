"""Tests for the bounded drain-pass orchestrator (pure; injected execution)."""

from __future__ import annotations

from aragora.swarm.drain_pass import (
    DrainPassPolicy,
    plan_drain_pass,
    run_drain_pass,
)
from aragora.swarm.drain_policy import DrainAction, DrainCandidate, DrainPolicy


def _mergeable(pr: int, tier: int = 0) -> DrainCandidate:
    return DrainCandidate(
        pr=pr,
        has_changes=True,
        required_checks_green=True,
        quorum_satisfied=True,
        mergeable=True,
        tier=tier,
    )


def _repairable(pr: int) -> DrainCandidate:
    return DrainCandidate(pr=pr, has_changes=True, required_checks_green=False, mergeable=False)


def test_leave_is_never_executed() -> None:
    cands = [
        DrainCandidate(pr=1, off_limits=True, has_changes=True),  # Factory branch
        DrainCandidate(pr=2, owned_by_other_agent=True, has_changes=True),
        _mergeable(3, tier=4),  # over auto_settle tier -> LEAVE
    ]
    calls: list[int] = []
    res = run_drain_pass(DrainPassPolicy(), cands, lambda pr, a: calls.append(pr) or True)
    assert calls == []  # nothing executed
    assert {p.pr for p in res.left} == {1, 2, 3}


def test_off_limits_factory_pr_left_even_if_green() -> None:
    # A green, mergeable PR that is pinned off-limits must still be LEFT.
    factory_pr = DrainCandidate(
        pr=1,
        off_limits=True,
        has_changes=True,
        required_checks_green=True,
        quorum_satisfied=True,
        mergeable=True,
    )
    res = run_drain_pass(DrainPassPolicy(), [factory_pr], lambda pr, a: True)
    assert res.executed == ()
    assert res.left[0].action is DrainAction.LEAVE


def test_merge_close_repair_routed_and_executed() -> None:
    cands = [
        _mergeable(10),  # MERGE
        DrainCandidate(pr=11, has_changes=False),  # empty -> CLOSE_SUPERSEDED
        DrainCandidate(pr=12, has_changes=True, superseded=True),  # CLOSE_SUPERSEDED
        _repairable(13),  # REPAIR
    ]
    seen: dict[int, DrainAction] = {}
    res = run_drain_pass(DrainPassPolicy(), cands, lambda pr, a: seen.__setitem__(pr, a) or True)
    assert seen[10] is DrainAction.MERGE
    assert seen[11] is DrainAction.CLOSE_SUPERSEDED
    assert seen[12] is DrainAction.CLOSE_SUPERSEDED
    assert seen[13] is DrainAction.REPAIR
    assert set(res.executed) == {10, 11, 12, 13}


def test_repair_is_tightly_capped_no_storm() -> None:
    # 50 repairable PRs, cap repairs at 2 -> only 2 planned, 48 deferred.
    cands = [_repairable(100 + i) for i in range(50)]
    policy = DrainPassPolicy(max_repairs_per_pass=2)
    planned, deferred, left = plan_drain_pass(policy, cands)
    assert len(planned) == 2
    assert len(deferred) == 48
    assert all(p.action is DrainAction.REPAIR for p in planned)


def test_merge_cap_and_deferral() -> None:
    cands = [_mergeable(200 + i) for i in range(9)]
    policy = DrainPassPolicy(max_merges_per_pass=5)
    planned, deferred, _ = plan_drain_pass(policy, cands)
    assert len(planned) == 5
    assert len(deferred) == 4


def test_priority_order_merge_before_close_before_repair() -> None:
    # Mixed batch; verify planned ordering follows MERGE -> CLOSE -> REPAIR.
    cands = [
        _repairable(1),
        DrainCandidate(pr=2, has_changes=False),  # CLOSE
        _mergeable(3),
    ]
    planned, _, _ = plan_drain_pass(DrainPassPolicy(), cands)
    actions = [p.action for p in planned]
    assert actions == [DrainAction.MERGE, DrainAction.CLOSE_SUPERSEDED, DrainAction.REPAIR]


def test_execute_failure_does_not_abort_pass() -> None:
    cands = [_mergeable(1), _mergeable(2), _mergeable(3)]

    def flaky(pr: int, a: DrainAction) -> bool:
        return pr != 2  # #2 "fails"

    res = run_drain_pass(DrainPassPolicy(), cands, flaky)
    assert set(res.executed) == {1, 3}
    assert res.failed == (2,)


def test_execute_exception_is_caught_as_failure() -> None:
    def boom(pr: int, a: DrainAction) -> bool:
        raise RuntimeError("gh down")

    res = run_drain_pass(DrainPassPolicy(), [_mergeable(1)], boom)
    assert res.failed == (1,)
    assert res.executed == ()


def test_tier_bound_respects_drain_policy() -> None:
    # tier 3 with default auto_settle_max_tier=2 -> LEAVE (parks for operator).
    res = run_drain_pass(
        DrainPassPolicy(drain=DrainPolicy(auto_settle_max_tier=2)),
        [_mergeable(1, tier=3)],
        lambda pr, a: True,
    )
    assert res.executed == ()
    assert res.left[0].pr == 1


def test_negative_cap_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="max_repairs_per_pass"):
        DrainPassPolicy(max_repairs_per_pass=-1)


def test_to_dict_shape() -> None:
    res = run_drain_pass(DrainPassPolicy(), [_mergeable(1)], lambda pr, a: True)
    d = res.to_dict()
    assert set(d) == {"planned", "deferred", "left", "executed", "failed", "counts"}
    assert d["counts"]["executed"] == 1
