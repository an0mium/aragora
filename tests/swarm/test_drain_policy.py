"""Tests for the drain-mode decision core (pure, deterministic)."""

from __future__ import annotations

import pytest

from aragora.swarm.drain_policy import (
    DrainAction,
    DrainCandidate,
    DrainDecision,
    DrainPolicy,
    decide_drain_action,
)

POL = DrainPolicy(auto_settle_max_tier=2)


def _green(**kw: object) -> DrainCandidate:
    base: dict[str, object] = {
        "pr": 1,
        "has_changes": True,
        "required_checks_green": True,
        "quorum_satisfied": True,
        "mergeable": True,
        "tier": 0,
    }
    base.update(kw)
    return DrainCandidate(**base)  # type: ignore[arg-type]


def test_fully_gated_in_bound_tier_merges() -> None:
    assert decide_drain_action(POL, _green()).action is DrainAction.MERGE


def test_off_limits_always_left_untouched() -> None:
    # The anti-Factory-collision guarantee: off-limits wins over everything.
    d = decide_drain_action(POL, _green(off_limits=True))
    assert d.action is DrainAction.LEAVE and "off-limits" in d.reason


def test_owned_by_other_agent_left() -> None:
    assert decide_drain_action(POL, _green(owned_by_other_agent=True)).action is DrainAction.LEAVE


def test_empty_pr_closed_superseded() -> None:
    d = decide_drain_action(POL, DrainCandidate(pr=2, has_changes=False))
    assert d.action is DrainAction.CLOSE_SUPERSEDED


def test_explicitly_superseded_closed() -> None:
    d = decide_drain_action(POL, DrainCandidate(pr=2, has_changes=True, superseded=True))
    assert d.action is DrainAction.CLOSE_SUPERSEDED


def test_red_useful_pr_is_repaired_never_closed() -> None:
    # The operator's rule: a merely-red useful PR is REPAIR, NOT close.
    d = decide_drain_action(
        POL, DrainCandidate(pr=3, has_changes=True, required_checks_green=False)
    )
    assert d.action is DrainAction.REPAIR


def test_green_but_no_quorum_is_repaired() -> None:
    d = decide_drain_action(POL, _green(quorum_satisfied=False))
    assert d.action is DrainAction.REPAIR


def test_green_but_not_mergeable_is_repaired() -> None:
    d = decide_drain_action(POL, _green(mergeable=False))
    assert d.action is DrainAction.REPAIR


def test_over_tier_left_for_operator_never_auto_merged() -> None:
    # Tier 3 is fully gated but above the autonomous bound -> LEAVE, never MERGE.
    d = decide_drain_action(POL, _green(tier=3))
    assert d.action is DrainAction.LEAVE and "operator" in d.reason


def test_off_limits_precedes_superseded() -> None:
    # Even an empty PR that's off-limits is left alone (don't act on another fleet's PR).
    d = decide_drain_action(POL, DrainCandidate(pr=4, has_changes=False, off_limits=True))
    assert d.action is DrainAction.LEAVE


def test_close_requires_truly_superseded_not_just_red() -> None:
    # Guard against scope creep: nothing closes a PR that merely has failing checks.
    red = DrainCandidate(pr=5, has_changes=True, required_checks_green=False, tier=0)
    assert decide_drain_action(POL, red).action is not DrainAction.CLOSE_SUPERSEDED


def test_action_vocabulary_is_fixed() -> None:
    assert {a.value for a in DrainAction} == {"merge", "repair", "close_superseded", "leave"}


def test_policy_validation() -> None:
    with pytest.raises(ValueError, match="auto_settle_max_tier"):
        DrainPolicy(auto_settle_max_tier=5)


def test_to_dict() -> None:
    d: DrainDecision = decide_drain_action(POL, _green())
    assert d.to_dict() == {"action": "merge", "reason": d.reason}


def test_deterministic() -> None:
    c = _green(tier=3)
    assert decide_drain_action(POL, c) == decide_drain_action(POL, c)
