"""Tests for the unattended Tier 0-2 auto-merge decision core.

The decision core (:func:`aragora.swarm.auto_merge_green.decide_auto_merge`) is
pure: it takes an already-fetched PR context and returns whether the PR may be
merged unattended, plus the blockers that prevented it. It encodes the *same*
authorization the merge-quorum gate already grants for Tier 0-2 PRs whose
merge-packet reaches ``status=satisfied`` -- it never makes a new risk judgment,
it only decides whether to *execute* an already-authorized merge without a human.

Safety is the whole point, so every guard that keeps a not-fully-authorized PR
from auto-merging gets its own test asserting the specific blocker.
"""

from __future__ import annotations

import dataclasses

import pytest

from aragora.swarm.auto_merge_green import (
    MAX_AUTO_MERGE_TIER,
    REQUIRED_CHECKS,
    PRMergeContext,
    context_from_gh,
    decide_auto_merge,
    first_error_line,
)


def _green_checks() -> dict[str, str]:
    states = dict.fromkeys(REQUIRED_CHECKS, "SUCCESS")
    states["aragora-merge-quorum"] = "SUCCESS"
    return states


def _authorized_context(**overrides) -> PRMergeContext:
    """A fully-authorized Tier-2 PR: every guard passes unless overridden."""
    base = dict(
        number=8447,
        head_sha="a" * 40,
        packet_head_sha="a" * 40,
        packet_pr_number=8447,
        tier=2,
        packet_status="satisfied",
        packet_verdict="admin_squash_allowed",
        requires_human_risk_settlement=False,
        unresolved_dissent=False,
        admin_squash_allowed=True,
        is_draft=False,
        mergeable="MERGEABLE",
        merge_state_status="BLOCKED",
        check_states=_green_checks(),
    )
    base.update(overrides)
    return PRMergeContext(**base)


def test_fully_authorized_tier2_pr_is_merged():
    decision = decide_auto_merge(_authorized_context())
    assert decision.should_merge is True
    assert decision.blockers == ()
    assert decision.number == 8447
    assert decision.head_sha == "a" * 40


def test_packet_head_mismatch_is_blocked():
    # The merge-packet is fetched in a separate subprocess from the gh view; if
    # the head moved between them we'd be deciding on mismatched data.
    decision = decide_auto_merge(_authorized_context(packet_head_sha="f" * 40))
    assert decision.should_merge is False
    assert any("head" in b.lower() for b in decision.blockers)


def test_absent_packet_head_does_not_add_mismatch_blocker():
    # packet=None -> packet_head_sha="" -> tier=None already blocks; no spurious
    # head-mismatch blocker should pile on.
    decision = decide_auto_merge(_authorized_context(packet_head_sha="", tier=None))
    assert decision.should_merge is False
    assert not any("head" in b.lower() and "mismatch" in b.lower() for b in decision.blockers)


def test_packet_pr_number_mismatch_is_blocked():
    # Defense-in-depth: a packet entry bound to a different PR must not merge.
    decision = decide_auto_merge(_authorized_context(packet_pr_number=9999))
    assert decision.should_merge is False
    assert any("pr mismatch" in b.lower() for b in decision.blockers)


def test_absent_packet_pr_number_blocks():
    # A packet with no concrete PR identity cannot safely be bound to the gh
    # view. Fail closed instead of treating "undisclosed" as acceptable.
    decision = decide_auto_merge(_authorized_context(packet_pr_number=0))
    assert decision.should_merge is False
    assert any("packet pr number" in b.lower() for b in decision.blockers)


def test_negative_tier_is_blocked():
    decision = decide_auto_merge(_authorized_context(tier=-1))
    assert decision.should_merge is False
    assert any("tier" in b.lower() for b in decision.blockers)


def test_clean_merge_state_is_also_mergeable():
    # A Tier-0 docs PR can reach CLEAN (no branch-protection block); still merge.
    decision = decide_auto_merge(_authorized_context(tier=0, merge_state_status="CLEAN"))
    assert decision.should_merge is True
    assert decision.blockers == ()


def test_tier_three_is_blocked_for_human_settlement():
    decision = decide_auto_merge(_authorized_context(tier=3))
    assert decision.should_merge is False
    assert any("tier" in b.lower() for b in decision.blockers)


def test_tier_four_is_blocked():
    decision = decide_auto_merge(_authorized_context(tier=4))
    assert decision.should_merge is False
    assert any("tier" in b.lower() for b in decision.blockers)


def test_unknown_tier_is_blocked():
    decision = decide_auto_merge(_authorized_context(tier=None))
    assert decision.should_merge is False
    assert any("tier" in b.lower() for b in decision.blockers)


def test_requires_human_risk_settlement_is_blocked():
    decision = decide_auto_merge(_authorized_context(requires_human_risk_settlement=True))
    assert decision.should_merge is False
    assert any("human" in b.lower() for b in decision.blockers)


def test_packet_status_not_satisfied_is_blocked():
    decision = decide_auto_merge(_authorized_context(packet_status="needs_model_review_quorum"))
    assert decision.should_merge is False
    assert any("satisfied" in b.lower() for b in decision.blockers)


def test_packet_verdict_not_admin_squash_is_blocked():
    decision = decide_auto_merge(
        _authorized_context(packet_verdict="collect_model_quorum_before_merge")
    )
    assert decision.should_merge is False
    assert any("verdict" in b.lower() for b in decision.blockers)


def test_admin_squash_not_allowed_is_blocked():
    decision = decide_auto_merge(_authorized_context(admin_squash_allowed=False))
    assert decision.should_merge is False
    assert any("admin squash" in b.lower() for b in decision.blockers)


def test_unresolved_dissent_is_blocked():
    decision = decide_auto_merge(_authorized_context(unresolved_dissent=True))
    assert decision.should_merge is False
    assert any("dissent" in b.lower() for b in decision.blockers)


def test_draft_is_blocked():
    decision = decide_auto_merge(_authorized_context(is_draft=True))
    assert decision.should_merge is False
    assert any("draft" in b.lower() for b in decision.blockers)


def test_conflicting_is_blocked():
    decision = decide_auto_merge(_authorized_context(mergeable="CONFLICTING"))
    assert decision.should_merge is False
    assert any("mergeable" in b.lower() for b in decision.blockers)


def test_unknown_mergeability_is_blocked():
    decision = decide_auto_merge(_authorized_context(mergeable="UNKNOWN"))
    assert decision.should_merge is False
    assert any("mergeable" in b.lower() for b in decision.blockers)


def test_dirty_merge_state_is_blocked():
    decision = decide_auto_merge(_authorized_context(merge_state_status="DIRTY"))
    assert decision.should_merge is False
    assert any("merge state" in b.lower() for b in decision.blockers)


def test_unstable_merge_state_is_blocked():
    # UNSTABLE = a non-required check is failing; skip rather than risk it.
    decision = decide_auto_merge(_authorized_context(merge_state_status="UNSTABLE"))
    assert decision.should_merge is False
    assert any("merge state" in b.lower() for b in decision.blockers)


def test_quorum_not_green_is_blocked():
    states = _green_checks()
    states["aragora-merge-quorum"] = "FAILURE"
    decision = decide_auto_merge(_authorized_context(check_states=states))
    assert decision.should_merge is False
    assert any("quorum" in b.lower() for b in decision.blockers)


def test_quorum_missing_is_blocked():
    states = _green_checks()
    del states["aragora-merge-quorum"]
    decision = decide_auto_merge(_authorized_context(check_states=states))
    assert decision.should_merge is False
    assert any("quorum" in b.lower() for b in decision.blockers)


def test_any_failing_required_check_is_blocked():
    states = _green_checks()
    states["lint"] = "FAILURE"
    decision = decide_auto_merge(_authorized_context(check_states=states))
    assert decision.should_merge is False
    assert any("lint" in b for b in decision.blockers)


def test_pending_required_check_is_blocked():
    states = _green_checks()
    states["typecheck"] = "PENDING"
    decision = decide_auto_merge(_authorized_context(check_states=states))
    assert decision.should_merge is False
    assert any("typecheck" in b for b in decision.blockers)


def test_missing_required_check_is_blocked():
    states = _green_checks()
    del states["sdk-parity"]
    decision = decide_auto_merge(_authorized_context(check_states=states))
    assert decision.should_merge is False
    assert any("sdk-parity" in b for b in decision.blockers)


def test_multiple_blockers_are_all_reported():
    decision = decide_auto_merge(
        _authorized_context(tier=4, is_draft=True, mergeable="CONFLICTING")
    )
    assert decision.should_merge is False
    # all three independent problems surface, not just the first
    assert len(decision.blockers) >= 3


def test_max_tier_is_two_by_default():
    assert MAX_AUTO_MERGE_TIER == 2


def test_context_is_immutable():
    ctx = _authorized_context()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.tier = 4  # type: ignore[misc]


def test_failing_non_required_check_blocks_even_when_blocked():
    # The target population is ~always mergeStateStatus=BLOCKED, so a failing
    # *non-required* check would otherwise pass every guard and get --admin
    # merged. Any failing check in the rollup must block.
    states = _green_checks()
    states["Baseline Determinism"] = "FAILURE"  # non-required, failing
    decision = decide_auto_merge(
        _authorized_context(check_states=states, merge_state_status="BLOCKED")
    )
    assert decision.should_merge is False
    assert any(
        "baseline determinism" in b.lower() or "failing" in b.lower() for b in decision.blockers
    )


def test_cancelled_check_blocks():
    states = _green_checks()
    states["some-check"] = "CANCELLED"
    decision = decide_auto_merge(_authorized_context(check_states=states))
    assert decision.should_merge is False


def test_first_error_line_whitespace_only_is_safe():
    # Regression: "\n".strip().splitlines()[0] used to raise IndexError mid-pass.
    assert first_error_line("\n", "") == "merge failed"
    assert first_error_line("", "") == "merge failed"
    assert first_error_line("   ", "  ") == "merge failed"


def test_first_error_line_returns_first_line():
    assert first_error_line("boom\nmore detail", "") == "boom"
    assert first_error_line("", "stdout only") == "stdout only"
    assert first_error_line("stderr wins", "stdout loses") == "stderr wins"


def _quorum_row(conclusion: str, completed_at: str) -> dict[str, object]:
    return {
        "__typename": "CheckRun",
        "name": "aragora-merge-quorum",
        "conclusion": conclusion,
        "startedAt": completed_at,
        "completedAt": completed_at,
    }


def test_stale_success_after_current_failure_does_not_read_green():
    """A superseded SUCCESS ordered after the current FAILURE must not win.

    Observed live on PR #9571: the rollup carried
    [FAILURE @21:39 (current), SUCCESS @18:43 (stale draft-phase run)] in that
    order, so last-write-wins reported the quorum check green while the
    current run had in fact failed.
    """
    view = {
        "number": 9571,
        "headRefOid": "a" * 40,
        "statusCheckRollup": [
            _quorum_row("FAILURE", "2026-07-24T21:39:21Z"),
            _quorum_row("SUCCESS", "2026-07-24T18:43:02Z"),
        ],
    }
    ctx = context_from_gh(view, {"tier": 2})
    assert ctx.check_states["aragora-merge-quorum"] == "FAILURE"


def test_rerun_success_after_earlier_failure_reads_green():
    """The legitimate rerun case must still resolve to SUCCESS.

    Guards against over-correcting into "any failure wins", which would block
    every PR whose check was rerun to green (common after cancelled runs).
    """
    view = {
        "number": 1,
        "headRefOid": "a" * 40,
        "statusCheckRollup": [
            _quorum_row("FAILURE", "2026-07-24T18:00:00Z"),
            _quorum_row("SUCCESS", "2026-07-24T21:00:00Z"),
        ],
    }
    ctx = context_from_gh(view, {"tier": 2})
    assert ctx.check_states["aragora-merge-quorum"] == "SUCCESS"


def test_rollup_reduction_is_order_independent():
    """Same rows, either order, same verdict — order must not decide merges."""
    rows = [
        _quorum_row("FAILURE", "2026-07-24T21:39:21Z"),
        _quorum_row("SUCCESS", "2026-07-24T18:43:02Z"),
    ]
    forward = context_from_gh({"statusCheckRollup": rows}, {"tier": 2})
    reverse = context_from_gh({"statusCheckRollup": list(reversed(rows))}, {"tier": 2})
    assert forward.check_states == reverse.check_states


def test_untimestamped_rows_fail_closed():
    """Without timestamps a success cannot be proven current, so failure wins."""
    view = {
        "statusCheckRollup": [
            {"name": "aragora-merge-quorum", "conclusion": "SUCCESS"},
            {"name": "aragora-merge-quorum", "conclusion": "FAILURE"},
        ]
    }
    assert context_from_gh(view, {"tier": 2}).check_states["aragora-merge-quorum"] == "FAILURE"
    # ...and the reverse order agrees.
    view["statusCheckRollup"].reverse()
    assert context_from_gh(view, {"tier": 2}).check_states["aragora-merge-quorum"] == "FAILURE"


def test_stale_success_blocks_the_actual_merge_decision():
    """End-to-end: the stale-success rollup must not authorise a merge."""
    states = _green_checks()
    ctx = _authorized_context(check_states=states)
    stale = context_from_gh(
        {
            "statusCheckRollup": [
                _quorum_row("FAILURE", "2026-07-24T21:39:21Z"),
                _quorum_row("SUCCESS", "2026-07-24T18:43:02Z"),
            ]
        },
        {"tier": 2},
    )
    merged_states = dict(states)
    merged_states["aragora-merge-quorum"] = stale.check_states["aragora-merge-quorum"]
    decision = decide_auto_merge(dataclasses.replace(ctx, check_states=merged_states))
    assert decision.should_merge is False


def test_in_flight_current_run_outranks_stale_completed_success():
    """A re-run still in flight must not lose to a stale completed SUCCESS.

    Regression on the first attempt at this fix: ranking on ``completedAt``
    first put an in-progress row (which has none) *below* an older finished
    row, so a stale SUCCESS won and the merge read as green while the current
    quorum run was still deciding.
    """
    view = {
        "statusCheckRollup": [
            {
                "name": "aragora-merge-quorum",
                "status": "IN_PROGRESS",
                "conclusion": None,
                "startedAt": "2026-07-24T22:00:00Z",
                "completedAt": None,
            },
            _quorum_row("SUCCESS", "2026-07-24T18:43:02Z"),
        ]
    }
    assert context_from_gh(view, {"tier": 2}).check_states["aragora-merge-quorum"] != "SUCCESS"


def test_completed_rerun_still_beats_older_in_flight_row():
    """Ordering by start time must not flip the legitimate rerun case."""
    view = {
        "statusCheckRollup": [
            {
                "name": "aragora-merge-quorum",
                "status": "IN_PROGRESS",
                "conclusion": None,
                "startedAt": "2026-07-24T18:00:00Z",
                "completedAt": None,
            },
            _quorum_row("SUCCESS", "2026-07-24T21:00:00Z"),
        ]
    }
    assert context_from_gh(view, {"tier": 2}).check_states["aragora-merge-quorum"] == "SUCCESS"


@pytest.mark.parametrize("blocking", ["PENDING", "QUEUED", "IN_PROGRESS", ""])
def test_untimestamped_success_loses_to_any_non_success(blocking):
    """An unrankable SUCCESS must not beat *any* non-success row.

    Review finding: the first tie-break only preferred explicitly failing
    states, so PENDING/QUEUED/IN_PROGRESS/unknown could still lose to an
    untimestamped SUCCESS purely on input order.
    """
    rows = [
        {"name": "aragora-merge-quorum", "conclusion": "SUCCESS"},
        {"name": "aragora-merge-quorum", "status": blocking, "conclusion": None},
    ]
    assert (
        context_from_gh({"statusCheckRollup": rows}, {"tier": 2}).check_states[
            "aragora-merge-quorum"
        ]
        != "SUCCESS"
    )
    assert (
        context_from_gh({"statusCheckRollup": list(reversed(rows))}, {"tier": 2}).check_states[
            "aragora-merge-quorum"
        ]
        != "SUCCESS"
    )


def test_rollup_states_are_case_normalised():
    """Lower-cased states must not slip past the reduction or the decision."""
    rows = [
        {"name": "aragora-merge-quorum", "conclusion": "success"},
        {"name": "aragora-merge-quorum", "conclusion": "failure"},
    ]
    states = context_from_gh({"statusCheckRollup": rows}, {"tier": 2}).check_states
    assert states["aragora-merge-quorum"] == "FAILURE"
