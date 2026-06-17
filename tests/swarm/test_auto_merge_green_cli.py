"""Tests for the glue layer of unattended Tier 0-2 auto-merge.

These cover the impure-but-injectable pieces around the pure decision core:
- ``context_from_gh``: map a ``gh pr view`` payload + merge-packet entry into a
  :class:`PRMergeContext` (parses the heterogeneous statusCheckRollup shape).
- ``merge_eligible``: decide a batch of contexts.
- ``apply_merges``: the bounded apply loop, with the merge side-effect injected
  so we test orchestration (dry-run, cap, failure handling) without touching gh.
"""

from __future__ import annotations

from aragora.swarm.auto_merge_green import (
    REQUIRED_CHECKS,
    MergeDecision,
    apply_merges,
    context_from_gh,
    decide_auto_merge,
    merge_eligible,
)


def _rollup_all_green() -> list[dict]:
    rollup = [{"name": name, "conclusion": "SUCCESS"} for name in REQUIRED_CHECKS]
    rollup.append({"name": "aragora-merge-quorum", "conclusion": "SUCCESS"})
    return rollup


def _view(**overrides) -> dict:
    base = dict(
        number=8447,
        headRefOid="b" * 40,
        isDraft=False,
        mergeable="MERGEABLE",
        mergeStateStatus="BLOCKED",
        statusCheckRollup=_rollup_all_green(),
    )
    base.update(overrides)
    return base


def _packet(**overrides) -> dict:
    base = dict(
        tier=2,
        status="satisfied",
        verdict="admin_squash_allowed",
        requires_human_risk_settlement=False,
        unresolved_dissent=False,
        admin_squash_allowed=True,
        head_sha="b" * 40,  # matches _view() headRefOid
    )
    base.update(overrides)
    return base


# --- context_from_gh -------------------------------------------------------


def test_context_from_gh_authorized_pr_decides_merge():
    ctx = context_from_gh(_view(), _packet())
    assert decide_auto_merge(ctx).should_merge is True


def test_context_from_gh_extracts_head_and_number():
    ctx = context_from_gh(_view(number=99, headRefOid="c" * 40), _packet())
    assert ctx.number == 99
    assert ctx.head_sha == "c" * 40


def test_context_from_gh_parses_status_style_rollup_entry():
    # Commit *statuses* (vs check *runs*) use "context"+"state", not "name"+"conclusion".
    rollup = _rollup_all_green()
    rollup.append({"context": "legacy-status", "state": "SUCCESS"})
    ctx = context_from_gh(_view(statusCheckRollup=rollup), _packet())
    assert ctx.check_states["legacy-status"] == "SUCCESS"
    assert ctx.check_states["aragora-merge-quorum"] == "SUCCESS"


def test_context_from_gh_missing_packet_yields_unknown_tier():
    ctx = context_from_gh(_view(), None)
    assert ctx.tier is None
    assert decide_auto_merge(ctx).should_merge is False


def test_context_from_gh_carries_human_settlement_flag():
    ctx = context_from_gh(_view(), _packet(tier=4, requires_human_risk_settlement=True))
    assert ctx.requires_human_risk_settlement is True
    assert decide_auto_merge(ctx).should_merge is False


def test_context_from_gh_extracts_packet_head():
    ctx = context_from_gh(_view(headRefOid="e" * 40), _packet(head_sha="e" * 40))
    assert ctx.packet_head_sha == "e" * 40
    assert ctx.head_sha == "e" * 40


def test_context_from_gh_packet_head_mismatch_blocks_merge():
    # packet computed against a different head than the view -> must not merge.
    ctx = context_from_gh(_view(headRefOid="1" * 40), _packet(head_sha="2" * 40))
    assert decide_auto_merge(ctx).should_merge is False
    assert any("head" in b.lower() for b in decide_auto_merge(ctx).blockers)


def test_context_fail_closed_on_missing_settlement_flag():
    # A safety-critical auto-merge must fail CLOSED if the packet omits the
    # human-settlement flag (e.g. a schema rename), not silently permit.
    pkt = _packet()
    del pkt["requires_human_risk_settlement"]
    ctx = context_from_gh(_view(), pkt)
    assert ctx.requires_human_risk_settlement is True
    assert decide_auto_merge(ctx).should_merge is False


def test_context_fail_closed_on_missing_dissent_flag():
    pkt = _packet()
    del pkt["unresolved_dissent"]
    ctx = context_from_gh(_view(), pkt)
    assert ctx.unresolved_dissent is True
    assert decide_auto_merge(ctx).should_merge is False


# --- merge_eligible --------------------------------------------------------


def test_merge_eligible_decides_each_context():
    good = context_from_gh(_view(number=1), _packet())
    bad = context_from_gh(_view(number=2), _packet(tier=4))
    decisions = merge_eligible([good, bad])
    by_pr = {d.number: d for d in decisions}
    assert by_pr[1].should_merge is True
    assert by_pr[2].should_merge is False


# --- apply_merges ----------------------------------------------------------


def _decision(number: int, should_merge: bool) -> MergeDecision:
    return MergeDecision(
        number=number,
        head_sha="d" * 40,
        should_merge=should_merge,
        blockers=() if should_merge else ("blocked for test",),
    )


def test_apply_merges_dry_run_never_calls_merger():
    calls: list[tuple[int, str]] = []

    def merger(pr, head):  # pragma: no cover - must NOT be invoked
        calls.append((pr, head))
        return (True, "ok")

    results = apply_merges([_decision(1, True)], merge_fn=merger, dry_run=True)
    assert calls == []
    assert results[0]["pr"] == 1
    assert results[0]["action"] == "would-merge"


def test_apply_merges_applies_only_eligible():
    calls: list[int] = []

    def merger(pr, head):
        calls.append(pr)
        return (True, "merged")

    results = apply_merges(
        [_decision(1, True), _decision(2, False)], merge_fn=merger, dry_run=False
    )
    assert calls == [1]
    actions = {r["pr"]: r["action"] for r in results}
    assert actions[1] == "merged"
    assert actions[2] == "skip"


def test_apply_merges_respects_max_merges():
    calls: list[int] = []

    def merger(pr, head):
        calls.append(pr)
        return (True, "merged")

    results = apply_merges(
        [_decision(1, True), _decision(2, True), _decision(3, True)],
        merge_fn=merger,
        dry_run=False,
        max_merges=1,
    )
    assert calls == [1]
    actions = {r["pr"]: r["action"] for r in results}
    assert actions[1] == "merged"
    assert "deferred" in actions[2]
    assert "deferred" in actions[3]


def test_apply_merges_failed_merge_does_not_consume_cap():
    def merger(pr, head):
        if pr == 1:
            return (False, "gh rejected")
        return (True, "merged")

    results = apply_merges(
        [_decision(1, True), _decision(2, True)],
        merge_fn=merger,
        dry_run=False,
        max_merges=1,
    )
    actions = {r["pr"]: r["action"] for r in results}
    assert actions[1] == "merge-failed"
    # the failed one must not have eaten the single merge slot
    assert actions[2] == "merged"
