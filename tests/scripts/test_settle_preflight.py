from __future__ import annotations

import scripts.settle_preflight as settle_preflight


def _entry(**overrides):
    base = {
        "pr_number": 9001,
        "title": "test pr",
        "head_sha": "a" * 40,
        "tier": 0,
        "status": "satisfied",
        "verdict": "admin_squash_allowed",
        "admin_squash_allowed": True,
        "requires_human_risk_settlement": False,
        "checks_summary": "6/6 green",
        "reasons": ["docs/tests/status-only"],
    }
    base.update(overrides)
    return base


def _metadata(**overrides):
    base = {
        "number": 9001,
        "title": "test pr",
        "headRefOid": "a" * 40,
        "isDraft": False,
        "mergeable": "MERGEABLE",
        "mergeStateStatus": "CLEAN",
        "files": [{"path": "docs/example.md"}],
    }
    base.update(overrides)
    return base


def test_main_red_halt_verdict() -> None:
    result = settle_preflight.classify_pr(entry=_entry(), metadata=_metadata(), main_red=True)

    assert result.verdict == settle_preflight.MAIN_RED_HALT
    assert "main-red" in result.action
    assert result.recheck_rule == settle_preflight.RECHECK_RULE


def test_draft_skip_verdict() -> None:
    result = settle_preflight.classify_pr(entry=_entry(), metadata=_metadata(isDraft=True))

    assert result.verdict == settle_preflight.DRAFT_SKIP
    assert "marked ready" in result.action


def test_human_gated_for_tier_above_two() -> None:
    result = settle_preflight.classify_pr(entry=_entry(tier=4), metadata=_metadata())

    assert result.verdict == settle_preflight.HUMAN_GATED
    assert "Tier 4" in result.reasons
    assert "human settlement" in result.action


def test_human_gated_for_unsettled_human_risk() -> None:
    result = settle_preflight.classify_pr(
        entry=_entry(tier=2, requires_human_risk_settlement=True),
        metadata=_metadata(),
    )

    assert result.verdict == settle_preflight.HUMAN_GATED
    assert any("requires_human_risk_settlement" in reason for reason in result.reasons)


def test_head_blocked_for_conflicting_or_behind_state() -> None:
    dirty = settle_preflight.classify_pr(
        entry=_entry(admin_squash_allowed=False),
        metadata=_metadata(mergeable="CONFLICTING", mergeStateStatus="DIRTY"),
    )
    behind = settle_preflight.classify_pr(
        entry=_entry(admin_squash_allowed=False),
        metadata=_metadata(mergeStateStatus="BEHIND"),
    )

    assert dirty.verdict == settle_preflight.HEAD_BLOCKED
    assert behind.verdict == settle_preflight.HEAD_BLOCKED


def test_github_unstable_for_model_authorized_unstable_state() -> None:
    result = settle_preflight.classify_pr(
        entry=_entry(),
        metadata=_metadata(mergeStateStatus="UNSTABLE"),
    )

    assert result.verdict == settle_preflight.GITHUB_UNSTABLE
    assert "do not merge" in result.action


def test_ready_for_model_authorized_clean_state() -> None:
    result = settle_preflight.classify_pr(entry=_entry(), metadata=_metadata())

    assert result.verdict == settle_preflight.READY
    assert "normal protected squash merge" in result.action


def test_head_blocked_when_packet_not_authorized() -> None:
    result = settle_preflight.classify_pr(
        entry=_entry(
            status="needs_model_review_quorum",
            verdict="collect_model_quorum_before_merge",
            admin_squash_allowed=False,
        ),
        metadata=_metadata(),
    )

    assert result.verdict == settle_preflight.HEAD_BLOCKED
    assert "satisfied model packet" in result.action
